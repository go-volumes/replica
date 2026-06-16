package replica

import (
	"errors"
	"fmt"
	"sync"
)

// targets snapshots, under the lock, the replicas a write/flush must reach: the
// in-sync set (the authoritative copies, counted toward the minimum) plus any
// rebuilding ones (kept current but not counted). It returns the in-sync slice
// and the rebuilding slice separately so the caller can apply the degrade rule
// to in-sync failures only.
func (e *Engine) targets() (inSync, rebuilding []*replicaState, closed bool) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed {
		return nil, nil, true
	}
	for _, rs := range e.reps {
		switch rs.state {
		case InSync:
			inSync = append(inSync, rs)
		case Rebuilding:
			rebuilding = append(rebuilding, rs)
		}
	}
	return inSync, rebuilding, false
}

// opResult pairs a replica with the error its mirrored operation returned.
type opResult struct {
	rs  *replicaState
	err error
}

// fanout runs op against every replica in reps concurrently and collects the
// per-replica errors.
func fanout(reps []*replicaState, op func(*replicaState) error) []opResult {
	results := make([]opResult, len(reps))
	var wg sync.WaitGroup
	wg.Add(len(reps))
	for i, rs := range reps {
		go func(i int, rs *replicaState) {
			defer wg.Done()
			results[i] = opResult{rs: rs, err: op(rs)}
		}(i, rs)
	}
	wg.Wait()
	return results
}

// demote marks every replica that errored out-of-sync (in-sync → OutOfSync) or,
// for a rebuilding replica, OutOfSync too (its rebuild is now invalid). It
// returns the count of in-sync replicas that survived (acked without error).
func (e *Engine) demote(inSync, rebuilding []opResult) int {
	e.mu.Lock()
	defer e.mu.Unlock()
	survivors := 0
	for _, r := range inSync {
		if r.err != nil {
			if r.rs.state == InSync {
				r.rs.state = OutOfSync
			}
		} else {
			survivors++
		}
	}
	for _, r := range rebuilding {
		if r.err != nil && r.rs.state == Rebuilding {
			// A rebuilding replica that fails a live write is no longer a valid
			// rebuild target; drop it back to out-of-sync.
			r.rs.state = OutOfSync
		}
	}
	return survivors
}

// mirror applies op to all in-sync (counted) and rebuilding (uncounted)
// replicas concurrently, demotes failures, and enforces the min-in-sync rule.
// It returns nil when at least minInSync in-sync replicas acked, or an error
// describing the failure otherwise.
func (e *Engine) mirror(what string, op func(*replicaState) error) error {
	inSync, rebuilding, closed := e.targets()
	if closed {
		return ErrClosed
	}
	if len(inSync) == 0 {
		return fmt.Errorf("%s: %w", what, ErrNoInSync)
	}

	inRes := fanout(inSync, op)
	rebRes := fanout(rebuilding, op)
	survivors := e.demote(inRes, rebRes)

	if survivors < e.minInSync {
		return fmt.Errorf("%s: %w: %d in-sync replica(s) acked, need %d (%s)",
			what, ErrNoInSync, survivors, e.minInSync, joinErrs(inRes))
	}
	return nil
}

// joinErrs summarizes the per-replica errors for a degraded-but-OK operation's
// error message.
func joinErrs(results []opResult) error {
	var errs []error
	for _, r := range results {
		if r.err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", r.rs.name, r.err))
		}
	}
	return errors.Join(errs...)
}

// WriteAt mirrors p at off to every in-sync replica (and any rebuilding one)
// concurrently. It returns len(p) once at least MinInSync in-sync replicas ack;
// a replica that errors is marked out-of-sync. The write fails only if fewer
// than MinInSync in-sync replicas remain.
func (e *Engine) WriteAt(p []byte, off int64) (int, error) {
	err := e.mirror("WriteAt", func(rs *replicaState) error {
		n, werr := rs.dev.WriteAt(p, off)
		if werr != nil {
			return werr
		}
		if n != len(p) {
			return fmt.Errorf("short write: %d of %d bytes", n, len(p))
		}
		return nil
	})
	if err != nil {
		return 0, err
	}
	return len(p), nil
}

// Sync flushes every in-sync (and rebuilding) replica concurrently, applying
// the same degrade rule as WriteAt. Durability of the bytes is each replica's
// own responsibility.
func (e *Engine) Sync() error {
	return e.mirror("Sync", func(rs *replicaState) error {
		return rs.dev.Sync()
	})
}

// ReadAt serves from one in-sync replica, preferring the configured local one,
// then trying the others in order on error. It returns the first successful
// read; if every in-sync replica errors, it returns the last error, and if
// there are no in-sync replicas it returns [ErrNoInSync].
func (e *Engine) ReadAt(p []byte, off int64) (int, error) {
	order, closed := e.readOrder()
	if closed {
		return 0, ErrClosed
	}
	if len(order) == 0 {
		return 0, fmt.Errorf("ReadAt: %w", ErrNoInSync)
	}
	var lastErr error
	for _, rs := range order {
		n, err := rs.dev.ReadAt(p, off)
		if err == nil {
			return n, nil
		}
		lastErr = fmt.Errorf("replica %q: %w", rs.name, err)
		// A read error does not by itself prove the replica is out-of-sync (it
		// could be a transient transport blip), so reads do not demote; the next
		// in-sync replica is tried instead.
	}
	return 0, fmt.Errorf("ReadAt: all in-sync replicas failed: %w", lastErr)
}

// readOrder returns the in-sync replicas in read-preference order: the local
// replica first (when it is in-sync), then the rest in configuration order.
func (e *Engine) readOrder() (order []*replicaState, closed bool) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed {
		return nil, true
	}
	local := e.reps[e.localIx]
	if local.state == InSync {
		order = append(order, local)
	}
	for i, rs := range e.reps {
		if i == e.localIx {
			continue
		}
		if rs.state == InSync {
			order = append(order, rs)
		}
	}
	return order, false
}

// Close closes every replica device once and marks the Engine closed. It joins
// the per-replica Close errors. Subsequent operations return [ErrClosed].
func (e *Engine) Close() error {
	e.mu.Lock()
	if e.closed {
		e.mu.Unlock()
		return nil
	}
	e.closed = true
	reps := e.reps
	e.mu.Unlock()

	var errs []error
	for _, rs := range reps {
		if err := rs.dev.Close(); err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", rs.name, err))
		}
	}
	return errors.Join(errs...)
}

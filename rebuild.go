package replica

import (
	"context"
	"fmt"
)

// DefaultChunkSize is the rebuild streaming granularity when Rebuild does not
// specify one.
const DefaultChunkSize = 1 << 20 // 1 MiB

// Rebuild resynchronizes the named replica from a healthy in-sync source by
// streaming the whole volume (source ReadAt → target WriteAt) in chunks, then
// marks the target in-sync.
//
// Concurrency: the target is first flipped to Rebuilding, which makes [Engine]'s
// write path mirror live writes to it alongside the in-sync set (see mirror).
// Each chunk is then read from a live in-sync source — which is itself receiving
// those same live writes — and written to the target. Because both source and
// target see every concurrent write, and the source is the authority for each
// chunk's current contents, the target converges to the source once the full
// span has been copied. A live write that demotes the target back to
// OutOfSync aborts the rebuild.
//
// chunkSize <= 0 uses [DefaultChunkSize]. Rebuild honors ctx cancellation
// between chunks.
func (e *Engine) Rebuild(ctx context.Context, name string, chunkSize int) error {
	if chunkSize <= 0 {
		chunkSize = DefaultChunkSize
	}

	// One rebuild at a time.
	e.rebuildMu.Lock()
	defer e.rebuildMu.Unlock()

	target, err := e.beginRebuild(name)
	if err != nil {
		return err
	}

	size := e.size
	buf := make([]byte, chunkSize)
	for off := int64(0); off < size; off += int64(chunkSize) {
		select {
		case <-ctx.Done():
			e.abortRebuild(target)
			return fmt.Errorf("rebuild %q: %w", name, ctx.Err())
		default:
		}

		n := chunkSize
		if rem := size - off; rem < int64(n) {
			n = int(rem)
		}

		src, ok := e.pickSource(target)
		if !ok {
			e.abortRebuild(target)
			return fmt.Errorf("rebuild %q: %w", name, ErrNoInSync)
		}
		if _, rerr := src.dev.ReadAt(buf[:n], off); rerr != nil {
			e.abortRebuild(target)
			return fmt.Errorf("rebuild %q: read source %q at %d: %w", name, src.name, off, rerr)
		}

		if !e.stillRebuilding(target) {
			// A concurrent live write failed on the target and demoted it.
			return fmt.Errorf("rebuild %q: target demoted mid-rebuild: %w", name, ErrNoInSync)
		}
		if _, werr := target.dev.WriteAt(buf[:n], off); werr != nil {
			e.abortRebuild(target)
			return fmt.Errorf("rebuild %q: write target at %d: %w", name, off, werr)
		}
	}

	return e.finishRebuild(target)
}

// beginRebuild validates the target and flips it to Rebuilding under the lock.
func (e *Engine) beginRebuild(name string) (*replicaState, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, ErrClosed
	}
	target, ok := e.byName[name]
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrUnknownReplica, name)
	}
	// There must be at least one OTHER in-sync replica to rebuild from.
	hasSource := false
	for _, rs := range e.reps {
		if rs != target && rs.state == InSync {
			hasSource = true
			break
		}
	}
	if !hasSource {
		return nil, fmt.Errorf("rebuild %q: %w", name, ErrNoInSync)
	}
	target.state = Rebuilding
	return target, nil
}

// pickSource returns an in-sync replica other than target to read from.
func (e *Engine) pickSource(target *replicaState) (*replicaState, bool) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	// Prefer the local replica when it is a valid source.
	if local := e.reps[e.localIx]; local != target && local.state == InSync {
		return local, true
	}
	for _, rs := range e.reps {
		if rs != target && rs.state == InSync {
			return rs, true
		}
	}
	return nil, false
}

// stillRebuilding reports whether target is still in the Rebuilding state (a
// concurrent failed write may have demoted it to OutOfSync).
func (e *Engine) stillRebuilding(target *replicaState) bool {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return target.state == Rebuilding
}

// finishRebuild flips a still-rebuilding target to InSync. If it was demoted
// mid-flight (a live write failed), it stays OutOfSync and an error is returned.
func (e *Engine) finishRebuild(target *replicaState) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return ErrClosed
	}
	if target.state != Rebuilding {
		return fmt.Errorf("rebuild %q: target demoted mid-rebuild: %w", target.name, ErrNoInSync)
	}
	target.state = InSync
	return nil
}

// abortRebuild returns a still-rebuilding target to OutOfSync (a demoted target
// is left as-is).
func (e *Engine) abortRebuild(target *replicaState) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if target.state == Rebuilding {
		target.state = OutOfSync
	}
}

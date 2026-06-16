// Package replica is the data plane for replicated, highly-available block
// volumes in the go-volumes family. It fronts N synchronous replicas — each any
// [volume.Device] (in production an NBD client to a remote pool volume, in
// tests an in-memory device) — behind a single [Engine] that itself satisfies
// [volume.Device], so a filesystem-format driver or an NBD server can write
// through it unchanged.
//
// The model is single-active-writer with synchronous replication (RPO 0): one
// Engine owns a volume and mirrors every write to all in-sync replicas,
// acking only once they all ack. A replica that errors is marked out-of-sync
// and dropped from the write set; the write still succeeds as long as at least
// MinInSync replicas remain in sync, otherwise it fails. Reads are served from
// one healthy replica (preferring the configured local one) with failover.
//
// Failover and fencing of a stale writer (STONITH before promotion) are a
// future control-plane concern; this package leaves a [Fencer] seam for it but
// implements no consensus here.
//
// Durability between the Engine and its replicas relies on each replica's own
// Sync: the Engine's Sync flushes every in-sync replica, and each replica is
// responsible for making those bytes durable (a pool fsync, an S3 PUT, an NBD
// FLUSH to a remote backing). The Engine adds no buffering of its own.
package replica

import (
	"context"
	"errors"
	"fmt"
	"sync"

	volume "github.com/go-volumes/interface"
)

// State is a replica's replication state within an [Engine].
type State int

const (
	// InSync means the replica holds the current data and participates in every
	// write and read.
	InSync State = iota
	// OutOfSync means the replica errored (or is new) and is excluded from
	// writes and reads until a [Engine.Rebuild] brings it current.
	OutOfSync
	// Rebuilding means the replica is being resynced: live writes are applied to
	// it alongside the in-sync set, but it is not yet trusted for reads.
	Rebuilding
)

// String renders a State for logs and status output.
func (s State) String() string {
	switch s {
	case InSync:
		return "in-sync"
	case OutOfSync:
		return "out-of-sync"
	case Rebuilding:
		return "rebuilding"
	default:
		return fmt.Sprintf("State(%d)", int(s))
	}
}

// Replica names a backing [volume.Device] that an [Engine] mirrors a volume to.
type Replica struct {
	// Name uniquely identifies the replica within an Engine (e.g. the pool or
	// host it lives on). Names must be unique.
	Name string
	// Dev is the backing device. In production this is typically an NBD client
	// to a remote pool volume; in tests it is an in-memory device.
	Dev volume.Device
}

// ReplicaStatus is a snapshot of one replica's state, returned by
// [Engine.Status].
type ReplicaStatus struct {
	Name  string
	State State
}

// Fencer is the seam a control plane uses to STONITH a stale writer before
// promoting a new one. The Engine does not call it (single-active-writer fencing
// is a control-plane decision — see github.com/go-volumes/replica-ha); it is
// held so a higher layer can fence through the same object that owns the
// replicas.
type Fencer interface {
	// Fence must isolate the named writer so it can no longer issue a single
	// write to the shared replicas, and return nil ONLY once that is definitively
	// true. A control plane opens the new leader's write gate the moment Fence
	// returns nil, so a Fencer that returns nil WITHOUT actually stopping the old
	// writer silently re-introduces split-brain and corrupts the volume. When in
	// doubt, return an error: a failed fence keeps the new leader passive (safe);
	// a falsely-successful one is unsafe.
	//
	// Implementations back this with whatever the substrate offers, strongest
	// first: power/STONITH (hard-stop the node or micro-VM), a cloud force-detach
	// or power-off, an IPMI/PDU cut or network ACL, or revoking the credential the
	// writer presents to the replicas. Fence should be idempotent (fencing an
	// already-dead writer returns nil) and must honour ctx (a timeout → error).
	Fence(ctx context.Context, writer string) error
}

// Config tunes an [Engine]. The zero value is valid: MinInSync defaults to 1
// and Local defaults to the first replica.
type Config struct {
	// MinInSync is the minimum number of in-sync replicas a write/flush must
	// reach to succeed. It defaults to 1 (any single surviving replica keeps the
	// volume writable). Set it higher for quorum-style durability. It is clamped
	// to [1, len(replicas)].
	MinInSync int
	// Local names the replica reads prefer (e.g. a same-host replica). Empty (or
	// unmatched) prefers the first replica.
	Local string
	// Fencer, if set, is exposed via [Engine.Fencer] for the control plane.
	Fencer Fencer
}

// Errors returned by the Engine.
var (
	// ErrNoReplicas is returned by New when no replicas are supplied.
	ErrNoReplicas = errors.New("replica: no replicas")
	// ErrSizeMismatch is returned by New when replicas disagree on size.
	ErrSizeMismatch = errors.New("replica: replica size mismatch")
	// ErrNoInSync is returned when an operation cannot meet the minimum number
	// of in-sync replicas (the volume has lost too many replicas).
	ErrNoInSync = errors.New("replica: not enough in-sync replicas")
	// ErrUnknownReplica is returned by Rebuild for an unknown replica name.
	ErrUnknownReplica = errors.New("replica: unknown replica")
	// ErrDuplicateName is returned by New when two replicas share a name.
	ErrDuplicateName = errors.New("replica: duplicate replica name")
	// ErrClosed is returned by operations after the Engine is closed.
	ErrClosed = errors.New("replica: engine closed")
)

// replicaState is the Engine's mutable per-replica bookkeeping.
type replicaState struct {
	name  string
	dev   volume.Device
	state State
}

// Engine fronts a set of replicas as a single replicated [volume.Device]. It is
// safe for concurrent use. Construct it with [New].
type Engine struct {
	size      int64
	minInSync int
	fencer    Fencer

	mu      sync.RWMutex // guards reps, localIdx, closed
	reps    []*replicaState
	byName  map[string]*replicaState
	localIx int // index in reps of the preferred read replica
	closed  bool

	rebuildMu sync.Mutex // serializes Rebuild so at most one runs at a time

	// writeMu serializes live writes against each other AND against a rebuild's
	// per-chunk copy. Holding it makes a WriteAt's fan-out and a rebuild's
	// (read-source → write-target) mutually exclusive, so a rebuild can never
	// overwrite a concurrent live write on the target with a stale value
	// (which would diverge the target from the source), and all replicas see
	// writes in one order.
	writeMu sync.Mutex
}

// Compile-time assertion that *Engine satisfies the volume contract.
var _ volume.Device = (*Engine)(nil)

// New builds an Engine over replicas. It validates that the replica set is
// non-empty, that names are unique, and that every replica reports the same
// size (returning [ErrSizeMismatch] otherwise). All replicas start in-sync.
func New(replicas []Replica, cfg Config) (*Engine, error) {
	if len(replicas) == 0 {
		return nil, ErrNoReplicas
	}

	reps := make([]*replicaState, 0, len(replicas))
	byName := make(map[string]*replicaState, len(replicas))
	var size int64
	for i, r := range replicas {
		if _, dup := byName[r.Name]; dup {
			return nil, fmt.Errorf("%w: %q", ErrDuplicateName, r.Name)
		}
		sz, err := r.Dev.Size()
		if err != nil {
			return nil, fmt.Errorf("replica %q size: %w", r.Name, err)
		}
		if i == 0 {
			size = sz
		} else if sz != size {
			return nil, fmt.Errorf("%w: %q is %d, want %d", ErrSizeMismatch, r.Name, sz, size)
		}
		rs := &replicaState{name: r.Name, dev: r.Dev, state: InSync}
		reps = append(reps, rs)
		byName[r.Name] = rs
	}

	minInSync := cfg.MinInSync
	if minInSync < 1 {
		minInSync = 1
	}
	if minInSync > len(reps) {
		minInSync = len(reps)
	}

	localIx := 0
	if cfg.Local != "" {
		for i, rs := range reps {
			if rs.name == cfg.Local {
				localIx = i
				break
			}
		}
	}

	return &Engine{
		size:      size,
		minInSync: minInSync,
		fencer:    cfg.Fencer,
		reps:      reps,
		byName:    byName,
		localIx:   localIx,
	}, nil
}

// Size reports the replicated volume size (validated equal across replicas at
// construction).
func (e *Engine) Size() (int64, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed {
		return 0, ErrClosed
	}
	return e.size, nil
}

// Fencer returns the configured Fencer seam (nil if none). The control plane
// uses it to fence a stale writer before promotion; the Engine never calls it.
func (e *Engine) Fencer() Fencer { return e.fencer }

// Status returns a snapshot of every replica's state, in configuration order.
func (e *Engine) Status() []ReplicaStatus {
	e.mu.RLock()
	defer e.mu.RUnlock()
	out := make([]ReplicaStatus, len(e.reps))
	for i, rs := range e.reps {
		out[i] = ReplicaStatus{Name: rs.name, State: rs.state}
	}
	return out
}

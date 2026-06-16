package replica

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"

	volume "github.com/go-volumes/interface"
)

// memDevice is an in-memory volume.Device with injectable failures, used as a
// replica backing in tests.
type memDevice struct {
	mu     sync.Mutex
	data   []byte
	synced int
	closed bool

	// Injected failures (under mu). A *Count value fails for the next N calls.
	readErr   error
	writeErr  error
	syncErr   error
	closeErr  error
	sizeErr   error
	sizeValue *int64 // override reported size when non-nil
}

func newMem(n int) *memDevice { return &memDevice{data: make([]byte, n)} }

func (m *memDevice) ReadAt(p []byte, off int64) (int, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.readErr != nil {
		return 0, m.readErr
	}
	if off < 0 || off > int64(len(m.data)) {
		return 0, fmt.Errorf("read out of range off=%d", off)
	}
	return copy(p, m.data[off:]), nil
}

func (m *memDevice) WriteAt(p []byte, off int64) (int, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.writeErr != nil {
		return 0, m.writeErr
	}
	if off < 0 || off+int64(len(p)) > int64(len(m.data)) {
		return 0, fmt.Errorf("write out of range off=%d len=%d", off, len(p))
	}
	return copy(m.data[off:], p), nil
}

func (m *memDevice) Size() (int64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.sizeErr != nil {
		return 0, m.sizeErr
	}
	if m.sizeValue != nil {
		return *m.sizeValue, nil
	}
	return int64(len(m.data)), nil
}

func (m *memDevice) Sync() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.syncErr != nil {
		return m.syncErr
	}
	m.synced++
	return nil
}

func (m *memDevice) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closed = true
	return m.closeErr
}

func (m *memDevice) snapshot() []byte {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]byte(nil), m.data...)
}

func (m *memDevice) setReadErr(err error) {
	m.mu.Lock()
	m.readErr = err
	m.mu.Unlock()
}

func (m *memDevice) setWriteErr(err error) {
	m.mu.Lock()
	m.writeErr = err
	m.mu.Unlock()
}

// shortWriteDevice writes one byte fewer than requested to drive the short-write
// branch in WriteAt.
type shortWriteDevice struct{ *memDevice }

func (d shortWriteDevice) WriteAt(p []byte, off int64) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	_, _ = d.memDevice.WriteAt(p[:len(p)-1], off)
	return len(p) - 1, nil
}

// --- construction -----------------------------------------------------------

func mustEngine(t *testing.T, devs []*memDevice, cfg Config) *Engine {
	t.Helper()
	reps := make([]Replica, len(devs))
	for i, d := range devs {
		reps[i] = Replica{Name: fmt.Sprintf("r%d", i), Dev: d}
	}
	e, err := New(reps, cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return e
}

func TestNewValidates(t *testing.T) {
	if _, err := New(nil, Config{}); !errors.Is(err, ErrNoReplicas) {
		t.Fatalf("empty err = %v", err)
	}

	// Duplicate names.
	dup := []Replica{
		{Name: "a", Dev: newMem(512)},
		{Name: "a", Dev: newMem(512)},
	}
	if _, err := New(dup, Config{}); !errors.Is(err, ErrDuplicateName) {
		t.Fatalf("dup err = %v", err)
	}

	// Size mismatch.
	mm := []Replica{
		{Name: "a", Dev: newMem(512)},
		{Name: "b", Dev: newMem(1024)},
	}
	if _, err := New(mm, Config{}); !errors.Is(err, ErrSizeMismatch) {
		t.Fatalf("mismatch err = %v", err)
	}

	// Size error from a replica.
	bad := newMem(512)
	bad.sizeErr = errors.New("stat boom")
	if _, err := New([]Replica{{Name: "a", Dev: bad}}, Config{}); err == nil {
		t.Fatal("expected size error")
	}
}

func TestNewConfigClamping(t *testing.T) {
	devs := []*memDevice{newMem(1024), newMem(1024)}
	// MinInSync above replica count is clamped to len(reps).
	e := mustEngine(t, devs, Config{MinInSync: 99, Local: "r1"})
	if e.minInSync != 2 {
		t.Fatalf("minInSync = %d, want 2", e.minInSync)
	}
	if e.localIx != 1 {
		t.Fatalf("localIx = %d, want 1", e.localIx)
	}
	// Negative MinInSync defaults to 1; unmatched Local defaults to 0.
	e2 := mustEngine(t, devs, Config{MinInSync: -5, Local: "nope"})
	if e2.minInSync != 1 || e2.localIx != 0 {
		t.Fatalf("minInSync=%d localIx=%d", e2.minInSync, e2.localIx)
	}
}

// --- synchronous write / read ----------------------------------------------

func TestSynchronousWriteAndRead(t *testing.T) {
	devs := []*memDevice{newMem(4096), newMem(4096), newMem(4096)}
	e := mustEngine(t, devs, Config{})

	if sz, err := e.Size(); err != nil || sz != 4096 {
		t.Fatalf("Size = %d, %v", sz, err)
	}

	payload := bytes.Repeat([]byte{0x5A}, 1024)
	n, err := e.WriteAt(payload, 512)
	if err != nil || n != 1024 {
		t.Fatalf("WriteAt = %d, %v", n, err)
	}
	// Every replica got the write (synchronous, RPO 0).
	for i, d := range devs {
		if !bytes.Equal(d.snapshot()[512:512+1024], payload) {
			t.Fatalf("replica %d missing the write", i)
		}
	}
	// Read back.
	got := make([]byte, 1024)
	if n, err := e.ReadAt(got, 512); err != nil || n != 1024 || !bytes.Equal(got, payload) {
		t.Fatalf("ReadAt = %d, %v, equal=%v", n, err, bytes.Equal(got, payload))
	}

	// Sync flushes all in-sync replicas.
	if err := e.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	for i, d := range devs {
		if d.synced != 1 {
			t.Fatalf("replica %d synced=%d", i, d.synced)
		}
	}

	st := e.Status()
	if len(st) != 3 {
		t.Fatalf("Status len = %d", len(st))
	}
	for _, s := range st {
		if s.State != InSync {
			t.Fatalf("replica %s state = %s", s.Name, s.State)
		}
	}
}

func TestReadPrefersLocalThenFailsOver(t *testing.T) {
	r0 := newMem(64)
	r1 := newMem(64)
	r2 := newMem(64)
	copy(r0.data, bytes.Repeat([]byte{0xA0}, 64))
	copy(r1.data, bytes.Repeat([]byte{0xA1}, 64))
	copy(r2.data, bytes.Repeat([]byte{0xA2}, 64))
	e := mustEngine(t, []*memDevice{r0, r1, r2}, Config{Local: "r1"})

	// Local r1 serves the read.
	got := make([]byte, 8)
	if _, err := e.ReadAt(got, 0); err != nil {
		t.Fatalf("ReadAt: %v", err)
	}
	if !bytes.Equal(got, bytes.Repeat([]byte{0xA1}, 8)) {
		t.Fatalf("read not from local r1: %x", got)
	}

	// Fail the local read → fail over to the next in-sync replica (r0).
	r1.setReadErr(errors.New("local down"))
	if _, err := e.ReadAt(got, 0); err != nil {
		t.Fatalf("failover ReadAt: %v", err)
	}
	if !bytes.Equal(got, bytes.Repeat([]byte{0xA0}, 8)) {
		t.Fatalf("failover did not reach r0: %x", got)
	}
}

func TestReadAllFail(t *testing.T) {
	r0 := newMem(64)
	r1 := newMem(64)
	r0.setReadErr(errors.New("r0 down"))
	r1.setReadErr(errors.New("r1 down"))
	e := mustEngine(t, []*memDevice{r0, r1}, Config{})
	if _, err := e.ReadAt(make([]byte, 8), 0); err == nil {
		t.Fatal("expected all-replicas-failed read error")
	}
}

// --- degrade rule -----------------------------------------------------------

func TestWriteDegradesReplicaButSucceeds(t *testing.T) {
	r0 := newMem(1024)
	r1 := newMem(1024)
	r1.setWriteErr(errors.New("r1 write down"))
	e := mustEngine(t, []*memDevice{r0, r1}, Config{MinInSync: 1})

	// r1 errors but r0 acks and minInSync=1 → write succeeds, r1 → out-of-sync.
	if _, err := e.WriteAt(make([]byte, 16), 0); err != nil {
		t.Fatalf("WriteAt should degrade-succeed: %v", err)
	}
	st := e.Status()
	if st[0].State != InSync || st[1].State != OutOfSync {
		t.Fatalf("states = %v", st)
	}
	// A subsequent read avoids the out-of-sync replica.
	if _, err := e.ReadAt(make([]byte, 16), 0); err != nil {
		t.Fatalf("ReadAt after degrade: %v", err)
	}
}

func TestWriteFailsBelowMinimum(t *testing.T) {
	r0 := newMem(1024)
	r1 := newMem(1024)
	r0.setWriteErr(errors.New("r0 down"))
	r1.setWriteErr(errors.New("r1 down"))
	e := mustEngine(t, []*memDevice{r0, r1}, Config{MinInSync: 1})

	// Both error → 0 survivors < 1 → write fails, both out-of-sync.
	if _, err := e.WriteAt(make([]byte, 16), 0); !errors.Is(err, ErrNoInSync) {
		t.Fatalf("WriteAt err = %v, want ErrNoInSync", err)
	}
	for _, s := range e.Status() {
		if s.State != OutOfSync {
			t.Fatalf("replica %s state = %s", s.Name, s.State)
		}
	}
	// With no in-sync replicas left, further writes and reads fail fast.
	if _, err := e.WriteAt(make([]byte, 16), 0); !errors.Is(err, ErrNoInSync) {
		t.Fatalf("WriteAt with none in-sync = %v", err)
	}
	if _, err := e.ReadAt(make([]byte, 16), 0); !errors.Is(err, ErrNoInSync) {
		t.Fatalf("ReadAt with none in-sync = %v", err)
	}
	if err := e.Sync(); !errors.Is(err, ErrNoInSync) {
		t.Fatalf("Sync with none in-sync = %v", err)
	}
}

func TestWriteQuorumThreeReplicas(t *testing.T) {
	// MinInSync=2 over 3 replicas: one failure is tolerated, two is not.
	r0, r1, r2 := newMem(1024), newMem(1024), newMem(1024)
	e := mustEngine(t, []*memDevice{r0, r1, r2}, Config{MinInSync: 2})

	r2.setWriteErr(errors.New("r2 down"))
	if _, err := e.WriteAt(make([]byte, 16), 0); err != nil {
		t.Fatalf("one failure should be tolerated: %v", err)
	}
	// Now drop r1 too → only r0 acks, below quorum of 2.
	r1.setWriteErr(errors.New("r1 down"))
	if _, err := e.WriteAt(make([]byte, 16), 0); !errors.Is(err, ErrNoInSync) {
		t.Fatalf("two failures should fail quorum: %v", err)
	}
}

func TestShortWriteDegrades(t *testing.T) {
	r0 := newMem(1024)
	r1 := shortWriteDevice{newMem(1024)}
	e, err := New([]Replica{
		{Name: "r0", Dev: r0},
		{Name: "r1", Dev: r1},
	}, Config{MinInSync: 1})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := e.WriteAt(make([]byte, 16), 0); err != nil {
		t.Fatalf("short write should degrade-succeed: %v", err)
	}
	if e.Status()[1].State != OutOfSync {
		t.Fatal("short-writing replica should be out-of-sync")
	}
}

func TestSyncDegrades(t *testing.T) {
	r0 := newMem(1024)
	r1 := newMem(1024)
	r1.syncErr = errors.New("flush down")
	e := mustEngine(t, []*memDevice{r0, r1}, Config{MinInSync: 1})
	if err := e.Sync(); err != nil {
		t.Fatalf("Sync should degrade-succeed: %v", err)
	}
	if e.Status()[1].State != OutOfSync {
		t.Fatal("flush-failing replica should be out-of-sync")
	}
}

// --- rebuild ----------------------------------------------------------------

func TestRebuildResyncsOutOfSyncReplica(t *testing.T) {
	r0 := newMem(4096)
	r1 := newMem(4096)
	copy(r0.data, bytes.Repeat([]byte{0xCC}, 4096))
	copy(r1.data, bytes.Repeat([]byte{0xCC}, 4096))
	e := mustEngine(t, []*memDevice{r0, r1}, Config{MinInSync: 1})

	// Knock r1 out of sync via a failed write, then heal it.
	r1.setWriteErr(errors.New("transient"))
	if _, err := e.WriteAt(bytes.Repeat([]byte{0xEE}, 256), 0); err != nil {
		t.Fatalf("WriteAt: %v", err)
	}
	if e.Status()[1].State != OutOfSync {
		t.Fatal("r1 should be out-of-sync")
	}
	r1.setWriteErr(nil) // r1 is healthy again

	if err := e.Rebuild(context.Background(), "r1", 512); err != nil {
		t.Fatalf("Rebuild: %v", err)
	}
	if e.Status()[1].State != InSync {
		t.Fatal("r1 should be in-sync after rebuild")
	}
	if !bytes.Equal(r0.snapshot(), r1.snapshot()) {
		t.Fatal("rebuild did not converge r1 to r0")
	}
}

func TestRebuildWithLiveWrites(t *testing.T) {
	r0 := newMem(1 << 16)
	r1 := newMem(1 << 16)
	for i := range r0.data {
		r0.data[i] = byte(i)
		r1.data[i] = byte(i)
	}
	e := mustEngine(t, []*memDevice{r0, r1}, Config{MinInSync: 1})

	// Demote r1.
	r1.setWriteErr(errors.New("down"))
	if _, err := e.WriteAt([]byte{1, 2, 3, 4}, 0); err != nil {
		t.Fatalf("WriteAt: %v", err)
	}
	r1.setWriteErr(nil)

	// Rebuild concurrently with live writes. Because r1 is Rebuilding, the live
	// writes are mirrored to it too, and the source (r0) sees them as well, so
	// the copy converges.
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 200; i++ {
			off := int64((i * 97) % (1<<16 - 8))
			if _, err := e.WriteAt([]byte{byte(i), byte(i + 1), byte(i + 2), byte(i + 3)}, off); err != nil {
				t.Errorf("live WriteAt: %v", err)
				return
			}
		}
	}()
	if err := e.Rebuild(context.Background(), "r1", 4096); err != nil {
		t.Fatalf("Rebuild: %v", err)
	}
	wg.Wait()

	if e.Status()[1].State != InSync {
		t.Fatal("r1 should be in-sync after rebuild")
	}
	// After the rebuild and all writes, both replicas must agree.
	if !bytes.Equal(r0.snapshot(), r1.snapshot()) {
		t.Fatal("replicas diverged after rebuild with live writes")
	}
}

func TestRebuildUnknownReplica(t *testing.T) {
	e := mustEngine(t, []*memDevice{newMem(512), newMem(512)}, Config{})
	if err := e.Rebuild(context.Background(), "nope", 0); !errors.Is(err, ErrUnknownReplica) {
		t.Fatalf("err = %v, want ErrUnknownReplica", err)
	}
}

func TestRebuildNoSource(t *testing.T) {
	// Single replica: nothing to rebuild from.
	e := mustEngine(t, []*memDevice{newMem(512)}, Config{})
	if err := e.Rebuild(context.Background(), "r0", 0); !errors.Is(err, ErrNoInSync) {
		t.Fatalf("err = %v, want ErrNoInSync", err)
	}
}

func TestRebuildSourceReadError(t *testing.T) {
	r0 := newMem(4096)
	r1 := newMem(4096)
	e := mustEngine(t, []*memDevice{r0, r1}, Config{MinInSync: 1})
	// Demote r1.
	r1.setWriteErr(errors.New("down"))
	_, _ = e.WriteAt(make([]byte, 16), 0)
	r1.setWriteErr(nil)
	// Source r0 read fails during rebuild.
	r0.setReadErr(errors.New("source dead"))
	if err := e.Rebuild(context.Background(), "r1", 512); err == nil {
		t.Fatal("expected source read error")
	}
	if e.Status()[1].State != OutOfSync {
		t.Fatal("target should be back to out-of-sync after aborted rebuild")
	}
}

func TestRebuildTargetWriteError(t *testing.T) {
	r0 := newMem(4096)
	r1 := newMem(4096)
	e := mustEngine(t, []*memDevice{r0, r1}, Config{MinInSync: 1})
	r1.setWriteErr(errors.New("down"))
	_, _ = e.WriteAt(make([]byte, 16), 0)
	// Leave r1 write erroring so the rebuild's own WriteAt to the target fails.
	if err := e.Rebuild(context.Background(), "r1", 512); err == nil {
		t.Fatal("expected target write error")
	}
}

func TestRebuildCancelled(t *testing.T) {
	r0 := newMem(1 << 20)
	r1 := newMem(1 << 20)
	e := mustEngine(t, []*memDevice{r0, r1}, Config{MinInSync: 1})
	r1.setWriteErr(errors.New("down"))
	_, _ = e.WriteAt(make([]byte, 16), 0)
	r1.setWriteErr(nil)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel before the first chunk
	if err := e.Rebuild(ctx, "r1", 4096); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if e.Status()[1].State != OutOfSync {
		t.Fatal("target should remain out-of-sync after cancellation")
	}
}

func TestRebuildDefaultChunkSize(t *testing.T) {
	r0 := newMem(2048)
	r1 := newMem(2048)
	copy(r0.data, bytes.Repeat([]byte{0x9}, 2048))
	e := mustEngine(t, []*memDevice{r0, r1}, Config{MinInSync: 1})
	r1.setWriteErr(errors.New("down"))
	_, _ = e.WriteAt(make([]byte, 16), 0)
	r1.setWriteErr(nil)
	// chunkSize <= 0 → DefaultChunkSize (one chunk covers the whole 2KiB volume).
	if err := e.Rebuild(context.Background(), "r1", 0); err != nil {
		t.Fatalf("Rebuild: %v", err)
	}
	if !bytes.Equal(r0.snapshot(), r1.snapshot()) {
		t.Fatal("rebuild with default chunk did not converge")
	}
}

// copyChunk holds the engine's writeMu across each chunk's (read source → write
// target), so a concurrent live write to that chunk cannot interleave and leave
// the target stale — that atomicity is what TestRebuildWithLiveWrites exercises.
// Its mid-rebuild error branches (target demoted by a prior live write, or the
// last in-sync source lost) are driven deterministically here, white-box, since
// forcing the interleaving with a synchronous live write would (correctly) block
// on writeMu.
func TestCopyChunkTargetDemoted(t *testing.T) {
	r0, r1 := newMem(4096), newMem(4096)
	e := mustEngine(t, []*memDevice{r0, r1}, Config{MinInSync: 1})
	target := e.byName["r1"]
	target.state = OutOfSync // a prior live write demoted it mid-rebuild
	if err := e.copyChunk(target, make([]byte, 512), 0); !errors.Is(err, ErrNoInSync) {
		t.Fatalf("copyChunk on a demoted target: err=%v, want ErrNoInSync", err)
	}
}

func TestCopyChunkNoSource(t *testing.T) {
	r0, r1 := newMem(4096), newMem(4096)
	e := mustEngine(t, []*memDevice{r0, r1}, Config{MinInSync: 1})
	target := e.byName["r1"]
	target.state = Rebuilding
	e.byName["r0"].state = OutOfSync // the only other replica is gone → no source
	if err := e.copyChunk(target, make([]byte, 512), 0); !errors.Is(err, ErrNoInSync) {
		t.Fatalf("copyChunk with no in-sync source: err=%v, want ErrNoInSync", err)
	}
}

func TestWriteDemotesFailingRebuildingReplica(t *testing.T) {
	// A live write during a rebuild mirrors to the in-sync set AND the rebuilding
	// target. If the rebuilding target's write fails, it drops back to out-of-sync
	// (its rebuild is now invalid) while the write still succeeds via the in-sync
	// replica. This drives the Rebuilding arm of targets() and demote().
	r0, r1 := newMem(4096), newMem(4096)
	e := mustEngine(t, []*memDevice{r0, r1}, Config{MinInSync: 1})
	e.byName["r1"].state = Rebuilding
	r1.setWriteErr(errors.New("rebuild target write failed"))
	if _, err := e.WriteAt([]byte{1, 2, 3}, 0); err != nil {
		t.Fatalf("WriteAt: in-sync r0 acked, want success, got %v", err)
	}
	if e.Status()[1].State != OutOfSync {
		t.Fatalf("rebuilding replica state = %s, want out-of-sync after a failed write", e.Status()[1].State)
	}
}

// --- close / closed-state ---------------------------------------------------

func TestCloseAndClosedOps(t *testing.T) {
	r0 := newMem(512)
	r1 := newMem(512)
	r1.closeErr = errors.New("close boom")
	e := mustEngine(t, []*memDevice{r0, r1}, Config{})
	if err := e.Close(); err == nil {
		t.Fatal("expected joined close error")
	}
	if !r0.closed || !r1.closed {
		t.Fatal("replicas not closed")
	}
	// Idempotent.
	if err := e.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	// All ops fail after close.
	if _, err := e.Size(); !errors.Is(err, ErrClosed) {
		t.Fatalf("Size after close = %v", err)
	}
	if _, err := e.WriteAt(make([]byte, 4), 0); !errors.Is(err, ErrClosed) {
		t.Fatalf("WriteAt after close = %v", err)
	}
	if _, err := e.ReadAt(make([]byte, 4), 0); !errors.Is(err, ErrClosed) {
		t.Fatalf("ReadAt after close = %v", err)
	}
	if err := e.Sync(); !errors.Is(err, ErrClosed) {
		t.Fatalf("Sync after close = %v", err)
	}
	if err := e.Rebuild(context.Background(), "r0", 0); !errors.Is(err, ErrClosed) {
		t.Fatalf("Rebuild after close = %v", err)
	}
}

func TestRebuildLocalIsTarget(t *testing.T) {
	// When the preferred-local replica IS the rebuild target, pickSource must
	// fall through to another in-sync replica.
	r0 := newMem(2048)
	r1 := newMem(2048)
	r2 := newMem(2048)
	copy(r0.data, bytes.Repeat([]byte{0x3}, 2048))
	copy(r1.data, bytes.Repeat([]byte{0x3}, 2048))
	copy(r2.data, bytes.Repeat([]byte{0x3}, 2048))
	e := mustEngine(t, []*memDevice{r0, r1, r2}, Config{MinInSync: 1, Local: "r1"})

	// Demote r1 (the local).
	r1.setWriteErr(errors.New("down"))
	_, _ = e.WriteAt(make([]byte, 16), 0)
	r1.setWriteErr(nil)

	if err := e.Rebuild(context.Background(), "r1", 512); err != nil {
		t.Fatalf("Rebuild: %v", err)
	}
	if e.Status()[1].State != InSync {
		t.Fatal("r1 should be in-sync")
	}
}

func TestFinishRebuildDemoted(t *testing.T) {
	// Directly exercise finishRebuild's "demoted mid-rebuild" branch: the target
	// is OutOfSync (not Rebuilding) when finishRebuild runs.
	e := mustEngine(t, []*memDevice{newMem(512), newMem(512)}, Config{})
	target := e.reps[1]
	e.mu.Lock()
	target.state = OutOfSync
	e.mu.Unlock()
	if err := e.finishRebuild(target); !errors.Is(err, ErrNoInSync) {
		t.Fatalf("finishRebuild demoted = %v, want ErrNoInSync", err)
	}
}

func TestPickSourceNone(t *testing.T) {
	// pickSource returns false when no in-sync replica other than target exists.
	e := mustEngine(t, []*memDevice{newMem(512), newMem(512)}, Config{})
	t0, t1 := e.reps[0], e.reps[1]
	e.mu.Lock()
	t0.state = OutOfSync // only t1 in-sync; pick a source for t1 → none
	e.mu.Unlock()
	if _, ok := e.pickSource(t1); ok {
		t.Fatal("expected no source")
	}
}

func TestRebuildFinishAfterClose(t *testing.T) {
	// Close the engine between the stream completing and finishRebuild: covered
	// indirectly here by closing concurrently is racy, so instead exercise
	// finishRebuild's closed branch directly.
	e := mustEngine(t, []*memDevice{newMem(512), newMem(512)}, Config{})
	target := e.reps[1]
	e.mu.Lock()
	target.state = Rebuilding
	e.closed = true
	e.mu.Unlock()
	if err := e.finishRebuild(target); !errors.Is(err, ErrClosed) {
		t.Fatalf("finishRebuild after close = %v", err)
	}
}

// --- Fencer seam / State.String ---------------------------------------------

type recordingFencer struct {
	mu      sync.Mutex
	fenced  []string
	fireErr error
}

func (f *recordingFencer) Fence(ctx context.Context, replica string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.fenced = append(f.fenced, replica)
	return f.fireErr
}

func TestFencerSeam(t *testing.T) {
	// No fencer by default.
	e := mustEngine(t, []*memDevice{newMem(512)}, Config{})
	if e.Fencer() != nil {
		t.Fatal("expected nil Fencer")
	}

	f := &recordingFencer{fireErr: errors.New("stonith failed")}
	e2 := mustEngine(t, []*memDevice{newMem(512)}, Config{Fencer: f})
	got := e2.Fencer()
	if got == nil {
		t.Fatal("expected configured Fencer")
	}
	// The engine never calls Fence itself; the control plane would. Exercise it.
	if err := got.Fence(context.Background(), "r0"); err == nil {
		t.Fatal("expected fence error")
	}
	if len(f.fenced) != 1 || f.fenced[0] != "r0" {
		t.Fatalf("fenced = %v", f.fenced)
	}
}

func TestStateString(t *testing.T) {
	cases := map[State]string{
		InSync:     "in-sync",
		OutOfSync:  "out-of-sync",
		Rebuilding: "rebuilding",
		State(99):  "State(99)",
	}
	for s, want := range cases {
		if got := s.String(); got != want {
			t.Fatalf("State(%d).String() = %q, want %q", int(s), got, want)
		}
	}
}

// TestEngineIsDevice is a compile-and-run interface assertion.
func TestEngineIsDevice(t *testing.T) {
	var _ volume.Device = (*Engine)(nil)
}

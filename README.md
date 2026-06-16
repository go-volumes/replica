# go-volumes/replica

A pure-Go (`CGO_ENABLED=0`), standard-library-only **replication engine** for
highly-available block volumes in the go-volumes family. It fronts N synchronous
replicas — each any [`volume.Device`](https://github.com/go-volumes/interface)
(in production an [NBD client](https://github.com/go-volumes/nbd) to a remote
pool volume, in tests an in-memory device) — behind a single `Engine` that
itself satisfies `volume.Device`. A filesystem-format driver or an NBD server
writes through the `Engine` exactly as it would a local device.

## Model

Single-active-writer with synchronous replication (**RPO 0**):

- **WriteAt** mirrors every write to all in-sync replicas concurrently and acks
  only once they all ack. A replica that errors is marked **out-of-sync** and
  dropped from the write set; the write still succeeds as long as at least
  `MinInSync` in-sync replicas remain, otherwise it fails (`ErrNoInSync`).
- **ReadAt** serves from one in-sync replica, preferring the configured `Local`
  one, failing over to the next on a read error.
- **Sync** flushes every in-sync replica concurrently (same degrade rule).
- **Size** is the replicas' agreed size, validated equal at construction
  (`ErrSizeMismatch` otherwise).
- **Close** closes every replica.

Per-replica state is `InSync | OutOfSync | Rebuilding`, exposed via `Status()`.

`Rebuild(ctx, name, chunkSize)` resyncs an out-of-sync (or new) replica by
streaming the volume from a healthy in-sync source (`ReadAt` → `WriteAt`,
chunked). The target is flipped to `Rebuilding` first, so live writes are
mirrored to it alongside the in-sync set while the stream runs; it flips to
`InSync` once caught up. A live write that fails on the target aborts the
rebuild.

Failover and fencing of a stale writer (STONITH before promotion) are a future
control-plane concern. The engine leaves a `Fencer` seam for it but implements
no consensus here.

## Durability

The `Engine` adds no buffering of its own: durability between the engine and its
replicas relies on each replica's own `Sync` (a pool fsync, an S3 PUT, an NBD
`FLUSH` to a remote backing).

## Usage

```go
e, err := replica.New([]replica.Replica{
    {Name: "local",  Dev: localDev},   // e.g. a same-host pool volume
    {Name: "remote", Dev: nbdClient},  // e.g. an NBD client to another host
}, replica.Config{MinInSync: 1, Local: "local"})
if err != nil { /* ... */ }
defer e.Close()

var dev volume.Device = e             // *Engine satisfies volume.Device
_, _ = dev.WriteAt(buf, off)          // mirrored to all in-sync replicas
_, _ = dev.ReadAt(buf, off)           // served from the local replica
_ = dev.Sync()                        // flush all in-sync replicas

// Heal a degraded replica.
_ = e.Rebuild(context.Background(), "remote", 0) // 0 = DefaultChunkSize
```

## Install

```
go get github.com/go-volumes/replica
```

Dependencies: the Go standard library and `github.com/go-volumes/interface`
only.

## License

BSD-3-Clause — see [LICENSE](LICENSE).

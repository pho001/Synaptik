# ADR 0028: Explicit native parameter copy before training

## Status

Accepted — 2026-10-11. This record explains the [authoritative Training contract](../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle); it does not replace it.

## Context

NN's initialized `Linear` creates ordinary heap-backed parameter Tensors. The public
`TrainingSession` deliberately requires exact-capacity writable, shareable native storage and
captures fixed Parameter, Tensor, and storage identities at open. Callers can rebind Model host
storage themselves, but doing that for a complete Module tree requires repeating Training's
geometry, finite-value, width, alias, and lifetime checks. A partial copy could leave a loss
expression and its parameter tree referring to mixed old and new storage.

## Decision drivers

- Keep Parameter and Tensor identity and existing expressions stable.
- Reject unsupported sources and destination lifetime before any association changes.
- Preserve exact represented bytes, including finite low-precision and signed-zero encodings.
- Keep arena ownership and Engine preparation explicit, without a backend import.

## Options considered

| Option | Benefit | Cost or reason rejected |
|---|---|---|
| Keep only manual caller rebinding | No new API | Every caller must reproduce the complete atomic preparation checks. |
| Copy implicitly in `TrainingSession.open` | One apparent call | Hides allocation and lifetime ownership and changes the native-only open contract. |
| Add explicit Training-owned pre-session copy | One checked transition with caller-selected lifetime | Requires an extra call and may consume arena capacity even if a later allocation fails. |

The table compares lifecycle ownership and failure visibility, not copy throughput.

## Decision

Add `TrainingParameterStorage.copyToSharedNative(Module, Arena)` as a backend-neutral, explicit
pre-session operation. It validates a complete non-empty recursive parameter snapshot, rejects
duplicate Tensor identities and overlapping source regions, stages fresh native copies in path
order, then installs their associations without replacing wrappers or Tensors. An unexpected
ordinary install failure restores earlier exact associations. The caller supplies and later
closes a shareable Arena after any session using its storage has closed. `TrainingSession.open`
continues to reject heap-backed parameters and captures fixed storage after the copy.

## Rationale

Training already owns the supported optimizer storage domain, while NN owns parameter discovery
and Model owns borrowed storage association. An explicit copy aligns those responsibilities:
Training checks the transition, Model performs the association update, and Engine remains the
only compile/prepare/run path. Raw copying avoids rounding signed zero or finite subnormal
encodings. Complete validation and staging avoid a partially rebound Module on expected failure.

## Consequences

Existing native-backed callers need no change. Heap-initialized callers gain a one-call opt-in
before session open and may reuse a loss expression built from the same Tensors. The destination
Arena and its consumed allocations remain caller-owned; failed allocation/copy does not reclaim
individual allocations, and ordinary rollback cannot undo arena consumption. The caller must
grant exclusive Module/storage use during copy and throughout an open session. Read-only source
storage is allowed, but the destination is writable. No view/stride conversion, backend
residency transfer, implicit session conversion, or optimizer memory reduction follows.

Unit tests prove raw-bit preservation, alias rejection, and no rebinding on validation failure;
the real `Linear` CPU test proves the same-expression session path. A future backend transfer or
different session capture domain needs its own decision and contract update.

## Related documentation

- [Root Training ownership route](../../../ARCHITECTURE.md#module-ownership-routing)
- [Training lifecycle contract](../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)
- [Training graph explanation](../../architecture/training-graph.md#optimizer-as-a-backend-agnostic-step)
- [Training API](../../api/training-api.md)
- [Training user guide](../../user-guide/training.md)

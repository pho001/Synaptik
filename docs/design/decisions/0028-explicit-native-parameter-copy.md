# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0028: Explicit [native parameter copy](../../glossary.md#native-parameter-copy--trainingparameterstorage) before training

## Status

Accepted — 2026-10-11. This record explains the [authoritative Training contract](../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle); it does not replace it.

2026-10-16 [signature](../../glossary.md#term-signature) reconciliation: Training0007 replaces the original [Module](../../glossary.md#term-module)-taking [API](../../glossary.md#term-application-programming-interface-api) below
with `copyToSharedNative(TrainingParameters, Arena)` under
[ADR0039](0039-portable-graph-model-persistence.md#necessary-boundary-types-not-a-new-model-hierarchy).
[NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) callers pass `module.trainingParameters()`; the same adapter remains valid after copying and
can open a session because membership does not freeze storage associations. Training now lives
at `modules/training` without an NN edge. The zero-byte [Arena](../../glossary.md#term-arena) probe, failure precedence, complete
source [validation](../../glossary.md#term-validation), [represented-bit](../../glossary.md#storage-mapping-contiguity-and-injectivity) preservation, association [rollback](../../glossary.md#term-rollback) and caller lifetimes below
remain unchanged; live membership is rechecked before installation.

## Context

[NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)'s initialized [`Linear`](../../glossary.md#term-linear-meanings) creates ordinary [heap](../../glossary.md#term-heap)-backed [parameter](../../glossary.md#term-parameter) [Tensors](../../glossary.md#tensor). The public
[`TrainingSession`](../../glossary.md#training-session--trainingsession) deliberately requires exact-capacity writable, shareable native storage and
[captures](../../glossary.md#term-capture-meanings) fixed Parameter, Tensor, and storage [identities](../../glossary.md#term-identity-meanings) at open. Callers can [rebind](../../glossary.md#capture-graph-ports-and-reconstruction) [Model](../../glossary.md#term-model) [host
storage](../../glossary.md#host-storage) themselves, but doing that for a complete [Module](../../glossary.md#term-module) tree requires repeating Training's
geometry, [finite](../../glossary.md#term-finite)-value, width, [alias](../../glossary.md#aliasing-and-arena), and [lifetime](../../glossary.md#term-lifetime) checks. A partial copy could leave a [loss](../../glossary.md#loss)
expression and its parameter tree referring to mixed old and new storage.

## Decision drivers

- Keep [Parameter](../../glossary.md#term-parameter) and [Tensor](../../glossary.md#tensor) [identity](../../glossary.md#term-identity-meanings) and existing expressions stable.
- Reject unsupported sources and destination [lifetime](../../glossary.md#term-lifetime) before any association changes.
- Preserve exact represented bytes, including [finite](../../glossary.md#term-finite) low-[precision](../../glossary.md#term-precision) and [signed-zero](../../glossary.md#nan-infinity-and-signed-zero) encodings.
- Keep [arena](../../glossary.md#term-arena) [ownership](../../glossary.md#term-ownership-meanings) and [Engine](../../glossary.md#term-engine) preparation explicit, without a [backend](../../glossary.md#backend) import.

## Options considered

| Option | Benefit | Cost or reason rejected |
|---|---|---|
| Keep only manual caller rebinding | No new [API](../../glossary.md#term-application-programming-interface-api) | Every caller must reproduce the complete [atomic](../../glossary.md#resource-transaction-and-rollback) preparation checks. |
| Copy implicitly in `TrainingSession.open` | One apparent call | Hides allocation and [lifetime](../../glossary.md#term-lifetime) [ownership](../../glossary.md#term-ownership-meanings) and changes the native-only open contract. |
| Add explicit Training-owned pre-session copy | One checked transition with caller-selected lifetime | Requires an extra call and may consume [arena](../../glossary.md#term-arena) capacity even if a later allocation fails. |

The table compares [lifecycle](../../glossary.md#lifecycle) ownership and failure visibility, not copy [throughput](../../glossary.md#term-throughput).

## Decision

Add `TrainingParameterStorage.copyToSharedNative(Module, Arena)` as a [backend](../../glossary.md#backend)-neutral, explicit
pre-session [operation](../../glossary.md#operation). It validates a complete non-empty recursive [parameter](../../glossary.md#term-parameter) [snapshot](../../glossary.md#term-snapshot), rejects
duplicate [Tensor](../../glossary.md#tensor) [identities](../../glossary.md#term-identity-meanings) and overlapping source regions, stages fresh native copies in path
order, then installs their associations without replacing wrappers or Tensors. An unexpected
ordinary install failure restores earlier exact associations. The caller supplies and later
closes a shareable [Arena](../../glossary.md#term-arena) after any session using its storage has closed. `TrainingSession.open`
continues to reject [heap](../../glossary.md#term-heap)-backed parameters and [captures](../../glossary.md#term-capture-meanings) fixed storage after the copy.

## Rationale

Training already owns the supported [optimizer](../../glossary.md#term-optimizer) storage domain, while [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) owns [parameter](../../glossary.md#term-parameter) discovery
and [Model](../../glossary.md#term-model) owns [borrowed](../../glossary.md#ownership-borrowing-and-lease) storage association. An explicit copy aligns those responsibilities:
Training checks the transition, Model performs the association update, and [Engine](../../glossary.md#term-engine) remains the
only [compile](../../glossary.md#compile)/[prepare](../../glossary.md#prepare)/[run](../../glossary.md#term-run) path. Raw copying avoids [rounding](../../glossary.md#term-rounding) [signed zero](../../glossary.md#term-signed-zero) or [finite](../../glossary.md#term-finite) [subnormal](../../glossary.md#term-subnormal)
encodings. Complete [validation](../../glossary.md#term-validation) and staging avoid a partially rebound [Module](../../glossary.md#term-module) on expected failure.

## Consequences

Existing native-backed callers need no change. [Heap](../../glossary.md#term-heap)-initialized callers gain a one-call opt-in
before session open and may reuse a [loss](../../glossary.md#loss) expression built from the same [Tensors](../../glossary.md#tensor). The destination
[Arena](../../glossary.md#term-arena) and its consumed allocations remain caller-owned; failed allocation/copy does not reclaim
individual allocations, and ordinary [rollback](../../glossary.md#term-rollback) cannot undo arena consumption. The caller must
grant exclusive [Module](../../glossary.md#term-module)/storage use during copy and throughout an open session. Read-only source
storage is allowed, but the destination is writable. No [view](../../glossary.md#term-view-meanings)/[stride](../../glossary.md#term-stride) conversion, [backend](../../glossary.md#backend)
[residency](../../glossary.md#residency) transfer, implicit session conversion, or [optimizer](../../glossary.md#term-optimizer) memory reduction follows.

Unit tests prove raw-bit preservation, [alias](../../glossary.md#aliasing-and-arena) rejection, and no rebinding on [validation](../../glossary.md#term-validation) failure;
the real [`Linear`](../../glossary.md#term-linear-meanings) [CPU](../../glossary.md#term-cpu) test proves the same-expression session path. A future backend transfer or
different session [capture](../../glossary.md#term-capture-meanings) domain needs its own decision and contract update.

## Related documentation

- [Root Training ownership route](../../../ARCHITECTURE.md#module-ownership-routing)<span data-glossary-navigation="true"> ([ownership glossary](../../glossary.md#term-ownership-meanings))</span>
- [Training lifecycle contract](../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../glossary.md#lifecycle))</span>
- [Training graph explanation](../../architecture/training-graph.md#optimizer-as-a-backend-agnostic-step)<span data-glossary-navigation="true"> ([Training graph glossary](../../glossary.md#training-graph))</span>
- [Training API](../../api/training-api.md)<span data-glossary-navigation="true"> ([API glossary](../../glossary.md#term-application-programming-interface-api))</span>
- [Training user guide](../../user-guide/training.md)

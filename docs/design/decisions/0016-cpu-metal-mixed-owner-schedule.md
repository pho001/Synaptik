# ADR 0016: CPU/Metal Mixed-Owner Prepared Schedule

## Status

Accepted — 2026-09-23

## Context

Explicit Engine composition already freezes CPU and Metal capability and availability facts in
registration order, routes a complete single-owner plan by exact `BackendId`, and keeps Runtime on
immutable direct prepared references. Planning already records each logical value's producing and
consuming partitions, and Runtime already defines multiple representations per buffer, explicit
buffer-transfer steps, success-only destination validity, and transactional run-state cleanup.
The missing decision is how one preparation combines several owners without moving selection,
lookup, or physical transfer into Runtime.

The [Runtime, Prepare, and Engine contract](../../architecture/contracts/runtime-prepare-engine.md#compile-ownership-and-mixed-owner-prepare-routing)
and [backend execution contract](../../architecture/contracts/backend-execution.md#metal-backend)
are authoritative. This record explains the selected design and does not replace them.

## Decision drivers

- Preserve exact `BackendId` routing and deterministic registration-order Planning input.
- Reuse the existing logical-memory, staged Prepare, slot assignment, prepared schedule, transfer,
  representation-creation, publication, and rollback contracts.
- Keep graph and registry lookup out of per-run cold binding and the Runtime hot traversal.
- Make transfer direction, representation index, capability, failure, and ownership observable and
  fail closed.
- Support the real Metal NEG domain in both CPU-to-Metal and Metal-to-CPU directions without
  conversion or avoidable per-element copying.
- Preserve CPU-only and single-owner behavior and keep model autotuning CPU-only.

## Options considered

### Shared slot plus one representation per participating owner

Prepare keeps one logical buffer slot and assigns deterministic representation positions to its
producing and distinct consuming owners. Finalizers receive their exact position. Backend
contributions create only assigned representations and workspaces; one shared assembler orders
partition execution, required transfers, and publication.

### Separate physical buffer for every partition boundary

Prepare could allocate another slot for every owner edge. That would duplicate one immutable
logical value, abandon the existing multi-representation validity model, complicate fan-out, and
introduce a second memory and scheduling convention.

### Runtime registry or adapter lookup

Runtime could inspect a backend ID at every boundary and request a transfer implementation. This
would turn the Engine registry into a service locator, move capability failure to execution, and
violate the direct-reference prepared hot path.

### Canonical host-byte materialization between owners

Engine could download to canonical `byte[]` and re-upload through public ingress. That path performs
avoidable conversion and allocation, loses direct physical representation identity, and changes
materialization from an explicit public result operation into hidden execution work.

## Decision

Synaptik selects one shared prepared memory plan with one representation per distinct participating
backend owner.

For each materialized logical value, Prepare orders owners as follows: the producing owner first
when present, otherwise the first consuming owner, followed by remaining distinct consumers in
logical consumer-partition order. A repeated owner reuses its position. The representation index is
part of the backend's buffer assignment and therefore becomes the exact index selected by its
prepared executable. Plan-ordered buffer metadata retains the owner list, and workspace metadata
retains its owning partition.

Engine resolves every planned partition to an exact registered adapter before analysis and
supplies one positional preparation list to the existing `GraphPreparation` transaction. Before
calling any backend analysis, it validates the exact source/destination adapter pair, descriptor,
positive rank-1..16 extents, and checked element/byte geometry of every cross-owner edge. The first
implementation accepts only CPU-to-Metal and Metal-to-CPU transfer of fully static canonical
contiguous `FLOAT32` values within Metal's real NEG capability. Every other path, layout, type,
geometry, or conversion request fails closed without fallback or partial backend preparation.

Each backend contributes representation creators only for its assigned positions and workspace
creators only for its partition-owned positions. One shared assembler emits:

```text
representation creation
partition 0 execution
required transfers before the next owner's first consumer
partition 1 execution
...
dense publication suffix
```

A fan-out value receives one transfer recipe per distinct destination owner. Transfers always copy
from the producing representation, or from the first consumer's caller-input representation when
there is no producer. CPU run-owned native memory is the reusable host staging storage. CPU to
Metal performs one upload from it; Metal to CPU performs one download into it. Transfer recipes
retain direction, checked byte geometry, and direct backend-owned cold binders. Bound actions retain
direct typed source, destination, and staging references.

Exactly one caller-input occurrence is created for each logical input, at its first owner. Shared
buffer assignment begins with bindable input IDs in exact Compiler occurrence order rather than
partition-analysis or backend-declaration order, so Runtime's flattened caller-input recipes cannot
drift from public binding order. The outward prepared handle captures immutable direct
input-adapter and publication-adapter lists. Each run creates fresh borrowed input wrappers and
fresh run-owned representations; each result uses its captured publication adapter. Neither path
re-queries the registry.

Preparation remains transactional across every finalizer, contribution, transfer recipe, schedule
validation, and prepared aggregate. Run-state creation and failed execution retain the existing
reverse cleanup and suppression rules. A transfer failure preserves source validity, leaves the
destination invalid, and becomes the primary run failure. Transfer recipes own no persistent or
per-run physical resource.

`prepareTuned` remains a single-owner CPU workflow and rejects mixed ownership before
representative borrowing or trial work.

## Rationale

The selected design turns Planning's existing producer and consumer facts into physical residency
only during Prepare, where exact owners, descriptors, assignments, and backend collaborations are
available. One shared slot preserves logical identity, while owner-indexed representations fit the
existing Runtime validity model. Passing the representation index through the finalization
assignment prevents backend executables from relying on a global position-zero convention.

A shared assembler keeps one schedule convention and makes order testable. Preflight makes
unsupported transfers a capability failure rather than a partially prepared runtime surprise.
Direct cold binders preserve concrete-backend type ownership while giving Runtime a bound action
that contains no registry, backend ID, provider, reflection, or discovery operation.

## Consequences

### Positive

- CPU-only and single-owner recipes use the same composition path and retain position zero.
- Mixed schedules support interleaved partitions and fan-out without duplicate logical slots.
- Transfer ordering and validity transitions are explicit schedule behavior.
- Repeated prepared runs share immutable recipes but own independent representations and staging.
- Prepare and run failure cleanup remain deterministic and attempt-all.

### Negative and risks

- Buffer assignments and schedule contexts carry additional owner and representation-position
  metadata.
- Backend schedule assemblers must expose exact physical creation contributions instead of assuming
  they own the complete graph.
- The initial transfer domain is intentionally narrow; non-contiguous layouts, non-`FLOAT32` data,
  heap staging, conversion, and other backend pairs remain unavailable.
- Engine has a built-in CPU/Metal transfer adapter because no generic plugin or transfer registry is
  authorized.

### Migration, testing, and follow-up

The single-owner mixed rejection is removed atomically with caller migration. Durable tests cover
owner/index assignment, executed repeated/interleaved ordering and fan-out, both directions,
repeated runs, oversized/rank/extent preflight before either backend analysis, transactional
rollback and close failure, direct adapter capture, and absence of post-prepare registry lookup. A
real Apple Metal lifecycle test builds the native library, runs both directions, publishes Metal-
and CPU-owned values together, poisons registry lookup after preparation, and still reuses and
materializes through public `Engine.builder()` composition. Any later data type, layout, device,
or backend pair requires an explicit capability extension and evidence; it must not weaken the
fail-closed default.

## Related documentation

- [Authoritative Runtime, Prepare, and Engine contract](../../architecture/contracts/runtime-prepare-engine.md#compile-ownership-and-mixed-owner-prepare-routing)
- [Authoritative backend execution contract](../../architecture/contracts/backend-execution.md#metal-backend)
- [ADR 0015: Explicit Engine backend composition](0015-explicit-engine-backend-composition.md)
- [Engine task 0016](../../planning/modules/engine/tasks/0016-cpu-metal-mixed-owner-schedule.md)
- [Runtime, Prepare, and backend boundary explanation](../../architecture/runtime-prepare-backend-boundary.md)

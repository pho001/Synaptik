# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0016: [CPU](../../glossary.md#term-cpu)/[Metal](../../glossary.md#term-metal) Mixed-[Owner](../../glossary.md#term-owner) [Prepared Schedule](../../glossary.md#prepared-schedule--preparedschedule)

## Status

Accepted — 2026-09-23

## Context

Explicit [Engine](../../glossary.md#term-engine) composition already freezes [CPU](../../glossary.md#term-cpu) and [Metal](../../glossary.md#term-metal) capability and availability facts in
registration order, routes a complete single-[owner](../../glossary.md#term-owner) plan by exact [`BackendId`](../../glossary.md#backend-identity--backendid), and keeps [Runtime](../../glossary.md#term-runtime) on
[immutable](../../glossary.md#term-immutable) direct prepared references. [Planning](../../glossary.md#term-planning) already records each logical value's producing and
consuming partitions, and Runtime already defines multiple representations per [buffer](../../glossary.md#term-buffer), explicit
buffer-transfer steps, success-only destination validity, and transactional [run](../../glossary.md#term-run)-state cleanup.
The missing decision is how one preparation combines several owners without moving selection,
[lookup](../../glossary.md#term-lookup), or physical transfer into Runtime.

The [Runtime, Prepare, and Engine contract](../../architecture/contracts/runtime-prepare-engine.md#compile-ownership-and-mixed-owner-prepare-routing)<span data-glossary-navigation="true"> ([Prepare glossary](../../glossary.md#prepare))</span>
and [backend execution contract](../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([backend glossary](../../glossary.md#backend))</span>
are authoritative. This record explains the selected design and does not replace them.

## Decision drivers

- Preserve exact [`BackendId`](../../glossary.md#backend-identity--backendid) routing and deterministic registration-order [Planning](../../glossary.md#term-planning) input.
- Reuse the existing logical-memory, staged Prepare, slot assignment, prepared schedule, transfer,
  representation-creation, publication, and rollback contracts.<span data-glossary-navigation="true"> ([Prepare glossary](../../glossary.md#prepare); [prepared schedule glossary](../../glossary.md#prepared-schedule--preparedschedule); [publication glossary](../../glossary.md#term-publication); [rollback glossary](../../glossary.md#term-rollback))</span>
- Keep [graph](../../glossary.md#graph) and [registry](../../glossary.md#term-registry) [lookup](../../glossary.md#term-lookup) out of per-[run](../../glossary.md#term-run) cold [binding](../../glossary.md#term-binding) and the [Runtime](../../glossary.md#term-runtime) hot traversal.
- Make transfer direction, representation index, capability, failure, and ownership observable and
  fail closed.<span data-glossary-navigation="true"> ([ownership glossary](../../glossary.md#term-ownership-meanings); [fail closed glossary](../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- Support the real Metal NEG domain in both CPU-to-Metal and Metal-to-CPU directions without
  conversion or avoidable per-element copying.<span data-glossary-navigation="true"> ([Metal glossary](../../glossary.md#term-metal); [element glossary](../../glossary.md#term-element))</span>
- Preserve [CPU](../../glossary.md#term-cpu)-only and single-[owner](../../glossary.md#term-owner) behavior and keep [model autotuning](../../glossary.md#autotuning--model-autotuning) CPU-only.

## Options considered

### Shared slot plus one representation per participating [owner](../../glossary.md#term-owner)

[Prepare](../../glossary.md#prepare) keeps one logical [buffer slot](../../glossary.md#buffer-slot--bufferslot) and assigns deterministic representation positions to its
producing and distinct consuming owners. Finalizers receive their exact position. [Backend](../../glossary.md#backend)
contributions create only assigned representations and workspaces; one shared assembler orders
[partition](../../glossary.md#partition) execution, required transfers, and [publication](../../glossary.md#term-publication).

### Separate physical [buffer](../../glossary.md#term-buffer) for every [partition](../../glossary.md#partition) boundary

[Prepare](../../glossary.md#prepare) could allocate another slot for every [owner](../../glossary.md#term-owner) edge. That would duplicate one [immutable](../../glossary.md#term-immutable)
logical value, abandon the existing multi-representation validity [model](../../glossary.md#term-model), complicate [fan-out](../../glossary.md#term-fan-out), and
introduce a second memory and scheduling convention.

### [Runtime](../../glossary.md#term-runtime) [registry](../../glossary.md#term-registry) or adapter [lookup](../../glossary.md#term-lookup)

Runtime could inspect a [backend](../../glossary.md#backend) ID at every boundary and request a transfer implementation. This
would turn the [Engine](../../glossary.md#term-engine) registry into a [service locator](../../glossary.md#term-service-locator), move capability failure to execution, and
violate the direct-reference prepared [hot path](../../glossary.md#term-hot-path).

### [Canonical](../../glossary.md#term-canonical-form) [host](../../glossary.md#term-host)-byte [materialization](../../glossary.md#materialization) between [owners](../../glossary.md#term-owner)

[Engine](../../glossary.md#term-engine) could download to canonical `byte[]` and re-upload through public ingress. That path performs
avoidable conversion and allocation, loses direct physical representation [identity](../../glossary.md#term-identity-meanings), and changes
materialization from an explicit public result [operation](../../glossary.md#operation) into hidden execution work.

## Decision

Synaptik selects one shared [prepared memory plan](../../glossary.md#prepared-memory-plan--preparedmemoryplan) with one representation per distinct participating
[backend](../../glossary.md#backend) [owner](../../glossary.md#term-owner).

For each materialized logical value, [Prepare](../../glossary.md#prepare) orders owners as follows: the producing owner first
when present, otherwise the first consuming owner, followed by remaining distinct consumers in
logical consumer-[partition](../../glossary.md#partition) order. A repeated owner reuses its position. The representation index is
part of the backend's [buffer](../../glossary.md#term-buffer) assignment and therefore becomes the exact index selected by its
[prepared executable](../../glossary.md#prepared-executable--preparedexecutable). Plan-ordered buffer [metadata](../../glossary.md#term-metadata) retains the owner list, and workspace metadata
retains its owning partition.

[Engine](../../glossary.md#term-engine) resolves every planned partition to an exact registered adapter before analysis and
supplies one positional preparation list to the existing [`GraphPreparation`](../../glossary.md#graph-preparation--graphpreparation) [transaction](../../glossary.md#term-transaction). Before
calling any backend analysis, it validates the exact source/destination adapter pair, [descriptor](../../glossary.md#term-descriptor),
positive [rank](../../glossary.md#term-rank)-1..16 extents, and checked [element](../../glossary.md#term-element)/byte geometry of every cross-owner edge. The first
implementation accepts only [CPU](../../glossary.md#term-cpu)-to-[Metal](../../glossary.md#term-metal) and Metal-to-CPU transfer of fully static [canonical](../../glossary.md#term-canonical-form)
[contiguous](../../glossary.md#term-contiguous) [`FLOAT32`](../../glossary.md#term-float32) values within Metal's real NEG capability. Every other path, [layout](../../glossary.md#layout), type,
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

A [fan-out](../../glossary.md#term-fan-out) value receives one transfer recipe per distinct destination owner. Transfers always copy
from the producing representation, or from the first consumer's caller-input representation when
there is no [producer](../../glossary.md#term-producer). CPU [run](../../glossary.md#term-run)-owned [native memory](../../glossary.md#aliasing-and-arena) is the reusable [host](../../glossary.md#term-host) staging storage. CPU to
Metal performs one upload from it; Metal to CPU performs one download into it. Transfer recipes
retain direction, checked byte geometry, and direct backend-owned cold binders. Bound actions retain
direct typed source, destination, and staging references.

Exactly one caller-input [occurrence](../../glossary.md#term-occurrence) is created for each logical input, at its first owner. Shared
buffer assignment begins with [bindable input](../../glossary.md#bindable-input) IDs in exact [Compiler](../../glossary.md#term-compiler) occurrence order rather than
partition-analysis or backend-declaration order, so [Runtime](../../glossary.md#term-runtime)'s flattened caller-input recipes cannot
drift from public [binding](../../glossary.md#term-binding) order. The outward prepared handle [captures](../../glossary.md#term-capture-meanings) [immutable](../../glossary.md#term-immutable) direct
input-adapter and [publication](../../glossary.md#term-publication)-adapter lists. Each run creates fresh [borrowed](../../glossary.md#ownership-borrowing-and-lease) input wrappers and
fresh run-owned representations; each result uses its captured publication adapter. Neither path
re-queries the [registry](../../glossary.md#term-registry).

Preparation remains transactional across every finalizer, contribution, transfer recipe, schedule
[validation](../../glossary.md#term-validation), and prepared aggregate. Run-state creation and failed execution retain the existing
reverse cleanup and suppression rules. A transfer failure preserves source validity, leaves the
destination invalid, and becomes the primary run failure. Transfer recipes own no persistent or
per-run physical resource.

`prepareTuned` remains a single-owner CPU workflow and rejects mixed [ownership](../../glossary.md#term-ownership-meanings) before
representative [borrowing](../../glossary.md#term-borrowing) or trial work.

## Rationale

The selected design turns [Planning](../../glossary.md#term-planning)'s existing [producer](../../glossary.md#term-producer) and consumer facts into physical [residency](../../glossary.md#residency)
only during [Prepare](../../glossary.md#prepare), where exact [owners](../../glossary.md#term-owner), [descriptors](../../glossary.md#term-descriptor), assignments, and [backend](../../glossary.md#backend) collaborations are
available. One shared slot preserves logical [identity](../../glossary.md#term-identity-meanings), while [owner](../../glossary.md#term-owner)-indexed representations fit the
existing [Runtime](../../glossary.md#term-runtime) validity [model](../../glossary.md#term-model). Passing the representation index through the finalization
assignment prevents backend executables from relying on a global position-zero convention.

A shared assembler keeps one schedule convention and makes order testable. [Preflight](../../glossary.md#term-preflight) makes
unsupported transfers a capability failure rather than a partially prepared runtime surprise.
Direct cold binders preserve concrete-backend type [ownership](../../glossary.md#term-ownership-meanings) while giving Runtime a bound action
that contains no [registry](../../glossary.md#term-registry), backend ID, provider, reflection, or discovery [operation](../../glossary.md#operation).

## Consequences

### Positive

- [CPU](../../glossary.md#term-cpu)-only and single-[owner](../../glossary.md#term-owner) recipes use the same composition path and retain position zero.
- Mixed schedules support interleaved partitions and [fan-out](../../glossary.md#term-fan-out) without duplicate logical slots.
- Transfer ordering and validity transitions are explicit schedule behavior.
- Repeated prepared runs share [immutable](../../glossary.md#term-immutable) recipes but own independent representations and staging.
- [Prepare](../../glossary.md#prepare) and [run](../../glossary.md#term-run) failure cleanup remain deterministic and attempt-all.

### Negative and risks

- Buffer assignments and schedule contexts carry additional owner and representation-position
  metadata.<span data-glossary-navigation="true"> ([Buffer glossary](../../glossary.md#term-buffer); [carry glossary](../../glossary.md#term-carry); [owner glossary](../../glossary.md#term-owner); [metadata glossary](../../glossary.md#term-metadata))</span>
- Backend schedule assemblers must expose exact physical creation contributions instead of assuming
  they own the complete graph.<span data-glossary-navigation="true"> ([Backend glossary](../../glossary.md#backend); [graph glossary](../../glossary.md#graph))</span>
- The initial transfer domain is intentionally narrow; non-[contiguous](../../glossary.md#term-contiguous) [layouts](../../glossary.md#layout), non-[`FLOAT32`](../../glossary.md#term-float32) data,
  heap staging, conversion, and other backend pairs remain unavailable.<span data-glossary-navigation="true"> ([heap glossary](../../glossary.md#term-heap))</span>
- Engine has a built-in CPU/Metal transfer adapter because no generic plugin or transfer registry is
  authorized.<span data-glossary-navigation="true"> ([Engine glossary](../../glossary.md#term-engine); [CPU glossary](../../glossary.md#term-cpu); [Metal glossary](../../glossary.md#term-metal); [plugin glossary](../../glossary.md#term-plugin); [registry glossary](../../glossary.md#term-registry))</span>

### Migration, testing, and follow-up

The single-[owner](../../glossary.md#term-owner) mixed rejection is removed atomically with caller migration. Durable tests cover
owner/index assignment, executed repeated/interleaved ordering and [fan-out](../../glossary.md#term-fan-out), both directions,
repeated runs, oversized/[rank](../../glossary.md#term-rank)/[extent](../../glossary.md#term-extent) [preflight](../../glossary.md#term-preflight) before either [backend](../../glossary.md#backend) analysis, transactional
[rollback](../../glossary.md#term-rollback) and close failure, direct adapter [capture](../../glossary.md#term-capture-meanings), and absence of post-[prepare](../../glossary.md#prepare) [registry](../../glossary.md#term-registry) [lookup](../../glossary.md#term-lookup). A
real Apple [Metal](../../glossary.md#term-metal) [lifecycle](../../glossary.md#lifecycle) test builds the [native library](../../glossary.md#term-native-library), runs both directions, publishes Metal-
and [CPU](../../glossary.md#term-cpu)-owned values together, poisons registry lookup after preparation, and still reuses and
materializes through public `Engine.builder()` composition. Any later [data type](../../glossary.md#data-type--datatype), [layout](../../glossary.md#layout), [device](../../glossary.md#term-device),
or backend pair requires an explicit capability extension and evidence; it must not weaken the
[fail-closed](../../glossary.md#term-fail-closed) default.

## Related documentation

- [Authoritative Runtime, Prepare, and Engine contract](../../architecture/contracts/runtime-prepare-engine.md#compile-ownership-and-mixed-owner-prepare-routing)<span data-glossary-navigation="true"> ([Runtime glossary](../../glossary.md#term-runtime); [Prepare glossary](../../glossary.md#prepare); [Engine glossary](../../glossary.md#term-engine))</span>
- [Authoritative backend execution contract](../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([backend glossary](../../glossary.md#backend))</span>
- [ADR 0015: Explicit Engine backend composition](0015-explicit-engine-backend-composition.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr); [backend composition glossary](../../glossary.md#backend-composition))</span>
- [Engine task 0016](../../planning/modules/engine/tasks/0016-cpu-metal-mixed-owner-schedule.md)
- [Runtime, Prepare, and backend boundary explanation](../../architecture/runtime-prepare-backend-boundary.md)

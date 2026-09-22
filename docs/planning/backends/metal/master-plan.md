# Metal Backend Master Plan

## Goal

Keep Metal capability, backend-owned MPSGraph/custom-kernel routes, native resources, storage,
preparation, and execution truthful across the cold Prepare and hot Runtime lifecycle.

## Authority and contracts

[`ARCHITECTURE.md`](../../../../ARCHITECTURE.md) is authoritative. The exact applicable headings
are [`modules/runtime`](../../../architecture/contracts/runtime-prepare-engine.md#modulesruntime),
[`modules/prepare`](../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare),
[Concrete backend modules](../../../architecture/contracts/backend-execution.md#concrete-backend-modules),
[Performance evidence and optimization tooling](../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling),
[Metal backend](../../../architecture/contracts/backend-execution.md#metal-backend), and
[Prepare lifecycle](../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle).

The non-authoritative [Metal strategy note](../../../design/notes/metal-backend-strategy.md) records
the residency/synchronization direction. [ADR 0002](../../../design/decisions/0002-backend-owned-lowering.md)
owns backend lowering, and [ADR 0013](../../../design/decisions/0013-prepared-execution-persistent-resource-lifecycle.md)
owns persistent prepared resources. The [module-boundary](../../../architecture/module-boundaries.md),
[dependency](../../../architecture/dependency-rules.md), and
[Runtime/Prepare/backend](../../../architecture/runtime-prepare-backend-boundary.md) documents
explain the shared boundary.

## Scope and non-goals

Metal owns truthful capability, MPSGraph/custom lowering and route choice, Metal storage and
workspace, native application binary interface (ABI), prepared resources, execution, trace
contribution, typed route candidates, and compatible workload-cache lookup during prepare.

It does not own public Tensor semantics, global autograd, Planning ownership policy, shared
Prepare/Runtime interpretation of private knobs, Engine composition, CPU Apple routes, or a
Training-to-Metal optimizer bridge.

## Stable invariants and dependencies

- Planning selects only `owner = Metal`; Metal analysis selects a backend-private route and
  declares every shared requirement before slot assignment. Runtime invokes already-prepared work.
- Capability is implemented semantic truth, not hardware/library availability. Target, operation,
  type, Shape, layout, numerical policy, and resource validity filter before heuristic, cache, or
  measurement comparison.
- Candidate generators return complete valid typed route configurations, are version-controlled,
  tested, and colocated with their routes. Shared orchestration treats them opaquely.
- Safe heuristics remain correct without tuning. Model 0026 must define IEEE FLOAT16 and affected
  numerical contracts before Metal advertises FLOAT16; two-byte storage does not imply BFLOAT16
  or FLOAT16 capability.
- Production dependencies may point to Model, Config, Planning, Runtime, Prepare,
  Backend Contract, and Trace, never Engine or Training. Task 0002's Compiler edge is test-only.

## Package map

```text
io.github.pho001.synaptik.backend.metal
  deliberate public capability surface
io.github.pho001.synaptik.backend.metal.internal
  native ABI, device/queue, storage/workspace, preparation, routes, binding, and resources
io.github.pho001.synaptik.backend.metal.internal.prepare
  deferred extraction only after multiple preparation families prove the seam
io.github.pho001.synaptik.backend.metal.internal.route.mpsgraph
  deferred broader MPSGraph extraction; never a custom-kernel or CPU home
```

The first two executable routes remain in the coupled internal package. A later task must update
this map before extracting a package or widening visibility.

## Task list

| ID | Task | Status | Depends on | One-line result or intent |
|---|---|---|---|---|
| 0001 | [Metal capability, storage, and native foundation](tasks/0001-metal-capability-storage-and-native-foundation.md) | Complete | Complete shared planning, runtime, prepare, backend-contract, and trace contracts; no Model 0026 dependency while fail-closed | Added a fail-closed provider, native context, and run-owned storage without executable capability. |
| 0002 | [MPSGraph prepared execution route](tasks/0002-mpsgraph-prepared-execution-route.md) | Complete | 0001; Runtime 0016; Prepare 0006; Engine 0010; Compiler 0006B7 | Added maximal-partition positive-shape contiguous FLOAT32 NEG through reusable MPSGraph preparation/execution. |
| 0003 | [Single-NEG custom Metal kernel route](tasks/0003-single-neg-custom-metal-kernel-route.md) | Complete | 0001–0002 | Added a private custom route for one NEG/feed/target within `1..UINT32_MAX`; all other supported partitions retain MPSGraph. |
| 0004 | Typed Metal route candidate generators and cache compatibility | Draft | 0002–0003, opaque prepare/tuning boundary and artifact versioning | Add typed complete route candidates and canonical workload compatibility without exposing Metal knobs. |

## Milestones and current frontier

The native/storage foundation, first MPSGraph route, and bounded second-route proof are `Complete`
through 0003. Task 0004 is the next ordered Metal row and current planning frontier. It remains
`Draft`, has no task brief, and is not implementation authorization. No Metal task is `Ready` or
`In progress`.

Before a compact 0004 brief can become `Ready`, a planner must verify all four live gates against
current source and contracts:

1. the Prepare/tuning handoff keeps complete Metal candidates and decisions opaque;
2. candidate/compatibility schemas and any workload-cache artifact have explicit versions and
   owners;
3. route selection, lowering, decoding, and resource declaration remain Metal-owned; and
4. the production, test, and artifact file set is bounded and isolated except for explicit
   route-integration points.

## Delivered lifecycle and ABI boundary

- Task 0001 is deliberately non-executing and fail closed. Task 0002 truthfully admits only the
  implemented fully static positive rank-`1..16`, dense-contiguous zero-offset FLOAT32 NEG domain.
  Task 0003 changes route choice, not occurrence capability or partitioning.
- The caller-supplied macOS arm64 native library uses an Objective-C C ABI reached through JDK 26
  Foreign Function and Memory (FFM). It is not packaged or discovered by the backend. ABI version
  1 established seven foundation functions and statuses `0..7`; version 2 retained them, added
  three typed MPSGraph functions and statuses `8..11`; version 3 retained all ten, added three
  custom-NEG functions and status `12`. Opaque resource kinds are never reinterpreted.
- Analysis validates the complete maximal Metal partition, selects the route, and declares exact
  buffers/workspaces. Finalization cannot change that route or add undeclared shared requirements;
  it creates route-specific persistent resources only after slot assignment.
- Finalizer, shared Prepare, and `PreparedExecution` transfer persistent resources transactionally.
  The prepared execution is the sole long-lived Runtime owner; repeated executable or schedule
  occurrences add no ownership. Physical cleanup stays backend-private and deterministic.
- Caller buffers are borrowed per run; splat/output buffers and binding workspaces are run-owned.
  Cold binding performs checked type/address work. The hot route uses direct typed handles and one
  synchronous native submission/downcall into assigned destinations, without route selection.
- `MTLBuffer` remains the resident representation across adjacent Metal work. A prepared
  shape-specialized `MPSGraphExecutable` is not per-run workspace; transfers and waits occur only
  at explicit boundaries required by the synchronous lifecycle.

## Task 0004 constraints, risks, and open decisions

- 0004 must derive complete typed candidates from canonical workload facts, target capability,
  exact policy, and budget; operation family chooses a generator but is not a universal cache key.
- Compatibility must include explicit schema/version and target/workload identity. A cache hit may
  select only a compatible decision and must still lead to fresh authenticated preparation.
- Do not introduce `Map<String,Object>`, reflection, string dispatch, a central knob registry,
  generic parameter bags, Planning route choice, Runtime cache access, or hidden global resources.
- Exact route-specific record shapes, target fingerprints, and version ownership remain the Draft
  0004 planning decision. No broader operation/type, mixed-owner Engine path, async execution,
  packaging/discovery, or performance claim is implied.
- Main risks are moving lowering into shared layers, leaking Metal fields through opaque seams,
  confusing capability with availability, and extending native/prepared lifetimes beyond their
  explicit owners.

## History and update policy

Detailed ABI inventories, native/test evidence, context identifiers, and route implementation
chronology remain in tasks 0001–0003 and Git history. The strategy note is planning guidance, not
a capability, platform, ABI, or performance promise.

Update this map only for task order/status/result, dependencies, package direction, the 0004 gate,
or a live lifecycle/ABI risk. Keep detailed evidence in task briefs. If a planning change conflicts
with `ARCHITECTURE.md` or an accepted ADR, stop and use the architecture-decision process.

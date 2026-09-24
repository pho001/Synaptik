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
contribution, typed route candidates, and the private compatibility needed before any outer
workload-cache decision can be accepted during prepare. Tuning tooling owns cache orchestration.

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
  public capability, explicit native-library configuration, and closeable Engine integration;
  package-private native ABI, device/queue, storage/workspace, preparation, routes, binding,
  and resources
io.github.pho001.synaptik.backend.metal.prepare
  deferred extraction only after multiple preparation families prove the seam
io.github.pho001.synaptik.backend.metal.route.mpsgraph
  deferred broader MPSGraph extraction; never a custom-kernel or CPU home
```

The first two executable routes remain coupled package-private types beside the three public
facade types. A later task must update this map before extracting a package or widening
visibility.

## Task list

| ID | Task | Status | Depends on | Conflicts with | Parallel group | Integration order | Integration validation | Intent/result |
|---|---|---|---|---|---|---|---|---|
| 0001 | [Metal capability, storage, and native foundation](tasks/0001-metal-capability-storage-and-native-foundation.md) | Complete | Complete shared planning, runtime, prepare, backend-contract, and trace contracts; no Model 0026 dependency while fail-closed | None | None | Any | Focused Metal foundation suite | Added a fail-closed provider, native context, and run-owned storage without executable capability. |
| 0002 | [MPSGraph prepared execution route](tasks/0002-mpsgraph-prepared-execution-route.md) | Complete | 0001; Runtime 0016; Prepare 0006; Engine 0010; Compiler 0006B7 | None | None | Any | Focused Metal and conformance suites | Added maximal-partition positive-shape contiguous FLOAT32 NEG through reusable MPSGraph preparation/execution. |
| 0003 | [Single-NEG custom Metal kernel route](tasks/0003-single-neg-custom-metal-kernel-route.md) | Complete | 0001–0002 | None | None | Any | Focused Metal/native suites | Added a private custom route for one NEG/feed/target within `1..UINT32_MAX`; all other supported partitions retain MPSGraph. |
| 0004 | [Typed Metal route candidate generators and cache compatibility](tasks/0004-typed-metal-route-candidate-generators-and-cache-compatibility.md) | Complete | 0002–0003, opaque prepare/tuning boundary and artifact versioning | None | None | Any | Focused candidate, Metal, and conformance suites | Added typed NEG candidates and a session-compatible authenticated codec foundation without outer tuning integration. |
| 0005 | [MPSGraph mixed NEG/binary FLOAT32 whole-partition route](tasks/0005-mpsgraph-mixed-binary-whole-partition.md) | Complete | 0001–0004; Engine 0017; Compiler 0006B7; Prepare 0008; Runtime 0016 | Metal capability/preparation/native ABI/Engine Metal scopes | None | ABI/schema → capability/analysis → Prepare → public Engine tests → docs/review | Native export audit; focused Metal/conformance; real public Engine integration; architecture checks; `git diff --check` | Approved after initial Class C BLOCK, remediation `9f3a264`, and independent APPROVE with zero residual findings; ABI v4 exact. |
| 0006 | [MPSGraph FLOAT32 unary algebra route](tasks/0006-mpsgraph-float32-unary-algebra.md) | Ready | 0005; current Model unary semantics; Engine 0017; Compiler 0006B7; Prepare 0008; Runtime 0016 | Metal capability/preparation/native ABI/Engine Metal scopes; scalar schema work | None | Native typed unary schema → capability/analysis → Prepare → fake/native tests → CPU-free Engine → docs/review | Native export/schema audit; focused Metal/conformance; real CPU-free Engine; architecture/docs checks; `git diff --check` | Bounded parameterless ABS/RECIPROCAL/EXP/LOG/SQRT/RSQRT/RELU/SIGMOID/TANH; scalar ADD/SUB/MUL/DIV explicitly deferred. |

## Dependency DAG and authorized frontiers

`0001 → 0002 → 0003 → 0004 → 0005 → 0006`

Authorized frontier: `0006` only, `Ready`. No other Metal task is Ready; later work remains
unauthorized until 0006 completes or is explicitly blocked/superseded.

## Integration ownership and shared documents

- Integration owner: Main planner
- Shared documents: this master plan and `docs/planning/roadmap.md`; planner owns synchronized
  status/frontier edits.

## Milestones and current frontier

Metal 0001–0005 are Complete. 0005's initial Class C `BLOCK` (two P1/two P2) was remediated by
`9f3a264` and independently re-reviewed `APPROVE` with zero residual findings. Evidence includes
the exact ABI-v4 thirteen-export audit, 5,000-run/native CPU-free proof, focused 55-test Metal
pass, 2/2 conformance pass, and 2/2 real CPU-free Engine integration pass. Metal 0006 is the sole
Ready frontier and preserves ABI v4 exactly; its scalar family is deferred pending a typed carrier.


## Delivered lifecycle and ABI boundary

- Task 0001 is deliberately non-executing and fail closed. Task 0002 truthfully admits only the
  implemented fully static positive rank-`1..16`, dense-contiguous zero-offset FLOAT32 NEG domain.
  Task 0003 changes route choice, not occurrence capability or partitioning.
- The caller-supplied macOS arm64 native library uses an Objective-C C ABI reached through JDK 26
  Foreign Function and Memory (FFM). It is not packaged or discovered by the backend. ABI version
  1 established seven foundation functions and statuses `0..7`; version 2 retained them, added
  three typed MPSGraph functions and statuses `8..11`; version 3 retained all ten, added three
  custom-NEG functions and status `12`; version 4 replaces only the NEG-specific MPSGraph create
  operation with the version-one typed whole-partition create schema. Opaque resource kinds are
  never reinterpreted. ABI v4 exports exactly:

  ```text
  synaptik_metal_foundation_abi_version
  synaptik_metal_context_create
  synaptik_metal_context_release
  synaptik_metal_buffer_create
  synaptik_metal_buffer_release
  synaptik_metal_buffer_upload
  synaptik_metal_buffer_download
  synaptik_metal_mpsgraph_executable_create
  synaptik_metal_mpsgraph_executable_release
  synaptik_metal_mpsgraph_executable_run
  synaptik_metal_neg_kernel_pipeline_create
  synaptik_metal_neg_kernel_pipeline_release
  synaptik_metal_neg_kernel_pipeline_run
  ```

  The old `synaptik_metal_mpsgraph_neg_executable_create` symbol is absent.
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
- Public `MetalBackendConfiguration` snapshots one explicit absolute native-library path.
  `MetalBackendIntegration.open(...)` validates ABI, opens one default-device context, supplies
  capability/availability/Prepare/ingress/materialization roles, and rolls partial opening back.
  Ownership transfers only through Engine's explicit builder; Metal performs no discovery.

## Task 0004 delivered constraints and remaining risks

- 0004 derives complete typed candidates from canonical workload facts, target capability,
  exact policy, and budget; operation family chooses a generator but is not a universal cache key.
- Compatibility includes explicit schema/version and target/workload identity. An optional
  encoded decision selects only a compatible candidate and still leads to fresh authenticated
  preparation; 0004 adds no outer cache hit or measurement path.
- The delivered boundary contains no `Map<String,Object>`, reflection, string dispatch, central
  knob registry, generic parameter bag, Planning route choice, Runtime cache access, or hidden
  global resource.
- The completed 0004 brief fixes route-specific typed shapes and conservatively session-scoped
  target compatibility without changing cache-file or native ABI ownership. Its package-private
  codec remains unconsumed by current `tools/tuning` and Engine tuning composition. Explicit
  single-owner Metal Engine execution is current without consuming that codec. No broader
  operation/type, mixed-owner Engine path, async execution, packaging/discovery, or performance
  claim is implied.
- Main risks are moving lowering into shared layers, leaking Metal fields through opaque seams,
  confusing capability with availability, and extending native/prepared lifetimes beyond their
  explicit owners.

## History and update policy

Detailed ABI inventories, native/test evidence, context identifiers, and route implementation
chronology remain in tasks 0001–0004 and Git history. The strategy note is planning guidance, not
a capability, platform, ABI, or performance promise.

Update this map only for task order/status/result, dependencies, package direction, a future Metal
gate, or a live lifecycle/ABI risk. Keep detailed evidence in task briefs. If a planning change
conflicts with `ARCHITECTURE.md` or an accepted ADR, stop and use the architecture-decision
process.

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
  or FLOAT16 capability. Blocked 0010 added no BOOL capability; blocked 0011 added no scalar
  arithmetic capability. Ready 0012 is FLOAT32-only ordinary extrema reduction. Later Metal 0028
  and 0029 remain reserved for separately authorized FLOAT16 and BFLOAT16 scopes.
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
| 0006 | [MPSGraph FLOAT32 unary algebra route](tasks/0006-mpsgraph-float32-unary-algebra.md) | Blocked | 0005; current Model unary semantics; Engine 0017; Compiler 0006B7; Prepare 0008; Runtime 0016 | Metal capability/preparation/native ABI/Engine Metal scopes; scalar schema work | None | Replacement route → numerical gates → Class C review | Reproducible real probe and fail-closed evidence; no production change | MPSGraph passed ABS ULP0, EXP ULP1, SIGMOID ULP1 but failed exact RECIPROCAL/LOG/SQRT/RSQRT/RELU/TANH cases; custom kernels or relaxed gates are required. |
| 0007 | [MPSGraph FLOAT32 SUM/MEAN reduction foundation](tasks/0007-mpsgraph-float32-reductions.md) | Blocked | 0005; current Model reduction attrs and Compiler autograd; Engine 0017; Compiler 0006B7; Prepare 0008; Runtime 0016 | Metal capability/preparation/native ABI/Engine Metal scopes; 0006 is not a dependency | None | Exact replacement route → numerical gates → Class C review | Reproducible real direct-output probe and fail-closed evidence; no production change | Eight runs returned positive zero for cancellation cases requiring SUM `2.0f` and MEAN `0.5f`; selector/schema/rank-zero feasibility does not satisfy exact Model semantics. |
| 0008 | [MPSGraph FLOAT32 affine transforms](tasks/0008-mpsgraph-float32-affine-transforms.md) | Complete | 0005; current Model/Compiler affine Shape and layout contracts; Engine 0017; Compiler 0006B7/0006B11; Prepare 0008; Runtime 0016; not 0006/0007 | Metal capability/preparation/native ABI/Engine Metal materialization; shared aliasing or transfer work | None | Real selector/bit probe → typed schema → capability/analysis → Prepare/Runtime/materialization → Engine proof → docs/review | ABI-v4 export/schema audit; exact Shape/layout/raw-bit gates; direct affine publication; transfer non-widening; focused Metal/conformance/Engine/architecture checks | Approved after implementation `7e39f705`, remediation `9713e528`/`aa42ed711`, and independent Class C APPROVE with zero findings; five bounded affine transforms, ABI v4 exact. |
| 0009 | [MPSGraph FLOAT32 rank-two MATMUL training checkpoint](tasks/0009-mpsgraph-float32-rank2-matmul-training-checkpoint.md) | Blocked | 0008; Model/Compiler rank-two MATMUL and first-order rules; Engine 0017; Compiler 0006B7/0006B11; Prepare 0008; Runtime 0016; not 0006/0007 | 0006/0007; Metal capability/preparation/native schema/candidate/materialization/Engine scopes | None | Exact replacement route → precision gates → Class C review | Reproducible real direct-target probe and fail-closed evidence; no production change | `K=1` multiplication by one flushed positive/negative minimum and ordinary subnormals to zero; no permitted FLOAT32 association/FMA yields zero. Independent `APPROVE-BLOCKER`; signed zero is not a blocker. |
| 0010 | [MPSGraph FLOAT32/BOOL predicates and selection](tasks/0010-mpsgraph-float32-bool-predicates-and-selection.md) | Blocked | 0008; current Model predicate/BOOL/WHERE semantics; Compiler forward capture; Engine 0017; Compiler 0006B7/0006B11; Prepare 0008; Runtime 0016; not 0006/0007/0009 or Model 0026 | 0006/0007/0009; Metal capability/preparation/native schema/candidate/storage/materialization/Engine scopes | None | Exact replacement route → numerical gates → Class C review | Reproducible real direct-target comparison probe and fail-closed evidence; no production change | Apple M3 Max MPSGraph comparisons treated positive and negative minimum subnormal as equal to `±0` at optimization levels 0/1; raw identity, ordinary/special controls, canonical BOOL targets, canaries, and permutations passed. |
| 0011 | [MPSGraph FLOAT32 scalar pointwise arithmetic](tasks/0011-mpsgraph-float32-scalar-pointwise-arithmetic.md) | Blocked | 0005/0008; current Model scalar semantics and Compiler forward capture; Engine 0017; Compiler 0006B7/0006B11; Prepare 0008; Runtime 0016; not 0006/0007/0009/0010 or Model 0026 | Every concurrent Metal capability/preparation/native schema/candidate/materialization/Engine scope | None | Exact replacement route → numerical gates → Class C review | Reproducible real scalar-bit/direct-target probe and independent blocker; no production change | The optimization-level 0/1 matrix found 17,328 mismatches in 77,824 checks, including signed-zero, subnormal, finite-division, and infinity-division failures; controls passed. |
| 0012 | [MPSGraph FLOAT32 extrema reductions](tasks/0012-mpsgraph-float32-extrema-reductions.md) | Ready | 0005/0008; current Model ordinary-extrema semantics and Compiler forward capture; Engine 0017; Compiler 0006B7/0006B11; Prepare 0008; Runtime 0016; not 0006/0007/0009/0010/0011 or Model 0026 | Every concurrent Metal capability/preparation/native schema/candidate/materialization/Engine scope | None | Probe → schema/geometry → capability/topology → candidates/lifecycle/scalar publication → Engine proof → docs/review | Exhaustive extrema/direct-target probe; ABI-v4 exact exports; focused Metal/conformance/Engine/Compiler/architecture/Javadoc/docs checks | Sole Ready frontier: bounded forward FLOAT32 ordinary MIN/MAX through installed NaN-propagating selectors, independent of exact SUM/MEAN. |

## Dependency DAG and authorized frontiers

`0001 → 0002 → 0003 → 0004 → 0005 → 0008 → {0009, 0010, 0011, 0012}`

0006 and 0007 are independent `Blocked` branches from 0005. 0009, 0010, and 0011 are
independently `Blocked` from 0008 by exact MATMUL subnormal flushing, comparison subnormal
collapse, and scalar arithmetic failures, respectively. 0012 is the sole authorized frontier and
depends only on completed 0005/0008 plus current Model/Compiler forward ordinary-extrema
contracts; it does not depend on 0006, 0007, 0009, 0010, 0011, Model 0026, or reserved future
Metal 0028/0029 type work.

## Integration ownership and shared documents

- Integration owner: Main planner
- Shared documents: this master plan and `docs/planning/roadmap.md`; planner owns synchronized
  status/frontier edits.

## Milestones and current frontier

Metal 0001–0005 and 0008 are Complete. 0008 landed bounded forward affine transforms in
`7e39f705`, received evidence remediation in `9713e528` and `aa42ed711`, and passed independent
Class C review with zero findings. Metal 0006 remains `Blocked` after exact
RECIPROCAL/LOG/SQRT/RSQRT/RELU/TANH probe failures. Metal 0007 remains independently `Blocked`
after eight direct-output executions returned positive zero where the exact Model SUM/MEAN results
are `2.0f`/`0.5f`.

Metal 0009 is independently `Blocked`: at optimization levels `0` and `1`, three executables per
level and eight runs per case, `K=1` MATMUL by one flushed positive/negative minimum and ordinary
subnormals to zero, although minimum normals survived and every permitted host association/FMA
oracle preserved the minimum subnormal. Selector, direct-target, Shape, permutation, and transpose
gates passed; signed zero is not a blocker. Independent review returned `APPROVE-BLOCKER`. Tasks
0006, 0007, and 0009 retain no production, test, or probe changes.

Metal 0010 is independently `Blocked`: on Apple M3 Max, optimization levels `0` and `1`, three
independent executables per level, and eight runs per case, positive and negative minimum
subnormal compared equal to `±0` instead of remaining distinct. Raw identity preserved the input
bits; ordinary, infinity, NaN, canonical BOOL target, canary, and permutation controls passed.
No production, test, or probe changes remain.

Metal 0011 is independently `Blocked`: at optimization levels `0` and `1`, three independent
executables per level and eight runs per case, its exact embedded-scalar matrix found 17,328
mismatches in 77,824 checks. Reproducible failures include scalar `ADD +0` retaining input `-0`,
minimum-subnormal flushing, minimum-subnormal `DIV +0` producing NaN, maximum-finite self-division
producing zero, and infinity/maximum-finite division producing NaN. Exact constants, direct
targets, canaries, permutations, repetition, and identity controls passed. Independent review
confirmed no Model relaxation admits the results. No production, test, native, or probe changes
remain.

Metal 0012 is the sole `Ready` frontier. It adds only forward FLOAT32 ordinary reduction `MIN`
and `MAX`, using the installed NaN-propagating extrema selectors. It is independent of blocked
exact SUM/MEAN and scalar arithmetic, begins with an exhaustive raw-bit real-device probe, and
fails closed on any NaN, signed-zero, subnormal-ordering, infinity, Shape, direct-target, canary,
permutation, repetition, or executable-independence failure. Product, SUM/MEAN/SUM_TO_SHAPE,
advanced/statistical/Boolean/arg reductions, custom kernels, backward/training, transfer widening,
FLOAT16, and BFLOAT16 are excluded.


## Delivered lifecycle and ABI boundary

- Task 0001 is deliberately non-executing and fail closed. Task 0002 truthfully admits only the
  implemented fully static positive rank-`1..16`, dense-contiguous zero-offset FLOAT32 NEG domain.
  Task 0003 changes route choice, not occurrence capability or partitioning.
- The caller-supplied macOS arm64 native library uses an Objective-C C ABI reached through JDK 26
  Foreign Function and Memory (FFM). It is not packaged or discovered by the backend. ABI version
  1 established seven foundation functions and statuses `0..7`; version 2 retained them, added
  three typed MPSGraph functions and statuses `8..11`; version 3 retained all ten, added three
  custom-NEG functions and status `12`; version 4 replaces only the NEG-specific MPSGraph create
  operation, whose pointed-to node table is now the version-two typed whole-partition schema.
  Opaque resource kinds are never reinterpreted. ABI v4 exports exactly:

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
- Task 0008 retains the exact thirteen-symbol ABI while schema version 2 adds bounded typed target-
  Shape, permutation, and axis attributes. Affine logical views receive distinct full-logical-size
  represented-order Metal targets; only an exact finalized-route-authenticated target can use the
  affine materialization path. Canonical-only cross-owner transfer remains unchanged.
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

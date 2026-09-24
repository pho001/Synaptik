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
- Safe heuristics remain correct without tuning. Model 0026 independently owns future IEEE
  FLOAT16 semantics; two-byte storage implies neither BFLOAT16 nor FLOAT16 capability.
- Task 0005's strict `ADD`/`SUB`/`MUL`/`DIV` delivery remains historically withdrawn after its
  subnormal audit. Complete 0015 restores only tensor `FLOAT32` binary arithmetic under
  `ACCELERATOR`, with bounded DAZ/FTZ and no strict binary capability.
- Blocked 0010 added no BOOL capability; blocked 0011 added no scalar arithmetic capability;
  blocked 0012 added no extrema reduction capability; blocked 0013 added no cumulative-scan
  capability. Complete 0014 composes exact strict affine layouts plus `CONTIGUOUS`. Complete 0015
  gives accelerator graphs the four canonical tensor-binary operations. Ready 0019 now has a
  worker implementation adding exact canonical `ABS` to both profile matrices; independent Class C
  review is still pending.
- Historical 0006, 0007, and 0009 remain Blocked records. Profile-qualified 0016 is also Blocked:
  its new broad gate proved only exact `ABS`, while `EXP`/`SIGMOID` failed unchanged no-FTZ and
  one-ULP requirements. Task 0019 remains Ready pending independent review; 0017 remains Draft for
  accelerator `SUM`/`MEAN`/`SUM_TO_SHAPE`, and 0018 remains Draft for accelerator rank-two
  `MATMUL`. Those tasks share Metal capability/schema/lifecycle scope and must not overlap.
- Later Metal 0028 and 0029 remain reserved for separately authorized FLOAT16 and BFLOAT16 scopes.
- Model 0027, Config 0006, Engine 0018, CPU 0017, and Metal 0015 are Complete. Model 0026 remains
  an independent FLOAT16 Draft.
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
| 0005 | [MPSGraph mixed NEG/binary FLOAT32 whole-partition route](tasks/0005-mpsgraph-mixed-binary-whole-partition.md) | Complete | 0001–0004; Engine 0017; Compiler 0006B7; Prepare 0008; Runtime 0016 | Metal capability/preparation/native ABI/Engine Metal scopes | None | ABI/schema → capability/analysis → Prepare → public Engine tests → docs/review | Native export audit; focused Metal/conformance; real public Engine integration; architecture checks; `git diff --check` | Originally delivered and approved after remediation `9f3a264`; an independent post-completion subnormal audit later required withdrawing all four binary operations. ABI v4 remains exact; schema 3 rejects wires `2..5`. |
| 0006 | [MPSGraph FLOAT32 unary algebra route](tasks/0006-mpsgraph-float32-unary-algebra.md) | Blocked | 0005; current Model unary semantics; Engine 0017; Compiler 0006B7; Prepare 0008; Runtime 0016 | Metal capability/preparation/native ABI/Engine Metal scopes; scalar schema work | None | Replacement route → numerical gates → Class C review | Reproducible real probe and fail-closed evidence; no production change | MPSGraph passed ABS ULP0, EXP ULP1, SIGMOID ULP1 but failed exact RECIPROCAL/LOG/SQRT/RSQRT/RELU/TANH cases; custom kernels or relaxed gates are required. |
| 0007 | [MPSGraph FLOAT32 SUM/MEAN reduction foundation](tasks/0007-mpsgraph-float32-reductions.md) | Blocked | 0005; current Model reduction attrs and Compiler autograd; Engine 0017; Compiler 0006B7; Prepare 0008; Runtime 0016 | Metal capability/preparation/native ABI/Engine Metal scopes; 0006 is not a dependency | None | Exact replacement route → numerical gates → Class C review | Reproducible real direct-output probe and fail-closed evidence; no production change | Eight runs returned positive zero for cancellation cases requiring SUM `2.0f` and MEAN `0.5f`; selector/schema/rank-zero feasibility does not satisfy exact Model semantics. |
| 0008 | [MPSGraph FLOAT32 affine transforms](tasks/0008-mpsgraph-float32-affine-transforms.md) | Complete | 0005; current Model/Compiler affine Shape and layout contracts; Engine 0017; Compiler 0006B7/0006B11; Prepare 0008; Runtime 0016; not 0006/0007 | Metal capability/preparation/native ABI/Engine Metal materialization; shared aliasing or transfer work | None | Real selector/bit probe → typed schema → capability/analysis → Prepare/Runtime/materialization → Engine proof → docs/review | ABI-v4 export/schema audit; exact Shape/layout/raw-bit gates; direct affine publication; transfer non-widening; focused Metal/conformance/Engine/architecture checks | Approved after implementation `7e39f705`, remediation `9713e528`/`aa42ed711`, and independent Class C APPROVE with zero findings; five bounded affine transforms, ABI v4 exact. |
| 0009 | [MPSGraph FLOAT32 rank-two MATMUL training checkpoint](tasks/0009-mpsgraph-float32-rank2-matmul-training-checkpoint.md) | Blocked | 0008; Model/Compiler rank-two MATMUL and first-order rules; Engine 0017; Compiler 0006B7/0006B11; Prepare 0008; Runtime 0016; not 0006/0007 | 0006/0007; Metal capability/preparation/native schema/candidate/materialization/Engine scopes | None | Exact replacement route → precision gates → Class C review | Reproducible real direct-target probe and fail-closed evidence; no production change | `K=1` multiplication by one flushed positive/negative minimum and ordinary subnormals to zero; no permitted FLOAT32 association/FMA yields zero. Independent `APPROVE-BLOCKER`; signed zero is not a blocker. |
| 0010 | [MPSGraph FLOAT32/BOOL predicates and selection](tasks/0010-mpsgraph-float32-bool-predicates-and-selection.md) | Blocked | 0008; current Model predicate/BOOL/WHERE semantics; Compiler forward capture; Engine 0017; Compiler 0006B7/0006B11; Prepare 0008; Runtime 0016; not 0006/0007/0009 or Model 0026 | 0006/0007/0009; Metal capability/preparation/native schema/candidate/storage/materialization/Engine scopes | None | Exact replacement route → numerical gates → Class C review | Reproducible real direct-target comparison probe and fail-closed evidence; no production change | Apple M3 Max MPSGraph comparisons treated positive and negative minimum subnormal as equal to `±0` at optimization levels 0/1; raw identity, ordinary/special controls, canonical BOOL targets, canaries, and permutations passed. |
| 0011 | [MPSGraph FLOAT32 scalar pointwise arithmetic](tasks/0011-mpsgraph-float32-scalar-pointwise-arithmetic.md) | Blocked | 0005/0008; current Model scalar semantics and Compiler forward capture; Engine 0017; Compiler 0006B7/0006B11; Prepare 0008; Runtime 0016; not 0006/0007/0009/0010 or Model 0026 | Every concurrent Metal capability/preparation/native schema/candidate/materialization/Engine scope | None | Exact replacement route → numerical gates → Class C review | Reproducible real scalar-bit/direct-target probe and independent blocker; no production change | The optimization-level 0/1 matrix found 17,328 mismatches in 77,824 checks, including signed-zero, subnormal, finite-division, and infinity-division failures; controls passed. |
| 0012 | [MPSGraph FLOAT32 extrema reductions](tasks/0012-mpsgraph-float32-extrema-reductions.md) | Blocked | 0005/0008; current Model ordinary-extrema semantics and Compiler forward capture; Engine 0017; Compiler 0006B7/0006B11; Prepare 0008; Runtime 0016; not 0006/0007/0009/0010/0011 or Model 0026 | Every concurrent Metal capability/preparation/native schema/candidate/materialization/Engine scope | None | Probe → schema/geometry → capability/topology → candidates/lifecycle/scalar publication → Engine proof → docs/review | Reproducible exhaustive direct-target probe and independent blocker; no production change | Propagating MIN/MAX selectors were order-dependent for opposite signed zeros and zero/minimum-subnormal pairs: 2,016 MIN plus 2,016 MAX mismatches and 2,976 subnormal failures in 43,824 executions. |
| 0013 | [MPSGraph FLOAT32 cumulative scans](tasks/0013-mpsgraph-float32-cumulative-scans.md) | Blocked | 0005/0008; current Model cumulative-scan semantics and Compiler forward capture; Engine 0017; Compiler 0006B7/0006B11; Prepare 0008; Runtime 0016; not 0006/0007/0009–0012 or Model 0026 | Every concurrent Metal capability/preparation/native schema/candidate/materialization/Engine scope | None | Exact replacement route -> numerical gates -> Class C review | Reproducible exact cumulative-selector/direct-target probe and independent blocker; no production change | Inclusive length-one SUM/PRODUCT flushed all six subnormal corpus values at optimization levels 0/1; strict host controls preserved them. Axis 5 of rank 6 passed and axis 6 correctly rejected. Independent `APPROVE-BLOCKER`. |
| 0014 | [MPSGraph FLOAT32 affine layout composition](tasks/0014-mpsgraph-float32-affine-layout-composition.md) | Complete | 0005/0008; current Model layout semantics and Compiler inference; Engine 0017; Compiler 0006B7/0006B11; Prepare 0008; Runtime 0016; not 0006–0007/0009–0013 or Model 0026 | Every concurrent Metal capability/preparation/native schema/candidate/materialization/Engine Metal scope | None | Probe -> capability/topology -> schema/state validation -> candidates/lifecycle/authentication -> Engine proof -> docs/review | Raw-bit selector-chain/direct-target probe; ABI-v4 exact exports; focused Metal/conformance/Compiler/Engine/architecture/Javadoc/docs checks | Commit `01e81be2` delivered exact local affine/`CONTIGUOUS` composition, schema/candidate version 4, and real Engine proof; independent Class C review returned `APPROVE` with zero findings. |
| 0015 | [ACCELERATOR FLOAT32 tensor binary arithmetic](tasks/0015-accelerator-float32-tensor-binary-arithmetic.md) | Complete | 0014; Model 0027; Config 0006; Engine 0018 | Every Metal capability/preparation/native schema/candidate/materialization scope | `numerical-profile-backends` (complete) | Integrated after CPU 0017; shared documents last | Native/export, focused Metal/conformance/public Engine, architecture, full-build, and documentation checkpoint | Delivered accelerator-only tensor `ADD`/`SUB`/`MUL`/`DIV` under the bounded DAZ/FTZ oracle; strict retains NEG/affine/`CONTIGUOUS`; schema 5 and version-six identities fail closed. |
| 0016 | [Profile-qualified FLOAT32 ABS, EXP, and SIGMOID](tasks/0016-profile-qualified-float32-abs-exp-sigmoid.md) | Blocked | 0015; Model 0027; Config 0006; Engine 0018 | 0017–0019 and every Metal capability/preparation/native schema/candidate/materialization/Engine Metal scope | None | Exact replacement gate only | Reproducible real-device oracle; no production change | Exact `ABS` passed, but `EXP`/`SIGMOID` flushed representable subnormal results and `SIGMOID` also exceeded its one-ULP gate; all controls passed and the probe was removed. |
| 0017 | ACCELERATOR FLOAT32 SUM/MEAN reductions | Draft | 0019; Model 0027 | 0018 and any 0016 restart; every Metal capability/preparation/native schema/candidate/materialization/Engine Metal scope | None | Serial after 0019, before 0018 | Future all-term binary-tree oracle and focused Class C checkpoint | Add accelerator-only `SUM`, `MEAN`, and `SUM_TO_SHAPE` for the former 0007 full, one-axis, ordered multi-axis, and binding-aware forms; use all and only declared terms in a Model-permitted FLOAT32 binary tree with per-step DAZ/FTZ; divide `MEAN` by the declared positive count; never drop, duplicate, or invent a term or widen rank-zero transfer. |
| 0018 | ACCELERATOR FLOAT32 rank-two MATMUL seeded-gradient checkpoint | Draft | 0017; Model 0027 | 0017 and any 0016 restart; every Metal capability/preparation/native schema/candidate/materialization/Engine Metal scope | None | Serial after 0017 | Future contraction-set oracle and focused Class C checkpoint | Add accelerator-only positive rank-two `MATMUL`, local rank-two transpose composition, and explicitly seeded two-operand gradient evidence; permit only Model DAZ/FTZ plus existing FMA/reassociation, never reduced precision or term loss. |
| 0019 | [Exact profile-qualified FLOAT32 ABS](tasks/0019-exact-profile-qualified-float32-abs.md) | Ready | 0015; Model 0027; Config 0006; Engine 0018; not 0016 | Any 0016 restart; 0017–0018; every Metal capability/preparation/native schema/candidate/materialization/Engine Metal scope | None | Serial after 0015, before 0017–0018 | Exact both-profile Metal ABS checkpoint | Worker implementation and fresh schema-six/version-seven, real-dylib, conformance, and CPU-free Engine proof are complete; exact canonical `ABS` is present under both profiles, with independent Class C review still pending. |

## Dependency DAG and authorized frontiers

Historical implementation branch:

`0001 -> 0002 -> 0003 -> 0004 -> 0005 -> 0008 -> {0009, 0010, 0011, 0012, 0013, 0014}`

Completed profile spine and serial Metal successors:

`Model 0027 -> Config 0006 -> Engine 0018`

`0014 + Model 0027 + Config 0006 + Engine 0018 -> 0015`

`0015 -> {0016 (Blocked), 0019 (Ready) -> 0017 (Draft) -> 0018 (Draft)}`

Historical 0006–0007 and 0009–0013 keep their recorded `Blocked` status and evidence. Blocked
[0016](tasks/0016-profile-qualified-float32-abs-exp-sigmoid.md) also keeps its failed three-operation
contract and is not a dependency of any successor. [0019](tasks/0019-exact-profile-qualified-float32-abs.md)
is the sole Ready Metal frontier. Tasks 0017 and 0018 remain Draft because they and 0019 mutate the
same Metal capability, schema, candidate/codec, native, lifecycle, and Engine integration scopes;
their serial dependencies are deliberate conflict edges, not semantic claims that ABS is required
for reduction or reduction for MATMUL.

## Integration ownership and shared documents

- Integration owner: each serial task implementer, with mandatory independent Class C review.
- Shared documents: the active task owns synchronized Metal planning, architecture-status, API/user
  status, capability/preparer guides, and glossary updates after executable behavior stabilizes.
- No Metal 0016–0019 write scopes may overlap. Blocked 0016 has no active write scope. Promote 0017
  only after 0019 is Complete and its exact brief is authored; promote 0018 only after 0017 is
  Complete and its exact brief is authored.

## Milestones and current frontier

Metal 0001–0005, 0008, and 0014–0015 are Complete. Task 0015 landed at `42c4cfbf` plus
evidence-wording remediation `fb102a46`; its real-device oracle, exact thirteen-export audit,
focused Metal/conformance/public Engine checks, architecture and full-build checkpoint, and
independent Class C `APPROVE` with zero findings all passed. Ready task 0019 has completed its
worker implementation and awaits independent review. Current ABI v4 retains exactly thirteen
exports, points to node schema 6, and uses version-seven workload/candidate/compatibility/route/codec
identities.

Metal 0006 remains `Blocked` after exact RECIPROCAL/LOG/SQRT/RSQRT/RELU/TANH probe failures.
Metal 0007 remains independently `Blocked` after eight direct-output executions returned positive
zero where the current exact Model SUM/MEAN results are `2.0f`/`0.5f`.

Task 0005's original strict binary delivery remains historically withdrawn by its 12,096
subnormal-domain mismatches. Complete 0015 does not reopen that strict claim: it restores only
tensor `FLOAT32` `ADD`, `SUB`, `MUL`, and `DIV` under `ACCELERATOR` after a fresh bounded DAZ/FTZ
oracle passed every operation and control gate. Strict Metal still rejects binary arithmetic; an
explicit CPU may own it.

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

Metal 0012 is independently `Blocked`: at optimization levels `0` and `1`, reduced-precision fast
math disabled, three executables per length/level, and eight runs per case, the propagating MIN/MAX
selectors were order-dependent for opposite signed zeros and zero/minimum-subnormal pairs. The
43,824 executions recorded 2,016 MIN plus 2,016 MAX mismatches and 2,976 subnormal failures.
NaN classification, minimum normals, identity, canaries, and bindings passed. Independent review
returned `APPROVE-BLOCKER`; no production, native, test, or probe changes remain.

Metal 0013 is independently `Blocked`: at optimization levels `0` and `1`, reduced-precision fast
math disabled, twelve executables covering three per level/length pair for lengths `1` and `2`,
and eight direct runs for all eight kind/mode combinations, inclusive length-one SUM and PRODUCT
flushed all six subnormal corpus values to zero. Strict host `+0 + minimum-subnormal` and
`+1 * minimum-subnormal` controls preserved them; exclusive identities, raw identity, canaries,
and bindings passed. Corrected review evidence proves rank-six axis `5` valid for both kinds and
axis `6` correctly rejected, so axis is not a blocker. Independent review returned
`APPROVE-BLOCKER`; no production, native, test, or probe changes remain.

Metal 0015 is Complete. [0016](tasks/0016-profile-qualified-float32-abs-exp-sigmoid.md) is Blocked
without production changes: its 324-executable/2,592-run Apple M3 Max gate proved exact `ABS`, but
`EXP` and `SIGMOID` reproducibly flushed representable subnormal results to positive zero across
levels, Shapes, contexts, and forms; `SIGMOID` also produced two-ULP ordinary finite results. All
controls passed and the disposable probe was removed. The unchanged broad contract is not weakened.

[0019](tasks/0019-exact-profile-qualified-float32-abs.md) remains the sole Ready Metal frontier
while its independent Class C review is pending. The worker implementation adds only exact
canonical `ABS` under both profiles on node schema 6 and version-seven identities. `EXP`,
`SIGMOID`, `RECIPROCAL`, `LOG`, `SQRT`, `RSQRT`, `RELU`, and `TANH` remain closed. Tasks 0017 and
0018 are serial Draft successors for accelerator reduction and rank-two MATMUL. Comparison,
scalar, extrema, scan, every other unary operation, backward/training beyond the later bounded
seeded-gradient scope, and Model 0026 remain unauthorized.

## Delivered lifecycle and ABI boundary

- Task 0001 is deliberately non-executing and fail closed. Task 0002 established the strict
  positive-rank canonical FLOAT32 NEG domain, task 0014 added strict affine/`CONTIGUOUS`
  composition, task 0015 added the accelerator tensor-binary domain, and task 0019's worker
  implementation added exact canonical ABS to both profile matrices. Task 0003 changes
  singleton-NEG route choice, not occurrence capability or partitioning.
- The caller-supplied macOS arm64 native library uses an Objective-C C ABI reached through JDK 26
  Foreign Function and Memory (FFM). It is not packaged or discovered by the backend. ABI version
  1 established seven foundation functions and statuses `0..7`; version 2 retained them, added
  three typed MPSGraph functions and statuses `8..11`; version 3 retained all ten, added three
  custom-NEG functions and status `12`; version 4 replaced only the NEG-specific MPSGraph create
  operation with the typed whole-partition create operation. Its pointed-to node table currently
  uses schema version 6. Opaque resource kinds are never reinterpreted. ABI v4 exports exactly:

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
- Tasks 0008, 0014, 0015, and 0019 retain the exact thirteen-symbol ABI. Schema version 6 admits
  strict NEG/affine/`CONTIGUOUS` on wires `1` and `6..11`, exact ABS in both profiles on wire `12`,
  and accelerator tensor binary on wires `2..5`. Java profile preflight rejects every
  profile-incompatible node set before downcall. Explicit unavailable/canonical/affine-view states
  enforce graph-local view provenance; affine outputs use authenticated full-logical-size
  represented-order targets. Canonical-only cross-owner transfer remains unchanged.
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

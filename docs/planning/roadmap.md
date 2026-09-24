# Implementation Roadmap

## Purpose and authority

This roadmap is the active index for project priority, area status, blockers, and authorized
implementation frontiers. It is not an architecture contract.
[`ARCHITECTURE.md`](../../ARCHITECTURE.md) is authoritative and wins if a planning document
conflicts with it.

Read an area's linked master plan for its dependency DAG and the authorized task briefs for
executable scope. Completed-task narratives and validation evidence do not belong in this index.

## Execution policy

- Each independent workstream may expose one authorized frontier when its DAG dependencies are
  satisfied.
- Before launch, the planner verifies that the task is `Ready`, is at an authorized frontier, and
  has current dependency, conflict, parallel-group, base, integration-order, and
  integration-validation metadata.
- Task-table and project-area order express priority, not an undeclared dependency. Concurrent
  writes are authorized only under the worktree, ownership, contract-first, and integration rules
  in the [Planning Guide](planning-guide.md).
- Record active frontiers and parallel groups in the owning master plans; summarize them here only
  when they affect cross-area priority, dependencies, or integration.
- Allowed and forbidden dependencies remain defined only by `ARCHITECTURE.md`.

## Ordered project areas

The order is the default delivery priority. “Next gate” identifies an unfinished decision or task
boundary; it does not promote Draft work to `Ready` or create a DAG edge.

| Order | Project area | Current status | Entry or next gate |
|---:|---|---|---|
| 1 | [`modules/model`](modules/model/master-plan.md) | Complete through reviewed 0029; 0026 Draft | Model 0029 completed at `30826783` after proof, validation, synchronized documentation, and independent Class C approval; no Model task is Ready. |
| 2 | [`modules/trace`](modules/trace/master-plan.md) | In progress, deliberately interleaved; 0001–0002 Complete, 0003–0008 Draft | Resume 0003 only after its producer vocabulary is stable; no Trace task is Ready. |
| 3 | [`modules/backend-contract`](modules/backend-contract/master-plan.md) | Complete through 0004 | Reopen only for a concrete shared-contract need. |
| 4 | [`modules/config`](modules/config/master-plan.md) | In progress, interleaved; 0001–0003, 0006, and 0006A–0006B Complete; 0004–0005 and 0007–0008 Draft | 0006 completed at `314e049` plus `37e9e9db`; no Config task is Ready. |
| 5 | [`modules/planning`](modules/planning/master-plan.md) | Complete through documentation-only 0007; profile-qualified query cutover Complete through Engine 0018 | No Planning task is Ready. |
| 6 | [`modules/runtime`](modules/runtime/master-plan.md) | Complete through 0016; profile-free boundary preserved by Engine 0018 | No Runtime task is Ready. |
| 7 | [`modules/compiler`](modules/compiler/master-plan.md) | Complete through documentation-only 0006B10; 0006C and 0007 Draft; profile artifact cutover Complete through Engine 0018 | No Compiler task is Ready. |
| 8 | [`modules/prepare`](modules/prepare/master-plan.md) | Complete through documentation-only 0008; profile projection cutover Complete through Engine 0018 | No Prepare task is Ready. |
| 9 | [`backends/openblas-provider`](backends/openblas-provider/master-plan.md) | Required baseline Complete; optional 0004 Blocked and deferred | Resume 0004 only when both direct-BFLOAT16 application binary interface (ABI) and one-final-narrowing proofs exist. |
| 10 | [`backends/cpu`](backends/cpu/master-plan.md) | Complete through profile realization 0017; 0007A1D Review needed; 0010D1 and 0011 Blocked | CPU 0017 completed at `372a8b98`; both profiles use identical exact CPU capability, routes, execution, and profile-separated identities. No CPU task is Ready. |
| 11 | [`modules/engine`](modules/engine/master-plan.md) | Complete through numerical-profile spine 0018 | 0018 completed at `ce7a7dfa` plus `07a01b9c`; no Engine task is Ready. |
| 12 | [`backends/metal`](backends/metal/master-plan.md) | Complete through 0020; 0006–0007, 0009–0013, and 0016–0018 Blocked; 0021 Ready | [Metal 0018](backends/metal/tasks/0018-accelerator-float32-rank2-matmul-seeded-gradient-checkpoint.md) remains Blocked under its unchanged historical contract. Reverified [Metal 0021](backends/metal/tasks/0021-accelerator-float32-rank2-matmul-after-zero-sign-refinement.md) is the sole Ready frontier and must start a completely fresh full real-M3 oracle from clean base `30826783`. |

| 13 | [`backends/cuda`](backends/cuda/master-plan.md) | Draft | Create a detailed 0001 brief only when CUDA becomes the authorized frontier. |
| 14 | [`extensions/onnx`](extensions/onnx/master-plan.md) | Draft | Define the first bounded mapping task only at an authorized frontier. |
| 15 | [`extensions/data`](extensions/data/master-plan.md) | Draft; architecture decision required | 0001 must authorize the Data/Text/Vision modules, build edges, decision record, and architecture tests first. |
| 16 | [`extensions/nn`](extensions/nn/master-plan.md) | In progress under recorded interleaves; 0001–0020C, 0021A, and 0025–0025A Complete; 0021B–0024 Draft | 0021B needs a detailed brief and truthful concrete-backend recurrent coverage; no NN task is Ready. |
| 17 | [`extensions/text`](extensions/text/master-plan.md) | Draft; architecture decision required | Wait for Data 0001, then define the first tokenizer task. |
| 18 | [`extensions/vision`](extensions/vision/master-plan.md) | Draft; architecture decision required | Join the coordinated Data 0001 decision before decoder or image APIs. |
| 19 | [`extensions/training`](extensions/training/master-plan.md) | Complete through 0001 | No later Training task is Ready; parameter groups and broader optimizers remain Draft. |
| 20 | [`extensions/checkpoint`](extensions/checkpoint/master-plan.md) | Draft; architecture decision required | Authorize the model-only and optional Training adapter boundaries before 0001. |
| 21 | [`tools/benchmarks`](tools/benchmarks/master-plan.md) | Complete through 0001; 0002–0003 Draft | Report-only smoke, baseline, and evidence profiles are current; no benchmark task is Ready. |
| 22 | [`tools/tuning`](tools/tuning/master-plan.md) | Complete through 0004 | No Tuning task is Ready. |
| 23 | [`tools/cli`](tools/cli/master-plan.md) | Draft | Define commands only after their Engine and diagnostic contracts are stable. |

## Authorized frontiers

Numerical profiles

[CPU 0017](backends/cpu/tasks/0017-explicit-accelerator-numerical-profile-realization.md) is
Complete at implementation `372a8b98`. CPU admits every current supported occurrence under both
profiles through one identical capability matrix and unchanged portable/OpenBLAS routes. Requested
profiles remain distinct in plans, generated artifacts, OpenBLAS workloads, local tuning,
complete-plan tuning, and cache compatibility; CPU adds no DAZ, FTZ, reassociation, approximate
instruction, reduced precision, route change, or Runtime/Trace state.

[Metal 0015](backends/metal/tasks/0015-accelerator-float32-tensor-binary-arithmetic.md) is Complete
at implementation `42c4cfbf` plus evidence-wording remediation `fb102a46`. Strict Metal retains
exact NEG, five affine transforms, and `CONTIGUOUS`; accelerator Metal retains canonical tensor
`FLOAT32` `ADD`, `SUB`, `MUL`, and `DIV` under the bounded Model DAZ/FTZ result set. Complete
[Metal 0019](backends/metal/tasks/0019-exact-profile-qualified-float32-abs.md) adds exact canonical
`ABS` to both profile matrices. Complete Metal 0020 adds accelerator reductions, and Ready Metal
0021 now implements positive static rank-two MATMUL with authenticated local transposes and seeded
gradients while awaiting independent review. Strict binary/reduction/MATMUL, accelerator
NEG/layout operations outside the local MATMUL transpose rule, every other unary operation, and
every unproved family remain fail-closed.

Task 0019 landed at implementation `a6d1796d` plus mixed-owner test remediation `bcb717a6`. Its
native ABI/export, Metal, conformance, real Engine, architecture, full-build, documentation, and
diff checks passed; independent Class C final review returned `APPROVE` with zero findings.

The completed cross-area profile DAG is:

`Model 0027 -> Config 0006 -> Engine 0018 -> {CPU 0017, Metal 0015}`

The active semantic and Metal serial DAG is:

`Metal 0015 -> {Metal 0016 (Blocked), Metal 0019 (Complete) -> Metal 0017 (Blocked)}`

`Model 0028 (Complete) -> Metal 0020 (Complete) -> Metal 0018 (Blocked)`

`Metal 0018 blocker evidence -> Model 0029 (Complete) -> Metal 0021 (Ready: review pending)`

[Metal 0016](backends/metal/tasks/0016-profile-qualified-float32-abs-exp-sigmoid.md) is Blocked
without production changes. Its Apple M3 Max gate proved exact `ABS`, but `EXP` and `SIGMOID`
reproducibly flushed representable subnormal results to positive zero across every level, Shape,
context, and form; `SIGMOID` also exceeded its one-ULP gate on ordinary finite inputs. All controls
passed and the probe was removed. Historical 0006 remains Blocked under its own contract.

[Metal 0019](backends/metal/tasks/0019-exact-profile-qualified-float32-abs.md) is Complete. It adds
only canonical exact `ABS` under both profiles, with no relaxed unary semantics. `EXP`, `SIGMOID`,
and the six historically failing 0006 operations remain closed, including the forbidden
`RELU(NaN) -> +0` and `TANH(NaN) -> +1` observations.

[Metal 0017](backends/metal/tasks/0017-accelerator-float32-sum-mean-sum-to-shape-reductions.md)
remains Blocked under its unchanged old contract. After probe-only dangling autorelease-string and
per-cell pool fixes, a minimal valid rank-16 axis-8 cell passed and the corrected full gate completed
300 executables/2,400 runs. Accelerator `SUM([-0,-0])` returned positive zero, but the old
binary-tree/per-step-FLOAT32 plus DAZ/FTZ set permitted only negative zero because zero is not
subnormal. No production change or probe artifact remains.

[Model 0028](modules/model/tasks/0028-accelerator-reduction-exact-zero-sign-freedom.md) is
Complete at `fc003ab8`. Metal 0020 completed at implementation `9ddb75f6` plus documentation
remediation `5b77c742` after its full fresh corrected gate, validation, and independent Class C
`APPROVE` with zero findings.

[Metal 0018](backends/metal/tasks/0018-accelerator-float32-rank2-matmul-seeded-gradient-checkpoint.md)
is Blocked under its unchanged historical contract. Its fresh real-M3 smoke passed, then the first
full direct context-zero/optimization-zero `K=1` cell returned positive zero for
`+0.0f * -1.0f`, outside that task's set containing only negative zero. Reduced-precision-none,
selector, direct-target, binding, and probe controls were valid; the probe was removed and no
production change exists.
[Model 0029](modules/model/tasks/0029-accelerator-matmul-exact-zero-sign-freedom.md) is Complete at
`30826783` after its proof, 23-task validation, documentation/diff scope check, and independent
Class C `APPROVE` with zero findings. Metal 0021 passed its fresh full oracle, implemented the
scoped route, and remains Ready pending independent Class C review.

Engine
[0017](modules/engine/tasks/0017-reusable-inference-session-api.md) is Complete from exact base
`4fc4fd3d1e44613cdbdc44051e0bc637992d0de4`; 0018 is Complete at `ce7a7dfa` plus `07a01b9c`.
No Engine task is Ready.

Training
[0001](extensions/training/tasks/0001-public-training-session-and-sgd-lifecycle.md) is Complete
from exact base `2e110c7688ead480faa4215ff67f9814fa81f4ad`. Mandatory external Class C review
first returned `BLOCK` with six P1 and one P2 findings; after remediation and validation, external
re-review returned `APPROVE` with findings `0`. No later Training task is authorized.
Metal
[0008](backends/metal/tasks/0008-mpsgraph-float32-affine-transforms.md) is `Complete` after
implementation `7e39f705`, remediation `9713e528` and `aa42ed711`, and independent Class C
`APPROVE` with zero findings. It delivers the five bounded forward affine transforms with exact
logical views, authenticated dense represented-order targets, ABI v4's exact thirteen exports,
and unchanged canonical-only transfer.

[0005](backends/metal/tasks/0005-mpsgraph-mixed-binary-whole-partition.md) remains a completed
historical delivery whose strict binary claim was withdrawn after 12,096 subnormal-domain failures.
Complete 0015 does not rewrite that history: it restored wires `2..5` only for `ACCELERATOR` after
the fresh bounded oracle passed. Complete 0019 appends exact ABS wire `12` under both profiles.
Task 0020 adds accelerator `SUM=13` and `MEAN=14`; task 0021 appends accelerator `MATMUL=15`.
ABI v4 still exposes exactly thirteen symbols, while node schema 8 and version-nine identities
reject profile-incompatible programs.

[0006](backends/metal/tasks/0006-mpsgraph-float32-unary-algebra.md) remains `Blocked` by exact
RECIPROCAL/LOG/SQRT/RSQRT/RELU/TANH probe failures. [0007](backends/metal/tasks/0007-mpsgraph-float32-reductions.md)
remains independently `Blocked` because repeated direct-output probes returned positive zero where
the exact Model SUM/MEAN results are `2.0f`/`0.5f`.

[0009](backends/metal/tasks/0009-mpsgraph-float32-rank2-matmul-training-checkpoint.md) is
independently `Blocked`. Its mandatory Apple M3 Max probe passed selector, direct-target, Shape,
permutation, transpose, repetition, and independent-executable gates, but `K=1` multiplication by
one flushed positive/negative minimum and ordinary subnormals to zero. Strict host multiplication,
`+0.0f` accumulation, and `fmaf` preserve the minimum subnormal, so zero is outside every
Model-permitted association/FMA result. Signed zero is not a blocker. Independent review returned
`APPROVE-BLOCKER`. Tasks 0006, 0007, and 0009 retain no production, test, or probe changes.

[0010](backends/metal/tasks/0010-mpsgraph-float32-bool-predicates-and-selection.md) is independently
`Blocked`. On an Apple M3 Max at optimization levels `0` and `1`, three independent executables per
level and eight runs per case treated positive and negative minimum subnormal as equal to `±0`.
Raw identity preserved the input bits; ordinary, infinity, NaN, canonical BOOL target, canary, and
permutation controls passed. No production, test, or probe changes remain.

[0011](backends/metal/tasks/0011-mpsgraph-float32-scalar-pointwise-arithmetic.md) is independently
`Blocked`. At optimization levels `0` and `1`, three independently compiled executables per level
and eight runs per case produced 17,328 mismatches in 77,824 scalar checks. Failures include scalar
`ADD +0` retaining input `-0`, minimum-subnormal flushing, minimum-subnormal `DIV +0` producing
NaN, maximum-finite self-division producing zero, and infinity/maximum-finite division producing
NaN. Exact constants, direct targets, canaries, permutations, repetition, and identity controls
passed. Independent review confirmed no current Model relaxation admits the results. No
production, native, test, or probe changes remain.

[0012](backends/metal/tasks/0012-mpsgraph-float32-extrema-reductions.md) is independently
`Blocked`. At optimization levels `0` and `1`, fast math disabled, three executables per
length/level, and eight runs per case, its 43,824-execution gate found order-dependent opposite
signed zeros and zero/minimum-subnormal results: 2,016 MIN plus 2,016 MAX mismatches and 2,976
subnormal failures. NaN classification, minimum normals, identity, canaries, and bindings passed.
Independent review returned `APPROVE-BLOCKER`; no production, native, test, or probe changes remain.

[0013](backends/metal/tasks/0013-mpsgraph-float32-cumulative-scans.md) is independently
`Blocked`. At optimization levels `0` and `1`, fast math disabled, twelve executables covering
three per level/length pair for lengths `1` and `2`, and eight direct runs for all eight kind/mode
combinations, inclusive length-one SUM and PRODUCT flushed all six subnormal corpus values to
zero. Strict host operations preserved them; exclusive identities, raw identity, canaries, and
bindings passed. Corrected review evidence shows rank-six axis `5` valid for both kinds and axis
`6` correctly rejected, so axis is not a blocker. Independent review returned `APPROVE-BLOCKER`;
no production, native, test, or probe changes remain.

[0014](backends/metal/tasks/0014-mpsgraph-float32-affine-layout-composition.md) is `Complete` at
`01e81be2` with exact strict local affine/`CONTIGUOUS` composition. [0015](backends/metal/tasks/0015-accelerator-float32-tensor-binary-arithmetic.md)
is `Complete` at `42c4cfbf` plus `fb102a46` after the operation-by-operation DAZ/FTZ oracle, focused
validation, combined serial checkpoint, and independent Class C approval all passed.

Current Metal uses ABI v4 with exactly thirteen exports and node schema 8. Both profile matrices
admit exact canonical `FLOAT32` ABS; strict additionally admits NEG/affine/`CONTIGUOUS`, while
accelerator additionally admits tensor `FLOAT32` `ADD`/`SUB`/`MUL`/`DIV`, canonical
`SUM`/`MEAN`/`SUM_TO_SHAPE`, and positive static rank-two MATMUL with authenticated local
transposes. Java preflight rejects profile-incompatible node sets before native entry, version-nine
identities separate profile/topology compatibility, scalar reduction materialization remains
local-only, and Runtime/Trace remain profile-free.

Strategic gate: historical blocker evidence is preserved, and no backend task may define Model
semantics. Complete Model 0028 owns bounded reduction exact-zero sign freedom; Complete Metal 0020
implements that rule under accelerator only. Blocked Metal 0018 retains its unchanged historical
MATMUL contract and no production changes. Complete Model 0029 split the shared contraction row
and permits either sign only at publication when a complete nonempty FLOAT32 ACCELERATOR MATMUL
result is exact zero after an otherwise-permitted evaluation. Strict, convolution, products/
intermediates, empty contraction, all terms, nonzero values, classifications, DAZ/FTZ,
FMA/reassociation, reduced precision, identity, and tolerance rules stay unchanged. Metal 0021
completed its fresh full M3 oracle and implementation/validation checkpoint and remains `Ready`
only for its mandatory independent Class C review. No task may infer generic fast math, relax
unary semantics, or authorize gross special-value errors.

## Blocked, review-needed, and deferred work

- Metal 0006 is `Blocked` by the exact special-value/underflow failures recorded in its brief.
  Unblocking requires custom kernels or an explicitly accepted relaxed numerical contract.
- Metal 0016 is independently `Blocked` by reproducible `EXP`/`SIGMOID` result flushing and
  `SIGMOID` ordinary finite results beyond its one-ULP gate. Its exact three-operation contract is
  unchanged; an exact replacement for both failed operations is required to restart it.
- Metal 0017 is `Blocked` under its old accelerator reduction contract. Its corrected full probe
  completed 300 executables/2,400 runs and found `SUM([-0,-0]) -> +0`, while the old permitted set
  contained only `-0`; zero is not subnormal, so DAZ/FTZ did not admit the result. No production
  change or retained probe artifact remains. Complete Model 0028 owns the separate semantic
  decision. Metal 0020 is the dependent Complete successor at `9ddb75f6` plus `5b77c742`;
  historical 0017 remains Blocked rather than being reopened.
- Metal 0007 is independently `Blocked` by its repeated exact reduction counterexample. Unblocking
  requires an exact replacement route or an explicit Model contract change.
- Metal 0009 is independently `Blocked` by exact subnormal flushing in `K=1` MATMUL. Unblocking
  requires an exact replacement route or an explicit Model numerical-contract change; signed-zero
  behavior does not block it.
- Metal 0018 is `Blocked` under its unchanged historical full-contraction contract. Its mandatory
  fresh M3 smoke passed, but the first full direct `K=1` cell produced `+0` for
  `+0.0f * -1.0f`, where that task permitted only `-0`. The exact reduced-precision-none and probe
  controls passed; the probe was removed and no production edit exists. Model 0029 completed the
  separate final-publication semantic decision at `30826783`; the product remains `-0`, and only
  publication may choose `+0`. Ready Metal 0021, not 0018, is the implementation successor; its
  fresh full oracle and implementation checkpoint passed, and independent review remains.
- Metal 0010 is independently `Blocked` by exact subnormal comparison collapse. Unblocking requires
  an exact replacement route or an explicit Model numerical-contract change; canonical BOOL and
  ordinary/special-value controls do not remove the blocker.
- Metal 0011 is independently `Blocked` by 17,328 exact scalar mismatches in 77,824 checks.
  Unblocking requires an exact replacement route or explicit Model numerical-contract change;
  successful constant, target, canary, permutation, repetition, and identity controls do not
  remove the arithmetic blocker.
- Metal 0012 is independently `Blocked` by order-dependent signed-zero and subnormal extrema
  results. Unblocking requires an exact replacement route or explicit Model numerical-contract
  change; passing NaN classification, minimum-normal, identity, canary, and binding controls do
  not remove the blocker.
- Metal 0013 is independently `Blocked` by exact subnormal flushing in cumulative SUM/PRODUCT.
  Unblocking requires an exact replacement route or explicit Model numerical-contract change;
  passing exclusive identities, raw identity, canaries, bindings, and valid axis `5` do not remove
  the blocker. Axis `6` correctly rejects and is not a blocker.
- [OpenBLAS provider 0004](backends/openblas-provider/tasks/0004-optional-direct-bfloat16-output-gemm-capability.md)
  and dependent CPU 0010D1 remain an optional blocked branch. Pinned evidence proves neither the
  required exported direct BFLOAT16-output ABI nor full-contraction FLOAT32 accumulation followed
  by one final BFLOAT16 narrowing. Existing FLOAT32/FLOAT64 and portable BFLOAT16 work remain
  unaffected.
- CPU 0011 remains `Blocked` in the [CPU master plan](backends/cpu/master-plan.md) until a concrete
  Intel workload and supported oneMKL interface evidence both exist.
- CPU 0007A1D remains `Review needed`, not an active frontier; its stable semantic and structural
  evidence does not satisfy its failed performance gate.
- CPU 0010M is `Complete` with an evidence-backed no-change conclusion: production retains scalar
  compute, configured/available parallelism `1`/`1`, minimum elements per worker `1`, no worker
  group, and all existing fallbacks/thresholds. The report-only protocol is hardened; a future
  comparison still requires a separately reviewed, fully sealed matrix before measurement.
- Model 0026 remains an independent FLOAT16 Draft. Model 0027–0029, Config 0006, Engine 0018, CPU
  0017, and Metal 0015/0019/0020 are Complete. Metal 0016–0018 remain Blocked under their unchanged
  historical contracts. Detailed Metal 0021 is the sole Ready frontier; every other unfinished or
  blocked family remains unauthorized.
- Planning 0007 review found a stale glossary `Compile` status sentence and stale
  `GraphCompilationPort` Javadoc about the Engine facade. Compiler 0006B10 corrected and
  independently reviewed both without reopening Planning capability work.
- Prepare 0007 review found a stale CPU-finalizer status paragraph in the glossary's `Backend
  partition finalization` entry. Prepare 0008 corrected and independently reviewed it without
  reopening executable Prepare or CPU work.
- Data, Text, and Vision require the coordinated decision owned by
  [Data 0001](extensions/data/master-plan.md); Checkpoint requires its separate decision in the
  [Checkpoint plan](extensions/checkpoint/master-plan.md). Planning authorization alone does not
  create these Gradle projects or permit implementation.

## Nearest next step

Run the mandatory independent Class C review for
[Metal 0021](backends/metal/tasks/0021-accelerator-float32-rank2-matmul-after-zero-sign-refinement.md).
Its completely fresh full real-M3 oracle passed before production edits, the probe was removed,
and the scoped implementation and worker checkpoint are complete. Review must inspect the
final-publication-only zero-sign boundary, every-term and reduced-precision controls, strict-false
matrix, exact local-transpose authentication, schema-eight/version-nine identity, prepared
lifecycle/cleanup, CPU-free direct/linear/seeded-gradient Engine proof, Runtime/Trace absence,
changed scope, validation evidence, and clean probe removal. Keep task 0021 `Ready` until that
independent review approves it. Do not reopen Metal 0018, launch Model 0026, or change any unrelated
status.

## History policy

Completed task narratives, prior validation evidence, agent context identifiers, retrospectives,
and old reorder stories remain historical evidence in Git and task files. They are not default
executor input. Read historical evidence only when a current brief explicitly identifies an
unresolved question that requires it. Do not reconstruct global history before executing a
verified current brief.

## Update policy

Update this roadmap only when project priority, project-area status, authorized frontiers, a
blocker, or a cross-area parallel group changes. Keep detailed scope and result evidence in task
files; keep master plans and this roadmap to links and one-line status or gate summaries. If a
planning change conflicts with `ARCHITECTURE.md`, stop and use the architecture-decision process
instead of changing this roadmap alone.

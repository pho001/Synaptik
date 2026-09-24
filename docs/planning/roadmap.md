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
| 1 | [`modules/model`](modules/model/master-plan.md) | Complete through reviewed 0027; 0026 Draft | Model 0027 is Complete at implementation `ff86a302` after independent validation and Class C approval; 0026 remains an independent FLOAT16 Draft. |
| 2 | [`modules/trace`](modules/trace/master-plan.md) | In progress, deliberately interleaved; 0001–0002 Complete, 0003–0008 Draft | Resume 0003 only after its producer vocabulary is stable; no Trace task is Ready. |
| 3 | [`modules/backend-contract`](modules/backend-contract/master-plan.md) | Complete through 0004 | Reopen only for a concrete shared-contract need. |
| 4 | [`modules/config`](modules/config/master-plan.md) | In progress, interleaved; 0001–0003, 0006, and 0006A–0006B Complete; 0004–0005 and 0007–0008 Draft | 0006 completed at `314e049` plus `37e9e9db`; no Config task is Ready. |
| 5 | [`modules/planning`](modules/planning/master-plan.md) | Complete through documentation-only 0007; profile-qualified query cutover Complete through Engine 0018 | No Planning task is Ready. |
| 6 | [`modules/runtime`](modules/runtime/master-plan.md) | Complete through 0016; profile-free boundary preserved by Engine 0018 | No Runtime task is Ready. |
| 7 | [`modules/compiler`](modules/compiler/master-plan.md) | Complete through documentation-only 0006B10; 0006C and 0007 Draft; profile artifact cutover Complete through Engine 0018 | No Compiler task is Ready. |
| 8 | [`modules/prepare`](modules/prepare/master-plan.md) | Complete through documentation-only 0008; profile projection cutover Complete through Engine 0018 | No Prepare task is Ready. |
| 9 | [`backends/openblas-provider`](backends/openblas-provider/master-plan.md) | Required baseline Complete; optional 0004 Blocked and deferred | Resume 0004 only when both direct-BFLOAT16 application binary interface (ABI) and one-final-narrowing proofs exist. |
| 10 | [`backends/cpu`](backends/cpu/master-plan.md) | Mainline Complete through evidence-backed no-change 0010M; 0007A1D Review needed; 0010D1 and 0011 Blocked; 0017 profile realization Ready | [0017](backends/cpu/tasks/0017-explicit-accelerator-numerical-profile-realization.md) is a Ready parallel frontier from exact base `07a01b9c13c2ceea0922b9af82da3e6e08375306`. |
| 11 | [`modules/engine`](modules/engine/master-plan.md) | Complete through numerical-profile spine 0018 | 0018 completed at `ce7a7dfa` plus `07a01b9c`; no Engine task is Ready. |
| 12 | [`backends/metal`](backends/metal/master-plan.md) | Complete through 0014; 0005 binary capability withdrawn; 0006–0007 and 0009–0013 Blocked; 0015 profile realization Ready | [0015](backends/metal/tasks/0015-accelerator-float32-tensor-binary-arithmetic.md) is a Ready parallel frontier from exact base `07a01b9c13c2ceea0922b9af82da3e6e08375306`. |

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
[Config 0006](modules/config/tasks/0006-explicit-graph-numerical-profile-identity.md) is Complete at
implementation `314e049` plus remediation `37e9e9db`. Fresh Config tests and Javadoc passed 33
tests with zero failures, errors, or skips; generated enum/package pages were inspected;
baseline-aware Markdown evidence and `git diff --check` passed; and independent Class B review
returned final `APPROVE` with zero findings.

[Engine 0018](modules/engine/tasks/0018-explicit-numerical-profile-propagation-spine.md) is
Complete at implementation `ce7a7dfa` plus documentation/Javadoc remediation `07a01b9c`.
Required module tests/Javadocs, backend conformance, architecture tests, the actual
`EngineNumericalProfileIntegrationTest`, full build, documentation/diff checks, and the performance
gate passed; independent Class C review returned `APPROVE` with zero findings. `STRICT_IEEE`
remains the builder and `Engine.standard()` default; Runtime, run/session state, and Trace remain
profile-free.

The remaining DAG is:

`Engine 0018 -> {CPU 0017, Metal 0015}`

[CPU 0017](backends/cpu/tasks/0017-explicit-accelerator-numerical-profile-realization.md) and
[Metal 0015](backends/metal/tasks/0015-accelerator-float32-tensor-binary-arithmetic.md) are the two
Ready repository frontiers in parallel group `numerical-profile-backends` from exact common base
`07a01b9c13c2ceea0922b9af82da3e6e08375306`. They are safe to execute concurrently because the
completed spine fixes every shared API and semantic contract, CPU and Metal own disjoint backend
realizations/cache formats, and their integration tests are disjoint. Each worker owns only its
backend-specific guide; the Main planner exclusively owns shared planning, root/scoped architecture,
API/user documentation, shared capability/preparer guides, and glossary reconciliation. Executable
commits may integrate in either order; shared documents integrate last.

CPU 0017 admits `ACCELERATOR` with exact current CPU semantics and must prove both profiles through
observable forward/backward execution without relaxed math. Metal 0015 may restore only tensor
`FLOAT32` `ADD`/`SUB`/`MUL`/`DIV` under `ACCELERATOR` after its operation-by-operation bounded
DAZ/FTZ real-device probe; strict binary capability and every unproved row remain fail-closed.
Model 0026 remains an independent FLOAT16 Draft. Metal unary, reduction, MATMUL, comparison,
scalar, extrema, scan, and other blocked successors remain unauthorized.

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
historical delivery, but its `ADD`/`SUB`/`MUL`/`DIV` capability is withdrawn. An independent Apple
M3 Max audit found 12,096 subnormal-domain failures in 79,488 arithmetic comparisons across 576
runs while identity, permutation, input-preservation, and canary controls passed. Current Metal
rejects those occurrences and native wires `2..5`; CPU may own them when explicitly registered.
ABI v4 retains exactly thirteen exports.

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
implementation commit `01e81be2`. Its mandatory Apple M3 Max raw-bit probe, focused Metal,
conformance, Compiler, public Engine, Javadoc, architecture, and documentation validation passed;
independent Class C review returned `APPROVE` with zero findings. It composes the five existing
affine transforms plus exact `CONTIGUOUS` and retains ABI v4's thirteen exports and node schema 4.
Engine 0018 subsequently advanced profile-keyed Metal cache identities to version 5 while binary
wires `2..5`, cross-owner view transfer, blocked numerics, backward/training, FLOAT16, and BFLOAT16
remain excluded.

Strategic gate: the accepted Model contract now permits a bounded tensor-binary DAZ/FTZ result set,
but historical evidence across Metal 0006/0007 and 0009–0013 does not authorize those other
families. Ready Metal 0015 must pass its fresh operation-by-operation real-device oracle before
restoring only the four tensor binary operations under `ACCELERATOR`. Strict binary and all blocked
families remain fail-closed. Ready CPU 0017 independently uses exact current CPU results under both
profiles.

## Blocked, review-needed, and deferred work

- Metal 0006 is `Blocked` by the exact special-value/underflow failures recorded in its brief.
  Unblocking requires custom kernels or an explicitly accepted relaxed numerical contract.
- Metal 0007 is independently `Blocked` by its repeated exact reduction counterexample. Unblocking
  requires an exact replacement route or an explicit Model contract change.
- Metal 0009 is independently `Blocked` by exact subnormal flushing in `K=1` MATMUL. Unblocking
  requires an exact replacement route or an explicit Model numerical-contract change; signed-zero
  behavior does not block it.
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
- Model 0026 remains an independent FLOAT16 Draft. Model 0027, Config 0006, and Engine 0018 are
  Complete. CPU 0017 and Metal 0015 are the only Ready tasks; their briefs prohibit promotion of
  blocked unary/reduction/MATMUL/scalar/comparison/extrema/scan or other successors.
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

1. Launch [CPU 0017](backends/cpu/tasks/0017-explicit-accelerator-numerical-profile-realization.md)
   and [Metal 0015](backends/metal/tasks/0015-accelerator-float32-tensor-binary-arithmetic.md) from
   exact common base `07a01b9c13c2ceea0922b9af82da3e6e08375306` in isolated worktrees.
   They may execute and integrate in either order under parallel group `numerical-profile-backends`.
2. After both executable commits land, the Main planner reconciles the reserved shared documents
   once and runs the numerical-profile backend integration checkpoint. Do not launch Model 0026 or
   any blocked Metal successor.

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

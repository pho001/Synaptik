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
| 1 | [`modules/model`](modules/model/master-plan.md) | Complete through independently reviewed 0031; 0026 Draft | Model 0030/0031 completed the total recursive profiles and backend-independent strict unary baseline; remediation `97cb9d11` received independent `APPROVE` with zero findings. No Model task is Ready. |
| 2 | [`modules/trace`](modules/trace/master-plan.md) | Complete through 0003; 0004–0008 Draft | [Trace 0003](modules/trace/tasks/0003-backend-preparation-and-invocation-diagnostic-dtos.md) completed its JDK-only DTO/ID surface and validation; no Trace task is Ready. |
| 3 | [`modules/backend-contract`](modules/backend-contract/master-plan.md) | Complete through 0004 | Reopen only for a concrete shared-contract need. |
| 4 | [`modules/config`](modules/config/master-plan.md) | In progress, interleaved; 0001–0003, 0006, and 0006A–0006B Complete; 0004–0005 and 0007–0008 Draft | 0006 completed at `314e049` plus `37e9e9db`; no Config task is Ready. |
| 5 | [`modules/planning`](modules/planning/master-plan.md) | Complete through documentation-only 0007; profile-qualified query cutover Complete through Engine 0018 | No Planning task is Ready. |
| 6 | [`modules/runtime`](modules/runtime/master-plan.md) | Complete through 0016; profile-free boundary preserved by Engine 0018 | No Runtime task is Ready. |
| 7 | [`modules/compiler`](modules/compiler/master-plan.md) | Complete through documentation-only 0006B10; 0006C and 0007 Draft; profile artifact cutover Complete through Engine 0018 | No Compiler task is Ready. |
| 8 | [`modules/prepare`](modules/prepare/master-plan.md) | Complete through documentation-only 0008; profile projection cutover Complete through Engine 0018 | No Prepare task is Ready. |
| 9 | [`backends/openblas-provider`](backends/openblas-provider/master-plan.md) | Required baseline Complete; optional 0004 Blocked and deferred | Resume 0004 only when both direct-BFLOAT16 application binary interface (ABI) and one-final-narrowing proofs exist. |
| 10 | [`backends/cpu`](backends/cpu/master-plan.md) | Complete through profile realization 0017; 0007A1D Review needed; 0010D1 and 0011 Blocked | CPU 0017 completed at `372a8b98`; both profiles use identical exact CPU capability, routes, execution, and profile-separated identities. No CPU task is Ready. |
| 11 | [`modules/engine`](modules/engine/master-plan.md) | Complete through numerical-profile spine 0018 | 0018 completed at `ce7a7dfa` plus `07a01b9c`; no Engine task is Ready. |
| 12 | [`backends/metal`](backends/metal/master-plan.md) | Task 0055 Complete; 0052 Complete; 0054 historical | [Metal 0055](backends/metal/tasks/0055-private-schema-type-cardinality-foundation.md) completed at `ddeff1b2`: ABI 5, schema 13, all-six rank-`0..16` typed transfer, identity 14, variable cardinality, and zero capability widening are landed. No Metal task is Ready. |

| 13 | [`backends/cuda`](backends/cuda/master-plan.md) | Draft | Create a detailed 0001 brief only when CUDA becomes the authorized frontier. |
| 14 | [`extensions/onnx`](extensions/onnx/master-plan.md) | Draft | Define the first bounded mapping task only at an authorized frontier. |
| 15 | [`extensions/data`](extensions/data/master-plan.md) | Draft; architecture decision required | 0001 must authorize the Data/Text/Vision modules, build edges, decision record, and architecture tests first. |
| 16 | [`extensions/nn`](extensions/nn/master-plan.md) | In progress under recorded interleaves; 0001–0020C, 0021A, and 0025–0025A Complete; 0021B–0024 Draft | 0021B needs a detailed brief and truthful concrete-backend recurrent coverage; no NN task is Ready. |
| 17 | [`extensions/text`](extensions/text/master-plan.md) | Draft; architecture decision required | Wait for Data 0001, then define the first tokenizer task. |
| 18 | [`extensions/vision`](extensions/vision/master-plan.md) | Draft; architecture decision required | Join the coordinated Data 0001 decision before decoder or image APIs. |
| 19 | [`extensions/training`](extensions/training/master-plan.md) | Complete through 0001 | No later Training task is Ready; parameter groups and broader optimizers remain Draft. |
| 20 | [`extensions/checkpoint`](extensions/checkpoint/master-plan.md) | Draft; architecture decision required | Authorize the model-only and optional Training adapter boundaries before 0001. |
| 21 | [`tools/benchmarks`](tools/benchmarks/master-plan.md) | Complete through local task 0001; 0002–0003 Draft; Metal 0043 Complete | Metal-owned Task 0043 completed its fixed report-only two-route benchmark at remediation `77e6091b` after same-reviewer approval; no separate benchmark task is Ready. |
| 22 | [`tools/tuning`](tools/tuning/master-plan.md) | Complete through 0004 | No Tuning task is Ready. |
| 23 | [`tools/cli`](tools/cli/master-plan.md) | Draft | Define commands only after their Engine and diagnostic contracts are stable. |

## Authorized frontiers

Model numerical profiles

[Model 0031](modules/model/tasks/0031-strict-unary-numerical-baseline.md) and
[Model 0030](modules/model/tasks/0030-total-recursive-accelerator-numerical-floor.md) are
`Complete`. Their documentation/Javadoc-only decisions own the backend-independent strict unary
sets and total recursive accelerator contract. The recursive-contract remediation is `2d95ab71`;
the complete strict-result-set remediation is `97cb9d11`. Focused validation passed, and
independent Class C rereview after `97cb9d11` returned `APPROVE` with zero findings. Neither task
changed executable behavior or capability.

Trace and Metal diagnostics

[Trace 0003](modules/trace/tasks/0003-backend-preparation-and-invocation-diagnostic-dtos.md)
is Complete from clean base `c734f0bb52cd35b94cd794cc580326dca09ec62b`; its final independent
public-API/documentation review returned `APPROVE` after remediation at
`0e796a82f18fbaaa1d5210638f41e0785e22094c`. It added only trace-local backend, device,
prepared-unit, and invocation IDs plus closed neutral immutable preparation/invocation DTOs. It
preserved the existing event envelope/phase/level/payload marker and added no emitter, observer,
consumer, state, dependency, string map, or serialization behavior.

[Metal 0041](backends/metal/tasks/0041-prepared-route-and-invocation-trace.md) is Complete at
implementation `ba16d942` plus remediation `386705ca`; final independent Class C rereview returned
`APPROVE` with zero findings. Its bounded implementation adds the optional Metal observer overload,
disabled fast path, producer-local IDs/events, PREPARE finalization and RUN native-invocation
outcomes, truthful `NOT_QUERIED`, containment/redaction, and public Engine lifecycle observability
without an Engine production or native ABI change.

`Trace 0003 (Complete) -> Metal 0041 (Complete)`

Metal route tuning and fixed benchmark

[Metal 0042](backends/metal/tasks/0042-metal-route-tuning-workflow.md) is Complete at
`9feb2505705263b6efb417d606678c606c2b9598` after final independent Class C review returned
`APPROVE` with zero findings. It adds retained public `MetalLocalWorkloadTuning` and
`MetalCompletePlanTuning` collaborations around the existing private version-twelve
singleton-`NEG` candidates and a minimum private Engine Metal adapter plus backend-neutral
selected-owner fallback. Scope remains exactly one two-route Metal partition, one fixed
complete-plan candidate, session-only reuse, unchanged public Config/result, preserved CPU identity
bytes/behavior, and no native/schema/capability/benchmark work.

[Metal 0043](backends/metal/tasks/0043-reproducible-metal-route-benchmark.md) is Complete at
remediation `77e6091b2a452faa04fa2b674bc295ddc7be88b7` after same-reviewer Class C approval with zero
remaining findings. It implements one reviewed report-only `[1_048_576]` singleton FLOAT32 `NEG`
benchmark over both current opaque local candidates, with thread-safe snapshot attestation,
reconstructible generator schema 2, exact raw-bit checks, and no winner/cache/threshold/artifact.

[Metal 0044](backends/metal/tasks/0044-custom-singleton-neg-benchmark-evaluation.md) is Complete
from exact planning revision `3d458b7a1d356f32fcd2c04de408ae9aa22355b0`. It records only the
bounded descriptive evidence and a no-change conclusion: retain both routes and the existing
`CUSTOM_SINGLE_NEG` no-selected-decision safe heuristic without a winner, performance endorsement,
or production change.

[Metal 0045](backends/metal/tasks/0045-verified-local-native-package.md) is Complete at P1
remediation `26c6c911` after the same independent Class C reviewer returned `APPROVE` with zero
remaining findings. Its atomic macOS-arm64 build, explicitly ad-hoc-signed schema-1 local package,
independent verifier, original/P1 rejection proofs, and packaged-dylib Metal/Engine tests passed.
It preserves the ABI, native source, Java, Gradle, absolute-path loader, and public-release
boundary.

[Metal 0046](backends/metal/tasks/0046-explicit-verified-native-local-archive.md) is Complete at
implementation `4aad1ab6ced318107e65bb9beef0013f8a7ff6e5` after independent Class C review returned
`APPROVE` with zero findings. Its explicit opt-in reproducible local ZIP retains the reviewed
package, manual extraction, absolute-path loading, and unauthenticated non-release boundary.

[Metal 0047](backends/metal/tasks/0047-apple-silicon-metal-ci.md) is planning-only and Blocked.
Standard `macos-26` arm64 has no documented usable Metal GPU; documented M2 GPU runner
`macos-26-xlarge` requires an eligible organization/enterprise plan, billing, positive spending,
repository access, and a real provider proof. No workflow or compile-only substitute was added.

[Metal 0048](backends/metal/tasks/0048-persistent-immutable-splats-and-no-general-pool-closure.md)
is Complete at independently approved implementation `89f9fbb9`. It persists immutable source-owned
splats per prepared execution with fresh read-only run bindings and exact deferred release. It
deliberately does not pool mutable outputs or MPSGraph address workspaces because `RunResult` and
concurrent-run ownership lack a safe exclusive return/reset/eviction contract.

[Metal 0049](backends/metal/tasks/0049-synchronous-single-default-device-contract.md) is Complete
after P1 status-drift remediation `6d4246f7` and final cumulative independent Class C `APPROVE`
with zero findings. It codifies synchronous completed-state result semantics, one system-default
Metal context per integration, context-bound resources, distinct non-physical device/trace/tuning
identities, no cross-device behavior, concrete successor gates, and the no-general-pool decision.

[Metal 0050](backends/metal/tasks/0050-final-packaged-metal-repository-verification.md) is Complete.
From exact Task 0049 completion revision `2c6326a9`, the native package/archive/extraction verifier
chain passed and the sole final full build executed all 87 actionable tasks against the extracted
dylib. All 130 Metal-backend and 15 explicit Metal integration tests had zero failures, errors, or
skips; the two real-Metal public `prepareTuned` cases are included in the integration total.

`Metal 0004 (Complete) + Metal 0041 (Complete) + Engine 0009/0015–0018 (Complete) + tools/tuning 0001–0002 (Complete) -> Metal 0042 (Complete) -> Metal 0043 (Complete) -> Metal 0044 (Complete) -> Metal 0045 (Complete) -> Metal 0046 (Complete) -> Metal 0047 (Blocked)`

`Metal 0046 (Complete) + Runtime 0016 (Complete) + Prepare 0006 (Complete) + Engine 0010 (Complete) -> Metal 0048 (Complete)`

`Metal 0048 (Complete) + Runtime 0016 (Complete) + Prepare 0006 (Complete) + Engine 0010 (Complete) + Planning device eligibility + Trace 0003 + Metal 0041–0042 (Complete) -> Metal 0049 (Complete)`

`Metal 0049 (Complete) -> Metal 0050 (Complete final program verification)`

Numerical profiles

[Model 0030](modules/model/tasks/0030-total-recursive-accelerator-numerical-floor.md) is `Complete`
after remediation `2d95ab71683698753c9ac64d9373fde8301c9404` and strict-baseline remediation
`97cb9d116bd85ee6a0dfbf2e9b70d32604633c85`. Model 0031 owns public backend-independent strict
result sets for all unary kinds and accepted floating types; Model 0030 owns the separate inclusive
ordered-binary32 accelerator rule and closed composite primitive-site ledger. Independent Class C
rereview after `97cb9d11` returned `APPROVE` with zero findings. No executable behavior or backend
capability changed.

[Metal 0051](backends/metal/tasks/0051-accelerator-float32-exp-sigmoid-recursive-floor-realization.md)
is Blocked before complete-domain authorization, route adjudication, or capability. Its consumed
frozen 83-word Apple M3 Max oracle gave bounded numerical passes to direct MPSGraph EXP and the
MPSGraph stable SIGMOID composition, and rejected custom EXP/custom SIGMOID/direct SIGMOID. The
passes are regression evidence only: direct EXP lacks an authoritative all-binary32/rank-`1..16`
contract and is `DOMAIN-BLOCKED`; the composition inherits that blocker. Task 0051 froze the proof
obligations and remains historical. Task 0053 now owns the concrete source/certificates,
independent review, and one newly authorized successor oracle, but stays Blocked on that
complete-domain work. Task 0051 reserved no production vocabulary; Task 0052 later consumed wires
`20..34`, and Complete Task 0055 registers `EXP=55`/`SIGMOID=64` inside its zero-capability
schema-13 foundation.
CPU requires no migration because its exact realizations remain valid members of the widened result
set.

[Metal 0053](backends/metal/tasks/0053-certified-accelerator-float32-custom-exp-stable-sigmoid.md)
supersedes 0051 for all future EXP/SIGMOID realization while preserving that historical record.
Its custom EXP must have concrete source/constants and kernel-checked proof over all raw words,
DAZ, overflow/underflow, range reduction, table/index and approximation bounds, every RNE/FMA/FTZ
site, inclusive ordinary EXP distance `<=5`, and rank-`1..16` launch/index coverage. Stable
SIGMOID must be proved recursively from exact guard/NEG, the approved EXP, and explicit ADD/DIV
sites, never by a final envelope. Current opaque direct/inherited routes stay `DOMAIN-BLOCKED`;
only independently domain-proved alternatives may join the certified custom routes. Task 0053 is
Blocked until that proof package passes read-only review. Only then may its one new frozen-corpus
oracle run; Task 0051 is never rerun. A sole survivor needs no comparative cost gate; multiple
survivors may close without timing only through strict dispatch/temporary-byte dominance under
identical proven semantics/domain, otherwise selection awaits a controlled environment. Local
timing is diagnostic only. Capability, schema, identities, routes, and production remain unchanged.

[Metal 0052](backends/metal/tasks/0052-accelerator-float32-comparisons-extrema-scans.md)
is Complete for six FLOAT32 comparisons, tensor/scalar/clamp/reduction extrema, and
CUM_SUM/CUM_PROD. Every opaque direct candidate remains `DOMAIN-BLOCKED`; all sixteen auditable
custom candidates proved `DOMAIN-PASS` and passed the sole numerical oracle. Fourteen operation
rows had one survivor. Fused CLAMP is fixed solely because its one dispatch/zero route bytes
strictly dominate composed CLAMP's two dispatches/4,194,304 bytes; all 128 local timing samples
remain diagnostic history. The original cutover landed schema 12/identity 13; Complete Task 0055
later migrated current production to ABI 5/schema 13 and identity 14 without changing capability
or routes.

[Metal 0054](backends/metal/tasks/0054-current-accelerator-115-kind-completeness-audit.md) remains
the exact Complete read-only audit of base `93b3d379`: 40 enums, 115 constants, and
`19+2+15+79=115`. Its schema-11 and Task-0052 capability-false counts are historical pre-cutover
facts; the document explicitly names its supersession and grants no production scope.

[Metal 0055](backends/metal/tasks/0055-private-schema-type-cardinality-foundation.md) is Complete
at `ddeff1b2`. It atomically replaced the fixed 160-byte record with one bounded schema-13 program
image having `0..N` inputs, `1..N` outputs, explicit per-value descriptors, all six current Model
carriers, BOOL/INT/rank-zero structure where legal, closed attributes `0..41`, ABI 5 with the same
thirteen export names, typed CPU/Metal transfer, identity 14, independent validation, and package
metadata. Capability remains exactly 34 executable and 81 registered unsupported kinds.

[CPU 0017](backends/cpu/tasks/0017-explicit-accelerator-numerical-profile-realization.md) is
Complete at implementation `372a8b98`. For every backend, capability/behavior under
`STRICT_IEEE` is a subset of capability/behavior under `ACCELERATOR` for the same occurrence
domain: accelerator may always execute the strict result and may add only named Model-authorized
results. CPU already realizes this invariant through one identical capability matrix and unchanged
portable/OpenBLAS routes; requested profiles remain distinct in plans, generated artifacts,
workloads, tuning, and cache compatibility without Runtime/Trace state.

[Metal 0015](backends/metal/tasks/0015-accelerator-float32-tensor-binary-arithmetic.md) is Complete
at implementation `42c4cfbf` plus evidence-wording remediation `fb102a46`. Complete
[Metal 0019](backends/metal/tasks/0019-exact-profile-qualified-float32-abs.md) adds exact canonical
`ABS` to both profile matrices. Complete Metal 0020 adds accelerator reductions, and Complete Metal
0021 adds positive static rank-two MATMUL with authenticated local transposes and seeded gradients.
Complete Metal 0022 implements the required monotonic matrix: exact canonical NEG/ABS and
affine/`CONTIGUOUS` are common to both profiles, while tensor binary/reduction/MATMUL remains
accelerator-only and strict-false. Implementation `41517594`, documentation remediation
`90cd5fd9`, and final independent Class C `APPROVE` with zero findings close it. Complete Metal 0023
landed at `9a7911c9` plus evidence `80e6cedd`; final independent lean Class C review approved with
zero findings after one concrete-risk production-ABI SNaN probe. Complete Metal 0024 landed at
implementation `a947e574732273bee4469d42afe8935082d53109` plus documentation remediation
`aa191ca469010d081150e97dcddd504ec626dd9e`; final independent lean Class C review approved with
zero findings. Complete Metal 0025 delivered bounded common-profile exact FLOAT32 UNFOLD_AXIS at
implementation `44edd860`, remediation `d0a947fa`, and final reviewed remediation `f88066e3`;
final independent lean Class C review approved with zero findings. Historical Metal 0026 and 0027
remain Blocked under their frozen profile-common/no-FTZ gates. Complete Model 0030 makes their
observed SOFTMAX/BatchNorm FTZ values recursively reachable for `ACCELERATOR`, but neither sample
proves a complete opaque-selector subset or supplies a strict route. Metal 0030 remains Blocked by
signed-zero/NaN max-pool winner-selection failures. Metal 0031 is independently Blocked without a
probe because one execution cannot prove an undocumented shape-dependent Conv2d contraction is a
complete recursive subset. Metal 0032 remains Blocked under its frozen profile-common task: MSE
arithmetic is recursively reachable for `ACCELERATOR`, but MPSGraph has no direct selector and no
complete decomposition/custom route, schema, formula, or gradient proof exists. Metal 0033 remains
Blocked: attention arithmetic is recursively reachable for `ACCELERATOR`, but additive mask
semantics mismatch Model and the opaque direct selector lacks complete-domain proof for
contractions, scale, guards, softmax, specials, output, gradients, and shape-dependent algorithms.
Metal 0034 remains Blocked because exact ordering's stability, NaN, signed-zero, subnormal,
exact-bit, and shape-algorithm contracts are undocumented. Metal 0035 remains Blocked because
hidden-random one-output dropout and opaque Philox state cannot implement Model's exact INT64[2]
initialization/transition or mandatory three-output dropout, and current Metal lacks zero-input,
multi-output, and INT64 representation. Metal 0036 remains Blocked: direct Conv3d inherits the
opaque contraction boundary; average Pool3d arithmetic is recursively reachable but lacks
mapping/divisor/route proof, while max Pool3d retains exact winner selection. Metal 0037 remains
Blocked: recurrent arithmetic is recursively reachable for `ACCELERATOR`, but direct RNN lacks
runtime INT64 valid lengths, atomic validation, skipped padded work, and `finalHidden`, and no
complete recurrence/state/gradient route proof exists. None changed production.
Documentation/audit-only Metal 0038 is Complete. Metal 0040 is Blocked after its one BFLOAT16
Gather gate canonicalized the first selected signaling NaN.
Metal 0041 is Complete at implementation `ba16d942` plus remediation `386705ca` after final Class C
approval with zero findings. Metal 0042 is Complete at `9feb2505`; Metal 0043 is Complete at
`77e6091b`; Metal 0044 is Complete from planning revision `3d458b7a`; Metal 0045 is Complete at
same-reviewer-approved remediation `26c6c911`; Metal 0046 is Complete at independently approved
implementation `4aad1ab6`; Metal 0047 is Blocked; Metal 0048 is Complete at `89f9fbb9`; Metal 0049
is Complete after approved remediation `6d4246f7`; Metal 0050 final verification and read-only
Metal 0054 completeness audit are Complete.

Task 0019 landed at implementation `a6d1796d` plus mixed-owner test remediation `bcb717a6`. Its
native ABI/export, Metal, conformance, real Engine, architecture, full-build, documentation, and
diff checks passed; independent Class C final review returned `APPROVE` with zero findings.

The completed cross-area profile DAG is:

`Model 0027 -> Config 0006 -> Engine 0018 -> {CPU 0017, Metal 0015}`

The active semantic and Metal serial DAG is:

`Metal 0015 -> {Metal 0016 (Blocked), Metal 0019 (Complete) -> Metal 0017 (Blocked)}`

`Model 0028 (Complete) -> Metal 0020 (Complete) -> Metal 0018 (Blocked)`

`Metal 0018 blocker evidence -> Model 0029 (Complete) -> Metal 0021 (Complete) -> Metal 0022 (Complete) -> Metal 0023 (Complete) -> Metal 0024 (Complete) -> Metal 0025 (Complete) -> {Metal 0026 (Blocked), Metal 0027 (Blocked), Metal 0030 (Blocked), Metal 0031 (Blocked), Metal 0032 (Blocked), Metal 0033 (Blocked), Metal 0034 (Blocked), Metal 0035 (Blocked), Metal 0036 (Blocked), Metal 0037 (Blocked)}`

`Model 0031 (Complete) -> Model 0030 (Complete) -> {Metal 0051 (historical Blocked) -> Metal 0053 (Blocked on concrete proof/review), Metal 0052 (Complete)}`

`Metal 0051 + reviewed Metal 0052 evidence + Metal 0053 + historical source -> Metal 0054 (Complete pre-cutover audit); Metal 0052 production cutover (Complete) supersedes that inventory`

`Metal 0022/0023/0024/0025 (Complete) + finalized evidence through Metal 0037 -> Metal 0038 (Complete)`

[Metal 0016](backends/metal/tasks/0016-profile-qualified-float32-abs-exp-sigmoid.md) remains
historically Blocked without production changes. Its frozen no-FTZ/one-step Apple M3 Max gate
proved exact `ABS`, while `EXP`/`SIGMOID` flushed representable subnormal results and sigmoid
produced two-step ordinary finite results. All controls passed and the probe was removed.

[Metal 0019](backends/metal/tasks/0019-exact-profile-qualified-float32-abs.md) is Complete and adds
only canonical exact `ABS` under both profiles. Complete Model 0030 makes recursive `ACCELERATOR`
`EXP`/`SIGMOID` results semantically reachable. Metal 0051's bounded numerical oracle passed direct
EXP and composed SIGMOID, but both are `DOMAIN-BLOCKED` because the opaque EXP selector lacks an
authoritative full-domain contract; its other candidates failed and its invocation is never rerun.
Metal 0053 is the planning-only successor and remains Blocked until concrete custom source/constants
pass the complete machine-checked proof and independent review. Only then may its one new oracle
run. A sole survivor needs no comparison; multiple identical-domain survivors close only through
strict structural dominance, otherwise selection awaits a controlled environment. Local timing is
diagnostic only.
Metal 0052's independent proof, sole regression oracle, structural survivor adjudication, fixed
routes, and production cutover are Complete. Its opaque routes remain domain-blocked; fused CLAMP
is fixed solely by strict dispatch/temporary-byte dominance, and all other rows had a sole
survivor. The bounded corpus remains regression evidence, not complete-domain proof. Gross
class/sign-failing 0006 selectors remain unusable.

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
Class C `APPROVE` with zero findings. Metal 0021 passed its fresh full oracle and completed at
implementation `ef2c6a1a` plus evidence corrections `be5543f9`/`5631d51f`; final independent
Class C review returned `APPROVE` with zero findings. Metal 0022 completed at `41517594` plus
`90cd5fd9` and final independent approval; Metal 0023 completed at `9a7911c9` plus evidence
`80e6cedd` and final independent approval. Metal 0024 completed at
`a947e574732273bee4469d42afe8935082d53109` plus documentation remediation
`aa191ca469010d081150e97dcddd504ec626dd9e` and final independent approval. Metal 0025 completed at
implementation `44edd86092509348e1e72cf7f0f4c3b13d141fa8`, first remediation
`d0a947fa953bddb2714c1917c7114ac345c530b7`, and final remediation/reviewed revision
`f88066e3ad0547987bb03b2d18ed2813f97de223`; final approval had zero findings. Metal 0026,
Metal 0027, Metal 0030, Metal 0031, Metal 0032, Metal 0033, Metal 0034, Metal 0035, Metal 0036, and
Metal 0037 are Blocked without production changes; 0031–0037 ran no probe. Documentation/audit-only
Metal 0038 is Complete. Planning-only Metal 0039 and failed-gate Metal 0040 remain Blocked; Metal
0041 is Complete at implementation `ba16d942` plus remediation `386705ca` after final approval with
zero findings. Metal 0042 is Complete at `9feb2505`; Metal 0043 is Complete at `77e6091b`;
Metal 0044 is Complete from planning revision `3d458b7a`.

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
Task 0020 adds accelerator `SUM=13` and `MEAN=14`; Complete task 0021 appends accelerator
`MATMUL=15`. Complete task 0023 retains ABI v4's exactly thirteen exports, advances node schema to
9, appends `GATHER=16`, `ONE_HOT=17`, and `DEPTH=5`, and advances route, candidate, compatibility,
workload, exact-policy, and codec identities to version ten. Evidence `80e6cedd` and final
independent lean Class C approval close it. Complete 0024 advances schema to 10, appends
`SCATTER_ELEMENTS=18`, and advances those identities to version eleven. Complete 0025 implements
schema 11/wire 19/typed window-axis state and version-twelve identities for exact bounded canonical
FLOAT32 UNFOLD_AXIS; implementation `44edd860`, remediations `d0a947fa`/`f88066e3`, and final
independent approval close it. Blocked 0026 ran exactly one `[4,8]` production-settings direct
SOFTMAX gate, returned positive zero instead of reference subnormal `0x0008ec28` at flattened index
18, removed the probe, and did not begin its then-planned schema 12/wire 20/version-thirteen
implementation. Blocked 0027 independently referenced those planned next numbers but implemented
none: its sole `[2,8,2]` BatchNorm gate flushed both signed minimum-subnormal outputs to same-signed
zero. Blocked 0030 independently ran exactly one raw-winner MAX_POOL2D execution, preserved its
exact failure evidence, removed its artifacts, and implemented none of those values. Blocked Task
0040 likewise consumed none. These are historical per-task facts; Task 0052 later landed schema 12,
wires `20..34`, attributes `7..9`, and version-thirteen identities.
Blocked 0031 ran no probe: its direct Conv2d descriptor mapping is structurally complete, but the
selector's shape-dependent contraction algorithm is undocumented and lean policy prohibits the
matrix required to authorize full geometry.
Blocked 0032 also ran no probe and remains Blocked under its frozen profile-common premise.
Complete Model 0030 recursively reaches `ACCELERATOR` MSE arithmetic, but no direct selector
or complete decomposition/custom route, schema, formula, gradient, and complete-domain proof
exists.
Blocked [0033](backends/metal/tasks/0033-profile-common-canonical-float32-no-grad-unmasked-noncausal-default-scale-sdpa-forward.md)
also ran no probe. Complete Model 0030 recursively reaches `ACCELERATOR` attention arithmetic,
but the additive mask surface mismatches Model's BOOL eligibility semantics and the opaque direct
selector lacks complete proof for contractions, scale, guards, softmax, specials, output,
gradients, and shape-dependent algorithms.
Blocked [0034](backends/metal/tasks/0034-profile-common-canonical-float32-no-grad-ascending-sort.md)
also ran no probe. Its smallest candidate is profile-common canonical rank-one FLOAT32 no-grad
ascending SORT. The direct selector is runtime-available but omits the complete exact ordering
contract. ARGSORT returns INT32 rather than Model INT64, and mandatory two-output TOP_K cannot fit
the current Metal one-output schema.
Blocked [0035](backends/metal/tasks/0035-profile-common-canonical-float32-no-grad-explicit-state-dropout.md)
also ran no probe. Its smallest nondegenerate candidate is profile-common exact canonical rank-one
FLOAT32 no-grad INITIAL_STATE plus explicit-state DROPOUT over the complete probability domain.
Direct dropout hides randomness and returns one output; opaque Philox state cannot establish exact
Model key/counter-plus-N, mask, or next-state semantics. Exact custom random execution requires a
complete zero-input/multi-output/local-INT64 schema and lifecycle.

Blocked [0036](backends/metal/tasks/0036-extended-3d-inference.md) also ran no probe. Direct Conv3d
inherits 0031's opaque shape-dependent contraction. Average Pool3d arithmetic is recursively
reachable for `ACCELERATOR` but lacks complete NCDHW mapping, ceil-tail, divisor, and route proof;
max Pool3d retains exact winner selection exposed by 0030. UNFOLD3D/FOLD3D remain separate custom
movement/overlap work, and conditional schema is unimplemented and unreserved.

Blocked [0037](backends/metal/tasks/0037-profile-common-canonical-float32-no-grad-forward-rnn-tanh.md)
also ran no probe. Complete Model 0030 recursively reaches `ACCELERATOR` recurrent arithmetic,
but direct RNN has no runtime INT64 valid-length input, atomic validation, skipped padded work,
positive-zero padding contract, or final-hidden output. No complete recurrence/state/gradient
route proof or five-input/two-output/caller-INT64 schema exists.

[0006](backends/metal/tasks/0006-mpsgraph-float32-unary-algebra.md) remains historically `Blocked`;
its direct-selector gross class/sign failures remain outside Model 0030, while a future conforming
custom or composed `ACCELERATOR` route requires complete formula, schema, capability, and route
proof. [0007](backends/metal/tasks/0007-mpsgraph-float32-reductions.md)
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

Current Metal uses ABI 5, exactly thirteen exports, and one bounded schema-13 program image.
Both profiles retain exact canonical FLOAT32 NEG/ABS/affine/CONTIGUOUS, bounded UNFOLD_AXIS, and
exact typed GATHER, ONE_HOT, and SCATTER_ELEMENTS/NONE. Accelerator additionally admits the
existing arithmetic, SUM/MEAN/SUM_TO_SHAPE, and bounded MATMUL rows plus Task-0052 comparisons,
tensor/scalar/clamp/reduction extrema, and scans. The Task-0052 route materializes every logical
value in a declared run-owned buffer and invokes one fixed custom whole-program native executable
per hot run. Schema 13 carries type wires `1..6`, operation wires `1..115`, and attribute wires
`0..41`; only the existing 34 operation kinds are executable. Workload, policy, candidate,
compatibility, route, and codec identities are version fourteen; candidate wires and complete-plan
wrapper remain stable.

Metal 0025 remains Complete at reviewed revision `f88066e3`; its schema-11/version-twelve facts are
historical. Blocked 0026–0027/0030–0037/0039–0040 changed no executable capability. Complete 0041
changes no capability. Metal 0042 still exposes only bounded singleton-NEG local tuning. Historical
0051 and planning-only 0053 added no current wire. Metal 0052, superseded historical audit 0054,
and zero-new-kind foundation 0055 are Complete; Metal 0047/0051/0053 remain Blocked. Task 0055
landed type wires `1..6`, variable node cardinality, typed transfer, ABI 5/schema 13, and identity
14 without changing capability. No backend task may define Model semantics.

Complete Model 0028 owns bounded reduction exact-zero sign freedom; Complete Metal 0020
implements that rule under accelerator only. Blocked Metal 0018 retains its unchanged historical
MATMUL contract and no production changes. Complete Model 0029 split the shared contraction row,
and Complete Metal 0021 implemented the bounded successor at `ef2c6a1a`; evidence corrections
`be5543f9`/`5631d51f` and final independent Class C `APPROVE` with zero findings close that task.
Complete Metal 0022 reuses exact NEG/affine/`CONTIGUOUS` lowering under accelerator, supports valid
whole-partition composition, and retained its ABI/schema/identity versions. Implementation
`41517594`, documentation remediation `90cd5fd9`, and final independent Class C `APPROVE` with zero
findings close it. No numerical probe ran because no selector, wire, lowering, or result freedom
changed. Complete Metal 0023 passed its documented-selector/minimal-smoke evidence and final
independent review at implementation `9a7911c9` plus evidence `80e6cedd`. Complete Metal 0024
passed its one-run Set smoke, implementation at `a947e574732273bee4469d42afe8935082d53109`,
documentation remediation at `aa191ca469010d081150e97dcddd504ec626dd9e`, and final independent
lean Class C approval with zero findings. Complete Metal 0025 passed its one-run movement smoke,
implementation `44edd860`, selector-cap/capability remediation `d0a947fa`, wrong-kind remediation
`f88066e3`, and final independent approval. Metal 0026 and 0027 remain historically Blocked under
their frozen profile-common/no-FTZ gates. Complete Model 0030 makes those observed FTZ values
recursively reachable only for `ACCELERATOR`; samples do not prove complete opaque-selector
subsets, strict capability remains false, and production remains unchanged. Metal 0030 remains
Blocked because exact max-pool signed-zero/NaN winner selection failed. Metal 0031 remains Blocked
because an opaque shape-dependent Conv2d contraction lacks complete recursive-subset proof. Metal
0032 remains Blocked under its frozen profile-common premise; `ACCELERATOR` MSE arithmetic is
recursively reachable, but no direct selector or complete decomposition/custom route, schema,
formula, and gradient proof exists. Metal 0033 remains Blocked; `ACCELERATOR` attention arithmetic
is recursively reachable, but additive mask semantics mismatch Model and the opaque selector lacks
complete proof for contractions, scale, guards, softmax, specials, output, gradients, and its
shape-dependent algorithms. Metal 0034 remains Blocked because exact ordering properties are
undocumented and ARGSORT/TOP_K conflict with current type/arity boundaries. Metal 0035 remains
Blocked because direct dropout hides randomness, opaque Philox state cannot represent exact Model
INT64[2] state, and current Metal lacks zero-input/multi-output/INT64 support. It requires an exact
custom kernel plus the complete schema/lifecycle; `p=0` narrowing is prohibited. Metal 0036 remains
Blocked because Conv3d inherits the opaque contraction boundary; average Pool3d lacks complete
mapping/divisor/route proof, while max Pool3d retains exact winner selection. Metal 0037 remains
Blocked because direct RNN lacks runtime INT64 lengths, atomic validation, skipped padding
semantics, and `finalHidden`; no complete recurrence/state/gradient route proof exists. No task may
infer generic fast math, transfer row relaxations, replace a prohibited matrix with narrowing, or
reserve conditional schema.

## Blocked, review-needed, and deferred work

- Metal 0006 remains historically `Blocked`. Its direct-selector gross special-class/sign failures
  remain outside Model 0030; a future conforming custom or composed `ACCELERATOR` route still needs
  complete formula, schema, capability, and route proof.
- Metal 0016 remains historically `Blocked` by `EXP`/`SIGMOID` FTZ and its frozen one-step gate.
  Complete Model 0030 makes those recursive `ACCELERATOR` results semantically reachable. Metal
  0051's sole recursive oracle is bounded regression evidence: direct EXP and composed SIGMOID
  passed the sample but are `DOMAIN-BLOCKED`, the other candidates failed, and that oracle is never
  rerun. Metal 0053 supersedes it for future work but is Blocked before source execution on concrete
  machine-checked complete-domain proof plus independent approval. A future nondominated
  multi-survivor result additionally requires a controlled environment; local timing is diagnostic.
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
  publication may choose `+0`. Complete Metal 0021, not 0018, is the implementation successor;
  its fresh full oracle, implementation/evidence chain, and independent Class C review passed.
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
  0017, Trace 0003, and Metal
  0015/0019/0020/0021/0022/0023/0024/0025/0038/0041/0042/0043/0044/0045/0046/0048/0049/0050/0054/
  0055 are Complete. Metal 0016–0018, 0026–0027, 0030–0037, planning-only 0039, failed-gate 0040,
  provider-gated 0047, historical consumed-oracle 0051, and proof/review-gated planning successor
  0053 remain Blocked under their recorded contracts. Task 0052 is Complete; Task 0054 remains its
  exact historical pre-cutover inventory. Task 0055 is the completed zero-new-capability
  foundation. No Metal task is Ready.
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

No Metal task is Ready. Task 0055 completed the private ABI-5/schema-13/six-carrier/cardinality/
typed-transfer/identity/package foundation at `ddeff1b2` with zero capability widening. Metal 0053
remains Blocked on its independent concrete-source, complete-domain proof and review; a controlled
environment is additionally required only if its future oracle leaves multiple nondominated
survivors. No local timing, operation route, oracle, or capability work is authorized by the
completed foundation.

Metal 0046 is Complete at independently approved implementation
`4aad1ab6ced318107e65bb9beef0013f8a7ff6e5`. Its two opt-in module-local Gradle tasks directly
verify an explicitly supplied absolute Task 0045 package and create one reproducible unversioned
local ZIP. Ordinary build, native build/signing, installation, discovery, cache, Java/runtime
behavior, publication, coordinates, Developer ID, notarization, provenance, and public release
remain outside that completed scope.

Metal 0047 is planning-only and Blocked. Its exact proposed workflow has one ordinary
`ubuntu-24.04` portable build and one `macos-26-xlarge` M2-GPU functional lane with immutable
action/JDK/Gradle pins, native build/ad-hoc-sign/package/verify/archive/extract/reverify, actual
no-skip Metal and Engine integration tests, and one report-only smoke without a strategy or
performance/context/executable matrix. The current personal public repository cannot schedule the
required larger runner; no workflow, compile-only replacement, cache, upload, or release path was
added.

Metal 0048 is Complete at independently approved implementation `89f9fbb9`. It removes
deterministic per-run splat allocation/fill/upload while preserving fresh run bindings and exact
result-lifetime cleanup, and explicitly rejects a general mutable output/workspace pool under the
current ownership contract.

Metal 0049 is Complete after P1 status-drift remediation `6d4246f7` and final cumulative
independent Class C `APPROVE` with zero findings. Its contract records the synchronous completion
barrier, completed-state `RunResult` resource/publication lease, and one system-default Metal device
context per integration without adding an asynchronous API, physical-device selector, multi-device
execution, identity change, cross-device behavior, or general output/workspace pool. Historical
Metal 0051 and planning-only successor 0053 preserve
those boundaries and own no production scope. Complete Task 0055 preserves the same synchronous,
single-default-device, direct-target lifecycle; no Metal task is Ready.

Metal 0050 is Complete. Its final program verification built the current dylib, applied the fixed
ad-hoc identifier, packaged and independently verified it, created and permission-preservingly
extracted the explicit local ZIP, reverified the extracted package, and used that absolute dylib for
the sole final full repository build. The build passed with 87/87 actionable tasks; all 130 Metal-
backend and 15 explicit Metal integration tests had zero failures, errors, or skips. No benchmark,
second full build, implementation change, or authenticated release was added.

Metal 0040 is Blocked from exact clean planning revision
`c300582727ec568a7482dd974f9d1b2e2e13f82c`. Its sole disposable direct BFLOAT16 Gather program
created one graph, compiled once, and executed once over the exact 20-word corpus and reverse INT32
permutation. Both inputs were unchanged, but output ordinal zero returned canonical quiet-NaN
`0x7fc0` instead of selected negative signaling-NaN `0xffa6`; later output and guard checks make no
success claim. The source and binary were removed. There was no retry, matrix, exhaustive corpus,
production/test/schema/identity change, native build, or full build. Unblocking requires a
separately planned exact replacement route, not a weaker direct-selector claim.

Metal 0030 remains Blocked after its only permitted direct `MAX_POOL2D` execution. The frozen
`[1,1,1,18] -> [1,1,1,9]` run returned negative zero `0x80000000` for `[+0,-0]`, where positive
zero was required, and finite `3.0f` for all first/later qNaN/sNaN windows; its other documented
controls passed and its artifacts were removed.

Metal 0031 is now Blocked without a device probe. The accelerator-only canonical FLOAT32 unbiased
no-grad Conv2d candidate has a complete documented NCHW/OIHW descriptor mapping, but MPSGraph does
not document or pin its shape-dependent contraction algorithm. One allowed-set execution cannot
authorize full Model geometry, lean policy prohibits a geometry/algorithm matrix, and narrowing to
one probe geometry is unsound. Unblocking requires an exact custom kernel or preceding
Model/architecture broadening. Average Pool2d, composed Pool1d/Conv1d, larger Pool3d/Conv3d, and
backward topology retain separate or inherited blockers and provide no smaller sound route.
Metal 0032 remains Blocked without a device probe under its frozen profile-common premise. The
smallest first-class candidate is same-typed FLOAT32 no-grad MSE `NONE`. Complete Model 0030 makes
its subtraction and `delta*delta` sites recursively reachable for `ACCELERATOR`, and
other reductions recurse through exact membership, sums, and divisors. MPSGraph has no direct MSE
selector, however, and no complete decomposition/custom route, structural schema/capability,
full-formula, generated-gradient, or complete-domain proof exists. Dense and index categorical
losses retain their own max/selection/bounds/ignore, exponential/logarithmic, and reduction
obligations. Unblocking requires that complete proof, not one sample or transfer from binary rows.
Metal 0033 remains Blocked without a device probe. Its smallest candidate is canonical FLOAT32
no-grad unmasked noncausal default-scale one-output SDPA. The legacy direct selector is
runtime-available, while the descriptor route requires macOS 27 and is unavailable on the current
macOS 26.6.2 runtime. Complete Model 0030 recursively reaches `ACCELERATOR` attention
arithmetic, but the direct/additive mask contract does not implement Model's BOOL selection or
BOOL-plus-causal AND. The opaque selector lacks complete-domain proof for query/key contractions,
explicit/default scale, guards and masks, softmax sites, special classes, value/output
contractions, generated gradients, and shape-dependent algorithms. Unblocking requires that
complete proof or a conforming custom/composed route.
Metal 0034 is now Blocked without a device probe. Its smallest candidate is profile-common
canonical rank-one FLOAT32 no-grad ascending SORT. Direct one-output SORT is runtime-available, but
ordering has no accelerator relaxation and Apple does not document stable ties, NaNs-last,
signed-zero order, ordinary subnormal comparison, exact represented-bit movement, or one behavior
across shape-dependent algorithms. ARGSORT's direct INT32 result mismatches Model's mandatory INT64
and the current Metal type/publication path; mandatory two-output TOP_K cannot fit the current
one-output schema and lacks Model's output-order guarantees. One run cannot close these independent
gaps. Unblocking requires an authoritative complete direct-selector contract or an exact custom
stable-sort kernel.
Metal 0035 is now Blocked without a device probe. Its smallest nondegenerate candidate is
profile-common exact canonical rank-one FLOAT32 no-grad INITIAL_STATE plus explicit-state DROPOUT
over every valid binary64 probability. Direct MPSGraph dropout hides randomness and returns only
the value; its opaque Philox state does not establish Model's exact INT64[2] key/counter input,
counter-plus-N output, mandatory mask/nextState, threshold, or special/scaling rules. Current Metal
also lacks zero-input nodes, multi-output nodes, and local INT64. A `p=0` identity route is not a
sound slice. Unblocking requires an exact Metal-private custom random kernel and complete
multi-output/local-INT64 schema, native lifecycle, and proof.
Metal 0036 remains Blocked without a device probe. Its smallest candidate is accelerator-only
unbiased no-grad canonical positive static rank-five FLOAT32 Conv3d. The macOS-13.2 direct selector
has a complete NCDHW/OIDHW groups/stride/dilation/explicit-padding mapping, but its contraction
algorithm and shape-selection boundary are undocumented; it therefore inherits Task 0031's
documentation-first blocker. Complete Model 0030 recursively reaches average Pool3d
arithmetic, but Pooling4D/stencil do not prove the complete NCDHW mapping, ceil-tail, fixed divisor,
or route. Max Pool3d retains exact winner-selection obligations, and Task 0030's signed-zero/NaN
evidence forbids generic MPSGraph max assumptions. UNFOLD3D/FOLD3D require separate custom
movement/overlap planning. Unblocking requires a conforming route with complete structural and
recursive-subset proof.
Metal 0037 remains Blocked without a device probe. Its smallest candidate is profile-common
canonical positive static FLOAT32 no-grad bias-free FORWARD RNN_TANH over the complete runtime
valid-length domain. Complete Model 0030 recursively reaches `ACCELERATOR` contractions,
additions, tanh/sigmoid sites, and state arithmetic. The macOS-12.3 direct selector nevertheless
has no INT64 valid-length input, atomic validation, skipped padded work, positive-zero padding
contract, or `finalHidden`; its optional training output has the wrong role. The opaque selector
lacks complete recurrence/state-publication proof, and GRU/LSTM add their own output-role/order and
gradient obligations. Unblocking requires a conforming custom or proved selector route plus the
complete five-input/two-output/caller-INT64 schema, native lifecycle, and proof.

Schema 13, operation wires `1..115`, attributes `0..41`, local types `1..6`, ABI 5, and
version-fourteen identities are landed by Complete Task 0055. Metal 0053 consumes that foundation
but remains Blocked before any future production cutover. Metal 0026/0027 remain separately
finalized Blocked. Documentation/audit-only Metal 0038
is Complete. Planning-only Metal 0039 is Blocked on Draft Model 0026. Metal 0040 is Blocked by its
failed one-execution BFLOAT16 raw-bit gate. Metal 0041 is Complete at implementation `ba16d942`
plus remediation `386705ca`; Metal 0042 is Complete at `9feb2505705263b6efb417d606678c606c2b9598`;
Metal 0043 is Complete at remediation `77e6091b`; Metal 0044 is Complete; Metal 0045 is Complete at
remediation `26c6c911`; Metal 0046 is Complete at independently approved implementation
`4aad1ab6`; Metal 0047 is Blocked; Metal 0048 is Complete at independently approved implementation
`89f9fbb9`; documentation-only Metal 0049 is Complete after remediation `6d4246f7`; Metal 0050
final verification, Task 0052, and historical documentation/audit-only Metal 0054 are Complete.
Historical Metal 0051 and successor Metal 0053 are Blocked. Task 0054 remains the exact pre-cutover
`19+2+15+79=115` record, not a current count; current post-cutover count is `34+2+79=115`.

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

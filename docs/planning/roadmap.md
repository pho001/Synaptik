# Implementation Roadmap

## Purpose and authority

This is the active index for area status, dependencies and authorized frontiers, not an
architecture contract. [ARCHITECTURE.md](../../ARCHITECTURE.md) and its incorporated scoped
contracts govern implementation. Read the owning master plan and compact Ready brief for scope;
completed evidence belongs in linked task records and Git history.

Engine0020 relocates backend sources without relocating their planning directories or completed
reports. Use the [current path migration table](../developer-guide/repository-layout.md#backend-path-migration-and-historical-navigation)
for source/build navigation; historical commands and hashes remain original-revision evidence.

## Execution policy

- Each independent workstream may expose one authorized Ready frontier when dependencies are met.
- Verify status, exact contracts, base, conflicts, write/semantic ownership, integration order and
  validation before launch. Use isolated worktrees for concurrent writers.
- Table order is priority, not a hidden dependency. Draft is not execution authorization.
- Classes B/C use clean implementation contexts and the required independent review. Reuse
  successful focused evidence; run one combined checkpoint after ordered integration.
- Follow the [planning guide](planning-guide.md); never derive architecture from old task prose.

## Completed checkpoint: repository review repairs

The user authorized [project review 2026-10-13](project-review-2026-10-13.md) against production
base `728680ea`. Its [finding report](project-review-2026-10-13-findings.md) distinguishes
reproduced/source-confirmed bugs, coverage regressions, planning drift, intentional limitations,
and unmeasured risks. All ten scoped repairs below are independently reviewed and Complete;
the final source-matched `test javadoc` checkpoint passed. G1's stable finite-softmax contract is
now Complete after a separate implementation, bounded v2 performance diagnostic and independent
review; the bounded vector ERF shape follow-up remains Draft.

| Workstream | Completed repair | Current state | Dependency / integration |
|---|---|---|---|
| Compiler | [0006D](modules/compiler/tasks/0006d-autograd-audit-correctness-repairs.md) | Complete | Correct batched matvec gradients and fail-closed derivative paths. |
| Prepare | [0009](modules/prepare/tasks/0009-complete-resource-rollback-on-duplicate.md) | Complete | Complete rollback of all returned unique resources. |
| Training | [0002C](extensions/training/tasks/0002c-adaptive-update-operation-order.md) | Complete | Preserve documented Adam arithmetic order. |
| NN/build | [0025B](extensions/nn/tasks/0025b-public-model-dependency-export.md) | Complete | Export existing public Model edge; isolated consumer validation. |
| Tuning | [0005](tools/tuning/tasks/0005-model-plan-cache-encoding-bound.md) | Complete | Preflight aggregate cache size before encoding allocation. |
| CPU | [0021](backends/cpu/tasks/0021-composed-value-validation-and-loss-bindings.md) | Complete | F16 softmax, produced-value validation and composed loss bindings. |
| CPU successor | [0022](backends/cpu/tasks/0022-activation-extremes-and-low-working-precision.md) | Complete | Activation extremes and low F32 working sites; schema 70; bounded performance limits retained. |
| Metal | [0078](backends/metal/tasks/0078-native-boundary-and-low-value-audit-repairs.md) | Complete | Native metadata/context and low exact-value repairs; source-matched device validation. |
| Metal successor | [0079](backends/metal/tasks/0079-scalar-provider-preflight-alignment.md) | Complete | Align advertised scalar RELU/MUL with Java/native admission and rank-zero lowering; identity 36. |
| Engine/tests | [0019](modules/engine/tasks/0019-lifecycle-and-mixed-owner-test-repair.md) | Complete | Deterministic copy/close, genuine CPU↔Metal tests and tuning Javadoc; final device checkpoint passed. |

Closed parallel group: `PROJECT-REVIEW-2026-10-13-REPAIRS`. Exact launch bases, scopes, reviews,
integration commits and validation are recorded in the briefs and report. No writer remains active.

## Area map

The area rows summarize status. Only an explicitly Ready brief at an authorized frontier permits
new execution. Training 0004 and bounded Checkpoint
[0004A](extensions/checkpoint/tasks/0004a-direct-leaf-model-save.md) are Complete after independent
review and configured integration gates.
[Checkpoint 0005](extensions/checkpoint/tasks/0005-training-state-reconstruction.md) is Complete:
defensive reconstruction, external-package tests, final Javadoc/example validation, and independent
API review passed. [Checkpoint 0005A](extensions/checkpoint/tasks/0005a-model-envelope-composition.md)
is Complete after envelope implementation, independent Class C API/documentation approval, and
configured coordinator gates. CP0005B is Complete (2026-10-15): final independent code/docs APPROVE,
57 module tests, final Javadoc, architecture evidence and 3 new integration tests passed.
Authorized Checkpoint source frontier: None; [CP0013 Complete](extensions/checkpoint/tasks/0013-general-scan-graph-persistence-decision.md#result)
delivers bounded v4 body persistence after final independent approval/configured gates. [0012 Complete](extensions/checkpoint/tasks/0012-general-scan-in-memory-copy-and-rebind.md#result)
delivers bounded in-memory copy/rebind with NN gate preservation and explicit root environment exception.
[Training0008](extensions/training/tasks/0008-fresh-value-training-session-runs.md) is Complete after
Euclid's definitive code/docs APPROVE and main's source-matched root gate (2026-10-19).
Authorized Training frontier: None; [NN0028](extensions/nn/tasks/0028-owned-model-compute-training-and-persistence.md)
is Complete, accepted 2026-10-19. [NN0029](extensions/nn/tasks/0029-fluent-model-builder-and-checkpoint-independent-execution.md)
is Complete after Planck final APPROVE, zero findings, and main's source-matched native root gate (2026-10-19), including independently reviewed [Compiler0009](modules/compiler/tasks/0009-static-select-and-slice-logical-layout-closure.md) and [CPU0025](backends/cpu/tasks/0025-scatter-validation-coordinate-rank.md). [CPU0026 diagnostic](backends/cpu/tasks/0026-stacked-lstm-cold-preparation-diagnostic.md), [CPU0027 unit topology fix](backends/cpu/tasks/0027-unit-topology-adjacency-ordering.md) and CPU0028/NN0021B bounded scan execution are also Complete. NN/Compiler frontiers: None; current CPU planning frontier is indexed below. CPU0030 bounded milestone and Compiler0011 completion are below. CPU0029 scalar coverage is Complete below. NN0022B is Complete.
Engine0020's completed layout cutover is below.
The user revised the primary product to automatic usable-model load and a choice of inference or
Training-owned resume, without routine callbacks/manual construction/digests/Arena.
[ADR 0038](../design/decisions/0038-declarative-model-and-training-resume.md#accepted-nn-decision) replaces
the withdrawn Proposed direction of ADR0037. Its NN and CP-only whole-model decisions are accepted;
NN0026 and CP0005B are Complete after final reviews/validation.
NN0027/CP0005C/Training0006/adapter0006 remain Draft as optional older proposals, not prerequisites of the new
[ADR0039 graph path](../design/decisions/0039-portable-graph-model-persistence.md).
[CP0005D planning/review](extensions/checkpoint/tasks/0005d-portable-graph-persistence-decision.md)
is Complete after independent review, the sole P2 correction and bounded coordinator acceptance.
The user approved the direction and physical layout on 2026-10-16. The
[model execution/refactor program](model-execution-refactor.md) records phased ownership, exact
target paths/edges and the independently approved P2 packet. Generic Model APIs, old v1 and flexible
declarative models stay supported; Engine0021 intentionally removed the thin session API with an
explicit source migration. All selected program phases and their bounded semantic cutovers are
Complete; unselected extensions remain separate future work.
[Engine0020 backend/native relocation](modules/engine/tasks/0020-backend-native-layout-relocation.md)
is Complete: separate :backends:* identities, packages, kernels and ABI preserved; independent
review, native build/package/device checks and the combined test/Javadoc checkpoint passed.
All five product choices and the prior P2 packet are approved. The user additionally approved
removing InferenceSession/Engine.session, keeping Engine PreparedExecution and adding Engine.runValues.
Engine0021 is Complete after independent Class C approval and the coordinator's native-enabled
test/Javadoc gate. Its brief owns evidence. Authorized Engine frontier: None.
[Training0007](extensions/training/tasks/0007-generic-live-parameter-admission.md) is Complete after
final independent APPROVE and the coordinator gate. Authorized Training frontier: None.
[Compiler0008](modules/compiler/tasks/0008-cold-source-capture-and-reconstruction.md) is Complete:
bounded source API/authority cutover, final independent code/docs APPROVE and coordinator Class C
checkpoint passed. Compiler0009 is locally accepted and Complete; at that checkpoint Compiler frontier was None. [Checkpoint0008](extensions/checkpoint/tasks/0008-generic-graph-value-persistence-and-nn-glue-cutover.md)
is Complete after Halley's definitive APPROVE and the coordinator's source-matched test/Javadoc
gate on 2026-10-17. Its brief owns evidence. Checkpoint0009 is Complete; frontier None. Training0008 is Complete
after Euclid's final APPROVE and main's root gate; NN0028 is Complete under the bounded NN authority.

| Area | Delivered baseline / remaining gate |
|---|---|
| [Model](modules/model/master-plan.md) | Model0046/0047/0048 empty-carry/explicit-mode, shared ordinary guards and per-output coordinate/direct-input semantics are Complete after independent final approval and actual affected/API/retained-byte proof. Typed AD/execution and cold/wire widening remain independent; original failures stay recorded. |
| [Trace](modules/trace/master-plan.md) | Envelope/IDs and preparation/invocation/low-route facts delivered; broader payloads/encoding Draft. No numerical-profile DTO. |
| [Backend Contract](modules/backend-contract/master-plan.md) | Complete through 0004; reopen only for a concrete shared need. |
| [Config](modules/config/master-plan.md) | Compile leaves and two-phase tuning policy delivered. Cost consumer/aggregate/run configuration Draft; no numerical-profile API. |
| [Planning](modules/planning/master-plan.md) | Complete0008 body-bearing query baseline. Current neutral ScanActivityPlan/fifth-query source is the coupled Compiler0017 Ready slice under accepted frozen authority/APIs, not new AD ownership/execution or cost-model work. |
| [Runtime](modules/runtime/master-plan.md) | Complete through0016 and Engine0022 closed-payload repair. [Engine0023 Complete](modules/engine/tasks/0023-reentrant-close-and-lease-admission.md#result) repairs reproduced lease-token allocation rollback and reentrant shutdown after independent final approval and unchanged actual OOM/GPU consumers; no new lease protocol. |
| [Compiler](modules/compiler/master-plan.md) | Capture/AD/binding,0018 cold,0017A/B lexical/shared checked construction and0019 RESHAPE are Complete. General Scan0017 composes ordinary selected-proof/activity semantics withCPU0081. Conv3d0006C is Complete after CPU F32/F64 and Metal F32 numerical/train/persist proof, independent reviews, root test/build/distribution and fresh Maven/no-Git-source consumers. |
| [Prepare](modules/prepare/master-plan.md) | Complete0011–0015 includes accepted static/named-B F32/F64 length projection/execution in its recorded domain. Current neutral activity/context-copy/typed projection is the coupled CPU0081 Ready slice under accepted APIs and exact common source; no new executor or implemented wider capability. |
| [OpenBLAS provider](backends/openblas-provider/master-plan.md) | 0001–0003 Complete. Optional direct BF16 0004 Blocked on ABI and one-final-narrowing evidence; F32/F64 unaffected. |
| [CPU](backends/cpu/master-plan.md) | CPU0077/0078/0079/0080 finite-softmax, retained low lock, raw RESHAPE and same-class emission are Complete. General Scan0081 composes ordinary units under Compiler0017. Coupled Conv3d0006C is Complete: F32/F64 training and full-body Scan forward/reverse, raw four-floating-carrier UNFOLD3D and F32/F64 FOLD3D passed final root and installed/source consumers. Ordinary named-B and low-policy boundaries are unchanged. |
| [Engine](modules/engine/master-plan.md) | Engine0021/0022/0023/0024 Complete in recorded scope: lifecycle/OOM/GPU fixes and standard18-product binary/source/Javadoc plus exact-source distribution mechanism. Clean extracted Maven/Gradle consumers train CPU/Metal and persist. Owner selected private/internal distribution without a public licence/upload; final all-PURE artifact freeze and technical acceptance remain. |
| [Metal](backends/metal/master-plan.md) | 0080/0081 identity38 composition and0082 host-copy lifetime repair are Complete in recorded scope. Current identity39 qualifies FLOAT32 overlapping image folds and Conv3d0006C training after source-matched build/package, Metal-only numerical/train/save/load/new-train, independent review, root gate and fresh artifact/source consumers. Hosted GPU CI0047 remains blocked; no official trusted prebuilt release. |
| [CUDA](backends/cuda/master-plan.md) | Marker/native scaffold only; no executable CUDA backend. 0001 Draft. |
| [ONNX](extensions/onnx/master-plan.md) | Marker only; no importer/exporter. Area Draft; first mapping task not yet defined. |
| [Data](extensions/data/master-plan.md) | 0001–0003 Complete; [0003 shared padded representation and INT64 batching](extensions/data/tasks/0003-padded-integral-sequence-batching.md) accepted after final independent APPROVE and module/architecture-consumer gates, 2026-10-20; Text program root join complete, 2026-10-21. Frontier None; broader batching Draft, no runtime binding. |
| [NN](extensions/nn/master-plan.md) | Existing owned/static/fixed baseline and NN0031 Complete. [NN0033 runtime-length RNN](extensions/nn/tasks/0033-trainable-runtime-length-rnn-workflow.md#result) Complete after independent reviews, 620 affected tests/Javadocs and public F32/F64 train/save/close/load/new-train proof. NN frontier None; NN0032 ordinary .rnn migration remains Draft, with no higher-order waiver. Retained fixed/GRU/LSTM/low routes and mixed-source rejection unchanged. |
| [Text](extensions/text/master-plan.md) | [0001 frozen whitespace tokenizer and Data adapter](extensions/text/tasks/0001-whitespace-tokenizer-and-data-batching.md) Complete, accepted 2026-10-21 after independent code/final documentation APPROVE and numerical/root gates. Sole api(Data), Model transitive. Frontier None; training/artifact work Draft. |
| [Vision](extensions/vision/master-plan.md) | No module; frontier None. Separate Vision architecture approval before decoder/preprocessing work; later numeric batching retains its Data gate. Data0001 grants no Vision edge. |
| [Training](extensions/training/master-plan.md) | Sessions/optimizers/state, [0007 generic live admission](extensions/training/tasks/0007-generic-live-parameter-admission.md) and [0008 fresh-value runs](extensions/training/tasks/0008-fresh-value-training-session-runs.md) Complete; final independent reviews and coordinator gates passed. Authorized Training frontier: None; NN0028 Complete; 0006/adapter held. |
| [Checkpoint](extensions/checkpoint/master-plan.md) | CP0013 v4, inherited static length copy/rebind and [CP0014 v5](extensions/checkpoint/tasks/0014-length-aware-general-scan-v5.md#result) Complete. Independent final review, 619 affected tests/Javadocs and loaded F32/F64 Model numerical proof passed. Source frontier None; NN0033 owns recurrent composition. |
| [Benchmarks](tools/benchmarks/master-plan.md) | Fixed CPU/Metal report-only harnesses delivered; broader workload suites Draft. No universal performance claim. |
| [Tuning](tools/tuning/master-plan.md) | Local/complete-plan tuning and read-only inspection delivered, plus cache-bound repair. |
| [CLI](tools/cli/master-plan.md) | Marker only; commands await selected concrete Engine/diagnostic workflow. |

## Bounded activation-source successor

The user authorized implementation of [Checkpoint0009](extensions/checkpoint/tasks/0009-owned-model-activation-source-and-persistence.md)
on 2026-10-19. Compiler0008 + Checkpoint0008 + Training0008 + NN0028 (all Complete) fed this
completed capability. The selected model-execution program remains Complete; NN's ownership,
static signatures and prepared execution remain unchanged. Status: Complete, accepted after
Archimedes final Astra/xhigh APPROVE (zero findings) and main's integration/native root gates.

One atomic Class C packet adds RELU/TANH/SIGMOID to the existing five source kinds, covering owned
Model construction as well as compute/train/save/load. Compiler-only widening required an atomic
cutover because the former encoder defaulted unknown kinds to PERMUTE. Accepted format: graph-model.v2 with
new activation wires, while five-kind saves/readers retain graph-model.v1 and all older formats remain.
The brief owns exact wires, direct-oracle multilayer-perceptron acceptance, bounded contract cutover
before Java and validation. The eight-kind/selective-v2 authority and code are current and accepted.
Common base: main `7e880fb1044168043e45b10419a570d64c5283ae`; Parallel group None, shared contracts
serialized, main owns shared documents and the completed gates. Checkpoint/Compiler/NN frontiers:
None at CP0009 completion. No CP0009 follow-up or reopened program; the later Compiler0009 repair below does not widen source admission.

## Completed fluent Model builder and prerequisite join

User-approved [NN0029](extensions/nn/tasks/0029-fluent-model-builder-and-checkpoint-independent-execution.md)
depends on completed NN0028 + CP0009 + [CPU0025](backends/cpu/tasks/0025-scatter-validation-coordinate-rank.md)
and [Compiler0009](modules/compiler/tasks/0009-static-select-and-slice-logical-layout-closure.md), also Complete and locally accepted after final independent APPROVE, zero findings.
CPU0025 is locally accepted after independent APPROVE, zero
findings (2026-10-19), with no NN dependency; CPU0023 remains Draft.
DAG: Compiler0006B11 + Compiler0008 + CPU0025 → Compiler0009 (all Complete) →
NN0029 → main's single shared native root checkpoint: all Complete; at that checkpoint NN/Compiler frontiers None.
User approved bounded exact logical-view inference after MATMUL/CONCAT: SELECT uses IndexingInference,
coordinate SLICE uses LayoutInference. No forced NN contiguous copies or scope reduction; existing
contracts suffice, with negative-step/empty/unproved layouts unchanged. No extra plan-review cycle.
Main verified NN resume on 2026-10-19: actual brief, user approval, unchanged authority, Complete
predecessors/reviews, inactive conflicts/no other writer and unchanged full dirty base at
`7e880fb1044168043e45b10419a570d64c5283ae`, including accepted CP0009, frozen NN partial and
completed CPU0025/Compiler0009. Main owns shared documents and the combined gate.
Final NN evidence: 22 focused, 424 NN and 3 Fluent integration tests passed with zero skips;
Planck final APPROVE, zero findings. CPU/Compiler final approvals and local evidence remain valid.
Main's native root command passed, exit 0/5m14s: 107 tasks, 18 executed/89 UP-TO-DATE.
Current XML: 4279 tests, zero failures/errors, 41 skips; includes UP-TO-DATE evidence, not all newly run.
The 2291-file reviewed snapshot/native hash were unchanged; both configured Metal training cases
executed without failure/skip. Full result and skip breakdown are in the NN0029 brief.
One public ModelBuilder composes existing layers;
private leaf acquisition and retained ordinary cold composition remove construction/execution's
serialization limit without widening graph-model.v1/v2 or adding another graph/execution pipeline.
Main accepted the root/owning NN authority cutover before the existing partial implementation.
Compiler0009 needs no further authority cutover. Parallel group None; main owns serial integration/shared documents and one final
native root gate, now passed after final implementation/review. Exact evidence lives in the brief.
Scan BPTT, Conv3d gradients, backend/codec changes, Compiler work beyond 0009 and held recurrent-persistence plans stay outside scope.

## Bounded CPU cold preparation diagnostic

User-authorized [CPU0026](backends/cpu/tasks/0026-stacked-lstm-cold-preparation-diagnostic.md) is Complete
after Pasteur's binding APPROVE, zero findings, and main acceptance (2026-10-20). CPU0025 +
Compiler0009 + NN0029 → CPU0026 → CPU0027 are Complete; at that checkpoint CPU frontier None, CPU0023 Draft.
The reconstructed three-step LSTM probe attributes sustained sampled CPU analysis to repeated
finish/topological scans; combined prepare was censored at the 180-second JVM deadline.
Two launches yielded one valid measurement; setup failure consumed the sole follow-up.
The brief owns timings, recording coverage and limits; no cached-training or regression claim.
The user-authorized CPU-only Class C fix [CPU0027](backends/cpu/tasks/0027-unit-topology-adjacency-ordering.md)
is Complete after invocation-local adjacency, stable Kahn ordering and reused dependencies;
independent code/docs APPROVE, CPU module/Javadoc, focused conformance/Fluent integration and the
same bounded public probe passed. Main accepted the unchanged reviewed source at base
`2090e0a6e32485756a3dfd0d545b5dbce523cfe6` (2026-10-20); Parallel group None, no new frontier.
The brief owns timings and single-probe limits; no root suite or benchmark campaign was added.
The [CP0010 CNN/static-recurrent packet](extensions/checkpoint/tasks/0010-fluent-cnn-and-static-recurrent-graph-persistence.md)
and [CP0011 flat-scan supplement](extensions/checkpoint/tasks/0011-flat-recurrent-scan-persistence.md)
are jointly Complete on 2026-10-20: bounded CNN/static recurrence and multi-output v3 cold scan
persistence accepted after independent APPROVE and main's focused/native root gates;
[CP0010 Result](extensions/checkpoint/tasks/0010-fluent-cnn-and-static-recurrent-graph-persistence.md#result) owns evidence.
Completed Compiler0008/0009 + Checkpoint0008/0009 + Training0008 + NN0028/0029 and
Model0025E/0025F/Compiler0006A remain the predecessor DAG, on base
`8d5357e9a8698721a2252cb2ca3cb94150e705e8`. At that checkpoint, Checkpoint/Compiler/NN frontiers: None;
no successor authorized by that persistence checkpoint. It grants no execution/BPTT or optimizer
resume; the subsequently completed CPU0028 route is recorded below.
Parallel group None; main owns serialized integration and shared documents.

## Completed CPU recurrent scan execution

[CPU0028](backends/cpu/tasks/0028-fixed-recurrent-scan-forward-execution.md) and
[NN0021B](extensions/nn/tasks/0021b-fixed-recurrent-scan-implementation-program.md) are Complete,
accepted 2026-10-20 after final independent code/docs APPROVE and main's root test/Javadoc gate.
They deliver canonical FLOAT32 scalar Java RNN/GRU/LSTM, both directions/bias forms, valid gradient
flags with exact output OR, runtime INT64 lengths and all outputs on heap/native/mixed storage.
Full length preflight follows producers and precedes any scan write; scratch is run-isolated.
DAG: NN0021A + Model0025E/0025F + Compiler0006A + CPU0027 + Engine0021 + CP0010/CP0011
→ CPU0028 → NN0021B, all Complete. Parallel group None; main owns shared-document integration.
Task Results own exact evidence and base metadata. At that checkpoint CPU/NN frontiers: None. Data/NN migration, BPTT, FLOAT64/low types, noncontiguous layouts, vector/parallel/native
scan routes, tuning and performance claims remain outside scope; no public API, wire or authority change.

The completed NN-only chain is NN0022 (runtime-Tensor-length forward expressions) →
[NN0022B](extensions/nn/tasks/0022b-runtime-length-scan-model-workflow.md) (Model/builder inference),
then [Complete NN0022A](extensions/nn/tasks/0022a-padded-numeric-sequence-inference.md), delivering
RIGHT/TIME_MAJOR numeric F32 inference. Data0001–0003 and
NN0022/0022B/0028/0029 are Complete; the exact Data export and narrow contract cutover are approved
(2026-10-21, ADR0044). Base is HEAD=origin/main `6eabea0c1ebf77b9bd8382855c7131eb89b52e04`.
The brief records approval, inactive conflicts and all seven integration fields. Serial clean
implementation and independent Class C code/docs APPROVE are complete; worker gates pass.
Main accepted after the source-matched architecture/root checkpoint on 2026-10-21. No NN-to-Data edge
or new owner. Fluent Text requires separate builder support.
The [completed brief](extensions/nn/tasks/0022-runtime-length-recurrent-nn-api.md) accepts existing
Sequence/cell ownership and Model result carriers with descriptor-only atomic initialization;
no new production type/package, runner/session, dependency or authority change. Existing list Model
inputs already represent sequence/lengths/states; direct public Engine is the execution boundary,
with Model.compute/train widening outside NN0022's scope. Main owns serial review/integration/shared documents
at base `2d1c1d6d0867a0bebda34e9177ad52ba9f13071a`, parallel group None. NN0023/0024 retain later
Data scope. NN0022 is Complete on 2026-10-20 after clean implementation, Mencius final code/docs
APPROVE with zero findings, and coordinator validation; its brief owns exact evidence and limitations.
NN/Data/Text frontiers: None; CPU frontier is indexed in the CPU row. NN0022A is closed; CPU0029 Complete below.
NN0023/0024 remain Draft; no automatic successor launch.
NN0022 reused existing backend/root evidence; its completion added no root/conformance/performance campaign.

NN0022B is Complete: scan recipes/finalHidden, typed buildWithLengths and generic
two-input/one-output compute convenience, including after loadGraph, reusing Engine.prepare/runValues. No new
production type or execution owner. The user approved the exact API and limited inference-only
root/owning-NN cutover on 2026-10-20; root, scoped contract, explanation and ADR0041 preceded
implementation. Final independent contract/code/docs approvals and main's source-matched root
test/Javadoc gate passed. Parallel group None; main owns serial integration/shared documents. Base is HEAD/origin/main
`2d1c1d6d0867a0bebda34e9177ad52ba9f13071a` plus accepted uncommitted NN0022, not a merged predecessor;
the brief owns exact commands/counts, reviews and hashes, including cached root tasks and the unrelated
integration assumption skip. Execution remains inference-only/canonical FLOAT32 CPU; no BPTT, Data,
dtype, backend or format widening. No capability follow-up or automatic successor launch.

## Completed CPU scalar scan coverage expansion

User-selected [CPU0029](backends/cpu/tasks/0029-recurrent-scan-scalar-type-and-layout-coverage.md)
is Complete, accepted 2026-10-21 after Raman's final independent Class C code/docs APPROVE,
zero P1/P2, and main's source-matched native-enabled root test/Javadoc PASS.
DAG: CPU0022 + CPU0028 → CPU0029, all Complete. At that checkpoint CPU frontier None; no automatic successor launch.
Base `f918d7d7dab3fff0b144747f56d629733bd443c7`, verified 2026-10-21; Parallel group None,
scan/numerical/ledger/shared-document conflicts serialized, main owns integration/shared documents.
Clean implementation, final review and the single root checkpoint are complete; source remained
frozen. The brief owns exact metadata, commands, counts and source/protected-hash evidence.
Delivered scope is direct scalar FLOAT32/FLOAT64/BFLOAT16/FLOAT16 and representable
positive/zero-stride/offset input reads with canonical outputs; low work/accumulation stays FLOAT32
with final narrowing per published value and exact stored recurrent state. Bounded timing is
diagnostic only: parity/speed remain unproven with unresolved JIT/preflight confounds, not a new
mandatory task. No BPTT, Metal, public API, NN executable/workflow,
Engine orchestration, tuning or generic execution abstraction. CPU0023 remains Draft.

## Current priority and next design decision

The user-agreed [production release goal](production-release-goal.md) defines equal Java CPU
F32/F64 support, trainable MLP/CNN/RNN, general Scan and required production-supported Metal in v1.
Java CPU BF16/FP16 computation code is retained without expansion and its numerical capability is
temporarily locked by Complete CPU0078; exact/raw/storage/casts remain independently available.
Implementation/release readiness remains incomplete. The
[reviewed support matrix](../backend-guide/production-support.md) explicitly selects CNN1d/2d
training and retained bounded 3D forward-only; Metal's required release role is no longer open.

[Compiler0018](modules/compiler/tasks/0018-static-length-scan-cold-reconstruction.md#result) is
Complete after independent review, the eight-case public cold smoke and the final affected
Compiler/Checkpoint/NN test/Javadoc gate. [Checkpoint0014 v5](extensions/checkpoint/tasks/0014-length-aware-general-scan-v5.md#result)
is Complete after independent final review, 619 affected tests/Javadocs and loaded F32/F64 Model
proof. [NN0033](extensions/nn/tasks/0033-trainable-runtime-length-rnn-workflow.md#result) is Complete
after two independent review slices, 620 affected tests/Javadocs and public F32/F64 RNN proof.
The shared source/coordinate/formula/emission prerequisites and Engine0023 lifecycle repair are
Complete. Engine0024's18-product distribution mechanism and clean installed CPU/Metal workflows
are qualified, including original full F32/F64 CNN/RNN numerical/state oracles. Current work is
the coordinated full general-Scan typed selected-proof/activity/ordinary-unit cutover, not another
CPU prerequisite or ReLU-only subset. MAIN owns exact source integration and final acceptance.
Ordinary static-unrolled `.rnn` is not migrated; NN0032's second-order blocker remains. GRU/LSTM,
low fixed paths and higher-order/SIMD/performance work are outside this practical scope.
[Compiler0017](modules/compiler/tasks/0017-per-value-typed-length-scan-inference-and-ad.md) and
[CPU0081](backends/cpu/tasks/0081-ordinary-unit-general-scan-composition.md) are coherent full-target
Ready tasks after both independent authority/interface APPROVE0 reviews and MAIN's coordinated
root/scoped acceptance. Exact dirty-source/disjoint contexts and integration metadata govern launch.
[Compiler0017A Complete](modules/compiler/tasks/0017a-ordinary-lexical-primal-binding.md#result)
completes existing ordinary primal binding under unchanged authority/eligibility after independent
source review, affected gates and actual lexical/ordinary numerical proof. No new Scan admission.
The earlier Metal0080/0081 identity38 checkpoint passed unchanged real typed-backward and full F32
MLP/CNN GPU proof; its combined v2 preserved all290previous rows and added16original-typed witnesses.
Current identity39 adds the coupled Conv3d training closure below. oneDNN stays later work,
not a v1 build/performance grant.

[Compiler0006C](modules/compiler/master-plan.md) is Complete, coupled to the existing
[CPU](backends/cpu/master-plan.md) generated movement/fold owners
and [Metal](backends/metal/master-plan.md) FLOAT32 image folds. One grouped convolution formula
handles Conv2d/Conv3d direct and checked metadata construction, selected targets and ordinary
two-stage composition. First-order coverage is39/112/134; only fixed RNN/GRU/LSTM remain deferred.
Actual programs passed independent every-role finite differences for CPU F32/F64 and Metal-only
F32, including grouped anisotropic/overlap and depthwise geometry, three training steps, bit-exact
save/close/load and a fresh oracle-checked SGD step. CPU general Scan passed complete Conv3d
forward/reverse finite differences in both types, all-valid/length-aware modes and both directions.
Independent CPU/AD and Metal source reviews returned APPROVE0. MAIN's full root test/build/
distribution gate and fresh Maven/no-Git-source runtime consumers passed.
Metal identity39/schema20/ABI7 is source-matched freshly built, ad-hoc signed and package verified.
The initial reserved-identifier shader compile failure was corrected by source rename and a
reauthenticated rebuild before passing Metal execution; this is not hosted CI or a released prebuilt.
Ordinary public CPU named-B Conv3d, the CPU low-arithmetic lock, Pool3d's no-gradient CPU gate,
Metal F64/low/FOLD_AXIS overlap restrictions, Metal Scan and all cold/wire inventories remain
unchanged. No new op, dependency, ownership or ABI shape, ADR or performance promise is introduced.

Model source frontier: [0046 Ready](modules/model/tasks/0046-empty-carry-scan-and-explicit-length-mode.md).
Its accepted empty-carry/explicit-mode source direction has an authenticated frozen33-file
implementation under independent source/wire review; no code execution or MAIN integration is
claimed yet. CPU0078 is Complete. Model0045 retains source-only acceptance and its original
failed evidence; no clean root PASS or historical retry follows. Existing downstream/cold/wire
domains remain independently bounded until later explicit qualification.

[CPU0048 proven static typed-one count folding](backends/cpu/tasks/0048-static-typed-one-count-folding.md)
is Complete, MAIN accepted 2026-10-31 after independent source/docs APPROVE, zero unresolved
P1/P2, final filtered CPU/Javadoc, named conformance/Engine join and conservation PASS.
Complete CPU0047/accepted CPU0076 remain its dependencies; the brief retains exact
seams/contracts, seven integration fields, commands and final evidence.
[CPU0049 Complete](backends/cpu/tasks/0049-producer-reduction-fusion.md#result), MAIN accepted
2026-10-31: static F32/F64 scalar SUM/MEAN producer fusion and ADD/RELU tails, independent FINAL
APPROVE, filtered CPU/Javadoc and named join PASS. The Result owns scope, preservation, recovered
gate evidence and the sole nonstationary ordinary diagnostic; no steady-profitability/parity claim.
[CPU0050](backends/cpu/tasks/0050-average-pool2d-backward-broadcast-fold.md#result) Complete, MAIN accepted
2026-10-31 after Dirac FINAL source/docs/protocol/result/gate-evidence APPROVE (0 P1/P2) and all gates PASS.
Static F32/F64 SCALAR/single-range compact-source fold reuses existing IR/Geometry boolean/emitter,
retains stored DIV and removes private expanded buffers; no new production type or authority change.
Closed order: freeze/independent review → MAIN named conformance/Engine join once (reused) → sole
diagnostic → final filtered CPU/Javadoc → MAIN acceptance. The Result owns exact evidence, protections
and limitations; 15-Java-file + guide freeze, old literal artifacts, HEAD and empty index retained.
Parity/cause remain unresolved; no steady-state/C2 or speedup claim. BF16/F16 remain untouched
compatibility controls, not numerical qualification; its separate repair is now
[CPU0050A Complete](backends/cpu/tasks/0050a-low-fold-fp32-accumulation.md#result), MAIN accepted
2026-10-31 after Raman FINAL APPROVE, 0 open P1/P2, and qualified gate closure.
[CPU0051 Complete](backends/cpu/tasks/0051-softmax-logsoftmax-backward-fusion.md#result), MAIN accepted
2026-10-31 after independent FINAL APPROVE, filtered CPU/Javadoc and qualified consumer closure.
Static F32/F64 scalar two-pass backward fusion preserves exact SUM and typed sites using two private
types and existing lowering/resource/cache owners. The Result preserves the first failed join,
independently reviewed test-only correction and targeted closure with unchanged-success reuse.
The sole short synthetic diagnostic verified outputs but did not establish performance parity,
profitability or a JIT cause. Verified launch, matrix, seven integration fields and evidence remain
in the brief. Parallel group None; MAIN owns shared integration. CPU frontier None; no further timing,
automatic successor or Git publication. Serial DAG: Complete CPU0050 → Complete CPU0050A → Complete
CPU0051 → held Draft CPU0052.
CPU0050A restores existing Model low FP32-accumulation/final-narrowing authority using existing owners,
not a new compact-low capability or architecture decision. Its brief owns exact contracts, seven
integration fields and verified actual-before `/private/tmp/synaptik-cpu0050a.yqY321/before.json`
(2596 files at HEAD `07a281ed` plus accepted dirty series, empty index) and source archive.
Historical Ready verification/MAIN-LAUNCH proof retained; no parallel repo writers. MAIN owns integration:
focused evidence/pre-final review → MAIN named join once PASS → failed CPU gate → approved metadata/
environment recovery → targeted recovery → metadata/Javadoc closure → final read-only review/acceptance.
The Result records original full FAILED, targeted closure and unchanged-outcome reuse, never a full-green
claim. Recovery updates only test-derived hashes/provenance. Generic traversal is retained under explicit
technical deviation evidence, not whole-loop/dataflow/performance parity; geometry hoisting remains a
known follow-up without automatic task/campaign. CPU0051 has the separate admission above; CPU0052 held,
Scan budgets paused/closed; this numeric correctness fix grants no new timing,
historical/broad campaign, automatic successor or stage/commit/push. MAIN owns shared integration.
CPU0049's sole private aggregate IR adds no authority/semantic change.

[CPU0076](backends/cpu/tasks/0076-row-invariant-load-staging.md#result) is Complete following
Complete CPU0073/0074/0075. The implemented
[CPU exception](../architecture/contracts/backend-execution.md#cpu0076-prospective-invariant-external-cell-load-staging)
permits generic scalar external-cell staging, not carry promotion or an RNN special case.
Authority/packet → independent review/MAIN GO → clean implementation/untimed freeze → source/
protocol review/MAIN sole finite release → bounded-result acceptance → final CPU/Javadoc/named
Engine gates and independent documentation are complete. The brief owns frozen hashes, exact
evidence, complete-body/lifetime/rank fallback and timing qualifications, not universal speed or
Java parity. Parallel group None; MAIN owns integration with exclusive final docs delegation.
At CPU0076 acceptance the source/Ready frontier was None; no further measurement or automatic successor.
The later CPU0048–0051 admissions/completions are above; CPU0052 remains held;
defaults unchanged, no automatic launch or rollout.

[CPU0075](backends/cpu/tasks/0075-row-fusion-scalar-load-address-reuse.md#result) is Complete after
MAIN's final CPU/Javadoc PASS and independent code/Javadoc/docs review. Complete CPU0073/0074
→ scalar emission/cache cutover → sole bounded A/B/C → owning gate/review is closed; CPU frontier
None at that completion. The Result records structural savings, unmatched ordinary JIT states, no material Scan
speedup, and remaining Java/C1 gaps. Unaffected controls, mathematics/API/ABI, exact SUM and
production defaults remain unchanged. That completion itself authorized no successor, fresh budget,
rollout or historical campaign; CPU0048–0052 remained held/inactive at that completion.

[CPU0074 row-fusion SIMD/parallel](backends/cpu/tasks/0074-row-fusion-simd-parallel.md) is
Complete and MAIN accepted, 2026-10-30. Its Result centralizes final CPU/Javadoc, named integration,
retention and the sole bounded comparison. Functional support is complete; nonstationary timings
and concrete JIT compilation pressure leave steady-state profitability unqualified. Separate
investigation is recommended, not a new task/frontier/campaign. The CPU0074 source/measurement
frontier is closed and budget exhausted; no retry or historical campaign opens. CPU master/brief
retain integration metadata; guarded ordinary-row scratch changes no pointwise rule or scoring.
CPU0048–0052 stayed held/inactive at that acceptance; scalar/max1 defaults remain, no rollout or automatic successor.
Earlier closed-frontier statements below remain historical, not automatic successor permission.

[Prepare0015 symbolic-length Scan CPU vertical](modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result)
is Complete, MAIN accepted 2026-10-27 after independent code/docs APPROVE (0 P1/P2), owning validation,
one bounded timing campaign and the configured root test/Javadoc PASS. Complete Model0044 +
Compiler0015 + CPU0042/Prepare0014 and retained predecessors feed the delivered fixed-T/features,
named-B + INT64[B] length F32/F64 forward/actual reverse capability, including bounded typed outer
prerequisites. Active Prepare/Engine frontiers: None; CPU0044's bounded repair is Complete below.
[CPU0045](backends/cpu/tasks/0045-symbolic-length-scan-equivalent-java.md#result) is Complete,
accepted by MAIN on 2026-10-28 after source/evidence reviews and actual documentation-diff checks.
It delivered an equivalent Java reference and bounded v5 offset sharing, not a demonstrated speedup
or parity. All prior measurement budgets are closed.

The user approved structural repairs first: [Compiler0016](modules/compiler/tasks/0016-matmul-gradient-noop-reduction-elision.md)
removes redundant MATMUL-gradient shape copies, then CPU0046–0052 cover cache cleanup, private
PERMUTE/MATMUL absorption, proven counts, producer/reduction and epilogue fusion, AveragePool2d
backward, Softmax/LogSoftmax backward, and private recurrent-gate slices. The
[CPU master](backends/cpu/master-plan.md#active-dependency-dag-and-integration) owns the serial queue.
Compiler0016, CPU0046 cache cleanup and
[CPU0047 private transpose/MATMUL reads](backends/cpu/tasks/0047-private-permute-matmul-absorption.md#result)
are Complete. CPU0048 is now Complete under MAIN acceptance above. The user's 2026-10-28 request selects
[CPU0053 fresh Scan/Java measurement](backends/cpu/tasks/0053-scan-java-continuation-measurement.md)
after publication of only CPU0047. The user subsequently stopped CPU0053 after its untimed
historical-class-pin mismatch; no measured body started and no retry is authorized. CPU0049–0052
remain Draft; the later CPU0048 Ready packet does not reopen measurement.

The user's subsequent verification request is Complete as
[CPU0054](backends/cpu/tasks/0054-scan-classfile-and-abi-verification.md#result): actual ClassFiles
differ only in their generated name; the causal identity control, historical ABI and tested raw
current/reference outputs passed. MAIN accepted after independent APPROVE on 2026-10-28. This
untimed diagnostic did not resume CPU0053 or establish performance parity.
The subsequent user continuation delivered the distinct finite
[CPU0055 measurement](backends/cpu/tasks/0055-verified-scan-java-measurement.md#result), Complete
after independent actual-result APPROVE and MAIN acceptance on 2026-10-28. One preflight and three
serial uncensored JVMs retained verified ClassFile/ABI/raw outputs; the narrow gap remains.
The wrapper false-negative and evidence-only correction are preserved, not a repeated preflight.
At CPU0055 acceptance source/observational frontiers were None; old budgets and CPU0053 stay closed.
The user's subsequent approval delivered [CPU0056](backends/cpu/tasks/0056-scan-invocation-binding.md#result),
Complete after independent final APPROVE and MAIN acceptance on 2026-10-28. Hot/cold comparison
rejected per-run typed-handle insertion; exact withdrawal retains the original invocation and all
accepted source. No architecture change, native rebuild, commit or push occurred; frontiers None.
The user-authorized [CPU0057 offline audit](backends/cpu/tasks/0057-scan-body-java-offline-audit.md#result)
is Complete after independent APPROVE and MAIN acceptance on 2026-10-28. Its
[combined report](backends/cpu/0057-scan-body-java-audit.md) compares stages and existing JIT evidence,
identifies bounded address-code opportunities and explicitly leaves native timing causality unproved.
At CPU0057 acceptance all frontiers were None; that audit made no source change or new measurement.
The subsequent user-approved [CPU0058](backends/cpu/tasks/0058-symbolic-length-scan-zero-scratch-base.md#result)
is Complete: coefficient-proved zero scratch-base omission and combined baked v6 compatibility.
MAIN accepted independent code/docs APPROVE and owning/named validation on 2026-10-28.
At CPU0058 acceptance all frontiers closed; that repair made no new measurement or publication.
The user-approved [CPU0059](backends/cpu/tasks/0059-scan-v6-v5-java-measurement.md#result) is Complete:
MAIN accepted independent actual-result APPROVE (0 P1/P2), repaired preflight and three measured JVMs.
No consistent v6 speedup is demonstrated; local Java gap/full-path JIT limits remain. All slots consumed;
original failed preflight/evidence unchanged. That completion authorized no automatic successor or further measurement.
The subsequent [CPU0060 loop-shape fix](backends/cpu/tasks/0060-symbolic-length-scan-java-loop-shape.md#result)
is Complete, MAIN accepted 2026-10-28 after independent code/docs/prelaunch/result approval and
owning/named gates. It aligns generic combined F32/F64 loops and baked v7 with unchanged semantics
and noncombined bytes. Its sole diagnostic observed modest improvement, retaining the Java gap and
full-path JIT limits; no automatic successor or further measurement followed that acceptance.
The subsequent [CPU0061 experiment](backends/cpu/tasks/0061-scan-feature-contraction-addresses.md#result)
is closed, with implementation Cancelled: one feature-K address candidate failed the frozen benefit
gate and was restored exactly to accepted v7. No functional suites, retry or retained capability;
at that acceptance all CPU frontiers were None. This distinct grant is consumed; older budgets stay closed.
The user-authorized [CPU0062](backends/cpu/tasks/0062-decompiled-scan-java-lab.md#result) is Complete after MAIN prelaunch/result approval: two matched JVMs pass all raw checks, but no manual prototype beats G7. Compiled-path readiness is supported with stated limits; no generator port or functional suite. Both slots/frontier closed; further source exploration needs its own brief, not an old-budget retry.
[CPU0063](backends/cpu/tasks/0063-clean-scan-java-generated-abi.md) completed reviewed whole-loop Java alignment. Separately authorized [CPU0064](backends/cpu/tasks/0064-scan-clean-java-generator-port.md#result) ports the proved shape as combined baked v9: scoped matched timing benefit, independent review and CPU/Javadoc/named conformance/integration gates PASS. Frontier None; no universal parity or vector/parallel Scan claim.
The [CPU0066 planning delivery](backends/cpu/tasks/0066-shared-scan-body-emission-plan.md) remains
Complete historical evidence. User-authorized [CPU0070](backends/cpu/tasks/0070-existing-emitter-composition-replan.md)
replaces its scalar/vector/worker sequence with [existing-emitter composition](backends/cpu/existing-emitter-composition-plan.md),
independently approved and MAIN accepted: completed planning → CPU0071 common MATMUL setup/binding
foundation → CPU0072 current pointwise/SUM closure. CPU0071 is Complete on 2026-10-29 after
corrected independent source/Javadoc review and MAIN-accepted module/named validation gates.
Its [narrow static compatibility cutover](../architecture/contracts/backend-execution.md#cpu0071-approved-static-matmul-compatibility-cutover)
is implemented. [CPU0072](backends/cpu/tasks/0072-common-pointwise-sum-composition.md#result) is also
Complete after Banach's independent source/Javadoc/docs/provenance APPROVE/P2 closure and MAIN's
combined module/Javadoc/named gates on 2026-10-29. Their frontiers are closed; CPU0073 completion
is below. CPU0067/0068 and the CPU0069 outline remain superseded.
The [CPU0071 Result](backends/cpu/tasks/0071-common-matmul-binding-foundation.md#result)
records finite-corpus preservation and the sole completed eight-row diagnostic. All semantic
checks passed, but warming transitions and unresolved generated/direct differences permit no
speedup or parity claim. Measurement budget consumed/closed: no repeat, retry or additional fork.
CPU0072 implemented approved helpers in existing owners, retaining literal accepted-current bytes,
identities, ABI/resources/bindings and explicit exact ordinary versus sequential Scan SUM algorithms.
Its brief owns the immutable baseline/supplement provenance, finite literal controls, truthful
failed-module-plus-targeted-correction evidence and named gates. The conditional four-group transfer
passed with inherited limitations, not new numerical/performance certification or historical Java
parity. No type/authority, namespace, resource, worker or Runtime handoff changed.
Durable next priority is [why actual Scan backward trails optimal matched Java](backends/cpu/existing-emitter-composition-plan.md#post-integration-actual-scan-evaluation):
compare complete code and avoidable loops/address/activity/saved/staging/entry work, not just repeat
benchmarks. CPU0071's length-reverse ~1.50 F32/~1.42 F64 is diagnostic with unstable warming in some
arms and no compilation-log C2 proof, not universal parity evidence. No new timing grant follows.
The bounded independent-Java code comparison identifies a cross-operation staging gap.
[General row-map/multiple-terminal fusion proposal](backends/cpu/row-map-multi-reduction-fusion-plan.md)
and [CPU0073](backends/cpu/tasks/0073-row-map-multi-reduction-fusion.md) have user approval of exact
types/signatures/identities/contract packet and implementation start on 2026-10-30. The narrow
authority cutover has Boole's independent APPROVE (0 P1/P2) and MAIN acceptance/verified Ready
receipt on 2026-10-30. CPU0073 implementation, independent code/tests/Javadoc/evidence-transfer
APPROVE (no open findings), MAIN's named conformance/Engine join and independent documentation
closure are Complete. The [task Result](backends/cpu/tasks/0073-row-map-multi-reduction-fusion.md#result)
owns exact validation: the failed full-module run is not relabeled PASS; targeted repairs and
unchanged-outcome transfer were accepted. Ordinary full-body units and Scan-emitted regions reuse
current math, retaining exact ordinary SUM/D stores and Scan's distinct per-time carry ADDs.
Source/measurement frontiers None; performance unmeasured, no automatic successor or commit/push.
CPU0070–0072 stay Complete and all prior budgets stay closed.
General existing-emitter composition remains the target, with distinct forward/backward/parallel
qualification; no all-operation support claim or speculative implementation cascade.
Ordinary workers stay unchanged; bounded Scan SIMD and unresolved parallel composition require
separate design/authority/qualification gates, not automatic threading or a mandated phase protocol.
CPU0071 → CPU0072 is complete; [Job reuse remains optional, not a dependency or blocker](backends/cpu/existing-emitter-composition-plan.md#deferred-job-reuse-evidence).
Future ordinary synchronous worker composition requires narrow approved authority/budget gates,
not zero scheduler metadata allocations. Current scalar inline Scan restrictions remain unchanged.
CPU0055 uses clean planning/launcher contexts and independent prelaunch/result review; MAIN owns
preservation, analysis and shared-document integration. Each successor requires its scoped Ready
verification. CPU0053's earlier priority change remains historical, not its execution authority.

The short probe does not establish performance parity: B=32 reverse paired medians span 41.129–78.837
times direct Java and change sharply between rounds. The separately authorized
[CPU0043 diagnostic](backends/cpu/tasks/0043-symbolic-length-reverse-diagnostic.md#result) is Complete:
two F32 B32/B7 order/control runs show compilation-phase confounds, not a uniquely diagnosed historical
cause or isolated lookup cost. No production repair or third probe; residual entry-inclusive overhead
remains unknown. Its Result owns crossover observations, JIT windows, launcher deviation and validation.
Complete Prepare0015 + CPU0041/0042 → CPU0043; independent Class C APPROVE and module/preservation
gates passed, MAIN accepted frozen source/evidence 2026-10-27 and retains final shared-document integration.
At CPU0043 acceptance CPU frontier was None; that diagnostic granted no automatic successor or timing.
[CPU0044 Complete](backends/cpu/tasks/0044-symbolic-length-scan-code-shape.md#result): MAIN accepted
the user-authorized bounded four-fix v4 repair and sole same-JVM v3/v4/unchanged-Java comparison on
2026-10-27 after independent source/evidence APPROVE, owning gates and completed baked-version docs.
Complete CPU0043 + Prepare0015 + CPU0041 + CPU0042 → CPU0044; verified base/dependencies and
seven integration fields remain in the brief. Parallel group None; MAIN owns final serial integration.
Both measured cases improved versus v3, but broader Java parity is UNRESOLVED, not an achieved goal
or automatic follow-up. Historical protocols/negative v3 Result stay verbatim; all four bodies consumed.
CPU source/authority frontiers None; no fifth/retry, rollback, new task cascade or further measurement.
Other frontiers and historical records stay held.

The user approved the next cohesive [Compiler0015 symbolic-length capability](modules/compiler/tasks/0015-symbolic-length-scan-inference-ad-and-binding.md):
one named B plus runtime discrete INT64[B] lengths, fixed T/features, homogeneous F32/F64 inference,
actual first-order sibling AD and complete fresh scoped all-port binding together. Compiler0015 is
Complete, MAIN accepted 2026-10-26 after independent source/docs APPROVE and recorded final gates.
The worker is closed; source/Ready frontiers None, Parallel group None. Binder production is unchanged;
Model Scan changes are support-status comments only. Combined Prepare/CPU/Engine execution,
cold/persistence and NN/Training remain closed. The task Result owns detailed evidence and limits.

The user now explicitly approves [Model0044 static/one-B length Scan source foundation](modules/model/tasks/0044-symbolic-length-scan-source-foundation.md):
one cohesive Model source capability with limited root/scan-owner wording before Java, tests/Javadoc/docs
and explicit pre-allocation Compiler/pre-extent Prepare rejection fixes. After independent pre-Java
APPROVE and MAIN authority/Ready acceptance on 2026-10-26, a different clean worker implemented it.
Independent source/docs APPROVE has 0 P1/P2; MAIN accepted Model0044 as Complete on 2026-10-26 after
the configured root test/Javadoc checkpoint passed. Worker evidence remains 1734 passing tests and
three Javadocs. Root included-project XML totals 707 suites / 5016 tests, 0 failures/errors, 56 skips,
including reused UP-TO-DATE evidence; the task Result records exclusions and skip inventory.
Recorded Model0044 admission: PlanningAuthorized=true;
AuthorityAuthorized=true only for bounded Model-source wording; ImplementationAuthorized=true only
for this source/rejection scope. Active Model frontiers: None; Parallel group None, MAIN completed
shared integration and its one configured root checkpoint. Compiler0015 subsequently implements
inference/AD/binding; symbolic-length Prepare/CPU/Engine execution and all successor frontiers remain
closed. No new type/API or authority-only task is selected.

[Model0043 symbolic-B/runtime-length Scan integration design](modules/model/tasks/0043-symbolic-length-scan-integration-design.md)
is Complete for DESIGN DELIVERY ONLY, MAIN accepted 2026-10-26 after Hume the 4th final independent
Class C APPROVE (0 P1/P2, no unresolved findings), following Ready/dependency/base verification
and explicit launch confirmation. User “ok, pust se do toho” authorized only the concise
[F32/F64 integration proposal](length-aware-general-scan.md#proposed-symbolic-b-plus-lengths-integration).
Historical PlanningAuthorized=true; AuthorityAuthorized=false; ImplementationAuthorized=false. The proposal
keeps source → Compiler inference/actual AD/binding → Prepare/CPU/Engine vertical coarse and serial,
with narrow static-wording and real outer F64/control prerequisites, no extra successor brief.
At that design acceptance, planning/source/authority frontiers were None; MAIN's serial review/delivery integration is
closed, Parallel group None. No Java/native/performance run, commit or push; all other gates held.

[CPU0042 length-aware general Scan](backends/cpu/tasks/0042-length-aware-general-scan-execution.md#result)
is Complete, MAIN accepted 2026-10-26 after independent Class C code/docs APPROVE and recorded
CPU/Javadoc, named joined checkpoint and sole probe. Fixed T/B/features F32/F64 forward and actual
Compiler reverse reuse existing CPU owners and the Prepare handoff. Dynamic-B plus lengths,
cold/v5 and NN migration remain excluded. The probe's timing transitions across measured pairs are
nonstationary, not steady-state parity or a confirmed compiler cause; its raw evidence and limits
belong in the brief. Source/test frontier None. Parallel group None; MAIN owns integration and
delegated final shared docs exclusively after freeze. No successor, additional measurement,
commit or push is authorized; other frontiers are unchanged.

[CPU0039 symbolic Scan fixed-geometry specialization](backends/cpu/tasks/0039-symbolic-scan-fixed-geometry-specialization.md#result)
is Complete for its bounded structural/correctness/verification scope, as explicitly accepted by
MAIN. That checkpoint and [CPU0040 diagnostic](backends/cpu/tasks/0040-symbolic-scan-phase-code-diagnostic.md#result)
left the performance gap unresolved. [CPU0041](backends/cpu/tasks/0041-symbolic-scan-reverse-carrier-specialization.md#result)
is now Complete, MAIN accepted 2026-10-26 after Peirce final source/docs and evidence APPROVE and
all gates. Immutable entry receivers for demanded port-backed cuts resolve the measured large-case
reverse gap with bounded near-Java phase timings, not universal speed parity or an internal C2 phi
proof. At CPU0041 acceptance CPU source/test frontier was None; CPU0042 completion is above.
Parallel group None; MAIN-owned serial integration is closed. The brief owns admission, all seven
integration fields, exact validation, measurements and limits. No new type/API/Runtime handoff,
unsafe route, arithmetic/resource change, protocol/oracle edit or wider campaign. Other areas held. Earlier
admission/freeze statements below describe their completed checkpoints, not current authorization.

The user selected [true per-run symbolic batch B](dynamic-batch-execution.md) through one immutable
compiled/prepared execution: no mandatory public B_max, required per-B full-preparation cache or
graph rebuild per batch. [Prepare0012 reviewed DESIGN delivery](modules/prepare/tasks/0012-dynamic-batch-geometry-and-resource-contract.md)
is Complete: the [proposal](modules/prepare/dynamic-batch-geometry-contract-proposal.md) has independent
Class C APPROVE, zero open P1/P2, and MAIN delivery acceptance, 2026-10-24. User “ano” now explicitly
approves architecture first, subsequent ordinary CPU forward/actual first-order backward and
bounded final A/B. [Prepare0012A](modules/prepare/tasks/0012a-dynamic-batch-authority-cutover.md) is
Complete: authority/ADR0050/explanations/glossary enacted/MAIN-accepted 2026-10-24; Pasteur definitive
APPROVE, zero open P1/P2, docs/glossary review complete. The sole serial implementation,
[Prepare0013](modules/prepare/tasks/0013-ordinary-dynamic-batch-cpu-vertical.md#result), is Complete:
ordinary homogeneous FLOAT32 CPU forward and actual first-order backward are implemented for one
named B, fixed features/parameters, rank-at-most-two boundaries plus qualified AD views, generated
scalar single-range execution and operation-qualified B=0. Final code/Javadoc/performance and
shared-document reviews approve with zero open P1/P2; MAIN's native-enabled root gate passed
and MAIN accepted the capability on 2026-10-24. Prepare frontier None; no successor authorized.

MAIN verified Complete Prepare0011 + Runtime0016 + Engine0021 + Compiler0013 + CPU0037 from
current Status/Result at 2026-10-24 17:41:11 UTC, plus design Complete. Their links and all seven
integration fields are in Prepare0012A; baseline 2435 files, HEAD/origin/index unchanged.
Base HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus accepted uncommitted work,
not merged predecessors. No competing writer; Parallel group None; MAIN owns shared documents.
Implementation is frozen; no automatic successor is authorized.
The geometry decision precedes a conflicting CPU binder/resource ABI brief; ordinary dynamic-B
CPU forward and first-order backward precede general Scan extension. Static length skipping stays
independent and separately gated, not cancelled or held on distant NN/persistence. Initial Scan T
and feature/parameter dimensions stay fixed; variable T/native/mixed/persistence/tuning remain later.

The user's next “ok, pokracuj” admits only
[Model0039](modules/model/tasks/0039-general-scan-symbolic-batch-design.md), now Complete for
DESIGN DELIVERY ONLY after independent frozen APPROVE (zero open P1/P2) and MAIN acceptance
2026-10-24, for the reviewed [symbolic-Scan proposal](modules/model/general-scan-symbolic-batch-proposal.md).
PlanningAuthorized=true; ImplementationAuthorized=false. One B, fixed T/F/H, all-valid F32 CPU
forward/actual first-order backward is the proposed execution slice, not current capability;
Model PURE/dtype source validity and Compiler AD admission stay separate. Zero new production
types proposed; existing owners/recipes need explicit source and layout/geometry changes.
Complete Model0035/0038, Compiler0013, Prepare0013 and CPU0037 satisfy planning admission on
the same accepted dirty base, no competing writer, Parallel group None; MAIN owns integration.
At Model0039 acceptance, planning/implementation frontiers were None. User detailed semantic approval
and coordinated authority admission have since followed under Model0041 below; no automatic successor launch.
Later cohesive source/AD/CPU gates are not Ready briefs.
Prepare/Compiler/CPU implementation frontiers remain None. Length skipping then selected
NN/Training reuse remain later gates; no execution, commit or push is authorized.

The user's “ok, dopln to” admits only
[Model0040 same-geometry reuse design amendment](modules/model/tasks/0040-same-geometry-reuse-design-amendment.md)
as Ready on 2026-10-25, now Complete for DESIGN AMENDMENT ONLY. Complete Model0039 and Prepare0013
satisfied admission; MAIN verified the accepted dirty base/inactive conflicts. Proposed single-entry
exact-preparation metadata reuse retains per-call Compiler binding/obligations, current input-backed
publication metadata, input/storage validation and isolated fresh run resources. Model0041 now
enacts the coordinated immutable-facts/mutable-run-array and prepared-meaning clarification.
PlanningAuthorized=true; ImplementationAuthorized=false. Planning frontier: None.
Implementation frontier: None. Independent APPROVE, zero open P1/P2; MAIN accepted 2026-10-25.
That design delivery alone grants no authority; the subsequent explicit authority admission is below.

## Current symbolic Scan implementation

The user “jdi na to” authorized [Prepare0014 symbolic Scan CPU vertical](modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md#result)
implementation and independent review. Complete, MAIN accepted 2026-10-25 after independent code/docs
APPROVE and the configured source-matched root gate. PlanningAuthorized=true; ImplementationAuthorized=true record admission.
MAIN verified source admission 2026-10-25 17:00 UTC: Complete Compiler0014, Prepare0013/0011, CPU0037,
Runtime0016, Engine0021, Model0041/0042 and Planning0008; accepted 2461-file planning snapshot,
stable contracts/base, empty index/inactive conflicts and sole writer. One atomic existing-owner
Prepare→CPU→Engine F32 forward/actual-backward packet, no memo or new architecture. Parallel group
None; MAIN owns shared docs/serial integration and one configured vertical join. Planning review:
APPROVE, 0 open P1/P2. At Prepare0014 acceptance, production source frontiers were None; worker validation and MAIN join
completed. The brief records bounded CPU/Engine probe results and limits; no commit/push or successor launch.

Complete Prepare0014 + CPU0037 → [CPU0038](backends/cpu/tasks/0038-symbolic-scan-timing-methodology.md),
Complete, MAIN accepted 2026-10-25 after final independent approval and the single bounded probe.
Its serial integration is closed; the brief owns historical validation provenance and limits.
That acceptance itself authorized no production/Engine change, successor or further measurement.

Complete CPU0038 + Prepare0014 + CPU0037 →
[CPU0039](backends/cpu/tasks/0039-symbolic-scan-fixed-geometry-specialization.md#result) Complete,
MAIN accepted 2026-10-25 after independent final code/docs APPROVE, CPU module/Javadoc, named
conformance/Engine integration and exactly one bounded probe PASS. This accepts only the bounded
fixed-geometry structural/correctness/verification scope: the largest observed generated/direct
ratio remains 2.351869× and smaller rows are mixed, not performance parity or causal attribution.
The explicit unresolved performance follow-up is now separately authorized as
[CPU0040](backends/cpu/tasks/0040-symbolic-scan-phase-code-diagnostic.md#result) Complete for bounded
diagnostic only after final independent APPROVE and MAIN acceptance; its brief owns evidence/limits.
CPU0040's two test types and original joint oracle/timing remain immutable under CPU0041.
Complete CPU0040 + CPU0039 + CPU0037 + Prepare0014 →
[CPU0041](backends/cpu/tasks/0041-symbolic-scan-reverse-carrier-specialization.md#result) Complete,
MAIN accepted 2026-10-26 after Peirce final source/docs and evidence approval, full CPU/Javadoc,
named join and final fresh phase/code capture. CPU/Prepare/Engine frontiers None; no automatic successor.
Existing CPU emitter/IR owners only, no MATMUL-emitter change, new type or handoff. Symbolic baked
v3 in both identity streams rejects v1/v2; canonical symbolic v1/family 71/static shaped-MATMUL v4/
ordinary identities and bytes remain frozen. One DEV + one FINAL JVM, worker zero; protocol and
four immutable files unchanged. Measured large-case reverse gap resolved with bounded near-Java
timings, not universal parity; the brief owns exact evidence/limits. MAIN serial integration closed.

At Compiler0014 acceptance, implementation frontier was None. [Compiler0014 symbolic Scan inference/actual first-order AD/scoped
binding](modules/compiler/tasks/0014-symbolic-scan-inference-ad-and-binding.md#result) is Complete,
MAIN accepted 2026-10-25 after independent code/docs APPROVE (0 P1/P2), module/Javadoc PASS
(384 tests) and the named scoped compatibility checkpoint PASS (68 tests).
The user's “ok, pust se do prace” historically authorized source implementation on 2026-10-25:
PlanningAuthorized=true; ImplementationAuthorized=true. Complete Model0041/0042 + Compiler0011/0013
and Planning0008 + Prepare0013 ordinary binding baseline satisfy dependencies; MAIN verified accepted
base/contracts/inactive conflicts and sole frontier before launch. Independent plan APPROVE, zero open P1/P2. Existing owners,
no new type/signature selected; Model PURE/dtype inference, bounded F32/F64 AD and later initial CPU
F32 qualification remain separate. Parallel group None; MAIN owns serial integration; exact metadata,
files and named compatibility gate are in the brief. Compiler source/tests/docs and the explicitly
admitted Model support-status Javadoc correction only; no downstream launch.

Current source frontier: None; Model0044 is Complete above. The all-valid [Model0042 static/symbolic Scan source foundation](modules/model/tasks/0042-symbolic-scan-source-foundation.md)
is Complete after user “ok, pust se do prace” and the explicit static-dimension preservation request.
Static and named-symbol Model source is implemented; independent APPROVE, zero open P1/P2, configured
root test/Javadoc PASS and MAIN acceptance 2026-10-25 are recorded in the brief.
Complete Model0041 authority plus retained Model0038/Compiler0013 satisfy admission, verified
2026-10-25 on the unchanged accepted dirty base. ImplementationAuthorized=true is confined to
Model source/Javadocs/tests and focused consumer rejection evidence; no symbolic Compiler/CPU/Engine
enablement, native build, benchmark, commit/push or automatic successor.

Standalone authority-only frontier: None; Model0044 includes its bounded source wording.
[Model0041 symbolic Scan and geometry reuse](modules/model/tasks/0041-symbolic-scan-authority-cutover.md)
is Complete after the user's “ok, pokracuj tedy” authorized the contract-first next step on 2026-10-25.
Complete Model0039/0040 designs plus retained Model0038/Compiler0013/Prepare0013/CPU0037 satisfy
admission; MAIN verified unchanged accepted dirty base and inactive conflicts. AuthorityAuthorized=true;
ImplementationAuthorized=false. Accepted-but-unimplemented Scan semantics and exact Engine metadata
memo exception, coordinated boundaries, ADR0051, glossary and future workflow are enacted.
Independent frozen read-only Class C APPROVE, zero open P1/P2; MAIN accepted 2026-10-25. No new type/signature
or atomic mechanism is selected. No Java, executable capability, commit/push or successor launch.

Model0036 remains Complete for reviewed proposal delivery. The subsequent “ok, pust se do prace”
selects the [length-aware design](length-aware-general-scan.md) and immediate authority plus first
Model foundation. [Model0037](modules/model/tasks/0037-length-aware-general-scan-authority.md),
verified Ready before drafting, is now Complete for coordinated authority/ADR0049: Hume's final
overall independent Class C APPROVE, zero open P1/P2, and MAIN acceptance on 2026-10-24.
[Model0038](modules/model/tasks/0038-length-aware-scan-source-foundation.md)
is Complete, MAIN-accepted 2026-10-24 after independent Class C/API/docs APPROVE, zero open P1/P2,
and successful combined integration retry. Source header/API/row validation/structural IR
plus explicit old-owner rejection grant no compile/AD/runtime/cold/wire route enablement.
Both briefs record the seven integration fields, verified frontier/dependencies/conflicts and
HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus accepted uncommitted work,
not merged predecessors. Worker gates and MAIN's combined named compile/architecture/existing-native
root test/Javadoc passed: first attempt failed only ledger metadata reflection, then a bounded
Class A test-only repair and same-command retry passed, both scan performance flags false, no
rebuild/campaign, successful/cached work reused. MAIN owns serial integration, Parallel group None. Later owner
phases remain coarse separately gated dependencies, no detailed cascade or automatic launch.
Length-mode source/IR support does not implement compile/AD/runtime/cold/v5 capability. No fluent
migration, higher-order waiver or model-wide masking follows; all other frontiers stay held.

After the accepted planning review, user “ok, pokracuj dal” authorizes only implementation of
[Compiler0013](modules/compiler/tasks/0013-length-aware-scan-capture-and-first-order-ad.md), launched
after Popper final APPROVE (0 P1/P2) and MAIN admission 2026-10-24. MAIN accepted Complete
2026-10-24 after independent code/docs APPROVE, worker module, compatibility and final docs/Javadoc
PASS; the brief's [Result](modules/compiler/tasks/0013-length-aware-scan-capture-and-first-order-ad.md#result) owns evidence.
Immediate DAG: Complete Model0037 → Model0038 plus Complete Compiler0011
and Planning0008 → Compiler0013; no new Planning transport/type layer. The brief records current
dependency evidence and all seven integration fields at the same HEAD plus accepted uncommitted
302-path/2429-file tree. MAIN owns serial integration, Parallel group None; no competing writer.
Compiler Ready frontier None; ImplementationAuthorized=true applies only to the recorded Compiler0013
implementation/tests/docs; it grants no automatic successor launch.
Scope is Compiler inference/role-aware first-order AD/query/demand only; cold capture/reconstruction,
CPU and codec remain closed; Prepare0011's separate authorization is below. No semantic authority redesign, commits/push
or detailed downstream cascade is authorized.

The Prepare0011 launch admission and review state are recorded:
[Prepare0011](modules/prepare/tasks/0011-length-aware-scan-projection-and-resource-handoff.md) was Ready
after Tesla planning APPROVE, zero open P1/P2; MAIN metadata accepted 2026-10-24.
Complete Model0037/0038 + Compiler0013 + Planning0008 + Prepare0010 feed one existing-owner Class C
capability for static lengths/formal separation, mixed reverse roles/demand and staged resource
handoff. At Prepare0011 completion, Prepare Ready frontier was None;
ImplementationAuthorized=true records completed Prepare0011 only,
authorized by the user's latest “pust se do prace” request. MAIN verified
admission 2026-10-24 12:25 UTC; clean sole writer started 12:31 UTC, now Complete after worker
71-focused/92-full test and Javadoc PASS. MAIN SPI-consumer compile/four retained guards PASS;
independent code/docs APPROVE, zero open P1/P2; MAIN accepted 2026-10-24 13:00:15 UTC. MAIN owns serial shared-document
integration. No new launch, API/architecture decision, downstream tasks, commit or push is authorized;
CPU mixed-boundary/preflight/skipping, cold/v5 and NN remain separately held.

The user approved only source-grounded fluent `.rnn(hiddenSize)` migration planning in
[NN0032](extensions/nn/tasks/0032-fluent-rnn-general-scan-migration.md), now a Draft decision packet.
It retains a confirmed advanced second-order CPU compile regression, not runtime Hessian proof;
the length proposal does not waive that blocker, the singleton-CONCAT copy or mixed-engine gates.
CP0013 now closes the general-body persistence prerequisite; later Engine choice still needs exact
old-supported backend/layout qualification, not inferred Metal support from DAG genericity.
The user subsequently approved selective v4 and the existing bounded cold inventory (“ok, pust se do toho”):
[CP0013 Complete](extensions/checkpoint/tasks/0013-general-scan-graph-persistence-decision.md#result)
has main's acceptance of final independent Cicero APPROVE (zero P1/P2), module/Javadoc, named consumer,
architecture and one configured root PASS, 2026-10-23. The same cohesive capability delivers authority,
codec/save-load/tests/docs and loaded consumer proof; no cascade. Remaining migration gates:
Cold output descriptor/remaining compatibility → fluent authority → separately Ready NN migration;
CPU0037's narrow read qualification is now Complete, not a migration grant.
No new production types/API or dependency changes; `.rnn` unchanged. GRU/LSTM stays later Draft.
NN/Checkpoint/CPU source frontiers None; CPU0037 is Complete below. Parallel group None, main owns integration. Historical
verified launch HEAD plus 292 accepted dirty paths (291 + ADR0047), approval/preservation evidence
and exact final counts remain in CP0013 Result; no additional test/performance campaign.

User selected general differentiable scan, NOT arbitrary while. [Program](general-differentiable-scan.md)
and [Model0034 brief](modules/model/tasks/0034-general-differentiable-scan-decision.md) record
Model0034 **Complete for authority decision/cutover only**: root/six owners, focused explanations,
ADR/glossary/navigation, final independent Class C APPROVE by Ramanujan the 2nd (zero blockers),
and main acceptance on 2026-10-21. The brief owns authority result evidence; Model0035 implements the bounded source/IR foundation below.
[Compiler0010 fixed-RNN-adjoint proposal](modules/compiler/tasks/0010-rnn-tanh-scan-bptt.md) is Superseded,
historical content/evidence retained. Base main=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`.
After the concrete ScanBody/reverse-scan/canonical-residual proposal and source-checked framework
comparison, user's “ok, tak tedy pokracuj” approves that proposal and authority cutover, not arbitrary
control flow or all implementation slices. Earlier planning APPROVE remains historical evidence.
Clean planner verified actual approval, all 16 declared baseline dependencies Complete, matching
HEAD/origin/main and inactive conflicts on 2026-10-21; all seven metadata fields are in the brief.
[Model0035 source/IR brief](modules/model/tasks/0035-general-scan-source-and-compiled-body.md) is Complete,
accepted by main 2026-10-22 after independent pre-code authority approval, final code/docs APPROVE
and the named compatibility checkpoint. Historical Ready verification used the same base plus accepted
dirty Model0034; the brief owns all seven fields and final evidence. Existing valid pure Model
algebra/dtypes remain independent of CPU qualification; conservative outer eligibility and canonical
metadata-only residuals are implemented, not generic capture/AD/execution.
Serial DAG: completed reviewed cutover → completed Model source/IR → completed Planning
body query → Compiler joint-output body AD/capture → Prepare projection → bounded F32
CPU0030 pointwise forward/automatic-reverse Complete → CPU0031 F32 broadcast/SUM/layout Complete
→ CPU0032 rank-two MATMUL/actual F32 reverse Complete → CPU0033 full homogeneous F64 algebra/closure Complete
→ low-level root/native join Complete.
CPU emits an outer time loop + admitted typed body emitters with explicit bindings, direct time offsets
preferred and necessary materialization justified; clean-Java/Class-File/short warmed performance gate.
First scope is canonical/static T/all-valid; BF16/FP16 qualification follows under existing semantics.
Parallel group None, main owns integration/documents at the base plus accepted uncommitted
authority/Model0034/0035, Planning0008, Compiler0011, Prepare0010 and CPU0030–0033 work.
At that completed milestone CPU frontier was None; CPU0031–0033 and low-level join Complete;
Prepare/Planning/Model/Compiler/NN and authority frontiers None; Compiler0012 Complete below.
CPU0030 bounded milestone Complete; no successor launch.
[Planning0008](modules/planning/tasks/0008-general-scan-body-capability-query.md) is Complete,
accepted by main 2026-10-22 after final independent Feynman APPROVE and exact downstream compilation,
architecture and old-provider/query compatibility gates. Its brief owns evidence and historical
Ready verification; main's diff/index/base/protected/110-Model-source-test freeze checks passed.
Source/IR, Planning transport and [Compiler0011](modules/compiler/tasks/0011-general-scan-capture-and-first-order-ad.md)
scoped capture/inference, joint first-order AD and final body queries/demand are implemented.
Compiler0011 is Complete, accepted by main 2026-10-22 after independent Class C code/docs APPROVE
and compatibility checkpoint; its brief owns evidence.
[Prepare0010](modules/prepare/tasks/0010-general-scan-projection-and-resource-handoff.md) is Complete:
exact scoped body/roles/demand validation and resource handoff, main accepted 2026-10-22 after
Huygens final independent Class C code/docs APPROVE and the exact compatibility checkpoint.
At that completed milestone CPU frontier was None; CPU0031–0033 and low-level join Complete;
Prepare/Planning/Model/Compiler/NN frontiers: None; Compiler0012 Complete below.
Bounded CPU F32 pointwise forward + actual
automatic reverse [CPU0030](backends/cpu/tasks/0030-general-scan-f32-pointwise-vertical.md) is Complete,
accepted by main 2026-10-22 after independent review and focused checkpoint PASS; its brief owns evidence.
[CPU0031](backends/cpu/tasks/0031-general-scan-f32-broadcast-sum-layout-closure.md#result) is Complete:
F32 broadcast/SUM/layout forward and actual Compiler reverse with differently shaped carries and
explicit stage bindings/resources. Main accepted 2026-10-22, 14:38+ UTC after final independent
code/docs APPROVE (0 P1/P2), module and named checkpoint PASS; the brief owns evidence and limits.
[CPU0032 Result](backends/cpu/tasks/0032-general-scan-f32-matmul-and-reverse.md#result) records completed
rank-two F32 MATMUL and actual Compiler reverse, accepted by main 2026-10-22, 17:09+ UTC after
independent code/docs and test-addendum APPROVE, CPU module/Javadoc and named checkpoint PASS.
The brief retains seven integration fields and historical launch verification; evidence stays there.
Main owns shared documents. [CPU0033 Result](backends/cpu/tasks/0033-general-scan-f64-domain-closure.md#result)
records Complete full homogeneous F64 algebra/actual reverse, independent code/docs and identity-test
addendum approval, final CPU/Javadoc and source-matched native root PASS. At that acceptance CPU frontier was None;
low-level join Complete, no successor launch. The unexplained timing gap and nonstationary samples
remain limitations, not JIT-cause or parity claims. No new production type or AD/execution layer.
Padding/NN, low types and vector/parallel/OpenBLAS remain separately gated Draft; CP0013 persistence is now Complete;
old four-dtype/layout fixed and static routes remain.
CPU0063/0064 Complete; accepted combined baked v9 with scoped timing benefit and recorded gates, frontier None. CPU0062 measurement slots closed; CPU0061 Cancelled. CPU0060 Complete with stated limits; CPU0059 and CPU0054–0058 Complete, CPU0056 withdrawn. CPU0053 Cancelled; Compiler0016 and CPU0046–0047 Complete, CPU0048–0052 queued. CPU0041–0045 remain Complete without broader parity. [CPU0037](backends/cpu/tasks/0037-general-scan-dense-read-views.md#result)
is Complete, accepted by main 2026-10-24 after Newton final independent APPROVE and CPU/Javadoc,
named conformance/integration and one configured root PASS. It closes the audit's CPU dense-read
gap with real Conv1d-prefix/singleton-CONCAT actual reverse gradients, preserving formal/write
requirements and existing scalar reads. CPU0036 → CPU0037 Complete; no new performance claim.
Parallel group None, main-owned integration closed; no successor. NN0032 remains Draft for cold
output descriptor, remaining compatibility and fluent-authority gates; `.rnn` unchanged.
[CPU0036](backends/cpu/tasks/0036-general-scan-javac-loop-control.md#result) remains Complete,
accepted by main 2026-10-23 after final independent evidence/docs approval.
CPU0033 + low-level join and accepted Compiler0012 baseline →
[CPU0034](backends/cpu/tasks/0034-general-scan-coordinate-loops-and-carry-transition.md#result), all Complete.
Main accepted 2026-10-23 after serial Class C implementation/reviews, final CPU test/Javadoc,
targeted conformance/Engine checkpoint and independent audit of 900 paired samples/JIT state.
At CPU0034 acceptance staged compatibility payloads were v2, not global envelope 71 or canonical v1 projections; no new
type/capability. The brief owns integration metadata, failed-setup disclosure and bounded timing limits.
CPU0034 → [CPU0035](backends/cpu/tasks/0035-general-scan-emission-cleanup-and-specialized-oracle.md#result)
are Complete; CPU0035 staged v3 is historical. CPU0035 → CPU0036 are Complete for bounded
top-tested loop control, not algorithm/memory-strategy changes; current staged payloads are v4,
schema/family 71 and geometry-free canonical v1 unchanged. CPU0036 Result owns exact evidence/limits;
its brief retains verified base/dependencies/conflicts and integration metadata. Parallel group None;
serial clean implementation/independent reviews, main-owned shared documents/checkpoints/single campaign closed.
Observed timings are narrow evidence, not universal parity or causal proof; supplementary guard qualification
remains insufficient, not itself proof of JIT instability. No additional campaign/performance qualification implied.
Checkpoint/NN/other gates remain held; no automatic launch or full-root/timing-matrix campaign.
No if/while/general-control-flow or associative/parallel-scan prerequisite; generic no-copy/JIT/speed
parity and inherited CPU0029 timing proof are not claimed.

Approved scan direction: [NN0030](extensions/nn/tasks/0030-general-scan-owned-model-admission-decision.md),
Complete for limited direction/planning only; user-approved NN0031 and its native program join are now Complete.
Select existing Model.graph/defineTensor and compute/configureTraining/train, initially tested
for all-valid canonical static-T F32/F64 CPU scans. PURE-body source semantics/dtypes and exact
cold-copy reconstruction admission stay independent of CPU execution capability. Compiler cold
reconstruction, Checkpoint attachment-preserving in-memory normalization and NN workflow qualification
are Complete, without another runner/API. Serial order: accepted authority/inventory
→ Compiler cold source → Checkpoint copy/rebind with unchanged v1/v2/v3 rejection → NN0031/main join.
Padding, fixed-scan BPTT and low/vector/parallel/OpenBLAS execution remain separately
gated. [Compiler0012 Result](modules/compiler/tasks/0012-general-scan-cold-source-reconstruction.md#result)
records Complete bounded cold reconstruction after independent code/docs APPROVE and main's final
module/consumer gates (2026-10-23), retaining historical pre-code/Ready verification.
User-authorized [CP0012](extensions/checkpoint/tasks/0012-general-scan-in-memory-copy-and-rebind.md)
is Complete, accepted by main 2026-10-23 after clean implementation, independent code/docs approval
and scoped module/required consumer gates. It preserves attachments and independent body IDs, delegates
cold reconstruction to Compiler, keeps wires closed and adds only NN compatibility rejection guards.
Its [Result](extensions/checkpoint/tasks/0012-general-scan-in-memory-copy-and-rebind.md#result) records
the two root attempts' unrelated missing-native-environment failures; unrestricted root is NOT green.
Those failures authorize no native test-source/gating/skip correction in NN0031.
Historical base HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus 274 accepted dirty paths;
Parallel group None, main owns serialized integration/shared documents. At CP0012 acceptance,
Checkpoint/Compiler/NN frontiers were None; no successor was launched by that acceptance.

[NN0031 Result](extensions/nn/tasks/0031-general-scan-owned-model-workflow.md#result) records Complete
bounded owned acquisition, compute/first-order train and weight updates, accepted 2026-10-23 after
clean implementation, independent Class C code/docs PASS, final NN test/Javadoc and main's single
configured native-enabled root PASS. Historical launch verification and seven integration fields
remain in the brief. Main owns integration with exclusive final documentation delegation; both scan
performance flags stayed false. Required Metal TrainingSession and earlier failing native cases
executed unskipped/pass using the existing library. CP0012's earlier missing-environment exception is
historical, not a gating-code fix or removal of the explicit native environment requirement.
NN/Compiler/Checkpoint/CPU/other program frontiers None; no automatic successor or new campaign.

1. [Model 0033 stable-softmax task](modules/model/tasks/0033-stable-softmax-overflow-semantics.md)
   is Complete: Model authority, CPU conformance, bounded v2 generated/direct evidence and
   independent Class C approval close G1. No further numerical-contract task is implied here.
2. [Training 0003A](extensions/training/tasks/0003a-exact-resume-state-ownership-decision.md)
   is Complete after independent Class C approval at base
   `34fb1c8d6a48f9a9352c25597a86045180477fbc`. It selects a bounded state-complete
   claim, caller-owned next-input/external-state handoff, and no Training 0003 API.
   The bounded caller-owned private-candidate proof in
   [Training 0004](extensions/training/tasks/0004-joint-restore-preflight-and-candidate-commit.md)
   is Complete after independent Class C review and the integration-owner repository gate.
   It is in-memory, same-route evidence, not durable exact-resume persistence. Do not invent a
   DataLoader or imply the current fixed session advances graph RNG between runs.
3. Durable model save/load was the selected stream and is now Complete. [Checkpoint 0001](extensions/checkpoint/tasks/0001-checkpoint-module-boundary.md)
   has completed the model-only module boundary. [Checkpoint 0002](extensions/checkpoint/tasks/0002-bounded-artifact-envelope.md)
   is Complete with a bounded single-file codec and atomic-live-publication policy;
   [0003](extensions/checkpoint/tasks/0003-model-schema-and-materialization-handoff.md) is
   Complete after implementation, validation, and independent review; [0004](extensions/checkpoint/tasks/0004-public-model-checkpoint-save-inspect-load.md)
   is Complete after the target-alias/unbound-target policy decision, independently reviewed
   public implementation, [CPU 0024](backends/cpu/tasks/0024-default-segment-route-heap-carrier-binding.md)
   repair, direct restored-heap Engine regression, targeted documentation, and combined gate.
   [0004A](extensions/checkpoint/tasks/0004a-direct-leaf-model-save.md) is Complete: Checkpoint-owned
   raw `CONTIGUOUS` requests support direct leaf-model save and loaded re-save through existing
   Engine compute, removing the private recovery workaround. Independent review and the configured
   coordinator gate passed; generic zero-node Engine/Prepare support and the durable adapter
   remain outside scope.
4. [Data0001](extensions/data/tasks/0001-data-extension-boundary.md) is Complete, accepted
   2026-10-20 after independent approval and main's cached root gate. Root/single owning contract,
   ADR0042 and explanations preceded the empty Gradle foundation and architecture enforcement.
   That foundation delivers only Data's project/path/namespace and sole Model API edge; Text/Vision retain
   separate decisions. Parallel group None; main owns serial integration/shared documents.
   The brief owns base, launch verification and exact evidence. NN/CPU/Text/Vision frontiers:
   None at that foundation checkpoint; the subsequently completed Text capability is below.
   NN0022A/0023/0024 remain Draft with unchanged downstream gates.
   CPU0028 supplies the bounded runtime scan route; completed NN0022 expressions and NN0022B
   Model/builder inference are independent of Data. NN0022A retains Data0001/0002 integration
   gates, without automatic successor launch or BPTT cancellation.
   [Data0002](extensions/data/tasks/0002-valid-sequence-lengths-and-right-padded-batch.md) is Complete,
   accepted 2026-10-20: copied host lengths, a borrowed Tensor record, explicit logical axes and
   nested fluent Builder after independent APPROVE and the bounded coordinator metadata checkpoint.
   Its brief owns exact evidence and source freeze at the same base plus accepted uncommitted
   Data0001, not a merged predecessor. That completed slice added no padding algorithm or runtime
   binding. The user subsequently approved [ADR0043](../design/decisions/0043-whitespace-text-and-padded-integral-batches.md):
   [Data0003](extensions/data/tasks/0003-padded-integral-sequence-batching.md) is locally Complete
   after final independent APPROVE and main acceptance. Text0001 is Complete, accepted 2026-10-21
   after independent code and final documentation APPROVE and numerical/root gates.
   Main verified approved root/owning authority and unchanged HEAD
   `ef35f2ec001b0c161d1abe825ff3be725b9be49e` plus accepted dirty Data0001/0002 + authority +
   fixed Data0003/final docs, inactive conflicts and unchanged protected hashes on 2026-10-20;
   Text0001 is also accepted uncommitted work, not a merged predecessor.
   Serial order, no parallel writers; main owns shared integration and the final source-matched
   root gate at the Text program join, now complete; exact evidence is in Text0001 Result.
   No NN edge, runtime binding, scan/BPTT extension or native lifetime is granted.

Step 2 records the completed decision and Training 0004 proof, not a durable artifact. Step 3 records
completed Checkpoint 0003/0004/0004A and CPU 0024. The subsequently authorized Training-owned
Checkpoint 0005 prerequisite is Complete, not authorization for the Draft durable adapter.
Step 4 closes Data0001–0003 and Text0001; Data/Text/Vision frontiers None, broader successors Draft.
The subsequently completed NN0022A above is numeric-only; no fluent Text pipeline or BPTT.
The approved NN-only NN0022B capability is Complete.
An independent performance/shape follow-up, [CPU 0023](backends/cpu/tasks/0023-vector-erf-direct-java-shape.md),
is also Draft: the bounded vector ERF diagnostic did not demonstrate parity with direct Java.
It is not a prerequisite for unrelated Training/Checkpoint decisions or a new broad campaign.

The completed Training/Checkpoint chain remains:
`Training 0003A + Checkpoint 0001–0004 → Training 0004 → Checkpoint 0004A →`
`Training-owned Checkpoint 0005 → Checkpoint 0005A` (CPU 0024 also precedes 0004/0004A).
Training 0003 is Superseded. [ADR 0036](../design/decisions/0036-model-envelope-composition.md)
and published model.v1 envelope APIs remain implemented, not durable training recovery.

The earlier proposed serial DAG is held after its completed prefix:
`NN0026 Complete → CP0005B Complete | HOLD: NN0027 → CP0005C → Training0006 → optional CP0006`.
Completed 0005A still feeds any later adapter. NN0027/CP0005C are not silently cancelled or Ready.
The revised [program](model-execution-refactor.md) replaces outputs(actualInputs)-first execution
with the same proposed model.compute/train/save/load for application-built and file-restored tensors.
Load reconstructs ordinary expressions/weights before compile with no Engine or future batch;
same-static-signature calls reuse preparations with per-run borrowed storage and no shared-placeholder
mutation. Existing Tensor DAG/autograd/Prepare/Runtime/backend execution is reused.

Phases: planning review → backend/native-only relocation; separately reviewed P2 API/ownership/compatibility
packet → generic Training + fresh-input Engine binding and generic Checkpoint + Compiler cold
persistence → common model workflow. Backend targets are modules/backend/{cpu,metal,cuda,
openblas-provider}; Metal native target is modules/backend/metal/macos-arm64, without native nesting.
Backend projects remain separate with unchanged :backends:* identities. Generic Training/Checkpoint
target modules/training and modules/checkpoint; NN stays extensions/nn, ONNX a marker.
The program owns the exact symbol/edge map; existing area plans stay in their historical directories.

CP0005D is Complete for bounded planning/review; Engine0020 is Complete after its reviewed path-only
cutover and combined validation. The user's subsequent approval accepts all five product choices.
P2 selects ordinary Model.defineTensor(signature, definition)/Model.load with single-Tensor compute,
configured train and save; generic Model.define/forward remain. New tensor construction discards
its prototype callback and uses owned source leaves, including for explicit cold symbolic forward.
`Model.load(path[, limits]).withEngine(engine)` returns the same Model and borrows the
Engine for compute/train; binding freezes before first preparation setup, including failed setup.
Engine.standard is lazy/model-owned (CPU-only, no discovery/global); explicit overloads remain.
compute(batch) stays terminal/detached; NN0028 delivers this API. One owned weight set serves
compute/train/save; loaded values need no ownership-only copy. First graph signatures are fixed.
Old flexible declarative/v1 behavior and readers remain. Source migration is explicit in ADR0039.

Implementation DAG: Engine0021 → Training0007 → Compiler0008 → Checkpoint0008 → Training0008 →
NN0028 is the serial shared-document integration order; code dependencies are recorded in the
[program DAG](model-execution-refactor.md#phases-and-owner-dag). Training0007 does not need fresh
Engine binding; Training0008 does. Compiler0008 provides Checkpoint0008 cold reconstruction.
[Engine0021](modules/engine/tasks/0021-prepared-execution-fresh-value-binding.md) is Complete:
InferenceSession and Engine.session were removed, and Training/benchmark/test consumers migrated
atomically to public Engine.prepare/run. TrainingSession's optimizer/transaction state remains.
Engine.run keeps exact-ID/arbitrary-order binding; runValues accepts fresh positional values.
Both use prepared-before-input close admission, an explicit direct-run precedence change.
Engine/Runtime prepared boundaries, advanced lifecycle and detached results remain.
Training0007 is Complete after final independent approval and coordinator acceptance; its brief owns evidence.
Compiler0008 is Complete and pushed at main/origin/main base `d235c0d3b89d6f2eaad4c80a59fd1c5091ea2851`:
bounded API/root/compiler authority implemented, independently approved and coordinator-gated.
At that checkpoint Compiler/Checkpoint/NN frontiers were None. NN0029 is Complete. Checkpoint0009 is Complete. Checkpoint0008 is Complete; final independent
approval and coordinator gate are recorded in its brief. Training0008 is Complete and integrated at main
`7671bd4bebc8723fa4864c6423f78219bf63851b`; its brief records Euclid's definitive APPROVE
and main's source-matched root gate (2026-10-19). Its bounded root/owning Training lifecycle cutover
is enacted; no authority is changed by planning. NN0028 and high-level Model/withEngine are Complete.
The narrow post-preflight
call-local reconstruction exception is enacted; Checkpoint0008 now implements direct cold
Checkpoint→Compiler access and generic reversed edges in coordinated authority, with final
independent review/coordinator gate passed. Current contracts win.
No optimized scan/backend BPTT/Data/Text or broad performance prerequisite is added.

NN 0026's matrix/API is accepted and coordinated with root NN ownership, the owning scoped contract,
architecture explanations and ADR 0038's NN decision. It includes nested Sequential,
Linear/LayerNorm/Embedding/Conv1d/2d/3d and RELU/TANH/SIGMOID, source-faithful input constraints and
Tensor-identity aliases, not arbitrary Model.define capture. Final independent approval and coordinator
verification of dependencies, accepted contract base `4472320ceb5498c46a661af549d6de74aa27ab6a`,
inactive conflicts and launch frontier are recorded. NN0026 implementation, final independent reviews
and coordinator validation are complete. Evidence is in its brief; NN0028 is also Complete and current authorized NN frontier is None.

CP0005B is Complete with the accepted CP-only whole-model contract/API/wire/default packet.
Its brief records final independent code/docs APPROVE, worker module tests, final Javadoc and
coordinator architecture/integration evidence. Execution used accepted packet base
`7d65f660051becc9bc6adc6f08af155b08096b59`. Defaults: 256 MiB file/aggregate, 4 MiB
manifest, 16384 entries, 64-byte keys, 64 MiB payload; fixed 1 MiB definition/4096 nodes/depth64.
NN0027/CP0005C recurrent initial-state/output selection/format version and their Training admission
remain optional held proposals, not prerequisites of the new ordinary graph path.
Training0006/adapter0006 loss/recipe and owner-transfer/capture/restore
bridges remain Proposed, each requiring its own coordinated contract before Ready.
Inference constraints remain separate from static Training signatures. Existing ArtifactFile v1
and model.v1 stay unchanged; whole-model.v1 is implemented for the closed NN0026 matrix, not recurrent-file
support. training-model.v1 remains Proposed, not same-file compatibility with withdrawn unpublished
training-checkpoint.v1. CP short load rejects stored input digests; typed explicit expected identity
must match presence/bytes. No new public wrapper/validator or dependency edge.

NN0026 implements the accepted NN contract without changing dependency edges or build configuration.
Parallel group None; coordinator owns shared documents and recorded Ready verification.
NN worker runs affected-module test/Javadoc once; coordinator runs architecture and narrow CPU
integration validation after review. The full source-matched repository test/Javadoc gate belongs
to the selected program join, not NN alone. Planning review checks links/anchors/sizes/diff only,
with no Java reruns, native builds or performance campaign.
Data cursor/RNG/scheduler/scaler stay application-owned; no exact automatic continuation, advancing
dropout stream, BatchNorm training-buffer alignment, fixed RecurrentScan BPTT or Conv3d gradient widening.
Ordinary static unrolling depends on normal constituent-operation adjoints, not a blanket ban.

## Explicit non-goals and unresolved decisions

- Do not restart historical profile/certificate/proof gates as prerequisites for current work.
- Do not equate operation-kind coverage with all dtypes/layouts/gradients/strategies or speed.
- No broad performance campaign, new native provider, automatic backend discovery, rollback-mode
  change, or new numerical policy is authorized by the prior review. Checkpoint 0004's model
  save/inspect/load and direct Engine acceptance are Complete; future persistence work remains
  separately gated.
- GPU CI provisioning, large-model aggregate memory policy, new vendor routes, unselected recurrent
  execution beyond CPU0029's scalar scope and missing adjoints retain their decisions/owning plans.

## History

The 2026-10-13 reconciliation removed duplicated completion narratives and stale global-frontier
claims from this index. Completed and historically blocked task evidence remains in its original
briefs and Git history. Future updates change current status, dependencies and next gates only.

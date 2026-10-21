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
Authorized Checkpoint frontier: None; Checkpoint0008/0009 are Complete.
[Training0008](extensions/training/tasks/0008-fresh-value-training-session-runs.md) is Complete after
Euclid's definitive code/docs APPROVE and main's source-matched root gate (2026-10-19).
Authorized Training frontier: None; [NN0028](extensions/nn/tasks/0028-owned-model-compute-training-and-persistence.md)
is Complete, accepted 2026-10-19. [NN0029](extensions/nn/tasks/0029-fluent-model-builder-and-checkpoint-independent-execution.md)
is Complete after Planck final APPROVE, zero findings, and main's source-matched native root gate (2026-10-19), including independently reviewed [Compiler0009](modules/compiler/tasks/0009-static-select-and-slice-logical-layout-closure.md) and [CPU0025](backends/cpu/tasks/0025-scatter-validation-coordinate-rank.md). [CPU0026 diagnostic](backends/cpu/tasks/0026-stacked-lstm-cold-preparation-diagnostic.md), [CPU0027 unit topology fix](backends/cpu/tasks/0027-unit-topology-adjacency-ordering.md) and CPU0028/NN0021B bounded scan execution are also Complete. Authorized CPU/Compiler/NN frontiers: None; NN0022B is Complete.
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
checkpoint passed. Compiler0009 is locally accepted and Complete; authorized Compiler frontier: None. [Checkpoint0008](extensions/checkpoint/tasks/0008-generic-graph-value-persistence-and-nn-glue-cutover.md)
is Complete after Halley's definitive APPROVE and the coordinator's source-matched test/Javadoc
gate on 2026-10-17. Its brief owns evidence. Checkpoint0009 is Complete; frontier None. Training0008 is Complete
after Euclid's final APPROVE and main's root gate; NN0028 is Complete under the bounded NN authority.

| Area | Delivered baseline / remaining gate |
|---|---|
| [Model](modules/model/master-plan.md) | Complete through 0032, active profile-free semantics and FLOAT16. Obsolete 0026 is Superseded. |
| [Trace](modules/trace/master-plan.md) | Envelope/IDs and preparation/invocation/low-route facts delivered; broader payloads/encoding Draft. No numerical-profile DTO. |
| [Backend Contract](modules/backend-contract/master-plan.md) | Complete through 0004; reopen only for a concrete shared need. |
| [Config](modules/config/master-plan.md) | Compile leaves and two-phase tuning policy delivered. Cost consumer/aggregate/run configuration Draft; no numerical-profile API. |
| [Planning](modules/planning/master-plan.md) | Complete through 0007; cost model awaits a concrete consumer and units. Partition size is not fusion size. |
| [Runtime](modules/runtime/master-plan.md) | Complete through 0016; per-run isolation and prepared-resource leases remain current. |
| [Compiler](modules/compiler/master-plan.md) | Complete through 0006B11 plus 0006D/0008/0009; [0009 exact SELECT/coordinate-SLICE logical closure](modules/compiler/tasks/0009-static-select-and-slice-logical-layout-closure.md) locally accepted after final independent APPROVE, zero findings. Compiler frontier None; main's combined native root gate passed. Conv3d adjoints 0006C and exact rewrites 0007 Draft; BPTT unsupported. |
| [Prepare](modules/prepare/master-plan.md) | Complete through 0009; opaque backend collaboration and complete resource transaction. |
| [OpenBLAS provider](backends/openblas-provider/master-plan.md) | 0001–0003 Complete. Optional direct BF16 0004 Blocked on ABI and one-final-narrowing evidence; F32/F64 unaffected. |
| [CPU](backends/cpu/master-plan.md) | Complete through 0022 plus 0024–0028. [0028 fixed-scan execution](backends/cpu/tasks/0028-fixed-recurrent-scan-forward-execution.md) delivers canonical FLOAT32 scalar Java forward execution, accepted 2026-10-20 after independent approval and main's root gate. Frontier None; 0023 Draft, defaults/vendor gates unchanged. |
| [Engine](modules/engine/master-plan.md) | Complete through [0021 prepared-owner/fresh-value binding](modules/engine/tasks/0021-prepared-execution-fresh-value-binding.md), including thin-session removal and mechanical consumers; independent approval and native-enabled test/Javadoc gate passed. Authorized Engine frontier: None. |
| [Metal](backends/metal/master-plan.md) | Complete through 0079. Current occurrence support is independently typed; hosted GPU CI 0047 remains blocked. |
| [CUDA](backends/cuda/master-plan.md) | Marker/native scaffold only; no executable CUDA backend. 0001 Draft. |
| [ONNX](extensions/onnx/master-plan.md) | Marker only; no importer/exporter. Area Draft; first mapping task not yet defined. |
| [Data](extensions/data/master-plan.md) | 0001–0003 Complete; [0003 shared padded representation and INT64 batching](extensions/data/tasks/0003-padded-integral-sequence-batching.md) accepted after final independent APPROVE and module/architecture-consumer gates, 2026-10-20; Text program root join complete, 2026-10-21. Frontier None; broader batching Draft, no runtime binding. |
| [NN](extensions/nn/master-plan.md) | 0028/0029, [0021B CPU scan program](extensions/nn/tasks/0021b-fixed-recurrent-scan-implementation-program.md), [0022 NN-only expressions](extensions/nn/tasks/0022-runtime-length-recurrent-nn-api.md), [0022B Model/builder inference](extensions/nn/tasks/0022b-runtime-length-scan-model-workflow.md) and [0022A numeric inference](extensions/nn/tasks/0022a-padded-numeric-sequence-inference.md) Complete. NN0022A accepted 2026-10-21 after independent APPROVE and main's source-matched architecture/root checkpoint. NN/CPU/Data/Text frontiers None. 0023/0024 Draft; no fluent Text pipeline, scan training/BPTT or automatic successor launch. |
| [Text](extensions/text/master-plan.md) | [0001 frozen whitespace tokenizer and Data adapter](extensions/text/tasks/0001-whitespace-tokenizer-and-data-batching.md) Complete, accepted 2026-10-21 after independent code/final documentation APPROVE and numerical/root gates. Sole api(Data), Model transitive. Frontier None; training/artifact work Draft. |
| [Vision](extensions/vision/master-plan.md) | No module; frontier None. Separate Vision architecture approval before decoder/preprocessing work; later numeric batching retains its Data gate. Data0001 grants no Vision edge. |
| [Training](extensions/training/master-plan.md) | Sessions/optimizers/state, [0007 generic live admission](extensions/training/tasks/0007-generic-live-parameter-admission.md) and [0008 fresh-value runs](extensions/training/tasks/0008-fresh-value-training-session-runs.md) Complete; final independent reviews and coordinator gates passed. Authorized Training frontier: None; NN0028 Complete; 0006/adapter held. |
| [Checkpoint](extensions/checkpoint/master-plan.md) | Complete through joint [0010 CNN/static recurrence](extensions/checkpoint/tasks/0010-fluent-cnn-and-static-recurrent-graph-persistence.md) / [0011 cold flat scans](extensions/checkpoint/tasks/0011-flat-recurrent-scan-persistence.md), accepted 2026-10-20 after independent APPROVE and main's focused/native root gates. CP0010 owns evidence; authorized frontier None. |
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
Compiler0009 + NN0029 → CPU0026 → CPU0027 are Complete; authorized CPU frontier None, CPU0023 Draft.
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
Current NN/CPU/Data/Text frontiers: None; NN0022A is closed.
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

## Proposed next decisions

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
Current Compiler/Checkpoint/NN frontiers are None. NN0029 is Complete. Checkpoint0009 is Complete. Checkpoint0008 is Complete; final independent
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
- GPU CI provisioning, large-model aggregate memory policy, new vendor routes, recurrent execution
  beyond CPU0028's baseline and missing adjoints retain their explicit decisions and owning plans.

## History

The 2026-10-13 reconciliation removed duplicated completion narratives and stale global-frontier
claims from this index. Completed and historically blocked task evidence remains in its original
briefs and Git history. Future updates change current status, dependencies and next gates only.

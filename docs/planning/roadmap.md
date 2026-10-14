# Implementation Roadmap

## Purpose and authority

This is the active index for area status, dependencies and authorized frontiers, not an
architecture contract. [ARCHITECTURE.md](../../ARCHITECTURE.md) and its incorporated scoped
contracts govern implementation. Read the owning master plan and compact Ready brief for scope;
completed evidence belongs in linked task records and Git history.

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
configured coordinator gates. Authorized Checkpoint frontier: None.
The user revised the primary product to automatic usable-model load and a choice of inference or
Training-owned resume, without routine callbacks/manual construction/digests/Arena.
[Proposed ADR 0038](../design/decisions/0038-declarative-model-and-training-resume.md) replaces
the withdrawn Proposed direction of ADR 0037. The complete bounded Draft program is NN 0026,
Checkpoint 0005B, Training 0006 and revised Checkpoint 0006; no new Ready frontier, code or build
inclusion is authorized. Current published explicit APIs remain implemented and compatible.

| Area | Delivered baseline / remaining gate |
|---|---|
| [Model](modules/model/master-plan.md) | Complete through 0032, active profile-free semantics and FLOAT16. Obsolete 0026 is Superseded. |
| [Trace](modules/trace/master-plan.md) | Envelope/IDs and preparation/invocation/low-route facts delivered; broader payloads/encoding Draft. No numerical-profile DTO. |
| [Backend Contract](modules/backend-contract/master-plan.md) | Complete through 0004; reopen only for a concrete shared need. |
| [Config](modules/config/master-plan.md) | Compile leaves and two-phase tuning policy delivered. Cost consumer/aggregate/run configuration Draft; no numerical-profile API. |
| [Planning](modules/planning/master-plan.md) | Complete through 0007; cost model awaits a concrete consumer and units. Partition size is not fusion size. |
| [Runtime](modules/runtime/master-plan.md) | Complete through 0016; per-run isolation and prepared-resource leases remain current. |
| [Compiler](modules/compiler/master-plan.md) | Complete through 0006B11 plus repair 0006D. Conv3d adjoints 0006C and exact rewrites 0007 Draft; BPTT unsupported. |
| [Prepare](modules/prepare/master-plan.md) | Complete through 0009; opaque backend collaboration and complete resource transaction. |
| [OpenBLAS provider](backends/openblas-provider/master-plan.md) | 0001–0003 Complete. Optional direct BF16 0004 Blocked on ABI and one-final-narrowing evidence; F32/F64 unaffected. |
| [CPU](backends/cpu/master-plan.md) | Complete through 0022 plus [0024](backends/cpu/tasks/0024-default-segment-route-heap-carrier-binding.md), including heap binding and mixed-carrier overlap repair; vector ERF shape follow-up 0023 Draft. 0010M retained scalar/1/1/1 defaults; vendor peers remain use-case/ABI gated. |
| [Engine](modules/engine/master-plan.md) | Public composition, sessions, compute/backward, host materialization and bounded tuning delivered; profile spine is historical after Model 0032. |
| [Metal](backends/metal/master-plan.md) | Complete through 0079. Current occurrence support is independently typed; hosted GPU CI 0047 remains blocked. |
| [CUDA](backends/cuda/master-plan.md) | Marker/native scaffold only; no executable CUDA backend. 0001 Draft. |
| [ONNX](extensions/onnx/master-plan.md) | Marker only; no importer/exporter. Area Draft; first mapping task not yet defined. |
| [Data](extensions/data/master-plan.md) | No module. 0001 architecture decision precedes batching; current plan excludes sampler/epoch lifecycle. |
| [NN](extensions/nn/master-plan.md) | State/layers/static recurrence/convolution delivered. [0026 declarative reconstruction](extensions/nn/tasks/0026-declarative-model-specification.md) Draft for whole-model load; no Ready frontier. Runtime scan/Data 0021B–0024 separate Draft stream. |
| [Text](extensions/text/master-plan.md) | No module. Depends on Data architecture, then selected tokenizer/artifact/batching scope. |
| [Vision](extensions/vision/master-plan.md) | No module. Shares Data 0001 architecture decision before decoder/preprocessing work. |
| [Training](extensions/training/master-plan.md) | Sessions/optimizers/state and private-candidate proof delivered. [0006 recipe/owned run](extensions/training/tasks/0006-supported-recipe-and-owned-run.md) Draft prerequisite for adapter; no persistence or Ready frontier. Broader 0005 separate Draft. |
| [Checkpoint](extensions/checkpoint/master-plan.md) | Complete through 0005A. [0005B whole-model load](extensions/checkpoint/tasks/0005b-whole-model-save-load.md) and revised [0006 save/loadModel/resume](extensions/checkpoint/tasks/0006-training-checkpoint-adapter.md) Draft under ADR 0038. Authorized frontier: None; 0007 later Draft. |
| [Benchmarks](tools/benchmarks/master-plan.md) | Fixed CPU/Metal report-only harnesses delivered; broader workload suites Draft. No universal performance claim. |
| [Tuning](tools/tuning/master-plan.md) | Local/complete-plan tuning and read-only inspection delivered, plus cache-bound repair. |
| [CLI](tools/cli/master-plan.md) | Marker only; commands await selected concrete Engine/diagnostic workflow. |

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
4. In a separately selected stream, Data 0001 authorizes Data/Text/Vision architecture before
   batching or text/image APIs. Runtime recurrent adoption still needs a concrete backend route.

Step 2 records the completed decision and Training 0004 proof, not a durable artifact. Step 3 records
completed Checkpoint 0003/0004/0004A and CPU 0024. The subsequently authorized Training-owned
Checkpoint 0005 prerequisite is Complete, not authorization for the Draft durable adapter.
Step 4 is a proposal, not a Ready implementation task.
An independent performance/shape follow-up, [CPU 0023](backends/cpu/tasks/0023-vector-erf-direct-java-shape.md),
is also Draft: the bounded vector ERF diagnostic did not demonstrate parity with direct Java.
It is not a prerequisite for unrelated Training/Checkpoint decisions or a new broad campaign.

The completed Training/Checkpoint chain remains:
`Training 0003A + Checkpoint 0001–0004 → Training 0004 → Checkpoint 0004A →`
`Training-owned Checkpoint 0005 → Checkpoint 0005A` (CPU 0024 also precedes 0004/0004A).
Training 0003 is Superseded. [ADR 0036](../design/decisions/0036-model-envelope-composition.md)
and published model.v1 envelope APIs remain implemented, not durable training recovery.

The revised Draft dependency DAG is:
`NN 0026 → Checkpoint 0005B and Training 0006 → optional Checkpoint 0006`,
with completed 0005A also feeding the adapter. NN owns Model-only semantic specification/forward
reconstruction; Checkpoint owns whole-model bytes/file APIs; Training owns recipe/run/batch/lifetime;
the adapter has exactly Training + Checkpoint api edges and consumes these public owner capabilities.
No unrelated recurrent/Data/Text task or broader Training checkpoint is a prerequisite.
Each is a cohesive capability with separate clean implementation and independent targeted review.
No mechanical type/facade/test split or speculative detailed successor cascade.

Freeze the exact initial NN/activation/loss/input/sharing matrix, canonical wire/default limits
and owner-transfer/capture/restore bridge before Ready. Initial proposal includes nested Sequential,
Linear/LayerNorm/Embedding/Conv1d/2d/3d and RELU/TANH/SIGMOID; branches/contextual layers are later
explicit choices, not a claim about all Model.define code. Inference constraints are separate from
static Training signatures. ArtifactFile v1/model.v1 remain untouched; whole-model.v1 and
training-model.v1 are new proposed kinds, not same-file compatibility with withdrawn unpublished
training-checkpoint.v1. Short loads refuse present input fingerprints until explicit verification.

Before Ready coordinate root + owning extensions contract + explanations + ADR + architecture tests
as applicable. No contract/build change occurs now. Parallel group None; coordinator owns shared
documents and verifies source-matched base/dependencies/frontier before launch. Owner focused/module
validation once; one combined source-matched repository test/Javadoc checkpoint after integration.
Proposal review performs planning links/anchors/sizes/diff only, not Java reruns/performance work.
Data cursor/RNG/scheduler/scaler stay application-owned; no exact automatic continuation, advancing
dropout stream, BatchNorm training-buffer alignment, BPTT or Conv3d gradient widening.

## Explicit non-goals and unresolved decisions

- Do not restart historical profile/certificate/proof gates as prerequisites for current work.
- Do not equate operation-kind coverage with all dtypes/layouts/gradients/strategies or speed.
- No broad performance campaign, new native provider, automatic backend discovery, rollback-mode
  change, or new numerical policy is authorized by the prior review. Checkpoint 0004's model
  save/inspect/load and direct Engine acceptance are Complete; future persistence work remains
  separately gated.
- GPU CI provisioning, large-model aggregate memory policy, new vendor routes, recurrent execution
  and missing adjoints retain their explicit decisions and owning plans.

## History

The 2026-10-13 reconciliation removed duplicated completion narratives and stale global-frontier
claims from this index. Completed and historically blocked task evidence remains in its original
briefs and Git history. Future updates change current status, dependencies and next gates only.

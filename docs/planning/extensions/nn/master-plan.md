# NN Extension Master Plan

## Goal and authority

This plan is the current map for the stateful neural-network (NN) composition layer: ownership of
modules and state, forward behavior, typed composition, standard layers, and the remaining
recurrent/data integration program. It coordinates work; it is not an architecture contract.
[`ARCHITECTURE.md`](../../../../ARCHITECTURE.md) is authoritative if this plan conflicts with it.

Read the exact applicable contract headings rather than reconstructing architecture from task
history:

- [`extensions/nn`](../../../architecture/contracts/extensions-training.md#extensionsnn) and
  [`modules/training`](../../../architecture/contracts/extensions-training.md#modulestraining) define ownership;
- [dependency rules](../../../../ARCHITECTURE.md#dependency-rules) define allowed edges; and
- [fixed recurrent scan without graph regions](../../../architecture/contracts/recurrent-scan.md#fixed-recurrent-scan-without-graph-regions)
  defines the accepted runtime-valid-length direction and its lifecycle/performance boundaries.

Focused explanations are [architecture decision record (ADR) 0007](../../../design/decisions/0007-neural-network-module-and-training-boundary.md),
[ADR 0012](../../../design/decisions/0012-fixed-recurrent-scan-without-regions.md),
[module boundaries](../../../architecture/module-boundaries.md),
[dependency rules](../../../architecture/dependency-rules.md), and the
[training graph](../../../architecture/training-graph.md). The [planning guide](../../planning-guide.md)
controls plan and task-brief format.

## Scope and non-goals

NN owns `Module`, `Parameter`, `Buffer`, named state discovery/loading, train/eval forward mode,
typed model topology, parameter initialization, and stateful layers composed from Model Tensor
semantics. It also owns current static recurrent cells/sequences, directional composition, and
rank-specific channels-first convolution layers.

NN does not own generic Tensor/operation semantics, compiler automatic differentiation, optimizer
algorithms or training sessions, persistent checkpoint transport, runtime execution, backend
storage/lowering/kernel selection, tokenization, or raw-data padding.

## Module invariants and dependencies

The following describes current ownership, including the completed NN0028 cutover.

- A module declares and owns its `Parameter` and `Buffer` wrappers, child modules, stable names,
  and local forward mode. A parameter replacement must preserve its declaration schema.
- State dictionaries are ordered in-memory Tensor-reference snapshots with strict complete
  validation before installation. NN now owns retained old model persistence glue; generic graph
  values/artifact transport belong to Checkpoint and optimizer/session state belongs to Training.
- Public parameters remain bound. Input-dependent layers reserve state privately and fail closed
  on discovery/export until one complete layer-local publication or strict load succeeds. They
  infer only input feature/channel extents; output, hidden, vocabulary, embedding, and class widths
  remain explicit schema choices.
- NN composes ordinary Model expressions and retained persistence uses public Engine materialization.
  It owns no compiler graph, backward construction, backend route or second execution pipeline.
  NN0028 composes existing prepared/session owners under the bounded NN contract.
- Random initialization uses explicit caller sources or an exactly selected per-layer algorithm
  and seed. NN owns no global, retained, or mutable random generator.
- Current recurrent sequence APIs use static Java `long[]` valid lengths, explicit carried state,
  compact outputs, and one cell parameter set across a fresh expression occurrence per represented
  step. Runtime Tensor lengths require the accepted fixed scan and truthful backend execution.
- `BatchNorm` buffer updates and `Dropout` graph-random state remain explicit forward contracts;
  they are not hidden runtime mutation.
- Conv1d/2d/3d remain separate channels-first NCW, NCHW, and NCDHW layout layers for one, two, and
  three spatial axes. Conv3d forward execution is supported, but its gradient remains fail-closed
  until Compiler 0006C or a replacement owner decision completes.

Allowed direct public production dependencies are exactly Model, Training, Checkpoint and Engine,
as enacted by Training0007/Checkpoint0008 and the current NN build. Training supplies live parameter
admission/sessions; Checkpoint/Engine support owned Models and relocated NN persistence. Forbidden
direct edges: Compiler, Runtime, Prepare and every concrete backend. NN0028 established the bounded
root/owning NN contract before implementing high-level ownership; no dependency edge changed.

## Package map

| Package | Ownership and visibility |
|---|---|
| `io.github.pho001.synaptik.nn.module` | Public module/state/mode/topology/model/sequence-construction contracts. Package-private primitives may support atomic ownership and publication without widening the API. |
| `io.github.pho001.synaptik.nn.initialization` | Public closed initialization policies and stateless eager initializers; no retained source, registry, callback model, or layer schema. |
| `io.github.pho001.synaptik.nn.layers` | Public concrete layers, cells, sequences, and typed results. Rank-shared convolution binding/initialization remains package-private implementation detail. |
| `io.github.pho001.synaptik.nn.checkpoint` | Current retained old model.v1/whole-model.v1 APIs and private NN persistence glue, moved by Checkpoint0008. |
| `io.github.pho001.synaptik.nn.functional` | Reserved planning direction only; no current production package or public API. |

Training0007 delivered Module.trainingParameters and its private live adapter. The completed NN0028
package additions are private Model.SingleTensorModel/Model.TensorListModel, package-private
OwnedModelSource (with private StateNamespace) and ModelExecutionState, all in existing nn.module.
They separate typed entry points, source/state ownership and delegated execution lifetime; no public
model/session hierarchy. Exact scope is in the [NN0028 brief](tasks/0028-owned-model-compute-training-and-persistence.md).

NN0029 plans one public `ModelBuilder` in existing nn.module, with private construction/typed-step
adapters, reusing Model.SingleTensorModel, OwnedModelSource and ModelExecutionState. No new layer
registry, model hierarchy, graph representation or dependency. Exact types and cutover are in its brief.

## Task list

The table is the complete ordered NN queue. Completed rows link to their evidence-owning task;
Draft rows without links have no detailed task brief.

| ID | Task | Status | Depends on | One-line result or intent |
|---|---|---|---|---|
| [0001](tasks/0001-module-parameter-buffer-and-forward-context-foundation.md) | Module, parameter, buffer, and forward-context foundation | Complete | Completed `modules/model` Tensor contracts; ADR 0007 | Created the model-only NN module, state wrappers, mode/context contracts, Training edge, and architecture checks. |
| [0002](tasks/0002-module-tree-ownership-and-recursive-mode-propagation.md) | Module-tree ownership and recursive mode propagation | Complete | 0001 | Added exclusive child ownership, collision-free deterministic state paths, immutable discovery snapshots, and preflighted iterative mode propagation. |
| [0003](tasks/0003-validated-parameter-and-buffer-binding-replacement.md) | Validated parameter and buffer binding replacement | Complete | 0002 | Added schema-validated direct binding replacement while preserving wrapper/name identity and explicitly excluding concurrency guarantees. |
| [0004](tasks/0004-explicit-eager-parameter-initializers.md) | Explicit eager parameter initializers | Complete | 0001–0003; completed Model eager constant/random contracts | Added stateless floating zero/one, explicit-source normal/uniform, and rank-two Glorot/Kaiming-ReLU eager initializers. |
| [0004A](tasks/0004a-parameter-update-and-traversal-hardening.md) | Parameter update and traversal hardening | Complete | 0001–0004; post-0004 code review | Added public schema-safe `Parameter.replace`, iterative identity-defended traversal, and complete fan-initializer failure contracts. |
| [0005](tasks/0005-linear-layer.md) | Linear layer | Complete | 0001, 0004, 0004A; completed model `Tensor.linear` | Added final stateful `Linear` with stable parameters and exact Model linear composition. |
| [0006](tasks/0006-layer-normalization-layer.md) | Layer normalization layer | Complete | 0001–0005; completed Model 0021 | Added affine `LayerNorm` with exact normalized Shape, typed epsilon, initialized/supplied state, and Model delegation. |
| [0007](tasks/0007-embedding-layer.md) | Embedding layer | Complete | 0006; completed Model 0019A1 | Added supplied-table `Embedding` with axis-zero Model delegation and no padding-row policy. |
| [0008](tasks/0008-batch-normalization-layer.md) | Batch normalization layer | Complete | 0007; completed Model 0021B–0021C | Added mode-sensitive affine `BatchNorm` and explicit installation of pure next-statistic expressions into stable buffers. |
| [0009](tasks/0009-dropout-layer.md) | Dropout layer | Complete | 0008; completed Model 0019B–0019B1 | Added caller-threaded graph-random state, training delegation, and identity-preserving evaluation bypass without hidden state. |
| [0010](tasks/0010-state-dictionary-and-checkpoint-contract.md) | State dictionary and checkpoint contract | Complete | 0009; stable module-tree traversal and replacement contracts | Added immutable ordered in-memory state dictionaries and strict validate-before-install loading; persistent transport stayed deferred. |
| [0011](tasks/0011-unary-tensor-module-composition-and-sequential.md) | Unary Tensor module composition and Sequential | Complete | 0010; completed Linear, LayerNorm, and Embedding unary Tensor APIs | Added narrow `UnaryTensorModule` and immutable numeric-child `Sequential`; contextual/stateful signatures remain outside it. |
| [0012](tasks/0012-vanilla-rnn-cell.md) | Vanilla tanh RNN cell | Complete | 0011; completed Model linear, ADD, and TANH expressions | Added explicit-state vanilla tanh `RnnCell` with fixed projections and optional shared bias. |
| [0013](tasks/0013-gru-cell.md) | GRU cell | Complete | 0012 | Added explicit-state reset-after `GruCell` with fixed packed gate order, interpolation, and optional input-side bias. |
| [0014](tasks/0014-lstm-cell.md) | LSTM cell | Complete | 0013 | Added explicit hidden/cell-state `LstmCell`, fixed packed equations, zero-bias policy, and typed next-state result. |
| [0015](tasks/0015-static-packed-rnn-sequence.md) | Static packed RNN sequence | Complete | 0012–0014; completed Model SELECT, GATHER, STACK, and eager INT64 leaves | Added static valid-prefix RNN packing, compact outputs, and original-order final-hidden restoration. |
| [0016](tasks/0016-static-packed-gru-sequence.md) | Static packed GRU sequence | Complete | 0015 | Applied the static one-state packing/restoration contract to GRU without a shared recurrent abstraction. |
| [0017](tasks/0017-static-packed-lstm-sequence.md) | Static packed LSTM sequence | Complete | 0014–0015 | Added static LSTM packing with compact hidden outputs and restoration of final hidden and cell state. |
| [0018](tasks/0018-typed-functional-model-topology.md) | Typed functional Model topology | Complete | 0010–0017; stable Module ownership | Added typed `Model<I,O>`, sealed definition-scoped topology, atomic named ownership, and stable state paths. |
| [0019](tasks/0019-automatic-first-forward-linear-initialization.md) | Automatic first-forward Linear initialization | Complete | 0018; exact initialization/state-dictionary decision | Added private reserved state and atomic first-forward/strict-load binding while inferring only `inFeatures`. |
| [0020](tasks/0020-automatic-recurrent-initialization-and-sequence-defaults.md) | Automatic recurrent initialization and sequence defaults | Complete | 0019; current recurrent cell/sequence and Model provenance contracts | Unified closed initialization policy, automatic recurrent input-width binding, deterministic seeds, and zero-state/all-valid sequence conveniences. |
| [0020A](tasks/0020a-initialized-embedding.md) | Initialized Embedding | Complete | 0020; current Embedding and tokenizer/schema boundaries | Added eager initialized Embedding with explicit schema/policy/seed and ordinary trainable semantics for every row. |
| [0020B](tasks/0020b-stateless-standard-module-factory.md) | Stateless standard ModuleFactory | Complete | 0020–0020A | Added fresh standard Embedding/Linear/RNN/GRU/LSTM recipes without ownership, registry, or retained configuration. |
| [0020C](tasks/0020c-bidirectional-static-recurrent-composition.md) | Type-safe bidirectional static recurrent composition | Complete | 0020–0020B; stable static sequence/result and Model gather/composition contracts | Added concrete bidirectional RNN/GRU/LSTM containers with independent state, valid-prefix reversal, aligned forward-first concat, and typed final states. |
| [0021A](tasks/0021a-fixed-recurrent-scan-architecture-decision.md) | Fixed recurrent-scan architecture decision | Complete | 0020C; current flat Model/Compiler graph, Prepare, Runtime, and Engine contracts | Accepted fixed flat multi-output RNN/GRU/LSTM scan semantics, lifecycle ownership, fail-closed backpropagation-through-time boundary, and migration constraints. |
| 0021B | Fixed recurrent-scan implementation program | Draft | 0021A; Model 0025E; Compiler 0006A; Engine 0001–0002; truthful concrete-backend coverage | Reconcile completed Model/Compiler adoption with the remaining ordinary lifecycle and concrete-backend execution program; runtime lengths must not specialize graph structure. |
| 0022 | Valid-length recurrent API and Data integration | Draft | 0021B; Data 0001–0002 architecture and valid-length contracts | Add the Data-owned runtime-length NN API only after executable scan coverage; deliberately decide the static `long[]` compatibility path. |
| 0023 | Arbitrary dense validity-mask semantics | Draft | 0022; concrete attention/loss/recurrent consumer | Add a Boolean mask only for a proven holes-capable consumer, never as duplicate right-padding metadata or a skipped-work claim. |
| 0024 | Typed model/recurrent/data integration checkpoint | Draft | 0020–0022; 0023 only if selected; Checkpoint model-state and Training publication readiness | Validate state, binding/checkpoint compatibility, variable batches, continuation, autograd/training handoff, documentation, and integration boundaries. |
| [0025](tasks/0025-channels-first-conv1d-conv2d-conv3d-layers.md) | Channels-first Conv1d, Conv2d, and Conv3d layers | Complete | 0020B; Model 0025G–0025H; Compiler 0006B and 0006B6; CPU 0008–0008A; Engine 0008 | Added rank-specific unary convolution layers/factory recipes with group-aware initialization, atomic inferred-channel binding, strict loading, and forward-only Conv3d boundary. |
| [0025A](tasks/0025a-dimensional-convolution-user-capability-checkpoint.md) | Dimensional-convolution user-capability checkpoint | Complete | 0025; Engine 0004; CPU 0008A | Validated strict-loaded and factory convolution through public one-shot/reusable Engine workflows, numerical oracles, lifecycle ownership, and Conv3d backward rejection. |
| [0025B](tasks/0025b-public-model-dependency-export.md) | Export public Model dependency | Complete | Existing NN facade; project review | Correct Gradle API visibility and validate an isolated NN-only consumer. |
| [0026](tasks/0026-declarative-model-specification.md) | Declarative model specification and reconstruction | Complete | Verified Complete 0025/0025A/0025B; accepted NN matrix/contract | Describe/reconstruct/strict-load delivered with final reviews and coordinator gates; no persistent codec or new dependency. |
| 0027 | Semantic recurrent description and reconstruction | Draft | NN0026/CP0005B Complete; P2 review and any separately selected recurrent scope | Held despite CP0005D planning completion; no bytes/versions or detailed brief. |
| [0028](tasks/0028-owned-model-compute-training-and-persistence.md) | Owned ordinary tensor Model workflow | Complete | Engine0021; Training0007; Compiler0008; Checkpoint0008; Training0008 — Complete | Accepted 2026-10-19 after definitive independent approval and coordinator native-enabled root gate; evidence in brief. |
| [0029](tasks/0029-fluent-model-builder-and-checkpoint-independent-execution.md) | Fluent Model builder and checkpoint-independent execution | Complete | NN0028; Checkpoint0009; CPU0025; Compiler0009 — Complete | Planck final APPROVE, zero findings; worker/integration evidence and main's source-matched native root gate passed (2026-10-19). |

## Dependency DAG and authorized frontiers

Engine0021 + Training0007 + Compiler0008 + Checkpoint0008 + Training0008 → NN0028.
Compiler0006B11 + Compiler0008 + CPU0025 → Compiler0009 (all Complete).
NN0028 + Checkpoint0009 + CPU0025 + Compiler0009 → NN0029 → main's shared native root gate: Complete.
Authorized NN frontier: None. Neither prerequisite repair depends on NN0029; the shared gate passed
on the unchanged reviewed snapshot (2026-10-19), including UP-TO-DATE report evidence.
Main accepted the final independent review and verified launch on 2026-10-19; user authorized implementation.
The separate [Checkpoint0009 activation-source packet](../checkpoint/tasks/0009-owned-model-activation-source-and-persistence.md)
is Complete, accepted 2026-10-19 after Archimedes final APPROVE and main's integration/native root gates.
NN remains the unchanged owned-source consumer with
targeted Javadoc/tests and one direct-oracle MLP integration regression. Compiler/Checkpoint jointly
own the completed eight-kind/selective-v2 capability; NN0028 and the selected program stay Complete.
CP0009 introduced no NN implementation task, lifecycle change or new public type. Its final authorized
Checkpoint/Compiler/NN frontiers: None. Checkpoint0009's brief owns final evidence; no follow-up remains.
NN0029 conflicts: NN API/lifecycle, root/NN authority and shared documents; none active, main verified.
Parallel group: None. Common base: `7e880fb1044168043e45b10419a570d64c5283ae` plus the full dirty
snapshot: accepted CP0009, frozen NN partial and completed CPU0025/Compiler0009. Integration order:
reviewed [Compiler0009](../../modules/compiler/tasks/0009-static-select-and-slice-logical-layout-closure.md)
→ main's verified NN resume (2026-10-19) → finish NN implementation → independent Class C review → main's single native root gate.
Main owns shared-document integration; the NN0029 brief owns exact metadata, signatures, acceptance,
validation, frozen evidence and current resume verification. Main rechecked approval, completed/reviewed
predecessors, unchanged authority/full dirty base and inactive conflicts with no other writer.
Compiler frontier: None. NN acceptance/scope unchanged; no forced contiguous copies or new hierarchy.
No Checkpoint widening or scan/Data/NN0027 prerequisite; earlier held streams remain Draft.

## Proposed whole-model checkpoint program

[ADR 0038](../../../design/decisions/0038-declarative-model-and-training-resume.md#nn-decision-selected-for-review)
revises the primary product to automatic usable-model load, then inference or owned Training resume.
0026 is Complete after final independent reviews and coordinator validation.

Earlier proposed serial DAG (held):
`NN0026 Complete → CP0005B Complete | HOLD: NN0027 → CP0005C → Training0006 → optional CP0006`.
[CP0005D](../checkpoint/tasks/0005d-portable-graph-persistence-decision.md) is Complete for bounded
planning/review after independent review, the sole P2 correction and coordinator acceptance;
[ADR0039](../../../design/decisions/0039-portable-graph-model-persistence.md) records the accepted decision; root/scoped contracts enact it.
The user approved one model.compute/train/save/load direction for manual/file tensors and generic
Training/Checkpoint modules. The [cross-area program](../../model-execution-refactor.md) owns the
phases, exact edges and independently approved P2 planning packet. NN stays extensions/nn, supplies the
live named-parameter adapter and receives NN-specific Checkpoint glue after removal of reverse edges.
Same-signature execution targets Engine PreparedExecution plus Engine.runValues with fresh per-run
borrowed storage, not the InferenceSession/Engine.session API removed by Engine0021. Model's
high-level compute/train hides preparation; no replacement runner/session facade. Normal use
does not reconstruct via outputs(actualInputs) or mutate shared placeholders.

Current Model<I,O> permits arbitrary Java forward and Module.train() is mode-only; both remain.
P2 selects defineTensor(signature, definition) and load(Path) returning Model<Tensor,Tensor>,
with compute(Tensor), configured train(Tensor,Tensor), save and close. No manual forward/leaf
enumeration/second Model is required for ordinary NN use. The one-shot definition produces a
source graph; discard its callback after private leaf acquisition/reconstruction, so a captured
external weight cannot escape into later symbolic forward. Explicit symbolic forward reconstructs
the owned source cold; compute/train never do so per batch. Legacy Model.define stays unchanged.
Advanced graph/loadGraph supports Tensor lists, not arbitrary Java-record serialization.
Detached HostTensorValue results are not generic O; no cast or new public model/result hierarchy.
Keep ModelSpecification/DeclarativeModel and flexible v1 rank/batch behavior until explicit migration.
Source restoration creates ordinary Tensor expressions/weights before compile without Engine/batch.
No loaded-model hierarchy, second execution pipeline or per-RNN recipe prerequisite.

Implemented binding: `Model.load(path[, limits]).withEngine(engine)` returns the exact same
ordinary Model, as does application-built Model binding. It borrows one Engine for compute/train;
existing explicit borrowed-Engine overloads remain. Otherwise Engine.standard() is lazy/model-owned,
currently CPU-only, without discovery/global state. Binding freezes at first inference/training
preparation setup, including failed setup, before Engine/native/compile/prepare work. No later
rebinding or automatic reprepare after rebinding; null/closed candidates reject without mutation.
The [exact admission, errors and configureTraining timing](../../../design/decisions/0039-portable-graph-model-persistence.md#target-model-engine-binding)
are implemented in completed NN0028. compute(batch) returns detached HostTensorValue, not a request/wrapper;
no Model methods or lifecycle are implemented by Training0007.

NN0028 is Complete; authorized NN frontier: None. Engine0020's backend/native-only cutover is Complete after
integration, independent review and combined validation. The user's later five choices are now
accepted. Engine0021 delivered the thin-session removal, authority cutover and mechanical Training
migration; independent Class C approval and the native-enabled test/Javadoc gate passed. Its brief
owns evidence. Training0007, including its atomic NN adapter, is Complete after final independent
APPROVE and the coordinator gate. Compiler0008, Checkpoint0008 and Training0008 are also Complete
after final independent reviews and coordinator gates; the selected program has no remaining frontier.
NN0028's implementation is accepted. Training0007 removed its reverse edge and added
Module.trainingParameters; Checkpoint0008 moved the nine NN glue classes into nn.checkpoint and
removed Checkpoint→NN. Current dependencies are acyclic; bounded high-level NN ownership is now enacted.
NN0028 privately acquires external weights once, keeps one working set for compute/train/save,
creates Engine.standard lazily (or borrows explicit Engine), and rejects stale mode/binding changes.
One high-level operation at a time; detached compute/TrainingStep values outlive close.
Known source/signature enables save-before-compute; model-only save is not optimizer/RNG resume.
Package additions are private typed model implementations/state namespaces in nn.module, not a
public TensorModel or execution adapter. The generic boundary types belong to Training/Checkpoint.
Shared documents integrate serially under the model-execution coordinator; parallel group None,
common base is the revision recorded above, reverified before launch. Exact DAG and
migration map live in the program; the [Training0007 brief](../training/tasks/0007-generic-live-parameter-admission.md)
records final independent approval and successful coordinator validation; NN0028 is its completed successor.
NN0027 owns only semantic description/reconstruction using existing static Rnn/Gru/Lstm capabilities;
CP0005C owns explicit versioned recurrent definition wire/whole-save/load adoption. This is an owner split,
not a hard dependency on optimized RecurrentScan/backend BPTT/Data. Authorized NN frontier: None.
NN0026's API/matrix and NN-only root/contract/explanation/ADR packet are accepted after final
independent review; coordinator Ready verification is recorded in the brief. Downstream
CP0005B's whole-model contract/API/wire/default packet is accepted after final independent approval
and coordinator verification; CP0005B is Complete with final reviews/gates (2026-10-15), no new
frontier. NN0027/CP0005C/Training0006/adapter remain held Draft; their optional policies are not P2 gates.

NN0027 ordering metadata (held, revalidate after remaining ADR0039 semantic decisions): Depends on NN0026 and CP0005B; Conflicts with shared NN semantic/API
contracts and CP0005C/Training0006/adapter0006; Parallel group None; Common base revision
`f892cd3fbde4f8b7722b7eb5b14f4b0a5f3b2669` (reverify after CP integration); Integration order
after CP0005B and before CP0005C; Integration validation future NN semantic reconstruction tests
and semantic round trips plus architecture at its join; Shared-document integration owner NN/Training/Checkpoint
coordinator. Input/output/default carried-state policy is future before a detailed brief or Ready.
CP0005C format version and Training gradient admission are separate future decisions. Current
whole-model v1 rejects all recurrent IDs; NN0027 owns no persistent format or I/O.

Selected package map: ModelSpecification and DeclarativeModel in existing nn.module, distinct from
the ModelDefinition Java callback; closed Activation in nn.layers. Initial scope: nested Sequential,
Linear/LayerNorm/
Embedding/Conv1d/2d/3d and closed RELU/TANH/SIGMOID steps, not all Java models.
NN owns actual attrs/signatures/wiring/sharing. Children alone do not prove forward flow.
Inference constraints need not freeze batch dimensions; Training signatures are independently static.
Inference Tensor-identity aliases reconstruct faithfully; describe reads current bindings after
legal replacement/load, without cached alias authority. Training rejects repeated parameter-path
Tensor identities/overlapping stores, not every repeated forward use of one uniquely owned layer.
Declarative branches/contextual adapters are explicit successors, not detailed new tasks.
Descriptor-bound skeletons retain source numeric paths without a wrapper prefix and use supplied
state/strict load without random weights; focused tests extend that evidence. The brief records
exact records, typed configuration access and Linear.fromState guard policy; Model-only edges and
existing generic APIs stay intact. 4096/depth64 bound cold-tree processing, not backend capability.

NN0026 itself changed no dependency. Training0007 subsequently added the public generic Training
edge; Checkpoint0008 enacted the current public Checkpoint/Engine edges for retained NN persistence.
The coordinator owns shared documents; each capability uses separate clean implementation/review.
Parallel group None; accepted contract base `4472320ceb5498c46a661af549d6de74aa27ab6a`.
Implementation, final reviews and coordinator validation are complete; evidence is in the task brief.
Owner NN test/Javadoc once; coordinator architecture and narrow CPU integration after review, full
source-matched checkpoint only at the selected program join. No native/performance campaign.
Exact metadata and 2026-10-14 dependency/base/contracts/review/conflict verification live in the Ready brief.

## Milestones and current frontier

- Complete: foundation and state lifecycle (0001–0011), recurrent cells/static sequences and typed
  topology (0012–0020C), the fixed-scan architecture decision (0021A), and dimensional
  convolution plus its user checkpoint (0025–0025A).
- Draft: 0021B–0024. None has a detailed task brief. Audit repair 0025B is Complete.
- Complete: NN0026 with final independent reviews and coordinator validation.
  NN0027 remains Draft; CP0005B is Complete. Engine0020's completed layout cutover does not
  authorize NN execution.
- Complete: NN0028, accepted 2026-10-19; evidence in its brief.
- Complete: NN0029 after Planck final APPROVE, zero findings, 22 focused/424 NN/3 Fluent integration
  tests and main's native root gate (2026-10-19). Authorized NN frontier: None.
- NN0027 is the earlier Draft recurrent follow-up, held with CP0005C/Training0006 held as an optional older proposal,
  not a prerequisite of NN0028; no scan/backend BPTT/Data gate.
- 0021B remains a separate Draft execution candidate only after a planner revalidates its cross-area
  program and truthful concrete-backend coverage. It is not implementation authorization;
  Model 0032 is Complete, and the [roadmap](../../roadmap.md) owns current cross-area frontiers.
- Completed NN work used recorded roadmap interleaves. NN 0016/0017 were a bounded parallel pair,
  and 0025/0025A were an explicit out-of-order convolution branch ahead of unrelated Draft
  0021B–0024. Those historical exceptions authorize no further NN work.

Earlier prose that named CPU 0006D or tools/tuning 0002 as the global frontier, called 0010/0011
or 0025A Draft/Ready, treated 0021A's planned Model surface as still wholly unpublished, or called
0021B–0024 the active frontier has been normalized to the task table, linked task results, current
dependency plans, and roadmap above.

## Live gates, decisions, and risks

- **Nonblocking cold-prepare follow-up:** a development 3-step stacked-LSTM train was interrupted
  in active CPU topological preparation; reduced 2-step integration trains, and required 3-step RNN
  training/all-family 3-step stack numerics pass. Review found no acceptance gap, error or proved
  regression. Characterize larger-case preparation cost later; no Ready task, benchmark campaign or gate.
- **0021B execution gate:** [Model 0025E–0025F](../../modules/model/master-plan.md) and
  [Compiler 0006A](../../modules/compiler/master-plan.md) are Complete, and current
  [Engine](../../modules/engine/master-plan.md) has caller-input/publication lifecycles. No
  concrete backend currently claims recurrent-scan execution. The future brief must identify an
  honest backend route and verify whether existing Planning/Prepare/Runtime contracts suffice.
- **0022 Data gate:** [Data 0001–0002](../data/master-plan.md) remain Draft; Data/Text/Vision are
  not authorized modules until their coordinated architecture decision. Ordinary right padding
  has one canonical valid length per batch row; padding lengths and dense masks are derived, not
  stored second sources of truth.
- **Scan boundary:** specific runtime length values must not change Model topology or compiled
  graph structure. Invalid lengths fail before published-result mutation, padded coordinates are
  not recurrent inputs, and dense masking must not be described as skipped recurrent work. Static
  sequence APIs remain unchanged until 0022 makes an explicit compatibility/migration decision.
- **Differentiation gate:** fixed RecurrentScan backpropagation through time remains fail-closed and
  belongs to Compiler. Ordinary static unrolled recurrence depends on normal constituent-operation
  adjoints; it is not covered by a blanket recurrent-autograd prohibition. Conv3d gradients remain
  fail-closed while Compiler 0006C is Draft. NN/Training must not imply either blocked capability.
- **Persistence/training gate:** NN owns in-memory state paths and strict Tensor-reference
  load, with declarative reconstruction delivered in 0026. Durable artifacts belong to
  [Checkpoint](../checkpoint/master-plan.md), whose published explicit APIs remain implemented;
  optimizer/session/update orchestration belongs to [Training](../training/master-plan.md).
- **Still-open API decisions:** select a persistent codec only with a concrete consumer; add a
  frozen padding row only with an owning gradient/replacement/update contract; add a shared
  recurrent abstraction, new initialization preset, configurable convolution fan value, or
  arbitrary mask only when a concrete consumer proves the contract; likewise defer public state-
  schema inspection or model-wide deferred-binding lifecycle until a consumer requires it.
- **Primary risks:** reverse dependencies or access to Compiler/Runtime/Prepare/backend internals;
  bypassing the bounded owned Model lifecycle through public Training/Checkpoint/Engine edges; hidden runtime or random
  state; partially published inferred parameters; confusing static unrolling with runtime scan;
  specializing graphs to host lengths; erasing LSTM cell state in a shared abstraction; or moving
  tokenization/padding/checkpoint transport into NN.

## History and update policy

Detailed completion evidence, validation commands and results, file inventories, context IDs,
audit trails, and superseded frontier narratives remain in linked task files and Git history.
They are not default executor input. Current source, focused tests, authoritative contracts, this
table, and the current task brief take precedence over historical narrative.

New or materially replanned tasks use the [compact task-brief format](../../planning-guide.md#task-brief-format).
Completed tasks need no retrospective rewrite. Update this plan only for ownership/package
direction, task order or status, live dependencies/gates, risks, milestones, or frontier changes.

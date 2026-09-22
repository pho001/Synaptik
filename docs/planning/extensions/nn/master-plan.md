# NN Extension Master Plan

## Goal and authority

This plan is the current map for the stateful neural-network (NN) composition layer: ownership of
modules and state, forward behavior, typed composition, standard layers, and the remaining
recurrent/data integration program. It coordinates work; it is not an architecture contract.
[`ARCHITECTURE.md`](../../../../ARCHITECTURE.md) is authoritative if this plan conflicts with it.

Read the exact applicable contract headings rather than reconstructing architecture from task
history:

- [`extensions/nn`](../../../architecture/contracts/extensions-training.md#extensionsnn) and
  [`extensions/training`](../../../architecture/contracts/extensions-training.md#extensionstraining) define ownership;
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

- A module declares and owns its `Parameter` and `Buffer` wrappers, child modules, stable names,
  and local forward mode. A parameter replacement must preserve its declaration schema.
- State dictionaries are ordered in-memory Tensor-reference snapshots with strict complete
  validation before installation; bytes, codecs, files, versions, and optimizer state remain
  downstream concerns.
- Public parameters remain bound. Input-dependent layers reserve state privately and fail closed
  on discovery/export until one complete layer-local publication or strict load succeeds. They
  infer only input feature/channel extents; output, hidden, vocabulary, embedding, and class widths
  remain explicit schema choices.
- NN composes ordinary Model expressions. It owns no compiler graph, backward construction,
  prepared executable, run state, backend route, or execution service.
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

Allowed production dependency: `modules/model` only.

Forbidden dependencies: `extensions/training`, `modules/compiler`, `modules/runtime`,
`modules/prepare`, `modules/engine`, and every concrete backend. Training depends on NN, never the
reverse.

## Package map

| Package | Ownership and visibility |
|---|---|
| `io.github.pho001.synaptik.nn.module` | Public module/state/mode/topology/model/sequence-construction contracts. Package-private primitives may support atomic ownership and publication without widening the API. |
| `io.github.pho001.synaptik.nn.initialization` | Public closed initialization policies and stateless eager initializers; no retained source, registry, callback model, or layer schema. |
| `io.github.pho001.synaptik.nn.layers` | Public concrete layers, cells, sequences, and typed results. Rank-shared convolution binding/initialization remains package-private implementation detail. |
| `io.github.pho001.synaptik.nn.functional` | Reserved planning direction only; no current production package or public API. |

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

## Milestones and current frontier

- Complete: foundation and state lifecycle (0001–0011), recurrent cells/static sequences and typed
  topology (0012–0020C), the fixed-scan architecture decision (0021A), and dimensional
  convolution plus its user checkpoint (0025–0025A).
- Draft: 0021B–0024. None has a detailed task brief, and no NN task is `Ready` or `In progress`.
- 0021B is the next ordered NN Draft candidate only after a planner revalidates its now-partially
  completed cross-area program and identifies truthful concrete-backend coverage. It is not
  implementation authorization; the [roadmap](../../roadmap.md) currently selects Metal 0004.
- Completed NN work used recorded roadmap interleaves. NN 0016/0017 were a bounded parallel pair,
  and 0025/0025A were an explicit out-of-order convolution branch ahead of unrelated Draft
  0021B–0024. Those historical exceptions authorize no further NN work.

Earlier prose that named CPU 0006D or tools/tuning 0002 as the global frontier, called 0010/0011
or 0025A Draft/Ready, treated 0021A's planned Model surface as still wholly unpublished, or called
0021B–0024 the active frontier has been normalized to the task table, linked task results, current
dependency plans, and roadmap above.

## Live gates, decisions, and risks

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
- **Differentiation gate:** recurrent backpropagation through time remains fail-closed and belongs
  to Compiler. Conv3d gradients likewise remain fail-closed while Compiler 0006C is Draft. NN and
  Training must not imply either capability.
- **Persistence/training gate:** NN owns only in-memory state paths and strict Tensor-reference
  load. Durable artifacts belong to the Draft [Checkpoint plan](../checkpoint/master-plan.md);
  optimizer/session/update orchestration belongs to [Training](../training/master-plan.md).
- **Still-open API decisions:** select a persistent codec only with a concrete consumer; add a
  frozen padding row only with an owning gradient/replacement/update contract; add a shared
  recurrent abstraction, new initialization preset, configurable convolution fan value, or
  arbitrary mask only when a concrete consumer proves the contract; likewise defer public state-
  schema inspection or model-wide deferred-binding lifecycle until a consumer requires it.
- **Primary risks:** dependency reversal into Training/Engine/backends; hidden runtime or random
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

# [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) Extension Master Plan

> **Editorial reading context — not a status update.** There are two different structures:
> the NN module tree owns named state and children, while Tensor expressions describe which
> computations depend on that state. A forward call builds the latter; it is not automatically
> a numerical run or optimizer step. TrainingSession coordinates Engine execution and updates;
> Checkpoint transports supported source/value state. The application connects those boundaries
> and keeps borrowed resources alive. Today's [composition explanation and diagram](../../../reference/nn/composition.md#current-nn-typed-model-composition-contract),
> [owned execution](../../../reference/nn/owned-model.md#owned-tensor-model-capability) and
> [session roles](../../../reference/training/sessions.md#purpose-and-implementation-status)
> provide current context without replacing the capability gates recorded below.

## Goal and authority

This plan is the current map for the stateful neural-network ([NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)) composition layer: [ownership](../../../glossary.md#term-ownership-meanings) of
[modules](../../../glossary.md#term-module) and state, forward behavior, typed composition, standard layers, and the remaining
recurrent/data integration program. It [coordinates](../../../glossary.md#axis-rank-scalar-vector-and-matrix) work; it is not an [architecture contract](../../../glossary.md#architecture-contract).
[`ARCHITECTURE.md`](../../../../ARCHITECTURE.md) is authoritative if this plan conflicts with it.

Read the exact applicable contract headings rather than reconstructing architecture from task
history:

- [`extensions/nn`](../../../architecture/contracts/extensions-training.md#extensionsnn) and
  [`modules/training`](../../../architecture/contracts/extensions-training.md#modulestraining) define ownership;
- [dependency rules](../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([dependency glossary](../../../glossary.md#term-dependency))</span> define allowed edges; and
- [fixed recurrent scan without graph regions](../../../architecture/contracts/recurrent-scan.md#fixed-recurrent-scan-without-graph-regions)<span data-glossary-navigation="true"> ([scan glossary](../../../glossary.md#term-cumulative-scan); [graph glossary](../../../glossary.md#graph))</span>
  defines the accepted runtime-valid-length direction and its lifecycle/performance boundaries.<span data-glossary-navigation="true"> ([runtime glossary](../../../glossary.md#term-runtime); [lifecycle glossary](../../../glossary.md#lifecycle))</span>

Focused explanations are [architecture decision record (ADR) 0007](../../../design/decisions/0007-neural-network-module-and-training-boundary.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../glossary.md#adr-ci-and-qualification))</span><span data-glossary-navigation="true"> ([architecture decision record glossary](../../../glossary.md#term-architecture-decision-record-adr))</span>,
[ADR 0012](../../../design/decisions/0012-fixed-recurrent-scan-without-regions.md),
[module boundaries](../../../architecture/module-boundaries.md),
[dependency rules](../../../architecture/dependency-rules.md), and the
[training graph](../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([training graph glossary](../../../glossary.md#training-graph))</span>. The [planning guide](../../planning-guide.md)<span data-glossary-navigation="true"> ([planning glossary](../../../glossary.md#term-planning))</span>
controls plan and task-brief format.

## [Scope](../../../glossary.md#term-scope) and non-goals

[NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) owns [`Module`](../../../glossary.md#term-module), [`Parameter`](../../../glossary.md#term-parameter), [`Buffer`](../../../glossary.md#term-buffer), named state discovery/loading, train/eval [forward mode](../../../glossary.md#term-forward-mode),
typed [model](../../../glossary.md#term-model) [topology](../../../glossary.md#term-topology), parameter initialization, and stateful layers composed from Model [Tensor](../../../glossary.md#tensor)
semantics. It also owns current static recurrent cells/sequences, directional composition, and
[rank](../../../glossary.md#term-rank)-specific channels-first [convolution](../../../glossary.md#term-channels-first-convolution-layers) layers.

NN does not own generic Tensor/[operation](../../../glossary.md#operation) semantics, [compiler](../../../glossary.md#term-compiler) [automatic differentiation](../../../glossary.md#autograd), [optimizer](../../../glossary.md#term-optimizer)
algorithms or training sessions, persistent [checkpoint](../../../glossary.md#term-checkpoint) transport, [runtime](../../../glossary.md#term-runtime) execution, [backend](../../../glossary.md#backend)
storage/[lowering](../../../glossary.md#lowering)/[kernel](../../../glossary.md#kernel) selection, [tokenization](../../../glossary.md#frozen-whitespace-tokenizer), or raw-data [padding](../../../glossary.md#padding).

## [Module](../../../glossary.md#term-module) invariants and dependencies

The following describes current [ownership](../../../glossary.md#term-ownership-meanings), including the completed NN0028 cutover.

- A module declares and owns its [`Parameter`](../../../glossary.md#term-parameter) and [`Buffer`](../../../glossary.md#term-buffer) wrappers, child modules, stable names,
  and local forward mode. A parameter replacement must preserve its declaration schema.<span data-glossary-navigation="true"> ([forward mode glossary](../../../glossary.md#term-forward-mode); [schema glossary](../../../glossary.md#term-schema))</span>
- State dictionaries are ordered in-memory Tensor-reference snapshots with strict complete
  validation before installation. NN now owns retained old model persistence glue; generic graph
  values/artifact transport belong to Checkpoint and optimizer/session state belongs to Training.<span data-glossary-navigation="true"> ([Tensor glossary](../../../glossary.md#tensor); [validation glossary](../../../glossary.md#term-validation); [graph glossary](../../../glossary.md#graph); [artifact glossary](../../../glossary.md#term-artifact); [optimizer glossary](../../../glossary.md#term-optimizer))</span>
- Public parameters remain bound. Input-dependent layers reserve state privately and fail closed
  on discovery/export until one complete layer-local publication or strict load succeeds. They
  infer only input feature/channel extents; output, hidden, vocabulary, embedding, and class widths
  remain explicit schema choices.<span data-glossary-navigation="true"> ([fail closed glossary](../../../glossary.md#admission-preflight-and-fail-closed-validation); [publication glossary](../../../glossary.md#term-publication); [feature glossary](../../../glossary.md#term-feature); [channel glossary](../../../glossary.md#term-channel); [embedding glossary](../../../glossary.md#embedding))</span>
- NN composes ordinary Model expressions and retained persistence uses public Engine materialization.
  It owns no compiler graph, backward construction, backend route or second execution pipeline.
  NN0028 composes existing prepared/session owners under the bounded NN contract.<span data-glossary-navigation="true"> ([materialization glossary](../../../glossary.md#materialization); [backend route glossary](../../../glossary.md#backend-route))</span>
- Random initialization uses explicit caller sources or an exactly selected per-layer algorithm
  and seed. NN owns no global, retained, or mutable random generator.<span data-glossary-navigation="true"> ([seed glossary](../../../glossary.md#term-seed); [mutable glossary](../../../glossary.md#term-mutable))</span>
- Static recurrent `forward` APIs use Java `long[]` lengths, explicit carried state and compact
  outputs, sharing one cell across represented steps. Runtime Tensor lengths use fixed forwardScan
  or bounded F32/F64 RNN forwardGeneralScan with independently qualified backend execution.<span data-glossary-navigation="true"> ([F32 glossary](../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../glossary.md#floating-point-precision-and-rounding); [RNN glossary](../../../glossary.md#term-recurrent-neural-network-rnn); [backend glossary](../../../glossary.md#backend))</span>
- `BatchNorm` buffer updates and [`Dropout`](../../../glossary.md#dropout--inverted-dropout) graph-random state remain explicit forward contracts;
  they are not hidden runtime mutation.
- Conv1d/2d/3d remain separate channels-first NCW, NCHW, and NCDHW layout layers for one, two, and
  three spatial axes. Conv3d forward execution is supported, but its gradient remains fail-closed
  until Compiler 0006C or a replacement owner decision completes.<span data-glossary-navigation="true"> ([NCW glossary](../../../glossary.md#term-ncw); [NCHW glossary](../../../glossary.md#term-nchw); [NCDHW glossary](../../../glossary.md#conv3d--ncdhw); [layout glossary](../../../glossary.md#layout); [spatial glossary](../../../glossary.md#term-spatial-axes); [axes glossary](../../../glossary.md#term-axis); [Conv3d glossary](../../../glossary.md#conv3d--ncdhw); [gradient glossary](../../../glossary.md#term-gradient); [fail-closed glossary](../../../glossary.md#term-fail-closed); [owner glossary](../../../glossary.md#term-owner))</span>

Allowed direct public production dependencies are exactly [Model](../../../glossary.md#term-model), Training, [Checkpoint](../../../glossary.md#term-checkpoint) and [Engine](../../../glossary.md#term-engine),
as enacted by Training0007/Checkpoint0008 and the current [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) build. Training supplies live parameter
[admission](../../../glossary.md#term-admission)/sessions; Checkpoint/Engine support owned Models and relocated NN [persistence](../../../glossary.md#term-persistence). Forbidden
direct edges: [Compiler](../../../glossary.md#term-compiler), [Runtime](../../../glossary.md#term-runtime), [Prepare](../../../glossary.md#prepare) and every [concrete backend](../../../glossary.md#concrete-backend). NN0028 established the bounded
root/owning NN contract before implementing high-level ownership; no [dependency edge](../../../glossary.md#term-dependency-edge) changed.

## Package map

| Package | [Ownership](../../../glossary.md#term-ownership-meanings) and visibility |
|---|---|
| `io.github.pho001.synaptik.nn.module` | Public [module](../../../glossary.md#term-module)/state/mode/[topology](../../../glossary.md#term-topology)/[model](../../../glossary.md#term-model)/sequence-construction contracts. Package-private primitives may support [atomic](../../../glossary.md#resource-transaction-and-rollback) ownership and [publication](../../../glossary.md#term-publication) without [widening](../../../glossary.md#term-widening) the [API](../../../glossary.md#term-application-programming-interface-api). |
| `io.github.pho001.synaptik.nn.initialization` | Public closed initialization policies and stateless [eager](../../../glossary.md#term-eager-evaluation) initializers; no retained source, [registry](../../../glossary.md#term-registry), [callback](../../../glossary.md#term-callback) model, or layer [schema](../../../glossary.md#term-schema). |
| `io.github.pho001.synaptik.nn.layers` | Public concrete layers, cells, sequences, and typed results. [Rank](../../../glossary.md#term-rank)-shared [convolution](../../../glossary.md#term-channels-first-convolution-layers) [binding](../../../glossary.md#term-binding)/initialization remains package-private implementation detail. |
| `io.github.pho001.synaptik.nn.checkpoint` | Current retained old model.v1/whole-model.v1 APIs and private [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [persistence](../../../glossary.md#term-persistence) glue, moved by Checkpoint0008. |
| `io.github.pho001.synaptik.nn.functional` | Reserved [planning](../../../glossary.md#term-planning) direction only; no current production package or public API. |

Training0007 delivered Module.trainingParameters and its private live adapter. The completed NN0028
package additions are private Model.SingleTensorModel/Model.TensorListModel, package-private
OwnedModelSource (with private StateNamespace) and ModelExecutionState, all in existing nn.module.
They separate typed entry points, source/state ownership and delegated execution [lifetime](../../../glossary.md#term-lifetime); no public
model/session hierarchy. Exact [scope](../../../glossary.md#term-scope) is in the [NN0028 brief](tasks/0028-owned-model-compute-training-and-persistence.md).

NN0029 plans one public `ModelBuilder` in existing nn.module, with private construction/typed-step
adapters, reusing Model.SingleTensorModel, OwnedModelSource and ModelExecutionState. No new layer
registry, model hierarchy, [graph](../../../glossary.md#graph) representation or [dependency](../../../glossary.md#term-dependency). Exact types and cutover are in its brief.

NN0022 Complete delivers additive [runtime](../../../glossary.md#term-runtime)-length methods on existing concrete Sequences, reusing Model
[scan](../../../glossary.md#term-cumulative-scan) carriers, current cell [Parameter](../../../glossary.md#term-parameter) wrappers and atomic initialization. New production types/packages:
None. Exact API/initialization choices accepted 2026-10-20; no Model runner/session or Data dependency.

Complete [NN0022B](tasks/0022b-runtime-length-scan-model-workflow.md) delivers explicit scan recipes,
finalHidden and list-typed buildWithLengths, plus generic two-input/one-output Model.compute convenience.
New production types/packages: None. The user approved the exact API and Class C [inference](../../../glossary.md#term-inference)-only
contract cutover on 2026-10-20; independent reviews and the coordinator root gate passed. No Data prerequisite.

[NN0030 approved decision](tasks/0030-general-scan-owned-model-admission-decision.md) selects existing
Model/[Engine](../../../glossary.md#term-engine)/Training/[Checkpoint](../../../glossary.md#term-checkpoint) [owners](../../../glossary.md#term-owner) only; no new production type/package/API selected.
[Compiler0012](../../modules/compiler/tasks/0012-general-scan-cold-source-reconstruction.md#result) is Complete
after independent code/docs APPROVE and main's final gates (2026-10-23).
[CP0012](../checkpoint/tasks/0012-general-scan-in-memory-copy-and-rebind.md#result) is Complete for
attachment-preserving in-memory copy/[rebind](../../../glossary.md#capture-graph-ports-and-reconstruction) with body wire closed; its earlier root environment exception
is historical. [NN0031](tasks/0031-general-scan-owned-model-workflow.md#result) is Complete after clean
implementation, independent review, final NN module/[Javadoc](../../../glossary.md#term-javadoc) and main's configured native-enabled join.
Model [PURE](../../../glossary.md#term-pure)-body validity/[dtypes](../../../glossary.md#data-type--datatype) and exact [cold reconstruction](../../../glossary.md#term-cold-reconstruction) support stay independent of the
initially tested [F32](../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../glossary.md#floating-point-precision-and-rounding) [CPU](../../../glossary.md#term-cpu) compute/train domain. No added production type/package/API or builder recipe.

[NN0032 migration decision](tasks/0032-fluent-rnn-general-scan-migration.md) is Draft after source-grounded
planning only. New/moved production types/packages/API: None; reuse existing RnnStep/sequence/cell,
[Scan body](../../../glossary.md#backpropagation-through-time-and-recurrent-state) and [owned Model](../../../glossary.md#model-module-buffer-and-scan-distinctions) execution owners. Persistence and [backend](../../../glossary.md#backend)/[layout](../../../glossary.md#layout) compatibility decisions
must precede any fluent cutover; no new mode, provider selection or fallback is selected.
Its confirmed advanced second-order CPU [compile](../../../glossary.md#compile) [regression](../../../glossary.md#term-regression) remains unresolved, not runtime [Hessian](../../../glossary.md#term-hessian)
proof or a first-order waiver. The separately authorized
[Model0036](../../modules/model/tasks/0036-length-aware-general-scan-design.md) is Complete for
[proposal delivery](../../length-aware-general-scan.md) only, after independent final APPROVE and MAIN acceptance. Its rowCarryCount/sequenceRowAxes,
length overload and selective v5 are PROPOSED, not NN API/authority approval.

## Task list

The table is the complete ordered [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) queue. Completed rows link to their evidence-owning task;
Draft rows without links have no detailed task brief.

| ID | Task | Status | Depends on | One-line result or intent |
|---|---|---|---|---|
| [0001](tasks/0001-module-parameter-buffer-and-forward-context-foundation.md) | [Module](../../../glossary.md#term-module), [parameter](../../../glossary.md#term-parameter), [buffer](../../../glossary.md#term-buffer), and forward-context foundation | Complete | Completed `modules/model` [Tensor](../../../glossary.md#tensor) contracts; [ADR](../../../glossary.md#term-architecture-decision-record-adr) 0007 | Created the [model](../../../glossary.md#term-model)-only NN module, state wrappers, mode/context contracts, Training edge, and architecture checks. |
| [0002](tasks/0002-module-tree-ownership-and-recursive-mode-propagation.md) | Module-tree [ownership](../../../glossary.md#term-ownership-meanings) and recursive mode propagation | Complete | 0001 | Added exclusive child ownership, collision-free deterministic state paths, [immutable](../../../glossary.md#term-immutable) discovery snapshots, and preflighted iterative mode propagation. |
| [0003](tasks/0003-validated-parameter-and-buffer-binding-replacement.md) | Validated parameter and buffer [binding](../../../glossary.md#term-binding) replacement | Complete | 0002 | Added [schema](../../../glossary.md#term-schema)-validated direct binding replacement while preserving wrapper/name [identity](../../../glossary.md#term-identity-meanings) and explicitly excluding [concurrency](../../../glossary.md#term-concurrency) guarantees. |
| [0004](tasks/0004-explicit-eager-parameter-initializers.md) | Explicit [eager](../../../glossary.md#term-eager-evaluation) parameter initializers | Complete | 0001–0003; completed Model eager [constant](../../../glossary.md#term-constant)/random contracts | Added stateless floating zero/one, explicit-source normal/uniform, and [rank-two](../../../glossary.md#axis-rank-scalar-vector-and-matrix) [Glorot](../../../glossary.md#term-glorot-uniform)/[Kaiming](../../../glossary.md#term-kaiming)-[ReLU](../../../glossary.md#term-relu) eager initializers. |
| [0004A](tasks/0004a-parameter-update-and-traversal-hardening.md) | Parameter update and traversal hardening | Complete | 0001–0004; post-0004 code review | Added public schema-safe `Parameter.replace`, iterative identity-defended traversal, and complete fan-initializer failure contracts. |
| [0005](tasks/0005-linear-layer.md) | [Linear](../../../glossary.md#term-linear-meanings) layer | Complete | 0001, 0004, 0004A; completed model `Tensor.linear` | Added final stateful `Linear` with stable parameters and exact Model linear composition. |
| [0006](tasks/0006-layer-normalization-layer.md) | [Layer normalization](../../../glossary.md#layer-normalization) layer | Complete | 0001–0005; completed Model 0021 | Added affine `LayerNorm` with exact [normalized Shape](../../../glossary.md#normalized-shape), typed [epsilon](../../../glossary.md#epsilon), initialized/supplied state, and Model delegation. |
| [0007](tasks/0007-embedding-layer.md) | [Embedding](../../../glossary.md#embedding) layer | Complete | 0006; completed Model 0019A1 | Added supplied-table `Embedding` with [axis](../../../glossary.md#term-axis)-zero Model delegation and no [padding](../../../glossary.md#padding)-row policy. |
| [0008](tasks/0008-batch-normalization-layer.md) | [Batch normalization](../../../glossary.md#batch-normalization--batch-normalization-inference-and-training) layer | Complete | 0007; completed Model 0021B–0021C | Added mode-sensitive affine `BatchNorm` and explicit installation of pure next-[statistic](../../../glossary.md#term-statistic) expressions into stable buffers. |
| [0009](tasks/0009-dropout-layer.md) | [Dropout](../../../glossary.md#dropout--inverted-dropout) layer | Complete | 0008; completed Model 0019B–0019B1 | Added caller-threaded [graph](../../../glossary.md#graph)-random state, training delegation, and identity-preserving [evaluation](../../../glossary.md#term-evaluation) bypass without hidden state. |
| [0010](tasks/0010-state-dictionary-and-checkpoint-contract.md) | State dictionary and [checkpoint](../../../glossary.md#term-checkpoint) contract | Complete | 0009; stable module-tree traversal and replacement contracts | Added immutable ordered in-memory state dictionaries and strict validate-before-install loading; persistent transport stayed deferred. |
| [0011](tasks/0011-unary-tensor-module-composition-and-sequential.md) | [Unary](../../../glossary.md#term-unary) Tensor module composition and Sequential | Complete | 0010; completed Linear, LayerNorm, and Embedding unary Tensor APIs | Added narrow `UnaryTensorModule` and immutable numeric-child `Sequential`; contextual/stateful [signatures](../../../glossary.md#term-signature) remain outside it. |
| [0012](tasks/0012-vanilla-rnn-cell.md) | Vanilla [tanh](../../../glossary.md#term-tanh) [RNN](../../../glossary.md#term-recurrent-neural-network-rnn) cell | Complete | 0011; completed Model linear, ADD, and TANH expressions | Added explicit-state vanilla tanh `RnnCell` with fixed projections and optional shared [bias](../../../glossary.md#term-bias). |
| [0013](tasks/0013-gru-cell.md) | [GRU](../../../glossary.md#term-gated-recurrent-unit-gru) cell | Complete | 0012 | Added explicit-state reset-after `GruCell` with fixed packed gate order, interpolation, and optional input-side bias. |
| [0014](tasks/0014-lstm-cell.md) | [LSTM](../../../glossary.md#term-long-short-term-memory-lstm) cell | Complete | 0013 | Added explicit hidden/cell-state `LstmCell`, fixed packed equations, zero-bias policy, and typed next-state result. |
| [0015](tasks/0015-static-packed-rnn-sequence.md) | Static packed RNN sequence | Complete | 0012–0014; completed Model SELECT, [GATHER](../../../glossary.md#gather), STACK, and eager [INT64](../../../glossary.md#data-type--datatype) leaves | Added static valid-prefix RNN packing, compact outputs, and original-order final-hidden restoration. |
| [0016](tasks/0016-static-packed-gru-sequence.md) | Static packed GRU sequence | Complete | 0015 | Applied the static one-state packing/restoration contract to GRU without a shared recurrent abstraction. |
| [0017](tasks/0017-static-packed-lstm-sequence.md) | Static packed LSTM sequence | Complete | 0014–0015 | Added static LSTM packing with compact hidden outputs and restoration of final hidden and cell state. |
| [0018](tasks/0018-typed-functional-model-topology.md) | Typed functional Model [topology](../../../glossary.md#term-topology) | Complete | 0010–0017; stable Module ownership | Added typed `Model<I,O>`, [sealed](../../../glossary.md#term-sealed) definition-scoped topology, [atomic](../../../glossary.md#resource-transaction-and-rollback) named ownership, and stable state paths. |
| [0019](tasks/0019-automatic-first-forward-linear-initialization.md) | Automatic first-forward Linear initialization | Complete | 0018; exact initialization/state-dictionary decision | Added private reserved state and atomic first-forward/strict-load binding while inferring only `inFeatures`. |
| [0020](tasks/0020-automatic-recurrent-initialization-and-sequence-defaults.md) | Automatic recurrent initialization and sequence defaults | Complete | 0019; current recurrent cell/sequence and Model [provenance](../../../glossary.md#provenance) contracts | Unified closed initialization policy, automatic recurrent input-width binding, deterministic [seeds](../../../glossary.md#commit-release-distribution-log-and-seed-meanings), and zero-state/all-valid sequence conveniences. |
| [0020A](tasks/0020a-initialized-embedding.md) | Initialized Embedding | Complete | 0020; current Embedding and [tokenizer](../../../glossary.md#frozen-whitespace-tokenizer)/schema boundaries | Added eager initialized Embedding with explicit schema/policy/[seed](../../../glossary.md#term-seed) and ordinary trainable semantics for every row. |
| [0020B](tasks/0020b-stateless-standard-module-factory.md) | Stateless standard ModuleFactory | Complete | 0020–0020A | Added fresh standard Embedding/Linear/RNN/GRU/LSTM recipes without ownership, [registry](../../../glossary.md#term-registry), or retained configuration. |
| [0020C](tasks/0020c-bidirectional-static-recurrent-composition.md) | Type-safe bidirectional static recurrent composition | Complete | 0020–0020B; stable static sequence/result and Model gather/composition contracts | Added concrete bidirectional RNN/GRU/LSTM containers with independent state, valid-prefix reversal, aligned forward-first concat, and typed final states. |
| [0021A](tasks/0021a-fixed-recurrent-scan-architecture-decision.md) | Fixed recurrent-[scan](../../../glossary.md#term-cumulative-scan) architecture decision | Complete | 0020C; current flat Model/[Compiler](../../../glossary.md#term-compiler) graph, [Prepare](../../../glossary.md#prepare), [Runtime](../../../glossary.md#term-runtime), and [Engine](../../../glossary.md#term-engine) contracts | Accepted fixed flat multi-output RNN/GRU/LSTM scan semantics, [lifecycle](../../../glossary.md#lifecycle) ownership, [fail-closed](../../../glossary.md#term-fail-closed) [backpropagation](../../../glossary.md#term-backpropagation)-through-time boundary, and migration constraints. |
| [0021B](tasks/0021b-fixed-recurrent-scan-implementation-program.md) | Fixed recurrent-scan forward execution program | Complete | 0021A; Model0025E/0025F; Compiler0006A; CPU0027; Engine0021; CP0010/CP0011 Complete | CPU0028 [scalar](../../../glossary.md#term-scalar) [FLOAT32](../../../glossary.md#term-float32) forward baseline accepted 2026-10-20 after independent APPROVE and main's root gate; no NN migration/Data/[BPTT](../../../glossary.md#term-backpropagation-through-time-bptt). |
| [0022](tasks/0022-runtime-length-recurrent-nn-api.md) | NN runtime-Tensor-length recurrent forward expressions | Complete | NN0020/0020C; NN0021B/CPU0028; Model0025E/0025F; Compiler0006A; Engine0021 — Complete | Accepted 2026-10-20 after Mencius final APPROVE and focused/module/[Javadoc](../../../glossary.md#term-javadoc)/Engine integration [validation](../../../glossary.md#term-validation); additive forwardScan, no Data or compute/train/BPTT [widening](../../../glossary.md#term-widening). |
| [0022B](tasks/0022b-runtime-length-scan-model-workflow.md) | Runtime-length scan Model/builder [inference](../../../glossary.md#term-inference) workflow | Complete | 0022; 0028/0029; scan/Engine/[persistence](../../../glossary.md#term-persistence) prerequisites Complete; approved inference-only contract enacted | Accepted 2026-10-20 after final independent code/docs approval and coordinator root gate: scan recipes/finalHidden, typed buildWithLengths and compute(first,second), including after loadGraph. |
| [0022A](tasks/0022a-padded-numeric-sequence-inference.md) | Padded numeric sequence inference | Complete | NN0022/0022B/0028/0029; Data0001–0003 Complete; exact export approved | Accepted 2026-10-21 after independent APPROVE and main's source-matched architecture/root checkpoint; fresh count export and numeric reuse, no NN-to-Data edge, Text pipeline or BPTT. |
| 0023 | Arbitrary dense validity-[mask](../../../glossary.md#term-boolean-mask) semantics | Draft | 0022/0022B; 0022A for Data integration; concrete attention/[loss](../../../glossary.md#loss)/recurrent consumer | Add a [Boolean mask](../../../glossary.md#term-boolean-mask) only for a proven holes-capable consumer, never as duplicate right-padding [metadata](../../../glossary.md#term-metadata) or a skipped-work claim. |
| 0024 | Typed model/recurrent/data integration checkpoint | Draft | 0020–0022; 0022B; 0022A; 0023 only if selected; Checkpoint model-state and Training [publication](../../../glossary.md#term-publication) readiness | Retain later Data integration, state/binding/checkpoint compatibility, variable [batches](../../../glossary.md#term-batch-machine-learning-and-hyperparameters), continuation and supported training [handoff](../../../glossary.md#term-handoff); no scan BPTT implied. |
| [0025](tasks/0025-channels-first-conv1d-conv2d-conv3d-layers.md) | Channels-first Conv1d, [Conv2d](../../../glossary.md#convolution--conv2d), and [Conv3d](../../../glossary.md#conv3d--ncdhw) layers | Complete | 0020B; Model 0025G–0025H; Compiler 0006B and 0006B6; [CPU](../../../glossary.md#term-cpu) 0008–0008A; Engine 0008 | Added [rank](../../../glossary.md#term-rank)-specific unary [convolution](../../../glossary.md#term-channels-first-convolution-layers) layers/factory recipes with group-aware initialization, atomic inferred-[channel](../../../glossary.md#term-channel) binding, strict loading, and forward-only Conv3d boundary. |
| [0025A](tasks/0025a-dimensional-convolution-user-capability-checkpoint.md) | Dimensional-convolution user-capability checkpoint | Complete | 0025; Engine 0004; CPU 0008A | Validated strict-loaded and factory convolution through public one-shot/reusable Engine workflows, numerical oracles, lifecycle ownership, and Conv3d backward rejection. |
| [0025B](tasks/0025b-public-model-dependency-export.md) | Export public Model [dependency](../../../glossary.md#term-dependency) | Complete | Existing NN [facade](../../../glossary.md#term-facade); project review | Correct [Gradle](../../../glossary.md#term-gradle) [API](../../../glossary.md#term-application-programming-interface-api) visibility and validate an isolated NN-only consumer. |
| [0026](tasks/0026-declarative-model-specification.md) | Declarative model specification and reconstruction | Complete | Verified Complete 0025/0025A/0025B; accepted NN [matrix](../../../glossary.md#term-matrix-meanings)/contract | Describe/reconstruct/strict-load delivered with final reviews and coordinator gates; no persistent [codec](../../../glossary.md#term-codec) or new dependency. |
| 0027 | Semantic recurrent description and reconstruction | Draft | NN0026/CP0005B Complete; P2 review and any separately selected recurrent [scope](../../../glossary.md#term-scope) | Held despite CP0005D [planning](../../../glossary.md#term-planning) completion; no bytes/versions or detailed brief. |
| [0028](tasks/0028-owned-model-compute-training-and-persistence.md) | Owned ordinary tensor Model workflow | Complete | Engine0021; Training0007; Compiler0008; Checkpoint0008; Training0008 — Complete | Accepted 2026-10-19 after definitive independent approval and coordinator native-enabled root gate; evidence in brief. |
| [0029](tasks/0029-fluent-model-builder-and-checkpoint-independent-execution.md) | [Fluent Model builder](../../../glossary.md#fluent-model-builder) and checkpoint-independent execution | Complete | NN0028; Checkpoint0009; CPU0025; Compiler0009 — Complete | Planck final APPROVE, zero findings; worker/integration evidence and main's source-matched native root gate passed (2026-10-19). |
| [0030](tasks/0030-general-scan-owned-model-admission-decision.md) | General-scan [owned Model](../../../glossary.md#model-module-buffer-and-scan-distinctions) [admission](../../../glossary.md#term-admission) decision | Complete | Low-level scan/root join and owned Model/source/Training baseline Complete | User-approved direction/planning only; cold prerequisites now Complete, NN0031 separately owns workflow [qualification](../../../glossary.md#term-qualification). |
| [0031](tasks/0031-general-scan-owned-model-workflow.md#result) | General-scan owned Model workflow | Complete | NN0030; Compiler0012; CP0012; accepted CPU domain/low-level join and owned Model/Engine/Training baseline Complete | Bounded all-valid owned acquisition/compute/first-order train accepted after independent review, final module/Javadoc and configured native program join; no successor. |
| [0032](tasks/0032-fluent-rnn-general-scan-migration.md) | Fluent RNN general-scan migration decision | Draft | Complete baseline; confirmed second-order CPU [compile](../../../glossary.md#compile) [regression](../../../glossary.md#term-regression) and remaining compatibility/fluent-authority gates held | No higher-order waiver; CONCAT-copy/mixed-engine gates remain; `.rnn` unchanged. |
| [0033](tasks/0033-trainable-runtime-length-rnn-workflow.md#result) | Trainable runtime-length RNN Model workflow | Complete | Compiler0018 and Checkpoint0014 Complete; exact API/authority independently approved | Shared cell equation, general [F32](../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../glossary.md#floating-point-precision-and-rounding) rnnScan and semantic ports; 620 affected tests/Javadocs, public numerical train/save/load/new-train and runnable example accepted. |

## [Dependency](../../../glossary.md#term-dependency) [DAG](../../../glossary.md#term-directed-acyclic-graph-dag) and authorized frontiers

Current [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) implementation frontier: **None**. MAIN owns serial shared integration, Parallel
group None. Compiler0018, Checkpoint0014 and [NN0033](tasks/0033-trainable-runtime-length-rnn-workflow.md#result)
are Complete after independent reviews and source-matched consumer proof. The additive
RnnSequence.forwardGeneralScan/[F32](../../../glossary.md#floating-point-precision-and-rounding)-[F64](../../../glossary.md#floating-point-precision-and-rounding) rnnScan cutover reuses the cell equation, initialization
and [Model](../../../glossary.md#term-model)/[Engine](../../../glossary.md#term-engine)/Training [owners](../../../glossary.md#term-owner). [Canonical](../../../glossary.md#term-canonical-form) result/read domains and exact general/fixed ports
are explicit, without bootstrap fixed [producers](../../../glossary.md#term-producer), synthetic [layouts](../../../glossary.md#layout) or a new [carrier](../../../glossary.md#term-storage-carrier).
The next product phase is capability audit/[scope](../../../glossary.md#term-scope) freeze, not an automatic NN successor.
Ordinary static-[unrolled](../../../glossary.md#backpropagation-through-time-and-recurrent-state) `.rnn` is not migrated; NN0032 stays Draft with its second-order blocker.
[GRU](../../../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../../../glossary.md#term-long-short-term-memory-lstm) and low fixed paths remain unchanged. No new types/executors/provider-based cold gate,
[dtype](../../../glossary.md#data-type--datatype) expansion or higher-order/[SIMD](../../../glossary.md#simd)/performance work. Compiler0017 is deferred, not cancelled,
behind this practical workflow. [Scan](../../../glossary.md#term-cumulative-scan)-local [padding](../../../glossary.md#padding) does not define masked [losses](../../../glossary.md#loss)/[MEAN](../../../glossary.md#term-mean) behavior.
After this workflow, the user's coarse priority is missing Java [CPU](../../../glossary.md#term-cpu) capability audit → [Metal](../../../glossary.md#term-metal)
completion → [oneDNN](../../../glossary.md#term-onednn) [backend](../../../glossary.md#backend); establish scope after audit, with no detailed successor tasks,
new backend/build/tree/architecture cutover or claim that these capabilities are complete.

[Persistence](../../../glossary.md#term-persistence) prerequisite [CP0013 Complete](../checkpoint/tasks/0013-general-scan-graph-persistence-decision.md#result):
selective v4/existing cold inventory, final Cicero APPROVE and main configured root accepted 2026-10-23.
Audit demonstrates CPU F32/F64 Conv1d-prefix old-pass/Scan-reject and [singleton](../../../glossary.md#term-singleton)-CONCAT reverse-[seed](../../../glossary.md#term-seed)
dense [view](../../../glossary.md#term-view-meanings) rejection; [AD](../../../glossary.md#autograd) [admission](../../../glossary.md#term-admission)/forward [smoke](../../../glossary.md#term-smoke-test) are not backward numeric proof. Old/new Metal-only
probes both reject; mixed execution untested, no proved Metal [regression](../../../glossary.md#term-regression). Separately approved
[CPU0037](../../backends/cpu/tasks/0037-general-scan-dense-read-views.md#result) is Complete with actual-reverse
numeric [gradients](../../../glossary.md#term-gradient); main accepted independent review and final gates on 2026-10-24.
For the separate NN0032 ordinary `.rnn` migration, remaining gates are cold output [descriptor](../../../glossary.md#term-descriptor)/
remaining compatibility → fluent authority → separately [Ready](../../../glossary.md#term-ready) migration; it is not the selected
[runtime](../../../glossary.md#term-runtime)-length [target](../../../glossary.md#term-target). Persistence completion alone authorizes no migration.
Authorized NN frontier: **None**. MAIN retains shared integration and accepted dirty-source
evidence. NN0032 and later GRU/LSTM migration remain Draft; completed NN0033 opens no other
area's implementation frontier.

The [general scan program](../../general-differentiable-scan.md)<span data-glossary-navigation="true"> ([general scan glossary](../../../glossary.md#general-differentiable-scan))</span> and
[Model0034 decision](../../modules/model/tasks/0034-general-differentiable-scan-decision.md) record
Model0034 authority decision/cutover Complete, accepted by main 2026-10-21 after final independent
Class C APPROVE, zero blockers; the brief owns result evidence. Compiler0010 fixed-[adjoint](../../../glossary.md#term-adjoint) direction
is Superseded. Serial chain: completed reviewed authority cutover → Model source/[IR](../../../glossary.md#term-intermediate-representation-ir) →
[Planning](../../../glossary.md#term-planning) body query → [Compiler](../../../glossary.md#term-compiler) AD → [Prepare](../../../glossary.md#prepare) projection → complete bounded
CPU0030 F32 [pointwise](../../../glossary.md#term-pointwise) → CPU0031 broadcast/SUM/layout → CPU0032 [MATMUL](../../../glossary.md#matrix-multiplication--matmul) → CPU0033 F64/domain closure
→ low-level root/native join: all Complete. The subsequent
[NN0030 approved decision](tasks/0030-general-scan-owned-model-admission-decision.md), planning Complete;
all-valid NN admission is now qualified by NN0031 independently of padding and persistence. Base
main=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`, Parallel group None, shared [owner](../../../glossary.md#term-owner) main.
No new NN types/[API](../../../glossary.md#term-application-programming-interface-api) authorized, no automatic executor; NN0031 completed [qualification](../../../glossary.md#term-qualification)/cutover of
general [graph](../../../glossary.md#graph) acquisition and training, not fixed rnnScan training or body save/load.
Existing fixed/static routes and fixed-only rejection stay unchanged.
Actual user approval (“ok, tak tedy pokracuj”) covers the concrete body/reverse-scan/[residual](../../../glossary.md#scan-residual-port-and-history-demand) proposal
and authority cutover, not NN or all Java slices. The brief records Complete dependencies/base/inactive
conflicts verified at launch on 2026-10-21. Authority/Model/NN/CPU frontiers None; CPU0030 bounded milestone Complete.
[Model0035 source/IR](../../modules/model/tasks/0035-general-scan-source-and-compiled-body.md) is Complete,
accepted by main 2026-10-22 after independent approval and its compatibility [checkpoint](../../../glossary.md#term-checkpoint).
Planning0008 body query is Complete, accepted by main 2026-10-22 after final independent APPROVE and
compatibility gates; Planning frontier None. [Compiler0011](../../modules/compiler/tasks/0011-general-scan-capture-and-first-order-ad.md)
is Complete, accepted by main 2026-10-22 after independent code/docs APPROVE and compatibility checkpoint.
[Prepare0010](../../modules/prepare/tasks/0010-general-scan-projection-and-resource-handoff.md) is Complete:
scoped projection/demand [validation](../../../glossary.md#term-validation) and resource [handoff](../../../glossary.md#term-handoff), main accepted 2026-10-22 after Huygens
final Class C code/docs APPROVE and the exact compatibility checkpoint.
CPU/Prepare/Planning/Model/Compiler/NN frontiers: None; Compiler0012 is Complete. Bounded CPU F32 pointwise forward + actual
automatic reverse [CPU0030](../../backends/cpu/tasks/0030-general-scan-f32-pointwise-vertical.md)
is Complete, accepted by main 2026-10-22 after independent review and focused checkpoint PASS.
Declared algebra/actual first-order closure in F32/F64 and the low-level join are Complete;
[CPU0033 Result](../../backends/cpu/tasks/0033-general-scan-f64-domain-closure.md#result) owns evidence
and unresolved timing limitations, not a parity/[JIT](../../../glossary.md#term-just-in-time-compilation-jit)-cause claim. No successor launch or NN
train/padding/persistence grant. Current base includes accepted uncommitted
authority/Model0034/0035, Planning0008, Compiler0011, Prepare0010 and CPU0030–0033 work.
[BF16](../../../glossary.md#floating-point-precision-and-rounding)/[FP16](../../../glossary.md#term-fp16) qualification remains CPU follow-up, not whole-model conversion.

NN0030 direction/planning, Compiler0012 and CP0012 → NN0031 → configured native-enabled program join:
all Complete. [NN0031 Result](tasks/0031-general-scan-owned-model-workflow.md#result) owns final evidence;
its brief retains seven integration fields and historical launch verification at the same HEAD/origin
plus 284 accepted dirty paths. Parallel group None; main retains integration [ownership](../../../glossary.md#term-ownership-meanings), with exclusive
final docs/[Javadoc](../../../glossary.md#term-javadoc) delegation after source-writer yield. Clean implementation, independent Class C review,
final worker NN test/Javadoc and main's single root all passed. CP0012's historical environment exception
was resolved by explicit existing-library configuration, not native/test gating edits; required Metal
training cases executed unskipped/pass. NN/Compiler/Checkpoint/CPU/other program frontiers None; no successor.

Authorized NN frontier: None. [NN0022A](tasks/0022a-padded-numeric-sequence-inference.md) is Complete,
accepted 2026-10-21 for numeric F32 sequences at base `6eabea0c1ebf77b9bd8382855c7131eb89b52e04`.
The user approved the exact opt-in Data length export on 2026-10-21; root/owning Data contract and
ADR0044 enact the permission. No NN-to-Data dependency or new execution owner is authorized.
The brief records all seven integration fields and main's approval/base/dependency/conflict checks.
Parallel group None; serial clean implementation and independent Class C approval are complete.
Main's source-matched architecture/root checkpoint and acceptance are complete; no new executor.
Full fluent Text IDs → [Embedding](../../../glossary.md#embedding) → scan has a separate upstream gap: ModelBuilder accepts only
floating input, has no Embedding recipe and instantiates every recipe with the original input
dtype. Select an independent NN integer-input/Embedding capability with explicit dtype transitions
before promising that workflow; the length bridge alone does not deliver it. No Text frontier is opened.

Completed NN0021A + Model0025E/0025F + Compiler0006A + CPU0027 + Engine0021 + CP0010/CP0011
→ [CPU0028](../../backends/cpu/tasks/0028-fixed-recurrent-scan-forward-execution.md)
→ [NN0021B](tasks/0021b-fixed-recurrent-scan-implementation-program.md), all Complete.
Main accepted independent approval and the final root gate on 2026-10-20; task briefs own evidence
and base [metadata](../../../glossary.md#term-metadata). Parallel group None; main owns integration/shared documents.
At that completed scan checkpoint NN/Data/CPU frontiers were None; CPU0030 subsequently completed. Data0001–0003 are Complete in the
[Data plan](../data/master-plan.md); Text0001 is Complete in the [Text plan](../text/master-plan.md). No new NN production type,
Data/static-API migration or [BPTT](../../../glossary.md#term-backpropagation-through-time-bptt) is authorized; CPU0028 records the delivered baseline and exclusions.

Completed 2026-10-20: completed scan/cell/Engine prerequisites →
[NN0022 Complete](tasks/0022-runtime-length-recurrent-nn-api.md) (NN-only) →
[NN0022B Complete](tasks/0022b-runtime-length-scan-model-workflow.md) (Model/builder [inference](../../../glossary.md#term-inference)) → NN0022A Complete
(numeric Data integration, Data0001–0003 Complete and export approved) → later NN0024 checkpoint. NN0023 retains its
concrete-consumer gate and NN0022A when Data is involved. NN0023/0024 remain Draft; no automatic successor launch.
NN0022 records all seven integration fields at base `2d1c1d6d0867a0bebda34e9177ad52ba9f13071a`;
parallel group None, serial implementation/review/integration, main owns shared documents. Exact
[signatures](../../../glossary.md#term-signature) and descriptor-only [atomic](../../../glossary.md#resource-transaction-and-rollback) first-scan initialization (including T/B=0) are accepted;
launch verification and completion evidence are in the brief. Execution uses direct Engine;
Model.compute/train [widening](../../../glossary.md#term-widening) was outside NN0022's scope. Focused NN
test/Javadoc, Engine integration and final independent review passed; prior backend/root evidence reused.

NN0022B is Complete under the user-approved Class C inference-only cutover (2026-10-20).
Root/owning-NN contract, targeted explanation and ADR0041 were enacted before implementation.
The brief records all seven integration fields and pre-launch dependency/base/conflict checks;
HEAD=origin/main `2d1c1d6d0867a0bebda34e9177ad52ba9f13071a` plus accepted dirty NN0022 is the base,
not a merged predecessor. Independent contract/code/docs reviews and the coordinator's source-matched
root test/Javadoc gate passed; the brief owns exact evidence, including cached tasks and the unrelated
integration assumption skip. Main owns shared-document integration; parallel group None.
At NN0022B completion, NN/CPU frontiers were None; the subsequent NN0022A frontier is now closed.
NN0023/0024 remain Draft. No NN0022B capability follow-up remains;
no Data/type/format/backend or BPTT widening, automatic successor launch or new execution pipeline.

Engine0021 + Training0007 + Compiler0008 + Checkpoint0008 + Training0008 → NN0028.
Compiler0006B11 + Compiler0008 + CPU0025 → Compiler0009 (all Complete).
NN0028 + Checkpoint0009 + CPU0025 + Compiler0009 → NN0029 → main's shared native root gate: Complete.
At NN0029 and NN0022 completion checkpoints, NN frontier was None. Neither prerequisite repair depends on NN0029; the shared gate passed
on the unchanged reviewed [snapshot](../../../glossary.md#term-snapshot) (2026-10-19), including UP-TO-DATE report evidence.
Main accepted the final independent review and verified launch on 2026-10-19; user authorized implementation.
The separate [Checkpoint0009 activation-source packet](../checkpoint/tasks/0009-owned-model-activation-source-and-persistence.md)<span data-glossary-navigation="true"> ([activation glossary](../../../glossary.md#term-activation))</span>
is Complete, accepted 2026-10-19 after Archimedes final APPROVE and main's integration/native root gates.
NN remains the unchanged owned-source consumer with
targeted Javadoc/tests and one direct-[oracle](../../../glossary.md#term-test-oracle) [MLP](../../../glossary.md#term-multilayer-perceptron-mlp) integration regression. Compiler/Checkpoint jointly
own the completed eight-kind/selective-v2 capability; NN0028 and the selected program stay Complete.
CP0009 introduced no NN implementation task, [lifecycle](../../../glossary.md#lifecycle) change or new public type.
At Checkpoint0009 completion, Checkpoint/Compiler/NN frontiers: None. Its brief owns final evidence; no follow-up remains.
NN0029 conflicts: NN API/lifecycle, root/NN authority and shared documents; none active, main verified.
Parallel group: None. Common base: `7e880fb1044168043e45b10419a570d64c5283ae` plus the full dirty
snapshot: accepted CP0009, frozen NN partial and completed CPU0025/Compiler0009. Integration order:
reviewed [Compiler0009](../../modules/compiler/tasks/0009-static-select-and-slice-logical-layout-closure.md)
→ main's verified NN [resume](../../../glossary.md#schema-serialization-and-checkpoint) (2026-10-19) → finish NN implementation → independent Class C review → main's single native root gate.
Main owns shared-document integration; the NN0029 brief owns exact metadata, signatures, acceptance,
validation, frozen evidence and current resume verification. Main rechecked approval, completed/reviewed
predecessors, unchanged authority/full dirty base and inactive conflicts with no other writer.
At that checkpoint Compiler frontier was None; Compiler0012 is now Complete, frontier None. NN acceptance/scope
unchanged; no forced [contiguous](../../../glossary.md#term-contiguous) copies or new hierarchy.
No Checkpoint widening or scan/Data/NN0027 prerequisite; earlier held streams remain Draft.

## Proposed whole-[model](../../../glossary.md#term-model) [checkpoint](../../../glossary.md#term-checkpoint) program

The user-approved [CP0010 packet](../checkpoint/tasks/0010-fluent-cnn-and-static-recurrent-graph-persistence.md)
and [CP0011 companion](../checkpoint/tasks/0011-flat-recurrent-scan-persistence.md) are jointly Complete
on 2026-10-20: builder [CNN](../../../glossary.md#term-convolutional-neural-network-cnn)/static recurrence and cold flat-[scan](../../../glossary.md#term-cumulative-scan) [persistence](../../../glossary.md#term-persistence) accepted after independent
APPROVE and main's focused/native root gates; [CP0010 Result](../checkpoint/tasks/0010-fluent-cnn-and-static-recurrent-graph-persistence.md#result)
owns evidence. At that checkpoint, [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/[Compiler](../../../glossary.md#term-compiler)/Checkpoint source frontiers: None; NN0021B/CPU0028 subsequently
completed bounded forward execution, not [BPTT](../../../glossary.md#term-backpropagation-through-time-bptt). Existing predecessor [DAG](../../../glossary.md#term-directed-acyclic-graph-dag) and held NN0027/CP0005C [scope](../../../glossary.md#term-scope) unchanged; no new
[API](../../../glossary.md#term-application-programming-interface-api)/types/[lifecycle](../../../glossary.md#lifecycle)/adjoints. Base `8d5357e9a8698721a2252cb2ca3cb94150e705e8`, Parallel group None;
main owns serialized integration and shared documents.

[ADR 0038](../../../design/decisions/0038-declarative-model-and-training-resume.md#nn-decision-selected-for-review)
revises the primary product to automatic usable-model load, then [inference](../../../glossary.md#term-inference) or owned Training [resume](../../../glossary.md#schema-serialization-and-checkpoint).
0026 is Complete after final independent reviews and coordinator [validation](../../../glossary.md#term-validation).

Earlier proposed serial DAG (held):
`NN0026 Complete → CP0005B Complete | HOLD: NN0027 → CP0005C → Training0006 → optional CP0006`.
[CP0005D](../checkpoint/tasks/0005d-portable-graph-persistence-decision.md) is Complete for bounded
[planning](../../../glossary.md#term-planning)/review after independent review, the sole P2 correction and coordinator acceptance;
[ADR0039](../../../design/decisions/0039-portable-graph-model-persistence.md) records the accepted decision; root/scoped contracts enact it.
The user approved one model.compute/train/save/load direction for manual/file [tensors](../../../glossary.md#tensor) and generic
Training/Checkpoint [modules](../../../glossary.md#term-module). The [cross-area program](../../model-execution-refactor.md) owns the
phases, exact edges and independently approved P2 planning packet. NN stays extensions/nn, supplies the
live named-[parameter](../../../glossary.md#term-parameter) adapter and receives NN-specific Checkpoint glue after removal of reverse edges.
Same-[signature](../../../glossary.md#term-signature) execution [targets](../../../glossary.md#term-target) [Engine](../../../glossary.md#term-engine) [PreparedExecution](../../../glossary.md#prepared-execution--preparedexecution) plus [Engine.runValues](../../../glossary.md#fresh-value-binding--enginerunvalues) with fresh per-[run](../../../glossary.md#term-run)
[borrowed](../../../glossary.md#ownership-borrowing-and-lease) storage, not the [InferenceSession](../../../glossary.md#former-inference-session--inferencesession)/Engine.session API removed by Engine0021. Model's
high-level compute/train hides preparation; no replacement runner/session [facade](../../../glossary.md#term-facade). Normal use
does not reconstruct via outputs(actualInputs) or mutate shared placeholders.

Current Model<I,O> permits arbitrary Java forward and Module.train() is mode-only; both remain.
P2 selects defineTensor(signature, definition) and load(Path) returning Model<Tensor,Tensor>,
with compute(Tensor), configured train(Tensor,Tensor), save and close. No manual forward/[leaf](../../../glossary.md#term-leaf)
enumeration/second Model is required for ordinary NN use. The one-shot definition produces a
source [graph](../../../glossary.md#graph); discard its [callback](../../../glossary.md#term-callback) after private leaf acquisition/reconstruction, so a captured
external [weight](../../../glossary.md#term-weight) cannot escape into later symbolic forward. Explicit symbolic forward reconstructs
the owned source cold; compute/train never do so per [batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters). Legacy Model.define stays unchanged.
Advanced graph/loadGraph supports Tensor lists, not arbitrary Java-record [serialization](../../../glossary.md#term-serialization).
[Detached](../../../glossary.md#detached-model-results) [HostTensorValue](../../../glossary.md#host-tensor-value--hosttensorvalue) results are not generic O; no cast or new public model/result hierarchy.
Keep ModelSpecification/DeclarativeModel and flexible v1 [rank](../../../glossary.md#term-rank)/batch behavior until explicit migration.
Source restoration creates ordinary Tensor expressions/weights before [compile](../../../glossary.md#compile) without Engine/batch.
No loaded-model hierarchy, second execution pipeline or per-[RNN](../../../glossary.md#term-recurrent-neural-network-rnn) recipe prerequisite.

Implemented [binding](../../../glossary.md#term-binding): `Model.load(path[, limits]).withEngine(engine)` returns the exact same
ordinary Model, as does application-built Model binding. It borrows one Engine for compute/train;
existing explicit borrowed-Engine overloads remain. Otherwise Engine.standard() is [lazy](../../../glossary.md#term-lazy-evaluation)/model-owned,
currently [CPU](../../../glossary.md#term-cpu)-only, without discovery/global state. Binding freezes at first inference/training
preparation setup, including failed setup, before Engine/native/compile/[prepare](../../../glossary.md#prepare) work. No later
rebinding or automatic reprepare after rebinding; null/closed candidates reject without mutation.
The [exact admission, errors and configureTraining timing](../../../design/decisions/0039-portable-graph-model-persistence.md#target-model-engine-binding)
are implemented in completed NN0028. compute(batch) returns detached HostTensorValue, not a request/wrapper;
no Model methods or lifecycle are implemented by Training0007.

NN0028 is Complete; at its acceptance NN frontier was None. Engine0020's [backend](../../../glossary.md#backend)/native-only cutover is Complete after
integration, independent review and combined validation. The user's later five choices are now
accepted. Engine0021 delivered the thin-session removal, authority cutover and mechanical Training
migration; independent Class C approval and the native-enabled test/[Javadoc](../../../glossary.md#term-javadoc) gate passed. Its brief
owns evidence. Training0007, including its [atomic](../../../glossary.md#resource-transaction-and-rollback) NN adapter, is Complete after final independent
APPROVE and the coordinator gate. Compiler0008, Checkpoint0008 and Training0008 are also Complete
after final independent reviews and coordinator gates; the selected program has no remaining frontier.
NN0028's implementation is accepted. Training0007 removed its reverse edge and added
Module.trainingParameters; Checkpoint0008 moved the nine NN glue classes into nn.checkpoint and
removed Checkpoint→NN. Current dependencies are acyclic; bounded high-level NN [ownership](../../../glossary.md#term-ownership-meanings) is now enacted.
NN0028 privately acquires external weights once, keeps one working set for compute/train/save,
creates Engine.standard lazily (or borrows explicit Engine), and rejects stale mode/binding changes.
One high-level [operation](../../../glossary.md#operation) at a time; detached compute/TrainingStep values outlive close.
Known source/signature enables save-before-compute; model-only save is not [optimizer](../../../glossary.md#term-optimizer)/[RNG](../../../glossary.md#graph-rng-state--graphrngstate) resume.
Package additions are private typed model implementations/state namespaces in nn.module, not a
public TensorModel or execution adapter. The generic boundary types belong to Training/Checkpoint.
Shared documents integrate serially under the model-execution coordinator; parallel group None,
common base is the revision recorded above, reverified before launch. Exact DAG and
migration map live in the program; the [Training0007 brief](../training/tasks/0007-generic-live-parameter-admission.md)
records final independent approval and successful coordinator validation; NN0028 is its completed successor.
NN0027 owns only semantic description/reconstruction using existing static Rnn/Gru/Lstm capabilities;
CP0005C owns explicit versioned recurrent definition wire/whole-save/load adoption. This is an [owner](../../../glossary.md#term-owner) split,
not a hard [dependency](../../../glossary.md#term-dependency) on optimized RecurrentScan/backend BPTT/Data. At that completed program checkpoint NN frontier was None.
NN0026's API/[matrix](../../../glossary.md#term-matrix-meanings) and NN-only root/contract/explanation/[ADR](../../../glossary.md#term-architecture-decision-record-adr) packet are accepted after final
independent review; coordinator [Ready](../../../glossary.md#term-ready) verification is recorded in the brief. Downstream
CP0005B's whole-model contract/API/wire/default packet is accepted after final independent approval
and coordinator verification; CP0005B is Complete with final reviews/gates (2026-10-15), no new
frontier. NN0027/CP0005C/Training0006/adapter remain held Draft; their optional policies are not P2 gates.

NN0027 ordering [metadata](../../../glossary.md#term-metadata) (held, revalidate after remaining ADR0039 semantic decisions): Depends on NN0026 and CP0005B; Conflicts with shared NN semantic/API
contracts and CP0005C/Training0006/adapter0006; Parallel group None; Common base revision
`f892cd3fbde4f8b7722b7eb5b14f4b0a5f3b2669` (reverify after CP integration); Integration order
after CP0005B and before CP0005C; Integration validation future NN semantic reconstruction tests
and semantic round trips plus architecture at its join; Shared-document integration owner NN/Training/Checkpoint
coordinator. Input/output/default carried-state policy is future before a detailed brief or Ready.
CP0005C format version and Training [gradient](../../../glossary.md#term-gradient) [admission](../../../glossary.md#term-admission) are separate future decisions. Current
whole-model v1 rejects all recurrent IDs; NN0027 owns no persistent format or I/O.

Selected package map: ModelSpecification and DeclarativeModel in existing nn.module, distinct from
the ModelDefinition Java callback; closed [Activation](../../../glossary.md#term-activation) in nn.layers. Initial scope: nested Sequential,
[Linear](../../../glossary.md#term-linear-meanings)/LayerNorm/
[Embedding](../../../glossary.md#embedding)/Conv1d/2d/3d and closed RELU/[TANH](../../../glossary.md#term-tanh)/[SIGMOID](../../../glossary.md#term-sigmoid) steps, not all Java models.
NN owns actual attrs/signatures/wiring/sharing. Children alone do not prove forward flow.
Inference constraints need not freeze batch [dimensions](../../../glossary.md#dimension); Training signatures are independently static.
Inference Tensor-[identity](../../../glossary.md#term-identity-meanings) [aliases](../../../glossary.md#aliasing-and-arena) reconstruct faithfully; describe reads current bindings after
legal replacement/load, without cached alias authority. Training rejects repeated parameter-path
Tensor identities/overlapping stores, not every repeated forward use of one uniquely owned layer.
Declarative branches/contextual adapters are explicit successors, not detailed new tasks.
[Descriptor](../../../glossary.md#term-descriptor)-bound skeletons retain source numeric paths without a wrapper prefix and use supplied
state/strict load without random weights; focused tests extend that evidence. The brief records
exact records, typed configuration access and Linear.fromState [guard](../../../glossary.md#term-guard) policy; Model-only edges and
existing generic APIs stay intact. 4096/depth64 bound cold-tree processing, not [backend capability](../../../glossary.md#backend-capability).

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
  convolution plus its user checkpoint (0025–0025A).<span data-glossary-navigation="true"> ([lifecycle glossary](../../../glossary.md#lifecycle); [topology glossary](../../../glossary.md#term-topology); [scan glossary](../../../glossary.md#term-cumulative-scan); [convolution glossary](../../../glossary.md#term-channels-first-convolution-layers); [checkpoint glossary](../../../glossary.md#term-checkpoint))</span>
- Complete: NN0021B/CPU0028 canonical FLOAT32 scalar forward execution; evidence in the briefs.
  Complete: NN0022 additive runtime-length expressions after final review and validation;
  Complete: NN0022B Model/builder inference after independent reviews and the coordinator root gate (2026-10-20).
  NN0022A is Complete for numeric inference after independent APPROVE and main's architecture/root gate;
  0023/0024 remain Draft. Audit repair 0025B is Complete.<span data-glossary-navigation="true"> ([canonical glossary](../../../glossary.md#term-canonical-form); [FLOAT32 glossary](../../../glossary.md#term-float32); [scalar glossary](../../../glossary.md#term-scalar); [runtime glossary](../../../glossary.md#term-runtime); [validation glossary](../../../glossary.md#term-validation); [inference glossary](../../../glossary.md#term-inference))</span>
- Complete: NN0026 with final independent reviews and coordinator validation.
  NN0027 remains Draft; CP0005B is Complete. Engine0020's completed layout cutover does not
  authorize NN execution.<span data-glossary-navigation="true"> ([layout glossary](../../../glossary.md#layout); [NN glossary](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Complete: NN0028, accepted 2026-10-19; evidence in its brief.
- Complete: NN0029 after Planck final APPROVE, zero findings, 22 focused/424 NN/3 Fluent integration
  tests and main's native root gate (2026-10-19). At that checkpoint, NN frontier: None.
- NN0027 is the earlier Draft recurrent follow-up, held with CP0005C/Training0006 held as an optional older proposal,
  not a prerequisite of NN0028; no scan/backend BPTT/Data gate.<span data-glossary-navigation="true"> ([backend glossary](../../../glossary.md#backend); [BPTT glossary](../../../glossary.md#term-backpropagation-through-time-bptt))</span>
- NN0021B is accepted after CPU0028's independent approval and main's root gate (2026-10-20).
  At that completed checkpoint NN frontier was None; NN0022A, Data0001–0003 and Text0001 are Complete. Model 0032 is Complete; the [roadmap](../../roadmap.md)
  owns current cross-area frontiers.
- Completed NN work used recorded roadmap interleaves. NN 0016/0017 were a bounded parallel pair,
  and 0025/0025A were an explicit out-of-order convolution branch ahead of unrelated Draft
  0021B–0024. Those historical exceptions authorize no further NN work.
- Complete: [NN0031](tasks/0031-general-scan-owned-model-workflow.md#result), including independent review,
  final NN module/Javadoc and main's configured native program join (2026-10-23).<span data-glossary-navigation="true"> ([module glossary](../../../glossary.md#term-module); [Javadoc glossary](../../../glossary.md#term-javadoc))</span>
- Complete: [NN0033](tasks/0033-trainable-runtime-length-rnn-workflow.md#result), practical
  F32/F64 runtime-length RNN workflow, independent reviews, 620 affected tests/Javadocs and public proof.<span data-glossary-navigation="true"> ([F32 glossary](../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../glossary.md#floating-point-precision-and-rounding); [RNN glossary](../../../glossary.md#term-recurrent-neural-network-rnn))</span>

Earlier prose that named [CPU](../../../glossary.md#term-cpu) 0006D or tools/tuning 0002 as the global frontier, called 0010/0011
or 0025A Draft/[Ready](../../../glossary.md#term-ready), treated 0021A's planned [Model](../../../glossary.md#term-model) surface as still wholly unpublished, or called
0021B–0024 the active frontier has been normalized to the task table, linked task results, current
[dependency](../../../glossary.md#term-dependency) plans, and roadmap above.

## Live gates, decisions, and risks

- **Fluent [RNN](../../../glossary.md#term-recurrent-neural-network-rnn) migration Draft:** [NN0032](tasks/0032-fluent-rnn-general-scan-migration.md) records
  a confirmed advanced second-order CPU compile regression; no runtime Hessian proof or first-order
  migration waiver. Model0036's length-aware proposal does not resolve that gate. CP0013 closes
  all-valid persistence; Checkpoint0014 separately closes the static homogeneous v5 length wire,
  not NN0032's remaining engine/layout compatibility. Genericity does not prove Metal availability; compare actual qualified
  occurrences before claiming regression/parity. <span data-glossary-navigation="true"> ([CPU glossary](../../../glossary.md#term-cpu); [compile glossary](../../../glossary.md#compile); [regression glossary](../../../glossary.md#term-regression); [Hessian glossary](../../../glossary.md#term-hessian); [homogeneous glossary](../../../glossary.md#term-homogeneous); [engine glossary](../../../glossary.md#term-engine); [layout glossary](../../../glossary.md#layout); [Metal glossary](../../../glossary.md#term-metal))</span>[CP0013](../checkpoint/tasks/0013-general-scan-graph-persistence-decision.md)
  is Complete after final Cicero APPROVE/main configured gates. CPU0037's narrow read qualification
  is Complete. Cold output descriptor, remaining compatibility and fluent authority for NN0032
  remain held; existing <span data-glossary-navigation="true"> ([qualification glossary](../../../glossary.md#term-qualification); [descriptor glossary](../../../glossary.md#term-descriptor))</span>`.rnn` is unchanged. NN0033's explicit [runtime](../../../glossary.md#term-runtime)-length recipe is separate.
- **Bounded all-valid owned [scan](../../../glossary.md#term-cumulative-scan) Complete; [padding](../../../glossary.md#padding) separately gated:** Compiler0010 is Superseded, not pending fixed-adjoint approval.
  Completed Model0034 authority and Model0035 source/IR establish the source/body/residual foundation;
  Planning0008 transport and Compiler0011 scoped capture/inference, first-order AD and final
  body queries/demand and Prepare0010 scoped projection/resource handoff are Complete;<span data-glossary-navigation="true"> ([adjoint glossary](../../../glossary.md#term-adjoint); [IR glossary](../../../glossary.md#term-intermediate-representation-ir); [residual glossary](../../../glossary.md#scan-residual-port-and-history-demand); [capture glossary](../../../glossary.md#term-capture-meanings); [inference glossary](../../../glossary.md#term-inference); [AD glossary](../../../glossary.md#autograd); [handoff glossary](../../../glossary.md#term-handoff))</span>
  [CPU0030](../../backends/cpu/tasks/0030-general-scan-f32-pointwise-vertical.md) bounded F32 pointwise
  forward/actual automatic reverse and CPU0031–0033 declared F32/F64 closure/root join are Complete.
  Generic first-order low-level support
  cannot alone unlock Model.train. Arbitrary bodies may mix batch rows: per-row lengths need an approved
  separability/branch-before-padded-arithmetic design. NN0030 separately approves all-valid owned
  admission; NN0031 completed workflow qualification/lifecycle review without padding or wire support.
  Source validity/cold-copy support
  must not be gated by CPU qualification. Result masking is not skipping; static <span data-glossary-navigation="true"> ([F32 glossary](../../../glossary.md#floating-point-precision-and-rounding); [pointwise glossary](../../../glossary.md#term-pointwise); [F64 glossary](../../../glossary.md#floating-point-precision-and-rounding); [Model glossary](../../../glossary.md#term-model); [batch glossary](../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [admission glossary](../../../glossary.md#term-admission); [lifecycle glossary](../../../glossary.md#lifecycle))</span>`.rnn()` training
  and fixed scan inference remain.
- **Nonblocking cold-[prepare](../../../glossary.md#prepare) follow-up:** a development 3-step stacked-LSTM train was interrupted
  in active CPU topological preparation; reduced 2-step integration trains, and required 3-step RNN
  training/all-family 3-step stack numerics pass. Review found no acceptance gap, error or proved
  regression. <span data-glossary-navigation="true"> ([LSTM glossary](../../../glossary.md#term-long-short-term-memory-lstm); [topological glossary](../../../glossary.md#term-topological-order))</span>[CPU0026](../../backends/cpu/tasks/0026-stacked-lstm-cold-preparation-diagnostic.md)
  is Complete after Pasteur's binding APPROVE, zero findings, and main acceptance: repeated
  finish/topological scans dominate sampled CPU analysis. The separately user-authorized<span data-glossary-navigation="true"> ([binding glossary](../../../glossary.md#term-binding))</span>
  [CPU0027](../../backends/cpu/tasks/0027-unit-topology-adjacency-ordering.md) adjacency-ordering
  fix is Complete after independent APPROVE and validation; its public cold-preparation probe
  completed forward and combined preparation, but is not a training benchmark. Actual larger-case
  cached-training latency and a historical regression remain unmeasured; NN acceptance/frontier
  are unchanged.<span data-glossary-navigation="true"> ([validation glossary](../../../glossary.md#term-validation); [latency glossary](../../../glossary.md#term-latency); [NN glossary](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- **Completed 0021B baseline:** CPU0028 executes canonical scalar FLOAT32 RNN/GRU/LSTM in both
  directions and bias forms, with runtime INT64 lengths and all outputs through existing lifecycles.
  FLOAT64/low types, noncontiguous layouts, vector/parallel/native scan routes and tuning remain
  outside that capability; no Data/NN migration or BPTT is implied.<span data-glossary-navigation="true"> ([canonical glossary](../../../glossary.md#term-canonical-form); [scalar glossary](../../../glossary.md#term-scalar); [FLOAT32 glossary](../../../glossary.md#term-float32); [GRU glossary](../../../glossary.md#term-gated-recurrent-unit-gru); [bias glossary](../../../glossary.md#term-bias); [INT64 glossary](../../../glossary.md#data-type--datatype); [FLOAT64 glossary](../../../glossary.md#term-float64); [vector glossary](../../../glossary.md#term-vector); [BPTT glossary](../../../glossary.md#term-backpropagation-through-time-bptt))</span>
- **0022A Data gate:** [Data0001](../data/tasks/0001-data-extension-boundary.md) is Complete for
  the empty Data-only foundation; Data0002/0003 are Complete for host metadata and integral batching,
  not runtime binding. NN0022A's opt-in export is Complete after independent approval and main's
  architecture/root checkpoint. Text's sole Data API edge is approved and Text0001 is Complete; Vision still needs
  a separate module/edge decision. Ordinary right padding
  has one canonical valid length per batch row; padding lengths and dense masks are derived, not
  stored second sources of truth. NN0022 needs no Data module and retains existing static APIs;
  NN0022B is Complete for Model/builder inference before NN0022A; neither NN-only task depends on Data.
  NN0022A now authorizes only the approved export and numeric integration, not fluent Text or BPTT.<span data-glossary-navigation="true"> ([host glossary](../../../glossary.md#term-host); [metadata glossary](../../../glossary.md#term-metadata); [checkpoint glossary](../../../glossary.md#term-checkpoint); [module glossary](../../../glossary.md#term-module); [valid length glossary](../../../glossary.md#term-valid-length); [masks glossary](../../../glossary.md#term-boolean-mask))</span>
- **Scan boundary:** specific runtime length values must not change Model topology or compiled
  graph structure. Invalid lengths fail before published-result mutation, padded coordinates are
  not recurrent inputs, and dense masking must not be described as skipped recurrent work. Static
  sequence APIs remain unchanged; completed NN0022 adds a dense API, not migration or deprecation.<span data-glossary-navigation="true"> ([topology glossary](../../../glossary.md#term-topology); [compiled graph glossary](../../../glossary.md#compiled-graph--compiledgraphmodel); [coordinates glossary](../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- **Differentiation gate:** fixed RecurrentScan backpropagation through time remains fail-closed and
  belongs to Compiler. Ordinary static unrolled recurrence depends on normal constituent-operation
  adjoints; it is not covered by a blanket recurrent-autograd prohibition. Conv3d gradients remain
  fail-closed while Compiler 0006C is Draft. NN/Training must not imply either blocked capability.<span data-glossary-navigation="true"> ([backpropagation through time glossary](../../../glossary.md#backpropagation-through-time-and-recurrent-state))</span><span data-glossary-navigation="true"> ([fail-closed glossary](../../../glossary.md#term-fail-closed); [Compiler glossary](../../../glossary.md#term-compiler); [unrolled glossary](../../../glossary.md#backpropagation-through-time-and-recurrent-state); [operation glossary](../../../glossary.md#operation); [autograd glossary](../../../glossary.md#autograd); [Conv3d glossary](../../../glossary.md#conv3d--ncdhw); [gradients glossary](../../../glossary.md#term-gradient))</span>
- **[Persistence](../../../glossary.md#term-persistence)/training gate:** NN owns in-memory state paths and strict Tensor-reference
  load, with declarative reconstruction delivered in 0026. Durable artifacts belong to<span data-glossary-navigation="true"> ([Tensor glossary](../../../glossary.md#tensor); [artifacts glossary](../../../glossary.md#term-artifact))</span>
  [Checkpoint](../checkpoint/master-plan.md), whose published explicit APIs remain implemented;
  optimizer/session/update orchestration belongs to <span data-glossary-navigation="true"> ([optimizer glossary](../../../glossary.md#term-optimizer))</span>[Training](../training/master-plan.md).
- **Still-open [API](../../../glossary.md#term-application-programming-interface-api) decisions:** select a persistent codec only with a concrete consumer; add a
  frozen padding row only with an owning gradient/replacement/update contract; add a shared
  recurrent abstraction, new initialization preset, configurable convolution fan value, or
  arbitrary mask only when a concrete consumer proves the contract; likewise defer public state-
  schema inspection or model-wide deferred-binding lifecycle until a consumer requires it.<span data-glossary-navigation="true"> ([codec glossary](../../../glossary.md#term-codec); [convolution glossary](../../../glossary.md#term-channels-first-convolution-layers); [schema glossary](../../../glossary.md#term-schema))</span>
- **Primary risks:** reverse dependencies or access to Compiler/Runtime/Prepare/backend internals;
  bypassing the bounded owned Model lifecycle through public Training/Checkpoint/Engine edges; hidden runtime or random
  state; partially published inferred parameters; confusing static unrolling with runtime scan;
  specializing graphs to host lengths; erasing LSTM cell state in a shared abstraction; or moving
  tokenization/padding/checkpoint transport into NN.<span data-glossary-navigation="true"> ([backend glossary](../../../glossary.md#backend); [owned Model glossary](../../../glossary.md#model-module-buffer-and-scan-distinctions); [parameters glossary](../../../glossary.md#term-parameter); [unrolling glossary](../../../glossary.md#term-unrolling); [graphs glossary](../../../glossary.md#graph); [tokenization glossary](../../../glossary.md#frozen-whitespace-tokenizer))</span>

## History and update policy

Detailed completion evidence, [validation](../../../glossary.md#term-validation) commands and results, file inventories, context IDs,
audit trails, and superseded frontier narratives remain in linked task files and Git history.
They are not default executor input. Current source, focused tests, authoritative contracts, this
table, and the current task brief take precedence over historical narrative.

New or materially replanned tasks use the [compact task-brief format](../../planning-guide.md#task-brief-format).
Completed tasks need no retrospective rewrite. Update this plan only for [ownership](../../../glossary.md#term-ownership-meanings)/package
direction, task order or status, live dependencies/gates, risks, milestones, or frontier changes.

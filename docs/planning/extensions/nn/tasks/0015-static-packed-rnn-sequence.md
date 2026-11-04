# Task 0015: Static Packed [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn) Sequence

> **Editorial reading context — not a status update.** Java lengths are read during
> expression construction, so they determine which original rows appear in each compact step
> batch. The sequence container calls its owned cell once per represented step and assembles
> final hidden rows in original order. A list of compact step outputs is not a dense padded
> Tensor. Numeric zeros never decide activity. Today's [static recurrent contract](../../../../reference/nn/recurrent-layers.md#current-nn-recurrent-composition-contract)
> explains packing, while [runtime-length composition](../../../../reference/nn/recurrent.md#nn-runtime-length-recurrent-composition)
> explains the separate case in which new count values reuse one source instead of specializing a new unroll.

## Status

Complete

## Goal

Add one final public vanilla-[RNN](../../../../glossary.md#term-recurrent-neural-network-rnn) sequence container that statically unrolls a [time-major](../../../../glossary.md#term-time-major) [Tensor](../../../../glossary.md#tensor)
expression from explicit construction-time sequence lengths. At each represented time step it
gathers only rows whose sequence is still active, invokes the owned `RnnCell` once on that compact
[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters), exposes the exact compact step outputs, and restores each sequence's final hidden row to
original batch order. A zero Tensor value never means [padding](../../../../glossary.md#padding).

This task also records the recurrent-scan decision forced by current contracts:

```text
static input Shape + caller Java long[] lengths
  -> construction-time active-row sets are known
  -> ordinary SELECT + eager integer indices + GATHER + RnnCell + STACK can express the graph

runtime Tensor lengths or mask
  -> active rows and carried-state Shape depend on runtime values
  -> WHERE would still construct cell work for padded rows
  -> a genuine recurrent Model scan/control-flow contract is required and remains deferred
```

The first container is deliberately cell-specific. `RnnCell` and `GruCell` each [carry](../../../../glossary.md#term-carry) one hidden
Tensor, while `LstmCell` carries hidden and cell Tensors and returns a distinct result type. No
current consumer proves a shared sequence abstraction that preserves those [signatures](../../../../glossary.md#term-signature) without
state erasure. Later Draft tasks may reuse the proven packing policy for [GRU](../../../../glossary.md#term-gated-recurrent-unit-gru) and then extend it
truthfully for [LSTM](../../../../glossary.md#term-long-short-term-memory-lstm) final hidden and cell-state [capture](../../../../glossary.md#term-capture-meanings).

## [Scope](../../../../glossary.md#term-scope)

- Add final public `io.github.pho001.synaptik.nn.layers.RnnSequence` extending
  `io.github.pho001.synaptik.nn.module.Module` directly.
- Add final public record `io.github.pho001.synaptik.nn.layers.RnnSequenceForwardResult` with
  exactly the components `List<Tensor> packedOutputs` and `Tensor finalHidden`, in that order.
- Add exactly the constructor, cell accessor, and three-argument forward method in the public API
  table below. Add no overload, builder, options object, generic recurrent interface, functional
  facade, direction enum, size getter, mask overload, or default state.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [facade glossary](../../../../glossary.md#term-facade); [mask glossary](../../../../glossary.md#term-boolean-mask))</span>
- Permanently register the exact caller-supplied `RnnCell` as the sole child under local name
  `cell`. The container declares no direct [parameter](../../../../glossary.md#term-parameter) or [buffer](../../../../glossary.md#term-buffer) and retains no sequence state.
- Accept a [time-major](../../../../glossary.md#term-time-major) input with exact fully static [rank](../../../../glossary.md#term-rank)-three [Shape](../../../../glossary.md#shape)
  `[time, batch, inputSize]`, an exact fully static [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) initial hidden Shape
  `[batch, hiddenSize]`, and one snapshotted Java `long[]` length per original [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) row.
- Require every length to be in `[0, time]` and the maximum length to fit Java list indexing
  (`<= Integer.MAX_VALUE`). Lengths are explicit metadata and are never inferred from input
  values, hidden values, labels, storage, gradients, or padding contents.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [gradients glossary](../../../../glossary.md#term-gradient); [padding glossary](../../../../glossary.md#padding))</span>
- Clone the caller's lengths before traversing them. Do not retain either the caller array or the
  clone after forward construction returns.
- Use stable active-batch compaction without sorting: step `t` contains original row indices `b`
  in ascending order exactly when `lengths[b] > t`. The result must therefore preserve original
  relative order at every step and need no inverse permutation.<span data-glossary-navigation="true"> ([permutation glossary](../../../../glossary.md#permutation))</span>
- Statically unroll exactly `max(lengths)` cell steps. At each step select the time slice, gather
  its active original rows, gather the corresponding carried hidden rows, and call the owned cell
  once on Shapes <span data-glossary-navigation="true"> ([gather glossary](../../../../glossary.md#gather))</span>`[activeCount, inputSize]` and `[activeCount, hiddenSize]`.
- Create ordinary [eager](../../../../glossary.md#term-eager-evaluation) dense-[contiguous](../../../../glossary.md#term-contiguous) unlabeled [provenance](../../../../glossary.md#provenance)-free [INT64](../../../../glossary.md#data-type--datatype) index leaves with
  `requiresGrad == false` for each required gather. Index leaves are Model data, not Parameters,
  Buffers, module state, public packing metadata, or retained configuration.<span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- At step zero, use one active-original-index leaf for both input and initial-hidden gathers. At
  every later step, use one active-original-index leaf for input and one relative-survivor-index
  leaf for the previous packed hidden. Do not special-case identity gathers; the fixed producer
  structure keeps compaction provenance uniform and testable.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [producer glossary](../../../../glossary.md#term-producer))</span>
- Return an [immutable](../../../../glossary.md#term-immutable) [snapshot](../../../../glossary.md#term-snapshot) with exactly one packed output [Tensor](../../../../glossary.md#tensor) per constructed time step.
  `packedOutputs.get(t)` is the exact `RnnCell.forward` result shaped
  `[activeCount(t), hiddenSize]`, with rows in ascending original batch order.
- Restore final hidden state in original batch order. For row `b`, use the exact row selected from
  `initialHidden` when `lengths[b] == 0`; otherwise select the row corresponding to `b` from
  `packedOutputs.get(lengths[b] - 1)`. Stack those rows on [axis](../../../../glossary.md#term-axis) zero.
- When the batch is empty or every length is zero, return the exact `initialHidden` reference as
  `finalHidden` and an empty packed-output list, creating no Tensor or cell expression.
- Complete caller-controlled descriptor, length, type, feature, and step-shape validation before
  creating the first index leaf or expression. Once construction begins, preserve the exact
  no-rollback side effects described below.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [feature glossary](../../../../glossary.md#term-feature); [validation glossary](../../../../glossary.md#term-validation); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- Keep [forward mode](../../../../glossary.md#term-forward-mode)-insensitive. It accepts no `ForwardContext`, reads neither container nor cell
  mode, and does not mutate mode or module state. Existing recursive `train()`/`eval()` calls
  still propagate through the registered child normally.
- Preserve stable child identity, cell parameter replacement behavior, recursive discovery, and
  state-dictionary paths through existing [`Module`](../../../../glossary.md#term-module), `Parameter`, and `RnnCell` contracts.
- Add focused exact-surface/result-carrier, ownership, static packing, validation-order,
  provenance, final-state restoration, replacement, mode, zero-length, and exclusion tests.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Add complete type, record-component, constructor, member, and package Javadocs. After executable
  work and final NN testing, use a separate clean documentation-focused context to finalize the
  Javadocs, Training API explanation, glossary impact, planning evidence, generated Javadoc, and
  documented no-change conclusions.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [planning glossary](../../../../glossary.md#term-planning); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

For lengths `[5, 3, 1]`, Java calls `cell.forward` five times, once per non-empty time step. Their
active batch extents are `[3, 2, 2, 1, 1]`, so the [graph](../../../../glossary.md#graph) represents nine logical recurrent row
applications instead of the fifteen rows in a dense padded batch. “Nine” is not nine Java method
calls: one cell call constructs a batched expression for each active time [slice](../../../../glossary.md#slice).

## Out of [scope](../../../../glossary.md#term-scope)

- Runtime-dynamic Tensor lengths, a Tensor Boolean mask, a Java mask overload, synthesized masks,
  sentinel lengths, padding indices, or inference of padding from zero-valued data.<span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#masks-padding-and-eager-evaluation))</span><span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Tensor glossary](../../../../glossary.md#tensor); [Boolean mask glossary](../../../../glossary.md#term-boolean-mask); [padding glossary](../../../../glossary.md#padding); [inference glossary](../../../../glossary.md#term-inference-meanings))</span>
- Calling a cell on the full [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) and applying `where`, multiplication, zeroing, or another
  post-cell mask. Such composition would still represent padded cell work and does not satisfy
  packed-sequence semantics.
- A general recurrent Model scan, loop/body graph, carried-value tuple, dynamic active set,
  runtime-dependent result Shape, early exit, condition operation, or control-flow region.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [graph glossary](../../../../glossary.md#graph); [Shape glossary](../../../../glossary.md#shape); [operation glossary](../../../../glossary.md#operation))</span>
- Reinterpreting fixed associative `CUM_SUM` or `CUM_PROD` as recurrent scan. Their operation
  contracts have no cell body, carried hidden value, active-set transition, or final-state exit
  capture.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- `GruSequence`, `LstmSequence`, shared `RecurrentCell`, shared `RecurrentSequence`, generic state
  tuple, erased collection of states, or a public `PackedSequence` abstraction. GRU and LSTM are
  later cohesive tasks after this concrete contract is proven.<span data-glossary-navigation="true"> ([GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm))</span>
- Sorting by descending length, an exposed permutation, inverse-permutation metadata, unstable
  compaction, or reordering equal or unequal length rows. This task preserves original order
  throughout.<span data-glossary-navigation="true"> ([permutation glossary](../../../../glossary.md#permutation); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Dense padded output, caller-selected padding values, padding materialization, unpacking to a
  rectangular time/batch result, batch-first input, direction configuration, reverse traversal,
  bidirectionality, stacked recurrent layers, residuals, dropout, attention, or state detachment.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- Retaining initial, intermediate, or final hidden Tensors, lengths, index leaves, outputs, or a
  step counter in a field, Buffer, Parameter, state dictionary, thread-local, static, runtime
  object, session, or another hidden lifecycle.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Default or zero initial hidden construction, state initializers, reset APIs, stateful forward
  overloads, numerical hidden mutation, or in-place compaction.
- `UnaryTensorModule`, participation in `Sequential`, an adapter into `Sequential`, or changes to
  `Sequential`, `UnaryTensorModule`, [`Module`](../../../../glossary.md#term-module), [`Parameter`](../../../../glossary.md#term-parameter), [`Buffer`](../../../../glossary.md#term-buffer), or the recurrent cells.
- Backpropagation through time, gradient rule work, optimizer/session orchestration, checkpoint
  transport, serialization, graph capture, compiler scheduling, Runtime/Prepare/Engine behavior,
  backend lowering, kernels, numerical execution, or end-to-end support claims.<span data-glossary-navigation="true"> ([Backpropagation through time glossary](../../../../glossary.md#term-backpropagation-through-time-bptt); [gradient glossary](../../../../glossary.md#term-gradient); [optimizer glossary](../../../../glossary.md#term-optimizer); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [serialization glossary](../../../../glossary.md#term-serialization); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel))</span>
- A Model, Training executable API, Gradle, dependency, architecture-contract, ADR,
  architecture-test, global-roadmap, CPU, or backend source change during implementation.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Detailed task specifications or implementation for [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0016 or later.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [ADR 0007: Neural-network module and training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [Training API](../../../../api/training-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [NN master plan](../master-plan.md)<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- [Completed NN task 0011: Unary composition](0011-unary-tensor-module-composition-and-sequential.md)<span data-glossary-navigation="true"> ([Unary glossary](../../../../glossary.md#term-unary))</span>
- [Completed NN task 0012: Vanilla RNN cell](0012-vanilla-rnn-cell.md)<span data-glossary-navigation="true"> ([RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn))</span>
- [Completed NN task 0013: GRU cell](0013-gru-cell.md)<span data-glossary-navigation="true"> ([GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru))</span>
- [Completed NN task 0014: LSTM cell](0014-lstm-cell.md)<span data-glossary-navigation="true"> ([LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm))</span>
- [Completed Model task 0017H: Slice expressions](../../../modules/model/tasks/0017h-slice-tensor-expressions.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Slice glossary](../../../../glossary.md#slice))</span>
- [Completed Model task 0018D: Gather expressions](../../../modules/model/tasks/0018d-axis-gather-tensor-expressions.md)<span data-glossary-navigation="true"> ([Gather glossary](../../../../glossary.md#gather))</span>
- [Completed Model task 0017L: Composition expressions](../../../modules/model/tasks/0017l-tensor-composition-expressions.md)
- [Completed Model task 0023E: Cumulative scan normalization](../../../modules/model/tasks/0023e-cumulative-scan-normalization-and-product.md)<span data-glossary-navigation="true"> ([Cumulative scan glossary](../../../../glossary.md#cumulative-scan); [normalization glossary](../../../../glossary.md#term-normalization))</span>

## Architecture constraints

- `extensions/nn` continues to depend only on `modules/model`. Production must not import
  Training, Compiler, Runtime, Prepare, Engine, CPU, Metal, CUDA, or another backend.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal); [CUDA glossary](../../../../glossary.md#term-cuda); [backend glossary](../../../../glossary.md#backend))</span>
- NN owns the cell-specific module composition, construction-time length policy, and NN result
  carrier. Model remains the sole owner of SELECT, GATHER, STACK, eager leaves, type promotion,
  Shape algebra, descriptors, layouts, provenance, Tensor identity, and cell operation semantics.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Model glossary](../../../../glossary.md#term-model); [owner glossary](../../../../glossary.md#term-owner); [GATHER glossary](../../../../glossary.md#gather); [eager glossary](../../../../glossary.md#term-eager-evaluation); [Shape glossary](../../../../glossary.md#shape); [descriptors glossary](../../../../glossary.md#term-descriptor); [layouts glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance); [Tensor glossary](../../../../glossary.md#tensor); [identity glossary](../../../../glossary.md#term-identity-meanings); [operation glossary](../../../../glossary.md#operation))</span>
- Tensor identity, descriptors, storage association, and provenance remain immutable. The
  sequence container constructs new leaves and expressions but neither evaluates nor mutates
  them.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- Recurrent hidden state is caller-threaded Tensor data, never module-owned persistent state. The
  registered `RnnCell` owns only its [trainable parameters](../../../../glossary.md#trainable-parameter).
- [`Module`](../../../../glossary.md#term-module) retains no universal forward method. `RnnSequence` is a direct subclass with a
  truthful input/state/length signature and must not extend <span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>`UnaryTensorModule`.
- `Sequential` remains a container only for `UnaryTensorModule`; no adapter may capture, erase,
  default, or synthesize recurrent state or lengths.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [recurrent state glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state))</span>
- Static Java lengths are construction metadata, not graph values. Their snapshot may decide how
  many expressions and what static Shapes to construct. A runtime Tensor mask cannot make those
  same decisions under the current storage-free expression contract.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [graph glossary](../../../../glossary.md#graph); [snapshot glossary](../../../../glossary.md#term-snapshot); [mask glossary](../../../../glossary.md#term-boolean-mask))</span>
- Current `CUM_SUM` and `CUM_PROD` accept one Tensor and fixed associative operation semantics.
  They cannot express an arbitrary cell body with carried hidden state, active-batch compaction,
  or exit-state capture and must not be reused or renamed for this purpose.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- `RnnSequenceForwardResult` is an NN composition value, not a Model multi-output producer,
  Tensor, module state, checkpoint payload, runtime result, or execution handle.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>
- Construction produces eager host-backed index leaves and storage-free expression metadata only.
  It does not prove graph capture, gradient implementation, compiled execution, backend support,
  numerical values, skipped kernels, or storage residency. “Skipped padded work” means no cell
  expression contains a padded logical row.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [gradient glossary](../../../../glossary.md#term-gradient); [kernels glossary](../../../../glossary.md#kernel); [residency glossary](../../../../glossary.md#residency))</span>
- If implementation needs a new Model method or operation, runtime Tensor mask/length input,
  dynamic Shape, scan/body contract, shared recurrent abstraction, state Buffer, dependency,
  architecture rule, or a path outside the exact task scope, stop and report the exact blocker
  instead of widening this task.<span data-glossary-navigation="true"> ([Buffer glossary](../../../../glossary.md#term-buffer); [dependency glossary](../../../../glossary.md#term-dependency); [scope glossary](../../../../glossary.md#term-scope); [widening glossary](../../../../glossary.md#term-widening))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.nn.layers`
- `io.github.pho001.synaptik.nn.module`
- `io.github.pho001.synaptik.model.tensor`
- existing [Model](../../../../glossary.md#term-model) [descriptor](../../../../glossary.md#term-descriptor), data-type, [Shape](../../../../glossary.md#shape), and [layout](../../../../glossary.md#layout) packages needed by [eager](../../../../glossary.md#term-eager-evaluation) [INT64](../../../../glossary.md#data-type--datatype) leaves

Packages added or changed:

- `io.github.pho001.synaptik.nn.layers` — two public types and existing package documentation only

Type placement:

- `io.github.pho001.synaptik.nn.layers.RnnSequence` — NN owns the cell-specific module
  composition and construction-time packing policy.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [module glossary](../../../../glossary.md#term-module))</span>
- `io.github.pho001.synaptik.nn.layers.RnnSequenceForwardResult` — NN owns the typed result of
  that composition; it is not a Model producer result or runtime value.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>

No package is added. `nn.module`, Model packages, Training, [compiler](../../../../glossary.md#term-compiler)/[runtime](../../../../glossary.md#term-runtime)/[prepare](../../../../glossary.md#prepare)/[engine](../../../../glossary.md#term-engine), and
[backend](../../../../glossary.md#backend) packages remain unchanged.

## Public [API](../../../../glossary.md#term-application-programming-interface-api)

`RnnSequence` declares exactly:

```java
public RnnSequence(RnnCell cell)

public RnnCell cell()

public RnnSequenceForwardResult forward(
        Tensor input,
        Tensor initialHidden,
        long[] lengths)
```

| Member | Exact contract |
|---|---|
| `RnnSequence(cell)` | Permanently registers and retains the exact available non-null cell under child name `cell`; declares no direct state. |
| `cell()` | Returns the exact stable registered child reference. |
| `forward(input, initialHidden, lengths)` | Snapshots and validates static inputs and lengths, constructs stable compact time-step expressions, and returns [immutable](../../../../glossary.md#term-immutable) packed outputs plus original-order final hidden state. |

The class declares no other public or protected constructor, method, field, nested type,
interface, or overload. Inherited final [`Module`](../../../../glossary.md#term-module) APIs remain available normally.

`RnnSequenceForwardResult` declares exactly:

```java
public record RnnSequenceForwardResult(
        List<Tensor> packedOutputs,
        Tensor finalHidden)
```

Its [canonical](../../../../glossary.md#term-canonical-form) constructor rejects null `packedOutputs`, traverses once to reject the first null
[element](../../../../glossary.md#term-element), [captures](../../../../glossary.md#term-capture-meanings) an immutable structural [snapshot](../../../../glossary.md#term-snapshot) retaining every exact [Tensor](../../../../glossary.md#tensor) reference, and
then rejects null `finalHidden`. A success performs no [descriptor](../../../../glossary.md#term-descriptor) relationship [validation](../../../../glossary.md#term-validation),
expression construction, Tensor copy, mutation, [evaluation](../../../../glossary.md#term-evaluation), or Tensor-ID allocation. Generated
record accessors expose the immutable list and exact final-hidden reference. The record declares
no additional public method or nested type beyond its canonical constructor and ordinary
generated accessors, `equals`, [`hashCode`](../../../../glossary.md#hash-codes-and-content-digests), and `toString`.

## State [schema](../../../../glossary.md#term-schema) and [ownership](../../../../glossary.md#term-ownership-meanings)

`RnnSequence` declares no direct [Parameter](../../../../glossary.md#term-parameter) or [Buffer](../../../../glossary.md#term-buffer). Its sole child is the exact supplied cell:

| Kind | Local name | Value |
|---|---|---|
| Child | `cell` | exact caller-supplied `RnnCell` |

Direct `parameters()` and `buffers()` are empty. `children()` contains exactly `cell` in its sole
entry. Recursive parameter and state-dictionary paths are `cell.inputWeight`,
`cell.hiddenWeight`, and optional `cell.bias`, preserving the cell's declaration order. Input,
initial hidden, lengths, [eager](../../../../glossary.md#term-eager-evaluation) index leaves, selected time slices, compact hidden values, packed
outputs, and final hidden never enter [module](../../../../glossary.md#term-module) discovery or a state dictionary.

Construction rejects null `cell` before child registration. Ordinary `Module.child` [validation](../../../../glossary.md#term-validation)
then rejects a previously owned cell and preserves existing ownership. A successful constructor
does not copy the cell or synchronize its pre-existing mode with the new parent's initial mode.
Later `train()` and `eval()` calls on the sequence propagate through the child under existing
Module semantics. Forward is mode-insensitive.

## Forward input contract

The caller supplies:

| Input | Required contract |
|---|---|
| `input` | non-null floating fully static [rank](../../../../glossary.md#term-rank)-three [Tensor](../../../../glossary.md#tensor) shaped `[time, batch, inputSize]` |
| `initialHidden` | non-null floating fully static [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) Tensor shaped `[batch, hiddenSize]` |
| `lengths` | non-null Java array with exactly [`batch`](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) [elements](../../../../glossary.md#term-element), each in `[0, time]`, with maximum at most `Integer.MAX_VALUE`; snapshotted and not retained |

`time` and `batch` may be zero. `inputSize` and `hiddenSize` must equal the positive static sizes
configured by the current cell [weights](../../../../glossary.md#term-weight). Input and hidden types may differ only when the complete
cell projections and addition accept them under existing [Model](../../../../glossary.md#term-model) [numeric promotion](../../../../glossary.md#numeric-promotion). Both must be
floating. Fully static [Shapes](../../../../glossary.md#shape) are required because Java lengths decide concrete time traversal,
active batch extents, [eager](../../../../glossary.md#term-eager-evaluation) index storage, and final row restoration before [runtime](../../../../glossary.md#term-runtime).

The caller owns input, initial hidden, and the original lengths array. Forward does not mutate,
retain, label, evaluate, copy, or attach either Tensor. It clones the array so mutation racing
after the [snapshot](../../../../glossary.md#term-snapshot) cannot change the active-set plan. The method provides no [synchronization](../../../../glossary.md#term-synchronization) with
concurrent mutation of the caller array before cloning or concurrent cell [parameter](../../../../glossary.md#term-parameter) replacement.

## Packing and final-state semantics

For snapshotted length `L[b]`, define:

```text
steps = max(L), or zero for an empty batch
active(t) = [b in 0..<batch, in ascending order, where L[b] > t]
activeCount(t) = size(active(t))
```

For each `t` from zero through `steps - 1`:

1. select `input` [axis](../../../../glossary.md#term-axis) zero at `t`, yielding `[batch, inputSize]`;
2. create one [INT64](../../../../glossary.md#data-type--datatype) [eager](../../../../glossary.md#term-eager-evaluation) [leaf](../../../../glossary.md#term-leaf) holding `active(t)`;
3. [gather](../../../../glossary.md#gather) the selected input on axis zero with that leaf, yielding
   `[activeCount(t), inputSize]`;
4. for `t == 0`, gather `initialHidden` on axis zero with the same active-original leaf;
5. for `t > 0`, create one INT64 leaf whose entries are the positions in `active(t - 1)` occupied
   by the surviving original rows in `active(t)`, then gather the prior packed output on axis
   zero, yielding `[activeCount(t), hiddenSize]`;
6. call `cell.forward(compactInput, compactHidden)` exactly once; and
7. append that exact returned [Tensor](../../../../glossary.md#tensor) to the result list.

Because `active(t)` is a subsequence of `active(t - 1)`, relative survivor positions are defined
without sorting. Every padded row `L[b] <= t` is absent from both compact operands and therefore
from that cell expression. Input values equal to numeric zero remain ordinary active data when
their length says they are active.

Final hidden [Shape](../../../../glossary.md#shape) is `[batch, hiddenSize]`, in original [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) order. For each original row:

```text
L[b] == 0  -> initialHidden.select(0, b)
L[b] > 0   -> packedOutputs[L[b] - 1].select(0, position of b in active(L[b] - 1))
```

When at least one length is positive, stack those exact [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) row selections on axis zero in
`b = 0..<batch` order. When no length is positive, return the exact `initialHidden` reference and
construct neither selections nor a stack.

Examples:

| Lengths | Active original rows by time | Final row sources |
|---|---|---|
| `[5, 3, 1]` | `[0,1,2]`, `[0,1]`, `[0,1]`, `[0]`, `[0]` | row 0 from step 4; row 1 from step 2; row 2 from step 0 |
| `[1, 3, 2]` | `[0,1,2]`, `[1,2]`, `[1]` | row 0 from step 0 position 0; row 1 from step 2 position 0; row 2 from step 1 position 1 |
| `[0, 2, 0]` | `[1]`, `[1]` | rows 0 and 2 from initial hidden; row 1 from step 1 |
| `[0, 0]` | none | exact initial-hidden Tensor |

The result is packed by time step rather than flattened into one Tensor. Step boundaries and
active extents therefore remain explicit without a second public batch-size array or a generic
packed-sequence value. It contains no padded dense output.

## [Validation](../../../../glossary.md#term-validation) and side-effect order

`forward(input, initialHidden, lengths)` performs exactly:

1. reject null `input`, then null `initialHidden`, then null `lengths`;
2. clone `lengths` and use only that [snapshot](../../../../glossary.md#term-snapshot) afterward;
3. read the current cell input-weight, hidden-weight, and optional-bias bindings once for
   descriptor preflight only, in cell declaration order;<span data-glossary-navigation="true"> ([weight glossary](../../../../glossary.md#term-weight); [bias glossary](../../../../glossary.md#term-bias))</span>
4. revalidate the current cell [parameter](../../../../glossary.md#term-parameter) [schema](../../../../glossary.md#term-schema) and derive exact positive `inputSize` and
   `hiddenSize` without creating a [Tensor](../../../../glossary.md#tensor);
5. require input floating type, [rank](../../../../glossary.md#term-rank) exactly three, and fully static [Shape](../../../../glossary.md#shape);
6. require initial-hidden floating type, rank exactly two, and fully static Shape;
7. require input [axis](../../../../glossary.md#term-axis) two to equal `inputSize` and initial-hidden axis one to equal `hiddenSize`;
8. require the input and initial-hidden [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) extents to be equal;
9. require snapshotted length count to equal batch, then validate each element in original order
   as non-negative and no greater than time while deriving the maximum, and require that maximum
   to be at most <span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element))</span>`Integer.MAX_VALUE` for [immutable](../../../../glossary.md#term-immutable) Java-list indexing;
10. prevalidate the exact step-zero and survivor compact Shapes, cell projection promotion, ADD
    broadcast, and TANH eligibility for every distinct active count required by the snapshot;<span data-glossary-navigation="true"> ([TANH glossary](../../../../glossary.md#term-tanh))</span>
11. prevalidate every final row selection and, when needed, final STACK Shape/type contract; and
12. only after all caller-controlled checks succeed, construct eager indices and expressions in
    the fixed packing order above.

Validation must use [Model](../../../../glossary.md#term-model)-owned helpers or equivalent non-constructing [descriptor](../../../../glossary.md#term-descriptor) algebra. It
must not call `cell.forward` merely to discover whether a Shape is valid. Every null, rank,
dynamic-Shape, type, [feature](../../../../glossary.md#term-feature), batch, length-count, length-range, promotion, broadcast, select,
[gather](../../../../glossary.md#gather), or stack failure is detected before the first new Tensor [identity](../../../../glossary.md#term-identity-meanings), [eager](../../../../glossary.md#term-eager-evaluation) storage
allocation, or expression [producer](../../../../glossary.md#term-producer).

The descriptor [preflight](../../../../glossary.md#term-preflight) snapshot does not make the entire sequence call [atomic](../../../../glossary.md#resource-transaction-and-rollback) with concurrent
parameter replacement. Each actual `cell.forward` call observes the cell's then-current [bindings](../../../../glossary.md#term-binding)
under its existing contract; external replacement during one sequence construction can therefore
make later steps use different compatible parameter Tensors. Callers must coordinate replacement
and forward construction when a consistent multi-step parameter snapshot matters.

After construction begins, successful eager index leaves, SELECT/GATHER producers, cell
expression prefixes, row selections, and [host](../../../../glossary.md#term-host) allocations are not rolled back if Tensor-ID
exhaustion, allocation failure, or another unexpected delegated failure occurs later. No partial
result carrier is returned. The container itself retains none of the created values after the
call exits except through references reachable from the returned result expressions.

## Exact expression and [provenance](../../../../glossary.md#provenance) contract

- Every time step begins with one `input.select(0, t)` whose exact input is the caller's input.
- Each active-original index is an independent [eager](../../../../glossary.md#term-eager-evaluation) [INT64](../../../../glossary.md#data-type--datatype) [leaf](../../../../glossary.md#term-leaf) shaped `[activeCount(t)]`, dense
  contiguous, unlabeled, host-backed, gradient-ineligible, and provenance-free.<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [host glossary](../../../../glossary.md#term-host))</span>
- Step-zero input and initial-hidden gathers share that step's exact active-original index leaf.
- Each later relative-survivor index is an independent eager INT64 leaf shaped
  `[activeCount(t)]` with the same [metadata](../../../../glossary.md#term-metadata) policy.
- Every gather uses axis zero and preserves exact source/index references in ordinary Model
  provenance.<span data-glossary-navigation="true"> ([gather glossary](../../../../glossary.md#gather); [axis glossary](../../../../glossary.md#term-axis); [Model glossary](../../../../glossary.md#term-model))</span>
- Every packed output is exactly the TANH result returned by the owned cell, not a wrapper,
  relabel, copy, extra identity expression, or dense padded reconstruction.<span data-glossary-navigation="true"> ([TANH glossary](../../../../glossary.md#term-tanh); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Final row selections use axis zero and exact source positions. A nontrivial final hidden is the
  exact axis-zero STACK result over original-order row selections.
- The container creates no direct [`Operation`](../../../../glossary.md#operation), producer, descriptor, or Tensor wrapper outside
  public Model construction APIs.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [descriptor glossary](../../../../glossary.md#term-descriptor); [Tensor glossary](../../../../glossary.md#tensor))</span>

This provenance specifies expression metadata only. It does not assert numerical results,
[gradient](../../../../glossary.md#term-gradient) traversal, [compiler](../../../../glossary.md#term-compiler) [capture](../../../../glossary.md#term-capture-meanings), [backend](../../../../glossary.md#backend) [lowering](../../../../glossary.md#lowering), [kernel](../../../../glossary.md#kernel) count, work scheduling, or actual
[runtime](../../../../glossary.md#term-runtime) avoidance of a padded kernel.

## Affected files

Implementation is limited to these nine paths:

1. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/RnnSequence.java`
2. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/RnnSequenceForwardResult.java`
3. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/package-info.java`
4. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/RnnSequenceTest.java`
5. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/RnnSequencePackingTest.java`
6. `docs/api/training-api.md`
7. `docs/glossary.md`
8. `docs/planning/extensions/nn/tasks/0015-static-packed-rnn-sequence.md`
9. `docs/planning/extensions/nn/master-plan.md`

Do not edit `ARCHITECTURE.md`, architecture explanations, ADRs, [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api), Training Java source,
other tasks/master plans, global roadmap, build files, [Model](../../../../glossary.md#term-model) source/tests, recurrent-cell source,
architecture tests, [conformance](../../../../glossary.md#term-conformance) tests, integration tests, [compiler](../../../../glossary.md#term-compiler)/[runtime](../../../../glossary.md#term-runtime)/[prepare](../../../../glossary.md#prepare)/[engine](../../../../glossary.md#term-engine) code, or
[backend](../../../../glossary.md#backend) code. If a required change falls outside these nine paths, stop and report it.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most the nine exact paths listed above. If a tenth path, a new
package, or a change to an excluded path is needed, stop and propose a follow-up task or report an
architecture/[API](../../../../glossary.md#term-application-programming-interface-api) blocker.

## Test requirements

### Public surface and result value

- Verify `RnnSequence` is final, extends [`Module`](../../../../glossary.md#term-module) directly, and declares exactly its constructor,
  accessor, and forward method with no public field, overload, nested type, mask input, length
  Tensor, or unary signature.<span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask); [Tensor glossary](../../../../glossary.md#tensor); [unary glossary](../../../../glossary.md#term-unary); [signature glossary](../../../../glossary.md#term-signature))</span>
- Verify the result record declares exactly `packedOutputs` then `finalHidden`, rejects null list,
  first null list element, then null final hidden, snapshots a mutable source list, exposes an
  unmodifiable list, and retains every exact Tensor reference.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- Verify generated record equality remains ordinary list/Tensor value equality and no custom
  behavior or descriptor coupling is added.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor))</span>

### [Ownership](../../../../glossary.md#term-ownership-meanings), state, replacement, and mode

- Verify the constructor registers the exact cell under `cell`, exposes it through `cell()`, and
  leaves direct parameters/buffers empty.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter); [buffers glossary](../../../../glossary.md#term-buffer))</span>
- Verify null and already-owned cells fail under documented order without changing existing
  ownership.
- Verify recursive parameter, state-dictionary, and child paths/order are exactly the existing
  cell schema prefixed by <span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>`cell.`.
- Verify `train()` and `eval()` propagate to the child; forward is identical in both modes and
  does not alter mode or state.
- Verify compatible cell-parameter replacement affects later sequence calls while an already
  constructed result retains its earlier exact expression inputs.

### Static packing and restoration

- For `[5, 3, 1]`, verify five packed outputs with active [Shapes](../../../../glossary.md#shape)
  `[3,H]`, `[2,H]`, `[2,H]`, `[1,H]`, `[1,H]`; active extents total nine rather than dense fifteen;
  and final rows select step/position `(4,0)`, `(2,1)`, `(0,2)` before original-order stacking.
- For unsorted `[1, 3, 2]`, inspect index [host](../../../../glossary.md#term-host) data and [provenance](../../../../glossary.md#provenance) to verify active-original sets
  `[0,1,2]`, `[1,2]`, `[1]`, relative survivor sets `[1,2]`, `[0]`, no sort/permutation, and
  final row sources <span data-glossary-navigation="true"> ([permutation glossary](../../../../glossary.md#permutation))</span>`(0,0)`, `(2,0)`, `(1,1)` in original order.
- Verify an active input row whose stored values are all numeric zero remains present. No storage
  read or value-based padding decision may occur.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding))</span>
- Verify `[0,2,0]` gathers only original row one, uses initial-hidden rows zero and two for final
  state, and constructs no cell operand for either padded row.
- Verify all-zero lengths, zero time with valid all-zero lengths, and an empty batch each return an
  empty immutable list plus the exact initial-hidden reference with no new Tensor ID.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [immutable glossary](../../../../glossary.md#term-immutable); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Verify final output Shapes/types and fixed SELECT/GATHER/cell/STACK producer order without
  evaluating numerical values.<span data-glossary-navigation="true"> ([GATHER glossary](../../../../glossary.md#gather); [producer glossary](../../../../glossary.md#term-producer))</span>

### [Validation](../../../../glossary.md#term-validation) and failure effects

- Cover null order, non-floating input/hidden, wrong ranks, every dynamic input/hidden axis,
  feature mismatch, batch mismatch, length-count mismatch, first negative length, first length
  above time, maximum length above <span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [feature glossary](../../../../glossary.md#term-feature); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>`Integer.MAX_VALUE`, and mixed-type promotion failure.
- Verify every caller-controlled failure occurs before the first Tensor ID or producer and leaves
  module state, cell state, and caller values unchanged.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [producer glossary](../../../../glossary.md#term-producer); [module glossary](../../../../glossary.md#term-module))</span>
- Verify the lengths array is snapshotted rather than retained or reread after [planning](../../../../glossary.md#term-planning).
- Verify failures after valid construction begins preserve already created eager leaves and
  expression prefixes without returning a result or mutating module state.<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation))</span>

### Exclusions

- Verify there is no Tensor-length/mask overload, Java mask overload, retained length/index/output
  field, Buffer state, hidden default, dense padded output, <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [mask glossary](../../../../glossary.md#term-boolean-mask); [Buffer glossary](../../../../glossary.md#term-buffer))</span>`Sequential` compatibility, shared
  recurrent abstraction, direct Model [`Operation`](../../../../glossary.md#operation) construction, or [dependency](../../../../glossary.md#term-dependency) outside [Model](../../../../glossary.md#term-model).
- Verify `CUM_SUM` and `CUM_PROD` kinds never occur in constructed sequence [provenance](../../../../glossary.md#provenance).

## Documentation requirements

The implementation context may draft complete public and package Javadocs while coding. After
the final executable [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) test [run](../../../../glossary.md#term-run), a separate clean documentation-focused context must read the
final source, tests, generated Javadocs, documentation rules, General/[API](../../../../glossary.md#term-application-programming-interface-api)-[Javadoc](../../../../glossary.md#term-javadoc)/[Planning](../../../../glossary.md#term-planning)/Example
profiles, Training API, glossary, NN master/task, completed cell tasks, and relevant [Model](../../../../glossary.md#term-model)
SELECT/[GATHER](../../../../glossary.md#gather)/STACK/cumulative-[scan](../../../../glossary.md#term-cumulative-scan) contracts. It must finalize:

- class, record, constructor, accessor, forward, component, failure, ownership, mode,
  concurrency, side-effect, and provenance Javadocs;<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [concurrency glossary](../../../../glossary.md#term-concurrency); [provenance glossary](../../../../glossary.md#provenance))</span>
- `layers/package-info.java` with static packed sequence semantics and its distinction from
  one-step cells, `Sequential`, gate-[parameter](../../../../glossary.md#term-parameter) packing, and [runtime](../../../../glossary.md#term-runtime) scan;
- `docs/api/training-api.md` with the current NN public surface, a short `[5,3,1]` example, exact
  result interpretation, and current-vs-planned execution boundary;
- `docs/glossary.md` definitions for static packed recurrent sequence, active batch,
  construction-time lengths, and recurrent scan, without implying zero-value padding; and<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [padding glossary](../../../../glossary.md#padding))</span>
- final planning evidence, no-change conclusions, links, anchors, status, and completion summary.

The documentation pass reuses stable Java test evidence unless it changes executable Java
behavior. It must state reasoned no-change conclusions for `ARCHITECTURE.md`, architecture
explanations/[ADR](../../../../glossary.md#term-architecture-decision-record-adr)/tests, [Tensor](../../../../glossary.md#tensor) API, Training Java API and [training graph](../../../../glossary.md#training-graph), Model capabilities,
[compiler](../../../../glossary.md#term-compiler)/runtime/[prepare](../../../../glossary.md#prepare)/[engine](../../../../glossary.md#term-engine), [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance), integration tests, [Gradle](../../../../glossary.md#term-gradle) dependencies,
global roadmap, and other [modules](../../../../glossary.md#term-module).

## Acceptance criteria

- The exact two-type public [API](../../../../glossary.md#term-application-programming-interface-api) exists with no extra overload or abstraction.
- The sequence owns exactly one `RnnCell` child and no direct or hidden state.
- Fully static time-major input plus snapshotted Java lengths constructs one compact cell
  expression per non-empty time step.<span data-glossary-navigation="true"> ([time-major glossary](../../../../glossary.md#term-time-major))</span>
- Packed outputs preserve stable original row order, exclude padded logical rows, and expose exact
  step boundaries without a dense padded result.
- Final hidden rows come from each sequence's exit step, with zero-length rows from initial hidden,
  and are restored to exact original batch order.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- Zero-valued data is never interpreted as [padding](../../../../glossary.md#padding); only explicit lengths control activity.
- All-zero, zero-time, and empty-batch requests have the documented exact-reference no-expression
  result.
- Caller-controlled validation precedes every new Tensor ID and expression; later failures retain
  documented prefixes without module mutation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Tensor glossary](../../../../glossary.md#tensor); [module glossary](../../../../glossary.md#term-module))</span>
- Result collection ownership, cell ownership, parameter replacement, mode, state paths, and
  concurrency limits match existing contracts.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [parameter glossary](../../../../glossary.md#term-parameter); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- No runtime Tensor mask/length promise, recurrent scan, cumulative-scan misuse, dense masking,
  shared recurrent abstraction, GRU/LSTM sequence, execution claim, dependency change, or
  out-of-scope file is introduced.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [mask glossary](../../../../glossary.md#term-boolean-mask); [GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm); [dependency glossary](../../../../glossary.md#term-dependency); [scope glossary](../../../../glossary.md#term-scope))</span>
- Focused and full NN tests, NN Javadocs, API-surface/import/dependency checks, Markdown/link gates,
  exact-scope checks, and whitespace validation pass.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- A separate clean documentation pass finalizes all affected documentation and records no-change
  conclusions before the task becomes Complete.

## Tests / [validation](../../../../glossary.md#term-validation)

### Implementation context

[Run](../../../../glossary.md#term-run) once after executable source and tests stabilize:

```bash
./gradlew :extensions:nn:test \
  --tests io.github.pho001.synaptik.nn.layers.RnnSequenceTest \
  --tests io.github.pho001.synaptik.nn.layers.RnnSequencePackingTest

./gradlew :extensions:nn:test
```

Record test/suite counts and failures, errors, and skips. Do not repeat the full [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) suite after
documentation-only edits.

### Documentation context

[Run](../../../../glossary.md#term-run) after final documentation edits:

```bash
./gradlew :extensions:nn:javadoc

rg -n "RnnSequence|packed recurrent|active batch|recurrent scan|construction-time lengths" \
  extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers \
  docs/api/training-api.md \
  docs/glossary.md \
  docs/planning/extensions/nn

rg -n "extensions\.training|compiler|runtime|prepare|engine|backends" \
  extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/RnnSequence*.java

git diff --name-only
git diff --check
```

Also perform:

- reflection or `javap` verification of the exact two-type public surface and absence of retained
  length/output/index fields;
- import/[dependency](../../../../glossary.md#term-dependency) inspection proving [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) production still imports only Java and [Model](../../../../glossary.md#term-model)/NN types;
- generated-[Javadoc](../../../../glossary.md#term-javadoc) inspection for every public/protected member, record component, `@param`,
  `@return`, `@throws`, [nullability](../../../../glossary.md#term-nullability), [ownership](../../../../glossary.md#term-ownership-meanings), failure, and side-effect contract;
- Markdown heading/anchor/link/fence [validation](../../../../glossary.md#term-validation) for changed documentation;
- exact nine-path [scope](../../../../glossary.md#term-scope) verification;
- master-plan frontier verification: 0015 is Complete, while 0016 and later rows are concise Draft
  entries without task specifications;
- newline-at-[EOF](../../../../glossary.md#term-eof) and whitespace checks.

Repository-wide tests are not required: this task changes one [leaf](../../../../glossary.md#term-leaf) extension and explanatory
documentation without changing [module](../../../../glossary.md#term-module) dependencies, shared build configuration, architecture
rules, Model behavior, or another module. Architecture, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), and integration
tests are not required for the same reason.

## Dependencies

- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0011 complete: direct versus [unary](../../../../glossary.md#term-unary) composition and `Sequential` exclusion are fixed.
- NN 0012 complete: exact vanilla-RNN cell signature, state, Shape, promotion, provenance, and
  replacement contracts are available.<span data-glossary-navigation="true"> ([RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [signature glossary](../../../../glossary.md#term-signature); [Shape glossary](../../../../glossary.md#shape); [provenance glossary](../../../../glossary.md#provenance))</span>
- NN 0013 and 0014 complete: GRU and LSTM prove that a shared sequence/state API is not yet
  justified and define later cell-specific follow-up requirements.<span data-glossary-navigation="true"> ([GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Model SELECT, GATHER, STACK, eager INT64 leaf, static Shape, promotion, and provenance contracts
  complete.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [GATHER glossary](../../../../glossary.md#gather); [eager glossary](../../../../glossary.md#term-eager-evaluation); [INT64 glossary](../../../../glossary.md#data-type--datatype); [leaf glossary](../../../../glossary.md#term-leaf))</span>
- No CPU, compiler, runtime, Prepare, Engine, backend, optimizer, or numerical execution task is
  a prerequisite for static expression construction.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>

## Follow-up tasks

- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0016: add cell-specific static packed `GruSequence` only after 0015 validates the one-hidden-
  state packing and restoration contract.
- NN 0017: add cell-specific static packed `LstmSequence`, carrying and restoring both hidden and
  cell states as sequences leave the active set.
- A future Model task may specify a genuine recurrent scan/control-flow primitive if a concrete
  runtime-dynamic length/mask consumer supplies body, carried-value, dynamic-Shape, compiler, and
  execution requirements. It is not an automatic part of the NN sequence tasks.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [runtime glossary](../../../../glossary.md#term-runtime); [mask glossary](../../../../glossary.md#term-boolean-mask); [Shape glossary](../../../../glossary.md#shape); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- A later consumer may justify dense unpacking, batch-first views, sorting/permutation metadata,
  bidirectionality, stacking, or a shared recurrent abstraction. None is implied here.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [views glossary](../../../../glossary.md#term-view-meanings); [permutation glossary](../../../../glossary.md#permutation); [metadata glossary](../../../../glossary.md#term-metadata))</span>

## Architecture impact

Expected impact: None. This is a cell-specific [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) composition over completed public [Model](../../../../glossary.md#term-model)
contracts and the existing `extensions/nn -> modules/model` [dependency](../../../../glossary.md#term-dependency). No [module](../../../../glossary.md#term-module) boundary,
[dependency direction](../../../../glossary.md#java-platform-and-build-tools), [lifecycle](../../../../glossary.md#lifecycle) [owner](../../../../glossary.md#term-owner), [operation](../../../../glossary.md#operation) semantics, or architecture rule changes.

If implementation requires an architecture-contract, Model operation, [runtime](../../../../glossary.md#term-runtime)-dynamic scan,
dependency, package-boundary, or architecture-test change, stop and report the issue.

## Implementation prompt

Use this prompt in a separate clean agentic task/thread:

```text
You are working in the Synaptik repository. Do not use any GSD skill or workflow, and do not
commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/architecture/current-architecture-plan.md,
docs/planning/planning-guide.md, docs/planning/roadmap.md,
docs/planning/extensions/nn/master-plan.md, and
docs/planning/extensions/nn/tasks/0015-static-packed-rnn-sequence.md in full, plus every directly
required contract named by that task.

Implement task 0015 exactly as specified in its nine-path maximum scope. Do not implement
out-of-scope items. If current public contracts cannot express the specified static composition,
or architecture/scope uncertainty appears, stop and report the exact blocker instead of inventing
architecture.

After executable implementation and the final NN module validation, hand the diff and recorded
test evidence to a distinct clean documentation-focused agent/thread. That pass must follow
docs/developer-guide/documentation-rules.md, independently finalize affected Javadocs,
documentation, glossary, planning evidence, and no-change conclusions in the same overall change,
and must not repeat successful Java tests unless executable behavior changes or a concrete risk is
recorded.

At the end, update the task with local decisions, known limitations, validation evidence,
implementation notes, completion summary, and final status. Do not mark it Complete before the
documentation pass and every exact-scope gate finish.
```

## Local decisions

- Use direct `RnnSequence extends Module` with one exact owned `RnnCell`; no shared recurrent
  abstraction is introduced because the completed cell signatures still differ materially.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>
- Represent packed output as one immutable list entry per time step. This preserves static active
  extents and exact step boundaries without a second batch-size array or dense padded Tensor.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Keep original batch order throughout. Each later hidden gather uses positions relative to the
  prior active set, and final row positions are captured when a row exits rather than recovered by
  sorting or an inverse permutation.<span data-glossary-navigation="true"> ([gather glossary](../../../../glossary.md#gather); [permutation glossary](../../../../glossary.md#permutation))</span>
- Build each INT64 gather index as one ordinary eager dense-contiguous unlabeled Model leaf. The
  sequence retains none of these leaves; ordinary expression provenance remains their only owner.<span data-glossary-navigation="true"> ([INT64 glossary](../../../../glossary.md#data-type--datatype); [eager glossary](../../../../glossary.md#term-eager-evaluation); [contiguous glossary](../../../../glossary.md#term-contiguous); [Model glossary](../../../../glossary.md#term-model); [leaf glossary](../../../../glossary.md#term-leaf); [provenance glossary](../../../../glossary.md#provenance); [owner glossary](../../../../glossary.md#term-owner))</span>
- Permit mixed floating promotion when every final row comes from recurrent output. If a
  zero-length initial-hidden row would have a different exact type from recurrent rows, reject the
  final STACK contract before constructing any Tensor.

## Known limitations

- Lengths must be construction-time Java values and all [Tensor](../../../../glossary.md#tensor) [Shapes](../../../../glossary.md#shape) must be fully static.
- The returned outputs are compact by step and cannot be interpreted as one dense rectangular
  sequence without a later explicit unpacking capability.
- Parameter replacement and a multi-step forward construction are not an atomic snapshot. Each
  cell call observes its current compatible bindings under the existing cell contract.<span data-glossary-navigation="true"> ([Parameter glossary](../../../../glossary.md#term-parameter); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [snapshot glossary](../../../../glossary.md#term-snapshot); [bindings glossary](../../../../glossary.md#term-binding))</span>
- Construction expresses compact logical rows but does not claim compiler, backend, numerical,
  gradient, scheduling, or physical kernel support.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [gradient glossary](../../../../glossary.md#term-gradient); [kernel glossary](../../../../glossary.md#kernel))</span>
- Runtime Tensor lengths or masks remain blocked on a genuine Model recurrent scan/control-flow
  design; cumulative sum and product do not provide that contract.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [masks glossary](../../../../glossary.md#term-boolean-mask); [Model glossary](../../../../glossary.md#term-model))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Clean implementation context `/root/nn_0015_implementation` read the repository instructions,
  architecture and planning contracts, documentation profiles, NN recurrent history and current
  task, Module/state/cell APIs, and relevant Model Tensor/Shape/layout/composition contracts before
  editing.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [Tensor glossary](../../../../glossary.md#tensor); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout))</span>
- The first focused development run exposed one incorrect test assertion about the existing
  GATHER axis-attribute record; production behavior was correct and the test was corrected to the
  exact <span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [GATHER glossary](../../../../glossary.md#gather); [axis glossary](../../../../glossary.md#term-axis))</span>`IndexAxisAttrs(0)` contract.
- The stabilized focused command
  `./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.layers.RnnSequenceTest --tests io.github.pho001.synaptik.nn.layers.RnnSequencePackingTest`
  passes 2 suites and 14 tests with no skips, failures, or errors.
- After executable Java and tests stabilized, the authoritative
  `./gradlew :extensions:nn:test` passes 25 suites and 168 tests with no skips, failures, or
  errors. No executable Java or test changed afterward.
- Preliminary `./gradlew :extensions:nn:javadoc` passed before final documentation review.
  `git diff --check` also passed at that point.
- `javap -public` shows only the exact `RnnSequence(RnnCell)`, `cell()`, and
  `forward(Tensor, Tensor, long[])` surface plus the [canonical](../../../../glossary.md#term-canonical-form) two-component record surface.
  `javap -private` shows the sequence retains only one `RnnCell cell` field and no lengths,
  indices, inputs, outputs, or state. Production import inspection found only Java, Model, and NN
  module types; the existing NN Gradle dependency remains unchanged.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- New-file no-index whitespace checks passed for both production and both focused test files.
  Worktree inspection found exactly seven current NN implementation-owned paths; the Training API
  and glossary are intentionally reserved for the required documentation pass. Concurrent CPU
  source/test/planning work and the global roadmap remain untouched.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- Focused tests cover the exact public surfaces, immutable result snapshot, sole-child ownership,
  recursive state paths, mode propagation and independence, compatible replacement, all-zero,
  zero-time and empty-batch no-ID results, null/type/rank/every-dynamic-axis/feature/batch/length/
  mixed-stack failures before an expression ID, stable <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [snapshot glossary](../../../../glossary.md#term-snapshot); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [rank glossary](../../../../glossary.md#term-rank); [feature glossary](../../../../glossary.md#term-feature))</span>`[5,3,1]` and unsorted `[1,3,2]` packing,
  `[0,2,0]` restoration, copied lengths, exact eager-index metadata and host data, fixed one-step
  ID/provenance order, and absence of cumulative-scan operations.<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [metadata glossary](../../../../glossary.md#term-metadata); [host glossary](../../../../glossary.md#term-host); [provenance glossary](../../../../glossary.md#provenance); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- Repository-wide, architecture, Model, backend-conformance, and integration tests were not run:
  the task changes one leaf extension, adds no dependency or Model/backend behavior, and the task
  assigns final documentation gates to the independent documentation context.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [leaf glossary](../../../../glossary.md#term-leaf))</span>
- Independent clean documentation context `/root/nn_0015_docs` read the final implementation and
  tests, repository instructions, architecture/planning/documentation contracts, recurrent and
  module-state boundaries, relevant Model SELECT/GATHER/STACK/eager-index/cumulative-scan
  contracts, Training and Compile API boundaries, Model capabilities, the Training graph, the
  central glossary, generated Javadocs, and the exact diff. It found no executable/API defect,
  architecture uncertainty, scope blocker, or reason to change executable Java or tests.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [Training graph glossary](../../../../glossary.md#training-graph); [scope glossary](../../../../glossary.md#term-scope))</span>
- The documentation context finalized `RnnSequence`, `RnnSequenceForwardResult`, and layers-
  package Javadocs. It clarified the active-batch definition, defensive Java-array snapshot and
  caller coordination, exact result ownership, no-zero-value inference, allocation failures, and
  runtime recurrent-scan boundary without adding execution, numerical, gradient, or backend
  claims.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference-meanings); [runtime glossary](../../../../glossary.md#term-runtime); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- The Training [API](../../../../glossary.md#term-application-programming-interface-api) now records the current direct-cell [signatures](../../../../glossary.md#term-signature) and the current cell-specific
  `RnnSequence` surface, fully static time-major Shapes, copied explicit lengths, stable active-
  batch compaction, packed step ordering, <span data-glossary-navigation="true"> ([time-major glossary](../../../../glossary.md#term-time-major))</span>`[5,3,1]` example, zero-length/final-hidden restoration,
  and current-versus-planned execution boundary. Static GRU/LSTM sequence containers remain
  planned.<span data-glossary-navigation="true"> ([GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm))</span>
- The existing central `docs/glossary.md` now defines construction-time length, active batch,
  static packed recurrent sequence, and recurrent scan. The task's erroneous draft reference to
  absent `docs/reference/glossary.md` was corrected in the affected-file and validation lists;
  no duplicate glossary was created.
- The documentation pass reused the implementation context's focused two-suite/14-test result and
  authoritative 25-suite/168-test NN result because no executable Java or test changed afterward.
  Javadoc and prose edits do not stale that evidence, so no Java test suite was repeated.
- Final `./gradlew :extensions:nn:javadoc` passed after the [Javadoc](../../../../glossary.md#term-javadoc) edits (`BUILD SUCCESSFUL`; 3
  actionable tasks, 2 executed and 1 up-to-date). Generated `RnnSequence.html`,
  `RnnSequenceForwardResult.html`, and `layers/package-summary.html` contain the exact public
  signatures, inputs, results, failures, ownership, active-row packing, defensive lengths,
  zero-value, concurrency, side-effect, mode, and non-execution boundaries.<span data-glossary-navigation="true"> ([concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- Final `javap -public` and `javap -private` confirmed final direct [`Module`](../../../../glossary.md#term-module), exactly one public
  constructor plus `cell()` and `forward(Tensor, Tensor, long[])`, the ordinary two-component
  record surface, no protected surface or nested type, and only one retained `RnnCell cell` field
  on the sequence. The implementation focused tests provide the independent reflection checks for
  the same exact surfaces and absence of retained sequence fields.
- Production import inspection found only Java, [Model](../../../../glossary.md#term-model), and existing [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) types. The unchanged
  `extensions/nn` build retains only `implementation(project(":modules:model"))`; no Training,
  Compiler, Runtime, Prepare, Engine, concrete-backend, or new Gradle dependency appears.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine))</span>
- The targeted Markdown validator passed the Training API, glossary, NN master plan, and task with
  362 local links, 302 anchors, balanced fences, and final newlines. Exact-scope inspection found
  exactly the nine authorized NN 0015 paths while preserving concurrent CPU and global-roadmap
  work. Master/task status is Complete; NN 0016 and 0017 remain concise Draft rows without task
  specifications.
- `ARCHITECTURE.md`, focused architecture explanations, ADR 0007, and architecture tests require
  no change because this remains cell-specific Model-only NN composition on the existing
  dependency edge. Tensor and Compile APIs, the Training Java API and Training graph, Model
  source/master/capabilities, and related operation contracts require no change because the task
  only composes current metadata and adds no Model semantic, autograd, capture, optimizer, or
  runtime-dynamic control-flow contract.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [dependency edge glossary](../../../../glossary.md#term-dependency-edge); [operation glossary](../../../../glossary.md#operation); [autograd glossary](../../../../glossary.md#autograd); [capture glossary](../../../../glossary.md#term-capture-meanings); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- Compiler, Runtime, Prepare, Engine, CPU and other backends, backend conformance, integration
  tests, Gradle configuration, other recurrent cells/modules, the CPU master/task, and the global
  roadmap require no change because no executable lifecycle, dependency, backend behavior, or
  cross-module capability changed. Repository-wide and those specialized suites remain deferred
  to their owning checkpoints or CI.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [checkpoints glossary](../../../../glossary.md#term-checkpoint); [CI glossary](../../../../glossary.md#term-continuous-integration-ci))</span>
- Final new-file no-index whitespace checks, final-newline checks, exact-scope/status/frontier
  checks, and whole-worktree `git diff --check` passed. Concurrent CPU changes were preserved and
  were neither reviewed nor modified by this context.

## Implementation notes

- Added `RnnSequence` and `RnnSequenceForwardResult` under the existing public `nn.layers`
  package, with no new package, dependency, Model operation, shared recurrent interface, mask
  overload, retained sequence state, or unary adapter.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Model glossary](../../../../glossary.md#term-model); [operation glossary](../../../../glossary.md#operation); [mask glossary](../../../../glossary.md#term-boolean-mask); [unary glossary](../../../../glossary.md#term-unary))</span>
- Forward clones lengths before descriptor planning, snapshots the three current cell bindings for
  preflight, validates all caller-controlled contracts without Tensor creation, and then emits the
  specified SELECT/index/GATHER/cell chain. Later steps gather survivor positions from the prior
  compact output.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [planning glossary](../../../../glossary.md#term-planning); [bindings glossary](../../../../glossary.md#term-binding); [preflight glossary](../../../../glossary.md#term-preflight); [Tensor glossary](../../../../glossary.md#tensor); [GATHER glossary](../../../../glossary.md#gather))</span>
- Exit positions are captured in a temporary primitive array as each original row leaves the
  active set. Final rows select either that exit output or initial hidden for length zero and are
  stacked in original order.
- Draft Javadocs cover ownership, Shapes, explicit lengths, no-zero inference, side effects,
  concurrency, mode, result identity, and non-execution boundaries. The package draft distinguishes
  sequence packing from gate-parameter packing, unary <span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [Shapes glossary](../../../../glossary.md#shape); [inference glossary](../../../../glossary.md#term-inference-meanings); [concurrency glossary](../../../../glossary.md#term-concurrency); [identity glossary](../../../../glossary.md#term-identity-meanings); [parameter glossary](../../../../glossary.md#term-parameter))</span>`Sequential`, and a [runtime](../../../../glossary.md#term-runtime) recurrent scan.
- No required change was found for architecture contracts, Model, Training Java source, Gradle,
  global roadmap, CPU/backend work, or tests outside <span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend))</span>`extensions/nn`.

## Completion summary

- Completed changes: stable static packed vanilla-RNN composition, immutable result carrier,
  focused tests, finalized Java/package documentation, Training API reference, glossary terms,
  and synchronized planning records.<span data-glossary-navigation="true"> ([RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [immutable glossary](../../../../glossary.md#term-immutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: the exact nine authorized paths—two production files, two focused test
  files, layers package Javadoc, Training API, central glossary, NN task, and NN master plan.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Tests and validation: focused 14/14 and full NN 168/168 passed in the implementation context;
  final Javadoc, generated-page, public/private-surface, reflection, import/dependency, Markdown,
  exact-scope/status/frontier, newline, whitespace, and diff gates passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [dependency glossary](../../../../glossary.md#term-dependency); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: independent clean context `/root/nn_0015_docs` completed the final
  review and validation without changing executable Java or tests.
- Documentation impact: Training API now explains current static RNN sequence composition and its
  future runtime-scan boundary; architecture and unrelated API documentation remain unchanged for
  the reasons recorded above.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime))</span>
- Javadoc review: final public, record-component, constructor, accessor, forward, package,
  ownership, failure, side-effect, concurrency, and non-execution contracts passed generation and
  rendered-page inspection.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- Glossary impact: the existing central glossary defines static packed recurrent sequence, active
  batch, construction-time length, and recurrent scan without creating a duplicate file.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- Unresolved issues: None in executable implementation.
- Follow-up required: None for NN 0015. NN 0016 and NN 0017 remain optional ordered Draft
  cell-specific follow-ups.

Status: Complete

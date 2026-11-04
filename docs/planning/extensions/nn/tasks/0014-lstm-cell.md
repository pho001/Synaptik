# Task 0014: [LSTM](../../../../glossary.md#term-long-short-term-memory-lstm) Cell

> **Editorial reading context — not a status update.** Long short-term memory threads two
> different values: hidden output and cell memory. The application passes both previous states
> and must preserve both returned expressions when continuing recurrence. The NN result carrier
> names those references; owning that carrier does not make the cell state a persistent Buffer
> or allocate runtime history. Packed input/forget/candidate/output weights remain cell-owned
> parameters. Today's [recurrent state signatures](../../../../reference/nn/recurrent-layers.md#current-nn-recurrent-composition-contract)
> explain why LSTM cannot be reduced to a unary Tensor chain, and [state export](../../../../reference/nn/state.md#current-nn-state-dictionary-contract)
> distinguishes learned model state from caller-threaded recurrent state.

## Status

Complete

## Goal

Add one final public [long short-term memory](../../../../glossary.md#term-long-short-term-memory-lstm) ([LSTM](../../../../glossary.md#neural-network-families)) cell with explicit caller-threaded hidden and
cell state. The cell owns two gate-major packed projection [matrices](../../../../glossary.md#term-matrix-meanings) and one optional packed
input-side [bias](../../../../glossary.md#term-bias), constructs one fixed LSTM step entirely from current [Model](../../../../glossary.md#term-model) [Tensor](../../../../glossary.md#tensor) expressions,
and returns both next-state Tensors in one [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)-owned result [carrier](../../../../glossary.md#term-storage-carrier). It retains no [recurrent state](../../../../glossary.md#backpropagation-through-time-and-recurrent-state).

[Mental model](../../../../glossary.md#term-mental-model):

```text
caller input x + caller hidden h + caller cell c + current packed parameters
  -> input and hidden packed linear projections
  -> input, forget, candidate, and output slices in fixed gate order
  -> fixed gate activations
  -> next cell c' = forget * c + inputGate * candidate
  -> next hidden h' = output * tanh(c')
  -> LstmCellForwardResult(nextHidden=h', nextCell=c')
  -> caller explicitly threads both returned references into a later call
```

Unlike `RnnCell` and `GruCell`, hidden output and cell state are distinct Tensor expressions. The
public result therefore names and retains both exact references rather than erasing one state or
returning an untyped collection.

## [Scope](../../../../glossary.md#term-scope)

- Add final public `io.github.pho001.synaptik.nn.layers.LstmCell` extending
  `io.github.pho001.synaptik.nn.module.Module` directly.
- Add final public record `io.github.pho001.synaptik.nn.layers.LstmCellForwardResult` with exactly
  the components `Tensor nextHidden` and `Tensor nextCell`, in that order.
- Add exactly the constructors, parameter accessors, and three-input forward method in the public
  API table below. Add no overload, builder, options object, gate enum, size getter, functional
  facade, state tuple abstraction, or configurable equation policy.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [facade glossary](../../../../glossary.md#term-facade))</span>
- Declare positive fully static [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [parameters](../../../../glossary.md#term-parameter) under exact local names `inputWeight` then
  `hiddenWeight`. Their [Shapes](../../../../glossary.md#shape) are `[4 * hiddenSize, inputSize]` and
  `[4 * hiddenSize, hiddenSize]`.
- Pack both matrices gate-major on axis zero in exact input, forget, candidate, output order.
  After linear projection, the same order occupies the final result axis.<span data-glossary-navigation="true"> ([matrices glossary](../../../../glossary.md#term-matrix-meanings); [axis glossary](../../../../glossary.md#term-axis); [linear projection glossary](../../../../glossary.md#linear-projection))</span>
- Optionally declare one positive fully static [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) parameter under exact local name [`bias`](../../../../glossary.md#term-bias),
  with Shape `[4 * hiddenSize]`. It is added only by the packed input projection. There is no
  recurrent bias and no separate per-gate bias parameter.<span data-glossary-navigation="true"> ([bias parameter glossary](../../../../glossary.md#bias-and-activation))</span>
- Require all parameters to share one exact floating [data type](../../../../glossary.md#data-type--datatype) and have `requiresGrad == true`.
  Retain caller-supplied parameter Tensor references exactly.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- Provide supplied-state construction both without and with bias. Null never means absent bias;
  callers select absence through the two-Tensor constructor.
- Provide one initialized constructor with explicit positive `inputSize`, positive `hiddenSize`,
  bias presence, floating [`DataType`](../../../../glossary.md#data-type--datatype), and caller-owned `RandomGenerator`.
- Compute `packedHiddenSize = 4 * hiddenSize` with checked arithmetic before constructing Shapes.
  Initialize `inputWeight` then `hiddenWeight` through exact
  `ParameterInitializers.glorotUniform` calls using the same explicit source. When requested,
  initialize the complete packed bias afterward through exact `ParameterInitializers.zeros`;
  bias consumes no draw. The source is never retained.
- Use an all-zero initialized packed bias, including the forget interval. Current initializers do
  not provide one direct eager packed leaf with only the forget interval set to one. Do not hide
  host mutation, compose a parameter from slices/concatenation, or invent a new initializer in
  this task. A selected forget-bias policy requires a separate initializer/API decision.<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [leaf glossary](../../../../glossary.md#term-leaf); [host glossary](../../../../glossary.md#term-host))</span>
- Add exactly `LstmCellForwardResult forward(Tensor input, Tensor hidden, Tensor cell)`. All three
  semantic inputs are explicit on every call; the module never supplies, caches, updates,
  registers, or discovers hidden or cell state.
- Fix the gate order, input-side-only optional-bias association, activation placement, and
  equations specified below. Do not claim checkpoint or formula compatibility with a framework
  whose packing, bias, or equation convention differs.<span data-glossary-navigation="true"> ([activation glossary](../../../../glossary.md#term-activation); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Use independent `sliceAxis(-1, ...)` expressions for the eight projection gate slices. Current
  Model has no shared multi-output split operation; independent slices keep each exact packed
  source and interval visible without adding a Model API or per-gate parameters.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [operation glossary](../../../../glossary.md#operation))</span>
- Support input, hidden, and cell rank one or higher. Their final Dimensions respectively
  represent <span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [Dimensions glossary](../../../../glossary.md#dimension))</span>`inputSize`, `hiddenSize`, and `hiddenSize`.
- Treat all leading Dimensions as ordinary right-broadcastable batch metadata. No leading axis is
  named, interpreted, or traversed as time.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Complete all caller-controlled forward validation before creating the first projection
  expression. Then construct exactly the formula and producer order specified below.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [producer glossary](../../../../glossary.md#term-producer))</span>
- Keep the cell mode-insensitive. Forward accepts no `ForwardContext`, does not inspect `mode()`,
  and does not alter module mode or state.
- Preserve stable parameter wrappers, schema-compatible replacement snapshots, recursive
  discovery, and state-dictionary paths through current <span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>[`Module`](../../../../glossary.md#term-module) and `Parameter` contracts.
- Add focused exact-surface/result-carrier, supplied-state, initialization, validation-order,
  shape/type, gate-slicing, provenance, replacement, mode, discovery, and exclusion tests.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [provenance glossary](../../../../glossary.md#provenance))</span>
- Add complete type, record-component, constructor, member, and package Javadocs. After executable
  work and final NN testing, use a separate clean documentation-focused context to finalize
  Javadocs, glossary impact, planning evidence, generated Javadoc, and no-change conclusions.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [planning glossary](../../../../glossary.md#term-planning); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Retaining current or next hidden/cell [Tensors](../../../../glossary.md#tensor) in fields, [`Buffer`](../../../../glossary.md#term-buffer) values, children, state
  dictionaries, thread-locals, statics, runtime objects, sessions, or another hidden lifecycle.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Default or zero hidden/cell state, state initializers, reset/detach APIs, stateful forward
  overloads, or a carrier that omits either state transition.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- `UnaryTensorModule`, participation in `Sequential`, an adapter into `Sequential`, or changes to
  `Sequential`, `UnaryTensorModule`, [`Module`](../../../../glossary.md#term-module), [`Parameter`](../../../../glossary.md#term-parameter), or `Buffer`.
- Separate public gate parameters, eight per-gate matrices, separate input/recurrent biases,
  per-gate bias accessors, or parameter-packing configuration.<span data-glossary-navigation="true"> ([matrices glossary](../../../../glossary.md#term-matrix-meanings); [biases glossary](../../../../glossary.md#term-bias))</span>
- A non-zero forget-bias initializer, hidden eager storage mutation, parameter construction from
  slices or concatenation, or a new initializer convenience.<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation))</span>
- Peephole connections, output projection, coupled input-forget gates, layer normalization, cell
  clipping, recurrent dropout, residuals, attention, convolutional recurrence, or configurable
  gate activations.<span data-glossary-navigation="true"> ([layer normalization glossary](../../../../glossary.md#layer-normalization); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [activations glossary](../../../../glossary.md#term-activation))</span>
- Alternate gate order, bias association, activation placement, equation policy, framework-
  compatibility mode, checkpoint migration, or conversion helpers for another LSTM layout.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm); [layout glossary](../../../../glossary.md#layout))</span>
- A recurrent sequence container, loop, step counter, time-axis convention, masks, sequence
  lengths, packed sequence, variable-length handling, active-batch compaction, bidirectionality,
  stacking, static unrolling, or time traversal.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [masks glossary](../../../../glossary.md#term-boolean-mask); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [unrolling glossary](../../../../glossary.md#term-unrolling))</span>
- A general recurrent Model scan, loop operation, subgraph/body representation, carried-value
  tuple, scan result, or Tensor API. Current <span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [scan glossary](../../../../glossary.md#term-cumulative-scan); [operation glossary](../../../../glossary.md#operation); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>`CUM_SUM` and `CUM_PROD` are associative cumulative
  operations and are not recurrent scan primitives.
- Backpropagation through time, gradient detachment, gradient rules, optimizer/session,
  parameter groups, checkpoint transport, serialization, compiler capture, scheduling,
  Runtime/Prepare/Engine behavior, backend lowering, kernels, numerical execution, or end-to-end
  support claims.<span data-glossary-navigation="true"> ([Backpropagation through time glossary](../../../../glossary.md#term-backpropagation-through-time-bptt); [gradient glossary](../../../../glossary.md#term-gradient); [optimizer glossary](../../../../glossary.md#term-optimizer); [serialization glossary](../../../../glossary.md#term-serialization); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel))</span>
- A Model, Training, Gradle, dependency, architecture-contract, ADR, architecture-test, global-
  roadmap, CPU, or explanatory-architecture source change during implementation.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- A detailed task specification or implementation for [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0015.

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
- [Completed NN task 0004: Initializers](0004-explicit-eager-parameter-initializers.md)
- [Completed NN task 0004A: Replacement hardening](0004a-parameter-update-and-traversal-hardening.md)
- [Completed NN task 0011: Unary composition](0011-unary-tensor-module-composition-and-sequential.md)<span data-glossary-navigation="true"> ([Unary glossary](../../../../glossary.md#term-unary))</span>
- [Completed NN task 0012: Vanilla RNN cell](0012-vanilla-rnn-cell.md)<span data-glossary-navigation="true"> ([RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn))</span>
- [Completed NN task 0013: GRU cell](0013-gru-cell.md)<span data-glossary-navigation="true"> ([GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru))</span>
- [Completed Model task 0017H: Slice expressions](../../../modules/model/tasks/0017h-slice-tensor-expressions.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Slice glossary](../../../../glossary.md#slice))</span>
- [Completed Model task 0019D: Linear convenience](../../../modules/model/tasks/0019d-linear-convenience.md)<span data-glossary-navigation="true"> ([Linear glossary](../../../../glossary.md#term-linear-meanings))</span>
- [Completed Model task 0023E: Cumulative scan normalization](../../../modules/model/tasks/0023e-cumulative-scan-normalization-and-product.md)<span data-glossary-navigation="true"> ([Cumulative scan glossary](../../../../glossary.md#cumulative-scan); [normalization glossary](../../../../glossary.md#term-normalization))</span>

## Architecture constraints

- `extensions/nn` continues to depend only on `modules/model`. Production must not import
  Training, Compiler, Runtime, Prepare, Engine, CPU, Metal, CUDA, or another backend.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal); [CUDA glossary](../../../../glossary.md#term-cuda); [backend glossary](../../../../glossary.md#backend))</span>
- NN owns packed parameter bindings, the typed cell composition, and its explicit two-Tensor
  result carrier. Model remains the sole owner of LINEAR decomposition, SLICE, ADD, MUL, SIGMOID,
  TANH, type promotion, Shape algebra, descriptors, provenance, and Tensor identity.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [parameter glossary](../../../../glossary.md#term-parameter); [bindings glossary](../../../../glossary.md#term-binding); [Tensor glossary](../../../../glossary.md#tensor); [carrier glossary](../../../../glossary.md#term-storage-carrier); [Model glossary](../../../../glossary.md#term-model); [owner glossary](../../../../glossary.md#term-owner); [LINEAR glossary](../../../../glossary.md#term-linear-meanings); [decomposition glossary](../../../../glossary.md#term-decomposition); [SLICE glossary](../../../../glossary.md#slice); [SIGMOID glossary](../../../../glossary.md#term-sigmoid); [TANH glossary](../../../../glossary.md#term-tanh); [Shape glossary](../../../../glossary.md#shape); [descriptors glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Tensor identity, descriptors, and provenance remain immutable. Parameter replacement changes
  only the current exact Tensor returned by one stable wrapper; existing expressions remain
  unchanged.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- Recurrent hidden and cell state are caller-threaded Tensor values, never module-owned persistent
  state. Neither becomes a [`Buffer`](../../../../glossary.md#term-buffer) merely because the caller may pass it between calls.
- [`Module`](../../../../glossary.md#term-module) retains no universal forward method. `LstmCell` is a direct subclass with a truthful
  three-input signature and must not extend <span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>`UnaryTensorModule`.
- `Sequential` remains a container only for `UnaryTensorModule`; no adapter may capture, erase,
  or synthesize either recurrent-state input.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- Mode is NN composition metadata. This mode-insensitive cell neither consumes a context nor
  reads inherited mode during forward.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>
- Packed slices are ordinary independent one-output Model expressions. The task must not invent a
  shared split producer, Model tuple, hidden gate cache, direct <span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [cache glossary](../../../../glossary.md#term-cache))</span>[`Operation`](../../../../glossary.md#operation) construction, or
  special LSTM semantic kind.<span data-glossary-navigation="true"> ([LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm))</span>
- `LstmCellForwardResult` is an NN composition value, not a Model multi-output producer, Tensor,
  graph value, module state, checkpoint payload, runtime result, or execution handle.<span data-glossary-navigation="true"> ([graph value glossary](../../../../glossary.md#graph-value); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>
- Construction and forward create eager parameter leaves or storage-free expression metadata
  only. They do not prove gradient implementation, graph capture, compiled execution, backend
  support, numerical values, or storage residency.<span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [gradient glossary](../../../../glossary.md#term-gradient); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [residency glossary](../../../../glossary.md#residency))</span>
- If implementation needs a new Model operation/helper/public method, initializer, recurrent
  container, state Buffer, context/RNG input, dependency, architecture rule, or ninth task path,
  stop and report the exact blocker instead of widening this task.<span data-glossary-navigation="true"> ([RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [dependency glossary](../../../../glossary.md#term-dependency); [widening glossary](../../../../glossary.md#term-widening))</span>

## Public [API](../../../../glossary.md#term-application-programming-interface-api)

`LstmCell` declares exactly:

```java
public LstmCell(Tensor inputWeight, Tensor hiddenWeight)

public LstmCell(Tensor inputWeight, Tensor hiddenWeight, Tensor bias)

public LstmCell(
        long inputSize,
        long hiddenSize,
        boolean bias,
        DataType dataType,
        RandomGenerator randomGenerator)

public Parameter inputWeight()
public Parameter hiddenWeight()
public Optional<Parameter> bias()
public LstmCellForwardResult forward(Tensor input, Tensor hidden, Tensor cell)
```

| Member | Exact contract |
|---|---|
| two-[Tensor](../../../../glossary.md#tensor) constructor | Retains exact positive static packed input/hidden projection [weights](../../../../glossary.md#term-weight) and declares no [bias](../../../../glossary.md#term-bias). |
| three-Tensor constructor | Retains the same exact weights plus one exact packed input-side bias; null bias is invalid. |
| initialized constructor | Uses explicit sizes/type/source, creates packed input weight then packed hidden weight through [Glorot](../../../../glossary.md#term-glorot-uniform) uniform and optional all-zero bias afterward. |
| `inputWeight()` | Returns the exact stable wrapper declared under `inputWeight`. |
| `hiddenWeight()` | Returns the exact stable wrapper declared under `hiddenWeight`. |
| `bias()` | Returns an empty Optional or the exact stable wrapper declared under `bias`. |
| `forward(input, hidden, cell)` | Validates the complete three-input request, snapshots current [parameters](../../../../glossary.md#term-parameter), constructs the fixed [LSTM](../../../../glossary.md#term-long-short-term-memory-lstm) formula once, and returns exact next-hidden/next-cell references in one fresh [carrier](../../../../glossary.md#term-storage-carrier). |

The class declares no other public or protected constructor, method, field, nested type,
interface, or overload. Inherited final [`Module`](../../../../glossary.md#term-module) APIs remain available normally.

`LstmCellForwardResult` declares exactly:

```java
public record LstmCellForwardResult(Tensor nextHidden, Tensor nextCell)
```

Its [canonical](../../../../glossary.md#term-canonical-form) constructor rejects null `nextHidden` then null `nextCell`, using the component name
as the `NullPointerException` message. A success retains both exact references without copy,
mutation, [descriptor](../../../../glossary.md#term-descriptor) inspection, relationship [validation](../../../../glossary.md#term-validation), expression creation, or Tensor-ID
allocation. The record uses ordinary record value equality and exposes no other declared public
method or nested type beyond generated component accessors, `equals`, [`hashCode`](../../../../glossary.md#hash-codes-and-content-digests), and `toString`.

## State [schema](../../../../glossary.md#term-schema) and [ownership](../../../../glossary.md#term-ownership-meanings)

| Kind | Local name | [Shape](../../../../glossary.md#shape) orientation | Type | `requiresGrad` | Initialized policy |
|---|---|---|---|---|---|
| [Parameter](../../../../glossary.md#term-parameter) | `inputWeight` | `[4 * hiddenSize, inputSize]` | exact configured floating type | `true` | [Glorot](../../../../glossary.md#term-glorot-uniform) uniform from explicit source |
| Parameter | `hiddenWeight` | `[4 * hiddenSize, hiddenSize]` | same exact type | `true` | Glorot uniform from the same explicit source |
| Parameter, optional | [`bias`](../../../../glossary.md#term-bias) | `[4 * hiddenSize]` | same exact type | `true` | exact typed zero across all four gates |

[Axis](../../../../glossary.md#term-axis) zero of each [matrix](../../../../glossary.md#term-matrix-meanings) and axis zero of bias use these [contiguous](../../../../glossary.md#term-contiguous) intervals:

| Interval | Gate |
|---|---|
| `[0, hiddenSize)` | input |
| `[hiddenSize, 2 * hiddenSize)` | forget |
| `[2 * hiddenSize, 3 * hiddenSize)` | candidate |
| `[3 * hiddenSize, 4 * hiddenSize)` | output |

This fixed gate order is part of the parameter and state-dictionary [checkpoint](../../../../glossary.md#term-checkpoint) schema. No direct
compatibility is promised with frameworks using a different gate order, separate bias [vectors](../../../../glossary.md#term-vector),
recurrent bias, or a non-zero forget-bias default.

For supplied construction, `inputWeight` axis zero must be positive and divisible by four.
`hiddenSize` is exactly that static [extent](../../../../glossary.md#term-extent) divided by four. `hiddenWeight` axis zero and optional
bias axis zero must structurally equal the complete packed extent; `hiddenWeight` axis one must
structurally equal the derived `hiddenSize`. `inputSize` is `inputWeight` axis one. Both logical
sizes are positive.

Direct discovery and state-dictionary order are `inputWeight`, `hiddenWeight`, then optional
`bias`. Under a future parent child name `cell`, paths become `cell.inputWeight`,
`cell.hiddenWeight`, and optional `cell.bias`. Forward input, current/next hidden, current/next
cell, gates, and candidate never appear in parameter, [buffer](../../../../glossary.md#term-buffer), child, or state-dictionary
discovery.

## Supplied construction [validation](../../../../glossary.md#term-validation) and side effects

Both supplied constructors validate complete state before declaring any [parameter](../../../../glossary.md#term-parameter).

For the two-[Tensor](../../../../glossary.md#tensor) constructor:

1. reject null `inputWeight`, then null `hiddenWeight`;
2. validate input [weight](../../../../glossary.md#term-weight) floating type, `requiresGrad == true`, rank two, fully static Shape,
   positive packed axis zero, divisibility of packed axis zero by four, then positive input-size
   axis one;
3. validate hidden weight floating type, `requiresGrad == true`, rank two, fully static Shape,
   positive axis zero, then positive axis one;
4. require hidden weight exact [data type](../../../../glossary.md#data-type--datatype) to equal input weight data type;
5. require hidden weight [axis](../../../../glossary.md#term-axis) zero to equal the complete input-weight packed [extent](../../../../glossary.md#term-extent) structurally;
6. require hidden weight axis one to equal the derived hidden-size static [Dimension](../../../../glossary.md#dimension); and
7. declare `inputWeight` then `hiddenWeight` and retain no [bias](../../../../glossary.md#term-bias).

For the three-Tensor constructor, reject null `bias` after the two weight null checks and before
[schema](../../../../glossary.md#term-schema) validation. Apply steps 2–6, then validate bias floating type,
`requiresGrad == true`, [rank](../../../../glossary.md#term-rank) one, fully static [Shape](../../../../glossary.md#shape), exact common data type, and exact complete
packed extent in that order. Declare `inputWeight`, `hiddenWeight`, then `bias` only after all
checks pass.

Validation creates no Tensor, [producer](../../../../glossary.md#term-producer), storage, random draw, or Tensor [identity](../../../../glossary.md#term-identity-meanings) and never mutates
the supplied values. A failure returns no cell and leaves every supplied Tensor unchanged.

## Initialized construction and side effects

The initialized constructor performs exactly:

1. reject null `dataType`, then null `randomGenerator`;
2. require `inputSize > 0`;
3. require `hiddenSize > 0`;
4. require the [data type](../../../../glossary.md#data-type--datatype) to be floating;
5. compute `packedHiddenSize = Math.multiplyExact(hiddenSize, 4L)`;
6. construct [Shapes](../../../../glossary.md#shape) `[packedHiddenSize, inputSize]`, `[packedHiddenSize, hiddenSize]`, and, only
   when requested, `[packedHiddenSize]`;
7. obtain each requested Shape's checked known element count in state order and reject a count
   above <span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element))</span>`Integer.MAX_VALUE` before the first draw or [Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings) allocation;
8. create input [weight](../../../../glossary.md#term-weight) through exactly
   `ParameterInitializers.glorotUniform(inputWeightShape, dataType, randomGenerator)`;
9. create hidden weight through exactly
   `ParameterInitializers.glorotUniform(hiddenWeightShape, dataType, randomGenerator)` using the
   same now-advanced source;
10. when requested, create the complete packed [bias](../../../../glossary.md#term-bias) through exactly
    `ParameterInitializers.zeros(biasShape, dataType)` with no source call; and
11. after all requested leaves exist, declare [parameters](../../../../glossary.md#term-parameter) in exact state-table order.

Caller-controlled null, size, type, multiplication-[overflow](../../../../glossary.md#term-overflow), checked-count, and Java-array-limit
failures precede every draw and Tensor ID. Weight bounds are computed independently from each
packed [matrix](../../../../glossary.md#term-matrix-meanings)'s actual `[fanOut, fanIn]` Shape. Bias consumes no draw. No bias [slice](../../../../glossary.md#slice) is replaced,
mutated, or materialized separately, so the forget gate begins with the same exact zero-bias
policy as the other three gates.

A source failure keeps completed source calls and creates no Tensor for the failing weight. If it
occurs during hidden-weight creation, the already created input weight and its ID are not rolled
back, but no cell is returned. Later allocation or identifier failures similarly preserve
completed effects. The caller owns the source; the cell never retains, resets, closes,
synchronizes, splits, [seeds](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings), or serializes it.

## Forward [validation](../../../../glossary.md#term-validation) and side-effect order

`forward(input, hidden, cell)` performs exactly:

1. reject null `input`, then null `hidden`, then null `cell`;
2. read current `inputWeight`, `hiddenWeight`, and optional [`bias`](../../../../glossary.md#term-bias) bindings exactly once in
   parameter declaration order;<span data-glossary-navigation="true"> ([bindings glossary](../../../../glossary.md#term-binding))</span>
3. revalidate the complete current packed [parameter](../../../../glossary.md#term-parameter) [schema](../../../../glossary.md#term-schema) and derive exact positive
   `hiddenSize`, `2 * hiddenSize`, `3 * hiddenSize`, and `4 * hiddenSize` bounds with checked
   arithmetic;
4. prevalidate the packed input affine projection under Model linear order: promote input and
   input-weight numeric types, require input rank at least one, reject a proven unequal static
   input-feature contraction, and, when bias is present, promote product and bias types;<span data-glossary-navigation="true"> ([weight glossary](../../../../glossary.md#term-weight))</span>
5. prevalidate the packed hidden projection: promote hidden and hidden-weight numeric types,
   require hidden rank at least one, and reject a proven unequal static hidden-feature
   contraction;
6. require cell rank at least one and reject a proven unequal static final cell-feature
   Dimension against `hiddenSize`; unresolved final equality remains deferred exactly as for the
   linear contractions;
7. derive both packed projection Shapes without creating a Tensor, then derive the eight gate-
   slice Shapes by replacing the final packed extent with exact static <span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent))</span>`hiddenSize` while
   preserving every leading Dimension reference;<span data-glossary-navigation="true"> ([leading Dimension glossary](../../../../glossary.md#term-leading-dimension))</span>
8. in input, forget, candidate, output order, prevalidate each input-slice plus hidden-slice
   promotion and Shape broadcast, and require the resulting preactivation type to be floating;
9. prevalidate forget-gate times current-cell promotion and broadcast;
10. prevalidate input-gate times candidate promotion and broadcast;
11. prevalidate addition of those two products to derive the exact next-cell type and Shape, then
    require next-cell TANH eligibility;<span data-glossary-navigation="true"> ([TANH glossary](../../../../glossary.md#term-tanh))</span>
12. prevalidate output-gate times activated-next-cell promotion and broadcast to derive the exact
    next-hidden type and Shape; and
13. only after all preceding checks succeed, construct the exact formula below.

All parameters are floating, so current same-category promotion requires input, hidden, and cell
to be floating while allowing [BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) [widening](../../../../glossary.md#term-widening). A static final [feature](../../../../glossary.md#term-feature) [Dimension](../../../../glossary.md#dimension)
must equal its configured size. An unresolved input or hidden [contraction Dimension](../../../../glossary.md#contraction-dimension) may remain
deferred under current [Model](../../../../glossary.md#term-model) [linear](../../../../glossary.md#term-linear-meanings) rules. Every subsequent gate or state broadcast must still be
locally provable, so an unresolved cell-feature Dimension paired with the static gate width is
rejected before expression creation.

Leading-prefix [broadcasting](../../../../glossary.md#broadcasting) is conservative. Equal Dimensions and static [singleton](../../../../glossary.md#term-singleton) expansion
succeed; incompatible static sizes and locally unprovable symbolic combinations fail before
expression creation. The three input ranks need not equal. Examples:

| Input [Shape](../../../../glossary.md#shape) | Hidden Shape | Cell Shape | Next-hidden / next-cell Shape | Meaning |
|---|---|---|---|---|
| `[inputSize]` | `[hiddenSize]` | `[hiddenSize]` | `[hiddenSize]` | one unbatched cell application |
| `[batch, inputSize]` | `[batch, hiddenSize]` | `[batch, hiddenSize]` | `[batch, hiddenSize]` | matching [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) prefix |
| `[batch, inputSize]` | `[hiddenSize]` | `[hiddenSize]` | `[batch, hiddenSize]` | both states broadcast across batch |
| `[outer, 1, inputSize]` | `[batch, hiddenSize]` | `[batch, hiddenSize]` | `[outer, batch, hiddenSize]` | ordinary right-aligned leading broadcast |

These examples describe [metadata](../../../../glossary.md#term-metadata) composition only. `outer` and `batch` are leading [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix),
not time traversal or a sequence loop.

Every local null/schema/type/[rank](../../../../glossary.md#term-rank)/static-[contraction](../../../../glossary.md#contraction-dimension)/[slice](../../../../glossary.md#slice)/broadcast failure consumes no [Tensor](../../../../glossary.md#tensor) ID
and creates no expression prefix. Parameter reads, result-Shape/type [preflight](../../../../glossary.md#term-preflight), and record
validation are not Tensor-expression side effects. Identifier exhaustion during valid
construction may leave an expression prefix; no [rollback](../../../../glossary.md#term-rollback) is attempted.

## Formula, delegation, and [provenance](../../../../glossary.md#provenance)

Let `x` be input, `h` be hidden, `c` be cell, `W` be `inputWeight`, `U` be `hiddenWeight`, and
`b` be the optional input-side packed [bias](../../../../glossary.md#term-bias). Let `H` be `hiddenSize`. Packed projections are:

```text
P_x = x @ transpose(W) + b       when bias is present
P_x = x @ transpose(W)           otherwise
P_h = h @ transpose(U)
```

[Slice](../../../../glossary.md#slice) the final [axes](../../../../glossary.md#term-axis) independently in fixed order:

```text
x_i = P_x[..., 0:H]
x_f = P_x[..., H:2H]
x_g = P_x[..., 2H:3H]
x_o = P_x[..., 3H:4H]
h_i = P_h[..., 0:H]
h_f = P_h[..., H:2H]
h_g = P_h[..., 2H:3H]
h_o = P_h[..., 3H:4H]
```

The fixed [LSTM](../../../../glossary.md#term-long-short-term-memory-lstm) equations are:

```text
i = sigmoid(x_i + h_i)
f = sigmoid(x_f + h_f)
g = tanh(x_g + h_g)
o = sigmoid(x_o + h_o)
c' = f * c + i * g
h' = o * tanh(c')
```

[SIGMOID](../../../../glossary.md#term-sigmoid) applies only to input, forget, and output gates. [TANH](../../../../glossary.md#term-tanh) applies to the candidate
preactivation and again to the complete next cell before output gating. There are no peepholes,
projection, clipping, coupled gates, or recurrent bias hidden in these equations. These equations
and the input/forget/candidate/output packing define this [API](../../../../glossary.md#term-application-programming-interface-api); no equivalence to a differently
packed, biased, or initialized framework LSTM is promised.

After prevalidation, implementation delegates exactly in this association and order:

```java
Tensor inputProjection = currentBias.isPresent()
        ? input.linear(currentInputWeight, currentBias.orElseThrow())
        : input.linear(currentInputWeight);
Tensor hiddenProjection = hidden.linear(currentHiddenWeight);

Tensor inputGateProjection = inputProjection.sliceAxis(-1, 0L, hiddenSize);
Tensor forgetGateProjection = inputProjection.sliceAxis(-1, hiddenSize, twiceHiddenSize);
Tensor inputCandidate = inputProjection.sliceAxis(-1, twiceHiddenSize, thriceHiddenSize);
Tensor outputGateProjection = inputProjection.sliceAxis(-1, thriceHiddenSize, packedHiddenSize);
Tensor hiddenInputGate = hiddenProjection.sliceAxis(-1, 0L, hiddenSize);
Tensor hiddenForgetGate = hiddenProjection.sliceAxis(-1, hiddenSize, twiceHiddenSize);
Tensor hiddenCandidate = hiddenProjection.sliceAxis(-1, twiceHiddenSize, thriceHiddenSize);
Tensor hiddenOutputGate = hiddenProjection.sliceAxis(-1, thriceHiddenSize, packedHiddenSize);

Tensor inputGate = inputGateProjection.add(hiddenInputGate).sigmoid();
Tensor forgetGate = forgetGateProjection.add(hiddenForgetGate).sigmoid();
Tensor candidate = inputCandidate.add(hiddenCandidate).tanh();
Tensor outputGate = outputGateProjection.add(hiddenOutputGate).sigmoid();
Tensor nextCell = forgetGate.mul(cell).add(inputGate.mul(candidate));
Tensor nextHidden = outputGate.mul(nextCell.tanh());
return new LstmCellForwardResult(nextHidden, nextCell);
```

The no-bias successful chain creates exactly twenty-five fresh [Tensors](../../../../glossary.md#tensor)/IDs in order: input-[weight](../../../../glossary.md#term-weight)
PERMUTE, input [MATMUL](../../../../glossary.md#matrix-multiplication--matmul), hidden-weight PERMUTE, hidden MATMUL, eight SLICE occurrences in the order
shown, input ADD, input SIGMOID, forget ADD, forget SIGMOID, candidate ADD, candidate TANH, output
ADD, output SIGMOID, forget-cell MUL, input-candidate MUL, next-cell ADD, next-cell TANH, and
next-hidden MUL. The biased chain inserts one bias ADD immediately after input MATMUL for
twenty-six total. Constructing the result record allocates no Tensor ID. Tests lock exact ordered
[producer](../../../../glossary.md#term-producer) references and [operation](../../../../glossary.md#operation) association, not numerical execution.

## Result, replacement, mode, and state [ownership](../../../../glossary.md#term-ownership-meanings)

- `nextCell` is the exact ADD expression joining forget-retained old cell and input-selected
  candidate. `nextHidden` is the exact final MUL expression using output gate and TANH of that
  same exact <span data-glossary-navigation="true"> ([TANH glossary](../../../../glossary.md#term-tanh))</span>`nextCell` reference.
- Every valid forward returns one fresh `LstmCellForwardResult` containing those references in
  `nextHidden`, `nextCell` order. Repeated calls construct independent expression chains and
  carriers.
- Each forward reads every stable parameter wrapper once in declaration order before local
  validation. Schema-compatible replacement before a call affects that call; replacement after
  an expression is built cannot change its provenance references.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [validation glossary](../../../../glossary.md#term-validation); [Schema glossary](../../../../glossary.md#term-schema); [provenance glossary](../../../../glossary.md#provenance))</span>
- There is no atomic multi-parameter snapshot or thread-safety guarantee. Callers coordinate
  replacement and forward when one consistent view matters.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [snapshot glossary](../../../../glossary.md#term-snapshot); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- `train()` and `eval()` may propagate inherited mode normally, but forward constructs the same
  formula in either mode and receives no `ForwardContext`.
- The cell registers no [`Buffer`](../../../../glossary.md#term-buffer) or child. Input, hidden, cell, gates, candidate, and returned
  state are caller- or expression-owned values and never module state.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>
- `LstmCell` is not a `UnaryTensorModule` and cannot appear in `Sequential` by type.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.nn.layers` — owns concrete public NN layers, recurrent cells, and
  cell-specific forward-result values.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- `io.github.pho001.synaptik.nn.module` — supplies direct [`Module`](../../../../glossary.md#term-module) [ownership](../../../../glossary.md#term-ownership-meanings) and stable
  [`Parameter`](../../../../glossary.md#term-parameter) wrappers without modification.
- `io.github.pho001.synaptik.nn.initialization` — supplies current eager parameter initializers
  without modification.<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation))</span>
- [Model](../../../../glossary.md#term-model) data-type, [Shape](../../../../glossary.md#shape), and [Tensor](../../../../glossary.md#tensor) packages — supply existing semantics only.

Packages added or changed:

- No package is added. The existing `nn.layers` package gains `LstmCell`,
  `LstmCellForwardResult`, and package documentation for their explicit two-state contract.

Type placement:

- `io.github.pho001.synaptik.nn.layers.LstmCell` — owns one parameterized LSTM step and its exact
  explicit-state public contract.<span data-glossary-navigation="true"> ([LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm))</span>
- `io.github.pho001.synaptik.nn.layers.LstmCellForwardResult` — owns the named shallow result of
  that step because Model has no LSTM semantic producer and the two next-state values are NN
  composition outputs.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>
- `io.github.pho001.synaptik.nn.layers.LstmCellTest` — owns public surfaces, result validation,
  supplied state, forward/preflight, provenance, replacement, mode, discovery, and exclusions.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [preflight glossary](../../../../glossary.md#term-preflight); [provenance glossary](../../../../glossary.md#provenance))</span>
- `io.github.pho001.synaptik.nn.layers.LstmCellInitializationTest` — owns initialized constructor,
  source order/bounds, zero-bias policy, metadata, early failures, and non-rollback coverage.<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [metadata glossary](../../../../glossary.md#term-metadata); [rollback glossary](../../../../glossary.md#term-rollback))</span>

## Affected files

Expected production files:

- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/LstmCell.java`.
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/LstmCellForwardResult.java`.
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/package-info.java`.

Expected test files:

- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/LstmCellTest.java`.
- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/LstmCellInitializationTest.java`.

Expected documentation and [planning](../../../../glossary.md#term-planning) files:

- `docs/glossary.md`.
- `docs/planning/extensions/nn/master-plan.md`.
- this task specification.

No other file may change for this task. In particular, no [Model](../../../../glossary.md#term-model), Training, architecture, [Gradle](../../../../glossary.md#term-gradle),
[dependency](../../../../glossary.md#term-dependency)-test, sequence, [CPU](../../../../glossary.md#term-cpu), global-roadmap, or concurrent-work path is authorized.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify exactly the eight listed paths and at most:

- three production Java files;
- two [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) test files; and
- three documentation/[planning](../../../../glossary.md#term-planning) files.

If implementation needs another public type, helper file, test [owner](../../../../glossary.md#term-owner), [Model](../../../../glossary.md#term-model) [API](../../../../glossary.md#term-application-programming-interface-api), initializer,
[dependency](../../../../glossary.md#term-dependency), architecture change, or ninth path, stop and propose a separate follow-up task.

## Acceptance criteria

- `LstmCell` is final, extends [`Module`](../../../../glossary.md#term-module) directly, and exposes exactly the seven declared public
  members with no additional public/protected surface or nested type.
- `LstmCellForwardResult` is the exact public final record described above, rejects null
  components in order, and retains exact references without Tensor-expression side effects.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- Supplied constructors retain exact caller Tensors and fully prevalidate the packed schema before
  declaration. Initialized construction uses exact checked packed sizes, state order, Glorot
  calls, optional all-zero bias, source lifecycle, and failure side effects specified above.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [Glorot glossary](../../../../glossary.md#term-glorot-uniform); [bias glossary](../../../../glossary.md#term-bias); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Parameter names, order, Shapes, exact type, gradient eligibility, accessors, recursive paths,
  state-dictionary entries, and absence of buffers/children are exact.<span data-glossary-navigation="true"> ([Parameter glossary](../../../../glossary.md#term-parameter); [Shapes glossary](../../../../glossary.md#shape); [gradient glossary](../../../../glossary.md#term-gradient); [buffers glossary](../../../../glossary.md#term-buffer))</span>
- Gate order is input, forget, candidate, output. Bias is optional, packed, and input-side only.
  Gate activations, next-cell update, and output gate use the exact formula and association above.
  Tests use provenance to distinguish these choices from plausible alternatives.<span data-glossary-navigation="true"> ([activations glossary](../../../../glossary.md#term-activation); [provenance glossary](../../../../glossary.md#provenance))</span>
- Forward rejects every locally knowable invalid request before the first Tensor expression,
  snapshots each current binding once, creates exactly the twenty-five- or twenty-six-Tensor chain
  in documented order, and returns exact final MUL/ADD references as next hidden/cell.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- Valid rank-one and higher inputs use ordinary conservative right-aligned leading broadcasting.
  Final feature checks, mixed floating promotion, both output Shapes/types, slice intervals, and
  failure order match current Model semantics.<span data-glossary-navigation="true"> ([rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [broadcasting glossary](../../../../glossary.md#broadcasting); [feature glossary](../../../../glossary.md#term-feature); [slice glossary](../../../../glossary.md#slice); [Model glossary](../../../../glossary.md#term-model))</span>
- Replacement affects later calls only; existing expressions retain old exact parameter
  references. Mode changes do not affect the expression chain.
- No hidden [recurrent state](../../../../glossary.md#backpropagation-through-time-and-recurrent-state), [unary](../../../../glossary.md#term-unary) adapter, `Sequential` change, context, RNG state, sequence/time
  behavior, scan, Model/initializer change, optimizer, execution behavior, or framework-
  compatibility claim is introduced.<span data-glossary-navigation="true"> ([RNG state glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [scan glossary](../../../../glossary.md#term-cumulative-scan); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- Focused tests cover exact reflection surfaces, record behavior, state schema/order, supplied
  validation, initialized draw/ID order, all-zero forget bias, early failures, all gate intervals,
  exact operation/provenance association, broadcasting/type/rank/feature failures, replacement
  snapshots, mode independence, explicit two-state threading, discovery, and exclusions.<span data-glossary-navigation="true"> ([feature glossary](../../../../glossary.md#term-software-feature))</span><span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [operation glossary](../../../../glossary.md#operation); [rank glossary](../../../../glossary.md#term-rank); [state threading glossary](../../../../glossary.md#state-threading))</span>
- All new and affected public/package APIs have meaningful complete Javadoc for purpose,
  equations, packing, Shapes/types, ownership, nullability, source lifecycle, side effects,
  promotion, broadcasting, result semantics, failure order, concurrency, mode, initialization
  policy, checkpoint compatibility limits, and non-execution boundaries.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [nullability glossary](../../../../glossary.md#term-nullability); [concurrency glossary](../../../../glossary.md#term-concurrency); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- A separate documentation-focused clean context finalizes affected Javadoc, package docs,
  glossary impact, planning evidence, no-change conclusions, links, and generated Javadoc in the
  same overall change.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation pass runs focused tests while developing and, after executable Java stabilizes:

```bash
./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.layers.LstmCellTest --tests io.github.pho001.synaptik.nn.layers.LstmCellInitializationTest
./gradlew :extensions:nn:test
```

The second command is the sole authoritative final affected-[module](../../../../glossary.md#term-module) Java validation. It covers the
existing [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) suite plus the new cell. [Model](../../../../glossary.md#term-model) tests are not repeated: current exhaustive tests remain
authoritative for primitive [linear](../../../../glossary.md#term-linear-meanings), [slice](../../../../glossary.md#slice), ADD, MUL, [SIGMOID](../../../../glossary.md#term-sigmoid), [TANH](../../../../glossary.md#term-tanh), promotion, and broadcast
semantics, while focused NN tests lock cell composition and [preflight](../../../../glossary.md#term-preflight).

Documentation pass runs after final [Javadoc](../../../../glossary.md#term-javadoc) edits:

```bash
./gradlew :extensions:nn:javadoc
git diff --check
git diff --no-index --check /dev/null extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/LstmCell.java
git diff --no-index --check /dev/null extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/LstmCellForwardResult.java
git diff --no-index --check /dev/null extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/LstmCellTest.java
git diff --no-index --check /dev/null extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/LstmCellInitializationTest.java
```

The documentation pass also validates generated type/package pages; exact public and private
surfaces; Model/NN/[JDK](../../../../glossary.md#term-java-development-kit-jdk)-only imports; the unchanged sole [NN Model](../../../../glossary.md#model-module-buffer-and-scan-distinctions) [dependency](../../../../glossary.md#term-dependency); Markdown links,
anchors, fences, final newlines, and trailing whitespace; exact eight-path [scope](../../../../glossary.md#term-scope); task/master
status [synchronization](../../../../glossary.md#term-synchronization); exactly one In progress NN row/spec during implementation; and absence of
a task file for 0015. It changes synchronized status to Complete only after every final gate
passes. It reuses successful implementation tests unless executable Java changes afterward or it
records a concrete reason to rerun them.

Repository-wide, architecture, Model, Training, [CPU](../../../../glossary.md#term-cpu), [compiler](../../../../glossary.md#term-compiler), [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), and integration
tests are deferred to the recurrent NN milestone [checkpoint](../../../../glossary.md#term-checkpoint) or [CI](../../../../glossary.md#term-continuous-integration-ci). This task changes one
Model-only extension [API](../../../../glossary.md#term-application-programming-interface-api) and no dependency, architecture boundary, shared build, or execution
contract.

## Dependencies

- NN tasks 0001–0013 are Complete and provide direct Module ownership, stable schema-validated
  parameters, explicit-source initialization, deterministic discovery/state dictionaries,
  unary-composition exclusions, and recurrent validation/state-threading precedents.<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [schema glossary](../../../../glossary.md#term-schema); [parameters glossary](../../../../glossary.md#term-parameter); [unary glossary](../../../../glossary.md#term-unary); [validation glossary](../../../../glossary.md#term-validation))</span>
- [Model](../../../../glossary.md#term-model) `Tensor.linear`, `sliceAxis`, `add`, `mul`, [`sigmoid`](../../../../glossary.md#term-sigmoid), and [`tanh`](../../../../glossary.md#term-tanh), same-category promotion,
  `ShapeBroadcast`, [immutable](../../../../glossary.md#term-immutable) [descriptors](../../../../glossary.md#term-descriptor)/[provenance](../../../../glossary.md#provenance), and ID allocation are Complete.
- Current independent slice expressions truthfully retain exact packed projection provenance; no
  shared split API is required. Concat is not used because initialized parameters remain direct
  eager leaves and gate packing is one schema, not a runtime composition.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [eager glossary](../../../../glossary.md#term-eager-evaluation); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- `ParameterInitializers.glorotUniform` and `zeros` provide the complete selected initialized
  policy. No new initializer is required because all-zero bias is explicit.<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias))</span>
- [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0007 and existing [dependency](../../../../glossary.md#term-dependency) tests already permit Model-only [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) composition.
- Compiler, Training, CPU, Runtime, Prepare, Engine, backend, and numerical execution support are
  not prerequisites because this task constructs parameter leaves and expression metadata only.<span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [CPU glossary](../../../../glossary.md#term-cpu); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [metadata glossary](../../../../glossary.md#term-metadata))</span>

## Follow-up tasks

- NN 0015 remains Draft without a detailed task: decide static unrolling of concrete cell calls
  versus a genuinely general recurrent Model scan. <span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [unrolling glossary](../../../../glossary.md#term-unrolling); [Model glossary](../../../../glossary.md#term-model); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>`CUM_SUM` and `CUM_PROD` do not satisfy that
  need.
- NN 0015 must reserve packed variable-length sequencing for explicit lengths or an explicit mask,
  never inference from zero-valued data. Padded steps cause no cell invocation. Active-batch
  compaction or stable sorting must restore original batch order and capture final hidden state,
  plus final cell state for LSTM, exactly when each sequence leaves the active set.<span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask); [inference glossary](../../../../glossary.md#term-inference); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [capture glossary](../../../../glossary.md#term-capture-meanings); [LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm))</span>
- If NN 0015 selects containers, prefer cell-specific types unless completed RNN/GRU/LSTM
  signatures prove a shared recurrent contract without casts, state erasure, duplicate carriers,
  or hidden state.<span data-glossary-navigation="true"> ([RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [signatures glossary](../../../../glossary.md#term-signature))</span>
- A non-zero forget-bias default, alternate LSTM packing/bias/equation policy, or framework
  checkpoint conversion requires a concrete consumer and separate task.<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Truncation, masking implementation, bidirectionality, stacking, recurrent dropout, checkpoint
  persistence, training sessions, and backend execution remain separate future work.<span data-glossary-navigation="true"> ([Truncation glossary](../../../../glossary.md#term-truncation-toward-zero); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [persistence glossary](../../../../glossary.md#term-persistence); [backend glossary](../../../../glossary.md#backend))</span>

## Documentation and no-change review

Document profiles:

- Java/package [Javadoc](../../../../glossary.md#term-javadoc): General plus [API](../../../../glossary.md#term-application-programming-interface-api)/Javadoc.
- glossary: General reference style.
- task/master plan: General plus [Planning](../../../../glossary.md#term-planning).

Required implementation-phase documentation changes are `LstmCell`, `LstmCellForwardResult`, and
layers-package Javadocs, the glossary's explicit-state recurrent wording, and synchronized [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)
planning records.

The separate documentation pass must verify and record these reasoned no-change conclusions:

- `ARCHITECTURE.md`, focused architecture pages, ADR 0007, and architecture tests remain accurate
  because state ownership, module direction, and lifecycle boundaries do not change.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Tensor and Compile APIs plus Model source/master/capabilities remain accurate because the cell
  composes existing expressions and adds no Tensor method, semantic kind, split, recurrent scan,
  capture, derivative, or execution behavior.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [Model glossary](../../../../glossary.md#term-model); [scan glossary](../../../../glossary.md#term-cumulative-scan); [capture glossary](../../../../glossary.md#term-capture-meanings); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- Training API and training graph remain accurate because recursive parameter discovery and
  replacement consume this direct Module normally; explicit recurrent states are forward values,
  and no optimizer, session, gradient publication, backpropagation-through-time, truncation, or
  state orchestration is added.<span data-glossary-navigation="true"> ([training graph glossary](../../../../glossary.md#training-graph); [parameter glossary](../../../../glossary.md#term-parameter); [optimizer glossary](../../../../glossary.md#term-optimizer); [gradient glossary](../../../../glossary.md#term-gradient); [publication glossary](../../../../glossary.md#term-publication); [backpropagation glossary](../../../../glossary.md#term-backpropagation); [truncation glossary](../../../../glossary.md#term-truncation-toward-zero))</span>
- `RnnCell`, `GruCell`, `DropoutForwardResult`, `UnaryTensorModule`, `Sequential`, state-dictionary
  contracts, and their tests remain accurate. LSTM has a distinct carrier because it returns two
  different state expressions; transient gates and recurrent values are not module state.<span data-glossary-navigation="true"> ([LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- [Gradle](../../../../glossary.md#term-gradle) and [dependency](../../../../glossary.md#term-dependency) rules/tests remain accurate because `extensions/nn` retains only its
  existing Model dependency.
- Backend conformance and integration tests remain unnecessary because no numerical execution,
  backend capability, or end-to-end lifecycle changes.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [backend capability glossary](../../../../glossary.md#backend-capability))</span>
- The global roadmap, [CPU](../../../../glossary.md#term-cpu) work, other [modules](../../../../glossary.md#term-module), and Draft NN 0015 work remain untouched.

## Architecture impact

Expected impact: None.

This task realizes the existing [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) responsibility for [parameter](../../../../glossary.md#term-parameter)-owning layer composition while
keeping both recurrent states explicit. If implementation requires hidden [module](../../../../glossary.md#term-module)/[runtime](../../../../glossary.md#term-runtime) state, a
universal recurrent abstraction, a [general scan](../../../../glossary.md#general-differentiable-scan), another module [dependency](../../../../glossary.md#term-dependency), or an architecture
rule, stop and report the conflict rather than editing architecture within this task.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are the clean-context implementation agent for Synaptik NN task 0014. Work in the existing
shared worktree. Do not use GSD. Do not commit or push. Preserve every unrelated/concurrent CPU
source, test, documentation, master-plan, task, roadmap, and glossary change exactly.

Read root AGENTS.md, ARCHITECTURE.md, the current architecture plan, planning guide/roadmap,
documentation rules and General/API-Javadoc/Planning profiles, NN master plan and tasks 0001–0014,
ADR 0007, final Module/Parameter/initializers/UnaryTensorModule/Sequential/RnnCell/GruCell and
result-carrier APIs/tests/Javadocs, Model master and final Tensor linear/add/mul/sigmoid/tanh/
slice/Shape/broadcast APIs/tests/Javadocs, Operation/Shape/DataType contracts, Training API/graph,
glossary, and dependency/build rules in full.

Implement exactly the final direct-Module LstmCell and LstmCellForwardResult public APIs, packed
state schema, supplied and initialized validation, all-zero optional bias, explicit caller-
threaded hidden/cell contract, full pre-expression forward preflight, input/forget/candidate/
output slices, fixed equations and activation placement, provenance order, replacement snapshots,
mode independence, and Sequential exclusion in the eight authorized paths. Add no hidden Buffer/
state, UnaryTensorModule adapter, non-zero forget-bias mutation, ForwardContext, GraphRngState,
sequence loop, time traversal, recurrent scan, Model/initializer helper, optimizer/session, backend/
execution behavior, or ninth path.

Run the focused LstmCell tests and one authoritative NN module test after executable Java
stabilizes. Then hand the unchanged executable diff and exact evidence to a distinct clean
documentation-focused context. That context must independently finalize Javadocs, package docs,
glossary, planning evidence, no-change conclusions, generated Javadoc, Markdown, surface, scope,
status, newline, and whitespace gates without repeating successful Java tests unless executable
behavior changes or a concrete risk is recorded.

If current Model APIs cannot express the exact formula and full local preflight, if checked packed
initialization cannot precede the first draw, if the result carrier cannot retain both exact states
without another abstraction, or if explicit type-safe state needs another dependency, stop and
report the exact blocker. Mark Complete only after implementation, documentation, and every
required validation passes.
```

## Documentation-agent [handoff](../../../../glossary.md#term-handoff)

After executable Java/tests stabilize, give the clean documentation context:

- this task and exact eight-path limit;
- the final executable diff and exact focused/final [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) commands, counts, and results;
- exact class/record surfaces, direct-Module/Sequential exclusion, packed state names/order/
  Shapes/types, zero-bias policy, constructor prevalidation, and draw/ID order;<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module); [Shapes glossary](../../../../glossary.md#shape); [bias glossary](../../../../glossary.md#term-bias))</span>
- full three-input preflight, batch-broadcast examples, fixed equations and activation placement,
  twenty-five-/twenty-six-ID producer chains, exact result references, mode behavior, replacement
  snapshots, and failure effects;<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [activation glossary](../../../../glossary.md#term-activation); [producer glossary](../../../../glossary.md#term-producer))</span>
- directly relevant architecture/ADR, documentation profiles, final NN/Model source and tests,
  Tensor/Training APIs, glossary, dependency tests, and planning history;<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor); [dependency glossary](../../../../glossary.md#term-dependency); [planning glossary](../../../../glossary.md#term-planning))</span>
- the mandate to preserve concurrent glossary work and record every reasoned no-change conclusion;
- generated-[Javadoc](../../../../glossary.md#term-javadoc), reflection/`javap`, import/dependency, Markdown, exact-scope/status,
  task-0015-absence, newline, and whitespace gates; and<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- the required completion-summary and Status format from `AGENTS.md`.

## Local decisions

- Use `LstmCell`, following project acronym style (`RnnCell`, `GruCell`), and place it beside the
  concrete recurrent cells. The name represents one cell step, not sequence traversal.
- Extend [`Module`](../../../../glossary.md#term-module) directly. Three Tensor inputs are intrinsic; a unary adapter would have to hide
  input state or retain recurrent state.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [unary glossary](../../../../glossary.md#term-unary); [recurrent state glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state))</span>
- Add public `LstmCellForwardResult(nextHidden, nextCell)`. The two next states are distinct
  expressions and both are required for truthful caller threading. A public record follows the
  existing NN forward-result convention while keeping the carrier cell-specific.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Pack two matrices and optional bias in input, forget, candidate, output order. Independent
  slices make provenance exact with current Model APIs and keep the parameter surface as small as<span data-glossary-navigation="true"> ([matrices glossary](../../../../glossary.md#term-matrix-meanings); [bias glossary](../../../../glossary.md#term-bias); [provenance glossary](../../../../glossary.md#provenance); [Model glossary](../../../../glossary.md#term-model); [parameter glossary](../../../../glossary.md#term-parameter))</span>
  `GruCell`. This order is the [checkpoint](../../../../glossary.md#term-checkpoint) [schema](../../../../glossary.md#term-schema) and is not claimed to match another framework.
- Use one optional packed input-side bias and no recurrent bias. This is the smallest unambiguous
  policy supported by one biased packed linear call.<span data-glossary-navigation="true"> ([linear glossary](../../../../glossary.md#term-linear-meanings))</span>
- Initialize the complete bias to zero. Current initializers offer whole-Tensor zero/one leaves,
  not a direct packed leaf with a selected forget interval set to one. Hidden host mutation or a
  slice/concat-derived parameter would violate the selected eager leaf and provenance policy.<span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [slice glossary](../../../../glossary.md#slice); [eager glossary](../../../../glossary.md#term-eager-evaluation))</span>
- Fix standard uncoupled equations with SIGMOID input/forget/output gates, TANH candidate, additive
  cell update, and output-gated TANH of next cell. Peepholes, projection, clipping, and configurable
  activation placement require separate capabilities.<span data-glossary-navigation="true"> ([SIGMOID glossary](../../../../glossary.md#term-sigmoid); [TANH glossary](../../../../glossary.md#term-tanh); [activation glossary](../../../../glossary.md#term-activation))</span>
- Use Glorot uniform independently for the actual packed matrix Shapes. No orthogonal initializer
  exists, and this task does not introduce one.<span data-glossary-navigation="true"> ([Glorot glossary](../../../../glossary.md#term-glorot-uniform); [Shapes glossary](../../../../glossary.md#shape))</span>
- Permit current Model mixed floating promotion and conservative leading broadcasting, as the
  completed recurrent cells do. Preflight duplicates only local public algebra needed to prevent
  a late gate/state failure from leaving a projection prefix.<span data-glossary-navigation="true"> ([broadcasting glossary](../../../../glossary.md#broadcasting); [Preflight glossary](../../../../glossary.md#term-preflight))</span>
- Accept unresolved input and hidden contraction equality exactly as Model linear does. The cell
  state does not participate in linear contraction: its final Dimension must additionally
  broadcast against the static gate width, so an unresolved cell-feature Dimension is rejected
  when that compatibility cannot be proved locally. Fixed packed parameters and slice bounds make
  every gate final axis statically <span data-glossary-navigation="true"> ([contraction glossary](../../../../glossary.md#contraction-dimension); [Dimension glossary](../../../../glossary.md#dimension); [feature glossary](../../../../glossary.md#term-feature); [axis glossary](../../../../glossary.md#term-axis))</span>`hiddenSize`.
- Keep all leading axes semantically neutral batch coordinates. Task 0015, not this cell, owns any
  time-axis, active-set, or repeated-invocation contract.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>

## Known limitations

- Only this input/forget/candidate/output-packed LSTM with positive fully static parameters and
  one optional all-zero input-side packed bias is supported.<span data-glossary-navigation="true"> ([LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm); [parameters glossary](../../../../glossary.md#term-parameter); [bias glossary](../../../../glossary.md#term-bias))</span>
- Input, hidden, and cell Shapes may contain unresolved leading or final Dimensions only when
  current local Model rules can represent the projection and broadcast obligations. NN performs
  no binding.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [Dimensions glossary](../../../../glossary.md#dimension); [Model glossary](../../../../glossary.md#term-model); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [binding glossary](../../../../glossary.md#term-binding))</span>
- Higher-[rank](../../../../glossary.md#term-rank) inputs are one batched cell application, never an implicit sequence.
- Explicit caller-threaded recurrence may build deep provenance chains. This task adds no
  detachment, truncation, loop IR, scheduling, or memory-lifetime policy.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [truncation glossary](../../../../glossary.md#term-truncation-toward-zero); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- Parameter replacement and forward are not thread-safe as one snapshot. The cell retains no
  lock, version, or transaction.<span data-glossary-navigation="true"> ([thread-safe glossary](../../../../glossary.md#term-thread-safe); [snapshot glossary](../../../../glossary.md#term-snapshot); [transaction glossary](../../../../glossary.md#term-transaction))</span>
- Initialized eager parameters require Java-array-sized host leaves and may consume draws or IDs
  before a later resource failure; effects are not rolled back.<span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [host glossary](../../../../glossary.md#term-host))</span>
- Forward constructs metadata and proves no numerical values, gradient rules, compiler capture,
  backend support, storage allocation, publication, or execution.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [backend glossary](../../../../glossary.md#backend); [publication glossary](../../../../glossary.md#term-publication))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Clean [planning](../../../../glossary.md#term-planning) context `/root/nn_0014_planning` read the repository instructions, architecture
  contract and current plan, planning guide/roadmap, documentation rules and General/Planning/API-
  Javadoc profiles, NN master and completed task history through 0013, final recurrent/module/
  initializer APIs and tests, Model Tensor/Shape/type/operation contracts, Training API/graph,
  glossary, and build/dependency rules before selecting this contract.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor); [Shape glossary](../../../../glossary.md#shape); [operation glossary](../../../../glossary.md#operation); [graph glossary](../../../../glossary.md#graph))</span>
- Planning inspection confirmed current Model linear, independent final-axis slice, ADD, MUL,
  SIGMOID, TANH, promotion, and broadcast APIs can express the complete formula and full local
  preflight. No new Model operation, split, scan, initializer, or dependency is required.<span data-glossary-navigation="true"> ([linear glossary](../../../../glossary.md#term-linear-meanings); [axis glossary](../../../../glossary.md#term-axis); [slice glossary](../../../../glossary.md#slice); [SIGMOID glossary](../../../../glossary.md#term-sigmoid); [TANH glossary](../../../../glossary.md#term-tanh); [preflight glossary](../../../../glossary.md#term-preflight))</span>
- Planning selected input/forget/candidate/output packing, one input-side packed bias, zero-bias
  initialization, fixed activation placement, additive next-cell update, and the NN-owned<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [activation glossary](../../../../glossary.md#term-activation))</span>
  `LstmCellForwardResult(nextHidden, nextCell)` explicitly; no framework compatibility is claimed.
- Targeted planning validation passed for the two changed planning paths: local links and anchors
  resolve, fences balance, final newlines are present, and trailing whitespace is absent.
- The NN planning state has exactly one Ready row/spec (0014). NN 0015 remains the unchanged Draft
  row with no task file, including its explicit lengths/mask, no-zero-value-inference, active-
  batch compaction, original-order restoration, final hidden/cell capture, static-versus-dynamic
  packing, and scan/unroll decision boundaries.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [mask glossary](../../../../glossary.md#term-boolean-mask); [inference glossary](../../../../glossary.md#term-inference); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- Every unrelated concurrent CPU/source/test/documentation/master/task/roadmap/glossary change was
  preserved exactly. Whole-worktree <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>`git diff --check` and the new-file no-index whitespace check
  passed; no Java, Javadoc, Gradle, or test command was run during this planning-only task.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [run glossary](../../../../glossary.md#term-run))</span>
- Clean implementation context `/root/nn_0014_implementation` added the exact direct-[Module](../../../../glossary.md#term-module)
  `LstmCell`, the exact two-component `LstmCellForwardResult`, two focused test suites, and draft
  type/package Javadocs. It required no Model helper, initializer, hidden Buffer/state,
  `ForwardContext`, sequence/[scan](../../../../glossary.md#term-cumulative-scan) abstraction, [dependency](../../../../glossary.md#term-dependency), or ninth task path.
- The stabilized focused command
  `./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.layers.LstmCellTest --tests io.github.pho001.synaptik.nn.layers.LstmCellInitializationTest`
  passed 15 tests with no failures, errors, or skips. The initial development run exposed one
  incorrect test expectation for an unresolved cell-feature Dimension against a static gate
  extent; current conservative Model broadcasting correctly rejects that unprovable pair before
  expression creation, and the corrected test locks that no-ID preflight behavior.<span data-glossary-navigation="true"> ([feature glossary](../../../../glossary.md#term-feature); [Dimension glossary](../../../../glossary.md#dimension); [extent glossary](../../../../glossary.md#term-extent); [broadcasting glossary](../../../../glossary.md#broadcasting))</span>
- Focused provenance inspection covers the exact twenty-five-/twenty-six-ID chain, independent
  input/forget/candidate/output slices from both packed projections, input-side-only bias,
  SIGMOID/TANH placement, additive next-cell update, output-gated activated next cell, exact
  result references, mixed-floating promotion, batch broadcasting, mode independence,
  replacement snapshots, and explicit two-state threading without retained state.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [state threading glossary](../../../../glossary.md#state-threading))</span>
- After the exact result-record surface assertion was finalized, the final authoritative
  `./gradlew :extensions:nn:test` run passed 23 suites and 154 tests with no skips, failures, or
  errors. Executable Java and tests remained unchanged afterward.
- Preliminary `./gradlew :extensions:nn:javadoc` passed. Generated LSTM cell, result, and package
  pages are present. Independent <span data-glossary-navigation="true"> ([LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm))</span>`javap -protected` and `javap -private` inspection confirmed the
  exact final direct-Module public surface, exact generated record surface, only three private
  stable parameter-wrapper fields on the cell, and only the two private final Tensor components
  on the result.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter))</span>
- Source/import and build inspection found only Model, NN, and JDK imports and confirmed the
  unchanged sole production dependency on <span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>`:modules:model`. No Training, Compiler, Runtime,
  Prepare, Engine, backend, context, buffer, recurrent-state field, or retained RNG appears.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- Scope/status inspection found only the seven currently changed NN 0014 paths, with the glossary
  left untouched for the independent documentation pass. Task/master are synchronized at In
  progress; NN 0015 remains the unchanged Draft row with no task file and its packed-sequence
  wording preserved. All unrelated CPU, roadmap, and existing glossary changes remain present.<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope))</span>
- Whole-worktree `git diff --check` passed. Each of the four untracked Java/test files passed
  `git diff --no-index --check /dev/null <path>` with the expected content-difference status and
  no whitespace diagnostic; all four end with one newline.
- Independent clean documentation context `/root/nn_0014_docs` read the final implementation and
  tests, architecture/planning/documentation contracts, complete NN task lineage, recurrent/
  module/initializer/unary boundaries, Model expression/Shape/type contracts, Training API and
  graph, glossary, and build/dependency rules. It found no executable defect, architecture
  uncertainty, scope blocker, or reason to change executable Java or tests.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary))</span>
- The documentation context finalized `LstmCell`, `LstmCellForwardResult`, and layers-package
  Javadocs. It clarified that unresolved input/hidden contraction equality may remain deferred by
  Model linear while every gate/state broadcast must be locally provable, so an unresolved cell-
  feature Dimension paired with static <span data-glossary-navigation="true"> ([contraction glossary](../../../../glossary.md#contraction-dimension))</span>`hiddenSize` is rejected before expression creation. This
  matches the stabilized implementation and focused test rather than changing behavior.
- The glossary now defines the current LSTM cell, input/forget/candidate/output packed checkpoint
  schema, input-side-only bias, all-zero initialized bias including forget, two-state formula and
  exact result order, caller-threaded ownership, direct-Module/Sequential exclusion, and the
  distinctions among vanilla RNN, GRU, gate packing, future packed variable-length sequences,
  recurrent scan, and cumulative scan.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [schema glossary](../../../../glossary.md#term-schema); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [cumulative scan glossary](../../../../glossary.md#cumulative-scan))</span>
- The documentation pass reused the implementation context's focused two-suite/15-test result and
  authoritative 23-suite/154-test NN result because no executable Java or test changed afterward.
  Javadoc and prose edits do not stale that evidence, so no Java test suite was repeated.
- Final `./gradlew :extensions:nn:javadoc` passed after the [Javadoc](../../../../glossary.md#term-javadoc) edits (`BUILD SUCCESSFUL`; 3
  actionable tasks, 2 executed and 1 up-to-date). Inspection of generated `LstmCell.html`,
  `LstmCellForwardResult.html`, and `layers/package-summary.html` confirmed the exact signatures,
  parameters, returns, failures, packing, initialization/source lifecycle, equations, two-state
  ownership, broadcast/preflight boundary, mode/replacement semantics, Sequential exclusion, and
  non-execution limits.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Final `javap -protected` and `javap -private` confirmed direct final `Module`, exactly three
  public constructors and four declared public methods, no protected surface or nested type, only
  three private final parameter-wrapper fields, the exact generated record surface, and only two
  private final Tensor components in next-hidden/next-cell order. A standalone reflection program
  compiled and printed `LSTM reflection surface: PASS`. An exploratory JShell check independently
  printed the record result but was non-authoritative because one expression had a `Module` name
  ambiguity and JShell then failed to persist Preferences history; neither event changed a file.
- Source/import and build inspection found only Model, existing NN, and JDK production imports and
  retained the sole `implementation(project(":modules:model"))` dependency. No Training,
  Compiler, Runtime, Prepare, Engine, backend, `ForwardContext`, [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate), [`Buffer`](../../../../glossary.md#term-buffer), hidden
  recurrent Tensor field, or retained random source appears.
- The targeted Markdown validator passed the glossary, NN master plan, and task with 356 local
  links, 295 anchors, balanced fences, final newlines, and no trailing whitespace. Exact-scope
  inspection found exactly the eight authorized NN 0014 paths; this context did not edit
  concurrent CPU or roadmap work. The master/task both read Complete; NN 0015 remains Draft with
  no task file and retains its
  explicit lengths/mask, no-zero inference, padded-call omission, active compaction/order restore,
  final hidden/cell capture, static/dynamic, and scan/unroll boundaries.
- Final XML inspection retained 23 suites/154 tests with zero skips, failures, or errors. Each of
  the five untracked task-owned files passed `git diff --no-index --check /dev/null <path>` with
  expected content-difference status 1 and no whitespace diagnostic. Final whole-worktree
  `git diff --check` passed with no output.
- `ARCHITECTURE.md`, focused architecture pages, ADR 0007, and architecture tests require no
  change because the cell stays inside existing Model-only NN composition and keeps both recurrent
  states caller-owned. Tensor/Compile APIs, Model source/master/capabilities, and related operation
  contracts require no change because this task only composes current linear, slice, ADD, MUL,
  SIGMOID, and TANH metadata and adds no semantic kind, split, scan, derivative, capture, or
  execution contract.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Compile glossary](../../../../glossary.md#compile); [metadata glossary](../../../../glossary.md#term-metadata); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- Training API and the training graph require no change because recursive parameter discovery and
  replacement already consume this direct Module and the task adds no optimizer, session,
  gradient publication, backpropagation-through-time, truncation, or state orchestration.<span data-glossary-navigation="true"> ([training graph glossary](../../../../glossary.md#training-graph); [optimizer glossary](../../../../glossary.md#term-optimizer); [gradient glossary](../../../../glossary.md#term-gradient); [publication glossary](../../../../glossary.md#term-publication); [backpropagation glossary](../../../../glossary.md#term-backpropagation); [truncation glossary](../../../../glossary.md#term-truncation-toward-zero))</span>
  `RnnCell`, `GruCell`, `DropoutForwardResult`, `UnaryTensorModule`, `Sequential`, state-dictionary
  contracts, and their tests remain accurate: LSTM alone needs the distinct two-state carrier,
  while transient gates and recurrent values never become module state.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Gradle/dependency rules and tests, backend conformance, and integration tests require no change
  because no dependency, numerical execution, backend capability, or end-to-end lifecycle changed.
  The LSTM task did not edit the global roadmap, CPU work, other modules, Model capabilities, or
  Draft NN 0015; their concurrent worktree changes were preserved exactly.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [backend capability glossary](../../../../glossary.md#backend-capability))</span>

## Implementation notes

Clean implementation context `/root/nn_0014_implementation` completed the executable source and
focused tests in the planned paths. The focused two-suite/15-test selection and sole authoritative
23-suite/154-test [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [module](../../../../glossary.md#term-module) [run](../../../../glossary.md#term-run) pass. Independent clean documentation context
`/root/nn_0014_docs` then finalized type/package Javadocs, glossary and [planning](../../../../glossary.md#term-planning) evidence, and
passed generated-[Javadoc](../../../../glossary.md#term-javadoc), rendered-page, `javap`/reflection, [dependency](../../../../glossary.md#term-dependency)/import, Markdown, exact
eight-path [scope](../../../../glossary.md#term-scope), status, newline, whitespace, no-index, and final-diff gates without changing
executable behavior or repeating the stable Java suites.

## Completion summary

- Completed changes: added final direct-[Module](../../../../glossary.md#term-module) `LstmCell` with exact supplied and explicit-source
  construction, IFGO-packed state, complete forward preflight, fixed two-state formula, exact
  result carrier, and focused contract tests; finalized every affected documentation contract.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Files changed or created: exactly `LstmCell.java`, `LstmCellForwardResult.java`, layers
  `package-info.java`, `LstmCellTest.java`, `LstmCellInitializationTest.java`, `docs/glossary.md`,
  the NN master plan, and this task; unrelated concurrent paths were preserved.
- Tests and validation: reused the passing focused 2-suite/15-test and authoritative
  23-suite/154-test NN evidence with zero failures, errors, or skips. Final NN Javadoc/generated-
  page inspection, <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`javap`, standalone reflection, import/dependency, Markdown, exact-scope/
  status, newline/whitespace, no-index, and whole-diff checks passed.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: clean context `/root/nn_0014_docs` completed the mandatory
  independent source/test/API/architecture review and changed no executable Java or test.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Documentation impact: finalized the LSTM/result/package Javadocs, added the glossary's current
  LSTM checkpoint schema and explicit two-state example plus sequence/scan distinctions, and
  synchronized task/master status and evidence.<span data-glossary-navigation="true"> ([LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [schema glossary](../../../../glossary.md#term-schema); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- Javadoc review: complete; generated type/result/package pages contain the exact surface, state,
  initialization, source, equation, failure, reference, concurrency, mode, and non-execution
  contracts.<span data-glossary-navigation="true"> ([concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- Glossary impact: added the reusable LSTM and cell-state terms, IFGO packed schema, zero-bias
  policy, exact next-hidden/next-cell order, and distinctions from RNN, GRU, future packed
  sequences, recurrent/cumulative scan, and <span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [cumulative scan glossary](../../../../glossary.md#cumulative-scan))</span>`Sequential`.
- Unresolved issues: None.
- Follow-up required: None for task 0014; [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0015 remains separate Draft work.

Status: Complete

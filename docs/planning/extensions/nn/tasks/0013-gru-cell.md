# Task 0013: [GRU](../../../../glossary.md#term-gated-recurrent-unit-gru) Cell

> **Editorial reading context — not a status update.** The gated recurrent unit's reset,
> update and candidate slices are parts of packed projection parameters, not separately owned
> modules. The application supplies previous hidden state and receives one next-hidden
> expression; the cell retains only parameter bindings. Reset-after names where the reset
> control is applied relative to the hidden projection, so a similarly named GRU elsewhere
> need not have interchangeable parameters. Today's [cell/sequence contract](../../../../reference/nn/recurrent-layers.md#current-nn-recurrent-composition-contract)
> explains state threading, and [runtime-length composition](../../../../reference/nn/recurrent.md#nn-runtime-length-recurrent-composition)
> contrasts later scan construction without rewriting this single-cell history.

## Status

Complete

## Goal

Add one final public [gated recurrent unit](../../../../glossary.md#term-gated-recurrent-unit-gru) ([GRU](../../../../glossary.md#neural-network-families)) cell with explicit caller-threaded hidden state.
The cell owns two gate-major packed projection [matrices](../../../../glossary.md#term-matrix-meanings) and one optional packed input-side [bias](../../../../glossary.md#term-bias),
constructs a fixed reset-after GRU step entirely from current [Model](../../../../glossary.md#term-model) [Tensor](../../../../glossary.md#tensor) expressions, and
returns the next hidden Tensor as its sole result. It retains no [recurrent state](../../../../glossary.md#backpropagation-through-time-and-recurrent-state).

[Mental model](../../../../glossary.md#term-mental-model):

```text
caller input x + caller hidden h + current packed parameters
  -> input and hidden packed linear projections
  -> reset, update, and candidate slices in fixed gate order
  -> reset-after candidate and fixed update interpolation
  -> one Tensor h' returned to the caller
  -> caller explicitly threads h' into a later call when recurrence is intended
```

As with `RnnCell`, the visible cell output and next hidden state are the same Tensor. A result
record with `output` and `nextHidden` would duplicate one exact reference and is not introduced.

## [Scope](../../../../glossary.md#term-scope)

- Add final public `io.github.pho001.synaptik.nn.layers.GruCell` extending
  `io.github.pho001.synaptik.nn.module.Module` directly.
- Add exactly the constructors, parameter accessors, and two-input forward method in the public
  API table below. Add no overload, builder, options object, gate enum, state carrier, size getter,
  functional facade, or configurable equation policy.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [facade glossary](../../../../glossary.md#term-facade))</span>
- Declare positive fully static [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [parameters](../../../../glossary.md#term-parameter) under exact local names `inputWeight` then
  `hiddenWeight`. Their [Shapes](../../../../glossary.md#shape) are `[3 * hiddenSize, inputSize]` and
  `[3 * hiddenSize, hiddenSize]`.
- Pack both matrices gate-major on axis zero in exact reset, update, candidate order. After linear
  projection, that same order occupies the final result axis.<span data-glossary-navigation="true"> ([matrices glossary](../../../../glossary.md#term-matrix-meanings); [axis glossary](../../../../glossary.md#term-axis); [linear projection glossary](../../../../glossary.md#linear-projection))</span>
- Optionally declare one positive fully static [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) parameter under exact local name [`bias`](../../../../glossary.md#term-bias),
  with Shape `[3 * hiddenSize]`. It is added only by the packed input projection. There is no
  recurrent bias and no separate per-gate bias parameter.<span data-glossary-navigation="true"> ([bias parameter glossary](../../../../glossary.md#bias-and-activation))</span>
- Require all parameters to share one exact floating [data type](../../../../glossary.md#data-type--datatype) and have
  `requiresGrad == true`. Retain caller-supplied parameter [Tensors](../../../../glossary.md#tensor) exactly.
- Provide supplied-state construction both without and with bias. Null never means absent bias;
  callers select absence through the two-Tensor constructor.
- Provide one initialized constructor with explicit positive `inputSize`, positive `hiddenSize`,
  bias presence, floating [`DataType`](../../../../glossary.md#data-type--datatype), and caller-owned `RandomGenerator`.
- Compute `packedHiddenSize = 3 * hiddenSize` with checked arithmetic before constructing Shapes.
  Initialize `inputWeight` then `hiddenWeight` through exact
  `ParameterInitializers.glorotUniform` calls using the same explicit source. When requested,
  initialize bias afterward through exact `ParameterInitializers.zeros`; bias consumes no draw.
  The source is never retained.
- Add exactly `Tensor forward(Tensor input, Tensor hidden)`. Both semantic inputs are explicit on
  every call; the cell never supplies, caches, updates, registers, or discovers hidden state.
- Fix the reset-after convention, gate order, optional-bias association, and interpolation formula
  specified below. Do not claim compatibility with another framework whose bias or equation
  convention may differ.
- Use independent `sliceAxis(-1, ...)` expressions for the six projection gate slices. Current
  Model has no shared multi-output split operation; independent slices keep each exact packed
  source and interval visible without adding a Model API or constructing per-gate parameters.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- Support input and hidden [rank](../../../../glossary.md#term-rank) one or higher. Their final [Dimensions](../../../../glossary.md#dimension) contract respectively with
  `inputSize` and `hiddenSize` under current [Model](../../../../glossary.md#term-model) [linear](../../../../glossary.md#term-linear-meanings) semantics.
- Treat all leading Dimensions as ordinary right-broadcastable batch metadata. No leading axis is
  named, interpreted, or traversed as time.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Complete all caller-controlled forward validation before creating the first projection
  expression. Then construct exactly the formula and producer order specified below.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [producer glossary](../../../../glossary.md#term-producer))</span>
- Keep the cell mode-insensitive. Forward accepts no `ForwardContext`, does not inspect `mode()`,
  and does not alter module mode or state.
- Preserve stable parameter wrappers, schema-compatible replacement snapshots, recursive
  discovery, and state-dictionary paths through current <span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>[`Module`](../../../../glossary.md#term-module) and `Parameter` contracts.
- Add focused exact-surface, supplied-state, initialization, validation-order, shape/type,
  gate-slicing, provenance, replacement, mode, state-discovery, and exclusion tests.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- Add complete type, constructor, member, and package Javadocs. After executable work and final NN
  testing, use a separate clean documentation-focused context to finalize Javadocs, glossary
  impact, planning evidence, generated Javadoc, and reasoned no-change conclusions.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [planning glossary](../../../../glossary.md#term-planning); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Retaining the current or next hidden [Tensor](../../../../glossary.md#tensor) in a field, [`Buffer`](../../../../glossary.md#term-buffer), child, state dictionary,
  thread-local, static, runtime object, session, or any other hidden lifecycle.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- A default or zero hidden state, hidden-state initializer, reset/detach API, stateful forward
  overload, or result carrier that obscures caller state threading.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [state threading glossary](../../../../glossary.md#state-threading))</span>
- `UnaryTensorModule`, participation in `Sequential`, an adapter into `Sequential`, or any change
  to `Sequential`, `UnaryTensorModule`, [`Module`](../../../../glossary.md#term-module), [`Parameter`](../../../../glossary.md#term-parameter), or `Buffer`.
- Separate public gate parameters, six per-gate matrices, separate input/recurrent biases,
  per-gate bias accessors, or a parameter-packing configuration.<span data-glossary-navigation="true"> ([matrices glossary](../../../../glossary.md#term-matrix-meanings); [biases glossary](../../../../glossary.md#term-bias))</span>
- Reset-before candidate projection, reset applied to raw hidden state before its candidate
  matrix, alternate update interpolation, gate-order configuration, framework-compatibility mode,
  or migration/conversion helpers for another GRU layout.<span data-glossary-navigation="true"> ([GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [layout glossary](../../../../glossary.md#layout))</span>
- A recurrent sequence container, loop, step counter, time-axis convention, masks, sequence
  lengths, packed sequence, variable-length handling, bidirectionality, stacking, recurrent
  dropout, static unrolling, or time traversal.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [masks glossary](../../../../glossary.md#term-boolean-mask); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [unrolling glossary](../../../../glossary.md#term-unrolling))</span>
- A general recurrent Model scan, loop operation, subgraph/body representation, carried-value
  tuple, scan result, or Tensor API. Current <span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [operation glossary](../../../../glossary.md#operation))</span>`CUM_SUM` and `CUM_PROD` are associative cumulative
  operations and are not recurrent scan primitives.
- LSTM state, peepholes, projections, layer normalization, clipping, residuals, attention,
  convolutional recurrence, or another recurrent-cell family.<span data-glossary-navigation="true"> ([LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm); [layer normalization glossary](../../../../glossary.md#layer-normalization))</span>
- Dynamic or zero-sized parameters, integral/BOOL parameters, frozen parameters, quantized or
  sparse state, custom initializer objects, orthogonal initialization, or a retained/default RNG.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- `ForwardContext`, [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate), forward-time `RandomGenerator`, mode-dependent branching,
  Buffer transitions, or stochastic recurrence.
- Backpropagation through time, gradient detachment, gradient rules, optimizer/session,
  parameter groups, checkpoint transport, serialization, compiler capture, scheduling,
  Runtime/Prepare/Engine behavior, backend lowering, kernels, numerical execution, or end-to-end
  support claims.<span data-glossary-navigation="true"> ([Backpropagation through time glossary](../../../../glossary.md#term-backpropagation-through-time-bptt); [gradient glossary](../../../../glossary.md#term-gradient); [optimizer glossary](../../../../glossary.md#term-optimizer); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [serialization glossary](../../../../glossary.md#term-serialization); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel))</span>
- A Model, Training, Gradle, dependency, architecture-contract, ADR, architecture-test, global-
  roadmap, CPU, or explanatory-architecture source change during implementation.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Detailed task specifications for [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0014 or 0015.

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
- [Completed Model task 0017H: Slice expressions](../../../modules/model/tasks/0017h-slice-tensor-expressions.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Slice glossary](../../../../glossary.md#slice))</span>
- [Completed Model task 0019D: Linear convenience](../../../modules/model/tasks/0019d-linear-convenience.md)<span data-glossary-navigation="true"> ([Linear glossary](../../../../glossary.md#term-linear-meanings))</span>
- [Completed Model task 0023E: Cumulative scan normalization](../../../modules/model/tasks/0023e-cumulative-scan-normalization-and-product.md)<span data-glossary-navigation="true"> ([Cumulative scan glossary](../../../../glossary.md#cumulative-scan); [normalization glossary](../../../../glossary.md#term-normalization))</span>

## Architecture constraints

- `extensions/nn` continues to depend only on `modules/model`. Production must not import
  Training, Compiler, Runtime, Prepare, Engine, CPU, Metal, CUDA, or another backend.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal); [CUDA glossary](../../../../glossary.md#term-cuda); [backend glossary](../../../../glossary.md#backend))</span>
- NN owns packed parameter bindings and typed cell composition. Model remains the sole owner of
  LINEAR decomposition, SLICE, ADD, SUB, MUL, SIGMOID, TANH, type promotion, Shape algebra,
  descriptors, provenance, and Tensor identity.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [parameter glossary](../../../../glossary.md#term-parameter); [bindings glossary](../../../../glossary.md#term-binding); [Model glossary](../../../../glossary.md#term-model); [owner glossary](../../../../glossary.md#term-owner); [LINEAR glossary](../../../../glossary.md#term-linear-meanings); [decomposition glossary](../../../../glossary.md#term-decomposition); [SLICE glossary](../../../../glossary.md#slice); [SIGMOID glossary](../../../../glossary.md#term-sigmoid); [TANH glossary](../../../../glossary.md#term-tanh); [Shape glossary](../../../../glossary.md#shape); [descriptors glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [Tensor glossary](../../../../glossary.md#tensor); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Tensor identity, descriptors, and provenance remain immutable. Parameter replacement changes
  only the current exact Tensor returned by one stable wrapper; existing expressions remain
  unchanged.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- Recurrent state is a caller-threaded Tensor value, never module-owned persistent state. It is
  not a <span data-glossary-navigation="true"> ([Recurrent state glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state))</span>[`Buffer`](../../../../glossary.md#term-buffer) merely because the caller may pass it between calls.
- [`Module`](../../../../glossary.md#term-module) retains no universal forward method. `GruCell` is a direct subclass with a truthful
  two-input signature and must not extend <span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>`UnaryTensorModule`.
- `Sequential` remains a container only for `UnaryTensorModule`; no adapter may capture or erase
  the hidden-state input.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- Mode is NN composition metadata. This mode-insensitive cell neither consumes a context nor
  reads inherited mode during forward.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>
- Packed slices are ordinary independent one-output Model expressions. The task must not invent a
  shared split producer, tuple value, hidden gate cache, direct <span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [cache glossary](../../../../glossary.md#term-cache))</span>[`Operation`](../../../../glossary.md#operation) construction, or
  special GRU semantic kind.<span data-glossary-navigation="true"> ([GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru))</span>
- Construction and forward create eager parameter leaves or storage-free expression metadata
  only. They do not prove gradient implementation, graph capture, compiled execution, backend
  support, numerical values, or storage residency.<span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [gradient glossary](../../../../glossary.md#term-gradient); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [residency glossary](../../../../glossary.md#residency))</span>
- If implementation needs a new Model operation/helper/public method, public result type,
  recurrent container, state Buffer, context/RNG input, dependency, architecture rule, or eighth
  task path, stop and report the exact blocker instead of widening the task.<span data-glossary-navigation="true"> ([RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [dependency glossary](../../../../glossary.md#term-dependency); [widening glossary](../../../../glossary.md#term-widening))</span>

## Public [API](../../../../glossary.md#term-application-programming-interface-api)

`GruCell` declares exactly:

```java
public GruCell(Tensor inputWeight, Tensor hiddenWeight)

public GruCell(Tensor inputWeight, Tensor hiddenWeight, Tensor bias)

public GruCell(
        long inputSize,
        long hiddenSize,
        boolean bias,
        DataType dataType,
        RandomGenerator randomGenerator)

public Parameter inputWeight()
public Parameter hiddenWeight()
public Optional<Parameter> bias()
public Tensor forward(Tensor input, Tensor hidden)
```

| Member | Exact contract |
|---|---|
| two-[Tensor](../../../../glossary.md#tensor) constructor | Retains exact positive static packed input/hidden projection [weights](../../../../glossary.md#term-weight) and declares no [bias](../../../../glossary.md#term-bias). |
| three-Tensor constructor | Retains the same exact weights plus one exact packed input-side bias; null bias is invalid. |
| initialized constructor | Uses explicit sizes/type/source, creates packed input weight then packed hidden weight through [Glorot](../../../../glossary.md#term-glorot-uniform) uniform and optional zero bias afterward. |
| `inputWeight()` | Returns the exact stable wrapper declared under `inputWeight`. |
| `hiddenWeight()` | Returns the exact stable wrapper declared under `hiddenWeight`. |
| `bias()` | Returns an empty Optional or the exact stable wrapper declared under `bias`. |
| `forward(input, hidden)` | Validates the complete two-input request, snapshots current [parameters](../../../../glossary.md#term-parameter), constructs the fixed reset-after formula once, and returns the exact next-hidden Tensor. |

The class declares no other public or protected constructor, method, field, nested type,
interface, or overload. Inherited final [`Module`](../../../../glossary.md#term-module) APIs remain available normally.

## State [schema](../../../../glossary.md#term-schema) and [ownership](../../../../glossary.md#term-ownership-meanings)

| Kind | Local name | [Shape](../../../../glossary.md#shape) orientation | Type | `requiresGrad` | Initialized policy |
|---|---|---|---|---|---|
| [Parameter](../../../../glossary.md#term-parameter) | `inputWeight` | `[3 * hiddenSize, inputSize]` | exact configured floating type | `true` | [Glorot](../../../../glossary.md#term-glorot-uniform) uniform from explicit source |
| Parameter | `hiddenWeight` | `[3 * hiddenSize, hiddenSize]` | same exact type | `true` | Glorot uniform from the same explicit source |
| Parameter, optional | [`bias`](../../../../glossary.md#term-bias) | `[3 * hiddenSize]` | same exact type | `true` | exact typed zero |

[Axis](../../../../glossary.md#term-axis) zero of each [matrix](../../../../glossary.md#term-matrix-meanings) and axis zero of bias use these [contiguous](../../../../glossary.md#term-contiguous) intervals:

| Interval | Gate |
|---|---|
| `[0, hiddenSize)` | reset |
| `[hiddenSize, 2 * hiddenSize)` | update |
| `[2 * hiddenSize, 3 * hiddenSize)` | candidate |

For supplied construction, `inputWeight` axis zero must be positive and divisible by three.
`hiddenSize` is exactly that static [extent](../../../../glossary.md#term-extent) divided by three. `hiddenWeight` axis zero and optional
bias axis zero must structurally equal the complete packed extent; `hiddenWeight` axis one must
structurally equal the derived `hiddenSize`. `inputSize` is `inputWeight` axis one. Both logical
sizes are positive.

Direct discovery and state-dictionary order are `inputWeight`, `hiddenWeight`, then optional
`bias`. Under a future parent child name `cell`, paths become `cell.inputWeight`,
`cell.hiddenWeight`, and optional `cell.bias`. The input hidden [Tensor](../../../../glossary.md#tensor), gates, candidate, and next
hidden Tensor never appear in parameter, [buffer](../../../../glossary.md#term-buffer), child, or state-dictionary discovery.

## Supplied construction [validation](../../../../glossary.md#term-validation) and side effects

Both supplied constructors validate complete state before declaring any [parameter](../../../../glossary.md#term-parameter).

For the two-[Tensor](../../../../glossary.md#tensor) constructor:

1. reject null `inputWeight`, then null `hiddenWeight`;
2. validate input [weight](../../../../glossary.md#term-weight) floating type, `requiresGrad == true`, rank two, fully static Shape,
   positive packed axis zero, divisibility of packed axis zero by three, then positive input-size
   axis one;
3. validate hidden weight floating type, `requiresGrad == true`, rank two, fully static Shape,
   positive axis zero, then positive axis one;
4. require hidden weight exact [data type](../../../../glossary.md#data-type--datatype) to equal input weight data type;
5. require hidden weight [axis](../../../../glossary.md#term-axis) zero to equal the complete input-weight packed [extent](../../../../glossary.md#term-extent) structurally;
6. require hidden weight axis one to equal the derived hidden-size static [Dimension](../../../../glossary.md#dimension); and
7. declare `inputWeight` then `hiddenWeight` and retain no [bias](../../../../glossary.md#term-bias).

For the three-Tensor constructor, reject null `bias` after the two weight null checks and before
[schema](../../../../glossary.md#term-schema) validation. Apply steps 2–6, then validate bias floating type, `requiresGrad == true`, [rank](../../../../glossary.md#term-rank)
one, fully static [Shape](../../../../glossary.md#shape), exact common data type, and exact complete packed extent in that order.
Declare `inputWeight`, `hiddenWeight`, then `bias` only after all checks pass.

Validation creates no Tensor, [producer](../../../../glossary.md#term-producer), storage, random draw, or Tensor [identity](../../../../glossary.md#term-identity-meanings) and never mutates
the supplied values. A failure returns no cell and leaves every supplied Tensor unchanged.

## Initialized construction and side effects

The initialized constructor performs exactly:

1. reject null `dataType`, then null `randomGenerator`;
2. require `inputSize > 0`;
3. require `hiddenSize > 0`;
4. require the [data type](../../../../glossary.md#data-type--datatype) to be floating;
5. compute `packedHiddenSize = Math.multiplyExact(hiddenSize, 3L)`;
6. construct [Shapes](../../../../glossary.md#shape) `[packedHiddenSize, inputSize]`, `[packedHiddenSize, hiddenSize]`, and, only
   when requested, `[packedHiddenSize]`;
7. obtain each requested Shape's checked known element count in state order and reject a count
   above <span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element))</span>`Integer.MAX_VALUE` before the first draw or [Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings) allocation;
8. create input [weight](../../../../glossary.md#term-weight) through exactly
   `ParameterInitializers.glorotUniform(inputWeightShape, dataType, randomGenerator)`;
9. create hidden weight through exactly
   `ParameterInitializers.glorotUniform(hiddenWeightShape, dataType, randomGenerator)` using the
   same now-advanced source;
10. when requested, create [bias](../../../../glossary.md#term-bias) through exactly
    `ParameterInitializers.zeros(biasShape, dataType)` with no source call; and
11. after all requested leaves exist, declare [parameters](../../../../glossary.md#term-parameter) in exact state-table order.

Caller-controlled null, size, type, multiplication-[overflow](../../../../glossary.md#term-overflow), checked-count, and Java-array-limit
failures precede every draw and Tensor ID. Weight bounds are computed independently from each
packed [matrix](../../../../glossary.md#term-matrix-meanings)'s actual `[fanOut, fanIn]` Shape. Bias consumes no draw.

A source failure keeps completed source calls and creates no Tensor for the failing weight. If it
occurs during hidden-weight creation, the already created input weight and its ID are not rolled
back, but no cell is returned. Later allocation or identifier failures similarly preserve
completed effects. The caller owns the source; the cell never retains, resets, closes,
synchronizes, splits, [seeds](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings), or serializes it.

## Forward [validation](../../../../glossary.md#term-validation) and side-effect order

`forward(input, hidden)` performs exactly:

1. reject null `input`, then null `hidden`;
2. read current `inputWeight`, `hiddenWeight`, and optional [`bias`](../../../../glossary.md#term-bias) bindings exactly once in
   parameter declaration order;
3. prevalidate the packed input affine projection under Model linear order: promote input and
   input-weight numeric types, require input rank at least one, reject a proven unequal static
   input-feature contraction, and, when bias is present, promote product and bias types;<span data-glossary-navigation="true"> ([weight glossary](../../../../glossary.md#term-weight))</span>
4. prevalidate the packed hidden projection: promote hidden and hidden-weight numeric types,
   require hidden rank at least one, and reject a proven unequal static hidden-feature contraction;
5. derive both packed projection Shapes without creating a Tensor: preserve complete leading
   prefixes and append the exact static packed extent;
6. derive the six gate-[slice](../../../../glossary.md#slice) [Shapes](../../../../glossary.md#shape) by replacing the final packed [extent](../../../../glossary.md#term-extent) with exact static
   `hiddenSize`, preserving every [leading Dimension](../../../../glossary.md#term-leading-dimension) reference from its projection;
7. prevalidate reset preactivation promotion and Shape broadcasting for input-reset plus
   hidden-reset, then the same for update;
8. prevalidate reset-gate times hidden-candidate promotion and broadcast, then candidate-input
   plus that reset product, followed by candidate TANH eligibility;<span data-glossary-navigation="true"> ([TANH glossary](../../../../glossary.md#term-tanh))</span>
9. prevalidate hidden minus candidate promotion and broadcast, update times that difference, and
   candidate plus the weighted difference; and
10. only after all preceding checks succeed, construct the exact formula below.

All [parameters](../../../../glossary.md#term-parameter) are floating, so current same-category promotion requires both call inputs to be
floating while allowing [BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) [widening](../../../../glossary.md#term-widening). A static final input or hidden [feature](../../../../glossary.md#term-feature)
[Dimension](../../../../glossary.md#dimension) must equal its configured size. An unresolved final feature Dimension is accepted under
current [Model](../../../../glossary.md#term-model) [linear](../../../../glossary.md#term-linear-meanings) semantics and leaves that obligation for later [compiler](../../../../glossary.md#term-compiler) validation or
[binding](../../../../glossary.md#term-binding).

Leading-prefix [broadcasting](../../../../glossary.md#broadcasting) is conservative. Equal Dimensions and static [singleton](../../../../glossary.md#term-singleton) expansion
succeed; incompatible static sizes and locally unprovable symbolic combinations fail before
expression creation. Input and hidden ranks need not equal. Examples:

| Input Shape | Hidden Shape | Next-hidden Shape | Meaning |
|---|---|---|---|
| `[inputSize]` | `[hiddenSize]` | `[hiddenSize]` | one unbatched cell application |
| `[batch, inputSize]` | `[batch, hiddenSize]` | `[batch, hiddenSize]` | matching [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) prefix |
| `[batch, inputSize]` | `[hiddenSize]` | `[batch, hiddenSize]` | hidden [vector](../../../../glossary.md#term-vector) broadcast across batch |
| `[outer, 1, inputSize]` | `[batch, hiddenSize]` | `[outer, batch, hiddenSize]` | ordinary right-aligned batch broadcasting |

These examples describe [metadata](../../../../glossary.md#term-metadata) composition only. `outer` and `batch` are leading [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix),
not a time traversal or sequence loop.

Every local null/type/[rank](../../../../glossary.md#term-rank)/static-[contraction](../../../../glossary.md#contraction-dimension)/slice/broadcast failure consumes no [Tensor](../../../../glossary.md#tensor) ID and
creates no expression prefix. Parameter reads and temporary Shape/type values are not expression
side effects. Identifier exhaustion during valid construction may leave an expression prefix; no
[rollback](../../../../glossary.md#term-rollback) is attempted.

## Formula, delegation, and [provenance](../../../../glossary.md#provenance)

Let `x` be input, `h` be hidden, `W` be `inputWeight`, `U` be `hiddenWeight`, and `b` be the
optional input-side packed [bias](../../../../glossary.md#term-bias). Let `H` be `hiddenSize`. Packed input projection `P_x` and packed
hidden projection `P_h` are:

```text
P_x = x @ transpose(W) + b       when bias is present
P_x = x @ transpose(W)           otherwise
P_h = h @ transpose(U)
```

[Slice](../../../../glossary.md#slice) the final [axes](../../../../glossary.md#term-axis) independently in fixed order:

```text
x_r = P_x[..., 0:H]
x_z = P_x[..., H:2H]
x_n = P_x[..., 2H:3H]
h_r = P_h[..., 0:H]
h_z = P_h[..., H:2H]
h_n = P_h[..., 2H:3H]
```

The fixed reset-after [GRU](../../../../glossary.md#term-gated-recurrent-unit-gru) equations are:

```text
r = sigmoid(x_r + h_r)
z = sigmoid(x_z + h_z)
n = tanh(x_n + r * h_n)
h' = n + z * (h - n)
```

The reset gate therefore multiplies the recurrent candidate projection `h_n` after its [matrix](../../../../glossary.md#term-matrix-meanings)
projection. Because no recurrent bias exists, nothing is hidden inside that product. The update
gate is a retention gate: `z == 1` selects old hidden state and `z == 0` selects the candidate.
These equations define this [API](../../../../glossary.md#term-application-programming-interface-api); no equivalence to a differently packed or differently biased
framework GRU is promised.

After prevalidation, implementation delegates exactly in this association and order:

```java
Tensor inputProjection = currentBias.isPresent()
        ? input.linear(currentInputWeight, currentBias.orElseThrow())
        : input.linear(currentInputWeight);
Tensor hiddenProjection = hidden.linear(currentHiddenWeight);

Tensor inputReset = inputProjection.sliceAxis(-1, 0L, hiddenSize);
Tensor inputUpdate = inputProjection.sliceAxis(-1, hiddenSize, twiceHiddenSize);
Tensor inputCandidate = inputProjection.sliceAxis(-1, twiceHiddenSize, packedHiddenSize);
Tensor hiddenReset = hiddenProjection.sliceAxis(-1, 0L, hiddenSize);
Tensor hiddenUpdate = hiddenProjection.sliceAxis(-1, hiddenSize, twiceHiddenSize);
Tensor hiddenCandidate = hiddenProjection.sliceAxis(-1, twiceHiddenSize, packedHiddenSize);

Tensor reset = inputReset.add(hiddenReset).sigmoid();
Tensor update = inputUpdate.add(hiddenUpdate).sigmoid();
Tensor candidate = inputCandidate.add(reset.mul(hiddenCandidate)).tanh();
return candidate.add(update.mul(hidden.sub(candidate)));
```

`twiceHiddenSize` and `packedHiddenSize` are validated exact positive `long` bounds derived from
the static [schema](../../../../glossary.md#term-schema); no expression helper constructs [`Operation`](../../../../glossary.md#operation), `TensorDescriptor`,
`TensorProvenance`, or a derived [Tensor](../../../../glossary.md#tensor) directly.

The no-bias successful chain creates exactly twenty fresh Tensors/IDs in order: input-[weight](../../../../glossary.md#term-weight)
PERMUTE, input [MATMUL](../../../../glossary.md#matrix-multiplication--matmul), hidden-weight PERMUTE, hidden MATMUL, six SLICE occurrences in the order
shown, reset ADD, reset [SIGMOID](../../../../glossary.md#term-sigmoid), update ADD, update SIGMOID, reset MUL, candidate ADD, candidate
[TANH](../../../../glossary.md#term-tanh), hidden-minus-candidate SUB, update MUL, and final ADD. The biased chain inserts one bias ADD
immediately after input MATMUL for twenty-one total. Tests must lock exact ordered [producer](../../../../glossary.md#term-producer)
references and operation association, not numerical execution.

## Replacement snapshots, mode, and state [ownership](../../../../glossary.md#term-ownership-meanings)

- Each forward call reads each stable parameter wrapper once in declaration order before local
  validation. Schema-compatible replacement before a call affects that call; replacement after
  an expression is built cannot change its exact provenance references.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [validation glossary](../../../../glossary.md#term-validation); [Schema glossary](../../../../glossary.md#term-schema); [provenance glossary](../../../../glossary.md#provenance))</span>
- There is no atomic multi-parameter snapshot or thread-safety guarantee. Callers coordinate
  replacement and forward when one consistent view matters.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [snapshot glossary](../../../../glossary.md#term-snapshot); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- `train()` and `eval()` may propagate inherited mode normally, but forward constructs the same
  formula in either mode and receives no `ForwardContext`.
- The cell registers no [`Buffer`](../../../../glossary.md#term-buffer) or child. Hidden input, gates, candidate, and result are caller-
  or expression-owned values and never module state.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>
- `GruCell` is not a `UnaryTensorModule` and cannot appear in `Sequential` by type.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.nn.layers` — owns concrete public [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) layers and recurrent cells.
- `io.github.pho001.synaptik.nn.module` — supplies direct [`Module`](../../../../glossary.md#term-module) [ownership](../../../../glossary.md#term-ownership-meanings) and stable
  [`Parameter`](../../../../glossary.md#term-parameter) wrappers without modification.
- `io.github.pho001.synaptik.nn.initialization` — supplies current eager parameter initializers
  without modification.<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation))</span>
- [Model](../../../../glossary.md#term-model) data-type, [Shape](../../../../glossary.md#shape), and [Tensor](../../../../glossary.md#tensor) packages — supply existing semantics only.

Packages added or changed:

- No package is added. The existing `nn.layers` package gains `GruCell` and package documentation
  is extended for its explicit-state contract.

Type placement:

- `io.github.pho001.synaptik.nn.layers.GruCell` — owns one parameterized GRU step and its exact
  explicit-state public contract.<span data-glossary-navigation="true"> ([GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru))</span>
- `io.github.pho001.synaptik.nn.layers.GruCellTest` — owns public surface, supplied state,
  forward/preflight, provenance, replacement, mode, discovery, and exclusion coverage.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [provenance glossary](../../../../glossary.md#provenance))</span>
- `io.github.pho001.synaptik.nn.layers.GruCellInitializationTest` — owns initialized constructor,
  source order/bounds, metadata, early-failure, and non-rollback coverage.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [rollback glossary](../../../../glossary.md#term-rollback))</span>

## Affected files

Expected production files:

- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/GruCell.java`.
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/package-info.java`.

Expected test files:

- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/GruCellTest.java`.
- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/GruCellInitializationTest.java`.

Expected documentation and [planning](../../../../glossary.md#term-planning) files:

- `docs/glossary.md`.
- `docs/planning/extensions/nn/master-plan.md`.
- this task specification.

No other file may change for this task. In particular, no [Model](../../../../glossary.md#term-model), Training, architecture, [Gradle](../../../../glossary.md#term-gradle),
[dependency](../../../../glossary.md#term-dependency)-test, sequence, [LSTM](../../../../glossary.md#term-long-short-term-memory-lstm), [CPU](../../../../glossary.md#term-cpu), global-roadmap, or concurrent-work path is authorized.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify exactly the seven listed paths and at most:

- two production Java files;
- two [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) test files; and
- three documentation/[planning](../../../../glossary.md#term-planning) files.

If implementation needs another public type, helper file, test [owner](../../../../glossary.md#term-owner), [Model](../../../../glossary.md#term-model) [API](../../../../glossary.md#term-application-programming-interface-api), [dependency](../../../../glossary.md#term-dependency),
architecture change, or eighth path, stop and propose a separate follow-up task.

## Acceptance criteria

- `GruCell` is final, extends [`Module`](../../../../glossary.md#term-module) directly, and exposes exactly the seven declared public
  members with no additional public/protected surface or nested type.
- Supplied constructors retain exact caller Tensors and fully prevalidate the packed schema before
  declaration. Initialized construction uses exact checked packed sizes, state order, Glorot
  calls, optional zero bias, source lifecycle, and failure side effects specified above.<span data-glossary-navigation="true"> ([Tensors glossary](../../../../glossary.md#tensor); [schema glossary](../../../../glossary.md#term-schema); [Glorot glossary](../../../../glossary.md#term-glorot-uniform); [bias glossary](../../../../glossary.md#term-bias); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Parameter names, order, Shapes, exact type, gradient eligibility, accessors, recursive paths,
  state-dictionary entries, and absence of buffers/children are exact.<span data-glossary-navigation="true"> ([Parameter glossary](../../../../glossary.md#term-parameter); [Shapes glossary](../../../../glossary.md#shape); [gradient glossary](../../../../glossary.md#term-gradient); [buffers glossary](../../../../glossary.md#term-buffer))</span>
- Gate order is reset, update, candidate. Bias is optional, packed, and input-side only. The
  candidate uses reset-after projection and the output uses retention interpolation exactly as
  specified. Tests use provenance to distinguish these choices from plausible alternatives.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- Forward rejects every locally knowable invalid request before the first Tensor expression,
  snapshots each current binding once, creates exactly the twenty- or twenty-one-Tensor chain in
  documented order, and returns the exact final ADD Tensor as next hidden state.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- Valid rank-one and higher inputs use ordinary conservative right-aligned leading broadcasting.
  Final feature checks, mixed floating promotion, output Shape/type, gate slice intervals, and
  failure order match current Model semantics.<span data-glossary-navigation="true"> ([rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [broadcasting glossary](../../../../glossary.md#broadcasting); [feature glossary](../../../../glossary.md#term-feature); [slice glossary](../../../../glossary.md#slice); [Model glossary](../../../../glossary.md#term-model))</span>
- Replacement affects later calls only; existing expressions retain old exact parameter
  references. Mode changes do not affect the expression chain.
- No hidden [recurrent state](../../../../glossary.md#backpropagation-through-time-and-recurrent-state), result record, [unary](../../../../glossary.md#term-unary) adapter, `Sequential` change, context, RNG state,
  sequence/time behavior, scan, Model change, optimizer, execution behavior, or framework-
  compatibility claim is introduced.<span data-glossary-navigation="true"> ([RNG state glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- Focused tests cover exact reflection surface, state schema/order, supplied validation, initialized
  draw/ID order, early failures, all gate intervals, exact operation/provenance association,
  broadcasting/type/rank/feature failures, replacement snapshots, mode independence, state
  discovery, and all explicit exclusions.<span data-glossary-navigation="true"> ([feature glossary](../../../../glossary.md#term-feature))</span><span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [operation glossary](../../../../glossary.md#operation); [rank glossary](../../../../glossary.md#term-rank))</span>
- All new and affected public/package APIs have meaningful complete Javadoc for purpose, equations,
  packing, Shapes/types, ownership, nullability, source lifecycle, side effects, promotion,
  broadcasting, result semantics, failure order, concurrency, mode, and non-execution boundaries.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [nullability glossary](../../../../glossary.md#term-nullability); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- A separate documentation-focused clean context finalizes affected Javadoc, package docs,
  glossary impact, planning evidence, no-change conclusions, links, and generated Javadoc in the
  same overall change.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation pass runs focused tests while developing and, after executable Java stabilizes:

```bash
./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.layers.GruCellTest --tests io.github.pho001.synaptik.nn.layers.GruCellInitializationTest
./gradlew :extensions:nn:test
```

The second command is the sole authoritative final affected-[module](../../../../glossary.md#term-module) Java validation. It covers the
existing [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) suite plus the new cell. [Model](../../../../glossary.md#term-model) tests are not repeated: current exhaustive tests remain
authoritative for primitive [linear](../../../../glossary.md#term-linear-meanings), [slice](../../../../glossary.md#slice), ADD, SUB, MUL, [SIGMOID](../../../../glossary.md#term-sigmoid), [TANH](../../../../glossary.md#term-tanh), promotion, and broadcast
semantics, while focused NN tests lock cell composition and [preflight](../../../../glossary.md#term-preflight).

Documentation pass runs after final [Javadoc](../../../../glossary.md#term-javadoc) edits:

```bash
./gradlew :extensions:nn:javadoc
git diff --check
git diff --no-index --check /dev/null extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/GruCell.java
git diff --no-index --check /dev/null extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/GruCellTest.java
git diff --no-index --check /dev/null extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/GruCellInitializationTest.java
```

The documentation pass also validates generated `GruCell` and package pages; exact surface and
private fields; Model/NN/[JDK](../../../../glossary.md#term-java-development-kit-jdk)-only imports; the unchanged sole [NN Model](../../../../glossary.md#model-module-buffer-and-scan-distinctions) [dependency](../../../../glossary.md#term-dependency); Markdown links,
anchors, fences, final newlines, and trailing whitespace; exact seven-path [scope](../../../../glossary.md#term-scope); task/master
status [synchronization](../../../../glossary.md#term-synchronization); exactly one In progress NN row/spec during the pass; and absence of task
files for 0014–0015. The documentation pass changes that synchronized status to Complete only
after all required final gates pass.
It reuses successful implementation tests unless executable Java changes afterward or it records
a concrete reason to rerun them.

Repository-wide, architecture, Model, Training, [CPU](../../../../glossary.md#term-cpu), [compiler](../../../../glossary.md#term-compiler), [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), and integration
tests are deferred to the recurrent NN milestone [checkpoint](../../../../glossary.md#term-checkpoint) or [CI](../../../../glossary.md#term-continuous-integration-ci). This task changes one Model-
only extension [API](../../../../glossary.md#term-application-programming-interface-api) and no dependency, architecture boundary, shared build, or execution contract.

## Dependencies

- NN tasks 0001–0012 are Complete and provide direct Module ownership, stable schema-validated
  parameters, explicit-source initialization, deterministic discovery/state dictionaries,
  unary-composition exclusions, and the recurrent validation/state-threading precedent.<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [schema glossary](../../../../glossary.md#term-schema); [parameters glossary](../../../../glossary.md#term-parameter); [unary glossary](../../../../glossary.md#term-unary); [validation glossary](../../../../glossary.md#term-validation))</span>
- [Model](../../../../glossary.md#term-model) `Tensor.linear`, `sliceAxis`, `add`, `sub`, `mul`, [`sigmoid`](../../../../glossary.md#term-sigmoid), and [`tanh`](../../../../glossary.md#term-tanh), same-category
  promotion, `ShapeBroadcast`, [immutable](../../../../glossary.md#term-immutable) [descriptors](../../../../glossary.md#term-descriptor)/[provenance](../../../../glossary.md#provenance), and ID allocation are Complete.
- Current independent slice expressions truthfully retain exact packed projection provenance;
  no shared split API is required. Concat is not used because initialized parameters must remain
  direct eager leaves and gate packing is one schema, not a runtime composition.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [eager glossary](../../../../glossary.md#term-eager-evaluation); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- `ParameterInitializers.glorotUniform` and `zeros` provide the complete initialized policy.
- [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0007 and existing [dependency](../../../../glossary.md#term-dependency) tests already permit Model-only [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) composition.
- Compiler, Training, CPU, Runtime, Prepare, Engine, backend, and numerical execution support are
  not prerequisites because this task constructs parameter leaves and expression metadata only.<span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [CPU glossary](../../../../glossary.md#term-cpu); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [metadata glossary](../../../../glossary.md#term-metadata))</span>

## Follow-up tasks

- NN 0014 remains Draft: add an LSTM cell only after this task is Complete. It must accept and
  return both hidden and cell state explicitly and must not store either in a Buffer or hidden
  field.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm); [Buffer glossary](../../../../glossary.md#term-buffer))</span>
- NN 0015 remains Draft: decide static unrolling of concrete cell calls versus a genuinely
  general recurrent Model scan. <span data-glossary-navigation="true"> ([unrolling glossary](../../../../glossary.md#term-unrolling); [Model glossary](../../../../glossary.md#term-model))</span>`CUM_SUM` and `CUM_PROD` do not satisfy that need.
- If NN 0015 selects containers, prefer cell-specific types unless the completed RNN/GRU/LSTM
  signatures prove a shared recurrent contract without casts, state erasure, duplicate carriers,
  or hidden state.<span data-glossary-navigation="true"> ([RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [signatures glossary](../../../../glossary.md#term-signature))</span>
- Alternate GRU packing, bias, reset-before, migration, and framework-conversion policies require
  a concrete consumer and separate task; they must not be added as aliases here.<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Truncation, masking, variable lengths, bidirectionality, stacking, recurrent dropout, checkpoint
  persistence, training sessions, and backend execution remain separate future work.<span data-glossary-navigation="true"> ([dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [persistence glossary](../../../../glossary.md#term-persistence); [backend glossary](../../../../glossary.md#backend))</span>

## Documentation and no-change review

Document profiles:

- Java/package [Javadoc](../../../../glossary.md#term-javadoc): General plus [API](../../../../glossary.md#term-application-programming-interface-api)/Javadoc.
- glossary: General reference style.
- task/master plan: General plus [Planning](../../../../glossary.md#term-planning).

Required implementation-phase documentation changes are `GruCell` and layers-package Javadocs,
the existing glossary's explicit-state recurrent wording, and synchronized [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) planning records.

The separate documentation pass must verify and record these reasoned no-change conclusions:

- `ARCHITECTURE.md`, focused architecture pages, ADR 0007, and architecture tests remain accurate
  because state ownership, module direction, and lifecycle boundaries do not change.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Tensor and Compile APIs plus Model source/master/capabilities remain accurate because the cell
  composes existing expressions and adds no Tensor method, semantic kind, split, recurrent scan,
  capture, derivative, or execution behavior.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [Model glossary](../../../../glossary.md#term-model); [capture glossary](../../../../glossary.md#term-capture-meanings); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- Training API and training graph remain accurate because recursive parameter discovery and
  replacement consume this direct Module normally; no optimizer, session, gradient publication,
  truncated history, or state orchestration is added.<span data-glossary-navigation="true"> ([training graph glossary](../../../../glossary.md#training-graph); [parameter glossary](../../../../glossary.md#term-parameter); [optimizer glossary](../../../../glossary.md#term-optimizer); [gradient glossary](../../../../glossary.md#term-gradient); [publication glossary](../../../../glossary.md#term-publication))</span>
- `RnnCell`, `UnaryTensorModule`, `Sequential`, state-dictionary contracts, and their tests remain
  accurate. The GRU follows RNN explicit hidden threading and is independently excluded from
  unary composition; transient gates and hidden values are not module state.<span data-glossary-navigation="true"> ([GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [unary glossary](../../../../glossary.md#term-unary))</span>
- [Gradle](../../../../glossary.md#term-gradle) and [dependency](../../../../glossary.md#term-dependency) rules/tests remain accurate because `extensions/nn` retains only its
  existing Model dependency.
- Backend conformance and integration tests remain unnecessary because no numerical execution,
  backend capability, or end-to-end lifecycle changes.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [backend capability glossary](../../../../glossary.md#backend-capability))</span>
- The global roadmap, [CPU](../../../../glossary.md#term-cpu) work, other [modules](../../../../glossary.md#term-module), and Draft NN tasks 0014–0015 remain untouched.

## Architecture impact

Expected impact: None.

This task realizes the existing [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) responsibility for [parameter](../../../../glossary.md#term-parameter)-owning layer composition while
keeping [recurrent state](../../../../glossary.md#backpropagation-through-time-and-recurrent-state) explicit. If implementation requires hidden [module](../../../../glossary.md#term-module)/[runtime](../../../../glossary.md#term-runtime) state, a
universal recurrent abstraction, a [general scan](../../../../glossary.md#general-differentiable-scan), another module [dependency](../../../../glossary.md#term-dependency), or an architecture
rule, stop and report the conflict rather than editing architecture within this task.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are the clean-context implementation agent for Synaptik NN task 0013. Work in the existing
shared worktree. Do not use GSD. Do not commit or push. Preserve every unrelated/concurrent CPU
source, test, documentation, master-plan, task, roadmap, and glossary change exactly.

Read root AGENTS.md, ARCHITECTURE.md, the current architecture plan, planning guide/roadmap,
documentation rules and General/API-Javadoc/Planning profiles, NN master plan and tasks 0001–0013,
ADR 0007, final Module/Parameter/initializers/UnaryTensorModule/Sequential/RnnCell and layer APIs/
tests/Javadocs, Model master and final Tensor linear/add/sub/mul/sigmoid/tanh/slice/concat/Shape/
broadcast APIs/tests/Javadocs, Operation attributes, Training API/graph, glossary, and dependency/
build rules in full.

Implement exactly the final direct-Module GruCell public API, packed state schema, supplied and
initialized validation, explicit caller-threaded hidden-state contract, full pre-expression
forward preflight, reset-update-candidate slices, fixed reset-after equations, provenance order,
replacement snapshots, mode independence, and Sequential exclusion in the seven authorized paths.
Add no result record, hidden Buffer/state, UnaryTensorModule, adapter, ForwardContext,
GraphRngState, sequence loop, time traversal, recurrent scan, Model helper, optimizer/session,
backend/execution behavior, or eighth path.

Run the focused GruCell tests and one authoritative NN module test after executable Java
stabilizes. Then hand the unchanged executable diff and exact evidence to a distinct clean
documentation-focused context. That context must independently finalize Javadocs, package docs,
glossary, planning evidence, no-change conclusions, generated Javadoc, Markdown, surface, scope,
status, newline, and whitespace gates without repeating successful Java tests unless executable
behavior changes or a concrete risk is recorded.

If current Model APIs cannot express the exact formula and full local preflight, if checked packed
initialization cannot precede the first draw, or if explicit type-safe state needs another public
type or dependency, stop and report the exact blocker. Mark Complete only after implementation,
documentation, and every required validation passes.
```

## Documentation-agent [handoff](../../../../glossary.md#term-handoff)

After executable Java/tests stabilize, give the clean documentation context:

- this task and exact seven-path limit;
- the final executable diff and exact focused/final [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) commands, counts, and results;
- the exact public surface, direct-Module/Sequential exclusion, packed state names/order/Shapes/
  types, gate order, bias association, constructor prevalidation, and draw/ID order;<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module); [Shapes glossary](../../../../glossary.md#shape); [bias glossary](../../../../glossary.md#term-bias))</span>
- full forward preflight, batch-broadcast examples, reset-after equations, twenty-/twenty-one-ID
  producer chains, next-hidden semantics, mode behavior, replacement snapshots, and failures;<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [producer glossary](../../../../glossary.md#term-producer))</span>
- directly relevant architecture/ADR, documentation profiles, final NN/Model source and tests,
  Tensor/Training APIs, glossary, dependency tests, and planning history;<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor); [dependency glossary](../../../../glossary.md#term-dependency); [planning glossary](../../../../glossary.md#term-planning))</span>
- the mandate to preserve concurrent glossary work and record every reasoned no-change conclusion;
- generated-[Javadoc](../../../../glossary.md#term-javadoc), reflection/`javap`, import/dependency, Markdown, exact-scope/status,
  task-0014–0015-absence, newline, and whitespace gates; and<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- the required completion-summary and Status format from `AGENTS.md`.

## Local decisions

- Use `GruCell`, following the project's acronym style (`RnnCell`, `Rng`), and place it beside
  concrete layers. The name represents one cell step, not sequence traversal.
- Extend [`Module`](../../../../glossary.md#term-module) directly and return one Tensor. Two Tensor inputs are intrinsic, and output and
  next hidden are one exact value; a unary adapter or duplicate result carrier would obscure the
  contract.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [unary glossary](../../../../glossary.md#term-unary))</span>
- Pack two matrices and optional bias in reset, update, candidate order. Independent slices make
  gate provenance exact with the current Model API and keep the public state/accessor surface as
  small as <span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [provenance glossary](../../../../glossary.md#provenance); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>`RnnCell`. Six public per-gate [matrices](../../../../glossary.md#term-matrix-meanings) would enlarge state without a current consumer.
- Use one optional packed input-side bias. This is the smallest unambiguous bias policy supported
  by one biased packed linear call. There is no recurrent bias hidden inside reset placement.<span data-glossary-navigation="true"> ([linear glossary](../../../../glossary.md#term-linear-meanings))</span>
- Fix reset-after semantics: reset multiplies the already projected recurrent candidate lane.
  Reset-before would require a different projection association and is not an alias.<span data-glossary-navigation="true"> ([lane glossary](../../../../glossary.md#term-lane); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Fix update as a retention gate through `candidate + update * (hidden - candidate)`. This avoids
  inventing a scalar-one Tensor while making the exact operation order and endpoint meanings clear.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [operation glossary](../../../../glossary.md#operation))</span>
- Use Glorot uniform independently for the actual packed matrix Shapes. Concatenating per-gate
  eager leaves would make a derived expression the parameter and add unnecessary state/provenance;
  no orthogonal initializer exists.<span data-glossary-navigation="true"> ([Glorot glossary](../../../../glossary.md#term-glorot-uniform); [Shapes glossary](../../../../glossary.md#shape); [eager glossary](../../../../glossary.md#term-eager-evaluation); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- Permit current [Model](../../../../glossary.md#term-model) mixed floating promotion and conservative leading [broadcasting](../../../../glossary.md#broadcasting), as
  `RnnCell` does. Preflight duplicates only local public algebra needed to prevent a late gate or
  interpolation failure from leaving a projection prefix.<span data-glossary-navigation="true"> ([Preflight glossary](../../../../glossary.md#term-preflight))</span>
- Accept unresolved final feature equality exactly as Model linear does. The fixed parameter
  output and slice bounds still make every next-hidden final axis statically <span data-glossary-navigation="true"> ([feature glossary](../../../../glossary.md#term-feature); [slice glossary](../../../../glossary.md#slice); [axis glossary](../../../../glossary.md#term-axis))</span>`hiddenSize`.
- Keep all leading axes semantically neutral batch coordinates. Task 0015, not this cell, owns any
  time-axis contract and repeated invocation.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>

## Known limitations

- Only this reset-after, reset/update/candidate-packed GRU with positive fully static parameters
  and one optional input-side packed bias is supported.<span data-glossary-navigation="true"> ([GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [parameters glossary](../../../../glossary.md#term-parameter); [bias glossary](../../../../glossary.md#term-bias))</span>
- Input and hidden Shapes may contain unresolved leading or final Dimensions only when current
  local Model rules can represent the projection and broadcast obligations. NN performs no
  binding.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [Dimensions glossary](../../../../glossary.md#dimension); [Model glossary](../../../../glossary.md#term-model); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [binding glossary](../../../../glossary.md#term-binding))</span>
- Higher-[rank](../../../../glossary.md#term-rank) inputs are one batched cell application, never an implicit sequence.
- Explicit caller-threaded recurrence may build deep provenance chains. This task adds no
  detachment, truncation, loop IR, scheduling, or memory-lifetime policy.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- Parameter replacement and forward are not thread-safe as one snapshot. The cell retains no
  lock, version, or transaction.<span data-glossary-navigation="true"> ([thread-safe glossary](../../../../glossary.md#term-thread-safe); [snapshot glossary](../../../../glossary.md#term-snapshot); [transaction glossary](../../../../glossary.md#term-transaction))</span>
- Initialized eager parameters require Java-array-sized host leaves and may consume draws or IDs
  before a later resource failure; effects are not rolled back.<span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [host glossary](../../../../glossary.md#term-host))</span>
- Forward constructs metadata and proves no numerical values, gradient rules, compiler capture,
  backend support, storage allocation, publication, or execution.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [backend glossary](../../../../glossary.md#backend); [publication glossary](../../../../glossary.md#term-publication))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Clean [planning](../../../../glossary.md#term-planning) context `/root/nn_0013_planning` read the repository instructions, architecture
  contract and current plan, planning guide/roadmap, documentation rules and General/Planning/API-
  Javadoc profiles, NN master and complete task history through 0012, final recurrent/module/
  initializer APIs and tests, Model master and relevant Tensor/Shape/operation contracts, Training
  API/graph, glossary, and build/dependency rules before selecting this contract.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor); [Shape glossary](../../../../glossary.md#shape); [operation glossary](../../../../glossary.md#operation); [graph glossary](../../../../glossary.md#graph); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Planning inspection confirmed the current Model surface can express the complete formula with
  packed linear projections and exact independent <span data-glossary-navigation="true"> ([linear glossary](../../../../glossary.md#term-linear-meanings))</span>`sliceAxis` provenance. No Model split, concat,
  recurrent scan, semantic kind, result carrier, or new initializer is required.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- Planning selected reset/update/candidate packing, one input-side packed bias, reset-after
  candidate association, and retention-gate interpolation explicitly; no framework-compatibility
  equivalence is claimed.<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias))</span>
- Targeted planning validation passed for the two changed planning paths: local links and anchors
  resolve, fences balance, final newlines are present, and trailing whitespace is absent.
- The NN planning state initially had exactly one Ready row/spec (0013); implementation has moved
  that same sole frontier to In progress. Tasks 0014–0015 remain concise Draft rows with no task
  files. Every unrelated concurrent CPU/source/test/documentation/master/task/roadmap/glossary
  change remains untouched.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Clean implementation context `/root/nn_0013_implementation` added the direct final `GruCell`,
  focused state/forward and initialization tests, and draft type/package Javadocs. The production
  implementation has exactly the planned three packed parameter fields and seven-member public
  surface, snapshots current bindings in declaration order, prevalidates the complete local
  request before expression creation, then delegates the exact twenty-/twenty-one-ID reset-after
  expression chain in reset, update, candidate slice order.<span data-glossary-navigation="true"> ([bindings glossary](../../../../glossary.md#term-binding); [slice glossary](../../../../glossary.md#slice))</span>
- The focused two-suite selection passed 14 tests. The sole authoritative final affected-module
  Java run, <span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>`./gradlew :extensions:nn:test`, passed 21 suites and 139 tests with zero failures,
  errors, or skips; executable Java did not change afterward. Preliminary generated Javadoc also
  passed.
- Public/private `javap`, reflection coverage, import and existing sole-dependency inspection, and
  forbidden-state scans confirmed the exact direct-Module surface, only three private packed
  parameter fields, Model/NN/JDK production imports, unchanged Model-only NN dependency, and no
  hidden Tensor/Buffer/context/RNG/sequence/scan/execution state.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [Buffer glossary](../../../../glossary.md#term-buffer); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- Independent clean documentation context `/root/nn_0013_docs` read the final implementation and
  tests, architecture and planning contracts, documentation rules and General/API-Javadoc/
  Planning profiles, NN task lineage through 0012, Module/Parameter/initializer/unary/RNN
  boundaries, relevant Model Tensor/Shape/type/provenance contracts, Training API/graph, glossary,
  and build/dependency rules. It found no executable defect, architecture uncertainty, scope
  blocker, or reason to change executable Java or tests.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [scope glossary](../../../../glossary.md#term-scope))</span>
- The documentation context finalized `GruCell` and layers-package Javadocs. The rendered contract
  now states the reset/update/candidate packed schemas, input-side-only bias, six independent
  slices, reset-after equations, retention-gate endpoints, explicit hidden ownership, complete
  preflight, binding snapshots, caller-owned initializer source/order, mode independence,<span data-glossary-navigation="true"> ([schemas glossary](../../../../glossary.md#term-schema); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [preflight glossary](../../../../glossary.md#term-preflight))</span>
  `Sequential` exclusion, and non-execution boundary. `./gradlew :extensions:nn:javadoc` passed
  after those edits (`BUILD SUCCESSFUL`; three actionable tasks, two executed and one up-to-date),
  and generated `GruCell.html` plus `layers/package-summary.html` inspection found the expected
  links, signatures, parameters, returns, failures, formulas, and boundaries.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>
- `javap -public` and `javap -private` confirmed final direct [`Module`](../../../../glossary.md#term-module), exactly three public
  constructors and four declared public methods, no protected member or nested type, and only
  private final [`Parameter`](../../../../glossary.md#term-parameter), `Parameter`, and `Optional<Parameter>` fields. An exploratory JShell
  reflection inspection printed the same successful contract checks but exited nonzero while
  persisting tool history because the environment rejected Java Preferences synchronization; a
  standalone reflection program then compiled and ran successfully and is the authoritative
  reflection result.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- The targeted Markdown validator passed the three task-owned Markdown files with 353 links
  inspected, 295 local anchors resolved, 40 balanced fence markers, final newlines, and no trailing
  whitespace. Source/import and build inspection found only Model, existing NN, and JDK production
  imports and retained the sole NN dependency `implementation(project(":modules:model"))`.
- The validator's first local-anchor run reported false missing-anchor results because its
  temporary GitHub-heading slug approximation removed punctuation before converting spaces; the
  corrected approximation converted spaces first and the exact same documents passed. An initial
  no-index loop likewise used zsh's special `path` variable, which removed `git` from command
  lookup after the first iteration and exited 127; the rerun used a task-specific variable and all
  four new-file checks passed. Neither exploratory failure changed a repository file.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup))</span>
- Exact-scope inspection found the seven authorized NN 0013 paths amid preserved unrelated CPU
  source/master/task/roadmap work. New-file no-index whitespace checks returned only the expected
  content-difference status for the task, production class, and two tests; whole-worktree
  `git diff --check` passed. NN tasks 0001–0013 and the master row now read Complete. Tasks
  0014–0015 remain Draft rows without task specifications and no NN task is Ready or In progress.
- The documentation pass reused the implementation context's focused two-suite/14-test result and
  authoritative 21-suite/139-test NN module result because no executable Java or test changed
  afterward. Javadoc/prose-only edits do not make that evidence stale, so no Java test suite was
  repeated.
- `ARCHITECTURE.md`, focused architecture pages, ADR 0007, and architecture tests require no
  change because state remains caller-owned and NN retains its existing Model-only direction.
  Tensor/Compile APIs, Model source/master/capabilities, and operation contracts require no change
  because the cell composes existing linear, independent slice, ADD, SUB, MUL, SIGMOID, and TANH
  expressions without a new semantic kind, scan, capture, derivative, or execution contract.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Compile glossary](../../../../glossary.md#compile); [SIGMOID glossary](../../../../glossary.md#term-sigmoid); [TANH glossary](../../../../glossary.md#term-tanh); [capture glossary](../../../../glossary.md#term-capture-meanings); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- Training API and the training graph require no GRU-specific change because ordinary recursive
  parameter discovery/replacement already consumes a direct Module and the task adds no optimizer,
  session, gradient publication, truncated history, or state orchestration. <span data-glossary-navigation="true"> ([training graph glossary](../../../../glossary.md#training-graph); [GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [optimizer glossary](../../../../glossary.md#term-optimizer); [gradient glossary](../../../../glossary.md#term-gradient); [publication glossary](../../../../glossary.md#term-publication))</span>`RnnCell`,
  `UnaryTensorModule`, `Sequential`, state-dictionary contracts, and their tests remain accurate:
  the GRU is separately excluded and its transient gates and hidden values never become module
  state.
- Gradle/dependency rules and tests, backend conformance, and integration tests require no change
  because no dependency, numerical execution, backend capability, or end-to-end lifecycle changed.
  The global roadmap, concurrent CPU work, other modules, and Draft NN 0014–0015 work remain
  untouched.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [backend capability glossary](../../../../glossary.md#backend-capability); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## Implementation notes

- Added final `GruCell extends Module` directly. It owns stable `inputWeight`, `hiddenWeight`, and
  optional input-side [`bias`](../../../../glossary.md#term-bias) wrappers with exact packed reset/update/candidate schemas and retains
  no hidden Tensor, Buffer, child, context, RNG, or result state.<span data-glossary-navigation="true"> ([schemas glossary](../../../../glossary.md#term-schema); [Tensor glossary](../../../../glossary.md#tensor); [Buffer glossary](../../../../glossary.md#term-buffer); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- Supplied constructors validate the complete exact schema before declaration. Initialized
  construction validates every requested packed count before draws, calls Glorot uniform for
  input then hidden weights from the same caller-owned source, and creates optional zero bias
  afterward without a draw.<span data-glossary-navigation="true"> ([Glorot glossary](../../../../glossary.md#term-glorot-uniform); [weights glossary](../../../../glossary.md#term-weight))</span>
- Forward rejects every locally knowable invalid request before the first expression ID, derives
  independent final-axis gate slices, and constructs exactly<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>
  `n + z * (hidden - n)` with `n = tanh(x_n + r * h_n)`. Exact provenance tests distinguish the
  input-only bias, reset-after candidate, retention interpolation, and fixed call order.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- Tests cover exact public/finality/superclass and exclusion surface, packed registration and
  recursive discovery, constructor validation and failure side effects, initializer bounds and
  provenance, forward preflight, all gate slices and operation associations, mixed floating and
  leading broadcasting, mode independence, replacement snapshots, repeated explicit state
  threading, absence of retained state, and null/failure behavior.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [preflight glossary](../../../../glossary.md#term-preflight); [operation glossary](../../../../glossary.md#operation); [broadcasting glossary](../../../../glossary.md#broadcasting); [state threading glossary](../../../../glossary.md#state-threading))</span>
- No Model, Training, architecture, Gradle, dependency, compiler, runtime, backend, conformance,
  integration, global-roadmap, CPU, LSTM, or sequence file changed. Current contracts remain
  sufficient because this task only composes existing Model metadata expressions.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [CPU glossary](../../../../glossary.md#term-cpu); [LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Independent documentation context `/root/nn_0013_docs` finalized the public/package Javadocs,
  added the glossary's packed schema and one-step gate example, distinguished the GRU from vanilla
  RNN, future LSTM, recurrent scan, cumulative scan, and unary <span data-glossary-navigation="true"> ([GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [cumulative scan glossary](../../../../glossary.md#cumulative-scan); [unary glossary](../../../../glossary.md#term-unary))</span>`Sequential` composition, and
  synchronized all final evidence without changing executable Java or tests.

## Completion summary

- Completed changes: added final direct-[Module](../../../../glossary.md#term-module) `GruCell` with exact packed state, supplied and
  explicit-source construction, complete forward preflight, reset-after gate composition, sole
  next-hidden Tensor result, and focused contract tests; finalized all affected documentation.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Files changed or created: exactly `GruCell.java`, layers `package-info.java`,
  `GruCellTest.java`, `GruCellInitializationTest.java`, `docs/glossary.md`, the NN master plan, and
  this task; unrelated concurrent paths were preserved.
- Tests and validation: reused the passing focused 2-suite/14-test and authoritative
  21-suite/139-test NN evidence with zero failures, errors, or skips. Final NN Javadoc/generated-
  page inspection, <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`javap`, independent reflection, import/dependency, Markdown, exact-scope/
  status, newline/whitespace, no-index, and whole-diff checks passed.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: clean context `/root/nn_0013_docs` completed the mandatory
  independent source/test/API/architecture review and changed no executable Java or test.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Documentation impact: finalized the GRU and layers-package Javadocs, added the glossary's packed
  schema and explicit one-step example plus recurrent-boundary distinctions, and synchronized
  task/master status and evidence.<span data-glossary-navigation="true"> ([GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [schema glossary](../../../../glossary.md#term-schema))</span>
- Javadoc review: complete; generated type and package pages contain the exact surface, state,
  equation, failure, ownership, concurrency, mode, and non-execution contracts.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- Glossary impact: added the reusable GRU term, fixed packed checkpoint schema, reset-after gate
  example, and distinctions from vanilla RNN, future LSTM, recurrent/cumulative scan, and<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm); [cumulative scan glossary](../../../../glossary.md#cumulative-scan))</span>
  `Sequential`.
- Unresolved issues: None.
- Follow-up required: None for task 0013; [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0014–0015 remain separate Draft work.

Status: Complete

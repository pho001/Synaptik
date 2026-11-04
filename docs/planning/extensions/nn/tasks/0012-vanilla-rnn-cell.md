# Task 0012: Vanilla [Tanh](../../../../glossary.md#term-tanh) [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn) Cell

> **Editorial reading context — not a status update.** An application calls the cell with
> input and previous hidden state. The cell owns projection parameters but retains neither
> hidden state nor execution history; its returned Tensor describes the next state for the
> application to thread onward. One higher-rank call is not an implicit time loop. The current
> [recurrent layer contract](../../../../reference/nn/recurrent-layers.md#current-nn-recurrent-composition-contract)
> contrasts cells with sequence containers.
>
> **Mathematical reading example.** For finite real scalar input $x$, hidden state $h$,
> input weight $w$, hidden weight $u$ and bias $b$, one transition is
> $h'=\tanh(wx+b+uh)$. Taking $x=1,h=0,w=1,u=2,b=0$ gives $h'=\tanh(1)$,
> about $0.762$. The cell returns a description of that transition, not this evaluated number;
> actual arithmetic remains governed by the recorded Model operations.

## Status

Complete

## Goal

Add the smallest truthful recurrent neural-network capability: one final public vanilla [tanh](../../../../glossary.md#term-tanh)
`RnnCell` whose hidden state is an explicit caller-supplied [Tensor](../../../../glossary.md#tensor) and whose returned Tensor is the
next hidden state. The cell owns only trainable projection [parameters](../../../../glossary.md#term-parameter). It retains no hidden state
between calls and composes the complete result through existing [Model](../../../../glossary.md#term-model) [`linear`](../../../../glossary.md#term-linear-meanings), ordinary `add`,
and `tanh` expressions.

[Mental model](../../../../glossary.md#term-mental-model):

```text
caller input x + caller hidden h + current cell parameters
  -> input.linear(inputWeight, optional bias)
  -> hidden.linear(hiddenWeight)
  -> add both projections
  -> tanh
  -> one Tensor h' returned to the caller
  -> caller explicitly threads h' into a later call when recurrence is intended
```

For a vanilla [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn) cell, the visible output and next hidden state are the same Tensor. The [API](../../../../glossary.md#term-application-programming-interface-api)
therefore returns that Tensor once rather than introducing a result record with two fields that
would retain identical references.

## [Scope](../../../../glossary.md#term-scope)

- Add final public `io.github.pho001.synaptik.nn.layers.RnnCell` extending
  `io.github.pho001.synaptik.nn.module.Module` directly.
- Add exactly the constructors, parameter accessors, and two-input forward method in the public
  API table below. Add no overload, builder, options object, activation argument, state carrier,
  size getter, or functional facade.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [argument glossary](../../../../glossary.md#term-argument); [facade glossary](../../../../glossary.md#term-facade))</span>
- Declare positive fully static [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [parameters](../../../../glossary.md#term-parameter) under exact local names `inputWeight` then
  `hiddenWeight`. Their [Shapes](../../../../glossary.md#shape) are respectively `[hiddenSize, inputSize]` and
  `[hiddenSize, hiddenSize]`.
- Optionally declare one positive fully static [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) parameter under exact local name [`bias`](../../../../glossary.md#term-bias),
  with Shape `[hiddenSize]`. It is one shared affine bias, not separate input and recurrent biases.
- Require all declared parameters to have one exact floating [data type](../../../../glossary.md#data-type--datatype) and
  `requiresGrad == true`. Retain caller-supplied parameter [Tensor](../../../../glossary.md#tensor) references exactly.
- Provide supplied-state construction both without and with bias. Null never means absent bias;
  callers select absence through the two-Tensor constructor.
- Provide one initialized constructor with explicit positive `inputSize`, positive `hiddenSize`,
  bias presence, floating [`DataType`](../../../../glossary.md#data-type--datatype), and caller-owned `RandomGenerator`.
- Initialize `inputWeight` then `hiddenWeight` through exact
  `ParameterInitializers.glorotUniform` calls using the same explicit source. When requested,
  initialize bias afterward through exact `ParameterInitializers.zeros`; bias consumes no random
  draw. The source is never retained.
- Add exactly `Tensor forward(Tensor input, Tensor hidden)`. Both semantic inputs are explicit on
  every call; the cell never supplies, caches, updates, registers, or discovers a hidden state.
- Fix [activation](../../../../glossary.md#term-activation) to existing `Tensor.tanh()`. This is the vanilla tanh cell, not a configurable
  activation wrapper.<span data-glossary-navigation="true"> ([tanh glossary](../../../../glossary.md#term-tanh))</span>
- Support input and hidden [rank](../../../../glossary.md#term-rank) one or higher. The final input [Dimension](../../../../glossary.md#dimension) contracts with
  `inputWeight` input size, and the final hidden Dimension contracts with `hiddenWeight` hidden
  size under the current Model linear rule.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [linear glossary](../../../../glossary.md#term-linear-meanings))</span>
- Treat every leading input or hidden Dimension as ordinary batch metadata. The two projected
  Shapes use current right-aligned ordinary ADD broadcasting, so their leading prefixes may have
  different ranks or singleton extents when <span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [metadata glossary](../../../../glossary.md#term-metadata); [broadcasting glossary](../../../../glossary.md#broadcasting); [singleton glossary](../../../../glossary.md#term-singleton))</span>`ShapeBroadcast` can prove compatibility. No leading
  axis is named, interpreted, or traversed as time.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>
- Complete all caller-controlled forward validation before creating the first projection
  expression. Then construct exactly the formula and producer chain specified below.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [producer glossary](../../../../glossary.md#term-producer))</span>
- Keep the layer mode-insensitive. Forward accepts no `ForwardContext`, does not inspect
  `mode()`, and does not alter mode or state.
- Preserve stable parameter wrappers, compatible replacement snapshots, recursive discovery, and
  state-dictionary paths through the existing [`Module`](../../../../glossary.md#term-module) contracts.
- Add focused exact-surface, supplied-state, initialization, validation-order, shape/type,
  provenance, replacement, mode, state-discovery, and exclusion tests.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- Add complete type, constructor, member, and package Javadocs. After executable work and final NN
  testing, use the required separate clean documentation-focused context to finalize Javadocs,
  glossary impact, planning evidence, generated Javadoc, and no-change conclusions.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [planning glossary](../../../../glossary.md#term-planning); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Retaining the current or next hidden [Tensor](../../../../glossary.md#tensor) in a field, [`Buffer`](../../../../glossary.md#term-buffer), child, state dictionary,
  thread-local, static, runtime object, session, or other hidden lifecycle.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- A default/zero hidden state, hidden-state initializer, state reset/detach API, stateful forward
  overload, or a result that omits the caller's responsibility to thread state.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- A result record containing `output` and `nextHidden`. They would be the same exact Tensor for
  this cell, and no current consumer requires duplicate component names.
- `UnaryTensorModule`. The complete forward signature has two Tensor inputs, so the cell is not a
  unary Tensor-to-Tensor module.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [unary glossary](../../../../glossary.md#term-unary); [module glossary](../../../../glossary.md#term-module))</span>
- Participation in `Sequential`, an adapter into `Sequential`, a wrapper that captures hidden
  state, a context erasure mechanism, or a change to `Sequential`/`UnaryTensorModule`.
- A recurrent sequence container, loop, step counter, time-axis convention, sequence-length or
  mask input, packed sequence, variable-length handling, bidirectionality, stacking, dropout
  between steps, static unrolling, or any time traversal.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [mask glossary](../../../../glossary.md#term-boolean-mask); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [unrolling glossary](../../../../glossary.md#term-unrolling))</span>
- A general recurrent Model scan, loop operation, subgraph/body representation, carried-value
  tuple, scan result, or Tensor API. Existing <span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [operation glossary](../../../../glossary.md#operation))</span>`CUM_SUM` and `CUM_PROD` are ordered associative
  cumulative operations and must not be treated as recurrent scan.
- GRU gates, LSTM cell state, peepholes, projections, layer normalization, clipping, residuals,
  attention, convolutional recurrence, or another recurrent-cell family.<span data-glossary-navigation="true"> ([GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm); [layer normalization glossary](../../../../glossary.md#layer-normalization))</span>
- Configurable activation, activation enum, callback, function object, ReLU variant, sigmoid
  output, or fast/approximate tanh policy.<span data-glossary-navigation="true"> ([activation glossary](../../../../glossary.md#term-activation); [callback glossary](../../../../glossary.md#term-callback); [ReLU glossary](../../../../glossary.md#term-relu); [sigmoid glossary](../../../../glossary.md#term-sigmoid); [tanh glossary](../../../../glossary.md#term-tanh))</span>
- Separate input and recurrent biases, bias fusion policy, frozen parameters, mixed parameter
  types, integral/BOOL parameters, lazy/dynamic parameter Shapes, zero feature sizes, quantized or
  sparse state, custom initializer objects, orthogonal initialization, or a retained/default RNG.<span data-glossary-navigation="true"> ([biases glossary](../../../../glossary.md#term-bias); [fusion glossary](../../../../glossary.md#term-fusion); [parameters glossary](../../../../glossary.md#term-parameter); [BOOL glossary](../../../../glossary.md#data-type--datatype); [lazy glossary](../../../../glossary.md#term-lazy-evaluation); [Shapes glossary](../../../../glossary.md#shape); [feature glossary](../../../../glossary.md#term-feature); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- A `ForwardContext`, [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate), `RandomGenerator` during forward, mode-dependent branch,
  Buffer transition, or stochastic recurrence.
- Truncated backpropagation through time, gradient detachment, gradient rules, optimizer/session,
  parameter groups, checkpoint transport, serialization, compiler capture, graph scheduling,
  runtime/prepare/Engine behavior, backend lowering, kernels, numerical execution, or end-to-end
  claims.<span data-glossary-navigation="true"> ([backpropagation through time glossary](../../../../glossary.md#term-backpropagation-through-time-bptt); [gradient glossary](../../../../glossary.md#term-gradient); [optimizer glossary](../../../../glossary.md#term-optimizer); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [serialization glossary](../../../../glossary.md#term-serialization); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [graph glossary](../../../../glossary.md#graph); [prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel))</span>
- A Model, Training, Gradle, dependency, architecture-contract, ADR, architecture-test, global-
  roadmap, CPU, glossary, or explanatory-document edit during this planning pass.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [CPU glossary](../../../../glossary.md#term-cpu); [planning glossary](../../../../glossary.md#term-planning))</span>
- Detailed task specifications for [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0013–0015.

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
- [Completed NN task 0005: Linear layer](0005-linear-layer.md)<span data-glossary-navigation="true"> ([Linear glossary](../../../../glossary.md#term-linear-meanings))</span>
- [Completed NN task 0010: State dictionary](0010-state-dictionary-and-checkpoint-contract.md)
- [Completed NN task 0011: Unary Tensor composition](0011-unary-tensor-module-composition-and-sequential.md)<span data-glossary-navigation="true"> ([Unary glossary](../../../../glossary.md#term-unary); [Tensor composition glossary](../../../../glossary.md#tensor-composition))</span>
- [Completed Model task 0019D: Linear convenience](../../../modules/model/tasks/0019d-linear-convenience.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Completed Model task 0023E: Cumulative scan normalization and product](../../../modules/model/tasks/0023e-cumulative-scan-normalization-and-product.md)<span data-glossary-navigation="true"> ([Cumulative scan glossary](../../../../glossary.md#cumulative-scan); [normalization glossary](../../../../glossary.md#term-normalization))</span>

## Architecture constraints

- `extensions/nn` continues to depend only on `modules/model`. Production must not import
  Training, Compiler, Runtime, Prepare, Engine, CPU, Metal, CUDA, or another backend.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal); [CUDA glossary](../../../../glossary.md#term-cuda); [backend glossary](../../../../glossary.md#backend))</span>
- NN owns the two projection parameter bindings and typed recurrent-cell composition. Model
  remains the sole owner of linear, ADD, TANH, type promotion, local Shape algebra, Tensor
  descriptor/provenance construction, and expression identity.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [parameter glossary](../../../../glossary.md#term-parameter); [bindings glossary](../../../../glossary.md#term-binding); [Model glossary](../../../../glossary.md#term-model); [owner glossary](../../../../glossary.md#term-owner); [linear glossary](../../../../glossary.md#term-linear-meanings); [TANH glossary](../../../../glossary.md#term-tanh); [Shape glossary](../../../../glossary.md#shape); [Tensor descriptor glossary](../../../../glossary.md#tensor-descriptor); [provenance glossary](../../../../glossary.md#provenance); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Tensor identity, descriptors, and provenance remain immutable. Parameter replacement changes
  only the current Tensor reference returned by one stable NN wrapper; existing Tensors and
  expressions remain unchanged.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [descriptors glossary](../../../../glossary.md#term-descriptor); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Recurrent state is a caller-threaded Tensor value, not module-owned persistent state. A hidden
  Tensor must not be registered as a Buffer merely because a caller may pass it between calls.<span data-glossary-navigation="true"> ([Recurrent state glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state); [Buffer glossary](../../../../glossary.md#term-buffer))</span>
- [`Module`](../../../../glossary.md#term-module) remains without a universal forward method. `RnnCell` is a direct subclass with the
  truthful two-input signature and must not implement or extend <span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>`UnaryTensorModule`.
- `Sequential` remains a container only for `UnaryTensorModule`. This cell is excluded by type and
  by explicit contract; no adapter may hide one input or state transition.
- Mode is NN composition metadata. This mode-insensitive cell neither consumes a context nor reads
  inherited mode during forward.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>
- Construction and forward create eager parameter leaves or storage-free Model expression
  metadata only. They do not prove gradient implementation, graph capture, compiled execution,
  backend support, numerical values, or storage residency.<span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [gradient glossary](../../../../glossary.md#term-gradient); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [residency glossary](../../../../glossary.md#residency))</span>
- If implementation needs a new Model operation/helper/public method, result carrier, recurrent
  container, state Buffer, context/RNG input, dependency, architecture rule, or eighth task path,
  stop and report the exact blocker instead of widening this task.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [dependency glossary](../../../../glossary.md#term-dependency); [widening glossary](../../../../glossary.md#term-widening))</span>

## Public [API](../../../../glossary.md#term-application-programming-interface-api)

`RnnCell` declares exactly:

```java
public RnnCell(Tensor inputWeight, Tensor hiddenWeight)

public RnnCell(Tensor inputWeight, Tensor hiddenWeight, Tensor bias)

public RnnCell(
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
| two-[Tensor](../../../../glossary.md#tensor) constructor | Retains exact positive static input/hidden projection [weights](../../../../glossary.md#term-weight) and declares no [bias](../../../../glossary.md#term-bias). |
| three-Tensor constructor | Retains the same exact weights plus one exact shared bias; null bias is invalid. |
| initialized constructor | Uses explicit sizes/type/source, creates input weight then hidden weight through [Glorot](../../../../glossary.md#term-glorot-uniform) uniform and optional zero bias afterward. |
| `inputWeight()` | Returns the exact stable wrapper declared under `inputWeight`. |
| `hiddenWeight()` | Returns the exact stable wrapper declared under `hiddenWeight`. |
| `bias()` | Returns an empty Optional or the exact stable wrapper declared under `bias`. |
| `forward(input, hidden)` | Validates the complete two-input request, snapshots current [parameters](../../../../glossary.md#term-parameter), composes the fixed [tanh](../../../../glossary.md#term-tanh) formula once, and returns the exact next-hidden Tensor. |

The class declares no other public or protected constructor, method, field, nested type, interface,
or overload. Inherited final [`Module`](../../../../glossary.md#term-module) APIs remain available normally.

## State [schema](../../../../glossary.md#term-schema) and [ownership](../../../../glossary.md#term-ownership-meanings)

| Kind | Local name | [Shape](../../../../glossary.md#shape) orientation | Type | `requiresGrad` | Initialized policy |
|---|---|---|---|---|---|
| [Parameter](../../../../glossary.md#term-parameter) | `inputWeight` | `[hiddenSize, inputSize]` | exact configured floating type | `true` | [Glorot](../../../../glossary.md#term-glorot-uniform) uniform from explicit source |
| Parameter | `hiddenWeight` | `[hiddenSize, hiddenSize]` | same exact type | `true` | Glorot uniform from the same explicit source |
| Parameter, optional | [`bias`](../../../../glossary.md#term-bias) | `[hiddenSize]` | same exact type | `true` | exact typed zero |

The one hidden-size [Dimension](../../../../glossary.md#dimension) is the structurally equal static Dimension at `inputWeight` [axis](../../../../glossary.md#term-axis)
zero, both `hiddenWeight` axes, and optional bias axis zero. `inputSize` is `inputWeight` axis one.
Both sizes are strictly positive.

Direct discovery order and direct state-dictionary order are exactly `inputWeight`,
`hiddenWeight`, then optional `bias`. When an owning future [module](../../../../glossary.md#term-module) registers the cell as child
`cell`, recursive paths are `cell.inputWeight`, `cell.hiddenWeight`, and optional `cell.bias`.
The input hidden [Tensor](../../../../glossary.md#tensor) and returned next hidden Tensor never appear in parameter, [buffer](../../../../glossary.md#term-buffer), child,
or state-dictionary discovery.

## Supplied construction [validation](../../../../glossary.md#term-validation) and side effects

Both supplied constructors validate complete state before declaring any [parameter](../../../../glossary.md#term-parameter).

For the two-[Tensor](../../../../glossary.md#tensor) constructor:

1. reject null `inputWeight`, then null `hiddenWeight` with those exact parameter names;
2. validate input [weight](../../../../glossary.md#term-weight) floating type, `requiresGrad == true`, rank two, fully static Shape,
   positive hidden-size axis zero, then positive input-size axis one;
3. validate hidden weight floating type, `requiresGrad == true`, rank two, fully static Shape,
   positive axis zero, then positive axis one;
4. require hidden weight exact [data type](../../../../glossary.md#data-type--datatype) to equal input weight data type;
5. require hidden weight [axis](../../../../glossary.md#term-axis) zero to equal input weight hidden-size axis structurally;
6. require hidden weight axis one to equal that same hidden-size [Dimension](../../../../glossary.md#dimension) structurally; and
7. declare `inputWeight` then `hiddenWeight` and retain no [bias](../../../../glossary.md#term-bias).

For the three-Tensor constructor, reject null `bias` after the two weight null checks and before
state-[schema](../../../../glossary.md#term-schema) validation. Apply steps 2–6, then validate bias floating type,
`requiresGrad == true`, [rank](../../../../glossary.md#term-rank) one, fully static [Shape](../../../../glossary.md#shape), exact common parameter data type, and exact
hidden-size Dimension in that order. Declare `inputWeight`, `hiddenWeight`, then `bias` only after
all checks pass.

Validation creates no Tensor, [producer](../../../../glossary.md#term-producer), storage, random draw, or Tensor [identity](../../../../glossary.md#term-identity-meanings) and never mutates
the supplied values. A failure returns no cell and leaves every supplied Tensor unchanged.

## Initialized construction and side effects

The initialized constructor performs exactly:

1. reject null `dataType`, then null `randomGenerator`;
2. require `inputSize > 0`;
3. require `hiddenSize > 0`;
4. require the [data type](../../../../glossary.md#data-type--datatype) to be floating;
5. create [immutable](../../../../glossary.md#term-immutable) [Shapes](../../../../glossary.md#shape) `[hiddenSize, inputSize]`, `[hiddenSize, hiddenSize]`, and, only when
   requested, `[hiddenSize]`;
6. obtain each requested Shape's checked known element count in state order and reject a count
   above `Integer.MAX_VALUE` before the first random draw or [Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings) allocation;
7. create input [weight](../../../../glossary.md#term-weight) through exactly
   `ParameterInitializers.glorotUniform(inputWeightShape, dataType, randomGenerator)`;
8. create hidden weight through exactly
   `ParameterInitializers.glorotUniform(hiddenWeightShape, dataType, randomGenerator)` using the
   same now-advanced source;
9. when requested, create [bias](../../../../glossary.md#term-bias) through exactly
   `ParameterInitializers.zeros(biasShape, dataType)` with no source call; and
10. after all requested leaves exist, declare [parameters](../../../../glossary.md#term-parameter) in exact state-table order.

Caller-controlled null, size, type, checked-count, and Java-array-limit failures precede every
draw and Tensor ID. Input-weight [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) consume one bounded `nextDouble(origin, bound)` call per
[row-major](../../../../glossary.md#term-row-major) [element](../../../../glossary.md#term-element), followed by all hidden-weight calls; the bounds are computed independently by
each weight's [Glorot](../../../../glossary.md#term-glorot-uniform) fan pair. Bias consumes no draw.

A random-source failure keeps completed source calls and creates no Tensor for the failing weight.
If it occurs during hidden-weight creation, the already created input weight and its ID are not
rolled back, but no cell is returned. Later allocation or identifier failures likewise preserve
completed draws and already allocated IDs. The caller owns and coordinates the source; the cell
never retains, resets, closes, synchronizes, splits, [seeds](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings), or serializes it.

## Forward [validation](../../../../glossary.md#term-validation) and side-effect order

`forward(input, hidden)` performs exactly:

1. reject null `input`, then null `hidden`;
2. read the current `inputWeight`, `hiddenWeight`, and optional [`bias`](../../../../glossary.md#term-bias) bindings exactly once in
   parameter declaration order;
3. prevalidate the input affine projection under the current Model linear order: promote input
   and input-weight numeric types, require input rank at least one, reject a proven unequal static
   input-feature contraction, and, when bias is present, promote the product and bias types;<span data-glossary-navigation="true"> ([weight glossary](../../../../glossary.md#term-weight))</span>
4. prevalidate the hidden projection: promote hidden and hidden-weight numeric types, require
   hidden rank at least one, and reject a proven unequal static hidden-feature contraction;
5. derive the two projection Shapes without creating a Tensor: preserve each operand's complete
   leading prefix and append the exact static hidden-size Dimension;
6. promote the two projection result types, then call ordinary
   `ShapeBroadcast.broadcast(inputProjectionShape, hiddenProjectionShape)` once to validate and
   derive their exact ADD result Shape; and
7. only after every preceding check succeeds, construct the formula below.

Because every [parameter](../../../../glossary.md#term-parameter) is floating, current same-category promotion requires both call inputs to
be floating, while allowing [BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) [widening](../../../../glossary.md#term-widening). A static final [feature](../../../../glossary.md#term-feature) [Dimension](../../../../glossary.md#dimension)
must equal the configured size. An unresolved final feature Dimension is accepted under current
[Model](../../../../glossary.md#term-model) [linear](../../../../glossary.md#term-linear-meanings) semantics and leaves that [contraction](../../../../glossary.md#contraction-dimension) obligation for [compiler](../../../../glossary.md#term-compiler) validation or later
[binding](../../../../glossary.md#term-binding).

Ordinary leading-prefix [broadcasting](../../../../glossary.md#broadcasting) is conservative. Equal dimensions and static [singleton](../../../../glossary.md#term-singleton)
expansion succeed; different symbolic dimensions, incompatible static sizes, and a symbolic
dimension paired with a static non-singleton fail before expression creation. Input and hidden
ranks need not equal. For example:

| Input [Shape](../../../../glossary.md#shape) | Hidden Shape | Next-hidden Shape | Meaning |
|---|---|---|---|
| `[inputSize]` | `[hiddenSize]` | `[hiddenSize]` | one unbatched cell application |
| `[batch, inputSize]` | `[batch, hiddenSize]` | `[batch, hiddenSize]` | matching [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) prefix |
| `[batch, inputSize]` | `[hiddenSize]` | `[batch, hiddenSize]` | hidden [vector](../../../../glossary.md#term-vector) broadcast across batch |
| `[outer, 1, inputSize]` | `[batch, hiddenSize]` | `[outer, batch, hiddenSize]` | ordinary right-aligned batch broadcasting |

These examples define [metadata](../../../../glossary.md#term-metadata) composition only. `outer` and `batch` are leading [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix),
not a time traversal or sequence loop.

Every local null/type/[rank](../../../../glossary.md#term-rank)/static-contraction/broadcast failure consumes no [Tensor](../../../../glossary.md#tensor) ID and creates
no partial expression. Parameter reads and temporary Shape values are not Tensor-expression side
effects. Identifier exhaustion during the valid construction phase may leave a successful prefix
of expression IDs; no [rollback](../../../../glossary.md#term-rollback) is attempted.

## Formula, delegation, and [provenance](../../../../glossary.md#provenance)

Let `x` be `input`, `h` be `hidden`, `W_ih` be `inputWeight`, and `W_hh` be `hiddenWeight`.

Without [bias](../../../../glossary.md#term-bias):

```text
h' = tanh((x @ transpose(W_ih)) + (h @ transpose(W_hh)))
```

With the one shared bias:

```text
h' = tanh(((x @ transpose(W_ih)) + bias) + (h @ transpose(W_hh)))
```

After prevalidation, implementation delegates exactly as:

```java
Tensor inputProjection = currentBias.isPresent()
        ? input.linear(currentInputWeight, currentBias.orElseThrow())
        : input.linear(currentInputWeight);
Tensor hiddenProjection = hidden.linear(currentHiddenWeight);
return inputProjection.add(hiddenProjection).tanh();
```

No helper constructs [`Operation`](../../../../glossary.md#operation), `TensorDescriptor`, `TensorProvenance`, or derived [Tensor](../../../../glossary.md#tensor)
directly. [Model](../../../../glossary.md#term-model) owns each [occurrence](../../../../glossary.md#term-occurrence).

The no-bias successful chain creates exactly six fresh Tensors/IDs in order: input-[weight](../../../../glossary.md#term-weight)
PERMUTE, input [MATMUL](../../../../glossary.md#matrix-multiplication--matmul), hidden-weight PERMUTE, hidden MATMUL, projection ADD, then [TANH](../../../../glossary.md#term-tanh). The biased
chain creates seven: input-weight PERMUTE, input MATMUL, bias ADD, hidden-weight PERMUTE, hidden
MATMUL, projection ADD, then TANH.

The returned Tensor has TANH provenance over the exact projection ADD. That ADD retains exact
ordered `[inputProjection, hiddenProjection]` references. Each [linear projection](../../../../glossary.md#linear-projection) exposes the
existing PERMUTE-to-MATMUL chain, and the biased input projection exposes its existing final ADD
with ordered `[inputProduct, bias]`. The result has the prevalidated broadcast [Shape](../../../../glossary.md#shape), the widest
floating type selected in expression order, unresolved [layout](../../../../glossary.md#layout), [gradient](../../../../glossary.md#term-gradient) eligibility equal to the
logical OR of input, hidden, and all [parameter](../../../../glossary.md#term-parameter) requests, no label, and no [host storage](../../../../glossary.md#host-storage).

This provenance describes declarative expression construction. The task makes no claim about
floating [reassociation](../../../../glossary.md#term-reassociation), [fused](../../../../glossary.md#specialization-and-fusion) [kernels](../../../../glossary.md#kernel), numerical values, gradients, [compiler](../../../../glossary.md#term-compiler) [capture](../../../../glossary.md#term-capture-meanings), [backend](../../../../glossary.md#backend)
availability, or execution.

## Output and state-threading contract

The exact returned [Tensor](../../../../glossary.md#tensor) is simultaneously the cell's visible output and next hidden state. The
cell does not retain it. A caller expresses recurrence only by passing that exact reference, or
another explicitly selected compatible Tensor, as a later call's `hidden` [argument](../../../../glossary.md#term-argument):

```text
h1 = cell.forward(x1, h0)
h2 = cell.forward(x2, h1)
```

Reusing `h0` in two calls creates two explicit branches. Passing `h1` to the next call creates a
[dependency](../../../../glossary.md#term-dependency) chain. Neither pattern mutates a Tensor or the cell. There is no implicit call order,
step [identity](../../../../glossary.md#term-identity-meanings), truncated history, detachment, or sequence [ownership](../../../../glossary.md#term-ownership-meanings).

## Replacement, mode, and composition behavior

- Each forward call snapshots all current parameter bindings before validation and expression
  construction. A compatible <span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [bindings glossary](../../../../glossary.md#term-binding); [validation glossary](../../../../glossary.md#term-validation))</span>`Parameter.replace` affects only later calls.
- An expression created before replacement keeps the exact old parameter references. Stable
  accessors and recursive discovery keep the same wrappers and expose the new binding on later
  `value()` reads.
- Individual replacement and forward construction remain non-thread-safe as a combined operation.
  Callers must coordinate them when one multi-parameter snapshot matters.<span data-glossary-navigation="true"> ([thread-safe glossary](../../../../glossary.md#term-thread-safe); [operation glossary](../../../../glossary.md#operation); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- `train()` and `eval()` retain inherited structural behavior, but forward composition is
  identical in both modes and consumes no `ForwardContext`.
- `RnnCell.class.getSuperclass()` is exactly [`Module`](../../../../glossary.md#term-module). It is not assignable to
  `UnaryTensorModule`, and no `Sequential` constructor can accept it.
- A future recurrent-style container may own the cell as an ordinary child Module and call its
  two-input method explicitly. Task 0012 neither names nor implements that container.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.nn.layers` — concrete stateful layer/cell [ownership](../../../../glossary.md#term-ownership-meanings).
- `io.github.pho001.synaptik.nn.module` — existing direct [`Module`](../../../../glossary.md#term-module) and stable [`Parameter`](../../../../glossary.md#term-parameter) wrappers.
- `io.github.pho001.synaptik.nn.initialization` — existing [Glorot-uniform](../../../../glossary.md#term-glorot-uniform) and zero policies.
- `io.github.pho001.synaptik.model.tensor` — existing [Tensor](../../../../glossary.md#tensor) [linear](../../../../glossary.md#term-linear-meanings), ADD, and [TANH](../../../../glossary.md#term-tanh) expression [API](../../../../glossary.md#term-application-programming-interface-api).
- `io.github.pho001.synaptik.model.datatype` — exact floating types and promotion [preflight](../../../../glossary.md#term-preflight).
- `io.github.pho001.synaptik.model.shape` — static parameter Shapes and ordinary broadcast
  preflight.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape))</span>

No package is added. Exact type placement:

- `io.github.pho001.synaptik.nn.layers.RnnCell` — direct Module owning vanilla recurrent
  projection parameters.
- `io.github.pho001.synaptik.nn.layers.RnnCellTest` — exact surface, state, supplied validation,
  forward metadata/provenance, mode/exclusion, and replacement coverage.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance))</span>
- `io.github.pho001.synaptik.nn.layers.RnnCellInitializationTest` — initialized policy, source,
  type, count, allocation, identifier, and failure-side-effect coverage.

## Affected files

Expected production files:

- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/RnnCell.java` (new).
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/package-info.java`.

Expected test files:

- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/RnnCellTest.java` (new).
- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/RnnCellInitializationTest.java`
  (new).

Expected documentation and [planning](../../../../glossary.md#term-planning) files:

- `docs/glossary.md` — extend the existing NN/recurrent terminology with explicit hidden-state
  threading, cell-versus-sequence distinction, and the declarative formula; do not add a glossary
  term during this planning pass.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [state threading glossary](../../../../glossary.md#state-threading))</span>
- `docs/planning/extensions/nn/master-plan.md`.
- this task specification.

## Maximum [scope](../../../../glossary.md#term-scope)

Implementation may create or modify exactly the seven paths above: two production/[Javadoc](../../../../glossary.md#term-javadoc) paths,
two focused test paths, the glossary, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) master plan, and this task. If implementation needs
another result type, [Model](../../../../glossary.md#term-model) helper, existing Sequential test edit, Training [API](../../../../glossary.md#term-application-programming-interface-api) edit, [architecture
test](../../../../glossary.md#term-architecture-test), [dependency](../../../../glossary.md#term-dependency), build file, recurrent container, or eighth path, stop and propose a focused
revision rather than expanding this task.

## Acceptance criteria

- Public final `RnnCell extends Module` declares exactly the three constructors, three parameter
  accessors, and one `forward(Tensor, Tensor)` method in the API table, with no other declared
  public/protected member.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- It neither extends nor implements `UnaryTensorModule` and cannot participate in `Sequential`.
  No adapter, hidden-state capture, or Sequential overload is added.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- Supplied constructors retain exact Tensors, validate all nulls then complete input-weight,
  hidden-weight, and optional-bias schema in the stated order, and declare exact names/order only
  after validation without allocating Tensor IDs.<span data-glossary-navigation="true"> ([Tensors glossary](../../../../glossary.md#tensor); [weight glossary](../../../../glossary.md#term-weight); [bias glossary](../../../../glossary.md#term-bias); [validation glossary](../../../../glossary.md#term-validation))</span>
- Initialized construction supports exactly current floating types, validates all requested
  Shapes/counts before the first draw, creates input weight then hidden weight through Glorot
  uniform and optional zero bias afterward, uses one explicit source, and retains no source.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [Glorot glossary](../../../../glossary.md#term-glorot-uniform))</span>
- Direct and recursive discovery plus state dictionary expose exactly the selected parameter
  names/order and no Buffer, hidden state, input, output, or RNG state.<span data-glossary-navigation="true"> ([Buffer glossary](../../../../glossary.md#term-buffer); [RNG state glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- Forward null-checks input then hidden, snapshots parameters once, and prevalidates the complete
  input projection, hidden projection, type combination, and ordinary broadcast before creating
  the first expression.
- Tests cover rank-one and higher-rank calls, equal and broadcastable leading prefixes, hidden
  vector broadcast, all three floating types and mixed widening, static feature mismatches,
  incompatible leading prefixes, and representative symbolic compatibility/rejection.<span data-glossary-navigation="true"> ([rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [rank glossary](../../../../glossary.md#term-rank); [vector glossary](../../../../glossary.md#term-vector); [widening glossary](../../../../glossary.md#term-widening); [feature glossary](../../../../glossary.md#term-feature))</span>
- Valid no-bias and biased calls build exactly the selected primitive chains and ordered exact
  provenance with the documented ID order, result Shape/type/eligibility/layout/label/storage,
  and fresh identities without reading numerical values.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [layout glossary](../../../../glossary.md#layout))</span>
- The returned Tensor is used once as both output and next hidden. No duplicate-reference result
  carrier, state field, Buffer transition, state initialization, or mutation appears.
- Training and evaluation modes create the same expression contract. No context, graph RNG,
  random draw during forward, runtime state, or hidden side effect appears.<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation); [graph glossary](../../../../glossary.md#graph); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [runtime glossary](../../../../glossary.md#term-runtime); [side effect glossary](../../../../glossary.md#term-side-effect))</span>
- Compatible parameter replacement affects only later forward expressions; earlier expressions
  retain old exact bindings, wrappers and paths remain stable, and incompatible replacement stays
  governed by existing <span data-glossary-navigation="true"> ([bindings glossary](../../../../glossary.md#term-binding))</span>[`Parameter`](../../../../glossary.md#term-parameter) [schema](../../../../glossary.md#term-schema).
- Focused tests prove local validation consumes no Tensor ID, while identifier/source/allocation
  failures retain current non-rollback effects without returning a partially initialized cell or
  hidden state.<span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback))</span>
- Public/package Javadocs document purpose, fixed formula/activation, parameter orientation and
  names, initialization order/source ownership, exact state threading, leading batch broadcast,
  validation/failure order, result/provenance, mode, replacement snapshots, thread safety,
  Sequential exclusion, and no-execution boundaries with complete tags.<span data-glossary-navigation="true"> ([activation glossary](../../../../glossary.md#term-activation); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [state threading glossary](../../../../glossary.md#state-threading); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- The glossary distinguishes a one-step explicit-state cell from sequence recurrence and from
  Model cumulative sum/product scans, with no invented scan API or numerical-execution claim.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- A separate clean documentation-focused context finalizes all affected Javadocs, package docs,
  glossary, planning evidence, generated Javadoc, Markdown, exact scope/status, and no-change
  conclusions before the task becomes Complete.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- No Model, Training, architecture/ADR/test, Gradle/dependency, compiler/runtime/prepare/Engine/
  backend, conformance/integration, CPU, global-roadmap, task-0013–0015 file, or unrelated path
  enters the implementation diff.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [compiler glossary](../../../../glossary.md#term-compiler); [prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [CPU glossary](../../../../glossary.md#term-cpu))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Validation tier: normal task validation for the single affected `extensions/nn` [module](../../../../glossary.md#term-module) plus the
required targeted documentation pass. This task changes no project [dependency](../../../../glossary.md#term-dependency), architecture
boundary, [backend](../../../../glossary.md#backend) behavior, or end-to-end execution path.

Implementation runs focused tests while developing:

```text
./gradlew :extensions:nn:test \
  --tests io.github.pho001.synaptik.nn.layers.RnnCellTest \
  --tests io.github.pho001.synaptik.nn.layers.RnnCellInitializationTest
```

After executable Java and tests stabilize, [run](../../../../glossary.md#term-run) the affected module exactly once as final Java
evidence:

```text
./gradlew :extensions:nn:test
```

The separate documentation-focused pass reuses that successful evidence unless it changes
executable Java or records a concrete reason to rerun. After final [Javadoc](../../../../glossary.md#term-javadoc) edits it runs:

```text
./gradlew :extensions:nn:javadoc
git diff --check
```

Final implementation/documentation validation also checks:

- reflection and `javap -public` for exact finality, direct `Module` superclass, three
  constructors/four methods, optional accessor type, and absence of another public/protected API;<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- source/reflection proof that `RnnCell` is outside `UnaryTensorModule` and `Sequential`;
- exact parameter names, order, Shapes, types, gradient flags, discovery paths, and absence of
  Buffer/hidden state;<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [gradient glossary](../../../../glossary.md#term-gradient); [Buffer glossary](../../../../glossary.md#term-buffer))</span>
- manual [provenance](../../../../glossary.md#provenance) and ID-order inspection for both formulas;
- production imports and the unchanged sole [Model](../../../../glossary.md#term-model) [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) dependency;
- generated `RnnCell` and layers-package Javadoc pages;
- local Markdown links/anchors, required headings, balanced fences, terminal newlines, and
  trailing whitespace;
- exactly seven task-owned implementation paths;
- NN 0001–0011 Complete, exactly NN 0012 Ready/In progress/Complete as appropriate, 0013–0015
  concise Draft rows, and no detailed 0013–0015 task files;<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- `git diff --check`; and
- `git diff --no-index --check /dev/null <path>` for every untracked new file.

Repository-wide, architecture, Model, Training, [CPU](../../../../glossary.md#term-cpu), [compiler](../../../../glossary.md#term-compiler), backend-[conformance](../../../../glossary.md#term-conformance), and integration
tests are deferred to the recurrent NN milestone [checkpoint](../../../../glossary.md#term-checkpoint) or [CI](../../../../glossary.md#term-continuous-integration-ci). Existing Model tests remain the
authoritative exhaustive [linear](../../../../glossary.md#term-linear-meanings), ADD, [TANH](../../../../glossary.md#term-tanh), promotion, and [Shape](../../../../glossary.md#shape)-broadcast [matrices](../../../../glossary.md#term-matrix-meanings); focused NN
tests lock only the cell-owned composition and [preflight](../../../../glossary.md#term-preflight).

## Dependencies

- NN tasks 0001–0011 are Complete and provide direct Module ownership, stable parameters,
  explicit-source initialization, deterministic discovery/state dictionaries, and the proven
  boundary that non-unary modules remain outside Sequential.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [parameters glossary](../../../../glossary.md#term-parameter); [unary glossary](../../../../glossary.md#term-unary))</span>
- [Model](../../../../glossary.md#term-model) `Tensor.linear`, `Tensor.add`, and `Tensor.tanh`, same-category promotion,
  `ShapeBroadcast`, [immutable](../../../../glossary.md#term-immutable) [descriptors](../../../../glossary.md#term-descriptor)/[provenance](../../../../glossary.md#provenance), and identifier allocation are Complete.
- Existing `ParameterInitializers.glorotUniform` and `zeros` provide every selected initialized
  state policy. No new initializer is required.
- ADR 0007 and the existing architecture dependency test already permit Model-only NN
  composition.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Compiler, Training, CPU, runtime, prepare, Engine, backend, and numerical execution support are
  not prerequisites because this task constructs parameter leaves and Tensor-expression metadata
  only.<span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [CPU glossary](../../../../glossary.md#term-cpu); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [Tensor glossary](../../../../glossary.md#tensor); [metadata glossary](../../../../glossary.md#term-metadata))</span>

## Follow-up tasks

- NN 0013: GRU cell remains Draft. It must keep hidden state explicit and choose its exact gate
  packing, parameter/bias schema, formula order, and result only at that frontier.<span data-glossary-navigation="true"> ([GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [parameter glossary](../../../../glossary.md#term-parameter); [bias glossary](../../../../glossary.md#term-bias); [schema glossary](../../../../glossary.md#term-schema))</span>
- NN 0014: LSTM cell remains Draft. It must accept and return both hidden and cell state explicitly
  and must not store either in a Buffer or hidden runtime field.<span data-glossary-navigation="true"> ([LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm); [Buffer glossary](../../../../glossary.md#term-buffer); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- NN 0015: recurrent sequence composition and scan decision remains Draft. It must decide static
  unrolling of concrete cell calls versus first introducing a genuinely general recurrent Model
  scan. <span data-glossary-navigation="true"> ([unrolling glossary](../../../../glossary.md#term-unrolling); [Model glossary](../../../../glossary.md#term-model))</span>`CUM_SUM`/`CUM_PROD` do not satisfy that need.
- If [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0015 chooses a container, it must prefer type-safe cell-specific
  `RnnSequence`/`GruSequence`/`LstmSequence` APIs unless the actual cell [signatures](../../../../glossary.md#term-signature) prove a shared
  `RecurrentSequence` without casts, state erasure, duplicate carriers, or hidden state. This task
  reserves none of those type names.
- Truncated backpropagation, masking, variable lengths, bidirectionality, stacked recurrence,
  recurrent dropout, checkpoint persistence, training sessions, and backend execution require
  later concrete consumers and separate plans.<span data-glossary-navigation="true"> ([backpropagation glossary](../../../../glossary.md#term-backpropagation); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [persistence glossary](../../../../glossary.md#term-persistence); [backend glossary](../../../../glossary.md#backend))</span>

## Documentation and no-change review

Document profiles:

- Java/package [Javadoc](../../../../glossary.md#term-javadoc): General plus [API](../../../../glossary.md#term-application-programming-interface-api)/Javadoc.
- glossary: General reference style.
- task/master plan: General plus [Planning](../../../../glossary.md#term-planning).

Required implementation-phase documentation changes are the `RnnCell` and layers-package
Javadocs, the existing glossary's [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/recurrent wording, and synchronized NN planning records.

The separate documentation pass must verify and record these reasoned no-change conclusions:

- `ARCHITECTURE.md`, focused architecture documents, ADR 0007, and architecture tests remain
  accurate because state ownership, module direction, and lifecycle boundaries do not change.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Tensor and Compile APIs plus Model source/plans/capabilities remain accurate because the cell
  composes existing expressions and adds no Tensor method, semantic kind, scan, capture, gradient,
  or execution behavior.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [Model glossary](../../../../glossary.md#term-model); [capture glossary](../../../../glossary.md#term-capture-meanings); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Training API and training graph remain accurate because existing recursive parameter discovery
  and replacement consume this direct Module normally; no optimizer, session, gradient
  publication, truncated history, or state orchestration is added.<span data-glossary-navigation="true"> ([training graph glossary](../../../../glossary.md#training-graph); [parameter glossary](../../../../glossary.md#term-parameter); [optimizer glossary](../../../../glossary.md#term-optimizer); [publication glossary](../../../../glossary.md#term-publication))</span>
- `UnaryTensorModule`, `Sequential`, and their tests remain accurate because the two-input cell is
  intentionally excluded; the new focused cell test owns the exclusion assertion.
- State-dictionary contracts remain accurate because only ordinary parameter paths are added and
  caller-threaded hidden state is not module state.
- [Gradle](../../../../glossary.md#term-gradle) and [dependency](../../../../glossary.md#term-dependency) rules/tests remain accurate because `extensions/nn` retains only its
  existing Model dependency.
- Backend conformance and integration tests remain unnecessary because no numerical execution,
  backend support, or end-to-end lifecycle changes.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- The global roadmap, [CPU](../../../../glossary.md#term-cpu) work, other [modules](../../../../glossary.md#term-module), and Draft NN tasks 0013–0015 remain untouched.

## Architecture impact

Expected impact: None.

This task realizes the existing [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) responsibility for [parameter](../../../../glossary.md#term-parameter)-owning layer composition while
keeping [recurrent state](../../../../glossary.md#backpropagation-through-time-and-recurrent-state) explicit. If implementation requires hidden [module](../../../../glossary.md#term-module)/[runtime](../../../../glossary.md#term-runtime) state, a
universal forward abstraction, a [general scan](../../../../glossary.md#general-differentiable-scan), another module [dependency](../../../../glossary.md#term-dependency), or an architecture
rule, stop and report the conflict rather than editing architecture within this task.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are the clean-context implementation agent for Synaptik NN task 0012. Work in the existing
shared worktree. Do not use GSD. Do not commit or push. Preserve all unrelated/concurrent CPU
source, test, documentation, master-plan, task, roadmap, and glossary changes exactly.

Read root AGENTS.md, ARCHITECTURE.md, the current architecture plan, planning guide/roadmap,
documentation rules and General/API-Javadoc/Planning profiles, NN master plan and tasks 0001–0012,
ADR 0007, final Module/Parameter/initializers/ForwardContext/UnaryTensorModule/Sequential and layer
APIs/tests, Model master plan plus final Tensor linear/matmul/add/tanh/shape/broadcast APIs,
Javadocs and tests, recurrent/scan Model planning, Training API/graph, glossary, dependency/build
architecture tests, and this task in full.

Implement exactly the final direct-Module RnnCell public API, state schema, supplied/initialized
validation, explicit caller-threaded hidden-state contract, full forward preflight, fixed tanh
formula, provenance order, replacement snapshots, mode-insensitivity, and Sequential exclusion in
the seven authorized paths. Add no result record, hidden Buffer/state, UnaryTensorModule,
Sequential adapter, ForwardContext, GraphRngState, sequence loop, time traversal, scan API,
optimizer/session, Model change, backend/execution behavior, or eighth path.

Run the focused RnnCell tests and one authoritative NN module test after executable Java
stabilizes. Then hand the unchanged executable diff and exact evidence to a distinct clean
documentation-focused context. That context must independently finalize Javadocs, package docs,
glossary, planning evidence, no-change conclusions, generated Javadoc, Markdown, surface, scope,
status, newline, and whitespace gates without repeating successful Java tests unless executable
behavior changes or a concrete risk is recorded.

If current final Model APIs cannot express the exact formula/preflight, if initialized-state
validation cannot precede the first draw as specified, or if type-safe explicit state requires
another public type or dependency, stop and report the exact blocker rather than inventing a
contract. Mark Complete only after implementation, documentation, and all required validation
pass.
```

## Documentation-agent [handoff](../../../../glossary.md#term-handoff)

After executable Java/tests stabilize, give the required clean documentation context:

- this task and exact seven-path limit;
- the final executable diff and exact focused/final [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) commands, counts, and results;
- the exact public surface, direct-Module/Sequential exclusion, state names/order/Shapes/types,
  constructor prevalidation and initialization draw/ID order;<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module); [Shapes glossary](../../../../glossary.md#shape))</span>
- full forward preflight, batch-broadcast examples, exact no-bias/biased producer chains, returned
  next-hidden identity, mode behavior, replacement snapshots, and failure effects;<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [bias glossary](../../../../glossary.md#term-bias); [producer glossary](../../../../glossary.md#term-producer); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- directly relevant architecture/ADR, documentation profiles, final NN/Model source and tests,
  Tensor/Training APIs, glossary, dependency test, and planning history;<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor); [dependency glossary](../../../../glossary.md#term-dependency); [planning glossary](../../../../glossary.md#term-planning))</span>
- the mandate to finalize only task-owned Javadocs and prose, preserve concurrent glossary work,
  and record every reasoned no-change conclusion;
- generated-[Javadoc](../../../../glossary.md#term-javadoc), reflection/`javap`, import/dependency, Markdown, exact-scope/status,
  task-0013–0015-absence, newline, and whitespace gates; and<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- the required completion-summary and `Status` format from `AGENTS.md`.

## Local decisions

- Use `RnnCell`, following the project's ordinary Java acronym style such as `Rng`, and place it
  beside concrete layers. The public name says one recurrent step, not sequence traversal.
- Extend [`Module`](../../../../glossary.md#term-module) directly. Two Tensor inputs are intrinsic to the cell; forcing it into the unary
  base would require captured hidden state or an adapter and would make Sequential unsafe.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary))</span>
- Return one [Tensor](../../../../glossary.md#tensor). In a vanilla cell `output == nextHidden` by contract, so a record containing
  both names would duplicate one reference without additional information.
- Use one shared optional bias. It is the smallest conventional affine state and avoids separate
  input/recurrent biases whose only effect would be another parameterization of the same summed
  preactivation.
- Associate the shared [bias](../../../../glossary.md#term-bias) with the input [`linear`](../../../../glossary.md#term-linear-meanings) call, fixing expression association as
  `((xW^T + b) + hU^T)` before TANH. This reuses the existing fully prevalidated biased linear
  convenience and makes provenance unambiguous.<span data-glossary-navigation="true"> ([TANH glossary](../../../../glossary.md#term-tanh); [provenance glossary](../../../../glossary.md#provenance))</span>
- Fix TANH. Configurable activation would turn one cohesive vanilla-cell capability into a generic
  recurrent framework without a current consumer.<span data-glossary-navigation="true"> ([activation glossary](../../../../glossary.md#term-activation))</span>
- Use Glorot uniform independently for both rank-two matrices. It is the existing general
  unit-gain affine policy that supports both Shapes. No orthogonal initializer exists, and this
  task must not invent one.<span data-glossary-navigation="true"> ([Glorot glossary](../../../../glossary.md#term-glorot-uniform); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [matrices glossary](../../../../glossary.md#term-matrix-meanings); [Shapes glossary](../../../../glossary.md#shape))</span>
- Permit current Model mixed floating promotion and conservative leading broadcasting rather than
  imposing equal ranks or exact parameter input types. Preflight duplicates only the public local
  algebra necessary to prevent a late hidden/broadcast failure from leaving an input-projection
  prefix.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [broadcasting glossary](../../../../glossary.md#broadcasting); [parameter glossary](../../../../glossary.md#term-parameter); [Preflight glossary](../../../../glossary.md#term-preflight))</span>
- Accept unresolved final feature equality exactly as Model linear does. The fixed parameter
  output Dimension still makes every returned next-hidden final axis statically <span data-glossary-navigation="true"> ([feature glossary](../../../../glossary.md#term-feature); [Dimension glossary](../../../../glossary.md#dimension); [axis glossary](../../../../glossary.md#term-axis))</span>`hiddenSize`.
- Keep all leading axes semantically neutral batch coordinates. A future sequence task, not the
  cell, owns any time-axis contract and repeated invocation.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>

## Known limitations

- Only one vanilla tanh cell with positive fully static parameters and one optional shared bias is
  supported.<span data-glossary-navigation="true"> ([tanh glossary](../../../../glossary.md#term-tanh); [parameters glossary](../../../../glossary.md#term-parameter); [bias glossary](../../../../glossary.md#term-bias))</span>
- Input and hidden Shapes may contain unresolved leading or final Dimensions only when current
  local Model rules can represent the projection and broadcast obligations. No binding occurs in
  NN.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [Dimensions glossary](../../../../glossary.md#dimension); [Model glossary](../../../../glossary.md#term-model); [binding glossary](../../../../glossary.md#term-binding); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- A higher-rank input is processed as one batched cell application. It is never traversed as a
  sequence, even if a caller informally assigns a leading axis the meaning of time.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [axis glossary](../../../../glossary.md#term-axis))</span>
- Caller-threaded recurrence can build arbitrarily deep provenance chains; this task adds no
  detachment, truncation, loop IR, scheduling, or memory-lifetime policy.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- Parameter replacement and forward construction are not thread-safe as one snapshot. The cell
  retains no lock, version, or transaction.<span data-glossary-navigation="true"> ([thread-safe glossary](../../../../glossary.md#term-thread-safe); [snapshot glossary](../../../../glossary.md#term-snapshot); [transaction glossary](../../../../glossary.md#term-transaction))</span>
- Initialized eager parameters require Java-array-sized host leaves and may consume source calls
  or Tensor IDs before a later resource/identifier failure; those effects are not rolled back.<span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [host glossary](../../../../glossary.md#term-host); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Forward constructs metadata and proves no numerical value, gradient rule, compiler capture,
  backend support, storage allocation, publication, or execution.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [backend glossary](../../../../glossary.md#backend); [publication glossary](../../../../glossary.md#term-publication))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Clean [planning](../../../../glossary.md#term-planning) context `/root/nn_0012_planning` read the repository instructions, authoritative
  architecture and current architecture plan, planning guide/roadmap, documentation rules and
  General/Planning/API-Javadoc profiles, ADR 0007, complete NN master/task history through 0011,
  final NN module/initializer/layer APIs and tests, Model master and exact linear/MATMUL/ADD/TANH/
  Shape/broadcast contracts and tests, cumulative-scan planning/contracts, Training API/graph,
  glossary, and dependency/build architecture enforcement before selecting this API.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Model glossary](../../../../glossary.md#term-model); [linear glossary](../../../../glossary.md#term-linear-meanings); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [TANH glossary](../../../../glossary.md#term-tanh); [Shape glossary](../../../../glossary.md#shape); [scan glossary](../../../../glossary.md#term-cumulative-scan); [graph glossary](../../../../glossary.md#graph); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Planning inspection confirmed the cell is fully expressible through current Model operations.
  The selected preflight can reproduce every local promotion, rank, static-contraction, and
  projection-broadcast check before expression creation; no new Model helper or scan is required.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [rank glossary](../../../../glossary.md#term-rank); [contraction glossary](../../../../glossary.md#contraction-dimension))</span>
- Planning also confirmed that vanilla output and next hidden are one value, so a result record
  would duplicate an identical Tensor reference, and that <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>`UnaryTensorModule`/`Sequential` cannot
  truthfully represent a two-input call.
- Targeted planning validation resolved every Markdown link and local anchor in the two affected
  planning files; fences, final newlines, and trailing whitespace passed.
- The NN planning diff contains exactly this task and the NN master plan. The master has exactly
  one Ready row, this is the only Ready NN task, tasks 0013-0015 remain Draft rows without task
  specifications, whole-worktree <span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>`git diff --check` passed, and the new-file no-index whitespace
  check returned only the expected content-difference status.
- Concurrent CPU, source, test, documentation, master-plan, task, and global-roadmap changes were
  left untouched.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- Clean implementation context `/root/nn_0012_implementation` added the exact direct-[Module](../../../../glossary.md#term-module)
  `RnnCell`, its focused state/forward and initialization suites, and draft public/package
  Javadocs. The implementation required no Model helper, result carrier, hidden Buffer/state,
  ForwardContext, sequence/scan abstraction, dependency, or eighth task path.<span data-glossary-navigation="true"> ([Buffer glossary](../../../../glossary.md#term-buffer))</span>
- The stabilized focused command passed both new suites with 15 tests and no failures, errors, or
  skips. The sole authoritative `./gradlew :extensions:nn:test` run then passed 19 suites and 125
  tests with no failures, errors, or skips.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- Implementation provenance inspection confirmed the exact six-ID no-bias and seven-ID biased
  chains, ordered projection inputs, fixed TANH result, explicit hidden threading, mode
  independence, and old-versus-new binding snapshots. Constructor and forward failure tests
  confirmed the specified no-draw/no-ID and non-rollback boundaries.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [bias glossary](../../../../glossary.md#term-bias); [binding glossary](../../../../glossary.md#term-binding); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- Executable Java remained behaviorally unchanged after the implementation test runs. Independent
  clean documentation context `/root/nn_0012_docs` reviewed the final source/tests and inherited
  Module, Parameter, initializer, Linear, unary-composition, Model expression, Shape/type, and
  Training boundaries against the architecture and task. It refined the RnnCell type summary,
  accepted the complete constructor/member and package contracts, and added the glossary's
  explicit hidden-state, one-step recurrence, Sequential exclusion, and cumulative-scan
  distinction.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary))</span>
- The documentation context reused the implementation's focused two-suite/15-test evidence and
  authoritative 19-suite/125-test NN run because only Javadoc/prose changed afterward. The final
  XML reports independently still contain 19 suites and 125 tests with zero skips, failures, or
  errors.
- Final `./gradlew :extensions:nn:javadoc` passed after the [Javadoc](../../../../glossary.md#term-javadoc) refinement (`3 actionable
  tasks: 2 executed, 1 up-to-date`). Inspection of generated `RnnCell.html` and the layers
  `package-summary.html` confirmed the direct-Module/two-input boundary, parameter schemas,
  initialization and failure contracts, forward preflight/result semantics, and non-execution
  limits.<span data-glossary-navigation="true"> ([schemas glossary](../../../../glossary.md#term-schema))</span>
- Independent `javap -protected` and `javap -private` inspection plus a standalone reflection
  program passed: final direct `Module` superclass; exactly three public constructors and four
  declared public methods; no public/protected field or nested type; and only private stable
  [`Parameter`](../../../../glossary.md#term-parameter), `Parameter`, and `Optional<Parameter>` state. Source/import and build inspection
  confirmed Model/NN/JDK imports only, no hidden-state/Buffer/context/RNG/sequence dependency, and
  the unchanged sole Gradle dependency on <span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [Gradle glossary](../../../../glossary.md#term-gradle))</span>`:modules:model`.
- The targeted Markdown validator passed all three task-owned Markdown files with 349 local links,
  including 295 heading anchors, 42 balanced fence markers, final newlines, and no trailing
  whitespace. Exact-scope inspection found only the seven authorized NN 0012 paths amid preserved
  concurrent CPU work. NN 0001–0011 remain Complete; NN 0013–0015 remain Draft rows with no task
  specifications and no Ready NN task.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- `git diff --no-index --check /dev/null <path>` returned the expected content-difference status
  1 with no whitespace diagnostics for each of the four untracked task files, and final whole-
  worktree `git diff --check` passed with no output.
- `ARCHITECTURE.md`, focused architecture documents, ADR 0007, and architecture tests require no
  change: the cell stays inside existing Model-only NN composition, keeps hidden state caller-
  owned, and changes no boundary or dependency. Tensor/Compile APIs, Model source/master plan and
  capabilities also require no change because the cell only composes current linear, ADD, and TANH
  metadata and adds no operation, Tensor method, scan, gradient, capture, or execution behavior.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [metadata glossary](../../../../glossary.md#term-metadata); [operation glossary](../../../../glossary.md#operation); [gradient glossary](../../../../glossary.md#term-gradient); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- Training API and the training graph require no change because ordinary recursive parameter
  discovery/replacement already consumes a direct Module and this task adds no optimizer,
  gradient publication, truncated history, or state orchestration. <span data-glossary-navigation="true"> ([training graph glossary](../../../../glossary.md#training-graph); [optimizer glossary](../../../../glossary.md#term-optimizer); [publication glossary](../../../../glossary.md#term-publication))</span>`UnaryTensorModule`,
  `Sequential`, state-dictionary contracts, and their existing tests remain accurate: the focused
  RnnCell suite proves the cell's explicit exclusion and its caller-threaded hidden Tensor never
  becomes module state.
- Gradle/dependency rules and tests, backend conformance, and integration tests require no change
  because no dependency, numerical execution, backend support, or end-to-end lifecycle changed.
  Compile, Training, Model capabilities, other modules, the global roadmap, concurrent CPU work,
  and Draft NN tasks 0013–0015 were preserved unchanged.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## Implementation notes

Clean implementation context `/root/nn_0012_implementation` completed the exact executable
capability and tests. The focused two-suite/15-test selection and authoritative 19-suite/125-test
[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [module](../../../../glossary.md#term-module) [run](../../../../glossary.md#term-run) passed with no failures, errors, or skips. Independent documentation context
`/root/nn_0012_docs` then finalized type/package Javadocs, glossary and [planning](../../../../glossary.md#term-planning) evidence and
passed generated-[Javadoc](../../../../glossary.md#term-javadoc), surface/private-state, [dependency](../../../../glossary.md#term-dependency)/import, Markdown, seven-path [scope](../../../../glossary.md#term-scope),
status, newline, whitespace, and final-diff gates without changing executable behavior or
repeating the stable Java suites.

## Completion summary

- Completed changes: added final direct-[Module](../../../../glossary.md#term-module) `RnnCell` with exact supplied and explicit-source
  construction, stable two-weight/optional-bias state, complete forward preflight, fixed
  linear/add/tanh composition, one explicit next-hidden Tensor result, and focused contract tests.<span data-glossary-navigation="true"> ([weight glossary](../../../../glossary.md#term-weight); [bias glossary](../../../../glossary.md#term-bias); [preflight glossary](../../../../glossary.md#term-preflight); [linear glossary](../../../../glossary.md#term-linear-meanings); [tanh glossary](../../../../glossary.md#term-tanh); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Files changed or created: `RnnCell.java`, layers `package-info.java`, `RnnCellTest.java`,
  `RnnCellInitializationTest.java`, `docs/glossary.md`, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) `master-plan.md`, and this task.
- Tests and validation: reused passing focused 15-test and authoritative 19-suite/125-test NN
  evidence; final NN Javadoc, generated-page inspection, independent <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`javap`/reflection,
  dependency/import, Markdown, exact-scope/status, newline, whitespace, no-index, and diff checks
  passed.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: clean context `/root/nn_0012_docs` completed the mandatory
  independent source/test/API/architecture review and changed no executable behavior.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Documentation impact: finalized RnnCell and layers-package contracts, added the explicit-state
  recurrent glossary definition/example and cumulative-scan distinction, synchronized planning,
  and recorded reasoned no-change conclusions for architecture, Model/Tensor/Compile/Training,
  unary/state-dictionary, dependency/build, conformance/integration, roadmap, CPU, and later NN
  work.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [planning glossary](../../../../glossary.md#term-planning); [Model glossary](../../../../glossary.md#term-model); [Compile glossary](../../../../glossary.md#compile); [unary glossary](../../../../glossary.md#term-unary); [conformance glossary](../../../../glossary.md#term-conformance); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Unresolved issues: none.
- Required follow-up: none for task 0012; NN 0013–0015 remain separate Draft work.

```text
Status: Complete
```

# Task 0014: LSTM Cell

## Status

Complete

## Goal

Add one final public long short-term memory (LSTM) cell with explicit caller-threaded hidden and
cell state. The cell owns two gate-major packed projection matrices and one optional packed
input-side bias, constructs one fixed LSTM step entirely from current Model Tensor expressions,
and returns both next-state Tensors in one NN-owned result carrier. It retains no recurrent state.

Mental model:

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

## Scope

- Add final public `io.github.pho001.synaptik.nn.layers.LstmCell` extending
  `io.github.pho001.synaptik.nn.module.Module` directly.
- Add final public record `io.github.pho001.synaptik.nn.layers.LstmCellForwardResult` with exactly
  the components `Tensor nextHidden` and `Tensor nextCell`, in that order.
- Add exactly the constructors, parameter accessors, and three-input forward method in the public
  API table below. Add no overload, builder, options object, gate enum, size getter, functional
  facade, state tuple abstraction, or configurable equation policy.
- Declare positive fully static rank-two parameters under exact local names `inputWeight` then
  `hiddenWeight`. Their Shapes are `[4 * hiddenSize, inputSize]` and
  `[4 * hiddenSize, hiddenSize]`.
- Pack both matrices gate-major on axis zero in exact input, forget, candidate, output order.
  After linear projection, the same order occupies the final result axis.
- Optionally declare one positive fully static rank-one parameter under exact local name `bias`,
  with Shape `[4 * hiddenSize]`. It is added only by the packed input projection. There is no
  recurrent bias and no separate per-gate bias parameter.
- Require all parameters to share one exact floating data type and have `requiresGrad == true`.
  Retain caller-supplied parameter Tensor references exactly.
- Provide supplied-state construction both without and with bias. Null never means absent bias;
  callers select absence through the two-Tensor constructor.
- Provide one initialized constructor with explicit positive `inputSize`, positive `hiddenSize`,
  bias presence, floating `DataType`, and caller-owned `RandomGenerator`.
- Compute `packedHiddenSize = 4 * hiddenSize` with checked arithmetic before constructing Shapes.
  Initialize `inputWeight` then `hiddenWeight` through exact
  `ParameterInitializers.glorotUniform` calls using the same explicit source. When requested,
  initialize the complete packed bias afterward through exact `ParameterInitializers.zeros`;
  bias consumes no draw. The source is never retained.
- Use an all-zero initialized packed bias, including the forget interval. Current initializers do
  not provide one direct eager packed leaf with only the forget interval set to one. Do not hide
  host mutation, compose a parameter from slices/concatenation, or invent a new initializer in
  this task. A selected forget-bias policy requires a separate initializer/API decision.
- Add exactly `LstmCellForwardResult forward(Tensor input, Tensor hidden, Tensor cell)`. All three
  semantic inputs are explicit on every call; the module never supplies, caches, updates,
  registers, or discovers hidden or cell state.
- Fix the gate order, input-side-only optional-bias association, activation placement, and
  equations specified below. Do not claim checkpoint or formula compatibility with a framework
  whose packing, bias, or equation convention differs.
- Use independent `sliceAxis(-1, ...)` expressions for the eight projection gate slices. Current
  Model has no shared multi-output split operation; independent slices keep each exact packed
  source and interval visible without adding a Model API or per-gate parameters.
- Support input, hidden, and cell rank one or higher. Their final Dimensions respectively
  represent `inputSize`, `hiddenSize`, and `hiddenSize`.
- Treat all leading Dimensions as ordinary right-broadcastable batch metadata. No leading axis is
  named, interpreted, or traversed as time.
- Complete all caller-controlled forward validation before creating the first projection
  expression. Then construct exactly the formula and producer order specified below.
- Keep the cell mode-insensitive. Forward accepts no `ForwardContext`, does not inspect `mode()`,
  and does not alter module mode or state.
- Preserve stable parameter wrappers, schema-compatible replacement snapshots, recursive
  discovery, and state-dictionary paths through current `Module` and `Parameter` contracts.
- Add focused exact-surface/result-carrier, supplied-state, initialization, validation-order,
  shape/type, gate-slicing, provenance, replacement, mode, discovery, and exclusion tests.
- Add complete type, record-component, constructor, member, and package Javadocs. After executable
  work and final NN testing, use a separate clean documentation-focused context to finalize
  Javadocs, glossary impact, planning evidence, generated Javadoc, and no-change conclusions.

## Out of scope

- Retaining current or next hidden/cell Tensors in fields, `Buffer` values, children, state
  dictionaries, thread-locals, statics, runtime objects, sessions, or another hidden lifecycle.
- Default or zero hidden/cell state, state initializers, reset/detach APIs, stateful forward
  overloads, or a carrier that omits either state transition.
- `UnaryTensorModule`, participation in `Sequential`, an adapter into `Sequential`, or changes to
  `Sequential`, `UnaryTensorModule`, `Module`, `Parameter`, or `Buffer`.
- Separate public gate parameters, eight per-gate matrices, separate input/recurrent biases,
  per-gate bias accessors, or parameter-packing configuration.
- A non-zero forget-bias initializer, hidden eager storage mutation, parameter construction from
  slices or concatenation, or a new initializer convenience.
- Peephole connections, output projection, coupled input-forget gates, layer normalization, cell
  clipping, recurrent dropout, residuals, attention, convolutional recurrence, or configurable
  gate activations.
- Alternate gate order, bias association, activation placement, equation policy, framework-
  compatibility mode, checkpoint migration, or conversion helpers for another LSTM layout.
- A recurrent sequence container, loop, step counter, time-axis convention, masks, sequence
  lengths, packed sequence, variable-length handling, active-batch compaction, bidirectionality,
  stacking, static unrolling, or time traversal.
- A general recurrent Model scan, loop operation, subgraph/body representation, carried-value
  tuple, scan result, or Tensor API. Current `CUM_SUM` and `CUM_PROD` are associative cumulative
  operations and are not recurrent scan primitives.
- Backpropagation through time, gradient detachment, gradient rules, optimizer/session,
  parameter groups, checkpoint transport, serialization, compiler capture, scheduling,
  Runtime/Prepare/Engine behavior, backend lowering, kernels, numerical execution, or end-to-end
  support claims.
- A Model, Training, Gradle, dependency, architecture-contract, ADR, architecture-test, global-
  roadmap, CPU, or explanatory-architecture source change during implementation.
- A detailed task specification or implementation for NN 0015.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [ADR 0007: Neural-network module and training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Training API](../../../../api/training-api.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Planning guide](../../../planning-guide.md)
- [NN master plan](../master-plan.md)
- [Completed NN task 0004: Initializers](0004-explicit-eager-parameter-initializers.md)
- [Completed NN task 0004A: Replacement hardening](0004a-parameter-update-and-traversal-hardening.md)
- [Completed NN task 0011: Unary composition](0011-unary-tensor-module-composition-and-sequential.md)
- [Completed NN task 0012: Vanilla RNN cell](0012-vanilla-rnn-cell.md)
- [Completed NN task 0013: GRU cell](0013-gru-cell.md)
- [Completed Model task 0017H: Slice expressions](../../../modules/model/tasks/0017h-slice-tensor-expressions.md)
- [Completed Model task 0019D: Linear convenience](../../../modules/model/tasks/0019d-linear-convenience.md)
- [Completed Model task 0023E: Cumulative scan normalization](../../../modules/model/tasks/0023e-cumulative-scan-normalization-and-product.md)

## Architecture constraints

- `extensions/nn` continues to depend only on `modules/model`. Production must not import
  Training, Compiler, Runtime, Prepare, Engine, CPU, Metal, CUDA, or another backend.
- NN owns packed parameter bindings, the typed cell composition, and its explicit two-Tensor
  result carrier. Model remains the sole owner of LINEAR decomposition, SLICE, ADD, MUL, SIGMOID,
  TANH, type promotion, Shape algebra, descriptors, provenance, and Tensor identity.
- Tensor identity, descriptors, and provenance remain immutable. Parameter replacement changes
  only the current exact Tensor returned by one stable wrapper; existing expressions remain
  unchanged.
- Recurrent hidden and cell state are caller-threaded Tensor values, never module-owned persistent
  state. Neither becomes a `Buffer` merely because the caller may pass it between calls.
- `Module` retains no universal forward method. `LstmCell` is a direct subclass with a truthful
  three-input signature and must not extend `UnaryTensorModule`.
- `Sequential` remains a container only for `UnaryTensorModule`; no adapter may capture, erase,
  or synthesize either recurrent-state input.
- Mode is NN composition metadata. This mode-insensitive cell neither consumes a context nor
  reads inherited mode during forward.
- Packed slices are ordinary independent one-output Model expressions. The task must not invent a
  shared split producer, Model tuple, hidden gate cache, direct `Operation` construction, or
  special LSTM semantic kind.
- `LstmCellForwardResult` is an NN composition value, not a Model multi-output producer, Tensor,
  graph value, module state, checkpoint payload, runtime result, or execution handle.
- Construction and forward create eager parameter leaves or storage-free expression metadata
  only. They do not prove gradient implementation, graph capture, compiled execution, backend
  support, numerical values, or storage residency.
- If implementation needs a new Model operation/helper/public method, initializer, recurrent
  container, state Buffer, context/RNG input, dependency, architecture rule, or ninth task path,
  stop and report the exact blocker instead of widening this task.

## Public API

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
| two-Tensor constructor | Retains exact positive static packed input/hidden projection weights and declares no bias. |
| three-Tensor constructor | Retains the same exact weights plus one exact packed input-side bias; null bias is invalid. |
| initialized constructor | Uses explicit sizes/type/source, creates packed input weight then packed hidden weight through Glorot uniform and optional all-zero bias afterward. |
| `inputWeight()` | Returns the exact stable wrapper declared under `inputWeight`. |
| `hiddenWeight()` | Returns the exact stable wrapper declared under `hiddenWeight`. |
| `bias()` | Returns an empty Optional or the exact stable wrapper declared under `bias`. |
| `forward(input, hidden, cell)` | Validates the complete three-input request, snapshots current parameters, constructs the fixed LSTM formula once, and returns exact next-hidden/next-cell references in one fresh carrier. |

The class declares no other public or protected constructor, method, field, nested type,
interface, or overload. Inherited final `Module` APIs remain available normally.

`LstmCellForwardResult` declares exactly:

```java
public record LstmCellForwardResult(Tensor nextHidden, Tensor nextCell)
```

Its canonical constructor rejects null `nextHidden` then null `nextCell`, using the component name
as the `NullPointerException` message. A success retains both exact references without copy,
mutation, descriptor inspection, relationship validation, expression creation, or Tensor-ID
allocation. The record uses ordinary record value equality and exposes no other declared public
method or nested type beyond generated component accessors, `equals`, `hashCode`, and `toString`.

## State schema and ownership

| Kind | Local name | Shape orientation | Type | `requiresGrad` | Initialized policy |
|---|---|---|---|---|---|
| Parameter | `inputWeight` | `[4 * hiddenSize, inputSize]` | exact configured floating type | `true` | Glorot uniform from explicit source |
| Parameter | `hiddenWeight` | `[4 * hiddenSize, hiddenSize]` | same exact type | `true` | Glorot uniform from the same explicit source |
| Parameter, optional | `bias` | `[4 * hiddenSize]` | same exact type | `true` | exact typed zero across all four gates |

Axis zero of each matrix and axis zero of bias use these contiguous intervals:

| Interval | Gate |
|---|---|
| `[0, hiddenSize)` | input |
| `[hiddenSize, 2 * hiddenSize)` | forget |
| `[2 * hiddenSize, 3 * hiddenSize)` | candidate |
| `[3 * hiddenSize, 4 * hiddenSize)` | output |

This fixed gate order is part of the parameter and state-dictionary checkpoint schema. No direct
compatibility is promised with frameworks using a different gate order, separate bias vectors,
recurrent bias, or a non-zero forget-bias default.

For supplied construction, `inputWeight` axis zero must be positive and divisible by four.
`hiddenSize` is exactly that static extent divided by four. `hiddenWeight` axis zero and optional
bias axis zero must structurally equal the complete packed extent; `hiddenWeight` axis one must
structurally equal the derived `hiddenSize`. `inputSize` is `inputWeight` axis one. Both logical
sizes are positive.

Direct discovery and state-dictionary order are `inputWeight`, `hiddenWeight`, then optional
`bias`. Under a future parent child name `cell`, paths become `cell.inputWeight`,
`cell.hiddenWeight`, and optional `cell.bias`. Forward input, current/next hidden, current/next
cell, gates, and candidate never appear in parameter, buffer, child, or state-dictionary
discovery.

## Supplied construction validation and side effects

Both supplied constructors validate complete state before declaring any parameter.

For the two-Tensor constructor:

1. reject null `inputWeight`, then null `hiddenWeight`;
2. validate input weight floating type, `requiresGrad == true`, rank two, fully static Shape,
   positive packed axis zero, divisibility of packed axis zero by four, then positive input-size
   axis one;
3. validate hidden weight floating type, `requiresGrad == true`, rank two, fully static Shape,
   positive axis zero, then positive axis one;
4. require hidden weight exact data type to equal input weight data type;
5. require hidden weight axis zero to equal the complete input-weight packed extent structurally;
6. require hidden weight axis one to equal the derived hidden-size static Dimension; and
7. declare `inputWeight` then `hiddenWeight` and retain no bias.

For the three-Tensor constructor, reject null `bias` after the two weight null checks and before
schema validation. Apply steps 2–6, then validate bias floating type,
`requiresGrad == true`, rank one, fully static Shape, exact common data type, and exact complete
packed extent in that order. Declare `inputWeight`, `hiddenWeight`, then `bias` only after all
checks pass.

Validation creates no Tensor, producer, storage, random draw, or Tensor identity and never mutates
the supplied values. A failure returns no cell and leaves every supplied Tensor unchanged.

## Initialized construction and side effects

The initialized constructor performs exactly:

1. reject null `dataType`, then null `randomGenerator`;
2. require `inputSize > 0`;
3. require `hiddenSize > 0`;
4. require the data type to be floating;
5. compute `packedHiddenSize = Math.multiplyExact(hiddenSize, 4L)`;
6. construct Shapes `[packedHiddenSize, inputSize]`, `[packedHiddenSize, hiddenSize]`, and, only
   when requested, `[packedHiddenSize]`;
7. obtain each requested Shape's checked known element count in state order and reject a count
   above `Integer.MAX_VALUE` before the first draw or Tensor identity allocation;
8. create input weight through exactly
   `ParameterInitializers.glorotUniform(inputWeightShape, dataType, randomGenerator)`;
9. create hidden weight through exactly
   `ParameterInitializers.glorotUniform(hiddenWeightShape, dataType, randomGenerator)` using the
   same now-advanced source;
10. when requested, create the complete packed bias through exactly
    `ParameterInitializers.zeros(biasShape, dataType)` with no source call; and
11. after all requested leaves exist, declare parameters in exact state-table order.

Caller-controlled null, size, type, multiplication-overflow, checked-count, and Java-array-limit
failures precede every draw and Tensor ID. Weight bounds are computed independently from each
packed matrix's actual `[fanOut, fanIn]` Shape. Bias consumes no draw. No bias slice is replaced,
mutated, or materialized separately, so the forget gate begins with the same exact zero-bias
policy as the other three gates.

A source failure keeps completed source calls and creates no Tensor for the failing weight. If it
occurs during hidden-weight creation, the already created input weight and its ID are not rolled
back, but no cell is returned. Later allocation or identifier failures similarly preserve
completed effects. The caller owns the source; the cell never retains, resets, closes,
synchronizes, splits, seeds, or serializes it.

## Forward validation and side-effect order

`forward(input, hidden, cell)` performs exactly:

1. reject null `input`, then null `hidden`, then null `cell`;
2. read current `inputWeight`, `hiddenWeight`, and optional `bias` bindings exactly once in
   parameter declaration order;
3. revalidate the complete current packed parameter schema and derive exact positive
   `hiddenSize`, `2 * hiddenSize`, `3 * hiddenSize`, and `4 * hiddenSize` bounds with checked
   arithmetic;
4. prevalidate the packed input affine projection under Model linear order: promote input and
   input-weight numeric types, require input rank at least one, reject a proven unequal static
   input-feature contraction, and, when bias is present, promote product and bias types;
5. prevalidate the packed hidden projection: promote hidden and hidden-weight numeric types,
   require hidden rank at least one, and reject a proven unequal static hidden-feature
   contraction;
6. require cell rank at least one and reject a proven unequal static final cell-feature
   Dimension against `hiddenSize`; unresolved final equality remains deferred exactly as for the
   linear contractions;
7. derive both packed projection Shapes without creating a Tensor, then derive the eight gate-
   slice Shapes by replacing the final packed extent with exact static `hiddenSize` while
   preserving every leading Dimension reference;
8. in input, forget, candidate, output order, prevalidate each input-slice plus hidden-slice
   promotion and Shape broadcast, and require the resulting preactivation type to be floating;
9. prevalidate forget-gate times current-cell promotion and broadcast;
10. prevalidate input-gate times candidate promotion and broadcast;
11. prevalidate addition of those two products to derive the exact next-cell type and Shape, then
    require next-cell TANH eligibility;
12. prevalidate output-gate times activated-next-cell promotion and broadcast to derive the exact
    next-hidden type and Shape; and
13. only after all preceding checks succeed, construct the exact formula below.

All parameters are floating, so current same-category promotion requires input, hidden, and cell
to be floating while allowing BFLOAT16/FLOAT32/FLOAT64 widening. A static final feature Dimension
must equal its configured size. An unresolved input or hidden contraction Dimension may remain
deferred under current Model linear rules. Every subsequent gate or state broadcast must still be
locally provable, so an unresolved cell-feature Dimension paired with the static gate width is
rejected before expression creation.

Leading-prefix broadcasting is conservative. Equal Dimensions and static singleton expansion
succeed; incompatible static sizes and locally unprovable symbolic combinations fail before
expression creation. The three input ranks need not equal. Examples:

| Input Shape | Hidden Shape | Cell Shape | Next-hidden / next-cell Shape | Meaning |
|---|---|---|---|---|
| `[inputSize]` | `[hiddenSize]` | `[hiddenSize]` | `[hiddenSize]` | one unbatched cell application |
| `[batch, inputSize]` | `[batch, hiddenSize]` | `[batch, hiddenSize]` | `[batch, hiddenSize]` | matching batch prefix |
| `[batch, inputSize]` | `[hiddenSize]` | `[hiddenSize]` | `[batch, hiddenSize]` | both states broadcast across batch |
| `[outer, 1, inputSize]` | `[batch, hiddenSize]` | `[batch, hiddenSize]` | `[outer, batch, hiddenSize]` | ordinary right-aligned leading broadcast |

These examples describe metadata composition only. `outer` and `batch` are leading coordinates,
not time traversal or a sequence loop.

Every local null/schema/type/rank/static-contraction/slice/broadcast failure consumes no Tensor ID
and creates no expression prefix. Parameter reads, result-Shape/type preflight, and record
validation are not Tensor-expression side effects. Identifier exhaustion during valid
construction may leave an expression prefix; no rollback is attempted.

## Formula, delegation, and provenance

Let `x` be input, `h` be hidden, `c` be cell, `W` be `inputWeight`, `U` be `hiddenWeight`, and
`b` be the optional input-side packed bias. Let `H` be `hiddenSize`. Packed projections are:

```text
P_x = x @ transpose(W) + b       when bias is present
P_x = x @ transpose(W)           otherwise
P_h = h @ transpose(U)
```

Slice the final axes independently in fixed order:

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

The fixed LSTM equations are:

```text
i = sigmoid(x_i + h_i)
f = sigmoid(x_f + h_f)
g = tanh(x_g + h_g)
o = sigmoid(x_o + h_o)
c' = f * c + i * g
h' = o * tanh(c')
```

SIGMOID applies only to input, forget, and output gates. TANH applies to the candidate
preactivation and again to the complete next cell before output gating. There are no peepholes,
projection, clipping, coupled gates, or recurrent bias hidden in these equations. These equations
and the input/forget/candidate/output packing define this API; no equivalence to a differently
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

The no-bias successful chain creates exactly twenty-five fresh Tensors/IDs in order: input-weight
PERMUTE, input MATMUL, hidden-weight PERMUTE, hidden MATMUL, eight SLICE occurrences in the order
shown, input ADD, input SIGMOID, forget ADD, forget SIGMOID, candidate ADD, candidate TANH, output
ADD, output SIGMOID, forget-cell MUL, input-candidate MUL, next-cell ADD, next-cell TANH, and
next-hidden MUL. The biased chain inserts one bias ADD immediately after input MATMUL for
twenty-six total. Constructing the result record allocates no Tensor ID. Tests lock exact ordered
producer references and operation association, not numerical execution.

## Result, replacement, mode, and state ownership

- `nextCell` is the exact ADD expression joining forget-retained old cell and input-selected
  candidate. `nextHidden` is the exact final MUL expression using output gate and TANH of that
  same exact `nextCell` reference.
- Every valid forward returns one fresh `LstmCellForwardResult` containing those references in
  `nextHidden`, `nextCell` order. Repeated calls construct independent expression chains and
  carriers.
- Each forward reads every stable parameter wrapper once in declaration order before local
  validation. Schema-compatible replacement before a call affects that call; replacement after
  an expression is built cannot change its provenance references.
- There is no atomic multi-parameter snapshot or thread-safety guarantee. Callers coordinate
  replacement and forward when one consistent view matters.
- `train()` and `eval()` may propagate inherited mode normally, but forward constructs the same
  formula in either mode and receives no `ForwardContext`.
- The cell registers no `Buffer` or child. Input, hidden, cell, gates, candidate, and returned
  state are caller- or expression-owned values and never module state.
- `LstmCell` is not a `UnaryTensorModule` and cannot appear in `Sequential` by type.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.nn.layers` — owns concrete public NN layers, recurrent cells, and
  cell-specific forward-result values.
- `io.github.pho001.synaptik.nn.module` — supplies direct `Module` ownership and stable
  `Parameter` wrappers without modification.
- `io.github.pho001.synaptik.nn.initialization` — supplies current eager parameter initializers
  without modification.
- Model data-type, Shape, and Tensor packages — supply existing semantics only.

Packages added or changed:

- No package is added. The existing `nn.layers` package gains `LstmCell`,
  `LstmCellForwardResult`, and package documentation for their explicit two-state contract.

Type placement:

- `io.github.pho001.synaptik.nn.layers.LstmCell` — owns one parameterized LSTM step and its exact
  explicit-state public contract.
- `io.github.pho001.synaptik.nn.layers.LstmCellForwardResult` — owns the named shallow result of
  that step because Model has no LSTM semantic producer and the two next-state values are NN
  composition outputs.
- `io.github.pho001.synaptik.nn.layers.LstmCellTest` — owns public surfaces, result validation,
  supplied state, forward/preflight, provenance, replacement, mode, discovery, and exclusions.
- `io.github.pho001.synaptik.nn.layers.LstmCellInitializationTest` — owns initialized constructor,
  source order/bounds, zero-bias policy, metadata, early failures, and non-rollback coverage.

## Affected files

Expected production files:

- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/LstmCell.java`.
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/LstmCellForwardResult.java`.
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/package-info.java`.

Expected test files:

- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/LstmCellTest.java`.
- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/LstmCellInitializationTest.java`.

Expected documentation and planning files:

- `docs/glossary.md`.
- `docs/planning/extensions/nn/master-plan.md`.
- this task specification.

No other file may change for this task. In particular, no Model, Training, architecture, Gradle,
dependency-test, sequence, CPU, global-roadmap, or concurrent-work path is authorized.

## Maximum scope

This task may create or modify exactly the eight listed paths and at most:

- three production Java files;
- two NN test files; and
- three documentation/planning files.

If implementation needs another public type, helper file, test owner, Model API, initializer,
dependency, architecture change, or ninth path, stop and propose a separate follow-up task.

## Acceptance criteria

- `LstmCell` is final, extends `Module` directly, and exposes exactly the seven declared public
  members with no additional public/protected surface or nested type.
- `LstmCellForwardResult` is the exact public final record described above, rejects null
  components in order, and retains exact references without Tensor-expression side effects.
- Supplied constructors retain exact caller Tensors and fully prevalidate the packed schema before
  declaration. Initialized construction uses exact checked packed sizes, state order, Glorot
  calls, optional all-zero bias, source lifecycle, and failure side effects specified above.
- Parameter names, order, Shapes, exact type, gradient eligibility, accessors, recursive paths,
  state-dictionary entries, and absence of buffers/children are exact.
- Gate order is input, forget, candidate, output. Bias is optional, packed, and input-side only.
  Gate activations, next-cell update, and output gate use the exact formula and association above.
  Tests use provenance to distinguish these choices from plausible alternatives.
- Forward rejects every locally knowable invalid request before the first Tensor expression,
  snapshots each current binding once, creates exactly the twenty-five- or twenty-six-Tensor chain
  in documented order, and returns exact final MUL/ADD references as next hidden/cell.
- Valid rank-one and higher inputs use ordinary conservative right-aligned leading broadcasting.
  Final feature checks, mixed floating promotion, both output Shapes/types, slice intervals, and
  failure order match current Model semantics.
- Replacement affects later calls only; existing expressions retain old exact parameter
  references. Mode changes do not affect the expression chain.
- No hidden recurrent state, unary adapter, `Sequential` change, context, RNG state, sequence/time
  behavior, scan, Model/initializer change, optimizer, execution behavior, or framework-
  compatibility claim is introduced.
- Focused tests cover exact reflection surfaces, record behavior, state schema/order, supplied
  validation, initialized draw/ID order, all-zero forget bias, early failures, all gate intervals,
  exact operation/provenance association, broadcasting/type/rank/feature failures, replacement
  snapshots, mode independence, explicit two-state threading, discovery, and exclusions.
- All new and affected public/package APIs have meaningful complete Javadoc for purpose,
  equations, packing, Shapes/types, ownership, nullability, source lifecycle, side effects,
  promotion, broadcasting, result semantics, failure order, concurrency, mode, initialization
  policy, checkpoint compatibility limits, and non-execution boundaries.
- A separate documentation-focused clean context finalizes affected Javadoc, package docs,
  glossary impact, planning evidence, no-change conclusions, links, and generated Javadoc in the
  same overall change.

## Tests / validation

Implementation pass runs focused tests while developing and, after executable Java stabilizes:

```bash
./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.layers.LstmCellTest --tests io.github.pho001.synaptik.nn.layers.LstmCellInitializationTest
./gradlew :extensions:nn:test
```

The second command is the sole authoritative final affected-module Java validation. It covers the
existing NN suite plus the new cell. Model tests are not repeated: current exhaustive tests remain
authoritative for primitive linear, slice, ADD, MUL, SIGMOID, TANH, promotion, and broadcast
semantics, while focused NN tests lock cell composition and preflight.

Documentation pass runs after final Javadoc edits:

```bash
./gradlew :extensions:nn:javadoc
git diff --check
git diff --no-index --check /dev/null extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/LstmCell.java
git diff --no-index --check /dev/null extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/LstmCellForwardResult.java
git diff --no-index --check /dev/null extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/LstmCellTest.java
git diff --no-index --check /dev/null extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/LstmCellInitializationTest.java
```

The documentation pass also validates generated type/package pages; exact public and private
surfaces; Model/NN/JDK-only imports; the unchanged sole NN Model dependency; Markdown links,
anchors, fences, final newlines, and trailing whitespace; exact eight-path scope; task/master
status synchronization; exactly one In progress NN row/spec during implementation; and absence of
a task file for 0015. It changes synchronized status to Complete only after every final gate
passes. It reuses successful implementation tests unless executable Java changes afterward or it
records a concrete reason to rerun them.

Repository-wide, architecture, Model, Training, CPU, compiler, backend-conformance, and integration
tests are deferred to the recurrent NN milestone checkpoint or CI. This task changes one
Model-only extension API and no dependency, architecture boundary, shared build, or execution
contract.

## Dependencies

- NN tasks 0001–0013 are Complete and provide direct Module ownership, stable schema-validated
  parameters, explicit-source initialization, deterministic discovery/state dictionaries,
  unary-composition exclusions, and recurrent validation/state-threading precedents.
- Model `Tensor.linear`, `sliceAxis`, `add`, `mul`, `sigmoid`, and `tanh`, same-category promotion,
  `ShapeBroadcast`, immutable descriptors/provenance, and ID allocation are Complete.
- Current independent slice expressions truthfully retain exact packed projection provenance; no
  shared split API is required. Concat is not used because initialized parameters remain direct
  eager leaves and gate packing is one schema, not a runtime composition.
- `ParameterInitializers.glorotUniform` and `zeros` provide the complete selected initialized
  policy. No new initializer is required because all-zero bias is explicit.
- ADR 0007 and existing dependency tests already permit Model-only NN composition.
- Compiler, Training, CPU, Runtime, Prepare, Engine, backend, and numerical execution support are
  not prerequisites because this task constructs parameter leaves and expression metadata only.

## Follow-up tasks

- NN 0015 remains Draft without a detailed task: decide static unrolling of concrete cell calls
  versus a genuinely general recurrent Model scan. `CUM_SUM` and `CUM_PROD` do not satisfy that
  need.
- NN 0015 must reserve packed variable-length sequencing for explicit lengths or an explicit mask,
  never inference from zero-valued data. Padded steps cause no cell invocation. Active-batch
  compaction or stable sorting must restore original batch order and capture final hidden state,
  plus final cell state for LSTM, exactly when each sequence leaves the active set.
- If NN 0015 selects containers, prefer cell-specific types unless completed RNN/GRU/LSTM
  signatures prove a shared recurrent contract without casts, state erasure, duplicate carriers,
  or hidden state.
- A non-zero forget-bias default, alternate LSTM packing/bias/equation policy, or framework
  checkpoint conversion requires a concrete consumer and separate task.
- Truncation, masking implementation, bidirectionality, stacking, recurrent dropout, checkpoint
  persistence, training sessions, and backend execution remain separate future work.

## Documentation and no-change review

Document profiles:

- Java/package Javadoc: General plus API/Javadoc.
- glossary: General reference style.
- task/master plan: General plus Planning.

Required implementation-phase documentation changes are `LstmCell`, `LstmCellForwardResult`, and
layers-package Javadocs, the glossary's explicit-state recurrent wording, and synchronized NN
planning records.

The separate documentation pass must verify and record these reasoned no-change conclusions:

- `ARCHITECTURE.md`, focused architecture pages, ADR 0007, and architecture tests remain accurate
  because state ownership, module direction, and lifecycle boundaries do not change.
- Tensor and Compile APIs plus Model source/master/capabilities remain accurate because the cell
  composes existing expressions and adds no Tensor method, semantic kind, split, recurrent scan,
  capture, derivative, or execution behavior.
- Training API and training graph remain accurate because recursive parameter discovery and
  replacement consume this direct Module normally; explicit recurrent states are forward values,
  and no optimizer, session, gradient publication, backpropagation-through-time, truncation, or
  state orchestration is added.
- `RnnCell`, `GruCell`, `DropoutForwardResult`, `UnaryTensorModule`, `Sequential`, state-dictionary
  contracts, and their tests remain accurate. LSTM has a distinct carrier because it returns two
  different state expressions; transient gates and recurrent values are not module state.
- Gradle and dependency rules/tests remain accurate because `extensions/nn` retains only its
  existing Model dependency.
- Backend conformance and integration tests remain unnecessary because no numerical execution,
  backend capability, or end-to-end lifecycle changes.
- The global roadmap, CPU work, other modules, and Draft NN 0015 work remain untouched.

## Architecture impact

Expected impact: None.

This task realizes the existing NN responsibility for parameter-owning layer composition while
keeping both recurrent states explicit. If implementation requires hidden module/runtime state, a
universal recurrent abstraction, a general scan, another module dependency, or an architecture
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

## Documentation-agent handoff

After executable Java/tests stabilize, give the clean documentation context:

- this task and exact eight-path limit;
- the final executable diff and exact focused/final NN commands, counts, and results;
- exact class/record surfaces, direct-Module/Sequential exclusion, packed state names/order/
  Shapes/types, zero-bias policy, constructor prevalidation, and draw/ID order;
- full three-input preflight, batch-broadcast examples, fixed equations and activation placement,
  twenty-five-/twenty-six-ID producer chains, exact result references, mode behavior, replacement
  snapshots, and failure effects;
- directly relevant architecture/ADR, documentation profiles, final NN/Model source and tests,
  Tensor/Training APIs, glossary, dependency tests, and planning history;
- the mandate to preserve concurrent glossary work and record every reasoned no-change conclusion;
- generated-Javadoc, reflection/`javap`, import/dependency, Markdown, exact-scope/status,
  task-0015-absence, newline, and whitespace gates; and
- the required completion-summary and Status format from `AGENTS.md`.

## Local decisions

- Use `LstmCell`, following project acronym style (`RnnCell`, `GruCell`), and place it beside the
  concrete recurrent cells. The name represents one cell step, not sequence traversal.
- Extend `Module` directly. Three Tensor inputs are intrinsic; a unary adapter would have to hide
  input state or retain recurrent state.
- Add public `LstmCellForwardResult(nextHidden, nextCell)`. The two next states are distinct
  expressions and both are required for truthful caller threading. A public record follows the
  existing NN forward-result convention while keeping the carrier cell-specific.
- Pack two matrices and optional bias in input, forget, candidate, output order. Independent
  slices make provenance exact with current Model APIs and keep the parameter surface as small as
  `GruCell`. This order is the checkpoint schema and is not claimed to match another framework.
- Use one optional packed input-side bias and no recurrent bias. This is the smallest unambiguous
  policy supported by one biased packed linear call.
- Initialize the complete bias to zero. Current initializers offer whole-Tensor zero/one leaves,
  not a direct packed leaf with a selected forget interval set to one. Hidden host mutation or a
  slice/concat-derived parameter would violate the selected eager leaf and provenance policy.
- Fix standard uncoupled equations with SIGMOID input/forget/output gates, TANH candidate, additive
  cell update, and output-gated TANH of next cell. Peepholes, projection, clipping, and configurable
  activation placement require separate capabilities.
- Use Glorot uniform independently for the actual packed matrix Shapes. No orthogonal initializer
  exists, and this task does not introduce one.
- Permit current Model mixed floating promotion and conservative leading broadcasting, as the
  completed recurrent cells do. Preflight duplicates only local public algebra needed to prevent
  a late gate/state failure from leaving a projection prefix.
- Accept unresolved input and hidden contraction equality exactly as Model linear does. The cell
  state does not participate in linear contraction: its final Dimension must additionally
  broadcast against the static gate width, so an unresolved cell-feature Dimension is rejected
  when that compatibility cannot be proved locally. Fixed packed parameters and slice bounds make
  every gate final axis statically `hiddenSize`.
- Keep all leading axes semantically neutral batch coordinates. Task 0015, not this cell, owns any
  time-axis, active-set, or repeated-invocation contract.

## Known limitations

- Only this input/forget/candidate/output-packed LSTM with positive fully static parameters and
  one optional all-zero input-side packed bias is supported.
- Input, hidden, and cell Shapes may contain unresolved leading or final Dimensions only when
  current local Model rules can represent the projection and broadcast obligations. NN performs
  no binding.
- Higher-rank inputs are one batched cell application, never an implicit sequence.
- Explicit caller-threaded recurrence may build deep provenance chains. This task adds no
  detachment, truncation, loop IR, scheduling, or memory-lifetime policy.
- Parameter replacement and forward are not thread-safe as one snapshot. The cell retains no
  lock, version, or transaction.
- Initialized eager parameters require Java-array-sized host leaves and may consume draws or IDs
  before a later resource failure; effects are not rolled back.
- Forward constructs metadata and proves no numerical values, gradient rules, compiler capture,
  backend support, storage allocation, publication, or execution.

## Validation evidence

- Clean planning context `/root/nn_0014_planning` read the repository instructions, architecture
  contract and current plan, planning guide/roadmap, documentation rules and General/Planning/API-
  Javadoc profiles, NN master and completed task history through 0013, final recurrent/module/
  initializer APIs and tests, Model Tensor/Shape/type/operation contracts, Training API/graph,
  glossary, and build/dependency rules before selecting this contract.
- Planning inspection confirmed current Model linear, independent final-axis slice, ADD, MUL,
  SIGMOID, TANH, promotion, and broadcast APIs can express the complete formula and full local
  preflight. No new Model operation, split, scan, initializer, or dependency is required.
- Planning selected input/forget/candidate/output packing, one input-side packed bias, zero-bias
  initialization, fixed activation placement, additive next-cell update, and the NN-owned
  `LstmCellForwardResult(nextHidden, nextCell)` explicitly; no framework compatibility is claimed.
- Targeted planning validation passed for the two changed planning paths: local links and anchors
  resolve, fences balance, final newlines are present, and trailing whitespace is absent.
- The NN planning state has exactly one Ready row/spec (0014). NN 0015 remains the unchanged Draft
  row with no task file, including its explicit lengths/mask, no-zero-value-inference, active-
  batch compaction, original-order restoration, final hidden/cell capture, static-versus-dynamic
  packing, and scan/unroll decision boundaries.
- Every unrelated concurrent CPU/source/test/documentation/master/task/roadmap/glossary change was
  preserved exactly. Whole-worktree `git diff --check` and the new-file no-index whitespace check
  passed; no Java, Javadoc, Gradle, or test command was run during this planning-only task.
- Clean implementation context `/root/nn_0014_implementation` added the exact direct-Module
  `LstmCell`, the exact two-component `LstmCellForwardResult`, two focused test suites, and draft
  type/package Javadocs. It required no Model helper, initializer, hidden Buffer/state,
  `ForwardContext`, sequence/scan abstraction, dependency, or ninth task path.
- The stabilized focused command
  `./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.layers.LstmCellTest --tests io.github.pho001.synaptik.nn.layers.LstmCellInitializationTest`
  passed 15 tests with no failures, errors, or skips. The initial development run exposed one
  incorrect test expectation for an unresolved cell-feature Dimension against a static gate
  extent; current conservative Model broadcasting correctly rejects that unprovable pair before
  expression creation, and the corrected test locks that no-ID preflight behavior.
- Focused provenance inspection covers the exact twenty-five-/twenty-six-ID chain, independent
  input/forget/candidate/output slices from both packed projections, input-side-only bias,
  SIGMOID/TANH placement, additive next-cell update, output-gated activated next cell, exact
  result references, mixed-floating promotion, batch broadcasting, mode independence,
  replacement snapshots, and explicit two-state threading without retained state.
- After the exact result-record surface assertion was finalized, the final authoritative
  `./gradlew :extensions:nn:test` run passed 23 suites and 154 tests with no skips, failures, or
  errors. Executable Java and tests remained unchanged afterward.
- Preliminary `./gradlew :extensions:nn:javadoc` passed. Generated LSTM cell, result, and package
  pages are present. Independent `javap -protected` and `javap -private` inspection confirmed the
  exact final direct-Module public surface, exact generated record surface, only three private
  stable parameter-wrapper fields on the cell, and only the two private final Tensor components
  on the result.
- Source/import and build inspection found only Model, NN, and JDK imports and confirmed the
  unchanged sole production dependency on `:modules:model`. No Training, Compiler, Runtime,
  Prepare, Engine, backend, context, buffer, recurrent-state field, or retained RNG appears.
- Scope/status inspection found only the seven currently changed NN 0014 paths, with the glossary
  left untouched for the independent documentation pass. Task/master are synchronized at In
  progress; NN 0015 remains the unchanged Draft row with no task file and its packed-sequence
  wording preserved. All unrelated CPU, roadmap, and existing glossary changes remain present.
- Whole-worktree `git diff --check` passed. Each of the four untracked Java/test files passed
  `git diff --no-index --check /dev/null <path>` with the expected content-difference status and
  no whitespace diagnostic; all four end with one newline.
- Independent clean documentation context `/root/nn_0014_docs` read the final implementation and
  tests, architecture/planning/documentation contracts, complete NN task lineage, recurrent/
  module/initializer/unary boundaries, Model expression/Shape/type contracts, Training API and
  graph, glossary, and build/dependency rules. It found no executable defect, architecture
  uncertainty, scope blocker, or reason to change executable Java or tests.
- The documentation context finalized `LstmCell`, `LstmCellForwardResult`, and layers-package
  Javadocs. It clarified that unresolved input/hidden contraction equality may remain deferred by
  Model linear while every gate/state broadcast must be locally provable, so an unresolved cell-
  feature Dimension paired with static `hiddenSize` is rejected before expression creation. This
  matches the stabilized implementation and focused test rather than changing behavior.
- The glossary now defines the current LSTM cell, input/forget/candidate/output packed checkpoint
  schema, input-side-only bias, all-zero initialized bias including forget, two-state formula and
  exact result order, caller-threaded ownership, direct-Module/Sequential exclusion, and the
  distinctions among vanilla RNN, GRU, gate packing, future packed variable-length sequences,
  recurrent scan, and cumulative scan.
- The documentation pass reused the implementation context's focused two-suite/15-test result and
  authoritative 23-suite/154-test NN result because no executable Java or test changed afterward.
  Javadoc and prose edits do not stale that evidence, so no Java test suite was repeated.
- Final `./gradlew :extensions:nn:javadoc` passed after the Javadoc edits (`BUILD SUCCESSFUL`; 3
  actionable tasks, 2 executed and 1 up-to-date). Inspection of generated `LstmCell.html`,
  `LstmCellForwardResult.html`, and `layers/package-summary.html` confirmed the exact signatures,
  parameters, returns, failures, packing, initialization/source lifecycle, equations, two-state
  ownership, broadcast/preflight boundary, mode/replacement semantics, Sequential exclusion, and
  non-execution limits.
- Final `javap -protected` and `javap -private` confirmed direct final `Module`, exactly three
  public constructors and four declared public methods, no protected surface or nested type, only
  three private final parameter-wrapper fields, the exact generated record surface, and only two
  private final Tensor components in next-hidden/next-cell order. A standalone reflection program
  compiled and printed `LSTM reflection surface: PASS`. An exploratory JShell check independently
  printed the record result but was non-authoritative because one expression had a `Module` name
  ambiguity and JShell then failed to persist Preferences history; neither event changed a file.
- Source/import and build inspection found only Model, existing NN, and JDK production imports and
  retained the sole `implementation(project(":modules:model"))` dependency. No Training,
  Compiler, Runtime, Prepare, Engine, backend, `ForwardContext`, `GraphRngState`, `Buffer`, hidden
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
  execution contract.
- Training API and the training graph require no change because recursive parameter discovery and
  replacement already consume this direct Module and the task adds no optimizer, session,
  gradient publication, backpropagation-through-time, truncation, or state orchestration.
  `RnnCell`, `GruCell`, `DropoutForwardResult`, `UnaryTensorModule`, `Sequential`, state-dictionary
  contracts, and their tests remain accurate: LSTM alone needs the distinct two-state carrier,
  while transient gates and recurrent values never become module state.
- Gradle/dependency rules and tests, backend conformance, and integration tests require no change
  because no dependency, numerical execution, backend capability, or end-to-end lifecycle changed.
  The LSTM task did not edit the global roadmap, CPU work, other modules, Model capabilities, or
  Draft NN 0015; their concurrent worktree changes were preserved exactly.

## Implementation notes

Clean implementation context `/root/nn_0014_implementation` completed the executable source and
focused tests in the planned paths. The focused two-suite/15-test selection and sole authoritative
23-suite/154-test NN module run pass. Independent clean documentation context
`/root/nn_0014_docs` then finalized type/package Javadocs, glossary and planning evidence, and
passed generated-Javadoc, rendered-page, `javap`/reflection, dependency/import, Markdown, exact
eight-path scope, status, newline, whitespace, no-index, and final-diff gates without changing
executable behavior or repeating the stable Java suites.

## Completion summary

- Completed changes: added final direct-Module `LstmCell` with exact supplied and explicit-source
  construction, IFGO-packed state, complete forward preflight, fixed two-state formula, exact
  result carrier, and focused contract tests; finalized every affected documentation contract.
- Files changed or created: exactly `LstmCell.java`, `LstmCellForwardResult.java`, layers
  `package-info.java`, `LstmCellTest.java`, `LstmCellInitializationTest.java`, `docs/glossary.md`,
  the NN master plan, and this task; unrelated concurrent paths were preserved.
- Tests and validation: reused the passing focused 2-suite/15-test and authoritative
  23-suite/154-test NN evidence with zero failures, errors, or skips. Final NN Javadoc/generated-
  page inspection, `javap`, standalone reflection, import/dependency, Markdown, exact-scope/
  status, newline/whitespace, no-index, and whole-diff checks passed.
- Documentation-agent review: clean context `/root/nn_0014_docs` completed the mandatory
  independent source/test/API/architecture review and changed no executable Java or test.
- Documentation impact: finalized the LSTM/result/package Javadocs, added the glossary's current
  LSTM checkpoint schema and explicit two-state example plus sequence/scan distinctions, and
  synchronized task/master status and evidence.
- Javadoc review: complete; generated type/result/package pages contain the exact surface, state,
  initialization, source, equation, failure, reference, concurrency, mode, and non-execution
  contracts.
- Glossary impact: added the reusable LSTM and cell-state terms, IFGO packed schema, zero-bias
  policy, exact next-hidden/next-cell order, and distinctions from RNN, GRU, future packed
  sequences, recurrent/cumulative scan, and `Sequential`.
- Unresolved issues: None.
- Follow-up required: None for task 0014; NN 0015 remains separate Draft work.

Status: Complete

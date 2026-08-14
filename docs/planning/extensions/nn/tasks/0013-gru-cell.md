# Task 0013: GRU Cell

## Status

Complete

## Goal

Add one final public gated recurrent unit (GRU) cell with explicit caller-threaded hidden state.
The cell owns two gate-major packed projection matrices and one optional packed input-side bias,
constructs a fixed reset-after GRU step entirely from current Model Tensor expressions, and
returns the next hidden Tensor as its sole result. It retains no recurrent state.

Mental model:

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

## Scope

- Add final public `io.github.pho001.synaptik.nn.layers.GruCell` extending
  `io.github.pho001.synaptik.nn.module.Module` directly.
- Add exactly the constructors, parameter accessors, and two-input forward method in the public
  API table below. Add no overload, builder, options object, gate enum, state carrier, size getter,
  functional facade, or configurable equation policy.
- Declare positive fully static rank-two parameters under exact local names `inputWeight` then
  `hiddenWeight`. Their Shapes are `[3 * hiddenSize, inputSize]` and
  `[3 * hiddenSize, hiddenSize]`.
- Pack both matrices gate-major on axis zero in exact reset, update, candidate order. After linear
  projection, that same order occupies the final result axis.
- Optionally declare one positive fully static rank-one parameter under exact local name `bias`,
  with Shape `[3 * hiddenSize]`. It is added only by the packed input projection. There is no
  recurrent bias and no separate per-gate bias parameter.
- Require all parameters to share one exact floating data type and have
  `requiresGrad == true`. Retain caller-supplied parameter Tensors exactly.
- Provide supplied-state construction both without and with bias. Null never means absent bias;
  callers select absence through the two-Tensor constructor.
- Provide one initialized constructor with explicit positive `inputSize`, positive `hiddenSize`,
  bias presence, floating `DataType`, and caller-owned `RandomGenerator`.
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
  source and interval visible without adding a Model API or constructing per-gate parameters.
- Support input and hidden rank one or higher. Their final Dimensions contract respectively with
  `inputSize` and `hiddenSize` under current Model linear semantics.
- Treat all leading Dimensions as ordinary right-broadcastable batch metadata. No leading axis is
  named, interpreted, or traversed as time.
- Complete all caller-controlled forward validation before creating the first projection
  expression. Then construct exactly the formula and producer order specified below.
- Keep the cell mode-insensitive. Forward accepts no `ForwardContext`, does not inspect `mode()`,
  and does not alter module mode or state.
- Preserve stable parameter wrappers, schema-compatible replacement snapshots, recursive
  discovery, and state-dictionary paths through current `Module` and `Parameter` contracts.
- Add focused exact-surface, supplied-state, initialization, validation-order, shape/type,
  gate-slicing, provenance, replacement, mode, state-discovery, and exclusion tests.
- Add complete type, constructor, member, and package Javadocs. After executable work and final NN
  testing, use a separate clean documentation-focused context to finalize Javadocs, glossary
  impact, planning evidence, generated Javadoc, and reasoned no-change conclusions.

## Out of scope

- Retaining the current or next hidden Tensor in a field, `Buffer`, child, state dictionary,
  thread-local, static, runtime object, session, or any other hidden lifecycle.
- A default or zero hidden state, hidden-state initializer, reset/detach API, stateful forward
  overload, or result carrier that obscures caller state threading.
- `UnaryTensorModule`, participation in `Sequential`, an adapter into `Sequential`, or any change
  to `Sequential`, `UnaryTensorModule`, `Module`, `Parameter`, or `Buffer`.
- Separate public gate parameters, six per-gate matrices, separate input/recurrent biases,
  per-gate bias accessors, or a parameter-packing configuration.
- Reset-before candidate projection, reset applied to raw hidden state before its candidate
  matrix, alternate update interpolation, gate-order configuration, framework-compatibility mode,
  or migration/conversion helpers for another GRU layout.
- A recurrent sequence container, loop, step counter, time-axis convention, masks, sequence
  lengths, packed sequence, variable-length handling, bidirectionality, stacking, recurrent
  dropout, static unrolling, or time traversal.
- A general recurrent Model scan, loop operation, subgraph/body representation, carried-value
  tuple, scan result, or Tensor API. Current `CUM_SUM` and `CUM_PROD` are associative cumulative
  operations and are not recurrent scan primitives.
- LSTM state, peepholes, projections, layer normalization, clipping, residuals, attention,
  convolutional recurrence, or another recurrent-cell family.
- Dynamic or zero-sized parameters, integral/BOOL parameters, frozen parameters, quantized or
  sparse state, custom initializer objects, orthogonal initialization, or a retained/default RNG.
- `ForwardContext`, `GraphRngState`, forward-time `RandomGenerator`, mode-dependent branching,
  Buffer transitions, or stochastic recurrence.
- Backpropagation through time, gradient detachment, gradient rules, optimizer/session,
  parameter groups, checkpoint transport, serialization, compiler capture, scheduling,
  Runtime/Prepare/Engine behavior, backend lowering, kernels, numerical execution, or end-to-end
  support claims.
- A Model, Training, Gradle, dependency, architecture-contract, ADR, architecture-test, global-
  roadmap, CPU, or explanatory-architecture source change during implementation.
- Detailed task specifications for NN 0014 or 0015.

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
- [Completed Model task 0017H: Slice expressions](../../../modules/model/tasks/0017h-slice-tensor-expressions.md)
- [Completed Model task 0019D: Linear convenience](../../../modules/model/tasks/0019d-linear-convenience.md)
- [Completed Model task 0023E: Cumulative scan normalization](../../../modules/model/tasks/0023e-cumulative-scan-normalization-and-product.md)

## Architecture constraints

- `extensions/nn` continues to depend only on `modules/model`. Production must not import
  Training, Compiler, Runtime, Prepare, Engine, CPU, Metal, CUDA, or another backend.
- NN owns packed parameter bindings and typed cell composition. Model remains the sole owner of
  LINEAR decomposition, SLICE, ADD, SUB, MUL, SIGMOID, TANH, type promotion, Shape algebra,
  descriptors, provenance, and Tensor identity.
- Tensor identity, descriptors, and provenance remain immutable. Parameter replacement changes
  only the current exact Tensor returned by one stable wrapper; existing expressions remain
  unchanged.
- Recurrent state is a caller-threaded Tensor value, never module-owned persistent state. It is
  not a `Buffer` merely because the caller may pass it between calls.
- `Module` retains no universal forward method. `GruCell` is a direct subclass with a truthful
  two-input signature and must not extend `UnaryTensorModule`.
- `Sequential` remains a container only for `UnaryTensorModule`; no adapter may capture or erase
  the hidden-state input.
- Mode is NN composition metadata. This mode-insensitive cell neither consumes a context nor
  reads inherited mode during forward.
- Packed slices are ordinary independent one-output Model expressions. The task must not invent a
  shared split producer, tuple value, hidden gate cache, direct `Operation` construction, or
  special GRU semantic kind.
- Construction and forward create eager parameter leaves or storage-free expression metadata
  only. They do not prove gradient implementation, graph capture, compiled execution, backend
  support, numerical values, or storage residency.
- If implementation needs a new Model operation/helper/public method, public result type,
  recurrent container, state Buffer, context/RNG input, dependency, architecture rule, or eighth
  task path, stop and report the exact blocker instead of widening the task.

## Public API

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
| two-Tensor constructor | Retains exact positive static packed input/hidden projection weights and declares no bias. |
| three-Tensor constructor | Retains the same exact weights plus one exact packed input-side bias; null bias is invalid. |
| initialized constructor | Uses explicit sizes/type/source, creates packed input weight then packed hidden weight through Glorot uniform and optional zero bias afterward. |
| `inputWeight()` | Returns the exact stable wrapper declared under `inputWeight`. |
| `hiddenWeight()` | Returns the exact stable wrapper declared under `hiddenWeight`. |
| `bias()` | Returns an empty Optional or the exact stable wrapper declared under `bias`. |
| `forward(input, hidden)` | Validates the complete two-input request, snapshots current parameters, constructs the fixed reset-after formula once, and returns the exact next-hidden Tensor. |

The class declares no other public or protected constructor, method, field, nested type,
interface, or overload. Inherited final `Module` APIs remain available normally.

## State schema and ownership

| Kind | Local name | Shape orientation | Type | `requiresGrad` | Initialized policy |
|---|---|---|---|---|---|
| Parameter | `inputWeight` | `[3 * hiddenSize, inputSize]` | exact configured floating type | `true` | Glorot uniform from explicit source |
| Parameter | `hiddenWeight` | `[3 * hiddenSize, hiddenSize]` | same exact type | `true` | Glorot uniform from the same explicit source |
| Parameter, optional | `bias` | `[3 * hiddenSize]` | same exact type | `true` | exact typed zero |

Axis zero of each matrix and axis zero of bias use these contiguous intervals:

| Interval | Gate |
|---|---|
| `[0, hiddenSize)` | reset |
| `[hiddenSize, 2 * hiddenSize)` | update |
| `[2 * hiddenSize, 3 * hiddenSize)` | candidate |

For supplied construction, `inputWeight` axis zero must be positive and divisible by three.
`hiddenSize` is exactly that static extent divided by three. `hiddenWeight` axis zero and optional
bias axis zero must structurally equal the complete packed extent; `hiddenWeight` axis one must
structurally equal the derived `hiddenSize`. `inputSize` is `inputWeight` axis one. Both logical
sizes are positive.

Direct discovery and state-dictionary order are `inputWeight`, `hiddenWeight`, then optional
`bias`. Under a future parent child name `cell`, paths become `cell.inputWeight`,
`cell.hiddenWeight`, and optional `cell.bias`. The input hidden Tensor, gates, candidate, and next
hidden Tensor never appear in parameter, buffer, child, or state-dictionary discovery.

## Supplied construction validation and side effects

Both supplied constructors validate complete state before declaring any parameter.

For the two-Tensor constructor:

1. reject null `inputWeight`, then null `hiddenWeight`;
2. validate input weight floating type, `requiresGrad == true`, rank two, fully static Shape,
   positive packed axis zero, divisibility of packed axis zero by three, then positive input-size
   axis one;
3. validate hidden weight floating type, `requiresGrad == true`, rank two, fully static Shape,
   positive axis zero, then positive axis one;
4. require hidden weight exact data type to equal input weight data type;
5. require hidden weight axis zero to equal the complete input-weight packed extent structurally;
6. require hidden weight axis one to equal the derived hidden-size static Dimension; and
7. declare `inputWeight` then `hiddenWeight` and retain no bias.

For the three-Tensor constructor, reject null `bias` after the two weight null checks and before
schema validation. Apply steps 2–6, then validate bias floating type, `requiresGrad == true`, rank
one, fully static Shape, exact common data type, and exact complete packed extent in that order.
Declare `inputWeight`, `hiddenWeight`, then `bias` only after all checks pass.

Validation creates no Tensor, producer, storage, random draw, or Tensor identity and never mutates
the supplied values. A failure returns no cell and leaves every supplied Tensor unchanged.

## Initialized construction and side effects

The initialized constructor performs exactly:

1. reject null `dataType`, then null `randomGenerator`;
2. require `inputSize > 0`;
3. require `hiddenSize > 0`;
4. require the data type to be floating;
5. compute `packedHiddenSize = Math.multiplyExact(hiddenSize, 3L)`;
6. construct Shapes `[packedHiddenSize, inputSize]`, `[packedHiddenSize, hiddenSize]`, and, only
   when requested, `[packedHiddenSize]`;
7. obtain each requested Shape's checked known element count in state order and reject a count
   above `Integer.MAX_VALUE` before the first draw or Tensor identity allocation;
8. create input weight through exactly
   `ParameterInitializers.glorotUniform(inputWeightShape, dataType, randomGenerator)`;
9. create hidden weight through exactly
   `ParameterInitializers.glorotUniform(hiddenWeightShape, dataType, randomGenerator)` using the
   same now-advanced source;
10. when requested, create bias through exactly
    `ParameterInitializers.zeros(biasShape, dataType)` with no source call; and
11. after all requested leaves exist, declare parameters in exact state-table order.

Caller-controlled null, size, type, multiplication-overflow, checked-count, and Java-array-limit
failures precede every draw and Tensor ID. Weight bounds are computed independently from each
packed matrix's actual `[fanOut, fanIn]` Shape. Bias consumes no draw.

A source failure keeps completed source calls and creates no Tensor for the failing weight. If it
occurs during hidden-weight creation, the already created input weight and its ID are not rolled
back, but no cell is returned. Later allocation or identifier failures similarly preserve
completed effects. The caller owns the source; the cell never retains, resets, closes,
synchronizes, splits, seeds, or serializes it.

## Forward validation and side-effect order

`forward(input, hidden)` performs exactly:

1. reject null `input`, then null `hidden`;
2. read current `inputWeight`, `hiddenWeight`, and optional `bias` bindings exactly once in
   parameter declaration order;
3. prevalidate the packed input affine projection under Model linear order: promote input and
   input-weight numeric types, require input rank at least one, reject a proven unequal static
   input-feature contraction, and, when bias is present, promote product and bias types;
4. prevalidate the packed hidden projection: promote hidden and hidden-weight numeric types,
   require hidden rank at least one, and reject a proven unequal static hidden-feature contraction;
5. derive both packed projection Shapes without creating a Tensor: preserve complete leading
   prefixes and append the exact static packed extent;
6. derive the six gate-slice Shapes by replacing the final packed extent with exact static
   `hiddenSize`, preserving every leading Dimension reference from its projection;
7. prevalidate reset preactivation promotion and Shape broadcasting for input-reset plus
   hidden-reset, then the same for update;
8. prevalidate reset-gate times hidden-candidate promotion and broadcast, then candidate-input
   plus that reset product, followed by candidate TANH eligibility;
9. prevalidate hidden minus candidate promotion and broadcast, update times that difference, and
   candidate plus the weighted difference; and
10. only after all preceding checks succeed, construct the exact formula below.

All parameters are floating, so current same-category promotion requires both call inputs to be
floating while allowing BFLOAT16/FLOAT32/FLOAT64 widening. A static final input or hidden feature
Dimension must equal its configured size. An unresolved final feature Dimension is accepted under
current Model linear semantics and leaves that obligation for later compiler validation or
binding.

Leading-prefix broadcasting is conservative. Equal Dimensions and static singleton expansion
succeed; incompatible static sizes and locally unprovable symbolic combinations fail before
expression creation. Input and hidden ranks need not equal. Examples:

| Input Shape | Hidden Shape | Next-hidden Shape | Meaning |
|---|---|---|---|
| `[inputSize]` | `[hiddenSize]` | `[hiddenSize]` | one unbatched cell application |
| `[batch, inputSize]` | `[batch, hiddenSize]` | `[batch, hiddenSize]` | matching batch prefix |
| `[batch, inputSize]` | `[hiddenSize]` | `[batch, hiddenSize]` | hidden vector broadcast across batch |
| `[outer, 1, inputSize]` | `[batch, hiddenSize]` | `[outer, batch, hiddenSize]` | ordinary right-aligned batch broadcasting |

These examples describe metadata composition only. `outer` and `batch` are leading coordinates,
not a time traversal or sequence loop.

Every local null/type/rank/static-contraction/slice/broadcast failure consumes no Tensor ID and
creates no expression prefix. Parameter reads and temporary Shape/type values are not expression
side effects. Identifier exhaustion during valid construction may leave an expression prefix; no
rollback is attempted.

## Formula, delegation, and provenance

Let `x` be input, `h` be hidden, `W` be `inputWeight`, `U` be `hiddenWeight`, and `b` be the
optional input-side packed bias. Let `H` be `hiddenSize`. Packed input projection `P_x` and packed
hidden projection `P_h` are:

```text
P_x = x @ transpose(W) + b       when bias is present
P_x = x @ transpose(W)           otherwise
P_h = h @ transpose(U)
```

Slice the final axes independently in fixed order:

```text
x_r = P_x[..., 0:H]
x_z = P_x[..., H:2H]
x_n = P_x[..., 2H:3H]
h_r = P_h[..., 0:H]
h_z = P_h[..., H:2H]
h_n = P_h[..., 2H:3H]
```

The fixed reset-after GRU equations are:

```text
r = sigmoid(x_r + h_r)
z = sigmoid(x_z + h_z)
n = tanh(x_n + r * h_n)
h' = n + z * (h - n)
```

The reset gate therefore multiplies the recurrent candidate projection `h_n` after its matrix
projection. Because no recurrent bias exists, nothing is hidden inside that product. The update
gate is a retention gate: `z == 1` selects old hidden state and `z == 0` selects the candidate.
These equations define this API; no equivalence to a differently packed or differently biased
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
the static schema; no expression helper constructs `Operation`, `TensorDescriptor`,
`TensorProvenance`, or a derived Tensor directly.

The no-bias successful chain creates exactly twenty fresh Tensors/IDs in order: input-weight
PERMUTE, input MATMUL, hidden-weight PERMUTE, hidden MATMUL, six SLICE occurrences in the order
shown, reset ADD, reset SIGMOID, update ADD, update SIGMOID, reset MUL, candidate ADD, candidate
TANH, hidden-minus-candidate SUB, update MUL, and final ADD. The biased chain inserts one bias ADD
immediately after input MATMUL for twenty-one total. Tests must lock exact ordered producer
references and operation association, not numerical execution.

## Replacement snapshots, mode, and state ownership

- Each forward call reads each stable parameter wrapper once in declaration order before local
  validation. Schema-compatible replacement before a call affects that call; replacement after
  an expression is built cannot change its exact provenance references.
- There is no atomic multi-parameter snapshot or thread-safety guarantee. Callers coordinate
  replacement and forward when one consistent view matters.
- `train()` and `eval()` may propagate inherited mode normally, but forward constructs the same
  formula in either mode and receives no `ForwardContext`.
- The cell registers no `Buffer` or child. Hidden input, gates, candidate, and result are caller-
  or expression-owned values and never module state.
- `GruCell` is not a `UnaryTensorModule` and cannot appear in `Sequential` by type.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.nn.layers` — owns concrete public NN layers and recurrent cells.
- `io.github.pho001.synaptik.nn.module` — supplies direct `Module` ownership and stable
  `Parameter` wrappers without modification.
- `io.github.pho001.synaptik.nn.initialization` — supplies current eager parameter initializers
  without modification.
- Model data-type, Shape, and Tensor packages — supply existing semantics only.

Packages added or changed:

- No package is added. The existing `nn.layers` package gains `GruCell` and package documentation
  is extended for its explicit-state contract.

Type placement:

- `io.github.pho001.synaptik.nn.layers.GruCell` — owns one parameterized GRU step and its exact
  explicit-state public contract.
- `io.github.pho001.synaptik.nn.layers.GruCellTest` — owns public surface, supplied state,
  forward/preflight, provenance, replacement, mode, discovery, and exclusion coverage.
- `io.github.pho001.synaptik.nn.layers.GruCellInitializationTest` — owns initialized constructor,
  source order/bounds, metadata, early-failure, and non-rollback coverage.

## Affected files

Expected production files:

- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/GruCell.java`.
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/package-info.java`.

Expected test files:

- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/GruCellTest.java`.
- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/GruCellInitializationTest.java`.

Expected documentation and planning files:

- `docs/glossary.md`.
- `docs/planning/extensions/nn/master-plan.md`.
- this task specification.

No other file may change for this task. In particular, no Model, Training, architecture, Gradle,
dependency-test, sequence, LSTM, CPU, global-roadmap, or concurrent-work path is authorized.

## Maximum scope

This task may create or modify exactly the seven listed paths and at most:

- two production Java files;
- two NN test files; and
- three documentation/planning files.

If implementation needs another public type, helper file, test owner, Model API, dependency,
architecture change, or eighth path, stop and propose a separate follow-up task.

## Acceptance criteria

- `GruCell` is final, extends `Module` directly, and exposes exactly the seven declared public
  members with no additional public/protected surface or nested type.
- Supplied constructors retain exact caller Tensors and fully prevalidate the packed schema before
  declaration. Initialized construction uses exact checked packed sizes, state order, Glorot
  calls, optional zero bias, source lifecycle, and failure side effects specified above.
- Parameter names, order, Shapes, exact type, gradient eligibility, accessors, recursive paths,
  state-dictionary entries, and absence of buffers/children are exact.
- Gate order is reset, update, candidate. Bias is optional, packed, and input-side only. The
  candidate uses reset-after projection and the output uses retention interpolation exactly as
  specified. Tests use provenance to distinguish these choices from plausible alternatives.
- Forward rejects every locally knowable invalid request before the first Tensor expression,
  snapshots each current binding once, creates exactly the twenty- or twenty-one-Tensor chain in
  documented order, and returns the exact final ADD Tensor as next hidden state.
- Valid rank-one and higher inputs use ordinary conservative right-aligned leading broadcasting.
  Final feature checks, mixed floating promotion, output Shape/type, gate slice intervals, and
  failure order match current Model semantics.
- Replacement affects later calls only; existing expressions retain old exact parameter
  references. Mode changes do not affect the expression chain.
- No hidden recurrent state, result record, unary adapter, `Sequential` change, context, RNG state,
  sequence/time behavior, scan, Model change, optimizer, execution behavior, or framework-
  compatibility claim is introduced.
- Focused tests cover exact reflection surface, state schema/order, supplied validation, initialized
  draw/ID order, early failures, all gate intervals, exact operation/provenance association,
  broadcasting/type/rank/feature failures, replacement snapshots, mode independence, state
  discovery, and all explicit exclusions.
- All new and affected public/package APIs have meaningful complete Javadoc for purpose, equations,
  packing, Shapes/types, ownership, nullability, source lifecycle, side effects, promotion,
  broadcasting, result semantics, failure order, concurrency, mode, and non-execution boundaries.
- A separate documentation-focused clean context finalizes affected Javadoc, package docs,
  glossary impact, planning evidence, no-change conclusions, links, and generated Javadoc in the
  same overall change.

## Tests / validation

Implementation pass runs focused tests while developing and, after executable Java stabilizes:

```bash
./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.layers.GruCellTest --tests io.github.pho001.synaptik.nn.layers.GruCellInitializationTest
./gradlew :extensions:nn:test
```

The second command is the sole authoritative final affected-module Java validation. It covers the
existing NN suite plus the new cell. Model tests are not repeated: current exhaustive tests remain
authoritative for primitive linear, slice, ADD, SUB, MUL, SIGMOID, TANH, promotion, and broadcast
semantics, while focused NN tests lock cell composition and preflight.

Documentation pass runs after final Javadoc edits:

```bash
./gradlew :extensions:nn:javadoc
git diff --check
git diff --no-index --check /dev/null extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/GruCell.java
git diff --no-index --check /dev/null extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/GruCellTest.java
git diff --no-index --check /dev/null extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/GruCellInitializationTest.java
```

The documentation pass also validates generated `GruCell` and package pages; exact surface and
private fields; Model/NN/JDK-only imports; the unchanged sole NN Model dependency; Markdown links,
anchors, fences, final newlines, and trailing whitespace; exact seven-path scope; task/master
status synchronization; exactly one In progress NN row/spec during the pass; and absence of task
files for 0014–0015. The documentation pass changes that synchronized status to Complete only
after all required final gates pass.
It reuses successful implementation tests unless executable Java changes afterward or it records
a concrete reason to rerun them.

Repository-wide, architecture, Model, Training, CPU, compiler, backend-conformance, and integration
tests are deferred to the recurrent NN milestone checkpoint or CI. This task changes one Model-
only extension API and no dependency, architecture boundary, shared build, or execution contract.

## Dependencies

- NN tasks 0001–0012 are Complete and provide direct Module ownership, stable schema-validated
  parameters, explicit-source initialization, deterministic discovery/state dictionaries,
  unary-composition exclusions, and the recurrent validation/state-threading precedent.
- Model `Tensor.linear`, `sliceAxis`, `add`, `sub`, `mul`, `sigmoid`, and `tanh`, same-category
  promotion, `ShapeBroadcast`, immutable descriptors/provenance, and ID allocation are Complete.
- Current independent slice expressions truthfully retain exact packed projection provenance;
  no shared split API is required. Concat is not used because initialized parameters must remain
  direct eager leaves and gate packing is one schema, not a runtime composition.
- `ParameterInitializers.glorotUniform` and `zeros` provide the complete initialized policy.
- ADR 0007 and existing dependency tests already permit Model-only NN composition.
- Compiler, Training, CPU, Runtime, Prepare, Engine, backend, and numerical execution support are
  not prerequisites because this task constructs parameter leaves and expression metadata only.

## Follow-up tasks

- NN 0014 remains Draft: add an LSTM cell only after this task is Complete. It must accept and
  return both hidden and cell state explicitly and must not store either in a Buffer or hidden
  field.
- NN 0015 remains Draft: decide static unrolling of concrete cell calls versus a genuinely
  general recurrent Model scan. `CUM_SUM` and `CUM_PROD` do not satisfy that need.
- If NN 0015 selects containers, prefer cell-specific types unless the completed RNN/GRU/LSTM
  signatures prove a shared recurrent contract without casts, state erasure, duplicate carriers,
  or hidden state.
- Alternate GRU packing, bias, reset-before, migration, and framework-conversion policies require
  a concrete consumer and separate task; they must not be added as aliases here.
- Truncation, masking, variable lengths, bidirectionality, stacking, recurrent dropout, checkpoint
  persistence, training sessions, and backend execution remain separate future work.

## Documentation and no-change review

Document profiles:

- Java/package Javadoc: General plus API/Javadoc.
- glossary: General reference style.
- task/master plan: General plus Planning.

Required implementation-phase documentation changes are `GruCell` and layers-package Javadocs,
the existing glossary's explicit-state recurrent wording, and synchronized NN planning records.

The separate documentation pass must verify and record these reasoned no-change conclusions:

- `ARCHITECTURE.md`, focused architecture pages, ADR 0007, and architecture tests remain accurate
  because state ownership, module direction, and lifecycle boundaries do not change.
- Tensor and Compile APIs plus Model source/master/capabilities remain accurate because the cell
  composes existing expressions and adds no Tensor method, semantic kind, split, recurrent scan,
  capture, derivative, or execution behavior.
- Training API and training graph remain accurate because recursive parameter discovery and
  replacement consume this direct Module normally; no optimizer, session, gradient publication,
  truncated history, or state orchestration is added.
- `RnnCell`, `UnaryTensorModule`, `Sequential`, state-dictionary contracts, and their tests remain
  accurate. The GRU follows RNN explicit hidden threading and is independently excluded from
  unary composition; transient gates and hidden values are not module state.
- Gradle and dependency rules/tests remain accurate because `extensions/nn` retains only its
  existing Model dependency.
- Backend conformance and integration tests remain unnecessary because no numerical execution,
  backend capability, or end-to-end lifecycle changes.
- The global roadmap, CPU work, other modules, and Draft NN tasks 0014–0015 remain untouched.

## Architecture impact

Expected impact: None.

This task realizes the existing NN responsibility for parameter-owning layer composition while
keeping recurrent state explicit. If implementation requires hidden module/runtime state, a
universal recurrent abstraction, a general scan, another module dependency, or an architecture
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

## Documentation-agent handoff

After executable Java/tests stabilize, give the clean documentation context:

- this task and exact seven-path limit;
- the final executable diff and exact focused/final NN commands, counts, and results;
- the exact public surface, direct-Module/Sequential exclusion, packed state names/order/Shapes/
  types, gate order, bias association, constructor prevalidation, and draw/ID order;
- full forward preflight, batch-broadcast examples, reset-after equations, twenty-/twenty-one-ID
  producer chains, next-hidden semantics, mode behavior, replacement snapshots, and failures;
- directly relevant architecture/ADR, documentation profiles, final NN/Model source and tests,
  Tensor/Training APIs, glossary, dependency tests, and planning history;
- the mandate to preserve concurrent glossary work and record every reasoned no-change conclusion;
- generated-Javadoc, reflection/`javap`, import/dependency, Markdown, exact-scope/status,
  task-0014–0015-absence, newline, and whitespace gates; and
- the required completion-summary and Status format from `AGENTS.md`.

## Local decisions

- Use `GruCell`, following the project's acronym style (`RnnCell`, `Rng`), and place it beside
  concrete layers. The name represents one cell step, not sequence traversal.
- Extend `Module` directly and return one Tensor. Two Tensor inputs are intrinsic, and output and
  next hidden are one exact value; a unary adapter or duplicate result carrier would obscure the
  contract.
- Pack two matrices and optional bias in reset, update, candidate order. Independent slices make
  gate provenance exact with the current Model API and keep the public state/accessor surface as
  small as `RnnCell`. Six public per-gate matrices would enlarge state without a current consumer.
- Use one optional packed input-side bias. This is the smallest unambiguous bias policy supported
  by one biased packed linear call. There is no recurrent bias hidden inside reset placement.
- Fix reset-after semantics: reset multiplies the already projected recurrent candidate lane.
  Reset-before would require a different projection association and is not an alias.
- Fix update as a retention gate through `candidate + update * (hidden - candidate)`. This avoids
  inventing a scalar-one Tensor while making the exact operation order and endpoint meanings clear.
- Use Glorot uniform independently for the actual packed matrix Shapes. Concatenating per-gate
  eager leaves would make a derived expression the parameter and add unnecessary state/provenance;
  no orthogonal initializer exists.
- Permit current Model mixed floating promotion and conservative leading broadcasting, as
  `RnnCell` does. Preflight duplicates only local public algebra needed to prevent a late gate or
  interpolation failure from leaving a projection prefix.
- Accept unresolved final feature equality exactly as Model linear does. The fixed parameter
  output and slice bounds still make every next-hidden final axis statically `hiddenSize`.
- Keep all leading axes semantically neutral batch coordinates. Task 0015, not this cell, owns any
  time-axis contract and repeated invocation.

## Known limitations

- Only this reset-after, reset/update/candidate-packed GRU with positive fully static parameters
  and one optional input-side packed bias is supported.
- Input and hidden Shapes may contain unresolved leading or final Dimensions only when current
  local Model rules can represent the projection and broadcast obligations. NN performs no
  binding.
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

- Clean planning context `/root/nn_0013_planning` read the repository instructions, architecture
  contract and current plan, planning guide/roadmap, documentation rules and General/Planning/API-
  Javadoc profiles, NN master and complete task history through 0012, final recurrent/module/
  initializer APIs and tests, Model master and relevant Tensor/Shape/operation contracts, Training
  API/graph, glossary, and build/dependency rules before selecting this contract.
- Planning inspection confirmed the current Model surface can express the complete formula with
  packed linear projections and exact independent `sliceAxis` provenance. No Model split, concat,
  recurrent scan, semantic kind, result carrier, or new initializer is required.
- Planning selected reset/update/candidate packing, one input-side packed bias, reset-after
  candidate association, and retention-gate interpolation explicitly; no framework-compatibility
  equivalence is claimed.
- Targeted planning validation passed for the two changed planning paths: local links and anchors
  resolve, fences balance, final newlines are present, and trailing whitespace is absent.
- The NN planning state initially had exactly one Ready row/spec (0013); implementation has moved
  that same sole frontier to In progress. Tasks 0014–0015 remain concise Draft rows with no task
  files. Every unrelated concurrent CPU/source/test/documentation/master/task/roadmap/glossary
  change remains untouched.
- Clean implementation context `/root/nn_0013_implementation` added the direct final `GruCell`,
  focused state/forward and initialization tests, and draft type/package Javadocs. The production
  implementation has exactly the planned three packed parameter fields and seven-member public
  surface, snapshots current bindings in declaration order, prevalidates the complete local
  request before expression creation, then delegates the exact twenty-/twenty-one-ID reset-after
  expression chain in reset, update, candidate slice order.
- The focused two-suite selection passed 14 tests. The sole authoritative final affected-module
  Java run, `./gradlew :extensions:nn:test`, passed 21 suites and 139 tests with zero failures,
  errors, or skips; executable Java did not change afterward. Preliminary generated Javadoc also
  passed.
- Public/private `javap`, reflection coverage, import and existing sole-dependency inspection, and
  forbidden-state scans confirmed the exact direct-Module surface, only three private packed
  parameter fields, Model/NN/JDK production imports, unchanged Model-only NN dependency, and no
  hidden Tensor/Buffer/context/RNG/sequence/scan/execution state.
- Independent clean documentation context `/root/nn_0013_docs` read the final implementation and
  tests, architecture and planning contracts, documentation rules and General/API-Javadoc/
  Planning profiles, NN task lineage through 0012, Module/Parameter/initializer/unary/RNN
  boundaries, relevant Model Tensor/Shape/type/provenance contracts, Training API/graph, glossary,
  and build/dependency rules. It found no executable defect, architecture uncertainty, scope
  blocker, or reason to change executable Java or tests.
- The documentation context finalized `GruCell` and layers-package Javadocs. The rendered contract
  now states the reset/update/candidate packed schemas, input-side-only bias, six independent
  slices, reset-after equations, retention-gate endpoints, explicit hidden ownership, complete
  preflight, binding snapshots, caller-owned initializer source/order, mode independence,
  `Sequential` exclusion, and non-execution boundary. `./gradlew :extensions:nn:javadoc` passed
  after those edits (`BUILD SUCCESSFUL`; three actionable tasks, two executed and one up-to-date),
  and generated `GruCell.html` plus `layers/package-summary.html` inspection found the expected
  links, signatures, parameters, returns, failures, formulas, and boundaries.
- `javap -public` and `javap -private` confirmed final direct `Module`, exactly three public
  constructors and four declared public methods, no protected member or nested type, and only
  private final `Parameter`, `Parameter`, and `Optional<Parameter>` fields. An exploratory JShell
  reflection inspection printed the same successful contract checks but exited nonzero while
  persisting tool history because the environment rejected Java Preferences synchronization; a
  standalone reflection program then compiled and ran successfully and is the authoritative
  reflection result.
- The targeted Markdown validator passed the three task-owned Markdown files with 353 links
  inspected, 295 local anchors resolved, 40 balanced fence markers, final newlines, and no trailing
  whitespace. Source/import and build inspection found only Model, existing NN, and JDK production
  imports and retained the sole NN dependency `implementation(project(":modules:model"))`.
- The validator's first local-anchor run reported false missing-anchor results because its
  temporary GitHub-heading slug approximation removed punctuation before converting spaces; the
  corrected approximation converted spaces first and the exact same documents passed. An initial
  no-index loop likewise used zsh's special `path` variable, which removed `git` from command
  lookup after the first iteration and exited 127; the rerun used a task-specific variable and all
  four new-file checks passed. Neither exploratory failure changed a repository file.
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
  expressions without a new semantic kind, scan, capture, derivative, or execution contract.
- Training API and the training graph require no GRU-specific change because ordinary recursive
  parameter discovery/replacement already consumes a direct Module and the task adds no optimizer,
  session, gradient publication, truncated history, or state orchestration. `RnnCell`,
  `UnaryTensorModule`, `Sequential`, state-dictionary contracts, and their tests remain accurate:
  the GRU is separately excluded and its transient gates and hidden values never become module
  state.
- Gradle/dependency rules and tests, backend conformance, and integration tests require no change
  because no dependency, numerical execution, backend capability, or end-to-end lifecycle changed.
  The global roadmap, concurrent CPU work, other modules, and Draft NN 0014–0015 work remain
  untouched.

## Implementation notes

- Added final `GruCell extends Module` directly. It owns stable `inputWeight`, `hiddenWeight`, and
  optional input-side `bias` wrappers with exact packed reset/update/candidate schemas and retains
  no hidden Tensor, Buffer, child, context, RNG, or result state.
- Supplied constructors validate the complete exact schema before declaration. Initialized
  construction validates every requested packed count before draws, calls Glorot uniform for
  input then hidden weights from the same caller-owned source, and creates optional zero bias
  afterward without a draw.
- Forward rejects every locally knowable invalid request before the first expression ID, derives
  independent final-axis gate slices, and constructs exactly
  `n + z * (hidden - n)` with `n = tanh(x_n + r * h_n)`. Exact provenance tests distinguish the
  input-only bias, reset-after candidate, retention interpolation, and fixed call order.
- Tests cover exact public/finality/superclass and exclusion surface, packed registration and
  recursive discovery, constructor validation and failure side effects, initializer bounds and
  provenance, forward preflight, all gate slices and operation associations, mixed floating and
  leading broadcasting, mode independence, replacement snapshots, repeated explicit state
  threading, absence of retained state, and null/failure behavior.
- No Model, Training, architecture, Gradle, dependency, compiler, runtime, backend, conformance,
  integration, global-roadmap, CPU, LSTM, or sequence file changed. Current contracts remain
  sufficient because this task only composes existing Model metadata expressions.
- Independent documentation context `/root/nn_0013_docs` finalized the public/package Javadocs,
  added the glossary's packed schema and one-step gate example, distinguished the GRU from vanilla
  RNN, future LSTM, recurrent scan, cumulative scan, and unary `Sequential` composition, and
  synchronized all final evidence without changing executable Java or tests.

## Completion summary

- Completed changes: added final direct-Module `GruCell` with exact packed state, supplied and
  explicit-source construction, complete forward preflight, reset-after gate composition, sole
  next-hidden Tensor result, and focused contract tests; finalized all affected documentation.
- Files changed or created: exactly `GruCell.java`, layers `package-info.java`,
  `GruCellTest.java`, `GruCellInitializationTest.java`, `docs/glossary.md`, the NN master plan, and
  this task; unrelated concurrent paths were preserved.
- Tests and validation: reused the passing focused 2-suite/14-test and authoritative
  21-suite/139-test NN evidence with zero failures, errors, or skips. Final NN Javadoc/generated-
  page inspection, `javap`, independent reflection, import/dependency, Markdown, exact-scope/
  status, newline/whitespace, no-index, and whole-diff checks passed.
- Documentation-agent review: clean context `/root/nn_0013_docs` completed the mandatory
  independent source/test/API/architecture review and changed no executable Java or test.
- Documentation impact: finalized the GRU and layers-package Javadocs, added the glossary's packed
  schema and explicit one-step example plus recurrent-boundary distinctions, and synchronized
  task/master status and evidence.
- Javadoc review: complete; generated type and package pages contain the exact surface, state,
  equation, failure, ownership, concurrency, mode, and non-execution contracts.
- Glossary impact: added the reusable GRU term, fixed packed checkpoint schema, reset-after gate
  example, and distinctions from vanilla RNN, future LSTM, recurrent/cumulative scan, and
  `Sequential`.
- Unresolved issues: None.
- Follow-up required: None for task 0013; NN 0014–0015 remain separate Draft work.

Status: Complete

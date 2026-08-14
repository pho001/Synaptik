# Task 0012: Vanilla Tanh RNN Cell

## Status

Complete

## Goal

Add the smallest truthful recurrent neural-network capability: one final public vanilla tanh
`RnnCell` whose hidden state is an explicit caller-supplied Tensor and whose returned Tensor is the
next hidden state. The cell owns only trainable projection parameters. It retains no hidden state
between calls and composes the complete result through existing Model `linear`, ordinary `add`,
and `tanh` expressions.

Mental model:

```text
caller input x + caller hidden h + current cell parameters
  -> input.linear(inputWeight, optional bias)
  -> hidden.linear(hiddenWeight)
  -> add both projections
  -> tanh
  -> one Tensor h' returned to the caller
  -> caller explicitly threads h' into a later call when recurrence is intended
```

For a vanilla RNN cell, the visible output and next hidden state are the same Tensor. The API
therefore returns that Tensor once rather than introducing a result record with two fields that
would retain identical references.

## Scope

- Add final public `io.github.pho001.synaptik.nn.layers.RnnCell` extending
  `io.github.pho001.synaptik.nn.module.Module` directly.
- Add exactly the constructors, parameter accessors, and two-input forward method in the public
  API table below. Add no overload, builder, options object, activation argument, state carrier,
  size getter, or functional facade.
- Declare positive fully static rank-two parameters under exact local names `inputWeight` then
  `hiddenWeight`. Their Shapes are respectively `[hiddenSize, inputSize]` and
  `[hiddenSize, hiddenSize]`.
- Optionally declare one positive fully static rank-one parameter under exact local name `bias`,
  with Shape `[hiddenSize]`. It is one shared affine bias, not separate input and recurrent biases.
- Require all declared parameters to have one exact floating data type and
  `requiresGrad == true`. Retain caller-supplied parameter Tensor references exactly.
- Provide supplied-state construction both without and with bias. Null never means absent bias;
  callers select absence through the two-Tensor constructor.
- Provide one initialized constructor with explicit positive `inputSize`, positive `hiddenSize`,
  bias presence, floating `DataType`, and caller-owned `RandomGenerator`.
- Initialize `inputWeight` then `hiddenWeight` through exact
  `ParameterInitializers.glorotUniform` calls using the same explicit source. When requested,
  initialize bias afterward through exact `ParameterInitializers.zeros`; bias consumes no random
  draw. The source is never retained.
- Add exactly `Tensor forward(Tensor input, Tensor hidden)`. Both semantic inputs are explicit on
  every call; the cell never supplies, caches, updates, registers, or discovers a hidden state.
- Fix activation to existing `Tensor.tanh()`. This is the vanilla tanh cell, not a configurable
  activation wrapper.
- Support input and hidden rank one or higher. The final input Dimension contracts with
  `inputWeight` input size, and the final hidden Dimension contracts with `hiddenWeight` hidden
  size under the current Model linear rule.
- Treat every leading input or hidden Dimension as ordinary batch metadata. The two projected
  Shapes use current right-aligned ordinary ADD broadcasting, so their leading prefixes may have
  different ranks or singleton extents when `ShapeBroadcast` can prove compatibility. No leading
  axis is named, interpreted, or traversed as time.
- Complete all caller-controlled forward validation before creating the first projection
  expression. Then construct exactly the formula and producer chain specified below.
- Keep the layer mode-insensitive. Forward accepts no `ForwardContext`, does not inspect
  `mode()`, and does not alter mode or state.
- Preserve stable parameter wrappers, compatible replacement snapshots, recursive discovery, and
  state-dictionary paths through the existing `Module` contracts.
- Add focused exact-surface, supplied-state, initialization, validation-order, shape/type,
  provenance, replacement, mode, state-discovery, and exclusion tests.
- Add complete type, constructor, member, and package Javadocs. After executable work and final NN
  testing, use the required separate clean documentation-focused context to finalize Javadocs,
  glossary impact, planning evidence, generated Javadoc, and no-change conclusions.

## Out of scope

- Retaining the current or next hidden Tensor in a field, `Buffer`, child, state dictionary,
  thread-local, static, runtime object, session, or other hidden lifecycle.
- A default/zero hidden state, hidden-state initializer, state reset/detach API, stateful forward
  overload, or a result that omits the caller's responsibility to thread state.
- A result record containing `output` and `nextHidden`. They would be the same exact Tensor for
  this cell, and no current consumer requires duplicate component names.
- `UnaryTensorModule`. The complete forward signature has two Tensor inputs, so the cell is not a
  unary Tensor-to-Tensor module.
- Participation in `Sequential`, an adapter into `Sequential`, a wrapper that captures hidden
  state, a context erasure mechanism, or a change to `Sequential`/`UnaryTensorModule`.
- A recurrent sequence container, loop, step counter, time-axis convention, sequence-length or
  mask input, packed sequence, variable-length handling, bidirectionality, stacking, dropout
  between steps, static unrolling, or any time traversal.
- A general recurrent Model scan, loop operation, subgraph/body representation, carried-value
  tuple, scan result, or Tensor API. Existing `CUM_SUM` and `CUM_PROD` are ordered associative
  cumulative operations and must not be treated as recurrent scan.
- GRU gates, LSTM cell state, peepholes, projections, layer normalization, clipping, residuals,
  attention, convolutional recurrence, or another recurrent-cell family.
- Configurable activation, activation enum, callback, function object, ReLU variant, sigmoid
  output, or fast/approximate tanh policy.
- Separate input and recurrent biases, bias fusion policy, frozen parameters, mixed parameter
  types, integral/BOOL parameters, lazy/dynamic parameter Shapes, zero feature sizes, quantized or
  sparse state, custom initializer objects, orthogonal initialization, or a retained/default RNG.
- A `ForwardContext`, `GraphRngState`, `RandomGenerator` during forward, mode-dependent branch,
  Buffer transition, or stochastic recurrence.
- Truncated backpropagation through time, gradient detachment, gradient rules, optimizer/session,
  parameter groups, checkpoint transport, serialization, compiler capture, graph scheduling,
  runtime/prepare/Engine behavior, backend lowering, kernels, numerical execution, or end-to-end
  claims.
- A Model, Training, Gradle, dependency, architecture-contract, ADR, architecture-test, global-
  roadmap, CPU, glossary, or explanatory-document edit during this planning pass.
- Detailed task specifications for NN 0013–0015.

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
- [Completed NN task 0005: Linear layer](0005-linear-layer.md)
- [Completed NN task 0010: State dictionary](0010-state-dictionary-and-checkpoint-contract.md)
- [Completed NN task 0011: Unary Tensor composition](0011-unary-tensor-module-composition-and-sequential.md)
- [Completed Model task 0019D: Linear convenience](../../../modules/model/tasks/0019d-linear-convenience.md)
- [Completed Model task 0023E: Cumulative scan normalization and product](../../../modules/model/tasks/0023e-cumulative-scan-normalization-and-product.md)

## Architecture constraints

- `extensions/nn` continues to depend only on `modules/model`. Production must not import
  Training, Compiler, Runtime, Prepare, Engine, CPU, Metal, CUDA, or another backend.
- NN owns the two projection parameter bindings and typed recurrent-cell composition. Model
  remains the sole owner of linear, ADD, TANH, type promotion, local Shape algebra, Tensor
  descriptor/provenance construction, and expression identity.
- Tensor identity, descriptors, and provenance remain immutable. Parameter replacement changes
  only the current Tensor reference returned by one stable NN wrapper; existing Tensors and
  expressions remain unchanged.
- Recurrent state is a caller-threaded Tensor value, not module-owned persistent state. A hidden
  Tensor must not be registered as a Buffer merely because a caller may pass it between calls.
- `Module` remains without a universal forward method. `RnnCell` is a direct subclass with the
  truthful two-input signature and must not implement or extend `UnaryTensorModule`.
- `Sequential` remains a container only for `UnaryTensorModule`. This cell is excluded by type and
  by explicit contract; no adapter may hide one input or state transition.
- Mode is NN composition metadata. This mode-insensitive cell neither consumes a context nor reads
  inherited mode during forward.
- Construction and forward create eager parameter leaves or storage-free Model expression
  metadata only. They do not prove gradient implementation, graph capture, compiled execution,
  backend support, numerical values, or storage residency.
- If implementation needs a new Model operation/helper/public method, result carrier, recurrent
  container, state Buffer, context/RNG input, dependency, architecture rule, or eighth task path,
  stop and report the exact blocker instead of widening this task.

## Public API

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
| two-Tensor constructor | Retains exact positive static input/hidden projection weights and declares no bias. |
| three-Tensor constructor | Retains the same exact weights plus one exact shared bias; null bias is invalid. |
| initialized constructor | Uses explicit sizes/type/source, creates input weight then hidden weight through Glorot uniform and optional zero bias afterward. |
| `inputWeight()` | Returns the exact stable wrapper declared under `inputWeight`. |
| `hiddenWeight()` | Returns the exact stable wrapper declared under `hiddenWeight`. |
| `bias()` | Returns an empty Optional or the exact stable wrapper declared under `bias`. |
| `forward(input, hidden)` | Validates the complete two-input request, snapshots current parameters, composes the fixed tanh formula once, and returns the exact next-hidden Tensor. |

The class declares no other public or protected constructor, method, field, nested type, interface,
or overload. Inherited final `Module` APIs remain available normally.

## State schema and ownership

| Kind | Local name | Shape orientation | Type | `requiresGrad` | Initialized policy |
|---|---|---|---|---|---|
| Parameter | `inputWeight` | `[hiddenSize, inputSize]` | exact configured floating type | `true` | Glorot uniform from explicit source |
| Parameter | `hiddenWeight` | `[hiddenSize, hiddenSize]` | same exact type | `true` | Glorot uniform from the same explicit source |
| Parameter, optional | `bias` | `[hiddenSize]` | same exact type | `true` | exact typed zero |

The one hidden-size Dimension is the structurally equal static Dimension at `inputWeight` axis
zero, both `hiddenWeight` axes, and optional bias axis zero. `inputSize` is `inputWeight` axis one.
Both sizes are strictly positive.

Direct discovery order and direct state-dictionary order are exactly `inputWeight`,
`hiddenWeight`, then optional `bias`. When an owning future module registers the cell as child
`cell`, recursive paths are `cell.inputWeight`, `cell.hiddenWeight`, and optional `cell.bias`.
The input hidden Tensor and returned next hidden Tensor never appear in parameter, buffer, child,
or state-dictionary discovery.

## Supplied construction validation and side effects

Both supplied constructors validate complete state before declaring any parameter.

For the two-Tensor constructor:

1. reject null `inputWeight`, then null `hiddenWeight` with those exact parameter names;
2. validate input weight floating type, `requiresGrad == true`, rank two, fully static Shape,
   positive hidden-size axis zero, then positive input-size axis one;
3. validate hidden weight floating type, `requiresGrad == true`, rank two, fully static Shape,
   positive axis zero, then positive axis one;
4. require hidden weight exact data type to equal input weight data type;
5. require hidden weight axis zero to equal input weight hidden-size axis structurally;
6. require hidden weight axis one to equal that same hidden-size Dimension structurally; and
7. declare `inputWeight` then `hiddenWeight` and retain no bias.

For the three-Tensor constructor, reject null `bias` after the two weight null checks and before
state-schema validation. Apply steps 2–6, then validate bias floating type,
`requiresGrad == true`, rank one, fully static Shape, exact common parameter data type, and exact
hidden-size Dimension in that order. Declare `inputWeight`, `hiddenWeight`, then `bias` only after
all checks pass.

Validation creates no Tensor, producer, storage, random draw, or Tensor identity and never mutates
the supplied values. A failure returns no cell and leaves every supplied Tensor unchanged.

## Initialized construction and side effects

The initialized constructor performs exactly:

1. reject null `dataType`, then null `randomGenerator`;
2. require `inputSize > 0`;
3. require `hiddenSize > 0`;
4. require the data type to be floating;
5. create immutable Shapes `[hiddenSize, inputSize]`, `[hiddenSize, hiddenSize]`, and, only when
   requested, `[hiddenSize]`;
6. obtain each requested Shape's checked known element count in state order and reject a count
   above `Integer.MAX_VALUE` before the first random draw or Tensor identity allocation;
7. create input weight through exactly
   `ParameterInitializers.glorotUniform(inputWeightShape, dataType, randomGenerator)`;
8. create hidden weight through exactly
   `ParameterInitializers.glorotUniform(hiddenWeightShape, dataType, randomGenerator)` using the
   same now-advanced source;
9. when requested, create bias through exactly
   `ParameterInitializers.zeros(biasShape, dataType)` with no source call; and
10. after all requested leaves exist, declare parameters in exact state-table order.

Caller-controlled null, size, type, checked-count, and Java-array-limit failures precede every
draw and Tensor ID. Input-weight samples consume one bounded `nextDouble(origin, bound)` call per
row-major element, followed by all hidden-weight calls; the bounds are computed independently by
each weight's Glorot fan pair. Bias consumes no draw.

A random-source failure keeps completed source calls and creates no Tensor for the failing weight.
If it occurs during hidden-weight creation, the already created input weight and its ID are not
rolled back, but no cell is returned. Later allocation or identifier failures likewise preserve
completed draws and already allocated IDs. The caller owns and coordinates the source; the cell
never retains, resets, closes, synchronizes, splits, seeds, or serializes it.

## Forward validation and side-effect order

`forward(input, hidden)` performs exactly:

1. reject null `input`, then null `hidden`;
2. read the current `inputWeight`, `hiddenWeight`, and optional `bias` bindings exactly once in
   parameter declaration order;
3. prevalidate the input affine projection under the current Model linear order: promote input
   and input-weight numeric types, require input rank at least one, reject a proven unequal static
   input-feature contraction, and, when bias is present, promote the product and bias types;
4. prevalidate the hidden projection: promote hidden and hidden-weight numeric types, require
   hidden rank at least one, and reject a proven unequal static hidden-feature contraction;
5. derive the two projection Shapes without creating a Tensor: preserve each operand's complete
   leading prefix and append the exact static hidden-size Dimension;
6. promote the two projection result types, then call ordinary
   `ShapeBroadcast.broadcast(inputProjectionShape, hiddenProjectionShape)` once to validate and
   derive their exact ADD result Shape; and
7. only after every preceding check succeeds, construct the formula below.

Because every parameter is floating, current same-category promotion requires both call inputs to
be floating, while allowing BFLOAT16/FLOAT32/FLOAT64 widening. A static final feature Dimension
must equal the configured size. An unresolved final feature Dimension is accepted under current
Model linear semantics and leaves that contraction obligation for compiler validation or later
binding.

Ordinary leading-prefix broadcasting is conservative. Equal dimensions and static singleton
expansion succeed; different symbolic dimensions, incompatible static sizes, and a symbolic
dimension paired with a static non-singleton fail before expression creation. Input and hidden
ranks need not equal. For example:

| Input Shape | Hidden Shape | Next-hidden Shape | Meaning |
|---|---|---|---|
| `[inputSize]` | `[hiddenSize]` | `[hiddenSize]` | one unbatched cell application |
| `[batch, inputSize]` | `[batch, hiddenSize]` | `[batch, hiddenSize]` | matching batch prefix |
| `[batch, inputSize]` | `[hiddenSize]` | `[batch, hiddenSize]` | hidden vector broadcast across batch |
| `[outer, 1, inputSize]` | `[batch, hiddenSize]` | `[outer, batch, hiddenSize]` | ordinary right-aligned batch broadcasting |

These examples define metadata composition only. `outer` and `batch` are leading coordinates,
not a time traversal or sequence loop.

Every local null/type/rank/static-contraction/broadcast failure consumes no Tensor ID and creates
no partial expression. Parameter reads and temporary Shape values are not Tensor-expression side
effects. Identifier exhaustion during the valid construction phase may leave a successful prefix
of expression IDs; no rollback is attempted.

## Formula, delegation, and provenance

Let `x` be `input`, `h` be `hidden`, `W_ih` be `inputWeight`, and `W_hh` be `hiddenWeight`.

Without bias:

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

No helper constructs `Operation`, `TensorDescriptor`, `TensorProvenance`, or derived Tensor
directly. Model owns each occurrence.

The no-bias successful chain creates exactly six fresh Tensors/IDs in order: input-weight
PERMUTE, input MATMUL, hidden-weight PERMUTE, hidden MATMUL, projection ADD, then TANH. The biased
chain creates seven: input-weight PERMUTE, input MATMUL, bias ADD, hidden-weight PERMUTE, hidden
MATMUL, projection ADD, then TANH.

The returned Tensor has TANH provenance over the exact projection ADD. That ADD retains exact
ordered `[inputProjection, hiddenProjection]` references. Each linear projection exposes the
existing PERMUTE-to-MATMUL chain, and the biased input projection exposes its existing final ADD
with ordered `[inputProduct, bias]`. The result has the prevalidated broadcast Shape, the widest
floating type selected in expression order, unresolved layout, gradient eligibility equal to the
logical OR of input, hidden, and all parameter requests, no label, and no host storage.

This provenance describes declarative expression construction. The task makes no claim about
floating reassociation, fused kernels, numerical values, gradients, compiler capture, backend
availability, or execution.

## Output and state-threading contract

The exact returned Tensor is simultaneously the cell's visible output and next hidden state. The
cell does not retain it. A caller expresses recurrence only by passing that exact reference, or
another explicitly selected compatible Tensor, as a later call's `hidden` argument:

```text
h1 = cell.forward(x1, h0)
h2 = cell.forward(x2, h1)
```

Reusing `h0` in two calls creates two explicit branches. Passing `h1` to the next call creates a
dependency chain. Neither pattern mutates a Tensor or the cell. There is no implicit call order,
step identity, truncated history, detachment, or sequence ownership.

## Replacement, mode, and composition behavior

- Each forward call snapshots all current parameter bindings before validation and expression
  construction. A compatible `Parameter.replace` affects only later calls.
- An expression created before replacement keeps the exact old parameter references. Stable
  accessors and recursive discovery keep the same wrappers and expose the new binding on later
  `value()` reads.
- Individual replacement and forward construction remain non-thread-safe as a combined operation.
  Callers must coordinate them when one multi-parameter snapshot matters.
- `train()` and `eval()` retain inherited structural behavior, but forward composition is
  identical in both modes and consumes no `ForwardContext`.
- `RnnCell.class.getSuperclass()` is exactly `Module`. It is not assignable to
  `UnaryTensorModule`, and no `Sequential` constructor can accept it.
- A future recurrent-style container may own the cell as an ordinary child Module and call its
  two-input method explicitly. Task 0012 neither names nor implements that container.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.nn.layers` — concrete stateful layer/cell ownership.
- `io.github.pho001.synaptik.nn.module` — existing direct `Module` and stable `Parameter` wrappers.
- `io.github.pho001.synaptik.nn.initialization` — existing Glorot-uniform and zero policies.
- `io.github.pho001.synaptik.model.tensor` — existing Tensor linear, ADD, and TANH expression API.
- `io.github.pho001.synaptik.model.datatype` — exact floating types and promotion preflight.
- `io.github.pho001.synaptik.model.shape` — static parameter Shapes and ordinary broadcast
  preflight.

No package is added. Exact type placement:

- `io.github.pho001.synaptik.nn.layers.RnnCell` — direct Module owning vanilla recurrent
  projection parameters.
- `io.github.pho001.synaptik.nn.layers.RnnCellTest` — exact surface, state, supplied validation,
  forward metadata/provenance, mode/exclusion, and replacement coverage.
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

Expected documentation and planning files:

- `docs/glossary.md` — extend the existing NN/recurrent terminology with explicit hidden-state
  threading, cell-versus-sequence distinction, and the declarative formula; do not add a glossary
  term during this planning pass.
- `docs/planning/extensions/nn/master-plan.md`.
- this task specification.

## Maximum scope

Implementation may create or modify exactly the seven paths above: two production/Javadoc paths,
two focused test paths, the glossary, NN master plan, and this task. If implementation needs
another result type, Model helper, existing Sequential test edit, Training API edit, architecture
test, dependency, build file, recurrent container, or eighth path, stop and propose a focused
revision rather than expanding this task.

## Acceptance criteria

- Public final `RnnCell extends Module` declares exactly the three constructors, three parameter
  accessors, and one `forward(Tensor, Tensor)` method in the API table, with no other declared
  public/protected member.
- It neither extends nor implements `UnaryTensorModule` and cannot participate in `Sequential`.
  No adapter, hidden-state capture, or Sequential overload is added.
- Supplied constructors retain exact Tensors, validate all nulls then complete input-weight,
  hidden-weight, and optional-bias schema in the stated order, and declare exact names/order only
  after validation without allocating Tensor IDs.
- Initialized construction supports exactly current floating types, validates all requested
  Shapes/counts before the first draw, creates input weight then hidden weight through Glorot
  uniform and optional zero bias afterward, uses one explicit source, and retains no source.
- Direct and recursive discovery plus state dictionary expose exactly the selected parameter
  names/order and no Buffer, hidden state, input, output, or RNG state.
- Forward null-checks input then hidden, snapshots parameters once, and prevalidates the complete
  input projection, hidden projection, type combination, and ordinary broadcast before creating
  the first expression.
- Tests cover rank-one and higher-rank calls, equal and broadcastable leading prefixes, hidden
  vector broadcast, all three floating types and mixed widening, static feature mismatches,
  incompatible leading prefixes, and representative symbolic compatibility/rejection.
- Valid no-bias and biased calls build exactly the selected primitive chains and ordered exact
  provenance with the documented ID order, result Shape/type/eligibility/layout/label/storage,
  and fresh identities without reading numerical values.
- The returned Tensor is used once as both output and next hidden. No duplicate-reference result
  carrier, state field, Buffer transition, state initialization, or mutation appears.
- Training and evaluation modes create the same expression contract. No context, graph RNG,
  random draw during forward, runtime state, or hidden side effect appears.
- Compatible parameter replacement affects only later forward expressions; earlier expressions
  retain old exact bindings, wrappers and paths remain stable, and incompatible replacement stays
  governed by existing `Parameter` schema.
- Focused tests prove local validation consumes no Tensor ID, while identifier/source/allocation
  failures retain current non-rollback effects without returning a partially initialized cell or
  hidden state.
- Public/package Javadocs document purpose, fixed formula/activation, parameter orientation and
  names, initialization order/source ownership, exact state threading, leading batch broadcast,
  validation/failure order, result/provenance, mode, replacement snapshots, thread safety,
  Sequential exclusion, and no-execution boundaries with complete tags.
- The glossary distinguishes a one-step explicit-state cell from sequence recurrence and from
  Model cumulative sum/product scans, with no invented scan API or numerical-execution claim.
- A separate clean documentation-focused context finalizes all affected Javadocs, package docs,
  glossary, planning evidence, generated Javadoc, Markdown, exact scope/status, and no-change
  conclusions before the task becomes Complete.
- No Model, Training, architecture/ADR/test, Gradle/dependency, compiler/runtime/prepare/Engine/
  backend, conformance/integration, CPU, global-roadmap, task-0013–0015 file, or unrelated path
  enters the implementation diff.

## Tests / validation

Validation tier: normal task validation for the single affected `extensions/nn` module plus the
required targeted documentation pass. This task changes no project dependency, architecture
boundary, backend behavior, or end-to-end execution path.

Implementation runs focused tests while developing:

```text
./gradlew :extensions:nn:test \
  --tests io.github.pho001.synaptik.nn.layers.RnnCellTest \
  --tests io.github.pho001.synaptik.nn.layers.RnnCellInitializationTest
```

After executable Java and tests stabilize, run the affected module exactly once as final Java
evidence:

```text
./gradlew :extensions:nn:test
```

The separate documentation-focused pass reuses that successful evidence unless it changes
executable Java or records a concrete reason to rerun. After final Javadoc edits it runs:

```text
./gradlew :extensions:nn:javadoc
git diff --check
```

Final implementation/documentation validation also checks:

- reflection and `javap -public` for exact finality, direct `Module` superclass, three
  constructors/four methods, optional accessor type, and absence of another public/protected API;
- source/reflection proof that `RnnCell` is outside `UnaryTensorModule` and `Sequential`;
- exact parameter names, order, Shapes, types, gradient flags, discovery paths, and absence of
  Buffer/hidden state;
- manual provenance and ID-order inspection for both formulas;
- production imports and the unchanged sole Model NN dependency;
- generated `RnnCell` and layers-package Javadoc pages;
- local Markdown links/anchors, required headings, balanced fences, terminal newlines, and
  trailing whitespace;
- exactly seven task-owned implementation paths;
- NN 0001–0011 Complete, exactly NN 0012 Ready/In progress/Complete as appropriate, 0013–0015
  concise Draft rows, and no detailed 0013–0015 task files;
- `git diff --check`; and
- `git diff --no-index --check /dev/null <path>` for every untracked new file.

Repository-wide, architecture, Model, Training, CPU, compiler, backend-conformance, and integration
tests are deferred to the recurrent NN milestone checkpoint or CI. Existing Model tests remain the
authoritative exhaustive linear, ADD, TANH, promotion, and Shape-broadcast matrices; focused NN
tests lock only the cell-owned composition and preflight.

## Dependencies

- NN tasks 0001–0011 are Complete and provide direct Module ownership, stable parameters,
  explicit-source initialization, deterministic discovery/state dictionaries, and the proven
  boundary that non-unary modules remain outside Sequential.
- Model `Tensor.linear`, `Tensor.add`, and `Tensor.tanh`, same-category promotion,
  `ShapeBroadcast`, immutable descriptors/provenance, and identifier allocation are Complete.
- Existing `ParameterInitializers.glorotUniform` and `zeros` provide every selected initialized
  state policy. No new initializer is required.
- ADR 0007 and the existing architecture dependency test already permit Model-only NN
  composition.
- Compiler, Training, CPU, runtime, prepare, Engine, backend, and numerical execution support are
  not prerequisites because this task constructs parameter leaves and Tensor-expression metadata
  only.

## Follow-up tasks

- NN 0013: GRU cell remains Draft. It must keep hidden state explicit and choose its exact gate
  packing, parameter/bias schema, formula order, and result only at that frontier.
- NN 0014: LSTM cell remains Draft. It must accept and return both hidden and cell state explicitly
  and must not store either in a Buffer or hidden runtime field.
- NN 0015: recurrent sequence composition and scan decision remains Draft. It must decide static
  unrolling of concrete cell calls versus first introducing a genuinely general recurrent Model
  scan. `CUM_SUM`/`CUM_PROD` do not satisfy that need.
- If NN 0015 chooses a container, it must prefer type-safe cell-specific
  `RnnSequence`/`GruSequence`/`LstmSequence` APIs unless the actual cell signatures prove a shared
  `RecurrentSequence` without casts, state erasure, duplicate carriers, or hidden state. This task
  reserves none of those type names.
- Truncated backpropagation, masking, variable lengths, bidirectionality, stacked recurrence,
  recurrent dropout, checkpoint persistence, training sessions, and backend execution require
  later concrete consumers and separate plans.

## Documentation and no-change review

Document profiles:

- Java/package Javadoc: General plus API/Javadoc.
- glossary: General reference style.
- task/master plan: General plus Planning.

Required implementation-phase documentation changes are the `RnnCell` and layers-package
Javadocs, the existing glossary's NN/recurrent wording, and synchronized NN planning records.

The separate documentation pass must verify and record these reasoned no-change conclusions:

- `ARCHITECTURE.md`, focused architecture documents, ADR 0007, and architecture tests remain
  accurate because state ownership, module direction, and lifecycle boundaries do not change.
- Tensor and Compile APIs plus Model source/plans/capabilities remain accurate because the cell
  composes existing expressions and adds no Tensor method, semantic kind, scan, capture, gradient,
  or execution behavior.
- Training API and training graph remain accurate because existing recursive parameter discovery
  and replacement consume this direct Module normally; no optimizer, session, gradient
  publication, truncated history, or state orchestration is added.
- `UnaryTensorModule`, `Sequential`, and their tests remain accurate because the two-input cell is
  intentionally excluded; the new focused cell test owns the exclusion assertion.
- State-dictionary contracts remain accurate because only ordinary parameter paths are added and
  caller-threaded hidden state is not module state.
- Gradle and dependency rules/tests remain accurate because `extensions/nn` retains only its
  existing Model dependency.
- Backend conformance and integration tests remain unnecessary because no numerical execution,
  backend support, or end-to-end lifecycle changes.
- The global roadmap, CPU work, other modules, and Draft NN tasks 0013–0015 remain untouched.

## Architecture impact

Expected impact: None.

This task realizes the existing NN responsibility for parameter-owning layer composition while
keeping recurrent state explicit. If implementation requires hidden module/runtime state, a
universal forward abstraction, a general scan, another module dependency, or an architecture
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

## Documentation-agent handoff

After executable Java/tests stabilize, give the required clean documentation context:

- this task and exact seven-path limit;
- the final executable diff and exact focused/final NN commands, counts, and results;
- the exact public surface, direct-Module/Sequential exclusion, state names/order/Shapes/types,
  constructor prevalidation and initialization draw/ID order;
- full forward preflight, batch-broadcast examples, exact no-bias/biased producer chains, returned
  next-hidden identity, mode behavior, replacement snapshots, and failure effects;
- directly relevant architecture/ADR, documentation profiles, final NN/Model source and tests,
  Tensor/Training APIs, glossary, dependency test, and planning history;
- the mandate to finalize only task-owned Javadocs and prose, preserve concurrent glossary work,
  and record every reasoned no-change conclusion;
- generated-Javadoc, reflection/`javap`, import/dependency, Markdown, exact-scope/status,
  task-0013–0015-absence, newline, and whitespace gates; and
- the required completion-summary and `Status` format from `AGENTS.md`.

## Local decisions

- Use `RnnCell`, following the project's ordinary Java acronym style such as `Rng`, and place it
  beside concrete layers. The public name says one recurrent step, not sequence traversal.
- Extend `Module` directly. Two Tensor inputs are intrinsic to the cell; forcing it into the unary
  base would require captured hidden state or an adapter and would make Sequential unsafe.
- Return one Tensor. In a vanilla cell `output == nextHidden` by contract, so a record containing
  both names would duplicate one reference without additional information.
- Use one shared optional bias. It is the smallest conventional affine state and avoids separate
  input/recurrent biases whose only effect would be another parameterization of the same summed
  preactivation.
- Associate the shared bias with the input `linear` call, fixing expression association as
  `((xW^T + b) + hU^T)` before TANH. This reuses the existing fully prevalidated biased linear
  convenience and makes provenance unambiguous.
- Fix TANH. Configurable activation would turn one cohesive vanilla-cell capability into a generic
  recurrent framework without a current consumer.
- Use Glorot uniform independently for both rank-two matrices. It is the existing general
  unit-gain affine policy that supports both Shapes. No orthogonal initializer exists, and this
  task must not invent one.
- Permit current Model mixed floating promotion and conservative leading broadcasting rather than
  imposing equal ranks or exact parameter input types. Preflight duplicates only the public local
  algebra necessary to prevent a late hidden/broadcast failure from leaving an input-projection
  prefix.
- Accept unresolved final feature equality exactly as Model linear does. The fixed parameter
  output Dimension still makes every returned next-hidden final axis statically `hiddenSize`.
- Keep all leading axes semantically neutral batch coordinates. A future sequence task, not the
  cell, owns any time-axis contract and repeated invocation.

## Known limitations

- Only one vanilla tanh cell with positive fully static parameters and one optional shared bias is
  supported.
- Input and hidden Shapes may contain unresolved leading or final Dimensions only when current
  local Model rules can represent the projection and broadcast obligations. No binding occurs in
  NN.
- A higher-rank input is processed as one batched cell application. It is never traversed as a
  sequence, even if a caller informally assigns a leading axis the meaning of time.
- Caller-threaded recurrence can build arbitrarily deep provenance chains; this task adds no
  detachment, truncation, loop IR, scheduling, or memory-lifetime policy.
- Parameter replacement and forward construction are not thread-safe as one snapshot. The cell
  retains no lock, version, or transaction.
- Initialized eager parameters require Java-array-sized host leaves and may consume source calls
  or Tensor IDs before a later resource/identifier failure; those effects are not rolled back.
- Forward constructs metadata and proves no numerical value, gradient rule, compiler capture,
  backend support, storage allocation, publication, or execution.

## Validation evidence

- Clean planning context `/root/nn_0012_planning` read the repository instructions, authoritative
  architecture and current architecture plan, planning guide/roadmap, documentation rules and
  General/Planning/API-Javadoc profiles, ADR 0007, complete NN master/task history through 0011,
  final NN module/initializer/layer APIs and tests, Model master and exact linear/MATMUL/ADD/TANH/
  Shape/broadcast contracts and tests, cumulative-scan planning/contracts, Training API/graph,
  glossary, and dependency/build architecture enforcement before selecting this API.
- Planning inspection confirmed the cell is fully expressible through current Model operations.
  The selected preflight can reproduce every local promotion, rank, static-contraction, and
  projection-broadcast check before expression creation; no new Model helper or scan is required.
- Planning also confirmed that vanilla output and next hidden are one value, so a result record
  would duplicate an identical Tensor reference, and that `UnaryTensorModule`/`Sequential` cannot
  truthfully represent a two-input call.
- Targeted planning validation resolved every Markdown link and local anchor in the two affected
  planning files; fences, final newlines, and trailing whitespace passed.
- The NN planning diff contains exactly this task and the NN master plan. The master has exactly
  one Ready row, this is the only Ready NN task, tasks 0013-0015 remain Draft rows without task
  specifications, whole-worktree `git diff --check` passed, and the new-file no-index whitespace
  check returned only the expected content-difference status.
- Concurrent CPU, source, test, documentation, master-plan, task, and global-roadmap changes were
  left untouched.
- Clean implementation context `/root/nn_0012_implementation` added the exact direct-Module
  `RnnCell`, its focused state/forward and initialization suites, and draft public/package
  Javadocs. The implementation required no Model helper, result carrier, hidden Buffer/state,
  ForwardContext, sequence/scan abstraction, dependency, or eighth task path.
- The stabilized focused command passed both new suites with 15 tests and no failures, errors, or
  skips. The sole authoritative `./gradlew :extensions:nn:test` run then passed 19 suites and 125
  tests with no failures, errors, or skips.
- Implementation provenance inspection confirmed the exact six-ID no-bias and seven-ID biased
  chains, ordered projection inputs, fixed TANH result, explicit hidden threading, mode
  independence, and old-versus-new binding snapshots. Constructor and forward failure tests
  confirmed the specified no-draw/no-ID and non-rollback boundaries.
- Executable Java remained behaviorally unchanged after the implementation test runs. Independent
  clean documentation context `/root/nn_0012_docs` reviewed the final source/tests and inherited
  Module, Parameter, initializer, Linear, unary-composition, Model expression, Shape/type, and
  Training boundaries against the architecture and task. It refined the RnnCell type summary,
  accepted the complete constructor/member and package contracts, and added the glossary's
  explicit hidden-state, one-step recurrence, Sequential exclusion, and cumulative-scan
  distinction.
- The documentation context reused the implementation's focused two-suite/15-test evidence and
  authoritative 19-suite/125-test NN run because only Javadoc/prose changed afterward. The final
  XML reports independently still contain 19 suites and 125 tests with zero skips, failures, or
  errors.
- Final `./gradlew :extensions:nn:javadoc` passed after the Javadoc refinement (`3 actionable
  tasks: 2 executed, 1 up-to-date`). Inspection of generated `RnnCell.html` and the layers
  `package-summary.html` confirmed the direct-Module/two-input boundary, parameter schemas,
  initialization and failure contracts, forward preflight/result semantics, and non-execution
  limits.
- Independent `javap -protected` and `javap -private` inspection plus a standalone reflection
  program passed: final direct `Module` superclass; exactly three public constructors and four
  declared public methods; no public/protected field or nested type; and only private stable
  `Parameter`, `Parameter`, and `Optional<Parameter>` state. Source/import and build inspection
  confirmed Model/NN/JDK imports only, no hidden-state/Buffer/context/RNG/sequence dependency, and
  the unchanged sole Gradle dependency on `:modules:model`.
- The targeted Markdown validator passed all three task-owned Markdown files with 349 local links,
  including 295 heading anchors, 42 balanced fence markers, final newlines, and no trailing
  whitespace. Exact-scope inspection found only the seven authorized NN 0012 paths amid preserved
  concurrent CPU work. NN 0001–0011 remain Complete; NN 0013–0015 remain Draft rows with no task
  specifications and no Ready NN task.
- `git diff --no-index --check /dev/null <path>` returned the expected content-difference status
  1 with no whitespace diagnostics for each of the four untracked task files, and final whole-
  worktree `git diff --check` passed with no output.
- `ARCHITECTURE.md`, focused architecture documents, ADR 0007, and architecture tests require no
  change: the cell stays inside existing Model-only NN composition, keeps hidden state caller-
  owned, and changes no boundary or dependency. Tensor/Compile APIs, Model source/master plan and
  capabilities also require no change because the cell only composes current linear, ADD, and TANH
  metadata and adds no operation, Tensor method, scan, gradient, capture, or execution behavior.
- Training API and the training graph require no change because ordinary recursive parameter
  discovery/replacement already consumes a direct Module and this task adds no optimizer,
  gradient publication, truncated history, or state orchestration. `UnaryTensorModule`,
  `Sequential`, state-dictionary contracts, and their existing tests remain accurate: the focused
  RnnCell suite proves the cell's explicit exclusion and its caller-threaded hidden Tensor never
  becomes module state.
- Gradle/dependency rules and tests, backend conformance, and integration tests require no change
  because no dependency, numerical execution, backend support, or end-to-end lifecycle changed.
  Compile, Training, Model capabilities, other modules, the global roadmap, concurrent CPU work,
  and Draft NN tasks 0013–0015 were preserved unchanged.

## Implementation notes

Clean implementation context `/root/nn_0012_implementation` completed the exact executable
capability and tests. The focused two-suite/15-test selection and authoritative 19-suite/125-test
NN module run passed with no failures, errors, or skips. Independent documentation context
`/root/nn_0012_docs` then finalized type/package Javadocs, glossary and planning evidence and
passed generated-Javadoc, surface/private-state, dependency/import, Markdown, seven-path scope,
status, newline, whitespace, and final-diff gates without changing executable behavior or
repeating the stable Java suites.

## Completion summary

- Completed changes: added final direct-Module `RnnCell` with exact supplied and explicit-source
  construction, stable two-weight/optional-bias state, complete forward preflight, fixed
  linear/add/tanh composition, one explicit next-hidden Tensor result, and focused contract tests.
- Files changed or created: `RnnCell.java`, layers `package-info.java`, `RnnCellTest.java`,
  `RnnCellInitializationTest.java`, `docs/glossary.md`, NN `master-plan.md`, and this task.
- Tests and validation: reused passing focused 15-test and authoritative 19-suite/125-test NN
  evidence; final NN Javadoc, generated-page inspection, independent `javap`/reflection,
  dependency/import, Markdown, exact-scope/status, newline, whitespace, no-index, and diff checks
  passed.
- Documentation-agent review: clean context `/root/nn_0012_docs` completed the mandatory
  independent source/test/API/architecture review and changed no executable behavior.
- Documentation impact: finalized RnnCell and layers-package contracts, added the explicit-state
  recurrent glossary definition/example and cumulative-scan distinction, synchronized planning,
  and recorded reasoned no-change conclusions for architecture, Model/Tensor/Compile/Training,
  unary/state-dictionary, dependency/build, conformance/integration, roadmap, CPU, and later NN
  work.
- Unresolved issues: none.
- Required follow-up: none for task 0012; NN 0013–0015 remain separate Draft work.

```text
Status: Complete
```

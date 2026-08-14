# Task 0008: Batch Normalization Layer

## Status

Complete

## Goal

Add one final public affine `BatchNorm` module as the first NN layer that owns both trainable
parameters and persistent buffers. The layer has one explicit layout-neutral channel axis,
mandatory rank-one scale and bias parameters, rank-one running-mean and running-variance buffers,
and exact typed momentum and epsilon. One typed forward call uses an immutable
`ForwardContext` snapshot to select the existing Model inference or training expression. A
successful training call installs the two returned next-statistic expressions into the stable
buffer wrappers; evaluation changes no state.

Mental model:

```text
input + scale + bias + current running statistics + mode snapshot
  -> evaluation: Model batchNormInference -> output, buffers unchanged
  -> training:   Model batchNormTraining
                   -> output
                   -> next mean installed as runningMean
                   -> next variance installed as runningVariance
```

The installation is NN binding replacement of pure Tensor expressions. It is not eager numerical
execution, in-place Tensor mutation, a backend write, a training session, or an atomic generic
multi-binding API.

## Scope

- Add final public `io.github.pho001.synaptik.nn.layers.BatchNorm` extending `Module`.
- Add exactly the public constructors, accessors, and forward method in the public API table
  below. Add no overload, default, alias, builder, options object, or generic layer interface.
- Store one non-negative logical `channelAxis`. It is already non-negative as layer
  configuration, but Model validates it against each input rank and retains the normalized value
  in operation attributes. The layer neither assumes NCHW/NHWC nor distinguishes a batch axis.
- Declare mandatory parameters under exact local names `scale` then `bias`. Both have the same
  exact floating data type, the same positive fully static rank-one Shape `[featureCount]`, and
  `requiresGrad == true` at declaration and after every public compatible parameter replacement.
- Declare mandatory buffers under exact local names `runningMean` then `runningVariance`. Supplied
  initial buffers have the same exact data type and Shape as the parameters and require
  `requiresGrad == false`. The initialized path creates exact typed zero mean and one variance with
  `requiresGrad == false`.
- Treat `Buffer` as optimizer-excluded module state, not as a promise that every later bound Tensor
  has `requiresGrad == false`. A successful training Model result defines each next statistic's
  gradient-eligibility metadata. Because the old buffers begin non-gradient and the layer accepts
  only successful exact-result-typed calls, the first next mean/variance each have
  `requiresGrad == input.descriptor().requiresGrad()`; a later expression may remain gradient-
  eligible through its old-statistic dependency. Do not detach, cast, copy, evaluate, or
  reconstruct those results.
- Retain exact immutable `ScalarValue` references for momentum and epsilon. Momentum is the Model
  training convention: finite new-batch weight in `[0, 1]`. Epsilon is finite and strictly
  positive. Both must have the exact common state data type; no raw `double`, default, conversion,
  Tensor scalar, or cumulative-average sentinel is allowed.
- Provide one caller-supplied-state construction path and one initialized construction path.
  Supplied state supports explicit pre-existing parameter/statistic expressions. Initialized
  state uses the already selected declarative parameter policies and direct Model buffer leaves:
  `ParameterInitializers.ones` for scale, `ParameterInitializers.zeros` for bias,
  `TensorFactory.zeros(..., Optional.empty(), false)` for running mean, and
  `TensorFactory.ones(..., Optional.empty(), false)` for running variance, in that order.
- Make `forward(input, context)` validate its two explicit call arguments, read each current state
  binding exactly once, require input rank at least two, normalize the stored channel axis against
  that input, and require the input channel Dimension to equal the layer's static feature
  Dimension structurally. This deliberately resolves rather than defers the persistent-state
  schema. Treat `context.mode()` as authoritative for this call. The method does not consult
  `mode()` again, create another context, or infer mode from compiler/runtime state.
- In `EVALUATION`, delegate exactly once to
  `input.batchNormInference(channelAxis, scale, bias, runningMean, runningVariance, epsilon)` and
  return its output. Do not replace either buffer.
- In `TRAINING`, delegate exactly once to
  `input.batchNormTraining(channelAxis, scale, bias, runningMean, runningVariance, momentum,
  epsilon)`. Only after the complete non-null `BatchNormTrainingResult` exists, install
  `nextRunningMean` through `replaceBuffer("runningMean", ...)`, then install
  `nextRunningVariance` through `replaceBuffer("runningVariance", ...)`, then return `output`.
- Preserve exact per-call snapshots. Expressions already constructed from earlier parameter or
  buffer Tensors retain those exact inputs. Previously captured structural discovery maps retain
  the stable wrappers and therefore observe later successful binding replacement through
  `value()`.
- Add focused exact-surface, state-schema, initialization, context/mode, Model-delegation,
  transition-order, replacement-snapshot, validation, failure-side-effect, and no-execution tests.
- Add complete type/member and package Javadocs. After executable work and final NN testing, use a
  separate documentation-focused clean context to finalize those Javadocs, glossary impact,
  planning evidence, no-change conclusions, generated Javadoc, Markdown, scope, and whitespace.

## Out of scope

- An affine-free form; optional scale, bias, running mean, or running variance; nullable state;
  implicit ones/zeros inside a forward producer; or a partially affine Model call.
- `BatchNorm1d`, `BatchNorm2d`, `BatchNorm3d`, NCHW/NHWC aliases, a fixed batch axis, multiple or
  inferred channel axes, grouped normalization, synchronized/distributed statistics, or rank-
  specific wrappers. This `BatchNorm` uses Model's rank-at-least-two arbitrary-axis contract.
- Zero feature count, dynamic state Shape, scalar/full-rank/broadcast state, mixed state types,
  integral/BOOL state, frozen affine parameters, or caller-selected buffer gradient eligibility.
- Deferred equality between a dynamic input channel and the fixed state extent. Generic Model may
  carry that obligation, but this stateful layer requires an exact structural match so training
  cannot replace a fixed-schema buffer with a dynamically shaped next-statistic expression.
- A default channel axis, default data type, default epsilon, default momentum, raw binary64
  hyperparameters, configurable parameter/buffer initializers, RNG, or retained source.
- A batch counter, cumulative momentum, optional momentum, unbiased forward normalization,
  population running-variance update, variance repair/clamp, or changes to Model formulas.
- A public result carrier for the layer, public saved batch statistics, public sibling lookup,
  explicit next-statistic return, buffer setter, public buffer replacement, or another state
  transition type. Current buffers expose the installed next expressions through their stable
  wrappers and normal discovery APIs.
- A no-context `forward(Tensor)` overload. This first mode-sensitive layer consumes the existing
  explicit immutable context contract. Task 0011 remains responsible for any narrow shared unary
  composition contract required by `Sequential`.
- Reading `Module.mode()` during forward after accepting a context, rejecting a context created by
  another module, attaching an origin to `ForwardContext`, or introducing a recursive context.
- Atomic two-buffer replacement, rollback, locking, versioning, compare-and-set, thread safety,
  transaction receipts, state dictionaries, checkpoints, serialization, or cross-call/session
  coordination. The existing protected replacements remain individual operations.
- Eager execution of next statistics, host-storage reads or writes, in-place mutation, backend
  state, hidden runtime publication, automatic reuse across prepared runs, or a promise that
  installing an expression has numerically evaluated it.
- Optimizer algorithms, parameter groups, training sessions/steps, gradient publication,
  optimizer state, compiler capture/autograd/saved-value lifetime, runtime/prepare/Engine behavior,
  backend support/lowering/kernels, numerical conformance, or integration execution.
- Padding, embedding-row policy, convolution/pooling geometry, Dropout random-state threading,
  loss behavior, and every other unrelated layer concern. They neither affect the selected
  channel axis nor justify widening this task.
- Any new Model kind, attrs, result carrier, Tensor overload, Shape/data-type/scalar rule, Gradle
  dependency, architecture rule/test, global-roadmap change, CPU work, or unrelated refactor.
- Detailed specifications for NN 0009–0011.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md): NN state/mode ownership, Tensor
  invariants, extension dependency direction, and optimizer/training lifecycle.
- [Current architecture index](../../../../architecture/current-architecture-plan.md).
- [ADR 0007: Neural-network module and training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md).
- [Module boundaries](../../../../architecture/module-boundaries.md).
- [Dependency rules](../../../../architecture/dependency-rules.md).
- [Training graph](../../../../architecture/training-graph.md).
- [NN master plan](../master-plan.md).
- [Planning guide](../../../planning-guide.md).
- [Documentation rules](../../../../developer-guide/documentation-rules.md) with the General,
  Planning, and API/Javadoc profiles.
- [Completed NN task 0007](0007-embedding-layer.md) and its completed foundation dependencies.
- [Completed Model task 0021B: batch-normalization inference](../../../modules/model/tasks/0021b-batch-normalization-inference.md).
- [Completed Model task 0021C: batch-normalization training and statistic transition](../../../modules/model/tasks/0021c-batch-normalization-training-and-statistic-transition.md).
- [Tensor API batch-normalization inference](../../../../api/tensor-api.md#batch-normalization-inference-expressions).
- [Tensor API batch-normalization training](../../../../api/tensor-api.md#batch-normalization-training-and-statistic-transition-expressions).
- [Training API](../../../../api/training-api.md).

## Architecture constraints

- `extensions/nn` continues to depend only on `modules/model`. Production must not import
  training, compiler, runtime, prepare, Engine, or concrete backend code.
- NN owns layer state, train/eval composition, and protected direct buffer binding replacement.
  Model remains the sole owner of batch-normalization formulas, rank/axis/vector validation,
  promotion, output metadata, multi-output producer provenance, and exact scalar semantics.
- Tensor identity, descriptor, and provenance remain immutable. Replacing a Buffer binding changes
  which exact Tensor a later `value()` call returns; it never mutates the old Tensor, an existing
  expression, or producer output.
- `ForwardContext` is immutable NN composition metadata. It must not become mutable runtime state,
  a compile request, an execution handle, a backend flag, or a source of hidden global mode.
- The training Model occurrence is a pure five-output producer. The layer consumes only the public
  `output`, `nextRunningMean`, and `nextRunningVariance` carrier components and must not expose or
  reconstruct the saved producer slots.
- Existing `Module.replaceBuffer` is sufficient because the two Model next-statistic components
  are non-null exact Tensor expressions and direct buffer names are permanently declared. This
  task must not widen Buffer mutation access or add generic multi-binding atomicity.
- The authorized NN parallel exception is implementation-order only. Preserve every unrelated
  worktree change exactly, especially CPU source/tests/docs/planning, concurrent glossary edits,
  and the global roadmap.

## Public API

`BatchNorm` declares exactly:

```java
public BatchNorm(
        int channelAxis,
        Tensor scale,
        Tensor bias,
        Tensor runningMean,
        Tensor runningVariance,
        ScalarValue momentum,
        ScalarValue epsilon)

public BatchNorm(
        long featureCount,
        int channelAxis,
        DataType dataType,
        ScalarValue momentum,
        ScalarValue epsilon)

public Parameter scale()
public Parameter bias()
public Buffer runningMean()
public Buffer runningVariance()
public Tensor forward(Tensor input, ForwardContext context)
```

| Member | Contract |
|---|---|
| supplied constructor | Retains four exact caller Tensors plus exact scalar references after complete validation; creates no Tensor. |
| initialized constructor | Creates `[featureCount]` one scale, zero bias, zero running mean, and one running variance in exact order. |
| `scale()` / `bias()` | Return the exact stable parameter wrappers declared by this layer. |
| `runningMean()` / `runningVariance()` | Return the exact stable buffer wrappers; they expose current bindings but no public replacement. |
| `forward(input, context)` | Requires exact input-channel/state Dimension equality, then uses the explicit immutable context snapshot to select one current Model expression and returns only normalized output. |

No public or protected member beyond these declarations is added by `BatchNorm`. Inherited final
`Module` APIs remain available normally.

## State contract

| Kind | Local name | Initial Shape | Initial type | Initial `requiresGrad` | Initialized value |
|---|---|---|---|---|---|
| Parameter | `scale` | `[featureCount]` | exact configured floating type | `true` | exact typed one |
| Parameter | `bias` | `[featureCount]` | exact configured floating type | `true` | exact typed zero |
| Buffer | `runningMean` | `[featureCount]` | exact configured floating type | `false` | exact typed zero |
| Buffer | `runningVariance` | `[featureCount]` | exact configured floating type | `false` | exact typed one |

The supplied constructor requires the same initial schema and flags. `featureCount` is the static
positive extent of `scale` axis zero. Parameters and buffers remain separate discovery domains;
the table does not imply one combined declaration list or transaction.

After a successful training call, each buffer retains the exact corresponding next-statistic
Tensor from that one shared Model producer. Its Shape is a new rank-one Shape containing the exact
input channel Dimension, which the layer has already required to equal the static state Dimension
structurally. Its type equals the stored scalar/state type because Model accepts the call
only when momentum and epsilon exactly match the promoted result. Its gradient flag follows Model
dependencies and is not rewritten by NN.

## Constructor validation and side-effect order

### Supplied-state constructor

Validate before declaring any state:

1. require `channelAxis >= 0`;
2. null-check `scale`, `bias`, `runningMean`, `runningVariance`, `momentum`, then `epsilon`;
3. require scale floating type, `requiresGrad == true`, rank one, fully static Shape, then positive
   feature extent;
4. require bias floating type, `requiresGrad == true`, rank one, fully static Shape, exact scale
   data type, then structural Shape equality with scale;
5. require running mean floating type, `requiresGrad == false`, rank one, fully static Shape,
   exact scale data type, then structural Shape equality with scale;
6. apply the same checks to running variance;
7. construct `BatchNormTrainingAttrs(channelAxis, momentum, epsilon)` only for the existing
   intrinsic axis/momentum/epsilon validation and exact-reference retention precedent; discard
   this temporary value and construct no Tensor operation;
8. require momentum data type to equal the common state type, then epsilon data type to equal it;
9. retain configuration and declare scale, bias, running mean, and running variance under the
   exact names above.

Every local failure creates no Tensor, producer, output, storage, or Tensor ID and declares no
state on a returned object. Supplied Tensors and scalar values are never mutated or evaluated.

### Initialized constructor

Validate and construct in this order:

1. require `channelAxis >= 0`;
2. null-check `dataType`, `momentum`, then `epsilon`;
3. require `featureCount > 0`;
4. require floating data type;
5. apply the same intrinsic attrs validation, then exact momentum-type and epsilon-type checks;
6. create and retain `Shape.of(featureCount)`;
7. create scale through `ParameterInitializers.ones`;
8. create bias through `ParameterInitializers.zeros`;
9. create running mean through `TensorFactory.zeros` with no label and
   `requiresGrad == false`;
10. create running variance through `TensorFactory.ones` with no label and
    `requiresGrad == false`;
11. only after all four Tensors exist, declare the two parameters and two buffers in their
    respective orders.

Caller-controlled validation completes before Tensor allocation or identifier consumption.
Model Java-array-limit, checked arithmetic, allocation, and identifier failures retain their
current effects. A failure during later creation returns no `BatchNorm`; already created leaves
and consumed IDs are not rolled back. No random source or draw exists.

## Forward validation, snapshots, and transition order

`forward(input, context)` performs exactly:

1. null-check `input`, then `context`;
2. read current `scale`, `bias`, `runningMean`, and `runningVariance` bindings once, in that order;
3. require input rank at least two, normalize the stored non-negative channel axis through the
   current input Shape, and require that exact input Dimension to equal the retained static feature
   Dimension structurally; an unresolved or unequal Dimension fails locally;
4. read `context.mode()` once and branch on that immutable value;
5. for `EVALUATION`, call Model inference once and return it without any binding replacement;
6. for `TRAINING`, call Model training once using the four captured bindings and retained exact
   scalars;
7. only after the complete result exists, replace direct buffer `runningMean` with the exact
   `nextRunningMean` component;
8. replace direct buffer `runningVariance` with the exact `nextRunningVariance` component;
9. return the exact `output` component.

The supplied context, not the module's later `mode()`, is authoritative. Therefore a context
captured before `eval()` or `train()` preserves its original behavior. Callers normally pass
`layer.forwardContext()`; a context from another module is accepted because `ForwardContext`
contains mode only and has no origin identity.

A Model validation or factory failure occurs before buffer replacement and leaves both exact old
bindings current. Partial five-output Tensor-ID allocation retains Model's existing no-rollback
effect but still changes no buffer because no complete result exists. On a supported successful
call, both protected replacements are non-null direct-name operations over permanent layer
declarations and therefore install both result components. The specified mean-then-variance order
does not create a general atomicity guarantee: the existing individual replacement contract has
no rollback or transaction if an internal invariant is corrupted or future code introduces a
new failure.

Neither forward branch changes module mode or the context. Parameter replacement, buffer
transition, mode changes, and forward construction remain non-thread-safe as a combined activity;
callers must externally coordinate them when one consistent state snapshot matters.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.nn.layers` — existing concrete stateful layer package.
- `io.github.pho001.synaptik.nn.module` — existing `Module`, `Parameter`, `Buffer`, and
  `ForwardContext` contracts.
- `io.github.pho001.synaptik.nn.initialization` — existing exact one/zero parameter policies.
- `io.github.pho001.synaptik.model.tensor` — existing inference/training receivers, result
  carrier, Tensor factories, metadata, and provenance.
- `io.github.pho001.synaptik.model.datatype` and `.shape` — exact type, scalar, and Shape values.
- `io.github.pho001.synaptik.model.operation.normalization` — intrinsic attrs validation and
  focused test inspection only; production does not construct an operation directly.

Packages added or changed:

- No package is added. The existing `io.github.pho001.synaptik.nn.layers` package gains one final
  public class and its package documentation is extended for mode-sensitive state transitions.

Type placement:

- `io.github.pho001.synaptik.nn.layers.BatchNorm` — concrete stateful affine normalization layer.
- `io.github.pho001.synaptik.nn.layers.BatchNormTest` — exact public surface, supplied state,
  forward context, Model provenance, transition, failure, and snapshot tests.
- `io.github.pho001.synaptik.nn.layers.BatchNormInitializationTest` — initialized values,
  metadata, validation order, identifier effects, and all-three-floating-type tests.

## Affected files

Expected production files:

- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/package-info.java`.
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/BatchNorm.java`.

Expected test files:

- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/BatchNormTest.java`.
- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/BatchNormInitializationTest.java`.

Expected documentation and planning files:

- `docs/glossary.md` — update the existing batch-normalization and NN module/buffer entries with
  the current layer owner, state names, context-selected behavior, and expression-binding
  transition without changing Model mathematics.
- `docs/planning/extensions/nn/master-plan.md`.
- this task specification.

## Maximum scope

This task may create or modify exactly the seven paths listed above: two production Java files,
two NN test files, and three documentation/planning files. If implementation needs another type,
test owner, API document, module dependency, Model helper, public buffer API, context change,
atomic transition facility, architecture test, or eighth path, stop and propose a focused
follow-up instead of expanding this task.

## Acceptance criteria

- Public final `BatchNorm extends Module` declares exactly the two constructors, four state
  accessors, and one `forward(Tensor, ForwardContext)` method in the public API section, with no
  other declared public/protected member and no no-context overload.
- Supplied construction retains the exact four Tensor and two scalar references after the complete
  specified validation order. It declares exact names, stable wrappers, separate parameter/buffer
  order, same static positive rank-one Shape/type, parameter gradient eligibility, and initial
  buffer non-gradient eligibility without Tensor-ID or expression side effects.
- Initialized construction supports exactly FLOAT64, FLOAT32, and BFLOAT16; uses explicit positive
  feature count, non-negative channel axis, exact typed momentum/epsilon, one/zero/zero/one state,
  and exact four-ID creation order without an RNG or configurable policy.
- State accessor and direct/recursive discovery tests prove exact wrapper/value identity and
  names. Buffer exposes no public/protected update method; Parameter retains only its existing
  compatible public replacement capability.
- Evaluation uses the exact context snapshot, reads the four current bindings once, creates one
  Model `BATCH_NORM_INFERENCE` occurrence with exact ordered inputs and attrs after exact channel-
  schema validation, returns its output, and preserves exact buffer bindings.
- Training uses the exact context snapshot, creates one Model `BATCH_NORM_TRAINING` producer,
  installs producer slots one and two as mean then variance buffer bindings, returns exact slot
  zero, and exposes no saved slots or layer result carrier.
- Tests lock Shape/type/gradient/layout metadata, shared producer and output indices, ordered
  provenance, exact attrs scalar identity, state replacement order, structural discovery snapshot
  observation, earlier-expression stability, and repeated-call chaining without executing values.
- A captured training context remains training after `eval()`; a captured evaluation context
  remains evaluation after `train()`. Forward never changes mode/context, and a context's mode is
  the only mode value used by that call.
- Compatible scale/bias replacement appears in later calls only. Successful training buffer
  transition appears in later calls only. Earlier expressions retain old state. No atomic or
  thread-safe cross-binding claim is made.
- Null, axis, state schema, scalar, input rank/axis/exact-channel, reduction-domain, promotion/scalar-
  type, identifier-exhaustion, and allocation failures follow the layer-local or inherited Model
  order. Pre-result failures preserve both exact buffers; partial Model output-ID consumption is
  tested only to the extent needed to prove no buffer transition.
- Public/package Javadocs document purpose, arbitrary-axis semantics, state schema/names,
  initialization, typed scalars, context authority, Model delegation, buffer installation order,
  result/provenance, replacement snapshots, validation/failures, threading, and no-execution/
  no-session boundaries with complete tags.
- A separate clean-context documentation pass finalizes affected Javadocs, package docs, glossary,
  planning evidence, no-change conclusions, generated Javadoc, Markdown, exact scope/status, and
  whitespace before the task becomes Complete.
- No Model, Training, build, dependency, architecture, architecture-test, Tensor/Compile/Training
  API, compiler/runtime/prepare/Engine/backend, CPU, global-roadmap, later-task, or unrelated path
  enters the diff.

## Tests / validation

Validation tier: task validation for the single affected `extensions/nn` module plus the targeted
documentation pass.

Implementation pass runs focused tests while developing:

```bash
./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.layers.BatchNormTest --tests io.github.pho001.synaptik.nn.layers.BatchNormInitializationTest
```

After executable Java stabilizes, run exactly one final NN suite:

```bash
./gradlew :extensions:nn:test
```

The focused tests own the exact surface, constructor/state policy, context branch, pure Model
producer/provenance delegation, transition order, snapshots, and failure side effects. Existing
Model 0021B–0021C tests remain authoritative for exhaustive mathematical, Shape, scalar,
multi-output, and identifier semantics; do not repeat them or assert numerical execution.

The separate documentation pass runs after final Javadoc edits:

```bash
./gradlew :extensions:nn:javadoc
git diff --check
```

It also validates local Markdown links and anchors, balanced fences, terminal newlines, trailing
whitespace, generated `BatchNorm` and package pages, exact seven-path scope, package placement,
public surface, forbidden imports, 0001–0007 Complete, exactly 0008 Ready before implementation
and Complete only after all evidence, 0009–0011 Draft, and absence of detailed 0009–0011 task
files. It preserves unrelated CPU/global-roadmap/glossary work and may reuse the final NN test
evidence when executable Java has not changed.

Repository-wide and architecture-test validation are deferred to the NN capability checkpoint or
CI. This task changes one existing model-only extension, no Gradle edge, dependency rule,
architecture boundary, shared build contract, backend behavior, or end-to-end execution path.

## Dependencies

- NN 0001–0007 are Complete, including stable mode/context, parameter schema, protected direct
  buffer replacement, declarative one/zero parameter initialization, and concrete-layer patterns.
- Model tasks 0021B and 0021C are Complete with exact public inference/training methods,
  `BatchNormTrainingResult`, typed attrs, and multi-output provenance.
- Model `Shape`, `DataType`, `ScalarValue`, `TensorFactory`, producer, and identifier contracts are
  Complete and provide every state and failure fact used here.
- Accepted ADR 0007 and the model-only NN dependency direction remain unchanged.
- The user-authorized NN parallel exception remains in force; this task's seven implementation
  paths must remain disjoint from concurrent CPU/global-roadmap work.

## Follow-up tasks

- NN 0009 remains Draft and separately owns explicit `GraphRngState` threading and evaluation
  bypass for Dropout.
- NN 0010 remains Draft and separately owns state-dictionary/checkpoint schema, multi-binding
  validation/atomic load, and serialization boundaries. It must not infer atomicity from this
  layer's sequential training transition.
- NN 0011 remains Draft and may introduce a narrow unary composition contract only for the actual
  `Sequential` consumer. It must decide how mode-sensitive explicit-context layers participate.
- `extensions/training` later owns cross-step execution, publication, optimizer updates, and any
  coordination needed to publish or persist evaluated running statistics. This layer adds no
  session or runtime transport.
- Affine-free, rank-specific, cumulative-momentum/counter, synchronized/distributed, serialization,
  device/dtype conversion, and custom-initializer variants remain deliberately deferred until a
  concrete consumer justifies a separate task.

## Documentation and no-change review

- Finalize `BatchNorm` and `layers` package Javadocs and the two existing glossary areas listed in
  Affected files.
- `docs/api/tensor-api.md` remains unchanged because both delegated Model methods, formulas,
  output roles, and examples are already current; NN adds no Tensor API.
- `docs/api/training-api.md` remains unchanged because it already states that Buffer has no public
  update and a module subclass owns protected direct state transitions; this task adds no
  optimizer, session, publication, or prepared execution API.
- Training graph, architecture pages, ADR 0007, dependency rules/tests, and `ARCHITECTURE.md`
  remain unchanged because this task realizes their existing NN state/mode owner without changing
  a lifecycle or dependency.
- Model capabilities/master/tasks and Compile API remain unchanged because no Model or compiler
  contract changes. Training saved outputs remain Model producer metadata and compiler-owned
  lifetime inputs.
- Conformance/integration suites, Gradle, other modules, execution layers, backends, CPU planning,
  and the global roadmap remain unchanged because the task constructs metadata and changes only
  NN bindings; it claims no execution support.

The independent documentation pass must verify these conclusions against the final diff and
record any newly inaccurate statement rather than copying them blindly.

## Architecture impact

Expected impact: None.

This task implements the existing architecture allowance for module-owned parameters, buffers,
train/eval behavior, and Model-composed layers. If implementation requires hidden execution-side
state, public Buffer mutation, cross-binding atomicity, a new dependency, Model change, context
origin/lifecycle, or another architecture rule, stop and report the conflict instead of editing
architecture inside this task.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Work in the Synaptik repository without commit or push. Do not use any GSD skill or workflow.
Read AGENTS.md, ARCHITECTURE.md, the current architecture index, planning and documentation rules,
the NN master plan and completed tasks 0001–0007, Model tasks 0021B–0021C, current Module/
Parameter/Buffer/ForwardContext/initializer/layer contracts, final Tensor batch-normalization APIs
and tests, Shape/DataType/ScalarValue, Tensor and Training API references, glossary, dependency
enforcement, and docs/planning/extensions/nn/tasks/0008-batch-normalization-layer.md in full.

Implement task 0008 exactly inside its seven authorized paths. Preserve every unrelated worktree
change exactly, especially CPU source/tests/docs/planning, concurrent glossary edits, and the
global roadmap. Stop and report any architecture uncertainty, scope overflow, or need for another
public type, dependency, context/buffer API, atomic transition, Model change, or file.

Run the focused tests and one final NN suite after executable Java stabilizes. Then hand the actual
diff, exact Java evidence, and task contract to a separate documentation-focused clean context in
the same overall change. That pass finalizes package/type Javadocs, glossary and planning evidence,
generated Javadoc, Markdown, scope/status, and reasoned no-change conclusions without repeating
successful Java tests unless executable behavior changes. Mark Complete only after every
criterion passes.
```

## Documentation-agent handoff

Give the separate documentation-focused agent this task, the complete final implementation diff,
exact focused/final NN evidence and whether Java changed afterward, the exact public surface,
state schema and creation order, context authority, evaluation/training producer evidence,
mean-then-variance installation and failure boundaries, replacement snapshots, and the exact seven
authorized paths.

That agent independently reads repository instructions, the architecture contract/index and
focused NN/training boundaries, documentation rules plus General/API-Javadoc/Planning profiles,
ADR 0007, this task, final source/tests, generated Javadoc, Tensor/Training APIs, glossary, NN
master/task history, Model 0021B–0021C, batch-normalization Tensor/attrs/result contracts,
Module/Parameter/Buffer/ForwardContext/initializers/current layers, dependency tests, and Java 26
Gradle configuration. It finalizes only package/type Javadoc, glossary, this task, and NN master
plan. It records reasoned no-change conclusions for architecture/ADRs/tests, Tensor/Compile/
Training/public APIs, Model plans/contracts, conformance/integration, Gradle, execution layers,
backends, other modules, CPU work, and the global roadmap.

The documentation pass reuses successful Java evidence unless executable Java changes or it
records a concrete reason to rerun it. It records its clean-context identity, files/topics
reviewed, commands/results, glossary impact, limitations, and unresolved issues before completion.

## Local decisions

- The class is `BatchNorm`, not a rank-specific alias, because the completed Model contract is
  layout-neutral, accepts every rank at least two, and reduces every non-channel axis.
- Both affine operands are mandatory. This matches the exact Model signatures and avoids hidden
  constants, nullable state, and a second affine-free semantic composition.
- A non-negative channel axis is stored explicitly because construction has no input rank against
  which a negative axis could be normalized. Model validates it against each input and retains the
  same logical position.
- The explicit-context-only forward method is the first direct consumer of the foundation's mode
  snapshot. It lets a caller or parent compose one stable mode choice and prevents a second mode
  read during the call. A convenience or shared unary signature waits for task 0011.
- State type and Shape are exact and shared. This keeps stored momentum/epsilon exact-result-typed
  for every successful call and gives public parameter replacement its existing schema without a
  new layer conversion policy.
- Forward requires exact structural equality between the input channel Dimension and the retained
  static feature Dimension. Generic Model can defer unresolved equality, but installing such a
  result would silently change this layer's persistent buffer Shape because Buffer has no schema;
  the stateful layer therefore resolves this boundary locally.
- Parameters use existing declarative one/zero initializers. Buffers use direct Model one/zero
  factories with `requiresGrad == false` because `ParameterInitializers` intentionally creates
  trainable leaves and is not a buffer initializer abstraction.
- Training installs the exact returned next expressions after the complete Model result exists.
  Returning only normalized output keeps the layer API unary while stable Buffer wrappers expose
  the next symbolic state. Installation does not claim numerical evaluation or runtime persistence.
- Mean then variance replacement follows producer/result order. Both calls are non-failing under
  the final layer invariants, but the task deliberately does not widen this fact into generic
  atomic multi-binding or rollback semantics.

## Known limitations

- Only mandatory affine, positive fully static rank-one state is supported. There is no dynamic,
  empty, frozen, mixed-type, optional-state, lazy, or rank-specific layer form.
- The input may retain dynamic non-channel Dimensions, subject to Model's training-domain
  obligations, but its selected channel Dimension must equal the fixed static feature Dimension.
- Every successful call requires Model promotion to match stored momentum and epsilon exactly. A
  wider input can therefore fail rather than triggering an implicit state/scalar cast.
- Installed running statistics are Tensor expressions, not evaluated host values. Repeated
  training forward construction chains later expressions to prior transition outputs; execution,
  publication, cross-step detachment/materialization, and persistence remain future owners' work.
- A Buffer remains excluded from parameter discovery/optimizer targets, but a next-statistic
  expression may have `requiresGrad == true` through its Model dependencies. NN adds no stop-
  gradient operation or hidden rewrite.
- State mutation and forward construction are not thread-safe. There is no version, transaction,
  rollback, checkpoint, serialization, counter, or session contract.
- Forward construction proves no compiler capture, backend support, numerical value, tolerance,
  storage, publication, or execution behavior.

## Validation evidence

Planning context `/root/nn_0008_planning` read the repository instructions, authoritative
architecture contract/index, planning and documentation rules/profiles, NN master and completed
tasks 0001–0007, ADR 0007, Model master and completed tasks 0021B–0021C, final NN and Model source/
tests, Tensor/Training APIs, training graph, glossary, Shape/DataType/ScalarValue/operation
contracts, dependency enforcement, and Gradle files before selecting this API and scope.

Planning inspection found no architecture blocker. The existing explicit `ForwardContext` is a
complete immutable per-call mode selector, and protected direct buffer replacement can install the
two non-null public next-statistic Tensor expressions after one successful Model result. The
replacement methods are individual, so the task specifies deterministic sequential installation
and failure boundaries without inventing transaction or runtime semantics.

Targeted planning validation passed on 2026-08-14:

- a repository-local Markdown check resolved 33 local links, including both referenced heading
  anchors, across the NN master plan and this task;
- both planning files have balanced backtick/tilde fences, terminal newlines, and no trailing
  whitespace;
- status inspection found exactly one Ready NN row/task at 0008, kept 0009–0011 Draft, and found
  no detailed 0009–0011 task file;
- scoped status found exactly the modified NN master plan and this new task as the two NN planning
  paths; unrelated CPU, glossary, and global-roadmap work remained outside this planning diff;
- whole-worktree `git diff --check` passed with no output; and
- `git diff --no-index --check /dev/null
  docs/planning/extensions/nn/tasks/0008-batch-normalization-layer.md` produced no whitespace
  diagnostic (the expected status is one because the new file differs from `/dev/null`).

No Java test or Javadoc command was run because this context changed planning Markdown only.

Implementation context `/root/nn_0008_implementation` read the required repository instructions,
architecture and planning contracts, documentation rules/profiles, NN master and tasks 0001–0007,
ADR 0007, Model master/capabilities and tasks 0021B–0021C, final Model/NN source and tests,
Tensor/Training APIs, glossary, dependency/build contracts, and this task before editing. It found
no architecture, final-Model-API, dependency, package-placement, or seven-path scope conflict.

Executable development and final validation on 2026-08-14:

- The first focused command reached execution with 13 tests and one failure. The failure was
  test-only: one side-effect assertion took its Tensor-ID baseline before lazily constructing
  invalid test fixtures. The test now takes the baseline around a preconstructed failing request;
  production Java did not change for this correction.
- The next focused attempt failed test compilation with five errors after an assertion-helper
  signature was applied to the wrong adjacent helper. Correcting the two test-only signatures
  restored the intended exact scalar-identity assertions; production Java did not change.
- `./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.layers.BatchNormTest
  --tests io.github.pho001.synaptik.nn.layers.BatchNormInitializationTest` then passed twice. The
  second successful run covers the final executable tests after the last validation-matrix
  additions: 2 suites and 13 tests with zero failures, errors, or skips.
- After production and tests stabilized, the sole authoritative
  `./gradlew :extensions:nn:test` passed (`BUILD SUCCESSFUL`, five actionable tasks: one executed,
  four up-to-date). XML reports contain 14 suites and 74 tests with zero failures, errors, or
  skips. No executable Java or test changed afterward in this implementation context.
- Preliminary `./gradlew :extensions:nn:javadoc` passed (`BUILD SUCCESSFUL`, three actionable
  tasks: one executed, two up-to-date). Generated `BatchNorm.html` and the layers package page
  exist and contain the drafted state, context, transition, failure, threading, and no-execution
  contracts. This is implementation-draft evidence; the separate documentation context still
  owns final Javadoc review/editing and the authoritative post-edit Javadoc run.

Focused coverage locks public finality and the exact two-constructor/five-method surface; direct
state names/order/wrapper identity and separate discovery domains; every supplied/initialized
schema and scalar-validation stage; zero-ID caller prevalidation; all three floating initialized
types; exact one/zero/zero/one values and four-ID order; context authority when module mode later
changes; exact inference/training kind, attrs, ordered inputs, producer/output positions, metadata,
freshness, and scalar identity; evaluation no-transition; training mean-then-variance installation;
structural snapshot observation, earlier-expression retention, repeated-call chaining, compatible
parameter replacement, local exact-channel rejection, inherited Model failures, pre-result buffer
preservation, partial output-ID exhaustion, and a reflective corrupted-invariant proof of the
documented non-transactional second-replacement boundary. Tests inspect metadata and provenance
only and make no numerical-execution claim.

Preliminary manual validation passed:

- `javap -public` showed final `BatchNorm extends Module` with exactly the two planned
  constructors and `scale`, `bias`, `runningMean`, `runningVariance`, and
  `forward(Tensor, ForwardContext)`.
- An independent Java 26 reflection program passed the same finality/superclass and exact
  two-constructor/five-visible-method checks. The focused test also proves Buffer still exposes
  only public `name` and `value`, with no public/protected update method.
- Production import scans found only Model, existing NN module/initializer, and JDK imports and no
  training/compiler/runtime/prepare/Engine/backend import. The NN Gradle project retains its sole
  Model dependency.
- Generated-page inspection found the exact state names, explicit context forward, sequential
  transition, non-transactional/threading boundary, and package summary. Source inspection
  confirmed input-then-context null checks, one ordered read of each current binding, one context
  mode read, one branch delegation, and mean-then-variance replacement after a complete result.
- `git diff --check`, individual `git diff --no-index --check` for all three new Java files, and
  final-newline/trailing-whitespace checks for all four Java task paths passed with no diagnostic.

Preliminary no-change review concluded that Tensor/Compile/Training APIs, Model source/plans/
capabilities/tasks, architecture/ADRs/tests, conformance/integration suites, Gradle and dependency
rules, execution layers, backends, other modules, CPU work, and the global roadmap remain accurate
and outside this task: BatchNorm composes the final existing Model expressions and changes only NN
state bindings, with no semantic kind, compiler/backend/runtime behavior, dependency, or execution
claim. The mandatory documentation context must independently verify these conclusions, finalize
the package/type Javadocs, glossary and planning evidence, and run final documentation gates.

Independent documentation context `/root/nn_0008_docs` read the repository instructions,
architecture contract/index, planning and documentation rules plus General, API/Javadoc,
Planning, and Example profiles, NN master/tasks 0001–0008, ADR 0007, Model master and tasks
0021B–0021C, final Tensor batch-normalization APIs/attrs/result, Shape/Dimension/DataType/
ScalarValue contracts, Module/Parameter/Buffer/ForwardContext/initializers/current layers,
BatchNorm source/tests, Tensor/Compile/Training APIs, training graph, glossary, dependency/build
rules, and Java 26 configuration. It found no executable, architecture, API, documentation, or
scope blocker. The complete BatchNorm and layers package Javadocs already met the final contracts,
so it changed no Java source or test. It updated the NN glossary entry with the current layer
definition and declarative examples, updated the existing Model batch-normalization entry with the
distinct NN state-owner boundary, and finalized this task and NN master-plan evidence while
preserving concurrent CPU glossary and all other unrelated dirty changes exactly.

Final documentation validation on 2026-08-14 passed:

- `./gradlew :extensions:nn:javadoc` completed successfully with all three actionable tasks up to
  date; generated `BatchNorm.html` and `package-summary.html` were inspected for complete type,
  constructor, method, result, failure, context-authority, ordered-transition, non-transactional,
  threading, and no-execution contracts;
- `javap -public` and an independent Java 26 reflection program both showed final public
  `BatchNorm extends Module` with exactly two declared public constructors, five declared public
  methods, and no other declared public or protected API;
- production import and Gradle inspection found only Model, existing NN, and JDK imports and the
  unchanged sole `implementation(project(":modules:model"))` dependency;
- local Markdown target/anchor validation, balanced-fence checks, terminal-newline and trailing-
  whitespace checks passed for the task-owned documentation, including the glossary's concurrent
  CPU and new NN hunks;
- exact-scope inspection found the seven task-owned paths and distinguished all unrelated dirty
  CPU source/test/documentation/planning and global-roadmap paths; the two Java tests and
  executable `BatchNorm` source remained unchanged after the authoritative NN run;
- status inspection found NN 0001–0008 Complete, 0009–0011 Draft, no Ready NN task, and no
  detailed 0009–0011 task file; and
- final `git diff --check` passed without output.

The documentation pass reused the stable focused 2-suite/13-test and authoritative
14-suite/74-test NN evidence and did not repeat executable tests. Independent review confirmed the
recorded no-change conclusions: architecture/ADRs/tests, Tensor/Compile/Training APIs, Model plans/
capabilities/contracts, conformance/integration, Gradle/dependencies, runtime/prepare/Engine/
backends, other modules, later NN tasks, CPU work, and the global roadmap require no task-owned
change.

## Implementation notes

The isolated implementation context added final public `BatchNorm`, extended the existing layers
package contract, and added the two planned focused test owners. Supplied construction performs the
specified complete pre-declaration schema/scalar validation and retains exact state. Initialized
construction creates exact one scale, zero bias, zero mean, and one variance in four-ID order.
Forward validates a fixed exact channel Dimension, treats the supplied context as authoritative,
delegates to the matching final Model expression, and installs successful training statistics in
the specified sequential order. The independent documentation context finalized the glossary and
planning evidence, reviewed the complete package/type Javadocs unchanged, and passed all final
documentation gates. Executable Java and tests remained stable after their authoritative run.

## Completion summary

- Completed changes: Final `BatchNorm` state, initialization, context-selected Model composition,
  sequential running-statistic transition, focused regression coverage, package/type Javadocs,
  glossary definition/examples, and synchronized NN planning evidence are complete.
- Files changed or created: exactly `BatchNorm.java`, `layers/package-info.java`,
  `BatchNormTest.java`, `BatchNormInitializationTest.java`, `docs/glossary.md` for task-owned NN
  hunks atop preserved CPU edits, the NN master plan, and this task.
- Tests and validation: Reused final focused 2-suite/13-test and authoritative NN
  14-suite/74-test evidence with zero failures/errors/skips because no executable Java or test
  changed afterward. Final generated Javadoc, page inspection, `javap`, independent reflection,
  import/dependency, Markdown links/anchors/fences, exact-scope/status, newline, trailing-
  whitespace, and `git diff --check` checks passed.
- Documentation-agent review: Clean context `/root/nn_0008_docs` independently finalized the
  documentation and no-change review without repeating stable executable tests.
- Documentation and Javadoc impact: The existing complete BatchNorm/package Javadocs required no
  source edit. The glossary now defines the fourth concrete NN layer, distinguishes it from the
  existing Model batch-normalization meaning, and gives concise declarative evaluation/training
  Shape, provenance, and binding examples without numerical or execution claims. Planning records
  now contain final evidence and Complete status.
- Unresolved issues: None.
- Required follow-up: None for task 0008. NN 0009–0011 remain separately scoped Draft work.

Status: Complete

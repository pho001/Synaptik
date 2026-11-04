# Task 0008: [Batch Normalization](../../../../glossary.md#batch-normalization--batch-normalization-inference-and-training) Layer

> **Editorial reading context — not a status update.** The application passes both input and
> a ForwardContext to BatchNorm; that context, not an implicit training-session flag, selects the
> branch. Training forward replaces running-statistic bindings with next-statistic expressions,
> not evaluated host numbers. Buffers remain outside optimizer discovery even if those
> expressions depend on gradient-eligible inputs. Evaluation uses supplied running statistics
> without updating them. Today's [inference normalization semantics](../../../../reference/tensor/normalization.md#batch-normalization-inference-expressions)
> explain the Model operation, while [recovery participant requirements](../../../../reference/training/recovery.md#caller-coordinated-in-memory-recovery)
> explain why symbolic running-buffer transitions are not proof of step-aligned exact continuation.

## Status

Complete

## Goal

Add one final public affine `BatchNorm` [module](../../../../glossary.md#term-module) as the first [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) layer that owns both [trainable
parameters](../../../../glossary.md#trainable-parameter) and persistent [buffers](../../../../glossary.md#term-buffer). The layer has one explicit [layout](../../../../glossary.md#layout)-neutral [channel](../../../../glossary.md#term-channel) [axis](../../../../glossary.md#term-axis),
mandatory [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) scale and [bias](../../../../glossary.md#term-bias) [parameters](../../../../glossary.md#term-parameter), rank-one running-[mean](../../../../glossary.md#term-mean) and running-[variance](../../../../glossary.md#term-variance) buffers,
and exact typed [momentum](../../../../glossary.md#term-momentum) and [epsilon](../../../../glossary.md#epsilon). One typed forward call uses an [immutable](../../../../glossary.md#term-immutable)
`ForwardContext` [snapshot](../../../../glossary.md#term-snapshot) to select the existing [Model](../../../../glossary.md#term-model) [inference](../../../../glossary.md#term-inference) or training expression. A
successful training call installs the two returned next-[statistic](../../../../glossary.md#term-statistic) expressions into the stable
buffer wrappers; [evaluation](../../../../glossary.md#term-evaluation) changes no state.

[Mental model](../../../../glossary.md#term-mental-model):

```text
input + scale + bias + current running statistics + mode snapshot
  -> evaluation: Model batchNormInference -> output, buffers unchanged
  -> training:   Model batchNormTraining
                   -> output
                   -> next mean installed as runningMean
                   -> next variance installed as runningVariance
```

The installation is NN [binding](../../../../glossary.md#term-binding) replacement of pure [Tensor](../../../../glossary.md#tensor) expressions. It is not [eager](../../../../glossary.md#term-eager-evaluation) numerical
execution, in-place Tensor mutation, a [backend](../../../../glossary.md#backend) write, a [training session](../../../../glossary.md#training-session--trainingsession), or an [atomic](../../../../glossary.md#resource-transaction-and-rollback) generic
multi-binding [API](../../../../glossary.md#term-application-programming-interface-api).

## [Scope](../../../../glossary.md#term-scope)

- Add final public `io.github.pho001.synaptik.nn.layers.BatchNorm` extending [`Module`](../../../../glossary.md#term-module).
- Add exactly the public constructors, accessors, and forward method in the public API table
  below. Add no overload, default, alias, builder, options object, or generic layer interface.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Store one non-negative logical `channelAxis`. It is already non-negative as layer
  configuration, but Model validates it against each input rank and retains the normalized value
  in operation attributes. The layer neither assumes NCHW/NHWC nor distinguishes a batch axis.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [rank glossary](../../../../glossary.md#term-rank); [operation glossary](../../../../glossary.md#operation); [NCHW glossary](../../../../glossary.md#term-nchw); [NHWC glossary](../../../../glossary.md#term-nhwc); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [axis glossary](../../../../glossary.md#term-axis))</span>
- Declare mandatory [parameters](../../../../glossary.md#term-parameter) under exact local names `scale` then [`bias`](../../../../glossary.md#term-bias). Both have the same
  exact floating data type, the same positive fully static rank-one Shape <span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [Shape glossary](../../../../glossary.md#shape))</span>`[featureCount]`, and
  `requiresGrad == true` at declaration and after every public compatible parameter replacement.
- Declare mandatory [buffers](../../../../glossary.md#term-buffer) under exact local names `runningMean` then `runningVariance`. Supplied
  initial buffers have the same exact data type and Shape as the parameters and require
  `requiresGrad == false`. The initialized path creates exact typed zero [mean](../../../../glossary.md#term-mean) and one [variance](../../../../glossary.md#term-variance) with
  `requiresGrad == false`.
- Treat `Buffer` as optimizer-excluded module state, not as a promise that every later bound Tensor
  has <span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [Tensor glossary](../../../../glossary.md#tensor))</span>`requiresGrad == false`. A successful training Model result defines each next statistic's
  gradient-eligibility metadata. Because the old buffers begin non-gradient and the layer accepts
  only successful exact-result-typed calls, the first next mean/variance each have<span data-glossary-navigation="true"> ([statistic glossary](../../../../glossary.md#term-statistic); [gradient glossary](../../../../glossary.md#term-gradient); [metadata glossary](../../../../glossary.md#term-metadata))</span>
  `requiresGrad == input.descriptor().requiresGrad()`; a later expression may remain gradient-
  eligible through its old-statistic dependency. Do not detach, cast, copy, evaluate, or
  reconstruct those results.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- Retain exact [immutable](../../../../glossary.md#term-immutable) [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) references for momentum and epsilon. Momentum is the Model
  training convention: finite new-batch weight in <span data-glossary-navigation="true"> ([momentum glossary](../../../../glossary.md#term-momentum); [epsilon glossary](../../../../glossary.md#epsilon); [finite glossary](../../../../glossary.md#term-finite); [weight glossary](../../../../glossary.md#term-weight))</span>`[0, 1]`. Epsilon is finite and strictly
  positive. Both must have the exact common state data type; no raw `double`, default, conversion,
  Tensor scalar, or cumulative-average sentinel is allowed.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
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
  schema. Treat <span data-glossary-navigation="true"> ([arguments glossary](../../../../glossary.md#term-argument); [binding glossary](../../../../glossary.md#term-binding); [channel glossary](../../../../glossary.md#term-channel); [Dimension glossary](../../../../glossary.md#dimension); [feature glossary](../../../../glossary.md#term-feature); [schema glossary](../../../../glossary.md#term-schema))</span>`context.mode()` as authoritative for this call. The method does not consult
  `mode()` again, create another context, or infer mode from [compiler](../../../../glossary.md#term-compiler)/[runtime](../../../../glossary.md#term-runtime) state.
- In [`EVALUATION`](../../../../glossary.md#term-evaluation), delegate exactly once to
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
  transition-order, replacement-snapshot, validation, failure-side-effect, and no-execution tests.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [validation glossary](../../../../glossary.md#term-validation))</span>
- Add complete type/member and package Javadocs. After executable work and final NN testing, use a
  separate documentation-focused clean context to finalize those Javadocs, glossary impact,
  planning evidence, no-change conclusions, generated Javadoc, Markdown, scope, and whitespace.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [planning glossary](../../../../glossary.md#term-planning); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- An affine-free form; optional scale, bias, running mean, or running variance; nullable state;
  implicit ones/zeros inside a forward producer; or a partially affine Model call.<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [mean glossary](../../../../glossary.md#term-mean); [variance glossary](../../../../glossary.md#term-variance); [producer glossary](../../../../glossary.md#term-producer))</span>
- `BatchNorm1d`, `BatchNorm2d`, `BatchNorm3d`, NCHW/NHWC aliases, a fixed batch axis, multiple or
  inferred channel axes, grouped normalization, synchronized/distributed statistics, or rank-
  specific wrappers. This <span data-glossary-navigation="true"> ([NCHW glossary](../../../../glossary.md#term-nchw); [NHWC glossary](../../../../glossary.md#term-nhwc); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [channel glossary](../../../../glossary.md#term-channel); [normalization glossary](../../../../glossary.md#term-normalization))</span>`BatchNorm` uses [Model](../../../../glossary.md#term-model)'s [rank](../../../../glossary.md#term-rank)-at-least-two arbitrary-[axis](../../../../glossary.md#term-axis) contract.
- Zero feature count, dynamic state Shape, scalar/full-rank/broadcast state, mixed state types,
  integral/BOOL state, frozen affine parameters, or caller-selected buffer gradient eligibility.<span data-glossary-navigation="true"> ([feature glossary](../../../../glossary.md#term-software-feature); [Shape glossary](../../../../glossary.md#shape); [scalar glossary](../../../../glossary.md#term-scalar); [BOOL glossary](../../../../glossary.md#data-type--datatype); [parameters glossary](../../../../glossary.md#term-parameter); [buffer glossary](../../../../glossary.md#term-buffer); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Deferred equality between a dynamic input channel and the fixed state extent. Generic Model may
  carry that obligation, but this stateful layer requires an exact structural match so training
  cannot replace a fixed-schema buffer with a dynamically shaped next-statistic expression.<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent); [carry glossary](../../../../glossary.md#term-carry); [schema glossary](../../../../glossary.md#term-schema); [statistic glossary](../../../../glossary.md#term-statistic))</span>
- A default channel axis, default data type, default epsilon, default momentum, raw binary64
  hyperparameters, configurable parameter/buffer initializers, RNG, or retained source.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [epsilon glossary](../../../../glossary.md#epsilon); [momentum glossary](../../../../glossary.md#term-momentum); [binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [hyperparameters glossary](../../../../glossary.md#term-hyperparameter); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- A batch counter, cumulative momentum, optional momentum, unbiased forward normalization,
  population running-variance update, variance repair/clamp, or changes to Model formulas.
- A public result carrier for the layer, public saved batch statistics, public sibling lookup,
  explicit next-statistic return, buffer setter, public buffer replacement, or another state
  transition type. Current buffers expose the installed next expressions through their stable
  wrappers and normal discovery APIs.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- A no-context `forward(Tensor)` overload. This first mode-sensitive layer consumes the existing
  explicit immutable context contract. Task 0011 remains responsible for any narrow shared unary
  composition contract required by <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [unary glossary](../../../../glossary.md#term-unary))</span>`Sequential`.
- Reading `Module.mode()` during forward after accepting a context, rejecting a context created by
  another module, attaching an origin to <span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>`ForwardContext`, or introducing a recursive context.
- Atomic two-buffer replacement, rollback, locking, versioning, compare-and-set, thread safety,
  transaction receipts, state dictionaries, checkpoints, serialization, or cross-call/session
  coordination. The existing protected replacements remain individual operations.<span data-glossary-navigation="true"> ([Atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [rollback glossary](../../../../glossary.md#term-rollback); [compare-and-set glossary](../../../../glossary.md#term-compare-and-set-cas); [transaction glossary](../../../../glossary.md#term-transaction); [checkpoints glossary](../../../../glossary.md#term-checkpoint); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- Eager execution of next statistics, host-storage reads or writes, in-place mutation, backend
  state, hidden runtime publication, automatic reuse across prepared runs, or a promise that
  installing an expression has numerically evaluated it.<span data-glossary-navigation="true"> ([Eager glossary](../../../../glossary.md#term-eager-evaluation); [host glossary](../../../../glossary.md#term-host); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime); [publication glossary](../../../../glossary.md#term-publication))</span>
- Optimizer algorithms, parameter groups, training sessions/steps, gradient publication,
  optimizer state, compiler capture/autograd/saved-value lifetime, runtime/prepare/Engine behavior,
  backend support/lowering/kernels, numerical conformance, or integration execution.<span data-glossary-navigation="true"> ([Optimizer glossary](../../../../glossary.md#term-optimizer); [optimizer state glossary](../../../../glossary.md#term-optimizer-state); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [autograd glossary](../../../../glossary.md#autograd); [lifetime glossary](../../../../glossary.md#term-lifetime); [prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Padding, embedding-row policy, convolution/pooling geometry, Dropout random-state threading,
  loss behavior, and every other unrelated layer concern. They neither affect the selected
  channel axis nor justify widening this task.<span data-glossary-navigation="true"> ([Padding glossary](../../../../glossary.md#padding); [embedding glossary](../../../../glossary.md#embedding); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [pooling glossary](../../../../glossary.md#term-pooling); [Dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [state threading glossary](../../../../glossary.md#state-threading); [loss glossary](../../../../glossary.md#loss); [widening glossary](../../../../glossary.md#term-widening))</span>
- Any new Model kind, attrs, result carrier, Tensor overload, Shape/data-type/scalar rule, Gradle
  dependency, architecture rule/test, global-roadmap change, CPU work, or unrelated refactor.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Detailed specifications for [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0009–0011.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>: NN state/mode ownership, Tensor
  invariants, extension dependency direction, and optimizer/training lifecycle.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Tensor glossary](../../../../glossary.md#tensor); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [optimizer glossary](../../../../glossary.md#term-optimizer); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md).
- [ADR 0007: Neural-network module and training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module))</span>.
- [Module boundaries](../../../../architecture/module-boundaries.md).
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>.
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>.
- [NN master plan](../master-plan.md).
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>.
- [Documentation rules](../../../../developer-guide/documentation-rules.md) with the General,
  Planning, and API/Javadoc profiles.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- [Completed NN task 0007](0007-embedding-layer.md) and its completed foundation dependencies.
- [Completed Model task 0021B: batch-normalization inference](../../../modules/model/tasks/0021b-batch-normalization-inference.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [normalization glossary](../../../../glossary.md#term-normalization); [inference glossary](../../../../glossary.md#term-inference))</span>.
- [Completed Model task 0021C: batch-normalization training and statistic transition](../../../modules/model/tasks/0021c-batch-normalization-training-and-statistic-transition.md)<span data-glossary-navigation="true"> ([statistic glossary](../../../../glossary.md#term-statistic))</span>.
- [Tensor API batch-normalization inference](../../../../api/tensor-api.md#batch-normalization-inference-expressions).
- [Tensor API batch-normalization training](../../../../api/tensor-api.md#batch-normalization-training-and-statistic-transition-expressions).
- [Training API](../../../../api/training-api.md).

## Architecture constraints

- `extensions/nn` continues to depend only on `modules/model`. Production must not import
  training, compiler, runtime, prepare, Engine, or concrete backend code.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
- NN owns layer state, train/eval composition, and protected direct buffer binding replacement.
  Model remains the sole owner of batch-normalization formulas, rank/axis/vector validation,
  promotion, output metadata, multi-output producer provenance, and exact scalar semantics.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [buffer glossary](../../../../glossary.md#term-buffer); [binding glossary](../../../../glossary.md#term-binding); [owner glossary](../../../../glossary.md#term-owner); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [normalization glossary](../../../../glossary.md#term-normalization); [rank glossary](../../../../glossary.md#term-rank); [axis glossary](../../../../glossary.md#term-axis); [vector glossary](../../../../glossary.md#term-vector); [validation glossary](../../../../glossary.md#term-validation); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Tensor identity, descriptor, and provenance remain immutable. Replacing a Buffer binding changes
  which exact Tensor a later <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [identity glossary](../../../../glossary.md#term-identity-meanings); [descriptor glossary](../../../../glossary.md#term-descriptor); [immutable glossary](../../../../glossary.md#term-immutable))</span>`value()` call returns; it never mutates the old Tensor, an existing
  expression, or producer output.
- `ForwardContext` is immutable NN composition metadata. It must not become mutable runtime state,
  a compile request, an execution handle, a backend flag, or a source of hidden global mode.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [compile glossary](../../../../glossary.md#compile); [backend glossary](../../../../glossary.md#backend))</span>
- The training [Model](../../../../glossary.md#term-model) [occurrence](../../../../glossary.md#term-occurrence) is a pure five-output [producer](../../../../glossary.md#term-producer). The layer consumes only the public
  `output`, `nextRunningMean`, and `nextRunningVariance` carrier components and must not expose or
  reconstruct the saved producer slots.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Existing `Module.replaceBuffer` is sufficient because the two Model next-statistic components
  are non-null exact Tensor expressions and direct buffer names are permanently declared. This
  task must not widen Buffer mutation access or add generic multi-binding atomicity.<span data-glossary-navigation="true"> ([statistic glossary](../../../../glossary.md#term-statistic); [atomicity glossary](../../../../glossary.md#term-atomicity))</span>
- The authorized NN parallel exception is implementation-order only. Preserve every unrelated
  worktree change exactly, especially CPU source/tests/docs/planning, concurrent glossary edits,
  and the global roadmap.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [planning glossary](../../../../glossary.md#term-planning))</span>

## Public [API](../../../../glossary.md#term-application-programming-interface-api)

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
| supplied constructor | Retains four exact caller [Tensors](../../../../glossary.md#tensor) plus exact [scalar](../../../../glossary.md#term-scalar) references after complete [validation](../../../../glossary.md#term-validation); creates no Tensor. |
| initialized constructor | Creates `[featureCount]` one scale, zero [bias](../../../../glossary.md#term-bias), zero running [mean](../../../../glossary.md#term-mean), and one running [variance](../../../../glossary.md#term-variance) in exact order. |
| `scale()` / `bias()` | Return the exact stable [parameter](../../../../glossary.md#term-parameter) wrappers declared by this layer. |
| `runningMean()` / `runningVariance()` | Return the exact stable [buffer](../../../../glossary.md#term-buffer) wrappers; they expose current [bindings](../../../../glossary.md#term-binding) but no public replacement. |
| `forward(input, context)` | Requires exact input-[channel](../../../../glossary.md#term-channel)/state [Dimension](../../../../glossary.md#dimension) equality, then uses the explicit [immutable](../../../../glossary.md#term-immutable) context [snapshot](../../../../glossary.md#term-snapshot) to select one current [Model](../../../../glossary.md#term-model) expression and returns only normalized output. |

No public or protected member beyond these declarations is added by `BatchNorm`. Inherited final
[`Module`](../../../../glossary.md#term-module) APIs remain available normally.

## State contract

| Kind | Local name | Initial [Shape](../../../../glossary.md#shape) | Initial type | Initial `requiresGrad` | Initialized value |
|---|---|---|---|---|---|
| [Parameter](../../../../glossary.md#term-parameter) | `scale` | `[featureCount]` | exact configured floating type | `true` | exact typed one |
| Parameter | [`bias`](../../../../glossary.md#term-bias) | `[featureCount]` | exact configured floating type | `true` | exact typed zero |
| [Buffer](../../../../glossary.md#term-buffer) | `runningMean` | `[featureCount]` | exact configured floating type | `false` | exact typed zero |
| Buffer | `runningVariance` | `[featureCount]` | exact configured floating type | `false` | exact typed one |

The supplied constructor requires the same initial [schema](../../../../glossary.md#term-schema) and flags. `featureCount` is the static
positive [extent](../../../../glossary.md#term-extent) of `scale` [axis](../../../../glossary.md#term-axis) zero. Parameters and buffers remain separate discovery domains;
the table does not imply one combined declaration list or [transaction](../../../../glossary.md#term-transaction).

After a successful training call, each buffer retains the exact corresponding next-[statistic](../../../../glossary.md#term-statistic)
[Tensor](../../../../glossary.md#tensor) from that one shared [Model](../../../../glossary.md#term-model) [producer](../../../../glossary.md#term-producer). Its Shape is a new [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) Shape containing the exact
input [channel](../../../../glossary.md#term-channel) [Dimension](../../../../glossary.md#dimension), which the layer has already required to equal the static state Dimension
structurally. Its type equals the stored [scalar](../../../../glossary.md#term-scalar)/state type because Model accepts the call
only when [momentum](../../../../glossary.md#term-momentum) and [epsilon](../../../../glossary.md#epsilon) exactly match the promoted result. Its [gradient](../../../../glossary.md#term-gradient) flag follows Model
dependencies and is not rewritten by [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context).

## Constructor [validation](../../../../glossary.md#term-validation) and side-effect order

### Supplied-state constructor

Validate before declaring any state:

1. require `channelAxis >= 0`;
2. null-check `scale`, [`bias`](../../../../glossary.md#term-bias), `runningMean`, `runningVariance`, [`momentum`](../../../../glossary.md#term-momentum), then [`epsilon`](../../../../glossary.md#epsilon);
3. require scale floating type, `requiresGrad == true`, rank one, fully static Shape, then positive
   feature extent;<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [Shape glossary](../../../../glossary.md#shape); [feature glossary](../../../../glossary.md#term-software-feature); [extent glossary](../../../../glossary.md#term-extent))</span>
4. require bias floating type, `requiresGrad == true`, rank one, fully static Shape, exact scale
   data type, then structural Shape equality with scale;
5. require running [mean](../../../../glossary.md#term-mean) floating type, `requiresGrad == false`, rank one, fully static Shape,
   exact scale data type, then structural Shape equality with scale;
6. apply the same checks to running [variance](../../../../glossary.md#term-variance);
7. construct `BatchNormTrainingAttrs(channelAxis, momentum, epsilon)` only for the existing
   intrinsic axis/momentum/epsilon validation and exact-reference retention precedent; discard
   this temporary value and construct no Tensor operation;<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [validation glossary](../../../../glossary.md#term-validation); [operation glossary](../../../../glossary.md#operation))</span>
8. require momentum [data type](../../../../glossary.md#data-type--datatype) to equal the common state type, then epsilon data type to equal it;
9. retain configuration and declare scale, bias, running mean, and running variance under the
   exact names above.

Every local failure creates no [Tensor](../../../../glossary.md#tensor), [producer](../../../../glossary.md#term-producer), output, storage, or Tensor ID and declares no
state on a returned object. Supplied Tensors and [scalar](../../../../glossary.md#term-scalar) values are never mutated or evaluated.

### Initialized constructor

Validate and construct in this order:

1. require `channelAxis >= 0`;
2. null-check `dataType`, [`momentum`](../../../../glossary.md#term-momentum), then [`epsilon`](../../../../glossary.md#epsilon);
3. require `featureCount > 0`;
4. require floating [data type](../../../../glossary.md#data-type--datatype);
5. apply the same intrinsic attrs [validation](../../../../glossary.md#term-validation), then exact momentum-type and epsilon-type checks;
6. create and retain `Shape.of(featureCount)`;
7. create scale through `ParameterInitializers.ones`;
8. create [bias](../../../../glossary.md#term-bias) through `ParameterInitializers.zeros`;
9. create running [mean](../../../../glossary.md#term-mean) through `TensorFactory.zeros` with no label and
   `requiresGrad == false`;
10. create running [variance](../../../../glossary.md#term-variance) through `TensorFactory.ones` with no label and
    `requiresGrad == false`;
11. only after all four Tensors exist, declare the two parameters and two buffers in their
    respective orders.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter); [buffers glossary](../../../../glossary.md#term-buffer))</span>

Caller-controlled validation completes before [Tensor](../../../../glossary.md#tensor) allocation or identifier consumption.
[Model](../../../../glossary.md#term-model) Java-array-limit, checked arithmetic, allocation, and identifier failures retain their
current effects. A failure during later creation returns no `BatchNorm`; already created leaves
and consumed IDs are not rolled back. No random source or draw exists.

## Forward [validation](../../../../glossary.md#term-validation), snapshots, and transition order

`forward(input, context)` performs exactly:

1. null-check `input`, then `context`;
2. read current `scale`, [`bias`](../../../../glossary.md#term-bias), `runningMean`, and `runningVariance` [bindings](../../../../glossary.md#term-binding) once, in that order;
3. require input rank at least two, normalize the stored non-negative channel axis through the
   current input Shape, and require that exact input Dimension to equal the retained static feature
   Dimension structurally; an unresolved or unequal Dimension fails locally;<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [channel glossary](../../../../glossary.md#term-channel); [axis glossary](../../../../glossary.md#term-axis); [Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [feature glossary](../../../../glossary.md#term-feature))</span>
4. read `context.mode()` once and branch on that [immutable](../../../../glossary.md#term-immutable) value;
5. for [`EVALUATION`](../../../../glossary.md#term-evaluation), call [Model](../../../../glossary.md#term-model) [inference](../../../../glossary.md#term-inference) once and return it without any binding replacement;
6. for `TRAINING`, call Model training once using the four captured bindings and retained exact
   scalars;<span data-glossary-navigation="true"> ([scalars glossary](../../../../glossary.md#term-scalar))</span>
7. only after the complete result exists, replace direct [buffer](../../../../glossary.md#term-buffer) `runningMean` with the exact
   `nextRunningMean` component;
8. replace direct buffer `runningVariance` with the exact `nextRunningVariance` component;
9. return the exact `output` component.

The supplied context, not the [module](../../../../glossary.md#term-module)'s later `mode()`, is authoritative. Therefore a context
captured before `eval()` or `train()` preserves its original behavior. Callers normally pass
`layer.forwardContext()`; a context from another module is accepted because `ForwardContext`
contains mode only and has no origin [identity](../../../../glossary.md#term-identity-meanings).

A Model validation or factory failure occurs before buffer replacement and leaves both exact old
bindings current. Partial five-output [Tensor](../../../../glossary.md#tensor)-ID allocation retains Model's existing no-[rollback](../../../../glossary.md#term-rollback)
effect but still changes no buffer because no complete result exists. On a supported successful
call, both protected replacements are non-null direct-name operations over permanent layer
declarations and therefore install both result components. The specified [mean](../../../../glossary.md#term-mean)-then-[variance](../../../../glossary.md#term-variance) order
does not create a general [atomicity](../../../../glossary.md#term-atomicity) guarantee: the existing individual replacement contract has
no rollback or [transaction](../../../../glossary.md#term-transaction) if an internal [invariant](../../../../glossary.md#term-invariant) is corrupted or future code introduces a
new failure.

Neither forward branch changes module mode or the context. [Parameter](../../../../glossary.md#term-parameter) replacement, buffer
transition, mode changes, and forward construction remain non-[thread-safe](../../../../glossary.md#term-thread-safe) as a combined activity;
callers must externally [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) them when one consistent state [snapshot](../../../../glossary.md#term-snapshot) matters.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.nn.layers` — existing concrete stateful layer package.
- `io.github.pho001.synaptik.nn.module` — existing [`Module`](../../../../glossary.md#term-module), [`Parameter`](../../../../glossary.md#term-parameter), [`Buffer`](../../../../glossary.md#term-buffer), and
  `ForwardContext` contracts.
- `io.github.pho001.synaptik.nn.initialization` — existing exact one/zero parameter policies.
- `io.github.pho001.synaptik.model.tensor` — existing inference/training receivers, result
  carrier, Tensor factories, metadata, and provenance.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [carrier glossary](../../../../glossary.md#term-storage-carrier); [Tensor glossary](../../../../glossary.md#tensor); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance))</span>
- `io.github.pho001.synaptik.model.datatype` and `.shape` — exact type, [scalar](../../../../glossary.md#term-scalar), and [Shape](../../../../glossary.md#shape) values.
- `io.github.pho001.synaptik.model.operation.normalization` — intrinsic attrs validation and
  focused test inspection only; production does not construct an operation directly.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [operation glossary](../../../../glossary.md#operation))</span>

Packages added or changed:

- No package is added. The existing `io.github.pho001.synaptik.nn.layers` package gains one final
  public class and its package documentation is extended for mode-sensitive state transitions.

Type placement:

- `io.github.pho001.synaptik.nn.layers.BatchNorm` — concrete stateful affine [normalization](../../../../glossary.md#term-normalization) layer.
- `io.github.pho001.synaptik.nn.layers.BatchNormTest` — exact public surface, supplied state,
  forward context, Model provenance, transition, failure, and snapshot tests.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- `io.github.pho001.synaptik.nn.layers.BatchNormInitializationTest` — initialized values,
  metadata, validation order, identifier effects, and all-three-floating-type tests.

## Affected files

Expected production files:

- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/package-info.java`.
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/BatchNorm.java`.

Expected test files:

- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/BatchNormTest.java`.
- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/BatchNormInitializationTest.java`.

Expected documentation and [planning](../../../../glossary.md#term-planning) files:

- `docs/glossary.md` — update the existing batch-normalization and NN module/buffer entries with
  the current layer owner, state names, context-selected behavior, and expression-binding
  transition without changing Model mathematics.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [normalization glossary](../../../../glossary.md#term-normalization); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [module glossary](../../../../glossary.md#term-module); [buffer glossary](../../../../glossary.md#term-buffer); [owner glossary](../../../../glossary.md#term-owner); [binding glossary](../../../../glossary.md#term-binding); [Model glossary](../../../../glossary.md#term-model))</span>
- `docs/planning/extensions/nn/master-plan.md`.
- this task specification.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify exactly the seven paths listed above: two production Java files,
two [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) test files, and three documentation/[planning](../../../../glossary.md#term-planning) files. If implementation needs another type,
test [owner](../../../../glossary.md#term-owner), [API](../../../../glossary.md#term-application-programming-interface-api) document, [module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency), [Model](../../../../glossary.md#term-model) helper, public [buffer](../../../../glossary.md#term-buffer) API, context change,
[atomic](../../../../glossary.md#resource-transaction-and-rollback) transition facility, [architecture test](../../../../glossary.md#term-architecture-test), or eighth path, stop and propose a focused
follow-up instead of expanding this task.

## Acceptance criteria

- Public final `BatchNorm extends Module` declares exactly the two constructors, four state
  accessors, and one `forward(Tensor, ForwardContext)` method in the public API section, with no
  other declared public/protected member and no no-context overload.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Supplied construction retains the exact four Tensor and two scalar references after the complete
  specified validation order. It declares exact names, stable wrappers, separate parameter/buffer
  order, same static positive rank-one Shape/type, parameter gradient eligibility, and initial
  buffer non-gradient eligibility without Tensor-ID or expression side effects.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [scalar glossary](../../../../glossary.md#term-scalar); [validation glossary](../../../../glossary.md#term-validation); [parameter glossary](../../../../glossary.md#term-parameter); [buffer glossary](../../../../glossary.md#term-buffer); [rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [Shape glossary](../../../../glossary.md#shape); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Initialized construction supports exactly FLOAT64, FLOAT32, and BFLOAT16; uses explicit positive
  feature count, non-negative channel axis, exact typed momentum/epsilon, one/zero/zero/one state,
  and exact four-ID creation order without an RNG or configurable policy.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [feature glossary](../../../../glossary.md#term-feature); [channel glossary](../../../../glossary.md#term-channel); [axis glossary](../../../../glossary.md#term-axis); [momentum glossary](../../../../glossary.md#term-momentum); [epsilon glossary](../../../../glossary.md#epsilon); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- State accessor and direct/recursive discovery tests prove exact wrapper/value identity and
  names. Buffer exposes no public/protected update method; Parameter retains only its existing
  compatible public replacement capability.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Evaluation uses the exact context snapshot, reads the four current bindings once, creates one
  Model <span data-glossary-navigation="true"> ([Evaluation glossary](../../../../glossary.md#term-evaluation); [bindings glossary](../../../../glossary.md#term-binding))</span>`BATCH_NORM_INFERENCE` occurrence with exact ordered inputs and attrs after exact channel-
  schema validation, returns its output, and preserves exact buffer bindings.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [schema glossary](../../../../glossary.md#term-schema))</span>
- Training uses the exact context [snapshot](../../../../glossary.md#term-snapshot), creates one [Model](../../../../glossary.md#term-model) `BATCH_NORM_TRAINING` producer,
  installs producer slots one and two as mean then variance buffer bindings, returns exact slot
  zero, and exposes no saved slots or layer result carrier.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [mean glossary](../../../../glossary.md#term-mean); [variance glossary](../../../../glossary.md#term-variance); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Tests lock Shape/type/gradient/layout metadata, shared producer and output indices, ordered
  provenance, exact attrs scalar identity, state replacement order, structural discovery snapshot
  observation, earlier-expression stability, and repeated-call chaining without executing values.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance))</span>
- A captured training context remains training after `eval()`; a captured evaluation context
  remains evaluation after `train()`. Forward never changes mode/context, and a context's mode is
  the only mode value used by that call.
- Compatible scale/bias replacement appears in later calls only. Successful training buffer
  transition appears in later calls only. Earlier expressions retain old state. No atomic or
  thread-safe cross-binding claim is made.<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [thread-safe glossary](../../../../glossary.md#term-thread-safe))</span>
- Null, axis, state schema, scalar, input rank/axis/exact-channel, reduction-domain, promotion/scalar-
  type, identifier-exhaustion, and allocation failures follow the layer-local or inherited Model
  order. Pre-result failures preserve both exact buffers; partial Model output-ID consumption is
  tested only to the extent needed to prove no buffer transition.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [extent glossary](../../../../glossary.md#term-extent))</span>
- Public/package Javadocs document purpose, arbitrary-axis semantics, state schema/names,
  initialization, typed scalars, context authority, Model delegation, buffer installation order,
  result/provenance, replacement snapshots, validation/failures, threading, and no-execution/
  no-session boundaries with complete tags.
- A separate clean-context documentation pass finalizes affected Javadocs, package docs, glossary,
  planning evidence, no-change conclusions, generated Javadoc, Markdown, exact scope/status, and
  whitespace before the task becomes Complete.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- No Model, Training, build, dependency, architecture, architecture-test, Tensor/Compile/Training
  API, compiler/runtime/prepare/Engine/backend, CPU, global-roadmap, later-task, or unrelated path
  enters the diff.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Compile glossary](../../../../glossary.md#compile); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [CPU glossary](../../../../glossary.md#term-cpu))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Validation tier: task validation for the single affected `extensions/nn` [module](../../../../glossary.md#term-module) plus the targeted
documentation pass.

Implementation pass runs focused tests while developing:

```bash
./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.layers.BatchNormTest --tests io.github.pho001.synaptik.nn.layers.BatchNormInitializationTest
```

After executable Java stabilizes, [run](../../../../glossary.md#term-run) exactly one final [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) suite:

```bash
./gradlew :extensions:nn:test
```

The focused tests own the exact surface, constructor/state policy, context branch, pure [Model](../../../../glossary.md#term-model)
[producer](../../../../glossary.md#term-producer)/[provenance](../../../../glossary.md#provenance) delegation, transition order, snapshots, and failure side effects. Existing
Model 0021B–0021C tests remain authoritative for exhaustive mathematical, [Shape](../../../../glossary.md#shape), [scalar](../../../../glossary.md#term-scalar),
multi-output, and identifier semantics; do not repeat them or assert numerical execution.

The separate documentation pass runs after final [Javadoc](../../../../glossary.md#term-javadoc) edits:

```bash
./gradlew :extensions:nn:javadoc
git diff --check
```

It also validates local Markdown links and anchors, balanced fences, terminal newlines, trailing
whitespace, generated `BatchNorm` and package pages, exact seven-path [scope](../../../../glossary.md#term-scope), package placement,
public surface, forbidden imports, 0001–0007 Complete, exactly 0008 [Ready](../../../../glossary.md#term-ready) before implementation
and Complete only after all evidence, 0009–0011 Draft, and absence of detailed 0009–0011 task
files. It preserves unrelated [CPU](../../../../glossary.md#term-cpu)/global-roadmap/glossary work and may reuse the final NN test
evidence when executable Java has not changed.

Repository-wide and architecture-test validation are deferred to the NN capability [checkpoint](../../../../glossary.md#term-checkpoint) or
[CI](../../../../glossary.md#term-continuous-integration-ci). This task changes one existing model-only extension, no [Gradle](../../../../glossary.md#term-gradle) edge, [dependency](../../../../glossary.md#term-dependency) rule,
architecture boundary, shared build contract, [backend](../../../../glossary.md#backend) behavior, or end-to-end execution path.

## Dependencies

- NN 0001–0007 are Complete, including stable mode/context, parameter schema, protected direct
  buffer replacement, declarative one/zero parameter initialization, and concrete-layer patterns.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [schema glossary](../../../../glossary.md#term-schema); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- [Model](../../../../glossary.md#term-model) tasks 0021B and 0021C are Complete with exact public [inference](../../../../glossary.md#term-inference)/training methods,
  `BatchNormTrainingResult`, typed attrs, and multi-output [provenance](../../../../glossary.md#provenance).
- Model [`Shape`](../../../../glossary.md#shape), [`DataType`](../../../../glossary.md#data-type--datatype), [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue), `TensorFactory`, producer, and identifier contracts are
  Complete and provide every state and failure fact used here.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>
- Accepted [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0007 and the model-only [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [dependency direction](../../../../glossary.md#java-platform-and-build-tools) remain unchanged.
- The user-authorized NN parallel exception remains in force; this task's seven implementation
  paths must remain disjoint from concurrent CPU/global-roadmap work.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>

## Follow-up tasks

- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0009 remains Draft and separately owns explicit [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate) threading and evaluation
  bypass for Dropout.<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation); [Dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- NN 0010 remains Draft and separately owns state-dictionary/checkpoint schema, multi-binding
  validation/atomic load, and serialization boundaries. It must not infer atomicity from this
  layer's sequential training transition.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [schema glossary](../../../../glossary.md#term-schema); [binding glossary](../../../../glossary.md#term-binding); [validation glossary](../../../../glossary.md#term-validation); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [serialization glossary](../../../../glossary.md#term-serialization); [atomicity glossary](../../../../glossary.md#term-atomicity))</span>
- NN 0011 remains Draft and may introduce a narrow [unary](../../../../glossary.md#term-unary) composition contract only for the actual
  `Sequential` consumer. It must decide how mode-sensitive explicit-context layers participate.
- `extensions/training` later owns cross-step execution, publication, optimizer updates, and any
  coordination needed to publish or persist evaluated running statistics. This layer adds no
  session or runtime transport.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [optimizer glossary](../../../../glossary.md#term-optimizer); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Affine-free, rank-specific, cumulative-momentum/counter, synchronized/distributed, serialization,
  device/dtype conversion, and custom-initializer variants remain deliberately deferred until a
  concrete consumer justifies a separate task.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [momentum glossary](../../../../glossary.md#term-momentum); [device glossary](../../../../glossary.md#term-device); [dtype glossary](../../../../glossary.md#data-type--datatype))</span>

## Documentation and no-change review

- Finalize `BatchNorm` and `layers` package Javadocs and the two existing glossary areas listed in
  Affected files.
- `docs/api/tensor-api.md` remains unchanged because both delegated Model methods, formulas,
  output roles, and examples are already current; NN adds no Tensor API.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- `docs/api/training-api.md` remains unchanged because it already states that Buffer has no public
  update and a module subclass owns protected direct state transitions; this task adds no
  optimizer, session, publication, or prepared execution API.<span data-glossary-navigation="true"> ([Buffer glossary](../../../../glossary.md#term-buffer); [module glossary](../../../../glossary.md#term-module); [optimizer glossary](../../../../glossary.md#term-optimizer); [publication glossary](../../../../glossary.md#term-publication); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>
- [Training graph](../../../../glossary.md#training-graph), architecture pages, [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0007, [dependency](../../../../glossary.md#term-dependency) rules/tests, and `ARCHITECTURE.md`
  remain unchanged because this task realizes their existing NN state/mode owner without changing
  a lifecycle or dependency.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Model capabilities/master/tasks and Compile API remain unchanged because no Model or compiler
  contract changes. Training saved outputs remain Model producer metadata and compiler-owned
  lifetime inputs.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [compiler glossary](../../../../glossary.md#term-compiler); [producer glossary](../../../../glossary.md#term-producer); [metadata glossary](../../../../glossary.md#term-metadata); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- Conformance/integration suites, Gradle, other modules, execution layers, backends, CPU planning,
  and the global roadmap remain unchanged because the task constructs metadata and changes only
  NN bindings; it claims no execution support.<span data-glossary-navigation="true"> ([Conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [backends glossary](../../../../glossary.md#backend); [CPU glossary](../../../../glossary.md#term-cpu); [planning glossary](../../../../glossary.md#term-planning); [bindings glossary](../../../../glossary.md#term-binding))</span>

The independent documentation pass must verify these conclusions against the final diff and
record any newly inaccurate statement rather than copying them blindly.

## Architecture impact

Expected impact: None.

This task implements the existing architecture allowance for [module](../../../../glossary.md#term-module)-owned [parameters](../../../../glossary.md#term-parameter), [buffers](../../../../glossary.md#term-buffer),
train/eval behavior, and [Model](../../../../glossary.md#term-model)-composed layers. If implementation requires hidden execution-side
state, public Buffer mutation, cross-[binding](../../../../glossary.md#term-binding) [atomicity](../../../../glossary.md#term-atomicity), a new [dependency](../../../../glossary.md#term-dependency), Model change, context
origin/[lifecycle](../../../../glossary.md#lifecycle), or another architecture rule, stop and report the conflict instead of editing
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

## Documentation-agent [handoff](../../../../glossary.md#term-handoff)

Give the separate documentation-focused agent this task, the complete final implementation diff,
exact focused/final [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) evidence and whether Java changed afterward, the exact public surface,
state [schema](../../../../glossary.md#term-schema) and creation order, context authority, [evaluation](../../../../glossary.md#term-evaluation)/training [producer](../../../../glossary.md#term-producer) evidence,
[mean](../../../../glossary.md#term-mean)-then-[variance](../../../../glossary.md#term-variance) installation and failure boundaries, replacement snapshots, and the exact seven
authorized paths.

That agent independently reads repository instructions, the [architecture contract](../../../../glossary.md#architecture-contract)/index and
focused NN/training boundaries, documentation rules plus General/[API](../../../../glossary.md#term-application-programming-interface-api)-[Javadoc](../../../../glossary.md#term-javadoc)/[Planning](../../../../glossary.md#term-planning) profiles,
[ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0007, this task, final source/tests, generated Javadoc, [Tensor](../../../../glossary.md#tensor)/Training APIs, glossary, NN
master/task history, [Model](../../../../glossary.md#term-model) 0021B–0021C, [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[normalization](../../../../glossary.md#term-normalization) Tensor/attrs/result contracts,
[Module](../../../../glossary.md#term-module)/[Parameter](../../../../glossary.md#term-parameter)/[Buffer](../../../../glossary.md#term-buffer)/ForwardContext/initializers/current layers, [dependency](../../../../glossary.md#term-dependency) tests, and Java 26
[Gradle](../../../../glossary.md#term-gradle) configuration. It finalizes only package/type Javadoc, glossary, this task, and NN master
plan. It records reasoned no-change conclusions for architecture/ADRs/tests, Tensor/[Compile](../../../../glossary.md#compile)/
Training/public APIs, Model plans/contracts, [conformance](../../../../glossary.md#term-conformance)/integration, Gradle, execution layers,
[backends](../../../../glossary.md#backend), other modules, [CPU](../../../../glossary.md#term-cpu) work, and the global roadmap.

The documentation pass reuses successful Java evidence unless executable Java changes or it
records a concrete reason to rerun it. It records its clean-context [identity](../../../../glossary.md#term-identity-meanings), files/topics
reviewed, commands/results, glossary impact, limitations, and unresolved issues before completion.

## Local decisions

- The class is `BatchNorm`, not a rank-specific alias, because the completed Model contract is
  layout-neutral, accepts every rank at least two, and reduces every non-channel axis.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [alias glossary](../../../../glossary.md#aliasing-and-arena); [Model glossary](../../../../glossary.md#term-model); [layout glossary](../../../../glossary.md#layout); [channel glossary](../../../../glossary.md#term-channel); [axis glossary](../../../../glossary.md#term-axis))</span>
- Both affine operands are mandatory. This matches the exact Model signatures and avoids hidden
  constants, nullable state, and a second affine-free semantic composition.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [constants glossary](../../../../glossary.md#term-constant))</span>
- A non-negative channel axis is stored explicitly because construction has no input rank against
  which a negative axis could be normalized. Model validates it against each input and retains the
  same logical position.
- The explicit-context-only forward method is the first direct consumer of the foundation's mode
  snapshot. It lets a caller or parent compose one stable mode choice and prevents a second mode
  read during the call. A convenience or shared unary signature waits for task 0011.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [unary glossary](../../../../glossary.md#term-unary))</span>
- State type and Shape are exact and shared. This keeps stored momentum/epsilon exact-result-typed
  for every successful call and gives public parameter replacement its existing schema without a
  new layer conversion policy.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [momentum glossary](../../../../glossary.md#term-momentum); [epsilon glossary](../../../../glossary.md#epsilon); [parameter glossary](../../../../glossary.md#term-parameter); [schema glossary](../../../../glossary.md#term-schema))</span>
- Forward requires exact structural equality between the input channel Dimension and the retained
  static feature Dimension. Generic Model can defer unresolved equality, but installing such a
  result would silently change this layer's persistent buffer Shape because Buffer has no schema;
  the stateful layer therefore resolves this boundary locally.<span data-glossary-navigation="true"> ([structural equality glossary](../../../../glossary.md#term-structural-equality); [Dimension glossary](../../../../glossary.md#dimension); [feature glossary](../../../../glossary.md#term-feature); [persistent buffer glossary](../../../../glossary.md#model-module-buffer-and-scan-distinctions); [Buffer glossary](../../../../glossary.md#term-buffer))</span>
- Parameters use existing declarative one/zero initializers. Buffers use direct Model one/zero
  factories with `requiresGrad == false` because `ParameterInitializers` intentionally creates
  trainable leaves and is not a buffer initializer abstraction.<span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span>
- Training installs the exact returned next expressions after the complete Model result exists.
  Returning only normalized output keeps the layer API unary while stable Buffer wrappers expose
  the next symbolic state. Installation does not claim numerical evaluation or runtime persistence.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [evaluation glossary](../../../../glossary.md#term-evaluation); [runtime glossary](../../../../glossary.md#term-runtime); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- Mean then variance replacement follows producer/result order. Both calls are non-failing under
  the final layer invariants, but the task deliberately does not widen this fact into generic
  atomic multi-binding or rollback semantics.<span data-glossary-navigation="true"> ([Mean glossary](../../../../glossary.md#term-mean); [variance glossary](../../../../glossary.md#term-variance); [producer glossary](../../../../glossary.md#term-producer); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [binding glossary](../../../../glossary.md#term-binding); [rollback glossary](../../../../glossary.md#term-rollback))</span>

## Known limitations

- Only mandatory affine, positive fully static rank-one state is supported. There is no dynamic,
  empty, frozen, mixed-type, optional-state, lazy, or rank-specific layer form.<span data-glossary-navigation="true"> ([rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [lazy glossary](../../../../glossary.md#term-lazy-evaluation); [rank glossary](../../../../glossary.md#term-rank))</span>
- The input may retain dynamic non-channel Dimensions, subject to Model's training-domain
  obligations, but its selected channel Dimension must equal the fixed static feature Dimension.<span data-glossary-navigation="true"> ([channel glossary](../../../../glossary.md#term-channel); [Dimensions glossary](../../../../glossary.md#dimension); [Model glossary](../../../../glossary.md#term-model); [feature glossary](../../../../glossary.md#term-feature))</span>
- Every successful call requires Model promotion to match stored momentum and epsilon exactly. A
  wider input can therefore fail rather than triggering an implicit state/scalar cast.<span data-glossary-navigation="true"> ([momentum glossary](../../../../glossary.md#term-momentum); [epsilon glossary](../../../../glossary.md#epsilon); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Installed running statistics are Tensor expressions, not evaluated host values. Repeated
  training forward construction chains later expressions to prior transition outputs; execution,
  publication, cross-step detachment/materialization, and persistence remain future owners' work.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [host glossary](../../../../glossary.md#term-host); [publication glossary](../../../../glossary.md#term-publication); [materialization glossary](../../../../glossary.md#materialization); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- A Buffer remains excluded from parameter discovery/optimizer targets, but a next-statistic
  expression may have <span data-glossary-navigation="true"> ([Buffer glossary](../../../../glossary.md#term-buffer); [parameter glossary](../../../../glossary.md#term-parameter); [optimizer glossary](../../../../glossary.md#term-optimizer); [targets glossary](../../../../glossary.md#term-target); [statistic glossary](../../../../glossary.md#term-statistic))</span>`requiresGrad == true` through its Model dependencies. NN adds no stop-
  gradient operation or hidden rewrite.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [gradient glossary](../../../../glossary.md#term-gradient); [operation glossary](../../../../glossary.md#operation))</span>
- State mutation and forward construction are not thread-safe. There is no version, transaction,
  rollback, checkpoint, serialization, counter, or session contract.<span data-glossary-navigation="true"> ([thread-safe glossary](../../../../glossary.md#term-thread-safe); [transaction glossary](../../../../glossary.md#term-transaction); [rollback glossary](../../../../glossary.md#term-rollback); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- Forward construction proves no compiler capture, backend support, numerical value, tolerance,
  storage, publication, or execution behavior.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [backend glossary](../../../../glossary.md#backend); [tolerance glossary](../../../../glossary.md#term-tolerance))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) context `/root/nn_0008_planning` read the repository instructions, authoritative
[architecture contract](../../../../glossary.md#architecture-contract)/index, planning and documentation rules/profiles, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) master and completed
tasks 0001–0007, [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0007, [Model](../../../../glossary.md#term-model) master and completed tasks 0021B–0021C, final NN and Model source/
tests, [Tensor](../../../../glossary.md#tensor)/Training APIs, [training graph](../../../../glossary.md#training-graph), glossary, [Shape](../../../../glossary.md#shape)/[DataType](../../../../glossary.md#data-type--datatype)/[ScalarValue](../../../../glossary.md#typed-scalar-value--scalarvalue)/[operation](../../../../glossary.md#operation)
contracts, [dependency](../../../../glossary.md#term-dependency) enforcement, and [Gradle](../../../../glossary.md#term-gradle) files before selecting this [API](../../../../glossary.md#term-application-programming-interface-api) and [scope](../../../../glossary.md#term-scope).

Planning inspection found no architecture blocker. The existing explicit `ForwardContext` is a
complete [immutable](../../../../glossary.md#term-immutable) per-call mode selector, and protected direct [buffer](../../../../glossary.md#term-buffer) replacement can install the
two non-null public next-[statistic](../../../../glossary.md#term-statistic) Tensor expressions after one successful Model result. The
replacement methods are individual, so the task specifies deterministic sequential installation
and failure boundaries without inventing [transaction](../../../../glossary.md#term-transaction) or [runtime](../../../../glossary.md#term-runtime) semantics.

Targeted planning validation passed on 2026-08-14:

- a repository-local Markdown check resolved 33 local links, including both referenced heading
  anchors, across the NN master plan and this task;
- both planning files have balanced backtick/tilde fences, terminal newlines, and no trailing
  whitespace;
- status inspection found exactly one Ready NN row/task at 0008, kept 0009–0011 Draft, and found
  no detailed 0009–0011 task file;<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- scoped status found exactly the modified NN master plan and this new task as the two NN planning
  paths; unrelated CPU, glossary, and global-roadmap work remained outside this planning diff;
- whole-worktree `git diff --check` passed with no output; and
- `git diff --no-index --check /dev/null
  docs/planning/extensions/nn/tasks/0008-batch-normalization-layer.md` produced no whitespace
  diagnostic (the expected status is one because the new file differs from `/dev/null`).

No Java test or [Javadoc](../../../../glossary.md#term-javadoc) command was [run](../../../../glossary.md#term-run) because this context changed planning Markdown only.

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
  restored the intended exact scalar-identity assertions; production Java did not change.<span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [signature glossary](../../../../glossary.md#term-signature))</span>
- `./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.layers.BatchNormTest
  --tests io.github.pho001.synaptik.nn.layers.BatchNormInitializationTest` then passed twice. The
  second successful run covers the final executable tests after the last validation-matrix
  additions: 2 suites and 13 tests with zero failures, errors, or skips.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
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
state names/order/wrapper [identity](../../../../glossary.md#term-identity-meanings) and separate discovery domains; every supplied/initialized
[schema](../../../../glossary.md#term-schema) and [scalar](../../../../glossary.md#term-scalar)-validation stage; zero-ID caller prevalidation; all three floating initialized
types; exact one/zero/zero/one values and four-ID order; context authority when [module](../../../../glossary.md#term-module) mode later
changes; exact [inference](../../../../glossary.md#term-inference)/training kind, attrs, ordered inputs, [producer](../../../../glossary.md#term-producer)/output positions, [metadata](../../../../glossary.md#term-metadata),
freshness, and scalar identity; [evaluation](../../../../glossary.md#term-evaluation) no-transition; training [mean](../../../../glossary.md#term-mean)-then-[variance](../../../../glossary.md#term-variance) installation;
structural [snapshot](../../../../glossary.md#term-snapshot) observation, earlier-expression retention, repeated-call chaining, compatible
[parameter](../../../../glossary.md#term-parameter) replacement, local exact-[channel](../../../../glossary.md#term-channel) rejection, inherited Model failures, pre-result buffer
preservation, partial output-ID exhaustion, and a reflective corrupted-[invariant](../../../../glossary.md#term-invariant) proof of the
documented non-transactional second-replacement boundary. Tests inspect metadata and [provenance](../../../../glossary.md#provenance)
only and make no numerical-execution claim.

Preliminary manual validation passed:

- `javap -public` showed final `BatchNorm extends Module` with exactly the two planned
  constructors and `scale`, [`bias`](../../../../glossary.md#term-bias), `runningMean`, `runningVariance`, and
  `forward(Tensor, ForwardContext)`.
- An independent Java 26 reflection program passed the same finality/superclass and exact
  two-constructor/five-visible-method checks. The focused test also proves Buffer still exposes
  only public `name` and `value`, with no public/protected update method.
- Production import scans found only Model, existing NN module/initializer, and JDK imports and no
  training/compiler/runtime/prepare/Engine/backend import. The NN Gradle project retains its sole
  Model dependency.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>
- Generated-page inspection found the exact state names, explicit context forward, sequential
  transition, non-transactional/threading boundary, and package summary. Source inspection
  confirmed input-then-context null checks, one ordered read of each current binding, one context
  mode read, one branch delegation, and mean-then-variance replacement after a complete result.
- `git diff --check`, individual `git diff --no-index --check` for all three new Java files, and
  final-newline/trailing-whitespace checks for all four Java task paths passed with no diagnostic.

Preliminary no-change review concluded that Tensor/[Compile](../../../../glossary.md#compile)/Training APIs, Model source/plans/
capabilities/tasks, architecture/ADRs/tests, [conformance](../../../../glossary.md#term-conformance)/integration suites, Gradle and dependency
rules, execution layers, [backends](../../../../glossary.md#backend), other modules, [CPU](../../../../glossary.md#term-cpu) work, and the global roadmap remain accurate
and outside this task: BatchNorm composes the final existing Model expressions and changes only NN
state [bindings](../../../../glossary.md#term-binding), with no semantic kind, [compiler](../../../../glossary.md#term-compiler)/backend/runtime behavior, dependency, or execution
claim. The mandatory documentation context must independently verify these conclusions, finalize
the package/type Javadocs, glossary and planning evidence, and run final documentation gates.

Independent documentation context `/root/nn_0008_docs` read the repository instructions,
architecture contract/index, planning and documentation rules plus General, API/Javadoc,
Planning, and Example profiles, NN master/tasks 0001–0008, ADR 0007, Model master and tasks
0021B–0021C, final Tensor [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[normalization](../../../../glossary.md#term-normalization) APIs/attrs/result, Shape/[Dimension](../../../../glossary.md#dimension)/DataType/
ScalarValue contracts, Module/Parameter/Buffer/ForwardContext/initializers/current layers,
BatchNorm source/tests, Tensor/Compile/Training APIs, training graph, glossary, dependency/build
rules, and Java 26 configuration. It found no executable, architecture, API, documentation, or
scope blocker. The complete BatchNorm and layers package Javadocs already met the final contracts,
so it changed no Java source or test. It updated the NN glossary entry with the current layer
definition and declarative examples, updated the existing Model batch-normalization entry with the
distinct NN state-[owner](../../../../glossary.md#term-owner) boundary, and finalized this task and NN master-plan evidence while
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
  CPU and new NN hunks;<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- exact-scope inspection found the seven task-owned paths and distinguished all unrelated dirty
  CPU source/test/documentation/planning and global-roadmap paths; the two Java tests and
  executable `BatchNorm` source remained unchanged after the authoritative NN run;
- status inspection found NN 0001–0008 Complete, 0009–0011 Draft, no Ready NN task, and no
  detailed 0009–0011 task file; and
- final `git diff --check` passed without output.

The documentation pass reused the stable focused 2-suite/13-test and authoritative
14-suite/74-test NN evidence and did not repeat executable tests. Independent review confirmed the
recorded no-change conclusions: architecture/ADRs/tests, Tensor/Compile/Training APIs, Model plans/
capabilities/contracts, conformance/integration, Gradle/dependencies, runtime/[prepare](../../../../glossary.md#prepare)/[Engine](../../../../glossary.md#term-engine)/
backends, other modules, later NN tasks, CPU work, and the global roadmap require no task-owned
change.

## Implementation notes

The isolated implementation context added final public `BatchNorm`, extended the existing layers
package contract, and added the two planned focused test [owners](../../../../glossary.md#term-owner). Supplied construction performs the
specified complete pre-declaration [schema](../../../../glossary.md#term-schema)/[scalar](../../../../glossary.md#term-scalar) [validation](../../../../glossary.md#term-validation) and retains exact state. Initialized
construction creates exact one scale, zero [bias](../../../../glossary.md#term-bias), zero [mean](../../../../glossary.md#term-mean), and one [variance](../../../../glossary.md#term-variance) in four-ID order.
Forward validates a fixed exact [channel](../../../../glossary.md#term-channel) [Dimension](../../../../glossary.md#dimension), treats the supplied context as authoritative,
delegates to the matching final [Model](../../../../glossary.md#term-model) expression, and installs successful training statistics in
the specified sequential order. The independent documentation context finalized the glossary and
[planning](../../../../glossary.md#term-planning) evidence, reviewed the complete package/type Javadocs unchanged, and passed all final
documentation gates. Executable Java and tests remained stable after their authoritative [run](../../../../glossary.md#term-run).

## Completion summary

- Completed changes: Final `BatchNorm` state, initialization, context-selected Model composition,
  sequential running-statistic transition, focused regression coverage, package/type Javadocs,
  glossary definition/examples, and synchronized NN planning evidence are complete.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [statistic glossary](../../../../glossary.md#term-statistic); [regression glossary](../../../../glossary.md#term-regression); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: exactly `BatchNorm.java`, `layers/package-info.java`,
  `BatchNormTest.java`, `BatchNormInitializationTest.java`, `docs/glossary.md` for task-owned NN
  hunks atop preserved CPU edits, the NN master plan, and this task.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- Tests and validation: Reused final focused 2-suite/13-test and authoritative NN
  14-suite/74-test evidence with zero failures/errors/skips because no executable Java or test
  changed afterward. Final generated Javadoc, page inspection, <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`javap`, independent reflection,
  import/dependency, Markdown links/anchors/fences, exact-scope/status, newline, trailing-
  whitespace, and <span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` checks passed.
- Documentation-agent review: Clean context `/root/nn_0008_docs` independently finalized the
  documentation and no-change review without repeating stable executable tests.
- Documentation and Javadoc impact: The existing complete BatchNorm/package Javadocs required no
  source edit. The glossary now defines the fourth concrete NN layer, distinguishes it from the
  existing Model batch-normalization meaning, and gives concise declarative evaluation/training
  Shape, provenance, and binding examples without numerical or execution claims. Planning records
  now contain final evidence and Complete status.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [normalization glossary](../../../../glossary.md#term-normalization); [evaluation glossary](../../../../glossary.md#term-evaluation); [Shape glossary](../../../../glossary.md#shape); [provenance glossary](../../../../glossary.md#provenance); [binding glossary](../../../../glossary.md#term-binding))</span>
- Unresolved issues: None.
- Required follow-up: None for task 0008. [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0009–0011 remain separately scoped Draft work.

Status: Complete

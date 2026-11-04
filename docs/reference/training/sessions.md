<a id="training-api"></a>
# Training sessions and [owned Model](../../glossary.md#model-module-buffer-and-scan-distinctions) convenience

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/training/src/main/java/io/github/pho001/synaptik/training/TrainingSession.java)

A [training session](../../glossary.md#training-session--trainingsession) connects a [scalar](../../glossary.md#term-scalar) **[objective](../../glossary.md#term-objective)** (the quantity to optimize), selected
[parameters](../../glossary.md#term-parameter) and an [optimizer](../../glossary.md#term-optimizer) to one reusable [Engine](../../glossary.md#term-engine) execution. A **[gradient](../../glossary.md#term-gradient)** measures how that
objective changes with a selected parameter. The session computes and checks gradients before
committing an update; [model](../../glossary.md#term-model) train/[eval mode](../../glossary.md#inference-and-evaluation) is a separate forward-behavior choice and does not
itself start an [optimizer step](../../glossary.md#optimizer-and-stochastic-gradient-descent).

<a id="purpose-and-implementation-status"></a>
## Purpose and implementation status

`modules/training` ([Gradle](../../glossary.md#term-gradle) `:extensions:training`, unchanged Java packages) provides the public
[backend](../../glossary.md#backend)-neutral [optimizer](../../glossary.md#term-optimizer)/session [lifecycle](../../glossary.md#lifecycle). It accepts generic live [`TrainingParameters`](../../glossary.md#live-parameter-source--trainingparameters), [Model](../../glossary.md#term-model)
[Tensor](../../glossary.md#tensor)/storage types, and the public [Engine](../../glossary.md#term-engine) [facade](../../glossary.md#term-facade). Its only direct project dependencies are
public Model and Engine; it imports no [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context), [Runtime](../../glossary.md#term-runtime), [Prepare](../../glossary.md#prepare), tuning, or [concrete backend](../../glossary.md#concrete-backend).
One [`TrainingSession`](../../glossary.md#training-session--trainingsession) compiles and prepares a [scalar](../../glossary.md#term-scalar) forward/[backward graph](../../glossary.md#backward-graph) once, reuses it for
every [run](../../glossary.md#term-run), maps [gradients](../../glossary.md#term-gradient) by [Compiler](../../glossary.md#term-compiler) `targetIndex`, and [commits](../../glossary.md#commit-release-distribution-log-and-seed-meanings) optimizer changes only after
complete [validation](../../glossary.md#term-validation).

The public surface is:

| Type | Contract |
|---|---|
| `Optimizer` | [Sealed](../../glossary.md#term-sealed) marker for [immutable](../../glossary.md#term-immutable) built-in optimizer configuration. |
| `Sgd` | [Learning rate](../../glossary.md#term-learning-rate) plus optional [momentum](../../glossary.md#term-momentum), [dampening](../../glossary.md#term-dampening), coupled [weight decay](../../glossary.md#term-weight-decay), and [Nesterov](../../glossary.md#term-nesterov). |
| [`Adam`](../../glossary.md#adaptive-optimizer--adam-adamw) / [`AdamW`](../../glossary.md#adaptive-optimizer--adam-adamw) | [Bias](../../glossary.md#term-bias)-corrected first and second moments with coupled / decoupled weight decay. |
| [`ParameterGroup`](../../glossary.md#parameter-group--parametergroup) | Immutable ordered stable source `Entry.path()` values with one built-in optimizer configuration; recursive paths only for NN adapters. |
| `GradientMode` | `RESET_AND_STEP`, `ACCUMULATE`, or `ACCUMULATE_AND_STEP`. |
| `TrainingParameters` / `Entry` | Immutable ordered path/Tensor membership with live-[owner](../../glossary.md#term-owner) validation. |
| `TrainingSession` | Closeable owner of one Engine [PreparedExecution](../../glossary.md#prepared-execution--preparedexecution), [parameter](../../glossary.md#term-parameter) [bindings](../../glossary.md#term-binding), counters, accumulation, and optimizer slots. |
| [`TrainingParameterStorage`](../../glossary.md#native-parameter-copy--trainingparameterstorage) | Explicit copy of a complete live source into caller-owned shareable native [host](../../glossary.md#term-host) memory before session open. |
| `TrainingStep` | [Detached](../../glossary.md#detached-model-results) [objective](../../glossary.md#term-objective) and successful execution/update numbering for one run. |
| `TrainingState` | Detached immutable parameter/optimizer/accumulation [snapshot](../../glossary.md#term-snapshot) for strict in-memory restore. |

<a id="owned-model-training-convenience"></a>
## [Owned Model](../../glossary.md#model-module-buffer-and-scan-distinctions) training convenience

[NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)'s `Model.defineTensor`/[`graph`](../../glossary.md#graph) and `load`/`loadGraph` own one private static source.
`configureTraining(targetSignature, objectiveBuilder, optimizer)` invokes the pure builder once
per explicit attempt on owned outputs and a [target](../../glossary.md#term-target) [prototype](../../glossary.md#term-prototype). The list form takes explicit ordered
target prototypes and a list builder. The [scalar objective](../../glossary.md#loss) must be [gradient](../../glossary.md#term-gradient)-eligible [FLOAT32](../../glossary.md#term-float32)/[FLOAT64](../../glossary.md#term-float64);
arbitrary external [objective](../../glossary.md#term-objective) leaves reject. Configuration preserves [eval mode](../../glossary.md#inference-and-evaluation) and succeeds once.
Failed setup permits explicit retry but [Engine](../../glossary.md#term-engine) selection freezes before [callback](../../glossary.md#term-callback)/native/preparation
work, even on failure. Successful native [admission](../../glossary.md#term-admission) is not repeated on retry.

Qualified fixed recurrent sources now use that same configured [TrainingSession](../../glossary.md#training-session--trainingsession) and [Compiler](../../glossary.md#term-compiler)
first-order [AD](../../glossary.md#autograd), including mixed fixed/general sources, ordinary heads and final-state roots.
Fixed presence is not a separate [inference](../../glossary.md#term-inference)-only admission [guard](../../glossary.md#term-guard). Keep the source's original
fixed/general recipe, storage carriers and [rounding](../../glossary.md#term-rounding) boundaries; low-output objectives explicitly
cast to FLOAT32/FLOAT64. [CPU](../../glossary.md#term-cpu) low types retain the existing family and actual [derivative](../../glossary.md#term-derivative) guards.
Actual [backend](../../glossary.md#backend)/AD rejection follows the existing setup ordering: target/source [validation](../../glossary.md#term-validation),
Engine-choice freeze, objective callback and validation, inference-preparation closure and the
one native [parameter](../../glossary.md#term-parameter) transition, then session [compilation](../../glossary.md#compiler-engine-and-execution-stage-names)/preparation. A failed later attempt
can retain changed storage associations and closed inference preparations; it is not an
effect-free fixed-[scan](../../glossary.md#term-cumulative-scan) rejection. Retry retains a successful native transition.
No [optimizer](../../glossary.md#term-optimizer)/session history is restored by graph load; configure fresh training on restored values.

Qualified [general Scan](../../glossary.md#general-differentiable-scan) sources use the same [owners](../../glossary.md#term-owner), including CPU low types under actual forward,
first-order derivative and [binding](../../glossary.md#term-binding) guards. The explicit [F32](../../glossary.md#floating-point-precision-and-rounding)/[F64](../../glossary.md#floating-point-precision-and-rounding) `rnnScan` builder uses general
composition; [GRU](../../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../../glossary.md#term-long-short-term-memory-lstm), low [RNN](../../glossary.md#term-recurrent-neural-network-rnn) and old fixed files retain their fixed recipes and numerical sites.
Configure fresh training after graph load; graph files restore no session history.


`train(batch, target)` and the list form delegate `RESET_AND_STEP` to one TrainingSession.runValues.
NN maps declared [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)/target slots and owned [buffers](../../glossary.md#term-buffer)/[constants](../../glossary.md#term-constant) to final session input order;
Training retains parameters, [seed](../../glossary.md#term-seed), overlap checks and complete [transaction](../../glossary.md#term-transaction) validation. Inference
preparations close before the one [native parameter copy](../../glossary.md#native-parameter-copy--trainingparameterstorage); parameter [Tensor](../../glossary.md#tensor) [identity](../../glossary.md#term-identity-meanings) stays fixed.
Compute and graph save observe updated bytes, but a [model](../../glossary.md#term-model) file contains no objective or optimizer
continuation. The advanced TrainingSession [API](../../glossary.md#term-application-programming-interface-api) below is unchanged. See the
[owned training example](../../user-guide/training.md#train-an-owned-static-model).

<a id="session-construction-and-input-mapping"></a>
## Session construction and input mapping

`TrainingSession.open(engine, parameters, objective, optimizer)` borrows the [Engine](../../glossary.md#term-engine) and live source.
The [objective](../../glossary.md#term-objective) must be one [gradient](../../glossary.md#term-gradient)-eligible [scalar](../../glossary.md#term-scalar) [`FLOAT32`](../../glossary.md#term-float32) or [`FLOAT64`](../../glossary.md#term-float64) expression. The source
must expose at least one [parameter](../../glossary.md#term-parameter), and every parameter is an [identity](../../glossary.md#term-identity-meanings)-unique connected gradient
[target](../../glossary.md#term-target). Construction snapshots each path, exact [Tensor](../../glossary.md#tensor) and [host](../../glossary.md#term-host)-storage identity in caller
encounter order, never sorted, before calling `Engine.compile(...)` exactly once and
`Engine.prepare(...)` exactly once. It owns that prepared handle. Identity-based [`run`](../../glossary.md#term-run) executes
through `Engine.run(prepared, inputs)`; positional `runValues` executes through
`Engine.runValues(prepared, values)`. Both share the same [optimizer](../../glossary.md#term-optimizer) [transaction](../../glossary.md#term-transaction) and fixed [captures](../../glossary.md#term-capture-meanings).

`TrainingSession.open(engine, parameters, objective, defaultOptimizer, groups)` additionally assigns
an optimizer configuration to each path named by a [`ParameterGroup`](../../glossary.md#parameter-group--parametergroup). Paths are the stable
`Entry.path()` values in the `parameters.entries()` list (for [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context), `module.trainingParameters()`
derives recursive paths such as [`weight`](../../glossary.md#term-weight), [`bias`](../../glossary.md#term-bias), and dot-separated child paths). Unlisted
parameters use `defaultOptimizer`; an empty groups list is
equivalent to the four-[argument](../../glossary.md#term-argument) overload. Every parameter remains a gradient target. The
session defensively captures group order and each group's path order, and resolves one algorithm per
parameter in source encounter order before [compile](../../glossary.md#compile). A null or duplicate group/path, empty or
blank path, unknown path, or assigned [dtype](../../glossary.md#data-type--datatype)-incompatible coefficient fails before private [seed](../../glossary.md#term-seed)
allocation, compile, or [prepare](../../glossary.md#prepare). Group order and path order are exact state/restore identity,
not separate optimizer-step namespaces.

For example, after the [`Linear`](../../glossary.md#term-linear-meanings) parameters in the native-copy example below are [ready](../../glossary.md#term-ready), this
two-rate variation uses [learning rate](../../glossary.md#term-learning-rate) `0.05` for `weight` and the default `0.1` for `bias`:

```java
List<ParameterGroup> groups = List.of(
        new ParameterGroup(List.of("weight"), new Sgd(0.05)));
try (TrainingSession session = TrainingSession.open(
        engine, layer.trainingParameters(), objective, new Sgd(0.1), groups)) {
    session.run(List.of(input, target), GradientMode.RESET_AND_STEP);
    TrainingState snapshot = session.state();
    assert snapshot.parameterGroups().equals(groups);
}
```

Here `layer`, `objective`, `engine`, `input`, and `target` are the live values constructed in the
complete example under [explicit native parameter copy](parameters.md#explicit-native-parameter-copy)<span data-glossary-navigation="true"> ([native parameter copy glossary](../../glossary.md#native-parameter-copy--trainingparameterstorage))</span>. The
single run advances one global [optimizer step](../../glossary.md#optimizer-and-stochastic-gradient-descent), not one step per group.

For a three-parameter [Module](../../glossary.md#term-module) with recursive paths `first`, `second`, and `third`, this assignment
fragment shows all three algorithms in one session. The unlisted `first` uses [Adam](../../glossary.md#adaptive-optimizer--adam-adamw); the ordered
groups assign [SGD](../../glossary.md#term-stochastic-gradient-descent-sgd) to `second` and [AdamW](../../glossary.md#adaptive-optimizer--adam-adamw) to `third`. All three still update at the same global step:

```java
List<ParameterGroup> mixed = List.of(
        new ParameterGroup(List.of("second"), new Sgd(0.2)),
        new ParameterGroup(List.of("third"),
                new AdamW(0.1, 0.5, 0.25, 0.01, 0.1)));
TrainingSession session = TrainingSession.open(engine, module.trainingParameters(), objective,
        new Adam(0.1, 0.5, 0.25, 0.01, 0.1), mixed);
```

Here `module` must own those three live native parameters and `objective` must be a scalar
expression connected to all three; the caller closes `session` after use. The
[CPU integration test](../../../testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/TrainingSessionCpuIntegrationTest.java)<span data-glossary-navigation="true"> ([CPU glossary](../../glossary.md#term-cpu); [integration test glossary](../../glossary.md#term-integration-test))</span>
constructs and runs that complete scenario.

`session.inputs()` contains only the compiled inputs not supplied internally as captured
parameters or the private [cotangent seed](../../glossary.md#primal-adjoint-and-differentiation-seed), preserving relative final compiled order and including
targets and every other external bindable [leaf](../../glossary.md#term-leaf). Every `run(inputs, mode)` must provide each listed Tensor
identity exactly once; list order is irrelevant. Missing, duplicate, foreign, [descriptor](../../glossary.md#term-descriptor)-
incompatible, absent, dead, inaccessible, overlapping, or parameter-[aliasing](../../glossary.md#term-aliasing) storage is rejected.
Gradient publications must contain the [scalar objective](../../glossary.md#loss) followed by one [derivative](../../glossary.md#term-derivative)-order-one
[occurrence](../../glossary.md#term-occurrence) for each parameter, with the exact matching `targetIndex`.

`runValues(values, mode)` instead takes exactly one stored value per `session.inputs()` position,
with fresh TensorIds allowed. Full static descriptors must match exactly: dtype, [shape](../../glossary.md#shape), [layout](../../glossary.md#layout)
presence/geometry, and requiresGrad. Values are not expressions to execute or recapture. The
session inserts its captured parameters and private seed into final compiled positions without
copying or rebinding [prototype](../../glossary.md#term-prototype) storage. Engine checks referenced spans for resolved layouts and
[canonical](../../glossary.md#term-canonical-form) static geometry for absent layouts; unknown/overflowing geometry rejects. The extra
absent-layout capacity check does not change old `run` diagnostics or checks.

Training rejects nonempty external/external, external/parameter and seed overlap, including one
value repeated across matching positions; zero-byte overlap remains exempt. This is deliberately
stronger than generic Engine positional [binding](../../glossary.md#term-binding). Both methods borrow caller external storage only
through the [synchronous](../../glossary.md#term-synchronous) call and internal result closure, never for the next run. Keep it alive,
accessible and free of conflicting mutation until return. See the
[fresh-batch scalar example](../../user-guide/training.md#train-fresh-batches-and-targets)<span data-glossary-navigation="true"> ([batch glossary](../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span> for compiled-order mapping
and independently calculated [losses](../../glossary.md#loss) and updates.


---

[Next topic](parameters.md)

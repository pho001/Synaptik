# Train a [model](../glossary.md#term-model)

## Outcome

This guide runs the implemented public [backend](../glossary.md#backend)-neutral training [lifecycle](../glossary.md#lifecycle): one [scalar objective](../glossary.md#loss),
one reusable compiled/prepared session, repeated [SGD](../glossary.md#term-stochastic-gradient-descent-sgd), [Adam](../glossary.md#adaptive-optimizer--adam-adamw), or [AdamW](../glossary.md#adaptive-optimizer--adam-adamw) updates, optional [gradient
accumulation](../glossary.md#term-gradient-accumulation), and [detached](../glossary.md#detached-model-results) state. The worked example uses [CPU](../glossary.md#term-cpu) through `Engine.standard()`; the
session itself uses
the ordinary [Engine](../glossary.md#term-engine) path rather than a training-specific executor. Supported [Metal](../glossary.md#term-metal) or mixed
preparation follows the same lifecycle, but the low-[parameter](../glossary.md#term-parameter) CPU [qualification](../glossary.md#term-qualification) described here is
not a claim of [BFLOAT16](../glossary.md#term-bfloat16)/[FLOAT16](../glossary.md#term-float16) Metal training support.

A **parameter** is stored [model](../glossary.md#term-model) state an [optimizer](../glossary.md#term-optimizer) may change; a **[feature](../glossary.md#term-feature)** is an input
measurement, and a **[training target](../glossary.md#loss)** supplies the desired answer. A **[loss](../glossary.md#loss)** measures [prediction](../glossary.md#term-prediction)
error; a **reduction** combines [elementwise](../glossary.md#term-elementwise) losses into the one [scalar](../glossary.md#term-scalar) **[objective](../glossary.md#term-objective)** we minimize.
A **[gradient](../glossary.md#term-gradient)** measures that objective's local [sensitivity](../glossary.md#term-sensitivity) to the parameters. The **[learning
rate](../glossary.md#term-learning-rate)** scales an optimizer's update, rather than measuring accuracy.

For the introductory numbers, follow the [bias-free FLOAT32 `[2,1]` journey](../tutorials/index.md#the-experiment-in-concrete-terms)<span data-glossary-navigation="true"> ([bias glossary](../glossary.md#term-bias); [FLOAT32 glossary](../glossary.md#term-float32))</span>:
input `[1,3]`, [target](../glossary.md#term-target) `[2,6]`, [weight](../glossary.md#term-weight) `0`, summed squared error `40`, gradient `-40`,
and SGD rate `0.05` give weight `2`. The existing examples below address additional session
workflows with their own unchanged starting states. An **[epoch](../glossary.md#term-epoch-meanings)** is one pass through a [dataset](../glossary.md#term-dataset);
neither a loop iteration nor an accumulate-only [run](../glossary.md#term-run) necessarily completes one.

### Who performs each part of a training call?

For an owned Model, the application calls Model, which owns its private copied parameters and
configured session. For the explicit `TrainingSession` examples below, the application creates
the parameter storage and calls `TrainingSession.open(engine, parameters, objective, optimizer)`.
Opening a session **borrows**, rather than copies or adopts, that parameter storage.

TrainingSession in turn calls Engine to compile, prepare and execute the objective and
derivatives. Compiler constructs derivative formulas; Engine coordinates execution; Training
calculates and commits optimizer changes. A session's `run` borrows external inputs only for
the synchronous call and closes its internal result before returning detached `TrainingStep`
bytes. This is shorter than an explicit Engine `RunResult` lifetime, which lasts until the
application closes that result. The application closes the session before its borrowed Engine
and parameter Arena.

## Train an owned static [Model](../glossary.md#term-model)

For a supported static [graph](../glossary.md#graph), `Model.defineTensor` or `Model.load` can own the working [weights](../glossary.md#term-weight),
native [Arena](../glossary.md#term-arena) and cached [TrainingSession](../glossary.md#training-session--trainingsession) for you. Follow the complete
[owned Model example](running-models.md#owned-static-model-workflow)<span data-glossary-navigation="true"> ([owned Model glossary](../glossary.md#model-module-buffer-and-scan-distinctions))</span>: compute predicts 6 from
weight 2 and input 3, squared-error training against [target](../glossary.md#term-target) 1 reports [loss](../glossary.md#loss) 25, [SGD](../glossary.md#term-stochastic-gradient-descent-sgd)(0.1) changes
the private weight to -1, and restored compute on input 2 yields -2. External weight bytes remain 2.

Supply a pure [scalar](../glossary.md#term-scalar) [FLOAT32](../glossary.md#term-float32)/[FLOAT64](../glossary.md#term-float64) [objective](../glossary.md#term-objective) builder once with `configureTraining`; the builder
is invoked once per explicit attempt and is never retained for [runtime](../glossary.md#term-runtime). Do not [capture](../glossary.md#term-capture-meanings) arbitrary
external state leaves: a leaf has no producing expression, so capturing a stored Tensor from
outside the supplied arguments would add a hidden dependency. Configuration leaves eval mode
unchanged and succeeds once. Subsequent `train(batch, target)` calls supply current stored
values with exact descriptors and perform `RESET_AND_STEP`; list forms also support explicit
target prototypes. Training retains overlap, nonfinite and precommit checks.

**Eval mode** is a [Module](../glossary.md#term-module) behavior setting, distinct from **[evaluation](../glossary.md#term-evaluation)**, which assesses
[predictions](../glossary.md#term-prediction), and **[inference](../glossary.md#term-inference)**, which computes predictions without an [optimizer](../glossary.md#term-optimizer) update.
Here configuring an update mechanism does not toggle that setting. A **pure objective builder**
constructs the objective from the supplied prototypes without hidden [mutable](../glossary.md#term-mutable) dependencies.
**Cached** training setup retains reusable [compilation](../glossary.md#compiler-engine-and-execution-stage-names) and preparation, not a permanently fixed
prediction or [gradient](../glossary.md#term-gradient).
The inherited no-argument `model.train()` changes Module behavior, not parameters. In the
owned-source workflow it also changes captured structure and invalidates execution.
Use `model.train(batch,target)` for a configured optimizer transaction, and `compute(batch)`
for new predictions without an update; do not toggle modes around these calls.

[Engine](../glossary.md#term-engine) selection freezes before the first valid setup attempt, even on failure. A failed setup
may be retried explicitly on that selection without recopying successfully admitted native state.
Model close [releases](../glossary.md#commit-release-distribution-log-and-seed-meanings) the session before [parameter](../glossary.md#term-parameter) memory; [detached](../glossary.md#detached-model-results) TrainingStep remains readable.
Save records the current source/weights, not objective, [optimizer history](../glossary.md#term-optimizer-history), [RNG](../glossary.md#graph-rng-state--graphrngstate) or input progress.
For accumulation, parameter groups and detached optimizer-state restore, use the explicit session
workflow below rather than assuming that model-only load is [exact training continuation](../glossary.md#exact-training-continuation).

## Copy an initialized [Linear](../glossary.md#term-linear-meanings)

An initialized `Linear` has weight and bias values in Java heap storage before any execution.
In this explicit-session workflow the Java application calls
`TrainingParameterStorage.copyToSharedNative(parameters, arena)` after constructing its loss
expression. The helper copies bytes into the application's shared native Arena and updates
storage associations while keeping the same Tensor objects connected to the expression.
`TrainingSession.open` does not perform that copy. A shared Arena permits access from the
session's operation threads; keep it open through session close. Allocation may consume
capacity even if a later copy step fails, so choose the Arena lifetime beforehand.

This complete example starts with a [`FLOAT32`](../glossary.md#term-float32) one-input, one-output `Linear` and a [target](../glossary.md#term-target) of zero.
The input and target have native storage too, because the [CPU](../glossary.md#term-cpu) [Engine](../glossary.md#term-engine) needs live [host](../glossary.md#term-host) inputs.
[`TrainingParameters`](../glossary.md#live-parameter-source--trainingparameters) is a live ordered [admission](../glossary.md#term-admission) source. [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context)'s `layer.trainingParameters()`
retains the layer and its exact paths/wrappers/Tensors, revalidating the live tree rather than only
a frozen list. It does not freeze storage associations, so this example reuses the same source
after copying. For a non-NN [owner](../glossary.md#term-owner), implement that boundary as shown in the
[generic admission example](../reference/training/parameters.md#generic-live-parameter-admission).

**Native** memory is managed outside the Java heap. **Shareable** means the memory [scope](../glossary.md#term-scope)
permits the session's operation threads, not that concurrent unsynchronized writes are safe.
A **storage association** binds a Tensor to its actual memory. Copying bytes into a new store
while keeping Tensor [identity](../glossary.md#term-identity-meanings) is different from replacing the Tensor with another object.
The [ownership diagram](../explanation/ownership.md#a-result-is-not-always-an-independent-value)
shows why caller parameter storage must outlive dependent work.

```java
static Tensor nativeMatrix(Arena arena, float value) {
    Shape shape = Shape.of(1, 1);
    TensorDescriptor descriptor = new TensorDescriptor(
            DataType.FLOAT32, shape,
            Optional.of(LayoutDescriptor.contiguous(shape)), false);
    // The returned Tensor borrows arena storage; retaining it cannot keep a closed Arena live.
    MemorySegmentStorage storage = new MemorySegmentStorage(
            DataType.FLOAT32, 1, arena.allocate(Float.BYTES, Float.BYTES));
    storage.segment().set(ValueLayout.JAVA_FLOAT, 0, value);
    return TensorFactory.create(descriptor, Optional.empty(), Optional.of(storage));
}

Linear layer = new Linear(1, 1, true, DataType.FLOAT32, new Random(42));
Tensor originalWeight = layer.weight().value();
try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
    Tensor input = nativeMatrix(arena, 1.0f);
    Tensor target = nativeMatrix(arena, 0.0f);
    Tensor output = layer.forward(input);
    // forward/sub/mul build the expression; opening/running the session executes it later.
    Tensor error = output.sub(target);
    Tensor objective = error.mul(error).sum();

    // This call is explicit; open does not copy heap-backed parameters.
    TrainingParameters parameters = layer.trainingParameters();
    TrainingParameterStorage.copyToSharedNative(parameters, arena);
    assert layer.weight().value() == originalWeight;
    try (TrainingSession session = TrainingSession.open(
            engine, parameters, objective, new Sgd(0.1))) {
        // Each objective is measured before that call's committed optimizer update.
        float firstLoss = session.run(
                List.of(input, target), GradientMode.RESET_AND_STEP)
                .objective().bytes().getFloat();
        float secondLoss = session.run(
                List.of(input, target), GradientMode.RESET_AND_STEP)
                .objective().bytes().getFloat();
        assert secondLoss < firstLoss;
    }
} // Engine and parameter Arena close after the TrainingSession.
```

The second loss is lower than the first for this seeded example. The identity assertion shows
that the original weight Tensor remains the [graph](../glossary.md#graph) input after copying; it does not claim that
the old heap storage was closed or transferred. The caller retains [ownership](../glossary.md#term-ownership-meanings) of both source and
destination storage. Keep the shared Arena alive through session close, and never copy while a
session is open or another caller uses the [Module](../glossary.md#term-module)/storage. Without the copy, `TrainingSession.open`
rejects the heap-backed parameters before [compilation](../glossary.md#compiler-engine-and-execution-stage-names). A confined or closed destination Arena,
uninitialized deferred layer, non-[finite](../glossary.md#term-finite) source value, aliased parameter regions, unsupported
[layout](../glossary.md#layout), or invalid source storage also fails before any parameter association changes. Already
consumed destination Arena capacity is not individually recovered after allocation failure.

For details of raw-byte preservation and failure behavior, see the [Training API](../reference/training/parameters.md#explicit-native-parameter-copy)<span data-glossary-navigation="true"> ([API glossary](../glossary.md#term-application-programming-interface-api))</span>
and [architecture explanation](../architecture/training-graph.md#optimizer-as-a-backend-agnostic-step).

## Build a runnable [scalar](../glossary.md#term-scalar) problem

The first [optimizer](../glossary.md#term-optimizer) domain requires exact-capacity writable **shareable native** [parameter](../glossary.md#term-parameter) storage.
Parameters may use [`FLOAT64`](../glossary.md#term-float64), [`FLOAT32`](../glossary.md#term-float32), [`BFLOAT16`](../glossary.md#term-bfloat16), or [`FLOAT16`](../glossary.md#term-float16); the [scalar objective](../glossary.md#loss) remains
`FLOAT32` or `FLOAT64`. The example uses `FLOAT32` for both parameter and input and therefore gives
them an `Arena.ofShared()` [lifetime](../glossary.md#term-lifetime). [JVM](../glossary.md#term-java-virtual-machine-jvm)-[heap](../glossary.md#term-heap) storage created by convenience [Tensor](../glossary.md#tensor) factories is not
an accepted parameter [carrier](../glossary.md#term-storage-carrier) in this [lifecycle](../glossary.md#lifecycle).

For [BF16](../glossary.md#floating-point-precision-and-rounding)/[FP16](../glossary.md#term-fp16) parameter storage on [CPU](../glossary.md#term-cpu), cast the parameter explicitly to FLOAT32 before
numerical [graph](../glossary.md#graph) operations, for example `weight.cast(DataType.FLOAT32)`. The private [F32](../glossary.md#floating-point-precision-and-rounding)
optimizer master preserves update [precision](../glossary.md#term-precision); it does not silently convert a low graph or
bypass the [temporary numerical lock](../backend-guide/production-support.md#temporary-java-low-computation-lock).

Each parameter must also fit the in-memory state format: its logical payload and its
optimizer-width payload must each be no larger than `Integer.MAX_VALUE` bytes. The optimizer
width is eight bytes per FLOAT64 [element](../glossary.md#term-element) and four bytes per FLOAT32, BFLOAT16, or FLOAT16 element.
An oversized parameter is rejected when the session opens, before its own update [buffers](../glossary.md#term-buffer), the
private [seed](../glossary.md#term-seed), or the [compiled graph](../glossary.md#compiled-graph--compiledgraphmodel) are created. Earlier parameters may already have been captured.

```java
final class ScalarModel extends Module {
    private final Parameter weight;

    ScalarModel(Tensor initialWeight) {
        weight = parameter("weight", initialWeight);
    }

    Parameter weight() {
        return weight;
    }
}

static Tensor nativeScalar(Arena arena, float value, boolean requiresGrad) {
    // The application owns arena; TrainingSession borrows parameter bytes rather than adopting it.
    Shape shape = Shape.scalar();
    TensorDescriptor descriptor = new TensorDescriptor(
            DataType.FLOAT32,
            shape,
            Optional.of(LayoutDescriptor.contiguous(shape)),
            requiresGrad);
    MemorySegmentStorage storage = new MemorySegmentStorage(
            DataType.FLOAT32,
            1,
            arena.allocate(Float.BYTES, Float.BYTES));
    storage.segment().set(ValueLayout.JAVA_FLOAT, 0, value);
    return TensorFactory.create(descriptor, Optional.empty(), Optional.of(storage));
}
```

Create the forward/[loss](../glossary.md#loss) expression before opening the session. Here [SGD](../glossary.md#term-stochastic-gradient-descent-sgd) learns the scalar [target](../glossary.md#term-target)
`2.5` from initial [weight](../glossary.md#term-weight) `-4.0`:

```java
TrainingStep last;
try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
    ScalarModel model = new ScalarModel(nativeScalar(arena, -4.0f, true));
    Tensor target = nativeScalar(arena, 2.5f, false);
    Tensor error = model.weight().value().sub(target);
    Tensor loss = error.mul(error);

    // TrainingSession is closed before Engine and arena release the resources it borrows.
    try (TrainingSession training = TrainingSession.open(
            engine, model.trainingParameters(), loss, new Sgd(0.1))) {
        // inputs() excludes the captured parameter and private cotangent seed.
        assert training.inputs().size() == 1;
        assert training.inputs().getFirst().tensorId().equals(target.id());

        last = null;
        for (int iteration = 0; iteration < 40; iteration++) {
            last = training.run(List.of(target), GradientMode.RESET_AND_STEP);
        }

        float learned = model.weight().value().hostStorage().orElseThrow().segment()
                .get(ValueLayout.JAVA_FLOAT, 0);
        assert Math.abs(learned - 2.5f) < 0.001f;
        assert last.executionNumber() == 40;
        assert last.optimizerStep().orElseThrow() == 40;
    }
}

// TrainingStep owns detached canonical objective bytes.
float finalLoss = last.objective().bytes().getFloat();
```

### Why does the scalar approach the target?

For finite dimensionless weight $w$ and desired answer $t=2.5$, the objective is
$L(w)=(w-t)^2$ and its derivative is $L'(w)=2(w-t)$.
With learning rate $\eta=0.1$, update number $s\in\{0,\ldots,39\}$ gives
$w_{s+1}=w_s-0.1\cdot2(w_s-t)$, hence $w_{s+1}-t=0.8(w_s-t)$.
Starting at $w_0=-4$, the first derivative is $-13$ and the first new weight is
$-4-0.1(-13)=-2.7$. Each ideal real-number step shrinks the remaining error by a factor
of $0.8$; FLOAT32 rounds each actual update, so the code checks a tolerance after forty steps
instead of asserting a real-number identity on final bits.

`last.objective()` is the fortieth call's **pre-update** loss, not a new loss evaluated at the
weight read afterward. Reading it after session close is safe because the session returned
copied bytes, not borrowed workspace.

`TrainingSession.open` performs the only [gradient](../glossary.md#term-gradient) [compile](../glossary.md#compile) and the only preparation. Every [`run`](../glossary.md#term-run)
reuses that [Engine](../glossary.md#term-engine) [`PreparedExecution`](../glossary.md#prepared-execution--preparedexecution) through `engine.run(prepared, inputs)`.
Inputs to `run` may be supplied in any list order, but every [identity](../glossary.md#term-identity-meanings) listed by `training.inputs()`
must occur exactly once with a compatible live storage association. For a new stored [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) or
target Tensor, use positional `runValues` instead, as shown below.

## Train fresh [batches](../glossary.md#term-batch-machine-learning-and-hyperparameters) and [targets](../glossary.md#term-target)

Use the `ScalarModel` and `nativeScalar` definitions above with [CPU](../glossary.md#term-cpu) `Engine.standard()`. Start
with [weight](../glossary.md#term-weight) `2`, and build the squared-error [objective](../glossary.md#term-objective) `(weight * batch - target)^2` once. The
batch and target [prototypes](../glossary.md#term-prototype) fix [descriptors](../glossary.md#term-descriptor), not the values of future runs. Fresh values must
match [dtype](../glossary.md#data-type--datatype), fully static [shape](../glossary.md#shape), [layout](../glossary.md#layout) presence/geometry and requiresGrad exactly.

```java
TrainingStep first;
TrainingStep second;
try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
    Tensor weight = nativeScalar(arena, 2.0f, true);
    ScalarModel model = new ScalarModel(weight);
    Tensor batchPrototype = nativeScalar(arena, 1.0f, false);
    // These stored prototype values describe positions; runValues supplies fresh values instead.
    Tensor targetPrototype = nativeScalar(arena, 0.0f, false);
    Tensor error = weight.mul(batchPrototype).sub(targetPrototype);
    Tensor objective = error.mul(error);
    try (TrainingSession session = TrainingSession.open(
            engine, model.trainingParameters(), objective, new Sgd(0.1))) {
        try (Arena batchArena = Arena.ofShared()) {
            Tensor batch = nativeScalar(batchArena, 3.0f, false);
            Tensor target = nativeScalar(batchArena, 1.0f, false);
            var values = session.inputs().stream().map(input ->
                    input.tensorId().equals(batchPrototype.id()) ? batch : target).toList();
            first = session.runValues(values, GradientMode.RESET_AND_STEP);
            // Fresh inputs are borrowed only through this synchronous call's internal result close.
        } // Fresh storage may close as soon as the synchronous call has returned.
        assert first.objective().bytes().getFloat() == 25.0f;
        assert Math.abs(weight.hostStorage().orElseThrow().segment()
                .get(ValueLayout.JAVA_FLOAT, 0) + 1.0f) < 1e-6f;
        try (Arena batchArena = Arena.ofShared()) {
            Tensor batch = nativeScalar(batchArena, 2.0f, false);
            Tensor target = nativeScalar(batchArena, -3.0f, false);
            var values = session.inputs().stream().map(input ->
                    input.tensorId().equals(batchPrototype.id()) ? batch : target).toList();
            second = session.runValues(values, GradientMode.RESET_AND_STEP);
        }
        assert Math.abs(second.objective().bytes().getFloat() - 1.0f) < 1e-6f;
        assert Math.abs(weight.hostStorage().orElseThrow().segment()
                .get(ValueLayout.JAVA_FLOAT, 0) + 1.4f) < 1e-6f;
        assert session.optimizerSteps() == 2;
    }
}
assert first.objective().bytes().getFloat() == 25.0f;
```

For dimensionless weight $w$, batch value $x$ and desired answer $t$, define
$p=wx$, $L(w)=(p-t)^2$ and $g=dL/dw=2(wx-t)x$. Plain SGD with
learning rate $\eta=0.1$ uses $w_{\mathrm{new}}=w-\eta g$.
The first call has $w=2$, $x=3$, $t=1$, so $p=6$, $L=25$, $g=30$ and
$w_{\mathrm{new}}=2-0.1\cdot30=-1$.
The second call uses that new weight with $x=2$, $t=-3$, giving $p=-2$,
$L=1$, $g=4$ and $w_{\mathrm{new}}=-1-0.1\cdot4=-1.4$.
Each returned loss precedes its own update. These real-number calculations explain the
finite example; actual FLOAT32 arithmetic rounds, so the code uses tolerances for updated
weights. They are not an extension to nonfinite or empty-parameter training.
The mapping uses `session.inputs()` [metadata](../glossary.md#term-metadata), excluding [parameters](../glossary.md#term-parameter) and the private [seed](../glossary.md#term-seed); do not
assume expression construction order is final compiled order. Same-descriptor positions remain
positional: swapping batch and target changes the computation, not their [identities](../glossary.md#term-identity-meanings)' mapping.
The [CPU regression](../../testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/TrainingSessionCpuIntegrationTest.java)<span data-glossary-navigation="true"> ([regression glossary](../glossary.md#term-regression))</span>
executes this workflow and checks unchanged prototype values, parameter storage [identity](../glossary.md#term-identity-meanings), [graph](../glossary.md#graph)
reuse and [detached](../glossary.md#detached-model-results) results. Repeating one nonempty [Tensor](../glossary.md#tensor) in both positions rejects for overlap,
even when descriptors match. No prototype value is copied or rebound, and no new graph is built.

## Configure an [optimizer](../glossary.md#term-optimizer)

`new Sgd(learningRate)` selects plain [SGD](../glossary.md#term-stochastic-gradient-descent-sgd). The constructor with every coefficient is:

```java
// Configuration only; a successful step selects and applies the resulting update.
new Sgd(learningRate, momentum, dampening, weightDecay, nesterov)
```

The four-argument convenience uses zero dampening. To read the equations below, consider
one parameter element $w$ and its gradient $g$ on optimizer step $s\in\mathbb{Z}_{>0}$.
Let $\lambda\ge0$ be weight decay, $\mu\in[0,1)$ momentum, $d\in[0,1]$ dampening
and $\eta>0$ learning rate. The adjusted gradient is $a=g+\lambda w$.
Momentum history $v_s$ is $a$ on the first step and
$v_s=\mu v_{s-1}+(1-d)a$ on later steps.
Without momentum the selected direction is $u=a$; ordinary momentum selects $u=v_s$;
Nesterov selects $u=a+\mu v_s$ and requires $\mu>0$, $d=0$.
The candidate parameter is $w_{\mathrm{new}}=w-\eta u$.

These are elementwise real-number descriptions, not permission to reassociate floating
operations. Coefficients must be finite and satisfy their API bounds. FLOAT32, BFLOAT16
and FLOAT16 parameter sessions also validate them after narrowing to private FLOAT32
working arithmetic; FLOAT64 uses configured binary64 values.

**Momentum** retains a running direction from earlier gradients. **Dampening** reduces how
much a new gradient enters that history after its initial update. **Weight decay** discourages
large parameters; coupled decay enters the gradient calculation shown above.
**Nesterov** combines the new gradient with a momentum contribution using the stated formula.
These are choices beyond plain SGD, not additional learned [model](../glossary.md#term-model) parameters.

To give [`weight`](../glossary.md#term-weight) and [`bias`](../glossary.md#term-bias) different rates in the [`Linear`](../glossary.md#term-linear-meanings) example above, replace its session
opening and [run](../glossary.md#term-run) block with this variation after `copyToSharedNative` succeeds. A [parameter group](../glossary.md#parameter-group--parametergroup)
is an [immutable](../glossary.md#term-immutable) ordered list of distinct paths from `layer.trainingParameters().entries()`
plus one optimizer configuration. In this example the group's `weight` uses SGD rate `0.05`;
unlisted `bias` uses the default SGD rate `0.1`:

```java
// A group changes selected paths' optimizer settings, not their membership in the graph.
List<ParameterGroup> groups = List.of(
        new ParameterGroup(List.of("weight"), new Sgd(0.05)));
try (TrainingSession session = TrainingSession.open(
        engine, layer.trainingParameters(), objective, new Sgd(0.1), groups)) {
    TrainingState before = session.state();
    // Capture detached state before the update so restore can reinstate parameters and counters.
    session.run(List.of(input, target), GradientMode.RESET_AND_STEP);
    assert session.optimizerSteps() == 1;
    assert session.parameterGroups().equals(groups);
    session.restore(before);
    assert session.optimizerSteps() == 0;
}
```

The run uses one prepared [graph](../glossary.md#graph) and one global update step. Restore returns both parameters and
the counter to their earlier state. Group list order and path order are part of restore
compatibility even when reordered paths would have the same numerical updates. Unknown or
overlapping paths fail during open before [compilation](../glossary.md#compiler-engine-and-execution-stage-names); a group does not freeze a parameter or
remove it from differentiation.

For a mixed algorithm choice, use this alternative after the same native copy. Here `weight`
uses decoupled-decay [AdamW](../glossary.md#adaptive-optimizer--adam-adamw), while unlisted `bias` uses coupled-decay [Adam](../glossary.md#adaptive-optimizer--adam-adamw). Both update on the
same successful global step:

```java
// Both algorithms commit on the same global optimizer step; this is not two sessions.
List<ParameterGroup> groups = List.of(
        new ParameterGroup(List.of("weight"),
                new AdamW(0.001, 0.9, 0.999, 1e-8, 0.01)));
try (TrainingSession session = TrainingSession.open(engine, layer.trainingParameters(), objective,
        new Adam(0.001, 0.9, 0.999, 1e-8, 0.0), groups)) {
    session.run(List.of(input, target), GradientMode.RESET_AND_STEP);
    assert session.optimizerSteps() == 1;
}
```

Both constructors accept `(learningRate, beta1, beta2, epsilon, weightDecay)`. Rates and [epsilon](../glossary.md#epsilon)
must be [finite](../glossary.md#term-finite) and positive, betas finite in `[0, 1)`, and decay finite and non-negative. Adam
adds decay to the gradient before updating its two moments; AdamW instead scales the master
parameter by $1-\eta\lambda$ before the adaptive update, where $\eta$ is learning rate and
$\lambda$ is weight decay. Each moment is bias-corrected using the one-based global optimizer step.
For FLOAT32, BFLOAT16 and FLOAT16 parameters, coefficients narrow once to FLOAT32 at open;
invalid narrowed values are rejected before compilation.

**Adam**, adaptive moment estimation, tracks moving averages of gradients and squared gradients
to scale updates. Its **first moment** summarizes gradient direction; its **second moment**
summarizes squared magnitude. `beta1` and `beta2` control how strongly earlier values persist.
**[Bias correction](../glossary.md#term-bias-correction)** compensates for those averages starting at zero; it is unrelated to a
layer's additive [bias parameter](../glossary.md#bias-and-activation). **Epsilon** is the small positive denominator [constant](../glossary.md#term-constant).
**AdamW** applies weight decay separately from those gradient summaries, as described above.

Let $\widehat m$ and $\widehat v$ be the bias-corrected first and second moments for one
parameter element, $\eta>0$ its learning rate, and $\epsilon>0$ the configured denominator
constant. The adaptive amount subtracted is

$$
\Delta=\frac{\eta\widehat m}{\sqrt{\widehat v}+\epsilon}.
$$

This real-number formula is evaluated **multiply before divide**, in the parameter's working
precision. Its finite domain requires nonnegative $\widehat v$ and admitted finite coefficients
and state. The scaled numerator and final update must each be finite; a nonfinite intermediate
rejects the whole run without committing parameters, moments, counters or pending gradients.
Changing operation order can change overflow behavior even when real-number expressions agree.

For a dimensionless FLOAT32 scalar initially $w=1$, choose
$\eta=\epsilon=10^{-30}$, first-moment retention $\beta_1=0.5$,
second-moment retention $\beta_2=0$, no decay, and gradients $g_1=10^{19}$ then $g_2=0$.
Here subscript $s\in\{1,2\}$ identifies an optimizer step, not a sample.
The first update is too small to change the represented weight $1$.

On step two, the first moment is $m_2=0.5(0.5\cdot10^{19})=2.5\cdot10^{18}$.
Bias correction gives $\widehat m_2=m_2/(1-\beta_1^2)\approx3.333333\cdot10^{18}$;
the corrected second moment is $\widehat v_2=0$ because $\beta_2=0$ and $g_2=0$.
Thus $\Delta_2=(\eta\widehat m_2)/\epsilon\approx3.333333\cdot10^{18}$ is finite
in FLOAT32 and step two commits for either Adam or AdamW. The separately evaluated quotient
$\widehat m_2/\epsilon\approx3.333333\cdot10^{48}$ would overflow FLOAT32 before
later scaling. These approximations explain why the documented rounded operation order matters.

## Accumulate and clear [gradients](../glossary.md#term-gradient)

- `RESET_AND_STEP` ignores older pending gradients, applies this [run](../glossary.md#term-run), and clears accumulation.
- `ACCUMULATE` adds this run to the pending sum without updating [parameters](../glossary.md#term-parameter) or [momentum](../glossary.md#term-momentum).
- `ACCUMULATE_AND_STEP` applies the pending sum plus this run and clears it.
- `zeroGrad()` clears only the pending sum.

```java
// The first two calls execute backward but deliberately leave parameter bytes unchanged.
training.run(inputs, GradientMode.ACCUMULATE);
training.run(inputs, GradientMode.ACCUMULATE);
TrainingStep update = training.run(inputs, GradientMode.ACCUMULATE_AND_STEP);
assert update.accumulatedGradientRuns() == 0;
```

Execution numbers count every successful forward/backward run. [Optimizer](../glossary.md#term-optimizer)-step numbers count only
successful updates and are absent from an accumulate-only `TrainingStep`.

Accumulation is a **sum**, not an implicit average. The [objective](../glossary.md#term-objective) reduction still matters:
summing gradients from several [batches](../glossary.md#term-batch-machine-learning-and-hyperparameters) does not silently normalize by their total [sample](../glossary.md#term-sample-machine-learning-and-hyperparameters) count.
The [existing reverse-gradient visual](../explanation/tensors-and-gradients.md) shows how sample
contributions reach a shared parameter; accumulation adds contributions from separate runs
before the chosen update.

## [Snapshot](../glossary.md#term-snapshot) and restore

`TrainingState snapshot = training.state()` owns [canonical](../glossary.md#term-canonical-form) logical-[parameter](../glossary.md#term-parameter), master-parameter,
first-slot, optional second-moment, and pending-[gradient](../glossary.md#term-gradient) bytes keyed by stable recursive parameter
paths. Logical payloads
use the declared parameter width: two bytes per [BFLOAT16](../glossary.md#term-bfloat16)/[FLOAT16](../glossary.md#term-float16) [element](../glossary.md#term-element), four for [FLOAT32](../glossary.md#term-float32), and
eight for [FLOAT64](../glossary.md#term-float64). Master, [optimizer](../glossary.md#term-optimizer)-slot, and accumulation payloads use four bytes per element for
BFLOAT16, FLOAT16, and FLOAT32 parameters and eight for FLOAT64. `training.restore(snapshot)`
validates the complete optimizer configuration, including [dampening](../glossary.md#term-dampening), counters, paths, data types,
[Shapes](../glossary.md#shape), [finite](../glossary.md#term-finite) payload values, and exact payload lengths before installing any state. For grouped
sessions it also checks the exact default optimizer and ordered groups, including each path and
algorithm configuration. `momentumBytes()` remains the first-slot accessor: [SGD](../glossary.md#term-stochastic-gradient-descent-sgd) [momentum](../glossary.md#term-momentum) or
[Adam](../glossary.md#adaptive-optimizer--adam-adamw)/[AdamW](../glossary.md#adaptive-optimizer--adam-adamw) first moment. [`firstMomentBytes()`](../glossary.md#first-optimizer-slot--firstmomentbytes) [aliases](../glossary.md#aliasing-and-arena) it; `secondMomentBytes()` is present only
for Adam/AdamW. The returned [views](../glossary.md#term-view-meanings) are fresh and read-only. `TrainingState.optimizer()` now
returns `Optimizer`, not `Sgd` (a source and [binary](../glossary.md#term-binary-descriptor-and-metadata) [signature](../glossary.md#term-signature) change); check/cast when using
SGD-specific members. No `defaultOptimizer()` accessor was added. The snapshot remains readable
after session and [Engine](../glossary.md#term-engine) close. It is an in-memory
recovery [handoff](../glossary.md#term-handoff), not a file
or [checkpoint](../glossary.md#term-checkpoint) format.

A **snapshot** records state at a particular point; **restore** reinstalls a compatible snapshot.
A **[master parameter](../glossary.md#term-master-parameter)** is the private working-[precision](../glossary.md#term-precision) value used by the optimizer when logical
storage has less precision. Optimizer **slots** hold history such as momentum or moments.
**Canonical** bytes follow the defined standard representation; **[detached](../glossary.md#detached-model-results)** means they are
independent of the live session. Restoring this state still does not restore the caller's [dataset
cursor](../glossary.md#term-dataset-cursor), [random stream](../glossary.md#graph-rng-state--graphrngstate) or [epoch](../glossary.md#term-epoch-meanings) position.

## Lifetimes and [concurrency](../glossary.md#term-concurrency)

The caller owns the [Engine](../glossary.md#term-engine), [live parameter source](../glossary.md#live-parameter-source--trainingparameters) (including its [Module](../glossary.md#term-module) when applicable), and
[parameter](../glossary.md#term-parameter) storage; these must outlive the session and any admitted [operation](../glossary.md#operation). Fresh [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters)/[target](../glossary.md#term-target)
storage is [borrowed](../glossary.md#ownership-borrowing-and-lease) only through its [synchronous](../glossary.md#term-synchronous) `runValues` call and internal result closure,
not until session close. Exact-ID [`run`](../glossary.md#term-run) likewise borrows its supplied external storage only for
the call; keep every association used by that call usable and free from conflicting mutation.
The session owns one Engine [`PreparedExecution`](../glossary.md#prepared-execution--preparedexecution), its private
[cotangent seed](../glossary.md#primal-adjoint-and-differentiation-seed) memory, [optimizer](../glossary.md#term-optimizer) slots, accumulation, and counters. It admits one operation at a
time. [Admission](../glossary.md#term-admission) happens before [argument](../glossary.md#term-argument) [validation](../glossary.md#term-validation): Engine closure wins first, then session
closing/closure, then a busy operation. Close rejects later admission, waits for already-admitted
Training work to finish, attempts all owned cleanup once, and never closes caller-owned objects.
It closes the prepared [owner](../glossary.md#term-owner) before its private [seed](../glossary.md#term-seed) [arena](../glossary.md#term-arena), and each [run result](../glossary.md#run-result--runresult) closes before
that owner. Repeated close calls replay the retained cleanup result.

A failed validation, Engine run, [publication](../glossary.md#term-publication) check, [materialization](../glossary.md#materialization), decode, non-[finite](../glossary.md#term-finite) arithmetic,
represented [overflow](../glossary.md#term-overflow), or candidate calculation changes no parameter bytes, private masters,
optimizer slots, counters, or pending [gradients](../glossary.md#term-gradient). [BFLOAT16](../glossary.md#term-bfloat16), [FLOAT16](../glossary.md#term-float16), and [FLOAT32](../glossary.md#term-float32) parameters use
finite FLOAT32 masters, gradients, accumulation, [momentum](../glossary.md#term-momentum), intermediates, and candidates; [FLOAT64](../glossary.md#term-float64)
uses FLOAT64. A successful low-[precision](../glossary.md#term-precision) update narrows the validated master candidate once to the
declared logical type before [commit](../glossary.md#term-commit). [NaN](../glossary.md#term-nan) and [infinity](../glossary.md#term-infinity) are rejected, while negative and [positive
zero](../glossary.md#nan-infinity-and-signed-zero) are both finite. Callers must not replace or concurrently mutate captured parameter wrappers,
[Tensor](../glossary.md#tensor) storage associations, or parameter bytes while the session is open.

**Transactional** means the update is committed as a whole or not committed: a failed candidate
does not leave only some parameters updated. A **candidate** is the proposed next parameter value
before commit. **Finite** excludes infinity and NaN (“[not a number](../glossary.md#nan-infinity-and-signed-zero)”); **overflow** occurs when a
calculated magnitude cannot fit the required representation. **[Narrowing](../glossary.md#term-narrowing)** rounds a working value
to the logical storage type. These checks concern numerical validity, not whether the [model](../glossary.md#term-model)'s
[predictions](../glossary.md#term-prediction) are useful on independent data.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| Open rejects [parameter](../glossary.md#term-parameter) storage | It is [heap](../glossary.md#term-heap), read-only, thread-confined, wrong-sized, a [view](../glossary.md#term-view-meanings), or not native/aligned. | Use exact-capacity writable shareable native [`FLOAT64`](../glossary.md#term-float64), [`FLOAT32`](../glossary.md#term-float32), [`BFLOAT16`](../glossary.md#term-bfloat16), or [`FLOAT16`](../glossary.md#term-float16) storage with a dense offset-zero non-view [layout](../glossary.md#layout). |
| Open rejects an empty/non-[finite](../glossary.md#term-finite) parameter | Its [element](../glossary.md#term-element) count is zero or an initial value is [NaN](../glossary.md#term-nan)/infinite. | Supply a non-empty parameter with finite values before opening; rejection precedes [compile](../glossary.md#compile). |
| Open rejects a [snapshot](../glossary.md#term-snapshot) byte size | The logical or [optimizer](../glossary.md#term-optimizer)-width payload exceeds `Integer.MAX_VALUE` bytes for one parameter. | Use a smaller parameter; this in-memory session cannot snapshot a larger payload. |
| Open rejects a [parameter group](../glossary.md#parameter-group--parametergroup) | A path is blank, unknown, or repeated, or assigned optimizer coefficients cannot narrow for that parameter type. | Use the exact keys from `module.parametersRecursively()`, keep memberships disjoint, and choose [dtype](../glossary.md#data-type--datatype)-compatible values. |
| [Run](../glossary.md#term-run) rejects non-finite optimizer arithmetic | A [gradient](../glossary.md#term-gradient), accumulation, [momentum](../glossary.md#term-momentum) term, or candidate is NaN/infinite or overflowed. | Correct the [objective](../glossary.md#term-objective)/input scale; the failed run committed no parameter, [optimizer state](../glossary.md#term-optimizer-state), or counter. |
| Open rejects a [target](../glossary.md#term-target) as disconnected | A discovered [Module](../glossary.md#term-module) parameter does not contribute to the [scalar objective](../glossary.md#loss). | Build the objective from the Module's complete intended [trainable parameter](../glossary.md#trainable-parameter) set or use a Module containing only this session's parameters. |
| `run` reports a missing or foreign input | The list does not contain every [identity](../glossary.md#term-identity-meanings) from `training.inputs()` exactly once. | Bind those exact [Tensors](../glossary.md#tensor) in any order, or use `runValues` for fresh stored values. |
| `runValues` reports a count or [descriptor](../glossary.md#term-descriptor) mismatch | Values do not match all `inputs()` positions exactly. | Include targets and other external leaves in compiled order with exact full static descriptors; omit parameters and [seed](../glossary.md#term-seed). |
| `runValues` reports overlapping storage | A nonempty value is repeated, external slices overlap, or an external [aliases](../glossary.md#aliasing-and-arena) a parameter or seed. | Use distinct non-overlapping external stores; generic [Engine](../glossary.md#term-engine)'s repeated-value permission does not apply to Training. |
| Run reports a changed [binding](../glossary.md#term-binding)/storage | A `Parameter` or Tensor storage association changed after open. | Close the session, finish the Module change, rebuild the expression, and open a new session. |
| A [Metal](../glossary.md#term-metal)/mixed run is unsupported | Current [backend capability](../glossary.md#backend-capability), transfer, preparation, or [materialization](../glossary.md#materialization) does not cover the [graph](../glossary.md#graph). | Use a supported Engine composition; Training never falls back or mutates before that failure. |
| Another [operation](../glossary.md#operation) reports the session is busy | A run, state operation, restore, zero, or close [race](../glossary.md#term-race) is active. | Serialize operations on that session or use independent sessions with independent Module/storage state. |

## Related documentation

- [Training API](../reference/training/index.md)<span data-glossary-navigation="true"> ([API glossary](../glossary.md#term-application-programming-interface-api))</span>
- [Autograd](autograd.md)<span data-glossary-navigation="true"> ([Autograd glossary](../glossary.md#autograd))</span>
- [Training graph](../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../glossary.md#training-graph))</span>
- [Lifecycle](../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../glossary.md#lifecycle))</span>
- [Training master plan](../planning/extensions/training/master-plan.md)

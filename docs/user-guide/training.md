# Train a model

## Outcome

This guide runs the implemented public backend-neutral training lifecycle: one scalar objective,
one reusable compiled/prepared session, repeated SGD, Adam, or AdamW updates, optional gradient
accumulation, and detached state. The worked example uses CPU through `Engine.standard()`; the
session itself uses
the ordinary Engine path rather than a training-specific executor. Supported Metal or mixed
preparation follows the same lifecycle, but the low-parameter CPU qualification described here is
not a claim of BFLOAT16/FLOAT16 Metal training support.

## Copy an initialized Linear

An eager `Linear` initializes its weight and bias in Java heap storage. To train it without
replacing those Tensor objects, construct the loss first, then explicitly copy its parameters
into a caller-owned shared native arena before opening a session. `Arena.ofShared()` keeps those
parameter segments accessible to the session's operation threads. The copy may allocate the
entire parameter payload even if a later step fails; plan the Arena lifetime before calling it.

This complete example starts with a `FLOAT32` one-input, one-output `Linear` and a target of zero.
The input and target have native storage too, because the CPU Engine needs live host inputs.

```java
static Tensor nativeMatrix(Arena arena, float value) {
    Shape shape = Shape.of(1, 1);
    TensorDescriptor descriptor = new TensorDescriptor(
            DataType.FLOAT32, shape,
            Optional.of(LayoutDescriptor.contiguous(shape)), false);
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
    Tensor error = output.sub(target);
    Tensor objective = error.mul(error).sum();

    // This call is explicit; open does not copy heap-backed parameters.
    TrainingParameterStorage.copyToSharedNative(layer, arena);
    assert layer.weight().value() == originalWeight;
    try (TrainingSession session = TrainingSession.open(
            engine, layer, objective, new Sgd(0.1))) {
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
that the original weight Tensor remains the graph input after copying; it does not claim that
the old heap storage was closed or transferred. The caller retains ownership of both source and
destination storage. Keep the shared Arena alive through session close, and never copy while a
session is open or another caller uses the Module/storage. Without the copy, `TrainingSession.open`
rejects the heap-backed parameters before compilation. A confined or closed destination Arena,
uninitialized deferred layer, non-finite source value, aliased parameter regions, unsupported
layout, or invalid source storage also fails before any parameter association changes. Already
consumed destination Arena capacity is not individually recovered after allocation failure.

For details of raw-byte preservation and failure behavior, see the [Training API](../api/training-api.md#explicit-native-parameter-copy)
and [architecture explanation](../architecture/training-graph.md#optimizer-as-a-backend-agnostic-step).

## Build a runnable scalar problem

The first optimizer domain requires exact-capacity writable **shareable native** parameter storage.
Parameters may use `FLOAT64`, `FLOAT32`, `BFLOAT16`, or `FLOAT16`; the scalar objective remains
`FLOAT32` or `FLOAT64`. The example uses `FLOAT32` for both parameter and input and therefore gives
them an `Arena.ofShared()` lifetime. JVM-heap storage created by convenience Tensor factories is not
an accepted parameter carrier in this lifecycle.

Each parameter must also fit the in-memory state format: its logical payload and its
optimizer-width payload must each be no larger than `Integer.MAX_VALUE` bytes. The optimizer
width is eight bytes per FLOAT64 element and four bytes per FLOAT32, BFLOAT16, or FLOAT16 element.
An oversized parameter is rejected when the session opens, before its own update buffers, the
private seed, or the compiled graph are created. Earlier parameters may already have been captured.

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

Create the forward/loss expression before opening the session. Here SGD learns the scalar target
`2.5` from initial weight `-4.0`:

```java
TrainingStep last;
try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
    ScalarModel model = new ScalarModel(nativeScalar(arena, -4.0f, true));
    Tensor target = nativeScalar(arena, 2.5f, false);
    Tensor error = model.weight().value().sub(target);
    Tensor loss = error.mul(error);

    try (TrainingSession training = TrainingSession.open(
            engine, model, loss, new Sgd(0.1))) {
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

`TrainingSession.open` performs the only gradient compile and the only preparation. Every `run`
reuses that `InferenceSession`. Inputs may be supplied in any list order, but every identity listed
by `training.inputs()` must occur exactly once with a compatible live storage association.

## Configure an optimizer

`new Sgd(learningRate)` selects plain SGD. The constructor with every coefficient is:

```java
new Sgd(learningRate, momentum, dampening, weightDecay, nesterov)
```

The four-argument convenience omits dampening and uses zero. Weight decay is coupled:
`adjusted = gradient + weightDecay * parameter`. Momentum stores the adjusted gradient on the
first update without applying dampening. Later updates store
`momentum * previous + (1 - dampening) * adjusted`. Ordinary momentum uses the new slot; Nesterov
uses `adjusted + momentum * newSlot`. Dampening is in `[0, 1]`; Nesterov requires positive momentum
and exactly zero dampening. The selected update is subtracted as `learningRate * selected`.
FLOAT32, BFLOAT16, and FLOAT16 parameter sessions validate coefficients after narrowing to their
private FLOAT32 arithmetic domain as well as before it; FLOAT64 uses the configured binary64 values.

To give `weight` and `bias` different rates in the `Linear` example above, replace its session
opening and run block with this variation after `copyToSharedNative` succeeds. A parameter group
is an immutable ordered list of distinct recursive paths from `layer.parametersRecursively()`
plus one optimizer configuration. In this example the group's `weight` uses SGD rate `0.05`;
unlisted `bias` uses the default SGD rate `0.1`:

```java
List<ParameterGroup> groups = List.of(
        new ParameterGroup(List.of("weight"), new Sgd(0.05)));
try (TrainingSession session = TrainingSession.open(
        engine, layer, objective, new Sgd(0.1), groups)) {
    TrainingState before = session.state();
    session.run(List.of(input, target), GradientMode.RESET_AND_STEP);
    assert session.optimizerSteps() == 1;
    assert session.parameterGroups().equals(groups);
    session.restore(before);
    assert session.optimizerSteps() == 0;
}
```

The run uses one prepared graph and one global update step. Restore returns both parameters and
the counter to their earlier state. Group list order and path order are part of restore
compatibility even when reordered paths would have the same numerical updates. Unknown or
overlapping paths fail during open before compilation; a group does not freeze a parameter or
remove it from differentiation.

For a mixed algorithm choice, use this alternative after the same native copy. Here `weight`
uses decoupled-decay AdamW, while unlisted `bias` uses coupled-decay Adam. Both update on the
same successful global step:

```java
List<ParameterGroup> groups = List.of(
        new ParameterGroup(List.of("weight"),
                new AdamW(0.001, 0.9, 0.999, 1e-8, 0.01)));
try (TrainingSession session = TrainingSession.open(engine, layer, objective,
        new Adam(0.001, 0.9, 0.999, 1e-8, 0.0), groups)) {
    session.run(List.of(input, target), GradientMode.RESET_AND_STEP);
    assert session.optimizerSteps() == 1;
}
```

Both constructors accept `(learningRate, beta1, beta2, epsilon, weightDecay)`. Rates and epsilon
must be finite and positive, betas finite in `[0, 1)`, and decay finite and non-negative. Adam
adds decay to the gradient before updating its two moments; AdamW instead scales the master
parameter by `1 - learningRate * weightDecay` before the adaptive update. Each moment is
bias-corrected using the one-based global optimizer step. For FLOAT32, BFLOAT16, and FLOAT16
parameters, coefficients narrow once to FLOAT32 at open, and invalid narrowed values are
rejected before compilation.

## Accumulate and clear gradients

- `RESET_AND_STEP` ignores older pending gradients, applies this run, and clears accumulation.
- `ACCUMULATE` adds this run to the pending sum without updating parameters or momentum.
- `ACCUMULATE_AND_STEP` applies the pending sum plus this run and clears it.
- `zeroGrad()` clears only the pending sum.

```java
training.run(inputs, GradientMode.ACCUMULATE);
training.run(inputs, GradientMode.ACCUMULATE);
TrainingStep update = training.run(inputs, GradientMode.ACCUMULATE_AND_STEP);
assert update.accumulatedGradientRuns() == 0;
```

Execution numbers count every successful forward/backward run. Optimizer-step numbers count only
successful updates and are absent from an accumulate-only `TrainingStep`.

## Snapshot and restore

`TrainingState snapshot = training.state()` owns canonical logical-parameter, master-parameter,
first-slot, optional second-moment, and pending-gradient bytes keyed by stable recursive parameter
paths. Logical payloads
use the declared parameter width: two bytes per BFLOAT16/FLOAT16 element, four for FLOAT32, and
eight for FLOAT64. Master, optimizer-slot, and accumulation payloads use four bytes per element for
BFLOAT16, FLOAT16, and FLOAT32 parameters and eight for FLOAT64. `training.restore(snapshot)`
validates the complete optimizer configuration, including dampening, counters, paths, data types,
Shapes, finite payload values, and exact payload lengths before installing any state. For grouped
sessions it also checks the exact default optimizer and ordered groups, including each path and
algorithm configuration. `momentumBytes()` remains the first-slot accessor: SGD momentum or
Adam/AdamW first moment. `firstMomentBytes()` aliases it; `secondMomentBytes()` is present only
for Adam/AdamW. The returned views are fresh and read-only. `TrainingState.optimizer()` now
returns `Optimizer`, not `Sgd` (a source and binary signature change); check/cast when using
SGD-specific members. No `defaultOptimizer()` accessor was added. The snapshot remains readable
after session and Engine close. It is an in-memory
recovery handoff, not a file
or checkpoint format.

## Lifetimes and concurrency

The caller owns the Engine, Module, parameter storage, and external inputs; all must outlive the
session and any admitted operation. The session owns one Engine `InferenceSession`, its private
cotangent seed memory, optimizer slots, accumulation, and counters. It admits one operation at a
time. Admission happens before argument validation: Engine closure wins first, then session
closing/closure, then a busy operation. Close rejects later admission, waits for already-admitted
Training work to finish, attempts all owned cleanup once, and never closes caller-owned objects.
Repeated close calls replay the retained cleanup result.

A failed validation, Engine run, publication check, materialization, decode, non-finite arithmetic,
represented overflow, or candidate calculation changes no parameter bytes, private masters,
optimizer slots, counters, or pending gradients. BFLOAT16, FLOAT16, and FLOAT32 parameters use
finite FLOAT32 masters, gradients, accumulation, momentum, intermediates, and candidates; FLOAT64
uses FLOAT64. A successful low-precision update narrows the validated master candidate once to the
declared logical type before commit. NaN and infinity are rejected, while negative and positive
zero are both finite. Callers must not replace or concurrently mutate captured parameter wrappers,
Tensor storage associations, or parameter bytes while the session is open.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| Open rejects parameter storage | It is heap, read-only, thread-confined, wrong-sized, a view, or not native/aligned. | Use exact-capacity writable shareable native `FLOAT64`, `FLOAT32`, `BFLOAT16`, or `FLOAT16` storage with a dense offset-zero non-view layout. |
| Open rejects an empty/non-finite parameter | Its element count is zero or an initial value is NaN/infinite. | Supply a non-empty parameter with finite values before opening; rejection precedes compile. |
| Open rejects a snapshot byte size | The logical or optimizer-width payload exceeds `Integer.MAX_VALUE` bytes for one parameter. | Use a smaller parameter; this in-memory session cannot snapshot a larger payload. |
| Open rejects a parameter group | A path is blank, unknown, or repeated, or assigned optimizer coefficients cannot narrow for that parameter type. | Use the exact keys from `module.parametersRecursively()`, keep memberships disjoint, and choose dtype-compatible values. |
| Run rejects non-finite optimizer arithmetic | A gradient, accumulation, momentum term, or candidate is NaN/infinite or overflowed. | Correct the objective/input scale; the failed run committed no parameter, optimizer state, or counter. |
| Open rejects a target as disconnected | A discovered Module parameter does not contribute to the scalar objective. | Build the objective from the Module's complete intended trainable parameter set or use a Module containing only this session's parameters. |
| Run reports a missing or foreign input | The list does not contain every identity from `training.inputs()` exactly once. | Bind those exact Tensors; order does not matter. |
| Run reports a changed binding/storage | A `Parameter` or Tensor storage association changed after open. | Close the session, finish the Module change, rebuild the expression, and open a new session. |
| A Metal/mixed run is unsupported | Current backend capability, transfer, preparation, or materialization does not cover the graph. | Use a supported Engine composition; Training never falls back or mutates before that failure. |
| Another operation reports the session is busy | A run, state operation, restore, zero, or close race is active. | Serialize operations on that session or use independent sessions with independent Module/storage state. |

## Related documentation

- [Training API](../api/training-api.md)
- [Autograd](autograd.md)
- [Training graph](../architecture/training-graph.md)
- [Lifecycle](../architecture/lifecycle.md)
- [Training master plan](../planning/extensions/training/master-plan.md)

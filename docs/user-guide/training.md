# Train a model

## Outcome

This guide runs the implemented public CPU training lifecycle: one scalar objective, one reusable
compiled/prepared session, repeated SGD updates, optional gradient accumulation, and detached
state. The session uses the ordinary Engine path rather than a training-specific executor.

## Build a runnable scalar problem

The first optimizer domain requires exact-capacity writable **shareable native** parameter storage.
The example therefore gives the parameter and input an `Arena.ofShared()` lifetime. JVM-heap
storage created by convenience Tensor factories is not an accepted parameter carrier in this
initial lifecycle.

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

## Configure SGD

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
FLOAT32 sessions validate the coefficients after narrowing as well as before it.

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

`TrainingState snapshot = training.state()` owns canonical parameter, momentum, and pending-
gradient bytes keyed by stable recursive parameter paths. `training.restore(snapshot)` validates
the complete optimizer configuration, including dampening, counters, paths, data types, Shapes,
finite payload values, and payload lengths before installing any state. The snapshot remains
readable after session and Engine close. It is an in-memory recovery handoff, not a file or
checkpoint format.

## Lifetimes and concurrency

The caller owns the Engine, Module, parameter storage, and external inputs; all must outlive the
session and any admitted operation. The session owns one Engine `InferenceSession`, its private
cotangent seed memory, optimizer slots, accumulation, and counters. It admits one operation at a
time. Admission happens before argument validation: Engine closure wins first, then session
closing/closure, then a busy operation. Close rejects later admission, waits for already-admitted
Training work to finish, attempts all owned cleanup once, and never closes caller-owned objects.
Repeated close calls replay the retained cleanup result.

A failed validation, Engine run, publication check, materialization, decode, non-finite arithmetic,
represented overflow, or candidate calculation changes no parameter bytes, optimizer slots,
counters, or pending gradients. Current parameters, gradients, accumulation, momentum, arithmetic
intermediates, and candidates must stay finite in their parameter precision; NaN and infinity are
rejected, while negative and positive zero are both finite. Callers must not replace or
concurrently mutate captured parameter wrappers, Tensor storage associations, or parameter bytes
while the session is open.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| Open rejects parameter storage | It is heap, read-only, thread-confined, wrong-sized, a view, or not native/aligned. | Use exact-capacity writable shareable native `FLOAT32`/`FLOAT64` storage with a dense offset-zero non-view layout. |
| Open rejects an empty/non-finite parameter | Its element count is zero or an initial value is NaN/infinite. | Supply a non-empty parameter with finite values before opening; rejection precedes compile. |
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

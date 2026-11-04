# Optimizers, gradient modes and atomic state

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/training/src/main/java/io/github/pho001/synaptik/training/TrainingSession.java)

An **optimizer** turns gradients into parameter updates. Its **state** can include momentum
(a running update direction), Adam's first and second moments (running gradient and squared-gradient
estimates), and pending accumulated gradients. An optimizer **slot** is one such per-parameter
history value, not a Runtime memory slot. A **transaction** stages a complete update before commit;
its documented atomicity does not create a transaction across model files and external progress.

<a id="optimizers-and-gradient-modes"></a>
## Optimizers and gradient modes

For parameter `p`, published gradient `g`, coupled weight decay `wd`, momentum `m`, dampening `damp`,
and learning rate `lr`, SGD first computes `d = g + wd * p`. Without momentum, the update is
`p = p - lr * d`. With momentum, the first successful optimizer step stores `d` without applying
dampening; later steps store `m * previous + (1 - damp) * d`. Ordinary momentum selects that new
slot, while Nesterov selects `d + m * newSlot`. Dampening is in `[0, 1]`; Nesterov requires
positive momentum and exactly zero dampening. FLOAT32 and all four low parameter types validate
coefficients after one binary32 narrowing and use finite binary32 master, momentum, accumulation,
and candidate arithmetic. A successful low-precision update narrows the validated master candidate
once to the declared logical type. FLOAT64 parameters use binary64.

`Adam(lr, beta1, beta2, epsilon, weightDecay)` and
`AdamW(lr, beta1, beta2, epsilon, weightDecay)` require finite positive `lr` and `epsilon`,
betas in `[0, 1)`, and finite non-negative decay. On one-based global step `t`, both form
`m = beta1 * oldM + (1 - beta1) * g` and
`v = beta2 * oldV + (1 - beta2) * g * g`. They subtract
`lr * (m / (1 - beta1^t)) / (sqrt(v / (1 - beta2^t)) + epsilon)`.
Adam adds `weightDecay * masterParameter` to `g` first (coupled decay). AdamW keeps `g`
unchanged and multiplies the master parameter by `1 - lr * weightDecay` before subtraction
(decoupled decay). All parameters share the same `t`, including mixed SGD/Adam/AdamW groups.
Binary32 assignments narrow all five coefficients once at open, rejecting zero rate/epsilon,
betas rounded to one, and non-finite or out-of-range results. Bias corrections are calculated
outside each element loop. Every intermediate, both moments, and final logical narrowing must
remain finite. `ACCUMULATE_AND_STEP` supplies the pending sum, not an average.

For a `Linear` whose `weight` and `bias` are already in native storage, this replaces the opening
block in the example above. AdamW updates `weight`; the unlisted `bias` uses Adam. Both advance
one global step together:

```java
List<ParameterGroup> groups = List.of(
        new ParameterGroup(List.of("weight"),
                new AdamW(0.001, 0.9, 0.999, 1e-8, 0.01)));
try (TrainingSession session = TrainingSession.open(engine, layer.trainingParameters(), objective,
        new Adam(0.001, 0.9, 0.999, 1e-8, 0.0), groups)) {
    session.run(List.of(input, target), GradientMode.RESET_AND_STEP);
    assert session.optimizerSteps() == 1;
}
```

`RESET_AND_STEP` updates from only the current gradients and clears older pending accumulation.
`ACCUMULATE` adds the current gradients to the pending sum without changing parameters, momentum,
or optimizer-step count. `ACCUMULATE_AND_STEP` updates from the pending sum plus the current
gradients and then clears the sum. `zeroGrad()` clears only pending gradients. Execution numbers
count every successful forward/backward run from one; optimizer-step numbers count successful
updates from one.

<a id="atomicity-state-concurrency-and-close"></a>
## Atomicity, state, concurrency, and close

Both `run` and `runValues` use one private transaction. Engine execution, result closure,
publication validation, detached materialization, gradient decoding, candidate
arithmetic, and a second complete binding/storage validation all precede mutation. Current
parameters, decoded gradients, accumulation, momentum, every optimizer intermediate, and
parameter candidates must be finite; NaN, infinity, and represented overflow fail before commit,
while both signed zeros are finite. A supported ordinary failure changes no parameter bytes,
momentum, accumulation, or counters. A non-finite detached objective alone is permitted when its
required gradients and all optimizer arithmetic are finite. Each captured parameter keeps active
momentum and pending-gradient arrays, parameter candidate and rollback arrays, and (for low
precision) master and master-rollback arrays. An inactive pending-gradient candidate is allocated
only for accumulation or restore and is released after a successful update, `zeroGrad()`, or
restore commits. An inactive momentum candidate is allocated for a nonzero configured momentum
update or restore, and is not retained for exactly zero configured momentum. Even when positive
configured momentum narrows to zero in FLOAT32 arithmetic, its update still stages and swaps a
zero momentum candidate. A restored negative-zero momentum value round-trips exactly; a later
successful zero-momentum update changes that slot to positive zero. The active pending-gradient
bytes remain intact until an explicit clear, even if a restored state records zero pending runs.
All staging allocation precedes parameter writes; commits allocate nothing, and the optimizer
loop creates no object per element. This reduces idle candidate memory, not active state or
rollback storage.

`state()` returns canonical big-endian logical-parameter, master-parameter, first-slot, optional
second-moment, and pending-gradient bytes associated with stable paths and exact schemas. The
existing `momentumBytes()` is the first-slot payload: SGD momentum or Adam/AdamW first moment.
`firstMomentBytes()` is its neutral alias; `secondMomentBytes()` returns a fresh read-only view
for Adam/AdamW and `Optional.empty()` for SGD. All views are detached and fresh. Logical payloads
use the declared
type width: one byte for either FP8 type, two bytes for BFLOAT16/FLOAT16, four for FLOAT32,
and eight for FLOAT64. Master, first-slot, optional second-moment, and pending-gradient payloads
use four bytes per element for all four low types and FLOAT32, and eight for FLOAT64.
`restore(state)` validates the complete default
optimizer and ordered groups (including dampening), counters, paths, schemas, finite payloads,
and exact slot presence/lengths, including a non-negative finite adaptive second moment, before
installing anything. FLOAT32/FLOAT64 logical and master payloads must be raw-identical, including
signed zeros. Low logical payloads and FLOAT32 masters remain independent exact payloads; restore
does not require that one be the rounded form of the other. This is an in-memory handoff, not a
durable checkpoint format.
It restores the fixed session's documented state, not an entire program: NN buffer values and
per-module modes, the effective `ForwardContext` values used to build the objective, the next
input/cursor, and any external random, scheduler, or scaler state are separate caller-coordinated
participants. A saved local module mode does not establish the context passed to a forward.
Training-mode BatchNorm has no proven step-aligned detached running-buffer capture for exact
continuation today. The session compiles its objective once and does not
advance a graph RNG stream between runs. A whole-program exact-continuation claim requires all
affecting participants on the same admitted route. A focused integration fixture now exercises
a caller-coordinated private-candidate handoff, but the per-owner APIs do not provide an atomic
transaction. See [caller-coordinated recovery](recovery.md#caller-coordinated-in-memory-recovery) and the
[Training continuation boundary](../../architecture/training-graph.md#continuing-at-the-next-training-input).
`TrainingState.optimizer()` now returns the exact default `Optimizer`, including Adam/AdamW.
Its source and binary signature changed from `Sgd`; callers requiring SGD-specific members must
check and cast. There is no replacement `defaultOptimizer()` accessor. Group configurations
retain exact kind, coefficients, and ordered paths.

Every `run`, `runValues` or state operation performs lifecycle admission before argument validation. Engine
closure wins first, then session closing/closure, then busy admission; only an admitted operation
inspects its arguments. A racing operation fails. Close first rejects later admission, waits for
an already-admitted Training operation to finish, then attempts all owned cleanup once. Repeated
and concurrent close calls replay the same retained cleanup result. The borrowed Engine, Module,
and parameter storage are never closed. Detached `TrainingStep` and `TrainingState` values remain
readable after session and Engine close. CPU and supported configured Metal/mixed execution use
the same Engine route; a capability-classified unsupported route fails before optimizer mutation.
This backend-neutral lifecycle does not itself prove or widen BFLOAT16/FLOAT16 Metal graph
capability. Training contains no backend-specific branch.

<a id="example-reconstruct-a-detached-training-value"></a>
### Example: reconstruct a detached Training value

The public grouped `TrainingState` constructor and both `ParameterState` payload constructors
allow callers outside Training's package to rebuild a detached value. Use an empty group list
for ungrouped state. Both lists are copied in their supplied order, immutable configuration and
schema entries are retained, and all supplied payload arrays are defensively copied without
conversion. Construction checks required references, nonblank paths, floating types, fully
static Shapes, exact payload widths and size bounds, nonempty parameter lists, and counter
ranges. It does not check group membership, session schema compatibility, finiteness, or
algorithm-specific slot validity. `restore` performs those checks before any installation.

**Goal and input.** Given `source = session.state()` from an existing compatible open session,
reconstruct every entry from its exposed canonical bytes. For the external test's FLOAT32 Adam
snapshot after one reset-and-step and one accumulate run, the counters are `(2, 1, 1)` and each
`Shape.of(2)` entry has eight bytes in every payload. BFLOAT16/FLOAT16 logical payloads instead
have four bytes and FP8 logical payloads have two; their optimizer payloads still have eight.
This helper also handles FLOAT64, SGD, AdamW, and ordered mixed groups. Save the following as
`DetachedTrainingCopy.java` with Training
and its public dependencies on the classpath:

```java
import io.github.pho001.synaptik.training.TrainingState;
import java.nio.ByteBuffer;
import java.util.ArrayList;

public final class DetachedTrainingCopy {
    public static TrainingState reconstruct(TrainingState source) {
        var entries = new ArrayList<TrainingState.ParameterState>();
        for (var entry : source.parameters()) {
            byte[] logical = bytes(entry.parameterBytes());
            byte[] master = bytes(entry.masterParameterBytes());
            byte[] first = bytes(entry.firstMomentBytes());
            byte[] pending = bytes(entry.accumulatedGradientBytes());
            var second = entry.secondMomentBytes();
            entries.add(second.isPresent()
                    ? new TrainingState.ParameterState(entry.path(), entry.dataType(), entry.shape(),
                            logical, master, first, bytes(second.orElseThrow()), pending)
                    : new TrainingState.ParameterState(entry.path(), entry.dataType(), entry.shape(),
                            logical, master, first, pending));
        }
        return new TrainingState(source.optimizer(), source.parameterGroups(), source.executions(),
                source.optimizerSteps(), source.accumulatedGradientRuns(), entries);
    }

    private static byte[] bytes(ByteBuffer view) {
        byte[] result = new byte[view.remaining()];
        view.get(result);
        return result;
    }
}
```

Each accessor supplies a fresh position-zero read-only view; `bytes` consumes only that view,
not the stored value. Selecting the constructor by second-moment presence preserves SGD's
absence and adaptive slots, including mixed-group assignments. The outer constructor retains
immutable optimizer/group configurations and preserves both group and parameter order.
After `TrainingState rebuilt = DetachedTrainingCopy.reconstruct(source)`, the configurations,
schema, counters, slot presence, and every payload byte match `source`. Calling
`session.restore(rebuilt)` on that compatible session and then `session.state()` gives the same
snapshot contents. Reconstruction itself requires no open session or Engine, and the value
remains readable after both close. Signed zero and nonzero pending sums with a zero pending-run
count remain unchanged; the count does not normalize those bytes.

**Failure and boundary.** A correctly sized negative adaptive second moment is structurally
constructible, but restore rejects it before changing any parameter, slot, counter, or pending
sum, even in the last entry. Reconstruction proves neither session compatibility nor complete
automatic exact continuation, and supplies no durable codec or file format. The
[external reconstruction test](../../../modules/training/src/test/java/io/github/pho001/synaptik/training/external/TrainingStateReconstructionTest.java)
checks these byte, ownership, lifetime, and late-rejection properties through public APIs.


---

[Previous topic](parameters.md) · [Next topic](recovery.md)

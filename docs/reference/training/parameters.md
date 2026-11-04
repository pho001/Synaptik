# [Parameter](../../glossary.md#term-parameter) domains, [admission](../../glossary.md#term-admission) and native copying

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/training/src/main/java/io/github/pho001/synaptik/training/TrainingSession.java)

Parameter **admission** checks whether live state satisfies the session's update requirements,
not merely whether it is [floating-point](../../glossary.md#term-floating-point). A **[master parameter](../../glossary.md#term-master-parameter)** is the private [FLOAT32](../../glossary.md#term-float32) working
value retained for a low-[precision](../../glossary.md#term-precision) parameter; the [graph](../../glossary.md#graph) still sees its declared logical type.
The explicit native copy changes the [Tensor](../../glossary.md#tensor)'s [host](../../glossary.md#term-host)-storage association without replacing its
[identity](../../glossary.md#term-identity-meanings) and is not a [device](../../glossary.md#term-device) transfer.

<a id="initial-parameter-domain"></a>
## Initial [parameter](../../glossary.md#term-parameter) domain

The update path supports non-empty [`FLOAT64`](../../glossary.md#term-float64), [`FLOAT32`](../../glossary.md#term-float32), [`BFLOAT16`](../../glossary.md#term-bfloat16), [`FLOAT16`](../../glossary.md#term-float16),
`FLOAT8_E4M3`, and `FLOAT8_E5M2` parameters whose
[Shapes](../../glossary.md#shape) are fully static and whose [layouts](../../glossary.md#layout) are dense-[contiguous](../../glossary.md#term-contiguous), offset zero, non-[view](../../glossary.md#term-view-meanings), and exact-
span. Storage must be writable, exact-capacity, [element](../../glossary.md#term-element)-aligned native [host](../../glossary.md#term-host) memory shareable across
[operation](../../glossary.md#operation) threads. Initial parameter values must be [finite](../../glossary.md#term-finite). Empty or non-finite parameters fail
during [capture](../../glossary.md#term-capture-meanings) before private [seed](../../glossary.md#term-seed) allocation, [compile](../../glossary.md#compile), or [prepare](../../glossary.md#prepare). The session updates accepted
storage in place, preserving the [Tensor](../../glossary.md#tensor) [identity](../../glossary.md#term-identity-meanings) captured by the [compiled graph](../../glossary.md#compiled-graph--compiledgraphmodel). All four low
types retain private FLOAT32 masters and publish one narrowed logical value only after a
successful update. Each logical [snapshot](../../glossary.md#term-snapshot) [payload](../../glossary.md#artifact-envelope-and-atomic-publication) and each FLOAT32/FLOAT64-width master,
[momentum](../../glossary.md#term-momentum), or accumulation payload must fit in a Java byte array (at most `Integer.MAX_VALUE`
bytes). Capture rejects an oversized payload before inspecting that parameter's storage,
allocating its update [buffers](../../glossary.md#term-buffer) or the private seed, or compiling; earlier parameters may already
have been captured. [JVM](../../glossary.md#term-java-virtual-machine-jvm)-[heap](../../glossary.md#term-heap), read-only, thread-confined, strided/view,
integral, [BOOL](../../glossary.md#data-type--datatype), and [device](../../glossary.md#term-device)-only parameter storage are outside this domain. The caller grants the
session exclusive use of the live source's membership, storage associations and parameter bytes until close.

[CPU](../../glossary.md#term-cpu) numerical execution uses the existing shared [Compiler](../../glossary.md#term-compiler) [AD](../../glossary.md#autograd) and ordinary operation [owners](../../glossary.md#term-owner) for
all four low types under their family and geometry constraints. Masters do not implicitly change
[graph](../../glossary.md#graph) [dtypes](../../glossary.md#data-type--datatype): the [objective](../../glossary.md#term-objective) must still be a visible [scalar](../../glossary.md#term-scalar) FLOAT32/FLOAT64 value, using explicit
casts where needed. Both master candidates and their narrowed logical representations must be
finite before [publication](../../glossary.md#term-publication). [E4M3](../../glossary.md#term-e4m3) [overflow](../../glossary.md#term-overflow) to [NaN](../../glossary.md#term-nan) and [E5M2](../../glossary.md#term-e5m2) overflow to [infinity](../../glossary.md#term-infinity) reject the update;
failure preserves logical raw bytes, masters and [optimizer state](../../glossary.md#term-optimizer-state) transactionally. No implicit
[loss](../../glossary.md#loss) scaling or clipping is provided.

<a id="generic-live-parameter-admission"></a>
## Generic live [parameter](../../glossary.md#term-parameter) [admission](../../glossary.md#term-admission)

[`TrainingParameters`](../../glossary.md#live-parameter-source--trainingparameters) declares `List<Entry> entries()` and `void validateUnchanged()`; its nested
`Entry(String path, Tensor tensor)` rejects null path/[Tensor](../../glossary.md#tensor) and blank paths. Training rejects null
sources/lists/entries, empty membership and duplicate paths before [seed](../../glossary.md#term-seed) allocation, [compilation](../../glossary.md#compiler-engine-and-execution-stage-names),
preparation or copy mutation. Entries expose [immutable](../../glossary.md#term-immutable) ordered membership, not [detached](../../glossary.md#detached-model-results) values.
The source must retain its live [owner](../../glossary.md#term-owner) and validate count/order/paths/exact Tensors by reading that
owner again. Comparing only a frozen list cannot detect replacement.

For [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context), use `module.trainingParameters()`: its private adapter retains the root and exact ordered
paths/wrappers/Tensors, re-traverses on every [validation](../../glossary.md#term-validation), and rejects incomplete reservations at
creation. It deliberately ignores storage associations and bytes. Training checks [descriptors](../../glossary.md#term-descriptor),
exact storage [identity](../../glossary.md#term-identity-meanings)/[schema](../../glossary.md#term-schema)/capacity/[liveness](../../glossary.md#term-liveness)/accessibility and [finite](../../glossary.md#term-finite) values separately.
Validation occurs at [capture](../../glossary.md#term-capture-meanings), before [run](../../glossary.md#term-run)/state/restore, and at complete-[view](../../glossary.md#term-view-meanings) accumulation,
update and restore [precommit](../../glossary.md#term-precommit). A late live-owner failure publishes no Training state.

This complete NN-free example trains one [FLOAT32](../../glossary.md#term-float32) [scalar](../../glossary.md#term-scalar) from `2.0` to `1.6` with one [SGD](../../glossary.md#term-stochastic-gradient-descent-sgd) step on
`p * p`. The [mutable](../../glossary.md#term-mutable) `live` list stands for the owner's current [bindings](../../glossary.md#term-binding), not the frozen entries:

```java
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.training.*;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

Tensor p = TensorFactory.scalar(2.0f, Optional.empty(), true);
List<Tensor> live = new ArrayList<>(List.of(p));
TrainingParameters parameters = new TrainingParameters() {
    private final List<Entry> snapshot = List.of(new Entry("weight", p));
    public List<Entry> entries() { return snapshot; }
    public void validateUnchanged() {
        if (live.size() != 1 || live.getFirst() != p) {
            throw new IllegalStateException("live parameter binding changed");
        }
    }
};
try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
    TrainingParameterStorage.copyToSharedNative(parameters, arena);
    try (TrainingSession session = TrainingSession.open(engine, parameters, p.mul(p), new Sgd(0.1))) {
        session.run(List.of(), GradientMode.RESET_AND_STEP);
        float value = p.hostStorage().orElseThrow().segment().get(ValueLayout.JAVA_FLOAT, 0);
        assert Math.abs(value - 1.6f) < 1e-6f;
    }
}
```

Copy changes only storage, so the same live source opens successfully. The [derivative](../../glossary.md#term-derivative) is `2 * 2 = 4`;
SGD computes `2 - 0.1 * 4 = 1.6`. Replacing `live[0]` with another Tensor would reject the next
[operation](../../glossary.md#operation) even though `entries()` still returns the old Tensor. Callers exclude such mutation
while copying or [borrowing](../../glossary.md#term-borrowing) a session. Neither operation closes caller [arenas](../../glossary.md#term-arena) or transfers [backend](../../glossary.md#backend) values.

<a id="explicit-native-parameter-copy"></a>
## Explicit [native parameter copy](../../glossary.md#native-parameter-copy--trainingparameterstorage)

`TrainingParameterStorage.copyToSharedNative(parameters, arena)` is an opt-in transition for a
live source whose current [parameter](../../glossary.md#term-parameter) storage may be [heap](../../glossary.md#term-heap) or native. The destination
[`Arena`](../../glossary.md#term-arena) must be live and shareable (`Arena.ofShared()` is the normal choice). A zero-byte [scope](../../glossary.md#term-scope)
probe preserves destination-shareability failure precedence, including foreign-thread
`WrongThreadException`, before source storage inspection. The method snapshots ordered paths,
validates the complete non-empty source before destination parameter allocation, copies exact represented
bytes into fresh aligned writable native segments, then changes each existing [Tensor](../../glossary.md#tensor)'s [borrowed](../../glossary.md#ownership-borrowing-and-lease)
[host](../../glossary.md#term-host)-storage association in path order. It returns no [owner](../../glossary.md#term-owner) and never replaces a Parameter or
Tensor, so expressions already built from those Tensors retain their [identities](../../glossary.md#term-identity-meanings). Read-only or
unaligned source storage is accepted; the destination is aligned. Non-[finite](../../glossary.md#term-finite) values, incomplete
initialization, duplicate Tensor identities, overlapping source regions, invalid geometry,
unsupported types or widths, and
missing/dead/inaccessible storage are rejected before any association changes. An ordinary
installation failure restores earlier exact associations and suppresses distinct [rollback](../../glossary.md#term-rollback) failures;
live membership is checked again before installation. Allocations consumed from the supplied
Arena cannot be individually reclaimed.

For example, a caller can construct an initialized [`Linear`](../../glossary.md#term-linear-meanings) and its [objective](../../glossary.md#term-objective), then make the
storage transition before opening a [CPU](../../glossary.md#term-cpu) session:

```java
Linear layer = new Linear(1, 1, true, DataType.FLOAT32, new Random(42));
Tensor output = layer.forward(input); // input is a live FLOAT32 [1, 1] Tensor
Tensor difference = output.sub(target); // target is a live FLOAT32 [1, 1] Tensor
Tensor objective = difference.mul(difference).sum();
try (Arena parameterArena = Arena.ofShared(); Engine engine = Engine.standard()) {
    TrainingParameterStorage.copyToSharedNative(layer.trainingParameters(), parameterArena);
    try (TrainingSession session = TrainingSession.open(
            engine, layer.trainingParameters(), objective, new Sgd(0.1))) {
        session.run(List.of(input, target), GradientMode.RESET_AND_STEP);
    }
}
```

Here `input` and [`target`](../../glossary.md#term-target) must themselves retain valid storage through the [run](../../glossary.md#term-run); the [user guide's
complete example](../../user-guide/training.md#copy-an-initialized-linear) constructs both. The
copy does not [compile](../../glossary.md#compile) or [prepare](../../glossary.md#prepare), and `TrainingSession.open` remains native-only: omitting the
explicit call still rejects heap-backed parameters. The caller keeps both source storage and
the destination Arena under its own [lifetime](../../glossary.md#term-lifetime) policy, grants the copy exclusive [Module](../../glossary.md#term-module)/storage
use, never calls it while a session is open, and closes the destination Arena only after the
session. This is not a [device](../../glossary.md#term-device) transfer or a reduction in [optimizer](../../glossary.md#term-optimizer)-state memory.


---

[Previous topic](sessions.md) · [Next topic](optimizers.md)

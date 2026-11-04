# Parameter domains, admission and native copying

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/training/src/main/java/io/github/pho001/synaptik/training/TrainingSession.java)

<a id="initial-parameter-domain"></a>
## Initial parameter domain

The update path supports non-empty `FLOAT64`, `FLOAT32`, `BFLOAT16`, `FLOAT16`,
`FLOAT8_E4M3`, and `FLOAT8_E5M2` parameters whose
Shapes are fully static and whose layouts are dense-contiguous, offset zero, non-view, and exact-
span. Storage must be writable, exact-capacity, element-aligned native host memory shareable across
operation threads. Initial parameter values must be finite. Empty or non-finite parameters fail
during capture before private seed allocation, compile, or prepare. The session updates accepted
storage in place, preserving the Tensor identity captured by the compiled graph. All four low
types retain private FLOAT32 masters and publish one narrowed logical value only after a
successful update. Each logical snapshot payload and each FLOAT32/FLOAT64-width master,
momentum, or accumulation payload must fit in a Java byte array (at most `Integer.MAX_VALUE`
bytes). Capture rejects an oversized payload before inspecting that parameter's storage,
allocating its update buffers or the private seed, or compiling; earlier parameters may already
have been captured. JVM-heap, read-only, thread-confined, strided/view,
integral, BOOL, and device-only parameter storage are outside this domain. The caller grants the
session exclusive use of the live source's membership, storage associations and parameter bytes until close.

CPU numerical execution uses the existing shared Compiler AD and ordinary operation owners for
all four low types under their family and geometry constraints. Masters do not implicitly change
graph dtypes: the objective must still be a visible scalar FLOAT32/FLOAT64 value, using explicit
casts where needed. Both master candidates and their narrowed logical representations must be
finite before publication. E4M3 overflow to NaN and E5M2 overflow to infinity reject the update;
failure preserves logical raw bytes, masters and optimizer state transactionally. No implicit
loss scaling or clipping is provided.

<a id="generic-live-parameter-admission"></a>
## Generic live parameter admission

`TrainingParameters` declares `List<Entry> entries()` and `void validateUnchanged()`; its nested
`Entry(String path, Tensor tensor)` rejects null path/Tensor and blank paths. Training rejects null
sources/lists/entries, empty membership and duplicate paths before seed allocation, compilation,
preparation or copy mutation. Entries expose immutable ordered membership, not detached values.
The source must retain its live owner and validate count/order/paths/exact Tensors by reading that
owner again. Comparing only a frozen list cannot detect replacement.

For NN, use `module.trainingParameters()`: its private adapter retains the root and exact ordered
paths/wrappers/Tensors, re-traverses on every validation, and rejects incomplete reservations at
creation. It deliberately ignores storage associations and bytes. Training checks descriptors,
exact storage identity/schema/capacity/liveness/accessibility and finite values separately.
Validation occurs at capture, before run/state/restore, and at complete-view accumulation,
update and restore precommit. A late live-owner failure publishes no Training state.

This complete NN-free example trains one FLOAT32 scalar from `2.0` to `1.6` with one SGD step on
`p * p`. The mutable `live` list stands for the owner's current bindings, not the frozen entries:

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

Copy changes only storage, so the same live source opens successfully. The derivative is `2 * 2 = 4`;
SGD computes `2 - 0.1 * 4 = 1.6`. Replacing `live[0]` with another Tensor would reject the next
operation even though `entries()` still returns the old Tensor. Callers exclude such mutation
while copying or borrowing a session. Neither operation closes caller arenas or transfers backend values.

<a id="explicit-native-parameter-copy"></a>
## Explicit native parameter copy

`TrainingParameterStorage.copyToSharedNative(parameters, arena)` is an opt-in transition for a
live source whose current parameter storage may be heap or native. The destination
`Arena` must be live and shareable (`Arena.ofShared()` is the normal choice). A zero-byte scope
probe preserves destination-shareability failure precedence, including foreign-thread
`WrongThreadException`, before source storage inspection. The method snapshots ordered paths,
validates the complete non-empty source before destination parameter allocation, copies exact represented
bytes into fresh aligned writable native segments, then changes each existing Tensor's borrowed
host-storage association in path order. It returns no owner and never replaces a Parameter or
Tensor, so expressions already built from those Tensors retain their identities. Read-only or
unaligned source storage is accepted; the destination is aligned. Non-finite values, incomplete
initialization, duplicate Tensor identities, overlapping source regions, invalid geometry,
unsupported types or widths, and
missing/dead/inaccessible storage are rejected before any association changes. An ordinary
installation failure restores earlier exact associations and suppresses distinct rollback failures;
live membership is checked again before installation. Allocations consumed from the supplied
Arena cannot be individually reclaimed.

For example, a caller can construct an initialized `Linear` and its objective, then make the
storage transition before opening a CPU session:

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

Here `input` and `target` must themselves retain valid storage through the run; the [user guide's
complete example](../../user-guide/training.md#copy-an-initialized-linear) constructs both. The
copy does not compile or prepare, and `TrainingSession.open` remains native-only: omitting the
explicit call still rejects heap-backed parameters. The caller keeps both source storage and
the destination Arena under its own lifetime policy, grants the copy exclusive Module/storage
use, never calls it while a session is open, and closes the destination Arena only after the
session. This is not a device transfer or a reduction in optimizer-state memory.


---

[Previous topic](sessions.md) · [Next topic](optimizers.md)

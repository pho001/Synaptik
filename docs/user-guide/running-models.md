<a id="run-a-reusable-inference-session"></a>

# Run a prepared graph

## Outcome

This guide compiles one fixed inference graph, prepares one closeable `PreparedExecution`, and runs the same
prepared execution repeatedly with current caller-owned input storage. Each run returns typed,
ordered publication metadata and supports explicit copying into detached immutable
`HostTensorValue` objects.

## Prerequisites

- One open `Engine.standard()`, or one explicitly composed CPU/Metal Engine.
- A non-empty graph supported by the Engine's registered backends.
- Every Tensor reported by `CompiledGraph.inputs()`, with compatible live caller-owned host
  storage when each run begins.

The Engine remains the composition owner. A prepared handle borrows it and must not outlive it.
For a supported NN Sequential, [declarative reconstruction](declarative-models.md) can
first build a descriptor-only model with stable state paths. Load its parameter values
before constructing the forward expression to compile; reconstruction itself neither
executes the model nor reconstructs it from a stored-model file.

## Load a saved whole model

To recover a supported bound `Sequential` without rebuilding its layers, use
`ModelCheckpoint.save(path, source, engine)` and `ModelCheckpoint.load(path)`. Load returns a
value-loaded `DeclarativeModel`, not a descriptor-only skeleton. It needs no Engine or Arena;
ordinary forward construction and execution still use the caller's Engine. No backend or prepared
execution is stored in the file.

Use an explicit `:extensions:nn` dependency and import `nn.checkpoint.ModelCheckpoint`;
the generic Checkpoint project no longer exports NN. The old flexible readers/defaults are
unchanged, distinct from [static generic graph persistence](../api/graph-checkpoint-api.md).
This complete Java 26 example needs NN and its public dependencies on the classpath and
`jdk.incubator.vector` enabled. Its trusted local directory is created before save. The source has
FLOAT32 diagonal weights `[1, 0; 0, 2]` followed by RELU. Both source owners close before load;
the saved file is then deleted before fresh-CPU inference. The input `[-3, 4]` should produce
`[0, 8]`.

```java
import io.github.pho001.synaptik.nn.checkpoint.ModelCheckpoint;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.layers.Activation;
import io.github.pho001.synaptik.nn.layers.Linear;
import io.github.pho001.synaptik.nn.module.DeclarativeModel;
import io.github.pho001.synaptik.nn.module.ModelSpecification.ActivationKind;
import io.github.pho001.synaptik.nn.module.Sequential;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

public final class WholeModelExample {
    static Tensor tensor(MemorySegment storage, Shape shape, boolean gradientEligible) {
        var descriptor = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), gradientEligible);
        return TensorFactory.create(descriptor, Optional.empty(), Optional.of(
                new MemorySegmentStorage(DataType.FLOAT32,
                        shape.knownElementCount().orElseThrow(), storage)));
    }

    public static void main(String[] args) throws Exception {
        Path directory = Files.createTempDirectory("whole-model-example");
        Path file = directory.resolve("model.chk");
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            MemorySegment initial = MemorySegment.ofArray(new float[] {1, 0, 0, 2});
            MemorySegment storage = arena.allocate(initial.byteSize(), 4);
            storage.copyFrom(initial);
            Sequential source = new Sequential(List.of(
                    new Linear(tensor(storage, Shape.of(2, 2), true)),
                    new Activation(ActivationKind.RELU)));
            source.train();
            ModelCheckpoint.save(file, source, engine);
        }
        DeclarativeModel loaded = ModelCheckpoint.load(file);
        Files.delete(file);
        Tensor input = tensor(MemorySegment.ofArray(new float[] {-3, 4}), Shape.of(2), false);
        try (Engine fresh = Engine.standard()) {
            var result = fresh.compute(loaded.forward(input)).bytes().asFloatBuffer();
            float first = result.get();
            float second = result.get();
            if (first != 0 || second != 8) throw new AssertionError("restored inference");
            ModelCheckpoint.save(directory.resolve("resaved.chk"), loaded, fresh);
            System.out.println(first + ", " + second);
        }
    }
}
```

Expected output is `0.0, 8.0`. Linear first produces `[-3, 8]`; RELU produces `[0, 8]`.
The source's training mode is not persisted: the loaded tree starts in evaluation mode, while
parameters remain gradient-eligible. Its heap values survive source Arena/Engine closure and file
deletion; re-saving needs no caller expression wrappers or native parameter copy. This does not
prove a saved execution plan or training continuation. Exact repeated parameter identities are
preserved, and compatible batch dimensions can change when a new forward expression is built.

The short calls use absent input-artifact identity. If the file stores one, default load raises
`ArtifactFormatException`; pass an application-verified `InputArtifactFingerprint` and explicit
`ArtifactLimits` instead. A values-only `model.v1` file needs the existing fully bound target-load
API, not automatic topology inference. Custom modules, unbound state and recurrent models are not
supported by this whole-model workflow. See [whole-model limits and failures](../api/model-checkpoint-api.md#bounds-lifetime-and-failures)
before selecting advanced encoded bounds; they are not peak-heap budgets.

## Compile and prepare once

Compile the requested output boundary once, then retain its prepared owner:

```java
CompiledGraph graph = engine.compile(List.of(output));
try (PreparedExecution prepared = engine.prepare(graph)) {
    assert prepared.compiledGraph() == graph;
    // Reuse engine.run(prepared, ...) while both prepared handle and Engine remain open.
}
```

`engine.prepare(graph)` performs exactly one preparation before returning. It does not execute the
graph or bind input storage. The graph fixes both boundaries:

- `graph.inputs()` is the stable input membership and Compiler occurrence order; and
- every result uses the graph's stable forward-then-gradient publication occurrence order,
  including distinct occurrences that alias one inward representation.

## Run and materialize

The input list may use any order because Engine binds by exact Tensor identity:

```java
HostTensorValue retained;
try (PreparedExecution prepared = engine.prepare(graph)) {
    try (RunResult result = engine.run(prepared, List.of(rightInput, leftInput))) {
        RunResult.Publication first = result.publications().getFirst();
        retained = result.materialize(first, 2L * Float.BYTES);
    }

    // Replace a required Tensor's host-storage association here when the next call
    // should use another compatible caller-owned storage.
    try (RunResult result = engine.run(prepared, List.of(leftInput, rightInput))) {
        assert result.publications().getFirst().index() == 0;
    }
}

assert retained.byteSize() == 2L * Float.BYTES;
```

Each call verifies exact Tensor identity and the complete compiled descriptor, then snapshots the
current host-storage associations in `graph.inputs()` order. It validates storage data type,
resolved-layout capacity, liveness, and current-thread accessibility before borrowing. The caller
continues to own the storage and must keep it usable and free from conflicting mutation until that
run's result closes.

`RunResult` is the existing publication lease. Each `Publication` records its dense index,
`TensorId`, final descriptor, forward or gradient role, and applicable derivative metadata.
`materialize(...)` accepts only the exact occurrence object from that result and performs a fresh
bounded copy through the publication adapter captured during preparation. A detached
`HostTensorValue` remains readable after the result, prepared handle, Engine, and input-storage arena close.

## Run fresh values in compiled order

**Goal and inputs.** Prepare negation of a storage-free FLOAT32 [2] prototype once, then supply
fresh batches [1,-2] and [3,-4]. With Java 26, Engine and Model on the classpath and
`jdk.incubator.vector` enabled, this complete block uses their ordinary CPU heap-input support:

```java
var shape = Shape.of(2);
var descriptor = new TensorDescriptor(DataType.FLOAT32, shape,
        Optional.of(LayoutDescriptor.contiguous(shape)), false);
Tensor prototype = TensorFactory.create(descriptor);
Tensor firstBatch = TensorFactory.fromFlatArray(
        descriptor, Optional.empty(), new float[] {1, -2});
Tensor secondBatch = TensorFactory.fromFlatArray(
        descriptor, Optional.empty(), new float[] {3, -4});
try (Engine engine = Engine.standard()) {
    var graph = engine.compile(List.of(prototype.neg()));
    try (var prepared = engine.prepare(graph)) {
        for (Tensor batch : List.of(firstBatch, secondBatch)) {
            try (var result = engine.runValues(prepared, List.of(batch))) {
                var values = result.materialize(result.publications().getFirst(), 8).bytes();
                System.out.println(values.getFloat() + ", " + values.getFloat());
            }
        }
    }
}
```

Use the Model/Engine imports from the whole-model example above. Compilation fixes the single
input position; preparation is outside the loop. Each run binds that batch's current storage
without changing the prototype. Output is `-1.0, 2.0`, then `-3.0, 4.0`; detached values may be
retained after every owner closes. With multiple inputs, follow `graph.inputs()` order, not
expression discovery order. A value may fill multiple matching slots.

runValues requires exact count and full static descriptors (dtype, shape, layout including absence
versus presence, and requiresGrad). A compiled [32,784] input rejects [64,784]. All storage is
validated before any borrow, including checked referenced spans for resolved layouts or checked
canonical element/byte geometry for absent layouts. Unknown/overflow geometry rejects. This extra
absent-layout check does not change identity-based run; `engine.run(prepared, List.of(firstBatch))`
rejects here because the fresh ID is not the prototype ID.

## Reuse and concurrency

Sequential and concurrent calls share the immutable prepared recipe and direct captured
input/publication adapters. Every call receives a distinct mutable Runtime `RunState`, borrowed
input wrappers, internal buffers, and workspaces. A run performs no compilation, preparation,
provider or availability interrogation, Engine-registry lookup, backend selection, kernel
selection, or schedule assembly.

The prepared owner adds no cache or result type. `Engine.compute(...)` and
`Engine.backward(...)` make a different lifetime choice: every call performs fresh compilation,
preparation, execution, materialization, and cleanup.

## Close and in-flight runs

Close the prepared handle when its reuse window ends. Close is idempotent, closes the one prepared
execution exactly once, and immediately prevents later run admission. It does not close a
`RunResult` that was already returned.

If close races a run, Runtime's existing prepared lease decides the race:

- a run that acquired the lease first may finish; prepared close does not wait for it; and
- a close-first run is rejected.

Engine admission is checked first. Under an open Engine, both run methods check prepared nullness/exact ownership and the exact inward delegate's close
gate before reading the input list, Tensor metadata, storage, or adapters. A run attempted after
prepared close therefore reports `prepared execution is closed` without borrowing storage even when
the supplied inputs are invalid. If Engine closure has begun, `advanced engine is closed` wins
before any argument inspection.

Engine closure waits for admitted Engine operations, then closes retained results in reverse
successful-run order, ordinary and advanced prepared handles in reverse
preparation-publication order, and finally backend integrations. Engine closure is the final
cleanup boundary for a prepared handle the caller leaves open.

Both methods now check the exact prepared delegate before inputs. This intentionally changes
former direct-run diagnostic precedence: closed prepared wins null/malformed/dead inputs without
borrowing; null or foreign prepared handles also precede list errors.

`InferenceSession` and `Engine.session` are removed. Migrate to prepare/run explicitly and
recompile; no replacement wrapper is introduced.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| Preparation rejects a graph | The graph belongs to another Engine, Engine closure began, or cold preparation rejected its plan. | Compile and prepare with the same open Engine; inspect the prepare-time capability failure. |
| A run reports a missing, duplicate, or unexpected input | The supplied identities do not exactly match `graph.inputs()`. | Supply every required Tensor exactly once; list order is arbitrary. |
| A run rejects a descriptor or storage | The Tensor descriptor changed identity expectations, or storage type, capacity, lifetime, or thread accessibility is incompatible. | Use the exact compiled Tensor and attach compatible accessible live storage before the run. |
| A run reports `prepared execution is closed` | Prepared close won admission. | Prepare a new handle from the still-valid graph and Engine if more runs are needed. |
| Materialization rejects a publication | The occurrence belongs to another result, the result or Engine is closed, or the byte limit is too small. | Use an occurrence from the same open result and a sufficient checked bound. |

## Limitations

Prepared runs are synchronous and fixed to one compiled graph. Fresh identities are supported only
by runValues; neither method provides dynamic descriptors, asynchronous submission, implicit
materialization, persistence, generic device-result handles, or tuning controls. Explicit
CPU/Metal composition supports current single-owner execution and the bounded all-seven-carrier
rank-0..16 static canonical or positive-stride non-overlapping transfer domain; unsupported
capability or transfer combinations fail during compile or preparation rather than
falling back during a run.

## Related documentation

- [Whole-model and values-only checkpoint API](../api/model-checkpoint-api.md)
- [Describe, reconstruct and load a closed NN model](declarative-models.md)
- [Compile a graph](compiling-graphs.md)
- [Prepare execution](preparing-execution.md)
- [Runtime API](../api/runtime-api.md#current-ordinary-engine-boundary)
- [Public API status](../api/public-api.md#current-ordinary-explicit-composition-and-advanced-lifecycle)
- [Lifecycle architecture](../architecture/lifecycle.md#current-public-engine-lifecycle)

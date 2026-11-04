# File-free and file-backed recovery examples

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/graph/GraphCheckpoint.java)

<a id="complete-example"></a>
## Complete example

<a id="file-free-example"></a>
### File-free example

Goal: capture and restore exact stored values without file staging, and compare the complete
parameter descriptors and raw bytes using only public APIs. This complete Java 26 class uses
`Engine.standard()` on the supported CPU route. Its source has a native scalar FLOAT32 parameter
`重み` containing `-0.0f` and a heap INT32 buffer `step` containing `1`. The configuration SHA-256
identifies this example's versioned schema; a real application must cover its complete configuration
and intended sharing. No input artifact is present (`null`).

Save the following class as `ModelEnvelopeExample.java` and compile/run with the project's public
Checkpoint/NN/Model/Engine dependencies on the classpath and `jdk.incubator.vector` enabled.
No Training API, reflection, model-internal type, or filesystem operation is used by the example.

```java
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactEnvelope;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits;
import io.github.pho001.synaptik.nn.checkpoint.ModelCheckpoint;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.module.Module;
import io.github.pho001.synaptik.nn.module.StateKind;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Optional;

public final class ModelEnvelopeExample {
    static final class Tree extends Module {
        Tree(Tensor weight) {
            parameter("重み", weight);
            buffer("step", TensorFactory.scalar(1, Optional.empty(), false));
        }
    }

    public static void main(String[] args) throws Exception {
        var descriptor = new TensorDescriptor(DataType.FLOAT32, Shape.scalar(),
                Optional.of(LayoutDescriptor.contiguous(Shape.scalar())), true);
        byte[] config = MessageDigest.getInstance("SHA-256").digest(
                "scalar-weight-step-v1".getBytes(StandardCharsets.UTF_8));
        var limits = new ArtifactLimits(4096, 2048, 8, 32, 1024, 2048);
        ArtifactEnvelope envelope;
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            var segment = arena.allocate(4, 4);
            segment.set(ValueLayout.JAVA_FLOAT, 0, -0.0f);
            Tensor weight = TensorFactory.create(descriptor, Optional.empty(), Optional.of(
                    new MemorySegmentStorage(DataType.FLOAT32, 1, segment)));
            envelope = ModelCheckpoint.capture(new Tree(weight), engine, config, null, limits);
        }
        // Both source owners are now closed; the envelope still owns detached bytes.
        var inspection = ModelCheckpoint.inspect(envelope, limits);
        var bytes = ModelCheckpoint.canonicalStateBytes(envelope, limits);
        var parameters = inspection.states().stream()
                .filter(state -> state.kind() == StateKind.PARAMETER).toList();
        if (parameters.size() != 1 || bytes.size() != 2) throw new AssertionError("state set");
        for (var state : parameters) {
            if (!state.path().equals("重み") || state.dataType() != DataType.FLOAT32
                    || !state.shape().equals(Shape.scalar()) || state.byteLength() != 4
                    || bytes.get(state.path()).getInt(0) != 0x80000000)
                throw new AssertionError("descriptor or raw bits");
        }
        Tree target = new Tree(TensorFactory.scalar(1.0f, Optional.empty(), true));
        ModelCheckpoint.load(envelope, target, config, null, limits);
        try (Engine engine = Engine.standard()) {
            var restored = ModelCheckpoint.capture(target, engine, config, null, limits);
            if (!inspection.states().equals(ModelCheckpoint.inspect(restored, limits).states())
                    || !bytes.equals(ModelCheckpoint.canonicalStateBytes(restored, limits)))
                throw new AssertionError("round trip");
        }
        System.out.println("parameter 重み: FLOAT32 scalar, 4 bytes, raw 80000000; round trip OK");
    }
}
```

Capture preflights metadata and limits before materialization. After closing the source owners,
inspection reports `重み` (PARAMETER, FLOAT32, scalar, 4 bytes) followed by `step` (BUFFER,
INT32, scalar, 4 bytes); the byte map contains both in that order. Role filtering selects only
`重み`, without confusing buffers with optimizer parameters. Absolute `getInt(0)` observes raw
`80000000` without moving the buffer cursor. Load replaces the recreated tree's bindings with
fresh heap-backed leaves; recapture verifies matching descriptors and every stored byte.
The printed result is:

```text
parameter 重み: FLOAT32 scalar, 4 bytes, raw 80000000; round trip OK
```

For a future consumer comparing with TrainingState, require the complete parameter path set,
role, data type, Shape, byte length, and raw logical bytes to match by exact path; checking only
the first parameter or numerical equality is insufficient. Positive zero (`00000000`) compares
numerically equal to negative zero but differs as stored state. Raw comparison also distinguishes
NaN sign/payload bits; numerical tolerances and NaN normalization do not apply here. Low-type
logical bytes are not FLOAT32 master bytes. Matching model bytes alone proves neither Training
compatibility nor exact continuation; each owner still validates its own restore.

Failure variation: use a different 32-byte expected configuration digest at load. The call raises
`IllegalArgumentException` before changing any target binding. With `maxEntries` set to `2`,
this three-payload envelope (schema plus two paths) instead fails envelope decode with
`ArtifactFormatException`. A prior successful inspection does not waive the new call's limits.

<a id="file-example"></a>
### File example

This JShell-style example saves one rank-one `FLOAT32` buffer. The trusted local directory exists
before save. The source uses a live native host segment; the target starts with an independent heap
Tensor of the same descriptor. The result is one restored buffer, a typed inspection reporting
path `runningMean` and eight payload bytes, a direct save of the loaded tree, and Engine
materialization through an explicit contiguous expression.

```java
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits;
import io.github.pho001.synaptik.nn.checkpoint.ModelCheckpoint;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.module.Module;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Optional;

class StateModule extends Module {
    StateModule(Tensor value) { buffer("runningMean", value); }
}

Shape shape = Shape.of(2);
TensorDescriptor descriptor = new TensorDescriptor(DataType.FLOAT32, shape,
        Optional.of(LayoutDescriptor.contiguous(shape)), false);
byte[] config = MessageDigest.getInstance("SHA-256")
        .digest("example-model-config-v1".getBytes(StandardCharsets.UTF_8));
ArtifactLimits limits = new ArtifactLimits(4096, 2048, 4, 32, 1024, 2048);
Path directory = Files.createTempDirectory("synaptik-model-example");
Path file = directory.resolve("model.chk");

try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
    MemorySegment sourceBytes = arena.allocate(8, 4);
    sourceBytes.setAtIndex(ValueLayout.JAVA_FLOAT, 0, 1.0f);
    sourceBytes.setAtIndex(ValueLayout.JAVA_FLOAT, 1, -0.0f);
    Tensor source = TensorFactory.create(descriptor, Optional.empty(), Optional.of(
            new MemorySegmentStorage(DataType.FLOAT32, 2, sourceBytes)));
    ModelCheckpoint.save(file, new StateModule(source), engine,
            config, null, limits);
}

var inspected = ModelCheckpoint.inspect(file, limits);
// inspected.states().getFirst().path() is "runningMean"; byteLength() is 8.
StateModule target = new StateModule(TensorFactory.allocate(descriptor));
ModelCheckpoint.load(file, target, config, null, limits);
Tensor restored = target.stateDictionary().entries().getFirst().value();
try (Engine engine = Engine.standard()) {
    ModelCheckpoint.save(directory.resolve("resaved.chk"), target, engine, config, null, limits);
    var computed = engine.compute(restored.contiguous(), 1024);
    int firstBits = computed.bytes().getInt(0);  // 0x3f800000 (1.0f)
    int secondBits = computed.bytes().getInt(4); // 0x80000000 (-0.0f)
}
```

`restored` is a fresh heap-backed leaf Tensor. Saving `target` needs no caller wrapper and leaves
its bindings unchanged; `Engine.compute(restored.contiguous(), ...)` separately returns its two
values as canonical big-endian bytes, including the signed-zero bit. The example demonstrates model
state recovery and execution, not training-session resume. If `config` differs, or if the target
path, type, or Shape differs, load rejects the entire candidate and leaves the target's prior
binding unchanged. Inspection alone does not authorize loading a later-mutated file.

<a id="failures-and-limitations"></a>
## Failures and limitations

`ArtifactFormatException` reports malformed, unsupported, semantically invalid BOOL, or over-limit
models during final capture decode and every envelope read/load. Capture's metadata/bounds preflight
retains unchecked `IllegalArgumentException`/`ArithmeticException`; incomplete NN state and Engine
materialization failures remain unchecked rather than silently translated. The file overloads retain
`IOException` declarations: `ArtifactFormatException` is an `IOException` subtype and also reports
corrupt file framing/checksums; other `IOException`s report file and publication failures. Invalid expected
digests, target schema, or target alias-value conflicts raise `IllegalArgumentException`. An
unbound target raises `IllegalStateException`. SHA-256 is an integrity and identity check, not
adversarial authentication or encryption. Limits bound encoded bytes, not peak JVM heap or Engine
workspace. The retained aggregate payload bound counts schema plus every path's canonical bytes,
including repeated aliases. The Engine materialization bound counts only unique state outputs;
neither bound includes transient workspace, detached source values, defensive copies, or peak
heap. The v1 format has no migration or source-alias metadata.

All required reference arguments are non-null; only the input digest may be absent. File save
checks target path presence before capture. Both load overloads check target presence, expected
configuration presence, and expected digest lengths before reading/decoding; file source presence
then precedes limits presence. Malformed model bytes raise the checked format exception, while
expected digest/target compatibility failures remain unchecked. Cast an untyped `null` to `Path`
or `ArtifactEnvelope` when selecting an overloaded `inspect` or `load` in diagnostic code.

Every `ArtifactLimits` dimension applies even without file I/O: complete file-equivalent bytes,
manifest bytes, entry count, kind/key ASCII length, individual payload bytes (including schema),
and aggregate schema plus all paths. Unicode NN paths are metadata, not artifact keys and not
sorted or normalized; fixed ordinal keys carry their payloads. These are encoded/canonical-size
ceilings, not a bound on copies or peak memory. Keep resource budgets appropriate to the application.

Model-only capture, canonical-byte access, save, inspection, and load are available; restored heap Tensors can be passed
to Engine through supported expressions for computation on the validated CPU path. This API does not persist
`TrainingState`, module modes, caller progress, or other state needed for exact training
continuation. A caller-coordinated [private training restore](../training/recovery.md#caller-coordinated-in-memory-recovery)
can load a model artifact into a distinct, fully bound unpublished tree, but no durable
`TrainingCheckpoint` adapter or built-in cross-owner transaction exists.

The focused integration test saves the actual quiescent trained tree, then strict-loads and
directly re-saves its heap-backed replacement before any native parameter copy. Complete file
equality checks that intermediate round trip. Caller exclusion and live, accessible source
storage remain required throughout save; arbitrary Engine compositions must support the selected
movement and existing expression occurrences. No Engine/Prepare zero-node support is implied.

<a id="related-api-and-architecture"></a>
## Related API and architecture

- [Opaque artifact file API](artifacts.md)
- [Checkpoint boundary explanation](../../architecture/module-boundaries.md#retained-nn-checkpoint-glue)
- [Public model checkpoint decision](../../design/decisions/0034-public-model-checkpoint.md)
- [Model-envelope composition decision](../../design/decisions/0036-model-envelope-composition.md)

---

[Previous topic](values.md)

# File-free and file-backed recovery examples

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/graph/GraphCheckpoint.java)

These examples compare **[raw bits](../../glossary.md#storage-mapping-contiguity-and-injectivity)**, the exact stored representation of each [element](../../glossary.md#term-element), rather
than approximate numeric equality. For example, positive and negative [floating-point](../../glossary.md#term-floating-point) zero are
numerically equal but have different bits. [Canonical](../../glossary.md#term-canonical-form) bytes use one specified element order and
[byte order](../../glossary.md#text-encoding-and-file-boundaries) so values from different storage [owners](../../glossary.md#term-owner) can be compared without interpreting [native memory](../../glossary.md#aliasing-and-arena).

<a id="complete-example"></a>
## Complete example

<a id="file-free-example"></a>
### File-free example

Goal: [capture](../../glossary.md#term-capture-meanings) and restore exact stored values without file staging, and compare the complete
[parameter](../../glossary.md#term-parameter) [descriptors](../../glossary.md#term-descriptor) and raw bytes using only public APIs. This complete Java 26 class uses
`Engine.standard()` on the supported [CPU](../../glossary.md#term-cpu) route. Its source has a native [scalar](../../glossary.md#term-scalar) [FLOAT32](../../glossary.md#term-float32) parameter
`重み` containing `-0.0f` and a [heap](../../glossary.md#term-heap) [INT32](../../glossary.md#data-type--datatype) [buffer](../../glossary.md#term-buffer) `step` containing `1`. The configuration [SHA-256](../../glossary.md#term-sha-256)
identifies this example's versioned [schema](../../glossary.md#term-schema); a real application must cover its complete configuration
and intended sharing. No input [artifact](../../glossary.md#term-artifact) is present (`null`).

Save the following class as `ModelEnvelopeExample.java` and [compile](../../glossary.md#compile)/[run](../../glossary.md#term-run) with the project's public
[Checkpoint](../../glossary.md#term-checkpoint)/[NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/[Model](../../glossary.md#term-model)/[Engine](../../glossary.md#term-engine) dependencies on the [classpath](../../glossary.md#term-classpath) and `jdk.incubator.vector` enabled.
No Training [API](../../glossary.md#term-application-programming-interface-api), reflection, model-internal type, or filesystem [operation](../../glossary.md#operation) is used by the example.

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

Capture preflights [metadata](../../glossary.md#term-metadata) and limits before [materialization](../../glossary.md#materialization). After closing the source [owners](../../glossary.md#term-owner),
inspection reports `重み` (PARAMETER, FLOAT32, scalar, 4 bytes) followed by `step` (BUFFER,
INT32, scalar, 4 bytes); the byte map contains both in that order. Role filtering selects only
`重み`, without confusing buffers with [optimizer](../../glossary.md#term-optimizer) parameters. Absolute `getInt(0)` observes raw
`80000000` without moving the buffer cursor. Load replaces the recreated tree's [bindings](../../glossary.md#term-binding) with
fresh heap-backed leaves; recapture verifies matching descriptors and every stored byte.
The printed result is:

```text
parameter 重み: FLOAT32 scalar, 4 bytes, raw 80000000; round trip OK
```

For a future consumer comparing with TrainingState, require the complete parameter path set,
role, [data type](../../glossary.md#data-type--datatype), [Shape](../../glossary.md#shape), byte length, and raw logical bytes to match by exact path; checking only
the first parameter or numerical equality is insufficient. [Positive zero](../../glossary.md#nan-infinity-and-signed-zero) (`00000000`) compares
numerically equal to [negative zero](../../glossary.md#nan-infinity-and-signed-zero) but differs as stored state. Raw comparison also distinguishes
[NaN](../../glossary.md#term-nan) sign/[payload](../../glossary.md#artifact-envelope-and-atomic-publication) bits; numerical tolerances and NaN [normalization](../../glossary.md#term-normalization) do not apply here. Low-type
logical bytes are not FLOAT32 master bytes. Matching model bytes alone proves neither Training
compatibility nor exact continuation; each [owner](../../glossary.md#term-owner) still validates its own restore.

Failure variation: use a different 32-byte expected configuration [digest](../../glossary.md#term-digest) at load. The call raises
`IllegalArgumentException` before changing any [target](../../glossary.md#term-target) binding. With `maxEntries` set to `2`,
this three-payload envelope (schema plus two paths) instead fails envelope decode with
`ArtifactFormatException`. A prior successful inspection does not waive the new call's limits.

<a id="file-example"></a>
### File example

This JShell-style example saves one [rank-one](../../glossary.md#axis-rank-scalar-vector-and-matrix) [`FLOAT32`](../../glossary.md#term-float32) [buffer](../../glossary.md#term-buffer). The trusted local directory exists
before save. The source uses a live native [host](../../glossary.md#term-host) segment; the [target](../../glossary.md#term-target) starts with an independent [heap](../../glossary.md#term-heap)
[Tensor](../../glossary.md#tensor) of the same [descriptor](../../glossary.md#term-descriptor). The result is one restored buffer, a typed inspection reporting
path `runningMean` and eight [payload](../../glossary.md#artifact-envelope-and-atomic-publication) bytes, a direct save of the loaded tree, and [Engine](../../glossary.md#term-engine)
[materialization](../../glossary.md#materialization) through an explicit [contiguous](../../glossary.md#term-contiguous) expression.

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

`restored` is a fresh heap-backed [leaf](../../glossary.md#term-leaf) Tensor. Saving `target` needs no caller wrapper and leaves
its [bindings](../../glossary.md#term-binding) unchanged; `Engine.compute(restored.contiguous(), ...)` separately returns its two
values as [canonical](../../glossary.md#term-canonical-form) [big-endian](../../glossary.md#text-encoding-and-file-boundaries) bytes, including the [signed-zero](../../glossary.md#nan-infinity-and-signed-zero) bit. The example demonstrates [model](../../glossary.md#term-model)
state recovery and execution, not training-session [resume](../../glossary.md#schema-serialization-and-checkpoint). If `config` differs, or if the target
path, type, or [Shape](../../glossary.md#shape) differs, load rejects the entire candidate and leaves the target's prior
binding unchanged. Inspection alone does not authorize loading a later-mutated file.

<a id="failures-and-limitations"></a>
## Failures and limitations

`ArtifactFormatException` reports malformed, unsupported, semantically invalid [BOOL](../../glossary.md#data-type--datatype), or over-limit
models during final [capture](../../glossary.md#term-capture-meanings) decode and every envelope read/load. Capture's [metadata](../../glossary.md#term-metadata)/bounds [preflight](../../glossary.md#term-preflight)
retains unchecked `IllegalArgumentException`/`ArithmeticException`; incomplete [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) state and [Engine](../../glossary.md#term-engine)
[materialization](../../glossary.md#materialization) failures remain unchecked rather than silently translated. The file overloads retain
`IOException` declarations: `ArtifactFormatException` is an `IOException` subtype and also reports
corrupt file [framing](../../glossary.md#term-framing)/checksums; other `IOException`s report file and [publication](../../glossary.md#term-publication) failures. Invalid expected
digests, [target](../../glossary.md#term-target) [schema](../../glossary.md#term-schema), or target [alias](../../glossary.md#aliasing-and-arena)-value conflicts raise `IllegalArgumentException`. An
unbound target raises `IllegalStateException`. [SHA-256](../../glossary.md#term-sha-256) is an [integrity](../../glossary.md#term-integrity) and [identity](../../glossary.md#term-identity-meanings) check, not
adversarial [authentication](../../glossary.md#term-authentication) or encryption. Limits bound encoded bytes, not peak [JVM](../../glossary.md#term-java-virtual-machine-jvm) [heap](../../glossary.md#term-heap) or Engine
workspace. The retained aggregate [payload](../../glossary.md#artifact-envelope-and-atomic-publication) bound counts schema plus every path's [canonical](../../glossary.md#term-canonical-form) bytes,
including repeated aliases. The Engine materialization bound counts only unique state outputs;
neither bound includes transient workspace, [detached](../../glossary.md#detached-model-results) source values, defensive copies, or peak
heap. The v1 format has no migration or source-alias metadata.

All required reference [arguments](../../glossary.md#term-argument) are non-null; only the input [digest](../../glossary.md#term-digest) may be absent. File save
checks target path presence before capture. Both load overloads check target presence, expected
configuration presence, and expected digest lengths before reading/decoding; file source presence
then precedes limits presence. Malformed [model](../../glossary.md#term-model) bytes raise the checked format exception, while
expected digest/target compatibility failures remain unchecked. Cast an untyped `null` to `Path`
or `ArtifactEnvelope` when selecting an overloaded `inspect` or `load` in diagnostic code.

Every `ArtifactLimits` [dimension](../../glossary.md#dimension) applies even without file I/O: complete file-equivalent bytes,
[manifest](../../glossary.md#term-manifest) bytes, entry count, kind/key [ASCII](../../glossary.md#term-ascii) length, individual payload bytes (including schema),
and aggregate schema plus all paths. [Unicode](../../glossary.md#term-unicode) NN paths are metadata, not [artifact](../../glossary.md#term-artifact) keys and not
sorted or normalized; fixed ordinal keys [carry](../../glossary.md#term-carry) their payloads. These are encoded/canonical-size
ceilings, not a bound on copies or peak memory. Keep resource budgets appropriate to the application.

Model-only capture, canonical-byte access, save, inspection, and load are available; restored heap [Tensors](../../glossary.md#tensor) can be passed
to Engine through supported expressions for computation on the validated [CPU](../../glossary.md#term-cpu) path. This [API](../../glossary.md#term-application-programming-interface-api) does not persist
`TrainingState`, [module](../../glossary.md#term-module) modes, caller progress, or other state needed for [exact training
continuation](../../glossary.md#exact-training-continuation). A caller-coordinated [private training restore](../training/recovery.md#caller-coordinated-in-memory-recovery)
can load a model artifact into a distinct, fully bound unpublished tree, but no durable
`TrainingCheckpoint` adapter or built-in cross-[owner](../../glossary.md#term-owner) [transaction](../../glossary.md#term-transaction) exists.

The focused [integration test](../../glossary.md#term-integration-test) saves the actual [quiescent](../../glossary.md#term-quiescent) trained tree, then strict-loads and
directly re-saves its heap-backed replacement before any [native parameter copy](../../glossary.md#native-parameter-copy--trainingparameterstorage). Complete file
equality checks that intermediate [round trip](../../glossary.md#term-round-trip). Caller exclusion and live, accessible source
storage remain required throughout save; arbitrary Engine compositions must support the selected
movement and existing expression occurrences. No Engine/[Prepare](../../glossary.md#prepare) zero-[node](../../glossary.md#node) support is implied.

<a id="related-api-and-architecture"></a>
## Related [API](../../glossary.md#term-application-programming-interface-api) and architecture

- [Opaque artifact file API](artifacts.md)<span data-glossary-navigation="true"> ([Opaque glossary](../../glossary.md#term-opaque); [artifact glossary](../../glossary.md#term-artifact))</span>
- [Checkpoint boundary explanation](../../architecture/module-boundaries.md#retained-nn-checkpoint-glue)<span data-glossary-navigation="true"> ([Checkpoint glossary](../../glossary.md#term-checkpoint))</span>
- [Public model checkpoint decision](../../design/decisions/0034-public-model-checkpoint.md)<span data-glossary-navigation="true"> ([model glossary](../../glossary.md#term-model))</span>
- [Model-envelope composition decision](../../design/decisions/0036-model-envelope-composition.md)

---

[Previous topic](values.md)

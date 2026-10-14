# Model checkpoint API

## Purpose and mental model

`ModelCheckpoint` saves a fully bound neural-network (NN) module's parameter and buffer values,
inspects the resulting file, and loads those values into a recreated fully bound module of the same
schema. The [architecture contract](../architecture/contracts/extensions-training.md#extensionscheckpoint)
owns this model-only boundary. It does not persist a Java module class, optimizer, or training
continuation.

```text
quiescent Module --stateDictionary--> Checkpoint --public Engine.compute--> model.v1 file
model.v1 file --full validation--> typed inspection
model.v1 file --full validation + target alias checks--> NN strict state installation
```

The arrows show who owns each stage: NN exports and installs state, Checkpoint validates model
metadata and bytes, and Engine materializes save-time values. Loading does not use an Engine.

## Core contracts

`save(Path, Module, Engine, configDigest, inputDigest, ArtifactLimits)` snapshots the complete
state once through public Engine materialization and publishes the existing version-one envelope.
Fully bound native and matching primitive-array heap leaves, including previously loaded state,
can be saved directly on the validated CPU route; callers need no expression wrappers.
`inspect(Path, ArtifactLimits)`
validates the whole file, including checksums, the model schema, every payload, and canonical
`BOOL` bytes (`0` or `1`). It returns ordered typed path, role, data type, Shape, and byte-length
descriptors plus defensive digest copies. The inspection owns no file or live module/Engine data.
`load(Path, Module, expectedConfigDigest, expectedInputDigest, ArtifactLimits)` rereads and
revalidates its path; it does not trust a prior inspection of a mutable file.

Both caller digests are SHA-256 byte arrays of exactly 32 bytes. The configuration fingerprint
must include the application's versioned model configuration and intended sharing topology.
The optional input-artifact fingerprint is `null` when absent; presence and all bytes must match
exactly at load. Checkpoint does not interpret either fingerprint or serialize model-defining Java
code. The first version stores one value per state path and cannot prove source Tensor identity.

The target must already be fully bound. `Module.stateDictionary()` is called before installation
to reject lazy/unbound groups, even though NN's in-memory loader can initialize those groups.
The recreated target's exact Tensor-identity groups define aliases. Every path in one group must
have byte-identical stored values; one fresh dense-contiguous host Tensor is then installed for
that group. A group containing a parameter is gradient-eligible; buffer-only groups are not.
All seven current data types use compatible primitive-array heap carriers and native byte order.
Floating values are restored by raw bits, without conversion or NaN normalization. Rank-zero
shapes contain one element; a zero extent produces an empty payload.

## Inputs, outputs, and ownership

The caller owns the module, Engine, file path, and trusted local parent directory. Save and load
require quiescence of bindings and relevant storage for their whole calls. The returned inspection
is immutable and detached; its digests are copies and its state list cannot be changed. Loaded
heap storage has ordinary garbage-collected lifetime, with no Checkpoint-owned arena or close
operation. NN's wrappers remain, but their Tensor bindings change after a successful load.

## Lifecycle and invariants

Before Engine work or payload-sized allocation, capture preflights caller fingerprints, original
ordered paths/kinds/types/Shapes, checked byte lengths, and entry, payload, aggregate, manifest,
and final-file limits. Only afterward does it deduplicate original Tensor identities in
first-encounter order. Internally, each unique provenance-free leaf becomes a raw-preserving
`CONTIGUOUS` request; an existing expression stays an expression. These requests normalize
save-time outputs, not NN bindings or schema descriptors. One public `Engine.compute` call
materializes nonempty state and checks returned type, Shape, and byte length. Empty state makes
no Engine call, even with a closed Engine. Each path still receives a separate payload.

This raw movement preserves signed zeros, subnormals, infinities, and NaN payload/sign bits;
it uses neither identity arithmetic nor casts. Tests cover all seven data types on native and
matching heap carriers, scalar and zero-extent state, and mixed leaves/expressions. It is not a
generic promise that Engine prepares provenance-free outputs, or a backend capability expansion.

Checkpoint validates all file bytes, digests, paths, kinds, data types, Shapes, alias values, and
reconstructed candidates before one NN whole-tree installation attempt. Ordinary validation or
reconstruction failure changes no target binding. There is no cross-owner transaction or
concurrent-reader snapshot guarantee. Save reuses the artifact API's sibling temporary-file,
force, reopen, validation, and one atomic-move path. Successful return gives atomic live
visibility of a complete prior or new file, not portable power-loss durability of the rename.
A failed move can have provider-dependent target state; inspect the target before retrying.

## Complete example

This JShell-style example saves one rank-one `FLOAT32` buffer. The trusted local directory exists
before save. The source uses a live native host segment; the target starts with an independent heap
Tensor of the same descriptor. The result is one restored buffer, a typed inspection reporting
path `runningMean` and eight payload bytes, a direct save of the loaded tree, and Engine
materialization through an explicit contiguous expression.

```java
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits;
import io.github.pho001.synaptik.checkpoint.model.ModelCheckpoint;
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

## Failures and limitations

`ArtifactFormatException` reports malformed, unsupported, corrupt, semantically invalid BOOL,
or over-limit files; other `IOException`s report file and publication failures. Invalid expected
digests, target schema, or target alias-value conflicts raise `IllegalArgumentException`. An
unbound target raises `IllegalStateException`. SHA-256 is an integrity and identity check, not
adversarial authentication or encryption. Limits bound encoded bytes, not peak JVM heap or Engine
workspace. The retained aggregate payload bound counts schema plus every path's canonical bytes,
including repeated aliases. The Engine materialization bound counts only unique state outputs;
neither bound includes transient workspace, detached source values, defensive copies, or peak
heap. The v1 format has no migration or source-alias metadata.

Model-only save, inspection, and load are available; the restored heap Tensors can be passed
to Engine through supported expressions for computation on the validated CPU path. This API does not persist
`TrainingState`, module modes, caller progress, or other state needed for exact training
continuation. A caller-coordinated [private training restore](training-api.md#caller-coordinated-in-memory-recovery)
can load a model artifact into a distinct, fully bound unpublished tree, but no durable
`TrainingCheckpoint` adapter or built-in cross-owner transaction exists.

The focused integration test saves the actual quiescent trained tree, then strict-loads and
directly re-saves its heap-backed replacement before any native parameter copy. Complete file
equality checks that intermediate round trip. Caller exclusion and live, accessible source
storage remain required throughout save; arbitrary Engine compositions must support the selected
movement and existing expression occurrences. No Engine/Prepare zero-node support is implied.

## Related API and architecture

- [Opaque artifact file API](checkpoint-artifact-api.md)
- [Checkpoint boundary explanation](../architecture/module-boundaries.md#extensionscheckpoint)
- [Public model checkpoint decision](../design/decisions/0034-public-model-checkpoint.md)

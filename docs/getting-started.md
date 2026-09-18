# Getting started as a contributor

## Outcome

This guide gets a new contributor from a fresh checkout to a verified build and one runnable
CPU-only Engine computation. The current public lifecycle is:

```text
Tensor expression -> compile -> prepare -> run -> materialize
meaning             recipe     reusable  leased  detached value
```

`Engine.compute(...)` performs those stages freshly in one synchronous convenience call. The
reusable lifecycle is covered in the linked user guides.

## Prerequisites

Synaptik requires JDK 26. Confirm the active JDK before importing or building the project:

```bash
java -version
```

Import the repository as a Gradle project in IntelliJ IDEA and select a Java 26 SDK. The project
uses the checked-in Gradle wrapper, so a separate Gradle installation is not required.

## Verify the checkout

Run these commands from the repository root:

```bash
./gradlew projects
./gradlew test
./gradlew build
```

A successful command ends with `BUILD SUCCESSFUL`. The build verifies the repository under the
active JDK. It does not imply that planned Metal, CUDA, mixed-backend, persistence, or training
capabilities are available.

## Run one CPU computation

In another repository module, add the current Engine and Model projects:

```kotlin
dependencies {
    implementation(project(":modules:engine"))
    implementation(project(":modules:model"))
}
```

The following complete example creates a caller-owned `FLOAT32` input, constructs a non-empty
`contiguous()` expression, computes it through the standard CPU composition, and reads the
detached result:

```java
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Optional;

HostTensorValue retained;
try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
    Shape shape = Shape.of(2);
    TensorDescriptor descriptor = new TensorDescriptor(
            DataType.FLOAT32,
            shape,
            Optional.of(LayoutDescriptor.contiguous(shape)),
            false);
    MemorySegment source = MemorySegment.ofArray(new float[] {1.5f, -2.25f});
    MemorySegment storage = arena.allocate(source.byteSize(), Float.BYTES);
    MemorySegment.copy(source, 0, storage, 0, source.byteSize());
    Tensor input = TensorFactory.create(
            descriptor,
            Optional.empty(),
            Optional.of(new MemorySegmentStorage(DataType.FLOAT32, 2, storage)));

    retained = engine.compute(input.contiguous(), 2L * Float.BYTES);
}

assert retained.bytes().getFloat(0) == 1.5f;
assert retained.bytes().getFloat(Float.BYTES) == -2.25f;
```

The input storage remains owned by the caller and must stay live through synchronous computation.
The `HostTensorValue` owns an immutable canonical copy, so it remains readable after both the
Engine and arena close. Every `compute(...)` call compiles and prepares afresh.

## Common setup problems

| Symptom | Likely cause | Correction |
|---|---|---|
| Gradle reports an unsupported Java version | The wrapper is running with a JDK older than 26. | Set the IDE Gradle JVM and `JAVA_HOME` to JDK 26, then rerun `java -version`. |
| `Engine.standard()` cannot prepare an expression | The current CPU composition requires one non-empty supported CPU partition with fully static compatible descriptors. | Start with the `contiguous()` example, then check the operation and descriptor constraints in the public API status. |
| A detached result exceeds the caller limit | The canonical payload is larger than `maximumTotalBytes`. | Increase the explicit bound after checking the expected output shape and data type. |
| A Metal, CUDA, or mixed-owner example fails | Those execution paths are not current public capabilities. | Use the fixed CPU composition and follow the roadmap for later backends. |

Java preview features are disabled by default. Incubator or preview APIs are configured only by
focused module tasks when stable Java 26 APIs are insufficient.

## Next reading

- [Compile graphs](user-guide/compiling-graphs.md) explains ordinary and advanced compilation.
- [Prepare execution](user-guide/preparing-execution.md) explains reusable prepared state.
- [Run models](user-guide/running-models.md) explains leases and detached values.
- [Public API status](api/public-api.md) records current limitations and complete examples.
- [Glossary](glossary.md) defines project-specific terms.

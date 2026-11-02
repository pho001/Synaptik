# Getting started

## Outcome

Build from a checkout or install a versioned local candidate, then run a CPU-only Engine
computation. The current public lifecycle is:

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

## Install a versioned local candidate

The owner selected **private/internal distribution** for the first production target. No public
project licence or redistribution grant is selected; do not publish these artifacts to public
registries. Current archives remain qualification candidates until the final all-PURE/source-matched
technical gate passes. No official signing or bundled third-party native implementation is implied.
Java CPU BF16/FP16 numerical arithmetic remains locked; raw storage, casts and the documented
exact/discrete domain remain separate. Metal requires the explicitly configured Apple-silicon
package and its documented supported operation/type occurrences.

### Produce and verify the archives

From the source root, supply the actual base revision and the existing verified Metal package:

```bash
./gradlew releaseDistribution \
  -PreleaseVersion=0.1.0 \
  -PreleaseBaseRevision=<actual-base-revision> \
  -PsynaptikMetalNativePackage=/absolute/path/to/macos-arm64
```

`releaseBaseRevision` is mandatory for source provenance; it has no built-in revision fallback.
The reviewed `gradle/release-source-paths.txt` selects the exact current working-tree bytes,
including accepted dirty source, wrapper, build scripts, tests and documentation. It excludes
historical evidence, build/cache/IDE files and secrets. Update that reviewed inventory when
accepted source files are added or removed; the build does not discover arbitrary local files.
`source-provenance.json` records the version, base revision and each selected path/SHA256, both
beside the archives and inside the extracted source root. A base revision alone is not the source.

Outputs live under `build/release/0.1.0/`:

| Archive | Contents |
|---|---|
| `synaptik-0.1.0-sources.zip` | Buildable selected source and provenance; Unix executable modes retained |
| `synaptik-0.1.0-repository.zip` | Standard local Maven repository, binary/source/Javadoc artifacts and third-party notices |
| `synaptik-0.1.0-bin.zip` | Java binaries, the same Maven repository, verified inner Metal ZIP and notices |

Verify `SHA256SUMS` from that output directory before extraction (`shasum -a 256 -c SHA256SUMS`
on macOS). Preserve executable file modes when extracting source; `ditto -x -k archive.zip target`
does so on macOS. An extracted source root builds without `.git`; ordinary `./gradlew test` does
not require release properties. Recreating a source archive still requires its recorded base
revision. The native package retains its separate backend-owned verification:
[Metal package instructions](../modules/backend/metal/macos-arm64/README.md).

### Consume only the installed repository

Extract the repository ZIP outside the checkout. In a separate Gradle project, use its absolute
`repository` directory and the ordinary Java toolchain:

```kotlin
plugins { java }
repositories { maven { url = uri("/absolute/path/to/repository") } }
dependencies { implementation("io.github.pho001.synaptik:nn:0.1.0") }
java { toolchain { languageVersion.set(JavaLanguageVersion.of(26)) } }
```

For Maven, put these sections inside the consumer's `pom.xml`; the standard compiler plugin must
support Java 26:

```xml
<properties><maven.compiler.release>26</maven.compiler.release></properties>
<repositories>
  <repository><id>synaptik-local</id><url>file:///absolute/path/to/repository</url></repository>
</repositories>
<dependencies>
  <dependency>
    <groupId>io.github.pho001.synaptik</groupId><artifactId>nn</artifactId><version>0.1.0</version>
  </dependency>
</dependencies>
<build><plugins><plugin>
  <groupId>org.apache.maven.plugins</groupId><artifactId>maven-compiler-plugin</artifactId>
  <version>3.14.0</version>
</plugin></plugins></build>
```

Choose `engine` for the low-level Engine API or `nn` for owned Models; add `text` only when using
the Text extension. Their published metadata supplies the supported dependency closure. Do not
add checkout JARs, internal modules, a fat JAR or a manual hidden-transitive classpath.
Use the complete [MLP, CNN and runtime-length RNN workflows](user-guide/running-models.md).
Launch Engine-based Java consumers with `--enable-native-access=ALL-UNNAMED`, including the
standard CPU composition's optional OpenBLAS probe. Metal also requires the installed verified
library's absolute path in MetalBackendConfiguration, with no discovery or automatic backend fallback. See the [production support matrix](backend-guide/production-support.md)
before selecting operations and dtypes.

## Verify the checkout

Run these commands from the repository root:

```bash
./gradlew projects
./gradlew test
./gradlew build
```

A successful command ends with `BUILD SUCCESSFUL`. The build verifies the repository under the
active JDK. It does not exercise the opt-in real-device Metal path, publish a release, or turn
planned CUDA and durable-persistence work into current capabilities.

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

public final class GettingStartedExample {
    public static void main(String[] args) {
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
    }
}
```

The input storage remains owned by the caller and must stay live through synchronous computation.
The `HostTensorValue` owns an immutable canonical copy, so it remains readable after both the
Engine and arena close. Every `compute(...)` call compiles and prepares afresh.

Save the complete Java block as `GettingStartedExample.java` and use the installed Engine runtime
classpath. Run with assertions enabled so the retained-value checks actually execute:

```bash
java --enable-native-access=ALL-UNNAMED -ea \
  --class-path "<installed-engine-runtime-classpath>" GettingStartedExample.java
```

## Common setup problems

| Symptom | Likely cause | Correction |
|---|---|---|
| Gradle reports an unsupported Java version | The wrapper is running with a JDK older than 26. | Set the IDE Gradle JVM and `JAVA_HOME` to JDK 26, then rerun `java -version`. |
| `Engine.standard()` cannot prepare an expression | The current CPU composition requires one non-empty supported CPU partition with fully static compatible descriptors. | Start with the `contiguous()` example, then check the operation and descriptor constraints in the public API status. |
| A detached result exceeds the caller limit | The canonical payload is larger than `maximumTotalBytes`. | Increase the explicit bound after checking the expected output shape and data type. |
| A Metal example fails | Real Metal execution requires Apple silicon, a built native bridge, explicit `MetalBackendConfiguration` plus `Engine.builder()` ownership, and a supported operation occurrence. | Build the native bridge, supply its absolute path, and keep the graph inside the documented Metal domain; there is no numerical-profile selector. |
| A CUDA example fails | CUDA has no current public lifecycle integration. | Use fixed CPU or explicitly registered CPU/Metal integrations. |
| A mixed CPU/Metal example fails | The required edge, descriptor, or host carrier is outside the exact all-seven-carrier rank-0..16 static canonical or positive-stride non-overlapping storage-layout transfer domain. | Keep cross-owner values inside the documented checked physical-span boundary; there is no conversion or fallback. |

Java preview features are disabled by default. Incubator or preview APIs are configured only by
focused module tasks when stable Java 26 APIs are insufficient.

## Next reading

- [Compile graphs](user-guide/compiling-graphs.md) explains ordinary and advanced compilation.
- [Prepare execution](user-guide/preparing-execution.md) explains reusable sessions and direct
  prepared-handle ownership.
- [Run models](user-guide/running-models.md) explains session reuse, result leases, and detached
  values.
- [Public API status](api/public-api.md) records current limitations and complete examples.
- [Glossary](glossary.md) defines project-specific terms.

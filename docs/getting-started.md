# Getting started as a contributor

## Outcome

This guide gets a new contributor from a fresh checkout to a verified build and a small runnable use of the model module. Synaptik is still under development: the model value types shown below exist today, while the public compile, prepare, and run workflow is planned but not yet implemented.

The mental model for the finished architecture is:

```text
compile              prepare                 run
meaning and owner -> executable state -> one invocation
```

Only the model foundations used to describe meaning are currently available. See the [roadmap](planning/roadmap.md) for the exact implementation frontier.

## Prerequisites

Synaptik requires JDK 26. Confirm the active JDK before importing or building the project:

```bash
java -version
```

Import the repository as a Gradle project in IntelliJ IDEA and select a Java 26 SDK. The project uses the checked-in Gradle wrapper, so a separate Gradle installation is not required.

## Verify the checkout

Run these commands from the repository root:

```bash
./gradlew projects
./gradlew test
./gradlew build
```

- `projects` confirms that Gradle can load the multi-module build and shows the available project paths.
- `test` executes the current unit-test suite.
- `build` compiles, tests, and packages every configured module.

A successful command ends with `BUILD SUCCESSFUL`. This proves that the checkout builds under the active JDK; it does not mean the planned compiler, runtime, or backends already exist.

## Try the implemented model API

The following Java snippet uses the current `modules:model` API. In another repository module's `build.gradle.kts`, add:

```kotlin
dependencies {
    implementation(project(":modules:model"))
}
```

`implementation` makes model types available to that module's main source set. The project path selects the local model module rather than a published artifact.

```java
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;

Shape shape = Shape.of(2, 3, 4);
long elements = shape.knownElementCount().orElseThrow();
LayoutDescriptor layout = LayoutDescriptor.contiguous(shape);
long firstAxisStride = layout.stride(0);
```

`Shape.of(2, 3, 4)` creates a rank-3 static shape. Its element count is `2 × 3 × 4 = 24`. A canonical row-major layout has strides `[12, 4, 1]`, so moving one step on the first axis advances `12` elements. The resulting values are therefore `elements == 24` and `firstAxisStride == 12`.

This example describes logical shape and layout only. It does not allocate tensor storage, compile a graph, or choose a backend.

## Common setup problems

| Symptom | Likely cause | Correction |
|---|---|---|
| Gradle reports an unsupported Java version | The wrapper is running with a JDK older than 26. | Set `JAVA_HOME` and the IDE Gradle JVM to JDK 26, then rerun `java -version`. |
| An import under `io.github.pho001.synaptik.model` is missing | Model contracts live in responsibility subpackages. | Use the package names in the [Tensor API reference](api/tensor-api.md). |
| A compile/prepare/run example does not compile | That public lifecycle is not implemented yet. | Treat lifecycle snippets in architecture documents as conceptual and follow the roadmap. |

Java preview features are disabled by default. Incubator or preview APIs are configured only by focused module tasks when the capability cannot be implemented through stable Java 26 APIs.

## Next reading

- [Tensor API](api/tensor-api.md) documents the implemented model contracts.
- [Architecture overview](architecture/overview.md) explains the planned system layers.
- [Repository layout](developer-guide/repository-layout.md) helps contributors find code and tests.
- [Glossary](glossary.md) defines project-specific terms.

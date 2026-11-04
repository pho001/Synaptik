# Coding rules

Use this page to distinguish Java [features](../glossary.md#term-software-feature) that are [ready](../glossary.md#term-ready) for ordinary use from features that
need a deliberately limited experiment. The **[Java Development Kit](../glossary.md#term-java-development-kit-jdk) ([JDK](../glossary.md#java-platform-and-build-tools))** provides the [compiler](../glossary.md#term-compiler)
and development tools; the **[Java Virtual Machine](../glossary.md#term-java-virtual-machine-jvm) ([JVM](../glossary.md#java-platform-and-build-tools))** executes the compiled program.

## Choose the stable language baseline

The contributor writes Java source; Gradle invokes the configured JDK compiler and
launches test JVMs. Compiler flags affect which source is accepted; runtime flags
affect the launched process. A feature can therefore compile successfully yet fail
when a consumer starts a JVM without its required options.

The project compiles and tests against Java 26. Production code may use stable Java 26 language features and APIs when they preserve the [module](../glossary.md#term-module) boundaries in [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md).

## Keep experimental activation scoped

[Preview features](../glossary.md#java-preview-incubator-modules-and-compiler-tiers) are not enabled globally. A task that requires a [preview feature](../glossary.md#term-preview-feature) must justify it, limit compiler and [runtime](../glossary.md#term-runtime) flags to the owning module, document portability and [lifecycle](../glossary.md#lifecycle) risks, and add focused [validation](../glossary.md#term-validation). Incubator modules follow the same task-scoped rule.

A **preview feature** is a Java language or platform feature offered for [evaluation](../glossary.md#term-evaluation) before its
design is final. An **[incubator module](../glossary.md#term-incubator-module)** distributes an experimental library [API](../glossary.md#term-application-programming-interface-api) separately.
Both can change between JDK [releases](../glossary.md#commit-release-distribution-log-and-seed-meanings), but enabling one does not enable the other. Here an
**API**, or [application programming interface](../glossary.md#api-spi-and-dto), is the set of types and operations a caller uses.

The [Vector API](../glossary.md#term-vector-api) remains an incubator module in Java 26. A future vectorization task may configure `jdk.incubator.vector` for the applicable [backend](../glossary.md#backend) module; the core project does not add that module until it is used. Incubator module [activation](../glossary.md#term-activation) uses module flags and must not be confused with the unrelated `--enable-preview` flag.

## Connect the change to its owner

Use the [repository map](repository-layout.md) to find the project receiving a
feature or flag, and [architecture tests](architecture-tests.md) to understand
which dependency boundaries are enforced. Do not add an experimental module to a
shared convention merely because one backend needs it: that would impose the
requirement on otherwise unrelated consumers.

All changes must also follow [`../../AGENTS.md`](../../AGENTS.md), including testing, documentation review, complete [Javadoc](../glossary.md#term-javadoc), package-structure [planning](../glossary.md#term-planning), and task isolation.

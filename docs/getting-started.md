# Getting started

## Outcome

[Run](glossary.md#term-run) the complete commented [CPU](glossary.md#term-cpu)/[FLOAT32](glossary.md#term-float32) learning project against installed [artifacts](glossary.md#term-artifact), then follow
it through training, save/close/load and reusable execution. Start with an application, not the
framework's implementation roadmap.

## Prerequisites

Use [JDK](glossary.md#term-java-development-kit-jdk) **26**, [Maven](glossary.md#term-maven) **3.9+**, and the supplied private file Maven repository for
`0.1.0-public-audit.20261104`. Confirm both Java and Maven use that JDK. The
[executable setup](examples/README.md) gives the exact commands, required repository URL, [JVM](glossary.md#term-java-virtual-machine-jvm)
flags, source and expected output. These are POSIX-shell commands; Windows needs its platform's
[classpath](glossary.md#term-classpath)/environment syntax.

No [GPU](glossary.md#term-gpu), data download, external [BLAS](glossary.md#term-blas) installation, preview flag or [Vector API](glossary.md#term-vector-api) flag is required.
[Engine](glossary.md#term-engine) applications enable [FFM](glossary.md#term-ffm) native access even for the standard [CPU](glossary.md#term-cpu) composition's optional
[OpenBLAS](glossary.md#term-openblas) probe; that permission does not select a native route.

## Install a versioned local candidate

The current candidate remains **private/internal**. No public licence/grant or upload has been
authorized. Obtain the [sealed](glossary.md#term-sealed) [artifacts](glossary.md#term-artifact) from their existing private [distribution](glossary.md#term-distribution); do not assume
the version is on [Maven](glossary.md#term-maven) Central or substitute another version silently.

### Produce and verify the archives

Application users do not need to rebuild the framework. If you were supplied [release](glossary.md#term-release) archives,
verify their accompanying `SHA256SUMS` before extraction and preserve the supplied [Maven](glossary.md#term-maven) repository
layout. The [release guide](developer-guide/release-process.md) owns source selection, [provenance](glossary.md#provenance),
package verification and production commands; the
[audit receipt](planning/public-release-audit-2026-11-04.md#result) identifies the qualified candidate.
Earlier `0.1.0` receipts do not qualify changed sources.

### Consume only the installed repository

Follow the [standalone NN-only consumer setup](examples/README.md#build-the-complete-project)<span data-glossary-navigation="true"> ([NN glossary](glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>.
Supply the absolute `file:///` URL of the extracted [Maven](glossary.md#term-maven) repository root. Published [metadata](glossary.md#term-metadata)
resolves its supported transitive dependencies; do not add checkout class directories, hidden
internal [modules](glossary.md#term-module) or a replacement [fat JAR](glossary.md#term-fat-jar).

## [Run](glossary.md#term-run) one [CPU](glossary.md#term-cpu) computation

Open [First model](tutorials/first-model.md). It starts from the same tiny [model](glossary.md#term-model) used in all later
chapters and explains the concrete input/[weight](glossary.md#term-weight) [shapes](glossary.md#shape), [parameter](glossary.md#term-parameter)-copy boundary and retained
result. The [complete canonical source](examples/learning-journey/src/main/java/io/github/pho001/synaptik/examples/LearningJourney.java)<span data-glossary-navigation="true"> ([canonical glossary](glossary.md#term-canonical-form))</span>
and [expected stdout](examples/learning-journey/expected-stdout.txt) live with the [Maven](glossary.md#term-maven) project;
there is no second source block to drift from the program.

Then continue through [all tutorial chapters](tutorials/index.md). Use [First terms](terms.md)
when a [tensor](glossary.md#tensor) or [ML](glossary.md#term-machine-learning-ml) word is unfamiliar, or go directly to [reference goals](reference/index.md)
if you already know what [API](glossary.md#term-application-programming-interface-api) task you need.

## Verify the checkout

Framework contributors use the checked-in [Gradle](glossary.md#term-gradle) wrapper with JDK26:

```sh
./gradlew test
./gradlew build
```

`BUILD SUCCESSFUL` verifies those configured tasks; it does not by itself qualify a real [Metal](glossary.md#term-metal)
[device](glossary.md#term-device), publish a [release](glossary.md#term-release), enable [CUDA](glossary.md#term-cuda) or certify every persisted [graph](glossary.md#graph)'s execution. The learning
consumer is independently built/[run](glossary.md#term-run) through [Maven](glossary.md#term-maven). See the [contributor area](developer-guide/index.md)
for source and native development.

## Common setup problems

| Symptom | Check and correction |
|---|---|
| [Maven](glossary.md#term-maven) cannot resolve the candidate | The supplied file URL must identify the Maven repository root and full exact-version closure |
| Unsupported [release](glossary.md#term-release)/class-file version | Select JDK26 in both Maven and the launching Java process |
| Restricted native-access warning | Use the [JVM](glossary.md#term-java-virtual-machine-jvm) flag from executable setup on each [run](glossary.md#term-run) |
| Same [element](glossary.md#term-element) count but input rejected | Match the complete [descriptor](glossary.md#term-descriptor), not just count; the tutorial's `[2,1]` is not `[1,2]` |
| A retained output becomes inaccessible | Distinguish a live result [lease](glossary.md#term-lease) from the already [detached](glossary.md#detached-model-results) [host](glossary.md#term-host) output; follow [ownership](explanation/ownership.md)<span data-glossary-navigation="true"> ([ownership glossary](glossary.md#term-ownership-meanings))</span> |
| Save fails | Use an existing writable parent and a deliberate regular-file destination; the example can replace it |
| [Metal](glossary.md#term-metal)/low-type variant rejects | Check the [exact support matrix](backend-guide/production-support.md)<span data-glossary-navigation="true"> ([support matrix glossary](glossary.md#term-review-matrix))</span><span data-glossary-navigation="true"> ([matrix glossary](glossary.md#term-review-matrix))</span>; [CPU](glossary.md#term-cpu) support is not Metal parity |

## Next reading

- [Train, save and reuse the same example](tutorials/index.md).
- [Solve a particular task](how-to/index.md).
- [Understand the whole framework](explanation/framework.md).
- [Find every package/native responsibility](explanation/packages/index.md).
- [Look up an API or generated Javadoc](reference/index.md)<span data-glossary-navigation="true"> ([API glossary](glossary.md#term-application-programming-interface-api); [Javadoc glossary](glossary.md#term-javadoc))</span>.

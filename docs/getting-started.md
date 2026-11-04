# Getting started

## Outcome

Run the complete commented CPU/FLOAT32 learning project against installed artifacts, then follow
it through training, save/close/load and reusable execution. Start with an application, not the
framework's implementation roadmap.

## Prerequisites

Use JDK **26**, Maven **3.9+**, and the supplied private file Maven repository for
`0.1.0-public-audit.20261104`. Confirm both Java and Maven use that JDK. The
[executable setup](examples/README.md) gives the exact commands, required repository URL, JVM
flags, source and expected output. These are POSIX-shell commands; Windows needs its platform's
classpath/environment syntax.

No GPU, data download, external BLAS installation, preview flag or Vector API flag is required.
Engine applications enable FFM native access even for the standard CPU composition's optional
OpenBLAS probe; that permission does not select a native route.

## Install a versioned local candidate

The current candidate remains **private/internal**. No public licence/grant or upload has been
authorized. Obtain the sealed artifacts from their existing private distribution; do not assume
the version is on Maven Central or substitute another version silently.

### Produce and verify the archives

Application users do not need to rebuild the framework. If you were supplied release archives,
verify their accompanying `SHA256SUMS` before extraction and preserve the supplied Maven repository
layout. The [release guide](developer-guide/release-process.md) owns source selection, provenance,
package verification and production commands; the
[audit receipt](planning/public-release-audit-2026-11-04.md#result) identifies the qualified candidate.
Earlier `0.1.0` receipts do not qualify changed sources.

### Consume only the installed repository

Follow the [standalone NN-only consumer setup](examples/README.md#build-the-complete-project).
Supply the absolute `file:///` URL of the extracted Maven repository root. Published metadata
resolves its supported transitive dependencies; do not add checkout class directories, hidden
internal modules or a replacement fat JAR.

## Run one CPU computation

Open [First model](tutorials/first-model.md). It starts from the same tiny model used in all later
chapters and explains the concrete input/weight shapes, parameter-copy boundary and retained
result. The [complete canonical source](examples/learning-journey/src/main/java/io/github/pho001/synaptik/examples/LearningJourney.java)
and [expected stdout](examples/learning-journey/expected-stdout.txt) live with the Maven project;
there is no second source block to drift from the program.

Then continue through [all tutorial chapters](tutorials/index.md). Use [First terms](terms.md)
when a tensor or ML word is unfamiliar, or go directly to [reference goals](reference/index.md)
if you already know what API task you need.

## Verify the checkout

Framework contributors use the checked-in Gradle wrapper with JDK26:

```sh
./gradlew test
./gradlew build
```

`BUILD SUCCESSFUL` verifies those configured tasks; it does not by itself qualify a real Metal
device, publish a release, enable CUDA or certify every persisted graph's execution. The learning
consumer is independently built/run through Maven. See the [contributor area](developer-guide/index.md)
for source and native development.

## Common setup problems

| Symptom | Check and correction |
|---|---|
| Maven cannot resolve the candidate | The supplied file URL must identify the Maven repository root and full exact-version closure |
| Unsupported release/class-file version | Select JDK26 in both Maven and the launching Java process |
| Restricted native-access warning | Use the JVM flag from executable setup on each run |
| Same element count but input rejected | Match the complete descriptor, not just count; the tutorial's `[2,1]` is not `[1,2]` |
| A retained output becomes inaccessible | Distinguish a live result lease from the already detached host output; follow [ownership](explanation/ownership.md) |
| Save fails | Use an existing writable parent and a deliberate regular-file destination; the example can replace it |
| Metal/low-type variant rejects | Check the [exact support matrix](backend-guide/production-support.md); CPU support is not Metal parity |

## Next reading

- [Train, save and reuse the same example](tutorials/index.md).
- [Solve a particular task](how-to/index.md).
- [Understand the whole framework](explanation/framework.md).
- [Find every package/native responsibility](explanation/packages/index.md).
- [Look up an API or generated Javadoc](reference/index.md).

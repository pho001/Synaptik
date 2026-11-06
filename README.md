# Synaptik

Build [tensor](docs/glossary.md#tensor) computations and [neural-network models](docs/glossary.md#neural-network-module-parameter-buffer-and-forward-context)
in Java. Tensor expressions describe **what to calculate**; Model or Engine turns that description
into an execution using the current inputs. Start with [CPU](docs/glossary.md#term-cpu).
Use [Metal](docs/glossary.md#term-metal) or mixed execution only within the actual program's
[supported operations, types and layouts](docs/backend-guide/production-support.md).

The documentation starts from ordinary Java knowledge.
[First terms](docs/terms.md) introduces the vocabulary in reading order; the
[central glossary](docs/glossary.md) explains definitions, prerequisites and easily confused meanings.

## Start with your goal

| You want to… | Start here |
|---|---|
| [Run](docs/glossary.md#term-run) your first [model](docs/glossary.md#term-model) and learn the required concepts | [The complete learning journey](docs/tutorials/index.md) |
| Use existing [ML](docs/glossary.md#term-machine-learning-ml) knowledge with Java [tensor](docs/glossary.md#tensor)/model APIs | [Task guides](docs/how-to/index.md) and [topic-sized reference](docs/reference/index.md) |
| Understand or contribute to the implementation | [One program through the framework](docs/explanation/framework.md), then the [package/native atlas](docs/explanation/packages/index.md) |

The learning project is one commented, runnable application: construct a model, understand
[shapes](docs/glossary.md#shape), calculate [gradients](docs/glossary.md#term-gradient), train,
evaluate, save/close/load, and reuse execution. Its
[setup and complete source](docs/examples/README.md) use public APIs from an independent
[Maven](docs/glossary.md#term-maven) project depending on the neural-network (`nn`) library—not checkout classes or a hidden dependency list.

In that program, **your Java application is the caller** of `model.compute(input)`; Model is
the called object. Model's owned parameter values are private working copies, while the
current input storage is only borrowed for that synchronous call. The returned host value is
an independent copy. The [first-model chapter](docs/tutorials/first-model.md#who-supplies-the-weight-and-who-executes-the-rule)
and [ownership walkthrough](docs/explanation/ownership.md) explain these different lifetimes.

## Current implementation status

[Tensor](docs/glossary.md#tensor) expressions describe mathematical meaning.
[Compilation](docs/glossary.md#compiler-engine-and-execution-stage-names) checks and organizes that
description for supported execution routes; preparation creates reusable execution resources.
Each run supplies current inputs and publishes results. Follow
[one complete program through these stages](docs/explanation/framework.md) before learning repository modules.

The [production support matrix](docs/backend-guide/production-support.md) is the entry point
for current operation, [data type](docs/glossary.md#data-type--datatype),
[layout](docs/glossary.md#layout) and [derivative](docs/glossary.md#term-derivative) qualification.
CPU support is not a Metal capability promise. [CUDA](docs/glossary.md#term-cuda),
[ONNX](docs/glossary.md#term-onnx) markers are not usable products. The separate
[real CLI application](docs/how-to/train-with-cli.md) trains numeric CSV, saves/loads Models,
predicts actual values and displays live training state; its supported geometry is explicit.
[Saving a Model](docs/tutorials/save-close-load.md) stores admitted source and current values,
not prepared native executables, optimizer state or the application's complete training continuation.

[Distribution](docs/glossary.md#term-distribution) remains **private/internal**: no public project licence or redistribution grant has
been selected, and no public upload is authorized. The sealed
`0.1.0-public-audit.20261104` candidate has [source-matched audit evidence](docs/planning/public-release-audit-2026-11-04.md#result).
The learning-project qualification remains pinned to those sealed artifacts. The current
checkout's CLI image and [bounded autotuning repairs](docs/planning/autotuning-audit.md)
are separate work; they do not silently replace or requalify that release.

## Prerequisites

- [JDK](docs/glossary.md#term-java-development-kit-jdk) 26.
- [Maven](docs/glossary.md#term-maven) 3.9+ for the standalone learning project, plus the supplied private file Maven repository.
- The checked-in [Gradle](docs/glossary.md#term-gradle) wrapper for framework development; no separate Gradle installation.

Follow [setup](docs/getting-started.md), then [first model](docs/tutorials/first-model.md)<span data-glossary-navigation="true"> ([model glossary](docs/glossary.md#term-model))</span>.
The default [CPU](docs/glossary.md#term-cpu) route needs no [GPU](docs/glossary.md#term-gpu) or third-party [native library](docs/glossary.md#term-native-library). [Engine](docs/glossary.md#term-engine) consumers enable [FFM](docs/glossary.md#term-ffm)
native access as shown in the runnable setup; explicit [Metal](docs/glossary.md#term-metal) requires its separately verified package.

## [Gradle](docs/glossary.md#term-gradle) commands

From the framework source root:

```sh
# Check the framework source; use the separate Maven setup for the learning application.
./gradlew test
./gradlew build
# Generate the foundational Tensor API reference.
./gradlew :modules:model:javadoc
```

These are development commands, not a substitute for the tutorial's external-consumer setup or
real-[device](docs/glossary.md#term-device) [qualification](docs/glossary.md#term-qualification). [Java preview features](docs/glossary.md#java-preview-incubator-modules-and-compiler-tiers) are not enabled globally; the learning project
needs no preview or incubating [Vector API](docs/glossary.md#term-vector-api) flag.

## Contributor and maintenance navigation

[Documentation home](docs/index.md) separates learning, tasks, explanation and reference.
[Contributor guides](docs/developer-guide/index.md) lead to source navigation, debugging and [release](docs/glossary.md#term-release)
mechanics. [ARCHITECTURE.md](ARCHITECTURE.md) is the sole architecture authority index;
[AGENTS.md](AGENTS.md) defines automated-contributor workflow.
[Planning](docs/planning/README.md)<span data-glossary-navigation="true"> ([Planning glossary](docs/glossary.md#term-planning))</span>, [design history](docs/design/README.md) and audit receipts are
maintenance evidence, not prerequisites for a first [prediction](docs/glossary.md#term-prediction).

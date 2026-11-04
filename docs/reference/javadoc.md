# Generated [Javadoc](../glossary.md#term-javadoc): exact [signatures](../glossary.md#term-signature) and source contracts

The topic pages explain relationships and examples; generated Javadoc supplies exact Java signatures, [parameter](../glossary.md#term-parameter) constraints, returns and failures. **Every local generated-output link on this page is build-required.** Generated HTML is not committed, and a Markdown browser cannot navigate into a [JAR](../glossary.md#term-jar) directly. Obtain it by either route below. These instructions do not publish [artifacts](../glossary.md#term-artifact), alter licensing, or modify the sealed `0.1.0-public-audit.20261104` candidate.

[Reference goals](index.md) · [Framework package atlas](../explanation/packages/index.md)

Javadoc is generated [API](../glossary.md#term-application-programming-interface-api) documentation: it turns Java declarations and their documentation
comments into browsable pages. A signature tells you a method's name, [argument](../glossary.md#term-argument) types and return
type; its contract also explains valid arguments, [lifetime](../glossary.md#term-lifetime) and failure behavior. A JAR is an
archive file, so the supplied documentation must be extracted before a browser can follow it.

## Read the supplied candidate without rebuilding

Prerequisites: [JDK](../glossary.md#term-java-development-kit-jdk) 26's `jar` tool, the existing private [release](../glossary.md#term-release) repository under `build/release/0.1.0-public-audit.20261104/repository`, and a writable temporary directory. [Run](../glossary.md#term-run) from the repository root. This extracts only the 18 existing [Javadoc](../glossary.md#term-javadoc) archives into a new local temporary directory; it does not write into the supplied repository or replace candidate [artifacts](../glossary.md#term-artifact).

```sh
# Extract the exact supplied archives into a new temporary tree rather than modifying candidate artifacts.
version=0.1.0-public-audit.20261104
repository="$PWD/build/release/$version/repository/io/github/pho001/synaptik"
output=$(mktemp -d "${TMPDIR:-/tmp}/synaptik-javadoc.XXXXXX")
for artifact in trace backend-contract model config planning compiler runtime prepare engine openblas-provider cpu metal data text nn training checkpoint tuning; do
    destination="$output/$artifact"
    mkdir -p "$destination"
    archive="$repository/$artifact/$version/$artifact-$version-javadoc.jar"
    jar --extract --file "$archive" --dir "$destination" || exit 1
done
printf 'Extracted Javadoc directory: %s\n' "$output"
```

Open `index.html` inside the desired artifact directory printed by the command. For example, [Model](../glossary.md#term-model)'s `io/github/pho001/synaptik/model/tensor/Tensor.html` documents [Tensor](../glossary.md#tensor). [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context)'s corresponding entry is `io/github/pho001/synaptik/nn/module/Model.html`; [Engine](../glossary.md#term-engine)'s is `io/github/pho001/synaptik/engine/Engine.html`. Extraction of all 18 supplied archives was exercised during documentation [qualification](../glossary.md#term-qualification). If a supplied [JAR](../glossary.md#term-jar) is missing, obtain that exact private candidate first rather than silently replacing it with another version. Cross-[module](../glossary.md#term-module) links in module-generated Javadoc may require returning to the neighboring artifact's index.

## Generate HTML from this checkout

Prerequisites: the repository's supported [JDK](../glossary.md#term-java-development-kit-jdk) 26/[Gradle](../glossary.md#term-gradle) environment and any ordinary [dependency](../glossary.md#term-dependency) access required by its build. From the repository root, request only the [modules](../glossary.md#term-module) you need, for example:

```sh
# Generate only the requested source-checkout documentation; this does not reproduce sealed candidate evidence.
./gradlew :modules:model:javadoc :modules:engine:javadoc :extensions:nn:javadoc :extensions:training:javadoc :extensions:checkpoint:javadoc
```

Expected output: each requested module's `build/docs/javadoc/index.html` and package/type pages. The links below resolve only after those tasks produce HTML. Generating documentation from a later [source revision](../glossary.md#term-source-revision) is not proof that it matches the sealed candidate; use its supplied JARs when exact candidate documentation matters. Production [Javadoc](../glossary.md#term-javadoc) was not regenerated for this editorial change because its Java/Javadoc input bytes are unchanged. The standalone learning source was separately checked with `javadoc -Xdoclint:all`.

## All 18 supported [artifact](../glossary.md#term-artifact) indexes

| Artifact | [Gradle](../glossary.md#term-gradle) task | Local output, build required |
|---|---|---|
| `backend-contract` | `:modules:backend-contract:javadoc` | [Generated index](../../modules/backend-contract/build/docs/javadoc/index.html) |
| [`checkpoint`](../glossary.md#term-checkpoint) | `:extensions:checkpoint:javadoc` | [Generated index](../../modules/checkpoint/build/docs/javadoc/index.html) |
| [`compiler`](../glossary.md#term-compiler) | `:modules:compiler:javadoc` | [Generated index](../../modules/compiler/build/docs/javadoc/index.html) |
| `config` | `:modules:config:javadoc` | [Generated index](../../modules/config/build/docs/javadoc/index.html) |
| `cpu` | `:backends:cpu:javadoc` | [Generated index](../../modules/backend/cpu/build/docs/javadoc/index.html) |
| `data` | `:extensions:data:javadoc` | [Generated index](../../extensions/data/build/docs/javadoc/index.html) |
| [`engine`](../glossary.md#term-engine) | `:modules:engine:javadoc` | [Generated index](../../modules/engine/build/docs/javadoc/index.html) |
| [`metal`](../glossary.md#term-metal) | `:backends:metal:javadoc` | [Generated index](../../modules/backend/metal/build/docs/javadoc/index.html) |
| [`model`](../glossary.md#term-model) | `:modules:model:javadoc` | [Generated index](../../modules/model/build/docs/javadoc/index.html) |
| `nn` | `:extensions:nn:javadoc` | [Generated index](../../extensions/nn/build/docs/javadoc/index.html) |
| `openblas-provider` | `:backends:openblas-provider:javadoc` | [Generated index](../../modules/backend/openblas-provider/build/docs/javadoc/index.html) |
| [`planning`](../glossary.md#term-planning) | `:modules:planning:javadoc` | [Generated index](../../modules/planning/build/docs/javadoc/index.html) |
| [`prepare`](../glossary.md#prepare) | `:modules:prepare:javadoc` | [Generated index](../../modules/prepare/build/docs/javadoc/index.html) |
| [`runtime`](../glossary.md#term-runtime) | `:modules:runtime:javadoc` | [Generated index](../../modules/runtime/build/docs/javadoc/index.html) |
| `text` | `:extensions:text:javadoc` | [Generated index](../../extensions/text/build/docs/javadoc/index.html) |
| [`trace`](../glossary.md#trace) | `:modules:trace:javadoc` | [Generated index](../../modules/trace/build/docs/javadoc/index.html) |
| `training` | `:extensions:training:javadoc` | [Generated index](../../modules/training/build/docs/javadoc/index.html) |
| `tuning` | `:tools:tuning:javadoc` | [Generated index](../../tools/tuning/build/docs/javadoc/index.html) |

[CUDA](../glossary.md#term-cuda), [ONNX](../glossary.md#term-onnx), [CLI](../glossary.md#term-cli) and benchmarks are not additional supported publications in this candidate. Their source/project presence is not an instruction to invent matching [Javadoc](../glossary.md#term-javadoc) artifacts.

## Common application entry points

These links are also **build-required**, using the checkout's generated locations rather than the temporary extraction directory:

- [Tensor](../../modules/model/build/docs/javadoc/io/github/pho001/synaptik/model/tensor/Tensor.html): logical [identity](../glossary.md#term-identity-meanings), [descriptors](../glossary.md#term-descriptor) and expression construction.
- [Engine](../../modules/engine/build/docs/javadoc/io/github/pho001/synaptik/engine/Engine.html): [compile](../glossary.md#compile)/[prepare](../glossary.md#prepare)/[run](../glossary.md#term-run) and explicit [backend ownership](../glossary.md#backend-ownership).
- [Model](../../extensions/nn/build/docs/javadoc/io/github/pho001/synaptik/nn/module/Model.html): owned compute, training and [persistence](../glossary.md#term-persistence) convenience.
- [ModelBuilder](../../extensions/nn/build/docs/javadoc/io/github/pho001/synaptik/nn/module/ModelBuilder.html): closed fixed-[signature](../glossary.md#term-signature) recipes and supplied-state acquisition.
- [TrainingSession](../../modules/training/build/docs/javadoc/io/github/pho001/synaptik/training/TrainingSession.html): live [parameter](../glossary.md#term-parameter) [admission](../glossary.md#term-admission) and update transactions.
- [GraphCheckpoint](../../modules/checkpoint/build/docs/javadoc/io/github/pho001/synaptik/checkpoint/graph/GraphCheckpoint.html): cold source copying, rebinding and bounded wire persistence.

A Java-public class inside an internal package is not automatically supported application [API](../glossary.md#term-application-programming-interface-api). Use the [package atlas](../explanation/packages/index.md) and topic boundaries to distinguish application facades from inward service-provider collaborations.

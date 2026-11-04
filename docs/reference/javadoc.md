# Generated Javadoc: exact signatures and source contracts

The topic pages explain relationships and examples; generated Javadoc supplies exact Java signatures, parameter constraints, returns and failures. **Every local generated-output link on this page is build-required.** Generated HTML is not committed, and a Markdown browser cannot navigate into a JAR directly. Obtain it by either route below. These instructions do not publish artifacts, alter licensing, or modify the sealed `0.1.0-public-audit.20261104` candidate.

[Reference goals](index.md) · [Framework package atlas](../explanation/packages/index.md)

## Read the supplied candidate without rebuilding

Prerequisites: JDK 26's `jar` tool, the existing private release repository under `build/release/0.1.0-public-audit.20261104/repository`, and a writable temporary directory. Run from the repository root. This extracts only the 18 existing Javadoc archives into a new local temporary directory; it does not write into the supplied repository or replace candidate artifacts.

```sh
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

Open `index.html` inside the desired artifact directory printed by the command. For example, Model's `io/github/pho001/synaptik/model/tensor/Tensor.html` documents Tensor. NN's corresponding entry is `io/github/pho001/synaptik/nn/module/Model.html`; Engine's is `io/github/pho001/synaptik/engine/Engine.html`. Extraction of all 18 supplied archives was exercised during documentation qualification. If a supplied JAR is missing, obtain that exact private candidate first rather than silently replacing it with another version. Cross-module links in module-generated Javadoc may require returning to the neighboring artifact's index.

## Generate HTML from this checkout

Prerequisites: the repository's supported JDK 26/Gradle environment and any ordinary dependency access required by its build. From the repository root, request only the modules you need, for example:

```sh
./gradlew :modules:model:javadoc :modules:engine:javadoc :extensions:nn:javadoc :extensions:training:javadoc :extensions:checkpoint:javadoc
```

Expected output: each requested module's `build/docs/javadoc/index.html` and package/type pages. The links below resolve only after those tasks produce HTML. Generating documentation from a later source revision is not proof that it matches the sealed candidate; use its supplied JARs when exact candidate documentation matters. Production Javadoc was not regenerated for this editorial change because its Java/Javadoc input bytes are unchanged. The standalone learning source was separately checked with `javadoc -Xdoclint:all`.

## All 18 supported artifact indexes

| Artifact | Gradle task | Local output, build required |
|---|---|---|
| `backend-contract` | `:modules:backend-contract:javadoc` | [Generated index](../../modules/backend-contract/build/docs/javadoc/index.html) |
| `checkpoint` | `:extensions:checkpoint:javadoc` | [Generated index](../../modules/checkpoint/build/docs/javadoc/index.html) |
| `compiler` | `:modules:compiler:javadoc` | [Generated index](../../modules/compiler/build/docs/javadoc/index.html) |
| `config` | `:modules:config:javadoc` | [Generated index](../../modules/config/build/docs/javadoc/index.html) |
| `cpu` | `:backends:cpu:javadoc` | [Generated index](../../modules/backend/cpu/build/docs/javadoc/index.html) |
| `data` | `:extensions:data:javadoc` | [Generated index](../../extensions/data/build/docs/javadoc/index.html) |
| `engine` | `:modules:engine:javadoc` | [Generated index](../../modules/engine/build/docs/javadoc/index.html) |
| `metal` | `:backends:metal:javadoc` | [Generated index](../../modules/backend/metal/build/docs/javadoc/index.html) |
| `model` | `:modules:model:javadoc` | [Generated index](../../modules/model/build/docs/javadoc/index.html) |
| `nn` | `:extensions:nn:javadoc` | [Generated index](../../extensions/nn/build/docs/javadoc/index.html) |
| `openblas-provider` | `:backends:openblas-provider:javadoc` | [Generated index](../../modules/backend/openblas-provider/build/docs/javadoc/index.html) |
| `planning` | `:modules:planning:javadoc` | [Generated index](../../modules/planning/build/docs/javadoc/index.html) |
| `prepare` | `:modules:prepare:javadoc` | [Generated index](../../modules/prepare/build/docs/javadoc/index.html) |
| `runtime` | `:modules:runtime:javadoc` | [Generated index](../../modules/runtime/build/docs/javadoc/index.html) |
| `text` | `:extensions:text:javadoc` | [Generated index](../../extensions/text/build/docs/javadoc/index.html) |
| `trace` | `:modules:trace:javadoc` | [Generated index](../../modules/trace/build/docs/javadoc/index.html) |
| `training` | `:extensions:training:javadoc` | [Generated index](../../modules/training/build/docs/javadoc/index.html) |
| `tuning` | `:tools:tuning:javadoc` | [Generated index](../../tools/tuning/build/docs/javadoc/index.html) |

CUDA, ONNX, CLI and benchmarks are not additional supported publications in this candidate. Their source/project presence is not an instruction to invent matching Javadoc artifacts.

## Common application entry points

These links are also **build-required**, using the checkout's generated locations rather than the temporary extraction directory:

- [Tensor](../../modules/model/build/docs/javadoc/io/github/pho001/synaptik/model/tensor/Tensor.html): logical identity, descriptors and expression construction.
- [Engine](../../modules/engine/build/docs/javadoc/io/github/pho001/synaptik/engine/Engine.html): compile/prepare/run and explicit backend ownership.
- [Model](../../extensions/nn/build/docs/javadoc/io/github/pho001/synaptik/nn/module/Model.html): owned compute, training and persistence convenience.
- [ModelBuilder](../../extensions/nn/build/docs/javadoc/io/github/pho001/synaptik/nn/module/ModelBuilder.html): closed fixed-signature recipes and supplied-state acquisition.
- [TrainingSession](../../modules/training/build/docs/javadoc/io/github/pho001/synaptik/training/TrainingSession.html): live parameter admission and update transactions.
- [GraphCheckpoint](../../modules/checkpoint/build/docs/javadoc/io/github/pho001/synaptik/checkpoint/graph/GraphCheckpoint.html): cold source copying, rebinding and bounded wire persistence.

A Java-public class inside an internal package is not automatically supported application API. Use the [package atlas](../explanation/packages/index.md) and topic boundaries to distinguish application facades from inward service-provider collaborations.

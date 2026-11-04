# Repository layout

Use this map to find the [owner](../glossary.md#term-owner) of a change before choosing files. Three names can describe the
same component without being interchangeable: a **directory** is its location on disk, a
**[Gradle](../glossary.md#term-gradle) project path** identifies it to the build tool, and a **Java package** groups types under
an import namespace. A **[module](../glossary.md#term-module) boundary** describes responsibility and permitted dependencies;
moving a directory does not by itself change that boundary.

The repository is organized around the module boundaries defined by [`ARCHITECTURE.md`](../../ARCHITECTURE.md):

- `modules/` contains [model](../glossary.md#term-model), [compiler](../glossary.md#term-compiler), [planning](../glossary.md#term-planning), [prepare](../glossary.md#prepare), [runtime](../glossary.md#term-runtime), [engine](../glossary.md#term-engine), generic training, configuration, [backend](../glossary.md#backend) contracts, tracing, and [concrete backend](../glossary.md#concrete-backend) projects;
- `modules/backend/` groups the separate [CPU](../glossary.md#term-cpu), [Metal](../glossary.md#term-metal), [CUDA](../glossary.md#term-cuda) scaffold, and [OpenBLAS](../glossary.md#term-openblas) provider projects;
- `extensions/` contains NN composition (including old NN checkpoint readers), ONNX integration,
  Data host preparation, and frozen Text tokenization/adaptation;<span data-glossary-navigation="true"> ([checkpoint glossary](../glossary.md#term-checkpoint); [ONNX glossary](../glossary.md#term-onnx); [tokenization glossary](../glossary.md#frozen-whitespace-tokenizer))</span>
- `tools/` contains tuning, benchmark, and command-line tooling;
- `modules/backend/metal/macos-arm64/` contains Metal's platform-specific native integration;
- `testing/` contains architecture, backend-[conformance](../glossary.md#term-conformance), and [integration test](../glossary.md#term-integration-test) modules; and
- `docs/` contains explanatory documentation and working implementation plans.

Implementation plans live under [`docs/planning/`](../planning/README.md). They coordinate non-trivial work but do not override the root [architecture contract](../glossary.md#architecture-contract). Production Java packages use the `io.github.pho001.synaptik.*` namespace.

In this inventory, **native integration** means code connecting Java to a platform library outside
the [JVM](../glossary.md#term-java-virtual-machine-jvm). **Conformance tests** check required shared behavior; **integration tests** check that
components work together. A **scaffold** marks incomplete infrastructure, not an implemented or
qualified backend. The path-migration tables below explain why a current directory and a retained
historical command can legitimately use different names.

Data is registered at `extensions/data`, using the default Gradle path `:extensions:data`
and namespace `io.github.pho001.synaptik.data`, exporting only Model through
`api(project(":modules:model"))`. The [Data sequence batching API](../reference/data/sequences.md)<span data-glossary-navigation="true"> ([API glossary](../glossary.md#term-application-programming-interface-api))</span> provides
copied [host](../glossary.md#term-host) counts, explicit [padding](../glossary.md#padding) side/[axis](../glossary.md#term-axis) order, a [borrowed](../glossary.md#ownership-borrowing-and-lease) [Tensor](../glossary.md#tensor) [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) record and an
[INT64](../glossary.md#data-type--datatype) host-array batcher. [Metadata](../glossary.md#term-metadata) construction does not allocate values; integral batching does.
Neither executes models or binds runtime length inputs.

Text is registered at `extensions/text`, `:extensions:text`, namespace
`io.github.pho001.synaptik.text`. Its flat package contains `WhitespaceTokenizer` and
`TextSequenceBatcher`, with a nested configuration Builder. Its sole edge is api(Data), with
Model transitive. The [Text API](../reference/text/tokenization.md) connects frozen [token](../glossary.md#term-token) IDs to Data padding;
the integration-test project separately composes [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Engine. Vision remains absent and separately gated.

## [Backend](../glossary.md#backend) path migration and historical navigation

Engine0020 changes physical placement only. The table translates original source paths to current
locations; the [Gradle](../glossary.md#term-gradle) project identifiers and Java packages do not change.

| Original source path | Current location | Gradle [identity](../glossary.md#term-identity-meanings) |
|---|---|---|
| `backends/cpu` | [CPU](../../modules/backend/cpu)<span data-glossary-navigation="true"> ([CPU glossary](../glossary.md#term-cpu))</span> | `:backends:cpu` |
| `backends/metal` | [Metal](../../modules/backend/metal) | `:backends:metal` |
| `backends/cuda` | [CUDA scaffold](../../modules/backend/cuda/README.md)<span data-glossary-navigation="true"> ([CUDA glossary](../glossary.md#term-cuda))</span> | `:backends:cuda` |
| `backends/openblas-provider` | [OpenBLAS provider](../../modules/backend/openblas-provider)<span data-glossary-navigation="true"> ([OpenBLAS glossary](../glossary.md#term-openblas))</span> | `:backends:openblas-provider` |
| `native/metal-macos-arm64` | [Native build and ABI guide](../../modules/backend/metal/macos-arm64/README.md)<span data-glossary-navigation="true"> ([ABI glossary](../glossary.md#term-application-binary-interface-abi))</span> | Part of [Metal](../glossary.md#term-metal) |
| `native/cuda/README.md` | [Merged native scaffold](../../modules/backend/cuda/README.md#native-support-scaffold) | No executable native project |

For example, from the repository root, `./gradlew :backends:metal:test` still selects the Metal
project, now physically under `modules/backend/metal`. Settings explicitly map both the parent
`:backends` and its four children; leaving an old directory behind would hide a missing mapping.
The physical group is not the separate backend-neutral `modules/backend-contract` project and
does not change which [modules](../glossary.md#term-module) may depend on [Engine](../glossary.md#term-engine).

Source-side documentation remains under `docs/`: original links to
[CPU plans](../planning/backends/cpu/master-plan.md),
[Metal plans](../planning/backends/metal/master-plan.md), and completed task/report files retain
their locations. Historical reports, commands, source manifests, and [certificate](../glossary.md#formal-proof-certificate-and-trusted-assumptions) [hashes](../glossary.md#term-hash) describe
their original revisions, not fresh [validation](../glossary.md#term-validation). Their old source paths translate through the table
above; old checkout-relative source links have no filesystem redirect. Use current guide links for
live commands, or inspect the original bytes with, for example,
`git show 9e7e8aa15e0833a1bc91263a7add99a80d88cd92:backends/metal/build.gradle.kts`.
Do not rewrite historical outcomes or regenerate certificates merely to change their paths.

## Training path migration

Training0007 moves only Training's `build.gradle.kts` and `src/` from `extensions/training` to
`modules/training`; generated build output is not relocated. Settings preserve the project
`:extensions:training`, and all Java types remain in `io.github.pho001.synaptik.training`.
For example, `./gradlew :extensions:training:test` now tests sources under `modules/training/src`.
The relocation accompanies generic live [parameter](../glossary.md#term-parameter) [admission](../glossary.md#term-admission): Training has [Model](../glossary.md#term-model)/[Engine](../glossary.md#term-engine) [API](../glossary.md#term-application-programming-interface-api) edges
and no [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) edge; NN exposes Model/Training and owns its [Module](../glossary.md#term-module) adapter. This is a role-specific
Engine exception, not permission for other inward modules to depend on Engine.

Training plans remain under [the existing planning directory](../planning/extensions/training/master-plan.md)<span data-glossary-navigation="true"> ([planning glossary](../glossary.md#term-planning))</span>.
Completed task evidence retains its original paths and commands; use `modules/training` for
current source navigation, without rewriting historical completion records.

Generic [Checkpoint](../glossary.md#term-checkpoint) is physically `modules/checkpoint`, with unchanged [Gradle](../glossary.md#term-gradle) project [identity](../glossary.md#term-identity-meanings)
`:extensions:checkpoint` through `settings.gradle.kts`. Its six [opaque](../glossary.md#term-opaque) [artifact](../glossary.md#term-artifact) files keep their
packages; the nine old [NN model](../glossary.md#model-module-buffer-and-scan-distinctions) files now belong to `extensions/nn` under `nn.checkpoint`.
For example, a generic artifact user keeps its [dependency](../glossary.md#term-dependency)/import, while a ModelCheckpoint user
adds an explicit NN dependency and changes `checkpoint.model` imports to `nn.checkpoint`; no
compatibility forwarding package is retained. See the [graph API](../reference/checkpoint/graph.md)<span data-glossary-navigation="true"> ([graph glossary](../glossary.md#graph))</span>.

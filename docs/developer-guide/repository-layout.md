# Repository layout

The repository is organized around the module boundaries defined by [`ARCHITECTURE.md`](../../ARCHITECTURE.md):

- `modules/` contains model, compiler, planning, prepare, runtime, engine, generic training, configuration, backend contracts, tracing, and concrete backend projects;
- `modules/backend/` groups the separate CPU, Metal, CUDA scaffold, and OpenBLAS provider projects;
- `extensions/` contains NN composition, Checkpoint, and ONNX integration;
- `tools/` contains tuning, benchmark, and command-line tooling;
- `modules/backend/metal/macos-arm64/` contains Metal's platform-specific native integration;
- `testing/` contains architecture, backend-conformance, and integration test modules; and
- `docs/` contains explanatory documentation and working implementation plans.

Implementation plans live under [`docs/planning/`](../planning/README.md). They coordinate non-trivial work but do not override the root architecture contract. Production Java packages use the `io.github.pho001.synaptik.*` namespace.

## Backend path migration and historical navigation

Engine0020 changes physical placement only. The table translates original source paths to current
locations; the Gradle project identifiers and Java packages do not change.

| Original source path | Current location | Gradle identity |
|---|---|---|
| `backends/cpu` | [CPU](../../modules/backend/cpu) | `:backends:cpu` |
| `backends/metal` | [Metal](../../modules/backend/metal) | `:backends:metal` |
| `backends/cuda` | [CUDA scaffold](../../modules/backend/cuda/README.md) | `:backends:cuda` |
| `backends/openblas-provider` | [OpenBLAS provider](../../modules/backend/openblas-provider) | `:backends:openblas-provider` |
| `native/metal-macos-arm64` | [Native build and ABI guide](../../modules/backend/metal/macos-arm64/README.md) | Part of Metal |
| `native/cuda/README.md` | [Merged native scaffold](../../modules/backend/cuda/README.md#native-support-scaffold) | No executable native project |

For example, from the repository root, `./gradlew :backends:metal:test` still selects the Metal
project, now physically under `modules/backend/metal`. Settings explicitly map both the parent
`:backends` and its four children; leaving an old directory behind would hide a missing mapping.
The physical group is not the separate backend-neutral `modules/backend-contract` project and
does not change which modules may depend on Engine.

Source-side documentation remains under `docs/`: original links to
[CPU plans](../planning/backends/cpu/master-plan.md),
[Metal plans](../planning/backends/metal/master-plan.md), and completed task/report files retain
their locations. Historical reports, commands, source manifests, and certificate hashes describe
their original revisions, not fresh validation. Their old source paths translate through the table
above; old checkout-relative source links have no filesystem redirect. Use current guide links for
live commands, or inspect the original bytes with, for example,
`git show 9e7e8aa15e0833a1bc91263a7add99a80d88cd92:backends/metal/build.gradle.kts`.
Do not rewrite historical outcomes or regenerate certificates merely to change their paths.

## Training path migration

Training0007 moves only Training's `build.gradle.kts` and `src/` from `extensions/training` to
`modules/training`; generated build output is not relocated. Settings preserve the project
`:extensions:training`, and all Java types remain in `io.github.pho001.synaptik.training`.
For example, `./gradlew :extensions:training:test` now tests sources under `modules/training/src`.
The relocation accompanies generic live parameter admission: Training has Model/Engine API edges
and no NN edge; NN exposes Model/Training and owns its Module adapter. This is a role-specific
Engine exception, not permission for other inward modules to depend on Engine.

Training plans remain under [the existing planning directory](../planning/extensions/training/master-plan.md).
Completed task evidence retains its original paths and commands; use `modules/training` for
current source navigation, without rewriting historical completion records.

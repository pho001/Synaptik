rootProject.name = "synaptik"

include(
    ":modules:trace",
    ":modules:backend-contract",
    ":modules:model",
    ":modules:config",
    ":modules:planning",
    ":modules:compiler",
    ":modules:runtime",
    ":modules:prepare",
    ":modules:engine",
    ":backends:openblas-provider",
    ":backends:cpu",
    ":backends:metal",
    ":backends:cuda",
    ":extensions:nn",
    ":extensions:training",
    ":extensions:checkpoint",
    ":extensions:onnx",
    ":tools:tuning",
    ":tools:benchmarks",
    ":tools:cli",
    ":testing:architecture-tests",
    ":testing:backend-conformance",
    ":testing:integration-tests",
)

// Preserve Training's project and package identities after its NN-independent relocation.
project(":extensions:training").projectDir = file("modules/training")

// Preserve Checkpoint's public project identity after the generic/NN ownership split.
project(":extensions:checkpoint").projectDir = file("modules/checkpoint")

// Preserve backend project identities independently of their physical source directories.
project(":backends").projectDir = file("modules/backend")
project(":backends:cpu").projectDir = file("modules/backend/cpu")
project(":backends:metal").projectDir = file("modules/backend/metal")
project(":backends:cuda").projectDir = file("modules/backend/cuda")
project(":backends:openblas-provider").projectDir = file("modules/backend/openblas-provider")

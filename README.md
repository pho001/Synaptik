# Synaptik

Synaptik is a modular Java tensor and neural-network runtime for compiling, preparing and
executing computational graphs across multiple backends. It provides a runnable public lifecycle
with fixed CPU convenience and explicit single- or bounded mixed-owner CPU/Metal composition,
plus backend-independent Tensor and compiler capabilities.

[`ARCHITECTURE.md`](ARCHITECTURE.md) is the authoritative architecture root and sole authority
index; it links the six incorporated scoped contracts. The contributor and agent workflow is
defined in [`AGENTS.md`](AGENTS.md).

Start with the [documentation index](docs/index.md). New contributors can follow
[Getting started](docs/getting-started.md); the
[implementation roadmap](docs/planning/roadmap.md) distinguishes current capabilities from
planned work.

## Current implementation status

Synaptik separates tensor meaning from execution ownership:

```text
Tensor expressions -> compile -> prepare -> run -> materialize
model semantics      recipe     resources  lease  detached values
```

`Engine.compute(...)` performs this lifecycle synchronously for one computation.
Use a reusable prepared execution for repeated inputs, and close runs, preparations and owned
integrations in lifetime order. Materialized host values remain usable after their owners close.
`Engine.standard()` creates a CPU-only composition; `Engine.builder()` explicitly composes
opened CPU and/or Metal integrations. There is no automatic backend discovery or execution fallback.

The [current production support matrix](docs/backend-guide/production-support.md) is the single
entry point for operation, dtype, layout, derivative and backend qualification. Java CPU supports
FLOAT32/FLOAT64 and BFLOAT16, FLOAT16 and both FP8 formats through the same ordinary, Scan,
recurrent and Compiler automatic-differentiation owners, under their family-specific guards.
Low arithmetic uses FLOAT32 working values with declared logical narrowing, not hidden
quantization or a separate executor. Optional native oneDNN acceleration is selected per qualified
occurrence; it is not required for CPU numerical execution.

Metal has its own seven-carrier and predicate domain, including qualified attention,
normalization and Scan occurrences; CPU support does not imply Metal support. Metal has no FP8
support. Consult the matrix rather than inferring support from an operation name or route count.
Explicit mixed CPU/Metal composition retains checked transfer boundaries, with no implicit
dtype conversion.

Training provides SGD, Adam and AdamW, parameter groups and strict in-memory session restore.
Bounded graph/Model persistence stores source meaning and values, not prepared native artifacts
or optimizer/session continuation. See the [public API](docs/api/public-api.md) and
[training lifecycle](docs/api/training-api.md) for the precise boundaries.
CUDA, ONNX and CLI module markers are not usable products.

Distribution remains **private/internal**: no public project licence or redistribution grant
has been selected, and no public upload is authorized. The
[public-readiness audit](docs/planning/public-release-audit-2026-11-04.md#result) and isolated
local qualification passed for `0.1.0-public-audit.20261104`. See the
[current support receipt](docs/backend-guide/production-support.md#public-readiness-audit-current-source-and-installed-candidate);
earlier `0.1.0` receipts still describe only their original source and artifacts.


## Prerequisites

- JDK 26
- an IntelliJ IDEA version with Java 26 support, when using the IDE

The Gradle wrapper downloads the supported Gradle distribution. A separate Gradle installation is not required.

## Gradle commands

Use the repository wrapper:

```shell
./gradlew projects
./gradlew test
./gradlew build
./gradlew :modules:model:javadoc
```

Synaptik does not enable Java preview features globally. A focused implementation task may enable a required preview or incubator feature only for the owning module, with explicit build configuration, documentation, and validation.

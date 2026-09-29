# Synaptik

Synaptik is a modular Java foundation for compiling, preparing, and executing computational graphs
across multiple backends. The project is under active development and currently provides a
runnable public lifecycle with fixed CPU convenience and explicit single- or bounded mixed-owner
CPU/Metal composition, plus backend-independent Tensor and compiler capabilities.

[`ARCHITECTURE.md`](ARCHITECTURE.md) is the authoritative architecture root and sole authority
index; it links the six incorporated scoped contracts. The contributor and agent workflow is
defined in [`AGENTS.md`](AGENTS.md).

Start with the [documentation index](docs/index.md). New contributors can follow
[Getting started](docs/getting-started.md); the
[implementation roadmap](docs/planning/roadmap.md) distinguishes current capabilities from
planned work.

## Current implementation status

The current public surface includes Tensor expressions, graph compilation, reusable inference
sessions that prepare once and run with isolated invocation state, detached host materialization,
one-shot forward computation, and a bounded scalar-objective backward convenience.
`Engine.standard()` owns a fixed
CPU composition; `Engine.builder()` explicitly owns opened CPU and/or Metal integrations and
executes non-empty single-owner plans or bounded mixed CPU/Metal plans through deterministic
owner-indexed representations and ordered transfer steps. Current cross-owner transfer supports
fully static rank-0..16 values of all seven carriers over canonical or approved positive-stride
non-overlapping layouts. Metal executes a closed occurrence-specific matrix. Its common exact
domain includes unary operations, affine movement and canonicalization, indexing, BOOL,
replacement/fold/aggregate operations, unsigned-32-bit-bounded stable ordering, top-K, numeric
arg-extrema, and maximum pooling. `SORT`, `ARGSORT`, and `TOP_K` admit all seven carriers;
`ARG_MIN` and `ARG_MAX` admit the six numeric carriers.
`ACCELERATOR` additionally admits the documented arithmetic, scalar, reduction, scan, MATMUL,
MSE, convolution, average-pooling, dropout, L1, ScatterAdd, and singleton-variance occurrences.
Every supported homogeneous `FLOAT32` accelerator occurrence has corresponding `BFLOAT16` and
`FLOAT16` ownership with `FLOAT32` working values and accumulators and one final ties-to-even
narrowing. Direct mixed `BFLOAT16`/`FLOAT16` operations have no low-precision kernel; callers use
explicit casts to `FLOAT32`. Every low arithmetic partition uses `CUSTOM_PROGRAM`; exact
homogeneous no-gradient raw-preserving `RESHAPE`, simple `PERMUTE`, materializing `CONTIGUOUS`,
`SLICE`, `CONCAT`, and `TILE` images may additionally use MPSGraph only after an exact schema-1
environment/program certificate match. Classic MPS and MPP remain qualified-negative. There is no
MPS, CPU, retry, generated-pointwise, or fusion fallback.

Canonical or authenticated local last-two-axis transpose MATMUL operands and exact local
singleton-height Conv1d/Pool1d compositions are allowed. Existing rank-two FLOAT32 matrix products
retain direct MPSGraph; newly admitted MATMUL forms and convolution/pooling rows use the fixed
custom program. Gradient-bearing arithmetic, scalar, reduction, scan, dropout, non-overlapping
maximum-pool, and other admitted differentiable rows retain the exact compiler-generated
first-order and owned higher-order paths. Overlapping generated folds, Conv3d backward, attention,
convolution transpose, and every other unlisted occurrence remain fail-closed. Exact custom nodes
fix their whole partition to one custom program with compact run-owned materialized slots and
direct targets; top-K publishes paired values and INT64 indices from one step. Deterministic
generated Metal units continue to fuse only their qualified FLOAT32 chains; low-precision
partitions never enter those routes. ABI 6 and fourteen exports are fixed; the route-bearing
program image is schema 19, data-type wires are `1..7`, and backend-local route, workload, policy,
and codec identities are version 28. The active certificate store contains exactly 24 positive
raw-preserving MPSGraph rows; its complete key and separate accuracy and determinism records are
reported only for the selected certified route.
Standard-Metal convenience, generic plugin registration/discovery, CUDA, broader optimizers,
durable persistence, and generic graph/plan tuning remain planned.
Focused documentation identifies the exact current boundary for each area.

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

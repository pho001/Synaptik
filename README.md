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
fully static rank-0..16 values of all six carriers over canonical or approved positive-stride
non-overlapping layouts. Metal executes a closed occurrence-specific matrix. Its common domain
includes exact unary, affine, canonicalization, indexing, BOOL, Task-0059 raw-movement,
Task-0060 replacement/fold/aggregate, unsigned-32-bit-bounded stable ordering, top-K, numeric
arg-extrema, and exact FLOAT64/FLOAT32/BFLOAT16 maximum Pool2d/Pool3d rows. `SORT`, `ARGSORT`,
and `TOP_K` admit all six carriers; `ARG_MIN` and `ARG_MAX` admit the five numeric carriers.
`ACCELERATOR` additionally admits the documented FLOAT32 arithmetic, reductions, scans, every
positive-static FLOAT32 `MATMUL` vector, matrix, batched, and broadcast geometry, same-type
canonical positive-rank FLOAT32 MSE forward execution for `NONE`, `SUM`, and `MEAN`, exact
no-gradient rank-one FLOAT32 `L1_NORM` over ordered axis `[0]`, FLOAT32-result grouped
Conv2d/Conv3d over FLOAT32/BFLOAT16 roles, and FLOAT32 average Pool2d/Pool3d. Convolution mixed
operands are no-gradient; all-FLOAT32 convolution and pooling
preserve their exact supported gradient metadata. Both profiles admit no-gradient promoted
INT32/INT64 `MATMUL`; accelerator also admits no-gradient BFLOAT16/FLOAT32 mixed pairs with
FLOAT32 result. Canonical or authenticated local last-two-axis transpose MATMUL operands and the
exact local singleton-height Conv1d/Pool1d compositions are allowed. Existing rank-two FLOAT32
matrix products retain direct MPSGraph; new MATMUL geometries, carrier pairs, and all six
convolution/pooling rows use the fixed custom program. MSE uses fixed MPSGraph subtraction,
self-multiplication, and optional full reduction and grants no generated backward ownership.
Strict convolution/average pooling, generated Conv3d and maximum-pool gradients, attention,
convolution transpose, and every other unlisted occurrence fail closed. Eligible singleton
negation retains its custom alternative. Exact custom nodes fix their whole partition to one
custom program with compact run-owned materialized slots and direct targets; top-K publishes paired
values and INT64 indices from one step. Deterministic generated Metal units fuse eligible linear
canonical FLOAT32 `FLOOR`/`CEIL`/`SIGN`/`RELU` chains while preserving every barrier. ABI 5 and
thirteen exports remain fixed; the route-bearing program image is schema 17 and backend-local
route, workload, policy, and codec identities are version 26.
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

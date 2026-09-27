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
includes the exact unary, affine, canonicalization, indexing, BOOL, Task-0059 raw-movement, and
Task-0060 replacement/fold/aggregate rows. `ACCELERATOR` additionally admits the documented
FLOAT32 arithmetic, reductions, scans, every positive-static FLOAT32 `MATMUL` vector, matrix,
batched, and broadcast geometry, and same-type canonical positive-rank FLOAT32 MSE forward
execution for `NONE`, `SUM`, and `MEAN`. Both profiles admit no-gradient promoted INT32/INT64
`MATMUL`; accelerator also admits no-gradient BFLOAT16/FLOAT32 mixed pairs with FLOAT32 result.
Canonical or authenticated local last-two-axis transpose operands are allowed. Existing rank-two
FLOAT32 matrix products retain direct MPSGraph; new MATMUL geometries and carrier pairs use the
fixed custom program. MSE uses fixed MPSGraph subtraction, self-multiplication, and optional full
reduction and grants no generated backward ownership; every other normalization/loss kind remains
false. Every unlisted occurrence fails closed. Eligible singleton negation retains its custom
alternative, and exact custom nodes fix their whole partition to one custom program with declared
run-owned intermediates and direct targets. ABI 5, thirteen exports, and the schema-15 operation
and attribute registries remain fixed; backend-local route and workload identities are version 18.
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

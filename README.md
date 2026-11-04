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

The current public surface includes Tensor expressions, graph compilation, reusable prepared
executions with isolated per-run state, detached host materialization, one-shot forward
computation, and a bounded scalar-objective backward convenience.
The [first-production matrix and remaining gates](docs/backend-guide/production-support.md)
declare equal Java CPU FLOAT32/FLOAT64 targets and required Metal support. CPU BFLOAT16, FLOAT16,
FLOAT8_E4M3 and FLOAT8_E5M2 numerical execution uses the existing operation, shared Compiler AD,
Scan and recurrent owners under their family, geometry and derivative constraints. Homogeneous
low arithmetic uses FLOAT32 working values and accumulators with final ties-to-even narrowing;
there is no implicit scale or quantization. Native oneDNN selection is occurrence-specific, not
a prerequisite for ordinary CPU execution; FLOAT64 remains ordinary CPU/OpenBLAS, not oneDNN.
General Scan composes ordinary CPU-supported PURE operations through
the existing lowering/emitters in all-valid and length-aware modes, including actual selected
backward, typed activity and static/qualified one-B geometry. Cold/wire and Metal Scan support
remain independently bounded; component qualification is not a whole-release claim.
The current distribution target is **private/internal**. No public project licence or
redistribution grant is selected, and no public registry upload is part of this release.
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
The qualified floating domain additionally admits documented arithmetic, scalar, reduction,
scan, MATMUL, MSE, convolution, average-pooling, dropout, L1, ScatterAdd and singleton-variance
occurrences. Declared homogeneous FLOAT32 rows have independently queried BFLOAT16/FLOAT16
counterparts with FLOAT32 working values/accumulators and one final ties-to-even
narrowing. Direct mixed `BFLOAT16`/`FLOAT16` operations have no low-precision kernel; callers use
explicit casts to `FLOAT32`. Every partition containing BFLOAT16 or FLOAT16 values uses
`CUSTOM_PROGRAM`, including exact homogeneous no-gradient raw-preserving `RESHAPE`, simple
`PERMUTE`, materializing `CONTIGUOUS`, `SLICE`, `CONCAT`, and `TILE`. It exposes no MPSGraph,
classic-MPS, MPP, CPU, retry, generated-pointwise, fusion, or fallback route.

Exact zero-offset dense MATMUL operands (logical views or not), authenticated local last-two-axis
transposes rooted in them, and exact local singleton-height Conv1d/Pool1d compositions are allowed.
Existing rank-two FLOAT32 matrix products retain direct MPSGraph; newly admitted MATMUL forms
and convolution/pooling rows use the fixed
custom program. Gradient-bearing arithmetic, scalar, reduction, scan, dropout, non-overlapping
low/FLOAT64 window paths and other admitted differentiable rows retain compiler-generated
first-order and owned higher-order paths. Shared grouped Conv2d/Conv3d AD now supports
qualified static CPU FLOAT32/FLOAT64 and Metal FLOAT32 training, including input, weight and
bias gradients and overlapping image folds. Maximum and average Pool3d now support qualified
static CPU FLOAT32/FLOAT64 and Metal FLOAT32 training, including overlapping windows.
Metal overlap-add remains FLOAT32-only; low/FLOAT64 Metal non-overlap restrictions remain.
See the [production support matrix](docs/backend-guide/production-support.md#conv3d-training-and-independent-pool3d-boundary).
Attention, convolution transpose and every other unlisted Metal occurrence remain fail-closed. Exact custom nodes
fix their whole partition to one custom program with compact run-owned materialized slots and
direct targets; top-K publishes paired values and INT64 indices from one step. Deterministic
generated Metal units continue to fuse only their qualified FLOAT32 chains; every BFLOAT16/FLOAT16
partition is custom-program-only. ABI 7 exposes thirteen symbols; the route-bearing program image
remains schema 20, data-type wires remain `1..7`, and backend-local route, workload, policy and
codec identities are version 40. Low-precision preparation trace reports the selected custom route
and ordered logical dtype tuple; tracing is diagnostic, not a numerical selector.
Standard-Metal convenience, generic plugin discovery, CUDA and broader optimizers remain planned.
Source-domain-bounded graph/Model persistence selects v7 only for FP8; old-only sources retain
their smallest sufficient v1-v6 format and bytes. NN checkpoints likewise select their FP8 v2
formats only when needed. Neither workflow stores optimizer/session continuation. Metal retains
its own seven-carrier domain and has no FP8 support.
Focused documentation identifies the exact current boundary for each area.

CPU also supports explicit **oneDNN** (external native kernel library) 3.14.0 composition through
`CpuBackendIntegration.openOneDnn(absolutePath[, OneDnnApi])`. Its primitive and Graph APIs are
routes inside the same CPU owner, not new backends or automatic-differentiation systems.
Qualified F32 arithmetic and literal composite units extend the existing bare MATMUL route;
unsupported implementations and geometry keep the ordinary CPU plan before execution.
Model/AD, Metal and default CPU composition remain unchanged. API preference does not force
native selection, and heuristic whole-plan costs do not promise a speedup.
No binary is bundled. See the [configuration, exact qualification and external-binary requirements](docs/backend-guide/cpu-backend.md#explicit-onednn-composition),
including JDK native-access permission, ownership and report-only benchmarking. Expanded-F32
source evidence is currently local macOS arm64; earlier sealed release receipts remain unchanged.

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

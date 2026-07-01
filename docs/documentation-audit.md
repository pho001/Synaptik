<!-- generated-by: codex-docs-audit -->
# Documentation Audit

Navigation: [Index](index.md#recommended-reading-paths) | [Quickstart](quickstart.md#what-synaptik-is) | [README](../README.md#reading-guide) | [Public API](public-api.md#stability-map) | [Architecture](architecture.md#system-overview) | [ONNX](onnx.md#onnx-import-and-export) | [Release](release.md#release-process)

Chapters: [Audit Scope](#audit-scope) | [Executive Findings](#executive-findings) | [Terminology Baseline](#terminology-baseline) | [Documentation Inventory](#documentation-inventory) | [Source Of Truth Map](#source-of-truth-map) | [Accuracy Notes By Area](#accuracy-notes-by-area) | [Known Stale-Risk Areas](#known-stale-risk-areas) | [Examples And Snippet Policy](#examples-and-snippet-policy) | [Verification Procedure](#verification-procedure) | [Recommended Follow-Up](#recommended-follow-up)

Audit date: 2026-07-01.

This document records the current state of the documentation set and the boundaries that future doc updates should preserve. It is not a replacement for the detailed subsystem guides. It is the map that explains which guides exist, which code owns the facts, and where a future maintainer should be suspicious of drift.

## Audit Scope

The audit covered:

- top-level project docs such as `README.md`, `CHANGELOG.md`, `LICENSE.md`, and `VERSION`;
- user and maintainer guides under `docs/`;
- package-local source docs under `src/main/java/**/README.md` and adjacent Markdown files;
- public Java API surfaces used by examples in the docs;
- current ONNX import/export docs and generated coverage report;
- release, tuning, calibration, backend planning, accelerator, BF16, and testing guides.

The audit deliberately did not rewrite local benchmark or calibration artifacts under `profiles/platform/...`. Those are machine-local measurement outputs unless a plan explicitly promotes them as canonical fixtures or release evidence.

## Executive Findings

The documentation set contains 34 top-level Markdown guides and covers public API, lifecycle, planning, backends,
tuning, testing, and historical migration evidence. The current source uses top-level `planning`, `prepare`, `runtime`,
and `trace`, low-level `backend.provider.blas.openblas`, and an opt-in cpu1 direct prepare route. Backend ownership is
represented by `Partition`/`PlannedPartition`; executable unit planning is represented by
`PartitionExecutionPlan` under `planning.partition.execution`, and the compile artifact exposes the combined
`ExecutablePartitionPlan`. The former standalone intermediate planning/lowering model has been removed from the current worktree; current
compile accessors, traces, renderers, and backend execution contracts consistently use partition terminology.

The main issues found during this pass were:

| Finding | Status after this pass | Why it mattered |
|---|---|---|
| No single deep quickstart existed. | Fixed by adding [quickstart.md](quickstart.md). | New users needed one guided path from build to tensor graph, autodiff, explicit execution, ONNX, and troubleshooting. |
| Some overview language framed Synaptik too broadly as a "framework." | Fixed in primary overviews: `README.md`, `framework-concepts.md`, and this audit. | The intended core identity is autograd engine plus compiled tensor runtime, not a full neural-network framework. |
| `public-api.md` still referenced older `config.optimizer.*` as the main public config layer. | Fixed in this pass. | Current architecture centers `config.compile.*`, `config.runtime.*`, and `config.profile.*`. |
| The docs had no explicit audit artifact explaining source-of-truth ownership. | Fixed by this document. | Large docs need a map so future updates do not turn generated summaries into unverified claims. |
| ONNX support is expanding quickly and needs exact boundary wording. | Existing docs are good; audit calls out generated coverage and stale-risk areas. | ONNX docs must distinguish import/export support, CPU execution, GPU lowering, and static helper behavior. |
| Accelerator docs are intentionally capability-scoped and machine-dependent. | Left as explicit `Needs verification` where appropriate. | Metal/CUDA runtime availability cannot be proven from source alone. |
| Lifecycle and source maps still named removed graph-execution, backend-runtime, backend-memory, and central-dispatch classes. | Fixed across the documentation set. | Runtime now invokes prepared executable contracts and must not select concrete backends. |
| The dirty worktree adds `CpuExecutionPolicy` and expands cpu1 parity/readiness evidence. | Documented as an opt-in prepare route with explicit fallback policy. | The default remains the established CPU path; docs must not imply that cpu1 is globally enabled. |
| Planning docs still described the removed standalone intermediate planning/lowering model. | Fixed to distinguish ownership `Partition`/`PlannedPartition`, combined `ExecutablePartitionPlan`, and its `PartitionExecutionPlan`. | Source paths and ownership terminology must match the current dirty worktree, not the pre-migration model. |
| OpenCL prepare was described as metadata-only. | Fixed: prepare attaches `OpenClDirectPreparedExecutable`; the current registry contains only `NOOP` and unsupported operations fail during prepare. | The executable contract and the narrow capability boundary are both observable behavior. |
| Shared accelerator buffer contracts were assigned to a nonexistent backend package. | Fixed to `runtime.device.buffer`; Metal/CUDA bind those neutral decisions to backend-owned handles. | Buffer policy and run-state ownership belongs to runtime, while concrete handles stay backend-owned. |

## Terminology Baseline

Future documentation should use these terms consistently.

| Term | Meaning | Example |
|---|---|---|
| Tensor | Public graph node and typed storage object. | `new Tensor(...).add(...).relu()` |
| Semantic graph | The user-visible DAG built by `Tensor` operations. | `loss = x.mul(x).sum()` |
| DAG | Directed acyclic graph. Nodes point to dependencies and cannot form cycles. | `a -> add -> relu -> output` |
| Operation descriptor | Immutable description of what a node means, not how to execute it. | `operations.elementwise.binary.add` |
| Compile | Snapshot graph, canonicalize, build backward graph if needed, optimize, plan backend ownership, plan memory. | `CompiledGraph.compile(y, CompileConfig.inference())` |
| Prepare | Resolve runtime/backend policy into executable steps and backend metadata. | `compiled.prepare(RuntimeConfig.inferenceDefaults())` |
| Execute | Run prepared steps and publish selected outputs/gradients. | `prepared.execute(ExecutionMode.FORWARD)` |
| Graph optimization | Backend-neutral graph rewrite/simplification/lowering. | `AR`, `CF`, `CSE`, `DCE`, optional `LOWER` |
| Backend planning | Compile-time ownership planning for CPU or accelerator partitions. | `BackendPlanningConfig.autoAccelerator()` |
| Runtime policy | Hardware-facing execution policy. | CPU thresholds, BLAS, fused backend, accelerator availability |
| Publication | Copying run-scoped values back to public `Tensor` objects after execution. | `PublicationPolicy.OUTPUT_ONLY` |
| Calibration | Measuring platform/runtime defaults. | vector threshold, BLAS threshold |
| Graph autotune | Measuring executable graph-policy candidates for one workload. | current graph policy vs backend-planning variants |
| Static ONNX subset | ONNX forms whose shapes and shape-like parameters are known at import/export time. | static `Reshape`, static `Slice`, static `Range` folding |

The most important identity statement:

```text
Synaptik core is an autograd engine and compiled tensor runtime.
It is not the layer/module/model-zoo framework above that engine.
```

That does not mean high-level neural-network layers are impossible. It means those layers should be designed as a higher layer that builds ordinary Synaptik primitive DAGs and can optionally use ONNX interchange at a layer-aware boundary later.

## Documentation Inventory

Current top-level guides under `docs/`:

| Document | Current role | Audit status |
|---|---|---|
| [quickstart.md](quickstart.md) | Detailed first path through build, tensor graph, autodiff, explicit execution, publication, profiles, ONNX, accelerators, tuning, troubleshooting. | New in this pass. |
| [index.md](index.md) | Main navigation, recommended reading paths, document map. | Updated in this pass. |
| [architecture.md](architecture.md) | Architecture and lifecycle overview with backend dispatch and diagrams. | Strong; keep updated after major pipeline changes. |
| [framework-concepts.md](framework-concepts.md) | First-principles conceptual model. | Updated to avoid overclaiming full-framework identity. |
| [compute-flow.md](compute-flow.md) | Deep lifecycle guide from `Tensor.compute(...)` through traces and backend behavior. | Strong; high value, high stale risk because it is detailed. |
| [graph-optimizer.md](graph-optimizer.md) | Graph optimization stages, simplification fixpoint, lowering, diagnostics. | Strong; must stay separate from backend planning docs. |
| [backend-planning-and-partitions.md](backend-planning-and-partitions.md) | Backend ownership, partitions, memory planning, publication, benchmark semantics. | Strong; central architecture doc. |
| [configuration.md](configuration.md) | Build, runtime, profile, CLI, native lookup, tuning persistence. | Strong; update whenever config records move. |
| [public-api.md](public-api.md) | Public and probably-internal Java API surfaces. | Updated in this pass. |
| [tensor-api.md](tensor-api.md) | Detailed operation-level Tensor API. | Strong; large and high stale risk. |
| [sequence-tensor-primitives.md](sequence-tensor-primitives.md) | Detailed guide to N-D sequence-friendly tensor primitives: factories, shape helpers, last-dimension linear, stack/unstack, axis indexing, masked reductions, and masked cross entropy. | New public API guide; keep synchronized with `Tensor.java`, `TensorOps.java`, `tensor.ops.*`, and `NdTensorSequencePrimitivesTest`. |
| [adding-tensor-operation.md](adding-tensor-operation.md) | Contributor checklist for adding operations. | Strong; update for new operation families. |
| [examples.md](examples.md) | Executable-style snippets and usage flows. | Useful; should defer deeper onboarding to quickstart. |
| [onnx.md](onnx.md) | ONNX import/export boundary and semantics. | Strong; must track every ONNX wave. |
| [onnx-coverage.md](onnx-coverage.md) | Generated ONNX coverage matrix. | Treat as generated from `OnnxCoverageMatrix`. |
| [cpu-bf16.md](cpu-bf16.md) | BF16 CPU storage/compute/performance contract. | Strong; important for benchmark interpretation. |
| [metal-backend.md](metal-backend.md) | Metal backend internals, buffer ABI, traces, capability boundaries. | Strong; hardware-dependent sections remain `Needs verification`. |
| [cuda-backend.md](cuda-backend.md) | CUDA backend state and boundaries. | Useful; likely lower maturity than Metal docs. |
| [gpu-lowering-coverage.md](gpu-lowering-coverage.md) | GPU lowering coverage details. | Should be kept in sync with coverage matrix code. |
| [gpu-coverage-triage.md](gpu-coverage-triage.md) | GPU support triage and prioritization. | Useful as planning/triage evidence. |
| [gpu-lowered-partition-manifest.md](gpu-lowered-partition-manifest.md) | Manifest-style lowered-partition evidence. | Verify after major accelerator lowering changes. |
| [metal-operation-parity.md](metal-operation-parity.md) | Metal operation parity matrix. | Generated/derived style; high stale risk. |
| [native-bridges-and-blas.md](native-bridges-and-blas.md) | BLAS/GEMM/FFM bridge explanation and native lookup. | Strong; detailed educational doc. |
| [calibration-autotune.md](calibration-autotune.md) | Full calibration/autotune architecture and API guide. | Very strong; very large. Update carefully. |
| [modules.md](modules.md) | Package-by-package map. | Strong; update on package moves. |
| [mechanisms.md](mechanisms.md) | Mechanism-oriented explanation of graph/runtime features. | Useful supporting doc. |
| [testing.md](testing.md) | Test commands, organization, expensive tests. | Updated in release pass; keep focused commands current. |
| [development.md](development.md) | Local development practices and implementation notes. | Useful; some machine-observation notes are intentionally scoped. |
| [troubleshooting.md](troubleshooting.md) | Symptom-driven debugging. | Useful; should absorb repeated support answers. |
| [glossary.md](glossary.md) | Terminology list. | Useful; update when architecture terms become canonical. |
| [release.md](release.md) | Versioning, release files, verification, artifact hygiene. | New in release hardening pass. |

Package-local source docs:

| Area | Files | Role |
|---|---|---|
| Tensor API internals | `src/main/java/tensor/README.md`, `src/main/java/tensor/API.md` | Package-local API and operation detail. |
| Operation descriptors | `src/main/java/operations/README.md` | Primitive descriptor contract and operation taxonomy. |
| Graph compile | `src/main/java/graph/README.md` | Graph package lifecycle and compile objects. |
| Optimizer | `src/main/java/graph/optimizer/README.md`, `AR.md`, `CSE.md`, `FUSE.md`, `MEM.md` | Detailed optimizer implementation notes. |
| Backend | `src/main/java/backend/README.md`, `backend/cpu/README.md`, `prepare/README.md`, `backend/lowering/README.md`, `backend/partition/README.md` | Backend execution, lowering, partition, and prepare implementation docs. |
| Tuning | `src/main/java/tuning/README.md`, `ARCHITECTURE.md`, `KNOBS.md`, `PERSISTENCE.md`, `REPORTING.md`, `SEARCH.md`, `WORKLOADS.md`, `LEGACY-BENCHMARK-REVIEW.md` | Measurement and persistence internals. |
| Numerics | `src/main/java/numerics/README.md` | Numerical drift harness. |

## Source Of Truth Map

Use this table when updating docs.

| Documentation claim | Source of truth |
|---|---|
| Tensor constructors, methods, storage accessors | `src/main/java/tensor/Tensor.java`, `tensor/ops/**`, `TensorOps.java` |
| Sequence-friendly public tensor helpers | `Tensor.java`, `TensorOps.java`, `TensorDataFactory.java`, concrete `tensor.ops.*` operation builders, `NdTensorSequencePrimitivesTest.java` |
| DType support | `tensor/DataType.java`, storage classes, CPU dtype resolver, backend coverage matrices |
| Compile config presets | `src/main/java/config/compile/CompileConfig.java` and sibling records |
| Runtime config presets | `src/main/java/config/runtime/RuntimeConfig.java` and sibling records |
| Execution profiles | `src/main/java/config/profile/ExecutionProfile.java`, profile IO classes |
| Publication behavior | `src/main/java/runtime/execution/PublicationPolicy.java`, `PreparedExecution.java`, `PublicationPolicyTest.java` |
| Compile lifecycle | `CompiledGraph.java`, `GraphCompiler.java`, `CompileArtifacts.java` |
| Prepare lifecycle | `prepare/**`, `PreparedExecutionBuilder.java` |
| Execution lifecycle | `PreparedExecution.java`, `ExecutionState.java`, `PreparedExecutionRunner.java` |
| Graph optimizer stages | `GraphOptimizationConfig.java`, `OptimizerFactory.java`, `graph/optimizer/**` |
| Backend planning | `BackendPlanningConfig.java`, `planning/partition/**`, including `planning/partition/execution/**` and `planning/partition/specialization/**` |
| Memory planning | `MemoryPlanningConfig.java`, `planning/memory/**` |
| ONNX public API | `src/main/java/onnx/Onnx.java`, `OnnxModel.java`, `ImportedOnnxModel.java` |
| ONNX supported ops | `OnnxGraphImporter.java`, `OnnxGraphExporter.java`, `OnnxCoverageMatrix.java`, ONNX tests |
| GPU lowering coverage | `GpuLoweringCoverageMatrix.java`, backend-specific lowering/semantics classes |
| Metal behavior | `backend/metal/**`, Metal tests, `metalTest` |
| CUDA behavior | `backend/cuda/**`, CUDA tests when available |
| Calibration/autotune CLI | `synaptik/app/TuningCli.java`, `tuning/**`, profile store classes |
| Release version | `VERSION`, `build.gradle`, `CHANGELOG.md` |
| License statement | `LICENSE.md` |

Rule of thumb:

```text
If a documentation sentence says "Synaptik supports X",
there should be a source class, coverage row, or test that explains exactly what X means.
```

## Accuracy Notes By Area

### Identity And Product Boundary

The docs should consistently say Synaptik is an autograd engine and compiled tensor runtime. "Framework" is acceptable only in generic educational sentences, for example "a higher-level framework can keep its public API stable," but not as the primary product identity.

Good wording:

```text
Synaptik core is a Java autograd engine and compiled tensor runtime.
```

Risky wording:

```text
Synaptik is a full neural-network framework.
```

Why risky: a full NN framework implies high-level layer/module APIs, dataloaders, optimizers, checkpointing, training loops, and model interchange at the layer level. The core repository intentionally keeps those above the primitive tensor/autograd layer.

### Sequence-Shaped Tensor API

The sequence-related API should be documented as general N-D tensor mechanics, not as a neural-network layer system.

Good wording:

```text
`Tensor.linear` projects the last dimension of an N-D tensor.
`Tensor.stack` can convert several `[batch, features]` timestep tensors into one `[batch, time, features]` tensor.
Masked reductions ignore padded positions where a BOOL mask is false.
```

Risky wording:

```text
Synaptik provides RNN/LSTM/GRU sequence layers.
```

Why risky: the core repository provides primitives for a consumer framework to build layers above it. It does not own recurrent cell state management, layer parameter containers, packed sequence abstractions, training loops, or model-level serialization. The source-of-truth test for this boundary is `NdTensorSequencePrimitivesTest`: it verifies factory helpers, N-D linear, stack/unstack, axis indexing, masked reductions, and masked loss without introducing layer types.

### Compile And Runtime Configuration

The current model is:

```text
CompileConfig
  -> SemanticCanonicalizationConfig
  -> GraphOptimizationConfig
  -> BackendPlanningConfig
  -> PartitionExecutionConfig
  -> MemoryPlanningConfig

RuntimeConfig
  -> KernelTuningConfig
  -> ApproximationConfig
  -> BlasConfig
  -> Conv2dConfig
  -> FusedExecutionPolicy
  -> AcceleratorConfig

ExecutionProfile
  -> CompileConfig + RuntimeConfig + dtype + execution mode + workload metadata
```

Docs should not reintroduce `OffloadConfig` as the mental model. Backend planning is compile-time execution planning, not execute-time offload.

### Graph Optimization

Graph optimization owns backend-neutral graph structure:

```text
AR -> CF -> CSE -> DCE -> optional LOWER
```

Definitions:

- `AR`: algebraic rewrite.
- `CF`: constant folding.
- `CSE`: common subexpression elimination.
- `DCE`: dead-code elimination.
- `LOWER`: optional backend-neutral lowering.

Docs should not describe partitioning as an optimizer stage in the target architecture. Partitioning/backend ownership belongs to backend planning and partition planning.

### Autodiff And Gradient Primitives

Semantic Tensor API should prefer building gradients as ordinary primitive DAGs when possible.

Example:

```text
softmax gradient should be represented through visible primitive graph structure
unless a later optimizer/backend rewrite intentionally fuses the pattern.
```

Specialized gradient operation descriptors can remain for tests, backend fixtures, or future CPU/backend specialization, but docs should not imply they are the normal semantic API output when the current Tensor API builds primitive DAGs.

### Publication Policy

`PublicationPolicy` is now a first-class concept and should be documented wherever execution side effects are discussed.

Correct model:

```text
Publication controls which run-scoped values are synchronized to public Tensor objects.
```

Incorrect model:

```text
Publication changes graph planning or execution semantics.
```

Policy summary:

- `ALL`: publish every forward value plus gradients.
- `OUTPUT_AND_GRADIENTS`: publish root output plus gradients.
- `OUTPUT_ONLY`: publish only root output.
- `NONE`: publish nothing.

### ONNX

ONNX docs must separate four dimensions:

| Dimension | Meaning |
|---|---|
| Import support | ONNX node can be translated into a Synaptik graph. |
| Export support | Synaptik graph form can be serialized as ONNX. |
| CPU support | Imported graph has a CPU execution path. |
| GPU support | Specific accelerator backend can lower and execute the operation natively. |

Example:

```text
ScatterElements can be import/export/CPU-supported while still being CUDA-unsupported.
```

That is not a contradiction. ONNX interchange support and accelerator native support are different contracts.

### BF16

BF16 documentation should not imply automatic speedup on CPU.

Correct model:

- BF16 can reduce storage bandwidth and memory footprint.
- CPU elementary compute often promotes to F32/F64 or pays conversion costs.
- BF16 speedups are workload- and backend-dependent.
- Matmul-heavy or accelerator-native workloads are more likely to benefit.

### Metal And CUDA

Accelerator docs should keep scoped language:

- "supported for this operation family, dtype, layout, and backend";
- "fallback is visible in traces";
- "CPU remains correctness baseline";
- "runtime availability is machine-dependent."

Avoid broad claims such as "GPU supports ONNX" without naming operation families and dtype/layout limits.

### Calibration And Autotune

Docs should preserve this separation:

```text
calibration tunes runtime defaults for a platform
graph autotune compares graph/profile candidates for a workload
benchmarks measure real ExecutionProfile candidates
```

Calibration must not silently change graph ownership policy. Graph autotune may compare backend planning policies, but those policies still become explicit compile config inside measured `ExecutionProfile` objects.

## Known Stale-Risk Areas

These areas change often and should be checked whenever source changes land.

| Area | Why stale risk is high | Minimum verification |
|---|---|---|
| ONNX support tables | Import/export waves add many ops and caveats. | Regenerate or run `OnnxCoverageMatrixTest`; compare `docs/onnx-coverage.md`. |
| GPU lowering matrices | Backend support changes per op/dtype/layout. | Check `GpuLoweringCoverageMatrix.java` and backend tests. |
| Metal docs | Native bridge and MPSGraph behavior are hardware/runtime-dependent. | Run `./gradlew metalTest` where possible. |
| CUDA docs | CUDA backend is narrower and may lag source docs. | Run CUDA-specific tests where available. |
| Tensor API docs | The public `Tensor` surface is large. | `rg` method signatures and run targeted tests for changed op families. |
| Configuration docs | Records were consolidated recently. | Check `config.compile`, `config.runtime`, and `config.profile`. |
| Tuning docs | Calibration/autotune has many knobs and persistence fields. | Check CLI, profile IO, candidate metadata tests. |
| Release docs | Versioning and license status can change outside code. | Check `VERSION`, `CHANGELOG.md`, `LICENSE.md`, CI. |

`Needs verification` markers are acceptable when the claim cannot be proven from source alone. Examples include native runtime availability, exact local performance timings, or machine-specific bridge loading. They should not be used as a substitute for checking an ordinary Java signature or test.

## Examples And Snippet Policy

Examples should follow these rules:

1. Import real classes.
2. Prefer small shapes and exact expected values.
3. Explain shape transformations.
4. Distinguish graph construction from execution.
5. Use `DataType.FLOAT64` for simplest numerical examples unless the example specifically teaches F32/BF16/INT64/BOOL.
6. Use `CompileMode.TRAINING` only when a leaf has `setRequiresGrad(true)`.
7. Use explicit `CompiledGraph` and `PreparedExecution` examples for performance or repeated execution.
8. Mention `PublicationPolicy` whenever examples intentionally skip output or gradient publication.
9. For ONNX, use `OnnxLeafTensorPolicy.INPUTS` when leaf tensors should become runtime model inputs.
10. For accelerator examples, say whether fallback is allowed or required.

Bad example:

```java
Tensor y = model.forward(x);
```

Why it is bad in core docs: `model.forward` implies a high-level model/layer framework that is not the core API.

Better core example:

```java
Tensor logits = x.matmul(weight).add(bias);
Tensor loss = logits.logSoftmax(-1).nllLossFromIndices(targets, -1).mean();
```

This shows the primitive graph that a higher-level layer could build.

## Verification Procedure

For documentation-only changes:

```bash
git diff --check
./gradlew classes
```

For docs that touch ONNX:

```bash
./gradlew test --tests 'onnx.*' --tests SourceTreeHygieneTest
```

For docs that touch Tensor API operation semantics:

```bash
./gradlew test --tests '*Tensor*' --tests '*Execution*'
```

For docs that touch backend planning or accelerator behavior:

```bash
./gradlew test --tests 'planning.partition.*'
./gradlew metalTest
```

Use hardware-specific tests only when the machine can support them. If they cannot run, the final note should say they were not run and why.

## Recommended Follow-Up

The docs are now navigable, but these improvements would make them even stronger:

1. Add compile-tested Java snippets for the most important quickstart examples.
2. Add a generated docs coverage check that fails when `OnnxCoverageMatrix` and `docs/onnx-coverage.md` diverge.
3. Add a small "higher-level framework above Synaptik" design note once the layer API discussion becomes active.
4. Split very large docs only when a section becomes independently maintained. `calibration-autotune.md` is intentionally comprehensive today, but it may eventually deserve separate user, internals, and CLI guides.
5. Add a short release-readiness checklist that maps the current alpha release claims directly to tests.
6. Add a doc snippet style guide so future examples consistently explain shapes, dtypes, lifecycle stage, expected values, and fallback behavior.

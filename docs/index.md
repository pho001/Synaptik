<!-- generated-by: gsd-doc-writer -->
# Documentation Index

Navigation: [README](../README.md#reading-guide) | [Quickstart](quickstart.md#what-synaptik-is) | [Architecture](architecture.md#system-overview) | [Tensor API](tensor-api.md#api-surface-and-conventions) | [Sequence Tensor Primitives](sequence-tensor-primitives.md#scope) | [Adding Tensor Operation](adding-tensor-operation.md#implementation-checklist) | [Compute Flow](compute-flow.md#lifecycle-map) | [Graph Optimizer](graph-optimizer.md#graph-optimizer) | [Backend Planning](backend-planning-and-partitions.md#backend-planning-and-partitions) | [ONNX](onnx.md#onnx-import-and-export) | [CPU BF16](cpu-bf16.md#cpu-bf16-runtime) | [Native Bridges & BLAS](native-bridges-and-blas.md#term-map-at-a-glance) | [Metal Backend](metal-backend.md#end-to-end-flow) | [Calibration & Autotune](calibration-autotune.md#core-distinction) | [Public API](public-api.md#stability-map) | [Examples](examples.md#running-examples) | [Docs Audit](documentation-audit.md#audit-scope)

Chapters: [Recommended Reading Paths](#recommended-reading-paths) | [Document Map](#document-map) | [Release Notes](#release-notes) | [Source Documentation](#source-documentation) | [Verification Notes](#verification-notes)

This directory is the implementation-grounded documentation set for Synaptik. It is written for maintainers and technically strong readers who want to understand both how to use the autograd engine and how this kind of compiled tensor runtime works internally.

## Table Of Contents

- [Recommended Reading Paths](#recommended-reading-paths)
- [Document Map](#document-map)
- [Release Notes](#release-notes)
- [Source Documentation](#source-documentation)
- [Verification Notes](#verification-notes)

## Recommended Reading Paths

### New maintainer path

1. [Quickstart: What Synaptik Is](quickstart.md#what-synaptik-is) - the guided first path through build, tensor graphs, autodiff, explicit execution, ONNX, and troubleshooting.
2. [Architecture: System Overview](architecture.md#system-overview) - the layered system model, package boundaries, and lifecycle artifacts.
3. [Framework Concepts: Tensors As Graph Nodes](framework-concepts.md#tensors-as-graph-nodes) - core vocabulary and the mental model behind compiled tensor execution.
4. [Compute Flow: Lifecycle Map](compute-flow.md#lifecycle-map) - the detailed journey from `Tensor` graph construction through compile, prepare, execution, memory binding, and traces.
5. [Graph Optimizer](graph-optimizer.md#graph-optimizer) - backend-neutral graph simplification and lowering: `AR`, `CF`, `CSE`, `DCE`, and optional `LOWER`.
6. [Backend Planning And Partitions](backend-planning-and-partitions.md#backend-planning-and-partitions) - backend ownership planning, CPU natural partitions, accelerator partitions, partition optimization, memory planning, and publication.
7. [Native Bridges & BLAS: Term Map At A Glance](native-bridges-and-blas.md#term-map-at-a-glance) - what BLAS/GEMM are, how Java FFM calls native libraries, and how OpenBLAS is selected.
8. [Modules: Package Map](modules.md#package-map) - package-by-package responsibilities, dependencies, invariants, and failure modes.

### Tensor API user path

1. [Quickstart: First Tensor Graph](quickstart.md#first-tensor-graph) - small runnable examples with exact shapes and values.
2. [Tensor API Guide: API Surface And Conventions](tensor-api.md#api-surface-and-conventions) - operation-level API explanations with concrete input/output examples and calculations.
3. [Sequence Tensor Primitives](sequence-tensor-primitives.md#scope) - N-D linear, stack/unstack, axis indexing, masked reductions, masked cross entropy, and factory helpers for sequence-shaped consumer frameworks.
4. [Tensor API: Compute Convenience API](tensor-api.md#compute-convenience-api) - how to call `compute()`, `compute(CompileMode)`, `compute(ComputeOptions)`, `compute(ExecutionProfile)`, and prepared execution.
5. [Compute Flow: Tensor Compute API](compute-flow.md#tensor-compute-api) - what happens internally when `compute(...)` resolves a profile, compiles, prepares, executes, and optionally runs generic graph autotune.
6. [Public API: Stability Map](public-api.md#stability-map) - the externally usable Java surfaces and probably-internal implementation hooks.
7. [Examples: Running Examples](examples.md#running-examples) - executable-style snippets for tensor operations, compile/prepare/execute, and tuning flows.
8. [Troubleshooting: Performance Regressions](troubleshooting.md#performance-regressions) - common runtime, compile, dtype, backend, and tuning failures.

### Tensor operation contributor path

1. [Adding A Tensor Operation: Implementation Checklist](adding-tensor-operation.md#implementation-checklist) - end-to-end implementation guide for new operations.
2. [Development: Adding Tensor Ops](development.md#adding-tensor-ops) - compact checklist and current package conventions.
3. [Modules: `tensor`](modules.md#tensor-public-graph-building-surface) and [Modules: `operations`](modules.md#operations-primitive-semantic-descriptors) - where public graph builders and descriptors live.
4. [Graph Optimizer: CSE](graph-optimizer.md#cse) and [Backend Planning: Partition Optimization](backend-planning-and-partitions.md#partition-optimization) - what must be updated when an op has parameters or participates in fusion/partition planning.
5. [Testing: Targeted Test Patterns](testing.md#targeted-test-patterns) - focused Gradle commands and test organization.

### Optimizer and runtime path

1. [Compute Flow: Compile](compute-flow.md#compile) - the lifecycle from semantic graph to prepared runtime execution.
2. [Backend Planning And Partitions](backend-planning-and-partitions.md#backend-planning-and-partitions) - CPU natural partitions, accelerator ownership partitions, partition optimization, memory planning, and publication policy.
3. [Mechanisms: Prepared Execution](mechanisms.md#prepared-execution) - cross-cutting mechanisms such as graph construction, compilation, preparation, memory planning, and dispatch.
4. [Configuration: RuntimeConfig](configuration.md#runtimeconfig) - compile/runtime knobs, system properties, native lookup, and profile layout.

### Metal and accelerator debugging path

1. [Architecture: Accelerator Scaffolding](architecture.md#accelerator-scaffolding) - what the Metal/CUDA/OpenCL source layers contain and where the current Metal capability boundary is.
2. [Metal Backend: End-To-End Flow](metal-backend.md#end-to-end-flow) - detailed Java FFM, Objective-C shim, buffer ABI, storage residency, trace, and fallback mechanics.
3. [Architecture: Metal MPS Buffer Execution And Copy Chain](architecture.md#metal-mps-buffer-execution-and-copy-chain) - how tensor-array and buffer-binding transport differ.
4. [Compute Flow: Native buffer-binding Metal path](compute-flow.md#native-buffer-binding-metal-path) - how execution state keeps Metal outputs device-owned until materialization.
5. [Native Bridges & BLAS: How This Differs From Metal FFM](native-bridges-and-blas.md#how-this-differs-from-metal-ffm) - the shared Java FFM idea and the different execution models for BLAS and Metal.
6. [Backend Planning And Partitions: Accelerator Partitions](backend-planning-and-partitions.md#accelerator-partitions) - how accelerator ownership, legality, planning cost, runtime selection, and fallback evidence fit together.
7. [Calibration & Graph Autotune: Built-in workload catalogs](calibration-autotune.md#built-in-workload-catalogs) - transformer shape presets for stressing larger attention and FFN workloads.

### BF16 performance path

1. [CPU BF16 Runtime](cpu-bf16.md#cpu-bf16-runtime) - BF16 storage, F32/F64 compute and accumulation, conversion costs, fusion impact, and why BF16 can be slower than F32 on CPU.
2. [Native Bridges & BLAS](native-bridges-and-blas.md#term-map-at-a-glance) - where OpenBLAS and native GEMM can help BF16 matmul-heavy workloads.
3. [Backend Planning And Partitions: CPU Natural Partitions](backend-planning-and-partitions.md#cpu-natural-partitions) - why a large CPU partition is not the same as one monolithic fused BF16 kernel.
4. [Troubleshooting: Performance Regressions](troubleshooting.md#performance-regressions) - how to separate compile policy, runtime calibration, materialization, publication, and dtype conversion costs.

### Calibration and autotune path

1. [Calibration & Graph Autotune: Core Distinction](calibration-autotune.md#core-distinction) - calibration families, owned knobs, candidate values, graph autotune parameters, persistence, and progress.
2. [Configuration: Tuning And Calibration Persistence](configuration.md#tuning-and-calibration-persistence) - profile persistence paths and command-line configuration.
3. [Testing: Targeted Test Patterns](testing.md#targeted-test-patterns) - relevant tests and how to run focused validation.
4. Source package docs under `src/main/java/tuning/`, especially `ARCHITECTURE.md`, `KNOBS.md`, `PERSISTENCE.md`, `SEARCH.md`, and `WORKLOADS.md`.

## Document Map

| Document | Purpose |
|---|---|
| [quickstart.md](quickstart.md#what-synaptik-is) | Detailed first path through build, tensors, broadcasting, autodiff, explicit compile/prepare/execute, publication, profiles, ONNX, accelerators, autotune, and troubleshooting. |
| [architecture.md](architecture.md#system-overview) | High-level architecture, lifecycle boundaries, package responsibilities, backend dispatch, and extension points. |
| [framework-concepts.md](framework-concepts.md#tensors-as-graph-nodes) | First-principles mental models for tensors, semantic graphs, compiled graphs, prepared execution, backend policy, and tuning. |
| [compute-flow.md](compute-flow.md#lifecycle-map) | Deep end-to-end walkthrough from graph building to `Tensor.compute(...)`, compile, prepare, execution, traces, and reuse rules. |
| [graph-optimizer.md](graph-optimizer.md#graph-optimizer) | Backend-neutral graph optimization, simplification fixpoint behavior, lowering, snapshot safety, and optimizer diagnostics. |
| [backend-planning-and-partitions.md](backend-planning-and-partitions.md#backend-planning-and-partitions) | Backend ownership planning, CPU natural partitions, accelerator partitions, partition optimization, memory planning, publication policy, and benchmark semantics. |
| [onnx.md](onnx.md#onnx-import-and-export) | ONNX import/export boundary, public API, supported static dense inference subset, dtype/op mapping, and failure policy. |
| [onnx-coverage.md](onnx-coverage.md#summary) | Generated ONNX import/export coverage report derived from `OnnxCoverageMatrix`. |
| [cpu-bf16.md](cpu-bf16.md#cpu-bf16-runtime) | CPU BF16 storage/compute/accumulation contract, conversion costs, fusion limits, BLAS implications, and trace-reading guidance. |
| [native-bridges-and-blas.md](native-bridges-and-blas.md#term-map-at-a-glance) | Explanation of BLAS/GEMM, OpenBLAS dispatch, Java FFM bridges, native lookup, fallbacks, and performance tradeoffs. |
| [cpu-storage-rewrite-plan.md](cpu-storage-rewrite-plan.md#cpu-storage-rewrite-plan) | Current CPU storage rewrite scope, wave order, package checklist, fused exclusion, and verification baseline. |
| [cpu-kernels-wave0-baseline.md](cpu-kernels-wave0-baseline.md#cpu-kernels-wave-0-baseline) | Baseline owner map, execution-path classification, native import gate, and benchmark sanity list for the CPU kernels rewrite. |
| [metal-backend.md](metal-backend.md#end-to-end-flow) | Detailed Metal backend guide covering planner legality, Java FFM, Objective-C MPS shim, native buffer ABI, residency, traces, and fallbacks. |
| [cuda-backend.md](cuda-backend.md#purpose-and-current-status) | Current CUDA capability, native-buffer route, parity gaps, fallback semantics, and verification commands. |
| [gpu-lowering-coverage.md](gpu-lowering-coverage.md#status-legend) | Metal/CUDA lowering coverage, stable reason codes, operation-family rows, and regression gates. |
| [gpu-coverage-triage.md](gpu-coverage-triage.md#what-the-triage-ranks) | Portable GPU coverage-gap ranking, target expectations, router evidence, and artifact hygiene. |
| [gpu-lowered-partition-manifest.md](gpu-lowered-partition-manifest.md#purpose) | Structured lowered GPU partition metadata, value assumptions, rejection evidence, and trace/report boundary. |
| [metal-operation-parity.md](metal-operation-parity.md#metal-operation-parity-matrix) | Generated Metal operation parity matrix derived from backend capability data. |
| [adding-tensor-operation.md](adding-tensor-operation.md#implementation-checklist) | Contributor guide for adding a new tensor operation through descriptors, builders, public API, CPU kernels, autograd, optimizer/fusion integration, docs, and tests. |
| [tensor-api.md](tensor-api.md#api-surface-and-conventions) | Detailed public Tensor API guide with signatures, `compute(...)` options, edge cases, and value-level operation examples. |
| [sequence-tensor-primitives.md](sequence-tensor-primitives.md#scope) | Detailed guide to N-D sequence-friendly primitives: factories, shape helpers, last-dimension linear, stack/unstack, axis take/slice, masked reductions, and masked cross entropy. |
| [calibration-autotune.md](calibration-autotune.md#core-distinction) | Deep calibration/autotune guide covering families, parameters, candidate values, persistence, progress, and reports. |
| [mechanisms.md](mechanisms.md#graph-construction) | Mechanism-oriented guide using a repeated problem/mental-model/walkthrough/example format. |
| [modules.md](modules.md#package-map) | Source module map with responsibilities, inputs, outputs, dependencies, invariants, and failure modes. |
| [public-api.md](public-api.md#stability-map) | Public and probably-internal Java APIs, signatures, examples, side effects, and risks. |
| [configuration.md](configuration.md#build-requirements) | Build/runtime requirements, compile/runtime/profile configuration, CLI options, and native library lookup. |
| [examples.md](examples.md#running-examples) | Small examples for tensors, execution profiles, prepared execution, and tuning flows. |
| [development.md](development.md#local-setup) | Local development setup, build/test commands, package conventions, and maintenance workflow. |
| [testing.md](testing.md#test-framework-and-setup) | Test structure, focused test commands, expensive tests, and test failure interpretation. |
| [troubleshooting.md](troubleshooting.md#java-heap-space) | Symptom-driven debugging guide for compile, runtime, backend, tuning, and performance issues. |
| [glossary.md](glossary.md#a) | Project-specific terminology with source references. |
| [release.md](release.md#release-process) | Public-preview versioning, required release files, verification gates, artifact hygiene, tagging, and license boundary. |
| [documentation-audit.md](documentation-audit.md#audit-scope) | Documentation inventory, source-of-truth map, terminology baseline, stale-risk areas, example policy, and verification procedure. |

## Release Notes

- Current version source of truth: [`VERSION`](../VERSION).
- Current changelog: [`CHANGELOG.md`](../CHANGELOG.md).
- Release process: [release.md](release.md#release-process).
- License status: [`LICENSE.md`](../LICENSE.md).

## Source Documentation

Several source packages also contain package-local documentation. These are useful when editing a specific subsystem:

- `src/main/java/tensor/README.md` and `src/main/java/tensor/API.md`
- `src/main/java/operations/README.md`
- `src/main/java/graph/README.md`
- `src/main/java/graph/optimizer/README.md`, `AR.md`, `CSE.md`, `FUSE.md`, and `MEM.md`
- `src/main/java/backend/README.md`
- `src/main/java/prepare/README.md`, `backend/lowering/README.md`, and `backend/partition/README.md`
- `src/main/java/tuning/README.md`, `ARCHITECTURE.md`, `KNOBS.md`, `PERSISTENCE.md`, `REPORTING.md`, `SEARCH.md`, and `WORKLOADS.md`
- `src/main/java/numerics/README.md`

## Verification Notes

The docs intentionally mark machine-specific or source-incomplete claims as `Needs verification`. Typical examples are native Metal/CUDA runtime availability, missing CUDA native build instructions, direct `java` classpath workflows, and machine-dependent vector widths. These markers are not placeholders for missing writing; they are explicit boundaries where the repository cannot prove the claim without a local runtime environment.

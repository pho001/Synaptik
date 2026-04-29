<!-- generated-by: gsd-doc-writer -->
# Documentation Index

Navigation: [README](../README.md#reading-guide) | [Architecture](architecture.md#system-overview) | [Tensor API](tensor-api.md#api-surface-and-conventions) | [Adding Tensor Operation](adding-tensor-operation.md#implementation-checklist) | [Compute Flow](compute-flow.md#lifecycle-map) | [Graph Optimizer](graph-optimizer.md#stage-ordering) | [Native Bridges & BLAS](native-bridges-and-blas.md#term-map-at-a-glance) | [Metal Backend](metal-backend.md#end-to-end-flow) | [Calibration & Autotune](calibration-autotune.md#core-distinction) | [Public API](public-api.md#stability-map) | [Examples](examples.md#running-examples)

Chapters: [Recommended Reading Paths](#recommended-reading-paths) | [Document Map](#document-map) | [Source Documentation](#source-documentation) | [Verification Notes](#verification-notes)

This directory is the implementation-grounded documentation set for Synaptik. It is written for maintainers and technically strong readers who want to understand both how to use the framework and how this kind of compiled tensor runtime works internally.

## Table Of Contents

- [Recommended Reading Paths](#recommended-reading-paths)
- [Document Map](#document-map)
- [Source Documentation](#source-documentation)
- [Verification Notes](#verification-notes)

## Recommended Reading Paths

### New maintainer path

1. [Architecture: System Overview](architecture.md#system-overview) - the layered system model, package boundaries, and lifecycle artifacts.
2. [Framework Concepts: Tensors As Graph Nodes](framework-concepts.md#tensors-as-graph-nodes) - core vocabulary and the mental model behind compiled tensor execution.
3. [Compute Flow: Lifecycle Map](compute-flow.md#lifecycle-map) - the detailed journey from `Tensor` graph construction through compile, prepare, execution, memory binding, and traces.
4. [Graph Optimizer: Stage Ordering](graph-optimizer.md#stage-ordering) - detailed explanations of `AR`, `CSE`, `PART`, `FUSE`, and `MEM`.
5. [Native Bridges & BLAS: Term Map At A Glance](native-bridges-and-blas.md#term-map-at-a-glance) - what BLAS/GEMM are, how Java FFM calls native libraries, and how OpenBLAS is selected.
6. [Modules: Package Map](modules.md#package-map) - package-by-package responsibilities, dependencies, invariants, and failure modes.

### Tensor API user path

1. [Tensor API Guide: API Surface And Conventions](tensor-api.md#api-surface-and-conventions) - operation-level API explanations with concrete input/output examples and calculations.
2. [Tensor API: Compute Convenience API](tensor-api.md#compute-convenience-api) - how to call `compute()`, `compute(CompileMode)`, `compute(ComputeOptions)`, `compute(ExecutionProfile)`, and prepared execution.
3. [Compute Flow: Tensor Compute API](compute-flow.md#tensor-compute-api) - what happens internally when `compute(...)` resolves a profile, compiles, prepares, executes, and optionally runs generic graph autotune.
4. [Public API: Stability Map](public-api.md#stability-map) - the externally usable Java surfaces and probably-internal implementation hooks.
5. [Examples: Running Examples](examples.md#running-examples) - executable-style snippets for tensor operations, compile/prepare/execute, and tuning flows.
6. [Troubleshooting: Performance Regressions](troubleshooting.md#performance-regressions) - common runtime, compile, dtype, backend, and tuning failures.

### Tensor operation contributor path

1. [Adding A Tensor Operation: Implementation Checklist](adding-tensor-operation.md#implementation-checklist) - end-to-end implementation guide for new operations.
2. [Development: Adding Tensor Ops](development.md#adding-tensor-ops) - compact checklist and current package conventions.
3. [Modules: `tensor`](modules.md#tensor-public-graph-building-surface) and [Modules: `operations`](modules.md#operations-primitive-semantic-descriptors) - where public graph builders and descriptors live.
4. [Graph Optimizer: CSE](graph-optimizer.md#stage-cse-common-subexpression-elimination) and [Graph Optimizer: FUSE](graph-optimizer.md#stage-fuse-region-optimization-and-fusion) - what must be updated when an op has parameters or participates in fusion.
5. [Testing: Targeted Test Patterns](testing.md#targeted-test-patterns) - focused Gradle commands and test organization.

### Optimizer and runtime path

1. [Compute Flow: Compile](compute-flow.md#compile) - the lifecycle from semantic graph to prepared runtime execution.
2. [Graph Optimizer: Stage PART](graph-optimizer.md#stage-part-partition-planning) - stage-by-stage optimizer mechanics, including CPU natural regions, accelerator ownership regions, and Metal transfer-aware region scoring.
3. [Mechanisms: Prepared Execution](mechanisms.md#prepared-execution) - cross-cutting mechanisms such as graph construction, compilation, preparation, memory planning, and dispatch.
4. [Configuration: RuntimeConfig](configuration.md#runtimeconfig) - optimizer/runtime knobs, system properties, native lookup, and profile layout.

### Metal and accelerator debugging path

1. [Architecture: Accelerator Scaffolding](architecture.md#accelerator-scaffolding) - what the Metal/CUDA/OpenCL source layers contain and where the current Metal capability boundary is.
2. [Metal Backend: End-To-End Flow](metal-backend.md#end-to-end-flow) - detailed Java FFM, Objective-C shim, buffer ABI, storage residency, trace, and fallback mechanics.
3. [Architecture: Metal MPS Buffer Execution And Copy Chain](architecture.md#metal-mps-buffer-execution-and-copy-chain) - how tensor-array and buffer-binding transport differ.
4. [Compute Flow: Native buffer-binding Metal path](compute-flow.md#native-buffer-binding-metal-path) - how execution state keeps Metal outputs device-owned until materialization.
5. [Native Bridges & BLAS: How This Differs From Metal FFM](native-bridges-and-blas.md#how-this-differs-from-metal-ffm) - the shared Java FFM idea and the different execution models for BLAS and Metal.
6. [Graph Optimizer: Scored Candidate Planner Deep Dive](graph-optimizer.md#scored-candidate-planner-deep-dive) - how Metal partition scoring accounts for input bytes, output bytes, and avoided intermediate bytes.
7. [Calibration & Graph Autotune: Built-in workload catalogs](calibration-autotune.md#built-in-workload-catalogs) - transformer shape presets for stressing larger attention and FFN workloads.

### Calibration and autotune path

1. [Calibration & Graph Autotune: Core Distinction](calibration-autotune.md#core-distinction) - calibration families, owned knobs, candidate values, graph autotune parameters, persistence, and progress.
2. [Configuration: Tuning And Calibration Persistence](configuration.md#tuning-and-calibration-persistence) - profile persistence paths and command-line configuration.
3. [Testing: Targeted Test Patterns](testing.md#targeted-test-patterns) - relevant tests and how to run focused validation.
4. Source package docs under `src/main/java/tuning/`, especially `ARCHITECTURE.md`, `KNOBS.md`, `PERSISTENCE.md`, `SEARCH.md`, and `WORKLOADS.md`.

## Document Map

| Document | Purpose |
|---|---|
| [architecture.md](architecture.md#system-overview) | High-level architecture, lifecycle boundaries, package responsibilities, backend dispatch, and extension points. |
| [framework-concepts.md](framework-concepts.md#tensors-as-graph-nodes) | First-principles mental models for tensors, semantic graphs, compiled graphs, prepared execution, backend policy, and tuning. |
| [compute-flow.md](compute-flow.md#lifecycle-map) | Deep end-to-end walkthrough from graph building to `Tensor.compute(...)`, compile, prepare, execution, traces, and reuse rules. |
| [graph-optimizer.md](graph-optimizer.md#stage-ordering) | Deep explanation of optimizer configuration, state, trace, and every optimizer stage. |
| [native-bridges-and-blas.md](native-bridges-and-blas.md#term-map-at-a-glance) | Explanation of BLAS/GEMM, OpenBLAS dispatch, Java FFM bridges, native lookup, fallbacks, and performance tradeoffs. |
| [metal-backend.md](metal-backend.md#end-to-end-flow) | Detailed Metal backend guide covering planner legality, Java FFM, Objective-C MPS shim, native buffer ABI, residency, traces, and fallbacks. |
| [adding-tensor-operation.md](adding-tensor-operation.md#implementation-checklist) | Contributor guide for adding a new tensor operation through descriptors, builders, public API, CPU kernels, autograd, optimizer/fusion integration, docs, and tests. |
| [tensor-api.md](tensor-api.md#api-surface-and-conventions) | Detailed public Tensor API guide with signatures, `compute(...)` options, edge cases, and value-level operation examples. |
| [calibration-autotune.md](calibration-autotune.md#core-distinction) | Deep calibration/autotune guide covering families, parameters, candidate values, persistence, progress, and reports. |
| [mechanisms.md](mechanisms.md#graph-construction) | Mechanism-oriented guide using a repeated problem/mental-model/walkthrough/example format. |
| [modules.md](modules.md#package-map) | Source module map with responsibilities, inputs, outputs, dependencies, invariants, and failure modes. |
| [public-api.md](public-api.md#stability-map) | Public and probably-internal Java APIs, signatures, examples, side effects, and risks. |
| [configuration.md](configuration.md#build-requirements) | Build/runtime requirements, optimizer/runtime/profile configuration, CLI options, and native library lookup. |
| [examples.md](examples.md#running-examples) | Small examples for tensors, execution profiles, prepared execution, and tuning flows. |
| [development.md](development.md#local-setup) | Local development setup, build/test commands, package conventions, and maintenance workflow. |
| [testing.md](testing.md#test-framework-and-setup) | Test structure, focused test commands, expensive tests, and test failure interpretation. |
| [troubleshooting.md](troubleshooting.md#java-heap-space) | Symptom-driven debugging guide for compile, runtime, backend, tuning, and performance issues. |
| [glossary.md](glossary.md#a) | Project-specific terminology with source references. |

## Source Documentation

Several source packages also contain package-local documentation. These are useful when editing a specific subsystem:

- `src/main/java/tensor/README.md` and `src/main/java/tensor/API.md`
- `src/main/java/operations/README.md`
- `src/main/java/graph/README.md`
- `src/main/java/graph/optimizer/README.md`, `AR.md`, `CSE.md`, `FUSE.md`, and `MEM.md`
- `src/main/java/backend/README.md`
- `src/main/java/backend/prepare/README.md`, `backend/lowering/README.md`, and `backend/partition/README.md`
- `src/main/java/tuning/README.md`, `ARCHITECTURE.md`, `KNOBS.md`, `PERSISTENCE.md`, `REPORTING.md`, `SEARCH.md`, and `WORKLOADS.md`
- `src/main/java/numerics/README.md`

## Verification Notes

The docs intentionally mark machine-specific or source-incomplete claims as `Needs verification`. Typical examples are native Metal/CUDA runtime availability, missing CUDA native build instructions, direct `java` classpath workflows, and machine-dependent vector widths. These markers are not placeholders for missing writing; they are explicit boundaries where the repository cannot prove the claim without a local runtime environment.

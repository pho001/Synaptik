<!-- generated-by: gsd-doc-writer -->
# Documentation Index

Navigation: [README](../README.md) | [Architecture](architecture.md) | [Tensor API](tensor-api.md) | [Compute Flow](compute-flow.md) | [Graph Optimizer](graph-optimizer.md) | [Calibration & Autotune](calibration-autotune.md) | [Public API](public-api.md) | [Examples](examples.md)

Chapters: [Recommended Reading Paths](#recommended-reading-paths) | [Document Map](#document-map) | [Source Documentation](#source-documentation) | [Verification Notes](#verification-notes)

This directory is the implementation-grounded documentation set for Synaptik. It is written for maintainers and technically strong readers who want to understand both how to use the framework and how this kind of compiled tensor runtime works internally.

## Table Of Contents

- [Recommended Reading Paths](#recommended-reading-paths)
- [Document Map](#document-map)
- [Source Documentation](#source-documentation)
- [Verification Notes](#verification-notes)

## Recommended Reading Paths

### New maintainer path

1. [Architecture](architecture.md) - the layered system model, package boundaries, and lifecycle artifacts.
2. [Framework Concepts](framework-concepts.md) - core vocabulary and the mental model behind compiled tensor execution.
3. [Compute Flow](compute-flow.md) - the detailed journey from `Tensor` graph construction through compile, prepare, execution, memory binding, and traces.
4. [Graph Optimizer](graph-optimizer.md) - detailed explanations of `AR`, `CSE`, `PART`, `FUSE`, and `MEM`.
5. [Modules](modules.md) - package-by-package responsibilities, dependencies, invariants, and failure modes.

### Tensor API user path

1. [Tensor API Guide](tensor-api.md) - operation-level API explanations with concrete input/output examples and calculations.
2. [Tensor API: Compute Convenience API](tensor-api.md#compute-convenience-api) - how to call `compute()`, `compute(CompileMode)`, `compute(ComputeOptions)`, `compute(ExecutionProfile)`, and prepared execution.
3. [Compute Flow: Tensor Compute API](compute-flow.md#tensor-compute-api) - what happens internally when `compute(...)` resolves a profile, compiles, prepares, executes, and optionally runs generic graph autotune.
4. [Public API](public-api.md) - the externally usable Java surfaces and probably-internal implementation hooks.
5. [Examples](examples.md) - executable-style snippets for tensor operations, compile/prepare/execute, and tuning flows.
6. [Troubleshooting](troubleshooting.md) - common runtime, compile, dtype, backend, and tuning failures.

### Optimizer and runtime path

1. [Compute Flow](compute-flow.md) - the lifecycle from semantic graph to prepared runtime execution.
2. [Graph Optimizer](graph-optimizer.md) - stage-by-stage optimizer mechanics, including CPU natural regions, accelerator ownership regions, and Metal transfer-aware region scoring.
3. [Mechanisms](mechanisms.md) - cross-cutting mechanisms such as graph construction, compilation, preparation, memory planning, and dispatch.
4. [Configuration](configuration.md) - optimizer/runtime knobs, system properties, native lookup, and profile layout.

### Metal and accelerator debugging path

1. [Architecture: Accelerator Scaffolding](architecture.md#accelerator-scaffolding) - what the Metal/CUDA/OpenCL source layers contain and where the current Metal capability boundary is.
2. [Architecture: Metal MPS Copy Chain](architecture.md#metal-mps-copy-chain) - why the current Metal bridge is still copy-based and which bridge stats are reported.
3. [Compute Flow: Traces](compute-flow.md#traces) - how run traces expose Metal fallback, transfer timings, and storage residency.
4. [Graph Optimizer: Scored Candidate Planner Deep Dive](graph-optimizer.md#scored-candidate-planner-deep-dive) - how Metal partition scoring accounts for input bytes, output bytes, and avoided intermediate bytes.
5. [Calibration & Graph Autotune: Built-in workload catalogs](calibration-autotune.md#built-in-workload-catalogs) - transformer shape presets for stressing larger attention and FFN workloads.

### Calibration and autotune path

1. [Calibration & Graph Autotune](calibration-autotune.md) - calibration families, owned knobs, candidate values, graph autotune parameters, persistence, and progress.
2. [Configuration](configuration.md) - profile persistence paths and command-line configuration.
3. [Testing](testing.md) - relevant tests and how to run focused validation.
4. Source package docs under `src/main/java/tuning/`, especially `ARCHITECTURE.md`, `KNOBS.md`, `PERSISTENCE.md`, `SEARCH.md`, and `WORKLOADS.md`.

## Document Map

| Document | Purpose |
|---|---|
| [architecture.md](architecture.md) | High-level architecture, lifecycle boundaries, package responsibilities, backend dispatch, and extension points. |
| [framework-concepts.md](framework-concepts.md) | First-principles mental models for tensors, semantic graphs, compiled graphs, prepared execution, backend policy, and tuning. |
| [compute-flow.md](compute-flow.md) | Deep end-to-end walkthrough from graph building to `Tensor.compute(...)`, compile, prepare, execution, traces, and reuse rules. |
| [graph-optimizer.md](graph-optimizer.md) | Deep explanation of optimizer configuration, state, trace, and every optimizer stage. |
| [tensor-api.md](tensor-api.md) | Detailed public Tensor API guide with signatures, `compute(...)` options, edge cases, and value-level operation examples. |
| [calibration-autotune.md](calibration-autotune.md) | Deep calibration/autotune guide covering families, parameters, candidate values, persistence, progress, and reports. |
| [mechanisms.md](mechanisms.md) | Mechanism-oriented guide using a repeated problem/mental-model/walkthrough/example format. |
| [modules.md](modules.md) | Source module map with responsibilities, inputs, outputs, dependencies, invariants, and failure modes. |
| [public-api.md](public-api.md) | Public and probably-internal Java APIs, signatures, examples, side effects, and risks. |
| [configuration.md](configuration.md) | Build/runtime requirements, optimizer/runtime/profile configuration, CLI options, and native library lookup. |
| [examples.md](examples.md) | Small examples for tensors, execution profiles, prepared execution, and tuning flows. |
| [development.md](development.md) | Local development setup, build/test commands, package conventions, and maintenance workflow. |
| [testing.md](testing.md) | Test structure, focused test commands, expensive tests, and test failure interpretation. |
| [troubleshooting.md](troubleshooting.md) | Symptom-driven debugging guide for compile, runtime, backend, tuning, and performance issues. |
| [glossary.md](glossary.md) | Project-specific terminology with source references. |

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

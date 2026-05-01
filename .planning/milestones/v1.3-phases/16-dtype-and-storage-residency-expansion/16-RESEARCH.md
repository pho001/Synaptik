# Phase 16 Research: DType And Storage Residency Expansion

**Phase:** 16 - DType And Storage Residency Expansion
**Date:** 2026-05-01
**Status:** Complete

## RESEARCH COMPLETE

## Planning Question

What needs to be known to plan Phase 16 well?

Phase 16 should close dtype/storage residency gaps that force GPU regions back through CPU-readable arrays. The right unit of work is not broad native arithmetic. The right unit is making runtime memory binding, accelerator buffer metadata, capability diagnostics, and reports accurately represent `BFLOAT16`, `INT32`, and `BOOL` values when those values are allowed to remain resident.

## Current Architecture

### Runtime Memory Binding

`RuntimeMemoryBinder` already binds reusable runtime region slots for `FLOAT64` and `FLOAT32`. It creates one Java array per reusable region slot and points compatible runtime tensors at that shared storage. The same method explicitly no-ops for `BFLOAT16`, `INT32`, and `BOOL`.

The tensor API already has storage-specific support:

- `Tensor.setData(short[])` creates `BFLOAT16` storage.
- `Tensor.setData(int[])` creates `INT32` storage.
- `Tensor.setData(byte[])` creates `BOOL` storage.
- `Tensor.getBFloat16Data()`, `Tensor.getInt32Data()`, and `Tensor.getBoolData()` expose mutable typed arrays.

This means Phase 16 can add typed region slots without adding a public device tensor API.

### Accelerator Buffer Metadata

`AcceleratorBufferLayout` and `ExecutionState` already understand byte sizes for `BFLOAT16`, `INT32`, and `BOOL`. The missing piece is not layout byte length. The gap is that backend residency decisions and runtime binding do not yet consistently expose whether a non-floating dtype can stay device-owned, must split a region, or must materialize at a true CPU boundary.

### Backend Capability Boundaries

Metal currently supports `FLOAT32` compute/output and `BOOL` only for predicate-style external inputs. `MetalMpsCapabilities.unsupportedDTypeMessage(...)` already provides stable user-facing detail.

CUDA buffer allocation and binder paths are currently dense `FLOAT32` oriented. CUDA FFM dtype code handling includes limited dtype awareness, but buffer allocation and executable integration do not provide broad non-`FLOAT32` native residency.

The plan should preserve those backend contracts. It should make rejection explicit rather than silently materializing.

### Lowered Region Manifest And Coverage

Phase 15 added a Java-side `GpuLoweredRegionManifest` with dtype/layout/storage assumptions, lowered primitives, and rejection metadata. Phase 16 should enrich the assumptions and reports with dtype residency evidence instead of adding parallel diagnostics.

Phase 14 marked `mlp_classifier_small` and `layer_norm_small` as `GPUSTORAGE` targets. These are the preferred workloads for evidence, with synthetic tests used where native Metal/CUDA capability is unavailable.

## Gaps To Close

### Gap 1: Typed Runtime Slots Stop At FLOAT32/FLOAT64

Reusable memory slots for `BFLOAT16`, `INT32`, and `BOOL` currently fall back to per-tensor storage. This blocks storage-residency tests and makes region slot reuse incomplete.

Recommended shape:

- Add `Map<Integer, short[]>`, `Map<Integer, int[]>`, and `Map<Integer, byte[]>` in `RuntimeMemoryBinder`.
- Extend `bindTypedStorage(...)` to call `runtimeTensor.setData(short[])`, `runtimeTensor.setData(int[])`, and `runtimeTensor.setData(byte[])`.
- Keep alias/view and workspace-sensitive skips unchanged.

### Gap 2: DType Residency Decisions Are Scattered

Metal and CUDA have dtype checks, but Phase 16 needs a shared decision vocabulary for residency diagnostics. A small backend-neutral helper can classify role-specific legality and keep detail strings stable without taking ownership of backend execution.

Recommended shape:

- Add an internal accelerator dtype residency decision record/policy under `backend.accelerator`.
- Inputs: backend, role, dtype, layout class, and capability detail.
- Output: supported/resident/rejected plus `UNSUPPORTED_DTYPE` and backend-specific detail.
- Metal/CUDA adapters call into or mirror this contract in tests.

### Gap 3: Reports Do Not Highlight DType Residency Exits

Coverage reports already count materializations and storage residency, but they do not make dtype-specific residency blockers easy to audit.

Recommended shape:

- Extend manifest assumptions or backend extensions with dtype residency role evidence.
- Extend coverage summary/report renderers with dtype materialization/rejection counters only where data exists.
- Add tests proving `BFLOAT16`, `INT32`, and `BOOL` unsupported/native-limited paths use stable strings.

## Recommended Plan Shape

1. Runtime typed slot binding for `BFLOAT16`, `INT32`, and `BOOL`.
2. Backend-neutral dtype residency/capability diagnostics for Metal and CUDA.
3. Trace/report evidence for dtype-related residency, materialization, and rejection.
4. Docs, validation, and hygiene closure.

## Validation Architecture

### Automated Sampling

- After runtime binding changes, run `./gradlew test --tests graph.execution.RuntimeMemoryBinderTest`.
- After accelerator dtype residency decisions, run focused tests for `AcceleratorBufferLayoutClassifierTest`, Metal capability tests, CUDA binder/region lowerer tests, and any new dtype residency policy test.
- After trace/report changes, run `./gradlew test --tests CompiledGraphTraceTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest`.
- Before phase verification, run `./gradlew classes` and the full Phase 16 focused slice.

### Nyquist Targets

- Every plan must include automated verification.
- No three consecutive implementation tasks may rely only on manual inspection.
- Native CUDA and Metal execution remains capability-gated; portable synthetic tests must prove diagnostics when hardware is unavailable.
- `git status --short` must show only unrelated local `profiles/platform/.../tuning/abc/*` artifacts unstaged.

## Research Risks

- Risk: treating dtype residency as native compute support. Mitigation: capability decisions must separate storage representability, input legality, output legality, and compute legality.
- Risk: adding public GPU tensor concepts. Mitigation: keep all residency state inside runtime execution and accelerator internals.
- Risk: silent CPU materialization remains hidden. Mitigation: trace/report tests assert dtype-specific reason evidence.
- Risk: CPU memory reuse regression. Mitigation: keep `FLOAT32`/`FLOAT64` tests and add dtype slot reuse parity tests.

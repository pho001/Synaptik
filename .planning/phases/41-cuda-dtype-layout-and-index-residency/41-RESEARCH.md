---
phase: 41
type: research
status: complete
requirements:
  - CUDADTYPE-01
  - CUDADTYPE-02
  - CUDAINDEX-01
---

# Phase 41 Research: CUDA DType Layout And Index Residency

## Phase Goal

Expand CUDA dtype/layout/index residency and forward indexing coverage without overclaiming unsupported compute.

## Current Evidence

- CUDA buffer binding is real but still conservative: `CudaAcceleratorBufferBinder` accepts dense `FLOAT32` metadata and rejects other input/output dtypes with `INPUT_DTYPE_UNSUPPORTED` / `OUTPUT_DTYPE_UNSUPPORTED`.
- CUDA dense layout materialization exists through `CudaDeviceLayoutMaterializer`, but it currently supports only dense `FLOAT32` targets.
- Shared layout propagation already distinguishes `METADATA_ONLY_VIEW`, `DENSE_GPU_MATERIALIZATION`, `BROADCAST_GPU_MATERIALIZATION`, `STRIDED_NATIVE_COMPUTE`, and `UNSUPPORTED`.
- CUDA lowering coverage currently marks forward `GATHER` and `TAKE_ALONG_AXIS` as `CAPABILITY_MISSING`, while Metal supports a scoped dense `FLOAT32` value/output plus static in-bounds `INT32` index contract.
- Phase 40 added CUDA parity and capability reports, including dtype role, layout ABI, DAG primitive, vendor-library, and hot-path blocker dimensions.

## Key Constraint

`dtype residency is not native dtype compute`.

Phase 41 must allow CUDA to represent the roles required by selected regions, especially `INT32` index inputs and report-visible `BFLOAT16`/`BOOL` residency evidence, while keeping generic CUDA native compute/output unsupported unless execution evidence exists.

## Implementation Direction

1. Add CUDA-specific dtype role policy, not generic dtype widening.
   - `FLOAT32`: compute/input/output supported for existing dense buffer path.
   - `INT32`: admissible only as index-input role for forward gather/take candidate analysis.
   - `BOOL`: admissible only as predicate/external-mask residency evidence where a later CUDA primitive can consume it; no BOOL-producing CUDA output claim.
   - `BFLOAT16`: residency/report role only unless a later CUDA compute primitive proves support.
   - `FLOAT64`: remains unsupported for CUDA native buffer compute/output.

2. Make CUDA layout routing explain every decision.
   - Metadata-only views may keep a CUDA binding if the consumer can still be made legal.
   - Dense GPU materialization may repair supported `FLOAT32` layouts where the native bridge supports it.
   - Broadcast/zero-stride and arbitrary strided compute must remain explicit rejection unless native support exists.
   - Report fields must distinguish residency, layout repair, and CPU materialization.

3. Handle forward gather/take conservatively.
   - Preferred path: implement scoped CUDA forward `GATHER` and `TAKE_ALONG_AXIS` only if the native shim and Java lowering can provide CPU parity.
   - Acceptable Phase 41 closure: stable CUDA rejection after validating dtype, layout, rank/axis, bounds, and adjacent producer residency, so the report explains why the region shortened.
   - Do not promote gather/take support without native buffer execution or an explicitly routed primitive.

## Verification Targets

- CUDA dtype role tests prove role-specific acceptance/rejection without widening compute support.
- CUDA layout materializer and device-flow tests prove metadata-only/dense-materialization/unsupported decisions.
- CUDA gather/take tests prove either scoped execution or stable rejection detail for dtype/layout/bounds/rank.
- Coverage reports expose CUDA dtype residency, layout transform kinds, and index blocker evidence.

## Risk Notes

- Promoting `INT32` input residency as generic `INT32` compute would corrupt coverage truth.
- Treating CUDA capability skip as support would invalidate Phase 40 parity discipline.
- Hidden CPU materialization between a CUDA producer and unsupported index/layout consumer would make reports misleading.
- Native CUDA environments may capability-skip locally; portable tests must still verify policy and rejection behavior.


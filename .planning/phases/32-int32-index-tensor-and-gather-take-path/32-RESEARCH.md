# Phase 32 Research: INT32 Index Tensor And Gather Take Path

## Current Codebase Facts

- Public `Tensor` already supports `INT32` storage through `Int32Storage`, constructors, and `getInt32Data()`.
- CPU gather/take execution exists under `backend.cpu.kernels.index` and supports multiple value dtypes, including `FLOAT32`, `BFLOAT16`, `BOOL`, and `INT32`.
- `TensorIndexOps.gather(...)` removes the gathered axis; `takeAlongAxis(...)` returns the index tensor shape and requires matching non-axis dimensions.
- The accelerator coverage matrix currently marks `GATHER` and `TAKE_ALONG_AXIS` as `DAG_PRIMITIVE_UNSUPPORTED` for Metal/CUDA because INT32 index residency is not native index compute yet.
- Metal dtype ABI v3 can name `INT32`, and common accelerator buffer layout byte sizing already knows `INT32`.
- Metal capability truth currently treats `INT32` compute/output as unsupported. Phase 32 should keep that true for generic compute while allowing index-role external inputs.

## Implementation Approach

1. Add role-specific INT32 index input legality before native execution:
   - `MetalMpsCapabilities` should distinguish generic external input from `GATHER`/`TAKE_ALONG_AXIS` index input role.
   - `AcceleratorDTypeResidencyPolicy` and trace evidence should show `INT32` index-input residency without implying native INT32 arithmetic.
   - `MetalBufferAllocator`, binder, bridge validation, and materializer checks should support INT32 input bindings where the executable descriptor expects INT32.

2. Add shared DAG contract for forward index ops:
   - Add `GATHER` and `TAKE_ALONG_AXIS` ABI node types after the current BOOL op codes.
   - Lower operation axis metadata through existing node scalar metadata or a small structured metadata addition.
   - Keep backward/scatter rows unsupported with explicit duplicate-index reasons.

3. Add native Metal execution:
   - Prefer MPSGraph native gather/take-style primitives if the local SDK exposes stable methods and `metalTest` proves the path.
   - If MPSGraph API shape differs, implement only the minimal legal scoped cases or keep support disabled with a stable capability reason rather than overclaiming.
   - Value output dtype should stay tied to the input value dtype; start with `FLOAT32` if BF16/BOOL value gather adds too much parity risk.

4. Prove semantics:
   - Axis normalization, output shape, bounds behavior, and index dtype handling must match CPU.
   - Non-dense value/index tensors should initially reject unless the GPU layout router already produces an accepted dense materialization path.
   - Duplicate index behavior matters only for backward/scatter, so forward duplicates are legal if CPU gather/take permits them.

5. Harden reports:
   - Add coverage target/gate for a representative gather/take flow if existing `cross_entropy_small` remains loss-index fallback-focused.
   - Gate native Metal evidence on `BUFFER_BINDING`, `dtype=INT32` residency evidence, lowered primitive count, selected region length, zero CPU fallback, zero tensor-array fallback, and no unexpected CPU materialization.
   - CUDA expectations remain visible blockers.

## Risks

- MPSGraph index primitive availability or semantics may differ from CPU gather/take semantics. Mitigation: native `metalTest` parity first; keep unsupported if not proven.
- Index bounds behavior can silently diverge. Mitigation: parity tests for valid indices and explicit rejection/failure tests for out-of-bounds cases.
- INT32 residency could be misread as INT32 compute/output support. Mitigation: role-specific capability names, docs, and coverage evidence.
- Supporting non-dense index/value tensors too early may hide CPU layout materialization. Mitigation: reject non-dense initially and leave broad repair to Phase 33.

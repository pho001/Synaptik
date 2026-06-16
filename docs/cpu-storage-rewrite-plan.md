# CPU Storage Rewrite Plan

This plan scopes the `backend.cpu` storage rewrite. It is an implementation guide, not a historical baseline.

## Goals

- Make regular CPU kernels consume the existing `CpuKernelCall` / `CpuStorageBindings` contract instead of reaching
  directly for `TensorInternalAccess` in hot paths.
- Keep Java-array and CPU-native `MemorySegment` storage under the same operation-family ownership.
- Preserve concrete per-kernel hot loops where they protect JIT inlining, Vector API use, BF16 continuation behavior,
  or provider-specific dispatch.
- Keep fallback visible in runtime traces and benchmark reports.
- Avoid adding a new generic CPU dispatcher, transitional native facade, or speculative template hierarchy.

## Non-Goals

- Do not rewrite `backend.cpu.fused` as part of this plan. Fused execution has its own IR, numeric contract,
  generated/interpreted executables, ASM support, and `FusedNativeSegmentBindings`.
- Do not promote non-BLAS `MemorySegment` scalar loops to default `AUTO` hot paths without benchmark proof.
- Do not make legacy/direct-test gradient descriptors first-class rewrite targets unless a separate task proves they are
  production-generated or performance-critical.
- Do not replace OpenBLAS provider boundaries with generic storage loops.

## Core Contract

`CpuKernelExecutor` already binds runtime storage and passes it to kernels:

```text
CpuStorageResolver -> CpuStorageBindings -> CpuKernelCall -> CpuKernel
```

The rewrite should close the gap between this contract and existing kernels that still use `TensorInternalAccess`
directly. The preferred shape is:

1. operation-family executor reads `CpuStorageView` values from `CpuKernelCall`
2. family-local routing decides array vs `MemorySegment`, dtype, dense vs strided, and fallback
3. concrete kernels keep the inner compute loops when that preserves monomorphic code and vectorization

Shared helpers may own boilerplate such as range splitting, storage validation, trace publication, and typed reads/writes.
They should not own the arithmetic operation as a virtual callback inside a shared hot loop.

## Wave Order

### 0. Contract Plumbing

Scope:

- `backend.cpu.kernels.CpuKernel`
- `backend.cpu.kernels.TypedCpuKernel`
- `backend.cpu.kernels.CpuKernelCall`
- `backend.cpu.execution.CpuKernelExecutor`
- `backend.cpu.storage`

Work:

- Decide how typed kernels receive `CpuStorageBindings`; the current `TypedCpuKernel.execute(...)` unwraps tensors,
  output dtype, and context but does not expose storage views to `forward*` methods.
- Add a narrow, non-speculative path for family executors to access input/output storage views.
- Keep public `Tensor` API logical; backend residency stays in runtime execution state.

Validation:

- `./gradlew classes`
- focused kernel registry and architecture tests after API shape changes

### 1. Elementwise Dense Families

Scope:

- `backend.cpu.kernels.elementwise.binary`
- `backend.cpu.kernels.elementwise.unary`
- `backend.cpu.kernels.elementwise.compare`
- `backend.cpu.kernels.elementwise.logical`
- `backend.cpu.kernels.elementwise.where`
- shared elementwise support under `backend.cpu.kernels.elementwise`

Work:

- Migrate fallback array paths away from the monolithic `ElementwiseLoops` entrypoints.
- Keep concrete direct loops in kernels such as add/mul/sub/div/min/max where they currently protect vectorized hot paths.
- Treat compare/logical/where as distinct storage families; they have BOOL outputs or mixed BOOL/numeric inputs.
- Preserve BF16 float-continuation publish/consume behavior.

Validation:

- representative elementwise tests
- `CpuKernelFamilyArchitectureTest`
- hot-path sanity for F32/F64/BF16 add when touching direct loops

### 2. Strided Elementwise

Scope:

- `backend.cpu.kernels.elementwise.strided`
- `backend.cpu.nativecpu.layout` only where existing strided `MemorySegment` helpers are still the owner

Work:

- Make the current pre-kernel strided branch explicit in the rewrite plan. `CpuKernelExecutor` routes
  `plan.stridedPath()` to `CpuStridedElementWise` before normal storage binding reaches the selected kernel.
- Decide whether existing `NativeSegmentStridedKernels` remain native layout support or move under elementwise family
  ownership.
- Keep correctness-first scalar segment strided paths guarded by runtime policy.

Validation:

- strided elementwise tests
- native segment strided tests where available

### 3. Index Families

Scope:

- `backend.cpu.kernels.index`

Work:

- Rewrite read-only operations first: `GATHER`, `GATHER_AXIS`, `GATHER_ND`, `TAKE_ALONG_AXIS`.
- Rewrite write/accumulation operations second: `SCATTER_ADD`, `SCATTER_AXIS_ADD`, `SCATTER_ELEMENTS`, `SCATTER_ND`.
- Extend index reading for both Java arrays and `MemorySegment` with `INT32` / `INT64` index tensors.
- Keep duplicate-index accumulation semantics deterministic unless a specific parallel path proves disjoint writes.
- Make scatter/index ops storage-ready because Tensor-level backward paths depend on them.

Validation:

- index execution tests
- duplicate-index accumulation tests
- accelerator parity tests only if public support claims are touched

### 4. Layout And Materialization

Scope:

- `backend.cpu.kernels.layout`

Work:

- Migrate materialization operations: `CAST`, `CONTIGUOUS`, `CONCAT`, `PAD`, `TILE`, `PERMUTE`.
- Migrate layout carriers needed by Tensor-level backward, especially `PAD` and `SLICE_BACKWARD`.
- Preserve view-only alias behavior for `RESHAPE`, `EXPAND`, `SELECT`, `SLICE`, `EXPAND_DIMS`, and `SQUEEZE`.
- Replace array-only copy assumptions such as `TensorRemap` or `copyDataFrom` only where runtime storage can be native.

Validation:

- layout/index shape and execution tests
- native view/contiguous/cast tests

### 5. Reduction Families

Scope:

- `backend.cpu.kernels.reduction`

Work:

- Split reduction work by semantics:
  - sum/mean
  - min/max/prod
  - cumsum/argmax
  - bool reductions
  - softmax/logSoftmax
  - dense and index-target losses
- Keep accumulation dtype and BF16 continuation behavior explicit.
- Keep loss/reduction storage work focused on public Tensor ops and measured optimizer specializations.

Validation:

- reduction and loss execution tests
- BF16 reduction/loss tests
- numerical tolerance tests for accumulation-sensitive paths

### 6. Linalg And Providers

Scope:

- `backend.cpu.kernels.linalg`
- `backend.cpu.kernels.linalg.matmul`
- `backend.cpu.provider.linalg.matmul`
- `backend.cpu.prepare.linalg`
- `backend.cpu.plan.linalg`

Work:

- Preserve OpenBLAS as a provider route, including array-copy and native-segment variants.
- Audit `LINEAR` bias and BF16 continuation paths for array-only assumptions.
- Keep attention work aligned with public Tensor/linalg DAGs unless a measured specialization is introduced.
- Do not fold provider logic into generic elementwise or reduction helpers.

Validation:

- matmul and linear tests
- OpenBLAS array-copy/native-segment focused tests where available
- attention execution tests

### 7. Neural-Network Kernels

Scope:

- `backend.cpu.kernels.nn`
- `backend.cpu.prepare.nn`
- `backend.cpu.plan.nn`

Work:

- Keep conv and pool backward on canonical Tensor DAG primitives:
  - conv gradients use window layout plus matrix multiplication primitives
  - pool gradients use window layout, argmax/scatter writes, and fold layout primitives
- Include forward NN kernels:
  - conv direct/GEMM
  - max/avg pool
  - layer norm
  - RMS norm
- Treat normalization kernels as their own storage problem; do not force them through elementwise templates.

Validation:

- conv, pool, layer norm, RMS norm execution tests
- GEMM lowering tests when conv lowering behavior is touched

### 8. Fused Audit Only

Scope:

- `backend.cpu.fused`
- `backend.cpu.kernels.fused`
- `backend.cpu.prepare.fused`
- `backend.cpu.plan.fused`

Work:

- Do not migrate fused through the regular storage rewrite.
- After regular kernels settle, audit that fused execution still binds and publishes native segment storage correctly.
- Verify no accelerator or generic backend code imports CPU fused internals.

Validation:

- fused dispatch tests
- source hygiene grep for forbidden fused imports

## Package Coverage Checklist

| Package | Disposition |
|---|---|
| `backend.cpu` root | Keep as orchestration/artifact/trace ownership. No hot-loop code. |
| `backend.cpu.execution` | Contract plumbing and storage binding integration. |
| `backend.cpu.storage` | Canonical storage view/binding layer. Extend only when required by real family consumers. |
| `backend.cpu.prepare` | Planning and materialization policy owner. Update only for explicit route/hint needs. |
| `backend.cpu.plan` | Prepared metadata and hints. Avoid runtime execution logic here. |
| `backend.cpu.kernels` root | Registry and shared kernel interfaces only. No concrete op entrypoints. |
| `backend.cpu.kernels.elementwise` | Main rewrite scope, split by binary/unary/compare/logical/where/strided. |
| `backend.cpu.kernels.index` | Main rewrite scope after elementwise, with scatter accumulation treated carefully. |
| `backend.cpu.kernels.layout` | Main rewrite scope for materialization and view semantics. |
| `backend.cpu.kernels.reduction` | Main rewrite scope with explicit accumulation policy. |
| `backend.cpu.kernels.linalg` | Rewrite/audit scope; preserve matmul provider boundary. |
| `backend.cpu.kernels.nn` | Rewrite scope for active NN forward/backward primitives. |
| `backend.cpu.kernels.fused` | Audit only. |
| `backend.cpu.fused` | Out of primary rewrite; audit only. |
| `backend.cpu.nativecpu` | Runtime storage, allocator, materializer, trace support. Do not recreate native executor stacks. |
| `backend.cpu.nativecpu.layout` | Existing segment strided support; decide ownership during strided wave. |
| `backend.cpu.provider` | Provider boundary; preserve OpenBLAS route semantics. |
| `backend.cpu.lowering`, `backend.cpu.partition`, `backend.cpu.region` | Planning/region audit only unless route metadata changes. |

## Verification Baseline

Use focused tests by wave. Before and after broad hot-path changes, run:

```bash
./gradlew classes
./gradlew test --tests CpuKernelFamilyArchitectureTest
```

For family-specific changes, prefer targeted tests over the full default suite when benchmark/debug tests would make the
run too slow.

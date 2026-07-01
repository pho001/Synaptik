# backend.cpu

`backend.cpu` is the owner root for CPU backend implementation.

Target layout:

- `backend.cpu.prepare` owns CPU node preparation.
- `backend.cpu.lowering` owns CPU partition lowering.
- `backend.cpu.partition` owns CPU partition legality and plans.
- `backend.cpu.kernels` owns CPU kernel resolution and runtime kernels.
- `backend.cpu.fused` owns fused planning, codegen, generated executable preparation, and generated ASM support.

CPU execution flow:

1. `backend.cpu.prepare.CpuNodePreparer` resolves the `CpuKernel`, `CpuNodeExecutionPlan`, dispatch hints, storage
   policy, fused executable, and provider-specific prepared executables.
2. `backend.cpu.CpuBackend` resolves runtime tensors and applies the prepared input/layout policy.
3. `backend.cpu.execution.CpuKernelExecutor` binds runtime storage views, builds `CpuKernelCall`, invokes the
   selected kernel, and returns `CpuKernelResult`.
4. The selected operation-family kernel executes directly over Java arrays, CPU-native `MemorySegment` storage, or a
   provider route such as OpenBLAS.
5. Runtime residency/materialization publishes whether Java array storage, native CPU storage, or device storage is
   current.

`MemorySegment` is a CPU storage kind, not a second CPU backend. Non-BLAS native segment execution belongs in the
same family packages as the Java-array kernels:

- `backend.cpu.kernels.elementwise` for unary/binary/where/compare/logical storage loops.
- `backend.cpu.kernels.reduction` for reductions.
- `backend.cpu.kernels.layout` for casts, contiguous materialization, and view aliases.
- `backend.cpu.kernels.fused` plus `backend.cpu.fused.exec` for fused segment execution and binding lifecycle.

Do not add transitional native CPU facades, generic native executor registries, or plan/facts/parity stacks beside the
family kernels. `CpuNativeStorageSupport` is a small support/trace vocabulary; it is not a dispatcher. `AUTO` native
storage must remain performance-gated: provider routes and metadata-only views are eligible by default, while
non-BLAS segment scalar kernels require explicit benchmark proof before promotion.

`backend.cpu.fused` intentionally remains separate from `backend.cpu.kernels.fused`:

- `backend.cpu.fused` prepares generated or planned fused execution artifacts.
- `backend.cpu.kernels.fused` executes direct runtime fused kernels.

Rewrite planning references:

- `docs/cpu-storage-rewrite-plan.md` defines the current storage rewrite scope, wave order, package checklist, and fused
  exclusion.
- `docs/cpu-kernels-wave0-baseline.md` is historical baseline evidence for earlier native CPU ownership cleanup.

Root-level CPU classes in `backend` and the old split CPU kernel tree have been removed.
Do not add new CPU implementation code outside `backend.cpu`.

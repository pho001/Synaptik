# CUDA backend (planned implementation)

## Outcome and status

This guide records the intended CUDA boundary. No CUDA Java or native implementation exists yet.

## Planned scope

`backends/cuda` will own capability reporting, CUDA-specific lowering, specialization and fusion, kernel selection, executable construction, device storage, transfers, workspaces, native integration, and typed traces.

```text
planning chooses CUDA ownership
  -> CUDA prepare chooses a concrete executable route
  -> runtime invokes the prepared executable
```

Planning must not reference CUDA kernel classes, and runtime must not discover CUDA or choose a kernel. Public `Tensor` state must not become CUDA residency state.

## Native and resource prerequisites

Driver, toolkit, device, compute-capability, and library requirements will be fixed by focused tasks and validated environments. Future contracts must specify context, stream, event, allocation, workspace, and native-handle ownership and cleanup. No version matrix is promised by this pre-implementation guide.

## Failures and validation

Unsupported work is rejected during capability analysis or preparation, not by runtime fallback. Future work requires architecture tests, backend-conformance and integration tests on supported hardware, cleanup/failure tests, and reproducible benchmarks.

See the [CUDA master plan](../planning/backends/cuda/master-plan.md), [partition preparer](partition-preparer.md), and [kernel routes](kernel-routes.md).

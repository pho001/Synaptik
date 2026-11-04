# CUDA backend (planned implementation)

## Outcome and status

This guide records the intended CUDA boundary. No CUDA Java or native implementation exists yet.

**CUDA** is NVIDIA's platform for programming supported graphics processing units (GPUs).
In the planned vocabulary below, the **host** runs Java and submits work, while the **device**
performs GPU computation. A **kernel** is a routine executed on the device. These definitions
explain the intended boundary; they do not announce a supported CUDA device or implementation.

## Planned scope

`modules/backend/cuda` will own capability reporting, CUDA-specific lowering, specialization and fusion, kernel selection, executable construction, device storage, transfers, workspaces, native integration, and typed traces.

```text
planning chooses CUDA ownership
  -> CUDA prepare chooses a concrete executable route
  -> runtime invokes the prepared executable
```

Planning must not reference CUDA kernel classes, and runtime must not discover CUDA or choose a kernel. Public `Tensor` state must not become CUDA residency state.

## Native and resource prerequisites

A driver lets host software communicate with the device; a toolkit supplies development tools
and libraries. NVIDIA **compute capability** identifies a device's architectural feature level,
not Synaptik's answer to whether an operation is supported. A **context** groups device execution
state, a **stream** orders submitted work, and an **event** marks progress for synchronization.
Their eventual ownership rules must determine when submitted work has finished and when its
buffers can safely be released.

Driver, toolkit, device, compute-capability, and library requirements will be fixed by focused tasks and validated environments. Future contracts must specify context, stream, event, allocation, workspace, and native-handle ownership and cleanup. No version matrix is promised by this pre-implementation guide.

## Failures and validation

Unsupported work is rejected during capability analysis or preparation, not by runtime fallback. Future work requires architecture tests, backend-conformance and integration tests on supported hardware, cleanup/failure tests, and reproducible benchmarks.

See the [CUDA master plan](../planning/backends/cuda/master-plan.md), [partition preparer](partition-preparer.md), and [kernel routes](kernel-routes.md).

# [CUDA](../glossary.md#term-cuda) [backend](../glossary.md#backend) (planned implementation)

## Outcome and status

This guide records the intended [CUDA](../glossary.md#term-cuda) boundary. No CUDA Java or native implementation exists yet.

**CUDA** is NVIDIA's platform for programming supported graphics processing units (GPUs).
In the planned vocabulary below, the **[host](../glossary.md#term-host)** runs Java and submits work, while the **[device](../glossary.md#term-device)**
performs [GPU](../glossary.md#term-gpu) computation. A **[kernel](../glossary.md#kernel)** is a routine executed on the device. These definitions
explain the intended boundary; they do not announce a supported CUDA device or implementation.

## Planned [scope](../glossary.md#term-scope)

`modules/backend/cuda` will own capability reporting, [CUDA](../glossary.md#term-cuda)-specific [lowering](../glossary.md#lowering), [specialization and fusion](../glossary.md#specialization-and-fusion), [kernel](../glossary.md#kernel) selection, executable construction, [device](../glossary.md#term-device) storage, transfers, workspaces, native integration, and typed traces.

```text
planning chooses CUDA ownership
  -> CUDA prepare chooses a concrete executable route
  -> runtime invokes the prepared executable
```

The intended callers are shared Prepare (passing an owned partition's descriptions and assigned
slots) and Runtime (invoking the completed executable with current representations). There is
no implemented CUDA callee to accept either handoff today. In particular, constructing an ID
named `"cuda"` does not load a driver, allocate device memory or transfer ownership.
The [implemented CPU/Metal handoff](../architecture/runtime-prepare-backend-boundary.md#start-with-one-value-crossing-the-boundary)
provides the comparison; it is not evidence that CUDA already has matching resource leases.

[Planning](../glossary.md#term-planning) must not reference CUDA kernel classes, and [runtime](../glossary.md#term-runtime) must not discover CUDA or choose a kernel. Public [`Tensor`](../glossary.md#tensor) state must not become CUDA [residency](../glossary.md#residency) state.

## Native and resource prerequisites

A driver lets [host](../glossary.md#term-host) software communicate with the [device](../glossary.md#term-device); a toolkit supplies development tools
and libraries. NVIDIA **[compute capability](../glossary.md#term-compute-capability)** identifies a device's architectural [feature](../glossary.md#term-software-feature) level,
not Synaptik's answer to whether an [operation](../glossary.md#operation) is supported. A **context** groups device execution
state, a **stream** orders submitted work, and an **event** marks progress for [synchronization](../glossary.md#term-synchronization).
Their eventual [ownership](../glossary.md#term-ownership-meanings) rules must determine when submitted work has finished and when its
[buffers](../glossary.md#term-buffer) can safely be released.

Driver, toolkit, device, compute-capability, and library requirements will be fixed by focused tasks and validated environments. Future contracts must specify context, stream, event, allocation, workspace, and native-handle ownership and cleanup. No version [matrix](../glossary.md#term-matrix-meanings) is promised by this pre-implementation guide.

## Failures and [validation](../glossary.md#term-validation)

Unsupported work is rejected during capability analysis or preparation, not by [runtime](../glossary.md#term-runtime) fallback. Future work requires architecture tests, [backend](../glossary.md#backend)-[conformance](../glossary.md#term-conformance) and integration tests on supported hardware, cleanup/failure tests, and reproducible benchmarks.

See the [CUDA master plan](../planning/backends/cuda/master-plan.md)<span data-glossary-navigation="true"> ([CUDA glossary](../glossary.md#term-cuda))</span>, [partition preparer](partition-preparer.md)<span data-glossary-navigation="true"> ([partition glossary](../glossary.md#partition))</span>, and [kernel routes](kernel-routes.md)<span data-glossary-navigation="true"> ([kernel glossary](../glossary.md#kernel))</span>.

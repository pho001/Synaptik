# backend.lowering

`backend.lowering` owns backend-neutral lowering contracts.

This package defines how optimized partitions become backend-owned execution artifacts, but it does not implement CPU, Metal, CUDA, or OpenCL policy.

Allowed responsibilities:

- lowering request/result contracts
- lowered-partition and lowered-unit records
- lowering pipeline orchestration
- backend capability and workspace requirement contracts

Not allowed:

- importing concrete backend packages such as `backend.cpu`, `backend.metal`, or `backend.cuda`
- selecting kernels
- compiling bridge executables
- rebuilding optimizer or partition artifacts

Backend-specific lowering belongs under the backend root.

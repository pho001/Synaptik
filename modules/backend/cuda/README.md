# [CUDA](../../../docs/glossary.md#term-cuda) [backend](../../../docs/glossary.md#backend) scaffold

**CUDA** is NVIDIA's programming platform for computations on its GPUs. A **marker project** keeps
a named project in the repository/build [layout](../../../docs/glossary.md#layout) but supplies no usable execution implementation.
The placeholder below records [planning](../../../docs/glossary.md#term-planning) intent, not a backend that an application can select.

The `:backends:cuda` project lives in `modules/backend/cuda`. It is a marker only; no executable
CUDA backend, native integration, or platform directory is implemented. See the
[CUDA backend guide](../../../docs/backend-guide/cuda-backend.md).

## Native support scaffold

The former `native/cuda/README.md` contained only the following placeholder, retained here as
[planning](../../../docs/glossary.md#term-planning) context rather than executable capability:

TODO: This directory will contain [CUDA](../../../docs/glossary.md#term-cuda) native integration and its build instructions.

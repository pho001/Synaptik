# backend.partition

`backend.partition` is a descriptor composition layer.

It connects partition targets to backend-owned legality adapters and lowerers.
It should stay small and declarative.

Allowed responsibilities:

- backend partition descriptor records
- descriptor registry composition
- target-to-adapter wiring

Not allowed:

- kernel selection
- bridge/runtime executable imports
- cost modeling
- backend execution policy

CPU, Metal, CUDA, and OpenCL implementation details belong under their backend roots.

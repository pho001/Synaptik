# backend.cpu

`backend.cpu` is the owner root for CPU backend implementation.

Target layout:

- `backend.cpu.prepare` owns CPU node preparation.
- `backend.cpu.lowering` owns CPU region lowering.
- `backend.cpu.partition` owns CPU partition legality and plans.
- `backend.cpu.registry` owns CPU kernel resolution.
- `backend.cpu.kernels` owns CPU runtime kernels.
- `backend.cpu.fused` owns fused planning, codegen, generated executable preparation, and generated ASM support.

`backend.cpu.fused` intentionally remains separate from `backend.cpu.kernels.fused`:

- `backend.cpu.fused` prepares generated or planned fused execution artifacts.
- `backend.cpu.kernels.fused` executes direct runtime fused kernels.

Root-level CPU classes in `backend` and the old split CPU kernel tree have been removed.
Do not add new CPU implementation code outside `backend.cpu`.

# Implementation atlas

This atlas explains every Java package in the audited source and the Metal native components.
Start with [one program through the framework](../framework.md); use the atlas when you want to
find the mechanism behind a call or choose the correct place for a contribution.

The pages follow responsibilities, not alphabetical class lists. Each package has its own
full-name heading, source entry points, inputs/outputs, state/lifetime and error boundaries, and
its relationship to the same learning model. A package being public Java does not make it a
supported application interface; CPU internal packages are implementation details.

| Responsibility | Packages/components explained |
|---|---|
| [Tensor meaning and graph vocabulary](model.md) | Data types, shapes, layouts, Tensor/storage/graph identity and all operation families |
| [Compilation and execution ownership](execution.md) | Trace, backend contract, configuration, planning, compiler, prepare, runtime and Engine |
| [CPU realization and optional native providers](cpu.md) | Lowering, typed IR, generated code, caches, memory, reference/portable/native routes and OpenBLAS provider |
| [Metal realization and native components](metal.md) | Java boundary, native context/program/Scan/kernel components and build/package verification |
| [Extensions, training, persistence and tools](extensions-tools.md) | NN, Data, Text, Training, Checkpoint, tuning, benchmarks and explicitly inert marker packages |

Coverage includes the three Java testing marker packages so that a source explorer does not
mistake those directory names for a hidden runtime. Build aggregation projects without Java
packages are not extra execution components. CUDA/ONNX/CLI markers are described as markers, not
advertised capabilities. No native OpenBLAS or oneDNN implementation is bundled merely because
its Java foreign-function boundary exists.

For the actual dependency rules and scopes, use [ARCHITECTURE.md](../../../ARCHITECTURE.md).
For what can execute on a selected device/type/layout, use the
[support matrix](../../backend-guide/production-support.md), not this source atlas.

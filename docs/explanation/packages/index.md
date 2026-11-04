# Implementation atlas

This atlas explains every Java package in the audited source and the [Metal](../../glossary.md#term-metal) native components.
Start with [one program through the framework](../framework.md); use the atlas when you want to
find the mechanism behind a call or choose the correct place for a contribution.

The pages follow responsibilities, not alphabetical class lists. Each package has its own
full-name heading, source entry points, inputs/outputs, state/[lifetime](../../glossary.md#term-lifetime) and error boundaries, and
its relationship to the same learning [model](../../glossary.md#term-model). A package being public Java does not make it a
supported application interface; [CPU](../../glossary.md#term-cpu) internal packages are implementation details.

| Responsibility | Packages/components explained |
|---|---|
| [Tensor meaning and graph vocabulary](model.md) | Data types, [shapes](../../glossary.md#shape), [layouts](../../glossary.md#layout), [Tensor](../../glossary.md#tensor)/storage/[graph](../../glossary.md#graph) [identity](../../glossary.md#term-identity-meanings) and all [operation](../../glossary.md#operation) families |
| [Compilation and execution ownership](execution.md)<span data-glossary-navigation="true"> ([Compilation glossary](../../glossary.md#compiler-engine-and-execution-stage-names); [ownership glossary](../../glossary.md#term-ownership-meanings))</span> | [Trace](../../glossary.md#trace), [backend](../../glossary.md#backend) contract, configuration, [planning](../../glossary.md#term-planning), [compiler](../../glossary.md#term-compiler), [prepare](../../glossary.md#prepare), [runtime](../../glossary.md#term-runtime) and [Engine](../../glossary.md#term-engine) |
| [CPU realization and optional native providers](cpu.md) | [Lowering](../../glossary.md#lowering), typed [IR](../../glossary.md#term-intermediate-representation-ir), generated code, caches, memory, reference/portable/native routes and [OpenBLAS](../../glossary.md#term-openblas) provider |
| [Metal realization and native components](metal.md) | Java boundary, native context/program/[Scan](../../glossary.md#term-cumulative-scan)/[kernel](../../glossary.md#kernel) components and build/package verification |
| [Extensions, training, persistence and tools](extensions-tools.md)<span data-glossary-navigation="true"> ([persistence glossary](../../glossary.md#term-persistence))</span> | [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context), Data, Text, Training, [Checkpoint](../../glossary.md#term-checkpoint), tuning, benchmarks and explicitly inert marker packages |

Coverage includes the three Java testing marker packages so that a source explorer does not
mistake those directory names for a hidden runtime. Build aggregation projects without Java
packages are not extra execution components. [CUDA](../../glossary.md#term-cuda)/[ONNX](../../glossary.md#term-onnx)/[CLI](../../glossary.md#term-cli) markers are described as markers, not
advertised capabilities. No native OpenBLAS or [oneDNN](../../glossary.md#term-onednn) implementation is bundled merely because
its Java foreign-function boundary exists.

For the actual [dependency](../../glossary.md#term-dependency) rules and scopes, use [ARCHITECTURE.md](../../../ARCHITECTURE.md).
For what can execute on a selected [device](../../glossary.md#term-device)/type/layout, use the
[support matrix](../../backend-guide/production-support.md)<span data-glossary-navigation="true"> ([support matrix glossary](../../glossary.md#term-review-matrix))</span><span data-glossary-navigation="true"> ([matrix glossary](../../glossary.md#term-review-matrix))</span>, not this source atlas.

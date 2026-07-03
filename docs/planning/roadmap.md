# Implementation Roadmap

## Authority

This roadmap coordinates implementation order. It is not an architecture contract. The authoritative contract is [`ARCHITECTURE.md`](../../ARCHITECTURE.md), and it wins if this roadmap conflicts with it.

## Execution policy

Implementation advances through one active frontier at a time. Complete the current area's tasks in master-plan order before moving to the next area. Create a detailed task specification only for the next unfinished task.

Parallel work is not the default. It requires an explicit roadmap or master-plan note confirming that dependencies and affected files do not overlap.

## Ordered project areas

| Order | Project area | Status | Entry condition | Exit condition |
|---|---|---|---|---|
| 1 | [`modules/model`](modules/model/master-plan.md) | Draft | Repository and planning infrastructure are ready. | Selected model capabilities and all model task acceptance criteria are complete. |
| 2 | [`modules/trace`](modules/trace/master-plan.md) | Draft | Required model contracts are stable or confirmed unnecessary. | Typed trace DTO contracts and validation are complete. |
| 3 | [`modules/backend-contract`](modules/backend-contract/master-plan.md) | Draft | Foundational value-model conventions are stable. | Backend identity and declarative requirement contracts are complete. |
| 4 | [`modules/config`](modules/config/master-plan.md) | Draft | Model and backend identity contracts required by configuration are stable. | Compile, prepare, run, and profile configuration contracts are complete. |
| 5 | [`modules/planning`](modules/planning/master-plan.md) | Draft | Model, trace, backend-contract, and config contracts are ready. | Ownership, partitioning, scoring, and logical memory planning are complete. |
| 6 | [`modules/runtime`](modules/runtime/master-plan.md) | Draft | Runtime-facing config, backend identities, and trace contracts are ready. | Prepared runtime contracts and dynamic run-state foundations are complete. |
| 7 | [`modules/compiler`](modules/compiler/master-plan.md) | Draft | Model, config, planning, backend-contract, and trace contracts are ready. | Compile artifacts, graph transformations, and autograd compilation are complete. |
| 8 | [`modules/prepare`](modules/prepare/master-plan.md) | Draft | Compiler, planning, runtime, config, backend-contract, and trace contracts are ready. | Shared prepare contracts and validation are complete. |
| 9 | [`backends/openblas-provider`](backends/openblas-provider/master-plan.md) | Draft | Native interop conventions needed by the provider are decided. | The low-level provider contract and validation are complete. |
| 10 | [`backends/cpu`](backends/cpu/master-plan.md) | Draft | Model, config, planning, runtime, prepare, backend-contract, trace, and OpenBLAS contracts are ready. | CPU is a conforming reference backend for the selected capability set. |
| 11 | [`modules/engine`](modules/engine/master-plan.md) | Draft | Compiler, runtime, prepare, and the CPU backend can be composed. | The public compile, prepare, and run lifecycle works end to end on CPU. |
| 12 | [`backends/metal`](backends/metal/master-plan.md) | Draft | Shared backend contracts and CPU reference behavior are stable. | Metal passes the applicable backend-conformance suite. |
| 13 | [`backends/cuda`](backends/cuda/master-plan.md) | Draft | Shared backend contracts and CPU reference behavior are stable. | CUDA passes the applicable backend-conformance suite. |
| 14 | [`extensions/onnx`](extensions/onnx/master-plan.md) | Draft | The model representation and public tensor semantics are stable. | Selected import/export mappings and compatibility validation are complete. |
| 15 | [`extensions/training`](extensions/training/master-plan.md) | Draft | Model, config, compiler autograd, and runtime publication contracts are stable. | Backend-independent optimizer and training-session capabilities are complete. |
| 16 | [`tools/tuning`](tools/tuning/master-plan.md) | Draft | Config and planning profiles are stable. | Tuning produces validated immutable profiles. |
| 17 | [`tools/benchmarks`](tools/benchmarks/master-plan.md) | Draft | Engine and selected execution paths are operational. | Repeatable benchmark suites and reporting are complete. |
| 18 | [`tools/cli`](tools/cli/master-plan.md) | Draft | Engine and diagnostic contracts are stable. | Selected diagnostic and execution commands are complete. |

The order above is the default delivery sequence, not a new dependency rule. Allowed and forbidden dependencies remain defined only by `ARCHITECTURE.md`.

## Current frontier

The current project area is [`modules/model`](modules/model/master-plan.md).

Its next task is:

```text
0009 Compiled graph model
```

Package migrations `0003A` through `0003C` and tasks `0004`–`0008` are complete. Task `0009` is
the next ordered planning frontier and remains `Draft` without a detailed specification.

## Model task sequence

| Order | Task | Status |
|---|---|---|
| 1 | 0001 DataType model | Complete |
| 2 | 0002 Shape and dimension model | Complete |
| 3 | 0003 Layout descriptor model | Complete |
| 4 | 0003A Data type package migration | Complete |
| 5 | 0003B Shape package migration | Complete |
| 6 | 0003C Layout package migration | Complete |
| 7 | 0004 Typed identifiers | Complete |
| 8 | 0005 Operation semantic foundation | Complete |
| 9 | 0006 Operation model | Complete |
| 10 | [0007 Tensor descriptor model](modules/model/tasks/0007-tensor-descriptor-model.md) | Complete |
| 11 | [0008 Graph value and node model](modules/model/tasks/0008-graph-value-and-node-model.md) | Complete |
| 12 | 0009 Compiled graph model | Draft |
| 13 | 0010 Host storage abstraction | Draft |
| 14 | 0011 Public Tensor skeleton | Draft |
| 15 | 0012 Tensor factory | Draft |
| 16 | 0013 Tensor provenance skeleton | Draft |
| 17 | 0014 Elementwise arithmetic operations | Draft |
| 18 | 0015 Comparison, logical, selection, and cast operations | Draft |
| 19 | 0016 Reduction and scan operations | Draft |
| 20 | 0017 Layout and view operations | Draft |
| 21 | 0018 Indexing and scatter operations | Draft |
| 22 | 0019 Linear algebra and attention operations | Draft |
| 23 | 0020 Convolution and pooling operations | Draft |
| 24 | 0021 Normalization operations | Draft |
| 25 | 0022 Loss operations | Draft |
| 26 | 0023 Compiler-generated semantic operations | Draft |
| 27 | 0024 Model capability parity audit | Draft |

Task dependencies in the model master plan remain hard prerequisites. The table order is the default execution order even when a later task has no explicit dependency on an earlier task.

## Model foundation checkpoint

After task `0013`, review the completed value, graph, storage, tensor, and provenance contracts against the entry requirements of downstream modules. The default remains to continue with model operation-family task groups. Advancing a cross-module vertical slice earlier requires an explicit roadmap update that names the new frontier and preserves all architecture dependency rules.

This checkpoint is not permission to skip or execute tasks out of order. It prevents the decision to complete every operation family before downstream feedback from remaining an implicit assumption.

## Advancing the frontier

Before advancing to the next task or project area:

1. complete all acceptance criteria for the current task;
2. record validation evidence and the completion summary;
3. review documentation and Javadoc impact;
4. update the task and master-plan statuses;
5. update this roadmap when the active project area changes; and
6. create the next detailed task specification as a separate planning step.

## Roadmap changes

Update this roadmap when implementation order, active frontier, or project-area status changes. Record the reason for reordering. If reordering reveals an architecture conflict, stop and resolve it through the architecture process instead of changing this roadmap alone.

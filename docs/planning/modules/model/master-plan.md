# Model Master Plan

## Goal

Define Synaptik's backend-independent tensor semantics, immutable graph model, public tensor state, and host storage contracts.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)
- [Model capability baseline](capabilities.md)

## Scope

- data type, dimension, shape, and layout value models
- typed tensor, node, value, and operation identifiers
- backend-independent operations and immutable attributes
- tensor descriptors, immutable graph values and nodes, and compiled graph state
- public `Tensor`, host storage abstractions, factories, and minimal provenance
- model-level representation and public expression construction for the selected legacy capability baseline

## Out of scope

- graph compilation and optimization
- backend ownership and partition planning
- prepared execution, runtime residency, and device buffers
- backend-specific storage, lowering, or kernels

## Module invariants

- `Tensor` is public mutable API state and is not an IR node.
- `Operation` owns semantics and never backend support.
- Compiled graph state is immutable.
- Host storage never represents runtime device residency.

## Allowed dependencies

- JDK standard library
- No project dependency until a focused task demonstrates an architecture-compliant need.

## Forbidden dependencies

- planning, compiler, runtime, prepare, and engine modules
- concrete backend modules
- kernel selection, device residency, runtime state, and prepared execution

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [DataType model](tasks/0001-data-type-model.md) | Complete | - | Define data type categories, metadata, floating promotion, and BFLOAT16 conversion. |
| 0002 | Shape and dimension model | Draft | - | Define static/dynamic dimensions and shapes. |
| 0003 | Layout descriptor model | Draft | 0002 | Define layout kind, strides, offset/view metadata. |
| 0004 | Typed identifiers | Draft | - | Define TensorId, NodeId, ValueId, OperationId. |
| 0005 | Operation taxonomy and attribute foundation | Draft | 0001 | Define the core semantic taxonomy and typed immutable attribute contracts used by later operation families. |
| 0006 | Operation model | Draft | 0005 | Define backend-independent operation object. |
| 0007 | Tensor descriptor model | Draft | 0001, 0002, 0003 | Define data type, shape, layout, and requires-grad descriptors. |
| 0008 | Graph value and node model | Draft | 0004, 0006, 0007 | Define immutable graph value and node records. |
| 0009 | Compiled graph model | Draft | 0008 | Define immutable graph container and publication binding. |
| 0010 | Public Tensor skeleton | Draft | 0001, 0002, 0003, 0004 | Define public Tensor metadata API without backend state. |
| 0011 | Host storage abstraction | Draft | 0010 | Define host storage abstractions without device buffers. |
| 0012 | Tensor factory | Draft | 0010, 0011 | Define tensor creation API and validation. |
| 0013 | Tensor provenance skeleton | Draft | 0006, 0010 | Define minimal provenance for future graph capture. |
| 0014 | Elementwise arithmetic operations | Draft | 0013 | Represent binary, unary, scalar, activation, and clamp capabilities. |
| 0015 | Comparison, logical, selection, and cast operations | Draft | 0014 | Represent comparison, boolean, where, and explicit cast capabilities. |
| 0016 | Reduction and scan operations | Draft | 0015 | Represent numeric and boolean reductions, scans, softmax, and tie policies. |
| 0017 | Layout and view operations | Draft | 0016 | Represent reshape, view, slice, composition, pad, tile, unfold, and fold capabilities. |
| 0018 | Indexing and scatter operations | Draft | 0017 | Represent gather, take, select, and functional scatter capabilities. |
| 0019 | Linear algebra and attention operations | Draft | 0018 | Represent matmul, linear, and scaled dot-product attention capabilities. |
| 0020 | Convolution and pooling operations | Draft | 0019 | Represent NCHW convolution and two-dimensional pooling capabilities. |
| 0021 | Normalization operations | Draft | 0020 | Represent batch, layer, and RMS normalization capabilities. |
| 0022 | Loss operations | Draft | 0021 | Represent dense/index NLL and cross-entropy variants and reductions. |
| 0023 | Compiler-generated semantic operations | Draft | 0022 | Represent backend-neutral backward and compiler-generated operation descriptors without autograd rules. |
| 0024 | Model capability parity audit | Draft | 0023 | Verify model representation and public expression construction against the selected legacy baseline. |

## Milestones

- Value foundations: tasks 0001–0004
- Operation and immutable graph model: tasks 0005–0009
- Public tensor and host storage: tasks 0010–0013
- Public operation capability families: tasks 0014–0022
- Compiler-generated model semantics and model parity: tasks 0023–0024

## Current status

Draft.

The capability baseline is documented and the ordered task queue covers its model-level responsibilities. Task 0001 is complete. The current frontier is task 0002; its detailed specification has not yet been created.

## Open questions

- The exact representation of dynamic dimensions remains local to task 0002.
- The canonical scalar shape and initial zero-sized-dimension policy remain local to task 0002.
- The minimal provenance representation remains local to task 0013.
- Exact public overloads and operation-attribute record boundaries remain local to the applicable operation-family tasks.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.
- The initial data type baseline is `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT32`, `INT64`, and `BOOL`.
- All selected legacy public operation capabilities must be representable without backend knowledge.
- Model capability parity and end-to-end executable parity are tracked separately.
- Fusion is not a model-level mathematical operation capability.

## Risks

- Accidentally treating public `Tensor` as compiled IR.
- Leaking runtime storage or backend support into the model.
- Expanding the public API before value-model invariants are stable.
- Treating the operation inventory as permission to move graph inference, autograd rules, fallback, or execution into model.
- Reproducing accidental legacy behavior instead of specifying and testing the intended contract.

## Notes

Execute tasks `0001` through `0024` in ascending order. The operation-family rows are planning boundaries, not permission for oversized implementations; split a row into smaller sequential tasks before implementation if its detailed specification would exceed the limits in [the planning guide](../../planning-guide.md).

Create the detailed specification for the next unfinished task only. The legacy branch must be consulted read-only for capability and test evidence when preparing each applicable task.

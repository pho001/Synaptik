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
- typed tensor, node, and value identifiers
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

## Package structure

The module root `io.github.pho001.synaptik.model` is a namespace boundary, not the default destination for new types. Model contracts are grouped by cohesive responsibility:

```text
io.github.pho001.synaptik.model.datatype
  Data type metadata, promotion, and host-independent bit conversion.

io.github.pho001.synaptik.model.shape
  Dimensions, immutable shapes, axes, and local broadcasting.

io.github.pho001.synaptik.model.layout
  Resolved logical layout geometry and layout classification.

io.github.pho001.synaptik.model.tensor
  Public Tensor state, TensorId, TensorDescriptor, TensorFactory, and provenance.

io.github.pho001.synaptik.model.storage
  Host-visible storage contracts and implementations.

io.github.pho001.synaptik.model.operation
  Backend-independent operation semantics and immutable attributes.

io.github.pho001.synaptik.model.graph
  NodeId, ValueId, graph values/nodes, graph phase, publication binding,
  and immutable compiled graph state.
```

Package dependencies remain acyclic. `datatype` and `shape` are foundational leaves; `layout` may depend on `shape`; `storage` may depend on `datatype`; `operation` may consume foundational value contracts but must not depend on public `Tensor` or compiled graph state; `tensor` may compose foundational values, host storage, and operation provenance; and `graph` may compose tensor descriptors and operation semantics. Package-private helpers live in the package whose contracts they implement.

Operation-family subpackages are introduced only when a focused operation task demonstrates a cohesive boundary. Generic `util`, `common`, `internal`, and `misc` packages are not part of the planned structure.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [DataType model](tasks/0001-data-type-model.md) | Complete | - | Define data type categories, metadata, floating promotion, and BFLOAT16 conversion. |
| 0002 | [Shape and dimension model](tasks/0002-shape-and-dimension-model.md) | Complete | - | Define static and symbolic dimensions, immutable shapes, checked element counts, axes, and broadcasting. |
| 0003 | [Layout descriptor model](tasks/0003-layout-descriptor-model.md) | Complete | 0002 | Define resolved layout kinds, checked element strides, offset/span, and view metadata. |
| 0003A | [Data type package migration](tasks/0003a-data-type-package-migration.md) | Complete | 0001 | Move completed data type contracts into `model.datatype` without changing behavior. |
| 0003B | [Shape package migration](tasks/0003b-shape-package-migration.md) | Complete | - | Move completed dimension and shape contracts into `model.shape` without changing behavior. |
| 0003C | [Layout package migration](tasks/0003c-layout-package-migration.md) | Complete | 0003B | Move completed layout contracts into `model.layout` and preserve their shape imports. |
| 0004 | [Typed identifiers](tasks/0004-typed-identifiers.md) | Complete | 0003A–0003C | Define TensorId, NodeId, and ValueId in their owning domain packages. |
| 0005 | [Operation semantic foundation](tasks/0005-operation-semantic-foundation.md) | Complete | - | Define the minimal operation-kind and typed-attribute contracts without family-specific semantics. |
| 0006 | [Operation model](tasks/0006-operation-model.md) | Complete | 0005 | Define the minimal immutable backend-independent kind-and-attributes descriptor. |
| 0007 | [Tensor descriptor model](tasks/0007-tensor-descriptor-model.md) | Complete | 0001–0003, 0003A–0003C | Define data type, shape, explicit resolved/unresolved layout, and requires-grad descriptors. |
| 0008 | Graph value and node model | Draft | 0004, 0006, 0007 | Define immutable graph value and node records. |
| 0009 | Compiled graph model | Draft | 0008 | Define immutable graph container and publication binding. |
| 0010 | Host storage abstraction | Draft | 0001, 0003A | Define host storage abstractions without device buffers. |
| 0011 | Public Tensor skeleton | Draft | 0004, 0007, 0010 | Define public Tensor metadata and host-storage state without runtime device state. |
| 0012 | Tensor factory | Draft | 0010, 0011 | Define tensor creation API and validation. |
| 0013 | Tensor provenance skeleton | Draft | 0006, 0011 | Define minimal provenance for future graph capture. |
| 0014 | Elementwise arithmetic operations | Draft | 0013 | Represent binary, unary, scalar, activation, and clamp capabilities. |
| 0015 | Comparison, logical, selection, and cast operations | Draft | 0013 | Represent comparison, boolean, where, and explicit cast capabilities. |
| 0016 | Reduction and scan operations | Draft | 0013 | Represent numeric and boolean reductions, scans, softmax, and tie policies. |
| 0017 | Layout and view operations | Draft | 0002, 0003, 0013 | Represent reshape, view, slice, composition, pad, tile, unfold, and fold capabilities. |
| 0018 | Indexing and scatter operations | Draft | 0001, 0013 | Represent gather, take, select, and functional scatter capabilities. |
| 0019 | Linear algebra and attention operations | Draft | 0013 | Represent matmul, linear, and scaled dot-product attention capabilities. |
| 0020 | Convolution and pooling operations | Draft | 0013 | Represent NCHW convolution and two-dimensional pooling capabilities. |
| 0021 | Normalization operations | Draft | 0013 | Represent batch, layer, and RMS normalization capabilities. |
| 0022 | Loss operations | Draft | 0013 | Represent dense/index NLL and cross-entropy variants and reductions. |
| 0023 | Compiler-generated semantic operations | Draft | 0006, 0014–0022 | Represent backend-neutral backward and compiler-generated operation descriptors without autograd rules. |
| 0024 | Model capability parity audit | Draft | 0001–0023 | Verify model representation and public expression construction against the selected legacy baseline. |

## Milestones

- Value foundations and package organization: tasks 0001–0004, including 0003A–0003C
- Operation and immutable graph model: tasks 0005–0009
- Public tensor and host storage: tasks 0010–0013
- Public operation capability families: tasks 0014–0022
- Compiler-generated model semantics and model parity: tasks 0023–0024

## Current status

Draft.

The capability baseline is documented and the ordered task queue covers its model-level responsibilities. Tasks 0001 through 0007 and package migrations 0003A–0003C are complete. Task 0008, graph value and node model, is the next ordered planning frontier and remains `Draft` without a detailed specification.

## Open questions

- The minimal provenance representation remains local to task 0013.
- Exact public overloads and operation-attribute record boundaries remain local to the applicable operation-family tasks.
- Task 0010 must define ownership, lifetime, mutability, alignment, and bounds for stable Java 26 `MemorySegment` values without leaking native backend storage into the model.
- After task 0013, review whether to continue through all model operation families or explicitly advance a cross-module vertical slice. The default roadmap remains sequential until that checkpoint records a different decision.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.
- The initial data type baseline is `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT32`, `INT64`, and `BOOL`.
- Static dimensions use non-negative `long` sizes; dynamic dimensions use explicit canonical symbols rather than negative sentinels.
- Scalar shape is rank zero, and zero-sized static dimensions are supported.
- Local broadcasting is right-aligned and conservative for symbolic dimensions; graph-wide symbolic constraints remain a compiler responsibility.
- Numeric layout descriptors are resolved only for fully static shapes and use non-negative `long` element strides and offsets.
- Layout geometry distinguishes dense, offset-dense, strided, and broadcast views; planning remains responsible for materialization decisions.
- Model contracts are organized by `datatype`, `shape`, `layout`, `tensor`, `storage`, `operation`, and `graph` responsibilities; the module root is not a flat type container.
- Java 26 is the project baseline. Stable Java 26 APIs require no preview opt-in; preview and incubator features remain disabled unless a focused owning-module task explicitly configures and validates them.
- Typed identifiers live with their domains. The current plan includes `TensorId`, `NodeId`, and `ValueId`; `OperationId` is deferred unless a focused task demonstrates identity distinct from `NodeId`.
- Host storage contracts precede the public `Tensor`, and `Tensor` reuses `TensorDescriptor` rather than duplicating descriptor validation.
- Operation-family table order coordinates delivery; dependencies record only real contract prerequisites rather than the preceding row.
- All selected legacy public operation capabilities must be representable without backend knowledge.
- Model capability parity and end-to-end executable parity are tracked separately.
- Fusion is not a model-level mathematical operation capability.

## Risks

- Accidentally treating public `Tensor` as compiled IR.
- Leaking runtime storage or backend support into the model.
- Expanding the public API before value-model invariants are stable.
- Creating cycles between `operation`, `tensor`, and `graph` packages.
- Enabling preview or incubator features globally instead of containing them in the module that requires them.
- Treating the operation inventory as permission to move graph inference, autograd rules, fallback, or execution into model.
- Reproducing accidental legacy behavior instead of specifying and testing the intended contract.

## Notes

Execute tasks in table order, including package migrations 0003A through 0003C before task 0004. The operation-family rows are task groups, not permission for oversized implementations; replace the current frontier row with smaller sequential task rows before implementation when its detailed scope would exceed the limits in [the planning guide](../../planning-guide.md).

Package migrations 0003A–0003C and tasks 0004–0007 are complete. Task 0008 is the next ordered planning frontier and remains a master-plan row without a detailed specification. Concrete operation families remain in tasks 0014–0023. The legacy branch must be consulted read-only for capability and test evidence when preparing each applicable capability task.

# Model Capability Baseline

## Purpose and authority

This document records the capability baseline that the new Synaptik model must be able to represent. It is an implementation-planning document, not an architecture contract.

The authoritative contract is [`ARCHITECTURE.md`](../../../../ARCHITECTURE.md). The model boundaries are explained in [Module Boundaries](../../../architecture/module-boundaries.md). If this baseline conflicts with either, implementation must stop and the architecture conflict must be resolved first.

The legacy implementation on the read-only `legacy/pre-rewrite` branch is evidence for observable capabilities. It is not a source design. New APIs, types, validation, and tests must be written from scratch under the `io.github.pho001.synaptik.*` namespace.

## Capability-parity policy

The selected baseline is all public tensor operation capabilities present in the legacy project. Parity means preserving the useful mathematical operation, accepted options, shape behavior, data type behavior, and failure conditions after those contracts are specified and tested for the new design.

Parity does not require:

- copying legacy classes, package names, or implementation structure;
- preserving accidental bugs or architecture violations;
- keeping every legacy overload when one coherent new API covers the same capability;
- representing every public convenience method as a distinct operation kind; or
- implementing compiler, autograd, runtime, or backend behavior inside `modules/model`.

When legacy behavior is ambiguous or inconsistent, the applicable task must record a local decision and add tests. It must not silently turn that decision into an architecture rule.

## Meaning of support

Support has multiple layers. A capability is not end-to-end complete merely because its operation kind exists in `modules/model`.

| Layer | Required responsibility |
|---|---|
| Model | Represent the operation and immutable attributes without backend knowledge. |
| Public tensor API | Build the expression and validate arguments that can be checked locally. |
| Compiler | Capture the expression, perform graph-wide inference and validation, and create backward operations when required. |
| Planning | Select backend ownership from declarative capabilities; never select a kernel. |
| Backend prepare | Lower an owned region and select a concrete implementation route. |
| Runtime | Execute the already-prepared schedule without rediscovering or falling back to another backend. |
| Tests | Verify semantic behavior and applicable backend conformance. |

The model milestone is complete when the model and public-API portions are represented and tested. Full project parity additionally requires compiler, CPU reference backend, planning, prepare, runtime, and integration work in their own plans.

## DataType baseline

The initial data type set is:

| DataType | Category | Host representation baseline | Notes |
|---|---|---|---|
| `FLOAT64` | Floating | `double` / `double[]` | IEEE-754 binary64. |
| `FLOAT32` | Floating | `float` / `float[]` | IEEE-754 binary32 and the default floating data type. |
| `BFLOAT16` | Floating | bfloat16 bits in `short` / `short[]` | Conversion behavior must be specified by task 0001. |
| `INT32` | Integral | `int` / `int[]` | Includes index-tensor use. |
| `INT64` | Integral | `long` / `long[]` | Includes ONNX-compatible index values. |
| `BOOL` | Boolean | normalized `byte` / `byte[]` for host interchange | Logical values are restricted to false/true semantics. |

The data type model must expose category and element-width metadata. Only floating data types are differentiable. Initial floating promotion follows:

```text
BFLOAT16 < FLOAT32 < FLOAT64
```

No implicit promotion between floating, integral, and boolean categories is assumed. Cast is an explicit operation. `FLOAT16` is not part of the legacy-parity baseline and requires a future planned capability decision.

## Shape and dimension baseline

The model must be able to describe:

- tensor rank, dimensions, and checked element count;
- static dimensions represented by non-negative `long` sizes;
- explicit symbolic dynamic dimensions represented without negative numeric sentinels;
- rank-0 scalar shapes with element count one;
- zero-sized dimensions and empty tensors;
- normalized positive and negative axes;
- right-aligned NumPy-style broadcasting;
- broadcast dimensions represented through effective zero strides where applicable;
- reduction shapes with and without retained dimensions;
- reshape with one inferred `-1` dimension at the public API boundary;
- overflow-safe element-count and stride calculations; and
- operation-specific output-shape metadata without backend information.

The legacy implementation represented scalar results as shape `[1]`, rejected zero-sized dimensions, and limited element count to `Integer.MAX_VALUE`. The new model deliberately uses rank zero for scalars, permits zero-sized dimensions, and keeps model dimensions independent of Java array-size limits. Storage implementations may impose narrower validated limits later.

## Layout baseline

`LayoutDescriptor` must describe logical layout independently of physical device storage. Required capabilities are:

- resolved numeric layouts for fully static shapes;
- contiguous row-major and contiguous-with-offset layout kinds;
- general strided and zero-stride broadcast layout kinds;
- explicit non-negative `long` element strides;
- non-negative storage offset and checked referenced element span;
- explicit storage-alias/view metadata independent of layout kind;
- permuted views;
- sliced views;
- expanded/broadcast views;
- reshape and contiguity metadata;
- layout-preserving aliases; and
- sufficient immutable facts for planning to derive logical materialization requirements.

Numeric layout descriptors are not created for dynamic shapes until their required dimensions are resolved by later compiler/runtime contracts. Layout metadata must not contain device addresses, runtime residency, backend storage handles, kernel routes, prepared-execution state, or materialization policy.

## Identifier baseline

The model uses distinct immutable identifier types for distinct semantic domains:

- `TensorId` identifies public tensor state and belongs to `model.tensor`;
- `NodeId` identifies a node occurrence within an owning graph and belongs to `model.graph`; and
- `ValueId` identifies an input, intermediate, or output value within an owning graph and belongs to `model.graph`.

Identifiers use validated non-negative `long` values. They are identity values only: allocation, uniqueness, graph construction, tensor lifecycle, persistence, and serialization belong to later focused tasks. Graph-local numeric values may be reused by different graph containers and must be interpreted in their owning graph context. Negative sentinels and implicit conversion between identifier types are not supported.

`OperationId` is not part of the current baseline. `NodeId` identifies the occurrence of operation semantics in a graph; a separate operation identity requires a future demonstrated use case rather than a speculative contract. Trace identifiers remain local to `modules/trace` and do not reuse model identifier types.

## Host storage baseline

The model owns host-visible tensor storage only. The baseline includes:

- `HostTensorStorage` as the model-level abstraction;
- array-backed host storage for all six initial data types;
- `MemorySegmentStorage` for host memory represented by a JDK memory segment;
- element count and data type consistency;
- typed bulk import and export;
- mutation/version tracking needed by public mutable tensor state;
- explicit ownership and lifetime contracts for host memory; and
- multiple tensor views sharing the same host storage through layout metadata.

The exact class count and whether storage implementations are public are local API-design decisions for task 0010. Capability parity does not require direct exposure of mutable backing arrays.

The project uses Java 26, where `MemorySegment` is a stable API. Task 0010 may therefore implement `MemorySegmentStorage` without enabling preview features. The task must still define ownership, lifetime, mutability, alignment, and bounds behavior explicitly; the stable API does not decide those model contracts.

The following are explicitly outside `modules/model`:

- Metal buffers;
- CUDA allocations;
- backend-native workspaces;
- runtime residency records;
- physical device buffer slots; and
- legacy allocation handles coupled to runtime execution resources.

## Public Tensor baseline

`Tensor` remains public mutable API state and must not become an IR node. Its selected model-level capabilities are:

- data type, shape, layout, label, and typed identifier metadata;
- host storage access and replacement under explicit validation rules;
- `requiresGrad`, gradient publication state, and trainable-parameter metadata where required by the architecture;
- publication intent that can later be converted into immutable `PublicationBinding` data;
- typed scalar and element access;
- typed array copy/export;
- layout and contiguity inspection;
- expression-building methods backed by minimal provenance; and
- factory integration.

Compilation, preparation, execution, topological graph traversal, backend selection, device synchronization, and runtime residency do not belong to `Tensor` in the new model. Lifecycle convenience APIs belong to the engine facade.

## Tensor factory baseline

The factory capability set includes:

- scalar tensors;
- zeros and ones;
- zeros-like and ones-like;
- normally distributed random tensors;
- integer ranges with a non-zero step;
- tensors from flat typed arrays;
- tensors from supported nested Java arrays;
- strict-prefix and cyclic-prefix filling for fixture/data preparation parity;
- explicit data type, shape, label, and gradient metadata where applicable; and
- validation that logical element count matches supplied data.

Random-source configuration and reproducibility policy must be decided without introducing live services into `modules/model`.

## Operation foundation baseline

Operation semantics use two open typed contracts under `model.operation`:

- `OperationKind` identifies backend-independent semantic kinds through immutable typed values with stable diagnostic names; and
- `OperationAttrs` marks immutable typed semantic-attribute values.

`NoOperationAttrs.INSTANCE` is the canonical attribute value for a kind with no parameters. Attributes are never represented primarily as `Map<String, ?>`, and absence is not represented by `null`. Concrete kinds and attribute records are introduced progressively by the applicable operation-family task rather than through a speculative monolithic enum.

The model foundation does not expose computational cost, fusion eligibility, kernel routes, backend support, device facts, materialization decisions, or runtime behavior. `FUSED` is backend-prepare output rather than a semantic kind, and `UNKNOWN` is not a supported operation. `Operation` itself is a separate ordered task built on this foundation.

## Public operation baseline

The following sections inventory public mathematical capabilities. They do not prescribe one Java class per item or a one-to-one mapping between API methods and operation kinds.

### Elementwise arithmetic and unary operations

- binary `add`, `sub`, `mul`, `div`, `min`, `max`, and tensor `pow` with broadcasting;
- unary `abs`, `neg`, `inv`, `log`, `exp`, `erf`, `sqrt`, `floor`, `ceil`, and `sign`;
- activations `relu`, `sigmoid`, and `tanh`;
- approximation variants `fastExp` and `fastTanh`;
- scalar multiplication and scalar power; and
- `clamp`, `clampMin`, and `clampMax`.

### Comparison, logical, selection, and cast operations

- comparisons `greaterThan`, `greaterOrEqual`, `lessThan`, `lessOrEqual`, `equalTo`, and `notEqualTo`;
- boolean `logicalAnd`, `logicalOr`, and `logicalNot`;
- broadcast-aware `where`; and
- explicit data type `cast`.

Comparison and logical results use `BOOL`. Logical inputs must follow the new data type contract rather than relying on numeric truthiness unless a focused task explicitly decides otherwise.

### Layout and view operations

- `contiguous`;
- `reshape`;
- `expand`;
- `permute` and two-axis `transpose` convenience;
- `expandDims` and `squeeze`;
- general `slice` and single-axis slice convenience;
- constant `pad`;
- `tile`;
- `concat`, `stack`, and `unstack`;
- single-axis `unfold`;
- `unfold2d`; and
- `fold2d`.

The two-dimensional unfold/fold window contract includes kernel, stride, symmetric padding, dilation, and `ceilMode` options.

The model expresses alias and view semantics. Backend-neutral planning derives logical materialization requirements from those semantics, and prepare/runtime/backend layers realize the required storage and copies.

### Indexing and scatter operations

- scalar-index `select`;
- `gather`;
- `gatherAxis` / `take`;
- `gatherNd` with `batchDims`;
- `takeAlongAxis`;
- functional `scatterAdd`;
- `scatterAxisAdd`;
- `scatterElements`; and
- `scatterNd` with `batchDims`.

Scatter reductions include `NONE`, `ADD`, `MUL`, `MAX`, and `MIN`. Index tensors must use an integral data type (`INT32` or `INT64`); inconsistent legacy acceptance of other data types is not a compatibility requirement.

### Reduction and scan operations

- `sum`, including full reduction, axis reduction, retained dimensions, and masked forms;
- `mean`, including full reduction, axis reduction, retained dimensions, and masked forms;
- `prod`, including full reduction, axis reduction, and retained dimensions;
- reduction `min` and `max`, including full and retained-dimension forms;
- `argMax` with `FIRST_INDEX` and `LAST_INDEX` tie policies;
- `cumSum` with `exclusive` and `reverse` options;
- `softmax` and `logSoftmax`; and
- boolean `all` and `any`, including full and retained-dimension forms.

### Linear algebra and attention operations

- vector, matrix, and batched `matmul` under a documented shape contract;
- `linear` with optional bias; and
- scaled dot-product attention with optional mask, causal mode, and optional scale.

### Convolution and pooling operations

- NCHW `conv2d` with optional bias;
- convolution stride, padding, dilation, and groups;
- `maxPool2d`;
- `avgPool2d`; and
- pooling kernel, stride, symmetric padding, `ceilMode`, and `countIncludePad` options.

### Normalization operations

- batch normalization with statistics computed from the input;
- batch normalization with supplied mean and variance;
- layer normalization over trailing dimensions; and
- RMS normalization.

A public normalization capability may be expressed as a composition of semantic primitives. Capability parity does not require a distinct primitive operation kind when composition preserves the specified semantics and compiler visibility.

### Loss operations

- dense-target negative log-likelihood loss;
- dense-target cross-entropy loss;
- optional masks for applicable dense losses;
- index-target negative log-likelihood loss;
- index-target cross-entropy loss;
- `ignoreIndex`;
- optional class weights; and
- reductions `NONE`, `SUM`, and `MEAN`.

## Compiler-generated semantic operations

The immutable model must be able to represent backend-independent operations emitted by compiler transformations and autograd expansion. The legacy capability evidence includes:

- binary min/max gradients;
- reduction min/max gradients;
- softmax and log-softmax gradients;
- index-target cross-entropy gradient;
- gather, gather-axis, gather-ND, and take-along-axis gradients;
- slice backward;
- scaled dot-product attention backward; and
- attention-weight calculation used by backward expansion.

The operation descriptors and immutable attributes may live in `modules/model`. Gradient rules, backward graph construction, and decisions about when to emit them belong to `modules/compiler`.

Constant-scalar and semantic no-op descriptors may also be represented when compiler graph construction requires them. `FUSED` is not a model-level mathematical capability: concrete fusion and fused executable representation belong to backend prepare. `UNKNOWN` is a diagnostic or invalid sentinel, not a supported operation.

## Backend capability behavior

No operation descriptor may expose `supportedBackends()` or reference a concrete backend. Backend capability providers separately declare which combinations of operation, data type, shape, and layout they can prepare.

An unsupported accelerator operation is handled before runtime execution:

```text
capability analysis
  -> planning assigns backend ownership
  -> prepare lowers each owned partition
  -> runtime executes the prepared schedule
```

CPU ownership may provide the reference path for a region that an accelerator cannot support. This is a compile/prepare-time ownership decision, not a dynamic runtime fallback.

## Compatibility evidence and validation

Each operation-family task must inspect relevant legacy public APIs and tests before its new contract becomes `Ready`. Validation should record:

- selected public capability and options;
- input data type and shape constraints;
- output data type and shape semantics;
- broadcasting, axis, and reduction behavior;
- differentiability expectation;
- invalid-input behavior;
- model-level unit tests; and
- required compiler, backend-conformance, or integration follow-up.

The final model parity audit verifies representation and public expression construction only. End-to-end numeric parity is completed later by the owning compiler and backend tasks.

## Deferred capabilities

The following are not part of the initial legacy-parity baseline unless added through planning:

- additional data types such as `FLOAT16`, unsigned integers, or complex numbers;
- sparse, quantized, string, or distributed tensor storage;
- runtime device tensors in `modules/model`;
- backend-specific operation variants; and
- optimizer algorithms, which belong to `extensions/training`.

## Open design questions

- Which operation families use dedicated immutable attribute records versus a shared typed attribute vocabulary?
- Which public Tensor overloads are retained, consolidated, or replaced while preserving capability parity?
- What ownership modes does `MemorySegmentStorage` support for externally supplied host memory?

These questions are resolved by the applicable ordered task and recorded as local decisions. They do not block documenting the capability baseline.

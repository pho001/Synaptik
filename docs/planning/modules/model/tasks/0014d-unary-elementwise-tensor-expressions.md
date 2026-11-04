# Task 0014D: [Unary](../../../../glossary.md#term-unary) [Elementwise](../../../../glossary.md#term-elementwise) [Tensor](../../../../glossary.md#tensor) Expressions

**Editorial reading key (later context, not a status update).**

A unary elementwise transform acts independently at each input coordinate, rather than combining a row or axis into one total. The expression retains the input type/Shape while leaving output geometry unresolved. Gradient eligibility is only metadata here, including for discontinuous functions; the recorded task does not attach derivative rules. The application calls the public Tensor method; its model helper validates available descriptions and returns a fresh storage-free expression with exact input references. It neither copies input values nor calculates the result. Compiler consumes this provenance later; actual storage and execution are separate downstream concerns.

Focused current reading: [Current unary-transform reference](../../../../reference/tensor/elementwise.md#unary-numeric-transforms-and-floating-classifications). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Expose all fifteen implemented [unary](../../../../glossary.md#term-unary) [elementwise](../../../../glossary.md#term-elementwise) semantics as public, [backend](../../../../glossary.md#backend)-independent [Tensor](../../../../glossary.md#tensor)
expression methods. Each successful call must validate the floating input type, preserve the
logical [data type](../../../../glossary.md#data-type--datatype) and [shape](../../../../glossary.md#shape), derive one [immutable](../../../../glossary.md#term-immutable) unresolved-[layout](../../../../glossary.md#layout) [descriptor](../../../../glossary.md#term-descriptor), and return a fresh
storage-free Tensor whose [provenance](../../../../glossary.md#provenance) records the exact [operation](../../../../glossary.md#operation) and exact input Tensor [identity](../../../../glossary.md#term-identity-meanings).

This task makes unary arithmetic, transcendental, [activation](../../../../glossary.md#term-activation), and explicit fast-approximation
requests capturable through the public [model](../../../../glossary.md#term-model) [API](../../../../glossary.md#term-application-programming-interface-api). It does not calculate numeric values, inspect
mathematical domains, allocate result storage, [capture](../../../../glossary.md#term-capture-meanings) a [compiled graph](../../../../glossary.md#compiled-graph--compiledgraphmodel), define [gradient](../../../../glossary.md#term-gradient) rules, or
report backend support.

## [Scope](../../../../glossary.md#term-scope)

- Add public zero-[argument](../../../../glossary.md#term-argument) [`Tensor`](../../../../glossary.md#tensor) methods `abs`, `neg`, `inv`, [`log`](../../../../glossary.md#term-log), [`exp`](../../../../glossary.md#term-expx), [`erf`](../../../../glossary.md#term-erf), `sqrt`,
  `floor`, `ceil`, `sign`, `relu`, [`sigmoid`](../../../../glossary.md#term-sigmoid), [`tanh`](../../../../glossary.md#term-tanh), `fastExp`, and `fastTanh`.
- Add one package-private `TensorUnaryExpressions` helper that owns the shared local validation and
  derived-Tensor construction path.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Accept only [`BFLOAT16`](../../../../glossary.md#term-bfloat16), [`FLOAT32`](../../../../glossary.md#term-float32), and [`FLOAT64`](../../../../glossary.md#term-float64) inputs.
- Preserve the input [descriptor](../../../../glossary.md#term-descriptor)'s exact [data type](../../../../glossary.md#data-type--datatype) and exact [immutable](../../../../glossary.md#term-immutable) [shape](../../../../glossary.md#shape) reference.
- Create an unresolved-[layout](../../../../glossary.md#layout) result `TensorDescriptor` with the input descriptor's exact
  `requiresGrad` value.
- Construct exactly one [`Operation`](../../../../glossary.md#operation) from the selected `UnaryElementwiseKind` and
  `NoOperationAttrs.INSTANCE`.
- Construct exactly one `TensorProvenance` with the one exact input reference.
- Delegate final [identity](../../../../glossary.md#term-identity-meanings)-bearing construction exactly once to `TensorFactory.createDerived` with
  no label and no storage.
- Update the exact Tensor public-[API](../../../../glossary.md#term-application-programming-interface-api) reflection test and add one focused [unary](../../../../glossary.md#term-unary)-expression test.
- Finalize affected Javadocs, Tensor API, the authorized Compile API status correction, glossary,
  task evidence, model master plan, and roadmap through the required independent documentation
  pass during implementation.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- eager numeric execution, constant folding, algebraic simplification, canonicalization, or
  returning an existing Tensor for identities such as double negation or double reciprocal<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [constant folding glossary](../../../../glossary.md#term-constant-folding); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [Tensor glossary](../../../../glossary.md#tensor))</span>
- reading, copying, allocating, attaching, materializing, or validating [host storage](../../../../glossary.md#host-storage)
- integral or boolean unary arithmetic, implicit conversion, explicit cast insertion, or a new
  data-type promotion contract<span data-glossary-navigation="true"> ([boolean glossary](../../../../glossary.md#data-type--datatype); [unary glossary](../../../../glossary.md#term-unary))</span>
- [scalar](../../../../glossary.md#term-scalar) [arguments](../../../../glossary.md#term-argument), overloads, in-place methods, static [operation](../../../../glossary.md#operation) factories, [aliases](../../../../glossary.md#aliasing-and-arena), generic
  `apply`, or an expression-builder public type
- output labels, caller-supplied labels, symbols, serialization names, or diagnostic expression
  strings<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization))</span>
- resolved output layouts, input-layout preservation, view/alias behavior, materialization policy,
  or physical stride derivation<span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout); [view glossary](../../../../glossary.md#term-view-meanings); [materialization glossary](../../../../glossary.md#materialization); [stride glossary](../../../../glossary.md#term-stride))</span>
- operation-family attributes, factories, registries, parsers, reflection discovery, or changes to
  `UnaryElementwiseKind`, `Operation`, [`OperationKind`](../../../../glossary.md#operationkind), [`OperationAttrs`](../../../../glossary.md#operationattrs), or [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs)
- mathematical domain or range validation, overflow, underflow, rounding, signed-zero, NaN,
  infinity, exactness, strict-function accuracy, or fast-approximation accuracy and algorithms<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [rounding glossary](../../../../glossary.md#term-rounding); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity))</span>
- graph traversal, cycle checks, graph IDs, graph capture, compiled graph construction,
  common-subexpression elimination, compiler inference, or publication binding<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [compiled graph glossary](../../../../glossary.md#compiled-graph--compiledgraphmodel); [common-subexpression elimination glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [compiler glossary](../../../../glossary.md#term-compiler); [inference glossary](../../../../glossary.md#term-inference); [publication glossary](../../../../glossary.md#term-publication); [binding glossary](../../../../glossary.md#term-binding))</span>
- gradient values, gradient rules, subgradient choices, nondifferentiable-point policy, backward
  graph generation, autograd execution, or training behavior<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [backward graph glossary](../../../../glossary.md#backward-graph); [autograd glossary](../../../../glossary.md#autograd))</span>
- scalar multiplication, scalar power, `clamp`, `clampMin`, or `clampMax`; tasks 0014E–0014F own
  their attributes and expression surface
- capability providers, backend support, fusion, cost, lowering, kernels, runtime residency,
  prepare, execution, traces, or engine behavior<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [fusion glossary](../../../../glossary.md#term-fusion); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel); [runtime glossary](../../../../glossary.md#term-runtime); [residency glossary](../../../../glossary.md#residency); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine))</span>
- dependencies, Gradle changes, architecture changes, another module, another operation family, or
  a detailed task-0014E specification<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model master plan](../master-plan.md)
- [Task 0001](0001-data-type-model.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0003](0003-layout-descriptor-model.md)
- [Task 0006](0006-operation-model.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0011](0011-public-tensor-skeleton.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0014C](0014c-unary-elementwise-semantic-kinds.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes all fifteen selected [unary](../../../../glossary.md#term-unary) capabilities through
its public [Tensor](../../../../glossary.md#tensor)/TensorOps surface: `abs`, `neg`, `inv`, [`log`](../../../../glossary.md#term-log), [`exp`](../../../../glossary.md#term-expx), [`erf`](../../../../glossary.md#term-erf), `sqrt`, `floor`,
`ceil`, `sign`, `relu`, [`sigmoid`](../../../../glossary.md#term-sigmoid), [`tanh`](../../../../glossary.md#term-tanh), `fastExp`, and `fastTanh`. The legacy builders accept
floating values and create [shape](../../../../glossary.md#shape)-preserving unary operations. Legacy tests additionally exercise
expression chaining, [gradients](../../../../glossary.md#term-gradient), storage [layouts](../../../../glossary.md#layout), [kernel](../../../../glossary.md#kernel) execution, [fusion](../../../../glossary.md#term-fusion), approximation paths,
and selected [eager](../../../../glossary.md#term-eager-evaluation) [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form).

Only the useful public [operation](../../../../glossary.md#operation) capability, floating eligibility, shape preservation, and exact
semantic [identity](../../../../glossary.md#term-identity-meanings) are retained here. The legacy implementation's [mutable](../../../../glossary.md#term-mutable) [node](../../../../glossary.md#node) state, gradient
callbacks, eager storage coupling, expression labels, kernel/fusion details, and canonicalization
are not copied. [Compiler](../../../../glossary.md#term-compiler) optimization owns semantic rewrites, compiler [autograd](../../../../glossary.md#autograd) owns gradient
expansion, and prepared [backends](../../../../glossary.md#backend) own numerical execution and fast approximation algorithms.

Legacy `floor`, `ceil`, and `sign` forcibly disabled gradient state. The new [model](../../../../glossary.md#term-model) does not copy
that coupling into expression construction: `requiresGrad` is model-level eligibility [metadata](../../../../glossary.md#term-metadata),
not proof that an operation has a gradient rule. All fifteen methods therefore preserve the input
request uniformly, while a later compiler/autograd contract decides whether and how each operation
contributes to a [backward graph](../../../../glossary.md#backward-graph).

## Architecture constraints

- [`Tensor`](../../../../glossary.md#tensor) remains public [mutable](../../../../glossary.md#term-mutable) [API](../../../../glossary.md#term-application-programming-interface-api) state and must not become an [IR](../../../../glossary.md#term-intermediate-representation-ir) [node](../../../../glossary.md#node).
- `TensorUnaryExpressions` performs deterministic local model validation only. It must not inspect
  values, traverse provenance, capture a graph, evaluate mathematics, or inspect backend
  capability.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [validation glossary](../../../../glossary.md#term-validation); [provenance glossary](../../../../glossary.md#provenance); [capture glossary](../../../../glossary.md#term-capture-meanings); [graph glossary](../../../../glossary.md#graph); [backend capability glossary](../../../../glossary.md#backend-capability))</span>
- [`Operation`](../../../../glossary.md#operation) owns only the backend-independent semantic kind and attributes. It contains no input
  Tensor reference, result descriptor, backend support, or executable behavior.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- `TensorProvenance` owns the one exact input [identity](../../../../glossary.md#term-identity-meanings). It must contain exactly `List.of(input)` in
  that order and must not substitute an ancestor or equal Tensor.
- Result identity comes only from `TensorFactory.createDerived`. No second allocator,
  caller-supplied ID, registry, cache, interning table, or service is introduced.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [cache glossary](../../../../glossary.md#term-cache))</span>
- The result is storage-free. Public expression construction must not allocate a physical buffer
  or attach/alias the input's storage.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- The result [descriptor](../../../../glossary.md#term-descriptor) has `Optional.empty()` layout even when the input has a resolved layout
  and fully static shape. A semantic expression does not assert a future materialization route.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [materialization glossary](../../../../glossary.md#materialization))</span>
- [Unary](../../../../glossary.md#term-unary) [elementwise](../../../../glossary.md#term-elementwise) [shape](../../../../glossary.md#shape) preservation retains the exact [immutable](../../../../glossary.md#term-immutable) input `Shape` reference. No
  shape algebra, symbolic binding, or new constraint is required.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- The exact input data type is retained after floating validation. Integral and boolean inputs are
  rejected; the helper must not convert, promote, or insert a cast.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [boolean glossary](../../../../glossary.md#data-type--datatype))</span>
- Result `requiresGrad` equals the input descriptor's flag for every kind, including `FLOOR`,
  `CEIL`, and `SIGN`. This propagates eligibility metadata only and does not promise a derivative,
  gradient rule, or differentiable backend execution.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [derivative glossary](../../../../glossary.md#term-derivative); [gradient glossary](../../../../glossary.md#term-gradient); [differentiable glossary](../../../../glossary.md#derivative-and-chain-rule))</span>
- `FAST_EXP` and `FAST_TANH` remain exact distinct semantic requests. The helper must not alias them
  to strict variants or choose an approximation implementation.
- Package direction is `model.tensor -> model.operation.elementwise.unary`, plus existing
  `model.tensor -> model.operation` and `model.datatype`. The operation package must not import
  Tensor and no package cycle may be introduced.
- Stop if implementation requires a changed foundational contract, resolved layout, storage/value
  access, graph capture, gradient policy, dependency, or architecture decision.<span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns public Tensor expression methods, local result
  descriptor construction, provenance, and derived Tensor creation.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- `io.github.pho001.synaptik.model.operation` — supplies [`Operation`](../../../../glossary.md#operation) and
  `NoOperationAttrs.INSTANCE`.
- `io.github.pho001.synaptik.model.operation.elementwise.unary` — supplies the fifteen exact
  `UnaryElementwiseKind` values.
- `io.github.pho001.synaptik.model.datatype` — supplies the existing [immutable](../../../../glossary.md#term-immutable) [`DataType`](../../../../glossary.md#data-type--datatype) category
  query used for floating validation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>

No package is added.

Type placement:

- `io.github.pho001.synaptik.model.tensor.Tensor` — public fluent expression surface; it receives
  only the fifteen zero-argument methods and delegates shared behavior.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument))</span>
- `io.github.pho001.synaptik.model.tensor.TensorUnaryExpressions` — package-private, stateless
  construction boundary colocated with [`Tensor`](../../../../glossary.md#tensor), `TensorDescriptor`, `TensorProvenance`, and the
  package-private factory seam it must use.
- `TensorUnaryElementwiseTest` — same-package focused test so it can verify the package-private
  helper without widening production visibility.<span data-glossary-navigation="true"> ([widening glossary](../../../../glossary.md#term-widening))</span>

## Required contract

### Public [Tensor](../../../../glossary.md#tensor) surface

Add exactly these public methods to `Tensor`:

```java
public Tensor abs()
public Tensor neg()
public Tensor inv()
public Tensor log()
public Tensor exp()
public Tensor erf()
public Tensor sqrt()
public Tensor floor()
public Tensor ceil()
public Tensor sign()
public Tensor relu()
public Tensor sigmoid()
public Tensor tanh()
public Tensor fastExp()
public Tensor fastTanh()
```

Each method delegates exactly once to `TensorUnaryExpressions.apply(this, <KIND>)` and returns that
exact result. It performs no separate [validation](../../../../glossary.md#term-validation), [descriptor](../../../../glossary.md#term-descriptor) or [provenance](../../../../glossary.md#provenance) construction,
[canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form), allocation, storage access, or exception translation. There are no overloads in
this task.

Method-to-kind mapping is exact and name preserving:

| Tensor method | Kind |
|---|---|
| `abs()` | `ABS` |
| `neg()` | `NEG` |
| `inv()` | `INV` |
| `log()` | [`LOG`](../../../../glossary.md#term-log) |
| `exp()` | [`EXP`](../../../../glossary.md#term-expx) |
| `erf()` | [`ERF`](../../../../glossary.md#term-erf) |
| `sqrt()` | `SQRT` |
| `floor()` | `FLOOR` |
| `ceil()` | `CEIL` |
| `sign()` | `SIGN` |
| `relu()` | `RELU` |
| `sigmoid()` | [`SIGMOID`](../../../../glossary.md#term-sigmoid) |
| `tanh()` | [`TANH`](../../../../glossary.md#term-tanh) |
| `fastExp()` | `FAST_EXP` |
| `fastTanh()` | `FAST_TANH` |

### Package-private helper [shape](../../../../glossary.md#shape)

Create exactly one package-private final non-record class:

```java
final class TensorUnaryExpressions {
    private TensorUnaryExpressions() {
    }

    static Tensor apply(Tensor input, UnaryElementwiseKind kind) {
        // exact construction contract below
    }
}
```

The helper has no fields, nested types, public/protected members, overloads, caches, registries, or
kind-specific branches. Its constructor prevents instantiation. `apply` is package-private and
static so [Tensor](../../../../glossary.md#tensor) can delegate without exposing an independent public expression service.

### [Validation](../../../../glossary.md#term-validation) and construction order

`apply` performs these steps in exact order:

1. require non-null `input` and `kind`, in that order, with messages `input` and `kind`;
2. read the input [descriptor](../../../../glossary.md#term-descriptor)'s exact [`DataType`](../../../../glossary.md#data-type--datatype) and reject it when `isFloating()` is false with
   `IllegalArgumentException("input must be a floating data type, but was " + dataType)`;
3. create exactly one `TensorDescriptor` from that same [data type](../../../../glossary.md#data-type--datatype), the exact input [shape](../../../../glossary.md#shape),
   `Optional.empty()` [layout](../../../../glossary.md#layout), and the exact input `requiresGrad` flag;
4. create exactly one `Operation(kind, NoOperationAttrs.INSTANCE)`;
5. create exactly one `TensorProvenance(operation, List.of(input))`;
6. call `TensorFactory.createDerived(descriptor, Optional.empty(), provenance)` exactly once and
   return its exact result.

Null and data-type validation complete before ID allocation. Do not catch, translate, aggregate,
or replace failures. Validation failures before `createDerived` consume no [Tensor](../../../../glossary.md#tensor) ID. Exhausted
Tensor [identity](../../../../glossary.md#term-identity-meanings) space fails through `createDerived` only after the local [immutable](../../../../glossary.md#term-immutable) [model](../../../../glossary.md#term-model) values
have been constructed. Do not add a production ID-inspection hook merely to test ordering.

### Result [descriptor](../../../../glossary.md#term-descriptor)

For every successful expression:

- `dataType` is the exact enum value from the input descriptor;
- [`shape`](../../../../glossary.md#shape) is the exact [immutable](../../../../glossary.md#term-immutable) `Shape` reference from the input descriptor;
- [`layout`](../../../../glossary.md#layout) is empty for static, zero-sized, [scalar](../../../../glossary.md#term-scalar), and dynamic shapes; and
- `requiresGrad` equals the input descriptor's exact value.

The result descriptor is a new immutable value. It does not retain the input descriptor itself,
input layout, [host storage](../../../../glossary.md#host-storage), physical geometry, or a [materialization](../../../../glossary.md#materialization) decision.

### [Provenance](../../../../glossary.md#provenance) and [identity](../../../../glossary.md#term-identity-meanings)

The output is a fresh [Tensor](../../../../glossary.md#tensor) with:

- a new factory-assigned [opaque](../../../../glossary.md#term-opaque) [`TensorId`](../../../../glossary.md#tensorid);
- an empty label;
- no [host storage](../../../../glossary.md#host-storage);
- one exact `UnaryElementwiseKind` paired with `NoOperationAttrs.INSTANCE`; and
- [immutable](../../../../glossary.md#term-immutable) provenance containing exactly the one exact input reference.

Repeating an expression creates another Tensor identity. Chaining retains the immediately
preceding derived Tensor as the next expression's exact input. This task performs no interning,
[common-subexpression elimination](../../../../glossary.md#term-common-subexpression-elimination-cse), ancestor substitution, or [graph](../../../../glossary.md#graph) traversal.

### No [eager](../../../../glossary.md#term-eager-evaluation) domain checks or [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form)

Every valid floating [Tensor](../../../../glossary.md#tensor) creates the requested semantic expression regardless of current
storage, values, or [provenance](../../../../glossary.md#provenance). For example, [`log`](../../../../glossary.md#term-log) and `sqrt` do not inspect whether values are in a
real-valued domain; `inv` does not inspect zero; and `inv().inv()` does not return the original
Tensor. Strict and fast variants stay distinct. [Compiler](../../../../glossary.md#term-compiler) optimization and [backend](../../../../glossary.md#backend) numerical
contracts may later make valid decisions using [immutable](../../../../glossary.md#term-immutable) [graph](../../../../glossary.md#graph) semantics and selected execution
capabilities.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorUnaryExpressions.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorUnaryElementwiseTest.java`

Documentation/[planning](../../../../glossary.md#term-planning) during implementation:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a task-related inconsistency requires stopping:

- `docs/api/training-api.md`
- `docs/planning/modules/model/capabilities.md`
- Existing [`DataType`](../../../../glossary.md#data-type--datatype), [`Shape`](../../../../glossary.md#shape), `TensorDescriptor`, `TensorProvenance`, `TensorFactory`,
  [`Operation`](../../../../glossary.md#operation), [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), and `UnaryElementwiseKind` Javadocs and tests.
- Focused architecture documents, ADRs, architecture tests, backend-conformance tests, and
  integration tests.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

At most two production files, two test files, and six documentation/[planning](../../../../glossary.md#term-planning) files: ten paths
total. The tenth path, `docs/api/compile-api.md`, was explicitly authorized after the independent
documentation review found its public-[Tensor](../../../../glossary.md#tensor) expression status stale.

`Tensor.java` and `TensorTest.java` may change only for the fifteen exact public methods, their
Javadocs, exact [API](../../../../glossary.md#term-application-programming-interface-api)-[shape](../../../../glossary.md#shape) expectations, and non-[synchronization](../../../../glossary.md#term-synchronization) assertions. Do not change existing
fields, constructor, [metadata](../../../../glossary.md#term-metadata)/storage behavior, [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) expression methods, equality, hashing,
diagnostics, or unrelated tests.

If implementation needs another production/test concept, a changed foundational contract, a
resolved-[layout](../../../../glossary.md#layout) or [gradient](../../../../glossary.md#term-gradient) policy, storage/value access, [graph](../../../../glossary.md#graph)/[compiler](../../../../glossary.md#term-compiler) behavior, another
documentation file, or more than ten paths, stop and propose a follow-up or architecture decision.
The [Compile](../../../../glossary.md#compile) API edit is limited to current-status wording and must not introduce compiler behavior.
Do not create task 0014E.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Update Tensor type Javadoc only as needed to explain that public binary and unary expression
  methods build storage-free provenance without making Tensor an IR node or executable value.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [unary glossary](../../../../glossary.md#term-unary); [provenance glossary](../../../../glossary.md#provenance); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [node glossary](../../../../glossary.md#node))</span>
- Every new public method must explain its elementwise mathematical or activation meaning,
  floating-only eligibility, exact type and shape preservation, unresolved result layout, exact
  gradient-eligibility propagation, fresh identity, storage absence, exact semantic kind,
  one-input provenance, and deferral of numerical/domain/gradient/backend behavior.<span data-glossary-navigation="true"> ([elementwise glossary](../../../../glossary.md#term-elementwise); [activation glossary](../../../../glossary.md#term-activation); [shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient); [backend glossary](../../../../glossary.md#backend))</span>
- Every public method must document the fresh derived [Tensor](../../../../glossary.md#tensor) with `@return`, the non-floating input
  failure with `@throws IllegalArgumentException`, and [identity](../../../../glossary.md#term-identity-meanings) exhaustion with
  `@throws IllegalStateException`. Zero-[argument](../../../../glossary.md#term-argument) instance methods must not invent an `@param` tag.
- The strict and fast method Javadocs must state that the kinds are distinct requests and must not
  promise a specific approximation, accuracy, or backend implementation.
- Document the package-private helper and its `apply` method with exact validation/construction
  order, reference retention, ownership, ID side effects, and failure behavior. The private
  constructor needs meaningful documentation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Avoid fifteen copies of unexplained terminology: each method remains complete, while type-level
  explanation and links may centralize shared mechanics where generated Javadoc stays clear.
- Review related foundational Javadocs and record why they remain accurate, or stop on an
  out-of-scope inconsistency.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

## Acceptance criteria

- Tensor declares exactly the fifteen new public zero-argument methods returning Tensor; no
  overload or unrelated public API is added.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Every method maps to the exact specified `UnaryElementwiseKind` and delegates once to the shared
  package-private helper.
- `TensorUnaryExpressions` has exactly the specified visibility, finality, constructor, method,
  and zero-field surface.
- Helper null [validation](../../../../glossary.md#term-validation) order and messages are exact.
- All methods accept [`BFLOAT16`](../../../../glossary.md#term-bfloat16), [`FLOAT32`](../../../../glossary.md#term-float32), and [`FLOAT64`](../../../../glossary.md#term-float64), retain the exact input data type, and
  reject <span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype))</span>[`INT32`](../../../../glossary.md#data-type--datatype), [`INT64`](../../../../glossary.md#data-type--datatype), and [`BOOL`](../../../../glossary.md#data-type--datatype) with the exact message before [identity](../../../../glossary.md#term-identity-meanings) allocation.
- Scalar, zero-sized, ordinary static, and dynamic shapes are accepted and retain the exact input
  shape reference.<span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar); [shapes glossary](../../../../glossary.md#shape))</span>
- Every result [descriptor](../../../../glossary.md#term-descriptor) has empty [layout](../../../../glossary.md#layout) and `requiresGrad` equal to the input flag for all
  fifteen kinds.
- Every result is a fresh, unlabeled, storage-free [Tensor](../../../../glossary.md#tensor) with exact kind,
  `NoOperationAttrs.INSTANCE`, and exactly one [immutable](../../../../glossary.md#term-immutable) exact input reference.
- Repeated calls are not interned; chains preserve the immediately previous Tensor; double
  reciprocal, double negation, nested activations, and strict/fast requests are not canonicalized.<span data-glossary-navigation="true"> ([activations glossary](../../../../glossary.md#term-activation))</span>
- No method reads numeric values or rejects a represented mathematical domain before execution.
- No input metadata, provenance, label, storage association, or storage contents are mutated or
  retained as output storage.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance))</span>
- No numerical execution, result values, gradient rules, graph state, backend facts, dependency,
  or architecture change is added.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [graph glossary](../../../../glossary.md#graph); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Focused and aggregate model tests, model Javadoc, root tests, reflection/javap/import/bytecode/
  scope checks, documentation links/formatting, and status synchronization pass.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- A separate clean-context documentation-focused agent finalizes Javadocs, Tensor API, the
  authorized Compile API status correction, glossary, task evidence, master plan, and roadmap in
  the same change and records reasoned no-change conclusions for related APIs, capabilities,
  architecture, and existing contracts.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- Task 0014D becomes Complete only after both passes. Task 0014E remains Draft without a detailed
  specification.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorUnaryElementwiseTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test must cover:

- exact helper class, constructor, method, field, visibility, and finality [shape](../../../../glossary.md#shape);
- all fifteen public method-to-kind mappings and exact [singleton](../../../../glossary.md#term-singleton) attributes reference;
- exact one-input [provenance](../../../../glossary.md#provenance) reference for every method and across a chain;
- all three floating types with exact type and shape-reference preservation;
- [scalar](../../../../glossary.md#term-scalar), zero-sized, static resolved/unresolved, and dynamic input shapes;
- empty result [layout](../../../../glossary.md#layout), exact [gradient](../../../../glossary.md#term-gradient)-eligibility propagation including `floor`, `ceil`, and
  `sign`, empty label, absent [host storage](../../../../glossary.md#host-storage), and fresh [identity](../../../../glossary.md#term-identity-meanings);
- helper null input/kind failures, every non-floating type, exact messages, and no ID consumption;
- absence of domain inspection and canonicalization, including repeated, nested, strict, and fast
  requests; and
- preservation of input [metadata](../../../../glossary.md#term-metadata), provenance, labels, storage associations, and storage contents.

Manually inspect `javap -p -c -s` and reflection for the exact [Tensor](../../../../glossary.md#tensor) method [descriptors](../../../../glossary.md#term-descriptor) and helper
surface. Inspect source/[bytecode](../../../../glossary.md#term-bytecode) for one helper delegation per public method and exact local
construction order. Scan production imports and [Gradle](../../../../glossary.md#term-gradle) dependencies for forbidden layers. Confirm
that no storage accessor, segment [API](../../../../glossary.md#term-application-programming-interface-api), shape algebra, [graph](../../../../glossary.md#graph) ID, [compiler](../../../../glossary.md#term-compiler)/[runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) type,
gradient rule, domain check, [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form), [cache](../../../../glossary.md#term-cache), [registry](../../../../glossary.md#term-registry), or service [lookup](../../../../glossary.md#term-lookup) appears. Validate
generated [Javadoc](../../../../glossary.md#term-javadoc), Tensor and [Compile](../../../../glossary.md#compile) API current-versus-planned wording and newcomer example,
glossary terminology, links/anchors/fences/whitespace, exact ten-path [scope](../../../../glossary.md#term-scope), synchronized statuses,
and
absence of a task-0014E specification.

## Dependencies

- Task 0001 supplies exact floating data types and differentiability [metadata](../../../../glossary.md#term-metadata).
- Task 0002 supplies [immutable](../../../../glossary.md#term-immutable) static and dynamic [`Shape`](../../../../glossary.md#shape) values.
- Task 0007 supplies immutable result [descriptors](../../../../glossary.md#term-descriptor).
- Task 0011 supplies the public [Tensor](../../../../glossary.md#tensor) surface.
- Task 0012 and task 0013 supply factory [identity](../../../../glossary.md#term-identity-meanings) allocation, `createDerived`, and immutable
  provenance.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- Task 0014C supplies the exact fifteen [unary](../../../../glossary.md#term-unary) [elementwise](../../../../glossary.md#term-elementwise) semantic kinds.

## Follow-up tasks

- Task 0014E remains Draft for typed [scalar](../../../../glossary.md#term-scalar) arithmetic and clamp semantic contracts.
- Task 0014F remains Draft for matching [Tensor](../../../../glossary.md#tensor) expressions.
- Compiler capture, optimizer rewrites, autograd expansion, capability analysis, backend ownership,
  numerical kernels, approximation contracts, and conformance tests remain in their owning module
  tasks.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [optimizer glossary](../../../../glossary.md#term-optimizer); [autograd glossary](../../../../glossary.md#autograd); [backend ownership glossary](../../../../glossary.md#backend-ownership); [kernels glossary](../../../../glossary.md#kernel); [conformance glossary](../../../../glossary.md#term-conformance); [module glossary](../../../../glossary.md#term-module))</span>

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None. The architecture already assigns public [Tensor](../../../../glossary.md#tensor) expression semantics,
[backend](../../../../glossary.md#backend)-independent [Operation](../../../../glossary.md#operation) values, [descriptors](../../../../glossary.md#term-descriptor), and [provenance](../../../../glossary.md#provenance) to `modules/model`, while
forbidding [graph](../../../../glossary.md#graph) [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names), physical allocation, backend support, and execution there.

If implementation requires Tensor to become [IR](../../../../glossary.md#term-intermediate-representation-ir), resolved physical [layout](../../../../glossary.md#layout), storage/value access,
[compiler](../../../../glossary.md#term-compiler)/[autograd](../../../../glossary.md#autograd) logic, backend facts, another [module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency), or a changed architecture rule,
stop and report the issue.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0002/0003/0006/0007/0011/0012/0013/0014B/0014C/0014D,
Tensor API, Compile API, Training API, glossary, current DataType/Shape/TensorDescriptor/Tensor/
TensorFactory/TensorProvenance/Operation/UnaryElementwiseKind contracts and tests, and Java 26
Gradle configuration.

Implement task 0014D exactly. Modify Tensor.java and add package-private final
TensorUnaryExpressions.java for production. Update TensorTest only for the exact fifteen-method
API surface and add TensorUnaryElementwiseTest. Add exactly abs/neg/inv/log/exp/erf/sqrt/floor/
ceil/sign/relu/sigmoid/tanh/fastExp/fastTanh(), each delegating once to the shared helper and exact
matching kind.

The helper must null-check input/kind, accept only floating DataType, retain exact input data type
and Shape reference, create an empty-layout descriptor with unchanged requiresGrad, create
Operation(kind, NoOperationAttrs.INSTANCE), preserve exact one-input provenance, and delegate once
to TensorFactory.createDerived with no label/storage. Every valid call returns a fresh derived
Tensor. Do not inspect values/storage, execute mathematics, check numeric domains, canonicalize,
resolve layout, insert casts, define gradient rules, capture a graph, add overloads, change existing
contracts, or introduce compiler/runtime/backend behavior.

Stop beyond ten paths or on architecture uncertainty. The tenth path is limited to correcting
Compile API implementation-status wording without adding compiler behavior. Run every specified
focused/aggregate test,
Javadoc, javap/reflection/bytecode/import/manual, documentation/link/whitespace/scope/status check.
Then hand the actual diff and evidence to a separate clean-context documentation agent in the same
change. It must inspect source/tests/generated Javadoc, finalize permitted Javadocs/Tensor API/
Compile API/glossary/planning, record related-contract/capability/Training API/architecture
no-change conclusions, and rerun validation.

Update task 0014D, model master plan, and roadmap only for planning status/evidence. Do not mark
0014D Complete until both passes succeed. Leave 0014E Draft without a specification. Do not commit
or push.
```

## Local decisions

- All fifteen capabilities use one package-private helper because they share arity,
  parameterlessness, floating validation, shape preservation, descriptor, and provenance rules.<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [validation glossary](../../../../glossary.md#term-validation); [shape glossary](../../../../glossary.md#shape); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- The exact floating input type is retained. Unary expressions have no second operand requiring
  promotion and do not insert casts.<span data-glossary-navigation="true"> ([Unary glossary](../../../../glossary.md#term-unary))</span>
- The exact immutable input Shape is retained while layout becomes unresolved. Logical shape
  preservation does not imply preservation of physical geometry or materialization.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [layout glossary](../../../../glossary.md#layout); [materialization glossary](../../../../glossary.md#materialization))</span>
- Gradient eligibility propagates unchanged for every kind. This metadata tracks the caller's
  request through an expression dependency and does not encode a derivative or gradient rule.<span data-glossary-navigation="true"> ([Gradient glossary](../../../../glossary.md#term-gradient); [metadata glossary](../../../../glossary.md#term-metadata); [dependency glossary](../../../../glossary.md#term-dependency); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- Every valid request creates a fresh expression. Legacy eager double-inverse canonicalization and
  other value/provenance rewrites belong to future compiler optimization, not mutable Tensor state.<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [compiler glossary](../../../../glossary.md#term-compiler); [mutable glossary](../../../../glossary.md#term-mutable); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Strict and fast semantics stay distinct without selecting an algorithm or promising accuracy.
- Output labels are empty. Typed Operation and exact provenance carry meaning instead of generated
  expression strings.<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation); [carry glossary](../../../../glossary.md#term-carry))</span>

## Known limitations

- The returned Tensor has no computed values or host storage and cannot execute without future
  compiler, prepare, runtime, and backend work.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [host storage glossary](../../../../glossary.md#host-storage); [compiler glossary](../../../../glossary.md#term-compiler); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend))</span>
- Only floating unary expressions are supported. Integral/boolean transformations and explicit
  casts remain separate planned capabilities.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [boolean glossary](../../../../glossary.md#data-type--datatype))</span>
- The output [layout](../../../../glossary.md#layout) is unresolved even when input [shape](../../../../glossary.md#shape) and layout are static.
- No domain [validation](../../../../glossary.md#term-validation) or numerical edge behavior is promised before execution.
- No [gradient](../../../../glossary.md#term-gradient) rule is attached. `requiresGrad` is eligibility [metadata](../../../../glossary.md#term-metadata) only, including for
  `floor`, `ceil`, and `sign`.
- `FAST_EXP` and `FAST_TANH` identify approximation intent but define no error bound, algorithm, or
  backend availability.
- Repeated equivalent expressions receive distinct Tensor identities until a future compiler
  chooses a valid graph optimization.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([graph optimization glossary](../../../../glossary.md#forward-graph-optimization))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) reviewed the architecture and documentation rules; planning guide, [model](../../../../glossary.md#term-model) capability
baseline, master plan, and roadmap; completed tasks 0001, 0002, 0003, 0006, 0007, 0011, 0012,
0013, 0014B, and 0014C; current [DataType](../../../../glossary.md#data-type--datatype), [Shape](../../../../glossary.md#shape), TensorDescriptor, [Tensor](../../../../glossary.md#tensor),
TensorBinaryExpressions, TensorFactory.createDerived, TensorProvenance, [Operation](../../../../glossary.md#operation),
[NoOperationAttrs](../../../../glossary.md#nooperationattrs), and UnaryElementwiseKind source/tests; and the read-only legacy Tensor/TensorOps
[unary](../../../../glossary.md#term-unary) surface, unary builders, primitive-builder [gradient](../../../../glossary.md#term-gradient) behavior, [dtype](../../../../glossary.md#data-type--datatype) validation, and
[canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form) tests.

Planning confirmed that current contracts can implement all fifteen methods without a [dependency](../../../../glossary.md#term-dependency),
[Gradle](../../../../glossary.md#term-gradle), architecture, or foundational [API](../../../../glossary.md#term-application-programming-interface-api) change. Legacy capability selection retains floating,
shape-preserving semantics while explicitly rejecting [mutable](../../../../glossary.md#term-mutable) [graph](../../../../glossary.md#graph) state, [eager](../../../../glossary.md#term-eager-evaluation) storage,
gradient callbacks, expression labels, [kernel](../../../../glossary.md#kernel)/[fusion](../../../../glossary.md#term-fusion) coupling, and canonicalization. The planned
package direction remains acyclic.

Planning validation:

- `git diff --check` passed, and the three changed planning files contain no trailing whitespace.
- The [canonical](../../../../glossary.md#term-canonical-form) section scan found every required task-specification section.
- The relative Markdown-[target](../../../../glossary.md#term-target) scan resolved every local `.md` link in this task, the model master
  plan, and the roadmap.
- Status inspection found task 0014D [`Ready`](../../../../glossary.md#term-ready) exactly once in this specification, its model-master
  row, and its roadmap row.
- Scope inspection found exactly this new task plus the model master plan and roadmap changed; no
  Java, test, Gradle, AGENTS, architecture, API, glossary, or other module file changed during
  planning.<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [module glossary](../../../../glossary.md#term-module))</span>
- No task-0014E specification exists; 0014E remains only a Draft queue entry.

Implementation and documentation validation:

- Implementation context `/root/implement_model_0014d` added the fifteen exact public Tensor
  methods, package-private `TensorUnaryExpressions`, the focused unary-expression suite, and only
  the required Tensor API-shape assertions. Independent documentation context
  `/root/implement_model_0014b` then performed a mandatory fresh reread of the architecture,
  documentation/planning rules, current APIs, glossary, task chain, actual source/tests, generated
  reports, bytecode, and complete diff. It applied General plus API/Javadoc style to Java and API
  reference review, Planning style to planning files, and Example format to the new Tensor API
  example.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- The documentation review initially stopped without writing when it found that the Compile API
  still described only binary expression construction as current. Explicit user authorization
  expanded the task from nine to ten paths solely for that page's implementation-status wording.
  The final edit recognizes current binary and unary public Tensor expressions while preserving
  compiler entry, traversal, capture, inference, optimization, artifacts, and execution as planned.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [artifacts glossary](../../../../glossary.md#term-artifact))</span>
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorUnaryElementwiseTest` passed; XML reports 8 tests,
  zero failures, errors, or skips.
- `./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorTest` passed;
  XML reports 14 tests, zero failures, errors, or skips.
- `./gradlew :modules:model:test` passed; 38 XML suites report 287 tests with zero failures, errors,
  or skips.
- `./gradlew :modules:model:javadoc` passed. Generated `Tensor.html` contains the Tensor-level
  binary/unary boundary and all fifteen method contracts, including floating eligibility, exact
  type/shape and gradient-eligibility retention, unresolved layout, fresh identity, storage
  absence, exact semantic kind and input provenance, failures, and strict/fast limits. Standard
  public Javadoc omits the package-private helper; its type, private constructor, and <span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance))</span>`apply`
  contract were reviewed directly in source and already satisfy the API/Javadoc profile, so the
  documentation pass made no Java edit.
- `./gradlew test` passed for the repository; the final run reported all 36 actionable tasks
  up-to-date and no failing task.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- The complete `UnaryExpressionExample` added to the Tensor API compiled with Java 26 against
  `model-0.1.0-SNAPSHOT.jar` and printed the documented [`FLOAT32`](../../../../glossary.md#term-float32), exact-shape, unresolved-layout,
  gradient, label, storage, `FAST_EXP`, canonical-attributes, exact-input, and fresh-result values.
- `javap -p -c -s` confirmed all fifteen exact zero-[argument](../../../../glossary.md#term-argument) `()Tensor` methods, one matching enum
  constant and one helper call per method, and a final zero-field helper with one private
  constructor and one package-private static <span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>`apply`. Helper bytecode confirms input/kind null
  checks, floating validation, exact type/shape/gradient descriptor construction, canonical
  parameterless Operation, one-input provenance, and one <span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor))</span>`createDerived` call in order.
- Reflection tests confirm the exact Tensor/helper API, visibility, finality, zero fields and
  nested types, non-synchronization, method-to-kind mapping, and all result/identity contracts.
  Import and source scans found only permitted same-module and JDK types and no storage access,
  segment API, shape algebra, graph ID/capture, compiler/runtime/backend type, gradient rule,
  domain check, canonicalization, cache, registry, or service lookup.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [cache glossary](../../../../glossary.md#term-cache); [registry glossary](../../../../glossary.md#term-registry); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- The corrected local Markdown validator checked 196 local file targets and heading anchors across
  the six changed documentation/planning files with zero errors. A preliminary validator run
  exposed Ruby interpolation in a heading regular expression, and the next run exposed that
  GitHub preserves adjacent heading spaces as a double hyphen; neither run changed repository
  files. The corrected check passed. Fence counts are balanced (`72/0`, `4/0`, `0/0`, `8/0`,
  `2/0`, and `0/0` backtick/tilde fences respectively), targeted trailing-whitespace scans found
  no matches, and `git diff --check` passed.
- Final scope contains exactly the ten authorized paths: two production files, two test files,
  Tensor API, the narrowly authorized Compile API correction, glossary, this task, model master
  plan, and roadmap. No Training API, capabilities, architecture/ADR/test, Gradle, dependency,
  backend-conformance, integration-test, or task-0014E specification path changed.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Task 0014D is synchronized as Complete in this specification, the model master plan, and the
  roadmap. Task 0014E remains the next Draft frontier without a detailed specification.
- `DataType` remains accurate because the helper consumes its existing floating-category query
  without changing differentiability metadata. <span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>`Shape` remains accurate because its immutable
  identity is retained without shape algebra. <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>`TensorDescriptor` remains accurate because the new
  result uses its existing unresolved-layout and gradient-eligibility contracts.
- `TensorProvenance`, `TensorFactory.createDerived`, `Operation`, `NoOperationAttrs`, and
  `UnaryElementwiseKind` remain accurate because the helper composes their existing exact-reference,
  identity-allocation, parameterless, typed-kind, and one-input boundaries without changing them.
  Binary expression contracts and tests also remain accurate and unchanged.
- `capabilities.md` requires no edit because it already selects all fifteen public methods and
  distinguishes model/public-expression support from compiler/backend/runtime execution. The
  Training API requires no edit because propagating `requiresGrad` is eligibility metadata only;
  no gradient object, rule, autograd, optimizer, or training behavior changed.<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- `ARCHITECTURE.md`, focused architecture pages, ADRs, architecture tests, backend-conformance and
  integration tests, and build configuration require no edit because the implementation remains
  within model-owned public expression semantics and adds no module boundary, dependency,
  lifecycle, backend behavior, numerical execution, Java toolchain, preview/incubator, or
  end-to-end contract.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [toolchain glossary](../../../../glossary.md#term-toolchain))</span>

## Implementation notes

- Added exactly fifteen public zero-[argument](../../../../glossary.md#term-argument) [Tensor](../../../../glossary.md#tensor) methods, each delegating once to the matching
  `UnaryElementwiseKind` through one stateless package-private helper.
- The helper performs the specified null and floating validation, exact data type/Shape/gradient
  retention, unresolved descriptor construction, canonical parameterless Operation, exact
  one-input provenance, and one derived-factory delegation without value or storage access.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [data type glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [gradient glossary](../../../../glossary.md#term-gradient); [descriptor glossary](../../../../glossary.md#term-descriptor); [canonical glossary](../../../../glossary.md#term-canonical-form); [Operation glossary](../../../../glossary.md#operation); [provenance glossary](../../../../glossary.md#provenance))</span>
- The independent documentation pass found the implementation Javadocs complete, added the current
  unary-expression mental model, method table, runnable example, failures, ownership summary, and
  glossary distinctions, and corrected only the explicitly authorized Compile API status wording.<span data-glossary-navigation="true"> ([mental model glossary](../../../../glossary.md#term-mental-model))</span><span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- No Java declaration, executable logic, or test changed during the documentation pass.

## Completion summary

- Completed changes: Implemented and documented all fifteen floating unary elementwise Tensor
  expression methods with exact descriptor preservation and one-input provenance.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [elementwise glossary](../../../../glossary.md#term-elementwise); [Tensor glossary](../../../../glossary.md#tensor); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- Files changed or created: Exactly two production files, two tests, Tensor API, Compile API,
  glossary, this task, model master plan, and roadmap.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: Focused unary 8/8, Tensor 14/14, all 287 model tests across 38 suites,
  model Javadoc, root tests, bytecode/reflection/import/absence checks, the compiled exact example,
  196 local links/anchors, fences, terminology, whitespace, exact scope/status, and<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>
  `git diff --check` passed.
- Documentation-agent review: Clean documentation context `/root/implement_model_0014b` completed
  the independent pass using General, API/Javadoc, Planning, and Example-format guidance.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Tensor API and glossary now describe current unary expressions; the
  explicitly authorized Compile API correction recognizes them without claiming compiler behavior.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler))</span>
- Javadoc review: Tensor type/all fifteen methods and the helper type/constructor/method are final;
  reused data type, shape, descriptor, provenance, factory, Operation, no-attributes, unary-kind,
  and binary-expression contracts remain accurate unchanged.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [shape glossary](../../../../glossary.md#shape); [Operation glossary](../../../../glossary.md#operation); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>
- Glossary impact: Existing implementation-status, OperationKind, Provenance, Tensor, and common
  distinctions now include unary expression construction; no new reusable domain term was needed.<span data-glossary-navigation="true"> ([OperationKind glossary](../../../../glossary.md#operationkind))</span>
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0014D. Task 0014E remains Draft without a specification.

Status: Complete

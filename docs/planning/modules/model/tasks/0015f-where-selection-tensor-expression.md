# Task 0015F: Where Selection [Tensor](../../../../glossary.md#tensor) Expression

## Status

Complete

## Goal

Expose the implemented `WHERE` semantic [identity](../../../../glossary.md#term-identity-meanings) through one public, [backend](../../../../glossary.md#backend)-independent
`Tensor.where(condition, ifTrue, ifFalse)` expression method. The condition must be [BOOL](../../../../glossary.md#data-type--datatype), the two
branches must be floating [tensors](../../../../glossary.md#tensor), all three [shapes](../../../../glossary.md#shape) must be locally broadcast-compatible, and the
result must use the promoted branch type. Every successful call returns a fresh storage-free
Tensor whose [provenance](../../../../glossary.md#provenance) preserves the exact ordered condition, true branch, and false branch.

This task makes conditional selection capturable through the public [model](../../../../glossary.md#term-model) [API](../../../../glossary.md#term-application-programming-interface-api). It does not read
condition or branch values, choose a branch eagerly, allocate result storage, create [gradient](../../../../glossary.md#term-gradient)
rules, [capture](../../../../glossary.md#term-capture-meanings) a [compiled graph](../../../../glossary.md#compiled-graph--compiledgraphmodel), or define backend support.

## [Scope](../../../../glossary.md#term-scope)

- Add exactly one public static `Tensor.where(Tensor, Tensor, Tensor)` method.
- Add one package-private final `TensorWhereExpressions` helper with one package-private static
  `apply` entry and no other behavior.
- Require an exact `DataType.BOOL` condition without numeric truthiness or implicit conversion.
- Validate and promote the two floating branch data types through exactly one
  `DataTypePromotion.promoteFloating` call.
- Derive the three-way result [shape](../../../../glossary.md#shape) through exactly two ordered `ShapeBroadcast.broadcast` calls:
  first true branch with false branch, then condition with the resulting branch shape.
- Create an unresolved-layout result descriptor with the promoted branch type and gradient
  eligibility equal to <span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [descriptor glossary](../../../../glossary.md#term-descriptor); [gradient glossary](../../../../glossary.md#term-gradient))</span>`ifTrue.requiresGrad || ifFalse.requiresGrad`.
- Construct exactly one [`Operation`](../../../../glossary.md#operation) from `WhereSelectionKind.WHERE` and
  `NoOperationAttrs.INSTANCE`.
- Construct exactly one `TensorProvenance` with ordered exact inputs
  `[condition, ifTrue, ifFalse]`.
- Delegate final [identity](../../../../glossary.md#term-identity-meanings)-bearing construction exactly once to `TensorFactory.createDerived` with
  no label and no storage.
- Update the exact [Tensor](../../../../glossary.md#tensor) public-[API](../../../../glossary.md#term-application-programming-interface-api) reflection test and add one focused where-expression test.
- Finalize affected Javadocs, Tensor API, Compile API status, glossary, task evidence, model master
  plan, and roadmap through the required independent documentation pass during implementation.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- eager value selection, condition or branch storage reads, branch evaluation order, lazy control
  flow, short-circuiting, constant folding, canonicalization, interning, or returning an input<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [evaluation glossary](../../../../glossary.md#term-evaluation); [lazy glossary](../../../../glossary.md#term-lazy-evaluation); [constant folding glossary](../../../../glossary.md#term-constant-folding); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- BOOL, INT32, or INT64 branches; numeric conditions; implicit conversion; explicit cast
  insertion; nullable conditions; or three-valued logic<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype))</span>
- a generic conditional operator, instance `where` method, overload, scalar branch, caller label,
  variadic API, operator alias, expression-builder public type, or factory method<span data-glossary-navigation="true"> ([variadic glossary](../../../../glossary.md#term-variadic); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- resolved output layout, a ternary broadcast-plan type, effective strides, aliases, view
  preservation, materialization policy, or input-layout propagation<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [strides glossary](../../../../glossary.md#term-stride); [view glossary](../../../../glossary.md#term-view-meanings); [materialization glossary](../../../../glossary.md#materialization))</span>
- gradient values, branch gradient routing, backward operations, autograd expansion, optimizer,
  training-root policy, or training execution<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- changes to `WhereSelectionKind`, [`Operation`](../../../../glossary.md#operation), [`OperationKind`](../../../../glossary.md#operationkind), [`OperationAttrs`](../../../../glossary.md#operationattrs),
  [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), `DataTypePromotion`, `ShapeBroadcast`, `TensorDescriptor`,
  `TensorProvenance`, or `TensorFactory`
- [scalar](../../../../glossary.md#term-scalar)-index `select`, gather, take, scatter, mask indexing, reduction, cast, or another operation
  family<span data-glossary-navigation="true"> ([gather glossary](../../../../glossary.md#gather); [mask glossary](../../../../glossary.md#term-boolean-mask))</span>
- graph traversal, cycle checks, node/value IDs, graph capture, compiled graph construction,
  common-subexpression elimination, compiler inference, publication, or optimization<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [node glossary](../../../../glossary.md#node); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [compiled graph glossary](../../../../glossary.md#compiled-graph--compiledgraphmodel); [common-subexpression elimination glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [compiler glossary](../../../../glossary.md#term-compiler); [inference glossary](../../../../glossary.md#term-inference); [publication glossary](../../../../glossary.md#term-publication))</span>
- planning ownership, capability providers, backend support, fusion, cost, lowering, kernels,
  runtime residency, prepare, execution, tracing, ONNX mapping, or engine behavior<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [backend glossary](../../../../glossary.md#backend); [fusion glossary](../../../../glossary.md#term-fusion); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel); [runtime glossary](../../../../glossary.md#term-runtime); [residency glossary](../../../../glossary.md#residency); [prepare glossary](../../../../glossary.md#prepare); [ONNX glossary](../../../../glossary.md#term-onnx); [engine glossary](../../../../glossary.md#term-engine))</span>
- dependencies, Gradle changes, architecture changes, another module, unrelated refactors, or a
  detailed task-0015G specification<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

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
- [Task 0006](0006-operation-model.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0011](0011-public-tensor-skeleton.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0015B](0015b-binary-comparison-tensor-expressions.md)
- [Task 0015C](0015c-boolean-logical-semantic-kinds.md)
- [Task 0015D](0015d-boolean-logical-tensor-expressions.md)
- [Task 0015E](0015e-where-selection-semantic-kind.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes static
`Tensor.where(condition, ifTrue, ifFalse)`. Its public builder requires a [BOOL](../../../../glossary.md#data-type--datatype) condition, accepts
all pairings of [BFLOAT16](../../../../glossary.md#term-bfloat16), [FLOAT32](../../../../glossary.md#term-float32), and [FLOAT64](../../../../glossary.md#term-float64) branches, promotes branch [precision](../../../../glossary.md#term-precision), and broadcasts
the condition and both branches to one result [shape](../../../../glossary.md#shape). Legacy tests cover branch [broadcasting](../../../../glossary.md#broadcasting),
condition broadcasting, mixed floating precision, comparison [masks](../../../../glossary.md#term-boolean-mask), non-[contiguous](../../../../glossary.md#term-contiguous) inputs,
expression chaining, [gradient](../../../../glossary.md#term-gradient) routing, [ONNX](../../../../glossary.md#term-onnx) mapping, [fusion](../../../../glossary.md#term-fusion), and several [backend](../../../../glossary.md#backend) routes.

Legacy code also creates [mutable](../../../../glossary.md#term-mutable) ternary broadcast plans, installs backward callbacks on [Tensor](../../../../glossary.md#tensor),
reads storage during execution, and exposes [runtime](../../../../glossary.md#term-runtime)/backend traits. Those mechanisms are not
copied. This task retains only local public expression construction through [immutable](../../../../glossary.md#term-immutable) [descriptor](../../../../glossary.md#term-descriptor)
and [provenance](../../../../glossary.md#provenance) values. [Compiler](../../../../glossary.md#term-compiler) [autograd](../../../../glossary.md#autograd) owns gradient expansion; [planning](../../../../glossary.md#term-planning) and backend [prepare](../../../../glossary.md#prepare) own
[materialization](../../../../glossary.md#materialization) and executable choices; concrete backends own condition interpretation and value
selection.

## Architecture constraints

- [`Tensor`](../../../../glossary.md#tensor) remains public [mutable](../../../../glossary.md#term-mutable) [API](../../../../glossary.md#term-application-programming-interface-api) state and must not become an [IR](../../../../glossary.md#term-intermediate-representation-ir) [node](../../../../glossary.md#node).
- `TensorWhereExpressions` performs deterministic local model validation only. It must not inspect
  values or storage, traverse provenance, capture a graph, infer backend support, or execute
  conditional selection.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [validation glossary](../../../../glossary.md#term-validation); [provenance glossary](../../../../glossary.md#provenance); [capture glossary](../../../../glossary.md#term-capture-meanings); [graph glossary](../../../../glossary.md#graph))</span>
- [`Operation`](../../../../glossary.md#operation) owns only the exact [backend](../../../../glossary.md#backend)-independent `WHERE` kind and canonical no-attributes
  value. It contains no inputs, arity field, result descriptor, broadcast plan, or executable
  behavior.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [arity glossary](../../../../glossary.md#term-arity); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- `TensorProvenance` owns exact ordered input [identities](../../../../glossary.md#term-identity-meanings). Order is always
  `[condition, ifTrue, ifFalse]`; no branch or condition reordering is allowed.
- Result [identity](../../../../glossary.md#term-identity-meanings) comes only from the existing package-private `TensorFactory.createDerived` seam.
  No second allocator, caller-supplied ID, registry, cache, or service is introduced.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [cache glossary](../../../../glossary.md#term-cache))</span>
- Every result is storage-free, unlabeled, unresolved-layout model state. No physical buffer,
  alias, view, or input storage is attached.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [alias glossary](../../../../glossary.md#aliasing-and-arena); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- The result [data type](../../../../glossary.md#data-type--datatype) is the exact output of the existing floating promotion contract applied to
  `ifTrue` and `ifFalse`. The [BOOL](../../../../glossary.md#data-type--datatype) condition never participates in promotion.
- Result gradient eligibility is the logical OR of the two branch descriptor requests. The BOOL
  condition does not contribute, and eligibility does not imply that a gradient rule exists.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient))</span>
- Three-way dynamic [shapes](../../../../glossary.md#shape) are accepted only where two existing pairwise `ShapeBroadcast` calls
  prove compatibility locally. The helper creates no symbolic constraints or ternary shape type.
- The output [layout](../../../../glossary.md#layout) remains unresolved even when every input layout is resolved or identical.
- Package direction is `model.tensor -> model.operation.elementwise.selection`, plus existing
  `model.tensor -> model.operation`, `model.datatype`, and `model.shape`. The selection package
  must not import Tensor, and no package cycle may be introduced.
- Stop if implementation requires a changed foundational contract, new broadcast-plan or
  attributes type, storage access, graph capture, gradient rule, dependency, or architecture
  decision.<span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns the public static where surface, local validation,
  result construction, provenance, and package-private derived Tensor factory seam.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [provenance glossary](../../../../glossary.md#provenance); [Tensor factory glossary](../../../../glossary.md#tensor-factory))</span>
- `io.github.pho001.synaptik.model.operation` — supplies [`Operation`](../../../../glossary.md#operation) and
  `NoOperationAttrs.INSTANCE`.
- `io.github.pho001.synaptik.model.operation.elementwise.selection` — supplies the exact
  `WhereSelectionKind.WHERE` semantic value.
- `io.github.pho001.synaptik.model.datatype` — supplies BOOL identity and floating branch
  promotion.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `io.github.pho001.synaptik.model.shape` — supplies immutable shapes and pairwise local
  right-aligned broadcasting.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [shapes glossary](../../../../glossary.md#shape); [broadcasting glossary](../../../../glossary.md#broadcasting))</span>

No package is added.

Type placement:

- `io.github.pho001.synaptik.model.tensor.Tensor` — public static conditional-selection surface;
  it receives only the exact `where` method and delegates all shared behavior.
- `io.github.pho001.synaptik.model.tensor.TensorWhereExpressions` — package-private, stateless
  validation and construction boundary colocated with Tensor, descriptors, provenance, and the
  package-private factory seam it must use.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- `TensorWhereSelectionTest` — same-package focused test so it can verify the package-private
  helper without widening production visibility.<span data-glossary-navigation="true"> ([widening glossary](../../../../glossary.md#term-widening))</span>

## Required contract

### Public [Tensor](../../../../glossary.md#tensor) surface

Add exactly this public static method to `Tensor`:

```java
public static Tensor where(Tensor condition, Tensor ifTrue, Tensor ifFalse)
```

It delegates exactly once and returns the exact result:

```java
return TensorWhereExpressions.apply(condition, ifTrue, ifFalse);
```

The public method performs no separate null check, type [validation](../../../../glossary.md#term-validation), promotion, [broadcasting](../../../../glossary.md#broadcasting),
[descriptor](../../../../glossary.md#term-descriptor) creation, allocation, [provenance](../../../../glossary.md#provenance) construction, [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form), or storage access.
There is no instance form, overload, [alias](../../../../glossary.md#aliasing-and-arena), or generic conditional [API](../../../../glossary.md#term-application-programming-interface-api) in this task.

### Package-private helper [shape](../../../../glossary.md#shape)

Create exactly one package-private final non-record class:

```java
final class TensorWhereExpressions {
    private TensorWhereExpressions() {
    }

    static Tensor apply(Tensor condition, Tensor ifTrue, Tensor ifFalse) {
        // exact contract below
    }
}
```

The helper has no fields, nested types, public/protected members, overloads, caches, registries,
or additional methods. Its private zero-[argument](../../../../glossary.md#term-argument) constructor prevents instantiation. A kind
[parameter](../../../../glossary.md#term-parameter) is unnecessary because the owning semantic family contains exactly `WHERE`.

### [Validation](../../../../glossary.md#term-validation) and construction order

`apply` performs these steps in exact order:

1. require non-null `condition`, `ifTrue`, and `ifFalse`, in that order, with messages
   `condition`, `ifTrue`, and `ifFalse`;
2. require `condition.descriptor().dataType() == DataType.BOOL`; otherwise throw
   `IllegalArgumentException` with message
   `condition must have BOOL data type, but was <actual>`;
3. call `DataTypePromotion.promoteFloating(ifTrue.descriptor().dataType(),
   ifFalse.descriptor().dataType())` exactly once and retain its exact result [data type](../../../../glossary.md#data-type--datatype);
4. call `ShapeBroadcast.broadcast(ifTrue.descriptor().shape(),
   ifFalse.descriptor().shape())` exactly once to derive the common branch [shape](../../../../glossary.md#shape);
5. call `ShapeBroadcast.broadcast(condition.descriptor().shape(), branchShape)` exactly once to
   derive the final three-way result shape;
6. create exactly one `TensorDescriptor` from the promoted branch data type, final result shape,
   `Optional.empty()` [layout](../../../../glossary.md#layout), and
   `ifTrue.descriptor().requiresGrad() || ifFalse.descriptor().requiresGrad()`;
7. create exactly one
   `Operation(WhereSelectionKind.WHERE, NoOperationAttrs.INSTANCE)`;
8. create exactly one `TensorProvenance(operation, List.of(condition, ifTrue, ifFalse))`;
9. call `TensorFactory.createDerived(descriptor, Optional.empty(), provenance)` exactly once and
   return its exact result.

Do not catch, translate, aggregate, or replace failures from `DataTypePromotion`,
`ShapeBroadcast`, [descriptor](../../../../glossary.md#term-descriptor) construction, [provenance](../../../../glossary.md#provenance), or the factory.

### Branch promotion failures

The existing shared promotion contract receives `ifTrue` as its left operand and `ifFalse` as its
right operand. Therefore:

- an invalid true branch fails with
  `IllegalArgumentException("left must be a floating data type, but was <actual>")`; and
- after a valid true branch, an invalid false branch fails with
  `IllegalArgumentException("right must be a floating data type, but was <actual>")`.

This task does not duplicate floating eligibility checks or change `DataTypePromotion` merely to
rename its diagnostic operands. Public [Javadoc](../../../../glossary.md#term-javadoc) explains the branch restriction without promising
a new promotion-message vocabulary.

### Three-way [broadcasting](../../../../glossary.md#broadcasting)

Three-way broadcasting is composition of the current pairwise local rule, not a new [model](../../../../glossary.md#term-model) type:

```text
branchShape = broadcast(ifTrue.shape, ifFalse.shape)
resultShape = broadcast(condition.shape, branchShape)
```

The first call proves branch compatibility before the condition [shape](../../../../glossary.md#shape) is combined. The second
call proves that the condition can address the common branch result. Equal symbolic [dimensions](../../../../glossary.md#dimension) and
static [singleton](../../../../glossary.md#term-singleton) expansion succeed according to `ShapeBroadcast`; different symbols and symbolic
versus non-singleton static sizes remain locally unprovable and fail. Pairwise order affects only
deterministic [validation](../../../../glossary.md#term-validation) and diagnostics, not [provenance](../../../../glossary.md#provenance) order.

No `WhereBroadcastPlan`, effective [stride](../../../../glossary.md#term-stride), resolved [layout](../../../../glossary.md#layout), or symbolic constraint is stored.

### Failure and [identity](../../../../glossary.md#term-identity-meanings) side effects

Null, condition-type, promotion, and both [shape](../../../../glossary.md#shape) validations complete before ID allocation:

- `Tensor.where(null, null, null)` fails with `NullPointerException("condition")`;
- after a valid condition reference, null branches fail in true-then-false order;
- a non-[BOOL](../../../../glossary.md#data-type--datatype) condition fails before branch data types or any shape is inspected;
- branch promotion fails before either broadcast call;
- incompatible branches fail before the condition is broadcast;
- a compatible branch shape with an incompatible condition fails in the second broadcast call;
  and
- exhausted [Tensor](../../../../glossary.md#tensor) identity space fails through `TensorFactory.createDerived` only after all local
  model values have been constructed.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>

Failures before `createDerived` consume no Tensor ID. Do not add a production ID-inspection hook,
catch an exception, or roll back an allocated identity.

### Result [descriptor](../../../../glossary.md#term-descriptor), [provenance](../../../../glossary.md#provenance), and [identity](../../../../glossary.md#term-identity-meanings)

Every successful result has:

- the exact promoted floating branch [data type](../../../../glossary.md#data-type--datatype);
- the final locally proven three-way broadcast [shape](../../../../glossary.md#shape);
- empty [layout](../../../../glossary.md#layout) for static and dynamic shapes;
- [gradient](../../../../glossary.md#term-gradient) eligibility equal only to the OR of true- and false-branch eligibility;
- a fresh factory-assigned [`TensorId`](../../../../glossary.md#tensorid);
- an empty label and no [host storage](../../../../glossary.md#host-storage);
- exact `WhereSelectionKind.WHERE` with `NoOperationAttrs.INSTANCE`; and
- exact [immutable](../../../../glossary.md#term-immutable) provenance inputs `[condition, ifTrue, ifFalse]`.

Using the same [Tensor](../../../../glossary.md#tensor) as both branches is valid and records that reference twice. Repeating a valid
call creates another identity. A comparison or logical expression may serve directly as the [BOOL](../../../../glossary.md#data-type--datatype)
condition. Input labels, provenance, layouts, storage associations, and storage contents remain
unchanged and are not copied to the result.

### No [eager](../../../../glossary.md#term-eager-evaluation) selection or [gradients](../../../../glossary.md#term-gradient)

The helper is eager only for expression [metadata](../../../../glossary.md#term-metadata). It never reads a condition byte, chooses or
evaluates a branch, copies or converts values, simplifies equal branches, reorders inputs, or
collapses nested selections. It propagates branch gradient eligibility as [descriptor](../../../../glossary.md#term-descriptor) metadata but
does not define gradient routing, create backward operations, or mutate any [Tensor](../../../../glossary.md#tensor) gradient state.
[Compiler](../../../../glossary.md#term-compiler) and training work later own those responsibilities.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorWhereExpressions.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorWhereSelectionTest.java`

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
- Existing [`DataType`](../../../../glossary.md#data-type--datatype), `DataTypePromotion`, [`Shape`](../../../../glossary.md#shape), `ShapeBroadcast`, `TensorDescriptor`,
  `TensorProvenance`, `TensorFactory`, [`Operation`](../../../../glossary.md#operation), [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), and
  `WhereSelectionKind` Javadocs/tests.
- Existing arithmetic, comparison, logical, unary, and scalar expression contracts; focused
  architecture documents; ADRs; architecture tests; backend-conformance tests; and integration
  tests.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [scalar glossary](../../../../glossary.md#term-scalar); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

At most two production files, two test files, and six documentation/[planning](../../../../glossary.md#term-planning) files: ten paths
total.

`Tensor.java` and `TensorTest.java` may change only for the one exact public static `where` method,
its import/Javadocs, exact [API](../../../../glossary.md#term-application-programming-interface-api)-[shape](../../../../glossary.md#shape) expectations, and non-[synchronization](../../../../glossary.md#term-synchronization) assertions. Do not
change existing fields, constructor, [metadata](../../../../glossary.md#term-metadata)/storage behavior, expression methods, equality,
hashing, diagnostics, or unrelated tests.

If implementation needs another production/test concept, a changed foundational contract, local
branch-specific promotion API, ternary broadcast-plan type, resolved-[layout](../../../../glossary.md#layout) policy, storage
access, [graph](../../../../glossary.md#graph)/[compiler](../../../../glossary.md#term-compiler) behavior, [gradient](../../../../glossary.md#term-gradient) rule, another documentation file, or more than ten
paths, stop and propose a follow-up or architecture decision. Do not create task 0015G.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Update Tensor type Javadoc only as needed to include conditional selection while preserving the
  distinction between public Tensor state, provenance, graph IR, gradient eligibility, and
  executable values.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [provenance glossary](../../../../glossary.md#provenance); [graph glossary](../../../../glossary.md#graph); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- `Tensor.where` must document the ordered condition/true-branch/false-branch roles, exact BOOL
  condition, floating promotion, ordered two-stage broadcasting, promoted result data type,
  unresolved layout, branch-only gradient eligibility OR, fresh identity, storage absence,
  provenance, and deferred value selection and gradient rules.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [broadcasting glossary](../../../../glossary.md#broadcasting); [data type glossary](../../../../glossary.md#data-type--datatype); [layout glossary](../../../../glossary.md#layout); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Document all three non-null inputs with `@param`, including exact roles, reference retention in
  provenance, and absence of mutation. Document the fresh result with `@return` and null,
  condition-type, promotion, shape, and identity-exhaustion failures with <span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>`@throws`.
- Document the package-private helper, private constructor, and `apply` method with exact
  validation/construction order, input ownership, ID side effects, and failure behavior.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Explain that expression metadata construction is eager but condition/branch value evaluation is
  not performed or specified by this model API.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [eager glossary](../../../../glossary.md#term-eager-evaluation); [evaluation glossary](../../../../glossary.md#term-evaluation); [model glossary](../../../../glossary.md#term-model); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Explain why conditional `where` is distinct from [scalar](../../../../glossary.md#term-scalar)-index `select` and why no broadcast plan
  or resolved output layout is created.
- Review related foundational and expression Javadocs and record why they remain accurate or stop
  on an out-of-scope inconsistency.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

## Acceptance criteria

- [Tensor](../../../../glossary.md#tensor) declares exactly one new `public static Tensor where(Tensor, Tensor, Tensor)` method; no
  instance form, overload, alias, or unrelated public API is added.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- The public method delegates once to `TensorWhereExpressions.apply` and performs no other work.
- `TensorWhereExpressions` has exactly the specified visibility, finality, private constructor,
  one package-private static method, zero fields, and zero nested types.
- Null checks, condition validation, promotion, two broadcasts, descriptor/operation/provenance
  construction, and factory delegation occur in the exact specified order.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [descriptor glossary](../../../../glossary.md#term-descriptor); [operation glossary](../../../../glossary.md#operation); [provenance glossary](../../../../glossary.md#provenance))</span>
- The condition must be exact BOOL. All nine floating branch pairs succeed and produce the exact
  shared promoted type. BOOL, INT32, and INT64 fail in either branch position without conversion,
  storage access, or ID allocation.<span data-glossary-navigation="true"> ([INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype))</span>
- Scalar, zero-sized, rank-mismatched, singleton-expanded, multi-axis, equal-symbolic, and
  symbolic/singleton three-way broadcasts succeed. Incompatible branches, condition/result
  shapes, different symbols, and symbolic/non-singleton pairs fail locally.<span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar); [rank glossary](../../../../glossary.md#term-rank); [singleton glossary](../../../../glossary.md#term-singleton); [axis glossary](../../../../glossary.md#term-axis); [shapes glossary](../../../../glossary.md#shape))</span>
- `ShapeBroadcast.broadcast` is invoked exactly twice per valid construction and in branch-first,
  condition-second order. No ternary plan, stride, layout, or constraint object is introduced.<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [layout glossary](../../../../glossary.md#layout))</span>
- Every result has promoted floating data type, empty layout, exact final shape, branch-only
  gradient eligibility OR, empty label, no host storage, fresh identity, exact operation and
  attributes, and exact ordered three-input provenance.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [gradient glossary](../../../../glossary.md#term-gradient); [host storage glossary](../../../../glossary.md#host-storage); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Comparison and logical [BOOL](../../../../glossary.md#data-type--datatype) expressions can feed the condition without special handling.
- Same-branch use and repeated calls remain fresh and ordered; equal branches are not simplified.
- No input Tensor metadata, provenance, label, storage association, or storage contents are
  mutated or retained as output storage.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>
- No value selection, evaluation-order policy, gradient rule, graph state, backend fact,
  dependency, build option, or architecture change is added.<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation); [graph glossary](../../../../glossary.md#graph); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Focused and aggregate model tests, model Javadoc, root tests, reflection/javap/import/bytecode/
  scope checks, documentation links/formatting, and status synchronization pass.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- A separate documentation-focused agent finalizes Javadocs, Tensor API, Compile API, glossary,
  task evidence, master plan, and roadmap in the same change and records reasoned no-change
  conclusions for Training API, capabilities, architecture, and related contracts.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- Task 0015F becomes Complete only after both passes. Task 0015G remains Draft without a detailed
  specification.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorWhereSelectionTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test must cover:

- exact helper class/constructor/method/field visibility and [signature](../../../../glossary.md#term-signature);
- exact public static `where` signature, visibility, non-[synchronization](../../../../glossary.md#term-synchronization), and one delegation;
- exact [Operation](../../../../glossary.md#operation) kind and [singleton](../../../../glossary.md#term-singleton) attributes references;
- exact ordered three-input provenance, including repeated branch references and derived BOOL
  conditions;<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- every floating branch pairing and exact promoted result type;
- exact [BOOL](../../../../glossary.md#data-type--datatype) condition eligibility and rejection of every non-BOOL type;
- [scalar](../../../../glossary.md#term-scalar), zero-sized, different-[rank](../../../../glossary.md#term-rank), singleton, multi-[axis](../../../../glossary.md#term-axis), and dynamic three-way [broadcasting](../../../../glossary.md#broadcasting);
- all four branch `requiresGrad` combinations and exclusion of the condition from propagation;
- empty [layout](../../../../glossary.md#layout), empty label, absent storage, fresh [identity](../../../../glossary.md#term-identity-meanings), and no [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form);
- helper/public nulls, invalid condition, invalid true/false branch, incompatible branch shapes,
  incompatible condition/result shapes, and unprovable dynamic shapes; and
- preservation of input descriptors, provenance, labels, layouts, storage associations, and
  contents.

Manually inspect `javap -p -c -s`, method [bytecode](../../../../glossary.md#term-bytecode), reflection, and imports for the exact [Tensor](../../../../glossary.md#tensor)
method [descriptor](../../../../glossary.md#term-descriptor), helper [shape](../../../../glossary.md#shape), sole delegation, null/type/promotion/broadcast order, exactly one
promotion call, exactly two broadcast calls in the specified order, branch-only [gradient](../../../../glossary.md#term-gradient) OR,
fixed `WHERE` operation, and absence of synchronization on new public/helper entries. Confirm no
numeric access, storage operation, cast, resolved layout, gradient rule, [graph](../../../../glossary.md#graph)/[compiler](../../../../glossary.md#term-compiler)/[runtime](../../../../glossary.md#term-runtime)/
[backend](../../../../glossary.md#backend) type, cost, [fusion](../../../../glossary.md#term-fusion), route, [registry](../../../../glossary.md#term-registry), service, [dependency](../../../../glossary.md#term-dependency), or build change appears. Validate
generated [Javadoc](../../../../glossary.md#term-javadoc), Tensor/[Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api) status, glossary, links/anchors/fences/whitespace, exact
ten-path [scope](../../../../glossary.md#term-scope), synchronized statuses, and absence of a task-0015G specification.

## Dependencies

- Task 0001 supplies exact [BOOL](../../../../glossary.md#data-type--datatype) [identity](../../../../glossary.md#term-identity-meanings) and floating `DataTypePromotion`.
- Task 0002 supplies [`Shape`](../../../../glossary.md#shape) and pairwise `ShapeBroadcast` local right-aligned [broadcasting](../../../../glossary.md#broadcasting).
- Task 0006 supplies [immutable](../../../../glossary.md#term-immutable) generic [`Operation`](../../../../glossary.md#operation) composition.
- Task 0007 supplies `TensorDescriptor` and its differentiability [validation](../../../../glossary.md#term-validation).
- Task 0011 supplies public [Tensor](../../../../glossary.md#tensor) state and the exact [API](../../../../glossary.md#term-application-programming-interface-api) surface to extend.
- Task 0012 supplies centralized Tensor identity allocation through `TensorFactory`.
- Task 0013 supplies immutable ordered [provenance](../../../../glossary.md#provenance) and `TensorFactory.createDerived`.
- Task 0015B supplies current comparison expressions that produce BOOL conditions.
- Task 0015D supplies current logical expressions that can also produce BOOL conditions.
- Task 0015E supplies exact parameterless `WhereSelectionKind.WHERE` semantics.

## Follow-up tasks

- 0015G remains Draft for typed cast semantic kind and attributes.
- 0015H remains Draft for explicit public [cast expression](../../../../glossary.md#cast-expression) construction.
- Compiler tasks later own provenance traversal, graph capture, optimization, autograd expansion,
  and training-graph treatment.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [provenance glossary](../../../../glossary.md#provenance); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [autograd glossary](../../../../glossary.md#autograd); [graph glossary](../../../../glossary.md#graph))</span>
- Backend, ONNX, and conformance tasks later own mapping, BOOL condition storage interpretation,
  value selection, lowering, kernels, and execution.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [ONNX glossary](../../../../glossary.md#term-onnx); [conformance glossary](../../../../glossary.md#term-conformance); [BOOL glossary](../../../../glossary.md#data-type--datatype); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel))</span>

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None. The architecture already assigns public [Tensor](../../../../glossary.md#tensor) state, [backend](../../../../glossary.md#backend)-independent
[Operation](../../../../glossary.md#operation) semantics, [descriptors](../../../../glossary.md#term-descriptor), and minimal [provenance](../../../../glossary.md#provenance) to `modules/model`. The helper composes
those existing contracts without adding [compiler](../../../../glossary.md#term-compiler), [runtime](../../../../glossary.md#term-runtime), backend, [device](../../../../glossary.md#term-device), or executable state.

If implementation requires [graph capture](../../../../glossary.md#capture-graph-ports-and-reconstruction), storage interpretation, a [gradient](../../../../glossary.md#term-gradient) rule, backend
[metadata](../../../../glossary.md#term-metadata), another [dependency](../../../../glossary.md#term-dependency), or architecture change, stop and report it.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0002/0006/0007/0011/0012/0013/0015B/0015D/0015E/
0015F, Tensor API, Compile API, Training API, glossary, current DataType/DataTypePromotion/Shape/
ShapeBroadcast/TensorDescriptor/Tensor/TensorFactory/TensorProvenance/Operation/
WhereSelectionKind contracts and tests, and Java 26 Gradle configuration.

Implement task 0015F exactly. Modify Tensor.java and add package-private final
TensorWhereExpressions.java for production. Update TensorTest only for the exact one-method public
API surface and add TensorWhereSelectionTest. Add exactly public static
where(Tensor condition, Tensor ifTrue, Tensor ifFalse), delegating once to the helper.

The helper has exactly one package-private static apply(condition,ifTrue,ifFalse) method. Follow
the task's exact null/BOOL/promotion/two-stage-broadcast/construction order and messages. Promote
branches through DataTypePromotion exactly once; broadcast true/false branches first and condition
with that branch shape second. Create an unresolved descriptor with promoted type and branch-only
requiresGrad OR, exact WHERE/NoOperationAttrs Operation, ordered
[condition, ifTrue, ifFalse] provenance, and delegate once to TensorFactory.createDerived with no
label/storage. Every valid call returns a fresh Tensor.

Do not inspect values/storage, choose or evaluate branches, add overloads, insert casts, create a
ternary broadcast plan or resolved layout, define gradient rules, capture a graph, change existing
contracts, or introduce compiler/runtime/backend/ONNX behavior. Stop beyond ten paths or on
architecture uncertainty.

Run every specified focused/aggregate test, Javadoc, javap/reflection/bytecode/import/manual,
documentation/link/whitespace/scope/status check. Then hand the actual diff/evidence to a separate
clean-context documentation agent in the same change. It must inspect source/tests/generated
Javadoc, finalize permitted Javadocs/Tensor API/Compile API/glossary/planning, record related-
contract/capability/Training API/architecture no-change conclusions, and rerun validation.

Update task 0015F, model master plan, and roadmap only for planning status/evidence. Do not mark
0015F Complete until both passes succeed. Leave 0015G Draft without a specification. Do not commit
or push.
```

## Local decisions

- The public capability remains the legacy-compatible static
  `Tensor.where(condition, ifTrue, ifFalse)` form. An instance receiver would privilege one branch
  or the condition and obscure the three explicit ordered roles.
- One package-private helper method is sufficient because the family has exactly one semantic kind
  and one arity. A kind parameter or shared ternary framework would add abstraction without a
  second current use.<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- Branch promotion uses the existing shared contract exactly once. Its left/right diagnostic
  vocabulary maps deterministically to true/false branches and is not changed in this task.
- Branch shapes are broadcast first; the condition is combined with their common shape second.
  This exposes branch incompatibility before condition incompatibility without creating a ternary
  plan or changing the accepted local shape set.<span data-glossary-navigation="true"> ([shapes glossary](../../../../glossary.md#shape))</span>
- Result gradient eligibility is the OR of branch requests only. The BOOL condition remains
  non-differentiable, and no gradient routing rule is created.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [BOOL glossary](../../../../glossary.md#data-type--datatype); [differentiable glossary](../../../../glossary.md#derivative-and-chain-rule))</span>
- Result layout is unresolved even for resolved inputs because semantic expression construction
  does not assert physical geometry or materialization.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [materialization glossary](../../../../glossary.md#materialization))</span>

## Known limitations

- The expression contains conditional-selection semantics and [provenance](../../../../glossary.md#provenance) but no calculated values.
- Only floating branches are supported, matching the selected legacy capability baseline. BOOL
  and integral branch selection are not generalized here.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Dynamic broadcasting is conservative and accepts only compatibility provable by the current
  pairwise local rule.<span data-glossary-navigation="true"> ([broadcasting glossary](../../../../glossary.md#broadcasting))</span>
- [Gradient](../../../../glossary.md#term-gradient) eligibility is [metadata](../../../../glossary.md#term-metadata) only; [autograd](../../../../glossary.md#autograd) expansion and branch routing remain unimplemented.
- No compiler capture, simplification, ONNX mapping, backend support, storage interpretation, or
  execution is implied.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [ONNX glossary](../../../../glossary.md#term-onnx); [backend glossary](../../../../glossary.md#backend))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) reviewed the [architecture contract](../../../../glossary.md#architecture-contract) and focused [module](../../../../glossary.md#term-module)/[dependency](../../../../glossary.md#term-dependency) explanations;
documentation and planning rules; roadmap; [model](../../../../glossary.md#term-model) capabilities and master plan; tasks 0001, 0002,
0006, 0007, 0011, 0012, 0013, 0015B, 0015C, 0015D, and 0015E; current DataTypePromotion,
ShapeBroadcast, TensorDescriptor, [Tensor](../../../../glossary.md#tensor), TensorFactory, TensorProvenance, [Operation](../../../../glossary.md#operation),
WhereSelectionKind, and expression-helper source/tests; Tensor/[Compile](../../../../glossary.md#compile)/Training APIs and glossary;
and Java 26 [Gradle](../../../../glossary.md#term-gradle) configuration.

The read-only `legacy/pre-rewrite` branch was inspected directly. It confirms static public
`Tensor.where(condition, ifTrue, ifFalse)`, exact ordered roles, [BOOL](../../../../glossary.md#data-type--datatype)-only condition, all floating
branch pair promotions, condition/branch [broadcasting](../../../../glossary.md#broadcasting), branch [gradient](../../../../glossary.md#term-gradient) eligibility and later
routing, comparison-[mask](../../../../glossary.md#term-boolean-mask) chaining, non-[contiguous](../../../../glossary.md#term-contiguous) inputs, [ONNX](../../../../glossary.md#term-onnx) mapping, [fusion](../../../../glossary.md#term-fusion), and [backend](../../../../glossary.md#backend)
execution evidence. [Mutable](../../../../glossary.md#term-mutable) legacy ternary broadcast plans, [runtime](../../../../glossary.md#term-runtime) [graph](../../../../glossary.md#graph) coupling, Tensor
backward callbacks, storage access, operation traits, [kernels](../../../../glossary.md#kernel), [lowering](../../../../glossary.md#lowering), and execution are
excluded.

Planning selected one public static method and one package-private single-entry helper. Existing
promotion, pairwise [shape](../../../../glossary.md#shape) algebra, [descriptor](../../../../glossary.md#term-descriptor), [provenance](../../../../glossary.md#provenance), and factory contracts are sufficient;
no package, dependency, foundation contract, or architecture rule changes.

Planning validation:

- `git diff --check` passed, and targeted whitespace inspection found no trailing whitespace in
  the three changed planning paths.
- The required-section scan found every canonical task-specification section, including package
  impact, exact helper and public API shapes, bounded scope, validation, implementation handoff,
  decisions, limitations, and completion-evidence sections.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [canonical glossary](../../../../glossary.md#term-canonical-form); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [scope glossary](../../../../glossary.md#term-scope); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- Every local Markdown file and heading anchor linked from this task, the model master plan, and
  the roadmap resolves. Markdown fence counts are balanced.
- Status inspection found 0015F [`Ready`](../../../../glossary.md#term-ready) in this specification, its linked model-master row, and
  its linked roadmap row/current-frontier text. Task 0015G remains `Draft` in both queues.
- Package inspection found no new package. The planned direction remains from `model.tensor` to
  existing operation, selection, datatype, and shape contracts without a reverse dependency.
- Scope inspection found exactly this new task, the model master plan, and the roadmap changed. No
  Java, test, API, glossary, Gradle, architecture, AGENTS, or other-module path changed.<span data-glossary-navigation="true"> ([module path glossary](../../../../glossary.md#term-module-path))</span>
- No task-0015G specification exists.

Implementation and independent documentation validation:

- Clean implementation work changed exactly `Tensor.java`, `TensorWhereExpressions.java`,
  `TensorTest.java`, and `TensorWhereSelectionTest.java`. The public static method delegates once;
  the package-private final helper has one private constructor, one package-private static entry,
  zero fields, and zero nested types.
- Bytecode confirms ordered condition/true/false null checks, exact BOOL validation, one floating
  promotion call, branch-first and condition-second broadcasts, branch-only gradient OR, one
  unresolved descriptor, one exact <span data-glossary-navigation="true"> ([Bytecode glossary](../../../../glossary.md#term-bytecode))</span>`WHERE` operation, one ordered provenance value, and one
  central derived-construction call. No value or storage access appears.
- Independent documentation context
  `/root/implement_model_0015d/review_model_0015d_docs` reread the architecture contract,
  focused architecture pages, documentation and planning rules, model plans and related tasks,
  final source/tests/diff, foundational and expression contracts, generated reports/Javadoc,
  bytecode, APIs, glossary, and Java 26 build configuration. It applied General plus API/Javadoc
  style to Java and API review, Planning style to planning files, and Example style to the new
  complete conditional-selection example.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- The independent pass found the Tensor and helper Javadocs complete. They document exact roles,
  non-nullness, validation order, floating promotion, ordered broadcasts, result facts, branch-only
  eligibility, identity side effects, provenance ownership, failures, non-mutation, and deferred
  value selection/gradient/compiler/backend concerns. No Java or test edit was needed.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Tensor API now documents the current static method, exact result/provenance contract, complete
  metadata-only example, failures, scalar-index-selection distinction, and planned execution
  limits. Compile API includes conditional selection in the current capturable expression input
  while keeping traversal, capture, inference, optimization, artifacts, and execution planned.
  Glossary status, OperationKind, provenance, Tensor, and common distinctions now agree.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [scalar glossary](../../../../glossary.md#term-scalar); [capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [artifacts glossary](../../../../glossary.md#term-artifact); [OperationKind glossary](../../../../glossary.md#operationkind))</span>
- Training API remains accurate unchanged because no gradient routing/rule, autograd, optimizer,
  publication, or training execution was added. Capabilities remain accurate because they already
  select broadcast-aware <span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer); [publication glossary](../../../../glossary.md#term-publication))</span>`where` and distinguish model/public expression support from compiler and
  backend completion.
- Existing [DataType](../../../../glossary.md#data-type--datatype)/promotion, Shape/broadcast, descriptor, TensorFactory/provenance, Operation,
  `WhereSelectionKind`, and prior expression contracts remain accurate because they are composed
  without changed behavior. Architecture/focused docs/ADRs/tests, backend-conformance and
  integration tests, and Java 26 Gradle/build remain unchanged because no dependency, module
  boundary, lifecycle, backend, numerical-execution, preview/incubator, or end-to-end rule changed.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- The first focused implementation run executed 11 tests with one test-only failure: the input-
  preservation test compared three independently constructed, structurally equal layout values to
  one unrelated layout reference. The test was corrected to snapshot each input descriptor's exact
  retained layout reference before calling <span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [layout glossary](../../../../glossary.md#layout); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>`where`; no production code changed for this failure.
- Fresh `--rerun-tasks` focused runs passed: `TensorWhereSelectionTest` contains 11 tests and
  `TensorTest` contains 14 tests, both with zero failures, errors, or skips.
- A fresh `--rerun-tasks` aggregate model run passed. Its 46 XML suites contain 347 tests with zero
  failures, errors, or skips. A fresh model Javadoc run and root `./gradlew test` also passed; the
  root run reported 36 actionable tasks with none failing.
- Final `javap -p -c -s`, reflection tests, import/dependency scans, and generated-Javadoc checks
  confirm the exact public/helper surface and construction order with no synchronization, value or
  storage access, cast insertion, ternary plan, resolved layout, gradient rule, graph/compiler/
  runtime/backend type, registry, route, service, dependency, or build change.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization); [registry glossary](../../../../glossary.md#term-registry))</span>
- The local Markdown validator resolved all 222 file targets and heading anchors in the six changed
  documentation/planning files. Fence counts are balanced, terminology agrees, trailing-whitespace
  scans found no matches, and <span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>`git diff --check` passes.
- Final scope contains exactly the authorized ten paths: two production files, two tests, Tensor
  API, Compile API, glossary, this task, model master plan, and roadmap. Status is synchronized as
  Complete. Task 0015G remains Draft, and no task-0015G specification exists.

## Implementation notes

- Added exactly one public static `Tensor.where` method and one package-private stateless helper.
- Implemented exact BOOL condition validation, floating branch promotion, ordered two-stage local
  broadcasting, branch-only eligibility, unresolved result construction, and three-input
  provenance without inspecting values or storage.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [validation glossary](../../../../glossary.md#term-validation); [broadcasting glossary](../../../../glossary.md#broadcasting); [provenance glossary](../../../../glossary.md#provenance))</span>
- Updated only the exact public/helper surface tests and added the focused where-expression suite.
- The independent documentation pass changed only the permitted Tensor API, Compile API, glossary,
  and planning files; existing Javadocs required no correction.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [planning glossary](../../../../glossary.md#term-planning))</span>

## Completion summary

- Completed changes: Implemented and documented public storage-free conditional-selection Tensor
  expression construction over the existing <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>`WHERE` semantic [identity](../../../../glossary.md#term-identity-meanings).
- Files changed or created: `Tensor.java`, `TensorWhereExpressions.java`, `TensorTest.java`,
  `TensorWhereSelectionTest.java`, Tensor API, Compile API, glossary, this task, model master plan,
  and roadmap.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: Focused where and Tensor suites, all model tests, generated model Javadoc,
  root tests, bytecode/reflection/import/dependency checks, documentation checks, exact scope/status
  checks, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [dependency glossary](../../../../glossary.md#term-dependency); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` passed.
- Documentation review: The required independent clean-context pass completed in the same overall
  change. Existing Javadocs were complete; API and glossary documentation now distinguish current
  expression construction from planned value selection, gradients, compiler capture, ONNX/backend
  support, and execution.<span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [ONNX glossary](../../../../glossary.md#term-onnx); [backend glossary](../../../../glossary.md#backend))</span>
- Documentation impact: Training API, capabilities, architecture and focused explanations, ADRs,
  architecture tests, conformance/integration material, and build documentation remain accurate
  without modification.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0015F. Plan task 0015G separately before implementation.

Status: Complete

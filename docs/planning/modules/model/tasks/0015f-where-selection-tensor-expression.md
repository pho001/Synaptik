# Task 0015F: Where Selection Tensor Expression

## Status

Complete

## Goal

Expose the implemented `WHERE` semantic identity through one public, backend-independent
`Tensor.where(condition, ifTrue, ifFalse)` expression method. The condition must be BOOL, the two
branches must be floating tensors, all three shapes must be locally broadcast-compatible, and the
result must use the promoted branch type. Every successful call returns a fresh storage-free
Tensor whose provenance preserves the exact ordered condition, true branch, and false branch.

This task makes conditional selection capturable through the public model API. It does not read
condition or branch values, choose a branch eagerly, allocate result storage, create gradient
rules, capture a compiled graph, or define backend support.

## Scope

- Add exactly one public static `Tensor.where(Tensor, Tensor, Tensor)` method.
- Add one package-private final `TensorWhereExpressions` helper with one package-private static
  `apply` entry and no other behavior.
- Require an exact `DataType.BOOL` condition without numeric truthiness or implicit conversion.
- Validate and promote the two floating branch data types through exactly one
  `DataTypePromotion.promoteFloating` call.
- Derive the three-way result shape through exactly two ordered `ShapeBroadcast.broadcast` calls:
  first true branch with false branch, then condition with the resulting branch shape.
- Create an unresolved-layout result descriptor with the promoted branch type and gradient
  eligibility equal to `ifTrue.requiresGrad || ifFalse.requiresGrad`.
- Construct exactly one `Operation` from `WhereSelectionKind.WHERE` and
  `NoOperationAttrs.INSTANCE`.
- Construct exactly one `TensorProvenance` with ordered exact inputs
  `[condition, ifTrue, ifFalse]`.
- Delegate final identity-bearing construction exactly once to `TensorFactory.createDerived` with
  no label and no storage.
- Update the exact Tensor public-API reflection test and add one focused where-expression test.
- Finalize affected Javadocs, Tensor API, Compile API status, glossary, task evidence, model master
  plan, and roadmap through the required independent documentation pass during implementation.

## Out of scope

- eager value selection, condition or branch storage reads, branch evaluation order, lazy control
  flow, short-circuiting, constant folding, canonicalization, interning, or returning an input
- BOOL, INT32, or INT64 branches; numeric conditions; implicit conversion; explicit cast
  insertion; nullable conditions; or three-valued logic
- a generic conditional operator, instance `where` method, overload, scalar branch, caller label,
  variadic API, operator alias, expression-builder public type, or factory method
- resolved output layout, a ternary broadcast-plan type, effective strides, aliases, view
  preservation, materialization policy, or input-layout propagation
- gradient values, branch gradient routing, backward operations, autograd expansion, optimizer,
  training-root policy, or training execution
- changes to `WhereSelectionKind`, `Operation`, `OperationKind`, `OperationAttrs`,
  `NoOperationAttrs`, `DataTypePromotion`, `ShapeBroadcast`, `TensorDescriptor`,
  `TensorProvenance`, or `TensorFactory`
- scalar-index `select`, gather, take, scatter, mask indexing, reduction, cast, or another operation
  family
- graph traversal, cycle checks, node/value IDs, graph capture, compiled graph construction,
  common-subexpression elimination, compiler inference, publication, or optimization
- planning ownership, capability providers, backend support, fusion, cost, lowering, kernels,
  runtime residency, prepare, execution, tracing, ONNX mapping, or engine behavior
- dependencies, Gradle changes, architecture changes, another module, unrelated refactors, or a
  detailed task-0015G specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
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
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes static
`Tensor.where(condition, ifTrue, ifFalse)`. Its public builder requires a BOOL condition, accepts
all pairings of BFLOAT16, FLOAT32, and FLOAT64 branches, promotes branch precision, and broadcasts
the condition and both branches to one result shape. Legacy tests cover branch broadcasting,
condition broadcasting, mixed floating precision, comparison masks, non-contiguous inputs,
expression chaining, gradient routing, ONNX mapping, fusion, and several backend routes.

Legacy code also creates mutable ternary broadcast plans, installs backward callbacks on Tensor,
reads storage during execution, and exposes runtime/backend traits. Those mechanisms are not
copied. This task retains only local public expression construction through immutable descriptor
and provenance values. Compiler autograd owns gradient expansion; planning and backend prepare own
materialization and executable choices; concrete backends own condition interpretation and value
selection.

## Architecture constraints

- `Tensor` remains public mutable API state and must not become an IR node.
- `TensorWhereExpressions` performs deterministic local model validation only. It must not inspect
  values or storage, traverse provenance, capture a graph, infer backend support, or execute
  conditional selection.
- `Operation` owns only the exact backend-independent `WHERE` kind and canonical no-attributes
  value. It contains no inputs, arity field, result descriptor, broadcast plan, or executable
  behavior.
- `TensorProvenance` owns exact ordered input identities. Order is always
  `[condition, ifTrue, ifFalse]`; no branch or condition reordering is allowed.
- Result identity comes only from the existing package-private `TensorFactory.createDerived` seam.
  No second allocator, caller-supplied ID, registry, cache, or service is introduced.
- Every result is storage-free, unlabeled, unresolved-layout model state. No physical buffer,
  alias, view, or input storage is attached.
- The result data type is the exact output of the existing floating promotion contract applied to
  `ifTrue` and `ifFalse`. The BOOL condition never participates in promotion.
- Result gradient eligibility is the logical OR of the two branch descriptor requests. The BOOL
  condition does not contribute, and eligibility does not imply that a gradient rule exists.
- Three-way dynamic shapes are accepted only where two existing pairwise `ShapeBroadcast` calls
  prove compatibility locally. The helper creates no symbolic constraints or ternary shape type.
- The output layout remains unresolved even when every input layout is resolved or identical.
- Package direction is `model.tensor -> model.operation.elementwise.selection`, plus existing
  `model.tensor -> model.operation`, `model.datatype`, and `model.shape`. The selection package
  must not import Tensor, and no package cycle may be introduced.
- Stop if implementation requires a changed foundational contract, new broadcast-plan or
  attributes type, storage access, graph capture, gradient rule, dependency, or architecture
  decision.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns the public static where surface, local validation,
  result construction, provenance, and package-private derived Tensor factory seam.
- `io.github.pho001.synaptik.model.operation` — supplies `Operation` and
  `NoOperationAttrs.INSTANCE`.
- `io.github.pho001.synaptik.model.operation.elementwise.selection` — supplies the exact
  `WhereSelectionKind.WHERE` semantic value.
- `io.github.pho001.synaptik.model.datatype` — supplies BOOL identity and floating branch
  promotion.
- `io.github.pho001.synaptik.model.shape` — supplies immutable shapes and pairwise local
  right-aligned broadcasting.

No package is added.

Type placement:

- `io.github.pho001.synaptik.model.tensor.Tensor` — public static conditional-selection surface;
  it receives only the exact `where` method and delegates all shared behavior.
- `io.github.pho001.synaptik.model.tensor.TensorWhereExpressions` — package-private, stateless
  validation and construction boundary colocated with Tensor, descriptors, provenance, and the
  package-private factory seam it must use.
- `TensorWhereSelectionTest` — same-package focused test so it can verify the package-private
  helper without widening production visibility.

## Required contract

### Public Tensor surface

Add exactly this public static method to `Tensor`:

```java
public static Tensor where(Tensor condition, Tensor ifTrue, Tensor ifFalse)
```

It delegates exactly once and returns the exact result:

```java
return TensorWhereExpressions.apply(condition, ifTrue, ifFalse);
```

The public method performs no separate null check, type validation, promotion, broadcasting,
descriptor creation, allocation, provenance construction, canonicalization, or storage access.
There is no instance form, overload, alias, or generic conditional API in this task.

### Package-private helper shape

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
or additional methods. Its private zero-argument constructor prevents instantiation. A kind
parameter is unnecessary because the owning semantic family contains exactly `WHERE`.

### Validation and construction order

`apply` performs these steps in exact order:

1. require non-null `condition`, `ifTrue`, and `ifFalse`, in that order, with messages
   `condition`, `ifTrue`, and `ifFalse`;
2. require `condition.descriptor().dataType() == DataType.BOOL`; otherwise throw
   `IllegalArgumentException` with message
   `condition must have BOOL data type, but was <actual>`;
3. call `DataTypePromotion.promoteFloating(ifTrue.descriptor().dataType(),
   ifFalse.descriptor().dataType())` exactly once and retain its exact result data type;
4. call `ShapeBroadcast.broadcast(ifTrue.descriptor().shape(),
   ifFalse.descriptor().shape())` exactly once to derive the common branch shape;
5. call `ShapeBroadcast.broadcast(condition.descriptor().shape(), branchShape)` exactly once to
   derive the final three-way result shape;
6. create exactly one `TensorDescriptor` from the promoted branch data type, final result shape,
   `Optional.empty()` layout, and
   `ifTrue.descriptor().requiresGrad() || ifFalse.descriptor().requiresGrad()`;
7. create exactly one
   `Operation(WhereSelectionKind.WHERE, NoOperationAttrs.INSTANCE)`;
8. create exactly one `TensorProvenance(operation, List.of(condition, ifTrue, ifFalse))`;
9. call `TensorFactory.createDerived(descriptor, Optional.empty(), provenance)` exactly once and
   return its exact result.

Do not catch, translate, aggregate, or replace failures from `DataTypePromotion`,
`ShapeBroadcast`, descriptor construction, provenance, or the factory.

### Branch promotion failures

The existing shared promotion contract receives `ifTrue` as its left operand and `ifFalse` as its
right operand. Therefore:

- an invalid true branch fails with
  `IllegalArgumentException("left must be a floating data type, but was <actual>")`; and
- after a valid true branch, an invalid false branch fails with
  `IllegalArgumentException("right must be a floating data type, but was <actual>")`.

This task does not duplicate floating eligibility checks or change `DataTypePromotion` merely to
rename its diagnostic operands. Public Javadoc explains the branch restriction without promising
a new promotion-message vocabulary.

### Three-way broadcasting

Three-way broadcasting is composition of the current pairwise local rule, not a new model type:

```text
branchShape = broadcast(ifTrue.shape, ifFalse.shape)
resultShape = broadcast(condition.shape, branchShape)
```

The first call proves branch compatibility before the condition shape is combined. The second
call proves that the condition can address the common branch result. Equal symbolic dimensions and
static singleton expansion succeed according to `ShapeBroadcast`; different symbols and symbolic
versus non-singleton static sizes remain locally unprovable and fail. Pairwise order affects only
deterministic validation and diagnostics, not provenance order.

No `WhereBroadcastPlan`, effective stride, resolved layout, or symbolic constraint is stored.

### Failure and identity side effects

Null, condition-type, promotion, and both shape validations complete before ID allocation:

- `Tensor.where(null, null, null)` fails with `NullPointerException("condition")`;
- after a valid condition reference, null branches fail in true-then-false order;
- a non-BOOL condition fails before branch data types or any shape is inspected;
- branch promotion fails before either broadcast call;
- incompatible branches fail before the condition is broadcast;
- a compatible branch shape with an incompatible condition fails in the second broadcast call;
  and
- exhausted Tensor identity space fails through `TensorFactory.createDerived` only after all local
  model values have been constructed.

Failures before `createDerived` consume no Tensor ID. Do not add a production ID-inspection hook,
catch an exception, or roll back an allocated identity.

### Result descriptor, provenance, and identity

Every successful result has:

- the exact promoted floating branch data type;
- the final locally proven three-way broadcast shape;
- empty layout for static and dynamic shapes;
- gradient eligibility equal only to the OR of true- and false-branch eligibility;
- a fresh factory-assigned `TensorId`;
- an empty label and no host storage;
- exact `WhereSelectionKind.WHERE` with `NoOperationAttrs.INSTANCE`; and
- exact immutable provenance inputs `[condition, ifTrue, ifFalse]`.

Using the same Tensor as both branches is valid and records that reference twice. Repeating a valid
call creates another identity. A comparison or logical expression may serve directly as the BOOL
condition. Input labels, provenance, layouts, storage associations, and storage contents remain
unchanged and are not copied to the result.

### No eager selection or gradients

The helper is eager only for expression metadata. It never reads a condition byte, chooses or
evaluates a branch, copies or converts values, simplifies equal branches, reorders inputs, or
collapses nested selections. It propagates branch gradient eligibility as descriptor metadata but
does not define gradient routing, create backward operations, or mutate any Tensor gradient state.
Compiler and training work later own those responsibilities.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorWhereExpressions.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorWhereSelectionTest.java`

Documentation/planning during implementation:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a task-related inconsistency requires stopping:

- `docs/api/training-api.md`
- `docs/planning/modules/model/capabilities.md`
- Existing `DataType`, `DataTypePromotion`, `Shape`, `ShapeBroadcast`, `TensorDescriptor`,
  `TensorProvenance`, `TensorFactory`, `Operation`, `NoOperationAttrs`, and
  `WhereSelectionKind` Javadocs/tests.
- Existing arithmetic, comparison, logical, unary, and scalar expression contracts; focused
  architecture documents; ADRs; architecture tests; backend-conformance tests; and integration
  tests.

## Maximum scope

At most two production files, two test files, and six documentation/planning files: ten paths
total.

`Tensor.java` and `TensorTest.java` may change only for the one exact public static `where` method,
its import/Javadocs, exact API-shape expectations, and non-synchronization assertions. Do not
change existing fields, constructor, metadata/storage behavior, expression methods, equality,
hashing, diagnostics, or unrelated tests.

If implementation needs another production/test concept, a changed foundational contract, local
branch-specific promotion API, ternary broadcast-plan type, resolved-layout policy, storage
access, graph/compiler behavior, gradient rule, another documentation file, or more than ten
paths, stop and propose a follow-up or architecture decision. Do not create task 0015G.

## Javadoc requirements

- Update Tensor type Javadoc only as needed to include conditional selection while preserving the
  distinction between public Tensor state, provenance, graph IR, gradient eligibility, and
  executable values.
- `Tensor.where` must document the ordered condition/true-branch/false-branch roles, exact BOOL
  condition, floating promotion, ordered two-stage broadcasting, promoted result data type,
  unresolved layout, branch-only gradient eligibility OR, fresh identity, storage absence,
  provenance, and deferred value selection and gradient rules.
- Document all three non-null inputs with `@param`, including exact roles, reference retention in
  provenance, and absence of mutation. Document the fresh result with `@return` and null,
  condition-type, promotion, shape, and identity-exhaustion failures with `@throws`.
- Document the package-private helper, private constructor, and `apply` method with exact
  validation/construction order, input ownership, ID side effects, and failure behavior.
- Explain that expression metadata construction is eager but condition/branch value evaluation is
  not performed or specified by this model API.
- Explain why conditional `where` is distinct from scalar-index `select` and why no broadcast plan
  or resolved output layout is created.
- Review related foundational and expression Javadocs and record why they remain accurate or stop
  on an out-of-scope inconsistency.

## Acceptance criteria

- Tensor declares exactly one new `public static Tensor where(Tensor, Tensor, Tensor)` method; no
  instance form, overload, alias, or unrelated public API is added.
- The public method delegates once to `TensorWhereExpressions.apply` and performs no other work.
- `TensorWhereExpressions` has exactly the specified visibility, finality, private constructor,
  one package-private static method, zero fields, and zero nested types.
- Null checks, condition validation, promotion, two broadcasts, descriptor/operation/provenance
  construction, and factory delegation occur in the exact specified order.
- The condition must be exact BOOL. All nine floating branch pairs succeed and produce the exact
  shared promoted type. BOOL, INT32, and INT64 fail in either branch position without conversion,
  storage access, or ID allocation.
- Scalar, zero-sized, rank-mismatched, singleton-expanded, multi-axis, equal-symbolic, and
  symbolic/singleton three-way broadcasts succeed. Incompatible branches, condition/result
  shapes, different symbols, and symbolic/non-singleton pairs fail locally.
- `ShapeBroadcast.broadcast` is invoked exactly twice per valid construction and in branch-first,
  condition-second order. No ternary plan, stride, layout, or constraint object is introduced.
- Every result has promoted floating data type, empty layout, exact final shape, branch-only
  gradient eligibility OR, empty label, no host storage, fresh identity, exact operation and
  attributes, and exact ordered three-input provenance.
- Comparison and logical BOOL expressions can feed the condition without special handling.
- Same-branch use and repeated calls remain fresh and ordered; equal branches are not simplified.
- No input Tensor metadata, provenance, label, storage association, or storage contents are
  mutated or retained as output storage.
- No value selection, evaluation-order policy, gradient rule, graph state, backend fact,
  dependency, build option, or architecture change is added.
- Focused and aggregate model tests, model Javadoc, root tests, reflection/javap/import/bytecode/
  scope checks, documentation links/formatting, and status synchronization pass.
- A separate documentation-focused agent finalizes Javadocs, Tensor API, Compile API, glossary,
  task evidence, master plan, and roadmap in the same change and records reasoned no-change
  conclusions for Training API, capabilities, architecture, and related contracts.
- Task 0015F becomes Complete only after both passes. Task 0015G remains Draft without a detailed
  specification.

## Tests / validation

Run before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorWhereSelectionTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test must cover:

- exact helper class/constructor/method/field visibility and signature;
- exact public static `where` signature, visibility, non-synchronization, and one delegation;
- exact Operation kind and singleton attributes references;
- exact ordered three-input provenance, including repeated branch references and derived BOOL
  conditions;
- every floating branch pairing and exact promoted result type;
- exact BOOL condition eligibility and rejection of every non-BOOL type;
- scalar, zero-sized, different-rank, singleton, multi-axis, and dynamic three-way broadcasting;
- all four branch `requiresGrad` combinations and exclusion of the condition from propagation;
- empty layout, empty label, absent storage, fresh identity, and no canonicalization;
- helper/public nulls, invalid condition, invalid true/false branch, incompatible branch shapes,
  incompatible condition/result shapes, and unprovable dynamic shapes; and
- preservation of input descriptors, provenance, labels, layouts, storage associations, and
  contents.

Manually inspect `javap -p -c -s`, method bytecode, reflection, and imports for the exact Tensor
method descriptor, helper shape, sole delegation, null/type/promotion/broadcast order, exactly one
promotion call, exactly two broadcast calls in the specified order, branch-only gradient OR,
fixed `WHERE` operation, and absence of synchronization on new public/helper entries. Confirm no
numeric access, storage operation, cast, resolved layout, gradient rule, graph/compiler/runtime/
backend type, cost, fusion, route, registry, service, dependency, or build change appears. Validate
generated Javadoc, Tensor/Compile API status, glossary, links/anchors/fences/whitespace, exact
ten-path scope, synchronized statuses, and absence of a task-0015G specification.

## Dependencies

- Task 0001 supplies exact BOOL identity and floating `DataTypePromotion`.
- Task 0002 supplies `Shape` and pairwise `ShapeBroadcast` local right-aligned broadcasting.
- Task 0006 supplies immutable generic `Operation` composition.
- Task 0007 supplies `TensorDescriptor` and its differentiability validation.
- Task 0011 supplies public Tensor state and the exact API surface to extend.
- Task 0012 supplies centralized Tensor identity allocation through `TensorFactory`.
- Task 0013 supplies immutable ordered provenance and `TensorFactory.createDerived`.
- Task 0015B supplies current comparison expressions that produce BOOL conditions.
- Task 0015D supplies current logical expressions that can also produce BOOL conditions.
- Task 0015E supplies exact parameterless `WhereSelectionKind.WHERE` semantics.

## Follow-up tasks

- 0015G remains Draft for typed cast semantic kind and attributes.
- 0015H remains Draft for explicit public cast expression construction.
- Compiler tasks later own provenance traversal, graph capture, optimization, autograd expansion,
  and training-graph treatment.
- Backend, ONNX, and conformance tasks later own mapping, BOOL condition storage interpretation,
  value selection, lowering, kernels, and execution.

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None. The architecture already assigns public Tensor state, backend-independent
Operation semantics, descriptors, and minimal provenance to `modules/model`. The helper composes
those existing contracts without adding compiler, runtime, backend, device, or executable state.

If implementation requires graph capture, storage interpretation, a gradient rule, backend
metadata, another dependency, or architecture change, stop and report it.

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
  second current use.
- Branch promotion uses the existing shared contract exactly once. Its left/right diagnostic
  vocabulary maps deterministically to true/false branches and is not changed in this task.
- Branch shapes are broadcast first; the condition is combined with their common shape second.
  This exposes branch incompatibility before condition incompatibility without creating a ternary
  plan or changing the accepted local shape set.
- Result gradient eligibility is the OR of branch requests only. The BOOL condition remains
  non-differentiable, and no gradient routing rule is created.
- Result layout is unresolved even for resolved inputs because semantic expression construction
  does not assert physical geometry or materialization.

## Known limitations

- The expression contains conditional-selection semantics and provenance but no calculated values.
- Only floating branches are supported, matching the selected legacy capability baseline. BOOL
  and integral branch selection are not generalized here.
- Dynamic broadcasting is conservative and accepts only compatibility provable by the current
  pairwise local rule.
- Gradient eligibility is metadata only; autograd expansion and branch routing remain unimplemented.
- No compiler capture, simplification, ONNX mapping, backend support, storage interpretation, or
  execution is implied.

## Validation evidence

Planning reviewed the architecture contract and focused module/dependency explanations;
documentation and planning rules; roadmap; model capabilities and master plan; tasks 0001, 0002,
0006, 0007, 0011, 0012, 0013, 0015B, 0015C, 0015D, and 0015E; current DataTypePromotion,
ShapeBroadcast, TensorDescriptor, Tensor, TensorFactory, TensorProvenance, Operation,
WhereSelectionKind, and expression-helper source/tests; Tensor/Compile/Training APIs and glossary;
and Java 26 Gradle configuration.

The read-only `legacy/pre-rewrite` branch was inspected directly. It confirms static public
`Tensor.where(condition, ifTrue, ifFalse)`, exact ordered roles, BOOL-only condition, all floating
branch pair promotions, condition/branch broadcasting, branch gradient eligibility and later
routing, comparison-mask chaining, non-contiguous inputs, ONNX mapping, fusion, and backend
execution evidence. Mutable legacy ternary broadcast plans, runtime graph coupling, Tensor
backward callbacks, storage access, operation traits, kernels, lowering, and execution are
excluded.

Planning selected one public static method and one package-private single-entry helper. Existing
promotion, pairwise shape algebra, descriptor, provenance, and factory contracts are sufficient;
no package, dependency, foundation contract, or architecture rule changes.

Planning validation:

- `git diff --check` passed, and targeted whitespace inspection found no trailing whitespace in
  the three changed planning paths.
- The required-section scan found every canonical task-specification section, including package
  impact, exact helper and public API shapes, bounded scope, validation, implementation handoff,
  decisions, limitations, and completion-evidence sections.
- Every local Markdown file and heading anchor linked from this task, the model master plan, and
  the roadmap resolves. Markdown fence counts are balanced.
- Status inspection found 0015F `Ready` in this specification, its linked model-master row, and
  its linked roadmap row/current-frontier text. Task 0015G remains `Draft` in both queues.
- Package inspection found no new package. The planned direction remains from `model.tensor` to
  existing operation, selection, datatype, and shape contracts without a reverse dependency.
- Scope inspection found exactly this new task, the model master plan, and the roadmap changed. No
  Java, test, API, glossary, Gradle, architecture, AGENTS, or other-module path changed.
- No task-0015G specification exists.

Implementation and independent documentation validation:

- Clean implementation work changed exactly `Tensor.java`, `TensorWhereExpressions.java`,
  `TensorTest.java`, and `TensorWhereSelectionTest.java`. The public static method delegates once;
  the package-private final helper has one private constructor, one package-private static entry,
  zero fields, and zero nested types.
- Bytecode confirms ordered condition/true/false null checks, exact BOOL validation, one floating
  promotion call, branch-first and condition-second broadcasts, branch-only gradient OR, one
  unresolved descriptor, one exact `WHERE` operation, one ordered provenance value, and one
  central derived-construction call. No value or storage access appears.
- Independent documentation context
  `/root/implement_model_0015d/review_model_0015d_docs` reread the architecture contract,
  focused architecture pages, documentation and planning rules, model plans and related tasks,
  final source/tests/diff, foundational and expression contracts, generated reports/Javadoc,
  bytecode, APIs, glossary, and Java 26 build configuration. It applied General plus API/Javadoc
  style to Java and API review, Planning style to planning files, and Example style to the new
  complete conditional-selection example.
- The independent pass found the Tensor and helper Javadocs complete. They document exact roles,
  non-nullness, validation order, floating promotion, ordered broadcasts, result facts, branch-only
  eligibility, identity side effects, provenance ownership, failures, non-mutation, and deferred
  value selection/gradient/compiler/backend concerns. No Java or test edit was needed.
- Tensor API now documents the current static method, exact result/provenance contract, complete
  metadata-only example, failures, scalar-index-selection distinction, and planned execution
  limits. Compile API includes conditional selection in the current capturable expression input
  while keeping traversal, capture, inference, optimization, artifacts, and execution planned.
  Glossary status, OperationKind, provenance, Tensor, and common distinctions now agree.
- Training API remains accurate unchanged because no gradient routing/rule, autograd, optimizer,
  publication, or training execution was added. Capabilities remain accurate because they already
  select broadcast-aware `where` and distinguish model/public expression support from compiler and
  backend completion.
- Existing DataType/promotion, Shape/broadcast, descriptor, TensorFactory/provenance, Operation,
  `WhereSelectionKind`, and prior expression contracts remain accurate because they are composed
  without changed behavior. Architecture/focused docs/ADRs/tests, backend-conformance and
  integration tests, and Java 26 Gradle/build remain unchanged because no dependency, module
  boundary, lifecycle, backend, numerical-execution, preview/incubator, or end-to-end rule changed.
- The first focused implementation run executed 11 tests with one test-only failure: the input-
  preservation test compared three independently constructed, structurally equal layout values to
  one unrelated layout reference. The test was corrected to snapshot each input descriptor's exact
  retained layout reference before calling `where`; no production code changed for this failure.
- Fresh `--rerun-tasks` focused runs passed: `TensorWhereSelectionTest` contains 11 tests and
  `TensorTest` contains 14 tests, both with zero failures, errors, or skips.
- A fresh `--rerun-tasks` aggregate model run passed. Its 46 XML suites contain 347 tests with zero
  failures, errors, or skips. A fresh model Javadoc run and root `./gradlew test` also passed; the
  root run reported 36 actionable tasks with none failing.
- Final `javap -p -c -s`, reflection tests, import/dependency scans, and generated-Javadoc checks
  confirm the exact public/helper surface and construction order with no synchronization, value or
  storage access, cast insertion, ternary plan, resolved layout, gradient rule, graph/compiler/
  runtime/backend type, registry, route, service, dependency, or build change.
- The local Markdown validator resolved all 222 file targets and heading anchors in the six changed
  documentation/planning files. Fence counts are balanced, terminology agrees, trailing-whitespace
  scans found no matches, and `git diff --check` passes.
- Final scope contains exactly the authorized ten paths: two production files, two tests, Tensor
  API, Compile API, glossary, this task, model master plan, and roadmap. Status is synchronized as
  Complete. Task 0015G remains Draft, and no task-0015G specification exists.

## Implementation notes

- Added exactly one public static `Tensor.where` method and one package-private stateless helper.
- Implemented exact BOOL condition validation, floating branch promotion, ordered two-stage local
  broadcasting, branch-only eligibility, unresolved result construction, and three-input
  provenance without inspecting values or storage.
- Updated only the exact public/helper surface tests and added the focused where-expression suite.
- The independent documentation pass changed only the permitted Tensor API, Compile API, glossary,
  and planning files; existing Javadocs required no correction.

## Completion summary

- Completed changes: Implemented and documented public storage-free conditional-selection Tensor
  expression construction over the existing `WHERE` semantic identity.
- Files changed or created: `Tensor.java`, `TensorWhereExpressions.java`, `TensorTest.java`,
  `TensorWhereSelectionTest.java`, Tensor API, Compile API, glossary, this task, model master plan,
  and roadmap.
- Tests and validation: Focused where and Tensor suites, all model tests, generated model Javadoc,
  root tests, bytecode/reflection/import/dependency checks, documentation checks, exact scope/status
  checks, and `git diff --check` passed.
- Documentation review: The required independent clean-context pass completed in the same overall
  change. Existing Javadocs were complete; API and glossary documentation now distinguish current
  expression construction from planned value selection, gradients, compiler capture, ONNX/backend
  support, and execution.
- Documentation impact: Training API, capabilities, architecture and focused explanations, ADRs,
  architecture tests, conformance/integration material, and build documentation remain accurate
  without modification.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0015F. Plan task 0015G separately before implementation.

Status: Complete

# Task 0015B: Binary Comparison Tensor Expressions

## Status

Complete

## Goal

Expose the six implemented binary comparison semantics as public, backend-independent Tensor
expression methods. Each successful call must validate floating input types, apply the existing
floating compatibility hierarchy, derive a locally provable right-aligned broadcast shape, and
return a fresh storage-free `BOOL` Tensor whose provenance records the exact comparison and ordered
input Tensor identities.

This task makes comparisons capturable through the public model API. It does not compare numeric
values, allocate result storage, capture a compiled graph, define numerical edge behavior, create
gradient rules, or report backend support.

## Scope

- Add public `Tensor` methods `greaterThan`, `greaterOrEqual`, `lessThan`, `lessOrEqual`,
  `equalTo`, and `notEqualTo`.
- Add one package-private `TensorComparisonExpressions` helper that owns the shared local
  validation and derived-Tensor construction path.
- Accept every pair formed from `BFLOAT16`, `FLOAT32`, and `FLOAT64`.
- Validate the pair and its common floating comparison domain through the existing
  `DataTypePromotion.promoteFloating` contract without storing a promoted result type.
- Compute the result shape through the existing `ShapeBroadcast.broadcast` contract.
- Create an unresolved-layout `BOOL` result descriptor with `requiresGrad=false` regardless of
  either input descriptor's gradient eligibility.
- Construct exactly one `Operation` from the selected `BinaryComparisonKind` and
  `NoOperationAttrs.INSTANCE`.
- Construct exactly one `TensorProvenance` with ordered inputs `[left, right]`.
- Delegate final identity-bearing construction exactly once to `TensorFactory.createDerived` with
  no label and no storage.
- Update the exact Tensor public-API reflection test and add one focused binary-comparison test.
- Finalize affected Javadocs, Tensor API, Compile API status, glossary, task evidence, model master
  plan, and roadmap through the required independent documentation pass during implementation.

## Out of scope

- eager numeric comparison, reading input values, constant folding, canonicalization, interning,
  or returning a cached result
- defining NaN, infinity, signed-zero, tolerance, total-order, or approximate-equality behavior
- integral or boolean comparison inputs, numeric truthiness, implicit cross-category conversion,
  explicit cast insertion, or extension beyond the selected floating baseline
- scalar-number overloads, reverse comparisons, static factory forms, aliases, generic predicates,
  comparator objects, or an expression-builder public type
- output labels, caller-supplied labels, expression strings, symbols, or serialization names
- resolved output layouts, broadcast stride plans, zero-stride views, aliases, materialization
  policy, or layout preservation from either operand
- propagation of input `requiresGrad`, gradient values, comparison gradients, straight-through
  estimators, backward graph construction, training-root rules, or autograd execution
- operation-family attributes, factories, registries, parsers, reflection discovery, or changes to
  `BinaryComparisonKind`, `Operation`, `OperationKind`, `OperationAttrs`, or `NoOperationAttrs`
- graph traversal, cycle checks, node/value IDs, graph capture, compiled graph construction,
  common-subexpression elimination, compiler inference, or publication binding
- planning ownership, capability providers, backend support, fusion, cost, lowering, kernels,
  runtime residency, prepare, execution, tracing, or engine behavior
- dependencies, Gradle changes, architecture changes, another module, another operation family, or
  a detailed task-0015C specification

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
- [Task 0003](0003-layout-descriptor-model.md)
- [Task 0006](0006-operation-model.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0011](0011-public-tensor-skeleton.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0014B](0014b-binary-arithmetic-tensor-expressions.md)
- [Task 0015A](0015a-binary-comparison-semantic-kinds.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes `Tensor.greaterThan(Tensor)`,
`greaterOrEqual(Tensor)`, `lessThan(Tensor)`, `lessOrEqual(Tensor)`, `equalTo(Tensor)`, and
`notEqualTo(Tensor)`. Its public construction path accepts floating inputs, rejects integral and
boolean inputs, applies binary broadcasting, returns a boolean result, disables result gradient
eligibility, and retains left/right operand order. Legacy execution tests cover all six relations,
FLOAT32 and FLOAT64 values, broadcast masks, boolean chaining, and use as a `where` condition.

The legacy implementation also stored mutable broadcast plans inside operation objects, attached
runtime-oriented behavior, defined executable numerical behavior, and coupled comparison results
to graph compilation and backends. Those mechanisms are not copied. The new model retains only the
selected public capability and represents it through immutable descriptors and provenance.
Numerical comparison policy belongs to backend-conformance work, graph capture and inference belong
to the compiler, and execution belongs to prepared backends.

## Architecture constraints

- `Tensor` remains public mutable API state and must not become an IR node.
- `TensorComparisonExpressions` performs deterministic local model validation only. It must not
  traverse provenance, capture a graph, evaluate data, or inspect backend capability.
- `Operation` owns only backend-independent semantic kind and attributes. It contains no operands,
  broadcast result, backend support, numerical policy, or executable behavior.
- `TensorProvenance` owns the exact ordered `[left, right]` input identities. All six comparisons
  preserve caller order; even `EQUAL` and `NOT_EQUAL` must not reorder or canonicalize operands.
- Result identity comes only from the existing package-private `TensorFactory.createDerived` seam.
  No second allocator, caller-supplied ID, registry, cache, or service is introduced.
- The result is storage-free. Public expression construction must not allocate physical buffers,
  inspect input storage, or attach either operand's storage.
- The result descriptor has `DataType.BOOL`, `Optional.empty()` layout, and
  `requiresGrad=false`. This remains true for static shapes and for inputs requesting gradients.
- Dynamic shapes are accepted only where `ShapeBroadcast` can prove compatibility locally: equal
  symbolic dimensions and static singleton expansion. The helper must not create constraints or
  defer a known local incompatibility.
- Input compatibility uses `DataTypePromotion.promoteFloating` exactly once. Its returned common
  floating type validates the semantic comparison domain but is not the output type and is not
  stored in the BOOL result descriptor. Integral and boolean inputs are rejected; no cast is
  inserted.
- A BOOL comparison result is non-differentiable model metadata. This task does not define whether
  or how a later compiler treats comparisons inside a training graph.
- Package direction is `model.tensor -> model.operation.elementwise.comparison`, plus existing
  `model.tensor -> model.operation`, `model.datatype`, and `model.shape`. The operation package must
  not import Tensor and no package cycle may be introduced.
- Stop if implementation requires a changed foundational contract, resolved layout, storage
  access, graph capture, gradient rule, numerical policy, dependency, or architecture decision.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns public Tensor comparison methods, local result
  construction, descriptors, provenance, and the derived Tensor factory seam.
- `io.github.pho001.synaptik.model.operation` — supplies `Operation` and
  `NoOperationAttrs.INSTANCE`.
- `io.github.pho001.synaptik.model.operation.elementwise.comparison` — supplies the six exact
  `BinaryComparisonKind` values.
- `io.github.pho001.synaptik.model.datatype` — supplies floating compatibility validation and the
  fixed `BOOL` result data type.
- `io.github.pho001.synaptik.model.shape` — supplies local right-aligned broadcasting.

No package is added.

Type placement:

- `io.github.pho001.synaptik.model.tensor.Tensor` — public fluent expression surface; it receives
  only the six one-argument methods and delegates shared behavior.
- `io.github.pho001.synaptik.model.tensor.TensorComparisonExpressions` — package-private, stateless
  construction boundary colocated with Tensor, descriptors, provenance, and the package-private
  factory seam it must use.
- `TensorBinaryComparisonTest` — same-package focused test so it can verify the package-private
  helper without widening production visibility.

## Required contract

### Public Tensor surface

Add exactly these public methods to `Tensor`:

```java
public Tensor greaterThan(Tensor right)
public Tensor greaterOrEqual(Tensor right)
public Tensor lessThan(Tensor right)
public Tensor lessOrEqual(Tensor right)
public Tensor equalTo(Tensor right)
public Tensor notEqualTo(Tensor right)
```

Each method delegates exactly once to
`TensorComparisonExpressions.apply(this, right, <KIND>)` and returns that exact result:

| Tensor method | Exact kind |
|---|---|
| `greaterThan` | `GREATER_THAN` |
| `greaterOrEqual` | `GREATER_OR_EQUAL` |
| `lessThan` | `LESS_THAN` |
| `lessOrEqual` | `LESS_OR_EQUAL` |
| `equalTo` | `EQUAL` |
| `notEqualTo` | `NOT_EQUAL` |

The public methods perform no separate validation, promotion, broadcasting, descriptor creation,
allocation, provenance construction, canonicalization, or storage access. There are no overloads
in this task. The receiver is always the ordered left operand and the argument is always the
ordered right operand.

### Package-private helper shape

Create exactly one package-private final non-record class:

```java
final class TensorComparisonExpressions {
    private TensorComparisonExpressions() {
    }

    static Tensor apply(Tensor left, Tensor right, BinaryComparisonKind kind) {
        // exact construction contract below
    }
}
```

The helper has no fields, nested types, public/protected members, overloads, caches, registries, or
operation-specific branches. Its constructor prevents instantiation. `apply` is package-private
and static so Tensor can delegate without exposing an independent public expression service.

### Validation and construction order

`apply` performs these steps in exact order:

1. require non-null `left`, `right`, and `kind`, in that order, with messages `left`, `right`, and
   `kind`;
2. call `DataTypePromotion.promoteFloating(left.descriptor().dataType(),
   right.descriptor().dataType())` exactly once to validate and determine the common comparison
   domain; do not use its returned type as the result type;
3. call `ShapeBroadcast.broadcast(left.descriptor().shape(), right.descriptor().shape())` exactly
   once;
4. create exactly one `TensorDescriptor` from `DataType.BOOL`, the broadcast shape,
   `Optional.empty()` layout, and `false` gradient eligibility;
5. create exactly one `Operation(kind, NoOperationAttrs.INSTANCE)`;
6. create exactly one `TensorProvenance(operation, List.of(left, right))`;
7. call `TensorFactory.createDerived(descriptor, Optional.empty(), provenance)` exactly once and
   return its exact result.

Null and semantic validation complete before ID allocation. Existing delegated exceptions and
messages remain unchanged:

- a public method given null fails with `NullPointerException("right")`;
- a non-floating left or right data type fails through `DataTypePromotion`, identifying the first
  invalid side in left-to-right order;
- an incompatible or locally unprovable shape pair fails through `ShapeBroadcast` with its existing
  result-axis diagnostic; and
- exhausted Tensor identity space fails through `TensorFactory.createDerived` only after all local
  model values have been constructed.

Do not catch, translate, aggregate, or replace these failures. Validation failures before
`createDerived` consume no Tensor ID. Do not add a production ID-inspection hook to test ordering.

### Input compatibility and result descriptor

All nine ordered floating type pairs are accepted. The existing hierarchy
`BFLOAT16 < FLOAT32 < FLOAT64` determines the common conceptual comparison domain. The helper does
not add a cast Operation, mutate either descriptor, or expose that common type in the BOOL output.

For every successful expression:

- `dataType` is exactly `DataType.BOOL`;
- `shape` is the exact immutable result of existing right-aligned broadcasting;
- `layout` is empty for both static and dynamic results; and
- `requiresGrad` is always false, even when either or both operands request gradients.

The descriptor records logical result facts only. It does not retain either input descriptor,
input layout, host storage, promoted type, broadcast strides, or a materialization decision.

### Provenance and identity

The output is a fresh Tensor with:

- a new factory-assigned opaque `TensorId`;
- an empty label;
- no host storage;
- one exact `BinaryComparisonKind` paired with `NoOperationAttrs.INSTANCE`; and
- ordered immutable provenance inputs containing exact references `[left, right]`.

Repeated inputs are valid. `tensor.equalTo(tensor)` stores the same exact Tensor reference in both
ordered positions. Repeating an equal comparison creates another Tensor identity. Symmetric
mathematical meaning does not authorize operand sorting, interning, or common-subexpression
elimination.

### No numerical or gradient behavior

Every valid call creates the requested semantic expression without reading storage. This task does
not decide how NaN, infinity, or signed zero compare and does not implement tolerance-based
equality. It also does not create gradient provenance from the BOOL result back to floating inputs.
Compiler, backend-conformance, and execution tasks own those later concerns within their
architecture boundaries.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorComparisonExpressions.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryComparisonTest.java`

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
  `BinaryComparisonKind` Javadocs and tests.
- Existing arithmetic/unary/scalar expression contracts, focused architecture documents, ADRs,
  architecture tests, backend-conformance tests, and integration tests.

## Maximum scope

At most two production files, two test files, and six documentation/planning files: ten paths
total.

`Tensor.java` and `TensorTest.java` may change only for the six exact public comparison methods,
their Javadocs, exact API-shape expectations, and non-synchronization assertions. Do not change
existing fields, constructor, metadata/storage behavior, arithmetic/unary/scalar methods,
equality, hashing, diagnostics, or unrelated tests.

If implementation needs another production/test concept, a changed foundational contract,
result-gradient behavior, resolved-layout policy, storage access, graph/compiler behavior, a
numerical edge policy, another documentation file, or more than ten paths, stop and propose a
follow-up or architecture decision. Do not create task 0015C.

## Javadoc requirements

- Update Tensor type Javadoc only as needed to include comparison expressions while preserving the
  distinction between public Tensor state, provenance, graph IR, and executable values.
- Every new public method must explain ordered operand roles, floating-only compatibility,
  broadcasting, fixed BOOL output, unresolved result layout, false gradient eligibility, fresh
  identity, storage absence, provenance, and deferral of numerical execution and gradient rules.
- Every method must document the non-null right operand with `@param`, the fresh derived BOOL Tensor
  with `@return`, and delegated null, data-type, shape, and identity-exhaustion failures with
  `@throws`.
- Document the package-private helper and `apply` method with exact validation/construction order,
  input ownership, fixed result facts, side effects, and failure behavior. The private constructor
  needs meaningful documentation.
- Keep all six methods independently understandable while using links to shared contracts to avoid
  unexplained repetition in generated Javadoc.
- Review related foundational and expression Javadocs and record why they remain accurate or stop
  on an out-of-scope inconsistency.

## Acceptance criteria

- Tensor declares exactly the six new public one-argument methods with parameter and return type
  `Tensor`; no overload, alias, or unrelated public API is added.
- Every method maps to the exact `BinaryComparisonKind` in the contract table and delegates once to
  the shared package-private helper.
- `TensorComparisonExpressions` has exactly the specified visibility, finality, constructor,
  method, and zero-field surface.
- Null validation order and messages are exact, and public null operands fail as `right`.
- All six methods accept all nine floating type pairs with compatible shapes and validate the
  common comparison domain through the existing promotion contract.
- Integral and boolean inputs are rejected without implicit casting, ID allocation, or storage
  access.
- Static scalar, zero-sized, rank-mismatched, singleton-expanded, and multi-axis broadcast results
  are represented correctly.
- Equal symbolic dimensions and symbolic/static-singleton pairs succeed; incompatible static or
  locally unprovable dynamic pairs fail through `ShapeBroadcast`.
- Every result descriptor has exact BOOL data type, empty layout, and false gradient eligibility
  regardless of input gradient flags.
- Every result is a fresh, unlabeled, storage-free Tensor with exact kind,
  `NoOperationAttrs.INSTANCE`, and immutable ordered exact input references.
- Self-use preserves the same input reference twice; repeated and symmetric calls are not interned,
  reordered, or canonicalized.
- No input Tensor metadata, provenance, label, storage association, or storage contents are mutated
  or retained as output storage.
- No numeric comparison, tolerance, gradient rule, graph state, backend fact, dependency, or
  architecture change is added.
- Focused and aggregate model tests, model Javadoc, root tests, reflection/javap/import/bytecode/
  scope checks, documentation links/formatting, and status synchronization pass.
- A separate documentation-focused agent finalizes Javadocs, Tensor API, Compile API, glossary,
  task evidence, master plan, and roadmap in the same change and records reasoned no-change
  conclusions for Training API, capabilities, architecture, and related contracts.
- Task 0015B becomes Complete only after both passes. Task 0015C remains Draft without a detailed
  specification.

## Tests / validation

Run before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorBinaryComparisonTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test must cover:

- exact helper class/constructor/method/field visibility and shape;
- all six public method-to-kind mappings;
- exact Operation and singleton attributes references;
- ordered exact provenance inputs, including relation direction, symmetric kinds, and repeated
  self-input;
- all nine floating type pairs and fixed BOOL result type;
- scalar, zero-sized, different-rank, singleton, multi-axis, and dynamic broadcasting;
- empty layout, always-false gradient eligibility, empty label, absent host storage, fresh identity,
  and absence of canonicalization;
- null helper/public operands, non-floating operands, incompatible static shapes, and unprovable
  dynamic shapes; and
- preservation of input metadata, provenance, labels, gradient eligibility, and storage
  associations.

Manually inspect `javap -p -c -s`, method bytecode, reflection, and imports for the exact Tensor
method descriptors, helper shape, one-delegation paths, validation order, fixed BOOL descriptor,
and no synchronization on the new public methods. Confirm no numeric access, storage operation,
cast operation, resolved layout, gradient propagation/rule, graph/compiler/runtime/backend type,
cost, fusion, route, registry, service, dependency, or build change appears. Validate generated
Javadoc, Tensor/Compile API status, glossary, links/anchors/fences/whitespace, exact ten-path scope,
synchronized statuses, and absence of a task-0015C specification.

## Dependencies

- Task 0001 supplies `DataType`, `DataTypePromotion`, floating categories, and fixed BOOL metadata.
- Task 0002 supplies `Shape` and `ShapeBroadcast` local right-aligned broadcasting.
- Task 0006 supplies immutable generic `Operation` composition.
- Task 0007 supplies `TensorDescriptor` and enforces non-differentiable BOOL descriptors.
- Task 0011 supplies public Tensor state and the exact API surface to extend.
- Task 0012 supplies centralized Tensor identity allocation through `TensorFactory`.
- Task 0013 supplies immutable ordered provenance and `TensorFactory.createDerived`.
- Task 0015A supplies the exact six parameterless `BinaryComparisonKind` values.

## Follow-up tasks

- 0015C remains Draft for parameterless BOOL logical semantic kinds.
- 0015D remains Draft for BOOL-only logical Tensor expressions and broadcasting.
- 0015E–0015F remain Draft for ternary `where` semantics and public expression construction.
- 0015G–0015H remain Draft for typed cast semantics and public expression construction.
- Compiler tasks later own provenance traversal, graph capture, inference, optimization, and
  training-graph treatment.
- Backend and conformance tasks later own exact numerical comparison behavior, mixed-floating
  lowering, storage access, and execution.

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None. The architecture already assigns public Tensor state, backend-independent
Operation semantics, descriptors, and minimal provenance to `modules/model`. The new helper remains
inside that boundary and adds no compiler, runtime, backend, device, or executable state.

If implementation requires graph capture, numerical comparison policy, gradient behavior, backend
metadata, another dependency, or architecture change, stop and report it.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0002/0006/0007/0011/0012/0013/0014B/0015A/0015B,
Tensor API, Compile API, Training API, glossary, current DataTypePromotion/ShapeBroadcast/
TensorDescriptor/Tensor/TensorFactory/TensorProvenance/Operation/BinaryComparisonKind contracts and
tests, and Java 26 Gradle configuration.

Implement task 0015B exactly. Modify Tensor.java and add package-private final
TensorComparisonExpressions.java for production. Update TensorTest only for the exact six-method
API surface and add TensorBinaryComparisonTest. Add exactly greaterThan/greaterOrEqual/lessThan/
lessOrEqual/equalTo/notEqualTo(Tensor), each delegating once to the shared helper and exact kind.

The helper must null-check left/right/kind, validate all floating pairs exactly once through
DataTypePromotion, broadcast exactly once through ShapeBroadcast, create an unresolved BOOL
descriptor with requiresGrad=false, create Operation(kind, NoOperationAttrs.INSTANCE), preserve
exact ordered provenance [left, right], and delegate once to TensorFactory.createDerived with no
label/storage. Every valid call returns a fresh derived Tensor. Do not inspect values/storage,
execute comparisons, define NaN/signed-zero/tolerance behavior, canonicalize or reorder operands,
insert casts, propagate gradient eligibility, define gradient rules, capture a graph, add
overloads, change existing contracts, or introduce compiler/runtime/backend behavior.

Stop beyond ten paths or on architecture uncertainty. Run every specified focused/aggregate test,
Javadoc, javap/reflection/bytecode/import/manual, documentation/link/whitespace/scope/status check.
Then hand the actual diff and evidence to a separate clean-context documentation agent in the same
change. It must inspect source/tests/generated Javadoc, finalize permitted Javadocs/Tensor API/
Compile API/glossary/planning, record related-contract/capability/Training API/architecture
no-change conclusions, and rerun validation.

Update task 0015B, model master plan, and roadmap only for planning status/evidence. Do not mark
0015B Complete until both passes succeed. Leave 0015C Draft without a specification. Do not commit
or push.
```

## Local decisions

- Public method names preserve the selected legacy capability while semantic enum names remain
  descriptive uppercase identities.
- All six comparisons are floating-only, including equality and inequality. Integral and BOOL
  comparisons require a future explicit capability decision rather than accidental expansion.
- Mixed floating pairs are accepted because the legacy public validation allowed them and the
  current model has one explicit floating-promotion hierarchy. Promotion validates the comparison
  domain; the result descriptor remains BOOL and stores no promoted type.
- Comparison output always has `requiresGrad=false`. Input gradient eligibility is neither mutated
  nor propagated through a non-differentiable BOOL descriptor.
- Operand order is preserved for every relation. Symmetric equality does not authorize a different
  provenance contract or eager canonicalization.
- No comparison attrs record is introduced because every kind is parameterless and
  `NoOperationAttrs.INSTANCE` is complete.

## Known limitations

- The task constructs semantic comparison expressions but does not calculate boolean values.
- NaN, infinity, signed-zero, equality tolerance, and cross-precision execution behavior remain
  unspecified until backend/conformance work.
- Integral and BOOL comparisons are not supported by this selected public baseline.
- No compiler capture, training-graph handling, gradient rule, backend support, or execution is
  implied.

## Validation evidence

Planning reviewed the architecture contract and focused module/dependency explanations;
documentation and planning rules; roadmap; model capabilities and master plan; tasks 0001, 0002,
0006, 0007, 0011, 0012, 0013, 0014B, and 0015A; current Tensor, descriptor, provenance, promotion,
broadcast, operation, and comparison-kind source/tests; Tensor/Compile/Training APIs and glossary;
and Java 26 Gradle configuration.

The read-only `legacy/pre-rewrite` branch was inspected directly. It confirms six public Tensor
method names, floating-only local eligibility, binary broadcasting, BOOL/no-gradient results,
ordered operands, and execution uses as masks and `where` conditions. Mutable legacy broadcast
plans, runtime coupling, traits, kernels, execution, and gradient/compiler machinery are excluded.

Planning selected one helper and one focused test because all six methods share the same validation,
descriptor, provenance, and factory path. The task adds no package, dependency, foundation change,
or architecture decision. Task 0015C remains only the next Draft queue entry.

Planning validation:

- `git diff --check` passed, and targeted whitespace inspection found no trailing whitespace in
  the three changed planning paths.
- The required-section scan found every canonical task-specification section, including package
  impact, exact scope, validation, implementation handoff, decisions, limitations, and completion
  evidence sections.
- The relative Markdown-target scan resolved every local `.md` link in this task, the model master
  plan, and the roadmap. Markdown fence counts are balanced and no changed link uses an unresolved
  heading anchor.
- Status inspection found 0015B `Ready` in this specification, its linked model-master row, and
  its linked roadmap row/current-frontier text. Task 0015C remains `Draft` in both queues.
- Scope inspection found exactly this new task, the model master plan, and the roadmap changed. No
  Java, test, API, glossary, Gradle, architecture, AGENTS, or other module path changed.
- No task-0015C specification exists.

Implementation and documentation validation:

- Implementation context `/root/implement_model_0015b` added the six exact public Tensor methods,
  one package-private helper, the focused comparison test, and the exact Tensor API-shape test
  updates. Independent documentation context
  `/root/implement_model_0015b/review_model_0015b_docs` then reread the architecture contract,
  focused model/compiler/training architecture pages, documentation and planning rules, current
  APIs, glossary, capabilities/master plans, relevant completed tasks, actual diff/source/tests,
  test reports, bytecode, imports, and generated Javadoc. It applied General plus API/Javadoc
  style to API and Javadoc review, Planning style to planning files, and Example format to the new
  complete comparison-expression example.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorBinaryComparisonTest` passed during the final pass
  from the Gradle build cache. The final fresh aggregate execution contains all 9 focused tests
  with zero failures, errors, or skips.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorTest` passed during the final pass from the Gradle
  build cache. The final fresh aggregate execution contains all 14 Tensor tests with zero failures,
  errors, or skips.
- `./gradlew :modules:model:test` passed, then
  `./gradlew :modules:model:test --rerun-tasks` freshly compiled and executed the suite. The 42 XML
  suites contain 317 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:javadoc` passed up to date, then
  `./gradlew :modules:model:javadoc --rerun-tasks` freshly regenerated Javadoc without warnings.
  Generated `Tensor.html` contains exactly one detail section for each of the six methods, with
  parameter, return, failure, fixed BOOL/false-gradient, provenance, and deferred-behavior text.
  The package-private helper Javadoc was reviewed in source because public Javadoc generation does
  not publish package-private types.
- `./gradlew test` passed for the repository with 36 actionable tasks up to date and no failure.
  Architecture, backend-conformance, and integration test modules currently have no test sources;
  the root run still compiled their existing module markers without error.
- The complete `ComparisonExpressionExample` was compiled with
  `javac --release 26 -cp modules/model/build/classes/java/main -d /private/tmp` and run with the
  model classes. Its ten output lines exactly matched the documented `BOOL`, `Shape[2, 3]`,
  unresolved-layout, false-gradient, unlabeled, storage-free, `LESS_OR_EQUAL`, parameterless,
  ordered-input, and fresh-identity results. The temporary source was removed and is not in scope.
- `javap -p -c -s` confirms exactly six `(Tensor)Tensor` public methods, each loading the exact
  kind and invoking `TensorComparisonExpressions.apply` once. The helper is final and package-
  private with zero fields, one private zero-argument constructor, and one package-private static
  method; bytecode confirms left/right/kind null checks, one floating-promotion call, one broadcast
  call, fixed `DataType.BOOL`/empty-layout/false-gradient descriptor, one Operation, ordered
  `List.of(left, right)` provenance, and one `createDerived` call. No monitor instruction or
  synchronized comparison method exists.
- Reflection tests confirm the exact helper and six-method API shapes, no overloads, and no
  synchronization. Import and bytecode scans found only model-owned data-type, shape, operation,
  Tensor, and JDK collection/null-check contracts; no graph/compiler/planning/prepare/runtime/
  backend type, storage access, cast, gradient rule, registry, service, numerical execution, or
  build dependency appears.
- The first two local-link validator attempts failed because the inline Ruby checker used an
  interpolated heading regex and then a `filter_map` method unavailable in the installed Ruby.
  After replacing those checker constructs, one run reported the existing
  `compiled-graph--compiledgraphmodel` anchor falsely because whitespace was collapsed unlike
  GitHub heading rules. The corrected validator preserves adjacent heading spaces as adjacent
  hyphens and then resolved every local file target and heading anchor across all six changed
  documentation/planning files. These were validation-script defects, not documentation defects.
- Markdown fence counts are balanced; targeted scans found no trailing whitespace in any of the
  ten paths; terminology and implementation-status scans found no stale claim that comparison
  construction is planned; and `git diff --check` passed.
- Final scope contains exactly the ten authorized paths: two production files, two test files,
  Tensor API, Compile API, glossary, this task, model master plan, and roadmap. Task 0015B is
  synchronized as Complete in all three planning locations. Task 0015C remains Draft and no
  task-0015C specification exists.
- The independent documentation pass found the Tensor type, all six public method Javadocs, and
  helper type/constructor/apply Javadocs complete and accurate, so it changed no Java declaration,
  executable logic, or test. Tensor API now documents the implemented comparison surface and a
  complete runnable metadata example. Compile API now inventories the six current comparison
  methods while keeping compiler entry, traversal, capture, inference, optimization, artifacts,
  and engine behavior planned.
- Existing glossary terms were sufficient. Implementation status, `OperationKind`, Provenance,
  Tensor, and common distinctions were updated for comparison construction; no new reusable
  domain term was introduced.
- Training API remains accurate unchanged because false descriptor gradient eligibility adds no
  gradient object, rule, autograd, optimizer, session, or training-graph treatment. Model
  capabilities remain accurate unchanged because they already select all six public comparison
  names, floating-category rules, broadcasting, BOOL results, and layer separation.
- `ARCHITECTURE.md`, focused architecture pages, and ADRs remain accurate unchanged because the
  change stays within model-owned Tensor semantics, descriptors, and provenance and adds no module
  boundary, dependency, lifecycle, backend, runtime, or training decision. Architecture tests need
  no update because dependency rules did not change. Backend-conformance and integration tests
  need no update because no numerical backend behavior or end-to-end execution exists yet.
- Java 26/Gradle configuration remains accurate unchanged because the change uses existing stable
  language/library contracts and adds no dependency, preview/incubator feature, or build change.
  Existing `DataTypePromotion`, `ShapeBroadcast`, `TensorDescriptor`, `TensorProvenance`,
  `TensorFactory.createDerived`, `Operation`, `NoOperationAttrs`, `BinaryComparisonKind`, and
  arithmetic/unary/scalar expression Javadocs and tests remain accurate because this task composes
  their existing contracts without changing them.

## Implementation notes

- Added exactly six public Tensor comparison methods and one package-private stateless helper.
- Every successful comparison creates a fresh, unlabeled, storage-free unresolved BOOL result with
  false gradient eligibility and exact ordered operation provenance after existing floating and
  local-broadcast validation.
- The independent documentation pass finalized Tensor API, Compile API, glossary, and planning
  status/evidence. It preserved the implementation and already-complete Javadocs unchanged.
- No compiler, training, architecture, dependency, numerical execution, or backend behavior was
  added.

## Completion summary

- Completed changes: Implemented and documented the six floating-only, broadcast-aware public
  binary comparison Tensor expressions with fixed BOOL results and ordered provenance.
- Files changed or created: Exactly the authorized two production files, two tests, Tensor API,
  Compile API, glossary, this task, model master plan, and roadmap.
- Tests and validation: Focused comparison tests 9/9, focused Tensor tests 14/14, all 317 model
  tests across 42 suites, regenerated model Javadoc, root tests, example compilation/execution,
  bytecode/reflection/import/absence checks, local links/anchors, fences, terminology, whitespace,
  exact scope/status, and `git diff --check` passed.
- Documentation-agent review: Clean documentation context
  `/root/implement_model_0015b/review_model_0015b_docs` completed the independent pass using
  General, API/Javadoc, Planning, and Example-format guidance.
- Documentation impact: Tensor API and Compile API now describe current comparison expression
  construction while preserving all compiler and execution lifecycle behavior as planned.
  Training API, capabilities, architecture/ADRs/tests, conformance/integration tests, build
  configuration, and related foundational/expression documentation remain accurate unchanged for
  the reasons recorded above.
- Javadoc review: Tensor and helper Javadocs fully document ordered inputs, floating validation,
  broadcasting, fixed result facts, identity/storage/provenance, failures, and deferred behavior;
  no documentation-pass Java edit was needed.
- Glossary impact: Existing implementation-status, `OperationKind`, Provenance, Tensor, and common-
  distinction text now includes comparison construction; no new reusable term was needed.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0015B. Task 0015C remains Draft without a specification.

Status: Complete

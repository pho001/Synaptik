# Task 0015D: Boolean Logical Tensor Expressions

## Status

Complete

## Goal

Expose the three implemented boolean logical semantics as public, backend-independent Tensor
expression methods. Binary conjunction and disjunction must accept only BOOL operands and derive a
locally provable right-aligned broadcast shape. Unary negation must accept one BOOL input and retain
its exact shape. Every successful call returns a fresh storage-free, non-differentiable BOOL Tensor
whose provenance records the exact logical operation and input Tensor identities.

This task makes boolean logic capturable through the public model API. It does not read truth
values, execute logical operations, allocate result storage, capture a compiled graph, or define
backend support.

## Scope

- Add public `Tensor.logicalAnd(Tensor)`, `logicalOr(Tensor)`, and `logicalNot()` methods.
- Add one package-private `TensorLogicalExpressions` helper with separate binary and unary
  validation entries and one private shared result-construction path.
- Require exact `DataType.BOOL` inputs; reject every floating and integral input without numeric
  truthiness or implicit conversion.
- Permit only `BooleanLogicalKind.AND` and `OR` through the binary helper and only `NOT` through the
  unary helper.
- Compute binary result shape through exactly one `ShapeBroadcast.broadcast` call.
- Retain the exact unary input `Shape` reference without invoking broadcasting.
- Create an unresolved-layout BOOL result descriptor with `requiresGrad=false` for every operation.
- Construct exactly one `Operation` from the selected `BooleanLogicalKind` and
  `NoOperationAttrs.INSTANCE`.
- Construct exactly one `TensorProvenance`: ordered `[left, right]` for AND/OR and exact `[input]`
  for NOT.
- Delegate final identity-bearing construction exactly once to `TensorFactory.createDerived` with
  no label and no storage.
- Update the exact Tensor public-API reflection test and add one focused boolean-logical expression
  test.
- Finalize affected Javadocs, Tensor API, Compile API status, glossary, task evidence, model master
  plan, and roadmap through the required independent documentation pass during implementation.

## Out of scope

- eager truth-value evaluation, reading input storage, short-circuiting, constant folding,
  canonicalization, interning, or returning an existing input/result
- numeric truthiness, bitwise operations, floating/integral inputs, implicit conversion, explicit
  cast insertion, nullable/unknown values, or three-valued logic
- XOR, NAND, NOR, XNOR, implication, equivalence, reduction `all`/`any`, or operator aliases
- scalar boolean overloads, static factory forms, variadic logical methods, in-place mutation, or
  an expression-builder public type
- output labels, caller-supplied labels, expression strings, symbols, or serialization names
- resolved output layouts, broadcast stride plans, zero-stride views, aliases, materialization
  policy, or layout preservation from an input
- gradient eligibility propagation, gradient values/rules, straight-through estimators, backward
  graph construction, training-root policy, autograd, optimizer, or training execution
- operation-family attributes, factories, registries, parsers, reflection discovery, or changes to
  `BooleanLogicalKind`, `Operation`, `OperationKind`, `OperationAttrs`, or `NoOperationAttrs`
- graph traversal, cycle checks, node/value IDs, graph capture, compiled graph construction,
  common-subexpression elimination, compiler inference, or publication binding
- planning ownership, capability providers, backend support, fusion, cost, lowering, kernels,
  runtime residency, prepare, execution, tracing, or engine behavior
- dependencies, Gradle changes, architecture changes, another module, another operation family, or
  a detailed task-0015E specification

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
- [Task 0015A](0015a-binary-comparison-semantic-kinds.md)
- [Task 0015B](0015b-binary-comparison-tensor-expressions.md)
- [Task 0015C](0015c-boolean-logical-semantic-kinds.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes `Tensor.logicalAnd(Tensor)`,
`logicalOr(Tensor)`, and `logicalNot()`. Its public construction path accepts only BOOL inputs;
AND/OR apply right-aligned broadcasting, NOT preserves the input shape, and all three produce
non-gradient BOOL results. Legacy tests cover truth tables, AND broadcasting, logical chains,
comparison-mask inputs, `where` conditions, non-contiguous layouts, ONNX mappings, and multiple
backend execution routes.

The legacy implementation also stores mutable broadcast plans in binary operation descriptors,
attaches runtime graph state, reads BOOL storage during execution, and exposes fusion, cost,
result-kind, and backend-facing traits. Those mechanisms are not copied. The new model retains the
selected public expression capability through immutable descriptors and provenance. Graph capture
belongs to the compiler; storage interpretation, truth-value execution, and conformance belong to
prepared backends.

## Architecture constraints

- `Tensor` remains public mutable API state and must not become an IR node.
- `TensorLogicalExpressions` performs deterministic local model validation only. It must not read
  truth values, inspect storage, traverse provenance, capture a graph, or inspect backend support.
- `Operation` owns only backend-independent semantic kind and attributes. It contains no inputs,
  arity field, result descriptor, backend support, or executable behavior.
- Helper entry selection enforces family arity locally: `applyBinary` accepts only `AND` or `OR`,
  and `applyUnary` accepts only `NOT`. Generic `Operation` remains unchanged and family-agnostic.
- `TensorProvenance` owns exact input identities. AND/OR retain caller order `[left, right]` even
  though their truth functions are commutative; NOT retains `[input]`.
- Result identity comes only from the existing package-private `TensorFactory.createDerived` seam.
  No second allocator, caller-supplied ID, registry, cache, or service is introduced.
- Every result is storage-free and has `DataType.BOOL`, `Optional.empty()` layout, and
  `requiresGrad=false`. Public expression construction does not allocate physical buffers or
  attach input storage.
- Binary dynamic shapes are accepted only where `ShapeBroadcast` proves compatibility locally:
  equal symbolic dimensions and static singleton expansion. The helper creates no constraints.
- Unary NOT retains the exact immutable input `Shape` reference and does not invoke broadcasting,
  copy dimensions, or preserve a resolved layout.
- Exact BOOL descriptor identity is required. No floating/integral value, raw non-zero byte, or
  other representation is interpreted as logical truth.
- Package direction is `model.tensor -> model.operation.elementwise.logical`, plus existing
  `model.tensor -> model.operation`, `model.datatype`, and `model.shape`. The operation package must
  not import Tensor and no package cycle may be introduced.
- Stop if implementation requires a changed foundational contract, resolved layout, storage
  access, graph capture, gradient behavior, dependency, or architecture decision.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns public Tensor logical methods, local validation,
  result construction, provenance, and the derived Tensor factory seam.
- `io.github.pho001.synaptik.model.operation` — supplies `Operation` and
  `NoOperationAttrs.INSTANCE`.
- `io.github.pho001.synaptik.model.operation.elementwise.logical` — supplies the exact AND, OR, and
  NOT semantic values.
- `io.github.pho001.synaptik.model.datatype` — supplies exact BOOL identity for input validation and
  result descriptors.
- `io.github.pho001.synaptik.model.shape` — supplies binary local right-aligned broadcasting and
  the immutable unary shape value.

No package is added.

Type placement:

- `io.github.pho001.synaptik.model.tensor.Tensor` — public fluent expression surface; it receives
  only the three exact logical methods and delegates shared behavior.
- `io.github.pho001.synaptik.model.tensor.TensorLogicalExpressions` — package-private, stateless
  binary/unary validation and construction boundary colocated with Tensor, descriptors,
  provenance, and the package-private factory seam it must use.
- `TensorBooleanLogicalTest` — same-package focused test so it can verify the package-private
  helper without widening production visibility.

## Required contract

### Public Tensor surface

Add exactly these public methods to `Tensor`:

```java
public Tensor logicalAnd(Tensor right)
public Tensor logicalOr(Tensor right)
public Tensor logicalNot()
```

The methods map and delegate exactly once as follows:

| Tensor method | Exact delegation |
|---|---|
| `logicalAnd(right)` | `TensorLogicalExpressions.applyBinary(this, right, BooleanLogicalKind.AND)` |
| `logicalOr(right)` | `TensorLogicalExpressions.applyBinary(this, right, BooleanLogicalKind.OR)` |
| `logicalNot()` | `TensorLogicalExpressions.applyUnary(this, BooleanLogicalKind.NOT)` |

Each method returns the helper's exact result and performs no separate validation, broadcasting,
descriptor creation, allocation, provenance construction, canonicalization, or storage access.
There are no overloads or aliases in this task.

### Package-private helper shape

Create exactly one package-private final non-record class:

```java
final class TensorLogicalExpressions {
    private TensorLogicalExpressions() {
    }

    static Tensor applyBinary(Tensor left, Tensor right, BooleanLogicalKind kind) {
        // exact binary contract below
    }

    static Tensor applyUnary(Tensor input, BooleanLogicalKind kind) {
        // exact unary contract below
    }

    private static Tensor create(
            Shape shape,
            BooleanLogicalKind kind,
            List<Tensor> inputs) {
        // exact common construction below
    }
}
```

The helper has no fields, nested types, public/protected members, overloads, caches, registries, or
additional methods. Its constructor prevents instantiation. The two package-private entries make
the family arity explicit without adding arity metadata to `BooleanLogicalKind` or widening a
public expression service.

### Binary validation and construction order

`applyBinary` performs these steps in exact order:

1. require non-null `left`, `right`, and `kind`, in that order, with messages `left`, `right`, and
   `kind`;
2. reject `BooleanLogicalKind.NOT` with `IllegalArgumentException` message
   `binary logical expression kind must be AND or OR, but was NOT`;
3. require `left.descriptor().dataType() == DataType.BOOL`; otherwise throw
   `IllegalArgumentException` message `left must have BOOL data type, but was <actual>`;
4. require `right.descriptor().dataType() == DataType.BOOL`; otherwise throw
   `IllegalArgumentException` message `right must have BOOL data type, but was <actual>`;
5. call `ShapeBroadcast.broadcast(left.descriptor().shape(), right.descriptor().shape())` exactly
   once;
6. call private `create(shape, kind, List.of(left, right))` exactly once and return its exact
   result.

Kind compatibility is checked before input data types so package-private misuse fails on the
semantic family error without inspecting descriptors further. Left data type is checked before
right. Shape validation occurs only after both data types are valid.

### Unary validation and construction order

`applyUnary` performs these steps in exact order:

1. require non-null `input` and `kind`, in that order, with messages `input` and `kind`;
2. reject `AND` or `OR` with `IllegalArgumentException` message
   `unary logical expression kind must be NOT, but was <actual>`;
3. require `input.descriptor().dataType() == DataType.BOOL`; otherwise throw
   `IllegalArgumentException` message `input must have BOOL data type, but was <actual>`;
4. read the exact `input.descriptor().shape()` reference without broadcasting or reconstruction;
5. call private `create(shape, kind, List.of(input))` exactly once and return its exact result.

The public `logicalNot()` path always supplies `NOT`; wrong-kind failures exist to make the shared
package-private helper contract explicit and locally safe.

### Common result construction

Private `create` performs these steps in exact order:

1. create exactly one `TensorDescriptor` from `DataType.BOOL`, the exact supplied shape,
   `Optional.empty()` layout, and `false` gradient eligibility;
2. create exactly one `Operation(kind, NoOperationAttrs.INSTANCE)`;
3. create exactly one `TensorProvenance(operation, inputs)`;
4. call `TensorFactory.createDerived(descriptor, Optional.empty(), provenance)` exactly once and
   return its exact result.

Only the validated binary and unary entries call `create`; it performs no duplicate family-arity,
data-type, or shape validation. `TensorProvenance` snapshots the supplied ordered list under its
existing contract.

### Failure and identity side effects

Null, kind, data-type, and shape validation complete before ID allocation:

- public `logicalAnd(null)` and `logicalOr(null)` fail with `NullPointerException("right")`;
- non-BOOL inputs fail with the exact left/right/input messages above;
- incompatible or locally unprovable binary shapes fail through `ShapeBroadcast` with its existing
  result-axis diagnostic; and
- exhausted Tensor identity space fails through `TensorFactory.createDerived` only after local
  model values are constructed.

Do not catch, translate, aggregate, or replace `ShapeBroadcast` or factory failures. Failures before
`createDerived` consume no Tensor ID. Do not add a production ID-inspection hook to test ordering.

### Result descriptor, provenance, and identity

Every successful result has:

- exact `DataType.BOOL`;
- binary broadcast shape or the exact unary input `Shape` reference;
- empty layout for static and dynamic shapes;
- `requiresGrad=false`;
- a fresh factory-assigned `TensorId`;
- an empty label and no host storage;
- the exact logical kind paired with `NoOperationAttrs.INSTANCE`; and
- exact immutable provenance inputs `[left, right]` or `[input]`.

Self-use is valid for binary methods and stores the same Tensor reference twice. AND and OR do not
reorder operands. Repeating a valid call creates another identity. `logicalNot().logicalNot()`
creates two distinct NOT expressions rather than returning the original tensor.

### No eager logical behavior

The helper never reads input storage or interprets truth bytes. It does not short-circuit, simplify
`x AND x`, simplify `x OR x`, collapse double negation, apply De Morgan rewrites, or reorder
commutative inputs. Compiler optimization and backend execution may address those concerns later
under their own correctness and conformance contracts.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorLogicalExpressions.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBooleanLogicalTest.java`

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
- Existing `DataType`, `Shape`, `ShapeBroadcast`, `TensorDescriptor`, `TensorProvenance`,
  `TensorFactory`, `Operation`, `NoOperationAttrs`, and `BooleanLogicalKind` Javadocs/tests.
- Existing arithmetic, unary, scalar, and comparison expression contracts; focused architecture
  documents; ADRs; architecture tests; backend-conformance tests; and integration tests.

## Maximum scope

At most two production files, two test files, and six documentation/planning files: ten paths
total.

`Tensor.java` and `TensorTest.java` may change only for the three exact public logical methods,
their Javadocs, exact API-shape expectations, and non-synchronization assertions. Do not change
existing fields, constructor, metadata/storage behavior, other expression methods, equality,
hashing, diagnostics, or unrelated tests.

If implementation needs another production/test concept, a changed foundational contract,
resolved-layout policy, storage access, graph/compiler behavior, gradient behavior, another
documentation file, or more than ten paths, stop and propose a follow-up or architecture decision.
Do not create task 0015E.

## Javadoc requirements

- Update Tensor type Javadoc only as needed to include boolean logical expressions while preserving
  the distinction between public Tensor state, provenance, graph IR, and executable values.
- `logicalAnd` and `logicalOr` must document ordered operands, exact BOOL eligibility, binary
  broadcasting, fixed BOOL output, unresolved layout, false gradient eligibility, fresh identity,
  storage absence, provenance, and deferred truth-value execution.
- `logicalNot` must document exact BOOL eligibility, exact input-shape retention, fixed BOOL output,
  unresolved layout, false gradient eligibility, fresh identity, storage absence, one-input
  provenance, and deferred execution.
- Every public binary method must document non-null `right` with `@param`, and every method must
  document its fresh result with `@return` and applicable null, data-type, shape, and identity
  exhaustion failures with `@throws`. Helper Javadocs additionally document wrong-kind failures.
- Document the package-private helper, constructor, both entries, and private `create` method with
  exact validation/construction order, input ownership, side effects, and failure behavior.
- Explain that logical operations are eager only as expression metadata: there is no Java-style
  short-circuiting or storage access during construction.
- Review related foundational and expression Javadocs and record why they remain accurate or stop
  on an out-of-scope inconsistency.

## Acceptance criteria

- Tensor declares exactly `logicalAnd(Tensor)`, `logicalOr(Tensor)`, and `logicalNot()` with return
  type `Tensor`; no overload, alias, or unrelated public API is added.
- Every public method delegates once to the exact helper entry and semantic kind in the contract
  table.
- `TensorLogicalExpressions` has exactly the specified visibility, finality, constructor, three
  methods, and zero-field/zero-nested-type surface.
- Binary and unary null checks, kind compatibility, BOOL validation order, exception types, and
  messages are exact.
- Binary AND/OR call `ShapeBroadcast` exactly once; NOT calls it zero times and retains the exact
  input Shape reference.
- Binary scalar, zero-sized, rank-mismatched, singleton-expanded, multi-axis, equal-symbolic, and
  symbolic/singleton shapes succeed; incompatible or locally unprovable shapes fail.
- Every floating and integral data type is rejected on every logical input position without
  conversion, storage access, or ID allocation.
- Every result has exact BOOL data type, empty layout, false gradient eligibility, empty label, no
  host storage, fresh identity, exact operation/attributes, and exact ordered provenance.
- Comparison-expression BOOL results can feed AND/OR/NOT without special handling.
- Self-use and repeated calls remain fresh and ordered; double NOT is not collapsed.
- No input Tensor metadata, provenance, label, storage association, or storage contents are mutated
  or retained as output storage.
- No logical evaluation, truthiness, gradient rule, graph state, backend fact, dependency, or
  architecture change is added.
- Focused and aggregate model tests, model Javadoc, root tests, reflection/javap/import/bytecode/
  scope checks, documentation links/formatting, and status synchronization pass.
- A separate documentation-focused agent finalizes Javadocs, Tensor API, Compile API, glossary,
  task evidence, master plan, and roadmap in the same change and records reasoned no-change
  conclusions for Training API, capabilities, architecture, and related contracts.
- Task 0015D becomes Complete only after both passes. Task 0015E remains Draft without a detailed
  specification.

## Tests / validation

Run before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorBooleanLogicalTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test must cover:

- exact helper class/constructor/method/field visibility and signatures;
- all three public method-to-entry/kind mappings;
- exact Operation and singleton attributes references;
- binary ordered and unary exact provenance, including self-use and chained comparison masks;
- exact BOOL-only eligibility and rejection of all five non-BOOL types at every input position;
- scalar, zero-sized, different-rank, singleton, multi-axis, and dynamic binary broadcasting;
- exact unary Shape reference retention without broadcasting;
- empty layout, false gradient eligibility, empty label, absent storage, fresh identity, and no
  canonicalization;
- public/helper nulls, binary/unary wrong kinds, incompatible static shapes, and unprovable dynamic
  shapes; and
- preservation of input metadata, provenance, labels, and storage associations.

Manually inspect `javap -p -c -s`, method bytecode, reflection, and imports for the exact Tensor
method descriptors, helper shape, delegation paths, validation order/messages, one binary
broadcast call, zero unary broadcast calls, fixed BOOL descriptor, and absence of synchronization
on new public/helper entries. Confirm no numeric access, storage operation, cast, resolved layout,
gradient behavior, graph/compiler/runtime/backend type, cost, fusion, route, registry, service,
dependency, or build change appears. Validate generated Javadoc, Tensor/Compile API status,
glossary, links/anchors/fences/whitespace, exact ten-path scope, synchronized statuses, and absence
of a task-0015E specification.

## Dependencies

- Task 0001 supplies exact `DataType.BOOL` identity.
- Task 0002 supplies `Shape` and `ShapeBroadcast` local right-aligned broadcasting.
- Task 0006 supplies immutable generic `Operation` composition.
- Task 0007 supplies `TensorDescriptor` and its non-differentiable BOOL invariant.
- Task 0011 supplies public Tensor state and the exact API surface to extend.
- Task 0012 supplies centralized Tensor identity allocation through `TensorFactory`.
- Task 0013 supplies immutable ordered provenance and `TensorFactory.createDerived`.
- Task 0015B supplies current comparison expressions that produce BOOL inputs for logical chains.
- Task 0015C supplies exact parameterless AND, OR, and NOT semantic kinds.

## Follow-up tasks

- 0015E remains Draft for parameterless ternary `where` selection semantics.
- 0015F remains Draft for BOOL-condition and branch-validated broadcast `where` construction.
- 0015G–0015H remain Draft for typed cast semantics and expression construction.
- Compiler tasks later own provenance traversal, graph capture, optimization, and training-graph
  treatment.
- Backend and conformance tasks later own BOOL storage interpretation, truth-value execution,
  lowering, and kernel behavior.

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None. The architecture already assigns public Tensor state, backend-independent
Operation semantics, descriptors, and minimal provenance to `modules/model`. The helper remains
inside that boundary and adds no compiler, runtime, backend, device, or executable state.

If implementation requires graph capture, storage truth interpretation, gradient behavior,
backend metadata, another dependency, or architecture change, stop and report it.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0002/0006/0007/0011/0012/0013/0015A/0015B/0015C/
0015D, Tensor API, Compile API, Training API, glossary, current DataType/Shape/ShapeBroadcast/
TensorDescriptor/Tensor/TensorFactory/TensorProvenance/Operation/BooleanLogicalKind contracts and
tests, and Java 26 Gradle configuration.

Implement task 0015D exactly. Modify Tensor.java and add package-private final
TensorLogicalExpressions.java for production. Update TensorTest only for the exact three-method API
surface and add TensorBooleanLogicalTest. Add exactly logicalAnd(Tensor), logicalOr(Tensor), and
logicalNot(), each delegating once to the exact shared helper entry and semantic kind.

The helper has exactly applyBinary(left,right,kind), applyUnary(input,kind), and private
create(shape,kind,inputs). Follow the task's exact null/kind/BOOL/shape/construction order and
messages. Binary accepts only AND/OR and calls ShapeBroadcast exactly once; unary accepts only NOT,
retains the exact input Shape, and never broadcasts. Common construction creates an unresolved
BOOL descriptor with requiresGrad=false, exact Operation/NoOperationAttrs, exact ordered
provenance, and delegates once to TensorFactory.createDerived with no label/storage. Every valid
call returns a fresh Tensor.

Do not inspect values/storage, execute truth logic, support numeric truthiness, short-circuit,
canonicalize/reorder/collapse expressions, insert casts, propagate gradient eligibility, define
gradient rules, capture a graph, add overloads, change existing contracts, or introduce compiler/
runtime/backend behavior. Stop beyond ten paths or on architecture uncertainty.

Run every specified focused/aggregate test, Javadoc, javap/reflection/bytecode/import/manual,
documentation/link/whitespace/scope/status check. Then hand the actual diff/evidence to a separate
clean-context documentation agent in the same change. It must inspect source/tests/generated
Javadoc, finalize permitted Javadocs/Tensor API/Compile API/glossary/planning, record related-
contract/capability/Training API/architecture no-change conclusions, and rerun validation.

Update task 0015D, model master plan, and roadmap only for planning status/evidence. Do not mark
0015D Complete until both passes succeed. Leave 0015E Draft without a specification. Do not commit
or push.
```

## Local decisions

- One package-private helper owns both arities because AND, OR, and NOT share exact BOOL
  validation, fixed descriptor facts, parameterless Operation composition, provenance, and factory
  construction. Separate production helpers would duplicate the same boundary.
- `applyBinary` and `applyUnary` enforce kind compatibility locally because generic `Operation`
  intentionally stores no arity and performs no family compatibility validation.
- Exact BOOL descriptors are required. Numeric truthiness is rejected rather than inherited from
  storage bytes or legacy implementation details.
- Binary AND/OR preserve caller order despite mathematical commutativity. Compiler optimization may
  canonicalize later; public expression construction does not.
- NOT preserves the exact input Shape reference but leaves result layout unresolved because a
  semantic expression does not assert storage geometry.
- Every logical result is non-differentiable (`requiresGrad=false`); no gradient propagation or
  straight-through rule is introduced.

## Known limitations

- Logical expressions contain semantics and provenance but no calculated truth values.
- Only BOOL inputs are accepted; XOR-family, bitwise, numeric, nullable, and three-valued logic are
  unsupported.
- No compiler capture, logical simplification, training-graph rule, backend support, storage
  interpretation, or execution is implied.

## Validation evidence

Planning reviewed the architecture contract and focused module/dependency explanations;
documentation and planning rules; roadmap; model capabilities and master plan; tasks 0001, 0002,
0006, 0007, 0011, 0012, 0013, 0015A, 0015B, and 0015C; current DataType, ShapeBroadcast,
TensorDescriptor, Tensor, TensorFactory, TensorProvenance, Operation, BooleanLogicalKind, and
expression-helper source/tests; Tensor/Compile/Training APIs and glossary; and Java 26 Gradle
configuration.

The read-only `legacy/pre-rewrite` branch was inspected directly. It confirms public
`logicalAnd`, `logicalOr`, and `logicalNot` names, BOOL-only eligibility, binary broadcasting,
unary shape preservation, BOOL no-gradient results, logical/comparison chaining, and backend/ONNX
evidence. Mutable legacy broadcast plans, runtime graph coupling, numeric storage access, traits,
kernels, execution, and compiler behavior are excluded.

Planning selected one helper with explicit binary/unary entries and one private common constructor.
This keeps family arity validation visible without adding enum metadata and centralizes the shared
fixed BOOL descriptor, Operation, provenance, and factory path. No package, dependency,
foundation contract, or architecture rule changes.

Planning validation:

- `git diff --check` passed, and targeted whitespace inspection found no trailing whitespace in
  the three changed planning paths.
- The required-section scan found every canonical task-specification section, including package
  impact, exact scope, validation, implementation handoff, decisions, limitations, and completion
  evidence sections.
- The relative Markdown-target scan resolved every local `.md` link in this task, the model master
  plan, and the roadmap. Markdown fence counts are balanced and no changed link uses an unresolved
  heading anchor.
- Status inspection found 0015D `Ready` in this specification, its linked model-master row, and
  its linked roadmap row/current-frontier text. Task 0015E remains `Draft` in both queues.
- Scope inspection found exactly this new task, the model master plan, and the roadmap changed. No
  Java, test, API, glossary, Gradle, architecture, AGENTS, or other module path changed.
- The planned implementation scope contains exactly two production paths, two test paths, and six
  documentation/planning paths. No task-0015E specification exists.

Implementation and documentation validation:

- Implementation context `/root/implement_model_0015d` added exactly the three public Tensor
  methods, one package-private helper, the focused logical-expression test, and the exact Tensor
  API-shape test updates. Independent documentation context
  `/root/implement_model_0015d/review_model_0015d_docs` reread the architecture contract, focused
  architecture pages, documentation and planning rules, model capability/master plans, related
  tasks, Tensor/Compile/Training APIs, glossary, Java 26 build configuration, actual shared-tree
  diff, final source/tests, generated Javadoc, test reports, bytecode, and imports. It applied
  General plus API/Javadoc style to Tensor/helper and the API references and General plus Planning
  style to task/master/roadmap. No complete example was added or changed, so Example format did
  not apply.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorBooleanLogicalTest` passed. The XML report contains
  9 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorTest` passed.
  The XML report contains 14 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test` passed. The 44 XML suites contain 331 tests with zero failures,
  errors, or skips.
- `./gradlew :modules:model:javadoc` passed without warnings. Generated `Tensor.html` contains one
  detail section for each logical method, including exact BOOL eligibility, binary broadcasting or
  unary exact-shape retention, fixed result facts, ordered provenance, failures, fresh identity,
  storage absence, and deferred execution. Package-private helper Javadoc was reviewed in source
  because public Javadoc generation does not publish package-private types.
- `./gradlew test` passed for the repository with 36 actionable tasks and no failing task.
  Architecture, backend-conformance, and integration modules have no applicable behavior tests for
  this model-only expression-construction change.
- `javap -p -c -s` confirmed exactly one `(Tensor)Tensor` `logicalAnd`, one `(Tensor)Tensor`
  `logicalOr`, and one zero-argument `logicalNot`; each loads the exact kind and invokes its exact
  helper entry once. The helper is final and package-private, has zero fields/nested types, one
  private no-argument constructor, two package-private static entries, and one private static
  constructor. Bytecode confirms ordered null/kind/BOOL validation, exact messages, one binary
  `ShapeBroadcast.broadcast` call, no unary broadcast call, exact unary Shape retention, one fixed
  BOOL/empty-layout/false-gradient descriptor, one Operation, one ordered provenance value, and one
  `TensorFactory.createDerived` call. No logical method is synchronized.
- Reflection tests confirm the exact helper/public API shapes and absence of overloads. Source,
  import, and bytecode review found no value/storage access, numeric truthiness, cast, resolved
  layout, gradient propagation or rule, graph capture, compiler/planning/prepare/runtime/backend
  type, cost, fusion, route, registry, service, dependency, or build behavior.
- The local Markdown validator resolved every file target and heading anchor across all six changed
  documentation/planning paths. Markdown fences are balanced, terminology distinguishes current
  model expression construction from planned compiler/execution behavior, and targeted scans found
  no trailing whitespace in any authorized path. `git diff --check` passed.
- Final scope contains exactly the ten authorized paths: two production files, two test files,
  Tensor API, Compile API, glossary, this task, model master plan, and roadmap. Task 0015D is
  synchronized as Complete in all three planning locations. Task 0015E remains Draft, and no
  task-0015E specification exists.
- The documentation pass found the Tensor type and all three public method Javadocs complete. It
  clarified only the helper's derived-construction wording and internal null-failure contract; no
  executable code or test changed. Tensor API now documents current BOOL-only logical expression
  construction and Compile API inventories it as current public expression input without claiming
  compiler capture, inference, optimization, artifacts, or execution.
- Existing glossary terms were sufficient. Implementation status, `OperationKind`, Provenance,
  Tensor, and common distinctions now include logical expression construction; no new reusable
  domain term was introduced.
- Training API remains accurate unchanged because fixed false gradient eligibility adds no gradient
  object, rule, autograd, optimizer, training-root policy, session, or training-graph treatment.
  Model capabilities remain accurate unchanged because they already select all three public
  logical names, exact BOOL inputs, result type, shape behavior, and layer separation.
- `ARCHITECTURE.md`, focused architecture pages, and ADRs remain accurate unchanged because the
  implementation stays within model-owned Tensor semantics, descriptors, and provenance and adds
  no module boundary, dependency, lifecycle, backend, runtime, or training decision. Architecture
  tests need no update because dependency rules did not change. Backend-conformance and integration
  tests need no update because no truth-value backend behavior or end-to-end execution exists yet.
- Java 26/Gradle configuration remains accurate unchanged because the implementation uses existing
  stable language/library contracts and adds no dependency, preview/incubator feature, or build
  change. `DataType`, `Shape`, `ShapeBroadcast`, `TensorDescriptor`, `TensorFactory.createDerived`,
  `TensorProvenance`, `Operation`, `NoOperationAttrs`, `BooleanLogicalKind`, and existing arithmetic,
  comparison, unary, and scalar expression contracts remain accurate because task 0015D composes
  them without changing their behavior.

## Implementation notes

- Added exactly three public Tensor logical methods and one package-private stateless helper with
  explicit binary/unary entries and one common fixed-result constructor.
- Every valid operation creates a fresh, unlabeled, storage-free unresolved BOOL result with false
  gradient eligibility and exact parameterless operation provenance after deterministic local
  validation. Binary methods broadcast and preserve ordered inputs; unary NOT retains exact shape.
- The independent documentation pass finalized helper Javadoc, Tensor API, Compile API, glossary,
  and planning evidence/status while preserving executable behavior and tests unchanged.
- No compiler, training, architecture, dependency, truth-value execution, or backend behavior was
  added.

## Completion summary

- Completed changes: Implemented and documented the three BOOL-only public logical Tensor
  expressions with binary broadcasting, unary exact-shape retention, fixed non-differentiable BOOL
  results, fresh identity, and exact provenance.
- Files changed or created: Exactly the authorized two production files, two tests, Tensor API,
  Compile API, glossary, this task, model master plan, and roadmap.
- Tests and validation: Focused logical tests 9/9, focused Tensor tests 14/14, all 331 model tests
  across 44 suites, model Javadoc, root tests, bytecode/reflection/import/absence checks, generated
  documentation review, local links/anchors, fences, terminology, whitespace, exact scope/status,
  and `git diff --check` passed.
- Documentation-agent review: Clean documentation context
  `/root/implement_model_0015d/review_model_0015d_docs` completed the independent pass using
  General, API/Javadoc, and Planning profiles; Example format was unnecessary because no complete
  example changed.
- Documentation impact: Tensor API and Compile API now describe current logical expression
  construction while preserving compiler and execution lifecycle behavior as planned. Training
  API, capabilities, architecture/ADRs/tests, conformance/integration tests, build configuration,
  and related foundational/expression documentation remain accurate unchanged for the reasons
  recorded above.
- Javadoc review: Tensor type and public logical-method Javadocs were complete; helper Javadoc now
  precisely distinguishes empty label from storage absence and documents downstream null failures.
- Glossary impact: Existing implementation-status, `OperationKind`, Provenance, Tensor, and common-
  distinction text now includes logical construction; no new reusable term was needed.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0015D. Task 0015E remains Draft without a specification.

Status: Complete

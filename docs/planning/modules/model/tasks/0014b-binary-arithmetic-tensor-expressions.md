# Task 0014B: Binary Arithmetic Tensor Expressions

## Status

Complete

## Goal

Expose the seven implemented binary arithmetic semantics as public, backend-independent Tensor
expression methods. Each successful call must validate floating data types and locally provable
right-aligned broadcasting, derive one immutable result descriptor, and return a fresh storage-free
Tensor whose provenance records the exact operation and ordered input Tensor identities.

This task makes binary arithmetic capturable through the public model API. It does not calculate
numeric values, allocate result storage, capture a compiled graph, define gradient rules, or report
backend support.

## Scope

- Add public `Tensor` methods `add`, `sub`, `mul`, `div`, `min`, `max`, and tensor-valued `pow`.
- Add one package-private `TensorBinaryExpressions` helper that owns the shared local validation and
  derived-Tensor construction path.
- Accept only `BFLOAT16`, `FLOAT32`, and `FLOAT64` operands.
- Promote floating data types through the existing `DataTypePromotion.promoteFloating` contract.
- Compute the result shape through the existing `ShapeBroadcast.broadcast` contract.
- Create an unresolved-layout result `TensorDescriptor` whose gradient eligibility is the logical
  OR of the two input descriptors' `requiresGrad` values.
- Construct exactly one `Operation` from the selected `BinaryArithmeticKind` and
  `NoOperationAttrs.INSTANCE`.
- Construct exactly one `TensorProvenance` with ordered inputs `[left, right]`.
- Delegate final identity-bearing construction exactly once to `TensorFactory.createDerived` with
  no label and no storage.
- Update the exact Tensor public-API reflection test and add one focused binary-expression test.
- Finalize affected Javadocs, Tensor API, glossary, task evidence, model master plan, and roadmap
  through the required independent documentation pass during implementation.

## Out of scope

- eager numeric execution, constant folding, algebraic simplification, canonicalization, or
  returning an input Tensor for identities such as add-zero or multiply-one
- reading, copying, allocating, attaching, materializing, or validating host storage
- integral or boolean arithmetic, implicit cross-category conversion, explicit cast insertion, or
  promotion beyond the existing floating hierarchy
- scalar-number overloads, reverse arithmetic, in-place methods, operator aliases, static factory
  methods, variadic arithmetic, or an expression-builder public type
- output labels, caller-supplied labels, label expressions, symbols, serialization names, or
  diagnostic expression strings
- resolved output layouts, broadcast stride plans, zero-stride views, aliases, materialization
  policy, or layout preservation from either operand
- operation-family attributes, factories, registries, parsers, reflection discovery, or changes to
  `BinaryArithmeticKind`, `Operation`, `OperationKind`, `OperationAttrs`, or `NoOperationAttrs`
- graph traversal, cycle checks, node/value IDs, graph capture, compiled graph construction,
  common-subexpression elimination, compiler inference, or publication binding
- gradient values, gradient rules, backward graph generation, tie policy for min/max, pow-domain
  rules, autograd execution, or training behavior
- numerical rules for division by zero, NaN, infinity, signed zero, min/max ties, or power edge
  cases; those require executable backend and conformance work
- planning ownership, capability providers, backend support, fusion, cost, lowering, kernels,
  runtime residency, prepare, execution, traces, or engine behavior
- dependencies, Gradle changes, architecture changes, another module, another operation family, or
  a detailed task-0014C specification

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
- [Task 0014A](0014a-binary-arithmetic-semantic-kinds.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes `Tensor.add(Tensor)`, `sub(Tensor)`,
`mul(Tensor)`, `div(Tensor)`, `min(Tensor)`, `max(Tensor)`, and `pow(Tensor)`. Its public contract
accepts floating operands, right-aligns broadcast shapes, promotes `BFLOAT16 < FLOAT32 < FLOAT64`,
and preserves left/right operand order. Legacy tests cover scalar, rank-mismatched, multi-axis, and
incompatible broadcasting, the three floating types, expression chaining, and ordered arithmetic.

The legacy builders also performed eager scalar-value inspection and algebraic rewrites, attached
mutable broadcast plans and gradient callbacks, built expression strings, and depended on runtime
and backend behavior. Those mechanisms are not copied. The new model retains only the selected
public capability and represents it through immutable descriptors and provenance. Optimization
belongs to the compiler, gradient expansion belongs to compiler autograd, and numerical execution
belongs to prepared backends.

## Architecture constraints

- `Tensor` remains public mutable API state and must not become an IR node.
- `TensorBinaryExpressions` performs deterministic local model validation only. It must not
  traverse provenance, capture a graph, evaluate data, or inspect backend capability.
- `Operation` owns only the backend-independent semantic kind and attributes. It contains no input
  Tensor references, shape result, backend support, or executable behavior.
- `TensorProvenance` owns the exact ordered `[left, right]` input identities. `SUB`, `DIV`, and
  `POW` therefore preserve their non-commutative operand roles.
- Result identity comes only from the existing package-private `TensorFactory.createDerived` seam.
  No second allocator, caller-supplied ID, registry, cache, or service is introduced.
- The result is storage-free. Public expression construction must not allocate physical buffers or
  attach either operand's storage.
- The result descriptor has `Optional.empty()` layout even for fully static shapes. A storage-free
  semantic expression does not assert resolved logical layout geometry, broadcast strides, or a
  future materialization route.
- Dynamic shapes are accepted only where `ShapeBroadcast` can prove compatibility locally: equal
  symbolic dimensions and static singleton expansion. The helper must not create constraints or
  defer a known local incompatibility.
- Floating promotion uses the existing model contract. Integral and boolean inputs are rejected;
  the helper must not insert casts or define a new promotion table.
- Result `requiresGrad` is `left.descriptor().requiresGrad() ||
  right.descriptor().requiresGrad()`. This propagates eligibility metadata only and does not promise
  a gradient rule, backward graph, or differentiable backend execution.
- Package direction is `model.tensor -> model.operation.elementwise.binary`, plus existing
  `model.tensor -> model.operation`, `model.datatype`, and `model.shape`. The operation package must
  not import Tensor and no package cycle may be introduced.
- Stop if implementation requires a changed existing foundational contract, resolved layout,
  storage access, graph capture, gradient rule, dependency, or architecture decision.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns public Tensor expression methods, local expression
  construction, result descriptors, provenance, and derived Tensor creation.
- `io.github.pho001.synaptik.model.operation` — supplies `Operation` and
  `NoOperationAttrs.INSTANCE`.
- `io.github.pho001.synaptik.model.operation.elementwise.binary` — supplies the seven exact
  `BinaryArithmeticKind` values.
- `io.github.pho001.synaptik.model.datatype` — supplies floating promotion.
- `io.github.pho001.synaptik.model.shape` — supplies local right-aligned broadcasting.

No package is added.

Type placement:

- `io.github.pho001.synaptik.model.tensor.Tensor` — public fluent expression surface; it receives
  only the seven one-argument methods and delegates shared behavior.
- `io.github.pho001.synaptik.model.tensor.TensorBinaryExpressions` — package-private, stateless
  construction boundary colocated with `Tensor`, `TensorDescriptor`, `TensorProvenance`, and the
  package-private factory seam it must use.
- `TensorBinaryArithmeticTest` — same-package focused test so it can verify the package-private
  helper without widening production visibility.

## Required contract

### Public Tensor surface

Add exactly these public methods to `Tensor`:

```java
public Tensor add(Tensor right)
public Tensor sub(Tensor right)
public Tensor mul(Tensor right)
public Tensor div(Tensor right)
public Tensor min(Tensor right)
public Tensor max(Tensor right)
public Tensor pow(Tensor right)
```

Each method delegates exactly once to `TensorBinaryExpressions.apply(this, right, <KIND>)` and
returns that exact result. It performs no separate validation, normalization, canonicalization,
allocation, provenance construction, or storage access. There are no overloads in this task.

The receiver is the ordered left operand. The argument is the ordered right operand, including the
subtrahend for `sub`, denominator for `div`, and exponent for `pow`.

### Package-private helper shape

Create exactly one package-private final non-record class:

```java
final class TensorBinaryExpressions {
    private TensorBinaryExpressions() {
    }

    static Tensor apply(Tensor left, Tensor right, BinaryArithmeticKind kind) {
        // exact construction contract below
    }
}
```

The helper has no fields, nested types, public/protected members, overloads, caches, registries, or
operation-specific branches. Its constructor prevents instantiation. `apply` is package-private and
static so `Tensor` can delegate without exposing an independent public expression service.

### Validation and construction order

`apply` performs these steps in exact order:

1. require non-null `left`, `right`, and `kind`, in that order, with messages `left`, `right`, and
   `kind`;
2. call `DataTypePromotion.promoteFloating(left.descriptor().dataType(),
   right.descriptor().dataType())` exactly once;
3. call `ShapeBroadcast.broadcast(left.descriptor().shape(), right.descriptor().shape())` exactly
   once;
4. create exactly one `TensorDescriptor` from the promoted type, broadcast shape,
   `Optional.empty()` layout, and the OR of input gradient-eligibility flags;
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
`createDerived` consume no Tensor ID. Do not add a production ID-inspection hook merely to test
that ordering.

### Result descriptor

For every successful expression:

- `dataType` is the exact result of the existing floating promotion table;
- `shape` is the exact immutable result of existing right-aligned broadcasting;
- `layout` is empty for both static and dynamic results; and
- `requiresGrad` is true exactly when either input descriptor requests gradient eligibility.

The descriptor records logical result facts only. It does not retain either input descriptor,
input layout, host storage, broadcast strides, or a materialization decision.

### Provenance and identity

The output is a fresh Tensor with:

- a new factory-assigned opaque `TensorId`;
- an empty label;
- no host storage;
- one exact `BinaryArithmeticKind` paired with `NoOperationAttrs.INSTANCE`; and
- ordered immutable provenance inputs containing exact references `[left, right]`.

Repeated inputs are valid. `tensor.add(tensor)` stores the same exact Tensor reference in both
ordered positions. Repeating an expression with equal inputs creates another Tensor identity; this
task performs no common-subexpression elimination or interning.

### No eager canonicalization

Every valid call creates the requested semantic expression, even if operand storage currently
contains values that would permit an algebraic rewrite. The helper must not read storage to detect
zero, one, minus one, equal operands, or constant exponents. It must not return an input Tensor or
replace one binary kind with a unary/scalar kind. Compiler optimization may make such decisions
later from immutable graph semantics under its own correctness rules.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorBinaryExpressions.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`

Documentation/planning during implementation:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a task-related inconsistency requires stopping:

- `docs/planning/modules/model/capabilities.md`
- Existing `DataType`, `DataTypePromotion`, `Shape`, `ShapeBroadcast`, `TensorDescriptor`,
  `TensorProvenance`, `TensorFactory`, `Operation`, `NoOperationAttrs`, and
  `BinaryArithmeticKind` Javadocs and tests.
- Training API, focused architecture documents, ADRs, and architecture tests.

## Maximum scope

At most two production files, two test files, and six documentation/planning files: ten paths
total. The tenth path, `docs/api/compile-api.md`, was explicitly authorized after the independent
documentation review found its stale public-Tensor implementation status.

`Tensor.java` and `TensorTest.java` may change only for the seven exact public methods, their
Javadocs, exact API-shape expectations, and non-synchronization assertions. Do not change existing
fields, constructor, metadata/storage behavior, equality, hashing, diagnostics, or unrelated tests.

If implementation needs another production/test concept, a changed foundational contract, a
resolved-layout policy, storage access, graph/compiler/autograd behavior, another documentation
file, or more than ten paths, stop and propose a follow-up or architecture decision. Do not create
task 0014C.

## Javadoc requirements

- Update Tensor type Javadoc only as needed to explain that public expression methods build
  storage-free provenance without making Tensor an IR node or executable value.
- Every new public method must explain ordered operand roles, floating-only promotion, broadcasting,
  unresolved result layout, gradient-eligibility propagation, fresh identity, storage absence,
  provenance, and deferral of numerical execution and gradient rules.
- Every method must document the non-null right operand with `@param`, the fresh derived Tensor with
  `@return`, and delegated null, data-type, shape, and identity-exhaustion failures with `@throws`.
- Document the package-private helper and its `apply` method with the exact validation/construction
  order, ownership, side effects, and failure behavior. The private constructor needs meaningful
  documentation.
- Avoid seven copies of unexplained terminology: each method remains complete, while links to the
  helper/type-level explanation may centralize shared details where generated Javadoc remains clear.
- Review related foundational Javadocs and record why they remain accurate or stop on an out-of-
  scope inconsistency.

## Acceptance criteria

- Tensor declares exactly the seven new public one-argument methods with parameter and return type
  `Tensor`; no overload or unrelated public API is added.
- Every method maps to the same-named exact `BinaryArithmeticKind` and delegates once to the shared
  package-private helper.
- `TensorBinaryExpressions` has exactly the specified visibility, finality, constructor, method,
  and zero-field surface.
- Null validation order and messages are exact, and public null operands fail as `right`.
- All seven methods accept every floating pair with compatible shapes, promote through the existing
  hierarchy, broadcast through the existing shape contract, and preserve ordered operands.
- Integral and boolean inputs are rejected without implicit casting, ID allocation, or storage
  access.
- Static scalar, zero-sized, rank-mismatched, singleton-expanded, and multi-axis broadcast results
  are represented correctly.
- Equal symbolic dimensions and symbolic/static-singleton pairs succeed; incompatible static or
  locally unprovable dynamic pairs fail through `ShapeBroadcast`.
- Every result descriptor has empty layout and `requiresGrad` equal to the input OR.
- Every result is a fresh, unlabeled, storage-free Tensor with exact kind,
  `NoOperationAttrs.INSTANCE`, and immutable ordered exact input references.
- Self-use preserves the same input reference twice; repeated calls are not interned or
  canonicalized.
- No input Tensor metadata, provenance, label, storage association, or storage contents are mutated
  or retained as output storage.
- No numerical execution, result values, gradient rules, graph state, backend facts, dependency,
  or architecture change is added.
- Focused and aggregate model tests, model Javadoc, root tests, reflection/javap/import/bytecode/
  scope checks, documentation links/formatting, and status synchronization pass.
- A separate documentation-focused agent finalizes Javadocs, Tensor API, glossary, task evidence,
  master plan, and roadmap in the same change and records reasoned no-change conclusions for
  related APIs, capabilities, and architecture.
- Task 0014B becomes Complete only after both passes. Task 0014C remains Draft without a detailed
  specification.

## Tests / validation

Run before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorBinaryArithmeticTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test must cover:

- exact helper class/constructor/method/field visibility and shape;
- all seven public method-to-kind mappings;
- exact Operation and singleton attributes references;
- ordered exact provenance inputs, including non-commutative order and repeated self-input;
- all floating type pairs and promotion results;
- scalar, zero-sized, different-rank, singleton, multi-axis, and dynamic broadcasting;
- empty layout, OR gradient eligibility, empty label, absent host storage, fresh identity, and no
  canonicalization;
- null helper/public operands, non-floating operands, incompatible static shapes, and unprovable
  dynamic shapes; and
- preservation of input metadata, provenance, labels, and storage associations.

Manually inspect `javap -p -c -s` and reflection for the exact Tensor method descriptors and helper
surface. Inspect method bytecode/source for one helper delegation per public method and exact local
construction order. Scan production imports and Gradle dependencies for forbidden layers. Confirm
that no storage accessor, segment API, graph ID, compiler/runtime/backend type, gradient rule,
canonicalization, cache, or registry appears. Validate generated Javadoc, Tensor API current-versus-
planned wording and newcomer example, glossary terminology, links/anchors/fences/whitespace, exact
ten-path scope, synchronized statuses, and absence of a task-0014C specification.

## Dependencies

- Task 0001 supplies floating data types and `DataTypePromotion`.
- Task 0002 supplies `Shape` and `ShapeBroadcast`.
- Task 0007 supplies immutable result descriptors.
- Task 0011 supplies the public Tensor surface.
- Task 0012 and task 0013 supply factory identity allocation, `createDerived`, and immutable
  provenance.
- Task 0014A supplies the exact seven binary arithmetic semantic kinds.

## Follow-up tasks

- Task 0014C remains Draft for unary arithmetic and activation semantic kinds while the
  post-0014B cross-module vertical-slice checkpoint is reconsidered.
- Task 0014D will later expose unary/activation Tensor expressions.
- Tasks 0014E and 0014F remain Draft for scalar/clamp semantics and expressions.
- Compiler capture, optimizer rewrites, autograd expansion, capability analysis, backend ownership,
  numerical kernels, and conformance tests remain in their owning module tasks.
- Reconsider the planned cross-module vertical-slice checkpoint after task 0014B completes the first
  public capturable operation family.

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None. The architecture already assigns public Tensor expression semantics,
backend-independent Operation values, descriptors, and provenance to `modules/model`, while
forbidding graph compilation, physical allocation, backend support, and execution there.

If implementation requires Tensor to become IR, a resolved physical layout, storage access,
compiler/autograd logic, backend facts, another module dependency, or a changed architecture rule,
stop and report the issue.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0002/0003/0006/0007/0011/0012/0013/0014A/0014B,
Tensor API, Compile API, Training API, glossary, current DataTypePromotion/ShapeBroadcast/
TensorDescriptor/Tensor/TensorFactory/TensorProvenance/Operation/BinaryArithmeticKind contracts and
tests, and Java 26 Gradle configuration.

Implement task 0014B exactly. Modify Tensor.java and add package-private final
TensorBinaryExpressions.java for production. Update TensorTest only for the exact seven-method API
surface and add TensorBinaryArithmeticTest. Add exactly add/sub/mul/div/min/max/pow(Tensor), each
delegating once to the shared helper and exact matching kind.

The helper must null-check left/right/kind, promote only floating data types with
DataTypePromotion, broadcast with ShapeBroadcast, create an empty-layout descriptor with OR
requiresGrad, create Operation(kind, NoOperationAttrs.INSTANCE), preserve exact ordered provenance
[left, right], and delegate once to TensorFactory.createDerived with no label/storage. Every valid
call returns a fresh derived Tensor. Do not inspect storage, execute arithmetic, canonicalize,
resolve layout, insert casts, add gradient rules, capture a graph, add overloads, change existing
contracts, or introduce compiler/runtime/backend behavior.

Stop beyond ten paths or on architecture uncertainty; the tenth path is limited to correcting
Compile API implementation-status wording. Run every specified focused/aggregate test, Javadoc,
javap/reflection/bytecode/import/manual, documentation/link/whitespace/scope/status check. Then hand
the actual diff and evidence to a separate clean-context documentation agent in the same change.
It must inspect source/tests/generated Javadoc, finalize permitted Javadocs/Tensor API/Compile API/
glossary/planning, record related-contract/capability/Training API/architecture no-change
conclusions, and rerun validation.

Update task 0014B, model master plan, and roadmap only for planning status/evidence. Do not mark
0014B Complete until both passes succeed. Leave 0014C Draft without a specification. Do not commit
or push.
```

## Local decisions

- Public methods remain on Tensor to preserve the selected fluent capability. Shared construction
  lives in one package-private helper so Tensor does not duplicate seven validation/provenance paths
  and no new public service is introduced.
- All seven operations use floating promotion. Existing integral data types do not imply integral
  arithmetic support, and BOOL has no numeric truthiness.
- Result layout remains unresolved for static and dynamic shapes because expression construction
  proves logical shape, not resolved layout geometry or a materialization route.
- Gradient eligibility propagates as input OR, matching immutable Tensor metadata while deferring
  actual gradient semantics to compiler autograd.
- Valid identity expressions are not canonicalized. Returning an operand or inspecting eager values
  would conflate public expression construction with compiler optimization and storage state.
- Output labels are empty. Stable semantic meaning lives in typed Operation/provenance rather than
  generated expression strings.
- One generic package-private helper is sufficient because all seven kinds share identical arity,
  parameterlessness, promotion, broadcasting, descriptor, and provenance rules.

## Known limitations

- The returned Tensor has no computed values or host storage and cannot execute without future
  compiler, prepare, runtime, and backend work.
- Only floating binary arithmetic is supported. Integral arithmetic and explicit cast composition
  remain future planned capabilities.
- Dynamic broadcasting accepts only relationships provable by the current local shape contract; it
  does not create or solve symbolic constraints.
- The output layout is unresolved even when inputs and result shape are static.
- No gradient rule is attached. `requiresGrad` is eligibility metadata only.
- No numerical edge behavior or backend support is promised by expression construction.
- Repeated equivalent expressions receive distinct Tensor identities until a future compiler
  chooses to deduplicate immutable graph occurrences where valid.

## Validation evidence

Planning reviewed the current architecture and documentation rules; model capabilities, master plan,
and roadmap; completed tasks 0001, 0002, 0003, 0006, 0007, 0011, 0012, 0013, and 0014A; current
DataTypePromotion, ShapeBroadcast, TensorDescriptor, Tensor, TensorFactory.createDerived,
TensorProvenance, Operation, NoOperationAttrs, and BinaryArithmeticKind source/tests; and the
read-only legacy Tensor binary API, builders, dtype utilities, broadcasting tests, dtype execution
coverage, and canonicalization tests.

Planning confirmed that the current contracts can implement the task without a dependency, Gradle,
architecture, or foundational API change. Legacy capability selection retains seven floating,
broadcast-aware, promoted public methods while explicitly rejecting legacy storage inspection,
eager rewrites, mutable broadcast plans, gradient callbacks, expression strings, and execution
coupling. The planned package direction remains acyclic.

Planning validation:

- `git diff --check` passed for the tracked planning changes; a trailing-whitespace scan also passed
  for this new task file.
- The canonical task-section scan found every required planning section.
- The local Markdown-target check resolved every link in this task, the model master plan, and the
  roadmap.
- Scope inspection found exactly three documentation/planning paths, no Java, Gradle, AGENTS, or
  architecture change, and no task-0014C specification.
- At the planning checkpoint, task status was `Ready` in this specification, the model master
  plan, and the roadmap; task 0014C remained `Draft`.

Implementation and documentation validation:

- The first focused-test invocation failed during test compilation because the updated Tensor
  reflection test did not yet declare the checked `NoSuchMethodException`; adding
  `throws ReflectiveOperationException` corrected the exact API-shape test. The next focused run
  executed 9 tests with one failure because it compared two equal reflection `Method` wrappers by
  object identity; replacing that assertion with value equality corrected the test without
  changing production behavior. Every subsequent focused and aggregate run passed as recorded
  below.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorBinaryArithmeticTest` passed after the documentation
  pass; its XML reports 9 tests, 0 failures, 0 errors, and 0 skipped.
- `./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorTest` passed;
  its XML reports 14 tests, 0 failures, 0 errors, and 0 skipped.
- `./gradlew :modules:model:test` passed; the 36 XML suites report 273 tests, 0 failures, 0 errors,
  and 0 skipped in aggregate.
- `./gradlew :modules:model:javadoc` passed. Generated `Tensor.html` contains all seven public
  methods, exact ordered roles, floating/local-broadcast contract, unresolved layout, OR gradient
  eligibility, fresh unlabeled storage-free results, exact kinds with
  `NoOperationAttrs.INSTANCE`, ordered provenance, and deferred execution/gradient rules. Standard
  generated Javadoc omits the package-private helper, whose type, constructor, and method Javadocs
  were reviewed directly in source.
- `./gradlew test` passed. The final post-edit rerun reported all 36 actionable tasks up-to-date;
  the earlier validating run executed one task and found 35 up-to-date.
- The complete `BinaryExpressionExample` from the Tensor API compiled in Java 26 JShell against
  `modules/model/build/libs/model-0.1.0-SNAPSHOT.jar` and printed the documented `FLOAT32`,
  `Shape[2, 3]`, unresolved-layout, gradient, label, storage, `DIV`, canonical-attributes,
  ordered-input, and fresh-result values exactly. JShell later reported a macOS preferences
  history-flush exception while exiting; compilation and program output had already completed and
  the exception does not affect the example result or repository validation.
- `javap -p -c -s` confirmed the seven exact `(Tensor)Tensor` descriptors, one matching enum
  constant and one helper invocation per public method, helper zero-field/private-constructor/
  single-package-private-static-method shape, exact left/right/kind null-check order, one floating
  promotion, one broadcast, one descriptor, one operation, one provenance, and one derived-factory
  call in order. The methods and helper are not synchronized.
- Source/import/dependency scans found only the permitted local model and JDK imports, no project
  dependency in `modules/model/build.gradle.kts`, and no storage access, execution,
  canonicalization, cast insertion, gradient rule, graph identity/capture, cache, registry,
  compiler/planning/runtime/backend type, or route selection in the helper.
- The six documentation/planning files have 147 resolving local Markdown file/anchor links,
  balanced fences (`66/0`, `4/0`, `0/0`, `8/0`, `2/0`, and `0/0` backtick/tilde fences
  respectively), and no trailing whitespace. `git diff --check` passed.
- Final scope inspection found exactly the authorized ten changed paths: two production files,
  two test files, and six documentation/planning files. No Gradle, dependency, architecture, ADR,
  architecture-test, backend-conformance, integration-test, or task-0014C specification path
  changed.
- Status synchronization marks task 0014B Complete in this specification, the model master plan,
  and the roadmap. Task 0014C remains Draft without a detailed specification.

The clean-context documentation review was performed in
`/root/implement_model_0014b/review_model_0014b_docs` using the General, API/Javadoc, and Planning
profiles plus the Example format. It independently reviewed the full diff, actual source and
tests, generated Javadoc, bytecode, XML reports, public reference, glossary, and planning state.

Adjacent-contract and documentation conclusions:

- `DataType` and `DataTypePromotion` remain accurate: they already define differentiability and
  the exact floating-only hierarchy and failures used here. `Shape` and `ShapeBroadcast` already
  define scalar/zero/dynamic shapes and conservative right-aligned local broadcasting. No source,
  Javadoc, or focused-test change is required for those reused contracts.
- `TensorDescriptor` already permits unresolved static/dynamic layouts and distinguishes
  `requiresGrad` eligibility from a gradient rule. `TensorProvenance` already preserves immutable
  ordered exact Tensor references. `TensorFactory.createDerived` already owns one-ID storage-free
  construction. Their Javadocs and tests remain accurate unchanged.
- `Operation`, `NoOperationAttrs`, and `BinaryArithmeticKind` already document exact reference
  retention, canonical parameterless attributes, ordered operand meanings, and the absence of
  execution/backend behavior. Their Javadocs and tests remain accurate unchanged.
- `capabilities.md` already selects the seven broadcast-aware binary capabilities and clearly
  separates model/public-expression support from compiler/backend/runtime support, so it requires
  no implementation-status edit.
- The Training API requires no change because OR propagation is descriptor eligibility only; this
  task adds no gradient object, rule, autograd, optimizer, or training behavior.
- `ARCHITECTURE.md`, focused architecture pages, ADRs, and architecture tests require no change:
  the implementation stays within the existing model ownership and adds no module/dependency or
  lifecycle decision. Backend-conformance and integration tests require no change because no
  numerical backend or end-to-end execution behavior exists. Gradle and dependencies are
  unchanged because the helper composes only same-module and JDK contracts.
- The Compile API required a task-related status correction. Explicit user authorization expanded
  the task to ten paths solely for `docs/api/compile-api.md`. The page now describes public Tensor
  state and seven binary expression builders as current, while keeping compiler entry, traversal,
  capture, inference, optimization, `CompileArtifacts`, and the engine facade explicitly planned.
  It makes no compiler/runtime/backend execution claim.

## Implementation notes

- Added exactly seven public one-argument Tensor methods, all delegating once to the matching kind
  through one stateless package-private helper.
- The helper implements the specified null, promotion, broadcast, descriptor, operation,
  provenance, and derived-construction order without inspecting or mutating input storage.
- The focused implementation test covers exact API/helper shape, all kind mappings and floating
  pairs, static/dynamic broadcasts, descriptor/provenance/identity facts, failure ordering, and
  input preservation. Tensor's existing reflection test changed only for the new exact public
  surface and non-synchronization assertions.
- The independent documentation pass finalized Tensor/helper Javadocs, moved binary expressions
  from planned to current in the Tensor API, added and executed a newcomer-complete example, and
  updated existing glossary distinctions without introducing a new term.
- After explicit scope expansion, the same pass corrected only Compile API implementation-status
  wording and preserved every compiler and engine API as conceptual and planned.
- No executable Java logic or tests were changed during the documentation pass.

## Completion summary

- Completed changes: The implementation, focused tests, public Javadocs, Tensor API, glossary, and
  all authorized API/task/master-plan/roadmap updates are complete and validated within the exact
  ten-path scope.
- Files changed or created: `Tensor.java`, `TensorBinaryExpressions.java`, `TensorTest.java`,
  `TensorBinaryArithmeticTest.java`, Tensor API, Compile API, glossary, this task, model master
  plan, and roadmap.
- Tests and validation: Focused 9/9 and 14/14 suites, all 273 model tests across 36 suites, model
  Javadoc, root tests, bytecode/reflection/import/dependency/absence checks, exact runnable example,
  147 local links/anchors, fences, whitespace, exact scope, and `git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0014b/review_model_0014b_docs` completed the independent pass using
  General, API/Javadoc, Planning, and Example-format guidance.
- Documentation impact: Binary Tensor expressions are current in the Tensor API and glossary;
  the Compile API now distinguishes those current expressions from planned compiler capture.
  Gradient rules, numerical execution, and backend support remain unclaimed.
- Javadoc review: Tensor type/all seven methods and the package-private helper/type/constructor are
  finalized. Reused data type, shape, descriptor, provenance, factory, and operation-family
  contracts remain accurate unchanged.
- Glossary impact: Existing implementation-status, Tensor, Provenance, OperationKind, and common
  distinctions now describe public binary expression construction. No new reusable term was
  needed.
- Unresolved issues: None.
- Architecture impact: None.
- Task 0014C: Remains Draft without a detailed specification.

Status: Complete

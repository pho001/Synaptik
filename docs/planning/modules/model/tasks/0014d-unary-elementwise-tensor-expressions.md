# Task 0014D: Unary Elementwise Tensor Expressions

## Status

Complete

## Goal

Expose all fifteen implemented unary elementwise semantics as public, backend-independent Tensor
expression methods. Each successful call must validate the floating input type, preserve the
logical data type and shape, derive one immutable unresolved-layout descriptor, and return a fresh
storage-free Tensor whose provenance records the exact operation and exact input Tensor identity.

This task makes unary arithmetic, transcendental, activation, and explicit fast-approximation
requests capturable through the public model API. It does not calculate numeric values, inspect
mathematical domains, allocate result storage, capture a compiled graph, define gradient rules, or
report backend support.

## Scope

- Add public zero-argument `Tensor` methods `abs`, `neg`, `inv`, `log`, `exp`, `erf`, `sqrt`,
  `floor`, `ceil`, `sign`, `relu`, `sigmoid`, `tanh`, `fastExp`, and `fastTanh`.
- Add one package-private `TensorUnaryExpressions` helper that owns the shared local validation and
  derived-Tensor construction path.
- Accept only `BFLOAT16`, `FLOAT32`, and `FLOAT64` inputs.
- Preserve the input descriptor's exact data type and exact immutable shape reference.
- Create an unresolved-layout result `TensorDescriptor` with the input descriptor's exact
  `requiresGrad` value.
- Construct exactly one `Operation` from the selected `UnaryElementwiseKind` and
  `NoOperationAttrs.INSTANCE`.
- Construct exactly one `TensorProvenance` with the one exact input reference.
- Delegate final identity-bearing construction exactly once to `TensorFactory.createDerived` with
  no label and no storage.
- Update the exact Tensor public-API reflection test and add one focused unary-expression test.
- Finalize affected Javadocs, Tensor API, the authorized Compile API status correction, glossary,
  task evidence, model master plan, and roadmap through the required independent documentation
  pass during implementation.

## Out of scope

- eager numeric execution, constant folding, algebraic simplification, canonicalization, or
  returning an existing Tensor for identities such as double negation or double reciprocal
- reading, copying, allocating, attaching, materializing, or validating host storage
- integral or boolean unary arithmetic, implicit conversion, explicit cast insertion, or a new
  data-type promotion contract
- scalar arguments, overloads, in-place methods, static operation factories, aliases, generic
  `apply`, or an expression-builder public type
- output labels, caller-supplied labels, symbols, serialization names, or diagnostic expression
  strings
- resolved output layouts, input-layout preservation, view/alias behavior, materialization policy,
  or physical stride derivation
- operation-family attributes, factories, registries, parsers, reflection discovery, or changes to
  `UnaryElementwiseKind`, `Operation`, `OperationKind`, `OperationAttrs`, or `NoOperationAttrs`
- mathematical domain or range validation, overflow, underflow, rounding, signed-zero, NaN,
  infinity, exactness, strict-function accuracy, or fast-approximation accuracy and algorithms
- graph traversal, cycle checks, graph IDs, graph capture, compiled graph construction,
  common-subexpression elimination, compiler inference, or publication binding
- gradient values, gradient rules, subgradient choices, nondifferentiable-point policy, backward
  graph generation, autograd execution, or training behavior
- scalar multiplication, scalar power, `clamp`, `clampMin`, or `clampMax`; tasks 0014E–0014F own
  their attributes and expression surface
- capability providers, backend support, fusion, cost, lowering, kernels, runtime residency,
  prepare, execution, traces, or engine behavior
- dependencies, Gradle changes, architecture changes, another module, another operation family, or
  a detailed task-0014E specification

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
- [Task 0014C](0014c-unary-elementwise-semantic-kinds.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes all fifteen selected unary capabilities through
its public Tensor/TensorOps surface: `abs`, `neg`, `inv`, `log`, `exp`, `erf`, `sqrt`, `floor`,
`ceil`, `sign`, `relu`, `sigmoid`, `tanh`, `fastExp`, and `fastTanh`. The legacy builders accept
floating values and create shape-preserving unary operations. Legacy tests additionally exercise
expression chaining, gradients, storage layouts, kernel execution, fusion, approximation paths,
and selected eager canonicalization.

Only the useful public operation capability, floating eligibility, shape preservation, and exact
semantic identity are retained here. The legacy implementation's mutable node state, gradient
callbacks, eager storage coupling, expression labels, kernel/fusion details, and canonicalization
are not copied. Compiler optimization owns semantic rewrites, compiler autograd owns gradient
expansion, and prepared backends own numerical execution and fast approximation algorithms.

Legacy `floor`, `ceil`, and `sign` forcibly disabled gradient state. The new model does not copy
that coupling into expression construction: `requiresGrad` is model-level eligibility metadata,
not proof that an operation has a gradient rule. All fifteen methods therefore preserve the input
request uniformly, while a later compiler/autograd contract decides whether and how each operation
contributes to a backward graph.

## Architecture constraints

- `Tensor` remains public mutable API state and must not become an IR node.
- `TensorUnaryExpressions` performs deterministic local model validation only. It must not inspect
  values, traverse provenance, capture a graph, evaluate mathematics, or inspect backend
  capability.
- `Operation` owns only the backend-independent semantic kind and attributes. It contains no input
  Tensor reference, result descriptor, backend support, or executable behavior.
- `TensorProvenance` owns the one exact input identity. It must contain exactly `List.of(input)` in
  that order and must not substitute an ancestor or equal Tensor.
- Result identity comes only from `TensorFactory.createDerived`. No second allocator,
  caller-supplied ID, registry, cache, interning table, or service is introduced.
- The result is storage-free. Public expression construction must not allocate a physical buffer
  or attach/alias the input's storage.
- The result descriptor has `Optional.empty()` layout even when the input has a resolved layout
  and fully static shape. A semantic expression does not assert a future materialization route.
- Unary elementwise shape preservation retains the exact immutable input `Shape` reference. No
  shape algebra, symbolic binding, or new constraint is required.
- The exact input data type is retained after floating validation. Integral and boolean inputs are
  rejected; the helper must not convert, promote, or insert a cast.
- Result `requiresGrad` equals the input descriptor's flag for every kind, including `FLOOR`,
  `CEIL`, and `SIGN`. This propagates eligibility metadata only and does not promise a derivative,
  gradient rule, or differentiable backend execution.
- `FAST_EXP` and `FAST_TANH` remain exact distinct semantic requests. The helper must not alias them
  to strict variants or choose an approximation implementation.
- Package direction is `model.tensor -> model.operation.elementwise.unary`, plus existing
  `model.tensor -> model.operation` and `model.datatype`. The operation package must not import
  Tensor and no package cycle may be introduced.
- Stop if implementation requires a changed foundational contract, resolved layout, storage/value
  access, graph capture, gradient policy, dependency, or architecture decision.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns public Tensor expression methods, local result
  descriptor construction, provenance, and derived Tensor creation.
- `io.github.pho001.synaptik.model.operation` — supplies `Operation` and
  `NoOperationAttrs.INSTANCE`.
- `io.github.pho001.synaptik.model.operation.elementwise.unary` — supplies the fifteen exact
  `UnaryElementwiseKind` values.
- `io.github.pho001.synaptik.model.datatype` — supplies the existing immutable `DataType` category
  query used for floating validation.

No package is added.

Type placement:

- `io.github.pho001.synaptik.model.tensor.Tensor` — public fluent expression surface; it receives
  only the fifteen zero-argument methods and delegates shared behavior.
- `io.github.pho001.synaptik.model.tensor.TensorUnaryExpressions` — package-private, stateless
  construction boundary colocated with `Tensor`, `TensorDescriptor`, `TensorProvenance`, and the
  package-private factory seam it must use.
- `TensorUnaryElementwiseTest` — same-package focused test so it can verify the package-private
  helper without widening production visibility.

## Required contract

### Public Tensor surface

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
exact result. It performs no separate validation, descriptor or provenance construction,
canonicalization, allocation, storage access, or exception translation. There are no overloads in
this task.

Method-to-kind mapping is exact and name preserving:

| Tensor method | Kind |
|---|---|
| `abs()` | `ABS` |
| `neg()` | `NEG` |
| `inv()` | `INV` |
| `log()` | `LOG` |
| `exp()` | `EXP` |
| `erf()` | `ERF` |
| `sqrt()` | `SQRT` |
| `floor()` | `FLOOR` |
| `ceil()` | `CEIL` |
| `sign()` | `SIGN` |
| `relu()` | `RELU` |
| `sigmoid()` | `SIGMOID` |
| `tanh()` | `TANH` |
| `fastExp()` | `FAST_EXP` |
| `fastTanh()` | `FAST_TANH` |

### Package-private helper shape

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
static so Tensor can delegate without exposing an independent public expression service.

### Validation and construction order

`apply` performs these steps in exact order:

1. require non-null `input` and `kind`, in that order, with messages `input` and `kind`;
2. read the input descriptor's exact `DataType` and reject it when `isFloating()` is false with
   `IllegalArgumentException("input must be a floating data type, but was " + dataType)`;
3. create exactly one `TensorDescriptor` from that same data type, the exact input shape,
   `Optional.empty()` layout, and the exact input `requiresGrad` flag;
4. create exactly one `Operation(kind, NoOperationAttrs.INSTANCE)`;
5. create exactly one `TensorProvenance(operation, List.of(input))`;
6. call `TensorFactory.createDerived(descriptor, Optional.empty(), provenance)` exactly once and
   return its exact result.

Null and data-type validation complete before ID allocation. Do not catch, translate, aggregate,
or replace failures. Validation failures before `createDerived` consume no Tensor ID. Exhausted
Tensor identity space fails through `createDerived` only after the local immutable model values
have been constructed. Do not add a production ID-inspection hook merely to test ordering.

### Result descriptor

For every successful expression:

- `dataType` is the exact enum value from the input descriptor;
- `shape` is the exact immutable `Shape` reference from the input descriptor;
- `layout` is empty for static, zero-sized, scalar, and dynamic shapes; and
- `requiresGrad` equals the input descriptor's exact value.

The result descriptor is a new immutable value. It does not retain the input descriptor itself,
input layout, host storage, physical geometry, or a materialization decision.

### Provenance and identity

The output is a fresh Tensor with:

- a new factory-assigned opaque `TensorId`;
- an empty label;
- no host storage;
- one exact `UnaryElementwiseKind` paired with `NoOperationAttrs.INSTANCE`; and
- immutable provenance containing exactly the one exact input reference.

Repeating an expression creates another Tensor identity. Chaining retains the immediately
preceding derived Tensor as the next expression's exact input. This task performs no interning,
common-subexpression elimination, ancestor substitution, or graph traversal.

### No eager domain checks or canonicalization

Every valid floating Tensor creates the requested semantic expression regardless of current
storage, values, or provenance. For example, `log` and `sqrt` do not inspect whether values are in a
real-valued domain; `inv` does not inspect zero; and `inv().inv()` does not return the original
Tensor. Strict and fast variants stay distinct. Compiler optimization and backend numerical
contracts may later make valid decisions using immutable graph semantics and selected execution
capabilities.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorUnaryExpressions.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorUnaryElementwiseTest.java`

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
- Existing `DataType`, `Shape`, `TensorDescriptor`, `TensorProvenance`, `TensorFactory`,
  `Operation`, `NoOperationAttrs`, and `UnaryElementwiseKind` Javadocs and tests.
- Focused architecture documents, ADRs, architecture tests, backend-conformance tests, and
  integration tests.

## Maximum scope

At most two production files, two test files, and six documentation/planning files: ten paths
total. The tenth path, `docs/api/compile-api.md`, was explicitly authorized after the independent
documentation review found its public-Tensor expression status stale.

`Tensor.java` and `TensorTest.java` may change only for the fifteen exact public methods, their
Javadocs, exact API-shape expectations, and non-synchronization assertions. Do not change existing
fields, constructor, metadata/storage behavior, binary expression methods, equality, hashing,
diagnostics, or unrelated tests.

If implementation needs another production/test concept, a changed foundational contract, a
resolved-layout or gradient policy, storage/value access, graph/compiler behavior, another
documentation file, or more than ten paths, stop and propose a follow-up or architecture decision.
The Compile API edit is limited to current-status wording and must not introduce compiler behavior.
Do not create task 0014E.

## Javadoc requirements

- Update Tensor type Javadoc only as needed to explain that public binary and unary expression
  methods build storage-free provenance without making Tensor an IR node or executable value.
- Every new public method must explain its elementwise mathematical or activation meaning,
  floating-only eligibility, exact type and shape preservation, unresolved result layout, exact
  gradient-eligibility propagation, fresh identity, storage absence, exact semantic kind,
  one-input provenance, and deferral of numerical/domain/gradient/backend behavior.
- Every public method must document the fresh derived Tensor with `@return`, the non-floating input
  failure with `@throws IllegalArgumentException`, and identity exhaustion with
  `@throws IllegalStateException`. Zero-argument instance methods must not invent an `@param` tag.
- The strict and fast method Javadocs must state that the kinds are distinct requests and must not
  promise a specific approximation, accuracy, or backend implementation.
- Document the package-private helper and its `apply` method with exact validation/construction
  order, reference retention, ownership, ID side effects, and failure behavior. The private
  constructor needs meaningful documentation.
- Avoid fifteen copies of unexplained terminology: each method remains complete, while type-level
  explanation and links may centralize shared mechanics where generated Javadoc stays clear.
- Review related foundational Javadocs and record why they remain accurate, or stop on an
  out-of-scope inconsistency.

## Acceptance criteria

- Tensor declares exactly the fifteen new public zero-argument methods returning Tensor; no
  overload or unrelated public API is added.
- Every method maps to the exact specified `UnaryElementwiseKind` and delegates once to the shared
  package-private helper.
- `TensorUnaryExpressions` has exactly the specified visibility, finality, constructor, method,
  and zero-field surface.
- Helper null validation order and messages are exact.
- All methods accept `BFLOAT16`, `FLOAT32`, and `FLOAT64`, retain the exact input data type, and
  reject `INT32`, `INT64`, and `BOOL` with the exact message before identity allocation.
- Scalar, zero-sized, ordinary static, and dynamic shapes are accepted and retain the exact input
  shape reference.
- Every result descriptor has empty layout and `requiresGrad` equal to the input flag for all
  fifteen kinds.
- Every result is a fresh, unlabeled, storage-free Tensor with exact kind,
  `NoOperationAttrs.INSTANCE`, and exactly one immutable exact input reference.
- Repeated calls are not interned; chains preserve the immediately previous Tensor; double
  reciprocal, double negation, nested activations, and strict/fast requests are not canonicalized.
- No method reads numeric values or rejects a represented mathematical domain before execution.
- No input metadata, provenance, label, storage association, or storage contents are mutated or
  retained as output storage.
- No numerical execution, result values, gradient rules, graph state, backend facts, dependency,
  or architecture change is added.
- Focused and aggregate model tests, model Javadoc, root tests, reflection/javap/import/bytecode/
  scope checks, documentation links/formatting, and status synchronization pass.
- A separate clean-context documentation-focused agent finalizes Javadocs, Tensor API, the
  authorized Compile API status correction, glossary, task evidence, master plan, and roadmap in
  the same change and records reasoned no-change conclusions for related APIs, capabilities,
  architecture, and existing contracts.
- Task 0014D becomes Complete only after both passes. Task 0014E remains Draft without a detailed
  specification.

## Tests / validation

Run before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorUnaryElementwiseTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test must cover:

- exact helper class, constructor, method, field, visibility, and finality shape;
- all fifteen public method-to-kind mappings and exact singleton attributes reference;
- exact one-input provenance reference for every method and across a chain;
- all three floating types with exact type and shape-reference preservation;
- scalar, zero-sized, static resolved/unresolved, and dynamic input shapes;
- empty result layout, exact gradient-eligibility propagation including `floor`, `ceil`, and
  `sign`, empty label, absent host storage, and fresh identity;
- helper null input/kind failures, every non-floating type, exact messages, and no ID consumption;
- absence of domain inspection and canonicalization, including repeated, nested, strict, and fast
  requests; and
- preservation of input metadata, provenance, labels, storage associations, and storage contents.

Manually inspect `javap -p -c -s` and reflection for the exact Tensor method descriptors and helper
surface. Inspect source/bytecode for one helper delegation per public method and exact local
construction order. Scan production imports and Gradle dependencies for forbidden layers. Confirm
that no storage accessor, segment API, shape algebra, graph ID, compiler/runtime/backend type,
gradient rule, domain check, canonicalization, cache, registry, or service lookup appears. Validate
generated Javadoc, Tensor and Compile API current-versus-planned wording and newcomer example,
glossary terminology, links/anchors/fences/whitespace, exact ten-path scope, synchronized statuses,
and
absence of a task-0014E specification.

## Dependencies

- Task 0001 supplies exact floating data types and differentiability metadata.
- Task 0002 supplies immutable static and dynamic `Shape` values.
- Task 0007 supplies immutable result descriptors.
- Task 0011 supplies the public Tensor surface.
- Task 0012 and task 0013 supply factory identity allocation, `createDerived`, and immutable
  provenance.
- Task 0014C supplies the exact fifteen unary elementwise semantic kinds.

## Follow-up tasks

- Task 0014E remains Draft for typed scalar arithmetic and clamp semantic contracts.
- Task 0014F remains Draft for matching Tensor expressions.
- Compiler capture, optimizer rewrites, autograd expansion, capability analysis, backend ownership,
  numerical kernels, approximation contracts, and conformance tests remain in their owning module
  tasks.

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None. The architecture already assigns public Tensor expression semantics,
backend-independent Operation values, descriptors, and provenance to `modules/model`, while
forbidding graph compilation, physical allocation, backend support, and execution there.

If implementation requires Tensor to become IR, resolved physical layout, storage/value access,
compiler/autograd logic, backend facts, another module dependency, or a changed architecture rule,
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
  parameterlessness, floating validation, shape preservation, descriptor, and provenance rules.
- The exact floating input type is retained. Unary expressions have no second operand requiring
  promotion and do not insert casts.
- The exact immutable input Shape is retained while layout becomes unresolved. Logical shape
  preservation does not imply preservation of physical geometry or materialization.
- Gradient eligibility propagates unchanged for every kind. This metadata tracks the caller's
  request through an expression dependency and does not encode a derivative or gradient rule.
- Every valid request creates a fresh expression. Legacy eager double-inverse canonicalization and
  other value/provenance rewrites belong to future compiler optimization, not mutable Tensor state.
- Strict and fast semantics stay distinct without selecting an algorithm or promising accuracy.
- Output labels are empty. Typed Operation and exact provenance carry meaning instead of generated
  expression strings.

## Known limitations

- The returned Tensor has no computed values or host storage and cannot execute without future
  compiler, prepare, runtime, and backend work.
- Only floating unary expressions are supported. Integral/boolean transformations and explicit
  casts remain separate planned capabilities.
- The output layout is unresolved even when input shape and layout are static.
- No domain validation or numerical edge behavior is promised before execution.
- No gradient rule is attached. `requiresGrad` is eligibility metadata only, including for
  `floor`, `ceil`, and `sign`.
- `FAST_EXP` and `FAST_TANH` identify approximation intent but define no error bound, algorithm, or
  backend availability.
- Repeated equivalent expressions receive distinct Tensor identities until a future compiler
  chooses a valid graph optimization.

## Validation evidence

Planning reviewed the architecture and documentation rules; planning guide, model capability
baseline, master plan, and roadmap; completed tasks 0001, 0002, 0003, 0006, 0007, 0011, 0012,
0013, 0014B, and 0014C; current DataType, Shape, TensorDescriptor, Tensor,
TensorBinaryExpressions, TensorFactory.createDerived, TensorProvenance, Operation,
NoOperationAttrs, and UnaryElementwiseKind source/tests; and the read-only legacy Tensor/TensorOps
unary surface, unary builders, primitive-builder gradient behavior, dtype validation, and
canonicalization tests.

Planning confirmed that current contracts can implement all fifteen methods without a dependency,
Gradle, architecture, or foundational API change. Legacy capability selection retains floating,
shape-preserving semantics while explicitly rejecting mutable graph state, eager storage,
gradient callbacks, expression labels, kernel/fusion coupling, and canonicalization. The planned
package direction remains acyclic.

Planning validation:

- `git diff --check` passed, and the three changed planning files contain no trailing whitespace.
- The canonical section scan found every required task-specification section.
- The relative Markdown-target scan resolved every local `.md` link in this task, the model master
  plan, and the roadmap.
- Status inspection found task 0014D `Ready` exactly once in this specification, its model-master
  row, and its roadmap row.
- Scope inspection found exactly this new task plus the model master plan and roadmap changed; no
  Java, test, Gradle, AGENTS, architecture, API, glossary, or other module file changed during
  planning.
- No task-0014E specification exists; 0014E remains only a Draft queue entry.

Implementation and documentation validation:

- Implementation context `/root/implement_model_0014d` added the fifteen exact public Tensor
  methods, package-private `TensorUnaryExpressions`, the focused unary-expression suite, and only
  the required Tensor API-shape assertions. Independent documentation context
  `/root/implement_model_0014b` then performed a mandatory fresh reread of the architecture,
  documentation/planning rules, current APIs, glossary, task chain, actual source/tests, generated
  reports, bytecode, and complete diff. It applied General plus API/Javadoc style to Java and API
  reference review, Planning style to planning files, and Example format to the new Tensor API
  example.
- The documentation review initially stopped without writing when it found that the Compile API
  still described only binary expression construction as current. Explicit user authorization
  expanded the task from nine to ten paths solely for that page's implementation-status wording.
  The final edit recognizes current binary and unary public Tensor expressions while preserving
  compiler entry, traversal, capture, inference, optimization, artifacts, and execution as planned.
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
  public Javadoc omits the package-private helper; its type, private constructor, and `apply`
  contract were reviewed directly in source and already satisfy the API/Javadoc profile, so the
  documentation pass made no Java edit.
- `./gradlew test` passed for the repository; the final run reported all 36 actionable tasks
  up-to-date and no failing task.
- The complete `UnaryExpressionExample` added to the Tensor API compiled with Java 26 against
  `model-0.1.0-SNAPSHOT.jar` and printed the documented `FLOAT32`, exact-shape, unresolved-layout,
  gradient, label, storage, `FAST_EXP`, canonical-attributes, exact-input, and fresh-result values.
- `javap -p -c -s` confirmed all fifteen exact zero-argument `()Tensor` methods, one matching enum
  constant and one helper call per method, and a final zero-field helper with one private
  constructor and one package-private static `apply`. Helper bytecode confirms input/kind null
  checks, floating validation, exact type/shape/gradient descriptor construction, canonical
  parameterless Operation, one-input provenance, and one `createDerived` call in order.
- Reflection tests confirm the exact Tensor/helper API, visibility, finality, zero fields and
  nested types, non-synchronization, method-to-kind mapping, and all result/identity contracts.
  Import and source scans found only permitted same-module and JDK types and no storage access,
  segment API, shape algebra, graph ID/capture, compiler/runtime/backend type, gradient rule,
  domain check, canonicalization, cache, registry, or service lookup.
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
  backend-conformance, integration-test, or task-0014E specification path changed.
- Task 0014D is synchronized as Complete in this specification, the model master plan, and the
  roadmap. Task 0014E remains the next Draft frontier without a detailed specification.
- `DataType` remains accurate because the helper consumes its existing floating-category query
  without changing differentiability metadata. `Shape` remains accurate because its immutable
  identity is retained without shape algebra. `TensorDescriptor` remains accurate because the new
  result uses its existing unresolved-layout and gradient-eligibility contracts.
- `TensorProvenance`, `TensorFactory.createDerived`, `Operation`, `NoOperationAttrs`, and
  `UnaryElementwiseKind` remain accurate because the helper composes their existing exact-reference,
  identity-allocation, parameterless, typed-kind, and one-input boundaries without changing them.
  Binary expression contracts and tests also remain accurate and unchanged.
- `capabilities.md` requires no edit because it already selects all fifteen public methods and
  distinguishes model/public-expression support from compiler/backend/runtime execution. The
  Training API requires no edit because propagating `requiresGrad` is eligibility metadata only;
  no gradient object, rule, autograd, optimizer, or training behavior changed.
- `ARCHITECTURE.md`, focused architecture pages, ADRs, architecture tests, backend-conformance and
  integration tests, and build configuration require no edit because the implementation remains
  within model-owned public expression semantics and adds no module boundary, dependency,
  lifecycle, backend behavior, numerical execution, Java toolchain, preview/incubator, or
  end-to-end contract.

## Implementation notes

- Added exactly fifteen public zero-argument Tensor methods, each delegating once to the matching
  `UnaryElementwiseKind` through one stateless package-private helper.
- The helper performs the specified null and floating validation, exact data type/Shape/gradient
  retention, unresolved descriptor construction, canonical parameterless Operation, exact
  one-input provenance, and one derived-factory delegation without value or storage access.
- The independent documentation pass found the implementation Javadocs complete, added the current
  unary-expression mental model, method table, runnable example, failures, ownership summary, and
  glossary distinctions, and corrected only the explicitly authorized Compile API status wording.
- No Java declaration, executable logic, or test changed during the documentation pass.

## Completion summary

- Completed changes: Implemented and documented all fifteen floating unary elementwise Tensor
  expression methods with exact descriptor preservation and one-input provenance.
- Files changed or created: Exactly two production files, two tests, Tensor API, Compile API,
  glossary, this task, model master plan, and roadmap.
- Tests and validation: Focused unary 8/8, Tensor 14/14, all 287 model tests across 38 suites,
  model Javadoc, root tests, bytecode/reflection/import/absence checks, the compiled exact example,
  196 local links/anchors, fences, terminology, whitespace, exact scope/status, and
  `git diff --check` passed.
- Documentation-agent review: Clean documentation context `/root/implement_model_0014b` completed
  the independent pass using General, API/Javadoc, Planning, and Example-format guidance.
- Documentation impact: Tensor API and glossary now describe current unary expressions; the
  explicitly authorized Compile API correction recognizes them without claiming compiler behavior.
- Javadoc review: Tensor type/all fifteen methods and the helper type/constructor/method are final;
  reused data type, shape, descriptor, provenance, factory, Operation, no-attributes, unary-kind,
  and binary-expression contracts remain accurate unchanged.
- Glossary impact: Existing implementation-status, OperationKind, Provenance, Tensor, and common
  distinctions now include unary expression construction; no new reusable domain term was needed.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0014D. Task 0014E remains Draft without a specification.

Status: Complete

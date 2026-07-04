# Task 0014C: Unary Elementwise Semantic Kinds

## Status

Complete

## Goal

Introduce one typed, backend-independent vocabulary for the fifteen parameterless unary
elementwise capabilities selected by the model baseline: arithmetic transforms, transcendental
functions, activations, and explicit fast approximation variants.

This task defines semantic identities only. It gives the following public Tensor-expression task
stable kinds to compose with `Operation` and `NoOperationAttrs.INSTANCE`, without adding Tensor
methods, result descriptors, provenance, numerical execution, gradient behavior, or backend
metadata.

## Scope

- Add one public enum `UnaryElementwiseKind` implementing `OperationKind`.
- Define exactly `ABS`, `NEG`, `INV`, `LOG`, `EXP`, `ERF`, `SQRT`, `FLOOR`, `CEIL`, `SIGN`,
  `RELU`, `SIGMOID`, `TANH`, `FAST_EXP`, and `FAST_TANH` in that order.
- Document the mathematical or activation meaning of every kind.
- Establish that all fifteen kinds have one logical input but no intrinsic attributes and compose
  with `NoOperationAttrs.INSTANCE` when represented by `Operation`.
- Add one focused test proving exact enum vocabulary, typed OperationKind behavior, parameterless
  Operation composition, and absence of extra family state or behavior.
- Add the cohesive `model.operation.elementwise.unary` package to the model package map.
- Record the post-0014B checkpoint decision to continue the ordered model queue rather than hide a
  multi-module prerequisite chain inside a cross-module task.
- Finalize affected Javadocs, Tensor API, glossary, task evidence, model master plan, and roadmap
  through the required independent documentation pass during implementation.

## Out of scope

- public Tensor methods for any unary operation or activation
- result Tensor construction, descriptor creation, provenance, labels, storage, IDs, expression
  chaining, or `TensorFactory.createDerived`
- data-type eligibility, result data type, shape preservation, layout resolution, gradient
  eligibility propagation, or local/graph-wide inference
- numeric evaluation, domain or range rules, overflow/underflow, NaN, infinity, signed-zero,
  rounding, exactness, accuracy bounds, monotonicity, or approximation algorithm selection
- gradient rules, subgradient conventions at zero, backward graph operations, autograd, optimizer,
  or training behavior
- a family attributes type, factory, wrapper, registry, parser, aliases, symbols, string dispatch,
  serialization token, classpath scanning, or reflective discovery
- category fields, arithmetic/transcendental/activation tags, strict/fast flags, arity metadata,
  computational cost, fusion flags, result-kind tags, differentiability metadata, capability
  matrices, backend support, routes, lowering, or execution
- scalar multiplication, scalar power, clamp, clampMin, or clampMax; these carry scalar parameters
  and remain in tasks 0014E–0014F
- binary, comparison, logical, selection, cast, reduction, layout, indexing, linear algebra,
  convolution, normalization, loss, or compiler-generated operation kinds
- modifying existing Operation foundation, binary arithmetic, Tensor, descriptor, provenance,
  factory, graph, Gradle, dependency, architecture, another module, or task-0014D specification

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
- [Task 0005](0005-operation-semantic-foundation.md)
- [Task 0006](0006-operation-model.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0014A](0014a-binary-arithmetic-semantic-kinds.md)
- [Task 0014B](0014b-binary-arithmetic-tensor-expressions.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Post-0014B checkpoint decision

Task 0014B completed the first public Tensor operation family that a future compiler can capture.
The planned checkpoint therefore reconsidered whether the next task should leave `modules/model`
and begin a vertical compile-to-execution slice.

The project will continue the ordered model queue with task 0014C. The downstream `trace`,
`backend-contract`, `config`, `planning`, and `compiler` modules currently contain placeholder
classes and broad master plans but no detailed foundational tasks or implemented contracts. A
so-called vertical next task would therefore cross several architecture boundaries and conceal a
large prerequisite sequence instead of validating one bounded integration seam. The global roadmap
already places those areas after the model frontier.

This is an implementation-order decision, not a new dependency or architecture rule. A future
explicit roadmap decision may reorder work when a bounded cross-module task and its prerequisites
are concrete. Until then, the sequential model queue remains the safer executable plan.

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes public Tensor methods `abs`, `neg`, `inv`, `log`,
`exp`, `erf`, `sqrt`, `floor`, `ceil`, `sign`, `relu`, `sigmoid`, `tanh`, `fastExp`, and
`fastTanh`. Each legacy operation descriptor is parameterless and represents a shape-preserving
unary elementwise request. Legacy tests exercise the strict and fast variants across expression
chains, multiple floating data types, non-contiguous inputs, gradients, fusion, and backend paths.

Only the fifteen semantic identities and their stable short names are retained here. Legacy arity,
semantic-family, cost, fusion, result-kind, expression-string, gradient, capability, storage,
lowering, and execution traits belong to later owning layers and are not copied. Scalar power,
scalar multiplication, and clamp operations are deliberately excluded because their scalar values
are intrinsic typed attributes rather than parameterless unary semantics.

## Architecture constraints

- Production remains in `modules/model`, which owns backend-independent operation semantics.
- `UnaryElementwiseKind` identifies a mathematical or activation meaning. It is not an input,
  graph occurrence, Tensor provenance value, result descriptor, executable operation, backend
  capability, approximation implementation, or kernel route.
- The enum implements `OperationKind` directly through inherited `Enum.name()` and adds no
  duplicate name field or method.
- All fifteen kinds have no intrinsic parameters. When wrapped by the later expression task, their
  complete attributes value is `NoOperationAttrs.INSTANCE`, never null or an empty map.
- One-input arity is family context and documentation, not a stored arity field. Input identity
  belongs to Tensor provenance and graph nodes.
- `FAST_EXP` and `FAST_TANH` are explicit user-visible approximate semantic requests, distinct from
  `EXP` and `TANH`. This task does not choose an approximation algorithm or accuracy contract.
- Shape, data type, domain, gradient eligibility, differentiation, and numerical edge semantics are
  not enum fields. The public expression, compiler, autograd, and backend-conformance tasks own
  their applicable rules.
- Package direction is `model.operation.elementwise.unary -> model.operation`; it must not depend
  on Tensor, graph, compiler, planning, runtime, prepare, backend, storage, or training packages.
- Stable enum names are typed diagnostic vocabulary, not wire tokens, registry keys, reflection
  names, or string-dispatch contracts.
- Stop if implementation needs another production type, attributes, Tensor behavior, inference,
  numerical policy, backend metadata, dependency, or architecture change.

## Package impact

Existing package used:

- `io.github.pho001.synaptik.model.operation` — supplies `OperationKind`, `Operation`, and
  `NoOperationAttrs` for the public contract, focused tests, and explanatory composition.

Package added:

```text
io.github.pho001.synaptik.model.operation.elementwise.unary
  Typed parameterless semantic kinds for unary arithmetic, transcendental functions,
  activations, and explicit fast-approximation requests.
```

One family is cohesive because all fifteen capabilities consume one logical Tensor input, carry no
intrinsic attributes, preserve elementwise position, and share the same future public-expression
construction boundary. Separate enums for arithmetic, transcendental, and activation labels would
add type fragmentation without a different arity, attribute, validation, or ownership contract.

Type placement:

- `io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind` — public
  family enum under the owning operation hierarchy.
- `UnaryElementwiseKindTest` — same-package focused test for exact vocabulary and contract shape.

## Required contract

### Enum declaration

Create exactly:

```java
public enum UnaryElementwiseKind implements OperationKind {
    ABS,
    NEG,
    INV,
    LOG,
    EXP,
    ERF,
    SQRT,
    FLOOR,
    CEIL,
    SIGN,
    RELU,
    SIGMOID,
    TANH,
    FAST_EXP,
    FAST_TANH
}
```

The enum declares no fields, explicit constructor, methods, nested types, per-constant class body,
symbols, aliases, category values, or metadata. Compiler-generated enum machinery is not an
additional project API. `Enum.name()` satisfies `OperationKind.name()` and returns the exact
constant spelling.

The semantic identities are:

| Kind | Elementwise meaning |
|---|---|
| `ABS` | absolute magnitude of the input value |
| `NEG` | additive inverse of the input value |
| `INV` | multiplicative reciprocal of the input value |
| `LOG` | natural logarithm of the input value |
| `EXP` | natural exponential of the input value |
| `ERF` | Gaussian error function of the input value |
| `SQRT` | principal square root of the input value |
| `FLOOR` | greatest integer-valued result not greater than the input value |
| `CEIL` | least integer-valued result not less than the input value |
| `SIGN` | negative, zero, or positive sign classification represented numerically |
| `RELU` | rectified linear unit of the input value |
| `SIGMOID` | logistic sigmoid of the input value |
| `TANH` | hyperbolic tangent of the input value |
| `FAST_EXP` | explicitly approximate natural exponential request |
| `FAST_TANH` | explicitly approximate hyperbolic tangent request |

The table defines semantic identity only. It does not define accepted data types, output
descriptor, exact special-value behavior, differentiation, fast-approximation accuracy, execution,
or backend availability.

### Parameterless Operation composition

Every kind is intended to compose explicitly as:

```java
Operation operation = new Operation(
        UnaryElementwiseKind.EXP,
        NoOperationAttrs.INSTANCE);
```

Do not add an enum factory, `operation()` method, attributes type, or compatibility registry.
Existing `Operation` remains an open generic pairing and does not itself prevent a caller from
supplying another `OperationAttrs` implementation. This task documents and tests correct canonical
composition without changing that foundation.

### Typed identity and naming

- `values()` returns exactly the fifteen constants in declaration order.
- `name()` returns the exact constant spelling.
- Standard Java enum identity, equality, hashing, `valueOf`, and inherited `toString` remain.
- An equal diagnostic name in another OperationKind family does not imply equality.
- `EXP` and `FAST_EXP` are different semantic identities, as are `TANH` and `FAST_TANH`; neither
  pair is an alias.
- Names and text are diagnostics only, not external serialization or dispatch contracts.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/unary/UnaryElementwiseKind.java`

Test:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/elementwise/unary/UnaryElementwiseKindTest.java`

Documentation/planning during implementation:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a task-related inconsistency requires stopping:

- `docs/planning/modules/model/capabilities.md` — already selects all fifteen public capabilities.
- Existing OperationKind, OperationAttrs, NoOperationAttrs, Operation, BinaryArithmeticKind,
  Tensor, TensorProvenance, TensorDescriptor, DataType, and Shape Javadocs/tests.
- Compile API, Training API, focused architecture documentation, ADRs, architecture tests,
  backend-conformance material, and integration tests.

## Maximum scope

At most one new production file, one new focused test, and five documentation/planning files:
seven paths total.

Do not modify any existing Java source/test, capabilities, completed task, Compile API, Training
API, Gradle, AGENTS, architecture document/test, another module, or unrelated documentation. Do
not create task 0014D. Stop beyond seven paths or if another production concept, attributes,
Tensor/public expression behavior, numerical policy, dependency, or architecture decision is
required.

## Javadoc requirements

- Type Javadoc must define the family as backend-independent parameterless unary elementwise
  semantic identities and distinguish it from input provenance, result inference, execution,
  gradients, approximation algorithms, and backend support.
- Explain why every kind uses `NoOperationAttrs.INSTANCE` and why one-input arity is not stored as
  metadata.
- Explain typed equality and diagnostic-name limits without promising serialization or dispatch.
- Explain that `FAST_EXP` and `FAST_TANH` are explicit distinct approximate requests while accuracy
  and implementation remain deferred.
- Every enum constant must have detailed Javadoc explaining its mathematical or activation meaning
  and explicitly deferring type/domain/special-value/gradient/execution rules where applicable.
- Review existing Operation foundation, binary-family, and provenance Javadocs and record why they
  remain accurate or stop on an out-of-scope discrepancy.

## Acceptance criteria

- Exactly one public `UnaryElementwiseKind` enum is added in the planned package.
- It implements OperationKind and declares exactly fifteen constants in the specified order and
  spelling.
- It adds no project-declared field, method, constructor, nested type, constant body, alias,
  category, arity, cost, fusion, result, differentiability, approximation, backend, inference, or
  execution metadata.
- Inherited `name()` produces exact diagnostic text and standard enum equality/hash/toString
  behavior remains.
- Every constant constructs a valid Operation with exact kind identity and
  `NoOperationAttrs.INSTANCE`; no family attributes or factory is added.
- Tests prove that strict and fast pairs remain distinct and that equal names from a private
  test-local OperationKind remain typed and unequal.
- Production imports only OperationKind; no Tensor/graph/compiler/planning/runtime/prepare/backend/
  storage/training dependency appears.
- No Tensor method, descriptor, provenance, storage, shape/data-type, numerical, gradient,
  compiler, backend, or execution behavior changes.
- Focused and aggregate model tests, Javadoc, root tests, reflection/javap/import/scope checks,
  documentation links/formatting, and status synchronization pass.
- A separate documentation-focused agent finalizes Javadoc, Tensor API, glossary, task evidence,
  master plan, and roadmap in the same change and records reasoned no-change conclusions for
  related APIs, capabilities, and architecture.
- Task 0014C becomes Complete only after both passes. Task 0014D remains Draft without a detailed
  specification.

## Tests / validation

Run before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKindTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test covers exact declaration/order/names, OperationKind membership, inherited name
and toString, enum identity/equality/hash/valueOf, strict-versus-fast distinctness, no project
instance state or behavior, parameterless Operation composition for every kind, exact reference
retention, and typed separation from a private test enum with an equal diagnostic name.

Manually inspect `javap -p -c -s` and reflection for exact enum shape, fifteen constants, only
compiler-generated static enum machinery, no extra state/API, inherited names, and Operation
composition. Scan production imports/dependencies for forbidden layers. Validate generated
Javadoc, current-versus-planned Tensor API wording, glossary status, local links/anchors/fences/
whitespace, exact seven-path scope, synchronized status, checkpoint decision, and absence of a
task-0014D specification.

## Dependencies

- Task 0005 supplies OperationKind and NoOperationAttrs.INSTANCE.
- Task 0006 supplies the generic immutable Operation descriptor.
- Tasks 0014A–0014B establish the operation-family conventions and complete the checkpoint that
  selected continued sequential model work; their Java contracts are not modified here.

## Follow-up tasks

- Task 0014D will own public Tensor unary/activation expression methods, floating validation,
  shape/data-type preservation, unresolved result descriptors, gradient-eligibility policy, exact
  Operation construction, single-input provenance, and `TensorFactory.createDerived` use.
- Tasks 0014E–0014F remain Draft for scalar arithmetic and clamp attributes/expressions.
- Compiler capture, optimization, autograd expansion, capability analysis, numerical policies,
  backend ownership, kernels, and conformance remain in their owning module tasks.

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None. The architecture already assigns backend-independent Operation semantics to
`modules/model` and excludes backend support, execution, and graph compilation. The checkpoint
changes implementation order only and the enum implements the existing model boundary without a
module dependency change.

If implementation requires Tensor/public-expression behavior, inference, numerical policy,
mutable state, backend metadata, another module/package direction, or an architecture change, stop
and report it.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0005/0006/0013/0014A/0014B/0014C, Tensor API, Compile API,
Training API, glossary, current OperationKind/OperationAttrs/NoOperationAttrs/Operation/
BinaryArithmeticKind/TensorProvenance contracts and tests, and Java 26 Gradle configuration.

Implement task 0014C exactly. Add only UnaryElementwiseKind.java and
UnaryElementwiseKindTest.java under io.github.pho001.synaptik.model.operation.elementwise.unary for
Java code/tests. The public enum implements OperationKind and contains exactly ABS, NEG, INV, LOG,
EXP, ERF, SQRT, FLOOR, CEIL, SIGN, RELU, SIGMOID, TANH, FAST_EXP, FAST_TANH in that order, with no
project fields/methods/nested types/metadata. Every kind is parameterless and composes explicitly
with Operation plus NoOperationAttrs.INSTANCE.

Do not add Tensor methods, operation factories/attrs, dtype/shape/layout/result rules, provenance,
numeric or approximation policies, gradients, graph/compiler/planning/runtime/backend behavior,
legacy traits, dependencies/build/architecture changes, existing Java edits, or later specs. Stop
beyond seven paths or on architecture doubt.

Run every specified focused/aggregate test, Javadoc, javap/reflection/import/manual,
documentation/link/whitespace/scope/status check. Then hand the actual diff/evidence to a separate
clean-context documentation agent in the same change. It must inspect source/tests/generated
Javadoc, finalize permitted Javadocs/Tensor API/glossary/planning, record related-contract/
capability/Compile API/Training API/architecture no-change conclusions, and rerun validation.

Update task 0014C, model master plan, and roadmap only for planning status/evidence. Do not mark
0014C Complete until both passes succeed. Leave 0014D Draft without a specification. Do not commit
or push.
```

## Local decisions

- One `UnaryElementwiseKind` family contains arithmetic, transcendental, activation, and explicit
  fast variants because all share one input, no attributes, elementwise position, and the same
  future Tensor-expression validation. Multiple enums would add type fragmentation without a
  distinct contract boundary.
- The fifteen constants and their order follow the selected public capability baseline rather than
  the legacy class hierarchy.
- `FAST_EXP` and `FAST_TANH` remain distinct public semantic requests, not aliases or backend flags.
  Their exact numerical contracts require later backend/conformance planning.
- Scalar multiplication, scalar power, and clamp stay outside this enum because their scalar values
  are intrinsic immutable attributes owned by tasks 0014E–0014F.
- No enum factory is added. Explicit Operation construction keeps the generic semantic descriptor
  visible and matches the completed binary-family convention.
- The post-0014B checkpoint preserves the ordered model frontier because downstream prerequisite
  modules remain placeholders; it does not change architecture dependencies.

## Known limitations

- The enum alone cannot build a Tensor expression, validate input data type, preserve a descriptor,
  create provenance, compile a graph, execute mathematics, or report backend support.
- Numerical domains and special values are not defined here.
- Fast variants have no algorithm or accuracy guarantee until an owning numerical/conformance task
  defines one.
- Parameterless compatibility is documented and tested by correct construction but is not enforced
  against arbitrary callers by the intentionally open Operation record.
- No serialization or external stable wire-name contract is provided.

## Validation evidence

Planning reviewed the architecture, documentation and planning rules, model master plan and
roadmap, completed operation foundation and binary-family tasks, current Operation contracts and
tests, downstream module master plans/placeholders, the model capability baseline, and the
read-only legacy unary operation descriptors, Tensor methods, builders, tests, and backend
coverage evidence.

The post-0014B checkpoint found that a cross-module next task would require an unplanned chain of
trace, backend-contract, config, planning, and compiler foundations. It therefore selected the
next bounded task in the existing model queue. Legacy capability review confirmed fifteen
parameterless public unary/activation meanings and rejected legacy cost, fusion, expression,
gradient, storage, lowering, and execution coupling. The planned package direction remains acyclic
and requires no architecture or dependency change.

Planning validation:

- `git diff --check` passed for the tracked master-plan and roadmap changes; a trailing-whitespace
  scan also passed for this new task file.
- The canonical task-section scan found every required planning section.
- The local Markdown-target check resolved every link in this task, the model master plan, and the
  roadmap.
- Scope inspection found exactly three documentation/planning paths, no Java, Gradle, AGENTS,
  architecture, or dependency change, and no task-0014D specification.
- Task status is `Ready` in this specification, the model master plan, and the roadmap; task 0014D
  remains `Draft`.

Implementation and documentation validation:

- Implementation context `/root/implement_model_0014c` added only
  `UnaryElementwiseKind.java` and `UnaryElementwiseKindTest.java`. Independent clean-context
  documentation context `/root/implement_model_0014b/review_model_0014b_docs` then reread the
  governing architecture, documentation and planning rules, predecessor tasks, final source and
  tests, related operation/provenance contracts, generated Javadoc, XML reports, API references,
  glossary, and complete workspace diff. It applied General, API/Javadoc, and Planning style;
  Example format was reviewed, but no executable example changed because this task adds semantic
  vocabulary rather than a callable Tensor expression.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKindTest` passed;
  its XML report contains 6 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test` passed; 37 XML suites contain 279 tests with zero failures,
  errors, or skips.
- `./gradlew :modules:model:javadoc` passed. Generated `UnaryElementwiseKind.html` contains the
  type contract and all fifteen constant details, including parameterless composition, arity as
  family context, strict/fast distinctions, diagnostic-name limits, and deferred inference,
  numerical, gradient, execution, and backend behavior.
- `./gradlew test` passed for the repository with no failing task.
- `javap -p -c -s` confirmed the fifteen public constants in exact order, only compiler-generated
  enum fields, methods, constructor, and initialization, and no project state or behavior.
  Focused reflection tests independently confirm no instance fields, instance methods, nested
  types, or constant-specific bodies.
- Production import inspection found only `OperationKind`. The focused tests prove inherited
  enum naming and identity, strict/fast distinction, explicit
  `new Operation(kind, NoOperationAttrs.INSTANCE)` composition for every value, exact reference
  retention, and typed separation from an unrelated equally named kind.
- The Tensor API now lists both implemented production kind families, documents all fifteen unary
  meanings and explicit parameterless composition, and keeps unary Tensor expression methods,
  result inference, provenance construction, numerical policy, gradients, execution, and backend
  support planned. The glossary updates the implementation-status convention, `OperationKind`,
  and the operation-kind/attributes/operation distinction without adding a new domain term.
- The enum type and constant Javadocs already met the API profile, so the documentation pass made
  no Java edit. Existing `OperationKind`, `OperationAttrs`, `NoOperationAttrs`, `Operation`,
  `BinaryArithmeticKind`, and `TensorProvenance` Javadocs/tests remain accurate because the new
  enum implements and composes their contracts without changing validation, ownership, equality,
  parameter, provenance, or occurrence semantics. Tensor, TensorDescriptor, DataType, and Shape
  remain unchanged because this task adds no public expression, descriptor, type, or shape rule.
- `capabilities.md` remains unchanged because it already selects all fifteen capabilities and
  separates model vocabulary from public expression and executable support. Compile API remains
  unchanged because no Tensor expression, capture entry point, inference, graph conversion, or
  artifact changed. Training API remains unchanged because no gradient rule, autograd, optimizer,
  or training behavior changed.
- `ARCHITECTURE.md`, focused architecture documents, ADRs, architecture tests,
  backend-conformance/integration material and tests, Gradle, and dependencies remain unchanged
  because the enum stays inside existing model ownership and changes no module boundary,
  dependency direction, lifecycle, numerical execution, backend behavior, or build contract.
- Final documentation checks resolved all local Markdown file and heading links in the five
  changed documentation/planning files, found balanced fences and no trailing whitespace, and
  `git diff --check` passed. Final scope inspection found exactly the seven authorized paths: one
  production enum, one focused test, Tensor API, glossary, this task, model master plan, and
  roadmap. Task 0014C is Complete in all three planning locations; task 0014D remains Draft and no
  task-0014D specification exists.

## Implementation notes

- Added the exact public `UnaryElementwiseKind` enum with fifteen constants in the specified order
  and no project-declared fields, methods, constructor, nested types, constant bodies, aliases,
  categories, metadata, attributes, or factory.
- Added one six-test focused suite for vocabulary, inherited `OperationKind` behavior, standard
  enum identity, strict/fast distinction, absence of extra API/state, canonical parameterless
  Operation composition, and typed same-name separation.
- Finalized Tensor API, glossary, and planning status after independent documentation review. No
  executable Java logic or tests changed during the documentation pass.

## Completion summary

- Completed changes: Implemented and documented fifteen typed, parameterless unary elementwise
  semantic identities without adding Tensor expression or executable behavior.
- Files changed or created: Exactly the production enum, focused test, Tensor API, glossary, this
  task, model master plan, and roadmap authorized by the task.
- Tests and validation: Focused 6/6, all 279 model tests across 37 suites, model Javadoc, root
  tests, bytecode/reflection/import/API checks, generated-documentation review, Markdown links and
  anchors, fence/whitespace checks, exact scope/status checks, and `git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0014b/review_model_0014b_docs` completed the independent pass using
  General, API/Javadoc, and Planning profiles.
- Documentation impact: Tensor API and glossary now describe unary semantic vocabulary as current
  while retaining public unary expressions and every numerical, gradient, compiler, backend, and
  execution layer as planned. Capabilities, Compile API, Training API, architecture/ADRs/tests,
  backend-conformance/integration material/tests, and build structure require no change for the
  reasons recorded above.
- Javadoc review: The enum type and all fifteen constants are final and required no correction;
  adjacent operation, binary-family, provenance, Tensor, descriptor, data-type, and shape
  contracts remain accurate without edits.
- Glossary impact: Existing implementation-status and operation-kind distinctions were updated;
  no new reusable project term was introduced.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0014C. Task 0014D remains Draft without a detailed
  specification.

Status: Complete

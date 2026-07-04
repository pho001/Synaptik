# Task 0014A: Binary Arithmetic Semantic Kinds

## Status

Complete

## Goal

Introduce the first production operation-family vocabulary: seven typed, backend-independent,
parameterless kinds for tensor-to-tensor elementwise arithmetic. This gives future public Tensor
expression construction and compiler capture stable semantic identities without adding Tensor
methods, inference, execution, backend metadata, or legacy planning traits.

The task is deliberately limited to the semantic-kind layer. A following task will use these kinds
with `Operation`, `NoOperationAttrs.INSTANCE`, Tensor provenance, local broadcasting, and result-
descriptor rules.

## Scope

- Add one public enum `BinaryArithmeticKind` implementing `OperationKind`.
- Define exactly `ADD`, `SUB`, `MUL`, `DIV`, `MIN`, `MAX`, and `POW` in that order.
- Document the operand order and mathematical meaning of every kind.
- Establish that all seven kinds are parameterless and are paired with
  `NoOperationAttrs.INSTANCE` when an `Operation` is constructed.
- Add one focused test proving exact enum vocabulary, typed OperationKind behavior, stable names,
  parameterless Operation composition, and absence of extra family state/behavior.
- Introduce the cohesive `model.operation.elementwise.binary` package in the model package map.
- Finalize affected Javadocs, Tensor API, glossary, task evidence, master plan, and roadmap through
  the required independent documentation pass during implementation.

## Out of scope

- public `Tensor.add`, `sub`, `mul`, `div`, `min`, `max`, or tensor `pow` methods
- Tensor result creation, `TensorFactory.createDerived`, provenance, labels, host storage, IDs,
  descriptor construction, requires-grad propagation, or expression chaining
- shape broadcasting, broadcast plans/attributes/strides, output shape inference, dynamic-symbol
  constraints, or graph-wide validation
- data-type eligibility, numeric promotion, casting, output data type, integer division policy,
  BOOL rejection, floating NaN/signed-zero/tie behavior, or numerical execution
- an operation-family attribute type, factory, wrapper, registry, parser, alias, symbol table,
  string dispatch, classpath/reflection discovery, or serialization token
- arity classes, semantic-family tags, computational cost, result-kind tags, fusion flags,
  differentiability metadata, gradient rules, backend support, capability matrices, kernel routes,
  lowering, or execution
- unary/scalar arithmetic, activations, clamp, comparisons, logical operations, selection, cast,
  reduction, layout, indexing, linear algebra, normalization, loss, or compiler-generated kinds
- modifying existing Operation/OperationKind/OperationAttrs/NoOperationAttrs contracts or tests,
  another production type, dependency, Gradle, architecture, another module, or task-0014B spec

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
- [Task 0013A](0013a-full-value-and-identity-matrix-tensor-creation.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Glossary](../../../../glossary.md)

## Foundation checkpoint decision

The post-task-0013A checkpoint reviewed the implemented data type, shape, layout, descriptor,
identity, immutable graph, host storage, Tensor, factory, Operation foundation, and provenance
contracts against downstream entry needs.

The project will continue with sequential model operation-family work rather than move immediately
to a cross-module vertical slice. The compiler lifecycle has enough structure to consume an
expression later, but no production concrete `OperationKind` currently exists, so graph capture,
capability queries, backend ownership, prepare, and execution would have no real semantic operation
to process. Implementing the first concrete family, followed by its public expression construction,
creates a meaningful seam for later cross-module work while preserving the ordered capability plan.

The broad former task 0014 is therefore decomposed into small semantic-vocabulary and public-
expression tasks. This specification covers only the first semantic vocabulary.

## Capability origin

The selected legacy public capability includes tensor-to-tensor `add`, `sub`, `mul`, `div`, `min`,
`max`, and `pow`, including broadcasted operands. Legacy operation classes also carried mutable or
nullable broadcast plans, expression strings, arity categories, fusion flags, semantic-family
labels, cost classes, and result-kind metadata.

Only the seven mathematical identities and their stable short names are retained here. Broadcast
geometry is derived from input shapes, not stored as an operation attribute. Cost, fusion,
capability, lowering, and execution metadata belong to planning or concrete backend preparation,
not the model kind. No legacy source or class hierarchy is copied.

## Architecture constraints

- Production remains in `modules/model`, which owns backend-independent operation semantics.
- `BinaryArithmeticKind` is a typed semantic discriminator, not a graph node, occurrence ID,
  executable operation, backend capability, or kernel route.
- The enum implements the existing open `OperationKind` contract directly through inherited
  `Enum.name()`; it adds no duplicate name field or method.
- Every kind has no intrinsic parameters. When wrapped later, its complete attributes value is
  `NoOperationAttrs.INSTANCE`, never null or an empty map.
- Input order remains semantically meaningful for `SUB`, `DIV`, and `POW`. This task stores no
  inputs and performs no validation because inputs belong to Tensor provenance and graph nodes.
- Broadcasting is derived state from input shapes. It is not an attribute, plan, stride set, or
  mutable field on the operation kind.
- Package direction is `model.operation.elementwise.binary -> model.operation`; it must not depend
  on Tensor, graph, compiler, planning, runtime, prepare, backend, storage, or training packages.
- Stable kind names are diagnostic typed vocabulary. They are not serialization tokens, registry
  keys, reflection names, or string-dispatch contracts.
- Stop if implementation needs another type, attributes, Tensor/public expression behavior,
  inference, backend metadata, dependency, or architecture change.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.operation` — supplies `OperationKind`, `Operation`, and
  `NoOperationAttrs` for tests and explanatory composition.

Package added:

```text
io.github.pho001.synaptik.model.operation.elementwise.binary
  Typed parameterless semantic kinds for tensor-to-tensor elementwise arithmetic.
```

This package is cohesive because binary arithmetic shares two ordered Tensor operands,
broadcast-aware elementwise meaning, parameterless descriptors, and the same future public
expression boundary. It does not contain Tensor methods, attributes, inference, execution, or
backend variants.

Type placement:

- `io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind` — public
  family enum under the owning operation hierarchy.
- `BinaryArithmeticKindTest` — same-package focused test for exact vocabulary and contract shape.

## Required contract

### Enum declaration

Create exactly:

```java
public enum BinaryArithmeticKind implements OperationKind {
    ADD,
    SUB,
    MUL,
    DIV,
    MIN,
    MAX,
    POW
}
```

The enum declares no fields, explicit constructor, methods, nested types, per-constant class body,
symbols, aliases, or metadata. Compiler-generated enum machinery is not an additional project API.
`Enum.name()` satisfies `OperationKind.name()` and returns the exact constant spelling.

Constant meanings are:

| Kind | Ordered elementwise meaning |
|---|---|
| `ADD` | left value plus right value |
| `SUB` | left value minus right value |
| `MUL` | left value multiplied by right value |
| `DIV` | left value divided by right value |
| `MIN` | minimum selected from left and right values |
| `MAX` | maximum selected from left and right values |
| `POW` | left base raised to the right exponent |

The table defines mathematical identity and operand order only. It does not define supported data
types, promotion, broadcasting success, integer/floating edge behavior, differentiation, execution,
or backend availability.

### Parameterless Operation composition

Every kind is intended to compose as:

```java
Operation operation = new Operation(
        BinaryArithmeticKind.ADD,
        NoOperationAttrs.INSTANCE);
```

Do not add a family factory or an `operation()` method to the enum. Explicit construction keeps the
generic Operation contract visible and avoids introducing convenience behavior before the public
expression task owns validation and provenance. Existing Operation intentionally does not prevent a
caller from pairing a kind with another OperationAttrs implementation; task 0014A neither changes
that generic openness nor adds reflection/registry enforcement.

### Naming and typed identity

- `values()` returns exactly the seven constants in the declared order.
- `name()` returns exactly `ADD`, `SUB`, `MUL`, `DIV`, `MIN`, `MAX`, or `POW`.
- Enum identity/equality/hash semantics remain standard Java enum semantics.
- A future kind in another enum may have the same name without becoming equal. In particular,
  future scalar power may also use `POW`; the concrete family type keeps the meanings distinct.
- `toString()` remains inherited enum diagnostic text and is not overridden or treated as a wire
  format.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/binary/BinaryArithmeticKind.java`

Test:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/elementwise/binary/BinaryArithmeticKindTest.java`

Documentation/planning during implementation:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a task-related inconsistency requires stopping:

- `docs/planning/modules/model/capabilities.md` — already selects all seven public capabilities.
- Existing OperationKind, OperationAttrs, NoOperationAttrs, Operation, TensorProvenance, Shape,
  ShapeBroadcast, DataType, and DataTypePromotion Javadocs/tests.
- Compile API, focused architecture documentation, ADRs, and architecture tests.

## Maximum scope

At most one new production file, one new focused test, and five documentation/planning files:
seven paths total.

Do not modify any existing Java source/test, capabilities, completed task, Gradle, AGENTS,
architecture document/test, another module, or unrelated documentation. Do not create task 0014B.
Stop beyond seven paths or if another production concept, package, attribute, public expression,
inference rule, dependency, or architecture decision is required.

## Javadoc requirements

- Type Javadoc must define the family as backend-independent tensor-to-tensor elementwise
  arithmetic semantic identities and distinguish it from Operation, provenance, graph occurrence,
  inference, execution, and backend support.
- Explain why broadcast metadata is not stored and why every kind uses NoOperationAttrs.INSTANCE.
- Document typed equality and diagnostic-name limits without promising serialization or dispatch.
- Every enum constant must have detailed Javadoc explaining left/right order and mathematical
  meaning, plus explicit deferral of type/shape/numeric/execution semantics where useful.
- Review existing Operation foundation/provenance Javadocs and record why they remain accurate or
  stop on an out-of-scope discrepancy.

## Acceptance criteria

- Exactly one public `BinaryArithmeticKind` enum is added in the planned package.
- It implements OperationKind and declares exactly seven constants in exact order and spelling.
- It adds no project-declared field, method, constructor, nested type, constant body, alias, symbol,
  category, cost, fusion, backend, inference, or execution metadata.
- Inherited `name()` produces exact stable diagnostic text and standard enum equality/hash/toString
  behavior remains.
- Every constant constructs a valid Operation with exact kind identity and
  `NoOperationAttrs.INSTANCE`; no family-specific attrs or factory is added.
- Tests distinguish the typed family from a private test-local enum with an equal constant name.
- Production imports only OperationKind; no Tensor/graph/compiler/planning/runtime/prepare/backend/
  storage/training dependency appears.
- No Tensor method, provenance, descriptor, broadcast, dtype, numerical, execution, gradient, or
  backend behavior changes.
- Focused and aggregate model tests, Javadoc, root tests, reflection/javap/import/scope checks,
  documentation links/formatting, and status synchronization pass.
- A separate documentation-focused agent finalizes Javadoc, Tensor API, glossary, task evidence,
  master plan, and roadmap in the same change and records reasoned no-change conclusions for
  related APIs, capabilities, and architecture.
- Task 0014A becomes Complete only after both passes. Task 0014B remains Draft without a detailed
  specification.

## Tests / validation

Run before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKindTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test covers exact declaration/order/names, OperationKind membership, inherited name and
toString, enum identity/equality/hash, no project instance state or extra instance behavior,
parameterless Operation composition for every kind, exact reference retention, and typed
separation from a private test enum with equal `ADD` text.

Manually inspect `javap -p -c -s` and reflection for exact enum shape, seven constants, only
compiler-generated static enum machinery, no extra state/API, inherited `name`, and Operation
composition. Scan imports/dependencies for forbidden layers. Validate generated Javadoc, production
example/current-versus-planned wording, glossary status, local links/anchors/fences/whitespace,
exact seven paths, synchronized status, and absence of a task-0014B specification.

## Dependencies

- Task 0005 supplies OperationKind and NoOperationAttrs.INSTANCE.
- Task 0006 supplies the generic immutable Operation descriptor.
- Task 0013 supplies Tensor provenance for the later expression-construction task but is not used
  by this enum implementation.
- The completed model foundation checkpoint selects sequential model family work.

## Follow-up tasks

- Task 0014B will own public Tensor binary arithmetic expression construction, local shape/data-
  type validation, result descriptor creation, exact parameterless Operation construction, ordered
  provenance inputs, and use of TensorFactory.createDerived.
- Tasks 0014C–0014F remain Draft semantic/expression groups for unary/activation and scalar/clamp
  capabilities.
- Compiler, planning, backend, runtime, autograd, and numerical conformance remain in their owning
  modules/tasks.

Do not create detailed follow-up specifications in this task.

## Architecture impact

Expected impact: None. The architecture already assigns backend-independent Operation semantics to
modules/model and excludes backend support, execution, and graph compilation. The new typed enum
implements that existing boundary without a module dependency change.

If implementation requires Tensor/public-expression behavior, inference, mutable state, backend
metadata, another module/package direction, or a changed architecture rule, stop and report it.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0005/0006/0013/0013A/0014A, Tensor API, glossary, current
OperationKind/OperationAttrs/NoOperationAttrs/Operation/TensorProvenance contracts and tests, and
Java 26 Gradle configuration.

Implement task 0014A exactly. Add only BinaryArithmeticKind.java and BinaryArithmeticKindTest.java
under io.github.pho001.synaptik.model.operation.elementwise.binary for Java code/tests. The public
enum implements OperationKind and contains exactly ADD, SUB, MUL, DIV, MIN, MAX, POW in that order,
with no project fields/methods/nested types/metadata. Every kind is parameterless and composes
explicitly with Operation plus NoOperationAttrs.INSTANCE.

Do not add Tensor methods, operation factory/attrs, broadcasting/inference/dtype rules, provenance,
graph/compiler/planning/runtime/backend behavior, legacy traits, dependencies/build/architecture
changes, existing Java edits, or later specs. Stop beyond seven paths or on architecture doubt.

Run every specified focused/aggregate test, Javadoc, javap/reflection/import/manual,
documentation/link/whitespace/scope/status check. Then hand the actual diff/evidence to a separate
clean-context documentation agent in the same change. It must inspect source/tests/generated
Javadoc, finalize permitted Javadocs/Tensor API/glossary/planning, record related-contract/
capability/architecture no-change conclusions, and rerun validation.

Update task 0014A, model master plan, and roadmap only for planning status/evidence. Do not mark
0014A Complete until both passes succeed. Leave 0014B Draft without a specification. Do not commit
or push.
```

## Local decisions

- Short constants `ADD`, `SUB`, `MUL`, `DIV`, `MIN`, `MAX`, and `POW` match the public capability
  vocabulary and selected stable legacy semantic names without copying legacy classes.
- The concrete family type supplies semantic context, so a later scalar-family `POW` can remain a
  distinct typed kind even with equal diagnostic text.
- No BinaryArithmeticAttrs type is added because all seven operations have no intrinsic
  parameters. Broadcast geometry and input descriptors are derived from operands.
- No enum factory method is added. Explicit Operation construction belongs to the following public
  expression task and uses the existing generic descriptor visibly.
- Semantic vocabulary is separated from public Tensor expression behavior so local inference and
  provenance policy can be specified and reviewed independently.

## Known limitations

- The enum alone cannot build a Tensor expression, validate operands, infer shape/data type, create
  provenance, compile a graph, execute arithmetic, or report backend support.
- Parameterless compatibility is documented and tested by correct construction but is not enforced
  against arbitrary callers by the intentionally open generic Operation record.
- Numerical edge behavior, supported data types, promotion, broadcasting, gradients, and backend
  conformance remain deferred to owning tasks.
- No serialization or external stable wire-name contract is provided.

## Validation evidence

Planning completed the post-foundation checkpoint by reviewing the implemented model value, graph,
storage, Tensor/factory, Operation, and provenance contracts against the architecture lifecycle and
downstream module entry conditions. It selected continued sequential model work because no
production concrete OperationKind existed yet for a meaningful compiler/backend vertical slice.

Planning also inspected the selected legacy binary operation classes and public tests read-only.
It retained only the seven mathematical identities and stable names, rejecting legacy broadcast-
plan state, expression strings, arity/cost/result/fusion metadata, backend coupling, and execution
behavior. The focused enum fits one production/test pair plus five documentation/planning paths
with no dependency or architecture change.

- Implementation context `/root/implement_model_0014a` added only
  `BinaryArithmeticKind.java` and `BinaryArithmeticKindTest.java`. Independent clean documentation
  context `/root/implement_model_0014a/review_model_0014a_docs` inspected the architecture,
  planning contracts, final source/test, generated Javadoc, XML reports, bytecode, Java 26 build
  configuration, API reference, glossary, and complete diff. It applied General plus API/Javadoc
  style to the enum, Tensor API, and glossary, and Planning style to this task, the model master
  plan, and roadmap.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKindTest` — passed;
  XML reports 5 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — passed; 35 XML suites report 264 tests with zero failures,
  errors, or skips.
- `./gradlew :modules:model:javadoc` — passed without Javadoc errors. Generated
  `BinaryArithmeticKind.html` contains the type contract and all seven constant details, including
  ordered operand meaning, parameterless composition, derived broadcast geometry, typed equality,
  diagnostic-name limits, and deferred inference/execution/backend behavior.
- `./gradlew test` — passed for the repository with 36 actionable tasks and no failing task.
- `javap -classpath modules/model/build/classes/java/main -p -c -s` confirmed seven public
  constants in exact order, only compiler-generated enum fields/methods/constructor/initializer,
  and no project state or behavior. Reflection tests additionally confirmed no instance fields,
  methods, nested types, or constant-specific bodies; inherited `name`, `toString`, identity, and
  hashing remain standard enum behavior.
- Production import inspection found only `OperationKind`, preserving package direction
  `model.operation.elementwise.binary -> model.operation`. Tests prove explicit
  `new Operation(kind, NoOperationAttrs.INSTANCE)` composition for every value, exact reference
  retention, and typed separation from an unrelated private `ADD` enum.
- Tensor API now presents `BinaryArithmeticKind` as the first implemented production family,
  lists the seven ordered mathematical identities, shows explicit parameterless Operation
  composition, and distinguishes semantic vocabulary from Tensor expressions, provenance, graph
  occurrences, inference, compilation, execution, and backend support. Current-versus-planned
  wording now leaves only other kind families, family attributes, and expression/execution layers
  planned.
- Glossary review updated the implementation-status convention, `OperationKind`, and the
  operation-kind/attributes/operation distinction. No new glossary entry was added because
  `BinaryArithmeticKind` is a concrete implementation of the existing OperationKind domain term,
  not a new reusable architecture, lifecycle, or ownership concept; its full vocabulary belongs
  in the Tensor API.
- The submitted `BinaryArithmeticKind` type and constant Javadocs already satisfy the API profile,
  so the documentation context made no Java edit. Existing `OperationKind`, `OperationAttrs`,
  `NoOperationAttrs`, `Operation`, and `TensorProvenance` Javadocs/tests remain accurate because
  the new enum implements and composes their existing contracts without changing validation,
  ownership, equality, provenance, or occurrence semantics. `Shape`, `ShapeBroadcast`, `DataType`,
  and `DataTypePromotion` remain accurate because the enum stores and derives no shape, broadcast,
  type, promotion, or numerical rule.
- `docs/planning/modules/model/capabilities.md` remains unchanged because it already selects all
  seven public capabilities and correctly distinguishes model semantic representation from public
  expression and executable support. The Compile API remains unchanged because no capture entry
  point, inference, graph conversion, or compile artifact changed.
- `ARCHITECTURE.md`, focused architecture documents, ADRs, architecture tests,
  backend-conformance/integration documentation and tests, and build configuration remain
  unchanged because the enum stays within model-owned semantics, adds only the already-planned
  inward package dependency, and changes no module boundary, lifecycle, backend behavior,
  end-to-end execution, Java 26 toolchain/release, preview/incubator setting, or dependency.
- The corrected local Markdown validator resolved 171 file targets and heading anchors across the
  five changed documentation/planning files with zero errors. A preliminary invocation failed on
  Ruby interpolation in the heading regular expression, and the next correctly exposed that the
  first slug approximation collapsed a double hyphen used by an existing heading anchor; neither
  invocation changed repository files. The corrected GitHub-style check passed. Code fences are
  balanced, no changed path has trailing whitespace, terminology and status are synchronized, and
  `git diff --check` passed.
- Final scope review found exactly the seven authorized paths: one production enum, one focused
  test, Tensor API, glossary, this task, model master plan, and roadmap. Task 0014A is Complete in
  all three planning locations. Task 0014B remains Draft and no task-0014B specification exists.

## Implementation notes

- Added the exact public `BinaryArithmeticKind` enum with `ADD`, `SUB`, `MUL`, `DIV`, `MIN`, `MAX`,
  and `POW` in order and no project-declared state, methods, constructor, nested types, constant
  bodies, metadata, attributes, or factory.
- Added one five-test reflection and composition suite covering vocabulary, inherited
  `OperationKind` behavior, standard enum identity, absence of extra API/state, explicit canonical
  no-attributes composition, and typed same-name separation.
- Finalized the Tensor API and glossary for the implemented family and synchronized planning
  status after the independent documentation review and full validation.

## Completion summary

- Completed changes: Implemented and documented the first production concrete OperationKind
  family as seven typed parameterless binary arithmetic semantic identities.
- Files changed or created: Exactly the one production enum, one focused test, Tensor API,
  glossary, this task, model master plan, and roadmap authorized by the task.
- Tests and validation: Focused 5/5, all 264 model tests across 35 suites, model Javadoc, root
  tests, bytecode/reflection/import/API checks, generated-documentation review, Markdown links and
  anchors, fence/terminology/whitespace checks, exact-scope/status checks, and `git diff --check`
  passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0014a/review_model_0014a_docs` completed the independent pass using
  General, API/Javadoc, and Planning profiles.
- Documentation impact: Tensor API and existing glossary/status language now describe the current
  binary arithmetic vocabulary and retain Tensor expressions, inference, compilation, and
  execution as planned work. Capabilities, Compile API, architecture/ADRs/tests,
  backend-conformance/integration documentation/tests, and build structure require no change for
  the reasons recorded above.
- Javadoc review: The enum type and all seven constants are final and required no correction;
  adjacent operation, provenance, shape/broadcast, and data-type/promotion contracts remain
  accurate without edits.
- Glossary impact: Existing implementation-status and operation-kind distinctions were updated;
  no new reusable project term was introduced.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0014A. Task 0014B remains the next Draft planning frontier
  without a detailed specification.

Status: Complete

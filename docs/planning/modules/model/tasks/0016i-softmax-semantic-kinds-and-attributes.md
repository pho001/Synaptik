# Task 0016I: Softmax Semantic Kinds and Attributes

## Status

Complete

## Goal

Define the typed, backend-independent semantic identities and immutable normalized-axis parameter
for softmax and log-softmax. The model must distinguish probability normalization from
log-probability normalization without retaining a Tensor, deriving a result descriptor,
evaluating exponentials or logarithms, defining gradients, or reporting backend support.

This task creates the semantic foundation consumed by task 0016J. It does not add public
`Tensor.softmax` or `Tensor.logSoftmax` expression methods.

## Scope

- Add one public `SoftmaxKind` enum implementing `OperationKind`.
- Define exactly `SOFTMAX` and `LOG_SOFTMAX`, in that order.
- Add one public `SoftmaxAttrs` record implementing `OperationAttrs` with exactly one normalized
  non-negative `int axis` component.
- Reject every negative stored axis with the exact normalized-axis failure contract.
- Define explicit typed composition of both kinds with `SoftmaxAttrs` without changing generic
  `Operation` validation.
- Document one logical input, shape-preserving normalization slices, probability versus
  log-probability meaning, and their mathematical relationship without executing it.
- Document that the stored axis is already normalized while public negative-axis handling belongs
  to task 0016J.
- Add one focused same-package semantic-contract test.
- Add the cohesive `model.operation.normalization` package to the model package map.
- Finalize Javadocs, Tensor API semantic reference, glossary, task evidence, master plan, and
  roadmap through the required independent documentation pass during implementation.

## Out of scope

- public `Tensor.softmax`, `Tensor.logSoftmax`, another Tensor method, overload, factory, builder,
  expression helper, or task-0016J implementation
- temperature, scale, mask, bias, causal mask, epsilon, stabilization toggle, output type,
  precision, accumulation type, approximation mode, or algorithm option
- sparse, sampled, hierarchical, adaptive, masked, fused-attention, cross-entropy, NLL, loss, or
  optimizer semantics
- input Tensor, Shape, caller-axis normalization, shape preservation implementation, result
  descriptor, label, identity, layout, storage, provenance, or `TensorFactory.createDerived`
- input DataType eligibility, output type, promotion, cast, local or graph-wide inference, or
  compatibility validation
- negative stored axis, an all-axes sentinel, no-axis form, multiple axes, axis collection, named
  axis, keep-dimensions flag, or optional axis
- value/storage access, exponential, logarithm, maximum, sum, division, subtraction, allocation,
  materialization, mutation, aliasing, decomposition, or execution
- numerical stability algorithm, subtract-maximum implementation, reduction order, precision,
  overflow, underflow, NaN, infinity, signed zero, empty-axis, or error policy
- gradients, Jacobian-vector products, backward operation kinds, saved forward values, autograd,
  optimizer, or training behavior
- operation factory, registry, parser, visitor, string dispatch, arity, result-kind, family, cost,
  fusion, capability, backend route, kernel, or executable metadata
- changes to Operation foundations, aggregate/scan contracts, DataType, Shape, Tensor, graph
  records, existing Java tests, dependencies, Gradle, architecture, or another module
- compiler, planning, prepare, runtime, backend, engine, tracing, ONNX, conformance, integration,
  or a detailed task-0016J specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Task 0001](0001-data-type-model.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0005](0005-operation-semantic-foundation.md)
- [Task 0006](0006-operation-model.md)
- [Task 0016A](0016a-reduction-semantic-kinds-and-attributes.md)
- [Task 0016G](0016g-cumulative-sum-semantic-kind-and-attributes.md)
- [Task 0016H](0016h-cumulative-sum-tensor-expressions.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes exactly:

```java
Tensor softmax(int axis)
Tensor logSoftmax(int axis)
```

Both public methods require floating input, accept a positive or negative axis, preserve logical
shape and data type, and participate in legacy gradient construction. Legacy execution evidence
covers FLOAT64, FLOAT32, BFLOAT16, non-contiguous inputs, attention and loss composition, ONNX,
CPU, Metal, and CUDA paths.

Legacy public builders decompose both meanings into maximum, subtraction, exponential, sum,
division, and logarithm expressions, while separate historical operation descriptors also exist.
Neither representation is copied as an architectural rule. Synaptik's new model represents
SOFTMAX and LOG_SOFTMAX as first-class backend-independent semantics. A later compiler may
canonicalize, decompose, or preserve them, and concrete backend prepare may choose fused or
decomposed execution routes.

For one normalization slice `x = [1, 2, 3]`, the ideal mathematical meanings are approximately:

| Kind | Semantic result |
|---|---|
| `SOFTMAX` | `[0.09003057, 0.24472847, 0.66524096]` |
| `LOG_SOFTMAX` | `[-2.40760596, -1.40760596, -0.40760596]` |

The SOFTMAX values sum to approximately one. Exponentiating each LOG_SOFTMAX value yields the
corresponding SOFTMAX value. These examples explain semantics only; this task calculates none of
them and selects no finite-precision algorithm.

## Architecture constraints

- Operation kinds and attributes are immutable backend-independent model semantics owned by
  `modules/model`.
- `SoftmaxKind.SOFTMAX` identifies normalized probabilities along one axis.
- `SoftmaxKind.LOG_SOFTMAX` identifies the logarithms of those normalized probabilities as a
  distinct first-class semantic kind, not an implicit `SOFTMAX` plus `LOG` graph fragment.
- `SoftmaxAttrs` is the complete parameter type for both kinds. Its axis is already normalized
  and non-negative; a later Tensor expression validates and normalizes a caller axis against
  Shape.
- A normalization slice contains positions that differ only along the selected axis while all
  other logical coordinates remain fixed. This is semantic context, not stored Shape or arity.
- Both meanings have one logical input and preserve logical positions. Those facts are documented
  but not stored or implemented here.
- Ideal SOFTMAX meaning for slice values `x_i` is `exp(x_i) / sum_j(exp(x_j))`.
- Ideal LOG_SOFTMAX meaning is `x_i - log(sum_j(exp(x_j)))`, mathematically equal to the logarithm
  of SOFTMAX for the same slice. Stable finite-precision evaluation remains executable policy.
- Generic `Operation` continues to validate only non-null kind and attributes. It must not discover
  families, enforce pairing, arity, axis bounds, DataType, Shape, gradient, or backend support.
- Stable enum/record text is diagnostic, not serialization, ONNX, reflective dispatch, backend
  route, or kernel identity.
- Package direction is `model.operation.normalization -> model.operation` only. It must not depend
  on Tensor, datatype, shape, layout, storage, graph, compiler, planning, runtime, prepare,
  backend, or training packages.
- Stop if implementation requires another semantic component, Tensor/Shape/DataType behavior,
  numerical policy, compatibility validation, dependency, or architecture change.

## Package impact

Existing package used:

- `io.github.pho001.synaptik.model.operation` — supplies `OperationKind`, `OperationAttrs`, and
  generic immutable `Operation` composition.

Package added:

```text
io.github.pho001.synaptik.model.operation.normalization
  Typed shape-preserving normalization meanings and immutable normalization parameters.
```

The package is separate from aggregate reduction because neither operation removes or retains an
axis with extent one; each emits one normalized result at every input position. It is separate
from scan because every output depends on the complete selected slice rather than an ordered
prefix. The package can later host other cohesive normalization semantics only when their focused
tasks define them; this task does not predict or add those contracts.

Type placement:

- `io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind` — public probability and
  log-probability semantic identities.
- `io.github.pho001.synaptik.model.operation.normalization.SoftmaxAttrs` — public immutable
  normalized-axis parameter shared by those identities.
- `SoftmaxSemanticsTest` — same-package focused test for both cohesive contracts.

## Required contract

### Semantic kind vocabulary

Create exactly:

```java
public enum SoftmaxKind implements OperationKind {
    SOFTMAX,
    LOG_SOFTMAX
}
```

The enum declares no project field, explicit constructor, method, nested type, per-constant class
body, alias, symbol, or metadata. Compiler-generated enum machinery is not additional project API.
Inherited `Enum.name()` satisfies `OperationKind.name()` and returns exact text `SOFTMAX` or
`LOG_SOFTMAX`.

Each constant means one-input shape-preserving normalization along the normalized axis carried by
`SoftmaxAttrs`. Neither defines input eligibility, result descriptor, numerical algorithm,
gradient behavior, decomposition, execution, or backend availability.

### Axis attributes

Create exactly:

```java
public record SoftmaxAttrs(int axis) implements OperationAttrs
```

The record has exactly one component, one public canonical constructor, an explicit documented
`axis()` accessor, and record-generated `equals`, `hashCode`, and `toString`. Add no overload,
factory, builder, temperature, epsilon, stability flag, field, nested type, optional, sentinel,
cache, or helper API.

The canonical constructor rejects every negative axis before ordinary record assignment with
`IllegalArgumentException` and exact message:

```text
axis must be non-negative: <axis>
```

Accept zero, positive values, and `Integer.MAX_VALUE` structurally. The record cannot prove that
the axis exists for an eventual input Shape.

### Typed composition and mathematical meaning

Document these valid pairings without adding generic compatibility validation:

```java
new Operation(SoftmaxKind.SOFTMAX, attrs)
new Operation(SoftmaxKind.LOG_SOFTMAX, attrs)
```

For one normalization slice with values `x_0 ... x_n`:

- SOFTMAX output at position `i` is the positive normalized exponential of `x_i`; ideal outputs
  sum to one across that slice.
- LOG_SOFTMAX output at position `i` is the natural logarithm of the corresponding ideal SOFTMAX
  output.
- Both preserve every input position and the axis order. They do not reduce rank, reverse
  positions, or retain a singleton reduction axis.

Do not define finite-precision guarantees, evaluation decomposition, maximum subtraction,
summation order, special-value policy, empty-axis behavior, gradient formulas, or executable
support. Generic `Operation` remains unchanged and does not enforce the pairing or any family
rule.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/normalization/SoftmaxKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/normalization/SoftmaxAttrs.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/normalization/SoftmaxSemanticsTest.java`

Documentation/planning:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless inconsistent: Compile API, Training API, capabilities,
Operation foundations, aggregate/scan contracts, focused architecture, ADRs/tests,
conformance/integration tests, and Gradle configuration.

## Maximum scope

At most two production files, one test, and five documentation/planning files: eight paths total.
Do not modify an existing Java contract or test. Stop beyond this scope or if Tensor/Shape/DataType
behavior, another semantic component, numerical policy, compatibility validation, dependency, or
architecture change is needed. Do not create task 0016J.

## Javadoc requirements

- Document the enum, both constants, the record, canonical constructor, and explicit accessor.
- Explain normalization axis and slice, shape preservation, SOFTMAX probability meaning,
  LOG_SOFTMAX log-probability meaning, and their mathematical relationship for a newcomer.
- Include the concrete `[1, 2, 3]` example with approximate outputs and explain the sum-to-one and
  exponentiation relationship.
- Document every parameter, structural axis validation, accepted boundaries, accessor result, and
  exact failure.
- Clearly distinguish ideal mathematical semantics from finite-precision algorithms and defer
  DataType eligibility, descriptor/provenance construction, numerical policy, gradients,
  compiler decomposition, storage, backend support, and execution.
- Review Operation foundations and reduction/scan-family Javadocs and record reasoned no-change
  conclusions or stop.

## Acceptance criteria

- Exactly one public two-constant enum, one public one-component record, and one focused test are
  added under the planned normalization package; no other Java file or API changes.
- Exact enum constants/order/names and absence of project metadata are verified.
- Exact record component/type, interface, constructor/accessor, and absence of extra project
  API/state are verified.
- Negative axes fail with exact type/message; zero, positive, and maximum axes succeed.
- Both kinds compose with the exact `SoftmaxAttrs` reference through generic `Operation`.
- Record value semantics, diagnostic text, and typed distinction from aggregate and scan kinds or
  attributes are verified.
- Javadocs explain both mathematical meanings and their relationship without selecting an
  algorithm, result contract, gradient, or backend behavior.
- No Tensor, DataType, Shape, storage, graph, compiler, runtime, backend, dependency, build, or
  architecture behavior is added.
- Focused/aggregate tests, Javadoc, root tests, reflection/javap/import/scope and documentation
  validation pass.
- A separate clean-context documentation-focused agent finalizes permitted Javadocs, Tensor API,
  glossary, planning, examples, and no-change conclusions.
- 0016I becomes Complete only after both passes; 0016J remains Draft without a specification.

## Tests / validation

Run before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.normalization.SoftmaxSemanticsTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test covers enum/package/API shape; exact constant order/names; record component and
explicit accessor; negative/boundary axes; generated value semantics and exact diagnostic text;
both typed Operation compositions and exact attributes identity; distinct identities from
aggregate reduction and cumulative-sum contracts; ideal example relationships within a declared
tolerance; and absence of fields, nested types, Tensor/Shape/DataType/execution state, aliases,
registries, or metadata.

Manually inspect reflection, `javap -p -c -s`, source, and imports for the exact enum/record shape,
constructor validation before assignment, explicit accessor, no extra project state/API, and no
cross-layer type or behavior. Validate generated Javadoc, Tensor API/glossary, examples,
links/anchors/fences/whitespace, exact eight paths, synchronized statuses, package map, and absence
of a task-0016J specification.

## Dependencies

- 0005 supplies `OperationKind` and `OperationAttrs`.
- 0006 supplies immutable generic `Operation` composition.
- 0016A and 0016G establish normalized-axis conventions and typed separation from aggregate and
  scan semantics.

## Follow-up tasks

- 0016J remains Draft for public floating `Tensor.softmax` and `Tensor.logSoftmax` expression
  methods, caller-axis normalization, shape/type/eligibility retention, descriptor/provenance
  construction, and freshness.
- Compiler tasks own capture, inference validation, canonicalization, optional decomposition,
  autograd expansion, and graph optimization.
- Backend/config/conformance tasks own stable finite-precision algorithms, storage traversal,
  fused/decomposed routes, kernels, and cross-backend parity.
- Loss and attention tasks may later compose these semantics without extending this attributes
  contract silently.

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. This task adds immutable model-owned normalization semantics in a cohesive
operation package without changing module boundaries, dependency direction, or lifecycle
ownership. Stop if an architecture change is required.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0005/0006/0016A/0016G/0016H/0016I, Tensor API, Compile API,
Training API, glossary, current OperationKind/OperationAttrs/Operation and aggregate/scan
contracts/tests, and Java 26 Gradle configuration.

Implement task 0016I exactly. Add only SoftmaxKind.java, SoftmaxAttrs.java, and
SoftmaxSemanticsTest.java under io.github.pho001.synaptik.model.operation.normalization for Java
code/tests.

The public enum implements OperationKind and contains exactly SOFTMAX and LOG_SOFTMAX in order,
with no project fields, methods, nested types, aliases, arity, or metadata. The public record
implements OperationAttrs and contains exactly int axis, an explicit documented accessor, exact
negative-axis validation/message, and no extra API/state. Document shape-preserving normalization
slices, probability versus log-probability meaning, their mathematical relationship, and explicit
kind/attributes pairings.

Do not add Tensor methods, DataType/Shape/result/provenance rules, numerical algorithms,
decomposition, gradients, graph/compiler/planning/runtime/backend behavior, factories/registries,
dependencies, build/architecture changes, existing Java edits, or later specs. Stop beyond eight
paths or on architecture uncertainty.

Run every specified focused/aggregate test, Javadoc, javap/reflection/import/manual,
documentation/link/whitespace/scope/status check. Then hand the actual diff/evidence to a separate
clean-context documentation agent in the same change. It must inspect source/tests/generated
Javadoc, finalize permitted Javadocs/Tensor API/glossary/planning, record related-contract/
capability/Compile API/Training API/architecture no-change conclusions, and rerun validation.

Update task 0016I, model master plan, and roadmap only for planning status/evidence. Do not mark
0016I Complete until both passes succeed. Leave 0016J Draft without a specification. Do not commit
or push.
```

## Local decisions

- Use one `SoftmaxKind` enum for the two closely related meanings and one shared `SoftmaxAttrs`
  axis value. Separate kind classes or duplicate attribute records would split one cohesive
  family without a current semantic difference in parameters.
- Add `operation.normalization` rather than place shape-preserving normalization under aggregate
  reduction or ordered scan. The package reflects semantic responsibility, not the reductions a
  numerical algorithm may use internally.
- Represent SOFTMAX and LOG_SOFTMAX as first-class model meanings. Legacy public decomposition is
  implementation history; compiler canonicalization and backend prepare later own whether an
  executable path remains fused or becomes primitive operations.
- Store only a normalized axis. One input, Shape preservation, output type, floating eligibility,
  ideal sum-to-one/log relationships, and gradients are either documented context or later-task
  behavior, not duplicated attributes.
- Defer temperature, masks, scaling, and numerical-stability options. Existing selected capability
  evidence requires none of them on the public softmax contracts.

## Known limitations

- No public Tensor expressions, Shape normalization/descriptor, provenance, numerical evaluation,
  or gradients yet.
- No finite-precision, empty-axis, NaN/infinity, compiler-decomposition, backend, ONNX, or kernel
  policy.
- Only ordinary one-axis softmax and log-softmax meanings are represented.

## Validation evidence

Planning read the architecture contract and focused architecture explanations; documentation and
planning rules; roadmap; model capabilities/master plan; tasks 0005, 0006, 0016A, 0016G, and
0016H; current Operation, aggregate, and scan contracts/tests; Tensor/Compile/Training APIs;
glossary; and Java 26 Gradle configuration.

The legacy branch was read directly. It confirms exactly one-axis `softmax` and `logSoftmax`
public methods, floating eligibility, negative-axis normalization, shape/type preservation,
gradient composition, stable mathematical intent, FLOAT64/FLOAT32/BFLOAT16, non-contiguous input,
attention/loss composition, ONNX, CPU, Metal, and CUDA evidence. Legacy decomposition, mutable
gradient callbacks, operation traits, storage, lowering, fusion, and kernels are excluded or
assigned to later owners.

Planning selected one two-constant kind, one one-component attributes record, one focused test,
and the new normalization package. Existing Operation foundations suffice; no dependency,
foundational contract, or architecture change is required.

Pre-implementation planning validation after synchronizing this task, the model master plan, and
roadmap:

- `git diff --check` passed.
- The targeted trailing-whitespace scan returned no matches across the three planning files.
- All 175 relative Markdown links across the three planning files resolve locally.
- Markdown fence counts are balanced: sixteen in this task, two in the master plan, and zero in
  the roadmap.
- All 20 task-template headings are present, together with the focused Capability origin,
  Required contract, and Javadoc requirements sections.
- The `[1, 2, 3]` SOFTMAX and LOG_SOFTMAX examples were independently recalculated; the displayed
  rounded values match and the unrounded SOFTMAX values sum to one.
- Task, model master plan, and roadmap consistently identify 0016I as Ready and 0016J as Draft.
- No detailed task-0016J specification exists.
- Repository scope is exactly this task, the model master plan, and the roadmap; no Java, API,
  architecture, Gradle, or other file changed during planning.

Implementation and independent documentation validation:

- The implementation pass added exactly `SoftmaxKind`, `SoftmaxAttrs`, and the same-package
  `SoftmaxSemanticsTest`. It changed no existing Java source or test.
- Clean documentation-focused context `/root/review_model_0016a_docs` performed a fresh task-0016I
  review and applied General style, API and Javadoc style, Planning style, and Example format. It
  independently inspected the complete implementation diff, final normalization and related
  operation/reduction/scan source and tests, generated model Javadoc, bytecode, imports, APIs,
  glossary, planning state, architecture documents, and Java 26 build configuration.
- Both new production Javadocs are complete without revision. They document one logical input,
  shape-preserving normalization slices, the normalized-axis boundary, explicit kind/attributes
  pairings, probability and log-probability meanings, the `[1, 2, 3]` example, sum-to-one and
  exponentiation relationships, every constructor parameter and accessor result, the exact
  negative-axis failure, immutable value semantics, diagnostic-only text, and all deferred
  Tensor, descriptor, numerical, gradient, compiler, backend, and execution behavior.
- `docs/api/tensor-api.md` now classifies `SoftmaxKind` and `SoftmaxAttrs` as current semantic
  contracts, explains a normalization slice, gives both explicit pairings and the ideal
  `[1, 2, 3]` outputs, and keeps public `Tensor.softmax` and `Tensor.logSoftmax` construction
  planned for task 0016J. `docs/glossary.md` now defines softmax/log-softmax and synchronizes the
  normalized-axis, operation-kind, and operation-attributes status distinctions.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.normalization.SoftmaxSemanticsTest` —
  `BUILD SUCCESSFUL`; the XML report contains 9 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; 57 XML suites contain 441 tests with zero
  failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`; generated pages contain both public
  normalization types, both enum constants, the canonical constructor, explicit accessor,
  parameter/result/failure documentation, both ideal examples, and the semantic-versus-executable
  boundary.
- `./gradlew test` — `BUILD SUCCESSFUL`; the complete repository lifecycle finished 36 actionable
  tasks without a failing task.
- `javap -p -c -s` confirmed exactly two ordered enum constants with no project state or behavior,
  and exactly one private final record component, one public constructor, one explicit accessor,
  generated record value methods, and negative-axis validation before field assignment. Focused
  reflection tests independently confirm the exact public/final shapes, interfaces, constructors,
  declared methods, absence of nested types and extra API/state, boundary axes, exact diagnostic
  text, typed distinction from aggregate and scan contracts, and exact-reference `Operation`
  composition.
- Production import inspection found only `OperationKind` and `OperationAttrs`. No Tensor,
  DataType, Shape, layout, storage, graph, compiler, planning, runtime, prepare, backend, trace, or
  training dependency or state was introduced. Package placement and direction are exactly
  `model.operation.normalization -> model.operation`, matching the master-plan package map.
- Local Markdown target-and-anchor validation passed for all 261 relative links in the five
  changed documentation/planning files. Markdown fences are balanced, targeted trailing-whitespace
  and stale-status scans passed, the numerical examples were recalculated by the focused test,
  generated Javadoc rendering was inspected, and `git diff --check` passed.
- Final repository scope is exactly the authorized eight paths: two new production files, one new
  focused test, Tensor API, glossary, this task, model master plan, and roadmap. Task 0016I is
  synchronized as Complete. Task 0016J remains Draft, and no `0016j-*` specification exists.
- `OperationKind`, `OperationAttrs`, and `Operation` remain accurate unchanged because the new
  family implements their open typed contracts and does not change generic compatibility
  validation. Aggregate-reduction and cumulative-sum contracts and tests remain accurate
  unchanged because normalization preserves positions, depends on a complete slice, and uses a
  separate typed family without modifying either existing API.
- `capabilities.md` remains accurate unchanged because it already inventories softmax and
  log-softmax while distinguishing semantic representation from later public and executable
  support. Compile API remains accurate unchanged because this task adds no public expression,
  provenance, capture entry point, inference, canonicalization, artifact, or execution behavior.
  Training API remains accurate unchanged because no gradient eligibility, gradient rule,
  autograd, optimizer, publication, or session behavior changed.
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests,
  backend-conformance tests, and integration tests remain accurate unchanged because this task
  changes no module boundary, dependency direction, lifecycle, backend behavior, or end-to-end
  execution. Java 26 root/model Gradle configuration and all other modules remain accurate
  unchanged because no dependency, source set, language level, preview/incubator feature, task,
  module, or cross-module behavior changed.

## Implementation notes

- Added the exact two-constant `SoftmaxKind` semantic enum and exact one-component `SoftmaxAttrs`
  record in the new cohesive normalization package.
- Added the focused nine-test contract suite covering exact API shape, structural axis validation,
  generated record semantics, both explicit pairings, typed family separation, the ideal example,
  and cross-layer exclusions.
- The documentation-focused pass finalized Tensor API, glossary, task evidence, model master plan,
  and roadmap. It changed no Java source because the submitted Javadocs were already complete.
- No public Tensor method, descriptor, provenance, numerical algorithm, gradient, compiler,
  runtime, backend, dependency, build, or architecture behavior was added.

## Completion summary

- Completed changes: Implemented and documented the backend-independent softmax and log-softmax
  semantic identities and their immutable normalized-axis attributes.
- Files changed or created: Exactly two production Java files, one focused test, Tensor API,
  glossary, this task, model master plan, and roadmap.
- Tests and validation: Focused normalization semantics 9/9, all 441 model tests across 57 suites,
  generated model Javadoc, root tests, bytecode/reflection/import checks, numerical-example and
  rendered-documentation review, Markdown link/anchor/fence/whitespace checks, exact scope/status
  checks, and `git diff --check` passed.
- Documentation-agent review: Clean context `/root/review_model_0016a_docs` completed the mandatory
  fresh independent pass using General, API/Javadoc, Planning, and Example profiles.
- Documentation impact: Tensor API and glossary now distinguish current softmax/log-softmax
  semantics from planned public Tensor construction and executable behavior; task, master plan,
  and roadmap are synchronized.
- Javadoc review: Both new public contracts and all required members are complete unchanged;
  operation foundations and aggregate/scan Javadocs remain accurate.
- Glossary impact: Added softmax/log-softmax terminology and synchronized normalized-axis and
  operation-family status.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0016I. Task 0016J remains Draft without a detailed
  specification.

Status: Complete

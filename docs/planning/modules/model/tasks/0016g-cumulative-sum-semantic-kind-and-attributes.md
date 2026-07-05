# Task 0016G: Cumulative-Sum Semantic Kind and Attributes

## Status

Complete

## Goal

Define the typed, backend-independent semantic identity and immutable parameters for a
single-axis cumulative sum. The model must represent the normalized scan axis, whether the current
element is excluded, and whether traversal proceeds from the end of the axis, without retaining a
Tensor, deriving a descriptor, reading values, or reporting backend support.

This task creates the complete semantic foundation consumed by task 0016H. It does not add public
`Tensor.cumSum` expression methods.

## Scope

- Add one public `CumulativeSumKind` enum implementing `OperationKind`.
- Define exactly one enum constant, `CUM_SUM`.
- Add one public `CumulativeSumAttrs` record implementing `OperationAttrs` with exactly normalized
  non-negative `int axis`, `boolean exclusive`, and `boolean reverse` components, in that order.
- Reject negative stored axes with the exact common normalized-axis failure contract.
- Preserve both booleans exactly and support all four exclusive/reverse combinations.
- Define typed composition of `CUM_SUM` with `CumulativeSumAttrs` without changing generic
  `Operation` validation.
- Document inclusive/exclusive and forward/reverse meaning, including additive-zero boundary
  behavior for exclusive scans, without executing it.
- Document one logical input and shape-preserving ordered scan semantics as family context without
  storing arity, Shape, DataType, descriptor, or result state.
- Add one focused same-package semantic-contract test.
- Add the cohesive `model.operation.scan` package to the model package map.
- Finalize Javadocs, Tensor API semantic reference, glossary, task evidence, master plan, and
  roadmap through the required independent documentation pass during implementation.

## Out of scope

- public `Tensor.cumSum`, another Tensor method, overload, factory, builder, expression helper, or
  task-0016H implementation
- another scan such as cumulative product, minimum, maximum, logical scan, prefix count, segmented
  scan, rolling window, or parallel scan primitive
- input Tensor, Shape, axis normalization against rank, shape preservation implementation, result
  descriptor, label, identity, layout, storage, provenance, or `TensorFactory.createDerived`
- input eligibility, output DataType, promotion, overflow, accumulation precision, BFLOAT16
  conversion, integer width, BOOL rejection, or local/graph-wide inference
- negative caller axis, an all-axes sentinel, multiple axes, axis collection, named axis, optional
  axis, or no-axis form
- value/storage access, allocation, copying, materialization, mutation, aliasing, or execution
- parallel prefix algorithm, work partitioning, associativity optimization, numerical
  reproducibility, vectorization, fusion, cost, route, or kernel selection
- empty-axis execution, NaN, infinity, signed zero, overflow, underflow, or numerical error policy
- gradients, backward scan semantics, gradient rules, autograd, optimizer, or training behavior
- operation factory, compatibility validator, registry, parser, visitor, string dispatch, arity or
  result-kind metadata, backend support, lowering, or executable behavior
- changes to Operation foundations, aggregate/masked reduction contracts, DataType, Shape, Tensor,
  graph records, existing Java tests, dependencies, Gradle, architecture, or another module
- compiler, planning, prepare, runtime, backend, engine, tracing, ONNX, conformance, integration,
  or a detailed task-0016H specification

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
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0005](0005-operation-semantic-foundation.md)
- [Task 0006](0006-operation-model.md)
- [Task 0016A](0016a-reduction-semantic-kinds-and-attributes.md)
- [Task 0016F](0016f-masked-reduction-semantics-and-axis-mapping.md)
- [Task 0016F1](0016f1-masked-sum-and-mean-tensor-expressions.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes `Tensor.cumSum(axis)` and
`Tensor.cumSum(axis, exclusive, reverse)`. The short form means inclusive forward traversal. The
complete form represents all four direction/inclusion combinations. Public axes may be negative,
output shape and data type match the input, BOOL is rejected, and legacy execution evidence covers
floating, INT32, INT64, non-contiguous input, ONNX, CPU, and Metal paths.

For one axis vector `[1, 2, 3]`, the selected meanings are:

| Exclusive | Reverse | Semantic result |
|---|---|---|
| false | false | `[1, 3, 6]` |
| true | false | `[0, 1, 3]` |
| false | true | `[6, 5, 3]` |
| true | true | `[5, 3, 0]` |

These values explain operation meaning only; this task computes none of them. Legacy arity,
family, cost, fusion, result-kind, expression text, backend facts, storage loops, lowering, and
kernel routes are not copied. Task 0016H owns public Tensor validation, negative-axis
normalization, descriptor/provenance construction, type eligibility, and freshness. Compiler and
backend work later own graph behavior, gradients, numerical execution, storage, and kernels.

## Architecture constraints

- Operation kinds and attributes are immutable backend-independent model semantics owned by
  `modules/model`.
- `CumulativeSumKind.CUM_SUM` identifies one ordered cumulative-addition scan. It stores no axis,
  input, output, graph occurrence, executable behavior, or backend information.
- `CumulativeSumAttrs` is the complete attributes type. Its axis is already normalized and
  non-negative; a later Tensor expression validates and normalizes a caller axis against Shape.
- `exclusive == false` includes the current element in its output prefix. `exclusive == true`
  emits the accumulated prior elements and therefore emits additive zero at the first traversed
  position.
- `reverse == false` traverses from lower to higher indices. `reverse == true` traverses from
  higher to lower indices. Reverse changes traversal direction, not output indexing or Shape.
- Every exclusive/reverse combination is valid. There is no separate default attributes singleton,
  negative sentinel, mode enum, or kind per combination.
- Cumulative sum has one logical input and preserves logical Shape positions as semantic family
  context. Those facts are not stored in the kind or attributes and are not implemented here.
- Generic `Operation` continues to validate only non-null kind and attributes. It must not discover
  families, enforce typed pairing, arity, axis bounds, DataType, result Shape, or backend support.
- Stable enum/record text is diagnostic, not serialization, ONNX, reflective dispatch, backend
  route, or kernel identity.
- Package direction is `model.operation.scan -> model.operation` only. It must not depend on
  Tensor, datatype, shape, layout, storage, graph, compiler, planning, runtime, prepare, backend,
  or training packages.
- Stop if implementation requires another semantic type/component, Tensor/Shape/DataType behavior,
  compatibility validation, dependency, or architecture change.

## Package impact

Existing package used:

- `io.github.pho001.synaptik.model.operation` — supplies `OperationKind`, `OperationAttrs`, and
  generic immutable `Operation` composition.

Package added:

```text
io.github.pho001.synaptik.model.operation.scan
  Typed shape-preserving ordered scan meanings and immutable scan parameters.
```

The package is separate from `model.operation.reduction` because an aggregate reduction removes
or contracts logical positions, while cumulative sum preserves one output position for every input
position and changes only each position's dependency prefix.

Type placement:

- `io.github.pho001.synaptik.model.operation.scan.CumulativeSumKind` — public cumulative-sum
  semantic identity.
- `io.github.pho001.synaptik.model.operation.scan.CumulativeSumAttrs` — public immutable normalized
  axis, exclusion, and direction parameters.
- `CumulativeSumSemanticsTest` — same-package focused test for both cohesive contracts.

## Required contract

### Semantic kind vocabulary

Create exactly:

```java
public enum CumulativeSumKind implements OperationKind {
    CUM_SUM
}
```

The enum declares no project field, explicit constructor, method, nested type, per-constant class
body, alias, symbol, or metadata. Compiler-generated enum machinery is not additional project API.
Inherited `Enum.name()` satisfies `OperationKind.name()` and returns exact text `CUM_SUM`.

The constant means one-input cumulative addition along the normalized axis carried by
`CumulativeSumAttrs`. It does not define input eligibility, result descriptor, numerical policy,
gradient behavior, execution, or backend availability.

### Scan attributes

Create exactly:

```java
public record CumulativeSumAttrs(
        int axis,
        boolean exclusive,
        boolean reverse) implements OperationAttrs
```

The record has exactly three components in that order, one public canonical constructor, explicit
documented `axis()`, `exclusive()`, and `reverse()` accessors, and record-generated `equals`,
`hashCode`, and `toString`. Add no overload, factory, builder, mode enum, field, nested type,
optional, sentinel, cache, or helper API.

The canonical constructor rejects every negative `axis` before ordinary record assignment with
`IllegalArgumentException` and exact message:

```text
axis must be non-negative: <axis>
```

Accept zero, positive values, and `Integer.MAX_VALUE` structurally. The record cannot prove that
the axis exists for an eventual input Shape. Both booleans are primitive, require no validation,
and are retained exactly.

### Typed composition and meaning

Document this valid pairing without adding generic compatibility validation:

```java
new Operation(CumulativeSumKind.CUM_SUM, attrs)
```

The four attribute combinations mean:

- `(exclusive=false, reverse=false)` — inclusive forward prefix;
- `(exclusive=true, reverse=false)` — exclusive forward prefix with zero at the lowest index;
- `(exclusive=false, reverse=true)` — inclusive reverse suffix-style accumulation;
- `(exclusive=true, reverse=true)` — exclusive reverse accumulation with zero at the highest
  index.

All forms retain output positions in input order. `reverse` controls accumulation traversal only;
it does not reverse the returned Tensor. `exclusive` excludes only the current element, not the
entire prefix. Additive zero is semantic identity, not an allocated Tensor or stored attribute.

Generic `Operation` remains unchanged and does not enforce pairing, input count, axis bounds,
Shape preservation, type eligibility, numerical behavior, gradients, or backend support.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/scan/CumulativeSumKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/scan/CumulativeSumAttrs.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/scan/CumulativeSumSemanticsTest.java`

Documentation/planning:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless inconsistent: Compile API, Training API, capabilities,
Operation foundations, aggregate/masked reduction contracts, focused architecture, ADRs/tests,
conformance/integration tests, and Gradle configuration.

## Maximum scope

At most two production files, one test, and five documentation/planning files: eight paths total.
Do not modify an existing Java contract or test. Stop beyond this scope or if Tensor/Shape/DataType
behavior, another semantic type/component, a compatibility validator, dependency, or architecture
change is needed. Do not create task 0016H.

## Javadoc requirements

- Document the kind, its sole constant, the record, canonical constructor, and every accessor.
- Explain cumulative sum, one logical input, shape preservation as semantic context, normalized
  axis, inclusive/exclusive behavior, traversal direction, output indexing, and all four modes.
- Include the concrete `[1, 2, 3]` examples and explain why each boundary position is zero or
  contains a suffix/prefix total.
- Document every parameter, structural axis validation, accepted boundaries, return meaning, and
  exact failure.
- Clearly defer DataType eligibility, descriptor/provenance construction, values, numerical policy,
  gradients, compiler behavior, storage, and execution.
- Review Operation foundations and reduction-family Javadocs and record reasoned no-change
  conclusions or stop.

## Acceptance criteria

- Exactly one public one-constant enum, one public three-component record, and one focused test are
  added under the planned scan package; no other Java file or API changes.
- Exact enum constant/name/order and absence of project metadata are verified.
- Exact record components/order/types, interface, constructor/accessors, and absence of extra
  project API/state are verified.
- Negative axes fail with exact type/message; zero, positive, and maximum axes succeed.
- All four boolean combinations retain exact values and documented meanings.
- `Operation` composes the exact kind and attributes reference without generic family validation.
- Record equality, hashing, diagnostic text, and distinct typed identity from aggregate reduction
  kinds/attributes are verified.
- No Tensor, DataType, Shape, storage, graph, compiler, runtime, backend, dependency, build, or
  architecture behavior is added.
- Focused/aggregate tests, Javadoc, root tests, reflection/javap/import/scope and documentation
  validation pass.
- A separate clean-context documentation-focused agent finalizes permitted Javadocs/API/glossary/
  planning and records reasoned no-change conclusions.
- 0016G becomes Complete only after both passes; 0016H remains Draft without a specification.

## Tests / validation

Run before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.scan.CumulativeSumSemanticsTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test covers enum/package/API shape; record component order; explicit accessors;
negative/boundary axes; all four boolean combinations; generated value semantics and exact text;
typed Operation composition and attributes identity; and absence of fields, nested types,
Tensor/Shape/DataType/execution state, aliases, registries, or metadata.

Manually inspect reflection, `javap -p -c -s`, source, and imports for the exact enum/record shape,
constructor validation before assignment, explicit accessors, no extra project state/API, and no
cross-layer type or behavior. Validate generated Javadoc, Tensor API/glossary, links/anchors/
fences/whitespace, exact eight paths, synchronized statuses, package map, and absence of a
task-0016H specification.

## Dependencies

- 0005 supplies `OperationKind` and `OperationAttrs`.
- 0006 supplies immutable generic `Operation` composition.
- 0016A establishes normalized-axis attribute conventions and the distinct aggregate-reduction
  family.

## Follow-up tasks

- 0016H remains Draft for public shape-preserving `Tensor.cumSum` expressions, negative-axis
  normalization, numeric eligibility, descriptor/provenance construction, and freshness.
- 0016I–0016J remain Draft for softmax and log-softmax semantics/expressions.
- Compiler tasks own capture/canonicalization/autograd; backend/config/conformance tasks own scan
  algorithms, numerical behavior, storage, kernels, routes, and cross-backend parity.

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. This task adds immutable model-owned operation semantics in a cohesive scan
package without changing module boundaries, dependency direction, or lifecycle ownership. Stop if
an architecture change is required.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0005/0006/0016A/0016F/0016F1/0016G, Tensor API, Compile API,
Training API, glossary, current OperationKind/OperationAttrs/Operation and aggregate/masked
reduction contracts/tests, and Java 26 Gradle configuration.

Implement task 0016G exactly. Add only CumulativeSumKind.java, CumulativeSumAttrs.java, and
CumulativeSumSemanticsTest.java under io.github.pho001.synaptik.model.operation.scan for Java
code/tests.

The public enum implements OperationKind and contains exactly CUM_SUM with no project fields,
methods, nested types, aliases, arity, or metadata. The public record implements OperationAttrs
and contains exactly int axis, boolean exclusive, and boolean reverse in order, explicit documented
accessors, exact negative-axis validation/message, and no extra API/state. Document all four modes,
one logical input, shape-preserving/output-order meaning, and explicit CUM_SUM/attributes pairing.

Do not add Tensor methods, DataType/Shape/result/provenance rules, value execution, scan algorithm,
gradients, graph/compiler/planning/runtime/backend behavior, factories/registries, dependencies,
build/architecture changes, existing Java edits, or later specs. Stop beyond eight paths or on
architecture uncertainty.

Run every specified focused/aggregate test, Javadoc, javap/reflection/import/manual,
documentation/link/whitespace/scope/status check. Then hand the actual diff/evidence to a separate
clean-context documentation agent in the same change. It must inspect source/tests/generated
Javadoc, finalize permitted Javadocs/Tensor API/glossary/planning, record related-contract/
capability/Compile API/Training API/architecture no-change conclusions, and rerun validation.

Update task 0016G, model master plan, and roadmap only for planning status/evidence. Do not mark
0016G Complete until both passes succeed. Leave 0016H Draft without a specification. Do not commit
or push.
```

## Local decisions

- Use `CumulativeSumKind.CUM_SUM` rather than legacy class/name casing. The underscore matches
  typed semantic enum naming such as `ARG_MAX` while public fluent naming remains a later Tensor
  concern.
- Use one attributes record for all four modes rather than four kinds or a mode enum; axis,
  inclusion, and direction are independent semantic parameters.
- Add a separate `operation.scan` package because scans preserve logical positions and differ from
  aggregate reductions that contract positions.
- Store only the normalized axis and two primitive flags. One input, Shape preservation, additive
  identity, and output order are documented meaning, not duplicated state.
- Treat exclusive boundary zero as fixed cumulative-sum semantics while deferring numeric
  representation and execution.
- Keep public Tensor construction and DataType eligibility in 0016H so this task remains a small
  semantic foundation.

## Known limitations

- No public Tensor expression, Shape normalization/descriptor, provenance, value execution, or
  gradients yet.
- Single-axis cumulative sum only; no segmented, multi-axis, or other cumulative family.
- No numerical accuracy, overflow, backend, ONNX, compiler, or kernel behavior.

## Validation evidence

Planning reviewed the architecture contract and focused architecture docs; documentation/planning
rules; roadmap; model capabilities/master plan; tasks 0005, 0006, 0016A, 0016F, and 0016F1;
current Operation and reduction contracts/tests; Tensor/Compile/Training APIs; glossary; and Java
26 Gradle configuration.

The legacy branch was read directly. It confirms exactly `cumSum(axis)` and
`cumSum(axis, exclusive, reverse)`, negative public axes, four inclusion/direction modes,
shape/type preservation, BOOL rejection, floating/INT32/INT64, non-contiguous input, ONNX, CPU,
and Metal evidence. Legacy operation traits, callbacks, storage, execution loops, lowering,
backend support, and kernels are excluded or reassigned.

Planning selected one single-constant kind, one three-component attributes record, one focused
test, and the new scan package. Existing Operation foundations suffice; no dependency,
foundational contract, or architecture change is required.

Pre-implementation planning validation after synchronizing this task, the model master plan, and
roadmap:

- `git diff --check` passed.
- The targeted trailing-whitespace scan returned no matches across the three planning files.
- All 167 relative Markdown links across the three planning files resolve locally.
- Markdown fence counts are balanced: fourteen in this task, two in the master plan, and zero in
  the roadmap.
- All 20 task-template headings are present, together with the focused Capability origin,
  Required contract, and Javadoc requirements sections.
- Task, model master plan, and roadmap consistently identify 0016G as Ready and 0016H as Draft.
- No detailed task-0016H specification exists.
- Repository scope is exactly this task, the model master plan, and the roadmap; no Java, API,
  architecture, Gradle, or other file changed during planning.

Implementation and independent documentation validation:

- The implementation pass added exactly `CumulativeSumKind`, `CumulativeSumAttrs`, and the
  same-package `CumulativeSumSemanticsTest`. It changed no existing Java source or test.
- Clean documentation-focused Codex context
  `019f3258-da07-7f00-9b6b-6895d9e94e0d` applied General style, API and Javadoc style, Planning
  style, and Example format. It independently inspected the complete implementation diff, final
  scan and related operation/reduction source and tests, generated model Javadoc, bytecode,
  imports, APIs, glossary, planning state, architecture documents, and Java 26 build configuration.
- Both new production Javadocs are complete without revision. They document one logical input,
  shape preservation, normalized-axis ownership, explicit `CUM_SUM`/`CumulativeSumAttrs` pairing,
  inclusive/exclusive and forward/reverse behavior, output order, all four `[1, 2, 3]` examples,
  the exclusive additive-zero boundary, every constructor parameter and accessor result, the
  exact negative-axis failure, immutable record semantics, and all deferred behavior.
- `docs/api/tensor-api.md` now classifies the cumulative-sum semantic pair as current, gives the
  exact composition and four-mode example, and keeps public `Tensor.cumSum` explicitly planned
  for task 0016H. `docs/glossary.md` now defines cumulative sum/scan and synchronizes operation
  kind/attribute status; its adjacent review also corrected two stale statements that still
  described already-completed public masked reduction construction as planned.
- The first post-review focused-test attempt failed before Gradle task execution because the
  restricted sandbox could not open
  `/Users/phujka/.gradle/wrapper/dists/gradle-9.6.1-bin/4ticwg1pgcbps2hj28r8so764/gradle-9.6.1-bin.zip.lck`
  (`FileNotFoundException`, `Operation not permitted`). This was environment-only; the approved
  rerun of the exact command passed from cache with `BUILD SUCCESSFUL` and 3 actionable tasks:
  1 from cache and 2 up-to-date.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.scan.CumulativeSumSemanticsTest` — `BUILD SUCCESSFUL`;
  the XML report contains 8 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; 55 XML suites contain 425 tests with zero
  failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`; generated pages contain both scan types,
  `CUM_SUM`, the canonical constructor, all three explicit accessors, the four concrete modes,
  return/parameter/failure documentation, and the semantic-versus-executable boundary.
- `./gradlew test` — `BUILD SUCCESSFUL`; the repository lifecycle completed 36 actionable tasks
  with no failing task.
- `git diff --check` passed with no output.
- `javap -p -c -s` confirmed one enum constant and no project fields/methods/nested types, plus
  exactly the three record fields/components/accessors, generated record value methods, and the
  negative-axis validation before all record field assignments. Focused reflection tests confirm
  the same public/final shapes, exact interfaces and constructors, typed distinction from
  aggregate reduction, valid boundary axes and all four flag combinations, exact diagnostic text,
  and exact-reference `Operation` composition.
- Production import inspection found only `OperationKind` and `OperationAttrs`. No Tensor,
  DataType, Shape, layout, storage, graph, compiler, planning, runtime, prepare, backend, trace, or
  training dependency or state was introduced. Package placement and direction are exactly
  `model.operation.scan -> model.operation`, matching the master-plan package map.
- Local Markdown target/anchor validation passed for all 252 relative links in the five changed
  documentation/planning files. Markdown fences are balanced, targeted trailing-whitespace and
  terminology checks passed, the four numerical examples were recalculated, and generated
  Javadoc rendering was inspected directly.
- Final repository scope is exactly the authorized eight paths: two new production files, one new
  focused test, Tensor API, glossary, this task, model master plan, and roadmap. Task 0016G is
  synchronized as Complete. Task 0016H remains Draft, and no `0016h-*` task specification exists.
- `OperationKind`, `OperationAttrs`, `NoOperationAttrs`, and `Operation` remain accurate unchanged:
  the new family implements their open typed contracts and does not change generic compatibility
  validation. Aggregate and masked reduction contracts remain accurate unchanged because a scan
  preserves positions and uses a separate typed family without modifying any reduction API,
  attributes, expression construction, or test.
- `capabilities.md` remains accurate unchanged because it already inventories `cumSum` with
  exclusive/reverse options and distinguishes semantic representation from public and executable
  support. Compile API remains accurate unchanged because no public expression, capture,
  inference, canonicalization, artifact, or execution behavior was added. Training API remains
  accurate unchanged because no gradient, autograd, optimizer, publication, or session behavior
  changed.
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests,
  backend-conformance tests, and integration tests remain accurate unchanged because this task
  changes no module boundary, dependency direction, lifecycle, backend behavior, or end-to-end
  execution. Java 26 root/model Gradle configuration remains accurate unchanged because no
  dependency, source set, language level, preview/incubator feature, task, or module changed.

## Implementation notes

- Added the exact one-constant `CumulativeSumKind` semantic enum and exact three-component
  `CumulativeSumAttrs` record in the new cohesive scan package.
- Added the focused eight-test contract suite covering exact API shape, structural axis validation,
  all four modes, generated record semantics, typed composition, and cross-layer exclusions.
- The documentation-focused pass finalized Tensor API, glossary, task evidence, model master plan,
  and roadmap. It changed no Java source because the submitted Javadocs were already complete.
- No public Tensor method, result descriptor, provenance, value execution, scan algorithm,
  gradient, compiler, runtime, backend, dependency, build, or architecture behavior was added.

## Completion summary

- Completed changes: Implemented and documented the backend-independent cumulative-sum semantic
  identity and immutable normalized-axis, exclusive, and reverse attributes.
- Files changed or created: Exactly two production Java files, one focused test, Tensor API,
  glossary, this task, model master plan, and roadmap.
- Tests and validation: Focused scan semantics 8/8, all 425 model tests across 55 suites, generated
  model Javadoc, root tests, bytecode/reflection/import checks, numerical-example and rendered-doc
  review, Markdown link/anchor/fence/whitespace checks, exact scope/status checks, and
  `git diff --check` passed.
- Documentation-agent review: Clean Codex context
  `019f3258-da07-7f00-9b6b-6895d9e94e0d` completed the mandatory independent pass using the
  API/Javadoc and Planning profiles plus General style and Example format.
- Documentation impact: Tensor API and glossary now distinguish current cumulative-sum semantic
  representation from planned public Tensor/executable behavior; task, master plan, and roadmap
  are synchronized.
- Javadoc review: Both new public contracts and all required members are complete unchanged;
  operation foundations and aggregate/masked reduction Javadocs remain accurate.
- Glossary impact: Added cumulative sum/scan terminology and corrected stale masked-reduction
  implementation-status wording found during adjacent-contract review.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0016G. Task 0016H remains Draft without a detailed
  specification.

Status: Complete

# Task 0016F: Masked Reduction Semantics and Axis Mapping

## Status

Complete

## Goal

Define the typed, backend-independent attributes that distinguish masked single-axis `SUM` and
`MEAN` from their ordinary forms and preserve how each mask dimension aligns to an input axis.
The immutable contract must make legacy-compatible non-right-aligned masks representable without
storing Tensor inputs, Shapes, mutable broadcast plans, or executable behavior in an Operation.

This task adds semantic vocabulary only. It does not add public masked Tensor methods, resolve a
mapping from concrete Shapes, derive a result descriptor, construct provenance, inspect values, or
execute a reduction.

## Scope

- Add one public `MaskedReductionAttrs` record implementing `OperationAttrs`.
- Give the record exactly normalized non-negative reduction `axis` and immutable
  `List<Integer> maskInputAxes` components.
- Define `maskInputAxes[i]` as the input-axis position to which mask dimension `i` aligns. Input
  axes omitted from the list are implicit broadcast dimensions.
- Require every mapped input axis to be non-negative and the list to be strictly increasing, so
  mask dimension order is preserved and no two mask dimensions claim one input axis.
- Permit an empty mapping for a rank-zero scalar mask.
- Snapshot caller-owned list state and expose only the immutable value.
- Document `AggregateReductionKind.SUM` and `MEAN` composition with these attributes while keeping
  their ordinary full and axis forms unchanged.
- Define the fixed masked meanings: false positions are excluded; masked sum over no selected
  values is zero; masked mean divides by selected-count and returns zero when that count is zero.
- Add one focused same-package test for record shape, validation, immutability, value semantics,
  mapping examples, typed Operation composition, and exclusions.
- Split the former broad expression task by adding Draft task 0016F1 for public masked sum/mean
  construction and deterministic Shape-based mapping resolution. Do not create its detailed spec.
- Finalize Javadocs, Tensor API semantic reference, glossary, task evidence, master plan, and
  roadmap through the required independent documentation pass.

## Out of scope

- public `Tensor.sum(axis, mask)`, `mean(axis, mask)`, another overload, helper, builder, factory,
  or expression construction
- Shape-based mask-alignment resolution, candidate generation, scoring, ambiguity handling,
  broadcasting validation, result-shape derivation, or axis normalization against an input rank
- Tensor inputs, provenance `[input, mask]`, descriptor construction, data-type eligibility,
  gradient eligibility, label, identity, layout, storage, or `TensorFactory.createDerived`
- full/all-axes masked reduction, `keepDimensions`, multiple reduction axes, named axes, or an
  empty reduction-axis selection
- masked product/min/max/all/any/arg-max, masked loss, attention mask, indexing mask, or a generic
  mask-alignment contract shared across unrelated families
- caller-selected all-masked policy, denominator mode, weights, numeric mask, nullable mask,
  three-valued mask, or implicit mask conversion
- value/storage access, zero filling, counting, division, short-circuiting, eager composition,
  allocation, copying, materialization, mutation, or output storage
- numerical accumulation, NaN, infinity, signed-zero, precision, overflow, or backend behavior
- gradient values/rules, autograd, optimizer, or training execution
- new OperationKind values, changes to enum order, family registry, compatibility validator,
  parser, visitor, arity/result-kind metadata, costs, fusion, routes, or kernels
- changes to `Operation`, `OperationAttrs`, ordinary/arg-max attribute record shapes, DataType,
  Shape, Tensor, graph records, or existing Java tests
- compiler, planning, prepare, runtime, backend, engine, tracing, ONNX, dependencies, Gradle,
  architecture, another module, or a detailed task-0016F1 specification

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
- [Task 0016A](0016a-reduction-semantic-kinds-and-attributes.md)
- [Task 0016B](0016b-sum-mean-and-product-tensor-expressions.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes only axis-removing
`sum(axis, mask)` and `mean(axis, mask)`. It requires a BOOL mask, accepts masks whose rank is no
greater than the input rank, and aligns mask dimensions to compatible ordered input axes. This is
more expressive than ordinary right-aligned broadcasting: mask `[batch, time]` for input
`[batch, time, features]` and reduction axis `time` aligns as `[batch, time, 1]`. Legacy masked sum
treats false positions as zero. Masked mean divides by the number of true mask positions per
output and returns zero when every contributing mask value is false.

Legacy stores alignment through reshape/expand Tensor expressions and mutable runtime graph state.
Those mechanisms are not copied. The new semantic value records only the resolved ordered mapping
needed to interpret the mask later. Task 0016F1 will own deterministic local Shape resolution and
public expression construction. Compiler and backend work will own graph lowering, value
selection, counting, numerical execution, storage, and kernels.

## Architecture constraints

- Operation kinds and attributes are immutable backend-independent model semantics owned by
  `modules/model`.
- `AggregateReductionKind.SUM` or `MEAN` plus `MaskedReductionAttrs` identifies a masked reduction.
  The later provenance order is `[input, mask]`; inputs are not duplicated in attributes.
- Ordinary full `SUM`/`MEAN` with `NoOperationAttrs.INSTANCE` and ordinary single-axis forms with
  `AxisReductionAttrs` remain unchanged.
- The reduction axis is already normalized and non-negative. The record cannot prove that it
  exists for an eventual input rank.
- `maskInputAxes` maps mask dimensions in their original order to input axes. Strictly increasing
  positions make the mapping injective and preserve order without storing implicit singleton
  dimensions.
- The record validates only mapping structure. It cannot know mask rank, input rank, dimension
  extents, symbolic compatibility, selected reduction axis, or output Shape.
- An empty mapping is valid and represents the only structural mapping for a scalar mask. A
  non-empty mapping need not include the reduction axis; such a mask may broadcast across that
  axis when later Shape validation permits it.
- Masked forms are axis-removing in the selected baseline. No `keepDimensions` component or
  negative all-axis sentinel is added.
- The fixed zero-result behavior for an all-false masked sum/mean is part of the requested
  operation meaning but does not execute in this record.
- Generic `Operation` remains family-agnostic and validates only non-null kind/attributes. It does
  not enforce that only SUM/MEAN use these attributes.
- Package direction remains `model.operation.reduction -> model.operation` plus JDK value types.
  The reduction package must not depend on Tensor, Shape, datatype, graph, compiler, runtime,
  backend, or training packages.
- Stop if implementation requires Tensor/Shape behavior, a new operation kind, another component,
  compatibility registry, dependency, or architecture change.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.operation` — supplies `OperationAttrs` and generic `Operation`
  composition.
- `io.github.pho001.synaptik.model.operation.reduction` — owns aggregate meanings and their typed
  immutable parameter values.

No package is added.

Type placement:

- `io.github.pho001.synaptik.model.operation.reduction.MaskedReductionAttrs` — public immutable
  normalized reduction axis and explicit ordered mask-to-input axis mapping.
- `MaskedReductionAttrsTest` — same-package focused semantic-contract test.

## Required contract

### Record surface

Create exactly:

```java
public record MaskedReductionAttrs(
        int axis,
        List<Integer> maskInputAxes) implements OperationAttrs
```

The record has exactly two components in that order, one public canonical constructor, explicit
documented `axis()` and `maskInputAxes()` accessors, and record-generated `equals`, `hashCode`, and
`toString`. Add no other constructor, method, field, nested type, factory, builder, resolver,
candidate, score, Tensor/Shape reference, or cache.

`maskInputAxes[i]` is the zero-based input-axis position corresponding to mask dimension `i`.
Examples:

- empty mapping `[]` describes a scalar mask;
- `[1]` maps a rank-one mask to input axis one;
- `[0, 1]` maps `[batch, time]` onto `[batch, time, features]` and leaves input axis two as an
  implicit singleton/broadcast axis;
- `[0, 2]` maps a rank-two mask to the first and third input axes while preserving mask dimension
  order.

These examples describe structural positions only. This task does not prove that actual dimension
values are compatible.

### Validation and snapshot order

The canonical constructor performs exactly:

1. Reject negative `axis` with `IllegalArgumentException` and exact message
   `axis must be non-negative: <axis>`.
2. `Objects.requireNonNull(maskInputAxes, "maskInputAxes")`.
3. Iterate caller elements from index zero in order. For each index:
   - reject null with `NullPointerException` and exact message `maskInputAxes[<index>]`;
   - reject a negative value with `IllegalArgumentException` and exact message
     `maskInputAxes[<index>] must be non-negative: <value>`;
   - from index one onward, reject a value less than or equal to its predecessor with
     `IllegalArgumentException` and exact message
     `maskInputAxes must be strictly increasing at index <index>: previous=<previous>, current=<current>`.
4. Snapshot the validated list with `List.copyOf(maskInputAxes)` and assign that immutable value to
   the record component.

Axis failure precedes list validation. Within the list, null and negativity at the current index
precede order comparison. Empty and non-empty valid mappings are accepted. `Integer.MAX_VALUE` is
structurally valid because input-rank bounds belong to later Shape resolution.

The accessor returns the stored immutable list value. Tests and Javadocs use equality and
immutability rather than promising list object identity. Mutating the caller list after successful
construction cannot affect the record, and the returned list rejects mutation.

### Semantic pairing

Document these valid pairings without adding generic compatibility validation:

```java
new Operation(AggregateReductionKind.SUM, maskedAttrs)
new Operation(AggregateReductionKind.MEAN, maskedAttrs)
```

The eventual provenance order is exact `[input, mask]`. `SUM` excludes values whose aligned mask
position is false; selecting no values produces zero. `MEAN` sums selected values, divides by the
selected true-count for each output, and produces zero when that count is zero.

Other aggregate kinds do not use `MaskedReductionAttrs` in this selected contract. Generic
`Operation` remains unchanged and does not enforce the pairing, arity, input roles, mapping/Shape
compatibility, result type, or numerical behavior.

### Existing kind documentation

Update only the explanatory Javadoc of `AggregateReductionKind` and its `SUM` and `MEAN` constants
as needed to document the additional typed pairing and fixed masked meaning. Do not change enum
constants, order, fields, constructors, methods, bytecode-visible project API, or ordinary
full/axis semantics. Record a reasoned review of other constants without changing them.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/reduction/MaskedReductionAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/reduction/AggregateReductionKind.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/reduction/MaskedReductionAttrsTest.java`

Documentation/planning:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless inconsistent: Compile API, Training API, capabilities,
`ReductionSemanticsTest`, existing reduction attributes/Javadocs, focused architecture, ADRs/tests,
conformance/integration tests, and Gradle configuration.

## Maximum scope

At most two production files, one test, and five documentation/planning files: eight paths total.
`AggregateReductionKind` may change only in Javadoc. Stop beyond this scope or if Tensor/Shape
behavior, another semantic type/component, public expression construction, a compatibility
validator, dependency, or architecture change is needed. Do not create a detailed task-0016F1
specification.

## Javadoc requirements

- Document the record's purpose, mapping mental model, valid SUM/MEAN pairings, later provenance
  order, axis-removing scope, fixed masked sum/mean meanings, and all deferred behavior.
- Document the canonical constructor with every validation, exact order/message, snapshot
  ownership, accepted empty mapping, and failure conditions.
- Document both accessors with normalized-axis and immutable mapping semantics and no identity
  promise.
- Give at least the `[0, 1]` mapping example in type Javadoc and explain every value.
- Update aggregate enum/SUM/MEAN Javadocs without implying that public Tensor masked expressions
  already exist.
- Review `AxisReductionAttrs`, `ArgMaxAttrs`, Operation foundations, and other kind constants and
  record reasoned no-change conclusions or stop.

## Acceptance criteria

- Exactly one public two-component record and one focused test are added; the aggregate enum has
  Javadoc-only changes and unchanged executable shape.
- Exact component names/order/types, interface, constructor/accessors, and no extra project API are
  verified.
- Exact validation order, exception types, messages, accepted boundaries, and immutable snapshot
  behavior hold.
- Empty, singleton, contiguous, gapped, and maximum-index mappings have tested structural value
  semantics.
- SUM/MEAN compose with the exact attributes reference; ordinary attributes and all existing
  semantic contracts remain unchanged.
- Documentation clearly separates implemented semantic representability from planned Shape
  resolution and Tensor expressions.
- No Tensor, Shape, datatype, storage, graph, compiler, runtime, backend, dependency, build, or
  architecture behavior is added.
- Focused/aggregate tests, Javadoc, root tests, reflection/javap/import/scope and docs validation
  pass.
- A separate clean-context docs agent finalizes permitted Javadocs/API/glossary/planning and
  records no-change conclusions.
- 0016F becomes Complete only after both passes; 0016F1 remains Draft without a specification.

## Tests / validation

Run before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.reduction.MaskedReductionAttrsTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.reduction.ReductionSemanticsTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test covers exact record/package/API shape; canonical-constructor validation order and
messages; empty/singleton/contiguous/gapped/maximum mappings; defensive snapshot and exposed-list
immutability; generated equality/hash/text; exact SUM/MEAN Operation composition and attributes
identity; and absence of Tensor/Shape/execution state.

Manually inspect `javap -p -c -s`, reflection, imports, and source for exact record components,
constructor/accessors, assignment after validation, one `List.copyOf`, no extra method/state, enum
Javadoc-only change, and absence of Tensor/Shape/datatype/storage/graph/cross-layer imports,
resolver/scoring logic, registry/service, dependency, or build change. Validate generated
Javadoc, Tensor API/glossary, links/anchors/fences/whitespace, exact eight paths, synchronized
statuses, 0016F1 Draft row, and absence of a task-0016F1 spec.

## Dependencies

- 0005 supplies the open `OperationKind`/`OperationAttrs` semantic foundation.
- 0006 supplies immutable generic Operation composition without family discovery.
- 0016A supplies SUM/MEAN identities and existing ordinary reduction attributes.
- 0016B establishes public ordinary sum/mean semantics that later masked expressions extend
  without changing.

## Follow-up tasks

- 0016F1 remains Draft for Shape-based mask mapping resolution and public axis-removing masked
  sum/mean Tensor expressions.
- 0016G–0016J remain Draft for cumulative sum and softmax families.
- Compiler tasks own capture/canonicalization/autograd; backend/conformance tasks own aligned mask
  interpretation, value selection, counting, numerical accumulation/division, storage, kernels,
  and cross-backend results.

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. This task adds model-owned immutable operation attributes without changing
module boundaries, dependency direction, or lifecycle ownership. Stop if architecture change is
required.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0005/0006/0016A/0016B/0016F, Tensor API, Compile API,
Training API, glossary, current OperationKind/OperationAttrs/Operation/AggregateReductionKind/
AxisReductionAttrs/ArgMaxAttrs contracts/tests, and Java 26 Gradle configuration.

Implement task 0016F exactly. Add only MaskedReductionAttrs.java and
MaskedReductionAttrsTest.java for new Java files; modify AggregateReductionKind.java Javadoc only.
The record has exactly int axis and immutable List<Integer> maskInputAxes, explicit documented
accessors, exact validation/snapshot order and messages, and no extra API. Document SUM/MEAN plus
these attributes as masked axis-removing semantics with later ordered [input, mask] provenance,
false-value exclusion, valid-count mean denominator, and zero for no selected values.

Do not add Tensor methods, Shape resolver, mapping candidates/scoring, provenance, descriptor,
value behavior, new kinds, compatibility validation, other operations, dependencies, build, or
architecture changes. Stop beyond eight paths or on architecture uncertainty.

Run all task validation, then hand the actual diff/evidence to a separate clean-context docs agent
in the same change. It must inspect source/tests/generated Javadoc, finalize permitted Javadocs/
Tensor API/glossary/planning, record Compile API/Training API/capability/architecture/related-
contract no-change conclusions, and rerun validation.

Update task 0016F, model master plan, and roadmap only for planning status/evidence. Do not mark
0016F Complete until both passes succeed. Leave 0016F1 Draft without a specification. Do not
commit or push.
```

## Local decisions

- Split semantic representability from public expression construction because legacy-compatible
  mask alignment is not ordinary right-aligned broadcasting and must not be inferred from
  provenance arity or hidden in a later backend.
- Reuse SUM/MEAN kinds. Attribute type distinguishes masked forms without adding duplicate kinds;
  ordered future provenance supplies the input and mask identities.
- Store the resolved mask-dimension-to-input-axis mapping explicitly. This is smaller and more
  stable than storing Shapes, reshapes, candidate scores, broadcast strides, or a mutable plan.
- Preserve mask dimension order with a strictly increasing list. An empty list represents a
  scalar mask; omitted input axes are implicit broadcast dimensions.
- Keep masked forms axis-removing because that is the selected legacy public surface. Do not add a
  keep-dimensions option speculatively.
- Fix all-false sum/mean at zero to preserve selected legacy behavior; no caller policy is added.
- Add Draft task 0016F1 rather than combining a new public semantic contract, Shape resolver, and
  Tensor expression API in one implementation session.

## Known limitations

- No public masked Tensor expression or automatic mapping resolution yet.
- Mapping bounds and dimension compatibility require future input/mask Shapes.
- Axis-removing SUM/MEAN only; no full, retained-axis, weighted, or other masked reduction.
- No value execution, gradients, compiler capture, ONNX/backend support, or kernels.

## Validation evidence

Planning reviewed the architecture contract and focused architecture documentation;
documentation/planning rules; roadmap; model capabilities/master plan; tasks 0005, 0006, 0016A,
0016B, and the completed reduction frontier through 0016E; current operation/reduction contracts
and tests; current Shape broadcasting, Tensor where/cast/sum/mean, factory constants, Tensor/
Compile/Training APIs; glossary; and Java 26 Gradle configuration.

The legacy branch was read directly. It confirms axis-removing `sum(axis, mask)` and
`mean(axis, mask)`, BOOL masks, ordered mask-dimension placement beyond right-aligned broadcast,
the `[batch, time] -> [batch, time, 1]` use case, false-value exclusion, valid-count mean
denominator, and zero all-masked result. Legacy reshape/expand graph construction, mutable
broadcast candidates/scores, callbacks, storage, runtime state, lowering, and kernels are excluded
or reassigned.

Planning selected one two-component attribute record, SUM/MEAN Javadoc clarification, one focused
test, and a new Draft expression frontier. Existing Operation foundations suffice; no package,
dependency, foundation, or architecture change is required.

Planning validation after synchronizing this task, the model master plan, and the roadmap:

- `git diff --check` passed.
- The trailing-whitespace scan returned no matches across the three planning files.
- All 160 relative Markdown links across the three planning files resolve locally.
- Markdown fence counts are balanced: eight in this task, two in the master plan, and zero in the
  roadmap.
- All 21 required task-specification headings are present.
- At planning time, the task/master/roadmap statuses consistently identified 0016F as Ready and
  0016F1 as Draft.
- No detailed task-0016F1 specification exists.
- Repository scope is exactly this task, the model master plan, and the roadmap; no Java, API,
  architecture, Gradle, or other file changed during planning.

Implementation and independent documentation validation:

- The clean implementation context `/root/implement_model_0016f` added exactly
  `MaskedReductionAttrs`, its focused test, and Javadoc-only `AggregateReductionKind` changes.
  The record has exactly the specified two components, validation order, immutable snapshot, and
  explicit accessors; no public Tensor, Shape resolver, value execution, dependency, or additional
  API was added.
- Clean documentation context `/root/implement_model_0016f/review_0016f_docs` independently read
  the architecture contract; focused overview, lifecycle, module-boundary, and dependency
  explanations; documentation workflow and General, API/Javadoc, Planning, and Example profiles;
  planning guide and roadmap; model capabilities/master plan; tasks 0005, 0006, 0016A, 0016B,
  and 0016F; Tensor, Compile, and Training API references; glossary; final source/tests; generated
  Javadoc; related operation/reduction contracts; and the actual workspace diff.
- The documentation pass finalized `MaskedReductionAttrs` constructor Javadoc with every exact
  caller-visible failure message, retained the already-complete type/component/accessor semantics,
  and found the `AggregateReductionKind` type/SUM/MEAN Javadocs complete. It finalized the Tensor
  API semantic reference and glossary mapping terminology while preserving the explicit boundary
  before public masked Tensor expressions and Shape resolution. The mapping examples use concrete
  axes, explain omitted broadcast dimensions, and state their structural limitation.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.reduction.MaskedReductionAttrsTest` —
  `BUILD SUCCESSFUL`; 10 tests, zero failures, errors, or skips.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.reduction.ReductionSemanticsTest` —
  `BUILD SUCCESSFUL`; 11 tests, zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; 53 XML suites contain 406 tests with zero
  failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`. Generated pages contain the complete
  record purpose and mapping example, both components, canonical constructor validation order and
  exact messages, both accessor results, immutable ownership, SUM/MEAN pairing, all-false
  semantics, and deferred Tensor/Shape/execution boundary. Generated aggregate-kind pages retain
  all constants and render the masked SUM/MEAN additions.
- `./gradlew test` — `BUILD SUCCESSFUL`; the root lifecycle completed 36 actionable tasks with no
  failing task.
- `javap -p -c -s` confirmed exactly two private final record fields/components in order, one
  public constructor, explicit accessors plus generated record methods, axis-first/list/indexed
  validation, one `List.copyOf` after the complete validation loop, and direct immutable-list
  assignment. The aggregate enum retains the exact eight constants/order and no project field,
  method, or executable shape change; its source diff is Javadoc only.
- Production import and source inspection found only `OperationAttrs`, `List`, and `Objects` for
  the new record. No Tensor, Shape, datatype, layout, storage, provenance, graph, compiler,
  planning, runtime, prepare, backend, engine, trace, training, resolver, scoring, registry, or
  service dependency or behavior was introduced.
- A local Markdown target-and-GitHub-heading checker resolved all 242 links and anchors across the
  five changed documentation/planning files with zero errors. Fence counts are balanced, targeted
  trailing-whitespace inspection found no match, and `git diff --check` passed.
- Final scope is exactly eight authorized paths: two production files, one test, Tensor API,
  glossary, this task, model master plan, and roadmap. Task 0016F is synchronized as Complete;
  0016F1 remains Draft, and no task-0016F1 specification exists. No commit or push occurred.
- `docs/api/compile-api.md` remains accurate unchanged because it inventories current public
  expressions and still claims no masked Tensor method, graph capture, inference, canonicalization,
  compile artifact, or execution behavior. `docs/api/training-api.md` remains accurate unchanged
  because no gradient eligibility, gradient rule, autograd, optimizer, session, or training
  execution behavior changed.
- `capabilities.md` remains accurate unchanged because it already inventories masked sum/mean and
  separates model semantic representation, public Tensor construction, compiler work, and
  executable support. `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture
  tests, backend-conformance tests, and integration tests remain accurate unchanged because the
  task changes no module boundary, dependency direction, lifecycle, backend behavior, or
  end-to-end execution. Java 26 Gradle configuration remains accurate unchanged because no build,
  dependency, preview, incubator, or toolchain behavior changed.
- `OperationKind`, `OperationAttrs`, and `Operation` remain accurate unchanged because the new
  record implements their open typed contracts without changing generic family compatibility.
  `AxisReductionAttrs`, `ArgMaxAttrs`, `ArgMaxTiePolicy`, and `ReductionSemanticsTest` remain
  accurate unchanged because ordinary and arg-max forms did not change. `PROD`, `MIN`, `MAX`,
  `ALL`, `ANY`, and `ARG_MAX` Javadocs remain accurate unchanged because masked attributes apply
  only to SUM/MEAN. Existing Tensor expression contracts remain accurate because no public masked
  overload, provenance, descriptor, or Shape behavior exists yet.

## Implementation notes

- Added `MaskedReductionAttrs(axis, maskInputAxes)` as the exact two-component public record with
  normalized non-negative reduction axis, strictly increasing immutable mapping, exact ordered
  validation, and no additional state or API.
- Updated only `AggregateReductionKind` documentation to describe the existing SUM/MEAN pairing,
  future `[input, mask]` provenance, false-value exclusion, selected-count denominator, and
  all-false zero result. Enum bytecode-visible shape is unchanged.
- Added one focused 10-test suite covering exact shape, boundaries, validation messages and order,
  immutable ownership, record value semantics, SUM/MEAN composition, and cross-layer exclusions.
- Finalized Tensor API and glossary semantic documentation and synchronized task, master-plan,
  and roadmap status while leaving Draft 0016F1 without a detailed specification.

## Completion summary

- Completed changes: Implemented and documented immutable masked SUM/MEAN semantic attributes with
  explicit ordered mask-dimension-to-input-axis mapping and fixed all-false behavior.
- Files changed or created: Exactly two production Java files, one focused test, Tensor API,
  glossary, this task, model master plan, and roadmap.
- Tests and validation: Focused suites 10/10 and 11/11, all 406 model tests across 53 suites,
  model Javadoc, root tests, bytecode/import/dependency/generated-documentation checks, 242 local
  Markdown link/anchor checks, fence/whitespace checks, exact scope/status/spec-absence checks,
  and `git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0016f/review_0016f_docs` completed the independent pass using General,
  API/Javadoc, Planning, and Example profiles.
- Documentation impact: Tensor API and glossary now define current masked reduction semantics and
  mapping while keeping public Tensor expressions and Shape resolution planned. Compile API,
  Training API, capabilities, architecture/ADRs/tests, conformance/integration tests, and build
  configuration remain accurate unchanged for the recorded reasons.
- Javadoc review: `MaskedReductionAttrs` and aggregate SUM/MEAN contracts are complete; related
  operation foundations, ordinary/arg-max attributes, other aggregate constants, and existing
  Tensor expressions remain accurate unchanged.
- Glossary impact: Added masked reduction and mask-to-input axis mapping terminology and
  synchronized aggregate/operation-attribute status.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0016F. Task 0016F1 remains Draft without a detailed
  specification.

Status: Complete

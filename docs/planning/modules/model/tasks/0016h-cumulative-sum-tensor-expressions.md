# Task 0016H: Cumulative-Sum Tensor Expressions

## Status

Complete

## Goal

Add the public model-level expression boundary for a one-axis cumulative sum. A caller must be
able to request the inclusive forward default or explicitly select exclusive and reverse modes.
Construction validates local metadata, normalizes the axis, preserves the input Shape and data
type, records exact one-input provenance, and returns a fresh storage-free Tensor without reading
or accumulating values.

This task completes public expression construction for the semantic contracts introduced by task
0016G. It does not execute a scan, define a gradient rule, capture a compiled graph, or report
backend support.

## Scope

- Add exactly `Tensor.cumSum(int axis)` and
  `Tensor.cumSum(int axis, boolean exclusive, boolean reverse)`.
- Make the short overload explicitly select inclusive forward semantics by passing
  `exclusive=false` and `reverse=false`.
- Add one package-private final `TensorCumulativeSumExpressions` construction boundary in the
  existing tensor package.
- Accept exactly the five current numeric data types: FLOAT64, FLOAT32, BFLOAT16, INT32, and
  INT64.
- Reject BOOL without promotion, conversion, or truthiness semantics.
- Normalize one positive or negative axis exactly once through the exact input Shape.
- Retain the exact input Shape and data type in a new unresolved-layout descriptor.
- Preserve the input descriptor's `requiresGrad` eligibility metadata. Current integral inputs
  necessarily carry false eligibility; this task adds no gradient rule.
- Construct exact `CumulativeSumKind.CUM_SUM` and `CumulativeSumAttrs` metadata.
- Record exact ordered one-input provenance `[input]` and create one fresh derived Tensor with no
  label or host storage.
- Update only the exact Tensor API surface assertion in `TensorTest` and add one focused expression
  test.
- Finalize Javadocs, Tensor API, Compile API current-expression inventory, glossary, task evidence,
  master plan, and roadmap through the required independent documentation pass during
  implementation.

## Out of scope

- another public Tensor method, overload, static factory, builder, alias, or task-0016I
  specification
- cumulative product, minimum, maximum, logical scan, segmented scan, prefix count, rolling
  window, or other scan family
- a no-axis, full-tensor, multiple-axis, axis-list, named-axis, keep-dimensions, output-type, or
  destination form
- BOOL or string truthiness, implicit cast, widening, promotion, accumulation data type, or
  caller-selectable result type
- value or host-storage inspection, allocation, copy, materialization, cumulative addition,
  exclusive-zero emission, reverse traversal, parallel prefix algorithm, mutation, or execution
- numerical accumulation order, precision, associativity, reproducibility, overflow, underflow,
  NaN, infinity, signed-zero, empty-axis, or error policy
- layout preservation, view creation, strides, offset, storage aliasing, or materialization policy
- a gradient rule, backward scan, autograd expansion, gradient Tensor, optimizer, or training
  behavior
- graph capture, `NodeId`, `ValueId`, compiled graph records, compiler inference,
  canonicalization, fusion, cost, planning ownership, prepare, runtime, backend, route, kernel,
  ONNX, or execution behavior
- changing `CumulativeSumKind`, `CumulativeSumAttrs`, `Operation`, `TensorDescriptor`, Shape,
  DataType, TensorFactory, TensorProvenance, existing expression helpers, dependencies, Gradle,
  architecture, or another module
- implementation or detailed specification of task 0016I or any later task

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
- [Task 0016G](0016g-cumulative-sum-semantic-kind-and-attributes.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes exactly:

```java
Tensor cumSum(int axis)
Tensor cumSum(int axis, boolean exclusive, boolean reverse)
```

The short form delegates to inclusive forward mode. The complete form supports all four
inclusion/direction combinations. Legacy evidence confirms positive and negative axis
normalization, shape and data-type preservation, FLOAT64, FLOAT32, BFLOAT16, INT32, and INT64
inputs, BOOL rejection, non-contiguous input, ONNX round trips, CPU execution, and Metal lowering.

Legacy implementation details are not copied. In particular, its operation class casing, generic
reduction family label, expression text, callback builder, forced eager no-grad flag, storage
loops, planner traits, lowering, and kernels do not belong in this model task. In the new
architecture, `requiresGrad` is descriptor eligibility metadata rather than a legacy eager
gradient callback. This task preserves that existing metadata for floating inputs without
claiming or implementing a cumulative-sum gradient rule.

For logical input `[1, 2, 3]`, the semantic contracts already selected by task 0016G are:

| Call mode | Semantic result |
|---|---|
| `cumSum(axis)` or `(false, false)` | `[1, 3, 6]` |
| `(true, false)` | `[0, 1, 3]` |
| `(false, true)` | `[6, 5, 3]` |
| `(true, true)` | `[5, 3, 0]` |

The present task constructs metadata for these meanings but calculates none of these values.

## Architecture constraints

- Public Tensor expression construction and backend-independent operation semantics belong to
  `modules/model`.
- `Tensor` remains public mutable API state and is not an IR node. The returned Tensor carries
  immutable descriptor/provenance metadata plus no initial host storage.
- The helper performs local expression validation and composition only. It must not capture a
  compiled graph, traverse provenance, infer graph-wide facts, or select backend support.
- Validation order is deterministic: non-null input at the helper boundary, numeric input
  eligibility, exact Shape access, one axis normalization, attributes construction, then common
  descriptor/operation/provenance/factory construction.
- The exact input Shape reference is retained because cumulative sum preserves logical positions,
  including dynamic and zero extents. No Shape or Dimension copy is needed.
- Result layout is unresolved even when the input descriptor has a resolved layout. A later
  compiler/prepare/backend path owns layout and materialization decisions.
- Result data type is exactly the input data type. BOOL is rejected; no conversion or promotion is
  inserted.
- Result `requiresGrad` equals the input descriptor value. This is eligibility metadata only and
  does not define a gradient rule or executable backward scan.
- `CumulativeSumAttrs` stores the normalized non-negative axis and the two exact caller mode flags.
- Provenance contains exactly `[input]` in order. It stores no graph IDs, output grouping,
  executable closure, or runtime state.
- Each valid invocation creates one fresh Tensor identity, including repeated identical requests.
  Model construction performs no common-subexpression elimination or canonicalization.
- Package direction remains `model.tensor -> model.operation.scan` and existing model
  foundations. No dependency leaves `modules/model`.
- Stop if implementation requires a semantic-contract change, another public method or type,
  value/storage access, gradient behavior, dependency, or architecture change.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns public Tensor methods and the package-private
  expression-construction helper.
- `io.github.pho001.synaptik.model.operation.scan` — supplies the completed cumulative-sum kind
  and attributes.
- existing datatype, shape, descriptor, operation, and provenance packages supply immutable model
  contracts without modification.

No package is added or renamed.

Type placement:

- `Tensor.cumSum(...)` — public fluent expression surface on the existing Tensor API.
- `io.github.pho001.synaptik.model.tensor.TensorCumulativeSumExpressions` — package-private
  stateless boundary that validates and composes cumulative-sum metadata.
- `TensorCumulativeSumExpressionTest` — same-package test of helper and public expression behavior.

## Required contract

### Public Tensor surface

Add exactly:

```java
public Tensor cumSum(int axis)

public Tensor cumSum(int axis, boolean exclusive, boolean reverse)
```

Both are public, non-static, non-synchronized instance methods returning Tensor. The short overload
performs exactly one delegation equivalent to:

```java
TensorCumulativeSumExpressions.apply(this, axis, false, false)
```

The complete overload performs exactly one delegation with the exact caller booleans. Neither
method performs validation, reads state, constructs metadata, accesses storage, or allocates an
identity itself.

### Construction helper

Add one package-private final `TensorCumulativeSumExpressions` with:

- no field, nested type, interface, instance state, or public/protected member;
- one private zero-argument constructor;
- exactly one package-private static
  `apply(Tensor input, int axis, boolean exclusive, boolean reverse)` entry;
- one private static `validateNumericInput(Tensor input)` method; and
- one private static
  `create(Tensor input, Shape shape, CumulativeSumAttrs attrs)` method.

Add no overload, generic operation-family helper, factory, service, registry, cache, state, or test
hook.

### Validation and construction order

`apply` performs exactly this sequence:

1. `Objects.requireNonNull(input, "input")`;
2. validate that `input.descriptor().dataType()` is floating or integral;
3. read the exact `Shape inputShape = input.descriptor().shape()`;
4. normalize `axis` exactly once through `inputShape.normalizeAxis(axis)`;
5. create one `CumulativeSumAttrs(normalizedAxis, exclusive, reverse)`;
6. call `create(input, inputShape, attrs)` exactly once.

BOOL fails at step 2 with `IllegalArgumentException` and exact message:

```text
input must have a numeric data type, but was BOOL
```

Invalid axes use the existing Shape failure type and exact message. Type validation precedes axis
validation, including for BOOL scalar input. Failures before factory delegation consume no Tensor
identity.

`create` performs exactly this sequence:

1. construct one `TensorDescriptor` from the exact input data type, exact supplied Shape,
   `Optional.empty()` layout, and unchanged input `requiresGrad`;
2. construct one `Operation(CumulativeSumKind.CUM_SUM, attrs)` retaining the exact attributes;
3. construct one `TensorProvenance(operation, List.of(input))`;
4. call `TensorFactory.createDerived(descriptor, Optional.empty(), provenance)` exactly once and
   return its result.

Construction reads no input label, provenance, host storage, layout geometry, dimensions, element
count, or values. The result has no label or storage and a present exact one-input provenance.

### Shape, type, and mode behavior

- Every numeric input rank greater than zero is structurally eligible, including dynamic and zero
  extents.
- Every scalar axis is invalid because rank zero has no axis.
- Positive and negative caller axes normalize to the same non-negative stored axis when they
  address the same input position.
- The result descriptor retains the exact input Shape reference, exact data type, and exact
  gradient-eligibility value; it never retains a resolved input layout.
- All four exclusive/reverse combinations retain their exact primitive flags.
- Reverse changes semantic traversal direction only. Neither Shape nor dimension order is
  reversed.
- Every valid invocation is fresh and explicit. Repeated equivalent calls must not return the
  input or reuse another result.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorCumulativeSumExpressions.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorCumulativeSumExpressionTest.java`

Documentation/planning:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless inconsistent: Training API, capabilities, cumulative-sum
semantic contracts/tests, descriptor/factory/provenance/Shape/DataType contracts, other expression
families, focused architecture, ADRs/tests, conformance/integration tests, and Gradle.

## Maximum scope

At most two production files, two tests, and six documentation/planning files: ten paths total.
Do not modify another existing Java contract or test. Stop beyond this scope or if implementation
requires semantic-contract changes, another public surface, value/storage behavior, gradient
rules, dependencies, or architecture changes. Do not create task 0016I.

## Javadoc requirements

- Document both public Tensor methods and the package-private helper, constructor, entry, and
  private methods.
- Explain cumulative sum for a newcomer, including axis, prefix, inclusive, exclusive, forward,
  reverse, output order, and shape preservation.
- Include the concrete `[1, 2, 3]` four-mode example and comment on each boundary/result.
- Document every parameter, exact mode default, accepted numeric types, axis range and negative
  normalization, returned metadata, freshness, unresolved layout, absent label/storage, and
  one-input provenance.
- Document every caller-visible failure type and deterministic ordering, including the exact BOOL
  failure and scalar/out-of-range axis behavior.
- Explain that construction does not inspect or accumulate values and does not define numerical,
  gradient, compiler, backend, or execution behavior.
- Review Tensor, descriptor, factory, provenance, Shape, DataType, Operation, and cumulative-sum
  semantic Javadocs. Record why unchanged contracts remain accurate or stop on an out-of-scope
  discrepancy.

## Acceptance criteria

- Tensor exposes exactly the two requested `cumSum` overloads with exact signatures and one helper
  delegation each; no other public API changes.
- The short form explicitly supplies `(false, false)` and the complete form retains both caller
  flags exactly.
- The helper has exactly the planned package visibility, finality, constructor, three methods,
  imports, and no state or extra API.
- All five numeric data types succeed; BOOL fails with exact type/message before axis validation.
- Positive and negative axes normalize exactly once; invalid/scalar axes preserve the existing
  Shape failure contract.
- Static, dynamic, and zero-extent Shapes are accepted when the axis exists. The exact Shape
  reference, input data type, and gradient eligibility are retained; layout is unresolved.
- Exact CUM_SUM/attributes semantics and exact ordered `[input]` provenance are present. Result
  label and host storage are absent.
- Every valid call produces a fresh identity without returning or mutating the input, inspecting
  values/storage, canonicalizing, or executing a scan.
- `TensorTest` changes only for the deliberate two-method public API shape. One focused test owns
  all cumulative-sum behavior.
- Focused and aggregate tests, Javadoc, root tests, reflection/javap/bytecode/import/scope and
  documentation validation pass.
- A separate clean-context documentation-focused agent finalizes permitted Javadocs, Tensor API,
  Compile API, glossary, planning, examples, and no-change conclusions.
- 0016H becomes Complete only after both passes; 0016I remains Draft without a specification.

## Tests / validation

Run before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorCumulativeSumExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test covers exact public/helper API shape; short-form delegation; all four mode flags;
all five numeric data types and BOOL rejection; positive/negative/scalar/out-of-range axes;
static, dynamic, zero-extent, unresolved/resolved-input-layout Shapes; exact descriptor facts;
same Shape/type/eligibility references or values; exact kind/attributes/provenance references;
absent label/storage; input non-mutation; freshness; failure order and identity side effects; and
absence of value, storage, graph, compiler, runtime, or backend behavior.

Manually inspect reflection, `javap -p -c -s`, source, imports, and bytecode for the exact two
public delegations, one axis normalization, validation/construction order, exact helper surface,
one descriptor/operation/provenance/factory path, no hidden state, no value/storage access, and no
cross-layer imports. Validate generated Javadoc, Tensor API/Compile API/glossary, newcomer example,
links/anchors/fences/whitespace, exact ten paths, synchronized statuses, and absence of a
task-0016I specification.

## Dependencies

- 0001 supplies numeric-versus-BOOL DataType categories and gradient eligibility.
- 0002 supplies exact immutable Shape retention and positive/negative axis normalization.
- 0006 supplies immutable generic Operation composition.
- 0007 supplies unresolved TensorDescriptor construction.
- 0011–0013 supply public Tensor metadata, central derived identity allocation, and immutable
  operation/input provenance.
- 0016G supplies exact CUM_SUM identity and normalized-axis/exclusive/reverse attributes.

## Follow-up tasks

- 0016I remains Draft for softmax and log-softmax semantic kinds and attributes.
- 0016J remains Draft for their public floating Tensor expression construction.
- Compiler tasks own capture, inference validation, canonicalization, autograd expansion, and
  graph optimization.
- Backend/config/conformance tasks own executable cumulative-sum algorithms, numerical policy,
  storage traversal, kernels, routes, and parity.

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. This task composes existing model-owned public Tensor, descriptor,
operation, provenance, Shape, and factory contracts without changing module boundaries or
dependency direction. Stop if an architecture change is required.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0002/0006/0007/0011/0012/0013/0016G/0016H, Tensor API,
Compile API, Training API, glossary, current DataType/Shape/TensorDescriptor/Tensor/TensorFactory/
TensorProvenance/Operation/CumulativeSumKind/CumulativeSumAttrs contracts and tests, and Java 26
Gradle configuration.

Implement task 0016H exactly. Modify Tensor.java and add package-private final
TensorCumulativeSumExpressions.java for production. Update TensorTest only for the exact two-method
API surface and add TensorCumulativeSumExpressionTest. Add exactly cumSum(axis) and
cumSum(axis,exclusive,reverse); each delegates once to the shared helper, and the short form uses
false/false.

The helper has exactly apply, validateNumericInput, and create. Null-check input, accept exactly
floating or integral data types, retain the exact input Shape, normalize the axis once, construct
exact CumulativeSumAttrs, preserve exact type and requiresGrad in an unresolved descriptor, create
exact CUM_SUM Operation and one-input provenance, and call createDerived once with no label/storage.
Every call is fresh.

Do not inspect or accumulate values/storage, preserve resolved layout, convert/promote types,
define numerical or gradient rules, capture graphs, add overloads, change existing contracts, or
introduce compiler/runtime/backend behavior. Stop beyond ten paths or on architecture uncertainty.

Run all task validation, then hand the actual diff/evidence to a separate clean-context docs agent
in the same change. It must inspect source/tests/generated Javadoc, finalize permitted Javadocs/
Tensor API/Compile API/glossary/planning, record related-contract/capability/Training API/
architecture no-change conclusions, and rerun validation.

Update task 0016H, model master plan, and roadmap only for planning status/evidence. Do not mark
0016H Complete until both passes succeed. Leave 0016I Draft without a specification. Do not commit
or push.
```

## Local decisions

- Provide exactly the two legacy public signatures and make the short form explicitly inclusive
  forward. A separate default, options object, or mode enum would add surface without a current
  need.
- Accept all five current numeric types and reject BOOL. This aligns with the selected capability
  evidence, including later legacy INT64 execution that supersedes one stale legacy error message
  mentioning only INT32.
- Preserve exact input data type rather than promote an accumulation type. Numerical accumulation
  width and overflow are executable policy outside the model expression.
- Preserve `requiresGrad` eligibility metadata for floating input, matching current model
  expression composition. This does not copy the legacy eager builder's forced no-grad callback
  behavior and does not claim a gradient implementation.
- Retain the exact input Shape because cumulative sum is shape-preserving. Leave layout unresolved
  because expression construction does not select output materialization.
- Use a dedicated tensor helper rather than extending aggregate-reduction helpers. Scans preserve
  positions and use a distinct semantic family and parameter set.

## Known limitations

- No values are accumulated and no output storage exists at model construction time.
- No numerical accuracy, overflow, empty-axis, gradient, compiler, ONNX, backend, or kernel
  behavior is implemented.
- Only one-axis cumulative sum exists; other scans and multi-axis forms remain unsupported.

## Validation evidence

Planning read the architecture contract and focused architecture explanations; documentation and
planning rules; roadmap; model capabilities/master plan; tasks 0001, 0002, 0006, 0007, 0011,
0012, 0013, and 0016G; current Tensor, descriptor, factory, provenance, Shape, DataType,
cumulative-sum semantic contracts/tests; Tensor/Compile/Training APIs; glossary; and Java 26 Gradle
configuration.

The legacy branch was read directly. It confirms exactly two public overloads, inclusive-forward
defaults, negative axis normalization, all four modes, exact shape/type preservation, all five
numeric types, BOOL rejection, non-contiguous input, ONNX, CPU, and Metal evidence. Legacy
operation traits, stale error text, eager no-grad callback policy, storage, lowering, and kernels
are excluded or assigned to later owners.

Planning selected two public methods, one three-method package-private helper, one focused test,
and no new package. Existing contracts suffice; no semantic, dependency, build, or architecture
change is required.

Pre-implementation planning validation after synchronizing this task, the model master plan, and
roadmap:

- `git diff --check` passed.
- The targeted trailing-whitespace scan returned no matches across the three planning files.
- All 172 relative Markdown links across the three planning files resolve locally.
- Markdown fence counts are balanced: twelve in this task, two in the master plan, and zero in the
  roadmap.
- All 20 task-template headings are present, together with the focused Capability origin,
  Required contract, and Javadoc requirements sections.
- At that pre-implementation point, task, model master plan, and roadmap consistently identified
  this task as Ready and 0016I as Draft.
- No detailed task-0016I specification exists.
- Repository scope is exactly this task, the model master plan, and the roadmap; no Java, API,
  architecture, Gradle, or other file changed during planning.

Implementation and independent documentation validation:

- The implementation pass added the exact two public `Tensor.cumSum` overloads, the dedicated
  package-private `TensorCumulativeSumExpressions` helper, the seven-test focused expression
  suite, and only the two-method public-surface adjustment in `TensorTest`.
- Clean documentation-focused Codex context
  `019f3281-986c-70a1-abf7-3c03944a1355` applied General style, API and Javadoc style, Planning
  style, and Example format. It independently inspected the implementation diff, final source and
  tests, related foundational and cumulative-sum contracts, generated model Javadoc, bytecode,
  imports, public API references, glossary, planning state, architecture documents, and Gradle
  configuration.
- The two public method Javadocs and package-private helper Javadocs now explain the prefix mental
  model, all four `[1, 2, 3]` modes position by position, positive/negative axes, accepted numeric
  types, BOOL and axis failures, exact metadata retention, fresh identity, unresolved layout,
  absent label/storage, one-input provenance, and all deferred value/gradient/compiler/backend/
  execution behavior. Related `DataType`, `Shape`, `TensorDescriptor`, `TensorFactory`,
  `TensorProvenance`, `Operation`, `CumulativeSumKind`, and `CumulativeSumAttrs` Javadocs remain
  accurate unchanged.
- `docs/api/tensor-api.md` now documents the current two-method expression surface, construction
  boundary, four-mode example with line-level numerical commentary, metadata/ownership behavior,
  deterministic failures, and semantic-versus-executable distinction. `docs/api/compile-api.md`
  now includes cumulative sum in the current expression inventory while retaining compiler
  capture, inference, canonicalization, artifacts, and execution as planned. `docs/glossary.md`
  synchronizes cumulative-sum, normalized-axis, provenance, Tensor, kind, and attribute status.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorCumulativeSumExpressionTest --tests
  io.github.pho001.synaptik.model.tensor.TensorTest` — `BUILD SUCCESSFUL`; 21 tests across the two
  XML suites, with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; 56 XML suites contain 432 tests with zero
  failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`; generated `Tensor.html` contains both
  overloads, their parameters, return facts, failures, four-mode explanation, and boundary text.
- `./gradlew test` — `BUILD SUCCESSFUL`; the repository lifecycle completed 36 actionable tasks,
  with one executed and 35 up-to-date and no failing task.
- `javap -p -c -s` confirms each public overload delegates exactly once, the short overload passes
  `false, false`, the complete overload forwards both flags, and the helper has exactly one private
  constructor plus `apply`, `validateNumericInput`, and `create`. Bytecode confirms one axis
  normalization and one descriptor/operation/provenance/factory construction path in the required
  order, with no fields or hidden state.
- Import and source inspection found only model-owned datatype, Shape, operation, scan, tensor,
  and JDK collection/null-check dependencies. No graph, compiler, planning, prepare, runtime,
  backend, engine, trace, training, storage-access, or value-execution dependency was introduced.
- Local Markdown target/anchor validation passed for all 262 relative links in the six changed
  documentation/planning files. Markdown fences are balanced, targeted trailing-whitespace and
  stale-status scans passed, the four numerical examples were recalculated, generated Javadoc was
  reviewed, and `git diff --check` passed.
- Final repository scope is exactly the authorized ten paths: two production files, two tests,
  Tensor API, Compile API, glossary, this task, model master plan, and roadmap. Task 0016H is
  synchronized as Complete; task 0016I remains Draft, and no `0016i-*` specification exists.
- Training API and capabilities remain accurate unchanged: the former contains only planned
  training concepts, and the latter already inventories cumulative-sum modes while distinguishing
  model expression representation from compiler/backend execution. Foundational, cumulative-sum
  semantic, and other expression contracts remain accurate because their signatures and ownership
  did not change.
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests,
  backend-conformance tests, integration tests, Gradle configuration, and other modules remain
  accurate unchanged because this task changes no dependency direction, lifecycle, backend or
  end-to-end behavior, build configuration, or code outside `modules/model`.
- No sandbox-only Gradle lock failure occurred during this documentation pass. No validation was
  skipped, and no blocker remains.

## Implementation notes

- Added exactly two public delegating methods and one stateless three-method construction helper.
- The helper accepts all five numeric types, rejects BOOL before axis validation, normalizes one
  axis, preserves exact Shape/type/eligibility metadata with unresolved layout, records exact
  one-input provenance, and delegates once to central derived identity allocation.
- Added the focused seven-test contract suite and adjusted only Tensor's exact public API count and
  method set in the existing Tensor test.
- The documentation-focused pass finalized Javadocs, Tensor API, Compile API, glossary, task
  evidence, model master plan, and roadmap without adding executable behavior.

## Completion summary

- Completed changes: Implemented and documented public cumulative-sum expression construction for
  inclusive/exclusive and forward/reverse modes.
- Files changed or created: Exactly two production Java files, two tests, Tensor API, Compile API,
  glossary, this task, model master plan, and roadmap.
- Tests and validation: Focused 21/21, model 432/432 across 56 suites, generated model Javadoc,
  root tests, bytecode/reflection/import/manual checks, Markdown links/anchors/fences/whitespace,
  exact scope/status checks, and `git diff --check` passed.
- Documentation-agent review: Clean Codex context
  `019f3281-986c-70a1-abf7-3c03944a1355` completed the mandatory independent pass.
- Documentation impact: Tensor and Compile API references, glossary, task, master plan, and roadmap
  now describe current cumulative-sum expression construction and its non-executable boundary.
- Javadoc review: Public and package-private cumulative-sum Javadocs are complete; related
  foundational and semantic contracts remain accurate unchanged.
- Glossary impact: Current cumulative-sum construction, normalized-axis ownership, provenance, and
  Tensor/kind/attribute status are synchronized.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0016H. Task 0016I remains Draft without a detailed
  specification.

Status: Complete

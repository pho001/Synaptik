# Task 0019A: Modern Activation Semantics and Tensor Expressions

## Status

Complete

## Goal

Add exact Gaussian error linear unit (GELU), the conventional tanh GELU approximation, and
sigmoid linear unit (SiLU) as one cohesive backend-independent unary activation capability. The
task introduces three parameterless first-class semantic kinds and three zero-argument public
`Tensor` methods while preserving the current floating unary descriptor, producer, provenance,
and failure contracts.

This task records storage-free model metadata only. It does not evaluate activations, choose an
algorithm, define an accuracy tolerance or gradient rule, capture a graph, lower an operation, or
claim compiler, backend, runtime, or end-to-end support.

## Why the former 0019A frontier is split

The former row combined three independently reviewable capability boundaries. GELU and SiLU are
same-shape floating unary activations and fit the existing unary family. Embedding is a rank-two
table lookup that can be expressed by canonical axis-zero gather. One-hot encoding instead changes
rank and result type and cannot use the unary construction contract. Combining their kinds,
public APIs, validation, tests, and documentation would exceed normal task granularity and obscure
three different ownership decisions.

Task 0019A therefore owns only modern activations. Draft task 0019A1 owns embedding, and Draft task
0019A2 owns one-hot. Existing Draft tasks 0019B through 0019E retain their identifiers, meanings,
and order after these inserted follow-ups.

## Mental model and examples

All three activations transform each floating input coordinate independently and retain the input
Shape:

```text
GELU exact:       x * Phi(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
GELU tanh target: 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
SiLU:             x * sigmoid(x) = x / (1 + exp(-x))
```

For conceptual finite inputs:

```text
input Shape [3], FLOAT32: [-1.0, 0.0, 1.0]
gelu() target:             about [-0.158655, 0.0, 0.841345]
silu() target:             about [-0.268941, 0.0, 0.731059]
```

The decimal values explain mathematical meaning only. Model construction neither reads those
inputs nor computes the results, and this task does not select rounded expected bits or an
execution tolerance.

## Scope

- Add exactly `GELU`, `GELU_TANH_APPROXIMATION`, and `SILU` to `UnaryElementwiseKind`.
- Keep all three parameterless with the existing exact `NoOperationAttrs`, one-input, one-output
  unary signature.
- Add exactly `Tensor.gelu()`, `Tensor.geluTanhApproximation()`, and `Tensor.silu()`.
- Reuse `TensorUnaryExpressions.apply` unchanged in behavior, expanding only its documented kind
  count.
- Accept only BFLOAT16, FLOAT32, and FLOAT64 and preserve exact input type, Shape reference, and
  gradient-eligibility request.
- Fix the mathematical and special-value policies below without prescribing an implementation.
- Construct one fresh unresolved-layout result with exact one-input provenance, no label, and no
  storage for every valid call.
- Extend the focused unary semantic and public-expression tests plus the exact public API
  inventories.
- Finalize Javadocs, Tensor/Compile API references, glossary, capabilities, and planning records
  through a mandatory separate clean-context documentation pass.

## Out of scope

- embedding, padding indices, one-hot, dropout, graph random-number-generator state, sorting,
  top-K, linear, attention, normalization, losses, convolution, or pooling
- ReLU, sigmoid, tanh, or any other already implemented unary method; completed tasks
  0014C–0014D own the current `RELU`/`relu()` contract, which this task must not duplicate or alter
- a `swish` alias, an approximation enum or attribute, string-valued modes, configurable GELU
  coefficients, SiLU beta, or additional activation variants
- composition into ERF, TANH, SIGMOID, scalar, binary, or WHERE producer chains
- eager value reads, host-storage allocation, result storage, resolved result layout, mutation,
  or numerical evaluation
- a fixed unit-in-the-last-place bound, relative-error tolerance, rounding mode, NaN payload,
  machine instruction, polynomial/rational evaluation scheme, or backend route
- compiler capture, canonicalization, decomposition, constant folding, fusion, autograd, gradient
  formulas, subgradient conventions, or backward operations
- capability providers, backend lowering, kernels, runtime execution, conformance, integration,
  or end-to-end support
- changes to operation attributes/signatures, `DataType`, Shape/layout/descriptor,
  producer/provenance/factory foundations, Gradle, another module, `ARCHITECTURE.md`, or focused
  architecture documentation

## Exact semantic and API contract

### Kinds, attributes, signatures, and public methods

Extend the existing enum with exactly:

```java
GELU,
GELU_TANH_APPROXIMATION,
SILU
```

Every kind continues to use the family's one stable signature:

```java
OperationSignature.fixed(NoOperationAttrs.class, 1, 1)
```

Every operation uses exactly `NoOperationAttrs.INSTANCE`. The tanh approximation is a separate
semantic kind, not an attribute variant of `GELU`, because it names a different portable function
with different finite results and no tunable state. Keeping it typed lets compiler and backend
work preserve the caller's selected target without string inspection. Add no activation-specific
attribute type or signature variant.

Add exactly these public methods:

```java
public Tensor gelu()
public Tensor geluTanhApproximation()
public Tensor silu()
```

`gelu()` selects the exact error-function definition. `geluTanhApproximation()` selects the fixed
formula above; it is not permission for a backend to substitute an arbitrary fast GELU.
`silu()` is the sole public spelling. Do not add `swish`, because two names for the same exact
parameterless semantic would add API surface without adding meaning or compatibility value.

### First-class rather than composed semantics

All three are first-class unary kinds. Literal expression composition is not semantically
equivalent at important special values: straightforward `x * sigmoid(x)` and
`x * (1 + erf(...))` chains encounter `infinity * zero` at negative infinity and may produce NaN
instead of the selected limiting signed zero. Composition would also hide the caller's exact
versus tanh GELU choice behind a larger producer chain and constrain later accuracy-preserving
compiler/backend treatment. First-class meaning fixes the mathematical target while leaving
decomposition, fusion, and algorithm choice to compiler and backend owners.

### Data type, Shape, layout, and gradient eligibility

Each method accepts exactly BFLOAT16, FLOAT32, or FLOAT64. INT32, INT64, and BOOL fail through the
existing unary helper message; no implicit cast is inserted.

Every result retains:

- the exact input `DataType`;
- the exact immutable input `Shape` reference, including scalar, zero-sized, named-symbolic, and
  expression dimensions; and
- the exact input `requiresGrad` value as model-level gradient eligibility only.

Every result uses `Optional.empty()` layout even when the input layout is resolved. Construction
does not claim a derivative, differentiability at every coordinate, a subgradient convention, or
compiler autograd support. Compiler-owned gradient rules may later consume the preserved
eligibility metadata.

### Numerical and special-value meaning

The mathematical definitions are evaluated over the represented floating type by a future
execution owner. They select these portable special-value classes:

| Kind | Negative infinity | Negative zero | Positive zero | Positive infinity | NaN |
|---|---|---|---|---|---|
| `GELU` | negative zero | negative zero | positive zero | positive infinity | NaN |
| `GELU_TANH_APPROXIMATION` | negative zero | negative zero | positive zero | positive infinity | NaN |
| `SILU` | negative zero | negative zero | positive zero | positive infinity | NaN |

The infinity entries are the continuous extensions of the selected functions, not a required
literal order of primitive floating operations. Finite overflow, underflow, subnormal handling,
rounding, and NaN payload propagation follow future per-data-type execution and conformance
contracts. This task promises neither bitwise equality nor a fixed accuracy bound and chooses no
backend algorithm.

### Result metadata and provenance

Every successful method creates:

- one fresh `TensorDescriptor` with exact input type and Shape, empty layout, and unchanged
  gradient eligibility;
- one `Operation` with the exact selected unary kind and `NoOperationAttrs.INSTANCE`;
- one fresh single-output `TensorProducer` retaining exactly `[this]` and the exact result
  descriptor; and
- one fresh unlabeled, storage-free Tensor with output index zero and one newly allocated ID.

Inputs and all input metadata, provenance, labels, storage, and IDs remain unchanged. Repeated
calls create distinct Tensor and producer identities. Chaining records the immediately preceding
Tensor as the exact input. No canonicalization or eager simplification occurs.

### Validation order, messages, and ID effects

The public zero-argument methods delegate directly to `TensorUnaryExpressions.apply(this, kind)`.
That helper retains this exact order:

1. null-check `input`, then `kind`, with the parameter name as message;
2. read the descriptor type and require a floating type;
3. construct the result descriptor;
4. construct the exact parameterless operation; and
5. delegate exactly once to `TensorFactory.createDerived`.

The existing task-owned type failure remains exact:

```text
input must be a floating data type, but was <dataType>
```

Every null/type failure before step 5 consumes no Tensor ID. Only the final factory delegation may
allocate an ID; identifier exhaustion retains the existing exact message and behavior. Focused
tests inspect the shared ID counter only for this existing concrete failure-side-effect risk.

### Operation-signature compatibility

The existing unary signature remains the sole family variant and accepts all nineteen enum
constants through exact `NoOperationAttrs` class matching. `Operation` and `TensorProducer`
continue to validate the one-input, one-output occurrence. Add no registry, schema, reflective
dispatch, backend support, result facts, or gradient metadata to the signature.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Planning guide](../../../planning-guide.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)

## Architecture constraints

- Work remains entirely inside `modules/model` plus its documentation/planning records.
- `Tensor` remains public mutable API state and is not graph IR.
- `Operation` and unary kinds record backend-independent meaning only and expose no support,
  lowering, algorithm, or execution route.
- Package direction remains `model.tensor -> model.operation.elementwise.unary`; the operation
  package must not import Tensor, graph, compiler, backend, or runtime state.
- Compiler later owns capture, graph-wide validation, canonicalization/decomposition, and gradient
  construction. Backend prepare owns lowering, numerical implementation, fusion, and kernel route.
- No architecture, dependency, lifecycle, module-boundary, Gradle, or cross-module change.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.operation`
- `io.github.pho001.synaptik.model.operation.elementwise.unary`
- `io.github.pho001.synaptik.model.tensor`

Packages added or changed:

- No package is added. The existing unary operation package gains three enum constants, and the
  existing Tensor package gains three facade methods.

Type placement:

- `io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind` — existing
  public owner of parameterless floating-preserving unary mathematical and activation identities.
- `io.github.pho001.synaptik.model.tensor.TensorUnaryExpressions` — existing package-private
  shared floating-unary validation and derived-construction boundary.
- `io.github.pho001.synaptik.model.tensor.Tensor` — existing public fluent API owner.

Tests mirror the production packages whose typed and package-private surfaces they inspect.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/unary/UnaryElementwiseKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorUnaryExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/elementwise/unary/UnaryElementwiseKindTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorUnaryElementwiseTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java` — update only
  the exact public Tensor API inventory and method count from 157 to 160.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java` —
  update only the shared exact public Tensor method count from 157 to 160.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java` —
  update only the stale shared exact public Tensor method count from 157 to 160; preserve every
  MATMUL-specific assertion and behavior.

Documentation/planning:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless the final implementation makes an existing statement inaccurate:
Training API; scalar/binary/where/cast/indexing/factory/producer/provenance/signature contracts;
architecture/ADRs/tests; conformance/integration; Gradle; other modules.

## Maximum scope

At most three production, five test, and seven documentation/planning files: exactly 15 paths.
`TensorUnaryExpressions.java` changes only for its kind-count documentation.
`Tensor.java` changes only for the three exact methods, required import/reference text, and its
Javadoc. `TensorMatmulExpressionTest.java` changes only for its stale total public Tensor method
count. No new Java type or package is permitted. Stop for a sixteenth path, another source or test
contract, any executable behavior outside model, cross-module work, or architecture change.

## Javadoc and documentation requirements

- Update unary kind, helper, and Tensor Javadocs for the three exact semantics, formula choices,
  floating domain, Shape/type/eligibility/layout retention, special values, provenance, freshness,
  validation, failure side effects, and owning-layer boundaries.
- Document every new public method with meaningful description, `@return`, all applicable
  `@throws`, and no invented execution or gradient promise.
- Update Tensor API's current inventory, unary table, formulas, special-value table, and metadata
  example or add a focused planned-versus-current example that observes only model metadata.
- Update Compile API only enough to list these current expressions while preserving the explicit
  planned capture/decomposition/gradient boundary. Do not claim compiler support.
- Update the glossary's unary-family inventory and define GELU and SiLU only if the final prose
  uses them as reusable project terms; distinguish SiLU from the omitted `swish` alias.
- Keep capabilities, task, master plan, and roadmap synchronized. Record reasoned no-change
  conclusions for Training API, related contracts, architecture/tests, conformance/integration,
  Gradle, and other modules.

## Acceptance criteria

- `UnaryElementwiseKind` contains exactly nineteen values, adding only `GELU`,
  `GELU_TANH_APPROXIMATION`, and `SILU` with the unchanged exact no-attributes one-input,
  one-output signature.
- Tensor adds exactly `gelu()`, `geluTanhApproximation()`, and `silu()`; its exact public method
  count becomes 160, and no `swish` or configurable activation overload exists.
- Exact GELU, fixed tanh-approximation GELU, and exact SiLU meanings and special-value classes
  match this specification without composition producers or algorithm promises.
- Every floating type and Shape state preserves exact type/Shape/eligibility metadata, leaves
  layout unresolved, and creates exact parameterless one-input, output-index-zero provenance.
- Every valid call is fresh, unlabeled, storage-free, and non-canonicalized; inputs remain
  unchanged.
- INT32, INT64, and BOOL fail with the existing exact message and no ID consumption. Null helper
  tests retain input-then-kind order; only final factory exhaustion may consume/fail an ID.
- Tests validate kinds, signatures, API shape, descriptor/provenance/freshness, and failures
  without pretending to execute activation values.
- No embedding, one-hot, swish alias, attributes, gradients, compiler/backend/runtime behavior,
  dependencies, Gradle, architecture, or other-module work lands.
- Focused tests, the final model suite after Java stabilizes, Javadoc, documentation/link, exact
  15-path scope, status, formatting, newline, fence, and whitespace checks pass. If the first
  final suite exposes only a stale total-method inventory outside the original focused paths, one
  authorized scope-repair rerun may follow that exact inventory correction and must be recorded.
- A separate clean-context documentation pass finalizes all authorized documentation and records
  reused Java evidence plus reasoned no-change conclusions.
- Task 0019A becomes Complete only after all evidence is recorded. Task 0019 remains Complete;
  0019A1, 0019A2, and existing 0019B through 0019E remain Draft without detailed specifications.

## Tests / validation

Required focused command:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKindTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorUnaryElementwiseTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorBinaryArithmeticTest
```

After executable Java stabilizes, normally run exactly one final model suite:

```bash
./gradlew :modules:model:test
```

The first final suite exposed the stale total public Tensor method count in
`TensorMatmulExpressionTest`. The authorized 15-path scope repair changes only 157 to 160 there;
run one necessary final model-suite rerun afterward and record that reason. Do not perform any
additional model-suite rerun.

The separate documentation pass runs after final Javadoc edits:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

Also validate local Markdown links and anchors, balanced fences, terminology, generated Javadoc,
final newlines, trailing whitespace, exact 15 paths, package placement, exact nineteen-kind and
160-method surfaces, forbidden aliases, synchronized status, the model frontier, and the absence
of detailed 0019A1-or-later specifications. The documentation pass reuses the successful Java
evidence and must not rerun either Java command unless it changes executable Java behavior or
records a concrete cross-check reason.

Repository-wide validation is deferred to the selected-modern-operation-family checkpoint after
task 0022 and to CI. This task changes one existing family inside one module and does not alter a
dependency, architecture boundary, shared build configuration, or multiple modules.

## Dependencies

- Tasks 0005–0006 and 0018K: operation kinds, typed attributes, exact family-owned signatures,
  and occurrence-cardinality validation.
- Tasks 0011–0013 and 0018L: public Tensor construction, identity, producer, and indexed
  provenance contracts.
- Tasks 0014C–0014D: the parameterless unary semantic family and public floating-unary
  construction boundary.
- Task 0018P: final portable unary vocabulary and removal of uncontracted fast variants.

## Follow-up tasks

- Draft task 0019A1 owns `weights.embedding(indices)` as a rank-two floating embedding-table
  convenience that validates its public table contract and delegates to exact
  `weights.gather(indices, 0)` provenance. It will add no padding index, embedding kind, or gradient
  rule; compiler may reject captured constant bounds failures, while backend preparation and the
  prepared executable own safe handling of bound execution-time index values. Runtime does not
  inspect the original Gather operation.
- Draft task 0019A2 owns `indices.oneHot(long depth)` as a trailing-axis, BOOL-valued first-class
  encoding with `false` off, `true` on, false gradient eligibility, positive static depth, and
  all-false rows for negative or too-large indices. It will add no output-type, axis, or on/off
  configuration and no eager TensorFactory depth vector. Its selected Draft semantic shape is
  `OneHotKind.ONE_HOT` plus `OneHotAttrs(long depth)` with a fixed one-input, one-output signature.
- Existing tasks 0019B through 0019E retain their Draft identities and scopes.
- In particular, unchanged Draft task 0019B remains the sole owner of dropout because dropout
  requires explicit state-consuming/state-producing graph RNG semantics rather than another
  deterministic unary activation.
- No detailed follow-up specification is created by this planning task.

## Architecture impact

Expected impact: None.

The task extends model-owned backend-independent unary semantics and the public Tensor expression
surface without changing module ownership, dependency direction, lifecycle contracts, or backend
responsibility. If implementation requires any architecture change, another package/type, or a
different semantic boundary, stop and report before editing architecture or expanding scope.

## Implementation prompt

Use this prompt in a separate agentic task/thread with clean context:

```text
You are working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, documentation/planning rules, roadmap, model capabilities/master
plan, tasks 0014C–0014D/0018K/0018L/0018P/0019A, Tensor and Compile APIs, glossary, and every
affected or review-only source/test named by task 0019A in full.

Implement task 0019A exactly. Add only first-class exact GELU, fixed tanh-approximation GELU, and
SiLU parameterless unary semantics plus gelu(), geluTanhApproximation(), and silu(). Preserve the
exact current floating-unary validation, metadata, producer/provenance, freshness, and ID-side-
effect contracts. Add no composition, swish alias, attribute/configuration, embedding, one-hot,
gradient, compiler/backend/runtime behavior, dependency, Gradle, architecture change, or later
spec. Stay within the exact fifteen paths and stop on scope or architecture conflict. Do not
commit or push.

Run the focused tests and the final model suite after executable Java stabilizes. The authorized
15-path repair permits one necessary rerun only if the first suite fails solely on the stale
MATMUL test's total Tensor method count; record that reason and perform no additional model rerun.
Then hand the actual diff and exact evidence to a separate clean-context documentation-focused
agent in the same overall change. That agent must inspect final source/tests, finalize permitted
Javadocs/Tensor API/Compile API/glossary/capability/planning documents, run model Javadoc and
documentation/scope checks, and must not repeat successful Java tests unless executable behavior
changes or it records a concrete reason.

Mark 0019A Complete only after both passes succeed. Leave 0019A1, 0019A2, and 0019B–0019E Draft
without detailed specifications.
```

## Local decisions

- GELU defaults to the exact error-function target; the tanh approximation is available only by
  the explicit `geluTanhApproximation()` spelling.
- Exact and tanh GELU are separate parameterless kinds because they are different fixed functions,
  not one kind with runtime configuration.
- GELU and SiLU are first-class because literal primitive chains do not preserve the selected
  negative-infinity limit and would erase useful semantic identity.
- `silu` is canonical and no `swish` alias is added.
- Model retains gradient eligibility but defines no gradient or subgradient rule.

## Known limitations

- No execution owner, per-type accuracy tolerance, correct-rounding promise, or backend support is
  included.
- No configurable approximation, coefficients, activation parameters, or compatibility aliases
  are included.
- Gradient eligibility does not imply that compiler autograd can differentiate these operations.
- Embedding and one-hot remain separate Draft follow-ups.

## Validation evidence

Planning context: clean documentation/planning task `/root/plan_0019a`.

- Read the architecture contract and focused architecture index/module/dependency/lifecycle/
  training documents; documentation rules and General/Planning/Example profiles; planning guide,
  roadmap, model capability baseline/master plan; completed semantic, Tensor, factory, indexing,
  provenance, signature-hardening, reset, and MATMUL task history; current relevant source/tests;
  Tensor/Compile/Training API references; glossary; and current public API inventories.
- Confirmed the current unary family has sixteen parameterless one-input/one-output kinds, all
  using exact `NoOperationAttrs`, and the shared helper already provides the required floating
  validation, descriptor, provenance, freshness, and no-ID local-failure behavior.
- Confirmed Tensor currently has 157 public methods, so three additions produce the exact 160-
  method surface and require only the two existing inventory/count updates named above.
- Confirmed literal GELU and SiLU primitive chains do not preserve the selected negative-infinity
  limiting result because they encounter zero multiplied by negative infinity. This supplies the
  semantic reason for first-class kinds despite the earlier provisional convenience label.
- Confirmed embedding maps exactly to axis-zero Gather for a rank-two table, whereas one-hot needs
  distinct rank-changing, fixed-BOOL semantics; they are separate Draft rows without detailed
  specifications.
- Planning-only validation found exactly the four authorized paths: capabilities, model master
  plan, this new task, and roadmap. Source/build/architecture/other-module scans found no changed
  path. Status scans found exactly one Ready model task and one matching Ready row in each queue;
  0019A1, 0019A2, and 0019B–0019E remain Draft, with no detailed later specification.
- A targeted Ruby Markdown check resolved all 345 local links and fragments across the four
  changed files. Fence parity, final-newline, trailing-whitespace, and exact-path checks passed.
  `git diff --check` passed with no output; the new untracked task separately passed newline,
  whitespace, and fence checks because ordinary diff checks do not include untracked files.
- Implementation, Java tests, Javadoc generation, and executable documentation validation remain
  deferred to the required implementation and documentation contexts.

Implementation context: clean task `/root/task_0019a_implementation`.

- The focused command in this task's validation section passed before the authorized scope repair:
  `BUILD SUCCESSFUL`, with three tasks executed.
- The first final `./gradlew :modules:model:test` ran 758 tests and failed only because
  `TensorMatmulExpressionTest` line 69 retained the stale total public Tensor method count of 157.
  No activation behavior failed. The user authorized expanding the exact scope from 14 to 15 paths
  for that count-only repair.
- After changing only that stale count to 160, the necessary final
  `./gradlew :modules:model:test` rerun passed: `BUILD SUCCESSFUL`, three actionable tasks, two
  executed and one up-to-date. Executable Java did not change afterward.

Documentation context: clean task `/root/task_0019a_implementation/docs_0019a` using the General,
API/Javadoc, Planning, and Example profiles.

- Independently read the required architecture, documentation, planning, API, glossary, source,
  tests, and directly relevant unary/provenance/signature contracts and inspected the actual
  shared-tree diff.
- Finalized the three production Javadocs and the Tensor/Compile API, glossary, capabilities, task,
  master-plan, and roadmap content. The public references now distinguish exact GELU, the fixed
  tanh approximation, and canonical SiLU; document formulas and function-level continuous
  extensions; and separate current metadata construction from planned compiler capture,
  decomposition, gradient construction, backend preparation, and execution.
- `./gradlew :modules:model:javadoc` passed after final Javadoc review: `BUILD SUCCESSFUL`, two
  actionable tasks, one executed and one up-to-date.
- A targeted Ruby Markdown check resolved all 509 local links and fragments across the seven
  documentation/planning paths. Separate format checks passed final-newline, trailing-whitespace,
  and balanced-fence checks for all seven paths.
- Source/scope checks found exactly nineteen unary kinds, exactly 160 public Tensor methods, and
  exactly the three new zero-argument methods. Production scans found no `swish`, embedding,
  one-hot, configurable activation attributes, or later-task implementation/specification.
- Final status checks found task 0019 and task 0019A Complete; 0019A1, 0019A2, and 0019B–0019E
  remain Draft without detailed specifications; no model task is currently Ready.
- Final combined scope is exactly 15 paths: three production, five tests, and seven documentation/
  planning paths. `git diff --check` passed with no output; the untracked task separately passed
  the same whitespace/newline checks.
- Training API requires no change because this task preserves eligibility metadata but adds no
  gradient rule, training workflow, or trainable-state contract. Related scalar, binary, where,
  cast, indexing, factory, producer, provenance, and signature documents require no change because
  their shared construction and cardinality contracts remain unchanged. Architecture documents,
  ADRs, and architecture tests require no change because module ownership and dependency rules do
  not change. Backend conformance and integration tests require no change because no executable
  behavior or backend support is added. Gradle and other modules require no change because the
  task adds no dependency, build configuration, or cross-module API.

## Implementation notes

- Added only `GELU`, `GELU_TANH_APPROXIMATION`, and `SILU` to the existing parameterless unary
  family and only `gelu()`, `geluTanhApproximation()`, and `silu()` to Tensor.
- Reused the existing floating-unary helper without executable modification; its Javadoc count is
  now nineteen and its validation, metadata, provenance, freshness, and ID-side-effect contracts
  remain unchanged.
- Repaired the shared MATMUL API-inventory test only after the first final suite exposed its stale
  count, as explicitly authorized. No MATMUL-specific assertion changed.
- No Java behavior changed during the documentation-focused pass.

## Completion summary

- Completed changes: added first-class exact GELU, fixed tanh-approximation GELU, and SiLU semantic
  identities plus their three public storage-free Tensor expression methods, tests, Javadocs, and
  synchronized public/planning documentation.
- Files changed or created: `UnaryElementwiseKind.java`, `TensorUnaryExpressions.java`,
  `Tensor.java`; `UnaryElementwiseKindTest.java`, `TensorUnaryElementwiseTest.java`, `TensorTest.java`,
  `TensorBinaryArithmeticTest.java`, `TensorMatmulExpressionTest.java`; `tensor-api.md`,
  `compile-api.md`, `glossary.md`, model `capabilities.md`, this task, model `master-plan.md`, and
  `roadmap.md`.
- Tests and validation: reused the successful focused and repaired final model-test evidence above;
  final model Javadoc, 509 local-link/anchor checks, formatting, exact-scope, surface, terminology,
  status, forbidden-feature, and `git diff --check` checks passed.
- Documentation-agent review: completed in clean context
  `/root/task_0019a_implementation/docs_0019a` under the required profiles.
- Documentation impact: Tensor API, Compile API, glossary, capabilities, task, master plan, and
  roadmap are synchronized with current model construction and planned downstream ownership.
- Javadoc review: all three affected production contracts are detailed and consistent; no further
  correction was required after independent review.
- Glossary impact: added reusable GELU and SiLU definitions and the exact-versus-approximation and
  canonical-SiLU/no-alias distinctions.
- Unresolved issues: None.
- Follow-up required: None for task 0019A. Separate 0019A1, 0019A2, and 0019B–0019E work remains
  Draft and outside this task.

Status: Complete

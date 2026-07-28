# Task 0025C: Portable Functional-Scatter Reduction Semantics

## Status

Complete

## Goal

Close the smallest model-owned forward-semantics prerequisite needed before Compiler task 0005C
can plan gradients for configurable functional scatter.

Make the current `SCATTER_ELEMENTS` and `SCATTER_ND` contracts state one portable represented-value
meaning for `ScatterReduction.MUL`, `MIN`, and `MAX`. The contract must fix base-value
participation, duplicate-target grouping, empty update groups, fixed data-type behavior, floating
special values, and signed-integral overflow without selecting a traversal, atomic schedule,
backend algorithm, or derivative convention.

The semantic relationship is:

```text
data value plus every update addressed to one logical result coordinate
  -> one model-owned candidate group
  -> MUL, MIN, or MAX represented value in the unchanged data type

fixed forward meaning
  -> later Compiler 0005C chooses derivative formulas and boundary policy separately
```

This is expected to be Javadoc and explanatory documentation only. It adds no evaluator,
operation, attribute, Tensor method, helper behavior, test-only numerical implementation,
compiler formula, backend behavior, or execution.

## Scope

- Finalize `ScatterReduction` Javadoc for the exact `MUL`, `MIN`, and `MAX` contracts below.
- Finalize the existing axis-scatter and Scatter-ND kind/attribute Javadocs so both configurable
  families use that same contract.
- Finalize the existing public `Tensor.scatterElements` and `Tensor.scatterNd` Javadocs and their
  package-private construction-helper Javadocs without changing executable construction.
- Clarify the Tensor API, Compile API, and glossary atomically.
- Record this task as `Complete` in the model master plan and roadmap after validation passes.
- Make Draft Compiler 0005C depend on Model 0025C without creating its detailed task file or
  promoting it.
- Record the downstream compiler planning decision that a future Compiler 0005C may construct one
  separate stable `ARGSORT` Tensor occurrence with exactly matching sort attributes as the
  ordinary backward formula for one-output `SORT`.
- Preserve Compiler 0005D, 0005E, and 0006 as concise `Draft` rows without detailed task files.
- During implementation, synchronize this task, the model master plan, compiler master plan, and
  roadmap only after all documentation and validation acceptance criteria pass.

## Fixed forward semantic contract

### Logical target groups

For one valid functional-scatter occurrence and one logical result coordinate `c`, let `U(c)` be
the logical multiset of update scalar values whose indices address `c`.

- The base value `data[c]` participates exactly once.
- Every logical update coordinate that addresses `c` contributes its update value exactly once.
- Duplicate targets therefore contribute multiple members, including when their represented
  values are equal.
- `SCATTER_ELEMENTS` forms the target coordinate by replacing the selected coordinate of each
  logical indices/updates coordinate with its index value.
- `SCATTER_ND` applies each index tuple to the indexed data prefix. When one tuple addresses a
  suffix slice, each logical scalar in the matching update slice contributes to the corresponding
  scalar result coordinate; duplicate tuples create duplicate contributions at every matching
  suffix coordinate.
- If `U(c)` is empty, the result at `c` is the exact unchanged representation of `data[c]`. A
  zero-element update domain therefore leaves every base representation unchanged.

For `MUL`, `MIN`, and `MAX`, the result is a symmetric reduction over
`{data[c]} multiset-union U(c)`. Logical update encounter order, physical layout, strides,
backend traversal, parallel grouping, and atomic scheduling are not observable. This is an exact
grouping rule, not an implementation-order rule. Implementations may reassociate or parallelize
only while reproducing the selected abstract result and future conformance requirements.

### Semantics matrix

| Reduction | FLOAT64, FLOAT32, BFLOAT16 | INT32, INT64 | Empty `U(c)` |
|---|---|---|---|
| `MUL` | Abstract product in the unchanged result format, with the floating rules below | Exact-width two's-complement product modulo `2^32` or `2^64` | Exact base representation |
| `MIN` | NaN-propagating numeric minimum; opposite signed zeros select negative zero | Ordinary signed minimum | Exact base representation |
| `MAX` | NaN-propagating numeric maximum; opposite signed zeros select positive zero | Ordinary signed maximum | Exact base representation |

`BOOL` remains ineligible for arithmetic scatter reductions under the current public construction
contract. Data and updates already have the same exact `DataType`; this task adds no promotion,
widened output, saturation, conversion, or accumulation attribute.

### Floating multiplication

For a non-empty update group, floating `MUL` follows the existing portable product meaning used by
ordinary reductions and cumulative product:

- any NaN factor produces NaN, without a payload, sign, signaling, or source-factor promise;
- a group containing both any zero and any infinity produces NaN;
- otherwise, an infinity produces infinity and a zero produces zero, with sign determined by the
  parity of negative factors, including negative zero and negative infinity;
- otherwise, finite non-zero factors denote their exact mathematical product, rounded to the
  unchanged result format with round-to-nearest, ties-to-even;
- finite overflow produces signed infinity, while underflow, subnormal results, and signed zero
  follow that result-format rounding and multiplication sign; and
- regrouping and an equal-or-wider implementation intermediate are lawful only when they conform
  to this abstract target and the future backend-conformance tolerance. Narrower accumulation,
  saturation, a model-visible promoted type, a fixed instruction sequence, NaN payload
  preservation, and a bitwise cross-backend algorithm are not selected.

FLOAT64 and FLOAT32 use their represented IEEE-754 binary64 and binary32 values and result
formats. BFLOAT16 uses the value represented by its exact current 16-bit storage and rounds the
abstract result to BFLOAT16 with the existing round-to-nearest, ties-to-even conversion contract.

The exact-base rule for an empty `U(c)` is stronger than applying a synthetic multiplicative
identity: no arithmetic occurs, so an existing NaN payload or other base representation is not
canonicalized merely because no update addresses that coordinate.

### Floating minimum and maximum

For a non-empty update group, floating `MIN` and `MAX` reuse completed Model 0025A's
represented-value extrema meaning across the entire candidate group:

- if any candidate is NaN, the result is NaN, without a payload, sign, source-candidate, or
  bitwise-result promise;
- if both zero signs occur, `MIN` produces negative zero and `MAX` produces positive zero,
  independent of how many occurrences contribute;
- infinities and unequal non-NaN values use ordinary numeric order; and
- equal non-zero candidates produce that numeric value without promising which candidate's
  representation or identity is selected.

No bounded empty identity is used because the base candidate is always present. When no update
addresses a coordinate, the exact-base rule applies instead.

### Signed-integral behavior

- INT32 `MUL` is multiplication modulo `2^32`; INT64 `MUL` is multiplication modulo `2^64`.
- Overflow wraps in the unchanged fixed width. It does not throw, saturate, widen, or become
  undefined.
- Modular multiplication is independent of candidate order and grouping.
- INT32 and INT64 `MIN`/`MAX` use ordinary signed order and cannot overflow.
- Empty update groups preserve the exact base bit pattern rather than applying an identity.

### `NONE` and `ADD` no-change boundary

This task does not redefine the completed replacement or addition contracts:

- `NONE` still replaces an addressed base with its single exact update representation and requires
  unique targets. Duplicate targets remain invalid rather than first-write or last-write ordered.
- `ADD` still combines the base and every addressed update, including duplicate accumulation.
  Existing fixed-width modular signed-integral addition and permitted floating reassociation/no
  bitwise-order guarantee remain unchanged.
- `SCATTER_ADD` remains the separate fixed-add Gather-compatible operation. This task changes
  neither its Shape relation nor its represented-value contract.

No new general update order is introduced for `NONE` or `ADD`.

## Out of scope

- any Java declaration, import, annotation, executable statement, validation branch, failure
  message, operation kind, signature, attribute component, Tensor method, helper method, data type,
  promotion rule, Shape rule, descriptor, provenance, or test behavior
- `SCATTER_ADD` semantics, `NONE` uniqueness, `ADD` numerical policy, index bounds, negative-index
  policy, duplicate detection, or value-aware validation
- a numerical-policy type, registry, service, selectable mode, generic evaluator, backend utility,
  runtime utility, or test-only fake evaluator
- derivative formulas or policies for product zeros, infinities, NaNs, extrema ties, signed zero,
  duplicate sharing, discontinuities, or invalid forward occurrences
- hidden sort outputs, a public sort result carrier, a multi-output `SORT`, changes to
  `Tensor.sort`/`argsort`, producer-output changes, or compiler implementation
- graph capture, inference, canonicalization, constant folding, planning, lowering, execution,
  backend conformance implementation, runtime, prepare, engine, training, ONNX, or trace work
- changing `ARCHITECTURE.md`, an ADR, focused architecture documentation, module boundaries,
  dependencies, Gradle, Java 26 configuration, architecture tests, backend-conformance tests,
  integration tests, capabilities/audits, or another module's source/tests
- creating or promoting Compiler 0005C, 0005D, 0005E, or 0006 task specifications
- unrelated documentation cleanup

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially model-owned operation
  semantics, compiler-owned gradient rules, and backend-owned execution
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)
- [Example format](../../../../developer-guide/documentation/example-format.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capabilities](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Model capability and contract closure audit](../model-capability-contract-closure-audit.md)
- [Adjoint expressibility audit](../adjoint-expressibility-audit.md), especially I5-I9 and O1-O4
- [Task 0018G](0018g-axis-scatter-semantics.md)
- [Task 0018H](0018h-axis-scatter-tensor-expressions.md)
- [Task 0018I](0018i-scatter-nd-semantics.md)
- [Task 0018J](0018j-scatter-nd-tensor-expression.md)
- [Task 0019C](0019c-sort-and-argsort.md)
- [Task 0025A](0025a-portable-floating-comparison-extrema-and-clamp-semantics.md)
- [Compiler master plan](../../compiler/master-plan.md)
- [Completed Compiler task 0005A](../../compiler/tasks/0005a-derivative-policy-and-elementwise-activation-gradient-completion.md)
- [Completed Compiler task 0005B](../../compiler/tasks/0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/model` owns backend-independent forward operation semantics. Recording the fixed
  represented-value meaning in existing scatter semantics and public API documentation is inside
  the current architecture.
- The model layer does not evaluate Tensor values. No executable helper may be added merely to
  make these rules unit-testable.
- Compiler owns derivative policy and gradient-expression construction. This task must not select
  a cotangent formula or imply a policy for zeros, infinities, NaNs, extrema ties, duplicate
  targets, or discontinuities.
- Forward and generated gradient expressions later share this same model numerical contract.
  Compiler 0005C may choose local derivative conventions but may not create a separate forward
  scatter algebra.
- Concrete backends later own evaluation. Backend-conformance tests must check the selected
  grouping and represented-value behavior when execution exists; this task implements no backend.
- The order-independent logical group is necessary portability, not a prescribed implementation
  traversal. No storage layout, atomic ordering, sequential fold, tree shape, or kernel route is
  part of model semantics.
- Existing `SCATTER_ELEMENTS` and `SCATTER_ND` kinds, attributes, Tensor methods, validation,
  descriptors, provenance, and data-type eligibility remain exact.
- The downstream stable-ARGSORT decision is compiler planning only. It does not change Model
  0019C's separate one-output `SORT` and `ARGSORT` occurrences or require a hidden forward output.
- If accurate implementation requires executable semantics, another type/member, changed
  validation, a derivative decision, or an architecture change, stop and report the exact
  conflict.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.operation.index` — owns shared scatter reduction vocabulary,
  axis-scatter semantics, Scatter-ND semantics, and their immutable attributes.
- `io.github.pho001.synaptik.model.tensor` — owns public Tensor method contracts and metadata-only
  expression construction.

Packages added, moved, or removed:

- None.

Type placement:

- No type is created or moved.
- Numerical meaning remains in `ScatterReduction` and the existing operation-family/public Tensor
  Javadocs. A new numerics package or policy object would create an unneeded abstraction and is
  forbidden.

## Affected files

Expected production Java, Javadoc-only:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/ScatterReduction.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/AxisScatterKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/ScatterElementsAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/ScatterNdKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/ScatterNdAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorAxisScatterExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorScatterNdExpressions.java`

Expected documentation and planning:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/modules/compiler/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification:

- `DataType`, `BFloat16Bits`, `TensorDescriptor`, `TensorFactory`, `TensorProducer`,
  `TensorProvenance`, `Operation`, and `OperationSignature` — their representations,
  construction, identity, and occurrence contracts remain exact.
- Axis-scatter and Scatter-ND semantic/expression tests — they lock declarations, reduction
  eligibility, validation, descriptor, provenance, and non-execution behavior. They do not
  evaluate Tensor values and must not gain a fake evaluator.
- `AggregateReductionKind`, `CumulativeScanKind`, binary/scalar extrema, and their documentation —
  they provide the existing product/extrema meaning reused here and do not change.
- `OrderingKind`, `SortAttrs`, `TensorSortExpressions`, and sort tests — the downstream compiler
  planning decision consumes their existing stable one-output contracts without modifying them.
- Model capabilities and both completed audits — historical selection and audit results remain
  unchanged; this focused prerequisite finalizes directly consumed Javadocs and APIs.
- Training and Runtime APIs — no training request, optimizer, prepared state, runtime, or
  execution contract changes.
- `ARCHITECTURE.md`, focused architecture documents, ADRs, and architecture tests — ownership and
  dependency rules do not change.
- Gradle files — Java 26 and current module dependencies remain exact.
- Backend-conformance and integration tests — no backend or end-to-end execution exists in this
  task; future executable implementations must receive conformance coverage.

## Maximum scope

At most the exact fifteen paths listed under expected production Java and expected documentation
and planning may change: eight Javadoc-only Java paths and seven documentation/planning paths.

No test, Gradle, architecture, capability/audit, sort Java, other Java, other module,
backend-conformance, or integration path may change. If another path is required for accuracy,
stop and request a replanning decision rather than expanding scope.

## Acceptance criteria

- `MUL`, `MIN`, and `MAX` define one candidate group containing the base exactly once and every
  addressed logical update exactly once.
- Duplicate target entries remain distinct contributions, including equal represented values.
- Scatter Elements and Scatter-ND suffix-slice grouping are both explicit.
- Empty update groups preserve the exact base representation without applying an identity,
  rounding, or NaN canonicalization.
- Group meaning is independent of logical encounter order, physical layout, backend traversal,
  parallel grouping, or atomic schedule.
- FLOAT64, FLOAT32, and BFLOAT16 `MUL` state exact product-format, rounding, overflow, underflow,
  zero, signed-zero, infinity, and NaN rules without selecting an executable algorithm or NaN
  payload.
- INT32/INT64 `MUL` uses unchanged-width two's-complement modular multiplication; integral
  `MIN`/`MAX` use signed order.
- Floating `MIN`/`MAX` reuse Model 0025A's NaN propagation, signed-zero selection, infinity order,
  and equal-candidate boundary.
- `NONE`, `ADD`, `SCATTER_ADD`, index bounds, negative indices, and duplicate detection retain
  their completed boundaries without a new order or behavior.
- No derivative formula, tie/subgradient sharing, zero-product case split, infinity/NaN
  convention, or generated compiler expression is selected.
- Every Java change is Javadoc-only. Declarations, imports, annotations, enum order, signatures,
  record components, executable statements, validation, messages, and bytecode-visible API remain
  unchanged.
- No evaluator, numerical-policy object, operation, attribute, Tensor method, helper behavior,
  data type, test change, backend behavior, or execution is added.
- Tensor API, Compile API, glossary, task, model master plan, compiler master plan, and roadmap use
  consistent current/planned wording.
- Compiler planning explicitly permits one separately constructed stable `ARGSORT` occurrence for
  one-output `SORT` backward only when kind/signature/axis/direction match exactly. It does not add
  a hidden sort output or public API. A missing or mismatched signature/attribute path must fail
  closed before partial backward construction.
- After implementation, Model 0025C is `Complete`; Compiler 0005C remains `Draft` without a task
  file and depends on Model 0025C plus Compiler 0005B; Compiler 0005D, 0005E, and 0006 remain
  `Draft` without task files.
- Exactly the fifteen authorized paths change. Markdown links/anchors, fences, final newlines,
  trailing whitespace, status/dependency consistency, and `git diff --check` pass.

## Tests / validation

This task changes semantic contracts in Javadocs and explanatory documentation but no executable
Java behavior. One clean documentation-focused implementation context is the required
documentation pass. It must read the final source and tests directly and finalize the Javadocs
and documentation itself; do not create a redundant second documentation context merely to repeat
that work.

Run one final model compile/test/Javadoc command after all Javadocs are stable:

```bash
./gradlew :modules:model:compileJava :modules:model:test :modules:model:javadoc
```

The model tests are regression evidence for unchanged declarations, attributes, validation,
public surfaces, descriptors, provenance, and expression construction. They are not evidence that
Tensor values were evaluated. No focused numerical execution test is appropriate because Model
has no evaluator and immutable declaration members do not change.

Then run:

```bash
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

If the temporary Markdown validator is absent, create an equivalent validator outside the
repository. It must check repository-local targets and heading anchors, balanced fences, final
newlines, and trailing whitespace.

Required manual/source checks:

- inspect every Java diff hunk and confirm that only Javadoc comments changed;
- mechanically compare the eight Java files with Javadoc removed and confirm all remaining bytes
  are unchanged;
- inspect generated Javadoc for all affected public index-operation types and Tensor methods;
- inspect the two package-private helper Javadocs in source;
- verify the exact fifteen-path ceiling and absence of test/Gradle/architecture/capability changes;
- verify Model 0025C status and Compiler 0005C dependency/status in all three planning indexes;
- verify no Compiler 0005C–0005E or 0006 task file exists and that those rows remain Draft;
- verify the compiler master plan and roadmap contain the stable-ARGSORT planning decision, exact
  match guard, and fail-closed boundary without changing Model sort semantics;
- verify no evaluator, numerical-policy type, registry, selectable mode, operation, attribute,
  Tensor method, data type, test, derivative convention, compiler implementation, backend
  behavior, dependency, or build change was added; and
- verify terminology, tables, links, anchors, fences, final newlines, trailing whitespace, and
  `git diff --check`.

Repository-wide and architecture validation are deferred to CI because this is one module's
Javadoc plus explanatory/planning documentation and changes no executable behavior, dependency,
build configuration, or architecture boundary.

## Dependencies

- Model 0018G–0018J functional scatter semantics and public expression construction — Complete.
- Model 0018U–0018U1 integral modular arithmetic and signed extrema — Complete.
- Model 0019C stable separate one-output sort/argsort — Complete.
- Model 0025A portable floating comparison/extrema contract — Complete.
- Compiler 0005B reduction/scan/normalization gradient completion — Complete; Compiler 0005C is
  the downstream consumer whose planning remains blocked until this task is implemented.

## Follow-up tasks

- Compiler 0005C — after Model 0025C is Complete, separately plan and implement compiler-owned
  layout/window/indexing/scatter/ordering/stochastic gradient completion. It must select every
  scatter derivative boundary and duplicate-sharing convention. For one-output `SORT`, it may
  construct one separate stable `ARGSORT` Tensor occurrence with the exact same input, normalized
  axis, and direction, then route the cotangent through unique-target `SCATTER_ELEMENTS/NONE`.
  Preflight must fail closed if the exact SORT or ARGSORT signature/attributes cannot be matched.
- Future concrete backends — when configurable scatter execution is implemented, add or update
  backend-conformance tests for duplicate groups, exact base preservation, zero-size update
  domains, floating zero signs/infinities/NaNs/overflow/underflow, modular integral overflow, and
  extrema signed-zero/NaN cases.

Do not create either follow-up specification during this task.

## Architecture impact

Expected impact: None.

The architecture already assigns forward operation semantics to Model, gradient rules to
Compiler, and concrete evaluation to backends. This task completes existing documentation without
changing ownership or dependencies.

If implementation requires an architecture change, executable model evaluation, another module,
or a derivative decision, stop and report the exact conflict.

## Implementation prompt

Use this prompt in one separate clean documentation-focused task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, the focused model/compiler/module/dependency/lifecycle/backend
documents, documentation rules plus General, API/Javadoc, Planning, and Example profiles,
planning guide and roadmap, model capabilities/master plan and both completed audits, completed
Model 0018G-0018J/0018U-0018U1/0019C/0025A, completed Compiler 0005A/0005B, compiler master plan,
this task, all eight affected production files, the review-only helpers/tests, Tensor/Compile/
Training/Runtime APIs, glossary, and Java 26/module Gradle boundaries.

Implement Model 0025C exactly within its fifteen authorized paths. Change Java only in Javadoc.
Finalize the portable SCATTER_ELEMENTS and SCATTER_ND MUL/MIN/MAX represented-value semantics
without adding an evaluator, policy object, operation, attribute, Tensor method, test change,
derivative convention, compiler implementation, backend behavior, or execution. Preserve NONE,
ADD, SCATTER_ADD, value-aware index validation, and every declaration/behavior. Stop on any
scope, behavior, completed-contract, or architecture conflict.

This clean documentation-focused context itself is the required documentation pass; do not create
a redundant second pass. Run the specified model compile/test/Javadoc and documentation/scope/
status checks, inspect generated Javadoc, record exact evidence and no-change conclusions, then
mark Model 0025C Complete only if every acceptance criterion passes. Keep Compiler 0005C-0005E
and 0006 Draft without task files. Do not commit or push.
```

## Local decisions

- A target group is a logical multiset, not a traversal sequence. Base participates once and each
  addressed update participates once; duplicate targets remain distinct contributions.
- `MUL`, `MIN`, and `MAX` are symmetric abstract reductions, so storage layout, encounter order,
  atomic order, and tree grouping are not model-visible.
- Floating `MUL` reuses the existing product target rather than inventing sequential intermediate
  rounding. Integral `MUL` reuses fixed-width modular meaning.
- Floating extrema reuse Model 0025A across the complete candidate group. No identity is needed
  because every group contains its base.
- An empty update group bypasses arithmetic and preserves the exact base representation.
- `NONE`, `ADD`, `SCATTER_ADD`, bounds, negative-index policy, and duplicate detection do not
  change.
- Stable `ARGSORT` recomputation is a future compiler formula choice, not a hidden Model output or
  public sort API change.

## Known limitations

- Index bounds, negative-index policy, and value-aware duplicate validation remain separately
  owned and unchanged.
- The completed contract will specify a mathematical floating target and exact special-value
  classes, not a sequential algorithm, NaN payload, instruction sequence, or universal bitwise
  backend implementation.
- Compiler 0005C remains Draft and must still choose derivative routing at zeros, infinities,
  NaNs, extrema ties, duplicate targets, and ordering discontinuities.
- Numerical execution cannot be verified until a concrete backend exists. Future implementations
  require backend-conformance coverage.

## Validation evidence

Planning context `/root/plan_model_scatter_semantics` read the required architecture, focused
architecture, documentation profiles, planning guide/roadmap, Model and Compiler plans, completed
scatter/gather/indexing/extrema/sort tasks, both Model audits, current APIs/glossary, production
contracts, helpers, and tests. It found no architecture or completed-contract conflict.

Planning-stage validation:

- A targeted Markdown validator passed for this task, both master plans, and the roadmap: four
  files and 550 repository-local links, including heading anchors, with balanced fences, final
  newlines, and no trailing whitespace.
- `git diff --check` passed.
- The union of tracked and untracked paths contains exactly this task, the Model master plan, the
  Compiler master plan, and the roadmap. No Java, test, Gradle, architecture, API, glossary,
  capability/audit, or other-module path changes in this planning-only diff.
- The future implementation scope contains exactly fifteen paths: eight existing Java files with
  Javadoc-only permission and seven API/glossary/planning files. It contains no test, Gradle,
  architecture, capability/audit, sort Java, or other-module path.
- Exact-status checks found this task as the only detailed task whose status line is `Ready`.
  Model/roadmap indexes agree; Compiler 0005C remains a Draft row dependent on Model 0025C and
  Compiler 0005B. Compiler 0005D, 0005E, and 0006 remain Draft.
- The Compiler task directory contains specifications only through completed 0005B. No 0005C,
  0005D, 0005E, or 0006 task file exists.
- The compiler master plan and roadmap both record the matching stable-ARGSORT allowance, reject
  hidden SORT outputs and public API changes, and require fail-closed exact kind/signature/axis/
  direction matching before partial backward construction.

No Java test or Javadoc-generation command ran because this change creates planning only and does
not implement task 0025C.

Implementation context `/root/implement_model_0025c_scatter` independently read the governing
architecture, documentation profiles, planning contracts, completed related tasks and audits,
current APIs, exact affected source, helpers, tests, generated Javadoc, and Java 26 build boundary.
It found no architecture, behavior, or scope conflict and completed the required documentation
pass without a second implementation context.

Implementation-stage validation:

- `./gradlew :modules:model:compileJava :modules:model:test :modules:model:javadoc` passed under
  OpenJDK 26.0.1. The Model suite reported 127 suites and 1,019 tests with no skips, failures, or
  errors.
- Generated Javadoc was inspected for all five public index-operation types and the affected
  public `Tensor.scatterElements`/`scatterNd` methods. Both package-private construction-helper
  Javadocs were inspected directly in source.
- Removing every Javadoc block mechanically from each of the eight before/after Java files
  produced identical remaining bytes. Independent `javac --release 26 -g:none` compilation of
  those before/after sources produced byte-for-byte identical class trees; declarations, imports,
  annotations, enum order, signatures, record components, validation, messages, and executable
  behavior are unchanged.
- The Markdown validator passed all 12 affected-or-linked Markdown files and 714 repository-local
  links, including heading anchors, balanced fences, final newlines, and trailing whitespace.
- The tracked/untracked union contains exactly the fifteen authorized paths: eight Javadoc-only
  Java files and seven API/glossary/planning files. No test, Gradle, architecture, capability/audit,
  sort Java, conformance/integration, other-module, or other path changed.
- Status/dependency checks agree that Model 0025C is Complete and Compiler 0005C remains Draft
  dependent on Model 0025C and Compiler 0005B. Compiler 0005C–0005E and 0006 have no task files and
  remain Draft. Compile API, compiler master plan, and roadmap retain the exact-match stable-
  ARGSORT allowance and fail-closed preflight boundary without a hidden SORT output or API change.
- `git diff --check` passed. No evaluator, numerical-policy type, registry, selectable mode,
  operation, attribute, Tensor method, data type, derivative convention, compiler implementation,
  backend behavior, dependency, or build change was added.

## Implementation notes

The shared `ScatterReduction` Javadoc now owns the complete target-group and represented-value
contract. Axis-scatter, Scatter-ND, public Tensor, and package-private helper Javadocs specialize
that contract only for their coordinate and suffix-slice mappings. Tensor API provides the full
reader-facing floating/integral matrix; glossary names the functional-scatter target group;
Compile API records preservation obligations and the future stable-ARGSORT allowance. No in-scope
decision refinement changed the fixed task contract.

## Completion summary

Completed the portable functional-scatter represented-value contract for configurable
`SCATTER_ELEMENTS` and `SCATTER_ND` `MUL`, `MIN`, and `MAX`. Base participation, distinct
duplicate contributions, suffix-slice scalar grouping, exact empty-group preservation, floating
product/extrema edge cases, modular integral multiplication, signed extrema, and order
independence are now explicit without executable Model evaluation or derivative policy.

Files changed: the exact eight Javadoc-only Java files and seven documentation/planning files
listed under [Affected files](#affected-files).

Validation: the final Model compile/test/Javadoc command passed with 127 suites and 1,019 tests;
generated and source Javadocs were inspected; mechanically Javadoc-stripped source and no-debug
bytecode were identical before/after; Markdown with 714 local links, exact fifteen-path scope,
status/dependency/spec-absence checks, Java 26/build boundaries, and `git diff --check` passed.

Documentation review: all affected Javadocs, Tensor API, Compile API, glossary, this task, both
master plans, and roadmap required updates. Model capabilities/audits, Training/Runtime APIs,
related aggregate/scan/extrema/ordering contracts, architecture/ADRs/tests, Gradle, dependencies,
other modules, backend-conformance, and integration required no change because this task changes
only the existing Model forward documentation contract.

Unresolved issues: None within Model task 0025C.

Required follow-up: Draft Compiler 0005C must separately choose scatter, ordering, layout/window,
and stochastic derivative policies while preserving this forward contract and the documented
stable-ARGSORT fail-closed planning boundary. Future executable backends require conformance
coverage.

Status: Complete

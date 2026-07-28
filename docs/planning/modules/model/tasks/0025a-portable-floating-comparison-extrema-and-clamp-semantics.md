# Task 0025A: Portable Floating Comparison, Extrema, and Clamp Semantics

## Status

Complete

## Goal

Close the smallest model-owned forward-semantics prerequisite needed before Compiler task 0005A
can plan floating-comparison-dependent gradient formulas.

Make the current comparison, pairwise minimum/maximum, scalar minimum/maximum, and first-class
range-clamp contracts state one fixed portable meaning for NaN, signed zero, equality, and exact
typed scalar values. The task changes Javadoc and explanatory documentation only. It adds no
evaluator, operation, attribute, Tensor method, test-only numerical implementation, compiler
formula, derivative convention, backend behavior, or execution.

The semantic relationship is:

```text
represented floating values
  -> fixed comparison and extrema meaning in modules/model
  -> later backend implementations reproduce that meaning

fixed forward meaning
  -> later Compiler 0005A selects derivative behavior separately
```

## Scope

- Finalize `BinaryComparisonKind` Javadoc so all six floating relations have explicit NaN and
  signed-zero behavior.
- Finalize binary and scalar `MIN`/`MAX` Javadoc so both families state the same complete
  NaN-propagating, signed-zero-selecting portable contract.
- Finalize `CLAMP` and `ClampRangeAttrs` Javadoc so first-class range clamp is exactly the ordered
  composition `MIN(MAX(input, minValue), maxValue)` under those extrema rules.
- Preserve the current `ClampRangeAttrs` validation that accepts equal bounds, both opposite-
  signed-zero bound orders, and one or two NaN endpoints while rejecting only a strictly inverted
  represented numeric range after its existing type/BOOL checks.
- Finalize the corresponding public `Tensor` method Javadocs for binary comparisons, pairwise
  binary/scalar extrema, range clamp, and one-bound clamp conveniences.
- Clarify the current Tensor API, the Compile API's model-semantics prerequisite status, and the
  glossary atomically.
- Record this task as the sole detailed `Ready` task in the model master plan and roadmap.
- Keep Compiler 0005A `Draft` without a task file, make it depend on completed Model 0025A before
  promotion, and preserve Compiler 0005B–0005E and 0006 as `Draft` without task files.
- During implementation, synchronize this task, the model master plan, compiler master plan, and
  roadmap only after all documentation and validation acceptance criteria pass.

## Fixed forward semantic contract

### Floating comparisons

For FLOAT64, FLOAT32, and BFLOAT16, comparisons use the ordinary numeric value represented by
each operand under the existing same-category promotion contract:

- `GREATER_THAN`, `GREATER_OR_EQUAL`, `LESS_THAN`, and `LESS_OR_EQUAL` are ordered numeric
  comparisons. If either operand is NaN, the result is false.
- Negative zero and positive zero compare numerically equal. Neither is strictly before or after
  the other; both inclusive relations are true in either operand order.
- `EQUAL` is exact numeric equality, not bit equality or approximate/tolerance equality. NaN is
  unequal to every value, including itself, and negative zero equals positive zero.
- `NOT_EQUAL` is the exact logical complement of `EQUAL`. It is therefore true when either operand
  is NaN and false for either ordering of opposite-signed zero.
- Finite values and infinities otherwise use ordinary numeric order.

The existing integral contract is unchanged: INT32/INT64 operands use current signed exact
comparison after current promotion, with exact equality and no floating special-value policy.

### Floating pairwise minimum and maximum

Binary and scalar floating `MIN`/`MAX` use one typed rule:

- if either candidate is NaN, the result is NaN, without a payload, sign, source-operand, or
  bitwise-result promise;
- for opposite-signed zeros, `MIN` produces negative zero and `MAX` produces positive zero,
  independent of operand order;
- infinities and unequal non-NaN values use ordinary numeric order; and
- equal nonzero candidates produce that numeric value without promising which operand's
  representation or identity is selected.

Scalar operations compare the input's represented value with the exact represented value in the
same-typed `ScalarValue`. `ScalarValue` continues to preserve exact raw floating bits for metadata
equality and hashing. That storage/equality contract does not turn elementwise numeric equality
into bit equality.

The existing integral `MIN`/`MAX` contract remains ordinary signed order after current promotion.

### First-class range clamp

For each logical input value:

```text
CLAMP(input, minValue, maxValue)
  = MIN(MAX(input, minValue), maxValue)
```

The order is semantic: apply `MAX` with `minValue` first, then `MIN` with `maxValue`. The operation
remains one first-class `CLAMP` occurrence with no stored intermediate producers.

Consequences that documentation must state explicitly:

| Input or bounds | Result |
|---|---|
| input, `minValue`, or `maxValue` is NaN | NaN, with no payload promise |
| equal non-NaN, same-representation bounds | that bound for every non-NaN input |
| bounds `[-0, +0]` | negative inputs and negative zero produce `-0`; positive inputs and positive zero produce `+0` |
| bounds `[+0, -0]` | every non-NaN input produces `-0` |

These outcomes follow from the ordered extrema definition; they do not add validation. Existing
construction continues to accept either signed-zero bound order and NaN bounds because primitive
`>` is false for those cases. `clampMin` remains exactly scalar `MAX`, and `clampMax` remains
exactly scalar `MIN`.

### Data-type and representation boundary

- FLOAT64 and FLOAT32 use their represented IEEE-754 binary64 and binary32 values.
- BFLOAT16 uses the value represented by its current exact 16-bit storage; widening for the
  existing comparison domain does not introduce a new scalar conversion or rounding contract.
- Exact `ScalarValue` raw bits remain immutable operation metadata, including signed-zero and NaN
  payload bits.
- This task adds no FLOAT16 data type and promises no NaN payload preservation, operand-source
  selection, instruction identity, intermediate precision, backend algorithm, or bitwise result
  beyond existing contracts.

## Out of scope

- any Java declaration, executable statement, validation branch, failure message, operation kind,
  operation signature, attribute component, Tensor method, data type, promotion rule, helper
  method, or test behavior
- a `model.numerics` package, numerical-policy object, registry, service, selectable mode,
  generic evaluator, backend utility, runtime utility, or test-only fake evaluator
- FLOAT16, implicit scalar conversion, a new scalar carrier, payload preservation, exact
  instruction selection, bitwise reproducibility, or backend tolerance
- changes to reduction `MIN`/`MAX`, arg extrema, sorting/top-K, pooling, scatter reduction, or
  other operation families
- constant folding, algebraic rewriting, canonicalization, graph capture, inference, validation,
  planning, lowering, execution, or backend conformance implementation
- gradient formulas or any derivative choice, including MIN/MAX ties, clamp endpoints, ABS/RELU
  zero, discontinuities, infinities, NaNs, singularities, or invalid-domain behavior
- changing `ARCHITECTURE.md`, an ADR, architecture documentation, module boundaries, dependencies,
  Gradle, Java 26 configuration, architecture tests, backend-conformance tests, integration tests,
  or another module's source/tests
- creating a Compiler 0005A task file or promoting Compiler 0005A–0005E or 0006
- unrelated documentation cleanup

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially model-owned operation
  semantics, compiler-owned gradient rules, and backend-owned execution
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
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
- [Adjoint expressibility audit](../adjoint-expressibility-audit.md), especially E8–E17
- [Task 0018N](0018n-typed-scalar-value-contract.md)
- [Task 0018T](0018t-scalar-arithmetic-family-normalization.md)
- [Task 0018U](0018u-integral-elementwise-arithmetic-and-comparisons.md)
- [Task 0025](0025-canonical-tensor-producer-outputs.md)
- [Compiler master plan](../../compiler/master-plan.md)
- [Completed Compiler task 0005](../../compiler/tasks/0005-publication-planning-orchestration-and-compile-artifacts.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/model` owns backend-independent forward operation semantics. Recording the fixed
  comparison/extrema/clamp meaning in its existing semantic kinds and public API documentation is
  inside the current architecture.
- The model layer does not evaluate Tensor values. No executable helper may be added merely to
  make these rules unit-testable.
- Compiler owns derivative policy and gradient-expression construction. This task must not select
  or imply any derivative at a tie, endpoint, discontinuity, singularity, infinity, or NaN.
- Forward and generated gradient expressions will later share this same fixed forward numerical
  contract. Compiler 0005A may choose local derivative conventions but may not add a separate
  forward comparison/extrema algebra.
- Concrete backends later own evaluation and must be checked through backend-conformance tests
  when execution exists. This task does not implement or test a backend.
- `ScalarValue` remains the exact typed-bit metadata carrier. Its Java equality is intentionally
  distinct from numeric elementwise equality.
- `ClampRangeAttrs` retains its current primitive represented-value validation and exact bound
  references. The task clarifies consequences; it does not change accepted or rejected ranges.
- No module dependency, package, architecture rule, operation shape, or public surface changes.
- If accurate implementation requires executable semantics, a new type or member, a changed
  validation rule, or a derivative decision, stop and report the exact conflict.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.operation.elementwise.comparison` — owns the six comparison
  meanings whose floating edge behavior becomes explicit.
- `io.github.pho001.synaptik.model.operation.elementwise.binary` — owns pairwise Tensor/Tensor
  `MIN` and `MAX`.
- `io.github.pho001.synaptik.model.operation.elementwise.scalar` — owns scalar `MIN`, scalar
  `MAX`, first-class `CLAMP`, and its exact typed attributes.
- `io.github.pho001.synaptik.model.tensor` — owns public Tensor method contracts and metadata-only
  expression construction.
- `io.github.pho001.synaptik.model.datatype` — owns unchanged `ScalarValue` represented bits and
  current promotion contracts; review only.

Packages added, moved, or removed:

- None.

Type placement:

- No type is created or moved.
- Numerical meaning remains in the existing operation-family Javadocs and public Tensor contract.
  A general numerics package or policy object would create an unneeded abstraction and is
  forbidden by this task.

## Affected files

Expected production Java, Javadoc-only:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/comparison/BinaryComparisonKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/binary/BinaryArithmeticKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/scalar/ScalarElementwiseKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/scalar/ClampRangeAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Expected documentation and planning:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/modules/compiler/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification:

- `ScalarValue`, `DataType`, `DataTypePromotion`, `ScalarValueAttrs`,
  `TensorBinaryExpressions`, `TensorScalarExpressions`, and `TensorComparisonExpressions` — their
  declarations, storage, promotion, validation, and metadata-construction behavior stay exact.
- `BinaryComparisonKindTest`, `BinaryArithmeticKindTest`, `ScalarElementwiseSemanticsTest`,
  `ScalarValueTest`, `TensorBinaryComparisonTest`, `TensorBinaryArithmeticTest`, and
  `TensorScalarElementwiseTest` — existing tests lock the immutable declarations, attributes,
  type domains, construction, provenance, and accepted clamp bounds. They do not evaluate Tensor
  values and must not be expanded with a fake evaluator.
- model capabilities and both completed audits — they already record that forward numerical
  policy must be fixed while derivative policy remains compiler-owned; this task makes the
  directly consumed Javadocs and APIs agree with that closure-level statement.
- Training and Runtime APIs — no training request, optimizer, prepared state, runtime, or
  execution contract changes.
- `ARCHITECTURE.md`, focused architecture documents, ADRs, and architecture tests — ownership and
  dependency rules do not change.
- Gradle files — the repository already uses Java 26 and current module dependencies remain exact.
- backend-conformance and integration tests — no backend or end-to-end execution exists in this
  task; future executable implementations must receive conformance coverage.

## Maximum scope

At most the exact twelve paths listed under expected production Java and expected documentation
and planning may change: five Javadoc-only Java paths and seven documentation/planning paths.

No test, Gradle, architecture, capability/audit, other Java, other module, backend-conformance, or
integration path may change. If another path is required for accuracy, stop and request a
replanning decision rather than expanding scope.

## Acceptance criteria

- Every floating comparison relation states the exact NaN and signed-zero result defined in this
  task; no tolerance or bit-equality ambiguity remains.
- `NOT_EQUAL` is documented as the exact logical complement of `EQUAL`.
- Binary and scalar floating `MIN`/`MAX` use one complete rule for NaN, infinities, opposite-
  signed zero, and otherwise equal or ordered values.
- Scalar Javadocs distinguish exact `ScalarValue` bit retention from represented-value numerical
  comparison.
- First-class CLAMP is exactly ordered `MIN(MAX(input, minValue), maxValue)` and documents NaN
  input/bounds, equal bounds, both opposite-signed-zero endpoint orders, and the absence of stored
  intermediate producers.
- `ClampRangeAttrs` retains and accurately documents its current acceptance of equal, signed-zero,
  infinity, and NaN bounds and its unchanged strict-inversion rejection.
- FLOAT64, FLOAT32, and BFLOAT16 represented-value behavior is explicit; FLOAT16 and NaN payload
  preservation are not promised.
- Integral comparison and extrema semantics remain unchanged and exact.
- All Java changes are Javadoc-only. Declarations, imports, annotations, enum order, signatures,
  record components, executable statements, validation, messages, and bytecode-visible API remain
  unchanged.
- No executable comparison/extrema/clamp helper or test evaluator is added. Existing focused tests
  remain unchanged and pass as regression evidence for the unmodified declarations and
  construction contracts.
- Tensor API, Compile API, glossary, task, model master plan, compiler master plan, and roadmap
  use consistent current/planned wording and do not imply execution or gradient completion.
- The clean documentation-focused implementation context records reasoned no-change conclusions
  for tests, capabilities/audits, Training/Runtime APIs, architecture/ADRs/tests, Gradle, other
  modules, backend conformance, and integration.
- After implementation, Model 0025A is `Complete`; Compiler 0005A remains `Draft` without a task
  file and depends on Model 0025A plus Compiler 0005; Compiler 0005B–0005E and 0006 remain `Draft`
  without task files.
- Exactly the twelve authorized paths change. Markdown links/anchors, fences, final newlines,
  trailing whitespace, status/dependency consistency, and `git diff --check` pass.

## Tests / validation

This task changes semantic contracts in Javadocs and explanatory documentation but no executable
Java behavior. A single clean documentation-focused implementation context is the required
documentation pass: it is separate from this planning context, reads the final source/tests
directly, and finalizes the Javadocs and documentation itself. Do not create a redundant second
documentation context solely to repeat that work.

Run one final model compile/test/Javadoc command after all Javadocs are stable:

```bash
./gradlew :modules:model:compileJava :modules:model:test :modules:model:javadoc
```

The module test is regression evidence for unchanged declaration, attribute, validation, public
surface, descriptor, and provenance behavior. It is not evidence that Tensor values were
evaluated. No focused semantic execution test is appropriate because the model has no evaluator
and no immutable declaration member changes; adding one would test a fake implementation rather
than the model contract.

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
- inspect generated Javadoc for all five affected production types and the relevant Tensor
  methods;
- verify the exact twelve-path ceiling and absence of test/Gradle/architecture changes;
- verify Model 0025A status and Compiler 0005A dependency/status in all three planning indexes;
- verify that no Compiler 0005A task file exists and that 0005B–0005E/0006 remain Draft without
  task files;
- verify no `model.numerics`, policy, registry, selectable mode, operation, attribute, Tensor
  method, data type, FLOAT16, evaluator, or derivative convention was added; and
- verify terminology, examples/tables, links, anchors, fences, final newlines, trailing
  whitespace, and `git diff --check`.

Repository-wide and architecture validation are deferred to CI because this is one module's
Javadoc plus explanatory/planning documentation and changes no executable behavior, dependency,
build configuration, or architecture boundary.

## Dependencies

- Model 0018N typed scalar value contract — Complete.
- Model 0018T scalar arithmetic family normalization and selected extrema/clamp meaning —
  Complete.
- Model 0018U integral arithmetic/comparison expansion — Complete and unchanged by this task.
- Model 0025 canonical producer outputs — Complete and the immediate model planning predecessor.
- Compiler 0005 publication/planning/artifacts — Complete; Compiler 0005A is the downstream
  consumer whose planning remains blocked until this task is implemented.

## Follow-up tasks

- Compiler 0005A — after Model 0025A is Complete, separately plan and implement compiler-owned
  elementwise/activation gradient completion. It must choose MIN/MAX tie routing or sharing,
  clamp endpoint behavior, ABS/RELU zero conventions, and singular/exceptional formula behavior.
  This task chooses none of those derivative policies.
- Future concrete backends — when comparison/extrema/clamp execution is implemented, add or update
  backend-conformance tests for NaN, infinities, both zero signs, equal candidates, scalar exact
  values, NaN bounds, equal bounds, and both signed-zero clamp endpoint orders.

Do not create either follow-up specification during this task.

## Architecture impact

Expected impact: None.

The architecture already assigns forward operation semantics to model, compiler-owned gradient
rules to compiler, and concrete evaluation to backends. This task makes existing documentation
complete without changing ownership or dependencies.

If implementation requires an architecture change, executable model evaluation, another module,
or a derivative decision, stop and report the exact conflict.

## Implementation prompt

Use this prompt in one separate clean documentation-focused task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, the focused model/compiler/backend-conformance boundary docs,
documentation rules plus the General, API/Javadoc, Planning, and Example profiles, planning guide
and roadmap, model capabilities/master plan, closure and adjoint audits E8-E17, completed Model
0018N/0018T/0018U/0025, compiler master plan and completed Compiler 0005, this task, all five
affected production files, the review-only helpers/tests, Tensor/Compile/Training/Runtime APIs,
glossary, and Java 26/module Gradle boundaries.

Implement Model 0025A exactly within its twelve authorized paths. Change Java only in Javadoc.
Finalize the fixed portable floating comparison, pairwise/scalar MIN/MAX, and ordered first-class
CLAMP semantics without adding an evaluator, policy object, operation, attribute, Tensor method,
data type, test change, derivative convention, compiler behavior, backend behavior, or execution.
Preserve current ClampRangeAttrs validation and exact ScalarValue bit storage. Stop on any scope,
behavior, or architecture conflict.

This clean documentation-focused context itself is the required documentation pass; do not create
a redundant second pass. Run the specified model compile/test/Javadoc and documentation/scope/
status checks, inspect generated Javadoc, record exact evidence and no-change conclusions, then
mark Model 0025A Complete only if every acceptance criterion passes. Keep Compiler 0005A Draft
without a task file, dependent on completed Model 0025A and Compiler 0005; keep 0005B-0005E and
0006 Draft without task files.
```

## Local decisions

- Ordinary represented numeric comparison is the sole portable floating relation. No total-order,
  raw-bit, approximate, tolerance, or selectable-policy alternative was added.
- `NOT_EQUAL` is the exact logical complement of numeric `EQUAL`. This supplies the NaN and
  opposite-signed-zero outcomes without a second independent policy.
- Binary and scalar MIN/MAX now use the same complete NaN, infinity, signed-zero, unequal-value,
  and equal-nonzero contract. No payload or source-operand identity becomes observable.
- Scalar numerical comparison uses the exact same-typed `ScalarValue` represented value.
  `ScalarValue` raw-bit equality and hashing remain distinct immutable metadata behavior.
- First-class CLAMP remains one operation with ordered
  `MIN(MAX(input, minValue), maxValue)` value meaning and no stored intermediate producers.
  Existing primitive bound validation remains unchanged.
- The model intentionally has no evaluator. The existing API/metadata tests, source-only checks,
  model compilation, generated Javadoc, and documentation validation are the correct evidence
  boundary.
- This clean documentation-focused implementation context performed the required final Javadoc,
  API, glossary, and planning pass. No second documentation context or duplicate test run was
  needed.

## Known limitations

- The completed contract specifies forward values, not an algorithm, tolerance, NaN payload,
  operand identity, instruction sequence, bitwise cross-backend result, or backend availability.
- Reduction extrema, arg extrema, sort/top-K, pooling, scatter extrema, and other comparison-using
  families remain outside this focused task.
- Compiler 0005A remains Draft without a task specification and must still choose derivative tie,
  endpoint, discontinuity, singularity, and exceptional-value policies.
- Numerical execution cannot be verified until a concrete backend exists. That future
  implementation must receive backend-conformance coverage.

## Validation evidence

- `./gradlew :modules:model:compileJava :modules:model:test :modules:model:javadoc` passed with
  `BUILD SUCCESSFUL`. The model report contains 127 suites and 1,018 tests with zero failures,
  errors, or skipped tests.
- `python3 /tmp/validate_synaptik_markdown.py` passed after the semantic edits with 12 Markdown
  files and 681 repository-local links checked. The final planning/status edit was validated again
  before completion.
- Generated Javadoc exists for all five affected types. Manual inspection found the four ordered
  NaN/signed-zero outcomes, exact equality and complement wording, both extrema zero selections,
  exact ordered CLAMP composition, and the signed-zero clamp endpoint consequences in the rendered
  HTML.
- A mechanical before/after comparison removed Javadoc blocks from each affected Java source and
  found every remaining byte identical. Imports, annotations, declarations, enum order,
  signatures, record components, executable statements, messages, and bytecode-visible API
  therefore remain unchanged.
- `git diff --check` passed. The union of tracked and untracked changed paths is exactly the five
  authorized Javadoc-only Java files and seven authorized documentation/planning files.
- Manual semantic review confirmed all six floating comparison truth rules, ordinary infinity
  order, NaN propagation, MIN negative-zero selection, MAX positive-zero selection, equal-nonzero
  wording, same-typed scalar represented-value comparison, all four required CLAMP cases, and the
  unchanged signed-integral contracts.
- Planning review confirmed Model 0025A Complete in its task, model master plan, and roadmap.
  Compiler 0005A still depends on completed Model 0025A plus Compiler 0005 and remains Draft;
  Compiler 0005B–0005E and 0006 also remain Draft. No task specification exists for any of those
  compiler rows.

Reasoned no-change conclusions:

- The seven named model tests remain unchanged. They already lock declarations, exact scalar
  metadata, range validation, domains, descriptors, provenance, and the public surface; adding a
  fake value evaluator would test behavior the model does not implement.
- Model capabilities and both completed audits remain unchanged. They already assign fixed forward
  semantics to model and derivative choices to compiler.
- Training and Runtime APIs remain unchanged because no training request, optimizer, prepared
  state, runtime, or execution contract changed. Compile API alone needed its prerequisite-status
  correction.
- `ARCHITECTURE.md`, focused architecture documentation, ADRs, architecture tests, module
  boundaries, dependencies, Gradle, and Java 26 configuration remain unchanged because ownership,
  structure, and executable behavior did not change.
- No other module source or tests changed. Backend-conformance and integration tests remain
  unchanged because this task adds no backend or end-to-end execution.

## Implementation notes

- Updated only Javadoc in `BinaryComparisonKind`, `BinaryArithmeticKind`,
  `ScalarElementwiseKind`, `ClampRangeAttrs`, and `Tensor`.
- Finalized current represented-value behavior in the Tensor API, updated the Compile API from an
  incomplete-model prerequisite to deferred compiler derivative policy, and synchronized glossary
  terminology.
- Synchronized this task, the model and compiler master plans, and the roadmap. Compiler
  0005A–0005E and 0006 were not promoted and no later task specification was created.
- Added no declaration, evaluator, numerical-policy object, operation, attribute, Tensor method,
  data type, FLOAT16 support, test, derivative convention, compiler implementation, backend
  behavior, dependency, or build change.

## Completion summary

Completed the portable represented-value contracts for all floating comparisons, binary and scalar
MIN/MAX, and ordered first-class CLAMP. Finalized the five affected Javadocs, Tensor API, Compile
API, glossary, and three planning indexes without changing executable Java or any public surface.

Files changed: the exact five Javadoc-only Java files and seven documentation/planning files listed
under [Affected files](#affected-files).

Validation: the final model compile/test/Javadoc command passed; 127 suites and 1,018 tests passed;
generated Javadoc was inspected; Markdown, local links/anchors, fences, final newlines, trailing
whitespace, exact scope, Java-comment isolation, status/dependencies, absence of later compiler
specifications, semantic wording, and `git diff --check` passed.

Documentation review: Tensor API, Compile API, glossary, and planning required updates. Model
capabilities/audits, Training/Runtime APIs, architecture/ADRs/tests, Gradle, other modules,
backend-conformance, and integration required no change for the reasons recorded above.

Unresolved issues: None within task scope.

Required follow-up: Compiler 0005A may be separately planned to choose derivative policy while
reusing this fixed forward contract. Future backends must receive conformance coverage when they
implement these operations.

Status: Complete

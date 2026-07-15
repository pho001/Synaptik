# Task 0003A: Exact Arithmetic Rewriting

## Status

Complete

## Goal

Add one deliberately narrow compiler-owned arithmetic-rewrite scan over a successful canonical
`ValidatedGraph`. The closed rule set contains exactly these seven semantic rules:

```text
BinaryArithmeticKind.MIN(x, x) -> x
BinaryArithmeticKind.MAX(x, x) -> x
ScalarElementwiseKind.MUL(x, exact typed +1) -> x  [BFLOAT16/FLOAT32/FLOAT64/INT32/INT64]
ScalarElementwiseKind.DIV(x, exact typed +1) -> x  [BFLOAT16/FLOAT32/FLOAT64]
ScalarElementwiseKind.POW(x, exact typed +1) -> x  [BFLOAT16/FLOAT32/FLOAT64]
ScalarElementwiseKind.ADD(x, exact typed 0) -> x   [INT32/INT64]
ScalarElementwiseKind.SUB(x, exact typed 0) -> x   [INT32/INT64]
```

Every rewrite is permitted only for an internal one-output `FORWARD` occurrence whose output
descriptor equals the input descriptor in all four components, whose equal descriptor has
`requiresGrad == false`, and whose output is not a graph output. Binary extrema additionally
require both already-remapped input positions to identify the same graph value. Scalar rules
require the exact kind, `ScalarValueAttrs`, scalar data type, and typed scalar value listed above.
These conditions preserve descriptor, phase, occurrence-boundary, future-autograd, exceptional-
value, modular-integral, and public-output contracts.

Insert one topological rewrite scan after task 0003's mandatory canonicalization and validation,
but before task 0003's unchanged one-shot optional sequence:

```text
successful captured-graph validation
  -> mandatory canonicalization
  -> Compiler 0002 validation
  -> optional exact arithmetic rewrite scan
  -> Compiler 0002 validation when changed
  -> optional DCE -> CSE -> DCE
  -> Compiler 0002 validation after each changed candidate
  -> successful canonical ValidatedGraph
```

Here dead-code elimination (DCE) and common-subexpression elimination (CSE) retain the exact
meanings established by task 0003. Placing rewriting before that existing sequence lets its first
DCE remove newly unreachable dependencies and lets CSE observe rewritten inputs, without adding
another cleanup loop or changing the completed `DCE -> CSE -> DCE` order.

`ScalarValueAttrs` and its `ScalarValue` are immutable compiler-visible operation metadata with an
exact type and representation; inspecting them does not read Tensor storage, execute a value, or
invent a compiler constant. Tensor zeros/ones, new result constants, and rewrites that depend on
Tensor values remain Compiler 0003B work. This task intentionally excludes every rule outside the
seven-row table, including floating ADD/SUB zero, multiplication by zero, cancellation,
reassociation, commutation, bounds/clamp identities, Tensor constants, and relaxed fast math.

## Scope

- Add one package-private stateless helper with this exact shape:

  ```java
  static CompiledGraphModel rewrite(CompiledGraphModel graph)
  ```

- Name the helper `ForwardExactArithmeticRewriting` and keep it package-private, final, and
  stateless in `io.github.pho001.synaptik.compiler`, with a private constructor and no instance
  fields.
- Reject a null graph with `NullPointerException("graph")`.
- Consume only a graph that has already passed Compiler 0002 validation. The helper is a
  transformer, not another structural or semantic validator.
- Visit nodes once in stored topological order and use already-remapped input IDs, so one earlier
  rewrite can make a later selected occurrence eligible during the same bounded scan.
- Rewrite exactly the seven semantic rules when every applicable condition in
  [Selected exact rules](#selected-exact-rules) holds.
- Return the exact graph argument when no occurrence rewrites.
- When at least one occurrence rewrites, return one new immutable graph in the same dense
  input-first, retained-node/output-slot canonical allocation order used by task 0003.
- Preserve exact operation, descriptor, and phase references for retained nodes and values.
- Preserve every graph input in exact boundary order and every graph output in exact boundary
  order. Do not merge, replace, or remove a graph-output producer.
- Integrate the helper into `ForwardGraphOptimization` immediately after canonicalization's
  successful Compiler 0002 validation and before the first DCE invocation.
- Run the helper only when `GraphOptimizationConfig.optionalOptimizationsEnabled()` is true.
  `disabled()` remains canonicalization plus validation only.
- Immediately validate a changed rewrite candidate through
  `CapturedGraphInference.inferAndValidate`. Do not revalidate the exact unchanged graph.
- Preserve task 0003's existing one-shot `DCE -> CSE -> DCE` sequence exactly once after the
  rewrite boundary, including validation after each changed candidate and no fixed-point loop.
- Add meaningful Javadocs to the new helper and update the orchestrator Javadocs for the exact
  ordering, identity behavior, immutable result, and failures.
- Add focused helper and orchestration tests for the complete seven-rule proof, negative-rule
  matrix, descriptor and boundary guards, phase behavior, deterministic remapping, task-0002
  revalidation, and bounded source order.
- Update the Compile API, targeted Tensor API status text, glossary, this task, compiler master
  plan, and roadmap after implementation.
- Complete the required separate documentation-focused review in the same overall change.

### Selected exact rules

For a node encountered in topological order, rewrite its sole output to its already-remapped input
only when all of the following common conditions are true:

1. `graph.nodePhases().get(node.id()) == GraphPhase.FORWARD`;
2. the operation kind and exact attributes class match one row of the closed rule table below;
3. the validated operation occurrence has the row's fixed input count and exactly one output, as
   guaranteed by its family signature and consumed through ordinary collection access rather
   than a new validator;
4. the output is not present in `graph.outputs()`;
5. the output `TensorDescriptor` equals the bypass input descriptor, covering data type,
   `Shape`, resolved-or-unresolved layout, and `requiresGrad`; and
6. that equal descriptor has `requiresGrad == false`.

The row-specific conditions are exactly:

| Rule | Kind and attributes | Data types | Exact value check |
|---|---|---|---|
| duplicate minimum | `BinaryArithmeticKind.MIN` with `NoOperationAttrs.INSTANCE` | all model-valid floating and integral binary domains | the two already-remapped input IDs are equal |
| duplicate maximum | `BinaryArithmeticKind.MAX` with `NoOperationAttrs.INSTANCE` | all model-valid floating and integral binary domains | the two already-remapped input IDs are equal |
| scalar multiply by one | `ScalarElementwiseKind.MUL` with exact `ScalarValueAttrs` | `BFLOAT16`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64` | typed positive one |
| scalar divide by one | `ScalarElementwiseKind.DIV` with exact `ScalarValueAttrs` | `BFLOAT16`, `FLOAT32`, `FLOAT64` | typed positive one |
| scalar power of one | `ScalarElementwiseKind.POW` with exact `ScalarValueAttrs` | `BFLOAT16`, `FLOAT32`, `FLOAT64` | typed positive one |
| scalar add zero | `ScalarElementwiseKind.ADD` with exact `ScalarValueAttrs` | `INT32`, `INT64` | typed zero |
| scalar subtract zero | `ScalarElementwiseKind.SUB` with exact `ScalarValueAttrs` | `INT32`, `INT64` | typed zero |

For every scalar row, `attrs.value().dataType()` must equal both the input and output descriptor
data type. Test the immutable `ScalarValue` through its existing public typed inspector only:

- `FLOAT64`: `float64Value() == 1.0d`;
- `FLOAT32`: `float32Value() == 1.0f`;
- `BFLOAT16`: `bfloat16Bits() == (short) 0x3F80`, the sole exact positive-one encoding;
- `INT32`: `int32Value() == 1` for one and `int32Value() == 0` for zero; and
- `INT64`: `int64Value() == 1L` for one and `int64Value() == 0L` for zero.

Call an inspector only after switching on the exact `ScalarValue.dataType()`. BOOL, floating zero
for ADD/SUB, integral DIV/POW, and every type/kind pairing absent from the table are ineligible.
Do not compare diagnostic text, convert through `double`, allocate replacement scalars, use
reflection, or add a generic scalar evaluator.

If any condition is false, retain the exact occurrence unchanged. The helper does not partially
rewrite, repair, normalize, or reject it.

For an eligible occurrence, map the original output to the already rebuilt bypass input and do
not retain the node or output value. Later nodes consume that remapping. This is a whole-occurrence
bypass: every selected family has exactly one output, and no multi-output family is eligible.

### Exactness proof

The selected rules are exact in their closed domains when gradient eligibility is false:

- **Floating values:** Current `MIN` propagates NaN, orders infinities normally, and selects
  negative zero for opposite signed zeros; current `MAX` propagates NaN, orders infinities
  normally, and selects positive zero for opposite signed zeros. Comparing one value with itself
  therefore preserves its value class for finite values, either infinity, either signed zero, and
  every NaN. The model promises no NaN payload or bitwise result, so bypassing the occurrence does
  not discard a promised payload transformation.
- **Integral values:** Current `MIN` and `MAX` use signed order. Selecting between one signed value
  and itself yields that value and performs no ADD, SUB, or MUL, so fixed-width modular overflow is
  not involved.
- **Scalar multiplication by positive one:** The exact typed multiplier is the multiplicative
  identity for every current floating and fixed-width signed-integral scalar domain. Floating
  finite values, either infinity, either signed zero, and NaN classification are preserved;
  integral modular arithmetic also preserves every bit pattern. No NaN-payload or bitwise-result
  promise is claimed by the model.
- **Scalar division and power by positive one:** DIV and POW are currently floating-only. Dividing
  by exact positive one and using exact positive-one exponent preserve the input semantic value,
  including finite values, infinities, signed zeros, and NaN classification, without selecting a
  numerical algorithm or promising a NaN payload. The bypass relies only on the current ordered
  base/exponent and input/divisor semantics; it does not generalize to any other exponent.
- **Integral scalar addition and subtraction of zero:** Exact typed zero is an identity in both
  current signed fixed-width domains, including at their endpoints; modular overflow is unchanged
  because the result is the original bit pattern. Floating ADD/SUB are excluded because opposite
  zero signs can change the result sign under IEEE-754 addition or subtraction.
- **BOOL:** Binary arithmetic rejects BOOL during model construction and Compiler 0002 validation.
  The helper receives only a successful `ValidatedGraph`, so BOOL can never satisfy the selected
  operation rule.
- **Promotion and mixed types:** Equal input IDs identify one logical graph value and therefore one
  descriptor and data type. Numeric promotion is idempotent for equal types. A mixed-type pair
  necessarily uses two distinct graph values and is not eligible, even if one concrete execution
  could happen to contain equal element values.
- **Shape and broadcasting:** Broadcasting a Shape with the same Shape is structurally
  idempotent. The stronger descriptor-equality guard also prevents bypass when result metadata
  differs for any reason, including resolved input layout versus the unresolved arithmetic result.
- **Gradient eligibility and future autograd:** No current autograd contract proves that deleting
  any selected operation preserves its gradient occurrence, saved-value, or operand-multiplicity
  semantics. Duplicate extrema additionally need a future tie-gradient rule. The explicit
  `requiresGrad == false` guard therefore retains every potentially differentiable occurrence.
  Integral descriptors already require false; floating descriptors rewrite only when the caller
  did not request gradient eligibility.
- **NaN, infinity, and signed zero:** Safety follows from the current extrema and positive-one
  identity semantics, not from an assumption that exceptional values are absent. No finite-only
  or nonzero precondition is introduced.
- **Producer and provenance:** `TensorProducer` and `TensorProvenance` are pre-capture model
  identities and are not retained in `CompiledGraphModel`. Rewriting an internal compiled
  occurrence does not intern or mutate public Tensor expressions. The graph-output exclusion
  preserves every requested expression occurrence at the publication boundary.
- **Graph outputs and public identity:** A node producing a graph output is never rewritten. Graph
  input/output ordering and distinct output IDs remain intact. Changed internal graph-local IDs
  are rebuilt deterministically and are not public Tensor IDs.
- **Phase:** Only `FORWARD` occurrences are eligible. Current or future `BACKWARD` occurrences are
  retained even if their local shape resembles the selected rule; post-autograd rewriting remains
  task 0004 work.
- **Multi-output occurrences:** Every selected binary or scalar kind has a fixed one-output
  signature. Every multi-output occurrence is retained.
- **Attributes and constants:** Binary selected kinds accept `NoOperationAttrs.INSTANCE`. Scalar
  selected kinds accept exact `ScalarValueAttrs`; their immutable `ScalarValue` is already a
  semantic operation parameter visible in the graph. Reading its typed value is not Tensor
  constant discovery, constant propagation, result construction, or host-storage inspection.

### Rebuild and identity contract

Use one construction-local mapping from each original value ID to its rebuilt or bypass target.
Allocate all graph inputs first in boundary order. For each retained node, remap ordered inputs,
allocate all outputs in output-position order, then allocate the node and retain its exact phase.
For each rewritten node, map its sole original output to the already-remapped bypass input and
allocate neither an output nor a node.

The changed result must be a structurally closed immutable `CompiledGraphModel`. Its graph inputs
and outputs preserve exact order, retained nodes preserve topological order, retained input
positions remain ordered, and retained operations/descriptors/phases retain exact references.
The helper may allocate construction-local maps and lists but retains no mutable index or rewrite
state after return.

Return the exact input graph reference when no rewrite occurs. A changed result makes no identity
promise for graph, node, value, ID, or list containers. This matches task 0003's helper contract
and allows the orchestrator to skip redundant validation on no change.

### Exact pipeline placement and cleanup

`ForwardGraphOptimization.optimize` keeps its current argument validation and mandatory work:

1. reject null `validatedGraph`;
2. reject null `optimizationConfig`;
3. canonicalize `validatedGraph.graph()`;
4. validate the canonical candidate through Compiler 0002; and
5. if optional optimization is disabled, return that exact validated canonical result.

When optional optimization is enabled, continue exactly:

6. invoke `ForwardExactArithmeticRewriting.rewrite` once;
7. if changed, validate the rewrite candidate once and use that exact `ValidatedGraph`;
8. invoke task 0003's first forward DCE once and validate only if changed;
9. invoke task 0003's exact forward CSE once and validate only if changed;
10. invoke task 0003's cleanup forward DCE once and validate only if changed; and
11. return the last exact successful `ValidatedGraph`.

This is the complete cleanup policy. Do not run a second rewrite scan, another CSE, another DCE,
or a convergence loop. A single topological rewrite scan uses prior remappings, and the existing
one-shot sequence supplies all authorized cleanup.

### Explicitly rejected rewrite matrix

The helper is a closed typed check for the seven selected rules, not a registry. Tests and Javadocs
must keep these tempting transformations absent:

| Candidate | Classification under current contracts |
|---|---|
| Tensor `x + 0`, `x - 0`, `x * 1`, `x / 1`, or `x ** 1` | Not selected. Recognizing a Tensor zero/one requires Compiler 0003B immutable constant facts. The seven scalar rows inspect only immutable operation attributes. |
| floating scalar ADD/SUB with either zero sign | Not selected. Signed zero makes a generic floating zero-identity rule invalid, and this closed task does not split out narrower zero-sign/operator cases. |
| scalar or Tensor `x * 0 -> 0` | Unsafe for NaN, infinity, and signed-zero sign; constructing a result zero also requires Compiler 0003B constant representation. |
| scalar `x * +1`, `x / +1`, or `x ** +1` | Selected only for the exact data-type rows above. Integral DIV/POW do not exist in the current scalar domain, BOOL is invalid, and every non-one typed bit pattern remains. |
| scalar integral `x + 0` or `x - 0` | Selected only for exact `INT32`/`INT64` zero. Floating variants remain excluded. |
| `x / x -> 1` | Unsafe for zero, infinity, and NaN; no result constant representation exists. |
| `x - x -> 0` | Unsafe for infinity and NaN and can change signed-zero behavior; no result constant representation exists. |
| reassociation or distributivity | Not valid as one current-family rule: IEEE-754 rounding, overflow, underflow, and NaN/infinity behavior can change, while mixed-width integral trees can select different promoted intermediate domains. Same-width modular subcases are not selected here. |
| commutation or canonical operand reordering | Not selected. Ordered operand positions remain semantic, and no numerical or identity benefit justifies changing evaluation/provenance order. Exact CSE continues to use ordered inputs. |
| `x ** 0`, `x ** 2`, `x ** -1`, or other small-integer exponent rewrites | Not selected. Zero, infinities, NaN, negative bases, signed zero, rounding, and instruction sequencing need explicit numerical/conformance contracts. Exponents 2, -1, and other small integers are future backend-prepare strength-reduction candidates, not graph rewrites, if a backend can prove the applicable contract. |
| extrema identities | Only non-gradient binary `MIN(x, x)` and `MAX(x, x)` under the exact guards above are selected. Scalar bounds, different inputs, gradient-eligible occurrences, operand reordering, and reduction extrema are not rewritten. |
| clamp rewrites | Not selected. `CLAMP` is a first-class occurrence; NaN endpoints are accepted, bound equality alone does not determine the input, and decomposition would change occurrence structure. |
| reciprocal/division substitution | Not selected. `RECIPROCAL` and `DIV` are distinct first-class semantics, and composition can change rounding and exceptional-value behavior. |
| mixed-type bypass | Unsafe. Promotion can change the operation domain and result descriptor; equal runtime values do not make distinct typed graph values interchangeable. |

Compiler 0003B may later use compiler-owned immutable constant facts for separately proved exact
constant folding and Tensor-constant identities. A future explicit numerical-permission contract
may permit relaxed reassociation, finite-only assumptions, reciprocal substitution, or other
fast-math transformations. Future backend prepare may consider POW strength reduction for
exponents such as 2, -1, or other small integers only under explicit numerical and backend-
conformance contracts. This task does not edit or pre-decide backend plans.

## Out of scope

- any arithmetic rewrite other than the exact guarded seven-row rule set
- scalar-attribute inspection outside the selected `ScalarValueAttrs` typed positive-one and
  integral-zero checks, including negative zero, NaN, infinity, or clamp-bound recognition
- reading public Tensor host storage or treating provenance-free leaves as constants
- a compiler-owned constant fact/ingress representation, constant propagation, constant folding,
  partial evaluation, eager value execution, or Compiler 0003B
- floating ADD/SUB zero; MUL zero; `x - x`; `x / x`; POW zero, two, negative one, or other
  small-integer graph rewrites; scalar MIN/MAX; unary, reduction, comparison, logical, selection,
  cast, layout, indexing, normalization, loss, random, or multi-output rewrites
- reassociation, commutation, operand sorting, canonical operand reordering, distributivity,
  strength reduction, reciprocal substitution, fast math, finite-only assumptions, or any relaxed
  numerical permission
- modification of task 0003's completed implementation history or its detailed specification
- another optional cleanup pass, pass repetition, fixed-point iteration, candidate comparison, or
  graph search
- a pass manager, pass/rewrite registry, generic optimizer framework, annotation, reflection,
  service loader, plugin, callback, string dispatch, or public pass enum
- public compiler, optimizer, rewrite, pass, candidate, result, diagnostic, or configuration API
- graph capture, inference-rule, deferred-constraint, model semantic, Tensor, operation,
  descriptor, graph DTO, config API, planning, trace, runtime, prepare, engine, backend, or
  training changes
- numerical diagnostics, numerical-permission APIs, autograd, backward graph construction,
  backward rewrite application, saved values, gradient
  rules, gradient accumulation, or post-autograd optimization
- publication binding, capability analysis, owner selection, partitions, logical memory,
  `CompileArtifacts`, or diagnostics
- lowering, fusion, specialization, kernel or route selection, physical memory, schedule,
  executable, runtime state, residency, or execution
- module dependencies, architecture rules/docs/tests, ADRs, Gradle/build configuration, Java
  version, backend conformance tests, integration tests, or another module's executable tests
- a detailed Compiler 0003B, 0004, or 0005 specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially compiler ownership,
  immutable compile-time state, dependency rules, and the compile lifecycle
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)
- [Planning documentation style](../../../../developer-guide/documentation/planning-style.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Compiler master plan](../master-plan.md)
- [Compiler task 0001: graph capture](0001-tensor-expression-graph-capture.md)
- [Compiler task 0002: inference and validation](0002-captured-graph-inference-and-validation.md)
- [Compiler task 0003: canonicalization and forward optimization](0003-canonicalization-and-forward-optimization.md)
- [Model master plan](../../model/master-plan.md)
- [Model capability baseline](../../model/capabilities.md)
- [Model task 0014A: binary arithmetic semantic kinds](../../model/tasks/0014a-binary-arithmetic-semantic-kinds.md)
- [Model task 0014B: binary arithmetic Tensor expressions](../../model/tasks/0014b-binary-arithmetic-tensor-expressions.md)
- [Model task 0018N: typed scalar values](../../model/tasks/0018n-typed-scalar-value-contract.md)
- [Model task 0018T: scalar arithmetic normalization](../../model/tasks/0018t-scalar-arithmetic-family-normalization.md)
- [Model task 0018U: integral elementwise arithmetic](../../model/tasks/0018u-integral-elementwise-arithmetic-and-comparisons.md)
- [Compile API](../../../../api/compile-api.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `Tensor` remains public mutable API state and is not graph IR. This task reads no Tensor,
  `TensorProducer`, `TensorProvenance`, label, or host storage.
- `Operation` remains backend-independent semantic state and exposes no backend support.
- `CompiledGraphModel` remains immutable compile-time state. The helper constructs no physical,
  prepared, backend-specific, executable, or runtime state.
- Compiler owns semantics-preserving graph rewriting and may reuse Compiler 0002 to validate each
  changed candidate.
- Optional rewriting remains controlled only by the existing coarse
  `GraphOptimizationConfig` permission. Pass identities and order do not enter config.
- The compiler keeps its existing allowed dependencies and no dependency direction changes.
- Runtime hot paths continue to see neither `Operation` nor `CompiledNode`.
- `ARCHITECTURE.md` remains unchanged. Stop if implementation needs an architecture, dependency,
  model-semantics, config-surface, or public-lifecycle change.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.compiler` — owns the package-private helper, orchestration, and
  same-package tests.
- `io.github.pho001.synaptik.model.graph` — supplies immutable graph, node, value, identifier, and
  phase contracts; read only.
- `io.github.pho001.synaptik.model.operation.elementwise.binary` — supplies the selected typed
  `MIN` and `MAX` semantic identities; read only.
- `io.github.pho001.synaptik.model.operation.elementwise.scalar` — supplies the selected scalar
  kinds and immutable exact `ScalarValueAttrs`; read only.
- `io.github.pho001.synaptik.model.datatype` — supplies exact scalar data types and typed
  `ScalarValue` inspectors; read only.
- `io.github.pho001.synaptik.model.tensor` — supplies immutable descriptor values; read only.
- `io.github.pho001.synaptik.config.compile` — supplies the existing coarse optional-optimization
  permission; read only.

Packages added or changed:

- No package is added. The compiler root package remains the cohesive internal capture,
  validation, canonicalization, and bounded forward-transformation boundary.

Type placement:

- `io.github.pho001.synaptik.compiler.ForwardExactArithmeticRewriting` — package-private stateless
  one-scan exact forward arithmetic transformer, colocated with the graph representation and
  orchestrator it supports.
- `io.github.pho001.synaptik.compiler.ForwardExactArithmeticRewritingTest` — same-package focused
  contract tests without widening production visibility.
- `io.github.pho001.synaptik.compiler.ForwardGraphOptimization` — existing package-private
  orchestrator updated only to place and validate the new helper before its unchanged task-0003
  sequence.

No generic rewrite subpackage, public optimizer package, rule registry, or cross-package surface is
justified by seven closed typed rules.

## Affected files

Expected implementation paths:

- add
  `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ForwardExactArithmeticRewriting.java`
- add
  `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/ForwardExactArithmeticRewritingTest.java`
- update
  `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ForwardGraphOptimization.java`
- update
  `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/ForwardGraphOptimizationTest.java`
- update [Compile API](../../../../api/compile-api.md)
- update targeted current/planned wording in [Tensor API](../../../../api/tensor-api.md)
- update [Glossary](../../../../glossary.md)
- update and finalize this task specification
- update [Compiler master plan](../master-plan.md)
- update [Roadmap](../../../roadmap.md)

Review without modification unless a documented conflict requires stopping:

- task 0003 and tasks 0001–0002;
- `ElementwiseInference`, graph capture/canonicalization/CSE/DCE, and unrelated compiler inference
  helpers/tests;
- `BinaryArithmeticKind`, `ScalarElementwiseKind`, `ScalarValueAttrs`, unrelated scalar/unary kinds
  and attributes, `DataType`, `ScalarValue`,
  `DataTypePromotion`, `TensorDescriptor`, `TensorProducer`, `TensorProvenance`, Shape/broadcast,
  graph DTOs, and their focused tests;
- Public API, Training API, model capabilities/master plan, config Javadocs/master plan,
  architecture documents/tests, Gradle files, other modules, backend conformance, and integration
  paths.

## Maximum scope

This task may create or modify at most the exact ten paths listed under
[Affected files](#affected-files): two production paths, two test paths, and six documentation or
planning paths.

If implementation or accurate documentation requires an eleventh path, stop and report the
concrete reason rather than silently expanding scope. In particular, do not modify task 0003,
model/config source, architecture, dependencies, or build configuration to make the rewrite fit.

## Acceptance criteria

- The compiler contains one new package-private final stateless
  `ForwardExactArithmeticRewriting` type with one package-private static
  `rewrite(CompiledGraphModel)` entry point and no public declaration.
- A null graph fails with `NullPointerException("graph")`.
- The helper implements exactly seven eligible forward rules: binary `MIN(x, x)`, binary
  `MAX(x, x)`, scalar MUL by exact typed positive one in all five current numeric domains, scalar
  DIV and POW by exact typed positive one in the three current floating domains, and scalar ADD
  and SUB by exact typed zero in the two current integral domains.
- Scalar eligibility requires exact `ScalarValueAttrs`, scalar/input/output data-type equality, and
  the matching typed inspector: binary64 `1.0d`, binary32 `1.0f`, BFLOAT16 bits `0x3F80`, signed
  integer one, or signed integer zero. BOOL, conversions, text comparison, and every kind/type/value
  combination outside the closed table remain ineligible.
- Eligibility uses already-remapped equal input IDs, so a later eligible occurrence may be exposed
  by an earlier rewrite in the same single topological scan.
- Floating duplicate extrema and MUL/DIV/POW-by-positive-one safety includes NaN classification,
  either infinity, and both signed zeros without finite-only or nonzero assumptions. Integral
  extrema and scalar MUL/ADD/SUB identities preserve fixed-width modular values. BOOL and mixed-
  type cases cannot rewrite.
- Exact descriptor equality covers data type, Shape, layout, and `requiresGrad`, and the equal
  descriptor must have `requiresGrad == false`. A valid arithmetic result with unresolved layout
  does not bypass a resolved-layout input, and every gradient-eligible occurrence remains intact
  for task 0004's future tie and operand-multiplicity decisions.
- No graph-output producer, non-forward node, multi-output node, aggregate reduction, different-
  input extrema occurrence, or scalar occurrence outside the exact typed table rewrites.
- A changed graph preserves every input/output boundary position, retained topological order, all
  retained ordered input positions and output slots, and exact retained operation, descriptor,
  and phase references while rebuilding dense graph-local IDs.
- An unchanged helper returns the exact graph argument; a changed helper returns a new immutable
  structurally closed graph and mutates no input object or collection.
- `ForwardGraphOptimization` invokes the rewrite exactly once after canonicalization validation and
  before the first DCE. A changed candidate is validated through Compiler 0002 before DCE consumes
  it; an unchanged candidate is not revalidated.
- The existing optional `DCE -> CSE -> DCE` sequence remains in exact order, runs once, and has no
  added loop or extra cleanup invocation. Disabled optimization skips rewriting and all three
  task-0003 optional passes while retaining canonicalization and validation.
- Focused tests prove direct non-gradient MIN/MAX rewrites across every floating and integral data
  type and every selected scalar kind/type row, including BFLOAT16 raw positive-one matching,
  explicit retention of gradient-eligible floating occurrences, descriptor/layout blocking,
  graph-output and phase blocking, same-pass remapping, no-change identity, deterministic rebuild,
  immutability, and exact boundary/descriptor/phase retention.
- Negative tests retain floating scalar ADD/SUB with both zero signs; scalar MUL by zero or any
  non-one; DIV/POW by non-one; POW by zero, two, negative one, and other small integers; x-minus-x;
  x-divided-by-x; Tensor-constant-shaped identities; scalar MIN/MAX and clamp; unary reciprocal;
  bounds identities; reductions; distinct-input extrema; mixed-type cases; gradient-eligible
  occurrences; and a genuine multi-output occurrence. Exact typed checks exclude every other
  family, domain, carrier, and value.
- Pipeline tests prove rewrite-before-DCE/CSE behavior, task-0002 constraint regeneration after a
  changed rewrite candidate, source-level one-shot ordering, and absence of iteration or a second
  rewrite invocation.
- Production source and rewrite-semantic tests contain no Tensor constant/storage inspection,
  generic scalar evaluator, pass registry, reflection-based dispatch, annotation, string dispatch,
  candidate collection, autograd,
  publication/planning,
  trace, prepare/runtime/backend/engine, dependency, build, or architecture behavior.
- Javadocs explain exact inputs, immutable and identity behavior, result, ordering, and failures
  with complete `@param`, `@return`, and applicable `@throws` coverage.
- Compile API, targeted Tensor API text, and glossary describe the current internal guarded seven-
  rule rewrite and distinguish immutable scalar operation attributes from Compiler 0003B Tensor
  constants, without claiming value execution, broader algebra, constant folding, public
  compilation, gradients, preparation, backend support, or runtime execution.
- Config Javadocs/master plan require no change because their coarse boolean permission does not
  expose pass identity or order. Public API and Training API require no change because no public
  Java, training, gradient, or optimizer contract changes.
- Model capabilities/master plan and model source/tests require no change because this task
  consumes their established semantics without changing model representation or public behavior.
- Architecture docs/tests, Gradle, dependencies, other modules, conformance, and integration paths
  remain unchanged for recorded reasons.
- Exactly the ten authorized paths change. Compiler task 0003 remains `Complete`; this task, its
  master-plan row, and the roadmap become `Complete` only after implementation, documentation, and
  validation finish. Compiler 0003B, 0004, and 0005 remain Draft rows without detailed specs.
- A separate documentation-focused agent pass has finalized affected source Javadocs, API text,
  glossary impact, planning status/evidence, terminology, links, and no-change conclusions in this
  same overall change.

## Tests / validation

The focused helper suite must parameterize the positive matrix rather than sampling it:

- binary MIN and MAX for `BFLOAT16`, `FLOAT32`, `FLOAT64`, `INT32`, and `INT64` with repeated
  already-remapped input IDs;
- scalar MUL with exact typed positive one for `BFLOAT16`, `FLOAT32`, `FLOAT64`, `INT32`, and
  `INT64`;
- scalar DIV and POW with exact typed positive one for `BFLOAT16`, `FLOAT32`, and `FLOAT64`; and
- scalar ADD and SUB with exact typed zero for `INT32` and `INT64`.

The focused negative matrix must cover adjacent or easily confused typed values, including raw
BFLOAT16 patterns neighboring `0x3F80`; floating positive and negative zero for ADD/SUB; negative
one, zero, two, NaN, and infinities where valid; integral nonzero/non-one values; model-valid
unselected kind/domain combinations; valid descriptor inequality such as resolved input layout;
`requiresGrad == true`; graph outputs; BACKWARD phase; and the broader rejected-rule matrix.
Existing model and Compiler 0002 tests remain the evidence that invalid kind/attributes and
scalar/input data-type combinations cannot reach this transformer; helper tests must not invent
behavior for invalid input graphs. Tests must prove that the helper reads only `ScalarValueAttrs`
and typed `ScalarValue` accessors, never Tensor storage or a constructed result constant.

During implementation, run focused tests as needed:

```bash
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.ForwardExactArithmeticRewritingTest
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.ForwardGraphOptimizationTest
```

After executable Java stabilizes, the implementation context runs exactly one final compiler
module suite:

```bash
./gradlew :modules:compiler:test
```

This is task-tier validation. The change stays inside one module, adds no public API, dependency,
architecture boundary, shared build configuration, or second module's executable behavior.
Repository-wide and architecture validation remain deferred to the compiler
transformation-and-autograd capability checkpoint after task 0004 or continuous integration,
unless implementation reveals a concrete cross-module risk.

The implementation context hands the exact focused/final commands and XML test counts to the
documentation context. The documentation pass does not repeat successful Java tests unless it
changes executable Java behavior or records a concrete stale-evidence risk.

Documentation pass:

```bash
./gradlew :modules:compiler:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

The documentation pass must also verify:

- the new type and method are package-private, the constructor and implementation-only helpers are
  private, and no public declaration is added;
- complete meaningful source and generated-Javadoc coverage;
- all local Markdown links and anchors, balanced fences, final newlines, and trailing whitespace;
- exact ten-path scope and synchronized task/master/roadmap status;
- one rewrite invocation after canonical validation and before the unchanged single
  `DCE -> CSE -> DCE` sequence;
- Compiler 0002 validation after every changed candidate and none for an unchanged helper result;
- exact binary/scalar kind and attributes checks, already-remapped duplicate input IDs, exact typed
  scalar carriers and values, descriptor equality, `requiresGrad == false`, graph-output
  exclusion, forward phase, and one-output behavior;
- the complete negative-rule matrix and absence of Tensor constant/storage inspection, a generic
  scalar evaluator, registry, reflection, string dispatch, looped optimization, autograd,
  numerical diagnostics, downstream lifecycle, dependency/build, or architecture changes;
- task 0003 remains `Complete`; task 0003A, its master-plan row, and the roadmap are `Complete`
  after final evidence; and no detailed 0003B-or-later specification exists.

## Dependencies

- Compiler task 0001, deterministic structural graph capture — Complete.
- Compiler task 0002, complete captured-graph inference/validation and typed deferred constraints
  — Complete.
- Compiler task 0003, mandatory canonicalization and one-shot forward DCE/CSE/DCE — Complete.
- Model binary extrema, scalar arithmetic kinds, exact typed immutable scalar attributes,
  same-category numeric promotion, exact descriptors, immutable graph DTOs, and public provenance
  boundaries — Complete.
- Config task 0002, stable coarse optional-optimization permission — Complete.

Compiler 0003B constant representation/folding, autograd, planning orchestration, trace payloads,
runtime, prepare, training, and concrete backends are not dependencies. The selected rules are
provable from current immutable compile-time graph metadata, strict extrema semantics, and exact
typed scalar operation attributes alone.

## Follow-up tasks

Future Draft compiler rows, in order:

- 0003B — define compiler-owned immutable constant facts and ingress, then perform exact
  deterministic constant folding and any separately proved constant-based identities without
  reading mutable Tensor host storage as authoritative compile-time data.
- 0004 — construct backward graph state after forward rewriting/folding, define post-autograd
  optimization separately, and run the compiler transformation-and-autograd capability
  checkpoint.
- 0005 — define publication, planning, diagnostics, and immutable compile artifacts after their
  concrete collaboration boundaries are stable.

A future numerical-permission task may add relaxed reassociation, finite-only assumptions,
reciprocal substitution, or other fast-math behavior only after an explicit contract exists.
POW exponents 2, -1, and other small integers may be considered later as backend-prepare strength-
reduction candidates only when explicit numerical and backend-conformance contracts justify the
backend-local lowering; they are not Compiler 0003A graph rewrites, and this task changes no
backend plan.

## Architecture impact

Expected impact: None.

This task implements the architecture-authorized compiler ownership of exact arithmetic
simplification over immutable compile-time graph state. It changes no module ownership,
dependency direction, public lifecycle, config surface, artifact shape, numerical permission, or
execution boundary. If implementation requires changed model semantics, public identity rules,
another module's source, a dependency/build change, or architecture clarification, stop and
report the conflict.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the compiler master plan, and
docs/planning/modules/compiler/tasks/0003a-exact-arithmetic-rewriting.md. Read the directly
referenced compiler/model/config source, tests, API documentation, and completed contracts needed
to verify the complete rewrite-safety matrix.

Implement Compiler 0003A exactly within its ten authorized paths. Add only the guarded internal
non-gradient seven-rule set: duplicate binary MIN/MAX; exact typed scalar MUL-by-positive-one for
all current numeric domains; exact typed scalar DIV/POW-by-positive-one for current floating
domains; and exact typed scalar ADD/SUB-zero for current integral domains. Use immutable
ScalarValueAttrs metadata and existing typed ScalarValue inspectors only. Place the one scan after
canonical validation and before the unchanged one-shot DCE/CSE/DCE sequence, and revalidate a
changed candidate through Compiler 0002.
Do not add Tensor constants/folding, rules outside the seven-row table, broader algebra,
relaxed/fast-math, autograd, publication/planning/diagnostics, public surfaces, downstream
lifecycle work, or dependency/build/architecture changes. Stop on a scope, safety-proof, or
architecture conflict.

After executable implementation and the final compiler test evidence, hand the actual diff and
exact evidence to a separate documentation-focused agent or thread with clean context. That pass
must follow docs/developer-guide/documentation-rules.md, finalize affected Javadocs, Compile/Tensor
API text, glossary, planning status/evidence, terminology, links, and no-change conclusions in the
same overall change, and must not repeat successful Java tests unless executable behavior changes
or a concrete stale-evidence risk is recorded.

Update this task with local decisions, exact validation evidence, implementation notes,
completion summary, and final status. Do not mark it Complete before that pass and every
acceptance criterion finish.
```

## Local decisions

- Select exactly seven semantic rules: two duplicate binary extrema, three scalar positive-one
  identities across their explicit current domains, and two integral scalar-zero identities.
- Treat `ScalarValueAttrs` as immutable compiler-visible semantic metadata, not Tensor storage or a
  compiler-owned result constant. Match its existing exact type and typed value only; leave Tensor
  zero/one facts and new result constants to Compiler 0003B.
- Require complete descriptor equality before bypass. This deliberately leaves valid arithmetic
  nodes in place when an input has resolved layout and the arithmetic output is unresolved.
- Require `requiresGrad == false` even though it is already one descriptor component. Current
  forward value semantics prove each identity, but no task-0004 autograd contract yet proves that
  removing an operation preserves its gradient occurrence and saved-value semantics; duplicate
  extrema additionally carry repeated-position and tie-gradient uncertainty.
- Exclude graph-output producers rather than remap publication roots. Internal graph-local
  occurrence removal is permitted; requested public expression occurrences remain distinct.
- Use already-remapped inputs in one scan so the helper is locally complete without iteration.
- Place rewriting before task 0003's existing one-shot sequence. That sequence is the complete
  deterministic cleanup and receives no added pass or fixed-point loop.
- Keep the helper package-private and typed. Seven closed semantic rows do not justify a registry,
  framework, public optimizer, or new package.

## Known limitations

- Only internal non-gradient forward occurrences matching one of the exact seven semantic rows
  with equal input/output descriptors are rewritten.
- Every `requiresGrad == true` occurrence remains intact until task 0004 defines and can preserve
  its operand-multiplicity and tie-gradient semantics.
- A graph-output producer remains even when it otherwise matches, preserving requested occurrence
  identity at the boundary.
- Resolved-layout inputs normally prevent bypass because arithmetic results currently have
  unresolved layout.
- No Tensor constant, floating ADD/SUB-zero, MUL-zero, cancellation, POW-zero/two/negative-one,
  cast, view, reduction, bounds/clamp, reciprocal substitution, commutation, or reassociation rule
  is included.
- POW 2, -1, and other small-integer strength reductions remain possible future backend-prepare
  work subject to explicit numerical/conformance contracts; no backend plan is changed here.
- The helper runs only before autograd. Post-autograd and backward rewriting remain task 0004 work.
- No public compiler entry point exists; the result remains package-private internal graph state.

## Validation evidence

- Implementation-context focused rewrite suite:
  `./gradlew :modules:compiler:test --tests
  io.github.pho001.synaptik.compiler.ForwardExactArithmeticRewritingTest` — `BUILD SUCCESSFUL`;
  JUnit XML records 10 tests, zero skipped, zero failures, and zero errors.
- Implementation-context focused pipeline suite:
  `./gradlew :modules:compiler:test --tests
  io.github.pho001.synaptik.compiler.ForwardGraphOptimizationTest` — `BUILD SUCCESSFUL`; JUnit XML
  records 11 tests, zero skipped, zero failures, and zero errors.
- Implementation-context final executable validation after Java stabilization:
  `./gradlew :modules:compiler:test` — `BUILD SUCCESSFUL`; 12 JUnit XML suites record 80 tests,
  zero skipped, zero failures, and zero errors. The documentation context reused this evidence and
  did not rerun Java tests because it changed only Javadocs and documentation after stabilization.
- One redundant focused XML-count attempt initially met a non-authoritative sandbox Gradle lock
  and then returned the already cached successful result. It was not an executable validation
  failure and was not repeated by the documentation context.
- Required separate clean-context documentation-focused review (this context) applied the General,
  API/Javadoc, Planning, and Example profiles. It reviewed the final production/test diff, the
  compiler/model contracts that establish the rule matrix, Compile API, targeted Tensor API
  boundary, glossary, task, compiler master plan, roadmap, and generated Javadoc. It finalized both
  affected source Javadocs and all six authorized documentation/planning paths without changing an
  executable statement or test behavior.
- Documentation-context `./gradlew :modules:compiler:javadoc` — `BUILD SUCCESSFUL`; seven
  actionable tasks, two executed and five up-to-date. Generated pages for both affected internal
  types were inspected and contain the guarded rule set, ordering, identity/null contract, and
  `@param`, `@return`, and applicable `@throws` documentation.
- Documentation-context `python3 /tmp/validate_synaptik_markdown.py` — passed 235 Markdown files,
  4,276 local links, 262 local anchors, 2,976 fence markers, final newlines, and trailing
  whitespace.
- Documentation-context `javap -classpath modules/compiler/build/classes/java/main -p ...` for
  both affected production types — confirmed package-private final types, package-private static
  entry points, private constructors, private implementation helpers, and no public declaration.
- Source/manual/import inspection confirmed one rewrite invocation after canonical validation and
  before exactly one `DCE -> CSE -> DCE`; `validateWhenChanged` calls Compiler 0002 only for a
  changed candidate. The helper has one topological scan with prior remappings; exact typed
  `ScalarValueAttrs` checks; complete descriptor equality; false-gradient, internal graph-output,
  `FORWARD`, and one-output guards; and exact unchanged-reference behavior. Production uses no
  Tensor/storage/constants, generic evaluator, registry, reflection/string dispatch, orchestration
  loop, autograd, downstream lifecycle, backend, build, dependency, or architecture mechanism.
  The existing task-0003 test-only reflection assertion remains limited to checking the internal
  optimization type's visibility and is not rewrite dispatch or production behavior.
- Generated-Javadoc source inspection found the affected pages and verified the rendered rule,
  ordering, `ScalarValueAttrs`, gradient guard, DCE/CSE, and null-failure text.
- `git diff --check` — passed with no whitespace errors.
- `{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u` — exactly the ten
  authorized paths. `git status --short` reports only those same modified or untracked paths.
- No-change conclusions: Config Javadocs and its master plan remain accurate because the existing
  coarse boolean grants optional optimization permission without naming or ordering passes. Public
  API and Training API do not change because the implementation adds no public Java, training,
  gradient, or optimizer contract. Model capabilities/master plan/source/tests remain unchanged
  because the compiler only consumes established immutable graph, scalar, arithmetic, and
  descriptor semantics. Architecture documents/tests remain unchanged because ownership and
  dependency rules do not change. Gradle/dependencies, other modules, backend conformance, and
  integration remain unchanged because this is one internal compiler transformation with no
  build, backend, or end-to-end execution effect.

## Implementation notes

- Added package-private `ForwardExactArithmeticRewriting` with the exact seven typed rules and one
  deterministic topological rebuild scan. An unchanged scan returns its input graph by identity.
- Added focused positive and negative proof matrices for extrema, scalar typed values, descriptor,
  gradient, phase, graph-boundary, multi-output, same-scan remapping, identity, immutability, and
  rebuild behavior.
- Inserted the helper once after canonicalization validation and before task 0003's unchanged
  one-shot DCE/CSE/DCE sequence. The existing identity-based validation helper revalidates the
  rewrite result only when it changed.
- Finalized Javadocs and current/planned documentation around immutable `ScalarValueAttrs` versus
  future Compiler 0003B Tensor constants. No constant folding, broader algebra, public compiler,
  autograd, execution, backend, or architecture behavior was added or documented as current.

## Completion summary

- Completed changes: implemented and documented the guarded internal seven-rule exact arithmetic
  rewrite, its deterministic pipeline placement, complete focused proof, and synchronized planning
  status.
- Files changed or created: `ForwardExactArithmeticRewriting.java`,
  `ForwardExactArithmeticRewritingTest.java`, `ForwardGraphOptimization.java`,
  `ForwardGraphOptimizationTest.java`, `docs/api/compile-api.md`, `docs/api/tensor-api.md`,
  `docs/glossary.md`, this task, the compiler master plan, and the roadmap.
- Tests and validation: reused successful focused suites with 10 and 11 tests and the final
  compiler suite with 12 XML suites/80 tests; documentation Javadoc, Markdown, bytecode/source,
  generated-output, exact-scope, status, and whitespace checks all passed.
- Documentation-agent review: completed in the required separate clean context; no executable Java
  or test behavior changed during the pass.
- Documentation impact: Compile API, targeted Tensor API status, glossary, task, master plan, and
  roadmap now describe only the implemented internal boundary and its limitations.
- Javadoc review: both affected internal production contracts were finalized and generated output
  inspected successfully.
- Glossary impact: current compiler status now includes the bounded guarded scan and distinguishes
  scalar operation metadata from future Tensor constants.
- Unresolved issues: None.
- Follow-up required: None for task 0003A; Compiler 0003B, 0004, and 0005 remain Draft future work.

Status: Complete

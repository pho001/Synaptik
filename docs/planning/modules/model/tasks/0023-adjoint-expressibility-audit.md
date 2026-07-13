# Task 0023: Adjoint Expressibility Audit

## Status

Complete

## Goal

Audit whether the adjoint of every selected differentiable public Tensor operation can be written
exactly as a backend-neutral compiler composition of current general model operations and current
shared producer outputs.

The task produces one planning-only expressibility matrix, identifies the minimum generally useful
public Tensor primitives needed to close proven gaps, and inserts concise Draft follow-up rows
before task 0024. It does not add an operation kind, Tensor method, gradient rule, backward graph,
compiler pass, executable behavior, or detailed follow-up specification.

Mental model:

```text
current exact forward semantics
  -> exact adjoint formula for each differentiable input role
     -> current composition, existing auxiliary output, public-primitive gap,
        genuine semantic gap, non-differentiable role, or deferred policy
        -> smallest evidence-backed follow-up queue before 0024
```

This is deliberately an audit-only task. The selected operation surface spans elementwise,
reduction, linear algebra, stochastic, normalization, attention, convolution, pooling, layout,
indexing, ordering, and loss contracts. Current evidence already exposes independent questions
about dynamic-shape unbroadcasting, zero-base construction for scatter, public general-axis fold,
slice placement, convolution correlation geometry, pooling selection, and derivative conventions.
Choosing or implementing one primitive before the complete audit would preselect a solution and
could hide a second generally useful gap. No small implementation slice is logically inseparable
from recording these decisions.

## Scope

- Create `docs/planning/modules/model/adjoint-expressibility-audit.md` as the sole detailed
  planning artifact for the audit result.
- Define the artifact's notation for an output cotangent, input adjoint, reduction-to-input-Shape,
  saved forward value, recomputed forward value, and non-differentiable input.
- Inventory every current public Tensor semantic operation family whose floating input or output
  can participate in a differentiable expression. Treat conveniences such as `linear`,
  `embedding`, `transpose`, `flip`, `clampMin`, `clampMax`, and `unstack` through their exact
  current primitive producer chains rather than inventing separate semantic kinds.
- Inventory public operations and output roles that are non-differentiable so the boundary is
  explicit rather than silently omitted.
- Write exact backend-neutral vector-Jacobian-product or adjoint compositions at an appropriate
  family level. A family row may cover overloads only when their operation kind, differentiable
  input roles, formula, Shape obligations, special-value obligations, and classification are the
  same. Split rows by input role or semantic variant when any of those differ.
- Audit required forward inputs, public outputs, recomputable intermediates, and existing
  producer-described auxiliary outputs. In particular, inspect the dropout keep-mask slot,
  dropout state outputs, batch-normalization saved mean/inverse-standard-deviation slots, top-K
  value/index slots, and the absence of attention weights or maximum-pooling indices.
- Classify every audited row using the fixed decision vocabulary below and explain the evidence
  for the classification.
- Identify only the minimum generally useful public model primitives proved necessary by the
  completed matrix. Insert one concise Draft master-plan and roadmap row per selected primitive or
  cohesive primitive family immediately after 0023 and before 0024. Do not create a detailed task
  specification for any follow-up.
- Keep `capabilities.md`, the model master plan, and the roadmap synchronized with the audit
  result, follow-up queue, dependencies, statuses, and preserved completed history.
- Record explicit no-change conclusions for Tensor, Compile, Runtime, and Training API references,
  the glossary, architecture documents and tests, Java source and tests, Gradle, other modules,
  backend conformance, and integration tests.
- Use current repository contracts as the primary source. If an external formula or convention is
  needed to resolve a factual question, cite only a primary paper or official specification/API
  reference, minimally, and distinguish that evidence from Synaptik's selected semantics.

### Required family coverage

The matrix must cover at least these current public families and must add any current public
semantic family found during the final source/API inventory:

- binary and scalar arithmetic, clamp/minimum/maximum/power, unary numeric operations, modern
  activations, floating casts, and conditional selection;
- full, single-axis, ordered multi-axis, statistical, norm, and masked reductions, plus cumulative
  sum and extrema/arg-extrema distinctions;
- matrix multiplication and the exact primitive `linear` composition;
- softmax and log-softmax;
- explicit-state dropout and its hidden keep mask;
- layer, RMS, batch-inference, and batch-training normalization, including affine inputs and saved
  training statistics;
- scaled dot-product attention with and without explicit/causal masking;
- grouped NCHW convolution, maximum pooling, and count-padding average pooling;
- contiguous, reshape, expand, dimension insertion/removal, permutation/transpose, slice/flip,
  scalar select, padding, tiling, concatenation, stack/unstack, unfold, unfold2d, and fold2d;
- Gather, Gather Elements, Gather-ND, embedding, Scatter Elements, and Scatter-ND, including every
  selected scatter reduction whose differentiable roles differ;
- stable sort and top-K value outputs, with index outputs separated from value outputs; and
- mean-squared error plus dense-target and index-target categorical cross entropy with logits for
  every differentiable operand role and reduction.

Comparison, logical, floating-classification, one-hot, graph-RNG-state, argsort, arg-extrema, and
integral/index roles must appear as non-differentiable or explicitly policy-deferred boundaries as
appropriate. A true `requiresGrad` metadata flag is eligibility evidence only and is not proof that
an exact derivative policy has already been selected.

## Out of scope

- Java production or test changes
- a new or changed `OperationKind`, `OperationAttrs`, operation signature, Tensor method, result
  carrier, helper, descriptor, producer, provenance contract, or factory behavior
- any catalog or advance selection of operation-specific `*_BACKWARD`, `BACKWARD_*`, gradient,
  derivative, or adjoint kinds
- implementation of `foldAxis`, slice scatter/add, reduce-to-Shape, zero-like expression,
  convolution transpose/correlation, pooling indices, or any other candidate primitive
- a detailed task specification for 0024 or for any follow-up inserted by this audit
- choosing subgradient, discontinuity, extrema-tie, duplicate-target, or undefined-point policy
  merely to make a formula table complete
- compiler capture, automatic-differentiation traversal, gradient accumulation, backward graph
  construction, saved-value lifetime implementation, optimization, or publication
- backend capability declarations, lowering, fusion, specialization, algorithms, kernels,
  tolerances, execution, or performance promises
- runtime, prepare, engine, training-extension, or backend changes
- architecture, ADR, dependency, Gradle, build-logic, conformance-test, or integration-test changes
- copying legacy implementation structure or treating legacy gradient callbacks as current policy

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [Planning documentation style](../../../../developer-guide/documentation/planning-style.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)
- completed operation, signature, shared-producer, symbolic-Shape, typed-scalar, indexing, window,
  modern-model, normalization, and loss tasks from 0005 through 0022B

## Architecture constraints

- `modules/model` owns backend-independent operation semantics, reusable public Tensor primitives,
  descriptors, and pre-capture producer metadata. It does not own gradient rules or backward graph
  construction.
- `modules/compiler` owns automatic differentiation, exact gradient-rule selection, backward graph
  construction, saved-value capture/lifetime decisions, and post-autograd optimization.
- Backend prepare owns lowering, fusion, specialization, route selection, and kernels. Complexity,
  operation count, anticipated performance, or fusion opportunity is never evidence for a new
  model semantic kind.
- A stable tensor transformation useful outside automatic differentiation is planned as a general
  public model primitive and Tensor API capability, not hidden under a backward-only name.
- A compiler-only semantic kind is eligible only when the audit supplies a concrete proof that
  neither current exact composition nor an appropriate general public primitive can represent the
  meaning. Failure to find a short formula is not such a proof.
- Shared `TensorProducer` output descriptors and indexed provenance are the existing representation
  for genuine auxiliary forward values. The audit may require their compiler capture; it may not
  redesign producer identity, expose private outputs, or invent saved-state storage.
- Planning documents remain non-authoritative coordination artifacts. The audit must explain
  current contracts without changing `ARCHITECTURE.md` or claiming implementation.
- If exact semantics are absent or contradictory, classify the row as deferred policy or report an
  architecture/scope conflict. Do not invent behavior.

## Package impact

Existing packages used:

- All current `io.github.pho001.synaptik.model.operation.*` families and
  `io.github.pho001.synaptik.model.tensor` are inspected as evidence only.
- `io.github.pho001.synaptik.model.shape`, `.datatype`, and `.layout` are inspected where exact
  Shape, type, or view semantics constrain expressibility.

Packages added or changed:

- None. This task changes planning documentation only.

Type placement:

- None. Candidate types or methods may be named only as non-binding follow-up design questions or
  as the minimum selected public primitive in a concise Draft queue row after proof in the matrix.

## Affected files

Expected:

- `docs/planning/modules/model/adjoint-expressibility-audit.md`
- `docs/planning/modules/model/tasks/0023-adjoint-expressibility-audit.md`
- `docs/planning/modules/model/capabilities.md`
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Reviewed without modification; if the audit finds an actual contradiction, stop and report it:

- `ARCHITECTURE.md` and focused architecture documentation
- `docs/api/tensor-api.md`, `docs/api/compile-api.md`, `docs/api/runtime-api.md`, and
  `docs/api/training-api.md`
- `docs/glossary.md`
- current model production and test sources
- completed task specifications relevant to each matrix row

## Maximum scope

This task may create or modify exactly the five expected planning paths above. It must not change
Java, tests, API or glossary documentation, architecture or ADR files, Gradle, another module, or
create another task specification. If a correct audit requires a sixth path, stop and propose the
scope change before editing it.

## Required audit artifact

`adjoint-expressibility-audit.md` is authoritative only for subsequent planning decisions. It is
not an architecture contract, public API reference, gradient implementation, or backend promise.
`capabilities.md` must retain only a concise result summary and links; it must not duplicate the
matrix or become an autograd manual.

### Matrix fields

Each matrix row must contain or link locally to all of these fields:

1. public Tensor operation/family and exact `OperationKind`/attributes or primitive producer chain;
2. differentiable input role under audit and explicitly non-differentiable roles;
3. forward result and incoming cotangent Shape/type assumptions;
4. required forward inputs, public outputs, recomputed intermediates, and exact shared producer
   output positions, if any;
5. exact backend-neutral adjoint formula using current public general operations where possible,
   including every broadcast reversal, axis insertion/removal, permutation, reduction, scatter,
   and gradient-accumulation step needed to restore the input Shape;
6. operation-specific axis, rank, type, empty-domain, dynamic-Shape binding, index-bound,
   duplicate-target, random-state/mask, special-value, discontinuity, and tie obligations;
7. one fixed classification and a concise proof tied to current semantics;
8. the minimum general public primitive candidate, if classification requires one, plus why it is
   useful beyond autograd and why a narrower backward-only kind is rejected;
9. architecture owner for later implementation; and
10. repository evidence and any minimal primary external reference.

Rows may share a formula block or evidence block, but no field may be silently inherited across
semantically different variants. Mathematical notation must define shapes and axes. Pseudocode or
proposed API names must be labeled as planned and non-binding.

### Fixed classifications

Use exactly one of these classifications per differentiable input-role row:

- `EXACT_CURRENT_COMPOSITION` — current public operations represent the exact adjoint for every
  accepted current Shape/semantic case, without depending on an unavailable forward output.
- `EXACT_WITH_EXISTING_AUXILIARY_OUTPUT` — the exact composition uses a producer-described output
  that already exists, such as the dropout keep mask or batch-normalization saved statistics.
- `MISSING_GENERAL_PUBLIC_PRIMITIVE` — exact meaning is representable by a stable tensor
  transformation useful outside autograd, but that public model primitive is absent.
- `GENUINELY_NON_EXPRESSIBLE_SEMANTIC_GAP` — after rejecting both current composition and every
  appropriate general public primitive, a stable compiler-only semantic meaning remains. This
  classification requires an explicit impossibility argument, not convenience or performance.
- `NON_DIFFERENTIABLE` — the role is integral, Boolean, state, index, or otherwise outside the
  selected differentiable domain.
- `POLICY_DEFERRED` — current forward semantics do not select the derivative/subgradient behavior
  required at discontinuities, ties, duplicate targets, undefined points, or another observable
  boundary. Record the exact missing policy; do not choose it in this task.

If a family contains roles with different classifications, split it. If a formula works only for
fully static Shapes while the forward operation accepts unresolved Shapes, it is not
`EXACT_CURRENT_COMPOSITION` for the general row.

### Required decision probes

The audit must explicitly resolve or defer all of these questions:

- **Broadcast reversal:** prove how binary, `where`, matmul, attention, affine normalization, and
  loss adjoints reduce to each original input Shape. Treat a Dimension that may bind to one or to
  the result extent as a binding-dependent obligation; do not assume compile-time reduction axes.
- **Zero/one construction:** distinguish eager fully static `TensorFactory.zerosLike`/`onesLike`
  leaves from graph-expressible constants for unresolved Shapes. Prove how scatter bases, masks,
  and empty identities are constructed without contaminating NaN/infinity semantics.
- **Gather:** test Gather, Gather Elements, and Gather-ND adjoints against current Scatter Elements
  and Scatter-ND Shape, batch-dimension, bounds, duplicate-index, reduction, and data-base
  semantics. Use addition only where repeated indices require accumulation.
- **Scatter:** audit adjoints separately for data and updates and for `NONE`, `ADD`, `MUL`, `MAX`,
  and `MIN`; account for replacement uniqueness, duplicate reduction, base participation, extrema
  ties, and saved/recomputed values.
- **Slice/select:** prove whether current scatter operations, padding, or other current primitives
  place signed-step, multi-axis, empty, and dynamic-Shape slice cotangents exactly. If not, assess a
  general slice update/scatter/add primitive rather than a slice-backward kind.
- **Windows:** prove the `unfold` adjoint against the retained `FOLD_AXIS` semantics and current
  absence of public `Tensor.foldAxis`; distinguish `unfold2d`/`fold2d` and audit the reverse
  direction of every public fold.
- **Convolution:** write both input and weight adjoints for grouped NCHW cross-correlation,
  including stride, symmetric padding, dilation, groups, bias, output padding/cropping, and dynamic
  Shapes. Test composition through current unfold2d/fold2d, matmul, reductions, and layouts before
  proposing convolution-transpose or correlation primitives.
- **Pooling:** audit average-pool overlap accumulation and fixed count-padding divisor. For
  max-pool, preserve excluded padding, all-padding negative infinity, NaN selection, signed zero,
  and first logical-kernel tie semantics; assess recomputation versus generally useful public
  indices/selection output without inventing a tie rule.
- **Reduction extrema and piecewise operations:** distinguish a representation gap from an
  unselected subgradient/tie convention for min/max, clamp, absolute value, ReLU, sign, floor,
  ceil, sort/top-K, and related discontinuities.
- **Softmax, log-softmax, normalization, attention, and losses:** write exact formulas using public
  outputs or saved values where sufficient. Preserve current masked/all-masked, positive-infinity,
  NaN, empty-domain, correction, epsilon, ignore-index, and reduction-denominator semantics rather
  than relying on a generic textbook formula that covers only finite non-empty inputs.
- **Randomness:** dropout adjoints must reuse the exact forward keep mask and scaling. They must not
  resample, infer a mask from the output, advance state, or differentiate RNG-state inputs.
- **Dynamic Shapes:** list every formula whose axes, base Shape, count, crop, or index construction
  depends on later binding. State whether current attributes can carry the exact obligation and
  whether a general public primitive is needed.

### Decision gate for follow-up rows

After every matrix row is complete:

1. merge gaps only when one cohesive general public primitive closes them under one semantic
   contract;
2. select the smallest set of required public model primitives, with `foldAxis` restoration or
   redesign explicitly decided;
3. add one concise Draft row per selected primitive/capability immediately after 0023 and before
   0024 in the model master plan and roadmap;
4. make each row depend on 0023 and order the rows by semantic dependency;
5. make 0024 depend on 0023 and every required inserted model follow-up;
6. leave optional optimizations and compiler implementation work outside the model queue; and
7. create no detailed follow-up task file.

If no public primitive is needed for a candidate, record that no-change conclusion rather than
adding a placeholder row. A compiler-only kind may receive a Draft row only with the required
impossibility proof and exact backend-neutral semantic gap in the matrix.

## Acceptance criteria

- The audit artifact states its planning-only authority and defines its notation and
  classifications before the matrix.
- Source, public API, operation-kind/signature, producer-output, and completed-task inventories
  account for every current public semantic family and convenience producer chain in scope.
- Every selected differentiable input role has an exact family-level or operation-level adjoint
  formula and every required matrix field.
- Every non-differentiable role is explicit; no role is omitted merely because its result metadata
  cannot request gradients.
- Every formula restores the exact input Shape and type or identifies the precise unresolved
  policy/primitive gap. Static-only constructions are visibly distinguished from the full accepted
  dynamic-Shape contract.
- The required gather/scatter, slice/select, window, convolution, pooling, extrema, softmax,
  normalization, attention, loss, random-state, saved-output, and dynamic-Shape probes are each
  answered with repository evidence.
- The audit does not preselect or catalog operation-specific backward kinds. Formula length,
  performance, kernel availability, or fusion never appears as justification for model semantics.
- Every `MISSING_GENERAL_PUBLIC_PRIMITIVE` row identifies a reusable public transformation and
  rejects a narrower backward-only spelling. Every `GENUINELY_NON_EXPRESSIBLE_SEMANTIC_GAP` row
  contains the required concrete impossibility proof.
- The minimum evidence-backed Draft follow-up rows are inserted before preserved task 0024, in
  dependency order, without detailed specifications. No speculative row is added merely to cover
  a table category.
- `capabilities.md` contains only a concise linked result summary; the detailed formulas and matrix
  exist only in `adjoint-expressibility-audit.md`.
- The model master plan, roadmap, capabilities, task status, dependencies, frontier text, and
  completed history agree. Task 0024 remains Draft and has no detailed specification.
- A separate clean documentation-focused agent reviews and finalizes the five planning paths in
  the same overall change using the General and Planning profiles. It records glossary and API
  no-change reasons and does not rerun successful Java suites.
- Exactly the five permitted planning paths change. There are no Java, test, API, glossary,
  architecture, ADR, Gradle, module, conformance, or integration changes.
- Local Markdown links and heading anchors resolve, code fences are balanced, all changed files
  end with one newline, no trailing whitespace exists, and `git diff --check` passes.

## Tests / validation

This task changes planning documentation only. Do not run Java tests or Javadoc. Reuse the
recorded post-0022B capability-checkpoint evidence: root tests passed with 966 tests across 124
suites, model Javadoc passed, the public Tensor surface contained 188 methods, 657 local links and
176 anchors passed, and no architecture conflict was found. Record the originating checkpoint
context and confirm that this audit changed no executable or API/Javadoc source afterward.

Run final scope and status inspection:

```bash
git status --short
git diff --name-only
git ls-files --others --exclude-standard
rg -n "Task 0023|Task 0024|0023|0024" \
  docs/planning/modules/model/capabilities.md \
  docs/planning/modules/model/master-plan.md \
  docs/planning/roadmap.md \
  docs/planning/modules/model/tasks/0023-adjoint-expressibility-audit.md \
  docs/planning/modules/model/adjoint-expressibility-audit.md
find docs/planning/modules/model/tasks -maxdepth 1 -type f -name '0024*.md' -print
```

The final path inventory must contain exactly the five paths under [Affected files](#affected-files),
and the `find` command must print nothing. Inspect the matrix for all fixed classification tokens,
every required family, exact shared output indices, required decision probes, no backward-kind
catalog, and no implementation claim.

Run a read-only Markdown validation over the five changed files that checks relative file targets,
GitHub-style heading anchors, balanced code fences, and final newlines. Record the exact checker
command and the number of links and anchors checked. Then run:

```bash
git diff --check
```

Repository-wide validation is not repeated because the recorded post-0022B capability checkpoint
already closed the model-family frontier and this task changes no executable, dependency, build,
architecture, or public API contract.

## Dependencies

- Tasks 0005–0006 for backend-neutral semantic operation ownership.
- Tasks 0013, 0018K, and 0018L for exact producer/signature/indexed multi-output provenance.
- Tasks 0018M–0018M1 and current Shape contracts for symbolic and dynamic extent obligations.
- Task 0018N for exact typed scalar configuration.
- Completed operation-family tasks 0014A–0022B, including indexing cleanup 0018O, slice/window
  cleanup 0018R, modern-model tasks 0019–0020A1, normalization tasks 0021–0021C, and loss tasks
  0022–0022B.
- The completed post-0022B capability checkpoint with the evidence recorded above.

## Follow-up tasks

- Required: the minimum public model primitive tasks selected by the completed matrix. The audit
  inserts only concise Draft rows before 0024 and creates no detailed specifications.
- Required after those rows: task 0024, the preserved Draft model capability selection audit.
- Later compiler planning owns gradient rules, graph traversal/capture, saved-value lifetime,
  gradient accumulation, and backward graph construction after the required model primitives are
  available.
- Later backend/runtime work owns lowering, conformance, algorithms, prepared execution, and
  numerical execution.

## Architecture impact

Expected impact: None.

This audit applies existing model/compiler/backend ownership and does not change dependency rules.
If the audit discovers that an exact required meaning cannot fit those boundaries, stop and report
the conflicting rule and decision needed instead of changing architecture documentation.

## Implementation prompt

Use this prompt in a separate clean agentic task/thread:

```text
You are working in the Synaptik repository on planning/design documentation only.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/developer-guide/documentation-rules.md, the General and Planning documentation profiles,
and docs/planning/modules/model/tasks/0023-adjoint-expressibility-audit.md. Read the current
operation/source/test/API/task evidence required by that specification.

Execute task 0023 exactly as specified. Create the one planning-only audit artifact and update
only the five permitted planning paths. Do not implement Java/tests, create a backward-kind
catalog, create a detailed follow-up specification, edit architecture/API/glossary/Gradle, commit,
or push. Stop on an architecture conflict, missing exact semantic policy, or required scope
expansion rather than inventing behavior.

After the audit draft and planning validation, hand the exact diff and reused post-0022B
checkpoint evidence to a separate documentation-focused agent/thread with clean context. That
agent must independently finalize the five planning paths under the General and Planning profiles,
validate links/anchors/fences/newlines/whitespace, record API/glossary/Javadoc no-change reasons,
and must not repeat successful Java tests.

Finally update the task file with decisions, limitations, exact validation evidence, documentation
review, completion summary, and status. Do not mark Complete until the matrix, evidence-backed
follow-up queue, synchronization, and documentation pass are final.
```

## Local decisions

- Task 0023 is audit-only. The breadth and independent semantic gaps make implementation before
  the matrix unsafe and non-cohesive.
- The detailed result belongs in one sibling planning artifact rather than capabilities, an API
  guide, the glossary, or architecture documentation.
- Fixed row classifications make absence, non-differentiability, unselected policy, reusable
  primitive gaps, and genuine compiler-only gaps independently reviewable.
- Follow-up rows were selected only after the full matrix; the Ready specification did not
  preselect `foldAxis`, slice placement, convolution transpose/correlation, sum-to-Shape, or any
  compiler-only kind.
- Task 0024 keeps its established ID, Draft status, and lack of a detailed specification.
- Exact typed scalar zero/one leaves followed by `expand(targetShape)` are sufficient graph
  constants for unresolved Shapes. They replace the initial zero-like candidate without reading
  template values or contaminating NaN/infinity semantics.
- Current Scatter Elements ADD exactly expresses Gather Elements adjoints, and Scatter-ND ADD
  exactly expresses Gather-ND adjoints. Rank-changing Gather still needs a distinct generally
  useful axis scatter-add for an unresolved gathered extent because neither current scatter has
  its Shape relation and current one-hot requires positive static depth. Positive static extents
  already compose through one-hot selection and reduction; a statically zero valid domain returns
  the zero base.
- Ordinary binary arithmetic, `where`, and `EXPAND` expose only leading or statically known
  singleton expansion axes, so fixed reductions reverse their current accepted broadcasts.
  Binding-aware sum-to-Shape remains required for MATMUL and attention because those helpers
  explicitly defer unresolved singleton-or-equal batch obligations.
- Signed non-zero slice coordinates are injective. Draft 0023C needs general target-Shape
  placement/update and target-relative crop, but the audit does not require an additive overlap
  mode or a slice-backward kind.
- Maximum-pool routing does not need a new producer output. Negative-infinity padding, dynamic
  window materialization, existing first-index arg-max, numeric `where` over one-hot selection,
  and exact fold/crop compose it once the general window gap is closed.
- The matrix selects six Draft public-capability rows: binding-aware sum-to-Shape,
  Gather-compatible axis scatter-add, signed slice placement plus target-relative dynamic crop,
  public foldAxis plus dynamic/configurable 2D windows, cumulative product, and same-occurrence
  attention weights.
- Dynamic 2D windows cannot be obtained by only relaxing current helper validation: canonical
  rank-three columns flatten `outputHeight*outputWidth`, while current Dimension expressions do
  not multiply two unresolved extents. Draft 0023D must select either sufficient Shape algebra or
  a non-flattened dynamic window result/target contract.
- No row survives as `GENUINELY_NON_EXPRESSIBLE_SEMANTIC_GAP`; no compiler-only or
  operation-specific backward semantic row is added.

## Known limitations

- This task does not implement or test automatic differentiation. Its formulas are planning
  evidence for later compiler tasks.
- Rows may remain `POLICY_DEFERRED`; that is a valid audit result when current forward contracts do
  not select an observable derivative convention. Such rows must name the later decision owner.
- The audit identifies minimum model prerequisites but does not plan the later compiler traversal,
  accumulation, capture, optimization, or publication sequence.
- Regular-domain formulas do not choose derivatives at NaNs, infinities, singularities,
  discontinuities, extrema ties, scatter MUL/extrema edge cases, empty/all-masked domains, or
  ordering cutoffs. The matrix records each as `POLICY_DEFERRED` rather than inventing behavior.
- Concise Draft rows 0023A–0023F are planning labels only. They do not stabilize method names,
  attributes, overloads, result carriers, package placement, or detailed task scope.

## Validation evidence

- Reused the completed post-0022B capability checkpoint because this task changes planning text
  only: 966 root tests across 124 suites passed, model Javadoc passed, 188 public Tensor methods
  were inventoried, and the checkpoint documentation validation passed 657 links and 176 anchors.
  Java tests and Javadoc were not rerun.
- Inspected the current public Tensor methods, operation kinds, attributes, signatures, expression
  helpers, shared-producer output positions, completed task evidence, and dynamic Shape contracts.
  The matrix covers every required family and decision probe and selects no operation-specific
  backward kind or compiler-only semantic gap.
- Independent documentation context
  `/root/execute_0023_audit/docs_review_0023` finalized the five planning paths under the General
  and Planning profiles. It corrected broadcast reversal, static Gather composition,
  batch-normalization auxiliary use, injective signed-slice placement, product Shape restoration,
  and the unresolved-product constraint on dynamic two-dimensional windows.
- Exact five-path inventory passed. No `0024*.md` task specification exists. Tasks 0023A–0023F
  and 0024 remain Draft, and 0024 depends through 0023F.
- The independent read-only Markdown checker and the local bounded checker each passed 486 local
  links and one heading anchor, balanced fences, and single final newlines across the five paths.
  Required family, classification, producer-slot, decision-probe, and no-backward-kind searches
  passed. Trailing-whitespace search and `git diff --check` passed.
- Tensor, Compile, Runtime, and Training API references remain unchanged because no public or
  executable API changed. Javadoc remains accurate because no Java contract changed. No glossary
  update is needed because the audit defines local notation without stabilizing a public term.
  Architecture documents/tests, Java/tests, Gradle, other modules, backend conformance, and
  integration tests remain unchanged because ownership, dependencies, build structure, and
  executable behavior did not change.

## Implementation notes

- Created the planning-only [adjoint expressibility audit](../adjoint-expressibility-audit.md).
- Inventoried all current public Tensor methods, operation families/attributes/signatures,
  expression helpers/tests, completed task evidence, API boundaries, dynamic Shape contracts, and
  shared producer slots.
- Synchronized capabilities, the model master plan, and the roadmap with the six evidence-backed
  Draft rows before preserved task 0024. No follow-up task specification was created.
- No Java, tests, Javadoc, API, glossary, architecture, ADR, Gradle, other-module, conformance, or
  integration path was changed.

## Completion summary

- Completed changes: produced the planning-only adjoint expressibility matrix and selected the six
  minimum general public-capability follow-ups without adding a backward-specific or compiler-only
  semantic kind.
- Files changed or created: `adjoint-expressibility-audit.md`, this task specification,
  `capabilities.md`, `master-plan.md`, and the repository roadmap.
- Tests and validation: reused the post-0022B executable/Javadoc checkpoint and passed the exact
  scope, status, matrix-probe, Markdown, link, anchor, fence, newline, whitespace, and diff checks.
- Documentation review: the required independent clean-context documentation pass completed with
  no blocker and synchronized all five planning paths.
- Documentation, Javadoc, and glossary impact: planning documents only; the API references,
  Javadoc, glossary, architecture documents, and implementation documentation require no change.
- Unresolved issues: none for this audit.
- Required follow-up: execute Draft tasks 0023A–0023F in order before Draft task 0024; their
  detailed specifications are intentionally not part of this task.

Status: Complete

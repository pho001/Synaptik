# Task 0025I: NCW Max/Average Pool1d Composition

## Status

Complete

## Goal

Add public rank-specific maximum and fixed-count average pooling for NCW tensors as an exact,
visible composition through the existing Pool2d families. A successful call must create exactly:

```text
NCW input -> EXPAND_DIMS(axis 2) -> MAX_POOL2D or AVERAGE_POOL2D -> SQUEEZE(axis 2) -> NCW result
```

The inserted height is one and its kernel, stride, padding, and dilation are `(1, 1, 0, 1)`.
Pool2d therefore remains the sole semantic occurrence and the sole owner of current Compiler
gradients. No Pool1d operation kind, signature, producer, captured node, or backend capability is
introduced.

## Scope

- Add immutable public `MaxPool1dAttrs` and `AveragePool1dAttrs` values with width kernel, stride,
  symmetric padding, dilation, and literal `ceilMode` components.
- Add `Tensor.maxPool1d(MaxPool1dAttrs)` and
  `Tensor.averagePool1d(AveragePool1dAttrs)`.
- Add one package-private, final, field-free `TensorPool1dExpressions` composition owner.
- Accept floating NCW `[N, C, W]` input, prevalidate rank and exact static/symbolic width geometry,
  expand axis `2`, invoke the matching Pool2d expression, and squeeze axis `2`.
- Map maximum attributes to
  `MaxPool2dAttrs(1, kernelWidth, 1, strideWidth, 0, paddingWidth, 1, dilationWidth, ceilMode)`;
  map average attributes identically through `AveragePool2dAttrs`.
- Preserve canonical wrappers, fresh producer occurrences, ordered provenance, unresolved layout,
  and existing Model failure ordering.
- Finalize all affected Javadocs and explanatory documentation in the required separate clean
  documentation-focused pass before marking this task Complete.

## Out of scope

- Any `MAX_POOL1D`, `AVERAGE_POOL1D`, `Pool1dKind`, Pool1d operation signature, Compiler inventory
  entry, first-class backend capability, or direct Pool1d semantic producer.
- Pool3d semantics, `unfold3d`/`fold3d`, Pool3d gradients, CPU execution, lowering, generated code,
  fusion, materialization, Runtime behavior, Engine behavior, or NN layers.
- A public/private `PoolNd`, dynamic-rank or geometry-array API, asymmetric intrinsic padding,
  valid-sample average divisors, saved maximum indices, or another layout such as NWC.
- Changes to current Pool2d, rank-editing, Shape/Dimension, gradient, or backend contracts.
- Detailed specifications for Model 0025J/0025K, Compiler 0006B1/0006B2, CPU 0008G1, or 0026.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Model master plan](../master-plan.md)
- [NCW Conv1d composition precedent](0025g-ncw-conv1d-composition.md)
- [NCHW Max Pool2d semantics](0020a-nchw-max-pool2d-semantics-and-tensor-expression.md)
- [NCHW Average Pool2d semantics](0020a1-nchw-average-pool2d-semantics-and-tensor-expression.md)
- [Compiler pooling gradients](../../compiler/tasks/0005d-attention-convolution-pooling-and-loss-gradient-completion.md)

## Architecture constraints

- Work stays in Model and directly affected documentation/planning. Model owns immutable Tensor
  expressions; it does not capture graphs, select execution, or depend on downstream modules.
- `Tensor` remains the fluent facade and delegates to the package-private construction owner.
- The successful graph is exactly one `EXPAND_DIMS`, one existing Pool2d occurrence, and one
  `SQUEEZE`; no implementation may conceal, tag, fuse, or replace that topology.
- Pool1d attribute records implement `OperationAttrs` only as public immutable parameter values;
  no `OperationKind` accepts them. The Pool2d producer retains a fresh mapped Pool2d attrs value.
- Compiler continues to own capture and derivatives. Planning reports only operation-kind support,
  so it must report the visible rank edit and Pool2d kinds individually and never advertise a
  Pool1d operation capability.
- No architecture, dependency, module boundary, lifecycle, build, or architecture-test rule changes.

## Package impact

- Add the two attribute records in `io.github.pho001.synaptik.model.operation.pooling`.
- Add the composition owner and two receiver methods in `io.github.pho001.synaptik.model.tensor`.
- Add no package and widen no existing factory or helper seam.

## Exact equivalence contract

For inserted height `H = 1`, `K_h = 1`, `S_h = 1`, `P_h = 0`, and `D_h = 1`, the effective
height kernel is one and the output-height numerator is zero. Both literal floor and literal ceil
geometry therefore produce `H_out = 1`; no height window begins in padding. Width output geometry
is byte-for-byte the existing Pool2d formula with the supplied width components, including a
literal terminal ceil window.

Maximum pooling excludes padding. Because height contributes exactly one eligible position,
Pool2d height-major/width-minor traversal degenerates to increasing width order. NaN dominance,
`+0.0 > -0.0`, first logical winner, and all-padding negative infinity are consequently identical
to the one-dimensional contract.

Average pooling uses the fixed logical divisor `1 * kernelWidth`, exactly `kernelWidth`. Its
height-major accumulation order also degenerates to increasing width. BFLOAT16 and FLOAT32
accumulate and divide in FLOAT32, FLOAT64 does so in FLOAT64, and BFLOAT16 narrows only once at the
final output. NaN, infinities, signed zero, and all-padding positive zero remain unchanged.

Autograd differentiates through `SQUEEZE -> Pool2d -> EXPAND_DIMS`. Maximum therefore inherits the
same reconstructed first eligible logical winner, including NaN/signed-zero occurrence matching;
average inherits the same fixed divisor and overlap fold. No Pool1d gradient rule is needed.

## Affected files

Production and tests may change only as needed in these areas:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/pooling/MaxPool1dAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/pooling/AveragePool1dAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorPool1dExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- focused attribute/expression tests under the matching Model test packages;
- existing exact public-method-count tests affected by the two new Tensor methods;
- directly affected Tensor API, glossary, Model capabilities, this task, master plan, and roadmap.

Compile API and Training API are review-only unless their current Pool2d/composition wording becomes
false. Architecture documents, architecture tests, Compiler production/tests, CPU production/tests,
conformance, integration, Gradle, and other modules are review-only and should remain unchanged.

## Maximum scope

- Four production files: two new attributes, one new composition owner, and `Tensor`.
- Two focused new test owners plus only existing public-method-count owners made stale by the two
  receiver methods.
- Directly affected Javadocs and documentation/planning only.
- Stop and request clarification if completion needs a new operation kind, downstream production
  change, Shape/Dimension change, or broader public API.

## Acceptance criteria

- Both APIs accept valid floating NCW input and return exact NCW output Shape for static and
  supported symbolic width geometry.
- Intrinsic attrs validation covers positive kernel/stride/dilation and non-negative padding;
  expression validation covers nulls, floating type, rank three, checked geometry, and impossible
  output, with no partial producer construction on failure.
- Provenance proves exactly three producers in the required order and an existing Pool2d kind with
  exact mapped attributes; no Pool1d kind or signature exists.
- Tests lock floor/ceil width geometry, terminal all-padding windows, excluded maximum padding,
  fixed-count average padding, NaN, infinities, signed zero, first-winner order, BFLOAT16 final
  rounding, canonical wrappers, and freshness. Source-backed Compiler review confirms gradients
  are inherited through composition without a new derivative rule.
- Capability documentation states that Pool1d is visible composition and that backend support is
  the conjunction of its actual component kinds, never a synthetic Pool1d claim.
- Javadocs document every parameter, return value, constraint, failure mode, and composition effect.
- Only 0025I has a detailed specification; 0025J/0025K, Compiler 0006B1/0006B2, and CPU 0008G1
  remain Draft rows without task files.

## Tests / validation

- Run focused new attribute and expression tests plus all changed method-count tests.
- Run `./gradlew :modules:model:test` and `./gradlew :modules:model:javadoc` once after implementation.
- Review current Compiler pooling/autograd source and existing tests to prove inherited composition
  behavior; run focused Compiler tests only if concrete implementation evidence requires it, and do
  not change Compiler sources or repeat an already successful suite in the documentation pass.
- Inspect public declarations and producer topology with `javap`, reflection, and source-backed
  operation-signature checks.
- Validate Markdown links, anchors, fences, task status/order/dependencies, exact changed paths,
  and `git diff --check`.
- Repository-wide tests are not required unless implementation changes shared dependencies,
  architecture boundaries, build configuration, or multiple modules.

## Dependencies

- Complete Model 0020A and 0020A1 provide the exact Pool2d contracts.
- Complete Model 0017F1 provides visible singleton rank editing.
- Complete Compiler 0005D provides Pool2d gradient formulas consumed unchanged through composition.

## Follow-up tasks

- Model 0025J: first-class NCDHW maximum/average Pool3d semantics.
- Model 0025K: public NCDHW `unfold3d`/`fold3d` window algebra.
- Compiler 0006B1: Pool3d forward adoption and explicit gradient boundary.
- Compiler 0006B2: Pool3d gradient closure.
- CPU 0008G1: Pool1d composition validation and direct Pool3d generated execution.
- CPU 0008H remains after this complete inserted chain.

## Architecture impact

None. This is a new rank-specific public composition entirely within existing Model ownership and
operation semantics. `ARCHITECTURE.md`, the current architecture plan, ADRs, dependency rules, and
architecture tests remain unchanged.

## Implementation prompt

Implement Model task 0025I exactly within this specification. Work in a separate clean
implementation context; read repository instructions and the named contracts in full. Add only
the two rank-specific attrs values, the package-private composition owner, the two Tensor methods,
focused tests, and affected Javadocs. Preserve the exact three-producer topology and all equivalence
rules above. Do not add a Pool1d kind or downstream behavior. Then use a distinct clean
documentation-focused context to finalize affected documentation and no-change conclusions. Run
the stated validation once, report exact paths/evidence, and mark Complete only if every criterion
passes. Do not commit or push unless a later user request explicitly authorizes it.

## Local decisions

- Pool1d is composition, not first-class. Unlike Pool3d, it has an exact bounded representation:
  one singleton-axis insertion, one existing Pool2d occurrence, and one squeeze, independent of W.
- Separate max/average Pool1d attrs keep public rank-specific names without implying semantic kinds.
- Literal ceil mode is preserved, not normalized to framework-style terminal-window suppression.
- Intrinsic validation occurs in attrs; input-aware geometry validation occurs before any producer.
- Planning and backends may recognize the exact topology for optimization, but must retain truthful
  component-kind capability reporting and semantic equivalence.
- Documentation uses the general, API/Javadoc, planning, and example profiles. The glossary adds
  the reusable Pool1d-composition distinction; architecture documents remain unchanged because
  the exact composition introduces no architecture rule or boundary.

## Known limitations

- Only floating NCW input and symmetric intrinsic width padding are included.
- Average pooling is fixed count-padding only; valid-sample division is absent.
- No saved maximum indices, asymmetric padding, NWC, adaptive/global pooling, or PoolNd exists.
- Composition remains visible; performance specialization is deferred to CPU 0008G1.

## Validation evidence

Implementation-owned validation completed before this documentation pass:

- `./gradlew :modules:model:test` — passed; retained Gradle XML reports contain 1,082 tests across
  135 suites, with zero skipped, failures, or errors. Executable Java did not change afterward.
- `./gradlew :modules:model:javadoc` — passed; the generated Javadoc index was retained at
  `modules/model/build/docs/javadoc/index.html`. This pass changed no Java or Javadoc source, so it
  did not repeat the successful task.

Clean-context documentation-focused pass:

- Reviewed `AGENTS.md`, `ARCHITECTURE.md`, the current architecture plan, documentation rules and
  General/API-Javadoc/Planning/Example profiles, planning guide, roadmap, Model master plan, this
  task, the Conv1d and Pool2d precedents, final Pool1d source/Javadocs/tests, Tensor/Compile/Training
  APIs, Model capabilities, glossary, and related Compiler/CPU capability and gradient source.
- Finalized the Tensor API, Model capabilities, glossary, task evidence, Model master-plan status,
  and roadmap frontier. Public and package-private Pool1d Javadocs were source/test-accurate and
  required no further edit.
- `javac -cp modules/model/build/classes/java/main -d /tmp /tmp/Pool1dMetadataExample.java` and
  `java -cp modules/model/build/classes/java/main:/tmp Pool1dMetadataExample` — passed; the API
  example printed both `Shape[2, 3, 5]` results and the exact
  `EXPAND_DIMS -> MAX_POOL2D -> SQUEEZE` provenance.
- `javap -classpath modules/model/build/classes/java/main ...MaxPool1dAttrs
  ...AveragePool1dAttrs ...Tensor` with a focused declaration filter — passed; found exactly the
  two five-component public records and the two intended Tensor receiver methods, with no
  synthetic kind declaration.
- `ruby /tmp/check_synaptik_docs.rb` over the eight changed Markdown/planning files — passed;
  local file targets, heading anchors, and fenced blocks all validated.
- `git diff --check` — passed with no whitespace errors.
- Focused `rg` absence scan across Model production, Compiler, CPU, Planning, Prepare, testing,
  Compile/Training APIs, architecture docs, and `ARCHITECTURE.md` — passed; no `MAX_POOL1D`,
  `AVERAGE_POOL1D`, or `Pool1dKind` exists.
- File/status checks — passed: the worktree contains exactly 30 changed or new paths, all inside
  the authorized implementation, method-count-test, documentation, and planning scope; no task
  file exists for Model 0025J/0025K, Compiler 0006B1/0006B2, or CPU 0008G1. Task, Model master
  plan, and roadmap consistently mark 0025I Complete and retain 0025J as the next Draft frontier.
- Compile API: no change. It documents actual captured kinds and Compiler-owned first-order
  formulas; Pool1d adds neither, and source inspection confirms the visible rank-edit/Pool2d nodes
  already use its current inference and gradient rows.
- Training API: no change. The task adds no layer, module, parameter, training workflow, gradient
  publication, optimizer, or Tensor gradient state.
- Architecture docs/ADRs/tests: no change because module ownership, dependencies, lifecycle, and
  architecture rules are unchanged. Compiler and CPU production/tests, conformance/integration,
  Gradle, and other modules likewise need no change: Pool1d introduces no downstream kind or
  signature, backend support remains component-wise, and later topology recognition remains Draft
  CPU 0008G1 work.

## Implementation notes

- Added immutable width-only maximum and average attributes, exact prevalidation, two Tensor
  receivers, and a field-free composition owner.
- Every successful call creates one axis-2 expansion, one matching existing Pool2d occurrence,
  and one axis-2 squeeze with fresh canonical wrappers and mapped singleton-height attributes.
- Existing rank-edit and Pool2d Compiler rules supply inference and gradients; no downstream
  operation family, signature, capability, production behavior, or test was added.

## Completion summary

- Completed changes: added the public NCW maximum/fixed-count-average Pool1d composition and
  finalized its Javadocs, API explanation, capability boundary, glossary term, and planning state.
- Files changed or created: four production/Javadoc files, two focused new test owners, existing
  Tensor public-method-count tests, Tensor API, Model capabilities, glossary, this task, Model
  master plan, roadmap, and the already-planned Compiler/CPU master-plan rows.
- Tests and validation: reused the successful 1,082-test Model run and successful Model Javadoc
  generation; documentation-specific validation passed as recorded above.
- Documentation-agent review: completed in this mandatory separate clean context using the
  General, API/Javadoc, Planning, and Example profiles.
- Documentation impact: Tensor API now gives the exact composition, geometry, validation,
  exceptional-value/rounding inheritance, provenance/freshness, gradients, and capability boundary.
- Javadoc review: all affected public parameters, results, validation failures, metadata effects,
  and composition semantics are complete and source-backed; no additional edit was required.
- Glossary impact: added `Pool1d composition` because it is a reusable distinction from a
  first-class Pool1d operation and from synthetic backend capability.
- Unresolved issues: None.
- Follow-up required: None. Model 0025J remains the next Draft frontier and has no detailed
  specification.

Status: Complete

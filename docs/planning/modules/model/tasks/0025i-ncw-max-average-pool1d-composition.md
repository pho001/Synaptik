# Task 0025I: [NCW](../../../../glossary.md#term-ncw) Max/Average [Pool1d Composition](../../../../glossary.md#pool1d-composition)

## Status

Complete

## Goal

Add public [rank](../../../../glossary.md#term-rank)-specific maximum and fixed-count average [pooling](../../../../glossary.md#term-pooling) for [NCW](../../../../glossary.md#term-ncw) [tensors](../../../../glossary.md#tensor) as an exact,
visible composition through the existing Pool2d families. A successful call must create exactly:

```text
NCW input -> EXPAND_DIMS(axis 2) -> MAX_POOL2D or AVERAGE_POOL2D -> SQUEEZE(axis 2) -> NCW result
```

The inserted height is one and its [kernel](../../../../glossary.md#kernel), [stride](../../../../glossary.md#term-stride), [padding](../../../../glossary.md#padding), and [dilation](../../../../glossary.md#term-dilation) are `(1, 1, 0, 1)`.
Pool2d therefore remains the sole semantic [occurrence](../../../../glossary.md#term-occurrence) and the sole [owner](../../../../glossary.md#term-owner) of current [Compiler](../../../../glossary.md#term-compiler)
[gradients](../../../../glossary.md#term-gradient). No Pool1d [operation](../../../../glossary.md#operation) kind, [signature](../../../../glossary.md#term-signature), [producer](../../../../glossary.md#term-producer), captured [node](../../../../glossary.md#node), or [backend capability](../../../../glossary.md#backend-capability) is
introduced.

## [Scope](../../../../glossary.md#term-scope)

- Add [immutable](../../../../glossary.md#term-immutable) public `MaxPool1dAttrs` and `AveragePool1dAttrs` values with width kernel, stride,
  symmetric padding, dilation, and literal <span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [stride glossary](../../../../glossary.md#term-stride); [padding glossary](../../../../glossary.md#padding); [dilation glossary](../../../../glossary.md#term-dilation))</span>`ceilMode` components.
- Add `Tensor.maxPool1d(MaxPool1dAttrs)` and
  `Tensor.averagePool1d(AveragePool1dAttrs)`.
- Add one package-private, final, field-free `TensorPool1dExpressions` composition [owner](../../../../glossary.md#term-owner).
- Accept floating [NCW](../../../../glossary.md#term-ncw) `[N, C, W]` input, prevalidate rank and exact static/symbolic width geometry,
  expand axis <span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank))</span>`2`, invoke the matching Pool2d expression, and squeeze [axis](../../../../glossary.md#term-axis) `2`.
- Map maximum attributes to
  `MaxPool2dAttrs(1, kernelWidth, 1, strideWidth, 0, paddingWidth, 1, dilationWidth, ceilMode)`;
  map average attributes identically through `AveragePool2dAttrs`.
- Preserve canonical wrappers, fresh producer occurrences, ordered provenance, unresolved layout,
  and existing Model failure ordering.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [layout glossary](../../../../glossary.md#layout); [Model glossary](../../../../glossary.md#term-model))</span>
- Finalize all affected Javadocs and explanatory documentation in the required separate clean
  documentation-focused pass before marking this task Complete.

## Out of [scope](../../../../glossary.md#term-scope)

- Any `MAX_POOL1D`, `AVERAGE_POOL1D`, `Pool1dKind`, Pool1d operation signature, Compiler inventory
  entry, first-class backend capability, or direct Pool1d semantic producer.<span data-glossary-navigation="true"> ([operation signature glossary](../../../../glossary.md#operation-signature); [backend capability glossary](../../../../glossary.md#backend-capability); [producer glossary](../../../../glossary.md#term-producer))</span>
- [Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling) semantics, `unfold3d`/`fold3d`, Pool3d gradients, CPU execution, lowering, generated code,
  fusion, materialization, Runtime behavior, Engine behavior, or NN layers.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [fusion glossary](../../../../glossary.md#term-fusion); [materialization glossary](../../../../glossary.md#materialization); [Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- A public/private `PoolNd`, dynamic-rank or geometry-array API, asymmetric intrinsic padding,
  valid-sample average divisors, saved maximum indices, or another layout such as NWC.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [padding glossary](../../../../glossary.md#padding); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [layout glossary](../../../../glossary.md#layout))</span>
- Changes to current Pool2d, [rank](../../../../glossary.md#term-rank)-editing, [Shape](../../../../glossary.md#shape)/[Dimension](../../../../glossary.md#dimension), [gradient](../../../../glossary.md#term-gradient), or [backend](../../../../glossary.md#backend) contracts.
- Detailed specifications for [Model](../../../../glossary.md#term-model) 0025J/0025K, [Compiler](../../../../glossary.md#term-compiler) 0006B1/0006B2, [CPU](../../../../glossary.md#term-cpu) 0008G1, or 0026.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Roadmap](../../../roadmap.md)
- [Model master plan](../master-plan.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [NCW Conv1d composition precedent](0025g-ncw-conv1d-composition.md)<span data-glossary-navigation="true"> ([NCW glossary](../../../../glossary.md#term-ncw); [Conv1d composition glossary](../../../../glossary.md#conv1d-composition))</span>
- [NCHW Max Pool2d semantics](0020a-nchw-max-pool2d-semantics-and-tensor-expression.md)<span data-glossary-navigation="true"> ([NCHW glossary](../../../../glossary.md#term-nchw))</span>
- [NCHW Average Pool2d semantics](0020a1-nchw-average-pool2d-semantics-and-tensor-expression.md)
- [Compiler pooling gradients](../../compiler/tasks/0005d-attention-convolution-pooling-and-loss-gradient-completion.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [pooling glossary](../../../../glossary.md#term-pooling); [gradients glossary](../../../../glossary.md#term-gradient))</span>

## Architecture constraints

- Work stays in Model and directly affected documentation/planning. Model owns immutable Tensor
  expressions; it does not capture graphs, select execution, or depend on downstream modules.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [planning glossary](../../../../glossary.md#term-planning); [immutable glossary](../../../../glossary.md#term-immutable); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- [`Tensor`](../../../../glossary.md#tensor) remains the fluent [facade](../../../../glossary.md#term-facade) and delegates to the package-private construction [owner](../../../../glossary.md#term-owner).
- The successful [graph](../../../../glossary.md#graph) is exactly one `EXPAND_DIMS`, one existing Pool2d [occurrence](../../../../glossary.md#term-occurrence), and one
  `SQUEEZE`; no implementation may conceal, tag, fuse, or replace that [topology](../../../../glossary.md#term-topology).
- Pool1d attribute records implement [`OperationAttrs`](../../../../glossary.md#operationattrs) only as public immutable parameter values;
  no <span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter))</span>[`OperationKind`](../../../../glossary.md#operationkind) accepts them. The Pool2d [producer](../../../../glossary.md#term-producer) retains a fresh mapped Pool2d attrs value.
- Compiler continues to own capture and derivatives. Planning reports only operation-kind support,
  so it must report the visible rank edit and Pool2d kinds individually and never advertise a
  Pool1d operation capability.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [derivatives glossary](../../../../glossary.md#term-derivative); [operation glossary](../../../../glossary.md#operation); [rank glossary](../../../../glossary.md#term-rank))</span>
- No architecture, [dependency](../../../../glossary.md#term-dependency), [module](../../../../glossary.md#term-module) boundary, [lifecycle](../../../../glossary.md#lifecycle), build, or architecture-test rule changes.

## Package impact

- Add the two attribute records in `io.github.pho001.synaptik.model.operation.pooling`.
- Add the composition [owner](../../../../glossary.md#term-owner) and two receiver methods in `io.github.pho001.synaptik.model.tensor`.
- Add no package and widen no existing factory or helper seam.

## Exact equivalence contract

For inserted height `H = 1`, `K_h = 1`, `S_h = 1`, `P_h = 0`, and `D_h = 1`, the effective
height [kernel](../../../../glossary.md#kernel) is one and the output-height numerator is zero. Both literal floor and literal ceil
geometry therefore produce `H_out = 1`; no height window begins in [padding](../../../../glossary.md#padding). Width output geometry
is byte-for-byte the existing Pool2d formula with the supplied width components, including a
literal terminal ceil window.

Maximum [pooling](../../../../glossary.md#term-pooling) excludes padding. Because height contributes exactly one eligible position,
Pool2d height-major/width-minor traversal degenerates to increasing width order. [NaN](../../../../glossary.md#term-nan) dominance,
`+0.0 > -0.0`, first logical winner, and all-padding [negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero) are consequently identical
to the one-dimensional contract.

Average pooling uses the fixed logical divisor `1 * kernelWidth`, exactly `kernelWidth`. Its
height-major accumulation order also degenerates to increasing width. [BFLOAT16](../../../../glossary.md#term-bfloat16) and [FLOAT32](../../../../glossary.md#term-float32)
accumulate and divide in FLOAT32, [FLOAT64](../../../../glossary.md#term-float64) does so in FLOAT64, and BFLOAT16 narrows only once at the
final output. NaN, [infinities](../../../../glossary.md#nan-infinity-and-signed-zero), [signed zero](../../../../glossary.md#term-signed-zero), and all-padding [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) remain unchanged.

[Autograd](../../../../glossary.md#autograd) differentiates through `SQUEEZE -> Pool2d -> EXPAND_DIMS`. Maximum therefore inherits the
same reconstructed first eligible logical winner, including NaN/[signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) [occurrence](../../../../glossary.md#term-occurrence) matching;
average inherits the same fixed divisor and overlap fold. No Pool1d [gradient](../../../../glossary.md#term-gradient) rule is needed.

## Affected files

Production and tests may change only as needed in these areas:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/pooling/MaxPool1dAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/pooling/AveragePool1dAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorPool1dExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- focused attribute/expression tests under the matching [Model](../../../../glossary.md#term-model) test packages;
- existing exact public-method-count tests affected by the two new [Tensor](../../../../glossary.md#tensor) methods;
- directly affected Tensor [API](../../../../glossary.md#term-application-programming-interface-api), glossary, Model capabilities, this task, master plan, and roadmap.

[Compile](../../../../glossary.md#compile) API and Training API are review-only unless their current Pool2d/composition wording becomes
false. Architecture documents, architecture tests, [Compiler](../../../../glossary.md#term-compiler) production/tests, [CPU](../../../../glossary.md#term-cpu) production/tests,
[conformance](../../../../glossary.md#term-conformance), integration, [Gradle](../../../../glossary.md#term-gradle), and other [modules](../../../../glossary.md#term-module) are review-only and should remain unchanged.

## Maximum [scope](../../../../glossary.md#term-scope)

- Four production files: two new attributes, one new composition [owner](../../../../glossary.md#term-owner), and [`Tensor`](../../../../glossary.md#tensor).
- Two focused new test owners plus only existing public-method-count owners made stale by the two
  receiver methods.
- Directly affected Javadocs and documentation/[planning](../../../../glossary.md#term-planning) only.
- Stop and request clarification if completion needs a new operation kind, downstream production
  change, Shape/Dimension change, or broader public API.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Acceptance criteria

- Both APIs accept valid floating NCW input and return exact NCW output Shape for static and
  supported symbolic width geometry.<span data-glossary-navigation="true"> ([NCW glossary](../../../../glossary.md#term-ncw); [Shape glossary](../../../../glossary.md#shape))</span>
- Intrinsic attrs validation covers positive kernel/stride/dilation and non-negative padding;
  expression validation covers nulls, floating type, rank three, checked geometry, and impossible
  output, with no partial producer construction on failure.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [kernel glossary](../../../../glossary.md#kernel); [stride glossary](../../../../glossary.md#term-stride); [dilation glossary](../../../../glossary.md#term-dilation); [padding glossary](../../../../glossary.md#padding); [rank glossary](../../../../glossary.md#term-rank); [producer glossary](../../../../glossary.md#term-producer))</span>
- Provenance proves exactly three producers in the required order and an existing Pool2d kind with
  exact mapped attributes; no Pool1d kind or signature exists.<span data-glossary-navigation="true"> ([Provenance glossary](../../../../glossary.md#provenance); [signature glossary](../../../../glossary.md#term-signature))</span>
- Tests lock floor/ceil width geometry, terminal all-padding windows, excluded maximum padding,
  fixed-count average padding, NaN, infinities, signed zero, first-winner order, BFLOAT16 final
  rounding, canonical wrappers, and freshness. Source-backed Compiler review confirms gradients
  are inherited through composition without a new derivative rule.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [signed zero glossary](../../../../glossary.md#term-signed-zero); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [rounding glossary](../../../../glossary.md#term-rounding); [canonical glossary](../../../../glossary.md#term-canonical-form); [Compiler glossary](../../../../glossary.md#term-compiler); [gradients glossary](../../../../glossary.md#term-gradient); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- Capability documentation states that Pool1d is visible composition and that backend support is
  the conjunction of its actual component kinds, never a synthetic Pool1d claim.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- Javadocs document every [parameter](../../../../glossary.md#term-parameter), return value, constraint, failure mode, and composition effect.
- Only 0025I has a detailed specification; 0025J/0025K, Compiler 0006B1/0006B2, and CPU 0008G1
  remain Draft rows without task files.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

- [Run](../../../../glossary.md#term-run) focused new attribute and expression tests plus all changed method-count tests.
- Run `./gradlew :modules:model:test` and `./gradlew :modules:model:javadoc` once after implementation.
- Review current Compiler pooling/autograd source and existing tests to prove inherited composition
  behavior; run focused Compiler tests only if concrete implementation evidence requires it, and do
  not change Compiler sources or repeat an already successful suite in the documentation pass.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [pooling glossary](../../../../glossary.md#term-pooling); [autograd glossary](../../../../glossary.md#autograd))</span>
- Inspect public declarations and [producer](../../../../glossary.md#term-producer) [topology](../../../../glossary.md#term-topology) with `javap`, reflection, and source-backed
  operation-signature checks.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [signature glossary](../../../../glossary.md#term-signature))</span>
- Validate Markdown links, anchors, fences, task status/order/dependencies, exact changed paths,
  and `git diff --check`.
- Repository-wide tests are not required unless implementation changes shared dependencies,
  architecture boundaries, build configuration, or multiple modules.<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span>

## Dependencies

- Complete [Model](../../../../glossary.md#term-model) 0020A and 0020A1 provide the exact Pool2d contracts.
- Complete Model 0017F1 provides visible [singleton](../../../../glossary.md#term-singleton) [rank](../../../../glossary.md#term-rank) editing.
- Complete [Compiler](../../../../glossary.md#term-compiler) 0005D provides Pool2d [gradient](../../../../glossary.md#term-gradient) formulas consumed unchanged through composition.

## Follow-up tasks

- [Model](../../../../glossary.md#term-model) 0025J: first-class [NCDHW](../../../../glossary.md#conv3d--ncdhw) maximum/average [Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling) semantics.
- Model 0025K: public NCDHW `unfold3d`/`fold3d` window algebra.
- [Compiler](../../../../glossary.md#term-compiler) 0006B1: Pool3d forward adoption and explicit [gradient](../../../../glossary.md#term-gradient) boundary.
- Compiler 0006B2: Pool3d gradient closure.
- [CPU](../../../../glossary.md#term-cpu) 0008G1: [Pool1d composition](../../../../glossary.md#pool1d-composition) [validation](../../../../glossary.md#term-validation) and direct Pool3d generated execution.
- CPU 0008H remains after this complete inserted chain.

## Architecture impact

None. This is a new [rank](../../../../glossary.md#term-rank)-specific public composition entirely within existing [Model](../../../../glossary.md#term-model) [ownership](../../../../glossary.md#term-ownership-meanings) and
[operation](../../../../glossary.md#operation) semantics. `ARCHITECTURE.md`, the current architecture plan, ADRs, [dependency](../../../../glossary.md#term-dependency) rules, and
architecture tests remain unchanged.

## Implementation prompt

Implement [Model](../../../../glossary.md#term-model) task 0025I exactly within this specification. Work in a separate clean
implementation context; read repository instructions and the named contracts in full. Add only
the two [rank](../../../../glossary.md#term-rank)-specific attrs values, the package-private composition [owner](../../../../glossary.md#term-owner), the two [Tensor](../../../../glossary.md#tensor) methods,
focused tests, and affected Javadocs. Preserve the exact three-[producer](../../../../glossary.md#term-producer) [topology](../../../../glossary.md#term-topology) and all equivalence
rules above. Do not add a Pool1d kind or downstream behavior. Then use a distinct clean
documentation-focused context to finalize affected documentation and no-change conclusions. [Run](../../../../glossary.md#term-run)
the stated [validation](../../../../glossary.md#term-validation) once, report exact paths/evidence, and mark Complete only if every criterion
passes. Do not [commit](../../../../glossary.md#term-commit) or push unless a later user request explicitly authorizes it.

## Local decisions

- Pool1d is composition, not first-class. Unlike Pool3d, it has an exact bounded representation:
  one singleton-axis insertion, one existing Pool2d occurrence, and one squeeze, independent of W.<span data-glossary-navigation="true"> ([Pool3d glossary](../../../../glossary.md#pool3d--ncdhw-pooling); [singleton glossary](../../../../glossary.md#term-singleton); [axis glossary](../../../../glossary.md#term-axis); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Separate max/average Pool1d attrs keep public [rank](../../../../glossary.md#term-rank)-specific names without implying semantic kinds.
- Literal ceil mode is preserved, not normalized to framework-style terminal-window suppression.
- Intrinsic [validation](../../../../glossary.md#term-validation) occurs in attrs; input-aware geometry validation occurs before any [producer](../../../../glossary.md#term-producer).
- Planning and backends may recognize the exact topology for optimization, but must retain truthful
  component-kind capability reporting and semantic equivalence.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [backends glossary](../../../../glossary.md#backend); [topology glossary](../../../../glossary.md#term-topology))</span>
- Documentation uses the general, API/Javadoc, planning, and example profiles. The glossary adds
  the reusable Pool1d-composition distinction; architecture documents remain unchanged because
  the exact composition introduces no architecture rule or boundary.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## Known limitations

- Only floating [NCW](../../../../glossary.md#term-ncw) input and symmetric intrinsic width [padding](../../../../glossary.md#padding) are included.
- Average [pooling](../../../../glossary.md#term-pooling) is fixed count-padding only; valid-[sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) division is absent.
- No saved maximum indices, asymmetric padding, NWC, adaptive/global pooling, or PoolNd exists.
- Composition remains visible; performance [specialization](../../../../glossary.md#term-specialization) is deferred to [CPU](../../../../glossary.md#term-cpu) 0008G1.

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation-owned validation completed before this documentation pass:

- `./gradlew :modules:model:test` — passed; retained Gradle XML reports contain 1,082 tests across
  135 suites, with zero skipped, failures, or errors. Executable Java did not change afterward.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle))</span>
- `./gradlew :modules:model:javadoc` — passed; the generated [Javadoc](../../../../glossary.md#term-javadoc) index was retained at
  `modules/model/build/docs/javadoc/index.html`. This pass changed no Java or Javadoc source, so it
  did not repeat the successful task.

Clean-context documentation-focused pass:

- Reviewed `AGENTS.md`, `ARCHITECTURE.md`, the current architecture plan, documentation rules and
  General/API-Javadoc/Planning/Example profiles, planning guide, roadmap, Model master plan, this
  task, the Conv1d and Pool2d precedents, final Pool1d source/Javadocs/tests, Tensor/Compile/Training
  APIs, Model capabilities, glossary, and related Compiler/CPU capability and gradient source.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Planning glossary](../../../../glossary.md#term-planning); [Model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [Compiler glossary](../../../../glossary.md#term-compiler); [CPU glossary](../../../../glossary.md#term-cpu); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Finalized the Tensor API, Model capabilities, glossary, task evidence, Model master-plan status,
  and roadmap frontier. Public and package-private Pool1d Javadocs were source/test-accurate and
  required no further edit.
- `javac -cp modules/model/build/classes/java/main -d /tmp /tmp/Pool1dMetadataExample.java` and
  `java -cp modules/model/build/classes/java/main:/tmp Pool1dMetadataExample` — passed; the API
  example printed both `Shape[2, 3, 5]` results and the exact
  `EXPAND_DIMS -> MAX_POOL2D -> SQUEEZE` [provenance](../../../../glossary.md#provenance).
- `javap -classpath modules/model/build/classes/java/main ...MaxPool1dAttrs
  ...AveragePool1dAttrs ...Tensor` with a focused declaration filter — passed; found exactly the
  two five-component public records and the two intended Tensor receiver methods, with no
  synthetic kind declaration.
- `ruby /tmp/check_synaptik_docs.rb` over the eight changed Markdown/planning files — passed;
  local file targets, heading anchors, and fenced blocks all validated.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>
- `git diff --check` — passed with no whitespace errors.
- Focused `rg` absence scan across Model production, Compiler, CPU, Planning, Prepare, testing,
  Compile/Training APIs, architecture docs, and <span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [Prepare glossary](../../../../glossary.md#prepare))</span>`ARCHITECTURE.md` — passed; no `MAX_POOL1D`,
  `AVERAGE_POOL1D`, or `Pool1dKind` exists.
- File/status checks — passed: the worktree contains exactly 30 changed or new paths, all inside
  the authorized implementation, method-count-test, documentation, and planning scope; no task
  file exists for Model 0025J/0025K, Compiler 0006B1/0006B2, or CPU 0008G1. Task, Model master
  plan, and roadmap consistently mark 0025I Complete and retain 0025J as the next Draft frontier.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- Compile API: no change. It documents actual captured kinds and Compiler-owned first-order
  formulas; Pool1d adds neither, and source inspection confirms the visible rank-edit/Pool2d nodes
  already use its current inference and gradient rows.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [nodes glossary](../../../../glossary.md#node); [inference glossary](../../../../glossary.md#term-inference))</span>
- Training API: no change. The task adds no layer, module, parameter, training workflow, gradient
  publication, optimizer, or Tensor gradient state.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [parameter glossary](../../../../glossary.md#term-parameter); [publication glossary](../../../../glossary.md#term-publication); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- Architecture docs/ADRs/tests: no change because module ownership, dependencies, lifecycle, and
  architecture rules are unchanged. Compiler and CPU production/tests, conformance/integration,
  Gradle, and other modules likewise need no change: Pool1d introduces no downstream kind or
  signature, backend support remains component-wise, and later topology recognition remains Draft
  CPU 0008G1 work.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle); [conformance glossary](../../../../glossary.md#term-conformance); [signature glossary](../../../../glossary.md#term-signature); [backend glossary](../../../../glossary.md#backend); [topology glossary](../../../../glossary.md#term-topology))</span>

## Implementation notes

- Added immutable width-only maximum and average attributes, exact prevalidation, two Tensor
  receivers, and a field-free composition owner.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [Tensor glossary](../../../../glossary.md#tensor); [owner glossary](../../../../glossary.md#term-owner))</span>
- Every successful call creates one axis-2 expansion, one matching existing Pool2d occurrence,
  and one axis-2 squeeze with fresh canonical wrappers and mapped singleton-height attributes.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [occurrence glossary](../../../../glossary.md#term-occurrence); [canonical glossary](../../../../glossary.md#term-canonical-form); [singleton glossary](../../../../glossary.md#term-singleton))</span>
- Existing rank-edit and Pool2d Compiler rules supply inference and gradients; no downstream
  operation family, signature, capability, production behavior, or test was added.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [Compiler glossary](../../../../glossary.md#term-compiler); [inference glossary](../../../../glossary.md#term-inference); [gradients glossary](../../../../glossary.md#term-gradient); [operation glossary](../../../../glossary.md#operation); [signature glossary](../../../../glossary.md#term-signature))</span>

## Completion summary

- Completed changes: added the public NCW maximum/fixed-count-average Pool1d composition and
  finalized its Javadocs, API explanation, capability boundary, glossary term, and planning state.<span data-glossary-navigation="true"> ([NCW glossary](../../../../glossary.md#term-ncw); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: four production/Javadoc files, two focused new test owners, existing
  Tensor public-method-count tests, Tensor API, Model capabilities, glossary, this task, Model
  master plan, roadmap, and the already-planned Compiler/CPU master-plan rows.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Tensor glossary](../../../../glossary.md#tensor); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Tests and validation: reused the successful 1,082-test Model run and successful Model Javadoc
  generation; documentation-specific validation passed as recorded above.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [run glossary](../../../../glossary.md#term-run))</span>
- Documentation-agent review: completed in this mandatory separate clean context using the
  General, API/Javadoc, Planning, and Example profiles.
- Documentation impact: Tensor API now gives the exact composition, geometry, validation,
  exceptional-value/rounding inheritance, provenance/freshness, gradients, and capability boundary.<span data-glossary-navigation="true"> ([rounding glossary](../../../../glossary.md#term-rounding); [provenance glossary](../../../../glossary.md#provenance); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Javadoc review: all affected public parameters, results, validation failures, metadata effects,
  and composition semantics are complete and source-backed; no additional edit was required.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Glossary impact: added [`Pool1d composition`](../../../../glossary.md#pool1d-composition) because it is a reusable distinction from a
  first-class Pool1d operation and from synthetic backend capability.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [backend capability glossary](../../../../glossary.md#backend-capability))</span>
- Unresolved issues: None.
- Follow-up required: None. Model 0025J remains the next Draft frontier and has no detailed
  specification.

Status: Complete

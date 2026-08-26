# Task 0006B: Conv3d Forward Adoption and Explicit Gradient Boundary

## Status

Complete

## Goal

Adopt Model's first-class grouped NCDHW `CONV3D` operation in Compiler's ordinary flat
forward-graph pipeline without lowering, decomposition, execution, or gradient construction.

For each biased or unbiased occurrence, Compiler must preserve the exact operation kind,
`Conv3dAttrs`, ordered inputs, and one output; independently infer and verify the complete
promoted descriptor and NCDHW geometry from captured graph state; retain every unresolved
channel, group, bias, and spatial-fit relation as an ordered deferred graph constraint; and run
the same checks during initial and final validation.

`CompileMode.FORWARD_ONLY` must compile the occurrence through the existing capture,
optimization, diagnostics, publication, and Planning handoff. `FORWARD_AND_BACKWARD` and
`TRAINING_STEP` must reject any complete original forward inventory containing `CONV3D` during
allocation-free autograd preflight, before a seed, derivative constant, formula Tensor, Tensor ID,
or combined graph is created. Conv3d derivative expressibility and formula construction remain
owned only by Draft Compiler task 0006C.

The representation remains:

```text
one Conv3d TensorProducer occurrence
  -> one ordinary flat CompiledNode
     operation = CONV3D + exact Conv3dAttrs
     inputs    = [input, weight] or [input, weight, bias]
     output    = one promoted NCDHW ValueId
  -> existing diagnostics, publication, and OperationCapabilityQuery paths
```

## Scope

- Route `Conv3dKind.CONV3D` through the current structured-operation inference boundary.
- Independently validate its exact `Conv3dAttrs`, two-or-three-input cardinality, one-output
  cardinality, role types, ranks, static kernel extents, ordered floating promotion, output
  gradient eligibility, and exact output descriptor.
- Derive NCDHW result geometry from graph descriptors and attributes without trusting the stored
  Model output descriptor.
- Prove or retain, in deterministic order, exact obligations for:
  1. input-channel divisibility by `groups`;
  2. output-channel divisibility by `groups`;
  3. `weightChannelsPerGroup * groups == inputChannels`;
  4. optional bias length equal to output channels;
  5. non-negative padded depth numerator;
  6. non-negative padded height numerator; and
  7. non-negative padded width numerator.
- Preserve the exact input batch Dimension and weight output-channel Dimension in the inferred
  result and derive the three spatial Dimensions using the existing canonical Dimension
  expression vocabulary.
- Preserve one ordinary flat node. Do not lower or decompose it into window, matrix, fold,
  pointwise, loop, region, or backend operations.
- Preserve ordinary exact common-subexpression elimination (CSE) semantics: graph-output
  producers remain ineligible as today; equal eligible internal Conv3d occurrences may merge only
  when phase, derivative order, complete operation value including attributes, remapped inputs,
  and output descriptors are equal; distinct attributes or inputs must not merge. Conv3d has no
  recurrent-style identity exclusion.
- Preserve whole-node dead-code elimination and ordinary canonical reindexing.
- Carry surviving exact operation attributes and ordered descriptors through publication and the
  existing `OperationCapabilityQuery`; add no provider capability.
- Add an allocation-free, deterministic Conv3d guard to the complete-original-forward-inventory
  autograd preflight for both backward-capable modes.
- Keep `FirstOrderGradientCoverage.SIGNATURES`, its 128 supported rows, and its family-owner
  vocabulary unchanged. Update only the source-backed boundary test so the complete 132-signature
  Model inventory is partitioned into 128 supported signatures plus four deferred signatures:
  three recurrent signatures and one Conv3d signature.
- Update directly affected Compiler/Model explanatory documentation and planning status in the
  implementation change, with a separate clean documentation-focused pass.

## Out of scope

- Conv3d input, weight, or bias gradient formulas; `ConvolutionGradientRules` changes; a Conv3d
  gradient family owner; adding Conv3d to `FirstOrderGradientCoverage.SIGNATURES`; or claiming
  adjoint or higher-order expressibility
- any part of Draft Compiler 0006C, including its proof, Model prerequisite selection, detailed
  specification, saved-state policy, overlap-accumulation formula, or gradient closure
- changes to Model Conv3d semantics, `Conv3dAttrs`, `Conv3dKind`, Tensor methods, descriptors,
  provenance, factory behavior, numerical policy, or tests beyond focused unchanged-boundary
  validation
- a `CONV1D` kind or Compiler inventory row; NCW Conv1d remains the visible
  `EXPAND_DIMS -> CONV2D -> SQUEEZE` composition from Model 0025G and continues to use existing
  inference and gradient rows
- Conv3d lowering or decomposition, window extraction, matrix multiplication, fold, explicit
  loops, graph regions, nested graphs, backend-specific graphs, executable units, kernels,
  generated code, or numerical evaluation
- capability advertisement, owner-policy changes, concrete backend work, CPU 0008/0008A, general
  partition-DAG work, fusion, profitability, materialization, or performance claims
- changes to Planning, Prepare, Runtime, Engine, Config, Trace, backend-contract, concrete
  backends, extensions, public Compiler API, dependencies, Gradle, architecture rules, or build
  structure
- dynamic-rank, channels-last, asymmetric intrinsic padding, transposed, deformable,
  causal-specialized, quantized, sparse, depthwise-specific, separable, or arbitrary-rank
  convolution
- Compiler 0007 exact/relaxed algebra or any later detailed task specification
- architecture, backend-conformance, or integration tests unless implementation reveals an actual
  boundary change, in which case stop rather than expanding this task

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially Model and Compiler
  ownership, flat graph state, autograd preflight, compile lifecycle, and forbidden Compiler
  dependencies
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Training graph](../../../../architecture/training-graph.md)
- [ADR 0009: Compiler-owned pre-capture Tensor-expression autograd](../../../../design/decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Compiler master plan](../master-plan.md)
- [Compiler 0002: Captured-graph inference and validation](0002-captured-graph-inference-and-validation.md)
- [Compiler 0004: Compiler-owned pre-capture autograd](0004-compiler-owned-pre-capture-autograd-and-combined-graph-compilation.md)
- [Compiler 0005D: Attention, convolution, pooling, and loss gradients](0005d-attention-convolution-pooling-and-loss-gradient-completion.md)
- [Compiler 0005E: First-order gradient coverage closure](0005e-first-order-gradient-coverage-closure-checkpoint.md)
- [Compiler 0006: Explicit functional gradient requests](0006-explicit-functional-gradient-requests-and-higher-order-differentiation.md)
- [Compiler 0006A: Fixed recurrent-scan forward adoption](0006a-fixed-recurrent-scan-forward-adoption-and-bptt-boundary.md)
- [Model 0020: NCHW Conv2d semantics](../../model/tasks/0020-nchw-conv2d-semantics-and-tensor-expressions.md)
- [Model 0025G: NCW Conv1d composition](../../model/tasks/0025g-ncw-conv1d-composition.md)
- [Model 0025H: NCDHW Conv3d semantics](../../model/tasks/0025h-ncdhw-conv3d-semantics-and-tensor-expressions.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)
- [Planning documentation style](../../../../developer-guide/documentation/planning-style.md)

## Architecture constraints

- Model remains the owner of Conv3d meaning, immutable attributes, public Tensor construction,
  floating numerical policy, descriptor metadata, and canonical provenance.
- Compiler independently derives and validates captured semantics; it must not trust a stored
  output descriptor merely because Model construction produced it.
- One `CONV3D` occurrence remains one ordinary flat `CompiledNode`. Compiler adds no convolution
  IR, nested graph, region, decomposition, executable work, physical state, or backend payload.
- Compiler may retain unresolved logical Shape obligations only through the existing typed
  `DeferredGraphConstraint`/`GraphPredicate` boundary. It must not bind Dimensions or move
  validation into Runtime.
- Compiler-owned autograd must fail closed before derivative Tensor allocation when a complete
  forward inventory contains an unsupported forward-only occurrence.
- Forward inference support and first-order gradient coverage remain separate closed inventories.
  This task changes the former from 131 to 132 signatures and leaves the latter at 128.
- CSE remains phase- and derivative-order-local and uses exact operation value, ordered remapped
  inputs, and complete output descriptors. Conv3d is functionally pure and gains no identity-
  distinct exemption.
- Compiler has no dependency on Runtime, Prepare, Engine, or concrete backends and introduces no
  public facade or registry.
- If implementation requires a public API, another module, a new predicate vocabulary, an
  architecture change, or a Conv3d derivative formula, stop and report the conflict.

## Package impact

Existing package used and changed:

- `io.github.pho001.synaptik.compiler` — retains all implementation and tests in the existing
  package-private Compiler front-end boundary.

Packages added or changed:

- no package is added;
- no public package surface is widened.

Type placement:

- `io.github.pho001.synaptik.compiler.CapturedGraphInference` — adds only closed dispatch of
  `Conv3dKind` to the existing structured inference owner.
- `io.github.pho001.synaptik.compiler.StructuredOperationInference` — owns the rank-five Conv3d
  descriptor derivation and ordered constraint requests beside its existing Conv2d inference.
- `io.github.pho001.synaptik.compiler.AutogradPreflight` — owns the complete-forward-inventory
  forward-only boundary and deterministic pre-allocation rejection.
- `io.github.pho001.synaptik.compiler.FirstOrderGradientCoverage` — remains unchanged; only its
  source-backed package-local test records Conv3d as the fourth deferred signature.

Do not add a generic convolution helper, `ConvNd` abstraction, public diagnostic type, gradient
registry, or another package.

## Forward inference and validation contract

### Accepted operation, attributes, and cardinality

Accept only:

```text
kind:       Conv3dKind.CONV3D
attributes: Conv3dAttrs
inputs:     2 or 3
outputs:    exactly 1
```

Ordered inputs are `[input, weight]` or `[input, weight, bias]`. The exact attribute object and
ordered inputs captured in the `Operation`/`CompiledNode` remain unchanged. Any unknown kind,
wrong attribute type, invalid input/output count, or later unrecognized operation continues to
fail closed through the current structural or inference boundary.

### Descriptor derivation

Inference derives the result in this order:

1. Require floating `input`, then floating `weight`; if present, require floating `bias`.
2. Promote `input` with `weight`, then promote the intermediate type with present `bias`, using
   `DataTypePromotion.promoteFloating`.
3. Require input Shape `[N, C_in, D, H, W]`, weight Shape
   `[C_out, C_in/groups, K_d, K_h, K_w]`, and optional bias Shape `[C_out]` by exact ranks
   five, five, and one.
4. Require `K_d`, `K_h`, and `K_w` to be statically known and positive, in depth, height, width
   order.
5. Compute each effective kernel with checked signed-`long` arithmetic:

   ```text
   effectiveKernel_x = dilation_x * (K_x - 1) + 1
   offset_x          = 2 * padding_x - effectiveKernel_x
   numerator_x       = X + offset_x
   X_out             = floor(numerator_x / stride_x) + 1
   ```

6. Construct the result Shape
   `[N, C_out, D_out, H_out, W_out]`, retaining the exact input batch Dimension and weight output-
   channel Dimension references.
7. Derive unresolved layout and `requiresGrad` equal to the logical OR of input, weight, and
   present bias.

The inferred descriptor must exactly equal the stored output descriptor. Existing node-index,
`NodeId`, kind, output-index, `ValueId`, expected-descriptor, and stored-descriptor context remains
the final mismatch diagnostic.

### Ordered constraints and diagnostics

Inference emits candidate constraints in this exact order:

```text
conv3d input channels divisible by groups
conv3d output channels divisible by groups
conv3d weight channels per group
conv3d bias channels                         only when bias is present
conv3d depth numerator non-negative
conv3d height numerator non-negative
conv3d width numerator non-negative
```

Use existing predicates:

- `DimensionDivisible(C_in, groups)`;
- `DimensionDivisible(C_out, groups)`;
- `DimensionEqual(multiply(weightChannelsPerGroup, groups), C_in)`;
- `DimensionEqual(biasLength, C_out)` when biased; and
- `DimensionAtLeast(numerator_x, 0)` for each spatial axis.

Static contradictions fail during inference with existing full node context. Proven constraints
are discarded. Unresolved constraints become exact ordered `DeferredGraphConstraint` values and
public `CompileDiagnostics.DeferredConstraintDiagnostic` projections. The constraint subjects are
stable diagnostic text for this task; no new public diagnostic taxonomy or rejection aggregate
is added.

For a static spatial input, checked addition computes the numerator and negative values fail. For
an unresolved input, use the same canonical `DimensionExpressions.addConstant`, `floorDivide`,
and final `addConstant(..., 1)` sequence as Model. Checked arithmetic overflow is reported through
the current node-context descriptor-derivation failure path; it is never treated as a deferred
valid binding.

### Repeated validation and downstream handoff

The same `CapturedGraphInference.inferAndValidate` boundary must run before canonicalization and
after canonicalization or every changed optional optimization candidate, as it does for current
operations. Final deferred constraints and their order must reflect only the final validated
graph.

Existing publication and Planning orchestration then carry:

- the exact surviving `CONV3D` operation and attributes;
- exact ordered input descriptors;
- the exact one-output descriptor; and
- the final ordered deferred diagnostics.

A recording test provider may accept the query to prove the unchanged handoff. It must not be a
production capability claim, and no current provider is changed.

## Autograd and inventory boundary

For both backward-capable modes, `AutogradPreflight` builds the complete original forward producer
inventory without allocating a Tensor, then scans deterministic producer postorder before request
stage, seed, route, occurrence-policy, or formula validation. The first `CONV3D` occurrence fails
with stable context containing producer-postorder index, exact kind class/name, exact attributes
class, and this explanation:

```text
Conv3d is forward-only until Compiler task 0006C closes its gradients
```

The guard applies even when Conv3d is a separate forward root and the requested target lies on an
unrelated non-Conv3d branch. Rejection must leave the next Tensor ID unchanged and construct no
default or explicit seed normalization, typed splat, local formula, gradient accumulator,
matching auxiliary, second-stage expression, combined capture, or partial backward graph.

Do not add Conv3d to `FirstOrderGradientCoverage.SIGNATURES`, make it resolve to
`FamilyOwner.CONVOLUTION`, or add it to `ConvolutionGradientRules`. The existing Conv2d row and
formulas remain unchanged. The
source-backed test must prove:

```text
Model production inventory:           39 families, 111 constants, 132 signatures
Compiler supported gradient inventory: 37 families, 107 kinds, 128 signatures
Deferred exact signatures:             4 = 3 recurrent + 1 Conv3d
Supported intersection deferred:       empty
Supported union deferred:              exact complete Model inventory
```

Every Conv3d output/input boundary classification remains `FC` with exact existing reason
`unknown or unclassified operation kind/attributes pairing`, without Tensor-ID allocation.
Forward inference adoption must not be mistaken for gradient support.

## Affected files

Expected Compiler production source (3):

- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CapturedGraphInference.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/StructuredOperationInference.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradPreflight.java`

Expected Compiler tests (4):

- add `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/Conv3dCompilerTest.java`
- update `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/StructuredOperationInferenceTest.java`
- update `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/AutogradPreflightTest.java`
- update `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/FirstOrderGradientCoverageTest.java`

Expected documentation and planning (8):

- `docs/api/compile-api.md`
- `docs/api/tensor-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- `docs/planning/modules/model/master-plan.md`
- this task specification
- `docs/planning/modules/compiler/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless current evidence makes a statement inaccurate: `GraphCapture`,
`ForwardCommonSubexpressionElimination`, `CompileDiagnostics`, `FirstOrderGradientCoverage`,
`ConvolutionGradientRules`, their focused tests, Model Conv3d source/tests/Javadocs, the
authoritative architecture and focused architecture pages, Runtime and Training APIs, Planning,
Prepare, Runtime, Engine, CPU and other backend plans, architecture/conformance/integration tests,
Gradle, dependencies, and other modules.

## Maximum scope

Exactly fifteen paths maximum: three Compiler production files, four Compiler test files, and
eight documentation/planning files. No Model Java or test, public API, other module, build,
architecture, backend, conformance, or integration path may change.

If implementation needs another path, a new graph predicate, another module, a public contract,
or a gradient formula, stop and update this task through a separate planning decision before
continuing. Do not silently use the maximum as permission for unrelated cleanup.

## Javadoc and documentation requirements

- Review and finalize the class and method Javadocs of all three changed Compiler production
  files. Explain independent Conv3d inference, ordered deferred obligations, repeated final
  validation, and allocation-free backward rejection. Every changed or added method contract must
  have meaningful `@param`, `@return`, and applicable `@throws` documentation.
- Update the Compile API to make the 132-signature forward inventory current, retain the exact
  128-signature first-order inventory, explain the four-signature deferred set, and distinguish
  successful forward compilation from absent execution/capability and absent gradients.
- Update only the Conv3d downstream-status sentence in the Tensor API; Model semantics and public
  API remain unchanged.
- Update the glossary's Conv3d and inventory entries to distinguish current Compiler forward
  inference/final proof from Draft 0006C gradients and later execution.
- Update Model capabilities and the current Model master-plan status wording only where they call
  Compiler forward adoption absent or Conv3d planned. Do not change completed Model 0025H's
  historical task record.
- Synchronize this task, Compiler master plan, and roadmap to `Complete` only after implementation,
  validation, and the separate documentation pass. Keep 0006C and 0007 Draft without detailed
  specifications; keep Model 0025H Complete and Model 0026 Draft without a specification.
- Record reasoned no-change conclusions for Tensor public signatures/Javadocs, Runtime and
  Training APIs, Model capabilities beyond status, architecture/ADRs/tests, Planning/Prepare/
  Runtime/Engine, backend plans and capability providers, conformance/integration, Gradle,
  dependencies, and other modules.

## Acceptance criteria

- `FORWARD_ONLY` accepts biased and unbiased `CONV3D` as one ordinary flat captured node with
  exact operation/attribute/input/output preservation and no decomposition or execution work.
- Inference independently checks exact kind/attributes/cardinalities, role types/ranks, ordered
  floating promotion, positive static K-d/K-h/K-w, checked geometry, result descriptor, unresolved
  layout, and gradient-eligibility OR.
- Static valid geometry and zero batch/output-channel cases pass. Every static contradiction and
  overflow category fails with deterministic node context. Dynamic channel/group/bias and three
  spatial-fit obligations are proven, rejected, or retained exactly in the specified order.
- Final stored-descriptor mismatch diagnostics identify node position, `NodeId`, kind, output
  position, `ValueId`, expected descriptor, and stored descriptor.
- Ordinary CSE semantics remain unchanged: equal eligible internal Conv3d expressions may merge;
  differing attributes or inputs do not; graph-output producers remain unmerged. DCE,
  canonicalization, phase/order metadata, and unrelated operation behavior remain unchanged.
- Publication and a recording Planning provider receive the exact surviving operation attributes
  and ordered descriptors; no production provider advertises Conv3d.
- Both backward-capable modes reject the first Conv3d in complete producer postorder with the
  stable task-owned diagnostic before any derivative Tensor/ID/node allocation, including an
  unrelated-gradient-branch case.
- The source-backed inventories are exactly 39 families, 111 constants, and 132 Model signatures;
  128 supported gradient signatures plus the exact four deferred signatures partition that set;
  Conv3d remains fail-closed with the existing unknown/unclassified reason.
- Conv1d remains visible rank-edit/Conv2d/rank-edit composition and gains no kind, inference row,
  gradient row, or special Compiler treatment.
- No Conv3d formula, `FirstOrderGradientCoverage` production row, `FamilyOwner` change,
  `ConvolutionGradientRules` change, public API, shared predicate type, other module, dependency,
  Gradle, architecture, backend, conformance, or integration behavior changes.
- Focused tests, unchanged Model boundary tests, one final Compiler suite, Compiler Javadoc,
  documentation validation, exact fifteen-path scope, status/order/specification-absence checks,
  empty staging, and `git diff --check` pass.
- A separate clean documentation-focused context independently finalizes all affected Javadocs,
  explanatory docs, glossary impact, planning status/evidence, and reasoned no-change conclusions
  in the same overall change.

## Tests / validation

During implementation, run the focused Compiler matrix:

```bash
./gradlew :modules:compiler:test \
  --tests io.github.pho001.synaptik.compiler.Conv3dCompilerTest \
  --tests io.github.pho001.synaptik.compiler.StructuredOperationInferenceTest \
  --tests io.github.pho001.synaptik.compiler.AutogradPreflightTest \
  --tests io.github.pho001.synaptik.compiler.FirstOrderGradientCoverageTest \
  --tests io.github.pho001.synaptik.compiler.ForwardCommonSubexpressionEliminationTest \
  --tests io.github.pho001.synaptik.compiler.CompileDiagnosticsTest
```

Run the committed Model boundary once without changing Model:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.operation.convolution.Conv3dSemanticsTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorConv3dExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorConv2dExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorConv1dExpressionTest
```

After executable Compiler code stabilizes, run exactly one final affected-module suite:

```bash
./gradlew :modules:compiler:test
```

Record Gradle outcomes and JUnit XML suite/test/skip/failure/error counts. Do not run the full
Model suite because Model executable behavior does not change. Repository-wide, architecture,
backend-conformance, and integration validation is deferred to continuous integration: this task
changes one executable module, no dependency or architecture rule, no shared build configuration,
and no backend or end-to-end path.

The implementation context must hand the actual diff and exact executable evidence to a separate
clean documentation-focused context. That context must not repeat successful Java tests unless it
changes executable behavior or records a concrete stale-evidence risk.

Documentation pass:

```bash
./gradlew :modules:compiler:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git diff --cached --name-only
git status --short
```

The documentation pass must also inspect the generated/public Compiler Javadocs and the source
Javadocs of package-private changed types; validate local Markdown links and anchors, unique
headings, balanced fences, final newlines, and trailing whitespace; verify exact operation,
constraint, diagnostic, inventory, and status wording; confirm the exact fifteen authorized
paths; confirm no Conv3d gradient or provider-capability claim; confirm 0006C and 0007 are Draft
without detailed task files; confirm Model 0025H remains Complete and Model 0026 remains Draft
without a specification; preserve roadmap order Model 0025H -> Compiler 0006B -> CPU 0008 -> CPU
0008A -> CPU 0008B–0008E; and confirm an empty Git index.

## Dependencies

Hard prerequisites, all Complete:

- Model 0018K–0018N and 0025 provide operation-signature, symbolic-Dimension, descriptor, and
  canonical-producer foundations.
- [Model 0020](../../model/tasks/0020-nchw-conv2d-semantics-and-tensor-expressions.md) provides the
  rank-two grouped-convolution semantic and inference oracle.
- [Model 0025G](../../model/tasks/0025g-ncw-conv1d-composition.md) fixes Conv1d as visible ordinary
  composition with no new operation kind.
- [Model 0025H](../../model/tasks/0025h-ncdhw-conv3d-semantics-and-tensor-expressions.md) provides
  exact first-class Conv3d semantics, attributes, signatures, static/symbolic geometry, numerical
  policy, and canonical provenance.
- Compiler 0001–0003B provide identity capture, inference/final-validation orchestration,
  canonicalization, exact optimization, DCE/CSE, and constant sidecars.
- Compiler 0004–0005E provide allocation-free autograd preflight, combined capture, publication/
  Planning orchestration, and the source-backed 128-signature first-order closure.
- [Compiler 0006](0006-explicit-functional-gradient-requests-and-higher-order-differentiation.md)
  provides current one/two-stage functional requests and backward-capable modes.
- [Compiler 0006A](0006a-fixed-recurrent-scan-forward-adoption-and-bptt-boundary.md) provides the
  complete-original-forward-inventory forward-only guard precedent and the intentional split
  between forward inference and deferred gradient inventories.

Existing Compiler dependencies on Model, Config, Planning, Backend Contract, and Trace are
sufficient. No new dependency or architecture decision is required.

## Follow-up tasks

- Draft Compiler 0006C remains the sole owner of Conv3d adjoint-expressibility proof and gradient
  closure. It must prove group isolation, dilation/padding, overlap accumulation, symbolic Shape,
  and higher-order formula closure through current public Tensor algebra, or select the smallest
  separate Model prerequisite. It is not implemented or specified here.
- Draft CPU 0008 remains the portable Conv2d execution foundation after 0006B. Draft CPU 0008A
  then validates visible Conv1d composition and adds direct grouped Conv3d forward execution.
  Compiler 0006C does not block those forward CPU tasks.
- CPU 0008B–0008E remain the later ordered general decomposition, recognition, profitability, and
  bounded multi-input materialization program. No need for those facilities is inferred here.
- Engine 0004 and NN 0025/0025A remain later public lifecycle and layer consumers. Model 0026
  remains the separate Draft FLOAT16 semantic foundation.
- Compiler 0007 remains the separate Draft exact/permission-aware algebra owner.

Do not create any follow-up detailed specification in this task.

## Architecture impact

Expected impact: None.

This task implements the existing Compiler ownership and pre-capture fail-closed contract with
the current flat graph, typed constraints, exact optimization, publication, diagnostics, and
Planning handoff. It changes no module ownership, dependency direction, public API, architecture
rule, or execution contract. If implementation reveals a need for any such change, stop and
report the exact conflict.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository. Do not commit, stage, or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the Compiler master plan, and
docs/planning/modules/compiler/tasks/0006b-conv3d-forward-adoption-and-explicit-gradient-boundary.md.
Read the task's directly referenced Model Conv3d/Conv2d/Conv1d, Compiler capture/inference/
constraint/validation/autograd/coverage/CSE/diagnostic/publication/Planning source, tests, and
documentation contracts.

Implement Compiler 0006B exactly within its fifteen authorized paths. Adopt Conv3d only in the
ordinary flat forward pipeline, independently derive its complete descriptor and ordered deferred
obligations, and reject both backward-capable modes before derivative Tensor allocation. Preserve
the 128 supported gradient rows plus exact four-signature deferred boundary. Do not implement
0006C, Conv3d formulas, decomposition, execution, capability advertisement, public API, another
module, dependencies, Gradle, architecture, conformance, integration, or later specifications.
Stop on any architecture, inventory, or scope conflict.

After executable implementation and the recorded focused, Model-boundary, and final Compiler
validation, hand the actual diff and exact evidence to a separate documentation-focused agent or
thread with clean context. That pass must follow docs/developer-guide/documentation-rules.md and
finalize affected Javadocs, APIs, glossary/capability impact, planning status/evidence, and
documentation validation in the same overall change without repeating successful Java tests
unless executable behavior changes or a concrete stale-evidence risk is recorded.

Update this task with local decisions, exact validation evidence, implementation notes,
completion summary, and final status. Do not mark it Complete before every acceptance criterion
and the documentation pass finish.
```

## Local decisions

- Conv3d uses the existing ordinary `CompiledNode`, structured inference, typed constraint,
  diagnostics, optimization, publication, and Planning boundaries. No new IR or public type is
  justified.
- Rank-five Conv3d inference remains beside Conv2d in `StructuredOperationInference`; no generic
  ConvNd helper or new package is introduced.
- Spatial-fit validity is an explicit `DimensionAtLeast(numerator, 0)` obligation for each axis.
  This makes deferred binding requirements observable rather than treating a symbolic output
  expression as proof of fit.
- Conv3d remains eligible for ordinary exact CSE. Unlike recurrent scan, the architecture does
  not make equivalent Conv3d producer identities semantically distinct.
- Backward-capable compilation rejects Conv3d from the complete original forward inventory, not
  only from the selected gradient ancestry, so `FORWARD_ONLY` is the sole adopting mode until
  0006C.
- Production first-order coverage remains byte-for-byte unchanged. Its source-backed test expands
  only the deferred set and complete Model inventory counts.

## Known limitations

- No current production capability provider is expected to accept `CONV3D`, so forward Compiler
  adoption does not make the operation executable through Engine or a concrete backend.
- Deferred constraints are compile diagnostics and later binding obligations; this task does not
  provide a binding API or prove arbitrary runtime Shapes.
- Conv3d gradients and every differentiable role remain fail-closed until separate task 0006C.
- Conv3d execution, algorithms, resources, numerical realization, and performance remain future
  CPU/backend work.
- Repository-wide and downstream suites remain deferred under the recorded task-validation tier.

## Validation evidence

- Reused the implementation context's focused Compiler matrix: 6 suites, 40 tests, 0 skipped,
  failures, or errors.
- Reused the focused unchanged Model Conv1d/Conv2d/Conv3d boundary: 4 suites, 30 tests, 0 skipped,
  failures, or errors.
- Reused the single final `./gradlew :modules:compiler:test`: 34 suites, 231 tests, 0 skipped,
  failures, or errors. One intermediate focused fixture was corrected before this final passing
  evidence; no production defect remained.
- The documentation-focused pass ran `./gradlew :modules:compiler:javadoc`, inspected generated
  public Compiler Javadocs and all three changed package-private source Javadocs, validated local
  Markdown links and anchors, heading uniqueness, balanced fences, final newlines, CR/trailing
  whitespace, status/inventory wording, exact fifteen-path scope, and the empty Git index, then
  passed final `git diff --check`.
- Java tests were not repeated because the documentation pass changed no executable Java behavior
  and found no stale-evidence risk.

## Implementation notes

- `CapturedGraphInference` dispatches `CONV3D` to the existing structured owner.
  `StructuredOperationInference` independently derives its promoted rank-five descriptor and
  requests the seven channel/spatial obligations in the specified order; final graph validation
  checks the stored descriptor and retained constraints.
- Conv3d remains an ordinary CSE-eligible flat node. Existing publication, diagnostics, and
  `OperationCapabilityQuery`/Planning handoff carry it unchanged; no production provider accepts
  it and no lowering, preparation, execution, or gradient behavior was added.
- `AutogradPreflight` rejects the first Conv3d occurrence in complete original-forward postorder
  for both backward-capable modes before seeds or derivative Tensors. Production
  `FirstOrderGradientCoverage` and `ConvolutionGradientRules` remain unchanged; the boundary test
  proves 39/111/132 forward inventory against unchanged 37/107/128 first-order support plus the
  exact four deferred signatures.
- The affected Javadocs are meaningful and complete for their package-private contracts. No
  public Tensor or Compiler API signature changed.
- Reviewed and recorded no-change conclusions for Runtime and Training APIs; architecture, ADRs,
  and architecture tests; Planning, Prepare, Runtime, and Engine behavior; capability providers
  and backend plans; conformance and integration tests; Gradle and dependencies; Model Java and
  tests; and all other modules. None required an update within task scope.

## Completion summary

Completed Compiler 0006B within the authorized fifteen paths. Three package-private production
files adopt first-class Conv3d for independent forward inference/final validation and preserve a
pre-allocation backward boundary; four tests prove forward behavior, ordered constraints,
diagnostics, CSE/publication/Planning behavior, and the exact supported/deferred inventories.
The documentation pass finalized the Compiler and Tensor APIs, glossary, Model downstream-status
statements, task evidence, Compiler master plan, and roadmap. Compiler 0006C and 0007 remain Draft
without specifications; Model 0025H remains Complete and 0026 remains Draft without a
specification; CPU 0008 remains the next Draft task to plan separately, followed by CPU 0008A and
CPU 0008B–0008E. No unresolved issue remains and no follow-up is required for 0006B.

Status: Complete

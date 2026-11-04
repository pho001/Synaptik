# Task 0024: [Model](../../../../glossary.md#term-model) Capability and Contract Closure Audit

## Status

Complete

## Goal

[Run](../../../../glossary.md#term-run) the final [planning](../../../../glossary.md#term-planning)-only exit gate for the selected `modules/model` capability milestone.
Inventory the completed public and semantic contracts, compare them with the intentional
[inference](../../../../glossary.md#term-inference)/training baseline and its explicit exclusions, verify that earlier cleanup and
[adjoint](../../../../glossary.md#term-adjoint)-expressibility decisions remain closed, and execute the recorded repository [checkpoint](../../../../glossary.md#term-checkpoint).

The task produces one durable audit [artifact](../../../../glossary.md#term-artifact) with a falsifiable closure verdict. It does not add
or change Java behavior. A successful `CLOSED` verdict confirms that the selected [model](../../../../glossary.md#term-model) surface is
stable enough for downstream [compiler](../../../../glossary.md#term-compiler) and [lifecycle](../../../../glossary.md#lifecycle) work without claiming execution, [kernels](../../../../glossary.md#kernel),
[backend](../../../../glossary.md#backend) support, or a complete general-purpose [tensor](../../../../glossary.md#tensor) library.

[Mental model](../../../../glossary.md#term-mental-model):

```text
authoritative architecture
  + selected capability baseline
  + completed Java/Javadoc/API contracts
  + completed reset and adjoint decisions
  + repository checkpoint evidence
    -> model capability and contract closure verdict
       CLOSED
       BLOCKING_GAP
       ARCHITECTURE_DECISION_REQUIRED
```

## [Scope](../../../../glossary.md#term-scope)

- Create `docs/planning/modules/model/model-capability-contract-closure-audit.md` as the sole
  detailed result artifact.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact))</span>
- Inventory the complete current `modules/model` production and test trees rather than relying on
  task completion prose alone.
- Record the exact current public [`Tensor`](../../../../glossary.md#tensor) method count and group every method into one owning
  semantic family or documented convenience. Category counts must sum to the exact total.
- Inventory every current [`OperationKind`](../../../../glossary.md#operationkind) [constant](../../../../glossary.md#term-constant), accepted exact [`OperationAttrs`](../../../../glossary.md#operationattrs) class,
  `OperationSignature` input/output range, public or package-private construction owner, and
  output count. Record kinds without public Tensor construction and explain their selected owner.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- Verify that each kind has a non-empty immutable signature list, one unique signature per exact
  attributes class, locally valid count ranges, and no permissive kind/attributes pairing.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [signature glossary](../../../../glossary.md#term-signature))</span>
- Inventory foundation contracts for data types and typed scalars, static/named/expression
  Dimensions and Shapes, layouts/descriptors, identifiers, Tensor identity/storage, factories,
  producers/provenance, graph values/nodes/models, and public multi-output result carriers.<span data-glossary-navigation="true"> ([scalars glossary](../../../../glossary.md#term-scalar); [Dimensions glossary](../../../../glossary.md#dimension); [Shapes glossary](../../../../glossary.md#shape); [layouts glossary](../../../../glossary.md#layout); [descriptors glossary](../../../../glossary.md#term-descriptor); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance); [graph glossary](../../../../glossary.md#graph); [nodes glossary](../../../../glossary.md#node))</span>
- Verify every genuine multi-output occurrence's exact ordered public/hidden output roles,
  descriptors, producer sharing, output indices, and public carrier boundary.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [producer glossary](../../../../glossary.md#term-producer); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Assess the selected capability baseline by cohesive area: construction/import/random sources;
  elementwise numeric/comparison/logical/selection/cast operations; reductions/statistics/scans;
  layout/indexing/composition; linear algebra/attention; convolution/pooling; normalization;
  losses; explicit graph RNG/dropout; graph and storage foundations.<span data-glossary-navigation="true"> ([elementwise glossary](../../../../glossary.md#term-elementwise); [linear glossary](../../../../glossary.md#term-linear-meanings); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [pooling glossary](../../../../glossary.md#term-pooling); [normalization glossary](../../../../glossary.md#term-normalization); [losses glossary](../../../../glossary.md#loss); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- Classify each capability area as current public primitive, current public convenience,
  model semantic without public construction, deliberately deferred, or rejected from the core
  baseline. Do not turn the artifact or `capabilities.md` into a full library reference.
- Verify that the minimum selected inference/training baseline is representable as model metadata,
  including dynamic/symbolic Shapes and the public primitives selected by the completed
  adjoint-expressibility audit.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [metadata glossary](../../../../glossary.md#term-metadata); [adjoint glossary](../../../../glossary.md#term-adjoint))</span>
- Verify that every selected numerical contract either fixes its applicable special-value,
  empty-domain, accumulation, ordering/tie, duplicate-index, bounds, approximation, and
  determinism policies or explicitly records the unresolved policy and its later owner.
- Re-audit the cleanup decisions listed below against current source, tests, Javadoc, APIs,
  glossary, and capability prose.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Verify architecture ownership and dependency direction without restating architecture rules:
  model semantics remain backend-independent; compiler/autograd construction, prepare, runtime,
  backend selection, kernels, and execution remain outside <span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [backend glossary](../../../../glossary.md#backend); [compiler glossary](../../../../glossary.md#term-compiler); [autograd glossary](../../../../glossary.md#autograd); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [kernels glossary](../../../../glossary.md#kernel))</span>`modules/model`.
- Review Tensor, Compile, Training, Runtime, and public API references plus the glossary for exact
  current-versus-planned wording. Correct documentation-only drift only within the conditional
  authorized paths below; record any behavioral or architectural contradiction as a gap.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Run one final combined repository, architecture-test, model-Javadoc, Markdown, scope, and status
  checkpoint after the audit artifact and documentation are stable.<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Synchronize the capability baseline, [model](../../../../glossary.md#term-model) master plan, and roadmap with the verdict.
- If the verdict is `CLOSED`, mark the selected model milestone complete and identify the next
  project area without creating its detailed task specification.
- If the verdict is `BLOCKING_GAP`, keep the model milestone open and add at most one concise next
  Draft row for the first cohesive blocking frontier. Record all other findings in the artifact;
  do not create a detailed follow-up specification.
- If an architecture decision is required, stop before changing `ARCHITECTURE.md` or explanatory
  architecture documents and report `ARCHITECTURE_DECISION_REQUIRED` with task status
  `Incomplete`.

### Required cleanup-closure coverage

The result [artifact](../../../../glossary.md#term-artifact) must explicitly verify at least these earlier decisions and add any comparable
current issue found during inventory:

- `FAST_EXP`/`FAST_TANH` and public `fastExp`/`fastTanh` remain absent from the selected baseline.
- [Elementwise](../../../../glossary.md#term-elementwise) reciprocal uses `RECIPROCAL`/`reciprocal`, not `INV`/`inv` or parallel [aliases](../../../../glossary.md#aliasing-and-arena).
- Floating Tensor/Tensor and Tensor/scalar arithmetic use the complete normalized seven-operation
  family, with pairwise <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [scalar glossary](../../../../glossary.md#term-scalar))</span>`minimum`/`maximum` distinct from reductions.
- Selected integral arithmetic, comparisons, reductions, arg-min/arg-max, and their rejected
  mixed-category or unsupported operations remain coherent.
- Gather, Gather Elements, Gather-ND, Select, Slice, Scatter Add, Scatter Elements, and Scatter-ND
  names and Shape relationships remain distinct; <span data-glossary-navigation="true"> ([Gather glossary](../../../../glossary.md#gather); [Gather Elements glossary](../../../../glossary.md#gather-elements); [Gather-ND glossary](../../../../glossary.md#gather-nd); [Slice glossary](../../../../glossary.md#slice); [Scatter Add glossary](../../../../glossary.md#scatter-add); [Scatter Elements glossary](../../../../glossary.md#scatter-elements); [Scatter-ND glossary](../../../../glossary.md#scatter-nd); [Shape glossary](../../../../glossary.md#shape))</span>`take` and first-class UNSTACK semantics remain
  absent.
- Public `unstack` remains explicit independent SELECT composition rather than false shared
  multi-output provenance.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- Masked sum/[mean](../../../../glossary.md#term-mean) use explicit ordinary right-aligned [broadcasting](../../../../glossary.md#broadcasting) and no [heuristic](../../../../glossary.md#term-heuristic) [axis](../../../../glossary.md#term-axis) mapping.
- Prefix population is test-data infrastructure rather than `TensorFactory` public surface.
- `TensorFactory`, `TensorRandoms`, `TensorRanges`, and constant/import construction ownership
  remains separated as selected by cleanup.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Typed [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) remains the sole exact scalar-attribute representation for supported data
  types, including padding constants; raw-double semantic alternatives remain absent.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding))</span>
- Public slicing supports signed non-zero steps, while resolved view geometry remains limited to
  representable positive-stride cases.<span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings); [stride glossary](../../../../glossary.md#term-stride))</span>
- Primitive-array `take` and its [eager](../../../../glossary.md#term-eager-evaluation) prevalidation/ID [side effect](../../../../glossary.md#term-side-effect) remain absent.
- `foldAxis` remains an intentional generally useful public overlap-add primitive, while no
  operation-specific backward kind is introduced.
- Symbolic Dimension expressions cover the selected addition, product, and division-derived Shape
  needs without pretending to bind runtime extents in model construction.<span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension); [runtime glossary](../../../../glossary.md#term-runtime); [model glossary](../../../../glossary.md#term-model))</span>
- Invalid kind/attributes combinations remain prevented by family-owned typed [signatures](../../../../glossary.md#term-signature).
- Multi-output producer identity and output indices remain sufficient for dropout, batch-normal
  training, top-K, and attention weights without introducing a generic tuple hierarchy.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [identity glossary](../../../../glossary.md#term-identity-meanings); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [weights glossary](../../../../glossary.md#term-weight))</span>
- No [operation](../../../../glossary.md#operation)-specific `*_BACKWARD` family has appeared where the completed adjoint matrix chose
  composition from general public primitives and existing producer outputs.<span data-glossary-navigation="true"> ([adjoint glossary](../../../../glossary.md#term-adjoint); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Java production or test creation, deletion, renaming, or modification
- Gradle/build, dependency, module, package, architecture-test, conformance-test, or
  integration-test changes<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- a new or changed Tensor method, constructor, factory, helper, result carrier, descriptor,
  producer, provenance, graph, Shape, Dimension, layout, data type, scalar, operation kind,
  attributes type, signature, or semantic policy<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [carrier glossary](../../../../glossary.md#term-storage-carrier); [descriptor glossary](../../../../glossary.md#term-descriptor); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [graph glossary](../../../../glossary.md#graph); [Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [layout glossary](../../../../glossary.md#layout); [data type glossary](../../../../glossary.md#data-type--datatype); [scalar glossary](../../../../glossary.md#term-scalar); [operation glossary](../../../../glossary.md#operation); [signature glossary](../../../../glossary.md#term-signature))</span>
- implementing, repairing, or redesigning a capability found by the audit
- reopening completed numerical policies merely because another framework differs
- choosing compiler graph capture, gradient rules, saved-value lifetime, optimization,
  accumulation, backend lowering, kernel selection, execution, or runtime state<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [gradient glossary](../../../../glossary.md#term-gradient); [lifetime glossary](../../../../glossary.md#term-lifetime); [lowering glossary](../../../../glossary.md#lowering); [kernel glossary](../../../../glossary.md#kernel); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- claiming [backend](../../../../glossary.md#backend) support or numerical execution from [model](../../../../glossary.md#term-model) [metadata](../../../../glossary.md#term-metadata) tests
- a broad comparison against external tensor libraries, legacy parity exercise, or speculative
  catalog of every possible operation/data type
- changing `ARCHITECTURE.md`, focused architecture documentation, ADRs, or architecture tests
- creating a detailed task 0024A, task 0025, or downstream project-area specification
- editing completed task specifications or removing their historical evidence
- unrelated documentation cleanup or turning the audit [artifact](../../../../glossary.md#term-artifact) into an [API](../../../../glossary.md#term-application-programming-interface-api) tutorial

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Documentation profile index](../../../../developer-guide/documentation/README.md)
- [General style](../../../../developer-guide/documentation/general-style.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model master plan](../master-plan.md)
- [Adjoint-expressibility audit task](0023-adjoint-expressibility-audit.md)<span data-glossary-navigation="true"> ([Adjoint glossary](../../../../glossary.md#term-adjoint))</span>
- [Adjoint-expressibility result](../adjoint-expressibility-audit.md)
- [Task 0023F](0023f-scaled-dot-product-attention-weights-output.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Runtime API](../../../../api/runtime-api.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [Public API status](../../../../api/public-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `ARCHITECTURE.md` remains authoritative. Planning and audit findings cannot silently create or
  change an architecture rule.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- The audit assesses whether the model-owned semantic representation is sufficient and coherent;
  it does not judge model completeness by current compiler, runtime, or backend implementation.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend))</span>
- [`Tensor`](../../../../glossary.md#tensor) remains public mutable API state rather than an IR node. Producer/provenance metadata
  remains pre-capture occurrence identity rather than compiled graph state.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [node glossary](../../../../glossary.md#node); [Producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [metadata glossary](../../../../glossary.md#term-metadata); [capture glossary](../../../../glossary.md#term-capture-meanings); [occurrence glossary](../../../../glossary.md#term-occurrence); [identity glossary](../../../../glossary.md#term-identity-meanings); [compiled graph glossary](../../../../glossary.md#compiled-graph--compiledgraphmodel))</span>
- [`Operation`](../../../../glossary.md#operation) remains backend-independent and contains no backend support, lowering, kernel, or
  execution behavior.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [kernel glossary](../../../../glossary.md#kernel))</span>
- Gradient rules and backward graph construction remain compiler-owned. The audit may verify
  expressibility but must not implement compiler adoption.<span data-glossary-navigation="true"> ([Gradient glossary](../../../../glossary.md#term-gradient); [backward graph glossary](../../../../glossary.md#backward-graph))</span>
- [Runtime](../../../../glossary.md#term-runtime) hot paths remain free of `Operation` and `CompiledNode`.
- The audit distinguishes selected public primitives, public conveniences, compiler-composable
  semantics, intentional deferrals, and rejected capabilities without duplicating the architecture
  contract in planning prose.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- Current source/tests are primary implementation evidence. Completed tasks and legacy are
  supporting evidence, not design authority.
- No Java or architecture edit is authorized. A contradiction requiring one changes the verdict,
  not the scope.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

## Package impact

No Java package is added, changed, or moved.

The audit reviews the complete current `io.github.pho001.synaptik.model` package tree and mirrors
its existing [ownership](../../../../glossary.md#term-ownership-meanings) boundaries in the result [artifact](../../../../glossary.md#term-artifact). It must not propose package movement as
incidental cleanup. A proven package problem is a separately bounded finding.

## Affected files

Always created or updated — exactly five paths:

- add `docs/planning/modules/model/model-capability-contract-closure-audit.md`
- add and then finalize this task
- modify `docs/planning/modules/model/capabilities.md`
- modify `docs/planning/modules/model/master-plan.md`
- modify `docs/planning/roadmap.md`

Conditionally permitted for documentation-only drift found by the audit — at most six paths:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/api/training-api.md`
- `docs/api/runtime-api.md`
- `docs/api/public-api.md`
- `docs/glossary.md`

Review without modification: every file under `modules/model/src/main/java` and
`modules/model/src/test/java`; `AGENTS.md`; `ARCHITECTURE.md`; focused architecture docs/ADRs/tests;
all completed [model](../../../../glossary.md#term-model) tasks and result [artifacts](../../../../glossary.md#term-artifact) needed as evidence; [Gradle](../../../../glossary.md#term-gradle); other [modules](../../../../glossary.md#term-module);
[backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) and integration tests; legacy reference material.

## Maximum [scope](../../../../glossary.md#term-scope)

This [planning](../../../../glossary.md#term-planning)-only task may create or modify at most the eleven documentation/planning paths above.
It must modify none of the review-only Java, test, [Gradle](../../../../glossary.md#term-gradle), architecture, [ADR](../../../../glossary.md#term-architecture-decision-record-adr), or cross-[module](../../../../glossary.md#term-module) paths.

The five always-affected paths form the expected result. Conditional [API](../../../../glossary.md#term-application-programming-interface-api)/glossary paths may change
only to correct documentation status, terminology, link, or example drift established against
already completed behavior. If a correction would select new behavior, conceal a Java mismatch,
or require a twelfth path, stop and record a blocking gap rather than expanding scope.

## Required audit [artifact](../../../../glossary.md#term-artifact)

The result artifact must be newcomer-readable but evidence-dense. It contains exactly these
top-level sections, with subsections and tables as needed:

1. `Executive conclusion and closure verdict`
2. `Authority, scope, and method`
3. `Architecture-boundary assessment`
4. `Foundation-contract inventory`
5. `Public Tensor and construction inventory`
6. `Operation kind, attributes, and signature inventory`
7. `Multi-output occurrence inventory`
8. `Selected capability baseline assessment`
9. `Dynamic Shape and typed-scalar assessment`
10. `Numerical-policy closure assessment`
11. `Legacy-cleanup and unusual-capability closure`
12. `Adjoint-prerequisite closure`
13. `Documentation and terminology consistency`
14. `Deferred, rejected, and downstream-owned capabilities`
15. `Findings, severity, and disposition`
16. `Checkpoint evidence and model-milestone decision`

### Closure verdict

The first section records exactly one verdict:

- `CLOSED` — no blocking model representation, selected-baseline, documentation, or architecture
  conflict remains. Explicit deferrals are permitted when their owner and non-blocking rationale
  are recorded.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- `BLOCKING_GAP` — the audit is complete, but at least one model-owned gap prevents milestone
  closure. Each gap names evidence, impact, owner, dependency, and the first bounded follow-up.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- `ARCHITECTURE_DECISION_REQUIRED` — evidence conflicts with the authoritative architecture or
  requires a new ownership/dependency decision. Stop without editing architecture and leave task
  0024 <span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>`Incomplete`.

For `CLOSED`, the task, [model](../../../../glossary.md#term-model) master-plan frontier, and model roadmap area become `Complete`.
For `BLOCKING_GAP`, the audit task may become `Complete` because the requested audit finished, but
the model master plan and roadmap area remain open and expose at most one concise next Draft row.

### Inventory requirements

- Public Tensor inventory groups exact signatures by semantic family or convenience and includes
  a category subtotal whose sum equals the reflected public-method total.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [signatures glossary](../../../../glossary.md#term-signature))</span>
- Operation inventory includes every kind constant and exact signature. A family-level row may
  group constants only when attributes class and input/output ranges are identical; otherwise it
  must split them.<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation); [constant glossary](../../../../glossary.md#term-constant))</span>
- Construction inventory maps every public primitive or convenience to its actual helper/producer
  chain and distinguishes direct one-occurrence construction from explicit public composition.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Multi-output inventory includes public, hidden, and non-differentiable roles and verifies exact
  producer/output-index relationships.<span data-glossary-navigation="true"> ([differentiable glossary](../../../../glossary.md#derivative-and-chain-rule))</span>
- Foundation inventory records contracts and readiness, not every accessor or test method.
- Capability baseline tables distinguish current primitive, convenience, semantic-only,
  deferred, and rejected statuses.
- Findings use exactly `BLOCKING`, `NON_BLOCKING_DEFERRED`, `DOCUMENTATION_DRIFT`, or
  `NO_CHANGE_CONFIRMED` severity/disposition labels.
- Every claim links to current source/test/API evidence or names the exact inspection command.
  Completed task prose alone is insufficient evidence.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Acceptance criteria

- The result [artifact](../../../../glossary.md#term-artifact) contains every required top-level section and exactly one closure verdict.
- Every current production and test file in `modules/model` was included in a recorded inventory
  process; no current operation family is omitted.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- The public Tensor family subtotals equal the exact reflected method count and match current API
  documentation.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Every OperationKind constant maps to exact attrs and signature ranges; duplicate attrs variants,
  malformed ranges, missing signatures, and unowned semantic-only kinds are explicitly checked.<span data-glossary-navigation="true"> ([OperationKind glossary](../../../../glossary.md#operationkind); [constant glossary](../../../../glossary.md#term-constant); [signature glossary](../../../../glossary.md#term-signature))</span>
- Every public direct expression maps to an accepted occurrence signature and every public
  convenience maps to its exact primitive producer chain.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [producer glossary](../../../../glossary.md#term-producer))</span>
- Every multi-output operation records exact output roles, descriptors, public/hidden status,
  shared producer identity, and output indices.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Foundation, dynamic Shape, typed scalar, storage, identity, producer/provenance, and graph model
  contracts are assessed against current code rather than assumed from task status.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [scalar glossary](../../../../glossary.md#term-scalar); [provenance glossary](../../../../glossary.md#provenance); [graph glossary](../../../../glossary.md#graph); [model glossary](../../../../glossary.md#term-model))</span>
- The selected minimal inference/training capability areas are each classified with a concrete
  readiness conclusion.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference))</span>
- All required cleanup-closure items are checked against current source and documentation.
- Numerical-policy gaps distinguish blocking forward-semantic ambiguity from intentionally
  compiler/backend-owned derivative or execution policy.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- The completed 0023A–0023F capabilities are rechecked against the adjoint matrix; no narrow
  operation-specific backward kind is introduced or requested without new evidence.<span data-glossary-navigation="true"> ([adjoint glossary](../../../../glossary.md#term-adjoint); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Architecture, dependency, lifecycle, runtime-hot-path, and cross-layer ownership checks produce
  explicit evidence-backed conclusions.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [lifecycle glossary](../../../../glossary.md#lifecycle); [runtime glossary](../../../../glossary.md#term-runtime); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Tensor, Compile, Training, Runtime, public API, glossary, capability, master-plan, and roadmap
  current/planned wording is mutually consistent or the discrepancy is reported.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- No Java, test, Gradle, architecture, ADR, another-module, conformance, or integration file
  changes.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Exactly one final combined repository [checkpoint](../../../../glossary.md#term-checkpoint) passes after audit/documentation stability.
- Markdown links/anchors/fences, final newlines, status [synchronization](../../../../glossary.md#term-synchronization), permitted [scope](../../../../glossary.md#term-scope), and
  `git diff --check` pass.
- No detailed 0024A, 0025, downstream module, compiler, runtime, or backend task specification is
  created.
- The task records local decisions, limitations, commands/results, agent identity, inventory
  method, findings, milestone disposition, and canonical completion summary.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

This documentation-focused audit changes no executable Java. Perform source/test inventories and
documentation edits first. Then [run](../../../../glossary.md#term-run) exactly one final combined [Gradle](../../../../glossary.md#term-gradle) [checkpoint](../../../../glossary.md#term-checkpoint):

```bash
./gradlew test :testing:architecture-tests:test :modules:model:javadoc
```

Gradle deduplicates the explicitly named architecture-test task if root `test` already selects it.
Record executed/skipped task counts, repository test counts by available report evidence, and any
project with no tests. Do not rerun successful suites merely to produce different formatting.

Create temporary inspection helpers only under `/tmp`; do not add repository tooling. Use current
compiled classes plus Java 26 reflection/`javap` and source inventories to verify at least:

- exact public [Tensor](../../../../glossary.md#tensor) method count and family subtotals;
- every [`OperationKind`](../../../../glossary.md#operationkind) implementation, enum [constant](../../../../glossary.md#term-constant), attrs pairing, and [signature](../../../../glossary.md#term-signature) range;
- every direct [producer](../../../../glossary.md#term-producer)'s accepted input/output count;
- all multi-output kinds, public carriers, hidden slots, and output indices;
- absence/presence claims in the required cleanup [matrix](../../../../glossary.md#term-matrix-meanings); and
- production/test file coverage counts.

Documentation and [scope](../../../../glossary.md#term-scope) validation:

```bash
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

Also confirm manually that all changed files are among the eleven authorized paths, no detailed
`0024A` or later task file exists, completed task history remains intact, and active status is
consistent across this task, the result [artifact](../../../../glossary.md#term-artifact), capabilities, master plan, and roadmap.

This is the recorded [model](../../../../glossary.md#term-model) capability checkpoint, so repository-wide and architecture validation
is required rather than deferred to [CI](../../../../glossary.md#term-continuous-integration-ci).

## Dependencies

- Tasks 0001–0013A for value, Shape/layout, operation, graph, storage, Tensor, factory, and
  producer/provenance foundations.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [graph glossary](../../../../glossary.md#graph); [Tensor glossary](../../../../glossary.md#tensor); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- Tasks 0014A–0018J and follow-ups for the initial [operation](../../../../glossary.md#operation) and expression families.
- Tasks 0018K–0018V for capability reset, signature hardening, multi-output provenance, symbolic
  extents, typed scalars, taxonomy cleanup, and missing core numeric/reduction semantics.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [scalars glossary](../../../../glossary.md#term-scalar))</span>
- Tasks 0019–0022B and follow-ups for the selected modern [inference](../../../../glossary.md#term-inference)/training operation baseline.
- Task 0023 and its result [artifact](../../../../glossary.md#term-artifact) for the complete [adjoint](../../../../glossary.md#term-adjoint)-expressibility decision [matrix](../../../../glossary.md#term-matrix-meanings).
- Tasks 0023A–0023F for all six evidence-selected general public prerequisites.

## Follow-up tasks

- A `CLOSED` verdict ends the selected model capability milestone. The roadmap may identify the
  next project area, but this task creates no downstream detailed specification.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- A `BLOCKING_GAP` verdict may add at most one concise next Draft model row for the first cohesive
  blocking frontier. It creates no detailed specification and does not implement the gap.
- An `ARCHITECTURE_DECISION_REQUIRED` verdict stops the task and requests an explicit architecture
  decision before any architecture or implementation edit.
- Non-blocking deferred capabilities retain their named later owner and do not keep the selected
  model milestone open.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- Final verdict: `BLOCKING_GAP`. Draft 0024A is the one concise next model row for correcting the
  stale `GraphValue` [Javadoc](../../../../glossary.md#term-javadoc) [Tensor](../../../../glossary.md#tensor)-status sentence. This task creates no detailed 0024A file.

## Architecture impact

Expected impact: None.

This task audits [conformance](../../../../glossary.md#term-conformance) to the existing architecture and updates non-authoritative [planning](../../../../glossary.md#term-planning)
status. It must not change architecture. A contrary finding changes the verdict to
`ARCHITECTURE_DECISION_REQUIRED` and stops execution.

## Implementation prompt

Use this prompt in one separate clean documentation-focused task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, documentation/planning rules and profiles, roadmap, model
capabilities/master plan, completed task history and result artifacts required by task 0024,
Tensor/Compile/Training/Runtime/Public APIs, glossary, and the complete current modules/model
production and test inventories.

Execute docs/planning/modules/model/tasks/0024-model-capability-and-contract-closure-audit.md
exactly inside its eleven authorized documentation/planning paths. This is the required clean
documentation-focused audit context; there is no Java implementation pass. You may use up to
three read-only subagents for disjoint foundation, operation-surface, and documentation/history
inventories, but you own synthesis, edits, evidence, and the final verdict.

Add no Java/test/Gradle/architecture/cross-module change and do not implement a finding. Stop on
architecture uncertainty or scope conflict. Run the single final combined checkpoint only after
the artifact and documentation are stable. Synchronize model milestone status from the verdict,
do not create a detailed follow-up specification, and mark task 0024 Complete only when every
audit and validation criterion succeeds. No second documentation pass is required solely to
duplicate this already clean documentation-focused task.
```

## Local decisions

- Keep 0024 as one cohesive planning-only closure audit rather than split inventory, cleanup,
  architecture, and checkpoint mechanics into separate tasks.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Treat source/tests as primary implementation evidence and completed task prose as supporting
  history.
- Use a permanent result artifact separate from the task execution record, consistently with the
  completed adjoint audit.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [adjoint glossary](../../../../glossary.md#term-adjoint))</span>
- Close the selected model milestone on representational and contract readiness, not on compiler,
  backend, runtime, or numerical execution availability.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Permit documentation-only corrections in the established API/glossary files, while every Java
  or architecture mismatch becomes a finding rather than hidden scope expansion.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [scope glossary](../../../../glossary.md#term-scope))</span>
- Allow non-blocking explicit deferrals. The selected baseline is intentionally smaller than a
  general-purpose tensor library.<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor))</span>
- Limit a blocking verdict to one next concise Draft frontier so progressive planning remains
  dependency-ordered.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- Use parallel read-only inventory lanes only to reduce audit latency; one primary
  documentation-focused agent owns the final evidence and edits.<span data-glossary-navigation="true"> ([lanes glossary](../../../../glossary.md#term-lane); [latency glossary](../../../../glossary.md#term-latency))</span>
- Record the stale `GraphValue` Javadoc as blocking rather than edit a review-only Java path or
  conceal the contradiction. The correction is mechanically clear and requires no architecture
  decision, but documentation closure cannot claim success while the contract calls a current API
  planned.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Correct the independent Public API and Training API drifts inside the authorized conditional
  paths because both changes only synchronize documentation with completed source/architecture.

## Known limitations

- The audit cannot prove backend numerical conformance or execution support because those layers
  are not implemented by <span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>`modules/model`.
- Reflection and source inventories prove declared contracts and construction coverage, not the
  mathematical correctness of future kernels.<span data-glossary-navigation="true"> ([kernels glossary](../../../../glossary.md#kernel))</span>
- Explicit compiler/backend-owned policy deferrals may remain after model closure when they do not
  make forward model semantics ambiguous.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [model glossary](../../../../glossary.md#term-model))</span>
- The audit does not promise that no future use case will justify another model capability; it
  closes only the selected current inference/training baseline.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference))</span>
- Legacy evidence can show an omitted capability but cannot override current selected design.

## [Validation](../../../../glossary.md#term-validation) evidence

- Planning preparation read the repository instructions, authoritative architecture, focused
  architecture index, documentation rules and profiles, planning guide, roadmap, model
  capabilities/master plan, the completed adjoint audit and task 0023F, and current validation
  configuration needed to bound this audit.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [adjoint glossary](../../../../glossary.md#term-adjoint))</span>
- The specification defines five always-affected and six conditionally affected paths, for an
  exact maximum of eleven documentation/planning paths and no executable or architecture path.
- [Planning](../../../../glossary.md#term-planning) validation originally confirmed 0024 as the sole [`Ready`](../../../../glossary.md#term-ready) model task. Execution found
  one bounded blocker, completed the audit, added only a concise Draft 0024A row, and introduced no
  detailed 0024A or later task specification.
- Relative-link inspection found no broken task links. The repository Markdown validator passed
  211 files, 3,564 local links, 207 local anchors, 2,670 fence markers, final newlines, and
  trailing whitespace.
- `git diff --check` passed. The final planning diff contains exactly this task, model
  capabilities, model master plan, and roadmap; it contains no Java, test, Gradle, architecture,
  or completed-task edit.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle))</span>
- Clean documentation-focused [execution context](../../../../glossary.md#term-execution-context) `/root/execute_0024` read the authoritative
  architecture, focused architecture documents, documentation/planning profiles, roadmap,
  capability/master plans, completed task/result history, all five API references, glossary, and
  the complete model trees. Read-only foundation lane<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [lane glossary](../../../../glossary.md#term-lane))</span>
  `/root/execute_0024/foundation_inventory` independently inventoried foundations and shared
  outputs; the primary context owned synthesis, edits, commands, and verdict.
- Sorted inventories covered 176 production Java paths and 129 test-tree paths (128 Java plus one
  `.gitkeep`). Java 26 reflection and `/tmp/ModelAudit.java` confirmed exactly 200 declared public
  Tensor methods whose 20 family subtotals sum to 200; 37 kind enum types, 107 constants, 47
  concrete attributes types, and 127 immutable exact signatures; valid ranges, unique exact attrs
  variants, fail-closed pairing, and complete attrs coverage passed.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [constants glossary](../../../../glossary.md#term-constant); [immutable glossary](../../../../glossary.md#term-immutable); [signatures glossary](../../../../glossary.md#term-signature); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Source/helper/tests rechecked all direct producer counts, all four genuine multi-output forms,
  public/hidden roles and descriptors, convenience chains, dynamic Shape/typed-scalar contracts,
  numerical-policy ownership, cleanup decisions, and 0023A–0023F prerequisites. No behavioral or
  architecture gap was found. <span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [descriptors glossary](../../../../glossary.md#term-descriptor); [Shape glossary](../../../../glossary.md#shape); [scalar glossary](../../../../glossary.md#term-scalar); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>`GraphValue.java` lines 9–12 supply the sole blocking stale-Javadoc
  finding; it was not edited.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- The single final combined command
  `./gradlew test :testing:architecture-tests:test :modules:model:javadoc` passed in two seconds.
  Gradle reported 39 actionable tasks: 2 executed and 37 up-to-date. XML evidence reports 1,016
  model tests across 127 suites plus one architecture test, with zero failures, errors, or skips.
  All other selected projects reported no tests.<span data-glossary-navigation="true"> ([architecture test glossary](../../../../glossary.md#term-architecture-test))</span>
- `python3 /tmp/validate_synaptik_markdown.py` passed 212 Markdown files, 3,621 local links, 207
  local anchors, 2,676 fence markers, final newlines, and trailing whitespace. `git diff --check`
  passed.
- The final diff contains exactly seven authorized paths: the five always-affected planning paths
  plus `docs/api/public-api.md` and `docs/api/training-api.md`. Scope/status scans found no Java,
  test, Gradle, architecture, ADR, other-module, conformance, or integration change; no detailed
  0024A/0025 file; preserved completed history; task 0024 Complete; model milestone open; and
  Draft 0024A as the sole next model frontier.<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Implementation notes

Created the durable audit [artifact](../../../../glossary.md#term-artifact) and classified every required foundation, [Tensor](../../../../glossary.md#tensor) method,
[operation signature](../../../../glossary.md#operation-signature), multi-output [occurrence](../../../../glossary.md#term-occurrence), capability area, numerical-policy family, cleanup
decision, [adjoint](../../../../glossary.md#term-adjoint) prerequisite, documentation boundary, and deferral. The verdict is
`BLOCKING_GAP`, not `ARCHITECTURE_DECISION_REQUIRED`: [model](../../../../glossary.md#term-model) behavior is coherent, but one stale
review-only `GraphValue` [Javadoc](../../../../glossary.md#term-javadoc) sentence requires the bounded follow-up. Authorized Public [API](../../../../glossary.md#term-application-programming-interface-api)
and Training API drift was corrected. The master plan and roadmap leave the model area open and
expose only Draft 0024A without a detailed specification.

## Completion summary

- Completed changes: executed the full [model](../../../../glossary.md#term-model) capability/contract audit, recorded the
  `BLOCKING_GAP` result, corrected two authorized API-documentation drifts, ran the single
  repository checkpoint, and synchronized the open model milestone with one Draft follow-up.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Files changed or created: exactly seven —
  `docs/planning/modules/model/model-capability-contract-closure-audit.md`, this task,
  `docs/planning/modules/model/capabilities.md`, `docs/planning/modules/model/master-plan.md`,
  `docs/planning/roadmap.md`, `docs/api/public-api.md`, and `docs/api/training-api.md`.
- Tests and validation: combined root/architecture/model-Javadoc checkpoint passed; 1,017 tests
  across 128 suites with no failures, errors, or skips; exact reflection/signature/inventory
  checks, Markdown links/anchors/fences/newlines/whitespace, authorized scope, status, and<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [signature glossary](../../../../glossary.md#term-signature); [scope glossary](../../../../glossary.md#term-scope))</span>
  `git diff --check` passed.
- Unresolved issue: `GraphValue` Javadoc incorrectly calls the implemented public mutable Tensor
  API planned. It is the sole blocking finding and was preserved because Java is review-only.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Follow-up required: plan and execute concise Draft 0024A to correct that Javadoc and close the
  model milestone. No detailed specification was created by task 0024.
- Documentation/Javadoc review: Tensor, Compile, Runtime, and glossary remain accurate without
  modification; Public and Training API drift was corrected; <span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [Runtime glossary](../../../../glossary.md#term-runtime))</span>`GraphValue` Javadoc is the recorded
  blocker. No Java behavior or architecture changed.

Status: Complete

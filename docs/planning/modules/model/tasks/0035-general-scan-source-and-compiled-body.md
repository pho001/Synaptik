# Task Model0035: [General scan](../../../../glossary.md#general-differentiable-scan) source and compiled body

**Editorial reading key (current context, not a status update).**

This record closes only Model0035’s source/IR foundation; current references describe later layers,
not additions to its historical acceptance. The caller supplies `ScanBody` formals (placeholders)
and positional actual Tensors to `Scan.apply`: carry passes state, sequences supply time slices, and
invariants stay unchanged. A leaf has no producing expression, not necessarily stored values;
declared formals here are storage-free. Descriptions cross this construction boundary, not evaluated
values; storage-free expressions neither execute nor own result buffers. Model semantics defines
meaning; providers realize supported routes. Follow [canonical results and
eligibility](../../../../reference/tensor/general-scan.md#canonical-results-residual-metadata-and-eligibility)
for these distinctions: emitting previous carry can receive true outer gradient eligibility from a
gradient-enabled sequence, without promising nonzero derivatives.

Canonical ownership identifies result wrappers, not storage ownership. Residuals name retainable
internal values; demand selects needed histories without allocating them. Follow [compiled graph
boundaries](../../../../reference/tensor/compiled-graph.md#graph-values-and-compiled-nodes) for the
separate callee contract: `CompiledScanBody` accepts a supplied graph and ordered role/demand
bindings, not source capture. Body IDs are graph-local; repeated role bindings do not merge outer
ports. Structural validity grants neither descriptor inference nor backend execution.

## Status

Complete — bounded [Model](../../../../glossary.md#term-model) source/intermediate-representation ([IR](../../../../glossary.md#term-intermediate-representation-ir)) foundation accepted by main on
2026-10-22 after required reviews and the named compatibility [checkpoint](../../../../glossary.md#term-checkpoint); no successor launch.

## Change class

Class C — public source-body [API](../../../../glossary.md#term-application-programming-interface-api), [canonical](../../../../glossary.md#term-canonical-form) [producer](../../../../glossary.md#term-producer) [ownership](../../../../glossary.md#term-ownership-meanings) and a scoped compiled-[node](../../../../glossary.md#node) boundary.

## Goal

Deliver an independently reviewable [Model](../../../../glossary.md#term-model) foundation: reusable source bodies, [canonical](../../../../glossary.md#term-canonical-form) scan
results/[residual](../../../../glossary.md#scan-residual-port-and-history-demand) ports and structurally validated compiled-body attachments. Construction describes
meaning; it does not [capture](../../../../glossary.md#term-capture-meanings), differentiate, [prepare](../../../../glossary.md#prepare) or execute a scan.

## [Scope](../../../../glossary.md#term-scope) and non-goals

- Add only the six production types in Files and symbols; extend existing producer/node construction
  without removing old constructor forms. Include focused tests, Javadoc and targeted documentation.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [node glossary](../../../../glossary.md#node); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Enact the two user-approved clarifications below in the root, owning recurrent contract,
  relevant explanation and ADR0045; no other authority changes are authorized.
- No Compiler capture/inference/AD, collective source escape checking, demand analysis, Planning
  query, Prepare projection, backend execution, NN/padding or persistence implementation.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [AD glossary](../../../../glossary.md#autograd); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [padding glossary](../../../../glossary.md#padding); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- No new module/dependency, public runner, global ownership registry or extra helper abstraction;
  no custom/unknown semantics, numerical policy, native/performance campaign or execution claim.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [registry glossary](../../../../glossary.md#term-registry))</span>

## Contracts

Start with [root scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>.
Read these exact headings, not unrelated contracts or completed task histories:

- Root [Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence),
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>, and
  [Documentation authority and architecture updates](../../../../../ARCHITECTURE.md#documentation-authority-and-architecture-updates).
- Foundational [modules/model](../../../../architecture/contracts/foundational-modules.md#modulesmodel)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module); [model glossary](../../../../glossary.md#term-model))</span>:
  producer ownership, immutable records and scalar-only attributes.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [immutable glossary](../../../../glossary.md#term-immutable); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Recurrent [General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)<span data-glossary-navigation="true"> ([General differentiable scan glossary](../../../../glossary.md#general-differentiable-scan))</span>,
  including **Source body, attachments and scoped [identities](../../../../glossary.md#term-identity-meanings)**, **[Canonical](../../../../glossary.md#term-canonical-form) [residual](../../../../glossary.md#scan-residual-port-and-history-demand) ports and demand**,
  **Reverse scan and exact [primal](../../../../glossary.md#term-primal) [bindings](../../../../glossary.md#term-binding)**, **Traversal, empty domains and declared [qualification](../../../../glossary.md#term-qualification)**,
  and **Qualification, performance and migration gates**: sole complete scan semantic [owner](../../../../glossary.md#term-owner).

If a new contract conflict appears, stop and report it. The following choices were approved and
independently accepted in authority before code; this [planning](../../../../glossary.md#term-planning) packet is not their semantic owner.

## Accepted pre-code clarifications

User approved both choices on 2026-10-22; authority review/main acceptance preceded code:

1. Existing valid PURE Model algebra/dtypes are independent of CPU qualification (FLOAT16 TANH and
   F32 RELU construct). Existing source exclusions still reject; no new semantics or AD/backend grant.<span data-glossary-navigation="true"> ([PURE glossary](../../../../glossary.md#term-pure); [Model glossary](../../../../glossary.md#term-model); [dtypes glossary](../../../../glossary.md#data-type--datatype); [CPU glossary](../../../../glossary.md#term-cpu); [qualification glossary](../../../../glossary.md#term-qualification); [FLOAT16 glossary](../../../../glossary.md#term-float16); [TANH glossary](../../../../glossary.md#term-tanh); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [AD glossary](../../../../glossary.md#autograd); [backend glossary](../../../../glossary.md#backend))</span>
2. `G` ORs every actual [carry](../../../../glossary.md#term-carry)/sequence/[invariant](../../../../glossary.md#term-invariant) `requiresGrad` flag. Differentiable outer ports get G;
   nondifferentiable ports get false; formals retain actual flags. Unused inputs/T=0 are included.
   Regression: <span data-glossary-navigation="true"> ([Differentiable glossary](../../../../glossary.md#derivative-and-chain-rule); [Regression glossary](../../../../glossary.md#term-regression))</span>`h0=false`, `x=true`, `next=h+xStep`, `emit=h` preserves temporal outer eligibility,
   not a nonzero-dependency or AD/backend guarantee. Full acceptance remains below.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>

## Dependencies and integration

- Depends on: Model0034 Complete, including independent authority APPROVE and main acceptance (2026-10-21).
- Conflicts with: scan source/IR/roles/residuals, shared authority/docs, and downstream query/AD/capture/
  projection/backend/NN/persistence scan changes; serialize all overlapping work.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [AD glossary](../../../../glossary.md#autograd); [capture glossary](../../../../glossary.md#term-capture-meanings); [backend glossary](../../../../glossary.md#backend); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- Parallel group: None.
- Common base revision: `6da23a0b42f015979a88f5e84416799f5ada3f40` (HEAD=origin/main), plus accepted
  uncommitted Model0034 authority/planning work; do not treat dirty accepted work as a merged commit.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [commit glossary](../../../../glossary.md#term-commit))</span>
- Integration order: clarification writer → independent authority review/main acceptance → Model code
  → independent code/docs review → main foundation checkpoint; downstream slices remain Draft.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- Integration [validation](../../../../glossary.md#term-validation): named **Model0035 CompiledNode compatibility [checkpoint](../../../../glossary.md#term-checkpoint)**, commands below;
  program-wide native root `test javadoc` remains at the later low-level join.
- Shared-document integration [owner](../../../../glossary.md#term-owner): Main; delegated writers own only their serial packet [scope](../../../../glossary.md#term-scope).

Historical [Ready](../../../../glossary.md#term-ready) verification (2026-10-22): Model0034's final Class C APPROVE (zero blockers), main
acceptance and closed dependencies verified; HEAD/origin/main matched the base, conflicts inactive,
sole writer confirmed. Ready authorized clarification; code awaited authority review/main acceptance,
subsequently satisfied. Accepted dirty work and the three protected files were preserved.

## Files and symbols

- New `model/tensor/ScanBody.java`, `Scan.java`, `ScanResult.java`: source formals/roots, static
  `Scan.apply(initialCarry, sequences, invariants, body, reverse)` and [canonical](../../../../glossary.md#term-canonical-form) grouped result lists.
- New `model/operation/scan/ScanKind.java`, `ScanAttrs.java`: fixed-count meaning and scalar header
  (T, carry/sequence/invariant/emission/residual counts, reverse; axis zero).<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [carry glossary](../../../../glossary.md#term-carry); [invariant glossary](../../../../glossary.md#term-invariant); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand); [axis glossary](../../../../glossary.md#term-axis))</span>
- New `model/graph/CompiledScanBody.java`: existing CompiledGraphModel, ordered role-result ValueId
  bindings and demanded outer-port indices; no second graph representation.<span data-glossary-navigation="true"> ([CompiledGraphModel glossary](../../../../glossary.md#compiled-graph--compiledgraphmodel); [ValueId glossary](../../../../glossary.md#logical-valueid-versus-physical-memory-slot); [bindings glossary](../../../../glossary.md#term-binding); [graph glossary](../../../../glossary.md#graph))</span>
- Existing `model/tensor/TensorProducer.java` and `TensorFactory.java`: `Optional<ScanBody>` attachment
  and validated atomic construction; retain empty-attachment old forms. Existing<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>
  `model/graph/CompiledNode.java`: add `Optional<CompiledScanBody>` and retain four-[argument](../../../../glossary.md#term-argument) constructor.
- Existing 46 internal [Tensor](../../../../glossary.md#tensor) expression helpers (including `TensorFloatingClassifications`):
  explicit synchronous descriptor-check seams reuse ordinary guards without allocating Tensor IDs
  or constructing validation expressions. Ordinary null-sink paths preserve existing behavior;
  corresponding exact internal-method inventory tests account for these non-public seams.<span data-glossary-navigation="true"> ([synchronous glossary](../../../../glossary.md#term-synchronous); [descriptor glossary](../../../../glossary.md#term-descriptor); [validation glossary](../../../../glossary.md#term-validation))</span>
- Tests under matching `modules/model/src/test/java/.../model/{tensor,operation/scan,graph}`:
  ScanBodyTest, ScanExpressionTest, ScanResultTest, ScanSemanticsTest, CompiledScanBodyTest;
  update TensorProducerTest/CompiledNodeTest and focused compatibility coverage.
- Documentation: affected [Javadoc](../../../../glossary.md#term-javadoc), `docs/api/tensor-api.md`, focused general-scan explanation and
  targeted glossary entries; main owns master/program/roadmap integration, including delegated closure.

## Acceptance criteria

- Immutable defensive ordered snapshots; identity-distinct declared storage/provenance-free static
  canonical formals, with unused formals allowed. Actual positional repeats remain intact and actuals
  are disjoint from formals. Carry/invariant descriptors match actual shape/type/flags; step descriptors
  remove sequence axis zero and use canonical geometry. At least one carry and sequence, common static
  T≥0, matching next-carry arity/shape/type, zero or more invariants/emissions and checked counts/shapes.<span data-glossary-navigation="true"> ([Immutable glossary](../../../../glossary.md#term-immutable); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance); [canonical glossary](../../../../glossary.md#term-canonical-form); [Carry glossary](../../../../glossary.md#term-carry); [invariant glossary](../../../../glossary.md#term-invariant); [descriptors glossary](../../../../glossary.md#term-descriptor); [shape glossary](../../../../glossary.md#shape); [axis glossary](../../../../glossary.md#term-axis); [arity glossary](../../../../glossary.md#term-arity))</span>
- Validate the WHOLE contained body, including canonical siblings, against existing known pure Model
  semantics, declared leaves and static descriptors. Reject effects/RNG/mutation, free/host-backed
  leaves and nested/recursive scans. Do not narrow source construction to CPU's future supported subset.
  Local validation belongs here; collective cross-template sharing/outer producer escape detection,
  scoped capture, inference, demand and AD remain future Compiler responsibilities, with no global registry.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [host glossary](../../../../glossary.md#term-host); [CPU glossary](../../../../glossary.md#term-cpu); [validation glossary](../../../../glossary.md#term-validation); [producer glossary](../../../../glossary.md#term-producer); [capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [AD glossary](../../../../glossary.md#autograd); [Compiler glossary](../../../../glossary.md#term-compiler); [registry glossary](../../../../glossary.md#term-registry))</span>
- Reuse one immutable body/producers across distinct scan occurrences. Never unroll by T or retain a
  callback. Derive identity-unique latent residual inventory deterministically by producer postorder/
  ascending sibling slots, including unused siblings and previous carry; exclude step/invariant formals.
  Previous-carry residual meaning is pre-step; emission/history indices are original-time aligned.<span data-glossary-navigation="true"> ([callback glossary](../../../../glossary.md#term-callback); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand))</span>
- Original producer reserves every distinct canonical wrapper in final-carry/emission/residual order
  before public return. Duplicate role bindings do not merge outer wrappers. ScanResult groups exact
  canonical immutable lists. No storage/history allocation, producer replacement or postcapture extension.<span data-glossary-navigation="true"> ([bindings glossary](../../../../glossary.md#term-binding))</span>
- Whole-body and descriptor/type/shape/arity/T/role failures precede output ID allocation, including
  direct attachment construction paths. Preserve existing identifier-exhaustion/no-partial-return behavior.
  T=0 still creates the occurrence/empty stacked descriptors without executing or aliasing away ports.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [aliasing glossary](../../../../glossary.md#term-aliasing))</span>
- Pin positive construction cases outside CPU's first domain (FLOAT16 TANH and F32 RELU), and negative
  unknown/custom/effectful/free/host-backed/nested cases, including an invalid unused canonical sibling.<span data-glossary-navigation="true"> ([FLOAT16 glossary](../../../../glossary.md#term-float16); [TANH glossary](../../../../glossary.md#term-tanh); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>
- Apply approved `G` eligibility to all outer groups. Test all-false, each actual role true, unused
  gradient inputs, T=0, nondifferentiable outputs and the critical <span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient))</span>`emit=h` [regression](../../../../glossary.md#term-regression) above.
- CompiledScanBody validates immutable local graph/binding/demand structure: known local result IDs,
  unique graph outputs with repeatable ordered role bindings, valid ordered demand indices and no
  nested scan bodies. Body graph inputs retain declared formal positions including unused ones.
  CompiledNode validates attachment/kind/header cardinality compatibility; ordinary/fixed nodes remain
  empty-attached. Do not compare numeric IDs across scopes or implement Compiler descriptor inference.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [cardinality glossary](../../../../glossary.md#term-cardinality); [nodes glossary](../../../../glossary.md#node))</span>
- Compatibility tests intentionally change CompiledNode's exact four-component/one-constructor assertion
  to five components plus the compatible old constructor; cover attachment equality/hash/text/null rules.
  Update TensorProducerTest's exact field/accessor assertions only for the approved attachment.
  Existing ordinary/multi-output/fixed scan provenance and graph-record validation remain intact.<span data-glossary-navigation="true"> ([hash glossary](../../../../glossary.md#term-hash))</span>
- Six-type inventory only; focused tests and final Javadoc pass; independent reviews/main checkpoint
  recorded. Explain source/IR-only status without claiming generic scan execution, training or persistence.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [persistence glossary](../../../../glossary.md#term-persistence))</span>

## [Validation](../../../../glossary.md#term-validation)

Focused new/compatibility command (passed):

```bash
./gradlew :modules:model:test --tests '*ScanBodyTest' --tests '*ScanExpressionTest' --tests '*ScanResultTest' --tests '*ScanSemanticsTest' --tests '*CompiledScanBodyTest' --tests '*TensorProducerTest' --tests '*TensorProvenanceTest' --tests '*TensorFactoryTest' --tests '*CompiledNodeTest' --tests '*CompiledGraphModelTest' --tests '*RecurrentScanExpressionTest' --tests '*RecurrentScanSemanticsTest'
```

Final affected-[module](../../../../glossary.md#term-module) command (passed; reviewer regenerated [Javadoc](../../../../glossary.md#term-javadoc) after comment-only edits):

```bash
./gradlew :modules:model:test :modules:model:javadoc
```

Main's CompiledNode compatibility [checkpoint](../../../../glossary.md#term-checkpoint) (all passed):

```bash
./gradlew :modules:planning:compileTestJava :modules:compiler:compileTestJava :modules:prepare:compileTestJava :backends:cpu:compileTestJava :extensions:checkpoint:compileTestJava
./gradlew :testing:architecture-tests:test --tests '*RuntimeDependencyAndHotPathContractTest' --tests '*PlanningDependencyContractTest' --tests '*EngineCompositionContractTest' --tests '*CpuDependencyContractTest' --tests '*CheckpointDependencyContractTest'
./gradlew :testing:integration-tests:test --tests '*EngineCpuRecurrentScanIntegrationTest'
git diff --check
```

Validation reused the existing [Gradle](../../../../glossary.md#term-gradle)/[JDK](../../../../glossary.md#term-java-development-kit-jdk)/[cache](../../../../glossary.md#term-cache) setup. Successful tests were not repeated by review;
local links/anchors and new-file whitespace were checked. No root/native/[conformance](../../../../glossary.md#term-conformance)/performance
campaign here; the program-wide root gate belongs to the later low-level join.

## Documentation and review impact

Independent authority/code/docs reviews finalized affected contracts, examples, glossary and [Javadoc](../../../../glossary.md#term-javadoc)
using general/architecture/decision/[API](../../../../glossary.md#term-application-programming-interface-api)/Javadoc/example profiles. Helper/factory Javadoc remains accurate.
This [planning](../../../../glossary.md#term-planning)-only closure uses general/planning profiles and documentation rules; no authority/API/
glossary/Java changes. API example [validation](../../../../glossary.md#term-validation) used
`jshell --execution local --class-path modules/model/build/classes/java/main --feedback concise -`.

## Result

- Main accepted this bounded foundation on 2026-10-22 at the recorded base plus accepted uncommitted
  Model0034/0035; conflicts serialized, no active writer, protected files/index/HEAD unchanged.
- Source/CPU admission separation and conservative G were independently approved by Helmholtz the 2nd
  and accepted by main before code. Final Parfit the 2nd code/docs verdict: APPROVE, zero blockers.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [admission glossary](../../../../glossary.md#term-admission))</span>
- Implemented six new production types plus 49 existing production files (46 guard helpers), and
  55 test files (49 inventory/compatibility updates plus six new suites): 110 Model files total.<span data-glossary-navigation="true"> ([guard glossary](../../../../glossary.md#term-guard); [Model glossary](../../../../glossary.md#term-model))</span>
- Optional source/compiled attachments, canonical ports/formals/roles, checked static descriptors,
  G/T=0 and metadata-only histories are implemented; no extra type/module/dependency or Tensor-ID
  allocation in reused guards. ID/exhaustion, late-storage and transpose regressions are covered;
  validation/diagnostic defects are fixed; 111 normalized ordinary-method comparisons had zero mismatches.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [descriptors glossary](../../../../glossary.md#term-descriptor); [metadata glossary](../../../../glossary.md#term-metadata); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [Tensor glossary](../../../../glossary.md#tensor); [transpose glossary](../../../../glossary.md#term-transpose); [validation glossary](../../../../glossary.md#term-validation))</span>
- Prescribed focused command: PASS, 116 tests/15 suites. Final Model test/Javadoc command: PASS,
  1,165 tests/146 suites, zero failures/errors/skips or Javadoc warnings.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Reviewer finalized API/examples/glossary/status docs and eight Javadocs without executable changes;
  Javadoc-only rerun PASS, zero warnings; compiled/run jshell API example PASS; 502 local links/anchors,
  whitespace and diff checks PASS. Successful module-test evidence was reused.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [run glossary](../../../../glossary.md#term-run))</span>
- Main's exact Validation commands all PASS: five-module compileTestJava (5s, 16 tasks: 10 executed,
  six up-to-date); architecture (five suites/seven tests; 2s, 27 tasks: 12/15); fixed-scan integration
  (one suite/five tests; 3s, 42 tasks: 3/39), zero failures/errors/skips. Existing unchecked javac note
  and FFM native-access/incubator-vector warnings only.<span data-glossary-navigation="true"> ([FFM glossary](../../../../glossary.md#term-ffm); [vector glossary](../../../../glossary.md#term-vector))</span>
- Main's final 110-file hash comparison differed only in eight reviewer-declared Javadoc files;
  protected hashes/index/HEAD and <span data-glossary-navigation="true"> ([hash glossary](../../../../glossary.md#term-hash))</span>`git diff --check` passed. [Planning](../../../../glossary.md#term-planning) closure synchronizes seven files.
- Limits: source/IR only, not whole generic scan; no capture/query/AD/Prepare/execution/training/
  persistence grant or root/native/performance campaign. Later low-level join owns the root gate.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [capture glossary](../../../../glossary.md#term-capture-meanings); [AD glossary](../../../../glossary.md#autograd); [Prepare glossary](../../../../glossary.md#prepare); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- Model/authority frontiers None. Sole next slice: Planning body query, Draft/not yet briefed;
  Compiler AD/capture → Prepare → CPU and later gates remain Draft/deferred, with fixed/static routes retained.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- Unresolved issues: none within this foundation; no successor launched. Status: Complete.
- Closure: 582 local links/anchors, whitespace/status and diff checks PASS; only seven planning files changed.

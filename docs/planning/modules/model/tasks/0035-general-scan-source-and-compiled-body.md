# Task Model0035: General scan source and compiled body

## Status

Complete — bounded Model source/intermediate-representation (IR) foundation accepted by main on
2026-10-22 after required reviews and the named compatibility checkpoint; no successor launch.

## Change class

Class C — public source-body API, canonical producer ownership and a scoped compiled-node boundary.

## Goal

Deliver an independently reviewable Model foundation: reusable source bodies, canonical scan
results/residual ports and structurally validated compiled-body attachments. Construction describes
meaning; it does not capture, differentiate, prepare or execute a scan.

## Scope and non-goals

- Add only the six production types in Files and symbols; extend existing producer/node construction
  without removing old constructor forms. Include focused tests, Javadoc and targeted documentation.
- Enact the two user-approved clarifications below in the root, owning recurrent contract,
  relevant explanation and ADR0045; no other authority changes are authorized.
- No Compiler capture/inference/AD, collective source escape checking, demand analysis, Planning
  query, Prepare projection, backend execution, NN/padding or persistence implementation.
- No new module/dependency, public runner, global ownership registry or extra helper abstraction;
  no custom/unknown semantics, numerical policy, native/performance campaign or execution claim.

## Contracts

Start with [root scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts).
Read these exact headings, not unrelated contracts or completed task histories:

- Root [Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence),
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules), and
  [Documentation authority and architecture updates](../../../../../ARCHITECTURE.md#documentation-authority-and-architecture-updates).
- Foundational [modules/model](../../../../architecture/contracts/foundational-modules.md#modulesmodel):
  producer ownership, immutable records and scalar-only attributes.
- Recurrent [General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan),
  including **Source body, attachments and scoped identities**, **Canonical residual ports and demand**,
  **Reverse scan and exact primal bindings**, **Traversal, empty domains and declared qualification**,
  and **Qualification, performance and migration gates**: sole complete scan semantic owner.

If a new contract conflict appears, stop and report it. The following choices were approved and
independently accepted in authority before code; this planning packet is not their semantic owner.

## Accepted pre-code clarifications

User approved both choices on 2026-10-22; authority review/main acceptance preceded code:

1. Existing valid PURE Model algebra/dtypes are independent of CPU qualification (FLOAT16 TANH and
   F32 RELU construct). Existing source exclusions still reject; no new semantics or AD/backend grant.
2. `G` ORs every actual carry/sequence/invariant `requiresGrad` flag. Differentiable outer ports get G;
   nondifferentiable ports get false; formals retain actual flags. Unused inputs/T=0 are included.
   Regression: `h0=false`, `x=true`, `next=h+xStep`, `emit=h` preserves temporal outer eligibility,
   not a nonzero-dependency or AD/backend guarantee. Full acceptance remains below.

## Dependencies and integration

- Depends on: Model0034 Complete, including independent authority APPROVE and main acceptance (2026-10-21).
- Conflicts with: scan source/IR/roles/residuals, shared authority/docs, and downstream query/AD/capture/
  projection/backend/NN/persistence scan changes; serialize all overlapping work.
- Parallel group: None.
- Common base revision: `6da23a0b42f015979a88f5e84416799f5ada3f40` (HEAD=origin/main), plus accepted
  uncommitted Model0034 authority/planning work; do not treat dirty accepted work as a merged commit.
- Integration order: clarification writer → independent authority review/main acceptance → Model code
  → independent code/docs review → main foundation checkpoint; downstream slices remain Draft.
- Integration validation: named **Model0035 CompiledNode compatibility checkpoint**, commands below;
  program-wide native root `test javadoc` remains at the later low-level join.
- Shared-document integration owner: Main; delegated writers own only their serial packet scope.

Historical Ready verification (2026-10-22): Model0034's final Class C APPROVE (zero blockers), main
acceptance and closed dependencies verified; HEAD/origin/main matched the base, conflicts inactive,
sole writer confirmed. Ready authorized clarification; code awaited authority review/main acceptance,
subsequently satisfied. Accepted dirty work and the three protected files were preserved.

## Files and symbols

- New `model/tensor/ScanBody.java`, `Scan.java`, `ScanResult.java`: source formals/roots, static
  `Scan.apply(initialCarry, sequences, invariants, body, reverse)` and canonical grouped result lists.
- New `model/operation/scan/ScanKind.java`, `ScanAttrs.java`: fixed-count meaning and scalar header
  (T, carry/sequence/invariant/emission/residual counts, reverse; axis zero).
- New `model/graph/CompiledScanBody.java`: existing CompiledGraphModel, ordered role-result ValueId
  bindings and demanded outer-port indices; no second graph representation.
- Existing `model/tensor/TensorProducer.java` and `TensorFactory.java`: `Optional<ScanBody>` attachment
  and validated atomic construction; retain empty-attachment old forms. Existing
  `model/graph/CompiledNode.java`: add `Optional<CompiledScanBody>` and retain four-argument constructor.
- Existing 46 internal Tensor expression helpers (including `TensorFloatingClassifications`):
  explicit synchronous descriptor-check seams reuse ordinary guards without allocating Tensor IDs
  or constructing validation expressions. Ordinary null-sink paths preserve existing behavior;
  corresponding exact internal-method inventory tests account for these non-public seams.
- Tests under matching `modules/model/src/test/java/.../model/{tensor,operation/scan,graph}`:
  ScanBodyTest, ScanExpressionTest, ScanResultTest, ScanSemanticsTest, CompiledScanBodyTest;
  update TensorProducerTest/CompiledNodeTest and focused compatibility coverage.
- Documentation: affected Javadoc, `docs/api/tensor-api.md`, focused general-scan explanation and
  targeted glossary entries; main owns master/program/roadmap integration, including delegated closure.

## Acceptance criteria

- Immutable defensive ordered snapshots; identity-distinct declared storage/provenance-free static
  canonical formals, with unused formals allowed. Actual positional repeats remain intact and actuals
  are disjoint from formals. Carry/invariant descriptors match actual shape/type/flags; step descriptors
  remove sequence axis zero and use canonical geometry. At least one carry and sequence, common static
  T≥0, matching next-carry arity/shape/type, zero or more invariants/emissions and checked counts/shapes.
- Validate the WHOLE contained body, including canonical siblings, against existing known pure Model
  semantics, declared leaves and static descriptors. Reject effects/RNG/mutation, free/host-backed
  leaves and nested/recursive scans. Do not narrow source construction to CPU's future supported subset.
  Local validation belongs here; collective cross-template sharing/outer producer escape detection,
  scoped capture, inference, demand and AD remain future Compiler responsibilities, with no global registry.
- Reuse one immutable body/producers across distinct scan occurrences. Never unroll by T or retain a
  callback. Derive identity-unique latent residual inventory deterministically by producer postorder/
  ascending sibling slots, including unused siblings and previous carry; exclude step/invariant formals.
  Previous-carry residual meaning is pre-step; emission/history indices are original-time aligned.
- Original producer reserves every distinct canonical wrapper in final-carry/emission/residual order
  before public return. Duplicate role bindings do not merge outer wrappers. ScanResult groups exact
  canonical immutable lists. No storage/history allocation, producer replacement or postcapture extension.
- Whole-body and descriptor/type/shape/arity/T/role failures precede output ID allocation, including
  direct attachment construction paths. Preserve existing identifier-exhaustion/no-partial-return behavior.
  T=0 still creates the occurrence/empty stacked descriptors without executing or aliasing away ports.
- Pin positive construction cases outside CPU's first domain (FLOAT16 TANH and F32 RELU), and negative
  unknown/custom/effectful/free/host-backed/nested cases, including an invalid unused canonical sibling.
- Apply approved `G` eligibility to all outer groups. Test all-false, each actual role true, unused
  gradient inputs, T=0, nondifferentiable outputs and the critical `emit=h` regression above.
- CompiledScanBody validates immutable local graph/binding/demand structure: known local result IDs,
  unique graph outputs with repeatable ordered role bindings, valid ordered demand indices and no
  nested scan bodies. Body graph inputs retain declared formal positions including unused ones.
  CompiledNode validates attachment/kind/header cardinality compatibility; ordinary/fixed nodes remain
  empty-attached. Do not compare numeric IDs across scopes or implement Compiler descriptor inference.
- Compatibility tests intentionally change CompiledNode's exact four-component/one-constructor assertion
  to five components plus the compatible old constructor; cover attachment equality/hash/text/null rules.
  Update TensorProducerTest's exact field/accessor assertions only for the approved attachment.
  Existing ordinary/multi-output/fixed scan provenance and graph-record validation remain intact.
- Six-type inventory only; focused tests and final Javadoc pass; independent reviews/main checkpoint
  recorded. Explain source/IR-only status without claiming generic scan execution, training or persistence.

## Validation

Focused new/compatibility command (passed):

```bash
./gradlew :modules:model:test --tests '*ScanBodyTest' --tests '*ScanExpressionTest' --tests '*ScanResultTest' --tests '*ScanSemanticsTest' --tests '*CompiledScanBodyTest' --tests '*TensorProducerTest' --tests '*TensorProvenanceTest' --tests '*TensorFactoryTest' --tests '*CompiledNodeTest' --tests '*CompiledGraphModelTest' --tests '*RecurrentScanExpressionTest' --tests '*RecurrentScanSemanticsTest'
```

Final affected-module command (passed; reviewer regenerated Javadoc after comment-only edits):

```bash
./gradlew :modules:model:test :modules:model:javadoc
```

Main's CompiledNode compatibility checkpoint (all passed):

```bash
./gradlew :modules:planning:compileTestJava :modules:compiler:compileTestJava :modules:prepare:compileTestJava :backends:cpu:compileTestJava :extensions:checkpoint:compileTestJava
./gradlew :testing:architecture-tests:test --tests '*RuntimeDependencyAndHotPathContractTest' --tests '*PlanningDependencyContractTest' --tests '*EngineCompositionContractTest' --tests '*CpuDependencyContractTest' --tests '*CheckpointDependencyContractTest'
./gradlew :testing:integration-tests:test --tests '*EngineCpuRecurrentScanIntegrationTest'
git diff --check
```

Validation reused the existing Gradle/JDK/cache setup. Successful tests were not repeated by review;
local links/anchors and new-file whitespace were checked. No root/native/conformance/performance
campaign here; the program-wide root gate belongs to the later low-level join.

## Documentation and review impact

Independent authority/code/docs reviews finalized affected contracts, examples, glossary and Javadoc
using general/architecture/decision/API/Javadoc/example profiles. Helper/factory Javadoc remains accurate.
This planning-only closure uses general/planning profiles and documentation rules; no authority/API/
glossary/Java changes. API example validation used
`jshell --execution local --class-path modules/model/build/classes/java/main --feedback concise -`.

## Result

- Main accepted this bounded foundation on 2026-10-22 at the recorded base plus accepted uncommitted
  Model0034/0035; conflicts serialized, no active writer, protected files/index/HEAD unchanged.
- Source/CPU admission separation and conservative G were independently approved by Helmholtz the 2nd
  and accepted by main before code. Final Parfit the 2nd code/docs verdict: APPROVE, zero blockers.
- Implemented six new production types plus 49 existing production files (46 guard helpers), and
  55 test files (49 inventory/compatibility updates plus six new suites): 110 Model files total.
- Optional source/compiled attachments, canonical ports/formals/roles, checked static descriptors,
  G/T=0 and metadata-only histories are implemented; no extra type/module/dependency or Tensor-ID
  allocation in reused guards. ID/exhaustion, late-storage and transpose regressions are covered;
  validation/diagnostic defects are fixed; 111 normalized ordinary-method comparisons had zero mismatches.
- Prescribed focused command: PASS, 116 tests/15 suites. Final Model test/Javadoc command: PASS,
  1,165 tests/146 suites, zero failures/errors/skips or Javadoc warnings.
- Reviewer finalized API/examples/glossary/status docs and eight Javadocs without executable changes;
  Javadoc-only rerun PASS, zero warnings; compiled/run jshell API example PASS; 502 local links/anchors,
  whitespace and diff checks PASS. Successful module-test evidence was reused.
- Main's exact Validation commands all PASS: five-module compileTestJava (5s, 16 tasks: 10 executed,
  six up-to-date); architecture (five suites/seven tests; 2s, 27 tasks: 12/15); fixed-scan integration
  (one suite/five tests; 3s, 42 tasks: 3/39), zero failures/errors/skips. Existing unchecked javac note
  and FFM native-access/incubator-vector warnings only.
- Main's final 110-file hash comparison differed only in eight reviewer-declared Javadoc files;
  protected hashes/index/HEAD and `git diff --check` passed. Planning closure synchronizes seven files.
- Limits: source/IR only, not whole generic scan; no capture/query/AD/Prepare/execution/training/
  persistence grant or root/native/performance campaign. Later low-level join owns the root gate.
- Model/authority frontiers None. Sole next slice: Planning body query, Draft/not yet briefed;
  Compiler AD/capture → Prepare → CPU and later gates remain Draft/deferred, with fixed/static routes retained.
- Unresolved issues: none within this foundation; no successor launched. Status: Complete.
- Closure: 582 local links/anchors, whitespace/status and diff checks PASS; only seven planning files changed.

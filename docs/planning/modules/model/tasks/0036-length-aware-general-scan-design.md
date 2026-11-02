# Task Model0036: Length-aware general scan and first-order AD proposal

## Status

Complete — proposal delivery only, after independent final Class C APPROVE and MAIN acceptance
on 2026-10-24. No semantic/API/wire approval, active planning/source/authority frontier or successor.

## Change class

Class C design — proposed row binding, AD reductions, saved values, generated-loop skipping and
persistence cross high-risk boundaries. This task writes non-authoritative planning only.

## Goal

Deliver a compact, independently reviewed [proposal](../../../length-aware-general-scan.md) for
runtime INT64 `[B]` valid lengths on existing Scan and first-order reverse, using existing owners.
Choose a bounded mechanism and expose every genuinely new API/metadata concept as PROPOSED.

## Scope

F32/F64 CPU target, static T/B and L in [0,T]; branch before padding loads/arithmetic; exact empty-row
initial/final state and final cotangent; original-time zero padding; active-prefix reverse;
shared invariant gradients without mandatory B×parameter state; cold copy/rebind and additive wire.

Exclusive write scope (all under `docs/planning/`): `length-aware-general-scan.md`, this brief,
`modules/model/master-plan.md`, `general-differentiable-scan.md`,
`extensions/nn/tasks/0032-fluent-rnn-general-scan-migration.md`, `extensions/nn/master-plan.md`,
`roadmap.md`. Other accepted dirty work belongs to the user and must remain byte-preserved.

## Non-goals

No source/test/Javadoc/build/contract/ADR changes, implementation, `.rnn` migration, new RNN API or
execution owner, higher-order waiver, packing/sorting, low types, Metal, if/while/regions,
runtime tape, callbacks/interpreter/per-step allocation, commit/push or successor launch. No GSD.
No Java/Gradle/native/performance runs. Do not turn this task into a speculative module cascade.

## Contracts

Read [root scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts), then:

- Root: [Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence),
  [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle),
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules),
  [Documentation authority and architecture updates](../../../../../ARCHITECTURE.md#documentation-authority-and-architecture-updates),
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Recurrent owner: [General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)
  and every subheading: `Source body, attachments and scoped identities`, `Canonical residual
  ports and demand`, `Reverse scan and exact primal bindings`, `Traversal, empty domains and
  declared qualification`, `Qualification, performance and migration gates`, including its final
  per-row-length gate. This proposal requests a future decision, not a current interpretation.
- Extensions/training: [Owned tensor Models](../../../../architecture/contracts/extensions-training.md#owned-tensor-models)
  and [Canonical graph-model version 4](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-4).
- Compiler: [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation),
  [Compile lifecycle](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle),
  [Bounded general-scan cold reconstruction](../../../../architecture/contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction).
- Boundaries: [modules/prepare](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare),
  [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle),
  [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes).

Planning guide/roadmap/active Model, NN and general-scan program reading is required for this
planning context. Apply documentation rules, general/planning style, decision-record rationale and
example profiles. Search relevant glossary terms only. Stop on a conflict that prevents a coherent
proposal; do not invent accepted architecture. Current contracts remain unchanged.

## Dependencies and integration

- Depends on: accepted Model0034/0035, Compiler0011/0012, Planning0008, Prepare0010,
  CPU0030–0037, CP0012/0013, NN0030/0031 and their recorded joins; current source/contract baseline.
  Their recorded Complete status is reused, not independently rerun here.
- Conflicts with: scan binding/AD/history/identity/wire decisions, fluent migration and all seven
  shared planning documents; sole documentation writer, no concurrent code writer.
- Parallel group: None. Planner closed; independent reviewer holds only the exclusive seven-path
  factual review/status closure grant after final APPROVE.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`, plus 295
  accepted dirty paths; 2423 Git-listed baseline files, empty index. Preservation baseline/archive:
  `/private/tmp/synaptik-length-scan-plan.Q3c8m1/baseline.json` and `accepted.tar` beside it.
- Integration order: sole-writer proposal/index draft → freeze → independent targeted review →
  corrections if needed → main acceptance of proposal delivery only. No next-task execution.
- Integration validation: scoped local-link/anchor, brief size, whitespace and baseline/hash
  preservation checks; no executable checkpoint in this design-only context.
- Shared-document integration owner: MAIN; the planner's delegated edits are closed. The
  independent reviewer owns only final factual closure in the seven paths, not semantic approval.

Planning-only Ready verification, 2026-10-24: latest explicit user approval authorizes this immediate
proposal and independent review only. Model0036 number/path were free; HEAD/origin and empty index
matched. Main verified archive extraction against all 2423 baseline hashes, zero mismatch, and no
other writer. Dependencies are the accepted working-tree baseline, not newly merged commits.
This satisfies the planning-only frontier; it does not satisfy an implementation Ready gate.

## Files and symbols

Read-only evidence uses package prefix `io.github.pho001.synaptik`:

- `modules/model`: `model.tensor.Scan.apply/validateBindings`, `ScanBody.retainedValues`,
  `model.operation.scan.ScanAttrs.inputCount`, `model.graph.CompiledScanBody`.
- `modules/compiler`: `ScanBodyInference.capture/infer/refreshDemand`,
  `ScanGradientRules.preflight/apply`, `FirstOrderAutograd.expandBody`,
  `LinearAlgebraGradientRules.apply`, the primal-bound `ReductionGradientRules.apply` overload,
  `SourceGraphReconstruction.preflight/reconstructScan`, `AutogradPreflight` scan-order rejection.
- `modules/planning`: `OperationCapabilityQuery`; `modules/prepare`: `PrepareContext`,
  `GraphPreparation` demand validation. Existing scoped body attachment remains the carrier.
- `modules/backend/cpu`: `CpuGeneralScanLowering`, `CpuGeneralScanIr`, `CpuGeneralScanEmitter`,
  typed reduction/MATMUL emitters, `CpuGeneralScanPreparedExecutable`; fixed
  `CpuRecurrentScanPreparedExecutable.executeBound` supplies preflight-order evidence only.
- `modules/checkpoint`: `GraphArtifactCodec.writeNodes/attributes/Reader`, graph copy/rebind;
  `extensions/nn`: `OwnedModelSource`, `ModelBuilder.RnnStep.forward`, current Model workflow.
- Existing focused tests to adapt are named in the proposal's
  [future acceptance](../../../length-aware-general-scan.md#focused-future-acceptance-and-present-evidence-limits),
  not new tests or claimed executable evidence in this task.

## Acceptance criteria

- Preferred whole-batch bounded eligibility is compared with row-local binding; real ordinary
  linear/RNN MATMUL, broadcast and SUM survive without a fixed adjoint or axis-proof framework.
- Proposed length role, rowCarryCount and per-sequence optional row axes are explicit. Shared
  W-transpose histories and equal-size row seeds remain distinguishable through pruning/capture/
  query/Prepare/cold/v5; no shape-based inference or temporary AD-only metadata shortcut.
- Temporal order, active SUM/MATMUL contributor domain, once-per-active-time shared ADD rounding,
  O(P) accumulator storage and inactive row identity are concrete; no B×parameter requirement.
- First-order AD requires rowCarryCount==carryCount AND every original sequenceRowAxes entry
  present0 (ROW(0)), including unused/non-gradient positions. Original shared-carry or
  SHARED/ROW(1)/other nonzero-axis sequence AD rejects before derivative allocation; future
  negative tests cover each. Broader forward validity, generated reverse mixed-role sequence
  inputs/shared sums, the ordinary TBF RNN target and old all-valid AD remain unchanged.
- Exact saved/previous primals, shared/row history activity and positive-zero padding are specified;
  no padded seed/history reads, activation replay or unguarded hoisted body work.
- Runtime lengths fully preflight after producers and before this scan's first write; no upstream
  rollback claim. O(body) code/prepared objects, dense output/live-history costs remain truthful.
- Old all-valid behavior/constructors/identities and v1–v4 bytes remain separate; new public record
  fields/overload and selective v5/new opcode are proposed, not silently accepted or schema-free.
- Current fluent/fixed APIs and NN0032 second-order compile blocker, CONCAT-copy and mixed-engine
  gates remain. Scan lengths imply neither model-wide masking nor a masked loss denominator.
- One compact immediate brief, coarse Draft future phases, concise synchronized indexes, no
  normative edits, independent review and main acceptance before proposal-delivery Complete.

## Validation

Design-only worker validation: scoped `git diff --check`, new-file whitespace and line/byte counts;
local Markdown target/fragment validation; seven-field/status/DAG/example review; compare every
non-allowlisted baseline hash and Git state with the supplied preservation baseline.
Reuse source/contract evidence; proposed APIs/examples are not compiled or execution-tested.

Main integration: accept the independent review and preserved-tree report before marking this
proposal task Complete. No Java, Gradle, native, performance or repository test gate is authorized.

## Documentation and review impact

Primary planning profile; decision-record option/rationale and conceptual-example requirements
apply to the proposal. No current glossary meaning, API Javadoc or authority is changed.
Independent targeted Class C review is mandatory; it reviews the frozen exact seven-file draft
and current source/contracts, with no competing writer or duplicate executable validation.

## Result

Prepared the proposal, this brief and five concise planning/index updates; no other paths changed.
Scoped `git diff --check --` with the seven Scope paths passed. A read-only inline Ruby checker
validated local Markdown targets/fragments, final newlines and trailing whitespace; final counts
are recorded in the handoff. `wc -lc` confirms this brief is below 200 lines/15 KB.
Read-only Ruby SHA-256 comparison against the supplied baseline found only the five allowed
existing documents changed; all other 2418 baseline files matched. The other two allowed documents
are new. `git rev-parse HEAD origin/main` retained the declared base; staged index remained empty.
No code/Javadoc/contract/build edits or Java/Gradle/native/generated/performance validation occurred.
Targeted glossary review found no current meaning change; proposed row roles are defined locally.
Independent final Class C review: APPROVE (2026-10-24), no unresolved P1/P2. The initial P2 on
non-ROW(0) original sequence gradients was corrected in the proposal, future negative-test wording
and this brief; targeted re-review confirms rejection before derivative allocation even for
unused/non-gradient positions, without restricting generated reverse mixed-role inputs.
The reviewer reused the full seven-document/source/contract review and successful link/math
evidence; MAIN's preserved-tree report covers 2425 files, seven allowed deltas and no other changes.
MAIN's conditional acceptance is satisfied by final APPROVE, for proposal delivery only.
Final factual closure changed only the seven Scope documents; scoped `git diff --check --`,
read-only inline Ruby local-link/fragment/newline/whitespace checks and `wc -lc` passed afterward.
No executable reruns or semantic rewriting. All new semantics/API/v5 remain PROPOSED; current
Javadoc/glossary meanings and NN0032's migration/second-order gates are unchanged.

Status: Complete
Proposal delivery only; no active planning/source/authority frontier, semantic/API/wire approval
or successor. No delivery follow-up remains; any future decision needs separate authorization.

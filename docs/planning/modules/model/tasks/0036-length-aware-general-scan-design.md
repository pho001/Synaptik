# Task Model0036: Length-aware [general scan](../../../../glossary.md#general-differentiable-scan) and first-order [AD](../../../../glossary.md#autograd) proposal

**Editorial reading key (current context, not a status update).**

This record closes proposal delivery only; its length roles, AD rules and wire changes remain
historical proposals, not approvals. For later descriptions, follow [current length-aware Scan
boundaries](../../../../reference/tensor/general-scan.md#length-aware-source-construction-and-qualified-execution),
not this task’s status. A caller binds actual Tensors to body formals: carry passes state between
steps, sequences supply time slices, and invariants supply unchanged inputs. Lengths are an extra
actual, not a body formal. Model defines meaning; Compiler constructs derivatives, Prepare validates
bindings/demand, and execution providers realize supported routes rather than redefine semantics.
These responsibilities do not confer buffer ownership: storage-free expressions neither execute nor
own result buffers; a leaf has no producing expression, not necessarily stored values.

ROW(axis) declares a proved independent row coordinate; SHARED has none, regardless of equal sizes.
Original length-AD requires every carry and original sequence to be ROW(0), including unused
positions; generated reverse inputs differ. A residual names a retainable internal value; [canonical
results and
demand](../../../../reference/tensor/general-scan.md#canonical-results-residual-metadata-and-eligibility)
explain why a port does not allocate history. Concretely, an empty row preserves initial carry and
final cotangent raw bits, while inactive emissions receive positive zero; skipping padded arithmetic
is not evaluating it then masking. Lengths do not change an outer mean’s denominator.

**Length-guard reading illustration.** For one row with non-negative integer length $\ell\le T$, let
$T$ be allocated time positions and $h_t$ carry before time $t\in\{0,\ldots,T-1\}$. A forward body
runs only when $t<\ell$; otherwise carry stays unchanged and that emission is positive-zero padding.
With $T=3$ and $\ell=1$, only time zero runs. A padded NaN is not first evaluated then multiplied by
zero: inactive work is skipped. The example explains the proposed guard, not implementation
delivery.

## Status

Complete — proposal delivery only, after independent final Class C APPROVE and MAIN acceptance
on 2026-10-24. No semantic/[API](../../../../glossary.md#term-application-programming-interface-api)/wire approval, active [planning](../../../../glossary.md#term-planning)/source/authority frontier or successor.

## Change class

Class C design — proposed row [binding](../../../../glossary.md#term-binding), [AD](../../../../glossary.md#autograd) reductions, saved values, generated-loop skipping and
[persistence](../../../../glossary.md#term-persistence) cross high-risk boundaries. This task writes non-authoritative [planning](../../../../glossary.md#term-planning) only.

## Goal

Deliver a compact, independently reviewed [proposal](../../../length-aware-general-scan.md) for
[runtime](../../../../glossary.md#term-runtime) [INT64](../../../../glossary.md#data-type--datatype) `[B]` valid lengths on existing Scan and first-order reverse, using existing [owners](../../../../glossary.md#term-owner).
Choose a bounded mechanism and expose every genuinely new [API](../../../../glossary.md#term-application-programming-interface-api)/[metadata](../../../../glossary.md#term-metadata) concept as PROPOSED.

## [Scope](../../../../glossary.md#term-scope)

[F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) [CPU](../../../../glossary.md#term-cpu) [target](../../../../glossary.md#term-target), static T/B and L in [0,T]; branch before [padding](../../../../glossary.md#padding) loads/arithmetic; exact empty-row
initial/final state and final [cotangent](../../../../glossary.md#cotangent); original-time zero padding; active-prefix reverse;
shared [invariant](../../../../glossary.md#term-invariant) [gradients](../../../../glossary.md#term-gradient) without mandatory B×[parameter](../../../../glossary.md#term-parameter) state; cold copy/[rebind](../../../../glossary.md#capture-graph-ports-and-reconstruction) and additive wire.

Exclusive write scope (all under `docs/planning/`): `length-aware-general-scan.md`, this brief,
`modules/model/master-plan.md`, `general-differentiable-scan.md`,
`extensions/nn/tasks/0032-fluent-rnn-general-scan-migration.md`, `extensions/nn/master-plan.md`,
`roadmap.md`. Other accepted dirty work belongs to the user and must remain byte-preserved.

## Non-goals

No source/test/[Javadoc](../../../../glossary.md#term-javadoc)/build/contract/[ADR](../../../../glossary.md#term-architecture-decision-record-adr) changes, implementation, `.rnn` migration, new [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn) [API](../../../../glossary.md#term-application-programming-interface-api) or
execution [owner](../../../../glossary.md#term-owner), higher-order waiver, packing/sorting, low types, [Metal](../../../../glossary.md#term-metal), if/while/regions,
[runtime](../../../../glossary.md#term-runtime) tape, callbacks/interpreter/per-step allocation, [commit](../../../../glossary.md#term-commit)/push or successor launch. No GSD.
No Java/[Gradle](../../../../glossary.md#term-gradle)/native/performance runs. Do not turn this task into a speculative [module](../../../../glossary.md#term-module) cascade.

## Contracts

Read [root scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>, then:

- Root: [Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence),
  [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>,
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>,
  [Documentation authority and architecture updates](../../../../../ARCHITECTURE.md#documentation-authority-and-architecture-updates),
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Recurrent [owner](../../../../glossary.md#term-owner): [General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)<span data-glossary-navigation="true"> ([General differentiable scan glossary](../../../../glossary.md#general-differentiable-scan))</span>
  and every subheading: `Source body, attachments and scoped identities`, `Canonical residual
  ports and demand`, `Reverse scan and exact primal bindings`, `Traversal, empty domains and
  declared qualification`, `Qualification, performance and migration gates`, including its final
  per-row-length gate. This proposal requests a future decision, not a current interpretation.
- Extensions/training: [Owned tensor Models](../../../../architecture/contracts/extensions-training.md#owned-tensor-models)<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor))</span>
  and [Canonical graph-model version 4](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-4)<span data-glossary-navigation="true"> ([Canonical glossary](../../../../glossary.md#term-canonical-form); [graph glossary](../../../../glossary.md#graph))</span>.
- [Compiler](../../../../glossary.md#term-compiler): [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([automatic differentiation glossary](../../../../glossary.md#autograd))</span>,
  [Compile lifecycle](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>,
  [Bounded general-scan cold reconstruction](../../../../architecture/contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction)<span data-glossary-navigation="true"> ([cold reconstruction glossary](../../../../glossary.md#term-cold-reconstruction))</span>.
- Boundaries: [modules/prepare](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module); [prepare glossary](../../../../glossary.md#prepare))</span>,
  [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle),
  [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend))</span>.

[Planning](../../../../glossary.md#term-planning) guide/roadmap/active [Model](../../../../glossary.md#term-model), [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) and general-scan program reading is required for this
planning context. Apply documentation rules, general/planning style, decision-record rationale and
example profiles. Search relevant glossary terms only. Stop on a conflict that prevents a coherent
proposal; do not invent accepted architecture. Current contracts remain unchanged.

## Dependencies and integration

- Depends on: accepted Model0034/0035, Compiler0011/0012, Planning0008, Prepare0010,
  CPU0030–0037, CP0012/0013, NN0030/0031 and their recorded joins; current source/contract baseline.
  Their recorded Complete status is reused, not independently rerun here.
- Conflicts with: scan binding/AD/history/identity/wire decisions, fluent migration and all seven
  shared planning documents; sole documentation writer, no concurrent code writer.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [AD glossary](../../../../glossary.md#autograd); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Parallel group: None. Planner closed; independent reviewer holds only the exclusive seven-path
  factual review/status closure grant after final APPROVE.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`, plus 295
  accepted dirty paths; 2423 Git-listed baseline files, empty index. Preservation baseline/archive:
  `/private/tmp/synaptik-length-scan-plan.Q3c8m1/baseline.json` and `accepted.tar` beside it.
- Integration order: sole-writer proposal/index draft → freeze → independent targeted review →
  corrections if needed → main acceptance of proposal delivery only. No next-task execution.
- Integration validation: scoped local-link/anchor, brief size, whitespace and baseline/hash
  preservation checks; no executable checkpoint in this design-only context.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [hash glossary](../../../../glossary.md#term-hash); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Shared-document integration owner: MAIN; the planner's delegated edits are closed. The
  independent reviewer owns only final factual closure in the seven paths, not semantic approval.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>

[Planning](../../../../glossary.md#term-planning)-only [Ready](../../../../glossary.md#term-ready) verification, 2026-10-24: latest explicit user approval authorizes this immediate
proposal and independent review only. Model0036 number/path were free; HEAD/origin and empty index
matched. Main verified archive extraction against all 2423 baseline hashes, zero mismatch, and no
other writer. Dependencies are the accepted working-tree baseline, not newly merged [commits](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings).
This satisfies the planning-only frontier; it does not satisfy an implementation Ready gate.

## Files and symbols

Read-only evidence uses package prefix `io.github.pho001.synaptik`:

- `modules/model`: `model.tensor.Scan.apply/validateBindings`, `ScanBody.retainedValues`,
  `model.operation.scan.ScanAttrs.inputCount`, `model.graph.CompiledScanBody`.
- `modules/compiler`: `ScanBodyInference.capture/infer/refreshDemand`,
  `ScanGradientRules.preflight/apply`, `FirstOrderAutograd.expandBody`,
  `LinearAlgebraGradientRules.apply`, the [primal](../../../../glossary.md#term-primal)-bound `ReductionGradientRules.apply` overload,
  `SourceGraphReconstruction.preflight/reconstructScan`, `AutogradPreflight` scan-order rejection.
- `modules/planning`: [`OperationCapabilityQuery`](../../../../glossary.md#operation-capability-query--operationcapabilityquery); `modules/prepare`: [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext),
  [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation) demand [validation](../../../../glossary.md#term-validation). Existing scoped body attachment remains the [carrier](../../../../glossary.md#term-storage-carrier).
- `modules/backend/cpu`: `CpuGeneralScanLowering`, `CpuGeneralScanIr`, `CpuGeneralScanEmitter`,
  typed reduction/MATMUL emitters, <span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>`CpuGeneralScanPreparedExecutable`; fixed
  `CpuRecurrentScanPreparedExecutable.executeBound` supplies [preflight](../../../../glossary.md#term-preflight)-order evidence only.
- `modules/checkpoint`: `GraphArtifactCodec.writeNodes/attributes/Reader`, [graph](../../../../glossary.md#graph) copy/[rebind](../../../../glossary.md#capture-graph-ports-and-reconstruction);
  `extensions/nn`: `OwnedModelSource`, `ModelBuilder.RnnStep.forward`, current [Model](../../../../glossary.md#term-model) workflow.
- Existing focused tests to adapt are named in the proposal's
  [future acceptance](../../../length-aware-general-scan.md#proportional-future-validation-not-runs-now),
  not new tests or claimed executable evidence in this task.

## Acceptance criteria

- Preferred whole-batch bounded eligibility is compared with row-local binding; real ordinary
  linear/RNN MATMUL, broadcast and SUM survive without a fixed adjoint or axis-proof framework.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [binding glossary](../../../../glossary.md#term-binding); [linear glossary](../../../../glossary.md#term-linear-meanings); [RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [adjoint glossary](../../../../glossary.md#term-adjoint); [axis glossary](../../../../glossary.md#term-axis))</span>
- Proposed length role, rowCarryCount and per-sequence optional row axes are explicit. Shared
  W-transpose histories and equal-size row seeds remain distinguishable through pruning/capture/
  query/Prepare/cold/v5; no shape-based inference or temporary AD-only metadata shortcut.<span data-glossary-navigation="true"> ([transpose glossary](../../../../glossary.md#term-transpose); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [capture glossary](../../../../glossary.md#term-capture-meanings); [Prepare glossary](../../../../glossary.md#prepare); [shape glossary](../../../../glossary.md#shape); [inference glossary](../../../../glossary.md#term-inference); [AD glossary](../../../../glossary.md#autograd); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Temporal order, active SUM/MATMUL contributor domain, once-per-active-time shared ADD rounding,
  O(P) accumulator storage and inactive row identity are concrete; no B×parameter requirement.<span data-glossary-navigation="true"> ([rounding glossary](../../../../glossary.md#term-rounding); [accumulator glossary](../../../../glossary.md#accumulator-type); [identity glossary](../../../../glossary.md#term-identity-meanings); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- First-order AD requires rowCarryCount==carryCount AND every original sequenceRowAxes entry
  present0 (ROW(0)), including unused/non-gradient positions. Original shared-carry or
  SHARED/ROW(1)/other nonzero-axis sequence AD rejects before derivative allocation; future
  negative tests cover each. Broader forward validity, generated reverse mixed-role sequence
  inputs/shared sums, the ordinary TBF RNN target and old all-valid AD remain unchanged.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [carry glossary](../../../../glossary.md#term-carry); [derivative glossary](../../../../glossary.md#term-derivative); [TBF glossary](../../../../glossary.md#term-tbf); [target glossary](../../../../glossary.md#term-target))</span>
- Exact saved/previous primals, shared/row history activity and positive-zero padding are specified;
  no padded seed/history reads, activation replay or unguarded hoisted body work.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [seed glossary](../../../../glossary.md#term-seed); [activation glossary](../../../../glossary.md#term-activation))</span>
- Runtime lengths fully preflight after producers and before this scan's first write; no upstream
  rollback claim. O(body) code/prepared objects, dense output/live-history costs remain truthful.<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span><span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [preflight glossary](../../../../glossary.md#term-preflight); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- Old all-valid behavior/constructors/identities and v1–v4 bytes remain separate; new public record
  fields/overload and selective v5/new opcode are proposed, not silently accepted or schema-free.<span data-glossary-navigation="true"> ([opcode glossary](../../../../glossary.md#term-opcode); [schema glossary](../../../../glossary.md#term-schema))</span>
- Current fluent/fixed APIs and NN0032 second-order compile blocker, CONCAT-copy and mixed-engine
  gates remain. Scan lengths imply neither model-wide masking nor a masked loss denominator.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [engine glossary](../../../../glossary.md#term-engine); [model glossary](../../../../glossary.md#term-model); [loss glossary](../../../../glossary.md#loss))</span>
- One compact immediate brief, coarse Draft future phases, concise synchronized indexes, no
  normative edits, independent review and main acceptance before proposal-delivery Complete.<span data-glossary-navigation="true"> ([normative glossary](../../../../glossary.md#term-normative))</span>

## [Validation](../../../../glossary.md#term-validation)

Design-only worker validation: scoped `git diff --check`, new-file whitespace and line/byte counts;
local Markdown [target](../../../../glossary.md#term-target)/fragment validation; seven-field/status/[DAG](../../../../glossary.md#term-directed-acyclic-graph-dag)/example review; compare every
non-allowlisted baseline [hash](../../../../glossary.md#term-hash) and Git state with the supplied preservation baseline.
Reuse source/contract evidence; proposed APIs/examples are not compiled or execution-tested.

Main integration: accept the independent review and preserved-tree report before marking this
proposal task Complete. No Java, [Gradle](../../../../glossary.md#term-gradle), native, performance or repository test gate is authorized.

## Documentation and review impact

Primary [planning](../../../../glossary.md#term-planning) profile; decision-record option/rationale and conceptual-example requirements
apply to the proposal. No current glossary meaning, [API](../../../../glossary.md#term-application-programming-interface-api) [Javadoc](../../../../glossary.md#term-javadoc) or authority is changed.
Independent targeted Class C review is mandatory; it reviews the frozen exact seven-file draft
and current source/contracts, with no competing writer or duplicate executable [validation](../../../../glossary.md#term-validation).

## Result

Prepared the proposal, this brief and five concise [planning](../../../../glossary.md#term-planning)/index updates; no other paths changed.
Scoped `git diff --check --` with the seven [Scope](../../../../glossary.md#term-scope) paths passed. A read-only inline Ruby checker
validated local Markdown [targets](../../../../glossary.md#term-target)/fragments, final newlines and trailing whitespace; final counts
are recorded in the [handoff](../../../../glossary.md#term-handoff). `wc -lc` confirms this brief is below 200 lines/15 KB.
Read-only Ruby [SHA-256](../../../../glossary.md#term-sha-256) comparison against the supplied baseline found only the five allowed
existing documents changed; all other 2418 baseline files matched. The other two allowed documents
are new. `git rev-parse HEAD origin/main` retained the declared base; staged index remained empty.
No code/[Javadoc](../../../../glossary.md#term-javadoc)/contract/build edits or Java/[Gradle](../../../../glossary.md#term-gradle)/native/generated/performance [validation](../../../../glossary.md#term-validation) occurred.
Targeted glossary review found no current meaning change; proposed row roles are defined locally.
Independent final Class C review: APPROVE (2026-10-24), no unresolved P1/P2. The initial P2 on
non-ROW(0) original sequence [gradients](../../../../glossary.md#term-gradient) was corrected in the proposal, future negative-test wording
and this brief; targeted re-review confirms rejection before [derivative](../../../../glossary.md#term-derivative) allocation even for
unused/non-gradient positions, without restricting generated reverse mixed-role inputs.
The reviewer reused the full seven-document/source/contract review and successful link/math
evidence; MAIN's preserved-tree report covers 2425 files, seven allowed deltas and no other changes.
MAIN's conditional acceptance is satisfied by final APPROVE, for proposal delivery only.
Final factual closure changed only the seven Scope documents; scoped `git diff --check --`,
read-only inline Ruby local-link/fragment/newline/whitespace checks and `wc -lc` passed afterward.
No executable reruns or semantic rewriting. All new semantics/[API](../../../../glossary.md#term-application-programming-interface-api)/v5 remain PROPOSED; current
Javadoc/glossary meanings and NN0032's migration/second-order gates are unchanged.

Status: Complete
Proposal delivery only; no active planning/source/authority frontier, semantic/API/wire approval
or successor. No delivery follow-up remains; any future decision needs separate authorization.

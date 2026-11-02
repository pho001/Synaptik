# Task 0043: Symbolic-batch and runtime-length Scan integration design

## Status

Complete — DESIGN DELIVERY ONLY; independent APPROVE and MAIN acceptance, 2026-10-26.

## Change class

Class C planning packet: the proposed combination touches scoped automatic differentiation (AD),
run geometry/resources and generated hot loops. This task changes planning documents only and
does not enact those contracts or capabilities. MAIN selected the independent targeted reviewer.

## Goal

Deliver a concise, source-backed integration proposal for one named dynamic batch B and runtime
INT64[B] sequence lengths in general Scan: fixed T/features, FLOAT32 and FLOAT64 forward plus
actual Compiler-generated first-order backward. Extend the existing length-aware plan, not a new
framework. Proposal delivery is not user acceptance of new semantics or implementation authority.

## Scope

Admission writes only this brief. After MAIN confirms launch, the exclusive planning write scope is:

- this brief and [length-aware general Scan](../../../length-aware-general-scan.md);
- [dynamic-batch execution](../../../dynamic-batch-execution.md), crosslink/status only;
- relevant active sections of the [Model](../master-plan.md),
  [Compiler](../../compiler/master-plan.md), [Prepare](../../prepare/master-plan.md) and
  [CPU](../../../backends/cpu/master-plan.md) master plans;
- concise current-frontier/status updates in the [roadmap](../../../roadmap.md).

Preserve completed histories and every preexisting change. No additional successor brief without
MAIN approval. Inspect only current Scan algebra and real outer first-order closure/plumbing,
not all backend operations. Reuse current owners; production type/API additions: NONE proposed.
No production replacement/deletion is selected. Identify overlapping mutual-exclusion guards for
careful combined-case replacement while retaining unrelated validation.

## Non-goals

No Java/tests/build/TSV, normative contracts, ARCHITECTURE.md, ADRs or explanatory source-doc edits;
no implementation, measurement, native/Java runs, delegation, Git mutations, commit or push.
Exclude variable T/features, persistence/cold/v5, NN/TrainingSession reuse or fluent migration,
higher-order, low/mixed types, vector/parallel, OpenBLAS/Metal, new abstractions/executor/session
and the full Engine geometry memo. NN0032's higher-order compile, singleton-CONCAT and mixed-engine
gates remain held. CPU0042's nonstationary probe is not stable parity proof; no remeasurement.

## Contracts

Use the root scope index and these exact current headings; descendants apply where named:

- [ARCHITECTURE.md](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts):
  `Core lifecycle`, `Core invariants`, `Scope-indexed normative contracts`.
- [Recurrent scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan):
  `General differentiable scan`, `Accepted symbolic-batch scan` and descendants,
  `Length-aware whole-batch scan` and descendants: source roles, reverse, demand and preflight.
- [Compiler/autograd](../../../../architecture/contracts/compiler-autograd.md#modulescompiler):
  `modules/compiler`, `Compile artifacts`, `Accepted symbolic-batch metadata binding`,
  `Compiler-owned automatic differentiation`: independent inference, actual AD and fresh binding.
- [Runtime/Prepare/Engine](../../../../architecture/contracts/runtime-prepare-engine.md#accepted-symbolic-batch-resource-contract):
  `Accepted symbolic-batch resource contract`, `modules/prepare`, `Prepare lifecycle`,
  `Run lifecycle`, `modules/engine`, `Public reusable prepared execution`.
  `Accepted Engine geometry memo` is accepted but UNIMPLEMENTED, not a dependency of this proposal.
- [Backend execution](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes):
  `CPU backend routes`, `Accepted ordinary symbolic-batch CPU contract`.

Admission identifies a real boundary: length source/projection currently requires static shapes/B;
symbolic lengths and wider symbolic CPU dtype remain separately unqualified. Do not infer their
combination from either completed capability. The design must identify exact limited conflicting
headings and proposed changes, if needed, without editing authority or inventing approval gates.
Report any missing/ambiguous contract explicitly to MAIN; never silently select architecture.

## Dependencies and integration

- Depends on: Complete Model0042; Compiler0013/0014; Prepare0011/0013/0014;
  CPU0037/0041/0042; MAIN's verified accepted baseline and user planning authority.
- Conflicts with: Scan semantics/roles, AD/binding, geometry/resources, generated identities and
  the listed shared planning documents; serialize any writer to those scopes.
- Parallel group: None.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`
  plus the accepted uncommitted 2477-file snapshot, not merged predecessors.
- Integration order: admission brief → MAIN launch confirmation → sole-planner proposal/status
  edits → independent targeted review → MAIN integration/delivery acceptance.
- Integration validation: named Model0043 documentation-only checkpoint: owned links/anchors,
  whitespace, brief guardrails, source/contract traceability, frontier/authority consistency and
  accepted-base preservation; no executable validation or performance work now.
- Shared-document integration owner: MAIN; this planner writes only the explicitly delegated
  scope after confirmation, serially. MAIN integrates independent review findings.

### Admission verification — 2026-10-26

PlanningAuthorized=true; AuthorityAuthorized=false; ImplementationAuthorized=false.
The user's “ok, pust se do toho” accepted the explicit concise integration-planning proposal only.
MAIN directly checked Status/Result for all 12 listed baseline predecessors: Complete with final
reviews/gates; the required Scan integration predecessors are recorded above. Current active
Model/Compiler/Prepare/CPU masters and roadmap confirm source/authority frontiers None.
At admission, the only newly admitted frontier was Model0043 planning; Ready granted no implementation.
HEAD/origin match the recorded revision; MAIN verified unchanged index/full 2477-file snapshot.
Baseline: `/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/symbolic-length-planning-baseline.json`
and sibling `symbolic-length-planning-baseline-files/`. Preserve all preexisting contents.
Required planner readings completed: AGENTS.md, full planning guide/roadmap, relevant active master
sections and admission contract routing. No competing writer; independent reviewer is read-only.
No worktree was required for the sole serial writer; at admission, the source/F64 audit awaited launch.

MAIN launch verification: read the complete Ready brief, directly reverified predecessor
Status/Results and current frontiers, and compared the full accepted baseline. Exactly this new
brief differed; HEAD/origin/index were unchanged. All seven fields and user planning-only authority
were satisfied, conflicts inactive, sole writer and independent read-only reviewer confirmed.
MAIN authorized only Model0043 planning in the eight listed paths, with no extra successor brief.
At launch, MAIN required Review needed at delivery; independent review/final acceptance were MAIN-owned.

## Files and symbols

Read current affected source and focused tests after launch, using these bounded seams:

- Model `Scan.validateRows/requireLengthStatic`, ScanBody and ScanAttrs;
  ScanSymbolicBatchSourceTest and current length/row tests.
- Compiler ScanBodyInference's symbolic-length rejection/static row extent, ScanGradientRules,
  BatchBindingPlan and LengthAwareScan*/SymbolicScan*/binding tests.
- Prepare `analysis/PrepareContext` scoped projection/bindings and focused resource tests.
- CPU CpuGeneralScanLowering.analyzeSymbolic, CpuPartitionLowering's ordinary F32 guards,
  CpuGeneralScanIr.Stages, existing scan emitters/prepared executable/carrier/resource owners;
  CpuGeneralScan*, CpuSymbolicScan* and CpuOrdinaryDynamicBatch* focused tests.
- Engine symbolic admission/runValues and existing symbolic/length conformance/integration tests.

## Acceptance criteria

- Show one prepared execution at B=32 → 32 with different lengths → 7 → 1 → 32; B=0 separately.
  Lengths may be supplied or computed by producers; shape admission never reads L.
- Preserve producer ordering: all B lengths checked after producers, before ANY Scan writes,
  including invalid last length and T=0. T=0/B=0 are not whole-run shortcuts.
- Preserve raw carry/empty-row seeds, positive-zero padding, active NaN/Inf and nonactive load/
  arithmetic skipping; ROW(0), ROW(1), SHARED derive from provenance, never equal sizes.
- Specify true generated reverse, O(P) shared accumulators, distinct per-time contribution ADD,
  demanded/dead row/shared histories and exact saved activation sites, without replay.
- Keep one graph/prepared code/routes/topology; only numeric per-run geometry and mutable resources
  vary. No Bmax banks, per-B recompilation/reprepare/cache or element/time callbacks.
- Preserve fixed/static/all-valid behavior, identities and generated bytes. Separately version
  changed combined identities; no silent compatibility redesign.
- Provide a short source-backed F32/F64 map separating already typed Scan owners from genuinely
  F32-specific outer adjoint/seed/constant/SUM/MEAN/cast and INT64-length plumbing. Distinguish
  source validity, AD validity and CPU capability. Propose a minimal F64 prerequisite only if
  demonstrated; no F64 narrowing, F32-only target or arbitrary ordinary-op expansion.
- Give a short serial capability DAG, preferring cohesive source/AD/vertical phases. Detail only
  a stable immediate successor; keep later phases coarse. A narrow coordinated future contract
  cutover is proposed only for concrete static-only conflicts, not repeated authority-only tasks.
- Plan focused/affected-module validation and one combined integration checkpoint. Require an
  optimal clean-Java semantic/hot-loop/dataflow oracle, ClassFile/decompile checks and bounded
  warmed matched generated/direct evidence for future changed hot paths; run none now.

## Validation

Documentation-only worker checks after owned edits:

```bash
git diff --check -- docs/planning/modules/model/tasks/0043-symbolic-length-scan-integration-design.md
git diff --no-index --check /dev/null docs/planning/modules/model/tasks/0043-symbolic-length-scan-integration-design.md
wc -lc docs/planning/modules/model/tasks/0043-symbolic-length-scan-integration-design.md
```

Validate all owned relative Markdown targets/heading anchors and new-file whitespace; compare
accepted-baseline hashes and unchanged HEAD/origin/index. The brief targets ≤200 lines/15 KB.
MAIN's named documentation checkpoint also checks the final delegated planning diff and review.
No Java/native/benchmark run: this task changes no executable behavior. Future implementation
validation belongs in the proposal, not admission execution or duplicated predecessor runs.

## Documentation and review impact

Primary profile: implementation planning, with general style and documentation-rules workflow.
Only listed planning documents may change; relevant glossary terms are existing symbolic batch
binding and length-aware row roles. No glossary or Javadoc edits: no term/API/code change is enacted.
MAIN-selected independent Class C review is complete; existing successful executable evidence
was reused. The Result records reviewer approval and MAIN acceptance, not planner self-acceptance.

## Result

Delivered the [bounded integration proposal](../../../length-aware-general-scan.md#proposed-symbolic-b-plus-lengths-integration)
and concise synchronized status/crosslinks in all eight allowed planning paths.
Files: this new brief, length-aware-general-scan.md, dynamic-batch-execution.md, active
Model/Compiler/Prepare/CPU master sections and roadmap.md; no other file changed.
F64 decision: real typed symbolic packing/fold and bounded outer/control prerequisites;
proposed exact SUM/MEAN uses 34 limbs/280-byte scratch, not F32 narrowing or all-op expansion.
DAG: cohesive source with limited approved wording cutover → Compiler inference/actual AD/binding
→ Prepare/CPU/Engine vertical, with F64/control/ROW(1) prerequisites inside it. No extra brief.
Reused validation PASS: `git diff --check -- docs/planning`; 8/8 baseline-relative `git diff --no-index --check`
checks; `wc -lc` guardrails; inline `ruby -rjson -rdigest -ropen3` checked 46 added Markdown
links/anchors, seven fields/flags, baseline hashes, exact changed-path scope and Git state.
Final checkpoint: `node /private/tmp/synaptik-symbolic-length-main.jD6Dh2/check-planning.mjs` — PASS
(exit 0): 46 links/13 fragments, 2470 protected hashes unchanged; successful evidence reused, not rerun.
Accepted-base preservation: only seven allowlisted existing plans plus this new brief differ;
all other baseline files and HEAD/origin/index remain unchanged. No Java/native/benchmark runs.
Documentation impact: planning only; Javadoc unchanged (no code/API edits); glossary unchanged
after targeted symbolic-batch/row-role search (no new term or enacted semantics).
Hume the 4th final independent Class C APPROVE: 0 P1/P2, no unresolved findings.
MAIN accepted DESIGN DELIVERY ONLY on 2026-10-26; reviewed semantic prose/map/phases/bounds unchanged.
PlanningAuthorized=true is historical; AuthorityAuthorized=false; ImplementationAuthorized=false.
Current planning/source/authority frontiers: None. No new semantics or source authority is accepted.
Status: Complete
Required follow-up: None; future semantic/source capability needs separate approval, not task completion.

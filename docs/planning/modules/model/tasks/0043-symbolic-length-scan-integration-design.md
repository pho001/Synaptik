# Task 0043: Symbolic-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) and [runtime](../../../../glossary.md#term-runtime)-length Scan integration design

**Editorial reading key (current context, not a status update).**

This record closes a design-only proposal, not implementation authority; later capabilities
described in the current reference do not change that historical scope. The caller supplies
`ScanBody` formals (declared input placeholders) and positional actual Tensors to `Scan.apply`,
which describes a recurrence without executing it or owning result buffers. A leaf has no producing
expression, not necessarily stored values. Model semantics defines meaning; Compiler captures and
differentiates, Prepare handles bindings/resources, and execution providers realize independently
supported routes rather than redefine semantics. Semantic responsibility is not storage ownership.
Follow [source
construction](../../../../reference/tensor/general-scan.md#general-scan-source-construction) for
these boundaries.

Here `B` names batch size, `T` fixes time, and INT64 `[B]` lengths supply per-row active counts.
ROW(axis) requires coordinate provenance; SHARED means no row coordinate, not equal-sized storage.
Descriptor binding reads no lengths; execution checks all produced lengths before any Scan-local
write. For `T=0`, every length must be zero; carry preserves raw bits, while inactive emissions use
positive zero. Follow [length-aware construction and
execution](../../../../reference/tensor/general-scan.md#length-aware-source-construction-and-qualified-execution)
to distinguish metadata, AD and runtime qualification.

**Length-domain reading illustration.** For integer batch extent $B\ge0$ and fixed integer time
extent $T\ge0$, each runtime length $\ell_b$ must satisfy $0\le\ell_b\le T$ for row
$b\in\{0,\ldots,B-1\}$. At $B=2$, $T=3$, lengths $(0,2)$ mean the first row runs no steps and the
second runs positions zero and one. These are value obligations; an INT64 `[B]` descriptor alone
cannot prove them. No row checks exist when $B=0$, but other shared obligations remain.

## Status

Complete — DESIGN DELIVERY ONLY; independent APPROVE and MAIN acceptance, 2026-10-26.

## Change class

Class C [planning](../../../../glossary.md#term-planning) packet: the proposed combination touches scoped [automatic differentiation](../../../../glossary.md#autograd) ([AD](../../../../glossary.md#autograd)),
[run](../../../../glossary.md#term-run) geometry/resources and generated hot loops. This task changes planning documents only and
does not enact those contracts or capabilities. MAIN selected the independent targeted reviewer.

## Goal

Deliver a concise, source-backed integration proposal for one named dynamic [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) B and [runtime](../../../../glossary.md#term-runtime)
[INT64](../../../../glossary.md#data-type--datatype)[B] sequence lengths in [general Scan](../../../../glossary.md#general-differentiable-scan): fixed T/[features](../../../../glossary.md#term-feature), [FLOAT32](../../../../glossary.md#term-float32) and [FLOAT64](../../../../glossary.md#term-float64) forward plus
actual [Compiler](../../../../glossary.md#term-compiler)-generated first-order backward. Extend the existing length-aware plan, not a new
framework. Proposal delivery is not user acceptance of new semantics or implementation authority.

## [Scope](../../../../glossary.md#term-scope)

[Admission](../../../../glossary.md#term-admission) writes only this brief. After MAIN confirms launch, the exclusive [planning](../../../../glossary.md#term-planning) write scope is:

- this brief and [length-aware general Scan](../../../length-aware-general-scan.md)<span data-glossary-navigation="true"> ([general Scan glossary](../../../../glossary.md#general-differentiable-scan))</span>;
- [dynamic-batch execution](../../../dynamic-batch-execution.md)<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>, crosslink/status only;
- relevant active sections of the [Model](../master-plan.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>,
  [Compiler](../../compiler/master-plan.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>, [Prepare](../../prepare/master-plan.md)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span> and
  [CPU](../../../backends/cpu/master-plan.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span> master plans;
- concise current-frontier/status updates in the [roadmap](../../../roadmap.md).

Preserve completed histories and every preexisting change. No additional successor brief without
MAIN approval. Inspect only current Scan algebra and real outer first-order closure/plumbing,
not all [backend](../../../../glossary.md#backend) operations. Reuse current [owners](../../../../glossary.md#term-owner); production type/[API](../../../../glossary.md#term-application-programming-interface-api) additions: NONE proposed.
No production replacement/deletion is selected. Identify overlapping mutual-exclusion guards for
careful combined-case replacement while retaining unrelated [validation](../../../../glossary.md#term-validation).

## Non-goals

No Java/tests/build/TSV, [normative](../../../../glossary.md#term-normative) contracts, ARCHITECTURE.md, ADRs or explanatory source-doc edits;
no implementation, measurement, native/Java runs, delegation, Git mutations, [commit](../../../../glossary.md#term-commit) or push.
Exclude variable T/[features](../../../../glossary.md#term-feature), [persistence](../../../../glossary.md#term-persistence)/cold/v5, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/[TrainingSession](../../../../glossary.md#training-session--trainingsession) reuse or fluent migration,
higher-order, low/mixed types, [vector](../../../../glossary.md#term-vector)/parallel, [OpenBLAS](../../../../glossary.md#term-openblas)/[Metal](../../../../glossary.md#term-metal), new abstractions/executor/session
and the full [Engine geometry memo](../../../../glossary.md#engine-geometry-memo). NN0032's higher-order [compile](../../../../glossary.md#compile), [singleton](../../../../glossary.md#term-singleton)-CONCAT and mixed-[engine](../../../../glossary.md#term-engine)
gates remain held. CPU0042's [nonstationary](../../../../glossary.md#term-nonstationary) probe is not stable parity proof; no remeasurement.

## Contracts

Use the root [scope](../../../../glossary.md#term-scope) index and these exact current headings; descendants apply where named:

- [ARCHITECTURE.md](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts):
  `Core lifecycle`, `Core invariants`, `Scope-indexed normative contracts`.
- [Recurrent scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan):
  [`General differentiable scan`](../../../../glossary.md#general-differentiable-scan), `Accepted symbolic-batch scan` and descendants,
  `Length-aware whole-batch scan` and descendants: source roles, reverse, demand and [preflight](../../../../glossary.md#term-preflight).
- [Compiler/autograd](../../../../architecture/contracts/compiler-autograd.md#modulescompiler)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [autograd glossary](../../../../glossary.md#autograd))</span>:
  `modules/compiler`, [`Compile artifacts`](../../../../glossary.md#compile-artifacts), `Accepted symbolic-batch metadata binding`,
  `Compiler-owned automatic differentiation`: independent [inference](../../../../glossary.md#term-inference), actual [AD](../../../../glossary.md#autograd) and fresh [binding](../../../../glossary.md#term-binding).
- [Runtime/Prepare/Engine](../../../../architecture/contracts/runtime-prepare-engine.md#accepted-symbolic-batch-resource-contract)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine))</span>:
  `Accepted symbolic-batch resource contract`, `modules/prepare`, `Prepare lifecycle`,
  `Run lifecycle`, `modules/engine`, `Public reusable prepared execution`.
  `Accepted Engine geometry memo` is accepted but UNIMPLEMENTED, not a [dependency](../../../../glossary.md#term-dependency) of this proposal.
- [Backend execution](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend))</span>:
  `CPU backend routes`, `Accepted ordinary symbolic-batch CPU contract`.

[Admission](../../../../glossary.md#term-admission) identifies a real boundary: length source/projection currently requires static [shapes](../../../../glossary.md#shape)/B;
symbolic lengths and wider symbolic [CPU](../../../../glossary.md#term-cpu) [dtype](../../../../glossary.md#data-type--datatype) remain separately unqualified. Do not infer their
combination from either completed capability. The design must identify exact limited conflicting
headings and proposed changes, if needed, without editing authority or inventing approval gates.
Report any missing/ambiguous contract explicitly to MAIN; never silently select architecture.

## Dependencies and integration

- Depends on: Complete Model0042; Compiler0013/0014; Prepare0011/0013/0014;
  CPU0037/0041/0042; MAIN's verified accepted baseline and user planning authority.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Conflicts with: Scan semantics/roles, AD/binding, geometry/resources, generated identities and
  the listed shared planning documents; serialize any writer to those scopes.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([AD glossary](../../../../glossary.md#autograd); [binding glossary](../../../../glossary.md#term-binding))</span>
- Parallel group: None.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`
  plus the accepted uncommitted 2477-file snapshot, not merged predecessors.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Integration order: admission brief → MAIN launch confirmation → sole-planner proposal/status
  edits → independent targeted review → MAIN integration/delivery acceptance.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission))</span>
- Integration validation: named Model0043 documentation-only checkpoint: owned links/anchors,
  whitespace, brief guardrails, source/contract traceability, frontier/authority consistency and
  accepted-base preservation; no executable validation or performance work now.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Shared-document integration owner: MAIN; this planner writes only the explicitly delegated
  scope after confirmation, serially. MAIN integrates independent review findings.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [scope glossary](../../../../glossary.md#term-scope))</span>

### [Admission](../../../../glossary.md#term-admission) verification — 2026-10-26

PlanningAuthorized=true; AuthorityAuthorized=false; ImplementationAuthorized=false.
The user's “ok, pust se do toho” accepted the explicit concise integration-[planning](../../../../glossary.md#term-planning) proposal only.
MAIN directly checked Status/Result for all 12 listed baseline predecessors: Complete with final
reviews/gates; the required Scan integration predecessors are recorded above. Current active
[Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler)/[Prepare](../../../../glossary.md#prepare)/[CPU](../../../../glossary.md#term-cpu) masters and roadmap confirm source/authority frontiers None.
At admission, the only newly admitted frontier was Model0043 planning; [Ready](../../../../glossary.md#term-ready) granted no implementation.
HEAD/origin match the recorded revision; MAIN verified unchanged index/full 2477-file [snapshot](../../../../glossary.md#term-snapshot).
Baseline: `/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/symbolic-length-planning-baseline.json`
and sibling `symbolic-length-planning-baseline-files/`. Preserve all preexisting contents.
Required planner readings completed: AGENTS.md, full planning guide/roadmap, relevant active master
sections and admission contract routing. No competing writer; independent reviewer is read-only.
No worktree was required for the sole serial writer; at admission, the source/[F64](../../../../glossary.md#floating-point-precision-and-rounding) audit awaited launch.

MAIN launch verification: read the complete Ready brief, directly reverified predecessor
Status/Results and current frontiers, and compared the full accepted baseline. Exactly this new
brief differed; HEAD/origin/index were unchanged. All seven fields and user planning-only authority
were satisfied, conflicts inactive, sole writer and independent read-only reviewer confirmed.
MAIN authorized only Model0043 planning in the eight listed paths, with no extra successor brief.
At launch, MAIN required Review needed at delivery; independent review/final acceptance were MAIN-owned.

## Files and symbols

Read current affected source and focused tests after launch, using these bounded seams:

- [Model](../../../../glossary.md#term-model) `Scan.validateRows/requireLengthStatic`, ScanBody and ScanAttrs;
  ScanSymbolicBatchSourceTest and current length/row tests.
- Compiler ScanBodyInference's symbolic-length rejection/static row extent, ScanGradientRules,
  BatchBindingPlan and LengthAwareScan*/SymbolicScan*/binding tests.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [extent glossary](../../../../glossary.md#term-extent))</span>
- [Prepare](../../../../glossary.md#prepare) `analysis/PrepareContext` scoped projection/[bindings](../../../../glossary.md#term-binding) and focused resource tests.
- CPU CpuGeneralScanLowering.analyzeSymbolic, CpuPartitionLowering's ordinary F32 guards,
  CpuGeneralScanIr.Stages, existing scan emitters/prepared executable/carrier/resource owners;
  CpuGeneralScan*, CpuSymbolicScan* and CpuOrdinaryDynamicBatch* focused tests.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- [Engine](../../../../glossary.md#term-engine) symbolic [admission](../../../../glossary.md#term-admission)/runValues and existing symbolic/length [conformance](../../../../glossary.md#term-conformance)/integration tests.

## Acceptance criteria

- Show one prepared execution at B=32 → 32 with different lengths → 7 → 1 → 32; B=0 separately.
  Lengths may be supplied or computed by producers; shape admission never reads L.<span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [shape glossary](../../../../glossary.md#shape); [admission glossary](../../../../glossary.md#term-admission))</span>
- Preserve producer ordering: all B lengths checked after producers, before ANY Scan writes,
  including invalid last length and T=0. T=0/B=0 are not whole-run shortcuts.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [run glossary](../../../../glossary.md#term-run))</span>
- Preserve raw carry/empty-row seeds, positive-zero padding, active NaN/Inf and nonactive load/
  arithmetic skipping; ROW(0), ROW(1), SHARED derive from provenance, never equal sizes.<span data-glossary-navigation="true"> ([carry glossary](../../../../glossary.md#term-carry); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [padding glossary](../../../../glossary.md#padding); [NaN glossary](../../../../glossary.md#term-nan); [provenance glossary](../../../../glossary.md#provenance))</span>
- Specify true generated reverse, O(P) shared accumulators, distinct per-time contribution ADD,
  demanded/dead row/shared histories and exact saved activation sites, without replay.<span data-glossary-navigation="true"> ([activation glossary](../../../../glossary.md#term-activation))</span>
- Keep one graph/prepared code/routes/topology; only numeric per-run geometry and mutable resources
  vary. No Bmax banks, per-B recompilation/reprepare/cache or element/time callbacks.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [topology glossary](../../../../glossary.md#term-topology); [mutable glossary](../../../../glossary.md#term-mutable); [cache glossary](../../../../glossary.md#term-cache); [element glossary](../../../../glossary.md#term-element))</span>
- Preserve fixed/static/all-valid behavior, identities and generated bytes. Separately version
  changed combined identities; no silent compatibility redesign.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span>
- Provide a short source-backed F32/F64 map separating already typed Scan owners from genuinely
  F32-specific outer adjoint/seed/constant/SUM/MEAN/cast and INT64-length plumbing. Distinguish
  source validity, AD validity and CPU capability. Propose a minimal F64 prerequisite only if
  demonstrated; no F64 narrowing, F32-only target or arbitrary ordinary-op expansion.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [adjoint glossary](../../../../glossary.md#term-adjoint); [seed glossary](../../../../glossary.md#term-seed); [constant glossary](../../../../glossary.md#term-constant); [MEAN glossary](../../../../glossary.md#term-mean); [INT64 glossary](../../../../glossary.md#data-type--datatype); [AD glossary](../../../../glossary.md#autograd); [CPU glossary](../../../../glossary.md#term-cpu); [narrowing glossary](../../../../glossary.md#term-narrowing); [target glossary](../../../../glossary.md#term-target))</span>
- Give a short serial capability DAG, preferring cohesive source/AD/vertical phases. Detail only
  a stable immediate successor; keep later phases coarse. A narrow coordinated future contract
  cutover is proposed only for concrete static-only conflicts, not repeated authority-only tasks.<span data-glossary-navigation="true"> ([DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag))</span>
- Plan focused/affected-module validation and one combined integration checkpoint. Require an
  optimal clean-Java semantic/hot-loop/dataflow oracle, ClassFile/decompile checks and bounded
  warmed matched generated/direct evidence for future changed hot paths; run none now.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [validation glossary](../../../../glossary.md#term-validation); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [dataflow glossary](../../../../glossary.md#term-dataflow); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>

## [Validation](../../../../glossary.md#term-validation)

Documentation-only worker checks after owned edits:

```bash
git diff --check -- docs/planning/modules/model/tasks/0043-symbolic-length-scan-integration-design.md
git diff --no-index --check /dev/null docs/planning/modules/model/tasks/0043-symbolic-length-scan-integration-design.md
wc -lc docs/planning/modules/model/tasks/0043-symbolic-length-scan-integration-design.md
```

Validate all owned relative Markdown [targets](../../../../glossary.md#term-target)/heading anchors and new-file whitespace; compare
accepted-baseline [hashes](../../../../glossary.md#term-hash) and unchanged HEAD/origin/index. The brief targets ≤200 lines/15 KB.
MAIN's named documentation [checkpoint](../../../../glossary.md#term-checkpoint) also checks the final delegated [planning](../../../../glossary.md#term-planning) diff and review.
No Java/native/benchmark [run](../../../../glossary.md#term-run): this task changes no executable behavior. Future implementation
validation belongs in the proposal, not [admission](../../../../glossary.md#term-admission) execution or duplicated predecessor runs.

## Documentation and review impact

Primary profile: implementation [planning](../../../../glossary.md#term-planning), with general style and documentation-rules workflow.
Only listed planning documents may change; relevant glossary terms are existing [symbolic batch
binding](../../../../glossary.md#symbolic-batch-binding) and length-aware row roles. No glossary or [Javadoc](../../../../glossary.md#term-javadoc) edits: no term/[API](../../../../glossary.md#term-application-programming-interface-api)/code change is enacted.
MAIN-selected independent Class C review is complete; existing successful executable evidence
was reused. The Result records reviewer approval and MAIN acceptance, not planner self-acceptance.

## Result

Delivered the [bounded integration proposal](../../../length-aware-general-scan.md#proposed-symbolic-b-plus-lengths-integration)
and concise synchronized status/crosslinks in all eight allowed [planning](../../../../glossary.md#term-planning) paths.
Files: this new brief, length-aware-general-scan.md, dynamic-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-execution.md, active
[Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler)/[Prepare](../../../../glossary.md#prepare)/[CPU](../../../../glossary.md#term-cpu) master sections and roadmap.md; no other file changed.
[F64](../../../../glossary.md#floating-point-precision-and-rounding) decision: real typed symbolic packing/fold and bounded outer/control prerequisites;
proposed exact SUM/[MEAN](../../../../glossary.md#term-mean) uses 34 limbs/280-byte scratch, not [F32](../../../../glossary.md#floating-point-precision-and-rounding) [narrowing](../../../../glossary.md#term-narrowing) or all-op expansion.
[DAG](../../../../glossary.md#term-directed-acyclic-graph-dag): cohesive source with limited approved wording cutover → Compiler [inference](../../../../glossary.md#term-inference)/actual [AD](../../../../glossary.md#autograd)/[binding](../../../../glossary.md#term-binding)
→ Prepare/CPU/[Engine](../../../../glossary.md#term-engine) vertical, with F64/control/ROW(1) prerequisites inside it. No extra brief.
Reused [validation](../../../../glossary.md#term-validation) PASS: `git diff --check -- docs/planning`; 8/8 baseline-relative `git diff --no-index --check`
checks; `wc -lc` guardrails; inline `ruby -rjson -rdigest -ropen3` checked 46 added Markdown
links/anchors, seven fields/flags, baseline [hashes](../../../../glossary.md#term-hash), exact changed-path [scope](../../../../glossary.md#term-scope) and Git state.
Final [checkpoint](../../../../glossary.md#term-checkpoint): `node /private/tmp/synaptik-symbolic-length-main.jD6Dh2/check-planning.mjs` — PASS
(exit 0): 46 links/13 fragments, 2470 protected hashes unchanged; successful evidence reused, not rerun.
Accepted-base preservation: only seven allowlisted existing plans plus this new brief differ;
all other baseline files and HEAD/origin/index remain unchanged. No Java/native/benchmark runs.
Documentation impact: planning only; [Javadoc](../../../../glossary.md#term-javadoc) unchanged (no code/[API](../../../../glossary.md#term-application-programming-interface-api) edits); glossary unchanged
after targeted symbolic-batch/row-role search (no new term or enacted semantics).
Hume the 4th final independent Class C APPROVE: 0 P1/P2, no unresolved findings.
MAIN accepted DESIGN DELIVERY ONLY on 2026-10-26; reviewed semantic prose/map/phases/bounds unchanged.
PlanningAuthorized=true is historical; AuthorityAuthorized=false; ImplementationAuthorized=false.
Current planning/source/authority frontiers: None. No new semantics or source authority is accepted.
Status: Complete
Required follow-up: None; future semantic/source capability needs separate approval, not task completion.

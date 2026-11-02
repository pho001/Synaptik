# Task Model0041: Symbolic Scan and geometry-reuse authority cutover

## Status

Complete — AUTHORITY ONLY; MAIN accepted 2026-10-25 after independent frozen review
APPROVE, zero open P1/P2. PlanningAuthorized=true; AuthorityAuthorized=true;
ImplementationAuthorized=false. Authority frontier: None. Implementation frontier: None.

## Change class

Class C: scoped Scan semantics, Compiler first-order automatic differentiation (AD),
Prepare/Runtime resource geometry, Engine admission/concurrency and generated hot-loop contracts.
This cutover changes documents, not executable support.

## Goal and scope

Enact the user-approved [symbolic-Scan design](../general-scan-symbolic-batch-proposal.md),
including Model0040's same-geometry amendment, as a bounded accepted-but-unimplemented contract.
One named B, fixed T/features/state widths, one compilation/preparation and initial homogeneous
FLOAT32 scalar single-range CPU forward plus actual generated first-order backward are selected.
Model PURE/dtype source validity and Compiler's bounded F32/F64 AD admission remain separate.

Use existing owners and numeric protocols. The Scan contract alone owns complete Scan semantics;
Compiler, Runtime/Prepare/Engine and CPU own only their boundaries. Resolve the metadata-only
single-entry memo exception explicitly, without changing current static behavior or claiming
symbolic Scan, memoization, NN/Training reuse or wider execution is implemented.

## Non-goals

No Java/Javadoc/test-source/build/API/ABI/native edits, code generation, benchmarks, commits or push.
No module/dependency change, second graph/session/executor, new production type or public signature.
No static Scan regression, fixed RNN/GRU/LSTM BPTT, length execution, variable T/features,
higher-order, low/wider CPU dtype, native/mixed/vector/parallel, cold/copy/wire or NN/Training enablement.
Do not retrospectively rewrite completed task briefs or ADR0050's historical evidence.

## Contracts

Read root [Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence),
[Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle),
[Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
[Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts),
[Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules),
[Documentation authority and architecture updates](../../../../../ARCHITECTURE.md#documentation-authority-and-architecture-updates)
and [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).

Owning scoped headings (all current clauses remain constraints except the explicit cutover):

- Scan: [Source body, attachments and scoped identities](../../../../architecture/contracts/recurrent-scan.md#source-body-attachments-and-scoped-identities),
  [Canonical residual ports and demand](../../../../architecture/contracts/recurrent-scan.md#canonical-residual-ports-and-demand),
  [Reverse scan and exact primal bindings](../../../../architecture/contracts/recurrent-scan.md#reverse-scan-and-exact-primal-bindings),
  [Traversal, empty domains and declared qualification](../../../../architecture/contracts/recurrent-scan.md#traversal-empty-domains-and-declared-qualification),
  [Qualification, performance and migration gates](../../../../architecture/contracts/recurrent-scan.md#qualification-performance-and-migration-gates).
- Compiler: [Accepted symbolic-batch metadata binding](../../../../architecture/contracts/compiler-autograd.md#accepted-symbolic-batch-metadata-binding)
  and [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation).
- Runtime/Prepare/Engine: [Accepted symbolic-batch resource contract](../../../../architecture/contracts/runtime-prepare-engine.md#accepted-symbolic-batch-resource-contract),
  [`modules/prepare`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare),
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle),
  [Public reusable prepared execution](../../../../architecture/contracts/runtime-prepare-engine.md#public-reusable-prepared-execution).
- CPU: [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)
  and [Accepted ordinary symbolic-batch CPU contract](../../../../architecture/contracts/backend-execution.md#accepted-ordinary-symbolic-batch-cpu-contract).

No foundational numerical or extension capability contract changes are selected. On an additional
missing/ambiguous owner, unapproved type/protocol or scope expansion, stop and report it.

## Dependencies and integration

- Depends on: Complete Model0039 and Model0040 reviewed design deliveries; Complete Model0038,
  Compiler0013, Prepare0013 and CPU0037 retained source/AD/static/dynamic-ordinary baseline.
- Conflicts with: Scan/source/AD/resource/admission/identity semantics and all thirteen write paths;
  serialize these concerns, including shared authoritative documents.
- Parallel group: None.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`
  plus accepted uncommitted baseline, not merged predecessors; index empty/unchanged.
- Integration order: sole clean authority writer → freeze → independent read-only Class C review
  → MAIN acceptance metadata and final document validation; no source launch.
- Integration validation: exact-scope baseline hash audit, local targets/anchors, terminology,
  status/authority/frontier/size checks and `git diff --check`; no executable behavior changed.
- Shared-document integration owner: MAIN; the thirteen paths below are exclusively delegated
  to the writer until freeze. Independent reviewer is read-only.

Admission verified by MAIN on 2026-10-25: all six predecessor Status/Result headers Complete,
reviewed design deliveries accepted, IDs Model0041/ADR0051 free, HEAD/origin/index unchanged and
no competing writer. User confirmed the original plan after discussing static respecialization,
then explicitly said “ok, pokracuj tedy” following the stated contract-first next step.
This approves the bounded authority cutover only; no automatic Java successor is authorized.
Snapshot: `/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/pre-scan-symbolic-authority.json`.

## Files and symbols

Exclusive repository write scope (thirteen files):

- `ARCHITECTURE.md` — lifecycle/scope routing and accepted-future status, no detailed duplication.
- `docs/architecture/contracts/recurrent-scan.md` — sole complete symbolic Scan semantics.
- `docs/architecture/contracts/compiler-autograd.md` — scoped inference/actual AD/multi-output binding.
- `docs/architecture/contracts/runtime-prepare-engine.md` — symbolic projection, numeric-only sizing,
  actual publication and bounded immutable-geometry memo ownership/admission/lifecycle exception.
- `docs/architecture/contracts/backend-execution.md` — bounded CPU resources/emission/identity gates.
- `docs/architecture/runtime-prepare-backend-boundary.md` — focused conceptual workflow and rationale.
- `docs/design/decisions/0051-symbolic-scan-and-geometry-reuse.md` — new accepted rationale, not authority.
- `docs/glossary.md` — targeted symbolic-batch/Scan/run-geometry/prepared-owner wording only.
- This brief, Model `master-plan.md`, `general-scan-symbolic-batch-proposal.md`,
  `docs/planning/dynamic-batch-execution.md`, `docs/planning/roadmap.md` — synchronized current state.

Inspect source blockers from the proposal's existing-owner map and current focused tests;
no source edits. In particular: Scan/ScanBody static guards, ScanBodyInference/ScanGradientRules,
BatchBindingPlan multi-output and per-call bind, PrepareContext projection, CpuGeneralScanIr/
Lowering/Emitter and provider dispatch, Engine dynamic entry, RunMemoryGeometry/Test.
Proposed new/moved production types: zero; existing owner field/method changes and any necessary
exact signatures remain later scoped-brief decisions, with the explicit type-approval stop gate.

## Acceptance criteria

- Add explicit accepted-but-unimplemented symbolic Scan qualification; preserve implemented static
  F32/F64, dense-read exceptions and fixed-scan restrictions. Never import CPU F32 limits into Model.
- Define named-axis roles, unresolved template vs resolved actual layouts, every multi-output port,
  fixed parameter gradients, actual AD closure/count/K=B, scoped body obligations and exact saved sites.
- Keep T=0 a separate fixed-T plan; B=0 never skips required shared work/parameter zeros or whole-body
  validity. Preserve constituent arithmetic and independent length-value preflight/unsupported gates.
- Reuse immutable numeric size/access recipes and exact-plan geometry with fixed routes/code/slots;
  piecewise checked alignment/empty scratch is allowed, not fake B or B_max/topology selection.
- Own one bounded metadata memo at exact Engine preparation entry, keyed by exact semantic/inward/
  memory-plan identity and B/mode; no Runtime lookup, per-B executable cache or new owner/protocol.
- Keep complete fresh Compiler bind/private obligations and all actual input/storage/physical checks.
  Cache only covered pure Engine geometry and template-derived non-input metadata; never Binding,
  supplied descriptors, storage/addresses/leases/results or shared mutable invocation arrays.
- Explicitly distinguish reusable encapsulated immutable numeric facts from mutable run-owned arrays;
  prepared meaning remains immutable. Require atomic complete-entry publication, local snapshots,
  bounded active candidates, failure/retry and close coordination without an execution lock/second lease.
- Preserve generated clean-Java algorithm/hot-loop/dataflow, static keys/bytes, runtime-B exclusion
  from generated compatibility identity and independent actual reverse qualification. No speed promise.
- ADR compares status quo, per-B static respecialization and selected numeric binding/reuse; explanation
  shows a labeled future Scan example and 64→64→40→40→64 miss/hit witness with fresh resources.
- Future executable/conformance/integration/structure/bounded-performance obligations are explicit;
  no new dependency tests needed now because dependency rules are unchanged. Independent review has
  zero open blocking findings, and only then MAIN accepts the authority cutover.

## Validation

Worker: targeted documentation/contract consistency and local links/anchors, glossary search,
brief <=200 lines/15 KB, `git diff --check` and exact-thirteen-path audit against the snapshot.
MAIN supplies a dedicated read-only document validator during drafting. Reuse existing executable
evidence only as retained-baseline evidence, never as dynamic Scan/memo qualification.
No Gradle/Java/native/performance run is needed for a document-only future permission with unchanged
dependencies/API/executable behavior; later implementation owns matching architecture/conformance/
integration and generated-code tests. No test or Javadoc is claimed newly validated here.

## Documentation and review impact

Apply general, architecture, decision-record, planning and conceptual-example profiles according
to each file type. Review affected guide/Javadoc/glossary wording; do not broaden guides to imply
unimplemented support. Exact future Javadoc obligations stay with source implementation.
Independent clean targeted Class C documentation/contract review is mandatory before completion.

## Result

Authority/docs enacted across exactly the thirteen paths above: sole Scan semantics, linked boundary
owners, immutable-only Engine memo exception, ADR0051, glossary, future example and synchronized plans.
The accepted direction is unimplemented; current static/source/AD/CPU distinctions and identities remain.
Documentation profiles: general, architecture, decision-record, planning and conceptual examples.
Javadoc impact: no Java/Javadoc edited or generated; future implementation must update affected contracts.
Review-stage validation: `node /private/tmp/synaptik-scan-authority.ODRQ6j/check-authority.mjs` PASS
(review mode): 908 local links/anchors, 71 added links, zero failures; exact thirteen-path baseline audit,
preserved authoritative anchors, unchanged root incorporation/dependencies, HEAD/origin/index and
`git diff --check` PASS. Brief remains below 200 lines/15 KB. Targeted glossary/terminology and
conceptual byte/hit witness reviewed. No Gradle/Java/Javadoc/native/performance run; no executable claim.
Independent read-only Class C review: APPROVE, zero open P1/P2. One P2 wording finding was fixed:
CPU-int representability bounds apply to the initial CPU domain, not future other backends.
V2 differed from the first freeze only in that scoped paragraph; reviewer verified all thirteen
hashes/copies and cleared the finding. MAIN accepted 2026-10-25; subsequent edits are acceptance
metadata only. No production type/public signature, source change or dependency change.
Final validation: `node /private/tmp/synaptik-scan-authority.ODRQ6j/check-authority.mjs --final` —
PASS: links/anchors, status/authority/size/whitespace, exact thirteen-path hash scope and protected
HEAD/origin/index/outside files; unchanged root incorporation/dependencies and retained anchors.
Authority frontier: None. Implementation frontier: None. ImplementationAuthorized=false.
Next gate is a separately Ready Model source foundation, then Compiler and the numeric vertical;
no automatic launch, executable capability, source/test/native/performance run, commit or push.
Status: Complete

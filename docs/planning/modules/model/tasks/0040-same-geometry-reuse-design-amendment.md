# Task Model0040: Same-geometry reuse design amendment

## Status

Complete — DESIGN DELIVERY ONLY; MAIN accepted 2026-10-25 after independent frozen review
APPROVE, zero open P1/P2. PlanningAuthorized=true; ImplementationAuthorized=false.
Planning frontier: None. Implementation frontier: None. User “ok, dopln to” authorizes
this design addition, not the wider symbolic-Scan authority cutover or Java implementation.

## Change class

Class C design amendment: proposed Engine admission memoization, numeric resource geometry,
concurrent-run isolation and generated hot-path boundaries. No normative contract changes now.

## Goal and scope

Amend the existing [symbolic-Scan proposal](../general-scan-symbolic-batch-proposal.md) so repeated
matching geometry reuses checked immutable metadata without repeating pure geometry work.
Distinguish compile/prepare work, run-entry work and numeric loops. Keep current per-call input,
storage and lifecycle validation, fresh mutable run state and exact actual-size resources.

Exclusive writes: this brief, the proposal, Model master plan,
[dynamic-batch map](../../../dynamic-batch-execution.md) and [roadmap](../../../roadmap.md).
Do not rewrite completed Model0039's brief or its frozen review evidence.

## Non-goals

No Java/tests/build/ABI/API/normative/ADR/glossary edits, execution qualification, benchmarks,
commits or push. No buffer pool, per-B compiled/prepared-plan cache, new session/executor,
global cache, mutable shared invocation geometry or implicit broader Scan approval.

## Contracts

Read root [Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence),
[Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle) and
[Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts).
Scoped headings constrain the proposal; this task cannot amend them:

- Runtime/Prepare/Engine: [Accepted symbolic-batch resource contract](../../../../architecture/contracts/runtime-prepare-engine.md#accepted-symbolic-batch-resource-contract),
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle),
  [Public reusable prepared execution](../../../../architecture/contracts/runtime-prepare-engine.md#public-reusable-prepared-execution).
- Compiler: [Accepted symbolic-batch metadata binding](../../../../architecture/contracts/compiler-autograd.md#accepted-symbolic-batch-metadata-binding).
- CPU: [Accepted ordinary symbolic-batch CPU contract](../../../../architecture/contracts/backend-execution.md#accepted-ordinary-symbolic-batch-cpu-contract).
- Scan owner: [Source body, attachments and scoped identities](../../../../architecture/contracts/recurrent-scan.md#source-body-attachments-and-scoped-identities).

Current clauses assign resolved private size arrays to a run and prohibit mutable prepared
geometry. If proposing shared immutable resolved facts or a memo slot, identify the exact future
authority clarification, distinguish it from mutable invocation arrays, and enact nothing here.
Stop for any other missing contract or required scope expansion.

## Dependencies and integration

- Depends on: Complete Model0039 design delivery and Prepare0013 ordinary dynamic-B vertical.
- Conflicts with: the five planning paths, Scan geometry/admission semantics and future
  Engine/Runtime ownership wording; serialize all writes.
- Parallel group: None.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`
  plus accepted uncommitted baseline; index empty and unchanged.
- Integration order: sole clean planner → frozen independent read-only review → MAIN acceptance.
- Integration validation: local links/anchors, metadata/status, size, whitespace and exact-five-path
  comparison to the pre-amendment snapshot; no executable validation.
- Shared-document integration owner: MAIN; five planning paths exclusively delegated to the planner
  until freeze. Independent review is read-only.

Admission: MAIN verified both predecessors Complete, unused ID, inactive write conflicts,
unchanged HEAD/origin/index and the authorized planning-only frontier on 2026-10-25 before launch.
Baseline: `/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/pre-geometry-reuse-amendment.json`.
No implementation frontier or authority grant follows from this admission.

## Files and symbols

Inspect existing `compiler.BatchBindingPlan.bind`/`Binding.descriptor`, Engine
`PreparedExecution.validateBatchGeometry`/`publicationDescriptors` and
`AdvancedEngine.runOrdinaryBatchOpen`, plus Runtime `RunMemoryGeometry.resolve` and
`RunMemoryGeometryTest`. These currently recompute geometry at Engine run entry; the Runtime
test permits the same immutable geometry with distinct physical run resources, not Engine memoization.
No new/moved production types selected; if an implementation later needs one, use the existing
exact-name/package/purpose/signature approval gate rather than inventing it in this amendment.

## Acceptance criteria

- Label the optimization proposed, not implemented. Retain all symbolic-Scan dtype, empty-domain,
  scope, identity and later-authority gates from Model0039.
- Propose bounded same-preparation reuse of immutable B-dependent checks/sizes/offsets and
  template-derived publication metadata, with a consecutive same-B hit scenario and changed-B miss.
- Specify exact preparation/plan, explicit B/mode key validity (also `[B,0]`) and atomic complete
  publication; retain per-call `bind` and its private Compiler obligations. Never cache `Binding`, supplied Tensor
  identities/descriptors, host storage, addresses, leases, mutable arrays or runtime results.
- Validate every actual input occurrence, including inconsistent B, fixed axes, dtype, flags and
  layout, before trusting a hit; refresh storage associations and physical admission every call.
  Input-backed publication descriptors must come from the current invocation.
- Preserve numeric-only Runtime, exact-plan authentication, fresh physical resources and isolated
  invocation arrays; no per-element/step cache lookup or symbol evaluation, no zero-overhead claim.
- Record future tests for hit/miss, invalid same-B inputs, B=0/1/overflow, foreign preparations,
  failure/retry, close and concurrent mixed-B runs; bounded counters/probe, no benchmark campaign.
- Synchronize the three planning indexes; independent review has no unresolved blocking findings.

## Validation

Planning only: validate local Markdown targets/anchors, all seven integration fields, authorization
and frontier/status consistency; brief <=200 lines/15 KB and focused proposal <=250 lines.
Run `git diff --check` and an exact-five-path hash audit against the snapshot above, preserving
HEAD/origin/index and all other files. MAIN supplies the reusable check command before review.
No Gradle/Java/native/performance runs: production behavior is unchanged.

## Documentation and review impact

Apply general, planning and conceptual-example profiles from the documentation rules. Targeted
glossary review: symbolic batch binding, prepared execution and run geometry; no term redefinition
or glossary update now. No Java/Javadoc changes. Independent clean targeted review is mandatory.

## Result

Amended the proposal with a single-entry exact-preparation immutable metadata memo, full per-call
Compiler binding/storage admission, current input-backed publications and isolated run resources.
Recorded future ownership clarification, bounded memory/races/failure/close, hot-loop separation
and later tests; selected no API/type or authority change. Synchronized the three planning indexes.
Changed files: this brief, `../general-scan-symbolic-batch-proposal.md`, `../master-plan.md`,
`../../../dynamic-batch-execution.md`, `../../../roadmap.md`; all other accepted dirty files protected.
Validation: `node /private/tmp/synaptik-geometry-reuse.ZKvc5T/check-planning.mjs` — PASS (review stage):
400 local links, zero link/anchor failures; seven integration fields/status/authorization and size
guards pass (proposal 250 lines; brief below 200 lines/15 KB). Initial run failed only proposal
length; shortened before PASS. Exact-five-path SHA-256 audit passes against the supplied baseline;
HEAD/origin/index and all other files unchanged, including protected Model0039. `git diff --check`
passes. No Java/native/benchmark runs, commits/staging/push. MAIN used a clean planning context
and an independent read-only review context; the writer spawned no agents.
Documentation: general/planning/conceptual-example profiles applied; targeted glossary reviewed,
no redefinition or glossary change. No Java/Javadoc change; current API work remains per invocation.
Independent frozen review: APPROVE, zero open P1/P2; all five reviewed hashes matched the frozen
snapshot. MAIN accepted 2026-10-25. This approval covers the design amendment, not implementation.
MAIN finalized acceptance metadata only; no approval is inferred from Model0039.
Final validation: `node /private/tmp/synaptik-geometry-reuse.ZKvc5T/check-planning.mjs --final` —
PASS: 400 local links, status/authorization, size, whitespace and exact-five-path audit;
HEAD/origin/index and all files outside scope unchanged. No executable validation was required.
Future coordinated authority must resolve run-owned size-array wording and immutable prepared
semantics before any memo implementation; this is an explicit later gate, not enacted permission.
Planning frontier: None. Implementation frontier: None. ImplementationAuthorized=false.
Required future work: separately approve/admit the coordinated authority cutover before source work.
Status: Complete

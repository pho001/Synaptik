# Task Model0040: Same-geometry reuse design amendment

**Editorial reading key (current context, not a status update).**

This record approves a design amendment, not memoization implementation or broader symbolic-Scan
authority. Here, geometry means checked sizes, offsets and publication metadata; a memo is proposed
reuse within one exact preparation, keyed by numeric batch extent and mode. Consecutive calls with
batch extent three could reuse immutable facts, but not input identities, storage or results;
changing that extent requires a miss. Engine still calls Compiler binding for every invocation and
admits current storage; Runtime receives numeric geometry with fresh physical resources and isolated
mutable arrays.

For current context, [source
construction](../../../../reference/tensor/general-scan.md#general-scan-source-construction)
distinguishes Model-defined recurrence meaning from provider-supported execution. Formals describe
positions; actual Tensors fill them. Storage-free expressions neither execute nor own result
buffers; semantic responsibility is not storage ownership. The [length-aware
boundary](../../../../reference/tensor/general-scan.md#length-aware-source-construction-and-qualified-execution)
explains why descriptor binding does not read length values: execution separately validates them. A
zero feature extent does not erase the batch extent or its checks. These current descriptions do not
enlarge this historical approval.

## Status

Complete — DESIGN DELIVERY ONLY; MAIN accepted 2026-10-25 after independent frozen review
APPROVE, zero open P1/P2. PlanningAuthorized=true; ImplementationAuthorized=false.
[Planning](../../../../glossary.md#term-planning) frontier: None. Implementation frontier: None. User “ok, dopln to” authorizes
this design addition, not the wider symbolic-[Scan](../../../../glossary.md#term-cumulative-scan) authority cutover or Java implementation.

## Change class

Class C design amendment: proposed [Engine](../../../../glossary.md#term-engine) [admission](../../../../glossary.md#term-admission) [memoization](../../../../glossary.md#term-memoization), numeric resource geometry,
concurrent-[run](../../../../glossary.md#term-run) isolation and generated hot-path boundaries. No [normative](../../../../glossary.md#term-normative) contract changes now.

## Goal and [scope](../../../../glossary.md#term-scope)

Amend the existing [symbolic-Scan proposal](../general-scan-symbolic-batch-proposal.md)<span data-glossary-navigation="true"> ([Scan glossary](../../../../glossary.md#term-cumulative-scan))</span> so repeated
matching geometry reuses checked [immutable](../../../../glossary.md#term-immutable) [metadata](../../../../glossary.md#term-metadata) without repeating pure geometry work.
Distinguish [compile](../../../../glossary.md#compile)/[prepare](../../../../glossary.md#prepare) work, [run](../../../../glossary.md#term-run)-entry work and numeric loops. Keep current per-call input,
storage and [lifecycle](../../../../glossary.md#lifecycle) [validation](../../../../glossary.md#term-validation), fresh [mutable](../../../../glossary.md#term-mutable) [run state](../../../../glossary.md#run-state--runstate) and exact actual-size resources.

Exclusive writes: this brief, the proposal, [Model](../../../../glossary.md#term-model) master plan,
[dynamic-batch map](../../../dynamic-batch-execution.md)<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span> and [roadmap](../../../roadmap.md).
Do not rewrite completed Model0039's brief or its frozen review evidence.

## Non-goals

No Java/tests/build/[ABI](../../../../glossary.md#term-application-binary-interface-abi)/[API](../../../../glossary.md#term-application-programming-interface-api)/[normative](../../../../glossary.md#term-normative)/[ADR](../../../../glossary.md#term-architecture-decision-record-adr)/glossary edits, execution [qualification](../../../../glossary.md#term-qualification), benchmarks,
[commits](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) or push. No [buffer](../../../../glossary.md#term-buffer) pool, per-B compiled/prepared-plan [cache](../../../../glossary.md#term-cache), new session/executor,
global cache, [mutable](../../../../glossary.md#term-mutable) shared invocation geometry or implicit broader [Scan](../../../../glossary.md#term-cumulative-scan) approval.

## Contracts

Read root [Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence),
[Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span> and
[Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([normative glossary](../../../../glossary.md#term-normative))</span>.
Scoped headings constrain the proposal; this task cannot amend them:

- [Runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare)/[Engine](../../../../glossary.md#term-engine): [Accepted symbolic-batch resource contract](../../../../architecture/contracts/runtime-prepare-engine.md#accepted-symbolic-batch-resource-contract)<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>,
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle),
  [Public reusable prepared execution](../../../../architecture/contracts/runtime-prepare-engine.md#public-reusable-prepared-execution)<span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>.
- [Compiler](../../../../glossary.md#term-compiler): [Accepted symbolic-batch metadata binding](../../../../architecture/contracts/compiler-autograd.md#accepted-symbolic-batch-metadata-binding)<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [binding glossary](../../../../glossary.md#term-binding))</span>.
- [CPU](../../../../glossary.md#term-cpu): [Accepted ordinary symbolic-batch CPU contract](../../../../architecture/contracts/backend-execution.md#accepted-ordinary-symbolic-batch-cpu-contract).
- [Scan](../../../../glossary.md#term-cumulative-scan) [owner](../../../../glossary.md#term-owner): [Source body, attachments and scoped identities](../../../../architecture/contracts/recurrent-scan.md#source-body-attachments-and-scoped-identities)<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span>.

Current clauses assign resolved private size arrays to a [run](../../../../glossary.md#term-run) and prohibit [mutable](../../../../glossary.md#term-mutable) prepared
geometry. If proposing shared [immutable](../../../../glossary.md#term-immutable) resolved facts or a [memo](../../../../glossary.md#cache-and-memoization) slot, identify the exact future
authority clarification, distinguish it from mutable invocation arrays, and enact nothing here.
Stop for any other missing contract or required [scope](../../../../glossary.md#term-scope) expansion.

## Dependencies and integration

- Depends on: Complete Model0039 design delivery and Prepare0013 ordinary dynamic-B vertical.
- Conflicts with: the five planning paths, Scan geometry/admission semantics and future
  Engine/Runtime ownership wording; serialize all writes.<span data-glossary-navigation="true"> ([Scan glossary](../../../../glossary.md#term-cumulative-scan); [Engine glossary](../../../../glossary.md#term-engine); [Runtime glossary](../../../../glossary.md#term-runtime); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Parallel group: None.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`
  plus accepted uncommitted baseline; index empty and unchanged.
- Integration order: sole clean planner → frozen independent read-only review → MAIN acceptance.
- Integration validation: local links/anchors, metadata/status, size, whitespace and exact-five-path
  comparison to the pre-amendment snapshot; no executable validation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [metadata glossary](../../../../glossary.md#term-metadata); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Shared-document integration owner: MAIN; five planning paths exclusively delegated to the planner
  until freeze. Independent review is read-only.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>

[Admission](../../../../glossary.md#term-admission): MAIN verified both predecessors Complete, unused ID, inactive write conflicts,
unchanged HEAD/origin/index and the authorized [planning](../../../../glossary.md#term-planning)-only frontier on 2026-10-25 before launch.
Baseline: `/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/pre-geometry-reuse-amendment.json`.
No implementation frontier or authority grant follows from this admission.

## Files and symbols

Inspect existing `compiler.BatchBindingPlan.bind`/`Binding.descriptor`, [Engine](../../../../glossary.md#term-engine)
`PreparedExecution.validateBatchGeometry`/`publicationDescriptors` and
`AdvancedEngine.runOrdinaryBatchOpen`, plus [Runtime](../../../../glossary.md#term-runtime) `RunMemoryGeometry.resolve` and
`RunMemoryGeometryTest`. These currently recompute geometry at Engine [run](../../../../glossary.md#term-run) entry; the Runtime
test permits the same [immutable](../../../../glossary.md#term-immutable) geometry with distinct physical run resources, not Engine [memoization](../../../../glossary.md#term-memoization).
No new/moved production types selected; if an implementation later needs one, use the existing
exact-name/package/purpose/[signature](../../../../glossary.md#term-signature) approval gate rather than inventing it in this amendment.

## Acceptance criteria

- Label the optimization proposed, not implemented. Retain all symbolic-Scan dtype, empty-domain,
  scope, identity and later-authority gates from Model0039.<span data-glossary-navigation="true"> ([Scan glossary](../../../../glossary.md#term-cumulative-scan); [dtype glossary](../../../../glossary.md#data-type--datatype); [scope glossary](../../../../glossary.md#term-scope); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Propose bounded same-preparation reuse of immutable B-dependent checks/sizes/offsets and
  template-derived publication metadata, with a consecutive same-B hit scenario and changed-B miss.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [publication glossary](../../../../glossary.md#term-publication); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Specify exact preparation/plan, explicit B/mode key validity (also `[B,0]`) and atomic complete
  publication; retain per-call <span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>`bind` and its private [Compiler](../../../../glossary.md#term-compiler) obligations. Never [cache](../../../../glossary.md#term-cache) [`Binding`](../../../../glossary.md#term-binding), supplied Tensor
  identities/descriptors, host storage, addresses, leases, mutable arrays or runtime results.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [descriptors glossary](../../../../glossary.md#term-descriptor); [host storage glossary](../../../../glossary.md#host-storage); [leases glossary](../../../../glossary.md#term-lease); [mutable glossary](../../../../glossary.md#term-mutable); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Validate every actual input occurrence, including inconsistent B, fixed axes, dtype, flags and
  layout, before trusting a hit; refresh storage associations and physical admission every call.
  Input-backed publication descriptors must come from the current invocation.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [axes glossary](../../../../glossary.md#term-axis); [layout glossary](../../../../glossary.md#layout); [admission glossary](../../../../glossary.md#term-admission))</span>
- Preserve numeric-only Runtime, exact-plan authentication, fresh physical resources and isolated
  invocation arrays; no per-element/step cache lookup or symbol evaluation, no zero-overhead claim.<span data-glossary-navigation="true"> ([authentication glossary](../../../../glossary.md#term-authentication); [element glossary](../../../../glossary.md#term-element); [lookup glossary](../../../../glossary.md#term-lookup); [symbol glossary](../../../../glossary.md#term-symbol); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- Record future tests for hit/miss, invalid same-B inputs, B=0/1/overflow, foreign preparations,
  failure/retry, close and concurrent mixed-B runs; bounded counters/probe, no benchmark campaign.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow))</span>
- Synchronize the three [planning](../../../../glossary.md#term-planning) indexes; independent review has no unresolved blocking findings.

## [Validation](../../../../glossary.md#term-validation)

[Planning](../../../../glossary.md#term-planning) only: validate local Markdown [targets](../../../../glossary.md#term-target)/anchors, all seven integration fields, authorization
and frontier/status consistency; brief <=200 lines/15 KB and focused proposal <=250 lines.
[Run](../../../../glossary.md#term-run) `git diff --check` and an exact-five-path [hash](../../../../glossary.md#term-hash) audit against the [snapshot](../../../../glossary.md#term-snapshot) above, preserving
HEAD/origin/index and all other files. MAIN supplies the reusable check command before review.
No [Gradle](../../../../glossary.md#term-gradle)/Java/native/performance runs: production behavior is unchanged.

## Documentation and review impact

Apply general, [planning](../../../../glossary.md#term-planning) and conceptual-example profiles from the documentation rules. Targeted
glossary review: [symbolic batch binding](../../../../glossary.md#symbolic-batch-binding), [prepared execution](../../../../glossary.md#prepared-execution--preparedexecution) and [run](../../../../glossary.md#term-run) geometry; no term redefinition
or glossary update now. No Java/[Javadoc](../../../../glossary.md#term-javadoc) changes. Independent clean targeted review is mandatory.

## Result

Amended the proposal with a single-entry exact-preparation [immutable](../../../../glossary.md#term-immutable) [metadata](../../../../glossary.md#term-metadata) [memo](../../../../glossary.md#cache-and-memoization), full per-call
[Compiler](../../../../glossary.md#term-compiler) [binding](../../../../glossary.md#term-binding)/storage [admission](../../../../glossary.md#term-admission), current input-backed publications and isolated [run](../../../../glossary.md#term-run) resources.
Recorded future [ownership](../../../../glossary.md#term-ownership-meanings) clarification, bounded memory/races/failure/close, hot-loop separation
and later tests; selected no [API](../../../../glossary.md#term-application-programming-interface-api)/type or authority change. Synchronized the three [planning](../../../../glossary.md#term-planning) indexes.
Changed files: this brief, `../general-scan-symbolic-batch-proposal.md`, `../master-plan.md`,
`../../../dynamic-batch-execution.md`, `../../../roadmap.md`; all other accepted dirty files protected.
[Validation](../../../../glossary.md#term-validation): `node /private/tmp/synaptik-geometry-reuse.ZKvc5T/check-planning.mjs` — PASS (review stage):
400 local links, zero link/anchor failures; seven integration fields/status/authorization and size
guards pass (proposal 250 lines; brief below 200 lines/15 KB). Initial run failed only proposal
length; shortened before PASS. Exact-five-path [SHA-256](../../../../glossary.md#term-sha-256) audit passes against the supplied baseline;
HEAD/origin/index and all other files unchanged, including protected Model0039. `git diff --check`
passes. No Java/native/benchmark runs, [commits](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings)/staging/push. MAIN used a clean planning context
and an independent read-only review context; the writer spawned no agents.
Documentation: general/planning/conceptual-example profiles applied; targeted glossary reviewed,
no redefinition or glossary change. No Java/[Javadoc](../../../../glossary.md#term-javadoc) change; current API work remains per invocation.
Independent frozen review: APPROVE, zero open P1/P2; all five reviewed [hashes](../../../../glossary.md#term-hash) matched the frozen
[snapshot](../../../../glossary.md#term-snapshot). MAIN accepted 2026-10-25. This approval covers the design amendment, not implementation.
MAIN finalized acceptance metadata only; no approval is inferred from Model0039.
Final validation: `node /private/tmp/synaptik-geometry-reuse.ZKvc5T/check-planning.mjs --final` —
PASS: 400 local links, status/authorization, size, whitespace and exact-five-path audit;
HEAD/origin/index and all files outside [scope](../../../../glossary.md#term-scope) unchanged. No executable validation was required.
Future coordinated authority must resolve run-owned size-array wording and immutable prepared
semantics before any memo implementation; this is an explicit later gate, not enacted permission.
Planning frontier: None. Implementation frontier: None. ImplementationAuthorized=false.
Required future work: separately approve/admit the coordinated authority cutover before source work.
Status: Complete

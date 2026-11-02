# Task Model0039: General Scan symbolic-batch design

## Status

Complete — DESIGN DELIVERY ONLY; MAIN accepted 2026-10-24 after independent frozen review
APPROVE, zero open P1/P2. PlanningAuthorized=true records the completed planning admission;
ImplementationAuthorized=false. Planning frontier: None. Implementation frontier: None.
This is not architecture approval or Java readiness; no automatic successor launch is authorized.

## Change class

Class C design review: scoped Scan semantics, first-order automatic differentiation (AD),
Prepare/Runtime sizing and generated hot loops. This task changes no normative contract or code.

## Goal and scope

Deliver a concise source-grounded [proposal](../general-scan-symbolic-batch-proposal.md) for one
named per-run batch B, fixed time T/features F/state H, through existing compilation/preparation.
First execution qualification is all-valid homogeneous FLOAT32 CPU forward and actual generated
first-order backward. Separate source validity, AD admission and backend qualification.

Writes are exclusively this brief, the focused proposal, Model master plan,
[dynamic-batch map](../../../dynamic-batch-execution.md) and [roadmap](../../../roadmap.md).
Preserve the accepted dirty baseline and completed briefs. No other writer is active.

## Non-goals

No Java/tests/build/API-guide/ADR/normative edits, implementation, measurements, commits or push.
No lengths execution, fixed RecurrentScan BPTT, dynamic T/features, higher-order, wider CPU dtypes,
native/mixed/vector/parallel routes, cold/copy/checkpoint/wire or NN/Training enablement.
No new facade/session/executor, runtime Model interpreter, second graph or proof framework.

## Contracts

Current contracts are constraints, not changes enacted by this proposal. Read these exact headings:

- Root [Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence),
  [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle),
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules) and
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Scan owner: [General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan),
  [Source body, attachments and scoped identities](../../../../architecture/contracts/recurrent-scan.md#source-body-attachments-and-scoped-identities)
  (including accepted-future symbolic B), [Canonical residual ports and demand](../../../../architecture/contracts/recurrent-scan.md#canonical-residual-ports-and-demand),
  [Reverse scan and exact primal bindings](../../../../architecture/contracts/recurrent-scan.md#reverse-scan-and-exact-primal-bindings),
  [Traversal, empty domains and declared qualification](../../../../architecture/contracts/recurrent-scan.md#traversal-empty-domains-and-declared-qualification),
  [Qualification, performance and migration gates](../../../../architecture/contracts/recurrent-scan.md#qualification-performance-and-migration-gates)
  and [Runtime preflight and implementation gates](../../../../architecture/contracts/recurrent-scan.md#runtime-preflight-and-implementation-gates).
- Compiler: [Accepted symbolic-batch metadata binding](../../../../architecture/contracts/compiler-autograd.md#accepted-symbolic-batch-metadata-binding)
  and [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation).
- Runtime/Prepare/Engine: [Accepted symbolic-batch resource contract](../../../../architecture/contracts/runtime-prepare-engine.md#accepted-symbolic-batch-resource-contract),
  [`modules/prepare`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare),
  [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle),
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle) and
  [Public reusable prepared execution](../../../../architecture/contracts/runtime-prepare-engine.md#public-reusable-prepared-execution).
- CPU: [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)
  and [Accepted ordinary symbolic-batch CPU contract](../../../../architecture/contracts/backend-execution.md#accepted-ordinary-symbolic-batch-cpu-contract).

If a future implementation finds a missing/ambiguous applicable contract, stop for clarification.
This design explicitly proposes extending current static Scan guards; it does not override them.

## Dependencies and integration

- Depends on: Complete Model0035/0038, Compiler0013, Prepare0013 ordinary dynamic-B vertical,
  CPU0037; current static Scan/AD/Prepare and ordinary symbolic-B contracts.
- Conflicts with: Scan semantics/API, scoped capture/AD/demand, CPU geometry/identities,
  resource/admission/publication contracts and the five shared planning paths; serialize all writes.
- Parallel group: None.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`
  plus the accepted uncommitted baseline, not merged predecessors; index empty/unchanged.
- Integration order: sole writer drafts → freeze → independent read-only review → writer fixes
  only on MAIN continuation → final review/MAIN design acceptance; no implementation launch.
- Integration validation: planning-only link/anchor, metadata, size, whitespace and baseline-scope
  checkpoint; no executable validation in this task.
- Shared-document integration owner: MAIN, with these five planning paths exclusively delegated
  to this writer until freeze; independent reviewer is read-only.

Admission: MAIN verified one planning-only Model0039 frontier from the user's “ok, pokracuj”
request, dependencies Complete, ID free and inactive write conflicts. Writer rechecked dependency
Status headers and baseline hashes before edits: HEAD/origin/index unchanged, zero file deltas.
PlanningAuthorized=true records that request; it does not approve the proposed detailed semantics.
Prepare/Compiler/CPU implementation frontiers remain None. No agents are spawned by this writer.

## Files and symbols

The [proposal's ownership map](../general-scan-symbolic-batch-proposal.md#existing-owner-change-map)
names inspected source paths, current blockers and proposed extensions. Important correction:
there is no `ScopedScanCapture` type in current Compiler source; the owners are
`ScanBodyInference.capture` and `GraphCapture.captureBody`. Reuse them.
Proposed new/moved production types: zero; this is a design choice, not an implementation proof.
Existing record/field/method changes require the later approved authority and scoped briefs.

## Acceptance criteria

- Distinguish proposed symbolic Scan from current static F32/F64 Scan and implemented ordinary B.
- Specify X[T,B,F], h[B,H], fixed Wx/Wh/bias, emissions versus final carry and actual first-order
  reverse, with 64→40→1→64 on one fixed-T plan; T=0 uses a separate fixed-T plan.
- Define named-axis versus shared geometry, demanded histories/cuts, empty policies, checked
  pre-borrow closure, numeric-only Runtime, generated-loop oracle and static compatibility.
- Explicitly resolve the symbolic multi-output layout blocker, not just loops or a binder.
- Name minimal existing owners/types; preserve the nine additions of completed Prepare0013,
  without treating nine as a future blanket type-count ceiling.
- Select authority cutover as the immediate conditional successor; keep source/AD/CPU and later
  lengths/NN/Training gates cohesive and coarse, with no speculative Ready implementation briefs.
- Independent review returns approval with no unresolved blocking findings, then MAIN accepts
  design delivery before this task can become Complete. Detailed semantic approval remains separate.

## Validation

Planning only: local links/anchors against current files; seven metadata fields and authorization
flags; brief <=200 lines/15 KB; focused proposal about 150–200 lines; trailing whitespace;
`git diff --check`; hash comparison to the pre-scan baseline allowing only the five listed paths.
Exact commands/results and final sizes are recorded at the frozen review checkpoint.
No Gradle/Java/native/performance run is authorized or needed for this design delivery.

Future qualification, not commands to run now: each worker owns focused tests and one affected-module
test/Javadoc gate; independent review reuses successful evidence. MAIN owns one recorded boundary
join after ordered integration, including conformance/integration/architecture and the configured
root gate when applicable. Only a tiny bounded matched-work probe accompanies changed hot loops.

## Follow-up

After independent design approval and MAIN acceptance, obtain the user's explicit detailed
semantic/authority approval. Only then admit a coordinated authority cutover (not source work).
The [ordered gates](../general-scan-symbolic-batch-proposal.md#ordered-gates-and-approval-choices)
state its bounded scope and the later implementation sequence; no successor is Ready now.

## Documentation and review impact

Planning profile plus general/example style applied. Targeted glossary review covered symbolic
batch binding, general Scan, formals/primal bindings, row roles and residual demand. No term is
redefined; glossary/API/Javadoc updates belong to later authorized changes. Java/Javadocs unchanged.
Independent clean review is mandatory for this Class C proposal; MAIN supplies that context.

## Result

Complete DESIGN DELIVERY ONLY; MAIN accepted 2026-10-24.
Independent clean read-only review of the frozen five files: APPROVE, zero open P1/P2;
reviewed hashes matched. Both numeric-recipe and explicit capability-provider corrections cleared.
Completed the source-grounded proposal, compact brief and synchronized three planning indexes.
Files: this brief, `../general-scan-symbolic-batch-proposal.md`, `../master-plan.md`,
`../../../dynamic-batch-execution.md` and `../../../roadmap.md`; no other repository path changed.
Review-stage validation passed 406 local links/anchors (55 added), metadata, size and whitespace.
Final validation: `node /private/tmp/synaptik-scan-batch-planning.kVZX2c/check-planning.mjs --final`
PASS: the same links/anchors, seven integration fields, authorization/status, size and whitespace;
`git diff --check` and exact-five-path baseline hash audit passed, HEAD/origin/index unchanged.
Strict checker permits Complete only with `--final`; default mode still requires Review needed.
Planning frontier: None. Implementation frontier: None. ImplementationAuthorized=false.
No unresolved design-delivery findings. Detailed semantic approval is still required from the user,
then a separately admitted coordinated authority cutover; no automatic launch or Java readiness.
Zero new/moved production types remains a proposal; future exact CPU record signatures stay gated.
Planning documentation only; no term, normative contract, API or Java/Javadoc change.
No executable tests, native/performance runs, commits or push. Final files are frozen.
Status: Complete

# Task Model0039: [General Scan](../../../../glossary.md#general-differentiable-scan) symbolic-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) design

**Editorial reading key (current context, not a status update).**

This record completes a design delivery, not semantic authority or implementation authorization; its
proposed 64→40→1→64 batch sequence holds time, feature and state extents fixed. Current references
describe later capabilities, not changes enacted here. In [Scan
construction](../../../../reference/tensor/general-scan.md#formals-actual-positions-and-source-closure),
follow the caller’s positional binding of actual Tensors to body formals: carry passes state,
sequences supply time slices, invariants stay unchanged, and emissions stack per-step outputs rather
than denote final carry. A leaf has no producing expression, not necessarily stored values.

Model semantics defines meaning; Compiler captures scoped bodies, differentiates and emits demand;
Prepare validates bindings/resources; qualified execution providers realize supported routes without
redefining semantics. These responsibilities do not imply storage ownership: storage-free
expressions and result ports neither execute nor own result buffers, and residual demand identifies
needed histories rather than allocating them. Follow [shape
broadcasting](../../../../reference/tensor/shapes.md#broadcasting) to distinguish a named batch
extent from concrete binding: matching names or singleton expansion does not prove an independent
ROW coordinate; SHARED means no such coordinate, not scalar shape.

**Shape reading illustration.** Let $T=10$ be fixed time steps, $F=16$ fixed features and $B\ge0$ a
named integer batch extent. A sequence description has Shape $[T,B,F]$ and each step slice $[B,F]$.
Binding $B=40$ yields numeric sizes $[10,40,16]$ without changing $T$ or $F$. This substitution
describes geometry only, not reuse of values, prepared resources, or proof that a coordinate is
independently ROW.

## Status

Complete — DESIGN DELIVERY ONLY; MAIN accepted 2026-10-24 after independent frozen review
APPROVE, zero open P1/P2. PlanningAuthorized=true records the completed [planning](../../../../glossary.md#term-planning) [admission](../../../../glossary.md#term-admission);
ImplementationAuthorized=false. Planning frontier: None. Implementation frontier: None.
This is not architecture approval or Java readiness; no automatic successor launch is authorized.

## Change class

Class C design review: scoped Scan semantics, first-order [automatic differentiation](../../../../glossary.md#autograd) ([AD](../../../../glossary.md#autograd)),
[Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime) sizing and generated hot loops. This task changes no [normative](../../../../glossary.md#term-normative) contract or code.

## Goal and [scope](../../../../glossary.md#term-scope)

Deliver a concise source-grounded [proposal](../general-scan-symbolic-batch-proposal.md) for one
named per-[run](../../../../glossary.md#term-run) [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) B, fixed time T/[features](../../../../glossary.md#term-feature) F/state H, through existing [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names)/preparation.
First execution [qualification](../../../../glossary.md#term-qualification) is all-valid [homogeneous](../../../../glossary.md#term-homogeneous) [FLOAT32](../../../../glossary.md#term-float32) [CPU](../../../../glossary.md#term-cpu) forward and actual generated
first-order backward. Separate source validity, [AD](../../../../glossary.md#autograd) [admission](../../../../glossary.md#term-admission) and [backend](../../../../glossary.md#backend) qualification.

Writes are exclusively this brief, the focused proposal, [Model](../../../../glossary.md#term-model) master plan,
[dynamic-batch map](../../../dynamic-batch-execution.md) and [roadmap](../../../roadmap.md).
Preserve the accepted dirty baseline and completed briefs. No other writer is active.

## Non-goals

No Java/tests/build/[API](../../../../glossary.md#term-application-programming-interface-api)-guide/[ADR](../../../../glossary.md#term-architecture-decision-record-adr)/[normative](../../../../glossary.md#term-normative) edits, implementation, measurements, [commits](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) or push.
No lengths execution, fixed RecurrentScan [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt), dynamic T/[features](../../../../glossary.md#term-feature), higher-order, wider [CPU](../../../../glossary.md#term-cpu) [dtypes](../../../../glossary.md#data-type--datatype),
native/mixed/[vector](../../../../glossary.md#term-vector)/parallel routes, cold/copy/[checkpoint](../../../../glossary.md#term-checkpoint)/wire or [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training enablement.
No new [facade](../../../../glossary.md#term-facade)/session/executor, [runtime](../../../../glossary.md#term-runtime) [Model](../../../../glossary.md#term-model) interpreter, second [graph](../../../../glossary.md#graph) or proof framework.

## Contracts

Current contracts are constraints, not changes enacted by this proposal. Read these exact headings:

- Root [Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence),
  [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>,
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [normative glossary](../../../../glossary.md#term-normative))</span>,
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span> and
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Scan [owner](../../../../glossary.md#term-owner): [General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)<span data-glossary-navigation="true"> ([General differentiable scan glossary](../../../../glossary.md#general-differentiable-scan))</span>,
  [Source body, attachments and scoped identities](../../../../architecture/contracts/recurrent-scan.md#source-body-attachments-and-scoped-identities)<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span>
  (including accepted-future symbolic B), [Canonical residual ports and demand](../../../../architecture/contracts/recurrent-scan.md#canonical-residual-ports-and-demand)<span data-glossary-navigation="true"> ([Canonical glossary](../../../../glossary.md#term-canonical-form); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand))</span>,
  [Reverse scan and exact primal bindings](../../../../architecture/contracts/recurrent-scan.md#reverse-scan-and-exact-primal-bindings)<span data-glossary-navigation="true"> ([primal glossary](../../../../glossary.md#term-primal); [bindings glossary](../../../../glossary.md#term-binding))</span>,
  [Traversal, empty domains and declared qualification](../../../../architecture/contracts/recurrent-scan.md#traversal-empty-domains-and-declared-qualification)<span data-glossary-navigation="true"> ([qualification glossary](../../../../glossary.md#term-qualification))</span>,
  [Qualification, performance and migration gates](../../../../architecture/contracts/recurrent-scan.md#qualification-performance-and-migration-gates)
  and [Runtime preflight and implementation gates](../../../../architecture/contracts/recurrent-scan.md#runtime-preflight-and-implementation-gates)<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight))</span>.
- [Compiler](../../../../glossary.md#term-compiler): [Accepted symbolic-batch metadata binding](../../../../architecture/contracts/compiler-autograd.md#accepted-symbolic-batch-metadata-binding)<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [metadata glossary](../../../../glossary.md#term-metadata))</span>
  and [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([automatic differentiation glossary](../../../../glossary.md#autograd))</span>.
- [Runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare)/[Engine](../../../../glossary.md#term-engine): [Accepted symbolic-batch resource contract](../../../../architecture/contracts/runtime-prepare-engine.md#accepted-symbolic-batch-resource-contract),
  [`modules/prepare`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare),
  [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle),
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span> and
  [Public reusable prepared execution](../../../../architecture/contracts/runtime-prepare-engine.md#public-reusable-prepared-execution)<span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>.
- [CPU](../../../../glossary.md#term-cpu): [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
  and [Accepted ordinary symbolic-batch CPU contract](../../../../architecture/contracts/backend-execution.md#accepted-ordinary-symbolic-batch-cpu-contract).

If a future implementation finds a missing/ambiguous applicable contract, stop for clarification.
This design explicitly proposes extending current static Scan guards; it does not override them.

## Dependencies and integration

- Depends on: Complete Model0035/0038, Compiler0013, Prepare0013 ordinary dynamic-B vertical,
  CPU0037; current static Scan/AD/Prepare and ordinary symbolic-B contracts.<span data-glossary-navigation="true"> ([AD glossary](../../../../glossary.md#autograd))</span>
- Conflicts with: Scan semantics/API, scoped capture/AD/demand, CPU geometry/identities,
  resource/admission/publication contracts and the five shared planning paths; serialize all writes.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [capture glossary](../../../../glossary.md#term-capture-meanings); [publication glossary](../../../../glossary.md#term-publication))</span>
- Parallel group: None.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`
  plus the accepted uncommitted baseline, not merged predecessors; index empty/unchanged.
- Integration order: sole writer drafts → freeze → independent read-only review → writer fixes
  only on MAIN continuation → final review/MAIN design acceptance; no implementation launch.
- Integration validation: planning-only link/anchor, metadata, size, whitespace and baseline-scope
  checkpoint; no executable validation in this task.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [metadata glossary](../../../../glossary.md#term-metadata); [scope glossary](../../../../glossary.md#term-scope); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Shared-document integration owner: MAIN, with these five planning paths exclusively delegated
  to this writer until freeze; independent reviewer is read-only.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>

[Admission](../../../../glossary.md#term-admission): MAIN verified one [planning](../../../../glossary.md#term-planning)-only Model0039 frontier from the user's “ok, pokracuj”
request, dependencies Complete, ID free and inactive write conflicts. Writer rechecked [dependency](../../../../glossary.md#term-dependency)
Status headers and baseline [hashes](../../../../glossary.md#term-hash) before edits: HEAD/origin/index unchanged, zero file deltas.
PlanningAuthorized=true records that request; it does not approve the proposed detailed semantics.
[Prepare](../../../../glossary.md#prepare)/[Compiler](../../../../glossary.md#term-compiler)/[CPU](../../../../glossary.md#term-cpu) implementation frontiers remain None. No agents are spawned by this writer.

## Files and symbols

The [proposal's ownership map](../general-scan-symbolic-batch-proposal.md#existing-owner-change-map)<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
names inspected source paths, current blockers and proposed extensions. Important correction:
there is no `ScopedScanCapture` type in current [Compiler](../../../../glossary.md#term-compiler) source; the [owners](../../../../glossary.md#term-owner) are
`ScanBodyInference.capture` and `GraphCapture.captureBody`. Reuse them.
Proposed new/moved production types: zero; this is a design choice, not an implementation proof.
Existing record/field/method changes require the later approved authority and scoped briefs.

## Acceptance criteria

- Distinguish proposed [symbolic Scan](../../../../glossary.md#symbolic-scan) from current static [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) Scan and implemented ordinary B.
- Specify X[T,B,F], h[B,H], fixed Wx/Wh/bias, emissions versus final carry and actual first-order
  reverse, with 64→40→1→64 on one fixed-T plan; T=0 uses a separate fixed-T plan.<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [carry glossary](../../../../glossary.md#term-carry))</span>
- Define named-axis versus shared geometry, demanded histories/cuts, empty policies, checked
  pre-borrow closure, numeric-only Runtime, generated-loop oracle and static compatibility.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [Runtime glossary](../../../../glossary.md#term-runtime); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- Explicitly resolve the symbolic multi-output [layout](../../../../glossary.md#layout) blocker, not just loops or a binder.
- Name minimal existing owners/types; preserve the nine additions of completed Prepare0013,
  without treating nine as a future blanket type-count ceiling.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span>
- Select authority cutover as the immediate conditional successor; keep source/AD/CPU and later
  lengths/NN/Training gates cohesive and coarse, with no speculative Ready implementation briefs.<span data-glossary-navigation="true"> ([AD glossary](../../../../glossary.md#autograd); [CPU glossary](../../../../glossary.md#term-cpu); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Ready glossary](../../../../glossary.md#term-ready))</span>
- Independent review returns approval with no unresolved blocking findings, then MAIN accepts
  design delivery before this task can become Complete. Detailed semantic approval remains separate.

## [Validation](../../../../glossary.md#term-validation)

[Planning](../../../../glossary.md#term-planning) only: local links/anchors against current files; seven [metadata](../../../../glossary.md#term-metadata) fields and authorization
flags; brief <=200 lines/15 KB; focused proposal about 150–200 lines; trailing whitespace;
`git diff --check`; [hash](../../../../glossary.md#term-hash) comparison to the pre-scan baseline allowing only the five listed paths.
Exact commands/results and final sizes are recorded at the frozen review [checkpoint](../../../../glossary.md#term-checkpoint).
No [Gradle](../../../../glossary.md#term-gradle)/Java/native/performance [run](../../../../glossary.md#term-run) is authorized or needed for this design delivery.

Future [qualification](../../../../glossary.md#term-qualification), not commands to run now: each worker owns focused tests and one affected-[module](../../../../glossary.md#term-module)
test/[Javadoc](../../../../glossary.md#term-javadoc) gate; independent review reuses successful evidence. MAIN owns one recorded boundary
join after ordered integration, including [conformance](../../../../glossary.md#term-conformance)/integration/architecture and the configured
root gate when applicable. Only a tiny bounded matched-work probe accompanies changed hot loops.

## Follow-up

After independent design approval and MAIN acceptance, obtain the user's explicit detailed
semantic/authority approval. Only then admit a coordinated authority cutover (not source work).
The [ordered gates](../general-scan-symbolic-batch-proposal.md#ordered-gates-and-approval-choices)
state its bounded [scope](../../../../glossary.md#term-scope) and the later implementation sequence; no successor is [Ready](../../../../glossary.md#term-ready) now.

## Documentation and review impact

[Planning](../../../../glossary.md#term-planning) profile plus general/example style applied. Targeted glossary review covered [symbolic
batch binding](../../../../glossary.md#symbolic-batch-binding), [general Scan](../../../../glossary.md#general-differentiable-scan), formals/[primal](../../../../glossary.md#term-primal) [bindings](../../../../glossary.md#term-binding), row roles and [residual](../../../../glossary.md#scan-residual-port-and-history-demand) demand. No term is
redefined; glossary/[API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc) updates belong to later authorized changes. Java/Javadocs unchanged.
Independent clean review is mandatory for this Class C proposal; MAIN supplies that context.

## Result

Complete DESIGN DELIVERY ONLY; MAIN accepted 2026-10-24.
Independent clean read-only review of the frozen five files: APPROVE, zero open P1/P2;
reviewed [hashes](../../../../glossary.md#term-hash) matched. Both numeric-recipe and explicit capability-provider corrections cleared.
Completed the source-grounded proposal, compact brief and synchronized three [planning](../../../../glossary.md#term-planning) indexes.
Files: this brief, `../general-scan-symbolic-batch-proposal.md`, `../master-plan.md`,
`../../../dynamic-batch-execution.md` and `../../../roadmap.md`; no other repository path changed.
Review-stage [validation](../../../../glossary.md#term-validation) passed 406 local links/anchors (55 added), [metadata](../../../../glossary.md#term-metadata), size and whitespace.
Final validation: `node /private/tmp/synaptik-scan-batch-planning.kVZX2c/check-planning.mjs --final`
PASS: the same links/anchors, seven integration fields, authorization/status, size and whitespace;
`git diff --check` and exact-five-path baseline [hash](../../../../glossary.md#term-hash) audit passed, HEAD/origin/index unchanged.
Strict checker permits Complete only with `--final`; default mode still requires Review needed.
Planning frontier: None. Implementation frontier: None. ImplementationAuthorized=false.
No unresolved design-delivery findings. Detailed semantic approval is still required from the user,
then a separately admitted coordinated authority cutover; no automatic launch or Java readiness.
Zero new/moved production types remains a proposal; future exact [CPU](../../../../glossary.md#term-cpu) record [signatures](../../../../glossary.md#term-signature) stay gated.
Planning documentation only; no term, [normative](../../../../glossary.md#term-normative) contract, [API](../../../../glossary.md#term-application-programming-interface-api) or Java/[Javadoc](../../../../glossary.md#term-javadoc) change.
No executable tests, native/performance runs, [commits](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) or push. Final files are frozen.
Status: Complete

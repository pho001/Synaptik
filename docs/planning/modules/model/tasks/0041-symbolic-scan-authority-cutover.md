# Task Model0041: [Symbolic Scan](../../../../glossary.md#symbolic-scan) and geometry-reuse authority cutover

**Editorial reading key (current context, not a status update).**

This record accepts authority, not implementation; its review evidence concerns documents, not
symbolic execution. Current references describe later, separately qualified support. A Scan body is
a step template: carry changes, sequences supply slices, invariants stay fixed. The caller supplies
formal Tensor placeholders to `ScanBody`, then positional actual Tensors to `Scan.apply`;
construction checks descriptions without reading values. A leaf has no producing expression, not
necessarily stored values. Storage-free expressions neither execute nor own result buffers. Follow
[source
binding](../../../../reference/tensor/general-scan.md#formals-actual-positions-and-source-closure)
for these distinctions.

Model semantics defines meaning; Compiler captures and differentiates, Prepare projects resources,
and execution providers realize qualified routes rather than redefine semantics. Semantic
responsibility is not buffer ownership. One named batch extent leaves time/features fixed; [shape
rules](../../../../reference/tensor/shapes.md#shapes-and-dimensions) explain symbolic dimensions.
The historical 64→64→40→40→64 witness describes a single-entry geometry memo: miss, hit, miss, hit,
miss—not executable caching or result reuse. Only immutable numeric metadata qualifies; binding and
run resources stay fresh. Zero batch still requires shared work; zero time preserves initial carry
raw bits.

## Status

Complete — AUTHORITY ONLY; MAIN accepted 2026-10-25 after independent frozen review
APPROVE, zero open P1/P2. PlanningAuthorized=true; AuthorityAuthorized=true;
ImplementationAuthorized=false. Authority frontier: None. Implementation frontier: None.

## Change class

Class C: scoped Scan semantics, [Compiler](../../../../glossary.md#term-compiler) first-order [automatic differentiation](../../../../glossary.md#autograd) ([AD](../../../../glossary.md#autograd)),
[Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime) resource geometry, [Engine](../../../../glossary.md#term-engine) [admission](../../../../glossary.md#term-admission)/[concurrency](../../../../glossary.md#term-concurrency) and generated hot-loop contracts.
This cutover changes documents, not executable support.

## Goal and [scope](../../../../glossary.md#term-scope)

Enact the user-approved [symbolic-Scan design](../general-scan-symbolic-batch-proposal.md),
including Model0040's same-geometry amendment, as a bounded accepted-but-unimplemented contract.
One named B, fixed T/[features](../../../../glossary.md#term-feature)/state widths, one [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names)/preparation and initial [homogeneous](../../../../glossary.md#term-homogeneous)
[FLOAT32](../../../../glossary.md#term-float32) [scalar](../../../../glossary.md#term-scalar) single-range [CPU](../../../../glossary.md#term-cpu) forward plus actual generated first-order backward are selected.
[Model](../../../../glossary.md#term-model) [PURE](../../../../glossary.md#term-pure)/[dtype](../../../../glossary.md#data-type--datatype) source validity and [Compiler](../../../../glossary.md#term-compiler)'s bounded [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) [AD](../../../../glossary.md#autograd) [admission](../../../../glossary.md#term-admission) remain separate.

Use existing [owners](../../../../glossary.md#term-owner) and numeric protocols. The Scan contract alone owns complete Scan semantics;
Compiler, [Runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare)/[Engine](../../../../glossary.md#term-engine) and CPU own only their boundaries. Resolve the [metadata](../../../../glossary.md#term-metadata)-only
single-entry [memo](../../../../glossary.md#cache-and-memoization) exception explicitly, without changing current static behavior or claiming
[symbolic Scan](../../../../glossary.md#symbolic-scan), [memoization](../../../../glossary.md#term-memoization), [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training reuse or wider execution is implemented.

## Non-goals

No Java/[Javadoc](../../../../glossary.md#term-javadoc)/test-source/build/[API](../../../../glossary.md#term-application-programming-interface-api)/[ABI](../../../../glossary.md#term-application-binary-interface-abi)/native edits, [code generation](../../../../glossary.md#term-code-generation), benchmarks, [commits](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) or push.
No [module](../../../../glossary.md#term-module)/[dependency](../../../../glossary.md#term-dependency) change, second [graph](../../../../glossary.md#graph)/session/executor, new production type or public [signature](../../../../glossary.md#term-signature).
No static Scan [regression](../../../../glossary.md#term-regression), fixed [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn)/[GRU](../../../../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../../../../glossary.md#term-long-short-term-memory-lstm) [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt), length execution, variable T/[features](../../../../glossary.md#term-feature),
higher-order, low/wider [CPU](../../../../glossary.md#term-cpu) [dtype](../../../../glossary.md#data-type--datatype), native/mixed/[vector](../../../../glossary.md#term-vector)/parallel, cold/copy/wire or [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training enablement.
Do not retrospectively rewrite completed task briefs or ADR0050's historical evidence.

## Contracts

Read root [Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence),
[Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>,
[Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
[Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([normative glossary](../../../../glossary.md#term-normative))</span>,
[Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>,
[Documentation authority and architecture updates](../../../../../ARCHITECTURE.md#documentation-authority-and-architecture-updates)
and [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).

Owning scoped headings (all current clauses remain constraints except the explicit cutover):

- Scan: [Source body, attachments and scoped identities](../../../../architecture/contracts/recurrent-scan.md#source-body-attachments-and-scoped-identities)<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span>,
  [Canonical residual ports and demand](../../../../architecture/contracts/recurrent-scan.md#canonical-residual-ports-and-demand)<span data-glossary-navigation="true"> ([Canonical glossary](../../../../glossary.md#term-canonical-form); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand))</span>,
  [Reverse scan and exact primal bindings](../../../../architecture/contracts/recurrent-scan.md#reverse-scan-and-exact-primal-bindings)<span data-glossary-navigation="true"> ([primal glossary](../../../../glossary.md#term-primal); [bindings glossary](../../../../glossary.md#term-binding))</span>,
  [Traversal, empty domains and declared qualification](../../../../architecture/contracts/recurrent-scan.md#traversal-empty-domains-and-declared-qualification)<span data-glossary-navigation="true"> ([qualification glossary](../../../../glossary.md#term-qualification))</span>,
  [Qualification, performance and migration gates](../../../../architecture/contracts/recurrent-scan.md#qualification-performance-and-migration-gates).
- [Compiler](../../../../glossary.md#term-compiler): [Accepted symbolic-batch metadata binding](../../../../architecture/contracts/compiler-autograd.md#accepted-symbolic-batch-metadata-binding)<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [metadata glossary](../../../../glossary.md#term-metadata))</span>
  and [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([automatic differentiation glossary](../../../../glossary.md#autograd))</span>.
- [Runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare)/[Engine](../../../../glossary.md#term-engine): [Accepted symbolic-batch resource contract](../../../../architecture/contracts/runtime-prepare-engine.md#accepted-symbolic-batch-resource-contract),
  [`modules/prepare`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare),
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>,
  [Public reusable prepared execution](../../../../architecture/contracts/runtime-prepare-engine.md#public-reusable-prepared-execution)<span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>.
- [CPU](../../../../glossary.md#term-cpu): [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
  and [Accepted ordinary symbolic-batch CPU contract](../../../../architecture/contracts/backend-execution.md#accepted-ordinary-symbolic-batch-cpu-contract).

No foundational numerical or extension capability contract changes are selected. On an additional
missing/ambiguous [owner](../../../../glossary.md#term-owner), unapproved type/protocol or [scope](../../../../glossary.md#term-scope) expansion, stop and report it.

## Dependencies and integration

- Depends on: Complete Model0039 and Model0040 reviewed design deliveries; Complete Model0038,
  Compiler0013, Prepare0013 and CPU0037 retained source/AD/static/dynamic-ordinary baseline.<span data-glossary-navigation="true"> ([AD glossary](../../../../glossary.md#autograd))</span>
- Conflicts with: Scan/source/AD/resource/admission/identity semantics and all thirteen write paths;
  serialize these concerns, including shared authoritative documents.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Parallel group: None.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`
  plus accepted uncommitted baseline, not merged predecessors; index empty/unchanged.
- Integration order: sole clean authority writer → freeze → independent read-only Class C review
  → MAIN acceptance metadata and final document validation; no source launch.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [validation glossary](../../../../glossary.md#term-validation))</span>
- Integration validation: exact-scope baseline hash audit, local targets/anchors, terminology,
  status/authority/frontier/size checks and <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [hash glossary](../../../../glossary.md#term-hash); [targets glossary](../../../../glossary.md#term-target))</span>`git diff --check`; no executable behavior changed.
- Shared-document integration owner: MAIN; the thirteen paths below are exclusively delegated
  to the writer until freeze. Independent reviewer is read-only.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>

[Admission](../../../../glossary.md#term-admission) verified by MAIN on 2026-10-25: all six predecessor Status/Result headers Complete,
reviewed design deliveries accepted, IDs Model0041/ADR0051 free, HEAD/origin/index unchanged and
no competing writer. User confirmed the original plan after discussing static respecialization,
then explicitly said “ok, pokracuj tedy” following the stated contract-first next step.
This approves the bounded authority cutover only; no automatic Java successor is authorized.
[Snapshot](../../../../glossary.md#term-snapshot): `/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/pre-scan-symbolic-authority.json`.

## Files and symbols

Exclusive repository write [scope](../../../../glossary.md#term-scope) (thirteen files):

- `ARCHITECTURE.md` — [lifecycle](../../../../glossary.md#lifecycle)/scope routing and accepted-future status, no detailed duplication.
- `docs/architecture/contracts/recurrent-scan.md` — sole complete [symbolic Scan](../../../../glossary.md#symbolic-scan) semantics.
- `docs/architecture/contracts/compiler-autograd.md` — scoped [inference](../../../../glossary.md#term-inference)/actual [AD](../../../../glossary.md#autograd)/multi-output [binding](../../../../glossary.md#term-binding).
- `docs/architecture/contracts/runtime-prepare-engine.md` — symbolic projection, numeric-only sizing,
  actual publication and bounded immutable-geometry memo ownership/admission/lifecycle exception.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [immutable glossary](../../../../glossary.md#term-immutable); [memo glossary](../../../../glossary.md#cache-and-memoization); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [admission glossary](../../../../glossary.md#term-admission))</span>
- `docs/architecture/contracts/backend-execution.md` — bounded [CPU](../../../../glossary.md#term-cpu) resources/emission/[identity](../../../../glossary.md#term-identity-meanings) gates.
- `docs/architecture/runtime-prepare-backend-boundary.md` — focused conceptual workflow and rationale.
- `docs/design/decisions/0051-symbolic-scan-and-geometry-reuse.md` — new accepted rationale, not authority.
- `docs/glossary.md` — targeted symbolic-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)/Scan/[run](../../../../glossary.md#term-run)-geometry/prepared-[owner](../../../../glossary.md#term-owner) wording only.
- This brief, [Model](../../../../glossary.md#term-model) `master-plan.md`, `general-scan-symbolic-batch-proposal.md`,
  `docs/planning/dynamic-batch-execution.md`, `docs/planning/roadmap.md` — synchronized current state.

Inspect source blockers from the proposal's existing-owner map and current focused tests;
no source edits. In particular: Scan/ScanBody static guards, ScanBodyInference/ScanGradientRules,
BatchBindingPlan multi-output and per-call bind, [PrepareContext](../../../../glossary.md#prepare-context-preparecontext) projection, CpuGeneralScanIr/
[Lowering](../../../../glossary.md#lowering)/Emitter and provider [dispatch](../../../../glossary.md#term-dispatch), [Engine](../../../../glossary.md#term-engine) dynamic entry, [RunMemoryGeometry](../../../../glossary.md#run-memory-geometry--runmemorygeometry)/Test.
Proposed new/moved production types: zero; existing owner field/method changes and any necessary
exact [signatures](../../../../glossary.md#term-signature) remain later scoped-brief decisions, with the explicit type-approval stop gate.

## Acceptance criteria

- Add explicit accepted-but-unimplemented symbolic Scan qualification; preserve implemented static
  F32/F64, dense-read exceptions and fixed-scan restrictions. Never import CPU F32 limits into Model.<span data-glossary-navigation="true"> ([symbolic Scan glossary](../../../../glossary.md#symbolic-scan); [qualification glossary](../../../../glossary.md#term-qualification); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [CPU glossary](../../../../glossary.md#term-cpu); [Model glossary](../../../../glossary.md#term-model))</span>
- Define named-axis roles, unresolved template vs resolved actual layouts, every multi-output port,
  fixed parameter gradients, actual AD closure/count/K=B, scoped body obligations and exact saved sites.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [layouts glossary](../../../../glossary.md#layout); [output port glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [parameter glossary](../../../../glossary.md#term-parameter); [gradients glossary](../../../../glossary.md#term-gradient); [AD glossary](../../../../glossary.md#autograd))</span>
- Keep T=0 a separate fixed-T plan; B=0 never skips required shared work/parameter zeros or whole-body
  validity. Preserve constituent arithmetic and independent length-value preflight/unsupported gates.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight))</span>
- Reuse immutable numeric size/access recipes and exact-plan geometry with fixed routes/code/slots;
  piecewise checked alignment/empty scratch is allowed, not fake B or B_max/topology selection.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [alignment glossary](../../../../glossary.md#term-alignment); [topology glossary](../../../../glossary.md#term-topology))</span>
- Own one bounded metadata memo at exact Engine preparation entry, keyed by exact semantic/inward/
  memory-plan identity and B/mode; no Runtime lookup, per-B executable cache or new owner/protocol.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [memo glossary](../../../../glossary.md#cache-and-memoization); [Engine glossary](../../../../glossary.md#term-engine); [identity glossary](../../../../glossary.md#term-identity-meanings); [Runtime glossary](../../../../glossary.md#term-runtime); [lookup glossary](../../../../glossary.md#term-lookup); [cache glossary](../../../../glossary.md#term-cache); [owner glossary](../../../../glossary.md#term-owner))</span>
- Keep complete fresh Compiler bind/private obligations and all actual input/storage/physical checks.
  Cache only covered pure Engine geometry and template-derived non-input metadata; never Binding,
  supplied descriptors, storage/addresses/leases/results or shared mutable invocation arrays.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Binding glossary](../../../../glossary.md#term-binding); [descriptors glossary](../../../../glossary.md#term-descriptor); [leases glossary](../../../../glossary.md#term-lease); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- Explicitly distinguish reusable encapsulated immutable numeric facts from mutable run-owned arrays;
  prepared meaning remains immutable. Require atomic complete-entry publication, local snapshots,
  bounded active candidates, failure/retry and close coordination without an execution lock/second lease.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [publication glossary](../../../../glossary.md#term-publication))</span>
- Preserve generated clean-Java algorithm/hot-loop/dataflow, static keys/bytes, runtime-B exclusion
  from generated compatibility identity and independent actual reverse qualification. No speed promise.<span data-glossary-navigation="true"> ([compatibility identity glossary](../../../../glossary.md#identity-meanings-and-algebraic-identities))</span><span data-glossary-navigation="true"> ([dataflow glossary](../../../../glossary.md#term-dataflow))</span>
- ADR compares status quo, per-B static respecialization and selected numeric binding/reuse; explanation
  shows a labeled future Scan example and 64→64→40→40→64 miss/hit witness with fresh resources.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- Future executable/conformance/integration/structure/bounded-performance obligations are explicit;
  no new dependency tests needed now because dependency rules are unchanged. Independent review has
  zero open blocking findings, and only then MAIN accepts the authority cutover.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker: targeted documentation/contract consistency and local links/anchors, glossary search,
brief <=200 lines/15 KB, `git diff --check` and exact-thirteen-path audit against the [snapshot](../../../../glossary.md#term-snapshot).
MAIN supplies a dedicated read-only document validator during drafting. Reuse existing executable
evidence only as retained-baseline evidence, never as dynamic Scan/[memo](../../../../glossary.md#cache-and-memoization) [qualification](../../../../glossary.md#term-qualification).
No [Gradle](../../../../glossary.md#term-gradle)/Java/native/performance [run](../../../../glossary.md#term-run) is needed for a document-only future permission with unchanged
dependencies/[API](../../../../glossary.md#term-application-programming-interface-api)/executable behavior; later implementation owns matching architecture/[conformance](../../../../glossary.md#term-conformance)/
integration and generated-code tests. No test or [Javadoc](../../../../glossary.md#term-javadoc) is claimed newly validated here.

## Documentation and review impact

Apply general, architecture, decision-record, [planning](../../../../glossary.md#term-planning) and conceptual-example profiles according
to each file type. Review affected guide/[Javadoc](../../../../glossary.md#term-javadoc)/glossary wording; do not broaden guides to imply
unimplemented support. Exact future Javadoc obligations stay with source implementation.
Independent clean targeted Class C documentation/contract review is mandatory before completion.

## Result

Authority/docs enacted across exactly the thirteen paths above: sole Scan semantics, linked boundary
[owners](../../../../glossary.md#term-owner), [immutable](../../../../glossary.md#term-immutable)-only [Engine](../../../../glossary.md#term-engine) [memo](../../../../glossary.md#cache-and-memoization) exception, ADR0051, glossary, future example and synchronized plans.
The accepted direction is unimplemented; current static/source/[AD](../../../../glossary.md#autograd)/[CPU](../../../../glossary.md#term-cpu) distinctions and [identities](../../../../glossary.md#term-identity-meanings) remain.
Documentation profiles: general, architecture, decision-record, [planning](../../../../glossary.md#term-planning) and conceptual examples.
[Javadoc](../../../../glossary.md#term-javadoc) impact: no Java/Javadoc edited or generated; future implementation must update affected contracts.
Review-stage [validation](../../../../glossary.md#term-validation): `node /private/tmp/synaptik-scan-authority.ODRQ6j/check-authority.mjs` PASS
(review mode): 908 local links/anchors, 71 added links, zero failures; exact thirteen-path baseline audit,
preserved authoritative anchors, unchanged root incorporation/dependencies, HEAD/origin/index and
`git diff --check` PASS. Brief remains below 200 lines/15 KB. Targeted glossary/terminology and
conceptual byte/hit witness reviewed. No [Gradle](../../../../glossary.md#term-gradle)/Java/Javadoc/native/performance [run](../../../../glossary.md#term-run); no executable claim.
Independent read-only Class C review: APPROVE, zero open P1/P2. One P2 wording finding was fixed:
CPU-int representability bounds apply to the initial CPU domain, not future other [backends](../../../../glossary.md#backend).
V2 differed from the first freeze only in that scoped paragraph; reviewer verified all thirteen
[hashes](../../../../glossary.md#term-hash)/copies and cleared the finding. MAIN accepted 2026-10-25; subsequent edits are acceptance
[metadata](../../../../glossary.md#term-metadata) only. No production type/public [signature](../../../../glossary.md#term-signature), source change or [dependency](../../../../glossary.md#term-dependency) change.
Final validation: `node /private/tmp/synaptik-scan-authority.ODRQ6j/check-authority.mjs --final` —
PASS: links/anchors, status/authority/size/whitespace, exact thirteen-path [hash](../../../../glossary.md#term-hash) [scope](../../../../glossary.md#term-scope) and protected
HEAD/origin/index/outside files; unchanged root incorporation/dependencies and retained anchors.
Authority frontier: None. Implementation frontier: None. ImplementationAuthorized=false.
Next gate is a separately [Ready](../../../../glossary.md#term-ready) [Model](../../../../glossary.md#term-model) source foundation, then [Compiler](../../../../glossary.md#term-compiler) and the numeric vertical;
no automatic launch, executable capability, source/test/native/performance run, [commit](../../../../glossary.md#term-commit) or push.
Status: Complete

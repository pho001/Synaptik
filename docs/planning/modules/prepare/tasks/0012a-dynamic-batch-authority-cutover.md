# Task Prepare0012A: Dynamic [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) authority cutover

> **Editorial historical reading key.**
>
> This document records authority enactment between design acceptance and Java implementation. Its
> nine-type/signature map and ADR packet changed normative documentation, not executable APIs. The
> ownership chain remains explicit: Compiler binds semantic metadata, Prepare preserves declarations,
> CPU supplies numeric recipes, Runtime owns invocation geometry/resources, and Engine publishes
> actual descriptors. Batch-required mode remains meaningful even when payload bytes are constant.
>
> The [current metadata
> reference](../../../../reference/compiler/batch-binding.md#ordinary-named-batch-metadata-binding)
> describes storage-free binding; the [current Engine
> lifecycle](../../../../reference/engine/lifecycle.md#current-ordinary-explicit-composition-and-advanced-lifecycle)
> describes subsequently qualified execution. Caller storage is borrowed until result closure; closing
> preparation stops admissions without revoking returned results, and detached copies have independent
> lifetimes. The historical unsupported-Java statement and proposed ordinary witness must remain dated
> to this cutover. Documentation approval was not numerical, concurrency or timing evidence, nor a
> broader workflow grant.

## Status

Complete — authority enacted and MAIN-accepted 2026-10-24; Pasteur definitive APPROVE, zero open P1/P2.
User architecture approval recorded 2026-10-24: “ano” accepts the reviewed contract, architecture
first, then ordinary [CPU](../../../../glossary.md#term-cpu) forward/actual first-order backward and bounded final A/B timing.
Exact [signatures](../../../../glossary.md#term-signature)/nine-type map preserved; current Java dynamic [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) unsupported. Implementation
approval exists; no [Ready](../../../../glossary.md#term-ready) Java brief until MAIN admits planner output, no automatic successor.

## Change class

Class C — coordinated [Runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare) sizing, [ownership](../../../../glossary.md#term-ownership-meanings), [admission](../../../../glossary.md#term-admission) and [publication](../../../../glossary.md#term-publication) contract.
Independent targeted authority/documentation review is mandatory; design review is already accepted.

## Goal

Enact the accepted [Prepare0012 proposal](../dynamic-batch-geometry-contract-proposal.md) as one
coherent root/scoped-contract, [architecture decision record](../../../../glossary.md#term-architecture-decision-record-adr) ([ADR](../../../../glossary.md#adr-ci-and-qualification)) and explanation cutover.
Preserve its exact [signatures](../../../../glossary.md#term-signature)/nine-type map; neither reopen design nor invent another [API](../../../../glossary.md#term-application-programming-interface-api).

## [Scope](../../../../glossary.md#term-scope)

- Apply the proposal's `Exact proposed authority and ADR packet`, `Exact proposed signatures and
  semantics` and `Complete type and replacement map` to their existing authority [owners](../../../../glossary.md#term-owner).
- Distinguish accepted future protocol from current fixed-only implementation. Preserve static
  admission, ownership, empty semantics, all-valid/length Scan and persistence compatibility.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Scan glossary](../../../../glossary.md#general-differentiable-scan); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- Record Compiler semantic binding → Prepare stable declarations → CPU numeric recipes →
  Runtime run-owned geometry/resources → Engine actual-descriptor publication.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [binding glossary](../../../../glossary.md#term-binding); [Prepare glossary](../../../../glossary.md#prepare); [CPU glossary](../../../../glossary.md#term-cpu); [Runtime glossary](../../../../glossary.md#term-runtime); [run glossary](../../../../glossary.md#term-run); [Engine glossary](../../../../glossary.md#term-engine); [descriptor glossary](../../../../glossary.md#term-descriptor); [publication glossary](../../../../glossary.md#term-publication))</span>
- Synchronize only the exact explanations, glossary entries and six planning files listed below.
  Reserve ADR0050 (currently free); recheck before creation and stop if occupied.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Non-goals

No Java/test/build/[dependency](../../../../glossary.md#term-dependency)/native/generated [artifact](../../../../glossary.md#term-artifact) changes, [Javadoc](../../../../glossary.md#term-javadoc) edits, builds, executable
tests or timing here. No [commit](../../../../glossary.md#term-commit)/push, own agents, design reopening, new [API](../../../../glossary.md#term-application-programming-interface-api)/type, new execution
[owner](../../../../glossary.md#term-owner), [runtime](../../../../glossary.md#term-runtime) [graph](../../../../glossary.md#graph) interpreter, B_max, per-B fullprepare [cache](../../../../glossary.md#term-cache) or prepared mutation.
No automatic Java/successor launch, [Scan](../../../../glossary.md#general-differentiable-scan)/[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training/cold/wire/native/mixed capability grant.

## Contracts

Start at the root [scope](../../../../glossary.md#term-scope) index. Current restrictions are replaced only by this admitted coordinated
cutover; the proposal and this brief cannot independently override authority.

- Root: [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>,
  [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([normative glossary](../../../../glossary.md#term-normative))</span>,
  [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>,
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span> (edges unchanged).
- [Runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare)/[Engine](../../../../glossary.md#term-engine): [`modules/runtime`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesruntime),
  [`modules/prepare`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare),
  [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle),
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>,
  [Public reusable prepared execution](../../../../architecture/contracts/runtime-prepare-engine.md#public-reusable-prepared-execution)<span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>.
- [Compiler](../../../../glossary.md#term-compiler): [Compile artifacts](../../../../architecture/contracts/compiler-autograd.md#compile-artifacts)<span data-glossary-navigation="true"> ([Compile artifacts glossary](../../../../glossary.md#compile-artifacts))</span>,
  [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([automatic differentiation glossary](../../../../glossary.md#autograd))</span>.
- [Backend](../../../../glossary.md#backend): [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>.
- [Scan](../../../../glossary.md#general-differentiable-scan): [General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)<span data-glossary-navigation="true"> ([General differentiable scan glossary](../../../../glossary.md#general-differentiable-scan))</span>,
  [Length-aware whole-batch scan](../../../../architecture/contracts/recurrent-scan.md#length-aware-whole-batch-scan)<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>.
- Extensions: [Owned tensor Models](../../../../architecture/contracts/extensions-training.md#owned-tensor-models)<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor))</span>,
  [`modules/training`](../../../../architecture/contracts/extensions-training.md#modulestraining),
  [Generic graph source and values](../../../../architecture/contracts/extensions-training.md#generic-graph-source-and-values)<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>.

If another applicable rule conflicts or the accepted map is ambiguous, stop and report the exact
conflict. Do not widen authority [ownership](../../../../glossary.md#term-ownership-meanings) or change the accepted design silently.

## Dependencies and integration

- Depends on: [Prepare0011](0011-length-aware-scan-projection-and-resource-handoff.md#result),
  [Runtime0016](../../runtime/tasks/0016-persistent-prepared-resource-lifecycle.md),
  [Engine0021](../../engine/tasks/0021-prepared-execution-fresh-value-binding.md#result),
  [Compiler0013](../../compiler/tasks/0013-length-aware-scan-capture-and-first-order-ad.md#result),
  [CPU0037](../../../backends/cpu/tasks/0037-general-scan-dense-read-views.md#result), and
  [Prepare0012 design](0012-dynamic-batch-geometry-and-resource-contract.md#result): all Complete.
- Conflicts with: all geometry/declaration/creator/publication, CPU binder/resource ABI and shared
  authority/planning writes; serial writer, no competing repository writer active.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [CPU glossary](../../../../glossary.md#term-cpu); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- Parallel group: None.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus accepted
  dirty tree, not merged predecessors; MAIN baseline has 2435 files at
  `/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/baseline.json`.
- Integration order: Ready brief → explicit MAIN admission → same-context authority cutover →
  independent Class C authority/docs review → MAIN docs checkpoint/acceptance → separately
  admitted ordinary vertical implementation. User approval does not bypass these gates.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Integration validation: MAIN docs-only link/fragment, accepted-map, whitespace and baseline
  protection checkpoint; no Java/build/native/performance validation in this task.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): MAIN; this serial context writes only the admitted list.

[Admission](../../../../glossary.md#term-admission) facts: MAIN reverified source Status/Result of all five dependencies Complete, plus
design Complete, at **2026-10-24 17:41:11 UTC**; HEAD/origin and index unchanged. MAIN captured
2435 baseline files before writer changes. Independent design Class C APPROVE, zero open P1/P2,
and design delivery acceptance are retained. User “ano” explicitly approves architecture cutover,
subsequent implementation and bounded final A/B. Planner confirms the stable contract/type map,
free ADR0050 and serial [scope](../../../../glossary.md#term-scope). MAIN inspected this [Ready](../../../../glossary.md#term-ready) brief, reverified the six Complete
dependencies, exact approved map, unchanged HEAD/origin/index and six-file [planning](../../../../glossary.md#term-planning)-only delta,
and explicitly admitted the sole serial authority/docs launch on 2026-10-24. No competing writer
is active. Java implementation remains held until independent authority review and MAIN acceptance.

## Files and symbols

Exact post-[admission](../../../../glossary.md#term-admission) authority/explanation write [scope](../../../../glossary.md#term-scope) (no production symbols created):

- `ARCHITECTURE.md` — Core [lifecycle](../../../../glossary.md#lifecycle) and scope-index routing only; [dependency](../../../../glossary.md#term-dependency) edges unchanged.
- `docs/architecture/contracts/runtime-prepare-engine.md` — the five headings above; exact sizing,
  explicit batch-required mode, same-plan geometry, creator inputs, admission/publication/lifetime.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [publication glossary](../../../../glossary.md#term-publication); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- `docs/architecture/contracts/compiler-autograd.md` — the two headings above; BatchBindingPlan,
  private typed obligations, eight CompileArtifacts components and actual first-order closure.
- `docs/architecture/contracts/backend-execution.md` — [CPU](../../../../glossary.md#term-cpu) [backend](../../../../glossary.md#backend) routes; prepared numeric [binding](../../../../glossary.md#term-binding),
  `ordinary-batch-v1`, retained static [identities](../../../../glossary.md#term-identity-meanings) and fixed-five-[limb](../../../../glossary.md#exact-superaccumulator-and-limbs) [FLOAT32](../../../../glossary.md#term-float32) exact SUM.
- `docs/architecture/contracts/recurrent-scan.md` — the two headings above; separate symbolic-B
  qualification, fixed T/features, shape versus producer-ordered length-value preflight.<span data-glossary-navigation="true"> ([qualification glossary](../../../../glossary.md#term-qualification); [features glossary](../../../../glossary.md#term-feature); [shape glossary](../../../../glossary.md#shape); [producer glossary](../../../../glossary.md#term-producer); [preflight glossary](../../../../glossary.md#term-preflight))</span>
- `docs/architecture/contracts/extensions-training.md` — the three headings above; separately
  gated one-state reuse, cold/builder/wire acquisition and retained static support.
- `docs/design/decisions/0050-dynamic-batch-geometry.md` — new accepted [ADR](../../../../glossary.md#term-architecture-decision-record-adr) from reviewed packet.
- `docs/architecture/runtime-prepare-backend-boundary.md` — focused future sizing/binding explanation.
- `docs/architecture/module-boundaries.md` — same [owners](../../../../glossary.md#term-owner) and unchanged edges; future [handoff](../../../../glossary.md#term-handoff).
- `docs/glossary.md` — targeted [dimension](../../../../glossary.md#dimension)-binding/prepared-memory/representation/[run](../../../../glossary.md#term-run)-state terms only.

[Planning](../../../../glossary.md#term-planning)/status scope, both at [Ready](../../../../glossary.md#term-ready) preparation and after acceptance:

- This brief; `docs/planning/modules/prepare/master-plan.md`; `docs/planning/roadmap.md`;
  `docs/planning/dynamic-batch-execution.md` — current frontier, status and navigation only.
- `docs/planning/modules/prepare/dynamic-batch-geometry-contract-proposal.md` — Purpose/status and
  Acceptance/next gate only; preserve reviewed design/signature/type/ADR packet.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>
- `docs/planning/modules/prepare/tasks/0012-dynamic-batch-geometry-and-resource-contract.md` — current
  Status, Documentation and Result follow-up only; preserve historical admission evidence.

## Acceptance criteria

1. Root remains sole index; each of five affected scoped owners carries only its concern. Enact
   the reviewed nine-type/signature map without changing dependency direction or current support claims.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools))</span>
2. Exact fixed/[run](../../../../glossary.md#term-run)-size declarations, explicit requiresBatchSize even for `X[B,0]`, authenticated
   same-plan geometry, exact-byte/alignment creators, static overload/accessor/SAM migration and
   pre-borrow checked admission/publication are coherent. No fake sizes or unsafe dynamic adapters.<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment); [admission glossary](../../../../glossary.md#term-admission); [publication glossary](../../../../glossary.md#term-publication))</span>
3. Preserve B=0 under operation predicates, representability/overflow and conditional 0/48-byte
   scratch, constant-five-limb F32 oracle, run isolation/leases/cleanup and static compatibility.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [overflow glossary](../../../../glossary.md#term-overflow); [constant glossary](../../../../glossary.md#term-constant); [limb glossary](../../../../glossary.md#exact-superaccumulator-and-limbs); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [oracle glossary](../../../../glossary.md#term-test-oracle); [leases glossary](../../../../glossary.md#term-lease))</span>
4. Distinguish accepted protocol from implemented capability everywhere edited. Preserve all-B
   produced-length pre-write checks, fixed T and separate Scan/NN/Training/cold/wire/native gates.<span data-glossary-navigation="true"> ([Scan glossary](../../../../glossary.md#general-differentiable-scan); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
5. ADR and two explanations/glossary agree with the accepted packet; independent Class C review
   has no unresolved P1/P2. MAIN accepts the docs-only checkpoint before any Java frontier.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker: compare final scoped text with the accepted [signatures](../../../../glossary.md#term-signature)/type map and existing named headings;
check local links/fragments, terminology, conceptual labels, brief size and whitespace:

```bash
node /private/tmp/synaptik-dynamic-batch-plan.ImRhfs/check-links.mjs \
  ARCHITECTURE.md docs/architecture/contracts/{runtime-prepare-engine,compiler-autograd,backend-execution,recurrent-scan,extensions-training}.md \
  docs/architecture/{runtime-prepare-backend-boundary,module-boundaries}.md docs/glossary.md \
  docs/design/decisions/0050-dynamic-batch-geometry.md docs/planning/{roadmap,dynamic-batch-execution}.md \
  docs/planning/modules/prepare/{master-plan,dynamic-batch-geometry-contract-proposal}.md \
  docs/planning/modules/prepare/tasks/{0012-dynamic-batch-geometry-and-resource-contract,0012a-dynamic-batch-authority-cutover}.md
wc -l -c docs/planning/modules/prepare/tasks/0012a-dynamic-batch-authority-cutover.md
git diff --check -- ARCHITECTURE.md docs/architecture/contracts docs/architecture/module-boundaries.md docs/architecture/runtime-prepare-backend-boundary.md docs/glossary.md docs/planning/modules/prepare docs/planning/roadmap.md docs/planning/dynamic-batch-execution.md
node /private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/audit.mjs
```

Check new [ADR](../../../../glossary.md#term-architecture-decision-record-adr)/brief whitespace separately; validate every edited Markdown file's links/fragments,
extending the checker input list if needed without repository writes. Audit must show only the exact
admitted list changed from MAIN baseline, unchanged HEAD/origin/index; never compare all dirty HEAD
changes as this task. MAIN owns final [checkpoint](../../../../glossary.md#term-checkpoint); independent review reuses successful evidence.
Architecture/[conformance](../../../../glossary.md#term-conformance)/integration executable checks belong to the later Java join because no
[dependency](../../../../glossary.md#term-dependency) or executable change occurs here. No [Gradle](../../../../glossary.md#term-gradle)/[Javadoc](../../../../glossary.md#term-javadoc)/native/benchmark runs.

## Follow-up: exact ordinary vertical gate

After authority acceptance, MAIN admits a separate compact Java brief: genuine named symbolic B,
[homogeneous](../../../../glossary.md#term-homogeneous) [FLOAT32](../../../../glossary.md#term-float32) [CPU](../../../../glossary.md#term-cpu), fixed [features](../../../../glossary.md#term-feature)/[parameters](../../../../glossary.md#term-parameter), [rank](../../../../glossary.md#term-rank)-at-most-two boundaries plus required [AD](../../../../glossary.md#autograd)
[views](../../../../glossary.md#term-view-meanings). [Compile](../../../../glossary.md#compile)/[prepare](../../../../glossary.md#prepare) once: grad-eligible X `[B,4]`, W `[4,3]`, Y=X@W, L=[mean](../../../../glossary.md#term-mean)(Y), actual dX/dW
from L/[scalar](../../../../glossary.md#term-scalar)-one [seed](../../../../glossary.md#term-seed); [run](../../../../glossary.md#term-run) 64→40→1→64 with all-one inputs and no updates. Prove Y=L=4,
dX=1/B, dW=1/3 under existing [rounding](../../../../glossary.md#term-rounding), actual [descriptors](../../../../glossary.md#term-descriptor)/bytes, K=B adjoints, unchanged code/
routes/slots, fresh [run state](../../../../glossary.md#run-state--runstate); qualify the entire actual AD closure, not forward-only answers.
Include B=0/1, [constant](../../../../glossary.md#term-constant)-byte [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-required rejection, mismatch/[overflow](../../../../glossary.md#term-overflow)/storage/failure/cleanup/
retry/[concurrency](../../../../glossary.md#term-concurrency)/static regressions and focused/[module](../../../../glossary.md#term-module) plus MAIN architecture/[conformance](../../../../glossary.md#term-conformance)/
integration join. Compare optimal specialized clean Java and generated Class-Files/hot-loop/[dataflow](../../../../glossary.md#term-dataflow);
bounded warmed matched-work A/B uses invocation B on both sides, separates cold generation/classload/
[JIT](../../../../glossary.md#term-just-in-time-compilation-jit), [admission](../../../../glossary.md#term-admission) and [kernel](../../../../glossary.md#kernel) costs, and diagnoses overhead without a zero-penalty/JIT claim.
This ordinary gate precedes [general Scan](../../../../glossary.md#general-differentiable-scan); [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training one-state reuse and cold/wire/builder/native/
mixed/tuning remain separate. Static length skipping is independent, not held on distant NN work.

## Documentation and review impact

[Planning](../../../../glossary.md#term-planning)/general profiles apply now; after [admission](../../../../glossary.md#term-admission) also architecture/[ADR](../../../../glossary.md#term-architecture-decision-record-adr) profiles and conceptual
example format. MAIN arranges independent clean Class C authority/docs review; no own agents.
Documentation impact: coordinated future contract/ADR/explanations/glossary only. [Javadoc](../../../../glossary.md#term-javadoc) impact:
none here; no Java changes, existing implementation Javadoc remains accurate for current behavior.
Later Java task finalizes every affected [signature](../../../../glossary.md#term-signature)/[invariant](../../../../glossary.md#term-invariant)'s Javadoc. No [API](../../../../glossary.md#term-application-programming-interface-api) guide support claim.

## Result

AUTHOR FROZEN: root + five scoped [owners](../../../../glossary.md#term-owner), ADR0050, two explanations, targeted glossary and six
[planning](../../../../glossary.md#term-planning) files updated; exactly 16 admitted paths. Approved [signatures](../../../../glossary.md#term-signature)/nine-type map unchanged;
current fixed-only truth, static paths/keys and actual-B-free dynamic [identities](../../../../glossary.md#term-identity-meanings) retained.
[Validation](../../../../glossary.md#term-validation) commands above: links PASS (16 files, 794 links/532 fragments); scoped diff/whitespace
PASS; protection PASS (only 16 paths, HEAD/origin/index unchanged); brief within 200 lines/15 KB.
Documentation impact: authority/[ADR](../../../../glossary.md#term-architecture-decision-record-adr)/explanations/glossary/planning. [Javadoc](../../../../glossary.md#term-javadoc) impact: none, no Java edits.
No builds/native/timing/[commit](../../../../glossary.md#term-commit)/push. Pasteur definitive APPROVE, zero open P1/P2; docs/glossary
review complete, both P2 fixes verified. MAIN inspected [owner](../../../../glossary.md#term-owner) blocks/fixes and accepted 2026-10-24.
Final [metadata](../../../../glossary.md#term-metadata) links/whitespace/protection PASS: 7 files, 275 links/85 fragments; contract bodies unchanged.
Status: Complete
Next gate: MAIN separately admits ordinary Java planner output; no automatically runnable successor.

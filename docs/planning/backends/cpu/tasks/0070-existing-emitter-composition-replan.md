# CPU0070: Replan composition of existing [operation](../../../../glossary.md#operation) emitters

## Reader context

The replacement plan distinguishes three binding spaces. CodeBuilder allocates JVM local
indexes while generating a method; emitter callers select operands and temporal addresses
while composing operation regions; a later invocation supplies numeric geometry and actual
storage. These are not three names for Runtime memory slots. Scan can surround a common
arithmetic region with activity/history work without invoking a nested prepared executable.
Non-fused regions still need declared intermediate storage and completion before consumption.

Read [common MATMUL binding](../../../../backend-guide/cpu-backend.md#common-rank-two-matmul-binding)
for the concrete caller/owner split and a non-fused intermediate example, and
[common pointwise/SUM composition](../../../../backend-guide/cpu-backend.md#common-pointwise-and-sum-composition)
for later shared ownership without numerical unification. Completion below means accepted
planning delivery, not permission to launch its successor commands or measurements.

## Status and goal

Complete for [planning](../../../../glossary.md#term-planning) delivery; independent APPROVE and MAIN acceptance on 2026-10-29.
At launch this was the verified [Ready](../../../../glossary.md#term-ready) planning-only frontier. User authorization on 2026-10-29 replaces the CPU0066 three-stage
[scalar](../../../../glossary.md#term-scalar)/[vector](../../../../glossary.md#term-vector)/worker proposal with composition of the existing standard [operation](../../../../glossary.md#operation) implementations.
MAIN verified CPU0064/0065 and CPU0066 planning delivery Complete, no authorized source writer,
HEAD=origin `07a281edd3086efde2853211b71b48288db027e1` and empty index. Preserve the accepted
dirty tree (431 changed/untracked paths); the [snapshot](../../../../glossary.md#term-snapshot) is outside the repository.
This Class C design uses a clean planner and independent targeted review. No Java, build,
benchmark, [normative](../../../../glossary.md#term-normative) contract, [dependency](../../../../glossary.md#term-dependency), [commit](../../../../glossary.md#term-commit) or push change is authorized.

Plan one [owner](../../../../glossary.md#term-owner) for each operation's scalar/vector/tiled mathematics, used by ordinary [kernels](../../../../glossary.md#kernel)
and Scan. Scan emits traversal, step/state/activity/history [bindings](../../../../glossary.md#term-binding) and transitions around
existing operation regions. Separate [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) local allocation, generation-time operand bindings and
numeric invocation geometry. Reuse CodeBuilder allocation rather than a second generic slot
manager; do not prescribe new types unless source proves their need.

## Contracts and source packet

Read AGENTS, [planning](../../../../glossary.md#term-planning) guide, roadmap/current [CPU](../../../../glossary.md#term-cpu) frontier and documentation rules plus General,
Planning and Example profiles. Root [Core invariants](../../../../../ARCHITECTURE.md#core-invariants)
and [scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span> route to [backend](../../../../glossary.md#backend)
[CPU routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes), Scan
[General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)<span data-glossary-navigation="true"> ([General differentiable scan glossary](../../../../glossary.md#general-differentiable-scan))</span>,
[Actual first-order closure and arithmetic](../../../../architecture/contracts/recurrent-scan.md#actual-first-order-closure-and-arithmetic),
[Qualification and mechanism gates](../../../../architecture/contracts/recurrent-scan.md#qualification-and-mechanism-gates)<span data-glossary-navigation="true"> ([Qualification glossary](../../../../glossary.md#term-qualification))</span>,
[Activity, raw identity and retained histories](../../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories)<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
and [Runtime](../../../../glossary.md#term-runtime) [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>.
Read additional exact [owner](../../../../glossary.md#term-owner) headings only for boundaries the source proves relevant.
Inspect CpuClassFileKernelGenerator, ordinary family emitters/lowerers/[IR](../../../../glossary.md#term-intermediate-representation-ir), CpuCarrierEmitter,
CpuGeneralScanLowering/IR/emitter, its math subemitters, preparer/finalizer, both prepared
executables, [CpuWorkerGroup](../../../../glossary.md#cpu-worker-group--cpuworkergroup) and focused tests. CPU0066 proposal is supersession input, not authority.

## Dependencies and integration

- Depends on: accepted CPU0064/0065 source and retained Compiler0015/0016/Prepare0015 boundaries;
  CPU0066 is completed planning history, not implementation.
- Conflicts with: [CPU](../../../../glossary.md#term-cpu) emission, Scan, geometry/[cache](../../../../glossary.md#term-cache), workers and CPU0048–0052 writers.
- Parallel group: None; one clean [planning](../../../../glossary.md#term-planning) writer, then independent read-only review.
- Common base revision: HEAD above plus preserved accepted dirty work; [snapshot](../../../../glossary.md#term-snapshot) cpu0070-start.
- Integration order: planning draft → independent targeted review → fixes/freeze → MAIN acceptance.
- Integration validation: source citations, contract/DAG/frontier consistency, links/anchors,
  brief sizes, whitespace and changed-file/hash preservation; no JVM/test/benchmark launch.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [hash glossary](../../../../glossary.md#term-hash); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm))</span>
- Shared-document integration owner: MAIN; planner receives exclusive writes to the old proposal,
  CPU0066–0068 status annotations, this brief, new CPU planning deliverables, CPU master and its
  focused roadmap entries until handoff. MAIN does not edit those concurrently.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [handoff glossary](../../../../glossary.md#term-handoff))</span>

## Deliverable and acceptance

Replace the active proposal; annotate superseded successors and preserve completed historical
results. Inventory all current standard generated/direct-delegating [operation](../../../../glossary.md#operation) families, grouping
their actual operand-[binding](../../../../glossary.md#term-binding), setup, resource and execution seams. Do not claim all are presently
Scan-capable. Give minimal cohesive migration tasks rather than [scalar](../../../../glossary.md#term-scalar)/[vector](../../../../glossary.md#term-vector)/parallel reimplementations;
detail only the first actionable capability/foundation and stable immediate successor.
Show before/after ordinary versus Scan binding and a non-[fused](../../../../glossary.md#specialization-and-fusion) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)→operation boundary: declared
reusable intermediate, no mandatory extra copy, full [dependency](../../../../glossary.md#term-dependency) completion and preserved saved sites.
Reuse existing [SIMD](../../../../glossary.md#simd)/tiled math and standard worker/range infrastructure. Source must distinguish
emission from CpuPreparedExecutable's worker orchestration; no automatic threading by inline emission.
Do not mandate a new PhaseState/barrier protocol, once-per-Scan worker cohort, nested executable/[run](../../../../glossary.md#term-run),
per-step prepared [callback](../../../../glossary.md#term-callback), [runtime](../../../../glossary.md#term-runtime) interpreter, new pool, public [API](../../../../glossary.md#term-application-programming-interface-api) or second [DAG](../../../../glossary.md#term-directed-acyclic-graph-dag). State genuine
parallel integration/authority decisions honestly and leave unproved execution Draft.
Preserve exact versus represented-sequential SUM as distinct existing algorithms under common
[ownership](../../../../glossary.md#term-ownership-meanings), current static/symbolic/length [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) forward/actual [AD](../../../../glossary.md#autograd) [scope](../../../../glossary.md#term-scope) and v9 code-shape gains.
New types/[signatures](../../../../glossary.md#term-signature) need exact justification and existing user approval gates; no old four-type
mandate or diagnostic budget carries over. No measurement is authorized by this [planning](../../../../glossary.md#term-planning) task.
Report superseded/deleted versus retained code, limitations, review/doc impact, exact checks and
Status: Complete or Status: Incomplete with specific follow-up.

## Result

- Clean writer delivery and independent targeted review are complete, 2026-10-29. The
  [replacement plan](../existing-emitter-composition-plan.md) inventories all standard generated/
  direct-delegating families in grouped seams, distinguishes the three binding spaces, and
  details setup dominance, a non-fused boundary, activity/saved/lifetime proofs and v9 retention.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [fused glossary](../../../../glossary.md#specialization-and-fusion); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- Draft [CPU0071](0071-common-matmul-binding-foundation.md) is the common MATMUL foundation;
  Draft <span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>[CPU0072](0072-common-pointwise-sum-composition.md) closes current pointwise/SUM reuse.
  No new type/package/signature mandate or scalar/vector/worker algorithm cascade. Scan SIMD
  needs bounded authority/whole-binder qualification; parallel composition remains a genuine
  unresolved design/authority gate, not a selected phase protocol or required cohort optimization.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [signature glossary](../../../../glossary.md#term-signature); [scalar glossary](../../../../glossary.md#term-scalar); [vector glossary](../../../../glossary.md#term-vector); [SIMD glossary](../../../../glossary.md#simd); [qualification glossary](../../../../glossary.md#term-qualification))</span>
- Changed: this brief, old proposal archive notice, CPU0066–0068 supersession annotations,
  focused CPU master/roadmap entries. Created: replacement plan and CPU0071/0072 briefs.
  CPU0066's completed Result/review is preserved. No executable code deleted; proposed math
  subemitter retirements are conditional on each family's future qualification.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- Validation: MAIN Node local-target/heading-anchor/trailing-whitespace check across all
  10 owned files PASS (524 local links); all six briefs below 200 lines/15 KB; scoped<span data-glossary-navigation="true"> ([Validation glossary](../../../../glossary.md#term-validation); [Node glossary](../../../../glossary.md#node); [target glossary](../../../../glossary.md#term-target))</span>
  `git diff --check` PASS. Planner's focused checks also passed.
  Manual source-dispatch/matrix, contract/DAG/status and authority-gate checks completed.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag))</span>
- Preservation: MAIN SHA-256 comparison against the 2514-file cpu0070-start snapshot finds
  exactly 6 owned planning files changed and 4 planning files added; no outside-scope change
  or removal. <span data-glossary-navigation="true"> ([SHA-256 glossary](../../../../glossary.md#term-sha-256); [snapshot glossary](../../../../glossary.md#term-snapshot); [planning glossary](../../../../glossary.md#term-planning); [scope glossary](../../../../glossary.md#term-scope))</span>`git rev-parse
  HEAD refs/remotes/origin/main` remains the recorded base; `git diff --cached --name-only`
  remains empty. No Git mutation, Java/JVM/build/test/benchmark/native or contract edit occurred.<span data-glossary-navigation="true"> ([JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm))</span>
- Documentation profiles: Planning, General and Example. Targeted glossary search found no
  changed domain meaning; glossary and Java/Javadocs remain unchanged because this is planning only.
- Independent read-only final review: APPROVE, 0 P1/P2 and no unresolved findings; MAIN accepted
  the planning delivery. Review reused passing documentation/preservation checks; no Java tests.
- Status: Complete. Successor Ready/implementation, new-type, capability-authority and measurement
  approvals remain separate future gates, not incomplete planning acceptance.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>

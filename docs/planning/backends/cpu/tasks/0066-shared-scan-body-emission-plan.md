# CPU0066: Plan shared ordinary and Scan body emission

> Historical completed planning delivery preserved. CPU0070 replaces its active direction with
> [existing-emitter composition](../existing-emitter-composition-plan.md); CPU0067/0068 and the
> CPU0069 outline are superseded. The old type/diagnostic/worker proposals are not current gates.
> The original Result and independent approval below remain historical evidence, not source authority.

## Status and goal

Complete for planning delivery only; successor source launch remains unauthorized. At planning launch, user authorization (2026-10-28) and MAIN verification established CPU0064/0065 Complete, accepted source/gates, HEAD=origin `07a281edd3086efde2853211b71b48288db027e1`, empty index and no conflicting writer. This documentation-only task designs a Class C generated-code/concurrency change; a clean planner and independent targeted review are required. It authorizes no implementation, build, benchmark, contract cutover, commit or push.
Plan one common CPU lowering/emission path for ordinary graphs and the existing captured Scan body. Scan supplies traversal, operand/state/activity/history bindings; it must not own duplicate pointwise, MATMUL, reduction algorithms or a parallel math implementation. Reuse `CompiledGraphModel`, existing typed CPU IR, emitters and worker pool. Do not introduce a second body DAG, public API, runtime interpreter, per-step callback/allocation, generic control-flow framework or unrelated module work.

## Contracts and source packet

Read AGENTS, planning guide, roadmap and current CPU master-plan frontier; documentation rules plus General/Planning profiles. Root [Core invariants](../../../../../ARCHITECTURE.md#core-invariants) and [scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts) route to [Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)/[CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes), Scan [General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan), [Actual first-order closure and arithmetic](../../../../architecture/contracts/recurrent-scan.md#actual-first-order-closure-and-arithmetic), [Activity, raw identity and retained histories](../../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories), and Runtime [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle). Read exact compiler/prepare/memory headings only if source proves their boundary affected. Plans must identify authority changes needed before capability expansion, not change authority themselves.
Inspect `CpuGeneralScanLowering`/IR/emitter and MATMUL/reduction subemitters; ordinary `CpuMatmulLowering`/IR/emitter, pointwise generator/vector/scalar emitters, ordinary aggregate lowering/IR/emission, `CpuCarrierEmitter`, preparer/finalizer, `CpuGeneralScanPreparedExecutable`, `CpuPreparedExecutable`, `CpuWorkerGroup`/`CpuConcurrencyBudget`, and focused Scan/MATMUL/aggregate tests. Existing ordinary vector TANH is complete. Source facts beat conversational pseudocode.

## Dependencies and integration

- Depends on: CPU0064/0065 Complete; accepted Compiler0015/0016 and Prepare0015 boundaries retained.
- Conflicts with: CPU common-emission, Scan, MATMUL/aggregate, cache, worker/concurrency and shared-plan writers.
- Parallel group: None; sole planning writer plus independent read-only review. No concurrent write worktree needed.
- Common base revision: HEAD above plus accepted dirty tree; preserve every unrelated change.
- Integration order: source-grounded draft → independent targeted review → planner fixes/freeze → MAIN validates and accepts planning delivery.
- Integration validation: Markdown targets/anchors, brief completeness/size, source citations, DAG/frontier consistency, scoped whitespace and preservation checks. No Java test/JVM launch for planning.
- Shared-document integration owner: MAIN; clean planner has exclusive writes to this brief, new CPU planning files, CPU master plan and its focused roadmap entries until freeze.

## Deliverable and acceptance

Write one compact source-grounded shared-body plan and a cohesive dependency DAG with roughly two to four implementation stages (not mechanical per-method tasks); detailed briefs only for the first executable stage and immediate successors. Give exact files/symbols, minimal named generation-only types if truly necessary, caller adaptations, retirements, contracts, acceptance and proportional commands. Keep plans non-authoritative and successor implementation unlaunched; propose first Ready status only when contracts/dependencies actually suffice after review.
Resolve or explicitly gate ordinary exact-state SUM versus current represented-dtype sequential Scan SUM: never silently change algorithms, rounding, raw singleton/empty behavior or saved values to obtain reuse. Plan one common owner for needed existing algorithm variants rather than a renamed Scan-only math path.
Cover current static/symbolic/length-aware F32/F64 forward and actual reverse scope, aliases/layouts/carry versions, activity guarding operand reads/contributors, publication/fan-out/saved-value barriers and exact shared resources. New vector eligibility is not automatic support for every CPU family/dtype/layout. Preserve current exclusions until explicitly qualified; avoid a permanent duplicate Scan opcode registry.
Separate vector realization from orchestration. Reuse existing range kernels and workers; plan one submission per Scan where proved safe, run-owned per-worker resources, and dependency-driven phase points where necessary. Address participant admission, no nested budget/worker submission, zero work, failure/cancellation/interruption/close/quiescence and segment thread accessibility. Do not present the current worker pool as a ready multi-phase protocol. Distinguish row-independent inference from shared-gradient/column-tiled bodies; retain a prepared legal single-thread choice when parallel legality is unproved.
Include clearly labeled emitted-wrapper examples for repeated MLP with distinct per-layer parameters and RNN time/state bindings. No automatic NN-layer packing, new Model API, persistence format, low/Metal/OpenBLAS Scan, associative scan or autotuning expansion. Retire duplicate math only after migrated ordinary/Scan and actual-AD qualification; preserve useful v9 optimizations.
Validation plan: focused semantic/structural/cache/conformance/integration checks, clean-Java shape oracle and a small bounded performance diagnostic; affected modules once per stabilized stage and one recorded cross-module checkpoint where necessary. No historical five-fork campaign or universal speed gate. Identify new/removed code and any decisions requiring user approval. Completion reports files, checks, unresolved issues and `Status: Complete` or `Status: Incomplete` with exact follow-up.

## Result and independent review

- [Proposal](../shared-body-emission-proposal.md): source/package/caller map, common scalar
  normalization/emission and explicit exact/sequential algorithms, v9/resource/carry proofs,
  actual MLP/RNN Scan wrappers, scoped authority gates and abortable worker-phase outline.
- [CPU0067](0067-common-scalar-body-lowering-and-emission.md) and
  [CPU0068](0068-qualified-shared-body-vector-emission.md) are compact Draft briefs;
  CPU0069 remains a master/proposal outline until preceding seams stabilize. No successor launch.
- MAIN's assigned read-only SUM and worker lifecycle audits are incorporated, not duplicated.
  Compiler-body recursive optimization is explicitly outside this CPU composition scope.
- CPU master/roadmap synchronize Draft DAG, package map, sole-writer conflicts and closed source
  frontiers. Private extraction alone does not require architecture cutover; actual compatibility
  changes and bounded vector/parallel capability gates remain explicit.
- Preservation baseline: `/private/tmp/synaptik-cpu-repairs.8TWk5O/cpu0066-ready.json`;
  pre-edit comparison: all 2511 files unchanged, HEAD/origin equal, index empty.
- Draft checks: Ruby local-target/heading-anchor/trailing-whitespace check across all six owned
  files PASS (457 links); all task briefs below 200 lines/15 KB. Scoped `git diff --check` PASS.
  SHA-256 comparison against the baseline finds exactly three changed baseline planning files
  and three new planning files, no other change; HEAD/origin/index preserved. DAG/status/source
  call sites reviewed; all successors Draft, CPU0069 outline only. No executable validation run.
- Independent read-only review: APPROVE, no blocking findings; MAIN accepted planning delivery
  on 2026-10-28. Saved-site exactness refers to bits from the selected forward realization;
  ordinary-unrolling oracles must retain the exact/sequential reduction distinction. These
  clarifications introduce no new scope. No Java/build/JVM/benchmark/native/Git mutation or
  authority edit. Javadocs unchanged because no Java was edited. Glossary impact: none.
- Status: Complete. CPU0067 remains Draft pending user approval of four proposed named types
  and its bounded diagnostic budget, then the recorded Ready verification; no source launch.

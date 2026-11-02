# Task Model0034: General differentiable scan architecture decision/cutover

## Status

Complete — ARCHITECTURE DECISION/CUTOVER ONLY, accepted by main on 2026-10-21 after final independent
Class C APPROVE. Authority frontier None; all Java implementation frontiers remain None.
User approval (“ok, tak tedy pokracuj”, 2026-10-21) followed the concrete ScanBody/reverse-scan/
canonical-residual proposal and source-checked comparison. Main recorded authority-only approval,
not arbitrary control flow or Java slices; earlier planning-only approval is separate evidence.

## Change class

Class C — source/compiled body ownership, residual bindings, pre-capture automatic differentiation
(AD), capability queries, Prepare projection/resources and generated hot-path invariants.
Executed in a clean authority context, then reviewed in a separate independent Class C context.

## Goal and scope

Enact the already approved [program](../../../general-differentiable-scan.md), without redesign:
root + existing owning scoped sections + focused explanations + significant ADR in one coordinated
documentation-only cutover. Broaden the recurrent semantic owner in place, retaining the six-contract
authority index. The bounded exception is now enacted in the
[scan semantic owner](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan);
this brief/program is coordination, not authority or Java implementation permission.

Non-goals: Java/build/test-source changes, a seventh contract, new module/dependency edge, additional
types/layers, if/while/general control flow, recursion/nested or parallel/associative scans, runtime tape,
mixed-owner bodies, another AD algebra/executor, fixed public RNN adjoint, automatic NN/padding/source/
persistence admission, measurements, staging, commit or push. Do not retire existing fixed/static routes.
Model source/IR is the sole next successor, still Draft/not yet briefed; new Ready verification is required.

## Contracts

Read these exact headings; amend only scan deltas in their existing owners, preserving other rules.
If approved scope leaves an actual ambiguity, stop and report it; do not invent architecture.

- [Root](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts): `Scope-indexed normative
  contracts`, `Core invariants`, `Dependency rules`, `Explicit non-goals`. Update index/routing and
  narrow body exception, not general control flow, dependency direction or the single RunState rule.
- [Foundational](../../../../architecture/contracts/foundational-modules.md#modulesmodel):
  `modules/model`, `modules/planning`, `Profile-free capability queries`, `Profile-free numerical
  semantics`. Source attachment/canonical ports versus compiled body; Tensor-free queries/demand;
  retain family arithmetic/saved sites and no Tensor/graph dependency from operation attrs.
- [Recurrent owner](../../../../architecture/contracts/recurrent-scan.md): `Scope`, `Fixed recurrent
  scan without graph regions`, `Fixed family and current Model surface`, `Static Shape and runtime-value
  boundary`, `Valid lengths, traversal, and outputs`, `Purity and lifecycle ownership`, `Performance
  and migration invariants`. Broaden in place for the approved general scan; keep fixed semantics
  and anchors valid. Own formals/results/residuals, scoping, traversal, empties and exclusions here.
- [Compiler](../../../../architecture/contracts/compiler-autograd.md#modulescompiler): `modules/compiler`,
  `Training graph model`, `Compiler-owned automatic differentiation`, `Compile lifecycle`. Scoped
  capture, joint-output pre-capture AD and exact primal binding; keep cold source admission closed.
- [Runtime/Prepare](../../../../architecture/contracts/runtime-prepare-engine.md#modulesruntime):
  `modules/runtime`, `modules/prepare`, `Prepare lifecycle`, `Run lifecycle`. Validated body projection,
  disjoint local/outer descriptors, ordinary saved edges, pre-slot exact resources, one RunState;
  no runtime graph inspection, nested owner or opaque saved-workspace handshake.
- [Backend](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules):
  `Concrete backend modules`, `CPU backend routes`. Whole-body/binder qualification, emitted time loop,
  typed step bindings, isolated resources and cache identity; no backend global AD or support claim.
- [Extensions](../../../../architecture/contracts/extensions-training.md#owned-tensor-models):
  `Owned tensor Models`, `Generic graph source and values`, `Canonical graph-model version 3`.
  Preserve closed wire/source domains and early scan-training rejection; unsupported body-containing
  NN acquisition rejects before acquisition/setup effects. Low-level AD grants no NN/train/save/load admission.

## Dependencies and integration

- Depends on: Model0025E/0025F/0032/0033; Compiler0006A/0006D/0008/0009; CPU0028/0029;
  Engine0021; Training0008; NN0021B/0022B/0028/0029 — all Complete, accepted baseline.
- Conflicts with: scan API/IR/residuals, capture/AD, Planning queries, Prepare projection, CPU
  resources/cache/ledger, NN admission/persistence and shared authority/docs; serialize all.
- Parallel group: None; sole writer, no concurrent write task or isolated worktree needed.
- Common base revision: HEAD=main=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`.
- Integration order: approved frozen packet → clean authority executor → separate independent Class C
  review/corrections → main acceptance → separately Ready Model source/IR → serial program DAG.
- Integration validation: `Model0034 authority checkpoint` below; future module/conformance/integration
  gates and one main-owned source-matched native-enabled root `test javadoc` at low-level join.
- Shared-document integration owner: main; executor exclusively owns authority/explanations/ADR,
  independent reviewer works serially; main delegated this final factual planning/ADR-status closure
  to the same executor under exclusive shared-document integration ownership.

Historical prelaunch verification (clean planner, 2026-10-21): actual approval, planning APPROVE
(zero blockers), all 16 Complete predecessors and accepted validation verified; no full history audit.
HEAD/origin/main matched the recorded base; the then-current prohibition was this task's explicit
replacement target. Main reconfirmed protected hashes and inactive conflicts; no parallel writer.
Only Model0034 authority was Ready at launch; no Java frontier was authorized.

## Files and symbols

Executor write scope: `ARCHITECTURE.md` and the six existing scoped files named above (owning sections
only), plus these exact documentation paths; no source/type/package additions in this task:

- `docs/architecture/general-differentiable-scan.md` — create focused explanation with conceptual
  body/forward/residual/reverse example, metadata/history distinction and capability/migration limits.
- `docs/architecture/module-boundaries.md` — affected Model/Planning/Compiler/Prepare/backend/NN explanations.
- `docs/architecture/training-graph.md` — retain fixed BPTT boundary; explain approved generic-body AD.
- `docs/architecture/runtime-prepare-backend-boundary.md` — generic handoff beside unchanged fixed route.
- `docs/architecture/current-architecture-plan.md` — main-authorized owner label and navigation links only.
- `docs/design/decisions/0045-general-differentiable-scan.md` — new significant ADR: actual approval,
  alternatives, costs, exclusions and superseded prohibition only, not implemented capability.
- `docs/glossary.md` — targeted scan/residual/formal/cotangent definitions only if needed; otherwise
  record a reasoned no-change conclusion. Initial authority execution excluded planning writes;
  main later delegated closure of the seven existing planning files and factual ADR Status only.

Source roots end in `src/main/java/io/github/pho001/synaptik/`; evidence only, not write authority.
Model: `tensor/TensorProducer`, `graph/CompiledNode`, `CompiledGraphModel`, `operation/OperationAttrs`.
Compiler: `GraphCapture`, `AutogradPreflight`, `FirstOrderAutograd`, `ElementwiseGradientRules`,
`CapturedGraphInference`, `StaticResultLogicalLayoutClosure`, `ForwardGraphOptimization`,
`ForwardDeadCodeElimination`, `ForwardCommonSubexpressionElimination`, `GraphCompiler`,
`GraphCompilationPort`, `SourceGraphReconstruction`. Planning: `capability/OperationCapabilityQuery`,
`memory/LogicalMemoryPlanning`. Prepare: `analysis/PrepareContext`, `PartitionDag`.
CPU internal: `lowering/CpuRecurrentScanLowering`, `executable/CpuRecurrentScanPreparedExecutable`,
`route/portable/CpuRecurrentScanKernel`, `codegen/emit/{CpuLoopEmitter,CpuScalarEmitter,
CpuClassFileKernelGenerator,CpuMatmulEmitter,CpuCarrierEmitter}`, `ir/CpuKernelIr`.
NN: `module/ModelExecutionState.rejectScanTraining`. Future types remain exactly the
[program inventory](../../../general-differentiable-scan.md#proposed-source-body-and-minimum-apiir).

Focused tests: Compiler `RecurrentScanCompilerTest`, `AutogradPreflightTest`; CPU
`CpuClassFileKernelGeneratorTest`, `CpuRecurrentScanExecutionTest`, `CpuRecurrentScanLoweringTest`;
conformance `CpuRecurrentScanConformanceTest`; integration `EngineCpuRecurrentScanIntegrationTest`,
`RuntimeLengthScanModelIntegrationTest`; NN `RuntimeLengthScanModelTest`. Inspect relevant cases only;
fixed-scan evidence does not prove generic-body semantics and needs no rerun for this docs-only task.

## Acceptance criteria

- Root, existing scoped owners, explanations and ADR agree on the complete approved program
  (all three protocol sections plus serial acceptance gates); preserve anchors/fixed routes and
  distinguish accepted decision from unimplemented capability. No new design layer/type.
- Canonical residual ports are reserved at ORIGINAL immutable producer construction, not replacement
  or replay: OUR immutable-producer design choice, not an industry requirement. Ordinary users write
  Tensor algebra, never saved-tensor recipes. Retained canonical siblings/previous carry and explicit
  step/invariant bindings stay precise. Metadata != allocated history; only publication/downstream
  demand creates history. Propagate/revalidate demand through query/Prepare; internal uses need step
  temporaries, dead ports zero bytes/stores and no fusion restriction, including reverse residuals.
- Preserve exact original Tensor → residual/step/invariant reverse formal binding for EVERY primal
  formula operand (including saved TANH output), original contribution identities and joint seeds once
  per scan; tied positional/cross-occurrence terms sum once each. No captured-IR AD or ValueId→Tensor AD map.
- Same immutable template/producers may serve multiple occurrences with independent capture scopes;
  reject outer escape/cross-template sharing. No outer ValueIds inside a body; role bindings may repeat.
  Close TANH/SIGMOID oneLike leaves through invariant formals/outer registered splats; zero seeds and
  accumulators through outer constants/expressions, carry initializers through carry formals.
- Codify program purity/free-leaf/alias, T=0/empty and residual-root/higher-order/createGraph rejection
  rules, whole derivative closure including ordinary floating tensor/scalar DIV, reductions/expands/
  views/contiguous/constants/casts, and existing family arithmetic/saved-value rules without widening.
- Preserve whole-body/binder capability (child support insufficient), one-owner projection/preparation,
  exact checked staging/carry/temporary/history bytes BEFORE slots, finalization-emitted outer time loop
  + admitted typed emitters, hoisted invariants/direct time offsets and justified materialization/live
  saved/MATMUL storage. No T-fold growth, runtime interpreter/per-op lookup/per-step callback or allocation.
  Static direct helpers require clean-Java oracle rationale; no giant-method requirement.
- Changing carry/addresses/geometry remain run-owned, never shared prepared state. CpuKernelIr excludes
  extents/slots/run bindings; key every baked body/role/demand fact and reject stale identity/meaning.
- Preserve complete bounded F32 pointwise forward/automatic-reverse milestone, then remaining forms/
  closure + F64 domain completion; low types later under existing semantics. All-valid/canonical/static T
  first. Padding row-separability/branch-before-padded-read/arithmetic, NN admission and persistence
  remain independent gates; masking != skipping. Old four-dtype/layout fixed/static routes remain.
- Retain later CPU optimal clean-Java algorithm/hot-loop/dataflow oracle, Class-File inspection and short
  warmed matched-work generated/direct evidence; separate cold generation/class-load/JIT, entry/preflight
  and kernel/step costs. Fix/justify diagnosed avoidable overhead; no broad/five-fork, no-copy/JIT/
  zero-penalty or CPU0029 parity promise. No measurement now.
- Independent Class C review approves final authority/explanation/ADR diff before main acceptance;
  no successor automatically Ready. Preserve Superseded Compiler0010 unchanged.

## Validation and documentation/review impact

`Model0034 authority checkpoint`: check changed local links/anchors, exact root scope ownership and
six-contract count, cross-document semantics/limits above, ADR approval/consequences and no implemented-
capability claim; `git diff --check` plus untracked-file trailing-whitespace checks. Confirm no Java/
build/dependency/test-source/Git-index change and protected hashes unchanged; preserve unrelated dirty
notice/cast oracle and Compiler0010. No Gradle/native build/Javadoc/measurements for this docs-only delta:
dependency rules/code are unchanged. Future implementation owns tests; main owns one low-level join.

Apply `docs/developer-guide/documentation-rules.md` plus profiles under
`docs/developer-guide/documentation/`: `general-style.md` throughout, `architecture-style.md` for root/
scoped contracts/explanations, `decision-record-style.md` for ADR, `example-format.md` for examples.
Planning/general profiles apply to this packet. Independent review is mandatory in its own clean thread;
inspect actual diff/current source/focused tests, reuse successful evidence, review targeted glossary.
Java/Javadoc unchanged. Main accepted the checkpoint/review and delegated factual closure integration.

## Result

Status: Complete — authority decision/cutover only, accepted 2026-10-21.

Completed root/six scoped owners, focused general-scan explanation, module/training/handoff updates,
ADR0045, three targeted glossary entries and authorized current-architecture-plan.md navigation.
Initial scope was 14 files (12 changed, two new); delegated closure updates seven existing planning
files plus ADR Status only, for 21 unique files overall. No Java/build/dependency/test-source change.

Ramanujan the 2nd returned final independent Class C APPROVE, zero blocking findings, after reading
current source/focused tests; no edits or Java reruns. Main accepted all 14 frozen hashes, protected
hashes, unchanged source/build except the protected preexisting oracle, and empty Git index.
Reused authority checkpoint: 550 local links/anchors, zero failures/lost anchors/trailing whitespace;
exactly six owners with non-overlapping scope. Executor and main `git diff --check` passed.

No tests/Javadoc/Gradle/native build/measurements: docs-only, no code/dependency delta. Java/Javadoc
remain unchanged; protected notice, cast oracle, Superseded Compiler0010 and index are unchanged.
Closure passed 594 local links/anchors, status/frontier/metadata/whitespace/size and protected-hash
checks, with no Java reruns. No unresolved issue. Model source/IR remains Draft/not yet briefed,
then serial query/AD/Prepare/CPU/domain/join slices; none is implemented or Ready. Padding/NN and
persistence stay separately gated. Earlier 576-link/planning APPROVE evidence was planning-only.

Size exception: this completed packet modestly exceeds 15 KB to retain its atomic Class C
acceptance, exact contracts and compact result; splitting would detach approval/validation from scope.

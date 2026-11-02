# Dynamic batch through one reusable prepared execution

## Purpose and selected direction

Coordinate true per-run symbolic batch `B` through one immutable compiled graph and prepared
execution. The user selected this direction on 2026-10-24. The
[Prepare0012 proposal](modules/prepare/dynamic-batch-geometry-contract-proposal.md) is Complete as
reviewed DESIGN delivery: independent Class C APPROVE, zero open P1/P2, and MAIN delivery acceptance.
User “ano” now explicitly approves authority cutover, subsequent ordinary CPU forward/actual
first-order backward and bounded final A/B. [Prepare0012A](modules/prepare/tasks/0012a-dynamic-batch-authority-cutover.md)
is Complete: authority enacted/MAIN-accepted 2026-10-24 after Pasteur definitive APPROVE, zero open
P1/P2, docs/glossary review complete.
[Prepare0013](modules/prepare/tasks/0013-ordinary-dynamic-batch-cpu-vertical.md#result) implements bounded
ordinary homogeneous FLOAT32 CPU forward and actual first-order backward and is Complete.
Final independent reviews and the native-enabled root gate passed; MAIN accepted it on 2026-10-24.
The design [brief](modules/prepare/tasks/0012-dynamic-batch-geometry-and-resource-contract.md)
retains launch/review evidence. This does not reopen the selected direction.
This is a non-authoritative coordination map, not an architecture contract or a broader capability grant.

The user's subsequent “ok, pokracuj” authorizes only
[Model0039 symbolic-Scan design](modules/model/tasks/0039-general-scan-symbolic-batch-design.md),
now Complete for DESIGN DELIVERY ONLY: independent frozen APPROVE, zero open P1/P2, and MAIN
acceptance on 2026-10-24, not by itself detailed semantic approval or implementation. The
[focused proposal](modules/model/general-scan-symbolic-batch-proposal.md) targets all-valid
FLOAT32 CPU forward/actual first-order backward, fixed T/F/H and one reusable per-run B.
It explicitly covers symbolic multi-output layout binding, source/inference/AD/Prepare guards
and CPU numeric-resource/generated-loop changes. Proposed new production types: zero;
existing owners/recipes are candidates, not an implementation proof. Model source/dtypes and
Compiler AD admission remain separate from CPU execution qualification. PlanningAuthorized=true;
ImplementationAuthorized=false. At Model0039 acceptance, planning/implementation frontiers were None.

[Model0040](modules/model/tasks/0040-same-geometry-reuse-design-amendment.md) is Complete for DESIGN
AMENDMENT ONLY, authorized by “ok, dopln to” on 2026-10-25. It proposes one exact-preparation
immutable metadata memo, retaining per-call Compiler binding/obligations, input/storage validation
and fresh resources. Independent APPROVE, zero open P1/P2; MAIN accepted 2026-10-25.
The user then confirmed that direction and explicitly authorized the contract-first next step.
[Model0041](modules/model/tasks/0041-symbolic-scan-authority-cutover.md) is Complete for
authority/docs only. The accepted
[Scan semantic owner](../architecture/contracts/recurrent-scan.md#accepted-symbolic-batch-scan) and
[Engine memo owner](../architecture/contracts/runtime-prepare-engine.md#accepted-engine-geometry-memo)
now distinguish shareable immutable numeric facts/private backing from mutable run arrays and
retain current input-backed descriptors plus full fresh Compiler bind/private obligations.
Independent frozen APPROVE, zero open P1/P2; MAIN accepted 2026-10-25.
PlanningAuthorized=true; AuthorityAuthorized=true; ImplementationAuthorized=false.
At that authority acceptance, planning/authority/implementation frontiers were None.
No new type/public signature or atomic mechanism was selected by that authority-only delivery.
The separately user-admitted [Model0042](modules/model/tasks/0042-symbolic-scan-source-foundation.md)
implements static/bounded-symbolic Model source only; independent APPROVE, zero open P1/P2,
configured root test/Javadoc PASS and MAIN acceptance 2026-10-25 are recorded in its brief.
At that source acceptance, implementation frontiers were None; Compiler0014 below is now Complete.

The user next accepted MAIN's recommendation to prepare and then implement
[Compiler0014](modules/compiler/tasks/0014-symbolic-scan-inference-ad-and-binding.md), a compact
now Complete packet for independent symbolic inference, actual first-order AD and scoped all-port
`BatchBindingPlan` binding through existing owners. PlanningAuthorized=true;
ImplementationAuthorized=true after user source approval and MAIN Ready verification 2026-10-25.
Independent planning and code/docs APPROVE, zero open P1/P2; MAIN accepted 2026-10-25 after
module/Javadoc and scoped compatibility validation. At that acceptance, frontier None. Forward inference retains
accepted Model PURE/dtype breadth, separately from bounded F32/F64 AD and later initial CPU F32
qualification. No symbolic execution/cold/memo enablement, new type/signature or automatic successor.

Completed packet: [Prepare0014 symbolic Scan CPU vertical](modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md#result),
MAIN accepted 2026-10-25 after independent code/docs APPROVE and the configured source-matched
root gate. PlanningAuthorized=true; ImplementationAuthorized=true record historical admission. User “jdi na to”
authorized source; MAIN verified Ready 2026-10-25 17:00 UTC with Complete predecessors, accepted planning
snapshot and inactive conflicts/sole writer. One atomic F32 forward/actual-backward capability
across existing Prepare/CPU/Engine owners is delivered; memo remains excluded. Bounded CPU/Engine
measurements and their limits are recorded in the brief, not a parity guarantee. Source frontier None;
Parallel group None, MAIN completed shared-document integration and the vertical join. No commit/push/successor.

[Model0043 symbolic-B plus runtime lengths](modules/model/tasks/0043-symbolic-length-scan-integration-design.md)
is Complete for DESIGN DELIVERY ONLY: Hume the 4th final Class C APPROVE, 0 P1/P2/no unresolved
findings; MAIN accepted 2026-10-26 after the verified planning-only admission/launch.
The [bounded integration proposal](length-aware-general-scan.md#proposed-symbolic-b-plus-lengths-integration)
targets F32/F64 forward/actual first-order reverse and identifies the narrow outer F64/control
prerequisites. Historical PlanningAuthorized=true; AuthorityAuthorized=false; ImplementationAuthorized=false.
At that design acceptance, planning/source/authority frontiers were None. The subsequent explicit
user semantic/source approval admits only [Model0044](modules/model/tasks/0044-symbolic-length-scan-source-foundation.md):
static/one-B length source, bounded root/scan-owner wording before Java and deliberate consumer
rejection. Source is implemented under the authority/Ready acceptance of 2026-10-26, with independent
source/docs APPROVE and 0 P1/P2. MAIN's configured root test/Javadoc checkpoint passed and MAIN
accepted Model0044 as Complete on 2026-10-26. Active Model frontiers are None.
Compiler0015 subsequently implements the separately approved inference/actual first-order AD/fresh
scoped all-port binding intersection. Its [Result](modules/compiler/tasks/0015-symbolic-length-scan-inference-ad-and-binding.md#result)
records its completed review, validation and MAIN acceptance.
Prepare0015 implements bounded combined Prepare/CPU/Engine execution and is Complete, MAIN accepted
2026-10-27; its [Result](modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result) records
validation and nonstationary timing limits. Engine memo and new API/type remain outside that task.

Active frontier: None. [Prepare0015 symbolic-length Scan CPU vertical](modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result)
is Complete, MAIN accepted 2026-10-27 after independent code/docs APPROVE (0 P1/P2), owning validation,
the sole bounded probe and configured root test/Javadoc PASS. Complete Model0044 + Compiler0015 +
CPU0042/Prepare0014 and retained predecessors feed the delivered F32/F64 vertical, including bounded
outer/control/ROW(1). No new type/API/memo or prerequisite chain. Probe gaps remain nonstationary
and unexplained, not performance parity or a JIT-cause qualification. No successor is authorized.

Compile retains symbolic meaning; prepare fixes code, routes, slots and typed recipes; each run
supplies actual numeric geometry and owns its resources. Existing `DynamicDimension` and
`DimensionExpressions` remain. Feature and parameter dimensions stay fixed; the first general
Scan extension also keeps time `T` fixed. Variable `T` is a later coarse phase, not solved by lengths.
There is no mandatory public `B_max`, required per-B full-preparation cache or graph rebuild per batch.
Allocate actual per-run bytes, not mandatory maximum-capacity banks.

## Current truth

Symbolic Model values alone do not grant executable binding. Runtime0010's completed task name
“dynamic execution” means mutable run state, not symbolic-extent admission. Completed briefs and
all existing static paths, including their empty-domain behavior, remain unchanged.

Prepare0013 implements the ordinary named-B path across the existing owners below. Its domain is
rank-at-most-two boundaries plus qualified AD views, generated scalar single-range CPU execution;
other operations, dtypes, layouts and workflows do not gain support by analogy.

| Boundary | Current source and consequence |
|---|---|
| Symbolic meaning | [DynamicDimension](../../modules/model/src/main/java/io/github/pho001/synaptik/model/shape/DynamicDimension.java) is identity-only. [DimensionExpressions.unknown](../../modules/model/src/main/java/io/github/pho001/synaptik/model/shape/DimensionExpressions.java) creates a distinct unknown, not bounds attached to named B. |
| Logical output and semantic binding | [LogicalMemoryRequirement](../../modules/planning/src/main/java/io/github/pho001/synaptik/planning/memory/LogicalMemoryRequirement.java) retains descriptors, not bytes. [BatchBindingPlan](../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/BatchBindingPlan.java) preindexes metadata and private typed diagnostic obligations; binding derives consistent B and lazily resolves requested descriptors. |
| Descriptor and prepare | [TensorDescriptor](../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorDescriptor.java) still rejects resolved layouts on dynamic shapes; [PrepareContext](../../modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/analysis/PrepareContext.java) admits bounded ordinary and all-valid symbolic Scan named-B projections alongside static descriptors. |
| Resources | [PreparationResourceRequirement](../../modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/analysis/PreparationResourceRequirement.java) and [PreparedMemoryPlan](../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/memory/PreparedMemoryPlan.java) carry fixed/run-size declarations and explicit mode. [RunMemoryGeometry](../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/RunMemoryGeometry.java) resolves exact sizes; [PreparedRepresentationPlan](../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/resource/PreparedRepresentationPlan.java) creators receive actual bytes/alignment, and CPU allocates fresh storage per run. |
| Engine admission and output | [AdvancedEngine](../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/AdvancedEngine.java) validates semantic/numeric geometry and storage before borrowing exact prefixes. [RunResult.Publication](../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/RunResult.java) receives actual occurrence descriptors. Static descriptor matching remains unchanged. |
| NN and Training admission | [ModelExecutionState.validateValues](../../extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/ModelExecutionState.java) and [TrainingSession.bindValues](../../modules/training/src/main/java/io/github/pho001/synaptik/training/TrainingSession.java) retain static external matching; dynamic one-state reuse is separately gated. |
| CPU binding and loops | [CpuPreparedExecutable](../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutable.java) validates actual bytes. Prepared numeric access/MATMUL/aggregate recipes bind invocation B once; generated loops keep fixed code/routes/carriers. Exact FLOAT32 SUM/MEAN uses five limbs with conditional 0/48-byte scratch. |
| General Scan | [CpuGeneralScanEmitter](../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneralScanEmitter.java) retains static baked loops and separately emits bounded F32 symbolic forward/actual reverse with invocation-bound numeric geometry. Prepare0014 is independently reviewed and MAIN accepted; fixed T/features/state, named B. Symbolic cold/copy/wire, lengths/skipping and memo remain unsupported. |

Existing [Engine lifecycle tests](../../modules/engine/src/test/java/io/github/pho001/synaptik/engine/EngineTypedLifecycleTest.java),
[run creation tests](../../modules/runtime/src/test/java/io/github/pho001/synaptik/runtime/run/RunStateCreationTest.java)
and [Training fresh-value tests](../../modules/training/src/test/java/io/github/pho001/synaptik/training/TrainingSessionValuesTest.java)
provide static reuse/admission/cleanup evidence, not dynamic-B qualification.
The focused [Engine dynamic-B tests](../../modules/engine/src/test/java/io/github/pho001/synaptik/engine/EngineOrdinaryDynamicBatchTest.java),
[CPU semantic/Class-File tests](../../modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuOrdinaryDynamicBatchTest.java),
[conformance tests](../../testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/CpuOrdinaryDynamicBatchConformanceTest.java)
and [public integration test](../../testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineCpuOrdinaryDynamicBatchIntegrationTest.java)
provide bounded ordinary dynamic forward/actual first-order backward evidence.

## Qualified reuse scenario

Goal: vary only batch size without recompilation or preparation. Assume an admitted FLOAT32
ordinary CPU slice with `X[B,4]`, fixed `W[4,3]`, `Y=X@W`, and scalar `mean(Y)`.
This implemented/tested scenario is not runnable Java or a dtype-wide promise; the
[running guide](../user-guide/running-models.md) supplies the public seeded scalar-loss example.

| Run | Actual B | Actual Y descriptor shape | Y payload bytes |
|---|---:|---|---:|
| 1 | 64 | `[64,3]` | 768 |
| 2 | 40 | `[40,3]` | 480 |
| 3 | 1 | `[1,3]` | 12 |
| 4 | 64 | `[64,3]` | 768 |

Each row uses the same compiled/prepared identities and fresh run-owned writable storage; all
other resources are sized separately from their actual requirements. With all-one X/W, Y contains
4 and the mathematical mean is 4; each weight gradient is 1/3 for every listed B. Input gradients
have actual `[B,4]` descriptors; parameter gradients remain `[4,3]`. A changed feature width or
inconsistent B across inputs rejects at admission. Qualified B=0 produces empty Y/dX, scalar NaN
empty MEAN and nonempty typed-zero `[4,3]` dW; it does not skip the whole run. The retained Y+L
witness has 16 units, 19 buffers and four workspaces; the public explicit-seed timing closure has
20 buffers/four workspaces. These are observed closures, not universal resource totals.

## Accepted boundaries and remaining gates

1. **Compile handoff.** Preserve symbolic descriptors and exact typed obligations through the
   compiler-owned boundary; diagnostic strings are not predicates. Preserve the eight
   `CompileArtifacts` components unless a later explicit decision changes that contract.
2. **Run admission.** Before borrowing/creation/writes, validate input ranks, fixed axes, consistent
   B bindings, exact dtype and requiresGrad, admitted layout/strides/offsets, storage accessibility,
   alignment/capacity, checked element/byte spans, overflow and deferred shape constraints.
   Dynamic-template inputs require resolved zero-offset canonical dense reads; absent layouts reject.
3. **Resource declarations.** Fixed/run-size recipes, explicit mode and actual-size/alignment
   creators implement the accepted handoff. All requirements, slot identities and representation positions are known
   before finalization; prepared typed recipes derive exact run sizes without discovering resources.
   Prepare stages/validates the handoff but neither executes nor interprets backend geometry.
   Concrete backends own physical allocation and numeric address mechanics.
4. **Runtime isolation.** Runtime keeps no Model/compiler dependency, Shape/DimensionExpression/
   Operation/graph interpreter, route search, code generation or cache lookup. The typed
   numeric handoff binds once per invocation, never per element or scan step.
   One RunState owns all mutable geometry, addresses, carry banks, temporaries and resources;
   no mutable B, working numeric arrays or storage is written into shared prepared state.
   The accepted future Engine-only metadata exception reuses immutable exact-plan facts, not
   Runtime cache lookup, mutable working arrays or resources; it remains unimplemented.
5. **Publication and lifetime.** Hand actual run output descriptors to Engine publication and
   materialization, preserving occurrence identities and gradient roles. Keep borrowed-input
   lifetime, attempt-all reverse cleanup, rollback/retry, prepared close/run leases, result leases,
   NN exclusion and concurrent-run isolation. Failure publishes no partial result; it never closes
   caller-owned input storage. Concurrent results with different B must remain independent.
6. **Separate future Scan preflights.** Shape/byte geometry is checked at run admission. Length VALUE checks
   occur after length producers execute and before this Scan's first output/padding/history/scratch
   writes; never move produced-length checks to run start. Validate all actual B lengths in `[0,T]`
   at that point, including the last invalid value and T=0 cases under the accepted empty policy.

For the future symbolic Scan extension, TBF (time/batch/features) time stride is `B*F`;
row histories/banks depend on `B*H`, while demanded shared
histories have no invented B axis. Check aggregate sizes/offsets and representability limits before
allocation; do not add B×parameter gradient banks. Preserve zero bytes/stores for dead residuals.
Empty domains cannot globally skip a different nonempty reduction or shared output.

[ReductionGradientRules.mean/count](../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ReductionGradientRules.java)
already uses `oneLike(input).sum(...)`, with
[DerivativeConstants.oneLike](../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/FirstOrderAutograd.java)
expanding a typed one. Compiler does not bake a denominator of 64: dynamic reduction geometry must
make that existing algebra correct. Parameter MATMUL adjoints may contract B. Training accumulation
is a sum of supplied gradients, not automatic sample weighting; mixed-batch objective weighting
must stay explicit. Scan-local lengths imply neither a masked outer loss nor a changed MEAN domain.

## Owners and coarse dependency order

The [architecture scope index](../../ARCHITECTURE.md#scope-indexed-normative-contracts) remains
the sole authority router. This map describes current ownership and separately gated extensions;
the exact nine-type inventory remains in the task and scoped contracts.

| Owner | Decision responsibility |
|---|---|
| Model, Compiler and Planning | Symbol identity/expressions, descriptor semantics, typed constraints, actual derivative algebra and symbolic logical requirements. |
| Prepare | Cross-owner handoff, stable slots/positions, declaration/creator coverage and transactional preparation; no backend geometry interpreter. |
| Runtime | Model-free prepared contracts, run-owned numeric state, creation orchestration, leases, cleanup and cold-bound schedule execution. |
| CPU | Prepared numeric recipes, lowering/emission, actual-byte binding, code identity and independently qualified forward/backward loops. |
| Engine | Public input admission and actual-descriptor publication/materialization through the existing prepared owner. |
| NN and Training (future dynamic reuse) | Separately qualify one owned parameter/optimizer state across B values through public Engine, preserving exclusion and transactions. NN acquisition and symbolic persistence qualify separately below. |

Read this order as gates, not an automatically executable task queue:

```text
Complete baseline -> Prepare0012 reviewed DESIGN delivery Complete -> user architecture approved
  -> Prepare0012A Complete, authority enacted/MAIN-accepted 2026-10-24
  -> Prepare0013 ordinary dynamic-B CPU forward AND first-order backward implemented
       (Complete: final reviews/root gate passed, MAIN-accepted 2026-10-24)
       -> separately qualified ordinary NN/Training reuse of one state
       -> Prepare0014 symbolic-B Scan vertical Complete; fixed T/F/H, F32 forward/actual reverse
Length-aware static projection (Prepare0011 Complete) -> CPU0042 fixed-length F32/F64 Complete
Dynamic-B Scan + fixed length execution -> Model0043 combined F32/F64 DESIGN DELIVERY Complete
  -> Model0044 source Complete; independent APPROVE, MAIN root PASS/acceptance 2026-10-26
  -> Compiler0015 inference/actual AD/binding Complete
  -> Prepare0015 vertical Complete; reviews/root PASS, MAIN accepted 2026-10-27; no performance-parity claim
```

The general-Scan branch has completed reviewed/MAIN-accepted Model0039 and Model0040 design
deliveries. User semantic approval and authority admission followed; Model0041 is Complete
for coordinated accepted-future authority/ADR0051/docs, not implementation. Independent frozen
APPROVE and MAIN acceptance (2026-10-25) precede separately admitted Model source, Compiler actual AD/all-port
binding and Prepare/CPU/Engine vertical gates. Model0042 is the separately authorized source-only
delivery, Complete after independent review and MAIN acceptance; no downstream successor is Ready.
Its immediate Compiler0014 successor is Complete: scoped inference, actual first-order AD and
all-port binding are implemented; module/Javadoc PASS (384 tests), independent APPROVE (0 P1/P2)
and MAIN's scoped compatibility checkpoint PASS (68 tests), accepted 2026-10-25. Dependency/base/
conflict, launch verification and evidence are recorded in that brief. Prepare0014 is separately
Complete after numeric/conformance checks, bounded CPU/Engine probes, independent reviews and
the configured source-matched root join. Its brief records results and measurement limits.
The memo's exact publication/close mechanism remains later scoped work: after inward admission,
close suppresses publication without a new outward-close error, and Runtime's sole lease arbitrates
the later race. CPU0042 now completes fixed-geometry length skipping; Model0044 admits the source
foundation for symbolic-B plus lengths and Compiler0015 adds inference/actual AD/all-port binding,
not execution. NN/Training reuse remains separately gated; neither this planning
delivery nor the fixed-length capability admits symbolic-length execution or state reuse.

NN entry points qualify separately: `Model.graph` and `defineTensor` need future symbolic
Compiler cold reconstruction and Checkpoint copy support; symbolic graph save/load additionally
needs its own wire qualification. Builder acquisition bypasses that copy path and requires
independent qualification, not inferred support. Existing static saved Models/readers remain
supported. None of NN cold acquisition, symbolic persistence or builder qualification blocks the
ordinary low-level Engine+CPU vertical.

Static length support is not cancelled and need not wait for distant NN/persistence work. The
ordinary geometry handoff is implemented; further conflicting CPU binder/resource changes require
separate admission. Variable T, native/OpenBLAS/Metal, mixed-backend transfers, persistence and tuning retain
separate later qualification. [SourceGraphReconstruction](../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/SourceGraphReconstruction.java)
and [GraphArtifactCodec](../../modules/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/graph/GraphArtifactCodec.java)
are static; Metal is not qualified, and OpenBLAS invokes plan m/n/k. No support follows by inference.

## Admission and acceptance gates

Prepare0012 reviewed DESIGN delivery: Complete, independent Class C APPROVE (zero open P1/P2),
MAIN delivery accepted 2026-10-24. Prepare0012A Complete, authority enacted/MAIN-accepted 2026-10-24;
Pasteur definitive APPROVE, zero open P1/P2, docs/glossary review complete. MAIN reverified all five dependencies plus
design Complete at 2026-10-24 17:41:11 UTC, HEAD/origin/index unchanged, baseline 2435 files.
Prepare0013 is Complete after frozen implementation, successful worker validation, final
code/Javadoc/performance and shared-document APPROVE with zero open P1/P2, and MAIN's successful
native-enabled root join. MAIN accepted the capability on 2026-10-24; authorized frontier None.
Base: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus accepted uncommitted work,
not merged predecessors. MAIN owns serial shared-document integration; Parallel group None.
No automatic successor execution or commit/push is authorized.

MAIN admitted Model0039 as the sole planning-only frontier on this same accepted dirty base,
with Complete Model0035/0038, Compiler0013, Prepare0013 and CPU0037 verified. The brief owns
admission and all seven integration fields. No competing writer; Parallel group None; MAIN owns
shared integration. Model0039 is now Complete for reviewed DESIGN DELIVERY ONLY, MAIN accepted
2026-10-24; planning frontier was None at that acceptance, no automatic successor launch.
Prepare/Compiler/CPU implementation frontiers remain None. T=0 qualification uses a separate
fixed-T plan; B=0 never globally skips required nonempty shared work or parameter gradients.
Current authority frontier: None; Model0041 is Complete. Complete fresh bind, all scoped ports and
bounded rank-three actual closure, immutable-only memo snapshots and current input-backed metadata
are future matching validation obligations, not evidence supplied by this document-only cutover.

The task Result records 64→40→1→64 reuse, actual outputs/gradients/resources, admission,
empty/overflow/cleanup/retry/concurrency, retained static paths and generated-versus-clean-Java
algorithm/Class-File evidence. Bounded timing separates cold setup from invocation measurements;
the original harness predates the final MATMUL repair and one authorized post-fix probe also exists.
The latter's generated/direct sample ratios are tiny MATMUL 0.968–1.337, medium MATMUL 1.008–1.035
and exact MEAN 1.398–1.414. Its callsite/warmup/input changes are joint confounds, not a replacement
acceptance row or isolated B tax. The original unfavorable ranges, unmatched medium Engine routes
and differing allocation topology remain disclosed; no general parity or negligible-overhead claim follows.
No certificates, general proof engine, new session/executor or second graph/DAG is proposed.

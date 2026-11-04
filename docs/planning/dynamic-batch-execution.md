# Dynamic [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) through one reusable [prepared execution](../glossary.md#prepared-execution--preparedexecution)

## Purpose and selected direction

[Coordinate](../glossary.md#axis-rank-scalar-vector-and-matrix) true per-[run](../glossary.md#term-run) symbolic [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) `B` through one [immutable](../glossary.md#term-immutable) [compiled graph](../glossary.md#compiled-graph--compiledgraphmodel) and [prepared
execution](../glossary.md#prepared-execution--preparedexecution). The user selected this direction on 2026-10-24. The
[Prepare0012 proposal](modules/prepare/dynamic-batch-geometry-contract-proposal.md) is Complete as
reviewed DESIGN delivery: independent Class C APPROVE, zero open P1/P2, and MAIN delivery acceptance.
User “ano” now explicitly approves authority cutover, subsequent ordinary [CPU](../glossary.md#term-cpu) forward/actual
first-order backward and bounded final A/B. [Prepare0012A](modules/prepare/tasks/0012a-dynamic-batch-authority-cutover.md)
is Complete: authority enacted/MAIN-accepted 2026-10-24 after Pasteur definitive APPROVE, zero open
P1/P2, docs/glossary review complete.
[Prepare0013](modules/prepare/tasks/0013-ordinary-dynamic-batch-cpu-vertical.md#result) implements bounded
ordinary [homogeneous](../glossary.md#term-homogeneous) [FLOAT32](../glossary.md#term-float32) CPU forward and actual first-order backward and is Complete.
Final independent reviews and the native-enabled root gate passed; MAIN accepted it on 2026-10-24.
The design [brief](modules/prepare/tasks/0012-dynamic-batch-geometry-and-resource-contract.md)
retains launch/review evidence. This does not reopen the selected direction.
This is a non-authoritative coordination map, not an [architecture contract](../glossary.md#architecture-contract) or a broader capability grant.

The user's subsequent “ok, pokracuj” authorizes only
[Model0039 symbolic-Scan design](modules/model/tasks/0039-general-scan-symbolic-batch-design.md),
now Complete for DESIGN DELIVERY ONLY: independent frozen APPROVE, zero open P1/P2, and MAIN
acceptance on 2026-10-24, not by itself detailed semantic approval or implementation. The
[focused proposal](modules/model/general-scan-symbolic-batch-proposal.md) [targets](../glossary.md#term-target) all-valid
FLOAT32 CPU forward/actual first-order backward, fixed T/F/H and one reusable per-run B.
It explicitly covers symbolic multi-output [layout](../glossary.md#layout) [binding](../glossary.md#term-binding), source/[inference](../glossary.md#term-inference)/[AD](../glossary.md#autograd)/[Prepare](../glossary.md#prepare) guards
and CPU numeric-resource/generated-loop changes. Proposed new production types: zero;
existing [owners](../glossary.md#term-owner)/recipes are candidates, not an implementation proof. [Model](../glossary.md#term-model) source/[dtypes](../glossary.md#data-type--datatype) and
[Compiler](../glossary.md#term-compiler) AD [admission](../glossary.md#term-admission) remain separate from CPU execution [qualification](../glossary.md#term-qualification). PlanningAuthorized=true;
ImplementationAuthorized=false. At Model0039 acceptance, [planning](../glossary.md#term-planning)/implementation frontiers were None.

[Model0040](modules/model/tasks/0040-same-geometry-reuse-design-amendment.md) is Complete for DESIGN
AMENDMENT ONLY, authorized by “ok, dopln to” on 2026-10-25. It proposes one exact-preparation
immutable [metadata](../glossary.md#term-metadata) [memo](../glossary.md#cache-and-memoization), retaining per-call Compiler binding/obligations, input/storage [validation](../glossary.md#term-validation)
and fresh resources. Independent APPROVE, zero open P1/P2; MAIN accepted 2026-10-25.
The user then confirmed that direction and explicitly authorized the contract-first next step.
[Model0041](modules/model/tasks/0041-symbolic-scan-authority-cutover.md) is Complete for
authority/docs only. The accepted
[Scan semantic owner](../architecture/contracts/recurrent-scan.md#accepted-symbolic-batch-scan) and
[Engine memo owner](../architecture/contracts/runtime-prepare-engine.md#accepted-engine-geometry-memo)
now distinguish shareable immutable numeric facts/private backing from [mutable](../glossary.md#term-mutable) run arrays and
retain current input-backed [descriptors](../glossary.md#term-descriptor) plus full fresh Compiler bind/private obligations.
Independent frozen APPROVE, zero open P1/P2; MAIN accepted 2026-10-25.
PlanningAuthorized=true; AuthorityAuthorized=true; ImplementationAuthorized=false.
At that authority acceptance, planning/authority/implementation frontiers were None.
No new type/public [signature](../glossary.md#term-signature) or [atomic](../glossary.md#resource-transaction-and-rollback) mechanism was selected by that authority-only delivery.
The separately user-admitted [Model0042](modules/model/tasks/0042-symbolic-scan-source-foundation.md)
implements static/bounded-symbolic Model source only; independent APPROVE, zero open P1/P2,
configured root test/[Javadoc](../glossary.md#term-javadoc) PASS and MAIN acceptance 2026-10-25 are recorded in its brief.
At that source acceptance, implementation frontiers were None; Compiler0014 below is now Complete.

The user next accepted MAIN's recommendation to prepare and then implement
[Compiler0014](modules/compiler/tasks/0014-symbolic-scan-inference-ad-and-binding.md), a compact
now Complete packet for independent symbolic inference, actual first-order AD and scoped all-port
`BatchBindingPlan` binding through existing owners. PlanningAuthorized=true;
ImplementationAuthorized=true after user source approval and MAIN [Ready](../glossary.md#term-ready) verification 2026-10-25.
Independent planning and code/docs APPROVE, zero open P1/P2; MAIN accepted 2026-10-25 after
[module](../glossary.md#term-module)/Javadoc and scoped compatibility validation. At that acceptance, frontier None. Forward inference retains
accepted Model [PURE](../glossary.md#term-pure)/dtype breadth, separately from bounded [F32](../glossary.md#floating-point-precision-and-rounding)/[F64](../glossary.md#floating-point-precision-and-rounding) AD and later initial CPU F32
qualification. No symbolic execution/cold/memo enablement, new type/signature or automatic successor.

Completed packet: [Prepare0014 symbolic Scan CPU vertical](modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md#result)<span data-glossary-navigation="true"> ([symbolic Scan glossary](../glossary.md#symbolic-scan))</span>,
MAIN accepted 2026-10-25 after independent code/docs APPROVE and the configured source-matched
root gate. PlanningAuthorized=true; ImplementationAuthorized=true record historical admission. User “jdi na to”
authorized source; MAIN verified Ready 2026-10-25 17:00 UTC with Complete predecessors, accepted planning
[snapshot](../glossary.md#term-snapshot) and inactive conflicts/sole writer. One atomic F32 forward/actual-backward capability
across existing Prepare/CPU/[Engine](../glossary.md#term-engine) owners is delivered; memo remains excluded. Bounded CPU/Engine
measurements and their limits are recorded in the brief, not a parity guarantee. Source frontier None;
Parallel group None, MAIN completed shared-document integration and the vertical join. No [commit](../glossary.md#term-commit)/push/successor.

[Model0043 symbolic-B plus runtime lengths](modules/model/tasks/0043-symbolic-length-scan-integration-design.md)<span data-glossary-navigation="true"> ([runtime glossary](../glossary.md#term-runtime))</span>
is Complete for DESIGN DELIVERY ONLY: Hume the 4th final Class C APPROVE, 0 P1/P2/no unresolved
findings; MAIN accepted 2026-10-26 after the verified planning-only admission/launch.
The [bounded integration proposal](length-aware-general-scan.md#proposed-symbolic-b-plus-lengths-integration)
targets F32/F64 forward/actual first-order reverse and identifies the narrow outer F64/control
prerequisites. Historical PlanningAuthorized=true; AuthorityAuthorized=false; ImplementationAuthorized=false.
At that design acceptance, planning/source/authority frontiers were None. The subsequent explicit
user semantic/source approval admits only [Model0044](modules/model/tasks/0044-symbolic-length-scan-source-foundation.md):
static/one-B length source, bounded root/[scan](../glossary.md#term-cumulative-scan)-[owner](../glossary.md#term-owner) wording before Java and deliberate consumer
rejection. Source is implemented under the authority/Ready acceptance of 2026-10-26, with independent
source/docs APPROVE and 0 P1/P2. MAIN's configured root test/Javadoc [checkpoint](../glossary.md#term-checkpoint) passed and MAIN
accepted Model0044 as Complete on 2026-10-26. Active Model frontiers are None.
Compiler0015 subsequently implements the separately approved inference/actual first-order AD/fresh
scoped all-port binding intersection. Its [Result](modules/compiler/tasks/0015-symbolic-length-scan-inference-ad-and-binding.md#result)
records its completed review, validation and MAIN acceptance.
Prepare0015 implements bounded combined Prepare/CPU/Engine execution and is Complete, MAIN accepted
2026-10-27; its [Result](modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result) records
validation and [nonstationary](../glossary.md#term-nonstationary) timing limits. Engine memo and new [API](../glossary.md#term-application-programming-interface-api)/type remain outside that task.

Active frontier: None. [Prepare0015 symbolic-length Scan CPU vertical](modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result)
is Complete, MAIN accepted 2026-10-27 after independent code/docs APPROVE (0 P1/P2), owning validation,
the sole bounded probe and configured root test/Javadoc PASS. Complete Model0044 + Compiler0015 +
CPU0042/Prepare0014 and retained predecessors feed the delivered F32/F64 vertical, including bounded
outer/control/ROW(1). No new type/API/memo or prerequisite chain. Probe gaps remain nonstationary
and unexplained, not performance parity or a [JIT](../glossary.md#term-just-in-time-compilation-jit)-cause qualification. No successor is authorized.

[Compile](../glossary.md#compile) retains symbolic meaning; prepare fixes code, routes, slots and typed recipes; each run
supplies actual numeric geometry and owns its resources. Existing `DynamicDimension` and
`DimensionExpressions` remain. [Feature](../glossary.md#term-software-feature) and [parameter](../glossary.md#term-parameter) [dimensions](../glossary.md#dimension) stay fixed; the first [general
Scan](../glossary.md#general-differentiable-scan) extension also keeps time `T` fixed. Variable `T` is a later coarse phase, not solved by lengths.
There is no mandatory public `B_max`, required per-B full-preparation [cache](../glossary.md#term-cache) or [graph](../glossary.md#graph) rebuild per batch.
Allocate actual per-run bytes, not mandatory maximum-capacity banks.

## Current truth

Symbolic [Model](../glossary.md#term-model) values alone do not grant executable [binding](../glossary.md#term-binding). Runtime0010's completed task name
“dynamic execution” means [mutable](../glossary.md#term-mutable) [run state](../glossary.md#run-state--runstate), not symbolic-[extent](../glossary.md#term-extent) [admission](../glossary.md#term-admission). Completed briefs and
all existing static paths, including their empty-domain behavior, remain unchanged.

Prepare0013 implements the ordinary named-B path across the existing [owners](../glossary.md#term-owner) below. Its domain is
[rank](../glossary.md#term-rank)-at-most-two boundaries plus qualified [AD](../glossary.md#autograd) [views](../glossary.md#term-view-meanings), generated [scalar](../glossary.md#term-scalar) single-range [CPU](../glossary.md#term-cpu) execution;
other operations, [dtypes](../glossary.md#data-type--datatype), [layouts](../glossary.md#layout) and workflows do not gain support by analogy.

| Boundary | Current source and consequence |
|---|---|
| Symbolic meaning | [DynamicDimension](../../modules/model/src/main/java/io/github/pho001/synaptik/model/shape/DynamicDimension.java) is [identity](../glossary.md#term-identity-meanings)-only. [DimensionExpressions.unknown](../../modules/model/src/main/java/io/github/pho001/synaptik/model/shape/DimensionExpressions.java) creates a distinct unknown, not bounds attached to named B. |
| Logical output and semantic binding | [LogicalMemoryRequirement](../../modules/planning/src/main/java/io/github/pho001/synaptik/planning/memory/LogicalMemoryRequirement.java) retains [descriptors](../glossary.md#term-descriptor), not bytes. [BatchBindingPlan](../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/BatchBindingPlan.java) preindexes [metadata](../glossary.md#term-metadata) and private typed diagnostic obligations; binding derives consistent B and lazily resolves requested descriptors. |
| Descriptor and [prepare](../glossary.md#prepare) | [TensorDescriptor](../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorDescriptor.java) still rejects resolved layouts on dynamic [shapes](../glossary.md#shape); [PrepareContext](../../modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/analysis/PrepareContext.java)<span data-glossary-navigation="true"> ([PrepareContext glossary](../glossary.md#prepare-context-preparecontext))</span> admits bounded ordinary and all-valid [symbolic Scan](../glossary.md#symbolic-scan) named-B projections alongside static descriptors. |
| Resources | [PreparationResourceRequirement](../../modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/analysis/PreparationResourceRequirement.java) and [PreparedMemoryPlan](../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/memory/PreparedMemoryPlan.java)<span data-glossary-navigation="true"> ([PreparedMemoryPlan glossary](../glossary.md#prepared-memory-plan--preparedmemoryplan))</span> [carry](../glossary.md#term-carry) fixed/[run](../glossary.md#term-run)-size declarations and explicit mode. [RunMemoryGeometry](../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/RunMemoryGeometry.java)<span data-glossary-navigation="true"> ([RunMemoryGeometry glossary](../glossary.md#run-memory-geometry--runmemorygeometry))</span> resolves exact sizes; [PreparedRepresentationPlan](../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/resource/PreparedRepresentationPlan.java)<span data-glossary-navigation="true"> ([PreparedRepresentationPlan glossary](../glossary.md#prepared-representation-plan--preparedrepresentationplan))</span> creators receive actual bytes/[alignment](../glossary.md#term-alignment), and CPU allocates fresh storage per run. |
| [Engine](../glossary.md#term-engine) admission and output | [AdvancedEngine](../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/AdvancedEngine.java) validates semantic/numeric geometry and storage before [borrowing](../glossary.md#term-borrowing) exact prefixes. [RunResult.Publication](../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/RunResult.java)<span data-glossary-navigation="true"> ([RunResult glossary](../glossary.md#run-result--runresult); [Publication glossary](../glossary.md#term-publication))</span> receives actual [occurrence](../glossary.md#term-occurrence) descriptors. Static descriptor matching remains unchanged. |
| [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) and Training admission | [ModelExecutionState.validateValues](../../extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/ModelExecutionState.java) and [TrainingSession.bindValues](../../modules/training/src/main/java/io/github/pho001/synaptik/training/TrainingSession.java)<span data-glossary-navigation="true"> ([TrainingSession glossary](../glossary.md#training-session--trainingsession))</span> retain static external matching; dynamic one-state reuse is separately gated. |
| CPU binding and loops | [CpuPreparedExecutable](../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutable.java) validates actual bytes. Prepared numeric access/[MATMUL](../glossary.md#matrix-multiplication--matmul)/aggregate recipes bind invocation B once; generated loops keep fixed code/routes/carriers. Exact [FLOAT32](../glossary.md#term-float32) SUM/[MEAN](../glossary.md#term-mean) uses five limbs with conditional 0/48-byte scratch. |
| [General Scan](../glossary.md#general-differentiable-scan) | [CpuGeneralScanEmitter](../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneralScanEmitter.java) retains static baked loops and separately emits bounded [F32](../glossary.md#floating-point-precision-and-rounding) symbolic forward/actual reverse with invocation-bound numeric geometry. Prepare0014 is independently reviewed and MAIN accepted; fixed T/[features](../glossary.md#term-software-feature)/state, named B. Symbolic cold/copy/wire, lengths/skipping and [memo](../glossary.md#cache-and-memoization) remain unsupported. |

Existing [Engine lifecycle tests](../../modules/engine/src/test/java/io/github/pho001/synaptik/engine/EngineTypedLifecycleTest.java)<span data-glossary-navigation="true"> ([lifecycle glossary](../glossary.md#lifecycle))</span>,
[run creation tests](../../modules/runtime/src/test/java/io/github/pho001/synaptik/runtime/run/RunStateCreationTest.java)
and [Training fresh-value tests](../../modules/training/src/test/java/io/github/pho001/synaptik/training/TrainingSessionValuesTest.java)
provide static reuse/admission/cleanup evidence, not dynamic-B [qualification](../glossary.md#term-qualification).
The focused [Engine dynamic-B tests](../../modules/engine/src/test/java/io/github/pho001/synaptik/engine/EngineOrdinaryDynamicBatchTest.java),
[CPU semantic/Class-File tests](../../modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuOrdinaryDynamicBatchTest.java),
[conformance tests](../../testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/CpuOrdinaryDynamicBatchConformanceTest.java)<span data-glossary-navigation="true"> ([conformance glossary](../glossary.md#term-conformance))</span>
and [public integration test](../../testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineCpuOrdinaryDynamicBatchIntegrationTest.java)<span data-glossary-navigation="true"> ([integration test glossary](../glossary.md#term-integration-test))</span>
provide bounded ordinary dynamic forward/actual first-order backward evidence.

## Qualified reuse scenario

Goal: vary only [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) size without recompilation or preparation. Assume an admitted [FLOAT32](../glossary.md#term-float32)
ordinary [CPU](../glossary.md#term-cpu) [slice](../glossary.md#slice) with `X[B,4]`, fixed `W[4,3]`, `Y=X@W`, and [scalar](../glossary.md#term-scalar) `mean(Y)`.
This implemented/tested scenario is not runnable Java or a [dtype](../glossary.md#data-type--datatype)-wide promise; the
[running guide](../user-guide/running-models.md) supplies the public seeded scalar-[loss](../glossary.md#loss) example.

| [Run](../glossary.md#term-run) | Actual B | Actual Y [descriptor](../glossary.md#term-descriptor) [shape](../glossary.md#shape) | Y [payload](../glossary.md#artifact-envelope-and-atomic-publication) bytes |
|---|---:|---|---:|
| 1 | 64 | `[64,3]` | 768 |
| 2 | 40 | `[40,3]` | 480 |
| 3 | 1 | `[1,3]` | 12 |
| 4 | 64 | `[64,3]` | 768 |

Each row uses the same compiled/prepared [identities](../glossary.md#term-identity-meanings) and fresh run-owned writable storage; all
other resources are sized separately from their actual requirements. With all-one X/W, Y contains
4 and the mathematical [mean](../glossary.md#term-mean) is 4; each [weight](../glossary.md#term-weight) [gradient](../glossary.md#term-gradient) is 1/3 for every listed B. Input gradients
have actual `[B,4]` descriptors; [parameter](../glossary.md#term-parameter) gradients remain `[4,3]`. A changed [feature](../glossary.md#term-feature) width or
inconsistent B across inputs rejects at [admission](../glossary.md#term-admission). Qualified B=0 produces empty Y/dX, scalar [NaN](../glossary.md#term-nan)
empty MEAN and nonempty typed-zero `[4,3]` dW; it does not skip the whole run. The retained Y+L
witness has 16 units, 19 [buffers](../glossary.md#term-buffer) and four workspaces; the public explicit-[seed](../glossary.md#term-seed) timing closure has
20 buffers/four workspaces. These are observed closures, not universal resource totals.

## Accepted boundaries and remaining gates

1. **[Compile](../glossary.md#compile) [handoff](../glossary.md#term-handoff).** Preserve symbolic descriptors and exact typed obligations through the
   compiler-owned boundary; diagnostic strings are not predicates. Preserve the eight<span data-glossary-navigation="true"> ([descriptors glossary](../glossary.md#term-descriptor))</span>
   `CompileArtifacts` components unless a later explicit decision changes that contract.
2. **[Run](../glossary.md#term-run) [admission](../glossary.md#term-admission).** Before borrowing/creation/writes, validate input ranks, fixed axes, consistent
   B bindings, exact dtype and requiresGrad, admitted layout/strides/offsets, storage accessibility,
   alignment/capacity, checked element/byte spans, overflow and deferred shape constraints.
   Dynamic-template inputs require resolved zero-offset canonical dense reads; absent layouts reject.<span data-glossary-navigation="true"> ([borrowing glossary](../glossary.md#term-borrowing); [bindings glossary](../glossary.md#term-binding); [dtype glossary](../glossary.md#data-type--datatype); [layout glossary](../glossary.md#layout); [alignment glossary](../glossary.md#term-alignment); [element glossary](../glossary.md#term-element); [overflow glossary](../glossary.md#term-overflow); [shape glossary](../glossary.md#shape); [canonical glossary](../glossary.md#term-canonical-form))</span>
3. **Resource declarations.** Fixed/run-size recipes, explicit mode and actual-size/alignment
   creators implement the accepted handoff. All requirements, slot identities and representation positions are known
   before finalization; prepared typed recipes derive exact run sizes without discovering resources.
   Prepare stages/validates the handoff but neither executes nor interprets backend geometry.
   Concrete backends own physical allocation and numeric address mechanics.<span data-glossary-navigation="true"> ([identities glossary](../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([Prepare glossary](../glossary.md#prepare); [backend glossary](../glossary.md#backend))</span>
4. **[Runtime](../glossary.md#term-runtime) isolation.** Runtime keeps no Model/compiler dependency, Shape/DimensionExpression/
   Operation/graph interpreter, route search, code generation or cache lookup. The typed
   numeric handoff binds once per invocation, never per element or scan step.
   One RunState owns all mutable geometry, addresses, carry banks, temporaries and resources;
   no mutable B, working numeric arrays or storage is written into shared prepared state.
   The accepted future Engine-only metadata exception reuses immutable exact-plan facts, not
   Runtime cache lookup, mutable working arrays or resources; it remains unimplemented.<span data-glossary-navigation="true"> ([Model glossary](../glossary.md#term-model); [dependency glossary](../glossary.md#term-dependency); [Operation glossary](../glossary.md#operation); [graph glossary](../glossary.md#graph); [code generation glossary](../glossary.md#term-code-generation); [cache glossary](../glossary.md#term-cache); [lookup glossary](../glossary.md#term-lookup); [RunState glossary](../glossary.md#run-state--runstate); [mutable glossary](../glossary.md#term-mutable); [carry glossary](../glossary.md#term-carry); [Engine glossary](../glossary.md#term-engine); [metadata glossary](../glossary.md#term-metadata); [immutable glossary](../glossary.md#term-immutable))</span>
5. **[Publication](../glossary.md#term-publication) and [lifetime](../glossary.md#term-lifetime).** Hand actual run output descriptors to Engine publication and
   materialization, preserving occurrence identities and gradient roles. Keep borrowed-input
   lifetime, attempt-all reverse cleanup, rollback/retry, prepared close/run leases, result leases,
   NN exclusion and concurrent-run isolation. Failure publishes no partial result; it never closes
   caller-owned input storage. Concurrent results with different B must remain independent.<span data-glossary-navigation="true"> ([materialization glossary](../glossary.md#materialization); [occurrence glossary](../glossary.md#term-occurrence); [borrowed glossary](../glossary.md#ownership-borrowing-and-lease); [rollback glossary](../glossary.md#term-rollback); [leases glossary](../glossary.md#term-lease); [NN glossary](../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
6. **Separate future [Scan](../glossary.md#term-cumulative-scan) preflights.** Shape/byte geometry is checked at run admission. Length VALUE checks
   occur after length producers execute and before this Scan's first output/padding/history/scratch
   writes; never move produced-length checks to run start. Validate all actual B lengths in <span data-glossary-navigation="true"> ([producers glossary](../glossary.md#term-producer))</span><span data-glossary-navigation="true"> ([padding glossary](../glossary.md#padding))</span>`[0,T]`
   at that point, including the last invalid value and T=0 cases under the accepted empty policy.

For the future [symbolic Scan](../glossary.md#symbolic-scan) extension, [TBF](../glossary.md#term-tbf) (time/[batch](../glossary.md#term-batch-machine-learning-and-hyperparameters)/[features](../glossary.md#term-software-feature)) time [stride](../glossary.md#term-stride) is `B*F`;
row histories/banks depend on `B*H`, while demanded shared
histories have no invented B [axis](../glossary.md#term-axis). Check aggregate sizes/offsets and representability limits before
allocation; do not add B×[parameter](../glossary.md#term-parameter) [gradient](../glossary.md#term-gradient) banks. Preserve zero bytes/stores for dead residuals.
Empty domains cannot globally skip a different nonempty reduction or shared output.

[ReductionGradientRules.mean/count](../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ReductionGradientRules.java)
already uses `oneLike(input).sum(...)`, with
[DerivativeConstants.oneLike](../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/FirstOrderAutograd.java)
expanding a typed one. [Compiler](../glossary.md#term-compiler) does not bake a denominator of 64: dynamic reduction geometry must
make that existing algebra correct. Parameter [MATMUL](../glossary.md#matrix-multiplication--matmul) adjoints may contract B. Training accumulation
is a sum of supplied gradients, not automatic [sample](../glossary.md#term-sample-machine-learning-and-hyperparameters) weighting; mixed-batch [objective](../glossary.md#term-objective) weighting
must stay explicit. Scan-local lengths imply neither a masked outer [loss](../glossary.md#loss) nor a changed [MEAN](../glossary.md#term-mean) domain.

## [Owners](../glossary.md#term-owner) and coarse [dependency](../glossary.md#term-dependency) order

The [architecture scope index](../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([scope glossary](../glossary.md#term-scope))</span> remains
the sole authority router. This map describes current [ownership](../glossary.md#term-ownership-meanings) and separately gated extensions;
the exact nine-type inventory remains in the task and scoped contracts.

| [Owner](../glossary.md#term-owner) | Decision responsibility |
|---|---|
| [Model](../glossary.md#term-model), [Compiler](../glossary.md#term-compiler) and [Planning](../glossary.md#term-planning) | [Symbol](../glossary.md#term-symbol) [identity](../glossary.md#term-identity-meanings)/expressions, [descriptor](../glossary.md#term-descriptor) semantics, typed constraints, actual [derivative](../glossary.md#term-derivative) algebra and symbolic logical requirements. |
| [Prepare](../glossary.md#prepare) | Cross-owner [handoff](../glossary.md#term-handoff), stable slots/positions, declaration/creator coverage and transactional preparation; no [backend](../glossary.md#backend) geometry interpreter. |
| [Runtime](../glossary.md#term-runtime) | Model-free prepared contracts, [run](../glossary.md#term-run)-owned numeric state, creation orchestration, [leases](../glossary.md#term-lease), cleanup and cold-bound schedule execution. |
| [CPU](../glossary.md#term-cpu) | Prepared numeric recipes, [lowering](../glossary.md#lowering)/emission, actual-byte [binding](../glossary.md#term-binding), code identity and independently qualified forward/backward loops. |
| [Engine](../glossary.md#term-engine) | Public input [admission](../glossary.md#term-admission) and actual-descriptor [publication](../glossary.md#term-publication)/[materialization](../glossary.md#materialization) through the existing prepared owner. |
| [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) and Training (future dynamic reuse) | Separately qualify one owned [parameter](../glossary.md#term-parameter)/[optimizer state](../glossary.md#term-optimizer-state) across B values through public Engine, preserving exclusion and transactions. NN acquisition and symbolic [persistence](../glossary.md#term-persistence) qualify separately below. |

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

The general-[Scan](../glossary.md#term-cumulative-scan) branch has completed reviewed/MAIN-accepted Model0039 and Model0040 design
deliveries. User semantic approval and authority admission followed; Model0041 is Complete
for coordinated accepted-future authority/ADR0051/docs, not implementation. Independent frozen
APPROVE and MAIN acceptance (2026-10-25) precede separately admitted Model source, Compiler actual [AD](../glossary.md#autograd)/all-port
binding and Prepare/CPU/Engine vertical gates. Model0042 is the separately authorized source-only
delivery, Complete after independent review and MAIN acceptance; no downstream successor is [Ready](../glossary.md#term-ready).
Its immediate Compiler0014 successor is Complete: scoped [inference](../glossary.md#term-inference), actual first-order AD and
all-port binding are implemented; [module](../glossary.md#term-module)/[Javadoc](../glossary.md#term-javadoc) PASS (384 tests), independent APPROVE (0 P1/P2)
and MAIN's scoped compatibility [checkpoint](../glossary.md#term-checkpoint) PASS (68 tests), accepted 2026-10-25. Dependency/base/
conflict, launch verification and evidence are recorded in that brief. Prepare0014 is separately
Complete after numeric/[conformance](../glossary.md#term-conformance) checks, bounded CPU/Engine probes, independent reviews and
the configured source-matched root join. Its brief records results and measurement limits.
The [memo](../glossary.md#cache-and-memoization)'s exact publication/close mechanism remains later scoped work: after inward admission,
close suppresses publication without a new outward-close error, and Runtime's sole lease arbitrates
the later [race](../glossary.md#term-race). CPU0042 now completes fixed-geometry length skipping; Model0044 admits the source
foundation for symbolic-B plus lengths and Compiler0015 adds inference/actual AD/all-port binding,
not execution. NN/Training reuse remains separately gated; neither this planning
delivery nor the fixed-length capability admits symbolic-length execution or state reuse.

NN entry points qualify separately: `Model.graph` and `defineTensor` need future symbolic
Compiler [cold reconstruction](../glossary.md#term-cold-reconstruction) and Checkpoint copy support; symbolic [graph](../glossary.md#graph) save/load additionally
needs its own wire [qualification](../glossary.md#term-qualification). Builder acquisition bypasses that copy path and requires
independent qualification, not inferred support. Existing static saved Models/readers remain
supported. None of NN cold acquisition, symbolic persistence or builder qualification blocks the
ordinary low-level Engine+CPU vertical.

Static length support is not cancelled and need not wait for distant NN/persistence work. The
ordinary geometry handoff is implemented; further conflicting CPU binder/resource changes require
separate admission. Variable T, native/[OpenBLAS](../glossary.md#term-openblas)/[Metal](../glossary.md#term-metal), mixed-backend transfers, persistence and tuning retain
separate later qualification. [SourceGraphReconstruction](../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/SourceGraphReconstruction.java)
and [GraphArtifactCodec](../../modules/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/graph/GraphArtifactCodec.java)
are static; Metal is not qualified, and OpenBLAS invokes plan m/n/k. No support follows by inference.

## [Admission](../glossary.md#term-admission) and acceptance gates

Prepare0012 reviewed DESIGN delivery: Complete, independent Class C APPROVE (zero open P1/P2),
MAIN delivery accepted 2026-10-24. Prepare0012A Complete, authority enacted/MAIN-accepted 2026-10-24;
Pasteur definitive APPROVE, zero open P1/P2, docs/glossary review complete. MAIN reverified all five dependencies plus
design Complete at 2026-10-24 17:41:11 UTC, HEAD/origin/index unchanged, baseline 2435 files.
Prepare0013 is Complete after frozen implementation, successful worker [validation](../glossary.md#term-validation), final
code/[Javadoc](../glossary.md#term-javadoc)/performance and shared-document APPROVE with zero open P1/P2, and MAIN's successful
native-enabled root join. MAIN accepted the capability on 2026-10-24; [authorized frontier](../glossary.md#term-authorized-frontier) None.
Base: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus accepted uncommitted work,
not merged predecessors. MAIN owns serial shared-document integration; Parallel group None.
No automatic successor execution or [commit](../glossary.md#term-commit)/push is authorized.

MAIN admitted Model0039 as the sole [planning](../glossary.md#term-planning)-only frontier on this same accepted dirty base,
with Complete Model0035/0038, Compiler0013, Prepare0013 and CPU0037 verified. The brief owns
admission and all seven integration fields. No competing writer; Parallel group None; MAIN owns
shared integration. Model0039 is now Complete for reviewed DESIGN DELIVERY ONLY, MAIN accepted
2026-10-24; planning frontier was None at that acceptance, no automatic successor launch.
[Prepare](../glossary.md#prepare)/[Compiler](../glossary.md#term-compiler)/[CPU](../glossary.md#term-cpu) implementation frontiers remain None. T=0 [qualification](../glossary.md#term-qualification) uses a separate
fixed-T plan; B=0 never globally skips required nonempty shared work or [parameter](../glossary.md#term-parameter) [gradients](../glossary.md#term-gradient).
Current authority frontier: None; Model0041 is Complete. Complete fresh bind, all scoped ports and
bounded [rank](../glossary.md#term-rank)-three actual closure, [immutable](../glossary.md#term-immutable)-only [memo](../glossary.md#cache-and-memoization) snapshots and current input-backed [metadata](../glossary.md#term-metadata)
are future matching validation obligations, not evidence supplied by this document-only cutover.

The task Result records 64→40→1→64 reuse, actual outputs/gradients/resources, admission,
empty/[overflow](../glossary.md#term-overflow)/cleanup/retry/[concurrency](../glossary.md#term-concurrency), retained static paths and generated-versus-clean-Java
algorithm/Class-File evidence. Bounded timing separates cold setup from invocation measurements;
the original harness predates the final [MATMUL](../glossary.md#matrix-multiplication--matmul) repair and one authorized post-fix probe also exists.
The latter's generated/direct [sample](../glossary.md#term-sample-machine-learning-and-hyperparameters) ratios are tiny MATMUL 0.968–1.337, medium MATMUL 1.008–1.035
and exact [MEAN](../glossary.md#term-mean) 1.398–1.414. Its callsite/[warmup](../glossary.md#term-warmup)/input changes are joint confounds, not a replacement
acceptance row or isolated B tax. The original unfavorable ranges, unmatched medium [Engine](../glossary.md#term-engine) routes
and differing allocation [topology](../glossary.md#term-topology) remain disclosed; no general parity or negligible-overhead claim follows.
No certificates, general proof engine, new session/executor or second [graph](../glossary.md#graph)/[DAG](../glossary.md#term-directed-acyclic-graph-dag) is proposed.

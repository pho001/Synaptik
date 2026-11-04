# Length-aware [general scan](../glossary.md#general-differentiable-scan) and first-order [AD](../glossary.md#autograd): selected design

## Reading context: which samples have a real step

An application can pack sequences of different lengths into one rectangular Tensor by
adding [padding](../glossary.md#padding). It supplies a non-gradient INT64 vector containing
one valid length $\ell_b$ for each batch row $b$. For a time extent $T$,
$0\le\ell_b\le T$; a physical step $t\in\{0,\ldots,T-1\}$ is active for that row only when
$t<\ell_b$. A zero-length row has no active steps.

The distinction is **skip before loading or calculating**, not calculate a padded value and
then multiply it by zero. This protocol controls the Scan body; it is not an automatic mask
for every loss in the model. Read the
[current length protocol](../architecture/contracts/recurrent-scan.md#length-aware-whole-batch-scan)
and [concrete Scan API](../reference/tensor/general-scan.md) before interpreting the dated
implementation scopes below. The [dynamic-batch program](dynamic-batch-execution.md) explains
the separate question of changing the number of batch rows between calls.

## Purpose and status

Specify selected [runtime](../glossary.md#term-runtime) valid lengths for the existing [general Scan](../glossary.md#general-differentiable-scan) pipeline, so padded rows are skipped
before arithmetic or loads during execution. [Compiler](../glossary.md#term-compiler) [automatic differentiation](../glossary.md#autograd) ([AD](../glossary.md#autograd))
now constructs the existing sibling reverse [scan](../glossary.md#general-differentiable-scan), not an [RNN](../glossary.md#term-recurrent-neural-network-rnn)-specific [adjoint](../glossary.md#term-adjoint) or another execution system.

**Source/[IR](../glossary.md#term-intermediate-representation-ir) and bounded Compiler support Complete; MAIN accepted 2026-10-24.** Model0036 remains Complete for proposal
delivery only after independent final APPROVE/MAIN acceptance. The subsequent user approval
“ok, pust se do prace” selects this design and authorizes immediate authority plus first [Model](../glossary.md#term-model)
foundation. [Model0037](modules/model/tasks/0037-length-aware-general-scan-authority.md) is Complete
for authority only, accepted by MAIN on 2026-10-24 after Hume's final overall independent Class C
APPROVE, zero open P1/P2. [Model0038](modules/model/tasks/0038-length-aware-scan-source-foundation.md)
is Complete, MAIN-accepted 2026-10-24 after independent Class C/[API](../glossary.md#term-application-programming-interface-api)/docs APPROVE, zero open P1/P2,
and combined [compile](../glossary.md#compile)/architecture/root retry PASS after a test-only ledger repair. Model frontier None;
Compiler0013 is Complete, MAIN accepted 2026-10-24. No fluent migration, higher-order
waiver, later [owner](../glossary.md#term-owner) execution, [commit](../glossary.md#term-commit)/push or distant task cascade is authorized.

After the accepted [planning](../glossary.md#term-planning) review, user “ok, pokracuj dal” authorizes only
[Compiler0013 implementation](modules/compiler/tasks/0013-length-aware-scan-capture-and-first-order-ad.md).
MAIN recorded launch [admission](../glossary.md#term-admission) 2026-10-24 after Popper final APPROVE (0 P1/P2).
The frozen implementation delivers [inference](../glossary.md#term-inference-meanings), existing first-order protocol extensions and final
role/demand queries. Independent code/docs APPROVE (0 P1/P2), worker [module](../glossary.md#term-module), MAIN compatibility
and final docs/[Javadoc](../glossary.md#term-javadoc) PASS; MAIN accepted Complete 2026-10-24.
At that Compiler0013 [checkpoint](../glossary.md#term-checkpoint), cold [capture](../glossary.md#term-capture-meanings)/reconstruction, [CPU](../glossary.md#term-cpu) and [codec](../glossary.md#term-codec) support remained closed;
the later Prepare0011 and CPU0042 completions are recorded below.

[Prepare0011](modules/prepare/tasks/0011-length-aware-scan-projection-and-resource-handoff.md) Complete;
MAIN accepted 2026-10-24 13:00:15 UTC. At that completion checkpoint, [Prepare](../glossary.md#prepare) [Ready](../glossary.md#term-ready) frontier was None. Tesla planning APPROVE,
zero open P1/P2; MAIN [metadata](../glossary.md#term-metadata) accepted 2026-10-24. ImplementationAuthorized=true records completed Prepare0011 only, from
the user's latest “pust se do prace” request. MAIN verified admission 2026-10-24 12:25 UTC;
clean sole writer started 2026-10-24 12:31 UTC. Worker
71-focused/92-full tests and Javadoc PASS, zero failures/errors/skips; MAIN [SPI](../glossary.md#term-service-provider-interface-spi)/four-[guard](../glossary.md#term-guard) checkpoint
PASS. Independent code/docs APPROVE, zero open P1/P2; MAIN acceptance complete.
No new launch, downstream tasks, commit or push is authorized.
Complete Model0037/0038 + Compiler0013 + Planning0008 + Prepare0010 feed this existing-owner static
projection/resource [handoff](../glossary.md#term-handoff). At that checkpoint CPU mixed-boundary/[preflight](../glossary.md#term-preflight)/skipping and cold/v5/[NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context)
remained held.

The dynamic-B geometry decision/authority and ordinary/symbolic CPU handoffs are now Complete,
including [Prepare0014](modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md#result); no obsolete
geometry-decision prerequisite blocks static length execution.
[CPU0042](backends/cpu/tasks/0042-length-aware-general-scan-execution.md#result) is Complete, MAIN accepted
2026-10-26 after independent Class C code/docs APPROVE and the recorded CPU/Javadoc, named joined
checkpoint and sole bounded probe. It implements fixed T/B/[features](../glossary.md#term-feature) [F32](../glossary.md#floating-point-precision-and-rounding)/[F64](../glossary.md#floating-point-precision-and-rounding) forward and actual
first-order reverse, including mixed [INT64](../glossary.md#data-type--datatype) control, [producer](../glossary.md#term-producer)-ordered preflight and skipping.
The probe is [nonstationary](../glossary.md#term-nonstationary), not steady-state parity or a compiler-cause proof. The brief owns
evidence and limits. Source/test frontier None; no successor or additional measurement is authorized.
Length skipping and dynamic-B combination have independent qualifications; Prepare0015 now
implements the bounded combined [scope](../glossary.md#term-scope) below, Complete after MAIN acceptance on 2026-10-27.
[Model0043](modules/model/tasks/0043-symbolic-length-scan-integration-design.md) delivered only the
[bounded integration proposal](#proposed-symbolic-b-plus-lengths-integration) below. Subsequent explicit
user semantic/source approval now admits [Model0044](modules/model/tasks/0044-symbolic-length-scan-source-foundation.md),
Complete under the source authority accepted on 2026-10-26. Independent source/docs review and
MAIN's root test/Javadoc checkpoint passed; MAIN accepted the task on 2026-10-26. This source capability
retains source semantics independently of downstream enablement. Compiler0015 now implements
combined inference/actual first-order AD/fresh scoped all-port [binding](../glossary.md#term-binding); completion evidence is in its
[Result](modules/compiler/tasks/0015-symbolic-length-scan-inference-ad-and-binding.md#result).
Prepare0015 implements combined Prepare/CPU/[Engine](../glossary.md#term-engine) execution; independent code/docs review and
configured gates passed. Its [Result](modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result)
records MAIN acceptance and nonstationary timing limits, not performance parity. [Shape](../glossary.md#shape)/byte checks do not
read lengths; length VALUE checks follow producers and precede this Scan's first writes. No cold/v5,
NN migration, variable T or model-wide [loss](../glossary.md#loss) [mask](../glossary.md#term-boolean-mask) is granted.

The current [scan contract](../architecture/contracts/recurrent-scan.md#general-differentiable-scan)
governs all-valid Scan and the [length protocol](../architecture/contracts/recurrent-scan.md#length-aware-whole-batch-scan).
This planning document does not own semantics; [ADR0049](../design/decisions/0049-length-aware-general-scan.md)
records the accepted decision. Model0038 now implements bounded source construction and structural
IR counts. Compiler0013 adds bounded compile-time support, Prepare0011 static projection/resource
handoff and CPU0042 the separately qualified generated execution path; cold/[persistence](../glossary.md#term-persistence) remains gated.
The [mental model](../glossary.md#term-mental-model) stays: ordinary [Tensor](../glossary.md#tensor) body → Compiler capture/AD → Prepare → one typed emitted
CPU loop → existing Checkpoint and owned NN workflows. No module edge or execution owner changes.

## Proposed symbolic-B plus lengths integration

Active frontier: None. [Prepare0015 symbolic-length Scan CPU vertical](modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result)
is Complete, MAIN accepted 2026-10-27 after independent code/docs APPROVE (0 P1/P2), owning [validation](../glossary.md#term-validation),
the sole bounded probe and configured root test/[Javadoc](../glossary.md#term-javadoc) PASS. Complete Model0044/Compiler0015 and
retained [Prepare](../glossary.md#prepare)/[CPU](../glossary.md#term-cpu)/[lifecycle](../glossary.md#lifecycle) predecessors feed the delivered [F32](../glossary.md#floating-point-precision-and-rounding)/[F64](../glossary.md#floating-point-precision-and-rounding) capability, including bounded
outer F64/control/ROW(1), combined projection/resources/generated forward/actual reverse and [Engine](../glossary.md#term-engine)
[publication](../glossary.md#term-publication). No new type/[API](../glossary.md#term-application-programming-interface-api)/[memo](../glossary.md#cache-and-memoization) or successor. Timing remains [nonstationary](../glossary.md#term-nonstationary) and does not qualify parity.

Model0043 is Complete for DESIGN DELIVERY ONLY: Hume the 4th final independent Class C APPROVE,
0 P1/P2/no unresolved findings; MAIN accepted 2026-10-26. Historical PlanningAuthorized=true;
AuthorityAuthorized=false; ImplementationAuthorized=false. MAIN confirmed its launch after [Ready](../glossary.md#term-ready)
[admission](../glossary.md#term-admission). This proposal combines one named B with the existing length header/overload for fixed
T/[features](../glossary.md#term-feature), [homogeneous](../glossary.md#term-homogeneous) F32 **and** F64 forward and actual [Compiler](../glossary.md#term-compiler)-generated first-order reverse.
That design delivery itself approved none of these capabilities. Subsequent user approval first
delivered Model0044's static/one-B source foundation, including coordinated source wording
before Java and explicit downstream rejection. Model0044 source is Complete after independent
source/docs review, configured root test/Javadoc PASS and MAIN acceptance on 2026-10-26.
The separately approved Compiler0015 implements [inference](../glossary.md#term-inference-meanings)/actual [AD](../glossary.md#autograd)/all-port [binding](../glossary.md#term-binding) and is
Complete after MAIN acceptance. Prepare0015 now implements the bounded real Prepare/CPU/Engine path
under the accepted pre-Java gate above; MAIN accepted the completed vertical on 2026-10-27.
The later cold/v5/[NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) phases elsewhere in this document are not dependencies of this low-level [slice](../glossary.md#slice).

### One-preparation witness and unchanged semantics

Conceptual acceptance scenario, not runnable/current support: choose T=4, F=3, H=2, symbolic
h0 `[B,2]`, X `[4,B,3]`, Wx `[3,2]`, Wh `[2,2]`, [bias](../glossary.md#term-bias) `[2]`, and the existing body
`next=tanh(xStep@Wx+h@Wh+bias)`, emitting next. For each [dtype](../glossary.md#data-type--datatype)/traversal/ingress variant, [compile](../glossary.md#compile)
and [prepare](../glossary.md#prepare) once; keep [graph](../glossary.md#graph), code, routes, slots, representation positions and resource [topology](../glossary.md#term-topology)
unchanged throughout this sequence. Supply nonuniform [finite](../glossary.md#term-finite) data and joint final/emission [seeds](../glossary.md#commit-release-distribution-log-and-seed-meanings).
Two fixed graph variants qualify separately: direct non-[gradient](../glossary.md#term-gradient) [INT64](../glossary.md#data-type--datatype) L `[B]`, and a non-gradient
same-body-dtype raw-length input `[B]` followed by ordinary `cast(INT64)`. Ingress does not switch
within a preparation. The [producer](../glossary.md#term-producer) variant proves that [admission](../glossary.md#term-admission) cannot read L before execution.

Read each row as a new [run](../glossary.md#term-run) with fresh actual [descriptors](../glossary.md#term-descriptor)/storage and the same prepared [owner](../glossary.md#term-owner):

| Run | B | L for zero-based row b | Observable check |
|---|---:|---|---|
| 1 | 32 | `b % 5` | Mixed/empty/full prefixes; final `[32,2]`, emissions `[4,32,2]` |
| 2 | 32 | `4 - (b % 5)` | Different results at the same geometry; no retained length facts |
| 3 | 7 | `b % 5` | Actual smaller [buffers](../glossary.md#term-buffer)/history; fixed [parameter](../glossary.md#term-parameter)-gradient [shapes](../glossary.md#shape) |
| 4 | 1 | `[0]` | Raw h0/final-[seed](../glossary.md#term-seed) bypass, positive-zero emissions/dX, typed-zero parameter sums |
| 5 | 32 | all 4 | Full prefixes through the original preparation, not a new all-valid route |

Expected values/gradients come from independent active-prefix [unrolling](../glossary.md#term-unrolling) and specialized clean Java,
not a handwritten reverse substituted for [Compiler](../glossary.md#term-compiler) [AD](../glossary.md#autograd). Repeat B=0 separately: L/dX/emissions are
empty, nonempty final shared [accumulator](../glossary.md#accumulator-type) outputs are typed zero, and no shared/hoisted [Scan body](../glossary.md#backpropagation-through-time-and-recurrent-state)
executes. Required other graph work still executes. A separate fixed-T=0 preparation checks every
L is zero before writes and preserves [carry](../glossary.md#term-carry)/dInitialCarry [identities](../glossary.md#term-identity-meanings); also cover T=0/B=0 together.
Add invalid-last-length, wrong B/type, F=B/H=B, [signed-zero](../glossary.md#nan-infinity-and-signed-zero)/[NaN payload](../glossary.md#nan-infinity-and-signed-zero) carry and poisoned padded
sequence/emission-seed cases; failures are Scan-local, not [rollback](../glossary.md#term-rollback) of already executed producers.

Keep existing length semantics: original-source AD requires every carry and every original sequence
ROW(0), including unused/non-gradient positions; broader forward validity does not grant AD.
Generated reverse keeps ROW(0), [rank-two](../glossary.md#axis-rank-scalar-vector-and-matrix) ROW(1) and SHARED [provenance](../glossary.md#provenance) explicitly, never inferring
roles from equal sizes or from B=0/1. Preserve joint seeds, positional ties/cross-[occurrence](../glossary.md#term-occurrence) sums,
false-flag carry propagation and exact saved [TANH](../glossary.md#term-tanh)/[SIGMOID](../glossary.md#term-sigmoid) outputs, never [activation](../glossary.md#term-activation) replay.
Inactive rows perform no dependent loads/arithmetic; carry and empty-row dInitialCarry preserve
raw initial/seed bits. [Padding](../glossary.md#padding)/dX is raw +0; active [NaN](../glossary.md#term-nan)/Inf retains constituent meaning.
At each active time, reduce active contributors in increasing b, then one distinct represented
shared-carry ADD; inactive times bypass it. Accumulator/contribution banks are O(P), not B×P.
Demanded row histories cost O(T×B×row-[elements](../glossary.md#term-element)), SHARED histories O(T×P) with no invented B [axis](../glossary.md#term-axis);
save previous carry before the step and produced primals at their exact sites. Dead histories,
including reverse residuals, allocate/store zero. Retain raw carry backing independently of
zero-padded histories; a demanded port is not automatically a legal inactive-carry bank.

### Bounded source map and concrete [F64](../glossary.md#floating-point-precision-and-rounding) prerequisite

The following map is the retained **pre-implementation inspection [snapshot](../glossary.md#term-snapshot)**, not current support
status. Its required changes are implemented by Prepare0015 through the same [owners](../glossary.md#term-owner); current [scope](../glossary.md#term-scope)
is in the [CPU contract](../architecture/contracts/backend-execution.md#accepted-ordinary-symbolic-batch-cpu-contract).
No new production type/[API](../glossary.md#term-application-programming-interface-api) or production replacement/deletion was selected.

| Seam / source evidence | Existing reusable facts and required change |
|---|---|
| [Scan](../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Scan.java) `validateRows`; [ScanBodyInference](../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ScanBodyInference.java) `infer/validateRows/rowResult` | Model0044 source and Compiler0015 [inference](../glossary.md#term-inference-meanings) now use exact named-[dimension](../glossary.md#dimension)/role checks for the [homogeneous](../glossary.md#term-homogeneous) [F32](../glossary.md#floating-point-precision-and-rounding)/F64 intersection. Static [canonical](../glossary.md#term-canonical-form) [layouts](../glossary.md#layout), unresolved symbolic templates, the bounded source algebra and unrelated guards remain. ScanAttrs/header/overload are unchanged. |
| [ScanGradientRules](../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ScanGradientRules.java), [BatchBindingPlan](../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/BatchBindingPlan.java), [PrepareContext](../../modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/analysis/PrepareContext.java)<span data-glossary-navigation="true"> ([PrepareContext glossary](../glossary.md#prepare-context-preparecontext))</span> | Compiler0015 retains the actual sibling/[primal](../glossary.md#term-primal)-[binding](../glossary.md#term-binding) protocol, pruned mixed roles, all latent ports and [occurrence](../glossary.md#term-occurrence)-scoped obligations. Existing binder production passes complete fresh [admission](../glossary.md#term-admission), including unused/tied inputs and [INT64](../glossary.md#data-type--datatype) L, without a production edit. [Prepare](../glossary.md#prepare) still deliberately rejects the combination; its future projection change needs separate [qualification](../glossary.md#term-qualification). |
| [CpuGeneralScanLowering](../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuGeneralScanLowering.java) `analyzeSymbolic/lowerSymbolic/Facts.pack`; CpuGeneralScanReductionEmitter (historical source path: `../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneralScanReductionEmitter.java`; no longer present in this checkout) `emitSymbolicFold` | Symbolic analysis fixes F32 values/[scalars](../glossary.md#term-scalar)/casts and four-byte sequence/frame/cut/[carry](../glossary.md#term-carry) packing; symbolic folds emit `0f/fadd`. Parameterize by the proved floating [dtype](../glossary.md#data-type--datatype)/width and emit double folds for F64. Existing typed scalar/[carrier](../glossary.md#term-storage-carrier)/[MATMUL](../glossary.md#matrix-multiplication--matmul) and fixed-length emitters are reusable, not sufficient qualification of this new combination. |
| [CpuPartitionLowering](../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java) `supportsOrdinaryBatch/batchByteSizes/lowerOrdinaryBatch`; [CpuPartitionPreparer](../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java) `analyzeOrdinaryBatch/analyzeGeneralScan` | Ordinary symbolic admission/[IR](../glossary.md#term-intermediate-representation-ir)/[specialization](../glossary.md#term-specialization) fix F32, four-byte spans/[alignment](../glossary.md#term-alignment) and float-array-or-segment carriers. Qualify only actual outer closure occurrences below with per-value dtype/width, checked-segment carriers for added cases, and separate eight-byte INT64 control. Existing initialized-[buffer](../glossary.md#term-buffer) recipes already fill typed F64/INT64 spans. No [heap](../glossary.md#term-heap)-carrier or ordinary-op-wide [widening](../glossary.md#term-widening) is required. |
| [CpuAggregateLowering](../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuAggregateLowering.java) `BatchGeometry/stateLimbCount`; [CpuExactSumEmitter](../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuExactSumEmitter.java) | Static exact reductions/emission are typed; symbolic SUM/[MEAN](../glossary.md#term-mean) is F32-only, five limbs/48 bytes. A real F64 prerequisite is dtype-qualified exact numeric geometry/state and compatible specialization, not [FLOAT32](../glossary.md#term-float32) reduction followed by cast. Scan's producing-site SUM stays its distinct typed sequential algorithm. |
| [CpuGeneralScanIr](../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuGeneralScanIr.java) `Stages`; [CpuGeneralScanPreparedExecutable](../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuGeneralScanPreparedExecutable.java) `executeBound` | Stages forbids length+symbolic; the length recheck uses static hooks/fixed [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters). Replace mutual exclusion only after proving combined control/roles/recipes. Recheck with [run](../glossary.md#term-run)-aware exact spans/type/alignment/overlap after [producers](../glossary.md#term-producer), then validate all actual B values before ANY output/[padding](../glossary.md#padding)/history/scratch write. Static hook rejection must not be removed wholesale. |

Select the outer prerequisite from the **actual generated closure**, not every [backend](../glossary.md#backend) [operation](../glossary.md#operation):
F32/F64 registered [seed](../glossary.md#term-seed)/zero/one splats; ADD for tied/cross-occurrence terms; required MUL/DIV and
SUM/MEAN/sumToShape; expand, [feature](../glossary.md#term-feature)-only reshape/[rank](../glossary.md#term-rank) edits, [transpose](../glossary.md#term-transpose)/[contiguous](../glossary.md#term-contiguous) and same-dtype
CAST where that closure uses them. In-body [rank-two](../glossary.md#axis-rank-scalar-vector-and-matrix) MATMUL remains in the typed Scan emitter;
this witness needs no new ordinary symbolic MATMUL capability. No other ordinary family gains F64.
For both dtypes, `ScanGradientRules.rowConstant/stackedShape` also produces ROW(1) literal/zero
sequences `[T,F,B]`: the current ordinary rank-three [guard](../glossary.md#term-guard) rejects nonsingleton T/F. Qualify this
mapping and `[T,B,F]` seed/history plumbing, not arbitrary rank-three operations. Ordinary provider
admission has no generated-by-Scan-[AD](../glossary.md#autograd) [provenance](../glossary.md#provenance): use explicit family/dtype/[shape](../glossary.md#shape)/attribute/access
predicates, with the same capability answer for identical qualifying occurrences outside Scan.
No provenance [registry](../glossary.md#term-registry) or new [metadata](../glossary.md#term-metadata) is proposed. Scan-internal Dimension/ROW provenance remains
the independent semantic proof above, not an ordinary provider origin test.
Add only the non-[gradient](../glossary.md#term-gradient) `[B]` F32→INT64 and F64→INT64 CAST [producer](../glossary.md#term-producer) with existing cast semantics;
direct INT64 lengths need no producer. No general integral arithmetic or cross-dtype body is added.

For ordinary F64 exact SUM/MEAN, existing static sizing gives signed bits
`2099 + ceil(log2(N))` ([cardinality](../glossary.md#term-cardinality) term zero for N≤1). For admitted int-bounded N, at most 2130 bits
fit 34 limbs; the accepted [target](../glossary.md#term-target) uses a fixed 34-[limb](../glossary.md#exact-superaccumulator-and-limbs) specialization, 280-byte single-range scratch when output
cells exist, zero otherwise. F32 retains five limbs/48 bytes and its exact old bytes. Keep workspace
[identity](../glossary.md#term-identity-meanings) even at zero bytes. Qualify F64 exact sum/rational-mean [rounding](../glossary.md#term-rounding), [subnormals](../glossary.md#term-subnormal), signed zeros,
[NaN](../glossary.md#term-nan)/Inf and empty domains with the existing typed algorithm; this accepted qualification is not measured
parity or permission to change static reductions. [ReductionGradientRules.mean/count](../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ReductionGradientRules.java)
already uses `oneLike(input).sum(...)` and restored division: `mean(finalCarry)` divides by actual
B×H, not active length sum or [prototype](../glossary.md#term-prototype) B. At B=0 it is NaN; row gradients are empty and [parameter](../glossary.md#term-parameter)
sums still typed zero. Source validity, first-order AD validity and [CPU](../glossary.md#term-cpu) capability remain separate.
Static `stateLimbCount` remains domain-sized: F64 N=3 still uses 33 limbs/272 bytes including flags,
not the accepted, still-unimplemented symbolic int-bounded 34-limb/280-byte specialization.

### Numeric resources, [admission](../glossary.md#term-admission) and compatibility

Use current staged declarations/creators, typed access recipes, [RunMemoryGeometry](../glossary.md#run-memory-geometry--runmemorygeometry) and one [RunState](../glossary.md#run-state--runstate).
Geometry admission reads [descriptors](../glossary.md#term-descriptor) only; [Engine](../glossary.md#term-engine)'s current semantic/byte admission and [publication](../glossary.md#term-publication)
are [dtype](../glossary.md#data-type--datatype)-generic, not a reason for an Engine [memo](../glossary.md#cache-and-memoization) or new [handoff](../glossary.md#term-handoff). Preindex every outer/scoped
domain and latent port; bind exact checked numeric facts before [borrowing](../glossary.md#term-borrowing), including empty leading
domains with overflowing trailing [features](../glossary.md#term-feature). Carry B explicitly in the combined [run](../glossary.md#term-run)-owned numeric
pack, not `firstFrame/H`: `[B,0]` still requires B and all B length checks. All changing arrays,
addresses, banks and results remain invocation-owned. Reuse fixed symbolic time offsets B×F and
row frames B×H, separately width-scaled; [INT64](../glossary.md#data-type--datatype) L is eight bytes per row, never a floating formal.
Declare all banks/cuts/demanded histories and structural zero-byte workspaces before slots; prove
cut [lifetime](../glossary.md#term-lifetime)/size dominance for every B or keep disjoint regions. No Bmax banks, per-B [compile](../glossary.md#compile),
reprepare/[cache](../glossary.md#term-cache), [topology](../glossary.md#term-topology) selection, [runtime](../glossary.md#term-runtime) [graph](../glossary.md#graph) walk or per-[element](../glossary.md#term-element)/time [callback](../glossary.md#term-callback) is introduced.

Extend existing CpuGeneralScanEmitter with an explicitly combined path: numeric symbolic setup
plus current length activity/terminal-contributor rules. B=0/all-zero L permits [identity](../glossary.md#term-identity-meanings)/zero output
initialization but no hoisted/shared body work; it is not a whole-run shortcut. Preserve [immutable](../glossary.md#term-immutable)
entry-[carrier](../glossary.md#term-storage-carrier) [bindings](../glossary.md#term-binding) from CPU0041 where proved safe, independent formal-carry selection and all
saved-site guards. Do not select generic fallback overhead merely because both modes coexist.
Use the accepted distinct combined [canonical](../glossary.md#term-canonical-form)/baked markers `:symbolic-length-v1` and
`|scan-symbolic-length-v1=` in both [specialization](../glossary.md#term-specialization) streams, covering dtype, roles, terminal/demand,
control position, numeric pack and baked recipes. Runtime B/L/bytes/addresses/slots stay excluded.
Retain [schema](../glossary.md#term-schema)/family 71, static shaped/[MATMUL](../glossary.md#matrix-multiplication--matmul) v4, all-valid symbolic v3, fixed-length v1 and ordinary
[F32](../glossary.md#floating-point-precision-and-rounding) identities/generated bytes exactly. Added outer [F64](../glossary.md#floating-point-precision-and-rounding)/control/new [rank](../glossary.md#term-rank)-three cases use
`ordinary-batch-typed-v1` in [IR](../glossary.md#term-intermediate-representation-ir) and both specialization streams, keying exact dtype/carrier/mapping/
reduction-state facts, with validated-generation and [artifact](../glossary.md#term-artifact) load/reuse/stale-rejection tests.

### Minimal coordinated delta and serial capability order

Source and [Compiler](../glossary.md#term-compiler) authority now qualify the bounded symbolic-length intersection; no generic new
architecture requirement is needed. Source/Compiler and Prepare0015's bounded execution are implemented.
The following records the coordinated [qualification](../glossary.md#term-qualification) delta, not new semantics or a successor plan:

- recurrent-scan `Header, bindings and source admission`: retain static T/features but permit one
  exact named B for length-mode row axes/L, with unresolved templates and static actual runs.<span data-glossary-navigation="true"> ([features glossary](../glossary.md#term-feature); [axes glossary](../glossary.md#term-axis))</span>
  `Bounded row-role analysis` compares identified [dimensions](../glossary.md#dimension)/[provenance](../glossary.md#provenance), not sampled numeric sizes.
  `Accepted symbolic-batch scan`/`Qualification and mechanism gates` records the combined F32/F64
  Compiler-only slice without relaxing original-source AD or cold/NN exclusions; this is implemented.<span data-glossary-navigation="true"> ([F32 glossary](../glossary.md#floating-point-precision-and-rounding); [slice glossary](../glossary.md#slice); [NN glossary](../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- [runtime](../glossary.md#term-runtime)-[prepare](../glossary.md#prepare)-[engine](../glossary.md#term-engine) `modules/prepare` and `Prepare lifecycle`: replace static-only length
  projection wording for this qualified combined case. Existing resource, Run lifecycle and Engine
  rules already prescribe the handoff/preflight; only precise support-status updates are needed.<span data-glossary-navigation="true"> ([Run glossary](../glossary.md#term-run); [lifecycle glossary](../glossary.md#lifecycle); [handoff glossary](../glossary.md#term-handoff); [preflight glossary](../glossary.md#term-preflight))</span>
- [backend](../glossary.md#backend)-execution `CPU backend routes` and `Accepted ordinary symbolic-batch CPU contract`:
  add only combined F32/F64 and the named outer/control prerequisite, including exact F64 state,
  while retaining the old F32 route. Compiler headings need matching qualification/status, not a
  new AD algorithm, binder API or Runtime dependency. Root lifecycle/invariants/index get matching
  scoped status links only. Authority/explanation/ADR updates, if approved, accompany owner changes
  before conflicting code; none was performed by Model0043. Prepare0015 enacted that bounded gate
  after independent pre-Java review and MAIN acceptance.<span data-glossary-navigation="true"> ([dependency glossary](../glossary.md#term-dependency); [ADR glossary](../glossary.md#term-architecture-decision-record-adr); [owner glossary](../glossary.md#term-owner))</span>

Read the following as coarse capabilities, not authorized tasks:

| Serial phase | Complete capability / entry condition |
|---|---|
| Source | Model0044 is Complete for static/one-B length source and existing overloads. Independent source/docs APPROVE and MAIN root test/[Javadoc](../glossary.md#term-javadoc) PASS; MAIN accepted 2026-10-26. Source alone grants no downstream capability. |
| Compiler | Compiler0015 is Complete for symbolic length [inference](../glossary.md#term-inference-meanings), actual first-order sibling [AD](../glossary.md#autograd) and fresh scoped all-port [binding](../glossary.md#term-binding) together. Mixed generated roles/[constants](../glossary.md#term-constant) and old AD restrictions are preserved. |
| Prepare→[CPU](../glossary.md#term-cpu)→Engine vertical | Prepare0015 Complete: bounded outer [F64](../glossary.md#floating-point-precision-and-rounding)/control/ROW(1), combined typed projection/resources, generated forward/actual reverse and Engine [admission](../glossary.md#term-admission)/[publication](../glossary.md#term-publication); independent reviews/configured [checkpoint](../glossary.md#term-checkpoint) PASS, MAIN accepted 2026-10-27. |

Model0044 is Complete; active [Model](../glossary.md#term-model) frontiers are None. Its bounded authority was accepted before
a different clean worker implemented it; independent source/docs review and MAIN's root checkpoint
passed, with MAIN task acceptance on 2026-10-26. Compiler0015 is implemented and independently reviewed;
Prepare0015 is Complete after independent code/docs APPROVE, recorded gates and MAIN acceptance.
No further successor brief, repeated authority-only chain, new [API](../glossary.md#term-application-programming-interface-api)/type/executor/session or production
deletion is selected. Later phases stay coarse. Independent proposal review/final acceptance is
MAIN-owned; successful proposal delivery itself accepts neither new semantics nor implementation.

### Proportional future [validation](../glossary.md#term-validation), not runs now

Extend existing ScanSymbolicBatchSource/LengthAwareScan*/SymbolicScan* [Compiler](../glossary.md#term-compiler) tests and
CpuGeneralScan*/CpuSymbolicScan*/CpuOrdinaryDynamicBatch* focused tests; add no validation framework.
Use raw-[IR](../glossary.md#term-intermediate-representation-ir)/[descriptor](../glossary.md#term-descriptor) bypasses to exercise each independent validator, not source guards alone.
Ordinary provider tests must accept identical qualifying occurrences inside/outside the Scan closure
and reject cases outside the explicit [dtype](../glossary.md#data-type--datatype)/family/[rank](../glossary.md#term-rank)/[axis](../glossary.md#term-axis)/attribute/access predicates, including
unsupported casts, unproved [layouts](../glossary.md#layout) and independent symbols; never consult an [AD](../glossary.md#autograd)-origin tag.
Require actual reverse for both dtypes, both traversals/joint [seeds](../glossary.md#commit-release-distribution-log-and-seed-meanings), ROW(1) [constants](../glossary.md#term-constant)/shared [transpose](../glossary.md#term-transpose)
histories, F=B/B=0/1, all-zero/mixed/full/invalid-last lengths, exact saved [activation](../glossary.md#term-activation) bits, dead/live
history bytes, source AD rejections, [overflow](../glossary.md#term-overflow)/[aliases](../glossary.md#aliasing-and-arena)/cleanup/retry/close and concurrent mixed-B
results. Extend EngineCpuGeneralScanIntegrationTest's produced-CAST witness with the five-[run](../glossary.md#term-run)
sequence above and actual [publication](../glossary.md#term-publication)/resource sizes; qualify direct L separately. Retain fixed,
static length, symbolic all-valid, old [identity](../glossary.md#term-identity-meanings)/byte fixtures and existing cold/higher-order/[NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) gates.
Worker final checks are owning [module](../glossary.md#term-module) `test` plus [`javadoc`](../glossary.md#term-javadoc) once after source freeze ([Model](../glossary.md#term-model),
Compiler, then [Prepare](../glossary.md#prepare)/[CPU](../glossary.md#term-cpu)/[Engine](../glossary.md#term-engine) for the vertical), with focused tests during development.
MAIN runs one configured combined `./gradlew test javadoc` [checkpoint](../glossary.md#term-checkpoint) after ordered integration,
covering [conformance](../glossary.md#term-conformance)/integration and unchanged architecture boundaries, reusing successful evidence.
Use the established native environment without a rebuild; no duplicate review Java runs.

For future changed hot paths require optimal specialized clean-Java semantic/algorithm/hot-loop/
[dataflow](../glossary.md#term-dataflow) oracles, including [F64](../glossary.md#floating-point-precision-and-rounding) exact outer reductions, and ClassFile/decompile inspection for
hidden helpers/allocation/boxing/reflection/[dispatch](../glossary.md#term-dispatch). Use one bounded warmed matched-work
generated/direct check, same carriers/inputs/represented sites/preallocation; separate cold
generation/classload/[JIT](../glossary.md#term-just-in-time-compilation-jit), entry/[admission](../glossary.md#term-admission) and [kernel](../glossary.md#kernel) costs. Preserve CPU0042's [nonstationary](../glossary.md#term-nonstationary) limits:
neither its probe nor this plan is stable parity proof. No Java/native/benchmark or remeasurement
is run by Model0043, and no performance campaign, variable T/[features](../glossary.md#term-feature), cold/v5, NN/Training reuse,
fluent migration, higher-order, low/mixed, [vector](../glossary.md#term-vector)/parallel, [OpenBLAS](../glossary.md#term-openblas) or [Metal](../glossary.md#term-metal) [scope](../glossary.md#term-scope) follows.

## Preferred minimum and alternative

Prefer whole-[batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) bodies with **explicit boundary row roles and a closed local eligibility check**.
A row domain means dependence on exactly one batch row, not a [dimension](../glossary.md#dimension) whose numeric size equals B.
Keep existing body [shapes](../glossary.md#shape) and ordinary [MATMUL](../glossary.md#matrix-multiplication--matmul)/broadcast/SUM formulas. This requires genuinely new
length/[binding](../glossary.md#term-binding) semantics; lengths as an ordinary [invariant](../glossary.md#term-invariant) cannot specify skipping, output [padding](../glossary.md#padding),
[carry](../glossary.md#term-carry) [identity](../glossary.md#term-identity-meanings), active reduction contributors or saved-history roles.

| Candidate | Benefit | Cost and disposition |
|---|---|---|
| Bounded whole-batch eligibility | Retains existing source/formal shapes, saved ports and batched [derivative](../glossary.md#term-derivative) algebra. | Must explicitly classify bindings and reject unproved coupling; selected. |
| Explicit row-local body binding | Removing the batch [axis](../glossary.md#term-axis) from formals makes row isolation structural. | Changes actual/formal binding and history geometry; still needs shared [gradient](../glossary.md#term-gradient) folds and a specified accumulation order. Not selected. |

Neither candidate needs B independent [parameter](../glossary.md#term-parameter)-gradient accumulators. Replicating P parameter
[elements](../glossary.md#term-element) across B rows would cost B×P elements, versus O(P) shared accumulation; that is not the
preferred design. No general [tensor](../glossary.md#tensor)-axis proof framework or public row-body type is proposed.

## Implemented source [API](../glossary.md#term-application-programming-interface-api) and semantic header

Model0038 extends existing `model.operation.scan.ScanAttrs` with `rowCarryCount` and [immutable](../glossary.md#term-immutable) ordered
`sequenceRowAxes: List<OptionalInt>`. These implemented source [metadata](../glossary.md#term-metadata)/API additions use existing
standard-library values, with no new production class or enum.

- `rowCarryCount=0` and an empty axes list select unchanged all-valid semantics. Preserve the old
  seven-argument constructor and five-argument <span data-glossary-navigation="true"> ([axes glossary](../glossary.md#term-axis); [argument glossary](../glossary.md#term-argument))</span>`Scan.apply` as that default.
- Length mode has `1 <= rowCarryCount <= carryCount` and one axis entry per sequence. Present
  means ROW(axis in the step formal); absent means SHARED, with no batch axis. Initial scope uses
  axis 0 and the axis-1 rank-two transpose closure only. Outer time prepends one axis.<span data-glossary-navigation="true"> ([batch glossary](../glossary.md#term-batch-machine-learning-and-hyperparameters); [scope glossary](../glossary.md#term-scope); [rank-two glossary](../glossary.md#axis-rank-scalar-vector-and-matrix); [transpose glossary](../glossary.md#term-transpose))</span>
- Leading carries are ROW(0); trailing carries are SHARED terminal additive accumulators. Shared
  carry formals may occur only as the old operand of their own `old.add(contribution)` root;
  they cannot affect row roots, emissions, other accumulators or their own contribution.
- Actual order is [carry](../glossary.md#term-carry), sequence, [invariant](../glossary.md#term-invariant), then one non-[gradient](../glossary.md#term-gradient) [INT64](../glossary.md#data-type--datatype) `[B]` lengths Tensor.
  Lengths has no body formal. `inputCount()` therefore differs from formal count in this mode.
  Invariants are SHARED bindings; normal input sequences are explicitly ROW(0).
- Validate common static T/B, role-axis extents, formal/actual descriptors, counts and containment
  before allocating canonical output identities. Never infer roles from equal extents, rank or<span data-glossary-navigation="true"> ([identities glossary](../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([rank glossary](../glossary.md#term-rank))</span>
  `requiresGrad`. Repeated actual positions retain positional contributions and compatible roles.

**Implemented source overload:**

```java
ScanResult Scan.apply(List<Tensor> initialCarry, List<Tensor> sequences,
        List<Tensor> invariants, Tensor validLengths, int rowCarryCount,
        List<OptionalInt> sequenceRowAxes, ScanBody body, boolean reverse);
```

For an ordinary [RNN](../glossary.md#term-recurrent-neural-network-rnn)-shaped body, let [canonical](../glossary.md#term-canonical-form) storage-free formals be h `[B,H]`, x `[B,F]`,
w `[H,F]`, u `[H,H]`, [bias](../glossary.md#term-bias) `[H]`, with matching actual h0, values `[T,B,F]`, W/U/b and lengths.
The existing [Tensor](../glossary.md#tensor) operations remain the algebra; the call below constructs source metadata
only. Compiler0013 supports bounded [inference](../glossary.md#term-inference-meanings)/first-order [AD](../glossary.md#autograd), Prepare0011 transports static
[bindings](../glossary.md#term-binding) and CPU0042 separately qualifies fixed-geometry [F32](../glossary.md#floating-point-precision-and-rounding)/[F64](../glossary.md#floating-point-precision-and-rounding) execution. Cold [capture](../glossary.md#term-capture-meanings)/
reconstruction and encoding still reject length mode:

```java
Tensor next = x.matmul(w.permute(1, 0)).add(bias)
        .add(h.matmul(u.permute(1, 0))).tanh();
ScanBody body = new ScanBody(List.of(h), List.of(x), List.of(w, u, bias),
        List.of(next), List.of(next));
ScanResult result = Scan.apply(List.of(h0), List.of(values), List.of(W, U, b),
        validLengths, 1, List.of(OptionalInt.of(0)), body, false);
```

The selected result is dense `[T,B,H]` emissions and `[B,H]` final carry, with no [callback](../glossary.md#term-callback).
Execution rejects lengths outside `[0,T]` after [producers](../glossary.md#term-producer) and before scan-local writes;
[descriptor](../glossary.md#term-descriptor)-only construction does not read values or grant a [backend capability](../glossary.md#backend-capability) answer.

## Closed row-eligibility rules

Derive/check domains inside existing Scan source [validation](../glossary.md#term-validation) and `compiler.ScanBodyInference`;
derive corresponding [backend](../glossary.md#backend) facts in `CpuGeneralScanLowering`. Keep no general proof [artifact](../glossary.md#term-artifact).
Recheck complete bodies, [canonical](../glossary.md#term-canonical-form) siblings and actual reverse closure, even for zero lengths/T.
The implemented length-mode source/[inference](../glossary.md#term-inference-meanings) domain is [homogeneous](../glossary.md#term-homogeneous) [F32](../glossary.md#floating-point-precision-and-rounding)/[F64](../glossary.md#floating-point-precision-and-rounding) over the bounded
declared algebra, further restricted by these rules; ordinary all-valid source [admission](../glossary.md#term-admission) is not
narrowed. CPU0042 independently qualifies this fixed-geometry F32/F64 algebra and actual reverse.

- Elementwise/scalar operations preserve a row domain. Two row operands must identify the same
  row coordinate. SHARED operands may broadcast over features or a singleton row axis, not
  introduce a non-singleton row coordinate. Bias <span data-glossary-navigation="true"> ([features glossary](../glossary.md#term-feature))</span><span data-glossary-navigation="true"> ([Elementwise glossary](../glossary.md#term-elementwise); [scalar glossary](../glossary.md#term-scalar); [coordinate glossary](../glossary.md#axis-rank-scalar-vector-and-matrix); [singleton glossary](../glossary.md#term-singleton); [axis glossary](../glossary.md#term-axis); [Bias glossary](../glossary.md#term-bias))</span>`[H]` into `[B,H]` is supported; ambiguous
  same-rank broadcast patterns fail closed rather than treating an extent equal to B as proof.<span data-glossary-navigation="true"> ([rank glossary](../glossary.md#term-rank); [fail closed glossary](../glossary.md#admission-preflight-and-fail-closed-validation); [extent glossary](../glossary.md#term-extent))</span>
- [Rank-two](../glossary.md#axis-rank-scalar-vector-and-matrix) ROW(0) `[B,F]` @ SHARED `[F,H]` gives ROW(0). SHARED @ ROW(1) gives ROW(1) when
  contraction is over features. Rank-two <span data-glossary-navigation="true"> ([contraction glossary](../glossary.md#contraction-dimension))</span>`[1,0]` PERMUTE changes ROW(0) to ROW(1), and conversely;
  a SHARED transpose stays SHARED. This includes real W/U transpose values, not just RNN spelling.<span data-glossary-navigation="true"> ([transpose glossary](../glossary.md#term-transpose); [RNN glossary](../glossary.md#term-recurrent-neural-network-rnn))</span>
- ROW(1) `[F,B]` @ ROW(0) `[B,H]` is a shared reduction over batch contributors. Admit it only
  in the terminal shared contribution path, never as feedback into a row result. Other products
  contracting a row coordinate or mixing row coordinates reject. Shared-only products stay SHARED.<span data-glossary-navigation="true"> ([batch glossary](../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- Feature-only SUM/sumToShape preserves the row domain. A reduction removing the row axis is a
  terminal shared contribution, with the active contributor domain below. Shared-only SUM is
  ordinary SUM. No changed meaning is assigned to global SUM elsewhere in the outer graph.<span data-glossary-navigation="true"> ([graph glossary](../glossary.md#graph))</span>
- Expand, reshape, rank edits, contiguous and same-dtype cast must preserve the identified row
  coordinate and feature grouping. Initially allow row-leading feature edits and the stated
  rank-two transpose closure; reject flattening/reinterpreting batch into features or unproved
  layout/axis transformations. Shared-only local algebra remains SHARED.<span data-glossary-navigation="true"> ([contiguous glossary](../glossary.md#term-contiguous); [dtype glossary](../glossary.md#data-type--datatype); [layout glossary](../glossary.md#layout))</span>
- Emissions and row next-carry roots must be ROW(0). Reject row-reduction feedback, batch mixing,
  unsupported source/generated operations and unresolved mappings. Shared contribution paths can
  combine ordinary shared derivatives, but cannot turn row-reduced data back into row feedback.<span data-glossary-navigation="true"> ([carry glossary](../glossary.md#term-carry); [derivatives glossary](../glossary.md#term-derivative))</span>

[Model](../glossary.md#term-model) and [Compiler](../glossary.md#term-compiler) validate these rules with focused positive/negative tests; CPU0042 independently
proves row/terminal facts before activity-dominated emission.
The [owning bounded row analysis](../architecture/contracts/recurrent-scan.md#bounded-row-role-analysis)
pins exact broadcast, row-leading [feature](../glossary.md#term-feature)-reshape and terminal-contribution legality. New length-mode
Model source is homogeneous F32/F64 over that closed algebra; this is semantic admission, not a
backend predicate. Old all-valid [PURE](../glossary.md#term-pure)/type admission remains unchanged. A supported child alone is insufficient.

## Traversal, skipping and exact histories

Let `active(t,b) = t < L[b]` and `active(t) = exists b: active(t,b)`. Preserve original time order:
forward t=0..T−1, reverse t=T−1..0. At each time, visit [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) contributors in increasing b.
Reverse traverses each valid prefix L[b]−1..0, not the last L cells of the padded [tensor](../glossary.md#tensor).

Branch on activity **before** any padded sequence, [seed](../glossary.md#term-seed) or row-history load and before dependent
arithmetic. Inactive row state is raw [identity](../glossary.md#term-identity-meanings), with no `0*x`, `where`-after-compute or [accumulator](../glossary.md#accumulator-type)
update. Emissions and dSequence at inactive original-time [coordinates](../glossary.md#axis-rank-scalar-vector-and-matrix) are exact [positive zero](../glossary.md#nan-infinity-and-signed-zero).
Zero-length final [carry](../glossary.md#term-carry) is the represented initial carry; dInitialCarry is the represented
dFinalCarry seed. [NaN](../glossary.md#term-nan)/Inf in [padding](../glossary.md#padding) cannot contaminate valid values or [parameter](../glossary.md#term-parameter) sums; active
special values retain ordinary family semantics, not sanitization.

Keep `ScanBody.retainedValues()` and [canonical](../glossary.md#term-canonical-form) [residual](../glossary.md#scan-residual-port-and-history-demand) ports. Save previous carries before the
active step, and save produced primals at their exact represented sites, without [activation](../glossary.md#term-activation) replay.
Demand remains ordinary [publication](../glossary.md#term-publication)/use demand, including reverse residuals; dead history costs zero.

- ROW([axis](../glossary.md#term-axis)) retained values keep their existing `[T] + localShape` geometry. Store exact active
  row cells and raw positive zero in inactive cells, addressing the declared row axis, not shape.<span data-glossary-navigation="true"> ([shape glossary](../glossary.md#shape))</span>
- SHARED retained values have `[T] + localShape` and **no added B axis**. At active(t), store the
  exact value at its prescribed body site; at nonactive(t), store raw positive zero. Shared prior
  accumulator history is its pre-time value, before that time's single accumulator ADD.
- A shared history is read only at active(t); a row history/seed only at active(t,b). Forward
  zero-filling does not license reverse padded reads. Guard shared-only hoisted arithmetic and
  history publications too: no body work when every length is zero, including T=0/B=0.<span data-glossary-navigation="true"> ([Guard glossary](../glossary.md#term-guard))</span>

Crucial example: ordinary [linear](../glossary.md#term-linear-meanings) algebra retains shared `w.permute(1,0)` `[F,H]`. Its demanded
history is `[T,F,H]`, whereas an output seed may be `[T,B,H]`. Even when F=B, the reverse builder
passes SHARED for the former and ROW(0) for the latter. A transposed row history passes ROW(1).
`ScanGradientRules` derives these entries from the checked original [binding](../glossary.md#term-binding)/[dataflow](../glossary.md#term-dataflow) domains,
then prunes axes **together with** `usedStep`/`usedSequences`. The roles live in reverse `ScanAttrs`,
survive [capture](../glossary.md#term-capture-meanings), demand refresh and query; Prepare0011 implements their exact static projection, while cold/v5
roundtrip remains future work. They are not temporary [AD](../glossary.md#autograd)-only labels. `CompiledScanBody` keeps
its existing [graph](../glossary.md#graph)/role/demand representation.

## First-order reverse and shared accumulation

Current `ScanGradientRules.apply` [seeds](../glossary.md#commit-release-distribution-log-and-seed-meanings) leading reverse carries from final-[carry](../glossary.md#term-carry) cotangents,
appends [invariant](../glossary.md#term-invariant)-shaped zeros, and constructs `carry.get(c+i).add(gradients.get(c+s+i))`.
It binds original step/invariant/retained [Tensors](../glossary.md#tensor) through `primalBindings`, calls
`FirstOrderAutograd.expandBody`, and creates the opposite-traversal sibling Scan.

For initial first-order [qualification](../glossary.md#term-qualification), require every **original length-aware source** to have
`rowCarryCount == carryCount` AND every original `sequenceRowAxes` entry present and equal to 0
(ROW(0)), including unused and non-[gradient](../glossary.md#term-gradient) sequence positions. `ScanGradientRules.preflight`
must reject original trailing SHARED carries and original SHARED/ROW(1)/other nonzero-[axis](../glossary.md#term-axis)
sequences before [derivative](../glossary.md#term-derivative) allocation. A final shared-carry [seed](../glossary.md#term-seed) could affect row derivatives;
a ROW(1) sequence gradient would require an emission outside the ROW(0)-only result domain.
Broader forward validity remains separate and grants no [AD](../glossary.md#autograd) capability. Generated reverse inputs
may still mix explicit ROW(0), ROW(1) and SHARED sequence roles for seeds and exact saved primals;
this original-source restriction does not apply to them. The ordinary [TBF](../glossary.md#term-tbf) [RNN](../glossary.md#term-recurrent-neural-network-rnn) [target](../glossary.md#term-target) and old
all-valid AD are unchanged.

Within that bound, extend the protocol, not derivative formulas: original carry cotangents are
the ROW(0) prefix; the generated reverse appends SHARED invariant-gradient suffix carries. No
differentiation of that generated shared-carry scan is admitted. Carry the same lengths into the
reverse [occurrence](../glossary.md#term-occurrence) and keep joint final/emission seeds, false-flag internal carry propagation,
closed derivative [constants](../glossary.md#term-constant), tied positional contributions and cross-occurrence additions.
Lengths are discrete/non-gradient, not a new [differentiation target](../glossary.md#functional-gradient-request). Existing [residual](../glossary.md#scan-residual-port-and-history-demand)-root and
higher-order/createGraph rejection remain before derivative allocation.

At active(t), ordinary generated `sumToShape` and transposed [MATMUL](../glossary.md#matrix-multiplication--matmul) [parameter](../glossary.md#term-parameter) formulas reduce
**only active row contributors**, in original increasing-b order. The gate dominates both MATMUL
operand loads and each SUM contributor's dependent arithmetic. A per-output reduction starts at
the family's typed zero, uses its existing represented multiply/accumulation/result boundaries,
then the existing shared carry ADD rounds once at its own [dtype](../glossary.md#data-type--datatype) site for that time. Do not
silently fuse that ADD into the [contraction](../glossary.md#contraction-dimension) or replace it with per-row updates. At nonactive(t),
the shared [accumulator](../glossary.md#accumulator-type) is unchanged, not `old + 0`. [Feature](../glossary.md#term-feature) contractions/reductions still use
ordinary constituent domains; this active-row restriction belongs only to length-aware Scan.

Thus dW for x @ W uses active terms in x-[transpose](../glossary.md#term-transpose) @ dZ; [bias](../glossary.md#term-bias) uses active-row sumToShape.
Inactive row [cotangent](../glossary.md#cotangent) carries bypass the body, including padded [NaN](../glossary.md#term-nan) emission seeds. Shared
derivative work consumes exact shared histories only at active times. Active NaN final seeds
remain meaningful; an empty row's seed is copied to dInitialCarry without entering shared sums.

Use one invariant-shaped accumulator plus bounded same-[shape](../glossary.md#shape) contribution/old-new scratch as
needed by existing stage lifetimes, not B×parameter state. Shared residual history, when genuinely
demanded, can still cost O(T×P); row history O(T×B×retained-row-[elements](../glossary.md#term-element)). Do not hide those costs.
The new length-mode reduction [oracle](../glossary.md#term-test-oracle) does not promise bit-identical reductions to a differently
grouped whole-[batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) unroll. Existing all-valid Scan behavior and bytes remain unchanged.

### Numerical walkthrough (conceptual, not a measured [fixture](../glossary.md#term-fixture))

For hNext=h+x*w, emit=hNext, T=3/B=3, L=[3,1,0], h0=[10,20,30], w=2 and
x=[[1,4,[NaN](../glossary.md#term-nan)],[2,NaN,NaN],[3,NaN,NaN]], four transitions execute. Emissions are
[[12,28,0],[16,0,0],[22,0,0]], final=[22,28,30]. With [loss](../glossary.md#loss)=sum(emissions)+sum(final)=158,
dw=24, dh0=[4,2,1], dx=[[8,4,0],[6,0,0],[4,0,0]]. This illustrates shared accumulation,
empty-row [identity](../glossary.md#term-identity-meanings) and no [padding](../glossary.md#padding) contamination, not a masked-loss [API](../glossary.md#term-application-programming-interface-api) or [backend](../glossary.md#backend) [qualification](../glossary.md#term-qualification).

## [Prepare](../glossary.md#prepare), [runtime](../glossary.md#term-runtime) rejection and generated loop

Complete [Prepare0011](modules/prepare/tasks/0011-length-aware-scan-projection-and-resource-handoff.md)
implements existing [`PrepareContext`](../glossary.md#prepare-context-preparecontext) static [validation](../glossary.md#term-validation), exact header/body/role projection and
[`GraphPreparation`](../glossary.md#graph-preparation--graphpreparation) demand/resource [handoff](../glossary.md#term-handoff), with lengths separated from local formals. Prepare
does not duplicate [Model](../glossary.md#term-model)/[Compiler](../glossary.md#term-compiler) row-algebra proof or impose original-source [AD](../glossary.md#autograd) restrictions on
generated reverse roles. CPU0042 analysis declares all typed boundaries/resources before slot assignment;
the [INT64](../glossary.md#data-type--datatype) lengths read is distinct from the [homogeneous](../glossary.md#term-homogeneous) floating body. Existing
`CpuGeneralScanPreparedExecutable` binds ordered boundary [dtypes](../glossary.md#data-type--datatype) and separate eight-byte control,
not a fabricated floating [invariant](../glossary.md#term-invariant).

Reuse the fixed scan's **ordering principle**, not its executor: after length [producers](../glossary.md#term-producer) execute,
recheck accessible typed spans/[alignment](../glossary.md#term-alignment)/overlap, then validate all B length values against [0,T]
before this scan's first output, [padding](../glossary.md#padding), history or scratch write. A bad last length rejects
before earlier rows write; T=0 still validates every length. This is scan-local rejection, not
[rollback](../glossary.md#term-rollback) of upstream [graph](../glossary.md#graph) operations. Caller storage remains [borrowed](../glossary.md#ownership-borrowing-and-lease) and stable during the [run](../glossary.md#term-run).

`CpuGeneralScanLowering`, `CpuGeneralScanIr`, `CpuGeneralScanEmitter` and typed reduction/[MATMUL](../glossary.md#matrix-multiplication--matmul)
emitters use the proved activity [bindings](../glossary.md#term-binding). Hoisted shared work must be dominated by any-active
[admission](../glossary.md#term-admission); per-time shared work by active(t); row work by active(t,b). No physical packing,
compaction or sorting. One generated invocation, no interpreter/[callback](../glossary.md#term-callback)/per-step allocation.
All [mutable](../glossary.md#term-mutable) bindings/banks belong to the existing [RunState](../glossary.md#run-state--runstate); prepared recipes/code stay O(body).
Dense requested emissions remain O([TBH](../glossary.md#term-tbh)); row banks/stage cuts may be O(BH), shared sums O(P),
and only demanded histories grow with T. An empty first [carry](../glossary.md#term-carry) cannot skip other nonempty work.

CPU0042 validation uses a specialized optimal clean-Java loop with the same temporal order,
active contributor domain and represented sites, plus Class-File inspection and bounded matched-work
generated/direct evidence. The original design pass ran none of those checks; CPU0042's linked
Result records subsequent implementation validation and [nonstationary](../glossary.md#term-nonstationary) probe limits, not speed parity.
Version every changed length-mode [canonical](../glossary.md#term-canonical-form)/[specialization](../glossary.md#term-specialization) [identity](../glossary.md#term-identity-meanings) and keyed role/demand/baked
geometry fact; keep old all-valid/fixed identity streams and generated bytes on their old paths.

## Cold copy, [rebind](../glossary.md#capture-graph-ports-and-reconstruction) and [persistence](../glossary.md#term-persistence)

Extend `SourceGraphReconstruction.preflight/reconstructScan` and existing [Checkpoint](../glossary.md#term-checkpoint) [normalization](../glossary.md#term-normalization)
to preserve lengths role, row-[carry](../glossary.md#term-carry) count and ordered sequence [axes](../glossary.md#term-axis). [Preflight](../glossary.md#term-preflight) the complete source
and every body, including late/dead/T=0 occurrences, before formal/[producer](../glossary.md#term-producer) allocation or [host](../glossary.md#term-host)
copy/import. Reconstruct each body/[occurrence](../glossary.md#term-occurrence) once with positional substitutions and fresh scopes.
Preserve exact [primal](../glossary.md#term-primal) roles, all [canonical](../glossary.md#term-canonical-form) ports, [aliases](../glossary.md#aliasing-and-arena) and raw [descriptors](../glossary.md#term-descriptor); never unroll at save.

Current `GraphArtifactCodec` v4 [opcode](../glossary.md#term-opcode) 23 writes T/counts/reverse, actual IDs and an inline body;
it has **no length or sequence-domain fields**. Therefore unchanged-v4-[schema](../glossary.md#term-schema) compatibility is
not possible for this design. **Selected future format:** selective `graph-model.v5`, magic
`SYNGRF05`, version 5, new outer opcode 33 (verified free after current 1..32) for the changed
actual/header roles and existing inline body. Exact new field encoding belongs to the later [codec](../glossary.md#term-codec) task.
Preserve ordered optional step-axis entries explicitly; length values are [runtime](../glossary.md#term-runtime) input payloads,
never serialized snapshots. Preserve existing body inventory and no nested scan.

Continue selecting byte-identical v1–v4 for sources needing no length mode. Old readers reject v5;
new readers retain exact old tuples/opcodes and default old Scan records to all-valid. Preserve
the old ScanAttrs constructor as a source/[binary](../glossary.md#term-binary-descriptor-and-metadata)-call bridge, but acknowledge new record components
and overloads as public [API](../glossary.md#term-application-programming-interface-api)/record-shape changes needing compatibility review. Do not derive [cache](../glossary.md#term-cache)
[identity](../glossary.md#term-identity-meanings) from changed default record text and accidentally invalidate unchanged old paths.
Dry-size/count/descriptor/role checks include the new fields before allocation/import; no new
limits default or runtime histories/compiled state in the file. Wire and [CPU](../glossary.md#term-cpu) generator identities
are separate versioning duties, not one shared version number.

## [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) boundary and future [dependency](../glossary.md#term-dependency) map

Generic length-aware Scan comes first. Later owned [graph](../glossary.md#graph)/list Models would use existing
`Model.graph`, compute, configureTraining/train and save/loadGraph [owners](../glossary.md#term-owner) with explicit length
inputs. No new [RNN](../glossary.md#term-recurrent-neural-network-rnn) [API](../glossary.md#term-application-programming-interface-api), manager, executor or runner. Current public `.rnn`, `.rnnScan`,
`buildWithLengths`, fixed [inference](../glossary.md#term-inference)-only rejection, [GRU](../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../glossary.md#term-long-short-term-memory-lstm) and low-type routes remain unchanged.

Lengths govern **this scan only**. Later [bias](../glossary.md#term-bias)/head/[loss](../glossary.md#loss) operations remain ordinary dense operations;
zero emissions neither stop their [padding](../glossary.md#padding) arithmetic nor select a masked [MEAN](../glossary.md#term-mean) denominator.
Future NN/loss integration must be explicit, with no implicit [model](../glossary.md#term-model)-wide [mask](../glossary.md#term-boolean-mask).

| Coarse phase | Dependencies / [owner](../glossary.md#term-owner) | Status |
|---|---|---|
| Model0036 proposal delivery | Current accepted source baseline; independent final APPROVE and MAIN acceptance | Complete; proposal delivery only |
| Model0037 coordinated authority packet | User decision on reviewed proposal; Model/scan owner with affected boundary owners | Complete authority only; Hume final overall Class C APPROVE, zero open P1/P2; MAIN accepted 2026-10-24 |
| Model0038 source/[IR](../glossary.md#term-intermediate-representation-ir) foundation | Model0037 independent approval and MAIN acceptance Complete | Complete; MAIN accepted 2026-10-24 after independent APPROVE and combined integration retry PASS; source/IR and guards only |
| [Compiler0013 capture/inference/first-order AD](modules/compiler/tasks/0013-length-aware-scan-capture-and-first-order-ad.md)<span data-glossary-navigation="true"> ([capture glossary](../glossary.md#term-capture-meanings); [AD glossary](../glossary.md#autograd))</span> | Model0037/0038, Compiler0011 and Planning0008 Complete; existing cold baseline preserved | Complete; MAIN accepted 2026-10-24 after independent code/docs APPROVE, worker [module](../glossary.md#term-module), compatibility and final docs/[Javadoc](../glossary.md#term-javadoc) PASS; [evidence](modules/compiler/tasks/0013-length-aware-scan-capture-and-first-order-ad.md#result). No [runtime](../glossary.md#term-runtime)/cold/v5 grant |
| [Prepare0011 projection/resource handoff](modules/prepare/tasks/0011-length-aware-scan-projection-and-resource-handoff.md#result)<span data-glossary-navigation="true"> ([handoff glossary](../glossary.md#term-handoff))</span> | Model0037/0038, Compiler0013, Planning0008 and Prepare0010 Complete | Complete, MAIN accepted 2026-10-24; static [metadata](../glossary.md#term-metadata) implemented, focused 71/full [Prepare](../glossary.md#prepare) 92 tests, zero failures/errors/skips, Javadoc and MAIN [SPI](../glossary.md#term-service-provider-interface-spi)/four-[guard](../glossary.md#term-guard) [checkpoint](../glossary.md#term-checkpoint) PASS. Independent code/docs APPROVE, zero open P1/P2; at Prepare0011 completion, Prepare [Ready](../glossary.md#term-ready) frontier was None |
| [CPU0042 generated length execution](backends/cpu/tasks/0042-length-aware-general-scan-execution.md#result) | Model0037/0038, Compiler0013, Prepare0011/0014, CPU0033/0037/0041 Complete | Complete, MAIN accepted 2026-10-26 after independent code/docs APPROVE and recorded gates; fixed-geometry [F32](../glossary.md#floating-point-precision-and-rounding)/[F64](../glossary.md#floating-point-precision-and-rounding) forward/actual reverse, no steady-state parity claim; frontier None |
| Later length-aware cold/[persistence](../glossary.md#term-persistence) capability | Cold Checkpoint/v5 before persistence claims | Separately gated coarse [ownership](../glossary.md#term-ownership-meanings), no detailed cascade/launch |
| Generic owned NN [qualification](../glossary.md#term-qualification) and integration join | Qualified forward/actual first-order reverse + cold/v5 support; NN/Training/Checkpoint/[Engine](../glossary.md#term-engine) existing owners | Draft |
| Any fluent migration | Separate NN0032 authorization and compatibility resolution, in addition to any selected length capability | Draft/held |

This table is dependency ownership, not execution authorization. Main owns serial shared-document
integration; no parallel group. [NN0032](extensions/nn/tasks/0032-fluent-rnn-general-scan-migration.md)
retains its confirmed advanced second-order [CPU](../glossary.md#term-cpu) **[compile](../glossary.md#compile)** [regression](../glossary.md#term-regression), not runtime [Hessian](../glossary.md#term-hessian) proof.
No first-order migration waiver follows. [Singleton](../glossary.md#term-singleton) CONCAT still copies the dense output; CPU0037
read qualification does not remove that copy. Mixed CPU/[Metal](../glossary.md#term-metal) parity remains separately unproved;
no Metal, low-type, higher-order, if/while/regions or [vector](../glossary.md#term-vector)/parallel capability is added.

## [Qualification](../glossary.md#term-qualification) coverage and remaining evidence limits

Adapt existing focused tests, rather than creating another [validation](../glossary.md#term-validation) framework:

- [Model](../glossary.md#term-model) `ScanBodyTest`, `ScanExpressionTest`, [scan semantic tests](../../modules/model/src/test/java/io/github/pho001/synaptik/model/operation/scan/ScanSemanticsTest.java), `CompiledScanBodyTest`: binding
  roles, equal-size role ambiguity, row-coupling rejection, old constructor/default compatibility.<span data-glossary-navigation="true"> ([binding glossary](../glossary.md#term-binding))</span>
- [Compiler](../glossary.md#term-compiler) [ScanGradientRulesTest](../../modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/ScanGradientRulesTest.java), `GeneralScanCompilerTest`, `ScanBodyInferenceTest`,
  `SourceGraphReconstructionTest`: both traversals/seeds, false-flag carry, shared W-transpose
  history with F=B, axes pruning, tied roles, exact previous values and pre-allocation rejection
  of original shared-carry AD and original SHARED/ROW(1)/other nonzero-axis sequence AD, including
  unused/non-gradient positions. Preserve broader forward validity, generated reverse mixed-role
  inputs and old all-valid AD.<span data-glossary-navigation="true"> ([seeds glossary](../glossary.md#commit-release-distribution-log-and-seed-meanings); [carry glossary](../glossary.md#term-carry); [transpose glossary](../glossary.md#term-transpose); [axes glossary](../glossary.md#term-axis); [AD glossary](../glossary.md#autograd); [gradient glossary](../glossary.md#term-gradient))</span>
- [CPU](../glossary.md#term-cpu) `CpuGeneralScanCapabilityTest`, `CpuGeneralScanLoweringTest`, `CpuGeneralScanPreparationTest`,
  `CpuGeneralScanExecutionTest`, [CpuGeneralScanGeneratedKernelTest](../../modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneralScanGeneratedKernelTest.java) and clean-Java oracle:
  F32/F64 active SUM/MATMUL, NaN/Inf padding/seeds, zero/full/mixed lengths, T/B/features empty,
  invalid last length before writes, hoisting/history guards, reuse/concurrency/overlap and identities.<span data-glossary-navigation="true"> ([identities glossary](../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([oracle glossary](../glossary.md#term-test-oracle); [F32 glossary](../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../glossary.md#floating-point-precision-and-rounding); [MATMUL glossary](../glossary.md#matrix-multiplication--matmul); [NaN glossary](../glossary.md#term-nan); [padding glossary](../glossary.md#padding); [features glossary](../glossary.md#term-feature); [concurrency glossary](../glossary.md#term-concurrency))</span>
- [Checkpoint](../glossary.md#term-checkpoint) `GraphArtifactCodecTest`, [GraphCheckpointGeneralScanTest](../../modules/checkpoint/src/test/java/io/github/pho001/synaptik/checkpoint/graph/GraphCheckpointGeneralScanTest.java): selective old byte
  fixtures, v5 roles, old-reader rejection, bounds/late-invalid/import/ID probes, no lengths payload.<span data-glossary-navigation="true"> ([payload glossary](../glossary.md#artifact-envelope-and-atomic-publication))</span>
- `CpuGeneralScanConformanceTest`, [EngineCpuGeneralScanIntegrationTest](../../testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineCpuGeneralScanIntegrationTest.java),
  `GeneralScanOwnedModelIntegrationTest`, `RuntimeLengthScanModelIntegrationTest`: ordinary
  unrolled active-prefix oracles, current all-valid/fixed regressions and later owned load/train
  lifecycle. Preserve NN0032's separate higher-order compile and CONCAT/mixed-engine gates.<span data-glossary-navigation="true"> ([unrolled glossary](../glossary.md#backpropagation-through-time-and-recurrent-state); [lifecycle glossary](../glossary.md#lifecycle); [engine glossary](../glossary.md#term-engine))</span>

Design/authority evidence used source/contract inspection and scoped documentation validation,
not Java/native/generated-code or performance runs. Later Model0038 source tests and five [module](../glossary.md#term-module)
Javadocs now pass; its [result](modules/model/tasks/0038-length-aware-scan-source-foundation.md#result)
records reused evidence and independent review. Numerical examples are conceptual walkthroughs,
not measured fixtures. CPU0042's Result records the later execution/structural checks and sole
[nonstationary](../glossary.md#term-nonstationary) probe; cold/v5 and owned [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) remain future qualification. MAIN owns shared integration,
no parallel group. Authority acceptance is
Complete; MAIN accepted Model0038 Complete on 2026-10-24 after the combined named [compile](../glossary.md#compile)/
architecture/existing-native root test/[Javadoc](../glossary.md#term-javadoc) retry passed. Attempt one failed solely test-ledger
[serialization](../glossary.md#term-serialization); the bounded test-only repair preserved fixtures/production. Both performance flags
stayed false, successful/cached work was reused, and no native rebuild/campaign occurred.
Compiler0013 now delivers bounded [inference](../glossary.md#term-inference-meanings)/first-order reverse construction/query/demand,
with independent code/docs APPROVE and worker module, MAIN compatibility and final docs/Javadoc
PASS. MAIN accepted Complete 2026-10-24; no execution/[persistence](../glossary.md#term-persistence) grant follows.
Prepare0011 is Complete, MAIN accepted 2026-10-24 13:00:15 UTC after Tesla [planning](../glossary.md#term-planning) APPROVE, zero open P1/P2, and launch
[admission](../glossary.md#term-admission) 2026-10-24 12:25 UTC. ImplementationAuthorized=true applies only to Prepare0011 from
the user's latest “pust se do prace” request; clean sole writer started 12:31 UTC.
Implementation is frozen with focused 71/full [Prepare](../glossary.md#prepare) 92 tests and Javadoc PASS, zero failures/errors/skips.
MAIN's narrow [SPI](../glossary.md#term-service-provider-interface-spi)-consumer compile/four-[guard](../glossary.md#term-guard) checkpoint PASS. Independent code/docs APPROVE,
zero open P1/P2; MAIN overall acceptance complete, with no [runtime](../glossary.md#term-runtime) skipping claim or successor launch.

# Row-map and multiple-terminal [CPU](../../../glossary.md#term-cpu) [fusion](../../../glossary.md#term-fusion) proposal

## Reader context

This is the preserved CPU0073 scalar design. “Row map” names shared per-row calculation;
“terminal” names an output that cannot finish until all contributing rows have been visited.
The ordinary generator or `CpuGeneralScanEmitter` calls `CpuRowMapReductionEmitter.emitBound`
during code generation, passing local-variable handles and actions that emit bytecode.
Those handles are not Runtime buffer slots. CPU analysis must already have declared retained
stores and scratch before shared Prepare assigns resources; finalization does not invent storage.

For the dense-gradient example below, use zero-based row index $r\in\{0,1\}$ and input-feature
index $f\in\{0,1\}$. Here $X\in\mathbb{R}^{2\times2}$, $D\in\mathbb{R}^{2\times1}$ and
$dW\in\mathbb{R}^{2\times1}$. Its real-number interpretation is
$(dW)_{f,0}=\sum_{r=0}^{1}X_{r,f}D_{r,0}$: the first cell is
$1\cdot14+3\cdot28=98$ and the second is $2\cdot14+4\cdot28=140$.
These finite integer-valued example products and sums are exactly representable in F32.
The equation explains terminal dimensions and completion, not permission to simplify actual
Compiler AD, reorder rounded operations, or replace the separate ordinary exact bias SUM.

The [implemented row-map explanation](../../../backend-guide/cpu-backend.md#implemented-row-map-and-multiple-terminal-fusion)
connects retained stores to terminal completion; the
[current general Scan composition](../../../backend-guide/cpu-backend.md#current-general-scan-composition)
distinguishes this historical region's exclusions from whole-feature admission. The small
off-tree oracle observation remains provenance only; its temporary evidence is unavailable
to this reader revision and no new raw-cell comparison is claimed.

## Purpose and status

**Implemented design, not architecture authority.** The user approved this
linked reviewed proposal on 2026-10-30, including exact types/[signatures](../../../glossary.md#term-signature)/[cache](../../../glossary.md#term-cache) [identities](../../../glossary.md#term-identity-meanings) and
contract packet, and approved starting implementation ("ok, pust se do toho"). Approved pre-edit
[SHA-256](../../../glossary.md#term-sha-256): `a1f510c71aa14eef07b56fc48bde736c8dbf1079467c5142f8c69a457a37132b`.
The [owning mechanism contract](../../../architecture/contracts/backend-execution.md#cpu0073-accepted-row-map-and-multiple-terminal-reduction-fusion)
now carries the narrow packet, MAIN-accepted 2026-10-30 after Boole's independent APPROVE
(0 P1/P2). CPU0073's clean implementation, Hooke's final code/tests/[Javadoc](../../../glossary.md#term-javadoc)/evidence-transfer
APPROVE (zero open findings), MAIN's named join and independent documentation closure are complete.
This plan cannot authorize anything beyond that packet. The implemented general [CPU](../../../glossary.md#term-cpu)
capability shares a row-local calculation across row outputs and multiple terminal reductions
or contractions. A row map is a pure [directed acyclic graph](../../../glossary.md#term-directed-acyclic-graph-dag) ([DAG](../../../glossary.md#intermediate-representation-dag-and-ssa)) evaluated independently for each
ordered row R; a terminal consumes those rows but cannot feed the row map in the same pass.
This is not an [RNN](../../../glossary.md#term-recurrent-neural-network-rnn) [operation](../../../glossary.md#operation), a new numerical algorithm, or another [pointwise](../../../glossary.md#term-pointwise)-[fusion](../../../glossary.md#term-fusion) project.

```text
existing semantic graph -> CPU cold fusion proof -> one fused traversal + existing operation math
ordinary: one lowered execution unit     Scan: region inside the existing emitted Scan unit
```

No [Tensor](../../../glossary.md#tensor)/[Compiler](../../../glossary.md#term-compiler) semantic [node](../../../glossary.md#node) is added. A dedicated fusion emitter owns loop/local and
init/contributor-step/finish orchestration, not clone [scalar](../../../glossary.md#term-scalar), [MATMUL](../../../glossary.md#matrix-multiplication--matmul) or SUM mathematics.
Inside [Scan](../../../glossary.md#term-cumulative-scan) there is no per-step [Runtime](../../../glossary.md#term-runtime) unit, prepared [callback](../../../glossary.md#term-callback), worker submission or Job.
[CPU0073](tasks/0073-row-map-multi-reduction-fusion.md#result) is Complete, MAIN accepted 2026-10-30;
its Result owns exact [validation](../../../glossary.md#term-validation) and limitations. Source frontier: None; no automatic successor.
Measurement frontier: **None**. CPU0070/0071/0072 remain Complete; all old timing grants stay closed.
No new measurement, [commit](../../../glossary.md#term-commit) or push is authorized.

## Motivation and evidence limits

The actual backward-versus-independent-Java comparison found a cross-[operation](../../../glossary.md#operation) [fusion](../../../glossary.md#term-fusion) gap, not
merely untidy loops. The examined generated reverse has seven stages: delta scratch, dH, dW, dU, [bias](../../../glossary.md#term-bias) SUM,
bias ADD and dX separately. Existing Java A/B/L/I/F transcriptions reproduce staging and are not
the [target](../../../glossary.md#term-target) optimal-Java [oracle](../../../glossary.md#term-test-oracle). A separately derived small T4/B7/F3/H2 oracle kept delta row-local,
used twelve typed per-time contributions and separate per-time [carry](../../../glossary.md#term-carry) ADDs. Its 90 paired calls
matched all 9900 public raw cells; that one geometry is neither general proof nor timing evidence.
Oracle source [SHA-256](../../../glossary.md#term-sha-256): `7ad4153b8f6073441f3e944475311ec07c08b0f7386b8c8d1371fb4234b55a6a`.
Temporary evidence is [provenance](../../../glossary.md#provenance) only, not an executor [dependency](../../../glossary.md#term-dependency) or acceptance authority.

Two optimizations must stay distinct: row fusion removes repeated traversals/intermediate traffic;
across-time carry promotion retains [parameter](../../../glossary.md#term-parameter)-[gradient](../../../glossary.md#term-gradient) carries in locals across t. The oracle did
both, plus [invariant](../../../glossary.md#term-invariant)-load hoisting. V1 implements row fusion only; cross-time carry promotion is not
mandatory, and no performance contribution may be attributed to it without separate evidence.

## Legal pattern and conceptual examples

Admit the same ordered row [axis](../../../glossary.md#term-axis) R throughout a pure row-local [DAG](../../../glossary.md#term-directed-acyclic-graph-dag), including bounded [feature](../../../glossary.md#term-software-feature)
contractions. Permit independent row outputs and terminal increasing-R sums/bilinear products
`X.T @ D`, with each terminal's original ordered operands and represented sites. Terminal results
finish after all row contributions; a downstream consumer executes only after completion.

Non-[RNN](../../../glossary.md#term-recurrent-neural-network-rnn) dense-[gradient](../../../glossary.md#term-gradient) example: static [F32](../../../glossary.md#floating-point-precision-and-rounding), R=2/F=2/J=1/H=1, [canonical](../../../glossary.md#term-canonical-form) disjoint storage,
X=`[[1,2],[3,4]]`, Z=`[[1],[2]]`, W=`[[1],[2]]`, U=`[[1]]`, [bias](../../../glossary.md#term-bias)=`[1]`.
[Compile](../../../glossary.md#compile) [loss](../../../glossary.md#loss) `sum(Y*Y)`, Y=`X@W + Z@U + bias`, with [scalar](../../../glossary.md#term-scalar) [seed](../../../glossary.md#term-seed) 1 through real [Compiler](../../../glossary.md#term-compiler) [AD](../../../glossary.md#autograd).
Conceptual equations below explain the case; the test must preserve the actual AD expression's
ordered sites, not replace it with a hand-simplified [derivative](../../../glossary.md#term-derivative):

```text
Y = [[7],[14]]; D = [[14],[28]]
for r in increasing R:
    compute represented D[r,:] once from row-local producer sites
    dX[r,:] = full-H projection D[r,:] @ W.T
    dZ[r,:] = full-H projection D[r,:] @ U.T
    contribute X[r,:].T * D[r,:] to dW; Z[r,:].T * D[r,:] to dU
    retain D[r,:] store for the separate ordinary exact db = SUM_R(D)
finish dW, dU; then execute unchanged exact db unit
```

Expected dX=`[[14,28],[28,56]]`, dZ=`[[14],[28]]`, dW=`[[98],[140]]`, dU=`[[70]]`,
db=`[42]`. This demonstrates two terminal contractions plus row projections and a retained SUM
boundary; it does not establish general [dtype](../../../glossary.md#data-type--datatype)/[layout](../../../glossary.md#layout) support. A one-input variant drops Z/U and
still demonstrates delta + dX + dW with db outside. [Producer](../../../glossary.md#term-producer) + ordinary SUM alone is queued
CPU0049 work, not the new capability or evidence of newly invented [pointwise](../../../glossary.md#term-pointwise) [fusion](../../../glossary.md#term-fusion).

[Scan](../../../glossary.md#term-cumulative-scan) example: at each active original t, compute D[b,:] once, produce dX/dH row projections,
and accumulate Xstep.T@D, previousH.T@D and represented bias SUM in increasing active b.
Only after reduction, perform each SHARED [carry](../../../glossary.md#term-carry) ADD once at active(t), at a separate typed site.
Inactive t leaves shared state raw unchanged; never fold the carry into each row contribution.

Reject `s=SUM_R(D); E[r]=D[r]/s` as a single-pass group: s feeds row work. Reject a path leaving
the candidate through an outside [node](../../../glossary.md#node) and reentering it ([dependency](../../../glossary.md#term-dependency)-convexity failure), even if
source ordinals are adjacent. Equal F and B do not prove [transpose](../../../glossary.md#term-transpose)/row-axis compatibility.

## V1 [admission](../../../glossary.md#term-admission) and exclusions

This is the user-approved bounded [scope](../../../glossary.md#term-scope), now qualified by CPU0073. Selection requires a genuinely
computed row [producer](../../../glossary.md#term-producer) reaching a terminal and another retained or terminal sink. Input/movement
sharing alone is ineligible; this does not implement CPU0049's queued producer/SUM capability.

| Concern | Included | Excluded / retained fallback |
|---|---|---|
| Callers | Ordinary static [F32](../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../glossary.md#floating-point-precision-and-rounding); existing ordinary symbolic F32; static/symbolic/length/combined [Scan](../../../glossary.md#term-cumulative-scan) where separately admitted | Ordinary symbolic F64 [MATMUL](../../../glossary.md#matrix-multiplication--matmul); all-valid symbolic F64 Scan; any new [operation](../../../glossary.md#operation) [occurrence](../../../glossary.md#term-occurrence) |
| Access | [Canonical](../../../glossary.md#term-canonical-form) boundaries, existing dense-read exception/[scalar](../../../glossary.md#term-scalar) reads, proved internal [transpose](../../../glossary.md#term-transpose)/expand/[feature](../../../glossary.md#term-software-feature)-only edits | New affine/offset/[carrier](../../../glossary.md#term-storage-carrier) capability; infer [axes](../../../glossary.md#term-axis) from sizes |
| Arithmetic | [Homogeneous](../../../glossary.md#term-homogeneous) F32 or F64, scalar single-range; existing [pointwise](../../../glossary.md#term-pointwise) and [contraction](../../../glossary.md#contraction-dimension) math | Low/mixed types, new [FMA](../../../glossary.md#term-fused-multiply-add-fma)/[reassociation](../../../glossary.md#term-reassociation), new numerical algorithm |
| [Shape](../../../glossary.md#shape) | Fixed bounded features, R static or existing named B, existing fixed T | B/T [unrolling](../../../glossary.md#term-unrolling), per-B executable [cache](../../../glossary.md#term-cache), [runtime](../../../glossary.md#term-runtime)-value profitability heuristics |
| Scheduling | One complete ordinary work item; existing Scan outer traversal | [SIMD](../../../glossary.md#simd), threads, [OpenBLAS](../../../glossary.md#term-openblas), new executor/manager/interpreter, Job reuse |
| Reductions | Existing Scan represented SUM; ordinary exact SUM remains separate | Replace ordinary exact SUM with `+=`; exact-state [fusion](../../../glossary.md#term-fusion) in v1 |

Approved cold limits: 32 row sites, 4 terminal outputs, 64 terminal scalar cells, feature extents
at most 16 each, 128 simultaneously live [JVM](../../../glossary.md#term-java-virtual-machine-jvm) local slots (double counts two), 32 boundary carriers,
64 candidate attempts per anchor, conservative generated method size at most 48000 bytes.
Reject over-limit candidates before declaring resources; do not cap whole [partition](../../../glossary.md#partition) size or silently
truncate work. Every attempted candidate is bounded; retain existing [decomposition](../../../glossary.md#term-decomposition) for the rest.
The small live-value budget is primary; 48000 bytes is only a Class-File safety [guard](../../../glossary.md#term-guard), not [JIT](../../../glossary.md#term-just-in-time-compilation-jit)
profitability evidence. These are approved structural limits, not measured thresholds. Finalization verifies
the bound but cannot change [topology](../../../glossary.md#term-topology)/resources; a failed size proof is a defect, not late fallback.

## Legality, resources and range proof

1. Use complete partition/body uses, demanded/public/saved/external uses and physical origin aliases,
   not candidate-local counts. Prove a dependency-convex group and acyclic quotient of the full DAG.
   Include both contraction inputs, views and temporal old/new reads. Saved or external uses retain
   exact producing stores or split the boundary; never replay activations or delete live histories.<span data-glossary-navigation="true"> ([partition glossary](../../../glossary.md#partition); [aliases glossary](../../../glossary.md#aliasing-and-arena); [dependency-convex glossary](../../../glossary.md#dependency-convex-fusion-region); [DAG glossary](../../../glossary.md#term-directed-acyclic-graph-dag); [contraction glossary](../../../glossary.md#contraction-dimension); [views glossary](../../../glossary.md#term-view-meanings); [activations glossary](../../../glossary.md#term-activation))</span>
2. Prove row/index-axis provenance through each view using named Dimensions and affine recipes,
   never sample B or F=B. Prove injective writes, legal read aliases, no overwritten future reads,
   and inclusive cut lifetimes/size dominance for every admitted B. Unknown aliases reject fusion.<span data-glossary-navigation="true"> ([axis glossary](../../../glossary.md#term-axis); [provenance glossary](../../../glossary.md#provenance); [Dimensions glossary](../../../glossary.md#dimension); [sample glossary](../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [injective glossary](../../../glossary.md#term-injective); [fusion glossary](../../../glossary.md#term-fusion))</span>
3. Preserve each output's contributor order, operand order, typed MUL/ADD/result rounding and raw
   singleton/no-axis behavior. No new FMA or algebraic derivative simplification. Activity must
   dominate both operand loads and dependent math, including invariant/shared hoists.<span data-glossary-navigation="true"> ([rounding glossary](../../../glossary.md#term-rounding); [singleton glossary](../../../glossary.md#term-singleton); [FMA glossary](../../../glossary.md#term-fused-multiply-add-fma); [derivative glossary](../../../glossary.md#term-derivative); [invariant glossary](../../../glossary.md#term-invariant))</span>
4. Handle B=0, B=1, T=0 and all-inactive separately: required shared zeros still publish, row state
   preserves raw identity (including -0), inactive histories/dX use +0, and no poisoned padding is
   read. Preserve length validation after producers before any Scan-local write, even at T=0.<span data-glossary-navigation="true"> ([producers glossary](../../../glossary.md#term-producer))</span><span data-glossary-navigation="true"> ([identity glossary](../../../glossary.md#term-identity-meanings); [padding glossary](../../../glossary.md#padding); [validation glossary](../../../glossary.md#term-validation); [Scan glossary](../../../glossary.md#term-cumulative-scan))</span>
5. Ordinary SUM retains exact accumulation: symbolic F32 five limbs/48 bytes and bounded symbolic
   F64 34 limbs/280 bytes; static F64 remains domain-sized. Keep D materialization needed by db,
   its exact workspace slot/creator and empty/singleton behavior. Scan SUM uses the existing
   represented algorithm owner; ordinary MATMUL admission does not qualify the Scan binder.<span data-glossary-navigation="true"> ([F32 glossary](../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../glossary.md#floating-point-precision-and-rounding); [materialization glossary](../../../glossary.md#materialization); [workspace slot glossary](../../../glossary.md#workspace-slot--workspaceslot); [owner glossary](../../../glossary.md#term-owner); [MATMUL glossary](../../../glossary.md#matrix-multiplication--matmul); [admission glossary](../../../glossary.md#term-admission))</span>
6. Ordinary fused work owns rows AND shared outputs. Use the existing one-work-item/full-body
   convention (<span data-glossary-navigation="true"> ([fused glossary](../../../glossary.md#specialization-and-fusion))</span>`CpuLossLowering.lower` reduced-[loss](../../../glossary.md#loss) `work=1` precedent): prepared work count 1,
   entry range [0,1), including B=0; [0,0) does no work and other
   ranges reject before writes. Never use rowCount as unit work count, arbitrary pointwise ranges,
   or parallel splitting. Current reduced-loss lowering confirms the work=1 precedent; the
   ordinary batch binder now implements the approved full-body/all-output adaptation;
   no Runtime protocol change was needed. Empty R with nonempty dW/dU zero outputs is tested.<span data-glossary-navigation="true"> ([pointwise glossary](../../../glossary.md#term-pointwise); [lowering glossary](../../../glossary.md#lowering); [batch glossary](../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [Runtime glossary](../../../glossary.md#term-runtime))</span>
7. Detect groups before cuts/declarations: ordinary before decomposition/materialization selection;
   Scan before <span data-glossary-navigation="true"> ([decomposition glossary](../../../glossary.md#term-decomposition))</span>`shapedResources`. Recompute liveness and exact scratch from the selected region,
   retaining banks/history/old-carry hazards and retained D stores. Finalization only binds assigned
   slots and emits; it neither discovers requirements nor reruns fusion. One RunState remains owner.<span data-glossary-navigation="true"> ([liveness glossary](../../../glossary.md#term-liveness); [carry glossary](../../../glossary.md#term-carry); [RunState glossary](../../../glossary.md#run-state--runstate))</span>

## Minimal proposed production map and [signatures](../../../glossary.md#term-signature)

All names/signatures below are **implemented internal design**, not supported [API](../../../glossary.md#term-application-programming-interface-api).
The owning contract, not this plan, authorizes their bounded use.
Prefix packages with `io.github.pho001.synaptik.backend.cpu.`;
files live under `modules/backend/cpu/src/main/java/` at the matching package path.

| Implemented type | Purpose and source-based necessity |
|---|---|
| `internal.ir.CpuRowMapReductionIr` | [Immutable](../../../glossary.md#term-immutable) derived region, implementing `CpuPortableKernelIr`; existing [Scan](../../../glossary.md#term-cumulative-scan) `Stage` has one output/input/right/reduction anchor, while ordinary `CpuKernelIr` only carries [pointwise](../../../glossary.md#term-pointwise) instructions. Neither expresses several terminal accumulators sharing one row map. |
| Nested `CpuRowMapReductionIr.Contraction(int output, int left, int right)` | Ordered value [bindings](../../../glossary.md#term-binding); `CpuMatmulLowering.validateRankTwo` checks only [rank](../../../glossary.md#term-rank)/[Dimension](../../../glossary.md#dimension) equations. Region proof separately establishes row [provenance](../../../glossary.md#provenance), access and [alias](../../../glossary.md#aliasing-and-arena) safety; never fabricate ordinary [F64](../../../glossary.md#floating-point-precision-and-rounding) [batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters) [MATMUL](../../../glossary.md#matrix-multiplication--matmul) [IR](../../../glossary.md#term-intermediate-representation-ir). |
| Nested `CpuRowMapReductionIr.Sum(int output, int input, List<Integer> axes, boolean sumToShape)` | Explicit represented Scan SUM binding; never relabel exact `CpuAggregateIr.Kind.SUM` as sequential. |
| `internal.lowering.CpuRowMapReductionLowering` | One cold legality/bounds proof shared by ordinary and Scan adapters; the pre-[fusion](../../../glossary.md#term-fusion) recognizer's single-anchor/`ORDINARY_SPLIT` [model](../../../glossary.md#term-model) could not express this group. |
| `internal.codegen.emit.CpuRowMapReductionEmitter` | One generation-only row/terminal loop and local-[lifetime](../../../glossary.md#term-lifetime) [owner](../../../glossary.md#term-owner); existing MATMUL/SUM complete traversals remain fallback paths. |

Internal Java declaration excerpts, not supported API or runnable snippets:

```java
// CpuRowMapReductionIr constructor: immutable snapshots; ordinal references validated cold.
public CpuRowMapReductionIr(CpuKernelIr arithmetic, List<Integer> rowSites,
    List<Contraction> contractions, List<Sum> sums, List<Integer> terminalOutputs,
    List<CpuAccessPlan.BatchBinding> accesses, List<Integer> viewOrigins,
    List<Integer> rowAxes, List<Integer> retainedStores);
public CpuKernelIr encodedKernelIr(); // ordinary generated envelope, not a semantic graph
String structuralKey(); // geometry-free topology, ordered sites, algorithm/site distinctions
String bakedSpecializationIdentity(); // every baked access, feature, demand and limit fact
// CpuRowMapReductionLowering; caller supplies complete ordinal-normalized uses/alias order edges.
public static Optional<CpuRowMapReductionIr> prove(CpuRowMapReductionIr candidate,
    List<List<Integer>> allUses, List<List<Integer>> requiredOrder);
// CpuRowMapReductionEmitter: callbacks emit bytecode during generation only.
static void emitBound(CodeBuilder code, CpuRowMapReductionIr ir,
    int rowIndex, int rowLimit, int[][] valueLocals, int[][] terminalAccumulatorLocals,
    Runnable init, Consumer<Runnable> activeRow, IntConsumer emitRowSite,
    IntConsumer emitTerminalStep, Runnable finish);
```

`rowSites` interleaves pointwise instruction outputs and row-[feature](../../../glossary.md#term-software-feature) [contraction](../../../glossary.md#contraction-dimension) outputs in one
[topological order](../../../glossary.md#term-topological-order); each ordinal resolves to exactly one kind, completing its bounded feature cells
before dependent sites. Terminals are excluded and ordered by `terminalOutputs`; every operand is
dominated. Contractions use typed loaders, never whole ordinary route IR (symbolic MATMUL [F32](../../../glossary.md#floating-point-precision-and-rounding)-only).
`viewOrigins[value]` maps every value to its self-rooted storage-producing ordinal; accesses encode
each [view](../../../glossary.md#term-view-meanings)'s geometry relative to that origin (static recipes have zero batch factors). BatchBinding
does not encode alias [identity](../../../glossary.md#term-identity-meanings). Validate total coverage, only self-root cycles, legal view composition,
rowAxes provenance and physical read/write hazards, including actual backing aliases at binding.
The proof lists supply complete uses and must-precede pairs, including conservative alias edges.

The caller allocates shared locals with the same CodeBuilder: INT rowIndex/rowLimit, valueLocals indexed
by value ordinal then fixed feature cell, and terminalAccumulatorLocals by terminal position/cell.
Use -1 only for direct-storage values; validate typed widths/nonoverlapping slots and load dominance.
Callbacks close over these exact immutable handle tables; emitRowSite/emitTerminalStep receive the
site ordinal, never allocate replacement [accumulator](../../../glossary.md#accumulator-type)/row locals or mutate tables. Emitter owns
lifetime/control, not allocation: init once, rowIndex=0, increasing loop to prebound rowLimit,
activeRow wrapping all rowSites then terminal steps, finish once after the loop (also at R=0).
Callbacks emit math/stores using those handles; rowIndex is read-only to them. Finish publishes
completed contributions and distinct [carry](../../../glossary.md#term-carry) ADDs; Scan caller's active-time gate encloses emission.
All arrays/callbacks exist only during generation; no [runtime](../../../glossary.md#term-runtime) protocol or per-step objects result.

Existing-file impact: extend `CpuPortableKernelIr` permits and existing `CpuPortableRoutePlan`,
`CpuPartitionLowering.LoweredPartition`, `CpuPartitionPreparationPlan`/`CpuPartitionPreparer` to
carry the optional ordinary region and its full-body work contract; extend existing
`CpuPreparedExecutable.bindOrdinaryBatch` for work=1 and all-output alias checks, not its former
last-output/count assumptions. Extend `CpuPartitionDagDecomposer`
and `CpuSpecializedSubgraphRecognizer` selection without changing unrelated recognition budgets.
Scan adapters in `CpuGeneralScanLowering`/`CpuGeneralScanIr.Stages` carry the optional same region
before cuts; `CpuGeneralScanEmitter` emits it. `CpuGeneralScanStageSchedule.order` remains a
post-slot reorder proof, not the fusion mechanism, and must see region read/write/lifetime facts.
`CpuClassFileKernelGenerator` selects the new emitter. The explicit row-region case in
`CpuPortableRoutePlan.encoded(CpuPortableKernelIr)` returns `region.encodedKernelIr()` before its
final aggregate cast; the envelope exposes boundary stores/family identity and derives the
same structuralKey, while typed region facts select emission. Test route construction through
`kernelIr()`, [artifact](../../../glossary.md#term-artifact) generation/load/reuse and rejection, not merely suffix [serialization](../../../glossary.md#term-serialization).

Expose generation-only `emitInit(CodeBuilder,DataType,int)`,
`emitContributorStep(CodeBuilder,DataType,int,Runnable,Runnable)` and
`emitFinish(CodeBuilder,DataType,int,IntConsumer)` in `CpuMatmulEmitter`; loaders leave ordered
typed values on stack, accumulator stays typed. `CpuAggregateEmitter` implements
`emitRepresentedInit(CodeBuilder,DataType,int,boolean directSingleton)`,
`emitRepresentedContributorStep(CodeBuilder,DataType,int,boolean rawSingleton,Runnable load)` and
`emitRepresentedFinish(CodeBuilder,DataType,int,IntConsumer sink)`. All seams are static void,
package-private and generation-only; preserve raw [singleton](../../../glossary.md#term-singleton) behavior. Reuse these from
old complete loops and the new orchestration, preserving old byte/local order. Reuse
`CpuClassFileKernelGenerator.emitBoundPointwise`, `CpuScalarEmitter` and [carrier](../../../glossary.md#term-storage-carrier) access; no cloned
[opcode](../../../glossary.md#term-opcode) switch/math. `CpuExactSumEmitter` is unchanged. Any extra named production type,
common-seam change beyond approved signatures or cross-owner protocol/runtime edge needs approval.
Ordinary private helpers within existing owners and this capability do not need fresh approval.

## Compatibility and retirement recommendation

Keep global generator envelope `CURRENT_VERSION=71`, [Scan](../../../glossary.md#term-cumulative-scan) projection 71 and existing namespaces.
Use approved ordinary class-[identity](../../../glossary.md#term-identity-meanings) projection **72** only for selected
ordinary regions; `CpuKernelSpecialization` keeps projection 71 iff `generalScanIr` exists.
Append `Optional<CpuRowMapReductionIr> rowMapReductionIr` to `CpuKernelSpecialization`; projection
72 iff it is present, mutually exclusive with matmulIr/generalScanIr. Scan embeds regions only
inside generalScanIr and retains projection 71. Validate [fingerprint](../../../glossary.md#term-fingerprint), boundary [dtype](../../../glossary.md#data-type--datatype)/[carrier](../../../glossary.md#term-storage-carrier)/access,
[scalar](../../../glossary.md#term-scalar)/species-zero, materializedSourcePosition=-1 and scratch requirements against the region;
ordinary symbolic groups are [F32](../../../glossary.md#floating-point-precision-and-rounding) only, not a bypass of typed-[batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters) [F64](../../../glossary.md#floating-point-precision-and-rounding)-[MATMUL](../../../glossary.md#matrix-multiplication--matmul) rejection. Preserve
every existing constructor overload (including the old [canonical](../../../glossary.md#term-canonical-form) [signature](../../../glossary.md#term-signature)) delegating with
Optional.empty(); update selected-region creation sites and add positive/negative constructor,
route/factory and both-stream identity tests. Keep 54 iff matmulIr and 71 iff generalScanIr.
Add geometry-free ordinary family marker
`row-map-multi-reduction-v1`; append `|ordinary-row-map-multi-reduction-v1=` only for selected
ordinary groups. Selected Scan groups append `|scan-row-map-multi-reduction-v1=` to their existing
baked mode [payload](../../../glossary.md#artifact-envelope-and-atomic-publication). Both `compatibilityBytes()` and `classIdentityBytes()` consume the same payload;
old unfused [artifacts](../../../glossary.md#term-artifact) must reject when supplied for a selected [fused](../../../glossary.md#specialization-and-fusion) [specialization](../../../glossary.md#term-specialization).
Encode [topology](../../../glossary.md#term-topology), ordered [bindings](../../../glossary.md#term-binding), dtype/carrier/access, algorithm/[singleton](../../../glossary.md#term-singleton)/[rounding](../../../glossary.md#term-rounding) sites,
row/activity roles, retained stores/demand, baked [feature](../../../glossary.md#term-feature)/coefficient/resource/[lifetime](../../../glossary.md#term-lifetime) facts,
range convention and all code-shaping limits, including fixed R/T/[strides](../../../glossary.md#term-stride)/resource bytes if baked
into ClassFiles. Exclude invocation-resolved symbolic B, length values, addresses, assigned slots
and invocation-resolved resource bytes only. No diagnostic text as identity or blanket [schema](../../../glossary.md#term-schema) bump.
Noneligible ordinary/static/symbolic/length/combined, [vector](../../../glossary.md#term-vector)/tiled/[TANH](../../../glossary.md#term-tanh)/fixed controls retain literal
old keys/bytes/resources. Freeze before-source controls; do not derive expectations from candidate code.

Only after both callers work may superseded eligible whole-domain traversal glue/cuts be removed.
Keep old fallback emitters, stage scheduling/temporal proofs, exact ordinary SUM, saved stores and
old test oracles. CPU0071/0072 already retired Scan math subemitters; do not recreate them.

## Delivery, [validation](../../../glossary.md#term-validation) and approval decisions

One Class C task: (1) approval/authority gate; (2) durable equations/actual-[AD](../../../glossary.md#autograd) fixtures/negatives;
(3) common [IR](../../../glossary.md#term-intermediate-representation-ir)/proof plus ordinary full-body unit; (4) [Scan](../../../glossary.md#term-cumulative-scan) adapter/resources;
(5) shared math seams/emission/[identity](../../../glossary.md#term-identity-meanings) and generated-class inspection; (6) stabilized [module](../../../glossary.md#term-module) gate,
named [conformance](../../../glossary.md#term-conformance)/integration and independent code/docs review. These are milestones, not a task cascade.

Implemented test-only `internal.codegen.emit.CpuRowMapReductionJavaOracle` and
`CpuRowMapReductionGeneratedKernelTest`, plus `internal.lowering.CpuRowMapReductionLoweringTest`.
Use self-contained typed equations/cases, explicit saved primals, both traversals/[seeds](../../../glossary.md#commit-release-distribution-log-and-seed-meanings), actual
[Compiler](../../../glossary.md#term-compiler) reverse and non-[RNN](../../../glossary.md#term-recurrent-neural-network-rnn) dense [gradients](../../../glossary.md#term-gradient); no temporary-[classpath](../../../glossary.md#term-classpath) [dependency](../../../glossary.md#term-dependency). Add multi-[shape](../../../glossary.md#shape),
F=B/ROW(1), poisoned [padding](../../../glossary.md#padding), empty/[singleton](../../../glossary.md#term-singleton), [cancellation](../../../glossary.md#term-cancellation)/special-value and [alias](../../../glossary.md#aliasing-and-arena)/reentry negatives.
Inspect generated classes for one row traversal, retained stores, separate [carry](../../../glossary.md#term-carry) ADDs and no hidden
allocation/boxing/reflection/[dispatch](../../../glossary.md#term-dispatch)/helper arithmetic. Match optimal specialized Java, not staged
transcriptions. [CPU0073 validation](tasks/0073-row-map-multi-reduction-fusion.md#validation) names gates.

Performance is unmeasured. Proposed later protocol requires fresh explicit approval of exact durable
fixtures, seeds and entry [ABI](../../../glossary.md#term-application-binary-interface-abi): at most six rows (ordinary static [F32](../../../glossary.md#floating-point-precision-and-rounding), ordinary symbolic F32, combined
F32/[F64](../../../glossary.md#floating-point-precision-and-rounding) reverse, two noneligible controls), one [JVM](../../../glossary.md#term-java-virtual-machine-jvm), 60-second watchdog, no retry/additional [fork](../../../glossary.md#term-fork-meanings).
Compare retained split, [fused](../../../glossary.md#specialization-and-fusion) and optimal same-algorithm Java; cap [warmup](../../../glossary.md#term-warmup) at five 100ms windows and
five alternating paired 100ms [samples](../../../glossary.md#term-sample-machine-learning-and-hyperparameters) per row. Separately report cold generation/classload/[JIT](../../../glossary.md#term-just-in-time-compilation-jit),
[admission](../../../glossary.md#term-admission), matched entry/[kernel](../../../glossary.md#kernel) time and entry allocation. Instability is inconclusive, not a reason
to extend. No five-fork campaign, speed/parity claim or implicit measurement grant follows.

The 2026-10-30 user approval covers the five exact names, [signatures](../../../glossary.md#term-signature)/common seams,
admission/limits/range, identity strings and coordinated contract packet. Root identity/index
[alignment](../../../glossary.md#term-alignment) and the owning [backend](../../../glossary.md#backend) mechanism subsection are accepted; only scoped gate/retention
references change in Scan. `Actual first-order closure and arithmetic`, `Activity, raw identity
and retained histories` and `First-order length-mode reverse` remain unchanged. [Runtime](../../../glossary.md#term-runtime)
`Prepare lifecycle`/`Run lifecycle` and foundational `Floating arithmetic primitives`/[`Aggregate
floor`](../../../glossary.md#term-aggregate-floor) remain constraints, not changes. The Scan explanation, [CPU](../../../glossary.md#term-cpu) guide and
[ADR 0052](../../../design/decisions/0052-row-map-multi-reduction-fusion.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../glossary.md#term-architecture-decision-record-adr))</span> distinguish
bounded implementation from unmeasured performance. Boole's authority APPROVE preceded the verified
[Ready](../../../glossary.md#term-ready) receipt; CPU0073's clean implementation, independent final reviews, MAIN named join and
documentation closure are Complete on 2026-10-30. Its Result records the failed full-module [run](../../../glossary.md#term-run),
passing repairs and approved unchanged-outcome transfer, not a fresh full-suite pass.
MAIN owns shared integration; source/measurement frontiers are None and no successor is automatic.
Targeted glossary review requires no edit: row map/terminal are locally defined
CPU optimization labels and change no existing public semantic meaning. No measurement is granted.

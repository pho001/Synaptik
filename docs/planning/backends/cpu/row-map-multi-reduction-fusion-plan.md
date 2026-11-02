# Row-map and multiple-terminal CPU fusion proposal

## Purpose and status

**Implemented design, not architecture authority.** The user approved this
linked reviewed proposal on 2026-10-30, including exact types/signatures/cache identities and
contract packet, and approved starting implementation ("ok, pust se do toho"). Approved pre-edit
SHA-256: `a1f510c71aa14eef07b56fc48bde736c8dbf1079467c5142f8c69a457a37132b`.
The [owning mechanism contract](../../../architecture/contracts/backend-execution.md#cpu0073-accepted-row-map-and-multiple-terminal-reduction-fusion)
now carries the narrow packet, MAIN-accepted 2026-10-30 after Boole's independent APPROVE
(0 P1/P2). CPU0073's clean implementation, Hooke's final code/tests/Javadoc/evidence-transfer
APPROVE (zero open findings), MAIN's named join and independent documentation closure are complete.
This plan cannot authorize anything beyond that packet. The implemented general CPU
capability shares a row-local calculation across row outputs and multiple terminal reductions
or contractions. A row map is a pure directed acyclic graph (DAG) evaluated independently for each
ordered row R; a terminal consumes those rows but cannot feed the row map in the same pass.
This is not an RNN operation, a new numerical algorithm, or another pointwise-fusion project.

```text
existing semantic graph -> CPU cold fusion proof -> one fused traversal + existing operation math
ordinary: one lowered execution unit     Scan: region inside the existing emitted Scan unit
```

No Tensor/Compiler semantic node is added. A dedicated fusion emitter owns loop/local and
init/contributor-step/finish orchestration, not clone scalar, MATMUL or SUM mathematics.
Inside Scan there is no per-step Runtime unit, prepared callback, worker submission or Job.
[CPU0073](tasks/0073-row-map-multi-reduction-fusion.md#result) is Complete, MAIN accepted 2026-10-30;
its Result owns exact validation and limitations. Source frontier: None; no automatic successor.
Measurement frontier: **None**. CPU0070/0071/0072 remain Complete; all old timing grants stay closed.
No new measurement, commit or push is authorized.

## Motivation and evidence limits

The actual backward-versus-independent-Java comparison found a cross-operation fusion gap, not
merely untidy loops. The examined generated reverse has seven stages: delta scratch, dH, dW, dU, bias SUM,
bias ADD and dX separately. Existing Java A/B/L/I/F transcriptions reproduce staging and are not
the target optimal-Java oracle. A separately derived small T4/B7/F3/H2 oracle kept delta row-local,
used twelve typed per-time contributions and separate per-time carry ADDs. Its 90 paired calls
matched all 9900 public raw cells; that one geometry is neither general proof nor timing evidence.
Oracle source SHA-256: `7ad4153b8f6073441f3e944475311ec07c08b0f7386b8c8d1371fb4234b55a6a`.
Temporary evidence is provenance only, not an executor dependency or acceptance authority.

Two optimizations must stay distinct: row fusion removes repeated traversals/intermediate traffic;
across-time carry promotion retains parameter-gradient carries in locals across t. The oracle did
both, plus invariant-load hoisting. V1 implements row fusion only; cross-time carry promotion is not
mandatory, and no performance contribution may be attributed to it without separate evidence.

## Legal pattern and conceptual examples

Admit the same ordered row axis R throughout a pure row-local DAG, including bounded feature
contractions. Permit independent row outputs and terminal increasing-R sums/bilinear products
`X.T @ D`, with each terminal's original ordered operands and represented sites. Terminal results
finish after all row contributions; a downstream consumer executes only after completion.

Non-RNN dense-gradient example: static F32, R=2/F=2/J=1/H=1, canonical disjoint storage,
X=`[[1,2],[3,4]]`, Z=`[[1],[2]]`, W=`[[1],[2]]`, U=`[[1]]`, bias=`[1]`.
Compile loss `sum(Y*Y)`, Y=`X@W + Z@U + bias`, with scalar seed 1 through real Compiler AD.
Conceptual equations below explain the case; the test must preserve the actual AD expression's
ordered sites, not replace it with a hand-simplified derivative:

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
boundary; it does not establish general dtype/layout support. A one-input variant drops Z/U and
still demonstrates delta + dX + dW with db outside. Producer + ordinary SUM alone is queued
CPU0049 work, not the new capability or evidence of newly invented pointwise fusion.

Scan example: at each active original t, compute D[b,:] once, produce dX/dH row projections,
and accumulate Xstep.T@D, previousH.T@D and represented bias SUM in increasing active b.
Only after reduction, perform each SHARED carry ADD once at active(t), at a separate typed site.
Inactive t leaves shared state raw unchanged; never fold the carry into each row contribution.

Reject `s=SUM_R(D); E[r]=D[r]/s` as a single-pass group: s feeds row work. Reject a path leaving
the candidate through an outside node and reentering it (dependency-convexity failure), even if
source ordinals are adjacent. Equal F and B do not prove transpose/row-axis compatibility.

## V1 admission and exclusions

This is the user-approved bounded scope, now qualified by CPU0073. Selection requires a genuinely
computed row producer reaching a terminal and another retained or terminal sink. Input/movement
sharing alone is ineligible; this does not implement CPU0049's queued producer/SUM capability.

| Concern | Included | Excluded / retained fallback |
|---|---|---|
| Callers | Ordinary static F32/F64; existing ordinary symbolic F32; static/symbolic/length/combined Scan where separately admitted | Ordinary symbolic F64 MATMUL; all-valid symbolic F64 Scan; any new operation occurrence |
| Access | Canonical boundaries, existing dense-read exception/scalar reads, proved internal transpose/expand/feature-only edits | New affine/offset/carrier capability; infer axes from sizes |
| Arithmetic | Homogeneous F32 or F64, scalar single-range; existing pointwise and contraction math | Low/mixed types, new FMA/reassociation, new numerical algorithm |
| Shape | Fixed bounded features, R static or existing named B, existing fixed T | B/T unrolling, per-B executable cache, runtime-value profitability heuristics |
| Scheduling | One complete ordinary work item; existing Scan outer traversal | SIMD, threads, OpenBLAS, new executor/manager/interpreter, Job reuse |
| Reductions | Existing Scan represented SUM; ordinary exact SUM remains separate | Replace ordinary exact SUM with `+=`; exact-state fusion in v1 |

Approved cold limits: 32 row sites, 4 terminal outputs, 64 terminal scalar cells, feature extents
at most 16 each, 128 simultaneously live JVM local slots (double counts two), 32 boundary carriers,
64 candidate attempts per anchor, conservative generated method size at most 48000 bytes.
Reject over-limit candidates before declaring resources; do not cap whole partition size or silently
truncate work. Every attempted candidate is bounded; retain existing decomposition for the rest.
The small live-value budget is primary; 48000 bytes is only a Class-File safety guard, not JIT
profitability evidence. These are approved structural limits, not measured thresholds. Finalization verifies
the bound but cannot change topology/resources; a failed size proof is a defect, not late fallback.

## Legality, resources and range proof

1. Use complete partition/body uses, demanded/public/saved/external uses and physical origin aliases,
   not candidate-local counts. Prove a dependency-convex group and acyclic quotient of the full DAG.
   Include both contraction inputs, views and temporal old/new reads. Saved or external uses retain
   exact producing stores or split the boundary; never replay activations or delete live histories.
2. Prove row/index-axis provenance through each view using named Dimensions and affine recipes,
   never sample B or F=B. Prove injective writes, legal read aliases, no overwritten future reads,
   and inclusive cut lifetimes/size dominance for every admitted B. Unknown aliases reject fusion.
3. Preserve each output's contributor order, operand order, typed MUL/ADD/result rounding and raw
   singleton/no-axis behavior. No new FMA or algebraic derivative simplification. Activity must
   dominate both operand loads and dependent math, including invariant/shared hoists.
4. Handle B=0, B=1, T=0 and all-inactive separately: required shared zeros still publish, row state
   preserves raw identity (including -0), inactive histories/dX use +0, and no poisoned padding is
   read. Preserve length validation after producers before any Scan-local write, even at T=0.
5. Ordinary SUM retains exact accumulation: symbolic F32 five limbs/48 bytes and bounded symbolic
   F64 34 limbs/280 bytes; static F64 remains domain-sized. Keep D materialization needed by db,
   its exact workspace slot/creator and empty/singleton behavior. Scan SUM uses the existing
   represented algorithm owner; ordinary MATMUL admission does not qualify the Scan binder.
6. Ordinary fused work owns rows AND shared outputs. Use the existing one-work-item/full-body
   convention (`CpuLossLowering.lower` reduced-loss `work=1` precedent): prepared work count 1,
   entry range [0,1), including B=0; [0,0) does no work and other
   ranges reject before writes. Never use rowCount as unit work count, arbitrary pointwise ranges,
   or parallel splitting. Current reduced-loss lowering confirms the work=1 precedent; the
   ordinary batch binder now implements the approved full-body/all-output adaptation;
   no Runtime protocol change was needed. Empty R with nonempty dW/dU zero outputs is tested.
7. Detect groups before cuts/declarations: ordinary before decomposition/materialization selection;
   Scan before `shapedResources`. Recompute liveness and exact scratch from the selected region,
   retaining banks/history/old-carry hazards and retained D stores. Finalization only binds assigned
   slots and emits; it neither discovers requirements nor reruns fusion. One RunState remains owner.

## Minimal proposed production map and signatures

All names/signatures below are **implemented internal design**, not supported API.
The owning contract, not this plan, authorizes their bounded use.
Prefix packages with `io.github.pho001.synaptik.backend.cpu.`;
files live under `modules/backend/cpu/src/main/java/` at the matching package path.

| Implemented type | Purpose and source-based necessity |
|---|---|
| `internal.ir.CpuRowMapReductionIr` | Immutable derived region, implementing `CpuPortableKernelIr`; existing Scan `Stage` has one output/input/right/reduction anchor, while ordinary `CpuKernelIr` only carries pointwise instructions. Neither expresses several terminal accumulators sharing one row map. |
| Nested `CpuRowMapReductionIr.Contraction(int output, int left, int right)` | Ordered value bindings; `CpuMatmulLowering.validateRankTwo` checks only rank/Dimension equations. Region proof separately establishes row provenance, access and alias safety; never fabricate ordinary F64 batch MATMUL IR. |
| Nested `CpuRowMapReductionIr.Sum(int output, int input, List<Integer> axes, boolean sumToShape)` | Explicit represented Scan SUM binding; never relabel exact `CpuAggregateIr.Kind.SUM` as sequential. |
| `internal.lowering.CpuRowMapReductionLowering` | One cold legality/bounds proof shared by ordinary and Scan adapters; the pre-fusion recognizer's single-anchor/`ORDINARY_SPLIT` model could not express this group. |
| `internal.codegen.emit.CpuRowMapReductionEmitter` | One generation-only row/terminal loop and local-lifetime owner; existing MATMUL/SUM complete traversals remain fallback paths. |

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

`rowSites` interleaves pointwise instruction outputs and row-feature contraction outputs in one
topological order; each ordinal resolves to exactly one kind, completing its bounded feature cells
before dependent sites. Terminals are excluded and ordered by `terminalOutputs`; every operand is
dominated. Contractions use typed loaders, never whole ordinary route IR (symbolic MATMUL F32-only).
`viewOrigins[value]` maps every value to its self-rooted storage-producing ordinal; accesses encode
each view's geometry relative to that origin (static recipes have zero batch factors). BatchBinding
does not encode alias identity. Validate total coverage, only self-root cycles, legal view composition,
rowAxes provenance and physical read/write hazards, including actual backing aliases at binding.
The proof lists supply complete uses and must-precede pairs, including conservative alias edges.

The caller allocates shared locals with the same CodeBuilder: INT rowIndex/rowLimit, valueLocals indexed
by value ordinal then fixed feature cell, and terminalAccumulatorLocals by terminal position/cell.
Use -1 only for direct-storage values; validate typed widths/nonoverlapping slots and load dominance.
Callbacks close over these exact immutable handle tables; emitRowSite/emitTerminalStep receive the
site ordinal, never allocate replacement accumulator/row locals or mutate tables. Emitter owns
lifetime/control, not allocation: init once, rowIndex=0, increasing loop to prebound rowLimit,
activeRow wrapping all rowSites then terminal steps, finish once after the loop (also at R=0).
Callbacks emit math/stores using those handles; rowIndex is read-only to them. Finish publishes
completed contributions and distinct carry ADDs; Scan caller's active-time gate encloses emission.
All arrays/callbacks exist only during generation; no runtime protocol or per-step objects result.

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
`kernelIr()`, artifact generation/load/reuse and rejection, not merely suffix serialization.

Expose generation-only `emitInit(CodeBuilder,DataType,int)`,
`emitContributorStep(CodeBuilder,DataType,int,Runnable,Runnable)` and
`emitFinish(CodeBuilder,DataType,int,IntConsumer)` in `CpuMatmulEmitter`; loaders leave ordered
typed values on stack, accumulator stays typed. `CpuAggregateEmitter` implements
`emitRepresentedInit(CodeBuilder,DataType,int,boolean directSingleton)`,
`emitRepresentedContributorStep(CodeBuilder,DataType,int,boolean rawSingleton,Runnable load)` and
`emitRepresentedFinish(CodeBuilder,DataType,int,IntConsumer sink)`. All seams are static void,
package-private and generation-only; preserve raw singleton behavior. Reuse these from
old complete loops and the new orchestration, preserving old byte/local order. Reuse
`CpuClassFileKernelGenerator.emitBoundPointwise`, `CpuScalarEmitter` and carrier access; no cloned
opcode switch/math. `CpuExactSumEmitter` is unchanged. Any extra named production type,
common-seam change beyond approved signatures or cross-owner protocol/runtime edge needs approval.
Ordinary private helpers within existing owners and this capability do not need fresh approval.

## Compatibility and retirement recommendation

Keep global generator envelope `CURRENT_VERSION=71`, Scan projection 71 and existing namespaces.
Use approved ordinary class-identity projection **72** only for selected
ordinary regions; `CpuKernelSpecialization` keeps projection 71 iff `generalScanIr` exists.
Append `Optional<CpuRowMapReductionIr> rowMapReductionIr` to `CpuKernelSpecialization`; projection
72 iff it is present, mutually exclusive with matmulIr/generalScanIr. Scan embeds regions only
inside generalScanIr and retains projection 71. Validate fingerprint, boundary dtype/carrier/access,
scalar/species-zero, materializedSourcePosition=-1 and scratch requirements against the region;
ordinary symbolic groups are F32 only, not a bypass of typed-batch F64-MATMUL rejection. Preserve
every existing constructor overload (including the old canonical signature) delegating with
Optional.empty(); update selected-region creation sites and add positive/negative constructor,
route/factory and both-stream identity tests. Keep 54 iff matmulIr and 71 iff generalScanIr.
Add geometry-free ordinary family marker
`row-map-multi-reduction-v1`; append `|ordinary-row-map-multi-reduction-v1=` only for selected
ordinary groups. Selected Scan groups append `|scan-row-map-multi-reduction-v1=` to their existing
baked mode payload. Both `compatibilityBytes()` and `classIdentityBytes()` consume the same payload;
old unfused artifacts must reject when supplied for a selected fused specialization.
Encode topology, ordered bindings, dtype/carrier/access, algorithm/singleton/rounding sites,
row/activity roles, retained stores/demand, baked feature/coefficient/resource/lifetime facts,
range convention and all code-shaping limits, including fixed R/T/strides/resource bytes if baked
into ClassFiles. Exclude invocation-resolved symbolic B, length values, addresses, assigned slots
and invocation-resolved resource bytes only. No diagnostic text as identity or blanket schema bump.
Noneligible ordinary/static/symbolic/length/combined, vector/tiled/TANH/fixed controls retain literal
old keys/bytes/resources. Freeze before-source controls; do not derive expectations from candidate code.

Only after both callers work may superseded eligible whole-domain traversal glue/cuts be removed.
Keep old fallback emitters, stage scheduling/temporal proofs, exact ordinary SUM, saved stores and
old test oracles. CPU0071/0072 already retired Scan math subemitters; do not recreate them.

## Delivery, validation and approval decisions

One Class C task: (1) approval/authority gate; (2) durable equations/actual-AD fixtures/negatives;
(3) common IR/proof plus ordinary full-body unit; (4) Scan adapter/resources;
(5) shared math seams/emission/identity and generated-class inspection; (6) stabilized module gate,
named conformance/integration and independent code/docs review. These are milestones, not a task cascade.

Implemented test-only `internal.codegen.emit.CpuRowMapReductionJavaOracle` and
`CpuRowMapReductionGeneratedKernelTest`, plus `internal.lowering.CpuRowMapReductionLoweringTest`.
Use self-contained typed equations/cases, explicit saved primals, both traversals/seeds, actual
Compiler reverse and non-RNN dense gradients; no temporary-classpath dependency. Add multi-shape,
F=B/ROW(1), poisoned padding, empty/singleton, cancellation/special-value and alias/reentry negatives.
Inspect generated classes for one row traversal, retained stores, separate carry ADDs and no hidden
allocation/boxing/reflection/dispatch/helper arithmetic. Match optimal specialized Java, not staged
transcriptions. [CPU0073 validation](tasks/0073-row-map-multi-reduction-fusion.md#validation) names gates.

Performance is unmeasured. Proposed later protocol requires fresh explicit approval of exact durable
fixtures, seeds and entry ABI: at most six rows (ordinary static F32, ordinary symbolic F32, combined
F32/F64 reverse, two noneligible controls), one JVM, 60-second watchdog, no retry/additional fork.
Compare retained split, fused and optimal same-algorithm Java; cap warmup at five 100ms windows and
five alternating paired 100ms samples per row. Separately report cold generation/classload/JIT,
admission, matched entry/kernel time and entry allocation. Instability is inconclusive, not a reason
to extend. No five-fork campaign, speed/parity claim or implicit measurement grant follows.

The 2026-10-30 user approval covers the five exact names, signatures/common seams,
admission/limits/range, identity strings and coordinated contract packet. Root identity/index
alignment and the owning backend mechanism subsection are accepted; only scoped gate/retention
references change in Scan. `Actual first-order closure and arithmetic`, `Activity, raw identity
and retained histories` and `First-order length-mode reverse` remain unchanged. Runtime
`Prepare lifecycle`/`Run lifecycle` and foundational `Floating arithmetic primitives`/`Aggregate
floor` remain constraints, not changes. The Scan explanation, CPU guide and
[ADR 0052](../../../design/decisions/0052-row-map-multi-reduction-fusion.md) distinguish
bounded implementation from unmeasured performance. Boole's authority APPROVE preceded the verified
Ready receipt; CPU0073's clean implementation, independent final reviews, MAIN named join and
documentation closure are Complete on 2026-10-30. Its Result records the failed full-module run,
passing repairs and approved unchanged-outcome transfer, not a fresh full-suite pass.
MAIN owns shared integration; source/measurement frontiers are None and no successor is automatic.
Targeted glossary review requires no edit: row map/terminal are locally defined
CPU optimization labels and change no existing public semantic meaning. No measurement is granted.

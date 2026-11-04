# Shared ordinary and [Scan body](../../../glossary.md#backpropagation-through-time-and-recurrent-state) [lowering](../../../glossary.md#lowering) and emission

> Archived by CPU0070 on 2026-10-29. The active replacement is
> [Compose existing CPU operation emission](existing-emitter-composition-plan.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../glossary.md#term-cpu); [operation glossary](../../../glossary.md#operation))</span>.
> The historical scalar/vector/worker sequence, four-type proposal, PhaseState outline and
> diagnostic budgets below are not current requirements, approvals or launch authority.
> CPU0066's completed planning result remains preserved.<span data-glossary-navigation="true"> ([scalar glossary](../../../glossary.md#term-scalar); [vector glossary](../../../glossary.md#term-vector); [planning glossary](../../../glossary.md#term-planning))</span>

## Reader context

Read the archived proposal below as a superseded implementation direction, not as declarations
that applications should instantiate. Its useful distinction is between describing a Scan body
and supplying stored values for execution. In the conceptual `ScanBody` example, `h`,
`wStep` and `bStep` are formal Tensor descriptions with no producing expression in that body;
being such a leaf does not require allocated storage. `Scan.apply` relates those formals to
outer Tensor descriptions `h0`, `W` and `b`; it does not itself execute layers or copy their cells.
Later execution binds actual representations. The example's borrowed inputs remain the
supplier's responsibility and are not updated by the generated wrapper.

For its finite scalar-width example, let row $r\in\{0,1\}$ have initial value
$h_{0,r}\in\{1,2\}$, with layer weights $w_0=2$, $w_1=3$ and biases $b_0=1$, $b_1=-1$.
The real-number recurrence is $h_{\ell+1,r}=\tanh(h_{\ell,r}w_\ell+b_\ell)$ for
$\ell\in\{0,1\}$, so $h_{1,0}=\tanh(3)$ and
$h_{2,0}=\tanh(3\tanh(3)-1)$. This interprets the preserved example;
actual dtype producing sites round separately and must retain any demanded saved values.

The replacement's [binding-space explanation](existing-emitter-composition-plan.md#three-different-bindings-and-two-emission-lifetimes)
shows why no new generic binding object follows from sharing math.
[Source construction without input storage](../../../architecture/general-differentiable-scan.md#static-and-named-batch-source-foundation)
and [residual metadata versus history memory](../../../architecture/general-differentiable-scan.md#residual-metadata-is-not-history-memory)
explain the description/value boundary; the
[current CPU composition](../../../backend-guide/cpu-backend.md#current-general-scan-composition)
is the navigation point for present implementation rather than the archived type proposal.

## Purpose and status

This CPU0066 proposal removes two implementations of the same [CPU](../../../glossary.md#term-cpu) math without changing what
ordinary [graphs](../../../glossary.md#graph) or captured [Scan](../../../glossary.md#term-cumulative-scan) bodies [mean](../../../glossary.md#term-mean). It is non-authoritative [planning](../../../glossary.md#term-planning), not source-launch
permission. The proposed serial stages are CPU0067 common [scalar](../../../glossary.md#term-scalar) bodies, CPU0068 qualified [vector](../../../glossary.md#term-vector)
bodies, then CPU0069 worker integration. All are Draft; CPU0069 has only the bounded outline below.
A body is the already captured [`CompiledGraphModel`](../../../glossary.md#compiled-graph--compiledgraphmodel), not another graph or [runtime](../../../glossary.md#term-runtime) instruction list.

The relationship is `existing graph → common family lowering → typed IR → bound family emission`.
An ordinary caller binds entry operands/ranges; Scan binds time, old/new state, activity and histories.
Only those [bindings](../../../glossary.md#term-binding) differ. No generated call traverses a graph, dispatches an [operation](../../../glossary.md#operation) or allocates
per [element](../../../glossary.md#term-element)/step. Generation-time callbacks below build [bytecode](../../../glossary.md#term-bytecode); they are not runtime callbacks.

## Source-grounded seams

Paths in this table are under `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/`.
Each row identifies the existing limitation and the common [owner](../../../glossary.md#term-owner) proposed to remove it.

| Current source/[symbol](../../../glossary.md#term-symbol) | Proposed [ownership](../../../glossary.md#term-ownership-meanings) and caller adaptation |
|---|---|
| `lowering/CpuGeneralScanLowering.analyzePointwise/analyzeShaped/analyzeSymbolic` | Keep whole-[occurrence](../../../glossary.md#term-occurrence)/binder checks, roles, demand and temporal resources. Delegate instruction/[shape](../../../glossary.md#shape)/[axis](../../../glossary.md#term-axis)/[MATMUL](../../../glossary.md#matrix-multiplication--matmul) [normalization](../../../glossary.md#term-normalization) to existing family lowerers; remove their duplicated [operation](../../../glossary.md#operation) cases. |
| `lowering/CpuPartitionLowering.opcode/scalarImmediate/batchCanonical/batchAccess` | Shared [pointwise](../../../glossary.md#term-pointwise) instruction and access normalization, called with scoped local [descriptors](../../../glossary.md#term-descriptor)/ordinals or ordinary descriptors. No fabricated outer [`PrepareContext`](../../../glossary.md#prepare-context-preparecontext) for a body. |
| `lowering/CpuMatmulLowering.lower`, `ir/CpuMatmulIr` | Factor a facts-only [rank-two](../../../glossary.md#axis-rank-scalar-vector-and-matrix) body seam, separate from ordinary entry/route selection. [Scan](../../../glossary.md#term-cumulative-scan) supplies effective [views](../../../glossary.md#term-view-meanings) and coefficient recipes, not an ordinary-[batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters) entry. |
| `lowering/CpuAggregateLowering`, `ir/CpuAggregateIr` | Common axis/domain/[SUM-to-Shape](../../../glossary.md#sum-to-shape) normalization and explicit reduction algorithm/resource requirements. Ordinary exact-state and Scan represented-sequential variants remain different. |
| `codegen/emit/CpuClassFileKernelGenerator` [scalar](../../../glossary.md#term-scalar) body and `emitVectorBody`; `CpuScalarEmitter`, `CpuVectorInstructionEmitter`, `CpuLoopEmitter` | Extract common bound pointwise load/instruction/store composition; retain existing scalar/[vector](../../../glossary.md#term-vector) instruction math. Move reusable affine-nest/address proofs out of Scan into [binding](../../../glossary.md#term-binding)/loop seams. |
| `codegen/emit/CpuMatmulEmitter.emit`, `emitOrdinaryBatch`, direct/tiled/vector forms | Entry adapters decode carriers/geometry/ranges, then common family body methods emit full-K math. Move Scan's indexed/cursor [contraction](../../../glossary.md#contraction-dimension) variants here; preserve direct/tiled ordinary forms and CPU0065 [TANH](../../../glossary.md#term-tanh). |
| `codegen/emit/CpuAggregateEmitter.emit`, `CpuExactSumEmitter` | Entry adapter retains ordinary exact algorithm. Common body entry also owns represented-sequential SUM, raw movement and contributor gating, rather than calling a renamed Scan reduction implementation. |
| `codegen/emit/CpuGeneralScanEmitter.Shaped`, `CpuGeneralScanMatmulEmitter`, `CpuGeneralScanReductionEmitter` | Retain traversal/binding/[publication](../../../glossary.md#term-publication) only in Scan; retire both math subemitters after migration and [qualification](../../../glossary.md#term-qualification). `Shaped.arithmeticCell` delegates common pointwise emission. |
| `CpuGeneralScanStageSchedule.order`; [lowering](../../../glossary.md#lowering) `fuseMatmulStages/fuseReductionStages/shapedResources` | Separate common same-cell [fusion](../../../glossary.md#term-fusion) legality and inclusive cut/resource proofs from temporal [carry](../../../glossary.md#term-carry)/history constraints. Shared family seams consume proof facts, not a Scan [optimizer](../../../glossary.md#term-optimizer) hidden behind a [facade](../../../glossary.md#term-facade). |
| `prepare/CpuPartitionPreparer.analyzeGeneralScan`, `CpuPartitionFinalizer.finalizeComposite` | Analyze exact declarations before slots; finalize only those declarations. Currently forces scalar/one range; vector/workers require later explicit qualification. |
| `executable/CpuGeneralScanPreparedExecutable.Invocation.executeBound`, `CpuPreparedExecutable` | Preserve typed cold binding and [producer](../../../glossary.md#term-producer)-ordered length [preflight](../../../glossary.md#term-preflight). Later adapt one Scan invocation to existing range-call ownership, never one submission per stage/time. |

## Concrete common call flow and proposed types

CPU0067 proposes exactly two new package-private final top-level generation-only classes in
`io.github.pho001.synaptik.backend.cpu.internal.codegen.emit`:

- `CpuBodyBindings`: selected carrier/base/stride/extent/local bindings and exact write sinks for
  one emitted region. Minimal method surface: <span data-glossary-navigation="true"> ([carrier glossary](../../../glossary.md#term-storage-carrier))</span>`load(int value, int[] coordinateLocals)`,
  `store(int value, int[] coordinateLocals, int representedLocal)`,
  `active(int rowCoordinateLocal, Runnable emitBody)`. It borrows a `CodeBuilder`; value numbers
  index existing typed IR. Loads leave the represented value on the generated stack. Stores may
  target demanded distinct wrappers as well as a necessary cut. It owns no math, graph or schedule.
- `CpuPointwiseBodyEmitter`: `emitScalar(CpuKernelIr ir, List<CpuKernelIr.Instruction> instructions,
  CpuBodyBindings bindings, int[] representedLocals, int[] coordinateLocals)`. This owns common
  load/instruction/store ordering; `CpuScalarEmitter` remains the arithmetic owner. CPU0068 adds
  vector composition here using <span data-glossary-navigation="true"> ([owner glossary](../../../glossary.md#term-owner))</span>`CpuVectorInstructionEmitter`, not another [Scan](../../../glossary.md#term-cumulative-scan) [vector](../../../glossary.md#term-vector) emitter.

Two new nested facts live in existing `io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateIr`;
proposed declaration surface (not implemented):

```java
public enum Accumulation { EXACT_STATE, REPRESENTED_SEQUENTIAL }
public record Body(DataType dataType, Form form, List<Integer> selectedAxes, boolean keepDimensions,
        boolean divideByCount, Accumulation accumulation, CpuAccessPlan inputAccess,
        CpuAccessPlan outputAccess, int stateLimbCount, long scratchSliceBytes) {}
```

`Body` is the one shared normalized SUM/[MEAN](../../../glossary.md#term-mean) family [descriptor](../../../glossary.md#term-descriptor) for both callers, derived once by
common aggregate [lowering](../../../glossary.md#lowering), not duplicated [normalization](../../../glossary.md#term-normalization) or Scan-only [IR](../../../glossary.md#term-intermediate-representation-ir). Existing exact
constructors/[cache](../../../glossary.md#term-cache) invariants are preserved through the ordinary adapter. `divideByCount` admits only
exact MEAN; sequential admits only existing [F32](../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../glossary.md#floating-point-precision-and-rounding) SUM forms, zero limbs and zero exact-state
bytes. Raw-copy cases require no state. Existing `Kind.SUM` still means exact real sum; its
ordinary constructors keep their guards. No accuracy selector/`NumericalProfile` is introduced.
Numeric domains/[extent](../../../glossary.md#term-extent) recipes remain in existing `Geometry`/`BatchBinding`, not [canonical](../../../glossary.md#term-canonical-form) [identity](../../../glossary.md#term-identity-meanings).
Nested visibility permits the existing internal lowering/emission packages to share these facts;
it adds no supported public [API](../../../glossary.md#term-application-programming-interface-api), storage, [Runtime](../../../glossary.md#term-runtime) [handoff](../../../glossary.md#term-handoff) or execution methods.

The explicit proposed family entry points are `CpuMatmulEmitter.emitBody(CodeBuilder,
CpuMatmulIr, CpuBodyBindings)` and `CpuAggregateEmitter.emitBody(CodeBuilder,
CpuAggregateIr.Body, CpuBodyBindings)`. Their [binding](../../../glossary.md#term-binding) supplies complete-output ranges and
dominating extent/[stride](../../../glossary.md#term-stride) locals; existing entry methods remain adapters. Generation-only
[coordinate](../../../glossary.md#axis-rank-scalar-vector-and-matrix) loops use `CpuLoopEmitter`; no binding method switches on an [operation](../../../glossary.md#operation) family.

1. Ordinary [partition](../../../glossary.md#partition) analysis and [Scan body](../../../glossary.md#backpropagation-through-time-and-recurrent-state) analysis both call facts-only helpers in
   `CpuPartitionLowering`, `CpuMatmulLowering` and `CpuAggregateLowering`. Extract shared validators,
   not just emitter calls. Use existing affine/access facts for layout edits; ordinary supported
   occurrence predicates and Scan's whole-binder/role predicates remain independent admission.<span data-glossary-navigation="true"> ([layout glossary](../../../glossary.md#layout); [occurrence glossary](../../../glossary.md#term-occurrence); [admission glossary](../../../glossary.md#term-admission))</span>
2. Scan's existing `Stage` references existing pointwise instructions or typed MATMUL/aggregate
   body facts. Do not encode sequential SUM as an exact ordinary SUM with missing scratch, or
   infer a family/algorithm by parsing diagnostic text. Retain original operand order and alias
   origins. A maintained second Scan opcode-to-math registry is not an acceptable end state.<span data-glossary-navigation="true"> ([pointwise glossary](../../../glossary.md#term-pointwise); [MATMUL glossary](../../../glossary.md#matrix-multiplication--matmul); [alias glossary](../../../glossary.md#aliasing-and-arena); [opcode glossary](../../../glossary.md#term-opcode); [registry glossary](../../../glossary.md#term-registry))</span>
3. Common analysis derives math requirements, fusion legality and cut lifetimes. Scan contributes
   temporal old/new versions, activity scopes, saved/publication barriers and port sinks. Both
   callers use those same proofs; no shared facade delegates back to Scan-specific math/fusion.<span data-glossary-navigation="true"> ([publication glossary](../../../glossary.md#term-publication); [facade glossary](../../../glossary.md#term-facade))</span>
4. CPU analysis declares exact buffers/workspaces; shared Prepare assigns outer slots/owner
   positions; finalization binds those slots to code-generation locals and emits adapters/body.
   Local <span data-glossary-navigation="true"> ([buffers glossary](../../../glossary.md#term-buffer); [Prepare glossary](../../../glossary.md#prepare))</span>[`ValueId`](../../../glossary.md#logical-valueid-versus-physical-memory-slot)s belong only to their `CompiledScanBody`; numeric coincidence with outer IDs
   never selects a buffer. Local cuts use declared workspace offsets or demanded outer-port
   bindings, not new shared logical slots. Only invocation-owned numeric arrays reach execution.

[Compiler](../../../glossary.md#term-compiler) optimization is a separate boundary: `ForwardGraphOptimization` preserves body
attachments/saved sites and explicitly performs no body rewrite/[fusion](../../../glossary.md#term-fusion)/recomputation/history
allocation; `ScanBodyInference.capture` [captures](../../../glossary.md#term-capture-meanings)/infers the existing body, and `GraphCompiler`
optimizes the outer [graph](../../../glossary.md#graph) then refreshes demand. This plan shares [CPU](../../../glossary.md#term-cpu) [target](../../../glossary.md#term-target) lowering/fusion,
not recursive ordinary Compiler optional passes. Such a Compiler extension would need separate
[scope](../../../glossary.md#term-scope) and saved-site/[AD](../../../glossary.md#autograd) remapping; it is neither included nor a prerequisite here.

## Semantics, resources and retained optimizations

MAIN's read-only SUM audit is incorporated: `recurrent-scan.md` **Actual first-order closure and
arithmetic** explicitly distinguishes the algorithms; `CpuAggregateIr.Kind.SUM`/constructor,
`CpuAggregateEmitter` and `CpuExactSumEmitter` own exact-state ordinary sums, while
`CpuGeneralScanReductionEmitter.emit/emitSymbolic` currently owns represented fadd/dadd folds.
Common [ownership](../../../glossary.md#term-ownership-meanings) means both explicit variants, not uniform results. Preserve raw [singleton](../../../glossary.md#term-singleton)/no-[axis](../../../glossary.md#term-axis)
movement, empty [positive zero](../../../glossary.md#nan-infinity-and-signed-zero), [NaN](../../../glossary.md#term-nan)/Inf/zero rules, [F64](../../../glossary.md#floating-point-precision-and-rounding) [subnormals](../../../glossary.md#term-subnormal) and ordinary rational [MEAN](../../../glossary.md#term-mean) [rounding](../../../glossary.md#term-rounding).
Preserve ordinary [F32](../../../glossary.md#floating-point-precision-and-rounding) five-[limb](../../../glossary.md#exact-superaccumulator-and-limbs)/48-byte and bounded symbolic F64 34-limb/280-byte state rules;
static F64 remains domain-sized. Sequential bodies gain no exact workspace. Requirements must be
typed and fixed before shared assignment, including structural zero workspaces and all-B bounds.

Preserve public/[fan-out](../../../glossary.md#term-fan-out)/demanded saved sites, exact stored [TANH](../../../glossary.md#term-tanh)/[SIGMOID](../../../glossary.md#term-sigmoid) values and separate per-time
shared [carry](../../../glossary.md#term-carry) ADD. No recomputation or cross-boundary [reassociation](../../../glossary.md#term-reassociation). Activity dominates both operand
loads and dependent contributor math; SHARED time gating differs from ROW(0)/ROW(1) gating. Preserve
raw inactive carry and positive-zero [padding](../../../glossary.md#padding); histories store at original time, including previous
carry before the step. T=0, B=0, empty K and zero [features](../../../glossary.md#term-software-feature) remain distinct semantic cases.

Move, do not discard, v9's direct stable entry carriers/temporal bases, explicit-B affine facts,
coefficient-proved zero scratch bases, cell-scoped address sharing and [pointwise](../../../glossary.md#term-pointwise) prefixes,
indexed feature/active-row contractions and bank-free stage scheduling. Retain fallback only
where those proofs fail. Inclusive scratch lifetimes, physical port [alias](../../../glossary.md#aliasing-and-arena) hazards and simultaneous
old/new carry versions constrain scheduling. Do not enlarge scratch/copy work to simplify adapters.

Current [scope](../../../glossary.md#term-scope) is static all-valid F32/F64, symbolic all-valid F32 only, and static/symbolic length
F32/F64, each with actual first-order siblings and existing dense/[scalar](../../../glossary.md#term-scalar)-read exceptions. A shared
body recipe must not relax `CpuMatmulIr.batchGeometry`'s ordinary F32/direct-scalar/no-[epilogue](../../../glossary.md#term-epilogue) [guard](../../../glossary.md#term-guard).
[Scan](../../../glossary.md#term-cumulative-scan) F64 coefficient [binding](../../../glossary.md#term-binding) is not ordinary symbolic F64 [MATMUL](../../../glossary.md#matrix-multiplication--matmul) [admission](../../../glossary.md#term-admission). Low types, arbitrary
[layouts](../../../glossary.md#layout), new body families, [Metal](../../../glossary.md#term-metal)/[OpenBLAS](../../../glossary.md#term-openblas) Scan, higher-order [AD](../../../glossary.md#autograd) and [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) migration remain excluded.

## [Vector](../../../glossary.md#term-vector) realization, then worker integration

CPU0068 first qualifies single-thread [feature](../../../glossary.md#term-software-feature)-[lane](../../../glossary.md#term-lane) [pointwise](../../../glossary.md#term-pointwise) and full-K/N-vector [MATMUL](../../../glossary.md#matrix-multiplication--matmul) bodies
only where exact [contiguous](../../../glossary.md#term-contiguous) feature accesses and complete lane activity are proved. Reuse existing
vector math/[TANH](../../../glossary.md#term-tanh) and [scalar](../../../glossary.md#term-scalar) tails. Do not vector-reduce sequential SUM or split K; transposed
[batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters) contractions may remain scalar. Qualify saved/state sites independently: save exactly the
bits produced by the selected forward realization and use those values in reverse. This does not
require vector TANH/[SIGMOID](../../../glossary.md#term-sigmoid) to match scalar last bits. Keep a site scalar if its actual retained-value
contract cannot be satisfied. Ordinary-[unrolling](../../../glossary.md#term-unrolling) comparisons must account for the distinct exact
and represented-sequential SUM algorithms; use matching-algorithm oracles for exact arithmetic
checks. No [dtype](../../../glossary.md#data-type--datatype)/[layout](../../../glossary.md#layout)/family support follows from a child vector [opcode](../../../glossary.md#term-opcode). Keep an explicit legal
scalar choice during [prepare](../../../glossary.md#prepare).

CPU0069 is one later cohesive [CPU](../../../glossary.md#term-cpu)-private orchestration stage, not new math. MAIN's worker audit
finds that `CpuWorkerGroup.execute` dynamically claims ranges, charges min(workers,calls) once and
joins synchronously; failures/close/interruption cancel unclaimed work but cannot [release](../../../glossary.md#term-release) an
arbitrary barrier. Thus calls>workers or a cancelled unstarted party can [deadlock](../../../glossary.md#term-deadlock) a naive barrier.

- Proved row-independent inference: each admitted call owns its rows across the entire time loop;
  one unchanged <span data-glossary-navigation="true"> ([inference glossary](../../../glossary.md#term-inference))</span>`execute` submission, no phases. Prove all carries, emitted/history writes and
  local scratch disjoint; a row dependency or shared write defeats this proof.<span data-glossary-navigation="true"> ([dependency glossary](../../../glossary.md#term-dependency))</span>
- Shared-gradient/column-tiled bodies: prepare dependency phases and complete-output ownership.
  A worker owns each shared-gradient cell's full ordered K and one represented carry ADD per
  time. Never make partial/reassociated sums or B×parameter banks. Dependencies, old/new carry
  reads and cut reuse determine phase points, not an unconditional barrier after every operation.<span data-glossary-navigation="true"> ([gradient glossary](../../../glossary.md#term-gradient); [ownership glossary](../../../glossary.md#term-ownership-meanings); [carry glossary](../../../glossary.md#term-carry); [parameter glossary](../../../glossary.md#term-parameter); [operation glossary](../../../glossary.md#operation))</span>
- Proposed nested `CpuWorkerGroup.PhaseState` is run-owned CPU-private coordination only;
  its final signature is deferred to CPU0069's Ready packet. Admit exactly the actual concurrent
  participant calls, at most worker capacity, with one long-lived call per party; tile work stays
  inside each call rather than increasing barrier parties. All parties cross matching phase
  points, even when their local rows are empty/inactive. No collective sits inside a length guard.<span data-glossary-navigation="true"> ([run glossary](../../../glossary.md#term-run); [signature glossary](../../../glossary.md#term-signature); [Ready glossary](../../../glossary.md#term-ready); [guard glossary](../../../glossary.md#term-guard))</span>
- Integrate a common abort notification with worker failure, admission failure, close and coordinator
  interruption <span data-glossary-navigation="true"> ([admission glossary](../../../glossary.md#term-admission))</span>**before** joining started waiters, including calls never started. Abort terminates
  every phase wait. Preserve deterministic range-index failures/suppression and restored interrupt
  status. No public asynchronous/cancellation API or new executor/pool is proposed.<span data-glossary-navigation="true"> ([asynchronous glossary](../../../glossary.md#term-asynchronous); [cancellation glossary](../../../glossary.md#term-cancellation); [API glossary](../../../glossary.md#term-application-programming-interface-api))</span>
- Replace [Scan](../../../glossary.md#term-cumulative-scan)'s inline `budget.acquire(1)` on the parallel branch with the group's single N-permit
  acquisition. No nested acquisition/submission. Zero work submits none; one participant runs
  inline. Preflight all produced lengths before Scan writes/submission. Check every selected
  segment/workspace with <span data-glossary-navigation="true"> ([Preflight glossary](../../../glossary.md#term-preflight))</span>`workersCanAccess`/`isAccessibleBy`, not coordinator accessibility alone.
- Declare per-worker/inter-stage scratch, histories and versioned alias lifetimes before slots,
  with fresh phase/control/argument arrays under the one RunState. Fixed participant capacity and
  storage topology are prepared; actual B resolves only numeric sizes/ranges under approved rules.
  Success/failure returns only after quiescence; run cleanup and lease release follow that barrier.
  Keep the prepared single-thread path for unproved dependencies/resources/accessibility.<span data-glossary-navigation="true"> ([alias glossary](../../../glossary.md#aliasing-and-arena); [argument glossary](../../../glossary.md#term-argument); [RunState glossary](../../../glossary.md#run-state--runstate); [topology glossary](../../../glossary.md#term-topology); [quiescence glossary](../../../glossary.md#term-quiescence); [lease glossary](../../../glossary.md#term-lease))</span>

## Emitted-wrapper examples (conceptual, not APIs)

These illustrate [binding](../../../glossary.md#term-binding), not [runtime](../../../glossary.md#term-runtime) callbacks, automatic [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) packing or a new [Model](../../../glossary.md#term-model) [API](../../../glossary.md#term-application-programming-interface-api).
For a repeated-width [multilayer perceptron](../../../glossary.md#term-multilayer-perceptron-mlp) ([MLP](../../../glossary.md#neural-network-families)), callers already supply h0 `[B,H]`, distinct
[weight](../../../glossary.md#term-weight) blocks in a [Tensor](../../../glossary.md#tensor) sequence W `[L,H,H]` and [bias](../../../glossary.md#term-bias) blocks b `[L,H]`. This is an actual
existing low-level all-valid [Scan](../../../glossary.md#term-cumulative-scan), not conversion/packing of existing NN layer objects.
With ordinary Tensor formals h `[B,H]`, wStep `[H,H]`, bStep `[H]`, source construction is:

```java
// Conceptual use of existing API; formals and actual Tensor sequences already exist.
var body = new ScanBody(List.of(h), List.of(wStep, bStep), List.of(),
        List.of(h.matmul(wStep).add(bStep).tanh()), List.of());
var layers = Scan.apply(List.of(h0), List.of(W, b), List.of(), body, false);
```

Schematic generated wrapper (inline code from the normal shared emitters, not helper calls):

```text
bind raw initial h; bind invariant extents and W/b sequence strides
for (layer = 0; layer < L; layer++) {
    bind W_step = W[layer], b_step = b[layer], old_h and disjoint/proved next_h
    inline common full-K MATMUL body(old_h, W_step)
    inline common ordered ADD b_step and TANH body
    save demanded represented sites; transition h only after all old_h reads
}
publish final h
```

For B=2, H=1, L=2, h0=[1,2], W blocks [2],[3] and b blocks [1],[-1], the first layer is
[[tanh](../../../glossary.md#term-tanh)(3),tanh(5)]; final h is [tanh(3*tanh(3)-1),tanh(3*tanh(5)-1)] at the [dtype](../../../glossary.md#data-type--datatype)'s producing sites.
Compare outputs/actual [AD](../../../glossary.md#autograd) with two ordinary Tensor layers using those same distinct blocks.
At L=0, final [carry](../../../glossary.md#term-carry) is raw h0 without body work. Inputs remain [borrowed](../../../glossary.md#ownership-borrowing-and-lease) under existing [lifetime](../../../glossary.md#term-lifetime)
rules, outputs use existing distinct [canonical](../../../glossary.md#term-canonical-form) ports, and caller storage is never updated.
Demanded layer histories/saved sites remain represented storage; [fusion](../../../glossary.md#term-fusion) cannot remove them.

For a forward [RNN](../../../glossary.md#term-recurrent-neural-network-rnn) with T=2, B=2, F=H=1, zero h0, Wx=Wh=[1], bias=[0], X0=[1,2], X1=[3,4]
and lengths [2,1], the generated wrapper binds each original-time X [slice](../../../glossary.md#slice) and old h to the same
common [MATMUL](../../../glossary.md#matrix-multiplication--matmul)/[pointwise](../../../glossary.md#term-pointwise) body. After t=0, h=[tanh(1),tanh(2)]; after t=1,
h=[tanh(3+tanh(1)),tanh(2)] and the inactive second emission is +0. Previous-state histories
precede updates. Reverse sibling binds saved primals/[seeds](../../../glossary.md#commit-release-distribution-log-and-seed-meanings) at original time with opposite traversal;
it does not replay TANH. Vectorization needs [feature](../../../glossary.md#term-software-feature) [lanes](../../../glossary.md#term-lane), so this H=1 example may remain [scalar](../../../glossary.md#term-scalar).

## Authority and launch gates

CPU0067 remains Draft until **user** approval of all four named types/seams above (including nested
`CpuAggregateIr.Body`/`Accumulation`), as required by [Scan](../../../glossary.md#term-cumulative-scan) **[Qualification](../../../glossary.md#term-qualification) and mechanism gates**
and the [backend](../../../glossary.md#backend) [owner](../../../glossary.md#term-owner); private generation-only types are not exempt and MAIN cannot self-approve.
The packet needs exact packages/names/purposes/[signatures](../../../glossary.md#term-signature) plus the diagnostic budget in CPU0067.
Reviewed [planning](../../../glossary.md#term-planning)/dependencies and MAIN acceptance are also necessary before [Ready](../../../glossary.md#term-ready). Extraction
alone needs no root/[ADR](../../../glossary.md#term-architecture-decision-record-adr) rule. Preserve old bytes where possible; enumerate changed [artifacts](../../../glossary.md#term-artifact) and
advance both [identity](../../../glossary.md#term-identity-meanings) streams. Update only affected root **Core invariants**/backend **[CPU](../../../glossary.md#term-cpu) backend
routes** retention/version facts before conflicting generation; ADR only for a significant actual
decision. No authority edits or uniform-numerics change occur here.

Before CPU0068/0069, separately approve bounded [vector](../../../glossary.md#term-vector)/parallel qualification in root/backend
and Scan **Qualification, performance and migration gates**, **Qualification and mechanism gates**
and length **[Runtime](../../../glossary.md#term-runtime) [preflight](../../../glossary.md#term-preflight) and implementation gates**. Runtime **[Run](../../../glossary.md#term-run) [lifecycle](../../../glossary.md#lifecycle)** remains
[synchronous](../../../glossary.md#term-synchronous) and unchanged; any necessary [handoff](../../../glossary.md#term-handoff)/[lease](../../../glossary.md#term-lease)/[API](../../../glossary.md#term-application-programming-interface-api) change is a stop, not part of this plan.
No uniform-numerics, ordinary [F64](../../../glossary.md#floating-point-precision-and-rounding) [MATMUL](../../../glossary.md#matrix-multiplication--matmul) or broad vector/parallel grant is proposed.

## [Validation](../../../glossary.md#term-validation) and task sequence

[CPU0067](tasks/0067-common-scalar-body-lowering-and-emission.md) and its immediate successor
[CPU0068](tasks/0068-qualified-shared-body-vector-emission.md) contain focused acceptance/commands.
CPU0069 gets a compact executable brief only after its predecessors stabilize and its participant/
abort design receives independent review. All three require clean Class C implementation and
independent targeted code/docs review. Affected [CPU](../../../glossary.md#term-cpu) tests/[Javadoc](../../../glossary.md#term-javadoc) [run](../../../glossary.md#term-run) once per stable stage;
MAIN owns one named [conformance](../../../glossary.md#term-conformance)/[Engine](../../../glossary.md#term-engine) [checkpoint](../../../glossary.md#term-checkpoint) at each capability cutover, reusing evidence.
New shared build/[dependency](../../../glossary.md#term-dependency)/[Runtime](../../../glossary.md#term-runtime) work would require a revised [scope](../../../glossary.md#term-scope) and repository checkpoint.
Use clean specialized Java and Class-File inspection, ordinary-[unrolled](../../../glossary.md#backpropagation-through-time-and-recurrent-state) forward/actual-[AD](../../../glossary.md#autograd) oracles,
[cache](../../../glossary.md#term-cache)/stale-[identity](../../../glossary.md#term-identity-meanings) checks and each brief's newly proposed one-[JVM](../../../glossary.md#term-java-virtual-machine-jvm)/120-second diagnostic, not an
old five-[fork](../../../glossary.md#term-fork-meanings) campaign or universal speed gate. MAIN accepts the [finite](../../../glossary.md#term-finite) protocol before [Ready](../../../glossary.md#term-ready).

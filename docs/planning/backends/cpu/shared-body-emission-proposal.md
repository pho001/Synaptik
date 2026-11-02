# Shared ordinary and Scan body lowering and emission

> Archived by CPU0070 on 2026-10-29. The active replacement is
> [Compose existing CPU operation emission](existing-emitter-composition-plan.md).
> The historical scalar/vector/worker sequence, four-type proposal, PhaseState outline and
> diagnostic budgets below are not current requirements, approvals or launch authority.
> CPU0066's completed planning result remains preserved.

## Purpose and status

This CPU0066 proposal removes two implementations of the same CPU math without changing what
ordinary graphs or captured Scan bodies mean. It is non-authoritative planning, not source-launch
permission. The proposed serial stages are CPU0067 common scalar bodies, CPU0068 qualified vector
bodies, then CPU0069 worker integration. All are Draft; CPU0069 has only the bounded outline below.
A body is the already captured `CompiledGraphModel`, not another graph or runtime instruction list.

The relationship is `existing graph → common family lowering → typed IR → bound family emission`.
An ordinary caller binds entry operands/ranges; Scan binds time, old/new state, activity and histories.
Only those bindings differ. No generated call traverses a graph, dispatches an operation or allocates
per element/step. Generation-time callbacks below build bytecode; they are not runtime callbacks.

## Source-grounded seams

Paths in this table are under `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/`.
Each row identifies the existing limitation and the common owner proposed to remove it.

| Current source/symbol | Proposed ownership and caller adaptation |
|---|---|
| `lowering/CpuGeneralScanLowering.analyzePointwise/analyzeShaped/analyzeSymbolic` | Keep whole-occurrence/binder checks, roles, demand and temporal resources. Delegate instruction/shape/axis/MATMUL normalization to existing family lowerers; remove their duplicated operation cases. |
| `lowering/CpuPartitionLowering.opcode/scalarImmediate/batchCanonical/batchAccess` | Shared pointwise instruction and access normalization, called with scoped local descriptors/ordinals or ordinary descriptors. No fabricated outer `PrepareContext` for a body. |
| `lowering/CpuMatmulLowering.lower`, `ir/CpuMatmulIr` | Factor a facts-only rank-two body seam, separate from ordinary entry/route selection. Scan supplies effective views and coefficient recipes, not an ordinary-batch entry. |
| `lowering/CpuAggregateLowering`, `ir/CpuAggregateIr` | Common axis/domain/SUM-to-Shape normalization and explicit reduction algorithm/resource requirements. Ordinary exact-state and Scan represented-sequential variants remain different. |
| `codegen/emit/CpuClassFileKernelGenerator` scalar body and `emitVectorBody`; `CpuScalarEmitter`, `CpuVectorInstructionEmitter`, `CpuLoopEmitter` | Extract common bound pointwise load/instruction/store composition; retain existing scalar/vector instruction math. Move reusable affine-nest/address proofs out of Scan into binding/loop seams. |
| `codegen/emit/CpuMatmulEmitter.emit`, `emitOrdinaryBatch`, direct/tiled/vector forms | Entry adapters decode carriers/geometry/ranges, then common family body methods emit full-K math. Move Scan's indexed/cursor contraction variants here; preserve direct/tiled ordinary forms and CPU0065 TANH. |
| `codegen/emit/CpuAggregateEmitter.emit`, `CpuExactSumEmitter` | Entry adapter retains ordinary exact algorithm. Common body entry also owns represented-sequential SUM, raw movement and contributor gating, rather than calling a renamed Scan reduction implementation. |
| `codegen/emit/CpuGeneralScanEmitter.Shaped`, `CpuGeneralScanMatmulEmitter`, `CpuGeneralScanReductionEmitter` | Retain traversal/binding/publication only in Scan; retire both math subemitters after migration and qualification. `Shaped.arithmeticCell` delegates common pointwise emission. |
| `CpuGeneralScanStageSchedule.order`; lowering `fuseMatmulStages/fuseReductionStages/shapedResources` | Separate common same-cell fusion legality and inclusive cut/resource proofs from temporal carry/history constraints. Shared family seams consume proof facts, not a Scan optimizer hidden behind a facade. |
| `prepare/CpuPartitionPreparer.analyzeGeneralScan`, `CpuPartitionFinalizer.finalizeComposite` | Analyze exact declarations before slots; finalize only those declarations. Currently forces scalar/one range; vector/workers require later explicit qualification. |
| `executable/CpuGeneralScanPreparedExecutable.Invocation.executeBound`, `CpuPreparedExecutable` | Preserve typed cold binding and producer-ordered length preflight. Later adapt one Scan invocation to existing range-call ownership, never one submission per stage/time. |

## Concrete common call flow and proposed types

CPU0067 proposes exactly two new package-private final top-level generation-only classes in
`io.github.pho001.synaptik.backend.cpu.internal.codegen.emit`:

- `CpuBodyBindings`: selected carrier/base/stride/extent/local bindings and exact write sinks for
  one emitted region. Minimal method surface: `load(int value, int[] coordinateLocals)`,
  `store(int value, int[] coordinateLocals, int representedLocal)`,
  `active(int rowCoordinateLocal, Runnable emitBody)`. It borrows a `CodeBuilder`; value numbers
  index existing typed IR. Loads leave the represented value on the generated stack. Stores may
  target demanded distinct wrappers as well as a necessary cut. It owns no math, graph or schedule.
- `CpuPointwiseBodyEmitter`: `emitScalar(CpuKernelIr ir, List<CpuKernelIr.Instruction> instructions,
  CpuBodyBindings bindings, int[] representedLocals, int[] coordinateLocals)`. This owns common
  load/instruction/store ordering; `CpuScalarEmitter` remains the arithmetic owner. CPU0068 adds
  vector composition here using `CpuVectorInstructionEmitter`, not another Scan vector emitter.

Two new nested facts live in existing `io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateIr`;
proposed declaration surface (not implemented):

```java
public enum Accumulation { EXACT_STATE, REPRESENTED_SEQUENTIAL }
public record Body(DataType dataType, Form form, List<Integer> selectedAxes, boolean keepDimensions,
        boolean divideByCount, Accumulation accumulation, CpuAccessPlan inputAccess,
        CpuAccessPlan outputAccess, int stateLimbCount, long scratchSliceBytes) {}
```

`Body` is the one shared normalized SUM/MEAN family descriptor for both callers, derived once by
common aggregate lowering, not duplicated normalization or Scan-only IR. Existing exact
constructors/cache invariants are preserved through the ordinary adapter. `divideByCount` admits only
exact MEAN; sequential admits only existing F32/F64 SUM forms, zero limbs and zero exact-state
bytes. Raw-copy cases require no state. Existing `Kind.SUM` still means exact real sum; its
ordinary constructors keep their guards. No accuracy selector/`NumericalProfile` is introduced.
Numeric domains/extent recipes remain in existing `Geometry`/`BatchBinding`, not canonical identity.
Nested visibility permits the existing internal lowering/emission packages to share these facts;
it adds no supported public API, storage, Runtime handoff or execution methods.

The explicit proposed family entry points are `CpuMatmulEmitter.emitBody(CodeBuilder,
CpuMatmulIr, CpuBodyBindings)` and `CpuAggregateEmitter.emitBody(CodeBuilder,
CpuAggregateIr.Body, CpuBodyBindings)`. Their binding supplies complete-output ranges and
dominating extent/stride locals; existing entry methods remain adapters. Generation-only
coordinate loops use `CpuLoopEmitter`; no binding method switches on an operation family.

1. Ordinary partition analysis and Scan body analysis both call facts-only helpers in
   `CpuPartitionLowering`, `CpuMatmulLowering` and `CpuAggregateLowering`. Extract shared validators,
   not just emitter calls. Use existing affine/access facts for layout edits; ordinary supported
   occurrence predicates and Scan's whole-binder/role predicates remain independent admission.
2. Scan's existing `Stage` references existing pointwise instructions or typed MATMUL/aggregate
   body facts. Do not encode sequential SUM as an exact ordinary SUM with missing scratch, or
   infer a family/algorithm by parsing diagnostic text. Retain original operand order and alias
   origins. A maintained second Scan opcode-to-math registry is not an acceptable end state.
3. Common analysis derives math requirements, fusion legality and cut lifetimes. Scan contributes
   temporal old/new versions, activity scopes, saved/publication barriers and port sinks. Both
   callers use those same proofs; no shared facade delegates back to Scan-specific math/fusion.
4. CPU analysis declares exact buffers/workspaces; shared Prepare assigns outer slots/owner
   positions; finalization binds those slots to code-generation locals and emits adapters/body.
   Local `ValueId`s belong only to their `CompiledScanBody`; numeric coincidence with outer IDs
   never selects a buffer. Local cuts use declared workspace offsets or demanded outer-port
   bindings, not new shared logical slots. Only invocation-owned numeric arrays reach execution.

Compiler optimization is a separate boundary: `ForwardGraphOptimization` preserves body
attachments/saved sites and explicitly performs no body rewrite/fusion/recomputation/history
allocation; `ScanBodyInference.capture` captures/infers the existing body, and `GraphCompiler`
optimizes the outer graph then refreshes demand. This plan shares CPU target lowering/fusion,
not recursive ordinary Compiler optional passes. Such a Compiler extension would need separate
scope and saved-site/AD remapping; it is neither included nor a prerequisite here.

## Semantics, resources and retained optimizations

MAIN's read-only SUM audit is incorporated: `recurrent-scan.md` **Actual first-order closure and
arithmetic** explicitly distinguishes the algorithms; `CpuAggregateIr.Kind.SUM`/constructor,
`CpuAggregateEmitter` and `CpuExactSumEmitter` own exact-state ordinary sums, while
`CpuGeneralScanReductionEmitter.emit/emitSymbolic` currently owns represented fadd/dadd folds.
Common ownership means both explicit variants, not uniform results. Preserve raw singleton/no-axis
movement, empty positive zero, NaN/Inf/zero rules, F64 subnormals and ordinary rational MEAN rounding.
Preserve ordinary F32 five-limb/48-byte and bounded symbolic F64 34-limb/280-byte state rules;
static F64 remains domain-sized. Sequential bodies gain no exact workspace. Requirements must be
typed and fixed before shared assignment, including structural zero workspaces and all-B bounds.

Preserve public/fan-out/demanded saved sites, exact stored TANH/SIGMOID values and separate per-time
shared carry ADD. No recomputation or cross-boundary reassociation. Activity dominates both operand
loads and dependent contributor math; SHARED time gating differs from ROW(0)/ROW(1) gating. Preserve
raw inactive carry and positive-zero padding; histories store at original time, including previous
carry before the step. T=0, B=0, empty K and zero features remain distinct semantic cases.

Move, do not discard, v9's direct stable entry carriers/temporal bases, explicit-B affine facts,
coefficient-proved zero scratch bases, cell-scoped address sharing and pointwise prefixes,
indexed feature/active-row contractions and bank-free stage scheduling. Retain fallback only
where those proofs fail. Inclusive scratch lifetimes, physical port alias hazards and simultaneous
old/new carry versions constrain scheduling. Do not enlarge scratch/copy work to simplify adapters.

Current scope is static all-valid F32/F64, symbolic all-valid F32 only, and static/symbolic length
F32/F64, each with actual first-order siblings and existing dense/scalar-read exceptions. A shared
body recipe must not relax `CpuMatmulIr.batchGeometry`'s ordinary F32/direct-scalar/no-epilogue guard.
Scan F64 coefficient binding is not ordinary symbolic F64 MATMUL admission. Low types, arbitrary
layouts, new body families, Metal/OpenBLAS Scan, higher-order AD and NN migration remain excluded.

## Vector realization, then worker integration

CPU0068 first qualifies single-thread feature-lane pointwise and full-K/N-vector MATMUL bodies
only where exact contiguous feature accesses and complete lane activity are proved. Reuse existing
vector math/TANH and scalar tails. Do not vector-reduce sequential SUM or split K; transposed
batch contractions may remain scalar. Qualify saved/state sites independently: save exactly the
bits produced by the selected forward realization and use those values in reverse. This does not
require vector TANH/SIGMOID to match scalar last bits. Keep a site scalar if its actual retained-value
contract cannot be satisfied. Ordinary-unrolling comparisons must account for the distinct exact
and represented-sequential SUM algorithms; use matching-algorithm oracles for exact arithmetic
checks. No dtype/layout/family support follows from a child vector opcode. Keep an explicit legal
scalar choice during prepare.

CPU0069 is one later cohesive CPU-private orchestration stage, not new math. MAIN's worker audit
finds that `CpuWorkerGroup.execute` dynamically claims ranges, charges min(workers,calls) once and
joins synchronously; failures/close/interruption cancel unclaimed work but cannot release an
arbitrary barrier. Thus calls>workers or a cancelled unstarted party can deadlock a naive barrier.

- Proved row-independent inference: each admitted call owns its rows across the entire time loop;
  one unchanged `execute` submission, no phases. Prove all carries, emitted/history writes and
  local scratch disjoint; a row dependency or shared write defeats this proof.
- Shared-gradient/column-tiled bodies: prepare dependency phases and complete-output ownership.
  A worker owns each shared-gradient cell's full ordered K and one represented carry ADD per
  time. Never make partial/reassociated sums or B×parameter banks. Dependencies, old/new carry
  reads and cut reuse determine phase points, not an unconditional barrier after every operation.
- Proposed nested `CpuWorkerGroup.PhaseState` is run-owned CPU-private coordination only;
  its final signature is deferred to CPU0069's Ready packet. Admit exactly the actual concurrent
  participant calls, at most worker capacity, with one long-lived call per party; tile work stays
  inside each call rather than increasing barrier parties. All parties cross matching phase
  points, even when their local rows are empty/inactive. No collective sits inside a length guard.
- Integrate a common abort notification with worker failure, admission failure, close and coordinator
  interruption **before** joining started waiters, including calls never started. Abort terminates
  every phase wait. Preserve deterministic range-index failures/suppression and restored interrupt
  status. No public asynchronous/cancellation API or new executor/pool is proposed.
- Replace Scan's inline `budget.acquire(1)` on the parallel branch with the group's single N-permit
  acquisition. No nested acquisition/submission. Zero work submits none; one participant runs
  inline. Preflight all produced lengths before Scan writes/submission. Check every selected
  segment/workspace with `workersCanAccess`/`isAccessibleBy`, not coordinator accessibility alone.
- Declare per-worker/inter-stage scratch, histories and versioned alias lifetimes before slots,
  with fresh phase/control/argument arrays under the one RunState. Fixed participant capacity and
  storage topology are prepared; actual B resolves only numeric sizes/ranges under approved rules.
  Success/failure returns only after quiescence; run cleanup and lease release follow that barrier.
  Keep the prepared single-thread path for unproved dependencies/resources/accessibility.

## Emitted-wrapper examples (conceptual, not APIs)

These illustrate binding, not runtime callbacks, automatic NN packing or a new Model API.
For a repeated-width multilayer perceptron (MLP), callers already supply h0 `[B,H]`, distinct
weight blocks in a Tensor sequence W `[L,H,H]` and bias blocks b `[L,H]`. This is an actual
existing low-level all-valid Scan, not conversion/packing of existing NN layer objects.
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
[tanh(3),tanh(5)]; final h is [tanh(3*tanh(3)-1),tanh(3*tanh(5)-1)] at the dtype's producing sites.
Compare outputs/actual AD with two ordinary Tensor layers using those same distinct blocks.
At L=0, final carry is raw h0 without body work. Inputs remain borrowed under existing lifetime
rules, outputs use existing distinct canonical ports, and caller storage is never updated.
Demanded layer histories/saved sites remain represented storage; fusion cannot remove them.

For a forward RNN with T=2, B=2, F=H=1, zero h0, Wx=Wh=[1], bias=[0], X0=[1,2], X1=[3,4]
and lengths [2,1], the generated wrapper binds each original-time X slice and old h to the same
common MATMUL/pointwise body. After t=0, h=[tanh(1),tanh(2)]; after t=1,
h=[tanh(3+tanh(1)),tanh(2)] and the inactive second emission is +0. Previous-state histories
precede updates. Reverse sibling binds saved primals/seeds at original time with opposite traversal;
it does not replay TANH. Vectorization needs feature lanes, so this H=1 example may remain scalar.

## Authority and launch gates

CPU0067 remains Draft until **user** approval of all four named types/seams above (including nested
`CpuAggregateIr.Body`/`Accumulation`), as required by Scan **Qualification and mechanism gates**
and the backend owner; private generation-only types are not exempt and MAIN cannot self-approve.
The packet needs exact packages/names/purposes/signatures plus the diagnostic budget in CPU0067.
Reviewed planning/dependencies and MAIN acceptance are also necessary before Ready. Extraction
alone needs no root/ADR rule. Preserve old bytes where possible; enumerate changed artifacts and
advance both identity streams. Update only affected root **Core invariants**/backend **CPU backend
routes** retention/version facts before conflicting generation; ADR only for a significant actual
decision. No authority edits or uniform-numerics change occur here.

Before CPU0068/0069, separately approve bounded vector/parallel qualification in root/backend
and Scan **Qualification, performance and migration gates**, **Qualification and mechanism gates**
and length **Runtime preflight and implementation gates**. Runtime **Run lifecycle** remains
synchronous and unchanged; any necessary handoff/lease/API change is a stop, not part of this plan.
No uniform-numerics, ordinary F64 MATMUL or broad vector/parallel grant is proposed.

## Validation and task sequence

[CPU0067](tasks/0067-common-scalar-body-lowering-and-emission.md) and its immediate successor
[CPU0068](tasks/0068-qualified-shared-body-vector-emission.md) contain focused acceptance/commands.
CPU0069 gets a compact executable brief only after its predecessors stabilize and its participant/
abort design receives independent review. All three require clean Class C implementation and
independent targeted code/docs review. Affected CPU tests/Javadoc run once per stable stage;
MAIN owns one named conformance/Engine checkpoint at each capability cutover, reusing evidence.
New shared build/dependency/Runtime work would require a revised scope and repository checkpoint.
Use clean specialized Java and Class-File inspection, ordinary-unrolled forward/actual-AD oracles,
cache/stale-identity checks and each brief's newly proposed one-JVM/120-second diagnostic, not an
old five-fork campaign or universal speed gate. MAIN accepts the finite protocol before Ready.

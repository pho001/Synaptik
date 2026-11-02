# General Scan with one symbolic per-run batch

## Purpose and authority

Record the selected extension from static general Scan to one named per-run batch B, with fixed
time T, features F and state width H. Compile retains meaning; prepare fixes code/routes/slots and
numeric recipes; each invocation resolves B and owns storage. This is a non-authoritative design
for [Model0039](tasks/0039-general-scan-symbolic-batch-design.md), not an authoritative contract or capability.
Model0039 baseline: Complete — DESIGN DELIVERY ONLY; independent frozen review APPROVE, zero open
P1/P2, MAIN accepted 2026-10-24. [Model0040 amendment](tasks/0040-same-geometry-reuse-design-amendment.md):
Complete DESIGN AMENDMENT ONLY; independent APPROVE, zero open P1/P2, MAIN accepted 2026-10-25.
The user subsequently approved the original symbolic-B direction plus geometry reuse and explicitly
authorized [Model0041](tasks/0041-symbolic-scan-authority-cutover.md), now Complete for the
authority-only cutover. PlanningAuthorized=true; AuthorityAuthorized=true; ImplementationAuthorized=false.
Planning frontier: None. Authority frontier: None. Implementation frontier: None.
Accepted semantics live only in the
[Scan owner](../../../architecture/contracts/recurrent-scan.md#accepted-symbolic-batch-scan) and
[memo owner](../../../architecture/contracts/runtime-prepare-engine.md#accepted-engine-geometry-memo).
The separately admitted [Model0042](tasks/0042-symbolic-scan-source-foundation.md) source foundation
is Complete after independent APPROVE and MAIN acceptance 2026-10-25; it grants no downstream capability.

## Current truth and accepted future admission

Ordinary FLOAT32 CPU dynamic-B forward/actual first-order backward is implemented by Prepare0013.
Model now constructs static and bounded one-name symbolic all-valid Scan source with no API/header
change, preserving canonical static and unresolved symbolic layouts. Compiler0014 inference/
actual first-order AD/scoped binding is independently reviewed and MAIN accepted 2026-10-25 after
compatibility validation. Prepare still rejects dynamic Scan descriptors; CPU additionally bakes staged extents, strides,
singleton/empty decisions and banks. Public cold capture/reconstruction remain static-only.
Changing only an invocation binder is insufficient. Existing all-valid FLOAT32/FLOAT64 execution,
dense actual-read views, static length source/IR/AD/projection and fixed scans remain unchanged.

Implemented and accepted source choice: extend the existing all-valid `Scan.apply`/`ScanBody` shape admission,
without a new overload/header mode, to fixed extents or one named B axis per value. Retain Model's
known PURE-operation and applicable dtype inventory under each family's existing constructor
guards; do not impose CPU's FLOAT32 restriction on Model. Static source validity is not weakened.
Compiler's separate proposed symbolic first-order domain retains its bounded FLOAT32/FLOAT64
algebra and exact primal-binding protocol; unsupported rules and higher-order still reject.
Initial CPU execution qualifies only homogeneous FLOAT32 and the complete actual generated closure.
Neither source nor AD acceptance grants symbolic FLOAT64 CPU execution or all-PURE execution.

## Concrete qualification example

Conceptual future scenario, not runnable or currently supported symbolic-Scan Java. Use fixed
T=3, F=4, H=3; X[T,B,F], h0[B,H], Wx[F,H], Wh[H,H], bias[H]. Storage-free formals describe:

```text
hNext = tanh(xStep @ Wx + h @ Wh + bias)
nextCarry = hNext; emission = hNext
finalCarry[B,H] is the last traversal state; Y[T,B,H] stores original-time emissions.
```

With zero h0/Wx/Wh and bias=atanh(0.25), each step is approximately 0.25 after FLOAT32 rounding.
Numerical qualification instead uses nonuniform signed X, nonzero h0/weights/bias and seeds,
against independent ordinary Tensor unrolling and a specialized clean-Java reference. Test both
traversals and joint final-carry/emission cotangents; never substitute a handwritten reverse scan
for actual Compiler-generated first-order AD. Also use L=mean(finalCarry) to qualify count geometry.

For 64→40→1→64, reuse one compiled graph/preparation/generated artifact set with fresh run values.
For each B, X has 12B elements, finalCarry 3B and Y 9B; their FLOAT32 payloads are 48B, 12B and 36B
bytes respectively. B=40 therefore gives 1,920/480/1,440 bytes. dX/dh0 have actual row shapes;
dWx[4,3], dWh[3,3], dbias[3] remain fixed. Check real values, descriptors, resources and stable
identities, not just these formulas or all-one frontend smoke. Changed F/H or inconsistent B rejects.

## Named axes, reverse closure and empty domains

B is identified by the exact declared named dimension, never by an equal numeric size. Fixed
parameter axes remain shared even when F=H=B. Transpose moves the named axis from `[B,F]` to
`[F,B]`; SUM over it removes row dependence. Shared `[F,H]` history is `[T,F,H]`, not `[T,B,F,H]`.
Do not reuse length-mode `rowCarryCount` to tag an all-valid symbolic scan: it adds a lengths actual.
Original recurrent row carries/steps and generated mixed row/shared accumulator positions remain
distinct. Tied actual positions retain contributions; physical read aliases do not merge roles.

The complete qualified closure includes pointwise ADD/SUB/MUL/DIV, scalar forms, NEG/TANH/SIGMOID,
rank-two MATMUL/transpose, SUM/sumToShape, explicit constants and required expand/rank edits/
contiguous/same-dtype casts under existing guards. Preserve exact saved activation outputs, not
activation replay. Reverse uses the opposite traversal, original-time histories and parameter-
shaped additive carry sums. MATMUL must bind forward M/K/N=B/F/H, dX=B/H/F and dWx=F/B/H;
the last contracts K=B, with actual transposed strides. Wh's adjoint likewise contracts B.

Bias unbroadcast must really reduce B, including B=0/1; equal sizes cannot erase axis provenance.
Keep `ReductionGradientRules.count = oneLike(input).sum(...)`, restored/expanded divide and actual
B*H count for L, never a denominator sampled at B=64. Qualify the needed rank-three outer
seed/zero/constant/pointwise/layout plumbing for `[T,B,*]` through existing ordinary owners;
this is an explicit bounded extension beyond ordinary rank-two admission, not arbitrary rank support.
Local feature-only rank edits may introduce a singleton axis; reject unproved reshaping of B.

T=0 is a **separate fixed-T compiled/prepared plan**, not a changing-T run of the T=3 plan.
No body or hoisted body operation executes; final carry preserves initial represented bits in a
distinct canonical output, dh0 equals the final-carry seed, stacked emissions/residuals/dX are
empty and invariant-gradient sums are typed zero. Structural checks still inspect the full body.
B=0 on a fixed-T plan empties row carries/sequences/emissions/histories but does not skip the
whole Scan/run: shared carries/emissions, parameter-shaped empty-contraction/SUM outputs and
initializers remain nonempty where declared. K=0 MATMUL and empty SUM publish typed zero without
reading nonexistent row cells. L=mean(empty finalCarry) is NaN; empty dX/dh0 do not license skipping
nonempty parameter-gradient outputs. Test B=0 and T=0 independently and together, including shared
work beside empty row work. Unsupported bodies reject even on empty domains.

## Existing-owner change map

Paths below are relative to `modules/`; packages start `io.github.pho001.synaptik.`. These are
future extensions of inspected owners, not permission to edit source in Model0041.

| Owner / inspected source | Necessary bounded change |
|---|---|
| [Model ScanBody](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/ScanBody.java) / [Scan](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Scan.java) | Replace `requireStatic`-only admission and numeric canonical-formal comparison for the symbolic branch; extend `validateBindings`, `validatePort`, `portDescriptor`. Dynamic templates have unresolved numeric layouts, not fabricated resolved layouts; static formals/outputs retain exact canonical checks. Preserve storage-free containment, siblings, gradient OR and canonical residual wrappers. |
| [ScanAttrs](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/scan/ScanAttrs.java) / [CompiledScanBody](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/graph/CompiledScanBody.java) | Keep `timeCount` fixed and all-valid header defaults/counts. Retain `graph`, `nextCarry`, `emissions`, `residuals`, `demandedOutputIndices`; no B value, new component or second body representation is selected. |
| [ScanBodyInference](../../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ScanBodyInference.java) / [ScanGradientRules](../../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ScanGradientRules.java) | Extend `infer`/`preflight` static guards and formal creation to named-B templates, independently check symbolic actual/formal/result obligations. Reuse `capture` → `GraphCapture.captureBody`, exact primal bindings, derivative constants and per-invariant accumulator algebra. There is no current `ScopedScanCapture` type. Preserve scope/demand after rewrites and reject residual roots/higher-order before derivative allocation. |
| [BatchBindingPlan](../../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/BatchBindingPlan.java) / [StaticResultLogicalLayoutClosure](../../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/StaticResultLogicalLayoutClosure.java) | `prepare` currently rejects any dynamic output when `node.outputs().size()!=1`. Add an exact body-bearing SCAN case that preindexes canonical actual layouts for **every** final/emission/residual port, then retains rejection for unknown symbolic multi-output kinds. Keep `bind`/`Binding.descriptor(int)` lazy and occurrence-correct. Do not make static `closeValue` fabricate dynamic layouts; symbolic layout recipes stay in the metadata/numeric owners. |
| [PrepareContext](../../../../modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/analysis/PrepareContext.java) / [GraphPreparation](../../../../modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/GraphPreparation.java) | Extend `ordinaryBatchShape`'s rank-two gate only for qualified scan boundaries/closure; extend `validateScan` local static/canonical matching to symbolic templates. Preserve complete-graph demand checks before analyzers, local/outer ID separation and `requiresBatchSize()`, even with constant bytes. No CPU geometry interpretation. |
| [CpuGeneralScanIr](../../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuGeneralScanIr.java) / [Lowering](../../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuGeneralScanLowering.java) | Extend existing `View`/`Stages`/`Cut` and `Facts` storage for immutable B-coefficient access/frame/cut/carry-offset recipes alongside retained static fields. Reuse `analyze`, `shapedResources` and stage lifetimes; no representative fake B, post-bind re-lowering or new semantic IR. Exact component signatures need the later CPU brief/approval, not speculative new records here. |
| [CpuCapabilityProvider](../../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java) / [CpuPartitionLowering](../../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java) / [CpuPartitionPreparer](../../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java) | `CpuCapabilityProvider.supports(query)` returns `supportsOrdinaryBatch` for symbolic descriptors before its SCAN branch; preparer `requiresBatchSize()` likewise enters `analyzeOrdinaryBatch`. Add explicit Scan dispatch to existing whole-body analysis at both boundaries, plus bounded rank-three closure support. Adapt `analyzeGeneralScan` fixed-byte declarations/workspace-presence test to fixed or run-size declarations; finalization emits only after slots. |
| [CpuGeneralScanPreparedExecutable](../../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuGeneralScanPreparedExecutable.java) / [Emitter](../../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneralScanEmitter.java) | Replace symbolic use of fixed `spans`, `elements`, `scratchBytes` with once-bound actual numeric facts; add existing state-aware representation-hook overrides and extend `bindCompatible`/run-owned `Invocation.geometry`. Extend `Shaped.loop/load/bytes`, MATMUL and reduction emitters to prepared dynamic axis/broadcast roles, with B/strides loaded outside hot contributor loops. |
| [Engine PreparedExecution](../../../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/PreparedExecution.java) / [AdvancedEngine](../../../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/AdvancedEngine.java) | Preindex scoped local domains in `batchNumericChecks`/`validateBatchGeometry`, not only outer graph values. Reuse `runOrdinaryBatchOpen`, exact-prefix borrowing and actual publication descriptors. No per-run body traversal, inference or synthetic Tensor construction. |

## Numeric resources and minimal type decision

Proposed new/moved production types: **zero**. Existing owners are candidates, not a proof that
every resource is affine. Raw payloads and admitted strides are fixed or affine in one B;
alignment/padding, conditional 0/48-byte aggregate scratch and B=0/1 cases can require piecewise
checked numeric sizes/offsets. Existing CPU owners and `PreparedMemoryPlan.ByteSize` can express
those numeric recipes without unknown maxima or per-B route/slot choices. Reuse existing numeric
`CpuAccessPlan.BatchBinding.bind(long)`, `CpuRepresentationRecipes.BatchByteSize.byteSize(long)`,
`CpuMatmulLowering.BatchGeometry.bind(long)` and `CpuAggregateLowering.BatchGeometry.bind(long)`;
the last describes ordinary outer reductions, not automatic reuse of its entry ABI inside Scan.
Reuse `PreparedMemoryPlan.ByteSize` and `RunMemoryGeometry.resolve(plan, OptionalLong)` unchanged.
CPU may extend fields/methods on its existing Scan records; if a new named type or changed Runtime
handoff proves necessary, stop for an exact package/name/purpose/signature and user approval first.
Zero is a proposal, not a claim that all implementation branches have been proved.

Preserve completed Prepare0013's nine additions: `compiler.BatchBindingPlan` and `.Binding`;
`runtime.memory.PreparedMemoryPlan.ByteSize` and `.FixedByteSize`; `runtime.run.RunMemoryGeometry`;
`backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize`;
`backend.cpu.internal.ir.CpuAccessPlan.BatchBinding`;
`backend.cpu.internal.lowering.CpuMatmulLowering.BatchGeometry` and
`backend.cpu.internal.lowering.CpuAggregateLowering.BatchGeometry`. Nine is historical inventory,
not a blanket future-count limit. No new public method/facade or Prepare/Runtime type is selected.

Declare exact actual-size recipes before slots: X time stride B*F, row frame B*H, demanded row
history T*B*localFeatures; demanded shared history T*sharedElements. Parameter gradients/shared
reverse accumulators never acquire B banks. Preserve dead-residual zero bytes/stores, repeated
outer ports, saved-site aliases and inclusive cut lifetimes. Fixed topology chooses demanded-frame
backing or disjoint old/new carry banks; offsets use checked aligned prefix recipes, including
conditional sizes where needed, never per-B route, slot or storage-topology selection.
Reuse a dead cut only with lifetime and size dominance proved for all admitted B, otherwise retain
disjoint exact regions. Do not discover/remove workspaces at B=0; a structurally needed zero-byte
workspace still has its fixed slot/creator. Each invocation gets fresh storage, including zero bytes.

## Admission, generation and identity

Before borrowing/creation/writes, validate every declared outer/local domain and latent port,
including unused or scalar-backed ones: ranks, fixed T/F/H, B identity/agreement, dtype/flags,
constraints, canonical actual reads/dense read views, extents, trailing products, strides, offsets,
referenced spans and byte counts. Counts/extents/element strides/spans fit CPU int; checked bytes,
aligned offsets and aggregate representation-multiplied totals fit long. B=0/T=0 cannot hide an
overflowing trailing domain. Build scoped numeric checks once during preparation; Runtime gets
only authenticated numeric geometry. Keep storage/lifetime/alignment/overlap checks and reverse
rollback/retry/concurrent-run isolation; no currentB/shared arrays/ThreadLocal or per-step allocation.

Generate the optimal fixed-F/H clean-Java algorithm with invocation B: direct time/row/feature
loops, cursor increments and typed MATMUL contributors. Hoist B/strides/frame offsets once per
invocation or necessary row setup, not per-element recipe lookup/dispatch. Keep structural
broadcast axes prepared and B=0/1 entry guards correct. Scan's existing typed producing-site SUM
and ordinary outer exact five-limb SUM/MEAN are distinct qualified algorithms; do not silently
replace one with the other. Required sumToShape singleton/raw-copy versus empty-reduction behavior
must remain operation-correct. Class-File evidence compares algorithm, dataflow and hidden overhead.

Use a distinct symbolic-Scan compatibility branch in both specialization streams, binding all
baked dtype/body/role/demand/mapping/recipe facts; actual B/bytes/slots/addresses stay excluded.
Preserve current static Scan/fixed/ordinary keys and generated bytes. Exact new discriminant and
existing-record component layout are later CPU decisions requiring an approved scoped brief.
No key based on changed diagnostic text; no per-B artifact/full-prepare cache or B_max allocation.

## Accepted future same-geometry reuse at Engine entry

A memo retains pure geometry results, not an invocation. Prepare preindexes checks/recipes/publications.
`BatchBindingPlan.bind` still checks all descriptors/private obligations and returns a fresh `Binding`
retaining supplied descriptors every call; an Engine memo cannot skip its work. Later Compiler-owner
refactoring/signature/protocol changes need exact approval; no new API or production type is selected.

Select one future replaceable entry owned by the exact Engine `PreparedExecution`, empty initially.
Key: exact compiled/binding plan, inward preparation/memory-plan references, fixed recipes/publications
and explicit B/mode (even `[B,0]`); equal bytes/B or foreign equal plans/shapes never authenticate it.
Retain checked B-dependent domain validity, sizes/strides/offsets, aligned aggregate bounds and
template-derived non-input publication metadata; never Tensor identity, supplied descriptors,
`Binding`, storage, addresses, leases, mutable arrays or results.
One graph-sized snapshot, no B-sized payload/map; old snapshots survive only via live calls/results.
Bound candidates by active admissions; no global/ThreadLocal cache, pool or per-B plan cache.

Every call keeps Engine/owner/closed-prepared precedence, snapshots/counts/nulls and exact-ID or
positional ordering, then binds every input occurrence, including unused/repeated roles. Check all
B occurrences, rank/fixed axes, dtype/requiresGrad, layout presence, canonical strides/offsets/view
policy before a hit; first-input B equality is not admission. Avoid only covered Engine pure numeric/
size checks and template-only publication reconstruction. Snapshot each current storage association
once; revalidate presence/type/liveness/thread access/capacity, borrow exact current prefixes and
retain physical type/alignment/access/overlap admission before execution. All other checks remain.

Input-backed publications use the fresh binding's current supplied descriptor, never a prior caller's;
same-B admitted dense read views can differ in `isView`. Cache only template-derived non-input facts.
Preserve order, tensorId/role/derivativeOrder/targetIndex, aliases and generated gradient flags.
Runtime gets only exact-plan numeric geometry, never the memo/symbols/semantic metadata. Each run
gets fresh resources (even zero bytes), mutable packed/access/address/carry arrays and outputs;
copy facts into isolated mutable invocation arrays, never share them.

Publish a complete immutable candidate only after pre-borrow descriptor/numeric/storage admission.
Atomic snapshot reads/replacement need future bounded publication-vs-close coordination through
existing lifecycle; exact mechanism gated, no execution lock/second lease. Mixed-B races may compute
duplicate candidates; either valid replacement may win, but each call keeps its selected snapshot.
Failed admission publishes nothing and preserves the good entry; later allocation/bind/execution failure
may retain valid metadata only, not resources/results/failures; reverse cleanup is unchanged.
Retry revalidates inputs/storage and creates fresh resources/leases/results. Close clears and prevents
late republishing; admitted calls keep local facts, while Runtime's unique lease decides close races.
After existing inward-delegate admission, a concurrent close suppresses/clears memo publication,
not the run through a new late outward-close rejection. No new lock/API mechanism is selected.

Model0041's completed authority cutover clarifies [resource ownership](../../../architecture/contracts/runtime-prepare-engine.md#accepted-symbolic-batch-resource-contract),
[run lifecycle](../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle) and
[prepared semantics](../../../architecture/contracts/runtime-prepare-engine.md#public-reusable-prepared-execution):
shareable never-mutated facts/private backing are distinct from mutable run arrays, and the slot
cannot change prepared meaning. Independent APPROVE; MAIN accepted 2026-10-25.
`RunMemoryGeometryTest` shares immutable geometry across fresh states, not authority for Engine
memoization. Root lifecycle/index and CPU ownership are coordinated in that cutover; affected
Javadocs remain a future source obligation. Existing `RunMemoryGeometry.resolve(plan, OptionalLong)`
is not a hit API. This design record enacts no authority by itself.
Memo lookup/publication/copying stays at cold entry, never time/row/contributor loops;
generated code consumes direct numeric facts. No zero-overhead claim.

Conceptual serial witness: one preparation, fresh values, B=64→64→40→40→64: miss/hit/miss/hit/miss;
unchanged code/routes/slots, actual-sized resources. Use bounded test-local counters/probes for
avoided Engine work, not binder work or a benchmark campaign. Test invalid same-B/fixed-axis/type/
flag/layout/storage, current input-backed views, B=0/1, constant-byte `[B,0]`, overflow, foreign
preparations, failure/retry, close/concurrent mixed-B results and generated loops without memo access.

## Ordered gates and approval choices

1. Model0039 design freeze → independent read-only APPROVE (zero open P1/P2) → MAIN accepted
   DESIGN DELIVERY ONLY on 2026-10-24; this gate is Complete, not implementation authorization.
   Model0040 reuse amendment Complete DESIGN ONLY; independent APPROVE, MAIN accepted 2026-10-25;
   its ownership clarification is included in completed Model0041 authority, not implemented reuse.
2. User detailed semantic approval and authority-only admission were supplied on 2026-10-25.
   Model0041 is Complete: root/index, sole Scan semantic owner, coordinated Compiler/Runtime/
   CPU boundaries, ADR0051, focused explanation and glossary are enacted. Independent frozen
   APPROVE, zero open P1/P2; MAIN accepted 2026-10-25. Source gates require separate admission.
3. Model0042 is separately admitted, implemented and MAIN accepted 2026-10-25, with explicit
   old-consumer rejection. Then separately admit one Compiler inference/actual-first-order-AD/binding-layout capability; then one
   coordinated Prepare/CPU/Engine numeric forward+backward vertical. Exact packets/classes/files
   are finalized at each frontier; no mechanical task cascade or Ready downstream implementation brief now.
4. Later, separately qualify length skipping and its symbolic-B combination, then selected
   NN/Training one-state reuse. Cold/copy/wire, variable T, wider execution dtypes and native/mixed
   remain independent gates, never inferred from this design or made prerequisites by speculation.

All future gates are serial (Parallel group None), conflict on Scan/resource/identity semantics,
and depend on their immediately preceding accepted gate plus the retained baseline. MAIN owns
shared-document integration and revalidates the accepted dirty base before any launch. Workers
run focused plus affected-module validation once; independent reviewers reuse it; MAIN runs one
recorded boundary join. Future tests cover both seeds/traversals, nonuniform actual AD versus
unrolling, exact demanded/dead/shared bytes, B64→40→1→64, separate T0, B0/B1, overflow/preborrow
failure, aliases, cleanup/retry/concurrency and retained static/length rejection paths. A tiny
warmed matched-work generated/direct probe separates cold setup/admission/kernel costs; no long
benchmark campaign, universal speed/parity claim or new certification machinery.

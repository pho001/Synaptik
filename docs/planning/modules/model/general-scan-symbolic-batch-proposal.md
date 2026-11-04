# [General Scan](../../../glossary.md#general-differentiable-scan) with one symbolic per-[run](../../../glossary.md#term-run) [batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters)

## Purpose and authority

Record the selected extension from static [general Scan](../../../glossary.md#general-differentiable-scan) to one named per-[run](../../../glossary.md#term-run) [batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters) B, with fixed
time T, [features](../../../glossary.md#term-software-feature) F and state width H. [Compile](../../../glossary.md#compile) retains meaning; [prepare](../../../glossary.md#prepare) fixes code/routes/slots and
numeric recipes; each invocation resolves B and owns storage. This is a non-authoritative design
for [Model0039](tasks/0039-general-scan-symbolic-batch-design.md), not an authoritative contract or capability.
Model0039 baseline: Complete — DESIGN DELIVERY ONLY; independent frozen review APPROVE, zero open
P1/P2, MAIN accepted 2026-10-24. [Model0040 amendment](tasks/0040-same-geometry-reuse-design-amendment.md):
Complete DESIGN AMENDMENT ONLY; independent APPROVE, zero open P1/P2, MAIN accepted 2026-10-25.
The user subsequently approved the original symbolic-B direction plus geometry reuse and explicitly
authorized [Model0041](tasks/0041-symbolic-scan-authority-cutover.md), now Complete for the
authority-only cutover. PlanningAuthorized=true; AuthorityAuthorized=true; ImplementationAuthorized=false.
[Planning](../../../glossary.md#term-planning) frontier: None. Authority frontier: None. Implementation frontier: None.
Accepted semantics live only in the
[Scan owner](../../../architecture/contracts/recurrent-scan.md#accepted-symbolic-batch-scan)<span data-glossary-navigation="true"> ([Scan glossary](../../../glossary.md#term-cumulative-scan); [owner glossary](../../../glossary.md#term-owner))</span> and
[memo owner](../../../architecture/contracts/runtime-prepare-engine.md#accepted-engine-geometry-memo)<span data-glossary-navigation="true"> ([memo glossary](../../../glossary.md#cache-and-memoization))</span>.
The separately admitted [Model0042](tasks/0042-symbolic-scan-source-foundation.md) source foundation
is Complete after independent APPROVE and MAIN acceptance 2026-10-25; it grants no downstream capability.

## Current truth and accepted future [admission](../../../glossary.md#term-admission)

Ordinary [FLOAT32](../../../glossary.md#term-float32) [CPU](../../../glossary.md#term-cpu) dynamic-B forward/actual first-order backward is implemented by Prepare0013.
[Model](../../../glossary.md#term-model) now constructs static and bounded one-name symbolic all-valid [Scan](../../../glossary.md#term-cumulative-scan) source with no [API](../../../glossary.md#term-application-programming-interface-api)/header
change, preserving [canonical](../../../glossary.md#term-canonical-form) static and unresolved symbolic [layouts](../../../glossary.md#layout). Compiler0014 [inference](../../../glossary.md#term-inference)/
actual first-order [AD](../../../glossary.md#autograd)/scoped [binding](../../../glossary.md#term-binding) is independently reviewed and MAIN accepted 2026-10-25 after
compatibility [validation](../../../glossary.md#term-validation). [Prepare](../../../glossary.md#prepare) still rejects dynamic Scan [descriptors](../../../glossary.md#term-descriptor); CPU additionally bakes staged extents, [strides](../../../glossary.md#term-stride),
[singleton](../../../glossary.md#term-singleton)/empty decisions and banks. Public cold [capture](../../../glossary.md#term-capture-meanings)/reconstruction remain static-only.
Changing only an invocation binder is insufficient. Existing all-valid FLOAT32/[FLOAT64](../../../glossary.md#term-float64) execution,
dense actual-read [views](../../../glossary.md#term-view-meanings), static length source/[IR](../../../glossary.md#term-intermediate-representation-ir)/AD/projection and fixed scans remain unchanged.

Implemented and accepted source choice: extend the existing all-valid `Scan.apply`/`ScanBody` [shape](../../../glossary.md#shape) admission,
without a new overload/header mode, to fixed extents or one named B [axis](../../../glossary.md#term-axis) per value. Retain Model's
known [PURE](../../../glossary.md#term-pure)-[operation](../../../glossary.md#operation) and applicable [dtype](../../../glossary.md#data-type--datatype) inventory under each family's existing constructor
guards; do not impose CPU's FLOAT32 restriction on Model. Static source validity is not weakened.
[Compiler](../../../glossary.md#term-compiler)'s separate proposed symbolic first-order domain retains its bounded FLOAT32/FLOAT64
algebra and exact [primal](../../../glossary.md#term-primal)-binding protocol; unsupported rules and higher-order still reject.
Initial CPU execution qualifies only [homogeneous](../../../glossary.md#term-homogeneous) FLOAT32 and the complete actual generated closure.
Neither source nor AD acceptance grants symbolic FLOAT64 CPU execution or all-PURE execution.

## Concrete [qualification](../../../glossary.md#term-qualification) example

Conceptual future scenario, not runnable or currently supported symbolic-[Scan](../../../glossary.md#term-cumulative-scan) Java. Use fixed
T=3, F=4, H=3; X[T,B,F], h0[B,H], Wx[F,H], Wh[H,H], [bias](../../../glossary.md#term-bias)[H]. Storage-free formals describe:

```text
hNext = tanh(xStep @ Wx + h @ Wh + bias)
nextCarry = hNext; emission = hNext
finalCarry[B,H] is the last traversal state; Y[T,B,H] stores original-time emissions.
```

With zero h0/Wx/Wh and bias=atanh(0.25), each step is approximately 0.25 after [FLOAT32](../../../glossary.md#term-float32) [rounding](../../../glossary.md#term-rounding).
Numerical qualification instead uses nonuniform signed X, nonzero h0/[weights](../../../glossary.md#term-weight)/bias and [seeds](../../../glossary.md#commit-release-distribution-log-and-seed-meanings),
against independent ordinary [Tensor](../../../glossary.md#tensor) [unrolling](../../../glossary.md#term-unrolling) and a specialized clean-Java reference. Test both
traversals and joint final-[carry](../../../glossary.md#term-carry)/emission cotangents; never substitute a handwritten reverse scan
for actual [Compiler](../../../glossary.md#term-compiler)-generated first-order [AD](../../../glossary.md#autograd). Also use L=[mean](../../../glossary.md#term-mean)(finalCarry) to qualify count geometry.

For 64→40→1→64, reuse one [compiled graph](../../../glossary.md#compiled-graph--compiledgraphmodel)/preparation/generated [artifact](../../../glossary.md#term-artifact) set with fresh [run](../../../glossary.md#term-run) values.
For each B, X has 12B [elements](../../../glossary.md#term-element), finalCarry 3B and Y 9B; their FLOAT32 payloads are 48B, 12B and 36B
bytes respectively. B=40 therefore gives 1,920/480/1,440 bytes. dX/dh0 have actual row [shapes](../../../glossary.md#shape);
dWx[4,3], dWh[3,3], dbias[3] remain fixed. Check real values, [descriptors](../../../glossary.md#term-descriptor), resources and stable
[identities](../../../glossary.md#term-identity-meanings), not just these formulas or all-one frontend [smoke](../../../glossary.md#term-smoke-test). Changed F/H or inconsistent B rejects.

## Named [axes](../../../glossary.md#term-axis), reverse closure and empty domains

B is identified by the exact declared named [dimension](../../../glossary.md#dimension), never by an equal numeric size. Fixed
[parameter](../../../glossary.md#term-parameter) axes remain shared even when F=H=B. [Transpose](../../../glossary.md#term-transpose) moves the named axis from `[B,F]` to
`[F,B]`; SUM over it removes row dependence. Shared `[F,H]` history is `[T,F,H]`, not `[T,B,F,H]`.
Do not reuse length-mode `rowCarryCount` to tag an all-valid [symbolic scan](../../../glossary.md#symbolic-scan): it adds a lengths actual.
Original recurrent row carries/steps and generated mixed row/shared [accumulator](../../../glossary.md#accumulator-type) positions remain
distinct. Tied actual positions retain contributions; physical read [aliases](../../../glossary.md#aliasing-and-arena) do not merge roles.

The complete qualified closure includes [pointwise](../../../glossary.md#term-pointwise) ADD/SUB/MUL/DIV, [scalar](../../../glossary.md#term-scalar) forms, NEG/[TANH](../../../glossary.md#term-tanh)/[SIGMOID](../../../glossary.md#term-sigmoid),
[rank-two](../../../glossary.md#axis-rank-scalar-vector-and-matrix) [MATMUL](../../../glossary.md#matrix-multiplication--matmul)/transpose, SUM/sumToShape, explicit [constants](../../../glossary.md#term-constant) and required expand/[rank](../../../glossary.md#term-rank) edits/
[contiguous](../../../glossary.md#term-contiguous)/same-[dtype](../../../glossary.md#data-type--datatype) casts under existing guards. Preserve exact saved [activation](../../../glossary.md#term-activation) outputs, not
activation replay. Reverse uses the opposite traversal, original-time histories and parameter-
shaped additive [carry](../../../glossary.md#term-carry) sums. MATMUL must bind forward M/K/N=B/F/H, dX=B/H/F and dWx=F/B/H;
the last contracts K=B, with actual transposed [strides](../../../glossary.md#term-stride). Wh's [adjoint](../../../glossary.md#term-adjoint) likewise contracts B.

[Bias](../../../glossary.md#term-bias) unbroadcast must really reduce B, including B=0/1; equal sizes cannot erase axis [provenance](../../../glossary.md#provenance).
Keep `ReductionGradientRules.count = oneLike(input).sum(...)`, restored/expanded divide and actual
B*H count for L, never a denominator sampled at B=64. Qualify the needed rank-three outer
[seed](../../../glossary.md#term-seed)/zero/constant/pointwise/[layout](../../../glossary.md#layout) plumbing for `[T,B,*]` through existing ordinary [owners](../../../glossary.md#term-owner);
this is an explicit bounded extension beyond ordinary rank-two [admission](../../../glossary.md#term-admission), not arbitrary rank support.
Local [feature](../../../glossary.md#term-feature)-only rank edits may introduce a [singleton](../../../glossary.md#term-singleton) axis; reject unproved reshaping of B.

T=0 is a **separate fixed-T compiled/prepared plan**, not a changing-T [run](../../../glossary.md#term-run) of the T=3 plan.
No body or hoisted body [operation](../../../glossary.md#operation) executes; final carry preserves initial [represented bits](../../../glossary.md#term-represented-bits) in a
distinct [canonical](../../../glossary.md#term-canonical-form) output, dh0 equals the final-carry seed, stacked emissions/residuals/dX are
empty and [invariant](../../../glossary.md#term-invariant)-[gradient](../../../glossary.md#term-gradient) sums are typed zero. Structural checks still inspect the full body.
B=0 on a fixed-T plan empties row carries/sequences/emissions/histories but does not skip the
whole [Scan](../../../glossary.md#term-cumulative-scan)/run: shared carries/emissions, parameter-shaped empty-[contraction](../../../glossary.md#contraction-dimension)/SUM outputs and
initializers remain nonempty where declared. K=0 MATMUL and empty SUM publish typed zero without
reading nonexistent row cells. L=[mean](../../../glossary.md#term-mean)(empty finalCarry) is [NaN](../../../glossary.md#term-nan); empty dX/dh0 do not license skipping
nonempty parameter-gradient outputs. Test B=0 and T=0 independently and together, including shared
work beside empty row work. Unsupported bodies reject even on empty domains.

## Existing-[owner](../../../glossary.md#term-owner) change map

Paths below are relative to `modules/`; packages start `io.github.pho001.synaptik.`. These are
future extensions of inspected owners, not permission to edit source in Model0041.

| Owner / inspected source | Necessary bounded change |
|---|---|
| [Model ScanBody](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/ScanBody.java)<span data-glossary-navigation="true"> ([Model glossary](../../../glossary.md#term-model))</span> / [Scan](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Scan.java) | Replace `requireStatic`-only [admission](../../../glossary.md#term-admission) and numeric [canonical](../../../glossary.md#term-canonical-form)-formal comparison for the symbolic branch; extend `validateBindings`, `validatePort`, `portDescriptor`. Dynamic templates have unresolved numeric [layouts](../../../glossary.md#layout), not fabricated resolved layouts; static formals/outputs retain exact canonical checks. Preserve storage-free containment, siblings, [gradient](../../../glossary.md#term-gradient) OR and canonical [residual](../../../glossary.md#scan-residual-port-and-history-demand) wrappers. |
| [ScanAttrs](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/scan/ScanAttrs.java) / [CompiledScanBody](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/graph/CompiledScanBody.java) | Keep `timeCount` fixed and all-valid header defaults/counts. Retain [`graph`](../../../glossary.md#graph), `nextCarry`, `emissions`, `residuals`, `demandedOutputIndices`; no B value, new component or second body representation is selected. |
| [ScanBodyInference](../../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ScanBodyInference.java) / [ScanGradientRules](../../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ScanGradientRules.java) | Extend `infer`/[`preflight`](../../../glossary.md#term-preflight) static guards and formal creation to named-B templates, independently check symbolic actual/formal/result obligations. Reuse [`capture`](../../../glossary.md#term-capture-meanings) → `GraphCapture.captureBody`, exact [primal](../../../glossary.md#term-primal) [bindings](../../../glossary.md#term-binding), [derivative](../../../glossary.md#term-derivative) [constants](../../../glossary.md#term-constant) and per-[invariant](../../../glossary.md#term-invariant) [accumulator](../../../glossary.md#accumulator-type) algebra. There is no current `ScopedScanCapture` type. Preserve [scope](../../../glossary.md#term-scope)/demand after rewrites and reject residual roots/higher-order before derivative allocation. |
| [BatchBindingPlan](../../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/BatchBindingPlan.java) / [StaticResultLogicalLayoutClosure](../../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/StaticResultLogicalLayoutClosure.java) | [`prepare`](../../../glossary.md#prepare) currently rejects any dynamic output when `node.outputs().size()!=1`. Add an exact body-bearing [SCAN](../../../glossary.md#term-cumulative-scan) case that preindexes canonical actual layouts for **every** final/emission/[residual port](../../../glossary.md#scan-residual-port-and-history-demand), then retains rejection for unknown symbolic multi-output kinds. Keep `bind`/`Binding.descriptor(int)` [lazy](../../../glossary.md#term-lazy-evaluation) and [occurrence](../../../glossary.md#term-occurrence)-correct. Do not make static `closeValue` fabricate dynamic layouts; symbolic layout recipes stay in the [metadata](../../../glossary.md#term-metadata)/numeric owners. |
| [PrepareContext](../../../../modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/analysis/PrepareContext.java)<span data-glossary-navigation="true"> ([PrepareContext glossary](../../../glossary.md#prepare-context-preparecontext))</span> / [GraphPreparation](../../../../modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/GraphPreparation.java)<span data-glossary-navigation="true"> ([GraphPreparation glossary](../../../glossary.md#graph-preparation--graphpreparation))</span> | Extend `ordinaryBatchShape`'s [rank-two](../../../glossary.md#axis-rank-scalar-vector-and-matrix) gate only for qualified scan boundaries/closure; extend `validateScan` local static/canonical matching to symbolic templates. Preserve complete-graph demand checks before analyzers, local/outer ID separation and `requiresBatchSize()`, even with constant bytes. No [CPU](../../../glossary.md#term-cpu) geometry interpretation. |
| [CpuGeneralScanIr](../../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuGeneralScanIr.java) / [Lowering](../../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuGeneralScanLowering.java) | Extend existing [`View`](../../../glossary.md#term-view-meanings)/`Stages`/`Cut` and `Facts` storage for [immutable](../../../glossary.md#term-immutable) B-coefficient access/frame/cut/[carry](../../../glossary.md#term-carry)-offset recipes alongside retained static fields. Reuse `analyze`, `shapedResources` and stage lifetimes; no representative fake B, post-bind re-[lowering](../../../glossary.md#lowering) or new semantic [IR](../../../glossary.md#term-intermediate-representation-ir). Exact component [signatures](../../../glossary.md#term-signature) need the later CPU brief/approval, not speculative new records here. |
| [CpuCapabilityProvider](../../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java) / [CpuPartitionLowering](../../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java) / [CpuPartitionPreparer](../../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java) | `CpuCapabilityProvider.supports(query)` returns `supportsOrdinaryBatch` for symbolic [descriptors](../../../glossary.md#term-descriptor) before its SCAN branch; preparer `requiresBatchSize()` likewise enters `analyzeOrdinaryBatch`. Add explicit Scan [dispatch](../../../glossary.md#term-dispatch) to existing whole-body analysis at both boundaries, plus bounded [rank](../../../glossary.md#term-rank)-three closure support. Adapt `analyzeGeneralScan` fixed-byte declarations/workspace-presence test to fixed or [run](../../../glossary.md#term-run)-size declarations; finalization emits only after slots. |
| [CpuGeneralScanPreparedExecutable](../../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuGeneralScanPreparedExecutable.java) / [Emitter](../../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneralScanEmitter.java) | Replace symbolic use of fixed `spans`, [`elements`](../../../glossary.md#term-element), `scratchBytes` with once-bound actual numeric facts; add existing state-aware representation-hook overrides and extend `bindCompatible`/run-owned `Invocation.geometry`. Extend `Shaped.loop/load/bytes`, [MATMUL](../../../glossary.md#matrix-multiplication--matmul) and reduction emitters to prepared dynamic [axis](../../../glossary.md#term-axis)/broadcast roles, with B/[strides](../../../glossary.md#term-stride) loaded outside hot contributor loops. |
| [Engine PreparedExecution](../../../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/PreparedExecution.java)<span data-glossary-navigation="true"> ([Engine glossary](../../../glossary.md#term-engine); [PreparedExecution glossary](../../../glossary.md#prepared-execution--preparedexecution))</span> / [AdvancedEngine](../../../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/AdvancedEngine.java) | Preindex scoped local domains in `batchNumericChecks`/`validateBatchGeometry`, not only outer graph values. Reuse `runOrdinaryBatchOpen`, exact-prefix [borrowing](../../../glossary.md#term-borrowing) and actual [publication](../../../glossary.md#term-publication) descriptors. No per-run body traversal, [inference](../../../glossary.md#term-inference) or synthetic [Tensor](../../../glossary.md#tensor) construction. |

## Numeric resources and minimal type decision

Proposed new/moved production types: **zero**. Existing [owners](../../../glossary.md#term-owner) are candidates, not a proof that
every resource is affine. Raw payloads and admitted [strides](../../../glossary.md#term-stride) are fixed or affine in one B;
[alignment](../../../glossary.md#term-alignment)/[padding](../../../glossary.md#padding), conditional 0/48-byte aggregate scratch and B=0/1 cases can require piecewise
checked numeric sizes/offsets. Existing [CPU](../../../glossary.md#term-cpu) owners and `PreparedMemoryPlan.ByteSize` can express
those numeric recipes without unknown maxima or per-B route/slot choices. Reuse existing numeric
`CpuAccessPlan.BatchBinding.bind(long)`, `CpuRepresentationRecipes.BatchByteSize.byteSize(long)`,
`CpuMatmulLowering.BatchGeometry.bind(long)` and `CpuAggregateLowering.BatchGeometry.bind(long)`;
the last describes ordinary outer reductions, not automatic reuse of its entry [ABI](../../../glossary.md#term-application-binary-interface-abi) inside [Scan](../../../glossary.md#term-cumulative-scan).
Reuse `PreparedMemoryPlan.ByteSize` and `RunMemoryGeometry.resolve(plan, OptionalLong)` unchanged.
CPU may extend fields/methods on its existing Scan records; if a new named type or changed [Runtime](../../../glossary.md#term-runtime)
[handoff](../../../glossary.md#term-handoff) proves necessary, stop for an exact package/name/purpose/[signature](../../../glossary.md#term-signature) and user approval first.
Zero is a proposal, not a claim that all implementation branches have been proved.

Preserve completed Prepare0013's nine additions: `compiler.BatchBindingPlan` and `.Binding`;
`runtime.memory.PreparedMemoryPlan.ByteSize` and `.FixedByteSize`; `runtime.run.RunMemoryGeometry`;
`backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize`;
`backend.cpu.internal.ir.CpuAccessPlan.BatchBinding`;
`backend.cpu.internal.lowering.CpuMatmulLowering.BatchGeometry` and
`backend.cpu.internal.lowering.CpuAggregateLowering.BatchGeometry`. Nine is historical inventory,
not a blanket future-count limit. No new public method/[facade](../../../glossary.md#term-facade) or [Prepare](../../../glossary.md#prepare)/Runtime type is selected.

Declare exact actual-size recipes before slots: X time stride B*F, row frame B*H, demanded row
history T*B*localFeatures; demanded shared history T*sharedElements. [Parameter](../../../glossary.md#term-parameter) [gradients](../../../glossary.md#term-gradient)/shared
reverse accumulators never acquire B banks. Preserve dead-[residual](../../../glossary.md#scan-residual-port-and-history-demand) zero bytes/stores, repeated
outer ports, saved-site [aliases](../../../glossary.md#aliasing-and-arena) and inclusive cut lifetimes. Fixed [topology](../../../glossary.md#term-topology) chooses demanded-frame
backing or disjoint old/new [carry](../../../glossary.md#term-carry) banks; offsets use checked aligned prefix recipes, including
conditional sizes where needed, never per-B route, slot or storage-topology selection.
Reuse a dead cut only with [lifetime](../../../glossary.md#term-lifetime) and size dominance proved for all admitted B, otherwise retain
disjoint exact regions. Do not discover/remove workspaces at B=0; a structurally needed zero-byte
workspace still has its fixed slot/creator. Each invocation gets fresh storage, including zero bytes.

## [Admission](../../../glossary.md#term-admission), generation and [identity](../../../glossary.md#term-identity-meanings)

Before [borrowing](../../../glossary.md#term-borrowing)/creation/writes, validate every declared outer/local domain and latent port,
including unused or [scalar](../../../glossary.md#term-scalar)-backed ones: ranks, fixed T/F/H, B identity/agreement, [dtype](../../../glossary.md#data-type--datatype)/flags,
constraints, [canonical](../../../glossary.md#term-canonical-form) actual reads/dense read [views](../../../glossary.md#term-view-meanings), extents, trailing products, [strides](../../../glossary.md#term-stride), offsets,
referenced spans and byte counts. Counts/extents/[element](../../../glossary.md#term-element) strides/spans fit [CPU](../../../glossary.md#term-cpu) int; checked bytes,
aligned offsets and aggregate representation-multiplied totals fit long. B=0/T=0 cannot hide an
overflowing trailing domain. Build scoped numeric checks once during preparation; [Runtime](../../../glossary.md#term-runtime) gets
only authenticated numeric geometry. Keep storage/[lifetime](../../../glossary.md#term-lifetime)/[alignment](../../../glossary.md#term-alignment)/overlap checks and reverse
[rollback](../../../glossary.md#term-rollback)/retry/concurrent-[run](../../../glossary.md#term-run) isolation; no currentB/shared arrays/ThreadLocal or per-step allocation.

Generate the optimal fixed-F/H clean-Java algorithm with invocation B: direct time/row/[feature](../../../glossary.md#term-feature)
loops, cursor increments and typed [MATMUL](../../../glossary.md#matrix-multiplication--matmul) contributors. Hoist B/strides/frame offsets once per
invocation or necessary row setup, not per-element recipe [lookup](../../../glossary.md#term-lookup)/[dispatch](../../../glossary.md#term-dispatch). Keep structural
broadcast [axes](../../../glossary.md#term-axis) prepared and B=0/1 entry guards correct. [Scan](../../../glossary.md#term-cumulative-scan)'s existing typed producing-site SUM
and ordinary outer exact five-[limb](../../../glossary.md#exact-superaccumulator-and-limbs) SUM/[MEAN](../../../glossary.md#term-mean) are distinct qualified algorithms; do not silently
replace one with the other. Required sumToShape [singleton](../../../glossary.md#term-singleton)/raw-copy versus empty-reduction behavior
must remain [operation](../../../glossary.md#operation)-correct. Class-File evidence compares algorithm, [dataflow](../../../glossary.md#term-dataflow) and hidden overhead.

Use a distinct symbolic-Scan compatibility branch in both [specialization](../../../glossary.md#term-specialization) streams, [binding](../../../glossary.md#term-binding) all
baked dtype/body/role/demand/mapping/recipe facts; actual B/bytes/slots/addresses stay excluded.
Preserve current static Scan/fixed/ordinary keys and generated bytes. Exact new discriminant and
existing-record component [layout](../../../glossary.md#layout) are later CPU decisions requiring an approved scoped brief.
No key based on changed diagnostic text; no per-B [artifact](../../../glossary.md#term-artifact)/full-[prepare](../../../glossary.md#prepare) [cache](../../../glossary.md#term-cache) or B_max allocation.

## Accepted future same-geometry reuse at [Engine](../../../glossary.md#term-engine) entry

A [memo](../../../glossary.md#cache-and-memoization) retains pure geometry results, not an invocation. [Prepare](../../../glossary.md#prepare) preindexes checks/recipes/publications.
`BatchBindingPlan.bind` still checks all [descriptors](../../../glossary.md#term-descriptor)/private obligations and returns a fresh [`Binding`](../../../glossary.md#term-binding)
retaining supplied descriptors every call; an Engine memo cannot skip its work. Later [Compiler](../../../glossary.md#term-compiler)-[owner](../../../glossary.md#term-owner)
refactoring/[signature](../../../glossary.md#term-signature)/protocol changes need exact approval; no new [API](../../../glossary.md#term-application-programming-interface-api) or production type is selected.

Select one future replaceable entry owned by the exact Engine [`PreparedExecution`](../../../glossary.md#prepared-execution--preparedexecution), empty initially.
Key: exact compiled/binding plan, inward preparation/memory-plan references, fixed recipes/publications
and explicit B/mode (even `[B,0]`); equal bytes/B or foreign equal plans/[shapes](../../../glossary.md#shape) never authenticate it.
Retain checked B-dependent domain validity, sizes/[strides](../../../glossary.md#term-stride)/offsets, aligned aggregate bounds and
template-derived non-input [publication](../../../glossary.md#term-publication) [metadata](../../../glossary.md#term-metadata); never [Tensor](../../../glossary.md#tensor) [identity](../../../glossary.md#term-identity-meanings), supplied descriptors,
`Binding`, storage, addresses, [leases](../../../glossary.md#term-lease), [mutable](../../../glossary.md#term-mutable) arrays or results.
One [graph](../../../glossary.md#graph)-sized [snapshot](../../../glossary.md#term-snapshot), no B-sized [payload](../../../glossary.md#artifact-envelope-and-atomic-publication)/map; old snapshots survive only via live calls/results.
Bound candidates by active admissions; no global/ThreadLocal [cache](../../../glossary.md#term-cache), pool or per-B plan cache.

Every call keeps Engine/owner/closed-prepared precedence, snapshots/counts/nulls and exact-ID or
positional ordering, then binds every input [occurrence](../../../glossary.md#term-occurrence), including unused/repeated roles. Check all
B occurrences, [rank](../../../glossary.md#term-rank)/fixed [axes](../../../glossary.md#term-axis), [dtype](../../../glossary.md#data-type--datatype)/requiresGrad, [layout](../../../glossary.md#layout) presence, [canonical](../../../glossary.md#term-canonical-form) strides/offsets/[view](../../../glossary.md#term-view-meanings)
policy before a hit; first-input B equality is not [admission](../../../glossary.md#term-admission). Avoid only covered Engine pure numeric/
size checks and template-only publication reconstruction. Snapshot each current storage association
once; revalidate presence/type/[liveness](../../../glossary.md#term-liveness)/thread access/capacity, borrow exact current prefixes and
retain physical type/[alignment](../../../glossary.md#term-alignment)/access/overlap admission before execution. All other checks remain.

Input-backed publications use the fresh binding's current supplied descriptor, never a prior caller's;
same-B admitted dense read views can differ in `isView`. Cache only template-derived non-input facts.
Preserve order, tensorId/role/derivativeOrder/targetIndex, [aliases](../../../glossary.md#aliasing-and-arena) and generated [gradient](../../../glossary.md#term-gradient) flags.
[Runtime](../../../glossary.md#term-runtime) gets only exact-plan numeric geometry, never the memo/symbols/semantic metadata. Each [run](../../../glossary.md#term-run)
gets fresh resources (even zero bytes), mutable packed/access/address/[carry](../../../glossary.md#term-carry) arrays and outputs;
copy facts into isolated mutable invocation arrays, never share them.

Publish a complete [immutable](../../../glossary.md#term-immutable) candidate only after pre-borrow descriptor/numeric/storage admission.
[Atomic](../../../glossary.md#resource-transaction-and-rollback) snapshot reads/replacement need future bounded publication-vs-close coordination through
existing [lifecycle](../../../glossary.md#lifecycle); exact mechanism gated, no execution lock/second lease. Mixed-B races may compute
duplicate candidates; either valid replacement may win, but each call keeps its selected snapshot.
Failed admission publishes nothing and preserves the good entry; later allocation/bind/execution failure
may retain valid metadata only, not resources/results/failures; reverse cleanup is unchanged.
Retry revalidates inputs/storage and creates fresh resources/leases/results. Close clears and prevents
late republishing; admitted calls keep local facts, while Runtime's unique lease decides close races.
After existing inward-delegate admission, a concurrent close suppresses/clears memo publication,
not the run through a new late outward-close rejection. No new lock/API mechanism is selected.

Model0041's completed authority cutover clarifies [resource ownership](../../../architecture/contracts/runtime-prepare-engine.md#accepted-symbolic-batch-resource-contract)<span data-glossary-navigation="true"> ([resource ownership glossary](../../../glossary.md#term-resource-ownership))</span>,
[run lifecycle](../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle) and
[prepared semantics](../../../architecture/contracts/runtime-prepare-engine.md#public-reusable-prepared-execution):
shareable never-mutated facts/private backing are distinct from mutable run arrays, and the slot
cannot change prepared meaning. Independent APPROVE; MAIN accepted 2026-10-25.
`RunMemoryGeometryTest` shares immutable geometry across fresh states, not authority for Engine
[memoization](../../../glossary.md#term-memoization). Root lifecycle/index and [CPU](../../../glossary.md#term-cpu) [ownership](../../../glossary.md#term-ownership-meanings) are coordinated in that cutover; affected
Javadocs remain a future source obligation. Existing `RunMemoryGeometry.resolve(plan, OptionalLong)`
is not a hit API. This design record enacts no authority by itself.
Memo [lookup](../../../glossary.md#term-lookup)/publication/copying stays at cold entry, never time/row/contributor loops;
generated code consumes direct numeric facts. No zero-overhead claim.

Conceptual serial witness: one preparation, fresh values, B=64→64→40→40→64: miss/hit/miss/hit/miss;
unchanged code/routes/slots, actual-sized resources. Use bounded test-local counters/probes for
avoided Engine work, not binder work or a benchmark campaign. Test invalid same-B/fixed-axis/type/
flag/layout/storage, current input-backed views, B=0/1, [constant](../../../glossary.md#term-constant)-byte `[B,0]`, [overflow](../../../glossary.md#term-overflow), foreign
preparations, failure/retry, close/concurrent mixed-B results and generated loops without memo access.

## Ordered gates and approval choices

1. Model0039 design freeze → independent read-only APPROVE (zero open P1/P2) → MAIN accepted
   DESIGN DELIVERY ONLY on 2026-10-24; this gate is Complete, not implementation authorization.
   Model0040 reuse amendment Complete DESIGN ONLY; independent APPROVE, MAIN accepted 2026-10-25;
   its ownership clarification is included in completed Model0041 authority, not implemented reuse.<span data-glossary-navigation="true"> ([ownership glossary](../../../glossary.md#term-ownership-meanings))</span>
2. User detailed semantic approval and authority-only admission were supplied on 2026-10-25.
   Model0041 is Complete: root/index, sole Scan semantic owner, coordinated Compiler/Runtime/
   CPU boundaries, ADR0051, focused explanation and glossary are enacted. Independent frozen
   APPROVE, zero open P1/P2; MAIN accepted 2026-10-25. Source gates require separate admission.<span data-glossary-navigation="true"> ([owner glossary](../../../glossary.md#term-owner); [Compiler glossary](../../../glossary.md#term-compiler); [Runtime glossary](../../../glossary.md#term-runtime); [CPU glossary](../../../glossary.md#term-cpu))</span>
3. Model0042 is separately admitted, implemented and MAIN accepted 2026-10-25, with explicit
   old-consumer rejection. Then separately admit one Compiler inference/actual-first-order-AD/binding-layout capability; then one
   coordinated Prepare/CPU/Engine numeric forward+backward vertical. Exact packets/classes/files
   are finalized at each frontier; no mechanical task cascade or Ready downstream implementation brief now.<span data-glossary-navigation="true"> ([inference glossary](../../../glossary.md#term-inference); [binding glossary](../../../glossary.md#term-binding); [layout glossary](../../../glossary.md#layout); [Prepare glossary](../../../glossary.md#prepare); [Engine glossary](../../../glossary.md#term-engine); [Ready glossary](../../../glossary.md#term-ready))</span>
4. Later, separately qualify length skipping and its symbolic-B combination, then selected
   NN/Training one-state reuse. Cold/copy/wire, variable T, wider execution dtypes and native/mixed
   remain independent gates, never inferred from this design or made prerequisites by speculation.<span data-glossary-navigation="true"> ([NN glossary](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [dtypes glossary](../../../glossary.md#data-type--datatype))</span>

All future gates are serial (Parallel group None), conflict on [Scan](../../../glossary.md#term-cumulative-scan)/resource/[identity](../../../glossary.md#term-identity-meanings) semantics,
and depend on their immediately preceding accepted gate plus the retained baseline. MAIN owns
shared-document integration and revalidates the accepted dirty base before any launch. Workers
[run](../../../glossary.md#term-run) focused plus affected-[module](../../../glossary.md#term-module) [validation](../../../glossary.md#term-validation) once; independent reviewers reuse it; MAIN runs one
recorded boundary join. Future tests cover both [seeds](../../../glossary.md#commit-release-distribution-log-and-seed-meanings)/traversals, nonuniform actual [AD](../../../glossary.md#autograd) versus
[unrolling](../../../glossary.md#term-unrolling), exact demanded/dead/shared bytes, B64→40→1→64, separate T0, B0/B1, [overflow](../../../glossary.md#term-overflow)/preborrow
failure, [aliases](../../../glossary.md#aliasing-and-arena), cleanup/retry/[concurrency](../../../glossary.md#term-concurrency) and retained static/length rejection paths. A tiny
warmed matched-work generated/direct probe separates cold setup/[admission](../../../glossary.md#term-admission)/[kernel](../../../glossary.md#kernel) costs; no long
benchmark campaign, universal speed/parity claim or new certification machinery.

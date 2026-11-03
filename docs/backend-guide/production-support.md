# First-production support matrix

## Status and interpretation

Reviewed scope freeze, 2026-11-01; component acceptance updated 2026-11-03 after the CPU/Metal
operation audit. This is the declared release target and evidence ledger; final candidate
acceptance requires separately retained exact-source/artifact/installation/root receipts.
General Scan and the required CPU/Metal neural journeys are component-qualified. The
[product goal](../planning/production-release-goal.md) and
[owning contracts](../../ARCHITECTURE.md#scope-indexed-normative-contracts) control semantics.
A source predicate, a generated-entry test and an actual Compiler backward run are different evidence.

## Common Java CPU domain

FLOAT32 and FLOAT64 are equal homogeneous production targets. The common baseline uses fully static
shapes, exact operation geometry/promotion, resolved nonnegative read strides/offsets, checked
referenced byte spans and injective outputs. Binding rejects illegal writable overlap. A family can
impose narrower layouts, extents or value guards; this table does not promise every Model layout.
Detached host publications are canonical. Scalar, vector and synchronous parallel strategies remain
operation/access/size qualified; equal support is not equal lane counts, thresholds, speed or bits.
Default scalar execution needs no incubator module. See [JVM prerequisites](../user-guide/running-models.md#jvm-prerequisites).

The static baseline does not remove advanced symbolic paths. Standalone ordinary symbolic
MATMUL/unary retain their family qualification. General Scan separately composes static and
qualified one-B all-valid/length bodies in both floating types through ordinary owners, with
typed discrete intermediates and proved coordinate semantics. This is not arbitrary dynamic-
dimension or mixed-floating backward parity; ordinary family and selected-VJP predicates remain.

| Family | Common static F32/F64 domain | First-order closure and current release evidence |
|---|---|---|
| Pointwise and selection | Floating tensor/scalar arithmetic, 19 unary kinds, clamp, comparisons, BOOL logic and WHERE under exact dtype/broadcast rules; Model-valid promoted F32/F64 and I32/I64 tuples | Original cotangent carrier conversion and exact stored-value comparisons passed public mixed-operand and WHERE witnesses. Discrete conditions receive no cotangent; implicit low arithmetic remains locked. Actual two-layer ReLU MLP forward/SGD/save/load/new-train passed in both types. |
| MATMUL/linear | Vector/matrix rank cases and broadcast batches; direct/tiled scalar, eligible N-vector and range parallelism | Ordinary rank edits, transpose and SUM-to-Shape derivatives. Homogeneous MLP/RNN paths passed; optional qualified OpenBLAS is not proof of portable Java coverage. |
| Conv1d/Conv2d | NCW/NCHW; bias/no bias, groups/depthwise, stride/padding/dilation, family-qualified noncanonical reads | Actual dX/dW/db passed 16 public cases across both types, grouped/dilated/strided, depthwise multiplier and asymmetric geometry. Both ranks passed complete owned Model journeys. |
| Max/AveragePool1d/2d | Exact floor/literal-ceil grids, padding/dilation, fixed average divisor, overlap | Actual backward passed 24 public cases: finite overlap, dilated ceil tails, NaN, signed zero, real -infinity and all-padding windows. Uses existing winner/fold rules, not a new derivative. |
| Conv3d | Fully static resolved-layout NCDHW; bias/no bias, groups/depthwise, stride/padding/dilation and overlapping windows | Shared Conv2d/Conv3d AD; actual dX/dW/db and selected weight-only gradients passed independent direct-loop central finite differences in both types. Grouped biased batch-two anisotropic and unbiased depthwise cases passed; both types also passed three-step parameter training/save/close/load/new-train. See the bounded volumetric qualification below. |
| Max/AveragePool3d | Fully static resolved-layout NCDHW; matching input/output gradient flags in F32/F64, checked kernel/stride/padding/dilation and literal ceil mode | Both types passed actual forward and explicit-cotangent gradients against independent spatial scatter and per-output finite differences, including batch/channel overlap and anisotropic geometry. Existing shared pooling adjoints and 3D windows close trainable Conv3d → pool → flatten → linear, all parameter roles, three SGD steps and save/close/load/fresh configured training. |
| SUM/MEAN/PROD/MIN/MAX and SUM-to-Shape | Full/single/multiple axes, exact keep/remove axes and target shape | Both types passed actual axis SUM/MEAN/PROD gradients, including zero products. MIN/MAX tie/other geometry qualification remains separate; generated cumulative/layout closure is not inferred. |
| Masked/statistical reductions | Masked SUM/MEAN, LOG_SUM_EXP, variance/std, L1/L2; correction/domain guards | Both types passed actual masked SUM/MEAN, LOG_SUM_EXP, corrected variance/std and L1/L2 against independent scalar forward/derivative oracles. BOOL masks have no cotangent; wider domains remain qualified separately. |
| Cumulative operations | Axis, direction and inclusive/exclusive CUM_SUM/CUM_PROD | Both types passed all inclusive/exclusive x forward/reverse combinations with nonuniform cotangents and zero-product inputs. Uses ordinary rules; not a universal layout/strategy claim. |
| SOFTMAX/LOG_SOFTMAX | Nonempty selected axis, matching type/shape, finite input domain | CPU0077 parity repair accepted: both types admit finite negative-overflow shifts. 45 targeted tests/Javadocs and actual public both-kind/type forward/Compiler-backward passed; tied maxima give nonzero gradients, with pre-write invalid/freshness/loss controls. |
| LayerNorm/RMSNorm | Static trailing normalized shape, typed epsilon and optional affine roles | Actual both-type affine LayerNorm three-role and RMSNorm two-role gradients passed independent scalar oracles. Wider normalized shapes/layouts retain family predicates. |
| BatchNorm inference/training | Exact channel geometry; training contributor minimum and five output roles | Both types passed all five input-role gradients for inference and the joint three public training roots (normalized output/next mean/next variance). Saved auxiliaries remain invalid independent derivative roots. |
| MSE and dense/index cross entropy | NONE/SUM/MEAN, exact target signatures, ignore/exclusion guards | Both types passed actual MSE/dense CE data+target gradients, including nonunit dense target sums, and index CE with ignore control. Integral targets are nondifferentiable; other reductions/domains retain their own qualification. |
| Affine/movement/windows | Declared reshape/rank edits/permute, SELECT/checked signed-step SLICE, PAD/TILE, bounded one-pass CONCAT/STACK/replacement, UNFOLD_AXIS/2D/3D and FOLD_AXIS/2D/3D | Signed/empty materialization and negative-step extraction/update adjoints passed. Seventeen-input CONCAT/STACK passed every-input and repeated-identity cotangents; composed 65-input consumers also passed within checked generation limits. Raw movement covers all seven carriers. UNFOLD3D preserves raw F32/F64/BF16/F16 words; arithmetic FOLD3D remains F32/F64. CPU0079 noncontiguous RESHAPE and the low-arithmetic lock remain. |
| Gather/scatter/one-hot | Exact index shape and INT32/INT64 bounds; declared NONE/ADD/MUL/MIN/MAX modes | Both types passed gather/gatherElements/gatherNd and all five scatterElements/scatterNd modes, data/update roles, duplicates and zero products. Bounds/uniqueness pre-write tests remain independent; no index cotangent. |
| Ordering and extrema | Stable SORT/ARGSORT/TOP_K and exact tie/index behavior | Both types passed actual SORT/TOP_K value gradients against scalar oracles; indices have no cotangent. Tie/special/other geometry qualification is not inferred from separated-value cases. |
| Attention | Rank>=2 Q/K/V, exact contraction and broadcast batches, optional BOOL/causal mask and normalized weights | Both types passed joint output/weights gradients for all Q/K/V roles, masked and unmasked. Literal EXP overflow/underflow and nonempty negative-zero value contractions passed separately, including controlled single-output Scan with inactive NaNs. Ordinary AD requires canonical two-output form; one-output construction is forward-only for AD. Masks have no cotangent, weights no V route. |
| Explicit RNG/dropout | Existing state/mask/next-state signatures and typed value semantics | No derivative for state/mask. Scan purity must follow the actual effect contract, not assume every random operation is pure. |
| Fixed recurrent scans | Existing explicit RNN/GRU/LSTM forward constructors and kernels | Remain forward-only. Shared nonempty projections retain first-product signs; both-type RNN negative-zero sequence/final-state publication passed in both directions. Old fixed files and fixed-containing Models still reject training before setup. |
| General Scan / practical RNN | Ordinary CPU-supported PURE bodies, static/qualified one-B F32/F64, per-value typed intermediates; all-valid whole-Tensor or length-mode proved ROW(axis)/SHARED semantics | Compiler0017/CPU0081 are integrated and source/consumer-qualified: ordinary lowering/emission, actual selected VJPs, structural activity, guards, empty/T0 caller publication, reuse and genuine old-recipe replay. The bounded public RNN train/save/load/new-train workflow remains separately qualified. Metal Scan has independent qualification below; broader cold/wire support is not inferred. |

All entries retain [Model numerical semantics](../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics).
Finite tolerances do not replace exact discrete, special-value, saved-value or pre-write validation.
Current exact predicates remain in CpuCapabilityProvider and the CPU lowerers; a family row is not
a declaration that arbitrary partitions or every possible backward target are eligible.

## CPU operation-audit closure

Local qualification covers all 137 ordinary signatures across 115 kinds, with general Scan
recorded separately. The current catalog completed 804 forward and 736 selected first-order AD
runs without an execution-stage failure. Its 622 selection rejections are confined to low-storage
projections. Rejected forward kinds were classified under the intentional arithmetic lock;
selected low VJPs may also require locked arithmetic. Illegal Model projections and
nondifferentiable/deferred roles are classified separately. These are bounded fixtures, not
every possible Model geometry; execution hashes are not numerical correctness proof.

Independent forward/derivative oracles passed 178 cases: 81 ordinary cases per floating type
and sixteen Pool3d Scan cases. Additional public witnesses cover promoted operands/original
cotangent types, signed slicing/update, variadic/repeated identities and composite special values.
A separate generated TILE witness checked `[200001,2,3]`, strides `[8,3,2]`, offset `5`, split
ranges, heap/segment outputs and untouched holes. Current-quotient checks prevent cyclic CPU units.
Changed affine empty-range and literal-attention classes use selective projections 73/74;
the generator envelope stays 71. Source review approved the affected owners. Metal evidence
and joined exact-source/artifact/installation acceptance are recorded separately below.

## Conv3d training and independent Pool3d boundary

The user's explicit Conv3d training request supersedes the earlier 1D/2D-only release-rank
decision. Conv1d/2d and Conv3d train through existing public Tensor algebra; this is not a
guard-only unlock. The shared convolution rule retains groups in every matrix contraction:
input gradients use transposed weights and overlap-add fold, weight gradients use unfold,
matrix multiplication and batch reduction, and bias gradients reduce batch and all spatial
axes. Each selected gradient returns to its target's exact Shape and data type.

Public CPU F32/F64 proof covers every input/weight/bias role for grouped biased batch-two
Conv3d with anisotropic stride, symmetric padding, dilation and overlap, plus unbiased
depthwise groups-two geometry and selected weight-only gradients. Both types trained every
weight/bias for three steps, saved, closed and loaded bit-exact parameters, then configured
fresh training and passed another independent oracle-checked SGD step. CPU general Scan also
composed actual Conv3d forward and reverse in eight independent finite-difference cases:
all-valid/length-aware, both traversal directions, and F32/F64. There is no new 3D blacklist;
complete body/binder qualification and Scan higher-order restrictions still apply.

Pool3d completion now independently qualifies CPU F32/F64 and Metal-only F32 forward and actual
backward through shared pooling adjoints and `UNFOLD3D`/`FOLD3D`, without new standalone backward
kernels or fallback. Both max and average passed independent spatial-scatter and finite-difference
oracles, plus Conv3d → pool → flatten → linear all-parameter finite differences, three SGD steps
and save/close/load with bit-exact state and fresh configured SGD. CPU Scan passed F32/F64,
max/average, all-valid/runtime-length and both traversal directions, including actual backward
input-role finite differences and save/close/load/fresh execution.

These are locally runtime-qualified capabilities, not joined-release or published-candidate
acceptance. Ordinary public CPU named-B admission, arbitrary layouts, the CPU low-arithmetic
lock and performance boundaries remain unchanged. Metal Scan has separate qualification below.
CPU low maximum-pool forward remains within its exact domain; low pooling gradients and average
arithmetic are not unlocked. Metal FLOAT64/low pooling retains its qualified non-overlap limits.

## Temporary Java low-computation lock

The required lock is implemented. Preserve BF16/FP16 types, storage/borrowing/export,
raw transfer/movement, initialized literals and casts/conversions. INT32/INT64/BOOL control remains.
An explicit low-storage -> F32 cast -> F32 computation path stays valid, including existing
Training storage/conversion behavior; this is not an implicit emulated-low backend fallback.

Reject low numerical operand roles, even if the result is promoted: arithmetic/transcendentals,
matmul/convolution, arithmetic reductions/cumulative operations, average pooling, normalization,
loss/attention, dropout scaling, overlap-fold addition and scatter ADD/MUL. Preserve exact
comparison/classification/WHERE, MIN/MAX/arg-extrema/ordering/winner selection and exact unary
ABS/NEG/SIGN/FLOOR/CEIL/RELU/clamp behavior under their existing contracts. Their backward may
still require prohibited low numerical operations. Do not disable every low-containing partition.

Admission must be independent at occurrence/provider, direct preparation/IR realization before
cache hits, and direct fixed-recurrent execution. Retain implementation; add no runtime opt-in
backdoor and no new low capability. Current ledger v2 changes, historical v1 and Metal do not.

[CPU0078](../planning/backends/cpu/tasks/0078-temporary-low-arithmetic-lock.md#result) records the
passed affected gate and actual both-type public rejection/retention proof. A real six-step
changing-target TrainingSession consumer also verifies exact F32 master values, low logical
narrowing, snapshot restore and fresh resumed training. Complete original owner/query domains
remain; retained generated rows are literal, historical v1 and Metal unchanged by this CPU cutover.

## Owned source and persistence boundaries

MLP, default persistable CNN1d/2d/3d and runtime-length tanh RNN support compute, one configured
training session, repeated fresh batches, save, close, load and newly configured training in
their qualified CPU domains. Metal F32 MLP/CNN1d/2d/3d and runtime-length general-Scan RNN also
passed actual-GPU parameter updates/save/load/fresh training without CPU fallback.

Tensor-level numerical support is broader than cold acquisition/serialization. An outer or
contained Pool3d occurrence selects v6; other admitted flat/all-valid/length sources retain their
byte-identical smallest sufficient v1–v5 format. Arbitrary PURE Scan execution does not
automatically widen cold reconstruction or add wire opcodes. Old fixed files are never implicitly
migrated. No optimizer/history/compiled-plan continuation is promised.

## Metal production target

Metal is required and is **not experimental**. Its independent target includes qualified static
F32 ordinary numerical families, Conv3d/Pool3d training, attention and native general Scan with
runtime-length recurrent training. Independently qualified BF16/FP16 and exact/raw F64/integral
rows remain separate. General F64 neural arithmetic and the three fixed-recurrent BPTT signatures
remain exclusions, not CPU fallback paths. The [Metal guide](metal-backend.md) and
[owning contract](../architecture/contracts/backend-execution.md#metal-backend) define exact limits.

The complete operation catalog covers 5,018 rows, all 115 ordinary kinds and all 137 legal
signature fixtures. It completed 744 forward and 494 actual selected first-order AD runs with
zero execution-stage gaps. The 924 selection rejections are classified separately from Model-
illegal projections and nondifferentiable/deferred roles; they are not execution successes.
The remaining F32 selection fixtures are seventeen-input CONCAT/STACK, beyond the retained
sixteen-input/five-output bound per ordinary instruction. Dtype-specific exclusions remain independent.

Independent Metal-only numerical proof passed 93 F32 forward/actual all-role derivative cases:
81 ordinary cases, eight Pool3d Scan cases and four signed-update/overlapping-axis-fold,
scalar-reciprocal and repeated-STACK accumulation edges. Offset-view binary arithmetic uses
the existing physical-stride kernels; packed canonical operands retain MPSGraph.
Another 136 exact-domain cases passed, including all 49 casts, four floating WHERE cases,
stored F64 decisions and integral modular/promotion paths. Numerical correctness is not inferred
from catalog hashes. Public runtime-length RNN separately passed all-parameter finite
differences, repeated SGD with changing lengths, all-inactive rows, invalid-length nonmutation/
retry, save/close/load raw-exact state and fresh configured SGD on a Metal-only Engine.

General Scan uses a true native time loop and once-prepared existing ordinary kernels, not
Java child invocation, time unrolling or a CPU evaluator. Controlled operands and original
canonical output roles retain structural inactivity and guard-before-write semantics.
GeneralScan, selected reverse bodies, source-only gradients and empty storage remain
independently checked; they do not confer arbitrary symbolic/empty or higher-order support.

The previously integrated Metal identity39 source passed full F32 MLP/CNN scalar-oracle compute,
all-parameter SGD/save/close/load/new training on the actual GPU without CPU fallback.
All fifteen original-typed F32/BF16/F16 five-form forward/both-generated-gradient graphs now pass unchanged,
including transpose-of-dense materializing RESHAPE. Public raw reshape additionally passed all
seven carriers. [Metal0081](../planning/backends/metal/tasks/0081-materializing-reshape-composition.md#result)
retains exact source, package, device/skip and before-failure evidence; this is not arbitrary backward closure.
Final independent source/evidence/documentation review accepted this declared Metal matrix;
Metal0080/0081 are Complete. This is component acceptance, not a complete release.

The integrated [Metal0082 host-copy correction](../planning/backends/metal/tasks/0082-host-copy-autorelease-boundaries.md#result)
also passes a rebuilt source-matched native install, full GPU/public integration and unchanged
MLP/CNN training/persistence and all-seven-carrier raw movement. Raw and mixed public before/after
probes isolate and repair Foundation host-copy retention: the named IOAccelerator category stays
flat across repeated windows, including 3,288 retained closed handles. Full-process footprint,
which includes warmup and retained Java metadata, is not claimed constant. This finite observation
is not universal allocation proof. Final independent source/evidence/docs review accepted Metal0082
with zero P1/P2 findings; the component is Complete.

The Conv3d extension additionally passed the grouped/bias/depthwise all-role finite-difference
and real NN training/persistence program in a Metal-only Engine containing no CPU backend.
FLOAT32 FOLD_AXIS/FOLD2D/FOLD3D overlap addition is output-owned and deterministic, with no
atomics, scratch, host fallback/repair or software FLOAT64. Other carriers retain qualified
non-overlap limits. Signed FLOAT32 SLICE materializes canonical storage; it does not authorize
negative physical strides. Both additions close actual selected movement adjoints.
The current coordinated identity is 41, program schema 21/header124, native ABI7/thirteen
exports, operation wires1..116 and attribute wires0..42. The fixed corpus has 22 authenticated
components, 192,958 UTF-8 bytes and 295 compiled functions. Source review and device compilation
are component evidence; final joined release acceptance still requires the receipts below.

The exact new native source was built, ad-hoc signed, ZIP-extracted outside checkout and
reverified before the GPU gate. The supported reproducible local source-build installation uses
the existing [native build/package/install commands](../../modules/backend/metal/macos-arm64/README.md#build-prerequisites-and-local-package)
and an explicit absolute MetalBackendConfiguration path. This is not a trusted public prebuilt
distribution: no signing identity or hosted GPU runner is established. Official prebuilt signing/
publication remains distinct from a caller's verified local source build; no verifier bypass or relabeling.

## Evidence and candidate qualification

Accepted bounded RNN and wire records: [Compiler0018](../planning/modules/compiler/tasks/0018-static-length-scan-cold-reconstruction.md#result),
[Checkpoint0014](../planning/extensions/checkpoint/tasks/0014-length-aware-general-scan-v5.md#result),
[NN0033](../planning/extensions/nn/tasks/0033-trainable-runtime-length-rnn-workflow.md#result).
CPU MLP/CNN numerical logs and the 16 convolution/24 pooling actual-backward cases are retained
in the coordinator's source-snapshot evidence, not inferred from operation counts.
Another 54 ordinary and 30 indexing/ordering public cases passed independent scalar forward and
actual all-role Compiler gradients in both CPU types. These are finite canonical geometries,
not automatic special-value/layout/strategy or all-PURE Scan proof.

The original general-Scan baseline rejected all 24 attempted F32/F64 RELU/SOFTMAX/LayerNorm
combinations across both modes/traversals. Those before-failures remain retained. The unchanged
program now passes, together with ordinary84, composition268 and named144 numerical cases,
structural/materialized boundaries, typed raw/T0 sources and genuine legacy recipe replay.
The unfiltered affected twelve-project recovery passed 4,942 tests, 64 skips, zero failures/errors
and all twelve Javadocs. Independent final semantic/physical ownership reviews approved with
zero findings. This is component acceptance; it does not imply universal layout or performance.

[Engine0022](../planning/modules/engine/tasks/0022-result-publication-and-closed-payload-cleanup.md#result)
repairs the reproduced closed-publication heap-payload retention and advanced publication rollback
gap. Independent approval, affected gates and public both-type heap/lifetime/precedence/race proof
passed. The heap diagnostic is not native-memory plateau evidence or a whole-release qualification.

The integrated [Engine0023 correction](../planning/modules/engine/tasks/0023-reentrant-close-and-lease-admission.md#result)
passes the unchanged real-GPU self-close and actual bounded-heap token-OOM consumers plus affected
module/public-integration gates and final independent source/evidence/docs review. Engine0023 is
Complete for that correction. No native OOM was induced; Metal0082 supplies the separate finite,
category-specific native-memory evidence above, not a universal allocation guarantee.

Engine0024's standard eighteen-product distribution mechanism is Complete after clean extracted
source builds and external Maven/Gradle consumers. A fresh Maven project declaring only `nn`
also resolves its sixteen-JAR closure and passes the original full both-type MLP/CNN/RNN
numerical, optimizer and save/load/new-train programs without the incubator module. The documented
CPU owner-close and fluent examples run against those installed artifacts. Standard CPU startup
still performs its documented optional OpenBLAS probe; grant FFM access under the
[JVM prerequisites](../user-guide/running-models.md#jvm-prerequisites).
This qualifies the installation mechanism, not the final all-PURE release candidate.

General Scan composition/typed activity and final-source CPU/Metal neural replay are accepted.
Candidate acceptance requires the versioned source/artifact freeze, clean installed artifact/API
replay and source-matched repository checkpoint. Short composition timings remain report-only
observations with warmup/host/storage limits, not a speed or zero-allocation promise.
The owner chose private/internal distribution: no public licence or registry publication is
selected. Cephes/Gradle notices remain packaged; they do not grant public rights to the project's
own code. Final candidate acceptance is recorded separately against its actual frozen artifacts.

## Accepted local candidate: Pool3d and operation-audit closure

The private/internal 0.1.0 candidate passed the joined `test build releaseDistribution` gate
on 2026-11-03. The gate reported 220 actionable tasks: 106 executed, five from cache and 109
up-to-date; it does not claim every test was freshly rerun in that command. The preceding
unfiltered joined `test build` also passed. Earlier failed commands remain failed evidence.

Fresh catalog replay against the qualified code reproduced the CPU 804 forward/736 actual-AD
and Metal 744 forward/494 actual-AD totals above, with zero execution-stage gaps. Separate
independent numerical replay passed CPU 178 and Metal 93 all-role cases plus 136 exact-domain
cases in each of two isolated installations: a Maven consumer declaring only `nn:0.1.0`, and
a clean no-Git source build with build cache disabled and all sixteen classes tasks rerun.
Every one of the sixteen resolved runtime JARs matched the published bytes. Neither runtime
classpath used checkout classes or old product JARs.

Both installations passed CPU F32/F64 and Metal-only F32 MLP/CNN/runtime-length RNN workflows,
Conv3d → max/average Pool3d → dense all-parameter training/save/close/load/fresh training, and
24 Pool3d Scan finite-difference/cold replay cases across the qualified types, both activity
modes and both traversal directions. The documented FluentModelExample produced `6`, `25`,
`-2` from installed artifacts and independently compiled extracted classes.

The native package was rebuilt from the qualified source, locally signed, extracted and
verified outside the checkout, then exercised on Apple M3 Max with JDK 26 and explicit native
access, without preview or the incubator Vector module. Short promoted-ADD and offset-SELECT/ADD
timings are retained as report-only public-execution observations; they include ownership/
scheduling and GPU-transfer costs, unlike the clean-Java kernel-only reference. No global
speed parity, allocation plateau or throughput SLA is inferred.

The versioned `build/release/0.1.0/verification.json`, `SHA256SUMS` and `qualification/` records
bind source provenance, artifact/native hashes, commands, failure/recovery history, exact
catalog rows, installation logs and limitations. The final documentation-only artifact freeze
must retain the qualified non-documentation bytes. Acceptance is for this explicit matrix,
not arbitrary geometry, newly unlocked CPU low arithmetic, general Metal F64 arithmetic,
fixed-recurrent BPTT, public licensing/registry publication or a trusted public native prebuilt.

The successful root gate accepted 794 retained JUnit suite reports: 5,852 tests, 65 skips,
zero failures and zero errors. Metal contributes 545 tests with ten skips; actual-GPU numerical
and installed-consumer proof above is separate from those skips. Cached/up-to-date report
acceptance is not reported as a fresh execution of every test.

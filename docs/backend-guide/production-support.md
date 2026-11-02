# First-production support matrix

## Status and interpretation

Reviewed scope freeze, 2026-11-01; component acceptance updated 2026-11-02. This is the declared
release target and evidence ledger; final candidate acceptance requires its separately retained
exact-source/artifact/installation/root receipts. General Scan and the required CPU/Metal neural
journeys are component-qualified. The
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
| Pointwise and selection | Floating tensor/scalar arithmetic, 19 unary kinds, clamp, comparisons, BOOL logic and WHERE under exact dtype/broadcast rules | Ordinary formula owners; discrete conditions receive no cotangent. Actual two-layer ReLU MLP forward/SGD/save/load/new-train passed in both types. Broader closure remains a release gate. |
| MATMUL/linear | Vector/matrix rank cases and broadcast batches; direct/tiled scalar, eligible N-vector and range parallelism | Ordinary rank edits, transpose and SUM-to-Shape derivatives. Homogeneous MLP/RNN paths passed; optional qualified OpenBLAS is not proof of portable Java coverage. |
| Conv1d/Conv2d | NCW/NCHW; bias/no bias, groups/depthwise, stride/padding/dilation, family-qualified noncanonical reads | Actual dX/dW/db passed 16 public cases across both types, grouped/dilated/strided, depthwise multiplier and asymmetric geometry. Both ranks passed complete owned Model journeys. |
| Max/AveragePool1d/2d | Exact floor/literal-ceil grids, padding/dilation, fixed average divisor, overlap | Actual backward passed 24 public cases: finite overlap, dilated ceil tails, NaN, signed zero, real -infinity and all-padding windows. Uses existing winner/fold rules, not a new derivative. |
| Conv3d | Existing NCDHW grouped/bias/stride/padding/dilation forward domain | **No v1 training.** Any Conv3d in original forward inventory rejects backward, even for another requested target. See the explicit rank decision below. |
| Max/AveragePool3d | Existing NCDHW forward domain, only input/output requiresGrad=false | **No v1 training.** Gradient-bearing pooling can reject even forward composition. Volumetric windows lack a CPU owner; formula construction alone is not execution. |
| SUM/MEAN/PROD/MIN/MAX and SUM-to-Shape | Full/single/multiple axes, exact keep/remove axes and target shape | Both types passed actual axis SUM/MEAN/PROD gradients, including zero products. MIN/MAX tie/other geometry qualification remains separate; generated cumulative/layout closure is not inferred. |
| Masked/statistical reductions | Masked SUM/MEAN, LOG_SUM_EXP, variance/std, L1/L2; correction/domain guards | Both types passed actual masked SUM/MEAN, LOG_SUM_EXP, corrected variance/std and L1/L2 against independent scalar forward/derivative oracles. BOOL masks have no cotangent; wider domains remain qualified separately. |
| Cumulative operations | Axis, direction and inclusive/exclusive CUM_SUM/CUM_PROD | Both types passed all inclusive/exclusive x forward/reverse combinations with nonuniform cotangents and zero-product inputs. Uses ordinary rules; not a universal layout/strategy claim. |
| SOFTMAX/LOG_SOFTMAX | Nonempty selected axis, matching type/shape, finite input domain | CPU0077 parity repair accepted: both types admit finite negative-overflow shifts. 45 targeted tests/Javadocs and actual public both-kind/type forward/Compiler-backward passed; tied maxima give nonzero gradients, with pre-write invalid/freshness/loss controls. |
| LayerNorm/RMSNorm | Static trailing normalized shape, typed epsilon and optional affine roles | Actual both-type affine LayerNorm three-role and RMSNorm two-role gradients passed independent scalar oracles. Wider normalized shapes/layouts retain family predicates. |
| BatchNorm inference/training | Exact channel geometry; training contributor minimum and five output roles | Both types passed all five input-role gradients for inference and the joint three public training roots (normalized output/next mean/next variance). Saved auxiliaries remain invalid independent derivative roots. |
| MSE and dense/index cross entropy | NONE/SUM/MEAN, exact target signatures, ignore/exclusion guards | Both types passed actual MSE/dense CE data+target gradients, including nonunit dense target sums, and index CE with ignore control. Integral targets are nondifferentiable; other reductions/domains retain their own qualification. |
| Affine/movement/windows | Declared reshape/rank edits/permute, SELECT/positive SLICE, PAD/TILE/CONCAT/STACK/replacement, UNFOLD_AXIS/2D and FOLD_AXIS/2D | Existing inverse views, crop/update, SUM and matching fold rules. CPU0079 additionally passed all-seven-carrier public noncontiguous ordinary RESHAPE raw-bit/order proof through the existing copy. No implicit UNFOLD3D/FOLD3D promise. |
| Gather/scatter/one-hot | Exact index shape and INT32/INT64 bounds; declared NONE/ADD/MUL/MIN/MAX modes | Both types passed gather/gatherElements/gatherNd and all five scatterElements/scatterNd modes, data/update roles, duplicates and zero products. Bounds/uniqueness pre-write tests remain independent; no index cotangent. |
| Ordering and extrema | Stable SORT/ARGSORT/TOP_K and exact tie/index behavior | Both types passed actual SORT/TOP_K value gradients against scalar oracles; indices have no cotangent. Tie/special/other geometry qualification is not inferred from separated-value cases. |
| Attention | Rank>=2 Q/K/V, exact contraction and broadcast batches, optional BOOL/causal mask and normalized weights | Both types passed joint output/weights gradients for all Q/K/V roles, masked and unmasked. Ordinary AD requires canonical two-output form; one-output constructor is forward-only for AD. Masks have no cotangent, weights no V route; guarded semantics remain separate. |
| Explicit RNG/dropout | Existing state/mask/next-state signatures and typed value semantics | No derivative for state/mask. Scan purity must follow the actual effect contract, not assume every random operation is pure. |
| Fixed recurrent scans | Existing explicit RNN/GRU/LSTM forward constructors and kernels | Remain forward-only. Old fixed files and any fixed-containing Model reject training before setup. |
| General Scan / practical RNN | Ordinary CPU-supported PURE bodies, static/qualified one-B F32/F64, per-value typed intermediates; all-valid whole-Tensor or length-mode proved ROW(axis)/SHARED semantics | Compiler0017/CPU0081 are integrated and source/consumer-qualified: ordinary lowering/emission, actual selected VJPs, structural activity, guards, empty/T0 caller publication, reuse and genuine old-recipe replay. The bounded public RNN train/save/load/new-train workflow remains separately qualified. No wider cold/wire or Metal Scan support follows. |

All entries retain [Model numerical semantics](../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics).
Finite tolerances do not replace exact discrete, special-value, saved-value or pre-write validation.
Current exact predicates remain in CpuCapabilityProvider and the CPU lowerers; a family row is not
a declaration that arbitrary partitions or every possible backward target are eligible.

## Explicit CNN rank decision

The first release trains **Conv1d/2d and Max/AveragePool1d/2d**, with the declared geometries and
actual whole-backward qualification. It retains **3D forward-only**, precisely as bounded above;
it does not promise trainable Conv3d or a gradient-bearing Conv3d -> Pool3d inference chain.
This is an explicit product-scope choice under the delegated release decisions, not removal of
existing code or a claim that 3D training works.

Reason: Conv3d AD is globally closed; Pool3d gradient flags independently narrow forward support;
CPU UNFOLD3D/FOLD3D are absent. Opening only a guard would be incorrect. A proper later 3D training
extension needs the existing grouped convolution/window mathematics generalized with resource
and performance proof, not a copied 2D algorithm. CNN training is already a mandatory, exercised
1D/2D feature; volumetric training is not required to satisfy that declared first-release rank scope.

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

MLP, default persistable CNN1d/2d and runtime-length tanh RNN support compute, one configured
training session, repeated fresh batches, save, close, load and newly configured training.
The unchanged independent final-source consumers passed both CPU types; Metal F32 MLP/CNN
also passed actual-GPU forward/parameter updates/save/load/fresh training without CPU fallback.

Tensor-level numerical support is broader than cold acquisition/serialization. Existing bounded
flat/all-valid/length constructor inventories select the smallest v1-v5 format. Arbitrary PURE
Scan execution does not automatically widen cold reconstruction or add wire opcodes. Old fixed
files are never implicitly migrated. No optimizer/history/compiled-plan continuation is promised.

## Metal production target

Metal is required and is **not experimental**. Its independent matrix includes qualified F32
neural MLP/CNN domains and independently qualified BF16/FP16 rows; it does not inherit CPU F64
or operation parity. F64 neural MATMUL, general Scan/runtime RNN, attention and unsupported
occurrences remain explicit exclusions, not CPU fallback paths. Exact domains stay with the
[Metal owner](../architecture/contracts/backend-execution.md#metal-backend).

The integrated Metal identity38 source passes unchanged full F32 MLP/CNN scalar-oracle compute,
all-parameter SGD/save/close/load/new training on the actual GPU without CPU fallback. All fifteen
original-typed F32/BF16/F16 five-form forward/both-generated-gradient graphs now pass unchanged,
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

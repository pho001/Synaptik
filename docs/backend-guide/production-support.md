# First-production support matrix

## Status and interpretation

Current public-readiness source and isolated candidate qualification, 2026-11-04, are recorded in
the [audit receipt below](#public-readiness-audit-current-source-and-installed-candidate).
The earlier bounded private/internal `0.1.0` matrix retains its own operation-catalog, root and
isolated source/artifact/consumer evidence. None is a claim about every Model geometry, public
registry publication or a trusted native prebuilt; sealed receipts apply only to their frozen bytes.
General Scan and the bounded CPU/Metal neural journeys have component evidence. The
[product goal](../planning/production-release-goal.md) and
[owning contracts](../../ARCHITECTURE.md#scope-indexed-normative-contracts) control semantics.
A source predicate, a generated-entry test and an actual Compiler backward run are different evidence.

Post-candidate source work is recorded in the
[interchangeability follow-up](backend-interchangeability-audit.md#15-navazující-opravy-a-aktuální-hranice)
and [public-release audit](../planning/public-release-audit-2026-11-04.md#result).
Neither relabels the earlier distribution as rebuilt or reissued.

Read each row as bounded evidence, not a promise for every tensor. A **domain** is the allowed
combination of numerical types, dimensions, storage mappings and values. **Qualification**
records which implementation and environment were checked. A **catalog** exercises representative
cases; an independent **numerical oracle** calculates an expected answer without reusing the
implementation being checked. Passing a catalog and agreeing with an oracle are different claims.

## Common Java CPU domain

F32/F64 abbreviate the FLOAT32/FLOAT64 storage formats; BF16/F16 abbreviate BFLOAT16/FLOAT16.
**Homogeneous** means the relevant numerical values use one common type. A **stride** is the
storage step for advancing one coordinate, an **offset** is the mapping's starting displacement,
and an **injective** output mapping gives every distinct logical coordinate a distinct address.
These properties matter even when two tensors have the same shape.

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
| Pointwise and selection | Floating tensor/scalar arithmetic, 19 unary kinds, clamp, comparisons, BOOL logic and WHERE under exact dtype/broadcast rules; Model-valid promoted F32/F64 and I32/I64 tuples | Original cotangent carrier conversion and exact stored-value comparisons passed public mixed-operand and WHERE witnesses. Discrete conditions receive no cotangent. Low arithmetic now follows the parity section below; these historical F32/F64 receipts do not qualify it. Actual two-layer ReLU MLP forward/SGD/save/load/new-train passed in both types. |
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
| Affine/movement/windows | Declared reshape/rank edits/permute, SELECT/checked signed-step SLICE, PAD/TILE, bounded one-pass CONCAT/STACK/replacement, UNFOLD_AXIS/2D/3D and FOLD_AXIS/2D/3D | Signed/empty materialization and negative-step extraction/update adjoints passed. Seventeen-input CONCAT/STACK passed every-input and repeated-identity cotangents; composed 65-input consumers also passed within checked generation limits. Raw movement covers all seven carriers. UNFOLD3D preserves raw F32/F64/BF16/F16 words; arithmetic FOLD3D remains F32/F64. Static noncontiguous RESHAPE materializes through the existing ordinal copy; the low-arithmetic lock remains. |
| Gather/scatter/one-hot | Exact index shape and INT32/INT64 bounds; declared NONE/ADD/MUL/MIN/MAX modes | Both types passed gather/gatherElements/gatherNd and all five scatterElements/scatterNd modes, data/update roles, duplicates and zero products. Bounds/uniqueness pre-write tests remain independent; no index cotangent. |
| Ordering and extrema | Stable SORT/ARGSORT/TOP_K and exact tie/index behavior | Both types passed actual SORT/TOP_K value gradients against scalar oracles; indices have no cotangent. Tie/special/other geometry qualification is not inferred from separated-value cases. |
| Attention | Rank>=2 Q/K/V, exact contraction and broadcast batches, optional BOOL/causal mask and normalized weights | Both types passed joint output/weights gradients for all Q/K/V roles, masked and unmasked. Literal EXP overflow/underflow and nonempty negative-zero value contractions passed separately, including controlled single-output Scan with inactive NaNs. Ordinary AD requires canonical two-output form; one-output construction is forward-only for AD. Masks have no cotangent, weights no V route. |
| Explicit RNG/dropout | Existing state/mask/next-state signatures and typed value semantics | No derivative for state/mask. Scan purity must follow the actual effect contract, not assume every random operation is pure. |
| Fixed recurrent scans | Existing explicit RNN/GRU/LSTM constructors and CPU forward kernels | CPU evidence here covers forward execution, including shared first-product signs and both-direction RNN negative-zero sequence/final-state publication. Shared fixed-cell first-order rules are no longer categorically deferred; actual Metal F32/BF16/F16 all-role proof is recorded below and does not itself qualify CPU fixed-cell training. Fixed recipes retain their own working/publication semantics rather than becoming GeneralScan recipes. |
| General Scan / practical RNN | Ordinary CPU-supported PURE bodies, static/qualified one-B F32/F64, per-value typed intermediates; all-valid whole-Tensor or length-mode proved ROW(axis)/SHARED semantics | Compiler0017/CPU0081 are integrated and source/consumer-qualified: ordinary lowering/emission, actual selected VJPs, structural activity, guards, empty/T0 caller publication, reuse and genuine old-recipe replay. The bounded public RNN train/save/load/new-train workflow remains separately qualified. Metal Scan has independent qualification below; broader cold/wire support is not inferred. |

All entries retain [Model numerical semantics](../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics).
Finite tolerances do not replace exact discrete, special-value, saved-value or pre-write validation.
Current exact predicates remain in CpuCapabilityProvider and the CPU lowerers; a family row is not
a declaration that arbitrary partitions or every possible backward target are eligible.

The optional [oneDNN CPU peer](cpu-backend.md#explicit-onednn-composition) retains its bare
F32 MATMUL baseline and adds qualified common F32 occurrences under the
[API/implementation/geometry limits](cpu-backend.md#numerical-qualification-and-training).
It does not broaden Model admission or substitute unqualified numerical formulas: every
unsupported unit keeps its preplanned ordinary CPU realization. API preferences do not force
provider selection. New local expanded-F32 source evidence is distinct from the earlier sealed
artifact and cross-platform receipts below; no new release or performance claim follows.

## CPU operation-audit closure

Local qualification covers all 137 ordinary signatures across 115 kinds, with general Scan
recorded separately. The current catalog completed 804 forward and 874 selected first-order AD
runs without an execution-stage failure. Its 760 selection rejections are confined to low-storage
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

Pool3d completion independently qualifies CPU F32/F64 and Metal F32/BF16/F16 forward and
actual backward through shared pooling adjoints and `UNFOLD3D`/`FOLD3D`, without new standalone
backward kernels or fallback. The earlier F32 Metal and both CPU types passed independent
spatial-scatter and finite-difference oracles, plus Conv3d → pool → flatten → linear
all-parameter finite differences, three SGD steps and save/close/load with bit-exact state and
fresh configured SGD. Current Metal low-type owned numerical journeys are recorded below.
CPU Scan passed F32/F64, max/average, all-valid/runtime-length and both traversal directions,
including actual backward input-role finite differences and save/close/load/fresh execution.

These are locally runtime-qualified capabilities, not joined-release or published-candidate
acceptance. Ordinary public CPU named-B admission, arbitrary layouts and performance remain
bounded by their family contracts. The low-parity cutover below supersedes the historical CPU
low-arithmetic lock; these earlier volumetric receipts do not themselves qualify low gradients.
Metal FLOAT64 pooling/folds retain their qualified non-overlap limits; Metal F32/BF16/F16
overlapping pooling adjoints use shared output-owned FLOAT32 working arithmetic.

## Temporary Java low-computation lock

FP8 means an eight-bit floating-point storage format; E4M3 and E5M2 split their non-sign bits
between exponent and fraction differently, so their ranges and special values differ.
**Working arithmetic** is the precision used during calculation, while **narrowing** converts
the result to a smaller stored format. CPU support for those stored formats does not imply a
native FP8 instruction, a Metal FP8 route, or a qualified device capability.

This anchor is retained for existing links; the CPU0078 lock is historical and superseded by
the [CPU low-precision parity cutover](../architecture/contracts/backend-execution.md#cpu-low-precision-parity-cutover).
BFLOAT16, FLOAT16, FLOAT8_E4M3 and FLOAT8_E5M2 now reuse ordinary CPU numerical owners,
shared Compiler first-order AD, fixed/general/length-aware Scan, qualified symbolic geometry
and Training. This is not a second executor, hidden quantization scale or runtime unlock.
Each family still checks its exact dtype/promotion, shape, layout, value, resource and alias domain;
functional parity is not universal geometry, strategy or performance qualification.

| Boundary | Current CPU low behavior |
|---|---|
| Storage | BF16/F16 use raw shorts; E4M3/E5M2 use raw bytes, never BOOL normalization. |
| Numerical operations | Homogeneous low outputs retain their type; FLOAT32 working arithmetic and accumulators end in one final RNE (round-to-nearest, ties-to-even) narrowing per ordinary operation/public/saved boundary. Unlike low types promote to F32; F64 dominates. Existing F32/F64 behavior remains. |
| Exact operations | Raw movement, same-type casts, ABS/NEG, comparison, ordering, winner/WHERE selection, indices, masks and RNG state retain exact contracts. Arithmetic NaN payload latitude does not apply to selected/copied stored NaN bits. |
| Casts and FP8 range | Direct F64/integer casts round directly to the target. Widening preserves representable NaN sign/payload. E4M3 has max 448 and no infinity: the finite 464 tie rounds to 448, above it/infinity input becomes canonical NaN 0x7f. E5M2 has max 57344 and signed infinity; numeric NaNs narrow to 0x7e. Same-type/raw import preserves every bit. |
| Extrema | Nonempty extrema select an original represented winner; an unrepresentable E4M3 infinity sentinel must not poison it. Empty infinity results narrow to E4M3 NaN. |
| Training | Low parameters have F32 master/optimizer values, explicitly F32/F64 scalar objectives, final logical narrowing and transactional nonfinite-candidate rejection/rollback. |
| Native MATMUL | Qualified typed low inputs, strict F32 native stage, then generated final RNE narrowing; bare K > 1 geometry only. Primitive/Graph/API/ISA availability is independent. Graph 3.14 rejects FP8; no native FP8 on Arm is claimed. Unsupported native recipes leave ordinary CPU selected cold, never runtime fallback. F64 remains ordinary/OpenBLAS, not oneDNN. |

The historical [CPU0078 receipt](../planning/backends/cpu/tasks/0078-temporary-low-arithmetic-lock.md#result)
records the earlier rejection/retention gate and six-step F32-master Training consumer, not today's
low admission. The public cutover smoke passed 205 numerical scenarios across all four low types:
arithmetic/shared Compiler AD, Pool2d/3d, changing-B (including B=0), length-aware Scan with
inactive NaNs, gradients, repeated SGD, bit-exact save/close/load and fresh SGD. It also covered
all 256 FP8 raw encodings through transpose and 12 unlike-low ADD promotions. Another 80
numerical scenarios passed active RNN/GRU/LSTM forward and actual Compiler gradients in both
directions, plus owned runtime-length Model save/load, for every low type.
The nine-type operation catalog completed 6,872 rows across all 115 ordinary kinds and 137
signatures: 1,398 forward and 2,160 actual AD runs, with no preparation/execution-stage gaps.
Illegal dtype/body projections and nondifferentiable roles remain separate classifications.
Catalog execution is not an independent numerical oracle for every row.
Three configured policies with changed weights confirmed F16 native AUTO/PRIMITIVES/GRAPH and
BF16 native AUTO/PRIMITIVES; BF16 GRAPH kept ordinary execution for that shape. Both FP8 types
stayed ordinary CPU on M3 under all policies. Local descriptor diagnostics returned status 3 for
FP8 and F64. The final repository `test build` gate and Javadoc for CPU, Metal, Model,
Compiler, Engine, NN, Training and Checkpoint passed with the configured native libraries.
This is source qualification, not rebuilt-release acceptance. No native FP8 execution,
unbounded geometry or performance qualification is claimed.

## Owned source and persistence boundaries

MLP, default persistable CNN1d/2d/3d and runtime-length tanh RNN support compute, one configured
training session, repeated fresh batches, save, close, load and newly configured training in
their qualified CPU domains. Earlier Metal F32 MLP/CNN1d/2d/3d and runtime-length general-Scan
RNN passed actual-GPU parameter updates/save/load/fresh training without CPU fallback. Current
F32/BF16/F16 owned MLP, Conv3d/overlapping Pool3d and explicit GeneralScan RNN evidence is below.

Tensor-level numerical support is broader than cold acquisition/serialization. Within the
existing recipe domain, cold reconstruction accepts Model-legal per-role floating types and casts.
Any source containing FP8 selects graph domain v7, with dtype wire IDs 8 (E4M3) and 9 (E5M2).
Old dtype IDs 1–7 and non-FP8 smallest-sufficient v1–v6 bytes remain unchanged: Pool3d selects
v6 when FP8 is absent. NN FP8 persistence uses conditional model.v2/SYNMOD02,
definition SYNDEF02 and whole-model.v2 formats; non-FP8 values retain old compatible formats.
Strict descriptor rebind remains. Arbitrary PURE Scan execution does not automatically widen
cold reconstruction or add wire opcodes. Fixed recipes retain their distinct working/publication boundaries.
No optimizer/history/compiled-plan continuation is promised.

## Metal production target

Metal is required and is **not experimental**. Its independent target includes qualified static
F32/BF16/F16 ordinary numerical families, Conv3d/overlapping Pool3d training, attention, native
GeneralScan and fixed RNN/GRU/LSTM selected first-order derivatives. Exact/raw F64/integral rows
remain separate: stored F64 movement is not F64 numerical arithmetic. General F64 neural
arithmetic, unsupported mixed-low domains and unqualified higher-order recurrent derivatives
remain exclusions, not CPU fallback paths. The [Metal guide](metal-backend.md) and
[owning contract](../architecture/contracts/backend-execution.md#metal-backend) define exact limits.

The prior operation catalog covered 5,018 rows, all 115 ordinary kinds and all 137 legal signature
fixtures: 744 forward and 494 actual selected first-order AD runs, with zero execution-stage gaps.
Its 924 selection rejections are historical baseline classifications, not current cutover results;
Model-illegal projections and nondifferentiable/deferred roles were counted separately. In
particular, its seventeen-input CONCAT/STACK exclusion is obsolete. Current composition uses
dynamic operand references and one disjoint direct-copy child per operand through the same
ordinary/Scan owner, not N-wide argument binding or an intermediate pairwise tensor chain.
The common rank-sixteen and checked geometry/span/dispatch/workspace bounds remain. Fresh
complete per-operation catalogs contain 5,018 rows per backend. Metal completed 908 forward and
1,156 selected first-order AD runs without an execution-stage failure. Each F32/BF16/F16 projection
has the same 187 forward and 358 AD cases, without a Model-valid selection gap; the 374 Metal
selection rejections are confined to the intentionally unsupported numerical F64 domain.

Independent Metal-only numerical proof passed 93 F32 forward/actual all-role derivative cases:
81 ordinary cases, eight Pool3d Scan cases and four signed-update/overlapping-axis-fold,
scalar-reciprocal and repeated-STACK accumulation edges. Offset-view binary arithmetic uses
the existing physical-stride kernels; packed canonical operands retain MPSGraph.
Another 136 exact-domain cases passed, including all 49 casts, four floating WHERE cases,
stored F64 decisions and integral modular/promotion paths. Numerical correctness is not inferred
from catalog hashes. Public runtime-length RNN separately passed all-parameter finite
differences, repeated SGD with changing lengths, all-inactive rows, invalid-length nonmutation/
retry, save/close/load raw-exact state and fresh configured SGD on a Metal-only Engine.

Current genuine M3 Max component evidence exercises the original F32/BF16/F16 carriers through
shared typed storage/POD roles, FLOAT32 working arithmetic and per-output publication. The
separate low-precision library is retired in favor of common owners and
`synaptik_numeric_codec.h`; there is no FLOAT32 tensor proxy, CPU fallback, FP64
arithmetic/emulation. This Metal receipt does not qualify the independent CPU low-parity cutover.

- CONCAT/STACK passed 17/33/257 inputs in all three types, including original-carrier views and
  last-operand-only AD. Zero-output CONCAT/STACK empty cotangents passed public GPU
  run/materialize. Signed raw SLICE passed all seven carriers; INT32/INT64 cumulative operations
  use exact modular arithmetic.
- Fixed RNN/GRU/LSTM first AD passed 36 public numerical cases (three types, three cells, both
  directions and both bias choices), covering every floating input/state/weight/bias role,
  lengths 0/1/3 and inactive NaNs against independent FLOAT32 finite differences and typed scalar
  adjoints. The fixed-cell builder retains its own working/publication rounding schedule.
- Owned MLP and Conv3d with overlapping average/max Pool3d passed nine cases across the three
  types: independent forward/MSE/input-and-parameter AD/SGD, save/close/load/rebind and fresh
  training. This is bounded numerical journey evidence, not universal shape/layout proof.
- Explicit NN GeneralScan passed twelve cases across the three types, both directions and bias
  choices, lengths `[4,2,0]`/`[1,3,4]`, inactive NaNs, independent ordinary RnnCell unroll and
  parameter AD, save/load and fresh SGD. It uses the ordinary-cell typed path, not a fixed-cell
  compatibility fallback; the two paths' rounding semantics remain distinct.
- Native public GeneralScan with 33 distinct operands passed 36 three-family concat/stack,
  direction and activity configurations. Only the last invariant requires gradients; complete
  emissions/final carries/cotangents match independent exact-integer oracles, including inactive NaNs.
- Normalization passed 36 full-forward cases and 114 explicit-seed AD routes across the three
  families against independent double formulas and finite differences, including all public
  BatchNorm training siblings and correction-one running variance.
- Dropout passed 18 dtype/mode/probability configurations against independent published-mask
  forward/VJP equations, exact dropped zeros with NaN payloads, replay, state advancement/counter
  wrap and fresh-input/seed reuse. Evaluation remains a Model identity, not a native random draw.

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

The earlier Conv3d extension passed grouped/bias/depthwise all-role finite differences and real
NN training/persistence in a Metal-only Engine containing no CPU backend. Current
F32/BF16/F16 FOLD_AXIS/FOLD2D/FOLD3D overlap addition is output-owned and deterministic, with
FLOAT32 working arithmetic and per-output typed publication, no atomics, scratch, host
fallback/repair or software FLOAT64. FLOAT64/integral folds retain qualified non-overlap limits.
Signed SLICE materializes canonical storage for all seven carriers; it does not authorize
negative physical strides. These shared owners close the qualified selected movement adjoints.
The current coordinated identity is 43, program schema22/`SM22`/header124, native ABI7/thirteen
exports, operation wires1..116 and attribute wires0..42. Shared native source has 23 authenticated
components and 188,490 UTF-8 bytes, SHA-256
`9d0d08724f5deac665bb4e09c08af3b0fad08efff050dad2d8ae265f39a030d4`.
The genuine M3 Max package builds with `-Wall -Wextra -Werror`, signs and packages; this is
component evidence, not a new sealed release or installation acceptance.

The earlier component source was built, ad-hoc signed, ZIP-extracted outside checkout and
reverified before its GPU gate. The supported reproducible local source-build installation uses
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

Prior GeneralScan composition/typed activity and CPU/Metal neural replay have accepted component
evidence. The current typed cutover has passed complete operation catalogs and the root gate;
versioned source/artifact freeze and clean installed API replay are separate acceptance receipts.
Short composition timings remain report-only observations with warmup/host/storage limits,
not a speed or zero-allocation promise.
The owner chose private/internal distribution: no public licence or registry publication is
selected. Cephes/Gradle notices remain packaged; they do not grant public rights to the project's
own code. Final candidate acceptance is recorded separately against its actual frozen artifacts.

## oneDNN and exact singleton root qualification

The integrated macOS arm64/M3 Max source passed unfiltered `test build --continue` plus CPU
and Metal Javadocs on JDK 26.0.1. The final command reported 159 actionable tasks: one executed
and 158 up-to-date. The complete retained inventory contains 793 JUnit suite reports:
5,920 tests, 64 skips, zero failures and zero errors. This accepts retained successful reports;
it does not claim every test was freshly executed in the last command.

Ten real oneDNN native-contract cases and twelve public Engine cases passed, including
same-address current weights, dense/transposed/broadcast and guarded-gap layouts, exact point
identities, separate epilogues and concurrent run isolation. Public PRIMITIVES/GRAPH/AUTO
MLP/CNN forward/AD/SGD/save/load/fresh-training consumers passed independent numerical oracles
with actual native execution records. The [CPU guide](cpu-backend.md#evidence-and-report-only-benchmarking)
records exact target identity, numerical exclusions and report-only complete-plan timings.

The rebuilt, locally signed and verified Metal package passed actual F32/BF16/F16 singleton
publication, ordinary/vector/batched/transposed and controlled Scan numerical checks. Its
library SHA-256 is `f9f18b2378403b65629de1900b21235b3a44e4e06b709bac81263e9904bf6334`;
semantic identity is 43, with ABI7/schema22 unchanged.

### Linux x86-64 qualification

The independent [Actions run](https://github.com/pho001/Synaptik/actions/runs/41663584505)
at `5b6a10af` authenticated and built upstream oneDNN 3.14.0 SEQ+Graph, then passed the complete
CPU suite and public Engine oneDNN qualification: 247 suites, 1,613 tests, 51 optional skips,
zero failures and zero errors. The ten native-contract and twelve public oneDNN cases all
executed without skips. The report-only whole-plan baseline also passed its independent
forward/AD checks; exact target, library identity and timings are in the
[CPU guide](cpu-backend.md#evidence-and-report-only-benchmarking).

The passing full-suite target is EPYC 7763/AVX2. Earlier EPYC 9V45/AVX512 evidence proves the
22 oneDNN cases, not a passing complete CPU gate. Original failed runs and their recovery remain
in the qualification receipt; arithmetic NaN payloads, generated-code hashes, default vector
widths and host-specific output paths are not numerical contracts.

### oneDNN installed-source and artifact replay

The eighteen-product 0.1.0 distribution was checksum-verified and extracted outside any checkout.
A fresh Maven consumer declaring only `nn:0.1.0` resolved sixteen runtime JARs in an isolated
cache; every JAR matched its installed repository bytes. A separate standard Gradle consumer
compiled and ran the documented oneDNN example, producing `64.0` without checkout classes.

The extracted NoGit source built all eighteen products with the build cache disabled and all
eighteen classes tasks executed. The final exact-path inventory selects 2,603 source files after
the portability cleanup; `source-provenance.json` binds their individual hashes.
Its independently rebuilt, locally signed and verified Metal image is byte-identical to the
installed binary image with the current `f9f18b...` SHA-256 recorded above.

Both installations passed the same twelve public programs: all three oneDNN API-policy
MLP/CNN forward/AD/SGD/save/load/fresh-training workflows, both primitive/Graph special-value
workflows, the canonical example, exact three-carrier Metal point products, fixed recurrent
all-role AD, original-carrier owned models, length-aware GeneralScan, normalization and dropout.
Their native execution records and independent numerical checks are retained under `qualification/`.
Neither classpath contains checkout classes, previous product JARs or test fixtures.
This installed replay is bounded local technical acceptance, separate from the Linux gate above:
not a universal geometry claim, performance SLA, CPU low-arithmetic unlock or registry/signing authorization.


## Prior shared typed Metal root qualification

This prior receipt and its source/artifacts are preserved unchanged under
`build/release/0.1.0-before-onednn-20261103T143942Z/`. Its counts and native hash below do not
describe the later oneDNN/singleton cutover.

The source checkpoint through `583b6495` passed unfiltered `test build --continue` on the
actual M3 Max with that checkpoint's authenticated native package. The final gate reported 159
actionable tasks: two executed and 157 up-to-date; the preceding repaired full gate also passed
with nine executed and 150 up-to-date. This accepts retained successful reports, not a claim
that every test was freshly executed in the final command. Earlier failed commands remain
failed evidence.

The current numerical, operation-catalog, wide Scan, owned training and cold-replay evidence
above is distinct from the ten optional native dispatch-observer skips. CPU low arithmetic,
general Metal F64 arithmetic, arbitrary geometry/higher-order closure and trusted public
prebuilt publication remain outside this acceptance.

The complete retained inventory contains 796 JUnit suite reports: 5,916 tests, 65 skips,
zero failures and zero errors. Metal contributes 580 tests with ten optional observer skips.

### Isolated versioned distribution acceptance

The eighteen-product 0.1.0 Maven/binary/exact-source distribution was rebuilt from the qualified
source, checksum-verified and extracted outside the checkout. A new Maven consumer declaring
only `nn:0.1.0` resolved sixteen runtime JARs in an isolated cache; every JAR matched its published
bytes. A separate extracted NoGit source tree built all eighteen products with build cache
disabled and all eighteen classes tasks executed. Neither runtime classpath contains checkout
classes or previous product JARs.

Both installations passed the same original-carrier F32/BF16/F16 public qualifications:
fixed RNN/GRU/LSTM all-role AD (36 configurations), owned MLP/Conv3d-overlapping-Pool3d
forward/AD/SGD/cold reload (nine models), explicit NN GeneralScan lengths/training/cold replay
(twelve configurations), normalization (36 forward/114 selected-AD routes) and Dropout
mask/state/reuse/VJP (eighteen configurations). Independent numerical oracles are described above.

The extracted source tree independently rebuilt, locally signed and verified its native package
before actual GPU replay. Its library is byte-identical to the installed binary library:
SHA-256 `03df3cc3196295a22329c2bfb48780818641a6c118f7db30d4f4e8eb3b3c6ed9`.
That preserved package uses semantic identity42 and source SHA-256
`f1f2429fdbb7a87c3bacd5358564f4abffca3aac2fc4d3e1e86429c5ca6dff35`, not the later
singleton-correction corpus; its ABI7/schema22 are unchanged.

The final documentation-only freeze retains the qualified non-documentation and binary bytes.
`source-provenance.json`, `SHA256SUMS`, `verification.json` and `qualification/` bind the final
archive source, artifact identities, commands, retained reports, failure/recovery history and
installed replay. This is local technical acceptance, not a performance SLA, CPU low unlock,
general Metal F64 arithmetic, public registry upload or official native signing claim.

## Accepted local candidate: Pool3d and operation-audit closure

This earlier candidate and its original receipts were preserved unchanged under
`build/release/0.1.0-before-metal-parity-20261103T112058Z/`. The paths and counts below describe
that archived checkpoint, not the current typed candidate.

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

## Public-readiness audit: current source and installed candidate

The framework-wide [audit](../planning/public-release-audit-2026-11-04.md#result) is complete.
Its local candidate is `0.1.0-public-audit.20261104`, separate from the historical `0.1.0`
archives above. Public publication is still blocked by the owner's retained no-licence/
no-redistribution decision. Qualification is local macOS arm64 with JDK26, actual Metal and the
explicit oneDNN 3.14.0 LP64 SEQ library; no x86 or trusted-public-prebuilt claim follows.

| Evidence | Observed result and boundary |
|---|---|
| Joined root gate | `test build` and affected Javadoc: BUILD SUCCESSFUL; 159 actionable tasks, 22 executed and 137 up-to-date. |
| Active-layout JUnit receipts | 790 suites, 6,012 tests, 67 skips, zero failures/errors. These accepted reports include up-to-date tasks; they are not a claim that every test was freshly executed. |
| Metal module | 624 tests, ten skips, zero failures/errors. Independent actual-GPU public scenarios are additional evidence, not inferred from the suite count. |
| Ordinary CPU workflow | 205 exact public checks across BF16/F16/E4M3/E5M2: arithmetic, shared Compiler AD, Pool2d/3d, changing/zero batch, length-aware Scan, repeated SGD and save/load with fresh training. Includes configured oneDNN policies where independently eligible. |
| CPU recurrent workflow | 80 checks: active RNN/GRU/LSTM, both directions, actual Compiler gradients and length-aware Model save/load across all four low types. |
| Metal recurrent workflow | 40 corresponding real-device checks across BF16/F16, both directions and persistence. No FP8 Metal domain is admitted. |
| Targeted public repros and diagnostics | Overflow versus L2 point identity, categorical zero gradients, exact pooling divisor bits, contraction-before-bias, mixed-owner excess capacity, transactional restore, deep Scan/namespace and session-only tuning. CPU cold/fallback observers and native bounds/duplicate/BOOL failures preserve actual context and native causes. |
| Fresh Maven installation | All 18 Java products published with binary/source/Javadoc artifacts. A consumer declaring only `nn` resolved 16 JARs from a fresh Maven cache and passed the same public programs using only extracted candidate native code. |
| Extracted source installation | All 2,632 selected source hashes matched. All 16 Java compilation actions executed with project build caching disabled; native source rebuilt, ad-hoc signed and package-verified. The same public programs passed against those independently built Java/native outputs without checkout runtime paths. |

The supported numerical, geometry, ownership and persistence boundaries remain the owning
contracts and family-specific matrix above. The audit repaired defects; successful operation
catalog rows still do not independently certify every legal value, shape or numerical outcome.
Native FP8 is unavailable on this Arm target, so both FP8 formats stay ordinary CPU, without
advertising widened FLOAT32 input as native FP8. Metal general F64 arithmetic remains excluded.
No optimizer continuation, exhaustive allocation-failure injection or universal performance
qualification is implied.

`build/release/0.1.0-public-audit.20261104/verification.json`, `SHA256SUMS` and `qualification/`
bind exact source/artifact hashes, commands, current-layout reports, exercised source programs,
installed/source output and failed-gate recovery history. The final documentation-only refresh
must retain the 1,813 frozen non-Markdown source hashes, all 18 runtime JARs and the installed
native-library fingerprint; record the comparison without pretending to re-execute every test.

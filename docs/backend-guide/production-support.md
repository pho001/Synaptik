# First-production support [matrix](../glossary.md#term-review-matrix)

## Status and interpretation

Current public-readiness source and isolated candidate [qualification](../glossary.md#term-qualification), 2026-11-04, are recorded in
the [audit receipt below](#public-readiness-audit-current-source-and-installed-candidate).
The earlier bounded private/internal `0.1.0` [matrix](../glossary.md#term-matrix-meanings) retains its own [operation](../glossary.md#operation)-catalog, root and
isolated source/[artifact](../glossary.md#term-artifact)/consumer evidence. None is a claim about every [Model](../glossary.md#term-model) geometry, public
[registry](../glossary.md#term-registry) [publication](../glossary.md#term-publication) or a trusted native prebuilt; [sealed](../glossary.md#term-sealed) receipts apply only to their frozen bytes.
[General Scan](../glossary.md#general-differentiable-scan) and the bounded [CPU](../glossary.md#term-cpu)/[Metal](../glossary.md#term-metal) neural journeys have component evidence. The
[product goal](../planning/production-release-goal.md) and
[owning contracts](../../ARCHITECTURE.md#scope-indexed-normative-contracts) control semantics.
A source predicate, a generated-entry test and an actual [Compiler](../glossary.md#term-compiler) backward [run](../glossary.md#term-run) are different evidence.

Post-candidate source work is recorded in the
[interchangeability follow-up](backend-interchangeability-audit.md#15-navazující-opravy-a-aktuální-hranice)
and [public-release audit](../planning/public-release-audit-2026-11-04.md#result)<span data-glossary-navigation="true"> ([release glossary](../glossary.md#term-release))</span>.
Neither relabels the earlier [distribution](../glossary.md#term-distribution) as rebuilt or reissued.

Read each row as bounded evidence, not a promise for every [tensor](../glossary.md#tensor). A **domain** is the allowed
combination of numerical types, [dimensions](../glossary.md#dimension), storage mappings and values. **Qualification**
records which implementation and environment were checked. A **catalog** exercises representative
cases; an independent **[numerical oracle](../glossary.md#adr-ci-and-qualification)** calculates an expected answer without reusing the
implementation being checked. Passing a catalog and agreeing with an [oracle](../glossary.md#term-test-oracle) are different claims.

## Common Java [CPU](../glossary.md#term-cpu) domain

[F32](../glossary.md#floating-point-precision-and-rounding)/[F64](../glossary.md#floating-point-precision-and-rounding) abbreviate the [FLOAT32](../glossary.md#term-float32)/[FLOAT64](../glossary.md#term-float64) storage formats; [BF16](../glossary.md#floating-point-precision-and-rounding)/[F16](../glossary.md#floating-point-precision-and-rounding) abbreviate [BFLOAT16](../glossary.md#term-bfloat16)/[FLOAT16](../glossary.md#term-float16).
**[Homogeneous](../glossary.md#term-homogeneous)** means the relevant numerical values use one common type. A **[stride](../glossary.md#term-stride)** is the
storage step for advancing one [coordinate](../glossary.md#axis-rank-scalar-vector-and-matrix), an **offset** is the mapping's starting displacement,
and an **[injective](../glossary.md#term-injective)** output mapping gives every distinct logical coordinate a distinct address.
These properties matter even when two [tensors](../glossary.md#tensor) have the same [shape](../glossary.md#shape).

FLOAT32 and FLOAT64 are equal homogeneous production [targets](../glossary.md#term-target). The common baseline uses fully static
shapes, exact [operation](../glossary.md#operation) geometry/promotion, resolved nonnegative read strides/offsets, checked
referenced byte spans and injective outputs. [Binding](../glossary.md#term-binding) rejects illegal writable overlap. A family can
impose narrower [layouts](../glossary.md#layout), extents or value guards; this table does not promise every [Model](../glossary.md#term-model) layout.
[Detached](../glossary.md#detached-model-results) [host](../glossary.md#term-host) publications are [canonical](../glossary.md#term-canonical-form). [Scalar](../glossary.md#term-scalar), [vector](../glossary.md#term-vector) and [synchronous](../glossary.md#term-synchronous) parallel strategies remain
operation/access/size qualified; equal support is not equal [lane](../glossary.md#term-lane) counts, thresholds, speed or bits.
Default scalar execution needs no [incubator module](../glossary.md#term-incubator-module). See [JVM prerequisites](../user-guide/running-models.md#jvm-prerequisites)<span data-glossary-navigation="true"> ([JVM glossary](../glossary.md#term-java-virtual-machine-jvm))</span>.

The static baseline does not remove advanced symbolic paths. Standalone ordinary symbolic
[MATMUL](../glossary.md#matrix-multiplication--matmul)/[unary](../glossary.md#term-unary) retain their family [qualification](../glossary.md#term-qualification). [General Scan](../glossary.md#general-differentiable-scan) separately composes static and
qualified one-B all-valid/length bodies in both floating types through ordinary [owners](../glossary.md#term-owner), with
typed discrete intermediates and proved coordinate semantics. This is not arbitrary dynamic-
[dimension](../glossary.md#dimension) or mixed-floating backward parity; ordinary family and selected-[VJP](../glossary.md#term-vector-jacobian-product-vjp) predicates remain.

| Family | Common static F32/F64 domain | First-order closure and current [release](../glossary.md#term-release) evidence |
|---|---|---|
| [Pointwise](../glossary.md#term-pointwise) and selection | Floating tensor/scalar arithmetic, 19 unary kinds, clamp, comparisons, [BOOL](../glossary.md#data-type--datatype) logic and WHERE under exact [dtype](../glossary.md#data-type--datatype)/broadcast rules; Model-valid promoted F32/F64 and I32/I64 tuples | Original [cotangent](../glossary.md#cotangent) [carrier](../glossary.md#term-storage-carrier) conversion and exact stored-value comparisons passed public mixed-operand and WHERE witnesses. Discrete conditions receive no cotangent. Low arithmetic now follows the parity section below; these historical F32/F64 receipts do not qualify it. Actual two-layer [ReLU](../glossary.md#term-relu) [MLP](../glossary.md#term-multilayer-perceptron-mlp) forward/[SGD](../glossary.md#term-stochastic-gradient-descent-sgd)/save/load/new-train passed in both types. |
| MATMUL/[linear](../glossary.md#term-linear-meanings) | Vector/[matrix](../glossary.md#term-matrix-meanings) [rank](../glossary.md#term-rank) cases and broadcast [batches](../glossary.md#term-batch-machine-learning-and-hyperparameters); direct/tiled scalar, eligible N-vector and range parallelism | Ordinary rank edits, [transpose](../glossary.md#term-transpose) and [SUM-to-Shape](../glossary.md#sum-to-shape) [derivatives](../glossary.md#term-derivative). Homogeneous MLP/[RNN](../glossary.md#term-recurrent-neural-network-rnn) paths passed; optional qualified [OpenBLAS](../glossary.md#term-openblas) is not proof of portable Java coverage. |
| Conv1d/[Conv2d](../glossary.md#convolution--conv2d) | [NCW](../glossary.md#term-ncw)/[NCHW](../glossary.md#term-nchw); [bias](../glossary.md#term-bias)/no bias, groups/depthwise, stride/[padding](../glossary.md#padding)/[dilation](../glossary.md#term-dilation), family-qualified noncanonical reads | Actual dX/dW/db passed 16 public cases across both types, grouped/dilated/strided, depthwise multiplier and asymmetric geometry. Both ranks passed complete [owned Model](../glossary.md#model-module-buffer-and-scan-distinctions) journeys. |
| Max/AveragePool1d/2d | Exact floor/literal-ceil grids, padding/dilation, fixed average divisor, overlap | Actual backward passed 24 public cases: [finite](../glossary.md#term-finite) overlap, dilated ceil tails, [NaN](../glossary.md#term-nan), [signed zero](../glossary.md#term-signed-zero), real -[infinity](../glossary.md#term-infinity) and all-padding windows. Uses existing winner/fold rules, not a new derivative. |
| [Conv3d](../glossary.md#conv3d--ncdhw) | Fully static resolved-layout [NCDHW](../glossary.md#conv3d--ncdhw); bias/no bias, groups/depthwise, stride/padding/dilation and overlapping windows | Shared Conv2d/Conv3d [AD](../glossary.md#autograd); actual dX/dW/db and selected [weight](../glossary.md#term-weight)-only [gradients](../glossary.md#term-gradient) passed independent direct-loop central [finite differences](../glossary.md#arithmetic-rearrangement-and-finite-difference-checks) in both types. Grouped biased batch-two anisotropic and unbiased depthwise cases passed; both types also passed three-step [parameter](../glossary.md#term-parameter) training/save/close/load/new-train. See the bounded volumetric qualification below. |
| Max/AveragePool3d | Fully static resolved-layout NCDHW; matching input/output gradient flags in F32/F64, checked [kernel](../glossary.md#kernel)/stride/padding/dilation and literal ceil mode | Both types passed actual forward and explicit-cotangent gradients against independent [spatial](../glossary.md#term-spatial-axes) scatter and per-output finite differences, including batch/[channel](../glossary.md#term-channel) overlap and anisotropic geometry. Existing shared [pooling](../glossary.md#term-pooling) adjoints and 3D windows close trainable Conv3d → pool → flatten → linear, all parameter roles, three SGD steps and save/close/load/fresh configured training. |
| SUM/[MEAN](../glossary.md#term-mean)/PROD/MIN/MAX and SUM-to-Shape | Full/single/multiple [axes](../glossary.md#term-axis), exact keep/remove axes and target shape | Both types passed actual axis SUM/MEAN/PROD gradients, including zero products. MIN/MAX tie/other geometry qualification remains separate; generated cumulative/layout closure is not inferred. |
| Masked/statistical reductions | Masked SUM/MEAN, LOG_SUM_EXP, [variance](../glossary.md#term-variance)/std, [L1](../glossary.md#term-l1)/[L2](../glossary.md#term-l2); correction/domain guards | Both types passed actual masked SUM/MEAN, LOG_SUM_EXP, corrected variance/std and L1/L2 against independent scalar forward/derivative oracles. BOOL [masks](../glossary.md#term-boolean-mask) have no cotangent; wider domains remain qualified separately. |
| Cumulative operations | Axis, direction and inclusive/exclusive CUM_SUM/CUM_PROD | Both types passed all inclusive/exclusive x forward/reverse combinations with nonuniform cotangents and zero-product inputs. Uses ordinary rules; not a universal layout/strategy claim. |
| [SOFTMAX](../glossary.md#softmax--log-softmax)/LOG_SOFTMAX | Nonempty selected axis, matching type/shape, finite input domain | CPU0077 parity repair accepted: both types admit finite negative-[overflow](../glossary.md#term-overflow) shifts. 45 targeted tests/Javadocs and actual public both-kind/type forward/[Compiler](../glossary.md#term-compiler)-backward passed; tied maxima give nonzero gradients, with pre-write invalid/freshness/[loss](../glossary.md#loss) controls. |
| LayerNorm/RMSNorm | Static trailing [normalized shape](../glossary.md#normalized-shape), typed [epsilon](../glossary.md#epsilon) and optional affine roles | Actual both-type affine LayerNorm three-role and RMSNorm two-role gradients passed independent scalar oracles. Wider normalized shapes/layouts retain family predicates. |
| BatchNorm [inference](../glossary.md#term-inference)/training | Exact channel geometry; training contributor minimum and five output roles | Both types passed all five input-role gradients for inference and the joint three public training roots (normalized output/next mean/next variance). Saved auxiliaries remain invalid independent derivative roots. |
| [MSE](../glossary.md#mean-squared-error--mse) and dense/index cross entropy | NONE/SUM/MEAN, exact target [signatures](../glossary.md#term-signature), ignore/exclusion guards | Both types passed actual MSE/dense CE data+target gradients, including nonunit [dense target](../glossary.md#dense-target) sums, and index CE with ignore control. Integral targets are nondifferentiable; other reductions/domains retain their own qualification. |
| Affine/movement/windows | Declared reshape/rank edits/permute, SELECT/checked signed-step [SLICE](../glossary.md#slice), [PAD](../glossary.md#padding)/TILE, bounded one-pass CONCAT/STACK/replacement, UNFOLD_AXIS/2D/3D and FOLD_AXIS/2D/3D | Signed/empty [materialization](../glossary.md#materialization) and negative-step extraction/update adjoints passed. Seventeen-input CONCAT/STACK passed every-input and repeated-[identity](../glossary.md#term-identity-meanings) cotangents; composed 65-input consumers also passed within checked generation limits. Raw movement covers all seven carriers. UNFOLD3D preserves raw F32/F64/BF16/F16 words; arithmetic FOLD3D remains F32/F64. Static noncontiguous RESHAPE materializes through the existing ordinal copy; the low-arithmetic lock remains. |
| [Gather](../glossary.md#gather)/scatter/one-hot | Exact index shape and [INT32](../glossary.md#data-type--datatype)/[INT64](../glossary.md#data-type--datatype) bounds; declared NONE/ADD/MUL/MIN/MAX modes | Both types passed gather/gatherElements/gatherNd and all five scatterElements/scatterNd modes, data/update roles, duplicates and zero products. Bounds/uniqueness pre-write tests remain independent; no index cotangent. |
| Ordering and [extrema](../glossary.md#saturation-pole-and-extremum) | Stable SORT/ARGSORT/TOP_K and exact tie/index behavior | Both types passed actual SORT/TOP_K value gradients against scalar oracles; indices have no cotangent. Tie/special/other geometry qualification is not inferred from separated-value cases. |
| Attention | Rank>=2 Q/K/V, exact [contraction](../glossary.md#contraction-dimension) and broadcast batches, optional BOOL/[causal mask](../glossary.md#causal-mask) and normalized weights | Both types passed joint output/weights gradients for all Q/K/V roles, masked and unmasked. Literal [EXP](../glossary.md#term-expx) overflow/[underflow](../glossary.md#term-underflow) and nonempty negative-zero value contractions passed separately, including controlled single-output [Scan](../glossary.md#term-cumulative-scan) with inactive NaNs. Ordinary AD requires canonical two-output form; one-output construction is forward-only for AD. Masks have no cotangent, weights no V route. |
| Explicit [RNG](../glossary.md#graph-rng-state--graphrngstate)/[dropout](../glossary.md#dropout--inverted-dropout) | Existing state/mask/next-state signatures and typed value semantics | No derivative for state/mask. Scan purity must follow the actual effect contract, not assume every random operation is pure. |
| Fixed recurrent scans | Existing explicit RNN/[GRU](../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../glossary.md#term-long-short-term-memory-lstm) constructors and CPU forward [kernels](../glossary.md#kernel) | CPU evidence here covers forward execution, including shared first-product signs and both-direction RNN negative-zero sequence/final-state [publication](../glossary.md#term-publication). Shared fixed-cell first-order rules are no longer categorically deferred; actual [Metal](../glossary.md#term-metal) F32/BF16/F16 all-role proof is recorded below and does not itself qualify CPU fixed-cell training. Fixed recipes retain their own working/publication semantics rather than becoming GeneralScan recipes. |
| General Scan / practical RNN | Ordinary CPU-supported [PURE](../glossary.md#term-pure) bodies, static/qualified one-B F32/F64, per-value typed intermediates; all-valid whole-Tensor or length-mode proved ROW(axis)/SHARED semantics | Compiler0017/CPU0081 are integrated and source/consumer-qualified: ordinary [lowering](../glossary.md#lowering)/emission, actual selected VJPs, structural activity, guards, empty/T0 caller publication, reuse and genuine old-recipe replay. The bounded public RNN train/save/load/new-train workflow remains separately qualified. Metal Scan has independent qualification below; broader cold/wire support is not inferred. |

All entries retain [Model numerical semantics](../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics).
Finite tolerances do not replace exact discrete, special-value, saved-value or pre-write [validation](../glossary.md#term-validation).
Current exact predicates remain in CpuCapabilityProvider and the CPU lowerers; a family row is not
a declaration that arbitrary partitions or every possible backward target are eligible.

The optional [oneDNN CPU peer](cpu-backend.md#explicit-onednn-composition)<span data-glossary-navigation="true"> ([oneDNN glossary](../glossary.md#term-onednn))</span> retains its bare
F32 MATMUL baseline and adds qualified common F32 occurrences under the
[API/implementation/geometry limits](cpu-backend.md#numerical-qualification-and-training).
It does not broaden Model [admission](../glossary.md#term-admission) or substitute unqualified numerical formulas: every
unsupported unit keeps its preplanned ordinary CPU realization. [API](../glossary.md#term-application-programming-interface-api) preferences do not force
provider selection. New local expanded-F32 source evidence is distinct from the earlier [sealed](../glossary.md#term-sealed)
[artifact](../glossary.md#term-artifact) and cross-platform receipts below; no new release or performance claim follows.

## [CPU](../glossary.md#term-cpu) [operation](../glossary.md#operation)-audit closure

Local [qualification](../glossary.md#term-qualification) covers all 137 ordinary [signatures](../glossary.md#term-signature) across 115 kinds, with [general Scan](../glossary.md#general-differentiable-scan)
recorded separately. The current catalog completed 804 forward and 874 selected first-order [AD](../glossary.md#autograd)
runs without an execution-stage failure. Its 760 selection rejections are confined to low-storage
projections. Rejected forward kinds were classified under the intentional arithmetic lock;
selected low VJPs may also require locked arithmetic. Illegal [Model](../glossary.md#term-model) projections and
nondifferentiable/deferred roles are classified separately. These are bounded fixtures, not
every possible Model geometry; execution [hashes](../glossary.md#term-hash) are not numerical correctness proof.

Independent forward/[derivative](../glossary.md#term-derivative) oracles passed 178 cases: 81 ordinary cases per floating type
and sixteen [Pool3d](../glossary.md#pool3d--ncdhw-pooling) [Scan](../glossary.md#term-cumulative-scan) cases. Additional public witnesses cover promoted operands/original
[cotangent](../glossary.md#cotangent) types, signed slicing/update, [variadic](../glossary.md#term-variadic)/repeated [identities](../glossary.md#term-identity-meanings) and composite special values.
A separate generated TILE witness checked `[200001,2,3]`, [strides](../glossary.md#term-stride) `[8,3,2]`, offset `5`, split
ranges, [heap](../glossary.md#term-heap)/segment outputs and untouched holes. Current-quotient checks prevent cyclic CPU units.
Changed affine empty-range and literal-attention classes use selective projections 73/74;
the generator envelope stays 71. Source review approved the affected [owners](../glossary.md#term-owner). [Metal](../glossary.md#term-metal) evidence
and joined exact-source/[artifact](../glossary.md#term-artifact)/installation acceptance are recorded separately below.

## [Conv3d](../glossary.md#conv3d--ncdhw) training and independent [Pool3d](../glossary.md#pool3d--ncdhw-pooling) boundary

The user's explicit Conv3d training request supersedes the earlier 1D/2D-only [release](../glossary.md#term-release)-[rank](../glossary.md#term-rank)
decision. Conv1d/2d and Conv3d train through existing public [Tensor](../glossary.md#tensor) algebra; this is not a
[guard](../glossary.md#term-guard)-only unlock. The shared [convolution](../glossary.md#term-channels-first-convolution-layers) rule retains groups in every [matrix](../glossary.md#term-matrix-meanings) [contraction](../glossary.md#contraction-dimension):
input [gradients](../glossary.md#term-gradient) use transposed [weights](../glossary.md#term-weight) and overlap-add fold, weight gradients use unfold,
[matrix multiplication](../glossary.md#matrix-multiplication--matmul) and [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) reduction, and [bias](../glossary.md#term-bias) gradients reduce batch and all [spatial](../glossary.md#term-spatial-axes)
[axes](../glossary.md#term-axis). Each selected gradient returns to its [target](../glossary.md#term-target)'s exact [Shape](../glossary.md#shape) and [data type](../glossary.md#data-type--datatype).

Public [CPU](../glossary.md#term-cpu) [F32](../glossary.md#floating-point-precision-and-rounding)/[F64](../glossary.md#floating-point-precision-and-rounding) proof covers every input/weight/bias role for grouped biased batch-two
Conv3d with anisotropic [stride](../glossary.md#term-stride), symmetric [padding](../glossary.md#padding), [dilation](../glossary.md#term-dilation) and overlap, plus unbiased
depthwise groups-two geometry and selected weight-only gradients. Both types trained every
weight/bias for three steps, saved, closed and loaded bit-exact [parameters](../glossary.md#term-parameter), then configured
fresh training and passed another independent [oracle](../glossary.md#term-test-oracle)-checked [SGD](../glossary.md#term-stochastic-gradient-descent-sgd) step. CPU [general Scan](../glossary.md#general-differentiable-scan) also
composed actual Conv3d forward and reverse in eight independent [finite-difference](../glossary.md#term-finite-difference-gradient-check) cases:
all-valid/length-aware, both traversal directions, and F32/F64. There is no new 3D blacklist;
complete body/binder [qualification](../glossary.md#term-qualification) and [Scan](../glossary.md#term-cumulative-scan) higher-order restrictions still apply.

Pool3d completion independently qualifies CPU F32/F64 and [Metal](../glossary.md#term-metal) F32/[BF16](../glossary.md#floating-point-precision-and-rounding)/[F16](../glossary.md#floating-point-precision-and-rounding) forward and
actual backward through shared [pooling](../glossary.md#term-pooling) adjoints and `UNFOLD3D`/`FOLD3D`, without new standalone
backward [kernels](../glossary.md#kernel) or fallback. The earlier F32 Metal and both CPU types passed independent
spatial-scatter and finite-difference oracles, plus Conv3d → pool → flatten → [linear](../glossary.md#term-linear-meanings)
all-parameter [finite differences](../glossary.md#arithmetic-rearrangement-and-finite-difference-checks), three SGD steps and save/close/load with bit-exact state and
fresh configured SGD. Current Metal low-type owned numerical journeys are recorded below.
CPU Scan passed F32/F64, max/average, all-valid/[runtime](../glossary.md#term-runtime)-length and both traversal directions,
including actual backward input-role finite differences and save/close/load/fresh execution.

These are locally runtime-qualified capabilities, not joined-release or published-candidate
acceptance. Ordinary public CPU named-B [admission](../glossary.md#term-admission), arbitrary [layouts](../glossary.md#layout) and performance remain
bounded by their family contracts. The low-parity cutover below supersedes the historical CPU
low-arithmetic lock; these earlier volumetric receipts do not themselves qualify low gradients.
Metal [FLOAT64](../glossary.md#term-float64) pooling/folds retain their qualified non-overlap limits; Metal F32/BF16/F16
overlapping pooling adjoints use shared output-owned [FLOAT32](../glossary.md#term-float32) [working arithmetic](../glossary.md#accumulator-type).

## Temporary Java low-computation lock

[FP8](../glossary.md#term-fp8) means an eight-bit [floating-point](../glossary.md#term-floating-point) storage format; [E4M3](../glossary.md#term-e4m3) and [E5M2](../glossary.md#term-e5m2) split their non-sign bits
between [exponent](../glossary.md#term-exponent) and fraction differently, so their ranges and special values differ.
**[Working arithmetic](../glossary.md#accumulator-type)** is the [precision](../glossary.md#term-precision) used during calculation, while **[narrowing](../glossary.md#term-narrowing)** converts
the result to a smaller stored format. [CPU](../glossary.md#term-cpu) support for those stored formats does not imply a
native FP8 instruction, a [Metal](../glossary.md#term-metal) FP8 route, or a qualified [device](../glossary.md#term-device) capability.

This anchor is retained for existing links; the CPU0078 lock is historical and superseded by
the [CPU low-precision parity cutover](../architecture/contracts/backend-execution.md#cpu-low-precision-parity-cutover).
[BFLOAT16](../glossary.md#term-bfloat16), [FLOAT16](../glossary.md#term-float16), FLOAT8_E4M3 and FLOAT8_E5M2 now reuse ordinary CPU numerical [owners](../glossary.md#term-owner),
shared [Compiler](../glossary.md#term-compiler) first-order [AD](../glossary.md#autograd), fixed/general/length-aware [Scan](../glossary.md#term-cumulative-scan), qualified symbolic geometry
and Training. This is not a second executor, hidden quantization scale or [runtime](../glossary.md#term-runtime) unlock.
Each family still checks its exact [dtype](../glossary.md#data-type--datatype)/promotion, [shape](../glossary.md#shape), [layout](../glossary.md#layout), value, resource and [alias](../glossary.md#aliasing-and-arena) domain;
functional parity is not universal geometry, strategy or performance [qualification](../glossary.md#term-qualification).

| Boundary | Current CPU low behavior |
|---|---|
| Storage | [BF16](../glossary.md#floating-point-precision-and-rounding)/[F16](../glossary.md#floating-point-precision-and-rounding) use raw shorts; E4M3/E5M2 use raw bytes, never [BOOL](../glossary.md#data-type--datatype) [normalization](../glossary.md#term-normalization). |
| Numerical operations | [Homogeneous](../glossary.md#term-homogeneous) low outputs retain their type; [FLOAT32](../glossary.md#term-float32) working arithmetic and accumulators end in one final RNE ([round-to-nearest, ties-to-even](../glossary.md#floating-point-precision-and-rounding)) narrowing per ordinary [operation](../glossary.md#operation)/public/saved boundary. Unlike low types promote to [F32](../glossary.md#floating-point-precision-and-rounding); [F64](../glossary.md#floating-point-precision-and-rounding) dominates. Existing F32/F64 behavior remains. |
| Exact operations | Raw movement, same-type casts, ABS/NEG, comparison, ordering, winner/WHERE selection, indices, [masks](../glossary.md#term-boolean-mask) and [RNG state](../glossary.md#graph-rng-state--graphrngstate) retain exact contracts. Arithmetic [NaN payload](../glossary.md#nan-infinity-and-signed-zero) latitude does not apply to selected/copied stored [NaN](../glossary.md#term-nan) bits. |
| Casts and FP8 range | Direct F64/integer casts round directly to the [target](../glossary.md#term-target). [Widening](../glossary.md#term-widening) preserves representable NaN sign/[payload](../glossary.md#artifact-envelope-and-atomic-publication). E4M3 has max 448 and no [infinity](../glossary.md#term-infinity): the [finite](../glossary.md#term-finite) 464 tie rounds to 448, above it/infinity input becomes [canonical NaN](../glossary.md#term-canonical-nan) 0x7f. E5M2 has max 57344 and signed infinity; numeric NaNs narrow to 0x7e. Same-type/raw import preserves every bit. |
| [Extrema](../glossary.md#saturation-pole-and-extremum) | Nonempty extrema select an original represented winner; an unrepresentable E4M3 infinity sentinel must not poison it. Empty infinity results narrow to E4M3 NaN. |
| Training | Low [parameters](../glossary.md#term-parameter) have F32 master/[optimizer](../glossary.md#term-optimizer) values, explicitly F32/F64 [scalar](../glossary.md#term-scalar) objectives, final logical narrowing and transactional nonfinite-candidate rejection/[rollback](../glossary.md#term-rollback). |
| Native [MATMUL](../glossary.md#matrix-multiplication--matmul) | Qualified typed low inputs, strict F32 native stage, then generated final RNE narrowing; bare K > 1 geometry only. Primitive/[Graph](../glossary.md#graph)/[API](../glossary.md#term-application-programming-interface-api)/[ISA](../glossary.md#term-instruction-set-architecture-isa) availability is independent. Graph 3.14 rejects FP8; no native FP8 on Arm is claimed. Unsupported native recipes leave ordinary CPU selected cold, never runtime fallback. F64 remains ordinary/[OpenBLAS](../glossary.md#term-openblas), not [oneDNN](../glossary.md#term-onednn). |

The historical [CPU0078 receipt](../planning/backends/cpu/tasks/0078-temporary-low-arithmetic-lock.md#result)
records the earlier rejection/retention gate and six-step F32-master Training consumer, not today's
low [admission](../glossary.md#term-admission). The public cutover [smoke](../glossary.md#term-smoke-test) passed 205 numerical scenarios across all four low types:
arithmetic/shared Compiler AD, Pool2d/3d, changing-B (including B=0), length-aware Scan with
inactive NaNs, [gradients](../glossary.md#term-gradient), repeated [SGD](../glossary.md#term-stochastic-gradient-descent-sgd), bit-exact save/close/load and fresh SGD. It also covered
all 256 FP8 raw encodings through [transpose](../glossary.md#term-transpose) and 12 unlike-low ADD promotions. Another 80
numerical scenarios passed active [RNN](../glossary.md#term-recurrent-neural-network-rnn)/[GRU](../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../glossary.md#term-long-short-term-memory-lstm) forward and actual Compiler gradients in both
directions, plus owned runtime-length [Model](../glossary.md#term-model) save/load, for every low type.
The nine-type operation catalog completed 6,872 rows across all 115 ordinary kinds and 137
[signatures](../glossary.md#term-signature): 1,398 forward and 2,160 actual AD runs, with no preparation/execution-stage gaps.
Illegal dtype/body projections and nondifferentiable roles remain separate classifications.
Catalog execution is not an independent [numerical oracle](../glossary.md#adr-ci-and-qualification) for every row.
Three configured policies with changed [weights](../glossary.md#term-weight) confirmed F16 native AUTO/PRIMITIVES/GRAPH and
BF16 native AUTO/PRIMITIVES; BF16 GRAPH kept ordinary execution for that shape. Both FP8 types
stayed ordinary CPU on M3 under all policies. Local [descriptor](../glossary.md#term-descriptor) diagnostics returned status 3 for
FP8 and F64. The final repository `test build` gate and [Javadoc](../glossary.md#term-javadoc) for CPU, Metal, Model,
Compiler, [Engine](../glossary.md#term-engine), [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context), Training and [Checkpoint](../glossary.md#term-checkpoint) passed with the configured native libraries.
This is source qualification, not rebuilt-[release](../glossary.md#term-release) acceptance. No native FP8 execution,
unbounded geometry or performance qualification is claimed.

## Owned source and [persistence](../glossary.md#term-persistence) boundaries

[MLP](../glossary.md#term-multilayer-perceptron-mlp), default persistable CNN1d/2d/3d and [runtime](../glossary.md#term-runtime)-length [tanh](../glossary.md#term-tanh) [RNN](../glossary.md#term-recurrent-neural-network-rnn) support compute, one configured
[training session](../glossary.md#training-session--trainingsession), repeated fresh [batches](../glossary.md#term-batch-machine-learning-and-hyperparameters), save, close, load and newly configured training in
their qualified [CPU](../glossary.md#term-cpu) domains. Earlier [Metal](../glossary.md#term-metal) [F32](../glossary.md#floating-point-precision-and-rounding) MLP/CNN1d/2d/3d and runtime-length general-[Scan](../glossary.md#term-cumulative-scan)
RNN passed actual-[GPU](../glossary.md#term-gpu) [parameter](../glossary.md#term-parameter) updates/save/load/fresh training without CPU fallback. Current
F32/[BF16](../glossary.md#floating-point-precision-and-rounding)/[F16](../glossary.md#floating-point-precision-and-rounding) owned MLP, [Conv3d](../glossary.md#conv3d--ncdhw)/overlapping [Pool3d](../glossary.md#pool3d--ncdhw-pooling) and explicit GeneralScan RNN evidence is below.

[Tensor](../glossary.md#tensor)-level numerical support is broader than cold acquisition/[serialization](../glossary.md#term-serialization). Within the
existing recipe domain, [cold reconstruction](../glossary.md#term-cold-reconstruction) accepts [Model](../glossary.md#term-model)-legal per-role floating types and casts.
Any source containing [FP8](../glossary.md#term-fp8) selects [graph](../glossary.md#graph) domain v7, with [dtype](../glossary.md#data-type--datatype) wire IDs 8 ([E4M3](../glossary.md#term-e4m3)) and 9 ([E5M2](../glossary.md#term-e5m2)).
Old dtype IDs 1–7 and non-FP8 smallest-sufficient v1–v6 bytes remain unchanged: Pool3d selects
v6 when FP8 is absent. [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) FP8 persistence uses conditional model.v2/SYNMOD02,
definition SYNDEF02 and whole-model.v2 formats; non-FP8 values retain old compatible formats.
Strict [descriptor](../glossary.md#term-descriptor) [rebind](../glossary.md#capture-graph-ports-and-reconstruction) remains. Arbitrary [PURE](../glossary.md#term-pure) Scan execution does not automatically widen
cold reconstruction or add wire opcodes. Fixed recipes retain their distinct working/[publication](../glossary.md#term-publication) boundaries.
No [optimizer](../glossary.md#term-optimizer)/history/compiled-plan continuation is promised.

## [Metal](../glossary.md#term-metal) production [target](../glossary.md#term-target)

Metal is required and is **not experimental**. Its independent target includes qualified static
[F32](../glossary.md#floating-point-precision-and-rounding)/[BF16](../glossary.md#floating-point-precision-and-rounding)/[F16](../glossary.md#floating-point-precision-and-rounding) ordinary numerical families, [Conv3d](../glossary.md#conv3d--ncdhw)/overlapping [Pool3d](../glossary.md#pool3d--ncdhw-pooling) training, attention, native
GeneralScan and fixed [RNN](../glossary.md#term-recurrent-neural-network-rnn)/[GRU](../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../glossary.md#term-long-short-term-memory-lstm) selected first-order [derivatives](../glossary.md#term-derivative). Exact/raw [F64](../glossary.md#floating-point-precision-and-rounding)/integral rows
remain separate: stored F64 movement is not F64 numerical arithmetic. General F64 neural
arithmetic, unsupported mixed-low domains and unqualified higher-order recurrent derivatives
remain exclusions, not [CPU](../glossary.md#term-cpu) fallback paths. The [Metal guide](metal-backend.md) and
[owning contract](../architecture/contracts/backend-execution.md#metal-backend) define exact limits.

The prior [operation](../glossary.md#operation) catalog covered 5,018 rows, all 115 ordinary kinds and all 137 legal [signature](../glossary.md#term-signature)
fixtures: 744 forward and 494 actual selected first-order [AD](../glossary.md#autograd) runs, with zero execution-stage gaps.
Its 924 selection rejections are historical baseline classifications, not current cutover results;
[Model](../glossary.md#term-model)-illegal projections and nondifferentiable/deferred roles were counted separately. In
particular, its seventeen-input CONCAT/STACK exclusion is obsolete. Current composition uses
dynamic operand references and one disjoint direct-copy child per operand through the same
ordinary/[Scan](../glossary.md#term-cumulative-scan) [owner](../glossary.md#term-owner), not N-wide [argument](../glossary.md#term-argument) [binding](../glossary.md#term-binding) or an intermediate pairwise [tensor](../glossary.md#tensor) chain.
The common [rank](../glossary.md#term-rank)-sixteen and checked geometry/span/[dispatch](../glossary.md#term-dispatch)/workspace bounds remain. Fresh
complete per-operation catalogs contain 5,018 rows per [backend](../glossary.md#backend). Metal completed 908 forward and
1,156 selected first-order AD runs without an execution-stage failure. Each F32/BF16/F16 projection
has the same 187 forward and 358 AD cases, without a Model-valid selection gap; the 374 Metal
selection rejections are confined to the intentionally unsupported numerical F64 domain.

Independent Metal-only numerical proof passed 93 F32 forward/actual all-role derivative cases:
81 ordinary cases, eight Pool3d Scan cases and four signed-update/overlapping-[axis](../glossary.md#term-axis)-fold,
[scalar](../glossary.md#term-scalar)-reciprocal and repeated-STACK accumulation edges. Offset-[view](../glossary.md#term-view-meanings) [binary](../glossary.md#term-binary-descriptor-and-metadata) arithmetic uses
the existing physical-[stride](../glossary.md#term-stride) [kernels](../glossary.md#kernel); packed [canonical](../glossary.md#term-canonical-form) operands retain [MPSGraph](../glossary.md#term-mpsgraph).
Another 136 exact-domain cases passed, including all 49 casts, four floating WHERE cases,
stored F64 decisions and integral modular/promotion paths. Numerical correctness is not inferred
from catalog [hashes](../glossary.md#term-hash). Public [runtime](../glossary.md#term-runtime)-length RNN separately passed all-[parameter](../glossary.md#term-parameter) [finite
differences](../glossary.md#arithmetic-rearrangement-and-finite-difference-checks), repeated [SGD](../glossary.md#term-stochastic-gradient-descent-sgd) with changing lengths, all-inactive rows, invalid-length nonmutation/
retry, save/close/load raw-exact state and fresh configured SGD on a Metal-only [Engine](../glossary.md#term-engine).

Current genuine M3 Max component evidence exercises the original F32/BF16/F16 carriers through
shared typed storage/[POD](../glossary.md#term-pod) roles, [FLOAT32](../glossary.md#term-float32) [working arithmetic](../glossary.md#accumulator-type) and per-output [publication](../glossary.md#term-publication). The
separate low-[precision](../glossary.md#term-precision) library is retired in favor of common owners and
`synaptik_numeric_codec.h`; there is no FLOAT32 tensor proxy, CPU fallback, [FP64](../glossary.md#term-fp64)
arithmetic/emulation. This Metal receipt does not qualify the independent CPU low-parity cutover.

- CONCAT/STACK passed 17/33/257 inputs in all three types, including original-carrier views and
  last-operand-only AD. Zero-output CONCAT/STACK empty cotangents passed public GPU
  run/materialize. Signed raw SLICE passed all seven carriers; INT32/INT64 cumulative operations
  use exact modular arithmetic.<span data-glossary-navigation="true"> ([run glossary](../glossary.md#term-run); [INT32 glossary](../glossary.md#data-type--datatype); [INT64 glossary](../glossary.md#data-type--datatype); [modular arithmetic glossary](../glossary.md#term-modular-arithmetic))</span>
- Fixed RNN/GRU/LSTM first AD passed 36 public numerical cases (three types, three cells, both
  directions and both bias choices), covering every floating input/state/weight/bias role,
  lengths 0/1/3 and inactive NaNs against independent FLOAT32 finite differences and typed scalar
  adjoints. The fixed-cell builder retains its own working/publication rounding schedule.<span data-glossary-navigation="true"> ([weight glossary](../glossary.md#term-weight); [rounding glossary](../glossary.md#term-rounding))</span>
- Owned MLP and Conv3d with overlapping average/max Pool3d passed nine cases across the three
  types: independent forward/MSE/input-and-parameter AD/SGD, save/close/load/rebind and fresh
  training. This is bounded numerical journey evidence, not universal shape/layout proof.<span data-glossary-navigation="true"> ([MSE glossary](../glossary.md#mean-squared-error--mse); [rebind glossary](../glossary.md#capture-graph-ports-and-reconstruction); [shape glossary](../glossary.md#shape); [layout glossary](../glossary.md#layout))</span>
- Explicit NN GeneralScan passed twelve cases across the three types, both directions and bias
  choices, lengths `[4,2,0]`/`[1,3,4]`, inactive NaNs, independent ordinary RnnCell unroll and
  parameter AD, save/load and fresh SGD. It uses the ordinary-cell typed path, not a fixed-cell
  compatibility fallback; the two paths' rounding semantics remain distinct.
- Native public GeneralScan with 33 distinct operands passed 36 three-family concat/stack,
  direction and activity configurations. Only the last invariant requires gradients; complete
  emissions/final carries/cotangents match independent exact-integer oracles, including inactive NaNs.<span data-glossary-navigation="true"> ([invariant glossary](../glossary.md#term-invariant))</span>
- Normalization passed 36 full-forward cases and 114 explicit-seed AD routes across the three
  families against independent double formulas and finite differences, including all public
  BatchNorm training siblings and correction-one running variance.<span data-glossary-navigation="true"> ([Normalization glossary](../glossary.md#term-normalization); [seed glossary](../glossary.md#term-seed); [variance glossary](../glossary.md#term-variance))</span>
- Dropout passed 18 dtype/mode/probability configurations against independent published-mask
  forward/VJP equations, exact dropped zeros with NaN payloads, replay, state advancement/counter
  wrap and fresh-input/seed reuse. Evaluation remains a Model identity, not a native random draw.<span data-glossary-navigation="true"> ([Dropout glossary](../glossary.md#dropout--inverted-dropout); [dtype glossary](../glossary.md#data-type--datatype); [mask glossary](../glossary.md#term-boolean-mask); [VJP glossary](../glossary.md#term-vector-jacobian-product-vjp); [NaN glossary](../glossary.md#term-nan); [Evaluation glossary](../glossary.md#term-evaluation))</span>

[General Scan](../glossary.md#general-differentiable-scan) uses a true native time loop and once-prepared existing ordinary kernels, not
Java child invocation, time [unrolling](../glossary.md#term-unrolling) or a CPU evaluator. Controlled operands and original
canonical output roles retain structural inactivity and [guard](../glossary.md#term-guard)-before-write semantics.
GeneralScan, selected reverse bodies, source-only [gradients](../glossary.md#term-gradient) and empty storage remain
independently checked; they do not confer arbitrary symbolic/empty or higher-order support.

The previously integrated Metal identity39 source passed full F32 [MLP](../glossary.md#term-multilayer-perceptron-mlp)/[CNN](../glossary.md#term-convolutional-neural-network-cnn) scalar-[oracle](../glossary.md#term-test-oracle) compute,
all-parameter SGD/save/close/load/new training on the actual [GPU](../glossary.md#term-gpu) without CPU fallback.
All fifteen original-typed F32/BF16/F16 five-form forward/both-generated-gradient [graphs](../glossary.md#graph) now pass unchanged,
including [transpose](../glossary.md#term-transpose)-of-dense materializing RESHAPE. Public raw reshape additionally passed all
seven carriers. [Metal0081](../planning/backends/metal/tasks/0081-materializing-reshape-composition.md#result)
retains exact source, package, [device](../glossary.md#term-device)/skip and before-failure evidence; this is not arbitrary backward closure.
Final independent source/evidence/documentation review accepted this declared Metal [matrix](../glossary.md#term-matrix-meanings);
Metal0080/0081 are Complete. This is component acceptance, not a complete [release](../glossary.md#term-release).

The integrated [Metal0082 host-copy correction](../planning/backends/metal/tasks/0082-host-copy-autorelease-boundaries.md#result)
also passes a rebuilt source-matched native install, full GPU/public integration and unchanged
MLP/CNN training/[persistence](../glossary.md#term-persistence) and all-seven-[carrier](../glossary.md#term-storage-carrier) raw movement. Raw and mixed public before/after
probes isolate and repair Foundation [host](../glossary.md#term-host)-copy retention: the named IOAccelerator category stays
flat across repeated windows, including 3,288 retained closed handles. Full-process footprint,
which includes [warmup](../glossary.md#term-warmup) and retained Java [metadata](../glossary.md#term-metadata), is not claimed [constant](../glossary.md#term-constant). This [finite](../glossary.md#term-finite) observation
is not universal allocation proof. Final independent source/evidence/docs review accepted Metal0082
with zero P1/P2 findings; the component is Complete.

The earlier Conv3d extension passed grouped/[bias](../glossary.md#term-bias)/depthwise all-role finite differences and real
[NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) training/persistence in a Metal-only Engine containing no CPU backend. Current
F32/BF16/F16 FOLD_AXIS/FOLD2D/FOLD3D overlap addition is output-owned and deterministic, with
FLOAT32 working arithmetic and per-output typed publication, no atomics, scratch, host
fallback/repair or software [FLOAT64](../glossary.md#term-float64). FLOAT64/integral folds retain qualified non-overlap limits.
Signed [SLICE](../glossary.md#slice) materializes canonical storage for all seven carriers; it does not authorize
negative physical strides. These shared owners close the qualified selected movement adjoints.
The current coordinated [identity](../glossary.md#term-identity-meanings) is 43, program schema22/`SM22`/header124, native ABI7/thirteen
exports, operation wires1..116 and attribute wires0..42. Shared native source has 23 authenticated
components and 188,490 [UTF-8](../glossary.md#term-utf-8) bytes, [SHA-256](../glossary.md#term-sha-256)
`9d0d08724f5deac665bb4e09c08af3b0fad08efff050dad2d8ae265f39a030d4`.
The genuine M3 Max package builds with `-Wall -Wextra -Werror`, signs and packages; this is
component evidence, not a new [sealed](../glossary.md#term-sealed) release or installation acceptance.

The earlier component source was built, ad-hoc signed, ZIP-extracted outside checkout and
reverified before its GPU gate. The supported reproducible local source-build installation uses
the existing [native build/package/install commands](../../modules/backend/metal/macos-arm64/README.md#build-prerequisites-and-local-package)
and an explicit absolute MetalBackendConfiguration path. This is not a trusted public prebuilt
[distribution](../glossary.md#term-distribution): no [signing identity](../glossary.md#identity-meanings-and-algebraic-identities) or hosted GPU runner is established. Official prebuilt signing/
publication remains distinct from a caller's verified local source build; no verifier bypass or relabeling.

## Evidence and candidate [qualification](../glossary.md#term-qualification)

Accepted bounded [RNN](../glossary.md#term-recurrent-neural-network-rnn) and wire records: [Compiler0018](../planning/modules/compiler/tasks/0018-static-length-scan-cold-reconstruction.md#result),
[Checkpoint0014](../planning/extensions/checkpoint/tasks/0014-length-aware-general-scan-v5.md#result),
[NN0033](../planning/extensions/nn/tasks/0033-trainable-runtime-length-rnn-workflow.md#result).
[CPU](../glossary.md#term-cpu) [MLP](../glossary.md#term-multilayer-perceptron-mlp)/[CNN](../glossary.md#term-convolutional-neural-network-cnn) numerical [logs](../glossary.md#commit-release-distribution-log-and-seed-meanings) and the 16 [convolution](../glossary.md#term-channels-first-convolution-layers)/24 [pooling](../glossary.md#term-pooling) actual-backward cases are retained
in the coordinator's source-[snapshot](../glossary.md#term-snapshot) evidence, not inferred from [operation](../glossary.md#operation) counts.
Another 54 ordinary and 30 indexing/ordering public cases passed independent [scalar](../glossary.md#term-scalar) forward and
actual all-role [Compiler](../glossary.md#term-compiler) [gradients](../glossary.md#term-gradient) in both CPU types. These are [finite](../glossary.md#term-finite) [canonical](../glossary.md#term-canonical-form) geometries,
not automatic special-value/[layout](../glossary.md#layout)/strategy or all-[PURE](../glossary.md#term-pure) [Scan](../glossary.md#term-cumulative-scan) proof.

The original general-Scan baseline rejected all 24 attempted [F32](../glossary.md#floating-point-precision-and-rounding)/[F64](../glossary.md#floating-point-precision-and-rounding) RELU/[SOFTMAX](../glossary.md#softmax--log-softmax)/LayerNorm
combinations across both modes/traversals. Those before-failures remain retained. The unchanged
program now passes, together with ordinary84, composition268 and named144 numerical cases,
structural/materialized boundaries, typed raw/T0 sources and genuine legacy recipe replay.
The unfiltered affected twelve-project recovery passed 4,942 tests, 64 skips, zero failures/errors
and all twelve Javadocs. Independent final semantic/physical [ownership](../glossary.md#term-ownership-meanings) reviews approved with
zero findings. This is component acceptance; it does not imply universal layout or performance.

[Engine0022](../planning/modules/engine/tasks/0022-result-publication-and-closed-payload-cleanup.md#result)
repairs the reproduced closed-[publication](../glossary.md#term-publication) [heap](../glossary.md#term-heap)-[payload](../glossary.md#artifact-envelope-and-atomic-publication) retention and advanced publication [rollback](../glossary.md#term-rollback)
gap. Independent approval, affected gates and public both-type heap/[lifetime](../glossary.md#term-lifetime)/precedence/[race](../glossary.md#term-race) proof
passed. The heap diagnostic is not native-memory plateau evidence or a whole-[release](../glossary.md#term-release) qualification.

The integrated [Engine0023 correction](../planning/modules/engine/tasks/0023-reentrant-close-and-lease-admission.md#result)
passes the unchanged real-[GPU](../glossary.md#term-gpu) self-close and actual bounded-heap [token](../glossary.md#term-token)-[OOM](../glossary.md#term-out-of-memory-oom) consumers plus affected
[module](../glossary.md#term-module)/public-integration gates and final independent source/evidence/docs review. Engine0023 is
Complete for that correction. No native OOM was induced; Metal0082 supplies the separate finite,
category-specific native-memory evidence above, not a universal allocation guarantee.

Engine0024's standard eighteen-product [distribution](../glossary.md#term-distribution) mechanism is Complete after clean extracted
source builds and external [Maven](../glossary.md#term-maven)/[Gradle](../glossary.md#term-gradle) consumers. A fresh Maven project declaring only `nn`
also resolves its sixteen-[JAR](../glossary.md#term-jar) closure and passes the original full both-type MLP/CNN/RNN
numerical, [optimizer](../glossary.md#term-optimizer) and save/load/new-train programs without the [incubator module](../glossary.md#term-incubator-module). The documented
CPU [owner](../glossary.md#term-owner)-close and fluent examples [run](../glossary.md#term-run) against those installed [artifacts](../glossary.md#term-artifact). Standard CPU startup
still performs its documented optional [OpenBLAS](../glossary.md#term-openblas) probe; grant [FFM](../glossary.md#term-ffm) access under the
[JVM prerequisites](../user-guide/running-models.md#jvm-prerequisites)<span data-glossary-navigation="true"> ([JVM glossary](../glossary.md#term-java-virtual-machine-jvm))</span>.
This qualifies the installation mechanism, not the final all-PURE release candidate.

Prior GeneralScan composition/typed activity and CPU/[Metal](../glossary.md#term-metal) neural replay have accepted component
evidence. The current typed cutover has passed complete operation catalogs and the root gate;
versioned source/artifact freeze and clean installed [API](../glossary.md#term-application-programming-interface-api) replay are separate acceptance receipts.
Short composition timings remain report-only observations with [warmup](../glossary.md#term-warmup)/[host](../glossary.md#term-host)/storage limits,
not a speed or zero-allocation promise.
The owner chose private/internal distribution: no public licence or [registry](../glossary.md#term-registry) publication is
selected. Cephes/Gradle notices remain packaged; they do not grant public rights to the project's
own code. Final candidate acceptance is recorded separately against its actual frozen artifacts.

## [oneDNN](../glossary.md#term-onednn) and exact [singleton](../glossary.md#term-singleton) root [qualification](../glossary.md#term-qualification)

The integrated macOS [arm64](../glossary.md#native-packaging-and-platform-vocabulary)/M3 Max source passed unfiltered `test build --continue` plus [CPU](../glossary.md#term-cpu)
and [Metal](../glossary.md#term-metal) Javadocs on [JDK](../glossary.md#term-java-development-kit-jdk) 26.0.1. The final command reported 159 actionable tasks: one executed
and 158 up-to-date. The complete retained inventory contains 793 [JUnit](../glossary.md#term-junit) suite reports:
5,920 tests, 64 skips, zero failures and zero errors. This accepts retained successful reports;
it does not claim every test was freshly executed in the last command.

Ten real oneDNN native-contract cases and twelve public [Engine](../glossary.md#term-engine) cases passed, including
same-address current [weights](../glossary.md#term-weight), dense/transposed/broadcast and guarded-gap [layouts](../glossary.md#layout), exact point
[identities](../glossary.md#term-identity-meanings), separate epilogues and concurrent [run](../glossary.md#term-run) isolation. Public PRIMITIVES/[GRAPH](../glossary.md#graph)/AUTO
[MLP](../glossary.md#term-multilayer-perceptron-mlp)/[CNN](../glossary.md#term-convolutional-neural-network-cnn) forward/[AD](../glossary.md#autograd)/[SGD](../glossary.md#term-stochastic-gradient-descent-sgd)/save/load/fresh-training consumers passed independent numerical oracles
with actual native execution records. The [CPU guide](cpu-backend.md#evidence-and-report-only-benchmarking)
records exact [target](../glossary.md#term-target) [identity](../glossary.md#term-identity-meanings), numerical exclusions and report-only complete-plan timings.

The rebuilt, locally signed and verified Metal package passed actual [F32](../glossary.md#floating-point-precision-and-rounding)/[BF16](../glossary.md#floating-point-precision-and-rounding)/[F16](../glossary.md#floating-point-precision-and-rounding) singleton
[publication](../glossary.md#term-publication), ordinary/[vector](../glossary.md#term-vector)/batched/transposed and controlled [Scan](../glossary.md#term-cumulative-scan) numerical checks. Its
library [SHA-256](../glossary.md#term-sha-256) is `f9f18b2378403b65629de1900b21235b3a44e4e06b709bac81263e9904bf6334`;
semantic identity is 43, with ABI7/schema22 unchanged.

### Linux [x86-64](../glossary.md#term-x86-64) [qualification](../glossary.md#term-qualification)

The independent [Actions run](https://github.com/pho001/Synaptik/actions/runs/41663584505)<span data-glossary-navigation="true"> ([run glossary](../glossary.md#term-run))</span>
at `5b6a10af` authenticated and built upstream [oneDNN](../glossary.md#term-onednn) 3.14.0 SEQ+[Graph](../glossary.md#graph), then passed the complete
[CPU](../glossary.md#term-cpu) suite and public [Engine](../glossary.md#term-engine) oneDNN qualification: 247 suites, 1,613 tests, 51 optional skips,
zero failures and zero errors. The ten native-contract and twelve public oneDNN cases all
executed without skips. The report-only whole-plan baseline also passed its independent
forward/[AD](../glossary.md#autograd) checks; exact [target](../glossary.md#term-target), library [identity](../glossary.md#term-identity-meanings) and timings are in the
[CPU guide](cpu-backend.md#evidence-and-report-only-benchmarking).

The passing full-suite target is EPYC 7763/AVX2. Earlier EPYC 9V45/AVX512 evidence proves the
22 oneDNN cases, not a passing complete CPU gate. Original failed runs and their recovery remain
in the qualification receipt; arithmetic [NaN](../glossary.md#term-nan) payloads, generated-code [hashes](../glossary.md#term-hash), default [vector](../glossary.md#term-vector)
widths and [host](../glossary.md#term-host)-specific output paths are not numerical contracts.

### [oneDNN](../glossary.md#term-onednn) installed-source and [artifact](../glossary.md#term-artifact) replay

The eighteen-product 0.1.0 [distribution](../glossary.md#term-distribution) was [checksum](../glossary.md#term-checksum)-verified and extracted outside any checkout.
A fresh [Maven](../glossary.md#term-maven) consumer declaring only `nn:0.1.0` resolved sixteen [runtime](../glossary.md#term-runtime) JARs in an isolated
[cache](../glossary.md#term-cache); every [JAR](../glossary.md#term-jar) matched its installed repository bytes. A separate standard [Gradle](../glossary.md#term-gradle) consumer
compiled and ran the documented oneDNN example, producing `64.0` without checkout classes.

The extracted NoGit source built all eighteen products with the build cache disabled and all
eighteen classes tasks executed. The final exact-path inventory selects 2,603 source files after
the portability cleanup; `source-provenance.json` binds their individual [hashes](../glossary.md#term-hash).
Its independently rebuilt, locally signed and verified [Metal](../glossary.md#term-metal) image is byte-identical to the
installed [binary](../glossary.md#term-binary-descriptor-and-metadata) image with the current `f9f18b...` [SHA-256](../glossary.md#term-sha-256) recorded above.

Both installations passed the same twelve public programs: all three oneDNN [API](../glossary.md#term-application-programming-interface-api)-policy
[MLP](../glossary.md#term-multilayer-perceptron-mlp)/[CNN](../glossary.md#term-convolutional-neural-network-cnn) forward/[AD](../glossary.md#autograd)/[SGD](../glossary.md#term-stochastic-gradient-descent-sgd)/save/load/fresh-training workflows, both primitive/[Graph](../glossary.md#graph) special-value
workflows, the [canonical](../glossary.md#term-canonical-form) example, exact three-[carrier](../glossary.md#term-storage-carrier) Metal point products, fixed recurrent
all-role AD, original-carrier owned models, length-aware GeneralScan, [normalization](../glossary.md#term-normalization) and [dropout](../glossary.md#dropout--inverted-dropout).
Their native execution records and independent numerical checks are retained under `qualification/`.
Neither [classpath](../glossary.md#term-classpath) contains checkout classes, previous product JARs or test fixtures.
This installed replay is bounded local technical acceptance, separate from the Linux gate above:
not a universal geometry claim, performance SLA, [CPU](../glossary.md#term-cpu) low-arithmetic unlock or [registry](../glossary.md#term-registry)/signing authorization.


## Prior shared typed [Metal](../glossary.md#term-metal) root [qualification](../glossary.md#term-qualification)

This prior receipt and its source/[artifacts](../glossary.md#term-artifact) are preserved unchanged under
`build/release/0.1.0-before-onednn-20261103T143942Z/`. Its counts and native [hash](../glossary.md#term-hash) below do not
describe the later [oneDNN](../glossary.md#term-onednn)/[singleton](../glossary.md#term-singleton) cutover.

The source [checkpoint](../glossary.md#term-checkpoint) through `583b6495` passed unfiltered `test build --continue` on the
actual M3 Max with that checkpoint's authenticated native package. The final gate reported 159
actionable tasks: two executed and 157 up-to-date; the preceding repaired full gate also passed
with nine executed and 150 up-to-date. This accepts retained successful reports, not a claim
that every test was freshly executed in the final command. Earlier failed commands remain
failed evidence.

The current numerical, [operation](../glossary.md#operation)-catalog, wide [Scan](../glossary.md#term-cumulative-scan), owned training and cold-replay evidence
above is distinct from the ten optional native [dispatch](../glossary.md#term-dispatch)-[observer](../glossary.md#term-observer) skips. [CPU](../glossary.md#term-cpu) low arithmetic,
general Metal [F64](../glossary.md#floating-point-precision-and-rounding) arithmetic, arbitrary geometry/higher-order closure and trusted public
prebuilt [publication](../glossary.md#term-publication) remain outside this acceptance.

The complete retained inventory contains 796 [JUnit](../glossary.md#term-junit) suite reports: 5,916 tests, 65 skips,
zero failures and zero errors. Metal contributes 580 tests with ten optional observer skips.

### Isolated versioned [distribution](../glossary.md#term-distribution) acceptance

The eighteen-product 0.1.0 [Maven](../glossary.md#term-maven)/[binary](../glossary.md#term-binary-descriptor-and-metadata)/exact-source distribution was rebuilt from the qualified
source, [checksum](../glossary.md#term-checksum)-verified and extracted outside the checkout. A new Maven consumer declaring
only `nn:0.1.0` resolved sixteen [runtime](../glossary.md#term-runtime) JARs in an isolated [cache](../glossary.md#term-cache); every [JAR](../glossary.md#term-jar) matched its published
bytes. A separate extracted NoGit source tree built all eighteen products with build cache
disabled and all eighteen classes tasks executed. Neither runtime [classpath](../glossary.md#term-classpath) contains checkout
classes or previous product JARs.

Both installations passed the same original-[carrier](../glossary.md#term-storage-carrier) [F32](../glossary.md#floating-point-precision-and-rounding)/[BF16](../glossary.md#floating-point-precision-and-rounding)/[F16](../glossary.md#floating-point-precision-and-rounding) public qualifications:
fixed [RNN](../glossary.md#term-recurrent-neural-network-rnn)/[GRU](../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../glossary.md#term-long-short-term-memory-lstm) all-role [AD](../glossary.md#autograd) (36 configurations), owned [MLP](../glossary.md#term-multilayer-perceptron-mlp)/[Conv3d](../glossary.md#conv3d--ncdhw)-overlapping-[Pool3d](../glossary.md#pool3d--ncdhw-pooling)
forward/AD/[SGD](../glossary.md#term-stochastic-gradient-descent-sgd)/cold reload (nine models), explicit [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) GeneralScan lengths/training/cold replay
(twelve configurations), [normalization](../glossary.md#term-normalization) (36 forward/114 selected-AD routes) and [Dropout](../glossary.md#dropout--inverted-dropout)
[mask](../glossary.md#term-boolean-mask)/state/reuse/[VJP](../glossary.md#term-vector-jacobian-product-vjp) (eighteen configurations). Independent numerical oracles are described above.

The extracted source tree independently rebuilt, locally signed and verified its native package
before actual [GPU](../glossary.md#term-gpu) replay. Its library is byte-identical to the installed binary library:
[SHA-256](../glossary.md#term-sha-256) `03df3cc3196295a22329c2bfb48780818641a6c118f7db30d4f4e8eb3b3c6ed9`.
That preserved package uses semantic identity42 and source SHA-256
`f1f2429fdbb7a87c3bacd5358564f4abffca3aac2fc4d3e1e86429c5ca6dff35`, not the later
[singleton](../glossary.md#term-singleton)-correction corpus; its ABI7/schema22 are unchanged.

The final documentation-only freeze retains the qualified non-documentation and binary bytes.
`source-provenance.json`, `SHA256SUMS`, `verification.json` and `qualification/` bind the final
archive source, [artifact](../glossary.md#term-artifact) [identities](../glossary.md#term-identity-meanings), commands, retained reports, failure/recovery history and
installed replay. This is local technical acceptance, not a performance SLA, [CPU](../glossary.md#term-cpu) low unlock,
general [Metal](../glossary.md#term-metal) [F64](../glossary.md#floating-point-precision-and-rounding) arithmetic, public [registry](../glossary.md#term-registry) upload or official native signing claim.

## Accepted local candidate: [Pool3d](../glossary.md#pool3d--ncdhw-pooling) and [operation](../glossary.md#operation)-audit closure

This earlier candidate and its original receipts were preserved unchanged under
`build/release/0.1.0-before-metal-parity-20261103T112058Z/`. The paths and counts below describe
that archived [checkpoint](../glossary.md#term-checkpoint), not the current typed candidate.

The private/internal 0.1.0 candidate passed the joined `test build releaseDistribution` gate
on 2026-11-03. The gate reported 220 actionable tasks: 106 executed, five from [cache](../glossary.md#term-cache) and 109
up-to-date; it does not claim every test was freshly rerun in that command. The preceding
unfiltered joined `test build` also passed. Earlier failed commands remain failed evidence.

Fresh catalog replay against the qualified code reproduced the [CPU](../glossary.md#term-cpu) 804 forward/736 actual-[AD](../glossary.md#autograd)
and [Metal](../glossary.md#term-metal) 744 forward/494 actual-AD totals above, with zero execution-stage gaps. Separate
independent numerical replay passed CPU 178 and Metal 93 all-role cases plus 136 exact-domain
cases in each of two isolated installations: a [Maven](../glossary.md#term-maven) consumer declaring only `nn:0.1.0`, and
a clean no-Git source build with build cache disabled and all sixteen classes tasks rerun.
Every one of the sixteen resolved [runtime](../glossary.md#term-runtime) JARs matched the published bytes. Neither runtime
[classpath](../glossary.md#term-classpath) used checkout classes or old product JARs.

Both installations passed CPU [F32](../glossary.md#floating-point-precision-and-rounding)/[F64](../glossary.md#floating-point-precision-and-rounding) and Metal-only F32 [MLP](../glossary.md#term-multilayer-perceptron-mlp)/[CNN](../glossary.md#term-convolutional-neural-network-cnn)/runtime-length [RNN](../glossary.md#term-recurrent-neural-network-rnn) workflows,
[Conv3d](../glossary.md#conv3d--ncdhw) → max/average Pool3d → dense all-[parameter](../glossary.md#term-parameter) training/save/close/load/fresh training, and
24 Pool3d [Scan](../glossary.md#term-cumulative-scan) [finite-difference](../glossary.md#term-finite-difference-gradient-check)/cold replay cases across the qualified types, both activity
modes and both traversal directions. The documented FluentModelExample produced `6`, `25`,
`-2` from installed [artifacts](../glossary.md#term-artifact) and independently compiled extracted classes.

The native package was rebuilt from the qualified source, locally signed, extracted and
verified outside the checkout, then exercised on Apple M3 Max with [JDK](../glossary.md#term-java-development-kit-jdk) 26 and explicit native
access, without preview or the incubator [Vector](../glossary.md#term-vector) [module](../glossary.md#term-module). Short promoted-ADD and offset-SELECT/ADD
timings are retained as report-only public-execution observations; they include [ownership](../glossary.md#term-ownership-meanings)/
scheduling and [GPU](../glossary.md#term-gpu)-transfer costs, unlike the clean-Java [kernel](../glossary.md#kernel)-only reference. No global
speed parity, allocation plateau or [throughput](../glossary.md#term-throughput) SLA is inferred.

The versioned `build/release/0.1.0/verification.json`, `SHA256SUMS` and `qualification/` records
bind [source provenance](../glossary.md#term-source-provenance), artifact/native [hashes](../glossary.md#term-hash), commands, failure/recovery history, exact
catalog rows, installation [logs](../glossary.md#commit-release-distribution-log-and-seed-meanings) and limitations. The final documentation-only artifact freeze
must retain the qualified non-documentation bytes. Acceptance is for this explicit [matrix](../glossary.md#term-matrix-meanings),
not arbitrary geometry, newly unlocked CPU low arithmetic, general Metal F64 arithmetic,
fixed-recurrent [BPTT](../glossary.md#term-backpropagation-through-time-bptt), public licensing/[registry](../glossary.md#term-registry) [publication](../glossary.md#term-publication) or a trusted public native prebuilt.

The successful root gate accepted 794 retained [JUnit](../glossary.md#term-junit) suite reports: 5,852 tests, 65 skips,
zero failures and zero errors. Metal contributes 545 tests with ten skips; actual-GPU numerical
and installed-consumer proof above is separate from those skips. Cached/up-to-date report
acceptance is not reported as a fresh execution of every test.

## Public-readiness audit: current source and installed candidate

The framework-wide [audit](../planning/public-release-audit-2026-11-04.md#result) is complete.
Its local candidate is `0.1.0-public-audit.20261104`, separate from the historical `0.1.0`
archives above. Public [publication](../glossary.md#term-publication) is still blocked by the [owner](../glossary.md#term-owner)'s retained no-licence/
no-redistribution decision. [Qualification](../glossary.md#term-qualification) is local macOS [arm64](../glossary.md#native-packaging-and-platform-vocabulary) with JDK26, actual [Metal](../glossary.md#term-metal) and the
explicit [oneDNN](../glossary.md#term-onednn) 3.14.0 [LP64](../glossary.md#term-lp64) SEQ library; no x86 or trusted-public-prebuilt claim follows.

| Evidence | Observed result and boundary |
|---|---|
| Joined root gate | `test build` and affected [Javadoc](../glossary.md#term-javadoc): BUILD SUCCESSFUL; 159 actionable tasks, 22 executed and 137 up-to-date. |
| Active-[layout](../glossary.md#layout) [JUnit](../glossary.md#term-junit) receipts | 790 suites, 6,012 tests, 67 skips, zero failures/errors. These accepted reports include up-to-date tasks; they are not a claim that every test was freshly executed. |
| Metal [module](../glossary.md#term-module) | 624 tests, ten skips, zero failures/errors. Independent actual-[GPU](../glossary.md#term-gpu) public scenarios are additional evidence, not inferred from the suite count. |
| Ordinary [CPU](../glossary.md#term-cpu) workflow | 205 exact public checks across [BF16](../glossary.md#floating-point-precision-and-rounding)/[F16](../glossary.md#floating-point-precision-and-rounding)/[E4M3](../glossary.md#term-e4m3)/[E5M2](../glossary.md#term-e5m2): arithmetic, shared [Compiler](../glossary.md#term-compiler) [AD](../glossary.md#autograd), Pool2d/3d, changing/zero [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters), length-aware [Scan](../glossary.md#term-cumulative-scan), repeated [SGD](../glossary.md#term-stochastic-gradient-descent-sgd) and save/load with fresh training. Includes configured oneDNN policies where independently eligible. |
| CPU recurrent workflow | 80 checks: active [RNN](../glossary.md#term-recurrent-neural-network-rnn)/[GRU](../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../glossary.md#term-long-short-term-memory-lstm), both directions, actual Compiler [gradients](../glossary.md#term-gradient) and length-aware [Model](../glossary.md#term-model) save/load across all four low types. |
| Metal recurrent workflow | 40 corresponding real-[device](../glossary.md#term-device) checks across BF16/F16, both directions and [persistence](../glossary.md#term-persistence). No [FP8](../glossary.md#term-fp8) Metal domain is admitted. |
| Targeted public repros and diagnostics | [Overflow](../glossary.md#term-overflow) versus [L2](../glossary.md#term-l2) point [identity](../glossary.md#term-identity-meanings), categorical zero gradients, exact [pooling](../glossary.md#term-pooling) divisor bits, [contraction](../glossary.md#contraction-dimension)-before-[bias](../glossary.md#term-bias), mixed-owner excess capacity, transactional restore, deep Scan/namespace and session-only tuning. CPU cold/fallback observers and native bounds/duplicate/[BOOL](../glossary.md#data-type--datatype) failures preserve actual context and native causes. |
| Fresh [Maven](../glossary.md#term-maven) installation | All 18 Java products published with [binary](../glossary.md#term-binary-descriptor-and-metadata)/source/Javadoc [artifacts](../glossary.md#term-artifact). A consumer declaring only `nn` resolved 16 JARs from a fresh Maven [cache](../glossary.md#term-cache) and passed the same public programs using only extracted candidate [native code](../glossary.md#native-numerical-library-and-instruction-architecture). |
| Extracted source installation | All 2,632 selected source [hashes](../glossary.md#term-hash) matched. All 16 Java [compilation](../glossary.md#compiler-engine-and-execution-stage-names) actions executed with project build caching disabled; native source rebuilt, ad-hoc signed and package-verified. The same public programs passed against those independently built Java/native outputs without checkout [runtime](../glossary.md#term-runtime) paths. |

The supported numerical, geometry, [ownership](../glossary.md#term-ownership-meanings) and persistence boundaries remain the owning
contracts and family-specific [matrix](../glossary.md#term-matrix-meanings) above. The audit repaired defects; successful [operation](../glossary.md#operation)
catalog rows still do not independently certify every legal value, [shape](../glossary.md#shape) or numerical outcome.
Native FP8 is unavailable on this Arm [target](../glossary.md#term-target), so both FP8 formats stay ordinary CPU, without
advertising widened [FLOAT32](../glossary.md#term-float32) input as native FP8. Metal general [F64](../glossary.md#floating-point-precision-and-rounding) arithmetic remains excluded.
No [optimizer](../glossary.md#term-optimizer) continuation, exhaustive allocation-failure injection or universal performance
qualification is implied.

`build/release/0.1.0-public-audit.20261104/verification.json`, `SHA256SUMS` and `qualification/`
bind exact source/artifact hashes, commands, current-layout reports, exercised source programs,
installed/source output and failed-gate recovery history. The final documentation-only refresh
must retain the 1,813 frozen non-Markdown source hashes, all 18 runtime JARs and the installed
native-library [fingerprint](../glossary.md#term-fingerprint); record the comparison without pretending to re-execute every test.

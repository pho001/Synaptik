/**
 * Supplies explicit capability, configuration, and lifecycle integration for the Metal backend.
 *
 * <p>{@link io.github.pho001.synaptik.backend.metal.MetalCapabilityProvider} reports a common exact
 * domain under both profiles: parameterless {@code NEG}, {@code ABS}, {@code FLOOR}, {@code CEIL},
 * {@code SIGN}, and {@code RELU}; {@code RESHAPE}, {@code EXPAND}, {@code PERMUTE}, {@code
 * EXPAND_DIMS}, {@code SQUEEZE}, affine positive-step {@code SELECT}/{@code SLICE}, and the
 * explicit {@code CONTIGUOUS} canonicalization barrier over all six carriers. Authenticated local
 * zero-stride broadcast views remain logically affine while every physical descriptor stays safe;
 * external zero-stride, negative-stride, overlapping, empty, unresolved, and dynamic layouts remain
 * false. All six carriers also enter exact raw movement for {@code GATHER}, {@code
 * GATHER_ELEMENTS}, {@code GATHER_ND}, {@code SCATTER_ELEMENTS}, {@code SCATTER_ND}, {@code PAD},
 * {@code SLICE_UPDATE}, {@code CONCAT}, {@code STACK}, {@code TILE}, and {@code UNFOLD_AXIS}, while
 * FLOAT64/FLOAT32/BFLOAT16 enter non-overlapping fold/window rows. Index roles accept INT32 or
 * INT64 and every replacement validates the complete index domain and destination uniqueness before
 * any mutation.
 *
 * <p>The common domain includes all 36 ordered{@code CAST} pairs with Model-exact bit conversion;
 * FLOAT64/FLOAT32/BFLOAT16 classification; scalar and right-aligned broadcast BOOL logic; and all
 * nine promoted floating {@code WHERE} signatures. The nine floating-to-floating casts preserve
 * legal gradient metadata, while float-to-integral may consume a gradient-eligible input without
 * producing a differentiable output. WHERE differentiability is the exact branch-role OR and its
 * condition is never differentiable. All selected occurrences use one deterministic custom
 * whole-program route with no fallback, retry, timing, or autotuning.
 *
 * <p> {@code ACCELERATOR} additionally admits tensor {@code ADD}, {@code SUB}, {@code MUL}, {@code
 * DIV}, {@code MIN}, and {@code MAX}; all six binary comparisons with canonical one-byte {@code
 * BOOL} output; exact FLOAT32 scalar {@code MIN}, {@code MAX}, and fused {@code CLAMP}; canonical
 * {@code FLOAT32} {@code SUM}, {@code MEAN}, {@code MIN}, {@code MAX}, binding-resolved {@code
 * SUM_TO_SHAPE}, and the Task-0069 source-owned {@code L1_NORM} slice; all four exclusive/reverse
 * modes of {@code CUM_SUM} and {@code CUM_PROD}; and every positive-static FLOAT32 {@code MATMUL}
 * vector, matrix, batched, and right-aligned broadcast geometry. The L1 slice is exactly one
 * canonical no-gradient rank-one input, ordered axes {@code [0]}, positive extent and four-byte
 * span bounded to unsigned 32 bits, and a canonical scalar or retained {@code [1]} output. Both
 * profiles admit no-gradient INT32/INT64 MATMUL pairs with INT64-dominant promotion and modular
 * result arithmetic. Accelerator additionally admits no-gradient BFLOAT16/FLOAT32 and
 * FLOAT32/BFLOAT16 operands with FLOAT32 result. Other admitted reductions support full,
 * normalized single-axis, ordered normalized multi-axis including empty, and exact keep-dimensions
 * forms; reduction inputs remain positive-rank while their target may be rank zero.
 * Canonical host ingress/materialization and bidirectional CPU/Metal transfer accept ranks
 * {@code 0..16} for all six public data types
 * with exact widths; transfer additionally accepts resolved positive-stride non-overlapping storage
 * layouts and rejects unresolved, zero-stride, negative-stride, or overlapping geometry. Logical
 * BOOL bytes are validated while layout holes remain uninterpreted. Exact BOOL operation ingress
 * and scalar or affine BOOL intermediates use canonical logical bytes; custom outputs are written
 * as exact zero or one, and owned MPSGraph ONE_HOT output is validated by its typed selector
 * contract. Accelerator binary inputs and outputs are canonical dense non-views and use exact
 * right-aligned broadcasting. Each MATMUL operand is canonical or the authenticated exact local
 * identity-prefix, last-two-axis {@code PERMUTE} of a canonical source; its output is canonical.
 * FLOAT32 output gradient metadata is the operand OR; integral and mixed-carrier rows are no-grad.
 * A partition containing any shared exact custom node or a MATMUL outside the retained direct
 * rank-two FLOAT32 MPSGraph slice selects one fixed custom whole-program resource: fixed custom
 * kernels and cold-compiled nested existing-node executables consume the stable declared value
 * table in program order behind one Java/native run call. Every logical intermediate is an assigned
 * run-owned buffer; targets remain the direct assigned buffers. Direct {@code NEG}, {@code ABS},
 * {@code FLOOR}, {@code CEIL}, {@code SIGN}, and {@code RELU} operands/outputs and their graph
 * feeds are canonical. SELECT/SLICE may consume exact supported storage-layout feeds and produce
 * materialized-layout values. Other affine inputs may be exact resolved zero-offset views produced
 * earlier in the same maximal partition; nested MPSGraph steps materialize those views into dense
 * assigned buffers before downstream custom steps index their physical representation. Their
 * published outputs retain exact Model view geometry. {@code CONTIGUOUS} produces canonical
 * geometry. Metal lowers one complete profile-homogeneous partition as a typed whole-partition
 * program during preparation.
 *
 * <p>A package-private exhaustive catalog describes all 115 schema-fifteen operation kinds as
 * MPSGraph {@code DIRECT}, {@code COMPOSED}, or {@code UNAVAILABLE} and custom-kernel {@code
 * AVAILABLE}, {@code PENDING}, or {@code UNAVAILABLE_WITH_PROOF}, with closed source reasons. It is
 * cold descriptive metadata only: capability remains authoritative and the catalog is never
 * consulted by Runtime. One package-private canonical prepared-route identity owns the stable
 * candidate wires {@code 1..3} and the MPSGraph/custom-kernel family. Every plan retains exactly
 * one such identity before finalization; candidate encoding delegates to it, and finalization,
 * tracing, cold binding, and execution never replace it. Package-private tests may force only a
 * route already approved by the freshly validated exact candidate batch. Production exposes no
 * force, selector, fallback, retry, or hot route decision.
 *
 * <p>{@link io.github.pho001.synaptik.backend.metal.MetalBackendConfiguration} names one explicit
 * native bridge. {@link io.github.pho001.synaptik.backend.metal.MetalBackendIntegration} opens and
 * owns one default-device native context and supplies availability, partition preparation, retained
 * {@link io.github.pho001.synaptik.backend.metal.MetalLocalWorkloadTuning local-workload} and
 * {@link io.github.pho001.synaptik.backend.metal.MetalCompletePlanTuning complete-plan} tuning,
 * backend-local schedule assembly, a shared physical schedule contribution, checked native
 * upload/download binders, caller-host ingress, and bounded canonical host materialization to
 * Engine's explicit builder. Local tuning admits only an exact singleton NEG having both the
 * custom-kernel and MPSGraph routes. Complete-plan tuning authenticates that local selection and
 * exposes one whole-plan candidate fixed to the selected route. Both collaborations use opaque
 * exact associations, fresh authoritative preparation, and session-only compatibility; they perform
 * no execution, measurement, cache input/output, or fallback-policy work. Exact prepared
 * SELECT/SLICE targets materialize by gathering logical elements from authenticated backend-private
 * physical storage using their exact positive strides and offset; unrelated prefix and gap bytes
 * remain intact. Other prepared affine targets retain the existing dense represented-order
 * publication path. Engine uses these contributions and typed binders for the bounded bidirectional
 * CPU/Metal mixed-owner schedule. Native internals remain package-private. There is no library
 * discovery, CPU fallback, or backend-global integration.
 *
 * <p>The optional {@link MetalBackendIntegration#open(MetalBackendConfiguration,
 * MetalTraceObserver) traced open} overload retains but never closes one caller-owned thread-safe
 * observer. The same trace producer is retained by ordinary, local-trial, complete-plan-trial, and
 * selected preparations. It reports only final preparation and native-invocation outcomes with
 * stream-local correlations; tuning eligibility validation creates no ghost trace unit. Observer
 * runtime failures disable later tracing without changing backend outcomes. An observer error
 * propagates from success reporting; during failure reporting, the backend failure remains primary
 * and receives a distinct acyclic observer error as a suppressed failure. The ordinary open
 * overload creates no trace producer or trace work.
 *
 * <p>Task 0060 adds exact profile-common no-gradient replacement SCATTER_ND and signed
 * SLICE_UPDATE, non-overlapping FLOAT64/FLOAT32/BFLOAT16 FOLD_AXIS/FOLD2D/FOLD3D, modular
 * INT32/INT64 PROD, and canonical BOOL ALL/ANY. Task 0066 additionally admits non-overlapping
 * INT32/INT64 FOLD_AXIS; BOOL FOLD_AXIS and integral or BOOL FOLD2D/FOLD3D remain false. Every
 * replacement validates all indices and destination uniqueness before writes; every admitted fold
 * has one output writer and skips conceptual padding. Empty reduction axes are point identities
 * without admitting zero-dimensional extents. LOG_SUM_EXP, VARIANCE, STANDARD_DEVIATION, and
 * L2_NORM have forceable structural recipes only and remain production-false. The exact bounded
 * L1_NORM occurrence described above instead uses the fixed source-owned custom program.
 *
 * <p>Task 0062 adds only ACCELERATOR same-type canonical positive-rank FLOAT32 MEAN_SQUARED_ERROR
 * with NONE, SUM, or MEAN reduction. Native lowering fixes the Model formula to one subtraction,
 * multiplication of that exact difference by itself, and the already qualified full SUM or MEAN
 * reduction. Output gradient metadata is the exact input logical OR; generated backward graphs
 * receive no new ownership. Strict IEEE, scalar inputs, other carriers or mixed carriers, and every
 * other normalization or loss kind remain false.
 *
 * <p>Task 0063 adds profile-common canonical dense {@code SORT}, {@code ARGSORT}, and positive-K
 * {@code TOP_K} for all six carriers, plus {@code ARG_MAX} and {@code ARG_MIN} for the five numeric
 * carriers. Ranks must be in {@code 1..16}; dimensions, element counts, strides, selected extents,
 * K, output counts, and one-dimensional grid widths must be positive and no greater than {@code
 * UINT32_MAX}; derived logical coordinates and indices must fit unsigned 32 bits. Validation and
 * checked byte/span calculation precede resource creation. One integer-only raw-word comparator
 * fixes stable order, NaNs last in both directions, negative zero below positive zero before
 * direction reversal, exact selected-bit copying, top-K selected-set/output order, and
 * NaN-preferred arg extrema with explicit first/last ties. TOP_K is one native step that
 * materializes and publishes paired values and INT64 indices. These five wires always select {@code
 * CUSTOM_PROGRAM}; there is no direct MPSGraph, host repair, timing, retry, fallback, or autotuning
 * route. Floating value outputs retain input gradient metadata. Task 0066 subsequently owns
 * generated floating SORT and TOP_K values-output backward through, respectively, one matching
 * stable ARGSORT or the retained canonical indices followed by replacement scatter; index roles
 * remain no-grad.
 *
 * <p>Task 0064 adds one fixed custom-program route for {@code CONV2D}, {@code CONV3D}, {@code
 * MAX_POOL2D}, {@code AVERAGE_POOL2D}, {@code MAX_POOL3D}, and {@code AVERAGE_POOL3D}. Convolution
 * is ACCELERATOR-only with FLOAT32 output and FLOAT32/BFLOAT16 operands, at least one FLOAT32
 * operand, no-gradient mixed operands, and exact all-FLOAT32 gradient metadata. Maximum pooling is
 * profile-common for FLOAT64, FLOAT32, and BFLOAT16 and preserves exact input/output gradient
 * metadata; average pooling is ACCELERATOR-only FLOAT32. All six rows require fully static positive
 * geometry bounded to unsigned 32 bits, exact symmetric padding and fixed layouts, and checked
 * result geometry. Pooling additionally caps the product of kernel extents at 65,536 positions so
 * padding-heavy metadata cannot create unbounded per-output work with tiny tensors. Conv1d and
 * Pool1d enter only through authenticated local singleton-height affine views. Maximum pooling
 * fixes NaN-first, positive-zero-over-negative-zero, first-logical-winner, raw-bit publication;
 * average pooling fixes a full-kernel divisor and conceptual positive-zero padding. Strict
 * convolution/average pooling, generated maximum-pool or Conv3d gradients, mixed convolution
 * gradients, overlap-accumulating generated folds, attention, and convolution transpose remain
 * fail-closed.
 *
 * <p>Task 0065 adds profile-common zero-input {@code INITIAL_STATE} and ACCELERATOR-only canonical
 * FLOAT32 {@code DROPOUT}. INITIAL_STATE publishes one raw {@code INT64[2]} {@code [key,counter]}
 * value whose words are interpreted unsigned. DROPOUT consumes that state plus a canonical
 * positive-static input and emits the scaled FLOAT32 result, canonical BOOL saved mask, and
 * advanced state through one fixed custom-program route. Its private SplitMix64 V1 identity uses
 * the exact top-53-bit threshold policy, raw positive-zero drops, FLOAT32 division then
 * multiplication for kept elements, and modulo-{@code 2^64} counter advancement by the logical
 * element count. It makes no entropy, cryptographic-quality, or cross-backend portable-stream
 * claim. Evaluation bypasses DROPOUT, generated backward reuses the saved mask without consuming
 * state, direct MPSGraph remains unavailable, and RNN, GRU, and LSTM remain fail-closed under both
 * profiles.
 *
 *<p>Task 0066 closes the selected dtype/layout/gradient occurrence gap without adding a kind,
 * wire, schema field, export, or route. All selected partitions use deterministic {@code
 * CUSTOM_PROGRAM}; native and JVM preflight independently authenticate logical versus physical
 * layouts, 36 cast semantics, promoted predicates/WHERE, INT64 index parity, exact saved-role
 * liveness, and one-writer replacement/window geometry. Dynamic, empty, negative, overlapping,
 * reduction-dependent, additive, transcendental, attention, and recurrent blockers remain
 * fail-closed.
 *
 * <p>Task 0069 Slice 1 adds one accelerator-only no-gradient FLOAT32 L1 norm occurrence at wire
 * 114. One output thread raw-ABS-transforms every rank-one contributor, initializes from ordinal
 * zero, performs exactly {@code N-1} safe binary32 additions in increasing ordinal order, and
 * stores once; {@code N=1} is a direct raw ABS with no addition. Java and native validation
 * independently enforce ordered axes {@code [0]}, canonical scalar or retained {@code [1]}
 * output, unsigned-32-bit element/span bounds, and distinct value-table buffers before execution.
 * The proof and source/compiler-site certificate are pinned under {@code evidence/0069}; no
 * SCATTER_ADD or VARIANCE production row is included.
 *
 * <p>The selected numerical profile participates in partition-plan, route, tuning, decision-codec,
 * and workload identity. Java rejects profile/schema mismatches before native entry. ABI version
 * five retains thirteen exports. Node schema version fifteen is one bounded self-describing
 * route-bearing image over stable type wires {@code 1..6}, operation wires {@code 1..115},
 * attribute wires {@code 0..41}, and complete optional storage-layout geometry. Native structural
 * execution covers exactly 101 wires and leaves 14 nonexecutable. Production capability is exactly
 * 84 operation kinds and 31 remain false. Backend-local workload, exact-policy, candidate,
 * compatibility, route-policy, and codec identities are version twenty-three; schema fourteen and
 * identity version twenty-two and earlier fail closed.
 */
package io.github.pho001.synaptik.backend.metal;

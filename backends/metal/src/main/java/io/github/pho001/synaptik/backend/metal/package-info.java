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
 * SUM_TO_SHAPE}, and the Task-0069 source-owned {@code L1_NORM} and {@code SCATTER_ADD} slices; all
 * four exclusive/reverse modes of {@code CUM_SUM} and {@code CUM_PROD}; and every positive-static
 * FLOAT32 {@code MATMUL} vector, matrix, batched, and right-aligned broadcast geometry. The L1 slice
 * is exactly one canonical no-gradient rank-one input, ordered axes {@code [0]}, positive extent
 * and four-byte span bounded to unsigned 32 bits, and a canonical scalar or retained {@code [1]}
 * output.
 * Both profiles admit no-gradient INT32/INT64 MATMUL pairs with INT64-dominant promotion and
 * modular result arithmetic. Accelerator additionally admits no-gradient BFLOAT16/FLOAT32 and
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
 * kernels and cold-compiled nested existing-node executables consume the stable compact
 * materialized-slot table in program order behind one Java/native run call. Only the authenticated
 * materialized-value set receives run-owned buffers; targets remain their direct assigned buffers.
 * Direct {@code NEG}, {@code ABS}, {@code FLOOR}, {@code CEIL}, {@code SIGN}, and {@code RELU}
 * operands/outputs and their graph feeds are canonical. Eligible maximal linear canonical
 * FLOAT32 chains of the latter four operations require positive static rank one through sixteen
 * and an unsigned-32-bit positive element count, then deterministically fuse into generated units
 * of length two through eight with one boundary load, one boundary store, and no intermediate
 * slot. Java carries only integer source byte counts; native owns generated source and every
 * source hash. Native authenticates all frozen fixed-source components and the assembled source
 * before compile, and requires exact strongest available binding/type reflection for all three
 * buffers and every {@code PointMeta} member.
 * <p>Anchor-epilogue fusion is ACCELERATOR-only and preserves exact source order. MATMUL accepts
 * optional literal {@code SCALAR_MUL}, at most one ordinary right-aligned tensor {@code ADD}, and
 * optional terminal {@code RELU} or no-gradient {@code CLAMP}. Conv2d accepts at most one external
 * ADD followed by the same terminal pair and never scalar multiplication. Its intrinsic rank-one
 * {@code [C]} bias remains only the third Conv2d input; an external rank-one addend is {@code [W]},
 * while {@code [1,C,1,1]} is the ordinary channel-broadcast form. Every absorbed intermediate is
 * private, single-consumer, non-target, canonical, and absent from the materialized set. An
 * admitted anchor uses one safe-math dispatch and one final store with no intermediate slot,
 * native retry, or fallback.
 *
 * SELECT/SLICE may consume exact supported storage-layout feeds and produce materialized-layout
 * values. Other affine inputs may be exact resolved zero-offset views produced earlier in the same
 * maximal partition; nested MPSGraph steps materialize those views into dense assigned buffers
 * before downstream custom steps index their physical representation. Their published outputs
 * retain exact Model view geometry. {@code CONTIGUOUS} produces canonical geometry. Metal lowers
 * one complete profile-homogeneous partition as a typed whole-partition program during preparation.
 *
 * <p>A package-private exhaustive catalog describes all 115 schema-eighteen operation kinds as
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
 * selected preparations. After final route and anchor recognition, PREPARE reports a bounded typed
 * structural payload with family, exact epilogue order, member/count facts, schema/generator
 * versions, and canonical digest. Immediately before native execution, RUN reports one bounded
 * invocation plan with route counts, aggregate input/output/internal/splat/workspace bytes,
 * splat/workspace counts, and bounded custom-step
 * summaries. Payloads contain no source/data/scalar bits, pointers, handles, paths, names, or
 * per-buffer records. Structural PREPARE and pre-run planning callback failures disable tracing
 * and are contained without aborting native finalization or execution. Observer runtime failures
 * from outcome reporting also
 * disable later tracing without changing backend outcomes. An observer error propagates only from
 * successful outcome reporting; during failure reporting, the backend failure remains primary and
 * receives a distinct acyclic observer error as a suppressed failure. The ordinary open overload
 * creates no trace producer or trace work. Per-unit close
 * payloads remain deferred because logical units share/refcount native resources and would be
 * misleading lifecycle noise.
 *
 * <p>Task 0060 adds exact profile-common no-gradient replacement SCATTER_ND and signed
 * SLICE_UPDATE, non-overlapping FLOAT64/FLOAT32/BFLOAT16 FOLD_AXIS/FOLD2D/FOLD3D, modular
 * INT32/INT64 PROD, and canonical BOOL ALL/ANY. Task 0066 additionally admits non-overlapping
 * INT32/INT64 FOLD_AXIS; BOOL FOLD_AXIS and integral or BOOL FOLD2D/FOLD3D remain false. Every
 * replacement validates all indices and destination uniqueness before writes; every admitted fold
 * has one output writer and skips conceptual padding. Empty reduction axes are point identities
 * without admitting zero-dimensional extents. LOG_SUM_EXP, STANDARD_DEVIATION, and L2_NORM have
 * forceable structural recipes only and remain production-false. The exact bounded L1_NORM and
 * singleton VARIANCE occurrences described below instead use fixed source-owned custom programs.
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
 * <p>Task 0069 Slices 1 through 3 add accelerator-only no-gradient FLOAT32 rank-one L1 norm,
 * scatter-add, and singleton variance occurrences at wires 114, 70, and 112. L1
 * raw-ABS-transforms every contributor and uses exactly {@code N-1} source-ordered safe additions.
 * Scatter requires axis zero, a materialized canonical INT32/INT64 index feed, and canonical
 * base/update/output roles. The compiler places explicit {@code CONTIGUOUS} between its generated
 * zero-base expansion and Scatter. Its complete CPU index scan precedes every dispatch and
 * mutation; one thread per target loads the raw base once, retains every matching duplicate in
 * source order, and stores once, while unaddressed cells are raw copies. Variance requires input
 * {@code [1]}, axis {@code [0]}, correction zero, and canonical scalar or retained {@code [1]}
 * output. It executes the literal DIV-SUB-MUL-DIV sequence on one custom writer thread. Other
 * variance occurrences retain the existing descriptive direct-MPSGraph path but remain
 * production-false. The shared proof and source/compiler/AIR certificates are pinned under
 * {@code evidence/0069}.
 *
 * <p>Task 0071 adds schema-eighteen typed anchor instructions, generator schema two, and the
 * anchor-epilogue/trace contracts above. Schema and digest bind anchor family, suffix order, ADD
 * side and role, raw scalar/clamp words, profile, types, Shapes, layouts, and gradient facts. Native
 * independently validates the records before one dispatch. The checked Lean model, authenticated
 * source certificate, pinned compiled-MSL/AIR audit, real-device dispatch observer, special-value
 * cases, and public Engine smoke are retained under {@code evidence/0071}.
 *
 * <p>The selected numerical profile participates in partition-plan, route, tuning, decision-codec,
 * and workload identity. Java rejects profile/schema mismatches before native entry. ABI version
 * five retains thirteen exports. Node schema version eighteen is one bounded self-describing
 * route-bearing image over stable type wires {@code 1..6}, operation wires {@code 1..115},
 * attribute wires {@code 0..41}, and complete optional storage-layout geometry. Its fixed
 * 128-byte header binds the exact numerical-profile wire and, only for the custom-program route,
 * the canonical execution-plan extension and manifest digest. Native structural execution covers
 * exactly 101 wires and leaves 14 nonexecutable. Production capability is exactly 86 operation
 * kinds and 29 remain false. Backend-local workload, exact-policy, candidate, compatibility,
 * route-policy, and codec identities are version twenty-seven. Only schema eighteen and identity
 * twenty-seven are accepted; every other schema or identity value fails closed.
 */
package io.github.pho001.synaptik.backend.metal;

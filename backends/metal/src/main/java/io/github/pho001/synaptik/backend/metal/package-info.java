/**
 * Supplies explicit capability, configuration, and lifecycle integration for the Metal backend.
 *
 * <p>{@link io.github.pho001.synaptik.backend.metal.MetalCapabilityProvider} reports an exact
 * occurrence matrix. The common domain covers represented-bit unary operations, seven-carrier
 * affine movement/canonicalization/indexing/replacement, all 49 casts, four-carrier floating
 * classification, BOOL logic, the fourteen non-mixed-low promoted WHERE signatures, stable
 * ordering/top-K/numeric arg-extrema, non-overlapping folds, stored comparisons, maximum pooling,
 * promoted integral MATMUL, and raw initial state. Authenticated affine views remain logical while
 * physical descriptors stay safe; unresolved, empty, unsafe-stride, overlapping, or over-limit
 * geometry remains false. Index roles accept INT32 or INT64 and replacement validates complete
 * bounds and destination uniqueness before mutation.
 *
 * <p>The low-precision capability path checks homogeneous BFLOAT16 and FLOAT16 descriptors and
 * scalar attributes through a FLOAT32 proxy of the frozen, formerly ACCELERATOR support baseline.
 * Direct BFLOAT16/FLOAT16 mixed-low operations are false except explicit CAST;
 * callers establish mixed-low computation with explicit FLOAT32 casts.
 * Every BFLOAT16/FLOAT16 occurrence uses the fixed authenticated {@code CUSTOM_PROGRAM} route;
 * no low-precision partition exposes an MPSGraph, MPS, retry, CPU-fallback, pointwise-generation,
 * or anchor-epilogue candidate. Arithmetic decodes represented inputs exactly to FLOAT32, retains
 * FLOAT32 working values and accumulators, and integer-narrows each observable low result exactly
 * once with round-to-nearest, ties-to-even. Raw-preserving operations move represented bits without
 * a numerical accumulator.
 *
 * <p>The selected low-precision closure includes tensor and scalar arithmetic, comparisons, extrema,
 * SUM/MEAN/MIN/MAX and sum-to-Shape reductions, cumulative SUM/PROD, positive-static MATMUL, MSE,
 * convolution, average pooling, dropout, and the source-owned L1, ScatterAdd, and singleton
 * VARIANCE kinds. Their low-type occurrences use separate {@code lp_*} custom kernels, not the
 * FLOAT32-only Task 0069 specialized kernels. Binary output differentiability is the operand OR;
 * scalar, reciprocal,
 * reduction, scan, pooling, and dropout value roles preserve their declared gradient relationship;
 * comparisons publish no-gradient BOOL. The provider also admits no-gradient INT32/INT64 MATMUL
 * pairs with modular result arithmetic. Canonical host ingress/materialization and bidirectional
 * CPU/Metal transfer accept ranks {@code 0..16} for all seven carriers and resolved positive-stride
 * non-overlapping storage layouts.
 *
 * <p>A partition containing low values or a selected exact custom node uses one fixed custom
 * whole-program resource: fixed kernels and cold-compiled nested existing-node executables consume
 * the compact materialized-slot table in program order behind one Java/native run call. Only the
 * authenticated materialized set receives run-owned buffers; targets remain direct assigned
 * buffers. Eligible existing FLOAT32 routes remain separately qualified, including direct
 * rank-two FLOAT32 MATMUL and bounded generated fusion; low values never enter those
 * MPSGraph/generated/fused routes. Direct {@code NEG}, {@code ABS},
 * {@code FLOOR}, {@code CEIL}, {@code SIGN}, and {@code RELU}
 * operands/outputs and their graph feeds are canonical. Eligible maximal linear canonical
 * FLOAT32 chains of the latter four operations require positive static rank one through sixteen
 * and an unsigned-32-bit positive element count, then deterministically fuse into generated units
 * of length two through eight with one boundary load, one boundary store, and no intermediate
 * slot. Java carries only integer source byte counts; native owns generated source and every
 * source hash. The emitted source identifies generator schema 2; predecessor schema 1 remains
 * incompatible. Native authenticates all frozen fixed-source components and the assembled source
 * before compile, and requires exact strongest available binding/type reflection for all three
 * buffers and every {@code PointMeta} member.
 * <p>Anchor-epilogue fusion applies only to its qualified FLOAT32 custom-program occurrences and
 * preserves exact source order. MATMUL covers
 * the already-admitted positive-static vector, matrix, dot, batched, and right-broadcast batch
 * geometries. It accepts optional literal {@code SCALAR_MUL}, at most one ordinary right-aligned
 * tensor {@code ADD}, and optional terminal {@code RELU} or no-gradient {@code CLAMP}. Conv2d
 * accepts at most one external
 * ADD followed by the same terminal pair and never scalar multiplication. Its intrinsic rank-one
 * {@code [C]} bias remains only the third Conv2d input; an external rank-one addend is {@code [W]},
 * while {@code [1,C,1,1]} is the ordinary channel-broadcast form. Every absorbed intermediate is
 * private, single-consumer, non-target, canonical, and absent from the materialized set.
 * Rank-zero {@code SCALAR_MUL} and {@code RELU} are admitted only in their exact authenticated
 * ordered anchor-member roles; standalone or unrelated custom-program occurrences fail closed.
 * An admitted anchor uses one safe-math dispatch and one final store with no intermediate slot,
 * native retry, or fallback. Fusion is all-or-none at 64 anchors. Structural trace scans at most
 * 65 eligible source anchors; count 65 is the exceeded-cap sentinel and reports the bounded member
 * facts as composed without allocating an unbounded diagnostic list.
 *
 * SELECT/SLICE may consume exact supported storage-layout feeds and produce materialized-layout
 * values. Other affine inputs may be exact resolved zero-offset views produced earlier in the same
 * maximal partition; nested MPSGraph steps materialize those views into dense assigned buffers
 * before downstream custom steps index their physical representation. Their published outputs
 * retain exact Model view geometry. {@code CONTIGUOUS} produces canonical geometry. Metal lowers
 * one complete homogeneous partition as a typed whole-partition program during preparation.
 *
 * <p>A package-private exhaustive catalog describes all 115 schema-twenty operation kinds as
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
 * splat/workspace counts, and bounded custom-step summaries. Splat facts describe the exact
 * validated invocation bindings, including a shared splat prepared by an earlier owner and
 * excluding any source-owned resource not bound by that invocation. Payloads contain no
 * source/data/scalar bits, pointers, handles, paths, names, or per-buffer records. Structural
 * PREPARE and pre-run planning callback failures disable tracing and are contained without
 * aborting native finalization or execution. A disabled producer skips structural trace planning
 * and digest construction on later preparations. When the structural PREPARE callback disables
 * tracing, analysis reuses a matching heuristic plan or constructs the selected no-trace plan once.
 * An enabled forced fallback constructs only the heuristic and traced selected plans, never an
 * eager third copy. Observer runtime failures from outcome reporting also disable later tracing
 * without changing backend outcomes. An observer error propagates only
 * from
 * successful outcome reporting; during failure reporting, the backend failure remains primary and
 * receives a distinct acyclic observer error as a suppressed failure. The ordinary open overload
 * creates no trace producer or trace work. Per-unit close
 * payloads remain deferred because logical units share/refcount native resources and would be
 * misleading lifecycle noise.
 *
 * <p>Task 0060 adds exact no-gradient replacement SCATTER_ND and signed
 * SLICE_UPDATE, non-overlapping FLOAT64/FLOAT32/BFLOAT16/FLOAT16 FOLD_AXIS/FOLD2D/FOLD3D,
 * modular INT32/INT64 PROD, and canonical BOOL ALL/ANY. FOLD_AXIS also admits INT32/INT64;
 * BOOL FOLD_AXIS and integral or BOOL FOLD2D/FOLD3D remain false. Every
 * replacement validates all indices and destination uniqueness before writes; every admitted fold
 * has one output writer and skips conceptual padding. Empty reduction axes are point identities
 * without admitting zero-dimensional extents. LOG_SUM_EXP, STANDARD_DEVIATION, and L2_NORM have
 * forceable structural recipes only and remain production-false. The exact bounded L1_NORM and
 * singleton VARIANCE occurrences described below instead use fixed source-owned custom programs.
 *
 * <p>Task 0062 adds selected same-type canonical positive-rank MSE with NONE, SUM, or MEAN
 * reduction. Native lowering fixes the Model formula to one subtraction, multiplication of that
 * exact difference by itself, and the already qualified full SUM or MEAN reduction. P9 admits
 * homogeneous FLOAT32/BFLOAT16/FLOAT16 descriptors and their compiler-generated gradients.
 * Output differentiability is the exact input logical OR. Scalar inputs,
 * direct mixed-low inputs, and every other normalization or loss kind remain false.
 *
 * <p>Task 0063 adds canonical dense {@code SORT}, {@code ARGSORT}, and positive-K
 * {@code TOP_K} for all seven carriers, plus {@code ARG_MAX} and {@code ARG_MIN} for the six numeric
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
 * MAX_POOL2D}, {@code AVERAGE_POOL2D}, {@code MAX_POOL3D}, and {@code AVERAGE_POOL3D}. P9 admits
 * homogeneous FLOAT32/BFLOAT16/FLOAT16 convolution and pooling wherever the
 * corresponding FLOAT32 occurrence is valid; maximum pooling additionally retains
 * FLOAT64. One-low-plus-FLOAT32 widening convolution remains no-gradient. All six kinds require
 * fully static positive geometry bounded to unsigned 32 bits, exact symmetric padding and fixed
 * layouts, and checked result geometry. Pooling additionally caps the product of kernel extents at
 * 65,536 positions so padding-heavy metadata cannot create unbounded per-output work with tiny
 * tensors. Conv1d and Pool1d enter only through authenticated local singleton-height affine views.
 * Maximum pooling fixes NaN-first, positive-zero-over-negative-zero, first-logical-winner raw-bit
 * publication; average pooling fixes a full-kernel divisor and conceptual positive-zero padding.
 * Conv3d gradients, direct mixed-low operations, and
 * overlap-accumulating generated folds remain fail-closed. Non-overlapping maximum-pool gradients
 * close through the compiler formula.
 *
 * <p>Task 0065 adds zero-input {@code INITIAL_STATE} and canonical
 * FLOAT32/BFLOAT16/FLOAT16 {@code DROPOUT}. INITIAL_STATE publishes one raw {@code INT64[2]}
 * {@code [key,counter]} value whose words are interpreted unsigned. DROPOUT emits the scaled value,
 * canonical BOOL saved mask, and advanced state through one fixed custom-program route. Its private
 * SplitMix64 V1 identity uses the exact top-53-bit threshold policy, raw positive-zero drops,
 * FLOAT32 division and multiplication, one final low narrowing, and modulo-{@code 2^64} counter
 * advancement by the logical element count. It makes no entropy, cryptographic-quality, or
 * cross-backend portable-stream claim. Evaluation bypasses DROPOUT; generated backward reuses the
 * saved mask without consuming state; recurrent operations remain fail-closed.
 *
 * <p>Task 0066 closes the selected dtype/layout/gradient occurrence gap without adding a kind,
 * wire, schema field, export, or route. All selected partitions use deterministic {@code
 * CUSTOM_PROGRAM}; native and JVM preflight independently authenticate logical versus physical
 * layouts, 49 cast semantics, promoted predicates/WHERE, INT64 index parity, exact saved-role
 * liveness, and one-writer replacement/window geometry. Dynamic, empty, unsafe, overlapping,
 * additive-scatter, attention, and recurrent blockers remain fail-closed.
 *
 * <p>Task 0069 Slices 1 through 3 add narrow no-gradient FLOAT32 rank-one L1 norm,
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
 * production-false. The retained Task 0069 proof and source/compiler/AIR audits under
 * {@code evidence/0069} concern these FLOAT32 specialized kernels; they do not certify the
 * low-type kernels or authenticate the current schema-twenty image.
 *
 * <p>The current schema-twenty image retains the typed anchor instructions and generator-schema-two
 * lineage introduced by Task 0071. Its schema and digest bind anchor family, suffix order, ADD
 * side and role, raw scalar/clamp words, types, Shapes, layouts, and gradient facts. Native
 * independently validates the records before one dispatch. The checked Lean model, authenticated
 * source audit, pinned compiled-MSL/AIR audit, real-device dispatch observer, special-value
 * cases, and public Engine smoke retained under {@code evidence/0071}. That historical evidence
 * does not by itself authenticate the current schema-twenty image or qualify a post-cutover route.
 *
 * <p>ABI version seven has thirteen exports; the former certification-environment query is absent.
 * Node schema version twenty is one bounded self-describing route-bearing image over stable type wires
 * {@code 1..7}, operation wires {@code 1..115}, attribute wires {@code 0..41}, and complete
 * optional storage-layout geometry. Its fixed 124-byte header binds the route and, only for the
 * custom-program route, the canonical execution-plan extension and manifest
 * digest. Native structural execution covers exactly 103 wires and leaves 12 nonexecutable.
 * Production capability is exactly 88 operation kinds and 27 remain false. The no-gradient
 * canonical FLOAT32 EXP occurrence uses one fixed direct MPSGraph exponent route, including an
 * internal MPSGraph boundary step after an explicit low-to-FLOAT32 cast in a custom partition.
 * Separately qualified canonical BFLOAT16 and FLOAT16 EXP occurrences use typed fixed custom
 * kernels with FLOAT32 working evaluation and one low narrowing; gradient-bearing EXP remains
 * unsupported. Bounded canonical no-gradient FLOAT32 SIGMOID uses wire 64 with an exact
 * stored-bit sign guard and the Model's stable branch formula in a composed MPSGraph step;
 * BFLOAT16/FLOAT16 SIGMOID stay false. An explicit low-to-FLOAT32 cast may put that FLOAT32
 * step inside a custom partition without making SIGMOID low-valued. Backend-local workload,
 * exact-policy, candidate, compatibility, route-policy, and codec identities are version
 * thirty-three. Only schema twenty and identity thirty-three are accepted; every other schema or
 * identity value fails closed.
 */
package io.github.pho001.synaptik.backend.metal;

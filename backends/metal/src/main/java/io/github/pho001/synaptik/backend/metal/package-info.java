/**
 * Supplies explicit capability, configuration, and lifecycle integration for the Metal backend.
 *
 * <p>{@link io.github.pho001.synaptik.backend.metal.MetalCapabilityProvider} reports a common exact
 * domain under both profiles: parameterless {@code NEG}, {@code ABS}, {@code FLOOR}, {@code CEIL},
 * {@code SIGN}, and {@code RELU}; {@code RESHAPE}, {@code EXPAND}, {@code PERMUTE},
 * {@code EXPAND_DIMS}, {@code SQUEEZE}; the explicit {@code CONTIGUOUS} canonicalization barrier;
 * canonical FLOAT32 {@code UNFOLD_AXIS} materializing one normalized axis into exact floor-count
 * windows with a trailing size axis; canonical positive-rank {@code FLOAT32} data {@code GATHER}
 * with canonical {@code INT32} indices; canonical positive-rank {@code INT32}-to-{@code BOOL}
 * {@code ONE_HOT}; canonical positive-rank {@code SCATTER_ELEMENTS} replacement with ordered
 * {@code FLOAT32}/{@code INT32}/{@code FLOAT32} data, indices, and updates; exact FLOAT32
 * classification, BOOL logic, and FLOAT32 {@code WHERE}. The four discrete unary operations and
 * the seven BOOL-domain operations require canonical rank {@code 1..16}; logic uses right-aligned
 * broadcasting and WHERE uses branch-first then condition broadcasting. UNFOLD_AXIS is bounded to
 * input ranks {@code 1..15} and sizes {@code 1..16}, requires a positive step and
 * size-at-most-axis extent, returns a fresh canonical rank-plus-one value, and preserves exact
 * represented FLOAT32 bits. Scatter accepts only {@code ScatterReduction.NONE}, returns a
 * canonical data-shaped value, preserves exact represented bits, and validates complete bounds
 * then target uniqueness before dispatch.
 * The common domain also includes the nineteen proved canonical no-gradient {@code CAST} carrier
 * pairs; exact raw {@code GATHER_ELEMENTS} and {@code GATHER_ND} over all six data carriers with
 * {@code INT32} or {@code INT64} indices; all-six-carrier {@code SELECT}, positive-step
 * {@code SLICE}, {@code PAD}, {@code CONCAT}, {@code STACK}, and {@code TILE}; and
 * floating-carrier {@code UNFOLD2D}/{@code UNFOLD3D}. SELECT and SLICE require fully static,
 * resolved, positive-stride, non-overlapping input and output storage layouts with exact
 * offset/stride/span relationships; unresolved, zero-stride, negative-stride, overlapping, empty,
 * or gradient-bearing occurrences remain false. The other routes require fully static canonical
 * rank {@code 0..16} descriptors where Model permits the rank, exact Model-derived Shapes and
 * attributes, and bounds validation before any indexed write. They use one fixed custom-program
 * route with no fallback, retry, timing, or autotuning.
 * {@code ACCELERATOR} additionally admits tensor {@code ADD}, {@code SUB}, {@code MUL},
 * {@code DIV}, {@code MIN}, and {@code MAX}; all six binary comparisons with canonical one-byte
 * {@code BOOL} output; exact FLOAT32 scalar {@code MIN}, {@code MAX}, and fused {@code CLAMP};
 * canonical {@code FLOAT32} {@code SUM}, {@code MEAN}, {@code MIN}, {@code MAX}, and
 * binding-resolved {@code SUM_TO_SHAPE}; all four exclusive/reverse modes of {@code CUM_SUM} and
 * {@code CUM_PROD}; and every positive-static FLOAT32 {@code MATMUL} vector, matrix, batched, and
 * right-aligned broadcast geometry. Both profiles admit no-gradient INT32/INT64 MATMUL pairs with
 * INT64-dominant promotion and modular result arithmetic. Accelerator additionally admits
 * no-gradient BFLOAT16/FLOAT32 and FLOAT32/BFLOAT16 operands with FLOAT32 result. Reductions support
 * full, normalized single-axis, ordered normalized multi-axis including empty, and exact
 * keep-dimensions forms; reduction inputs remain positive-rank while their target may be rank
 * zero. Canonical host ingress/materialization and bidirectional CPU/Metal transfer accept ranks
 * {@code 0..16} for all six public data types with exact widths; transfer additionally accepts
 * resolved positive-stride non-overlapping storage layouts and rejects unresolved, zero-stride,
 * negative-stride, or overlapping geometry. Logical BOOL bytes are validated while layout holes
 * remain uninterpreted. Exact BOOL operation ingress is restricted further to positive-rank
 * canonical bytes; custom outputs are written as exact zero or one, and owned MPSGraph ONE_HOT
 * output is validated by its typed selector contract. Accelerator binary inputs and outputs are
 * canonical dense non-views and use exact right-aligned broadcasting. Each MATMUL operand is
 * canonical or the authenticated exact local identity-prefix, last-two-axis {@code PERMUTE} of a
 * canonical source; its output is canonical. FLOAT32 output gradient metadata is the operand OR;
 * integral and mixed-carrier rows are no-grad. A partition containing any shared exact custom node
 * or a MATMUL outside the retained direct rank-two FLOAT32 MPSGraph slice selects one fixed custom
 * whole-program resource: fixed custom kernels and cold-compiled nested existing-node executables
 * consume the stable declared value table in program order behind one Java/native run call.
 * Every logical intermediate is an assigned run-owned buffer; targets remain the direct assigned
 * buffers.
 * Direct {@code NEG}, {@code ABS}, {@code FLOOR}, {@code CEIL}, {@code SIGN}, and {@code RELU}
 * operands/outputs and their graph feeds are canonical. SELECT/SLICE may consume exact supported
 * storage-layout feeds and produce materialized-layout values. Other affine inputs may be exact
 * resolved zero-offset views produced earlier in the same maximal partition; their outputs retain
 * exact Model view geometry. {@code CONTIGUOUS} produces canonical geometry. Metal lowers one
 * complete profile-homogeneous partition as a typed whole-partition program during preparation.</p>
 *
 * <p>A package-private exhaustive catalog describes all 115 schema-fifteen operation kinds as
 * MPSGraph {@code DIRECT}, {@code COMPOSED}, or {@code UNAVAILABLE} and custom-kernel
 * {@code AVAILABLE}, {@code PENDING}, or {@code UNAVAILABLE_WITH_PROOF}, with closed source
 * reasons. It is cold descriptive metadata only: capability remains authoritative and the catalog
 * is never consulted by Runtime. One package-private canonical prepared-route identity owns the
 * stable candidate wires {@code 1..3} and the MPSGraph/custom-kernel family. Every plan retains
 * exactly one such identity before finalization; candidate encoding delegates to it, and
 * finalization, tracing, cold binding, and execution never replace it. Package-private tests may
 * force only a route already approved by the freshly validated exact candidate batch. Production
 * exposes no force, selector, fallback, retry, or hot route decision.</p>
 *
 * <p>{@link io.github.pho001.synaptik.backend.metal.MetalBackendConfiguration} names one explicit
 * native bridge. {@link io.github.pho001.synaptik.backend.metal.MetalBackendIntegration} opens and
 * owns one default-device native context and supplies availability, partition preparation,
 * retained {@link io.github.pho001.synaptik.backend.metal.MetalLocalWorkloadTuning local-workload}
 * and {@link io.github.pho001.synaptik.backend.metal.MetalCompletePlanTuning complete-plan}
 * tuning, backend-local schedule assembly, a shared physical schedule contribution, checked
 * native upload/download binders, caller-host ingress, and bounded canonical host materialization
 * to Engine's explicit builder. Local tuning admits only an exact singleton NEG having both the
 * custom-kernel and MPSGraph routes. Complete-plan tuning authenticates that local selection and
 * exposes one whole-plan candidate fixed to the selected route. Both collaborations use opaque
 * exact associations, fresh authoritative preparation, and session-only compatibility; they
 * perform no execution, measurement, cache input/output, or fallback-policy work. Exact prepared
 * SELECT/SLICE targets materialize by gathering logical elements from authenticated backend-private
 * physical storage using their exact positive strides and offset; unrelated prefix and gap bytes
 * remain intact. Other prepared affine targets retain the existing dense represented-order
 * publication path. Engine uses these contributions and typed binders for the bounded
 * bidirectional CPU/Metal mixed-owner schedule.
 * Native internals remain package-private.
 * There is no library discovery, CPU fallback, or backend-global integration.</p>
 *
 * <p>The optional
 * {@link MetalBackendIntegration#open(MetalBackendConfiguration, MetalTraceObserver) traced open}
 * overload retains but never closes one caller-owned thread-safe observer. The same trace producer
 * is retained by ordinary, local-trial, complete-plan-trial, and selected preparations. It reports
 * only final preparation and native-invocation outcomes with stream-local correlations; tuning
 * eligibility validation creates no ghost trace unit. Observer runtime failures disable later
 * tracing without changing backend outcomes. An observer error propagates from success reporting;
 * during failure reporting, the backend failure remains primary and receives a distinct acyclic
 * observer error as a suppressed failure. The ordinary open overload creates no trace producer or
 * trace work.</p>
 *
 * <p>Task 0060 adds exact profile-common no-gradient replacement SCATTER_ND and signed
 * SLICE_UPDATE, non-overlapping FLOAT64/FLOAT32/BFLOAT16 FOLD_AXIS/FOLD2D/FOLD3D, modular
 * INT32/INT64 PROD, and canonical BOOL ALL/ANY. Integral and BOOL folds remain false. Every
 * replacement validates all indices and destination uniqueness before writes; every admitted fold
 * has one output writer and skips conceptual padding. Empty reduction axes are point identities
 * without admitting zero-dimensional extents. LOG_SUM_EXP, VARIANCE,
 * STANDARD_DEVIATION, L1_NORM, and L2_NORM have forceable structural recipes only and remain
 * production-false.</p>
 *
 * <p>The selected numerical profile participates in partition-plan, route, tuning,
 * decision-codec, and workload identity. Java rejects profile/schema mismatches before native
 * entry. ABI version five retains thirteen exports. Node schema version fifteen is one bounded
 * self-describing route-bearing image over stable type wires {@code 1..6}, operation wires
 * {@code 1..115}, attribute wires {@code 0..41}, and complete optional storage-layout geometry.
 * Native structural execution covers exactly 87 wires and leaves 28 nonexecutable. Production
 * capability is exactly 69 operation kinds and 46 remain false. Backend-local workload,
 * exact-policy, candidate, compatibility, route-policy, and codec identities are version
 * seventeen; schema fourteen and identity version sixteen fail closed.</p>
 */
package io.github.pho001.synaptik.backend.metal;

/**
 * Supplies explicit capability, configuration, and lifecycle integration for the Metal backend.
 *
 * <p>{@link io.github.pho001.synaptik.backend.metal.MetalCapabilityProvider} reports a common exact
 * domain under both profiles: parameterless {@code NEG} and {@code ABS}, {@code RESHAPE},
 * {@code EXPAND}, {@code PERMUTE}, {@code EXPAND_DIMS}, {@code SQUEEZE}, the explicit
 * {@code CONTIGUOUS} canonicalization barrier, canonical FLOAT32 {@code UNFOLD_AXIS} materializing
 * one normalized axis into exact floor-count windows with a trailing size axis, canonical
 * positive-rank {@code FLOAT32} data {@code GATHER} with canonical {@code INT32} indices,
 * canonical positive-rank {@code INT32}-to-{@code BOOL} {@code ONE_HOT}, and canonical
 * positive-rank {@code SCATTER_ELEMENTS} replacement with ordered
 * {@code FLOAT32}/{@code INT32}/{@code FLOAT32} data, indices, and updates. UNFOLD_AXIS is bounded
 * to input ranks {@code 1..15} and sizes {@code 1..16}, requires a positive step and
 * size-at-most-axis extent, returns a fresh canonical rank-plus-one value, and preserves exact
 * represented FLOAT32 bits. Scatter accepts only {@code ScatterReduction.NONE}, returns a
 * canonical data-shaped value, preserves exact represented bits, and validates complete bounds
 * then target uniqueness before dispatch. {@code ACCELERATOR} additionally admits tensor
 * {@code ADD}, {@code SUB}, {@code MUL}, {@code DIV}, {@code MIN}, and {@code MAX}; all six
 * binary comparisons with canonical one-byte {@code BOOL} output; exact FLOAT32 scalar
 * {@code MIN}, {@code MAX}, and fused {@code CLAMP}; canonical {@code FLOAT32} {@code SUM},
 * {@code MEAN}, {@code MIN}, {@code MAX}, and binding-resolved {@code SUM_TO_SHAPE}; all four
 * exclusive/reverse modes of {@code CUM_SUM} and {@code CUM_PROD}; and positive static rank-two
 * {@code FLOAT32} {@code MATMUL}. Reductions support full, normalized single-axis, ordered
 * normalized multi-axis including empty, and exact keep-dimensions forms; reduction inputs remain
 * positive-rank while their target may be rank zero. Canonical host ingress/materialization and
 * bidirectional CPU/Metal transfer accept ranks {@code 0..16} for all six public data types with
 * exact widths, canonical non-view geometry, and strict BOOL-byte validation. This transfer
 * foundation does not widen Metal operation capability: current BOOL operation feeds/consumers
 * remain unsupported. Accelerator binary inputs and outputs are canonical dense non-views and use
 * exact right-aligned broadcasting.
 * Each MATMUL operand is canonical or the authenticated exact local rank-two
 * {@code PERMUTE [1,0]} of a canonical source; its output is canonical. Strict MATMUL remains
 * unsupported. A partition containing any Task-0052 node selects one fixed custom whole-program
 * resource: fixed custom kernels and cold-compiled nested existing-node executables consume the
 * stable declared value table in program order behind one Java/native run call. Every logical
 * intermediate is an assigned run-owned buffer; targets remain the direct assigned buffers.
 * Every direct {@code NEG} or {@code ABS} operand/output and graph feed is canonical;
 * affine inputs may also be exact resolved zero-offset views produced earlier in the same maximal
 * partition. Affine outputs retain their exact Model view geometry, while {@code CONTIGUOUS}
 * produces canonical geometry. Metal lowers one complete profile-homogeneous partition as a typed
 * whole-partition program during preparation.</p>
 *
 * <p>A package-private exhaustive catalog describes all 115 schema-thirteen operation kinds as
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
 * affine-view targets may materialize from their authenticated backend-private dense
 * represented-order buffers without rewriting logical view descriptors. Engine uses those
 * contributions and typed binders for the bounded bidirectional CPU/Metal mixed-owner schedule.
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
 * <p>The selected numerical profile participates in partition-plan, route, tuning,
 * decision-codec, and workload identity. Java rejects profile/schema mismatches before native
 * entry. ABI version five retains thirteen exports. Node schema version thirteen is one bounded
 * self-describing image over stable type wires {@code 1..6}, operation wires {@code 1..115}, and
 * attribute wires {@code 0..41}; structural registry coverage does not widen the current
 * executable capability of operation wires {@code 1..34}. Backend-local workload, exact-policy,
 * candidate, compatibility, route-policy, and codec identities are version fourteen.</p>
 */
package io.github.pho001.synaptik.backend.metal;

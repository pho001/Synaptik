/**
 * Supplies explicit capability, configuration, and lifecycle integration for the Metal backend.
 *
 * <p>{@link io.github.pho001.synaptik.backend.metal.MetalCapabilityProvider} reports a common exact
 * domain under both profiles: parameterless {@code NEG} and {@code ABS}, {@code RESHAPE},
 * {@code EXPAND}, {@code PERMUTE}, {@code EXPAND_DIMS}, {@code SQUEEZE}, the explicit
 * {@code CONTIGUOUS} canonicalization barrier, canonical positive-rank {@code FLOAT32} data
 * {@code GATHER} with canonical {@code INT32} indices, canonical positive-rank
 * {@code INT32}-to-{@code BOOL} {@code ONE_HOT}, and canonical positive-rank
 * {@code SCATTER_ELEMENTS} replacement with ordered {@code FLOAT32}/{@code INT32}/{@code FLOAT32}
 * data, indices, and updates. Scatter accepts only {@code ScatterReduction.NONE}, returns a
 * canonical data-shaped value, preserves exact represented bits, and validates complete bounds
 * then target uniqueness before dispatch. {@code ACCELERATOR} additionally admits tensor
 * {@code ADD}, {@code SUB}, {@code MUL}, and {@code DIV}, canonical {@code FLOAT32} {@code SUM},
 * {@code MEAN}, and binding-resolved {@code SUM_TO_SHAPE}, and positive static rank-two
 * {@code FLOAT32} {@code MATMUL}. Reductions support full, normalized single-axis, ordered
 * normalized multi-axis including empty, and exact keep-dimensions forms. Feeds and ordinary
 * outputs are fully static positive-rank {@code 1..16}; a locally produced reduction target may be
 * rank zero and materializes as exactly four canonical bytes. Caller ingress accepts only exact
 * canonical {@code FLOAT32} and {@code INT32}; local canonical {@code BOOL} results may publish as
 * exact one-byte elements. CPU/Metal transfer remains canonical non-view {@code FLOAT32} only.
 * Accelerator binary inputs and outputs are canonical dense non-views and use exact right-aligned
 * broadcasting. Each MATMUL operand is canonical or the authenticated exact local rank-two
 * {@code PERMUTE [1,0]} of a canonical source; its output is canonical. Strict MATMUL remains
 * unsupported. Every direct {@code NEG} or {@code ABS} operand/output and graph feed is canonical;
 * affine inputs may also be exact resolved zero-offset views produced earlier in the same maximal
 * partition. Affine outputs retain their exact Model view geometry, while {@code CONTIGUOUS}
 * produces canonical geometry. Metal lowers one complete profile-homogeneous partition as a typed
 * whole-partition program during preparation.</p>
 *
 * <p>{@link io.github.pho001.synaptik.backend.metal.MetalBackendConfiguration} names one explicit
 * native bridge. {@link io.github.pho001.synaptik.backend.metal.MetalBackendIntegration} opens and
 * owns one default-device native context and supplies availability, partition preparation,
 * backend-local schedule assembly, a shared physical schedule contribution, checked native
 * upload/download binders, caller-host ingress, and bounded canonical host materialization to
 * Engine's explicit builder. Exact prepared affine-view targets may materialize from their
 * authenticated backend-private dense represented-order buffers without rewriting logical view
 * descriptors or widening the canonical non-view CPU/Metal transfer predicate. Engine uses those
 * contributions and binders for the bounded bidirectional CPU/Metal mixed-owner schedule. Native
 * internals remain package-private. There is no library discovery, CPU fallback, or backend-global
 * integration.</p>
 *
 * <p>The selected numerical profile participates in partition-plan, route, tuning,
 * decision-codec, and workload identity. Java rejects profile/schema mismatches before native
 * entry. ABI version four remains stable; node schema version ten retains wires {@code 1..17}
 * and appends {@code SCATTER_ELEMENTS=18} with its third input in the typed auxiliary cell.
 * Backend-local workload, exact-policy, candidate, compatibility, route-policy, and codec
 * identities are version eleven.</p>
 */
package io.github.pho001.synaptik.backend.metal;

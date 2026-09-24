/**
 * Supplies explicit capability, configuration, and lifecycle integration for the Metal backend.
 *
 * <p>{@link io.github.pho001.synaptik.backend.metal.MetalCapabilityProvider} reports support only
 * for parameterless {@code NEG}, {@code RESHAPE}, {@code EXPAND}, {@code PERMUTE},
 * {@code EXPAND_DIMS}, {@code SQUEEZE}, and the explicit {@code CONTIGUOUS} canonicalization
 * barrier. Binary arithmetic is intentionally not advertised. The domain is fully static,
 * positive rank {@code 1..16} {@code FLOAT32}. Graph feeds and {@code NEG} operands are canonical
 * dense non-views. Affine inputs may also be exact resolved zero-offset views produced earlier in
 * the same maximal partition; affine outputs retain the exact Model view geometry.
 * {@code CONTIGUOUS} produces canonical geometry and is required before a view feeds {@code NEG}.
 * Metal lowers the complete partition as one typed whole-partition program during preparation.</p>
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
 * <p>Capability and preparation currently support only {@code STRICT_IEEE} and fail closed for
 * {@code ACCELERATOR}. The selected numerical profile participates in partition-plan, route,
 * tuning, decision-codec, and workload identity; no native ABI or MPSGraph schema is changed.</p>
 */
package io.github.pho001.synaptik.backend.metal;

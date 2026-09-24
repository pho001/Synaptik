/**
 * Supplies explicit capability, configuration, and lifecycle integration for the Metal backend.
 *
 * <p>{@link io.github.pho001.synaptik.backend.metal.MetalCapabilityProvider} reports two disjoint
 * profile-qualified domains. {@code STRICT_IEEE} retains parameterless {@code NEG},
 * {@code RESHAPE}, {@code EXPAND}, {@code PERMUTE}, {@code EXPAND_DIMS}, {@code SQUEEZE}, and the
 * explicit {@code CONTIGUOUS} canonicalization barrier. {@code ACCELERATOR} admits only tensor
 * {@code ADD}, {@code SUB}, {@code MUL}, and {@code DIV}. Every descriptor is fully static,
 * positive rank {@code 1..16} {@code FLOAT32}. Accelerator binary inputs and outputs are canonical
 * dense non-views and use exact right-aligned broadcasting. Strict graph feeds and {@code NEG}
 * operands are canonical; strict affine inputs may also be exact resolved zero-offset views
 * produced earlier in the same maximal partition. Strict affine outputs retain their exact Model
 * view geometry, while {@code CONTIGUOUS} produces canonical geometry. Metal lowers one complete
 * profile-homogeneous partition as a typed whole-partition program during preparation.</p>
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
 * entry. ABI version four remains stable; node schema version five adds ordered binary wires.</p>
 */
package io.github.pho001.synaptik.backend.metal;

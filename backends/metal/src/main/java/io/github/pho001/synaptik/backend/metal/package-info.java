/**
 * Supplies explicit capability, configuration, and lifecycle integration for the Metal backend.
 *
 * <p>{@link io.github.pho001.synaptik.backend.metal.MetalCapabilityProvider} reports support only
 * for unary {@code NEG} occurrences whose input and output are equal-shape, fully static,
 * positive rank {@code 1..16}, resolved dense-contiguous, non-view, zero-offset {@code FLOAT32}
 * descriptors with equal gradient-eligibility flags. Planning may consequently form one maximal
 * Metal-owned partition from adjacent supported occurrences; the backend lowers that complete
 * partition during preparation.</p>
 *
 * <p>{@link io.github.pho001.synaptik.backend.metal.MetalBackendConfiguration} names one explicit
 * native bridge. {@link io.github.pho001.synaptik.backend.metal.MetalBackendIntegration} opens and
 * owns one default-device native context and supplies availability, partition preparation,
 * backend-local schedule assembly, a shared physical schedule contribution, checked native
 * upload/download binders, caller-host ingress, canonical host materialization, and cleanup to
 * Engine's explicit builder. Engine uses those contributions and binders for the bounded
 * bidirectional CPU/Metal mixed-owner schedule. Native internals remain package-private. There is
 * no library discovery, CPU fallback, or backend-global integration.</p>
 */
package io.github.pho001.synaptik.backend.metal;

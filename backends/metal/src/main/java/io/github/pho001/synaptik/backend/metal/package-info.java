/**
 * Supplies the explicitly constructed public capability surface for the Metal backend.
 *
 * <p>{@link io.github.pho001.synaptik.backend.metal.MetalCapabilityProvider} reports support only
 * for unary {@code NEG} occurrences whose input and output are equal-shape, fully static,
 * positive rank {@code 1..16}, resolved dense-contiguous, non-view, zero-offset {@code FLOAT32}
 * descriptors with equal gradient-eligibility flags. Planning may consequently form one maximal
 * Metal-owned partition from adjacent supported occurrences; the backend lowers that complete
 * partition during preparation.</p>
 *
 * <p>The capability provider is the package's only public type. Device discovery, native-library
 * loading, preparation, executable ownership, storage, and execution remain package-private.
 * There is no public Metal Engine composition, mixed-owner composition, automatic fallback, or
 * backend discovery surface.</p>
 */
package io.github.pho001.synaptik.backend.metal;

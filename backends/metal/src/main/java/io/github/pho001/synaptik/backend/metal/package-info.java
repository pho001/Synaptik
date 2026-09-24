/**
 * Supplies explicit capability, configuration, and lifecycle integration for the Metal backend.
 *
 * <p>{@link io.github.pho001.synaptik.backend.metal.MetalCapabilityProvider} reports support only
 * for parameterless {@code NEG}, {@code ADD}, {@code SUB}, {@code MUL}, and {@code DIV}, plus
 * terminal {@code RESHAPE}, {@code EXPAND}, {@code PERMUTE}, {@code EXPAND_DIMS}, and
 * {@code SQUEEZE} occurrences. The domain is fully static, positive rank {@code 1..16}
 * {@code FLOAT32}. Elementwise descriptors and affine inputs are canonical dense-contiguous
 * non-views; affine outputs retain the exact Model view shape, offset, and strides. Planning may
 * form one maximal Metal-owned partition from any supported mixture, which the backend lowers as
 * one typed whole-partition program during preparation.</p>
 *
 * <p>{@link io.github.pho001.synaptik.backend.metal.MetalBackendConfiguration} names one explicit
 * native bridge. {@link io.github.pho001.synaptik.backend.metal.MetalBackendIntegration} opens and
 * owns one default-device native context and supplies availability, partition preparation,
 * backend-local schedule assembly, a shared physical schedule contribution, checked native
 * upload/download binders, caller-host ingress, and bounded canonical host materialization to
 * Engine's explicit builder. Exact prepared affine targets may materialize from their
 * authenticated backend-private dense represented-order buffers without rewriting logical view
 * descriptors or widening the canonical non-view CPU/Metal transfer predicate. Engine uses those
 * contributions and binders for the bounded bidirectional CPU/Metal mixed-owner schedule. Native
 * internals remain package-private. There is no library discovery, CPU fallback, or backend-global
 * integration.</p>
 */
package io.github.pho001.synaptik.backend.metal;

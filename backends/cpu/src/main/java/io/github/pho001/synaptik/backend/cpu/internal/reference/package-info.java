/**
 * Owns the unsupported scalar conformance and fail-closed reference realization.
 *
 * <p>The reference code expresses the exact fused arithmetic and shared scalar error-function
 * behavior used to verify generated results. It is test and conformance support: it is not a Model
 * operation interpreter, canonical-IR interpreter, Runtime fallback dispatcher, or independently
 * selectable numerical route.
 *
 * <p>Reference evaluation is outside the generated Runtime hot path and owns no buffers, slots, or
 * artifact lifecycle.
 */
package io.github.pho001.synaptik.backend.cpu.internal.reference;

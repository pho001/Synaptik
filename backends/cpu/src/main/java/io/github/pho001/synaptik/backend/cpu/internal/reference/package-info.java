/**
 * Owns the unsupported scalar conformance and fail-closed reference realization.
 *
 * <p>The reference code evaluates the already-lowered typed CPU opcode sequence, including the
 * shared scalar error-function behavior, over the same normalized heap/segment access bindings
 * used by generated code across all five access regimes. It is test and conformance support: it
 * does not interpret Model operations, run inside Runtime, dispatch a Runtime fallback, or define
 * an independently selectable numerical route.
 *
 * <p>Reference evaluation is outside the generated Runtime hot path and owns no buffers, slots, or
 * artifact lifecycle.
 */
package io.github.pho001.synaptik.backend.cpu.internal.reference;

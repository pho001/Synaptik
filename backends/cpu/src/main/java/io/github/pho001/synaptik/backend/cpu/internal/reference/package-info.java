/**
 * Owns the unsupported scalar conformance and fail-closed reference realization.
 *
 * <p>The reference code evaluates the already-lowered typed CPU opcode sequence, including the
 * shared scalar error-function behavior, primitive floating division, represented-value extrema,
 * first-class clamp, direct Tensor power, canonical-BOOL logic, the closed same-typed
 * FLOAT32/FLOAT64 unary matrix, and selected exact/default scalar-power realization, over the same
 * normalized heap/segment access bindings
 * used by generated code across all five access regimes. It is test and conformance support: it
 * does not interpret Model operations, run inside Runtime, dispatch a Runtime fallback, or define
 * an independently selectable numerical route. Its pure error-function and activation helpers are
 * also callable by generated scalar bytecode, so the generated and reference realizations share
 * one fixed formula and exceptional-value policy.
 * The reference realization also evaluates compact static movement and window mappings plus
 * functional SLICE_UPDATE over unique boundary carriers for generated/reference differential
 * tests; it copies represented bits, selects base outside the update region, and does not define
 * a Runtime fallback route.
 * It also validates and evaluates the four compact indexing mappings independently for
 * generated/reference differential tests, preserving deterministic first-invalid order and
 * strict no-wrap/no-clamp bounds meaning.
 *
 * <p>Reference evaluation is outside the generated Runtime hot path and owns no buffers, slots, or
 * artifact lifecycle.
 */
package io.github.pho001.synaptik.backend.cpu.internal.reference;

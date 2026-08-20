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
 * It also validates and evaluates SCATTER_ELEMENTS, Gather-compatible SCATTER_ADD, and SCATTER_ND
 * independently for generated/reference differential tests. The oracle preserves functional base
 * semantics, deterministic bounds-before-duplicates failure, represented reductions, and exact
 * once-rounded floating products without serving as generated execution or a Runtime fallback.
 * It independently evaluates FOLD_AXIS and FOLD2D from logical coordinates, positive-zero
 * initialization, padding exclusion, and represented sequential addition without sharing the
 * generated packed-coordinate walk.
 * It independently evaluates stable SORT, ARGSORT, and TOP_K with primitive-index insertion,
 * fixed NaN-last and signed-zero order, logical INT64 coordinates, represented-bit value copies,
 * and deterministic increasing-index order for unsorted selected pairs.
 *
 * <p>The explicit-state random oracle independently implements the CPU-private V1 word mapping,
 * top-53-bit uniform conversion, exact keep comparison, FLOAT64/FLOAT32 scaling, canonical BOOL
 * mask, and modulo state advancement for generated differential coverage.</p>
 *
 * <p>The cumulative-scan oracle independently reconstructs logical slice coordinates and applies
 * forward/reverse, inclusive/exclusive typed accumulation across FLOAT64, FLOAT32, BFLOAT16,
 * INT32, and INT64. It shares neither the generated packed-coordinate walk nor the emitter.</p>
 *
 * <p>The ordinary aggregate oracle independently maps full, single-axis, and multi-axis logical
 * coordinates and applies exact numerical sum/product state with independent integer/rational
 * ties-to-even conversion, modular integral arithmetic, exact identities, first-NaN bits,
 * signed-zero extrema, signed integral order, and canonical Boolean folds. It calls no production
 * aggregate body, emitter rounding helper, packer, or lowering coordinate helper.</p>
 *
 * <p>Reference evaluation is outside the generated Runtime hot path and owns no buffers, slots, or
 * artifact lifecycle.
 */
package io.github.pho001.synaptik.backend.cpu.internal.reference;

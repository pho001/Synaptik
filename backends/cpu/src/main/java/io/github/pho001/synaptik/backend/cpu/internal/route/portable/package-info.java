/**
 * Defines unsupported portable-route realization facts over common CPU lowering.
 *
 * <p>The route plan pairs route-independent canonical IR with the exact Class-File specialization
 * selected by common CPU analysis. It may be consumed by artifact realization and finalization,
 * but it does not interpret graph semantics, own the scalar reference implementation, select a
 * native provider, or declare resource lifetime.
 *
 * <p>The plan is immutable cold-path state; route dispatch never occurs inside the generated loop.
 */
package io.github.pho001.synaptik.backend.cpu.internal.route.portable;

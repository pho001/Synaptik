package io.github.pho001.synaptik.prepare.analysis;

/**
 * Marks one concrete backend's immutable result of partition analysis.
 *
 * <p>A backend-owned implementation retains its selected lowering, route, and private
 * configuration for a later finalization stage. Shared Prepare carries the plan opaquely and
 * never inspects, downcasts, copies, or interprets it. The plan is not an executable, resource
 * handle, assigned slot, physical allocation, or mutable tuning result.</p>
 */
public interface BackendPreparationPlan {}

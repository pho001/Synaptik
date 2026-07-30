package io.github.pho001.synaptik.prepare.analysis;

/**
 * Marks one concrete backend's immutable inputs to partition analysis.
 *
 * <p>A backend-owned implementation may combine its resolved target capabilities, applicable
 * configuration, and a compatible cached decision. Shared Prepare retains the object opaquely:
 * it does not inspect, downcast, copy, or interpret backend-private fields. Implementations must
 * therefore expose immutable state and must not use this role to smuggle executable, physical
 * resource, or mutable cache state across the analysis boundary.</p>
 */
public interface BackendAnalysisInputs {}

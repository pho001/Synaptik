package io.github.pho001.synaptik.backend.contract;

/**
 * Marks a closed family of immutable hard-eligibility targets for later backend planning.
 *
 * <p>A requirement contains exactly one requested target. A {@link BackendIdRequirement} targets
 * later ownership by an equal backend identity. A {@link BackendDeviceIdRequirement} targets an
 * equal backend-scoped device identity and therefore also its owning backend. A
 * {@link DeviceClassRequirement} accepts any later eligible device in the requested coarse
 * class.</p>
 *
 * <p>Later configuration decides whether to supply a requirement. Later planning combines the
 * supplied target with availability and capability facts and owns failure when no eligible
 * target remains; it must not silently weaken the requirement.</p>
 *
 * <p>This method-free family expresses no preference, fallback, score, combination, discovery,
 * capability, ownership decision, preparation, or execution behavior.</p>
 */
public sealed interface BackendRequirement
        permits BackendIdRequirement, BackendDeviceIdRequirement, DeviceClassRequirement {}

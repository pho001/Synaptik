package io.github.pho001.synaptik.config.compile;

import io.github.pho001.synaptik.backend.contract.DeviceClass;
import java.util.Optional;

/**
 * Records an optional soft device-class preference for later ownership ranking.
 *
 * <p>Planning applies this immutable input only after hard eligibility has established candidate
 * ownership choices. An empty preference means only that no explicit coarse device-class
 * preference was supplied; it promises no default, fallback, equal candidate scores, discovery,
 * or successful ownership selection. A present preference does not make another eligible
 * candidate ineligible, weaken a hard backend requirement, or guarantee selection of a candidate
 * in the preferred class.</p>
 *
 * <p>This value does not evaluate eligibility or candidates, calculate or compare scores, contain
 * calibrated profile data, select an owner, device, route, or kernel, or perform compilation,
 * preparation, runtime, or execution work. Equality, hashing, and diagnostic text follow ordinary
 * record semantics over the optional preference.</p>
 *
 * @param preferredDeviceClass optional soft device-class preference; must not be {@code null};
 *     the exact optional reference and, when present, exact device-class reference are retained
 */
public record PartitionScoringConfig(Optional<DeviceClass> preferredDeviceClass) {
    /**
     * Creates scoring configuration from an exact caller-supplied optional preference.
     *
     * <p>Construction performs no snapshot, normalization, lookup, matching, scoring, or other
     * evaluation.</p>
     *
     * @param preferredDeviceClass optional soft device-class preference; must not be {@code null};
     *     the exact optional reference and, when present, exact device-class reference are
     *     retained
     * @throws NullPointerException if {@code preferredDeviceClass} is {@code null}; the exception
     *     message is {@code preferredDeviceClass}
     */
    public PartitionScoringConfig(Optional<DeviceClass> preferredDeviceClass) {
        if (preferredDeviceClass == null) {
            throw new NullPointerException("preferredDeviceClass");
        }
        this.preferredDeviceClass = preferredDeviceClass;
    }

    /**
     * Creates configuration with no explicit device-class preference.
     *
     * <p>This factory does not select an aggregate default, owner, fallback, or successful
     * ownership result.</p>
     *
     * @return a new configuration containing an empty optional; never {@code null}
     */
    public static PartitionScoringConfig neutral() {
        return new PartitionScoringConfig(Optional.empty());
    }

    /**
     * Creates configuration that softly prefers one exact device class after hard eligibility.
     *
     * <p>The preference does not filter other eligible candidates, weaken a hard requirement, or
     * guarantee selection.</p>
     *
     * @param deviceClass device class to retain by reference as a soft preference; must not be
     *     {@code null}
     * @return a new configuration containing the exact supplied device-class reference; never
     *     {@code null}
     * @throws NullPointerException if {@code deviceClass} is {@code null}; the exception message
     *     is {@code deviceClass}
     */
    public static PartitionScoringConfig preferring(DeviceClass deviceClass) {
        if (deviceClass == null) {
            throw new NullPointerException("deviceClass");
        }
        return new PartitionScoringConfig(Optional.of(deviceClass));
    }

    /**
     * Returns the optional soft device-class preference retained for later ownership ranking.
     *
     * @return the exact non-null optional reference supplied at construction, containing the
     *     exact supplied device-class reference when present
     */
    public Optional<DeviceClass> preferredDeviceClass() {
        return preferredDeviceClass;
    }
}

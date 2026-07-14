package io.github.pho001.synaptik.planning.capability;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Selects one backend owner from an existing hard-eligible identity list.
 *
 * <p>This package-private stateless operation treats the identities retained by
 * {@link BackendEligibility} as the complete candidate set. It applies only the optional coarse
 * device-class preference and provider-order baseline; it does not re-evaluate eligibility or
 * select a device, route, kernel, executable, or partition.</p>
 */
final class BackendOwnerSelection {
    /**
     * Prevents construction because this class retains no selection state.
     */
    private BackendOwnerSelection() {}

    /**
     * Selects one backend owner from the supplied hard-eligible identities.
     *
     * <p>The top-level inputs are validated in parameter order. An empty eligible list then fails
     * terminally before any availability-snapshot element is read. Otherwise, the complete
     * snapshot list is scanned in caller order before selection: snapshots must be non-null and
     * unique by equal backend identity, and every eligible identity must have one equal associated
     * snapshot. Extra unique snapshots are allowed and never become candidates.</p>
     *
     * <p>With no preferred class, the first eligible identity wins. With a preferred class, the
     * first eligible identity whose associated snapshot reports at least one device of that class
     * wins; if none matches, the first eligible identity wins. Provider encounter order therefore
     * resolves every tie. A matching snapshot with no devices is a preference nonmatch, not a
     * renewed eligibility filter. Selection never removes a nonpreferred eligible backend,
     * restores an ineligible backend, or selects or retains a device, route, or kernel.</p>
     *
     * @param eligibility the non-null hard-eligibility result whose ordered identity list is the
     *     complete candidate set; inspected but not mutated or retained
     * @param scoringConfig the non-null immutable optional coarse-class preference; inspected but
     *     not mutated or retained
     * @param availabilitySnapshots the non-null caller-ordered snapshot list; every element must
     *     be non-null and have a unique backend identity, equal identities associate snapshots
     *     with eligible candidates, extra unique snapshots are allowed, and the list is inspected
     *     but not mutated or retained
     * @return the exact non-null {@link BackendId} reference from the eligible identity list
     * @throws NullPointerException if {@code eligibility} is null, with message
     *     {@code eligibility}; if {@code scoringConfig} is null, with message
     *     {@code scoringConfig}; if {@code availabilitySnapshots} is null, with message
     *     {@code availabilitySnapshots}; or if its first null element is at index {@code i}, with
     *     message {@code availabilitySnapshots[i]}
     * @throws IllegalStateException if no hard-eligible backend exists, with message
     *     {@code no hard-eligible backend is available for ownership selection}
     * @throws IllegalArgumentException if a second snapshot with an equal backend identity is
     *     encountered, with message
     *     {@code duplicate availability snapshot backendId: <backendId.value()>}; or if the first
     *     eligible identity without an equal snapshot is encountered, with message
     *     {@code missing availability snapshot for backendId: <backendId.value()>}
     */
    static BackendId select(
            BackendEligibility eligibility,
            PartitionScoringConfig scoringConfig,
            List<BackendAvailabilitySnapshot> availabilitySnapshots) {
        Objects.requireNonNull(eligibility, "eligibility");
        Objects.requireNonNull(scoringConfig, "scoringConfig");
        Objects.requireNonNull(availabilitySnapshots, "availabilitySnapshots");

        List<BackendId> eligibleBackendIds = eligibility.eligibleBackendIds();
        if (eligibleBackendIds.isEmpty()) {
            throw new IllegalStateException(
                    "no hard-eligible backend is available for ownership selection");
        }

        Map<BackendId, BackendAvailabilitySnapshot> snapshotsByBackendId =
                new HashMap<>(availabilitySnapshots.size());
        for (int index = 0; index < availabilitySnapshots.size(); index++) {
            BackendAvailabilitySnapshot snapshot =
                    Objects.requireNonNull(
                            availabilitySnapshots.get(index),
                            "availabilitySnapshots[" + index + "]");
            BackendId backendId = snapshot.backendId();
            if (snapshotsByBackendId.putIfAbsent(backendId, snapshot) != null) {
                throw new IllegalArgumentException(
                        "duplicate availability snapshot backendId: " + backendId.value());
            }
        }

        for (BackendId backendId : eligibleBackendIds) {
            if (!snapshotsByBackendId.containsKey(backendId)) {
                throw new IllegalArgumentException(
                        "missing availability snapshot for backendId: " + backendId.value());
            }
        }

        DeviceClass preferredDeviceClass = scoringConfig.preferredDeviceClass().orElse(null);
        if (preferredDeviceClass != null) {
            for (BackendId backendId : eligibleBackendIds) {
                if (snapshotsByBackendId
                        .get(backendId)
                        .devices()
                        .containsValue(preferredDeviceClass)) {
                    return backendId;
                }
            }
        }
        return eligibleBackendIds.getFirst();
    }
}

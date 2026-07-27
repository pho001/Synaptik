package io.github.pho001.synaptik.planning.capability;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;
import java.util.List;
import java.util.Objects;

/**
 * Selects one backend owner for one structurally valid operation occurrence.
 *
 * <p>This public stateless collaboration is the narrow per-occurrence Planning seam used by the
 * package-private compiler orchestrator. It composes the package-private hard-eligibility and
 * baseline owner-selection operations without exposing their intermediate result. It preserves
 * provider order, exact backend-identity references, hard requirements, current availability,
 * and the optional coarse device-class preference.</p>
 *
 * <p>The collaboration does not assemble graph-wide ownership, retain a provider or availability
 * snapshot, select a device, calculate a numeric cost, or choose a route, kernel, executable, or
 * runtime state. It is not a public graph compiler, planner workflow, registry, discovery
 * mechanism, or service object.</p>
 */
public final class BackendOwnerPlanning {
    private BackendOwnerPlanning() {}

    /**
     * Selects the backend owner for one operation occurrence.
     *
     * <p>Top-level references are validated in declaration order. The complete hard-eligibility
     * operation then runs exactly once, followed by exactly one baseline selection. Provider,
     * snapshot, validation, association, filtering, preference, and failure semantics are those
     * of the composed operations.</p>
     *
     * @param query non-null immutable operation occurrence passed by exact reference to eligible
     *     capability providers
     * @param intent non-null immutable optional hard backend requirement
     * @param providers non-null ordered capability providers; inspected but not retained
     * @param availabilitySnapshots non-null point-in-time backend availability snapshots;
     *     inspected but not retained
     * @param scoringConfig non-null immutable optional coarse device-class preference
     * @return the exact non-null backend identity reference selected from the eligible
     *     provider-order candidates
     * @throws NullPointerException if a top-level reference or a required nested value is
     *     {@code null}, in the validation order documented by the composed operations
     * @throws IllegalArgumentException if provider and snapshot composition is invalid
     * @throws IllegalStateException if no hard-eligible backend is available; the message is
     *     {@code no hard-eligible backend is available for ownership selection}, and the
     *     package-private selection failure is retained as the cause
     * @throws RuntimeException if a queried provider throws; the same failure propagates
     *     unchanged
     */
    public static BackendId selectOwner(
            OperationCapabilityQuery query,
            BackendIntent intent,
            List<BackendCapabilityProvider> providers,
            List<BackendAvailabilitySnapshot> availabilitySnapshots,
            PartitionScoringConfig scoringConfig) {
        Objects.requireNonNull(query, "query");
        Objects.requireNonNull(intent, "intent");
        Objects.requireNonNull(providers, "providers");
        Objects.requireNonNull(availabilitySnapshots, "availabilitySnapshots");
        Objects.requireNonNull(scoringConfig, "scoringConfig");

        BackendEligibility eligibility =
                BackendEligibility.evaluate(query, intent, providers, availabilitySnapshots);
        try {
            return BackendOwnerSelection.select(
                    eligibility, scoringConfig, availabilitySnapshots);
        } catch (IllegalStateException failure) {
            throw new IllegalStateException(failure.getMessage(), failure);
        }
    }
}

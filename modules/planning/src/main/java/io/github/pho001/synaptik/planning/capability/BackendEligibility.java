package io.github.pho001.synaptik.planning.capability;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.contract.BackendDeviceIdRequirement;
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.contract.BackendIdRequirement;
import io.github.pho001.synaptik.backend.contract.BackendRequirement;
import io.github.pho001.synaptik.backend.contract.DeviceClassRequirement;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Records the hard-eligible backend identities for one operation-capability query.
 *
 * <p>The eligible identities are the exact identities returned by capability providers, in
 * provider encounter order. This package-private value retains no provider, availability
 * snapshot, requirement, device, score, owner choice, route, executable, or runtime state.</p>
 *
 * <p>Equality, hashing, and diagnostic text follow ordinary record semantics over the query and
 * ordered immutable identity snapshot.</p>
 *
 * @param query the non-null operation occurrence; retained by exact reference and returned from
 *     the public component accessor on this package-private record
 * @param eligibleBackendIds the non-null ordered eligible identities; every element must be
 *     non-null and unique by equality, list membership is copied, and identity references are
 *     retained; the public component accessor returns that immutable membership snapshot
 */
record BackendEligibility(
        OperationCapabilityQuery query, List<BackendId> eligibleBackendIds) {
    /**
     * Creates one immutable per-query hard-eligibility result.
     *
     * <p>Validation checks {@code query}, then {@code eligibleBackendIds}, then scans identity
     * elements in encounter order for nulls and duplicate equal values. Membership is snapshotted
     * only after that scan succeeds.</p>
     *
     * @param query the non-null operation occurrence; retained by exact reference
     * @param eligibleBackendIds the non-null ordered eligible identities; every element must be
     *     non-null and unique by equality, list membership is copied, and identity references are
     *     retained
     * @throws NullPointerException if {@code query} is null, with message {@code query}; if
     *     {@code eligibleBackendIds} is null, with message {@code eligibleBackendIds}; or if its
     *     first null element is at index {@code i}, with message
     *     {@code eligibleBackendIds[i]}
     * @throws IllegalArgumentException if a second equal identity is encountered, with message
     *     {@code duplicate eligible backendId: <backendId.value()>}
     */
    BackendEligibility {
        query = Objects.requireNonNull(query, "query");
        Objects.requireNonNull(eligibleBackendIds, "eligibleBackendIds");

        Set<BackendId> seenBackendIds = new HashSet<>();
        for (int index = 0; index < eligibleBackendIds.size(); index++) {
            BackendId backendId =
                    Objects.requireNonNull(
                            eligibleBackendIds.get(index), "eligibleBackendIds[" + index + "]");
            if (!seenBackendIds.add(backendId)) {
                throw new IllegalArgumentException(
                        "duplicate eligible backendId: " + backendId.value());
            }
        }
        eligibleBackendIds = List.copyOf(eligibleBackendIds);
    }

    /**
     * Evaluates backend-level support, current availability, and one optional hard requirement.
     *
     * <p>Top-level references are validated in parameter order. Providers are then scanned
     * completely in encounter order, calling each {@link BackendCapabilityProvider#backendId()}
     * exactly once, before availability snapshots are scanned. Duplicate and missing equal
     * backend identities are rejected before any capability call. Provider identities and
     * snapshot identities match by {@link BackendId#equals(Object)}, so snapshot order does not
     * affect capability-call or result order.</p>
     *
     * <p>After complete composition validation, a provider is skipped without a capability call
     * when its matching snapshot is empty or when it does not satisfy the hard requirement. An
     * exact-backend requirement matches an equal provider identity. An exact-device requirement
     * also requires an equal device key in that backend's snapshot. A device-class requirement
     * requires that class among the snapshot values. Device and class matches prove current
     * matching availability only; capability remains backend-level and no device is selected or
     * retained. Every provider still able to qualify receives the exact query once, in provider
     * order.</p>
     *
     * <p>A queried provider's runtime failure propagates unchanged and prevents later capability
     * calls. A valid no-match evaluation, including two empty input lists, returns an immutable
     * empty identity list; it does not weaken a hard requirement or invent a fallback.</p>
     *
     * @param query the non-null immutable operation occurrence passed by exact reference to each
     *     provider that survives availability and requirement filtering; retained by the result
     * @param intent the non-null immutable optional hard target; inspected but not retained
     * @param providers the non-null ordered capability providers; every provider and its
     *     exactly-once returned backend identity must be non-null, and returned identities must
     *     be unique by equality; inspected but not retained by the result
     * @param availabilitySnapshots the non-null caller-supplied point-in-time snapshots; every
     *     snapshot must be non-null, snapshot identities must be unique by equality, and their
     *     identity set must exactly match the provider identity set; inspected but not retained
     * @return a new non-null result retaining the exact query and an immutable provider-order
     *     snapshot of the exact eligible provider identity references; the list may be empty
     * @throws NullPointerException if a top-level argument is null, with its parameter name as
     *     the message; if provider {@code i} is null, with message {@code providers[i]}; if its
     *     returned identity is null, with message {@code providers[i].backendId()}; or if
     *     availability snapshot {@code i} is null, with message {@code availabilitySnapshots[i]}
     * @throws IllegalArgumentException if a second equal provider identity is encountered, with
     *     message {@code duplicate provider backendId: <backendId.value()>}; if a second equal
     *     snapshot identity is encountered, with message
     *     {@code duplicate availability snapshot backendId: <backendId.value()>}; if the first
     *     provider without a snapshot is encountered, with message
     *     {@code missing availability snapshot for backendId: <backendId.value()>}; or if the
     *     first snapshot without a provider is encountered, with message
     *     {@code missing capability provider for backendId: <backendId.value()>}
     * @throws RuntimeException if a queried provider throws; the same failure propagates unchanged
     */
    static BackendEligibility evaluate(
            OperationCapabilityQuery query,
            BackendIntent intent,
            List<BackendCapabilityProvider> providers,
            List<BackendAvailabilitySnapshot> availabilitySnapshots) {
        Objects.requireNonNull(query, "query");
        Objects.requireNonNull(intent, "intent");
        Objects.requireNonNull(providers, "providers");
        Objects.requireNonNull(availabilitySnapshots, "availabilitySnapshots");

        List<BackendCapabilityProvider> retainedProviders = new ArrayList<>(providers.size());
        List<BackendId> providerBackendIds = new ArrayList<>(providers.size());
        Set<BackendId> seenProviderBackendIds = new HashSet<>();
        for (int index = 0; index < providers.size(); index++) {
            BackendCapabilityProvider provider =
                    Objects.requireNonNull(providers.get(index), "providers[" + index + "]");
            BackendId backendId =
                    Objects.requireNonNull(
                            provider.backendId(), "providers[" + index + "].backendId()");
            retainedProviders.add(provider);
            providerBackendIds.add(backendId);
            if (!seenProviderBackendIds.add(backendId)) {
                throw new IllegalArgumentException(
                        "duplicate provider backendId: " + backendId.value());
            }
        }

        List<BackendId> snapshotBackendIds = new ArrayList<>(availabilitySnapshots.size());
        Map<BackendId, BackendAvailabilitySnapshot> snapshotsByBackendId =
                new HashMap<>(availabilitySnapshots.size());
        for (int index = 0; index < availabilitySnapshots.size(); index++) {
            BackendAvailabilitySnapshot snapshot =
                    Objects.requireNonNull(
                            availabilitySnapshots.get(index),
                            "availabilitySnapshots[" + index + "]");
            BackendId backendId = snapshot.backendId();
            snapshotBackendIds.add(backendId);
            if (snapshotsByBackendId.putIfAbsent(backendId, snapshot) != null) {
                throw new IllegalArgumentException(
                        "duplicate availability snapshot backendId: " + backendId.value());
            }
        }

        for (BackendId backendId : providerBackendIds) {
            if (!snapshotsByBackendId.containsKey(backendId)) {
                throw new IllegalArgumentException(
                        "missing availability snapshot for backendId: " + backendId.value());
            }
        }
        for (BackendId backendId : snapshotBackendIds) {
            if (!seenProviderBackendIds.contains(backendId)) {
                throw new IllegalArgumentException(
                        "missing capability provider for backendId: " + backendId.value());
            }
        }

        BackendRequirement hardRequirement = intent.hardRequirement().orElse(null);
        List<BackendId> eligibleBackendIds = new ArrayList<>();
        for (int index = 0; index < retainedProviders.size(); index++) {
            BackendCapabilityProvider provider = retainedProviders.get(index);
            BackendId backendId = providerBackendIds.get(index);
            BackendAvailabilitySnapshot snapshot = snapshotsByBackendId.get(backendId);

            if (snapshot.devices().isEmpty()) {
                continue;
            }
            boolean satisfiesRequirement =
                    hardRequirement == null
                            || switch (hardRequirement) {
                                case BackendIdRequirement requirement ->
                                        backendId.equals(requirement.backendId());
                                case BackendDeviceIdRequirement requirement ->
                                        backendId.equals(requirement.deviceId().backendId())
                                                && snapshot.devices()
                                                        .containsKey(requirement.deviceId());
                                case DeviceClassRequirement requirement ->
                                        snapshot.devices()
                                                .containsValue(requirement.deviceClass());
                            };
            if (!satisfiesRequirement) {
                continue;
            }
            if (provider.supports(query)) {
                eligibleBackendIds.add(backendId);
            }
        }
        return new BackendEligibility(query, eligibleBackendIds);
    }

    /**
     * Returns the operation occurrence evaluated by this result.
     *
     * <p>Java record component accessors are necessarily public. The enclosing record remains
     * package-private, so this declaration does not expose the result outside this package.</p>
     *
     * @return the exact non-null query reference supplied at construction
     */
    @Override
    public OperationCapabilityQuery query() {
        return query;
    }

    /**
     * Returns the hard-eligible backend identities in provider encounter order.
     *
     * <p>Java record component accessors are necessarily public. The enclosing record remains
     * package-private, so this declaration does not expose the result outside this package.</p>
     *
     * @return the non-null immutable membership snapshot; elements are the exact non-null
     *     provider-returned identity references supplied at construction
     */
    @Override
    public List<BackendId> eligibleBackendIds() {
        return eligibleBackendIds;
    }
}

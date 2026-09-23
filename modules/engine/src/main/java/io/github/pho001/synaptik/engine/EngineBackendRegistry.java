package io.github.pho001.synaptik.engine;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.prepare.GraphPreparation;
import io.github.pho001.synaptik.prepare.PartitionPreparation;
import io.github.pho001.synaptik.runtime.execution.PreparedExecution;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Private ordered construction registry and cold complete-owner-set routing boundary.
 *
 * <p>Construction validates each already-captured entry exactly once and freezes provider and
 * availability lists in registration order. Cold prepare validates the complete owner set and
 * transfer support before invoking any backend analysis. Execution receives only exact
 * occurrence adapters and direct prepared references captured in outward handles; this registry
 * has no run-path lookup role.</p>
 */
final class EngineBackendRegistry implements AutoCloseable {
    private final List<EngineBackendComposition> entries;
    private final List<BackendCapabilityProvider> capabilityProviders;
    private final List<BackendAvailabilitySnapshot> availabilitySnapshots;
    private final Map<BackendId, EngineBackendComposition> entriesById;
    private boolean closed;
    private Throwable closeFailure;
    private volatile RuntimeException lookupFailureForTesting;

    /** Immutable registration fact captured before Builder mutates its sole ordered collection. */
    record Registration(
            BackendId backendId,
            BackendCapabilityProvider provider,
            BackendAvailabilitySnapshot snapshot,
            EngineBackendComposition adapter) {
        Registration {
            Objects.requireNonNull(backendId, "backendId");
            Objects.requireNonNull(provider, "provider");
            Objects.requireNonNull(snapshot, "snapshot");
            Objects.requireNonNull(adapter, "adapter");
            if (!backendId.equals(snapshot.backendId())) {
                throw new IllegalArgumentException(
                        "provider and availability snapshot backend IDs must be equal");
            }
        }
    }

    /**
     * Captures one adapter's provider, availability, and identity exactly once.
     *
     * @param entry non-null adapter being transferred
     * @return one non-null immutable registration
     */
    static Registration capture(EngineBackendComposition entry) {
        Objects.requireNonNull(entry, "entry");
        List<BackendCapabilityProvider> providers = Objects.requireNonNull(
                entry.capabilityProviders(), "capabilityProviders");
        List<BackendAvailabilitySnapshot> availability = Objects.requireNonNull(
                entry.availabilitySnapshots(), "availabilitySnapshots");
        if (providers.size() != 1 || availability.size() != 1) {
            throw new IllegalArgumentException(
                    "backend entry must expose exactly one provider and snapshot");
        }
        BackendCapabilityProvider provider = Objects.requireNonNull(
                providers.getFirst(), "capabilityProvider");
        BackendAvailabilitySnapshot snapshot = Objects.requireNonNull(
                availability.getFirst(), "availabilitySnapshot");
        BackendId backendId = Objects.requireNonNull(provider.backendId(), "backendId");
        return new Registration(backendId, provider, snapshot, entry);
    }

    /**
     * Takes ownership of non-empty entries and captures their facts in list order.
     *
     * @param entries non-null registration-ordered adapters
     */
    EngineBackendRegistry(List<EngineBackendComposition> entries) {
        this(captureAll(entries), true);
    }

    /**
     * Takes ownership of immutable facts already captured by Builder.
     *
     * @param registrations non-null non-empty registration-ordered facts
     * @return a new owning registry
     */
    static EngineBackendRegistry fromCaptured(List<Registration> registrations) {
        return new EngineBackendRegistry(registrations, true);
    }

    private EngineBackendRegistry(List<Registration> registrations, boolean alreadyCaptured) {
        Objects.requireNonNull(registrations, "registrations");
        if (registrations.isEmpty()) {
            throw new IllegalArgumentException("at least one backend integration is required");
        }
        var entrySnapshot =
                new ArrayList<EngineBackendComposition>(registrations.size());
        var providerSnapshot =
                new ArrayList<BackendCapabilityProvider>(registrations.size());
        var availabilitySnapshot =
                new ArrayList<BackendAvailabilitySnapshot>(registrations.size());
        var byId = new LinkedHashMap<BackendId, EngineBackendComposition>();
        for (int index = 0; index < registrations.size(); index++) {
            Registration registration = Objects.requireNonNull(
                    registrations.get(index), "registrations[" + index + "]");
            if (byId.putIfAbsent(registration.backendId(), registration.adapter()) != null) {
                throw new IllegalArgumentException(
                        "duplicate backend ID: " + registration.backendId().value());
            }
            entrySnapshot.add(registration.adapter());
            providerSnapshot.add(registration.provider());
            availabilitySnapshot.add(registration.snapshot());
        }
        this.entries = List.copyOf(entrySnapshot);
        this.capabilityProviders = List.copyOf(providerSnapshot);
        this.availabilitySnapshots = List.copyOf(availabilitySnapshot);
        this.entriesById = Collections.unmodifiableMap(new LinkedHashMap<>(byId));
    }

    private static List<Registration> captureAll(
            List<EngineBackendComposition> entries) {
        Objects.requireNonNull(entries, "entries");
        var registrations = new ArrayList<Registration>(entries.size());
        for (int index = 0; index < entries.size(); index++) {
            registrations.add(capture(Objects.requireNonNull(
                    entries.get(index), "entries[" + index + "]")));
        }
        return registrations;
    }

    /** @return immutable provider references in registration order */
    List<BackendCapabilityProvider> capabilityProviders() {
        return capabilityProviders;
    }

    /** @return immutable registration-time snapshots aligned with the provider list */
    List<BackendAvailabilitySnapshot> availabilitySnapshots() {
        return availabilitySnapshots;
    }

    /**
     * Preflights every owner and cross-owner path, then performs one shared graph preparation.
     *
     * @param artifacts non-null compile artifacts whose complete owner and transfer set is
     *     inspected before any backend analysis callback
     * @return non-null inward recipe plus direct input/publication adapter snapshots
     * @throws IllegalArgumentException if the plan is empty, an exact owner is unregistered, or a
     *     cross-owner descriptor/path is unsupported
     * @throws IllegalStateException if registry closure has begun
     */
    EnginePreparation prepare(CompileArtifacts artifacts) {
        Objects.requireNonNull(artifacts, "artifacts");
        synchronized (this) {
            if (closed) {
                throw new IllegalStateException("backend registry is closed");
            }
        }
        if (artifacts.partitions().isEmpty()) {
            throw new IllegalArgumentException("preparation requires a non-empty partition plan");
        }

        for (int index = 0; index < artifacts.partitions().size(); index++) {
            BackendId owner = Objects.requireNonNull(
                    artifacts.partitions().get(index).owner(), "partition owner");
            requireAdapter(owner);
        }
        preflightTransfers(artifacts.memory().requirements());

        var preparations = new ArrayList<PartitionPreparation<?, ?>>(
                artifacts.partitions().size());
        for (int index = 0; index < artifacts.partitions().size(); index++) {
            preparations.add(requireAdapter(
                    artifacts.partitions().get(index).owner()).partitionPreparation());
        }
        var orderedAdapters = new LinkedHashMap<BackendId, EngineBackendComposition>();
        entriesById.forEach(orderedAdapters::put);
        PreparedExecution execution = GraphPreparation.prepare(
                artifacts,
                preparations,
                new CompositePreparedScheduleAssembler(orderedAdapters));
        try {
            Map<ValueId, LogicalMemoryRequirement> requirements = requirementMap(artifacts);
            List<EngineBackendComposition> inputAdapters =
                    adaptersForValues(artifacts.constants().bindableInputs(), requirements);
            var publicationValues = new ArrayList<ValueId>();
            artifacts.publication().forwardBindings().forEach(
                    binding -> publicationValues.add(binding.valueId()));
            artifacts.publication().gradientBindings().forEach(
                    binding -> publicationValues.add(binding.valueId()));
            List<EngineBackendComposition> publicationAdapters =
                    adaptersForValues(publicationValues, requirements);
            return new EnginePreparation(
                    execution, inputAdapters, publicationAdapters);
        } catch (RuntimeException | Error failure) {
            try {
                execution.close();
            } catch (RuntimeException | Error cleanupFailure) {
                if (cleanupFailure != failure) {
                    failure.addSuppressed(cleanupFailure);
                }
            }
            throw failure;
        }
    }

    private EngineBackendComposition requireAdapter(BackendId owner) {
        RuntimeException lookupFailure = lookupFailureForTesting;
        if (lookupFailure != null) {
            throw lookupFailure;
        }
        EngineBackendComposition adapter = entriesById.get(owner);
        if (adapter == null) {
            throw new IllegalArgumentException(
                    "no registered backend owns plan: " + owner.value());
        }
        return adapter;
    }

    private void preflightTransfers(List<LogicalMemoryRequirement> requirements) {
        for (LogicalMemoryRequirement requirement : requirements) {
            List<BackendId> owners =
                    CompositePreparedScheduleAssembler.ownerOrder(requirement);
            for (BackendId owner : owners) {
                requireAdapter(owner);
            }
            if (owners.size() < 2) {
                continue;
            }
            EngineBackendComposition source = requireAdapter(owners.getFirst());
            for (int destinationIndex = 1;
                    destinationIndex < owners.size();
                    destinationIndex++) {
                EngineBackendComposition destination =
                        requireAdapter(owners.get(destinationIndex));
                if (!source.supportsTransferTo(destination, requirement.descriptor())) {
                    throw new IllegalArgumentException(
                            "unsupported cross-owner transfer from "
                                    + owners.getFirst().value()
                                    + " to "
                                    + owners.get(destinationIndex).value()
                                    + " for "
                                    + requirement.valueId());
                }
            }
        }
    }

    private static Map<ValueId, LogicalMemoryRequirement> requirementMap(
            CompileArtifacts artifacts) {
        var requirements = new HashMap<ValueId, LogicalMemoryRequirement>();
        for (LogicalMemoryRequirement requirement : artifacts.memory().requirements()) {
            requirements.put(requirement.valueId(), requirement);
        }
        return requirements;
    }

    private List<EngineBackendComposition> adaptersForValues(
            List<ValueId> valueIds,
            Map<ValueId, LogicalMemoryRequirement> requirements) {
        var result = new ArrayList<EngineBackendComposition>(valueIds.size());
        for (int index = 0; index < valueIds.size(); index++) {
            ValueId valueId = valueIds.get(index);
            LogicalMemoryRequirement requirement = requirements.get(valueId);
            if (requirement == null) {
                throw new IllegalArgumentException(
                        "external value has no logical memory requirement: " + valueId);
            }
            List<BackendId> owners =
                    CompositePreparedScheduleAssembler.ownerOrder(requirement);
            if (owners.isEmpty()) {
                result.add(producerlessConstantAdapter());
            } else {
                result.add(requireAdapter(owners.getFirst()));
            }
        }
        return List.copyOf(result);
    }

    private EngineBackendComposition producerlessConstantAdapter() {
        for (EngineBackendComposition entry : entries) {
            if (entry instanceof CpuEngineBackendComposition) {
                return entry;
            }
        }
        throw new IllegalArgumentException(
                "producerless published constants require a registered CPU adapter");
    }

    /**
     * Selects the exact adapter for one complete plan without invoking backend preparation.
     *
     * @param artifacts non-null immutable compile artifacts
     * @return the exact registered adapter selected by equal {@link BackendId}
     * @throws IllegalArgumentException if the plan is empty, mixed-owner, or unregistered
     * @throws IllegalStateException if registry closure has begun
     */
    EngineBackendComposition selectedAdapter(CompileArtifacts artifacts) {
        Objects.requireNonNull(artifacts, "artifacts");
        synchronized (this) {
            if (closed) throw new IllegalStateException("backend registry is closed");
        }
        if (artifacts.partitions().isEmpty()) {
            throw new IllegalArgumentException("preparation requires a non-empty partition plan");
        }
        BackendId owner = Objects.requireNonNull(
                artifacts.partitions().getFirst().owner(), "partition owner");
        for (int index = 1; index < artifacts.partitions().size(); index++) {
            BackendId candidate = Objects.requireNonNull(
                    artifacts.partitions().get(index).owner(), "partition owner");
            if (!owner.equals(candidate)) {
                throw new IllegalArgumentException(
                        "preparation requires exactly one backend owner");
            }
        }
        EngineBackendComposition adapter = entriesById.get(owner);
        if (adapter == null) {
            throw new IllegalArgumentException(
                    "no registered backend owns plan: " + owner.value());
        }
        return adapter;
    }

    /**
     * Makes every later owner lookup fail for focused proof that prepared run/materialization uses
     * only captured direct references.
     */
    void poisonLookupForTesting() {
        lookupFailureForTesting =
                new IllegalStateException("backend registry lookup poisoned after preparation");
    }

    /**
     * Returns the sole entry for the CPU-only advanced representation surface.
     *
     * @return the exact sole registered adapter
     * @throws IllegalStateException if more than one adapter is registered
     */
    EngineBackendComposition soleAdapter() {
        if (entries.size() != 1) {
            throw new IllegalStateException(
                    "advanced representation operations require one backend integration");
        }
        return entries.getFirst();
    }

    /**
     * Closes every entry in reverse registration order with attempt-all failure retention.
     * Repeated calls replay the exact first failure.
     *
     * @throws RuntimeException if cleanup first reports an unchecked failure
     * @throws Error if cleanup first reports a fatal failure
     */
    @Override
    public synchronized void close() {
        if (!closed) {
            closed = true;
            Throwable first = null;
            for (int index = entries.size() - 1; index >= 0; index--) {
                try {
                    entries.get(index).close();
                } catch (RuntimeException | Error failure) {
                    if (first == null) first = failure;
                    else if (failure != first) first.addSuppressed(failure);
                }
            }
            closeFailure = first;
        }
        rethrow(closeFailure);
    }

    private static void rethrow(Throwable failure) {
        if (failure instanceof RuntimeException runtime) throw runtime;
        if (failure instanceof Error error) throw error;
    }
}

package planning.memory;

import planning.partition.execution.MaterializationDecision;
import planning.value.GraphValueRef;
import tensor.DataType;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class PartitionBindingAllocator {
    private PartitionBindingAllocator() {
    }

    static PartitionBindingAssignment allocate(List<PartitionValueLifetime> lifetimes) {
        if (lifetimes.isEmpty()) {
            return new PartitionBindingAssignment(Map.of(), Map.of(), Map.of());
        }
        List<PartitionValueLifetime> allocatable = lifetimes.stream()
                .filter(lifetime -> lifetime.decision() != MaterializationDecision.VIRTUALIZE)
                .sorted(Comparator.comparingInt(PartitionValueLifetime::birthStep).thenComparingInt(PartitionValueLifetime::lastUseStep))
                .toList();
        ArrayList<PartitionBindingState> active = new ArrayList<>();
        ArrayList<PartitionBindingState> free = new ArrayList<>();
        LinkedHashMap<GraphValueRef, PartitionMemoryBinding> bindings = new LinkedHashMap<>();
        LinkedHashMap<GraphValueRef, Integer> slotByValueRef = new LinkedHashMap<>();
        LinkedHashMap<Integer, Integer> slotSizes = new LinkedHashMap<>();
        int nextBindingId = 0;
        for (PartitionValueLifetime lifetime : allocatable) {
            releaseExpiredPartitionBindings(active, free, lifetime.birthStep());
            PartitionBindingState chosen = choosePartitionBinding(free, lifetime);
            if (chosen == null) {
                chosen = new PartitionBindingState(
                        nextBindingId++,
                        bindingKindFor(lifetime.decision()),
                        storageTypeFor(lifetime),
                        transportTypeFor(lifetime),
                        slotSizeElementsFor(lifetime),
                        Integer.MIN_VALUE
                );
                slotSizes.put(chosen.bindingId, slotSizeElementsFor(lifetime));
            } else {
                free.remove(chosen);
            }
            chosen.lastUseStep = lifetime.lastUseStep();
            active.add(chosen);
            bindings.put(lifetime.valueRef(), new PartitionMemoryBinding(
                    lifetime.valueRef(),
                    chosen.kind,
                    chosen.bindingId,
                    chosen.storageType,
                    chosen.transportType,
                    true
            ));
            slotByValueRef.put(lifetime.valueRef(), chosen.bindingId);
            slotSizes.merge(chosen.bindingId, slotSizeElementsFor(lifetime), Math::max);
        }
        for (PartitionValueLifetime lifetime : lifetimes) {
            if (bindings.containsKey(lifetime.valueRef())) {
                continue;
            }
            bindings.put(lifetime.valueRef(), new PartitionMemoryBinding(
                    lifetime.valueRef(),
                    PartitionMemoryBindingKind.NONE,
                    null,
                    storageTypeFor(lifetime),
                    transportTypeFor(lifetime),
                    false
            ));
        }
        return new PartitionBindingAssignment(Map.copyOf(bindings), Map.copyOf(slotByValueRef), Map.copyOf(slotSizes));
    }

    private static void releaseExpiredPartitionBindings(
            List<PartitionBindingState> active,
            List<PartitionBindingState> free,
            int currentBirthStep
    ) {
        List<PartitionBindingState> released = new ArrayList<>();
        for (PartitionBindingState state : active) {
            if (state.lastUseStep < currentBirthStep) {
                released.add(state);
            }
        }
        active.removeAll(released);
        free.addAll(released);
    }

    private static PartitionBindingState choosePartitionBinding(List<PartitionBindingState> free, PartitionValueLifetime lifetime) {
        PartitionMemoryBindingKind kind = bindingKindFor(lifetime.decision());
        DataType storageType = storageTypeFor(lifetime);
        DataType transportType = transportTypeFor(lifetime);
        int slotSize = slotSizeElementsFor(lifetime);
        for (PartitionBindingState state : free) {
            if (state.kind == kind
                    && state.storageType == storageType
                    && state.transportType == transportType
                    && state.size == slotSize) {
                return state;
            }
        }
        return null;
    }

    private static PartitionMemoryBindingKind bindingKindFor(MaterializationDecision decision) {
        return switch (decision) {
            case CONTINUE -> PartitionMemoryBindingKind.CONTINUATION;
            case MATERIALIZE -> PartitionMemoryBindingKind.MATERIALIZED;
            case VIRTUALIZE -> PartitionMemoryBindingKind.NONE;
        };
    }

    private static DataType storageTypeFor(PartitionValueLifetime lifetime) {
        return lifetime.typeContract().storageType();
    }

    private static DataType transportTypeFor(PartitionValueLifetime lifetime) {
        return switch (lifetime.decision()) {
            case CONTINUE -> lifetime.typeContract().transportType();
            case MATERIALIZE, VIRTUALIZE -> lifetime.typeContract().storageType();
        };
    }

    private static int slotSizeElementsFor(PartitionValueLifetime lifetime) {
        return Math.max(0, lifetime.elementCount());
    }

    private static final class PartitionBindingState {
        private final int bindingId;
        private final PartitionMemoryBindingKind kind;
        private final DataType storageType;
        private final DataType transportType;
        private final int size;
        private int lastUseStep;

        private PartitionBindingState(
                int bindingId,
                PartitionMemoryBindingKind kind,
                DataType storageType,
                DataType transportType,
                int size,
                int lastUseStep
        ) {
            this.bindingId = bindingId;
            this.kind = kind;
            this.storageType = storageType;
            this.transportType = transportType;
            this.size = size;
            this.lastUseStep = lastUseStep;
        }
    }
}

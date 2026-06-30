package planning.memory;

import planning.region.MaterializationDecision;
import planning.value.GraphValueRef;
import tensor.DataType;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class RegionBindingAllocator {
    private RegionBindingAllocator() {
    }

    static RegionBindingAssignment allocate(List<RegionValueLifetime> lifetimes) {
        if (lifetimes.isEmpty()) {
            return new RegionBindingAssignment(Map.of(), Map.of(), Map.of());
        }
        List<RegionValueLifetime> allocatable = lifetimes.stream()
                .filter(lifetime -> lifetime.decision() != MaterializationDecision.VIRTUALIZE)
                .sorted(Comparator.comparingInt(RegionValueLifetime::birthStep).thenComparingInt(RegionValueLifetime::lastUseStep))
                .toList();
        ArrayList<RegionBindingState> active = new ArrayList<>();
        ArrayList<RegionBindingState> free = new ArrayList<>();
        LinkedHashMap<GraphValueRef, RegionMemoryBinding> bindings = new LinkedHashMap<>();
        LinkedHashMap<GraphValueRef, Integer> slotByValueRef = new LinkedHashMap<>();
        LinkedHashMap<Integer, Integer> slotSizes = new LinkedHashMap<>();
        int nextBindingId = 0;
        for (RegionValueLifetime lifetime : allocatable) {
            releaseExpiredRegionBindings(active, free, lifetime.birthStep());
            RegionBindingState chosen = chooseRegionBinding(free, lifetime);
            if (chosen == null) {
                chosen = new RegionBindingState(
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
            bindings.put(lifetime.valueRef(), new RegionMemoryBinding(
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
        for (RegionValueLifetime lifetime : lifetimes) {
            if (bindings.containsKey(lifetime.valueRef())) {
                continue;
            }
            bindings.put(lifetime.valueRef(), new RegionMemoryBinding(
                    lifetime.valueRef(),
                    RegionMemoryBindingKind.NONE,
                    null,
                    storageTypeFor(lifetime),
                    transportTypeFor(lifetime),
                    false
            ));
        }
        return new RegionBindingAssignment(Map.copyOf(bindings), Map.copyOf(slotByValueRef), Map.copyOf(slotSizes));
    }

    private static void releaseExpiredRegionBindings(
            List<RegionBindingState> active,
            List<RegionBindingState> free,
            int currentBirthStep
    ) {
        List<RegionBindingState> released = new ArrayList<>();
        for (RegionBindingState state : active) {
            if (state.lastUseStep < currentBirthStep) {
                released.add(state);
            }
        }
        active.removeAll(released);
        free.addAll(released);
    }

    private static RegionBindingState chooseRegionBinding(List<RegionBindingState> free, RegionValueLifetime lifetime) {
        RegionMemoryBindingKind kind = bindingKindFor(lifetime.decision());
        DataType storageType = storageTypeFor(lifetime);
        DataType transportType = transportTypeFor(lifetime);
        int slotSize = slotSizeElementsFor(lifetime);
        for (RegionBindingState state : free) {
            if (state.kind == kind
                    && state.storageType == storageType
                    && state.transportType == transportType
                    && state.size == slotSize) {
                return state;
            }
        }
        return null;
    }

    private static RegionMemoryBindingKind bindingKindFor(MaterializationDecision decision) {
        return switch (decision) {
            case CONTINUE -> RegionMemoryBindingKind.CONTINUATION;
            case MATERIALIZE -> RegionMemoryBindingKind.MATERIALIZED;
            case VIRTUALIZE -> RegionMemoryBindingKind.NONE;
        };
    }

    private static DataType storageTypeFor(RegionValueLifetime lifetime) {
        return lifetime.typeContract().storageType();
    }

    private static DataType transportTypeFor(RegionValueLifetime lifetime) {
        return switch (lifetime.decision()) {
            case CONTINUE -> lifetime.typeContract().transportType();
            case MATERIALIZE, VIRTUALIZE -> lifetime.typeContract().storageType();
        };
    }

    private static int slotSizeElementsFor(RegionValueLifetime lifetime) {
        return Math.max(0, lifetime.elementCount());
    }

    private static final class RegionBindingState {
        private final int bindingId;
        private final RegionMemoryBindingKind kind;
        private final DataType storageType;
        private final DataType transportType;
        private final int size;
        private int lastUseStep;

        private RegionBindingState(
                int bindingId,
                RegionMemoryBindingKind kind,
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

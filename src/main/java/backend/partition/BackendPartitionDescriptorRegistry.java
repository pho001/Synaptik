package backend.partition;

import backend.metal.lowering.MetalBackendPartitionCapability;
import backend.metal.lowering.MetalRegionLowerer;
import backend.cpu.lowering.CpuRegionLowerer;
import backend.cpu.partition.CpuBackendPartitionCapability;
import backend.cuda.lowering.CudaGpuBackendPartitionCapability;
import backend.cuda.lowering.CudaRegionLowerer;
import backend.lowering.RegionLowerer;
import graph.compile.planning.partition.PartitionTarget;
import graph.compile.planning.partition.BackendPartitionCapability;
import graph.compile.planning.partition.UnsupportedBackendPartitionCapability;

import java.util.List;
import java.util.function.Supplier;

public final class BackendPartitionDescriptorRegistry {
    private static final BackendPartitionDescriptorRegistry DEFAULTS = new BackendPartitionDescriptorRegistry(List.of(
            descriptor(PartitionTarget.CPU, CpuBackendPartitionCapability::new, List.of(CpuRegionLowerer::new)),
            descriptor(PartitionTarget.GPU_METAL, MetalBackendPartitionCapability::new, List.of(MetalRegionLowerer::new)),
            descriptor(PartitionTarget.GPU_CUDA, CudaGpuBackendPartitionCapability::new, List.of(CudaRegionLowerer::new))
    ));

    private final List<BackendPartitionDescriptor> descriptors;

    public BackendPartitionDescriptorRegistry(List<BackendPartitionDescriptor> descriptors) {
        this.descriptors = List.copyOf(descriptors == null ? List.of() : descriptors);
    }

    public static BackendPartitionDescriptorRegistry defaults() {
        return DEFAULTS;
    }

    public BackendPartitionDescriptor descriptorFor(PartitionTarget target) {
        PartitionTarget resolved = target == null ? PartitionTarget.NONE : target;
        for (BackendPartitionDescriptor descriptor : descriptors) {
            if (descriptor.target() == resolved) {
                return descriptor;
            }
        }
        return descriptor(resolved, () -> new UnsupportedBackendPartitionCapability(resolved), List.of());
    }

    public BackendPartitionCapability partitionCapabilityFor(PartitionTarget target) {
        return descriptorFor(target).partitionCapability();
    }

    public List<RegionLowerer> lowerers() {
        return descriptors.stream()
                .flatMap(descriptor -> descriptor.lowerers().stream())
                .toList();
    }

    private static BackendPartitionDescriptor descriptor(
            PartitionTarget target,
            Supplier<BackendPartitionCapability> partitionCapability,
            List<Supplier<RegionLowerer>> lowerers
    ) {
        return new DefaultBackendPartitionDescriptor(target, partitionCapability, lowerers);
    }

    private static final class DefaultBackendPartitionDescriptor implements BackendPartitionDescriptor {
        private final PartitionTarget target;
        private final Supplier<BackendPartitionCapability> partitionCapabilitySupplier;
        private final List<Supplier<RegionLowerer>> lowererSuppliers;

        private DefaultBackendPartitionDescriptor(
                PartitionTarget target,
                Supplier<BackendPartitionCapability> partitionCapability,
                List<Supplier<RegionLowerer>> lowerers
        ) {
            this.target = target == null ? PartitionTarget.NONE : target;
            this.partitionCapabilitySupplier = partitionCapability == null
                    ? () -> new UnsupportedBackendPartitionCapability(this.target)
                    : partitionCapability;
            this.lowererSuppliers = List.copyOf(lowerers == null ? List.of() : lowerers);
        }

        @Override
        public PartitionTarget target() {
            return target;
        }

        @Override
        public BackendPartitionCapability partitionCapability() {
            return partitionCapabilitySupplier.get();
        }

        @Override
        public List<RegionLowerer> lowerers() {
            return lowererSuppliers.stream()
                    .map(Supplier::get)
                    .toList();
        }
    }
}

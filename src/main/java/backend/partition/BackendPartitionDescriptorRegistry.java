package backend.partition;

import backend.metal.lowering.MetalBackendPartitionCapability;
import backend.metal.lowering.MetalPartitionLowerer;
import backend.cpu.lowering.CpuPartitionLowerer;
import backend.cpu.partition.CpuBackendPartitionCapability;
import backend.cuda.lowering.CudaGpuBackendPartitionCapability;
import backend.cuda.lowering.CudaPartitionLowerer;
import backend.lowering.PartitionLowerer;
import planning.partition.BackendPartitionCapability;
import planning.backend.BackendPartitionCapabilityRegistry;
import planning.partition.PartitionTarget;
import planning.partition.UnsupportedBackendPartitionCapability;

import java.util.List;
import java.util.function.Supplier;

public final class BackendPartitionDescriptorRegistry implements BackendPartitionCapabilityRegistry {
    private static final BackendPartitionDescriptorRegistry DEFAULTS = new BackendPartitionDescriptorRegistry(List.of(
            descriptor(PartitionTarget.CPU, CpuBackendPartitionCapability::new, List.of(CpuPartitionLowerer::new)),
            descriptor(PartitionTarget.GPU_METAL, MetalBackendPartitionCapability::new, List.of(MetalPartitionLowerer::new)),
            descriptor(PartitionTarget.GPU_CUDA, CudaGpuBackendPartitionCapability::new, List.of(CudaPartitionLowerer::new))
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

    @Override
    public BackendPartitionCapability partitionCapabilityFor(PartitionTarget target) {
        return descriptorFor(target).partitionCapability();
    }

    public List<PartitionLowerer> lowerers() {
        return descriptors.stream()
                .flatMap(descriptor -> descriptor.lowerers().stream())
                .toList();
    }

    private static BackendPartitionDescriptor descriptor(
            PartitionTarget target,
            Supplier<BackendPartitionCapability> partitionCapability,
            List<Supplier<PartitionLowerer>> lowerers
    ) {
        return new DefaultBackendPartitionDescriptor(target, partitionCapability, lowerers);
    }

    private static final class DefaultBackendPartitionDescriptor implements BackendPartitionDescriptor {
        private final PartitionTarget target;
        private final Supplier<BackendPartitionCapability> partitionCapabilitySupplier;
        private final List<Supplier<PartitionLowerer>> lowererSuppliers;

        private DefaultBackendPartitionDescriptor(
                PartitionTarget target,
                Supplier<BackendPartitionCapability> partitionCapability,
                List<Supplier<PartitionLowerer>> lowerers
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
        public List<PartitionLowerer> lowerers() {
            return lowererSuppliers.stream()
                    .map(Supplier::get)
                    .toList();
        }
    }
}

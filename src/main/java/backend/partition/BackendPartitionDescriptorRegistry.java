package backend.partition;

import backend.metal.lowering.MetalRegionLegalityAdapter;
import backend.metal.lowering.MetalRegionLowerer;
import backend.cpu.lowering.CpuRegionLowerer;
import backend.cpu.partition.CpuRegionLegalityAdapter;
import backend.cuda.lowering.CudaGpuRegionLegalityAdapter;
import backend.cuda.lowering.CudaRegionLowerer;
import backend.lowering.RegionLowerer;
import graph.optimizer.partition.PartitionTarget;
import graph.optimizer.partition.RegionLegalityAdapter;
import graph.optimizer.partition.UnsupportedRegionLegalityAdapter;

import java.util.List;
import java.util.function.Supplier;

public final class BackendPartitionDescriptorRegistry {
    private static final BackendPartitionDescriptorRegistry DEFAULTS = new BackendPartitionDescriptorRegistry(List.of(
            descriptor(PartitionTarget.CPU, CpuRegionLegalityAdapter::new, List.of(CpuRegionLowerer::new)),
            descriptor(PartitionTarget.GPU_METAL, MetalRegionLegalityAdapter::new, List.of(MetalRegionLowerer::new)),
            descriptor(PartitionTarget.GPU_CUDA, CudaGpuRegionLegalityAdapter::new, List.of(CudaRegionLowerer::new))
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
        return descriptor(resolved, () -> new UnsupportedRegionLegalityAdapter(resolved), List.of());
    }

    public RegionLegalityAdapter legalityAdapterFor(PartitionTarget target) {
        return descriptorFor(target).legalityAdapter();
    }

    public List<RegionLowerer> lowerers() {
        return descriptors.stream()
                .flatMap(descriptor -> descriptor.lowerers().stream())
                .toList();
    }

    private static BackendPartitionDescriptor descriptor(
            PartitionTarget target,
            Supplier<RegionLegalityAdapter> legalityAdapter,
            List<Supplier<RegionLowerer>> lowerers
    ) {
        return new DefaultBackendPartitionDescriptor(target, legalityAdapter, lowerers);
    }

    private static final class DefaultBackendPartitionDescriptor implements BackendPartitionDescriptor {
        private final PartitionTarget target;
        private final Supplier<RegionLegalityAdapter> legalityAdapterSupplier;
        private final List<Supplier<RegionLowerer>> lowererSuppliers;

        private DefaultBackendPartitionDescriptor(
                PartitionTarget target,
                Supplier<RegionLegalityAdapter> legalityAdapter,
                List<Supplier<RegionLowerer>> lowerers
        ) {
            this.target = target == null ? PartitionTarget.NONE : target;
            this.legalityAdapterSupplier = legalityAdapter == null
                    ? () -> new UnsupportedRegionLegalityAdapter(this.target)
                    : legalityAdapter;
            this.lowererSuppliers = List.copyOf(lowerers == null ? List.of() : lowerers);
        }

        @Override
        public PartitionTarget target() {
            return target;
        }

        @Override
        public RegionLegalityAdapter legalityAdapter() {
            return legalityAdapterSupplier.get();
        }

        @Override
        public List<RegionLowerer> lowerers() {
            return lowererSuppliers.stream()
                    .map(Supplier::get)
                    .toList();
        }
    }
}

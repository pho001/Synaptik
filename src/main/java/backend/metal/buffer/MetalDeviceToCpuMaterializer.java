package backend.metal.buffer;

import backend.accelerator.buffer.AcceleratorBufferLayout;
import backend.memory.CpuMaterializationReason;
import backend.memory.CpuMaterializationResult;
import backend.memory.DeviceBufferBinding;
import backend.memory.DeviceToCpuMaterializer;
import tensor.DataType;
import tensor.Tensor;

import java.util.Arrays;
import java.util.Objects;

/**
 * Device-to-CPU materializer for run-scoped Metal buffer bindings.
 */
public final class MetalDeviceToCpuMaterializer implements DeviceToCpuMaterializer {
    private final MetalBufferAllocator allocator;

    /**
     * Creates a materializer backed by the active run's Metal allocator.
     *
     * @param allocator allocator that can read Metal buffers into CPU storage
     */
    public MetalDeviceToCpuMaterializer(MetalBufferAllocator allocator) {
        this.allocator = Objects.requireNonNull(allocator, "allocator cannot be null");
    }

    /**
     * Returns whether the binding can be synchronized into the target tensor.
     */
    @Override
    public boolean supports(DeviceBufferBinding binding, Tensor target, CpuMaterializationReason reason) {
        if (!(binding instanceof MetalBufferBinding metalBinding) || target == null) {
            return false;
        }
        AcceleratorBufferLayout targetLayout = AcceleratorBufferLayout.fromTensor(target);
        return metalBinding.available()
                && metalBinding.layout().dataType() == DataType.FLOAT32
                && target.getDataType() == DataType.FLOAT32
                && Arrays.equals(metalBinding.layout().shape(), targetLayout.shape())
                && Arrays.equals(metalBinding.layout().strides(), targetLayout.strides())
                && metalBinding.layout().storageOffset() == targetLayout.storageOffset()
                && metalBinding.layout().logicalElementCount() == targetLayout.logicalElementCount();
    }

    /**
     * Reads the active Metal buffer into the target tensor's CPU-visible storage.
     */
    @Override
    public CpuMaterializationResult materialize(
            DeviceBufferBinding binding,
            Tensor target,
            CpuMaterializationReason reason
    ) {
        if (!(binding instanceof MetalBufferBinding metalBinding)) {
            throw new IllegalArgumentException("Metal materializer requires MetalBufferBinding.");
        }
        return allocator.readToCpu(metalBinding, target, reason);
    }
}

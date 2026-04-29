package backend.metal.buffer;

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
        return metalBinding.available()
                && metalBinding.layout().dataType() == DataType.FLOAT32
                && target.getDataType() == DataType.FLOAT32
                && Arrays.equals(metalBinding.layout().shape(), target.getShape())
                && metalBinding.layout().logicalElementCount() == target.getFlatDataSize();
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

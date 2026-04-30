package backend.metal.buffer;

import backend.accelerator.buffer.AcceleratorBufferLayout;
import backend.accelerator.buffer.AcceleratorBufferLayoutClass;
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
        AcceleratorBufferLayout bindingLayout = metalBinding.layout();
        AcceleratorBufferLayoutClass layoutClass = bindingLayout.layoutClass();
        MetalLayoutPolicy.Decision layoutDecision = MetalLayoutPolicy.output(bindingLayout);
        return metalBinding.available()
                && bindingLayout.dataType() == DataType.FLOAT32
                && target.getDataType() == DataType.FLOAT32
                && Arrays.equals(bindingLayout.shape(), targetLayout.shape())
                && Arrays.equals(bindingLayout.strides(), targetLayout.strides())
                && bindingLayout.storageOffset() == targetLayout.storageOffset()
                && bindingLayout.logicalElementCount() == targetLayout.logicalElementCount()
                && layoutDecision.accepted()
                && layoutClass != AcceleratorBufferLayoutClass.BROADCAST_ZERO_STRIDE_VIEW
                && layoutClass != AcceleratorBufferLayoutClass.UNSUPPORTED;
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

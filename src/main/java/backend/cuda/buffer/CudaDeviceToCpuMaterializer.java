package backend.cuda.buffer;

import runtime.device.buffer.AcceleratorBufferLayout;
import runtime.device.buffer.AcceleratorBufferLayoutClass;
import runtime.contract.CpuMaterializationReason;
import runtime.memory.CpuMaterializationResult;
import runtime.device.buffer.DeviceBufferBinding;
import runtime.memory.DeviceToCpuMaterializer;
import tensor.DataType;
import tensor.Tensor;

import java.util.Arrays;
import java.util.Objects;

/**
 * Device-to-CPU materializer for CUDA buffer bindings.
 */
public final class CudaDeviceToCpuMaterializer implements DeviceToCpuMaterializer {
    private final CudaBufferAllocator allocator;

    public CudaDeviceToCpuMaterializer(CudaBufferAllocator allocator) {
        this.allocator = Objects.requireNonNull(allocator, "allocator cannot be null");
    }

    @Override
    public boolean supports(DeviceBufferBinding binding, Tensor target, CpuMaterializationReason reason) {
        if (!(binding instanceof CudaBufferBinding cudaBinding) || target == null) {
            return false;
        }
        AcceleratorBufferLayout bindingLayout = cudaBinding.layout();
        AcceleratorBufferLayout targetLayout = AcceleratorBufferLayout.fromTensor(target);
        return cudaBinding.available()
                && bindingLayout.dataType() == DataType.FLOAT32
                && target.getDataType() == DataType.FLOAT32
                && bindingLayout.layoutClass() == AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS
                && Arrays.equals(bindingLayout.shape(), targetLayout.shape())
                && Arrays.equals(bindingLayout.strides(), targetLayout.strides())
                && bindingLayout.storageOffset() == targetLayout.storageOffset()
                && bindingLayout.logicalElementCount() == targetLayout.logicalElementCount();
    }

    @Override
    public CpuMaterializationResult materialize(
            DeviceBufferBinding binding,
            Tensor target,
            CpuMaterializationReason reason
    ) {
        if (!(binding instanceof CudaBufferBinding cudaBinding)) {
            throw new IllegalArgumentException("CUDA materializer requires CudaBufferBinding.");
        }
        return allocator.readToCpu(cudaBinding, target, reason);
    }
}

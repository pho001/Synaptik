package backend.metal.buffer;

import backend.accelerator.buffer.AcceleratorBufferLayout;
import backend.accelerator.buffer.AcceleratorBufferLayoutClass;
import backend.memory.CpuMaterializationReason;
import backend.memory.CpuMaterializationResult;
import backend.memory.DeviceBufferBinding;
import backend.memory.DeviceToNativeMaterializer;
import tensor.DataType;
import tensor.NativeFloat32Storage;
import tensor.NativeTensorStorage;
import tensor.Tensor;

import java.util.Arrays;
import java.util.Objects;

/**
 * Device-to-native materializer for dense FLOAT32 Metal buffer bindings.
 */
public final class MetalDeviceToNativeMaterializer implements DeviceToNativeMaterializer {
    private final MetalBufferAllocator allocator;

    public MetalDeviceToNativeMaterializer(MetalBufferAllocator allocator) {
        this.allocator = Objects.requireNonNull(allocator, "allocator cannot be null");
    }

    @Override
    public boolean supports(
            DeviceBufferBinding binding,
            Tensor target,
            NativeTensorStorage nativeStorage,
            CpuMaterializationReason reason
    ) {
        if (!(binding instanceof MetalBufferBinding metalBinding)
                || target == null
                || !(nativeStorage instanceof NativeFloat32Storage nativeF32)
                || !metalBinding.available()) {
            return false;
        }
        AcceleratorBufferLayout bindingLayout = metalBinding.layout();
        AcceleratorBufferLayout targetLayout = AcceleratorBufferLayout.fromTensor(target);
        return bindingLayout.dataType() == DataType.FLOAT32
                && target.getDataType() == DataType.FLOAT32
                && bindingLayout.layoutClass() == AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS
                && targetLayout.layoutClass() == AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS
                && target.isContiguous()
                && !target.hasStorageOffset()
                && Arrays.equals(bindingLayout.shape(), targetLayout.shape())
                && bindingLayout.logicalElementCount() == targetLayout.logicalElementCount()
                && nativeF32.getSize() == target.getFlatDataSize()
                && nativeF32.byteSize() >= bindingLayout.logicalByteLength();
    }

    @Override
    public CpuMaterializationResult materialize(
            DeviceBufferBinding binding,
            Tensor target,
            NativeTensorStorage nativeStorage,
            CpuMaterializationReason reason
    ) {
        if (!(binding instanceof MetalBufferBinding metalBinding)
                || !(nativeStorage instanceof NativeFloat32Storage nativeF32)) {
            throw new IllegalArgumentException("Metal native materializer requires MetalBufferBinding and NativeFloat32Storage.");
        }
        return allocator.readToNativeFloat32(metalBinding, target, nativeF32, reason);
    }
}

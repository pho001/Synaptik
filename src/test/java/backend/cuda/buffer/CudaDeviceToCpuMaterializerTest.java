package backend.cuda.buffer;

import backend.accelerator.buffer.AcceleratorBufferLayout;
import runtime.contract.CpuMaterializationReason;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CudaDeviceToCpuMaterializerTest {
    @Test
    void supportsOnlyMatchingCudaFloat32Binding() {
        CudaDeviceToCpuMaterializer materializer = new CudaDeviceToCpuMaterializer(allocator(new float[]{1.0f, 2.0f}));
        Tensor target = tensor(new float[]{0.0f, 0.0f});
        CudaBufferBinding binding = binding(3, layout(new int[]{2}, new int[]{1}, DataType.FLOAT32));
        CudaBufferBinding mismatched = binding(3, layout(new int[]{2}, new int[]{2}, DataType.FLOAT32));

        assertTrue(materializer.supports(binding, target, CpuMaterializationReason.GRAPH_OUTPUT));
        assertFalse(materializer.supports(mismatched, target, CpuMaterializationReason.GRAPH_OUTPUT));
        assertFalse(materializer.supports(null, target, CpuMaterializationReason.GRAPH_OUTPUT));
    }

    @Test
    void materializeDelegatesToAllocator() {
        CudaDeviceToCpuMaterializer materializer = new CudaDeviceToCpuMaterializer(allocator(new float[]{9.0f, 10.0f}));
        Tensor target = tensor(new float[]{0.0f, 0.0f});

        materializer.materialize(binding(5, layout(new int[]{2}, new int[]{1}, DataType.FLOAT32)), target, CpuMaterializationReason.CPU_CONSUMER);

        assertArrayEquals(new float[]{9.0f, 10.0f}, target.toFloat32ArrayCopy(), 0.0f);
    }

    private static CudaBufferAllocator allocator(float[] values) {
        return CudaBufferAllocator.available(new CudaBufferAllocator.NativeAccess() {
            @Override
            public CudaBufferHandle createBuffer(long byteLength, MemorySegment initialData, long initialDataBytes) {
                return new CudaBufferHandle(MemorySegment.ofAddress(99), byteLength, true);
            }

            @Override
            public void readBuffer(CudaBufferHandle handle, MemorySegment destination, long byteLength) {
                destination.copyFrom(MemorySegment.ofArray(values));
            }

            @Override
            public void destroyBuffer(CudaBufferHandle handle) {
            }
        });
    }

    private static CudaBufferBinding binding(int nodeId, AcceleratorBufferLayout layout) {
        return new CudaBufferBinding(
                nodeId,
                layout,
                new CudaBufferHandle(MemorySegment.ofAddress(nodeId + 10L), layout.logicalByteLength(), true),
                CudaBufferAccess.READ_WRITE
        );
    }

    private static AcceleratorBufferLayout layout(int[] shape, int[] strides, DataType dataType) {
        return AcceleratorBufferLayout.of(dataType, shape, strides, 0, shape[0]);
    }

    private static Tensor tensor(float[] values) {
        return new Tensor(values.clone(), new int[]{values.length}, null, "target", DataType.FLOAT32);
    }
}

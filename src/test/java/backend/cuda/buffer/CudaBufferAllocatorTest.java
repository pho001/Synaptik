package backend.cuda.buffer;

import backend.accelerator.buffer.AcceleratorBufferLayout;
import backend.memory.CpuMaterializationReason;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.lang.foreign.MemorySegment;
import java.util.HashMap;
import java.util.Map;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CudaBufferAllocatorTest {
    @Test
    void createInputBindingUploadsFloat32Tensor() {
        RecordingNativeAccess nativeAccess = new RecordingNativeAccess();
        CudaBufferAllocator allocator = CudaBufferAllocator.available(nativeAccess);
        Tensor tensor = tensor(new float[]{1.0f, -2.0f, 3.0f, 4.0f});

        CudaBufferBinding binding = allocator.createInputBinding(7, tensor);

        assertEquals(7, binding.nodeId());
        assertEquals(CudaBufferAccess.READ, binding.access());
        assertEquals("GPU_CUDA", binding.backendId());
        assertTrue(binding.available());
        assertEquals(16L, nativeAccess.lastCreatedBytes);
        assertArrayEquals(new float[]{1.0f, -2.0f, 3.0f, 4.0f}, nativeAccess.values(binding.handle()), 0.0f);
    }

    @Test
    void createOutputBindingAllocatesDenseFloat32Buffer() {
        RecordingNativeAccess nativeAccess = new RecordingNativeAccess();
        CudaBufferAllocator allocator = CudaBufferAllocator.available(nativeAccess);

        CudaBufferBinding binding = allocator.createOutputBinding(9, denseLayout());

        assertEquals(9, binding.nodeId());
        assertEquals(CudaBufferAccess.READ_WRITE, binding.access());
        assertEquals(16L, binding.logicalByteLength());
        assertTrue(binding.describe().contains("backend=GPU_CUDA"));
    }

    @Test
    void readToCpuCopiesFloat32Data() {
        RecordingNativeAccess nativeAccess = new RecordingNativeAccess();
        CudaBufferAllocator allocator = CudaBufferAllocator.available(nativeAccess);
        CudaBufferBinding binding = allocator.createOutputBinding(11, denseLayout());
        nativeAccess.put(binding.handle(), new float[]{5.0f, 6.0f, 7.0f, 8.0f});
        Tensor destination = tensor(new float[]{0.0f, 0.0f, 0.0f, 0.0f});

        var result = allocator.readToCpu(binding, destination, CpuMaterializationReason.GRAPH_OUTPUT);

        assertArrayEquals(new float[]{5.0f, 6.0f, 7.0f, 8.0f}, destination.getFloat32Data(), 0.0f);
        assertTrue(result.detail().contains("cuda read_buffer materialized nodeId=11"));
    }

    @Test
    void unavailableAllocatorRejectsAllocation() {
        CudaBufferAllocator allocator = CudaBufferAllocator.unavailable("missing CUDA buffer ABI");

        UnsupportedOperationException failure = assertThrows(
                UnsupportedOperationException.class,
                () -> allocator.createInputBinding(1, tensor(new float[]{1.0f}))
        );

        assertEquals("missing CUDA buffer ABI", failure.getMessage());
    }

    private static Tensor tensor(float[] data) {
        return new Tensor(data.clone(), new int[]{data.length}, null, "x", DataType.FLOAT32);
    }

    private static AcceleratorBufferLayout denseLayout() {
        return AcceleratorBufferLayout.of(DataType.FLOAT32, new int[]{4}, new int[]{1}, 0, 4);
    }

    private static final class RecordingNativeAccess implements CudaBufferAllocator.NativeAccess {
        private final Map<Long, float[]> valuesByAddress = new HashMap<>();
        private long nextAddress = 1L;
        private long lastCreatedBytes;

        @Override
        public CudaBufferHandle createBuffer(long byteLength, MemorySegment initialData, long initialDataBytes) {
            lastCreatedBytes = byteLength;
            CudaBufferHandle handle = new CudaBufferHandle(MemorySegment.ofAddress(nextAddress++), byteLength, true);
            float[] values = new float[(int) byteLength / Float.BYTES];
            if (initialData != null && !initialData.equals(MemorySegment.NULL) && initialDataBytes > 0) {
                MemorySegment.ofArray(values).copyFrom(initialData.reinterpret(initialDataBytes));
            }
            valuesByAddress.put(handle.handle().address(), values);
            return handle;
        }

        @Override
        public void readBuffer(CudaBufferHandle handle, MemorySegment destination, long byteLength) {
            float[] values = values(handle);
            destination.copyFrom(MemorySegment.ofArray(values));
        }

        @Override
        public void destroyBuffer(CudaBufferHandle handle) {
            valuesByAddress.remove(handle.handle().address());
        }

        void put(CudaBufferHandle handle, float[] values) {
            valuesByAddress.put(handle.handle().address(), values.clone());
        }

        float[] values(CudaBufferHandle handle) {
            return valuesByAddress.get(handle.handle().address()).clone();
        }
    }
}

package backend.metal.buffer;

import backend.memory.CpuMaterializationReason;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class MetalBufferAllocatorTest {
    @Test
    void bfloat16InputBindingAndReadbackPreserveRawStorageBits() {
        FakeNativeAccess nativeAccess = new FakeNativeAccess();
        MetalBufferAllocator allocator = MetalBufferAllocator.available(nativeAccess);
        short[] raw = new short[]{(short) 0x3f80, (short) 0xc020, (short) 0x0000};
        Tensor source = new Tensor(raw.clone(), new int[]{3}, null, "bf16", DataType.BFLOAT16);

        MetalBufferBinding binding = allocator.createInputBinding(7, source);
        Tensor destination = new Tensor(new short[]{0, 0, 0}, new int[]{3}, null, "out", DataType.BFLOAT16);
        allocator.readToCpu(binding, destination, CpuMaterializationReason.PUBLIC_DATA_ACCESS);

        assertEquals(DataType.BFLOAT16, binding.layout().dataType());
        assertEquals(raw.length * Short.BYTES, binding.logicalByteLength());
        assertArrayEquals(raw, destination.getBFloat16Data());
    }

    @Test
    void bfloat16OutputBindingAllocatesLogicalByteLength() {
        FakeNativeAccess nativeAccess = new FakeNativeAccess();
        MetalBufferAllocator allocator = MetalBufferAllocator.available(nativeAccess);
        Tensor output = new Tensor(new short[]{0, 0, 0, 0}, new int[]{2, 2}, null, "out", DataType.BFLOAT16);

        MetalBufferBinding binding = allocator.createOutputBinding(9, backend.accelerator.buffer.AcceleratorBufferLayout.fromTensor(output));

        assertEquals(DataType.BFLOAT16, binding.layout().dataType());
        assertEquals(4L * Short.BYTES, binding.logicalByteLength());
        assertEquals(4L * Short.BYTES, binding.handle().byteLength());
    }

    private static final class FakeNativeAccess implements MetalBufferAllocator.NativeAccess {
        private long nextAddress = 1_000L;
        private final Map<Long, byte[]> buffers = new HashMap<>();

        @Override
        public MetalBufferHandle createBuffer(long byteLength, int storageMode, MemorySegment initialData, long initialDataBytes) {
            long address = nextAddress++;
            byte[] storage = new byte[Math.toIntExact(byteLength)];
            if (initialData != null && !initialData.equals(MemorySegment.NULL) && initialDataBytes > 0) {
                byte[] initial = new byte[Math.toIntExact(initialDataBytes)];
                MemorySegment.ofArray(initial).copyFrom(initialData.reinterpret(initialDataBytes));
                System.arraycopy(initial, 0, storage, 0, Math.min(initial.length, storage.length));
            }
            buffers.put(address, storage);
            return new MetalBufferHandle(MemorySegment.ofAddress(address), byteLength, storageMode == 1 ? "shared" : "test", "test", true);
        }

        @Override
        public void readBuffer(MetalBufferHandle handle, MemorySegment destination, long byteLength) {
            byte[] storage = buffers.get(handle.nativeHandle().address());
            byte[] source = Arrays.copyOf(storage, Math.toIntExact(byteLength));
            destination.copyFrom(MemorySegment.ofArray(source));
        }

        @Override
        public void destroyBuffer(MetalBufferHandle handle) {
            buffers.remove(handle.nativeHandle().address());
        }
    }
}

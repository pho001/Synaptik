package backend.metal.buffer;

import backend.memory.CpuMaterializationReason;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class MetalBufferAllocatorTest {
    @Test
    void readToCpuRejectsNonContiguousDestinationBeforeNativeRead() {
        AtomicInteger reads = new AtomicInteger();
        MetalBufferAllocator allocator = MetalBufferAllocator.available(new MetalBufferAllocator.NativeAccess() {
            @Override
            public MetalBufferHandle createBuffer(long byteLength, int storageMode, MemorySegment initialData, long initialDataBytes) {
                throw new UnsupportedOperationException("not used");
            }

            @Override
            public void readBuffer(MetalBufferHandle handle, MemorySegment destination, long byteLength) {
                reads.incrementAndGet();
            }

            @Override
            public void destroyBuffer(MetalBufferHandle handle) {
            }
        });
        MetalBufferBinding binding = new MetalBufferBinding(
                7,
                DataType.FLOAT32,
                new int[]{2, 2},
                4,
                new MetalBufferHandle(MemorySegment.ofAddress(7), 16, "shared", "test", false),
                MetalBufferAccess.READ_WRITE
        );
        Tensor base = new Tensor(new float[]{0f, 0f, 0f, 0f}, new int[]{2, 2}, null, "base", DataType.FLOAT32);
        Tensor destination = base.permute(1, 0);

        assertThrows(
                UnsupportedOperationException.class,
                () -> allocator.readToCpu(binding, destination, CpuMaterializationReason.PUBLIC_DATA_ACCESS)
        );
        assertEquals(0, reads.get());
    }
}

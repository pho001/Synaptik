package backend.metal.buffer;

import org.junit.jupiter.api.Test;

import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class MetalBufferResourceTest {
    @Test
    void closeRemainsRetryableWhenDestroyFails() {
        AtomicInteger destroys = new AtomicInteger();
        MetalBufferAllocator allocator = MetalBufferAllocator.available(new MetalBufferAllocator.NativeAccess() {
            @Override
            public MetalBufferHandle createBuffer(long byteLength, int storageMode, MemorySegment initialData, long initialDataBytes) {
                throw new UnsupportedOperationException("not used");
            }

            @Override
            public void readBuffer(MetalBufferHandle handle, MemorySegment destination, long byteLength) {
                throw new UnsupportedOperationException("not used");
            }

            @Override
            public void destroyBuffer(MetalBufferHandle handle) {
                if (destroys.incrementAndGet() == 1) {
                    throw new UnsupportedOperationException("synthetic destroy failure");
                }
            }
        });
        MetalBufferHandle handle = new MetalBufferHandle(MemorySegment.ofAddress(11), 4, "shared", "test", true);
        MetalBufferResource resource = new MetalBufferResource(allocator, handle);

        assertThrows(UnsupportedOperationException.class, resource::close);
        assertDoesNotThrow(resource::close);
        assertEquals(2, destroys.get());
    }
}

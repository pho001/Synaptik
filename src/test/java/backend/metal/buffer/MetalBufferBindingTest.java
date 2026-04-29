package backend.metal.buffer;

import org.junit.jupiter.api.Test;
import tensor.DataType;

import java.lang.foreign.MemorySegment;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MetalBufferBindingTest {
    @Test
    void bindingReportsLogicalPayloadCoverage() {
        MetalBufferHandle handle = new MetalBufferHandle(MemorySegment.ofAddress(1), 16, "shared", "test", false);
        MetalBufferBinding binding = new MetalBufferBinding(
                7,
                DataType.FLOAT32,
                new int[]{2, 2},
                4,
                handle,
                MetalBufferAccess.READ
        );

        assertTrue(binding.bufferCoversLogicalPayload());
        assertTrue(binding.available());
        assertTrue(binding.backendId().equals("GPU_METAL"));
        assertTrue(binding.describe().contains("nodeId=7"));
        assertTrue(binding.describe().contains("bytes=16"));
    }

    @Test
    void handleIdentifiesOnlySharedStorageAsHostShared() {
        MetalBufferHandle shared = new MetalBufferHandle(MemorySegment.ofAddress(1), 16, "shared", "test", false);
        MetalBufferHandle mixedCaseShared = new MetalBufferHandle(MemorySegment.ofAddress(2), 16, "SHARED", "test", false);
        MetalBufferHandle privateStorage = new MetalBufferHandle(MemorySegment.ofAddress(3), 16, "private", "test", false);
        MetalBufferHandle unknownStorage = new MetalBufferHandle(MemorySegment.ofAddress(4), 16, "", "test", false);

        assertTrue(shared.hostShared());
        assertTrue(mixedCaseShared.hostShared());
        assertFalse(privateStorage.hostShared());
        assertFalse(unknownStorage.hostShared());
    }

    @Test
    void shapeAccessorReturnsDefensiveCopy() {
        MetalBufferHandle handle = new MetalBufferHandle(MemorySegment.ofAddress(1), 32, "shared", "test", false);
        int[] shape = {2, 4};
        MetalBufferBinding binding = new MetalBufferBinding(3, DataType.FLOAT32, shape, 8, handle, MetalBufferAccess.WRITE);

        shape[0] = 99;
        int[] read = binding.shape();
        read[1] = 99;

        assertArrayEquals(new int[]{2, 4}, binding.shape());
    }

    @Test
    void unavailableOrSmallHandleDoesNotCoverPayload() {
        MetalBufferBinding unavailable = new MetalBufferBinding(
                1,
                DataType.FLOAT32,
                new int[]{2},
                2,
                new MetalBufferHandle(MemorySegment.NULL, 8, "shared", "test", false),
                MetalBufferAccess.READ
        );
        MetalBufferBinding tooSmall = new MetalBufferBinding(
                2,
                DataType.FLOAT64,
                new int[]{2},
                2,
                new MetalBufferHandle(MemorySegment.ofAddress(1), 8, "shared", "test", false),
                MetalBufferAccess.READ
        );

        assertFalse(unavailable.bufferCoversLogicalPayload());
        assertFalse(tooSmall.bufferCoversLogicalPayload());
    }
}

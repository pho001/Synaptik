package backend.metal.buffer;

import backend.accelerator.buffer.AcceleratorBufferAccessMode;
import backend.accelerator.buffer.AcceleratorBufferLayout;
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
                AcceleratorBufferLayout.of(DataType.FLOAT32, new int[]{2, 2}, new int[]{2, 1}, 0, 4),
                handle,
                MetalBufferAccess.READ
        );

        assertTrue(binding.bufferCoversLogicalPayload());
        assertTrue(binding.available());
        assertTrue(binding.backendId().equals("GPU_METAL"));
        assertTrue(binding.nativeHandleIdentity().contains("GPU_METAL"));
        assertTrue(binding.nativeHandleIdentity().contains("owner=test"));
        assertTrue(binding.nativeHandleIdentity().contains("storageMode=shared"));
        assertTrue(binding.nativeHandleIdentity().contains("bytes=16"));
        assertTrue(binding.accessMode().equals(AcceleratorBufferAccessMode.READ));
        assertTrue(binding.describe().contains("nodeId=7"));
        assertTrue(binding.describe().contains("layoutClass=DENSE_CONTIGUOUS"));
        assertTrue(binding.describe().contains("strides=[2, 1]"));
        assertTrue(binding.describe().contains("storageOffset=0"));
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
        AcceleratorBufferLayout layout = AcceleratorBufferLayout.of(DataType.FLOAT32, shape, new int[]{4, 1}, 0, 8);
        MetalBufferBinding binding = new MetalBufferBinding(3, layout, handle, MetalBufferAccess.WRITE);

        shape[0] = 99;
        int[] read = binding.layout().shape();
        read[1] = 99;

        assertArrayEquals(new int[]{2, 4}, binding.layout().shape());
        assertTrue(binding.accessMode().equals(AcceleratorBufferAccessMode.WRITE));
    }

    @Test
    void unavailableOrSmallHandleDoesNotCoverPayload() {
        MetalBufferBinding unavailable = new MetalBufferBinding(
                1,
                AcceleratorBufferLayout.of(DataType.FLOAT32, new int[]{2}, new int[]{1}, 0, 2),
                new MetalBufferHandle(MemorySegment.NULL, 8, "shared", "test", false),
                MetalBufferAccess.READ
        );
        MetalBufferBinding tooSmall = new MetalBufferBinding(
                2,
                AcceleratorBufferLayout.of(DataType.FLOAT64, new int[]{2}, new int[]{1}, 0, 2),
                new MetalBufferHandle(MemorySegment.ofAddress(1), 8, "shared", "test", false),
                MetalBufferAccess.READ
        );

        assertFalse(unavailable.bufferCoversLogicalPayload());
        assertFalse(tooSmall.bufferCoversLogicalPayload());
    }
}

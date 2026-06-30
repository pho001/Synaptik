package runtime.device.buffer;

import org.junit.jupiter.api.Test;
import tensor.DataType;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AcceleratorLayoutAbiV2DescriptorTest {
    @Test
    void describesDenseBindingWithoutBackendHandleLeak() {
        AcceleratorBufferLayout layout = AcceleratorBufferLayout.of(
                DataType.FLOAT32,
                new int[]{2, 3},
                new int[]{3, 1},
                0,
                6
        );

        AcceleratorLayoutAbiV2Descriptor descriptor = AcceleratorLayoutAbiV2Descriptor.fromBinding(binding(layout));

        assertEquals("GPU_TEST", descriptor.backendId());
        assertEquals(7, descriptor.nodeId());
        assertEquals(DataType.FLOAT32, descriptor.dataType());
        assertEquals(2, descriptor.rank());
        assertArrayEquals(new int[]{2, 3}, descriptor.shape());
        assertArrayEquals(new int[]{3, 1}, descriptor.strides());
        assertEquals(0, descriptor.storageOffset());
        assertEquals(6, descriptor.logicalElementCount());
        assertEquals(24, descriptor.logicalByteLength());
        assertEquals(24, descriptor.physicalByteSpan());
        assertEquals(AcceleratorBufferAccessMode.READ_WRITE, descriptor.accessMode());
        assertEquals(AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS, descriptor.layoutClass());
        assertEquals("GPU_TEST:test-handle", descriptor.nativeHandleIdentity());
    }

    @Test
    void computesPhysicalSpanForNonZeroOffsetDenseView() {
        AcceleratorBufferLayout layout = AcceleratorBufferLayout.of(
                DataType.FLOAT32,
                new int[]{2, 3},
                new int[]{3, 1},
                2,
                6
        );

        AcceleratorLayoutAbiV2Descriptor descriptor = AcceleratorLayoutAbiV2Descriptor.fromBinding(binding(layout));

        assertEquals(32L, descriptor.physicalByteSpan());
    }

    @Test
    void computesPhysicalSpanForPermutedView() {
        AcceleratorBufferLayout layout = AcceleratorBufferLayout.of(
                DataType.FLOAT32,
                new int[]{2, 3},
                new int[]{1, 2},
                0,
                6
        );

        AcceleratorLayoutAbiV2Descriptor descriptor = AcceleratorLayoutAbiV2Descriptor.fromBinding(binding(layout));

        assertEquals(24L, descriptor.physicalByteSpan());
    }

    @Test
    void representsBroadcastZeroStrideView() {
        AcceleratorBufferLayout layout = AcceleratorBufferLayout.of(
                DataType.BFLOAT16,
                new int[]{2, 3},
                new int[]{0, 1},
                0,
                6
        );

        AcceleratorLayoutAbiV2Descriptor descriptor = AcceleratorLayoutAbiV2Descriptor.fromBinding(binding(layout));

        assertEquals(AcceleratorBufferLayoutClass.BROADCAST_ZERO_STRIDE_VIEW, descriptor.layoutClass());
        assertEquals(6L, descriptor.physicalByteSpan());
    }

    @Test
    void rejectsNegativeStrideMetadata() {
        AcceleratorBufferLayout layout = AcceleratorBufferLayout.of(
                DataType.FLOAT32,
                new int[]{2, 3},
                new int[]{3, -1},
                0,
                6
        );

        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
                () -> AcceleratorLayoutAbiV2Descriptor.fromBinding(binding(layout)));

        assertTrue(failure.getMessage().contains("negative stride"));
    }

    @Test
    void rejectsPhysicalSpanOverflow() {
        assertThrows(ArithmeticException.class,
                () -> AcceleratorLayoutAbiV2Descriptor.physicalByteSpan(
                        DataType.FLOAT64,
                        new int[]{Integer.MAX_VALUE, Integer.MAX_VALUE},
                        new int[]{Integer.MAX_VALUE, Integer.MAX_VALUE},
                        Integer.MAX_VALUE
                ));
    }

    @Test
    void descriptorAccessorsReturnDefensiveCopies() {
        AcceleratorBufferLayout layout = AcceleratorBufferLayout.of(
                DataType.FLOAT32,
                new int[]{2, 3},
                new int[]{3, 1},
                0,
                6
        );
        AcceleratorLayoutAbiV2Descriptor descriptor = AcceleratorLayoutAbiV2Descriptor.fromBinding(binding(layout));

        int[] shape = descriptor.shape();
        int[] strides = descriptor.strides();
        shape[0] = 99;
        strides[0] = 99;

        assertArrayEquals(new int[]{2, 3}, descriptor.shape());
        assertArrayEquals(new int[]{3, 1}, descriptor.strides());
    }

    private static DeviceBufferBinding binding(AcceleratorBufferLayout layout) {
        return new DeviceBufferBinding() {
            @Override
            public int nodeId() {
                return 7;
            }

            @Override
            public String backendId() {
                return "GPU_TEST";
            }

            @Override
            public AcceleratorBufferLayout layout() {
                return layout;
            }

            @Override
            public AcceleratorBufferAccessMode accessMode() {
                return AcceleratorBufferAccessMode.READ_WRITE;
            }

            @Override
            public String nativeHandleIdentity() {
                return "GPU_TEST:test-handle";
            }

            @Override
            public boolean available() {
                return true;
            }

            @Override
            public String describe() {
                return "test binding";
            }
        };
    }
}

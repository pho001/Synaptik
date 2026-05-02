package backend.accelerator.buffer;

import backend.memory.DeviceBufferBinding;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AcceleratorLayoutTransformPlannerTest {
    @Test
    void metadataOnlyPermuteViewIsAcceptedWhenSourceBindingAvailable() {
        AcceleratorBufferLayout sourceLayout = layout(new int[]{2, 3}, new int[]{3, 1});
        AcceleratorBufferLayout targetLayout = layout(new int[]{3, 2}, new int[]{1, 3});

        AcceleratorLayoutTransformDecision decision = AcceleratorLayoutTransformPlanner.decide(
                request(Operation.OpType.PERMUTE, sourceLayout, targetLayout, binding(sourceLayout)));

        assertTrue(decision.accepted());
        assertEquals(AcceleratorLayoutTransformKind.METADATA_ONLY_VIEW, decision.kind());
        assertEquals(AcceleratorBufferReasonCode.GPU_LAYOUT_VIEW_BINDING_AVAILABLE, decision.reasonCode());
        assertTrue(decision.reason().contains("metadata-only view"));
    }

    @Test
    void metadataOnlyExpandViewIsAcceptedAsReadOnlyView() {
        AcceleratorBufferLayout sourceLayout = layout(new int[]{1, 3}, new int[]{3, 1});
        AcceleratorBufferLayout targetLayout = layout(new int[]{2, 3}, new int[]{0, 1});

        AcceleratorLayoutTransformDecision decision = AcceleratorLayoutTransformPlanner.decide(
                request(Operation.OpType.EXPAND, sourceLayout, targetLayout, binding(sourceLayout)));

        assertTrue(decision.accepted());
        assertEquals(AcceleratorLayoutTransformKind.METADATA_ONLY_VIEW, decision.kind());
        assertEquals(AcceleratorBufferReasonCode.GPU_LAYOUT_VIEW_BINDING_AVAILABLE, decision.reasonCode());
        assertTrue(decision.reason().contains("metadata-only view"));
    }

    @Test
    void contiguousRequestsDenseGpuMaterialization() {
        AcceleratorBufferLayout sourceLayout = layout(new int[]{3, 2}, new int[]{1, 3});
        AcceleratorBufferLayout targetLayout = layout(new int[]{3, 2}, new int[]{2, 1});

        AcceleratorLayoutTransformDecision decision = AcceleratorLayoutTransformPlanner.decide(
                request(Operation.OpType.CONTIGUOUS, sourceLayout, targetLayout, binding(sourceLayout)));

        assertTrue(decision.accepted());
        assertEquals(AcceleratorLayoutTransformKind.DENSE_GPU_MATERIALIZATION, decision.kind());
        assertEquals(AcceleratorBufferReasonCode.GPU_LAYOUT_DENSE_MATERIALIZATION_AVAILABLE, decision.reasonCode());
        assertTrue(decision.reason().contains("dense GPU materialization"));
    }

    @Test
    void broadcastContiguousRequestsBroadcastGpuMaterialization() {
        AcceleratorBufferLayout sourceLayout = layout(new int[]{2, 3}, new int[]{0, 1});
        AcceleratorBufferLayout targetLayout = layout(new int[]{2, 3}, new int[]{3, 1});

        AcceleratorLayoutTransformDecision decision = AcceleratorLayoutTransformPlanner.decide(
                request(Operation.OpType.CONTIGUOUS, sourceLayout, targetLayout, binding(sourceLayout)));

        assertTrue(decision.accepted());
        assertEquals(AcceleratorLayoutTransformKind.BROADCAST_GPU_MATERIALIZATION, decision.kind());
        assertEquals(AcceleratorBufferReasonCode.GPU_LAYOUT_BROADCAST_MATERIALIZATION_AVAILABLE, decision.reasonCode());
        assertTrue(decision.reason().contains("broadcast GPU materialization"));
    }

    @Test
    void broadcastContiguousRejectsNonDenseTargetWithStableReason() {
        AcceleratorBufferLayout sourceLayout = layout(new int[]{2, 3}, new int[]{0, 1});
        AcceleratorBufferLayout targetLayout = layout(new int[]{2, 3}, new int[]{0, 1});

        AcceleratorLayoutTransformDecision decision = AcceleratorLayoutTransformPlanner.decide(
                request(Operation.OpType.CONTIGUOUS, sourceLayout, targetLayout, binding(sourceLayout)));

        assertFalse(decision.accepted());
        assertEquals(AcceleratorLayoutTransformKind.UNSUPPORTED, decision.kind());
        assertEquals(AcceleratorBufferReasonCode.GPU_LAYOUT_BROADCAST_MATERIALIZATION_UNSUPPORTED, decision.reasonCode());
        assertTrue(decision.reason().contains("requires dense contiguous target"));
    }

    @Test
    void nonContiguousReshapeRequestsDenseGpuMaterialization() {
        AcceleratorBufferLayout sourceLayout = layout(new int[]{3, 2}, new int[]{1, 3});
        AcceleratorBufferLayout targetLayout = layout(new int[]{6}, new int[]{1});

        AcceleratorLayoutTransformDecision decision = AcceleratorLayoutTransformPlanner.decide(
                request(Operation.OpType.RESHAPE, sourceLayout, targetLayout, binding(sourceLayout)));

        assertTrue(decision.accepted());
        assertEquals(AcceleratorLayoutTransformKind.DENSE_GPU_MATERIALIZATION, decision.kind());
        assertEquals(AcceleratorBufferReasonCode.GPU_LAYOUT_DENSE_MATERIALIZATION_AVAILABLE, decision.reasonCode());
        assertTrue(decision.reason().contains("dense GPU materialization"));
    }

    @Test
    void stridedComputeRejectsWithStridedNativeReason() {
        AcceleratorBufferLayout sourceLayout = layout(new int[]{3, 2}, new int[]{1, 3});
        AcceleratorBufferLayout targetLayout = layout(new int[]{3, 2}, new int[]{1, 3});

        AcceleratorLayoutTransformDecision decision = AcceleratorLayoutTransformPlanner.decide(
                request(Operation.OpType.RELU, sourceLayout, targetLayout, binding(sourceLayout)));

        assertFalse(decision.accepted());
        assertEquals(AcceleratorLayoutTransformKind.UNSUPPORTED, decision.kind());
        assertEquals(AcceleratorBufferReasonCode.GPU_LAYOUT_STRIDED_NATIVE_COMPUTE_UNSUPPORTED, decision.reasonCode());
        assertTrue(decision.reason().contains("direct strided native compute unsupported"));
    }

    @Test
    void missingSourceBindingRejectsWithStableReason() {
        AcceleratorBufferLayout sourceLayout = layout(new int[]{2, 3}, new int[]{3, 1});
        AcceleratorBufferLayout targetLayout = layout(new int[]{3, 2}, new int[]{1, 3});

        AcceleratorLayoutTransformDecision decision = AcceleratorLayoutTransformPlanner.decide(
                request(Operation.OpType.PERMUTE, sourceLayout, targetLayout, null));

        assertFalse(decision.accepted());
        assertEquals(AcceleratorLayoutTransformKind.UNSUPPORTED, decision.kind());
        assertEquals(AcceleratorBufferReasonCode.GPU_LAYOUT_SOURCE_BINDING_UNAVAILABLE, decision.reasonCode());
        assertTrue(decision.reason().contains("source binding unavailable"));
    }

    @Test
    void backendMismatchRejectsWithStableReason() {
        AcceleratorBufferLayout sourceLayout = layout(new int[]{2, 3}, new int[]{3, 1});
        AcceleratorBufferLayout targetLayout = layout(new int[]{3, 2}, new int[]{1, 3});

        AcceleratorLayoutTransformDecision decision = AcceleratorLayoutTransformPlanner.decide(
                new AcceleratorLayoutTransformRequest(
                        "GPU_OTHER",
                        11,
                        12,
                        Operation.OpType.PERMUTE,
                        sourceLayout,
                        targetLayout,
                        binding(sourceLayout),
                        false
                ));

        assertFalse(decision.accepted());
        assertEquals(AcceleratorLayoutTransformKind.UNSUPPORTED, decision.kind());
        assertEquals(AcceleratorBufferReasonCode.GPU_LAYOUT_BACKEND_MISMATCH, decision.reasonCode());
        assertTrue(decision.reason().contains("backend mismatch"));
    }

    @Test
    void negativeStrideRejectsWithNativeLayoutReason() {
        AcceleratorBufferLayout sourceLayout = layout(new int[]{2, 3}, new int[]{3, -1});
        AcceleratorBufferLayout targetLayout = layout(new int[]{3, 2}, new int[]{1, 3});

        AcceleratorLayoutTransformDecision decision = AcceleratorLayoutTransformPlanner.decide(
                request(Operation.OpType.PERMUTE, sourceLayout, targetLayout, binding(sourceLayout)));

        assertFalse(decision.accepted());
        assertEquals(AcceleratorLayoutTransformKind.UNSUPPORTED, decision.kind());
        assertEquals(AcceleratorBufferReasonCode.NATIVE_LAYOUT_METADATA_UNSUPPORTED, decision.reasonCode());
        assertTrue(decision.reason().contains("native layout metadata unsupported"));
    }

    private static AcceleratorLayoutTransformRequest request(
            Operation.OpType opType,
            AcceleratorBufferLayout sourceLayout,
            AcceleratorBufferLayout targetLayout,
            DeviceBufferBinding sourceBinding
    ) {
        return new AcceleratorLayoutTransformRequest(
                "GPU_TEST",
                11,
                12,
                opType,
                sourceLayout,
                targetLayout,
                sourceBinding,
                false
        );
    }

    private static AcceleratorBufferLayout layout(int[] shape, int[] strides) {
        return AcceleratorBufferLayout.of(DataType.FLOAT32, shape, strides, 0, elementCount(shape));
    }

    private static long elementCount(int[] shape) {
        long count = 1L;
        for (int dimension : shape) {
            count *= dimension;
        }
        return count;
    }

    private static DeviceBufferBinding binding(AcceleratorBufferLayout layout) {
        return binding(layout, true);
    }

    private static DeviceBufferBinding binding(AcceleratorBufferLayout layout, boolean available) {
        return new DeviceBufferBinding() {
            @Override
            public int nodeId() {
                return 11;
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
                return "GPU_TEST:view-source";
            }

            @Override
            public boolean available() {
                return available;
            }

            @Override
            public String describe() {
                return "test binding";
            }
        };
    }
}

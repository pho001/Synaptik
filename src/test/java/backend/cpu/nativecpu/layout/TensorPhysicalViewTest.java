package backend.cpu.nativecpu.layout;

import backend.cpu.nativecpu.NativeCpuStorageFactory;
import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptor;
import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.storage.NativeTensorStorage;
import tensor.Tensor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TensorPhysicalViewTest {
    @Test
    void describesDenseOffsetTransposeAndBroadcastViewsFromCompiledDescriptors() {
        Tensor denseTensor = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "dense", DataType.FLOAT32);
        Tensor selectedTensor = denseTensor.select(0, 1);
        Tensor transposedTensor = denseTensor.permute(1, 0);
        Tensor broadcastTensor = new Tensor(new short[]{1, 2, 3}, new int[]{1, 3}, null, "bf16_bias", DataType.BFLOAT16)
                .expand(2, 3);

        TensorPhysicalView dense = TensorPhysicalView.fromDescriptor(descriptorFor(denseTensor, 0), NativeCpuStorageFamily.CPU_NATIVE);
        TensorPhysicalView selected = TensorPhysicalView.fromDescriptor(descriptorFor(selectedTensor, 1), NativeCpuStorageFamily.CPU_NATIVE);
        TensorPhysicalView transposed = TensorPhysicalView.fromDescriptor(descriptorFor(transposedTensor, 1), NativeCpuStorageFamily.CPU_NATIVE);
        TensorPhysicalView broadcast = TensorPhysicalView.fromDescriptor(descriptorFor(broadcastTensor, 1), NativeCpuStorageFamily.CPU_NATIVE);

        assertEquals(NativeCpuLayoutClass.DENSE_CONTIGUOUS, dense.layoutClass());
        assertArrayEquals(new int[]{2, 3}, dense.shape());
        assertArrayEquals(new int[]{3, 1}, dense.elementStrides());
        assertEquals(24L, dense.physicalByteSpan());

        assertEquals(NativeCpuLayoutClass.OFFSET_CONTIGUOUS, selected.layoutClass());
        assertArrayEquals(new int[]{3}, selected.shape());
        assertArrayEquals(new int[]{1}, selected.elementStrides());
        assertEquals(3, selected.storageOffsetElements());
        assertEquals(12L, selected.baseByteOffset());
        assertEquals(24L, selected.physicalByteSpan());

        assertEquals(NativeCpuLayoutClass.TRANSPOSE_2D_READ_DENSE_WRITE, transposed.layoutClass());
        assertArrayEquals(new int[]{3, 2}, transposed.shape());
        assertArrayEquals(new int[]{1, 3}, transposed.elementStrides());
        assertEquals(24L, transposed.physicalByteSpan());

        assertEquals(NativeCpuLayoutClass.LAST_DIM_BIAS_BROADCAST, broadcast.layoutClass());
        assertArrayEquals(new int[]{2, 3}, broadcast.shape());
        assertArrayEquals(new int[]{0, 1}, broadcast.elementStrides());
        assertEquals(12L, broadcast.logicalByteLength());
        assertEquals(6L, broadcast.physicalByteSpan());
        assertTrue(broadcast.hasZeroStride());

        TensorPhysicalView rowBroadcast = TensorPhysicalView.of(
                2,
                DataType.FLOAT32,
                new int[]{2, 3},
                new int[]{1, 0},
                0,
                NativeCpuStorageFamily.CPU_NATIVE
        );
        assertEquals(NativeCpuLayoutClass.BROADCAST_READ_DENSE_WRITE, rowBroadcast.layoutClass());
    }

    @Test
    void lowersNativeSegmentViewWithByteStridesAndBounds() {
        TensorPhysicalView view = TensorPhysicalView.of(
                7,
                DataType.FLOAT32,
                new int[]{3},
                new int[]{1},
                3,
                NativeCpuStorageFamily.CPU_NATIVE
        );
        NativeTensorStorage storage = new NativeCpuStorageFactory().allocate(DataType.FLOAT32, 6, "segment-view");
        try {
            NativeSegmentView segmentView = NativeSegmentView.from(view, storage);

            assertEquals(12L, segmentView.baseByteOffset());
            assertArrayEquals(new long[]{4L}, segmentView.byteStrides());
            assertEquals(4L, segmentView.elementSizeBytes());
            assertEquals(24L, segmentView.physicalByteSpan());
            assertEquals(12L, segmentView.byteOffsetForLogicalIndex(0));
            assertEquals(20L, segmentView.byteOffsetForLogicalIndex(2));
        } finally {
            storage.close();
        }
    }

    @Test
    void rejectsInvalidNativeSegmentViews() {
        TensorPhysicalView f32View = TensorPhysicalView.of(
                7,
                DataType.FLOAT32,
                new int[]{3},
                new int[]{1},
                3,
                NativeCpuStorageFamily.CPU_NATIVE
        );
        NativeTensorStorage tooSmall = new NativeCpuStorageFactory().allocate(DataType.FLOAT32, 3, "too-small");
        NativeTensorStorage wrongDType = new NativeCpuStorageFactory().allocate(DataType.FLOAT64, 6, "wrong-dtype");
        try {
            assertThrows(IllegalArgumentException.class, () -> NativeSegmentView.from(f32View, tooSmall));
            assertThrows(IllegalArgumentException.class, () -> NativeSegmentView.from(f32View, wrongDType));
        } finally {
            tooSmall.close();
            wrongDType.close();
        }
    }

    @Test
    void rejectsNegativeStrideAndDefensivelyCopiesArrays() {
        assertThrows(IllegalArgumentException.class, () -> TensorPhysicalView.of(
                3,
                DataType.FLOAT32,
                new int[]{2, 3},
                new int[]{3, -1},
                0,
                NativeCpuStorageFamily.CPU_NATIVE
        ));

        int[] shape = new int[]{2, 3};
        int[] strides = new int[]{3, 1};
        TensorPhysicalView view = TensorPhysicalView.of(4, DataType.FLOAT32, shape, strides, 0, NativeCpuStorageFamily.CPU_NATIVE);
        shape[0] = 99;
        strides[0] = 99;
        int[] returnedShape = view.shape();
        int[] returnedStrides = view.elementStrides();
        returnedShape[0] = 77;
        returnedStrides[0] = 77;

        assertArrayEquals(new int[]{2, 3}, view.shape());
        assertArrayEquals(new int[]{3, 1}, view.elementStrides());
    }

    private static CompiledTensorDescriptor descriptorFor(Tensor tensor, int nodeId) {
        List<CompiledNode> nodes = CompiledNode.snapshot(tensor.topologicalSort());
        return CompiledTensorDescriptorBuilder.build(nodes).byNodeId(nodeId);
    }
}

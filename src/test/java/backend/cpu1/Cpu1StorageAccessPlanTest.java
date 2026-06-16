package backend.cpu1;

import backend.cpu1.storage.Cpu1StorageAccessKind;
import backend.cpu1.storage.Cpu1StorageAccessPlan;
import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptor;
import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import graph.compile.intent.BackendIntentPlan;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

class Cpu1StorageAccessPlanTest {
    @Test
    void classifiesDenseContiguousDescriptor() {
        Tensor tensor = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "dense",
                DataType.FLOAT32
        );

        Cpu1StorageAccessPlan plan = Cpu1StorageAccessPlan.fromDescriptor(descriptorFor(tensor));

        assertEquals(Cpu1StorageAccessKind.DENSE_CONTIGUOUS, plan.kind());
        assertArrayEquals(new int[]{2, 3}, plan.shape());
        assertArrayEquals(new int[]{3, 1}, plan.strides());
        assertEquals(0, plan.storageOffset());
        assertEquals(6L, plan.elementCount());
        assertNull(plan.rejectionReason());
    }

    @Test
    void classifiesDenseContiguousNodeWithStorageOffset() {
        Tensor base = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "base",
                DataType.FLOAT32
        );
        Tensor selected = base.select(0, 1);

        Cpu1StorageAccessPlan plan = Cpu1StorageAccessPlan.fromNode(nodeFor(selected));

        assertEquals(Cpu1StorageAccessKind.DENSE_WITH_OFFSET, plan.kind());
        assertArrayEquals(new int[]{3}, plan.shape());
        assertArrayEquals(new int[]{1}, plan.strides());
        assertEquals(3, plan.storageOffset());
        assertEquals(3L, plan.elementCount());
        assertNull(plan.rejectionReason());
    }

    @Test
    void classifiesStridedDescriptor() {
        Tensor base = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "base",
                DataType.FLOAT32
        );
        Tensor transposed = base.permute(1, 0);

        Cpu1StorageAccessPlan plan = Cpu1StorageAccessPlan.fromDescriptor(descriptorFor(transposed));

        assertEquals(Cpu1StorageAccessKind.STRIDED, plan.kind());
        assertArrayEquals(new int[]{3, 2}, plan.shape());
        assertArrayEquals(new int[]{1, 3}, plan.strides());
        assertEquals(0, plan.storageOffset());
        assertEquals(6L, plan.elementCount());
        assertNull(plan.rejectionReason());
    }

    @Test
    void classifiesBroadcastDescriptor() {
        Tensor row = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f},
                new int[]{1, 3},
                null,
                "row",
                DataType.FLOAT32
        );
        Tensor broadcast = row.expand(2, 3);

        Cpu1StorageAccessPlan plan = Cpu1StorageAccessPlan.fromDescriptor(descriptorFor(broadcast));

        assertEquals(Cpu1StorageAccessKind.BROADCAST, plan.kind());
        assertArrayEquals(new int[]{2, 3}, plan.shape());
        assertArrayEquals(new int[]{0, 1}, plan.strides());
        assertEquals(0, plan.storageOffset());
        assertEquals(6L, plan.elementCount());
        assertNull(plan.rejectionReason());
    }

    @Test
    void classifiesBroadcastedLogicalSameShapeAsDenseContiguous() {
        Tensor tensor = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "dense",
                DataType.FLOAT32
        );

        Cpu1StorageAccessPlan plan = Cpu1StorageAccessPlan.forBroadcastedLogicalShape(
                descriptorFor(tensor),
                new int[]{2, 3}
        );

        assertEquals(Cpu1StorageAccessKind.DENSE_CONTIGUOUS, plan.kind());
        assertArrayEquals(new int[]{2, 3}, plan.shape());
        assertArrayEquals(new int[]{3, 1}, plan.strides());
        assertEquals(0, plan.storageOffset());
        assertEquals(6L, plan.elementCount());
        assertNull(plan.rejectionReason());
    }

    @Test
    void classifiesBroadcastedLogicalExpandedRowAsBroadcast() {
        Tensor row = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f},
                new int[]{1, 3},
                null,
                "row",
                DataType.FLOAT32
        );

        Cpu1StorageAccessPlan plan = Cpu1StorageAccessPlan.forBroadcastedLogicalShape(
                descriptorFor(row),
                new int[]{2, 3}
        );

        assertEquals(Cpu1StorageAccessKind.BROADCAST, plan.kind());
        assertArrayEquals(new int[]{2, 3}, plan.shape());
        assertArrayEquals(new int[]{0, 1}, plan.strides());
        assertEquals(0, plan.storageOffset());
        assertEquals(6L, plan.elementCount());
        assertNull(plan.rejectionReason());
    }

    @Test
    void defensivelyCopiesShapeAndStrides() {
        int[] shape = new int[]{2, 3};
        int[] strides = new int[]{3, 1};
        Cpu1StorageAccessPlan plan = new Cpu1StorageAccessPlan(
                Cpu1StorageAccessKind.DENSE_CONTIGUOUS,
                shape,
                strides,
                0,
                6L,
                null
        );

        shape[0] = 99;
        strides[0] = 99;
        int[] planShape = plan.shape();
        int[] planStrides = plan.strides();
        planShape[1] = 99;
        planStrides[1] = 99;

        assertArrayEquals(new int[]{2, 3}, plan.shape());
        assertArrayEquals(new int[]{3, 1}, plan.strides());
    }

    @Test
    void rejectsInvalidMetadata() {
        assertThrows(NullPointerException.class, () -> new Cpu1StorageAccessPlan(
                Cpu1StorageAccessKind.DENSE_CONTIGUOUS,
                null,
                new int[]{1},
                0,
                1L,
                null
        ));
        assertThrows(IllegalArgumentException.class, () -> new Cpu1StorageAccessPlan(
                Cpu1StorageAccessKind.DENSE_CONTIGUOUS,
                new int[]{2, 3},
                new int[]{1},
                0,
                6L,
                null
        ));
        assertThrows(IllegalArgumentException.class, () -> new Cpu1StorageAccessPlan(
                Cpu1StorageAccessKind.UNSUPPORTED,
                new int[]{2},
                new int[]{1},
                0,
                2L,
                null
        ));
    }

    private static CompiledTensorDescriptor descriptorFor(Tensor tensor) {
        List<CompiledNode> nodes = CompiledNode.snapshot(tensor.topologicalSort(), BackendIntentPlan.empty());
        return CompiledTensorDescriptorBuilder.build(nodes).byNodeId(nodes.getLast().id());
    }

    private static CompiledNode nodeFor(Tensor tensor) {
        List<CompiledNode> nodes = CompiledNode.snapshot(tensor.topologicalSort(), BackendIntentPlan.empty());
        return nodes.getLast();
    }
}

package graph.compile.descriptor;

import graph.CompiledNode;
import graph.execution.state.ExecutionState;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CompiledTensorDescriptorIndexTest {
    @Test
    void describesDenseContiguousLeafTensor() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "input", DataType.FLOAT32);

        CompiledTensorDescriptor descriptor = descriptorFor(input, 0);

        assertEquals(0, descriptor.nodeId());
        assertNull(descriptor.opType());
        assertEquals(DataType.FLOAT32, descriptor.dataType());
        assertArrayEquals(new int[]{2, 3}, descriptor.shape());
        assertEquals(2, descriptor.rank());
        assertArrayEquals(new int[]{3, 1}, descriptor.strides());
        assertEquals(0, descriptor.storageOffset());
        assertEquals(6L, descriptor.logicalElementCount());
        assertEquals(6L, descriptor.physicalSpan());
        assertEquals(24L, descriptor.logicalByteLength());
        assertEquals(24L, descriptor.physicalByteSpan());
        assertEquals(LayoutClass.DENSE_CONTIGUOUS, descriptor.layoutClass());
        assertTrue(descriptor.contiguous());
        assertTrue(descriptor.dense());
        assertFalse(descriptor.hasStorageOffset());
        assertFalse(descriptor.hasZeroStride());
        assertFalse(descriptor.broadcastView());
        assertTrue(descriptor.leaf());
        assertFalse(descriptor.backwardNode());
        assertFalse(descriptor.requiresGrad());
        assertEquals(List.of(), descriptor.inputIds());
    }

    @Test
    void classifiesStorageOffsetStridedAndBroadcastViews() {
        Tensor base = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "base", DataType.FLOAT32);
        Tensor selected = base.select(0, 1);
        Tensor permuted = base.permute(1, 0);
        Tensor expanded = new Tensor(new float[]{1f, 2f, 3f}, new int[]{1, 3}, null, "bias", DataType.FLOAT32)
                .expand(2, 3);

        CompiledTensorDescriptor selectedDescriptor = descriptorFor(selected, 1);
        CompiledTensorDescriptor permutedDescriptor = descriptorFor(permuted, 1);
        CompiledTensorDescriptor expandedDescriptor = descriptorFor(expanded, 1);

        assertArrayEquals(new int[]{3}, selectedDescriptor.shape());
        assertArrayEquals(new int[]{1}, selectedDescriptor.strides());
        assertEquals(3, selectedDescriptor.storageOffset());
        assertEquals(3L, selectedDescriptor.logicalElementCount());
        assertEquals(6L, selectedDescriptor.physicalSpan());
        assertEquals(LayoutClass.DENSE_WITH_OFFSET, selectedDescriptor.layoutClass());
        assertTrue(selectedDescriptor.dense());
        assertTrue(selectedDescriptor.hasStorageOffset());

        assertArrayEquals(new int[]{3, 2}, permutedDescriptor.shape());
        assertArrayEquals(new int[]{1, 3}, permutedDescriptor.strides());
        assertEquals(6L, permutedDescriptor.logicalElementCount());
        assertEquals(6L, permutedDescriptor.physicalSpan());
        assertEquals(LayoutClass.STRIDED_VIEW, permutedDescriptor.layoutClass());
        assertFalse(permutedDescriptor.dense());

        assertArrayEquals(new int[]{2, 3}, expandedDescriptor.shape());
        assertArrayEquals(new int[]{0, 1}, expandedDescriptor.strides());
        assertEquals(6L, expandedDescriptor.logicalElementCount());
        assertEquals(3L, expandedDescriptor.physicalSpan());
        assertEquals(LayoutClass.BROADCAST_ZERO_STRIDE, expandedDescriptor.layoutClass());
        assertFalse(expandedDescriptor.dense());
        assertTrue(expandedDescriptor.hasZeroStride());
        assertTrue(expandedDescriptor.broadcastView());
    }

    @Test
    void preservesDTypeRequiresGradAndInputLookupFromSnapshot() {
        Tensor left = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "left", DataType.BFLOAT16);
        Tensor right = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{2, 2}, null, "right", DataType.BFLOAT16);
        left.setRequiresGrad(true);
        Tensor out = left.add(right).relu();
        List<CompiledNode> nodes = CompiledNode.snapshot(out.topologicalSort());

        CompiledTensorDescriptorIndex index = CompiledTensorDescriptorBuilder.build(nodes);
        CompiledTensorDescriptor add = index.byNodeId(2);
        CompiledTensorDescriptor relu = index.byNodeId(3);

        assertEquals(DataType.BFLOAT16, add.dataType());
        assertTrue(add.requiresGrad());
        assertFalse(index.byNodeId(1).requiresGrad());
        assertEquals(List.of(0, 1), add.inputIds());
        assertEquals(index.byNodeId(0), index.input(2, 0));
        assertEquals(index.byNodeId(1), index.input(2, 1));
        assertEquals(List.of(add), index.inputs(3));
        assertThrows(IllegalArgumentException.class, () -> index.byNodeId(99));
        assertThrows(IllegalArgumentException.class, () -> index.input(2, 2));
    }

    @Test
    void descriptorArraysAreDefensiveCopies() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "input", DataType.FLOAT32);
        CompiledTensorDescriptor descriptor = descriptorFor(input, 0);

        int[] shape = descriptor.shape();
        int[] strides = descriptor.strides();
        shape[0] = 99;
        strides[0] = 99;

        assertArrayEquals(new int[]{2, 2}, descriptor.shape());
        assertArrayEquals(new int[]{2, 1}, descriptor.strides());
    }

    @Test
    void executionStateUsesCompiledRequiresGradSnapshotNotMutableSemanticTensor() {
        Tensor input = new Tensor(new float[]{1f, -2f}, new int[]{2}, null, "input", DataType.FLOAT32);
        input.setRequiresGrad(true);
        Tensor out = input.relu();
        List<CompiledNode> nodes = CompiledNode.snapshot(out.topologicalSort());
        CompiledTensorDescriptorIndex index = CompiledTensorDescriptorBuilder.build(nodes);

        input.setRequiresGrad(false);
        out.setRequiresGrad(false);

        ExecutionState state = ExecutionState.create(nodes, index, Map.of(), nodes.getLast().id());

        assertTrue(index.byNodeId(0).requiresGrad());
        assertTrue(index.byNodeId(1).requiresGrad());
        assertTrue(state.runtimeTensorForNodeId(0).getRequiresGrad());
        assertTrue(state.runtimeTensorForNodeId(1).getRequiresGrad());
    }

    private static CompiledTensorDescriptor descriptorFor(Tensor tensor, int nodeId) {
        return CompiledTensorDescriptorBuilder.build(CompiledNode.snapshot(tensor.topologicalSort())).byNodeId(nodeId);
    }
}

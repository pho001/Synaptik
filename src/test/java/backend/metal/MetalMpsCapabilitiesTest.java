package backend.metal;

import graph.CompiledNode;
import org.junit.jupiter.api.Test;
import operations.Operation;
import tensor.Tensor;
import tensor.DataType;
import tensor.internal.TensorPrimitiveBuilder;
import graph.compile.intent.BackendIntentPlan;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MetalMpsCapabilitiesTest {
    @Test
    void exposesConservativeDTypeCapabilities() {
        assertTrue(MetalMpsCapabilities.supportsComputeDType(DataType.FLOAT32));
        assertTrue(MetalMpsCapabilities.supportsOutputDType(DataType.FLOAT32));
        assertTrue(MetalMpsCapabilities.supportsExternalInputDType(DataType.FLOAT32));
        assertTrue(MetalMpsCapabilities.supportsExternalInputDType(DataType.BOOL));
        assertTrue(MetalMpsCapabilities.supportsExternalInputDType(DataType.BFLOAT16));
        assertTrue(MetalMpsCapabilities.supportsExternalInputDType(DataType.INT32));
        assertTrue(MetalMpsCapabilities.supportsComputeDType(DataType.BFLOAT16));
        assertTrue(MetalMpsCapabilities.supportsOutputDType(DataType.BFLOAT16));

        assertTrue(MetalMpsCapabilities.supportsComputeDType(DataType.BOOL));
        assertTrue(MetalMpsCapabilities.supportsOutputDType(DataType.BOOL));
        assertTrue(MetalMpsCapabilities.supportsOutputDType(DataType.INT64));

        assertTrue(MetalMpsCapabilities.operationDecision(Operation.OpType.MATMUL, DataType.BFLOAT16).supported());
        assertTrue(MetalMpsCapabilities.operationDecision(Operation.OpType.CONV2D, DataType.BFLOAT16).supported());
        assertTrue(MetalMpsCapabilities.operationDecision(Operation.OpType.CROSS_ENTROPY_LOSS_INDICES, DataType.BFLOAT16).supported());
        assertTrue(MetalMpsCapabilities.operationDecision(Operation.OpType.GATHER_ND, DataType.BFLOAT16).supported());
        assertTrue(MetalMpsCapabilities.operationDecision(Operation.OpType.SCATTER_ADD, DataType.BFLOAT16).supported());
        assertTrue(MetalMpsCapabilities.operationDecision(Operation.OpType.GE, DataType.BOOL).supported());
        assertTrue(MetalMpsCapabilities.operationDecision(Operation.OpType.LOGICAL_AND, DataType.BOOL).supported());
        assertTrue(MetalMpsCapabilities.operationDecision(Operation.OpType.REDUCE_ANY, DataType.BOOL).supported());
        assertFalse(MetalMpsCapabilities.operationDecision(Operation.OpType.ARGMAX, DataType.INT32).supported());
        assertTrue(MetalMpsCapabilities.operationDecision(Operation.OpType.ARGMAX, DataType.INT64).supported());
        assertFalse(MetalMpsCapabilities.operationDecision(Operation.OpType.CUMSUM, DataType.INT32).supported());
        assertTrue(MetalMpsCapabilities.operationDecision(Operation.OpType.EXPAND, DataType.BOOL).supported());
        assertTrue(MetalMpsCapabilities.operationDecision(Operation.OpType.PERMUTE, DataType.BOOL).supported());
        assertTrue(MetalMpsCapabilities.operationDecision(Operation.OpType.RESHAPE, DataType.BOOL).supported());
        assertTrue(MetalMpsCapabilities.operationDecision(Operation.OpType.SQUEEZE, DataType.BOOL).supported());
        assertFalse(MetalMpsCapabilities.operationDecision(Operation.OpType.MATMUL, DataType.BOOL).supported());
        assertTrue(MetalMpsCapabilities.unsupportedDTypeMessage(DataType.INT32)
                .contains("FLOAT32/BFLOAT16 compute/output tensors for supported floating operation families, scoped BOOL outputs, BOOL predicate inputs, INT32 index inputs, and scoped INT64 ARGMAX index outputs"));
    }

    @Test
    void exposesRoleSpecificDTypeDecisionsForEveryPublicDType() {
        for (DataType dtype : DataType.values()) {
            assertTrue(MetalMpsCapabilities.storageDecision(dtype).storageRepresentable());
            assertEquals(dtype == DataType.FLOAT32 || dtype == DataType.BFLOAT16 || dtype == DataType.BOOL || dtype == DataType.INT64, MetalMpsCapabilities.computeDecision(dtype).supported());
            assertEquals(dtype == DataType.FLOAT32 || dtype == DataType.BFLOAT16 || dtype == DataType.BOOL || dtype == DataType.INT64, MetalMpsCapabilities.outputDecision(dtype).supported());
        }

        assertTrue(MetalMpsCapabilities.externalInputDecision(DataType.BOOL).supported());
        assertEquals(
                MetalDTypeReasonCode.SUPPORTED_PREDICATE_INPUT_ONLY,
                MetalMpsCapabilities.externalInputDecision(DataType.BOOL).reasonCode()
        );
        assertTrue(MetalMpsCapabilities.externalInputDecision(DataType.BFLOAT16).supported());
        assertTrue(MetalMpsCapabilities.externalInputDecision(DataType.INT32).supported());
        assertEquals(
                MetalDTypeReasonCode.SUPPORTED_STORAGE_ONLY,
                MetalMpsCapabilities.externalInputDecision(DataType.INT32).reasonCode()
        );
        assertEquals(MetalDTypeReasonCode.FLOAT64_UNSUPPORTED, MetalMpsCapabilities.computeDecision(DataType.FLOAT64).reasonCode());
        assertEquals(MetalDTypeReasonCode.FLOAT64_UNSUPPORTED, MetalMpsCapabilities.outputDecision(DataType.FLOAT64).reasonCode());
    }

    @Test
    void externalInputRoleKeepsBoolLimitedToWherePredicate() {
        Tensor mask = new Tensor(new byte[]{1, 0}, new int[]{2}, null, "mask", DataType.BOOL);
        Tensor left = new Tensor(new float[]{1.0f, 2.0f}, new int[]{2}, null, "left", DataType.FLOAT32);
        Tensor right = new Tensor(new float[]{3.0f, 4.0f}, new int[]{2}, null, "right", DataType.FLOAT32);
        Tensor where = Tensor.where(mask, left, right);
        List<CompiledNode> nodes = CompiledNode.snapshot(List.of(mask, left, right, where), BackendIntentPlan.empty());

        assertTrue(MetalMpsCapabilities.externalInputRoleDecision(nodes.get(0), nodes.get(3), 0).supported());
        assertTrue(MetalMpsCapabilities.externalInputRoleDecision(nodes.get(1), nodes.get(3), 1).supported());
        assertTrue(MetalMpsCapabilities.externalInputRoleDecision(nodes.get(2), nodes.get(3), 2).supported());
        assertFalse(MetalMpsCapabilities.externalInputRoleDecision(nodes.get(0), nodes.get(3), 1).supported());
        assertEquals(
                MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                MetalMpsCapabilities.externalInputRoleDecision(nodes.get(0), nodes.get(3), 1).reasonCode()
        );
    }

    @Test
    void externalInputRoleAllowsOnlyNumericDataForBoolCompareInputs() {
        Tensor left = new Tensor(new float[]{1.0f, 2.0f}, new int[]{2}, null, "left", DataType.FLOAT32);
        Tensor right = new Tensor(new float[]{3.0f, 4.0f}, new int[]{2}, null, "right", DataType.FLOAT32);
        Tensor compare = left.greaterOrEqual(right);
        List<CompiledNode> nodes = CompiledNode.snapshot(List.of(left, right, compare), BackendIntentPlan.empty());

        assertTrue(MetalMpsCapabilities.externalInputRoleDecision(nodes.get(0), nodes.get(2), 0).supported());
        assertTrue(MetalMpsCapabilities.externalInputRoleDecision(nodes.get(1), nodes.get(2), 1).supported());
        assertFalse(MetalMpsCapabilities.externalInputRoleDecision(nodes.get(0), nodes.get(2), 2).supported());

        Tensor mask = new Tensor(new byte[]{1, 0}, new int[]{2}, null, "mask", DataType.BOOL);
        CompiledNode maskNode = CompiledNode.snapshot(List.of(mask), BackendIntentPlan.empty()).getFirst();
        assertFalse(MetalMpsCapabilities.externalInputRoleDecision(maskNode, nodes.get(2), 0).supported());
    }

    @Test
    void externalInputRoleAllowsBoolForLogicalAndReductionInputs() {
        Tensor left = new Tensor(new byte[]{1, 0}, new int[]{2}, null, "left", DataType.BOOL);
        Tensor right = new Tensor(new byte[]{1, 1}, new int[]{2}, null, "right", DataType.BOOL);
        Tensor logical = left.logicalAnd(right);
        Tensor reduced = logical.any(0, true);
        List<CompiledNode> logicalNodes = CompiledNode.snapshot(List.of(left, right, logical), BackendIntentPlan.empty());
        List<CompiledNode> reductionNodes = CompiledNode.snapshot(reduced.topologicalSort(), BackendIntentPlan.empty());

        assertTrue(MetalMpsCapabilities.externalInputRoleDecision(logicalNodes.get(0), logicalNodes.get(2), 0).supported());
        assertTrue(MetalMpsCapabilities.externalInputRoleDecision(logicalNodes.get(1), logicalNodes.get(2), 1).supported());
        assertFalse(MetalMpsCapabilities.externalInputRoleDecision(logicalNodes.get(0), logicalNodes.get(2), 2).supported());

        CompiledNode reductionInput = reductionNodes.get(reductionNodes.size() - 2);
        CompiledNode reduction = reductionNodes.getLast();
        assertTrue(MetalMpsCapabilities.externalInputRoleDecision(reductionInput, reduction, 0).supported());
        assertFalse(MetalMpsCapabilities.externalInputRoleDecision(reductionInput, reduction, 1).supported());
    }

    @Test
    void externalInputRoleAllowsBoolForDTypePreservingLayoutInputs() {
        Tensor mask = new Tensor(new byte[]{1, 0}, new int[]{1, 2}, null, "mask", DataType.BOOL);
        Tensor expanded = mask.expand(3, 2);
        List<CompiledNode> nodes = CompiledNode.snapshot(List.of(mask, expanded), BackendIntentPlan.empty());

        assertTrue(MetalMpsCapabilities.externalInputRoleDecision(nodes.get(0), nodes.get(1), 0).supported());
        assertFalse(MetalMpsCapabilities.externalInputRoleDecision(nodes.get(0), nodes.get(1), 1).supported());

        Tensor values = new Tensor(new float[]{1.0f, 2.0f}, new int[]{1, 2}, null, "values", DataType.FLOAT32);
        Tensor floatExpanded = values.expand(3, 2);
        List<CompiledNode> floatNodes = CompiledNode.snapshot(List.of(values, floatExpanded), BackendIntentPlan.empty());
        assertTrue(MetalMpsCapabilities.externalInputRoleDecision(floatNodes.get(0), floatNodes.get(1), 0).supported());

        Tensor intTensor = new Tensor(new int[]{1, 0}, new int[]{1, 2}, null, "indices", DataType.INT32);
        CompiledNode intNode = CompiledNode.snapshot(List.of(intTensor), BackendIntentPlan.empty()).getFirst();
        assertFalse(MetalMpsCapabilities.externalInputRoleDecision(intNode, nodes.get(1), 0).supported());
    }

    @Test
    void externalInputRoleAllowsInt32OnlyForIndexInputs() {
        Tensor values = new Tensor(new float[]{1.0f, 2.0f, 3.0f, 4.0f}, new int[]{2, 2}, null, "values", DataType.FLOAT32);
        Tensor indices = new Tensor(new int[]{1, 0}, new int[]{2}, null, "indices", DataType.INT32);
        Tensor gathered = values.gather(indices, 1);
        Tensor gatherNd = values.gatherNd(new Tensor(new int[]{1, 0}, new int[]{1, 2}, null, "gatherNdIndices", DataType.INT32));
        List<CompiledNode> nodes = CompiledNode.snapshot(List.of(values, indices, gathered), BackendIntentPlan.empty());
        List<CompiledNode> gatherNdNodes = CompiledNode.snapshot(gatherNd.topologicalSort(), BackendIntentPlan.empty());

        assertTrue(MetalMpsCapabilities.externalInputRoleDecision(nodes.get(0), nodes.get(2), 0).supported());
        assertTrue(MetalMpsCapabilities.externalInputRoleDecision(nodes.get(1), nodes.get(2), 1).supported());
        assertFalse(MetalMpsCapabilities.externalInputRoleDecision(nodes.get(1), nodes.get(2), 0).supported());
        assertTrue(MetalMpsCapabilities.externalInputRoleDecision(gatherNdNodes.get(0), gatherNdNodes.get(2), 0).supported());
        assertTrue(MetalMpsCapabilities.externalInputRoleDecision(gatherNdNodes.get(1), gatherNdNodes.get(2), 1).supported());

        Tensor where = Tensor.where(
                new Tensor(new byte[]{1, 0}, new int[]{2}, null, "mask", DataType.BOOL),
                new Tensor(new float[]{1.0f, 2.0f}, new int[]{2}, null, "left", DataType.FLOAT32),
                new Tensor(new float[]{3.0f, 4.0f}, new int[]{2}, null, "right", DataType.FLOAT32)
        );
        CompiledNode intNode = CompiledNode.snapshot(List.of(indices), BackendIntentPlan.empty()).getFirst();
        CompiledNode whereNode = CompiledNode.snapshot(where.topologicalSort(), BackendIntentPlan.empty()).getLast();
        assertFalse(MetalMpsCapabilities.externalInputRoleDecision(intNode, whereNode, 0).supported());
    }

    @Test
    void externalInputRoleUsesCastPairPolicyForCastInputs() {
        Tensor f32 = new Tensor(new float[]{1.0f, 2.0f}, new int[]{2}, null, "f32CastInput", DataType.FLOAT32);
        Tensor bf16Cast = f32.cast(DataType.BFLOAT16);
        List<CompiledNode> f32ToBf16 = CompiledNode.snapshot(List.of(f32, bf16Cast), BackendIntentPlan.empty());

        assertTrue(MetalMpsCapabilities.externalInputRoleDecision(f32ToBf16.get(0), f32ToBf16.get(1), 0).supported());
        assertFalse(MetalMpsCapabilities.externalInputRoleDecision(f32ToBf16.get(0), f32ToBf16.get(1), 1).supported());

        Tensor bf16 = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "bf16CastInput", DataType.BFLOAT16);
        Tensor f32Cast = bf16.cast(DataType.FLOAT32);
        List<CompiledNode> bf16ToF32 = CompiledNode.snapshot(List.of(bf16, f32Cast), BackendIntentPlan.empty());
        assertTrue(MetalMpsCapabilities.externalInputRoleDecision(bf16ToF32.get(0), bf16ToF32.get(1), 0).supported());

        Tensor intCast = f32.cast(DataType.INT32);
        List<CompiledNode> f32ToInt = CompiledNode.snapshot(List.of(f32, intCast), BackendIntentPlan.empty());
        assertFalse(MetalMpsCapabilities.externalInputRoleDecision(f32ToInt.get(0), f32ToInt.get(1), 0).supported());
    }

    @Test
    void externalInputRoleAllowsInt32ForIndexTargetCrossEntropy() {
        Tensor logits = new Tensor(new float[]{1f, 2f, 3f, 1f, 0f, -1f}, new int[]{2, 3}, null, "logits", DataType.FLOAT32);
        Tensor targets = new Tensor(new int[]{2, 0}, new int[]{2}, null, "targets", DataType.INT32);
        Tensor loss = logits.crossEntropyLossFromIndices(targets, 1);
        List<CompiledNode> lossNodes = CompiledNode.snapshot(List.of(logits, targets, loss), BackendIntentPlan.empty());

        assertTrue(MetalMpsCapabilities.externalInputRoleDecision(lossNodes.get(0), lossNodes.get(2), 0).supported());
        assertTrue(MetalMpsCapabilities.externalInputRoleDecision(lossNodes.get(1), lossNodes.get(2), 1).supported());
        assertFalse(MetalMpsCapabilities.externalInputRoleDecision(lossNodes.get(1), lossNodes.get(2), 0).supported());

        Tensor scale = new Tensor(new float[]{0.5f, 0.5f}, new int[]{2}, null, "scale", DataType.FLOAT32);
        Tensor grad = TensorPrimitiveBuilder.ternaryNoGrad(
                logits,
                targets,
                scale,
                new int[]{2, 3},
                new operations.loss.crossEntropyLossIndicesGrad(1),
                "ceGrad",
                DataType.FLOAT32
        );
        List<CompiledNode> gradNodes = CompiledNode.snapshot(List.of(logits, targets, scale, grad), BackendIntentPlan.empty());
        assertTrue(MetalMpsCapabilities.externalInputRoleDecision(gradNodes.get(0), gradNodes.get(3), 0).supported());
        assertTrue(MetalMpsCapabilities.externalInputRoleDecision(gradNodes.get(1), gradNodes.get(3), 1).supported());
        assertTrue(MetalMpsCapabilities.externalInputRoleDecision(gradNodes.get(2), gradNodes.get(3), 2).supported());

        Tensor bf16Logits = new Tensor(new double[]{1, 2, 3, 1, 0, -1}, new int[]{2, 3}, null, "bf16Logits", DataType.BFLOAT16);
        Tensor bf16Loss = bf16Logits.crossEntropyLossFromIndices(targets, 1);
        List<CompiledNode> bf16LossNodes = CompiledNode.snapshot(List.of(bf16Logits, targets, bf16Loss), BackendIntentPlan.empty());
        assertTrue(MetalMpsCapabilities.externalInputRoleDecision(bf16LossNodes.get(0), bf16LossNodes.get(2), 0).supported());
        assertTrue(MetalMpsCapabilities.externalInputRoleDecision(bf16LossNodes.get(1), bf16LossNodes.get(2), 1).supported());

        Tensor bf16Scale = new Tensor(new double[]{0.5, 0.5}, new int[]{2}, null, "bf16Scale", DataType.BFLOAT16);
        Tensor bf16Grad = TensorPrimitiveBuilder.ternaryNoGrad(
                bf16Logits,
                targets,
                bf16Scale,
                new int[]{2, 3},
                new operations.loss.crossEntropyLossIndicesGrad(1),
                "bf16CeGrad",
                DataType.BFLOAT16
        );
        List<CompiledNode> bf16GradNodes = CompiledNode.snapshot(List.of(bf16Logits, targets, bf16Scale, bf16Grad), BackendIntentPlan.empty());
        assertTrue(MetalMpsCapabilities.externalInputRoleDecision(bf16GradNodes.get(0), bf16GradNodes.get(3), 0).supported());
        assertTrue(MetalMpsCapabilities.externalInputRoleDecision(bf16GradNodes.get(1), bf16GradNodes.get(3), 1).supported());
        assertTrue(MetalMpsCapabilities.externalInputRoleDecision(bf16GradNodes.get(2), bf16GradNodes.get(3), 2).supported());
    }

    @Test
    void descriptorAbiCodesCoverAllPublicDTypesButExecutionAbiStaysNarrow() {
        assertEquals(1, MetalMpsCapabilities.abiDescriptorDataTypeCode(DataType.FLOAT32));
        assertEquals(2, MetalMpsCapabilities.abiDescriptorDataTypeCode(DataType.BOOL));
        assertEquals(3, MetalMpsCapabilities.abiDescriptorDataTypeCode(DataType.BFLOAT16));
        assertEquals(4, MetalMpsCapabilities.abiDescriptorDataTypeCode(DataType.INT32));
        assertEquals(5, MetalMpsCapabilities.abiDescriptorDataTypeCode(DataType.FLOAT64));
        assertEquals(6, MetalMpsCapabilities.abiDescriptorDataTypeCode(DataType.INT64));

        assertEquals(1, MetalMpsCapabilities.abiDataTypeCode(DataType.FLOAT32));
        assertEquals(2, MetalMpsCapabilities.abiDataTypeCode(DataType.BOOL));
        assertTrue(MetalMpsCapabilities.outputDecision(DataType.BOOL).supported());
    }
}

package backend.metal;

import graph.CompiledNode;
import org.junit.jupiter.api.Test;
import tensor.Tensor;
import tensor.DataType;

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

        assertFalse(MetalMpsCapabilities.supportsComputeDType(DataType.BOOL));
        assertFalse(MetalMpsCapabilities.supportsOutputDType(DataType.BOOL));
        assertFalse(MetalMpsCapabilities.supportsComputeDType(DataType.BFLOAT16));
        assertFalse(MetalMpsCapabilities.supportsOutputDType(DataType.INT32));

        assertTrue(MetalMpsCapabilities.unsupportedDTypeMessage(DataType.BFLOAT16)
                .contains("FLOAT32 compute/output tensors and BOOL only for predicate inputs"));
    }

    @Test
    void exposesRoleSpecificDTypeDecisionsForEveryPublicDType() {
        for (DataType dtype : DataType.values()) {
            assertTrue(MetalMpsCapabilities.storageDecision(dtype).storageRepresentable());
            assertEquals(dtype == DataType.FLOAT32, MetalMpsCapabilities.computeDecision(dtype).supported());
            assertEquals(dtype == DataType.FLOAT32, MetalMpsCapabilities.outputDecision(dtype).supported());
        }

        assertTrue(MetalMpsCapabilities.externalInputDecision(DataType.BOOL).supported());
        assertEquals(
                MetalDTypeReasonCode.SUPPORTED_PREDICATE_INPUT_ONLY,
                MetalMpsCapabilities.externalInputDecision(DataType.BOOL).reasonCode()
        );
        assertFalse(MetalMpsCapabilities.externalInputDecision(DataType.BFLOAT16).supported());
        assertFalse(MetalMpsCapabilities.externalInputDecision(DataType.INT32).supported());
        assertEquals(MetalDTypeReasonCode.FLOAT64_UNSUPPORTED, MetalMpsCapabilities.computeDecision(DataType.FLOAT64).reasonCode());
        assertEquals(MetalDTypeReasonCode.FLOAT64_UNSUPPORTED, MetalMpsCapabilities.outputDecision(DataType.FLOAT64).reasonCode());
    }

    @Test
    void externalInputRoleKeepsBoolLimitedToWherePredicate() {
        Tensor mask = new Tensor(new byte[]{1, 0}, new int[]{2}, null, "mask", DataType.BOOL);
        Tensor left = new Tensor(new float[]{1.0f, 2.0f}, new int[]{2}, null, "left", DataType.FLOAT32);
        Tensor right = new Tensor(new float[]{3.0f, 4.0f}, new int[]{2}, null, "right", DataType.FLOAT32);
        Tensor where = Tensor.where(mask, left, right);
        List<CompiledNode> nodes = CompiledNode.snapshot(List.of(mask, left, right, where));

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
    void descriptorAbiCodesCoverAllPublicDTypesButExecutionAbiStaysNarrow() {
        assertEquals(1, MetalMpsCapabilities.abiDescriptorDataTypeCode(DataType.FLOAT32));
        assertEquals(2, MetalMpsCapabilities.abiDescriptorDataTypeCode(DataType.BOOL));
        assertEquals(3, MetalMpsCapabilities.abiDescriptorDataTypeCode(DataType.BFLOAT16));
        assertEquals(4, MetalMpsCapabilities.abiDescriptorDataTypeCode(DataType.INT32));
        assertEquals(5, MetalMpsCapabilities.abiDescriptorDataTypeCode(DataType.FLOAT64));

        assertEquals(1, MetalMpsCapabilities.abiDataTypeCode(DataType.FLOAT32));
        assertEquals(2, MetalMpsCapabilities.abiDataTypeCode(DataType.BOOL));
        assertFalse(MetalMpsCapabilities.outputDecision(DataType.BOOL).supported());
    }
}

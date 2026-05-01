package backend.metal;

import org.junit.jupiter.api.Test;
import tensor.DataType;

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
}

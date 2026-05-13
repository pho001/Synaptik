package backend.metal;

import org.junit.jupiter.api.Test;
import tensor.DataType;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MetalCastPolicyTest {
    @Test
    void supportsOnlyIdentityAndFloatBfloat16Pairs() {
        assertSupported(DataType.FLOAT32, DataType.FLOAT32);
        assertSupported(DataType.BFLOAT16, DataType.BFLOAT16);
        assertSupported(DataType.BOOL, DataType.BOOL);
        assertSupported(DataType.INT32, DataType.INT32);
        assertSupported(DataType.FLOAT32, DataType.BFLOAT16);
        assertSupported(DataType.BFLOAT16, DataType.FLOAT32);

        assertUnsupported(DataType.FLOAT32, DataType.INT32, MetalDTypeReasonCode.UNSUPPORTED_CAST_PAIR);
        assertUnsupported(DataType.INT32, DataType.FLOAT32, MetalDTypeReasonCode.UNSUPPORTED_CAST_PAIR);
        assertUnsupported(DataType.FLOAT32, DataType.BOOL, MetalDTypeReasonCode.UNSUPPORTED_CAST_PAIR);
        assertUnsupported(DataType.BOOL, DataType.FLOAT32, MetalDTypeReasonCode.UNSUPPORTED_CAST_PAIR);
        assertUnsupported(DataType.FLOAT64, DataType.FLOAT32, MetalDTypeReasonCode.FLOAT64_UNSUPPORTED);
        assertUnsupported(DataType.FLOAT32, DataType.FLOAT64, MetalDTypeReasonCode.FLOAT64_UNSUPPORTED);
    }

    private static void assertSupported(DataType source, DataType target) {
        MetalCastPolicy.Decision decision = MetalCastPolicy.decide(source, target);
        assertTrue(decision.supported(), source + " -> " + target);
        assertEquals(MetalDTypeReasonCode.SUPPORTED, decision.reasonCode());
    }

    private static void assertUnsupported(DataType source, DataType target, MetalDTypeReasonCode reasonCode) {
        MetalCastPolicy.Decision decision = MetalCastPolicy.decide(source, target);
        assertFalse(decision.supported(), source + " -> " + target);
        assertEquals(reasonCode, decision.reasonCode());
    }
}

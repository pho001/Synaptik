import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.calibration.family.CalibrationFamilyId;
import tuning.calibration.family.CalibrationFamilyRegistry;

import java.util.HashSet;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class CalibrationFamilyRegistryTest {
    @Test
    void standardSuiteHasDeterministicProductionOrder() {
        assertEquals(
                java.util.List.of(
                        CalibrationFamilyId.SCHEDULER,
                        CalibrationFamilyId.MATMUL,
                        CalibrationFamilyId.ATTENTION_MATMUL,
                        CalibrationFamilyId.CONV2D_GEMM_DISPATCH,
                        CalibrationFamilyId.ELEMENTWISE_DISPATCH,
                        CalibrationFamilyId.FUSED_DISPATCH,
                        CalibrationFamilyId.FUSED_CHEAP_CONTIGUOUS_WIDTH,
                        CalibrationFamilyId.FUSED_CHEAP_STRIDED_WIDTH,
                        CalibrationFamilyId.FUSED_NON_CHEAP_CONTIGUOUS_WIDTH,
                        CalibrationFamilyId.FUSED_NON_CHEAP_STRIDED_WIDTH,
                        CalibrationFamilyId.REDUCTION,
                        CalibrationFamilyId.ATTENTION_THRESHOLDS,
                        CalibrationFamilyId.MATERIALIZATION
                ),
                CalibrationFamilyRegistry.standardSuite()
        );
    }

    @Test
    void removedFamilyIdsAreNotParseable() {
        assertThrows(IllegalArgumentException.class, () -> CalibrationFamilyRegistry.parse("fused-thresholds"));
        assertThrows(IllegalArgumentException.class, () -> CalibrationFamilyRegistry.parse("fused-arithmetic"));
        assertThrows(IllegalArgumentException.class, () -> CalibrationFamilyRegistry.parse("conv2d-f32"));
        assertThrows(IllegalArgumentException.class, () -> CalibrationFamilyRegistry.parse("numerics"));
        assertThrows(IllegalArgumentException.class, () -> CalibrationFamilyRegistry.parse("accelerator-metal"));
    }

    @Test
    void standardFamiliesDoNotOverlapOwnedKnobs() {
        HashSet<String> seen = new HashSet<>();
        for (CalibrationFamilyId family : CalibrationFamilyRegistry.standardSuite()) {
            for (String knob : CalibrationFamilyRegistry.spec(family).ownedKnobs()) {
                assertTrue(seen.add(knob), () -> "Duplicate calibration knob ownership: " + knob);
            }
        }
    }

    @Test
    void metalSelectionIsOptInAndF32Only() {
        assertFalse(CalibrationFamilyRegistry.standardSuite().contains(CalibrationFamilyId.METAL_SELECTION));
        assertTrue(CalibrationFamilyRegistry.fullSuite(true).contains(CalibrationFamilyId.METAL_SELECTION));
        assertTrue(CalibrationFamilyRegistry.supportsDType(CalibrationFamilyId.METAL_SELECTION, DataType.FLOAT32));
        assertFalse(CalibrationFamilyRegistry.supportsDType(CalibrationFamilyId.METAL_SELECTION, DataType.FLOAT64));
        assertFalse(CalibrationFamilyRegistry.supportsDType(CalibrationFamilyId.METAL_SELECTION, DataType.BFLOAT16));
    }
}

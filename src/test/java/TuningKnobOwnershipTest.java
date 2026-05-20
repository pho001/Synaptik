import org.junit.jupiter.api.Test;
import tuning.ownership.TuningKnobOwner;
import tuning.ownership.TuningKnobOwnership;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TuningKnobOwnershipTest {
    @Test
    void acceleratorBufferModeIsPlatformDtypeOwned() {
        assertEquals(
                TuningKnobOwner.PLATFORM_DTYPE,
                TuningKnobOwnership.ownerOf("runtime.accelerator.metal.buffer.bindingMode")
        );
    }

    @Test
    void metalSelectionIsPlatformDtypeOwned() {
        assertEquals(
                TuningKnobOwner.PLATFORM_DTYPE,
                TuningKnobOwnership.ownerOf("runtime.accelerator.metal.minimumEstimatedWork")
        );
    }

    @Test
    void transferCostPresetIsGraphWorkloadOwned() {
        assertEquals(
                TuningKnobOwner.GRAPH_WORKLOAD,
                TuningKnobOwnership.ownerOf("compile.backendPlanning.cost.transferCostPreset")
        );
    }

    @Test
    void unknownKnobIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> TuningKnobOwnership.ownerOf("unknown.knob"));
    }

    @Test
    void graphValidationRejectsPlatformKnobs() {
        IllegalStateException thrown = assertThrows(IllegalStateException.class, () ->
                TuningKnobOwnership.validateGraphWorkload(
                        Map.of("runtime.accelerator.metal.minimumEstimatedWork", "4096"),
                        "graph-test"
                ));
        assertTrue(thrown.getMessage().contains("runtime.accelerator.metal.minimumEstimatedWork"));
    }

    @Test
    void platformValidationRejectsGraphKnobs() {
        IllegalStateException thrown = assertThrows(IllegalStateException.class, () ->
                TuningKnobOwnership.validatePlatformDtype(
                        Map.of("compile.backendPlanning.cost.transferCostPreset", "MEASURED"),
                        "platform-test"
                ));
        assertTrue(thrown.getMessage().contains("compile.backendPlanning.cost.transferCostPreset"));
    }
}

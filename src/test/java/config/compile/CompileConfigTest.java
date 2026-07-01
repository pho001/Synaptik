package config.compile;

import config.optimizer.CpuPartitionConfig;
import config.optimizer.MemoryConfig;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CompileConfigTest {
    @Test
    void trainingDefaultsHonorExplicitAcceleratorIntentsAndKeepCompileLayersEnabled() {
        CompileConfig config = CompileConfig.training();

        assertTrue(config.semanticCanonicalization().enabled());
        assertTrue(config.graphOptimization().commonSubexpressionElimination());
        assertEquals(BackendDiscoveryMode.EXPLICIT, config.backendPlanning().discoveryMode());
        assertTrue(config.partitionExecution().enabled());
        assertTrue(config.memoryPlanning().enabled());
    }

    @Test
    void autoAcceleratorIsExplicitOptIn() {
        CompileConfig config = CompileConfig.trainingAutoAccelerator();

        assertEquals(BackendDiscoveryMode.AUTO, config.backendPlanning().discoveryMode());
        assertEquals(BackendPlanningFailurePolicy.OPTIONAL, config.backendPlanning().failurePolicy());
        assertTrue(config.backendPlanning().targets().contains(BackendTarget.GPU_METAL));
    }

    @Test
    void noGraphOptimizationOnlyDisablesGraphOptimization() {
        CompileConfig config = CompileConfig.noGraphOptimization();

        assertTrue(config.semanticCanonicalization().enabled());
        assertFalse(config.graphOptimization().algebraicRewrite());
        assertFalse(config.graphOptimization().commonSubexpressionElimination());
        assertEquals(BackendDiscoveryMode.EXPLICIT, config.backendPlanning().discoveryMode());
        assertTrue(config.partitionExecution().enabled());
        assertTrue(config.memoryPlanning().enabled());
    }

    @Test
    void backendPlanningValidationRejectsInvalidRequiredCpuOnlyMode() {
        assertThrows(IllegalArgumentException.class, () -> new BackendPlanningConfig(
                BackendDiscoveryMode.CPU_ONLY,
                BackendPlanningFailurePolicy.REQUIRE_ACCELERATOR_PARTITION,
                BackendPlanningRequirementScope.ANY_TARGET,
                java.util.Set.of(),
                PartitionOwnershipPlannerStrategy.ANCHOR,
                PartitionSearchConfig.defaults(),
                CpuPartitionConfig.defaults(),
                BackendPlanningCostConfig.conservative()
        ));
    }

    @Test
    void memoryPlanningDisabledUnlessRequiredKeepsMemoryPolicyAvailable() {
        MemoryPlanningConfig config = MemoryPlanningConfig.disabledUnlessRequired();

        assertFalse(config.enabled());
        assertEquals(MemoryConfig.defaults(), config.memory());
    }
}

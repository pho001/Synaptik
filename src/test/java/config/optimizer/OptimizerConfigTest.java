package config.optimizer;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class OptimizerConfigTest {
    @Test
    void trainingDefaultsIncludePartFuseAndMem() {
        assertEquals(
                List.of(OptimizerStage.AR, OptimizerStage.CSE, OptimizerStage.PART, OptimizerStage.FUSE, OptimizerStage.MEM),
                OptimizerConfig.trainingDefaults().stageOrder()
        );
    }

    @Test
    void rejectsFuseWithoutPartitionStage() {
        assertThrows(IllegalArgumentException.class, () -> new OptimizerConfig(
                List.of(OptimizerStage.AR, OptimizerStage.FUSE, OptimizerStage.MEM),
                RewriteConfig.defaults(),
                CseConfig.strictDefaults(),
                FuseConfig.inferenceDefaults(),
                MemoryConfig.defaults(),
                PartitionConfig.defaults()
        ));
    }

    @Test
    void rejectsStageOrderWhenPartitionRunsAfterFuse() {
        assertThrows(IllegalArgumentException.class, () -> new OptimizerConfig(
                List.of(OptimizerStage.AR, OptimizerStage.FUSE, OptimizerStage.PART, OptimizerStage.MEM),
                RewriteConfig.defaults(),
                CseConfig.strictDefaults(),
                FuseConfig.inferenceDefaults(),
                MemoryConfig.defaults(),
                PartitionConfig.defaults()
        ));
    }

    @Test
    void rejectsMemWithoutFuseStage() {
        assertThrows(IllegalArgumentException.class, () -> new OptimizerConfig(
                List.of(OptimizerStage.AR, OptimizerStage.CSE, OptimizerStage.PART, OptimizerStage.MEM),
                RewriteConfig.defaults(),
                CseConfig.strictDefaults(),
                FuseConfig.inferenceDefaults(),
                MemoryConfig.defaults(),
                PartitionConfig.defaults()
        ));
    }
}

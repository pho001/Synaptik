package synaptik.app;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TuningCliShapeTest {
    @Test
    void parseTuningCommandAcceptsTransformerShapePreset() {
        TuningCli.ParsedCommand command = TuningCli.parseCommand(new String[]{
                "autotune.run",
                "--dtype", "f32",
                "--workload", "transformer-block",
                "--shape", "large"
        });

        assertEquals(TuningCli.CommandKind.AUTOTUNE, command.kind());
        assertEquals(TuningCli.DTypeTarget.F32, command.tuning().dtype());
        assertEquals(TuningCli.WorkloadTarget.TRANSFORMER_BLOCK, command.tuning().workload());
        assertEquals(TuningCli.WorkloadShape.LARGE, command.tuning().shape());
    }

    @Test
    void nonMediumTransformerShapeGetsSeparatePersistenceNamespace() {
        assertEquals(
                "transformer_block_hot_path_large",
                TuningCli.WorkloadTarget.TRANSFORMER_BLOCK.namespace(TuningCli.WorkloadShape.LARGE)
        );
        assertEquals(
                "transformer_block_hot_path",
                TuningCli.WorkloadTarget.TRANSFORMER_BLOCK.namespace(TuningCli.WorkloadShape.MEDIUM)
        );
    }
}

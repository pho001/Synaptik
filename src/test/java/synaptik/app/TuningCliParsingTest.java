package synaptik.app;

import backend.runtime.ExecutionMode;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.autotune.GraphAutotuneMode;
import tuning.calibration.family.CalibrationFamilyId;
import tuning.calibration.run.CalibrationCommand;
import tuning.calibration.run.CalibrationScope;
import tuning.calibration.run.CalibrationSuite;
import tuning.preset.TuningPreset;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class TuningCliParsingTest {

    @Test
    void mainParsesDottedCalibrationCommandAsCanonicalCalibration() {
        TuningCli.ParsedCommand parsed = TuningCli.parseCommand(new String[]{
                "calibration.run", "--dtype", "f64", "--family", "matmul"
        });

        assertEquals(TuningCli.CommandKind.CALIBRATION, parsed.kind());
        assertEquals(CalibrationScope.SINGLE_FAMILY, parsed.calibration().scope());
        assertEquals(DataType.FLOAT64, parsed.calibration().dataTypes().getFirst());
        assertEquals(CalibrationFamilyId.MATMUL, parsed.calibration().family());
    }

    @Test
    void mainParsesDottedAutotuneCommandWithFlags() {
        TuningCli.ParsedCommand parsed = TuningCli.parseCommand(new String[]{
                "autotune.run",
                "--dtype", "f32",
                "--preset", "quick",
                "--graph-mode", "research",
                "--profile-root", "build/test-profiles",
                "--measurement", "1:2:3"
        });

        assertEquals(TuningCli.CommandKind.AUTOTUNE, parsed.kind());
        assertEquals(TuningPreset.QUICK, parsed.tuning().preset());
        assertEquals(GraphAutotuneMode.RESEARCH, parsed.tuning().graphMode());
        assertEquals(Path.of("build/test-profiles"), parsed.tuning().profileRoot());
        assertEquals(1, parsed.tuning().measurement().warmupIters());
        assertEquals(2, parsed.tuning().measurement().measureIters());
        assertEquals(3, parsed.tuning().measurement().repeats());
    }

    @Test
    void mainParsesGenericBenchmarkRunScenario() {
        TuningCli.ParsedCommand parsed = TuningCli.parseCommand(new String[]{
                "benchmark.run", "--scenario", "graph-space", "--dtype", "bf16"
        });

        assertEquals(TuningCli.CommandKind.BENCHMARK_GRAPH_SPACE, parsed.kind());
    }

    @Test
    void mainKeepsLegacyBenchmarkWinnerAlias() {
        TuningCli.ParsedCommand parsed = TuningCli.parseCommand(new String[]{
                "benchmark-winner", "f64"
        });

        assertEquals(TuningCli.CommandKind.BENCHMARK_WINNER, parsed.kind());
    }

    @Test
    void benchmarkCommandsAreProfileReadOnly() {
        assertEquals(false, TuningCli.CommandKind.HELP.writesProfileArtifacts());
        assertEquals(false, TuningCli.CommandKind.BENCHMARK_WINNER.writesProfileArtifacts());
        assertEquals(false, TuningCli.CommandKind.BENCHMARK_GRAPH_SPACE.writesProfileArtifacts());

        assertEquals(false, TuningCli.parseCommand(new String[]{
                "benchmark.winner", "--dtype", "f64"
        }).kind().writesProfileArtifacts());
        assertEquals(false, TuningCli.parseCommand(new String[]{
                "benchmark.run", "--scenario", "graph-space", "--dtype", "f32"
        }).kind().writesProfileArtifacts());
    }

    @Test
    void autotuneAndCalibrationCommandsWriteProfileArtifacts() {
        assertEquals(true, TuningCli.CommandKind.FULL.writesProfileArtifacts());
        assertEquals(true, TuningCli.CommandKind.CALIBRATION.writesProfileArtifacts());
        assertEquals(true, TuningCli.CommandKind.AUTOTUNE.writesProfileArtifacts());

        assertEquals(true, TuningCli.parseCommand(new String[]{
                "flow.full", "--dtype", "f64"
        }).kind().writesProfileArtifacts());
        assertEquals(true, TuningCli.parseCommand(new String[]{
                "calibration.run", "--dtype", "f64", "--families", "all"
        }).kind().writesProfileArtifacts());
        assertEquals(true, TuningCli.parseCommand(new String[]{
                "autotune.run", "--dtype", "bf16"
        }).kind().writesProfileArtifacts());
    }

    @Test
    void calibrateParsesSingleFamilyCommand() {
        CalibrationCommand command = CalibrationCommand.parse(new String[]{
                "calibrate", "--dtype", "f64", "--family", "matmul"
        });

        assertEquals(CalibrationScope.SINGLE_FAMILY, command.scope());
        assertEquals(DataType.FLOAT64, command.dataTypes().getFirst());
        assertEquals(CalibrationFamilyId.MATMUL, command.family());
        assertEquals(ExecutionMode.FORWARD_BACKWARD, command.mode());
    }

    @Test
    void calibrateParsesAllDtypesAndMeasurementOverride() {
        CalibrationCommand command = CalibrationCommand.parse(new String[]{
                "calibrate",
                "--dtypes", "all",
                "--families", "all",
                "--preset", "quick",
                "--measurement", "30:100:2",
                "--progress", "lines",
                "--color", "never"
        });

        assertEquals(CalibrationScope.ALL_FAMILIES, command.scope());
        assertEquals(3, command.dataTypes().size());
        assertEquals(TuningPreset.QUICK, command.preset());
        assertEquals(30, command.measurement().warmupIters());
        assertEquals(100, command.measurement().measureIters());
        assertEquals(2, command.measurement().repeats());
        assertEquals(1, command.passCount());
    }

    @Test
    void calibrateParsesOneFamilyAcrossAllDtypes() {
        CalibrationCommand command = CalibrationCommand.parse(new String[]{
                "calibrate", "--dtypes", "all", "--family", "matmul"
        });

        assertEquals(CalibrationScope.SINGLE_FAMILY, command.scope());
        assertEquals(3, command.dataTypes().size());
        assertEquals(CalibrationFamilyId.MATMUL, command.family());
        assertEquals(1, command.passCount());
    }

    @Test
    void balancedAllFamilyCommandUsesTwoPasses() {
        CalibrationCommand command = CalibrationCommand.parse(new String[]{
                "calibrate", "--dtype", "bf16", "--families", "all"
        });

        assertEquals(2, command.passCount());
    }

    @Test
    void calibrateRejectsOldPositionalSyntax() {
        assertThrows(
                IllegalArgumentException.class,
                () -> CalibrationCommand.parse(new String[]{"calibrate", "f64", "conv2d"})
        );
    }

    @Test
    void calibrateRejectsUnknownFamily() {
        assertThrows(
                IllegalArgumentException.class,
                () -> CalibrationCommand.parse(new String[]{"calibrate", "--dtype", "f64", "--family", "fused-arithmetic"})
        );
    }

    @Test
    void calibrateRejectsNonCalibrationDtypes() {
        assertThrows(
                IllegalArgumentException.class,
                () -> CalibrationCommand.parse(new String[]{"calibrate", "--dtype", "int32", "--family", "matmul"})
        );
    }

    @Test
    void materializationFamilyCreatesGenericAndWhereSteps() {
        var steps = CalibrationSuite.stepsFor(
                CalibrationFamilyId.MATERIALIZATION,
                "mat",
                TuningPreset.BALANCED,
                DataType.FLOAT64
        );

        assertEquals(2, steps.size());
        assertEquals("mat", steps.get(0).name());
        assertEquals("mat-where", steps.get(1).name());
    }
}

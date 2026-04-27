package synaptik.app;

import backend.runtime.ExecutionMode;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.calibration.family.CalibrationFamilyId;
import tuning.calibration.run.CalibrationCommand;
import tuning.calibration.run.CalibrationScope;
import tuning.calibration.run.CalibrationSuite;
import tuning.preset.TuningPreset;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class MainCliParsingTest {

    @Test
    void calibrateParsesSingleFamilyCommand() {
        CalibrationCommand command = CalibrationCommand.parse(new String[]{
                "calibrate", "--dtype", "f64", "--family", "conv2d-gemm-dispatch"
        });

        assertEquals(CalibrationScope.SINGLE_FAMILY, command.scope());
        assertEquals(DataType.FLOAT64, command.dataTypes().getFirst());
        assertEquals(CalibrationFamilyId.CONV2D_GEMM_DISPATCH, command.family());
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

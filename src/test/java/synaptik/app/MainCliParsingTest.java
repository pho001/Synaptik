package synaptik.app;

import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.measure.MeasurementPolicy;
import tuning.session.TuningPreset;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

class MainCliParsingTest {

    @Test
    void calibrateWithoutFamilyOrMeasurementUsesDefaults() {
        Main.CalibrationCliOptions options = Main.parseCalibrationOptions(new String[]{"calibrate", "f64"});

        assertNull(options.family());
        assertNull(options.measurement());
    }

    @Test
    void calibrateParsesFamilyOnly() {
        Main.CalibrationCliOptions options = Main.parseCalibrationOptions(new String[]{"calibrate", "f64", "conv2d"});

        assertEquals(Main.CalibrationFamilyTarget.CONV2D, options.family());
        assertNull(options.measurement());
    }

    @Test
    void calibrateParsesFamilyAndMeasurementOverride() {
        Main.CalibrationCliOptions options = Main.parseCalibrationOptions(
                new String[]{"calibrate", "f64", "conv2d", "30", "100", "2"}
        );

        assertEquals(Main.CalibrationFamilyTarget.CONV2D, options.family());
        MeasurementPolicy measurement = options.measurement();
        assertEquals(30, measurement.warmupIters());
        assertEquals(100, measurement.measureIters());
        assertEquals(2, measurement.repeats());
    }

    @Test
    void calibrateParsesMeasurementWithoutFamily() {
        Main.CalibrationCliOptions options = Main.parseCalibrationOptions(
                new String[]{"calibrate", "f32", "30", "100", "2"}
        );

        assertNull(options.family());
        MeasurementPolicy measurement = options.measurement();
        assertEquals(30, measurement.warmupIters());
        assertEquals(100, measurement.measureIters());
        assertEquals(2, measurement.repeats());
    }

    @Test
    void calibrateRejectsUnknownFamily() {
        assertThrows(
                IllegalArgumentException.class,
                () -> Main.parseCalibrationOptions(new String[]{"calibrate", "f64", "fused-arithmetic"})
        );
    }

    @Test
    void calibrateRejectsMalformedArgumentCount() {
        assertThrows(
                IllegalArgumentException.class,
                () -> Main.parseCalibrationOptions(new String[]{"calibrate", "f64", "conv2d", "30", "100"})
        );
    }

    @Test
    void materializationFamilyCreatesBothGenericAndWhereSteps() {
        var steps = Main.CalibrationFamilyTarget.MATERIALIZATION.createSteps("mat", TuningPreset.BALANCED, DataType.FLOAT64);

        assertEquals(2, steps.size());
        assertEquals("mat", steps.get(0).name());
        assertEquals("mat-where", steps.get(1).name());
    }
}

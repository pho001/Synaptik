import org.junit.jupiter.api.Test;
import tuning.calibration.progress.PlatformCalibrationProgressEvent;
import tuning.calibration.progress.PlatformCalibrationProgressPhase;
import tuning.calibration.progress.TerminalCalibrationProgressRenderer;
import tuning.calibration.progress.TerminalCapabilities;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TerminalCalibrationProgressRendererTest {
    @Test
    void rendererProducesBoundedReadablePanel() {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        TerminalCalibrationProgressRenderer renderer = new TerminalCalibrationProgressRenderer(
                new PrintStream(bytes, true, StandardCharsets.UTF_8),
                new TerminalCapabilities(false, false)
        );

        renderer.onEvent(new PlatformCalibrationProgressEvent(
                PlatformCalibrationProgressPhase.CANDIDATE_MEASURED,
                "platform",
                "MATMUL",
                2,
                13,
                "matmul_square",
                1,
                4,
                "candidate-a",
                3,
                8,
                "candidate-b",
                0.42d,
                "candidate measured"
        ));

        String output = bytes.toString(StandardCharsets.UTF_8);
        assertTrue(output.contains("Synaptik calibration"));
        assertTrue(output.contains("family   2/13"));
        assertTrue(output.contains("workload 1/4"));
        assertTrue(output.contains("candidate 3/8"));
        assertTrue(output.contains("candidate-b"));
    }

    @Test
    void rendererThrottlesRawCandidateEvents() {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        TerminalCalibrationProgressRenderer renderer = new TerminalCalibrationProgressRenderer(
                new PrintStream(bytes, true, StandardCharsets.UTF_8),
                new TerminalCapabilities(false, false)
        );

        for (int i = 1; i <= 5; i++) {
            renderer.onEvent(new PlatformCalibrationProgressEvent(
                    PlatformCalibrationProgressPhase.CANDIDATE_MEASURING,
                    "platform",
                    "MATMUL",
                    1,
                    1,
                    "matmul_square",
                    1,
                    1,
                    "candidate-" + i,
                    i,
                    5,
                    "",
                    Double.NaN,
                    "measuring"
            ));
        }

        String output = bytes.toString(StandardCharsets.UTF_8);
        assertEquals(1, output.split("Synaptik calibration", -1).length - 1);
    }

    @Test
    void liveRendererRedrawsSamePanelLines() {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        TerminalCalibrationProgressRenderer renderer = new TerminalCalibrationProgressRenderer(
                new PrintStream(bytes, true, StandardCharsets.UTF_8),
                new TerminalCapabilities(false, true)
        );

        renderer.onEvent(new PlatformCalibrationProgressEvent(
                PlatformCalibrationProgressPhase.STARTED,
                "platform",
                "",
                0,
                2,
                "",
                0,
                0,
                "",
                0,
                0,
                "",
                Double.NaN,
                "started"
        ));
        renderer.onEvent(new PlatformCalibrationProgressEvent(
                PlatformCalibrationProgressPhase.FAMILY_STARTED,
                "platform",
                "MATMUL",
                1,
                2,
                "",
                0,
                0,
                "candidate-with-a-very-long-name-that-should-not-wrap-the-live-progress-panel",
                0,
                0,
                "",
                Double.NaN,
                "family started"
        ));

        String output = bytes.toString(StandardCharsets.UTF_8);
        assertTrue(output.contains("\u001B[8F"));
        assertTrue(output.contains("\u001B[2K"));
        assertEquals(2, output.split("Synaptik calibration", -1).length - 1);
    }

    @Test
    void explicitLiveModeEnablesRedrawEvenWithoutInteractiveConsole() {
        TerminalCapabilities capabilities = TerminalCapabilities.detect(
                "auto",
                "live",
                new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8)
        );

        assertTrue(capabilities.liveRedrawEnabled());
    }
}

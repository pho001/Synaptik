import org.junit.jupiter.api.Test;
import tuning.calibration.family.CalibrationFamilyId;
import tuning.calibration.store.CalibrationArtifactLayout;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class CalibrationArtifactLayoutTest {
    @Test
    void schemaV2LayoutSeparatesLatestHistoryAndRunArtifacts() {
        CalibrationArtifactLayout layout = CalibrationArtifactLayout.of(Path.of("profiles"), "mac-arm64-test-10c");

        assertEquals(
                Path.of("profiles/platform/mac-arm64-test-10c/calibration/schema-v2/latest/f64/forward-backward/profile.json"),
                layout.latestProfilePath("f64", "forward_backward")
        );
        assertEquals(
                Path.of("profiles/platform/mac-arm64-test-10c/calibration/schema-v2/history/f64/forward-backward/matmul.jsonl"),
                layout.historyPath("f64", "forward_backward", CalibrationFamilyId.MATMUL)
        );
        assertEquals(
                Path.of("profiles/platform/mac-arm64-test-10c/calibration/schema-v2/runs/run-1/f64/forward-backward/matmul/result.json"),
                layout.resultJsonPath("run-1", "f64", "forward_backward", CalibrationFamilyId.MATMUL)
        );
    }
}

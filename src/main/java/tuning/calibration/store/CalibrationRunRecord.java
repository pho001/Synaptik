package tuning.calibration.store;

import tuning.calibration.PlatformCalibrationStepResult;
import tuning.calibration.family.CalibrationFamilyRegistry;

import java.nio.file.Path;
import java.time.OffsetDateTime;
import java.util.Locale;

public record CalibrationRunRecord(
        String runId,
        String dtype,
        String mode,
        int passIndex,
        int passCount,
        PlatformCalibrationStepResult step,
        Path selectedProfilePath
) {
    public String toJson() {
        return "{"
                + "\"runId\":\"" + escape(runId) + "\","
                + "\"createdAt\":\"" + OffsetDateTime.now() + "\","
                + "\"dtype\":\"" + escape(dtype) + "\","
                + "\"mode\":\"" + escape(mode) + "\","
                + "\"family\":\"" + escape(CalibrationFamilyRegistry.spec(step.family()).cliName()) + "\","
                + "\"passIndex\":" + passIndex + ","
                + "\"passCount\":" + passCount + ","
                + "\"winner\":\"" + escape(step.winner().candidateId()) + "\","
                + "\"score\":" + format(step.selectedScore().score()) + ","
                + "\"scoreMetric\":\"" + escape(step.scoreMetric()) + "\","
                + "\"candidateSpaceVersion\":\"" + escape(CalibrationFamilyRegistry.version()) + "\","
                + "\"selectedProfilePath\":\"" + escape(selectedProfilePath == null ? "" : selectedProfilePath.toString()) + "\""
                + "}";
    }

    private static String format(double value) {
        return Double.isFinite(value) ? String.format(Locale.US, "%.6f", value) : "null";
    }

    private static String escape(String value) {
        return value == null ? "" : value.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}

package tuning.report;

import tuning.session.PlatformCalibrationResult;
import tuning.session.PlatformCalibrationStepResult;

import java.util.Locale;

public final class TextPlatformCalibrationResultRenderer {
    private TextPlatformCalibrationResultRenderer() {
    }

    public static String render(PlatformCalibrationResult result) {
        if (result == null) {
            throw new IllegalArgumentException("result cannot be null");
        }
        StringBuilder sb = new StringBuilder();
        sb.append("Platform Calibration Result\n");
        sb.append("platformId=").append(result.platformId()).append('\n');
        sb.append("createdAt=").append(result.createdAt()).append('\n');
        sb.append("persisted=").append(result.persisted()).append('\n');
        sb.append("outputProfilePath=").append(result.outputProfilePath() == null ? "n/a" : result.outputProfilePath()).append('\n');
        sb.append("profileName=").append(result.profileName()).append('\n');
        sb.append("dataType=").append(result.finalRuntimeProfile().dataType()).append('\n');
        sb.append("mode=").append(result.finalRuntimeProfile().metadata().executionMode()).append('\n');
        sb.append("seedRuntimeProfile=").append(result.seedRuntimeProfile().metadata().platformProfileId()).append('\n');
        sb.append("finalRuntimeProfile=").append(result.finalRuntimeProfile().metadata().platformProfileId()).append("\n\n");

        sb.append("Hardware\n");
        sb.append("os=").append(result.hardware().os()).append('\n');
        sb.append("arch=").append(result.hardware().arch()).append('\n');
        sb.append("vm=").append(result.hardware().vm()).append('\n');
        sb.append("vendor=").append(result.hardware().vendor()).append('\n');
        sb.append("cores=").append(result.hardware().cores()).append("\n\n");

        if (!result.steps().isEmpty()) {
            sb.append("Steps\n");
            sb.append(String.format(Locale.US, "%-22s %-18s %-18s %-18s %-12s %-18s%n",
                    "name", "family", "seedRuntime", "selectedExec", "score", "metric"));
            for (PlatformCalibrationStepResult step : result.steps()) {
                sb.append(String.format(
                        Locale.US,
                        "%-22s %-18s %-18s %-18s %-12s %-18s%n",
                        step.name(),
                        step.family().name(),
                        step.seedRuntimeProfile().metadata().platformProfileId(),
                        step.selectedExecutionProfile().candidateName(),
                        Double.isFinite(step.selectedScore().score()) ? String.format(Locale.US, "%.6f", step.selectedScore().score()) : "n/a",
                        step.scoreMetric()
                ));
            }
        }

        return sb.toString();
    }
}

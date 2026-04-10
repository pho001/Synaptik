package tuning.report;

import tuning.session.PlatformCalibrationResult;
import tuning.session.PlatformCalibrationStepResult;

import java.util.Locale;

public final class JsonPlatformCalibrationResultRenderer {
    private JsonPlatformCalibrationResultRenderer() {
    }

    public static String render(PlatformCalibrationResult result) {
        if (result == null) {
            throw new IllegalArgumentException("result cannot be null");
        }
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"platformId\": \"").append(escape(result.platformId())).append("\",\n");
        sb.append("  \"createdAt\": \"").append(result.createdAt()).append("\",\n");
        sb.append("  \"persisted\": ").append(result.persisted()).append(",\n");
        sb.append("  \"outputProfilePath\": \"").append(escape(result.outputProfilePath() == null ? "" : result.outputProfilePath().toString())).append("\",\n");
        sb.append("  \"profileName\": \"").append(escape(result.profileName())).append("\",\n");
        sb.append("  \"dataType\": \"").append(result.finalRuntimeProfile().dataType().name()).append("\",\n");
        sb.append("  \"mode\": \"").append(result.finalRuntimeProfile().metadata().executionMode().name()).append("\",\n");
        sb.append("  \"seedRuntimeProfile\": \"").append(escape(result.seedRuntimeProfile().metadata().platformProfileId())).append("\",\n");
        sb.append("  \"finalRuntimeProfile\": \"").append(escape(result.finalRuntimeProfile().metadata().platformProfileId())).append("\",\n");
        sb.append("  \"hardware\": {\n");
        sb.append("    \"os\": \"").append(escape(result.hardware().os())).append("\",\n");
        sb.append("    \"arch\": \"").append(escape(result.hardware().arch())).append("\",\n");
        sb.append("    \"vm\": \"").append(escape(result.hardware().vm())).append("\",\n");
        sb.append("    \"vendor\": \"").append(escape(result.hardware().vendor())).append("\",\n");
        sb.append("    \"cores\": ").append(result.hardware().cores()).append("\n");
        sb.append("  },\n");
        sb.append("  \"steps\": [\n");
        for (int i = 0; i < result.steps().size(); i++) {
            if (i > 0) {
                sb.append(",\n");
            }
            PlatformCalibrationStepResult step = result.steps().get(i);
            sb.append("    {\n");
            sb.append("      \"name\": \"").append(escape(step.name())).append("\",\n");
            sb.append("      \"family\": \"").append(step.family().name()).append("\",\n");
            sb.append("      \"seedRuntimeProfile\": \"").append(escape(step.seedRuntimeProfile().metadata().platformProfileId())).append("\",\n");
            sb.append("      \"selectedRuntimeProfile\": \"").append(escape(step.selectedRuntimeProfile().metadata().platformProfileId())).append("\",\n");
            sb.append("      \"selectedExecutionProfile\": \"").append(escape(step.selectedExecutionProfile().candidateName())).append("\",\n");
            sb.append("      \"score\": ").append(format(step.selectedScore().score())).append(",\n");
            sb.append("      \"scoreMetric\": \"").append(escape(step.scoreMetric())).append("\",\n");
            sb.append("      \"winner\": \"").append(escape(step.winner().candidateId())).append("\",\n");
            sb.append("      \"suiteSummary\": {\n");
            sb.append("        \"workloadCount\": ").append(step.benchmarkReport().workloadReports().size()).append(",\n");
            sb.append("        \"candidateCount\": ").append(step.benchmarkReport().totalCandidateCount()).append(",\n");
            sb.append("        \"successCount\": ").append(step.benchmarkReport().totalSuccessCount()).append(",\n");
            sb.append("        \"failureCount\": ").append(step.benchmarkReport().totalFailureCount()).append("\n");
            sb.append("      },\n");
            sb.append("      \"candidateSummaries\": [\n");
            for (int j = 0; j < step.candidateSummaries().size(); j++) {
                if (j > 0) {
                    sb.append(",\n");
                }
                var candidate = step.candidateSummaries().get(j);
                sb.append("        {\n");
                sb.append("          \"candidateId\": \"").append(escape(candidate.candidateId())).append("\",\n");
                sb.append("          \"valid\": ").append(candidate.score().valid()).append(",\n");
                sb.append("          \"score\": ").append(format(candidate.score().score())).append(",\n");
                sb.append("          \"geometricMeanMs\": ").append(format(candidate.score().geometricMeanMs())).append(",\n");
                sb.append("          \"worstBucketMedianMs\": ").append(format(candidate.score().worstBucketMedianMs())).append(",\n");
                sb.append("          \"variancePenalty\": ").append(format(candidate.score().variancePenalty())).append(",\n");
                sb.append("          \"explanation\": \"").append(escape(candidate.score().explanation())).append("\"\n");
                sb.append("        }");
            }
            sb.append("\n      ]\n");
            sb.append("    }");
        }
        sb.append("\n  ]\n");
        sb.append("}\n");
        return sb.toString();
    }

    private static String format(double value) {
        return Double.isFinite(value) ? String.format(Locale.US, "%.6f", value) : "null";
    }

    private static String escape(String value) {
        return value == null ? "" : value.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}

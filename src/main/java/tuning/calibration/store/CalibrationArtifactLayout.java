package tuning.calibration.store;

import tuning.calibration.family.CalibrationFamilyId;

import java.nio.file.Path;
import java.util.Locale;
import java.util.Objects;

public record CalibrationArtifactLayout(Path root, String platformId) {
    public CalibrationArtifactLayout {
        root = Objects.requireNonNull(root, "root cannot be null")
                .resolve("platform")
                .resolve(platformId == null || platformId.isBlank() ? "platform" : platformId)
                .resolve("calibration")
                .resolve("schema-v2");
        platformId = platformId == null || platformId.isBlank() ? "platform" : platformId;
    }

    public static CalibrationArtifactLayout of(Path outputRoot, String platformId) {
        return new CalibrationArtifactLayout(outputRoot == null ? Path.of("profiles") : outputRoot, platformId);
    }

    public Path rootManifestPath() {
        return root.resolve("manifest.json");
    }

    public Path runRoot(String runId) {
        return root.resolve("runs").resolve(runId);
    }

    public Path runManifestPath(String runId) {
        return runRoot(runId).resolve("manifest.json");
    }

    public Path familyRoot(String runId, String dtype, String mode, CalibrationFamilyId family) {
        return runRoot(runId)
                .resolve(dtype)
                .resolve(modeId(mode))
                .resolve(familyId(family));
    }

    public Path resultJsonPath(String runId, String dtype, String mode, CalibrationFamilyId family) {
        return familyRoot(runId, dtype, mode, family).resolve("result.json");
    }

    public Path resultTextPath(String runId, String dtype, String mode, CalibrationFamilyId family) {
        return familyRoot(runId, dtype, mode, family).resolve("result.txt");
    }

    public Path selectedProfilePath(String runId, String dtype, String mode, CalibrationFamilyId family) {
        return familyRoot(runId, dtype, mode, family).resolve("selected-profile.json");
    }

    public Path candidatesPath(String runId, String dtype, String mode, CalibrationFamilyId family) {
        return familyRoot(runId, dtype, mode, family).resolve("candidates.jsonl");
    }

    public Path historyPath(String dtype, String mode, CalibrationFamilyId family) {
        return root.resolve("history").resolve(dtype).resolve(modeId(mode)).resolve(familyId(family) + ".jsonl");
    }

    public Path latestProfilePath(String dtype, String mode) {
        return root.resolve("latest").resolve(dtype).resolve(modeId(mode)).resolve("profile.json");
    }

    public Path latestManifestPath(String dtype, String mode) {
        return root.resolve("latest").resolve(dtype).resolve(modeId(mode)).resolve("manifest.json");
    }

    private static String familyId(CalibrationFamilyId family) {
        return family == null ? "unknown" : family.name().toLowerCase(Locale.ROOT).replace('_', '-');
    }

    private static String modeId(String mode) {
        return mode == null ? "forward-backward" : mode.toLowerCase(Locale.ROOT).replace('_', '-');
    }
}

package tuning.calibration.store;

import tuning.calibration.family.CalibrationFamilyId;
import tuning.calibration.family.CalibrationFamilyRegistry;
import tuning.calibration.run.CalibrationCommand;
import tuning.calibration.run.CalibrationScope;
import tuning.store.HardwareFingerprint;

import java.nio.file.Path;
import java.time.OffsetDateTime;
import java.util.List;
import java.util.Locale;

public record CalibrationRunManifest(
        String schemaVersion,
        String runId,
        String status,
        OffsetDateTime createdAt,
        OffsetDateTime completedAt,
        String platformId,
        HardwareFingerprint hardware,
        List<String> dtypes,
        String mode,
        List<String> families,
        String familyRegistryVersion,
        int passCount,
        String preset,
        String progressMode,
        String colorMode,
        Path runRoot
) {
    public static CalibrationRunManifest started(
            String runId,
            String platformId,
            HardwareFingerprint hardware,
            CalibrationCommand command,
            Path runRoot
    ) {
        return new CalibrationRunManifest(
                "schema-v2",
                runId,
                "started",
                OffsetDateTime.now(),
                null,
                platformId,
                hardware,
                command.dataTypes().stream().map(CalibrationCommand::dtypeId).toList(),
                command.mode().name().toLowerCase(Locale.ROOT),
                familyNames(command.families()),
                CalibrationFamilyRegistry.version(),
                command.passCount(),
                command.preset().name().toLowerCase(Locale.ROOT),
                command.progressMode(),
                command.colorMode(),
                runRoot
        );
    }

    public CalibrationRunManifest completed() {
        return withStatus("completed");
    }

    public CalibrationRunManifest failed() {
        return withStatus("failed");
    }

    public String latestJson(String dtype, String selectedProfilePath) {
        return "{\n"
                + "  \"schemaVersion\": \"" + schemaVersion + "\",\n"
                + "  \"runId\": \"" + escape(runId) + "\",\n"
                + "  \"status\": \"" + escape(status) + "\",\n"
                + "  \"dtype\": \"" + escape(dtype) + "\",\n"
                + "  \"mode\": \"" + escape(mode) + "\",\n"
                + "  \"familyIds\": " + jsonArray(families) + ",\n"
                + "  \"passCount\": " + passCount + ",\n"
                + "  \"selectedProfilePath\": \"" + escape(selectedProfilePath) + "\"\n"
                + "}\n";
    }

    public String toJson() {
        return "{\n"
                + "  \"schemaVersion\": \"" + schemaVersion + "\",\n"
                + "  \"runId\": \"" + escape(runId) + "\",\n"
                + "  \"status\": \"" + escape(status) + "\",\n"
                + "  \"createdAt\": \"" + createdAt + "\",\n"
                + "  \"completedAt\": \"" + (completedAt == null ? "" : completedAt) + "\",\n"
                + "  \"platformId\": \"" + escape(platformId) + "\",\n"
                + "  \"architecture\": \"" + escape(hardware == null ? "" : hardware.arch()) + "\",\n"
                + "  \"jvm\": \"" + escape(hardware == null ? "" : hardware.vm()) + "\",\n"
                + "  \"dtypes\": " + jsonArray(dtypes) + ",\n"
                + "  \"mode\": \"" + escape(mode) + "\",\n"
                + "  \"familyIds\": " + jsonArray(families) + ",\n"
                + "  \"familyRegistryVersion\": \"" + escape(familyRegistryVersion) + "\",\n"
                + "  \"knobSpaceVersion\": \"" + escape(familyRegistryVersion) + "\",\n"
                + "  \"workloadSetVersion\": \"" + escape(familyRegistryVersion) + "\",\n"
                + "  \"passCount\": " + passCount + ",\n"
                + "  \"preset\": \"" + escape(preset) + "\",\n"
                + "  \"scope\": \"" + (families.size() == 1 ? CalibrationScope.SINGLE_FAMILY : CalibrationScope.ALL_FAMILIES) + "\",\n"
                + "  \"progressMode\": \"" + escape(progressMode) + "\",\n"
                + "  \"colorMode\": \"" + escape(colorMode) + "\",\n"
                + "  \"runRoot\": \"" + escape(runRoot == null ? "" : runRoot.toString()) + "\"\n"
                + "}\n";
    }

    private CalibrationRunManifest withStatus(String nextStatus) {
        return new CalibrationRunManifest(
                schemaVersion,
                runId,
                nextStatus,
                createdAt,
                OffsetDateTime.now(),
                platformId,
                hardware,
                dtypes,
                mode,
                families,
                familyRegistryVersion,
                passCount,
                preset,
                progressMode,
                colorMode,
                runRoot
        );
    }

    private static List<String> familyNames(List<CalibrationFamilyId> families) {
        return families.stream().map(family -> CalibrationFamilyRegistry.spec(family).cliName()).toList();
    }

    private static String jsonArray(List<String> values) {
        return "[" + String.join(", ", values.stream().map(value -> "\"" + escape(value) + "\"").toList()) + "]";
    }

    private static String escape(String value) {
        return value == null ? "" : value.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}

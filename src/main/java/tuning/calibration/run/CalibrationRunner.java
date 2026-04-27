package tuning.calibration.run;

import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.GraphExecutionPolicy;
import config.profile.PlatformRuntimeProfile;
import config.profile.PlatformRuntimeProfileIO;
import config.profile.WorkloadProfile;
import tensor.DataType;
import tuning.calibration.PlatformCalibrationRequest;
import tuning.calibration.PlatformCalibrationResult;
import tuning.calibration.PlatformCalibrationSession;
import tuning.calibration.progress.LoggingPlatformCalibrationProgressListener;
import tuning.calibration.progress.PlatformCalibrationProgressListener;
import tuning.calibration.progress.SynaptikBanner;
import tuning.calibration.progress.TerminalCalibrationProgressRenderer;
import tuning.calibration.store.CalibrationArtifactLayout;
import tuning.calibration.store.CalibrationRunManifest;
import tuning.calibration.store.CalibrationRunStore;
import tuning.calibration.store.PlatformCalibrationPaths;
import tuning.store.HardwareFingerprint;

import java.io.PrintStream;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

public final class CalibrationRunner {
    private final PrintStream out;

    public CalibrationRunner(PrintStream out) {
        this.out = out == null ? System.out : out;
    }

    public static CalibrationRunner create() {
        return new CalibrationRunner(System.out);
    }

    public List<PlatformCalibrationResult> run(CalibrationCommand command) {
        Objects.requireNonNull(command, "command cannot be null");
        HardwareFingerprint hardware = HardwareFingerprint.capture();
        String platformId = PlatformCalibrationPaths.platformId(hardware);
        CalibrationArtifactLayout layout = CalibrationArtifactLayout.of(command.outputRoot(), platformId);
        String runId = runId();
        CalibrationRunStore store = new CalibrationRunStore(layout);
        CalibrationRunManifest manifest = CalibrationRunManifest.started(
                runId,
                platformId,
                hardware,
                command,
                layout.runRoot(runId)
        );
        store.writeManifest(manifest);

        if (!"quiet".equals(command.progressMode())) {
            out.print(SynaptikBanner.render(!"never".equals(command.colorMode())));
            out.println(command.describePlan());
            out.println("artifactRoot=" + layout.root());
            out.flush();
        }

        List<PlatformCalibrationResult> results = new ArrayList<>();
        Map<String, PlatformRuntimeProfile> completedProfiles = new LinkedHashMap<>();
        try {
            for (DataType dataType : command.dataTypes()) {
                PlatformRuntimeProfile current = loadSeedProfile(layout, platformId, hardware, dataType, command.mode());
                for (int passIndex = 1; passIndex <= command.passCount(); passIndex++) {
                    CalibrationPlan plan = CalibrationPlan.build(command, dataType, passIndex);
                    PlatformCalibrationRequest request = new PlatformCalibrationRequest(
                            platformId,
                            "platform-" + CalibrationCommand.dtypeId(dataType) + "-" + modeId(command.mode()) + "-calibration",
                            dataType,
                            command.mode(),
                            graphPolicy(command.mode()),
                            current,
                            plan.steps(),
                            null,
                            command.measurement(),
                            progressListener(command)
                    );
                    PlatformCalibrationResult result = PlatformCalibrationSession.create(request).run();
                    results.add(result);
                    String dtype = CalibrationCommand.dtypeId(dataType);
                    String mode = modeId(command.mode());
                    for (var step : result.steps()) {
                        store.saveStep(runId, dtype, mode, passIndex, command.passCount(), result, step);
                    }
                    current = result.finalRuntimeProfile();
                }
                completedProfiles.put(CalibrationCommand.dtypeId(dataType), current);
            }
            CalibrationRunManifest completed = manifest.completed();
            store.writeManifest(completed);
            for (Map.Entry<String, PlatformRuntimeProfile> entry : completedProfiles.entrySet()) {
                store.publishLatest(completed, entry.getKey(), modeId(command.mode()), entry.getValue());
            }
            return List.copyOf(results);
        } catch (RuntimeException ex) {
            store.writeManifest(manifest.failed());
            throw ex;
        }
    }

    private PlatformRuntimeProfile loadSeedProfile(
            CalibrationArtifactLayout layout,
            String platformId,
            HardwareFingerprint hardware,
            DataType dataType,
            ExecutionMode mode
    ) {
        ExecutionProfile fallbackProfile = seedExecutionProfile(dataType, mode);
        PlatformRuntimeProfile fallback = PlatformRuntimeProfile.fromExecutionProfile(
                platformId,
                hardware.key(),
                "built-in-defaults",
                fallbackProfile
        );
        var latest = layout.latestProfilePath(CalibrationCommand.dtypeId(dataType), modeId(mode));
        if (!Files.exists(latest)) {
            return fallback;
        }
        return PlatformRuntimeProfileIO.loadOrDefault(latest, fallback);
    }

    private static PlatformCalibrationProgressListener progressListener(CalibrationCommand command) {
        return switch (command.progressMode()) {
            case "quiet" -> PlatformCalibrationProgressListener.noop();
            case "lines" -> LoggingPlatformCalibrationProgressListener.defaults();
            default -> TerminalCalibrationProgressRenderer.create(command.colorMode(), command.progressMode());
        };
    }

    private static ExecutionProfile seedExecutionProfile(DataType dataType, ExecutionMode mode) {
        boolean training = mode == ExecutionMode.FORWARD_BACKWARD;
        return new ExecutionProfile(
                "platform-seed-" + CalibrationCommand.dtypeId(dataType) + "-" + modeId(mode),
                "platform-seed-" + CalibrationCommand.dtypeId(dataType) + "-" + modeId(mode),
                dataType,
                mode,
                training ? config.optimizer.OptimizerConfig.trainingDefaults() : config.optimizer.OptimizerConfig.inferenceDefaults(),
                training ? config.runtime.RuntimeConfig.trainingDefaults() : config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
    }

    private static GraphExecutionPolicy graphPolicy(ExecutionMode mode) {
        return mode == ExecutionMode.FORWARD_BACKWARD
                ? GraphExecutionPolicy.trainingDefaults()
                : GraphExecutionPolicy.inferenceDefaults();
    }

    private static String modeId(ExecutionMode mode) {
        return mode.name().toLowerCase(Locale.ROOT).replace('_', '-');
    }

    private static String runId() {
        return java.time.OffsetDateTime.now()
                .toString()
                .replace(":", "")
                .replace(".", "")
                .replace("+", "p")
                .replace("-", "")
                + "-"
                + java.util.UUID.randomUUID().toString().substring(0, 8);
    }
}

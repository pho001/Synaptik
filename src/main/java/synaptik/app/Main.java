package synaptik.app;

import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.ExecutionProfileAssembler;
import config.profile.GraphExecutionPolicy;
import config.profile.PlatformRuntimeProfile;
import config.profile.PlatformRuntimeProfileIO;
import config.profile.WorkloadProfile;
import tensor.DataType;
import tuning.candidate.ProfileGridCandidateSpace;
import tuning.candidate.ProfileMutators;
import tuning.measure.MeasurementPolicy;
import tuning.report.TextBenchmarkReportRenderer;
import tuning.report.TextPlatformCalibrationResultRenderer;
import tuning.report.TextTuningResultRenderer;
import tuning.search.ExhaustiveSearchStrategy;
import tuning.session.AutotuneSession;
import tuning.session.BenchmarkEntry;
import tuning.session.BenchmarkSession;
import tuning.session.LoggingPlatformCalibrationProgressListener;
import tuning.session.PlatformCalibrationDefaults;
import tuning.session.PlatformCalibrationRequest;
import tuning.session.PlatformCalibrationSession;
import tuning.session.TuningDefaults;
import tuning.session.TuningPreset;
import tuning.store.JsonFileBestProfileStore;
import tuning.store.JsonFileTuningHistoryStore;
import tuning.store.HardwareFingerprint;
import tuning.store.PersistencePolicy;
import tuning.store.PlatformCalibrationLayout;
import tuning.store.PlatformCalibrationPaths;
import tuning.store.PlatformCalibrationSaveHelper;
import tuning.validate.DefaultValidationEngine;
import tuning.workload.StandardWorkloads;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;

public final class Main {
    private Main() {
    }

    public static void main(String[] args) {
        if (args.length == 0) {
            runFull(DTypeTarget.F64);
            return;
        }

        if (args.length < 2) {
            printUsage();
            return;
        }

        Phase phase = Phase.parse(args[0]);
        DTypeTarget dtype = DTypeTarget.parse(args[1]);
        if (phase == null || dtype == null) {
            printUsage();
            return;
        }

        switch (phase) {
            case FULL -> {
                requireExactArgCount(args, 2);
                runFull(dtype);
            }
            case CALIBRATE -> runCalibration(dtype, parseCalibrationMeasurement(args));
            case AUTOTUNE -> {
                requireExactArgCount(args, 2);
                runAutotune(dtype);
            }
            case BENCHMARK_WINNER -> {
                requireExactArgCount(args, 2);
                runWinnerBenchmark(dtype);
            }
            case BENCHMARK_STAGE_SPACE -> {
                requireExactArgCount(args, 2);
                runStageSpaceBenchmark(dtype);
            }
        }
    }

    private static void runFull(DTypeTarget dtype) {
        System.out.println(header(dtype, "full flow"));
        System.out.println("note=convenience flow for local iteration; for the cleanest performance numbers prefer running phases separately");
        runCalibration(dtype, null);
        runAutotune(dtype);
        runWinnerBenchmark(dtype);
    }

    private static void runCalibration(DTypeTarget dtype, MeasurementPolicy measurement) {
        System.out.println(header(dtype, "training calibration"));
        printCalibrationMeasurement(measurement);
        ExecutionProfile seed = trainingSeedProfile(dtype);
        PlatformCalibrationLayout layout = calibrationLayout(dtype, seed);
        PlatformCalibrationRequest baseRequest = PlatformCalibrationDefaults.balancedTrainingFull(
                layout.platformId(),
                seed,
                layout.profilePath()
        );
        PlatformCalibrationRequest request = new PlatformCalibrationRequest(
                baseRequest.platformId(),
                baseRequest.profileName(),
                baseRequest.dataType(),
                baseRequest.executionMode(),
                baseRequest.graphPolicy(),
                baseRequest.seedRuntimeProfile(),
                baseRequest.steps(),
                baseRequest.outputProfilePath(),
                measurement,
                LoggingPlatformCalibrationProgressListener.defaults()
        );

        var result = PlatformCalibrationSession.create(request).run();
        PlatformCalibrationSaveHelper.saveAll(
                result,
                layout.profilePath(),
                layout.jsonReportPath(),
                layout.textReportPath()
        );

        System.out.println(TextPlatformCalibrationResultRenderer.render(result));
        System.out.println("profilePath=" + result.outputProfilePath());
    }

    private static void printCalibrationMeasurement(MeasurementPolicy measurement) {
        if (measurement == null) {
            return;
        }
        System.out.println("measurementOverride="
                + "warmup=" + measurement.warmupIters()
                + ", measure=" + measurement.measureIters()
                + ", repeats=" + measurement.repeats());
    }

    private static void runAutotune(DTypeTarget dtype) {
        System.out.println(header(dtype, "ABC autotune"));
        ExecutionProfile calibratedSeed = calibratedAbcSeed(dtype);
        var workload = abcWorkload(dtype);
        var candidateSpace = stageCandidateSpace(calibratedSeed);
        PersistencePolicy persistence = tuningPersistence(dtype);

        var request = TuningDefaults.balancedAutotune(
                workload,
                calibratedSeed,
                candidateSpace,
                persistence
        );
        var result = AutotuneSession.create(
                request,
                new ExhaustiveSearchStrategy(),
                new tuning.measure.DefaultMeasurementEngine(),
                new DefaultValidationEngine(),
                new JsonFileBestProfileStore(),
                new JsonFileTuningHistoryStore()
        ).run();

        System.out.println(TextTuningResultRenderer.render(result));
        System.out.println("autotuneBestProfilePath=" + persistence.bestProfilePath());
        System.out.println("autotuneHistoryPath=" + persistence.historyPath());
    }

    private static void runWinnerBenchmark(DTypeTarget dtype) {
        System.out.println(header(dtype, "ABC benchmark: baseline vs winner"));
        ExecutionProfile baseline = baselineProfile(dtype);
        ExecutionProfile winner = loadWinnerProfile(dtype);

        var request = TuningDefaults.benchmark(
                TuningPreset.BALANCED,
                abcWorkload(dtype),
                List.of(
                        BenchmarkEntry.baseline("baseline-no-opt", baseline),
                        BenchmarkEntry.candidate("best-profile", winner)
                )
        );
        var report = BenchmarkSession.create(request).run();
        System.out.println(TextBenchmarkReportRenderer.render(report));
    }

    private static void runStageSpaceBenchmark(DTypeTarget dtype) {
        System.out.println(header(dtype, "ABC benchmark: stage space exploration"));
        ExecutionProfile baseline = baselineProfile(dtype);
        ExecutionProfile calibratedSeed = calibratedAbcSeed(dtype);
        var workload = abcWorkload(dtype);
        var candidateSpace = stageCandidateSpace(calibratedSeed);

        List<BenchmarkEntry> entries = new ArrayList<>();
        entries.add(BenchmarkEntry.baseline("baseline-no-opt", baseline));
        candidateSpace.generate(workload).forEach(candidate ->
                entries.add(BenchmarkEntry.candidate(candidate.name(), candidate.profile()))
        );

        var request = TuningDefaults.benchmark(
                TuningPreset.BALANCED,
                workload,
                entries
        );
        var report = BenchmarkSession.create(request).run();
        System.out.println(TextBenchmarkReportRenderer.render(report));
    }

    private static ExecutionProfile trainingSeedProfile(DTypeTarget dtype) {
        return new ExecutionProfile(
                "platform-seed-" + dtype.id + "-training",
                "platform-seed-" + dtype.id + "-training",
                dtype.dataType,
                ExecutionMode.FORWARD_BACKWARD,
                config.optimizer.OptimizerConfig.trainingDefaults(),
                config.runtime.RuntimeConfig.trainingDefaults(),
                WorkloadProfile.none()
        );
    }

    private static PlatformCalibrationLayout calibrationLayout(DTypeTarget dtype, ExecutionProfile seed) {
        HardwareFingerprint hardware = HardwareFingerprint.capture();
        String platformId = PlatformCalibrationPaths.platformId(hardware);
        String variant = PlatformCalibrationPaths.variantId(seed);
        Path root = Path.of("profiles", "platform", platformId);
        return new PlatformCalibrationLayout(
                platformId,
                hardware,
                root.resolve("calibration").resolve(variant + ".json"),
                root.resolve("reports").resolve("calibration-" + variant + ".json"),
                root.resolve("reports").resolve("calibration-" + variant + ".txt")
        );
    }

    private static PlatformCalibrationLayout legacyCalibrationLayout(DTypeTarget dtype, ExecutionProfile seed) {
        return PlatformCalibrationPaths.defaultLayout(
                Path.of("build", "platform-calibration", dtype.id),
                seed
        );
    }

    private static PlatformRuntimeProfile loadCalibrationProfile(DTypeTarget dtype) {
        ExecutionProfile seed = trainingSeedProfile(dtype);
        PlatformCalibrationLayout layout = calibrationLayout(dtype, seed);
        Path path = ensurePreferredArtifact(
                layout.profilePath(),
                legacyCalibrationLayout(dtype, seed).profilePath(),
                "calibration profile"
        );
        if (!Files.exists(path)) {
            throw new IllegalStateException("Missing calibration profile: " + path + ". Run `calibrate " + dtype.id + "` first.");
        }
        PlatformRuntimeProfile fallback = PlatformRuntimeProfile.fromExecutionProfile(
                layout.platformId(),
                layout.hardware().key(),
                "fallback",
                seed
        );
        return PlatformRuntimeProfileIO.loadOrDefault(path, fallback);
    }

    private static ExecutionProfile calibratedAbcSeed(DTypeTarget dtype) {
        return ExecutionProfileAssembler.assemble(
                "abc-" + dtype.id + "-calibrated",
                "abc-" + dtype.id + "-calibrated",
                dtype.dataType,
                ExecutionMode.FORWARD_BACKWARD,
                loadCalibrationProfile(dtype),
                GraphExecutionPolicy.trainingDefaults(),
                WorkloadProfile.none()
        );
    }

    private static ProfileGridCandidateSpace stageCandidateSpace(ExecutionProfile seed) {
        return new ProfileGridCandidateSpace(
                seed,
                List.of(ProfileMutators.constrainedStageOrderSpace())
        );
    }

    private static ExecutionProfile baselineProfile(DTypeTarget dtype) {
        return new ExecutionProfile(
                "abc-baseline-no-opt-" + dtype.id,
                "abc-baseline-no-opt-" + dtype.id,
                dtype.dataType,
                ExecutionMode.FORWARD_BACKWARD,
                config.optimizer.OptimizerConfig.noOptimization(),
                config.runtime.RuntimeConfig.noOptNoVecNoPar(),
                WorkloadProfile.none()
        );
    }

    private static ExecutionProfile loadWinnerProfile(DTypeTarget dtype) {
        PersistencePolicy preferred = tuningPersistence(dtype);
        PersistencePolicy legacy = legacyTuningPersistence(dtype);
        ensurePreferredArtifact(
                preferred.bestProfilePath(),
                legacy.bestProfilePath(),
                "best profile"
        );
        ensurePreferredArtifact(
                preferred.historyPath(),
                legacy.historyPath(),
                "tuning history"
        );
        Path path = resolveExisting(
                tuningPersistence(dtype).bestProfilePath(),
                legacyTuningPersistence(dtype).bestProfilePath()
        );
        return new JsonFileBestProfileStore()
                .load(path)
                .orElseThrow(() -> new IllegalStateException("Missing best profile: " + path + ". Run `autotune " + dtype.id + "` first."))
                .profile();
    }

    private static PersistencePolicy tuningPersistence(DTypeTarget dtype) {
        String platformId = PlatformCalibrationPaths.platformId(HardwareFingerprint.capture());
        Path root = Path.of("profiles", "platform", platformId, "tuning", "abc");
        return new PersistencePolicy(
                true,
                true,
                root.resolve(dtype.id + "-best-profile.json"),
                root.resolve(dtype.id + "-history.jsonl")
        );
    }

    private static PersistencePolicy legacyTuningPersistence(DTypeTarget dtype) {
        return new PersistencePolicy(
                true,
                true,
                Path.of("build", "tuning", "best-profiles", "abc-" + dtype.id + "-best-profile.json"),
                Path.of("build", "tuning", "history", "abc-" + dtype.id + "-history.jsonl")
        );
    }

    private static Path resolveExisting(Path preferred, Path fallback) {
        if (preferred != null && Files.exists(preferred)) {
            return preferred;
        }
        return fallback;
    }

    private static Path ensurePreferredArtifact(Path preferred, Path legacy, String label) {
        if (preferred == null || Files.exists(preferred)) {
            return preferred;
        }
        if (legacy == null || !Files.exists(legacy)) {
            return preferred;
        }
        try {
            Files.createDirectories(preferred.getParent());
            Files.copy(legacy, preferred, StandardCopyOption.REPLACE_EXISTING);
            System.out.println("migratedLegacyArtifact=" + label + " -> " + preferred);
            return preferred;
        } catch (Exception e) {
            throw new IllegalStateException(
                    "Failed to migrate legacy " + label + " from " + legacy + " to " + preferred,
                    e
            );
        }
    }

    private static tuning.workload.WorkloadSpec abcWorkload(DTypeTarget dtype) {
        return StandardWorkloads.abcSequenceMatmul("abc_sequence_matmul_" + dtype.id, 64, 10_000);
    }

    private static String header(DTypeTarget dtype, String title) {
        return "\n==============================\n"
                + "Synaptik " + dtype.id.toUpperCase() + " " + title + "\n"
                + "==============================";
    }

    private static void printUsage() {
        System.out.println("""
                Usage:
                  ./gradlew run --args="full <f64|f32|bf16>"
                  ./gradlew run --args="calibrate <f64|f32|bf16> [warmup measure repeats]"
                  ./gradlew run --args="autotune <f64|f32|bf16>"
                  ./gradlew run --args="benchmark-winner <f64|f32|bf16>"
                  ./gradlew run --args="benchmark-stage-space <f64|f32|bf16>"

                Notes:
                  - no args defaults to `full f64`
                  - run phases separately to avoid cross-phase JVM warmup bias
                  - `calibrate` optionally accepts explicit measurement override, e.g. `calibrate f64 30 100 2`
                  - `autotune` expects an existing calibration profile
                  - `benchmark-winner` expects an existing best-profile artifact
                """);
    }

    private static void requireExactArgCount(String[] args, int expected) {
        if (args.length != expected) {
            printUsage();
            throw new IllegalArgumentException("Unexpected argument count.");
        }
    }

    private static MeasurementPolicy parseCalibrationMeasurement(String[] args) {
        if (args.length == 2) {
            return null;
        }
        if (args.length != 5) {
            printUsage();
            throw new IllegalArgumentException("Calibration measurement override expects exactly 3 integers: warmup measure repeats.");
        }
        try {
            int warmup = Integer.parseInt(args[2]);
            int measure = Integer.parseInt(args[3]);
            int repeats = Integer.parseInt(args[4]);
            MeasurementPolicy base = TuningPreset.BALANCED.benchmarkMeasurement();
            return new MeasurementPolicy(
                    warmup,
                    measure,
                    repeats,
                    base.measureCompile(),
                    base.measurePrepare(),
                    base.measureColdRun(),
                    base.measureSteadyState(),
                    base.captureStepTrace()
            );
        } catch (NumberFormatException ex) {
            printUsage();
            throw new IllegalArgumentException("Calibration measurement override must be integers.", ex);
        }
    }

    private enum Phase {
        FULL("full"),
        CALIBRATE("calibrate"),
        AUTOTUNE("autotune"),
        BENCHMARK_WINNER("benchmark-winner"),
        BENCHMARK_STAGE_SPACE("benchmark-stage-space");

        private final String cliName;

        Phase(String cliName) {
            this.cliName = cliName;
        }

        private static Phase parse(String value) {
            if (value == null) {
                return null;
            }
            for (Phase phase : values()) {
                if (phase.cliName.equalsIgnoreCase(value)) {
                    return phase;
                }
            }
            return null;
        }
    }

    private enum DTypeTarget {
        F64("f64", DataType.FLOAT64),
        F32("f32", DataType.FLOAT32),
        BF16("bf16", DataType.BFLOAT16);

        private final String id;
        private final DataType dataType;

        DTypeTarget(String id, DataType dataType) {
            this.id = id;
            this.dataType = dataType;
        }

        private static DTypeTarget parse(String value) {
            if (value == null) {
                return null;
            }
            for (DTypeTarget target : values()) {
                if (target.id.equalsIgnoreCase(value)) {
                    return target;
                }
            }
            return null;
        }
    }
}

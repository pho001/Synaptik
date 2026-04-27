package synaptik.app;

import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.GraphExecutionPolicy;
import config.profile.PlatformRuntimeProfile;
import config.profile.PlatformRuntimeProfileIO;
import config.profile.WorkloadProfile;
import tensor.DataType;
import tuning.calibration.run.CalibrationCommand;
import tuning.calibration.run.CalibrationRunner;
import tuning.calibration.run.CalibrationScope;
import tuning.autotune.GraphAutotuneMode;
import tuning.autotune.GraphAutotuneRequest;
import tuning.candidate.graph.GraphAutotuneCandidateSpace;
import tuning.benchmark.report.TextBenchmarkReportRenderer;
import tuning.autotune.report.TextTuningResultRenderer;
import tuning.search.SearchPolicy;
import tuning.autotune.AutotuneSession;
import tuning.benchmark.BenchmarkEntry;
import tuning.benchmark.BenchmarkSession;
import tuning.autotune.TuningDefaults;
import tuning.preset.TuningPreset;
import tuning.calibration.store.CalibrationArtifactLayout;
import tuning.store.JsonFileBestProfileStore;
import tuning.store.JsonFileTuningHistoryStore;
import tuning.store.HardwareFingerprint;
import tuning.store.PersistencePolicy;
import tuning.calibration.store.PlatformCalibrationPaths;
import tuning.validate.DefaultValidationEngine;
import tuning.workload.StandardWorkloads;

import java.nio.file.Files;
import java.nio.file.Path;
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

        Phase phase = Phase.parse(args[0]);
        if (phase == null) {
            printUsage();
            return;
        }

        switch (phase) {
            case FULL -> {
                requireExactArgCount(args, 2);
                DTypeTarget dtype = requireDType(args[1]);
                runFull(dtype);
            }
            case CALIBRATE -> runCalibration(CalibrationCommand.parse(args));
            case AUTOTUNE -> {
                requireExactArgCount(args, 2);
                DTypeTarget dtype = requireDType(args[1]);
                runAutotune(dtype);
            }
            case BENCHMARK_WINNER -> {
                requireExactArgCount(args, 2);
                DTypeTarget dtype = requireDType(args[1]);
                runWinnerBenchmark(dtype);
            }
            case BENCHMARK_GRAPH_SPACE -> {
                requireExactArgCount(args, 2);
                DTypeTarget dtype = requireDType(args[1]);
                runGraphSpaceBenchmark(dtype);
            }
        }
    }

    private static void runFull(DTypeTarget dtype) {
        System.out.println(header(dtype, "full flow"));
        System.out.println("note=convenience flow for local iteration; for the cleanest performance numbers prefer running phases separately");
        runCalibration(new CalibrationCommand(
                List.of(dtype.dataType),
                null,
                CalibrationScope.ALL_FAMILIES,
                TuningPreset.BALANCED,
                ExecutionMode.FORWARD_BACKWARD,
                null,
                "auto",
                "live",
                Path.of("profiles"),
                false
        ));
        runAutotune(dtype);
        runWinnerBenchmark(dtype);
    }

    private static void runCalibration(CalibrationCommand command) {
        CalibrationRunner.create().run(command);
    }

    private static void runAutotune(DTypeTarget dtype) {
        System.out.println(header(dtype, "ABC autotune"));
        PlatformRuntimeProfile runtimeProfile = loadCalibrationProfile(dtype);
        var workload = abcWorkload(dtype);
        PersistencePolicy persistence = tuningPersistence(dtype);

        var graphRequest = new GraphAutotuneRequest(
                workload,
                "abc-" + dtype.id + "-graph-autotune",
                dtype.dataType,
                ExecutionMode.FORWARD_BACKWARD,
                GraphExecutionPolicy.trainingDefaults(),
                runtimeProfile,
                GraphAutotuneMode.STANDARD,
                TuningPreset.BALANCED.autotuneMeasurement(),
                TuningPreset.BALANCED.autotuneValidation(),
                new SearchPolicy(1, 1, 1, false),
                persistence,
                null
        );
        var result = AutotuneSession.create(
                graphRequest.toAutotuneRequest(),
                new tuning.search.SingleCandidateSearchStrategy(),
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

    private static void runGraphSpaceBenchmark(DTypeTarget dtype) {
        System.out.println(header(dtype, "ABC benchmark: graph space exploration"));
        ExecutionProfile baseline = baselineProfile(dtype);
        PlatformRuntimeProfile runtimeProfile = loadCalibrationProfile(dtype);
        var workload = abcWorkload(dtype);
        var candidateSpace = new GraphAutotuneCandidateSpace(
                "abc-" + dtype.id + "-graph-space",
                dtype.dataType,
                ExecutionMode.FORWARD_BACKWARD,
                runtimeProfile,
                GraphExecutionPolicy.trainingDefaults(),
                GraphAutotuneMode.STANDARD
        );

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

    private static PlatformRuntimeProfile loadCalibrationProfile(DTypeTarget dtype) {
        ExecutionProfile seed = trainingSeedProfile(dtype);
        HardwareFingerprint hardware = HardwareFingerprint.capture();
        String platformId = PlatformCalibrationPaths.platformId(hardware);
        CalibrationArtifactLayout layout = CalibrationArtifactLayout.of(Path.of("profiles"), platformId);
        Path path = layout.latestProfilePath(dtype.id, ExecutionMode.FORWARD_BACKWARD.name());
        if (!Files.exists(path)) {
            throw new IllegalStateException("Missing calibration profile: " + path
                    + ". Run `calibrate --dtype " + dtype.id + " --families all` first.");
        }
        PlatformRuntimeProfile fallback = PlatformRuntimeProfile.fromExecutionProfile(
                platformId,
                hardware.key(),
                "fallback",
                seed
        );
        return PlatformRuntimeProfileIO.loadOrDefault(path, fallback);
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
        Path path = tuningPersistence(dtype).bestProfilePath();
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

    private static tuning.workload.WorkloadSpec abcWorkload(DTypeTarget dtype) {
        return StandardWorkloads.abcSequenceMatmulBlasBenchmark("abc_sequence_matmul_" + dtype.id);
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
                  ./gradlew run --args="calibrate --dtype <f64|f32|bf16> --family <family-id>"
                  ./gradlew run --args="calibrate --dtype <f64|f32|bf16> --families all"
                  ./gradlew run --args="calibrate --dtypes all --families all"
                  ./gradlew run --args="autotune <f64|f32|bf16>"
                  ./gradlew run --args="benchmark-winner <f64|f32|bf16>"
                  ./gradlew run --args="benchmark-graph-space <f64|f32|bf16>"

                Notes:
                  - no args defaults to `full f64`
                  - run phases separately to avoid cross-phase JVM warmup bias
                  - `calibrate` accepts explicit measurement override, e.g. `--measurement 30:100:2`
                  - supported calibration families: %s
                  - `autotune` expects an existing calibration profile
                  - `benchmark-winner` expects an existing best-profile artifact
                """.formatted(tuning.calibration.family.CalibrationFamilyRegistry.supportedCliNames()));
    }

    private static void requireExactArgCount(String[] args, int expected) {
        if (args.length != expected) {
            printUsage();
            throw new IllegalArgumentException("Unexpected argument count.");
        }
    }

    private enum Phase {
        FULL("full"),
        CALIBRATE("calibrate"),
        AUTOTUNE("autotune"),
        BENCHMARK_WINNER("benchmark-winner"),
        BENCHMARK_GRAPH_SPACE("benchmark-graph-space");

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

    private static DTypeTarget requireDType(String value) {
        DTypeTarget dtype = DTypeTarget.parse(value);
        if (dtype == null) {
            printUsage();
            throw new IllegalArgumentException("Unknown dtype: " + value);
        }
        return dtype;
    }
}

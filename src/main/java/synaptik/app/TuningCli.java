package synaptik.app;

import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.GraphExecutionPolicy;
import config.profile.PlatformRuntimeProfile;
import config.profile.PlatformRuntimeProfileIO;
import config.profile.WorkloadProfile;
import tensor.DataType;
import tuning.autotune.AutotuneSession;
import tuning.autotune.GraphAutotuneMode;
import tuning.autotune.GraphAutotuneRequest;
import tuning.autotune.TuningDefaults;
import tuning.autotune.report.TextTuningResultRenderer;
import tuning.benchmark.BenchmarkEntry;
import tuning.benchmark.BenchmarkRequest;
import tuning.benchmark.BenchmarkSession;
import tuning.benchmark.report.TextBenchmarkReportRenderer;
import tuning.calibration.run.CalibrationCommand;
import tuning.calibration.run.CalibrationRunner;
import tuning.calibration.run.CalibrationScope;
import tuning.calibration.store.CalibrationArtifactLayout;
import tuning.calibration.store.PlatformCalibrationPaths;
import tuning.candidate.graph.GraphAutotuneCandidateSpace;
import tuning.measure.MeasurementPolicy;
import tuning.preset.TuningPreset;
import tuning.search.SearchPolicy;
import tuning.store.BestProfileRecord;
import tuning.store.HardwareFingerprint;
import tuning.store.JsonFileBestProfileStore;
import tuning.store.PersistencePolicy;
import tuning.workload.StandardWorkloads;
import tuning.workload.WorkloadSpec;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Command-line entry point for local Synaptik calibration, graph autotune, and benchmark flows.
 *
 * <p>The application is intentionally a thin orchestration layer. It parses ergonomic CLI commands,
 * builds the existing request objects from the {@code tuning.*} packages, and delegates execution to
 * calibration, autotune, or benchmark sessions. It does not implement tuning algorithms directly.</p>
 *
 * <p>Typical production-style invocations are:</p>
 *
 * <pre>{@code
 * ./gradlew run --args="calibration.run --dtype f64 --families all"
 * ./gradlew run --args="autotune.run --dtype f64"
 * ./gradlew run --args="benchmark.winner --dtype f64"
 * }</pre>
 *
 * <p>The no-argument path runs the local convenience flow for {@code f64}. That flow is useful during
 * development, but separate JVM invocations usually produce cleaner performance measurements because
 * JVM warmup from one phase cannot bias the next phase.</p>
 *
 * <p>This type has no shared mutable runtime state. The sessions it launches may write calibration and
 * tuning artifacts below the configured profile root, which defaults to {@code profiles}.</p>
 */
public final class TuningCli {
    private static final Path DEFAULT_PROFILE_ROOT = Path.of("profiles");

    private TuningCli() {
    }

    /**
     * Runs the Synaptik command-line interface.
     *
     * @param args command tokens supplied by Gradle or the shell; may be empty to run the default
     *             {@code flow.full --dtype f64} convenience flow
     * @throws IllegalArgumentException if command-line options are invalid
     * @throws IllegalStateException if a requested autotune or benchmark flow depends on missing
     *                               calibration or best-profile artifacts
     */
    public static void main(String[] args) {
        ParsedCommand command;
        try {
            command = parseCommand(args);
        } catch (IllegalArgumentException ex) {
            printUsage();
            throw ex;
        }

        switch (command.kind()) {
            case HELP -> printUsage();
            case FULL -> runFull(command.tuning());
            case CALIBRATION -> runCalibration(command.calibration());
            case AUTOTUNE -> runAutotune(command.tuning());
            case BENCHMARK_WINNER -> runWinnerBenchmark(command.tuning());
            case BENCHMARK_GRAPH_SPACE -> runGraphSpaceBenchmark(command.tuning());
        }
    }

    static ParsedCommand parseCommand(String[] args) {
        if (args == null || args.length == 0) {
            return ParsedCommand.tuning(CommandKind.FULL, TuningOptions.defaults(DTypeTarget.F64));
        }

        String command = canonicalCommand(args[0]);
        return switch (command) {
            case "help" -> ParsedCommand.help();
            case "full" -> ParsedCommand.tuning(
                    CommandKind.FULL,
                    parseTuningOptions(args, 1, true, null)
            );
            case "calibrate" -> ParsedCommand.calibration(CalibrationCommand.parse(normalizeCalibrationArgs(args)));
            case "autotune" -> ParsedCommand.tuning(
                    CommandKind.AUTOTUNE,
                    parseTuningOptions(args, 1, false, null)
            );
            case "benchmark-winner" -> ParsedCommand.tuning(
                    CommandKind.BENCHMARK_WINNER,
                    parseTuningOptions(args, 1, false, null)
            );
            case "benchmark-graph-space" -> ParsedCommand.tuning(
                    CommandKind.BENCHMARK_GRAPH_SPACE,
                    parseTuningOptions(args, 1, false, null)
            );
            case "benchmark-run" -> {
                TuningOptions options = parseTuningOptions(args, 1, false, null);
                if (options.scenario() == null) {
                    throw new IllegalArgumentException("benchmark.run requires --scenario <winner|graph-space>.");
                }
                yield ParsedCommand.tuning(options.scenario().commandKind(), options);
            }
            default -> throw new IllegalArgumentException("Unknown command: " + args[0]);
        };
    }

    private static void runFull(TuningOptions options) {
        System.out.println(header(options.dtype(), "full flow"));
        System.out.println("note=convenience flow for local iteration; for the cleanest performance numbers prefer running phases separately");
        runCalibration(new CalibrationCommand(
                List.of(options.dtype().dataType),
                null,
                CalibrationScope.ALL_FAMILIES,
                options.preset(),
                ExecutionMode.FORWARD_BACKWARD,
                options.measurement(),
                "auto",
                "live",
                options.profileRoot(),
                false
        ));
        runAutotune(options);
        runWinnerBenchmark(options);
    }

    private static void runCalibration(CalibrationCommand command) {
        CalibrationRunner.create().run(command);
    }

    private static void runAutotune(TuningOptions options) {
        DTypeTarget dtype = options.dtype();
        WorkloadTarget workloadTarget = options.workload();
        System.out.println(header(dtype, workloadTarget.displayName() + " autotune"));
        PlatformRuntimeProfile runtimeProfile = loadCalibrationProfile(dtype, options.profileRoot());
        WorkloadSpec workload = workloadTarget.workload(dtype);
        PersistencePolicy persistence = tuningPersistence(dtype, options.profileRoot(), workloadTarget.namespace());
        GraphAutotuneMode graphMode = options.graphMode();
        TuningPreset preset = options.preset();
        MeasurementPolicy measurement = options.measurement() == null
                ? preset.autotuneMeasurement()
                : options.measurement();

        var graphRequest = new GraphAutotuneRequest(
                workload,
                workloadTarget.namespace() + "-" + dtype.id + "-graph-autotune",
                dtype.dataType,
                ExecutionMode.FORWARD_BACKWARD,
                GraphExecutionPolicy.trainingDefaults(),
                runtimeProfile,
                graphMode,
                measurement,
                preset.autotuneValidation(),
                searchPolicy(graphMode, preset),
                persistence,
                null
        );
        var result = AutotuneSession.create(graphRequest.toAutotuneRequest()).run();

        System.out.println(TextTuningResultRenderer.render(result));
        System.out.println("autotuneBestProfilePath=" + persistence.bestProfilePath());
        System.out.println("autotuneHistoryPath=" + persistence.historyPath());
    }

    private static void runWinnerBenchmark(TuningOptions options) {
        DTypeTarget dtype = options.dtype();
        WorkloadTarget workloadTarget = options.workload();
        System.out.println(header(dtype, workloadTarget.displayName() + " benchmark: baseline vs winner"));
        ExecutionProfile baseline = baselineProfile(dtype, workloadTarget);
        PlatformRuntimeProfile runtimeProfile = loadCalibrationProfile(dtype, options.profileRoot());
        ExecutionProfile winner = loadWinnerProfile(dtype, options.profileRoot(), workloadTarget.namespace(), runtimeProfile);

        var request = benchmarkRequest(
                options,
                workloadTarget.workload(dtype),
                List.of(
                        BenchmarkEntry.baseline("baseline-no-opt", baseline),
                        BenchmarkEntry.candidate("best-profile", winner)
                )
        );
        var report = BenchmarkSession.create(request).run();
        System.out.println(TextBenchmarkReportRenderer.render(report));
    }

    private static void runGraphSpaceBenchmark(TuningOptions options) {
        DTypeTarget dtype = options.dtype();
        WorkloadTarget workloadTarget = options.workload();
        System.out.println(header(dtype, workloadTarget.displayName() + " benchmark: graph space exploration"));
        ExecutionProfile baseline = baselineProfile(dtype, workloadTarget);
        PlatformRuntimeProfile runtimeProfile = loadCalibrationProfile(dtype, options.profileRoot());
        WorkloadSpec workload = workloadTarget.workload(dtype);
        var candidateSpace = new GraphAutotuneCandidateSpace(
                workloadTarget.namespace() + "-" + dtype.id + "-graph-space",
                dtype.dataType,
                ExecutionMode.FORWARD_BACKWARD,
                runtimeProfile,
                GraphExecutionPolicy.trainingDefaults(),
                options.graphMode()
        );

        List<BenchmarkEntry> entries = new ArrayList<>();
        entries.add(BenchmarkEntry.baseline("baseline-no-opt", baseline));
        candidateSpace.generate(workload).forEach(candidate ->
                entries.add(BenchmarkEntry.candidate(candidate.name(), candidate.profile()))
        );

        var request = benchmarkRequest(options, workload, entries);
        var report = BenchmarkSession.create(request).run();
        System.out.println(TextBenchmarkReportRenderer.render(report));
    }

    private static BenchmarkRequest benchmarkRequest(TuningOptions options, WorkloadSpec workload, List<BenchmarkEntry> entries) {
        if (options.measurement() == null) {
            return TuningDefaults.benchmark(options.preset(), workload, entries);
        }
        return new BenchmarkRequest(
                workload,
                entries,
                options.measurement(),
                options.preset().benchmarkValidation(),
                options.preset().reportPolicy()
        );
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

    private static PlatformRuntimeProfile loadCalibrationProfile(DTypeTarget dtype, Path profileRoot) {
        ExecutionProfile seed = trainingSeedProfile(dtype);
        HardwareFingerprint hardware = HardwareFingerprint.capture();
        String platformId = PlatformCalibrationPaths.platformId(hardware);
        CalibrationArtifactLayout layout = CalibrationArtifactLayout.of(profileRoot, platformId);
        Path path = layout.latestProfilePath(dtype.id, ExecutionMode.FORWARD_BACKWARD.name());
        if (!Files.exists(path)) {
            throw new IllegalStateException("Missing calibration profile: " + path
                    + ". Run `calibration.run --dtype " + dtype.id + " --families all` first.");
        }
        PlatformRuntimeProfile fallback = PlatformRuntimeProfile.fromExecutionProfile(
                platformId,
                hardware.key(),
                "fallback",
                seed
        );
        return PlatformRuntimeProfileIO.loadOrDefault(path, fallback);
    }

    private static ExecutionProfile baselineProfile(DTypeTarget dtype, WorkloadTarget workload) {
        return new ExecutionProfile(
                workload.namespace() + "-baseline-no-opt-" + dtype.id,
                workload.namespace() + "-baseline-no-opt-" + dtype.id,
                dtype.dataType,
                ExecutionMode.FORWARD_BACKWARD,
                config.optimizer.OptimizerConfig.noOptimization(),
                config.runtime.RuntimeConfig.noOptNoVecNoPar(),
                workload.profile()
        );
    }

    private static ExecutionProfile loadWinnerProfile(
            DTypeTarget dtype,
            Path profileRoot,
            String namespace,
            PlatformRuntimeProfile currentRuntimeProfile
    ) {
        Path path = tuningPersistence(dtype, profileRoot, namespace).bestProfilePath();
        BestProfileRecord record = new JsonFileBestProfileStore()
                .load(path)
                .orElseThrow(() -> new IllegalStateException("Missing best profile: " + path
                        + ". Run `autotune.run --dtype " + dtype.id + " --workload " + namespace + "` first."));
        ExecutionProfile winner = record.profile();
        if (currentRuntimeProfile == null) {
            return winner;
        }
        ExecutionProfile rebound = record.rebaseOnRuntime(currentRuntimeProfile);
        if (!record.runtimeProfileId().isBlank()
                && !record.runtimeProfileId().equals(currentRuntimeProfile.metadata().platformProfileId())) {
            System.out.println("note=best graph policy was measured with runtimeProfileId="
                    + record.runtimeProfileId()
                    + "; rebasing to current runtimeProfileId="
                    + currentRuntimeProfile.metadata().platformProfileId());
        }
        return rebound;
    }

    private static PersistencePolicy tuningPersistence(DTypeTarget dtype, Path profileRoot, String namespace) {
        String platformId = PlatformCalibrationPaths.platformId(HardwareFingerprint.capture());
        Path root = profileRoot.resolve("platform").resolve(platformId).resolve("tuning").resolve(namespace);
        return new PersistencePolicy(
                true,
                true,
                root.resolve(dtype.id + "-best-profile.json"),
                root.resolve(dtype.id + "-history.jsonl")
        );
    }

    private static SearchPolicy searchPolicy(GraphAutotuneMode graphMode, TuningPreset preset) {
        if (graphMode == GraphAutotuneMode.STANDARD) {
            return new SearchPolicy(16, 4, 1, false);
        }
        return preset.autotuneSearch();
    }

    private static String header(DTypeTarget dtype, String title) {
        return "\n==============================\n"
                + "Synaptik " + dtype.id.toUpperCase(Locale.ROOT) + " " + title + "\n"
                + "==============================";
    }

    private static String canonicalCommand(String value) {
        if (value == null) {
            return "help";
        }
        String normalized = value.trim().toLowerCase(Locale.ROOT).replace('_', '-');
        return switch (normalized) {
            case "-h", "--help", "help" -> "help";
            case "full", "flow.full", "tuning.full" -> "full";
            case "calibrate", "calibration", "calibration.run" -> "calibrate";
            case "autotune", "autotune.run", "autotune.graph", "tuning.autotune" -> "autotune";
            case "benchmark-winner", "benchmark.winner", "benchmark.best", "benchmark.best-profile" -> "benchmark-winner";
            case "benchmark-graph-space", "benchmark.graph-space", "benchmark.graph" -> "benchmark-graph-space";
            case "benchmark", "benchmark.run" -> "benchmark-run";
            default -> normalized;
        };
    }

    private static String[] normalizeCalibrationArgs(String[] args) {
        String[] normalized = args.clone();
        normalized[0] = "calibrate";
        return normalized;
    }

    private static TuningOptions parseTuningOptions(
            String[] args,
            int start,
            boolean dtypeOptional,
            BenchmarkScenario defaultScenario
    ) {
        DTypeTarget dtype = null;
        WorkloadTarget workload = WorkloadTarget.ABC;
        TuningPreset preset = TuningPreset.BALANCED;
        GraphAutotuneMode graphMode = GraphAutotuneMode.STANDARD;
        Path profileRoot = DEFAULT_PROFILE_ROOT;
        MeasurementPolicy measurement = null;
        BenchmarkScenario scenario = defaultScenario;

        for (int i = start; i < args.length; i++) {
            String arg = args[i];
            switch (arg) {
                case "--dtype" -> {
                    requireValue(args, i, arg);
                    ensureDTypeUnset(dtype);
                    dtype = requireDType(args[++i]);
                }
                case "--preset" -> {
                    requireValue(args, i, arg);
                    preset = parsePreset(args[++i]);
                }
                case "--workload" -> {
                    requireValue(args, i, arg);
                    workload = WorkloadTarget.parse(args[++i]);
                }
                case "--graph-mode", "--autotune-mode" -> {
                    requireValue(args, i, arg);
                    graphMode = parseGraphMode(args[++i]);
                }
                case "--profile-root", "--profiles", "--output-root" -> {
                    requireValue(args, i, arg);
                    profileRoot = Path.of(args[++i]);
                }
                case "--measurement" -> {
                    requireValue(args, i, arg);
                    measurement = parseMeasurement(args[++i], preset);
                }
                case "--scenario" -> {
                    requireValue(args, i, arg);
                    scenario = BenchmarkScenario.parse(args[++i]);
                }
                default -> {
                    if (arg.startsWith("-")) {
                        throw new IllegalArgumentException("Unknown option: " + arg);
                    }
                    ensureDTypeUnset(dtype);
                    dtype = requireDType(arg);
                }
            }
        }

        if (dtype == null) {
            if (!dtypeOptional) {
                throw new IllegalArgumentException("Command requires --dtype <f64|f32|bf16>.");
            }
            dtype = DTypeTarget.F64;
        }

        return new TuningOptions(dtype, workload, preset, graphMode, profileRoot, measurement, scenario);
    }

    private static void requireValue(String[] args, int index, String option) {
        if (index + 1 >= args.length) {
            throw new IllegalArgumentException(option + " requires a value.");
        }
    }

    private static void ensureDTypeUnset(DTypeTarget dtype) {
        if (dtype != null) {
            throw new IllegalArgumentException("Specify dtype only once.");
        }
    }

    private static DTypeTarget requireDType(String value) {
        DTypeTarget dtype = DTypeTarget.parse(value);
        if (dtype == null) {
            throw new IllegalArgumentException("Unknown dtype: " + value);
        }
        return dtype;
    }

    private static TuningPreset parsePreset(String value) {
        try {
            return TuningPreset.valueOf(value.trim().toUpperCase(Locale.ROOT));
        } catch (RuntimeException ex) {
            throw new IllegalArgumentException("Unknown preset: " + value, ex);
        }
    }

    private static GraphAutotuneMode parseGraphMode(String value) {
        return switch (value.trim().toLowerCase(Locale.ROOT)) {
            case "standard", "current" -> GraphAutotuneMode.STANDARD;
            case "research" -> GraphAutotuneMode.RESEARCH;
            default -> throw new IllegalArgumentException("Unknown graph mode: " + value);
        };
    }

    private static MeasurementPolicy parseMeasurement(String value, TuningPreset preset) {
        String[] parts = value.split(":");
        if (parts.length != 3) {
            throw new IllegalArgumentException("--measurement expects warmup:measure:repeats.");
        }
        try {
            MeasurementPolicy base = preset.benchmarkMeasurement();
            return new MeasurementPolicy(
                    Integer.parseInt(parts[0]),
                    Integer.parseInt(parts[1]),
                    Integer.parseInt(parts[2]),
                    base.measureCompile(),
                    base.measurePrepare(),
                    base.measureColdRun(),
                    base.measureSteadyState(),
                    base.captureStepTrace()
            );
        } catch (NumberFormatException ex) {
            throw new IllegalArgumentException("--measurement values must be integers.", ex);
        }
    }

    private static void printUsage() {
        System.out.println("""
                Usage:
                  ./gradlew run --args="flow.full --dtype <f64|f32|bf16>"
                  ./gradlew run --args="calibration.run --dtype <f64|f32|bf16> --family <family-id>"
                  ./gradlew run --args="calibration.run --dtype <f64|f32|bf16> --families all"
                  ./gradlew run --args="calibration.run --dtypes all --families all"
                  ./gradlew run --args="autotune.run --dtype <f64|f32|bf16>"
                  ./gradlew run --args="autotune.run --dtype <f64|f32|bf16> --workload <abc|transformer-block>"
                  ./gradlew run --args="autotune.run --dtype <f64|f32|bf16> --graph-mode <standard|research>"
                  ./gradlew run --args="benchmark.winner --dtype <f64|f32|bf16>"
                  ./gradlew run --args="benchmark.graph-space --dtype <f64|f32|bf16>"
                  ./gradlew run --args="benchmark.run --scenario <winner|graph-space> --dtype <f64|f32|bf16>"

                Short aliases still accepted:
                  full <dtype>
                  calibrate --dtype <dtype> --family <family-id>
                  autotune <dtype>
                  benchmark-winner <dtype>
                  benchmark-graph-space <dtype>

                Shared options for autotune and benchmark:
                  --workload <abc|transformer-block|transformer-hot-path>
                  --preset <quick|balanced|thorough>
                  --measurement <warmup>:<measure>:<repeats>
                  --profile-root <path>

                Calibration options:
                  --preset <quick|balanced|thorough>
                  --mode <forward|forward-backward>
                  --measurement <warmup>:<measure>:<repeats>
                  --progress <live|lines|quiet>
                  --color <auto|always|never>
                  --output-root <path>
                  --include-accelerators

                Notes:
                  - no args defaults to `flow.full --dtype f64`
                  - run phases separately to avoid cross-phase JVM warmup bias
                  - supported calibration families: %s
                  - `autotune.run` expects an existing calibration profile
                  - `benchmark.winner` expects an existing best-profile artifact
                """.formatted(tuning.calibration.family.CalibrationFamilyRegistry.supportedCliNames()));
    }

    enum CommandKind {
        HELP,
        FULL,
        CALIBRATION,
        AUTOTUNE,
        BENCHMARK_WINNER,
        BENCHMARK_GRAPH_SPACE
    }

    record ParsedCommand(
            CommandKind kind,
            CalibrationCommand calibration,
            TuningOptions tuning
    ) {
        static ParsedCommand help() {
            return new ParsedCommand(CommandKind.HELP, null, null);
        }

        static ParsedCommand calibration(CalibrationCommand calibration) {
            return new ParsedCommand(CommandKind.CALIBRATION, calibration, null);
        }

        static ParsedCommand tuning(CommandKind kind, TuningOptions tuning) {
            return new ParsedCommand(kind, null, tuning);
        }
    }

    record TuningOptions(
            DTypeTarget dtype,
            WorkloadTarget workload,
            TuningPreset preset,
            GraphAutotuneMode graphMode,
            Path profileRoot,
            MeasurementPolicy measurement,
            BenchmarkScenario scenario
    ) {
        TuningOptions {
            if (dtype == null) {
                throw new IllegalArgumentException("dtype cannot be null");
            }
            workload = workload == null ? WorkloadTarget.ABC : workload;
            preset = preset == null ? TuningPreset.BALANCED : preset;
            graphMode = graphMode == null ? GraphAutotuneMode.STANDARD : graphMode;
            profileRoot = profileRoot == null ? DEFAULT_PROFILE_ROOT : profileRoot;
        }

        static TuningOptions defaults(DTypeTarget dtype) {
            return new TuningOptions(dtype, WorkloadTarget.ABC, TuningPreset.BALANCED, GraphAutotuneMode.STANDARD, DEFAULT_PROFILE_ROOT, null, null);
        }
    }

    enum WorkloadTarget {
        ABC("abc", "ABC"),
        TRANSFORMER_BLOCK("transformer_block_hot_path", "Transformer block hot path"),
        TRANSFORMER_HOT_PATH("transformer_hot_path", "Transformer hot path");

        private final String namespace;
        private final String displayName;

        WorkloadTarget(String namespace, String displayName) {
            this.namespace = namespace;
            this.displayName = displayName;
        }

        private String namespace() {
            return namespace;
        }

        private String displayName() {
            return displayName;
        }

        private WorkloadSpec workload(DTypeTarget dtype) {
            return switch (this) {
                case ABC -> StandardWorkloads.abcSequenceMatmulBlasBenchmark("abc_sequence_matmul_" + dtype.id);
                case TRANSFORMER_BLOCK -> StandardWorkloads.transformerBlockHotPath("transformer_block_hot_path_" + dtype.id);
                case TRANSFORMER_HOT_PATH -> StandardWorkloads.transformerHotPath("transformer_hot_path_" + dtype.id);
            };
        }

        private WorkloadProfile profile() {
            return switch (this) {
                case ABC -> WorkloadProfile.none();
                case TRANSFORMER_BLOCK, TRANSFORMER_HOT_PATH -> WorkloadProfile.transformerHotPathDefaults();
            };
        }

        private static WorkloadTarget parse(String value) {
            String normalized = value == null ? "" : value.trim().toLowerCase(Locale.ROOT).replace('_', '-');
            return switch (normalized) {
                case "abc", "abc-sequence", "abc-sequence-matmul" -> ABC;
                case "transformer-block", "transformer-block-hot-path", "transformer-block-hotpath",
                     "transformer-block-hot-path-f32" -> TRANSFORMER_BLOCK;
                case "transformer", "transformer-hot-path", "transformer-hotpath" -> TRANSFORMER_HOT_PATH;
                default -> throw new IllegalArgumentException("Unknown workload: " + value);
            };
        }
    }

    enum BenchmarkScenario {
        WINNER(CommandKind.BENCHMARK_WINNER),
        GRAPH_SPACE(CommandKind.BENCHMARK_GRAPH_SPACE);

        private final CommandKind commandKind;

        BenchmarkScenario(CommandKind commandKind) {
            this.commandKind = commandKind;
        }

        private CommandKind commandKind() {
            return commandKind;
        }

        private static BenchmarkScenario parse(String value) {
            return switch (value.trim().toLowerCase(Locale.ROOT).replace('_', '-')) {
                case "winner", "best", "best-profile" -> WINNER;
                case "graph-space", "graph" -> GRAPH_SPACE;
                default -> throw new IllegalArgumentException("Unknown benchmark scenario: " + value);
            };
        }
    }

    enum DTypeTarget {
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

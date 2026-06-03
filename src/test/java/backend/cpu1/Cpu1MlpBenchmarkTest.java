package backend.cpu1;

import backend.ComputeBackend;
import backend.blas.OpenBlasRuntime;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.prepare.Cpu1NodePreparer;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.cpu1.provider.matmul.Cpu1MatmulRoute;
import backend.cpu1.storage.Cpu1StorageKind;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import config.runtime.RuntimeConfig;
import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.compile.intent.BackendIntentPlan;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.state.ExecutionState;
import operations.Operation;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.storage.NativeFloat32Storage;
import tensor.storage.NativeTensorStorage;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.OptionalInt;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

@Tag("benchmark")
class Cpu1MlpBenchmarkTest {
    private static final int WARMUP_ITERATIONS = 4;
    private static final int MEASURE_ITERATIONS = 10;
    private static final int WORKERS = 4;
    private static final int BATCH = 256;
    private static final int INPUT = 512;
    private static final int HIDDEN_1 = 384;
    private static final int HIDDEN_2 = 256;
    private static final int OUTPUT = 128;
    private static final BenchmarkProfile DEFAULT_PROFILE = new BenchmarkProfile(WARMUP_ITERATIONS, MEASURE_ITERATIONS);
    private static final BenchmarkProfile LARGE_PROFILE = new BenchmarkProfile(3, 7);
    private static final BenchmarkProfile CHAIN_PROFILE = new BenchmarkProfile(3, 7);
    private static final int THREAD_COUNT_BENCHMARK_ROUNDS = 5;
    private static final int THREAD_COUNT_BENCHMARK_FORKS = 3;
    private static final String FORK_RESULT_PREFIX = "CPU1_MLP_FORK_RESULT";

    @Test
    void benchmarkF32ThreeLayerMlpArrayVsAllNativeSegment() {
        Assumptions.assumeTrue(OpenBlasRuntime.isFloat32GemmAvailable(), OpenBlasRuntime.unavailableReason());
        MlpCase mlpCase = defaultMlpCase();
        MlpFixture fixture = mlpFixture(mlpCase);
        PreparedGraph arrayGraph = prepare(fixture, RouteKind.JAVA_ARRAY_OPENBLAS);
        PreparedGraph javaMatmulGraph = prepare(fixture, RouteKind.JAVA_ARRAY_JAVA_MATMUL);
        PreparedGraph nativeGraph = prepare(fixture, RouteKind.NATIVE_SEGMENT_OPENBLAS);
        attachNativeLeaves(nativeGraph.context(), fixture);

        BenchmarkTriple benchmark = benchmarkTriple(
                "java-array-openblas-copy",
                arrayGraph,
                "java-array-java-vector-parallel-matmul",
                javaMatmulGraph,
                "all-native-segment-openblas",
                nativeGraph,
                DEFAULT_PROFILE
        );
        List<NodeBenchmarkResult> arrayNodes = benchmarkNodes(arrayGraph, DEFAULT_PROFILE);
        List<NodeBenchmarkResult> javaMatmulNodes = benchmarkNodes(javaMatmulGraph, DEFAULT_PROFILE);
        List<NodeBenchmarkResult> nativeNodes = benchmarkNodes(nativeGraph, DEFAULT_PROFILE);

        assertArrayEquals(benchmark.array().output(), benchmark.nativeSegment().output(), 1.0e-3f);
        assertArrayEquals(benchmark.array().output(), benchmark.javaMatmul().output(), 1.0e-3f);
        assertEquals(0, nativeGraph.context().cpuMaterializationTraceCount());
        System.out.println(report(
                mlpCase,
                DEFAULT_PROFILE,
                benchmark.array(),
                benchmark.javaMatmul(),
                benchmark.nativeSegment(),
                arrayNodes,
                javaMatmulNodes,
                nativeNodes
        ));
    }

    @Test
    void benchmarkF32LargeThreeLayerMlpArrayVsAllNativeSegment() {
        Assumptions.assumeTrue(OpenBlasRuntime.isFloat32GemmAvailable(), OpenBlasRuntime.unavailableReason());
        List<MlpCase> cases = List.of(
                new MlpCase("large-b1024-i1024-h512-h256-o128-h2tanh", 1024, 1024, 512, 256, 128, ActivationKind.TANH),
                new MlpCase("large-b2048-i512-h512-h256-o128-h2tanh", 2048, 512, 512, 256, 128, ActivationKind.TANH),
                new MlpCase("large-b1024-i1024-h512-h256-o128-h2relu", 1024, 1024, 512, 256, 128, ActivationKind.RELU)
        );
        for (MlpCase mlpCase : cases) {
            MlpFixture fixture = mlpFixture(mlpCase);
            PreparedGraph arrayGraph = prepare(fixture, RouteKind.JAVA_ARRAY_OPENBLAS);
            PreparedGraph javaMatmulGraph = prepare(fixture, RouteKind.JAVA_ARRAY_JAVA_MATMUL);
            PreparedGraph nativeGraph = prepare(fixture, RouteKind.NATIVE_SEGMENT_OPENBLAS);
            attachNativeLeaves(nativeGraph.context(), fixture);

            BenchmarkTriple benchmark = benchmarkTriple(
                    "java-array-openblas-copy",
                    arrayGraph,
                    "java-array-java-vector-parallel-matmul",
                    javaMatmulGraph,
                    "all-native-segment-openblas",
                    nativeGraph,
                    LARGE_PROFILE
            );
            List<NodeBenchmarkResult> arrayNodes = benchmarkNodes(arrayGraph, LARGE_PROFILE);
            List<NodeBenchmarkResult> javaMatmulNodes = benchmarkNodes(javaMatmulGraph, LARGE_PROFILE);
            List<NodeBenchmarkResult> nativeNodes = benchmarkNodes(nativeGraph, LARGE_PROFILE);

            assertArrayEquals(benchmark.array().output(), benchmark.nativeSegment().output(), 1.0e-3f);
            assertArrayEquals(benchmark.array().output(), benchmark.javaMatmul().output(), 1.0e-3f);
            assertEquals(0, nativeGraph.context().cpuMaterializationTraceCount());
            System.out.println(report(
                    mlpCase,
                    LARGE_PROFILE,
                    benchmark.array(),
                    benchmark.javaMatmul(),
                    benchmark.nativeSegment(),
                    arrayNodes,
                    javaMatmulNodes,
                    nativeNodes
            ));
        }
    }

    @Test
    void benchmarkF32ThreeLayerMlpOpenBlasThreadCounts() {
        Assumptions.assumeTrue(OpenBlasRuntime.isFloat32GemmAvailable(), OpenBlasRuntime.unavailableReason());
        OptionalInt originalThreads = OpenBlasRuntime.getNumThreads();
        Assumptions.assumeTrue(originalThreads.isPresent(), "OpenBLAS openblas_get_num_threads unavailable");
        Assumptions.assumeTrue(
                OpenBlasRuntime.setNumThreads(originalThreads.getAsInt()),
                "OpenBLAS openblas_set_num_threads unavailable"
        );

        MlpCase mlpCase = forkedThreeLayerThreadCountMlpCase();
        List<ThreeLayerOpenBlasThreadBenchmarkResult> results = new ArrayList<>();
        float[] expectedOutput = null;
        try {
            for (int round = 1; round <= THREAD_COUNT_BENCHMARK_ROUNDS; round++) {
                for (int requestedThreads : threeLayerThreadCountOrder(round)) {
                    assertTrue(OpenBlasRuntime.setNumThreads(requestedThreads));
                    ThreeLayerOpenBlasThreadBenchmarkResult result = benchmarkThreeLayerOpenBlasThreadCountRound(
                            mlpCase,
                            LARGE_PROFILE,
                            round,
                            requestedThreads
                    );
                    BenchmarkTriple benchmark = result.benchmark();
                    if (expectedOutput == null) {
                        expectedOutput = benchmark.array().output();
                    } else {
                        assertArrayEquals(expectedOutput, benchmark.array().output(), 1.0e-3f);
                    }
                    assertArrayEquals(expectedOutput, benchmark.javaMatmul().output(), 1.0e-3f);
                    assertArrayEquals(expectedOutput, benchmark.nativeSegment().output(), 1.0e-3f);
                    results.add(result);
                }
            }
        } finally {
            OpenBlasRuntime.setNumThreads(originalThreads.getAsInt());
        }

        System.out.println(threeLayerOpenBlasThreadCountReport(
                mlpCase,
                LARGE_PROFILE,
                originalThreads.getAsInt(),
                THREAD_COUNT_BENCHMARK_ROUNDS,
                results
        ));
    }

    @Test
    void benchmarkF32ThreeLayerMlpOpenBlasThreadCountsForked() throws IOException, InterruptedException {
        Assumptions.assumeTrue(OpenBlasRuntime.isFloat32GemmAvailable(), OpenBlasRuntime.unavailableReason());
        OptionalInt originalThreads = OpenBlasRuntime.getNumThreads();
        Assumptions.assumeTrue(originalThreads.isPresent(), "OpenBLAS openblas_get_num_threads unavailable");
        Assumptions.assumeTrue(
                OpenBlasRuntime.setNumThreads(originalThreads.getAsInt()),
                "OpenBLAS openblas_set_num_threads unavailable"
        );

        List<ForkedThreeLayerOpenBlasThreadBenchmarkResult> results = new ArrayList<>();
        try {
            for (int fork = 1; fork <= THREAD_COUNT_BENCHMARK_FORKS; fork++) {
                results.addAll(runThreeLayerMlpOpenBlasThreadCountFork(fork));
            }
        } finally {
            OpenBlasRuntime.setNumThreads(originalThreads.getAsInt());
        }

        int expectedResultLines = THREAD_COUNT_BENCHMARK_FORKS * 2 * ThreeLayerRoute.values().length;
        assertEquals(expectedResultLines, results.size(), "missing fork result lines");
        MlpCase mlpCase = forkedThreeLayerThreadCountMlpCase();
        System.out.println(forkedThreeLayerOpenBlasThreadCountReport(
                mlpCase,
                LARGE_PROFILE,
                originalThreads.getAsInt(),
                THREAD_COUNT_BENCHMARK_FORKS,
                results
        ));
    }

    @Test
    void benchmarkF32MatmulActivationMatmulChainArrayVsNativeSegment() {
        Assumptions.assumeTrue(OpenBlasRuntime.isFloat32GemmAvailable(), OpenBlasRuntime.unavailableReason());
        List<ChainCase> cases = List.of(
                new ChainCase("large-chain-b1024-i1024-h512-o256-tanh", 1024, 1024, 512, 256, ActivationKind.TANH),
                new ChainCase("large-chain-b1024-i1024-h512-o256-relu", 1024, 1024, 512, 256, ActivationKind.RELU)
        );
        for (ChainCase chainCase : cases) {
            MlpFixture fixture = chainFixture(chainCase);
            PreparedGraph arrayGraph = prepare(fixture, RouteKind.JAVA_ARRAY_OPENBLAS);
            PreparedGraph javaMatmulGraph = prepare(fixture, RouteKind.JAVA_ARRAY_JAVA_MATMUL);
            PreparedGraph nativeGraph = prepare(fixture, RouteKind.NATIVE_SEGMENT_OPENBLAS);
            attachNativeLeaves(nativeGraph.context(), fixture);

            BenchmarkTriple benchmark = benchmarkTriple(
                    "java-array-openblas-copy",
                    arrayGraph,
                    "java-array-java-vector-parallel-matmul",
                    javaMatmulGraph,
                    "all-native-segment-openblas",
                    nativeGraph,
                    CHAIN_PROFILE
            );
            List<NodeBenchmarkResult> arrayNodes = benchmarkNodes(arrayGraph, CHAIN_PROFILE);
            List<NodeBenchmarkResult> javaMatmulNodes = benchmarkNodes(javaMatmulGraph, CHAIN_PROFILE);
            List<NodeBenchmarkResult> nativeNodes = benchmarkNodes(nativeGraph, CHAIN_PROFILE);

            assertArrayEquals(benchmark.array().output(), benchmark.nativeSegment().output(), 1.0e-3f);
            assertArrayEquals(benchmark.array().output(), benchmark.javaMatmul().output(), 1.0e-3f);
            assertEquals(0, nativeGraph.context().cpuMaterializationTraceCount());
            System.out.println(chainReport(
                    chainCase,
                    CHAIN_PROFILE,
                    benchmark.array(),
                    benchmark.javaMatmul(),
                    benchmark.nativeSegment(),
                    arrayNodes,
                    javaMatmulNodes,
                    nativeNodes
            ));
        }
    }

    @Test
    void benchmarkF32MatmulActivationMatmulChainOpenBlasThreadCounts() {
        Assumptions.assumeTrue(OpenBlasRuntime.isFloat32GemmAvailable(), OpenBlasRuntime.unavailableReason());
        OptionalInt originalThreads = OpenBlasRuntime.getNumThreads();
        Assumptions.assumeTrue(originalThreads.isPresent(), "OpenBLAS openblas_get_num_threads unavailable");
        Assumptions.assumeTrue(
                OpenBlasRuntime.setNumThreads(originalThreads.getAsInt()),
                "OpenBLAS openblas_set_num_threads unavailable"
        );

        ChainCase chainCase = new ChainCase("openblas-threads-b1024-i1024-h512-o256-tanh", 1024, 1024, 512, 256, ActivationKind.TANH);
        MlpFixture fixture = chainFixture(chainCase);
        PreparedGraph nativeGraph = prepare(fixture, RouteKind.NATIVE_SEGMENT_OPENBLAS);
        attachNativeLeaves(nativeGraph.context(), fixture);
        List<OpenBlasThreadCase> threadCases = openBlasThreadCases(originalThreads.getAsInt());
        List<OpenBlasThreadBenchmarkResult> results = new ArrayList<>();
        float[] expected = null;

        try {
            for (OpenBlasThreadCase threadCase : threadCases) {
                if (threadCase.requestedThreads() != null) {
                    assertTrue(OpenBlasRuntime.setNumThreads(threadCase.requestedThreads()));
                }
                OpenBlasThreadBenchmarkResult result = benchmarkOpenBlasThreadCase(threadCase, nativeGraph, CHAIN_PROFILE);
                if (expected == null) {
                    expected = result.output();
                } else {
                    assertArrayEquals(expected, result.output(), 1.0e-3f);
                }
                results.add(result);
            }
        } finally {
            OpenBlasRuntime.setNumThreads(originalThreads.getAsInt());
        }

        assertEquals(0, nativeGraph.context().cpuMaterializationTraceCount());
        System.out.println(openBlasThreadReport(chainCase, CHAIN_PROFILE, originalThreads.getAsInt(), results));
    }

    private static PreparedGraph prepare(MlpFixture fixture, RouteKind routeKind) {
        Cpu1NodePreparer preparer = new Cpu1NodePreparer();
        Map<Integer, CompiledNodeExecutionMetadata> metadata = new LinkedHashMap<>();
        for (CompiledNode node : fixture.nodes()) {
            if (node.operation() == null) {
                continue;
            }
            Cpu1PreparedArtifact artifact = preparer.prepare(node, fixture.descriptorIndex(), configFor(node, routeKind));
            metadata.put(node.id(), metadata(node, artifact));
        }
        ExecutionState state = ExecutionState.create(
                fixture.nodes(),
                fixture.descriptorIndex(),
                metadata,
                fixture.rootNode().id(),
                testsupport.PublicationPlans.forRoot(fixture.root(), fixture.nodes(), fixture.rootNode().id())
        );
        ExecutionContext context = ExecutionContext.fromRuntimeConfig(
                RuntimeConfig.inferenceDefaults(DataType.FLOAT32),
                ExecutionMode.FORWARD,
                metadata,
                state
        );
        return new PreparedGraph(fixture, Map.copyOf(metadata), context, state);
    }

    private static Cpu1PrepareConfig configFor(CompiledNode node, RouteKind routeKind) {
        Operation.OpType opType = node.operation().opType();
        if (opType == Operation.OpType.MATMUL) {
            return switch (routeKind) {
                case JAVA_ARRAY_OPENBLAS -> Cpu1PrepareConfig.scalarSingleThread()
                        .withMatmulRoute(Cpu1MatmulRoute.OPENBLAS_ARRAY_COPYING);
                case JAVA_ARRAY_JAVA_MATMUL -> Cpu1PrepareConfig.vectorParallel(WORKERS)
                        .withMatmulRoute(Cpu1MatmulRoute.JAVA_SCALAR);
                case NATIVE_SEGMENT_OPENBLAS -> Cpu1PrepareConfig.scalarMemorySegmentSingleThread()
                        .withMatmulRoute(Cpu1MatmulRoute.OPENBLAS_NATIVE_SEGMENT);
            };
        }
        if (opType == Operation.OpType.MEAN || opType == Operation.OpType.SUM) {
            return routeKind == RouteKind.NATIVE_SEGMENT_OPENBLAS
                    ? Cpu1PrepareConfig.scalarMemorySegmentSingleThread()
                    : Cpu1PrepareConfig.scalarSingleThread();
        }
        if (routeKind == RouteKind.NATIVE_SEGMENT_OPENBLAS) {
            return new Cpu1PrepareConfig(
                    Cpu1VectorizationKind.VECTOR,
                    Cpu1LaunchConfig.parallel(WORKERS),
                    Cpu1StorageKind.MEMORY_SEGMENT
            );
        }
        return Cpu1PrepareConfig.vectorParallel(WORKERS);
    }

    private static BenchmarkTriple benchmarkTriple(
            String arrayName,
            PreparedGraph arrayGraph,
            String javaMatmulName,
            PreparedGraph javaMatmulGraph,
            String nativeName,
            PreparedGraph nativeGraph,
            BenchmarkProfile profile
    ) {
        Cpu1Backend arrayBackend = new Cpu1Backend();
        Cpu1Backend javaMatmulBackend = new Cpu1Backend();
        Cpu1Backend nativeBackend = new Cpu1Backend();
        for (int i = 0; i < profile.warmupIterations(); i++) {
            switch (i % 3) {
                case 0 -> {
                    executeGraph(arrayBackend, arrayGraph);
                    executeGraph(javaMatmulBackend, javaMatmulGraph);
                    executeGraph(nativeBackend, nativeGraph);
                }
                case 1 -> {
                    executeGraph(javaMatmulBackend, javaMatmulGraph);
                    executeGraph(nativeBackend, nativeGraph);
                    executeGraph(arrayBackend, arrayGraph);
                }
                case 2 -> {
                    executeGraph(nativeBackend, nativeGraph);
                    executeGraph(arrayBackend, arrayGraph);
                    executeGraph(javaMatmulBackend, javaMatmulGraph);
                }
                default -> throw new IllegalStateException("unexpected warmup route order");
            }
        }
        NativeBenchmarkGuard nativeGuard = captureNativeBenchmarkGuard(nativeGraph);
        long[] arraySamples = new long[profile.measureIterations()];
        long[] javaMatmulSamples = new long[profile.measureIterations()];
        long[] nativeSamples = new long[profile.measureIterations()];
        for (int i = 0; i < profile.measureIterations(); i++) {
            switch (i % 3) {
                case 0 -> {
                    arraySamples[i] = measureGraph(arrayBackend, arrayGraph);
                    javaMatmulSamples[i] = measureGraph(javaMatmulBackend, javaMatmulGraph);
                    nativeSamples[i] = measureGraph(nativeBackend, nativeGraph);
                    nativeGuard.assertStillValid(nativeGraph);
                }
                case 1 -> {
                    javaMatmulSamples[i] = measureGraph(javaMatmulBackend, javaMatmulGraph);
                    nativeSamples[i] = measureGraph(nativeBackend, nativeGraph);
                    nativeGuard.assertStillValid(nativeGraph);
                    arraySamples[i] = measureGraph(arrayBackend, arrayGraph);
                }
                case 2 -> {
                    nativeSamples[i] = measureGraph(nativeBackend, nativeGraph);
                    nativeGuard.assertStillValid(nativeGraph);
                    arraySamples[i] = measureGraph(arrayBackend, arrayGraph);
                    javaMatmulSamples[i] = measureGraph(javaMatmulBackend, javaMatmulGraph);
                }
                default -> throw new IllegalStateException("unexpected measurement route order");
            }
        }
        assertEquals(0, nativeGraph.context().cpuMaterializationTraceCount());
        return new BenchmarkTriple(
                new BenchmarkResult(arrayName, medianMs(arraySamples), output(arrayGraph, false)),
                new BenchmarkResult(javaMatmulName, medianMs(javaMatmulSamples), output(javaMatmulGraph, false)),
                new BenchmarkResult(nativeName, medianMs(nativeSamples), output(nativeGraph, true))
        );
    }

    private static OpenBlasThreadBenchmarkResult benchmarkOpenBlasThreadCase(
            OpenBlasThreadCase threadCase,
            PreparedGraph graph,
            BenchmarkProfile profile
    ) {
        Cpu1Backend backend = new Cpu1Backend();
        for (int i = 0; i < profile.warmupIterations(); i++) {
            executeGraph(backend, graph);
        }
        NativeBenchmarkGuard nativeGuard = captureNativeBenchmarkGuard(graph);
        long[] samples = new long[profile.measureIterations()];
        for (int i = 0; i < profile.measureIterations(); i++) {
            samples[i] = measureGraph(backend, graph);
            nativeGuard.assertStillValid(graph);
        }
        return new OpenBlasThreadBenchmarkResult(
                threadCase.label(),
                threadCase.requestedThreads(),
                OpenBlasRuntime.getNumThreads().orElse(-1),
                OpenBlasRuntime.getParallelMode().orElse(-1),
                OpenBlasRuntime.parallelModeDescription(),
                medianMs(samples),
                output(graph, true)
        );
    }

    private static ThreeLayerOpenBlasThreadBenchmarkResult benchmarkThreeLayerOpenBlasThreadCountRound(
            MlpCase mlpCase,
            BenchmarkProfile profile,
            int round,
            int requestedThreads
    ) {
        MlpFixture fixture = mlpFixture(mlpCase);
        PreparedGraph arrayGraph = prepare(fixture, RouteKind.JAVA_ARRAY_OPENBLAS);
        PreparedGraph javaMatmulGraph = prepare(fixture, RouteKind.JAVA_ARRAY_JAVA_MATMUL);
        PreparedGraph nativeGraph = prepare(fixture, RouteKind.NATIVE_SEGMENT_OPENBLAS);
        attachNativeLeaves(nativeGraph.context(), fixture);

        BenchmarkTriple benchmark = benchmarkTriple(
                routeName(ThreeLayerRoute.ARRAY),
                arrayGraph,
                routeName(ThreeLayerRoute.JAVA_MATMUL),
                javaMatmulGraph,
                routeName(ThreeLayerRoute.NATIVE_SEGMENT),
                nativeGraph,
                profile
        );

        assertArrayEquals(benchmark.array().output(), benchmark.nativeSegment().output(), 1.0e-3f);
        assertArrayEquals(benchmark.array().output(), benchmark.javaMatmul().output(), 1.0e-3f);
        assertEquals(0, nativeGraph.context().cpuMaterializationTraceCount());
        return new ThreeLayerOpenBlasThreadBenchmarkResult(
                round,
                requestedThreads,
                OpenBlasRuntime.getNumThreads().orElse(-1),
                OpenBlasRuntime.getParallelMode().orElse(-1),
                OpenBlasRuntime.parallelModeDescription(),
                benchmark
        );
    }

    private static List<ForkedThreeLayerOpenBlasThreadBenchmarkResult> runThreeLayerMlpOpenBlasThreadCountFork(
            int fork
    ) throws IOException, InterruptedException {
        List<String> command = new ArrayList<>();
        command.add(currentJavaExecutable());
        command.add("--add-modules=jdk.incubator.vector");
        command.add("--enable-native-access=ALL-UNNAMED");
        addSystemPropertyIfPresent(command, "openblas.lib");
        addSystemPropertyIfPresent(command, "synaptik.metal.mps.lib");
        command.add("-cp");
        command.add(System.getProperty("java.class.path"));
        command.add(ForkedThreeLayerMlpBenchmarkMain.class.getName());
        command.add(Integer.toString(fork));

        ProcessBuilder processBuilder = new ProcessBuilder(command);
        processBuilder.redirectErrorStream(true);
        Process process = processBuilder.start();
        List<String> outputLines = new ArrayList<>();
        List<ForkedThreeLayerOpenBlasThreadBenchmarkResult> results = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                outputLines.add(line);
                ForkedThreeLayerOpenBlasThreadBenchmarkResult result = parseForkedThreeLayerResult(line);
                if (result != null) {
                    results.add(result);
                }
            }
        }
        int exitCode = process.waitFor();
        assertEquals(
                0,
                exitCode,
                "forked benchmark JVM failed for fork=" + fork + "\n" + String.join("\n", outputLines)
        );
        assertEquals(
                2 * ThreeLayerRoute.values().length,
                results.size(),
                "unexpected fork result line count for fork=" + fork + "\n" + String.join("\n", outputLines)
        );
        return results;
    }

    private static String currentJavaExecutable() {
        String executable = System.getProperty("os.name").toLowerCase(Locale.ROOT).contains("win")
                ? "java.exe"
                : "java";
        return Path.of(System.getProperty("java.home"), "bin", executable).toString();
    }

    private static void addSystemPropertyIfPresent(List<String> command, String propertyName) {
        String value = System.getProperty(propertyName);
        if (value != null && !value.isBlank()) {
            command.add("-D" + propertyName + "=" + value);
        }
    }

    private static void runThreeLayerMlpOpenBlasThreadCountForkMain(int fork) {
        assertTrue(OpenBlasRuntime.isFloat32GemmAvailable(), OpenBlasRuntime.unavailableReason());
        OptionalInt originalThreads = OpenBlasRuntime.getNumThreads();
        assertTrue(originalThreads.isPresent(), "OpenBLAS openblas_get_num_threads unavailable");
        assertTrue(OpenBlasRuntime.setNumThreads(originalThreads.getAsInt()), "OpenBLAS openblas_set_num_threads unavailable");

        MlpCase mlpCase = forkedThreeLayerThreadCountMlpCase();
        try {
            for (int requestedThreads : threeLayerThreadCountOrder(fork)) {
                assertTrue(OpenBlasRuntime.setNumThreads(requestedThreads));
                ThreeLayerOpenBlasThreadBenchmarkResult result = benchmarkThreeLayerOpenBlasThreadCountRound(
                        mlpCase,
                        LARGE_PROFILE,
                        fork,
                        requestedThreads
                );
                for (ThreeLayerRoute route : ThreeLayerRoute.values()) {
                    System.out.println(forkedThreeLayerResultLine(result, route));
                }
            }
        } finally {
            OpenBlasRuntime.setNumThreads(originalThreads.getAsInt());
        }
    }

    private static MlpCase forkedThreeLayerThreadCountMlpCase() {
        return new MlpCase("large-b1024-i1024-h512-h256-o128-h2tanh", 1024, 1024, 512, 256, 128, ActivationKind.TANH);
    }

    private static String forkedThreeLayerResultLine(
            ThreeLayerOpenBlasThreadBenchmarkResult result,
            ThreeLayerRoute route
    ) {
        BenchmarkResult routeResult = routeResult(result.benchmark(), route);
        return String.format(
                Locale.US,
                "%s fork=%d requestedThreads=%d effectiveThreads=%d parallelMode=%d parallelModeDescription=%s route=%s medianMs=%.6f firstOutput=%.8f",
                FORK_RESULT_PREFIX,
                result.round(),
                result.requestedThreads(),
                result.effectiveThreads(),
                result.parallelMode(),
                result.parallelModeDescription(),
                routeResult.name(),
                routeResult.medianMs(),
                routeResult.output()[0]
        );
    }

    private static ForkedThreeLayerOpenBlasThreadBenchmarkResult parseForkedThreeLayerResult(String line) {
        if (!line.startsWith(FORK_RESULT_PREFIX + " ")) {
            return null;
        }
        Map<String, String> fields = new LinkedHashMap<>();
        String[] tokens = line.substring(FORK_RESULT_PREFIX.length() + 1).split(" ");
        for (String token : tokens) {
            int equals = token.indexOf('=');
            if (equals <= 0 || equals == token.length() - 1) {
                continue;
            }
            fields.put(token.substring(0, equals), token.substring(equals + 1));
        }
        return new ForkedThreeLayerOpenBlasThreadBenchmarkResult(
                Integer.parseInt(fields.get("fork")),
                Integer.parseInt(fields.get("requestedThreads")),
                Integer.parseInt(fields.get("effectiveThreads")),
                Integer.parseInt(fields.get("parallelMode")),
                fields.get("parallelModeDescription"),
                fields.get("route"),
                Double.parseDouble(fields.get("medianMs")),
                Float.parseFloat(fields.get("firstOutput"))
        );
    }

    private static long measureGraph(Cpu1Backend backend, PreparedGraph graph) {
        long start = System.nanoTime();
        executeGraph(backend, graph);
        return System.nanoTime() - start;
    }

    private static List<NodeBenchmarkResult> benchmarkNodes(PreparedGraph graph, BenchmarkProfile profile) {
        Cpu1Backend backend = new Cpu1Backend();
        Map<Integer, long[]> samplesByNodeId = new LinkedHashMap<>();
        for (CompiledNode node : graph.fixture().nodes()) {
            if (graph.metadata().containsKey(node.id())) {
                samplesByNodeId.put(node.id(), new long[profile.measureIterations()]);
            }
        }
        for (int iteration = 0; iteration < profile.measureIterations(); iteration++) {
            for (CompiledNode node : graph.fixture().nodes()) {
                CompiledNodeExecutionMetadata metadata = graph.metadata().get(node.id());
                if (metadata == null) {
                    continue;
                }
                long start = System.nanoTime();
                backend.execute(node, metadata, graph.context());
                samplesByNodeId.get(node.id())[iteration] = System.nanoTime() - start;
            }
        }
        List<NodeBenchmarkResult> results = new ArrayList<>();
        for (CompiledNode node : graph.fixture().nodes()) {
            CompiledNodeExecutionMetadata metadata = graph.metadata().get(node.id());
            if (metadata == null) {
                continue;
            }
            Cpu1PreparedArtifact artifact = (Cpu1PreparedArtifact) metadata.artifact();
            results.add(new NodeBenchmarkResult(
                    node.id(),
                    node.operation().opType().name(),
                    Arrays.toString(node.shape()),
                    kernelLabel(artifact),
                    medianMs(samplesByNodeId.get(node.id()))
            ));
        }
        return results;
    }

    private static String kernelLabel(Cpu1PreparedArtifact artifact) {
        try {
            return artifact.preparedUnit().kernelId().name();
        } catch (IllegalStateException ignored) {
            // Not an elementwise unit.
        }
        try {
            return artifact.preparedMatmulUnit().kernelId().name();
        } catch (IllegalStateException ignored) {
            // Not a matmul unit.
        }
        try {
            return artifact.preparedReductionUnit().kernelId().name();
        } catch (IllegalStateException ignored) {
            // Not a reduction unit.
        }
        return artifact.executableUnit().getClass().getSimpleName();
    }

    private static void executeGraph(Cpu1Backend backend, PreparedGraph graph) {
        for (CompiledNode node : graph.fixture().nodes()) {
            CompiledNodeExecutionMetadata metadata = graph.metadata().get(node.id());
            if (metadata != null) {
                backend.execute(node, metadata, graph.context());
            }
        }
    }

    private static Map<Integer, NativeTensorStorage> captureNativeOutputStorages(PreparedGraph graph) {
        Map<Integer, NativeTensorStorage> outputs = new LinkedHashMap<>();
        for (CompiledNode node : graph.fixture().nodes()) {
            if (!graph.metadata().containsKey(node.id())) {
                continue;
            }
            NativeTensorStorage storage = graph.context().nativeStorageForNodeId(node.id());
            assertNotNull(storage, "missing native output storage for nodeId=" + node.id());
            outputs.put(node.id(), storage);
        }
        return outputs;
    }

    private static NativeBenchmarkGuard captureNativeBenchmarkGuard(PreparedGraph graph) {
        NativeTensorStorage expectedRootOutput = graph.context().nativeStorageForNodeId(graph.fixture().rootNode().id());
        assertNotNull(expectedRootOutput, "missing native output storage for root");
        return new NativeBenchmarkGuard(
                expectedRootOutput,
                captureNativeOutputStorages(graph),
                graph.state().nativeCpuMemoryTrace().allocationCount()
        );
    }

    private static void assertNativeOutputStoragesSame(
            PreparedGraph graph,
            Map<Integer, NativeTensorStorage> expectedStorages
    ) {
        for (Map.Entry<Integer, NativeTensorStorage> entry : expectedStorages.entrySet()) {
            assertSame(entry.getValue(), graph.context().nativeStorageForNodeId(entry.getKey()));
        }
    }

    private static MlpCase defaultMlpCase() {
        return new MlpCase("default-b256-i512-h384-h256-o128-h2tanh", BATCH, INPUT, HIDDEN_1, HIDDEN_2, OUTPUT, ActivationKind.TANH);
    }

    private static MlpFixture mlpFixture(MlpCase mlpCase) {
        float[] xData = values(mlpCase.batch() * mlpCase.input(), 37, 18, 0.03125f);
        float[] w1Data = values(mlpCase.input() * mlpCase.hidden1(), 41, 20, 0.015625f);
        float[] w2Data = values(mlpCase.hidden1() * mlpCase.hidden2(), 43, 21, 0.015625f);
        float[] w3Data = values(mlpCase.hidden2() * mlpCase.output(), 47, 23, 0.015625f);
        float[] targetData = values(mlpCase.batch() * mlpCase.output(), 31, 15, 0.0078125f);

        Tensor x = new Tensor(xData, new int[]{mlpCase.batch(), mlpCase.input()}, null, "mlp-x", DataType.FLOAT32);
        Tensor w1 = new Tensor(w1Data, new int[]{mlpCase.input(), mlpCase.hidden1()}, null, "mlp-w1", DataType.FLOAT32);
        Tensor w2 = new Tensor(w2Data, new int[]{mlpCase.hidden1(), mlpCase.hidden2()}, null, "mlp-w2", DataType.FLOAT32);
        Tensor w3 = new Tensor(w3Data, new int[]{mlpCase.hidden2(), mlpCase.output()}, null, "mlp-w3", DataType.FLOAT32);
        Tensor target = new Tensor(targetData, new int[]{mlpCase.batch(), mlpCase.output()}, null, "mlp-target", DataType.FLOAT32);

        Tensor h1 = x.matmul(w1).relu();
        Tensor h2 = activate(h1.matmul(w2), mlpCase.hidden2Activation());
        Tensor prediction = h2.matmul(w3);
        Tensor diff = prediction.sub(target);
        Tensor loss = diff.mul(diff).mean(1).mean(0, true);
        List<Tensor> tensors = loss.topologicalSort();
        List<CompiledNode> nodes = CompiledNode.snapshot(tensors, BackendIntentPlan.empty());
        CompiledTensorDescriptorIndex descriptorIndex = CompiledTensorDescriptorBuilder.build(nodes);
        Map<String, LeafData> leaves = Map.of(
                "mlp-x", new LeafData(xData),
                "mlp-w1", new LeafData(w1Data),
                "mlp-w2", new LeafData(w2Data),
                "mlp-w3", new LeafData(w3Data),
                "mlp-target", new LeafData(targetData)
        );
        return new MlpFixture(loss, nodes, descriptorIndex, nodes.getLast(), leaves);
    }

    private static MlpFixture chainFixture(ChainCase chainCase) {
        float[] xData = values(chainCase.batch() * chainCase.input(), 37, 18, 0.03125f);
        float[] w1Data = values(chainCase.input() * chainCase.hidden(), 41, 20, 0.015625f);
        float[] w2Data = values(chainCase.hidden() * chainCase.output(), 43, 21, 0.015625f);

        Tensor x = new Tensor(xData, new int[]{chainCase.batch(), chainCase.input()}, null, "chain-x", DataType.FLOAT32);
        Tensor w1 = new Tensor(w1Data, new int[]{chainCase.input(), chainCase.hidden()}, null, "chain-w1", DataType.FLOAT32);
        Tensor w2 = new Tensor(w2Data, new int[]{chainCase.hidden(), chainCase.output()}, null, "chain-w2", DataType.FLOAT32);

        Tensor h = activate(x.matmul(w1), chainCase.activation());
        Tensor out = h.matmul(w2);
        List<Tensor> tensors = out.topologicalSort();
        List<CompiledNode> nodes = CompiledNode.snapshot(tensors, BackendIntentPlan.empty());
        CompiledTensorDescriptorIndex descriptorIndex = CompiledTensorDescriptorBuilder.build(nodes);
        Map<String, LeafData> leaves = Map.of(
                "chain-x", new LeafData(xData),
                "chain-w1", new LeafData(w1Data),
                "chain-w2", new LeafData(w2Data)
        );
        return new MlpFixture(out, nodes, descriptorIndex, nodes.getLast(), leaves);
    }

    private static Tensor activate(Tensor input, ActivationKind activation) {
        return switch (activation) {
            case TANH -> input.tanh();
            case RELU -> input.relu();
        };
    }

    private static void attachNativeLeaves(ExecutionContext context, MlpFixture fixture) {
        for (CompiledNode node : fixture.nodes()) {
            if (!node.leaf()) {
                continue;
            }
            LeafData leaf = fixture.leaves().get(node.label());
            if (leaf != null) {
                attachNativeF32(context, node.id(), leaf.values());
            }
        }
    }

    private static void attachNativeF32(ExecutionContext context, int nodeId, float[] values) {
        NativeFloat32Storage storage = assertInstanceOf(
                NativeFloat32Storage.class,
                context.allocateNativeStorage(DataType.FLOAT32, values.length, "cpu1 mlp native leaf " + nodeId)
        );
        for (int i = 0; i < values.length; i++) {
            storage.setFloat32At(i, values[i]);
        }
        context.attachNativeStorage(nodeId, storage, "cpu1 MLP benchmark native leaf");
    }

    private static float[] output(PreparedGraph graph, boolean nativeOutput) {
        int rootNodeId = graph.fixture().rootNode().id();
        if (nativeOutput) {
            NativeTensorStorage storage = graph.context().nativeStorageForNodeId(rootNodeId);
            return nativeF32Values(assertInstanceOf(NativeFloat32Storage.class, storage));
        }
        return graph.context().runtimeTensorForNodeId(rootNodeId).toFloat32ArrayCopy();
    }

    private static float[] nativeF32Values(NativeFloat32Storage storage) {
        float[] out = new float[storage.getSize()];
        for (int i = 0; i < out.length; i++) {
            out[i] = storage.getFloat32At(i);
        }
        return out;
    }

    private static float[] values(int size, int modulus, int center, float scale) {
        float[] out = new float[size];
        for (int i = 0; i < out.length; i++) {
            out[i] = ((i % modulus) - center) * scale;
        }
        return out;
    }

    private static double medianMs(long[] samplesNs) {
        long[] sorted = samplesNs.clone();
        Arrays.sort(sorted);
        int middle = sorted.length / 2;
        if (sorted.length % 2 == 1) {
            return sorted[middle] / 1_000_000.0d;
        }
        return ((sorted[middle - 1] + sorted[middle]) / 2.0d) / 1_000_000.0d;
    }

    private static String report(
            MlpCase mlpCase,
            BenchmarkProfile profile,
            BenchmarkResult array,
            BenchmarkResult javaMatmul,
            BenchmarkResult nativeSegment,
            List<NodeBenchmarkResult> arrayNodes,
            List<NodeBenchmarkResult> javaMatmulNodes,
            List<NodeBenchmarkResult> nativeNodes
    ) {
        return String.format(
                Locale.US,
                """
                cpu1 F32 3-layer MLP benchmark
                case: %s
                graph: batch=%d, input=%d, hidden1=%d, hidden2=%d, hidden2Activation=%s, output=%d, loss=MSE mean(1).mean(0,true)
                  openblas: source=%s threadPolicy=%s sgemm=%s
                  warmup=%d, measure=%d, elementwiseWorkers=%d
                  total measurement: rotated end-to-end samples; route order rotates every iteration.
                  total %-28s medianMs=%8.4f ratio=%6.2fx loss=% .8f
                  total %-28s medianMs=%8.4f ratio=%6.2fx loss=% .8f
                  total %-28s medianMs=%8.4f ratio=%6.2fx loss=% .8f cpuMaterializations=0
                  per-node diagnostics: standalone node timings for route inspection only; these medians are not expected to sum to total.
                %s
                %s
                %s
                """,
                mlpCase.name(),
                mlpCase.batch(),
                mlpCase.input(),
                mlpCase.hidden1(),
                mlpCase.hidden2(),
                mlpCase.hidden2Activation(),
                mlpCase.output(),
                OpenBlasRuntime.lookupSource(),
                OpenBlasRuntime.threadPolicy(),
                OpenBlasRuntime.isFloat32GemmAvailable(),
                profile.warmupIterations(),
                profile.measureIterations(),
                WORKERS,
                array.name(),
                array.medianMs(),
                1.0d,
                array.output()[0],
                javaMatmul.name(),
                javaMatmul.medianMs(),
                array.medianMs() / javaMatmul.medianMs(),
                javaMatmul.output()[0],
                nativeSegment.name(),
                nativeSegment.medianMs(),
                array.medianMs() / nativeSegment.medianMs(),
                nativeSegment.output()[0],
                nodeBreakdownReport("java-array-openblas-copy nodes", arrayNodes),
                nodeBreakdownReport("java-array-java-vector-parallel-matmul nodes", javaMatmulNodes),
                nodeBreakdownReport("all-native-segment-openblas nodes", nativeNodes)
        );
    }

    private static String chainReport(
            ChainCase chainCase,
            BenchmarkProfile profile,
            BenchmarkResult array,
            BenchmarkResult javaMatmul,
            BenchmarkResult nativeSegment,
            List<NodeBenchmarkResult> arrayNodes,
            List<NodeBenchmarkResult> javaMatmulNodes,
            List<NodeBenchmarkResult> nativeNodes
    ) {
        return String.format(
                Locale.US,
                """
                cpu1 F32 matmul-activation-matmul chain benchmark
                case: %s
                graph: x.matmul(w1) -> %s -> matmul(w2), batch=%d, input=%d, hidden=%d, output=%d, loss/reduction=none
                  openblas: source=%s threadPolicy=%s sgemm=%s
                  warmup=%d, measure=%d, elementwiseWorkers=%d
                  total measurement: rotated end-to-end samples; route order rotates every iteration.
                  total %-28s medianMs=%8.4f ratio=%6.2fx firstOutput=% .8f
                  total %-28s medianMs=%8.4f ratio=%6.2fx firstOutput=% .8f
                  total %-28s medianMs=%8.4f ratio=%6.2fx firstOutput=% .8f cpuMaterializations=0
                  per-node diagnostics: standalone node timings for route inspection only; these medians are not expected to sum to total.
                %s
                %s
                %s
                """,
                chainCase.name(),
                chainCase.activation(),
                chainCase.batch(),
                chainCase.input(),
                chainCase.hidden(),
                chainCase.output(),
                OpenBlasRuntime.lookupSource(),
                OpenBlasRuntime.threadPolicy(),
                OpenBlasRuntime.isFloat32GemmAvailable(),
                profile.warmupIterations(),
                profile.measureIterations(),
                WORKERS,
                array.name(),
                array.medianMs(),
                1.0d,
                array.output()[0],
                javaMatmul.name(),
                javaMatmul.medianMs(),
                array.medianMs() / javaMatmul.medianMs(),
                javaMatmul.output()[0],
                nativeSegment.name(),
                nativeSegment.medianMs(),
                array.medianMs() / nativeSegment.medianMs(),
                nativeSegment.output()[0],
                nodeBreakdownReport("java-array-openblas-copy nodes", arrayNodes),
                nodeBreakdownReport("java-array-java-vector-parallel-matmul nodes", javaMatmulNodes),
                nodeBreakdownReport("all-native-segment-openblas nodes", nativeNodes)
        );
    }

    private static String openBlasThreadReport(
            ChainCase chainCase,
            BenchmarkProfile profile,
            int originalThreads,
            List<OpenBlasThreadBenchmarkResult> results
    ) {
        StringBuilder rows = new StringBuilder();
        double baselineMs = results.getFirst().medianMs();
        for (OpenBlasThreadBenchmarkResult result : results) {
            rows.append(String.format(
                    Locale.US,
                    "  %-18s requested=%-10s effective=%-3d parallel=%d/%-10s medianMs=%8.4f ratio=%6.2fx firstOutput=% .8f%n",
                    result.label(),
                    result.requestedThreads() == null ? "current" : result.requestedThreads().toString(),
                    result.effectiveThreads(),
                    result.parallelMode(),
                    result.parallelModeDescription(),
                    result.medianMs(),
                    baselineMs / result.medianMs(),
                    result.output()[0]
            ));
        }
        return String.format(
                Locale.US,
                """
                cpu1 F32 OpenBLAS thread control chain benchmark
                case: %s
                graph: x.matmul(w1) -> %s -> matmul(w2), batch=%d, input=%d, hidden=%d, output=%d, route=all-native-segment-openblas
                  openblas: source=%s threadPolicy=%s sgemm=%s originalThreads=%d parallel=%s
                  warmup=%d, measure=%d, thread changes use openblas_set_num_threads inside this JVM
                %s
                """,
                chainCase.name(),
                chainCase.activation(),
                chainCase.batch(),
                chainCase.input(),
                chainCase.hidden(),
                chainCase.output(),
                OpenBlasRuntime.lookupSource(),
                OpenBlasRuntime.threadPolicy(),
                OpenBlasRuntime.isFloat32GemmAvailable(),
                originalThreads,
                OpenBlasRuntime.parallelModeDescription(),
                profile.warmupIterations(),
                profile.measureIterations(),
                rows
        );
    }

    private static String threeLayerOpenBlasThreadCountReport(
            MlpCase mlpCase,
            BenchmarkProfile profile,
            int originalThreads,
            int rounds,
            List<ThreeLayerOpenBlasThreadBenchmarkResult> results
    ) {
        StringBuilder rows = new StringBuilder();
        for (int requestedThreads : List.of(4, 16)) {
            List<ThreeLayerOpenBlasThreadBenchmarkResult> threadResults = results.stream()
                    .filter(result -> result.requestedThreads() == requestedThreads)
                    .toList();
            if (threadResults.isEmpty()) {
                continue;
            }
            RouteRoundStats arrayStats = routeRoundStats(threadResults, ThreeLayerRoute.ARRAY);
            RouteRoundStats javaMatmulStats = routeRoundStats(threadResults, ThreeLayerRoute.JAVA_MATMUL);
            RouteRoundStats nativeStats = routeRoundStats(threadResults, ThreeLayerRoute.NATIVE_SEGMENT);
            double baselineMs = arrayStats.medianMs();
            rows.append(String.format(
                    Locale.US,
                    "  requested=%-3d rounds=%d%n",
                    requestedThreads,
                    threadResults.size()
            ));
            appendThreeLayerRouteStatsRow(rows, arrayStats, 1.0d);
            appendThreeLayerRouteStatsRow(rows, javaMatmulStats, baselineMs / javaMatmulStats.medianMs());
            appendThreeLayerRouteStatsRow(rows, nativeStats, baselineMs / nativeStats.medianMs());
            rows.append("    per-round medians:\n");
            for (ThreeLayerOpenBlasThreadBenchmarkResult result : threadResults) {
                BenchmarkTriple benchmark = result.benchmark();
                rows.append(String.format(
                        Locale.US,
                        "      round=%d effective=%-3d parallel=%d/%-10s arrayMs=%8.4f javaVectorMs=%8.4f nativeMs=%8.4f loss=% .8f%n",
                        result.round(),
                        result.effectiveThreads(),
                        result.parallelMode(),
                        result.parallelModeDescription(),
                        benchmark.array().medianMs(),
                        benchmark.javaMatmul().medianMs(),
                        benchmark.nativeSegment().medianMs(),
                        benchmark.array().output()[0]
                ));
            }
        }
        rows.append(threeLayerThreadCountComparison(results));
        return String.format(
                Locale.US,
                """
                cpu1 F32 3-layer MLP OpenBLAS thread-count repeated in-JVM benchmark
                case: %s
                graph: batch=%d, input=%d, hidden1=%d, hidden2=%d, hidden2Activation=%s, output=%d, loss=MSE mean(1).mean(0,true)
                  openblas: source=%s threadPolicy=%s sgemm=%s originalThreads=%d
                  rounds=%d, warmupPerRound=%d, measurePerRound=%d, elementwiseWorkers=%d
                  repeated path: independent in-JVM rounds with fresh fixtures/prepared graphs; this JUnit benchmark does not fork JVMs.
                  thread order rotates by round, and thread changes use openblas_set_num_threads before prepare and measurement.
                  java-array-java-vector-parallel-matmul does not use OpenBLAS; it is included as a side-by-side baseline.
                  summary rows use median/best/worst of each round's end-to-end sample median.
                %s
                """,
                mlpCase.name(),
                mlpCase.batch(),
                mlpCase.input(),
                mlpCase.hidden1(),
                mlpCase.hidden2(),
                mlpCase.hidden2Activation(),
                mlpCase.output(),
                OpenBlasRuntime.lookupSource(),
                OpenBlasRuntime.threadPolicy(),
                OpenBlasRuntime.isFloat32GemmAvailable(),
                originalThreads,
                rounds,
                profile.warmupIterations(),
                profile.measureIterations(),
                WORKERS,
                rows
        );
    }

    private static String forkedThreeLayerOpenBlasThreadCountReport(
            MlpCase mlpCase,
            BenchmarkProfile profile,
            int originalThreads,
            int forks,
            List<ForkedThreeLayerOpenBlasThreadBenchmarkResult> results
    ) {
        StringBuilder rows = new StringBuilder();
        for (int requestedThreads : List.of(4, 16)) {
            List<ForkedThreeLayerOpenBlasThreadBenchmarkResult> threadResults = results.stream()
                    .filter(result -> result.requestedThreads() == requestedThreads)
                    .toList();
            if (threadResults.isEmpty()) {
                continue;
            }
            rows.append(String.format(
                    Locale.US,
                    "  requested=%-3d forks=%d%n",
                    requestedThreads,
                    distinctForkCount(threadResults)
            ));
            for (ThreeLayerRoute route : ThreeLayerRoute.values()) {
                ForkedRouteStats stats = forkedRouteStats(threadResults, route);
                double baselineMs = forkedRouteStats(threadResults, ThreeLayerRoute.ARRAY).medianMs();
                appendForkedThreeLayerRouteStatsRow(rows, stats, baselineMs / stats.medianMs());
            }
            rows.append("    per-fork medians:\n");
            for (ForkedThreeLayerOpenBlasThreadBenchmarkResult result : threadResults) {
                rows.append(String.format(
                        Locale.US,
                        "      fork=%d effective=%-3d parallel=%d/%-10s route=%-38s medianMs=%8.4f firstOutput=% .8f%n",
                        result.fork(),
                        result.effectiveThreads(),
                        result.parallelMode(),
                        result.parallelModeDescription(),
                        result.route(),
                        result.medianMs(),
                        result.firstOutput()
                ));
            }
        }
        rows.append(forkedThreeLayerThreadCountComparison(results));
        return String.format(
                Locale.US,
                """
                cpu1 F32 3-layer MLP OpenBLAS thread-count forked JVM benchmark
                case: %s
                graph: batch=%d, input=%d, hidden1=%d, hidden2=%d, hidden2Activation=%s, output=%d, loss=MSE mean(1).mean(0,true)
                  parent openblas: source=%s threadPolicy=%s sgemm=%s originalThreads=%d
                  forks=%d, warmupPerFork=%d, measurePerFork=%d, elementwiseWorkers=%d
                  fork command: current java executable, current java.class.path, --add-modules=jdk.incubator.vector, --enable-native-access=ALL-UNNAMED
                  each fork runs OpenBLAS thread counts 4 and 16 across java-array-openblas-copy, java-array-java-vector-parallel-matmul, and all-native-segment-openblas.
                  summary rows use median/best/worst of each fork's end-to-end sample median.
                %s
                """,
                mlpCase.name(),
                mlpCase.batch(),
                mlpCase.input(),
                mlpCase.hidden1(),
                mlpCase.hidden2(),
                mlpCase.hidden2Activation(),
                mlpCase.output(),
                OpenBlasRuntime.lookupSource(),
                OpenBlasRuntime.threadPolicy(),
                OpenBlasRuntime.isFloat32GemmAvailable(),
                originalThreads,
                forks,
                profile.warmupIterations(),
                profile.measureIterations(),
                WORKERS,
                rows
        );
    }

    private static void appendThreeLayerRouteStatsRow(StringBuilder rows, RouteRoundStats result, double ratio) {
        rows.append(String.format(
                Locale.US,
                "    total %-38s medianOfRoundMediansMs=%8.4f bestRoundMedianMs=%8.4f worstRoundMedianMs=%8.4f spread=%5.1f%% ratio=%6.2fx%n",
                result.name(),
                result.medianMs(),
                result.bestMs(),
                result.worstMs(),
                result.spreadPercent(),
                ratio
        ));
    }

    private static void appendForkedThreeLayerRouteStatsRow(StringBuilder rows, ForkedRouteStats result, double ratio) {
        rows.append(String.format(
                Locale.US,
                "    total %-38s medianOfForkMediansMs=%8.4f bestForkMedianMs=%8.4f worstForkMedianMs=%8.4f spread=%5.1f%% ratio=%6.2fx%n",
                result.name(),
                result.medianMs(),
                result.bestMs(),
                result.worstMs(),
                result.spreadPercent(),
                ratio
        ));
    }

    private static String threeLayerThreadCountComparison(List<ThreeLayerOpenBlasThreadBenchmarkResult> results) {
        List<ThreeLayerOpenBlasThreadBenchmarkResult> fourThreadResults = results.stream()
                .filter(result -> result.requestedThreads() == 4)
                .toList();
        List<ThreeLayerOpenBlasThreadBenchmarkResult> sixteenThreadResults = results.stream()
                .filter(result -> result.requestedThreads() == 16)
                .toList();
        if (fourThreadResults.isEmpty() || sixteenThreadResults.isEmpty()) {
            return "";
        }

        StringBuilder rows = new StringBuilder("  16-vs-4 thread-count comparison, ratio > 1.0 means 16 threads was faster:\n");
        for (ThreeLayerRoute route : ThreeLayerRoute.values()) {
            RouteRoundStats four = routeRoundStats(fourThreadResults, route);
            RouteRoundStats sixteen = routeRoundStats(sixteenThreadResults, route);
            rows.append(String.format(
                    Locale.US,
                    "    %-38s 4threadMedianMs=%8.4f 16threadMedianMs=%8.4f speedup=%6.2fx%n",
                    four.name(),
                    four.medianMs(),
                    sixteen.medianMs(),
                    four.medianMs() / sixteen.medianMs()
            ));
        }
        return rows.toString();
    }

    private static String forkedThreeLayerThreadCountComparison(List<ForkedThreeLayerOpenBlasThreadBenchmarkResult> results) {
        List<ForkedThreeLayerOpenBlasThreadBenchmarkResult> fourThreadResults = results.stream()
                .filter(result -> result.requestedThreads() == 4)
                .toList();
        List<ForkedThreeLayerOpenBlasThreadBenchmarkResult> sixteenThreadResults = results.stream()
                .filter(result -> result.requestedThreads() == 16)
                .toList();
        if (fourThreadResults.isEmpty() || sixteenThreadResults.isEmpty()) {
            return "";
        }

        StringBuilder rows = new StringBuilder("  16-vs-4 thread-count comparison, ratio > 1.0 means 16 threads was faster:\n");
        for (ThreeLayerRoute route : ThreeLayerRoute.values()) {
            ForkedRouteStats four = forkedRouteStats(fourThreadResults, route);
            ForkedRouteStats sixteen = forkedRouteStats(sixteenThreadResults, route);
            rows.append(String.format(
                    Locale.US,
                    "    %-38s 4threadMedianMs=%8.4f 16threadMedianMs=%8.4f speedup=%6.2fx%n",
                    four.name(),
                    four.medianMs(),
                    sixteen.medianMs(),
                    four.medianMs() / sixteen.medianMs()
            ));
        }
        return rows.toString();
    }

    private static RouteRoundStats routeRoundStats(
            List<ThreeLayerOpenBlasThreadBenchmarkResult> results,
            ThreeLayerRoute route
    ) {
        double[] medians = new double[results.size()];
        String name = null;
        for (int i = 0; i < results.size(); i++) {
            BenchmarkResult routeResult = routeResult(results.get(i).benchmark(), route);
            medians[i] = routeResult.medianMs();
            name = routeResult.name();
        }
        Arrays.sort(medians);
        double median = median(medians);
        double best = medians[0];
        double worst = medians[medians.length - 1];
        double spreadPercent = best == 0.0d ? 0.0d : ((worst - best) / best) * 100.0d;
        return new RouteRoundStats(name, median, best, worst, spreadPercent);
    }

    private static ForkedRouteStats forkedRouteStats(
            List<ForkedThreeLayerOpenBlasThreadBenchmarkResult> results,
            ThreeLayerRoute route
    ) {
        String routeName = routeName(route);
        double[] medians = results.stream()
                .filter(result -> result.route().equals(routeName))
                .mapToDouble(ForkedThreeLayerOpenBlasThreadBenchmarkResult::medianMs)
                .toArray();
        assertTrue(medians.length > 0, "missing forked results for route " + routeName);
        Arrays.sort(medians);
        double median = median(medians);
        double best = medians[0];
        double worst = medians[medians.length - 1];
        double spreadPercent = best == 0.0d ? 0.0d : ((worst - best) / best) * 100.0d;
        return new ForkedRouteStats(routeName, median, best, worst, spreadPercent);
    }

    private static long distinctForkCount(List<ForkedThreeLayerOpenBlasThreadBenchmarkResult> results) {
        return results.stream()
                .map(ForkedThreeLayerOpenBlasThreadBenchmarkResult::fork)
                .distinct()
                .count();
    }

    private static BenchmarkResult routeResult(BenchmarkTriple benchmark, ThreeLayerRoute route) {
        return switch (route) {
            case ARRAY -> benchmark.array();
            case JAVA_MATMUL -> benchmark.javaMatmul();
            case NATIVE_SEGMENT -> benchmark.nativeSegment();
        };
    }

    private static String routeName(ThreeLayerRoute route) {
        return switch (route) {
            case ARRAY -> "java-array-openblas-copy";
            case JAVA_MATMUL -> "java-array-java-vector-parallel-matmul";
            case NATIVE_SEGMENT -> "all-native-segment-openblas";
        };
    }

    private static double median(double[] sortedSamples) {
        int middle = sortedSamples.length / 2;
        if (sortedSamples.length % 2 == 1) {
            return sortedSamples[middle];
        }
        return (sortedSamples[middle - 1] + sortedSamples[middle]) / 2.0d;
    }

    private static String nodeBreakdownReport(String title, List<NodeBenchmarkResult> nodes) {
        StringBuilder builder = new StringBuilder();
        builder.append("  ").append(title).append('\n');
        for (NodeBenchmarkResult node : nodes) {
            builder.append(String.format(
                    Locale.US,
                    "    node=%-3d op=%-8s shape=%-14s kernel=%-48s medianMs=%8.4f%n",
                    node.nodeId(),
                    node.opType(),
                    node.shape(),
                    node.kernel(),
                    node.medianMs()
            ));
        }
        return builder.toString();
    }

    private static CompiledNodeExecutionMetadata metadata(CompiledNode node, Cpu1PreparedArtifact artifact) {
        return new CompiledNodeExecutionMetadata(
                ComputeBackend.CPU,
                null,
                node.inputIds(),
                artifact
        );
    }

    private static List<OpenBlasThreadCase> openBlasThreadCases(int currentThreads) {
        List<OpenBlasThreadCase> cases = new ArrayList<>();
        cases.add(new OpenBlasThreadCase("default/current", null));
        addOpenBlasThreadCase(cases, currentThreads, "set-1", 1);
        addOpenBlasThreadCase(cases, currentThreads, "set-4", 4);
        int javaAvailableProcessors = Runtime.getRuntime().availableProcessors();
        addOpenBlasThreadCase(
                cases,
                currentThreads,
                "set-java-available-processors",
                javaAvailableProcessors
        );
        if (javaAvailableProcessors >= 16) {
            addOpenBlasThreadCase(cases, currentThreads, "set-16", 16);
        }
        return cases;
    }

    private static void addOpenBlasThreadCase(
            List<OpenBlasThreadCase> cases,
            int currentThreads,
            String label,
            int requestedThreads
    ) {
        if (requestedThreads == currentThreads) {
            return;
        }
        for (OpenBlasThreadCase threadCase : cases) {
            if (threadCase.requestedThreads() != null && threadCase.requestedThreads() == requestedThreads) {
                return;
            }
        }
        cases.add(new OpenBlasThreadCase(label, requestedThreads));
    }

    private static List<Integer> threeLayerThreadCountOrder(int round) {
        return round % 2 == 1 ? List.of(4, 16) : List.of(16, 4);
    }

    private enum RouteKind {
        JAVA_ARRAY_OPENBLAS,
        JAVA_ARRAY_JAVA_MATMUL,
        NATIVE_SEGMENT_OPENBLAS
    }

    private enum ActivationKind {
        TANH,
        RELU
    }

    private enum ThreeLayerRoute {
        ARRAY,
        JAVA_MATMUL,
        NATIVE_SEGMENT
    }

    private record BenchmarkProfile(int warmupIterations, int measureIterations) {
    }

    private record MlpCase(
            String name,
            int batch,
            int input,
            int hidden1,
            int hidden2,
            int output,
            ActivationKind hidden2Activation
    ) {
    }

    private record ChainCase(
            String name,
            int batch,
            int input,
            int hidden,
            int output,
            ActivationKind activation
    ) {
    }

    private record LeafData(float[] values) {
    }

    private record MlpFixture(
            Tensor root,
            List<CompiledNode> nodes,
            CompiledTensorDescriptorIndex descriptorIndex,
            CompiledNode rootNode,
            Map<String, LeafData> leaves
    ) {
    }

    private record PreparedGraph(
            MlpFixture fixture,
            Map<Integer, CompiledNodeExecutionMetadata> metadata,
            ExecutionContext context,
            ExecutionState state
    ) {
    }

    private record NativeBenchmarkGuard(
            NativeTensorStorage expectedRootOutput,
            Map<Integer, NativeTensorStorage> expectedOutputStorages,
            long allocationCountAfterWarmup
    ) {
        private void assertStillValid(PreparedGraph graph) {
            assertSame(expectedRootOutput, graph.context().nativeStorageForNodeId(graph.fixture().rootNode().id()));
            assertNativeOutputStoragesSame(graph, expectedOutputStorages);
            assertEquals(allocationCountAfterWarmup, graph.state().nativeCpuMemoryTrace().allocationCount());
        }
    }

    private record BenchmarkTriple(
            BenchmarkResult array,
            BenchmarkResult javaMatmul,
            BenchmarkResult nativeSegment
    ) {
    }

    private record BenchmarkResult(
            String name,
            double medianMs,
            float[] output
    ) {
    }

    private record NodeBenchmarkResult(
            int nodeId,
            String opType,
            String shape,
            String kernel,
            double medianMs
    ) {
    }

    private record OpenBlasThreadCase(
            String label,
            Integer requestedThreads
    ) {
    }

    private record OpenBlasThreadBenchmarkResult(
            String label,
            Integer requestedThreads,
            int effectiveThreads,
            int parallelMode,
            String parallelModeDescription,
            double medianMs,
            float[] output
    ) {
    }

    private record ThreeLayerOpenBlasThreadBenchmarkResult(
            int round,
            int requestedThreads,
            int effectiveThreads,
            int parallelMode,
            String parallelModeDescription,
            BenchmarkTriple benchmark
    ) {
    }

    private record RouteRoundStats(
            String name,
            double medianMs,
            double bestMs,
            double worstMs,
            double spreadPercent
    ) {
    }

    private record ForkedThreeLayerOpenBlasThreadBenchmarkResult(
            int fork,
            int requestedThreads,
            int effectiveThreads,
            int parallelMode,
            String parallelModeDescription,
            String route,
            double medianMs,
            float firstOutput
    ) {
    }

    private record ForkedRouteStats(
            String name,
            double medianMs,
            double bestMs,
            double worstMs,
            double spreadPercent
    ) {
    }

    public static final class ForkedThreeLayerMlpBenchmarkMain {
        private ForkedThreeLayerMlpBenchmarkMain() {
        }

        public static void main(String[] args) {
            try {
                int fork = args.length == 0 ? 1 : Integer.parseInt(args[0]);
                runThreeLayerMlpOpenBlasThreadCountForkMain(fork);
            } catch (Throwable t) {
                t.printStackTrace(System.err);
                System.exit(1);
            }
        }
    }
}

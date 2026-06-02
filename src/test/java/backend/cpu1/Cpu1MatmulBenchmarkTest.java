package backend.cpu1;

import backend.ComputeBackend;
import backend.blas.OpenBlasRuntime;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.kernels.matmul.Cpu1MatmulKernelId;
import backend.cpu1.prepare.Cpu1NodePreparer;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.cpu1.provider.matmul.Cpu1MatmulRoute;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import config.runtime.RuntimeConfig;
import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.compile.intent.BackendIntentPlan;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.state.ExecutionState;
import graph.execution.trace.StepTraceContribution;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

@Tag("benchmark")
class Cpu1MatmulBenchmarkTest {
    private static final int WARMUP_ITERATIONS = 4;
    private static final int MEASURE_ITERATIONS = 12;
    private static final int M = 96;
    private static final int K = 130;
    private static final int N = 96;

    @Test
    void benchmarkF32DenseMatmulExplicitScalarVectorAndOpenBlasArrayCopying() {
        Assumptions.assumeTrue(
                OpenBlasRuntime.isFloat32GemmAvailable(),
                "OpenBLAS sgemm unavailable: " + OpenBlasRuntime.unavailableReason()
        );

        BenchmarkResult scalar = benchmark(
                "java-scalar",
                preparedFixture(
                        DataType.FLOAT32,
                        Cpu1PrepareConfig.scalarSingleThread(),
                        Cpu1MatmulRoute.JAVA_SCALAR,
                        Cpu1MatmulKernelId.MATMUL_F32_DENSE_SCALAR
                )
        );
        BenchmarkResult vector = benchmark(
                "java-vector-packed-b",
                preparedFixture(
                        DataType.FLOAT32,
                        Cpu1PrepareConfig.vectorSingleThread(),
                        Cpu1MatmulRoute.JAVA_SCALAR,
                        Cpu1MatmulKernelId.MATMUL_F32_DENSE_PACKED_B_VECTOR
                )
        );
        BenchmarkResult openBlas = benchmark(
                "openblas-array-copying",
                preparedFixture(
                        DataType.FLOAT32,
                        Cpu1PrepareConfig.scalarSingleThread()
                                .withMatmulRoute(Cpu1MatmulRoute.OPENBLAS_ARRAY_COPYING),
                        Cpu1MatmulRoute.OPENBLAS_ARRAY_COPYING,
                        Cpu1MatmulKernelId.MATMUL_F32_OPENBLAS_ARRAY_COPYING
                )
        );

        assertArrayEquals(scalar.output(), vector.output(), 1.0e-3d);
        assertArrayEquals(scalar.output(), openBlas.output(), 1.0e-3d);
        System.out.println(report(DataType.FLOAT32, scalar, vector, openBlas));
    }

    @Test
    void benchmarkF64DenseMatmulExplicitScalarVectorAndOpenBlasArrayCopying() {
        Assumptions.assumeTrue(
                OpenBlasRuntime.isFloat64GemmAvailable(),
                "OpenBLAS dgemm unavailable: " + OpenBlasRuntime.unavailableReason()
        );

        BenchmarkResult scalar = benchmark(
                "java-scalar",
                preparedFixture(
                        DataType.FLOAT64,
                        Cpu1PrepareConfig.scalarSingleThread(),
                        Cpu1MatmulRoute.JAVA_SCALAR,
                        Cpu1MatmulKernelId.MATMUL_F64_DENSE_SCALAR
                )
        );
        BenchmarkResult vector = benchmark(
                "java-vector-packed-b",
                preparedFixture(
                        DataType.FLOAT64,
                        Cpu1PrepareConfig.vectorSingleThread(),
                        Cpu1MatmulRoute.JAVA_SCALAR,
                        Cpu1MatmulKernelId.MATMUL_F64_DENSE_PACKED_B_VECTOR
                )
        );
        BenchmarkResult openBlas = benchmark(
                "openblas-array-copying",
                preparedFixture(
                        DataType.FLOAT64,
                        Cpu1PrepareConfig.scalarSingleThread()
                                .withMatmulRoute(Cpu1MatmulRoute.OPENBLAS_ARRAY_COPYING),
                        Cpu1MatmulRoute.OPENBLAS_ARRAY_COPYING,
                        Cpu1MatmulKernelId.MATMUL_F64_OPENBLAS_ARRAY_COPYING
                )
        );

        assertArrayEquals(scalar.output(), vector.output(), 1.0e-9d);
        assertArrayEquals(scalar.output(), openBlas.output(), 1.0e-9d);
        System.out.println(report(DataType.FLOAT64, scalar, vector, openBlas));
    }

    private static PreparedFixture preparedFixture(
            DataType dataType,
            Cpu1PrepareConfig config,
            Cpu1MatmulRoute expectedRoute,
            Cpu1MatmulKernelId expectedKernelId
    ) {
        Fixture fixture = fixture(matmulTensor(dataType));
        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex(), config);
        assertEquals(expectedKernelId, artifact.preparedMatmulUnit().kernelId());
        assertEquals(expectedRoute, artifact.preparedMatmulUnit().route());
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, metadata);
        StepTraceContribution trace = artifact.traceContribution(fixture.node(), metadata, context);
        assertEquals(expectedKernelId.name(), trace.kernel());
        assertNotNull(trace.matMul());
        assertEquals(expectedRoute.name(), trace.matMul().route());
        if (expectedRoute == Cpu1MatmulRoute.OPENBLAS_ARRAY_COPYING) {
            assertTrue(trace.matMul().useBlas());
            assertEquals(expectedBlasSymbol(dataType), trace.matMul().blasSymbol());
            assertEquals(Cpu1MatmulRoute.OPENBLAS_ARRAY_COPYING.name(), trace.matMul().blasRoute());
        }
        return new PreparedFixture(fixture, metadata, artifact, context);
    }

    private static BenchmarkResult benchmark(String name, PreparedFixture fixture) {
        Cpu1Backend backend = new Cpu1Backend();
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            backend.execute(fixture.fixture().node(), fixture.metadata(), fixture.context());
        }
        long[] samples = new long[MEASURE_ITERATIONS];
        for (int i = 0; i < MEASURE_ITERATIONS; i++) {
            long start = System.nanoTime();
            backend.execute(fixture.fixture().node(), fixture.metadata(), fixture.context());
            samples[i] = System.nanoTime() - start;
        }
        StepTraceContribution trace = fixture.artifact()
                .traceContribution(fixture.fixture().node(), fixture.metadata(), fixture.context());
        return new BenchmarkResult(
                name,
                fixture.artifact().preparedMatmulUnit().route(),
                fixture.artifact().preparedMatmulUnit().vectorizationKind(),
                fixture.artifact().preparedMatmulUnit().kernelId(),
                trace.kernel(),
                trace.matMul().route(),
                trace.matMul().blasSymbol(),
                trace.matMul().microKernel(),
                (Integer) trace.attributes().get("cpu1MatmulVectorWidth"),
                fixture.artifact().preparedMatmulUnit().launchConfig().workerCount(),
                medianMs(samples),
                output(fixture)
        );
    }

    private static Tensor matmulTensor(DataType dataType) {
        return switch (dataType) {
            case FLOAT32 -> matmulF32Tensor();
            case FLOAT64 -> matmulF64Tensor();
            default -> throw new IllegalArgumentException("Unsupported benchmark dtype: " + dataType);
        };
    }

    private static Tensor matmulF32Tensor() {
        float[] leftData = new float[M * K];
        float[] rightData = new float[K * N];
        for (int i = 0; i < leftData.length; i++) {
            leftData[i] = f32Value(i, 17, 8, 0.125f);
        }
        for (int i = 0; i < rightData.length; i++) {
            rightData[i] = f32Value(i, 19, 9, 0.0625f);
        }
        Tensor left = new Tensor(leftData, new int[]{M, K}, null, "matmul-benchmark-left", DataType.FLOAT32);
        Tensor right = new Tensor(rightData, new int[]{K, N}, null, "matmul-benchmark-right", DataType.FLOAT32);
        return left.matmul(right);
    }

    private static Tensor matmulF64Tensor() {
        double[] leftData = new double[M * K];
        double[] rightData = new double[K * N];
        for (int i = 0; i < leftData.length; i++) {
            leftData[i] = f64Value(i, 17, 8, 0.125d);
        }
        for (int i = 0; i < rightData.length; i++) {
            rightData[i] = f64Value(i, 19, 9, 0.0625d);
        }
        Tensor left = new Tensor(leftData, new int[]{M, K}, null, "matmul-benchmark-left", DataType.FLOAT64);
        Tensor right = new Tensor(rightData, new int[]{K, N}, null, "matmul-benchmark-right", DataType.FLOAT64);
        return left.matmul(right);
    }

    private static float f32Value(int index, int modulus, int center, float scale) {
        return ((index % modulus) - center) * scale;
    }

    private static double f64Value(int index, int modulus, int center, double scale) {
        return ((index % modulus) - center) * scale;
    }

    private static double[] output(PreparedFixture fixture) {
        Tensor output = fixture.context().runtimeTensorForNodeId(fixture.fixture().node().id());
        return switch (fixture.artifact().preparedMatmulUnit().dataType()) {
            case FLOAT32 -> toDoubleArray(output.toFloat32ArrayCopy());
            case FLOAT64 -> output.toFloat64ArrayCopy();
            default -> throw new IllegalArgumentException(
                    "Unsupported benchmark dtype: " + fixture.artifact().preparedMatmulUnit().dataType()
            );
        };
    }

    private static double[] toDoubleArray(float[] values) {
        double[] result = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            result[i] = values[i];
        }
        return result;
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
            DataType dataType,
            BenchmarkResult scalar,
            BenchmarkResult vector,
            BenchmarkResult openBlas
    ) {
        return String.format(
                Locale.US,
                """
                cpu1 %s dense matmul route benchmark
                  shape: left=[%d,%d], right=[%d,%d], outputElements=%d
                  openblas: source=%s threadPolicy=%s sgemm=%s dgemm=%s
                  warmup=%d, measure=%d
                  %-24s preparedRoute=%-23s traceRoute=%-23s vector=%-6s microKernel=%-22s kernel=%-38s traceKernel=%-38s blas=%-12s vecWidth=%2d workers=%2d medianMs=%8.4f speedup=%6.2fx
                  %-24s preparedRoute=%-23s traceRoute=%-23s vector=%-6s microKernel=%-22s kernel=%-38s traceKernel=%-38s blas=%-12s vecWidth=%2d workers=%2d medianMs=%8.4f speedup=%6.2fx
                  %-24s preparedRoute=%-23s traceRoute=%-23s vector=%-6s microKernel=%-22s kernel=%-38s traceKernel=%-38s blas=%-12s vecWidth=%2d workers=%2d medianMs=%8.4f speedup=%6.2fx
                """,
                dataType,
                M,
                K,
                K,
                N,
                scalar.output().length,
                OpenBlasRuntime.lookupSource(),
                OpenBlasRuntime.threadPolicy(),
                OpenBlasRuntime.isFloat32GemmAvailable(),
                OpenBlasRuntime.isFloat64GemmAvailable(),
                WARMUP_ITERATIONS,
                MEASURE_ITERATIONS,
                scalar.name(),
                scalar.route(),
                scalar.traceRoute(),
                scalar.vectorizationKind(),
                scalar.microKernel(),
                scalar.kernelId(),
                scalar.traceKernel(),
                scalar.blasSymbol(),
                scalar.vectorWidth(),
                scalar.workers(),
                scalar.medianMs(),
                1.0d,
                vector.name(),
                vector.route(),
                vector.traceRoute(),
                vector.vectorizationKind(),
                vector.microKernel(),
                vector.kernelId(),
                vector.traceKernel(),
                vector.blasSymbol(),
                vector.vectorWidth(),
                vector.workers(),
                vector.medianMs(),
                scalar.medianMs() / vector.medianMs(),
                openBlas.name(),
                openBlas.route(),
                openBlas.traceRoute(),
                openBlas.vectorizationKind(),
                openBlas.microKernel(),
                openBlas.kernelId(),
                openBlas.traceKernel(),
                openBlas.blasSymbol(),
                openBlas.vectorWidth(),
                openBlas.workers(),
                openBlas.medianMs(),
                scalar.medianMs() / openBlas.medianMs()
        );
    }

    private static String expectedBlasSymbol(DataType dataType) {
        return switch (dataType) {
            case FLOAT32 -> "cblas_sgemm";
            case FLOAT64 -> "cblas_dgemm";
            default -> throw new IllegalArgumentException("Unsupported OpenBLAS benchmark dtype: " + dataType);
        };
    }

    private static Fixture fixture(Tensor out) {
        List<CompiledNode> nodes = CompiledNode.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
        CompiledTensorDescriptorIndex descriptorIndex = CompiledTensorDescriptorBuilder.build(nodes);
        return new Fixture(out, nodes, descriptorIndex, nodes.getLast());
    }

    private static ExecutionContext context(Fixture fixture, CompiledNodeExecutionMetadata metadata) {
        Map<Integer, CompiledNodeExecutionMetadata> metadataIndex = Map.of(fixture.node().id(), metadata);
        ExecutionState state = ExecutionState.create(
                fixture.nodes(),
                fixture.descriptorIndex(),
                metadataIndex,
                fixture.node().id(),
                testsupport.PublicationPlans.forRoot(fixture.root(), fixture.nodes(), fixture.node().id())
        );
        return ExecutionContext.fromRuntimeConfig(
                RuntimeConfig.inferenceDefaults(),
                ExecutionMode.FORWARD,
                metadataIndex,
                state
        );
    }

    private static CompiledNodeExecutionMetadata metadata(CompiledNode node, Cpu1PreparedArtifact artifact) {
        return new CompiledNodeExecutionMetadata(
                ComputeBackend.CPU,
                null,
                node.inputIds(),
                artifact
        );
    }

    private record Fixture(
            Tensor root,
            List<CompiledNode> nodes,
            CompiledTensorDescriptorIndex descriptorIndex,
            CompiledNode node
    ) {
    }

    private record PreparedFixture(
            Fixture fixture,
            CompiledNodeExecutionMetadata metadata,
            Cpu1PreparedArtifact artifact,
            ExecutionContext context
    ) {
    }

    private record BenchmarkResult(
            String name,
            Cpu1MatmulRoute route,
            Cpu1VectorizationKind vectorizationKind,
            Cpu1MatmulKernelId kernelId,
            String traceKernel,
            String traceRoute,
            String blasSymbol,
            String microKernel,
            int vectorWidth,
            int workers,
            double medianMs,
            double[] output
    ) {
    }
}

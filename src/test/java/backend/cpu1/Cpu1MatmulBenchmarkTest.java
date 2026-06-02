package backend.cpu1;

import backend.ComputeBackend;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.kernels.matmul.Cpu1MatmulKernelId;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.prepare.Cpu1NodePreparer;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.prepare.Cpu1PreparedArtifact;
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

@Tag("benchmark")
class Cpu1MatmulBenchmarkTest {
    private static final int WARMUP_ITERATIONS = 4;
    private static final int MEASURE_ITERATIONS = 12;
    private static final int M = 96;
    private static final int K = 130;
    private static final int N = 96;

    @Test
    void benchmarkF32DenseMatmulExplicitScalarVectorAndParallelVector() {
        BenchmarkResult scalar = benchmark(
                "scalar-single-thread",
                preparedFixture(Cpu1PrepareConfig.scalarSingleThread(), Cpu1MatmulKernelId.MATMUL_F32_DENSE_SCALAR)
        );
        BenchmarkResult vector = benchmark(
                "vector-single-thread",
                preparedFixture(Cpu1PrepareConfig.vectorSingleThread(), Cpu1MatmulKernelId.MATMUL_F32_DENSE_PACKED_B_VECTOR)
        );
        BenchmarkResult vectorParallel = benchmark(
                "vector-parallel",
                preparedFixture(
                        new Cpu1PrepareConfig(
                                Cpu1VectorizationKind.VECTOR,
                                Cpu1LaunchConfig.parallel(parallelWorkers()),
                                Cpu1StorageKind.JAVA_ARRAY
                        ),
                        Cpu1MatmulKernelId.MATMUL_F32_DENSE_PACKED_B_VECTOR
                )
        );

        assertArrayEquals(scalar.output(), vector.output(), 1.0e-4f);
        assertArrayEquals(scalar.output(), vectorParallel.output(), 1.0e-4f);
        System.out.println(report(scalar, vector, vectorParallel));
    }

    private static PreparedFixture preparedFixture(Cpu1PrepareConfig config, Cpu1MatmulKernelId expectedKernelId) {
        Fixture fixture = fixture(matmulTensor());
        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex(), config);
        assertEquals(expectedKernelId, artifact.preparedMatmulUnit().kernelId());
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, metadata);
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
        float[] output = fixture.context()
                .runtimeTensorForNodeId(fixture.fixture().node().id())
                .toFloat32ArrayCopy();
        return new BenchmarkResult(
                name,
                fixture.artifact().preparedMatmulUnit().kernelId(),
                fixture.artifact().preparedMatmulUnit().launchConfig().workerCount(),
                medianMs(samples),
                output
        );
    }

    private static Tensor matmulTensor() {
        float[] leftData = new float[M * K];
        float[] rightData = new float[K * N];
        for (int i = 0; i < leftData.length; i++) {
            leftData[i] = ((i % 17) - 8) * 0.125f;
        }
        for (int i = 0; i < rightData.length; i++) {
            rightData[i] = ((i % 19) - 9) * 0.0625f;
        }
        Tensor left = new Tensor(leftData, new int[]{M, K}, null, "matmul-benchmark-left", DataType.FLOAT32);
        Tensor right = new Tensor(rightData, new int[]{K, N}, null, "matmul-benchmark-right", DataType.FLOAT32);
        return left.matmul(right);
    }

    private static int parallelWorkers() {
        return Math.max(2, Math.min(4, Runtime.getRuntime().availableProcessors()));
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

    private static String report(BenchmarkResult scalar, BenchmarkResult vector, BenchmarkResult vectorParallel) {
        return String.format(
                Locale.US,
                """
                cpu1 F32 dense matmul benchmark
                  shape: left=[%d,%d], right=[%d,%d], outputElements=%d
                  warmup=%d, measure=%d
                  %-22s kernel=%-34s workers=%2d medianMs=%8.4f speedup=%6.2fx
                  %-22s kernel=%-34s workers=%2d medianMs=%8.4f speedup=%6.2fx
                  %-22s kernel=%-34s workers=%2d medianMs=%8.4f speedup=%6.2fx
                """,
                M,
                K,
                K,
                N,
                scalar.output().length,
                WARMUP_ITERATIONS,
                MEASURE_ITERATIONS,
                scalar.name(),
                scalar.kernelId(),
                scalar.workers(),
                scalar.medianMs(),
                1.0d,
                vector.name(),
                vector.kernelId(),
                vector.workers(),
                vector.medianMs(),
                scalar.medianMs() / vector.medianMs(),
                vectorParallel.name(),
                vectorParallel.kernelId(),
                vectorParallel.workers(),
                vectorParallel.medianMs(),
                scalar.medianMs() / vectorParallel.medianMs()
        );
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
            Cpu1MatmulKernelId kernelId,
            int workers,
            double medianMs,
            float[] output
    ) {
    }
}

package backend.cpu1;

import backend.contract.ComputeBackend;
import backend.cpu1.exec.Cpu1ScratchBufferSpec;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.kernels.layout.Cpu1LayoutKernelId;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.prepare.Cpu1NodePreparer;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.cpu1.prepare.Cpu1PreparedLayoutUnit;
import backend.cpu1.storage.Cpu1StorageKind;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import config.runtime.RuntimeConfig;
import graph.compile.CompiledNodeSnapshotter;
import graph.model.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.compile.intent.BackendIntentPlan;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.state.ExecutionState;
import operations.Operation;
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
class Cpu1LayoutTileBenchmarkTest {
    private static final int WARMUP_ITERATIONS = 10;
    private static final int MEASURE_ITERATIONS = 40;

    @Test
    void benchmarkDenseMultiAxisTileAgainstGenericScalar() {
        BenchmarkResult generic = benchmark(
                "generic-scalar",
                preparedFixture(Cpu1LayoutKernelId.TILE_COPY_SCALAR, Cpu1VectorizationKind.SCALAR)
        );
        BenchmarkResult denseScalar = benchmark(
                "dense-multi-axis-scalar",
                preparedFixture(Cpu1LayoutKernelId.TILE_DENSE_MULTI_AXIS_BLOCK_COPY_SCALAR, Cpu1VectorizationKind.SCALAR)
        );
        BenchmarkResult denseVector = benchmark(
                "dense-multi-axis-vector",
                preparedFixture(Cpu1LayoutKernelId.TILE_DENSE_MULTI_AXIS_BLOCK_COPY_VECTOR, Cpu1VectorizationKind.VECTOR)
        );

        assertArrayEquals(generic.output(), denseScalar.output(), 0.0f);
        assertArrayEquals(generic.output(), denseVector.output(), 0.0f);
        System.out.println(report(generic, denseScalar, denseVector));
    }

    private static PreparedFixture preparedFixture(Cpu1LayoutKernelId kernelId, Cpu1VectorizationKind vectorizationKind) {
        Fixture fixture = fixture(tiledTensor());
        Cpu1PreparedArtifact artifact;
        if (kernelId == Cpu1LayoutKernelId.TILE_COPY_SCALAR) {
            artifact = forcedLayoutArtifact(fixture, kernelId, vectorizationKind);
        } else {
            Cpu1PrepareConfig config = vectorizationKind == Cpu1VectorizationKind.VECTOR
                    ? Cpu1PrepareConfig.vectorSingleThread()
                    : Cpu1PrepareConfig.scalarSingleThread();
            artifact = new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex(), config);
            assertEquals(kernelId, artifact.preparedLayoutUnit().kernelId());
        }
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, metadata);
        return new PreparedFixture(fixture, metadata, context);
    }

    private static Cpu1PreparedArtifact forcedLayoutArtifact(
            Fixture fixture,
            Cpu1LayoutKernelId kernelId,
            Cpu1VectorizationKind vectorizationKind
    ) {
        CompiledNode node = fixture.node();
        Cpu1PreparedLayoutUnit unit = new Cpu1PreparedLayoutUnit(
                node.id(),
                node.inputIds(),
                Operation.OpType.TILE,
                node.dataType(),
                Cpu1StorageKind.JAVA_ARRAY,
                kernelId,
                RuntimeConfig.inferenceDefaults(node.dataType()).cpuKernelConfig().contiguousMaterializeThreshold(),
                vectorizationKind,
                Cpu1LaunchConfig.singleThread(),
                Cpu1ScratchBufferSpec.none(),
                -1,
                new int[0],
                new int[0],
                0.0d,
                -1,
                0,
                0,
                null,
                new int[0],
                new int[0],
                new int[0],
                new int[0]
        );
        return new Cpu1PreparedArtifact(unit);
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
        return new BenchmarkResult(name, medianMs(samples), output);
    }

    private static Tensor tiledTensor() {
        int rows = 256;
        int columns = 256;
        float[] values = new float[rows * columns];
        for (int i = 0; i < values.length; i++) {
            values[i] = (i % 1024) * 0.25f;
        }
        Tensor input = new Tensor(values, new int[]{rows, columns}, null, "tile-benchmark-input", DataType.FLOAT32);
        return input.tile(2, 3);
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

    private static String report(BenchmarkResult generic, BenchmarkResult denseScalar, BenchmarkResult denseVector) {
        return String.format(
                Locale.US,
                """
                cpu1 dense multi-axis tile benchmark
                  shape: input=[256,256], repeats=[2,3], outputElements=%d
                  warmup=%d, measure=%d
                  %-26s medianMs=%8.4f speedup=%6.2fx
                  %-26s medianMs=%8.4f speedup=%6.2fx
                  %-26s medianMs=%8.4f speedup=%6.2fx
                """,
                generic.output().length,
                WARMUP_ITERATIONS,
                MEASURE_ITERATIONS,
                generic.name(),
                generic.medianMs(),
                1.0d,
                denseScalar.name(),
                denseScalar.medianMs(),
                generic.medianMs() / denseScalar.medianMs(),
                denseVector.name(),
                denseVector.medianMs(),
                generic.medianMs() / denseVector.medianMs()
        );
    }

    private static Fixture fixture(Tensor out) {
        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
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
            ExecutionContext context
    ) {
    }

    private record BenchmarkResult(
            String name,
            double medianMs,
            float[] output
    ) {
    }
}

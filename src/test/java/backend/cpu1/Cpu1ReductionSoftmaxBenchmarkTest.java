package backend.cpu1;

import backend.contract.ComputeBackend;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.prepare.Cpu1NodePreparer;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.prepare.Cpu1PreparedArtifact;
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
import operations.reduction.logSoftmax;
import operations.reduction.softmax;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.storage.NativeFloat32Storage;
import tensor.storage.NativeTensorStorage;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;

@Tag("benchmark")
class Cpu1ReductionSoftmaxBenchmarkTest {
    private static final int[] GROUP_COUNTS = {1_024, 10_000, 100_000};
    private static final int CLASS_COUNT = 32;
    private static final int PARALLEL_WORKERS = 4;
    private static final int WARMUP_ITERATIONS = 2;
    private static final int MEASURE_ITERATIONS = 5;

    @Test
    void benchmarkSoftmaxAndLogSoftmaxGroupWidth() {
        StringBuilder report = new StringBuilder()
                .append("cpu1 softmax/logSoftmax group-width benchmark\n")
                .append("classes=").append(CLASS_COUNT)
                .append(", warmup=").append(WARMUP_ITERATIONS)
                .append(", measure=").append(MEASURE_ITERATIONS)
                .append(", workers=").append(PARALLEL_WORKERS)
                .append('\n');
        for (int groups : GROUP_COUNTS) {
            for (SoftmaxOp op : SoftmaxOp.values()) {
                CaseResult result = benchmarkCase(groups, op);
                report.append(String.format(
                        Locale.ROOT,
                        "%s groups=%d elements=%d | array %.4f -> %.4f ms (%.2fx), segment %.4f -> %.4f ms (%.2fx)%n",
                        op,
                        groups,
                        groups * CLASS_COUNT,
                        result.arraySingleMs(),
                        result.arrayParallelMs(),
                        speedup(result.arraySingleMs(), result.arrayParallelMs()),
                        result.segmentSingleMs(),
                        result.segmentParallelMs(),
                        speedup(result.segmentSingleMs(), result.segmentParallelMs())
                ));
            }
        }
        System.out.println(report);
    }

    private static CaseResult benchmarkCase(int groups, SoftmaxOp op) {
        Fixture fixture = fixture(outputTensor(groups, op));
        PreparedCase arraySingle = prepare(fixture, config(Cpu1StorageKind.JAVA_ARRAY, 1));
        PreparedCase arrayParallel = prepare(fixture, config(Cpu1StorageKind.JAVA_ARRAY, PARALLEL_WORKERS));
        PreparedCase segmentSingle = prepare(fixture, config(Cpu1StorageKind.MEMORY_SEGMENT, 1));
        PreparedCase segmentParallel = prepare(fixture, config(Cpu1StorageKind.MEMORY_SEGMENT, PARALLEL_WORKERS));
        attachNativeInput(segmentSingle.context(), fixture.inputNodeId(), values(groups));
        attachNativeInput(segmentParallel.context(), fixture.inputNodeId(), values(groups));

        BenchmarkResult arraySingleResult = benchmark(arraySingle);
        BenchmarkResult arrayParallelResult = benchmark(arrayParallel);
        BenchmarkResult segmentSingleResult = benchmark(segmentSingle);
        BenchmarkResult segmentParallelResult = benchmark(segmentParallel);

        assertArrayEquals(arraySingleResult.output(), arrayParallelResult.output(), 1.0e-5f);
        assertArrayEquals(arraySingleResult.output(), segmentSingleResult.output(), 1.0e-5f);
        assertArrayEquals(arraySingleResult.output(), segmentParallelResult.output(), 1.0e-5f);
        return new CaseResult(
                arraySingleResult.medianMs(),
                arrayParallelResult.medianMs(),
                segmentSingleResult.medianMs(),
                segmentParallelResult.medianMs()
        );
    }

    private static BenchmarkResult benchmark(PreparedCase preparedCase) {
        Cpu1Backend backend = new Cpu1Backend();
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            execute(preparedCase, backend);
        }
        long[] samples = new long[MEASURE_ITERATIONS];
        for (int i = 0; i < MEASURE_ITERATIONS; i++) {
            long start = System.nanoTime();
            execute(preparedCase, backend);
            samples[i] = System.nanoTime() - start;
        }
        return new BenchmarkResult(medianMs(samples), output(preparedCase));
    }

    private static void execute(PreparedCase preparedCase, Cpu1Backend backend) {
        backend.execute(preparedCase.fixture().node(), preparedCase.metadata(), preparedCase.context());
    }

    private static PreparedCase prepare(Fixture fixture, Cpu1PrepareConfig config) {
        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex(), config);
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, metadata);
        return new PreparedCase(fixture, artifact, metadata, context);
    }

    private static Cpu1PrepareConfig config(Cpu1StorageKind storageKind, int workers) {
        return new Cpu1PrepareConfig(
                Cpu1VectorizationKind.SCALAR,
                workers == 1 ? Cpu1LaunchConfig.singleThread() : Cpu1LaunchConfig.parallel(workers),
                storageKind
        );
    }

    private static Tensor outputTensor(int groups, SoftmaxOp op) {
        Tensor input = new Tensor(values(groups), new int[]{groups, CLASS_COUNT}, null,
                "cpu1-softmax-width-input", DataType.FLOAT32);
        return switch (op) {
            case SOFTMAX ->
                    new Tensor(input.getShape(), List.of(input), new softmax(1), "softmax", DataType.FLOAT32);
            case LOG_SOFTMAX ->
                    new Tensor(input.getShape(), List.of(input), new logSoftmax(1), "logSoftmax", DataType.FLOAT32);
        };
    }

    private static float[] values(int groups) {
        float[] values = new float[groups * CLASS_COUNT];
        for (int row = 0; row < groups; row++) {
            for (int col = 0; col < CLASS_COUNT; col++) {
                values[row * CLASS_COUNT + col] = ((col % 11) - 5) * 0.25f + (row % 7) * 0.03125f;
            }
        }
        return values;
    }

    private static float[] output(PreparedCase preparedCase) {
        if (preparedCase.artifact().preparedReductionUnit().storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            return nativeF32Values(preparedCase.context().nativeStorageForNodeId(preparedCase.fixture().node().id()));
        }
        return preparedCase.context()
                .runtimeTensorForNodeId(preparedCase.fixture().node().id())
                .toFloat32ArrayCopy();
    }

    private static void attachNativeInput(ExecutionContext context, int nodeId, float[] values) {
        NativeFloat32Storage storage = assertInstanceOf(
                NativeFloat32Storage.class,
                context.allocateNativeStorage(DataType.FLOAT32, values.length, "cpu1 softmax benchmark input")
        );
        for (int i = 0; i < values.length; i++) {
            storage.setFloat32At(i, values[i]);
        }
        context.attachNativeStorage(nodeId, storage, "cpu1 softmax benchmark native F32 input");
    }

    private static float[] nativeF32Values(NativeTensorStorage storage) {
        NativeFloat32Storage f32 = assertInstanceOf(NativeFloat32Storage.class, storage);
        float[] out = new float[f32.getSize()];
        for (int i = 0; i < out.length; i++) {
            out[i] = f32.getFloat32At(i);
        }
        return out;
    }

    private static Fixture fixture(Tensor out) {
        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
        CompiledTensorDescriptorIndex descriptorIndex = CompiledTensorDescriptorBuilder.build(nodes);
        CompiledNode node = nodes.getLast();
        return new Fixture(out, nodes, descriptorIndex, node, node.inputIds().getFirst());
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
                RuntimeConfig.inferenceDefaults(DataType.FLOAT32),
                ExecutionMode.FORWARD,
                metadataIndex,
                state
        );
    }

    private static CompiledNodeExecutionMetadata metadata(CompiledNode node, Cpu1PreparedArtifact artifact) {
        return new CompiledNodeExecutionMetadata(ComputeBackend.CPU, null, node.inputIds(), artifact);
    }

    private static double medianMs(long[] nanos) {
        long[] sorted = nanos.clone();
        Arrays.sort(sorted);
        if (sorted.length % 2 == 1) {
            return sorted[sorted.length / 2] / 1_000_000.0d;
        }
        int upper = sorted.length / 2;
        return ((sorted[upper - 1] + sorted[upper]) / 2.0d) / 1_000_000.0d;
    }

    private static double speedup(double single, double parallel) {
        return parallel <= 0.0d ? 0.0d : single / parallel;
    }

    private enum SoftmaxOp {
        SOFTMAX,
        LOG_SOFTMAX
    }

    private record BenchmarkResult(double medianMs, float[] output) {
    }

    private record CaseResult(
            double arraySingleMs,
            double arrayParallelMs,
            double segmentSingleMs,
            double segmentParallelMs
    ) {
    }

    private record PreparedCase(
            Fixture fixture,
            Cpu1PreparedArtifact artifact,
            CompiledNodeExecutionMetadata metadata,
            ExecutionContext context
    ) {
    }

    private record Fixture(
            Tensor root,
            List<CompiledNode> nodes,
            CompiledTensorDescriptorIndex descriptorIndex,
            CompiledNode node,
            int inputNodeId
    ) {
    }
}

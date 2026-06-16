package backend.cpu1;

import backend.ComputeBackend;
import backend.cpu1.exec.Cpu1ReductionExecutableUnit;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.kernels.reduction.Cpu1ReductionKernelId;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.launch.Cpu1RangeLauncher;
import backend.cpu1.prepare.Cpu1NodePreparer;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.cpu1.prepare.Cpu1PreparedReductionUnit;
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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;

@Tag("benchmark")
class Cpu1ReductionBenchmarkTest {
    private static final int ELEMENTS = 5_000_000;
    private static final int PARALLEL_WORKERS = 4;
    private static final int WARMUP_ITERATIONS = 4;
    private static final int MEASURE_ITERATIONS = 12;

    @Test
    void benchmarkScalarLargeSumMeanSingleThreadVsParallelPartials() {
        List<BenchmarkResult> results = List.of(
                benchmarkCase(DataType.FLOAT32, ReductionOp.SUM),
                benchmarkCase(DataType.FLOAT32, ReductionOp.MEAN),
                benchmarkCase(DataType.FLOAT64, ReductionOp.SUM),
                benchmarkCase(DataType.FLOAT64, ReductionOp.MEAN)
        );

        System.out.println(report(results));
    }

    private static BenchmarkResult benchmarkCase(DataType dataType, ReductionOp reductionOp) {
        Fixture fixture = fixture(reductionTensor(dataType, reductionOp));
        PreparedFixture single = preparedFixture(fixture, Cpu1PrepareConfig.scalarSingleThread());
        PreparedFixture parallel = preparedFixture(fixture, scalarParallel(PARALLEL_WORKERS));

        assertPreparedReduction(single, dataType, reductionOp, 1, 0);
        int expectedScratchSlots = Cpu1RangeLauncher.slotCount(
                ELEMENTS,
                parallel.artifact().preparedReductionUnit().launchConfig()
        );
        assertPreparedReduction(parallel, dataType, reductionOp, PARALLEL_WORKERS, expectedScratchSlots);

        PairedBenchmarkResult benchmark = benchmarkPaired(single, parallel);
        assertEquals(benchmark.singleOutput(), benchmark.parallelOutput(), tolerance(dataType, reductionOp));
        return new BenchmarkResult(
                dataType,
                reductionOp,
                ELEMENTS,
                single.artifact().preparedReductionUnit().kernelId(),
                parallel.artifact().preparedReductionUnit().kernelId(),
                parallel.artifact().preparedReductionUnit().scratchBufferSpec().f64ArrayElements(),
                benchmark.singleMedianMs(),
                benchmark.parallelMedianMs(),
                benchmark.singleOutput(),
                benchmark.parallelOutput()
        );
    }

    private static Cpu1PrepareConfig scalarParallel(int workerCount) {
        return new Cpu1PrepareConfig(
                Cpu1VectorizationKind.SCALAR,
                Cpu1LaunchConfig.parallel(workerCount),
                Cpu1StorageKind.JAVA_ARRAY
        );
    }

    private static void assertPreparedReduction(
            PreparedFixture fixture,
            DataType dataType,
            ReductionOp reductionOp,
            int expectedWorkers,
            int expectedF64ScratchSlots
    ) {
        Cpu1ReductionExecutableUnit executable = assertInstanceOf(
                Cpu1ReductionExecutableUnit.class,
                fixture.artifact().executableUnit()
        );
        Cpu1PreparedReductionUnit unit = fixture.artifact().preparedReductionUnit();
        assertSame(unit, executable.preparedUnit());
        assertEquals(kernelId(dataType, reductionOp), unit.kernelId());
        assertEquals(1, unit.outputElementCount());
        assertEquals(ELEMENTS, unit.axisSize());
        assertEquals(1, unit.outerSize());
        assertEquals(1, unit.innerSize());
        assertEquals(expectedWorkers, unit.launchConfig().workerCount());
        assertEquals(expectedF64ScratchSlots, unit.scratchBufferSpec().f64ArrayElements());
    }

    private static PairedBenchmarkResult benchmarkPaired(PreparedFixture single, PreparedFixture parallel) {
        Cpu1Backend singleBackend = new Cpu1Backend();
        Cpu1Backend parallelBackend = new Cpu1Backend();
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            if (i % 2 == 0) {
                execute(single, singleBackend);
                execute(parallel, parallelBackend);
            } else {
                execute(parallel, parallelBackend);
                execute(single, singleBackend);
            }
        }

        long[] singleSamples = new long[MEASURE_ITERATIONS];
        long[] parallelSamples = new long[MEASURE_ITERATIONS];
        for (int i = 0; i < MEASURE_ITERATIONS; i++) {
            if (i % 2 == 0) {
                singleSamples[i] = measure(single, singleBackend);
                parallelSamples[i] = measure(parallel, parallelBackend);
            } else {
                parallelSamples[i] = measure(parallel, parallelBackend);
                singleSamples[i] = measure(single, singleBackend);
            }
        }
        return new PairedBenchmarkResult(
                medianMs(singleSamples),
                medianMs(parallelSamples),
                scalarOutput(single),
                scalarOutput(parallel)
        );
    }

    private static long measure(PreparedFixture fixture, Cpu1Backend backend) {
        long start = System.nanoTime();
        execute(fixture, backend);
        return System.nanoTime() - start;
    }

    private static void execute(PreparedFixture fixture, Cpu1Backend backend) {
        backend.execute(fixture.fixture().node(), fixture.metadata(), fixture.context());
    }

    private static Tensor reductionTensor(DataType dataType, ReductionOp reductionOp) {
        Tensor input = inputTensor(dataType);
        return switch (reductionOp) {
            case SUM -> input.sum(0, true);
            case MEAN -> input.mean(0, true);
        };
    }

    private static Tensor inputTensor(DataType dataType) {
        return switch (dataType) {
            case FLOAT32 ->
                    new Tensor(f32Values(), new int[]{ELEMENTS}, null, "cpu1-reduction-benchmark-f32", dataType);
            case FLOAT64 ->
                    new Tensor(f64Values(), new int[]{ELEMENTS}, null, "cpu1-reduction-benchmark-f64", dataType);
            case BFLOAT16, BOOL, INT32, INT64 ->
                    throw new IllegalArgumentException("Unsupported benchmark dtype: " + dataType);
        };
    }

    private static float[] f32Values() {
        float[] values = new float[ELEMENTS];
        for (int i = 0; i < values.length; i++) {
            values[i] = (i % 5) + 1.0f;
        }
        return values;
    }

    private static double[] f64Values() {
        double[] values = new double[ELEMENTS];
        for (int i = 0; i < values.length; i++) {
            values[i] = (i % 5) + 1.0d;
        }
        return values;
    }

    private static PreparedFixture preparedFixture(Fixture fixture, Cpu1PrepareConfig config) {
        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex(), config);
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, metadata);
        return new PreparedFixture(fixture, artifact, metadata, context);
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
                RuntimeConfig.inferenceDefaults(fixture.node().dataType()),
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

    private static double scalarOutput(PreparedFixture fixture) {
        Tensor output = fixture.context().runtimeTensorForNodeId(fixture.fixture().node().id());
        return switch (fixture.fixture().node().dataType()) {
            case FLOAT32 -> output.toFloat32ArrayCopy()[0];
            case FLOAT64 -> output.toFloat64ArrayCopy()[0];
            case BFLOAT16, BOOL, INT32, INT64 ->
                    throw new IllegalArgumentException(
                            "Unsupported benchmark output dtype: " + fixture.fixture().node().dataType()
                    );
        };
    }

    private static Cpu1ReductionKernelId kernelId(DataType dataType, ReductionOp reductionOp) {
        return switch (reductionOp) {
            case SUM -> switch (dataType) {
                case FLOAT32 -> Cpu1ReductionKernelId.SUM_F32_DENSE_SCALAR;
                case FLOAT64 -> Cpu1ReductionKernelId.SUM_F64_DENSE_SCALAR;
                case BFLOAT16, BOOL, INT32, INT64 ->
                        throw new IllegalArgumentException("Unsupported benchmark dtype: " + dataType);
            };
            case MEAN -> switch (dataType) {
                case FLOAT32 -> Cpu1ReductionKernelId.MEAN_F32_DENSE_SCALAR;
                case FLOAT64 -> Cpu1ReductionKernelId.MEAN_F64_DENSE_SCALAR;
                case BFLOAT16, BOOL, INT32, INT64 ->
                        throw new IllegalArgumentException("Unsupported benchmark dtype: " + dataType);
            };
        };
    }

    private static double tolerance(DataType dataType, ReductionOp reductionOp) {
        if (dataType == DataType.FLOAT64) {
            return 1.0e-12d;
        }
        return reductionOp == ReductionOp.SUM ? 1.0e-3d : 1.0e-6d;
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

    private static String report(List<BenchmarkResult> results) {
        StringBuilder builder = new StringBuilder();
        builder.append(String.format(
                Locale.US,
                "cpu1 scalar large SUM/MEAN reduction benchmark%n  elements=%d, warmup=%d, measure=%d, parallelWorkers=%d%n",
                ELEMENTS,
                WARMUP_ITERATIONS,
                MEASURE_ITERATIONS,
                PARALLEL_WORKERS
        ));
        builder.append(String.format(
                Locale.US,
                "  %-7s %-5s %10s  %-26s %-26s %7s %12s %12s %9s %14s %14s%n",
                "dtype",
                "op",
                "elements",
                "singleKernel",
                "parallelKernel",
                "scratch",
                "singleMs",
                "parallelMs",
                "speedup",
                "singleOut",
                "parallelOut"
        ));
        for (BenchmarkResult result : results) {
            builder.append(String.format(
                    Locale.US,
                    "  %-7s %-5s %10d  %-26s %-26s %7d %12.4f %12.4f %8.2fx %14.6f %14.6f%n",
                    dtypeName(result.dataType()),
                    result.reductionOp(),
                    result.elements(),
                    result.singleKernelId(),
                    result.parallelKernelId(),
                    result.parallelScratchF64Slots(),
                    result.singleMedianMs(),
                    result.parallelMedianMs(),
                    result.singleMedianMs() / result.parallelMedianMs(),
                    result.singleOutput(),
                    result.parallelOutput()
            ));
        }
        return builder.toString();
    }

    private static String dtypeName(DataType dataType) {
        return switch (dataType) {
            case FLOAT32 -> "F32";
            case FLOAT64 -> "F64";
            case BFLOAT16 -> "BF16";
            case BOOL -> "BOOL";
            case INT32 -> "I32";
            case INT64 -> "I64";
        };
    }

    private enum ReductionOp {
        SUM,
        MEAN
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
            Cpu1PreparedArtifact artifact,
            CompiledNodeExecutionMetadata metadata,
            ExecutionContext context
    ) {
    }

    private record PairedBenchmarkResult(
            double singleMedianMs,
            double parallelMedianMs,
            double singleOutput,
            double parallelOutput
    ) {
    }

    private record BenchmarkResult(
            DataType dataType,
            ReductionOp reductionOp,
            int elements,
            Cpu1ReductionKernelId singleKernelId,
            Cpu1ReductionKernelId parallelKernelId,
            int parallelScratchF64Slots,
            double singleMedianMs,
            double parallelMedianMs,
            double singleOutput,
            double parallelOutput
    ) {
    }
}

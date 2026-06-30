package backend.cpu1;

import backend.contract.ComputeBackend;
import backend.cpu.nativecpu.NativeCpuStorageFactory;
import backend.cpu1.kernels.Cpu1VectorizationKind;
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
import operations.Operation;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;

@Tag("benchmark")
class Cpu1ElementwiseSegmentBenchmarkTest {
    private static final BenchmarkProfile MLP_PROFILE = new BenchmarkProfile("MLP sizes", 5, 12);
    private static final BenchmarkProfile MLP_HOT_PATH_PROFILE = new BenchmarkProfile("F32 MLP hotpath paired alternating", 5, 12);
    private static final BenchmarkProfile LARGE_PROFILE = new BenchmarkProfile("large sweep", 2, 5);
    private static final BenchmarkProfile F32_CHEAP_PROFILE = new BenchmarkProfile("F32 cheap sweep", 2, 5);
    private static final BenchmarkProfile F64_CHEAP_PROFILE = new BenchmarkProfile("F64 cheap sweep", 2, 5);
    private static final int[] MLP_HOT_PATH_ELEMENT_COUNTS = {
            131_072,
            262_144,
            524_288,
            1_048_576
    };
    private static final int[] LARGE_ELEMENT_COUNTS = {
            100_000,
            300_000,
            1_000_000,
            3_000_000,
            10_000_000
    };
    private static final int[] CHEAP_ELEMENT_COUNTS = {
            1_024,
            4_096,
            16_384,
            65_536,
            262_144,
            1_048_576,
            3_145_728
    };

    @Test
    void benchmarkF32MlpHotPathArrayVsMemorySegment() {
        List<CaseResult> hotPathResults = new ArrayList<>();
        for (int elements : MLP_HOT_PATH_ELEMENT_COUNTS) {
            hotPathResults.add(benchmarkCase(DataType.FLOAT32, BinaryOp.SUB, elements, MLP_HOT_PATH_PROFILE));
            hotPathResults.add(benchmarkCase(DataType.FLOAT32, BinaryOp.MUL, elements, MLP_HOT_PATH_PROFILE));
            hotPathResults.add(benchmarkCase(DataType.FLOAT32, UnaryOp.TANH, elements, MLP_HOT_PATH_PROFILE));
        }

        List<CaseResult> largeResults = new ArrayList<>();
        for (int elements : LARGE_ELEMENT_COUNTS) {
            largeResults.add(benchmarkCase(DataType.FLOAT32, BinaryOp.SUB, elements, LARGE_PROFILE));
            largeResults.add(benchmarkCase(DataType.FLOAT32, BinaryOp.MUL, elements, LARGE_PROFILE));
            largeResults.add(benchmarkCase(DataType.FLOAT32, UnaryOp.TANH, elements, LARGE_PROFILE));
            largeResults.add(benchmarkCase(DataType.FLOAT32, UnaryOp.RELU, elements, LARGE_PROFILE));
        }

        System.out.println(report(MLP_HOT_PATH_PROFILE, hotPathResults));
        System.out.println(report(LARGE_PROFILE, largeResults));
    }

    @Test
    void benchmarkF32CheapOpsVectorArrayVsMemorySegment() {
        List<CaseResult> results = new ArrayList<>();
        for (int elements : CHEAP_ELEMENT_COUNTS) {
            results.add(benchmarkCase(DataType.FLOAT32, BinaryOp.SUB, elements, F32_CHEAP_PROFILE));
            results.add(benchmarkCase(DataType.FLOAT32, BinaryOp.MUL, elements, F32_CHEAP_PROFILE));
            results.add(benchmarkCase(DataType.FLOAT32, UnaryOp.RELU, elements, F32_CHEAP_PROFILE));
        }
        System.out.println(report(F32_CHEAP_PROFILE, results));
    }

    @Test
    void benchmarkF64CheapOpsVectorArrayVsMemorySegment() {
        List<CaseResult> results = new ArrayList<>();
        for (int elements : CHEAP_ELEMENT_COUNTS) {
            results.add(benchmarkCase(DataType.FLOAT64, BinaryOp.SUB, elements, F64_CHEAP_PROFILE));
            results.add(benchmarkCase(DataType.FLOAT64, BinaryOp.MUL, elements, F64_CHEAP_PROFILE));
            results.add(benchmarkCase(DataType.FLOAT64, UnaryOp.RELU, elements, F64_CHEAP_PROFILE));
        }
        System.out.println(report(F64_CHEAP_PROFILE, results));
    }

    private static CaseResult benchmarkCase(DataType dataType, BinaryOp op, int elements, BenchmarkProfile profile) {
        Tensor leftTensor = tensor(dataType, values(elements, 37, 18, 0.03125d), op.name().toLowerCase(Locale.ROOT) + "-left");
        Tensor rightTensor = tensor(dataType, values(elements, 41, 20, 0.015625d), op.name().toLowerCase(Locale.ROOT) + "-right");
        Tensor out = switch (op) {
            case SUB -> leftTensor.sub(rightTensor);
            case MUL -> leftTensor.mul(rightTensor);
        };
        return benchmarkFixture(op.name(), fixture(out), profile);
    }

    private static CaseResult benchmarkCase(DataType dataType, UnaryOp op, int elements, BenchmarkProfile profile) {
        Tensor input = tensor(dataType, values(elements, 43, 21, 0.046875d), op.name().toLowerCase(Locale.ROOT) + "-input");
        Tensor out = switch (op) {
            case TANH -> input.tanh();
            case RELU -> input.relu();
        };
        return benchmarkFixture(op.name(), fixture(out), profile);
    }

    private static CaseResult benchmarkFixture(String opName, Fixture fixture, BenchmarkProfile profile) {
        PreparedCase arrayScalarCase = prepare(fixture, Cpu1PrepareConfig.scalarSingleThread());
        PreparedCase arrayVectorCase = prepare(fixture, Cpu1PrepareConfig.vectorSingleThread());
        PreparedCase segmentScalarCase = prepare(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        PreparedCase segmentVectorCase = prepare(fixture, Cpu1PrepareConfig.vectorMemorySegmentSingleThread());

        assertPreparedPath(arrayScalarCase.artifact(), Cpu1StorageKind.JAVA_ARRAY, Cpu1VectorizationKind.SCALAR);
        assertPreparedPath(arrayVectorCase.artifact(), Cpu1StorageKind.JAVA_ARRAY, Cpu1VectorizationKind.VECTOR);
        assertPreparedPath(segmentScalarCase.artifact(), Cpu1StorageKind.MEMORY_SEGMENT, Cpu1VectorizationKind.SCALAR);
        assertPreparedPath(segmentVectorCase.artifact(), Cpu1StorageKind.MEMORY_SEGMENT, Cpu1VectorizationKind.VECTOR);
        attachNativeInputs(segmentScalarCase.context(), fixture);
        attachNativeInputs(segmentVectorCase.context(), fixture);

        FourWayBenchmarkResult benchmark = benchmarkFourWayAlternating(
                arrayScalarCase,
                arrayVectorCase,
                segmentScalarCase,
                segmentVectorCase,
                profile
        );
        BenchmarkResult arrayScalar = benchmark.arrayScalar();
        BenchmarkResult arrayVector = benchmark.arrayVector();
        BenchmarkResult segmentScalar = benchmark.segmentScalar();
        BenchmarkResult segmentVector = benchmark.segmentVector();

        double tolerance = toleranceFor(fixture.node().operation().opType(), fixture.root().getDataType());
        assertArrayEquals(arrayScalar.output(), arrayVector.output(), tolerance);
        assertArrayEquals(arrayScalar.output(), segmentScalar.output(), tolerance);
        assertArrayEquals(arrayScalar.output(), segmentVector.output(), tolerance);
        assertEquals(0, segmentScalarCase.context().cpuMaterializationTraceCount());
        assertEquals(0, segmentVectorCase.context().cpuMaterializationTraceCount());
        return new CaseResult(
                fixture.root().getDataType(),
                opName,
                fixture.node().flatDataSize(),
                arrayScalarCase.artifact().preparedUnit().kernelId().name(),
                arrayVectorCase.artifact().preparedUnit().kernelId().name(),
                segmentScalarCase.artifact().preparedUnit().kernelId().name(),
                segmentVectorCase.artifact().preparedUnit().kernelId().name(),
                arrayScalar.medianMs(),
                arrayVector.medianMs(),
                segmentScalar.medianMs(),
                segmentVector.medianMs()
        );
    }

    private static PreparedCase prepare(Fixture fixture, Cpu1PrepareConfig config) {
        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex(), config);
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        Map<Integer, CompiledNodeExecutionMetadata> metadataIndex = Map.of(fixture.node().id(), metadata);
        ExecutionState state = ExecutionState.create(
                fixture.nodes(),
                fixture.descriptorIndex(),
                metadataIndex,
                fixture.node().id(),
                testsupport.PublicationPlans.forRoot(fixture.root(), fixture.nodes(), fixture.node().id())
        );
        ExecutionContext context = ExecutionContext.fromRuntimeConfig(
                RuntimeConfig.inferenceDefaults(fixture.root().getDataType()),
                ExecutionMode.FORWARD,
                metadataIndex,
                state
        );
        return new PreparedCase(fixture, artifact, metadata, context, state);
    }

    private static FourWayBenchmarkResult benchmarkFourWayAlternating(
            PreparedCase arrayScalarCase,
            PreparedCase arrayVectorCase,
            PreparedCase segmentScalarCase,
            PreparedCase segmentVectorCase,
            BenchmarkProfile profile
    ) {
        Cpu1Backend arrayScalarBackend = new Cpu1Backend();
        Cpu1Backend arrayVectorBackend = new Cpu1Backend();
        Cpu1Backend segmentScalarBackend = new Cpu1Backend();
        Cpu1Backend segmentVectorBackend = new Cpu1Backend();
        for (int i = 0; i < profile.warmupIterations(); i++) {
            if (i % 2 == 0) {
                execute(arrayScalarCase, arrayScalarBackend);
                execute(segmentScalarCase, segmentScalarBackend);
                execute(arrayVectorCase, arrayVectorBackend);
                execute(segmentVectorCase, segmentVectorBackend);
            } else {
                execute(segmentVectorCase, segmentVectorBackend);
                execute(arrayVectorCase, arrayVectorBackend);
                execute(segmentScalarCase, segmentScalarBackend);
                execute(arrayScalarCase, arrayScalarBackend);
            }
        }

        NativeCaseGuard segmentScalarGuard = captureNativeCaseGuard(segmentScalarCase);
        NativeCaseGuard segmentVectorGuard = captureNativeCaseGuard(segmentVectorCase);

        long[] arrayScalarSamples = new long[profile.measureIterations()];
        long[] arrayVectorSamples = new long[profile.measureIterations()];
        long[] segmentScalarSamples = new long[profile.measureIterations()];
        long[] segmentVectorSamples = new long[profile.measureIterations()];
        for (int i = 0; i < profile.measureIterations(); i++) {
            if (i % 2 == 0) {
                arrayScalarSamples[i] = measure(arrayScalarCase, arrayScalarBackend);
                segmentScalarSamples[i] = measure(segmentScalarCase, segmentScalarBackend);
                segmentScalarGuard.assertStillValid(segmentScalarCase);
                arrayVectorSamples[i] = measure(arrayVectorCase, arrayVectorBackend);
                segmentVectorSamples[i] = measure(segmentVectorCase, segmentVectorBackend);
                segmentVectorGuard.assertStillValid(segmentVectorCase);
            } else {
                segmentVectorSamples[i] = measure(segmentVectorCase, segmentVectorBackend);
                segmentVectorGuard.assertStillValid(segmentVectorCase);
                arrayVectorSamples[i] = measure(arrayVectorCase, arrayVectorBackend);
                segmentScalarSamples[i] = measure(segmentScalarCase, segmentScalarBackend);
                segmentScalarGuard.assertStillValid(segmentScalarCase);
                arrayScalarSamples[i] = measure(arrayScalarCase, arrayScalarBackend);
            }
        }
        assertEquals(0, segmentScalarCase.context().cpuMaterializationTraceCount());
        assertEquals(0, segmentVectorCase.context().cpuMaterializationTraceCount());
        return new FourWayBenchmarkResult(
                new BenchmarkResult(medianMs(arrayScalarSamples), output(arrayScalarCase, false)),
                new BenchmarkResult(medianMs(arrayVectorSamples), output(arrayVectorCase, false)),
                new BenchmarkResult(medianMs(segmentScalarSamples), output(segmentScalarCase, true)),
                new BenchmarkResult(medianMs(segmentVectorSamples), output(segmentVectorCase, true))
        );
    }

    private static long measure(PreparedCase preparedCase, Cpu1Backend backend) {
        long start = System.nanoTime();
        execute(preparedCase, backend);
        return System.nanoTime() - start;
    }

    private static void execute(PreparedCase preparedCase, Cpu1Backend backend) {
        backend.execute(preparedCase.fixture().node(), preparedCase.metadata(), preparedCase.context());
    }

    private static NativeCaseGuard captureNativeCaseGuard(PreparedCase preparedCase) {
        NativeTensorStorage expectedOutputStorage = preparedCase.context()
                .nativeStorageForNodeId(preparedCase.fixture().node().id());
        assertNotNull(expectedOutputStorage, "missing native output storage for nodeId=" + preparedCase.fixture().node().id());
        return new NativeCaseGuard(
                expectedOutputStorage,
                preparedCase.state().nativeCpuMemoryTrace().allocationCount()
        );
    }

    private static void assertPreparedPath(
            Cpu1PreparedArtifact artifact,
            Cpu1StorageKind storageKind,
            Cpu1VectorizationKind vectorizationKind
    ) {
        assertEquals(storageKind, artifact.preparedUnit().storageKind());
        assertEquals(vectorizationKind, artifact.preparedUnit().kernelId().key().vectorizationKind());
    }

    private static Fixture fixture(Tensor out) {
        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
        CompiledTensorDescriptorIndex descriptorIndex = CompiledTensorDescriptorBuilder.build(nodes);
        return new Fixture(out, nodes, descriptorIndex, nodes.getLast());
    }

    private static CompiledNodeExecutionMetadata metadata(CompiledNode node, Cpu1PreparedArtifact artifact) {
        return new CompiledNodeExecutionMetadata(
                ComputeBackend.CPU,
                null,
                node.inputIds(),
                artifact
        );
    }

    private static void attachNativeInputs(ExecutionContext context, Fixture fixture) {
        NativeCpuStorageFactory storageFactory = new NativeCpuStorageFactory();
        for (int inputNodeId : fixture.node().inputIds()) {
            Tensor tensor = context.runtimeTensorForNodeId(inputNodeId);
            NativeTensorStorage storage = storageFactory.allocate(
                    tensor.getDataType(),
                    tensor.size(),
                    "cpu1-elementwise-benchmark-input-" + inputNodeId
            );
            copyToNative(tensor, storage);
            context.attachNativeStorage(inputNodeId, storage, "cpu1 elementwise benchmark native input");
        }
    }

    private static void copyToNative(Tensor tensor, NativeTensorStorage storage) {
        MemorySegment segment = storage.segment();
        switch (tensor.getDataType()) {
            case FLOAT32 -> {
                float[] source = TensorInternalAccess.float32Data(tensor);
                for (int i = 0; i < storage.getSize(); i++) {
                    segment.set(JAVA_FLOAT, (long) i * Float.BYTES, source[i]);
                }
            }
            case FLOAT64 -> {
                double[] source = TensorInternalAccess.float64Data(tensor);
                for (int i = 0; i < storage.getSize(); i++) {
                    segment.set(JAVA_DOUBLE, (long) i * Double.BYTES, source[i]);
                }
            }
            case BFLOAT16, BOOL, INT32, INT64 ->
                    throw new UnsupportedOperationException("cpu1 elementwise benchmark native copy dtype=" + tensor.getDataType());
        }
        storage.markModified();
    }

    private static double[] output(PreparedCase preparedCase, boolean nativeOutput) {
        int nodeId = preparedCase.fixture().node().id();
        if (nativeOutput) {
            NativeTensorStorage storage = preparedCase.context().nativeStorageForNodeId(nodeId);
            return switch (preparedCase.fixture().root().getDataType()) {
                case FLOAT32 -> readNativeF32(storage);
                case FLOAT64 -> readNativeF64(storage);
                case BFLOAT16, BOOL, INT32, INT64 ->
                        throw new UnsupportedOperationException("cpu1 elementwise benchmark native output dtype=" + preparedCase.fixture().root().getDataType());
            };
        }
        return preparedCase.context().runtimeTensorForNodeId(nodeId).toDoubleArrayCopy();
    }

    private static double[] readNativeF32(NativeTensorStorage storage) {
        double[] out = new double[storage.getSize()];
        MemorySegment segment = storage.segment();
        for (int i = 0; i < out.length; i++) {
            out[i] = segment.get(JAVA_FLOAT, (long) i * Float.BYTES);
        }
        return out;
    }

    private static double[] readNativeF64(NativeTensorStorage storage) {
        double[] out = new double[storage.getSize()];
        MemorySegment segment = storage.segment();
        for (int i = 0; i < out.length; i++) {
            out[i] = segment.get(JAVA_DOUBLE, (long) i * Double.BYTES);
        }
        return out;
    }

    private static Tensor tensor(DataType dataType, double[] values, String label) {
        return switch (dataType) {
            case FLOAT32 -> new Tensor(f32(values), new int[]{values.length}, null, label, dataType);
            case FLOAT64 -> new Tensor(values, new int[]{values.length}, null, label, dataType);
            case BFLOAT16, BOOL, INT32, INT64 ->
                    throw new UnsupportedOperationException("cpu1 elementwise benchmark dtype=" + dataType);
        };
    }

    private static float[] f32(double[] values) {
        float[] out = new float[values.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = (float) values[i];
        }
        return out;
    }

    private static double[] values(int size, int modulus, int center, double scale) {
        double[] out = new double[size];
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

    private static double toleranceFor(Operation.OpType opType, DataType dataType) {
        if (dataType == DataType.FLOAT32) {
            return opType == Operation.OpType.TANH ? 1.0e-5d : 1.0e-6d;
        }
        return opType == Operation.OpType.TANH ? 1.0e-12d : 1.0e-12d;
    }

    private static String report(BenchmarkProfile profile, List<CaseResult> results) {
        StringBuilder builder = new StringBuilder();
        builder.append(String.format(
                Locale.US,
                "cpu1 elementwise array vs MemorySegment scalar/vector benchmark - %s%n  warmup=%d, measure=%d%n  measurement=paired alternating four-way; ratios are segment-over-array where named%n",
                profile.name(),
                profile.warmupIterations(),
                profile.measureIterations()
        ));
        builder.append(String.format(
                Locale.US,
                "  %-7s %-5s %8s  %-42s %-42s %-44s %-44s %13s %13s %15s %15s %23s %26s %19s %23s%n",
                "dtype",
                "op",
                "elements",
                "arrayScalarKernel",
                "arrayVectorKernel",
                "segmentScalarKernel",
                "segmentVectorKernel",
                "arrayScalarMs",
                "arrayVectorMs",
                "segmentScalarMs",
                "segmentVectorMs",
                "arrayVector/arrayScalar",
                "segmentVector/segmentScalar",
                "segmentScalar/arrayScalar",
                "segmentVector/arrayVector"
        ));
        for (CaseResult result : results) {
            builder.append(String.format(
                    Locale.US,
                    "  %-7s %-5s %8d  %-42s %-42s %-44s %-44s %13.4f %13.4f %15.4f %15.4f %23.3f %26.3f %19.3f %23.3f%n",
                    dtypeName(result.dataType()),
                    result.opName(),
                    result.elements(),
                    result.arrayScalarKernel(),
                    result.arrayVectorKernel(),
                    result.segmentScalarKernel(),
                    result.segmentVectorKernel(),
                    result.arrayScalarMedianMs(),
                    result.arrayVectorMedianMs(),
                    result.segmentScalarMedianMs(),
                    result.segmentVectorMedianMs(),
                    result.arrayVectorMedianMs() / result.arrayScalarMedianMs(),
                    result.segmentVectorMedianMs() / result.segmentScalarMedianMs(),
                    result.segmentScalarMedianMs() / result.arrayScalarMedianMs(),
                    result.segmentVectorMedianMs() / result.arrayVectorMedianMs()
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

    private enum BinaryOp {
        SUB,
        MUL
    }

    private enum UnaryOp {
        TANH,
        RELU
    }

    private record Fixture(
            Tensor root,
            List<CompiledNode> nodes,
            CompiledTensorDescriptorIndex descriptorIndex,
            CompiledNode node
    ) {
    }

    private record PreparedCase(
            Fixture fixture,
            Cpu1PreparedArtifact artifact,
            CompiledNodeExecutionMetadata metadata,
            ExecutionContext context,
            ExecutionState state
    ) {
    }

    private record BenchmarkResult(
            double medianMs,
            double[] output
    ) {
    }

    private record FourWayBenchmarkResult(
            BenchmarkResult arrayScalar,
            BenchmarkResult arrayVector,
            BenchmarkResult segmentScalar,
            BenchmarkResult segmentVector
    ) {
    }

    private record NativeCaseGuard(
            NativeTensorStorage expectedOutputStorage,
            long allocationCountAfterWarmup
    ) {
        private void assertStillValid(PreparedCase preparedCase) {
            assertSame(expectedOutputStorage, preparedCase.context().nativeStorageForNodeId(preparedCase.fixture().node().id()));
            assertEquals(allocationCountAfterWarmup, preparedCase.state().nativeCpuMemoryTrace().allocationCount());
        }
    }

    private record BenchmarkProfile(
            String name,
            int warmupIterations,
            int measureIterations
    ) {
    }

    private record CaseResult(
            DataType dataType,
            String opName,
            int elements,
            String arrayScalarKernel,
            String arrayVectorKernel,
            String segmentScalarKernel,
            String segmentVectorKernel,
            double arrayScalarMedianMs,
            double arrayVectorMedianMs,
            double segmentScalarMedianMs,
            double segmentVectorMedianMs
    ) {
    }
}

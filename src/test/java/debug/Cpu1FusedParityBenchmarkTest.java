package debug;

import runtime.memory.nativecpu.NativeCpuStorageFactory;
import backend.cpu.CpuFusedExecutionArtifact;
import backend.cpu1.exec.Cpu1FusedKernelArgs;
import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.fused.ir.Cpu1FusedAccessKind;
import backend.cpu1.fused.ir.Cpu1FusedExpressionPlan;
import backend.cpu1.fused.ir.Cpu1FusedInputPlan;
import backend.cpu1.fused.ir.Cpu1FusedNodePlan;
import backend.cpu1.fused.ir.Cpu1FusedScalarParameter;
import backend.cpu1.kernels.Cpu1LayoutKind;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.kernels.fused.codegen.Cpu1FusedCodegenKernel;
import backend.cpu1.kernels.fused.codegen.Cpu1FusedCodegenKernelFactory;
import backend.cpu1.kernels.fused.codegen.Cpu1FusedCodegenLoopKind;
import backend.cpu1.kernels.fused.codegen.Cpu1FusedCodegenPlan;
import backend.cpu1.kernels.fused.codegen.Cpu1FusedCodegenRejectionReason;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.launch.Cpu1LaunchPolicy;
import backend.cpu1.launch.Cpu1SingleThreadLaunch;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.cpu1.prepare.Cpu1PreparedFusedElementwiseUnit;
import backend.cpu1.prepare.dispatch.Cpu1CostClass;
import backend.cpu1.prepare.dispatch.Cpu1FusedDispatchDecision;
import backend.cpu1.storage.Cpu1StorageKind;
import runtime.contract.ExecutionMode;
import config.backend.CpuKernelConfig;
import config.compile.CompileConfig;
import config.runtime.ApproximationConfig;
import config.runtime.BlasConfig;
import config.runtime.FusedExecutionPolicy;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import runtime.execution.PreparedExecution;
import operations.Operation;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.function.Supplier;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

@Tag("benchmark")
final class Cpu1FusedParityBenchmarkTest {
    private static final String ENABLE_PROPERTY = "synaptik.benchmark.cpu1FusedParity";
    private static final String ENABLE_ENV = "SYNAPTIK_BENCHMARK_CPU1_FUSED_PARITY";
    private static final int WARMUP_ITERATIONS = 4;
    private static final int MEASURE_ITERATIONS = 20;
    private static final int REPEATS = 5;
    private static final int VECTOR_ELEMENTS = 8_192;
    private static final int SCALAR_ELEMENTS = 512;
    private static final String HEADER =
            "case,dtype,storage,elements,oldCpuMedianMs,cpu1JavaMedianMs,cpu1JavaRatio,"
                    + "javaToNativeMedianMs,nativeResidentMedianMs,nativeToJavaMedianMs,"
                    + "nativeEndToEndMedianMs,nativeResidentVsOldCpuRatio,"
                    + "nativeEndToEndVsOldCpuRatio,cpu1CodegenRejectionReason";

    @Test
    void benchmarkOldCpuFusedVsCpu1FusedRoute() {
        Assumptions.assumeTrue(
                benchmarkEnabled(),
                "Set " + ENABLE_ENV + "=true before Gradle, or -D" + ENABLE_PROPERTY
                        + "=true in the test JVM, to run the cpu1 fused parity benchmark."
        );

        System.out.println(HEADER);
        for (BenchmarkCase benchmarkCase : benchmarkCases()) {
            reportCase(benchmarkCase);
        }
        reportNativeCase("native-memory-segment-f32", DataType.FLOAT32, 1.0e-5);
        reportNativeCase("native-memory-segment-f64", DataType.FLOAT64, 1.0e-12);
    }

    private static boolean benchmarkEnabled() {
        return Boolean.getBoolean(ENABLE_PROPERTY) || Boolean.parseBoolean(System.getenv(ENABLE_ENV));
    }

    private static List<BenchmarkCase> benchmarkCases() {
        return List.of(
                new BenchmarkCase(
                        "cheap-contiguous-f32",
                        DataType.FLOAT32,
                        "JAVA_ARRAY",
                        VECTOR_ELEMENTS,
                        1.0e-5,
                        () -> cheapContiguous(DataType.FLOAT32, VECTOR_ELEMENTS)
                ),
                new BenchmarkCase(
                        "cheap-contiguous-f64",
                        DataType.FLOAT64,
                        "JAVA_ARRAY",
                        VECTOR_ELEMENTS,
                        1.0e-12,
                        () -> cheapContiguous(DataType.FLOAT64, VECTOR_ELEMENTS)
                ),
                new BenchmarkCase(
                        "bf16-chain",
                        DataType.BFLOAT16,
                        "JAVA_ARRAY",
                        SCALAR_ELEMENTS,
                        8.0e-3,
                        () -> cheapContiguous(DataType.BFLOAT16, SCALAR_ELEMENTS)
                ),
                new BenchmarkCase(
                        "broadcast-bias-f32",
                        DataType.FLOAT32,
                        "JAVA_ARRAY",
                        VECTOR_ELEMENTS,
                        1.0e-5,
                        Cpu1FusedParityBenchmarkTest::broadcastBias
                ),
                new BenchmarkCase(
                        "transcendental-f32",
                        DataType.FLOAT32,
                        "JAVA_ARRAY",
                        SCALAR_ELEMENTS,
                        1.0e-5,
                        Cpu1FusedParityBenchmarkTest::transcendental
                ),
                new BenchmarkCase(
                        "strided-input-view-f32",
                        DataType.FLOAT32,
                        "JAVA_ARRAY",
                        VECTOR_ELEMENTS,
                        1.0e-5,
                        Cpu1FusedParityBenchmarkTest::stridedInputView
                ),
                new BenchmarkCase(
                        "where-mask-f32",
                        DataType.FLOAT32,
                        "JAVA_ARRAY",
                        SCALAR_ELEMENTS,
                        1.0e-5,
                        Cpu1FusedParityBenchmarkTest::whereMask
                )
        );
    }

    private static void reportCase(BenchmarkCase benchmarkCase) {
        GraphRun oldRoute = prepare(benchmarkCase, false);
        try (PreparedExecution oldExecution = oldRoute.execution()) {
            assertTrue(hasOldCpuFusedArtifact(oldExecution), benchmarkCase.name() + " did not use old CPU fused artifact");
            BenchmarkResult oldResult = benchmark(oldExecution, oldRoute.root());

            GraphRun cpu1Route;
            try {
                cpu1Route = prepare(benchmarkCase, true);
            } catch (UnsupportedOperationException ex) {
                printJavaRow(benchmarkCase, oldResult.medianMs(), Double.NaN, rejectionReason(ex));
                return;
            }

            try (PreparedExecution cpu1Execution = cpu1Route.execution()) {
                assertTrue(hasCpu1FusedArtifact(cpu1Execution), benchmarkCase.name() + " did not use cpu1 fused artifact");
                BenchmarkResult cpu1Result = benchmark(cpu1Execution, cpu1Route.root());
                assertArrayEquals(
                        oldResult.output(),
                        cpu1Result.output(),
                        benchmarkCase.tolerance(),
                        benchmarkCase.name() + " route-on/off output mismatch"
                );
                printJavaRow(benchmarkCase, oldResult.medianMs(), cpu1Result.medianMs(), "NONE");
            } catch (UnsupportedOperationException ex) {
                printJavaRow(benchmarkCase, oldResult.medianMs(), Double.NaN, rejectionReason(ex));
            }
        }
    }

    private static void reportNativeCase(String name, DataType dataType, double tolerance) {
        BenchmarkCase javaBaseline = new BenchmarkCase(
                name,
                dataType,
                "MEMORY_SEGMENT",
                VECTOR_ELEMENTS,
                tolerance,
                () -> cheapContiguous(dataType, VECTOR_ELEMENTS)
        );
        GraphRun oldRoute = prepare(javaBaseline, false);
        try (PreparedExecution oldExecution = oldRoute.execution()) {
            assertTrue(hasOldCpuFusedArtifact(oldExecution), name + " Java baseline did not use old CPU fused artifact");
            BenchmarkResult oldResult = benchmark(oldExecution, oldRoute.root());

            GraphRun cpu1Route = prepare(javaBaseline, true);
            try (PreparedExecution cpu1Execution = cpu1Route.execution()) {
                assertTrue(hasCpu1FusedArtifact(cpu1Execution), name + " Java baseline did not use cpu1 fused artifact");
                BenchmarkResult cpu1Result = benchmark(cpu1Execution, cpu1Route.root());
                assertArrayEquals(
                        oldResult.output(),
                        cpu1Result.output(),
                        tolerance,
                        name + " Java baseline route-on/off output mismatch"
                );

                try (NativeBenchmarkCase nativeCase = NativeBenchmarkCase.create(dataType, VECTOR_ELEMENTS)) {
                    NativeBenchmarkResult nativeResult = benchmarkNative(nativeCase);
                    assertArrayEquals(
                            oldResult.output(),
                            nativeResult.output(),
                            tolerance,
                            name + " native output mismatch"
                    );
                    printNativeRow(javaBaseline, oldResult.medianMs(), cpu1Result.medianMs(), nativeResult);
                }
            }
        }
    }

    private static GraphRun prepare(BenchmarkCase benchmarkCase, boolean useCpu1Elementwise) {
        Tensor root = benchmarkCase.rootFactory().get();
        PreparedExecution execution = CompiledGraph.compile(root, CompileConfig.inference())
                .prepare(runtimeConfig(useCpu1Elementwise));
        return new GraphRun(root, execution);
    }

    private static RuntimeConfig runtimeConfig(boolean useCpu1Elementwise) {
        return new RuntimeConfig(
                CpuKernelConfig.defaultsInference(),
                ApproximationConfig.defaults(),
                BlasConfig.disabled(),
                new FusedExecutionPolicy(false, useCpu1Elementwise)
        );
    }

    private static BenchmarkResult benchmark(PreparedExecution execution, Tensor root) {
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            execution.execute(ExecutionMode.FORWARD);
        }
        double[] samples = new double[REPEATS];
        for (int repeat = 0; repeat < REPEATS; repeat++) {
            long start = System.nanoTime();
            for (int i = 0; i < MEASURE_ITERATIONS; i++) {
                execution.execute(ExecutionMode.FORWARD);
            }
            samples[repeat] = (System.nanoTime() - start) / 1_000_000.0d / MEASURE_ITERATIONS;
        }
        Arrays.sort(samples);
        return new BenchmarkResult(percentile(samples, 50), root.toDoubleArrayCopy());
    }

    private static NativeBenchmarkResult benchmarkNative(NativeBenchmarkCase nativeCase) {
        double javaToNativeMedianMs = benchmarkMedianMs(nativeCase::copyJavaInputsToNative);
        nativeCase.markNativeInputsModified();
        double nativeResidentMedianMs = benchmarkMedianMs(nativeCase::computeRange);
        double nativeToJavaMedianMs = benchmarkMedianMs(nativeCase::readNativeOutputToJava);
        double nativeEndToEndMedianMs = javaToNativeMedianMs + nativeResidentMedianMs + nativeToJavaMedianMs;
        return new NativeBenchmarkResult(
                javaToNativeMedianMs,
                nativeResidentMedianMs,
                nativeToJavaMedianMs,
                nativeEndToEndMedianMs,
                nativeCase.readNativeOutputAsDouble()
        );
    }

    private static double benchmarkMedianMs(Runnable action) {
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            action.run();
        }
        double[] samples = new double[REPEATS];
        for (int repeat = 0; repeat < REPEATS; repeat++) {
            long start = System.nanoTime();
            for (int i = 0; i < MEASURE_ITERATIONS; i++) {
                action.run();
            }
            samples[repeat] = (System.nanoTime() - start) / 1_000_000.0d / MEASURE_ITERATIONS;
        }
        Arrays.sort(samples);
        return percentile(samples, 50);
    }

    private static double percentile(double[] sortedValues, int percentile) {
        if (sortedValues.length == 1) {
            return sortedValues[0];
        }
        double rank = (percentile / 100.0d) * (sortedValues.length - 1);
        int low = (int) Math.floor(rank);
        int high = (int) Math.ceil(rank);
        if (low == high) {
            return sortedValues[low];
        }
        double weight = rank - low;
        return sortedValues[low] * (1.0d - weight) + sortedValues[high] * weight;
    }

    private static Tensor cheapContiguous(DataType dataType, int elements) {
        int[] shape = new int[]{elements};
        Tensor a = tensor(values(elements, -0.35, 0.75), shape, "a", dataType);
        Tensor b = tensor(values(elements, 0.20, 0.45), shape, "b", dataType);
        Tensor c = tensor(values(elements, -0.10, 0.25), shape, "c", dataType);
        return a.mul(b).add(c).relu();
    }

    private static Tensor broadcastBias() {
        int rows = 128;
        int columns = 64;
        int[] shape = new int[]{rows, columns};
        Tensor a = tensor(values(rows * columns, -0.25, 0.55), shape, "a", DataType.FLOAT32);
        Tensor bias = tensor(values(columns, 0.05, 0.20), new int[]{columns}, "bias", DataType.FLOAT32);
        return a.add(bias).relu();
    }

    private static Tensor transcendental() {
        int[] shape = new int[]{SCALAR_ELEMENTS};
        Tensor x = tensor(values(SCALAR_ELEMENTS, -0.05, 0.08), shape, "x", DataType.FLOAT32);
        Tensor y = tensor(values(SCALAR_ELEMENTS, 0.10, 0.05), shape, "y", DataType.FLOAT32);
        return x.exp().add(y).tanh();
    }

    private static Tensor stridedInputView() {
        int rows = 128;
        int columns = 64;
        Tensor base = tensor(values(rows * columns * 2, -0.30, 0.45), new int[]{rows, columns * 2}, "base", DataType.FLOAT32);
        Tensor strided = base.slice(new int[]{0, 0}, new int[]{rows, columns * 2}, new int[]{0, 1}, new int[]{1, 2});
        Tensor b = tensor(values(rows * columns, 0.20, 0.40), new int[]{rows, columns}, "b", DataType.FLOAT32);
        Tensor c = tensor(values(rows * columns, -0.05, 0.15), new int[]{rows, columns}, "c", DataType.FLOAT32);
        return strided.mul(b).add(c).relu();
    }

    private static Tensor whereMask() {
        int[] shape = new int[]{SCALAR_ELEMENTS};
        Tensor mask = new Tensor(maskValues(SCALAR_ELEMENTS), shape, null, "mask", DataType.BOOL);
        Tensor x = tensor(values(SCALAR_ELEMENTS, -0.20, 0.60), shape, "x", DataType.FLOAT32);
        Tensor fill = tensor(new double[]{-3.0d}, new int[]{1}, "fill", DataType.FLOAT32);
        return Tensor.where(mask, x.mul(0.25d), fill);
    }

    private static Tensor tensor(double[] values, int[] shape, String label, DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> new Tensor(values.clone(), shape.clone(), null, label, DataType.FLOAT64);
            case FLOAT32 -> new Tensor(toFloatArray(values), shape.clone(), null, label, DataType.FLOAT32);
            case BFLOAT16 -> new Tensor(toFloatArray(values), shape.clone(), null, label, DataType.BFLOAT16);
            default -> throw new IllegalArgumentException("Unsupported benchmark dtype: " + dataType);
        };
    }

    private static double[] values(int length, double offset, double scale) {
        double[] values = new double[length];
        for (int i = 0; i < values.length; i++) {
            values[i] = offset + scale * Math.sin(i * 0.013d) + 0.125d * Math.cos(i * 0.007d);
        }
        return values;
    }

    private static float[] toFloatArray(double[] values) {
        float[] floats = new float[values.length];
        for (int i = 0; i < values.length; i++) {
            floats[i] = (float) values[i];
        }
        return floats;
    }

    private static byte[] maskValues(int length) {
        byte[] mask = new byte[length];
        for (int i = 0; i < mask.length; i++) {
            mask[i] = (byte) (i % 3 == 0 || i % 5 == 0 ? 1 : 0);
        }
        return mask;
    }

    private static boolean hasOldCpuFusedArtifact(PreparedExecution execution) {
        return execution.forwardSteps().stream()
                .anyMatch(step -> step.metadata().executable() instanceof CpuFusedExecutionArtifact);
    }

    private static boolean hasCpu1FusedArtifact(PreparedExecution execution) {
        for (var step : execution.forwardSteps()) {
            if (step.metadata().executable() instanceof Cpu1PreparedArtifact artifact) {
                try {
                    artifact.preparedFusedElementwiseUnit();
                    return true;
                } catch (IllegalStateException ignored) {
                    // Other cpu1 artifacts are not relevant to this route benchmark.
                }
            }
        }
        return false;
    }

    private static void printJavaRow(BenchmarkCase benchmarkCase, double oldMs, double cpu1Ms, String reason) {
        printRow(
                benchmarkCase,
                oldMs,
                cpu1Ms,
                Double.NaN,
                Double.NaN,
                Double.NaN,
                Double.NaN,
                reason
        );
    }

    private static void printNativeRow(
            BenchmarkCase benchmarkCase,
            double oldMs,
            double cpu1JavaMs,
            NativeBenchmarkResult nativeResult
    ) {
        printRow(
                benchmarkCase,
                oldMs,
                cpu1JavaMs,
                nativeResult.javaToNativeMedianMs(),
                nativeResult.nativeResidentMedianMs(),
                nativeResult.nativeToJavaMedianMs(),
                nativeResult.nativeEndToEndMedianMs(),
                "NONE"
        );
    }

    private static void printRow(
            BenchmarkCase benchmarkCase,
            double oldMs,
            double cpu1JavaMs,
            double javaToNativeMs,
            double nativeResidentMs,
            double nativeToJavaMs,
            double nativeEndToEndMs,
            String reason
    ) {
        String oldValue = formatNullableMs(oldMs);
        String cpu1Value = formatNullableMs(cpu1JavaMs);
        String cpu1Ratio = Double.isFinite(oldMs) && Double.isFinite(cpu1JavaMs) && oldMs > 0.0d
                ? String.format(Locale.US, "%.6f", cpu1JavaMs / oldMs)
                : "NA";
        String nativeResidentRatio = Double.isFinite(oldMs) && Double.isFinite(nativeResidentMs) && oldMs > 0.0d
                ? String.format(Locale.US, "%.6f", nativeResidentMs / oldMs)
                : "NA";
        String nativeEndToEndRatio = Double.isFinite(oldMs) && Double.isFinite(nativeEndToEndMs) && oldMs > 0.0d
                ? String.format(Locale.US, "%.6f", nativeEndToEndMs / oldMs)
                : "NA";
        System.out.printf(
                Locale.US,
                "%s,%s,%s,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s%n",
                benchmarkCase.name(),
                dtypeId(benchmarkCase.dataType()),
                benchmarkCase.storage(),
                benchmarkCase.elements(),
                oldValue,
                cpu1Value,
                cpu1Ratio,
                formatNullableMs(javaToNativeMs),
                formatNullableMs(nativeResidentMs),
                formatNullableMs(nativeToJavaMs),
                formatNullableMs(nativeEndToEndMs),
                nativeResidentRatio,
                nativeEndToEndRatio,
                csvSafe(reason)
        );
    }

    private static String formatNullableMs(double value) {
        return Double.isFinite(value) ? String.format(Locale.US, "%.6f", value) : "NA";
    }

    private static String rejectionReason(UnsupportedOperationException ex) {
        String message = ex.getMessage();
        return message == null || message.isBlank() ? ex.getClass().getSimpleName() : message;
    }

    private static String csvSafe(String value) {
        return (value == null || value.isBlank() ? "NONE" : value)
                .replace(',', ';')
                .replace('\n', ' ')
                .replace('\r', ' ');
    }

    private static String dtypeId(DataType dataType) {
        return switch (dataType) {
            case FLOAT32 -> "F32";
            case FLOAT64 -> "F64";
            case BFLOAT16 -> "BF16";
            default -> fail("Unexpected benchmark dtype: " + dataType);
        };
    }

    private static Cpu1FusedExpressionPlan nativeCheapContiguousPlan(DataType dataType, int elements) {
        int[] shape = new int[]{elements};
        return new Cpu1FusedExpressionPlan(
                List.of(
                        node(0, Operation.OpType.MUL, List.of(0, 1), 3, dataType),
                        node(1, Operation.OpType.ADD, List.of(3, 2), 4, dataType),
                        node(2, Operation.OpType.RELU, List.of(4), 5, dataType)
                ),
                List.of(
                        contiguousInput(0, dataType, shape),
                        contiguousInput(1, dataType, shape),
                        contiguousInput(2, dataType, shape)
                ),
                5
        );
    }

    private static Cpu1FusedNodePlan node(
            int index,
            Operation.OpType opType,
            List<Integer> inputRefs,
            int outputRef,
            DataType outputType
    ) {
        return new Cpu1FusedNodePlan(
                index,
                10 + index,
                opType,
                inputRefs,
                outputRef,
                outputType,
                Cpu1FusedScalarParameter.NONE
        );
    }

    private static Cpu1FusedInputPlan contiguousInput(int ref, DataType dataType, int[] shape) {
        int[] strides = denseStrides(shape);
        return new Cpu1FusedInputPlan(
                ref,
                ref,
                dataType,
                shape,
                strides,
                shape,
                strides,
                0,
                strides,
                Cpu1FusedAccessKind.DIRECT_CONTIGUOUS
        );
    }

    private static int[] denseStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int stride = 1;
        for (int dim = shape.length - 1; dim >= 0; dim--) {
            strides[dim] = stride;
            stride *= shape[dim];
        }
        return strides;
    }

    private static Cpu1PreparedFusedElementwiseUnit nativePreparedUnit(
            Cpu1FusedExpressionPlan plan,
            DataType dataType,
            Cpu1FusedCodegenKernel kernel,
            int elements
    ) {
        Cpu1LaunchConfig launchConfig = Cpu1LaunchConfig.singleThread();
        Cpu1LaunchPolicy launchPolicy = new Cpu1SingleThreadLaunch(launchConfig);
        return new Cpu1PreparedFusedElementwiseUnit(
                "native-fused-parity-benchmark-" + dtypeId(dataType),
                plan.nodes().stream().map(Cpu1FusedNodePlan::nodeId).toList(),
                plan.inputs().stream().map(Cpu1FusedInputPlan::nodeId).toList(),
                100,
                dataType,
                elements,
                new int[]{elements},
                plan,
                Cpu1LayoutKind.CONTIGUOUS,
                Cpu1StorageKind.MEMORY_SEGMENT,
                launchPolicy,
                launchConfig,
                new Cpu1FusedDispatchDecision(
                        Cpu1CostClass.CHEAP_ELEMENTWISE,
                        Cpu1VectorizationKind.SCALAR,
                        launchConfig,
                        Cpu1StorageKind.MEMORY_SEGMENT,
                        1024,
                        1024,
                        1
                ),
                Cpu1FusedCodegenRejectionReason.NONE,
                kernel,
                false,
                false
        );
    }

    private static NativeTensorStorage nativeStorage(
            NativeCpuStorageFactory storageFactory,
            DataType dataType,
            int elements,
            String label
    ) {
        return storageFactory.allocate(dataType, elements, label);
    }

    private static void copyF32ToNative(float[] source, NativeTensorStorage target) {
        MemorySegment segment = target.segment();
        for (int i = 0; i < source.length; i++) {
            segment.set(JAVA_FLOAT, (long) i * Float.BYTES, source[i]);
        }
    }

    private static void copyF64ToNative(double[] source, NativeTensorStorage target) {
        MemorySegment segment = target.segment();
        for (int i = 0; i < source.length; i++) {
            segment.set(JAVA_DOUBLE, (long) i * Double.BYTES, source[i]);
        }
    }

    private static void readNativeF32ToJava(NativeTensorStorage source, float[] target) {
        MemorySegment segment = source.segment();
        for (int i = 0; i < target.length; i++) {
            target[i] = segment.get(JAVA_FLOAT, (long) i * Float.BYTES);
        }
    }

    private static void readNativeF64ToJava(NativeTensorStorage source, double[] target) {
        MemorySegment segment = source.segment();
        for (int i = 0; i < target.length; i++) {
            target[i] = segment.get(JAVA_DOUBLE, (long) i * Double.BYTES);
        }
    }

    private static double[] readNativeOutputAsDouble(DataType dataType, NativeTensorStorage source, int elements) {
        MemorySegment segment = source.segment();
        double[] values = new double[elements];
        switch (dataType) {
            case FLOAT32 -> {
                for (int i = 0; i < values.length; i++) {
                    values[i] = segment.get(JAVA_FLOAT, (long) i * Float.BYTES);
                }
            }
            case FLOAT64 -> {
                for (int i = 0; i < values.length; i++) {
                    values[i] = segment.get(JAVA_DOUBLE, (long) i * Double.BYTES);
                }
            }
            default -> throw new IllegalArgumentException("Unsupported native benchmark dtype: " + dataType);
        }
        return values;
    }

    private record BenchmarkCase(
            String name,
            DataType dataType,
            String storage,
            int elements,
            double tolerance,
            Supplier<Tensor> rootFactory
    ) {
    }

    private record GraphRun(Tensor root, PreparedExecution execution) {
    }

    private record BenchmarkResult(double medianMs, double[] output) {
    }

    private record NativeBenchmarkResult(
            double javaToNativeMedianMs,
            double nativeResidentMedianMs,
            double nativeToJavaMedianMs,
            double nativeEndToEndMedianMs,
            double[] output
    ) {
    }

    private static final class NativeBenchmarkCase implements AutoCloseable {
        private final DataType dataType;
        private final int elements;
        private final NativeTensorStorage leftStorage;
        private final NativeTensorStorage rightStorage;
        private final NativeTensorStorage biasStorage;
        private final NativeTensorStorage outputStorage;
        private final Cpu1FusedKernelArgs args;
        private final Cpu1PreparedFusedElementwiseUnit preparedUnit;
        private final float[] leftF32;
        private final float[] rightF32;
        private final float[] biasF32;
        private final float[] javaOutputF32;
        private final double[] leftF64;
        private final double[] rightF64;
        private final double[] biasF64;
        private final double[] javaOutputF64;

        private NativeBenchmarkCase(
                DataType dataType,
                int elements,
                NativeTensorStorage leftStorage,
                NativeTensorStorage rightStorage,
                NativeTensorStorage biasStorage,
                NativeTensorStorage outputStorage,
                Cpu1FusedKernelArgs args,
                Cpu1PreparedFusedElementwiseUnit preparedUnit,
                float[] leftF32,
                float[] rightF32,
                float[] biasF32,
                float[] javaOutputF32,
                double[] leftF64,
                double[] rightF64,
                double[] biasF64,
                double[] javaOutputF64
        ) {
            this.dataType = dataType;
            this.elements = elements;
            this.leftStorage = leftStorage;
            this.rightStorage = rightStorage;
            this.biasStorage = biasStorage;
            this.outputStorage = outputStorage;
            this.args = args;
            this.preparedUnit = preparedUnit;
            this.leftF32 = leftF32;
            this.rightF32 = rightF32;
            this.biasF32 = biasF32;
            this.javaOutputF32 = javaOutputF32;
            this.leftF64 = leftF64;
            this.rightF64 = rightF64;
            this.biasF64 = biasF64;
            this.javaOutputF64 = javaOutputF64;
        }

        private static NativeBenchmarkCase create(DataType dataType, int elements) {
            NativeCpuStorageFactory storageFactory = new NativeCpuStorageFactory();
            int[] shape = new int[]{elements};
            Tensor leftTensor = tensor(new double[elements], shape, "native-a", dataType);
            Tensor rightTensor = tensor(new double[elements], shape, "native-b", dataType);
            Tensor biasTensor = tensor(new double[elements], shape, "native-c", dataType);
            Tensor outputTensor = tensor(new double[elements], shape, "native-out", dataType);
            NativeTensorStorage leftStorage = nativeStorage(storageFactory, dataType, elements, "native-fused-a");
            NativeTensorStorage rightStorage = nativeStorage(storageFactory, dataType, elements, "native-fused-b");
            NativeTensorStorage biasStorage = nativeStorage(storageFactory, dataType, elements, "native-fused-c");
            NativeTensorStorage outputStorage = nativeStorage(storageFactory, dataType, elements, "native-fused-out");

            Cpu1FusedExpressionPlan expressionPlan = nativeCheapContiguousPlan(dataType, elements);
            Cpu1FusedCodegenPlan codegenPlan = Cpu1FusedCodegenPlan.from(
                    expressionPlan,
                    dataType,
                    Cpu1LayoutKind.CONTIGUOUS,
                    Cpu1StorageKind.MEMORY_SEGMENT,
                    Cpu1FusedCodegenLoopKind.CONTIGUOUS_SCALAR,
                    Cpu1PrepareConfig.scalarMemorySegmentSingleThread()
            );
            if (codegenPlan.rejectionReason() != Cpu1FusedCodegenRejectionReason.NONE) {
                fail("Native cpu1 fused benchmark codegen rejected: " + codegenPlan.rejectionReason());
            }
            Cpu1FusedCodegenKernel kernel = Cpu1FusedCodegenKernelFactory.prepareKernel(codegenPlan);
            Cpu1PreparedFusedElementwiseUnit preparedUnit =
                    nativePreparedUnit(expressionPlan, dataType, kernel, elements);

            List<Cpu1TensorView> inputs = new ArrayList<>(3);
            inputs.add(Cpu1TensorView.fromNativeStorage(leftTensor, leftStorage));
            inputs.add(Cpu1TensorView.fromNativeStorage(rightTensor, rightStorage));
            inputs.add(Cpu1TensorView.fromNativeStorage(biasTensor, biasStorage));
            Cpu1FusedKernelArgs args = new Cpu1FusedKernelArgs(
                    preparedUnit,
                    inputs,
                    Cpu1TensorView.fromNativeStorage(outputTensor, outputStorage)
            );

            double[] leftValues = values(elements, -0.35, 0.75);
            double[] rightValues = values(elements, 0.20, 0.45);
            double[] biasValues = values(elements, -0.10, 0.25);
            if (dataType == DataType.FLOAT32) {
                return new NativeBenchmarkCase(
                        dataType,
                        elements,
                        leftStorage,
                        rightStorage,
                        biasStorage,
                        outputStorage,
                        args,
                        preparedUnit,
                        toFloatArray(leftValues),
                        toFloatArray(rightValues),
                        toFloatArray(biasValues),
                        new float[elements],
                        null,
                        null,
                        null,
                        null
                );
            }
            if (dataType == DataType.FLOAT64) {
                return new NativeBenchmarkCase(
                        dataType,
                        elements,
                        leftStorage,
                        rightStorage,
                        biasStorage,
                        outputStorage,
                        args,
                        preparedUnit,
                        null,
                        null,
                        null,
                        null,
                        leftValues,
                        rightValues,
                        biasValues,
                        new double[elements]
                );
            }
            throw new IllegalArgumentException("Unsupported native benchmark dtype: " + dataType);
        }

        private void copyJavaInputsToNative() {
            switch (dataType) {
                case FLOAT32 -> {
                    copyF32ToNative(leftF32, leftStorage);
                    copyF32ToNative(rightF32, rightStorage);
                    copyF32ToNative(biasF32, biasStorage);
                }
                case FLOAT64 -> {
                    copyF64ToNative(leftF64, leftStorage);
                    copyF64ToNative(rightF64, rightStorage);
                    copyF64ToNative(biasF64, biasStorage);
                }
                default -> throw new IllegalStateException("Unsupported native benchmark dtype: " + dataType);
            }
        }

        private void computeRange() {
            preparedUnit.generatedKernel().computeRange(args, 0, elements);
        }

        private void markNativeInputsModified() {
            leftStorage.markModified();
            rightStorage.markModified();
            biasStorage.markModified();
        }

        private void readNativeOutputToJava() {
            switch (dataType) {
                case FLOAT32 -> readNativeF32ToJava(outputStorage, javaOutputF32);
                case FLOAT64 -> readNativeF64ToJava(outputStorage, javaOutputF64);
                default -> throw new IllegalStateException("Unsupported native benchmark dtype: " + dataType);
            }
        }

        private double[] readNativeOutputAsDouble() {
            return Cpu1FusedParityBenchmarkTest.readNativeOutputAsDouble(dataType, outputStorage, elements);
        }

        @Override
        public void close() {
            leftStorage.close();
            rightStorage.close();
            biasStorage.close();
            outputStorage.close();
        }
    }
}

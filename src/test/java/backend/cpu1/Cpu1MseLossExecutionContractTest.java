package backend.cpu1;

import backend.cpu1.kernels.loss.mse.Cpu1MseLossKernelId;
import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.cpu1.storage.Cpu1StorageKind;
import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.backend.CpuKernelConfig;
import config.backend.KernelTuningConfig;
import config.runtime.CpuStorageProfile;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.execution.PreparedExecution;
import graph.execution.PreparedExecutionStep;
import graph.execution.trace.ExecutionStepTrace;
import graph.execution.trace.RunTrace;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class Cpu1MseLossExecutionContractTest {
    @Test
    void preparedGraphExecutesF32MeanMseAsSingleCpu1SpecializedStep() {
        Tensor prediction = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f},
                new int[]{4},
                null,
                "mseF32Prediction",
                DataType.FLOAT32
        );
        Tensor target = new Tensor(
                new float[]{1.5f, 1.0f, 2.5f, 3.0f},
                new int[]{4},
                null,
                "mseF32Target",
                DataType.FLOAT32
        );
        Tensor diff = prediction.sub(target);
        Tensor loss = diff.mul(diff).mean();
        PreparedExecution execution = prepare(loss, DataType.FLOAT32);

        assertSingleMseStep(execution, Cpu1MseLossKernelId.MSE_MEAN_F32_DENSE_SCALAR);
        RunTrace trace = execution.executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(new float[]{0.625f}, loss.toFloat32ArrayCopy(), 1.0e-6f);
        assertMseTrace(trace, Cpu1MseLossKernelId.MSE_MEAN_F32_DENSE_SCALAR, "MEAN", 4);
    }

    @Test
    void preparedGraphExecutesF64SumMseAsSingleCpu1SpecializedStep() {
        Tensor prediction = new Tensor(
                new double[]{1.0d, -2.0d, 0.5d, 4.0d},
                new int[]{2, 2},
                null,
                "mseF64Prediction",
                DataType.FLOAT64
        );
        Tensor target = new Tensor(
                new double[]{0.0d, -1.5d, 1.5d, 1.0d},
                new int[]{2, 2},
                null,
                "mseF64Target",
                DataType.FLOAT64
        );
        Tensor diff = prediction.sub(target);
        Tensor loss = diff.mul(diff).sum();
        PreparedExecution execution = prepare(loss, DataType.FLOAT64);

        assertSingleMseStep(execution, Cpu1MseLossKernelId.MSE_SUM_F64_DENSE_SCALAR);
        RunTrace trace = execution.executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{11.25d}, loss.toDoubleArrayCopy(), 1.0e-12);
        assertMseTrace(trace, Cpu1MseLossKernelId.MSE_SUM_F64_DENSE_SCALAR, "SUM", 4);
    }

    @Test
    void preparedGraphExecutesBf16MeanMseAsSingleCpu1SpecializedStep() {
        Tensor prediction = new Tensor(
                bf16Bits(1.0f, 2.0f, 3.0f, 4.0f),
                new int[]{4},
                null,
                "mseBf16Prediction",
                DataType.BFLOAT16
        );
        Tensor target = new Tensor(
                bf16Bits(0.5f, 1.0f, 1.5f, 2.0f),
                new int[]{4},
                null,
                "mseBf16Target",
                DataType.BFLOAT16
        );
        Tensor diff = prediction.sub(target);
        Tensor loss = diff.mul(diff).mean();
        PreparedExecution execution = prepare(loss, DataType.BFLOAT16);

        assertSingleMseStep(execution, Cpu1MseLossKernelId.MSE_MEAN_BF16_DENSE_SCALAR);
        RunTrace trace = execution.executeTraced(ExecutionMode.FORWARD);

        float actual = TensorDTypeOps.fromBFloat16Bits(loss.toBFloat16BitsArrayCopy()[0]);
        float expected = TensorDTypeOps.fromBFloat16Bits(TensorDTypeOps.toBFloat16Bits(1.875f));
        assertEquals(expected, actual, 0.0f);
        assertMseTrace(trace, Cpu1MseLossKernelId.MSE_MEAN_BF16_DENSE_SCALAR, "MEAN", 4);
    }

    @Test
    void preparedGraphExecutesNestedF32MeanMseAsSingleCpu1SpecializedStep() {
        Tensor prediction = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "mseNestedF32Prediction",
                DataType.FLOAT32
        );
        Tensor target = new Tensor(
                new float[]{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f},
                new int[]{2, 3},
                null,
                "mseNestedF32Target",
                DataType.FLOAT32
        );
        Tensor diff = prediction.sub(target);
        Tensor loss = diff.mul(diff).mean(1).mean(0, true);
        PreparedExecution execution = prepare(loss, DataType.FLOAT32);

        Cpu1PreparedArtifact artifact = assertSingleMseStep(
                execution,
                Cpu1MseLossKernelId.MSE_MEAN_F32_DENSE_SCALAR,
                4,
                6
        );
        assertEquals(4, artifact.preparedMseLossUnit().orderedNodeIds().size());
        RunTrace trace = execution.executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(new float[]{1.0f}, loss.toFloat32ArrayCopy(), 1.0e-6f);
        assertMseTrace(trace, Cpu1MseLossKernelId.MSE_MEAN_F32_DENSE_SCALAR, "MEAN", 6,
                Cpu1StorageKind.JAVA_ARRAY, 6, 2);
    }

    @Test
    void preparedGraphExecutesF32MeanMseOnNativeSegmentWhenCpuNativeIsRequested() {
        Tensor prediction = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f},
                new int[]{4},
                null,
                "mseF32NativePrediction",
                DataType.FLOAT32
        );
        Tensor target = new Tensor(
                new float[]{1.5f, 1.0f, 2.5f, 3.0f},
                new int[]{4},
                null,
                "mseF32NativeTarget",
                DataType.FLOAT32
        );
        Tensor diff = prediction.sub(target);
        Tensor loss = diff.mul(diff).mean();
        PreparedExecution execution = prepare(
                loss,
                RuntimeConfig.inferenceDefaults(DataType.FLOAT32).withCpuStorageProfile(CpuStorageProfile.CPU_NATIVE)
        );

        Cpu1PreparedArtifact artifact = assertSingleMseStep(execution, Cpu1MseLossKernelId.MSE_MEAN_F32_DENSE_SCALAR);
        assertEquals(Cpu1StorageKind.MEMORY_SEGMENT, artifact.preparedMseLossUnit().storageKind());
        RunTrace trace = execution.executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(new float[]{0.625f}, loss.toFloat32ArrayCopy(), 1.0e-6f);
        assertMseTrace(trace, Cpu1MseLossKernelId.MSE_MEAN_F32_DENSE_SCALAR, "MEAN", 4, Cpu1StorageKind.MEMORY_SEGMENT);
    }

    @Test
    void preparedGraphExecutesNestedF32MeanMseOnNativeSegmentAsSingleStep() {
        Tensor prediction = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "mseNestedF32NativePrediction",
                DataType.FLOAT32
        );
        Tensor target = new Tensor(
                new float[]{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f},
                new int[]{2, 3},
                null,
                "mseNestedF32NativeTarget",
                DataType.FLOAT32
        );
        Tensor diff = prediction.sub(target);
        Tensor loss = diff.mul(diff).mean(1).mean(0, true);
        PreparedExecution execution = prepare(
                loss,
                RuntimeConfig.inferenceDefaults(DataType.FLOAT32).withCpuStorageProfile(CpuStorageProfile.CPU_NATIVE)
        );

        Cpu1PreparedArtifact artifact = assertSingleMseStep(
                execution,
                Cpu1MseLossKernelId.MSE_MEAN_F32_DENSE_SCALAR,
                4,
                6
        );
        assertEquals(Cpu1StorageKind.MEMORY_SEGMENT, artifact.preparedMseLossUnit().storageKind());
        RunTrace trace = execution.executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(new float[]{1.0f}, loss.toFloat32ArrayCopy(), 1.0e-6f);
        assertMseTrace(trace, Cpu1MseLossKernelId.MSE_MEAN_F32_DENSE_SCALAR, "MEAN", 6,
                Cpu1StorageKind.MEMORY_SEGMENT, 6, 2);
    }

    @Test
    void preparedGraphExecutesF64SumMseOnNativeSegmentWhenCpuNativeIsRequested() {
        Tensor prediction = new Tensor(
                new double[]{1.0d, -2.0d, 0.5d, 4.0d},
                new int[]{2, 2},
                null,
                "mseF64NativePrediction",
                DataType.FLOAT64
        );
        Tensor target = new Tensor(
                new double[]{0.0d, -1.5d, 1.5d, 1.0d},
                new int[]{2, 2},
                null,
                "mseF64NativeTarget",
                DataType.FLOAT64
        );
        Tensor diff = prediction.sub(target);
        Tensor loss = diff.mul(diff).sum();
        PreparedExecution execution = prepare(
                loss,
                RuntimeConfig.inferenceDefaults(DataType.FLOAT64).withCpuStorageProfile(CpuStorageProfile.CPU_NATIVE)
        );

        Cpu1PreparedArtifact artifact = assertSingleMseStep(execution, Cpu1MseLossKernelId.MSE_SUM_F64_DENSE_SCALAR);
        assertEquals(Cpu1StorageKind.MEMORY_SEGMENT, artifact.preparedMseLossUnit().storageKind());
        RunTrace trace = execution.executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{11.25d}, loss.toDoubleArrayCopy(), 1.0e-12);
        assertMseTrace(trace, Cpu1MseLossKernelId.MSE_SUM_F64_DENSE_SCALAR, "SUM", 4, Cpu1StorageKind.MEMORY_SEGMENT);
    }

    @Test
    void preparedGraphExecutesLargeF32MeanMseWithParallelPartialReduction() {
        assumeTrue(Runtime.getRuntime().availableProcessors() > 1, "parallel cpu1 MSE requires more than one worker");
        int elements = 20_000;
        float[] predictionValues = new float[elements];
        float[] targetValues = new float[elements];
        for (int i = 0; i < elements; i++) {
            predictionValues[i] = (i & 1) == 0 ? 2.0f : -2.0f;
            targetValues[i] = (i & 1) == 0 ? 1.0f : -3.0f;
        }
        Tensor prediction = new Tensor(predictionValues, new int[]{elements}, null, "mseParallelPrediction", DataType.FLOAT32);
        Tensor target = new Tensor(targetValues, new int[]{elements}, null, "mseParallelTarget", DataType.FLOAT32);
        Tensor diff = prediction.sub(target);
        Tensor loss = diff.mul(diff).mean();
        PreparedExecution execution = prepare(loss, lowThresholdRuntime(DataType.FLOAT32));

        Cpu1PreparedArtifact artifact = assertSingleMseStep(execution, Cpu1MseLossKernelId.MSE_MEAN_F32_DENSE_SCALAR);
        assertEquals(Cpu1StorageKind.JAVA_ARRAY, artifact.preparedMseLossUnit().storageKind());
        assumeTrue(artifact.preparedMseLossUnit().launchConfig().workerCount() > 1, "parallel launch was not selected");
        RunTrace trace = execution.executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(new float[]{1.0f}, loss.toFloat32ArrayCopy(), 1.0e-6f);
        assertMseTrace(trace, Cpu1MseLossKernelId.MSE_MEAN_F32_DENSE_SCALAR, "MEAN", elements);
        assertMseLaunchWorkers(trace, artifact.preparedMseLossUnit().launchConfig().workerCount());
    }

    @Test
    void cpuNativeBf16MseFailsWithExplicitUnsupportedStorageMessage() {
        Tensor prediction = new Tensor(
                bf16Bits(1.0f, 2.0f, 3.0f, 4.0f),
                new int[]{4},
                null,
                "mseBf16NativePrediction",
                DataType.BFLOAT16
        );
        Tensor target = new Tensor(
                bf16Bits(0.5f, 1.0f, 1.5f, 2.0f),
                new int[]{4},
                null,
                "mseBf16NativeTarget",
                DataType.BFLOAT16
        );
        Tensor diff = prediction.sub(target);
        Tensor loss = diff.mul(diff).mean();
        RuntimeConfig nativeBf16 = RuntimeConfig.inferenceDefaults(DataType.BFLOAT16)
                .withCpuStorageProfile(CpuStorageProfile.CPU_NATIVE);

        UnsupportedOperationException exception = assertThrows(
                UnsupportedOperationException.class,
                () -> prepare(loss, nativeBf16)
        );
        assertEquals(
                "cpu1 MSE_LOSS MEMORY_SEGMENT supports FLOAT32/FLOAT64 only; BFLOAT16 remains on the JAVA_ARRAY path.",
                exception.getMessage()
        );
    }

    private static PreparedExecution prepare(Tensor loss, DataType dataType) {
        return prepare(loss, RuntimeConfig.inferenceDefaults(dataType));
    }

    private static PreparedExecution prepare(Tensor loss, RuntimeConfig runtimeConfig) {
        CompiledGraph graph = CompiledGraph.compile(loss, CompileConfig.inference());
        return graph.prepare(runtimeConfig);
    }

    private static Cpu1PreparedArtifact assertSingleMseStep(
            PreparedExecution execution,
            Cpu1MseLossKernelId expectedKernelId
    ) {
        return assertSingleMseStep(execution, expectedKernelId, 3, -1);
    }

    private static Cpu1PreparedArtifact assertSingleMseStep(
            PreparedExecution execution,
            Cpu1MseLossKernelId expectedKernelId,
            int expectedMseNodeCount,
            long expectedReductionDivisor
    ) {
        assertEquals(2, execution.forwardSteps().size());
        PreparedExecutionStep step = execution.forwardSteps().stream()
                .filter(candidate -> candidate.metadata().artifact() instanceof Cpu1PreparedArtifact)
                .findFirst()
                .orElseThrow();
        Cpu1PreparedArtifact artifact = assertInstanceOf(Cpu1PreparedArtifact.class, step.metadata().artifact());
        assertEquals(expectedKernelId, artifact.preparedMseLossUnit().kernelId());
        assertEquals(expectedMseNodeCount, step.orderedNodeIds().size());
        if (expectedReductionDivisor > 0) {
            assertEquals(expectedReductionDivisor, artifact.preparedMseLossUnit().reductionDivisor());
        }
        return artifact;
    }

    private static void assertMseTrace(
            RunTrace trace,
            Cpu1MseLossKernelId expectedKernelId,
            String expectedReduction,
            int expectedElementCount
    ) {
        assertMseTrace(trace, expectedKernelId, expectedReduction, expectedElementCount, Cpu1StorageKind.JAVA_ARRAY);
    }

    private static void assertMseTrace(
            RunTrace trace,
            Cpu1MseLossKernelId expectedKernelId,
            String expectedReduction,
            int expectedElementCount,
            Cpu1StorageKind expectedStorageKind
    ) {
        long expectedDivisor = expectedReduction.equals("MEAN") ? expectedElementCount : 1L;
        assertMseTrace(trace, expectedKernelId, expectedReduction, expectedElementCount, expectedStorageKind,
                expectedDivisor, 1);
    }

    private static void assertMseTrace(
            RunTrace trace,
            Cpu1MseLossKernelId expectedKernelId,
            String expectedReduction,
            int expectedElementCount,
            Cpu1StorageKind expectedStorageKind,
            long expectedReductionDivisor,
            int expectedReductionNodeCount
    ) {
        assertEquals(2, trace.steps().size());
        ExecutionStepTrace step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("cpu1MseLossKernelId"))
                .findFirst()
                .orElseThrow();
        Map<String, Object> attrs = step.metadata().attributes();
        assertEquals(expectedKernelId.name(), attrs.get("cpu1MseLossKernelId"));
        assertEquals("MSE_LOSS", attrs.get("cpu1SpecializationKind"));
        assertEquals(expectedStorageKind.name(), attrs.get("cpu1StorageKind"));
        assertEquals(expectedReduction, attrs.get("cpu1MseLossReduction"));
        assertEquals(expectedElementCount, attrs.get("cpu1MseLossElementCount"));
        assertEquals(expectedReductionDivisor, attrs.get("cpu1MseLossReductionDivisor"));
        assertEquals(expectedReductionNodeCount, attrs.get("cpu1MseLossReductionNodeCount"));
    }

    private static void assertMseLaunchWorkers(RunTrace trace, int expectedWorkers) {
        ExecutionStepTrace step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("cpu1MseLossKernelId"))
                .findFirst()
                .orElseThrow();
        assertEquals(expectedWorkers, step.metadata().attributes().get("cpu1MseLossLaunchWorkers"));
    }

    private static RuntimeConfig lowThresholdRuntime(DataType dataType) {
        RuntimeConfig base = RuntimeConfig.inferenceDefaults(dataType);
        CpuKernelConfig cpu = new CpuKernelConfig(1, 16, 16, 16, 1, 1);
        KernelTuningConfig kernel = new KernelTuningConfig(cpu, base.kernel().cuda(), base.kernel().opencl());
        return new RuntimeConfig(
                kernel,
                base.approximation(),
                base.blas(),
                base.conv2d(),
                base.fused(),
                base.accelerator(),
                base.cpuStorageProfile(),
                base.nativeCpuFailurePolicy(),
                base.deviceTransferPolicy(),
                base.nativeCpuMemory(),
                base.bfloat16TrainingPolicy()
        );
    }

    private static short[] bf16Bits(float... values) {
        short[] bits = new short[values.length];
        for (int i = 0; i < values.length; i++) {
            bits[i] = TensorDTypeOps.toBFloat16Bits(values[i]);
        }
        return bits;
    }
}

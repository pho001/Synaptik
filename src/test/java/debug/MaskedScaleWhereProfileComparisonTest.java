package debug;

import backend.runtime.ExecutionMode;
import config.backend.CpuKernelConfig;
import config.backend.KernelTuningConfig;
import config.compile.CompileConfig;
import config.runtime.ApproximationConfig;
import config.runtime.BlasConfig;
import config.runtime.FusedExecutionPolicy;
import config.runtime.RuntimeConfig;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tuning.benchmark.report.TextBenchmarkReportRenderer;
import tuning.benchmark.BenchmarkEntry;
import tuning.benchmark.BenchmarkRequest;
import tuning.benchmark.BenchmarkSession;
import tuning.validate.ValidationPolicy;
import tuning.workload.TensorRootWorkloadSpec;
import tuning.workload.WorkloadKind;

import java.util.List;

final class MaskedScaleWhereProfileComparisonTest {
    private static final int SIZE = 4 * 8 * 64 * 64;
    private static final tuning.measure.MeasurementPolicy COMPARISON_MEASUREMENT = DebugMeasurementPolicies.STANDARD;

    @Test
    void compareMaskedScaleWhereProfiles() {
        var request = new BenchmarkRequest(
                maskedScaleWhereWorkload("compare_masked_scale_where_profiles"),
                List.of(
                        BenchmarkEntry.baseline("default-inference", runtimeProfile("default-inference", RuntimeConfig.inferenceDefaults())),
                        BenchmarkEntry.candidate("noncheap-strided-w2", runtimeWithFusedAsmWidths(1, 1, 1, 2, -1)),
                        BenchmarkEntry.candidate("noncheap-strided-w4", runtimeWithFusedAsmWidths(1, 1, 1, 4, -1)),
                        BenchmarkEntry.candidate("noncheap-strided-w4-vector-only", runtimeWithFusedAsmWidths(1, 1, 1, 4, Integer.MAX_VALUE))
                ),
                COMPARISON_MEASUREMENT,
                ValidationPolicy.disabled(),
                tuning.reporting.ReportPolicy.defaults()
        );

        var report = BenchmarkSession.create(request).run();
        System.out.println();
        System.out.println(TextBenchmarkReportRenderer.render(report));
    }

    private static TensorRootWorkloadSpec maskedScaleWhereWorkload(String name) {
        return new TensorRootWorkloadSpec(
                name,
                WorkloadKind.GENERIC,
                environment -> {
                    byte[] maskValues = new byte[SIZE];
                    float[] valueValues = new float[SIZE];
                    for (int i = 0; i < SIZE; i++) {
                        maskValues[i] = (byte) ((i & 3) == 0 ? 1 : 0);
                        valueValues[i] = (float) (Math.sin(i * 0.03125) + (i % 11) * 0.125);
                    }

                    Tensor mask = new Tensor(maskValues, new int[]{SIZE}, null, "mask", DataType.BOOL);
                    Tensor fill = new Tensor(new float[]{-1000.0f}, new int[]{1}, null, "fill", DataType.FLOAT32);
                    Tensor values = new Tensor(valueValues, new int[]{SIZE}, null, "values", DataType.FLOAT32);
                    return Tensor.where(mask, values.mul(0.25), fill);
                }
        );
    }

    private static config.profile.ExecutionProfile runtimeProfile(String candidateName, RuntimeConfig runtime) {
        return new config.profile.ExecutionProfile(
                candidateName,
                candidateName,
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                CompileConfig.inference(),
                runtime,
                config.profile.WorkloadProfile.none()
        );
    }

    private static config.profile.ExecutionProfile runtimeWithFusedAsmWidths(
            int cheapContiguous,
            int cheapStrided,
            int nonCheapContiguous,
            int nonCheapStrided,
            int fusedParallelMinOverride
    ) {
        return runtimeProfile(
                "masked-scale-where",
                runtimeWithFusedAsmWidthsRaw(
                        cheapContiguous,
                        fusedParallelMinOverride
                )
        );
    }

    private static RuntimeConfig runtimeWithFusedAsmWidthsRaw(
            int fusedAsmVectorWidth,
            int fusedParallelMinOverride
    ) {
        var base = RuntimeConfig.inferenceDefaults();
        CpuKernelConfig cpu = base.kernel().cpu();
        int fusedCheapParallelMinSize = fusedParallelMinOverride > 0
                ? fusedParallelMinOverride
                : cpu.fusedCheapParallelMinSize();
        return new RuntimeConfig(
                new KernelTuningConfig(
                        new CpuKernelConfig(
                                cpu.loopUnrollFactor(),
                                cpu.matMulTileM(),
                                cpu.matMulTileN(),
                                cpu.matMulTileK(),
                                cpu.cheapVectorMinSize(),
                                cpu.transcendentalVectorMinSize(),
                                cpu.fusedCheapVectorMinSize(),
                                cpu.fusedTranscendentalVectorMinSize(),
                                cpu.reductionVectorMinSize(),
                                cpu.cheapParallelMinSize(),
                                cpu.transcendentalParallelMinSize(),
                                fusedCheapParallelMinSize,
                                cpu.fusedTranscendentalParallelMinSize(),
                                cpu.reductionParallelMinSize(),
                                cpu.contiguousMaterializeThreshold(),
                                cpu.lowCostTargetChunksPerWorker(),
                                cpu.mediumCostTargetChunksPerWorker(),
                                cpu.highCostTargetChunksPerWorker(),
                                cpu.minScalarChunkSize(),
                                cpu.minVectorChunkSize(),
                                cpu.minReductionChunkSize(),
                                cpu.commonPoolLowCostMaxWorkPerWorker(),
                                fusedAsmVectorWidth,
                                cpu.sumAccuracyMode(),
                                cpu.matMulParallelMinSize(),
                                cpu.attentionMatMulPolicy()
                        ),
                        base.kernel().cuda(),
                        base.kernel().opencl()
                ),
                ApproximationConfig.defaults(),
                BlasConfig.disabled(),
                new FusedExecutionPolicy(true)
        );
    }
}

package debug;

import runtime.contract.ExecutionMode;
import config.compile.CompileConfig;
import config.compile.GraphOptimizationConfig;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
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

final class AttentionLoweringProfileComparisonTest {
    private static final int BATCH = 4;
    private static final int HEADS = 8;
    private static final int TOKENS = 64;
    private static final int HEAD_DIM = 32;
    private static final int VALUE_DIM = 32;
    private static final double SCALE = 1.0d / Math.sqrt(HEAD_DIM);
    private static final tuning.measure.MeasurementPolicy MEASUREMENT = DebugMeasurementPolicies.STANDARD;

    @Test
    void benchmarkMaskedAttentionForwardF32() {
        runBenchmark(maskedAttentionWorkload(DataType.FLOAT32, ExecutionMode.FORWARD), DataType.FLOAT32, ExecutionMode.FORWARD);
    }

    @Test
    void benchmarkMaskedAttentionForwardF64() {
        runBenchmark(maskedAttentionWorkload(DataType.FLOAT64, ExecutionMode.FORWARD), DataType.FLOAT64, ExecutionMode.FORWARD);
    }

    @Test
    void benchmarkMaskedAttentionForwardBackwardF32() {
        runBenchmark(maskedAttentionWorkload(DataType.FLOAT32, ExecutionMode.FORWARD_BACKWARD), DataType.FLOAT32, ExecutionMode.FORWARD_BACKWARD);
    }

    @Test
    void benchmarkMaskedAttentionForwardBackwardF64() {
        runBenchmark(maskedAttentionWorkload(DataType.FLOAT64, ExecutionMode.FORWARD_BACKWARD), DataType.FLOAT64, ExecutionMode.FORWARD_BACKWARD);
    }

    private static void runBenchmark(TensorRootWorkloadSpec workload, DataType dataType, ExecutionMode mode) {
        BenchmarkRequest request = new BenchmarkRequest(
                workload,
                List.of(
                        BenchmarkEntry.baseline("manual-no-opt", baselineProfile(dataType, mode)),
                        BenchmarkEntry.candidate("manual-ar-lowered", loweredProfile(dataType, mode))
                ),
                MEASUREMENT,
                ValidationPolicy.disabled(),
                tuning.reporting.ReportPolicy.defaults()
        );

        var report = BenchmarkSession.create(request).run();
        System.out.println();
        System.out.println(TextBenchmarkReportRenderer.render(report));
    }

    private static ExecutionProfile baselineProfile(DataType dataType, ExecutionMode mode) {
        return new ExecutionProfile(
                "manual-no-opt",
                "manual-no-opt",
                dataType,
                mode,
                CompileConfig.noGraphOptimizationBaseline(),
                mode == ExecutionMode.FORWARD ? RuntimeConfig.inferenceDefaults() : RuntimeConfig.trainingDefaults(),
                WorkloadProfile.none()
        );
    }

    private static ExecutionProfile loweredProfile(DataType dataType, ExecutionMode mode) {
        return new ExecutionProfile(
                "manual-ar-lowered",
                "manual-ar-lowered",
                dataType,
                mode,
                (mode == ExecutionMode.FORWARD ? CompileConfig.inference() : CompileConfig.training())
                        .withGraphOptimization(config.compile.GraphOptimizationConfig.stages(true, false, false, false, false)),
                mode == ExecutionMode.FORWARD ? RuntimeConfig.inferenceDefaults() : RuntimeConfig.trainingDefaults(),
                WorkloadProfile.none()
        );
    }

    private static TensorRootWorkloadSpec maskedAttentionWorkload(DataType dataType, ExecutionMode mode) {
        return new TensorRootWorkloadSpec(
                "masked_attention_manual_pattern_" + dataType.name().toLowerCase() + "_" + mode.name().toLowerCase(),
                WorkloadKind.GENERIC,
                environment -> {
                    Tensor q = tensor(queryDataF64(), new int[]{BATCH, HEADS, TOKENS, HEAD_DIM}, "q", dataType);
                    Tensor k = tensor(keyDataF64(), new int[]{BATCH, HEADS, TOKENS, HEAD_DIM}, "k", dataType);
                    Tensor v = tensor(valueDataF64(), new int[]{BATCH, HEADS, TOKENS, VALUE_DIM}, "v", dataType);
                    if (mode == ExecutionMode.FORWARD_BACKWARD) {
                        q.setRequiresGrad(true);
                        k.setRequiresGrad(true);
                        v.setRequiresGrad(true);
                    }
                    Tensor mask = new Tensor(maskData(), new int[]{BATCH, HEADS, TOKENS, TOKENS}, null, "mask", DataType.BOOL);
                    Tensor scores = q.matmul(k.permute(0, 1, 3, 2)).mul(SCALE);
                    Tensor out = Tensor.where(mask, scores, Tensor.scalar(maskFillValue(dataType), dataType))
                            .softmax(3)
                            .matmul(v);
                    return mode == ExecutionMode.FORWARD ? out : out.sum();
                }
        );
    }

    private static Tensor tensor(double[] values, int[] shape, String label, DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> new Tensor(values.clone(), shape, null, label, DataType.FLOAT64);
            case FLOAT32 -> new Tensor(toF32(values), shape, null, label, DataType.FLOAT32);
            case BFLOAT16 -> throw new IllegalArgumentException("BF16 benchmark is not used here.");
            case INT32, INT64, BOOL -> throw new IllegalArgumentException("attention benchmark requires floating dtype");
        };
    }

    private static double maskFillValue(DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> -1.0e30d;
            case FLOAT32 -> -1.0e9d;
            case BFLOAT16 -> -1.0e30d;
            case INT32, INT64, BOOL -> throw new IllegalArgumentException("attention benchmark requires floating dtype");
        };
    }

    private static float[] toF32(double[] src) {
        float[] out = new float[src.length];
        for (int i = 0; i < src.length; i++) {
            out[i] = (float) src[i];
        }
        return out;
    }

    private static double[] queryDataF64() {
        int size = BATCH * HEADS * TOKENS * HEAD_DIM;
        double[] out = new double[size];
        for (int i = 0; i < size; i++) {
            out[i] = Math.sin(i * 0.013) + Math.cos(i * 0.003) * 0.25 + (i % 7) * 0.03125;
        }
        return out;
    }

    private static double[] keyDataF64() {
        int size = BATCH * HEADS * TOKENS * HEAD_DIM;
        double[] out = new double[size];
        for (int i = 0; i < size; i++) {
            out[i] = Math.cos(i * 0.017) + Math.sin(i * 0.005) * 0.2 + (i % 11) * 0.015625;
        }
        return out;
    }

    private static double[] valueDataF64() {
        int size = BATCH * HEADS * TOKENS * VALUE_DIM;
        double[] out = new double[size];
        for (int i = 0; i < size; i++) {
            out[i] = Math.sin(i * 0.019) * 0.75 + Math.cos(i * 0.007) * 0.5 + ((i / VALUE_DIM) % 13) * 0.015625;
        }
        return out;
    }

    private static byte[] maskData() {
        int size = BATCH * HEADS * TOKENS * TOKENS;
        byte[] out = new byte[size];
        for (int batch = 0; batch < BATCH; batch++) {
            for (int head = 0; head < HEADS; head++) {
                for (int q = 0; q < TOKENS; q++) {
                    for (int k = 0; k < TOKENS; k++) {
                        int index = ((batch * HEADS + head) * TOKENS + q) * TOKENS + k;
                        out[index] = (byte) ((k <= q || ((q + k + head) & 3) != 0) ? 1 : 0);
                    }
                }
            }
        }
        return out;
    }
}

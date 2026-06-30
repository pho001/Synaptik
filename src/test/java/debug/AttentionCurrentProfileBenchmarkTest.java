package debug;

import runtime.contract.ExecutionMode;
import config.compile.CompileConfig;
import config.compile.GraphOptimizationConfig;
import config.profile.ExecutionProfile;
import config.profile.ExecutionProfileAssembler;
import config.profile.GraphExecutionPolicy;
import config.profile.PlatformRuntimeProfile;
import config.profile.PlatformRuntimeProfileIO;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tuning.benchmark.report.TextBenchmarkReportRenderer;
import tuning.benchmark.BenchmarkEntry;
import tuning.benchmark.BenchmarkRequest;
import tuning.benchmark.BenchmarkSession;
import tuning.calibration.store.PlatformCalibrationLayout;
import tuning.calibration.store.PlatformCalibrationPaths;
import tuning.workload.TensorRootWorkloadSpec;
import tuning.workload.WorkloadKind;

import java.nio.file.Path;
import java.util.List;

final class AttentionCurrentProfileBenchmarkTest {
    private static final int BATCH = 4;
    private static final int HEADS = 8;
    private static final int TOKENS = 64;
    private static final int HEAD_DIM = 32;
    private static final int VALUE_DIM = 32;
    private static final double SCALE = 1.0d / Math.sqrt(HEAD_DIM);
    private static final tuning.measure.MeasurementPolicy MEASUREMENT = DebugMeasurementPolicies.STANDARD_WITH_TRACE;

    @Test
    void benchmarkCurrentProfileAttentionF32ForwardBackward() {
        ExecutionProfile seed = new ExecutionProfile(
                "platform-seed-f32-training",
                "platform-seed-f32-training",
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                CompileConfig.training(),
                config.runtime.RuntimeConfig.trainingDefaults(),
                WorkloadProfile.none()
        );
        PlatformCalibrationLayout layout = PlatformCalibrationPaths.defaultLayout(
                Path.of("build", "platform-calibration", "f32"),
                seed
        );
        PlatformRuntimeProfile fallback = PlatformRuntimeProfile.fromExecutionProfile(
                layout.platformId(),
                layout.hardware().key(),
                "fallback",
                seed
        );
        PlatformRuntimeProfile current = PlatformRuntimeProfileIO.loadOrDefault(layout.profilePath(), fallback);
        GraphExecutionPolicy loweredPolicy = GraphExecutionPolicy.of(
                CompileConfig.training().withGraphOptimization(GraphOptimizationConfig.stages(true, false, false, false, false))
        );

        BenchmarkRequest request = new BenchmarkRequest(
                maskedAttentionWorkload(),
                List.of(
                        BenchmarkEntry.baseline("manual-no-opt", baselineProfile()),
                        BenchmarkEntry.candidate(
                                "manual-ar-current-profile",
                                ExecutionProfileAssembler.assemble(
                                        "manual-ar-current-profile",
                                        "manual-ar-current-profile",
                                        DataType.FLOAT32,
                                        ExecutionMode.FORWARD_BACKWARD,
                                        current,
                                        loweredPolicy
                                )
                        )
                ),
                MEASUREMENT,
                tuning.validate.ValidationPolicy.disabled(),
                tuning.reporting.ReportPolicy.defaults()
        );

        var report = BenchmarkSession.create(request).run();
        System.out.println("ATTENTION_CURRENT_PROFILE_BENCHMARK");
        System.out.println("profilePath=" + layout.profilePath());
        System.out.println(TextBenchmarkReportRenderer.render(report));
    }

    private static ExecutionProfile baselineProfile() {
        return new ExecutionProfile(
                "manual-no-opt",
                "manual-no-opt",
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                CompileConfig.noGraphOptimizationBaseline(),
                config.runtime.RuntimeConfig.noOptNoVecNoPar(),
                WorkloadProfile.none()
        );
    }

    private static TensorRootWorkloadSpec maskedAttentionWorkload() {
        return new TensorRootWorkloadSpec(
                "masked_attention_manual_pattern_float32_forward_backward",
                WorkloadKind.GENERIC,
                environment -> {
                    Tensor q = new Tensor(toF32(queryDataF64()), new int[]{BATCH, HEADS, TOKENS, HEAD_DIM}, null, "q", DataType.FLOAT32);
                    Tensor k = new Tensor(toF32(keyDataF64()), new int[]{BATCH, HEADS, TOKENS, HEAD_DIM}, null, "k", DataType.FLOAT32);
                    Tensor v = new Tensor(toF32(valueDataF64()), new int[]{BATCH, HEADS, TOKENS, VALUE_DIM}, null, "v", DataType.FLOAT32);
                    q.setRequiresGrad(true);
                    k.setRequiresGrad(true);
                    v.setRequiresGrad(true);
                    Tensor mask = new Tensor(maskData(), new int[]{BATCH, HEADS, TOKENS, TOKENS}, null, "mask", DataType.BOOL);
                    Tensor scores = q.matmul(k.permute(0, 1, 3, 2)).mul(SCALE);
                    Tensor out = Tensor.where(mask, scores, Tensor.scalar(-1.0e9d, DataType.FLOAT32))
                            .softmax(3)
                            .matmul(v);
                    return out.sum();
                }
        );
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

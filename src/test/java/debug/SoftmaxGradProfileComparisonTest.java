package debug;

import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import config.runtime.RuntimeConfig;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tuning.report.TextBenchmarkReportRenderer;
import tuning.session.BenchmarkEntry;
import tuning.session.BenchmarkRequest;
import tuning.session.BenchmarkSession;
import tuning.validate.ValidationPolicy;
import tuning.workload.TensorRootWorkloadSpec;
import tuning.workload.WorkloadKind;

import java.util.List;

final class SoftmaxGradProfileComparisonTest {
    private static final int BATCH = 4;
    private static final int HEADS = 8;
    private static final int TOKENS = 64;
    private static final int AXIS = 64;
    private static final tuning.measure.MeasurementPolicy MEASUREMENT = DebugMeasurementPolicies.STANDARD;

    @Test
    void benchmarkSoftmaxGradForwardBackward() {
        runBenchmark("softmax-grad", softmaxGradWorkload());
    }

    @Test
    void benchmarkLogSoftmaxGradForwardBackward() {
        runBenchmark("log-softmax-grad", logSoftmaxGradWorkload());
    }

    private static void runBenchmark(String name, TensorRootWorkloadSpec workload) {
        ExecutionProfile canonicalProfile = new ExecutionProfile(
                name + "-canonical",
                name + "-canonical",
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                OptimizerConfig.noOptimization(),
                RuntimeConfig.trainingDefaults(),
                WorkloadProfile.none()
        );
        ExecutionProfile specializedProfile = new ExecutionProfile(
                name + "-specialized",
                name + "-specialized",
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                OptimizerConfig.trainingDefaults(),
                RuntimeConfig.trainingDefaults(),
                WorkloadProfile.none()
        );

        BenchmarkRequest request = new BenchmarkRequest(
                workload,
                List.of(
                        BenchmarkEntry.candidate("canonical", canonicalProfile),
                        BenchmarkEntry.candidate("specialized", specializedProfile)
                ),
                MEASUREMENT,
                ValidationPolicy.disabled(),
                tuning.report.ReportPolicy.defaults()
        );

        var report = BenchmarkSession.create(request).run();
        System.out.println();
        System.out.println(TextBenchmarkReportRenderer.render(report));
    }

    private static TensorRootWorkloadSpec softmaxGradWorkload() {
        return new TensorRootWorkloadSpec(
                "softmax_grad_forward_backward",
                WorkloadKind.GENERIC,
                environment -> {
                    Tensor logits = new Tensor(logitsData(), new int[]{BATCH, HEADS, TOKENS, AXIS}, null, "logits", DataType.FLOAT32);
                    logits.setRequiresGrad(true);
                    Tensor weights = new Tensor(weightsData(), new int[]{BATCH, HEADS, TOKENS, AXIS}, null, "weights", DataType.FLOAT32);
                    return logits.softmax(3).mul(weights).sum();
                }
        );
    }

    private static TensorRootWorkloadSpec logSoftmaxGradWorkload() {
        return new TensorRootWorkloadSpec(
                "log_softmax_grad_forward_backward",
                WorkloadKind.GENERIC,
                environment -> {
                    Tensor logits = new Tensor(logitsData(), new int[]{BATCH, HEADS, TOKENS, AXIS}, null, "logits", DataType.FLOAT32);
                    logits.setRequiresGrad(true);
                    Tensor weights = new Tensor(weightsData(), new int[]{BATCH, HEADS, TOKENS, AXIS}, null, "weights", DataType.FLOAT32);
                    return logits.logSoftmax(3).mul(weights).sum();
                }
        );
    }

    private static float[] logitsData() {
        int size = BATCH * HEADS * TOKENS * AXIS;
        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            out[i] = (float) (Math.sin(i * 0.03125) + Math.cos(i * 0.0078125) * 0.25 + (i % 17) * 0.05);
        }
        return out;
    }

    private static float[] weightsData() {
        int size = BATCH * HEADS * TOKENS * AXIS;
        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            out[i] = (float) (Math.sin(i * 0.013) * 0.75 + Math.cos(i * 0.021) * 0.5 + ((i / AXIS) % 11) * 0.03125);
        }
        return out;
    }
}

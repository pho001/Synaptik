package debug;

import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
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

final class SoftmaxProfileComparisonTest {
    private static final int BATCH = 4;
    private static final int HEADS = 8;
    private static final int TOKENS = 64;
    private static final int AXIS = 64;
    private static final tuning.measure.MeasurementPolicy MEASUREMENT = DebugMeasurementPolicies.STANDARD;

    @Test
    void benchmarkTransformerLikeSoftmax() {
        var profile = new ExecutionProfile(
                "softmax-transformer-like",
                "softmax-transformer-like",
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                CompileConfig.inference(),
                RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );

        var request = new BenchmarkRequest(
                softmaxWorkload("benchmark_transformer_like_softmax"),
                List.of(BenchmarkEntry.candidate("default-inference", profile)),
                MEASUREMENT,
                ValidationPolicy.disabled(),
                tuning.reporting.ReportPolicy.defaults()
        );

        var report = BenchmarkSession.create(request).run();
        System.out.println();
        System.out.println(TextBenchmarkReportRenderer.render(report));
    }

    private static TensorRootWorkloadSpec softmaxWorkload(String name) {
        return new TensorRootWorkloadSpec(
                name,
                WorkloadKind.GENERIC,
                environment -> {
                    int size = BATCH * HEADS * TOKENS * AXIS;
                    float[] values = new float[size];
                    for (int i = 0; i < size; i++) {
                        values[i] = (float) (Math.sin(i * 0.03125) + Math.cos(i * 0.0078125) * 0.25 + (i % 17) * 0.05);
                    }
                    Tensor logits = new Tensor(values, new int[]{BATCH, HEADS, TOKENS, AXIS}, null, "logits", DataType.FLOAT32);
                    return logits.softmax(logits.getShapeUnsafe().length - 1);
                }
        );
    }
}

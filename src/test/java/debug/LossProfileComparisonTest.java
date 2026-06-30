package debug;

import runtime.contract.ExecutionMode;
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

final class LossProfileComparisonTest {
    private static final int BATCH = 2048;
    private static final int CLASSES = 256;
    private static final tuning.measure.MeasurementPolicy MEASUREMENT = DebugMeasurementPolicies.STANDARD;

    @Test
    void benchmarkDenseCrossEntropyLoss() {
        var profile = new ExecutionProfile(
                "dense-cross-entropy-loss",
                "dense-cross-entropy-loss",
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                CompileConfig.inference(),
                RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );

        var request = new BenchmarkRequest(
                denseCrossEntropyWorkload("benchmark_dense_cross_entropy_loss"),
                List.of(BenchmarkEntry.candidate("default-inference", profile)),
                MEASUREMENT,
                ValidationPolicy.disabled(),
                tuning.reporting.ReportPolicy.defaults()
        );

        var report = BenchmarkSession.create(request).run();
        System.out.println();
        System.out.println(TextBenchmarkReportRenderer.render(report));
    }

    @Test
    void benchmarkDenseNllLoss() {
        var profile = new ExecutionProfile(
                "dense-nll-loss",
                "dense-nll-loss",
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                CompileConfig.inference(),
                RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );

        var request = new BenchmarkRequest(
                denseNllWorkload("benchmark_dense_nll_loss"),
                List.of(BenchmarkEntry.candidate("default-inference", profile)),
                MEASUREMENT,
                ValidationPolicy.disabled(),
                tuning.reporting.ReportPolicy.defaults()
        );

        var report = BenchmarkSession.create(request).run();
        System.out.println();
        System.out.println(TextBenchmarkReportRenderer.render(report));
    }

    private static TensorRootWorkloadSpec denseCrossEntropyWorkload(String name) {
        return new TensorRootWorkloadSpec(
                name,
                WorkloadKind.GENERIC,
                environment -> {
                    float[] logitsData = logitsData();
                    float[] targetData = oneHotTargetData();
                    Tensor logits = new Tensor(logitsData, new int[]{BATCH, CLASSES}, null, "logits", DataType.FLOAT32);
                    Tensor targets = new Tensor(targetData, new int[]{BATCH, CLASSES}, null, "targets", DataType.FLOAT32);
                    return logits.crossEntropyLoss(targets, 1);
                }
        );
    }

    private static TensorRootWorkloadSpec denseNllWorkload(String name) {
        return new TensorRootWorkloadSpec(
                name,
                WorkloadKind.GENERIC,
                environment -> {
                    float[] logProbData = logProbData();
                    float[] targetData = oneHotTargetData();
                    Tensor logProbs = new Tensor(logProbData, new int[]{BATCH, CLASSES}, null, "logProbs", DataType.FLOAT32);
                    Tensor targets = new Tensor(targetData, new int[]{BATCH, CLASSES}, null, "targets", DataType.FLOAT32);
                    return logProbs.nllLoss(targets, 1);
                }
        );
    }

    private static float[] logitsData() {
        int size = BATCH * CLASSES;
        float[] logitsData = new float[size];
        for (int batch = 0; batch < BATCH; batch++) {
            int base = batch * CLASSES;
            for (int cls = 0; cls < CLASSES; cls++) {
                int offset = base + cls;
                logitsData[offset] = (float) (Math.sin(offset * 0.017) + Math.cos(offset * 0.003) * 0.5 + (cls % 7) * 0.03125);
            }
        }
        return logitsData;
    }

    private static float[] oneHotTargetData() {
        int size = BATCH * CLASSES;
        float[] targetData = new float[size];
        for (int batch = 0; batch < BATCH; batch++) {
            targetData[batch * CLASSES + (batch % CLASSES)] = 1.0f;
        }
        return targetData;
    }

    private static float[] logProbData() {
        int size = BATCH * CLASSES;
        float[] logProbData = new float[size];
        for (int batch = 0; batch < BATCH; batch++) {
            int base = batch * CLASSES;
            double rowMax = Double.NEGATIVE_INFINITY;
            for (int cls = 0; cls < CLASSES; cls++) {
                double value = Math.sin((base + cls) * 0.017) + Math.cos((base + cls) * 0.003) * 0.5 + (cls % 7) * 0.03125;
                logProbData[base + cls] = (float) value;
                rowMax = Math.max(rowMax, value);
            }
            double sumExp = 0.0d;
            for (int cls = 0; cls < CLASSES; cls++) {
                sumExp += Math.exp(logProbData[base + cls] - rowMax);
            }
            float logSumExp = (float) (rowMax + Math.log(sumExp));
            for (int cls = 0; cls < CLASSES; cls++) {
                logProbData[base + cls] -= logSumExp;
            }
        }
        return logProbData;
    }
}

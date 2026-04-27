package debug;

import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import config.runtime.RuntimeConfig;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.loss.LossReduction;
import tensor.Tensor;
import tuning.benchmark.report.TextBenchmarkReportRenderer;
import tuning.benchmark.BenchmarkEntry;
import tuning.benchmark.BenchmarkRequest;
import tuning.benchmark.BenchmarkSession;
import tuning.validate.ValidationPolicy;
import tuning.workload.TensorRootWorkloadSpec;
import tuning.workload.WorkloadKind;

import java.util.List;

final class CrossEntropyLossIndicesProfileComparisonTest {
    private static final int BATCH = 2048;
    private static final int CLASSES = 256;
    private static final tuning.measure.MeasurementPolicy MEASUREMENT = DebugMeasurementPolicies.STANDARD;

    @Test
    void benchmarkForward() {
        runBenchmark(ExecutionMode.FORWARD);
    }

    @Test
    void benchmarkForwardBackward() {
        runBenchmark(ExecutionMode.FORWARD_BACKWARD);
    }

    private static void runBenchmark(ExecutionMode mode) {
        ExecutionProfile profile = new ExecutionProfile(
                "cross-entropy-indices-profile",
                "cross-entropy-indices-profile",
                DataType.FLOAT32,
                mode,
                mode == ExecutionMode.FORWARD ? OptimizerConfig.noOptimization() : OptimizerConfig.trainingDefaults(),
                mode == ExecutionMode.FORWARD ? RuntimeConfig.inferenceDefaults() : RuntimeConfig.trainingDefaults(),
                WorkloadProfile.none()
        );

        BenchmarkEntry canonical = BenchmarkEntry.candidate("canonical-logsoftmax-nll", profile);
        BenchmarkEntry specialized = BenchmarkEntry.candidate("specialized-cross-entropy-indices", profile);

        var canonicalReport = BenchmarkSession.create(new BenchmarkRequest(
                canonicalWorkload(mode),
                List.of(canonical),
                MEASUREMENT,
                ValidationPolicy.disabled(),
                tuning.reporting.ReportPolicy.defaults()
        )).run();

        var specializedReport = BenchmarkSession.create(new BenchmarkRequest(
                specializedWorkload(mode),
                List.of(specialized),
                MEASUREMENT,
                ValidationPolicy.disabled(),
                tuning.reporting.ReportPolicy.defaults()
        )).run();

        System.out.println();
        System.out.println("=== canonical " + mode + " ===");
        System.out.println(TextBenchmarkReportRenderer.render(canonicalReport));
        System.out.println();
        System.out.println("=== specialized " + mode + " ===");
        System.out.println(TextBenchmarkReportRenderer.render(specializedReport));
    }

    private static TensorRootWorkloadSpec canonicalWorkload(ExecutionMode mode) {
        return new TensorRootWorkloadSpec(
                "cross_entropy_indices_canonical_" + mode.name().toLowerCase(),
                WorkloadKind.GENERIC,
                environment -> {
                    boolean requiresGrad = mode == ExecutionMode.FORWARD_BACKWARD;
                    Tensor logits = new Tensor(logitsData(), new int[]{BATCH, CLASSES}, null, "logits", DataType.FLOAT32);
                    logits.setRequiresGrad(requiresGrad);
                    Tensor targetIndices = new Tensor(targetIndices(), new int[]{BATCH}, null, "targetIndices", DataType.INT32);
                    return logits.logSoftmax(1).nllLossFromIndices(targetIndices, 1, LossReduction.MEAN);
                }
        );
    }

    private static TensorRootWorkloadSpec specializedWorkload(ExecutionMode mode) {
        return new TensorRootWorkloadSpec(
                "cross_entropy_indices_specialized_" + mode.name().toLowerCase(),
                WorkloadKind.GENERIC,
                environment -> {
                    boolean requiresGrad = mode == ExecutionMode.FORWARD_BACKWARD;
                    Tensor logits = new Tensor(logitsData(), new int[]{BATCH, CLASSES}, null, "logits", DataType.FLOAT32);
                    logits.setRequiresGrad(requiresGrad);
                    Tensor targetIndices = new Tensor(targetIndices(), new int[]{BATCH}, null, "targetIndices", DataType.INT32);
                    return logits.crossEntropyLossFromIndices(targetIndices, 1, LossReduction.MEAN);
                }
        );
    }

    private static float[] logitsData() {
        int size = BATCH * CLASSES;
        float[] out = new float[size];
        for (int batch = 0; batch < BATCH; batch++) {
            int base = batch * CLASSES;
            for (int cls = 0; cls < CLASSES; cls++) {
                int offset = base + cls;
                out[offset] = (float) (Math.sin(offset * 0.017) + Math.cos(offset * 0.003) * 0.5 + (cls % 7) * 0.03125);
            }
        }
        return out;
    }

    private static int[] targetIndices() {
        int[] out = new int[BATCH];
        for (int i = 0; i < BATCH; i++) {
            out[i] = i % CLASSES;
        }
        return out;
    }
}

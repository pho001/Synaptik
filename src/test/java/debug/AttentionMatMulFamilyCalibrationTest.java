package debug;

import backend.runtime.ExecutionMode;
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
import tuning.benchmark.report.TextBenchmarkSuiteReportRenderer;
import tuning.benchmark.BenchmarkEntry;
import tuning.benchmark.BenchmarkRequest;
import tuning.benchmark.BenchmarkSession;
import tuning.benchmark.BenchmarkSuiteRequest;
import tuning.benchmark.BenchmarkSuiteSession;
import tuning.calibration.PlatformCalibrationDefaults;
import tuning.calibration.PlatformCalibrationScore;
import tuning.calibration.PlatformCalibrationStep;
import tuning.calibration.runtime.RuntimeProfileCandidate;
import tuning.calibration.store.PlatformCalibrationLayout;
import tuning.calibration.store.PlatformCalibrationPaths;
import tuning.workload.TensorRootWorkloadSpec;
import tuning.workload.WorkloadKind;

import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;

final class AttentionMatMulFamilyCalibrationTest {
    private static final int BATCH = 4;
    private static final int HEADS = 8;
    private static final int TOKENS = 64;
    private static final int HEAD_DIM = 32;
    private static final int VALUE_DIM = 32;
    private static final double SCALE = 1.0d / Math.sqrt(HEAD_DIM);
    private static final tuning.measure.MeasurementPolicy MEASUREMENT = DebugMeasurementPolicies.STANDARD_WITH_TRACE;

    @Test
    void calibrateAttentionMatmulFamilyF32AndBenchmarkAttention() {
        ExecutionProfile seed = trainingSeedProfile(DataType.FLOAT32);
        GraphExecutionPolicy loweredPolicy = GraphExecutionPolicy.of(
                CompileConfig.training().withGraphOptimization(GraphOptimizationConfig.stages(true, false, false, false, false))
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

        PlatformCalibrationStep step = PlatformCalibrationDefaults.attentionMatmulStep("attention_matmul_family", tuning.preset.TuningPreset.BALANCED);
        List<RuntimeProfileCandidate> generated = step.candidateSpaceFactory().create(current).generate(step.workloads().get(0));
        List<BenchmarkEntry> calibrationEntries = generated.stream()
                .map(candidate -> BenchmarkEntry.candidate(
                        candidate.name(),
                        ExecutionProfileAssembler.assemble(
                                "attention-matmul-family-f32-calibration",
                                candidate.name(),
                                DataType.FLOAT32,
                                ExecutionMode.FORWARD_BACKWARD,
                                candidate.runtimeProfile(),
                                loweredPolicy
                        )
                ))
                .toList();

        var suiteRequest = new BenchmarkSuiteRequest(
                step.workloads(),
                calibrationEntries,
                MEASUREMENT,
                step.preset().benchmarkValidation(),
                step.preset().reportPolicy()
        );
        var suiteReport = BenchmarkSuiteSession.create(suiteRequest).run();
        RuntimeProfileCandidate winner = generated.stream()
                .min(Comparator.comparingDouble(candidate -> candidateScore(candidate.name(), step, suiteReport).score()))
                .orElseThrow();
        PlatformRuntimeProfileIO.save(layout.profilePath(), winner.runtimeProfile());

        System.out.println("ATTENTION_MATMUL_FAMILY_CALIBRATION");
        System.out.println("profilePath=" + layout.profilePath());
        System.out.println("winner=" + winner.name());
        System.out.println(TextBenchmarkSuiteReportRenderer.render(suiteReport));

        ExecutionProfile baseline = baselineProfile(DataType.FLOAT32);
        ExecutionProfile currentLowered = ExecutionProfileAssembler.assemble(
                "manual-ar-current-calib-f32",
                "manual-ar-current-calib-f32",
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                current,
                loweredPolicy
        );
        ExecutionProfile winnerLowered = ExecutionProfileAssembler.assemble(
                "manual-ar-attention-matmul-calib-f32",
                "manual-ar-attention-matmul-calib-f32",
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                winner.runtimeProfile(),
                loweredPolicy
        );

        BenchmarkRequest benchmarkRequest = new BenchmarkRequest(
                maskedAttentionWorkload(DataType.FLOAT32, ExecutionMode.FORWARD_BACKWARD),
                List.of(
                        BenchmarkEntry.baseline("manual-no-opt", baseline),
                        BenchmarkEntry.candidate("manual-ar-current-calib", currentLowered),
                        BenchmarkEntry.candidate("manual-ar-attention-matmul-calib", winnerLowered)
                ),
                MEASUREMENT,
                tuning.validate.ValidationPolicy.disabled(),
                tuning.reporting.ReportPolicy.defaults()
        );
        var benchmarkReport = BenchmarkSession.create(benchmarkRequest).run();
        System.out.println("ATTENTION_AFTER_ATTENTION_MATMUL_CALIBRATION");
        System.out.println(TextBenchmarkReportRenderer.render(benchmarkReport));
    }

    private static PlatformCalibrationScore candidateScore(
            String candidateName,
            PlatformCalibrationStep step,
            tuning.benchmark.report.BenchmarkSuiteReport suiteReport
    ) {
        PlatformCalibrationScore score = step.scorePolicy().score(candidateName, suiteReport);
        if (!score.valid()) {
            return PlatformCalibrationScore.invalid(score.explanation());
        }
        return score;
    }

    private static ExecutionProfile trainingSeedProfile(DataType dataType) {
        return new ExecutionProfile(
                "platform-seed-f32-training",
                "platform-seed-f32-training",
                dataType,
                ExecutionMode.FORWARD_BACKWARD,
                CompileConfig.training(),
                config.runtime.RuntimeConfig.trainingDefaults(),
                WorkloadProfile.none()
        );
    }

    private static ExecutionProfile baselineProfile(DataType dataType) {
        return new ExecutionProfile(
                "manual-no-opt",
                "manual-no-opt",
                dataType,
                ExecutionMode.FORWARD_BACKWARD,
                CompileConfig.noGraphOptimizationBaseline(),
                config.runtime.RuntimeConfig.noOptNoVecNoPar(),
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

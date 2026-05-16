package tuning.benchmark.report;

import graph.execution.trace.ExecutionTrace;
import graph.execution.trace.MatMulTraceMetadata;
import tensor.DataType;

import java.util.ArrayList;
import java.util.List;

/**
 * Aggregated BF16 route and policy evidence derived from existing run traces.
 */
public record Bf16PerformanceSummary(
        int matMulStepCount,
        int bgemmOutputCount,
        int sbgemmContinuationCount,
        int promotedF32Count,
        int javaRouteCount,
        int unavailableRouteCount,
        boolean openblasSbgemmAvailable,
        boolean openblasBgemmAvailable,
        long copyInBytes,
        long copyOutBytes,
        long nativeTempBytes,
        int optimizerTraceCount,
        int optimizerArrayFallbackCount,
        int optimizerNativeCount,
        int activationsOnlyPolicyCount,
        int f32MasterPolicyCount,
        int experimentalPolicyCount,
        List<String> fallbackReasons,
        boolean present
) {
    public Bf16PerformanceSummary {
        fallbackReasons = fallbackReasons == null ? List.of() : List.copyOf(fallbackReasons);
        copyInBytes = Math.max(0L, copyInBytes);
        copyOutBytes = Math.max(0L, copyOutBytes);
        nativeTempBytes = Math.max(0L, nativeTempBytes);
    }

    public static Bf16PerformanceSummary empty() {
        return new Bf16PerformanceSummary(
                0,
                0,
                0,
                0,
                0,
                0,
                false,
                false,
                0L,
                0L,
                0L,
                0,
                0,
                0,
                0,
                0,
                0,
                List.of(),
                false
        );
    }

    public static Bf16PerformanceSummary fromTrace(ExecutionTrace trace) {
        return trace == null ? empty() : fromRun(trace.run());
    }

    public static Bf16PerformanceSummary fromRun(graph.execution.trace.RunTrace run) {
        if (run == null) {
            return empty();
        }
        int matMulSteps = 0;
        int bgemm = 0;
        int sbgemm = 0;
        int promoted = 0;
        int javaRoute = 0;
        int unavailable = 0;
        boolean sbgemmAvailable = false;
        boolean bgemmAvailable = false;
        long copyIn = 0L;
        long copyOut = 0L;
        long nativeTemp = 0L;
        List<String> reasons = new ArrayList<>();

        for (var step : run.steps()) {
            if (step == null || step.metadata() == null) {
                continue;
            }
            MatMulTraceMetadata matMul = step.metadata().matMul();
            if (matMul == null || !isBf16MatMul(matMul)) {
                continue;
            }
            matMulSteps++;
            if (matMul.openblasSbgemmAvailable()) {
                sbgemmAvailable = true;
            }
            if (matMul.openblasBgemmAvailable()) {
                bgemmAvailable = true;
            }
            if ("BGEMM".equals(matMul.bf16OutputRoute())) {
                bgemm++;
            }
            if ("SBGEMM".equals(matMul.bf16ContinuationRoute())) {
                sbgemm++;
            }
            if ("PROMOTED_F32".equals(matMul.bf16OutputRoute())
                    || "F32_PROMOTED".equals(matMul.bf16ComputePrecision())) {
                promoted++;
            }
            if ("JAVA".equals(matMul.bf16OutputRoute()) || "JAVA".equals(matMul.bf16ContinuationRoute())) {
                javaRoute++;
            }
            if ("UNAVAILABLE".equals(matMul.bf16OutputRoute())
                    || "UNAVAILABLE".equals(matMul.bf16ContinuationRoute())
                    || "UNAVAILABLE".equals(matMul.bf16ComputePrecision())) {
                unavailable++;
            }
            if (matMul.copyInBytes() > 0L) {
                copyIn += matMul.copyInBytes();
            }
            if (matMul.copyOutBytes() > 0L) {
                copyOut += matMul.copyOutBytes();
            }
            if (matMul.nativeTempBytes() > 0L) {
                nativeTemp += matMul.nativeTempBytes();
            }
            addReason(reasons, matMul.nativeCpuFallbackReason());
            addReason(reasons, matMul.fallbackReason());
        }

        int optimizerCount = 0;
        int optimizerArrayFallback = 0;
        int optimizerNative = 0;
        int activationsOnly = 0;
        int f32Master = 0;
        int experimental = 0;
        for (var optimizer : run.nativeOptimizers()) {
            if (optimizer == null || optimizer.dataType() != DataType.BFLOAT16) {
                continue;
            }
            optimizerCount++;
            if ("CPU_ARRAY".equals(optimizer.route())) {
                optimizerArrayFallback++;
            }
            if ("CPU_NATIVE".equals(optimizer.route())) {
                optimizerNative++;
            }
            if ("ACTIVATIONS_ONLY".equals(optimizer.bf16TrainingPolicy())) {
                activationsOnly++;
            }
            if ("PARAMS_WITH_F32_MASTER".equals(optimizer.bf16TrainingPolicy())) {
                f32Master++;
            }
            if ("PARAMS_BF16_EXPERIMENTAL".equals(optimizer.bf16TrainingPolicy())) {
                experimental++;
            }
            addReason(reasons, optimizer.fallbackReason());
        }

        boolean present = matMulSteps > 0 || optimizerCount > 0;
        return new Bf16PerformanceSummary(
                matMulSteps,
                bgemm,
                sbgemm,
                promoted,
                javaRoute,
                unavailable,
                sbgemmAvailable,
                bgemmAvailable,
                copyIn,
                copyOut,
                nativeTemp,
                optimizerCount,
                optimizerArrayFallback,
                optimizerNative,
                activationsOnly,
                f32Master,
                experimental,
                reasons,
                present
        );
    }

    private static boolean isBf16MatMul(MatMulTraceMetadata matMul) {
        return !matMul.bf16ContinuationRoute().isBlank()
                || !matMul.bf16OutputRoute().isBlank()
                || !matMul.bf16ComputePrecision().isBlank()
                || !matMul.bf16OutputPrecision().isBlank();
    }

    private static void addReason(List<String> reasons, String reason) {
        if (reason != null && !reason.isBlank() && !reasons.contains(reason)) {
            reasons.add(reason);
        }
    }
}

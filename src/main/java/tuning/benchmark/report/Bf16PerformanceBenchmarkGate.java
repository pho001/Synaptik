package tuning.benchmark.report;

import trace.backend.MatMulTraceMetadata;
import tensor.DataType;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Fail-fast evaluator for Wave 8 BF16 performance truth evidence.
 */
public final class Bf16PerformanceBenchmarkGate {
    private Bf16PerformanceBenchmarkGate() {
    }

    public static List<String> evaluate(BenchmarkReport report) {
        if (report == null) {
            return List.of("missing BF16 performance benchmark report");
        }
        var failures = new ArrayList<String>();
        boolean sawEvidence = false;
        for (BenchmarkCandidateReport candidate : report.candidates()) {
            if (candidate == null || candidate.measurement() == null) {
                continue;
            }
            var trace = candidate.measurement().trace();
            Bf16PerformanceSummary summary = Bf16PerformanceSummary.fromTrace(trace);
            if (summary.present()) {
                sawEvidence = true;
            }
            if (trace == null || trace.run() == null) {
                continue;
            }
            for (var step : trace.run().steps()) {
                if (step == null || step.metadata() == null || step.metadata().matMul() == null) {
                    if (step != null && step.metadata() != null) {
                        evaluateNonBlasBf16(candidate.entry().name(), step.metadata().attributes(), failures);
                    }
                    continue;
                }
                evaluateNonBlasBf16(candidate.entry().name(), step.metadata().attributes(), failures);
                evaluateMatMul(candidate.entry().name(), step.metadata().matMul(), failures);
            }
            for (var optimizer : trace.run().nativeOptimizers()) {
                if (optimizer == null || !"BFLOAT16".equals(optimizer.dataType())) {
                    continue;
                }
                if (optimizer.bf16TrainingPolicy().isBlank()) {
                    failures.add("missing BF16 optimizer policy for " + candidate.entry().name());
                }
                if ("PARAMS_BF16_EXPERIMENTAL".equals(optimizer.bf16TrainingPolicy())) {
                    failures.add("experimental BF16 parameter update is not a Wave 8A default for "
                            + candidate.entry().name());
                }
            }
        }
        if (!sawEvidence) {
            failures.add("missing BF16 performance evidence");
        }
        return List.copyOf(failures);
    }

    public static void requirePass(BenchmarkReport report) {
        List<String> failures = evaluate(report);
        if (!failures.isEmpty()) {
            throw new IllegalStateException(String.join("; ", failures));
        }
    }

    private static void evaluateMatMul(String candidateName, MatMulTraceMetadata matMul, List<String> failures) {
        if (!isBf16MatMul(matMul)) {
            return;
        }
        if ("BGEMM".equals(matMul.bf16OutputRoute())) {
            if (!"cblas_bgemm".equals(matMul.blasSymbol())) {
                failures.add("BF16 output route BGEMM without cblas_bgemm for " + candidateName);
            }
            if (!matMul.openblasBgemmAvailable()) {
                failures.add("BF16 output route BGEMM while bgemm unavailable for " + candidateName);
            }
            if (!"BF16_OUTPUT".equals(matMul.bf16ComputePrecision())
                    || !"BF16".equals(matMul.bf16OutputPrecision())) {
                failures.add("BF16 output route BGEMM has wrong precision contract for " + candidateName);
            }
        }
        if ("SBGEMM".equals(matMul.bf16ContinuationRoute()) || "cblas_sbgemm".equals(matMul.blasSymbol())) {
            if (!"cblas_sbgemm".equals(matMul.blasSymbol())) {
                failures.add("BF16 continuation route SBGEMM without cblas_sbgemm for " + candidateName);
            }
            if (!matMul.openblasSbgemmAvailable()) {
                failures.add("BF16 continuation route SBGEMM while sbgemm unavailable for " + candidateName);
            }
            if (!"PROMOTED_F32".equals(matMul.bf16OutputRoute())
                    || !"F32_PROMOTED".equals(matMul.bf16ComputePrecision())
                    || !"F32".equals(matMul.bf16OutputPrecision())) {
                failures.add("SBGEMM must be reported as BF16 to F32 continuation for " + candidateName);
            }
        }
        if ("PROMOTED_F32".equals(matMul.bf16OutputRoute())
                && !"F32_PROMOTED".equals(matMul.bf16ComputePrecision())) {
            failures.add("promoted BF16 route missing F32_PROMOTED compute precision for " + candidateName);
        }
        if ("cblas_sbgemm".equals(matMul.blasSymbol()) && "BGEMM".equals(matMul.bf16OutputRoute())) {
            failures.add("sbgemm overclaimed as BF16 output route for " + candidateName);
        }
    }

    private static void evaluateNonBlasBf16(String candidateName, Map<String, Object> attrs, List<String> failures) {
        if (attrs == null || attrs.isEmpty()) {
            return;
        }
        boolean promotedStep = "BF16".equals(String.valueOf(attrs.getOrDefault("storagePrecision", "")))
                || "F32_PROMOTED".equals(String.valueOf(attrs.getOrDefault("computePrecision", "")));
        if (promotedStep
                && (!"BF16".equals(String.valueOf(attrs.getOrDefault("storagePrecision", "")))
                || !"F32_PROMOTED".equals(String.valueOf(attrs.getOrDefault("computePrecision", ""))))) {
            failures.add("BF16 non-BLAS promoted step missing BF16/F32_PROMOTED precision contract for " + candidateName);
        }
        int regionPromotedNodes = collectionSize(attrs.get("nativeCpuRegionBf16PromotedNodes"));
        if (regionPromotedNodes > 0) {
            if (!"BF16".equals(String.valueOf(attrs.getOrDefault("nativeCpuRegionBf16StoragePrecision", "")))
                    || !"F32_PROMOTED".equals(String.valueOf(attrs.getOrDefault("nativeCpuRegionBf16ComputePrecision", "")))) {
                failures.add("BF16 native region promoted nodes missing BF16/F32_PROMOTED precision contract for "
                        + candidateName);
            }
        }
    }

    private static boolean isBf16MatMul(MatMulTraceMetadata matMul) {
        return !matMul.bf16ContinuationRoute().isBlank()
                || !matMul.bf16OutputRoute().isBlank()
                || !matMul.bf16ComputePrecision().isBlank()
                || !matMul.bf16OutputPrecision().isBlank();
    }

    private static int collectionSize(Object value) {
        return value instanceof java.util.Collection<?> collection ? collection.size() : 0;
    }
}

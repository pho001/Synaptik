package tuning.benchmark.report;

import backend.ComputeBackend;
import graph.execution.trace.BackendSelectionDecisionTrace;
import graph.execution.trace.CpuMaterializationTrace;
import graph.execution.trace.ExecutionStepTrace;
import graph.execution.trace.ExecutionTrace;
import backend.accelerator.lowering.GpuLoweredRegionManifest;
import backend.accelerator.lowering.GpuLoweredRegionRejection;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;

/**
 * Backend-neutral GPU coverage diagnostics derived from prepare and run traces.
 *
 * <p>This summary intentionally separates selected region coverage, native buffer execution, tensor-array bridge
 * execution, CPU fallback, CPU materialization boundaries, storage residency, and device handoffs so reports cannot
 * hide a CPU exit behind generic accelerator presence.</p>
 *
 * @param backends coverage summary keyed by accelerator backend name
 */
public record GpuCoverageSummary(Map<String, BackendCoverage> backends) {
    public GpuCoverageSummary {
        backends = backends == null ? Map.of() : orderedMap(backends);
    }

    /**
     * Builds a coverage summary from an execution trace.
     *
     * @param trace execution trace
     * @return coverage summary, possibly empty
     */
    public static GpuCoverageSummary fromTrace(ExecutionTrace trace) {
        if (trace == null) {
            return new GpuCoverageSummary(Map.of());
        }
        List<ExecutionStepTrace> steps = trace.run().steps();
        List<CpuMaterializationTrace> materializations = trace.run().cpuMaterializations();
        LinkedHashMap<String, MutableBackendCoverage> mutable = new LinkedHashMap<>();

        collectBackendKeys(trace, mutable);

        for (var entry : mutable.entrySet()) {
            MutableBackendCoverage coverage = entry.getValue();
            coverage.totalStepCount = steps.size();
            addRunStepCoverage(entry.getKey(), steps, coverage);
            addBackendSelectionCoverage(entry.getKey(), trace.prepare().backendSelection().decisions(), coverage);
            addMaterializationCoverage(entry.getKey(), materializations, coverage);
            coverage.deviceHandoffCount = countDeviceHandoffs(entry.getKey(), steps) + coverage.cpuMaterializationCount;
        }

        LinkedHashMap<String, BackendCoverage> out = new LinkedHashMap<>();
        mutable.forEach((backend, coverage) -> out.put(backend, coverage.toImmutable()));
        return new GpuCoverageSummary(out);
    }

    /**
     * @return true when at least one accelerator backend has coverage data
     */
    public boolean present() {
        return !backends.isEmpty();
    }

    private static void collectBackendKeys(ExecutionTrace trace, LinkedHashMap<String, MutableBackendCoverage> out) {
        for (ExecutionStepTrace step : trace.run().steps()) {
            String backend = acceleratorBackend(step);
            if (isAcceleratorBackend(backend)) {
                out.computeIfAbsent(backend, ignored -> new MutableBackendCoverage());
            }
        }
        for (BackendSelectionDecisionTrace decision : trace.prepare().backendSelection().decisions()) {
            if (decision.selected() && decision.selectedBackend() != null) {
                String backend = decision.selectedBackend().name();
                if (isAcceleratorBackend(backend)) {
                    out.computeIfAbsent(backend, ignored -> new MutableBackendCoverage());
                }
            }
            for (ComputeBackend compatible : decision.compatibleBackends()) {
                if (compatible != null && isAcceleratorBackend(compatible.name())) {
                    out.computeIfAbsent(compatible.name(), ignored -> new MutableBackendCoverage());
                }
            }
        }
        for (CpuMaterializationTrace materialization : trace.run().cpuMaterializations()) {
            if (isAcceleratorBackend(materialization.materializedFrom())) {
                out.computeIfAbsent(materialization.materializedFrom(), ignored -> new MutableBackendCoverage());
            }
        }
    }

    private static void addRunStepCoverage(
            String backend,
            List<ExecutionStepTrace> steps,
            MutableBackendCoverage coverage
    ) {
        for (ExecutionStepTrace step : steps) {
            if (!backend.equals(acceleratorBackend(step))) {
                continue;
            }
            coverage.acceleratorStepCount++;
            Map<String, Object> attrs = attributes(step);
            String path = stringAttr(attrs, "acceleratorBufferExecutionPath");
            switch (path) {
                case "BUFFER_BINDING" -> coverage.bufferBindingStepCount++;
                case "TENSOR_ARRAY" -> coverage.tensorArrayStepCount++;
                case "CPU_FALLBACK" -> coverage.cpuFallbackStepCount++;
                default -> {
                    // Unknown accelerator paths still count as accelerator steps, but not as native buffer coverage.
                }
            }
            addNonBlank(coverage.reasonCodes, attrs.get("acceleratorBufferReasonCode"));
            addNonBlank(coverage.fallbackReasons, attrs.get("acceleratorBufferReason"));
            addNonBlank(coverage.fallbackReasons, attrs.get("metalFallbackReason"));
            addCount(coverage.storageResidencyCounts, stringAttr(attrs, "storageResidency"));
            coverage.copyDurationNs += firstLongAttr(attrs, "acceleratorJavaToNativeCopyNs", "metalJavaToNativeCopyNs");
            coverage.copyDurationNs += firstLongAttr(attrs, "acceleratorNativeToJavaCopyNs", "metalNativeToJavaCopyNs");
            coverage.copyDurationNs += firstLongAttr(attrs, "acceleratorNativeDeviceCopyNs", "metalNativeDeviceCopyNs");
        }
    }

    private static void addBackendSelectionCoverage(
            String backend,
            List<BackendSelectionDecisionTrace> decisions,
            MutableBackendCoverage coverage
    ) {
        int selectedTotalLength = 0;
        for (BackendSelectionDecisionTrace decision : decisions) {
            if (decision.selected() && decision.selectedBackend() != null && backend.equals(decision.selectedBackend().name())) {
                int length = decision.nodeIds().size();
                coverage.selectedRegionCount++;
                selectedTotalLength += length;
                coverage.maxSelectedRegionLength = Math.max(coverage.maxSelectedRegionLength, length);
                GpuLoweredRegionManifest manifest = decision.gpuLoweredRegionManifest();
                if (length > 1 || (manifest != null && manifest.selectedRegionLength() > 1)) {
                    coverage.multiOpGpuRegionCount++;
                }
                if (manifest != null) {
                    coverage.loweredPrimitiveCount += manifest.loweredPrimitives().size();
                }
                addDTypeResidencyCoverage(decision.gpuLoweredRegionManifest(), coverage);
                addFusedSubpatternCoverage(decision.gpuLoweredRegionManifest(), coverage);
                continue;
            }
            if (!decision.selected() && compatibleWith(backend, decision)) {
                coverage.rejectedCandidateCount++;
                String reason = decision.reason().isBlank() ? "unspecified" : decision.reason();
                addCount(coverage.rejectedCandidateReasonCounts, reason);
            }
        }
        coverage.selectedRegionTotalLength = selectedTotalLength;
    }

    private static void addMaterializationCoverage(
            String backend,
            List<CpuMaterializationTrace> materializations,
            MutableBackendCoverage coverage
    ) {
        for (CpuMaterializationTrace materialization : materializations) {
            if (!backend.equals(materialization.materializedFrom())) {
                continue;
            }
            coverage.cpuMaterializationCount++;
            addCount(coverage.cpuMaterializationReasonCounts, materialization.reason().name());
            addDTypeResidencyReason(coverage, materialization.detail());
            coverage.cpuMaterializationBytes += materialization.bytes();
            coverage.cpuMaterializationDurationNs += materialization.durationNs();
        }
    }

    private static void addDTypeResidencyCoverage(
            GpuLoweredRegionManifest manifest,
            MutableBackendCoverage coverage
    ) {
        if (manifest == null) {
            return;
        }
        manifest.backendExtensions().forEach((key, value) -> {
            if (key != null && key.startsWith("dtypeResidency.")) {
                addDTypeResidencyReason(coverage, "dtypeResidency " + value);
            }
        });
        for (GpuLoweredRegionRejection rejection : manifest.rejections()) {
            if (rejection.detail().contains("dtypeResidency")) {
                addDTypeResidencyReason(coverage, rejection.reason().name() + " " + rejection.detail());
            }
        }
    }

    private static void addDTypeResidencyReason(MutableBackendCoverage coverage, String detail) {
        if (detail == null || !detail.contains("dtypeResidency")) {
            return;
        }
        addCount(coverage.dtypeResidencyReasons, detail);
    }

    private static void addFusedSubpatternCoverage(
            GpuLoweredRegionManifest manifest,
            MutableBackendCoverage coverage
    ) {
        if (manifest == null || manifest.fusedSubpatterns().isEmpty()) {
            return;
        }
        for (var subpattern : manifest.fusedSubpatterns()) {
            coverage.gpuFusedSubpatternCount++;
            addNonBlank(coverage.gpuFusedSubpatternTypes, subpattern.patternType().name());
            addNonBlank(coverage.gpuFusedSubpatternOriginalNodeIds, subpattern.originalOperationNodeIds().toString());
            coverage.gpuFusedSubpatternLoweredPrimitiveCount += subpattern.loweredPrimitiveCount();
            addNonBlank(coverage.gpuFusedSubpatternReasons, subpattern.reason().name());
        }
    }

    private static int countDeviceHandoffs(String backend, List<ExecutionStepTrace> steps) {
        int count = 0;
        for (int i = 1; i < steps.size(); i++) {
            String previous = runBackend(steps.get(i - 1));
            String current = runBackend(steps.get(i));
            if (!previous.equals(current) && (backend.equals(previous) || backend.equals(current))) {
                count++;
            }
        }
        return count;
    }

    private static boolean compatibleWith(String backend, BackendSelectionDecisionTrace decision) {
        if (decision.selectedBackend() != null && backend.equals(decision.selectedBackend().name())) {
            return true;
        }
        return decision.compatibleBackends().stream()
                .filter(candidate -> candidate != null)
                .map(Enum::name)
                .anyMatch(backend::equals);
    }

    private static Map<String, Object> attributes(ExecutionStepTrace step) {
        if (step == null || step.metadata() == null || step.metadata().attributes() == null) {
            return Map.of();
        }
        return step.metadata().attributes();
    }

    private static String acceleratorBackend(ExecutionStepTrace step) {
        Map<String, Object> attrs = attributes(step);
        String backend = stringAttr(attrs, "acceleratorBufferBackend");
        return backend.isBlank() ? "" : backend;
    }

    private static String runBackend(ExecutionStepTrace step) {
        String backend = acceleratorBackend(step);
        if (!backend.isBlank()) {
            return backend;
        }
        return step == null || step.backend() == null ? "" : step.backend();
    }

    private static boolean isAcceleratorBackend(String backend) {
        return backend != null && (backend.equals("GPU_METAL") || backend.equals("GPU_CUDA"));
    }

    private static String stringAttr(Map<String, Object> attrs, String key) {
        if (attrs == null || !attrs.containsKey(key) || attrs.get(key) == null) {
            return "";
        }
        return String.valueOf(attrs.get(key)).trim();
    }

    private static long longAttr(Object value) {
        return value instanceof Number number ? number.longValue() : 0L;
    }

    private static long firstLongAttr(Map<String, Object> attrs, String preferredKey, String fallbackKey) {
        if (attrs == null) {
            return 0L;
        }
        Object preferred = attrs.get(preferredKey);
        return preferred instanceof Number ? ((Number) preferred).longValue() : longAttr(attrs.get(fallbackKey));
    }

    private static void addCount(LinkedHashMap<String, Integer> target, String key) {
        if (key == null || key.isBlank()) {
            return;
        }
        target.merge(key, 1, Integer::sum);
    }

    private static void addNonBlank(LinkedHashSet<String> target, Object value) {
        if (value == null) {
            return;
        }
        String text = String.valueOf(value).trim();
        if (!text.isBlank()) {
            target.add(text);
        }
    }

    /**
     * Per-backend coverage aggregate.
     */
    public record BackendCoverage(
            int totalStepCount,
            int acceleratorStepCount,
            double gpuCoverageRatio,
            int selectedRegionCount,
            int multiOpGpuRegionCount,
            int maxSelectedRegionLength,
            double averageSelectedRegionLength,
            int loweredPrimitiveCount,
            int rejectedCandidateCount,
            Map<String, Integer> rejectedCandidateReasonCounts,
            int bufferBindingStepCount,
            int tensorArrayStepCount,
            int cpuFallbackStepCount,
            int fallbackCount,
            int cpuMaterializationCount,
            Map<String, Integer> cpuMaterializationReasonCounts,
            long cpuMaterializationBytes,
            long cpuMaterializationDurationNs,
            long copyDurationNs,
            int deviceHandoffCount,
            Map<String, Integer> storageResidencyCounts,
            Map<String, Integer> dtypeResidencyReasons,
            int gpuFusedSubpatternCount,
            List<String> gpuFusedSubpatternTypes,
            List<String> gpuFusedSubpatternOriginalNodeIds,
            int gpuFusedSubpatternLoweredPrimitiveCount,
            List<String> gpuFusedSubpatternReasons,
            List<String> reasonCodes,
            List<String> fallbackReasons
    ) {
        public BackendCoverage {
            rejectedCandidateReasonCounts = rejectedCandidateReasonCounts == null
                    ? Map.of()
                    : orderedMap(rejectedCandidateReasonCounts);
            cpuMaterializationReasonCounts = cpuMaterializationReasonCounts == null
                    ? Map.of()
                    : orderedMap(cpuMaterializationReasonCounts);
            storageResidencyCounts = storageResidencyCounts == null ? Map.of() : orderedMap(storageResidencyCounts);
            dtypeResidencyReasons = dtypeResidencyReasons == null ? Map.of() : orderedMap(dtypeResidencyReasons);
            gpuFusedSubpatternTypes = gpuFusedSubpatternTypes == null ? List.of() : List.copyOf(gpuFusedSubpatternTypes);
            gpuFusedSubpatternOriginalNodeIds = gpuFusedSubpatternOriginalNodeIds == null ? List.of() : List.copyOf(gpuFusedSubpatternOriginalNodeIds);
            gpuFusedSubpatternReasons = gpuFusedSubpatternReasons == null ? List.of() : List.copyOf(gpuFusedSubpatternReasons);
            reasonCodes = reasonCodes == null ? List.of() : List.copyOf(reasonCodes);
            fallbackReasons = fallbackReasons == null ? List.of() : List.copyOf(fallbackReasons);
        }

        public int nativeBufferStepCount() {
            return bufferBindingStepCount;
        }
    }

    private static <K, V> Map<K, V> orderedMap(Map<K, V> source) {
        return Collections.unmodifiableMap(new LinkedHashMap<>(source));
    }

    private static final class MutableBackendCoverage {
        private int totalStepCount;
        private int acceleratorStepCount;
        private int selectedRegionCount;
        private int multiOpGpuRegionCount;
        private int maxSelectedRegionLength;
        private int selectedRegionTotalLength;
        private int loweredPrimitiveCount;
        private int rejectedCandidateCount;
        private final LinkedHashMap<String, Integer> rejectedCandidateReasonCounts = new LinkedHashMap<>();
        private int bufferBindingStepCount;
        private int tensorArrayStepCount;
        private int cpuFallbackStepCount;
        private int cpuMaterializationCount;
        private final LinkedHashMap<String, Integer> cpuMaterializationReasonCounts = new LinkedHashMap<>();
        private long cpuMaterializationBytes;
        private long cpuMaterializationDurationNs;
        private long copyDurationNs;
        private int deviceHandoffCount;
        private final LinkedHashMap<String, Integer> storageResidencyCounts = new LinkedHashMap<>();
        private final LinkedHashMap<String, Integer> dtypeResidencyReasons = new LinkedHashMap<>();
        private int gpuFusedSubpatternCount;
        private final LinkedHashSet<String> gpuFusedSubpatternTypes = new LinkedHashSet<>();
        private final LinkedHashSet<String> gpuFusedSubpatternOriginalNodeIds = new LinkedHashSet<>();
        private int gpuFusedSubpatternLoweredPrimitiveCount;
        private final LinkedHashSet<String> gpuFusedSubpatternReasons = new LinkedHashSet<>();
        private final LinkedHashSet<String> reasonCodes = new LinkedHashSet<>();
        private final LinkedHashSet<String> fallbackReasons = new LinkedHashSet<>();

        private BackendCoverage toImmutable() {
            int fallbackCount = tensorArrayStepCount + cpuFallbackStepCount;
            double ratio = totalStepCount == 0 ? 0.0d : (double) acceleratorStepCount / (double) totalStepCount;
            double averageLength = selectedRegionCount == 0
                    ? 0.0d
                    : (double) selectedRegionTotalLength / (double) selectedRegionCount;
            return new BackendCoverage(
                    totalStepCount,
                    acceleratorStepCount,
                    ratio,
                    selectedRegionCount,
                    multiOpGpuRegionCount,
                    maxSelectedRegionLength,
                    averageLength,
                    loweredPrimitiveCount,
                    rejectedCandidateCount,
                    new LinkedHashMap<>(rejectedCandidateReasonCounts),
                    bufferBindingStepCount,
                    tensorArrayStepCount,
                    cpuFallbackStepCount,
                    fallbackCount,
                    cpuMaterializationCount,
                    new LinkedHashMap<>(cpuMaterializationReasonCounts),
                    cpuMaterializationBytes,
                    cpuMaterializationDurationNs,
                    copyDurationNs,
                    deviceHandoffCount,
                    new LinkedHashMap<>(storageResidencyCounts),
                    new LinkedHashMap<>(dtypeResidencyReasons),
                    gpuFusedSubpatternCount,
                    new ArrayList<>(gpuFusedSubpatternTypes),
                    new ArrayList<>(gpuFusedSubpatternOriginalNodeIds),
                    gpuFusedSubpatternLoweredPrimitiveCount,
                    new ArrayList<>(gpuFusedSubpatternReasons),
                    new ArrayList<>(reasonCodes),
                    new ArrayList<>(fallbackReasons)
            );
        }
    }
}

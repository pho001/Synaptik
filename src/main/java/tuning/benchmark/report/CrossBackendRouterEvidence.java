package tuning.benchmark.report;

import graph.execution.trace.ExecutionStepTrace;
import graph.execution.trace.ExecutionTrace;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;

/**
 * Backend-neutral route evidence for accelerator router calibration and regression gates.
 *
 * <p>This model is intentionally report-side only. It derives selected routes, fallback paths, copy/write status,
 * layout/dtype residency, and CPU-exit evidence from existing traces without changing runtime execution.</p>
 *
 * @param backends evidence keyed by accelerator backend name
 */
public record CrossBackendRouterEvidence(Map<String, BackendEvidence> backends) {
    public CrossBackendRouterEvidence {
        backends = backends == null ? Map.of() : orderedMap(backends);
    }

    /**
     * Builds cross-backend router evidence from one execution trace.
     *
     * @param trace execution trace
     * @return router evidence, possibly empty
     */
    public static CrossBackendRouterEvidence fromTrace(ExecutionTrace trace) {
        if (trace == null) {
            return new CrossBackendRouterEvidence(Map.of());
        }
        GpuCoverageSummary coverageSummary = GpuCoverageSummary.fromTrace(trace);
        AcceleratorTraceSummary acceleratorSummary = AcceleratorTraceSummary.fromSteps(trace.run().steps());
        LinkedHashMap<String, MutableBackendEvidence> mutable = new LinkedHashMap<>();

        coverageSummary.backends().keySet().forEach(backend ->
                mutable.computeIfAbsent(backend, MutableBackendEvidence::new));
        acceleratorSummary.backends().keySet().forEach(backend ->
                mutable.computeIfAbsent(backend, MutableBackendEvidence::new));
        collectBackendKeysFromSteps(trace.run().steps(), mutable);

        for (var entry : mutable.entrySet()) {
            String backend = entry.getKey();
            MutableBackendEvidence evidence = entry.getValue();
            GpuCoverageSummary.BackendCoverage coverage = coverageSummary.backends().get(backend);
            if (coverage != null) {
                evidence.addCoverage(coverage);
            }
            AcceleratorTraceSummary.BackendSummary accelerator = acceleratorSummary.backends().get(backend);
            if (accelerator != null) {
                evidence.addAcceleratorSummary(accelerator);
            }
        }
        collectStepAttributes(trace.run().steps(), mutable);

        LinkedHashMap<String, BackendEvidence> out = new LinkedHashMap<>();
        mutable.forEach((backend, evidence) -> out.put(backend, evidence.toImmutable()));
        return new CrossBackendRouterEvidence(out);
    }

    /**
     * @return true when at least one accelerator backend has router evidence
     */
    public boolean present() {
        return !backends.isEmpty();
    }

    private static void collectBackendKeysFromSteps(
            List<ExecutionStepTrace> steps,
            LinkedHashMap<String, MutableBackendEvidence> out
    ) {
        for (ExecutionStepTrace step : steps) {
            String backend = backendFromStep(step);
            if (isAcceleratorBackend(backend)) {
                out.computeIfAbsent(backend, MutableBackendEvidence::new);
            }
        }
    }

    private static void collectStepAttributes(
            List<ExecutionStepTrace> steps,
            LinkedHashMap<String, MutableBackendEvidence> out
    ) {
        for (ExecutionStepTrace step : steps) {
            String backend = backendFromStep(step);
            if (!isAcceleratorBackend(backend)) {
                continue;
            }
            MutableBackendEvidence evidence = out.computeIfAbsent(backend, MutableBackendEvidence::new);
            Map<String, Object> attrs = attributes(step);
            addCount(evidence.acceleratorPathCounts, attrs.get("acceleratorBufferExecutionPath"));
            addCount(evidence.backendRouteCounts, attrs.get("metalExecutionRoute"));
            addCount(evidence.backendRouteCounts, attrs.get("cudaExecutionPath"));
            addListCounts(evidence.rejectedRouteReasonCounts, attrs.get("metalRouteRejectedReasonCodes"));
            addListCounts(evidence.rejectedRouteReasonCounts, attrs.get("cudaRouteRejectedReasonCodes"));
            addCount(evidence.reasonCodeCounts, attrs.get("acceleratorBufferReasonCode"));
            addCount(evidence.reasonCodeCounts, attrs.get("cudaReasonCode"));
            addCount(evidence.fallbackReasonCounts, attrs.get("acceleratorBufferReason"));
            addCount(evidence.fallbackReasonCounts, attrs.get("metalFallbackReason"));
            addCount(evidence.fallbackReasonCounts, attrs.get("cudaFallbackReason"));
            addCount(evidence.nativeCopyStrategyCounts, attrs.get("acceleratorNativeCopyStrategy"));
            addCount(evidence.nativeCopyStrategyCounts, attrs.get("metalNativeCopyStrategy"));
            addCount(evidence.nativeCopyStrategyCounts, attrs.get("cudaNativeCopyStrategy"));
            addCount(evidence.outputBufferWriteStatusCounts, attrs.get("acceleratorOutputBufferWriteStatus"));
            addCount(evidence.outputBufferWriteStatusCounts, attrs.get("metalOutputBufferWriteStatus"));
            addCount(evidence.outputBufferWriteStatusCounts, attrs.get("cudaOutputBufferWriteStatus"));
        }
    }

    private static String backendFromStep(ExecutionStepTrace step) {
        Map<String, Object> attrs = attributes(step);
        String acceleratorBackend = stringAttr(attrs, "acceleratorBufferBackend");
        if (!acceleratorBackend.isBlank()) {
            return acceleratorBackend;
        }
        return step == null ? "" : step.backend();
    }

    private static Map<String, Object> attributes(ExecutionStepTrace step) {
        if (step == null || step.metadata() == null || step.metadata().attributes() == null) {
            return Map.of();
        }
        return step.metadata().attributes();
    }

    private static String stringAttr(Map<String, Object> attrs, String key) {
        if (attrs == null || !attrs.containsKey(key) || attrs.get(key) == null) {
            return "";
        }
        return String.valueOf(attrs.get(key)).trim();
    }

    private static boolean isAcceleratorBackend(String backend) {
        return "GPU_METAL".equals(backend) || "GPU_CUDA".equals(backend);
    }

    private static void addCount(LinkedHashMap<String, Integer> target, Object value) {
        if (value == null) {
            return;
        }
        String text = String.valueOf(value).trim();
        if (!text.isBlank()) {
            target.merge(text, 1, Integer::sum);
        }
    }

    private static void addListCounts(LinkedHashMap<String, Integer> target, Object value) {
        if (value instanceof Iterable<?> iterable) {
            for (Object item : iterable) {
                addCount(target, item);
            }
            return;
        }
        addCount(target, value);
    }

    private static <K, V> Map<K, V> orderedMap(Map<K, V> source) {
        return Collections.unmodifiableMap(new LinkedHashMap<>(source));
    }

    private static List<String> orderedKeys(Map<String, Integer> source) {
        return List.copyOf(source.keySet());
    }

    /**
     * Per-backend router evidence aggregate.
     */
    public record BackendEvidence(
            String backend,
            int totalStepCount,
            int acceleratorStepCount,
            int selectedRegionCount,
            int multiOpGpuRegionCount,
            int maxSelectedRegionLength,
            int loweredPrimitiveCount,
            int gpuFusedSubpatternCount,
            int bufferBindingStepCount,
            int tensorArrayStepCount,
            int cpuFallbackStepCount,
            int unavailableStepCount,
            int preparedInputStepCount,
            int fallbackCount,
            int cpuMaterializationCount,
            int internalCpuMaterializationCount,
            int gradientPublicationMaterializationCount,
            int deviceHandoffCount,
            int gpuLayoutMaterializationCount,
            long gpuLayoutMaterializationBytes,
            long copyDurationNs,
            long inputBytes,
            long outputBytes,
            Map<String, Integer> acceleratorPathCounts,
            Map<String, Integer> backendRouteCounts,
            Map<String, Integer> rejectedRouteReasonCounts,
            Map<String, Integer> rejectedCandidateReasonCounts,
            Map<String, Integer> reasonCodeCounts,
            Map<String, Integer> fallbackReasonCounts,
            Map<String, Integer> nativeCopyStrategyCounts,
            Map<String, Integer> outputBufferWriteStatusCounts,
            Map<String, Integer> storageResidencyCounts,
            Map<String, Integer> dtypeResidencyReasonCounts,
            Map<String, Integer> gpuLayoutTransformKindCounts,
            Map<String, Integer> gpuLayoutTargetLayoutClassCounts
    ) {
        public BackendEvidence {
            backend = backend == null ? "" : backend;
            acceleratorPathCounts = acceleratorPathCounts == null ? Map.of() : orderedMap(acceleratorPathCounts);
            backendRouteCounts = backendRouteCounts == null ? Map.of() : orderedMap(backendRouteCounts);
            rejectedRouteReasonCounts = rejectedRouteReasonCounts == null ? Map.of() : orderedMap(rejectedRouteReasonCounts);
            rejectedCandidateReasonCounts = rejectedCandidateReasonCounts == null ? Map.of() : orderedMap(rejectedCandidateReasonCounts);
            reasonCodeCounts = reasonCodeCounts == null ? Map.of() : orderedMap(reasonCodeCounts);
            fallbackReasonCounts = fallbackReasonCounts == null ? Map.of() : orderedMap(fallbackReasonCounts);
            nativeCopyStrategyCounts = nativeCopyStrategyCounts == null ? Map.of() : orderedMap(nativeCopyStrategyCounts);
            outputBufferWriteStatusCounts = outputBufferWriteStatusCounts == null ? Map.of() : orderedMap(outputBufferWriteStatusCounts);
            storageResidencyCounts = storageResidencyCounts == null ? Map.of() : orderedMap(storageResidencyCounts);
            dtypeResidencyReasonCounts = dtypeResidencyReasonCounts == null ? Map.of() : orderedMap(dtypeResidencyReasonCounts);
            gpuLayoutTransformKindCounts = gpuLayoutTransformKindCounts == null ? Map.of() : orderedMap(gpuLayoutTransformKindCounts);
            gpuLayoutTargetLayoutClassCounts = gpuLayoutTargetLayoutClassCounts == null ? Map.of() : orderedMap(gpuLayoutTargetLayoutClassCounts);
        }

        public boolean hasRouteOrPath(String routeOrPath) {
            if (routeOrPath == null || routeOrPath.isBlank()) {
                return true;
            }
            String needle = routeOrPath.trim();
            return backendRouteCounts.containsKey(needle) || acceleratorPathCounts.containsKey(needle);
        }

        public boolean hasTextEvidence(String expectedText) {
            if (expectedText == null || expectedText.isBlank()) {
                return true;
            }
            String needle = expectedText.trim();
            return allVisibleEvidence().stream().anyMatch(item -> item.contains(needle));
        }

        public List<String> nativeCopyStrategies() {
            return orderedKeys(nativeCopyStrategyCounts);
        }

        public List<String> outputBufferWriteStatuses() {
            return orderedKeys(outputBufferWriteStatusCounts);
        }

        public boolean hasSupportClaim() {
            return bufferBindingStepCount > 0
                    || backendRouteCounts.keySet().stream().anyMatch(BackendEvidence::isNativeSupportRoute)
                    || reasonCodeCounts.containsKey("BUFFER_BINDING_AVAILABLE");
        }

        public List<String> allVisibleEvidence() {
            LinkedHashSet<String> evidence = new LinkedHashSet<>();
            evidence.addAll(acceleratorPathCounts.keySet());
            evidence.addAll(backendRouteCounts.keySet());
            evidence.addAll(rejectedRouteReasonCounts.keySet());
            evidence.addAll(rejectedCandidateReasonCounts.keySet());
            evidence.addAll(reasonCodeCounts.keySet());
            evidence.addAll(fallbackReasonCounts.keySet());
            evidence.addAll(nativeCopyStrategyCounts.keySet());
            evidence.addAll(outputBufferWriteStatusCounts.keySet());
            evidence.addAll(dtypeResidencyReasonCounts.keySet());
            evidence.addAll(gpuLayoutTransformKindCounts.keySet());
            evidence.addAll(gpuLayoutTargetLayoutClassCounts.keySet());
            return List.copyOf(evidence);
        }

        private static boolean isNativeSupportRoute(String route) {
            if (route == null || route.isBlank()) {
                return false;
            }
            return switch (route) {
                case "CPU_FALLBACK", "TENSOR_ARRAY", "UNAVAILABLE", "STATIC_CPU_FALLBACK", "UNAVAILABLE_REQUIRED" -> false;
                default -> true;
            };
        }
    }

    private static final class MutableBackendEvidence {
        private final String backend;
        private int totalStepCount;
        private int acceleratorStepCount;
        private int selectedRegionCount;
        private int multiOpGpuRegionCount;
        private int maxSelectedRegionLength;
        private int loweredPrimitiveCount;
        private int gpuFusedSubpatternCount;
        private int bufferBindingStepCount;
        private int tensorArrayStepCount;
        private int cpuFallbackStepCount;
        private int unavailableStepCount;
        private int preparedInputStepCount;
        private int fallbackCount;
        private int cpuMaterializationCount;
        private int internalCpuMaterializationCount;
        private int gradientPublicationMaterializationCount;
        private int deviceHandoffCount;
        private int gpuLayoutMaterializationCount;
        private long gpuLayoutMaterializationBytes;
        private long copyDurationNs;
        private long inputBytes;
        private long outputBytes;
        private final LinkedHashMap<String, Integer> acceleratorPathCounts = new LinkedHashMap<>();
        private final LinkedHashMap<String, Integer> backendRouteCounts = new LinkedHashMap<>();
        private final LinkedHashMap<String, Integer> rejectedRouteReasonCounts = new LinkedHashMap<>();
        private final LinkedHashMap<String, Integer> rejectedCandidateReasonCounts = new LinkedHashMap<>();
        private final LinkedHashMap<String, Integer> reasonCodeCounts = new LinkedHashMap<>();
        private final LinkedHashMap<String, Integer> fallbackReasonCounts = new LinkedHashMap<>();
        private final LinkedHashMap<String, Integer> nativeCopyStrategyCounts = new LinkedHashMap<>();
        private final LinkedHashMap<String, Integer> outputBufferWriteStatusCounts = new LinkedHashMap<>();
        private final LinkedHashMap<String, Integer> storageResidencyCounts = new LinkedHashMap<>();
        private final LinkedHashMap<String, Integer> dtypeResidencyReasonCounts = new LinkedHashMap<>();
        private final LinkedHashMap<String, Integer> gpuLayoutTransformKindCounts = new LinkedHashMap<>();
        private final LinkedHashMap<String, Integer> gpuLayoutTargetLayoutClassCounts = new LinkedHashMap<>();

        private MutableBackendEvidence(String backend) {
            this.backend = backend == null ? "" : backend;
        }

        private void addCoverage(GpuCoverageSummary.BackendCoverage coverage) {
            totalStepCount = Math.max(totalStepCount, coverage.totalStepCount());
            acceleratorStepCount = Math.max(acceleratorStepCount, coverage.acceleratorStepCount());
            selectedRegionCount = Math.max(selectedRegionCount, coverage.selectedRegionCount());
            multiOpGpuRegionCount = Math.max(multiOpGpuRegionCount, coverage.multiOpGpuRegionCount());
            maxSelectedRegionLength = Math.max(maxSelectedRegionLength, coverage.maxSelectedRegionLength());
            loweredPrimitiveCount = Math.max(loweredPrimitiveCount, coverage.loweredPrimitiveCount());
            gpuFusedSubpatternCount = Math.max(gpuFusedSubpatternCount, coverage.gpuFusedSubpatternCount());
            bufferBindingStepCount = Math.max(bufferBindingStepCount, coverage.bufferBindingStepCount());
            tensorArrayStepCount = Math.max(tensorArrayStepCount, coverage.tensorArrayStepCount());
            cpuFallbackStepCount = Math.max(cpuFallbackStepCount, coverage.cpuFallbackStepCount());
            fallbackCount = Math.max(fallbackCount, coverage.fallbackCount());
            cpuMaterializationCount = Math.max(cpuMaterializationCount, coverage.cpuMaterializationCount());
            internalCpuMaterializationCount = Math.max(internalCpuMaterializationCount, coverage.internalCpuMaterializationCount());
            gradientPublicationMaterializationCount = Math.max(
                    gradientPublicationMaterializationCount,
                    coverage.gradientPublicationMaterializationCount()
            );
            deviceHandoffCount = Math.max(deviceHandoffCount, coverage.deviceHandoffCount());
            gpuLayoutMaterializationCount = Math.max(gpuLayoutMaterializationCount, coverage.gpuLayoutMaterializationCount());
            gpuLayoutMaterializationBytes = Math.max(gpuLayoutMaterializationBytes, coverage.gpuLayoutMaterializationBytes());
            copyDurationNs = Math.max(copyDurationNs, coverage.copyDurationNs());
            addAll(rejectedCandidateReasonCounts, coverage.rejectedCandidateReasonCounts());
            addAll(storageResidencyCounts, coverage.storageResidencyCounts());
            addAll(dtypeResidencyReasonCounts, coverage.dtypeResidencyReasons());
            addAll(gpuLayoutTransformKindCounts, coverage.gpuLayoutTransformKindCounts());
            addAll(gpuLayoutTargetLayoutClassCounts, coverage.gpuLayoutTargetLayoutClassCounts());
        }

        private void addAcceleratorSummary(AcceleratorTraceSummary.BackendSummary summary) {
            acceleratorStepCount = Math.max(acceleratorStepCount, summary.steps());
            bufferBindingStepCount = Math.max(bufferBindingStepCount, summary.bufferBindingSteps());
            tensorArrayStepCount = Math.max(tensorArrayStepCount, summary.tensorArraySteps());
            cpuFallbackStepCount = Math.max(cpuFallbackStepCount, summary.cpuFallbackSteps());
            unavailableStepCount = Math.max(unavailableStepCount, summary.unavailableSteps());
            preparedInputStepCount = Math.max(preparedInputStepCount, summary.preparedInputSteps());
            inputBytes = Math.max(inputBytes, summary.inputBytes());
            outputBytes = Math.max(outputBytes, summary.outputBytes());
            copyDurationNs = Math.max(
                    copyDurationNs,
                    summary.javaToNativeCopyNs() + summary.nativeToJavaCopyNs() + summary.nativeDeviceCopyNs()
            );
        }

        private BackendEvidence toImmutable() {
            return new BackendEvidence(
                    backend,
                    totalStepCount,
                    acceleratorStepCount,
                    selectedRegionCount,
                    multiOpGpuRegionCount,
                    maxSelectedRegionLength,
                    loweredPrimitiveCount,
                    gpuFusedSubpatternCount,
                    bufferBindingStepCount,
                    tensorArrayStepCount,
                    cpuFallbackStepCount,
                    unavailableStepCount,
                    preparedInputStepCount,
                    fallbackCount,
                    cpuMaterializationCount,
                    internalCpuMaterializationCount,
                    gradientPublicationMaterializationCount,
                    deviceHandoffCount,
                    gpuLayoutMaterializationCount,
                    gpuLayoutMaterializationBytes,
                    copyDurationNs,
                    inputBytes,
                    outputBytes,
                    new LinkedHashMap<>(acceleratorPathCounts),
                    new LinkedHashMap<>(backendRouteCounts),
                    new LinkedHashMap<>(rejectedRouteReasonCounts),
                    new LinkedHashMap<>(rejectedCandidateReasonCounts),
                    new LinkedHashMap<>(reasonCodeCounts),
                    new LinkedHashMap<>(fallbackReasonCounts),
                    new LinkedHashMap<>(nativeCopyStrategyCounts),
                    new LinkedHashMap<>(outputBufferWriteStatusCounts),
                    new LinkedHashMap<>(storageResidencyCounts),
                    new LinkedHashMap<>(dtypeResidencyReasonCounts),
                    new LinkedHashMap<>(gpuLayoutTransformKindCounts),
                    new LinkedHashMap<>(gpuLayoutTargetLayoutClassCounts)
            );
        }

        private static void addAll(LinkedHashMap<String, Integer> target, Map<String, Integer> source) {
            source.forEach((key, value) -> {
                if (key != null && !key.isBlank() && value != null && value > 0) {
                    target.merge(key, value, Integer::sum);
                }
            });
        }
    }
}

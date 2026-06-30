package tuning.benchmark.report;

import trace.execution.ExecutionStepTrace;

import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;

/**
 * Aggregated accelerator buffer-path diagnostics derived from execution trace attributes.
 *
 * <p>The summary intentionally reads the backend-neutral {@code acceleratorBuffer*} attributes emitted by
 * prepared execution. Metal-specific copy counters are aggregated opportunistically when present, but the
 * path counts are not tied to Metal and can be reused by CUDA once it emits the same common attributes.</p>
 *
 * @param backends summary by accelerator backend name
 */
public record AcceleratorTraceSummary(Map<String, BackendSummary> backends) {
    public AcceleratorTraceSummary {
        backends = backends == null ? Map.of() : Map.copyOf(backends);
    }

    /**
     * Builds a summary from run steps.
     *
     * @param steps traced execution steps
     * @return accelerator summary, possibly empty when no accelerator step was traced
     */
    public static AcceleratorTraceSummary fromSteps(List<ExecutionStepTrace> steps) {
        if (steps == null || steps.isEmpty()) {
            return new AcceleratorTraceSummary(Map.of());
        }
        LinkedHashMap<String, MutableBackendSummary> mutable = new LinkedHashMap<>();
        for (ExecutionStepTrace step : steps) {
            if (step == null || step.metadata() == null || step.metadata().attributes() == null) {
                continue;
            }
            Map<String, Object> attrs = step.metadata().attributes();
            Object backendValue = attrs.get("acceleratorBufferBackend");
            if (backendValue == null || String.valueOf(backendValue).isBlank()) {
                continue;
            }
            String backend = String.valueOf(backendValue);
            MutableBackendSummary summary = mutable.computeIfAbsent(backend, ignored -> new MutableBackendSummary());
            summary.steps++;
            String path = String.valueOf(attrs.getOrDefault("acceleratorBufferExecutionPath", ""));
            switch (path) {
                case "BUFFER_BINDING" -> summary.bufferBindingSteps++;
                case "TENSOR_ARRAY" -> summary.tensorArraySteps++;
                case "CPU_FALLBACK" -> summary.cpuFallbackSteps++;
                default -> summary.unavailableSteps++;
            }
            if (Boolean.TRUE.equals(attrs.get("acceleratorBufferPreparedInputUsed"))) {
                summary.preparedInputSteps++;
            }
            addNonBlank(summary.reasonCodes, attrs.get("acceleratorBufferReasonCode"));
            addNonBlankItems(summary.reasonCodes, attrs.get("fallbackReasonCodes"));
            if (!attrs.containsKey("fallbackReasonCodes")) {
                addNonBlank(summary.reasonCodes, attrs.get("fallbackReasonCode"));
            }
            addNonBlank(summary.fallbackReasons, attrs.get("acceleratorBufferReason"));
            addNonBlank(summary.fallbackReasons, attrs.get("metalFallbackReason"));
            addNonBlankItems(summary.fallbackReasons, attrs.get("fallbackReasons"));
            if (!attrs.containsKey("fallbackReasons")) {
                addNonBlank(summary.fallbackReasons, attrs.get("fallbackReason"));
            }
            summary.javaToNativeCopyNs += firstLongAttr(attrs, "acceleratorJavaToNativeCopyNs", "metalJavaToNativeCopyNs");
            summary.nativeToJavaCopyNs += firstLongAttr(attrs, "acceleratorNativeToJavaCopyNs", "metalNativeToJavaCopyNs");
            summary.nativeDeviceCopyNs += firstLongAttr(attrs, "acceleratorNativeDeviceCopyNs", "metalNativeDeviceCopyNs");
            addNonBlank(summary.nativeCopyStrategies, attrs.get("acceleratorNativeCopyStrategy"));
            addNonBlank(summary.nativeCopyStrategies, attrs.get("metalNativeCopyStrategy"));
            addNonBlank(summary.outputBufferWriteStatuses, attrs.get("metalOutputBufferWriteStatus"));
            addCount(summary.executionRouteCounts, attrs.get("metalExecutionRoute"));
            addListCounts(summary.rejectedRouteReasonCounts, attrs.get("metalRouteRejectedReasonCodes"));
            summary.inputBytes += firstLongAttr(attrs, "acceleratorInputBytes", "metalInputBytes");
            summary.outputBytes += firstLongAttr(attrs, "acceleratorOutputBytes", "metalOutputBytes");
        }

        LinkedHashMap<String, BackendSummary> out = new LinkedHashMap<>();
        mutable.forEach((backend, summary) -> out.put(backend, summary.toImmutable()));
        return new AcceleratorTraceSummary(out);
    }

    /**
     * @return true when at least one accelerator step was summarized
     */
    public boolean present() {
        return !backends.isEmpty();
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

    private static void addNonBlank(LinkedHashSet<String> target, Object value) {
        if (value == null) {
            return;
        }
        String text = String.valueOf(value).trim();
        if (!text.isBlank()) {
            target.add(text);
        }
    }

    private static void addNonBlankItems(LinkedHashSet<String> target, Object value) {
        if (value instanceof Iterable<?> iterable) {
            for (Object item : iterable) {
                addNonBlank(target, item);
            }
            return;
        }
        addNonBlank(target, value);
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

    /**
     * Per-backend aggregate counters.
     */
    public record BackendSummary(
            int steps,
            int bufferBindingSteps,
            int tensorArraySteps,
            int cpuFallbackSteps,
            int unavailableSteps,
            int preparedInputSteps,
            long inputBytes,
            long outputBytes,
            long javaToNativeCopyNs,
            long nativeToJavaCopyNs,
            long nativeDeviceCopyNs,
            List<String> nativeCopyStrategies,
            List<String> outputBufferWriteStatuses,
            Map<String, Integer> executionRouteCounts,
            Map<String, Integer> rejectedRouteReasonCounts,
            List<String> fallbackReasons,
            List<String> reasonCodes
    ) {
        public BackendSummary {
            nativeCopyStrategies = nativeCopyStrategies == null ? List.of() : List.copyOf(nativeCopyStrategies);
            outputBufferWriteStatuses = outputBufferWriteStatuses == null ? List.of() : List.copyOf(outputBufferWriteStatuses);
            executionRouteCounts = executionRouteCounts == null ? Map.of() : Map.copyOf(executionRouteCounts);
            rejectedRouteReasonCounts = rejectedRouteReasonCounts == null ? Map.of() : Map.copyOf(rejectedRouteReasonCounts);
            fallbackReasons = fallbackReasons == null ? List.of() : List.copyOf(fallbackReasons);
            reasonCodes = reasonCodes == null ? List.of() : List.copyOf(reasonCodes);
        }
    }

    private static final class MutableBackendSummary {
        private int steps;
        private int bufferBindingSteps;
        private int tensorArraySteps;
        private int cpuFallbackSteps;
        private int unavailableSteps;
        private int preparedInputSteps;
        private long inputBytes;
        private long outputBytes;
        private long javaToNativeCopyNs;
        private long nativeToJavaCopyNs;
        private long nativeDeviceCopyNs;
        private final LinkedHashSet<String> nativeCopyStrategies = new LinkedHashSet<>();
        private final LinkedHashSet<String> outputBufferWriteStatuses = new LinkedHashSet<>();
        private final LinkedHashMap<String, Integer> executionRouteCounts = new LinkedHashMap<>();
        private final LinkedHashMap<String, Integer> rejectedRouteReasonCounts = new LinkedHashMap<>();
        private final LinkedHashSet<String> fallbackReasons = new LinkedHashSet<>();
        private final LinkedHashSet<String> reasonCodes = new LinkedHashSet<>();

        private BackendSummary toImmutable() {
            return new BackendSummary(
                    steps,
                    bufferBindingSteps,
                    tensorArraySteps,
                    cpuFallbackSteps,
                    unavailableSteps,
                    preparedInputSteps,
                    inputBytes,
                    outputBytes,
                    javaToNativeCopyNs,
                    nativeToJavaCopyNs,
                    nativeDeviceCopyNs,
                    List.copyOf(nativeCopyStrategies),
                    List.copyOf(outputBufferWriteStatuses),
                    new LinkedHashMap<>(executionRouteCounts),
                    new LinkedHashMap<>(rejectedRouteReasonCounts),
                    List.copyOf(fallbackReasons),
                    List.copyOf(reasonCodes)
            );
        }
    }
}

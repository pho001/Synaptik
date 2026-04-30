package tuning.benchmark.report;

import graph.execution.trace.ExecutionStepTrace;

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
            addNonBlank(summary.fallbackReasons, attrs.get("acceleratorBufferReason"));
            addNonBlank(summary.fallbackReasons, attrs.get("metalFallbackReason"));
            summary.javaToNativeCopyNs += longAttr(attrs.get("metalJavaToNativeCopyNs"));
            summary.nativeToJavaCopyNs += longAttr(attrs.get("metalNativeToJavaCopyNs"));
            summary.nativeDeviceCopyNs += longAttr(attrs.get("metalNativeDeviceCopyNs"));
            summary.inputBytes += longAttr(attrs.get("metalInputBytes"));
            summary.outputBytes += longAttr(attrs.get("metalOutputBytes"));
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
            List<String> fallbackReasons,
            List<String> reasonCodes
    ) {
        public BackendSummary {
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
                    List.copyOf(fallbackReasons),
                    List.copyOf(reasonCodes)
            );
        }
    }
}

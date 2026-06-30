package runtime.runner;

import runtime.execution.ExecutionContext;
import graph.model.CompiledNode;
import graph.execution.PreparedExecutionStep;
import runtime.execution.PreparedStepExecutable;
import trace.execution.ExecutionStepTrace;
import trace.execution.StepExecutionMetadata;
import trace.backend.StepTraceContribution;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Builds per-step execution trace records from prepared metadata and runtime diagnostics.
 */
public final class StepExecutionTracer {
    private StepExecutionTracer() {
    }

    public static ExecutionStepTrace toStepTrace(int index, PreparedExecutionStep step, long durationNs, ExecutionContext context) {
        CompiledNode node = step.compiledNode();
        var metadata = step.metadata();
        operations.Operation executionOperation = metadata.executionOperation() == null
                ? node.operation()
                : metadata.executionOperation();
        String opType = executionOperation == null ? "LEAF" : executionOperation.opType().name();
        StepTraceContribution contribution = contribution(node, step, context);
        return new ExecutionStepTrace(
                index,
                node.label(),
                opType,
                java.util.Arrays.stream(node.shape()).boxed().toList(),
                node.dataType().name(),
                metadata.backend().name(),
                contribution.kernel(),
                durationNs,
                buildStepMetadata(node, contribution, context)
        );
    }

    private static StepTraceContribution contribution(
            CompiledNode node,
            PreparedExecutionStep step,
            ExecutionContext context
    ) {
        PreparedStepExecutable executable = step.metadata().executable();
        return executable.traceContribution(node, step.metadata(), context);
    }

    private static StepExecutionMetadata buildStepMetadata(
            CompiledNode node,
            StepTraceContribution contribution,
            ExecutionContext context
    ) {
        LinkedHashMap<String, Object> attrs = new LinkedHashMap<>(contribution.attributes());
        addStorageAttrs(node, context, attrs);
        addFallbackSummary(attrs);
        return new StepExecutionMetadata(
                "node",
                attrs,
                contribution.compute(),
                contribution.layout(),
                contribution.dispatch(),
                contribution.reduction(),
                contribution.matMul(),
                contribution.conv(),
                contribution.fused()
        );
    }

    private static void addStorageAttrs(
            CompiledNode node,
            ExecutionContext context,
            LinkedHashMap<String, Object> attrs
    ) {
        var residency = context.residencyForNodeId(node.id());
        if (residency != null) {
            attrs.put("storageResidency", residency.residency().name());
            attrs.put("storageCpuCurrent", residency.cpuCurrent());
            attrs.put("storageDeviceCurrent", residency.deviceCurrent());
            attrs.put("storageDeviceBackend", residency.deviceBackend());
            attrs.put("storageTransitionReason", residency.lastTransitionReason());
        }
        var deviceBinding = context.deviceBufferBindingForNodeId(node.id());
        if (deviceBinding != null) {
            attrs.put("deviceBufferBackend", deviceBinding.backendId());
            attrs.put("deviceBufferBytes", deviceBinding.logicalByteLength());
            attrs.put("deviceBufferAvailable", deviceBinding.available());
            attrs.put("deviceBuffer", deviceBinding.describe());
        }
    }

    private static void addFallbackSummary(LinkedHashMap<String, Object> attrs) {
        ArrayList<String> kinds = new ArrayList<>();
        ArrayList<String> reasonCodes = new ArrayList<>();
        ArrayList<String> reasons = new ArrayList<>();

        String acceleratorPath = stringAttr(attrs, "acceleratorBufferExecutionPath");
        if ("CPU_FALLBACK".equals(acceleratorPath)) {
            addFallback(kinds, reasonCodes, reasons, "ACCELERATOR_CPU_FALLBACK", stringAttr(attrs, "acceleratorBufferReasonCode"), stringAttr(attrs, "acceleratorBufferReason"));
        } else if ("TENSOR_ARRAY".equals(acceleratorPath)) {
            addFallback(kinds, reasonCodes, reasons, "ACCELERATOR_TENSOR_ARRAY_FALLBACK", stringAttr(attrs, "acceleratorBufferReasonCode"), stringAttr(attrs, "acceleratorBufferReason"));
        } else if ("UNAVAILABLE".equals(acceleratorPath)) {
            addFallback(kinds, reasonCodes, reasons, "ACCELERATOR_BUFFER_UNAVAILABLE", stringAttr(attrs, "acceleratorBufferReasonCode"), stringAttr(attrs, "acceleratorBufferReason"));
        }

        if (Boolean.TRUE.equals(attrs.get("metalUsedCpuFallback"))) {
            addFallback(kinds, reasonCodes, reasons, "METAL_CPU_FALLBACK", stringAttr(attrs, "metalRouteReasonCode"), firstNonBlank(stringAttr(attrs, "metalFallbackReason"), stringAttr(attrs, "metalRouteReason")));
        }
        if ("TENSOR_ARRAY_COPY".equals(stringAttr(attrs, "metalExecutionPath"))
                || "TENSOR_ARRAY".equals(stringAttr(attrs, "metalExecutionRoute"))) {
            addFallback(kinds, reasonCodes, reasons, "METAL_TENSOR_ARRAY_FALLBACK", stringAttr(attrs, "metalRouteReasonCode"), firstNonBlank(stringAttr(attrs, "metalRouteReason"), stringAttr(attrs, "acceleratorBufferReason")));
        }

        if (Boolean.TRUE.equals(attrs.get("cudaUsedCpuFallback"))) {
            addFallback(kinds, reasonCodes, reasons, "CUDA_CPU_FALLBACK", stringAttr(attrs, "acceleratorBufferReasonCode"), firstNonBlank(stringAttr(attrs, "cudaFallbackReason"), stringAttr(attrs, "acceleratorBufferReason")));
        }
        if ("TENSOR_ARRAY".equals(stringAttr(attrs, "cudaExecutionPath"))) {
            addFallback(kinds, reasonCodes, reasons, "CUDA_TENSOR_ARRAY_FALLBACK", stringAttr(attrs, "acceleratorBufferReasonCode"), firstNonBlank(stringAttr(attrs, "cudaFallbackReason"), stringAttr(attrs, "acceleratorBufferReason")));
        }

        if (!stringAttr(attrs, "matMulFallbackReason").isBlank()) {
            addFallback(kinds, reasonCodes, reasons, "CPU_MATMUL_ROUTE_FALLBACK", stringAttr(attrs, "matMulRoute"), stringAttr(attrs, "matMulFallbackReason"));
        }

        if (!kinds.isEmpty()) {
            attrs.put("fallbackOccurred", true);
            attrs.put("fallbackKind", kinds.size() == 1 ? kinds.getFirst() : "MULTIPLE");
            attrs.put("fallbackKinds", List.copyOf(kinds));
            attrs.put("fallbackReasonCode", reasonCodes.size() == 1 ? reasonCodes.getFirst() : String.join(" | ", reasonCodes));
            attrs.put("fallbackReasonCodes", List.copyOf(reasonCodes));
            attrs.put("fallbackReason", reasons.size() == 1 ? reasons.getFirst() : String.join(" | ", reasons));
            attrs.put("fallbackReasons", List.copyOf(reasons));
        }
    }

    private static void addFallback(
            ArrayList<String> kinds,
            ArrayList<String> reasonCodes,
            ArrayList<String> reasons,
            String kind,
            String reasonCode,
            String reason
    ) {
        if (kind == null || kind.isBlank() || kinds.contains(kind)) {
            return;
        }
        kinds.add(kind);
        reasonCodes.add(reasonCode == null || reasonCode.isBlank() ? "UNKNOWN" : reasonCode);
        reasons.add(reason == null || reason.isBlank() ? kind : reason);
    }

    private static String stringAttr(Map<String, Object> attrs, String key) {
        Object value = attrs.get(key);
        return value == null ? "" : String.valueOf(value);
    }

    private static String firstNonBlank(String first, String second) {
        return first == null || first.isBlank() ? (second == null ? "" : second) : first;
    }
}

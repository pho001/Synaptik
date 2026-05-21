package graph.execution.trace.contrib;

import backend.blas.OpenBlasFfmBridge;
import backend.cpu.CpuFusedExecutionArtifact;
import backend.cpu.CpuNodeExecutionArtifact;
import backend.cpu.fused.plan.FusedOperation;
import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuNodeExecutionPlan;
import backend.cpu.kernels.linalg.matmul.exec.PreparedMatMulExecutable;
import backend.cpu.kernels.linalg.matmul.plan.MatMulExecutionRoute;
import backend.runtime.ExecutionContext;
import config.runtime.BlasStorageMode;
import config.runtime.CpuStorageProfile;
import graph.CompiledNode;
import graph.execution.PreparedExecutionStep;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.trace.ComputeTraceMetadata;
import graph.execution.trace.ConvTraceMetadata;
import graph.execution.trace.DispatchTraceMetadata;
import graph.execution.trace.ExecutionStepTrace;
import graph.execution.trace.FusedTraceMetadata;
import graph.execution.trace.LayoutTraceMetadata;
import graph.execution.trace.MatMulTraceMetadata;
import graph.execution.trace.ReductionTraceMetadata;
import graph.execution.trace.StepExecutionMetadata;
import tensor.Tensor;

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
        CpuKernel cpuKernel = cpuKernel(metadata);
        String kernel = cpuKernel == null ? "" : cpuKernel.getClass().getSimpleName();
        return new ExecutionStepTrace(
                index,
                node.label(),
                opType,
                java.util.Arrays.stream(node.shape()).boxed().toList(),
                node.dataType(),
                metadata.backend().name(),
                kernel,
                durationNs,
                buildStepMetadata(node, step, context)
        );
    }

    private static StepExecutionMetadata buildStepMetadata(CompiledNode node, PreparedExecutionStep step, ExecutionContext context) {
        var metadata = step.metadata();
        CpuNodeExecutionPlan cpuPlan = cpuPlan(metadata);
        LinkedHashMap<String, Object> attrs = new LinkedHashMap<>();
        ComputeTraceMetadata compute = null;
        LayoutTraceMetadata layout = new LayoutTraceMetadata(
                node.storageOffset(),
                node.contiguous(),
                cpuPlan != null && cpuPlan.stridedPath(),
                cpuPlan == null ? "" : cpuPlan.targetType().name()
        );
        DispatchTraceMetadata dispatch = null;
        ReductionTraceMetadata reduction = null;
        MatMulTraceMetadata matMul = null;
        ConvTraceMetadata conv = null;
        FusedTraceMetadata fusedMeta = null;

        if (cpuPlan != null) {
            var plan = cpuPlan;
            compute = new ComputeTraceMetadata(
                    plan.computeContract().computeType().name(),
                    plan.computeContract().storageType().name(),
                    plan.computeContract().computeType().name(),
                    plan.computeContract().backend().name(),
                    plan.computeContract().accumulateType().name()
            );
            if (plan.dispatchHints() != null) {
                dispatch = new DispatchTraceMetadata(
                        plan.dispatchHints().mode().name(),
                        plan.dispatchHints().vectorWidth(),
                        plan.dispatchHints().plannedWorkers(),
                        plan.dispatchHints().scalarChunkSize(),
                        plan.dispatchHints().vectorChunkSize()
                );
            }
            if (plan.reductionHints() != null) {
                reduction = new ReductionTraceMetadata(
                        plan.reductionHints().mode().name(),
                        plan.reductionHints().plannedWorkers(),
                        plan.reductionHints().chunkSize(),
                        plan.reductionHints().vectorWidth(),
                        plan.reductionHints().accuracyMode().name()
                );
            }
            if (plan.matMulHints() != null) {
                PreparedMatMulExecutable executable = plan.matMulExecutable();
                MatMulExecutionRoute route = executable == null || executable.lastExecutionRoute() == null
                        ? plan.matMulHints().route()
                        : executable.lastExecutionRoute();
                String blasProvider = matMulBlasProvider(context);
                String blasSymbol = matMulBlasSymbol(node, route, executable, plan);
                String nativeCpuFallbackReason = executable == null ? "" : executable.lastFallbackReason();
                boolean openblasProvider = "OPENBLAS_FFM".equals(blasProvider);
                matMul = new MatMulTraceMetadata(
                        plan.matMulHints().useBlas(),
                        plan.matMulHints().useBatchedBlas(),
                        blasProvider,
                        blasSymbol,
                        route.name(),
                        route.name(),
                        matMulCpuStorageProfile(context),
                        matMulNativeCpuFailurePolicy(context),
                        matMulRequestedCpuStorage(context),
                        matMulActualCpuStorage(route),
                        nativeCpuFallbackReason,
                        openblasProvider && OpenBlasFfmBridge.isFloat32GemmAvailable(),
                        openblasProvider && OpenBlasFfmBridge.isFloat64GemmAvailable(),
                        openblasProvider && OpenBlasFfmBridge.isBFloat16ToFloatGemmAvailable(),
                        openblasProvider && OpenBlasFfmBridge.isBFloat16OutputGemmAvailable(),
                        matMulBf16ContinuationRoute(node, route, blasSymbol),
                        matMulBf16OutputRoute(node, route, blasSymbol),
                        matMulBf16ComputePrecision(node, route, blasSymbol),
                        matMulBf16OutputPrecision(node, route, blasSymbol),
                        matMulCopyInBytes(node, step, context, executable, route),
                        matMulCopyOutBytes(node, executable, route),
                        matMulNativeTempBytes(route),
                        matMulThreadPolicy(context),
                        nativeCpuFallbackReason,
                        plan.matMulHints().parallel(),
                        plan.matMulHints().tileM(),
                        plan.matMulHints().tileN(),
                        plan.matMulHints().tileK(),
                        plan.matMulHints().plannedWorkers(),
                        plan.matMulHints().work(),
                        plan.matMulHints().microKernel().name()
                );
            }
        }

        ConvTraceMetadata trace = context.convTraceForNodeId(node.id());
        if (trace != null) {
            conv = trace;
        }

        operations.Operation executionOperation = metadata.executionOperation() == null
                ? node.operation()
                : metadata.executionOperation();
        if (executionOperation instanceof FusedOperation fused) {
            var fusedExecutable = fusedExecutable(step.metadata());
            String executionBackend = fusedExecutable == null
                    ? ""
                    : fusedExecutable.getClass().getSimpleName();
            fusedMeta = new FusedTraceMetadata(
                    fused.getPrecisionMode(),
                    fused.isLowCostHint(),
                    fused.getDispatchFamily().id(),
                    fused.getSchedulerSignature(),
                    executionBackend,
                    fused.getDispatchScale(),
                    fused.getPlan().nodeCount(),
                    fused.getPlan().inputCount()
            );
        }

        BackendRunTraceContributors.contribute(
                new BackendRunTraceContext(node, step, context, matMul),
                attrs
        );

        addFallbackSummary(attrs);
        return new StepExecutionMetadata("node", attrs, compute, layout, dispatch, reduction, matMul, conv, fusedMeta);
    }

    private static long matMulCopyInBytes(
            CompiledNode node,
            PreparedExecutionStep step,
            ExecutionContext context,
            PreparedMatMulExecutable executable,
            MatMulExecutionRoute route
    ) {
        if (executable != null && executable.lastCopyInBytes() >= 0L) {
            return executable.lastCopyInBytes();
        }
        if (route != MatMulExecutionRoute.OPENBLAS_ARRAY_COPYING) {
            return route == MatMulExecutionRoute.OPENBLAS_NATIVE_SEGMENT ? 0L : -1L;
        }
        List<Integer> inputIds = step.metadata().executionInputNodeIds().isEmpty()
                ? node.inputIds()
                : step.metadata().executionInputNodeIds();
        long bytes = 0L;
        for (int inputId : inputIds) {
            bytes += logicalByteLength(context.runtimeTensorForNodeId(inputId));
        }
        return bytes;
    }

    private static long matMulCopyOutBytes(
            CompiledNode node,
            PreparedMatMulExecutable executable,
            MatMulExecutionRoute route
    ) {
        if (executable != null && executable.lastCopyOutBytes() >= 0L) {
            return executable.lastCopyOutBytes();
        }
        if (route != MatMulExecutionRoute.OPENBLAS_ARRAY_COPYING) {
            return route == MatMulExecutionRoute.OPENBLAS_NATIVE_SEGMENT ? 0L : -1L;
        }
        return logicalByteLength(node.dataType(), node.shape());
    }

    private static long matMulNativeTempBytes(MatMulExecutionRoute route) {
        return route == MatMulExecutionRoute.OPENBLAS_NATIVE_SEGMENT ? 0L : -1L;
    }

    private static String matMulBlasProvider(ExecutionContext context) {
        if (context.runtimeConfig() == null || context.runtimeConfig().blas() == null) {
            return "";
        }
        return context.runtimeConfig().blas().provider().name();
    }

    private static String matMulThreadPolicy(ExecutionContext context) {
        if (context.runtimeConfig() == null
                || context.runtimeConfig().blas() == null
                || context.runtimeConfig().blas().provider() != backend.blas.BlasProvider.OPENBLAS_FFM) {
            return "";
        }
        return OpenBlasFfmBridge.threadPolicy();
    }

    private static String matMulCpuStorageProfile(ExecutionContext context) {
        return context.runtimeConfig() == null || context.runtimeConfig().cpuStorageProfile() == null
                ? ""
                : context.runtimeConfig().cpuStorageProfile().name();
    }

    private static String matMulNativeCpuFailurePolicy(ExecutionContext context) {
        return context.runtimeConfig() == null || context.runtimeConfig().nativeCpuFailurePolicy() == null
                ? ""
                : context.runtimeConfig().nativeCpuFailurePolicy().name();
    }

    private static String matMulRequestedCpuStorage(ExecutionContext context) {
        if (context.runtimeConfig() == null || context.runtimeConfig().blas() == null) {
            return "";
        }
        CpuStorageProfile profile = context.runtimeConfig().cpuStorageProfile();
        BlasStorageMode mode = switch (profile) {
            case CPU_ARRAY -> BlasStorageMode.CPU_ARRAY;
            case CPU_NATIVE -> BlasStorageMode.CPU_NATIVE;
            case AUTO -> context.runtimeConfig().blas().storageMode();
        };
        return mode.name();
    }

    private static String matMulActualCpuStorage(MatMulExecutionRoute route) {
        return route == MatMulExecutionRoute.OPENBLAS_NATIVE_SEGMENT ? "CPU_NATIVE" : "CPU_ARRAY";
    }

    private static String matMulBlasSymbol(CompiledNode node, MatMulExecutionRoute route, PreparedMatMulExecutable executable, CpuNodeExecutionPlan plan) {
        if (executable != null && !executable.lastBlasSymbol().isBlank()) {
            return executable.lastBlasSymbol();
        }
        if (route == MatMulExecutionRoute.JAVA_DIRECT) {
            return "";
        }
        if (isBFloat16LinearSbgemmRoute(node, plan)) {
            return "cblas_sbgemm";
        }
        return switch (node.dataType()) {
            case FLOAT32 -> "cblas_sgemm";
            case FLOAT64 -> "cblas_dgemm";
            case BFLOAT16 -> "cblas_bgemm";
            default -> "";
        };
    }

    private static boolean isBFloat16LinearSbgemmRoute(CompiledNode node, CpuNodeExecutionPlan plan) {
        if (node.dataType() != tensor.DataType.BFLOAT16
                || !(node.operation() instanceof operations.linalg.linear linearOp)
                || plan == null
                || plan.matMulHints() == null
                || (!plan.matMulHints().useBlas() && !plan.matMulHints().useBatchedBlas())) {
            return false;
        }
        return OpenBlasFfmBridge.isBFloat16ToFloatGemmAvailable()
                && (plan.publishFloatContinuation() || linearOp.hasBias());
    }

    private static String matMulBf16ContinuationRoute(CompiledNode node, MatMulExecutionRoute route, String blasSymbol) {
        if (node.dataType() != tensor.DataType.BFLOAT16) {
            return "";
        }
        if ("cblas_sbgemm".equals(blasSymbol)) {
            return "SBGEMM";
        }
        if (route == MatMulExecutionRoute.JAVA_DIRECT) {
            return "JAVA";
        }
        if ("cblas_bgemm".equals(blasSymbol)) {
            return "";
        }
        return "UNAVAILABLE";
    }

    private static String matMulBf16OutputRoute(CompiledNode node, MatMulExecutionRoute route, String blasSymbol) {
        if (node.dataType() != tensor.DataType.BFLOAT16) {
            return "";
        }
        if ("cblas_bgemm".equals(blasSymbol)) {
            return "BGEMM";
        }
        if ("cblas_sbgemm".equals(blasSymbol)) {
            return "PROMOTED_F32";
        }
        if (route == MatMulExecutionRoute.JAVA_DIRECT) {
            return "JAVA";
        }
        return "UNAVAILABLE";
    }

    private static String matMulBf16ComputePrecision(CompiledNode node, MatMulExecutionRoute route, String blasSymbol) {
        if (node.dataType() != tensor.DataType.BFLOAT16) {
            return "";
        }
        if ("cblas_bgemm".equals(blasSymbol)) {
            return "BF16_OUTPUT";
        }
        if ("cblas_sbgemm".equals(blasSymbol) || route == MatMulExecutionRoute.JAVA_DIRECT) {
            return "F32_PROMOTED";
        }
        return "UNAVAILABLE";
    }

    private static String matMulBf16OutputPrecision(CompiledNode node, MatMulExecutionRoute route, String blasSymbol) {
        if (node.dataType() != tensor.DataType.BFLOAT16) {
            return "";
        }
        if ("cblas_sbgemm".equals(blasSymbol)) {
            return "F32";
        }
        if ("cblas_bgemm".equals(blasSymbol) || route == MatMulExecutionRoute.JAVA_DIRECT) {
            return "BF16";
        }
        return "UNAVAILABLE";
    }

    private static long logicalByteLength(Tensor tensor) {
        if (tensor == null) {
            return 0L;
        }
        return Math.multiplyExact((long) tensor.getFlatDataSize(), elementBytes(tensor.getDataType()));
    }

    private static long logicalByteLength(tensor.DataType dataType, int[] shape) {
        long elements = 1L;
        for (int dim : shape == null ? new int[0] : shape) {
            elements = Math.multiplyExact(elements, Math.max(0, dim));
        }
        return Math.multiplyExact(elements, elementBytes(dataType));
    }

    private static int elementBytes(tensor.DataType dataType) {
        return switch (dataType) {
            case FLOAT64, INT64 -> Long.BYTES;
            case FLOAT32, INT32 -> Integer.BYTES;
            case BFLOAT16 -> Short.BYTES;
            case BOOL -> Byte.BYTES;
        };
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

    private static CpuKernel cpuKernel(CompiledNodeExecutionMetadata metadata) {
        if (metadata.artifact() instanceof CpuNodeExecutionArtifact artifact) {
            return artifact.cpuKernel();
        }
        if (metadata.artifact() instanceof CpuFusedExecutionArtifact artifact) {
            return artifact.cpuKernel();
        }
        return null;
    }

    private static CpuNodeExecutionPlan cpuPlan(CompiledNodeExecutionMetadata metadata) {
        if (metadata.artifact() instanceof CpuNodeExecutionArtifact artifact) {
            return artifact.cpuPlan();
        }
        if (metadata.artifact() instanceof CpuFusedExecutionArtifact artifact) {
            return artifact.cpuPlan();
        }
        return null;
    }

    private static backend.cpu.fused.exec.PreparedFusedExecutable fusedExecutable(CompiledNodeExecutionMetadata metadata) {
        return metadata.artifact() instanceof CpuFusedExecutionArtifact artifact
                ? artifact.fusedExecutable()
                : null;
    }
}

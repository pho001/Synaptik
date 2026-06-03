package backend.cpu;

import backend.blas.OpenBlasRuntime;
import backend.cpu.fused.plan.FusedOperation;
import backend.cpu.fused.plan.FusedVectorFallbackReason;
import backend.cpu.kernels.CpuKernel;
import backend.cpu.plan.CpuNodeExecutionPlan;
import backend.cpu.provider.linalg.matmul.PreparedMatMulExecutable;
import backend.cpu.plan.linalg.matmul.MatMulExecutionRoute;
import backend.memory.TensorResidencyState;
import backend.cpu.nativecpu.NativeCpuTraceState;
import backend.runtime.ExecutionContext;
import config.runtime.BlasStorageMode;
import config.runtime.CpuStorageProfile;
import graph.CompiledNode;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.trace.ComputeTraceMetadata;
import graph.execution.trace.ConvTraceMetadata;
import graph.execution.trace.DispatchTraceMetadata;
import graph.execution.trace.FusedTraceMetadata;
import graph.execution.trace.LayoutTraceMetadata;
import graph.execution.trace.MatMulTraceMetadata;
import graph.execution.trace.ReductionTraceMetadata;
import graph.execution.trace.StepTraceContribution;
import operations.Operation;
import tensor.Tensor;

import java.util.LinkedHashMap;
import java.util.List;

public final class CpuStepTraceContributor {
    private CpuStepTraceContributor() {
    }

    public static StepTraceContribution contribute(
            CompiledNode node,
            CompiledNodeExecutionMetadata metadata,
            ExecutionContext context
    ) {
        CpuKernel cpuKernel = cpuKernel(metadata);
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
        FusedTraceMetadata fusedMeta = null;

        if (cpuPlan != null) {
            compute = new ComputeTraceMetadata(
                    cpuPlan.computeContract().computeType().name(),
                    cpuPlan.computeContract().storageType().name(),
                    cpuPlan.computeContract().computeType().name(),
                    cpuPlan.computeContract().backend().name(),
                    cpuPlan.computeContract().accumulateType().name()
            );
            if (cpuPlan.dispatchHints() != null) {
                dispatch = new DispatchTraceMetadata(
                        cpuPlan.dispatchHints().mode().name(),
                        cpuPlan.dispatchHints().vectorWidth(),
                        cpuPlan.dispatchHints().plannedWorkers(),
                        cpuPlan.dispatchHints().scalarChunkSize(),
                        cpuPlan.dispatchHints().vectorChunkSize()
                );
            }
            if (cpuPlan.reductionHints() != null) {
                reduction = new ReductionTraceMetadata(
                        cpuPlan.reductionHints().mode().name(),
                        cpuPlan.reductionHints().plannedWorkers(),
                        cpuPlan.reductionHints().chunkSize(),
                        cpuPlan.reductionHints().vectorWidth(),
                        cpuPlan.reductionHints().accuracyMode().name()
                );
            }
            if (cpuPlan.matMulHints() != null) {
                matMul = matMulTrace(node, metadata, context, cpuPlan);
            }
        }

        ConvTraceMetadata conv = context.convTraceForNodeId(node.id());
        Operation executionOperation = metadata.executionOperation() == null
                ? node.operation()
                : metadata.executionOperation();
        if (executionOperation instanceof FusedOperation fused) {
            var fusedExecutable = metadata.artifact() instanceof CpuFusedExecutionArtifact artifact
                    ? artifact.fusedExecutable()
                    : null;
            String executionClass = fusedExecutable == null ? "" : fusedExecutable.getClass().getSimpleName();
            FusedVectorFallbackReason vectorFallbackReason = metadata.artifact() instanceof CpuFusedExecutionArtifact artifact
                    ? artifact.vectorFallbackReason()
                    : FusedVectorFallbackReason.NONE;
            attrs.put("fusedInputStorageKind", fused.getNumericContract().inputStorageKind().name());
            attrs.put("fusedOutputStorageKind", fused.getNumericContract().outputStorageKind().name());
            attrs.put("fusedExecutionClass", executionClass);
            attrs.put("fusedVectorFallbackReason", vectorFallbackReason.name());
            attrs.put("fusedVectorEligible", vectorFallbackReason == FusedVectorFallbackReason.NONE
                    && cpuPlan != null
                    && cpuPlan.dispatchHints() != null
                    && cpuPlan.dispatchHints().vectorWidth() > 1);
            addFusedNativeOutputWriteAttrs(attrs, fused, node, context);
            fusedMeta = new FusedTraceMetadata(
                    fused.getNumericContract().signatureToken(),
                    fused.isLowCostHint(),
                    fused.getDispatchFamily().id(),
                    fused.getSchedulerSignature(),
                    executionClass,
                    fused.getPlan().nodeCount(),
                    fused.getPlan().inputCount(),
                    vectorFallbackReason.name()
            );
        }

        addMatMulAttrs(attrs, matMul);
        addNativeCpuStateAttrs(attrs, node, context);
        return new StepTraceContribution(
                cpuKernel == null ? "" : cpuKernel.getClass().getSimpleName(),
                attrs,
                compute,
                layout,
                dispatch,
                reduction,
                matMul,
                conv,
                fusedMeta
        );
    }

    private static void addFusedNativeOutputWriteAttrs(
            LinkedHashMap<String, Object> attrs,
            FusedOperation fused,
            CompiledNode node,
            ExecutionContext context
    ) {
        if (fused == null || !fused.getNumericContract().usesMemorySegmentStorage()) {
            return;
        }
        TensorResidencyState residency = context.residencyForNodeId(node.id());
        if (residency == null || !residency.nativeCurrent() || context.nativeStorageForNodeId(node.id()) == null) {
            return;
        }
        attrs.put("fusedNativeOutputWritten", true);
        attrs.put("fusedNativeOutputResidency", residency.residency().name());
        attrs.put("fusedNativeOutputWriteReason", residency.lastTransitionReason());
    }

    private static MatMulTraceMetadata matMulTrace(
            CompiledNode node,
            CompiledNodeExecutionMetadata metadata,
            ExecutionContext context,
            CpuNodeExecutionPlan plan
    ) {
        PreparedMatMulExecutable executable = plan.matMulExecutable();
        MatMulExecutionRoute route = executable == null || executable.lastExecutionRoute() == null
                ? plan.matMulHints().route()
                : executable.lastExecutionRoute();
        String blasProvider = matMulBlasProvider(context);
        String blasSymbol = matMulBlasSymbol(node, route, executable, plan);
        String nativeCpuFallbackReason = executable == null ? "" : executable.lastFallbackReason();
        boolean openblasProvider = "OPENBLAS_FFM".equals(blasProvider);
        return new MatMulTraceMetadata(
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
                openblasProvider && OpenBlasRuntime.isFloat32GemmAvailable(),
                openblasProvider && OpenBlasRuntime.isFloat64GemmAvailable(),
                openblasProvider && OpenBlasRuntime.isBFloat16ToFloatGemmAvailable(),
                openblasProvider && OpenBlasRuntime.isBFloat16OutputGemmAvailable(),
                matMulBf16ContinuationRoute(node, route, blasSymbol),
                matMulBf16OutputRoute(node, route, blasSymbol),
                matMulBf16ComputePrecision(node, route, blasSymbol),
                matMulBf16OutputPrecision(node, route, blasSymbol),
                matMulCopyInBytes(node, metadata, context, executable, route),
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

    private static void addMatMulAttrs(LinkedHashMap<String, Object> attrs, MatMulTraceMetadata matMul) {
        if (matMul == null) {
            return;
        }
        attrs.put("matMulRoute", matMul.route());
        attrs.put("blasProvider", matMul.blasProvider());
        attrs.put("blasSymbol", matMul.blasSymbol());
        attrs.put("blasRoute", matMul.blasRoute());
        attrs.put("cpuStorageProfile", matMul.cpuStorageProfile());
        attrs.put("nativeCpuFailurePolicy", matMul.nativeCpuFailurePolicy());
        attrs.put("requestedCpuStorage", matMul.requestedCpuStorage());
        attrs.put("actualCpuStorage", matMul.actualCpuStorage());
        attrs.put("nativeCpuFallbackReason", matMul.nativeCpuFallbackReason());
        attrs.put("openblasSgemmAvailable", matMul.openblasSgemmAvailable());
        attrs.put("openblasDgemmAvailable", matMul.openblasDgemmAvailable());
        attrs.put("openblasSbgemmAvailable", matMul.openblasSbgemmAvailable());
        attrs.put("openblasBgemmAvailable", matMul.openblasBgemmAvailable());
        attrs.put("bf16ContinuationRoute", matMul.bf16ContinuationRoute());
        attrs.put("bf16OutputRoute", matMul.bf16OutputRoute());
        attrs.put("bf16ComputePrecision", matMul.bf16ComputePrecision());
        attrs.put("bf16OutputPrecision", matMul.bf16OutputPrecision());
        if ("OPENBLAS_FFM".equals(matMul.blasProvider())) {
            attrs.put("openblasLookupSource", OpenBlasRuntime.lookupSource());
        }
        attrs.put("matMulCopyInBytes", matMul.copyInBytes());
        attrs.put("matMulCopyOutBytes", matMul.copyOutBytes());
        attrs.put("matMulNativeTempBytes", matMul.nativeTempBytes());
        attrs.put("blasThreadPolicy", matMul.threadPolicy());
        if (!matMul.fallbackReason().isBlank()) {
            attrs.put("matMulFallbackReason", matMul.fallbackReason());
        }
    }

    private static void addNativeCpuStateAttrs(LinkedHashMap<String, Object> attrs, CompiledNode node, ExecutionContext context) {
        Tensor runtimeTensor = safeRuntimeTensor(context, node.id());
        NativeCpuTraceState nativeCpu = runtimeTensor == null
                ? null
                : context.runtimeStateFor(runtimeTensor, NativeCpuTraceState.class);
        if (nativeCpu == null) {
            return;
        }
        attrs.put("cpuStorageProfile", nativeCpu.cpuStorageProfile());
        attrs.put("nativeCpuFailurePolicy", nativeCpu.nativeCpuFailurePolicy());
        attrs.put("requestedCpuStorage", nativeCpu.requestedCpuStorage());
        attrs.put("actualCpuStorage", nativeCpu.actualCpuStorage());
        attrs.put("nativeCpuKernelStatus", nativeCpu.nativeCpuKernelStatus());
        attrs.put("nativeCpuKernelFamily", nativeCpu.nativeCpuKernelFamily());
        attrs.put("nativeCpuFallbackReason", nativeCpu.nativeCpuFallbackReason());
        if (!nativeCpu.storagePrecision().isBlank()) {
            attrs.put("storagePrecision", nativeCpu.storagePrecision());
        }
        if (!nativeCpu.computePrecision().isBlank()) {
            attrs.put("computePrecision", nativeCpu.computePrecision());
        }
    }

    private static long matMulCopyInBytes(
            CompiledNode node,
            CompiledNodeExecutionMetadata metadata,
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
        List<Integer> inputIds = metadata.executionInputNodeIds().isEmpty()
                ? node.inputIds()
                : metadata.executionInputNodeIds();
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
        return OpenBlasRuntime.threadPolicy();
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
        return OpenBlasRuntime.isBFloat16ToFloatGemmAvailable()
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

    private static Tensor safeRuntimeTensor(ExecutionContext context, int nodeId) {
        try {
            return context.runtimeTensorForNodeId(nodeId);
        } catch (RuntimeException ignored) {
            return null;
        }
    }

    private static CpuKernel cpuKernel(CompiledNodeExecutionMetadata metadata) {
        if (metadata == null) {
            return null;
        }
        if (metadata.artifact() instanceof CpuNodeExecutionArtifact artifact) {
            return artifact.cpuKernel();
        }
        if (metadata.artifact() instanceof CpuFusedExecutionArtifact artifact) {
            return artifact.cpuKernel();
        }
        return null;
    }

    private static CpuNodeExecutionPlan cpuPlan(CompiledNodeExecutionMetadata metadata) {
        if (metadata == null) {
            return null;
        }
        if (metadata.artifact() instanceof CpuNodeExecutionArtifact artifact) {
            return artifact.cpuPlan();
        }
        if (metadata.artifact() instanceof CpuFusedExecutionArtifact artifact) {
            return artifact.cpuPlan();
        }
        return null;
    }
}

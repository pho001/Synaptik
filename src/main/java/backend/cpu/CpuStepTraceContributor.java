package backend.cpu;

import backend.ComputeBackend;
import backend.accelerator.exec.AcceleratorExecutionArtifact;
import backend.blas.OpenBlasRuntime;
import backend.cpu.fused.plan.FusedOperation;
import backend.cpu.fused.plan.FusedVectorFallbackReason;
import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuNativeStorageSupport;
import backend.cpu.kernels.CpuNodeExecutionPlan;
import backend.cpu.kernels.linalg.matmul.exec.PreparedMatMulExecutable;
import backend.cpu.kernels.linalg.matmul.plan.MatMulExecutionRoute;
import backend.memory.TensorResidencyState;
import backend.cpu.nativecpu.NativeCpuTraceState;
import backend.cpu.nativecpu.layout.NativeCpuLayoutClass;
import backend.cpu.nativecpu.layout.NativeCpuStorageFamily;
import backend.cpu.nativecpu.layout.TensorPhysicalView;
import backend.cpu.region.PreparedCpuRegionExecutable;
import backend.lowering.region.CpuNativeRegionPayload;
import backend.lowering.region.RegionExecutionPlan;
import backend.lowering.region.RegionNodePlan;
import backend.runtime.ExecutionContext;
import config.runtime.BlasStorageMode;
import config.runtime.CpuStorageProfile;
import config.runtime.RuntimeConfig;
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

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

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
        addNativeCpuRegionRejectionAttrs(attrs, node, metadata, context);
        addNativeCpuRegionAttrs(attrs, metadata);
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

    private static void addNativeCpuRegionAttrs(
            LinkedHashMap<String, Object> attrs,
            CompiledNodeExecutionMetadata metadata
    ) {
        PreparedCpuRegionExecutable executable = cpuRegionExecutable(metadata);
        if (executable == null || executable.regionExecutionPlan() == null) {
            return;
        }
        RegionExecutionPlan regionPlan = executable.regionExecutionPlan();
        addRegionPlanAttrs(attrs, regionPlan);
        attrs.put("nativeCpuRegionId", regionPlan.regionId());
        attrs.put("nativeCpuRegionNodeCount", regionPlan.orderedNodeIds().size());
        attrs.put("nativeCpuRegionInputs", regionPlan.externalInputNodeIds());
        attrs.put("nativeCpuRegionOutputs", regionPlan.boundaryOutputNodeIds());
        attrs.put("nativeCpuRegionRoute", executable.lastRoute());
        attrs.put("nativeCpuRegionDecision", regionPlan.decision().selected() ? "SELECTED" : "REJECTED");
        attrs.put("nativeCpuRegionReason", regionPlan.decision().reason());
        attrs.put("nativeCpuRegionFallbackReason", executable.lastFallbackReason());
        attrs.put("nativeCpuRegionLocalKernelCount", executable.lastRegionLocalKernelCount());
        attrs.put("nativeCpuRegionLocalViewCount", executable.lastRegionLocalViewCount());
        attrs.put("nativeCpuRegionExecutedGroupCount", executable.lastExecutedGroupCount());
        if (regionPlan.backendPayload() instanceof CpuNativeRegionPayload payload) {
            attrs.put("nativeCpuRegionProviderKind", payload.providerKind());
            attrs.put("nativeCpuRegionProviderNodes", payload.providerNodeIds());
            attrs.put("nativeCpuRegionLocalKernelNodes", payload.localKernelNodeIds());
            attrs.put("nativeCpuRegionViewNodes", regionPlan.nodePlans().stream()
                    .filter(nodePlan -> nodePlan.regionRole() == backend.lowering.region.RegionRole.VIEW_ALIAS)
                    .map(RegionNodePlan::nodeId)
                    .toList());
            attrs.put("nativeCpuRegionPhysicalKernels", regionPlan.nodePlans().stream()
                    .map(RegionNodePlan::physicalKernel)
                    .toList());
            attrs.put("nativeCpuRegionSegmentKernelFamilies", regionPlan.nodePlans().stream()
                    .map(RegionNodePlan::segmentKernelFamily)
                    .toList());
            attrs.put("nativeCpuRegionAutoEligible", regionPlan.nodePlans().stream()
                    .map(nodePlan -> CpuNativeStorageSupport.autoNativeRegionEligible(nodePlan.opType(), nodePlan.dataType()))
                    .toList());
            attrs.put("nativeCpuRegionResultResidencies", regionPlan.nodePlans().stream()
                    .map(CpuStepTraceContributor::nativeCpuRegionResultResidency)
                    .toList());
            attrs.put("nativeCpuRegionLayoutClasses", regionPlan.nodePlans().stream()
                    .map(RegionNodePlan::layoutClass)
                    .toList());
            attrs.put("nativeCpuRegionInputLayoutClasses", regionPlan.nodePlans().stream()
                    .map(RegionNodePlan::inputLayoutClasses)
                    .toList());
            attrs.put("nativeCpuRegionOutputLayoutClasses", regionPlan.nodePlans().stream()
                    .map(RegionNodePlan::outputLayoutClass)
                    .toList());
            attrs.put("nativeCpuLayoutClassCounts", stringCounts(regionPlan.nodePlans().stream()
                    .map(RegionNodePlan::layoutClass)
                    .toList()));
            attrs.put("nativeCpuStridedNodeCount", nativeCpuStridedNodeCount(regionPlan.nodePlans()));
            attrs.put("nativeCpuStridedMaterializationCount", regionPlan.nodePlans().stream()
                    .filter(nodePlan -> !nodePlan.materializationReason().isBlank())
                    .count());
            attrs.put("nativeCpuStridedFallbackReasons", regionPlan.nodePlans().stream()
                    .map(RegionNodePlan::materializationReason)
                    .filter(reason -> reason != null && !reason.isBlank())
                    .distinct()
                    .toList());
            attrs.put("nativeCpuRegionExecutionKinds", regionPlan.nodePlans().stream()
                    .map(nodePlan -> nodePlan.executionKind().name())
                    .toList());
            attrs.put("nativeCpuRegionStorageContracts", regionPlan.nodePlans().stream()
                    .map(nodePlan -> nodePlan.storageContract().name())
                    .toList());
            attrs.put("nativeCpuRegionNodeReasons", regionPlan.nodePlans().stream()
                    .map(RegionNodePlan::reason)
                    .toList());
            attrs.put("nativeCpuRegionSegmentScalarNodes", regionPlan.nodePlans().stream()
                    .filter(CpuStepTraceContributor::isSegmentScalarNodePlan)
                    .map(RegionNodePlan::nodeId)
                    .toList());
            attrs.put("nativeCpuRegionBf16PromotedNodes", regionPlan.nodePlans().stream()
                    .filter(CpuStepTraceContributor::isBf16PromotedRegionNodePlan)
                    .map(RegionNodePlan::nodeId)
                    .toList());
            attrs.put("nativeCpuRegionBf16PromotedSegmentScalarNodes", regionPlan.nodePlans().stream()
                    .filter(CpuStepTraceContributor::isBf16PromotedRegionNodePlan)
                    .filter(CpuStepTraceContributor::isSegmentScalarNodePlan)
                    .map(RegionNodePlan::nodeId)
                    .toList());
            if (regionPlan.nodePlans().stream().anyMatch(CpuStepTraceContributor::isBf16PromotedRegionNodePlan)) {
                attrs.put("nativeCpuRegionBf16StoragePrecision", "BF16");
                attrs.put("nativeCpuRegionBf16ComputePrecision", "F32_PROMOTED");
            }
            attrs.put("nativeCpuRegionFallbackPlanCount", payload.fallbackPlans().size());
        }
    }

    private static void addNativeCpuRegionRejectionAttrs(
            LinkedHashMap<String, Object> attrs,
            CompiledNode node,
            CompiledNodeExecutionMetadata metadata,
            ExecutionContext context
    ) {
        if (node == null || metadata == null || context == null || context.runtimeConfig() == null) {
            return;
        }
        RuntimeConfig runtimeConfig = context.runtimeConfig();
        CpuNodeExecutionPlan cpuPlan = cpuPlan(metadata);
        if (metadata.backend() != ComputeBackend.CPU
                || cpuRegionExecutable(metadata) != null
                || metadata.artifact() instanceof AcceleratorExecutionArtifact
                || runtimeConfig.cpuStorageProfile() == CpuStorageProfile.CPU_ARRAY
                || cpuPlan == null) {
            return;
        }
        if (usesDirectNativeStorage(node, cpuPlan, runtimeConfig)) {
            return;
        }
        String reason = nativeRegionRejectionReason(node, runtimeConfig, context);
        attrs.put("nativeCpuRegionDecision", "REJECTED");
        attrs.put("nativeCpuRegionReason", reason);
        attrs.put("nativeCpuRegionRoute", "CPU_ARRAY");
        attrs.put("nativeCpuRegionFallbackReason", reason);
        attrs.put("nativeCpuRegionNodeCount", 1);
        attrs.put("nativeCpuRegionInputs", node.inputIds());
        attrs.put("nativeCpuRegionOutputs", List.of(node.id()));
        attrs.put("nativeCpuRegionRejectedNode", node.id());
        attrs.put("nativeCpuRegionRejectedOp", node.operation() == null
                ? "UNKNOWN"
                : node.operation().opType().name());
        attrs.put("nativeCpuRegionAutoEligible", List.of(node.operation() != null
                && CpuNativeStorageSupport.autoNativeRegionEligible(node.operation().opType(), node.dataType())));
        attrs.put("nativeCpuRegionResultResidencies", List.of(nativeCpuSingleNodeResultResidency(node)));
        String layoutClass = nodeLayoutClassName(node);
        List<String> inputLayoutClasses = node.inputIds().stream()
                .map(inputNodeId -> safeRuntimeTensor(context, inputNodeId))
                .map(CpuStepTraceContributor::tensorLayoutClassName)
                .toList();
        ArrayList<String> rejectionLayoutClasses = new ArrayList<>();
        rejectionLayoutClasses.add(layoutClass);
        rejectionLayoutClasses.addAll(inputLayoutClasses);
        attrs.put("nativeCpuRegionLayoutClasses", List.of(layoutClass));
        attrs.put("nativeCpuRegionInputLayoutClasses", inputLayoutClasses);
        attrs.put("nativeCpuRegionOutputLayoutClasses", List.of(layoutClass));
        attrs.put("nativeCpuLayoutClassCounts", stringCounts(rejectionLayoutClasses));
        attrs.put("nativeCpuStridedNodeCount", isDenseOrViewLayout(layoutClass) && !reason.startsWith("native-layout-") ? 0 : 1);
        attrs.put("nativeCpuStridedMaterializationCount", reason.startsWith("native-layout-materialization-required:") ? 1 : 0);
        attrs.put("nativeCpuStridedFallbackReasons", reason.startsWith("native-layout-") ? List.of(reason) : List.of());
    }

    private static String nativeRegionRejectionReason(
            CompiledNode node,
            RuntimeConfig runtimeConfig,
            ExecutionContext executionContext
    ) {
        String opLabel = node == null || node.operation() == null
                ? "unknown"
                : node.operation().opType().name().toLowerCase(Locale.ROOT);
        boolean providerOp = node != null
                && node.operation() != null
                && (node.operation().opType() == Operation.OpType.MATMUL
                || node.operation().opType() == Operation.OpType.LINEAR);
        String layoutReason = nativeLayoutRejectionReason(node, providerOp, executionContext);
        if (!layoutReason.isBlank()) {
            return layoutReason;
        }
        if (providerOp && runtimeConfig.blas().provider() == backend.blas.BlasProvider.NONE) {
            return "native-cpu-region-provider-unavailable:" + opLabel;
        }
        String opReason = nativeRegionUnsupportedReason(node);
        if (runtimeConfig.cpuStorageProfile() == CpuStorageProfile.AUTO
                && nativeRegionAutoRejectsSlowOp(node)) {
            return "native-cpu-region-auto-rejected-slow-op:" + opLabel;
        }
        if (runtimeConfig.cpuStorageProfile() == CpuStorageProfile.AUTO && opReason.isBlank()) {
            return "native-cpu-region-auto-rejected:no-region-selected";
        }
        if (!opReason.isBlank()) {
            return opReason.startsWith("native-cpu-region-")
                    ? opReason
                    : "native-cpu-region-rejected:" + opReason;
        }
        return "native-cpu-region-rejected:no-region-selected";
    }

    private static boolean usesDirectNativeStorage(
            CompiledNode node,
            CpuNodeExecutionPlan cpuPlan,
            RuntimeConfig runtimeConfig
    ) {
        if (node == null || node.operation() == null || cpuPlan == null) {
            return false;
        }
        Operation.OpType opType = node.operation().opType();
        if ((opType == Operation.OpType.MATMUL || opType == Operation.OpType.LINEAR)
                && cpuPlan.matMulHints() != null
                && cpuPlan.matMulHints().usesOpenBlasMemorySegment()) {
            return true;
        }
        return runtimeConfig != null
                && runtimeConfig.cpuStorageProfile() == CpuStorageProfile.CPU_NATIVE
                && CpuNativeStorageSupport.writesNativeOutput(opType, node.dataType());
    }

    private static String nativeRegionUnsupportedReason(CompiledNode node) {
        if (node == null || node.operation() == null) {
            return "native-kernel-unknown-op";
        }
        Operation.OpType opType = node.operation().opType();
        if (opType == Operation.OpType.MATMUL || opType == Operation.OpType.LINEAR) {
            return "native-cpu-region-provider-fallback:" + opType.name().toLowerCase(Locale.ROOT);
        }
        return CpuNativeStorageSupport.unsupportedReason(opType, node.dataType());
    }

    private static boolean nativeRegionAutoRejectsSlowOp(CompiledNode node) {
        return node != null
                && node.operation() != null
                && CpuNativeStorageSupport.nativeRegionSupported(node.operation().opType(), node.dataType())
                && !CpuNativeStorageSupport.autoNativeRegionEligible(node.operation().opType(), node.dataType());
    }

    private static String nativeLayoutRejectionReason(
            CompiledNode node,
            boolean providerOp,
            ExecutionContext executionContext
    ) {
        if (node == null || node.operation() == null) {
            return "";
        }
        String opLabel = node.operation().opType().name().toLowerCase(Locale.ROOT);
        String outputLayout = nodeLayoutClassName(node);
        if ("UNSUPPORTED_LAYOUT".equals(outputLayout)) {
            return "native-layout-unsupported:node-" + node.id();
        }
        for (int inputNodeId : node.inputIds()) {
            String inputLayout = tensorLayoutClassName(safeRuntimeTensor(executionContext, inputNodeId));
            if ("DENSE_CONTIGUOUS".equals(inputLayout)) {
                continue;
            }
            if ("UNSUPPORTED_LAYOUT".equals(inputLayout)) {
                return "native-layout-unsupported:input:" + opLabel;
            }
            if (providerOp) {
                return "native-layout-materialization-required:provider-dense-input";
            }
            if ("OFFSET_CONTIGUOUS".equals(inputLayout)) {
                return "native-layout-materialization-required:offset-input:" + opLabel;
            }
            if ("BROADCAST_READ_DENSE_WRITE".equals(inputLayout)
                    || "LAST_DIM_BIAS_BROADCAST".equals(inputLayout)) {
                return "native-layout-materialization-required:broadcast-input:" + opLabel;
            }
            return "native-layout-unsupported:strided-input:" + opLabel;
        }
        if (!isDenseOrViewLayout(outputLayout)) {
            return "native-layout-unsupported:strided-output:" + opLabel;
        }
        return "";
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

    private static PreparedCpuRegionExecutable cpuRegionExecutable(CompiledNodeExecutionMetadata metadata) {
        return metadata != null && metadata.artifact() instanceof CpuRegionExecutionArtifact artifact
                ? artifact.executable()
                : null;
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

    private static void addRegionPlanAttrs(LinkedHashMap<String, Object> attrs, RegionExecutionPlan regionPlan) {
        attrs.put("regionId", regionPlan.regionId());
        attrs.put("regionTarget", regionPlan.target().name());
        attrs.put("loweringFamily", regionPlan.loweringFamily().name());
        attrs.put("anchorNodeId", regionPlan.anchorNodeId());
        attrs.put("orderedNodeIds", regionPlan.orderedNodeIds());
        attrs.put("boundaryOutputNodeIds", regionPlan.boundaryOutputNodeIds());
        attrs.put("regionNodeCount", regionPlan.orderedNodeIds().size());
        attrs.put("regionDecision", regionPlan.decision().selected() ? "SELECTED" : "REJECTED");
        attrs.put("regionReason", regionPlan.decision().reason());
        attrs.put("regionExecutionKindSummary", regionPlan.executionGroups().stream()
                .map(group -> group.executionKind().name())
                .distinct()
                .toList());
        attrs.put("regionStorageContractSummary", regionPlan.executionGroups().stream()
                .map(group -> group.storageContract().name())
                .distinct()
                .toList());
    }

    private static Map<String, Integer> stringCounts(List<String> values) {
        LinkedHashMap<String, Integer> counts = new LinkedHashMap<>();
        if (values == null) {
            return counts;
        }
        for (String value : values) {
            String key = value == null || value.isBlank() ? "UNKNOWN" : value;
            counts.merge(key, 1, Integer::sum);
        }
        return counts;
    }

    private static long nativeCpuStridedNodeCount(List<RegionNodePlan> nodePlans) {
        if (nodePlans == null) {
            return 0L;
        }
        return nodePlans.stream()
                .filter(nodePlan -> nodePlan != null && !isDenseOrViewLayout(nodePlan.layoutClass()))
                .count();
    }

    private static boolean isSegmentScalarNodePlan(RegionNodePlan nodePlan) {
        if (nodePlan == null) {
            return false;
        }
        String physicalKernel = nodePlan.physicalKernel();
        String segmentKernelFamily = nodePlan.segmentKernelFamily();
        return "SEGMENT_SCALAR".equals(physicalKernel)
                || "SEGMENT_DENSE_SCALAR".equals(segmentKernelFamily)
                || "SEGMENT_STRIDED_SCALAR".equals(segmentKernelFamily);
    }

    private static List<String> nativeCpuRegionResultResidency(RegionNodePlan nodePlan) {
        if (nodePlan == null) {
            return List.of();
        }
        if (isBoolMaskNode(nodePlan.opType(), nodePlan.dataType())) {
            return List.of("BOOL_MASK_NATIVE");
        }
        return switch (nodePlan.storageContract()) {
            case CPU_NATIVE -> List.of("CPU_NATIVE");
            case VIEW_ALIAS -> List.of("VIEW_ALIAS");
            default -> List.of("CPU_ARRAY");
        };
    }

    private static List<String> nativeCpuSingleNodeResultResidency(CompiledNode node) {
        if (node == null || node.operation() == null) {
            return List.of("CPU_ARRAY");
        }
        if (isBoolMaskNode(node.operation().opType(), node.dataType())) {
            return List.of("BOOL_MASK_ARRAY", "BOOL_MASK_NATIVE");
        }
        if (CpuNativeStorageSupport.providerRoute(node.operation().opType(), node.dataType())) {
            return List.of("CPU_NATIVE");
        }
        if (CpuNativeStorageSupport.viewAlias(node.operation().opType(), node.dataType())) {
            return List.of("VIEW_ALIAS");
        }
        return CpuNativeStorageSupport.writesNativeOutput(node.operation().opType(), node.dataType())
                ? List.of("CPU_NATIVE")
                : List.of("CPU_ARRAY");
    }

    private static boolean isBoolMaskNode(Operation.OpType opType, tensor.DataType dataType) {
        return dataType == tensor.DataType.BOOL
                && (CpuNativeStorageSupport.supportsNativeCompare(opType)
                || CpuNativeStorageSupport.supportsNativeBoolLogical(opType, dataType)
                || CpuNativeStorageSupport.supportsNativeReduction(opType, dataType));
    }

    private static boolean isBf16PromotedRegionNodePlan(RegionNodePlan nodePlan) {
        if (nodePlan == null || nodePlan.dataType() != tensor.DataType.BFLOAT16) {
            return false;
        }
        return switch (nodePlan.opType()) {
            case ADD, SUB, MUL, DIV, MIN, MAX, MUL_SCALAR, NEG, RELU, ABS, CLAMP_MIN, CLAMP_MAX, WHERE, SUM, MEAN -> true;
            default -> false;
        };
    }

    private static boolean isDenseOrViewLayout(String layoutClass) {
        return "DENSE_CONTIGUOUS".equals(layoutClass) || "VIEW_ALIAS_ONLY".equals(layoutClass);
    }

    private static String nodeLayoutClassName(CompiledNode node) {
        if (node == null) {
            return NativeCpuLayoutClass.UNSUPPORTED_LAYOUT.name();
        }
        return layoutClassName(node.id(), node.dataType(), node.shape(), node.strides(), node.storageOffset());
    }

    private static String tensorLayoutClassName(Tensor tensor) {
        if (tensor == null) {
            return NativeCpuLayoutClass.UNSUPPORTED_LAYOUT.name();
        }
        return layoutClassName(
                0,
                tensor.getDataType(),
                tensor.getShapeUnsafe(),
                tensor.getStridesUnsafe(),
                tensor.getStorageOffsetUnsafe()
        );
    }

    private static String layoutClassName(
            int nodeId,
            tensor.DataType dataType,
            int[] shape,
            int[] strides,
            int storageOffset
    ) {
        try {
            return TensorPhysicalView.of(
                    Math.max(0, nodeId),
                    dataType,
                    shape,
                    strides,
                    storageOffset,
                    NativeCpuStorageFamily.CPU_NATIVE
            ).layoutClass().name();
        } catch (RuntimeException ignored) {
            return NativeCpuLayoutClass.UNSUPPORTED_LAYOUT.name();
        }
    }
}

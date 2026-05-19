package graph.execution;

import backend.ComputeBackend;
import backend.blas.OpenBlasFfmBridge;
import backend.cpu.nativecpu.NativeCpuParityMatrix;
import backend.cpu.nativecpu.NativeCpuTraceState;
import backend.cpu.nativecpu.PreparedNativeCpuPlan;
import backend.cpu.nativecpu.PreparedNativeCpuRoute;
import backend.lowering.region.CpuNativeRegionPayload;
import backend.lowering.region.RegionNodePlan;
import config.runtime.CpuStorageProfile;
import config.runtime.RuntimeConfig;
import graph.CompiledNode;
import graph.execution.trace.MatMulTraceMetadata;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;

final class CpuRunTraceContributor implements BackendRunTraceContributor {
    @Override
    public void contribute(BackendRunTraceContext context, LinkedHashMap<String, Object> attrs) {
        addMatMulAttrs(attrs, context.matMul());
        addNativeCpuRegionRejectionAttrs(attrs, context.node(), context.metadata(), context.executionContext());
        addNativeCpuRegionAttrs(attrs, context.metadata());
        addNativeCpuStateAttrs(attrs, context);
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
            attrs.put("openblasLookupSource", OpenBlasFfmBridge.lookupSource());
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
        if (metadata.cpuRegionExecutable() == null) {
            return;
        }
        var regionPlan = metadata.cpuRegionExecutable().regionExecutionPlan();
        if (regionPlan == null) {
            return;
        }
        BackendTraceSupport.addRegionPlanAttrs(attrs, regionPlan);
        attrs.put("nativeCpuRegionId", regionPlan.regionId());
        attrs.put("nativeCpuRegionNodeCount", regionPlan.orderedNodeIds().size());
        attrs.put("nativeCpuRegionInputs", regionPlan.externalInputNodeIds());
        attrs.put("nativeCpuRegionOutputs", regionPlan.boundaryOutputNodeIds());
        attrs.put("nativeCpuRegionRoute", metadata.cpuRegionExecutable().lastRoute());
        attrs.put("nativeCpuRegionDecision", regionPlan.decision().selected() ? "SELECTED" : "REJECTED");
        attrs.put("nativeCpuRegionReason", regionPlan.decision().reason());
        attrs.put("nativeCpuRegionFallbackReason", metadata.cpuRegionExecutable().lastFallbackReason());
        attrs.put("nativeCpuRegionLocalKernelCount", metadata.cpuRegionExecutable().lastRegionLocalKernelCount());
        attrs.put("nativeCpuRegionLocalViewCount", metadata.cpuRegionExecutable().lastRegionLocalViewCount());
        attrs.put("nativeCpuRegionExecutedGroupCount", metadata.cpuRegionExecutable().lastExecutedGroupCount());
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
            attrs.put("nativeCpuParityStoragePaths", regionPlan.nodePlans().stream()
                    .map(BackendTraceSupport::nativeCpuParityStoragePaths)
                    .toList());
            attrs.put("nativeCpuParityLayoutCapabilities", regionPlan.nodePlans().stream()
                    .map(BackendTraceSupport::nativeCpuParityLayoutCapabilities)
                    .toList());
            attrs.put("nativeCpuParityResultResidencies", regionPlan.nodePlans().stream()
                    .map(BackendTraceSupport::nativeCpuParityResultResidencies)
                    .toList());
            attrs.put("nativeCpuParityAutoEligible", regionPlan.nodePlans().stream()
                    .map(nodePlan -> NativeCpuParityMatrix.isAutoEligible(nodePlan.opType(), nodePlan.dataType()))
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
            attrs.put("nativeCpuLayoutClassCounts", BackendTraceSupport.stringCounts(regionPlan.nodePlans().stream()
                    .map(RegionNodePlan::layoutClass)
                    .toList()));
            attrs.put("nativeCpuStridedNodeCount", BackendTraceSupport.nativeCpuStridedNodeCount(regionPlan.nodePlans()));
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
                    .filter(BackendTraceSupport::isSegmentScalarNodePlan)
                    .map(RegionNodePlan::nodeId)
                    .toList());
            attrs.put("nativeCpuRegionBf16PromotedNodes", regionPlan.nodePlans().stream()
                    .filter(BackendTraceSupport::isBf16PromotedRegionNodePlan)
                    .map(RegionNodePlan::nodeId)
                    .toList());
            attrs.put("nativeCpuRegionBf16PromotedSegmentScalarNodes", regionPlan.nodePlans().stream()
                    .filter(BackendTraceSupport::isBf16PromotedRegionNodePlan)
                    .filter(BackendTraceSupport::isSegmentScalarNodePlan)
                    .map(RegionNodePlan::nodeId)
                    .toList());
            if (regionPlan.nodePlans().stream().anyMatch(BackendTraceSupport::isBf16PromotedRegionNodePlan)) {
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
            backend.runtime.ExecutionContext context
    ) {
        if (node == null || metadata == null || context == null || context.runtimeConfig() == null) {
            return;
        }
        RuntimeConfig runtimeConfig = context.runtimeConfig();
        if (metadata.backend() != ComputeBackend.CPU
                || metadata.cpuRegionExecutable() != null
                || metadata.acceleratorExecutable() != null
                || runtimeConfig.cpuStorageProfile() == CpuStorageProfile.CPU_ARRAY
                || metadata.cpuPlan() == null) {
            return;
        }
        PreparedNativeCpuPlan nativePlan = metadata.cpuPlan().nativeCpuPlan();
        if (nativePlan != null && nativePlan.route() == PreparedNativeCpuRoute.NATIVE_EXECUTABLE) {
            return;
        }
        String reason = nativeRegionRejectionReason(node, nativePlan, runtimeConfig);
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
        var parity = NativeCpuParityMatrix.entryFor(
                node.operation() == null ? operations.Operation.OpType.UNKNOWN : node.operation().opType(),
                node.dataType()
        );
        attrs.put("nativeCpuParityStoragePaths", List.of(parity.storagePaths().stream()
                .map(Enum::name)
                .sorted()
                .toList()));
        attrs.put("nativeCpuParityLayoutCapabilities", List.of(parity.layoutCapabilities().stream()
                .map(Enum::name)
                .sorted()
                .toList()));
        attrs.put("nativeCpuParityResultResidencies", List.of(parity.resultResidencies().stream()
                .map(Enum::name)
                .sorted()
                .toList()));
        attrs.put("nativeCpuParityAutoEligible", List.of(parity.autoEligible()));
        String layoutClass = BackendTraceSupport.nodeLayoutClassName(node);
        List<String> inputLayoutClasses = node.inputTensors().stream()
                .map(BackendTraceSupport::tensorLayoutClassName)
                .toList();
        ArrayList<String> rejectionLayoutClasses = new ArrayList<>();
        rejectionLayoutClasses.add(layoutClass);
        rejectionLayoutClasses.addAll(inputLayoutClasses);
        attrs.put("nativeCpuRegionLayoutClasses", List.of(layoutClass));
        attrs.put("nativeCpuRegionInputLayoutClasses", inputLayoutClasses);
        attrs.put("nativeCpuRegionOutputLayoutClasses", List.of(layoutClass));
        attrs.put("nativeCpuLayoutClassCounts", BackendTraceSupport.stringCounts(rejectionLayoutClasses));
        attrs.put("nativeCpuStridedNodeCount",
                BackendTraceSupport.isDenseOrViewLayout(layoutClass) && !reason.startsWith("native-layout-") ? 0 : 1);
        attrs.put("nativeCpuStridedMaterializationCount",
                reason.startsWith("native-layout-materialization-required:") ? 1 : 0);
        attrs.put("nativeCpuStridedFallbackReasons",
                reason.startsWith("native-layout-") ? List.of(reason) : List.of());
    }

    private static String nativeRegionRejectionReason(
            CompiledNode node,
            PreparedNativeCpuPlan nativePlan,
            RuntimeConfig runtimeConfig
    ) {
        String opLabel = node == null || node.operation() == null
                ? "unknown"
                : node.operation().opType().name().toLowerCase(Locale.ROOT);
        boolean providerOp = node != null
                && node.operation() != null
                && (node.operation().opType() == operations.Operation.OpType.MATMUL
                || node.operation().opType() == operations.Operation.OpType.LINEAR);
        String layoutReason = nativeLayoutRejectionReason(node, providerOp);
        if (!layoutReason.isBlank()) {
            return layoutReason;
        }
        if (providerOp && runtimeConfig.blas().provider() == backend.blas.BlasProvider.NONE) {
            return "native-cpu-region-provider-unavailable:" + opLabel;
        }
        String planReason = nativePlan == null ? "" : nativePlan.fallbackReason();
        if (runtimeConfig.cpuStorageProfile() == CpuStorageProfile.AUTO
                && (planReason.isBlank() || planReason.startsWith("cpu-storage-profile-not-native"))) {
            return "native-cpu-region-auto-rejected:no-region-selected";
        }
        if (!planReason.isBlank()) {
            return planReason.startsWith("native-cpu-region-")
                    ? planReason
                    : "native-cpu-region-rejected:" + planReason;
        }
        return "native-cpu-region-rejected:no-region-selected";
    }

    private static String nativeLayoutRejectionReason(CompiledNode node, boolean providerOp) {
        if (node == null || node.operation() == null) {
            return "";
        }
        String opLabel = node.operation().opType().name().toLowerCase(Locale.ROOT);
        String outputLayout = BackendTraceSupport.nodeLayoutClassName(node);
        if ("UNSUPPORTED_LAYOUT".equals(outputLayout)) {
            return "native-layout-unsupported:node-" + node.id();
        }
        for (Tensor input : node.inputTensors()) {
            String inputLayout = BackendTraceSupport.tensorLayoutClassName(input);
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
        if (!BackendTraceSupport.isDenseOrViewLayout(outputLayout)) {
            return "native-layout-unsupported:strided-output:" + opLabel;
        }
        return "";
    }

    private static void addNativeCpuStateAttrs(LinkedHashMap<String, Object> attrs, BackendRunTraceContext context) {
        Tensor runtimeTensor = safeRuntimeTensor(context.executionContext(), context.node().id());
        NativeCpuTraceState nativeCpu = runtimeTensor == null
                ? null
                : context.executionContext().runtimeStateFor(runtimeTensor, NativeCpuTraceState.class);
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

    private static Tensor safeRuntimeTensor(backend.runtime.ExecutionContext context, int nodeId) {
        try {
            return context.runtimeTensorForNodeId(nodeId);
        } catch (RuntimeException ignored) {
            return null;
        }
    }
}

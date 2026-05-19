package graph.execution;

import backend.metal.exec.PreparedMetalExecutable;
import graph.optimizer.cost.CostExplanation;

import java.util.LinkedHashMap;

final class MetalRunTraceContributor implements BackendRunTraceContributor {
    @Override
    public void contribute(BackendRunTraceContext context, LinkedHashMap<String, Object> attrs) {
        if (!(context.metadata().acceleratorExecutable() instanceof PreparedMetalExecutable metal)) {
            return;
        }
        var metalStats = metal.lastExecutionStats();
        var route = metal.routeDecision();
        CostExplanation routeCost = route.toCostScore().explain(route.reasonCode().name());
        attrs.put("metalBridgeAvailable", metal.bridge().isAvailable());
        attrs.put("metalBridgeContextAvailable", metal.bridgeContext().available());
        attrs.put("metalBridgeExecutableAvailable", metal.bridgeExecutable().available());
        attrs.put("metalBridgeCacheHit", metal.bridgeExecutable().cacheHit());
        attrs.put("metalSupportsBufferBindings", metal.bridge().supportsBufferBindings());
        attrs.put("metalExecutionRoute", route.selectedRoute().name());
        attrs.put("metalRouteReasonCode", route.reasonCode().name());
        attrs.put("metalRouteRejectedRoutes", route.rejectedRoutes().stream()
                .map(Enum::name)
                .toList());
        attrs.put("metalRouteRejectedReasonCodes", route.rejectedReasonCodes().stream()
                .map(Enum::name)
                .toList());
        attrs.put("metalRouteRejectedReasons", route.rejectedRouteReasons());
        attrs.put("metalRouteReason", route.detail());
        attrs.put("metalRouteEstimatedCost", route.estimatedRouteCost());
        attrs.put("metalRouteEstimatedCopyCost", route.estimatedCopyCost());
        attrs.put("metalRouteBridgeAvailable", route.bridgeAvailable());
        attrs.put("metalRouteExecutableAvailable", route.executableAvailable());
        attrs.put("metalRouteBufferAbiSupported", route.bufferAbiSupported());
        attrs.put("metalRouteCustomKernelAvailable", route.customKernelAvailable());
        attrs.put("metalRouteNativeCopyCostKnown", route.nativeCopyCostKnown());
        attrs.put("metalRouteCostModel", routeCost.modelName());
        attrs.put("metalRouteCostInputKind", routeCost.inputKind());
        attrs.put("metalRouteCostReason", routeCost.reasonCode());
        attrs.put("metalRouteCostComparison", routeCost.comparison().name());
        attrs.put("metalRouteCostTopContributors", routeCost.topContributors().stream()
                .map(BackendTraceSupport::costComponentSummary)
                .toList());
        attrs.put("metalRouteCostComponents", routeCost.rawComponents().stream()
                .map(BackendTraceSupport::costComponentSummary)
                .toList());
        attrs.put("metalBufferBindingDecision", metal.lastBufferBindingDecision());
        attrs.put("metalOutputBufferWriteProbeSupported", metal.bridge().supportsOutputBufferWriteProbe());
        attrs.put("metalSubgraphNodeCount", metal.plan().nodeIds().size());
        attrs.put("metalSubgraphOps", metal.plan().subgraph().ops().stream().map(op -> op.opType().name()).toList());
        attrs.put("metalEstimatedWork", metal.plan().estimatedWork());
        attrs.put("metalUsedCpuFallback", metalStats.usedCpuFallback());
        attrs.put("metalFallbackReason", metalStats.fallbackReason());
        attrs.put("metalExecutionPath", metalStats.executionPath().name());
        attrs.put("metalExternalInputCount", metalStats.externalInputCount());
        attrs.put("metalOutputCount", metalStats.outputCount());
        attrs.put("metalInputBytes", metalStats.inputBytes());
        attrs.put("metalOutputBytes", metalStats.outputBytes());
        attrs.put("metalJavaToNativeCopyNs", metalStats.javaToNativeCopyNs());
        attrs.put("metalOutputAllocationNs", metalStats.outputAllocationNs());
        attrs.put("metalNativeExecuteNs", metalStats.nativeExecuteNs());
        attrs.put("metalNativeCopyStrategy", metalStats.nativeCopyStrategy().name());
        attrs.put("metalOutputBufferWriteProven", metalStats.outputBufferWriteProven());
        attrs.put("metalOutputBufferWriteStatus", metalStats.outputBufferWriteStatus());
        attrs.put("metalNativeDeviceCopyNs", metalStats.nativeDeviceCopyNs());
        attrs.put("metalNativeToJavaCopyNs", metalStats.nativeToJavaCopyNs());
        attrs.put("metalBridgeTotalNs", metalStats.totalNs());
    }
}

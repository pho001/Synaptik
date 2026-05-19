package graph.execution.trace.contrib;

import java.util.LinkedHashMap;

final class StorageRunTraceContributor implements BackendRunTraceContributor {
    @Override
    public void contribute(BackendRunTraceContext context, LinkedHashMap<String, Object> attrs) {
        var residency = context.executionContext().residencyForNodeId(context.node().id());
        if (residency != null) {
            attrs.put("storageResidency", residency.residency().name());
            attrs.put("storageCpuCurrent", residency.cpuCurrent());
            attrs.put("storageDeviceCurrent", residency.deviceCurrent());
            attrs.put("storageDeviceBackend", residency.deviceBackend());
            attrs.put("storageTransitionReason", residency.lastTransitionReason());
        }
        var deviceBinding = context.executionContext().deviceBufferBindingForNodeId(context.node().id());
        if (deviceBinding != null) {
            attrs.put("deviceBufferBackend", deviceBinding.backendId());
            attrs.put("deviceBufferBytes", deviceBinding.logicalByteLength());
            attrs.put("deviceBufferAvailable", deviceBinding.available());
            attrs.put("deviceBuffer", deviceBinding.describe());
        }
    }
}

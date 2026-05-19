package graph.execution.residency;

import backend.memory.DeviceToCpuMaterializer;
import backend.memory.DeviceToNativeMaterializer;
import backend.memory.TensorResidencyState;
import graph.execution.trace.CpuMaterializationTrace;
import graph.execution.trace.HostDeviceTransferTrace;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Run-scoped tensor residency, materializers, and residency-related traces.
 */
public final class RuntimeResidencyStore {
    private final Map<Integer, TensorResidencyState> residencyByNodeId;
    private final Map<String, DeviceToCpuMaterializer> deviceToCpuMaterializerByBackend = new HashMap<>();
    private final Map<String, DeviceToNativeMaterializer> deviceToNativeMaterializerByBackend = new HashMap<>();
    private final List<CpuMaterializationTrace> cpuMaterializationTraces = new ArrayList<>();
    private final List<HostDeviceTransferTrace> hostDeviceTransferTraces = new ArrayList<>();

    public RuntimeResidencyStore(Map<Integer, TensorResidencyState> residencyByNodeId) {
        this.residencyByNodeId = Map.copyOf(residencyByNodeId);
    }

    public TensorResidencyState residencyForNodeId(int nodeId) {
        TensorResidencyState state = residencyByNodeId.get(nodeId);
        if (state == null) {
            throw new IllegalStateException("Missing runtime residency state for nodeId=" + nodeId);
        }
        return state;
    }

    public void registerDeviceToCpuMaterializer(String backendId, DeviceToCpuMaterializer materializer) {
        if (backendId == null || backendId.isBlank()) {
            throw new IllegalArgumentException("backendId cannot be blank");
        }
        deviceToCpuMaterializerByBackend.put(backendId, Objects.requireNonNull(materializer, "materializer cannot be null"));
    }

    public void registerDeviceToNativeMaterializer(String backendId, DeviceToNativeMaterializer materializer) {
        if (backendId == null || backendId.isBlank()) {
            throw new IllegalArgumentException("backendId cannot be blank");
        }
        deviceToNativeMaterializerByBackend.put(
                backendId,
                Objects.requireNonNull(materializer, "materializer cannot be null")
        );
    }

    public DeviceToCpuMaterializer deviceToCpuMaterializer(String backendId) {
        return deviceToCpuMaterializerByBackend.get(backendId);
    }

    public DeviceToNativeMaterializer deviceToNativeMaterializer(String backendId) {
        return deviceToNativeMaterializerByBackend.get(backendId);
    }

    public void recordCpuMaterialization(CpuMaterializationTrace trace) {
        cpuMaterializationTraces.add(Objects.requireNonNull(trace, "trace cannot be null"));
    }

    public void recordHostDeviceTransfer(HostDeviceTransferTrace trace) {
        hostDeviceTransferTraces.add(Objects.requireNonNull(trace, "trace cannot be null"));
    }

    public List<CpuMaterializationTrace> cpuMaterializationTraces() {
        return List.copyOf(cpuMaterializationTraces);
    }

    public List<HostDeviceTransferTrace> hostDeviceTransferTraces() {
        return List.copyOf(hostDeviceTransferTraces);
    }
}

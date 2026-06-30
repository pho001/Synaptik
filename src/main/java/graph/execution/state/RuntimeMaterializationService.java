package graph.execution.state;

import runtime.memory.nativecpu.NativeCpuMaterializer;
import runtime.contract.CpuMaterializationReason;
import runtime.memory.CpuMaterializationResult;
import runtime.device.buffer.DeviceBufferBinding;
import runtime.memory.DeviceToCpuMaterializer;
import runtime.memory.DeviceToNativeMaterializer;
import runtime.contract.StorageResidency;
import runtime.residency.TensorResidencyState;
import config.runtime.DeviceTransferPolicy;
import graph.execution.residency.DeviceBindingRegistry;
import graph.execution.residency.NativeCpuStorageRegistry;
import graph.execution.residency.RuntimeResidencyStore;
import trace.execution.CpuMaterializationTrace;
import runtime.contract.HostDeviceTransferKind;
import trace.execution.HostDeviceTransferTrace;
import tensor.DataType;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

import java.util.Objects;

/**
 * Run-scoped CPU/native/device materialization logic.
 */
public final class RuntimeMaterializationService {
    private final RuntimeTensorStore tensorStore;
    private final RuntimeResidencyStore residencyStore;
    private final DeviceBindingRegistry deviceBindingRegistry;
    private final NativeCpuStorageRegistry nativeStorageRegistry;
    private final RuntimeResourceRegistry resourceRegistry;

    public RuntimeMaterializationService(
            RuntimeTensorStore tensorStore,
            RuntimeResidencyStore residencyStore,
            DeviceBindingRegistry deviceBindingRegistry,
            NativeCpuStorageRegistry nativeStorageRegistry,
            RuntimeResourceRegistry resourceRegistry
    ) {
        this.tensorStore = Objects.requireNonNull(tensorStore, "tensorStore cannot be null");
        this.residencyStore = Objects.requireNonNull(residencyStore, "residencyStore cannot be null");
        this.deviceBindingRegistry = Objects.requireNonNull(deviceBindingRegistry, "deviceBindingRegistry cannot be null");
        this.nativeStorageRegistry = Objects.requireNonNull(nativeStorageRegistry, "nativeStorageRegistry cannot be null");
        this.resourceRegistry = Objects.requireNonNull(resourceRegistry, "resourceRegistry cannot be null");
    }

    public void markMaterializedToCpu(int nodeId, CpuMaterializationReason reason) {
        markMaterializedToCpu(nodeId, reason, 0L);
    }

    public void markMaterializedToCpu(int nodeId, CpuMaterializationReason reason, long durationNs) {
        markMaterializedToCpu(nodeId, reason, durationNs, "device value synchronized to CPU storage");
    }

    public void markMaterializedToCpu(
            int nodeId,
            CpuMaterializationReason reason,
            long durationNs,
            String detail
    ) {
        Objects.requireNonNull(reason, "reason cannot be null");
        TensorResidencyState state = residencyForNodeId(nodeId);
        String normalizedDetail = detail == null || detail.isBlank()
                ? "device value synchronized to CPU storage"
                : detail;
        residencyStore.recordCpuMaterialization(new CpuMaterializationTrace(
                nodeId,
                reason,
                state.deviceBackend(),
                state.residency(),
                logicalByteLength(nodeId),
                durationNs,
                true,
                normalizedDetail
        ));
        if (state.deviceCurrent()) {
            recordHostDeviceTransfer(new HostDeviceTransferTrace(
                    nodeId,
                    state.deviceBackend(),
                    runtimeTensorForNodeId(nodeId).getDataType().name(),
                    state.residency(),
                    StorageResidency.CPU_ARRAY,
                    HostDeviceTransferKind.DEVICE_TO_CPU_ARRAY_COPY,
                    logicalByteLength(nodeId),
                    logicalByteLength(nodeId),
                    0L,
                    logicalByteLength(nodeId),
                    durationNs,
                    false,
                    true,
                    true,
                    "",
                    detail == null || detail.isBlank() ? "device_to_cpu_array" : detail
            ));
        }
        deviceBindingRegistry.remove(nodeId);
        state.markMaterializedToCpu(reason.label());
    }

    public void recordHostDeviceTransfer(HostDeviceTransferTrace trace) {
        residencyStore.recordHostDeviceTransfer(trace);
    }

    public void requireCpuReadable(int nodeId, CpuMaterializationReason reason) {
        Objects.requireNonNull(reason, "reason cannot be null");
        TensorResidencyState state = residencyForNodeId(nodeId);
        if (state.requiresCpuMaterialization()) {
            tryMaterializeToCpu(nodeId, reason, state);
            return;
        }
        if (!state.cpuCurrent()) {
            residencyStore.recordCpuMaterialization(new CpuMaterializationTrace(
                    nodeId,
                    reason,
                    state.deviceBackend(),
                    state.residency(),
                    logicalByteLength(nodeId),
                    0L,
                    false,
                    "CPU storage is not current and no device representation is current"
            ));
            throw new IllegalStateException(
                    "CPU read requested for nodeId=" + nodeId
                            + " reason=" + reason.label()
                            + " but CPU storage is not current and no device representation is current."
            );
        }
    }

    public NativeTensorStorage requireNativeReadable(int nodeId, CpuMaterializationReason reason) {
        return requireNativeReadable(nodeId, reason, DeviceTransferPolicy.ALLOW_ARRAY_BRIDGE);
    }

    public NativeTensorStorage requireNativeReadable(
            int nodeId,
            CpuMaterializationReason reason,
            DeviceTransferPolicy deviceTransferPolicy
    ) {
        Objects.requireNonNull(reason, "reason cannot be null");
        DeviceTransferPolicy transferPolicy = deviceTransferPolicy == null
                ? DeviceTransferPolicy.ALLOW_ARRAY_BRIDGE
                : deviceTransferPolicy;
        TensorResidencyState state = residencyForNodeId(nodeId);
        if (state.nativeCurrent()) {
            NativeTensorStorage storage = nativeStorageRegistry.get(nodeId);
            if (storage == null) {
                throw new IllegalStateException("Native CPU residency is current for nodeId=" + nodeId
                        + " but no native storage is attached.");
            }
            storage.ensureOpen();
            return storage;
        }
        if (state.cpuCurrent()) {
            return materializeArrayToNative(nodeId, reason, "array_to_native");
        }
        if (state.deviceCurrent()) {
            NativeTensorStorage direct = tryMaterializeDeviceToNative(nodeId, reason, state);
            if (direct != null) {
                return direct;
            }
            if (!transferPolicy.allowsArrayBridge()) {
                throw new IllegalStateException(
                        "DeviceTransferPolicy.REQUIRE_DIRECT forbids Java array bridge for nodeId=" + nodeId
                                + " backend=" + state.deviceBackend()
                                + " sourceResidency=" + state.residency()
                                + " because direct device-to-native transfer is unavailable."
                );
            }
            StorageResidency sourceResidency = state.residency();
            String backend = state.deviceBackend();
            long bytes = logicalByteLength(nodeId);
            long start = System.nanoTime();
            requireCpuReadable(nodeId, reason);
            NativeTensorStorage storage = materializeArrayToNative(nodeId, reason, "device_to_array_to_native");
            recordHostDeviceTransfer(new HostDeviceTransferTrace(
                    nodeId,
                    backend,
                    runtimeTensorForNodeId(nodeId).getDataType().name(),
                    sourceResidency,
                    StorageResidency.CPU_NATIVE,
                    HostDeviceTransferKind.DEVICE_TO_ARRAY_TO_NATIVE_BRIDGE,
                    bytes,
                    bytes,
                    bytes,
                    bytes,
                    System.nanoTime() - start,
                    false,
                    false,
                    true,
                    "native-device-direct-transfer-unavailable",
                    "device_to_array_to_native"
            ));
            return storage;
        }
        throw new IllegalStateException(
                "Native CPU read requested for nodeId=" + nodeId
                        + " reason=" + reason.label()
                        + " but neither native, CPU array, nor device storage is current."
        );
    }

    public NativeTensorStorage materializeArrayToNative(int nodeId, CpuMaterializationReason reason, String detail) {
        Objects.requireNonNull(reason, "reason cannot be null");
        requireCpuReadable(nodeId, reason);
        Tensor tensor = runtimeTensorForNodeId(nodeId);
        NativeTensorStorage storage = nativeStorageRegistry.get(nodeId);
        if (storage == null || storage.closed() || storage.getType() != tensor.getDataType()
                || storage.getSize() != tensor.getFlatDataSize()) {
            storage = allocateNativeStorage(
                    tensor.getDataType(),
                    tensor.getFlatDataSize(),
                    "node-" + nodeId + ":" + tensor.getLabel()
            );
            nativeStorageRegistry.put(nodeId, storage);
            if (storage.ownsSegment()) {
                resourceRegistry.registerResource(storage.allocation());
            }
        }
        long start = System.nanoTime();
        NativeCpuMaterializer.arrayToNative(tensor, storage);
        long durationNs = System.nanoTime() - start;
        residencyStore.recordCpuMaterialization(new CpuMaterializationTrace(
                nodeId,
                reason,
                "",
                StorageResidency.CPU_ARRAY,
                logicalByteLength(nodeId),
                durationNs,
                true,
                detail == null || detail.isBlank() ? "array_to_native" : detail
        ));
        deviceBindingRegistry.remove(nodeId);
        residencyForNodeId(nodeId).markNativeCurrent(reason.label());
        return storage;
    }

    private NativeTensorStorage tryMaterializeDeviceToNative(
            int nodeId,
            CpuMaterializationReason reason,
            TensorResidencyState state
    ) {
        DeviceBufferBinding binding = deviceBindingRegistry.active(nodeId);
        DeviceToNativeMaterializer materializer = binding == null
                ? null
                : residencyStore.deviceToNativeMaterializer(binding.backendId());
        if (binding == null || materializer == null) {
            return null;
        }
        Tensor tensor = runtimeTensorForNodeId(nodeId);
        NativeTensorStorage storage = nativeStorageRegistry.get(nodeId);
        if (storage == null || storage.closed() || storage.getType() != tensor.getDataType()
                || storage.getSize() != tensor.getFlatDataSize()) {
            storage = allocateNativeStorage(
                    tensor.getDataType(),
                    tensor.getFlatDataSize(),
                    "node-" + nodeId + ":" + tensor.getLabel()
            );
            nativeStorageRegistry.put(nodeId, storage);
            if (storage.ownsSegment()) {
                resourceRegistry.registerResource(storage.allocation());
            }
        }
        if (!materializer.supports(binding, tensor, storage, reason)) {
            return null;
        }
        CpuMaterializationResult result = materializer.materialize(binding, tensor, storage, reason);
        if (result == null) {
            result = CpuMaterializationResult.unmeasured("device value synchronized to native CPU storage");
        }
        long bytes = logicalByteLength(nodeId);
        recordHostDeviceTransfer(new HostDeviceTransferTrace(
                nodeId,
                state.deviceBackend(),
                tensor.getDataType().name(),
                state.residency(),
                StorageResidency.CPU_NATIVE,
                HostDeviceTransferKind.DEVICE_TO_NATIVE_SEGMENT_COPY,
                bytes,
                0L,
                bytes,
                bytes,
                result.durationNs(),
                false,
                true,
                true,
                "",
                result.detail()
        ));
        deviceBindingRegistry.remove(nodeId);
        residencyForNodeId(nodeId).markNativeCurrent(reason.label());
        return storage;
    }

    private void tryMaterializeToCpu(int nodeId, CpuMaterializationReason reason, TensorResidencyState state) {
        if (state.nativeCurrent()) {
            NativeTensorStorage storage = nativeStorageRegistry.get(nodeId);
            if (storage != null) {
                long start = System.nanoTime();
                NativeCpuMaterializer.nativeToArray(storage, runtimeTensorForNodeId(nodeId));
                markMaterializedToCpu(
                        nodeId,
                        reason,
                        System.nanoTime() - start,
                        storage.getType() == DataType.BOOL
                                ? "bool_mask_published:native_to_array"
                                : "native_to_array"
                );
                return;
            }
            residencyStore.recordCpuMaterialization(new CpuMaterializationTrace(
                    nodeId,
                    reason,
                    "",
                    state.residency(),
                    logicalByteLength(nodeId),
                    0L,
                    false,
                    "native CPU storage is current but no native storage binding is available"
            ));
            throw new IllegalStateException(
                    "CPU materialization requested for nodeId=" + nodeId
                            + " reason=" + reason.label()
                            + " but native CPU storage is current and no native storage binding is available."
            );
        }
        DeviceBufferBinding binding = deviceBindingRegistry.active(nodeId);
        DeviceToCpuMaterializer materializer = binding == null
                ? null
                : residencyStore.deviceToCpuMaterializer(binding.backendId());
        if (binding != null
                && materializer != null
                && materializer.supports(binding, runtimeTensorForNodeId(nodeId), reason)) {
            CpuMaterializationResult result = materializer.materialize(binding, runtimeTensorForNodeId(nodeId), reason);
            if (result == null) {
                result = CpuMaterializationResult.unmeasured("device value synchronized to CPU storage");
            }
            markMaterializedToCpu(nodeId, reason, result.durationNs(), result.detail());
            return;
        }
        String detail = materializationFailureDetail(binding, materializer);
        residencyStore.recordCpuMaterialization(new CpuMaterializationTrace(
                nodeId,
                reason,
                state.deviceBackend(),
                state.residency(),
                logicalByteLength(nodeId),
                0L,
                false,
                detail
        ));
        throw new IllegalStateException(
                "CPU materialization requested for nodeId=" + nodeId
                        + " reason=" + reason.label()
                        + " but " + detail
                        + " for backend=" + state.deviceBackend()
                        + ", residency=" + state.residency()
                        + ". This prevents publishing stale CPU tensor storage."
        );
    }

    private NativeTensorStorage allocateNativeStorage(DataType dataType, int elements, String label) {
        return resourceRegistry.allocateNativeStorage(dataType, elements, label);
    }

    private Tensor runtimeTensorForNodeId(int nodeId) {
        return tensorStore.runtimeTensorForNodeId(nodeId);
    }

    private TensorResidencyState residencyForNodeId(int nodeId) {
        return residencyStore.residencyForNodeId(nodeId);
    }

    private long logicalByteLength(int nodeId) {
        return tensorStore.logicalByteLength(nodeId);
    }

    private static String materializationFailureDetail(
            DeviceBufferBinding binding,
            DeviceToCpuMaterializer materializer
    ) {
        if (binding == null) {
            return "no device buffer binding is available";
        }
        if (materializer == null) {
            return "no device-to-CPU materializer is available";
        }
        return "registered device-to-CPU materializer does not support the active binding";
    }
}

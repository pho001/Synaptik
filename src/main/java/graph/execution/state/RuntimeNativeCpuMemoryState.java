package graph.execution.state;

import runtime.memory.nativecpu.NativeCpuMemoryPool;
import runtime.contract.CpuMaterializationReason;
import runtime.residency.TensorResidencyState;
import config.runtime.NativeCpuMemoryConfig;
import graph.execution.residency.DeviceBindingRegistry;
import graph.execution.residency.NativeCpuStorageRegistry;
import graph.execution.residency.RuntimeResidencyStore;
import tensor.DataType;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

import java.util.Objects;

/**
 * Run-scoped native CPU memory state and invariants.
 */
public final class RuntimeNativeCpuMemoryState {
    private final RuntimeTensorStore tensorStore;
    private final RuntimeResidencyStore residencyStore;
    private final NativeCpuStorageRegistry nativeStorageRegistry;
    private final DeviceBindingRegistry deviceBindingRegistry;
    private final RuntimeResourceRegistry resourceRegistry;
    private final RuntimeMaterializationService materializationService;

    public RuntimeNativeCpuMemoryState(
            RuntimeTensorStore tensorStore,
            RuntimeResidencyStore residencyStore,
            NativeCpuStorageRegistry nativeStorageRegistry,
            DeviceBindingRegistry deviceBindingRegistry,
            RuntimeResourceRegistry resourceRegistry,
            RuntimeMaterializationService materializationService
    ) {
        this.tensorStore = Objects.requireNonNull(tensorStore, "tensorStore cannot be null");
        this.residencyStore = Objects.requireNonNull(residencyStore, "residencyStore cannot be null");
        this.nativeStorageRegistry = Objects.requireNonNull(nativeStorageRegistry, "nativeStorageRegistry cannot be null");
        this.deviceBindingRegistry = Objects.requireNonNull(deviceBindingRegistry, "deviceBindingRegistry cannot be null");
        this.resourceRegistry = Objects.requireNonNull(resourceRegistry, "resourceRegistry cannot be null");
        this.materializationService = Objects.requireNonNull(materializationService, "materializationService cannot be null");
    }

    public void configure(NativeCpuMemoryConfig config, NativeCpuMemoryPool preparedPool) {
        resourceRegistry.configureNativeCpuMemory(config, preparedPool);
    }

    public void attachNativeStorage(int nodeId, NativeTensorStorage storage, String reason) {
        Objects.requireNonNull(storage, "storage cannot be null");
        Tensor tensor = tensorStore.runtimeTensorForNodeId(nodeId);
        if (tensor.getDataType() != storage.getType()) {
            throw new IllegalArgumentException("Native storage dtype mismatch for nodeId=" + nodeId
                    + ". tensorType=" + tensor.getDataType() + ", storageType=" + storage.getType());
        }
        if (tensor.getFlatDataSize() != storage.getSize()) {
            throw new IllegalArgumentException("Native storage size mismatch for nodeId=" + nodeId
                    + ". tensorElements=" + tensor.getFlatDataSize() + ", storageElements=" + storage.getSize());
        }
        nativeStorageRegistry.put(nodeId, storage);
        if (storage.ownsSegment()) {
            resourceRegistry.registerResource(storage.allocation());
        }
        deviceBindingRegistry.remove(nodeId);
        residencyForNodeId(nodeId).markNativeCurrent(reason);
    }

    public void reserveNativeOutputStorage(int nodeId, NativeTensorStorage storage) {
        Objects.requireNonNull(storage, "storage cannot be null");
        Tensor tensor = tensorStore.runtimeTensorForNodeId(nodeId);
        if (tensor.getDataType() != storage.getType()) {
            throw new IllegalArgumentException("Native output storage dtype mismatch for nodeId=" + nodeId
                    + ". tensorType=" + tensor.getDataType() + ", storageType=" + storage.getType());
        }
        if (tensor.getFlatDataSize() != storage.getSize()) {
            throw new IllegalArgumentException("Native output storage size mismatch for nodeId=" + nodeId
                    + ". tensorElements=" + tensor.getFlatDataSize() + ", storageElements=" + storage.getSize());
        }
        storage.ensureOpen();
        nativeStorageRegistry.put(nodeId, storage);
        if (storage.ownsSegment()) {
            resourceRegistry.registerResource(storage.allocation());
        }
        deviceBindingRegistry.remove(nodeId);
        residencyForNodeId(nodeId).markNativeOutputReserved("native output storage reserved");
    }

    public NativeTensorStorage allocateNativeStorage(DataType dataType, int elements, String label) {
        return resourceRegistry.allocateNativeStorage(dataType, elements, label);
    }

    public void aliasNativeStorage(int targetNodeId, int sourceNodeId, String reason) {
        Tensor target = tensorStore.runtimeTensorForNodeId(targetNodeId);
        Tensor source = tensorStore.runtimeTensorForNodeId(sourceNodeId);
        if (target.getDataType() != source.getDataType()) {
            throw new IllegalArgumentException("Native view alias dtype mismatch. targetNodeId=" + targetNodeId
                    + ", sourceNodeId=" + sourceNodeId + ", targetType=" + target.getDataType()
                    + ", sourceType=" + source.getDataType());
        }
        NativeTensorStorage storage = materializationService.requireNativeReadable(
                sourceNodeId,
                CpuMaterializationReason.CPU_CONSUMER
        );
        long requiredElements = viewPhysicalElementSpan(target);
        if (requiredElements > storage.getSize()) {
            throw new IllegalArgumentException("Native view alias exceeds source storage. targetNodeId=" + targetNodeId
                    + ", sourceNodeId=" + sourceNodeId + ", requiredElements=" + requiredElements
                    + ", storageElements=" + storage.getSize());
        }
        nativeStorageRegistry.put(targetNodeId, storage);
        deviceBindingRegistry.remove(targetNodeId);
        residencyForNodeId(targetNodeId).markNativeCurrent(reason);
    }

    public NativeTensorStorage nativeStorageForNodeId(int nodeId) {
        residencyForNodeId(nodeId);
        return nativeStorageRegistry.get(nodeId);
    }

    private TensorResidencyState residencyForNodeId(int nodeId) {
        return residencyStore.residencyForNodeId(nodeId);
    }

    private static long viewPhysicalElementSpan(Tensor tensor) {
        int[] shape = tensor.getShapeUnsafe();
        int[] strides = tensor.getStridesUnsafe();
        long maxElementOffset = tensor.getStorageOffsetUnsafe();
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] < 0 || strides[i] < 0) {
                throw new IllegalArgumentException("Native view alias supports only non-negative shape/strides.");
            }
            if (shape[i] == 0) {
                return 0L;
            }
            maxElementOffset = Math.addExact(
                    maxElementOffset,
                    Math.multiplyExact((long) shape[i] - 1L, strides[i])
            );
        }
        return Math.addExact(maxElementOffset, 1L);
    }
}

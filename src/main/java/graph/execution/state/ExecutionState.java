package graph.execution.state;

import backend.cpu.nativecpu.NativeCpuMemoryPool;
import runtime.contract.CpuMaterializationReason;
import backend.memory.DeviceBufferBinding;
import backend.memory.DeviceToCpuMaterializer;
import backend.memory.DeviceToNativeMaterializer;
import backend.memory.ExecutionResource;
import runtime.contract.StorageResidency;
import backend.memory.TensorResidencyState;
import config.runtime.DeviceTransferPolicy;
import config.runtime.NativeCpuMemoryConfig;
import graph.model.CompiledNode;
import planning.descriptor.CompiledTensorDescriptorIndex;
import graph.compile.publication.PublicationPlan;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.residency.DeviceBindingRegistry;
import graph.execution.residency.NativeCpuStorageRegistry;
import graph.execution.residency.RuntimeResidencyStore;
import trace.execution.CpuMaterializationTrace;
import trace.execution.HostDeviceTransferTrace;
import trace.execution.NativeCpuMemoryTrace;
import tensor.DataType;
import tensor.storage.NativeTensorStorage;
import tensor.Tensor;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Per-run mutable execution state.
 *
 * <p>Prepared programs keep immutable compile/prepare metadata. Every execute call materializes its own
 * runtime tensor bindings and workspaces here so runs do not share mutable graph state.
 */
public final class ExecutionState {
    private final RuntimeTensorStore tensorStore;
    private final RuntimeWorkspaceStore workspaceStore;
    private final RuntimeResidencyStore residencyStore;
    private final DeviceBindingRegistry deviceBindingRegistry;
    private final NativeCpuStorageRegistry nativeStorageRegistry;
    private final RuntimeResourceRegistry resourceRegistry;
    private final RuntimeStorageSlotCache storageSlotCache;
    private final Map<Integer, RuntimeStorageSlotKey> runtimeStorageSlotByNodeId = new HashMap<>();
    private final RuntimeMaterializationService materializationService;
    private final RuntimeNativeCpuMemoryState nativeCpuMemoryState;
    private final RuntimeDeviceMemoryState deviceMemoryState;

    private ExecutionState(
            RuntimeTensorStore tensorStore,
            RuntimeWorkspaceStore workspaceStore,
            RuntimeResidencyStore residencyStore,
            DeviceBindingRegistry deviceBindingRegistry,
            NativeCpuStorageRegistry nativeStorageRegistry,
            RuntimeResourceRegistry resourceRegistry
    ) {
        this.tensorStore = Objects.requireNonNull(tensorStore, "tensorStore cannot be null");
        this.workspaceStore = Objects.requireNonNull(workspaceStore, "workspaceStore cannot be null");
        this.residencyStore = Objects.requireNonNull(residencyStore, "residencyStore cannot be null");
        this.deviceBindingRegistry = Objects.requireNonNull(deviceBindingRegistry, "deviceBindingRegistry cannot be null");
        this.nativeStorageRegistry = Objects.requireNonNull(nativeStorageRegistry, "nativeStorageRegistry cannot be null");
        this.resourceRegistry = Objects.requireNonNull(resourceRegistry, "resourceRegistry cannot be null");
        this.storageSlotCache = new RuntimeStorageSlotCache(this.resourceRegistry);
        this.materializationService = new RuntimeMaterializationService(
                this.tensorStore,
                this.residencyStore,
                this.deviceBindingRegistry,
                this.nativeStorageRegistry,
                this.resourceRegistry
        );
        this.nativeCpuMemoryState = new RuntimeNativeCpuMemoryState(
                this.tensorStore,
                this.residencyStore,
                this.nativeStorageRegistry,
                this.deviceBindingRegistry,
                this.resourceRegistry,
                this.materializationService
        );
        this.deviceMemoryState = new RuntimeDeviceMemoryState(
                this.residencyStore,
                this.deviceBindingRegistry,
                this.nativeStorageRegistry
        );
    }

    /**
     * Configures the run-owned native CPU allocator before execution starts.
     *
     * @param config native CPU memory policy; {@code null} disables pooling
     */
    public void configureNativeCpuMemory(NativeCpuMemoryConfig config) {
        configureNativeCpuMemory(config, null);
    }

    /**
     * Configures the run-owned native CPU allocator before execution starts.
     *
     * @param config native CPU memory policy; {@code null} disables pooling
     * @param preparedPool shared prepared-execution pool for {@code PER_PREPARED_EXECUTION}
     */
    public void configureNativeCpuMemory(NativeCpuMemoryConfig config, NativeCpuMemoryPool preparedPool) {
        nativeCpuMemoryState.configure(config, preparedPool);
    }

    /**
     * Creates per-run runtime tensors, prepared input buffers, and CPU workspaces.
     *
     * @param compiledNodes compiled node snapshots in graph order
     * @param descriptorIndex immutable tensor descriptor facts for {@code compiledNodes}
     * @param metadataIndex prepared execution metadata keyed by node id
     * @param forwardBoundaryNodeId last forward node id, used to decide leaf aliasing versus copying
     * @param publicationPlan runtime input and publication bindings captured at compile time
     * @return mutable execution state for one run
     */
    public static ExecutionState create(
            List<CompiledNode> compiledNodes,
            CompiledTensorDescriptorIndex descriptorIndex,
            Map<Integer, CompiledNodeExecutionMetadata> metadataIndex,
            int forwardBoundaryNodeId,
            PublicationPlan publicationPlan
    ) {
        Objects.requireNonNull(compiledNodes, "compiledNodes cannot be null");
        Objects.requireNonNull(descriptorIndex, "descriptorIndex cannot be null");
        Objects.requireNonNull(metadataIndex, "metadataIndex cannot be null");
        Objects.requireNonNull(publicationPlan, "publicationPlan cannot be null");

        Map<Integer, TensorResidencyState> residency = new HashMap<>(compiledNodes.size());
        RuntimeTensorStore tensorStore = RuntimeTensorStore.create(
                compiledNodes,
                descriptorIndex,
                forwardBoundaryNodeId,
                residency,
                publicationPlan
        );
        return new ExecutionState(
                tensorStore,
                RuntimeWorkspaceStore.create(metadataIndex),
                new RuntimeResidencyStore(residency),
                new DeviceBindingRegistry(Map.of()),
                new NativeCpuStorageRegistry(),
                new RuntimeResourceRegistry()
        );
    }

    /**
     * Returns the runtime tensor for a compiled node.
     *
     * @param nodeId compiled node id
     * @return runtime tensor
     */
    public Tensor runtimeTensorForNodeId(int nodeId) {
        return tensorStore.runtimeTensorForNodeId(nodeId);
    }

    /**
     * Returns the backend-owned runtime workspace for a compiled node.
     *
     * @param nodeId compiled node id
     * @return runtime workspace, or {@code null} when the node does not use one
     */
    public Object workspaceForNodeId(int nodeId) {
        return workspaceStore.workspaceForNodeId(nodeId);
    }

    /**
     * Returns a prepared runtime input tensor for a node input.
     *
     * @param nodeId compiled node id
     * @param inputIndex input index
     * @return prepared runtime tensor
     */
    public Tensor preparedInputTensorFor(int nodeId, int inputIndex) {
        return workspaceStore.preparedInputTensorFor(nodeId, inputIndex);
    }

    /**
     * Looks up the compiled node id for a runtime tensor.
     *
     * @param tensor runtime tensor
     * @return node id, or {@code null} when the tensor is unknown
     */
    public Integer nodeIdForRuntimeTensor(Tensor tensor) {
        return tensorStore.nodeIdForRuntimeTensor(tensor);
    }

    /**
     * Returns runtime residency state for a compiled node.
     *
     * @param nodeId compiled node id
     * @return mutable residency state for the runtime tensor
     */
    public TensorResidencyState residencyForNodeId(int nodeId) {
        return residencyStore.residencyForNodeId(nodeId);
    }

    /**
     * Marks a node output as current in CPU array storage.
     *
     * @param nodeId compiled node id
     * @param reason diagnostic transition reason
     */
    public void markCpuCurrent(int nodeId, String reason) {
        nativeStorageRegistry.remove(nodeId);
        deviceBindingRegistry.remove(nodeId);
        residencyForNodeId(nodeId).markCpuCurrent(reason);
    }

    /**
     * Attaches native CPU storage as the current representation for a runtime tensor.
     *
     * @param nodeId compiled node id
     * @param storage native CPU storage containing the current value
     * @param reason diagnostic transition reason
     */
    public void attachNativeStorage(int nodeId, NativeTensorStorage storage, String reason) {
        nativeCpuMemoryState.attachNativeStorage(nodeId, storage, reason);
    }

    /**
     * Reserves native CPU storage for a future output write and marks the output value stale.
     *
     * <p>Backends must call {@link #attachNativeStorage(int, NativeTensorStorage, String)} only after they
     * have successfully written the reserved storage.</p>
     *
     * @param nodeId compiled node id
     * @param storage native CPU storage available for an upcoming write
     */
    public void reserveNativeOutputStorage(int nodeId, NativeTensorStorage storage) {
        nativeCpuMemoryState.reserveNativeOutputStorage(nodeId, storage);
    }

    /**
     * Registers a planned region runtime storage slot for a node output.
     *
     * @param nodeId compiled node id
     * @param dataType storage dtype
     * @param slotId memory-plan region slot id
     * @param elements region slot size in elements
     * @return Java-array slot key for the registered region slot
     */
    public RuntimeStorageSlotKey registerRegionRuntimeStorageSlot(
            int nodeId,
            DataType dataType,
            int slotId,
            int elements
    ) {
        Tensor tensor = runtimeTensorForNodeId(nodeId);
        if (tensor.getDataType() != dataType) {
            throw new IllegalArgumentException("Runtime slot dtype mismatch for nodeId=" + nodeId
                    + ". tensorType=" + tensor.getDataType() + ", slotType=" + dataType);
        }
        if (tensor.getFlatDataSize() != elements) {
            throw new IllegalArgumentException("Runtime slot size mismatch for nodeId=" + nodeId
                    + ". tensorElements=" + tensor.getFlatDataSize() + ", slotElements=" + elements);
        }
        RuntimeStorageSlotKey key = RuntimeStorageSlotKey.regionSlot(
                RuntimeStorageKind.JAVA_ARRAY,
                dataType,
                slotId,
                elements
        );
        runtimeStorageSlotByNodeId.put(nodeId, key);
        return key;
    }

    /**
     * Binds a runtime tensor to Java array storage from the slot cache.
     *
     * @param nodeId compiled node id
     * @param key Java-array storage slot key
     */
    public void bindJavaStorageSlot(int nodeId, RuntimeStorageSlotKey key) {
        storageSlotCache.bindJavaStorage(runtimeTensorForNodeId(nodeId), key);
    }

    /**
     * Returns the planned runtime storage slot registered for a node.
     *
     * @param nodeId compiled node id
     * @return registered Java-array region slot key, or {@code null}
     */
    public RuntimeStorageSlotKey runtimeStorageSlotKeyForNodeId(int nodeId) {
        return runtimeStorageSlotByNodeId.get(nodeId);
    }

    /**
     * Returns writable native CPU output storage for a node and reserves it without marking it current.
     *
     * <p>When the memory plan registered a region slot for the node, the native output reuses that slot id.
     * Otherwise the storage is scoped to the node output.</p>
     *
     * @param nodeId compiled node id
     * @param dataType tensor dtype
     * @param elements number of logical elements
     * @param label diagnostic allocation label
     * @return writable native CPU tensor storage
     */
    public NativeTensorStorage requireNativeOutputStorage(
            int nodeId,
            DataType dataType,
            int elements,
            String label
    ) {
        Tensor tensor = runtimeTensorForNodeId(nodeId);
        if (tensor.getDataType() != dataType) {
            throw new IllegalArgumentException("Native output dtype mismatch for nodeId=" + nodeId
                    + ". tensorType=" + tensor.getDataType() + ", requestedType=" + dataType);
        }
        if (tensor.getFlatDataSize() != elements) {
            throw new IllegalArgumentException("Native output size mismatch for nodeId=" + nodeId
                    + ". tensorElements=" + tensor.getFlatDataSize() + ", requestedElements=" + elements);
        }
        RuntimeStorageSlotKey key = nativeOutputSlotKey(nodeId, dataType, elements);
        NativeTensorStorage storage = storageSlotCache.nativeCpuStorage(key, label);
        reserveNativeOutputStorage(nodeId, storage);
        return storage;
    }

    /**
     * Allocates run-owned native CPU tensor storage through this execution state's allocator.
     *
     * @param dataType tensor dtype
     * @param elements number of logical elements
     * @param label diagnostic allocation label
     * @return native tensor storage
     */
    public NativeTensorStorage allocateNativeStorage(DataType dataType, int elements, String label) {
        return nativeCpuMemoryState.allocateNativeStorage(dataType, elements, label);
    }

    private RuntimeStorageSlotKey nativeOutputSlotKey(int nodeId, DataType dataType, int elements) {
        RuntimeStorageSlotKey regionKey = runtimeStorageSlotByNodeId.get(nodeId);
        if (regionKey == null) {
            return RuntimeStorageSlotKey.nodeOutput(RuntimeStorageKind.NATIVE_CPU, dataType, nodeId, elements);
        }
        if (regionKey.dataType() != dataType || regionKey.elements() != elements) {
            throw new IllegalStateException("Registered runtime slot does not match native output request for nodeId="
                    + nodeId + ". slot=" + regionKey + ", requested=" + dataType + "[" + elements + "]");
        }
        return regionKey.withKind(RuntimeStorageKind.NATIVE_CPU);
    }

    /**
     * Attaches the current native CPU storage from one node to another view-only node without allocating.
     *
     * <p>The source native storage must already be current and registered with this execution state. The
     * target node gets a runtime alias to the same native storage, while its CPU array representation remains
     * stale until an explicit CPU materialization request writes into the target runtime tensor.</p>
     *
     * @param targetNodeId view-only target node id
     * @param sourceNodeId source node id whose native storage is current
     * @param reason diagnostic transition reason
     */
    public void aliasNativeStorage(int targetNodeId, int sourceNodeId, String reason) {
        nativeCpuMemoryState.aliasNativeStorage(targetNodeId, sourceNodeId, reason);
    }

    /**
     * Returns native CPU storage attached to a node.
     *
     * @param nodeId compiled node id
     * @return native storage, or {@code null} when none is attached
     */
    public NativeTensorStorage nativeStorageForNodeId(int nodeId) {
        return nativeCpuMemoryState.nativeStorageForNodeId(nodeId);
    }

    /**
     * Marks a node output as current only in a device-visible representation.
     *
     * <p>This method records state only; it does not allocate or populate a device buffer. Backends
     * should call it after they have actually written the value to the corresponding device storage.</p>
     *
     * @param nodeId compiled node id
     * @param residency device residency class; must not be {@link StorageResidency#CPU_ARRAY}
     * @param deviceBackend backend id such as {@code GPU_METAL}
     * @param reason diagnostic transition reason
     */
    public void markDeviceCurrent(int nodeId, StorageResidency residency, String deviceBackend, String reason) {
        deviceMemoryState.markDeviceCurrent(nodeId, residency, deviceBackend, reason);
    }

    /**
     * Reserves a writable device buffer for a future backend output without marking the value current.
     *
     * <p>This method is for output buffers that have been allocated but not written yet. Unlike
     * {@link #attachDeviceBufferBinding(int, DeviceBufferBinding, StorageResidency, String)}, it does not
     * change residency or CPU/device current flags. A backend must promote the reservation with
     * {@code attachDeviceBufferBinding(...)} only after it has written the output bytes.</p>
     *
     * @param nodeId compiled node id
     * @param binding backend-specific buffer binding
     */
    public void reserveDeviceBufferBinding(int nodeId, DeviceBufferBinding binding) {
        deviceMemoryState.reserveDeviceBufferBinding(nodeId, binding);
    }

    /**
     * Registers a usable device buffer binding for a runtime tensor and updates residency.
     *
     * <p>For {@link StorageResidency#HOST_SHARED_DEVICE_BUFFER}, both CPU and device representations
     * are marked current. For {@link StorageResidency#DEVICE_OWNED}, CPU storage is marked stale and
     * device storage is marked current. This method records metadata only; the caller must already
     * have created and populated the backend buffer.</p>
     *
     * @param nodeId compiled node id
     * @param binding backend-specific buffer binding
     * @param residency device residency class
     * @param reason diagnostic transition reason
     */
    public void attachDeviceBufferBinding(
            int nodeId,
            DeviceBufferBinding binding,
            StorageResidency residency,
            String reason
    ) {
        deviceMemoryState.attachDeviceBufferBinding(nodeId, binding, residency, reason);
    }

    /**
     * Returns the registered device buffer binding for a runtime tensor.
     *
     * @param nodeId compiled node id
     * @return device buffer binding, or {@code null} when none is registered
     */
    public DeviceBufferBinding deviceBufferBindingForNodeId(int nodeId) {
        return deviceMemoryState.deviceBufferBindingForNodeId(nodeId);
    }

    /**
     * Returns a writable device buffer binding for a future backend output.
     *
     * <p>The returned binding may be either an already active binding or a reserved output binding. Callers
     * must not treat a reserved binding as current data until the backend has written it and promoted it via
     * {@link #attachDeviceBufferBinding(int, DeviceBufferBinding, StorageResidency, String)}.</p>
     *
     * @param nodeId compiled node id
     * @return writable binding, or {@code null} when none is available
     */
    public DeviceBufferBinding writableDeviceBufferBindingForNodeId(int nodeId) {
        return deviceMemoryState.writableDeviceBufferBindingForNodeId(nodeId);
    }

    /**
     * Marks a completed device-to-CPU synchronization.
     *
     * <p>This method must only be called after the CPU array storage has been updated with the current
     * value. It intentionally does not perform the copy itself.</p>
     *
     * @param nodeId compiled node id
     * @param reason reason that forced CPU materialization
     */
    public void markMaterializedToCpu(int nodeId, CpuMaterializationReason reason) {
        materializationService.markMaterializedToCpu(nodeId, reason);
    }

    /**
     * Marks a completed device-to-CPU synchronization with measured duration.
     *
     * <p>This method must only be called after the CPU array storage has been updated with the current
     * value. It intentionally does not perform the copy itself.</p>
     *
     * @param nodeId compiled node id
     * @param reason reason that forced CPU materialization
     * @param durationNs measured materialization duration
     */
    public void markMaterializedToCpu(int nodeId, CpuMaterializationReason reason, long durationNs) {
        materializationService.markMaterializedToCpu(nodeId, reason, durationNs);
    }

    /**
     * Marks a completed device-to-CPU synchronization with measured duration and trace detail.
     *
     * <p>This method must only be called after the CPU array storage has been updated with the current
     * value. It intentionally does not perform the copy itself.</p>
     *
     * @param nodeId compiled node id
     * @param reason reason that forced CPU materialization
     * @param durationNs measured materialization duration
     * @param detail diagnostic trace detail
     */
    public void markMaterializedToCpu(
            int nodeId,
            CpuMaterializationReason reason,
            long durationNs,
            String detail
    ) {
        materializationService.markMaterializedToCpu(nodeId, reason, durationNs, detail);
    }

    /**
     * Records a host/device transfer trace entry.
     *
     * @param trace transfer trace entry
     */
    public void recordHostDeviceTransfer(HostDeviceTransferTrace trace) {
        materializationService.recordHostDeviceTransfer(trace);
    }

    /**
     * Registers or replaces the materializer used for one device backend during this run.
     *
     * <p>The registration is intentionally per execution state. It avoids a hidden global singleton and lets
     * prepared/runtime code decide which backend object owns synchronization for a particular run.</p>
     *
     * @param backendId backend id such as {@code GPU_METAL}
     * @param materializer materializer implementation
     */
    public void registerDeviceToCpuMaterializer(String backendId, DeviceToCpuMaterializer materializer) {
        if (backendId == null || backendId.isBlank()) {
            throw new IllegalArgumentException("backendId cannot be blank");
        }
        residencyStore.registerDeviceToCpuMaterializer(backendId, materializer);
    }

    /**
     * Registers or replaces the materializer used to read one device backend directly into native CPU storage.
     *
     * @param backendId backend id such as {@code GPU_METAL}
     * @param materializer materializer implementation
     */
    public void registerDeviceToNativeMaterializer(String backendId, DeviceToNativeMaterializer materializer) {
        if (backendId == null || backendId.isBlank()) {
            throw new IllegalArgumentException("backendId cannot be blank");
        }
        residencyStore.registerDeviceToNativeMaterializer(backendId, materializer);
    }

    /**
     * Registers a native/backend resource owned by this execution run.
     *
     * <p>Resources are closed in reverse allocation order by {@link #closeResources()}. Only owned resources
     * should be registered here; borrowed handles must remain under their original owner.</p>
     *
     * @param resource resource to close when this run finishes
     */
    public void registerResource(ExecutionResource resource) {
        resourceRegistry.registerResource(resource);
    }

    /**
     * Closes all registered execution resources in reverse allocation order.
     *
     * <p>All close attempts are made even if one resource fails. Active and reserved device buffer bindings are
     * cleared afterward so no closed handle remains discoverable from runtime state.</p>
     */
    public void closeResources() {
        try {
            resourceRegistry.closeResources();
        } finally {
            deviceBindingRegistry.clear();
            nativeStorageRegistry.clear();
        }
    }

    /**
     * Returns whether a CPU read would need device-to-CPU materialization.
     *
     * @param nodeId compiled node id
     * @return {@code true} when CPU storage is stale and device storage is current
     */
    public boolean requiresCpuMaterialization(int nodeId) {
        return residencyForNodeId(nodeId).requiresCpuMaterialization();
    }

    /**
     * Verifies that CPU array storage is current before a CPU read or publication.
     *
     * @param nodeId compiled node id
     * @param reason reason for the requested CPU access
     * @throws IllegalStateException if the current value is only device-resident
     */
    public void requireCpuReadable(int nodeId, CpuMaterializationReason reason) {
        materializationService.requireCpuReadable(nodeId, reason);
    }

    /**
     * Verifies that native CPU storage is current before a native CPU read.
     *
     * @param nodeId compiled node id
     * @param reason reason for the requested native access
     * @return current native CPU storage
     */
    public NativeTensorStorage requireNativeReadable(int nodeId, CpuMaterializationReason reason) {
        return materializationService.requireNativeReadable(nodeId, reason);
    }

    /**
     * Verifies that native CPU storage is current before a native CPU read.
     *
     * @param nodeId compiled node id
     * @param reason reason for the requested native access
     * @param deviceTransferPolicy host/device transfer fallback policy
     * @return current native CPU storage
     */
    public NativeTensorStorage requireNativeReadable(
            int nodeId,
            CpuMaterializationReason reason,
            DeviceTransferPolicy deviceTransferPolicy
    ) {
        return materializationService.requireNativeReadable(nodeId, reason, deviceTransferPolicy);
    }

    /**
     * Copies current CPU array storage into native CPU storage and marks it current.
     *
     * @param nodeId compiled node id
     * @param reason materialization reason
     * @param detail trace detail
     * @return current native CPU storage
     */
    public NativeTensorStorage materializeArrayToNative(int nodeId, CpuMaterializationReason reason, String detail) {
        return materializationService.materializeArrayToNative(nodeId, reason, detail);
    }

    /**
     * Returns CPU materialization trace entries recorded during this execution.
     *
     * @return immutable trace entries
     */
    public List<CpuMaterializationTrace> cpuMaterializationTraces() {
        return residencyStore.cpuMaterializationTraces();
    }

    /**
     * Returns host/device transfer trace entries recorded during this execution.
     *
     * @return immutable transfer trace entries
     */
    public List<HostDeviceTransferTrace> hostDeviceTransferTraces() {
        return residencyStore.hostDeviceTransferTraces();
    }

    /**
     * Returns native CPU allocation counters recorded so far in this execution.
     *
     * @return native CPU memory trace snapshot
     */
    public NativeCpuMemoryTrace nativeCpuMemoryTrace() {
        return resourceRegistry.nativeCpuMemoryTrace();
    }
}

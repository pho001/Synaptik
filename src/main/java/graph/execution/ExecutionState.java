package graph.execution;

import backend.cpu.kernels.CpuNodeWorkspace;
import backend.cpu.kernels.CpuNodeExecutionPlan;
import backend.cpu.nativecpu.NativeCpuMaterializer;
import backend.cpu.nativecpu.NativeCpuStorageFactory;
import backend.cpu.plan.CpuPreparedInput;
import backend.memory.CpuMaterializationReason;
import backend.memory.CpuMaterializationResult;
import backend.memory.DeviceBufferBinding;
import backend.memory.DeviceToCpuMaterializer;
import backend.memory.ExecutionResource;
import backend.memory.StorageResidency;
import backend.memory.TensorResidencyState;
import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptor;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.execution.trace.CpuMaterializationTrace;
import tensor.DataType;
import tensor.NativeTensorStorage;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.IdentityHashMap;
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
    private record PreparedInputKey(int nodeId, int inputIndex) {
    }

    private final Map<Integer, Tensor> runtimeTensorByNodeId;
    private final Map<Integer, CpuNodeWorkspace> cpuWorkspaceByNodeId;
    private final Map<PreparedInputKey, Tensor> preparedInputTensorByKey;
    private final Map<Tensor, Integer> runtimeNodeIdByTensor;
    private final Map<Integer, TensorResidencyState> residencyByNodeId;
    private final Map<Integer, DeviceBufferBinding> deviceBufferBindingByNodeId;
    private final Map<Integer, DeviceBufferBinding> reservedDeviceBufferBindingByNodeId;
    private final Map<Integer, NativeTensorStorage> nativeStorageByNodeId;
    private final Map<String, DeviceToCpuMaterializer> deviceToCpuMaterializerByBackend;
    private final List<CpuMaterializationTrace> cpuMaterializationTraces;
    private final List<ExecutionResource> executionResources;

    private ExecutionState(
            Map<Integer, Tensor> runtimeTensorByNodeId,
            Map<Integer, CpuNodeWorkspace> cpuWorkspaceByNodeId,
            Map<PreparedInputKey, Tensor> preparedInputTensorByKey,
            Map<Tensor, Integer> runtimeNodeIdByTensor,
            Map<Integer, TensorResidencyState> residencyByNodeId,
            Map<Integer, DeviceBufferBinding> deviceBufferBindingByNodeId
    ) {
        this.runtimeTensorByNodeId = Map.copyOf(runtimeTensorByNodeId);
        this.cpuWorkspaceByNodeId = Map.copyOf(cpuWorkspaceByNodeId);
        this.preparedInputTensorByKey = Map.copyOf(preparedInputTensorByKey);
        this.runtimeNodeIdByTensor = Map.copyOf(runtimeNodeIdByTensor);
        this.residencyByNodeId = Map.copyOf(residencyByNodeId);
        this.deviceBufferBindingByNodeId = new HashMap<>(deviceBufferBindingByNodeId == null ? Map.of() : deviceBufferBindingByNodeId);
        this.reservedDeviceBufferBindingByNodeId = new HashMap<>();
        this.nativeStorageByNodeId = new HashMap<>();
        this.deviceToCpuMaterializerByBackend = new HashMap<>();
        this.cpuMaterializationTraces = new ArrayList<>();
        this.executionResources = new ArrayList<>();
    }

    /**
     * Creates per-run runtime tensors, prepared input buffers, and CPU workspaces.
     *
     * @param compiledNodes compiled node snapshots in graph order
     * @param descriptorIndex immutable tensor descriptor facts for {@code compiledNodes}
     * @param metadataIndex prepared execution metadata keyed by node id
     * @param forwardBoundaryNodeId last forward node id, used to decide leaf aliasing versus copying
     * @return mutable execution state for one run
     */
    public static ExecutionState create(
            List<CompiledNode> compiledNodes,
            CompiledTensorDescriptorIndex descriptorIndex,
            Map<Integer, CompiledNodeExecutionMetadata> metadataIndex,
            int forwardBoundaryNodeId
    ) {
        Objects.requireNonNull(compiledNodes, "compiledNodes cannot be null");
        Objects.requireNonNull(descriptorIndex, "descriptorIndex cannot be null");
        Objects.requireNonNull(metadataIndex, "metadataIndex cannot be null");

        Map<Integer, Tensor> runtimeTensors = new HashMap<>(compiledNodes.size());
        Map<Tensor, Integer> runtimeNodeIds = new IdentityHashMap<>(compiledNodes.size());
        Map<Integer, TensorResidencyState> residency = new HashMap<>(compiledNodes.size());
        for (CompiledNode node : compiledNodes) {
            Tensor runtimeTensor = new Tensor(
                    node.shape(),
                    node.strides(),
                    node.storageOffset(),
                    null,
                    node.operation(),
                    node.label(),
                    node.dataType()
            );
            CompiledTensorDescriptor descriptor = descriptorIndex.byNodeId(node.id());
            runtimeTensor.setRequiresGrad(descriptor.requiresGrad());
            runtimeTensor.setTrainableParameter(descriptor.trainableParameter());
            if (node.leaf()) {
                if (node.id() <= forwardBoundaryNodeId) {
                    TensorInternalAccess.aliasRuntimeFrom(runtimeTensor, node.sourceTensor());
                } else {
                    runtimeTensor.copyDataFrom(node.sourceTensor());
                }
                residency.put(node.id(), TensorResidencyState.cpuArrayCurrent("leaf runtime binding"));
            } else {
                residency.put(node.id(), TensorResidencyState.cpuArrayStale("runtime tensor allocated"));
            }
            runtimeTensors.put(node.id(), runtimeTensor);
            runtimeNodeIds.put(runtimeTensor, node.id());
        }
        for (CompiledNode node : compiledNodes) {
            if (node.inputIds().isEmpty()) {
                continue;
            }
            java.util.ArrayList<Tensor> runtimeInputs = new java.util.ArrayList<>(node.inputIds().size());
            for (int inputId : node.inputIds()) {
                Tensor input = runtimeTensors.get(inputId);
                if (input == null) {
                    throw new IllegalStateException("Missing runtime input tensor for nodeId=" + node.id() + ", inputId=" + inputId);
                }
                runtimeInputs.add(input);
            }
            TensorInternalAccess.setPrevTensors(runtimeTensors.get(node.id()), runtimeInputs);
        }
        for (CompiledNode node : compiledNodes) {
            if (node.inputIds().isEmpty() || !isCpuAliasView(node, descriptorIndex)) {
                continue;
            }
            int sourceNodeId = node.inputIds().getFirst();
            TensorResidencyState sourceResidency = residency.get(sourceNodeId);
            if (sourceResidency != null && sourceResidency.cpuCurrent()) {
                TensorInternalAccess.aliasRuntimeFrom(runtimeTensors.get(node.id()), runtimeTensors.get(sourceNodeId));
                residency.get(node.id()).markCpuCurrent("alias view runtime binding");
            }
        }

        Map<Integer, CpuNodeWorkspace> workspaces = new HashMap<>();
        Map<CpuNodeWorkspace, CpuNodeWorkspace> runtimeWorkspaceByTemplate = new IdentityHashMap<>();
        Map<PreparedInputKey, Tensor> preparedInputs = new HashMap<>();
        for (Map.Entry<Integer, CompiledNodeExecutionMetadata> entry : metadataIndex.entrySet()) {
            CpuNodeWorkspace workspace = entry.getValue().cpuWorkspace();
            if (workspace != null) {
                CpuNodeWorkspace runtimeWorkspace = runtimeWorkspaceByTemplate.computeIfAbsent(workspace, ignored -> workspace.fork());
                workspaces.put(entry.getKey(), runtimeWorkspace);
            }
            allocatePreparedInputs(entry.getKey(), entry.getValue().cpuPlan(), preparedInputs);
            if (entry.getValue().acceleratorExecutable() != null) {
                for (var fallbackStep : entry.getValue().acceleratorExecutable().cpuFallbackSteps()) {
                    allocatePreparedInputs(fallbackStep.node().id(), fallbackStep.metadata().cpuPlan(), preparedInputs);
                }
            }
        }
        return new ExecutionState(runtimeTensors, workspaces, preparedInputs, runtimeNodeIds, residency, Map.of());
    }

    private static void allocatePreparedInputs(
            int nodeId,
            CpuNodeExecutionPlan cpuPlan,
            Map<PreparedInputKey, Tensor> preparedInputs
    ) {
        if (cpuPlan == null || cpuPlan.layoutPlan().preparedInputs().isEmpty()) {
            return;
        }
        for (CpuPreparedInput preparedInput : cpuPlan.layoutPlan().preparedInputs()) {
            Tensor template = preparedInput.runtimeTensor();
            Tensor runtimePrepared = new Tensor(
                    template.getShapeUnsafe().clone(),
                    template.getStridesUnsafe().clone(),
                    template.getStorageOffsetUnsafe(),
                    null,
                    null,
                    template.getLabel(),
                    template.getDataType()
            );
            runtimePrepared.setRequiresGrad(template.getRequiresGrad());
            preparedInputs.put(new PreparedInputKey(nodeId, preparedInput.inputIndex()), runtimePrepared);
        }
    }

    private static boolean isCpuAliasView(
            CompiledNode node,
            CompiledTensorDescriptorIndex descriptorIndex
    ) {
        if (node == null || node.operation() == null || node.inputIds().isEmpty()) {
            return false;
        }
        return switch (node.operation().opType()) {
            case NOOP, EXPAND, SELECT, PERMUTE, EXPAND_DIMS, SQUEEZE -> true;
            case RESHAPE -> {
                CompiledTensorDescriptor source = descriptorIndex.byNodeId(node.inputIds().getFirst());
                yield source.contiguous();
            }
            default -> false;
        };
    }

    /**
     * Returns the runtime tensor for a compiled node.
     *
     * @param nodeId compiled node id
     * @return runtime tensor
     */
    public Tensor runtimeTensorForNodeId(int nodeId) {
        Tensor tensor = runtimeTensorByNodeId.get(nodeId);
        if (tensor == null) {
            throw new IllegalStateException("Missing runtime tensor for nodeId=" + nodeId);
        }
        return tensor;
    }

    /**
     * Returns the CPU workspace fork for a compiled node.
     *
     * @param nodeId compiled node id
     * @return CPU workspace, or {@code null} when the node does not use one
     */
    public CpuNodeWorkspace cpuWorkspaceForNodeId(int nodeId) {
        return cpuWorkspaceByNodeId.get(nodeId);
    }

    /**
     * Returns a prepared runtime input tensor for a node input.
     *
     * @param nodeId compiled node id
     * @param inputIndex input index
     * @return prepared runtime tensor
     */
    public Tensor preparedInputTensorFor(int nodeId, int inputIndex) {
        Tensor tensor = preparedInputTensorByKey.get(new PreparedInputKey(nodeId, inputIndex));
        if (tensor == null) {
            throw new IllegalStateException("Missing prepared runtime tensor for nodeId=" + nodeId + ", inputIndex=" + inputIndex);
        }
        return tensor;
    }

    /**
     * Looks up the compiled node id for a runtime tensor.
     *
     * @param tensor runtime tensor
     * @return node id, or {@code null} when the tensor is unknown
     */
    public Integer nodeIdForRuntimeTensor(Tensor tensor) {
        return tensor == null ? null : runtimeNodeIdByTensor.get(tensor);
    }

    /**
     * Returns runtime residency state for a compiled node.
     *
     * @param nodeId compiled node id
     * @return mutable residency state for the runtime tensor
     */
    public TensorResidencyState residencyForNodeId(int nodeId) {
        TensorResidencyState state = residencyByNodeId.get(nodeId);
        if (state == null) {
            throw new IllegalStateException("Missing runtime residency state for nodeId=" + nodeId);
        }
        return state;
    }

    /**
     * Marks a node output as current in CPU array storage.
     *
     * @param nodeId compiled node id
     * @param reason diagnostic transition reason
     */
    public void markCpuCurrent(int nodeId, String reason) {
        nativeStorageByNodeId.remove(nodeId);
        deviceBufferBindingByNodeId.remove(nodeId);
        reservedDeviceBufferBindingByNodeId.remove(nodeId);
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
        Objects.requireNonNull(storage, "storage cannot be null");
        Tensor tensor = runtimeTensorForNodeId(nodeId);
        if (tensor.getDataType() != storage.getType()) {
            throw new IllegalArgumentException("Native storage dtype mismatch for nodeId=" + nodeId
                    + ". tensorType=" + tensor.getDataType() + ", storageType=" + storage.getType());
        }
        if (tensor.getFlatDataSize() != storage.getSize()) {
            throw new IllegalArgumentException("Native storage size mismatch for nodeId=" + nodeId
                    + ". tensorElements=" + tensor.getFlatDataSize() + ", storageElements=" + storage.getSize());
        }
        nativeStorageByNodeId.put(nodeId, storage);
        if (storage.ownsSegment()) {
            registerResource(storage.allocation());
        }
        deviceBufferBindingByNodeId.remove(nodeId);
        reservedDeviceBufferBindingByNodeId.remove(nodeId);
        residencyForNodeId(nodeId).markNativeCurrent(reason);
    }

    /**
     * Returns native CPU storage attached to a node.
     *
     * @param nodeId compiled node id
     * @return native storage, or {@code null} when none is attached
     */
    public NativeTensorStorage nativeStorageForNodeId(int nodeId) {
        residencyForNodeId(nodeId);
        return nativeStorageByNodeId.get(nodeId);
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
        nativeStorageByNodeId.remove(nodeId);
        deviceBufferBindingByNodeId.remove(nodeId);
        reservedDeviceBufferBindingByNodeId.remove(nodeId);
        residencyForNodeId(nodeId).markDeviceCurrent(residency, deviceBackend, reason);
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
        validateDeviceBufferBinding(nodeId, binding);
        reservedDeviceBufferBindingByNodeId.put(nodeId, binding);
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
        Objects.requireNonNull(binding, "binding cannot be null");
        Objects.requireNonNull(residency, "residency cannot be null");
        validateDeviceBufferBinding(nodeId, binding);
        if (residency == StorageResidency.CPU_ARRAY || residency == StorageResidency.CPU_NATIVE) {
            throw new IllegalArgumentException("Device buffer binding requires a device residency.");
        }
        nativeStorageByNodeId.remove(nodeId);
        deviceBufferBindingByNodeId.put(nodeId, binding);
        reservedDeviceBufferBindingByNodeId.remove(nodeId);
        if (residency == StorageResidency.HOST_SHARED_DEVICE_BUFFER) {
            residencyForNodeId(nodeId).markSharedBufferCurrent(binding.backendId(), reason);
            return;
        }
        residencyForNodeId(nodeId).markDeviceCurrent(residency, binding.backendId(), reason);
    }

    /**
     * Returns the registered device buffer binding for a runtime tensor.
     *
     * @param nodeId compiled node id
     * @return device buffer binding, or {@code null} when none is registered
     */
    public DeviceBufferBinding deviceBufferBindingForNodeId(int nodeId) {
        residencyForNodeId(nodeId);
        return deviceBufferBindingByNodeId.get(nodeId);
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
        residencyForNodeId(nodeId);
        DeviceBufferBinding reserved = reservedDeviceBufferBindingByNodeId.get(nodeId);
        return reserved == null ? deviceBufferBindingByNodeId.get(nodeId) : reserved;
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
        markMaterializedToCpu(nodeId, reason, 0L);
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
        markMaterializedToCpu(nodeId, reason, durationNs, "device value synchronized to CPU storage");
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
        Objects.requireNonNull(reason, "reason cannot be null");
        TensorResidencyState state = residencyForNodeId(nodeId);
        cpuMaterializationTraces.add(new CpuMaterializationTrace(
                nodeId,
                reason,
                state.deviceBackend(),
                state.residency(),
                logicalByteLength(nodeId),
                durationNs,
                true,
                detail == null || detail.isBlank() ? "device value synchronized to CPU storage" : detail
        ));
        deviceBufferBindingByNodeId.remove(nodeId);
        reservedDeviceBufferBindingByNodeId.remove(nodeId);
        state.markMaterializedToCpu(reason.label());
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
        deviceToCpuMaterializerByBackend.put(backendId, Objects.requireNonNull(materializer, "materializer cannot be null"));
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
        executionResources.add(Objects.requireNonNull(resource, "resource cannot be null"));
    }

    /**
     * Closes all registered execution resources in reverse allocation order.
     *
     * <p>All close attempts are made even if one resource fails. Active and reserved device buffer bindings are
     * cleared afterward so no closed handle remains discoverable from runtime state.</p>
     */
    public void closeResources() {
        RuntimeException closeFailure = null;
        for (int i = executionResources.size() - 1; i >= 0; i--) {
            try {
                executionResources.get(i).close();
            } catch (RuntimeException ex) {
                if (closeFailure == null) {
                    closeFailure = new RuntimeException("One or more execution resources failed to close.");
                }
                closeFailure.addSuppressed(ex);
            }
        }
        executionResources.clear();
        deviceBufferBindingByNodeId.clear();
        reservedDeviceBufferBindingByNodeId.clear();
        nativeStorageByNodeId.clear();
        if (closeFailure != null) {
            throw closeFailure;
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
        Objects.requireNonNull(reason, "reason cannot be null");
        TensorResidencyState state = residencyForNodeId(nodeId);
        if (state.requiresCpuMaterialization()) {
            tryMaterializeToCpu(nodeId, reason, state);
            return;
        }
        if (!state.cpuCurrent()) {
            cpuMaterializationTraces.add(new CpuMaterializationTrace(
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

    /**
     * Verifies that native CPU storage is current before a native CPU read.
     *
     * @param nodeId compiled node id
     * @param reason reason for the requested native access
     * @return current native CPU storage
     */
    public NativeTensorStorage requireNativeReadable(int nodeId, CpuMaterializationReason reason) {
        Objects.requireNonNull(reason, "reason cannot be null");
        TensorResidencyState state = residencyForNodeId(nodeId);
        if (state.nativeCurrent()) {
            NativeTensorStorage storage = nativeStorageByNodeId.get(nodeId);
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
            requireCpuReadable(nodeId, reason);
            return materializeArrayToNative(nodeId, reason, "device_to_array_to_native");
        }
        throw new IllegalStateException(
                "Native CPU read requested for nodeId=" + nodeId
                        + " reason=" + reason.label()
                        + " but neither native, CPU array, nor device storage is current."
        );
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
        Objects.requireNonNull(reason, "reason cannot be null");
        requireCpuReadable(nodeId, reason);
        Tensor tensor = runtimeTensorForNodeId(nodeId);
        NativeTensorStorage storage = nativeStorageByNodeId.get(nodeId);
        if (storage == null || storage.closed() || storage.getType() != tensor.getDataType()
                || storage.getSize() != tensor.getFlatDataSize()) {
            storage = new NativeCpuStorageFactory().allocate(
                    tensor.getDataType(),
                    tensor.getFlatDataSize(),
                    "node-" + nodeId + ":" + tensor.getLabel()
            );
            nativeStorageByNodeId.put(nodeId, storage);
            if (storage.ownsSegment()) {
                registerResource(storage.allocation());
            }
        }
        long start = System.nanoTime();
        NativeCpuMaterializer.arrayToNative(tensor, storage);
        long durationNs = System.nanoTime() - start;
        cpuMaterializationTraces.add(new CpuMaterializationTrace(
                nodeId,
                reason,
                "",
                StorageResidency.CPU_ARRAY,
                logicalByteLength(nodeId),
                durationNs,
                true,
                detail == null || detail.isBlank() ? "array_to_native" : detail
        ));
        deviceBufferBindingByNodeId.remove(nodeId);
        reservedDeviceBufferBindingByNodeId.remove(nodeId);
        residencyForNodeId(nodeId).markNativeCurrent(reason.label());
        return storage;
    }

    private void tryMaterializeToCpu(int nodeId, CpuMaterializationReason reason, TensorResidencyState state) {
        if (state.nativeCurrent()) {
            NativeTensorStorage storage = nativeStorageByNodeId.get(nodeId);
            if (storage != null) {
                long start = System.nanoTime();
                NativeCpuMaterializer.nativeToArray(storage, runtimeTensorForNodeId(nodeId));
                markMaterializedToCpu(
                        nodeId,
                        reason,
                        System.nanoTime() - start,
                        "native_to_array"
                );
                return;
            }
            cpuMaterializationTraces.add(new CpuMaterializationTrace(
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
        DeviceBufferBinding binding = deviceBufferBindingByNodeId.get(nodeId);
        DeviceToCpuMaterializer materializer = binding == null ? null : deviceToCpuMaterializerByBackend.get(binding.backendId());
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
        cpuMaterializationTraces.add(new CpuMaterializationTrace(
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

    private void validateDeviceBufferBinding(int nodeId, DeviceBufferBinding binding) {
        Objects.requireNonNull(binding, "binding cannot be null");
        if (binding.nodeId() != nodeId) {
            throw new IllegalArgumentException("Device buffer binding nodeId=" + binding.nodeId()
                    + " does not match requested nodeId=" + nodeId);
        }
        if (!binding.available()) {
            throw new IllegalArgumentException("Device buffer binding is not available: " + binding.describe());
        }
        residencyForNodeId(nodeId);
    }

    /**
     * Returns CPU materialization trace entries recorded during this execution.
     *
     * @return immutable trace entries
     */
    public List<CpuMaterializationTrace> cpuMaterializationTraces() {
        return List.copyOf(cpuMaterializationTraces);
    }

    private long logicalByteLength(int nodeId) {
        Tensor tensor = runtimeTensorForNodeId(nodeId);
        return (long) tensor.getFlatDataSize() * elementByteSize(tensor.getDataType());
    }

    private static int elementByteSize(DataType dataType) {
        if (dataType == null) {
            return 0;
        }
        return switch (dataType) {
            case FLOAT64 -> Double.BYTES;
            case FLOAT32 -> Float.BYTES;
            case BFLOAT16 -> Short.BYTES;
            case BOOL -> Byte.BYTES;
            case INT32 -> Integer.BYTES;
            case INT64 -> Long.BYTES;
        };
    }
}

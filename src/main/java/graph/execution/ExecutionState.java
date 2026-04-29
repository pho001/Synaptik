package graph.execution;

import backend.cpu.kernels.CpuNodeWorkspace;
import backend.cpu.plan.CpuPreparedInput;
import backend.memory.CpuMaterializationReason;
import backend.memory.DeviceBufferBinding;
import backend.memory.StorageResidency;
import backend.memory.TensorResidencyState;
import graph.CompiledNode;
import graph.execution.trace.CpuMaterializationTrace;
import tensor.DataType;
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
    private final List<CpuMaterializationTrace> cpuMaterializationTraces;

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
        this.cpuMaterializationTraces = new ArrayList<>();
    }

    /**
     * Creates per-run runtime tensors, prepared input buffers, and CPU workspaces.
     *
     * @param compiledNodes compiled node snapshots in graph order
     * @param metadataIndex prepared execution metadata keyed by node id
     * @param forwardBoundaryNodeId last forward node id, used to decide leaf aliasing versus copying
     * @return mutable execution state for one run
     */
    public static ExecutionState create(
            List<CompiledNode> compiledNodes,
            Map<Integer, CompiledNodeExecutionMetadata> metadataIndex,
            int forwardBoundaryNodeId
    ) {
        Objects.requireNonNull(compiledNodes, "compiledNodes cannot be null");
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
            runtimeTensor.setRequiresGrad(node.semanticTensor().getRequiresGrad());
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

        Map<Integer, CpuNodeWorkspace> workspaces = new HashMap<>();
        Map<CpuNodeWorkspace, CpuNodeWorkspace> runtimeWorkspaceByTemplate = new IdentityHashMap<>();
        Map<PreparedInputKey, Tensor> preparedInputs = new HashMap<>();
        for (Map.Entry<Integer, CompiledNodeExecutionMetadata> entry : metadataIndex.entrySet()) {
            CpuNodeWorkspace workspace = entry.getValue().cpuWorkspace();
            if (workspace != null) {
                CpuNodeWorkspace runtimeWorkspace = runtimeWorkspaceByTemplate.computeIfAbsent(workspace, ignored -> workspace.fork());
                workspaces.put(entry.getKey(), runtimeWorkspace);
            }
            if (entry.getValue().cpuPlan() != null) {
                for (CpuPreparedInput preparedInput : entry.getValue().cpuPlan().layoutPlan().preparedInputs()) {
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
                    preparedInputs.put(new PreparedInputKey(entry.getKey(), preparedInput.inputIndex()), runtimePrepared);
                }
            }
        }
        return new ExecutionState(runtimeTensors, workspaces, preparedInputs, runtimeNodeIds, residency, Map.of());
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
        deviceBufferBindingByNodeId.remove(nodeId);
        residencyForNodeId(nodeId).markCpuCurrent(reason);
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
        deviceBufferBindingByNodeId.remove(nodeId);
        residencyForNodeId(nodeId).markDeviceCurrent(residency, deviceBackend, reason);
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
        if (binding.nodeId() != nodeId) {
            throw new IllegalArgumentException("Device buffer binding nodeId=" + binding.nodeId()
                    + " does not match requested nodeId=" + nodeId);
        }
        if (residency == StorageResidency.CPU_ARRAY) {
            throw new IllegalArgumentException("Device buffer binding requires a device residency.");
        }
        if (!binding.available()) {
            throw new IllegalArgumentException("Device buffer binding is not available: " + binding.describe());
        }
        residencyForNodeId(nodeId);
        deviceBufferBindingByNodeId.put(nodeId, binding);
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
                "device value synchronized to CPU storage"
        ));
        deviceBufferBindingByNodeId.remove(nodeId);
        state.markMaterializedToCpu(reason.label());
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
            cpuMaterializationTraces.add(new CpuMaterializationTrace(
                    nodeId,
                    reason,
                    state.deviceBackend(),
                    state.residency(),
                    logicalByteLength(nodeId),
                    0L,
                    false,
                    "no device-to-CPU materializer is available"
            ));
            throw new IllegalStateException(
                    "CPU materialization requested for nodeId=" + nodeId
                            + " reason=" + reason.label()
                            + " but no device-to-CPU materializer is available for backend="
                            + state.deviceBackend()
                            + ", residency=" + state.residency()
                            + ". This prevents publishing stale CPU tensor storage."
            );
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
        };
    }
}

package backend.runtime;

import backend.memory.CpuMaterializationReason;
import backend.memory.DeviceBufferBinding;
import backend.memory.DeviceToCpuMaterializer;
import backend.memory.ExecutionResource;
import backend.memory.StorageResidency;
import backend.memory.TensorResidencyState;
import backend.accelerator.buffer.AcceleratorLayoutTransformDecision;
import config.runtime.RuntimeConfig;
import graph.execution.CompiledNodeExecutionMetadata;
import graph.execution.ExecutionState;
import graph.execution.trace.ConvTraceMetadata;
import tensor.Tensor;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Per-run runtime context passed to backend kernels.
 *
 * <p>The context exposes execution mode, approximation decisions, prepared metadata, runtime tensor
 * lookup, per-node workspaces, and small backend-specific state maps. It is created for a prepared
 * execution run and should not be treated as a global singleton.</p>
 *
 * <p>The runtime-state and conv-trace maps are synchronized because individual backend helpers can
 * publish state while executing. The graph execution scheduler still owns the higher-level ordering and
 * lifetime of tensors and workspaces.</p>
 */
public final class ExecutionContext {
    private final RuntimeConfig runtimeConfig;
    private final ExecutionMode mode;
    private final boolean useFastExpApprox;
    private final boolean useFastTanhApprox;
    private final Map<Integer, CompiledNodeExecutionMetadata> metadataIndex;
    private final ExecutionState executionState;
    private final Map<Tensor, Object> runtimeStateIndex;
    private final Map<Integer, ConvTraceMetadata> convTraceIndex;
    private final Map<Integer, AcceleratorLayoutTransformDecision> layoutTransformDecisionIndex;
    private final Map<Class<?>, Object> runtimeServices;

    /**
     * Creates a standalone context without prepared execution state.
     *
     * @param mode execution mode
     * @param useFastExpApprox whether exp may use fast approximation
     * @param useFastTanhApprox whether tanh may use fast approximation
     */
    public ExecutionContext(ExecutionMode mode, boolean useFastExpApprox, boolean useFastTanhApprox) {
        this(mode, useFastExpApprox, useFastTanhApprox, Map.of(), null);
    }

    public ExecutionContext(
            ExecutionMode mode,
            boolean useFastExpApprox,
            boolean useFastTanhApprox,
            Map<Integer, CompiledNodeExecutionMetadata> metadataIndex,
            ExecutionState executionState
    ) {
        this(null, mode, useFastExpApprox, useFastTanhApprox, metadataIndex, executionState);
    }

    private ExecutionContext(
            RuntimeConfig runtimeConfig,
            ExecutionMode mode,
            boolean useFastExpApprox,
            boolean useFastTanhApprox,
            Map<Integer, CompiledNodeExecutionMetadata> metadataIndex,
            ExecutionState executionState
    ) {
        this.runtimeConfig = runtimeConfig;
        this.mode = Objects.requireNonNull(mode, "mode cannot be null");
        this.metadataIndex = Map.copyOf(metadataIndex == null ? Map.of() : metadataIndex);
        this.executionState = executionState;
        this.runtimeStateIndex = Collections.synchronizedMap(new IdentityHashMap<>());
        this.convTraceIndex = Collections.synchronizedMap(new java.util.HashMap<>());
        this.layoutTransformDecisionIndex = Collections.synchronizedMap(new java.util.HashMap<>());
        this.runtimeServices = Collections.synchronizedMap(new java.util.HashMap<>());
        this.useFastExpApprox = useFastExpApprox;
        this.useFastTanhApprox = useFastTanhApprox;
    }

    /**
     * Creates a context from runtime config without prepared metadata or per-run execution state.
     *
     * @param runtimeConfig runtime config used to resolve approximation flags
     * @param mode execution mode
     * @return execution context
     */
    public static ExecutionContext fromRuntimeConfig(RuntimeConfig runtimeConfig, ExecutionMode mode) {
        return fromRuntimeConfig(runtimeConfig, mode, Map.of(), null);
    }

    /**
     * Creates a context from runtime config and prepared execution state.
     *
     * @param runtimeConfig runtime config used to resolve approximation flags
     * @param mode execution mode
     * @param metadataIndex prepared metadata indexed by compiled node id
     * @param executionState per-run tensor/workspace state; may be {@code null} for limited contexts
     * @return execution context
     */
    public static ExecutionContext fromRuntimeConfig(
            RuntimeConfig runtimeConfig,
            ExecutionMode mode,
            Map<Integer, CompiledNodeExecutionMetadata> metadataIndex,
            ExecutionState executionState
    ) {
        Objects.requireNonNull(runtimeConfig, "runtimeConfig cannot be null");
        Objects.requireNonNull(mode, "mode cannot be null");
        boolean backwardEnabled = mode == ExecutionMode.FORWARD_BACKWARD;
        return new ExecutionContext(
                runtimeConfig,
                mode,
                runtimeConfig.approximation().useFastExp(backwardEnabled),
                runtimeConfig.approximation().useFastTanh(backwardEnabled),
                metadataIndex,
                executionState
        );
    }

    /**
     * Returns the runtime configuration used to create this context, when available.
     *
     * @return runtime configuration, or {@code null} for standalone test contexts
     */
    public RuntimeConfig runtimeConfig() {
        return runtimeConfig;
    }

    /**
     * @return current execution mode
     */
    public ExecutionMode mode() {
        return mode;
    }

    /**
     * @return {@code true} when the current run includes backward execution
     */
    public boolean runsBackwardPass() {
        return mode == ExecutionMode.FORWARD_BACKWARD;
    }

    public boolean useFastExpApprox() {
        return useFastExpApprox;
    }

    public boolean useFastTanhApprox() {
        return useFastTanhApprox;
    }

    public CompiledNodeExecutionMetadata metadataForNodeId(int nodeId) {
        return metadataIndex.get(nodeId);
    }

    /**
     * Returns the runtime tensor allocated for a compiled node id.
     *
     * @param nodeId compiled node id
     * @return runtime tensor for the node
     * @throws IllegalStateException if this context was created without per-run execution state
     */
    public Tensor runtimeTensorForNodeId(int nodeId) {
        if (executionState == null) {
            throw new IllegalStateException("ExecutionContext does not carry per-run ExecutionState.");
        }
        return executionState.runtimeTensorForNodeId(nodeId);
    }

    public backend.cpu.kernels.CpuNodeWorkspace cpuWorkspaceForNodeId(int nodeId) {
        if (executionState == null) {
            return null;
        }
        return executionState.cpuWorkspaceForNodeId(nodeId);
    }

    public Tensor preparedInputTensorFor(int nodeId, int inputIndex) {
        if (executionState == null) {
            throw new IllegalStateException("ExecutionContext does not carry per-run ExecutionState.");
        }
        return executionState.preparedInputTensorFor(nodeId, inputIndex);
    }

    public Integer nodeIdForRuntimeTensor(Tensor tensor) {
        if (executionState == null) {
            return null;
        }
        return executionState.nodeIdForRuntimeTensor(tensor);
    }

    /**
     * Returns runtime residency state for a compiled node when this context has execution state.
     *
     * @param nodeId compiled node id
     * @return residency state, or {@code null} for standalone contexts
     */
    public TensorResidencyState residencyForNodeId(int nodeId) {
        if (executionState == null) {
            return null;
        }
        return executionState.residencyForNodeId(nodeId);
    }

    /**
     * Marks a node output as current in CPU array storage.
     *
     * @param nodeId compiled node id
     * @param reason diagnostic transition reason
     */
    public void markCpuCurrent(int nodeId, String reason) {
        if (executionState != null) {
            executionState.markCpuCurrent(nodeId, reason);
        }
    }

    /**
     * Marks a node output as current only in a device-visible representation.
     *
     * @param nodeId compiled node id
     * @param residency device residency class; must not be {@link StorageResidency#CPU_ARRAY}
     * @param deviceBackend backend id such as {@code GPU_METAL}
     * @param reason diagnostic transition reason
     */
    public void markDeviceCurrent(int nodeId, StorageResidency residency, String deviceBackend, String reason) {
        if (executionState != null) {
            executionState.markDeviceCurrent(nodeId, residency, deviceBackend, reason);
        }
    }

    /**
     * Registers a usable device buffer binding for a runtime tensor and updates residency.
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
        if (executionState != null) {
            executionState.attachDeviceBufferBinding(nodeId, binding, residency, reason);
        }
    }

    /**
     * Reserves a writable device buffer for a future backend output without marking it current.
     *
     * @param nodeId compiled node id
     * @param binding backend-specific buffer binding
     */
    public void reserveDeviceBufferBinding(int nodeId, DeviceBufferBinding binding) {
        if (executionState != null) {
            executionState.reserveDeviceBufferBinding(nodeId, binding);
        }
    }

    /**
     * Returns the registered device buffer binding for a runtime tensor.
     *
     * @param nodeId compiled node id
     * @return device buffer binding, or {@code null} when no execution state or binding exists
     */
    public DeviceBufferBinding deviceBufferBindingForNodeId(int nodeId) {
        if (executionState == null) {
            return null;
        }
        return executionState.deviceBufferBindingForNodeId(nodeId);
    }

    /**
     * Returns a writable device buffer binding for a future backend output.
     *
     * @param nodeId compiled node id
     * @return writable binding, or {@code null} when none is available
     */
    public DeviceBufferBinding writableDeviceBufferBindingForNodeId(int nodeId) {
        if (executionState == null) {
            return null;
        }
        return executionState.writableDeviceBufferBindingForNodeId(nodeId);
    }

    /**
     * Marks a completed device-to-CPU synchronization.
     *
     * @param nodeId compiled node id
     * @param reason reason that forced CPU materialization
     */
    public void markMaterializedToCpu(int nodeId, CpuMaterializationReason reason) {
        if (executionState != null) {
            executionState.markMaterializedToCpu(nodeId, reason);
        }
    }

    /**
     * Marks a completed device-to-CPU synchronization with measured duration.
     *
     * @param nodeId compiled node id
     * @param reason reason that forced CPU materialization
     * @param durationNs measured materialization duration
     */
    public void markMaterializedToCpu(int nodeId, CpuMaterializationReason reason, long durationNs) {
        if (executionState != null) {
            executionState.markMaterializedToCpu(nodeId, reason, durationNs);
        }
    }

    /**
     * Registers a backend hook that can synchronize device-current values into CPU tensor storage.
     *
     * @param backendId backend id such as {@code GPU_METAL}
     * @param materializer materializer implementation
     */
    public void registerDeviceToCpuMaterializer(String backendId, DeviceToCpuMaterializer materializer) {
        if (executionState != null) {
            executionState.registerDeviceToCpuMaterializer(backendId, materializer);
        }
    }

    /**
     * Registers a resource whose lifetime is owned by this execution run.
     *
     * @param resource resource to close when the run finishes
     */
    public void registerResource(ExecutionResource resource) {
        if (executionState != null) {
            executionState.registerResource(resource);
        }
    }

    /**
     * Verifies that CPU array storage is current before a CPU read or publication.
     *
     * @param nodeId compiled node id
     * @param reason reason for the requested CPU access
     */
    public void requireCpuReadable(int nodeId, CpuMaterializationReason reason) {
        if (executionState != null) {
            executionState.requireCpuReadable(nodeId, reason);
        }
    }

    /**
     * Reads backend-specific state attached to a tensor.
     *
     * @param tensor tensor identity used as key
     * @param type expected state type
     * @return state cast to {@code type}, or {@code null} when no state of that type exists
     */
    public <T> T runtimeStateFor(Tensor tensor, Class<T> type) {
        Objects.requireNonNull(type, "type cannot be null");
        Object state = runtimeStateIndex.get(tensor);
        return type.isInstance(state) ? type.cast(state) : null;
    }

    /**
     * Publishes backend-specific state for a tensor.
     *
     * @param tensor tensor identity used as key; must not be {@code null}
     * @param runtimeState state object, or {@code null} to remove the mapping
     */
    public void putRuntimeState(Tensor tensor, Object runtimeState) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        if (runtimeState == null) {
            runtimeStateIndex.remove(tensor);
            return;
        }
        runtimeStateIndex.put(tensor, runtimeState);
    }

    public void clearRuntimeState(Tensor tensor) {
        if (tensor != null) {
            runtimeStateIndex.remove(tensor);
        }
    }

    /**
     * Registers a run-scoped service object.
     */
    public <T> void registerRuntimeService(Class<T> type, T service) {
        Objects.requireNonNull(type, "type cannot be null");
        if (service == null) {
            runtimeServices.remove(type);
            return;
        }
        runtimeServices.put(type, type.cast(service));
    }

    /**
     * Returns a run-scoped service object.
     */
    public <T> T runtimeService(Class<T> type) {
        Objects.requireNonNull(type, "type cannot be null");
        Object service = runtimeServices.get(type);
        return type.isInstance(service) ? type.cast(service) : null;
    }

    public void mirrorRuntimeState(Tensor source, Tensor target) {
        Objects.requireNonNull(target, "target cannot be null");
        if (source == null) {
            runtimeStateIndex.remove(target);
            return;
        }
        Object state = runtimeStateIndex.get(source);
        if (state == null) {
            runtimeStateIndex.remove(target);
            return;
        }
        runtimeStateIndex.put(target, state);
    }

    public ConvTraceMetadata convTraceForNodeId(int nodeId) {
        return convTraceIndex.get(nodeId);
    }

    public void publishConvTrace(int nodeId, ConvTraceMetadata trace) {
        if (trace == null) {
            convTraceIndex.remove(nodeId);
            return;
        }
        convTraceIndex.put(nodeId, trace);
    }

    /**
     * Publishes the layout-transform decision observed for a node during this run.
     */
    public void publishLayoutTransformDecision(int nodeId, AcceleratorLayoutTransformDecision decision) {
        if (decision == null) {
            layoutTransformDecisionIndex.remove(nodeId);
            return;
        }
        layoutTransformDecisionIndex.put(nodeId, decision);
    }

    /**
     * Returns the layout-transform decision observed for a node during this run.
     */
    public AcceleratorLayoutTransformDecision layoutTransformDecisionForNodeId(int nodeId) {
        return layoutTransformDecisionIndex.get(nodeId);
    }
}

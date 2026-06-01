package backend.runtime;

import backend.memory.CpuMaterializationReason;
import backend.memory.DeviceBufferBinding;
import backend.memory.DeviceToCpuMaterializer;
import backend.memory.DeviceToNativeMaterializer;
import backend.memory.ExecutionResource;
import backend.memory.StorageResidency;
import backend.memory.TensorResidencyState;
import backend.accelerator.buffer.AcceleratorLayoutTransformDecision;
import config.runtime.RuntimeConfig;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.state.ExecutionState;
import graph.execution.trace.ConvTraceMetadata;
import graph.execution.trace.HostDeviceTransferTrace;
import tensor.DataType;
import tensor.storage.NativeTensorStorage;
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
        this(
                runtimeConfig,
                mode,
                useFastExpApprox,
                useFastTanhApprox,
                metadataIndex,
                executionState,
                null,
                null,
                null,
                null
        );
    }

    private ExecutionContext(
            RuntimeConfig runtimeConfig,
            ExecutionMode mode,
            boolean useFastExpApprox,
            boolean useFastTanhApprox,
            Map<Integer, CompiledNodeExecutionMetadata> metadataIndex,
            ExecutionState executionState,
            Map<Tensor, Object> runtimeStateIndex,
            Map<Integer, ConvTraceMetadata> convTraceIndex,
            Map<Integer, AcceleratorLayoutTransformDecision> layoutTransformDecisionIndex,
            Map<Class<?>, Object> runtimeServices
    ) {
        this.runtimeConfig = runtimeConfig;
        this.mode = Objects.requireNonNull(mode, "mode cannot be null");
        this.metadataIndex = Map.copyOf(metadataIndex == null ? Map.of() : metadataIndex);
        this.executionState = executionState;
        this.runtimeStateIndex = runtimeStateIndex == null
                ? Collections.synchronizedMap(new IdentityHashMap<>())
                : runtimeStateIndex;
        this.convTraceIndex = convTraceIndex == null
                ? Collections.synchronizedMap(new java.util.HashMap<>())
                : convTraceIndex;
        this.layoutTransformDecisionIndex = layoutTransformDecisionIndex == null
                ? Collections.synchronizedMap(new java.util.HashMap<>())
                : layoutTransformDecisionIndex;
        this.runtimeServices = runtimeServices == null
                ? Collections.synchronizedMap(new java.util.HashMap<>())
                : runtimeServices;
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
     * Returns a context view with the same per-run state but a different runtime policy.
     *
     * <p>This is used for private fallback subplans that must override routing policy while still
     * writing into the same runtime tensors, workspaces, traces, and services as the parent run.</p>
     *
     * @param newRuntimeConfig replacement runtime config
     * @return context sharing this run's mutable state
     */
    public ExecutionContext withRuntimeConfig(RuntimeConfig newRuntimeConfig) {
        Objects.requireNonNull(newRuntimeConfig, "newRuntimeConfig cannot be null");
        boolean backwardEnabled = mode == ExecutionMode.FORWARD_BACKWARD;
        return new ExecutionContext(
                newRuntimeConfig,
                mode,
                newRuntimeConfig.approximation().useFastExp(backwardEnabled),
                newRuntimeConfig.approximation().useFastTanh(backwardEnabled),
                metadataIndex,
                executionState,
                runtimeStateIndex,
                convTraceIndex,
                layoutTransformDecisionIndex,
                runtimeServices
        );
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

    public backend.cpu.execution.CpuNodeWorkspace cpuWorkspaceForNodeId(int nodeId) {
        Object workspace = workspaceForNodeId(nodeId);
        if (workspace == null) {
            return null;
        }
        if (workspace instanceof backend.cpu.execution.CpuNodeWorkspace cpuWorkspace) {
            return cpuWorkspace;
        }
        throw new IllegalStateException("Runtime workspace for nodeId=" + nodeId
                + " is not a CpuNodeWorkspace: " + workspace.getClass().getName());
    }

    public backend.cpu1.exec.Cpu1Workspace cpu1WorkspaceForNodeId(int nodeId) {
        Object workspace = workspaceForNodeId(nodeId);
        if (workspace == null) {
            return null;
        }
        if (workspace instanceof backend.cpu1.exec.Cpu1Workspace cpu1Workspace) {
            return cpu1Workspace;
        }
        throw new IllegalStateException("Runtime workspace for nodeId=" + nodeId
                + " is not a Cpu1Workspace: " + workspace.getClass().getName());
    }

    public Object workspaceForNodeId(int nodeId) {
        if (executionState == null) {
            return null;
        }
        return executionState.workspaceForNodeId(nodeId);
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
     * Attaches native CPU storage as the current representation for a runtime tensor.
     *
     * @param nodeId compiled node id
     * @param storage native CPU storage containing the current value
     * @param reason diagnostic transition reason
     */
    public void attachNativeStorage(int nodeId, NativeTensorStorage storage, String reason) {
        if (executionState != null) {
            executionState.attachNativeStorage(nodeId, storage, reason);
        }
    }

    /**
     * Reserves native CPU storage for a future output write without marking it current.
     *
     * @param nodeId compiled node id
     * @param storage native CPU storage available for an upcoming write
     */
    public void reserveNativeOutputStorage(int nodeId, NativeTensorStorage storage) {
        if (executionState != null) {
            executionState.reserveNativeOutputStorage(nodeId, storage);
        }
    }

    /**
     * Attaches a target node to the current native CPU storage of a source node for view-only aliases.
     *
     * @param targetNodeId target compiled node id
     * @param sourceNodeId source compiled node id
     * @param reason diagnostic transition reason
     */
    public void aliasNativeStorage(int targetNodeId, int sourceNodeId, String reason) {
        if (executionState == null) {
            throw new IllegalStateException("ExecutionContext does not carry per-run ExecutionState.");
        }
        executionState.aliasNativeStorage(targetNodeId, sourceNodeId, reason);
    }

    /**
     * Returns native CPU storage attached to a runtime tensor.
     *
     * @param nodeId compiled node id
     * @return native storage, or {@code null} when no execution state or storage exists
     */
    public NativeTensorStorage nativeStorageForNodeId(int nodeId) {
        if (executionState == null) {
            return null;
        }
        return executionState.nativeStorageForNodeId(nodeId);
    }

    /**
     * Allocates run-owned native CPU tensor storage.
     *
     * @param dataType tensor dtype
     * @param elements number of logical elements
     * @param label diagnostic allocation label
     * @return native tensor storage
     */
    public NativeTensorStorage allocateNativeStorage(DataType dataType, int elements, String label) {
        if (executionState == null) {
            throw new IllegalStateException("ExecutionContext does not carry per-run ExecutionState.");
        }
        return executionState.allocateNativeStorage(dataType, elements, label);
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
     * Registers a backend hook that can synchronize device-current values directly into native CPU storage.
     *
     * @param backendId backend id such as {@code GPU_METAL}
     * @param materializer materializer implementation
     */
    public void registerDeviceToNativeMaterializer(String backendId, DeviceToNativeMaterializer materializer) {
        if (executionState != null) {
            executionState.registerDeviceToNativeMaterializer(backendId, materializer);
        }
    }

    /**
     * Records a host/device transfer route observed by a backend helper.
     *
     * @param trace transfer trace entry
     */
    public void recordHostDeviceTransfer(HostDeviceTransferTrace trace) {
        if (executionState != null) {
            executionState.recordHostDeviceTransfer(trace);
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
     * Verifies that native CPU storage is current before a native CPU read.
     *
     * @param nodeId compiled node id
     * @param reason reason for the requested native access
     * @return current native CPU storage
     */
    public NativeTensorStorage requireNativeReadable(int nodeId, CpuMaterializationReason reason) {
        if (executionState == null) {
            throw new IllegalStateException("ExecutionContext does not carry per-run ExecutionState.");
        }
        return executionState.requireNativeReadable(
                nodeId,
                reason,
                runtimeConfig == null ? null : runtimeConfig.deviceTransferPolicy()
        );
    }

    /**
     * Returns the number of CPU materialization events recorded so far in this run.
     *
     * @return materialization trace count, or zero when no execution state is attached
     */
    public int cpuMaterializationTraceCount() {
        return executionState == null ? 0 : executionState.cpuMaterializationTraces().size();
    }

    /**
     * Returns the number of host/device transfer events recorded so far in this run.
     *
     * @return transfer trace count, or zero when no execution state is attached
     */
    public int hostDeviceTransferTraceCount() {
        return executionState == null ? 0 : executionState.hostDeviceTransferTraces().size();
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

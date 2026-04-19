package backend.runtime;

import config.runtime.RuntimeConfig;
import graph.execution.CompiledNodeExecutionMetadata;
import graph.execution.trace.ConvTraceMetadata;
import tensor.Tensor;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Objects;

public final class ExecutionContext {
    private final ExecutionMode mode;
    private final boolean useFastExpApprox;
    private final boolean useFastTanhApprox;
    private final Map<Tensor, CompiledNodeExecutionMetadata> metadataIndex;
    private final Map<Tensor, Object> runtimeStateIndex;
    private final Map<Tensor, ConvTraceMetadata> convTraceIndex;

    public ExecutionContext(ExecutionMode mode, boolean useFastExpApprox, boolean useFastTanhApprox) {
        this(mode, useFastExpApprox, useFastTanhApprox, Map.of());
    }

    public ExecutionContext(
            ExecutionMode mode,
            boolean useFastExpApprox,
            boolean useFastTanhApprox,
            Map<Tensor, CompiledNodeExecutionMetadata> metadataIndex
    ) {
        this.mode = Objects.requireNonNull(mode, "mode cannot be null");
        this.metadataIndex = Map.copyOf(metadataIndex == null ? Map.of() : metadataIndex);
        this.runtimeStateIndex = Collections.synchronizedMap(new IdentityHashMap<>());
        this.convTraceIndex = Collections.synchronizedMap(new IdentityHashMap<>());
        this.useFastExpApprox = useFastExpApprox;
        this.useFastTanhApprox = useFastTanhApprox;
    }

    public static ExecutionContext fromRuntimeConfig(RuntimeConfig runtimeConfig, ExecutionMode mode) {
        return fromRuntimeConfig(runtimeConfig, mode, Map.of());
    }

    public static ExecutionContext fromRuntimeConfig(
            RuntimeConfig runtimeConfig,
            ExecutionMode mode,
            Map<Tensor, CompiledNodeExecutionMetadata> metadataIndex
    ) {
        Objects.requireNonNull(runtimeConfig, "runtimeConfig cannot be null");
        Objects.requireNonNull(mode, "mode cannot be null");
        boolean backwardEnabled = mode == ExecutionMode.FORWARD_BACKWARD;
        return new ExecutionContext(
                mode,
                runtimeConfig.approximation().useFastExp(backwardEnabled),
                runtimeConfig.approximation().useFastTanh(backwardEnabled),
                metadataIndex
        );
    }

    public ExecutionMode mode() {
        return mode;
    }

    public boolean runsBackwardPass() {
        return mode == ExecutionMode.FORWARD_BACKWARD;
    }

    public boolean useFastExpApprox() {
        return useFastExpApprox;
    }

    public boolean useFastTanhApprox() {
        return useFastTanhApprox;
    }

    public CompiledNodeExecutionMetadata metadataFor(Tensor tensor) {
        return metadataIndex.get(tensor);
    }

    public <T> T runtimeStateFor(Tensor tensor, Class<T> type) {
        Objects.requireNonNull(type, "type cannot be null");
        Object state = runtimeStateIndex.get(tensor);
        return type.isInstance(state) ? type.cast(state) : null;
    }

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

    public ConvTraceMetadata convTraceFor(Tensor tensor) {
        return convTraceIndex.get(tensor);
    }

    public void publishConvTrace(Tensor tensor, ConvTraceMetadata trace) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        if (trace == null) {
            convTraceIndex.remove(tensor);
            return;
        }
        convTraceIndex.put(tensor, trace);
    }
}

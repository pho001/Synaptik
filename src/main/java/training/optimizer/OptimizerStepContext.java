package training.optimizer;

import backend.runtime.ExecutionContext;
import config.runtime.RuntimeConfig;
import graph.CompiledGradientBinding;
import graph.CompiledNode;
import graph.execution.trace.NativeOptimizerTrace;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Runtime view used by optimizers attached to prepared execution.
 */
public final class OptimizerStepContext {
    private final RuntimeConfig runtimeConfig;
    private final ExecutionContext executionContext;
    private final List<CompiledNode> allNodes;
    private final Map<Tensor, CompiledGradientBinding> gradientBindings;
    private final List<NativeOptimizerTrace> nativeOptimizerTraces;

    public OptimizerStepContext(
            RuntimeConfig runtimeConfig,
            ExecutionContext executionContext,
            List<CompiledNode> allNodes,
            Map<Tensor, CompiledGradientBinding> gradientBindings
    ) {
        this.runtimeConfig = Objects.requireNonNull(runtimeConfig, "runtimeConfig cannot be null");
        this.executionContext = Objects.requireNonNull(executionContext, "executionContext cannot be null");
        this.allNodes = List.copyOf(allNodes == null ? List.of() : allNodes);
        this.gradientBindings = Map.copyOf(gradientBindings == null ? Map.of() : gradientBindings);
        this.nativeOptimizerTraces = new ArrayList<>();
    }

    public RuntimeConfig runtimeConfig() {
        return runtimeConfig;
    }

    public ExecutionContext executionContext() {
        return executionContext;
    }

    public List<CompiledNode> allNodes() {
        return allNodes;
    }

    public Map<Tensor, CompiledGradientBinding> gradientBindings() {
        return gradientBindings;
    }

    public List<TrainableParameterRef> trainableParameters() {
        return allNodes.stream()
                .filter(node -> !node.backwardNode())
                .filter(CompiledNode::trainableParameter)
                .map(node -> {
                    CompiledGradientBinding binding = gradientBindings.get(node.sourceTensor());
                    return binding == null ? null : new TrainableParameterRef(node, binding);
                })
                .filter(Objects::nonNull)
                .toList();
    }

    public void recordNativeOptimizerTrace(NativeOptimizerTrace trace) {
        if (trace != null) {
            nativeOptimizerTraces.add(trace);
        }
    }

    public List<NativeOptimizerTrace> nativeOptimizerTraces() {
        return List.copyOf(nativeOptimizerTraces);
    }
}

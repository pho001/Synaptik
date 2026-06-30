package training.optimizer;

import runtime.execution.ExecutionContext;
import config.runtime.RuntimeConfig;
import graph.model.CompiledNode;
import graph.compile.publication.PublicationPlan;
import graph.execution.PublicationPolicy;
import trace.execution.NativeOptimizerTrace;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Runtime view used by optimizers attached to prepared execution.
 */
public final class OptimizerStepContext {
    private final RuntimeConfig runtimeConfig;
    private final ExecutionContext executionContext;
    private final PublicationPolicy publicationPolicy;
    private final List<CompiledNode> allNodes;
    private final PublicationPlan publicationPlan;
    private final List<NativeOptimizerTrace> nativeOptimizerTraces;

    public OptimizerStepContext(
            RuntimeConfig runtimeConfig,
            ExecutionContext executionContext,
            PublicationPolicy publicationPolicy,
            List<CompiledNode> allNodes,
            PublicationPlan publicationPlan
    ) {
        this.runtimeConfig = Objects.requireNonNull(runtimeConfig, "runtimeConfig cannot be null");
        this.executionContext = Objects.requireNonNull(executionContext, "executionContext cannot be null");
        this.publicationPolicy = publicationPolicy == null
                ? PublicationPolicy.defaultOptimizerStep()
                : publicationPolicy;
        this.allNodes = List.copyOf(allNodes == null ? List.of() : allNodes);
        this.publicationPlan = Objects.requireNonNull(publicationPlan, "publicationPlan cannot be null");
        this.nativeOptimizerTraces = new ArrayList<>();
    }

    public RuntimeConfig runtimeConfig() {
        return runtimeConfig;
    }

    public ExecutionContext executionContext() {
        return executionContext;
    }

    public PublicationPolicy publicationPolicy() {
        return publicationPolicy;
    }

    public List<CompiledNode> allNodes() {
        return allNodes;
    }

    public PublicationPlan publicationPlan() {
        return publicationPlan;
    }

    public List<TrainableParameterRef> trainableParameters() {
        return publicationPlan.trainableParameters().stream()
                .map(binding -> new TrainableParameterRef(
                        allNodes.get(binding.parameterNodeId()),
                        binding.parameterTensor(),
                        binding.gradientBinding()
                ))
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

package graph.compile;

import backend.partition.BackendPartitionDescriptorRegistry;
import config.compile.BackendPlanningConfig;
import graph.CompiledGradientBinding;
import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Inputs for compile-time backend ownership planning.
 */
public record BackendPlanningRequest(
        BackendPlanningConfig config,
        boolean supportsBackward,
        List<CompiledNode> compiledNodes,
        CompiledTensorDescriptorIndex descriptorIndex,
        CompiledNode forwardOutput,
        Map<?, CompiledGradientBinding> gradientBindings,
        BackendPartitionDescriptorRegistry backendPartitionDescriptors
) {
    public BackendPlanningRequest {
        config = config == null ? BackendPlanningConfig.cpuOnly() : config;
        compiledNodes = List.copyOf(compiledNodes == null ? List.of() : compiledNodes);
        descriptorIndex = Objects.requireNonNull(descriptorIndex, "descriptorIndex cannot be null");
        gradientBindings = Map.copyOf(gradientBindings == null ? Map.of() : gradientBindings);
        backendPartitionDescriptors = backendPartitionDescriptors == null
                ? BackendPartitionDescriptorRegistry.defaults()
                : backendPartitionDescriptors;
    }
}

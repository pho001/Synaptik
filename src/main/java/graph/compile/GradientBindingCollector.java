package graph.compile;

import graph.CompiledGradientBinding;
import graph.CompiledNode;
import tensor.Tensor;

import java.util.IdentityHashMap;
import java.util.Map;

/**
 * Captures how compiled gradient values are published back to source tensors.
 */
public final class GradientBindingCollector {
    private GradientBindingCollector() {
    }

    /**
     * Captures gradient bindings for every tensor in a compiled graph that has a gradient.
     *
     * @param graph compiled graph tensors
     * @param sourceTensors semantic-to-source tensor mapping
     * @param compiledNodeByTensor compiled node lookup by tensor
     * @return bindings keyed by published source tensor
     */
    public static Map<Tensor, CompiledGradientBinding> captureCompiledGradients(
            Iterable<Tensor> graph,
            Map<Tensor, Tensor> sourceTensors,
            Map<Tensor, CompiledNode> compiledNodeByTensor
    ) {
        IdentityHashMap<Tensor, CompiledGradientBinding> out = new IdentityHashMap<>();
        if (graph == null) {
            return Map.of();
        }
        Map<Tensor, Tensor> sources = sourceTensors == null ? Map.of() : sourceTensors;
        Map<Tensor, CompiledNode> compiledNodes = compiledNodeByTensor == null ? Map.of() : compiledNodeByTensor;
        for (Tensor tensor : graph) {
            Tensor gradient = tensor.getGradient();
            if (gradient == null) {
                continue;
            }
            Tensor publishedTensor = sources.getOrDefault(tensor, tensor);
            CompiledNode gradientNode = compiledNodes.get(gradient);
            if (gradientNode != null) {
                out.put(publishedTensor, CompiledGradientBinding.node(gradientNode.id()));
                continue;
            }
            if (gradient.getOperation() == null) {
                out.put(publishedTensor, CompiledGradientBinding.constant(gradient));
                continue;
            }
            throw new IllegalStateException("Gradient binding for tensor '" + publishedTensor.getLabel()
                    + "' does not resolve to a compiled node or constant.");
        }
        if (out.isEmpty()) {
            return Map.of();
        }
        return Map.copyOf(out);
    }

    /**
     * Captures the root gradient binding used to seed backward execution.
     *
     * @param forwardRoot forward root tensor
     * @param compiledNodeByTensor compiled node lookup by tensor
     * @return gradient binding, or {@code null} when no seed gradient exists
     */
    public static CompiledGradientBinding captureForwardSeedGradient(
            Tensor forwardRoot,
            Map<Tensor, CompiledNode> compiledNodeByTensor
    ) {
        Tensor gradient = forwardRoot == null ? null : forwardRoot.getGradient();
        if (gradient == null) {
            return null;
        }
        Map<Tensor, CompiledNode> compiledNodes = compiledNodeByTensor == null ? Map.of() : compiledNodeByTensor;
        CompiledNode gradientNode = compiledNodes.get(gradient);
        if (gradientNode != null) {
            return CompiledGradientBinding.node(gradientNode.id());
        }
        if (gradient.getOperation() == null) {
            return CompiledGradientBinding.constant(gradient);
        }
        throw new IllegalStateException("Forward seed gradient does not resolve to a compiled node or constant.");
    }
}

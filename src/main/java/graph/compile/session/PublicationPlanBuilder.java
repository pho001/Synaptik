package graph.compile.session;

import graph.CompiledGradientBinding;
import graph.CompiledNode;
import graph.compile.GraphStructureContract;
import graph.compile.publication.PublicationPlan;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Builds runtime publication bindings from compiled-node snapshots and semantic publication mappings.
 */
final class PublicationPlanBuilder {
    private PublicationPlanBuilder() {
    }

    static PublicationPlan build(
            Tensor rootTensor,
            GraphStructureContract graphContract,
            List<Tensor> graph,
            Map<Tensor, CompiledNode> compiledNodeByTensor,
            Map<Tensor, CompiledGradientBinding> compiledGradients,
            CompiledGradientBinding forwardSeedGradient,
            CompiledNode compiledForwardOutput,
            Tensor forwardOutput,
            int forwardBoundaryNodeId,
            Map<Tensor, Tensor> publicationTensors
    ) {
        Objects.requireNonNull(rootTensor, "rootTensor cannot be null");
        Objects.requireNonNull(compiledForwardOutput, "compiledForwardOutput cannot be null");
        Tensor actualForwardRoot = BackwardGraphCompiler.requireForwardRoot(forwardOutput);
        IdentityHashMap<Tensor, Tensor> sources = new IdentityHashMap<>();
        if (publicationTensors != null) {
            sources.putAll(publicationTensors);
        }
        mapComputedForwardRootForPublish(rootTensor, actualForwardRoot, sources);

        ArrayList<PublicationPlan.RuntimeInputBinding> runtimeInputs = new ArrayList<>();
        ArrayList<PublicationPlan.ForwardPublicationBinding> forwardPublications = new ArrayList<>();
        ArrayList<PublicationPlan.GradientPublicationBinding> gradientPublications = new ArrayList<>();
        ArrayList<PublicationPlan.TrainableParameterBinding> trainableParameters = new ArrayList<>();
        IdentityHashMap<Tensor, Boolean> gradientClearTargets = new IdentityHashMap<>();

        Map<Tensor, CompiledNode> compiledNodes = compiledNodeByTensor == null ? Map.of() : compiledNodeByTensor;
        Map<Tensor, CompiledGradientBinding> gradients = compiledGradients == null ? Map.of() : compiledGradients;
        for (Tensor tensor : List.copyOf(graph == null ? List.of() : graph)) {
            CompiledNode node = compiledNodes.get(tensor);
            if (node == null) {
                continue;
            }
            Tensor publicationTarget = sources.getOrDefault(tensor, tensor);
            if (node.leaf()) {
                runtimeInputs.add(new PublicationPlan.RuntimeInputBinding(
                        node.id(),
                        publicationTarget,
                        runtimeInputKind(node, forwardBoundaryNodeId)
                ));
            }
            if (node.backwardNode()) {
                continue;
            }
            forwardPublications.add(new PublicationPlan.ForwardPublicationBinding(
                    publicationTarget,
                    node.id(),
                    PublicationPlan.PublicationKind.FORWARD_VALUE,
                    PublicationPlan.aliasRepairChainFor(publicationTarget)
            ));
            gradientClearTargets.put(publicationTarget, Boolean.TRUE);
            CompiledGradientBinding gradientBinding = gradients.get(publicationTarget);
            if (node.trainableParameter() && gradientBinding != null) {
                trainableParameters.add(new PublicationPlan.TrainableParameterBinding(
                        publicationTarget,
                        node.id(),
                        gradientBinding
                ));
            }
        }

        for (Map.Entry<Tensor, CompiledGradientBinding> entry : gradients.entrySet()) {
            gradientPublications.add(new PublicationPlan.GradientPublicationBinding(entry.getKey(), entry.getValue()));
            gradientClearTargets.put(entry.getKey(), Boolean.TRUE);
        }

        return new PublicationPlan(
                rootTensor,
                graphContract,
                runtimeInputs,
                new PublicationPlan.ForwardPublicationBinding(
                        rootTensor,
                        rootOutputSourceNodeId(actualForwardRoot, compiledNodes, compiledForwardOutput),
                        PublicationPlan.PublicationKind.ROOT_OUTPUT,
                        PublicationPlan.aliasRepairChainFor(rootTensor)
                ),
                forwardPublications,
                gradientPublications,
                new ArrayList<>(gradientClearTargets.keySet()),
                forwardSeedGradient,
                trainableParameters
        );
    }

    private static void mapComputedForwardRootForPublish(
            Tensor rootTensor,
            Tensor actualForwardRoot,
            Map<Tensor, Tensor> publicationTensors
    ) {
        if (actualForwardRoot.getOperation() != null) {
            publicationTensors.put(actualForwardRoot, rootTensor);
        }
    }

    private static PublicationPlan.RuntimeInputBindingKind runtimeInputKind(
            CompiledNode node,
            int forwardBoundaryNodeId
    ) {
        if (node.id() <= forwardBoundaryNodeId) {
            return PublicationPlan.RuntimeInputBindingKind.FORWARD_LEAF_ALIAS;
        }
        return node.backwardNode()
                ? PublicationPlan.RuntimeInputBindingKind.BACKWARD_LEAF_COPY
                : PublicationPlan.RuntimeInputBindingKind.STATIC_LEAF_COPY;
    }

    private static int rootOutputSourceNodeId(
            Tensor actualForwardRoot,
            Map<Tensor, CompiledNode> compiledNodeByTensor,
            CompiledNode compiledForwardOutput
    ) {
        CompiledNode actualRootNode = compiledNodeByTensor.get(actualForwardRoot);
        if (actualRootNode != null) {
            return actualRootNode.id();
        }
        return compiledForwardOutput.id();
    }
}

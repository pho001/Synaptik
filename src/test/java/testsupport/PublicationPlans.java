package testsupport;

import graph.model.CompiledNode;
import graph.compile.GraphStructureContract;
import graph.compile.publication.PublicationPlan;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class PublicationPlans {
    private PublicationPlans() {
    }

    public static PublicationPlan forRoot(Tensor root, List<CompiledNode> nodes, int forwardOutputNodeId) {
        Objects.requireNonNull(root, "root cannot be null");
        List<CompiledNode> compiledNodes = nodes == null ? List.of() : nodes;
        List<Tensor> graph = root.topologicalSort();
        ArrayList<PublicationPlan.RuntimeInputBinding> runtimeInputs = new ArrayList<>();
        ArrayList<PublicationPlan.ForwardPublicationBinding> forwardPublications = new ArrayList<>();
        ArrayList<Tensor> gradientClearTargets = new ArrayList<>();

        for (CompiledNode node : compiledNodes) {
            Tensor tensor = node.id() < graph.size() ? graph.get(node.id()) : root;
            if (node.leaf()) {
                runtimeInputs.add(new PublicationPlan.RuntimeInputBinding(
                        node.id(),
                        tensor,
                        node.id() <= forwardOutputNodeId
                                ? PublicationPlan.RuntimeInputBindingKind.FORWARD_LEAF_ALIAS
                                : PublicationPlan.RuntimeInputBindingKind.STATIC_LEAF_COPY
                ));
            }
            if (!node.backwardNode()) {
                forwardPublications.add(new PublicationPlan.ForwardPublicationBinding(
                        tensor,
                        node.id(),
                        PublicationPlan.PublicationKind.FORWARD_VALUE,
                        PublicationPlan.aliasRepairChainFor(tensor)
                ));
                gradientClearTargets.add(tensor);
            }
        }

        return new PublicationPlan(
                root,
                GraphStructureContract.unchecked(),
                runtimeInputs,
                new PublicationPlan.ForwardPublicationBinding(
                        root,
                        forwardOutputNodeId,
                        PublicationPlan.PublicationKind.ROOT_OUTPUT,
                        PublicationPlan.aliasRepairChainFor(root)
                ),
                forwardPublications,
                List.of(),
                gradientClearTargets,
                null,
                List.of()
        );
    }
}

package graph.optimizer.simplify;

import graph.optimizer.cost.CostComponent;
import graph.optimizer.cost.CostScore;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

/**
 * Backend-neutral structural cost used to accept simplification fixpoint iterations.
 */
public record GraphOptimizationScore(
        int weightedOperationCost,
        int nodeCount,
        int edgeCount
) implements Comparable<GraphOptimizationScore> {
    private static final String COST_MODEL_NAME = "GraphSimplificationCostModel";
    private static final String COST_INPUT_KIND = "optimizer-simplification-graph";

    public static GraphOptimizationScore capture(List<Tensor> graph) {
        int weightedCost = 0;
        int edges = 0;
        for (Tensor tensor : graph) {
            Operation operation = tensor.getOperation();
            weightedCost += operationCost(operation);
            List<Tensor> inputs = tensor.getPrevTensors();
            if (inputs != null) {
                edges += inputs.size();
            }
        }
        return new GraphOptimizationScore(weightedCost, graph.size(), edges);
    }

    @Override
    public int compareTo(GraphOptimizationScore other) {
        int byCost = Integer.compare(weightedOperationCost, other.weightedOperationCost);
        if (byCost != 0) {
            return byCost;
        }
        int byNodes = Integer.compare(nodeCount, other.nodeCount);
        if (byNodes != 0) {
            return byNodes;
        }
        return Integer.compare(edgeCount, other.edgeCount);
    }

    /**
     * Exports this structural simplification score through the shared cost vocabulary.
     *
     * <p>The simplification fixpoint still uses {@link #compareTo(GraphOptimizationScore)} as its
     * source of truth. This method is report-only.</p>
     *
     * @return shared cost score explanation input
     */
    public CostScore toCostScore() {
        return CostScore.of(
                COST_MODEL_NAME,
                COST_INPUT_KIND,
                List.of(
                        CostComponent.lowerIsBetter(
                                "weightedOperationCost",
                                weightedOperationCost,
                                "lexicographic simplification priority: lower structural operation cost is better"
                        ),
                        CostComponent.lowerIsBetter(
                                "nodeCount",
                                nodeCount,
                                "simplification should reduce live graph nodes when operation cost ties"
                        ),
                        CostComponent.lowerIsBetter(
                                "edgeCount",
                                edgeCount,
                                "simplification should reduce graph edges when operation cost and node count tie"
                        )
                )
        );
    }

    private static int operationCost(Operation operation) {
        if (operation == null) {
            return 0;
        }
        int cost = metadataCost(operation.computationalCost());
        cost = applyResultKindPolicy(cost, operation.resultKind());
        cost = applySemanticPolicy(cost, operation.semanticFamily());
        return applyCategoryPolicy(cost, operation.arityClass());
    }

    private static int metadataCost(Operation.OpComputationalCost cost) {
        if (cost == null) {
            return 12;
        }
        return switch (cost) {
            case TRIVIAL -> 1;
            case CHEAP -> 2;
            case MEDIUM -> 4;
            case EXPENSIVE -> 4;
            case UNKNOWN -> 12;
        };
    }

    private static int applyResultKindPolicy(int cost, Operation.OpResultKind resultKind) {
        if (resultKind == Operation.OpResultKind.SHAPE_VIEW) {
            return Math.min(cost, 4);
        }
        return cost;
    }

    private static int applySemanticPolicy(int cost, Operation.OpSemanticFamily family) {
        if (family == null) {
            return Math.max(cost, 12);
        }
        return switch (family) {
            case LAYOUT -> Math.min(cost, 4);
            case REDUCTION -> Math.max(cost, 8);
            case LINEAR_ALGEBRA -> Math.max(cost, 16);
            case SPECIAL -> heavierNonTrivialCost(cost, 12);
            case FUSED -> 32;
            case UNKNOWN -> Math.max(cost, 12);
            case ARITHMETIC, TRANSCENDENTAL, COMPARISON, LOGICAL, SELECTION -> cost;
        };
    }

    private static int applyCategoryPolicy(int cost, Operation.OpArityClass category) {
        if (category == null) {
            return Math.max(cost, 12);
        }
        return switch (category) {
            case LAYOUT -> Math.min(cost, 4);
            case REDUCTION -> Math.max(cost, 8);
            case LINEAR_ALGEBRA -> Math.max(cost, 16);
            case SPECIAL -> heavierNonTrivialCost(cost, 12);
            case FUSED -> 32;
            case ELEMENT_WISE -> cost;
        };
    }

    private static int heavierNonTrivialCost(int cost, int floor) {
        return cost <= 2 ? cost : Math.max(cost, floor);
    }
}

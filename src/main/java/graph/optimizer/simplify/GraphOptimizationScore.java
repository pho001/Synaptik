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
        return switch (operation.opType()) {
            case NOOP, RESHAPE, EXPAND, SELECT, PERMUTE, EXPAND_DIMS, SQUEEZE, CONTIGUOUS -> 1;
            case ADD, SUB, MUL, DIV, MIN, MAX, GT, GE, LT, LE, EQ, NE,
                 LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT, WHERE, NEG, INV, ABS,
                 FLOOR, CEIL, SIGN, MUL_SCALAR, RELU, CLAMP_MIN, CLAMP_MAX -> 2;
            case SQRT, LOG, EXP, FAST_EXP, ERF, TANH, FAST_TANH, POW, SIGMOID -> 4;
            case SUM, MEAN, REDUCE_MIN, REDUCE_MAX, REDUCE_ALL, REDUCE_ANY,
                 SOFTMAX, SOFTMAX_GRAD, LOG_SOFTMAX, LOG_SOFTMAX_GRAD -> 8;
            case MATMUL, LINEAR -> 16;
            case CONV2D, CONV2D_GEMM, CONV2D_BACKWARD_INPUT, CONV2D_BACKWARD_WEIGHT,
                 CONV2D_BACKWARD_INPUT_GEMM, CONV2D_BACKWARD_WEIGHT_GEMM,
                 SCALED_DOT_PRODUCT_ATTENTION, SCALED_DOT_PRODUCT_ATTENTION_BACKWARD,
                 SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS -> 24;
            case FUSED -> 32;
            default -> 12;
        };
    }
}

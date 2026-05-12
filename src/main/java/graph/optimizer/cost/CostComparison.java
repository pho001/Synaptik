package graph.optimizer.cost;

/**
 * Generic comparison result for two scores from the same model family.
 */
public enum CostComparison {
    IMPROVED,
    UNCHANGED,
    WORSE,
    INCOMPARABLE
}

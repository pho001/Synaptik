package graph.optimizer.cost;

import graph.optimizer.simplify.GraphOptimizationScore;
import planning.partition.cost.AcceleratorPartitionScoreModel;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CostScoreTest {
    @Test
    void lexicographicComparisonUsesComponentDirections() {
        CostScore previous = CostScore.of(
                "GraphSimplificationCostModel",
                "optimizer-simplification-graph",
                List.of(
                        CostComponent.lowerIsBetter("weightedOperationCost", 10.0d, "cost"),
                        CostComponent.lowerIsBetter("nodeCount", 5.0d, "nodes")
                )
        );
        CostScore improved = CostScore.of(
                "GraphSimplificationCostModel",
                "optimizer-simplification-graph",
                List.of(
                        CostComponent.lowerIsBetter("weightedOperationCost", 8.0d, "cost"),
                        CostComponent.lowerIsBetter("nodeCount", 7.0d, "nodes")
                )
        );

        assertEquals(CostComparison.IMPROVED, improved.compare(previous));
        assertEquals(CostComparison.WORSE, previous.compare(improved));
    }

    @Test
    void incompatibleModelsDoNotCompareAcrossSpecializedScores() {
        CostScore simplification = CostScore.of(
                "GraphSimplificationCostModel",
                "optimizer-simplification-graph",
                List.of(CostComponent.lowerIsBetter("nodeCount", 3.0d, "nodes"))
        );
        CostScore accelerator = CostScore.of(
                "AcceleratorPartitionCostModel",
                "accelerator-partition-materialization",
                List.of(CostComponent.higherIsBetter("finalScore", 100.0d, "score"))
        );

        assertEquals(CostComparison.INCOMPARABLE, simplification.compare(accelerator));
    }

    @Test
    void graphOptimizationScoreExportsSimplificationVocabularyWithoutChangingOrdering() {
        GraphOptimizationScore previous = new GraphOptimizationScore(12, 5, 4);
        GraphOptimizationScore improved = new GraphOptimizationScore(10, 10, 10);

        assertTrue(improved.compareTo(previous) < 0);
        assertEquals(CostComparison.IMPROVED, improved.toCostScore().compare(previous.toCostScore()));

        CostExplanation explanation = improved.toCostScore().explain("simplification-fixpoint-improved");
        assertEquals("GraphSimplificationCostModel", explanation.modelName());
        assertEquals("optimizer-simplification-graph", explanation.inputKind());
        assertEquals("simplification-fixpoint-improved", explanation.reasonCode());
        assertTrue(explanation.rawComponents().stream().anyMatch(component -> component.name().equals("weightedOperationCost")));
    }

    @Test
    void acceleratorMaterializationSummaryExportsReportOnlyVocabulary() {
        var summary = new AcceleratorPartitionScoreModel.MaterializationCostSummary(
                "PROFILE_DERIVED",
                2,
                128L,
                64L,
                4096L,
                512L,
                125.0d,
                1024.0d,
                "accepted-static-profitable",
                "BUFFER_BINDING",
                "DENSE"
        );

        CostScore score = summary.toCostScore();
        CostExplanation explanation = score.explain(summary.reasonCode());

        assertEquals("AcceleratorPartitionCostModel", score.modelName());
        assertEquals("accelerator-partition-materialization", score.inputKind());
        assertEquals("accepted-static-profitable", explanation.reasonCode());
        assertFalse(explanation.topContributors().isEmpty());
        assertTrue(score.components().stream().anyMatch(component ->
                component.name().equals("finalScore") && component.direction() == CostDirection.HIGHER_IS_BETTER));
        assertTrue(score.components().stream().anyMatch(component ->
                component.name().equals("estimatedTransferBytes") && component.direction() == CostDirection.LOWER_IS_BETTER));
        assertTrue(score.components().stream().anyMatch(component ->
                component.name().equals("preset") && component.reason().equals("PROFILE_DERIVED")));
    }
}

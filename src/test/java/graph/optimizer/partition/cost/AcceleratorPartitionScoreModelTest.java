package graph.optimizer.partition.cost;

import config.optimizer.MetalTransferModel;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AcceleratorPartitionScoreModelTest {
    @Test
    void transferAwareScorePenalizesBoundaryCopies() {
        var metrics = new AcceleratorPartitionScoreModel.CandidateMetrics(3, 2, 2, 1, 2);
        var planner = new AcceleratorPartitionScoreModel.PlannerPolicy(
                16,
                512,
                10.0,
                5.0,
                2.0,
                1.0,
                1.0,
                1.0
        );
        var transferPolicy = new AcceleratorPartitionScoreModel.TransferPolicy(0.5, 1.0, 0.0);

        double noTransfer = AcceleratorPartitionScoreModel.acceptedScore(
                metrics,
                1000L,
                AcceleratorPartitionScoreModel.TransferMetrics.none(),
                planner,
                transferPolicy
        );
        double withTransfer = AcceleratorPartitionScoreModel.acceptedScore(
                metrics,
                1000L,
                new AcceleratorPartitionScoreModel.TransferMetrics(100L, 200L, 0L),
                planner,
                transferPolicy
        );

        assertEquals(250.0, noTransfer - withTransfer, 0.0001);
    }

    @Test
    void avoidedIntermediateBytesCanOffsetTransferCost() {
        var metrics = new AcceleratorPartitionScoreModel.CandidateMetrics(2, 1, 1, 0, 1);
        var planner = AcceleratorPartitionScoreModel.PlannerPolicy.defaults();
        var transferPolicy = new AcceleratorPartitionScoreModel.TransferPolicy(1.0, 1.0, 0.5);

        double smallRegion = AcceleratorPartitionScoreModel.acceptedScore(
                metrics,
                100L,
                new AcceleratorPartitionScoreModel.TransferMetrics(200L, 200L, 0L),
                planner,
                transferPolicy
        );
        double largerRegion = AcceleratorPartitionScoreModel.acceptedScore(
                metrics,
                100L,
                new AcceleratorPartitionScoreModel.TransferMetrics(200L, 200L, 1000L),
                planner,
                transferPolicy
        );

        assertTrue(largerRegion > smallRegion);
        assertEquals(500.0, largerRegion - smallRegion, 0.0001);
    }

    @Test
    void transferPolicyCanBeBuiltFromMetalTransferModel() {
        var conservative = AcceleratorPartitionScoreModel.TransferPolicy.fromMetalTransferModel(
                MetalTransferModel.CONSERVATIVE
        );
        var aggressive = AcceleratorPartitionScoreModel.TransferPolicy.fromMetalTransferModel(
                MetalTransferModel.AGGRESSIVE
        );

        assertTrue(aggressive.inputBytePenalty() < conservative.inputBytePenalty());
        assertTrue(aggressive.outputBytePenalty() < conservative.outputBytePenalty());
        assertTrue(aggressive.avoidedIntermediateByteCredit() > conservative.avoidedIntermediateByteCredit());
    }
}

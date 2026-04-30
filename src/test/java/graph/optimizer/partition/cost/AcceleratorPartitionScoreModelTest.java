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

    @Test
    void dispatchOverheadPenalizesTinyAcceleratorIslands() {
        var metrics = new AcceleratorPartitionScoreModel.CandidateMetrics(1, 0, 1, 0, 0);
        var planner = new AcceleratorPartitionScoreModel.PlannerPolicy(16, 512, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0);

        var summary = AcceleratorPartitionScoreModel.scoreMaterializationAware(
                metrics,
                50L,
                AcceleratorPartitionScoreModel.MaterializationSignals.none(),
                planner,
                new AcceleratorPartitionScoreModel.StaticCostPreset(
                        "TEST",
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        100.0,
                        1.0
                )
        );

        assertEquals("rejected-materialization-cost", summary.reasonCode());
        assertEquals(-40.0, summary.finalScore(), 0.0001);
        assertEquals(100.0, summary.dispatchCost(), 0.0001);
    }

    @Test
    void boundaryAndTransferBytesReduceStaticScore() {
        var metrics = new AcceleratorPartitionScoreModel.CandidateMetrics(3, 2, 2, 1, 2);
        var planner = new AcceleratorPartitionScoreModel.PlannerPolicy(16, 512, 10.0, 5.0, 2.0, 1.0, 1.0, 1.0);
        var preset = new AcceleratorPartitionScoreModel.StaticCostPreset(
                "TEST",
                10.0,
                0.5,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0
        );

        var noTransfer = AcceleratorPartitionScoreModel.scoreMaterializationAware(
                metrics,
                1000L,
                AcceleratorPartitionScoreModel.MaterializationSignals.none(),
                planner,
                preset
        );
        var withTransfer = AcceleratorPartitionScoreModel.scoreMaterializationAware(
                metrics,
                1000L,
                new AcceleratorPartitionScoreModel.MaterializationSignals(2, 100L, 200L, 0L, 0L, 0L, "BUFFER", "DENSE"),
                planner,
                preset
        );

        assertEquals(270.0, noTransfer.finalScore() - withTransfer.finalScore(), 0.0001);
        assertEquals(300L, withTransfer.estimatedTransferBytes());
        assertEquals(2, withTransfer.boundaryCount());
    }

    @Test
    void avoidedIntermediateBytesCanMakeLongerRegionWin() {
        var metrics = new AcceleratorPartitionScoreModel.CandidateMetrics(2, 1, 1, 0, 1);
        var planner = AcceleratorPartitionScoreModel.PlannerPolicy.defaults();
        var preset = new AcceleratorPartitionScoreModel.StaticCostPreset(
                "TEST",
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.5,
                0.0,
                1.0
        );

        var smallRegion = AcceleratorPartitionScoreModel.scoreMaterializationAware(
                metrics,
                100L,
                new AcceleratorPartitionScoreModel.MaterializationSignals(0, 200L, 200L, 0L, 0L, 0L, "BUFFER", "DENSE"),
                planner,
                preset
        );
        var largerRegion = AcceleratorPartitionScoreModel.scoreMaterializationAware(
                metrics,
                100L,
                new AcceleratorPartitionScoreModel.MaterializationSignals(0, 200L, 200L, 0L, 0L, 1000L, "BUFFER", "DENSE"),
                planner,
                preset
        );

        assertTrue(largerRegion.finalScore() > smallRegion.finalScore());
        assertEquals(500.0, largerRegion.finalScore() - smallRegion.finalScore(), 0.0001);
        assertEquals(1000L, largerRegion.avoidedIntermediateBytes());
    }

    @Test
    void layoutAndTensorArrayFallbackBytesCanRejectCandidate() {
        var metrics = new AcceleratorPartitionScoreModel.CandidateMetrics(1, 0, 1, 0, 0);
        var planner = new AcceleratorPartitionScoreModel.PlannerPolicy(16, 512, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        var preset = new AcceleratorPartitionScoreModel.StaticCostPreset(
                "TEST",
                0.0,
                0.0,
                0.0,
                1.0,
                2.0,
                0.0,
                0.0,
                1.0
        );

        var summary = AcceleratorPartitionScoreModel.scoreMaterializationAware(
                metrics,
                100L,
                new AcceleratorPartitionScoreModel.MaterializationSignals(
                        0,
                        0L,
                        0L,
                        75L,
                        25L,
                        0L,
                        "TENSOR_ARRAY",
                        "PERMUTED_OR_STRIDED_VIEW"
                ),
                planner,
                preset
        );

        assertEquals("rejected-materialization-cost", summary.reasonCode());
        assertEquals(-15.0, summary.finalScore(), 0.0001);
        assertEquals("TENSOR_ARRAY", summary.fallbackMode());
        assertEquals("PERMUTED_OR_STRIDED_VIEW", summary.layoutClass());
    }

    @Test
    void staticPresetsDoNotRequireProfileDerivedCosts() {
        var conservative = AcceleratorPartitionScoreModel.StaticCostPreset.conservative();
        var measured = AcceleratorPartitionScoreModel.StaticCostPreset.measured();
        var aggressive = AcceleratorPartitionScoreModel.StaticCostPreset.aggressive();

        assertEquals("CONSERVATIVE", conservative.name());
        assertEquals("MEASURED", measured.name());
        assertEquals("AGGRESSIVE", aggressive.name());
        assertTrue(aggressive.boundaryPenalty() < conservative.boundaryPenalty());
        assertTrue(aggressive.uploadBytePenalty() < conservative.uploadBytePenalty());
        assertTrue(aggressive.downloadBytePenalty() < conservative.downloadBytePenalty());
        assertTrue(aggressive.avoidedIntermediateByteCredit() > conservative.avoidedIntermediateByteCredit());
    }
}

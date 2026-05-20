package graph.compile.planning.partition.cost;

import backend.ComputeBackend;
import backend.accelerator.select.ProfileDerivedAcceleratorCostFactors;
import config.backend.CpuKernelConfig;
import config.runtime.AcceleratorBackendConfig;
import config.runtime.AcceleratorConfig;
import config.runtime.ApproximationConfig;
import config.runtime.BlasConfig;
import config.runtime.RuntimeConfig;
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

    @Test
    void profileDerivedCostFactorsUseRuntimeThresholds() {
        RuntimeConfig runtime = new RuntimeConfig(
                new CpuKernelConfig(1, 16, 16, 16, 1024, 1024, 2048),
                ApproximationConfig.defaults(),
                BlasConfig.disabled()
        ).withAccelerator(new AcceleratorConfig(
                AcceleratorBackendConfig.disabled(),
                AcceleratorBackendConfig.disabled(),
                new AcceleratorBackendConfig(true, false, 2_000_000L)
        ));

        var factors = ProfileDerivedAcceleratorCostFactors.fromRuntimeConfig(runtime, ComputeBackend.GPU_METAL);
        var conservative = AcceleratorPartitionScoreModel.StaticCostPreset.conservative();

        assertEquals("PROFILE_DERIVED", factors.presetName());
        assertEquals(2_000_000L, factors.minimumEstimatedWork());
        assertEquals(2048, factors.contiguousMaterializeThreshold());
        assertEquals(conservative.dispatchOverhead() + 2_000.0d, factors.dispatchOverhead(), 0.0001);
        assertEquals(conservative.uploadBytePenalty() * 2.0d, factors.uploadBytePenalty(), 0.0001);
        assertEquals(conservative.downloadBytePenalty() * 2.0d, factors.downloadBytePenalty(), 0.0001);
        assertEquals(conservative.layoutFallbackBytePenalty() * 2.0d, factors.layoutFallbackBytePenalty(), 0.0001);
    }

    @Test
    void profileDerivedPresetKeepsConservativeReasonCodes() {
        RuntimeConfig runtime = RuntimeConfig.trainingDefaults().withAccelerator(new AcceleratorConfig(
                AcceleratorBackendConfig.disabled(),
                AcceleratorBackendConfig.disabled(),
                new AcceleratorBackendConfig(true, false, 1_000_000L)
        ));
        var preset = ProfileDerivedAcceleratorCostFactors
                .fromRuntimeConfig(runtime, ComputeBackend.GPU_METAL)
                .toStaticCostPreset();
        var summary = AcceleratorPartitionScoreModel.scoreMaterializationAware(
                new AcceleratorPartitionScoreModel.CandidateMetrics(1, 0, 1, 0, 0),
                100L,
                AcceleratorPartitionScoreModel.MaterializationSignals.none(),
                new AcceleratorPartitionScoreModel.PlannerPolicy(16, 512, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                preset
        );

        assertEquals("PROFILE_DERIVED", preset.name());
        assertEquals("rejected-materialization-cost", summary.reasonCode());
        assertEquals("PROFILE_DERIVED", summary.preset());
    }
}

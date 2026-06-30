import org.junit.jupiter.api.Test;
import tuning.benchmark.report.GpuCoverageGateResult;
import tuning.benchmark.report.GpuCoverageGatePolicy;
import tuning.benchmark.report.GpuCoverageHotPathExpectation;
import tuning.benchmark.report.GpuCoverageRegressionGate;
import tuning.benchmark.report.GpuCoverageSummary;
import tuning.benchmark.report.GpuHotPathCoverageTargets;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class GpuCoverageRegressionGateTest {
    @Test
    void failsWhenGpuCoverageIsLost() {
        var summary = summary("GPU_METAL", coverage(0.25d, 1, 1, 0, 0, 1, 0));
        var policy = GpuCoverageGatePolicy.nativeBufferTarget("GPU_METAL", 0.5d, 3);

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> GpuCoverageRegressionGate.requirePass(summary, policy)
        );

        assertTrue(failure.getMessage().contains("lost GPU coverage"));
    }

    @Test
    void failsWhenUnexpectedCpuMaterializationAppears() {
        var summary = summary("GPU_METAL", coverage(0.75d, 3, 1, 0, 0, 0, 0));
        var policy = GpuCoverageGatePolicy.nativeBufferTarget("GPU_METAL", 0.5d, 3);

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> GpuCoverageRegressionGate.requirePass(summary, policy)
        );

        assertTrue(failure.getMessage().contains("unexpected CPU materialization"));
    }

    @Test
    void failsWhenTensorArrayFallbackIsHidden() {
        var summary = summary("GPU_METAL", coverage(0.75d, 3, 0, 1, 1, 0, 0));
        var policy = GpuCoverageGatePolicy.nativeBufferTarget("GPU_METAL", 0.5d, 3);

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> GpuCoverageRegressionGate.requirePass(summary, policy)
        );

        assertTrue(failure.getMessage().contains("hidden tensor-array fallback"));
    }

    @Test
    void failsWhenDeviceHandoffBudgetIsExceeded() {
        var summary = summary("GPU_METAL", coverage(0.75d, 3, 0, 0, 0, 3, 1));
        var policy = GpuCoverageGatePolicy.nativeBufferTarget("GPU_METAL", 0.5d, 3);

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> GpuCoverageRegressionGate.requirePass(summary, policy)
        );

        assertTrue(failure.getMessage().contains("unexpected device handoff"));
    }

    @Test
    void passesWhenCoverageMatchesPolicy() {
        var summary = summary("GPU_CUDA", coverage(0.75d, 3, 0, 0, 0, 1, 1));
        var policy = GpuCoverageGatePolicy.nativeBufferTarget("GPU_CUDA", 0.5d, 3);

        assertDoesNotThrow(() -> GpuCoverageRegressionGate.requirePass(summary, policy));
    }

    @Test
    void failsWhenNativeCopyStrategyRegresses() {
        var summary = summary("GPU_METAL", coverage(0.75d, 3, 0, 0, 0, 1, 1));
        var policy = GpuCoverageGatePolicy.nativeBufferTarget("GPU_METAL", 0.5d, 3)
                .withRequiredNativeCopyStrategy("TRUE_OUTPUT_BUFFER_WRITE");

        var result = GpuCoverageRegressionGate.evaluate(summary, policy);

        assertTrue(result.failures().contains("unexpected native copy strategy"));
    }

    @Test
    void failsWhenCoverageSummaryIsMissing() {
        var summary = new GpuCoverageSummary(Map.of());
        var policy = GpuCoverageGatePolicy.nativeBufferTarget("GPU_METAL", 0.5d, 3);

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> GpuCoverageRegressionGate.requirePass(summary, policy)
        );

        assertTrue(failure.getMessage().contains("missing coverage summary"));
    }

    @Test
    void phaseTwentyFailsWhenMultiOpRegionCoverageIsLost() {
        var summary = summary("GPU_METAL", coverage(0.75d, 3, 0, 0, 0, 1, 1, 0, 3, 1, 0));
        var policy = GpuCoverageGatePolicy.hotPathTarget("GPU_METAL", 0.5d, 3, 1, 3, 1);

        var result = GpuCoverageRegressionGate.evaluate(summary, policy);

        assertTrue(result.failures().contains("lost multi-op GPU region coverage"));
    }

    @Test
    void phaseTwentyFailsWhenLoweredPrimitiveCoverageIsLost() {
        var summary = summary("GPU_METAL", coverage(0.75d, 3, 0, 0, 0, 1, 1, 1, 2, 1, 0));
        var policy = GpuCoverageGatePolicy.hotPathTarget("GPU_METAL", 0.5d, 3, 1, 3, 1);

        var result = GpuCoverageRegressionGate.evaluate(summary, policy);

        assertTrue(result.failures().contains("lost lowered primitive coverage"));
    }

    @Test
    void phaseTwentyFailsWhenFusedSubpatternCoverageIsLost() {
        var summary = summary("GPU_METAL", coverage(0.75d, 3, 0, 0, 0, 1, 1, 1, 3, 0, 0));
        var policy = GpuCoverageGatePolicy.hotPathTarget("GPU_METAL", 0.5d, 3, 1, 3, 1);

        var result = GpuCoverageRegressionGate.evaluate(summary, policy);

        assertTrue(result.failures().contains("lost fused subpattern coverage"));
    }

    @Test
    void phaseTwentyFailureReasonsAreStableAndSpecific() {
        var summary = summary("GPU_METAL", coverage(0.25d, 1, 1, 1, 2, 3, 0, 0, 0, 0, 1));
        var policy = GpuCoverageGatePolicy.hotPathTarget("GPU_METAL", 0.5d, 3, 1, 1, 1);

        var result = GpuCoverageRegressionGate.evaluate(summary, policy);

        assertEquals(List.of(
                "lost GPU coverage",
                "lost GPU coverage",
                "lost multi-op GPU region coverage",
                "lost lowered primitive coverage",
                "lost fused subpattern coverage",
                "unexpected CPU materialization",
                "unexpected internal CPU materialization",
                "lost GPU coverage",
                "unexpected CPU fallback",
                "hidden tensor-array fallback",
                "lost native buffer binding",
                "unexpected device handoff"
        ), result.failures());

        var missing = GpuCoverageRegressionGate.evaluate(new GpuCoverageSummary(Map.of()), policy);
        assertEquals(List.of("missing coverage summary"), missing.failures());
    }

    @Test
    void phaseTwentyFiveSupportedSdpaPolicyFailsHiddenCpuOrTensorArrayExit() {
        var policy = new GpuCoverageGatePolicy(
                "GPU_METAL",
                0.1d,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                true
        );
        var summary = summary("GPU_METAL", coverage(0.5d, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1));

        var result = GpuCoverageRegressionGate.evaluate(summary, policy);

        assertTrue(result.failures().contains("unexpected CPU fallback"));
        assertTrue(result.failures().contains("hidden tensor-array fallback"));
        assertTrue(result.failures().contains("lost native buffer binding"));
    }

    @Test
    void phaseTwentyEightSupportedTargetPoliciesRejectHiddenFallbackModes() {
        for (String workload : List.of(
                "reduction_chain_small",
                "layer_norm_small",
                "rms_norm_small",
                "transformer_block_hot_path",
                "mlp_classifier_small"
        )) {
            GpuCoverageHotPathExpectation expectation = GpuHotPathCoverageTargets.defaultExpectations()
                    .stream()
                    .filter(item -> item.workloadName().equals(workload))
                    .findFirst()
                    .orElseThrow();

            var hiddenTensorArray = summary("GPU_METAL", coverage(0.75d, 3, 0, 1, 1, 1, 0, 1, 3, 1, 0));
            var tensorArrayResult = GpuCoverageRegressionGate.evaluate(hiddenTensorArray, expectation.policy());
            assertTrue(tensorArrayResult.failures().contains("hidden tensor-array fallback"), workload);
            assertTrue(tensorArrayResult.failures().contains("lost native buffer binding"), workload);

            var cpuFallback = summary("GPU_METAL", coverage(0.75d, 3, 0, 0, 1, 1, 1, 1, 3, 1, 1));
            var cpuFallbackResult = GpuCoverageRegressionGate.evaluate(cpuFallback, expectation.policy());
            assertTrue(cpuFallbackResult.failures().contains("unexpected CPU fallback"), workload);

            var cpuMaterialization = summary("GPU_METAL", coverage(0.75d, 3, 1, 0, 0, 1, 1, 1, 3, 1, 0));
            var materializationResult = GpuCoverageRegressionGate.evaluate(cpuMaterialization, expectation.policy());
            assertTrue(materializationResult.failures().contains("unexpected CPU materialization"), workload);
        }
    }

    @Test
    void phaseTwentyEightSupportedTargetPoliciesRejectLostPrimitiveOrRegionEvidence() {
        GpuCoverageHotPathExpectation reduction = GpuHotPathCoverageTargets.defaultExpectations()
                .stream()
                .filter(item -> item.workloadName().equals("reduction_chain_small"))
                .findFirst()
                .orElseThrow();
        var summary = summary("GPU_METAL", coverage(0.75d, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0));

        var result = GpuCoverageRegressionGate.evaluate(summary, reduction.policy());

        assertTrue(result.failures().contains("lost GPU coverage"));
        assertTrue(result.failures().contains("lost lowered primitive coverage"));
    }

    @Test
    void phaseTwentyEightMissingTargetCoverageNamesWorkloadAndBackend() {
        GpuCoverageHotPathExpectation expectation = GpuHotPathCoverageTargets.defaultExpectations().getFirst();

        var results = GpuCoverageRegressionGate.evaluateTargets(
                new tuning.benchmark.report.BenchmarkSuiteReport(null, List.of()),
                List.of(expectation)
        );

        assertEquals(1, results.size());
        assertTrue(results.getFirst().failures().getFirst().contains(expectation.workloadName()));
        assertTrue(results.getFirst().failures().getFirst().contains(expectation.backend()));
    }

    @Test
    void phaseThirtyBf16TargetsRequireDTypeResidencyEvidence() {
        GpuCoverageHotPathExpectation expectation = GpuHotPathCoverageTargets.defaultExpectations()
                .stream()
                .filter(item -> item.workloadName().equals("mlp_classifier_small_bf16"))
                .findFirst()
                .orElseThrow();
        var reportWithoutDTypeEvidence = reportFor(
                "mlp_classifier_small_bf16",
                coverage(0.75d, 3, 0, 0, 0, 1, 1, 1, 3, 1, 0)
        );
        var reportWithDTypeEvidence = reportFor(
                "mlp_classifier_small_bf16",
                coverageWithDTypeEvidence(0.75d, 3, 1, 3, 1, tensor.DataType.BFLOAT16)
        );

        List<GpuCoverageGateResult> missing = GpuCoverageRegressionGate.evaluateTargets(
                reportWithoutDTypeEvidence,
                List.of(expectation)
        );
        List<GpuCoverageGateResult> passing = GpuCoverageRegressionGate.evaluateTargets(
                reportWithDTypeEvidence,
                List.of(expectation)
        );

        assertTrue(missing.getFirst().failures().stream()
                .anyMatch(failure -> failure.contains("missing BF16 dtype residency evidence")));
        assertTrue(passing.getFirst().passed(), passing.getFirst().failures().toString());
    }

    @Test
    void phaseThirtyOneBoolCompareWhereRequiresBoolDTypeResidencyEvidence() {
        GpuCoverageHotPathExpectation expectation = GpuHotPathCoverageTargets.defaultExpectations()
                .stream()
                .filter(item -> item.workloadName().equals("bool_compare_where_small"))
                .findFirst()
                .orElseThrow();
        var reportWithoutDTypeEvidence = reportFor(
                "bool_compare_where_small",
                coverage(1.0d, 4, 0, 0, 0, 0, 1, 1, 4, 0, 0)
        );
        var reportWithDTypeEvidence = reportFor(
                "bool_compare_where_small",
                coverageWithDTypeEvidence(1.0d, 4, 1, 4, 0, tensor.DataType.BOOL)
        );

        List<GpuCoverageGateResult> missing = GpuCoverageRegressionGate.evaluateTargets(
                reportWithoutDTypeEvidence,
                List.of(expectation)
        );
        List<GpuCoverageGateResult> passing = GpuCoverageRegressionGate.evaluateTargets(
                reportWithDTypeEvidence,
                List.of(expectation)
        );

        assertTrue(missing.getFirst().failures().stream()
                .anyMatch(failure -> failure.contains("missing BOOL dtype residency evidence")));
        assertTrue(passing.getFirst().passed(), passing.getFirst().failures().toString());
    }

    @Test
    void phaseThirtyTwoGatherTakeRequiresInt32IndexDTypeResidencyEvidence() {
        GpuCoverageHotPathExpectation expectation = GpuHotPathCoverageTargets.defaultExpectations()
                .stream()
                .filter(item -> item.workloadName().equals("gather_take_small"))
                .findFirst()
                .orElseThrow();
        var reportWithoutDTypeEvidence = reportFor(
                "gather_take_small",
                coverage(1.0d, 4, 0, 0, 0, 0, 1, 1, 4, 0, 0)
        );
        var reportWithDTypeEvidence = reportFor(
                "gather_take_small",
                coverageWithDTypeEvidence(1.0d, 4, 1, 4, 0, tensor.DataType.INT32)
        );

        List<GpuCoverageGateResult> missing = GpuCoverageRegressionGate.evaluateTargets(
                reportWithoutDTypeEvidence,
                List.of(expectation)
        );
        List<GpuCoverageGateResult> passing = GpuCoverageRegressionGate.evaluateTargets(
                reportWithDTypeEvidence,
                List.of(expectation)
        );

        assertTrue(missing.getFirst().failures().stream()
                .anyMatch(failure -> failure.contains("missing INT32 index dtype residency evidence")));
        assertTrue(passing.getFirst().passed(), passing.getFirst().failures().toString());
    }

    @Test
    void phaseThirtyThreeLayoutRepairRequiresGpuLayoutMaterializationEvidence() {
        GpuCoverageHotPathExpectation expectation = GpuHotPathCoverageTargets.defaultExpectations()
                .stream()
                .filter(item -> item.workloadName().equals("layout_broadcast_repair_small"))
                .findFirst()
                .orElseThrow();
        var reportWithoutLayoutEvidence = reportFor(
                "layout_broadcast_repair_small",
                coverage(1.0d, 2, 0, 0, 0, 0, 1, 0, 1, 0, 0)
        );
        var reportWithLayoutEvidence = reportFor(
                "layout_broadcast_repair_small",
                coverageWithLayoutMaterialization()
        );
        var reportWithCpuConsumer = reportFor(
                "layout_broadcast_repair_small",
                coverageWithLayoutMaterializationAndCpuConsumer()
        );

        List<GpuCoverageGateResult> missing = GpuCoverageRegressionGate.evaluateTargets(
                reportWithoutLayoutEvidence,
                List.of(expectation)
        );
        List<GpuCoverageGateResult> passing = GpuCoverageRegressionGate.evaluateTargets(
                reportWithLayoutEvidence,
                List.of(expectation)
        );
        List<GpuCoverageGateResult> cpuConsumer = GpuCoverageRegressionGate.evaluateTargets(
                reportWithCpuConsumer,
                List.of(expectation)
        );

        assertTrue(missing.getFirst().failures().stream()
                .anyMatch(failure -> failure.contains("missing GPU layout materialization evidence")));
        assertTrue(passing.getFirst().passed(), passing.getFirst().failures().toString());
        assertTrue(cpuConsumer.getFirst().failures().stream()
                .anyMatch(failure -> failure.contains("unexpected CPU_CONSUMER layout materialization")));
    }

    @Test
    void phaseTwentyEightVisibleBlockerFailureNamesWorkloadAndBackend() {
        GpuCoverageHotPathExpectation expectation = GpuHotPathCoverageTargets.expectationsForBackend("GPU_CUDA")
                .stream()
                .filter(item -> item.workloadName().equals("bool_compare_where_small"))
                .findFirst()
                .orElseThrow();
        var report = reportWithRejectedReason(
                "bool_compare_where_small",
                "GPU_CUDA",
                backend.contract.ComputeBackend.GPU_CUDA,
                "unsupported-layout"
        );

        var results = GpuCoverageRegressionGate.evaluateTargets(report, List.of(expectation));

        assertEquals(1, results.size());
        assertTrue(results.getFirst().failures().stream()
                .anyMatch(failure -> failure.contains("bool_compare_where_small")
                        && failure.contains(expectation.backend())));
    }

    @Test
    void phaseThirtySixCudaScatterIndexGradientRequiresDuplicateIndexBlockerVisibility() {
        GpuCoverageHotPathExpectation expectation = GpuHotPathCoverageTargets.expectationsForBackend("GPU_CUDA")
                .stream()
                .filter(item -> item.workloadName().equals("scatter_index_gradient_small"))
                .findFirst()
                .orElseThrow();
        var missingReason = reportWithRejectedReason(
                "scatter_index_gradient_small",
                "GPU_CUDA",
                backend.contract.ComputeBackend.GPU_CUDA,
                "unsupported-layout"
        );
        var visibleReason = reportWithRejectedReason(
                "scatter_index_gradient_small",
                "GPU_CUDA",
                backend.contract.ComputeBackend.GPU_CUDA,
                "UNSUPPORTED_DUPLICATE_INDEX: operation SCATTER_ADD GPU_CUDA native duplicate-index accumulation is not proven"
        );

        List<GpuCoverageGateResult> missing = GpuCoverageRegressionGate.evaluateTargets(
                missingReason,
                List.of(expectation)
        );
        List<GpuCoverageGateResult> visible = GpuCoverageRegressionGate.evaluateTargets(
                visibleReason,
                List.of(expectation)
        );

        assertTrue(missing.getFirst().failures().stream()
                .anyMatch(failure -> failure.contains("scatter_index_gradient_small")
                        && failure.contains(expectation.backend())));
        assertTrue(visible.getFirst().passed(), visible.getFirst().failures().toString());
    }

    @Test
    void phaseFiftyEightMetalScatterIndexGradientRequiresHardNativeCoverage() {
        GpuCoverageHotPathExpectation expectation = GpuHotPathCoverageTargets.expectationsForBackend("GPU_METAL")
                .stream()
                .filter(item -> item.workloadName().equals("scatter_index_gradient_small"))
                .findFirst()
                .orElseThrow();
        var missing = GpuCoverageRegressionGate.evaluate(
                summary("GPU_METAL", coverage(1.0d, 3, 0, 1, 0, 0, 0, 1, 3, 0, 0)),
                expectation.policy()
        );
        var passed = GpuCoverageRegressionGate.evaluate(
                summary("GPU_METAL", coverage(1.0d, 3, 0, 0, 0, 0, 1, 1, 3, 0, 0)),
                expectation.policy()
        );

        assertTrue(missing.failures().contains("hidden tensor-array fallback"));
        assertTrue(missing.failures().contains("lost native buffer binding"));
        assertTrue(passed.passed(), passed.failures().toString());
    }

    @Test
    void phaseThirtySevenDenseLossRequiresHardNativeCoverage() {
        GpuCoverageHotPathExpectation expectation = GpuHotPathCoverageTargets.expectationsForBackend("GPU_METAL")
                .stream()
                .filter(item -> item.workloadName().equals("dense_loss_small"))
                .findFirst()
                .orElseThrow();
        var missing = GpuCoverageRegressionGate.evaluate(
                summary("GPU_METAL", coverage(1.0d, 1, 0, 1, 0, 0, 0, 1, 3, 0, 0)),
                expectation.policy()
        );
        var passed = GpuCoverageRegressionGate.evaluate(
                summary("GPU_METAL", coverage(1.0d, 1, 0, 0, 0, 0, 1, 1, 3, 0, 0)),
                expectation.policy()
        );

        assertTrue(missing.failures().contains("hidden tensor-array fallback"));
        assertTrue(missing.failures().contains("lost native buffer binding"));
        assertTrue(passed.passed(), passed.failures().toString());
    }

    @Test
    void phaseThirtySevenIndexLossTargetRequiresHardNativeCoverage() {
        GpuCoverageHotPathExpectation expectation = GpuHotPathCoverageTargets.expectationsForBackend("GPU_METAL")
                .stream()
                .filter(item -> item.workloadName().equals("cross_entropy_small"))
                .findFirst()
                .orElseThrow();
        var missing = GpuCoverageRegressionGate.evaluate(
                summary("GPU_METAL", coverage(1.0d, 1, 0, 1, 0, 0, 0, 1, 3, 0, 0)),
                expectation.policy()
        );
        var passed = GpuCoverageRegressionGate.evaluate(
                summary("GPU_METAL", coverage(1.0d, 1, 0, 0, 0, 0, 1, 1, 3, 0, 0)),
                expectation.policy()
        );

        assertTrue(missing.failures().contains("hidden tensor-array fallback"));
        assertTrue(missing.failures().contains("lost native buffer binding"));
        assertTrue(passed.passed(), passed.failures().toString());
    }

    @Test
    void phaseThirtyEightTrainingPolicyAllowsGradientPublicationButRejectsInternalCpuConsumer() {
        var policy = GpuCoverageGatePolicy.trainingHotPathTarget("GPU_METAL", 0.5d, 2, 1, 2, 0, 4);
        var gradientPublication = GpuCoverageRegressionGate.evaluate(
                summary("GPU_METAL", coverageWithMaterializationReasons(Map.of("GRADIENT_PUBLICATION", 2))),
                policy
        );
        var cpuConsumer = GpuCoverageRegressionGate.evaluate(
                summary("GPU_METAL", coverageWithMaterializationReasons(Map.of("GRADIENT_PUBLICATION", 2, "CPU_CONSUMER", 1))),
                policy
        );

        assertTrue(gradientPublication.passed(), gradientPublication.failures().toString());
        assertTrue(cpuConsumer.failures().contains("unexpected internal CPU materialization"));
    }

    @Test
    void reportNativeBufferPolicyAllowsPublicationBoundariesButRejectsInternalCpuExits() {
        var publicationCoverage = coverageWithMaterializationReasons(Map.of(
                "GRAPH_OUTPUT", 1,
                "GRADIENT_PUBLICATION", 2
        ));
        var publicationPolicy = GpuCoverageGatePolicy.reportNativeBufferTarget("GPU_METAL", publicationCoverage);
        var publicationResult = GpuCoverageRegressionGate.evaluate(
                summary("GPU_METAL", publicationCoverage),
                publicationPolicy
        );

        var internalCoverage = coverageWithMaterializationReasons(Map.of(
                "GRAPH_OUTPUT", 1,
                "GRADIENT_PUBLICATION", 2,
                "CPU_CONSUMER", 1
        ));
        var internalPolicy = GpuCoverageGatePolicy.reportNativeBufferTarget("GPU_METAL", internalCoverage);
        var internalResult = GpuCoverageRegressionGate.evaluate(
                summary("GPU_METAL", internalCoverage),
                internalPolicy
        );

        assertTrue(publicationResult.passed(), publicationResult.failures().toString());
        assertTrue(internalResult.failures().contains("unexpected CPU materialization"));
        assertTrue(internalResult.failures().contains("unexpected internal CPU materialization"));
    }

    @Test
    void phaseThirtyEightTrainingTargetsFailHiddenInternalCpuMaterializationByWorkload() {
        GpuCoverageHotPathExpectation expectation = GpuHotPathCoverageTargets.defaultExpectations()
                .stream()
                .filter(item -> item.workloadName().equals("training_dense_loss_small"))
                .findFirst()
                .orElseThrow();
        var report = reportFor(
                "training_dense_loss_small",
                coverageWithMaterializationReasons(Map.of("GRADIENT_PUBLICATION", 1, "CPU_CONSUMER", 1))
        );

        List<GpuCoverageGateResult> results = GpuCoverageRegressionGate.evaluateTargets(report, List.of(expectation));

        assertTrue(results.getFirst().failures().stream()
                .anyMatch(failure -> failure.contains("unexpected internal CPU materialization")
                        && failure.contains("training_dense_loss_small")));
    }

    private static config.profile.ExecutionProfile profile() {
        return new config.profile.ExecutionProfile(
                "phase28-gate-profile",
                "phase28-gate-profile",
                tensor.DataType.FLOAT32,
                runtime.contract.ExecutionMode.FORWARD,
                config.compile.CompileConfig.noGraphOptimizationBaseline(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                config.profile.WorkloadProfile.transformerHotPathDefaults()
        );
    }

    private static tuning.benchmark.report.BenchmarkSuiteReport reportFor(
            String workloadName,
            GpuCoverageSummary.BackendCoverage coverage
    ) {
        return new tuning.benchmark.report.BenchmarkSuiteReport(
                null,
                List.of(tuning.benchmark.report.BenchmarkReport.of(
                        workloadName,
                        List.of(tuning.benchmark.report.BenchmarkCandidateReport.success(
                                tuning.benchmark.BenchmarkEntry.candidate("candidate", profile()),
                                tuning.validate.ValidationResult.skipped(),
                                new tuning.measure.MeasurementResult(
                                        tuning.measure.MeasurementPolicy.defaults(),
                                        traceWithCoverage(coverage),
                                        new tuning.measure.MeasurementStatistics(1.0, 1.0, 1.0)
                                )
                        ))
                ))
        );
    }

    private static tuning.benchmark.report.BenchmarkSuiteReport reportWithRejectedReason(
            String workloadName,
            String backendName,
            backend.contract.ComputeBackend backend,
            String reason
    ) {
        return new tuning.benchmark.report.BenchmarkSuiteReport(
                null,
                List.of(tuning.benchmark.report.BenchmarkReport.of(
                        workloadName,
                        List.of(tuning.benchmark.report.BenchmarkCandidateReport.success(
                                tuning.benchmark.BenchmarkEntry.candidate("candidate", profile()),
                                tuning.validate.ValidationResult.skipped(),
                                new tuning.measure.MeasurementResult(
                                        tuning.measure.MeasurementPolicy.defaults(),
                                        traceWithRejectedReason(backendName, backend, reason),
                                        new tuning.measure.MeasurementStatistics(1.0, 1.0, 1.0)
                                )
                        ))
                ))
        );
    }

    private static trace.ExecutionTrace traceWithCoverage(GpuCoverageSummary.BackendCoverage coverage) {
        tensor.DataType evidenceDType = evidenceDType(coverage);
        var manifest = new backend.accelerator.lowering.GpuLoweredRegionManifest(
                "dtype-region",
                backend.contract.ComputeBackend.GPU_METAL,
                1,
                java.util.stream.IntStream.range(0, Math.max(1, coverage.maxSelectedRegionLength()))
                        .boxed()
                        .toList(),
                List.of(1),
                List.of(2),
                Math.max(1, coverage.maxSelectedRegionLength()),
                List.of(),
                java.util.stream.IntStream.range(0, coverage.loweredPrimitiveCount())
                        .mapToObj(index -> new backend.accelerator.lowering.GpuLoweredPrimitiveManifest(
                                "p" + index,
                                evidenceDType + "_TEST",
                                List.of(index),
                                List.of("external:0"),
                                "node:" + index,
                                evidenceDType,
                                List.of(2),
                                List.of()
                        ))
                        .toList(),
                List.of(),
                List.of(),
                coverage.gpuFusedSubpatternCount() == 0
                        ? backend.accelerator.lowering.GpuCompoundRegionSummary.none(backend.contract.ComputeBackend.GPU_METAL, List.of(1))
                        : backend.accelerator.lowering.GpuCompoundRegionSummary.supported(
                                backend.contract.ComputeBackend.GPU_METAL,
                                backend.accelerator.lowering.GpuCompoundPatternType.ELEMENTWISE_CHAIN,
                                List.of(1, 2),
                                List.of(1),
                                List.of(2),
                                List.of("ADD", "RELU"),
                                List.of(),
                                "bf16 synthetic fused subpattern"
                        ),
                List.of(),
                backend.accelerator.lowering.GpuLoweredRegionCandidateSpan.none(List.of(1)),
                coverage.dtypeResidencyReasons().isEmpty()
                        ? Map.of()
                        : Map.of("dtypeResidency.compute.1", "backend=GPU_METAL role=compute dtype=" + evidenceDType + " supported")
        );
        var attrs = new java.util.LinkedHashMap<String, Object>();
        attrs.put("acceleratorBufferBackend", "GPU_METAL");
        attrs.put("acceleratorBufferExecutionPath", "BUFFER_BINDING");
        attrs.put("acceleratorBufferReasonCode", "BUFFER_BINDING_AVAILABLE");
        attrs.put("acceleratorBufferReason", "using native buffer bindings");
        attrs.put("storageResidency", "DEVICE_OWNED");
        if (coverage.gpuLayoutMaterializationCount() > 0) {
            attrs.put("gpuLayoutMaterializationCount", coverage.gpuLayoutMaterializationCount());
            attrs.put("gpuLayoutMaterializationBytes", coverage.gpuLayoutMaterializationBytes());
            attrs.put("gpuLayoutTransformKind", coverage.gpuLayoutTransformKindCounts().keySet().iterator().next());
            attrs.put("gpuLayoutTransformTargetLayoutClass", coverage.gpuLayoutTargetLayoutClassCounts().keySet().iterator().next());
        }
        var gpuStep = new trace.execution.ExecutionStepTrace(
                0,
                "gpu_metal_dtype",
                evidenceDType + "_TEST",
                List.of(2),
                evidenceDType.name(),
                "GPU_METAL",
                "PreparedAcceleratorExecutable",
                1L,
                new trace.execution.StepExecutionMetadata(
                        "node",
                        attrs,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null
                )
        );
        var selection = new trace.prepare.BackendSelectionTrace(
                1,
                1,
                0,
                List.of(new trace.prepare.BackendSelectionDecisionTrace(
                        1,
                        manifest.orderedNodeIds(),
                        List.of(backend.contract.ComputeBackend.GPU_METAL.name()),
                        true,
                        backend.contract.ComputeBackend.GPU_METAL.name(),
                        "selected",
                        128L,
                        null,
                        List.of(),
                        testsupport.TraceSnapshotTestSupport.traceManifest(manifest)
                ))
        );
        List<trace.execution.CpuMaterializationTrace> materializations =
                coverage.cpuMaterializationReasonCounts().containsKey("CPU_CONSUMER")
                        ? List.of(new trace.execution.CpuMaterializationTrace(
                                1,
                                runtime.contract.CpuMaterializationReason.CPU_CONSUMER,
                                "GPU_METAL",
                                runtime.contract.StorageResidency.DEVICE_OWNED,
                                4096L,
                                1L,
                                true,
                                "synthetic CPU_CONSUMER"
                        ))
                        : List.of();
        return new trace.ExecutionTrace(
                new trace.compile.CompileTrace(true, 1L, 0, 0, false, trace.compile.PartitionCompileTrace.empty()),
                new trace.prepare.PrepareTrace(true, 1L, 0, 0, selection),
                new trace.execution.RunTrace(runtime.contract.ExecutionMode.FORWARD, 1L, List.of(gpuStep), materializations)
        );
    }

    private static trace.ExecutionTrace traceWithRejectedReason(
            String backendName,
            backend.contract.ComputeBackend computeBackend,
            String reason
    ) {
        var selection = new trace.prepare.BackendSelectionTrace(
                1,
                0,
                1,
                List.of(new trace.prepare.BackendSelectionDecisionTrace(
                        1,
                        List.of(1),
                        List.of(computeBackend.name()),
                        false,
                        null,
                        reason,
                        128L
                ))
        );
        return new trace.ExecutionTrace(
                new trace.compile.CompileTrace(true, 1L, 0, 0, false, trace.compile.PartitionCompileTrace.empty()),
                new trace.prepare.PrepareTrace(true, 1L, 0, 0, selection),
                new trace.execution.RunTrace(runtime.contract.ExecutionMode.FORWARD, 1L, List.of(), List.of())
        );
    }

    private static tensor.DataType evidenceDType(GpuCoverageSummary.BackendCoverage coverage) {
        String reasons = coverage.dtypeResidencyReasons().keySet().toString();
        if (reasons.contains("dtype=BOOL")) {
            return tensor.DataType.BOOL;
        }
        if (reasons.contains("dtype=INT32")) {
            return tensor.DataType.INT32;
        }
        if (reasons.contains("dtype=FLOAT64")) {
            return tensor.DataType.FLOAT64;
        }
        if (reasons.contains("dtype=FLOAT32")) {
            return tensor.DataType.FLOAT32;
        }
        return tensor.DataType.BFLOAT16;
    }

    private static GpuCoverageSummary summary(String backend, GpuCoverageSummary.BackendCoverage coverage) {
        return new GpuCoverageSummary(Map.of(backend, coverage));
    }

    private static GpuCoverageSummary.BackendCoverage coverage(
            double ratio,
            int maxSelectedRegionLength,
            int cpuMaterializationCount,
            int tensorArrayStepCount,
            int fallbackCount,
            int deviceHandoffCount,
            int bufferBindingStepCount
    ) {
        return coverage(
                ratio,
                maxSelectedRegionLength,
                cpuMaterializationCount,
                tensorArrayStepCount,
                fallbackCount,
                deviceHandoffCount,
                bufferBindingStepCount,
                1,
                maxSelectedRegionLength,
                0,
                0
        );
    }

    private static GpuCoverageSummary.BackendCoverage coverage(
            double ratio,
            int maxSelectedRegionLength,
            int cpuMaterializationCount,
            int tensorArrayStepCount,
            int fallbackCount,
            int deviceHandoffCount,
            int bufferBindingStepCount,
            int multiOpGpuRegionCount,
            int loweredPrimitiveCount,
            int gpuFusedSubpatternCount,
            int cpuFallbackStepCount
    ) {
        return new GpuCoverageSummary.BackendCoverage(
                4,
                3,
                ratio,
                1,
                multiOpGpuRegionCount,
                maxSelectedRegionLength,
                maxSelectedRegionLength,
                loweredPrimitiveCount,
                0,
                Map.of(),
                bufferBindingStepCount,
                tensorArrayStepCount,
                cpuFallbackStepCount,
                fallbackCount,
                cpuMaterializationCount,
                cpuMaterializationCount == 0 ? Map.of() : Map.of("CPU_CONSUMER", cpuMaterializationCount),
                4096L * cpuMaterializationCount,
                250_000L * cpuMaterializationCount,
                325_000L,
                Map.of("MPSGRAPH_RESULT_COPY", 1),
                deviceHandoffCount,
                0,
                0L,
                Map.of(),
                Map.of(),
                Map.of("DEVICE_OWNED", 1),
                Map.of(),
                gpuFusedSubpatternCount,
                gpuFusedSubpatternCount == 0 ? List.of() : List.of("ELEMENTWISE_CHAIN"),
                gpuFusedSubpatternCount == 0 ? List.of() : List.of("[1, 2]"),
                gpuFusedSubpatternCount,
                gpuFusedSubpatternCount == 0 ? List.of() : List.of("SUPPORTED_PATTERN"),
                List.of("BUFFER_BINDING_AVAILABLE"),
                List.of("using native buffer bindings")
        );
    }

    private static GpuCoverageSummary.BackendCoverage coverageWithDTypeEvidence(
            double ratio,
            int maxSelectedRegionLength,
            int multiOpGpuRegionCount,
            int loweredPrimitiveCount,
            int gpuFusedSubpatternCount,
            tensor.DataType dataType
    ) {
        return new GpuCoverageSummary.BackendCoverage(
                4,
                3,
                ratio,
                1,
                multiOpGpuRegionCount,
                maxSelectedRegionLength,
                maxSelectedRegionLength,
                loweredPrimitiveCount,
                0,
                Map.of(),
                1,
                0,
                0,
                0,
                0,
                Map.of(),
                0L,
                0L,
                0L,
                Map.of(),
                1,
                0,
                0L,
                Map.of(),
                Map.of(),
                Map.of("DEVICE_OWNED", 1),
                Map.of("dtypeResidency backend=GPU_METAL role=compute dtype=" + dataType + " supported", 1),
                gpuFusedSubpatternCount,
                gpuFusedSubpatternCount == 0 ? List.of() : List.of("ELEMENTWISE_CHAIN"),
                gpuFusedSubpatternCount == 0 ? List.of() : List.of("[1, 2]"),
                gpuFusedSubpatternCount,
                gpuFusedSubpatternCount == 0 ? List.of() : List.of("SUPPORTED"),
                List.of("BUFFER_BINDING_AVAILABLE"),
                List.of("using native buffer bindings")
        );
    }

    private static GpuCoverageSummary.BackendCoverage coverageWithMaterializationReasons(Map<String, Integer> reasons) {
        int materializationCount = reasons.values().stream().mapToInt(Integer::intValue).sum();
        return new GpuCoverageSummary.BackendCoverage(
                4,
                4,
                1.0d,
                1,
                1,
                2,
                2,
                3,
                0,
                Map.of(),
                1,
                0,
                0,
                0,
                materializationCount,
                reasons,
                4096L * materializationCount,
                250_000L * materializationCount,
                0L,
                Map.of(),
                Math.max(1, materializationCount + 1),
                0,
                0L,
                Map.of(),
                Map.of(),
                Map.of("DEVICE_OWNED", 1),
                Map.of(),
                0,
                List.of(),
                List.of(),
                0,
                List.of(),
                List.of("BUFFER_BINDING_AVAILABLE"),
                List.of("using native buffer bindings")
        );
    }

    private static GpuCoverageSummary.BackendCoverage coverageWithLayoutMaterialization() {
        return layoutCoverage(Map.of());
    }

    private static GpuCoverageSummary.BackendCoverage coverageWithLayoutMaterializationAndCpuConsumer() {
        return layoutCoverage(Map.of("CPU_CONSUMER", 1));
    }

    private static GpuCoverageSummary.BackendCoverage layoutCoverage(Map<String, Integer> cpuMaterializationReasons) {
        return new GpuCoverageSummary.BackendCoverage(
                4,
                3,
                1.0d,
                1,
                0,
                2,
                2,
                1,
                0,
                Map.of(),
                1,
                0,
                0,
                0,
                cpuMaterializationReasons.isEmpty() ? 0 : 1,
                cpuMaterializationReasons,
                0L,
                0L,
                0L,
                Map.of(),
                1,
                1,
                4096L,
                Map.of("BROADCAST_GPU_MATERIALIZATION", 1),
                Map.of("DENSE_CONTIGUOUS", 1),
                Map.of("DEVICE_OWNED", 1),
                Map.of(),
                0,
                List.of(),
                List.of(),
                0,
                List.of(),
                List.of("BUFFER_BINDING_AVAILABLE"),
                List.of("using native buffer bindings")
        );
    }
}

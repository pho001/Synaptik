import backend.contract.ComputeBackend;
import backend.accelerator.lowering.GpuCompoundPatternType;
import backend.accelerator.lowering.GpuCompoundRegionSummary;
import backend.accelerator.lowering.GpuLoweredPrimitiveManifest;
import backend.accelerator.lowering.GpuLoweredRegionCandidateSpan;
import backend.accelerator.lowering.GpuLoweredRegionManifest;
import backend.accelerator.lowering.GpuLoweredRegionOriginalOp;
import backend.accelerator.lowering.GpuLoweredRegionRejection;
import backend.accelerator.lowering.GpuLoweringUnsupportedReason;
import runtime.contract.CpuMaterializationReason;
import runtime.contract.StorageResidency;
import runtime.contract.ExecutionMode;
import trace.prepare.BackendSelectionDecisionTrace;
import trace.prepare.BackendSelectionTrace;
import trace.compile.CompileTrace;
import trace.execution.CpuMaterializationTrace;
import trace.execution.ExecutionStepTrace;
import trace.ExecutionTrace;
import trace.compile.PartitionCompileTrace;
import trace.prepare.PrepareTrace;
import trace.execution.RunTrace;
import trace.execution.StepExecutionMetadata;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.benchmark.report.GpuCoverageBaseline;
import tuning.benchmark.report.GpuCoverageComparison;
import tuning.benchmark.report.GpuCoverageSummary;
import tuning.benchmark.report.GpuTargetCoverageTruth;
import tuning.benchmark.report.GpuTargetExecutionStatus;

import java.util.List;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class GpuCoverageSummaryTest {
    @Test
    void v14TargetCoverageTruthKeepsRemainingGapsOutOfNativeStatus() {
        for (ComputeBackend backend : List.of(ComputeBackend.GPU_METAL, ComputeBackend.GPU_CUDA)) {
            Map<operations.Operation.OpType, GpuTargetCoverageTruth.Row> rows = GpuTargetCoverageTruth.rowsFor(backend)
                    .stream()
                    .collect(java.util.stream.Collectors.toMap(
                            GpuTargetCoverageTruth.Row::opType,
                            row -> row
                    ));

            assertEquals(
                    GpuTargetExecutionStatus.NATIVE_EXECUTABLE,
                    rows.get(operations.Operation.OpType.SUM).executionStatus()
            );
            assertEquals(
                    GpuTargetExecutionStatus.NATIVE_EXECUTABLE,
                    rows.get(operations.Operation.OpType.MEAN).executionStatus()
            );
            assertEquals(
                    GpuTargetExecutionStatus.NATIVE_EXECUTABLE,
                    rows.get(operations.Operation.OpType.REDUCE_MIN).executionStatus()
            );
            assertEquals(
                    GpuTargetExecutionStatus.NATIVE_EXECUTABLE,
                    rows.get(operations.Operation.OpType.REDUCE_MAX).executionStatus()
            );
            assertEquals(
                    GpuTargetExecutionStatus.NATIVE_EXECUTABLE,
                    rows.get(operations.Operation.OpType.LAYER_NORM).executionStatus()
            );
            assertEquals(
                    GpuTargetExecutionStatus.NATIVE_EXECUTABLE,
                    rows.get(operations.Operation.OpType.RMS_NORM).executionStatus()
            );
            if (backend == ComputeBackend.GPU_METAL) {
                assertEquals(
                        GpuTargetExecutionStatus.NATIVE_EXECUTABLE,
                        rows.get(operations.Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION).executionStatus()
                );
                assertEquals(
                        GpuTargetExecutionStatus.NATIVE_EXECUTABLE,
                        rows.get(operations.Operation.OpType.NLL_LOSS).executionStatus()
                );
                assertEquals(
                        GpuTargetExecutionStatus.NATIVE_EXECUTABLE,
                        rows.get(operations.Operation.OpType.CROSS_ENTROPY_LOSS).executionStatus()
                );
            } else {
                assertEquals(
                        GpuTargetExecutionStatus.EXPLICIT_CPU_FALLBACK,
                        rows.get(operations.Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION).executionStatus()
                );
                assertEquals(
                        GpuTargetExecutionStatus.UNSUPPORTED_REJECTION,
                        rows.get(operations.Operation.OpType.NLL_LOSS).executionStatus()
                );
                assertEquals(
                        GpuTargetExecutionStatus.UNSUPPORTED_REJECTION,
                        rows.get(operations.Operation.OpType.CROSS_ENTROPY_LOSS).executionStatus()
                );
            }
            assertEquals(
                    backend == ComputeBackend.GPU_METAL
                            ? GpuTargetExecutionStatus.NATIVE_EXECUTABLE
                            : GpuTargetExecutionStatus.UNSUPPORTED_REJECTION,
                    rows.get(operations.Operation.OpType.CROSS_ENTROPY_LOSS_INDICES).executionStatus()
            );
            assertEquals(
                    backend == ComputeBackend.GPU_METAL
                            ? GpuTargetExecutionStatus.NATIVE_EXECUTABLE
                            : GpuTargetExecutionStatus.UNSUPPORTED_REJECTION,
                    rows.get(operations.Operation.OpType.CROSS_ENTROPY_LOSS_INDICES_GRAD).executionStatus()
            );
            for (operations.Operation.OpType opType : List.of(
                    operations.Operation.OpType.SOFTMAX_GRAD,
                    operations.Operation.OpType.LOG_SOFTMAX_GRAD,
                    operations.Operation.OpType.REDUCE_MIN_GRAD,
                    operations.Operation.OpType.REDUCE_MAX_GRAD,
                    operations.Operation.OpType.MIN_GRAD,
                    operations.Operation.OpType.MAX_GRAD
            )) {
                assertEquals(
                        backend == ComputeBackend.GPU_METAL
                                ? GpuTargetExecutionStatus.NATIVE_EXECUTABLE
                                : GpuTargetExecutionStatus.MATRIX_SUPPORTED_ONLY,
                        rows.get(opType).executionStatus(),
                        opType.name()
                );
            }
            assertEquals(
                    backend == ComputeBackend.GPU_METAL
                            ? GpuTargetExecutionStatus.MATRIX_SUPPORTED_ONLY
                            : GpuTargetExecutionStatus.UNSUPPORTED_REJECTION,
                    rows.get(operations.Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_BACKWARD).executionStatus()
            );
            assertEquals(
                    backend == ComputeBackend.GPU_METAL
                            ? GpuTargetExecutionStatus.NATIVE_EXECUTABLE
                            : GpuTargetExecutionStatus.UNSUPPORTED_REJECTION,
                    rows.get(operations.Operation.OpType.GATHER).executionStatus()
            );
            assertEquals(
                    backend == ComputeBackend.GPU_METAL
                            ? GpuTargetExecutionStatus.NATIVE_EXECUTABLE
                            : GpuTargetExecutionStatus.UNSUPPORTED_REJECTION,
                    rows.get(operations.Operation.OpType.TAKE_ALONG_AXIS).executionStatus()
            );
            assertEquals(
                    backend == ComputeBackend.GPU_METAL
                            ? GpuTargetExecutionStatus.NATIVE_EXECUTABLE
                            : GpuTargetExecutionStatus.UNSUPPORTED_REJECTION,
                    rows.get(operations.Operation.OpType.GATHER_ND).executionStatus()
            );
            assertEquals(
                    backend == ComputeBackend.GPU_METAL
                            ? GpuTargetExecutionStatus.NATIVE_EXECUTABLE
                            : GpuTargetExecutionStatus.UNSUPPORTED_REJECTION,
                    rows.get(operations.Operation.OpType.GATHER_GRAD).executionStatus()
            );
            assertEquals(
                    backend == ComputeBackend.GPU_METAL
                            ? GpuTargetExecutionStatus.NATIVE_EXECUTABLE
                            : GpuTargetExecutionStatus.UNSUPPORTED_REJECTION,
                    rows.get(operations.Operation.OpType.TAKE_ALONG_AXIS_GRAD).executionStatus()
            );
            assertEquals(
                    backend == ComputeBackend.GPU_METAL
                            ? GpuTargetExecutionStatus.NATIVE_EXECUTABLE
                            : GpuTargetExecutionStatus.UNSUPPORTED_REJECTION,
                    rows.get(operations.Operation.OpType.SCATTER_ADD).executionStatus()
            );
            if (backend == ComputeBackend.GPU_CUDA) {
                assertTrue(rows.get(operations.Operation.OpType.GATHER_GRAD).detail().contains("UNSUPPORTED_DUPLICATE_INDEX"));
                assertTrue(rows.get(operations.Operation.OpType.TAKE_ALONG_AXIS_GRAD).detail().contains("UNSUPPORTED_DUPLICATE_INDEX"));
                assertTrue(rows.get(operations.Operation.OpType.SCATTER_ADD).detail().contains("UNSUPPORTED_DUPLICATE_INDEX"));
            }
            assertEquals(
                    backend == ComputeBackend.GPU_METAL
                            ? GpuTargetExecutionStatus.NATIVE_EXECUTABLE
                            : GpuTargetExecutionStatus.UNSUPPORTED_REJECTION,
                    rows.get(operations.Operation.OpType.CONV2D).executionStatus()
            );
            assertEquals(
                    backend == ComputeBackend.GPU_METAL
                            ? GpuTargetExecutionStatus.NATIVE_EXECUTABLE
                            : GpuTargetExecutionStatus.UNSUPPORTED_REJECTION,
                    rows.get(operations.Operation.OpType.MAX_POOL2D).executionStatus()
            );
            assertEquals(
                    backend == ComputeBackend.GPU_METAL
                            ? GpuTargetExecutionStatus.NATIVE_EXECUTABLE
                            : GpuTargetExecutionStatus.UNSUPPORTED_REJECTION,
                    rows.get(operations.Operation.OpType.AVG_POOL2D).executionStatus()
            );
            assertEquals(
                    backend == ComputeBackend.GPU_METAL
                            ? GpuTargetExecutionStatus.NATIVE_EXECUTABLE
                            : GpuTargetExecutionStatus.UNSUPPORTED_REJECTION,
                    rows.get(operations.Operation.OpType.GT).executionStatus()
            );
        }
    }

    @Test
    void summarizesMetalCoverageFromSyntheticTrace() {
        GpuCoverageSummary summary = GpuCoverageSummary.fromTrace(traceFor("GPU_METAL", ComputeBackend.GPU_METAL));
        GpuCoverageSummary.BackendCoverage coverage = summary.backends().get("GPU_METAL");

        assertEquals(2, coverage.totalStepCount());
        assertEquals(1, coverage.acceleratorStepCount());
        assertEquals(0.5d, coverage.gpuCoverageRatio(), 1e-9);
        assertEquals(1, coverage.bufferBindingStepCount());
        assertEquals(0, coverage.tensorArrayStepCount());
        assertEquals(0, coverage.cpuFallbackStepCount());
        assertEquals(0, coverage.fallbackCount());
        assertEquals(Map.of("DEVICE_OWNED", 1), coverage.storageResidencyCounts());
        assertEquals(Map.of("MPSGRAPH_RESULT_COPY", 1), coverage.nativeCopyStrategyCounts());
        assertTrue(coverage.reasonCodes().contains("BUFFER_BINDING_AVAILABLE"));
    }

    @Test
    void summarizesMetalExecutionRoutesAndRejectedAlternativesFromStepAttributes() {
        GpuCoverageSummary summary = GpuCoverageSummary.fromTrace(traceWithMetalRoute(
                "CUSTOM_KERNEL",
                "TRUE_OUTPUT_BUFFER_WRITE",
                List.of("MPS_GRAPH_SELECTED")
        ));
        GpuCoverageSummary.BackendCoverage coverage = summary.backends().get("GPU_METAL");

        assertEquals(Map.of("CUSTOM_KERNEL", 1), coverage.executionRouteCounts());
        assertEquals(Map.of("MPS_GRAPH_SELECTED", 1), coverage.rejectedRouteReasonCounts());
        assertEquals(Map.of("TRUE_OUTPUT_BUFFER_WRITE", 1), coverage.nativeCopyStrategyCounts());
    }

    @Test
    void summarizesCudaCoverageFromSyntheticTrace() {
        GpuCoverageSummary summary = GpuCoverageSummary.fromTrace(traceFor("GPU_CUDA", ComputeBackend.GPU_CUDA));
        GpuCoverageSummary.BackendCoverage coverage = summary.backends().get("GPU_CUDA");

        assertEquals(2, coverage.totalStepCount());
        assertEquals(1, coverage.acceleratorStepCount());
        assertEquals(0.5d, coverage.gpuCoverageRatio(), 1e-9);
        assertEquals(1, coverage.bufferBindingStepCount());
        assertEquals(325_000L, coverage.copyDurationNs());
        assertTrue(coverage.fallbackReasons().contains("using native buffer bindings"));
    }

    @Test
    void summarizesGpuLayoutMaterializationFromStepAttributes() {
        GpuCoverageSummary summary = GpuCoverageSummary.fromTrace(traceFor("GPU_METAL", ComputeBackend.GPU_METAL));
        GpuCoverageSummary.BackendCoverage coverage = summary.backends().get("GPU_METAL");

        assertEquals(1, coverage.gpuLayoutMaterializationCount());
        assertEquals(4096L, coverage.gpuLayoutMaterializationBytes());
        assertEquals(Map.of("DENSE_GPU_MATERIALIZATION", 1), coverage.gpuLayoutTransformKindCounts());
        assertEquals(Map.of("DENSE_CONTIGUOUS", 1), coverage.gpuLayoutTargetLayoutClassCounts());
    }

    @Test
    void countsRejectedCandidateReasonsAndSelectedRegionLengths() {
        GpuCoverageSummary summary = GpuCoverageSummary.fromTrace(traceFor("GPU_METAL", ComputeBackend.GPU_METAL));
        GpuCoverageSummary.BackendCoverage coverage = summary.backends().get("GPU_METAL");

        assertEquals(1, coverage.selectedRegionCount());
        assertEquals(3, coverage.maxSelectedRegionLength());
        assertEquals(3.0d, coverage.averageSelectedRegionLength(), 1e-9);
        assertEquals(1, coverage.rejectedCandidateCount());
        assertEquals(Map.of("unsupported-layout", 1), coverage.rejectedCandidateReasonCounts());
    }

    @Test
    void coverageSummaryIgnoresManifestWhenCountingSelectedRegions() {
        GpuLoweredRegionManifest manifest = new GpuLoweredRegionManifest(
                "gpu-metal-region-10",
                ComputeBackend.GPU_METAL,
                10,
                List.of(10, 11, 12),
                List.of(1),
                List.of(12),
                99,
                List.of(new GpuLoweredRegionOriginalOp(
                        10,
                        "LOG_SOFTMAX",
                        List.of(1),
                        List.of(12),
                        DataType.FLOAT32,
                        List.of(2, 3),
                        List.of("p0"),
                        List.of()
                )),
                List.of(new GpuLoweredPrimitiveManifest(
                        "p0",
                        "SOFTMAX",
                        List.of(10),
                        List.of("external:0"),
                        "node:0",
                        DataType.FLOAT32,
                        List.of(2, 3),
                        List.of()
                )),
                List.of(),
                List.of(),
                GpuCompoundRegionSummary.none(ComputeBackend.GPU_METAL, List.of(10, 11, 12)),
                List.of(),
                GpuLoweredRegionCandidateSpan.none(List.of(10, 11, 12)),
                Map.of()
        );
        ExecutionTrace trace = traceWithManifest("GPU_METAL", ComputeBackend.GPU_METAL, manifest);

        GpuCoverageSummary.BackendCoverage coverage = GpuCoverageSummary.fromTrace(trace).backends().get("GPU_METAL");

        assertEquals(1, coverage.selectedRegionCount());
        assertEquals(3, coverage.maxSelectedRegionLength());
        assertEquals(3.0d, coverage.averageSelectedRegionLength(), 1e-9);
    }

    @Test
    void countsCpuMaterializationReasonsAndDeviceHandoffs() {
        GpuCoverageSummary summary = GpuCoverageSummary.fromTrace(traceFor("GPU_METAL", ComputeBackend.GPU_METAL));
        GpuCoverageSummary.BackendCoverage coverage = summary.backends().get("GPU_METAL");

        assertEquals(1, coverage.cpuMaterializationCount());
        assertEquals(Map.of("CPU_CONSUMER", 1), coverage.cpuMaterializationReasonCounts());
        assertEquals(1, coverage.internalCpuMaterializationCount());
        assertEquals(0, coverage.gradientPublicationMaterializationCount());
        assertEquals(4096L, coverage.cpuMaterializationBytes());
        assertEquals(250_000L, coverage.cpuMaterializationDurationNs());
        assertEquals(2, coverage.deviceHandoffCount());
    }

    @Test
    void phaseThirtyEightSeparatesGradientPublicationFromInternalCpuMaterialization() {
        ExecutionTrace base = traceFor("GPU_METAL", ComputeBackend.GPU_METAL);
        CpuMaterializationTrace gradientPublication = new CpuMaterializationTrace(
                13,
                CpuMaterializationReason.GRADIENT_PUBLICATION,
                "GPU_METAL",
                StorageResidency.DEVICE_OWNED,
                2048L,
                125_000L,
                true,
                "gradient publication requested readable storage"
        );
        ExecutionTrace trace = new ExecutionTrace(
                base.compile(),
                base.prepare(),
                new RunTrace(
                        ExecutionMode.FORWARD_BACKWARD,
                        base.run().durationNs(),
                        base.run().steps(),
                        List.of(base.run().cpuMaterializations().getFirst(), gradientPublication)
                )
        );

        GpuCoverageSummary.BackendCoverage coverage = GpuCoverageSummary.fromTrace(trace).backends().get("GPU_METAL");

        assertEquals(2, coverage.cpuMaterializationCount());
        assertEquals(1, coverage.internalCpuMaterializationCount());
        assertEquals(1, coverage.gradientPublicationMaterializationCount());
        assertEquals(Map.of("CPU_CONSUMER", 1, "GRADIENT_PUBLICATION", 1), coverage.cpuMaterializationReasonCounts());
    }

    @Test
    void comparesCoverageAgainstBaselineWithoutTimingThresholds() {
        GpuCoverageBaseline baseline = new GpuCoverageBaseline("v1.1", "GPU_METAL", 1, 2, 1, 2);
        GpuCoverageSummary.BackendCoverage current = new GpuCoverageSummary.BackendCoverage(
                4,
                3,
                0.75d,
                1,
                1,
                3,
                3.0d,
                3,
                0,
                Map.of(),
                3,
                0,
                0,
                0,
                1,
                Map.of("CPU_CONSUMER", 1),
                4096L,
                250_000L,
                325_000L,
                Map.of(),
                1,
                0,
                0L,
                Map.of(),
                Map.of(),
                Map.of("DEVICE_OWNED", 3),
                Map.of(),
                0,
                List.of(),
                List.of(),
                0,
                List.of(),
                List.of("BUFFER_BINDING_AVAILABLE"),
                List.of("using native buffer bindings")
        );

        GpuCoverageComparison comparison = GpuCoverageComparison.compare(baseline, current);

        assertTrue(comparison.passes());
        assertEquals("v1.1", comparison.baselineName());
        assertEquals("GPU_METAL", comparison.backend());
        assertTrue(comparison.improvements().contains("longer selected region"));
        assertTrue(comparison.improvements().contains("fewer CPU materializations"));
        assertTrue(comparison.improvements().contains("fewer fallbacks"));
        assertTrue(comparison.improvements().contains("fewer device handoffs"));
        assertTrue(comparison.regressions().isEmpty());
    }

    @Test
    void coverageSummaryCountsDTypeMaterializationReasons() {
        GpuLoweredRegionManifest manifest = new GpuLoweredRegionManifest(
                "gpu-metal-region-dtype",
                ComputeBackend.GPU_METAL,
                30,
                List.of(30, 31, 32),
                List.of(20),
                List.of(32),
                3,
                List.of(),
                List.of(),
                List.of(),
                List.of(),
                GpuCompoundRegionSummary.none(ComputeBackend.GPU_METAL, List.of(30, 31, 32)),
                List.of(new GpuLoweredRegionRejection(
                        "dtype_residency.compute",
                        31,
                        "p1",
                        "",
                        GpuLoweringUnsupportedReason.UNSUPPORTED_DTYPE,
                        "dtypeResidency backend=GPU_CUDA role=compute dtype=INT32 unsupported"
                )),
                GpuLoweredRegionCandidateSpan.none(List.of(30, 31, 32)),
                Map.of(
                        "dtypeResidency.input.20", "backend=GPU_METAL role=externalInput dtype=BOOL residentRepresentable=true",
                        "dtypeResidency.compute.30", "backend=GPU_METAL role=compute dtype=BFLOAT16 residentRepresentable=true nativeCompute=true"
                )
        );

        GpuCoverageSummary.BackendCoverage coverage = GpuCoverageSummary.fromTrace(
                traceWithManifest("GPU_METAL", ComputeBackend.GPU_METAL, manifest)
        ).backends().get("GPU_METAL");
        String reasons = coverage.dtypeResidencyReasons().toString();

        assertTrue(reasons.contains("dtypeResidency"));
        assertTrue(reasons.contains("UNSUPPORTED_DTYPE"));
        assertTrue(reasons.contains("backend=GPU_METAL"));
        assertTrue(reasons.contains("backend=GPU_CUDA"));
        assertTrue(reasons.contains("dtype=BFLOAT16"));
        assertTrue(reasons.contains("dtype=INT32"));
        assertTrue(reasons.contains("dtype=BOOL"));
    }

    @Test
    void coverageSummaryCountsGpuFusedSubpatterns() {
        GpuLoweredRegionManifest manifest = new GpuLoweredRegionManifest(
                "gpu-metal-region-fused",
                ComputeBackend.GPU_METAL,
                30,
                List.of(30, 31, 32),
                List.of(20),
                List.of(32),
                3,
                List.of(),
                List.of(new GpuLoweredPrimitiveManifest(
                        "p0",
                        "ADD",
                        List.of(30),
                        List.of("external:0", "external:1"),
                        "node:0",
                        DataType.FLOAT32,
                        List.of(4),
                        List.of()
                )),
                List.of(),
                List.of(),
                GpuCompoundRegionSummary.supported(
                        ComputeBackend.GPU_METAL,
                        GpuCompoundPatternType.ELEMENTWISE_CHAIN,
                        List.of(30, 31, 32),
                        List.of(20),
                        List.of(32),
                        List.of("ADD", "RELU", "EXP"),
                        List.of(),
                        "synthetic elementwise fused subpattern"
                ),
                List.of(),
                GpuLoweredRegionCandidateSpan.none(List.of(30, 31, 32)),
                Map.of()
        );

        GpuCoverageSummary.BackendCoverage coverage = GpuCoverageSummary.fromTrace(
                traceWithManifest("GPU_METAL", ComputeBackend.GPU_METAL, manifest)
        ).backends().get("GPU_METAL");

        assertEquals(1, coverage.gpuFusedSubpatternCount());
        assertTrue(coverage.gpuFusedSubpatternTypes().contains("ELEMENTWISE_CHAIN"));
        assertTrue(coverage.gpuFusedSubpatternOriginalNodeIds().contains("[30, 31, 32]"));
        assertEquals(1, coverage.gpuFusedSubpatternLoweredPrimitiveCount());
        assertTrue(coverage.gpuFusedSubpatternReasons().contains("SUPPORTED"));
        assertEquals(1, coverage.cpuMaterializationCount());
    }

    @Test
    void phaseNineteenCoverageSummaryCountsMultiOpRegionLengthAndLoweredPrimitives() {
        GpuLoweredRegionManifest manifest = new GpuLoweredRegionManifest(
                "gpu-metal-region-phase19",
                ComputeBackend.GPU_METAL,
                30,
                List.of(30, 31, 32),
                List.of(20),
                List.of(32),
                3,
                List.of(),
                List.of(
                        primitive("p0", "MATMUL", 30),
                        primitive("p1", "ADD", 31),
                        primitive("p2", "RELU", 32),
                        primitive("p3", "LOG", 32)
                ),
                List.of(),
                List.of(),
                GpuCompoundRegionSummary.supported(
                        ComputeBackend.GPU_METAL,
                        GpuCompoundPatternType.ELEMENTWISE_CHAIN,
                        List.of(31, 32),
                        List.of(20),
                        List.of(32),
                        List.of("ADD", "RELU"),
                        List.of(),
                        "phase nineteen elementwise subpattern"
                ),
                List.of(),
                GpuLoweredRegionCandidateSpan.none(List.of(30, 31, 32)),
                Map.of()
        );

        GpuCoverageSummary.BackendCoverage coverage = GpuCoverageSummary.fromTrace(
                traceWithManifest("GPU_METAL", ComputeBackend.GPU_METAL, manifest)
        ).backends().get("GPU_METAL");

        assertEquals(1, coverage.multiOpGpuRegionCount());
        assertEquals(3, coverage.maxSelectedRegionLength());
        assertEquals(4, coverage.loweredPrimitiveCount());
        assertEquals(1, coverage.gpuFusedSubpatternCount());
        assertEquals(1, coverage.nativeBufferStepCount());
        assertEquals(0, coverage.tensorArrayStepCount());
        assertEquals(Map.of("CPU_CONSUMER", 1), coverage.cpuMaterializationReasonCounts());
        assertEquals(2, coverage.deviceHandoffCount());
    }

    @Test
    void coverageSummaryCountsUnsupportedNormVariantAndLossReasons() {
        String normReason = "UNSUPPORTED_LAYOUT: GPU_METAL normalization inputs require dense layout family=NORMALIZATION target=layer_norm_small";
        String lossReason = "UNSUPPORTED_INDEX_SEMANTICS: GPU_METAL index-target loss target out of range: 17 for classes=16 family=LOSS_ADJACENT target=transformer_block_hot_path";
        GpuLoweredRegionManifest manifest = new GpuLoweredRegionManifest(
                "gpu-metal-region-phase17",
                ComputeBackend.GPU_METAL,
                40,
                List.of(40, 41),
                List.of(30),
                List.of(41),
                2,
                List.of(new GpuLoweredRegionOriginalOp(
                        41,
                        "LOG_SOFTMAX",
                        List.of(40),
                        List.of(41),
                        DataType.FLOAT32,
                        List.of(2, 3),
                        List.of("p0", "p1"),
                        List.of()
                )),
                List.of(
                        new GpuLoweredPrimitiveManifest(
                                "p0",
                                "SOFTMAX",
                                List.of(41),
                                List.of("external:0"),
                                "node:0",
                                DataType.FLOAT32,
                                List.of(2, 3),
                                List.of()
                        ),
                        new GpuLoweredPrimitiveManifest(
                                "p1",
                                "LOG",
                                List.of(41),
                                List.of("node:0"),
                                "node:1",
                                DataType.FLOAT32,
                                List.of(2, 3),
                                List.of()
                        )
                ),
                List.of(),
                List.of(),
                GpuCompoundRegionSummary.none(ComputeBackend.GPU_METAL, List.of(40, 41)),
                List.of(),
                GpuLoweredRegionCandidateSpan.none(List.of(40, 41)),
                Map.of()
        );
        ExecutionTrace trace = traceWithPhaseSeventeenEvidence("GPU_METAL", ComputeBackend.GPU_METAL, manifest, normReason, lossReason);

        GpuCoverageSummary.BackendCoverage coverage = GpuCoverageSummary.fromTrace(trace).backends().get("GPU_METAL");
        String manifestText = backend.accelerator.lowering.GpuLoweredRegionManifestRenderer.renderCompact(manifest);

        assertEquals(1, coverage.selectedRegionCount());
        assertEquals(2, coverage.rejectedCandidateCount());
        assertEquals(1, coverage.rejectedCandidateReasonCounts().get(normReason));
        assertEquals(1, coverage.rejectedCandidateReasonCounts().get(lossReason));
        assertTrue(manifestText.contains("LOG_SOFTMAX"));
        assertTrue(manifestText.contains("SOFTMAX"));
        assertTrue(coverage.rejectedCandidateReasonCounts().toString().contains("family=NORMALIZATION"));
        assertTrue(coverage.rejectedCandidateReasonCounts().toString().contains("family=LOSS_ADJACENT"));
        assertTrue(coverage.rejectedCandidateReasonCounts().toString().contains("UNSUPPORTED_INDEX_SEMANTICS"));
        assertTrue(coverage.rejectedCandidateReasonCounts().toString().contains("target=layer_norm_small"));
        assertTrue(coverage.rejectedCandidateReasonCounts().toString().contains("target=transformer_block_hot_path"));
    }

    private static GpuLoweredPrimitiveManifest primitive(String id, String type, int nodeId) {
        return new GpuLoweredPrimitiveManifest(
                id,
                type,
                List.of(nodeId),
                List.of("external:0"),
                "node:" + id.substring(1),
                DataType.FLOAT32,
                List.of(2, 3),
                List.of()
        );
    }

    static ExecutionTrace traceFor(String backendName, ComputeBackend backend) {
        ExecutionStepTrace gpuStep = new ExecutionStepTrace(
                0,
                backendName.toLowerCase() + "_linear",
                "LINEAR",
                List.of(16, 16),
                DataType.FLOAT32.name(),
                backendName,
                "PreparedAcceleratorExecutable",
                2_000_000L,
                new StepExecutionMetadata(
                        "node",
                        Map.ofEntries(
                                Map.entry("acceleratorBufferBackend", backendName),
                                Map.entry("acceleratorBufferExecutionPath", "BUFFER_BINDING"),
                                Map.entry("acceleratorBufferReasonCode", "BUFFER_BINDING_AVAILABLE"),
                                Map.entry("acceleratorBufferReason", "using native buffer bindings"),
                                Map.entry("acceleratorJavaToNativeCopyNs", 100_000L),
                                Map.entry("acceleratorNativeToJavaCopyNs", 200_000L),
                                Map.entry("acceleratorNativeDeviceCopyNs", 25_000L),
                                Map.entry("acceleratorNativeCopyStrategy", "MPSGRAPH_RESULT_COPY"),
                                Map.entry("gpuLayoutMaterializationCount", 1),
                                Map.entry("gpuLayoutMaterializationBytes", 4096L),
                                Map.entry("gpuLayoutTransformKind", "DENSE_GPU_MATERIALIZATION"),
                                Map.entry("gpuLayoutTransformTargetLayoutClass", "DENSE_CONTIGUOUS"),
                                Map.entry("storageResidency", "DEVICE_OWNED")
                        ),
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null
                )
        );
        ExecutionStepTrace cpuStep = new ExecutionStepTrace(
                1,
                "cpu_consumer",
                "ADD",
                List.of(16, 16),
                DataType.FLOAT32.name(),
                "CPU",
                "CpuElementWiseKernel",
                1_000_000L,
                StepExecutionMetadata.none()
        );
        BackendSelectionTrace selection = new BackendSelectionTrace(
                2,
                1,
                1,
                List.of(
                        new BackendSelectionDecisionTrace(
                                10,
                                List.of(10, 11, 12),
                                List.of(backend.name()),
                                true,
                                backend.name(),
                                "selected",
                                4096L
                        ),
                        new BackendSelectionDecisionTrace(
                                20,
                                List.of(20),
                                List.of(backend.name()),
                                false,
                                null,
                                "unsupported-layout",
                                1024L
                        )
                )
        );
        CpuMaterializationTrace materialization = new CpuMaterializationTrace(
                12,
                CpuMaterializationReason.CPU_CONSUMER,
                backendName,
                StorageResidency.DEVICE_OWNED,
                4096L,
                250_000L,
                true,
                "CPU consumer requested readable storage"
        );
        return new ExecutionTrace(
                new CompileTrace(true, 1L, 0, 0, false, PartitionCompileTrace.empty()),
                new PrepareTrace(true, 1L, 0, 0, selection),
                new RunTrace(ExecutionMode.FORWARD, 3_000_000L, List.of(gpuStep, cpuStep), List.of(materialization))
        );
    }

    private static ExecutionTrace traceWithMetalRoute(
            String route,
            String nativeCopyStrategy,
            List<String> rejectedReasonCodes
    ) {
        ExecutionTrace trace = traceFor("GPU_METAL", ComputeBackend.GPU_METAL);
        ExecutionStepTrace original = trace.run().steps().getFirst();
        LinkedHashMap<String, Object> attrs = new LinkedHashMap<>(original.metadata().attributes());
        attrs.put("metalExecutionRoute", route);
        attrs.put("metalRouteRejectedReasonCodes", rejectedReasonCodes);
        attrs.put("metalNativeCopyStrategy", nativeCopyStrategy);
        attrs.remove("acceleratorNativeCopyStrategy");
        ExecutionStepTrace routedStep = new ExecutionStepTrace(
                original.index(),
                original.label(),
                original.opType(),
                original.shape(),
                original.dataType(),
                original.backend(),
                original.kernel(),
                original.durationNs(),
                new StepExecutionMetadata(
                        original.metadata().kind(),
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
        return new ExecutionTrace(
                trace.compile(),
                trace.prepare(),
                new RunTrace(
                        trace.run().mode(),
                        trace.run().durationNs(),
                        List.of(routedStep, trace.run().steps().get(1)),
                        trace.run().cpuMaterializations()
                )
        );
    }

    private static ExecutionTrace traceWithPhaseSeventeenEvidence(
            String backendName,
            ComputeBackend backend,
            GpuLoweredRegionManifest manifest,
            String normReason,
            String lossReason
    ) {
        ExecutionTrace trace = traceFor(backendName, backend);
        BackendSelectionTrace selection = new BackendSelectionTrace(
                3,
                1,
                2,
                List.of(
                        new BackendSelectionDecisionTrace(
                                40,
                                List.of(40, 41),
                                List.of(backend.name()),
                                true,
                                backend.name(),
                                "selected",
                                4096L,
                                null,
                                List.of(),
                                testsupport.TraceSnapshotTestSupport.traceManifest(manifest)
                ),
                        new BackendSelectionDecisionTrace(
                                90,
                                List.of(90),
                                List.of(backend.name()),
                                false,
                                null,
                                normReason,
                                1024L
                        ),
                        new BackendSelectionDecisionTrace(
                                91,
                                List.of(91),
                                List.of(backend.name()),
                                false,
                                null,
                                lossReason,
                                1024L
                        )
                )
        );
        return new ExecutionTrace(
                trace.compile(),
                new PrepareTrace(true, 1L, 0, 0, selection),
                trace.run()
        );
    }

    private static ExecutionTrace traceWithManifest(
            String backendName,
            ComputeBackend backend,
            GpuLoweredRegionManifest manifest
    ) {
        ExecutionTrace trace = traceFor(backendName, backend);
        BackendSelectionTrace selection = new BackendSelectionTrace(
                1,
                1,
                0,
                List.of(new BackendSelectionDecisionTrace(
                        10,
                        List.of(10, 11, 12),
                        List.of(backend.name()),
                        true,
                        backend.name(),
                        "selected",
                        4096L,
                        null,
                        List.of(),
                        testsupport.TraceSnapshotTestSupport.traceManifest(manifest)
                ))
        );
        return new ExecutionTrace(
                trace.compile(),
                new PrepareTrace(true, 1L, 0, 0, selection),
                trace.run()
        );
    }
}

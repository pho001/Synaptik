package backend.accelerator.lowering;

import backend.ComputeBackend;
import operations.Operation;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GpuLoweringCoverageMatrixTest {
    private static final Set<GpuLoweringOperationFamily> REQUIRED_PHASE_ELEVEN_FAMILIES = EnumSet.of(
            GpuLoweringOperationFamily.MATMUL_LINEAR,
            GpuLoweringOperationFamily.ELEMENTWISE_CHAIN,
            GpuLoweringOperationFamily.LAYOUT_VIEW_ADJACENT,
            GpuLoweringOperationFamily.SOFTMAX_LIKE,
            GpuLoweringOperationFamily.REDUCTION,
            GpuLoweringOperationFamily.NORMALIZATION,
            GpuLoweringOperationFamily.LOSS_ADJACENT,
            GpuLoweringOperationFamily.ATTENTION
    );

    @Test
    void matrixCoversRequiredPhaseElevenFamiliesForMetalAndCuda() {
        assertRequiredFamiliesCovered(ComputeBackend.GPU_METAL);
        assertRequiredFamiliesCovered(ComputeBackend.GPU_CUDA);
    }

    @Test
    void supportedEntriesHaveConcreteOperationTypes() {
        for (GpuLoweringCoverageEntry entry : GpuLoweringCoverageMatrix.entries()) {
            if (entry.status() == GpuLoweringCoverageStatus.SUPPORTED) {
                assertNotNull(entry.opType(), () -> "missing op type for supported entry " + entry);
                assertFalse(entry.opType() == Operation.OpType.UNKNOWN,
                        () -> "supported entry must not use UNKNOWN op type: " + entry);
                assertEquals(GpuLoweringUnsupportedReason.SUPPORTED, entry.reason(),
                        () -> "supported entry must use SUPPORTED reason: " + entry);
            }
        }
    }

    @Test
    void nonSupportedEntriesHaveStableReasons() {
        for (GpuLoweringCoverageEntry entry : GpuLoweringCoverageMatrix.entries()) {
            if (entry.status() == GpuLoweringCoverageStatus.FALLBACK
                    || entry.status() == GpuLoweringCoverageStatus.UNSUPPORTED) {
                assertNotNull(entry.reason(), () -> "missing reason for non-supported entry " + entry);
                assertFalse(entry.reason() == GpuLoweringUnsupportedReason.SUPPORTED,
                        () -> "non-supported entry must not use SUPPORTED reason: " + entry);
            } else {
                assertEquals(GpuLoweringCoverageStatus.SUPPORTED, entry.status(),
                        () -> "unknown coverage status: " + entry);
            }
        }
    }

    @Test
    void docsMatrixListsRequiredStatusesAndReasons() throws IOException {
        String docs = Files.readString(Path.of("docs/gpu-lowering-coverage.md"));

        assertTrue(docs.contains("GPU Lowering Coverage Matrix"));
        assertTrue(docs.contains("GPULOWER-01"));
        assertTrue(docs.contains("UNSUPPORTED_OPERATION"));
        assertTrue(docs.contains("UNSUPPORTED_DTYPE"));
        assertTrue(docs.contains("UNSUPPORTED_LAYOUT"));
        assertTrue(docs.contains("DEFERRED_FUSED_REGION"));
        assertTrue(docs.contains("supported"));
        assertTrue(docs.contains("fallback"));
        assertTrue(docs.contains("unsupported"));
    }

    @Test
    void compoundFusedRowsExposeStablePhaseTwelveReasons() {
        for (ComputeBackend backend : List.of(ComputeBackend.GPU_METAL, ComputeBackend.GPU_CUDA)) {
            GpuLoweringCoverageEntry entry = GpuLoweringCoverageMatrix.entryFor(backend, Operation.OpType.FUSED);

            assertFalse(entry.status() == GpuLoweringCoverageStatus.SUPPORTED);
            assertFalse(entry.reason() == GpuLoweringUnsupportedReason.SUPPORTED);
            assertEquals(GpuLoweringUnsupportedReason.CPU_FUSED_OPERATION_UNSUPPORTED, entry.reason());
            assertTrue(entry.note().contains("CPU Operation.OpType.FUSED remains CPU-only for Phase 12"));
        }
    }

    @Test
    void phaseSeventeenMatrixCoversNormalizationReductionSoftmaxAndLossRows() {
        List<Operation.OpType> phaseSeventeenOps = List.of(
                Operation.OpType.LAYER_NORM,
                Operation.OpType.RMS_NORM,
                Operation.OpType.SUM,
                Operation.OpType.MEAN,
                Operation.OpType.REDUCE_MIN,
                Operation.OpType.REDUCE_MAX,
                Operation.OpType.SOFTMAX,
                Operation.OpType.LOG_SOFTMAX,
                Operation.OpType.NLL_LOSS,
                Operation.OpType.CROSS_ENTROPY_LOSS,
                Operation.OpType.CROSS_ENTROPY_LOSS_INDICES,
                Operation.OpType.CROSS_ENTROPY_LOSS_INDICES_GRAD,
                Operation.OpType.CONV2D
        );

        for (ComputeBackend backend : List.of(ComputeBackend.GPU_METAL, ComputeBackend.GPU_CUDA)) {
            for (Operation.OpType opType : phaseSeventeenOps) {
                GpuLoweringCoverageEntry entry = GpuLoweringCoverageMatrix.entryFor(backend, opType);

                assertEquals(backend, entry.backend());
                assertEquals(opType, entry.opType());
                assertFalse(entry.opType() == Operation.OpType.UNKNOWN,
                        () -> backend + " must list Phase 17 op " + opType);
            }

            assertEquals(GpuLoweringCoverageStatus.SUPPORTED,
                    GpuLoweringCoverageMatrix.entryFor(backend, Operation.OpType.SOFTMAX).status());
            assertEquals(GpuLoweringCoverageStatus.SUPPORTED,
                    GpuLoweringCoverageMatrix.entryFor(backend, Operation.OpType.LOG_SOFTMAX).status());
            for (Operation.OpType opType : List.of(
                    Operation.OpType.SUM,
                    Operation.OpType.MEAN,
                    Operation.OpType.REDUCE_MIN,
                    Operation.OpType.REDUCE_MAX,
                    Operation.OpType.LAYER_NORM,
                    Operation.OpType.RMS_NORM
            )) {
                assertEquals(GpuLoweringCoverageStatus.SUPPORTED,
                        GpuLoweringCoverageMatrix.entryFor(backend, opType).status(),
                        () -> opType + " should be supported by native accelerator DAG execution");
            }
            assertFalse(GpuLoweringCoverageMatrix.entryFor(backend, Operation.OpType.CROSS_ENTROPY_LOSS_INDICES).reason()
                            == GpuLoweringUnsupportedReason.SUPPORTED,
                    "index-target loss must keep a stable non-supported reason");
        }
    }

    @Test
    void phaseTwentySixMatrixCoversLossAndIndexingFamilyExplicitly() {
        List<Operation.OpType> phaseTwentySixOps = List.of(
                Operation.OpType.NLL_LOSS,
                Operation.OpType.CROSS_ENTROPY_LOSS,
                Operation.OpType.CROSS_ENTROPY_LOSS_INDICES,
                Operation.OpType.CROSS_ENTROPY_LOSS_INDICES_GRAD,
                Operation.OpType.GATHER,
                Operation.OpType.GATHER_GRAD,
                Operation.OpType.TAKE_ALONG_AXIS,
                Operation.OpType.TAKE_ALONG_AXIS_GRAD,
                Operation.OpType.SCATTER_ADD
        );

        for (ComputeBackend backend : List.of(ComputeBackend.GPU_METAL, ComputeBackend.GPU_CUDA)) {
            for (Operation.OpType opType : phaseTwentySixOps) {
                GpuLoweringCoverageEntry entry = GpuLoweringCoverageMatrix.entryFor(backend, opType);

                assertEquals(backend, entry.backend());
                assertEquals(opType, entry.opType());
                if (backend == ComputeBackend.GPU_METAL
                        && (opType == Operation.OpType.GATHER || opType == Operation.OpType.TAKE_ALONG_AXIS)) {
                    assertEquals(GpuLoweringCoverageStatus.SUPPORTED, entry.status());
                    assertEquals(GpuLoweringUnsupportedReason.SUPPORTED, entry.reason());
                } else {
                    assertFalse(entry.status() == GpuLoweringCoverageStatus.SUPPORTED,
                            () -> opType + " must not be supported until native index/loss execution exists");
                    assertFalse(entry.reason() == GpuLoweringUnsupportedReason.SUPPORTED);
                }
                assertFalse(entry.note().isBlank());
            }

            assertEquals(GpuLoweringUnsupportedReason.UNSUPPORTED_INDEX_SEMANTICS,
                    GpuLoweringCoverageMatrix.entryFor(backend, Operation.OpType.CROSS_ENTROPY_LOSS_INDICES).reason());
            assertEquals(GpuLoweringUnsupportedReason.UNSUPPORTED_INDEX_SEMANTICS,
                    GpuLoweringCoverageMatrix.entryFor(backend, Operation.OpType.CROSS_ENTROPY_LOSS_INDICES_GRAD).reason());
            GpuLoweringUnsupportedReason expectedForwardIndexReason = backend == ComputeBackend.GPU_METAL
                    ? GpuLoweringUnsupportedReason.SUPPORTED
                    : GpuLoweringUnsupportedReason.CAPABILITY_MISSING;
            assertEquals(expectedForwardIndexReason,
                    GpuLoweringCoverageMatrix.entryFor(backend, Operation.OpType.GATHER).reason());
            assertEquals(expectedForwardIndexReason,
                    GpuLoweringCoverageMatrix.entryFor(backend, Operation.OpType.TAKE_ALONG_AXIS).reason());
            assertEquals(GpuLoweringUnsupportedReason.UNSUPPORTED_DUPLICATE_INDEX,
                    GpuLoweringCoverageMatrix.entryFor(backend, Operation.OpType.GATHER_GRAD).reason());
            assertEquals(GpuLoweringUnsupportedReason.UNSUPPORTED_DUPLICATE_INDEX,
                    GpuLoweringCoverageMatrix.entryFor(backend, Operation.OpType.TAKE_ALONG_AXIS_GRAD).reason());
            assertEquals(GpuLoweringUnsupportedReason.UNSUPPORTED_DUPLICATE_INDEX,
                    GpuLoweringCoverageMatrix.entryFor(backend, Operation.OpType.SCATTER_ADD).reason());
            assertTrue(GpuLoweringCoverageMatrix.entryFor(backend, Operation.OpType.GATHER_GRAD).note()
                    .contains("duplicate-index accumulation parity"));
            assertTrue(GpuLoweringCoverageMatrix.entryFor(backend, Operation.OpType.TAKE_ALONG_AXIS_GRAD).note()
                    .contains("rank-preserving static bounds checks"));
            assertTrue(GpuLoweringCoverageMatrix.entryFor(backend, Operation.OpType.SCATTER_ADD).note()
                    .contains("native write-add semantics"));
        }
    }

    @Test
    void phaseTwentySevenMatrixCoversConvPoolAndBoolOutputFamilyExplicitly() {
        List<Operation.OpType> convPoolOps = List.of(
                Operation.OpType.CONV2D,
                Operation.OpType.CONV2D_GEMM,
                Operation.OpType.CONV2D_BACKWARD_INPUT,
                Operation.OpType.CONV2D_BACKWARD_WEIGHT,
                Operation.OpType.CONV2D_BACKWARD_INPUT_GEMM,
                Operation.OpType.CONV2D_BACKWARD_WEIGHT_GEMM,
                Operation.OpType.MAX_POOL2D,
                Operation.OpType.MAX_POOL2D_BACKWARD_INPUT,
                Operation.OpType.AVG_POOL2D,
                Operation.OpType.AVG_POOL2D_BACKWARD_INPUT
        );
        List<Operation.OpType> boolCompareOps = List.of(
                Operation.OpType.GT,
                Operation.OpType.GE,
                Operation.OpType.LT,
                Operation.OpType.LE,
                Operation.OpType.EQ,
                Operation.OpType.NE
        );
        List<Operation.OpType> boolLogicalAndReductionOps = List.of(
                Operation.OpType.LOGICAL_AND,
                Operation.OpType.LOGICAL_OR,
                Operation.OpType.LOGICAL_NOT,
                Operation.OpType.REDUCE_ALL,
                Operation.OpType.REDUCE_ANY
        );

        for (ComputeBackend backend : List.of(ComputeBackend.GPU_METAL, ComputeBackend.GPU_CUDA)) {
            for (Operation.OpType opType : convPoolOps) {
                GpuLoweringCoverageEntry entry = GpuLoweringCoverageMatrix.entryFor(backend, opType);

                assertEquals(backend, entry.backend());
                assertEquals(opType, entry.opType());
                assertEquals(GpuLoweringOperationFamily.CONV_POOL, entry.family());
                if (backend == ComputeBackend.GPU_METAL
                        && (opType == Operation.OpType.CONV2D
                        || opType == Operation.OpType.CONV2D_GEMM
                        || opType == Operation.OpType.MAX_POOL2D
                        || opType == Operation.OpType.AVG_POOL2D)) {
                    assertEquals(GpuLoweringCoverageStatus.SUPPORTED, entry.status());
                    assertEquals(GpuLoweringUnsupportedReason.SUPPORTED, entry.reason(),
                            () -> opType + " should be supported for scoped Metal direct conv/pool forward execution");
                    assertTrue(entry.note().contains("MPSGraph"));
                } else {
                    assertEquals(GpuLoweringCoverageStatus.UNSUPPORTED, entry.status());
                    assertEquals(GpuLoweringUnsupportedReason.CAPABILITY_MISSING, entry.reason(),
                            () -> opType + " should be a capability-gated conv/pool rejection, not an unlisted operation");
                }
                assertFalse(entry.note().isBlank());
            }

            for (Operation.OpType opType : boolCompareOps) {
                GpuLoweringCoverageEntry entry = GpuLoweringCoverageMatrix.entryFor(backend, opType);

                assertEquals(backend, entry.backend());
                assertEquals(opType, entry.opType());
                assertEquals(GpuLoweringOperationFamily.COMPARE_BOOL, entry.family());
                if (backend == ComputeBackend.GPU_METAL) {
                    assertEquals(GpuLoweringCoverageStatus.SUPPORTED, entry.status());
                    assertEquals(GpuLoweringUnsupportedReason.SUPPORTED, entry.reason(),
                            () -> opType + " should be native-supported for Metal BOOL compare output");
                    assertTrue(entry.note().contains("native Metal BOOL output DAG execution"));
                } else {
                    assertEquals(GpuLoweringCoverageStatus.UNSUPPORTED, entry.status());
                    assertEquals(GpuLoweringUnsupportedReason.UNSUPPORTED_DTYPE, entry.reason(),
                            () -> opType + " should remain unsupported for CUDA until CUDA BOOL output compute is implemented");
                    assertTrue(entry.note().contains("BOOL output"));
                }
                assertTrue(entry.note().contains("WHERE"));
            }

            for (Operation.OpType opType : boolLogicalAndReductionOps) {
                GpuLoweringCoverageEntry entry = GpuLoweringCoverageMatrix.entryFor(backend, opType);

                assertEquals(backend, entry.backend());
                assertEquals(opType, entry.opType());
                assertEquals(GpuLoweringOperationFamily.COMPARE_BOOL, entry.family());
                if (backend == ComputeBackend.GPU_METAL) {
                    assertEquals(GpuLoweringCoverageStatus.SUPPORTED, entry.status());
                    assertEquals(GpuLoweringUnsupportedReason.SUPPORTED, entry.reason(),
                            () -> opType + " should be native-supported for Metal BOOL logical/reduction output");
                    assertTrue(entry.note().contains("native Metal BOOL output DAG execution"));
                } else {
                    assertEquals(GpuLoweringCoverageStatus.UNSUPPORTED, entry.status());
                    assertEquals(GpuLoweringUnsupportedReason.UNSUPPORTED_DTYPE, entry.reason(),
                            () -> opType + " should reject because native BOOL output compute is not implemented");
                    assertTrue(entry.note().contains("BOOL output"));
                }
                assertTrue(entry.note().contains("WHERE"));
            }
        }
    }

    @Test
    void phaseSeventeenCoverageRowsTieToHotPathTargets() {
        String notes = GpuLoweringCoverageMatrix.entries().stream()
                .map(GpuLoweringCoverageEntry::note)
                .collect(Collectors.joining("\n"));

        assertTrue(notes.contains("target=layer_norm_small"));
        assertTrue(notes.contains("target=conv2d_resnet_3x3"));
        assertTrue(notes.contains("target=max_pool2d_small"));
        assertTrue(notes.contains("target=avg_pool2d_small"));
        assertTrue(notes.contains("target=transformer_block_hot_path"));
    }

    @Test
    void phaseSeventeenNonSupportedRowsUseStableReasonCodes() {
        for (ComputeBackend backend : List.of(ComputeBackend.GPU_METAL, ComputeBackend.GPU_CUDA)) {
            List<Operation.OpType> nonSupportedOps = backend == ComputeBackend.GPU_METAL
                    ? List.of(
                    Operation.OpType.NLL_LOSS,
                    Operation.OpType.CROSS_ENTROPY_LOSS,
                    Operation.OpType.CROSS_ENTROPY_LOSS_INDICES,
                    Operation.OpType.CROSS_ENTROPY_LOSS_INDICES_GRAD
            )
                    : List.of(
                    Operation.OpType.NLL_LOSS,
                    Operation.OpType.CROSS_ENTROPY_LOSS,
                    Operation.OpType.CROSS_ENTROPY_LOSS_INDICES,
                    Operation.OpType.CROSS_ENTROPY_LOSS_INDICES_GRAD,
                    Operation.OpType.CONV2D
            );
            for (Operation.OpType opType : nonSupportedOps) {
                GpuLoweringCoverageEntry entry = GpuLoweringCoverageMatrix.entryFor(backend, opType);

                assertFalse(entry.status() == GpuLoweringCoverageStatus.SUPPORTED,
                        () -> opType + " should not be claimed supported by Phase 17 matrix contract alone");
                assertNotNull(entry.reason());
                assertFalse(entry.reason() == GpuLoweringUnsupportedReason.SUPPORTED,
                        () -> opType + " must keep a stable non-supported reason");
            }
        }
    }

    private static void assertRequiredFamiliesCovered(ComputeBackend backend) {
        List<GpuLoweringCoverageEntry> entries = GpuLoweringCoverageMatrix.entriesFor(backend);
        Set<GpuLoweringOperationFamily> coveredFamilies = entries.stream()
                .map(GpuLoweringCoverageEntry::family)
                .collect(Collectors.toCollection(() -> EnumSet.noneOf(GpuLoweringOperationFamily.class)));

        assertTrue(coveredFamilies.containsAll(REQUIRED_PHASE_ELEVEN_FAMILIES),
                () -> backend + " missing required families: " + missing(coveredFamilies));
        assertTrue(entries.stream().anyMatch(entry -> entry.status() == GpuLoweringCoverageStatus.SUPPORTED),
                () -> backend + " must have supported coverage rows");
        if (backend == ComputeBackend.GPU_CUDA) {
            assertTrue(entries.stream().anyMatch(entry -> entry.status() == GpuLoweringCoverageStatus.FALLBACK),
                    () -> backend + " must have fallback coverage rows");
        }
        assertTrue(entries.stream().anyMatch(entry -> entry.status() == GpuLoweringCoverageStatus.UNSUPPORTED),
                () -> backend + " must have unsupported coverage rows");
    }

    private static Set<GpuLoweringOperationFamily> missing(Set<GpuLoweringOperationFamily> coveredFamilies) {
        Set<GpuLoweringOperationFamily> missing = EnumSet.copyOf(REQUIRED_PHASE_ELEVEN_FAMILIES);
        missing.removeAll(coveredFamilies);
        return missing;
    }
}

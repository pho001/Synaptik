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
            GpuLoweringUnsupportedReason expectedIndexLossReason = backend == ComputeBackend.GPU_METAL
                    ? GpuLoweringUnsupportedReason.SUPPORTED
                    : GpuLoweringUnsupportedReason.UNSUPPORTED_INDEX_SEMANTICS;
            assertEquals(expectedIndexLossReason,
                    GpuLoweringCoverageMatrix.entryFor(backend, Operation.OpType.CROSS_ENTROPY_LOSS_INDICES).reason());
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
                Operation.OpType.GATHER_ND,
                Operation.OpType.GATHER_ND_GRAD,
                Operation.OpType.TAKE_ALONG_AXIS,
                Operation.OpType.TAKE_ALONG_AXIS_GRAD,
                Operation.OpType.SCATTER_ADD,
                Operation.OpType.SCATTER_ELEMENTS,
                Operation.OpType.SCATTER_ND
        );

        for (ComputeBackend backend : List.of(ComputeBackend.GPU_METAL, ComputeBackend.GPU_CUDA)) {
            for (Operation.OpType opType : phaseTwentySixOps) {
                GpuLoweringCoverageEntry entry = GpuLoweringCoverageMatrix.entryFor(backend, opType);

                assertEquals(backend, entry.backend());
                assertEquals(opType, entry.opType());
                if (backend == ComputeBackend.GPU_METAL
                        && (opType == Operation.OpType.NLL_LOSS
                        || opType == Operation.OpType.CROSS_ENTROPY_LOSS
                        || opType == Operation.OpType.CROSS_ENTROPY_LOSS_INDICES
                        || opType == Operation.OpType.CROSS_ENTROPY_LOSS_INDICES_GRAD
                        || opType == Operation.OpType.GATHER
                        || opType == Operation.OpType.GATHER_ND
                        || opType == Operation.OpType.TAKE_ALONG_AXIS
                        || opType == Operation.OpType.GATHER_GRAD
                        || opType == Operation.OpType.TAKE_ALONG_AXIS_GRAD
                        || opType == Operation.OpType.SCATTER_ADD
                        || opType == Operation.OpType.SCATTER_ELEMENTS
                        || opType == Operation.OpType.SCATTER_ND)) {
                    assertEquals(GpuLoweringCoverageStatus.SUPPORTED, entry.status());
                    assertEquals(GpuLoweringUnsupportedReason.SUPPORTED, entry.reason());
                } else {
                    assertFalse(entry.status() == GpuLoweringCoverageStatus.SUPPORTED,
                            () -> opType + " must not be supported until native index/loss execution exists");
                    assertFalse(entry.reason() == GpuLoweringUnsupportedReason.SUPPORTED);
                }
                assertFalse(entry.note().isBlank());
            }

            assertEquals(backend == ComputeBackend.GPU_METAL
                            ? GpuLoweringUnsupportedReason.SUPPORTED
                            : GpuLoweringUnsupportedReason.UNSUPPORTED_INDEX_SEMANTICS,
                    GpuLoweringCoverageMatrix.entryFor(backend, Operation.OpType.CROSS_ENTROPY_LOSS_INDICES).reason());
            assertEquals(backend == ComputeBackend.GPU_METAL
                            ? GpuLoweringUnsupportedReason.SUPPORTED
                            : GpuLoweringUnsupportedReason.UNSUPPORTED_INDEX_SEMANTICS,
                    GpuLoweringCoverageMatrix.entryFor(backend, Operation.OpType.CROSS_ENTROPY_LOSS_INDICES_GRAD).reason());
            GpuLoweringUnsupportedReason expectedForwardIndexReason = backend == ComputeBackend.GPU_METAL
                    ? GpuLoweringUnsupportedReason.SUPPORTED
                    : GpuLoweringUnsupportedReason.CAPABILITY_MISSING;
            assertEquals(expectedForwardIndexReason,
                    GpuLoweringCoverageMatrix.entryFor(backend, Operation.OpType.GATHER).reason());
            assertEquals(backend == ComputeBackend.GPU_METAL
                            ? GpuLoweringUnsupportedReason.SUPPORTED
                            : GpuLoweringUnsupportedReason.UNSUPPORTED_INDEX_SEMANTICS,
                    GpuLoweringCoverageMatrix.entryFor(backend, Operation.OpType.GATHER_ND).reason());
            assertEquals(GpuLoweringUnsupportedReason.UNSUPPORTED_INDEX_SEMANTICS,
                    GpuLoweringCoverageMatrix.entryFor(backend, Operation.OpType.GATHER_ND_GRAD).reason());
            assertEquals(expectedForwardIndexReason,
                    GpuLoweringCoverageMatrix.entryFor(backend, Operation.OpType.TAKE_ALONG_AXIS).reason());
            GpuLoweringUnsupportedReason expectedIndexWriteReason = backend == ComputeBackend.GPU_METAL
                    ? GpuLoweringUnsupportedReason.SUPPORTED
                    : GpuLoweringUnsupportedReason.UNSUPPORTED_DUPLICATE_INDEX;
            assertEquals(expectedIndexWriteReason,
                    GpuLoweringCoverageMatrix.entryFor(backend, Operation.OpType.GATHER_GRAD).reason());
            assertEquals(expectedIndexWriteReason,
                    GpuLoweringCoverageMatrix.entryFor(backend, Operation.OpType.TAKE_ALONG_AXIS_GRAD).reason());
            assertEquals(expectedIndexWriteReason,
                    GpuLoweringCoverageMatrix.entryFor(backend, Operation.OpType.SCATTER_ADD).reason());
            assertEquals(backend == ComputeBackend.GPU_METAL
                            ? GpuLoweringUnsupportedReason.SUPPORTED
                            : GpuLoweringUnsupportedReason.UNSUPPORTED_INDEX_SEMANTICS,
                    GpuLoweringCoverageMatrix.entryFor(backend, Operation.OpType.SCATTER_ELEMENTS).reason());
            assertEquals(backend == ComputeBackend.GPU_METAL
                            ? GpuLoweringUnsupportedReason.SUPPORTED
                            : GpuLoweringUnsupportedReason.UNSUPPORTED_INDEX_SEMANTICS,
                    GpuLoweringCoverageMatrix.entryFor(backend, Operation.OpType.SCATTER_ND).reason());
            String expectedNote = backend == ComputeBackend.GPU_METAL ? "scatterAlongAxis" : "duplicate-index";
            assertTrue(GpuLoweringCoverageMatrix.entryFor(backend, Operation.OpType.GATHER_GRAD).note()
                    .contains(expectedNote));
            assertTrue(GpuLoweringCoverageMatrix.entryFor(backend, Operation.OpType.TAKE_ALONG_AXIS_GRAD).note()
                    .contains(expectedNote));
            assertTrue(GpuLoweringCoverageMatrix.entryFor(backend, Operation.OpType.SCATTER_ADD).note()
                    .contains(backend == ComputeBackend.GPU_METAL ? "scatterAlongAxis" : "native write-add semantics"));
            assertTrue(GpuLoweringCoverageMatrix.entryFor(backend, Operation.OpType.SCATTER_ELEMENTS).note()
                    .contains(backend == ComputeBackend.GPU_METAL ? "scatterAlongAxis" : "rank-preserving write"));
            assertTrue(GpuLoweringCoverageMatrix.entryFor(backend, Operation.OpType.SCATTER_ND).note()
                    .contains(backend == ComputeBackend.GPU_METAL ? "scatterNDWithDataTensor" : "tuple-index write"));
        }
    }

    @Test
    void phaseThirtySevenMatrixSeparatesSupportedDenseLossFromIndexTargetLoss() {
        GpuLoweringCoverageEntry nll = GpuLoweringCoverageMatrix.entryFor(ComputeBackend.GPU_METAL, Operation.OpType.NLL_LOSS);
        GpuLoweringCoverageEntry denseCe = GpuLoweringCoverageMatrix.entryFor(ComputeBackend.GPU_METAL, Operation.OpType.CROSS_ENTROPY_LOSS);
        GpuLoweringCoverageEntry indexCe = GpuLoweringCoverageMatrix.entryFor(ComputeBackend.GPU_METAL, Operation.OpType.CROSS_ENTROPY_LOSS_INDICES);
        GpuLoweringCoverageEntry indexCeGrad = GpuLoweringCoverageMatrix.entryFor(ComputeBackend.GPU_METAL, Operation.OpType.CROSS_ENTROPY_LOSS_INDICES_GRAD);

        assertEquals(GpuLoweringCoverageStatus.SUPPORTED, nll.status());
        assertEquals(GpuLoweringCoverageStatus.SUPPORTED, denseCe.status());
        assertEquals(GpuLoweringUnsupportedReason.SUPPORTED, nll.reason());
        assertEquals(GpuLoweringUnsupportedReason.SUPPORTED, denseCe.reason());
        assertTrue(nll.note().contains("all-axis SUM"));
        assertTrue(denseCe.note().contains("SOFTMAX"));
        assertTrue(denseCe.note().contains("target=loss_dense_small"));

        assertEquals(GpuLoweringCoverageStatus.SUPPORTED, indexCe.status());
        assertEquals(GpuLoweringCoverageStatus.SUPPORTED, indexCeGrad.status());
        assertEquals(GpuLoweringUnsupportedReason.SUPPORTED, indexCe.reason());
        assertEquals(GpuLoweringUnsupportedReason.SUPPORTED, indexCeGrad.reason());
        assertTrue(indexCe.note().contains("ignore-index"));
        assertTrue(indexCeGrad.note().contains("scatterAlongAxis"));
    }

    @Test
    void phaseThirtyEightMatrixSeparatesBackwardSupportFromForwardSupport() {
        List<Operation.OpType> supportedBackwardAdjacent = List.of(
                Operation.OpType.SOFTMAX_GRAD,
                Operation.OpType.LOG_SOFTMAX_GRAD,
                Operation.OpType.REDUCE_MIN_GRAD,
                Operation.OpType.REDUCE_MAX_GRAD,
                Operation.OpType.MIN_GRAD,
                Operation.OpType.MAX_GRAD
        );
        for (ComputeBackend backend : List.of(ComputeBackend.GPU_METAL, ComputeBackend.GPU_CUDA)) {
            for (Operation.OpType opType : supportedBackwardAdjacent) {
                GpuLoweringCoverageEntry entry = GpuLoweringCoverageMatrix.entryFor(backend, opType);

                assertEquals(GpuLoweringCoverageStatus.SUPPORTED, entry.status(), opType.name());
                assertEquals(GpuLoweringUnsupportedReason.SUPPORTED, entry.reason(), opType.name());
                assertEquals(GpuLoweringOperationFamily.BACKWARD_ADJACENT, entry.family(), opType.name());
            }
        }

        assertEquals(
                GpuLoweringCoverageStatus.SUPPORTED,
                GpuLoweringCoverageMatrix.entryFor(ComputeBackend.GPU_METAL, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_BACKWARD).status()
        );
        assertEquals(
                GpuLoweringCoverageStatus.UNSUPPORTED,
                GpuLoweringCoverageMatrix.entryFor(ComputeBackend.GPU_CUDA, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_BACKWARD).status()
        );

        for (Operation.OpType opType : List.of(
                Operation.OpType.CONV2D_BACKWARD_INPUT,
                Operation.OpType.CONV2D_BACKWARD_WEIGHT,
                Operation.OpType.CONV2D_BACKWARD_INPUT_GEMM,
                Operation.OpType.CONV2D_BACKWARD_WEIGHT_GEMM,
                Operation.OpType.MAX_POOL2D_BACKWARD_INPUT,
                Operation.OpType.AVG_POOL2D_BACKWARD_INPUT
        )) {
            GpuLoweringCoverageEntry metalEntry = GpuLoweringCoverageMatrix.entryFor(ComputeBackend.GPU_METAL, opType);
            assertEquals(GpuLoweringCoverageStatus.SUPPORTED, metalEntry.status(), opType.name());
            assertEquals(GpuLoweringUnsupportedReason.SUPPORTED, metalEntry.reason(), opType.name());
        }

        for (Operation.OpType opType : List.of(
                Operation.OpType.CONV2D_BACKWARD_INPUT,
                Operation.OpType.CONV2D_BACKWARD_WEIGHT,
                Operation.OpType.CONV2D_BACKWARD_INPUT_GEMM,
                Operation.OpType.CONV2D_BACKWARD_WEIGHT_GEMM,
                Operation.OpType.MAX_POOL2D_BACKWARD_INPUT,
                Operation.OpType.AVG_POOL2D_BACKWARD_INPUT
        )) {
            GpuLoweringCoverageEntry cudaEntry = GpuLoweringCoverageMatrix.entryFor(ComputeBackend.GPU_CUDA, opType);
            assertEquals(GpuLoweringCoverageStatus.UNSUPPORTED, cudaEntry.status(), opType.name());
            assertEquals(GpuLoweringUnsupportedReason.CAPABILITY_MISSING, cudaEntry.reason(), opType.name());
        }

        GpuLoweringCoverageEntry cudaMaxPoolBackward = GpuLoweringCoverageMatrix.entryFor(
                ComputeBackend.GPU_CUDA,
                Operation.OpType.MAX_POOL2D_BACKWARD_INPUT
        );
        assertEquals(GpuLoweringCoverageStatus.UNSUPPORTED, cudaMaxPoolBackward.status());
        assertEquals(GpuLoweringUnsupportedReason.CAPABILITY_MISSING, cudaMaxPoolBackward.reason());
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
                        || opType == Operation.OpType.CONV2D_BACKWARD_INPUT
                        || opType == Operation.OpType.CONV2D_BACKWARD_WEIGHT
                        || opType == Operation.OpType.CONV2D_BACKWARD_INPUT_GEMM
                        || opType == Operation.OpType.CONV2D_BACKWARD_WEIGHT_GEMM
                        || opType == Operation.OpType.MAX_POOL2D
                        || opType == Operation.OpType.MAX_POOL2D_BACKWARD_INPUT
                        || opType == Operation.OpType.AVG_POOL2D
                        || opType == Operation.OpType.AVG_POOL2D_BACKWARD_INPUT)) {
                    assertEquals(GpuLoweringCoverageStatus.SUPPORTED, entry.status());
                    assertEquals(GpuLoweringUnsupportedReason.SUPPORTED, entry.reason(),
                            () -> opType + " should be supported for scoped Metal direct conv/pool execution");
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
    void castCoverageIsScopedToMetalDTypeConversionPolicy() {
        GpuLoweringCoverageEntry metal = GpuLoweringCoverageMatrix.entryFor(ComputeBackend.GPU_METAL, Operation.OpType.CAST);
        assertEquals(GpuLoweringCoverageStatus.SUPPORTED, metal.status());
        assertEquals(GpuLoweringUnsupportedReason.SUPPORTED, metal.reason());
        assertEquals(GpuLoweringOperationFamily.DTYPE_CONVERSION, metal.family());
        assertTrue(metal.note().contains("FLOAT32 <-> BFLOAT16"));
        assertTrue(metal.note().contains("general BOOL/INT32 numeric casts remain unsupported"));

        GpuLoweringCoverageEntry cuda = GpuLoweringCoverageMatrix.entryFor(ComputeBackend.GPU_CUDA, Operation.OpType.CAST);
        assertEquals(GpuLoweringCoverageStatus.UNSUPPORTED, cuda.status());
        assertEquals(GpuLoweringUnsupportedReason.CAPABILITY_MISSING, cuda.reason());
        assertEquals(GpuLoweringOperationFamily.DTYPE_CONVERSION, cuda.family());
    }

    @Test
    void sliceGradCoverageIsScopedToMetalPadBasedBackwardLayoutPolicy() {
        GpuLoweringCoverageEntry metal = GpuLoweringCoverageMatrix.entryFor(ComputeBackend.GPU_METAL, Operation.OpType.SLICE_GRAD);
        assertEquals(GpuLoweringCoverageStatus.SUPPORTED, metal.status());
        assertEquals(GpuLoweringUnsupportedReason.SUPPORTED, metal.reason());
        assertEquals(GpuLoweringOperationFamily.LAYOUT_VIEW_ADJACENT, metal.family());
        assertTrue(metal.note().contains("step=1"));
        assertTrue(metal.note().contains("zero-fill pad"));

        GpuLoweringCoverageEntry cuda = GpuLoweringCoverageMatrix.entryFor(ComputeBackend.GPU_CUDA, Operation.OpType.SLICE_GRAD);
        assertEquals(GpuLoweringCoverageStatus.UNSUPPORTED, cuda.status());
        assertEquals(GpuLoweringUnsupportedReason.UNSUPPORTED_OPERATION, cuda.reason());
        assertEquals(GpuLoweringOperationFamily.LAYOUT_VIEW_ADJACENT, cuda.family());
    }

    @Test
    void unaryMathParityOpsAreMetalSupportedAndCudaCapabilityMissing() {
        for (Operation.OpType opType : List.of(Operation.OpType.ERF, Operation.OpType.FLOOR, Operation.OpType.CEIL, Operation.OpType.SIGN)) {
            GpuLoweringCoverageEntry metal = GpuLoweringCoverageMatrix.entryFor(ComputeBackend.GPU_METAL, opType);
            assertEquals(GpuLoweringCoverageStatus.SUPPORTED, metal.status(), opType.name());
            assertEquals(GpuLoweringUnsupportedReason.SUPPORTED, metal.reason(), opType.name());
            assertEquals(GpuLoweringOperationFamily.ELEMENTWISE_CHAIN, metal.family(), opType.name());
            assertTrue(metal.note().contains("MPSGraph-first"), opType.name());

            GpuLoweringCoverageEntry cuda = GpuLoweringCoverageMatrix.entryFor(ComputeBackend.GPU_CUDA, opType);
            assertEquals(GpuLoweringCoverageStatus.UNSUPPORTED, cuda.status(), opType.name());
            assertEquals(GpuLoweringUnsupportedReason.CAPABILITY_MISSING, cuda.reason(), opType.name());
            assertEquals(GpuLoweringOperationFamily.ELEMENTWISE_CHAIN, cuda.family(), opType.name());
        }
    }

    @Test
    void phaseSeventeenNonSupportedRowsUseStableReasonCodes() {
        for (ComputeBackend backend : List.of(ComputeBackend.GPU_METAL, ComputeBackend.GPU_CUDA)) {
            List<Operation.OpType> nonSupportedOps = backend == ComputeBackend.GPU_METAL
                    ? List.of()
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

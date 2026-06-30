package backend.accelerator.lowering;

import backend.contract.ComputeBackend;
import org.junit.jupiter.api.Test;
import tensor.DataType;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GpuLoweredRegionManifestTest {
    @Test
    void normalizesNullCollectionsAndDefaultsBackendExtensions() {
        GpuLoweredRegionManifest manifest = new GpuLoweredRegionManifest(
                null,
                null,
                7,
                null,
                null,
                null,
                -1,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null
        );

        assertEquals("", manifest.regionId());
        assertEquals(ComputeBackend.CPU, manifest.backend());
        assertEquals(0, manifest.selectedRegionLength());
        assertTrue(manifest.orderedNodeIds().isEmpty());
        assertTrue(manifest.externalInputNodeIds().isEmpty());
        assertTrue(manifest.outputNodeIds().isEmpty());
        assertTrue(manifest.originalOps().isEmpty());
        assertTrue(manifest.loweredPrimitives().isEmpty());
        assertTrue(manifest.inputAssumptions().isEmpty());
        assertTrue(manifest.outputAssumptions().isEmpty());
        assertTrue(manifest.rejections().isEmpty());
        assertTrue(manifest.fusedSubpatterns().isEmpty());
        assertTrue(manifest.backendExtensions().isEmpty());
        assertNotNull(manifest.fusedSummary());
        assertNotNull(manifest.candidateSpan());
    }

    @Test
    void normalizesNullFusionSubpatternsToEmptyList() {
        GpuLoweredRegionManifest manifest = new GpuLoweredRegionManifest(
                "gpu-metal-region-1",
                ComputeBackend.GPU_METAL,
                1,
                List.of(1),
                List.of(0),
                List.of(1),
                1,
                List.of(),
                List.of(),
                List.of(),
                List.of(),
                GpuCompoundRegionSummary.none(ComputeBackend.GPU_METAL, List.of(1)),
                null,
                List.of(),
                GpuLoweredRegionCandidateSpan.none(List.of(1)),
                Map.of()
        );

        assertTrue(manifest.fusedSubpatterns().isEmpty());
    }

    @Test
    void recordsBidirectionalOriginalOpToPrimitiveMapping() {
        GpuLoweredRegionOriginalOp originalOp = new GpuLoweredRegionOriginalOp(
                10,
                "LOG_SOFTMAX",
                List.of(9),
                List.of(10),
                DataType.FLOAT32,
                List.of(2, 3),
                List.of("p0", "p1"),
                List.of()
        );
        GpuLoweredPrimitiveManifest softmax = new GpuLoweredPrimitiveManifest(
                "p0",
                "SOFTMAX",
                List.of(10),
                List.of("external:0"),
                "node:0",
                DataType.FLOAT32,
                List.of(2, 3),
                List.of()
        );
        GpuLoweredPrimitiveManifest log = new GpuLoweredPrimitiveManifest(
                "p1",
                "LOG",
                List.of(10),
                List.of("p0"),
                "node:1",
                DataType.FLOAT32,
                List.of(2, 3),
                List.of()
        );

        assertEquals(List.of("p0", "p1"), originalOp.loweredPrimitiveIds());
        assertTrue(softmax.sourceOriginalNodeIds().contains(10));
        assertTrue(log.sourceOriginalNodeIds().contains(10));
    }

    @Test
    void recordsDtypeLayoutAndStorageAssumptions() {
        GpuLoweredRegionValueAssumption assumption = new GpuLoweredRegionValueAssumption(
                12,
                "input",
                DataType.FLOAT32,
                2,
                List.of(4, 8),
                "CONTIGUOUS",
                true,
                false,
                0L
        );

        assertEquals(12, assumption.nodeId());
        assertEquals("input", assumption.role());
        assertEquals(DataType.FLOAT32, assumption.dataType());
        assertEquals(2, assumption.rank());
        assertEquals(List.of(4, 8), assumption.shape());
        assertEquals("CONTIGUOUS", assumption.layout());
        assertTrue(assumption.contiguous());
    }

    @Test
    void recordsPrimitiveBoundaryAndFusedRejectionReasons() {
        GpuLoweredRegionRejection primitive = new GpuLoweredRegionRejection(
                "primitive",
                10,
                "p1",
                "",
                GpuLoweringUnsupportedReason.DAG_PRIMITIVE_UNSUPPORTED,
                "primitive has no native support"
        );
        GpuLoweredRegionRejection boundary = new GpuLoweredRegionRejection(
                "region_boundary",
                -1,
                "",
                "",
                GpuLoweringUnsupportedReason.DAG_REGION_BOUNDARY_MATERIALIZATION,
                "CPU consumer requested readable storage"
        );
        GpuLoweredRegionRejection fused = new GpuLoweredRegionRejection(
                "fused_subpattern",
                11,
                "",
                "ELEMENTWISE_CHAIN",
                GpuLoweringUnsupportedReason.DAG_FUSED_SUBPATTERN_REJECTED,
                "fused subpattern rejected"
        );

        assertEquals(GpuLoweringUnsupportedReason.DAG_PRIMITIVE_UNSUPPORTED, primitive.reason());
        assertEquals(GpuLoweringUnsupportedReason.DAG_REGION_BOUNDARY_MATERIALIZATION, boundary.reason());
        assertEquals(GpuLoweringUnsupportedReason.DAG_FUSED_SUBPATTERN_REJECTED, fused.reason());
        assertEquals("ELEMENTWISE_CHAIN", fused.fusedPatternType());
    }

    @Test
    void recordsCandidateShorteningEvidence() {
        GpuLoweredRegionCandidateSpan span = new GpuLoweredRegionCandidateSpan(
                List.of(10, 11, 12),
                List.of(10, 11),
                12,
                "p2",
                GpuLoweringUnsupportedReason.DAG_CANDIDATE_SHORTENED
        );
        GpuLoweredRegionManifest manifest = new GpuLoweredRegionManifest(
                "gpu-metal-region-10",
                ComputeBackend.GPU_METAL,
                10,
                List.of(10, 11),
                List.of(1, 2),
                List.of(11),
                2,
                List.of(),
                List.of(),
                List.of(),
                List.of(),
                GpuCompoundRegionSummary.none(ComputeBackend.GPU_METAL, List.of(10, 11)),
                List.of(new GpuLoweredRegionRejection(
                        "primitive",
                        12,
                        "p2",
                        "",
                        GpuLoweringUnsupportedReason.DAG_CANDIDATE_SHORTENED,
                        "candidate shortened"
                )),
                span,
                Map.of("dagNodeCount", "2")
        );

        assertEquals(List.of(10, 11, 12), manifest.candidateSpan().originalCandidateNodeIds());
        assertEquals(List.of(10, 11), manifest.candidateSpan().acceptedNodeIds());
        assertEquals(12, manifest.candidateSpan().rejectedOriginalNodeId());
        assertEquals("p2", manifest.candidateSpan().rejectedPrimitiveId());
        assertEquals(GpuLoweringUnsupportedReason.DAG_CANDIDATE_SHORTENED, manifest.candidateSpan().reason());
        assertEquals("2", manifest.backendExtensions().get("dagNodeCount"));
    }

    @Test
    void compactRendererIncludesStableSectionsAndManifestEvidence() {
        GpuLoweredRegionManifest manifest = new GpuLoweredRegionManifest(
                "gpu-metal-region-10",
                ComputeBackend.GPU_METAL,
                10,
                List.of(10),
                List.of(9),
                List.of(10),
                1,
                List.of(new GpuLoweredRegionOriginalOp(
                        10,
                        "LOG_SOFTMAX",
                        List.of(9),
                        List.of(10),
                        DataType.FLOAT32,
                        List.of(2, 3),
                        List.of("p0"),
                        List.of(GpuLoweringUnsupportedReason.DAG_PRIMITIVE_UNSUPPORTED)
                )),
                List.of(new GpuLoweredPrimitiveManifest(
                        "p0",
                        "SOFTMAX",
                        List.of(10),
                        List.of("external:0"),
                        "node:0",
                        DataType.FLOAT32,
                        List.of(2, 3),
                        List.of(GpuLoweringUnsupportedReason.DAG_PRIMITIVE_UNSUPPORTED)
                )),
                List.of(),
                List.of(),
                GpuCompoundRegionSummary.none(ComputeBackend.GPU_METAL, List.of(10)),
                List.of(new GpuLoweredRegionRejection(
                        "primitive",
                        10,
                        "p0",
                        "",
                        GpuLoweringUnsupportedReason.DAG_PRIMITIVE_UNSUPPORTED,
                        "primitive rejected"
                )),
                GpuLoweredRegionCandidateSpan.none(List.of(10)),
                Map.of()
        );

        String rendered = GpuLoweredRegionManifestRenderer.renderCompact(manifest);

        assertTrue(rendered.contains("GPU Lowered Region"));
        assertTrue(rendered.contains("Original Ops"));
        assertTrue(rendered.contains("Lowered Primitives"));
        assertTrue(rendered.contains("Value Assumptions"));
        assertTrue(rendered.contains("Fused Subpatterns"));
        assertTrue(rendered.contains("Rejections"));
        assertTrue(rendered.contains("regionId:"));
        assertTrue(rendered.contains("selectedRegionLength:"));
        assertTrue(rendered.contains("p0"));
        assertTrue(rendered.contains("LOG_SOFTMAX"));
        assertTrue(rendered.contains("SOFTMAX"));
        assertTrue(rendered.contains("DAG_PRIMITIVE_UNSUPPORTED"));
    }
}

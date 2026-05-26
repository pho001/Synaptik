package backend.metal.lowering;

import operations.Operation;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MetalOperationParityMatrixTest {
    @Test
    void matrixMarksMinMaxPowAsMpsGraphMappedBufferExecutableOps() {
        assertMpsGraphMappedElementwise(Operation.OpType.MIN);
        assertMpsGraphMappedElementwise(Operation.OpType.MAX);
        assertMpsGraphMappedElementwise(Operation.OpType.POW);
    }

    @Test
    void matrixMarksExpandAndSelectAsMpsGraphMappedLayoutOps() {
        assertMpsGraphMappedLayout(Operation.OpType.EXPAND);
        assertMpsGraphMappedLayout(Operation.OpType.SELECT);
    }

    @Test
    void matrixMarksCastAsScopedMpsGraphDTypeConversion() {
        MetalOperationParityMatrix.Row row = row(Operation.OpType.CAST);

        assertTrue(row.cpuKernelAvailable());
        assertFalse(row.cpuFusable());
        assertEquals("SUPPORTED", row.metalCoverageStatus());
        assertEquals("SUPPORTED", row.metalReason());
        assertTrue(row.plannerSupported());
        assertTrue(row.dagLowerable());
        assertTrue(row.nativeMpsGraphMapped());
        assertTrue(row.bufferExecutable());
        assertFalse(row.cpuFallbackOnly());
        assertTrue(row.note().contains("FLOAT32 <-> BFLOAT16"));
    }

    @Test
    void matrixMarksSliceGradAsScopedPadBasedBackwardLayoutOp() {
        MetalOperationParityMatrix.Row row = row(Operation.OpType.SLICE_GRAD);

        assertFalse(row.cpuKernelAvailable());
        assertFalse(row.cpuFusable());
        assertEquals("SUPPORTED", row.metalCoverageStatus());
        assertEquals("SUPPORTED", row.metalReason());
        assertTrue(row.plannerSupported());
        assertTrue(row.dagLowerable());
        assertTrue(row.nativeMpsGraphMapped());
        assertTrue(row.bufferExecutable());
        assertFalse(row.cpuFallbackOnly());
        assertTrue(row.note().contains("zero-fill pad"));
    }

    @Test
    void matrixMarksUnaryMathParityOpsAsMpsGraphMapped() {
        for (Operation.OpType opType : List.of(Operation.OpType.ERF, Operation.OpType.FLOOR, Operation.OpType.CEIL, Operation.OpType.SIGN)) {
            MetalOperationParityMatrix.Row row = row(opType);

            assertTrue(row.cpuKernelAvailable(), opType.name());
            assertFalse(row.cpuFusable(), opType.name());
            assertEquals("SUPPORTED", row.metalCoverageStatus(), opType.name());
            assertEquals("SUPPORTED", row.metalReason(), opType.name());
            assertTrue(row.plannerSupported(), opType.name());
            assertTrue(row.dagLowerable(), opType.name());
            assertTrue(row.nativeMpsGraphMapped(), opType.name());
            assertTrue(row.bufferExecutable(), opType.name());
            assertFalse(row.cpuFallbackOnly(), opType.name());
        }
    }

    @Test
    void matrixMarksSdpaWeightsPublicationAsMpsGraphMappedAttentionOp() {
        MetalOperationParityMatrix.Row row = row(Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS);

        assertEquals("SUPPORTED", row.metalCoverageStatus());
        assertEquals("SUPPORTED", row.metalReason());
        assertTrue(row.plannerSupported());
        assertTrue(row.dagLowerable());
        assertTrue(row.nativeMpsGraphMapped());
        assertTrue(row.bufferExecutable());
        assertFalse(row.cpuFallbackOnly());
        assertTrue(row.note().contains("attention weights publication"));
    }

    @Test
    void matrixMarksMetalIndexWriteAndGradientOpsAsMpsGraphMapped() {
        assertMpsGraphMappedIndexWrite(Operation.OpType.SCATTER_ADD);
        assertMpsGraphMappedIndexWrite(Operation.OpType.SCATTER_ELEMENTS);
        assertMpsGraphMappedIndexWrite(Operation.OpType.SCATTER_ND);
        assertMpsGraphMappedIndexWrite(Operation.OpType.GATHER_GRAD);
        assertMpsGraphMappedIndexWrite(Operation.OpType.TAKE_ALONG_AXIS_GRAD);
    }

    @Test
    void matrixMarksGatherNdAsScopedMpsGraphMappedIndexRead() {
        MetalOperationParityMatrix.Row row = row(Operation.OpType.GATHER_ND);

        assertTrue(row.cpuKernelAvailable());
        assertFalse(row.cpuFusable());
        assertEquals("SUPPORTED", row.metalCoverageStatus());
        assertEquals("SUPPORTED", row.metalReason());
        assertTrue(row.plannerSupported());
        assertTrue(row.dagLowerable());
        assertTrue(row.nativeMpsGraphMapped());
        assertTrue(row.bufferExecutable());
        assertFalse(row.cpuFallbackOnly());
        assertTrue(row.note().contains("gatherNDWithUpdatesTensor"));
    }

    @Test
    void matrixKeepsCpuFusedOperationOutOfMetalLoweringSurface() {
        MetalOperationParityMatrix.Row row = row(Operation.OpType.FUSED);

        assertEquals("UNSUPPORTED", row.metalCoverageStatus());
        assertEquals("CPU_FUSED_OPERATION_UNSUPPORTED", row.metalReason());
        assertFalse(row.plannerSupported());
        assertFalse(row.dagLowerable());
        assertFalse(row.nativeMpsGraphMapped());
        assertFalse(row.bufferExecutable());
        assertTrue(row.cpuFallbackOnly());
    }

    @Test
    void matrixKeepsConstScalarAsInternalFusedPlanNodeOnly() {
        MetalOperationParityMatrix.Row row = row(Operation.OpType.CONST_SCALAR);

        assertEquals("UNSUPPORTED", row.metalCoverageStatus());
        assertEquals("UNSUPPORTED_OPERATION", row.metalReason());
        assertFalse(row.plannerSupported());
        assertFalse(row.dagLowerable());
        assertFalse(row.nativeMpsGraphMapped());
        assertFalse(row.bufferExecutable());
        assertTrue(row.cpuFallbackOnly());
        assertTrue(row.note().contains("internal CPU fused-plan scalar node"));
    }

    @Test
    void matrixCoversEveryOperationAndOnlyPolicyRowsAreCpuFallbackOnly() {
        Set<Operation.OpType> seen = EnumSet.noneOf(Operation.OpType.class);
        for (MetalOperationParityMatrix.Row row : MetalOperationParityMatrix.rows()) {
            seen.add(row.opType());
            if (row.cpuFallbackOnly()) {
                assertTrue(!row.plannerSupported() && !row.dagLowerable() && !row.bufferExecutable(),
                        () -> "unexpected Metal cpuFallbackOnly row: " + row);
            }
        }

        for (Operation.OpType opType : Operation.OpType.values()) {
            if (opType != Operation.OpType.UNKNOWN) {
                assertTrue(seen.contains(opType), () -> "missing Metal parity row for " + opType);
            }
        }
    }

    @Test
    void matrixDistinguishesCompoundDagLoweringFromDirectNativeMapping() {
        MetalOperationParityMatrix.Row row = row(Operation.OpType.LAYER_NORM);

        assertEquals("SUPPORTED", row.metalCoverageStatus());
        assertTrue(row.plannerSupported());
        assertTrue(row.dagLowerable());
        assertFalse(row.nativeMpsGraphMapped());
        assertTrue(row.bufferExecutable());
        assertFalse(row.cpuFallbackOnly());
    }

    @Test
    void markdownRendererPublishesStableMatrixColumnsAndGapClosureRows() {
        String markdown = MetalOperationParityMatrix.renderMarkdown();

        assertTrue(markdown.contains("| Operation | CPU kernel | CPU fusable | Metal coverage |"));
        assertTrue(markdown.contains("| MIN | yes | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED |"));
        assertTrue(markdown.contains("| MAX | yes | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED |"));
        assertTrue(markdown.contains("| POW | yes | yes | supported | yes | yes | yes | yes | no | no | SUPPORTED |"));
        assertTrue(markdown.contains("| ERF | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED |"));
        assertTrue(markdown.contains("| FLOOR | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED |"));
        assertTrue(markdown.contains("| CEIL | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED |"));
        assertTrue(markdown.contains("| SIGN | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED |"));
        assertTrue(markdown.contains("| EXPAND | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED |"));
        assertTrue(markdown.contains("| SELECT | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED |"));
        assertTrue(markdown.contains("| CAST | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED |"));
        assertTrue(markdown.contains("| SLICE_GRAD | no | no | supported | yes | yes | yes | yes | no | no | SUPPORTED |"));
        assertTrue(markdown.contains("| SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED |"));
        assertTrue(markdown.contains("| SCATTER_ADD | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED |"));
        assertTrue(markdown.contains("| SCATTER_ELEMENTS | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED |"));
        assertTrue(markdown.contains("| SCATTER_ND | yes | no | supported | yes | yes | yes | yes | no | no | SUPPORTED |"));
        assertTrue(markdown.contains("| GATHER_GRAD | no | no | supported | yes | yes | yes | yes | no | no | SUPPORTED |"));
        assertTrue(markdown.contains("| TAKE_ALONG_AXIS_GRAD | no | no | supported | yes | yes | yes | yes | no | no | SUPPORTED |"));
        assertTrue(markdown.contains("| CONST_SCALAR | no | no | unsupported | no | no | no | no | no | yes | UNSUPPORTED_OPERATION |"));
        assertTrue(markdown.contains("| FUSED |"));
        assertTrue(markdown.contains("CPU_FUSED_OPERATION_UNSUPPORTED"));
    }

    @Test
    void checkedInMarkdownMatchesGeneratedParityMatrix() throws IOException {
        assertEquals(MetalOperationParityMatrix.renderMarkdown(), Files.readString(Path.of("docs/metal-operation-parity.md")));
    }

    private static void assertMpsGraphMappedElementwise(Operation.OpType opType) {
        MetalOperationParityMatrix.Row row = row(opType);

        assertTrue(row.cpuKernelAvailable());
        assertTrue(row.cpuFusable());
        assertEquals("SUPPORTED", row.metalCoverageStatus());
        assertEquals("SUPPORTED", row.metalReason());
        assertTrue(row.plannerSupported());
        assertTrue(row.dagLowerable());
        assertTrue(row.nativeMpsGraphMapped());
        assertTrue(row.bufferExecutable());
        assertFalse(row.customRouteEligible());
        assertFalse(row.cpuFallbackOnly());
        assertTrue(row.note().contains("MPSGraph-first elementwise parity gap closed"));
    }

    private static void assertMpsGraphMappedLayout(Operation.OpType opType) {
        MetalOperationParityMatrix.Row row = row(opType);

        assertTrue(row.cpuKernelAvailable());
        assertFalse(row.cpuFusable());
        assertEquals("SUPPORTED", row.metalCoverageStatus());
        assertEquals("SUPPORTED", row.metalReason());
        assertTrue(row.plannerSupported());
        assertTrue(row.dagLowerable());
        assertTrue(row.nativeMpsGraphMapped());
        assertTrue(row.bufferExecutable());
        assertFalse(row.customRouteEligible());
        assertFalse(row.cpuFallbackOnly());
        assertTrue(row.note().contains("native accelerator DAG shape ops"));
    }

    private static void assertMpsGraphMappedIndexWrite(Operation.OpType opType) {
        MetalOperationParityMatrix.Row row = row(opType);

        if (opType == Operation.OpType.GATHER_GRAD || opType == Operation.OpType.TAKE_ALONG_AXIS_GRAD) {
            assertFalse(row.cpuKernelAvailable());
        } else {
            assertTrue(row.cpuKernelAvailable());
        }
        assertFalse(row.cpuFusable());
        assertEquals("SUPPORTED", row.metalCoverageStatus());
        assertEquals("SUPPORTED", row.metalReason());
        assertTrue(row.plannerSupported());
        assertTrue(row.dagLowerable());
        assertTrue(row.nativeMpsGraphMapped());
        assertTrue(row.bufferExecutable());
        assertFalse(row.cpuFallbackOnly());
        assertTrue(row.note().contains(opType == Operation.OpType.SCATTER_ND ? "scatterNDWithDataTensor" : "scatterAlongAxis"));
    }

    private static MetalOperationParityMatrix.Row row(Operation.OpType opType) {
        return MetalOperationParityMatrix.rows().stream()
                .filter(candidate -> candidate.opType() == opType)
                .findFirst()
                .orElseThrow();
    }
}

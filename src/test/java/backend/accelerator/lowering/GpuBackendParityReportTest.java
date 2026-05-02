package backend.accelerator.lowering;

import operations.Operation;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GpuBackendParityReportTest {
    @Test
    void metalSupportedCudaGapsAreReportedWithRequiredEvidence() {
        GpuBackendParityReport report = GpuBackendParityReporter.cudaAgainstMetal();
        Set<Operation.OpType> gapOps = report.gapRows().stream()
                .map(GpuBackendParityRow::opType)
                .collect(Collectors.toSet());

        assertTrue(gapOps.contains(Operation.OpType.CONV2D));
        assertTrue(gapOps.contains(Operation.OpType.GATHER));
        assertTrue(gapOps.contains(Operation.OpType.TAKE_ALONG_AXIS));
        assertTrue(gapOps.contains(Operation.OpType.NLL_LOSS));
        assertTrue(gapOps.contains(Operation.OpType.CROSS_ENTROPY_LOSS));
        assertTrue(gapOps.contains(Operation.OpType.GT));
        assertTrue(gapOps.contains(Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION));
    }

    @Test
    void sharedSupportedRowsAreNotParityGaps() {
        GpuBackendParityReport report = GpuBackendParityReporter.cudaAgainstMetal();
        Set<Operation.OpType> gapOps = report.gapRows().stream()
                .map(GpuBackendParityRow::opType)
                .collect(Collectors.toSet());

        assertFalse(gapOps.contains(Operation.OpType.MATMUL));
        assertFalse(gapOps.contains(Operation.OpType.LINEAR));
        assertFalse(gapOps.contains(Operation.OpType.SOFTMAX));
        assertFalse(gapOps.contains(Operation.OpType.SUM));
        assertFalse(gapOps.contains(Operation.OpType.MEAN));
        assertFalse(gapOps.contains(Operation.OpType.LAYER_NORM));
        assertFalse(gapOps.contains(Operation.OpType.RMS_NORM));
    }

    @Test
    void cudaCapabilityMissingRowsNeverCountAsSupported() {
        GpuBackendParityReport report = GpuBackendParityReporter.cudaAgainstMetal();

        for (GpuBackendParityRow row : report.rows()) {
            if (row.cudaReason() == GpuLoweringUnsupportedReason.CAPABILITY_MISSING) {
                assertFalse(row.cudaSupported(), row.opType().name());
            }
        }
    }

    @Test
    void evidenceGroupsExposeNativeAndDtypeWork() {
        Map<String, java.util.List<GpuBackendParityRow>> groups =
                GpuBackendParityReporter.cudaAgainstMetal().rowsByRequiredEvidence();

        assertTrue(groups.containsKey(GpuBackendParityRow.CUDA_NATIVE_EXECUTION_REQUIRED));
        assertTrue(groups.containsKey(GpuBackendParityRow.CUDA_DTYPE_OR_LAYOUT_CONTRACT_REQUIRED));
    }
}

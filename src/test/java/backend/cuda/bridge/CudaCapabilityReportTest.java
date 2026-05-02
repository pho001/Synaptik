package backend.cuda.bridge;

import org.junit.jupiter.api.Test;

import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CudaCapabilityReportTest {
    @Test
    void unavailableCapabilityReportNeverCountsSkipAsSupport() {
        CudaCapabilityReport report = CudaBridgeCapabilities
                .unavailable(CudaBridgeCapabilityCode.NATIVE_LIBRARY_UNAVAILABLE, "missing")
                .report();

        assertFalse(report.capabilitySkipCountsAsSupport());
        assertFalse(report.graphExecutable());
        assertFalse(report.bufferBindingExecutable());
        assertTrue(report.hasStatus(CudaCapabilityDimension.NATIVE_LIBRARY, CudaCapabilityDimensionStatus.UNAVAILABLE));
    }

    @Test
    void availableCapabilityReportTracksGraphBufferAndLayoutState() {
        CudaCapabilityReport report = CudaBridgeCapabilities
                .available(true, true, 2)
                .report();

        assertTrue(report.graphExecutable());
        assertTrue(report.bufferBindingExecutable());
        assertTrue(report.layoutAbiV2Executable());
        assertTrue(report.hasStatus(CudaCapabilityDimension.GRAPH_EXECUTION_ABI, CudaCapabilityDimensionStatus.AVAILABLE));
        assertTrue(report.hasStatus(CudaCapabilityDimension.BUFFER_BINDING_ABI, CudaCapabilityDimensionStatus.AVAILABLE));
    }

    @Test
    void reportSeparatesVendorLibrariesAndUnsupportedDTypeRoles() {
        CudaCapabilityReport report = CudaBridgeCapabilities.available(false).report();
        String details = report.entries().stream()
                .map(CudaCapabilityReport.Entry::detail)
                .collect(Collectors.joining("\n"));

        assertTrue(report.hasStatus(CudaCapabilityDimension.VENDOR_LIBRARY_ROUTE, CudaCapabilityDimensionStatus.NOT_INTEGRATED));
        assertTrue(details.contains("cuBLAS/cuDNN routing is not integrated in the CUDA graph bridge"));
        assertTrue(details.contains("role=COMPUTE_INPUT dtype=FLOAT32 code=SUPPORTED"));
        assertTrue(details.contains("role=COMPUTE_OUTPUT dtype=BFLOAT16 code=RESIDENCY_ONLY_NOT_COMPUTE"));
        assertTrue(details.contains("role=COMPUTE_OUTPUT dtype=BOOL code=RESIDENCY_ONLY_NOT_COMPUTE"));
        assertTrue(details.contains("role=COMPUTE_OUTPUT dtype=INT32 code=RESIDENCY_ONLY_NOT_COMPUTE"));
        assertTrue(details.contains("role=INDEX_INPUT dtype=INT32 code=SUPPORTED"));
        assertTrue(details.contains("role=PREDICATE_INPUT dtype=BOOL code=SUPPORTED"));
        assertTrue(details.contains("role=RESIDENCY_ONLY dtype=BFLOAT16 code=SUPPORTED"));
        assertTrue(details.contains("dtype residency is not native dtype compute"));
    }

    @Test
    void reportIncludesDagPrimitiveSummary() {
        CudaCapabilityReport report = CudaBridgeCapabilities.available(false).report();
        String details = report.entriesFor(CudaCapabilityDimension.DAG_PRIMITIVE).stream()
                .map(CudaCapabilityReport.Entry::detail)
                .collect(Collectors.joining("\n"));

        assertTrue(details.contains("CUDA supported DAG primitive rows="));
        assertTrue(details.contains("CUDA parity gap rows requiring evidence="));
    }
}

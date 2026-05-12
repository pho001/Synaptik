package onnx;

import backend.ComputeBackend;
import backend.accelerator.lowering.GpuLoweringCoverageMatrix;
import backend.accelerator.lowering.GpuLoweringCoverageStatus;
import operations.Operation;
import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class OnnxCoverageMatrixTest {
    @Test
    void matrixHasUniqueRowsForEveryImporterSupportedOp() {
        Set<String> names = new HashSet<>();
        for (OnnxCoverageMatrix.Entry entry : OnnxCoverageMatrix.entries()) {
            assertTrue(names.add(entry.onnxOp()), "duplicate ONNX coverage row: " + entry.onnxOp());
            assertFalse(entry.synaptikMapping().isBlank(), entry.onnxOp());
            assertNotNull(entry.importStatus(), entry.onnxOp());
            assertNotNull(entry.exportStatus(), entry.onnxOp());
            assertNotNull(entry.cpuStatus(), entry.onnxOp());
            assertNotNull(entry.metalStatus(), entry.onnxOp());
            assertNotNull(entry.cudaStatus(), entry.onnxOp());
        }
        Set<String> importSupportedRows = new HashSet<>();
        for (OnnxCoverageMatrix.Entry entry : OnnxCoverageMatrix.entries()) {
            if (entry.importStatus() != OnnxCoverageMatrix.CoverageStatus.UNSUPPORTED) {
                importSupportedRows.add(entry.onnxOp());
            }
        }
        assertEquals(OnnxGraphImporter.supportedOps(), importSupportedRows);
        assertTrue(names.contains("NonZero"));
        assertEquals(OnnxCoverageMatrix.CoverageStatus.UNSUPPORTED,
                OnnxCoverageMatrix.entryFor("NonZero").importStatus());
    }

    @Test
    void matrixSeparatesOnnxCoverageFromGpuCoverage() {
        assertEquals(OnnxCoverageMatrix.CoverageStatus.SUPPORTED,
                OnnxCoverageMatrix.entryFor("GatherND").importStatus());
        assertEquals(OnnxCoverageMatrix.CoverageStatus.UNSUPPORTED,
                OnnxCoverageMatrix.entryFor("GatherND").metalStatus());
        assertEquals(OnnxCoverageMatrix.CoverageStatus.SUPPORTED,
                OnnxCoverageMatrix.entryFor("Conv").metalStatus());
        assertEquals(OnnxCoverageMatrix.CoverageStatus.UNSUPPORTED,
                OnnxCoverageMatrix.entryFor("Conv").cudaStatus());
        assertEquals(OnnxCoverageMatrix.CoverageStatus.UNSUPPORTED,
                OnnxCoverageMatrix.entryFor("BatchNormalization").exportStatus());
    }

    @Test
    void singleOperationGpuStatusesMirrorGpuLoweringMatrix() {
        assertBackendStatus("Conv", Operation.OpType.CONV2D);
        assertBackendStatus("LayerNormalization", Operation.OpType.LAYER_NORM);
        assertBackendStatus("GatherElements", Operation.OpType.TAKE_ALONG_AXIS);
        assertBackendStatus("ScatterND", Operation.OpType.SCATTER_ND);
    }

    @Test
    void exportSupportedRowsAreRoundTrippedOrExplicitlyClassified() {
        Set<String> roundTripCovered = Set.of(
                "Add", "MatMul", "Conv", "LayerNormalization",
                "GatherElements", "GatherND", "ScatterElements", "ScatterND",
                "Reciprocal", "Erf", "Floor", "Ceil", "Sign"
        );
        Set<String> explicitlyClassified = Set.of(
                "Sub", "Mul", "Div", "Min", "Max", "Pow",
                "Neg", "Abs", "Relu", "Tanh", "Sigmoid", "Exp", "Log", "Sqrt",
                "Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual",
                "Not", "And", "Or", "Where", "Clip", "Cast", "Gemm",
                "MaxPool", "AveragePool",
                "Transpose", "Reshape", "Flatten", "Expand", "Pad", "Tile",
                "Squeeze", "Unsqueeze", "Slice", "Concat", "Gather",
                "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd",
                "ArgMax", "CumSum", "Softmax", "LogSoftmax"
        );

        for (OnnxCoverageMatrix.Entry entry : OnnxCoverageMatrix.entries()) {
            if (entry.exportStatus() == OnnxCoverageMatrix.CoverageStatus.SUPPORTED) {
                assertTrue(roundTripCovered.contains(entry.onnxOp()) || explicitlyClassified.contains(entry.onnxOp()),
                        "export-supported ONNX row needs a round-trip test or explicit classification: " + entry.onnxOp());
            }
        }
    }

    private static void assertBackendStatus(String onnxOp, Operation.OpType opType) {
        OnnxCoverageMatrix.Entry entry = OnnxCoverageMatrix.entryFor(onnxOp);
        assertEquals(toOnnxStatus(GpuLoweringCoverageMatrix.entryFor(ComputeBackend.GPU_METAL, opType).status()),
                entry.metalStatus(), onnxOp + " Metal status");
        assertEquals(toOnnxStatus(GpuLoweringCoverageMatrix.entryFor(ComputeBackend.GPU_CUDA, opType).status()),
                entry.cudaStatus(), onnxOp + " CUDA status");
    }

    private static OnnxCoverageMatrix.CoverageStatus toOnnxStatus(GpuLoweringCoverageStatus status) {
        return switch (status) {
            case SUPPORTED -> OnnxCoverageMatrix.CoverageStatus.SUPPORTED;
            case FALLBACK -> OnnxCoverageMatrix.CoverageStatus.PARTIAL;
            case UNSUPPORTED -> OnnxCoverageMatrix.CoverageStatus.UNSUPPORTED;
        };
    }
}

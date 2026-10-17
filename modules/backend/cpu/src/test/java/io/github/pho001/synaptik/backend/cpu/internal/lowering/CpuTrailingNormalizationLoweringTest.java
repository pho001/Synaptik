package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuTrailingNormalizationIr;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.normalization.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.List;
import org.junit.jupiter.api.Test;

public class CpuTrailingNormalizationLoweringTest {
    @Test void derivesCompleteSlicesUniqueBoundariesAndLayerOnlyScratch() {
        var layer = lower(context(true, true, DataType.FLOAT32, Shape.of(2, 3), Shape.of(3),
                List.of(0, 1, 1)));
        var rms = lower(context(false, true, DataType.FLOAT64, Shape.of(2, 3), Shape.of(3),
                List.of(0, 1)));
        var layerIr = (CpuTrailingNormalizationIr) layer.portableKernelIr();
        assertAll(() -> assertEquals(2, layer.elementCount()),
                () -> assertEquals(List.of(0, 1, 1), layerIr.positionToBoundary()),
                () -> assertEquals(3, layer.boundaryValues().size()),
                () -> assertTrue(layer.trailingNormalizationGeometry().orElseThrow()
                        .scratchSliceBytes() > 0),
                () -> assertEquals(0, rms.trailingNormalizationGeometry().orElseThrow()
                        .scratchSliceBytes()));
    }

    @Test void leadingAndNormalizedZeroExtentsNeedNoRangeOrWorkspace() {
        var leadingZero = lower(context(true, false, DataType.FLOAT64, Shape.of(0, 3),
                Shape.of(3), List.of(0)));
        var normalizedZero = lower(context(true, false, DataType.FLOAT32, Shape.of(2, 0),
                Shape.of(0), List.of(0)));
        assertAll(() -> assertEquals(0, leadingZero.elementCount()),
                () -> assertEquals(0, normalizedZero.elementCount()),
                () -> assertEquals(0, leadingZero.trailingNormalizationGeometry().orElseThrow()
                        .scratchSliceBytes()),
                () -> assertEquals(0, normalizedZero.trailingNormalizationGeometry().orElseThrow()
                        .scratchSliceBytes()));
    }

    public static CpuPartitionLowering.LoweredPartition lower(
            PrepareContext<CpuPartitionAnalysisInputs> context) {
        return new CpuPartitionLowering().lower(context);
    }

    public static PrepareContext<CpuPartitionAnalysisInputs> context(boolean layer, boolean affine,
            DataType type, Shape inputShape, Shape normalizedShape, List<Integer> occurrences) {
        Object attrs = layer ? affine
                ? new AffineLayerNormAttrs(normalizedShape, epsilon(type))
                : new LayerNormAttrs(normalizedShape, epsilon(type))
                : new RmsNormAttrs(normalizedShape, epsilon(type));
        var operation = new Operation(layer ? LayerNormKind.LAYER_NORM : RmsNormKind.RMS_NORM,
                (io.github.pho001.synaptik.model.operation.OperationAttrs) attrs);
        int inputs = affine ? (layer ? 3 : 2) : 1;
        var descriptors = new java.util.ArrayList<io.github.pho001.synaptik.model.tensor.TensorDescriptor>();
        descriptors.add(CpuScatterLoweringTest.desc(type, inputShape));
        for (int i = 1; i < inputs; i++) descriptors.add(CpuScatterLoweringTest.desc(type, normalizedShape));
        return CpuScatterLoweringTest.context(operation, occurrences, descriptors,
                CpuScatterLoweringTest.desc(type, inputShape));
    }

    private static ScalarValue epsilon(DataType type) {
        return type == DataType.FLOAT64 ? ScalarValue.float64(1e-5)
                : type == DataType.FLOAT32 ? ScalarValue.float32(1e-5f)
                : ScalarValue.bfloat16Bits((short) 0x3728);
    }
}

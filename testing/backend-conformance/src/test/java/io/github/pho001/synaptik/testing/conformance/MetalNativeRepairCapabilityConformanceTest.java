package io.github.pho001.synaptik.testing.conformance;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.metal.MetalCapabilityProvider;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Confirms the repaired native cases already belong to Metal's unchanged provider domain. */
final class MetalNativeRepairCapabilityConformanceTest {
    private final MetalCapabilityProvider metal = new MetalCapabilityProvider();

    @Test
    void rankSixteenFalseBranchAndLastAxisBroadcastRemainAdmitted() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            for (boolean broadcast : List.of(false, true)) {
                long[] output = new long[16];
                Arrays.fill(output, 1);
                output[14] = 2;
                output[15] = 3;
                long[] falseShape = output.clone();
                if (broadcast) falseShape[15] = 1;
                assertTrue(metal.supports(new OperationCapabilityQuery(
                        new Operation(WhereSelectionKind.WHERE, NoOperationAttrs.INSTANCE),
                        List.of(value(DataType.BOOL, 3), value(type, 2, broadcast ? 3 : 1),
                                value(type, falseShape)), List.of(value(type, output)))));
            }
        }
        assertFalse(metal.supports(new OperationCapabilityQuery(
                new Operation(WhereSelectionKind.WHERE, NoOperationAttrs.INSTANCE),
                List.of(value(DataType.BOOL, 3), value(DataType.BFLOAT16, 3),
                        value(DataType.FLOAT16, 3)), List.of(value(DataType.FLOAT32, 3)))));
    }

    @Test
    void bothLowTypesRetainScatterIndexWidthsAndEveryScanTraversal() {
        for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
            for (DataType indices : List.of(DataType.INT32, DataType.INT64)) {
                assertTrue(metal.supports(new OperationCapabilityQuery(
                        new Operation(AxisScatterKind.SCATTER_ADD, new IndexAxisAttrs(0)),
                        List.of(value(type, 9), value(indices, 2), value(type, 2)),
                        List.of(value(type, 9)))));
            }
            for (var kind : CumulativeScanKind.values()) {
                for (boolean exclusive : List.of(false, true)) {
                    for (boolean reverse : List.of(false, true)) {
                        assertTrue(metal.supports(new OperationCapabilityQuery(
                                new Operation(kind, new CumulativeScanAttrs(1, exclusive, reverse)),
                                List.of(value(type, 2, 3)), List.of(value(type, 2, 3)))));
                    }
                }
            }
        }
    }

    private static TensorDescriptor value(DataType type, long... dimensions) {
        Shape shape = Shape.of(dimensions);
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), false);
    }
}

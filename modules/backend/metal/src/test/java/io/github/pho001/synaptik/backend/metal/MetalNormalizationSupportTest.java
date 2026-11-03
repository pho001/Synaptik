package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Independent raw occurrence checks use original carriers, scalar bits and all sibling roles. */
class MetalNormalizationSupportTest {
    private static final List<DataType> TYPES = List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16);

    @Test
    void trainingRetainsEveryCombinationOfOriginalInputAndCanonicalSiblingGradientFlags() {
        for (DataType type : TYPES) {
            var node = node(MetalMpsGraphProgram.NodeKind.BATCH_NORM_TRAINING,
                    new int[] {0, 1, 2, 3, 4}, new int[] {5, 6, 7, 8, 9},
                    MetalMpsGraphProgram.AttributeKind.BATCH_NORM_TRAINING,
                    new long[] {1, scalarType(type), scalarBits(type, 0.5f), scalarType(type), scalarBits(type, 1)});
            for (int flags = 0; flags < 32; flags++) {
                var values = new ArrayList<MetalMpsGraphProgram.ValueDescriptor>();
                for (int role = 0; role < 5; role++) values.add(value(type, (flags & (1 << role)) != 0,
                        role == 0 ? new long[] {2, 2} : new long[] {2}));
                boolean x = (flags & 1) != 0;
                boolean[] gradients = {(flags & 7) != 0, x || (flags & 8) != 0,
                        x || (flags & 16) != 0, x, x};
                for (int role = 0; role < 5; role++) values.add(value(type, gradients[role],
                        role == 0 ? new long[] {2, 2} : new long[] {2}));
                assertDoesNotThrow(() -> MetalNormalizationSupport.validate(node, values));
                for (int role = 0; role < 5; role++) {
                    var corrupted = new ArrayList<>(values);
                    corrupted.set(5 + role, value(type, !gradients[role],
                            role == 0 ? new long[] {2, 2} : new long[] {2}));
                    assertThrows(IllegalArgumentException.class, () -> MetalNormalizationSupport.validate(node, corrupted));
                }
            }
        }
    }

    @Test
    void mixedAffineRolesPromoteButScalarAndOutputCarriersCannotBeProjected() {
        var layer = node(MetalMpsGraphProgram.NodeKind.LAYER_NORM,
                new int[] {0, 1, 2}, new int[] {3}, MetalMpsGraphProgram.AttributeKind.NORMALIZED_SHAPE_EPSILON,
                new long[] {1, 2, 1, scalarBits(DataType.FLOAT32, 1)});
        var values = new ArrayList<>(List.of(value(DataType.BFLOAT16, true, 2, 2),
                value(DataType.FLOAT16, false, 2), value(DataType.BFLOAT16, false, 2),
                value(DataType.FLOAT32, true, 2, 2)));
        assertDoesNotThrow(() -> MetalNormalizationSupport.validate(layer, values));
        values.set(3, value(DataType.BFLOAT16, true, 2, 2));
        assertThrows(IllegalArgumentException.class, () -> MetalNormalizationSupport.validate(layer, values));
        for (DataType type : TYPES) {
            var rms = node(MetalMpsGraphProgram.NodeKind.RMS_NORM, new int[] {0, 1}, new int[] {2},
                    MetalMpsGraphProgram.AttributeKind.NORMALIZED_SHAPE_EPSILON,
                    new long[] {1, 2, scalarType(type), scalarBits(type, 1)});
            var homogeneous = List.of(value(type, false, 3, 2), value(type, true, 2), value(type, true, 3, 2));
            assertDoesNotThrow(() -> MetalNormalizationSupport.validate(rms, homogeneous));
            var wrong = node(rms.kind(), rms.inputs(), rms.outputs(), rms.attributeKind(),
                    new long[] {1, 2, type == DataType.BFLOAT16 ? 1 : 5, 0x3f80});
            assertThrows(IllegalArgumentException.class, () -> MetalNormalizationSupport.validate(wrong, homogeneous));
        }
    }

    @Test
    void originalIntegerTargetsStayIntegerAndEveryLossReductionPreservesOutputRoles() {
        for (DataType type : TYPES) for (int reduction = 1; reduction <= 3; reduction++) {
            for (var kind : List.of(MetalMpsGraphProgram.NodeKind.MEAN_SQUARED_ERROR,
                    MetalMpsGraphProgram.NodeKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                    MetalMpsGraphProgram.NodeKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS)) {
                boolean mse = kind == MetalMpsGraphProgram.NodeKind.MEAN_SQUARED_ERROR;
                boolean index = kind == MetalMpsGraphProgram.NodeKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS;
                var attr = mse ? MetalMpsGraphProgram.AttributeKind.MSE : index
                        ? MetalMpsGraphProgram.AttributeKind.INDEX_CROSS_ENTROPY
                        : MetalMpsGraphProgram.AttributeKind.DENSE_CROSS_ENTROPY;
                var node = node(kind, new int[] {0, 1}, new int[] {2}, attr,
                        mse ? new long[] {reduction} : index ? new long[] {1, reduction, 0} : new long[] {1, reduction});
                long[] shape = reduction != 1 ? new long[0] : mse ? new long[] {2, 3} : new long[] {2};
                var values = List.of(value(type, true, 2, 3),
                        value(index ? DataType.INT64 : type, !index, index ? new long[] {2} : new long[] {2, 3}),
                        value(type, true, shape));
                assertDoesNotThrow(() -> MetalNormalizationSupport.validate(node, values));
                if (index) {
                    var malformed = new ArrayList<>(values);
                    malformed.set(1, value(DataType.INT64, true, 2));
                    assertThrows(IllegalArgumentException.class, () -> MetalNormalizationSupport.validate(node, malformed));
                }
            }
        }
    }

    private static long scalarType(DataType type) { return type == DataType.BFLOAT16 ? 5 : 1; }
    private static long scalarBits(DataType type, float value) {
        return type == DataType.BFLOAT16 ? Short.toUnsignedLong(io.github.pho001.synaptik.model.datatype.BFloat16Bits.fromFloat(value))
                : Integer.toUnsignedLong(Float.floatToRawIntBits(value));
    }
    private static MetalMpsGraphProgram.ValueDescriptor value(DataType type, boolean gradient, long... shape) {
        return new MetalMpsGraphProgram.ValueDescriptor(type, shape, gradient);
    }
    private static MetalMpsGraphProgram.Node node(MetalMpsGraphProgram.NodeKind kind, int[] inputs, int[] outputs,
            MetalMpsGraphProgram.AttributeKind attr, long[] words) {
        return MetalMpsGraphProgram.Node.generic(kind, inputs, outputs, attr, words);
    }
}

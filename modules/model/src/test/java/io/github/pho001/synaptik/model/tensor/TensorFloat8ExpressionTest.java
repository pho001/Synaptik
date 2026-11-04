package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionAttrs;
import io.github.pho001.synaptik.model.operation.normalization.LayerNormAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import org.junit.jupiter.api.Test;

class TensorFloat8ExpressionTest {
    private static final List<DataType> FP8 = List.of(DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2);

    @Test
    void rawByteImportCopiesAllPatternsWithoutBooleanNormalizationOrSourceRetention() {
        byte[] raw = new byte[256];
        for (int index = 0; index < raw.length; index++) raw[index] = (byte) index;
        for (DataType type : FP8) {
            Shape shape = Shape.of(256);
            TensorDescriptor descriptor = new TensorDescriptor(type, shape,
                    Optional.of(LayoutDescriptor.contiguous(shape)), true);
            Tensor tensor = TensorFactory.fromFlatArray(descriptor, Optional.empty(), raw);
            assertArrayEquals(raw, bytes(tensor));
            tensor.hostStorage().orElseThrow().segment().set(ValueLayout.JAVA_BYTE, 0, (byte) 42);
            assertEquals(0, raw[0]);
            assertEquals(1, type.byteWidth());
            assertTrue(tensor.descriptor().requiresGrad());
        }
        Shape shape = Shape.of(256);
        Tensor bool = TensorFactory.fromFlatArray(new TensorDescriptor(DataType.BOOL, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false), Optional.empty(), raw);
        byte[] normalized = new byte[256];
        java.util.Arrays.fill(normalized, 1, normalized.length, (byte) 1);
        assertArrayEquals(normalized, bytes(bool));
    }

    @Test
    void constantsRetainFp8TypeAndExactZeroOneAndConvertedFill() {
        for (DataType type : FP8) {
            byte one = (byte) (type == DataType.FLOAT8_E4M3 ? 0x38 : 0x3C);
            assertArrayEquals(new byte[] {0, 0}, bytes(TensorFactory.zeros(
                    Shape.of(2), type, Optional.empty(), true)));
            assertArrayEquals(new byte[] {one, one}, bytes(TensorFactory.ones(
                    Shape.of(2), type, Optional.empty(), true)));
            assertArrayEquals(new byte[] {one, 0, 0, one}, bytes(TensorFactory.identityMatrix(
                    2, 2, type, Optional.empty(), true)));
        }
        assertArrayEquals(new byte[] {(byte) 0x80}, bytes(
                TensorFactory.scalarFloat8E4m3(-0.0f, Optional.empty(), true)));
        assertArrayEquals(new byte[] {(byte) 0xFC}, bytes(
                TensorFactory.scalarFloat8E5m2(Float.NEGATIVE_INFINITY, Optional.empty(), false)));
        assertArrayEquals(new byte[] {0x7F, 0x7F}, bytes(TensorFactory.fullFloat8E4m3(
                Shape.of(2), 465.0f, Optional.empty(), false)));
        assertArrayEquals(new byte[] {0x3C, 0x3C}, bytes(TensorFactory.fullFloat8E5m2(
                Shape.of(2), 1.125f, Optional.empty(), false)));
    }

    @Test
    void unlikeLowOperandsPromoteToFloat32WhileHomogeneousAndFloat64BoundariesRemainVisible() {
        List<DataType> floating = List.of(DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2,
                DataType.FLOAT16, DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64);
        for (DataType left : floating) for (DataType right : floating) {
            DataType expected = left == DataType.FLOAT64 || right == DataType.FLOAT64
                    ? DataType.FLOAT64 : left == right ? left : DataType.FLOAT32;
            assertEquals(expected, DataTypePromotion.promoteFloating(left, right));
            Tensor result = leaf(left, Shape.of(2, 3)).add(leaf(right, Shape.of(2, 3)));
            assertEquals(expected, result.descriptor().dataType());
            assertTrue(result.descriptor().requiresGrad());
            assertTrue(result.hostStorage().isEmpty());
        }
    }

    @Test
    void normalizationAndAttentionUseExactFloat32ScalarMetadataForFp8() {
        ScalarValue epsilon = ScalarValue.float32(1.0e-5f);
        ScalarValue momentum = ScalarValue.float32(0.1f);
        for (DataType type : FP8) {
            Tensor input = leaf(type, Shape.of(2, 3));
            Tensor channel = leaf(type, Shape.of(3));
            Tensor layer = input.layerNorm(Shape.of(3), epsilon);
            assertEquals(type, layer.descriptor().dataType());
            assertSame(epsilon, ((LayerNormAttrs) layer.provenance().orElseThrow()
                    .operation().attrs()).epsilon());
            assertEquals(type, input.rmsNorm(Shape.of(3), epsilon).descriptor().dataType());
            assertEquals(type, input.batchNormInference(1, channel, channel, channel,
                    channel, epsilon).descriptor().dataType());
            input.batchNormTraining(1, channel, channel, channel, channel, momentum, epsilon);
            Tensor query = leaf(type, Shape.of(2, 3));
            ScaledDotProductAttentionAttrs attrs = new ScaledDotProductAttentionAttrs(
                    Optional.of(ScalarValue.float32(0.5f)), false);
            Tensor attention = query.scaledDotProductAttention(query, query, attrs);
            assertEquals(type, attention.descriptor().dataType());
            assertSame(attrs, attention.provenance().orElseThrow().operation().attrs());
            ScalarValue low = type == DataType.FLOAT8_E4M3
                    ? ScalarValue.float8E4m3(0.5f) : ScalarValue.float8E5m2(0.5f);
            assertThrows(IllegalArgumentException.class, () -> input.layerNorm(Shape.of(3), low));
            assertThrows(IllegalArgumentException.class, () -> query.scaledDotProductAttention(
                    query, query, new ScaledDotProductAttentionAttrs(Optional.of(low), false)));
        }
    }

    @Test
    void eagerRandomInitializersNarrowBinary64SamplesDirectlyToFp8() {
        for (DataType type : FP8) {
            // Both values lie just above an even lower endpoint's midpoint but round to the
            // midpoint in binary32. A hidden binary32 intermediate would produce 1, not the upper.
            double midpoint = type == DataType.FLOAT8_E4M3 ? 1.0625d : 1.125d;
            double witness = Math.nextUp(midpoint);
            Random source = new Random(0) {
                @Override public double nextGaussian() { return witness; }
                @Override public double nextDouble(double origin, double bound) { return witness; }
            };
            byte expected = (byte) (type == DataType.FLOAT8_E4M3 ? 0x39 : 0x3D);
            Tensor normal = TensorRandoms.randomNormal(Shape.of(2), type, 0.0d, 1.0d,
                    source, Optional.empty(), true);
            Tensor uniform = TensorRandoms.randomUniform(Shape.of(2), type, 0.0d, 2.0d,
                    source, Optional.empty(), false);
            assertArrayEquals(new byte[] {expected, expected}, bytes(normal));
            assertArrayEquals(new byte[] {expected, expected}, bytes(uniform));
            assertTrue(normal.descriptor().requiresGrad());
            assertFalse(uniform.descriptor().requiresGrad());
        }
    }

    private static Tensor leaf(DataType type, Shape shape) {
        return TensorFactory.create(new TensorDescriptor(type, shape, Optional.empty(), true));
    }

    private static byte[] bytes(Tensor tensor) {
        return tensor.hostStorage().orElseThrow().segment().toArray(ValueLayout.JAVA_BYTE);
    }
}

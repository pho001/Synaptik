package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Scan;
import io.github.pho001.synaptik.model.tensor.ScanBody;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.List;
import java.util.Optional;
import java.util.OptionalInt;
import org.junit.jupiter.api.Test;

/** Public execution distinguishes literal attention EXP sites from first-class shifted softmax. */
final class EngineAttentionNamedSitesIntegrationTest {
    @Test
    void lengthAwareSingleOutputScanPreservesActiveNegativeZeroAndSkipsInactiveNaNs() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (boolean reverse : List.of(false, true)) try (Engine engine = Engine.standard()) {
                Shape stepShape = Shape.of(2, 1, 1);
                Tensor query = TensorFactory.create(descriptor(type, stepShape));
                Tensor key = TensorFactory.create(descriptor(type, stepShape));
                Tensor value = TensorFactory.create(descriptor(type, stepShape));
                Tensor output = query.scaledDotProductAttention(key, value);
                var body = new ScanBody(List.of(), List.of(query, key, value), List.of(), List.of(), List.of(output));
                Shape sequenceShape = Shape.of(2, 2, 1, 1);
                Tensor queries = tensor(type, sequenceShape, 0, 0, Double.NaN, 0);
                Tensor keys = tensor(type, sequenceShape, 0, 0, Double.NaN, 0);
                Tensor values = tensor(type, sequenceShape, -0.0, -0.0, Double.NaN, -0.0);
                Tensor lengths = TensorFactory.fromFlatArray(descriptor(DataType.INT64, Shape.of(2)),
                        Optional.empty(), new long[] {1, 2});
                var scan = Scan.apply(List.of(), List.of(queries, keys, values), List.of(), lengths, 0,
                        List.of(OptionalInt.of(0), OptionalInt.of(0), OptionalInt.of(0)), body, reverse);
                var bytes = engine.compute(scan.outputs().getFirst()).bytes();
                for (int index = 0; index < 4; index++) {
                    if (type == DataType.FLOAT64) assertEquals(index == 2 ? 0L : Long.MIN_VALUE, bytes.getLong());
                    else assertEquals(index == 2 ? 0 : Integer.MIN_VALUE, bytes.getInt());
                }
            }
    }

    @Test
    void nonemptyValueContractionPreservesNegativeZeroIncludingMaskedFirstContributor() {
        try (Engine engine = Engine.standard()) {
            for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
                Tensor query = tensor(type, Shape.of(1, 1), 0);
                Tensor mask = TensorFactory.fromFlatArray(descriptor(DataType.BOOL, Shape.of(1, 2)),
                        Optional.empty(), new byte[] {0, 1});
                var singleton = query.scaledDotProductAttentionWithWeights(
                        tensor(type, Shape.of(1, 1), 0), tensor(type, Shape.of(1, 1), -0.0));
                var masked = query.scaledDotProductAttentionWithWeights(
                        tensor(type, Shape.of(2, 1), Double.NaN, 0),
                        tensor(type, Shape.of(2, 1), Double.NaN, -0.0), mask);
                for (HostTensorValue value : engine.compute(List.of(singleton.output(), masked.output()))) {
                    var bytes = value.bytes();
                    if (type == DataType.FLOAT64) assertEquals(Long.MIN_VALUE, bytes.getLong());
                    else assertEquals(Integer.MIN_VALUE, bytes.getInt());
                }
            }
        }
    }

    @Test
    void finiteScoreOverflowAndUnderflowDoNotBecomeShiftedSoftmax() {
        try (Engine engine = Engine.standard()) {
            for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
                double magnitude = type == DataType.FLOAT32 ? 100 : 1000;
                Tensor query = tensor(type, Shape.of(1, 1), 1);
                Tensor values = tensor(type, Shape.of(2, 1), 2, 4);
                for (double score : new double[] {magnitude, -2 * magnitude}) {
                    Tensor keys = tensor(type, Shape.of(2, 1), score, score - 1);
                    var result = query.scaledDotProductAttentionWithWeights(keys, values);
                    for (HostTensorValue publication : engine.compute(List.of(result.output(), result.weights()))) {
                        var bytes = publication.bytes();
                        while (bytes.hasRemaining()) {
                            double observed = type == DataType.FLOAT64 ? bytes.getDouble() : bytes.getFloat();
                            assertTrue(Double.isNaN(observed), "literal EXP normalization class");
                        }
                    }
                }
                var infinities = query.scaledDotProductAttentionWithWeights(
                        tensor(type, Shape.of(2, 1), Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY), values);
                var tied = engine.compute(List.of(infinities.output(), infinities.weights()));
                compare(type, tied.get(0), 3);
                compare(type, tied.get(1), 0.5, 0.5);
                Tensor mask = TensorFactory.fromFlatArray(descriptor(DataType.BOOL, Shape.of(1, 2)),
                        Optional.empty(), new byte[] {0, 1});
                var excluded = query.scaledDotProductAttentionWithWeights(
                        tensor(type, Shape.of(2, 1), Double.NaN, 0),
                        tensor(type, Shape.of(2, 1), Double.NaN, 4), mask);
                var guarded = engine.compute(List.of(excluded.output(), excluded.weights()));
                compare(type, guarded.get(0), 4);
                compare(type, guarded.get(1), 0, 1);
            }
        }
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape) {
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), false);
    }

    private static Tensor tensor(DataType type, Shape shape, double... values) {
        if (type == DataType.FLOAT64) return TensorFactory.fromFlatArray(descriptor(type, shape), Optional.empty(), values);
        float[] raw = new float[values.length];
        for (int i = 0; i < raw.length; i++) raw[i] = (float) values[i];
        return TensorFactory.fromFlatArray(descriptor(type, shape), Optional.empty(), raw);
    }

    private static void compare(DataType type, HostTensorValue value, double... expected) {
        var bytes = value.bytes();
        assertEquals(expected.length * type.byteWidth(), bytes.remaining());
        for (double lane : expected) assertEquals(lane, type == DataType.FLOAT64 ? bytes.getDouble() : bytes.getFloat(), 0);
    }
}

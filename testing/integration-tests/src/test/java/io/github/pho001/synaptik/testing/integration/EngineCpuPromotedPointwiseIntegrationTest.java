package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Public CPU promotion witnesses, with independent finite differences for both original carriers. */
final class EngineCpuPromotedPointwiseIntegrationTest {
    @Test void mixedFloatingArithmeticAndSelectionReturnOriginalCarrierCotangents() {
        try (Engine engine = Engine.standard()) {
            for (boolean reverse : List.of(false, true)) for (int operation = 0; operation < 8; operation++) {
                DataType leftType = reverse ? DataType.FLOAT64 : DataType.FLOAT32;
                DataType rightType = reverse ? DataType.FLOAT32 : DataType.FLOAT64;
                double[] leftValues = {1.25, 3.5, 2.25, 4.5};
                double[] rightValues = {2.5, 1.75, 3.75, 2.25};
                double[] seeds = {0.5, -1.25, 2, 0.75};
                Tensor left = floating(leftType, Shape.of(2, 2), true, leftValues);
                Tensor right = floating(rightType, Shape.of(2, 2), true, rightValues);
                Tensor condition = tensor(DataType.BOOL, Shape.of(2, 2), false, new byte[]{1, 0, 1, 0});
                Tensor output = expression(operation, left, right, condition);
                Tensor seed = floating(DataType.FLOAT64, Shape.of(2, 2), false, seeds);
                var compiled = engine.compile(List.of(output), List.of(seed), List.of(left, right));
                List<Tensor> actualInputs = operation == 7 ? List.of(left, right, condition, seed)
                        : List.of(left, right, seed);
                try (var prepared = engine.prepare(compiled);
                        var result = engine.run(prepared, actualInputs)) {
                    double[] expected = new double[4], dl = new double[4], dr = new double[4];
                    for (int i = 0; i < expected.length; i++) {
                        expected[i] = value(operation, leftValues[i], rightValues[i], i % 2 == 0);
                        double h = 1e-5;
                        dl[i] = seeds[i] * (value(operation, leftValues[i] + h, rightValues[i], i % 2 == 0)
                                - value(operation, leftValues[i] - h, rightValues[i], i % 2 == 0)) / (2 * h);
                        dr[i] = seeds[i] * (value(operation, leftValues[i], rightValues[i] + h, i % 2 == 0)
                                - value(operation, leftValues[i], rightValues[i] - h, i % 2 == 0)) / (2 * h);
                    }
                    compare(expected, result.materialize(result.publications().get(0), 1024), DataType.FLOAT64, 1e-12);
                    compare(dl, result.materialize(result.publications().get(1), 1024), leftType, 3e-5);
                    compare(dr, result.materialize(result.publications().get(2), 1024), rightType, 3e-5);
                }
            }
        }
    }

    @Test void unequalPointwiseBranchesRejoinWithoutLosingTheirStoredIntermediateOrder() {
        try (Engine engine = Engine.standard()) {
            for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
                double[] values = {-2, -0.5, 1, 3};
                Tensor input = floating(type, Shape.of(4), false, values);
                Tensor longBranch = input.neg().exp();
                Tensor shortBranch = input.abs();
                // Sibling fusion may join NEG and ABS, but their final ADD must still wait
                // for EXP outside that sibling unit.
                Tensor output = longBranch.add(shortBranch);
                try (var prepared = engine.prepare(engine.compile(List.of(output)));
                        var result = engine.run(prepared, List.of(input))) {
                    double[] expected = new double[values.length];
                    for (int i = 0; i < values.length; i++) {
                        double exponential = StrictMath.exp(-values[i]);
                        expected[i] = type == DataType.FLOAT32
                                ? (float) ((float) exponential + (float) Math.abs(values[i]))
                                : exponential + Math.abs(values[i]);
                    }
                    compare(expected, result.materialize(result.publications().getFirst(), 1024),
                            type, type == DataType.FLOAT32 ? 2e-6 : 1e-14);
                }
            }
        }
    }

    @Test void mixedBroadcastPullbacksReduceThenRestoreEachOriginalCarrier() {
        try (Engine engine = Engine.standard()) {
            Tensor x = floating(DataType.FLOAT32, Shape.of(2, 1), true, new double[]{1.25, 3.5});
            Tensor y = floating(DataType.FLOAT64, Shape.of(1, 3), true, new double[]{2, 3, 4});
            Tensor seed = floating(DataType.FLOAT64, Shape.of(2, 3), false, new double[]{1, 2, 3, 4, 5, 6});
            var compiled = engine.compile(List.of(x.mul(y)), List.of(seed), List.of(x, y));
            try (var prepared = engine.prepare(compiled); var result = engine.run(prepared, List.of(x, y, seed))) {
                compare(new double[]{2.5, 3.75, 5, 7, 10.5, 14},
                        result.materialize(result.publications().get(0), 1024), DataType.FLOAT64, 0);
                compare(new double[]{20, 47}, result.materialize(result.publications().get(1), 1024),
                        DataType.FLOAT32, 0);
                compare(new double[]{15.25, 20, 24.75}, result.materialize(result.publications().get(2), 1024),
                        DataType.FLOAT64, 0);
            }
        }
    }

    @Test void mixedIntegralArithmeticAndStoredComparisonsNeverRoundThroughDouble() {
        try (Engine engine = Engine.standard()) {
            int[] narrow = {-1, 0, Integer.MAX_VALUE, Integer.MIN_VALUE};
            long[] wide = {Long.MAX_VALUE, 9007199254740993L, 2147483648L, Long.MIN_VALUE};
            for (boolean reverse : List.of(false, true)) {
                Tensor a = tensor(DataType.INT32, Shape.of(4), false, narrow);
                Tensor b = tensor(DataType.INT64, Shape.of(4), false, wide);
                Tensor left = reverse ? b : a, right = reverse ? a : b;
                var outputs = List.of(left.add(right), left.sub(right), left.mul(right),
                        left.minimum(right), left.maximum(right), left.greaterThan(right),
                        left.greaterOrEqual(right), left.lessThan(right), left.lessOrEqual(right),
                        left.equalTo(right), left.notEqualTo(right));
                try (var prepared = engine.prepare(engine.compile(outputs)); var result = engine.run(prepared, List.of(a, b))) {
                    for (int operation = 0; operation < outputs.size(); operation++) {
                        var actual = result.materialize(result.publications().get(operation), 1024);
                        var bytes = actual.bytes();
                        for (int i = 0; i < narrow.length; i++) {
                            long x = reverse ? wide[i] : narrow[i], y = reverse ? narrow[i] : wide[i];
                            if (operation < 5) {
                                long expected = switch (operation) {
                                    case 0 -> x + y; case 1 -> x - y; case 2 -> x * y;
                                    case 3 -> Math.min(x, y); default -> Math.max(x, y);
                                };
                                assertEquals(DataType.INT64, actual.dataType());
                                assertEquals(expected, bytes.getLong());
                            } else {
                                boolean expected = switch (operation) {
                                    case 5 -> x > y; case 6 -> x >= y; case 7 -> x < y;
                                    case 8 -> x <= y; case 9 -> x == y; default -> x != y;
                                };
                                assertEquals(expected ? 1 : 0, bytes.get());
                            }
                        }
                    }
                }
            }
        }
    }

    @Test void floatingComparisonsObservePromotedStoredValuesAndSpecialClasses() {
        try (Engine engine = Engine.standard()) {
            float[] x = {Float.MIN_VALUE, -0f, Float.NaN, Float.POSITIVE_INFINITY};
            double[] y = {0, 0, 1, Double.POSITIVE_INFINITY};
            Tensor a = tensor(DataType.FLOAT32, Shape.of(4), false, x);
            Tensor b = tensor(DataType.FLOAT64, Shape.of(4), false, y);
            var outputs = List.of(a.greaterThan(b), a.greaterOrEqual(b), a.lessThan(b),
                    a.lessOrEqual(b), a.equalTo(b), a.notEqualTo(b));
            try (var prepared = engine.prepare(engine.compile(outputs)); var result = engine.run(prepared, List.of(a, b))) {
                for (int op = 0; op < outputs.size(); op++) {
                    var bytes = result.materialize(result.publications().get(op), 1024).bytes();
                    for (int i = 0; i < x.length; i++) {
                        boolean expected = switch (op) {
                            case 0 -> x[i] > y[i]; case 1 -> x[i] >= y[i]; case 2 -> x[i] < y[i];
                            case 3 -> x[i] <= y[i]; case 4 -> x[i] == y[i]; default -> x[i] != y[i];
                        };
                        assertEquals(expected ? 1 : 0, bytes.get());
                    }
                }
            }
        }
    }

    private static Tensor expression(int op, Tensor x, Tensor y, Tensor condition) {
        return switch (op) {
            case 0 -> x.add(y); case 1 -> x.sub(y); case 2 -> x.mul(y); case 3 -> x.div(y);
            case 4 -> x.minimum(y); case 5 -> x.maximum(y); case 6 -> x.pow(y);
            default -> Tensor.where(condition, x, y);
        };
    }

    private static double value(int op, double x, double y, boolean condition) {
        return switch (op) {
            case 0 -> x + y; case 1 -> x - y; case 2 -> x * y; case 3 -> x / y;
            case 4 -> Math.min(x, y); case 5 -> Math.max(x, y); case 6 -> StrictMath.pow(x, y);
            default -> condition ? x : y;
        };
    }

    private static Tensor floating(DataType type, Shape shape, boolean gradient, double[] values) {
        if (type == DataType.FLOAT64) return tensor(type, shape, gradient, values);
        float[] floats = new float[values.length];
        for (int i = 0; i < floats.length; i++) floats[i] = (float) values[i];
        return tensor(type, shape, gradient, floats);
    }

    private static Tensor tensor(DataType type, Shape shape, boolean gradient, Object values) {
        var descriptor = new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), gradient);
        return switch (values) {
            case double[] array -> TensorFactory.fromFlatArray(descriptor, Optional.empty(), array);
            case float[] array -> TensorFactory.fromFlatArray(descriptor, Optional.empty(), array);
            case int[] array -> TensorFactory.fromFlatArray(descriptor, Optional.empty(), array);
            case long[] array -> TensorFactory.fromFlatArray(descriptor, Optional.empty(), array);
            case byte[] array -> TensorFactory.fromFlatArray(descriptor, Optional.empty(), array);
            default -> throw new IllegalArgumentException("unsupported test carrier");
        };
    }

    private static void compare(double[] expected, HostTensorValue actual, DataType type, double tolerance) {
        assertEquals(type, actual.dataType());
        var bytes = actual.bytes();
        for (double value : expected) assertEquals(value,
                type == DataType.FLOAT64 ? bytes.getDouble() : bytes.getFloat(), tolerance);
    }
}

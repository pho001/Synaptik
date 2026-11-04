package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.Float16Bits;
import io.github.pho001.synaptik.model.datatype.Float8E4M3Bits;
import io.github.pho001.synaptik.model.datatype.Float8E5M2Bits;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.*;
import java.lang.foreign.*;
import java.math.BigDecimal;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Actual Compiler AD preserves saved softmax values and categorical-loss-specific guarded sites. */
class EngineCpuSoftmaxBackwardIntegrationTest {
    @Test void actualAdBothKindsReuseFreshInputsAndNonuniformSeeds() {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean log : List.of(false,true)) {
            try (var arena = Arena.ofShared(); var engine = Engine.builder().takeOwnership(CpuBackendIntegration.open()).build()) {
                var shape = Shape.of(2,3);
                var input = tensor(arena,type,shape,true,new double[]{-1,0,1,2,-2,.5});
                var seed = tensor(arena,type,shape,false,new double[]{1,-2,3,-4,5,-6});
                var output = log ? input.logSoftmax(1) : input.softmax(1);
                var compiled = engine.compile(List.of(output),List.of(seed),List.of(input));
                try (var prepared = engine.prepare(compiled)) {
                    for (int run = 1; run <= 2; run++) {
                        double[] x = {-run,.25*run,run,run+1,-run-1,.5};
                        double[] g = {run,-2*run,3*run,-4*run,5*run,-6*run};
                        var freshInput = tensor(arena,type,shape,true,x); var freshSeed = tensor(arena,type,shape,false,g);
                        var values = compiled.inputs().stream().map(i -> i.tensorId().equals(input.id()) ? freshInput : i.tensorId().equals(seed.id()) ? freshSeed : null).toList();
                        assertFalse(values.contains(null));
                        try (var result = engine.runValues(prepared,values)) {
                            assertEquals(2,result.publications().size());
                            var forward = result.materialize(result.publications().getFirst(),6L*type.byteWidth()).bytes();
                            var backward = result.materialize(result.publications().get(1),6L*type.byteWidth()).bytes();
                            double[] saved = new double[6];
                            for (int i = 0; i < 6; i++) saved[i] = type == DataType.FLOAT32 ? forward.getFloat() : forward.getDouble();
                            for (int row = 0; row < 2; row++) {
                                double m = Math.max(x[row*3],Math.max(x[row*3+1],x[row*3+2])), denominator = 0;
                                for (int f = 0; f < 3; f++) denominator += StrictMath.exp(x[row*3+f]-m);
                                for (int f = 0; f < 3; f++) assertEquals(log ? x[row*3+f]-m-StrictMath.log(denominator)
                                        : StrictMath.exp(x[row*3+f]-m)/denominator,saved[row*3+f],type == DataType.FLOAT32 ? 4e-7 : 2e-15);
                                BigDecimal exact = BigDecimal.ZERO;
                                for (int f = 0; f < 3; f++) { int i = row*3+f;
                                    exact = exact.add(new BigDecimal(log ? typed(type,g[i]) : typed(type,typed(type,g[i])*saved[i]))); }
                                double sum = type == DataType.FLOAT32 ? exact.floatValue() : exact.doubleValue();
                                for (int f = 0; f < 3; f++) {
                                    int i = row*3+f;
                                    double expected = log ? typed(type,typed(type,g[i])-typed(type,typed(type,StrictMath.exp(saved[i]))*sum))
                                            : typed(type,saved[i]*typed(type,typed(type,g[i])-sum));
                                    double actual = type == DataType.FLOAT32 ? backward.getFloat() : backward.getDouble();
                                    if (type == DataType.FLOAT32) assertEquals(Float.floatToRawIntBits((float)expected),Float.floatToRawIntBits((float)actual));
                                    else assertEquals(Double.doubleToRawLongBits(expected),Double.doubleToRawLongBits(actual));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    @Test void tiedFiniteExtremesHaveFiniteSelectedObjectivesAndNonzeroActualGradients() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (boolean log : List.of(false, true)) {
                try (var arena = Arena.ofShared(); var engine = Engine.builder()
                        .takeOwnership(CpuBackendIntegration.open()).build()) {
                    double maximum = type == DataType.FLOAT32 ? Float.MAX_VALUE : Double.MAX_VALUE;
                    Tensor input = tensor(arena, type, Shape.of(3), true,
                            new double[] {-maximum, maximum, maximum});
                    Tensor output = log ? input.logSoftmax(0) : input.softmax(0);
                    var backward = engine.backward(output.select(0, 1), List.of(input),
                            4L * type.byteWidth());
                    double objective = read(type, backward.objective().bytes());
                    assertTrue(Double.isFinite(objective));
                    assertEquals(log ? -Math.log(2) : 0.5, objective,
                            type == DataType.FLOAT32 ? 1e-7 : 1e-15);
                    var gradient = backward.gradients().getFirst().bytes();
                    assertEquals(Double.doubleToRawLongBits(log ? 0.0 : -0.0),
                            Double.doubleToRawLongBits(read(type, gradient)));
                    assertEquals(log ? 0.5 : 0.25, read(type, gradient));
                    assertEquals(log ? -0.5 : -0.25, read(type, gradient));
                }
            }
    }

    @Test void finiteExtremeDenseAndIndexLossActualBackwardRetainsZeroTargetAndIgnoreGuards() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (DataType targetType : List.of(type, DataType.INT32, DataType.INT64)) {
                try (var arena = Arena.ofShared(); var engine = Engine.builder()
                        .takeOwnership(CpuBackendIntegration.open()).build()) {
                    double maximum = type == DataType.FLOAT32 ? Float.MAX_VALUE : Double.MAX_VALUE;
                    Tensor logits = tensor(arena, type, Shape.of(2, 3), true,
                            new double[] {-maximum, maximum, maximum, maximum, -maximum, maximum});
                    Tensor target;
                    Tensor objective;
                    if (targetType == type) {
                        target = tensor(arena, type, Shape.of(2, 3), false,
                                new double[] {0, 1, 0, 0, 0, 0});
                        objective = logits.categoricalCrossEntropyWithLogits(target, 1, LossReduction.SUM);
                    } else {
                        MemorySegment values = targetType == DataType.INT32
                                ? MemorySegment.ofArray(new int[] {1, -1})
                                : MemorySegment.ofArray(new long[] {1, -1});
                        target = TensorFactory.create(new TensorDescriptor(targetType, Shape.of(2),
                                Optional.of(LayoutDescriptor.contiguous(Shape.of(2))), false),
                                Optional.empty(), Optional.of(new MemorySegmentStorage(targetType, 2, values)));
                        objective = logits.categoricalCrossEntropyWithLogits(target, 1, LossReduction.SUM,
                                targetType == DataType.INT32 ? ScalarValue.int32(-1) : ScalarValue.int64(-1));
                    }
                    var backward = engine.backward(objective, List.of(logits), 7L * type.byteWidth());
                    // Existing loss sites form lse = maximum + log(sum) before selected subtraction;
                    // at MAX that addition rounds back to MAX, unlike selected log-softmax above.
                    assertEquals(0.0, read(type, backward.objective().bytes()));
                    var gradient = backward.gradients().getFirst().bytes();
                    assertTrue(read(type, gradient) == 0.0);
                    assertEquals(-0.5, read(type, gradient));
                    assertEquals(0.5, read(type, gradient));
                    for (int i = 0; i < 3; i++) assertTrue(read(type, gradient) == 0.0);
                }
            }
    }

    @Test void denseTargetActualAdRetainsLossMaximumAdditionAtFiniteExtremes() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            try (var arena = Arena.ofShared(); var engine = Engine.builder()
                    .takeOwnership(CpuBackendIntegration.open()).build()) {
                double maximum = type == DataType.FLOAT32 ? Float.MAX_VALUE : Double.MAX_VALUE;
                Tensor logits = tensor(arena, type, Shape.of(3), true,
                        new double[] {-maximum, maximum, maximum});
                Tensor target = tensor(arena, type, Shape.of(3), true, new double[] {0, 2, 0});
                Tensor loss = logits.categoricalCrossEntropyWithLogits(target, 0, LossReduction.SUM);
                var backward = engine.backward(loss, List.of(logits, target), 7L * type.byteWidth());
                assertEquals(0.0, read(type, backward.objective().bytes()));
                var logitsGradient = backward.gradients().getFirst().bytes();
                assertEquals(0.0, read(type, logitsGradient));
                assertEquals(-1.0, read(type, logitsGradient));
                assertEquals(1.0, read(type, logitsGradient));
                // The target role differentiates lse-z, including the rounded maximum addition.
                // It must not inherit log-softmax's -log(2) at the two tied maxima.
                var targetGradient = backward.gradients().get(1).bytes();
                for (int i = 0; i < 3; i++) assertTrue(read(type, targetGradient) == 0.0);
            }
        }
    }

    @Test void categoricalMixedInfinityUsesLossShiftForLogitsAndDenseTargetActualAd() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64, DataType.BFLOAT16,
                DataType.FLOAT16, DataType.FLOAT8_E5M2))
            for (LossReduction reduction : LossReduction.values())
                for (DataType targetType : List.of(type, DataType.INT32, DataType.INT64)) {
                    try (var arena = Arena.ofShared(); var engine = Engine.builder()
                            .takeOwnership(CpuBackendIntegration.open()).build()) {
                        Tensor logits = tensor(arena, type, Shape.of(2, 2), true,
                                new double[] {Double.NEGATIVE_INFINITY, 1000, 1000, Double.NEGATIVE_INFINITY});
                        Tensor target = targetType == type
                                ? tensor(arena, type, Shape.of(2, 2), true, new double[] {0, 2, 1, 0})
                                : indexTarget(targetType, 1, 0);
                        Tensor loss = logits.categoricalCrossEntropyWithLogits(target, 1, reduction);
                        Tensor objective = reduction == LossReduction.NONE ? loss.sum() : loss;
                        List<Tensor> selected = targetType == type ? List.of(logits, target) : List.of(logits);
                        var backward = engine.backward(objective, selected,
                                (1L + 4L * selected.size()) * type.byteWidth());
                        assertEquals(0.0, read(type, backward.objective().bytes()));
                        for (var gradient : backward.gradients()) {
                            var bytes = gradient.bytes();
                            for (int i = 0; i < 4; i++) assertTrue(read(type, bytes) == 0.0);
                        }
                    }
                }
    }

    @Test void denseZeroRowsExcludeExceptionalLogitsButRetainFiniteTargetDerivativesAndMass() {
        for (DataType type : floatingTypes())
            for (LossReduction reduction : LossReduction.values()) {
                try (var arena = Arena.ofShared(); var engine = Engine.builder()
                        .takeOwnership(CpuBackendIntegration.open()).build()) {
                    Tensor logits = tensor(arena, type, Shape.of(5, 2), true, new double[] {
                            Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY,
                            Double.NaN, Double.POSITIVE_INFINITY,
                            Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY,
                            0, 2, 0, 0});
                    Tensor target = tensor(arena, type, Shape.of(5, 2), true,
                            new double[] {0, 0, 0, 0, 0, 0, 0, 0, 1, 2});
                    Tensor loss = logits.categoricalCrossEntropyWithLogits(target, 1, reduction);
                    Tensor seed = tensor(arena, type,
                            reduction == LossReduction.NONE ? Shape.of(5) : Shape.scalar(),
                            false, reduction == LossReduction.NONE
                                    ? new double[] {2, 4, 8, 16, 32} : new double[] {2});
                    Tensor objective = loss.mul(seed);
                    if (reduction == LossReduction.NONE) objective = objective.sum();
                    var backward = engine.backward(objective, List.of(logits, target), 21L * type.byteWidth());
                    double scale = reduction == LossReduction.NONE ? 32
                            : reduction == LossReduction.MEAN ? 2.0 / 5 : 2;
                    var logitsGradient = backward.gradients().getFirst().bytes();
                    for (int i = 0; i < 8; i++) assertEquals(0.0, read(type, logitsGradient));
                    assertEquals(rounded(type, .5 * scale), read(type, logitsGradient));
                    assertEquals(rounded(type, -.5 * scale), read(type, logitsGradient));
                    var targetGradient = backward.gradients().get(1).bytes();
                    for (int i = 0; i < 6; i++) assertEquals(0.0, read(type, targetGradient));
                    double finiteScale = reduction == LossReduction.NONE ? 16 : scale;
                    double normalizer = 2 + Math.log1p(Math.exp(-2));
                    assertEquals(rounded(type, normalizer * finiteScale),
                            read(type, targetGradient), tolerance(type, finiteScale));
                    assertEquals(rounded(type, (normalizer - 2) * finiteScale),
                            read(type, targetGradient), tolerance(type, finiteScale));
                    for (int i = 0; i < 2; i++)
                        assertEquals(rounded(type, Math.log(2) * scale),
                                read(type, targetGradient), tolerance(type, scale));
                    assertTrue(Double.isFinite(read(type, backward.objective().bytes())));
                }
            }
    }

    @Test void ignoredIndexRowsNeverEvaluateExceptionalLogitsAndMeanCountsOnlyActiveRows() {
        for (DataType type : floatingTypes())
            for (DataType targetType : List.of(DataType.INT32, DataType.INT64))
                for (LossReduction reduction : LossReduction.values())
                    for (boolean allIgnored : List.of(false, true)) {
                        try (var arena = Arena.ofShared(); var engine = Engine.builder()
                                .takeOwnership(CpuBackendIntegration.open()).build()) {
                            Tensor logits = tensor(arena, type, Shape.of(4, 2), true, new double[] {
                                    Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY,
                                    Double.NaN, Double.POSITIVE_INFINITY,
                                    Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY, 0, 0});
                            long ignore = targetType == DataType.INT32 ? Integer.MIN_VALUE : Long.MIN_VALUE;
                            Tensor target = indexTarget(targetType, ignore, ignore, ignore, allIgnored ? ignore : 0);
                            Tensor loss = logits.categoricalCrossEntropyWithLogits(target, 1, reduction,
                                    targetType == DataType.INT32 ? ScalarValue.int32((int) ignore)
                                            : ScalarValue.int64(ignore));
                            Tensor objective = reduction == LossReduction.NONE ? loss.sum() : loss;
                            var backward = engine.backward(objective, List.of(logits), 9L * type.byteWidth());
                            var gradient = backward.gradients().getFirst().bytes();
                            for (int i = 0; i < 6; i++) assertEquals(0.0, read(type, gradient));
                            assertEquals(allIgnored ? 0 : -.5, read(type, gradient));
                            assertEquals(allIgnored ? 0 : .5, read(type, gradient));
                            double value = read(type, backward.objective().bytes());
                            if (allIgnored && reduction == LossReduction.MEAN) assertTrue(Double.isNaN(value));
                            else if (allIgnored) assertEquals(0.0, value);
                            else assertEquals(rounded(type, Math.log(2)), value, tolerance(type, 1));
                        }
                    }
    }

    @Test void activePositiveInfinityAndNanRowsRemainInvalidRatherThanBeingZeroed() {
        for (DataType type : floatingTypes())
            for (double poison : new double[] {Double.POSITIVE_INFINITY, Double.NaN})
                for (boolean dense : List.of(false, true)) {
                    try (var arena = Arena.ofShared(); var engine = Engine.builder()
                            .takeOwnership(CpuBackendIntegration.open()).build()) {
                        Tensor logits = tensor(arena, type, Shape.of(2), true, new double[] {poison, 0});
                        Tensor target = dense
                                ? tensor(arena, type, Shape.of(2), true, new double[] {1, 0})
                                : TensorFactory.fromFlatArray(new TensorDescriptor(DataType.INT64, Shape.scalar(),
                                        Optional.of(LayoutDescriptor.contiguous(Shape.scalar())), false),
                                        Optional.empty(), new long[] {0});
                        Tensor loss = logits.categoricalCrossEntropyWithLogits(target, 0, LossReduction.SUM);
                        var backward = engine.backward(loss, dense ? List.of(logits, target) : List.of(logits),
                                5L * type.byteWidth());
                        assertTrue(Double.isNaN(read(type, backward.objective().bytes())));
                        var logitsGradient = backward.gradients().getFirst().bytes();
                        assertTrue(Double.isNaN(read(type, logitsGradient)));
                        assertTrue(Double.isNaN(read(type, logitsGradient)));
                        if (dense) {
                            var targetGradient = backward.gradients().get(1).bytes();
                            assertTrue(Double.isNaN(read(type, targetGradient)));
                            assertEquals(0.0, read(type, targetGradient));
                        }
                    }
                }
    }

    private static List<DataType> floatingTypes() {
        return List.of(DataType.FLOAT32, DataType.FLOAT64, DataType.BFLOAT16, DataType.FLOAT16,
                DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2);
    }

    private static Tensor indexTarget(DataType type, long... values) {
        Shape shape = Shape.of(values.length);
        var descriptor = new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        if (type == DataType.INT64) return TensorFactory.fromFlatArray(descriptor, Optional.empty(), values);
        int[] ints = new int[values.length];
        for (int i = 0; i < values.length; i++) ints[i] = Math.toIntExact(values[i]);
        return TensorFactory.fromFlatArray(descriptor, Optional.empty(), ints);
    }

    private static double tolerance(DataType type, double scale) {
        return Math.abs(scale) * switch (type) {
            case FLOAT64 -> 2e-15;
            case FLOAT32 -> 3e-7;
            case FLOAT16 -> .002;
            case BFLOAT16 -> .02;
            case FLOAT8_E4M3 -> .25;
            case FLOAT8_E5M2 -> .5;
            default -> throw new IllegalArgumentException("not floating");
        };
    }

    private static double rounded(DataType type, double value) {
        return switch (type) {
            case FLOAT64 -> value;
            case FLOAT32 -> (float) value;
            case FLOAT16 -> Float16Bits.toFloat(Float16Bits.fromFloat((float) value));
            case BFLOAT16 -> BFloat16Bits.toFloat(BFloat16Bits.fromFloat((float) value));
            case FLOAT8_E4M3 -> Float8E4M3Bits.toFloat(Float8E4M3Bits.fromDouble(value));
            case FLOAT8_E5M2 -> Float8E5M2Bits.toFloat(Float8E5M2Bits.fromDouble(value));
            default -> throw new IllegalArgumentException("not floating");
        };
    }

    private static double read(DataType type, java.nio.ByteBuffer bytes) {
        return switch (type) {
            case FLOAT32 -> bytes.getFloat();
            case FLOAT64 -> bytes.getDouble();
            case BFLOAT16 -> BFloat16Bits.toFloat(bytes.getShort());
            case FLOAT16 -> Float16Bits.toFloat(bytes.getShort());
            case FLOAT8_E4M3 -> Float8E4M3Bits.toFloat(bytes.get());
            case FLOAT8_E5M2 -> Float8E5M2Bits.toFloat(bytes.get());
            default -> throw new IllegalArgumentException("not floating");
        };
    }

    private static double typed(DataType type,double x) { return type == DataType.FLOAT32 ? (float)x : x; }
    private static Tensor tensor(Arena arena,DataType type,Shape shape,boolean gradient,double[] values) {
        var segment = arena.allocate(values.length*(long)type.byteWidth(),type.byteWidth());
        for (int i = 0; i < values.length; i++) {
            switch (type) {
                case FLOAT32 -> segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) values[i]);
                case FLOAT64 -> segment.setAtIndex(ValueLayout.JAVA_DOUBLE, i, values[i]);
                case BFLOAT16 -> segment.setAtIndex(ValueLayout.JAVA_SHORT, i, BFloat16Bits.fromFloat((float) values[i]));
                case FLOAT16 -> segment.setAtIndex(ValueLayout.JAVA_SHORT, i, Float16Bits.fromFloat((float) values[i]));
                case FLOAT8_E4M3 -> segment.setAtIndex(ValueLayout.JAVA_BYTE, i, Float8E4M3Bits.fromDouble(values[i]));
                case FLOAT8_E5M2 -> segment.setAtIndex(ValueLayout.JAVA_BYTE, i, Float8E5M2Bits.fromDouble(values[i]));
                default -> throw new IllegalArgumentException("not floating");
            }
        }
        return TensorFactory.create(new TensorDescriptor(type,shape,Optional.of(LayoutDescriptor.contiguous(shape)),gradient),
                Optional.empty(),Optional.of(new MemorySegmentStorage(type,values.length,segment)));
    }
}

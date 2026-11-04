package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.*;
import java.lang.foreign.*;
import java.math.BigDecimal;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Actual Compiler AD and Engine reuse preserve the exact published saved forward values. */
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

    private static double read(DataType type, java.nio.ByteBuffer bytes) {
        return type == DataType.FLOAT32 ? bytes.getFloat() : bytes.getDouble();
    }

    private static double typed(DataType type,double x) { return type == DataType.FLOAT32 ? (float)x : x; }
    private static Tensor tensor(Arena arena,DataType type,Shape shape,boolean gradient,double[] values) {
        var segment = arena.allocate(values.length*(long)type.byteWidth(),type.byteWidth());
        for (int i = 0; i < values.length; i++) {
            if (type == DataType.FLOAT32) segment.setAtIndex(ValueLayout.JAVA_FLOAT,i,(float)values[i]);
            else segment.setAtIndex(ValueLayout.JAVA_DOUBLE,i,values[i]);
        }
        return TensorFactory.create(new TensorDescriptor(type,shape,Optional.of(LayoutDescriptor.contiguous(shape)),gradient),
                Optional.empty(),Optional.of(new MemorySegmentStorage(type,values.length,segment)));
    }
}

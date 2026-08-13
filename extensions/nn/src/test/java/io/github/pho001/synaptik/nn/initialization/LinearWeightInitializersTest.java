package io.github.pho001.synaptik.nn.initialization;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import java.util.random.RandomGenerator;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

@Execution(ExecutionMode.SAME_THREAD)
class LinearWeightInitializersTest {
    @Test
    void usesExactOutThenInGlorotNormalFormulaAndOneGaussianPerElement() {
        Shape shape = Shape.of(3, 2);
        GaussianSource source = GaussianSource.repeating(6, 1.0d);
        double expectedScale = Math.sqrt(2.0d / ((double) 2L + (double) 3L));

        Tensor result = ParameterInitializers.glorotNormal(
                shape, DataType.FLOAT64, source);

        assertAll(
                () -> assertArrayEquals(
                        new double[] {
                                expectedScale, expectedScale, expectedScale,
                                expectedScale, expectedScale, expectedScale
                        },
                        heapArray(result, double[].class)),
                () -> source.assertCalls(expectedScale),
                () -> assertLeafMetadata(result, shape, DataType.FLOAT64));
    }

    @Test
    void usesExactOutThenInGlorotUniformFormulaAndBfloat16Conversion() {
        Shape shape = Shape.of(2, 3);
        double expectedBound = Math.sqrt(6.0d / ((double) 3L + (double) 2L));
        BoundedSource source = BoundedSource.repeating(6, expectedBound);

        Tensor result = ParameterInitializers.glorotUniform(
                shape, DataType.BFLOAT16, source);
        short expected = BFloat16Bits.fromFloat((float) expectedBound);

        assertAll(
                () -> assertArrayEquals(
                        new short[] {expected, expected, expected, expected, expected, expected},
                        heapArray(result, short[].class)),
                () -> source.assertCalls(-expectedBound, expectedBound),
                () -> assertLeafMetadata(result, shape, DataType.BFLOAT16));
    }

    @Test
    void usesFanInOnlyForBothFixedReluKaimingPolicies() {
        Shape shape = Shape.of(4, 2);
        double normalScale = Math.sqrt(2.0d / (double) 2L);
        double uniformBound = Math.sqrt(6.0d / (double) 2L);
        GaussianSource normalSource = GaussianSource.repeating(8, -0.5d);
        BoundedSource uniformSource = BoundedSource.repeating(8, -uniformBound);

        Tensor normal = ParameterInitializers.kaimingReluNormal(
                shape, DataType.FLOAT32, normalSource);
        Tensor uniform = ParameterInitializers.kaimingReluUniform(
                shape, DataType.FLOAT64, uniformSource);

        float expectedNormal = (float) (-0.5d * normalScale);
        assertAll(
                () -> assertArrayEquals(
                        new float[] {
                                expectedNormal, expectedNormal, expectedNormal, expectedNormal,
                                expectedNormal, expectedNormal, expectedNormal, expectedNormal
                        },
                        heapArray(normal, float[].class)),
                () -> assertArrayEquals(
                        new double[] {
                                -uniformBound, -uniformBound, -uniformBound, -uniformBound,
                                -uniformBound, -uniformBound, -uniformBound, -uniformBound
                        },
                        heapArray(uniform, double[].class)),
                () -> normalSource.assertCalls(normalScale),
                () -> uniformSource.assertCalls(-uniformBound, uniformBound),
                () -> assertLeafMetadata(normal, shape, DataType.FLOAT32),
                () -> assertLeafMetadata(uniform, shape, DataType.FLOAT64));
    }

    @Test
    void validatesNullsThenTypeStaticRankAndPositiveFansBeforeDrawingOrAllocatingId() {
        Shape dynamic = Shape.ofDimensions(
                new DynamicDimension("out"), new StaticDimension(2));
        GaussianSource source = GaussianSource.repeating(1, 0.0d);
        Tensor before = ParameterInitializers.zeros(Shape.scalar(), DataType.FLOAT32);

        NullPointerException shapeNull = assertThrows(
                NullPointerException.class,
                () -> ParameterInitializers.glorotNormal(null, null, null));
        NullPointerException typeNull = assertThrows(
                NullPointerException.class,
                () -> ParameterInitializers.glorotNormal(Shape.of(1, 1), null, null));
        NullPointerException sourceNull = assertThrows(
                NullPointerException.class,
                () -> ParameterInitializers.glorotNormal(
                        Shape.of(1, 1), DataType.FLOAT32, null));
        IllegalArgumentException type = assertThrows(
                IllegalArgumentException.class,
                () -> ParameterInitializers.glorotNormal(dynamic, DataType.INT32, source));
        IllegalArgumentException staticShape = assertThrows(
                IllegalArgumentException.class,
                () -> ParameterInitializers.glorotNormal(dynamic, DataType.FLOAT32, source));
        IllegalArgumentException scalar = assertThrows(
                IllegalArgumentException.class,
                () -> ParameterInitializers.glorotNormal(
                        Shape.scalar(), DataType.FLOAT32, source));
        IllegalArgumentException rankOne = assertThrows(
                IllegalArgumentException.class,
                () -> ParameterInitializers.glorotNormal(
                        Shape.of(2), DataType.FLOAT32, source));
        IllegalArgumentException rankThree = assertThrows(
                IllegalArgumentException.class,
                () -> ParameterInitializers.glorotNormal(
                        Shape.of(1, 2, 3), DataType.FLOAT32, source));
        IllegalArgumentException out = assertThrows(
                IllegalArgumentException.class,
                () -> ParameterInitializers.glorotNormal(
                        Shape.of(0, 0), DataType.FLOAT32, source));
        IllegalArgumentException in = assertThrows(
                IllegalArgumentException.class,
                () -> ParameterInitializers.glorotNormal(
                        Shape.of(2, 0), DataType.FLOAT32, source));
        Tensor after = ParameterInitializers.zeros(Shape.scalar(), DataType.FLOAT32);

        assertAll(
                () -> assertEquals("weightShape", shapeNull.getMessage()),
                () -> assertEquals("dataType", typeNull.getMessage()),
                () -> assertEquals("randomGenerator", sourceNull.getMessage()),
                () -> assertEquals(
                        "fan-based parameter initialization requires floating data type: INT32",
                        type.getMessage()),
                () -> assertEquals(
                        "fan-based parameter initialization requires a fully static shape: "
                                + dynamic,
                        staticShape.getMessage()),
                () -> assertEquals(
                        "fan-based parameter initialization requires rank-two weight shape: "
                                + Shape.scalar(),
                        scalar.getMessage()),
                () -> assertEquals(
                        "fan-based parameter initialization requires rank-two weight shape: "
                                + Shape.of(2),
                        rankOne.getMessage()),
                () -> assertEquals(
                        "fan-based parameter initialization requires rank-two weight shape: "
                                + Shape.of(1, 2, 3),
                        rankThree.getMessage()),
                () -> assertEquals(
                        "fan-based parameter initialization requires positive outFeatures: 0",
                        out.getMessage()),
                () -> assertEquals(
                        "fan-based parameter initialization requires positive inFeatures: 0",
                        in.getMessage()),
                () -> assertEquals(0, source.calls()),
                () -> assertEquals(before.id().value() + 1, after.id().value()));
    }

    @Test
    void allFanEntriesUseTheSameNoDrawPreflightAndDelegateSourceFailures() {
        Shape valid = Shape.of(1, 2);
        Shape invalid = Shape.of(1, 0);
        GaussianSource gaussian = GaussianSource.repeating(2, 0.0d).throwOnCall(1);
        BoundedSource uniform = BoundedSource.repeating(2, 0.0d).throwOnCall(1);
        GaussianSource unusedGaussian = GaussianSource.repeating(1, 0.0d);
        BoundedSource unusedUniform = BoundedSource.repeating(1, 0.0d);

        assertAll(
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> ParameterInitializers.kaimingReluNormal(
                                invalid, DataType.FLOAT64, unusedGaussian)),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> ParameterInitializers.glorotUniform(
                                invalid, DataType.FLOAT64, unusedUniform)),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> ParameterInitializers.kaimingReluUniform(
                                invalid, DataType.FLOAT64, unusedUniform)),
                () -> assertThrows(
                        DeliberateSourceFailure.class,
                        () -> ParameterInitializers.kaimingReluNormal(
                                valid, DataType.FLOAT64, gaussian)),
                () -> assertThrows(
                        DeliberateSourceFailure.class,
                        () -> ParameterInitializers.kaimingReluUniform(
                                valid, DataType.FLOAT64, uniform)),
                () -> assertEquals(0, unusedGaussian.calls()),
                () -> assertEquals(0, unusedUniform.calls()),
                () -> assertEquals(2, gaussian.calls()),
                () -> assertEquals(2, uniform.calls()));
    }

    @Test
    void positiveFanShapeAboveJavaArrayLimitFailsBeforeDrawOrTensorIdAllocation() {
        Shape overLimit = Shape.of(46_341, 46_341);
        GaussianSource gaussian = GaussianSource.repeating(1, 0.0d);
        BoundedSource uniform = BoundedSource.repeating(1, 0.0d);
        Tensor before = ParameterInitializers.zeros(Shape.scalar(), DataType.FLOAT32);

        assertAll(
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> ParameterInitializers.glorotNormal(
                                overLimit, DataType.FLOAT64, gaussian)),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> ParameterInitializers.glorotUniform(
                                overLimit, DataType.FLOAT64, uniform)),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> ParameterInitializers.kaimingReluNormal(
                                overLimit, DataType.FLOAT64, gaussian)),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> ParameterInitializers.kaimingReluUniform(
                                overLimit, DataType.FLOAT64, uniform)));

        Tensor after = ParameterInitializers.zeros(Shape.scalar(), DataType.FLOAT32);
        assertAll(
                () -> assertEquals(0, gaussian.calls()),
                () -> assertEquals(0, uniform.calls()),
                () -> assertEquals(before.id().value() + 1, after.id().value()));
    }

    private static void assertLeafMetadata(Tensor tensor, Shape shape, DataType dataType) {
        assertAll(
                () -> assertTrue(tensor.descriptor().requiresGrad()),
                () -> assertSame(shape, tensor.descriptor().shape()),
                () -> assertSame(dataType, tensor.descriptor().dataType()),
                () -> assertTrue(tensor.label().isEmpty()),
                () -> assertTrue(tensor.provenance().isEmpty()));
    }

    private static <T> T heapArray(Tensor tensor, Class<T> carrier) {
        return carrier.cast(
                tensor.hostStorage().orElseThrow().segment().heapBase().orElseThrow());
    }

    private static final class GaussianSource implements RandomGenerator {
        private final double[] values;
        private int calls;
        private int throwingCall = -1;

        private GaussianSource(double[] values) {
            this.values = values;
        }

        private static GaussianSource repeating(int count, double value) {
            double[] values = new double[count];
            java.util.Arrays.fill(values, value);
            return new GaussianSource(values);
        }

        private GaussianSource throwOnCall(int call) {
            throwingCall = call;
            return this;
        }

        @Override
        public long nextLong() {
            throw new AssertionError("nextLong must not be called");
        }

        @Override
        public double nextGaussian() {
            int call = calls++;
            if (call == throwingCall) {
                throw new DeliberateSourceFailure();
            }
            return values[call];
        }

        private int calls() {
            return calls;
        }

        private void assertCalls(double expectedScale) {
            assertEquals(values.length, calls);
            assertTrue(expectedScale > 0.0d);
        }
    }

    private static final class BoundedSource implements RandomGenerator {
        private final double[] values;
        private final double[] origins;
        private final double[] bounds;
        private int calls;
        private int throwingCall = -1;

        private BoundedSource(double[] values) {
            this.values = values;
            origins = new double[values.length];
            bounds = new double[values.length];
        }

        private static BoundedSource repeating(int count, double value) {
            double[] values = new double[count];
            java.util.Arrays.fill(values, value);
            return new BoundedSource(values);
        }

        private BoundedSource throwOnCall(int call) {
            throwingCall = call;
            return this;
        }

        @Override
        public long nextLong() {
            throw new AssertionError("nextLong must not be called");
        }

        @Override
        public double nextDouble() {
            throw new AssertionError("unbounded nextDouble must not be called");
        }

        @Override
        public double nextDouble(double origin, double bound) {
            int call = calls++;
            origins[call] = origin;
            bounds[call] = bound;
            if (call == throwingCall) {
                throw new DeliberateSourceFailure();
            }
            return values[call];
        }

        private int calls() {
            return calls;
        }

        private void assertCalls(double expectedOrigin, double expectedBound) {
            assertEquals(values.length, calls);
            for (int index = 0; index < calls; index++) {
                assertEquals(expectedOrigin, origins[index]);
                assertEquals(expectedBound, bounds[index]);
            }
        }
    }

    private static final class DeliberateSourceFailure extends RuntimeException {
    }
}

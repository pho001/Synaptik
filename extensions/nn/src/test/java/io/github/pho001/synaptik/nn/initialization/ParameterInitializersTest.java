package io.github.pho001.synaptik.nn.initialization;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.Set;
import java.util.random.RandomGenerator;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

@Execution(ExecutionMode.SAME_THREAD)
class ParameterInitializersTest {
    @Test
    void exposesExactlyEightStaticEntriesWithoutStateOrObjectConstruction()
            throws ReflectiveOperationException {
        Set<Method> expected = Set.of(
                ParameterInitializers.class.getDeclaredMethod(
                        "zeros", Shape.class, DataType.class),
                ParameterInitializers.class.getDeclaredMethod(
                        "ones", Shape.class, DataType.class),
                ParameterInitializers.class.getDeclaredMethod(
                        "normal",
                        Shape.class,
                        DataType.class,
                        double.class,
                        double.class,
                        RandomGenerator.class),
                ParameterInitializers.class.getDeclaredMethod(
                        "uniform",
                        Shape.class,
                        DataType.class,
                        double.class,
                        double.class,
                        RandomGenerator.class),
                ParameterInitializers.class.getDeclaredMethod(
                        "glorotNormal", Shape.class, DataType.class, RandomGenerator.class),
                ParameterInitializers.class.getDeclaredMethod(
                        "glorotUniform", Shape.class, DataType.class, RandomGenerator.class),
                ParameterInitializers.class.getDeclaredMethod(
                        "kaimingReluNormal", Shape.class, DataType.class, RandomGenerator.class),
                ParameterInitializers.class.getDeclaredMethod(
                        "kaimingReluUniform", Shape.class, DataType.class, RandomGenerator.class));
        Set<Method> actual = Arrays.stream(ParameterInitializers.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .collect(Collectors.toUnmodifiableSet());

        assertAll(
                () -> assertTrue(Modifier.isPublic(ParameterInitializers.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(ParameterInitializers.class.getModifiers())),
                () -> assertEquals(0, ParameterInitializers.class.getDeclaredFields().length),
                () -> assertEquals(0, ParameterInitializers.class.getDeclaredClasses().length),
                () -> assertEquals(1, ParameterInitializers.class.getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isPrivate(
                        ParameterInitializers.class.getDeclaredConstructors()[0].getModifiers())),
                () -> assertEquals(
                        0,
                        ParameterInitializers.class.getDeclaredConstructors()[0]
                                .getParameterCount()),
                () -> assertEquals(expected, actual),
                () -> assertTrue(actual.stream().allMatch(method ->
                        Modifier.isStatic(method.getModifiers())
                                && method.getReturnType() == Tensor.class)));
    }

    @Test
    void createsIndependentTypedZerosAndOnesWithParameterLeafMetadata() {
        Shape shape = Shape.of(2, 2);
        for (DataType dataType : new DataType[] {
                DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16
        }) {
            Tensor zeros = ParameterInitializers.zeros(shape, dataType);
            Tensor otherZeros = ParameterInitializers.zeros(shape, dataType);
            Tensor ones = ParameterInitializers.ones(shape, dataType);

            assertAll(
                    () -> assertParameterLeaf(zeros, shape, dataType),
                    () -> assertParameterLeaf(otherZeros, shape, dataType),
                    () -> assertParameterLeaf(ones, shape, dataType),
                    () -> assertNotEquals(zeros.id(), otherZeros.id()),
                    () -> assertNotSame(
                            zeros.hostStorage().orElseThrow(),
                            otherZeros.hostStorage().orElseThrow()),
                    () -> assertNotSame(heapBase(zeros), heapBase(otherZeros)));
            assertConstantValues(zeros, dataType, false);
            assertConstantValues(ones, dataType, true);
        }
    }

    @Test
    void delegatesNormalAndUniformSamplingWithExactArgumentsAndConversions() {
        Shape shape = Shape.of(2);
        GaussianSource gaussian = new GaussianSource(-1.25d, 0.5d);
        BoundedSource uniform = new BoundedSource(-0.75d, 0.25d);
        double mean = 0x1.0000000000001p4;
        double deviation = 0x1.0000000000001p-4;
        double lower = -2.25d;
        double upper = 3.75d;

        Tensor normal = ParameterInitializers.normal(
                shape, DataType.FLOAT32, mean, deviation, gaussian);
        Tensor bounded = ParameterInitializers.uniform(
                shape, DataType.BFLOAT16, lower, upper, uniform);

        assertAll(
                () -> assertArrayEquals(
                        new float[] {
                                (float) (mean + -1.25d * deviation),
                                (float) (mean + 0.5d * deviation)
                        },
                        heapArray(normal, float[].class)),
                () -> assertArrayEquals(
                        new short[] {
                                BFloat16Bits.fromFloat(-0.75f),
                                BFloat16Bits.fromFloat(0.25f)
                        },
                        heapArray(bounded, short[].class)),
                () -> assertEquals(2, gaussian.calls()),
                () -> uniform.assertCalls(lower, upper),
                () -> assertParameterLeaf(normal, shape, DataType.FLOAT32),
                () -> assertParameterLeaf(bounded, shape, DataType.BFLOAT16));
    }

    @Test
    void genericValidationAndSourceFailurePreserveModelSideEffects() {
        Shape dynamic = Shape.ofDimensions(new DynamicDimension("features"));
        GaussianSource unusedGaussian = new GaussianSource(0.0d);
        BoundedSource unusedUniform = new BoundedSource(0.0d);

        NullPointerException shapeNull = assertThrows(
                NullPointerException.class,
                () -> ParameterInitializers.normal(null, null, 0.0d, 1.0d, null));
        NullPointerException typeNull = assertThrows(
                NullPointerException.class,
                () -> ParameterInitializers.normal(
                        Shape.scalar(), null, 0.0d, 1.0d, null));
        NullPointerException sourceNull = assertThrows(
                NullPointerException.class,
                () -> ParameterInitializers.normal(
                        Shape.scalar(), DataType.FLOAT32, 0.0d, 1.0d, null));

        assertAll(
                () -> assertEquals("shape", shapeNull.getMessage()),
                () -> assertEquals("dataType", typeNull.getMessage()),
                () -> assertEquals("randomGenerator", sourceNull.getMessage()),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> ParameterInitializers.zeros(dynamic, DataType.FLOAT32)),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> ParameterInitializers.ones(Shape.scalar(), DataType.INT32)),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> ParameterInitializers.normal(
                                dynamic,
                                DataType.FLOAT64,
                                0.0d,
                                1.0d,
                                unusedGaussian)),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> ParameterInitializers.uniform(
                                Shape.scalar(),
                                DataType.INT64,
                                -1.0d,
                                1.0d,
                                unusedUniform)),
                () -> assertEquals(0, unusedGaussian.calls()),
                () -> assertEquals(0, unusedUniform.calls()));

        Tensor before = ParameterInitializers.zeros(Shape.scalar(), DataType.FLOAT32);
        GaussianSource throwing = new GaussianSource(0.25d, 0.5d).throwOnCall(1);
        assertThrows(
                DeliberateSourceFailure.class,
                () -> ParameterInitializers.normal(
                        Shape.of(2), DataType.FLOAT64, 0.0d, 1.0d, throwing));
        Tensor after = ParameterInitializers.zeros(Shape.scalar(), DataType.FLOAT32);

        assertAll(
                () -> assertEquals(2, throwing.calls()),
                () -> assertEquals(before.id().value() + 1, after.id().value()));
    }

    private static void assertParameterLeaf(Tensor tensor, Shape shape, DataType dataType) {
        assertAll(
                () -> assertSame(shape, tensor.descriptor().shape()),
                () -> assertSame(dataType, tensor.descriptor().dataType()),
                () -> assertEquals(
                        LayoutKind.DENSE_CONTIGUOUS,
                        tensor.descriptor().layout().orElseThrow().kind()),
                () -> assertFalse(tensor.descriptor().layout().orElseThrow().isView()),
                () -> assertTrue(tensor.descriptor().requiresGrad()),
                () -> assertTrue(tensor.label().isEmpty()),
                () -> assertTrue(tensor.provenance().isEmpty()),
                () -> assertEquals(
                        shape.knownElementCount().orElseThrow(),
                        tensor.hostStorage().orElseThrow().elementCapacity()));
    }

    private static void assertConstantValues(Tensor tensor, DataType dataType, boolean one) {
        switch (dataType) {
            case FLOAT64 -> assertArrayEquals(
                    new double[] {one ? 1.0d : 0.0d, one ? 1.0d : 0.0d,
                            one ? 1.0d : 0.0d, one ? 1.0d : 0.0d},
                    heapArray(tensor, double[].class));
            case FLOAT32 -> assertArrayEquals(
                    new float[] {one ? 1.0f : 0.0f, one ? 1.0f : 0.0f,
                            one ? 1.0f : 0.0f, one ? 1.0f : 0.0f},
                    heapArray(tensor, float[].class));
            case BFLOAT16 -> {
                short value = BFloat16Bits.fromFloat(one ? 1.0f : 0.0f);
                assertArrayEquals(
                        new short[] {value, value, value, value},
                        heapArray(tensor, short[].class));
            }
            case INT32, INT64, BOOL -> throw new AssertionError("unexpected non-floating type");
        }
    }

    private static Object heapBase(Tensor tensor) {
        return tensor.hostStorage().orElseThrow().segment().heapBase().orElseThrow();
    }

    private static <T> T heapArray(Tensor tensor, Class<T> carrier) {
        return carrier.cast(heapBase(tensor));
    }

    private static final class GaussianSource implements RandomGenerator {
        private final double[] values;
        private int calls;
        private int throwingCall = -1;

        private GaussianSource(double... values) {
            this.values = values.clone();
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
    }

    private static final class BoundedSource implements RandomGenerator {
        private final double[] values;
        private final double[] origins;
        private final double[] bounds;
        private int calls;

        private BoundedSource(double... values) {
            this.values = values.clone();
            origins = new double[values.length];
            bounds = new double[values.length];
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
            origins[calls] = origin;
            bounds[calls] = bound;
            return values[calls++];
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

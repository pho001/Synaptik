package io.github.pho001.synaptik.nn.initialization;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

@Execution(ExecutionMode.SAME_THREAD)
class ParameterInitializationTest {
    @Test
    void exposesOnlyTheClosedImmutableValueSurface() {
        Set<String> publicMethods = Arrays.stream(ParameterInitialization.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .map(Method::getName)
                .collect(Collectors.toSet());

        assertAll(
                () -> assertTrue(Modifier.isPublic(ParameterInitialization.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(ParameterInitialization.class.getModifiers())),
                () -> assertEquals(
                        Set.of("glorotNormal", "glorotUniform", "kaimingReluNormal",
                                "kaimingReluUniform", "normal", "uniform", "zeros", "ones",
                                "requiresRandomGenerator", "equals", "hashCode", "toString"),
                        publicMethods),
                () -> assertTrue(Arrays.stream(ParameterInitialization.class.getDeclaredFields())
                        .noneMatch(field -> Modifier.isPublic(field.getModifiers())
                                || Modifier.isProtected(field.getModifiers()))),
                () -> assertTrue(Arrays.stream(ParameterInitialization.class.getDeclaredClasses())
                        .noneMatch(type -> Modifier.isPublic(type.getModifiers())
                                || Modifier.isProtected(type.getModifiers()))),
                () -> assertTrue(Arrays.stream(ParameterInitialization.class.getDeclaredConstructors())
                        .map(Constructor::getModifiers)
                        .noneMatch(modifiers -> Modifier.isPublic(modifiers)
                                || Modifier.isProtected(modifiers))),
                () -> assertTrue(Arrays.stream(ParameterInitialization.class.getDeclaredFields())
                        .allMatch(field -> Modifier.isFinal(field.getModifiers()))),
                () -> assertTrue(Arrays.stream(ParameterInitialization.class.getDeclaredFields())
                        .noneMatch(field -> field.getType() == Shape.class
                                || field.getType() == DataType.class
                                || field.getType() == Tensor.class)));
    }

    @Test
    void reportsRandomRequirementsForExactlyTheSixSamplingPolicies() {
        List<ParameterInitialization> random = List.of(
                ParameterInitialization.glorotNormal(),
                ParameterInitialization.glorotUniform(),
                ParameterInitialization.kaimingReluNormal(),
                ParameterInitialization.kaimingReluUniform(),
                ParameterInitialization.normal(0.0d, 1.0d),
                ParameterInitialization.uniform(-1.0d, 1.0d));

        assertAll(
                () -> assertTrue(random.stream().allMatch(
                        ParameterInitialization::requiresRandomGenerator)),
                () -> assertFalse(ParameterInitialization.zeros().requiresRandomGenerator()),
                () -> assertFalse(ParameterInitialization.ones().requiresRandomGenerator()));
    }

    @Test
    void hasStructuralSignedZeroSensitiveValueAndDiagnosticSemantics() {
        ParameterInitialization first = ParameterInitialization.normal(-0.0d, 0.25d);
        ParameterInitialization equal = ParameterInitialization.normal(-0.0d, 0.25d);
        ParameterInitialization differentZero = ParameterInitialization.normal(0.0d, 0.25d);

        assertAll(
                () -> assertNotSame(first, equal),
                () -> assertEquals(first, equal),
                () -> assertEquals(first.hashCode(), equal.hashCode()),
                () -> assertNotEquals(first, differentZero),
                () -> assertNotEquals(ParameterInitialization.zeros(),
                        ParameterInitialization.ones()),
                () -> assertEquals(
                        "ParameterInitialization.normal(mean=-0.0, standardDeviation=0.25)",
                        first.toString()),
                () -> assertEquals(
                        "ParameterInitialization.uniform(lowerBoundInclusive=-2.0, "
                                + "upperBoundExclusive=3.0)",
                        ParameterInitialization.uniform(-2.0d, 3.0d).toString()),
                () -> assertEquals("ParameterInitialization.glorotUniform()",
                        ParameterInitialization.glorotUniform().toString()));
    }

    @Test
    void validatesConfiguredArgumentsBeforeAnyTensorIdentifierEffect() throws Exception {
        AtomicLong ids = nextTensorIdState();
        long before = ids.get();

        assertAll(
                () -> assertThrows(IllegalArgumentException.class,
                        () -> ParameterInitialization.normal(Double.NaN, Double.NaN)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> ParameterInitialization.normal(0.0d, Double.POSITIVE_INFINITY)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> ParameterInitialization.normal(0.0d, -0.25d)),
                () -> assertEquals(ParameterInitialization.normal(0.0d, -0.0d),
                        ParameterInitialization.normal(0.0d, -0.0d)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> ParameterInitialization.uniform(Double.NEGATIVE_INFINITY, 1.0d)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> ParameterInitialization.uniform(0.0d, Double.NaN)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> ParameterInitialization.uniform(1.0d, 1.0d)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> ParameterInitialization.uniform(2.0d, 1.0d)),
                () -> assertEquals(before, ids.get()));
    }

    private static AtomicLong nextTensorIdState() throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }
}

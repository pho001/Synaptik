package io.github.pho001.synaptik.model.operation.reduction;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class ReductionSemanticsTest {
    private static final List<AggregateReductionKind> ORDINARY_KINDS = List.of(
            AggregateReductionKind.SUM,
            AggregateReductionKind.MEAN,
            AggregateReductionKind.PROD,
            AggregateReductionKind.MIN,
            AggregateReductionKind.MAX,
            AggregateReductionKind.ALL,
            AggregateReductionKind.ANY);

    @Test
    void declaresExactlyTheRequiredAggregateVocabularyInOrder() {
        AggregateReductionKind[] values = AggregateReductionKind.values();

        assertAll(
                () -> assertArrayEquals(
                        new AggregateReductionKind[] {
                            AggregateReductionKind.SUM,
                            AggregateReductionKind.MEAN,
                            AggregateReductionKind.PROD,
                            AggregateReductionKind.MIN,
                            AggregateReductionKind.MAX,
                            AggregateReductionKind.ALL,
                            AggregateReductionKind.ANY,
                            AggregateReductionKind.ARG_MAX
                        },
                        values),
                () -> assertEquals(
                        List.of("SUM", "MEAN", "PROD", "MIN", "MAX", "ALL", "ANY", "ARG_MAX"),
                        Arrays.stream(values).map(AggregateReductionKind::name).toList()),
                () -> assertInstanceOf(OperationKind.class, AggregateReductionKind.SUM),
                () -> assertSame(
                        AggregateReductionKind.ARG_MAX,
                        AggregateReductionKind.valueOf("ARG_MAX")));
    }

    @Test
    void declaresExactlyTheRequiredTiePolicyVocabularyInOrder() {
        ArgMaxTiePolicy[] values = ArgMaxTiePolicy.values();

        assertAll(
                () -> assertArrayEquals(
                        new ArgMaxTiePolicy[] {
                            ArgMaxTiePolicy.FIRST_INDEX, ArgMaxTiePolicy.LAST_INDEX
                        },
                        values),
                () -> assertEquals(
                        List.of("FIRST_INDEX", "LAST_INDEX"),
                        Arrays.stream(values).map(ArgMaxTiePolicy::name).toList()),
                () -> assertSame(
                        ArgMaxTiePolicy.FIRST_INDEX,
                        ArgMaxTiePolicy.valueOf("FIRST_INDEX")));
    }

    @Test
    void exposesOnlyTheExactEnumShapes() {
        assertEnumShape(AggregateReductionKind.class, List.of(OperationKind.class));
        assertEnumShape(ArgMaxTiePolicy.class, List.of());
    }

    @Test
    void exposesOnlyTheExactRecordShapesAndAccessors() {
        assertAll(
                () -> assertRecordShape(
                        AxisReductionAttrs.class,
                        List.of("axis", "keepDimensions"),
                        List.of(int.class, boolean.class),
                        List.of(
                                "axis():int",
                                "equals(java.lang.Object):boolean",
                                "hashCode():int",
                                "keepDimensions():boolean",
                                "toString():java.lang.String")),
                () -> assertRecordShape(
                        ArgMaxAttrs.class,
                        List.of("axis", "keepDimensions", "tiePolicy"),
                        List.of(int.class, boolean.class, ArgMaxTiePolicy.class),
                        List.of(
                                "axis():int",
                                "equals(java.lang.Object):boolean",
                                "hashCode():int",
                                "keepDimensions():boolean",
                                "tiePolicy():io.github.pho001.synaptik.model.operation.reduction.ArgMaxTiePolicy",
                                "toString():java.lang.String")));
    }

    @Test
    void acceptsEveryNonNegativeOrdinaryAxisAndBothDimensionChoices() {
        for (int axis : new int[] {0, 1, 37, Integer.MAX_VALUE}) {
            for (boolean keepDimensions : new boolean[] {false, true}) {
                AxisReductionAttrs attrs = new AxisReductionAttrs(axis, keepDimensions);

                assertAll(
                        () -> assertEquals(axis, attrs.axis()),
                        () -> assertEquals(keepDimensions, attrs.keepDimensions()));
            }
        }
    }

    @Test
    void rejectsEveryRepresentativeNegativeOrdinaryAxisWithTheExactMessage() {
        for (int axis : new int[] {-1, -2, -37, Integer.MIN_VALUE}) {
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class,
                    () -> new AxisReductionAttrs(axis, false));

            assertEquals("axis must be non-negative: " + axis, failure.getMessage());
        }
    }

    @Test
    void validatesAndRetainsCompleteArgMaxAttributesInComponentOrder() {
        for (int axis : new int[] {0, 1, 37, Integer.MAX_VALUE}) {
            for (boolean keepDimensions : new boolean[] {false, true}) {
                for (ArgMaxTiePolicy tiePolicy : ArgMaxTiePolicy.values()) {
                    ArgMaxAttrs attrs = new ArgMaxAttrs(axis, keepDimensions, tiePolicy);

                    assertAll(
                            () -> assertEquals(axis, attrs.axis()),
                            () -> assertEquals(keepDimensions, attrs.keepDimensions()),
                            () -> assertSame(tiePolicy, attrs.tiePolicy()));
                }
            }
        }

        for (int axis : new int[] {-1, -37, Integer.MIN_VALUE}) {
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class,
                    () -> new ArgMaxAttrs(axis, true, null));
            assertEquals("axis must be non-negative: " + axis, failure.getMessage());
        }

        NullPointerException nullPolicy = assertThrows(
                NullPointerException.class,
                () -> new ArgMaxAttrs(0, false, null));
        assertEquals("tiePolicy", nullPolicy.getMessage());
    }

    @Test
    void usesGeneratedRecordValueSemanticsAndDiagnosticText() {
        AxisReductionAttrs axis = new AxisReductionAttrs(2, true);
        AxisReductionAttrs equalAxis = new AxisReductionAttrs(2, true);
        AxisReductionAttrs differentAxis = new AxisReductionAttrs(2, false);
        ArgMaxAttrs argMax = new ArgMaxAttrs(1, false, ArgMaxTiePolicy.FIRST_INDEX);
        ArgMaxAttrs equalArgMax = new ArgMaxAttrs(1, false, ArgMaxTiePolicy.FIRST_INDEX);
        ArgMaxAttrs differentArgMax = new ArgMaxAttrs(1, false, ArgMaxTiePolicy.LAST_INDEX);

        assertAll(
                () -> assertEquals(axis, equalAxis),
                () -> assertEquals(axis.hashCode(), equalAxis.hashCode()),
                () -> assertNotEquals(axis, differentAxis),
                () -> assertEquals(
                        "AxisReductionAttrs[axis=2, keepDimensions=true]", axis.toString()),
                () -> assertEquals(argMax, equalArgMax),
                () -> assertEquals(argMax.hashCode(), equalArgMax.hashCode()),
                () -> assertNotEquals(argMax, differentArgMax),
                () -> assertEquals(
                        "ArgMaxAttrs[axis=1, keepDimensions=false, tiePolicy=FIRST_INDEX]",
                        argMax.toString()));
    }

    @Test
    void composesEveryOrdinaryKindWithAxisAndFullAttributesByExactReference() {
        for (AggregateReductionKind kind : ORDINARY_KINDS) {
            AxisReductionAttrs axisAttrs = new AxisReductionAttrs(3, true);
            Operation axisOperation = new Operation(kind, axisAttrs);
            Operation fullOperation = new Operation(kind, NoOperationAttrs.INSTANCE);

            assertAll(
                    () -> assertSame(kind, axisOperation.kind()),
                    () -> assertSame(axisAttrs, axisOperation.attrs()),
                    () -> assertSame(kind, fullOperation.kind()),
                    () -> assertSame(NoOperationAttrs.INSTANCE, fullOperation.attrs()));
        }
    }

    @Test
    void composesArgMaxWithEveryExplicitTieAndDimensionChoice() {
        for (ArgMaxTiePolicy tiePolicy : ArgMaxTiePolicy.values()) {
            for (boolean keepDimensions : new boolean[] {false, true}) {
                ArgMaxAttrs attrs = new ArgMaxAttrs(1, keepDimensions, tiePolicy);
                Operation operation = new Operation(AggregateReductionKind.ARG_MAX, attrs);

                assertAll(
                        () -> assertSame(AggregateReductionKind.ARG_MAX, operation.kind()),
                        () -> assertSame(attrs, operation.attrs()));
            }
        }
    }

    @Test
    void keepsEqualDiagnosticNamesDistinctAcrossTypedKindFamilies() {
        OperationKind reductionMin = AggregateReductionKind.MIN;
        OperationKind reductionMax = AggregateReductionKind.MAX;
        OperationKind localMin = OtherKind.MIN;

        assertAll(
                () -> assertEquals(reductionMin.name(), BinaryArithmeticKind.MIN.name()),
                () -> assertEquals(reductionMax.name(), BinaryArithmeticKind.MAX.name()),
                () -> assertNotEquals(reductionMin, BinaryArithmeticKind.MIN),
                () -> assertNotEquals(reductionMax, BinaryArithmeticKind.MAX),
                () -> assertEquals(reductionMin.name(), localMin.name()),
                () -> assertNotEquals(reductionMin, localMin),
                () -> assertNotEquals(
                        new Operation(reductionMin, new AxisReductionAttrs(0, false)),
                        new Operation(localMin, new AxisReductionAttrs(0, false))));
    }

    private static void assertEnumShape(Class<?> type, List<Class<?>> interfaces) {
        var constructors = type.getDeclaredConstructors();
        var fields = type.getDeclaredFields();
        var methods = type.getDeclaredMethods();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.model.operation.reduction",
                        type.getPackageName()),
                () -> assertTrue(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isFinal(type.getModifiers())),
                () -> assertTrue(type.isEnum()),
                () -> assertEquals(interfaces, Arrays.asList(type.getInterfaces())),
                () -> assertEquals(1, constructors.length),
                () -> assertEquals(
                        List.of(String.class, int.class),
                        Arrays.asList(constructors[0].getParameterTypes())),
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertTrue(Arrays.stream(fields)
                        .filter(field -> !field.isEnumConstant())
                        .allMatch(field -> field.isSynthetic()
                                && Modifier.isStatic(field.getModifiers()))),
                () -> assertTrue(Arrays.stream(fields)
                        .filter(field -> !Modifier.isStatic(field.getModifiers()))
                        .findAny()
                        .isEmpty()),
                () -> assertEquals(
                        List.of("valueOf(java.lang.String):" + type.getName(),
                                        "values():[L" + type.getName() + ";")
                                .stream()
                                .sorted()
                                .toList(),
                        Arrays.stream(methods)
                                .filter(method -> !method.isSynthetic())
                                .map(ReductionSemanticsTest::methodSignature)
                                .sorted()
                                .toList()),
                () -> assertTrue(Arrays.stream(methods)
                        .filter(method -> !Modifier.isStatic(method.getModifiers()))
                        .findAny()
                        .isEmpty()),
                () -> assertEquals(0, type.getDeclaredClasses().length));
    }

    private static void assertRecordShape(
            Class<?> type,
            List<String> componentNames,
            List<Class<?>> componentTypes,
            List<String> methodSignatures) {
        var components = type.getRecordComponents();
        var constructors = type.getDeclaredConstructors();
        var fields = type.getDeclaredFields();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.model.operation.reduction",
                        type.getPackageName()),
                () -> assertTrue(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isFinal(type.getModifiers())),
                () -> assertTrue(type.isRecord()),
                () -> assertEquals(
                        List.of(OperationAttrs.class), Arrays.asList(type.getInterfaces())),
                () -> assertEquals(
                        componentNames,
                        Arrays.stream(components).map(component -> component.getName()).toList()),
                () -> assertEquals(
                        componentTypes,
                        Arrays.stream(components).map(component -> component.getType()).toList()),
                () -> assertEquals(1, constructors.length),
                () -> assertEquals(
                        componentTypes, Arrays.asList(constructors[0].getParameterTypes())),
                () -> assertTrue(Modifier.isPublic(constructors[0].getModifiers())),
                () -> assertEquals(
                        componentNames, Arrays.stream(fields).map(field -> field.getName()).toList()),
                () -> assertTrue(Arrays.stream(fields).allMatch(field ->
                        Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers())
                                && !Modifier.isStatic(field.getModifiers()))),
                () -> assertEquals(
                        methodSignatures,
                        Arrays.stream(type.getDeclaredMethods())
                                .map(ReductionSemanticsTest::methodSignature)
                                .sorted()
                                .toList()),
                () -> assertEquals(0, type.getDeclaredClasses().length));
    }

    private static String methodSignature(java.lang.reflect.Method method) {
        String parameters = Arrays.stream(method.getParameterTypes())
                .map(Class::getName)
                .collect(java.util.stream.Collectors.joining(","));
        return method.getName()
                + "("
                + parameters
                + "):"
                + method.getReturnType().getName();
    }

    private enum OtherKind implements OperationKind {
        MIN
    }
}

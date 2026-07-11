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
import io.github.pho001.synaptik.model.operation.OperationSignature;
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
                            AggregateReductionKind.ARG_MAX,
                            AggregateReductionKind.ARG_MIN,
                            AggregateReductionKind.LOG_SUM_EXP,
                            AggregateReductionKind.VARIANCE,
                            AggregateReductionKind.STANDARD_DEVIATION,
                            AggregateReductionKind.L1_NORM,
                            AggregateReductionKind.L2_NORM
                        },
                        values),
                () -> assertEquals(
                        List.of("SUM", "MEAN", "PROD", "MIN", "MAX", "ALL", "ANY", "ARG_MAX",
                                "ARG_MIN", "LOG_SUM_EXP", "VARIANCE", "STANDARD_DEVIATION",
                                "L1_NORM", "L2_NORM"),
                        Arrays.stream(values).map(AggregateReductionKind::name).toList()),
                () -> assertInstanceOf(OperationKind.class, AggregateReductionKind.SUM),
                () -> assertSame(
                        AggregateReductionKind.ARG_MAX,
                        AggregateReductionKind.valueOf("ARG_MAX")));
    }

    @Test
    void declaresExactlyTheRequiredTiePolicyVocabularyInOrder() {
        ArgExtremaTiePolicy[] values = ArgExtremaTiePolicy.values();

        assertAll(
                () -> assertArrayEquals(
                        new ArgExtremaTiePolicy[] {
                            ArgExtremaTiePolicy.FIRST_INDEX, ArgExtremaTiePolicy.LAST_INDEX
                        },
                        values),
                () -> assertEquals(
                        List.of("FIRST_INDEX", "LAST_INDEX"),
                        Arrays.stream(values).map(ArgExtremaTiePolicy::name).toList()),
                () -> assertSame(
                        ArgExtremaTiePolicy.FIRST_INDEX,
                        ArgExtremaTiePolicy.valueOf("FIRST_INDEX")));
    }

    @Test
    void exposesOnlyTheExactEnumShapes() {
        io.github.pho001.synaptik.model.operation.OperationSignatureTest
                .assertSignatureEnumShape(AggregateReductionKind.class);
        assertEnumShape(ArgExtremaTiePolicy.class, List.of());
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
                        ArgExtremaAttrs.class,
                        List.of("axis", "keepDimensions", "tiePolicy"),
                        List.of(int.class, boolean.class, ArgExtremaTiePolicy.class),
                        List.of(
                                "axis():int",
                                "equals(java.lang.Object):boolean",
                                "hashCode():int",
                                "keepDimensions():boolean",
                                "tiePolicy():io.github.pho001.synaptik.model.operation.reduction.ArgExtremaTiePolicy",
                                "toString():java.lang.String")),
                () -> assertRecordShape(
                        MultiAxisReductionAttrs.class,
                        List.of("axes", "keepDimensions"),
                        List.of(List.class, boolean.class),
                        List.of(
                                "axes():java.util.List",
                                "equals(java.lang.Object):boolean",
                                "hashCode():int",
                                "keepDimensions():boolean",
                                "toString():java.lang.String")),
                () -> assertRecordShape(
                        StatisticalReductionAttrs.class,
                        List.of("axes", "keepDimensions", "correction"),
                        List.of(List.class, boolean.class, long.class),
                        List.of(
                                "axes():java.util.List",
                                "correction():long",
                                "equals(java.lang.Object):boolean",
                                "hashCode():int",
                                "keepDimensions():boolean",
                                "toString():java.lang.String")));
    }

    @Test
    void validatesAndSnapshotsOrderedMultiAxisAttributes() {
        var source = new java.util.ArrayList<>(List.of(2, 0));
        MultiAxisReductionAttrs ordinary = new MultiAxisReductionAttrs(source, true);
        StatisticalReductionAttrs statistical =
                new StatisticalReductionAttrs(source, false, 1);
        source.set(0, 1);

        assertAll(
                () -> assertEquals(List.of(2, 0), ordinary.axes()),
                () -> assertTrue(ordinary.keepDimensions()),
                () -> assertEquals(List.of(2, 0), statistical.axes()),
                () -> assertEquals(1, statistical.correction()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> ordinary.axes().add(3)));

        assertEquals("axes", assertThrows(NullPointerException.class,
                () -> new MultiAxisReductionAttrs(null, false)).getMessage());
        assertEquals("axes[1]", assertThrows(NullPointerException.class,
                () -> new MultiAxisReductionAttrs(Arrays.asList(0, null), false)).getMessage());
        assertEquals("axes[1] must be non-negative: -1", assertThrows(
                IllegalArgumentException.class,
                () -> new MultiAxisReductionAttrs(List.of(0, -1), false)).getMessage());
        assertEquals("axes contains duplicate axis 0 at index 2", assertThrows(
                IllegalArgumentException.class,
                () -> new StatisticalReductionAttrs(List.of(0, 1, 0), false, -1)).getMessage());
        assertEquals("correction must be non-negative: -1", assertThrows(
                IllegalArgumentException.class,
                () -> new StatisticalReductionAttrs(List.of(), false, -1)).getMessage());
    }

    @Test
    void removesEveryArgMaxSpecificSemanticTypeWithoutAliases() {
        for (String oldType : List.of(
                "io.github.pho001.synaptik.model.operation.reduction.ArgMaxAttrs",
                "io.github.pho001.synaptik.model.operation.reduction.ArgMaxTiePolicy",
                "io.github.pho001.synaptik.model.tensor.TensorArgMaxExpressions")) {
            assertThrows(ClassNotFoundException.class, () -> Class.forName(oldType));
        }
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
                for (ArgExtremaTiePolicy tiePolicy : ArgExtremaTiePolicy.values()) {
                    ArgExtremaAttrs attrs = new ArgExtremaAttrs(axis, keepDimensions, tiePolicy);

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
                    () -> new ArgExtremaAttrs(axis, true, null));
            assertEquals("axis must be non-negative: " + axis, failure.getMessage());
        }

        NullPointerException nullPolicy = assertThrows(
                NullPointerException.class,
                () -> new ArgExtremaAttrs(0, false, null));
        assertEquals("tiePolicy", nullPolicy.getMessage());
    }

    @Test
    void usesGeneratedRecordValueSemanticsAndDiagnosticText() {
        AxisReductionAttrs axis = new AxisReductionAttrs(2, true);
        AxisReductionAttrs equalAxis = new AxisReductionAttrs(2, true);
        AxisReductionAttrs differentAxis = new AxisReductionAttrs(2, false);
        ArgExtremaAttrs argMax =
                new ArgExtremaAttrs(1, false, ArgExtremaTiePolicy.FIRST_INDEX);
        ArgExtremaAttrs equalArgMax =
                new ArgExtremaAttrs(1, false, ArgExtremaTiePolicy.FIRST_INDEX);
        ArgExtremaAttrs differentArgMax =
                new ArgExtremaAttrs(1, false, ArgExtremaTiePolicy.LAST_INDEX);

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
                        "ArgExtremaAttrs[axis=1, keepDimensions=false, tiePolicy=FIRST_INDEX]",
                        argMax.toString()));
    }

    @Test
    void composesEveryOrdinaryKindWithAxisAndFullAttributesByExactReference() {
        for (AggregateReductionKind kind : ORDINARY_KINDS) {
            AxisReductionAttrs axisAttrs = new AxisReductionAttrs(3, true);
            Operation axisOperation = new Operation(kind, axisAttrs);
            Operation fullOperation = new Operation(kind, NoOperationAttrs.INSTANCE);
            MultiAxisReductionAttrs multiAttrs =
                    new MultiAxisReductionAttrs(List.of(2, 0), false);
            Operation multiOperation = new Operation(kind, multiAttrs);

            assertAll(
                    () -> assertSame(kind, axisOperation.kind()),
                    () -> assertSame(axisAttrs, axisOperation.attrs()),
                    () -> assertSame(kind, fullOperation.kind()),
                    () -> assertSame(NoOperationAttrs.INSTANCE, fullOperation.attrs()),
                    () -> assertSame(multiAttrs, multiOperation.attrs()));
        }
    }

    @Test
    void composesAdvancedKindsOnlyWithTheirExactAttributeFamilies() {
        MultiAxisReductionAttrs axes = new MultiAxisReductionAttrs(List.of(1), true);
        StatisticalReductionAttrs statistics =
                new StatisticalReductionAttrs(List.of(1), true, 1);
        for (AggregateReductionKind kind : List.of(
                AggregateReductionKind.LOG_SUM_EXP,
                AggregateReductionKind.L1_NORM,
                AggregateReductionKind.L2_NORM)) {
            assertSame(axes, new Operation(kind, axes).attrs());
            assertThrows(IllegalArgumentException.class, () -> new Operation(kind, statistics));
        }
        for (AggregateReductionKind kind : List.of(
                AggregateReductionKind.VARIANCE,
                AggregateReductionKind.STANDARD_DEVIATION)) {
            assertSame(statistics, new Operation(kind, statistics).attrs());
            assertThrows(IllegalArgumentException.class, () -> new Operation(kind, axes));
        }
    }

    @Test
    void composesArgMaxWithEveryExplicitTieAndDimensionChoice() {
        for (ArgExtremaTiePolicy tiePolicy : ArgExtremaTiePolicy.values()) {
            for (boolean keepDimensions : new boolean[] {false, true}) {
                for (AggregateReductionKind kind : List.of(
                        AggregateReductionKind.ARG_MIN, AggregateReductionKind.ARG_MAX)) {
                    ArgExtremaAttrs attrs = new ArgExtremaAttrs(1, keepDimensions, tiePolicy);
                    Operation operation = new Operation(kind, attrs);

                    assertAll(
                            () -> assertSame(kind, operation.kind()),
                            () -> assertSame(attrs, operation.attrs()));
                }
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
        MIN;

        private static final List<OperationSignature> SIGNATURES =
                List.of(OperationSignature.fixed(AxisReductionAttrs.class, 1, 1));

        @Override
        public List<OperationSignature> signatures() {
            return SIGNATURES;
        }
    }
}

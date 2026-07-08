package io.github.pho001.synaptik.model.operation.elementwise.cast;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class CastSemanticsTest {
    @Test
    void declaresExactlyTheCastKindVocabularyWithTypedIdentity() {
        OperationKind cast = CastKind.CAST;
        OperationKind other = OtherKind.CAST;

        assertAll(
                () -> assertArrayEquals(new CastKind[] {CastKind.CAST}, CastKind.values()),
                () -> assertEquals("CAST", cast.name()),
                () -> assertEquals("CAST", cast.toString()),
                () -> assertSame(CastKind.CAST, CastKind.valueOf("CAST")),
                () -> assertInstanceOf(OperationKind.class, cast),
                () -> assertEquals(cast.name(), other.name()),
                () -> assertNotEquals(cast, other),
                () -> assertEquals(CastKind.CAST, CastKind.CAST),
                () -> assertEquals(CastKind.CAST.hashCode(), CastKind.CAST.hashCode()));
    }

    @Test
    void exposesOnlyTheExactEnumShape() {
        io.github.pho001.synaptik.model.operation.OperationSignatureTest
                .assertSignatureEnumShape(CastKind.class);
    }

    @Test
    void exposesOnlyTheExactTargetDataTypeRecordShape() {
        var components = CastAttrs.class.getRecordComponents();
        var constructors = CastAttrs.class.getDeclaredConstructors();
        var fields = CastAttrs.class.getDeclaredFields();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.model.operation.elementwise.cast",
                        CastAttrs.class.getPackageName()),
                () -> assertTrue(Modifier.isPublic(CastAttrs.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(CastAttrs.class.getModifiers())),
                () -> assertTrue(CastAttrs.class.isRecord()),
                () -> assertEquals(
                        List.of(OperationAttrs.class),
                        Arrays.asList(CastAttrs.class.getInterfaces())),
                () -> assertEquals(1, components.length),
                () -> assertEquals("targetDataType", components[0].getName()),
                () -> assertEquals(DataType.class, components[0].getType()),
                () -> assertEquals(1, constructors.length),
                () -> assertEquals(
                        List.of(DataType.class),
                        Arrays.asList(constructors[0].getParameterTypes())),
                () -> assertTrue(Modifier.isPublic(constructors[0].getModifiers())),
                () -> assertEquals(List.of("targetDataType"), Arrays.stream(fields)
                        .map(field -> field.getName())
                        .toList()),
                () -> assertTrue(Arrays.stream(fields).allMatch(field ->
                        Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers())
                                && !Modifier.isStatic(field.getModifiers()))),
                () -> assertEquals(
                        List.of(
                                "equals(java.lang.Object):boolean",
                                "hashCode():int",
                                "targetDataType():io.github.pho001.synaptik.model.datatype.DataType",
                                "toString():java.lang.String"),
                        Arrays.stream(CastAttrs.class.getDeclaredMethods())
                                .map(CastSemanticsTest::methodSignature)
                                .sorted()
                                .toList()),
                () -> assertEquals(0, CastAttrs.class.getDeclaredClasses().length));
    }

    @Test
    void acceptsRetainsAndComposesEveryCurrentTargetDataType() {
        for (DataType targetDataType : DataType.values()) {
            CastAttrs attrs = new CastAttrs(targetDataType);
            Operation operation = new Operation(CastKind.CAST, attrs);

            assertAll(
                    () -> assertSame(targetDataType, attrs.targetDataType()),
                    () -> assertSame(CastKind.CAST, operation.kind()),
                    () -> assertSame(attrs, operation.attrs()),
                    () -> assertNotSame(NoOperationAttrs.INSTANCE, operation.attrs()));
        }
    }

    @Test
    void rejectsNullTargetDataTypeWithTheExactMessage() {
        NullPointerException failure =
                assertThrows(NullPointerException.class, () -> new CastAttrs(null));

        assertEquals("targetDataType", failure.getMessage());
    }

    @Test
    void usesGeneratedRecordValueSemanticsAndDiagnosticText() {
        CastAttrs attrs = new CastAttrs(DataType.FLOAT32);
        CastAttrs equalAttrs = new CastAttrs(DataType.FLOAT32);
        CastAttrs differentAttrs = new CastAttrs(DataType.INT32);

        assertAll(
                () -> assertEquals(attrs, equalAttrs),
                () -> assertEquals(attrs.hashCode(), equalAttrs.hashCode()),
                () -> assertNotEquals(attrs, differentAttrs),
                () -> assertEquals(
                        "CastAttrs[targetDataType=FLOAT32]", attrs.toString()),
                () -> assertNotEquals(
                        new Operation(CastKind.CAST, attrs),
                        new Operation(OtherKind.CAST, equalAttrs)));
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
        CAST;

        private static final List<OperationSignature> SIGNATURES =
                List.of(OperationSignature.fixed(CastAttrs.class, 1, 1));

        @Override
        public List<OperationSignature> signatures() {
            return SIGNATURES;
        }
    }
}

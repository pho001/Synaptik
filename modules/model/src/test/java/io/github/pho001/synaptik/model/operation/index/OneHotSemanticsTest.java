package io.github.pho001.synaptik.model.operation.index;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class OneHotSemanticsTest {
    @Test
    void declaresExactlyTheOneHotKindAndExactImmutableSignature() {
        OperationKind kind = OneHotKind.ONE_HOT;
        List<OperationSignature> signatures = kind.signatures();

        assertAll(
                () -> assertArrayEquals(new OneHotKind[] {OneHotKind.ONE_HOT}, OneHotKind.values()),
                () -> assertEquals("ONE_HOT", kind.name()),
                () -> assertSame(OneHotKind.ONE_HOT, OneHotKind.valueOf("ONE_HOT")),
                () -> assertEquals(
                        List.of(OperationSignature.fixed(OneHotAttrs.class, 1, 1)), signatures),
                () -> assertSame(signatures, kind.signatures()),
                () -> assertThrows(
                        UnsupportedOperationException.class,
                        () -> signatures.add(OperationSignature.fixed(OneHotAttrs.class, 2, 1))));
    }

    @Test
    void exposesOnlyTheExactKindAndAttributesShapes() {
        io.github.pho001.synaptik.model.operation.OperationSignatureTest
                .assertSignatureEnumShape(OneHotKind.class);
        var components = OneHotAttrs.class.getRecordComponents();
        var constructors = OneHotAttrs.class.getDeclaredConstructors();
        var fields = OneHotAttrs.class.getDeclaredFields();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.model.operation.index",
                        OneHotAttrs.class.getPackageName()),
                () -> assertTrue(Modifier.isPublic(OneHotAttrs.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(OneHotAttrs.class.getModifiers())),
                () -> assertTrue(OneHotAttrs.class.isRecord()),
                () -> assertEquals(
                        List.of(OperationAttrs.class),
                        Arrays.asList(OneHotAttrs.class.getInterfaces())),
                () -> assertEquals(
                        List.of("depth"),
                        Arrays.stream(components).map(component -> component.getName()).toList()),
                () -> assertEquals(
                        List.of(long.class),
                        Arrays.stream(components).map(component -> component.getType()).toList()),
                () -> assertEquals(1, constructors.length),
                () -> assertEquals(
                        List.of(long.class), Arrays.asList(constructors[0].getParameterTypes())),
                () -> assertTrue(Modifier.isPublic(constructors[0].getModifiers())),
                () -> assertEquals(List.of("depth"),
                        Arrays.stream(fields).map(field -> field.getName()).toList()),
                () -> assertTrue(Arrays.stream(fields).allMatch(field ->
                        Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers())
                                && !Modifier.isStatic(field.getModifiers()))),
                () -> assertEquals(0, OneHotAttrs.class.getDeclaredClasses().length));
    }

    @Test
    void acceptsEverySelectedPositiveDepthAndRetainsItUnchanged() {
        for (long depth : new long[] {1L, 2L, 37L, Long.MAX_VALUE}) {
            assertEquals(depth, new OneHotAttrs(depth).depth());
        }
    }

    @Test
    void rejectsEverySelectedNonPositiveDepthWithExactMessage() {
        for (long depth : new long[] {0L, -1L, -37L, Long.MIN_VALUE}) {
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class, () -> new OneHotAttrs(depth));
            assertEquals("depth must be positive: " + depth, failure.getMessage());
        }
    }

    @Test
    void usesRecordValueSemanticsAndComposesExactOperation() {
        OneHotAttrs attrs = new OneHotAttrs(3);
        OneHotAttrs equal = new OneHotAttrs(3);
        OneHotAttrs different = new OneHotAttrs(4);
        Operation operation = new Operation(OneHotKind.ONE_HOT, attrs);

        assertAll(
                () -> assertEquals(attrs, equal),
                () -> assertEquals(attrs.hashCode(), equal.hashCode()),
                () -> assertNotEquals(attrs, different),
                () -> assertEquals("OneHotAttrs[depth=3]", attrs.toString()),
                () -> assertSame(OneHotKind.ONE_HOT, operation.kind()),
                () -> assertSame(attrs, operation.attrs()),
                () -> assertEquals(
                        OperationSignature.fixed(OneHotAttrs.class, 1, 1),
                        operation.signature()));
    }

    @Test
    void containsNoCrossLayerOrConfigurableSemanticState() {
        List<String> componentTypes = Arrays.stream(OneHotAttrs.class.getRecordComponents())
                .map(component -> component.getType().getName())
                .toList();
        List<String> kindNames = Arrays.stream(OneHotKind.values()).map(Enum::name).toList();

        assertAll(
                () -> assertEquals(List.of("long"), componentTypes),
                () -> assertEquals(List.of("ONE_HOT"), kindNames),
                () -> assertFalse(componentTypes.stream().anyMatch(name ->
                        name.contains("Tensor") || name.contains("DataType")
                                || name.contains("Shape") || name.contains("backend")
                                || name.contains("runtime") || name.contains("compiler"))));
    }
}

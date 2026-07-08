package io.github.pho001.synaptik.model.operation.layout;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class ShapeTransformSemanticsTest {
    @Test
    void declaresExactlyTheTwoOrderedKindsWithTypedIdentity() {
        OperationKind reshape = ShapeTransformKind.RESHAPE;
        OperationKind expand = ShapeTransformKind.EXPAND;
        OperationKind otherReshape = OtherKind.RESHAPE;

        assertAll(
                () -> assertArrayEquals(
                        new ShapeTransformKind[] {
                            ShapeTransformKind.RESHAPE, ShapeTransformKind.EXPAND
                        },
                        ShapeTransformKind.values()),
                () -> assertEquals("RESHAPE", reshape.name()),
                () -> assertEquals("EXPAND", expand.name()),
                () -> assertEquals("RESHAPE", reshape.toString()),
                () -> assertEquals("EXPAND", expand.toString()),
                () -> assertSame(
                        ShapeTransformKind.RESHAPE,
                        ShapeTransformKind.valueOf("RESHAPE")),
                () -> assertSame(
                        ShapeTransformKind.EXPAND,
                        ShapeTransformKind.valueOf("EXPAND")),
                () -> assertInstanceOf(OperationKind.class, reshape),
                () -> assertNotEquals(reshape, expand),
                () -> assertEquals(reshape.name(), otherReshape.name()),
                () -> assertNotEquals(reshape, otherReshape));
    }

    @Test
    void exposesOnlyTheExactEnumShape() {
        io.github.pho001.synaptik.model.operation.OperationSignatureTest
                .assertSignatureEnumShape(ShapeTransformKind.class);
    }

    @Test
    void exposesOnlyTheExactTargetShapeRecordShape() {
        var components = TargetShapeAttrs.class.getRecordComponents();
        var constructors = TargetShapeAttrs.class.getDeclaredConstructors();
        var fields = TargetShapeAttrs.class.getDeclaredFields();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.model.operation.layout",
                        TargetShapeAttrs.class.getPackageName()),
                () -> assertTrue(Modifier.isPublic(TargetShapeAttrs.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(TargetShapeAttrs.class.getModifiers())),
                () -> assertTrue(TargetShapeAttrs.class.isRecord()),
                () -> assertEquals(
                        List.of(OperationAttrs.class),
                        Arrays.asList(TargetShapeAttrs.class.getInterfaces())),
                () -> assertEquals(1, components.length),
                () -> assertEquals("targetShape", components[0].getName()),
                () -> assertEquals(Shape.class, components[0].getType()),
                () -> assertEquals(1, constructors.length),
                () -> assertEquals(
                        List.of(Shape.class),
                        Arrays.asList(constructors[0].getParameterTypes())),
                () -> assertTrue(Modifier.isPublic(constructors[0].getModifiers())),
                () -> assertEquals(
                        List.of("targetShape"),
                        Arrays.stream(fields).map(field -> field.getName()).toList()),
                () -> assertTrue(Arrays.stream(fields).allMatch(field ->
                        Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers())
                                && !Modifier.isStatic(field.getModifiers()))),
                () -> assertEquals(
                        List.of(
                                "equals(java.lang.Object):boolean",
                                "hashCode():int",
                                "targetShape():io.github.pho001.synaptik.model.shape.Shape",
                                "toString():java.lang.String"),
                        Arrays.stream(TargetShapeAttrs.class.getDeclaredMethods())
                                .map(ShapeTransformSemanticsTest::methodSignature)
                                .sorted()
                                .toList()),
                () -> assertEquals(0, TargetShapeAttrs.class.getDeclaredClasses().length));
    }

    @Test
    void acceptsAndRetainsEveryCurrentShapeCategoryByExactReference() {
        List<Shape> targetShapes = List.of(
                Shape.scalar(),
                Shape.of(2, 3),
                Shape.of(2, 0, 3),
                Shape.ofDimensions(
                        new StaticDimension(2),
                        new DynamicDimension("width")),
                Shape.ofDimensions(
                        new DynamicDimension("batch"),
                        new DynamicDimension("width")));

        for (Shape targetShape : targetShapes) {
            TargetShapeAttrs attrs = new TargetShapeAttrs(targetShape);

            assertSame(targetShape, attrs.targetShape());
        }
    }

    @Test
    void rejectsNullTargetShapeWithTheExactMessage() {
        NullPointerException failure =
                assertThrows(NullPointerException.class, () -> new TargetShapeAttrs(null));

        assertEquals("targetShape", failure.getMessage());
    }

    @Test
    void usesStructuralRecordValueSemanticsAndDiagnosticText() {
        Shape targetShape = Shape.ofDimensions(
                new DynamicDimension("batch"), new StaticDimension(4));
        TargetShapeAttrs attrs = new TargetShapeAttrs(targetShape);
        TargetShapeAttrs equalAttrs = new TargetShapeAttrs(Shape.ofDimensions(
                new DynamicDimension("batch"), new StaticDimension(4)));
        TargetShapeAttrs differentAttrs = new TargetShapeAttrs(Shape.of(2, 4));

        assertAll(
                () -> assertEquals(attrs, equalAttrs),
                () -> assertEquals(attrs.hashCode(), equalAttrs.hashCode()),
                () -> assertNotEquals(attrs, differentAttrs),
                () -> assertEquals(
                        "TargetShapeAttrs[targetShape=Shape[batch, 4]]",
                        attrs.toString()));
    }

    @Test
    void composesEachKindWithTheExactAttributesReference() {
        TargetShapeAttrs attrs = new TargetShapeAttrs(Shape.of(2, 3));
        Operation reshape = new Operation(ShapeTransformKind.RESHAPE, attrs);
        Operation expand = new Operation(ShapeTransformKind.EXPAND, attrs);

        assertAll(
                () -> assertSame(ShapeTransformKind.RESHAPE, reshape.kind()),
                () -> assertSame(attrs, reshape.attrs()),
                () -> assertSame(ShapeTransformKind.EXPAND, expand.kind()),
                () -> assertSame(attrs, expand.attrs()),
                () -> assertNotSame(NoOperationAttrs.INSTANCE, reshape.attrs()),
                () -> assertNotSame(NoOperationAttrs.INSTANCE, expand.attrs()),
                () -> assertNotEquals(reshape, expand),
                () -> assertNotEquals(
                        reshape,
                        new Operation(OtherKind.RESHAPE, attrs)));
    }

    @Test
    void attributesContainOnlyShapeStateWithoutCrossLayerTypes() {
        List<String> componentTypes = Arrays.stream(TargetShapeAttrs.class.getRecordComponents())
                .map(component -> component.getType().getName())
                .toList();

        assertAll(
                () -> assertEquals(
                        List.of("io.github.pho001.synaptik.model.shape.Shape"),
                        componentTypes),
                () -> assertFalse(componentTypes.stream().anyMatch(name ->
                        name.contains("Tensor")
                                || name.contains("layout.Layout")
                                || name.contains("storage")
                                || name.contains("graph")
                                || name.contains("compiler")
                                || name.contains("planning")
                                || name.contains("runtime")
                                || name.contains("backend"))));
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
        RESHAPE;

        private static final List<OperationSignature> SIGNATURES =
                List.of(OperationSignature.fixed(TargetShapeAttrs.class, 1, 1));

        @Override
        public List<OperationSignature> signatures() {
            return SIGNATURES;
        }
    }
}

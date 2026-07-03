package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class TensorDescriptorTest {
    @Test
    void isExactlyTheRequiredFourComponentRecord() {
        assertTrue(TensorDescriptor.class.isRecord());

        var components = TensorDescriptor.class.getRecordComponents();
        assertEquals(4, components.length);
        assertAll(
                () -> assertEquals("dataType", components[0].getName()),
                () -> assertEquals(DataType.class, components[0].getType()),
                () -> assertEquals("shape", components[1].getName()),
                () -> assertEquals(Shape.class, components[1].getType()),
                () -> assertEquals("layout", components[2].getName()),
                () -> assertEquals(Optional.class, components[2].getType()),
                () -> assertEquals("requiresGrad", components[3].getName()),
                () -> assertEquals(boolean.class, components[3].getType()));

        var instanceFields = Arrays.stream(TensorDescriptor.class.getDeclaredFields())
                .filter(field -> !Modifier.isStatic(field.getModifiers()))
                .toList();
        assertEquals(4, instanceFields.size());
        assertEquals(
                Set.of("dataType", "shape", "layout", "requiresGrad"),
                instanceFields.stream()
                        .map(field -> field.getName())
                        .collect(Collectors.toSet()));
        assertEquals(1, TensorDescriptor.class.getDeclaredConstructors().length);
    }

    @Test
    void rejectsEachNullReferenceWithItsComponentName() {
        NullPointerException nullDataType = assertThrows(
                NullPointerException.class,
                () -> new TensorDescriptor(null, Shape.scalar(), Optional.empty(), false));
        NullPointerException nullShape = assertThrows(
                NullPointerException.class,
                () -> new TensorDescriptor(DataType.FLOAT32, null, Optional.empty(), false));
        NullPointerException nullLayout = assertThrows(
                NullPointerException.class,
                () -> new TensorDescriptor(DataType.FLOAT32, Shape.scalar(), null, false));

        assertAll(
                () -> assertEquals("dataType", nullDataType.getMessage()),
                () -> assertEquals("shape", nullShape.getMessage()),
                () -> assertEquals("layout", nullLayout.getMessage()));
    }

    @Test
    void permitsUnresolvedDynamicShapeAndRejectsItsResolvedLayout() {
        Shape dynamic = Shape.ofDimensions(
                new DynamicDimension("batch"), new StaticDimension(3));
        TensorDescriptor unresolved =
                new TensorDescriptor(DataType.FLOAT32, dynamic, Optional.empty(), true);
        LayoutDescriptor resolvedForStaticShape = LayoutDescriptor.contiguous(Shape.of(2, 3));

        assertEquals(dynamic, unresolved.shape());
        assertFalse(unresolved.layout().isPresent());
        assertThrows(
                IllegalArgumentException.class,
                () -> new TensorDescriptor(
                        DataType.FLOAT32,
                        dynamic,
                        Optional.of(resolvedForStaticShape),
                        false));
    }

    @Test
    void permitsUnresolvedOrdinaryScalarAndEmptyStaticShapes() {
        Shape ordinary = Shape.of(2, 3);
        Shape scalar = Shape.scalar();
        Shape empty = Shape.of(2, 0, 4);

        TensorDescriptor ordinaryDescriptor =
                new TensorDescriptor(DataType.FLOAT64, ordinary, Optional.empty(), false);
        TensorDescriptor scalarDescriptor =
                new TensorDescriptor(DataType.FLOAT32, scalar, Optional.empty(), false);
        TensorDescriptor emptyDescriptor =
                new TensorDescriptor(DataType.BFLOAT16, empty, Optional.empty(), false);

        assertAll(
                () -> assertEquals(ordinary, ordinaryDescriptor.shape()),
                () -> assertFalse(ordinaryDescriptor.layout().isPresent()),
                () -> assertEquals(scalar, scalarDescriptor.shape()),
                () -> assertFalse(scalarDescriptor.layout().isPresent()),
                () -> assertEquals(empty, emptyDescriptor.shape()),
                () -> assertFalse(emptyDescriptor.layout().isPresent()));
    }

    @Test
    void acceptsContiguousScalarEmptyOffsetAndBroadcastLayouts() {
        Shape ordinaryShape = Shape.of(2, 3);
        Shape scalarShape = Shape.scalar();
        Shape emptyShape = Shape.of(2, 0, 4);
        LayoutDescriptor ordinary = LayoutDescriptor.contiguous(ordinaryShape);
        LayoutDescriptor scalar = LayoutDescriptor.contiguous(scalarShape);
        LayoutDescriptor empty = LayoutDescriptor.contiguous(emptyShape);
        LayoutDescriptor offset =
                LayoutDescriptor.of(ordinaryShape, new long[] {3, 1}, 5, true);
        LayoutDescriptor broadcast =
                LayoutDescriptor.of(ordinaryShape, new long[] {0, 1}, 0, true);

        assertAll(
                () -> assertEquals(
                        Optional.of(ordinary),
                        new TensorDescriptor(
                                        DataType.FLOAT32,
                                        ordinaryShape,
                                        Optional.of(ordinary),
                                        false)
                                .layout()),
                () -> assertEquals(
                        Optional.of(scalar),
                        new TensorDescriptor(
                                        DataType.FLOAT32,
                                        scalarShape,
                                        Optional.of(scalar),
                                        false)
                                .layout()),
                () -> assertEquals(
                        Optional.of(empty),
                        new TensorDescriptor(
                                        DataType.FLOAT32,
                                        emptyShape,
                                        Optional.of(empty),
                                        false)
                                .layout()),
                () -> assertEquals(
                        Optional.of(offset),
                        new TensorDescriptor(
                                        DataType.FLOAT32,
                                        ordinaryShape,
                                        Optional.of(offset),
                                        false)
                                .layout()),
                () -> assertEquals(
                        Optional.of(broadcast),
                        new TensorDescriptor(
                                        DataType.FLOAT32,
                                        ordinaryShape,
                                        Optional.of(broadcast),
                                        false)
                                .layout()));
    }

    @Test
    void rejectsRankMismatchAndSameRankIncompatibleGeometry() {
        LayoutDescriptor scalarLayout = LayoutDescriptor.contiguous(Shape.scalar());
        LayoutDescriptor layoutForTwoByThree = LayoutDescriptor.contiguous(Shape.of(2, 3));
        Shape incompatibleSameRankShape = Shape.of(3, 2);
        LayoutDescriptor reconstructedForIncompatibleShape = LayoutDescriptor.of(
                incompatibleSameRankShape,
                layoutForTwoByThree.strides(),
                layoutForTwoByThree.storageOffset(),
                layoutForTwoByThree.isView());

        assertThrows(
                IllegalArgumentException.class,
                () -> new TensorDescriptor(
                        DataType.FLOAT32,
                        Shape.of(2, 3),
                        Optional.of(scalarLayout),
                        false));
        assertEquals(layoutForTwoByThree.rank(), incompatibleSameRankShape.rank());
        assertNotEquals(layoutForTwoByThree.kind(), reconstructedForIncompatibleShape.kind());
        assertNotEquals(
                layoutForTwoByThree.referencedElementSpan(),
                reconstructedForIncompatibleShape.referencedElementSpan());
        assertThrows(
                IllegalArgumentException.class,
                () -> new TensorDescriptor(
                        DataType.FLOAT32,
                        incompatibleSameRankShape,
                        Optional.of(layoutForTwoByThree),
                        false));
    }

    @Test
    void acceptsCompatibleLayoutGeometryReusedForAnotherShape() {
        Shape sourceShape = Shape.of(0, 2);
        Shape compatibleShape = Shape.of(0, 3);
        LayoutDescriptor sharedGeometry =
                LayoutDescriptor.of(sourceShape, new long[] {7, 1}, 5, true);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32,
                compatibleShape,
                Optional.of(sharedGeometry),
                false);

        assertEquals(compatibleShape, descriptor.shape());
        assertEquals(Optional.of(sharedGeometry), descriptor.layout());
        assertSame(sharedGeometry, descriptor.layout().orElseThrow());
        assertNotEquals(sourceShape, compatibleShape);
        assertEquals(
                sharedGeometry,
                LayoutDescriptor.of(
                        compatibleShape,
                        sharedGeometry.strides(),
                        sharedGeometry.storageOffset(),
                        sharedGeometry.isView()));
    }

    @Test
    void propagatesArithmeticFailureFromLayoutReconstruction() {
        LayoutDescriptor reconstructingWillOverflow = LayoutDescriptor.of(
                Shape.of(1, 1, 1),
                new long[] {Long.MAX_VALUE, Long.MAX_VALUE, 1},
                0,
                true);

        assertThrows(
                ArithmeticException.class,
                () -> new TensorDescriptor(
                        DataType.FLOAT32,
                        Shape.of(1, Long.MAX_VALUE, 2),
                        Optional.of(reconstructingWillOverflow),
                        false));
    }

    @Test
    void enforcesGradientEligibilityForAllSixDataTypes() {
        DataType[] allDataTypes = {
            DataType.FLOAT64,
            DataType.FLOAT32,
            DataType.BFLOAT16,
            DataType.INT32,
            DataType.INT64,
            DataType.BOOL
        };
        for (DataType dataType : allDataTypes) {
            assertDoesNotThrow(
                    () -> new TensorDescriptor(dataType, Shape.scalar(), Optional.empty(), false));
        }

        DataType[] differentiable = {
            DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16
        };
        for (DataType dataType : differentiable) {
            TensorDescriptor descriptor =
                    new TensorDescriptor(dataType, Shape.scalar(), Optional.empty(), true);
            assertTrue(descriptor.requiresGrad());
        }

        DataType[] nonDifferentiable = {DataType.INT32, DataType.INT64, DataType.BOOL};
        for (DataType dataType : nonDifferentiable) {
            assertThrows(
                    IllegalArgumentException.class,
                    () -> new TensorDescriptor(
                            dataType, Shape.scalar(), Optional.empty(), true));
        }
    }

    @Test
    void equalityAndHashingCoverEveryComponent() {
        TensorDescriptor first = new TensorDescriptor(
                DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), false);
        TensorDescriptor equal = new TensorDescriptor(
                DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), false);
        TensorDescriptor differentDataType = new TensorDescriptor(
                DataType.BFLOAT16, Shape.of(2, 3), Optional.empty(), false);
        TensorDescriptor differentShape = new TensorDescriptor(
                DataType.FLOAT32, Shape.of(3, 2), Optional.empty(), false);
        LayoutDescriptor contiguous = LayoutDescriptor.contiguous(Shape.of(2, 3));
        LayoutDescriptor offset = LayoutDescriptor.of(
                Shape.of(2, 3), new long[] {3, 1}, 1, false);
        TensorDescriptor resolved = new TensorDescriptor(
                DataType.FLOAT32, Shape.of(2, 3), Optional.of(contiguous), false);
        TensorDescriptor differentLayout = new TensorDescriptor(
                DataType.FLOAT32, Shape.of(2, 3), Optional.of(offset), false);
        TensorDescriptor requiresGrad = new TensorDescriptor(
                DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), true);

        assertAll(
                () -> assertEquals(first, equal),
                () -> assertEquals(first.hashCode(), equal.hashCode()),
                () -> assertNotEquals(first, differentDataType),
                () -> assertNotEquals(first, differentShape),
                () -> assertNotEquals(first, resolved),
                () -> assertNotEquals(resolved, differentLayout),
                () -> assertNotEquals(first, requiresGrad));
    }

    @Test
    void usesOptionalValueSemanticsAndRetainsOnlyUnderlyingLayoutIdentity() {
        long[] strides = {3, 1};
        LayoutDescriptor suppliedLayout =
                LayoutDescriptor.of(Shape.of(2, 3), strides, 0, false);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32,
                Shape.of(2, 3),
                Optional.of(suppliedLayout),
                false);
        strides[0] = 99;
        long[] returnedStrides = descriptor.layout().orElseThrow().strides();
        returnedStrides[1] = 99;

        assertTrue(descriptor.layout().isPresent());
        assertEquals(Optional.of(suppliedLayout), descriptor.layout());
        assertSame(suppliedLayout, descriptor.layout().orElseThrow());
        assertArrayEquals(new long[] {3, 1}, descriptor.layout().orElseThrow().strides());
    }

    @Test
    void diagnosticTextNamesCompleteResolvedAndUnresolvedState() {
        TensorDescriptor unresolved = new TensorDescriptor(
                DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), false);
        LayoutDescriptor layout = LayoutDescriptor.contiguous(Shape.of(2, 3));
        TensorDescriptor resolved = new TensorDescriptor(
                DataType.FLOAT32, Shape.of(2, 3), Optional.of(layout), true);

        String unresolvedText = unresolved.toString();
        String resolvedText = resolved.toString();
        assertAll(
                () -> assertTrue(unresolvedText.contains("TensorDescriptor")),
                () -> assertTrue(unresolvedText.contains("dataType=FLOAT32")),
                () -> assertTrue(unresolvedText.contains("shape=Shape[2, 3]")),
                () -> assertTrue(unresolvedText.contains("layout=Optional.empty")),
                () -> assertTrue(unresolvedText.contains("requiresGrad=false")),
                () -> assertTrue(resolvedText.contains("TensorDescriptor")),
                () -> assertTrue(resolvedText.contains("dataType=FLOAT32")),
                () -> assertTrue(resolvedText.contains("shape=Shape[2, 3]")),
                () -> assertTrue(resolvedText.contains("layout=Optional[LayoutDescriptor")),
                () -> assertTrue(resolvedText.contains("requiresGrad=true")),
                () -> assertNotEquals(unresolvedText, resolvedText));
    }
}

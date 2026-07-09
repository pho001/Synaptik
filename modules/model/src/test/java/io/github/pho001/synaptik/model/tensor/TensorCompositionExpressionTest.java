package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.layout.CompositionAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.TensorCompositionKind;
import io.github.pho001.synaptik.model.operation.layout.UnstackOutputAttrs;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

class TensorCompositionExpressionTest {
    @Test
    void helperHasExactlyTheRequiredFieldFreeTenMethodShape() {
        var constructors = TensorCompositionExpressions.class.getDeclaredConstructors();
        var methods = Arrays.stream(TensorCompositionExpressions.class.getDeclaredMethods())
                .map(TensorCompositionExpressionTest::methodSignature)
                .sorted()
                .toList();

        assertAll(
                () -> assertFalse(Modifier.isPublic(
                        TensorCompositionExpressions.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(
                        TensorCompositionExpressions.class.getModifiers())),
                () -> assertEquals(0, TensorCompositionExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0, TensorCompositionExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertEquals(0, constructors[0].getParameterCount()),
                () -> assertEquals(
                        List.of(
                                "concat(int,[Lio.github.pho001.synaptik.model.tensor.Tensor;):io.github.pho001.synaptik.model.tensor.Tensor",
                                "concatShape(java.util.List,io.github.pho001.synaptik.model.shape.Shape,int):io.github.pho001.synaptik.model.shape.Shape",
                                "create(io.github.pho001.synaptik.model.datatype.DataType,io.github.pho001.synaptik.model.shape.Shape,boolean,io.github.pho001.synaptik.model.operation.Operation,java.util.List):io.github.pho001.synaptik.model.tensor.Tensor",
                                "normalizeExistingAxis(java.lang.String,int,int):int",
                                "normalizeInsertionAxis(int,int):int",
                                "snapshotInputs(java.lang.String,[Lio.github.pho001.synaptik.model.tensor.Tensor;):java.util.List",
                                "stack(int,[Lio.github.pho001.synaptik.model.tensor.Tensor;):io.github.pho001.synaptik.model.tensor.Tensor",
                                "stackShape(io.github.pho001.synaptik.model.shape.Shape,int,int):io.github.pho001.synaptik.model.shape.Shape",
                                "unstack(io.github.pho001.synaptik.model.tensor.Tensor,int):java.util.List",
                                "unstackShape(io.github.pho001.synaptik.model.shape.Shape,int):io.github.pho001.synaptik.model.shape.Shape"),
                        methods),
                () -> assertTrue(Arrays.stream(
                                TensorCompositionExpressions.class.getDeclaredMethods())
                        .allMatch(method -> Modifier.isStatic(method.getModifiers()))));
    }

    @Test
    void concatPreservesOrderMetadataAndAllCurrentDataTypes() {
        for (DataType dataType : DataType.values()) {
            StaticDimension rows = new StaticDimension(2);
            Tensor first = tensor(
                    dataType,
                    Shape.ofDimensions(rows, new StaticDimension(1)),
                    Optional.empty(),
                    dataType.isDifferentiable());
            Tensor second = tensor(
                    dataType,
                    Shape.ofDimensions(new StaticDimension(2), new StaticDimension(3)),
                    Optional.empty(),
                    false);
            Tensor third = tensor(
                    dataType,
                    Shape.ofDimensions(new StaticDimension(2), new StaticDimension(0)),
                    Optional.empty(),
                    false);
            Tensor[] callerInputs = {first, second, first, third};

            Tensor result = Tensor.concat(-1, callerInputs);
            callerInputs[0] = third;
            TensorProvenance provenance = result.provenance().orElseThrow();

            assertAll(
                    () -> assertEquals(dataType, result.descriptor().dataType()),
                    () -> assertEquals(Shape.of(2, 5), result.descriptor().shape()),
                    () -> assertSame(
                            rows, result.descriptor().shape().dimensions().get(0)),
                    () -> assertEquals(
                            dataType.isDifferentiable(), result.descriptor().requiresGrad()),
                    () -> assertTrue(result.descriptor().layout().isEmpty()),
                    () -> assertTrue(result.label().isEmpty()),
                    () -> assertTrue(result.hostStorage().isEmpty()),
                    () -> assertSame(TensorCompositionKind.CONCAT,
                            provenance.operation().kind()),
                    () -> assertEquals(
                            new CompositionAxisAttrs(1), provenance.operation().attrs()),
                    () -> assertEquals(List.of(first, second, first, third), provenance.inputs()),
                    () -> assertSame(first, provenance.inputs().get(0)),
                    () -> assertSame(first, provenance.inputs().get(2)),
                    () -> assertNotSame(first, result),
                    () -> assertNotSame(second, result));
        }
    }

    @Test
    void concatSupportsCanonicalStaticZeroAndDynamicSums() {
        DynamicDimension batch = new DynamicDimension("batch");
        DynamicDimension otherBatch = new DynamicDimension("otherBatch");
        StaticDimension columns = new StaticDimension(4);
        Tensor dynamic = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(batch, columns),
                Optional.empty(),
                false);
        Tensor firstZero = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(new StaticDimension(0), columns),
                Optional.empty(),
                false);
        Tensor secondZero = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(new StaticDimension(0), columns),
                Optional.empty(),
                true);

        Tensor result = Tensor.concat(0, firstZero, dynamic, secondZero);
        Tensor dynamicSum = Tensor.concat(
                0,
                dynamic,
                tensor(
                        DataType.FLOAT32,
                        Shape.ofDimensions(otherBatch, columns),
                        Optional.empty(),
                        false));
        Tensor repeated = Tensor.concat(0, dynamic, dynamic);
        Tensor allStaticZero = Tensor.concat(
                0,
                tensor(DataType.INT32, Shape.of(0, 3), Optional.empty(), false),
                tensor(DataType.INT32, Shape.of(0, 3), Optional.empty(), false));

        assertAll(
                () -> assertSame(batch, result.descriptor().shape().dimensions().get(0)),
                () -> assertSame(columns, result.descriptor().shape().dimensions().get(1)),
                () -> assertTrue(result.descriptor().requiresGrad()),
                () -> assertEquals(
                        DimensionExpressions.add(batch, otherBatch),
                        dynamicSum.descriptor().shape().dimensions().get(0)),
                () -> assertEquals(
                        DimensionExpressions.multiply(batch, 2),
                        repeated.descriptor().shape().dimensions().get(0)),
                () -> assertEquals(Shape.of(0, 3), allStaticZero.descriptor().shape()),
                () -> assertNotSame(
                        firstZero.descriptor().shape().dimensions().get(0),
                        allStaticZero.descriptor().shape().dimensions().get(0)));
    }

    @Test
    void concatRejectsInvalidInputsInSpecifiedOrderWithExactMessages() {
        Tensor base = tensor(DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), false);
        Tensor wrongType = tensor(DataType.FLOAT64, Shape.of(2, 3), Optional.empty(), false);
        Tensor wrongRank = tensor(DataType.FLOAT32, Shape.of(2, 3, 1), Optional.empty(), false);
        Tensor wrongNonAxis = tensor(DataType.FLOAT32, Shape.of(4, 3), Optional.empty(), false);
        NullPointerException nullArray = assertThrows(
                NullPointerException.class,
                () -> Tensor.concat(0, (Tensor[]) null));
        IllegalArgumentException empty = assertThrows(
                IllegalArgumentException.class, () -> Tensor.concat(0));
        NullPointerException nullElement = assertThrows(
                NullPointerException.class, () -> Tensor.concat(0, base, null, null));
        IndexOutOfBoundsException axis = assertThrows(
                IndexOutOfBoundsException.class, () -> Tensor.concat(Integer.MIN_VALUE, wrongType));
        IllegalArgumentException type = assertThrows(
                IllegalArgumentException.class, () -> Tensor.concat(1, base, wrongType));
        IllegalArgumentException rank = assertThrows(
                IllegalArgumentException.class, () -> Tensor.concat(1, base, wrongRank));
        IllegalArgumentException dimension = assertThrows(
                IllegalArgumentException.class, () -> Tensor.concat(1, base, wrongNonAxis));
        assertAll(
                () -> assertEquals("inputs", nullArray.getMessage()),
                () -> assertEquals("concat requires at least one input", empty.getMessage()),
                () -> assertEquals("inputs[1]", nullElement.getMessage()),
                () -> assertEquals(
                        "concat axis -2147483648 is outside shape rank 2", axis.getMessage()),
                () -> assertEquals(
                        "concat inputs must have matching data types: inputs[1] is FLOAT64, expected FLOAT32",
                        type.getMessage()),
                () -> assertEquals(
                        "concat inputs must have matching ranks: inputs[1] has 3, expected 2",
                        rank.getMessage()),
                () -> assertEquals(
                        "concat inputs differ at non-concat axis 0: inputs[1]",
                        dimension.getMessage()));
    }

    @Test
    void concatFlattensExistingExpressionsAndRetainsAtomicDynamicTerms() {
        DynamicDimension n = new DynamicDimension("N");
        DynamicDimension m = new DynamicDimension("M");
        StaticDimension columns = new StaticDimension(8);
        Dimension nPlusTwo = DimensionExpressions.addConstant(n, 2);
        Dimension mPlusThree = DimensionExpressions.addConstant(m, 3);
        Dimension divided = DimensionExpressions.ceilingDivide(n, 2);
        Dimension unknown = DimensionExpressions.unknown(1, Optional.of(nPlusTwo));
        Tensor first = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(nPlusTwo, columns),
                Optional.empty(),
                false);
        Tensor second = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(mPlusThree, new StaticDimension(8)),
                Optional.empty(),
                true);
        Tensor divisionInput = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(divided, columns),
                Optional.empty(),
                false);
        Tensor unknownInput = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(unknown, columns),
                Optional.empty(),
                false);
        StaticDimension zero = new StaticDimension(0);
        Tensor zeroInput = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(zero, columns),
                Optional.empty(),
                false);

        Tensor flattened = Tensor.concat(0, first, second);
        Tensor atomic = Tensor.concat(0, divisionInput, unknownInput);
        Tensor neutral = Tensor.concat(0, zeroInput, unknownInput);
        Tensor single = Tensor.concat(0, first);

        assertAll(
                () -> assertEquals(
                        DimensionExpressions.add(
                                DimensionExpressions.add(n, m), new StaticDimension(5)),
                        flattened.descriptor().shape().dimensions().get(0)),
                () -> assertSame(
                        columns, flattened.descriptor().shape().dimensions().get(1)),
                () -> assertTrue(flattened.descriptor().requiresGrad()),
                () -> assertEquals(
                        DimensionExpressions.add(divided, unknown),
                        atomic.descriptor().shape().dimensions().get(0)),
                () -> assertSame(
                        unknown, neutral.descriptor().shape().dimensions().get(0)),
                () -> assertSame(
                        nPlusTwo, single.descriptor().shape().dimensions().get(0)));
    }

    @Test
    void concatOverflowFailsBeforeIdentityAllocation() throws Exception {
        Tensor maximum = tensor(
                DataType.INT64, Shape.of(Long.MAX_VALUE), Optional.empty(), false);
        Tensor one = tensor(DataType.INT64, Shape.of(1), Optional.empty(), false);
        DynamicDimension n = new DynamicDimension("N");
        Tensor maximumOffset = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(DimensionExpressions.addConstant(n, Long.MAX_VALUE)),
                Optional.empty(),
                false);
        Tensor maximumCoefficient = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(DimensionExpressions.multiply(n, Long.MAX_VALUE)),
                Optional.empty(),
                false);
        Tensor dynamicOne = tensor(
                DataType.FLOAT32, Shape.ofDimensions(n), Optional.empty(), false);
        Tensor staticOne = tensor(DataType.FLOAT32, Shape.of(1), Optional.empty(), false);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertThrows(ArithmeticException.class, () -> Tensor.concat(0, maximum, one));
        assertThrows(
                ArithmeticException.class,
                () -> Tensor.concat(0, maximumOffset, staticOne));
        assertThrows(
                ArithmeticException.class,
                () -> Tensor.concat(0, maximumCoefficient, dynamicOne));

        assertEquals(before, next.get());
    }

    @Test
    void stackInsertsCountForScalarStaticAndDynamicShapes() {
        Tensor scalar = tensor(DataType.BOOL, Shape.scalar(), Optional.empty(), false);
        Tensor scalarResult = Tensor.stack(-1, scalar, scalar);
        DynamicDimension batch = new DynamicDimension("batch");
        StaticDimension columns = new StaticDimension(3);
        Shape shape = Shape.ofDimensions(batch, columns);
        Tensor first = tensor(DataType.FLOAT64, shape, Optional.empty(), false);
        Tensor second = tensor(DataType.FLOAT64, shape, Optional.empty(), true);
        Tensor third = tensor(DataType.FLOAT64, shape, Optional.empty(), false);

        Tensor middle = Tensor.stack(1, first, second, third);
        Tensor end = Tensor.stack(-1, first);
        TensorProvenance provenance = middle.provenance().orElseThrow();

        assertAll(
                () -> assertEquals(Shape.of(2), scalarResult.descriptor().shape()),
                () -> assertSame(batch, middle.descriptor().shape().dimensions().get(0)),
                () -> assertEquals(
                        new StaticDimension(3), middle.descriptor().shape().dimensions().get(1)),
                () -> assertSame(columns, middle.descriptor().shape().dimensions().get(2)),
                () -> assertTrue(middle.descriptor().requiresGrad()),
                () -> assertTrue(middle.descriptor().layout().isEmpty()),
                () -> assertEquals(
                        Shape.ofDimensions(batch, columns, new StaticDimension(1)),
                        end.descriptor().shape()),
                () -> assertSame(TensorCompositionKind.STACK, provenance.operation().kind()),
                () -> assertEquals(
                        new CompositionAxisAttrs(1), provenance.operation().attrs()),
                () -> assertEquals(List.of(first, second, third), provenance.inputs()));
    }

    @Test
    void stackAndUnstackRetainEveryDataTypeAndValidEligibilityState() {
        for (DataType dataType : DataType.values()) {
            for (boolean requestedEligibility : new boolean[] {false, true}) {
                if (requestedEligibility && !dataType.isDifferentiable()) {
                    continue;
                }
                Tensor ineligible = tensor(
                        dataType, Shape.of(2), Optional.empty(), false);
                Tensor requested = tensor(
                        dataType, Shape.of(2), Optional.empty(), requestedEligibility);

                Tensor stacked = Tensor.stack(0, ineligible, requested);
                List<Tensor> unstacked = requested.unstack(0);

                assertAll(
                        () -> assertEquals(dataType, stacked.descriptor().dataType()),
                        () -> assertEquals(
                                requestedEligibility, stacked.descriptor().requiresGrad()),
                        () -> assertTrue(stacked.descriptor().layout().isEmpty()),
                        () -> assertEquals(2, unstacked.size()),
                        () -> assertTrue(unstacked.stream().allMatch(output ->
                                output.descriptor().dataType() == dataType
                                        && output.descriptor().requiresGrad()
                                                == requestedEligibility
                                        && output.descriptor().layout().isEmpty())));
            }
        }
    }

    @Test
    void stackRejectsInvalidInputsAndInsertionAxesWithExactMessages() {
        Tensor base = tensor(DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), false);
        Tensor wrongType = tensor(DataType.INT32, Shape.of(2, 3), Optional.empty(), false);
        Tensor wrongShape = tensor(DataType.FLOAT32, Shape.of(2, 1), Optional.empty(), false);

        NullPointerException nullArray = assertThrows(
                NullPointerException.class, () -> Tensor.stack(0, (Tensor[]) null));
        IllegalArgumentException empty = assertThrows(
                IllegalArgumentException.class, () -> Tensor.stack(0));
        NullPointerException nullElement = assertThrows(
                NullPointerException.class, () -> Tensor.stack(0, base, null));
        IndexOutOfBoundsException negative = assertThrows(
                IndexOutOfBoundsException.class, () -> Tensor.stack(-4, wrongType));
        IndexOutOfBoundsException positive = assertThrows(
                IndexOutOfBoundsException.class, () -> Tensor.stack(3, base));
        IllegalArgumentException type = assertThrows(
                IllegalArgumentException.class, () -> Tensor.stack(0, base, wrongType));
        IllegalArgumentException shape = assertThrows(
                IllegalArgumentException.class, () -> Tensor.stack(0, base, wrongShape));

        assertAll(
                () -> assertEquals("inputs", nullArray.getMessage()),
                () -> assertEquals("stack requires at least one input", empty.getMessage()),
                () -> assertEquals("inputs[1]", nullElement.getMessage()),
                () -> assertEquals(
                        "stack axis -4 is outside insertion range for shape rank 2",
                        negative.getMessage()),
                () -> assertEquals(
                        "stack axis 3 is outside insertion range for shape rank 2",
                        positive.getMessage()),
                () -> assertEquals(
                        "stack inputs must have matching data types: inputs[1] is INT32, expected FLOAT32",
                        type.getMessage()),
                () -> assertEquals(
                        "stack inputs must have identical shapes: inputs[1] differs from inputs[0]",
                        shape.getMessage()));
    }

    @Test
    void unstackReturnsImmutableOrderedIndividuallyIndexedFreshOutputs() {
        StaticDimension rows = new StaticDimension(2);
        StaticDimension selected = new StaticDimension(3);
        DynamicDimension columns = new DynamicDimension("columns");
        Tensor input = tensor(
                DataType.BFLOAT16,
                Shape.ofDimensions(rows, selected, columns),
                Optional.empty(),
                true);

        List<Tensor> outputs = input.unstack(-2);

        assertEquals(3, outputs.size());
        for (int index = 0; index < outputs.size(); index++) {
            Tensor output = outputs.get(index);
            TensorProvenance provenance = output.provenance().orElseThrow();
            assertAll(
                    () -> assertEquals(DataType.BFLOAT16, output.descriptor().dataType()),
                    () -> assertSame(rows, output.descriptor().shape().dimensions().get(0)),
                    () -> assertSame(columns, output.descriptor().shape().dimensions().get(1)),
                    () -> assertTrue(output.descriptor().requiresGrad()),
                    () -> assertTrue(output.descriptor().layout().isEmpty()),
                    () -> assertTrue(output.label().isEmpty()),
                    () -> assertTrue(output.hostStorage().isEmpty()),
                    () -> assertSame(TensorCompositionKind.UNSTACK,
                            provenance.operation().kind()),
                    () -> assertEquals(
                            new UnstackOutputAttrs(1, outputs.indexOf(output)),
                            provenance.operation().attrs()),
                    () -> assertEquals(List.of(input), provenance.inputs()),
                    () -> assertSame(input, provenance.inputs().getFirst()),
                    () -> assertNotSame(input, output));
        }
        assertAll(
                () -> assertNotSame(outputs.get(0), outputs.get(1)),
                () -> assertNotEquals(outputs.get(0).id(), outputs.get(1).id()),
                () -> assertSame(
                        outputs.get(0).descriptor().shape(),
                        outputs.get(1).descriptor().shape()),
                () -> assertThrows(
                        UnsupportedOperationException.class,
                        () -> outputs.add(outputs.getFirst())));
    }

    @Test
    void unstackHandlesZeroAndRejectsUnknownOrExcessiveCountsWithoutIds() throws Exception {
        Tensor zero = tensor(DataType.INT32, Shape.of(2, 0, 4), Optional.empty(), false);
        Tensor dynamic = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("items")),
                Optional.empty(),
                false);
        Tensor excessive = tensor(
                DataType.INT64,
                Shape.of((long) Integer.MAX_VALUE + 1),
                Optional.empty(),
                false);
        Tensor scalar = tensor(DataType.BOOL, Shape.scalar(), Optional.empty(), false);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        List<Tensor> empty = zero.unstack(1);
        IllegalArgumentException unknown = assertThrows(
                IllegalArgumentException.class, () -> dynamic.unstack(0));
        IllegalArgumentException count = assertThrows(
                IllegalArgumentException.class, () -> excessive.unstack(0));
        IndexOutOfBoundsException scalarAxis = assertThrows(
                IndexOutOfBoundsException.class, () -> scalar.unstack(0));

        assertAll(
                () -> assertEquals(List.of(), empty),
                () -> assertThrows(
                        UnsupportedOperationException.class,
                        () -> empty.add(scalar)),
                () -> assertEquals(
                        "unstack axis 0 must have a statically known dimension",
                        unknown.getMessage()),
                () -> assertEquals(
                        "unstack axis 0 size 2147483648 exceeds maximum result count 2147483647",
                        count.getMessage()),
                () -> assertEquals(
                        "unstack axis 0 is outside shape rank 0", scalarAxis.getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void allCompositionResultsStayUnresolvedRegardlessOfInputLayoutAndStorage() {
        Shape shape = Shape.of(2, 3);
        List<Optional<LayoutDescriptor>> layouts = List.of(
                Optional.empty(),
                Optional.of(LayoutDescriptor.contiguous(shape)),
                Optional.of(LayoutDescriptor.of(shape, new long[] {4, 1}, 2, true)),
                Optional.of(LayoutDescriptor.of(shape, new long[] {0, 1}, 0, true)));

        for (Optional<LayoutDescriptor> layout : layouts) {
            TensorDescriptor descriptor =
                    new TensorDescriptor(DataType.FLOAT32, shape, layout, false);
            Tensor input = layout.isPresent()
                    ? TensorFactory.allocate(descriptor)
                    : TensorFactory.create(descriptor);
            Tensor concatenated = Tensor.concat(0, input);
            Tensor stacked = Tensor.stack(0, input);
            List<Tensor> unstacked = input.unstack(0);

            assertAll(
                    () -> assertEquals(layout.isPresent(), input.hostStorage().isPresent()),
                    () -> assertTrue(concatenated.descriptor().layout().isEmpty()),
                    () -> assertTrue(stacked.descriptor().layout().isEmpty()),
                    () -> assertTrue(unstacked.stream()
                            .allMatch(output -> output.descriptor().layout().isEmpty())),
                    () -> assertTrue(concatenated.hostStorage().isEmpty()),
                    () -> assertTrue(stacked.hostStorage().isEmpty()),
                    () -> assertTrue(unstacked.stream()
                            .allMatch(output -> output.hostStorage().isEmpty())));
        }
    }

    @Test
    void nestedCompositionAlwaysCreatesFreshExpressions() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2), Optional.empty(), true);

        Tensor firstConcat = Tensor.concat(0, input);
        Tensor secondConcat = Tensor.concat(0, input);
        Tensor nestedStack = Tensor.stack(0, firstConcat, secondConcat);
        List<Tensor> nestedUnstack = nestedStack.unstack(0);

        assertAll(
                () -> assertNotSame(firstConcat, secondConcat),
                () -> assertNotEquals(firstConcat.id(), secondConcat.id()),
                () -> assertEquals(List.of(firstConcat, secondConcat),
                        nestedStack.provenance().orElseThrow().inputs()),
                () -> assertEquals(2, nestedUnstack.size()),
                () -> assertSame(nestedStack,
                        nestedUnstack.getFirst().provenance().orElseThrow().inputs().getFirst()),
                () -> assertNotSame(nestedUnstack.get(0), nestedUnstack.get(1)));
    }

    @Test
    void unstackExhaustionMayConsumeEarlierIdsButReturnsNoPartialList() throws Exception {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2), Optional.empty(), false);
        AtomicLong next = nextTensorIdState();
        AtomicBoolean maximumClaimed = maximumTensorIdClaimedState();
        long savedNext = next.get();
        boolean savedMaximumClaimed = maximumClaimed.get();
        try {
            next.set(Long.MAX_VALUE);
            maximumClaimed.set(false);

            IllegalStateException failure = assertThrows(
                    IllegalStateException.class, () -> input.unstack(0));

            assertAll(
                    () -> assertEquals("tensor identifier space exhausted", failure.getMessage()),
                    () -> assertEquals(Long.MAX_VALUE, next.get()),
                    () -> assertTrue(maximumClaimed.get()));
        } finally {
            next.set(savedNext);
            maximumClaimed.set(savedMaximumClaimed);
        }
    }

    private static Tensor tensor(
            DataType dataType,
            Shape shape,
            Optional<LayoutDescriptor> layout,
            boolean requiresGrad) {
        return TensorFactory.create(new TensorDescriptor(dataType, shape, layout, requiresGrad));
    }

    private static AtomicLong nextTensorIdState() throws Exception {
        var field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    private static AtomicBoolean maximumTensorIdClaimedState() throws Exception {
        var field = TensorFactory.class.getDeclaredField("MAXIMUM_TENSOR_ID_CLAIMED");
        field.setAccessible(true);
        return (AtomicBoolean) field.get(null);
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
}

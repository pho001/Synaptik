package io.github.pho001.synaptik.data;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

/** Exercises prefix geometry, reusable fluent state and exact non-owning Tensor borrowing. */
final class PaddedSequenceBatchTest {
    @Test
    void exposesExactlyTheRecordAndPublicStaticFinalBuilderApi() throws NoSuchMethodException {
        assertTrue(PaddedSequenceBatch.class.isRecord());
        assertEquals(List.of("values", "validLengths", "axisOrder", "paddingSide"),
                Arrays.stream(PaddedSequenceBatch.class.getRecordComponents())
                        .map(component -> component.getName()).toList());
        assertEquals(List.of(Tensor.class, ValidSequenceLengths.class, SequenceAxisOrder.class, PaddingSide.class),
                Arrays.stream(PaddedSequenceBatch.class.getRecordComponents())
                        .map(component -> component.getType()).toList());
        assertEquals(1, PaddedSequenceBatch.class.getConstructors().length);
        assertEquals(Set.of("values", "validLengths", "axisOrder", "paddingSide", "builder", "equals", "hashCode", "toString"),
                Arrays.stream(PaddedSequenceBatch.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers()))
                        .map(method -> method.getName()).collect(Collectors.toSet()));
        assertEquals(8, PaddedSequenceBatch.class.getDeclaredMethods().length);
        assertEquals(List.of(PaddedSequenceBatch.Builder.class),
                Arrays.asList(PaddedSequenceBatch.class.getDeclaredClasses()));
        Class<?> builder = PaddedSequenceBatch.Builder.class;
        assertTrue(Modifier.isPublic(builder.getModifiers()));
        assertTrue(Modifier.isStatic(builder.getModifiers()));
        assertTrue(Modifier.isFinal(builder.getModifiers()));
        assertEquals(0, builder.getConstructors().length);
        assertEquals(1, builder.getDeclaredConstructors().length);
        assertTrue(Modifier.isPrivate(builder.getDeclaredConstructors()[0].getModifiers()));
        assertEquals(6, builder.getDeclaredMethods().length);
        assertEquals(Set.of("timeMajor", "batchMajor", "validLengths", "leftPadded", "rightPadded", "build"),
                Arrays.stream(builder.getDeclaredMethods()).map(method -> method.getName())
                        .collect(Collectors.toSet()));
        assertTrue(builder.getMethod("validLengths", long[].class).isVarArgs());
        assertEquals(builder, builder.getMethod("timeMajor").getReturnType());
        assertEquals(builder, builder.getMethod("batchMajor").getReturnType());
        assertEquals(builder, builder.getMethod("validLengths", long[].class).getReturnType());
        assertEquals(PaddedSequenceBatch.class, builder.getMethod("build").getReturnType());
        assertTrue(Modifier.isStatic(
                PaddedSequenceBatch.class.getMethod("builder", Tensor.class).getModifiers()));
        assertEquals(4, PaddedSequenceBatch.class.getDeclaredFields().length);
        assertFalse(AutoCloseable.class.isAssignableFrom(PaddedSequenceBatch.class));
        assertFalse(AutoCloseable.class.isAssignableFrom(builder));
    }

    @Test
    void directConstructionRetainsExactComponentsForBothNonSquareOrders() {
        var lengths = ValidSequenceLengths.of(4, 0, 2);
        for (var order : SequenceAxisOrder.values()) {
            Tensor values = values(sequenceShape(order, 4, 3));
            var batch = new PaddedSequenceBatch(values, lengths, order, PaddingSide.RIGHT);
            assertSame(values, batch.values());
            assertSame(lengths, batch.validLengths());
            assertSame(order, batch.axisOrder());
            assertTrue(values.hostStorage().isEmpty());
            assertTrue(values.provenance().isEmpty());
        }
    }

    @Test
    void fluentConstructionSupportsBothOrdersAndEitherSetterOrder() {
        Tensor timeMajor = values(Shape.of(4, 3));
        Tensor batchMajor = values(Shape.of(3, 4));
        assertEquals(new PaddedSequenceBatch(timeMajor, ValidSequenceLengths.of(3, 2, 1),
                        SequenceAxisOrder.TIME_MAJOR, PaddingSide.RIGHT),
                PaddedSequenceBatch.builder(timeMajor).rightPadded().timeMajor().validLengths(3, 2, 1).build());
        assertEquals(new PaddedSequenceBatch(batchMajor, ValidSequenceLengths.of(3, 2, 1),
                        SequenceAxisOrder.BATCH_MAJOR, PaddingSide.RIGHT),
                PaddedSequenceBatch.builder(batchMajor).rightPadded().validLengths(3, 2, 1).batchMajor().build());
        assertSame(timeMajor, PaddedSequenceBatch.builder(timeMajor).rightPadded()
                .validLengths(3, 2, 1).timeMajor().build().values());
        assertSame(batchMajor, PaddedSequenceBatch.builder(batchMajor).rightPadded()
                .batchMajor().validLengths(3, 2, 1).build().values());
    }

    @Test
    void builderRejectsNullValuesImmediately() {
        assertEquals("values", assertThrows(NullPointerException.class,
                () -> PaddedSequenceBatch.builder(null).rightPadded()).getMessage());
    }

    @Test
    void missingSelectionsFailBeforeInvalidGeometryAndNoAxisDefaultsExist() {
        var builder = assertDoesNotThrow(() -> PaddedSequenceBatch.builder(values(Shape.scalar())).rightPadded());
        assertThrows(IllegalStateException.class, builder::build);
        assertSame(builder, builder.timeMajor());
        assertThrows(IllegalStateException.class, builder::build);
        assertSame(builder, builder.validLengths());
        assertThrows(IllegalArgumentException.class, builder::build);
        var onlyLengths = PaddedSequenceBatch.builder(values(Shape.of(1))).rightPadded().validLengths(0);
        assertThrows(IllegalStateException.class, onlyLengths::build);
        onlyLengths.batchMajor();
        assertThrows(IllegalArgumentException.class, onlyLengths::build);
    }

    @Test
    void emptyLengthsAreExplicitAndDifferentFromMissingOrFailedSelection() {
        var builder = PaddedSequenceBatch.builder(values(Shape.of(0, 4))).rightPadded().batchMajor();
        assertThrows(IllegalStateException.class, builder::build);
        assertThrows(NullPointerException.class, () -> builder.validLengths((long[]) null));
        assertThrows(IllegalArgumentException.class, () -> builder.validLengths(-1));
        assertThrows(IllegalStateException.class, builder::build);
        builder.validLengths();
        assertEquals(0, builder.build().validLengths().batchSize());
    }

    @Test
    void settersChainOnSameInstanceAndLastSuccessfulAxisAndLengthsWin() {
        Tensor values = values(Shape.of(3, 4));
        var builder = PaddedSequenceBatch.builder(values).rightPadded();
        assertSame(builder, builder.timeMajor());
        assertSame(builder, builder.validLengths(1, 1, 1, 1));
        assertSame(builder, builder.batchMajor());
        assertSame(builder, builder.validLengths(3, 2, 1));
        var batch = builder.build();
        assertSame(values, batch.values());
        assertEquals(SequenceAxisOrder.BATCH_MAJOR, batch.axisOrder());
        assertArrayEquals(new long[] {3, 2, 1}, batch.validLengths().toLongArray());
        assertSame(builder, builder.timeMajor());
        assertSame(builder, builder.validLengths(1, 2, 3, 0));
        assertEquals(SequenceAxisOrder.TIME_MAJOR, builder.build().axisOrder());
    }

    @Test
    void snapshotsLengthsImmediatelyAndChecksNegativesBeforeBuild() {
        long[] source = {3, 2, 1};
        var builder = PaddedSequenceBatch.builder(values(Shape.of(3, 4))).rightPadded().validLengths(source);
        source[0] = -1;
        builder.batchMajor();
        assertArrayEquals(new long[] {3, 2, 1}, builder.build().validLengths().toLongArray());
        String message = assertThrows(IllegalArgumentException.class,
                () -> builder.validLengths(0, -1, -2)).getMessage();
        assertTrue(message.contains("row 1"));
        assertEquals("lengths", assertThrows(NullPointerException.class,
                () -> builder.validLengths((long[]) null)).getMessage());
        assertArrayEquals(new long[] {3, 2, 1}, builder.build().validLengths().toLongArray());
    }

    @Test
    void repeatBuildsAreFreshAndShareExactImmutableSelectionWithoutRecopy() {
        var builder = PaddedSequenceBatch.builder(values(Shape.of(3, 4))).rightPadded()
                .batchMajor().validLengths(3, 2, 1);
        var first = builder.build();
        var second = builder.build();
        assertNotSame(first, second);
        assertEquals(first, second);
        assertSame(first.values(), second.values());
        assertSame(first.validLengths(), second.validLengths());
        builder.validLengths(0, 1, 4);
        var third = builder.build();
        assertNotSame(first.validLengths(), third.validLengths());
        assertArrayEquals(new long[] {3, 2, 1}, first.validLengths().toLongArray());
        assertArrayEquals(new long[] {0, 1, 4}, third.validLengths().toLongArray());
    }

    @Test
    void previousBatchesSurviveSetterAndGeometryFailuresAndBuilderRecovers() {
        var builder = PaddedSequenceBatch.builder(values(Shape.of(3, 4))).rightPadded()
                .batchMajor().validLengths(3, 2, 1);
        var first = builder.build();
        var lengths = first.validLengths();
        int hash = first.hashCode();
        String text = first.toString();
        assertThrows(IllegalArgumentException.class, () -> builder.validLengths(-1));
        assertSame(lengths, builder.build().validLengths());
        builder.timeMajor();
        assertThrows(IllegalArgumentException.class, builder::build);
        builder.batchMajor();
        assertSame(lengths, builder.build().validLengths());
        builder.validLengths(5, 0, 0);
        assertThrows(IllegalArgumentException.class, builder::build);
        assertThrows(NullPointerException.class, () -> builder.validLengths((long[]) null));
        assertThrows(IllegalArgumentException.class, builder::build);
        builder.validLengths(4, 0, 1);
        assertArrayEquals(new long[] {4, 0, 1}, builder.build().validLengths().toLongArray());
        assertSame(lengths, first.validLengths());
        assertArrayEquals(new long[] {3, 2, 1}, lengths.toLongArray());
        assertEquals(hash, first.hashCode());
        assertEquals(text, first.toString());
    }

    @Test
    void sameChainFailurePreservesEarlierSuccessfulSelectionsInThatChain() {
        var builder = PaddedSequenceBatch.builder(values(Shape.of(3, 4))).rightPadded();
        assertThrows(IllegalArgumentException.class,
                () -> builder.timeMajor().batchMajor().validLengths(1, 2, 3).validLengths(-1));
        var batch = builder.build();
        assertEquals(SequenceAxisOrder.BATCH_MAJOR, batch.axisOrder());
        assertArrayEquals(new long[] {1, 2, 3}, batch.validLengths().toLongArray());
    }

    @Test
    void nullChecksPrecedeGeometryInDeclarationOrderWithParameterMessages() {
        assertEquals("values", assertThrows(NullPointerException.class,
                () -> new PaddedSequenceBatch(null, null, null, PaddingSide.RIGHT)).getMessage());
        Tensor invalid = values(Shape.scalar());
        assertEquals("validLengths", assertThrows(NullPointerException.class,
                () -> new PaddedSequenceBatch(invalid, null, null, PaddingSide.RIGHT)).getMessage());
        assertEquals("axisOrder", assertThrows(NullPointerException.class,
                () -> new PaddedSequenceBatch(invalid, ValidSequenceLengths.of(), null, PaddingSide.RIGHT)).getMessage());
    }

    @Test
    void rejectsScalarAndRankOneBeforeOtherGeometryChecks() {
        for (var order : SequenceAxisOrder.values()) {
            for (Shape shape : List.of(Shape.scalar(), Shape.of(3))) {
                String message = assertThrows(IllegalArgumentException.class,
                        () -> new PaddedSequenceBatch(values(shape), ValidSequenceLengths.of(), order, PaddingSide.RIGHT))
                        .getMessage();
                assertTrue(message.contains("rank"));
            }
        }
    }

    @Test
    void rejectsUnknownTimeOrBatchAxesIncludingExpressionDimensions() {
        for (var order : SequenceAxisOrder.values()) {
            for (int axis = 0; axis < 2; axis++) {
                var dimensions = sequenceShape(order, 4, 3).dimensions()
                        .toArray(Dimension[]::new);
                dimensions[axis] = new DynamicDimension("unknown");
                Tensor dynamic = values(Shape.ofDimensions(dimensions));
                String message = assertThrows(IllegalArgumentException.class,
                        () -> new PaddedSequenceBatch(dynamic, ValidSequenceLengths.of(3, 2, 1), order, PaddingSide.RIGHT))
                        .getMessage();
                assertTrue(message.contains("axis " + axis));
                dimensions[axis] = DimensionExpressions.unknown(0, Optional.empty());
                Tensor expression = values(Shape.ofDimensions(dimensions));
                assertThrows(IllegalArgumentException.class,
                        () -> new PaddedSequenceBatch(expression, ValidSequenceLengths.of(3, 2, 1), order, PaddingSide.RIGHT));
            }
        }
    }

    @Test
    void staticAxesAreCheckedBeforeCountAndCountBeforeUpperBounds() {
        Tensor unknown = values(Shape.ofDimensions(new StaticDimension(1), new DynamicDimension("B")));
        assertTrue(assertThrows(IllegalArgumentException.class,
                () -> new PaddedSequenceBatch(unknown, ValidSequenceLengths.of(9, 9),
                        SequenceAxisOrder.TIME_MAJOR, PaddingSide.RIGHT)).getMessage().contains("Batch axis"));
        String count = assertThrows(IllegalArgumentException.class,
                () -> new PaddedSequenceBatch(values(Shape.of(1, 3)), ValidSequenceLengths.of(9, 9),
                        SequenceAxisOrder.TIME_MAJOR, PaddingSide.RIGHT)).getMessage();
        assertTrue(count.contains("B=3"));
        assertTrue(count.contains("count=2"));
    }

    @Test
    void rejectsCountMismatchAndFirstExcessLengthInRowOrderForBothOrders() {
        for (var order : SequenceAxisOrder.values()) {
            Tensor values = values(sequenceShape(order, 4, 3));
            for (var lengths : List.of(ValidSequenceLengths.of(), ValidSequenceLengths.of(1, 2),
                    ValidSequenceLengths.of(1, 2, 3, 4))) {
                assertTrue(assertThrows(IllegalArgumentException.class,
                        () -> new PaddedSequenceBatch(values, lengths, order, PaddingSide.RIGHT))
                        .getMessage().contains("Batch"));
            }
            String message = assertThrows(IllegalArgumentException.class,
                    () -> new PaddedSequenceBatch(values, ValidSequenceLengths.of(0, 5, 6), order, PaddingSide.RIGHT))
                    .getMessage();
            assertTrue(message.contains("row 1"));
            assertTrue(message.contains("T=4"));
        }
    }

    @Test
    void zeroBatchAcceptsEmptyLengthsAtAnyTimeExtentAndZeroTimeRequiresZeros() {
        for (var order : SequenceAxisOrder.values()) {
            for (long time : new long[] {0, 4, Long.MAX_VALUE}) {
                assertDoesNotThrow(() -> new PaddedSequenceBatch(
                        values(sequenceShape(order, time, 0)), ValidSequenceLengths.of(), order, PaddingSide.RIGHT));
            }
            Tensor zeroTime = values(sequenceShape(order, 0, 3));
            assertDoesNotThrow(() -> new PaddedSequenceBatch(
                    zeroTime, ValidSequenceLengths.of(0, 0, 0), order, PaddingSide.RIGHT));
            String message = assertThrows(IllegalArgumentException.class,
                    () -> new PaddedSequenceBatch(zeroTime, ValidSequenceLengths.of(0, 1, 0), order, PaddingSide.RIGHT))
                    .getMessage();
            assertTrue(message.contains("row 1"));
            assertTrue(message.contains("T=0"));
        }
    }

    @Test
    void suffixDimensionsAreOpaqueIncludingZeroDynamicAndExpressionDimensions() {
        for (var order : SequenceAxisOrder.values()) {
            var leading = sequenceShape(order, 4, 3);
            Shape shape = Shape.ofDimensions(leading.dimension(0), leading.dimension(1),
                    new StaticDimension(0), new DynamicDimension("features"),
                    DimensionExpressions.unknown(1, Optional.empty()));
            Tensor values = values(shape);
            var batch = new PaddedSequenceBatch(values, ValidSequenceLengths.of(4, 3, 2), order, PaddingSide.RIGHT);
            assertSame(shape, batch.values().descriptor().shape());
            assertEquals(4, batch.validLengths().length(0));
            assertDoesNotThrow(() -> new PaddedSequenceBatch(
                    values(Shape.ofDimensions(leading.dimension(0), leading.dimension(1), new StaticDimension(0))),
                    ValidSequenceLengths.of(4, 3, 2), order, PaddingSide.RIGHT));
        }
    }

    @Test
    void maximumTimeAndLargeLengthsNeedNoProductsNarrowingOrTensorAllocation() {
        for (var order : SequenceAxisOrder.values()) {
            Tensor values = values(sequenceShape(order, Long.MAX_VALUE, 3));
            var lengths = ValidSequenceLengths.of(Long.MAX_VALUE, (long) Integer.MAX_VALUE + 1, 0);
            var batch = new PaddedSequenceBatch(values, lengths, order, PaddingSide.RIGHT);
            assertSame(lengths, batch.validLengths());
            assertTrue(values.hostStorage().isEmpty());
            assertTrue(values.descriptor().layout().isEmpty());
            var builder = PaddedSequenceBatch.builder(values).rightPadded().validLengths(lengths.toLongArray());
            if (order == SequenceAxisOrder.TIME_MAJOR) builder.timeMajor();
            else builder.batchMajor();
            assertSame(values, builder.build().values());
        }
    }

    @Test
    void oversizedBatchExtentsCannotMatchHostCountsWithoutNarrowing() {
        for (var order : SequenceAxisOrder.values()) {
            for (long batch : new long[] {(long) Integer.MAX_VALUE + 1, 1L << 32, Long.MAX_VALUE}) {
                String message = assertThrows(IllegalArgumentException.class,
                        () -> new PaddedSequenceBatch(values(sequenceShape(order, Long.MAX_VALUE, batch)),
                                ValidSequenceLengths.of(), order, PaddingSide.RIGHT)).getMessage();
                assertTrue(message.contains("B=" + batch));
                assertTrue(message.contains("count=0"));
            }
        }
    }

    @Test
    void acceptsEveryDtypeAndEveryModelValidGradientFlagWithoutChangingDescriptor() {
        for (DataType type : DataType.values()) {
            for (boolean requiresGrad : new boolean[] {false, true}) {
                if (requiresGrad && !type.isDifferentiable()) continue;
                var descriptor = new TensorDescriptor(type, Shape.of(3, 4), Optional.empty(), requiresGrad);
                Tensor values = TensorFactory.create(descriptor);
                var batch = PaddedSequenceBatch.builder(values).rightPadded().batchMajor().validLengths(3, 2, 1).build();
                assertSame(values, batch.values());
                assertSame(descriptor, batch.values().descriptor());
            }
        }
    }

    @Test
    void acceptsDerivedProvenanceWithoutCreatingAnotherExpression() {
        Tensor input = values(Shape.of(3, 4));
        Tensor derived = input.add(input);
        var provenance = derived.provenance().orElseThrow();
        var batch = PaddedSequenceBatch.builder(derived).rightPadded().batchMajor().validLengths(3, 2, 1).build();
        assertSame(derived, batch.values());
        assertSame(provenance, batch.values().provenance().orElseThrow());
        assertSame(derived, provenance.producer().output(provenance.outputIndex()));
        assertTrue(derived.hostStorage().isEmpty());
    }

    @Test
    void preservesResolvedStridedOffsetBroadcastAndUnresolvedLayoutsExactly() {
        Shape shape = Shape.of(3, 4);
        for (var layout : List.of(Optional.<LayoutDescriptor>empty(),
                Optional.of(LayoutDescriptor.contiguous(shape)),
                Optional.of(LayoutDescriptor.of(shape, new long[] {1, 3}, 0, true)),
                Optional.of(LayoutDescriptor.of(shape, new long[] {4, 1}, 7, true)),
                Optional.of(LayoutDescriptor.of(shape, new long[] {0, 1}, 0, true)))) {
            var descriptor = new TensorDescriptor(DataType.FLOAT32, shape, layout, false);
            int capacity = (int) layout.map(LayoutDescriptor::referencedElementSpan).orElse(1L).longValue();
            MemorySegment segment = MemorySegment.ofArray(new float[capacity]);
            var storage = new MemorySegmentStorage(DataType.FLOAT32, capacity, segment);
            Tensor values = TensorFactory.create(descriptor, Optional.empty(), Optional.of(storage));
            byte[] before = segment.toArray(JAVA_BYTE);
            var batch = PaddedSequenceBatch.builder(values).rightPadded().batchMajor().validLengths(3, 2, 1).build();
            assertSame(values, batch.values());
            assertSame(descriptor, values.descriptor());
            assertEquals(layout, values.descriptor().layout());
            if (layout.isPresent()) assertSame(layout.orElseThrow(), values.descriptor().layout().orElseThrow());
            assertSame(storage, values.hostStorage().orElseThrow());
            assertArrayEquals(before, segment.toArray(JAVA_BYTE));
            assertThrows(IllegalArgumentException.class,
                    () -> new PaddedSequenceBatch(values, ValidSequenceLengths.of(5, 2, 1),
                            SequenceAxisOrder.BATCH_MAJOR, PaddingSide.RIGHT));
            assertSame(storage, values.hostStorage().orElseThrow());
            assertArrayEquals(before, segment.toArray(JAVA_BYTE));
        }
    }

    @Test
    void realZerosAndArbitrarySuffixesAreNeitherValidatedNorChanged() {
        float[] cells = {0, 4, 0, 99, 7, 0, 99, 99, 0, 99, 99, 99};
        MemorySegment segment = MemorySegment.ofArray(cells);
        var storage = new MemorySegmentStorage(DataType.FLOAT32, cells.length, segment);
        Tensor values = withStorage(Shape.of(3, 4), storage);
        byte[] before = segment.toArray(JAVA_BYTE);
        var batch = PaddedSequenceBatch.builder(values).rightPadded().batchMajor().validLengths(3, 2, 1).build();
        assertArrayEquals(before, segment.toArray(JAVA_BYTE));
        assertSame(storage, values.hostStorage().orElseThrow());
        assertThrows(IllegalArgumentException.class,
                () -> PaddedSequenceBatch.builder(values).rightPadded().batchMajor().validLengths(3, 5, 1).build());
        assertArrayEquals(before, segment.toArray(JAVA_BYTE));
        assertSame(storage, values.hostStorage().orElseThrow());
        cells[3] = Float.NaN;
        cells[6] = Float.NEGATIVE_INFINITY;
        cells[7] = -15;
        cells[9] = 0;
        cells[10] = 123;
        cells[11] = Float.POSITIVE_INFINITY;
        byte[] changed = segment.toArray(JAVA_BYTE);
        assertEquals(batch, new PaddedSequenceBatch(values, ValidSequenceLengths.of(3, 2, 1),
                SequenceAxisOrder.BATCH_MAJOR, PaddingSide.RIGHT));
        assertArrayEquals(changed, segment.toArray(JAVA_BYTE));
        assertEquals(0, cells[0]);
        assertEquals(0, cells[2]);
        assertEquals(0, cells[5]);
        assertEquals(0, cells[8]);
    }

    @Test
    void acceptsReadOnlyStorageAndLeavesBytesAndAssociationUnchangedOnFailures() {
        byte[] cells = {0, 1, 2, 3, 4, 5};
        var storage = new MemorySegmentStorage(DataType.BOOL, 6, MemorySegment.ofArray(cells).asReadOnly());
        Tensor values = TensorFactory.create(new TensorDescriptor(DataType.BOOL, Shape.of(2, 3),
                Optional.of(LayoutDescriptor.contiguous(Shape.of(2, 3))), false),
                Optional.empty(), Optional.of(storage));
        var builder = PaddedSequenceBatch.builder(values).rightPadded().batchMajor().validLengths(1, 2);
        assertSame(values, builder.build().values());
        assertTrue(storage.isReadOnly());
        assertThrows(IllegalArgumentException.class, () -> builder.validLengths(-1));
        builder.validLengths(4, 0);
        assertThrows(IllegalArgumentException.class, builder::build);
        assertSame(storage, values.hostStorage().orElseThrow());
        assertArrayEquals(new byte[] {0, 1, 2, 3, 4, 5}, cells);
    }

    @Test
    void storageCanDieAfterAttachmentBeforeWrappingOrAfterWrappingWithoutNewOwnership() {
        Tensor values;
        HostTensorStorage storage;
        PaddedSequenceBatch first;
        try (Arena arena = Arena.ofConfined()) {
            storage = new MemorySegmentStorage(DataType.FLOAT32, 12, arena.allocate(48, 4));
            values = withStorage(Shape.of(3, 4), storage);
            first = PaddedSequenceBatch.builder(values).rightPadded().batchMajor().validLengths(3, 2, 1).build();
            assertTrue(storage.isAlive());
            assertSame(storage, first.values().hostStorage().orElseThrow());
        }
        assertFalse(storage.isAlive());
        var second = new PaddedSequenceBatch(values, first.validLengths(), first.axisOrder(), PaddingSide.RIGHT);
        var builder = PaddedSequenceBatch.builder(values).rightPadded().batchMajor().validLengths(3, 2, 1);
        assertEquals(first, builder.build());
        assertEquals(first, second);
        assertEquals(first.hashCode(), second.hashCode());
        assertDoesNotThrow(first::toString);
        builder.validLengths(5, 2, 1);
        assertThrows(IllegalArgumentException.class, builder::build);
        assertSame(storage, values.hostStorage().orElseThrow());
        assertSame(storage, values.clearHostStorage().orElseThrow());
    }

    @Test
    void storageAssociationRemainsMutableAndIsNeverSnapshottedByBatchOrBuilder() {
        Tensor values = values(Shape.of(3, 4));
        var builder = PaddedSequenceBatch.builder(values).rightPadded().batchMajor().validLengths(3, 2, 1);
        var batch = builder.build();
        int hash = batch.hashCode();
        String text = batch.toString();
        var first = new MemorySegmentStorage(DataType.FLOAT32, 0, MemorySegment.ofArray(new byte[0]));
        var second = new MemorySegmentStorage(DataType.FLOAT32, 1, MemorySegment.ofArray(new float[] {99}));
        assertTrue(values.replaceHostStorage(first).isEmpty());
        assertSame(first, batch.values().hostStorage().orElseThrow());
        assertSame(first, values.replaceHostStorage(second).orElseThrow());
        assertSame(second, batch.values().hostStorage().orElseThrow());
        assertSame(second, builder.build().values().hostStorage().orElseThrow());
        assertSame(second, values.clearHostStorage().orElseThrow());
        assertTrue(batch.values().hostStorage().isEmpty());
        assertEquals(hash, batch.hashCode());
        assertEquals(text, batch.toString());
        assertEquals(batch, builder.build());
    }

    @Test
    void equalityUsesTensorIdentityStructuralLengthsAndAxisIdentityNeverContents() {
        Tensor values = values(Shape.of(3, 3));
        long[] source = {3, 2, 1};
        var batch = new PaddedSequenceBatch(values, ValidSequenceLengths.of(source), SequenceAxisOrder.BATCH_MAJOR, PaddingSide.RIGHT);
        var equal = new PaddedSequenceBatch(values, ValidSequenceLengths.of(3, 2, 1), SequenceAxisOrder.BATCH_MAJOR, PaddingSide.RIGHT);
        int hash = batch.hashCode();
        String text = batch.toString();
        source[0] = -1;
        assertEquals(batch, batch);
        assertEquals(batch, equal);
        assertEquals(equal, batch);
        assertEquals(hash, equal.hashCode());
        assertEquals(text, batch.toString());
        assertNotEquals(batch, new PaddedSequenceBatch(values(Shape.of(3, 3)),
                equal.validLengths(), equal.axisOrder(), PaddingSide.RIGHT));
        assertNotEquals(batch, new PaddedSequenceBatch(values, ValidSequenceLengths.of(1, 2, 3), equal.axisOrder(), PaddingSide.RIGHT));
        assertNotEquals(batch, new PaddedSequenceBatch(values, equal.validLengths(), SequenceAxisOrder.TIME_MAJOR, PaddingSide.RIGHT));
        assertNotEquals(batch, null);
        assertNotEquals(batch, values);
        assertTrue(text.contains("PaddedSequenceBatch"));
        assertTrue(text.contains("ValidSequenceLengths"));
        assertTrue(text.contains("BATCH_MAJOR"));
        assertFalse(text.contains("hostStorage"));
    }

    @Test
    void sideIsExplicitRetainedAndParticipatesInEqualityWithoutChangingCounts() {
        Tensor values = values(Shape.of(3, 4));
        var builder = PaddedSequenceBatch.builder(values).batchMajor().validLengths(3, 2, 1);
        assertThrows(IllegalStateException.class, builder::build);
        assertSame(builder, builder.leftPadded());
        var left = builder.build();
        assertSame(PaddingSide.LEFT, left.paddingSide());
        assertSame(builder, builder.rightPadded());
        var right = builder.build();
        assertSame(PaddingSide.RIGHT, right.paddingSide());
        assertNotEquals(left, right);
        assertSame(left.validLengths(), right.validLengths());
        assertEquals(left, builder.leftPadded().build());
        assertEquals("paddingSide", assertThrows(NullPointerException.class,
                () -> new PaddedSequenceBatch(values(Shape.scalar()), ValidSequenceLengths.of(),
                        SequenceAxisOrder.TIME_MAJOR, null)).getMessage());
        for (var order : SequenceAxisOrder.values()) {
            for (var side : PaddingSide.values()) {
                var batch = new PaddedSequenceBatch(values(sequenceShape(order, Long.MAX_VALUE, 2)),
                        ValidSequenceLengths.of(Long.MAX_VALUE, 0), order, side);
                assertSame(side, batch.paddingSide());
                assertTrue(batch.values().hostStorage().isEmpty());
            }
        }
    }

    /**
     * Creates descriptor-only leaf values, deliberately leaving layout and storage unresolved.
     *
     * @param shape non-null Model shape, without calculating its element count
     * @return fresh FLOAT32 non-gradient leaf retaining that shape
     */
    private static Tensor values(Shape shape) {
        return TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, shape, Optional.empty(), false));
    }

    /**
     * Provides known leading geometry without products or data allocation.
     *
     * @param order non-null logical order
     * @param time non-negative long time extent
     * @param batch non-negative long row extent
     * @return non-null static rank-two shape in the selected order
     */
    private static Shape sequenceShape(SequenceAxisOrder order, long time, long batch) {
        return order == SequenceAxisOrder.TIME_MAJOR ? Shape.of(time, batch) : Shape.of(batch, time);
    }

    /**
     * Attaches the exact caller-owned storage to resolved FLOAT32 geometry using Model validation.
     *
     * @param shape non-null fully static shape compatible with the supplied storage capacity
     * @param storage non-null live borrowed FLOAT32 storage, retained without copying
     * @return fresh non-gradient Tensor with the exact storage association
     */
    private static Tensor withStorage(Shape shape, HostTensorStorage storage) {
        return TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false), Optional.empty(), Optional.of(storage));
    }
}

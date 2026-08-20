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
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentDirection;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

public final class TensorRecurrentScanExpressionTest {
    @Test
    void exposesExactlySixReceiverMethodsAndTwoExactResultRecords() throws Exception {
        assertTensorMethod("rnnScan", RecurrentScanResult.class,
                Tensor.class, Tensor.class, Tensor.class, Tensor.class,
                RecurrentDirection.class);
        assertTensorMethod("rnnScan", RecurrentScanResult.class,
                Tensor.class, Tensor.class, Tensor.class, Tensor.class, Tensor.class,
                RecurrentDirection.class);
        assertTensorMethod("gruScan", RecurrentScanResult.class,
                Tensor.class, Tensor.class, Tensor.class, Tensor.class,
                RecurrentDirection.class);
        assertTensorMethod("gruScan", RecurrentScanResult.class,
                Tensor.class, Tensor.class, Tensor.class, Tensor.class, Tensor.class,
                RecurrentDirection.class);
        assertTensorMethod("lstmScan", LstmRecurrentScanResult.class,
                Tensor.class, Tensor.class, Tensor.class, Tensor.class, Tensor.class,
                RecurrentDirection.class);
        assertTensorMethod("lstmScan", LstmRecurrentScanResult.class,
                Tensor.class, Tensor.class, Tensor.class, Tensor.class, Tensor.class, Tensor.class,
                RecurrentDirection.class);

        List<Method> scanMethods = Arrays.stream(Tensor.class.getDeclaredMethods())
                .filter(method -> Set.of("rnnScan", "gruScan", "lstmScan")
                        .contains(method.getName()))
                .toList();
        assertAll(
                () -> assertEquals(6, scanMethods.size()),
                () -> assertEquals(208, Arrays.stream(Tensor.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers())).count()),
                () -> assertRecord(
                        RecurrentScanResult.class,
                        List.of("outputs", "finalHidden")),
                () -> assertRecord(
                        LstmRecurrentScanResult.class,
                        List.of("outputs", "finalHidden", "finalCell")),
                () -> assertTrue(Modifier.isFinal(TensorRecurrentScanExpressions.class
                        .getModifiers())),
                () -> assertFalse(Modifier.isPublic(TensorRecurrentScanExpressions.class
                        .getModifiers())),
                () -> assertEquals(0, TensorRecurrentScanExpressions.class.getDeclaredFields()
                        .length),
                () -> assertEquals(0, TensorRecurrentScanExpressions.class.getDeclaredClasses()
                        .length),
                () -> assertEquals(1, TensorRecurrentScanExpressions.class
                        .getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isPrivate(TensorRecurrentScanExpressions.class
                        .getDeclaredConstructors()[0].getModifiers())));
    }

    @Test
    void constructsEveryKindDirectionAndBiasVariantWithExactOrderedInputs() {
        Fixtures f = fixtures(DataType.FLOAT32, 5, 2, 4, 3, false);
        for (RecurrentDirection direction : RecurrentDirection.values()) {
            assertOccurrence(
                    f.input.rnnScan(f.lengths, f.hidden, f.rnnInputWeight, f.rnnHiddenWeight,
                            direction),
                    RecurrentScanKind.RNN_TANH,
                    direction,
                    List.of(f.input, f.lengths, f.hidden, f.rnnInputWeight, f.rnnHiddenWeight));
            assertOccurrence(
                    f.input.rnnScan(f.lengths, f.hidden, f.rnnInputWeight, f.rnnHiddenWeight,
                            f.rnnBias, direction),
                    RecurrentScanKind.RNN_TANH,
                    direction,
                    List.of(f.input, f.lengths, f.hidden, f.rnnInputWeight, f.rnnHiddenWeight,
                            f.rnnBias));
            assertOccurrence(
                    f.input.gruScan(f.lengths, f.hidden, f.gruInputWeight, f.gruHiddenWeight,
                            direction),
                    RecurrentScanKind.GRU_RESET_AFTER,
                    direction,
                    List.of(f.input, f.lengths, f.hidden, f.gruInputWeight, f.gruHiddenWeight));
            assertOccurrence(
                    f.input.gruScan(f.lengths, f.hidden, f.gruInputWeight, f.gruHiddenWeight,
                            f.gruBias, direction),
                    RecurrentScanKind.GRU_RESET_AFTER,
                    direction,
                    List.of(f.input, f.lengths, f.hidden, f.gruInputWeight, f.gruHiddenWeight,
                            f.gruBias));
            assertLstmOccurrence(
                    f.input.lstmScan(f.lengths, f.hidden, f.cell, f.lstmInputWeight,
                            f.lstmHiddenWeight, direction),
                    direction,
                    List.of(f.input, f.lengths, f.hidden, f.cell, f.lstmInputWeight,
                            f.lstmHiddenWeight));
            assertLstmOccurrence(
                    f.input.lstmScan(f.lengths, f.hidden, f.cell, f.lstmInputWeight,
                            f.lstmHiddenWeight, f.lstmBias, direction),
                    direction,
                    List.of(f.input, f.lengths, f.hidden, f.cell, f.lstmInputWeight,
                            f.lstmHiddenWeight, f.lstmBias));
        }
    }

    @Test
    void derivesStaticDescriptorsAndCanonicalSharedProducerSlots() {
        Fixtures f = fixtures(DataType.FLOAT64, 5, 2, 4, 3, true);
        LstmRecurrentScanResult result = f.input.lstmScan(
                f.lengths, f.hidden, f.cell, f.lstmInputWeight, f.lstmHiddenWeight,
                f.lstmBias, RecurrentDirection.REVERSE);
        TensorProducer producer = result.outputs().provenance().orElseThrow().producer();
        assertAll(
                () -> assertEquals(Shape.of(5, 2, 3), result.outputs().descriptor().shape()),
                () -> assertSame(f.hidden.descriptor().shape(),
                        result.finalHidden().descriptor().shape()),
                () -> assertSame(f.cell.descriptor().shape(),
                        result.finalCell().descriptor().shape()),
                () -> assertSame(DataType.FLOAT64, result.outputs().descriptor().dataType()),
                () -> assertTrue(result.outputs().descriptor().requiresGrad()),
                () -> assertTrue(result.finalHidden().descriptor().requiresGrad()),
                () -> assertTrue(result.finalCell().descriptor().requiresGrad()),
                () -> assertTrue(result.outputs().descriptor().layout().isEmpty()),
                () -> assertTrue(result.finalHidden().descriptor().layout().isEmpty()),
                () -> assertTrue(result.finalCell().descriptor().layout().isEmpty()),
                () -> assertEquals(3, producer.outputCount()),
                () -> assertSame(result.outputs(), producer.output(0)),
                () -> assertSame(result.finalHidden(), producer.output(1)),
                () -> assertSame(result.finalCell(), producer.output(2)),
                () -> assertEquals(0, result.outputs().provenance().orElseThrow().outputIndex()),
                () -> assertEquals(1,
                        result.finalHidden().provenance().orElseThrow().outputIndex()),
                () -> assertEquals(2,
                        result.finalCell().provenance().orElseThrow().outputIndex()),
                () -> assertNotEquals(result.outputs().id(), result.finalHidden().id()),
                () -> assertNotEquals(result.finalHidden().id(), result.finalCell().id()));
    }

    @Test
    void createsIdentityDistinctFlatOccurrencesWithoutIntermediateTensors() {
        Fixtures f = fixtures(DataType.BFLOAT16, 3, 1, 2, 2, false);
        RecurrentScanResult first = f.input.gruScan(
                f.lengths, f.hidden, f.gruInputWeight, f.gruHiddenWeight,
                RecurrentDirection.FORWARD);
        RecurrentScanResult second = f.input.gruScan(
                f.lengths, f.hidden, f.gruInputWeight, f.gruHiddenWeight,
                RecurrentDirection.FORWARD);
        TensorProducer firstProducer = first.outputs().provenance().orElseThrow().producer();
        TensorProducer secondProducer = second.outputs().provenance().orElseThrow().producer();
        assertAll(
                () -> assertNotSame(first, second),
                () -> assertNotSame(firstProducer, secondProducer),
                () -> assertNotSame(firstProducer.operation(), secondProducer.operation()),
                () -> assertNotSame(first.outputs(), second.outputs()),
                () -> assertEquals(5, firstProducer.inputs().size()),
                () -> assertEquals(2, firstProducer.outputCount()),
                () -> assertTrue(firstProducer.inputs().stream()
                        .allMatch(tensor -> tensor.provenance().isEmpty())),
                () -> assertSame(first.finalHidden(), firstProducer.output(1)));
    }

    @Test
    void acceptsZeroTimeAndZeroBatchAndDoesNotRequireLengthStorage() {
        Fixtures zeroTime = fixtures(DataType.FLOAT32, 0, 2, 4, 3, false);
        RecurrentScanResult timeResult = zeroTime.input.rnnScan(
                zeroTime.lengths, zeroTime.hidden, zeroTime.rnnInputWeight,
                zeroTime.rnnHiddenWeight, RecurrentDirection.REVERSE);
        Fixtures zeroBatch = fixtures(DataType.FLOAT32, 4, 0, 2, 3, false);
        LstmRecurrentScanResult batchResult = zeroBatch.input.lstmScan(
                zeroBatch.lengths, zeroBatch.hidden, zeroBatch.cell,
                zeroBatch.lstmInputWeight, zeroBatch.lstmHiddenWeight,
                RecurrentDirection.FORWARD);
        assertAll(
                () -> assertEquals(Shape.of(0, 2, 3),
                        timeResult.outputs().descriptor().shape()),
                () -> assertEquals(Shape.of(2, 3),
                        timeResult.finalHidden().descriptor().shape()),
                () -> assertEquals(Shape.of(4, 0, 3),
                        batchResult.outputs().descriptor().shape()),
                () -> assertEquals(Shape.of(0, 3),
                        batchResult.finalCell().descriptor().shape()),
                () -> assertTrue(zeroTime.lengths.hostStorage().isEmpty()),
                () -> assertTrue(zeroBatch.lengths.hostStorage().isEmpty()));
    }

    @Test
    void combinesGradientRequestsFromFloatingRolesOnly() {
        Fixtures none = fixtures(DataType.FLOAT32, 2, 2, 3, 4, false);
        LstmRecurrentScanResult falseResult = none.input.lstmScan(
                none.lengths, none.hidden, none.cell, none.lstmInputWeight,
                none.lstmHiddenWeight, none.lstmBias, RecurrentDirection.FORWARD);
        Tensor gradBias = tensor(DataType.FLOAT32, Shape.of(16), true);
        LstmRecurrentScanResult trueResult = none.input.lstmScan(
                none.lengths, none.hidden, none.cell, none.lstmInputWeight,
                none.lstmHiddenWeight, gradBias, RecurrentDirection.FORWARD);
        assertAll(
                () -> assertFalse(falseResult.outputs().descriptor().requiresGrad()),
                () -> assertFalse(falseResult.finalHidden().descriptor().requiresGrad()),
                () -> assertFalse(falseResult.finalCell().descriptor().requiresGrad()),
                () -> assertTrue(trueResult.outputs().descriptor().requiresGrad()),
                () -> assertTrue(trueResult.finalHidden().descriptor().requiresGrad()),
                () -> assertTrue(trueResult.finalCell().descriptor().requiresGrad()));
    }

    @Test
    void validatesNullsInDeclarationOrder() {
        Fixtures f = fixtures(DataType.FLOAT32, 2, 2, 3, 4, false);
        assertAll(
                () -> assertEquals("input", assertThrows(NullPointerException.class,
                        () -> TensorRecurrentScanExpressions.rnnScan(
                                null, null, null, null, null, null)).getMessage()),
                () -> assertEquals("validLengths", assertThrows(NullPointerException.class,
                        () -> TensorRecurrentScanExpressions.rnnScan(
                                f.input, null, null, null, null, null)).getMessage()),
                () -> assertEquals("initialCell", assertThrows(NullPointerException.class,
                        () -> TensorRecurrentScanExpressions.lstmScan(
                                f.input, f.lengths, f.hidden, null, null, null, null)).getMessage()),
                () -> assertEquals("bias", assertThrows(NullPointerException.class,
                        () -> TensorRecurrentScanExpressions.gruScan(
                                f.input, f.lengths, f.hidden, f.gruInputWeight,
                                f.gruHiddenWeight, null, null)).getMessage()),
                () -> assertEquals("direction", assertThrows(NullPointerException.class,
                        () -> f.input.rnnScan(f.lengths, f.hidden, f.rnnInputWeight,
                                f.rnnHiddenWeight, (RecurrentDirection) null)).getMessage()));
    }

    @Test
    void rejectsDescriptorViolationsBeforeAllocatingOutputIdentifiers() throws Exception {
        Fixtures f = fixtures(DataType.FLOAT32, 2, 2, 3, 4, false);
        AtomicLong next = nextIds();
        Tensor integralInput = tensor(DataType.INT32, Shape.of(2, 2, 3), false);
        Tensor rankTwoInput = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        Tensor dynamicInput = tensor(DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("T"),
                        f.input.descriptor().shape().dimension(1),
                        f.input.descriptor().shape().dimension(2)), false);
        Tensor zeroFeatureInput = tensor(DataType.FLOAT32, Shape.of(2, 2, 0), false);
        Tensor intLengths = tensor(DataType.INT32, Shape.of(2), false);
        Tensor wrongLengths = tensor(DataType.INT64, Shape.of(3), false);
        Tensor wrongTypeHidden = tensor(DataType.FLOAT64, Shape.of(2, 4), false);
        Tensor zeroHidden = tensor(DataType.FLOAT32, Shape.of(2, 0), false);
        Tensor wrongInputWeight = tensor(DataType.FLOAT32, Shape.of(5, 3), false);
        Tensor wrongHiddenWeight = tensor(DataType.FLOAT32, Shape.of(4, 5), false);
        long before = next.get();
        assertAll(
                () -> assertFailure("floating data type",
                        integralInput, f.lengths, f.hidden,
                        f.rnnInputWeight, f.rnnHiddenWeight),
                () -> assertFailure("input rank must be 3",
                        rankTwoInput, f.lengths, f.hidden,
                        f.rnnInputWeight, f.rnnHiddenWeight),
                () -> assertFailure("input must have a fully static shape",
                        dynamicInput,
                        f.lengths, f.hidden, f.rnnInputWeight, f.rnnHiddenWeight),
                () -> assertFailure("inputSize must be positive",
                        zeroFeatureInput, f.lengths, f.hidden,
                        f.rnnInputWeight, f.rnnHiddenWeight),
                () -> assertFailure("validLengths data type must be INT64", f.input,
                        intLengths, f.hidden,
                        f.rnnInputWeight, f.rnnHiddenWeight),
                () -> assertFailure("validLengths batch extent mismatch", f.input,
                        wrongLengths, f.hidden,
                        f.rnnInputWeight, f.rnnHiddenWeight),
                () -> assertFailure("initialHidden data type must match input", f.input,
                        f.lengths, wrongTypeHidden,
                        f.rnnInputWeight, f.rnnHiddenWeight),
                () -> assertFailure("hiddenSize must be positive", f.input, f.lengths,
                        zeroHidden,
                        f.rnnInputWeight, f.rnnHiddenWeight),
                () -> assertFailure("inputWeight packedHiddenSize extent mismatch", f.input,
                        f.lengths, f.hidden, wrongInputWeight,
                        f.rnnHiddenWeight),
                () -> assertFailure("hiddenWeight hiddenSize extent mismatch", f.input,
                        f.lengths, f.hidden, f.rnnInputWeight,
                        wrongHiddenWeight),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void rejectsLstmCellAndPackedGateOverflowAtTheSpecifiedStages() throws Exception {
        Fixtures f = fixtures(DataType.FLOAT32, 2, 2, 3, 4, false);
        AtomicLong next = nextIds();
        Tensor hugeHidden = tensor(DataType.FLOAT32, Shape.of(2, Long.MAX_VALUE), false);
        Tensor wrongCell = tensor(DataType.FLOAT32, Shape.of(2, 5), false);
        long before = next.get();
        assertAll(
                () -> assertTrue(assertThrows(IllegalArgumentException.class,
                        () -> f.input.lstmScan(f.lengths, f.hidden,
                                wrongCell,
                                f.lstmInputWeight, f.lstmHiddenWeight,
                                RecurrentDirection.FORWARD)).getMessage().contains(
                                        "initialCell hiddenSize extent mismatch")),
                () -> assertTrue(assertThrows(IllegalArgumentException.class,
                        () -> f.input.gruScan(f.lengths, hugeHidden,
                                f.gruInputWeight, f.gruHiddenWeight,
                                RecurrentDirection.FORWARD)).getMessage().contains(
                                        "packed gate extent overflow")),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void factoryExhaustionAfterFirstOutputPublishesNoPartialResult() throws Exception {
        Fixtures f = fixtures(DataType.FLOAT32, 2, 2, 3, 4, false);
        AtomicLong next = nextIds();
        AtomicBoolean maximumClaimed = maximumClaimed();
        long savedNext = next.get();
        boolean savedClaimed = maximumClaimed.get();
        try {
            next.set(Long.MAX_VALUE);
            maximumClaimed.set(false);
            IllegalStateException failure = assertThrows(IllegalStateException.class,
                    () -> f.input.rnnScan(f.lengths, f.hidden, f.rnnInputWeight,
                            f.rnnHiddenWeight, RecurrentDirection.FORWARD));
            assertAll(
                    () -> assertEquals("tensor identifier space exhausted", failure.getMessage()),
                    () -> assertTrue(maximumClaimed.get()),
                    () -> assertEquals(Long.MAX_VALUE, next.get()));
        } finally {
            next.set(savedNext);
            maximumClaimed.set(savedClaimed);
        }
    }

    @Test
    void resultRecordsRejectNullInOrderAndRetainExactReferences() {
        Tensor first = tensor(DataType.FLOAT32, Shape.of(1), false);
        Tensor second = tensor(DataType.FLOAT32, Shape.of(1), false);
        Tensor third = tensor(DataType.FLOAT32, Shape.of(1), false);
        RecurrentScanResult recurrent = new RecurrentScanResult(first, second);
        LstmRecurrentScanResult lstm = new LstmRecurrentScanResult(first, second, third);
        assertAll(
                () -> assertEquals("outputs", assertThrows(NullPointerException.class,
                        () -> new RecurrentScanResult(null, null)).getMessage()),
                () -> assertEquals("finalHidden", assertThrows(NullPointerException.class,
                        () -> new RecurrentScanResult(first, null)).getMessage()),
                () -> assertEquals("finalCell", assertThrows(NullPointerException.class,
                        () -> new LstmRecurrentScanResult(first, second, null)).getMessage()),
                () -> assertSame(first, recurrent.outputs()),
                () -> assertSame(second, recurrent.finalHidden()),
                () -> assertSame(first, lstm.outputs()),
                () -> assertSame(second, lstm.finalHidden()),
                () -> assertSame(third, lstm.finalCell()));
    }

    private static void assertOccurrence(
            RecurrentScanResult result,
            RecurrentScanKind kind,
            RecurrentDirection direction,
            List<Tensor> inputs) {
        TensorProducer producer = result.outputs().provenance().orElseThrow().producer();
        assertAll(
                () -> assertSame(producer,
                        result.finalHidden().provenance().orElseThrow().producer()),
                () -> assertSame(kind, producer.operation().kind()),
                () -> assertSame(direction, producer.operation().attrs()),
                () -> assertEquals(inputs, producer.inputs()),
                () -> assertEquals(2, producer.outputCount()),
                () -> assertSame(result.outputs(), producer.output(0)),
                () -> assertSame(result.finalHidden(), producer.output(1)));
    }

    private static void assertLstmOccurrence(
            LstmRecurrentScanResult result,
            RecurrentDirection direction,
            List<Tensor> inputs) {
        TensorProducer producer = result.outputs().provenance().orElseThrow().producer();
        assertAll(
                () -> assertSame(producer,
                        result.finalHidden().provenance().orElseThrow().producer()),
                () -> assertSame(producer,
                        result.finalCell().provenance().orElseThrow().producer()),
                () -> assertSame(RecurrentScanKind.LSTM, producer.operation().kind()),
                () -> assertSame(direction, producer.operation().attrs()),
                () -> assertEquals(inputs, producer.inputs()),
                () -> assertEquals(3, producer.outputCount()),
                () -> assertSame(result.finalCell(), producer.output(2)));
    }

    private static void assertFailure(
            String message,
            Tensor input,
            Tensor lengths,
            Tensor hidden,
            Tensor inputWeight,
            Tensor hiddenWeight) {
        assertTrue(assertThrows(IllegalArgumentException.class,
                () -> input.rnnScan(lengths, hidden, inputWeight, hiddenWeight,
                        RecurrentDirection.FORWARD)).getMessage().contains(message));
    }

    private static void assertTensorMethod(
            String name, Class<?> returnType, Class<?>... parameters) throws Exception {
        Method method = Tensor.class.getDeclaredMethod(name, parameters);
        assertAll(
                () -> assertSame(returnType, method.getReturnType()),
                () -> assertTrue(Modifier.isPublic(method.getModifiers())),
                () -> assertFalse(Modifier.isStatic(method.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(method.getModifiers())),
                () -> assertFalse(method.isVarArgs()));
    }

    private static void assertRecord(Class<?> type, List<String> componentNames) {
        assertAll(
                () -> assertTrue(type.isRecord()),
                () -> assertEquals(componentNames,
                        Arrays.stream(type.getRecordComponents())
                                .map(component -> component.getName()).toList()),
                () -> assertTrue(Arrays.stream(type.getRecordComponents())
                        .allMatch(component -> component.getType() == Tensor.class)),
                () -> assertEquals(0, type.getDeclaredFields().length - componentNames.size()),
                () -> assertEquals(0, type.getDeclaredClasses().length));
    }

    private static Fixtures fixtures(
            DataType type, long time, long batch, long inputSize, long hiddenSize,
            boolean requiresGrad) {
        return new Fixtures(
                tensor(type, Shape.of(time, batch, inputSize), requiresGrad),
                tensor(DataType.INT64, Shape.of(batch), false),
                tensor(type, Shape.of(batch, hiddenSize), requiresGrad),
                tensor(type, Shape.of(batch, hiddenSize), requiresGrad),
                tensor(type, Shape.of(hiddenSize, inputSize), requiresGrad),
                tensor(type, Shape.of(hiddenSize, hiddenSize), requiresGrad),
                tensor(type, Shape.of(hiddenSize), requiresGrad),
                tensor(type, Shape.of(3 * hiddenSize, inputSize), requiresGrad),
                tensor(type, Shape.of(3 * hiddenSize, hiddenSize), requiresGrad),
                tensor(type, Shape.of(3 * hiddenSize), requiresGrad),
                tensor(type, Shape.of(4 * hiddenSize, inputSize), requiresGrad),
                tensor(type, Shape.of(4 * hiddenSize, hiddenSize), requiresGrad),
                tensor(type, Shape.of(4 * hiddenSize), requiresGrad));
    }

    private static Tensor tensor(DataType type, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(new TensorDescriptor(
                type, shape, Optional.empty(), requiresGrad));
    }

    private static AtomicLong nextIds() throws Exception {
        var field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    private static AtomicBoolean maximumClaimed() throws Exception {
        var field = TensorFactory.class.getDeclaredField("MAXIMUM_TENSOR_ID_CLAIMED");
        field.setAccessible(true);
        return (AtomicBoolean) field.get(null);
    }

    private record Fixtures(
            Tensor input,
            Tensor lengths,
            Tensor hidden,
            Tensor cell,
            Tensor rnnInputWeight,
            Tensor rnnHiddenWeight,
            Tensor rnnBias,
            Tensor gruInputWeight,
            Tensor gruHiddenWeight,
            Tensor gruBias,
            Tensor lstmInputWeight,
            Tensor lstmHiddenWeight,
            Tensor lstmBias) {
    }
}

package io.github.pho001.synaptik.nn.layers;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.initialization.ParameterInitialization;
import io.github.pho001.synaptik.nn.module.StateDictionary;
import io.github.pho001.synaptik.nn.module.StateEntry;
import io.github.pho001.synaptik.nn.module.StateKind;
import java.lang.reflect.Field;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

@Execution(ExecutionMode.SAME_THREAD)
class ConvolutionStateDictionaryTest {
    @Test
    void strictLoadRetainsExactCandidatesCreatesNoIdAndIgnoresPolicyAndSeed()
            throws ReflectiveOperationException {
        Conv2d first = conv2d(true, ParameterInitialization.glorotUniform(), 13L);
        Conv2d second = conv2d(true, ParameterInitialization.ones(), Long.MIN_VALUE);
        Tensor weight = parameter(Shape.of(6, 2, 3, 5), DataType.FLOAT32, true);
        Tensor bias = parameter(Shape.of(6), DataType.FLOAT32, true);
        StateDictionary candidate = dictionary(
                entry("bias", StateKind.PARAMETER, bias),
                entry("weight", StateKind.PARAMETER, weight));
        AtomicLong ids = nextTensorIdState();
        long before = ids.get();

        assertThrows(IllegalStateException.class, first::stateDictionary);
        first.loadStateDictionary(candidate);
        second.loadStateDictionary(candidate);

        assertAll(
                () -> assertEquals(before, ids.get()),
                () -> assertSame(weight, first.weight().value()),
                () -> assertSame(bias, first.bias().orElseThrow().value()),
                () -> assertSame(weight, second.weight().value()),
                () -> assertSame(bias, second.bias().orElseThrow().value()),
                () -> assertEquals(List.of("weight", "bias"), first.stateDictionary().entries()
                        .stream().map(StateEntry::path).toList()),
                () -> assertSame(weight, first.stateDictionary().entries().get(0).value()),
                () -> assertSame(bias, first.stateDictionary().entries().get(1).value()));
    }

    @Test
    void noBiasLoadPublishesOnlyWeightAndBiasAccessorStaysEmpty()
            throws ReflectiveOperationException {
        Conv1d layer = new Conv1d(4, 3, 1, 0, 1, 2, false, DataType.FLOAT64,
                ParameterInitialization.normal(7.0d, 0.0d), 99L);
        Tensor weight = parameter(Shape.of(4, 2, 3), DataType.FLOAT64, true);
        AtomicLong ids = nextTensorIdState();
        long before = ids.get();

        layer.loadStateDictionary(dictionary(entry("weight", StateKind.PARAMETER, weight)));

        assertAll(
                () -> assertEquals(before, ids.get()),
                () -> assertSame(weight, layer.weight().value()),
                () -> assertTrue(layer.bias().isEmpty()),
                () -> assertEquals(List.of("weight"), layer.stateDictionary().entries()
                        .stream().map(StateEntry::path).toList()));
    }

    @Test
    void exactJavaArrayLimitBoundaryIsAcceptedWithoutAllocatingStorage() {
        Conv1d layer = new Conv1d(1, 1, 1, 0, 1, 1, false, DataType.FLOAT32,
                ParameterInitialization.zeros(), 0L);
        Tensor boundary = parameter(Shape.of(1, Integer.MAX_VALUE, 1),
                DataType.FLOAT32, true);

        layer.loadStateDictionary(dictionary(entry("weight", StateKind.PARAMETER, boundary)));

        assertAll(
                () -> assertSame(boundary, layer.weight().value()),
                () -> assertEquals(List.of("weight"), layer.stateDictionary().entries()
                        .stream().map(StateEntry::path).toList()));
    }

    @Test
    void missingUnexpectedAndKindFailuresLeaveCompleteReservedGroupUnpublished() {
        Conv2d missing = conv2d(true, ParameterInitialization.zeros(), 1L);
        Conv2d unexpected = conv2d(false, ParameterInitialization.zeros(), 1L);
        Conv2d kind = conv2d(false, ParameterInitialization.zeros(), 1L);
        Tensor weight = parameter(Shape.of(6, 2, 3, 5), DataType.FLOAT32, true);
        Tensor bias = parameter(Shape.of(6), DataType.FLOAT32, true);

        assertAll(
                () -> assertThrows(IllegalArgumentException.class, () -> missing.loadStateDictionary(
                        dictionary(entry("weight", StateKind.PARAMETER, weight)))),
                () -> assertThrows(IllegalArgumentException.class, () -> unexpected.loadStateDictionary(
                        dictionary(entry("weight", StateKind.PARAMETER, weight),
                                entry("bias", StateKind.PARAMETER, bias)))),
                () -> assertThrows(IllegalArgumentException.class, () -> kind.loadStateDictionary(
                        dictionary(entry("weight", StateKind.BUFFER, weight)))),
                () -> assertUnbound(missing),
                () -> assertUnbound(unexpected),
                () -> assertUnbound(kind));
    }

    @Test
    void convolutionSpecificTypeRankShapeGradientGroupAndArrayValidatorsRollBack() {
        assertRejectedUnbound(parameter(Shape.of(4, 2, 2, 2, 2), DataType.FLOAT32, true),
                "data type");
        assertRejectedUnbound(parameter(Shape.of(4, 2, 2, 2), DataType.FLOAT64, true),
                "rank-five");
        assertRejectedUnbound(parameter(Shape.of(5, 2, 2, 2, 2), DataType.FLOAT64, true),
                "outChannels");
        assertRejectedUnbound(parameter(Shape.of(4, 2, 3, 2, 2), DataType.FLOAT64, true),
                "kernelDepth");
        assertRejectedUnbound(parameter(Shape.of(4, 2, 2, 2, 2), DataType.FLOAT64, false),
                "requiresGrad");
        assertRejectedUnbound(parameter(Shape.of(4, Long.MAX_VALUE, 2, 2, 2),
                DataType.FLOAT64, true), "overflow");
        assertRejectedUnbound(parameter(Shape.of(4, 67_108_864, 2, 2, 2),
                DataType.FLOAT64, true), "Java array limit");
    }

    @Test
    void invalidBiasRollsBackWeightAndLaterValidLoadPublishesExactOrder() {
        Conv3d layer = conv3d(true);
        Tensor weight = parameter(Shape.of(4, 2, 2, 2, 2), DataType.FLOAT64, true);
        Tensor wrongBias = parameter(Shape.of(5), DataType.FLOAT64, true);
        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
                () -> layer.loadStateDictionary(dictionary(
                        entry("weight", StateKind.PARAMETER, weight),
                        entry("bias", StateKind.PARAMETER, wrongBias))));
        assertAll(
                () -> assertTrue(failure.getMessage().contains("bias")),
                () -> assertUnbound(layer));

        Tensor bias = parameter(Shape.of(4), DataType.FLOAT64, true);
        layer.loadStateDictionary(dictionary(
                entry("bias", StateKind.PARAMETER, bias),
                entry("weight", StateKind.PARAMETER, weight)));
        assertAll(
                () -> assertSame(weight, layer.weight().value()),
                () -> assertSame(bias, layer.bias().orElseThrow().value()),
                () -> assertEquals(List.of("weight", "bias"), layer.stateDictionary().entries()
                        .stream().map(StateEntry::path).toList()));
    }

    @Test
    void boundReplacementIsStableAndFailedReloadDoesNotMutateEitherWrapper() {
        Conv2d layer = conv2d(true, ParameterInitialization.zeros(), 0L);
        Tensor firstWeight = parameter(Shape.of(6, 2, 3, 5), DataType.FLOAT32, true);
        Tensor firstBias = parameter(Shape.of(6), DataType.FLOAT32, true);
        layer.loadStateDictionary(dictionary(
                entry("weight", StateKind.PARAMETER, firstWeight),
                entry("bias", StateKind.PARAMETER, firstBias)));
        var weightWrapper = layer.weight();
        var biasWrapper = layer.bias().orElseThrow();
        Tensor nextWeight = parameter(Shape.of(6, 2, 3, 5), DataType.FLOAT32, true);
        Tensor wrongBias = parameter(Shape.of(7), DataType.FLOAT32, true);

        assertThrows(IllegalArgumentException.class, () -> layer.loadStateDictionary(dictionary(
                entry("weight", StateKind.PARAMETER, nextWeight),
                entry("bias", StateKind.PARAMETER, wrongBias))));
        assertAll(
                () -> assertSame(weightWrapper, layer.weight()),
                () -> assertSame(biasWrapper, layer.bias().orElseThrow()),
                () -> assertSame(firstWeight, layer.weight().value()),
                () -> assertSame(firstBias, layer.bias().orElseThrow().value()));

        Tensor nextBias = parameter(Shape.of(6), DataType.FLOAT32, true);
        layer.loadStateDictionary(dictionary(
                entry("bias", StateKind.PARAMETER, nextBias),
                entry("weight", StateKind.PARAMETER, nextWeight)));
        assertAll(
                () -> assertSame(weightWrapper, layer.weight()),
                () -> assertSame(biasWrapper, layer.bias().orElseThrow()),
                () -> assertSame(nextWeight, layer.weight().value()),
                () -> assertSame(nextBias, layer.bias().orElseThrow().value()));
    }

    private static void assertRejectedUnbound(Tensor weight, String messagePart) {
        Conv3d layer = conv3d(false);
        RuntimeException failure = assertThrows(RuntimeException.class,
                () -> layer.loadStateDictionary(
                        dictionary(entry("weight", StateKind.PARAMETER, weight))));
        assertAll(
                () -> assertTrue(failure.getMessage().contains(messagePart), failure::getMessage),
                () -> assertUnbound(layer));
    }

    private static Conv2d conv2d(
            boolean bias, ParameterInitialization initialization, long seed) {
        return new Conv2d(6, 3, 5, 1, 1, 0, 0, 1, 1, 2, bias,
                DataType.FLOAT32, initialization, seed);
    }

    private static Conv3d conv3d(boolean bias) {
        return new Conv3d(4, 2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, 2,
                bias, DataType.FLOAT64, ParameterInitialization.ones(), 0L);
    }

    private static Tensor parameter(Shape shape, DataType type, boolean requiresGrad) {
        return TensorFactory.create(new TensorDescriptor(type, shape, Optional.empty(), requiresGrad));
    }

    private static StateEntry entry(String path, StateKind kind, Tensor value) {
        return new StateEntry(path, kind, value);
    }

    private static StateDictionary dictionary(StateEntry... entries) {
        return new StateDictionary(List.of(entries));
    }

    private static void assertUnbound(Conv2d layer) {
        assertThrows(IllegalStateException.class, layer::weight);
        assertThrows(IllegalStateException.class, layer::stateDictionary);
    }

    private static void assertUnbound(Conv3d layer) {
        assertThrows(IllegalStateException.class, layer::weight);
        assertThrows(IllegalStateException.class, layer::stateDictionary);
    }

    private static AtomicLong nextTensorIdState() throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }
}

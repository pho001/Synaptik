package io.github.pho001.synaptik.nn.layers;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentDirection;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.LstmRecurrentScanResult;
import io.github.pho001.synaptik.model.tensor.RecurrentScanResult;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.initialization.ParameterInitialization;
import io.github.pho001.synaptik.nn.initialization.ParameterInitializers;
import io.github.pho001.synaptik.nn.module.Module;
import io.github.pho001.synaptik.nn.module.Parameter;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.MethodSource;

/** Descriptor-only NN scan composition; execution belongs to public Engine integration tests. */
@Execution(ExecutionMode.SAME_THREAD)
class RuntimeLengthRecurrentSequenceTest {
    private enum Binding { SUPPLIED, EAGER, AUTOMATIC, STRICT_LOADED }

    static Stream<Arguments> variants() {
        return Stream.of(RecurrentScanKind.values()).flatMap(kind -> Stream.of(false, true)
                .flatMap(bias -> Stream.of(RecurrentDirection.values()).flatMap(direction ->
                        Stream.of(Binding.values()).map(binding ->
                                Arguments.of(kind, bias, direction, binding)))));
    }

    @ParameterizedTest
    @MethodSource("variants")
    void exactSingleProducerPortsModesAndReplacement(
            RecurrentScanKind kind, boolean bias, RecurrentDirection direction, Binding binding) {
        Module sequence = sequence(kind, bias, DataType.FLOAT32, binding);
        Tensor input = tensor(DataType.FLOAT32, Shape.of(7, 2, 4), false).neg();
        Tensor lengths = tensor(DataType.FLOAT32, Shape.of(2), false).cast(DataType.INT64);
        Tensor hidden = tensor(DataType.FLOAT32, Shape.of(2, 3), false).neg();
        Tensor cell = tensor(DataType.FLOAT32, Shape.of(2, 3), true).neg();
        List<Tensor> outputs = scan(sequence, input, lengths, hidden, cell, direction);
        List<Parameter> parameters = sequence.children().get("cell").parameters();
        var expected = new ArrayList<>(List.of(input, lengths, hidden));
        if (kind == RecurrentScanKind.LSTM) expected.add(cell);
        parameters.forEach(parameter -> expected.add(parameter.value()));
        assertOccurrence(kind, direction, outputs, expected, Shape.of(7, 2, 3), Shape.of(2, 3));
        assertEquals(bias ? List.of("cell.inputWeight", "cell.hiddenWeight", "cell.bias")
                        : List.of("cell.inputWeight", "cell.hiddenWeight"),
                new ArrayList<>(sequence.parametersRecursively().keySet()));

        sequence.eval();
        List<Tensor> evaluation = scan(sequence, input, lengths, hidden, cell, direction);
        sequence.train();
        List<Tensor> training = scan(sequence, input, lengths, hidden, cell, direction);
        assertEquals(evaluation.getFirst().descriptor(), training.getFirst().descriptor());
        assertEquals(evaluation.getFirst().provenance().orElseThrow().operation(),
                training.getFirst().provenance().orElseThrow().operation());
        assertEquals(expected, evaluation.getFirst().provenance().orElseThrow().inputs());
        assertEquals(expected, training.getFirst().provenance().orElseThrow().inputs());

        for (Parameter parameter : parameters) {
            Tensor old = parameter.value();
            parameter.replace(old.neg()); // ordinary produced parameter, not a detached leaf
        }
        List<Tensor> later = scan(sequence, input, lengths, hidden, cell, direction);
        assertEquals(expected, outputs.getFirst().provenance().orElseThrow().inputs());
        int parameterStart = kind == RecurrentScanKind.LSTM ? 4 : 3;
        for (int index = 0; index < parameters.size(); index++) {
            assertSame(parameters.get(index).value(), later.getFirst().provenance().orElseThrow()
                    .inputs().get(parameterStart + index));
            assertNotSame(expected.get(parameterStart + index), parameters.get(index).value());
        }
    }

    @ParameterizedTest
    @EnumSource(RecurrentScanKind.class)
    void allFloatingTypesAreExpressionValidWithoutBackendGating(RecurrentScanKind kind) {
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32,
                DataType.BFLOAT16, DataType.FLOAT16)) {
            Module sequence = sequence(kind, true, type, Binding.AUTOMATIC);
            List<Tensor> outputs = scan(sequence,
                    tensor(type, Shape.of(2, 1, 4), false),
                    tensor(DataType.INT64, Shape.of(1), false),
                    tensor(type, Shape.of(1, 3), false),
                    tensor(type, Shape.of(1, 3), false), RecurrentDirection.REVERSE);
            outputs.forEach(value -> {
                assertSame(type, value.descriptor().dataType());
                assertTrue(value.descriptor().requiresGrad());
            });
        }
    }

    @ParameterizedTest
    @EnumSource(RecurrentScanKind.class)
    void invalidDescriptorsAndNullsHaveNoInitializationOrIdentifierEffects(RecurrentScanKind kind)
            throws Exception {
        Module sequence = sequence(kind, true, DataType.FLOAT32, Binding.AUTOMATIC);
        Tensor x = tensor(DataType.FLOAT32, Shape.of(2, 2, 4), false);
        Tensor l = tensor(DataType.INT64, Shape.of(2), false);
        Tensor h = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        Tensor c = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        var invalidInputs = List.of(
                tensor(DataType.INT32, Shape.of(2, 2, 4), false),
                tensor(DataType.FLOAT64, Shape.of(2, 2, 4), false),
                tensor(DataType.FLOAT16, Shape.of(2, 2, 4), false),
                tensor(DataType.FLOAT32, Shape.of(2, 4), false),
                tensor(DataType.FLOAT32, Shape.of(2, 2, 0), false),
                dynamicInput(0), dynamicInput(1), dynamicInput(2));
        var invalidLengths = List.of(
                tensor(DataType.INT32, Shape.of(2), false),
                tensor(DataType.INT64, Shape.of(2, 1), false),
                tensor(DataType.INT64, Shape.of(1), false),
                tensor(DataType.INT64, Shape.ofDimensions(new DynamicDimension("B")), false));
        var invalidStates = List.of(
                tensor(DataType.INT64, Shape.of(2, 3), false),
                tensor(DataType.FLOAT64, Shape.of(2, 3), false),
                tensor(DataType.BFLOAT16, Shape.of(2, 3), false),
                tensor(DataType.FLOAT32, Shape.of(3), false),
                tensor(DataType.FLOAT32, Shape.of(1, 3), false),
                tensor(DataType.FLOAT32, Shape.of(2, 0), false),
                tensor(DataType.FLOAT32, Shape.of(2, 4), false),
                tensor(DataType.FLOAT32, Shape.ofDimensions(
                        new DynamicDimension("B"), new StaticDimension(3)), false),
                tensor(DataType.FLOAT32, Shape.ofDimensions(
                        new StaticDimension(2), new DynamicDimension("H")), false));
        AtomicLong ids = ids();
        long before = ids.get();
        // Model rejects integral gradient eligibility before a Tensor exists; do not bypass it.
        assertThrows(IllegalArgumentException.class,
                () -> tensor(DataType.INT64, Shape.of(2), true));
        assertEquals("input", assertThrows(NullPointerException.class,
                () -> scan(sequence, null, null, null, null, null)).getMessage());
        assertEquals("validLengths", assertThrows(NullPointerException.class,
                () -> scan(sequence, invalidInputs.getFirst(), null, null, null, null)).getMessage());
        assertEquals("initialHidden", assertThrows(NullPointerException.class,
                () -> scan(sequence, x, l, null, null, null)).getMessage());
        if (kind == RecurrentScanKind.LSTM) {
            assertEquals("initialCell", assertThrows(NullPointerException.class,
                    () -> scan(sequence, x, l, h, null, null)).getMessage());
        }
        assertEquals("direction", assertThrows(NullPointerException.class,
                () -> scan(sequence, x, l, h, c, null)).getMessage());
        for (Tensor invalid : invalidInputs) assertThrows(IllegalArgumentException.class,
                () -> scan(sequence, invalid, l, h, c, RecurrentDirection.FORWARD));
        for (Tensor invalid : invalidLengths) assertThrows(IllegalArgumentException.class,
                () -> scan(sequence, x, invalid, h, c, RecurrentDirection.FORWARD));
        for (Tensor invalid : invalidStates) {
            assertThrows(IllegalArgumentException.class,
                    () -> scan(sequence, x, l, invalid, c, RecurrentDirection.FORWARD));
            if (kind == RecurrentScanKind.LSTM) assertThrows(IllegalArgumentException.class,
                    () -> scan(sequence, x, l, h, invalid, RecurrentDirection.FORWARD));
        }
        assertEquals(before, ids.get());
        assertThrows(IllegalStateException.class, () -> sequence.children().get("cell").parameters());
        scan(sequence, x, l, h, c, RecurrentDirection.FORWARD); // same reservations remain usable
        Tensor wrongWidth = tensor(DataType.FLOAT32, Shape.of(2, 2, 5), false);
        long boundBefore = ids.get();
        assertThrows(IllegalArgumentException.class,
                () -> scan(sequence, wrongWidth, l, h, c, RecurrentDirection.FORWARD));
        assertEquals(boundBefore, ids.get());
    }

    @ParameterizedTest
    @EnumSource(RecurrentScanKind.class)
    void prospectiveInitializationCountLimitsPrecedeAnyTensor(RecurrentScanKind kind) throws Exception {
        Module sequence = sequence(kind, true, DataType.FLOAT32, Binding.AUTOMATIC);
        Tensor l = tensor(DataType.INT64, Shape.of(1), false);
        Tensor h = tensor(DataType.FLOAT32, Shape.of(1, 3), false);
        Tensor overArray = tensor(DataType.FLOAT32, Shape.of(0, 1, Integer.MAX_VALUE), false);
        Tensor overflow = tensor(DataType.FLOAT32, Shape.of(0, 1, Long.MAX_VALUE), false);
        long before = ids().get();
        assertThrows(IllegalArgumentException.class,
                () -> scan(sequence, overArray, l, h, h, RecurrentDirection.FORWARD));
        assertThrows(ArithmeticException.class,
                () -> scan(sequence, overflow, l, h, h, RecurrentDirection.FORWARD));
        assertEquals(before, ids().get());
        assertThrows(IllegalStateException.class, sequence::stateDictionary);
    }

    @ParameterizedTest
    @EnumSource(RecurrentScanKind.class)
    void emptyAndZeroLengthScansBindWithoutDummyTensorIdentities(RecurrentScanKind kind) throws Exception {
        for (long[] extents : List.of(new long[] {0, 2}, new long[] {4, 0}, new long[] {3, 2})) {
            Module sequence = sequence(kind, true, DataType.FLOAT32, Binding.AUTOMATIC);
            long time = extents[0], batch = extents[1];
            Tensor x = tensor(DataType.FLOAT32, Shape.of(time, batch, 4), false);
            Tensor l = TensorFactory.zeros(Shape.of(batch), DataType.INT64, Optional.empty(), false);
            Tensor h = tensor(DataType.FLOAT32, Shape.of(batch, 3), false);
            long before = ids().get();
            List<Tensor> outputs = scan(sequence, x, l, h, h, RecurrentDirection.REVERSE);
            List<Parameter> parameters = sequence.children().get("cell").parameters();
            assertEquals(3, parameters.size());
            for (int index = 0; index < parameters.size(); index++) {
                assertEquals(before + index, parameters.get(index).value().id().value());
            }
            for (int index = 0; index < outputs.size(); index++) {
                assertEquals(before + 3 + index, outputs.get(index).id().value());
            }
            assertEquals(before + 3 + outputs.size(), ids().get());
            assertEquals(Shape.of(time, batch, 3), outputs.getFirst().descriptor().shape());
            assertNotSame(h, outputs.get(1));
            assertEquals(Shape.of(batch, 3), outputs.get(1).descriptor().shape());
        }
    }

    @ParameterizedTest
    @EnumSource(RecurrentScanKind.class)
    void scanShapeAndProducerCountDoNotGrowWithTimeOrLengthValues(RecurrentScanKind kind) throws Exception {
        Module sequence = sequence(kind, false, DataType.FLOAT32, Binding.SUPPLIED);
        // Metadata only: scans impose no static-unroll Java list/array limit on T or B.
        for (long time : List.of(1L, 100L, (long) Integer.MAX_VALUE + 1)) {
            long batch = (long) Integer.MAX_VALUE + 1;
            Tensor x = tensor(DataType.FLOAT32, Shape.of(time, batch, 4), false);
            Tensor l = tensor(DataType.INT64, Shape.of(batch), false);
            Tensor h = tensor(DataType.FLOAT32, Shape.of(batch, 3), false);
            long before = ids().get();
            List<Tensor> outputs = scan(sequence, x, l, h, h, RecurrentDirection.FORWARD);
            assertEquals(before + outputs.size(), ids().get());
            assertEquals(Shape.of(time, batch, 3), outputs.getFirst().descriptor().shape());
            var producer = outputs.getFirst().provenance().orElseThrow().producer();
            outputs.forEach(value -> assertSame(producer, value.provenance().orElseThrow().producer()));
        }
    }

    @ParameterizedTest
    @EnumSource(RecurrentScanKind.class)
    void randomScanInitializationMatchesInputThenHiddenStreamAndZeroBias(RecurrentScanKind kind) {
        for (ParameterInitialization policy : List.of(ParameterInitialization.glorotUniform(),
                ParameterInitialization.kaimingReluNormal(), ParameterInitialization.ones(),
                ParameterInitialization.zeros())) {
            Module sequence = automatic(kind, true, DataType.FLOAT32, policy);
            Tensor x = tensor(DataType.FLOAT32, Shape.of(0, 2, 4), false);
            Tensor l = tensor(DataType.INT64, Shape.of(2), false);
            Tensor h = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
            scan(sequence, x, l, h, h, RecurrentDirection.FORWARD);
            RandomGenerator source = random();
            Tensor expectedInput = initialize(Shape.of(gates(kind) * 3, 4), policy, source);
            Tensor expectedHidden = initialize(Shape.of(gates(kind) * 3, 3), policy, source);
            List<Parameter> parameters = sequence.children().get("cell").parameters();
            assertArrayEquals(values(expectedInput), values(parameters.get(0).value()));
            assertArrayEquals(values(expectedHidden), values(parameters.get(1).value()));
            assertArrayEquals(new float[gates(kind) * 3], values(parameters.get(2).value()));
        }
    }

    @ParameterizedTest
    @EnumSource(RecurrentScanKind.class)
    void failedInitializationLeavesGroupUnboundAndRetryRestartsSeed(RecurrentScanKind kind) throws Exception {
        Module sequence = automatic(kind, true, DataType.FLOAT32, ParameterInitialization.glorotUniform());
        Tensor x = tensor(DataType.FLOAT32, Shape.of(0, 2, 4), false);
        Tensor l = tensor(DataType.INT64, Shape.of(2), false);
        Tensor h = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        AtomicLong ids = ids();
        AtomicBoolean maximumClaimed = maximumClaimed();
        long saved = ids.get();
        boolean savedClaimed = maximumClaimed.get();
        try {
            ids.set(Long.MAX_VALUE);
            maximumClaimed.set(false);
            assertThrows(IllegalStateException.class,
                    () -> scan(sequence, x, l, h, h, RecurrentDirection.FORWARD));
            assertThrows(IllegalStateException.class, sequence::stateDictionary);
            assertTrue(maximumClaimed.get()); // input Tensor was created, hidden allocation failed
        } finally {
            ids.set(saved);
            maximumClaimed.set(savedClaimed);
        }
        scan(sequence, x, l, h, h, RecurrentDirection.FORWARD);
        RandomGenerator source = random();
        List<Parameter> parameters = sequence.children().get("cell").parameters();
        assertArrayEquals(values(initialize(Shape.of(gates(kind) * 3, 4),
                ParameterInitialization.glorotUniform(), source)), values(parameters.get(0).value()));
        assertArrayEquals(values(initialize(Shape.of(gates(kind) * 3, 3),
                ParameterInitialization.glorotUniform(), source)), values(parameters.get(1).value()));
    }

    @ParameterizedTest
    @EnumSource(RecurrentScanKind.class)
    void laterScanAllocationFailureKeepsPublishedParameters(RecurrentScanKind kind) throws Exception {
        Module sequence = sequence(kind, true, DataType.FLOAT32, Binding.AUTOMATIC);
        Tensor x = tensor(DataType.FLOAT32, Shape.of(0, 2, 4), false);
        Tensor l = tensor(DataType.INT64, Shape.of(2), false);
        Tensor h = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        AtomicLong ids = ids();
        AtomicBoolean maximumClaimed = maximumClaimed();
        long saved = ids.get();
        boolean savedClaimed = maximumClaimed.get();
        List<Parameter> published;
        try {
            ids.set(Long.MAX_VALUE - 2);
            maximumClaimed.set(false);
            assertThrows(IllegalStateException.class,
                    () -> scan(sequence, x, l, h, h, RecurrentDirection.FORWARD));
            published = sequence.children().get("cell").parameters();
            assertEquals(3, published.size());
        } finally {
            ids.set(saved);
            maximumClaimed.set(savedClaimed);
        }
        scan(sequence, x, l, h, h, RecurrentDirection.FORWARD);
        assertEquals(published, sequence.children().get("cell").parameters());
    }

    @ParameterizedTest
    @EnumSource(RecurrentScanKind.class)
    void compatibleConcurrentFirstScanAndForwardShareOneAtomicGroup(RecurrentScanKind kind) throws Exception {
        Module sequence = sequence(kind, true, DataType.FLOAT32, Binding.AUTOMATIC);
        Tensor x = tensor(DataType.FLOAT32, Shape.of(2, 2, 4), false);
        Tensor step = tensor(DataType.FLOAT32, Shape.of(2, 4), false);
        Tensor l = tensor(DataType.INT64, Shape.of(2), false);
        Tensor h = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        CountDownLatch ready = new CountDownLatch(3), start = new CountDownLatch(1);
        Module cell = sequence.children().get("cell");
        try (var executor = Executors.newFixedThreadPool(3)) {
            try {
                var scanA = executor.submit(() -> {
                    ready.countDown(); start.await();
                    return scan(sequence, x, l, h, h, RecurrentDirection.FORWARD);
                });
                var scanB = executor.submit(() -> {
                    ready.countDown(); start.await();
                    return scan(sequence, x, l, h, h, RecurrentDirection.REVERSE);
                });
                var ordinary = executor.submit(() -> {
                    ready.countDown(); start.await();
                    return switch (cell) {
                        case RnnCell rnn -> rnn.forward(step, h);
                        case GruCell gru -> gru.forward(step, h);
                        case LstmCell lstm -> lstm.forward(step, h, h).nextHidden();
                        default -> throw new AssertionError();
                    };
                });
                assertTrue(ready.await(10, TimeUnit.SECONDS));
                start.countDown();
                List<Tensor> a = scanA.get(10, TimeUnit.SECONDS), b = scanB.get(10, TimeUnit.SECONDS);
                Tensor result = ordinary.get(10, TimeUnit.SECONDS);
                var aInputs = a.getFirst().provenance().orElseThrow().inputs();
                var bInputs = b.getFirst().provenance().orElseThrow().inputs();
                int offset = kind == RecurrentScanKind.LSTM ? 4 : 3;
                for (int index = 0; index < 3; index++) {
                    Tensor parameter = cell.parameters().get(index).value();
                    assertSame(parameter, aInputs.get(offset + index));
                    assertSame(parameter, bInputs.get(offset + index));
                    assertTrue(reachable(result, parameter));
                }
            } finally {
                start.countDown();
            }
        }
    }

    @Test
    void publicScanSignaturesUseOnlyExistingModelCarriers() throws Exception {
        assertSame(RecurrentScanResult.class, RnnSequence.class.getDeclaredMethod("forwardScan",
                Tensor.class, Tensor.class, Tensor.class, RecurrentDirection.class).getReturnType());
        assertSame(RecurrentScanResult.class, GruSequence.class.getDeclaredMethod("forwardScan",
                Tensor.class, Tensor.class, Tensor.class, RecurrentDirection.class).getReturnType());
        assertSame(LstmRecurrentScanResult.class, LstmSequence.class.getDeclaredMethod("forwardScan",
                Tensor.class, Tensor.class, Tensor.class, Tensor.class, RecurrentDirection.class).getReturnType());
    }

    private static void assertOccurrence(RecurrentScanKind kind, RecurrentDirection direction,
            List<Tensor> outputs, List<Tensor> expected, Shape dense, Shape state) {
        var producer = outputs.getFirst().provenance().orElseThrow().producer();
        assertSame(kind, producer.operation().kind());
        assertSame(direction, producer.operation().attrs());
        assertEquals(expected.size(), producer.inputs().size());
        for (int index = 0; index < expected.size(); index++) {
            assertSame(expected.get(index), producer.inputs().get(index));
        }
        for (int index = 0; index < outputs.size(); index++) {
            Tensor output = outputs.get(index);
            assertSame(producer, output.provenance().orElseThrow().producer());
            assertEquals(index, output.provenance().orElseThrow().outputIndex());
            assertSame(output, producer.output(index));
            assertTrue(output.descriptor().requiresGrad());
            assertTrue(output.descriptor().layout().isEmpty());
            assertEquals(index == 0 ? dense : state, output.descriptor().shape());
        }
    }

    private static List<Tensor> scan(Module sequence, Tensor x, Tensor l, Tensor h, Tensor c,
            RecurrentDirection direction) {
        return switch (sequence) {
            case RnnSequence rnn -> outputs(rnn.forwardScan(x, l, h, direction));
            case GruSequence gru -> outputs(gru.forwardScan(x, l, h, direction));
            case LstmSequence lstm -> {
                var result = lstm.forwardScan(x, l, h, c, direction);
                yield List.of(result.outputs(), result.finalHidden(), result.finalCell());
            }
            default -> throw new AssertionError();
        };
    }

    private static List<Tensor> outputs(RecurrentScanResult result) {
        return List.of(result.outputs(), result.finalHidden());
    }

    private static Module sequence(RecurrentScanKind kind, boolean bias, DataType type, Binding binding) {
        if (binding == Binding.AUTOMATIC) return automatic(kind, bias, type, ParameterInitialization.zeros());
        if (binding == Binding.STRICT_LOADED) {
            Module target = automatic(kind, bias, type, ParameterInitialization.glorotUniform());
            Module donor = sequence(kind, bias, type, Binding.SUPPLIED);
            target.loadStateDictionary(donor.stateDictionary());
            return target;
        }
        if (binding == Binding.EAGER) return switch (kind) {
            case RNN_TANH -> new RnnSequence(new RnnCell(4, 3, bias, type, random()));
            case GRU_RESET_AFTER -> new GruSequence(new GruCell(4, 3, bias, type, random()));
            case LSTM -> new LstmSequence(new LstmCell(4, 3, bias, type, random()));
        };
        Tensor w = tensor(type, Shape.of(gates(kind) * 3, 4), true).neg();
        Tensor u = tensor(type, Shape.of(gates(kind) * 3, 3), true).neg();
        Tensor b = tensor(type, Shape.of(gates(kind) * 3), true).neg();
        return switch (kind) {
            case RNN_TANH -> new RnnSequence(bias ? new RnnCell(w, u, b) : new RnnCell(w, u));
            case GRU_RESET_AFTER -> new GruSequence(bias ? new GruCell(w, u, b) : new GruCell(w, u));
            case LSTM -> new LstmSequence(bias ? new LstmCell(w, u, b) : new LstmCell(w, u));
        };
    }

    private static Module automatic(RecurrentScanKind kind, boolean bias, DataType type,
            ParameterInitialization policy) {
        return switch (kind) {
            case RNN_TANH -> new RnnSequence(3, bias, type, policy, 42);
            case GRU_RESET_AFTER -> new GruSequence(3, bias, type, policy, 42);
            case LSTM -> new LstmSequence(3, bias, type, policy, 42);
        };
    }

    private static int gates(RecurrentScanKind kind) {
        return switch (kind) { case RNN_TANH -> 1; case GRU_RESET_AFTER -> 3; case LSTM -> 4; };
    }

    private static Tensor tensor(DataType type, Shape shape, boolean grad) {
        return TensorFactory.create(new TensorDescriptor(type, shape, Optional.empty(), grad));
    }

    private static Tensor dynamicInput(int axis) {
        var dimensions = new io.github.pho001.synaptik.model.shape.Dimension[] {
                new StaticDimension(2), new StaticDimension(2), new StaticDimension(4)};
        dimensions[axis] = new DynamicDimension("unknown");
        return tensor(DataType.FLOAT32, Shape.ofDimensions(dimensions), false);
    }

    private static RandomGenerator random() {
        return RandomGeneratorFactory.<RandomGenerator>of("L64X128MixRandom").create(42);
    }

    private static Tensor initialize(Shape shape, ParameterInitialization policy, RandomGenerator source) {
        return policy.requiresRandomGenerator()
                ? ParameterInitializers.initialize(shape, DataType.FLOAT32, policy, source)
                : ParameterInitializers.initialize(shape, DataType.FLOAT32, policy);
    }

    private static float[] values(Tensor tensor) {
        return (float[]) tensor.hostStorage().orElseThrow().segment().heapBase().orElseThrow();
    }

    private static boolean reachable(Tensor root, Tensor target) {
        return root == target || root.provenance().map(provenance ->
                provenance.inputs().stream().anyMatch(input -> reachable(input, target))).orElse(false);
    }

    private static AtomicLong ids() throws Exception {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    private static AtomicBoolean maximumClaimed() throws Exception {
        Field field = TensorFactory.class.getDeclaredField("MAXIMUM_TENSOR_ID_CLAIMED");
        field.setAccessible(true);
        return (AtomicBoolean) field.get(null);
    }
}

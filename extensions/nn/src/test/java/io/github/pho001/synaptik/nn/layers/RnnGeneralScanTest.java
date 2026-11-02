package io.github.pho001.synaptik.nn.layers;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentDirection;
import io.github.pho001.synaptik.model.operation.scan.ScanAttrs;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.*;
import io.github.pho001.synaptik.nn.initialization.ParameterInitialization;
import io.github.pho001.synaptik.nn.initialization.ParameterInitializers;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.random.RandomGeneratorFactory;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/** Direct additive composition and the existing cell's initialization/publication contract. */
@Execution(ExecutionMode.SAME_THREAD)
final class RnnGeneralScanTest {
    @ParameterizedTest @EnumSource(value = DataType.class, names = {"FLOAT32", "FLOAT64"})
    void suppliedEagerAutomaticAndStrictLoadedBindingsKeepRolesAndSnapshots(DataType type) {
        for (boolean biased : List.of(false, true)) for (String binding : List.of("supplied", "eager", "automatic", "loaded"))
            for (RecurrentDirection direction : RecurrentDirection.values()) {
                RnnSequence sequence = sequence(type, biased, binding);
                Tensor x = formal(type, Shape.of(4, 1, 3), true).neg();
                Tensor h = formal(type, Shape.of(1, 2), false).neg();
                Tensor l = formal(DataType.INT64, Shape.of(1), false);
                var result = sequence.forwardGeneralScan(x, l, h, direction);
                var producer = result.outputs().provenance().orElseThrow().producer();
                assertSame(result.outputs(), producer.output(1));
                assertSame(result.finalHidden(), producer.output(0));
                assertEquals(Shape.of(4, 1, 2), result.outputs().descriptor().shape());
                assertEquals(Optional.of(LayoutDescriptor.contiguous(Shape.of(4, 1, 2))), result.outputs().descriptor().layout());
                assertEquals(Optional.of(LayoutDescriptor.contiguous(Shape.of(1, 2))), result.finalHidden().descriptor().layout());
                assertEquals(h, producer.inputs().get(0)); assertSame(x, producer.inputs().get(1));
                assertSame(l, producer.inputs().getLast());
                ScanAttrs attrs = (ScanAttrs) producer.operation().attrs();
                assertEquals(1, attrs.rowCarryCount()); assertEquals(List.of(OptionalInt.of(0)), attrs.sequenceRowAxes());
                assertEquals(direction == RecurrentDirection.REVERSE, attrs.reverse());
                var body = producer.scanBody().orElseThrow();
                assertSame(body.nextCarry().getFirst(), body.emissions().getFirst());
                assertFalse(body.carry().getFirst().descriptor().requiresGrad());
                assertTrue(body.step().getFirst().descriptor().requiresGrad());
                var formals = new ArrayList<>(body.carry());
                formals.addAll(body.step()); formals.addAll(body.invariants());
                for (Tensor f : formals) {
                    assertTrue(f.hostStorage().isEmpty()); assertTrue(f.provenance().isEmpty());
                    assertEquals(Optional.of(LayoutDescriptor.contiguous(f.descriptor().shape())), f.descriptor().layout());
                }
                var parameters = sequence.cell().parameters();
                for (int i = 0; i < parameters.size(); i++) assertSame(parameters.get(i).value(), producer.inputs().get(2 + i));
                var old = List.copyOf(producer.inputs());
                sequence.eval();
                assertEquals(result.outputs().descriptor(), sequence.forwardGeneralScan(x, l, h, direction).outputs().descriptor());
                for (var p : parameters) p.replace(p.value().neg());
                var later = sequence.forwardGeneralScan(x, l, h, direction).outputs().provenance().orElseThrow().producer();
                assertEquals(old, producer.inputs());
                for (int i = 0; i < parameters.size(); i++) assertSame(parameters.get(i).value(), later.inputs().get(2 + i));
                assertEquals(biased ? List.of("cell.inputWeight", "cell.hiddenWeight", "cell.bias")
                        : List.of("cell.inputWeight", "cell.hiddenWeight"), List.copyOf(sequence.parametersRecursively().keySet()));
                var fixed = sequence.forwardScan(x, l, h, direction);
                assertSame(fixed.outputs(), fixed.outputs().provenance().orElseThrow().producer().output(0));
                assertSame(fixed.finalHidden(), fixed.outputs().provenance().orElseThrow().producer().output(1));
                assertTrue(fixed.outputs().descriptor().layout().isEmpty());
            }
    }

    @Test void invalidDescriptorsDoNotInitializeOrConstructBodyAndRetryRemainsValid() throws Exception {
        var sequence = sequence(DataType.FLOAT32, true, "automatic");
        Tensor x = formal(DataType.FLOAT32, Shape.of(3, 2, 3), false);
        Tensor l = formal(DataType.INT64, Shape.of(2), false);
        Tensor h = formal(DataType.FLOAT32, Shape.of(2, 2), false);
        var badX = List.of(formal(DataType.FLOAT64, Shape.of(3, 2, 3), false),
                formal(DataType.FLOAT32, Shape.of(3, 2), false), formal(DataType.FLOAT32, Shape.of(3, 2, 0), false),
                formal(DataType.FLOAT32, Shape.ofDimensions(new DynamicDimension("T"), new StaticDimension(2), new StaticDimension(3)), false));
        var badL = List.of(formal(DataType.INT32, Shape.of(2), false), formal(DataType.INT64, Shape.of(1), false));
        var badH = List.of(formal(DataType.FLOAT64, Shape.of(2, 2), false), formal(DataType.FLOAT32, Shape.of(1, 2), false));
        long before = ids().get();
        assertEquals("input", assertThrows(NullPointerException.class, () -> sequence.forwardGeneralScan(null, null, null, null)).getMessage());
        assertEquals("validLengths", assertThrows(NullPointerException.class, () -> sequence.forwardGeneralScan(x, null, null, null)).getMessage());
        assertEquals("initialHidden", assertThrows(NullPointerException.class, () -> sequence.forwardGeneralScan(x, l, null, null)).getMessage());
        assertEquals("direction", assertThrows(NullPointerException.class, () -> sequence.forwardGeneralScan(x, l, h, null)).getMessage());
        for (Tensor bad : badX) assertThrows(IllegalArgumentException.class, () -> sequence.forwardGeneralScan(bad, l, h, RecurrentDirection.FORWARD));
        for (Tensor bad : badL) assertThrows(IllegalArgumentException.class, () -> sequence.forwardGeneralScan(x, bad, h, RecurrentDirection.FORWARD));
        for (Tensor bad : badH) assertThrows(IllegalArgumentException.class, () -> sequence.forwardGeneralScan(x, l, bad, RecurrentDirection.FORWARD));
        assertEquals(before, ids().get()); assertThrows(IllegalStateException.class, sequence::stateDictionary);
        sequence.forwardGeneralScan(x, l, h, RecurrentDirection.FORWARD);
        Tensor wrongWidth = formal(DataType.FLOAT32, Shape.of(3, 2, 4), false);
        before = ids().get();
        assertThrows(IllegalArgumentException.class, () -> sequence.forwardGeneralScan(wrongWidth, l, h, RecurrentDirection.FORWARD));
        assertEquals(before, ids().get());
        for (DataType low : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
            var lowSequence = sequence(low, true, "automatic");
            Tensor lowX = formal(low, Shape.of(3, 2, 3), false), lowH = formal(low, Shape.of(2, 2), false);
            long lowBefore = ids().get();
            assertThrows(IllegalArgumentException.class, () -> lowSequence.forwardGeneralScan(lowX, l, lowH, RecurrentDirection.FORWARD));
            assertEquals(lowBefore, ids().get()); assertThrows(IllegalStateException.class, lowSequence::stateDictionary);
        }
    }

    @ParameterizedTest @EnumSource(value = DataType.class, names = {"FLOAT32", "FLOAT64"})
    void emptyDescriptorsInitializeWithTheSameSeedBytesAndBoundCellIsReusable(DataType type) {
        for (long[] geometry : List.of(new long[]{0, 2}, new long[]{3, 0}, new long[]{0, 0})) {
            var sequence = sequence(type, true, "automatic");
            var result = sequence.forwardGeneralScan(formal(type, Shape.of(geometry[0], geometry[1], 3), false),
                    formal(DataType.INT64, Shape.of(geometry[1]), false), formal(type, Shape.of(geometry[1], 2), false), RecurrentDirection.REVERSE);
            assertEquals(Shape.of(geometry[0], geometry[1], 2), result.outputs().descriptor().shape());
            var random = RandomGeneratorFactory.of("L64X128MixRandom").create(42);
            assertArrayEquals(bits(ParameterInitializers.glorotUniform(Shape.of(2, 3), type, random)), bits(sequence.cell().inputWeight().value()));
            assertArrayEquals(bits(ParameterInitializers.glorotUniform(Shape.of(2, 2), type, random)), bits(sequence.cell().hiddenWeight().value()));
            assertArrayEquals(new byte[2 * type.byteWidth()], bits(sequence.cell().bias().orElseThrow().value()));
            var wrappers = sequence.cell().parameters();
            sequence.cell().forward(formal(type, Shape.of(1, 3), false), formal(type, Shape.of(1, 2), false));
            assertEquals(wrappers, sequence.cell().parameters());
        }
    }

    @Test void sharedParameterActualsRemainDistinctInvariantPositions() {
        Tensor shared = formal(DataType.FLOAT32, Shape.of(2, 2), true);
        var sequence = new RnnSequence(new RnnCell(shared, shared));
        var result = sequence.forwardGeneralScan(formal(DataType.FLOAT32, Shape.of(2, 1, 2), false),
                formal(DataType.INT64, Shape.of(1), false), formal(DataType.FLOAT32, Shape.of(1, 2), false), RecurrentDirection.FORWARD);
        var p = result.outputs().provenance().orElseThrow().producer();
        assertSame(shared, p.inputs().get(2)); assertSame(shared, p.inputs().get(3));
        assertNotSame(p.scanBody().orElseThrow().invariants().get(0), p.scanBody().orElseThrow().invariants().get(1));
    }

    @Test void concurrentGeneralFixedAndOrdinaryFirstCallsPublishOneGroup() throws Exception {
        var sequence = sequence(DataType.FLOAT32, true, "automatic");
        Tensor x = formal(DataType.FLOAT32, Shape.of(2, 1, 3), false), h = formal(DataType.FLOAT32, Shape.of(1, 2), false);
        Tensor l = formal(DataType.INT64, Shape.of(1), false), step = formal(DataType.FLOAT32, Shape.of(1, 3), false);
        CountDownLatch ready = new CountDownLatch(3), start = new CountDownLatch(1);
        try (var pool = Executors.newFixedThreadPool(3)) {
            try {
                var general = pool.submit(() -> { ready.countDown(); start.await(); return sequence.forwardGeneralScan(x, l, h, RecurrentDirection.REVERSE); });
                var fixed = pool.submit(() -> { ready.countDown(); start.await(); return sequence.forwardScan(x, l, h, RecurrentDirection.FORWARD); });
                var ordinary = pool.submit(() -> { ready.countDown(); start.await(); return sequence.cell().forward(step, h); });
                assertTrue(ready.await(10, TimeUnit.SECONDS)); start.countDown();
                var g = general.get(10, TimeUnit.SECONDS).outputs().provenance().orElseThrow().inputs();
                var f = fixed.get(10, TimeUnit.SECONDS).outputs().provenance().orElseThrow().inputs();
                Tensor o = ordinary.get(10, TimeUnit.SECONDS);
                for (int i = 0; i < 3; i++) {
                    Tensor parameter = sequence.cell().parameters().get(i).value();
                    assertSame(parameter, g.get(2 + i)); assertSame(parameter, f.get(3 + i));
                    assertTrue(reachable(o, parameter));
                }
            } finally { start.countDown(); }
        }
    }

    @Test void prospectiveParameterLimitsRejectBeforeInitializationEvenAtZeroTime() throws Exception {
        var sequence = sequence(DataType.FLOAT32, true, "automatic");
        Tensor l = formal(DataType.INT64, Shape.of(1), false), h = formal(DataType.FLOAT32, Shape.of(1, 2), false);
        Tensor overArray = formal(DataType.FLOAT32, Shape.of(0, 1, Integer.MAX_VALUE), false);
        Tensor overflow = formal(DataType.FLOAT32, Shape.of(0, 1, Long.MAX_VALUE), false);
        long before = ids().get();
        assertThrows(IllegalArgumentException.class, () -> sequence.forwardGeneralScan(overArray, l, h, RecurrentDirection.FORWARD));
        assertThrows(ArithmeticException.class, () -> sequence.forwardGeneralScan(overflow, l, h, RecurrentDirection.FORWARD));
        assertEquals(before, ids().get()); assertThrows(IllegalStateException.class, sequence::stateDictionary);
    }

    @ParameterizedTest @EnumSource(value = DataType.class, names = {"FLOAT32", "FLOAT64"})
    void resultAndActualLatentPortOverflowPrecedeInitializationAndPermitDifferentWidthRetry(DataType type) throws Exception {
        // Emission overflow, then independent input/hidden transpose-history overflow.
        for (long[] geometry : List.of(
                new long[]{Long.MAX_VALUE, 1, 1},
                new long[]{Long.MAX_VALUE / 3, 1, 3},
                new long[]{Long.MAX_VALUE / 3, 1, 1},
                new long[]{Long.MAX_VALUE / 3, 0, 1})) {
            for (boolean bias : List.of(false, true)) for (RecurrentDirection direction : RecurrentDirection.values()) {
                var sequence = sequence(type, bias, "automatic");
                Tensor x = formal(type, Shape.of(geometry), false);
                Tensor l = formal(DataType.INT64, Shape.of(geometry[1]), false);
                Tensor h = formal(type, Shape.of(geometry[1], 2), false);
                long before = ids().get();
                assertThrows(ArithmeticException.class, () -> sequence.forwardGeneralScan(x, l, h, direction));
                assertEquals(before, ids().get());
                assertThrows(IllegalStateException.class, sequence::stateDictionary);
                Tensor valid = formal(type, Shape.of(2, geometry[1], 3), false);
                sequence.forwardGeneralScan(valid, l, h, direction);
                assertEquals(Shape.of(2, 3), sequence.cell().inputWeight().value().descriptor().shape());
            }
        }
    }

    @Test void stepAndActualInputCountsArePreflightedIndependentlyOfResultGeometry() throws Exception {
        for (long[] geometry : List.of(
                new long[]{0, Long.MAX_VALUE, 2},
                new long[]{2, Long.MAX_VALUE / 3, 3})) {
            var sequence = new RnnSequence(1, true, DataType.FLOAT32, ParameterInitialization.zeros(), 42);
            Tensor x = formal(DataType.FLOAT32, Shape.of(geometry), false);
            Tensor l = formal(DataType.INT64, Shape.of(geometry[1]), false);
            Tensor h = formal(DataType.FLOAT32, Shape.of(geometry[1], 1), false);
            long before = ids().get();
            assertThrows(ArithmeticException.class, () -> sequence.forwardGeneralScan(x, l, h, RecurrentDirection.FORWARD));
            assertEquals(before, ids().get());
            assertThrows(IllegalStateException.class, sequence::stateDictionary);
            sequence.forwardGeneralScan(formal(DataType.FLOAT32, Shape.of(1, 1, 3), false),
                    formal(DataType.INT64, Shape.of(1), false), formal(DataType.FLOAT32, Shape.of(1, 1), false),
                    RecurrentDirection.FORWARD);
            assertEquals(Shape.of(1, 3), sequence.cell().inputWeight().value().descriptor().shape());
        }
    }

    @Test void exactLatentGeometriesDoNotImposeAMaximumExtentProduct() {
        // All actual histories fit; multiplying unrelated maxima T*F*F would not.
        for (long[] geometry : List.of(
                new long[]{Long.MAX_VALUE / 7, 1, 3},
                new long[]{0, Long.MAX_VALUE / 3, 3},
                new long[]{Long.MAX_VALUE / 7, 0, 3})) {
            var sequence = sequence(DataType.FLOAT32, true, "automatic");
            var result = sequence.forwardGeneralScan(formal(DataType.FLOAT32, Shape.of(geometry), false),
                    formal(DataType.INT64, Shape.of(geometry[1]), false),
                    formal(DataType.FLOAT32, Shape.of(geometry[1], 2), false), RecurrentDirection.FORWARD);
            assertEquals(Shape.of(geometry[0], geometry[1], 2), result.outputs().descriptor().shape());
            assertEquals(Shape.of(2, 3), sequence.cell().inputWeight().value().descriptor().shape());
        }
    }

    @Test void failedInitializationRetriesAndLaterBodyFailureKeepsPublishedGroup() throws Exception {
        for (boolean failDuringInitialization : List.of(true, false)) {
            var sequence = sequence(DataType.FLOAT32, true, "automatic");
            Tensor x = formal(DataType.FLOAT32, Shape.of(0, 1, 3), false);
            Tensor l = formal(DataType.INT64, Shape.of(1), false), h = formal(DataType.FLOAT32, Shape.of(1, 2), false);
            var ids = ids();
            Field field = TensorFactory.class.getDeclaredField("MAXIMUM_TENSOR_ID_CLAIMED");
            field.setAccessible(true); var claimed = (AtomicBoolean) field.get(null);
            long saved = ids.get(); boolean savedClaimed = claimed.get();
            List<io.github.pho001.synaptik.nn.module.Parameter> published = null;
            try {
                ids.set(failDuringInitialization ? Long.MAX_VALUE : Long.MAX_VALUE - 2);
                claimed.set(false);
                assertThrows(IllegalStateException.class, () -> sequence.forwardGeneralScan(x, l, h, RecurrentDirection.FORWARD));
                if (failDuringInitialization) assertThrows(IllegalStateException.class, sequence::stateDictionary);
                else published = sequence.cell().parameters();
            } finally { ids.set(saved); claimed.set(savedClaimed); }
            sequence.forwardGeneralScan(x, l, h, RecurrentDirection.FORWARD);
            if (published != null) assertEquals(published, sequence.cell().parameters());
            var random = RandomGeneratorFactory.of("L64X128MixRandom").create(42);
            assertArrayEquals(bits(ParameterInitializers.glorotUniform(Shape.of(2, 3), DataType.FLOAT32, random)), bits(sequence.cell().inputWeight().value()));
            assertArrayEquals(bits(ParameterInitializers.glorotUniform(Shape.of(2, 2), DataType.FLOAT32, random)), bits(sequence.cell().hiddenWeight().value()));
        }
    }

    private static RnnSequence sequence(DataType type, boolean bias, String binding) {
        if (binding.equals("automatic")) return new RnnSequence(2, bias, type, ParameterInitialization.glorotUniform(), 42);
        if (binding.equals("eager")) return new RnnSequence(new RnnCell(3, 2, bias, type, RandomGeneratorFactory.of("L64X128MixRandom").create(42)));
        if (binding.equals("loaded")) {
            var target = sequence(type, bias, "automatic"); target.loadStateDictionary(sequence(type, bias, "supplied").stateDictionary()); return target;
        }
        Tensor w = formal(type, Shape.of(2, 3), true).neg(), u = formal(type, Shape.of(2, 2), true).neg();
        return new RnnSequence(bias ? new RnnCell(w, u, formal(type, Shape.of(2), true).neg()) : new RnnCell(w, u));
    }
    private static Tensor formal(DataType type, Shape shape, boolean grad) {
        return TensorFactory.create(new TensorDescriptor(type, shape, Optional.empty(), grad));
    }
    private static byte[] bits(Tensor tensor) { return tensor.hostStorage().orElseThrow().segment().toArray(ValueLayout.JAVA_BYTE); }
    private static boolean reachable(Tensor root, Tensor target) {
        return root == target || root.provenance().map(p -> p.inputs().stream().anyMatch(t -> reachable(t, target))).orElse(false);
    }
    private static AtomicLong ids() throws Exception {
        Field f = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID"); f.setAccessible(true); return (AtomicLong) f.get(null);
    }
}

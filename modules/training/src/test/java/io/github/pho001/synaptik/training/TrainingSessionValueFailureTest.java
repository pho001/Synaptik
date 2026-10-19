package io.github.pho001.synaptik.training;

import static io.github.pho001.synaptik.training.TrainingValueFixture.*;
import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.PreparedExecution;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

/** Fresh admission preserves rollback, result cleanup and deterministic lifecycle precedence. */
final class TrainingSessionValueFailureTest {
    /**
     * Proves failure cleanup, complete rollback and repaired retry with low-precision masters.
     * @throws Exception if private result state cannot be inspected
     */
    @Test
    void freshFailuresPreserveMastersAdaptiveSlotsPendingStateAndRetry() throws Exception {
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Tensor scalarPrototype = scalar(arena, 2, true);
            var lowDescriptor = new TensorDescriptor(DataType.FLOAT16,
                    scalarPrototype.descriptor().shape(), scalarPrototype.descriptor().layout(), true);
            var lowSegment = arena.allocate(2, 2);
            lowSegment.set(ValueLayout.JAVA_SHORT, 0, (short) 0x4000); // exactly 2
            Tensor low = TensorFactory.create(lowDescriptor, Optional.empty(), Optional.of(
                    new MemorySegmentStorage(DataType.FLOAT16, 1, lowSegment)));
            Tensor second = scalar(arena, -1, true);
            Tensor input = scalar(arena, 3, false);
            var source = new TestParameterSource();
            var binding = source.parameter("low", low);
            source.parameter("second", second);
            Tensor objective = low.cast(DataType.FLOAT32).mul(input).add(second);
            AtomicInteger fault = new AtomicInteger();
            RuntimeException injected = new IllegalStateException("injected fresh failure");
            TrainingSession.Hooks hooks = new TrainingSession.Hooks() {
                @Override void afterObjectiveMaterialization() {
                    if (fault.get() == 1) {
                        assertEquals(1, assertDoesNotThrow(() -> openResults(engine)));
                        throw injected;
                    }
                }
                @Override void afterExecution() {
                    assertEquals(0, assertDoesNotThrow(() -> openResults(engine)));
                    if (fault.get() == 2) binding.replace(second);
                }
                @Override void beforeParameterWrite(int index) {
                    if (fault.get() == 3 && index == 1) throw injected;
                }
            };
            try (TrainingSession session = TrainingSession.openForTesting(engine, source,
                    objective, new Adam(0.1, 0.5, 0.25, 0.01, 0), hooks)) {
                session.runValues(List.of(scalar(arena, 2, false)), GradientMode.ACCUMULATE);
                TrainingState before = session.state();
                assertEquals(1, before.accumulatedGradientRuns());
                assertThrows(ArithmeticException.class, () -> session.runValues(
                        List.of(scalar(arena, Float.NaN, false)), GradientMode.ACCUMULATE_AND_STEP));
                assertState(before, session.state());
                assertEquals(0, openResults(engine));
                fault.set(1);
                assertSame(injected, assertThrows(IllegalStateException.class, () -> session.runValues(
                        List.of(scalar(arena, 4, false)), GradientMode.ACCUMULATE_AND_STEP)));
                assertState(before, session.state());
                assertEquals(0, openResults(engine));
                fault.set(2);
                assertThrows(IllegalStateException.class, () -> session.runValues(
                        List.of(scalar(arena, 4, false)), GradientMode.ACCUMULATE_AND_STEP));
                binding.replace(low); // repair the deliberate live-owner violation before inspection
                assertState(before, session.state());
                fault.set(3);
                assertSame(injected, assertThrows(IllegalStateException.class, () -> session.runValues(
                        List.of(scalar(arena, 4, false)), GradientMode.ACCUMULATE_AND_STEP)));
                assertState(before, session.state());
                assertEquals(0, openResults(engine));
                fault.set(0);
                session.runValues(List.of(scalar(arena, 4, false)), GradientMode.ACCUMULATE_AND_STEP);
                TrainingState continued = session.state();
                assertEquals(2, continued.executions());
                assertEquals(1, continued.optimizerSteps());
                assertEquals(0, continued.accumulatedGradientRuns());
                assertNotEquals(before.parameters().getFirst().masterParameterBytes(),
                        continued.parameters().getFirst().masterParameterBytes());
                session.restore(before);
                session.runValues(List.of(scalar(arena, 4, false)), GradientMode.ACCUMULATE_AND_STEP);
                assertState(continued, session.state());
            }
        }
    }

    /**
     * Holds an admitted fresh call across close admission and observes cleanup and precedence.
     * @throws Exception if concurrent progress or private owner inspection fails
     */
    @Test
    void admittedFreshRunFinishesBeforeCloseAndLifecycleWinsInvalidArguments() throws Exception {
        CountDownLatch admitted = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        AtomicInteger closes = new AtomicInteger();
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard();
                var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            Tensor weight = scalar(arena, 2, true);
            Tensor prototype = scalar(arena, 0, false);
            Tensor fresh = scalar(arena, 3, false);
            var source = new TestParameterSource();
            source.parameter("weight", weight);
            TrainingSession.Hooks hooks = new TrainingSession.Hooks() {
                @Override void afterAdmission(TrainingSession.Operation operation) {
                    if (operation == TrainingSession.Operation.RUN) {
                        admitted.countDown();
                        await(release);
                    }
                }
                @Override void beforeOwnedResourceClose() { closes.incrementAndGet(); }
            };
            TrainingSession session = TrainingSession.openForTesting(engine, source,
                    weight.mul(prototype), new Sgd(0.1), hooks);
            PreparedExecution prepared = (PreparedExecution) field(session, "preparedExecution");
            Tensor seed = (Tensor) field(session, "cotangentSeed");
            try {
                var run = executor.submit(() -> session.runValues(List.of(fresh), GradientMode.RESET_AND_STEP));
                assertTrue(admitted.await(10, TimeUnit.SECONDS));
                assertEquals("training session already has an operation in flight", assertThrows(
                        IllegalStateException.class, () -> session.runValues(null, null)).getMessage());
                var closing = executor.submit(session::close);
                // Wait for the observable close admission gate, without timing-based sleeps.
                long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
                while (!session.isClosed() && System.nanoTime() < deadline) Thread.onSpinWait();
                assertTrue(session.isClosed());
                assertFalse(closing.isDone());
                assertEquals(0, closes.get());
                assertFalse(prepared.isClosed());
                assertTrue(seed.hostStorage().orElseThrow().isAlive());
                assertEquals("training session is closed", assertThrows(IllegalStateException.class,
                        () -> session.runValues(null, null)).getMessage());
                release.countDown();
                TrainingStep step = run.get(10, TimeUnit.SECONDS);
                closing.get(10, TimeUnit.SECONDS);
                assertEquals(6, step.objective().bytes().getFloat());
                assertEquals(1.7f, value(weight), 1e-6f);
                assertEquals(1, closes.get());
                assertEquals(0, openResults(engine));
                assertTrue(prepared.isClosed());
                assertFalse(seed.hostStorage().orElseThrow().isAlive());
                session.close();
                engine.close();
                assertEquals("training engine is closed", assertThrows(IllegalStateException.class,
                        () -> session.runValues(null, null)).getMessage());
                assertEquals(6, step.objective().bytes().getFloat());
            } finally {
                release.countDown();
                session.close();
            }
        }
    }

    /**
     * Waits for an explicit lifecycle release, bounding failed-test deadlock time.
     * @param latch deterministic release owned by the test thread
     * @throws AssertionError if release times out or waiting is interrupted
     */
    private static void await(CountDownLatch latch) {
        try {
            assertTrue(latch.await(10, TimeUnit.SECONDS));
        } catch (InterruptedException failure) {
            Thread.currentThread().interrupt();
            throw new AssertionError(failure);
        }
    }
}

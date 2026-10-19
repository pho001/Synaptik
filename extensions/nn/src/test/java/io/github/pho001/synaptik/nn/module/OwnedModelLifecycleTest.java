package io.github.pho001.synaptik.nn.module;

import static org.junit.jupiter.api.Assertions.*;
import static io.github.pho001.synaptik.nn.module.OwnedModelSourceTest.*;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.training.Sgd;
import java.lang.foreign.Arena;
import java.lang.reflect.Proxy;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Deterministic admission, invalidation, close wait/replay and failed setup cleanup. */
final class OwnedModelLifecycleTest {
    @TempDir Path directory;

    static Model<List<Tensor>, List<Tensor>> model() {
        Tensor x = prototype(), w = scalar(2, true);
        return Model.graph(List.of(x), List.of(w.mul(x)), List.of(parameter("weight", w)));
    }

    static void set(Object owner, String name, Object value) throws Exception {
        var field = owner.getClass().getDeclaredField(name);
        field.setAccessible(true); field.set(owner, value);
    }

    @Test void borrowedSelectionIdentityNullClosedCandidateAndFreeze() throws Exception {
        var model = model();
        try (Engine first = Engine.standard(); Engine second = Engine.standard(); Engine dead = Engine.standard()) {
            assertSame(model, model.withEngine(first));
            Object state = field(model, "execution");
            assertThrows(NullPointerException.class, () -> model.withEngine(null));
            assertSame(first, field(state, "engine"));
            dead.close();
            assertThrows(IllegalStateException.class, () -> model.withEngine(dead));
            assertSame(first, field(state, "engine"));
            assertSame(model, model.withEngine(second));
            assertFalse(first.isClosed());
            model.compute(scalar(3, false));
            assertThrows(IllegalStateException.class, () -> model.withEngine(null));
            assertThrows(IllegalStateException.class, () -> model.withEngine(second));
            second.close();
            assertTrue(assertThrows(IllegalStateException.class,
                    () -> model.compute((Tensor) null)).getMessage().contains("selected Engine"));
            model.close(); model.close();
            assertFalse(first.isClosed());
        }
    }

    @Test void invalidModeBindingOrStorageClosesCachedOwnersAndNeverRecaptures() throws Exception {
        for (String mutation : List.of("mode", "tensor", "storage")) {
            try (var model = model()) {
                model.compute(scalar(1, false));
                var state = field(model, "execution");
                var prepared = (io.github.pho001.synaptik.engine.PreparedExecution) field(state, "inference");
                Tensor owned = model.stateDictionary().entries().getFirst().value();
                switch (mutation) {
                    case "mode" -> model.train();
                    case "tensor" -> model.loadStateDictionary(new StateDictionary(
                            List.of(new StateEntry("weight", StateKind.PARAMETER, scalar(9, true)))));
                    default -> owned.replaceHostStorage(scalar(9, true).hostStorage().orElseThrow());
                }
                assertThrows(IllegalStateException.class, () -> model.compute(scalar(1, false)));
                assertTrue(prepared.isClosed());
                model.eval(); // restoring the mode does not make the invalid capability reusable
                assertThrows(IllegalStateException.class, () -> model.forward(List.of(prototype())));
                assertThrows(IllegalStateException.class, () -> model.save(directory.resolve(mutation)));
                assertThrows(IllegalStateException.class, () -> model.configureTraining(
                        prototype().descriptor(), OwnedModelTrainingTest::loss, new Sgd(0.1)));
            }
        }
    }

    @Test void busyPrecedesArgumentsAndCloseWaitsForAdmittedCallback() throws Exception {
        var model = model();
        CountDownLatch entered = new CountDownLatch(1), release = new CountDownLatch(1);
        try (var pool = Executors.newFixedThreadPool(3)) {
            var setup = pool.submit(() -> model.configureTraining(prototype().descriptor(), (p, t) -> {
                entered.countDown();
                try { assertTrue(release.await(10, TimeUnit.SECONDS)); }
                catch (InterruptedException failure) { throw new AssertionError(failure); }
                return OwnedModelTrainingTest.loss(p, t);
            }, new Sgd(0.1)));
            assertTrue(entered.await(10, TimeUnit.SECONDS));
            assertTrue(assertThrows(IllegalStateException.class,
                    () -> model.compute((Tensor) null)).getMessage().contains("busy"));
            assertTrue(assertThrows(IllegalStateException.class,
                    () -> model.save(null)).getMessage().contains("busy"));
            assertTrue(assertThrows(IllegalStateException.class,
                    () -> model.withEngine(null)).getMessage().contains("busy"));
            Object state = field(model, "execution");
            var closing = pool.submit(model::close);
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
            synchronized (state) {
                while (!(Boolean) field(state, "closing") && System.nanoTime() < deadline) state.wait(10);
            }
            assertTrue((Boolean) field(state, "closing"));
            assertFalse(closing.isDone());
            assertTrue(assertThrows(IllegalStateException.class,
                    () -> model.train((Tensor) null, null)).getMessage().contains("closing"));
            var concurrentClose = pool.submit(model::close);
            release.countDown();
            setup.get(10, TimeUnit.SECONDS);
            closing.get(10, TimeUnit.SECONDS);
            concurrentClose.get(10, TimeUnit.SECONDS);
            assertTrue((Boolean) field(state, "closed"));
            assertTrue(((Engine) field(state, "engine")).isClosed());
        } finally { release.countDown(); model.close(); }
    }

    @Test void closeAttemptsAllOwnersRetainsFailureAndConcurrentCallersReplay() throws Exception {
        var model = model();
        model.configureTraining(prototype().descriptor(), OwnedModelTrainingTest::loss, new Sgd(0.1));
        Object state = field(model, "execution");
        var session = (io.github.pho001.synaptik.training.TrainingSession) field(state, "session");
        Arena realArena = (Arena) field(state, "nativeArena");
        RuntimeException failure = new IllegalStateException("injected native close");
        RuntimeException earlier = new IllegalStateException("injected session seed close");
        Arena seed = (Arena) field(session, "seedArena");
        Arena failingSeed = (Arena) Proxy.newProxyInstance(Arena.class.getClassLoader(),
                new Class<?>[] { Arena.class }, (proxy, method, arguments) -> {
                    if (method.getName().equals("close")) { seed.close(); throw earlier; }
                    return method.invoke(seed, arguments);
                });
        set(session, "seedArena", failingSeed);
        CountDownLatch entered = new CountDownLatch(1), release = new CountDownLatch(1);
        Arena throwing = (Arena) Proxy.newProxyInstance(Arena.class.getClassLoader(),
                new Class<?>[] { Arena.class }, (proxy, method, arguments) -> {
                    if (method.getName().equals("close")) {
                        entered.countDown();
                        assertTrue(release.await(10, TimeUnit.SECONDS));
                        realArena.close(); throw failure;
                    }
                    return method.invoke(realArena, arguments);
                });
        set(state, "nativeArena", throwing);
        try (var pool = Executors.newFixedThreadPool(2)) {
            var one = pool.submit(() -> assertSame(earlier, assertThrows(IllegalStateException.class, model::close)));
            assertTrue(entered.await(10, TimeUnit.SECONDS));
            var two = pool.submit(() -> assertSame(earlier, assertThrows(IllegalStateException.class, model::close)));
            release.countDown();
            one.get(10, TimeUnit.SECONDS); two.get(10, TimeUnit.SECONDS);
        }
        assertTrue(session.isClosed());
        assertTrue(((Engine) field(state, "engine")).isClosed());
        assertFalse(realArena.scope().isAlive());
        assertArrayEquals(new Throwable[] {failure}, earlier.getSuppressed());
        assertSame(earlier, assertThrows(IllegalStateException.class, model::close));
    }

    @Test void failedPreparationRetainsNativeAdmissionAndFrozenEngineForRetry() throws Exception {
        try (var model = model()) {
            Object state = field(model, "execution");
            AtomicCounter callback = new AtomicCounter();
            assertThrows(RuntimeException.class, () -> model.configureTraining(
                    scalar(0, true).descriptor(), (p, t) -> { callback.value++; return t.mul(t); },
                    new Sgd(0.1)));
            Tensor weight = model.stateDictionary().entries().getFirst().value();
            Object admitted = weight.hostStorage().orElseThrow();
            assertTrue(weight.hostStorage().orElseThrow().segment().isNative());
            assertNull(field(state, "session"));
            assertThrows(IllegalStateException.class, () -> model.withEngine(null));
            model.configureTraining(prototype().descriptor(),
                    (p, t) -> { callback.value++; return OwnedModelTrainingTest.loss(p, t); }, new Sgd(0.1));
            assertSame(admitted, weight.hostStorage().orElseThrow());
            assertEquals(2, callback.value);
        }
    }

    @Test void borrowedEngineClosedFirstCannotPreventNativeResourceRelease() throws Exception {
        Engine borrowed = Engine.standard();
        var model = model().withEngine(borrowed);
        model.configureTraining(prototype().descriptor(), OwnedModelTrainingTest::loss, new Sgd(0.1));
        Object state = field(model, "execution");
        Arena arena = (Arena) field(state, "nativeArena");
        borrowed.close();
        assertThrows(IllegalStateException.class, () -> model.save(null));
        model.close(); model.close();
        assertFalse(arena.scope().isAlive());
    }

    private static final class AtomicCounter { int value; }
}

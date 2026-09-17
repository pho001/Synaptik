package io.github.pho001.synaptik.engine;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.runtime.execution.PreparedExecution;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

/** Exercises ordinary Engine ownership and delegated lifecycle behavior. */
final class EngineStandardCompositionTest {
    @Test
    void constructorTransfersTheExactNonNullOwner() {
        RecordingComposition composition = new RecordingComposition();
        AdvancedEngine owner = new AdvancedEngine(composition);
        Engine engine = new Engine(owner);

        assertFalse(engine.isClosed());
        engine.close();
        assertTrue(engine.isClosed());
        assertTrue(owner.isClosed());
        assertEquals(1, composition.closeCount.get());
    }

    @Test
    void nullConstructorArgumentTransfersNothing() {
        NullPointerException failure = assertThrows(NullPointerException.class,
                () -> new Engine(null));
        assertEquals("delegate", failure.getMessage());
        assertEquals(0, failure.getSuppressed().length);
    }

    @Test
    void independentWrappersOwnIndependentCompositions() {
        RecordingComposition firstComposition = new RecordingComposition();
        RecordingComposition secondComposition = new RecordingComposition();
        Engine first = new Engine(new AdvancedEngine(firstComposition));
        Engine second = new Engine(new AdvancedEngine(secondComposition));

        assertNotSame(first, second);
        first.close();
        assertTrue(first.isClosed());
        assertFalse(second.isClosed());
        assertEquals(1, firstComposition.closeCount.get());
        assertEquals(0, secondComposition.closeCount.get());

        second.close();
        assertEquals(1, secondComposition.closeCount.get());
    }

    @Test
    void repeatedAndConcurrentCloseRunsCleanupExactlyOnce() throws Exception {
        RecordingComposition composition = new RecordingComposition();
        Engine engine = new Engine(new AdvancedEngine(composition));

        try (var executor = Executors.newFixedThreadPool(4)) {
            var closes = List.of(
                    executor.submit(() -> { engine.close(); return null; }),
                    executor.submit(() -> { engine.close(); return null; }),
                    executor.submit(() -> { engine.close(); return null; }),
                    executor.submit(() -> { engine.close(); return null; }));
            for (var close : closes) {
                assertNull(close.get(10, TimeUnit.SECONDS));
            }
        }

        assertTrue(engine.isClosed());
        assertEquals(1, composition.closeCount.get());
        engine.close();
        assertEquals(1, composition.closeCount.get());
    }

    @Test
    void closePreservesExactCleanupFailureForEveryCaller() throws Exception {
        Error expected = new AssertionError("cleanup");
        RecordingComposition composition = new RecordingComposition();
        composition.closeFailure = expected;
        Engine engine = new Engine(new AdvancedEngine(composition));

        try (var executor = Executors.newFixedThreadPool(2)) {
            var first = executor.submit(() -> assertThrows(Error.class, engine::close));
            var second = executor.submit(() -> assertThrows(Error.class, engine::close));
            assertSame(expected, first.get(10, TimeUnit.SECONDS));
            assertSame(expected, second.get(10, TimeUnit.SECONDS));
        }

        assertEquals(1, composition.closeCount.get());
        assertSame(expected, assertThrows(Error.class, engine::close));
    }

    private static final class RecordingComposition implements EngineBackendComposition {
        private final AtomicInteger closeCount = new AtomicInteger();
        private Throwable closeFailure;

        @Override
        public List<BackendCapabilityProvider> capabilityProviders() {
            throw new AssertionError("unexpected capability request");
        }

        @Override
        public List<BackendAvailabilitySnapshot> availabilitySnapshots() {
            throw new AssertionError("unexpected availability request");
        }

        @Override
        public PreparedExecution prepare(CompileArtifacts artifacts) {
            throw new AssertionError("unexpected preparation");
        }

        @Override
        public BufferRepresentation borrow(HostTensorStorage storage) {
            throw new AssertionError("unexpected borrow");
        }

        @Override
        public byte[] copyToCanonicalHostBytes(
                BufferRepresentation representation,
                TensorDescriptor descriptor,
                long maximumBytes) {
            throw new AssertionError("unexpected copy");
        }

        @Override
        public void close() {
            closeCount.incrementAndGet();
            if (closeFailure instanceof RuntimeException runtimeFailure) {
                throw runtimeFailure;
            }
            if (closeFailure instanceof Error error) {
                throw error;
            }
        }
    }
}

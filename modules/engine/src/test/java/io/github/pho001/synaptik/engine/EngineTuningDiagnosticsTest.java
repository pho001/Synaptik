package io.github.pho001.synaptik.engine;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.config.tuning.ModelAutotuningConfig;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.trace.TraceEvent;
import io.github.pho001.synaptik.trace.TraceLevel;
import io.github.pho001.synaptik.trace.TracePhase;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.Proxy;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/** Exercises actual public absent-handoff fallback, passive observation, and exact rejection. */
final class EngineTuningDiagnosticsTest {
    @Test
    void actualAbsentCpuHandoffReportsFallbackAndPreparedRunPreservesValues() {
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Tensor input = input(arena);
            CompiledGraph graph = engine.compile(List.of(input.contiguous()));
            var events = new ArrayList<TraceEvent<EngineTuningDiagnostic>>();

            var preparation = engine.prepareTuned(graph, request(input, true), events::add);

            assertFallbackRuns(engine, input, preparation);
            assertEquals(1, events.size());
            var event = events.getFirst();
            assertEquals(0, event.id().value());
            assertEquals(TracePhase.PREPARE, event.phase());
            assertEquals(TraceLevel.WARN, event.level());
            assertEquals(new EngineTuningDiagnostic(
                    EngineTuningDiagnostic.Outcome.SAFE_HEURISTIC_FALLBACK,
                    EngineTuningDiagnostic.TuningPhase.LOCAL_WORKLOAD,
                    EngineTuningDiagnostic.Reason.NO_ELIGIBLE_HANDOFF), event.payload());
            // Running the returned owner does not emit per-run diagnostics.
            assertEquals(1, events.size());
        }
    }

    @ParameterizedTest
    @MethodSource("observerFailures")
    void observerFailureCannotChangeActualFallbackOrRun(Throwable callbackFailure) {
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Tensor input = input(arena);
            CompiledGraph graph = engine.compile(List.of(input.contiguous()));
            var events = new ArrayList<TraceEvent<EngineTuningDiagnostic>>();

            var preparation = engine.prepareTuned(graph, request(input, true), event -> {
                events.add(event);
                throwUnchecked(callbackFailure);
            });

            assertFallbackRuns(engine, input, preparation);
            assertEquals(1, events.size());
            assertEquals(EngineTuningDiagnostic.Reason.NO_ELIGIBLE_HANDOFF,
                    events.getFirst().payload().reason());
            assertEquals(0, callbackFailure.getSuppressed().length);
        }
    }

    @Test
    void actualAbsentCpuHandoffRequiredPolicyReportsRejection() {
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Tensor input = input(arena);
            CompiledGraph graph = engine.compile(List.of(input.contiguous()));
            var events = new ArrayList<TraceEvent<EngineTuningDiagnostic>>();

            var failure = assertThrows(IllegalStateException.class,
                    () -> engine.prepareTuned(graph, request(input, false), events::add));

            assertEquals("no eligible local-workload tuning handoff", failure.getMessage());
            assertEquals(0, failure.getSuppressed().length);
            assertEquals(1, events.size());
            assertEquals(TracePhase.PREPARE, events.getFirst().phase());
            assertEquals(TraceLevel.ERROR, events.getFirst().level());
            assertEquals(new EngineTuningDiagnostic(EngineTuningDiagnostic.Outcome.REJECTED,
                    EngineTuningDiagnostic.TuningPhase.LOCAL_WORKLOAD,
                    EngineTuningDiagnostic.Reason.NO_ELIGIBLE_HANDOFF),
                    events.getFirst().payload());
        }
    }

    @ParameterizedTest
    @MethodSource("observerFailures")
    void requiredPolicyRetainsExactTuningFailureDespiteObserverFailure(Throwable callbackFailure) {
        var tuningFailure = new IllegalArgumentException("sentinel tuning failure");
        var composition = new CpuEngineBackendComposition(CpuBackendIntegration.open());
        var tuning = (AdvancedEngine.ModelAutotuningTuning<?, ?, ?, ?, ?, ?>)
                Proxy.newProxyInstance(getClass().getClassLoader(),
                        new Class<?>[] {AdvancedEngine.ModelAutotuningTuning.class},
                        (proxy, method, arguments) -> {
                            if (method.getName().equals("candidateHandoff")) throw tuningFailure;
                            throw new AssertionError("unexpected collaboration: " + method);
                        });
        try (Arena arena = Arena.ofShared();
                Engine engine = new Engine(new AdvancedEngine(composition, tuning))) {
            Tensor input = input(arena);
            CompiledGraph graph = engine.compile(List.of(input.contiguous()));
            var events = new ArrayList<TraceEvent<EngineTuningDiagnostic>>();

            var observed = assertThrows(IllegalArgumentException.class,
                    () -> engine.prepareTuned(graph, request(input, false), event -> {
                        events.add(event);
                        throwUnchecked(callbackFailure);
                    }));

            assertSame(tuningFailure, observed);
            assertEquals(0, observed.getSuppressed().length);
            assertEquals(1, events.size());
            assertEquals(new EngineTuningDiagnostic(EngineTuningDiagnostic.Outcome.REJECTED,
                    EngineTuningDiagnostic.TuningPhase.LOCAL_WORKLOAD,
                    EngineTuningDiagnostic.Reason.OTHER_TUNING_FAILURE),
                    events.getFirst().payload());
            try (var prepared = engine.prepare(graph); var result = engine.run(prepared, List.of(input))) {
                assertEquals(1, result.resultCount());
            }
        }
    }

    @Test
    void terminalCallbackStillHoldsEngineAdmissionWithoutLifecycleLock() {
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Tensor input = input(arena);
            CompiledGraph graph = engine.compile(List.of(input.contiguous()));
            var selfCloseFailure = new AtomicReference<RuntimeException>();

            var preparation = engine.prepareTuned(graph, request(input, true), event -> {
                try {
                    engine.close();
                } catch (RuntimeException failure) {
                    selfCloseFailure.set(failure);
                }
            });

            assertNotNull(selfCloseFailure.get());
            assertTrue(selfCloseFailure.get() instanceof IllegalStateException);
            assertFalse(engine.isClosed());
            assertFallbackRuns(engine, input, preparation);
        }
    }

    @Test
    void observerNullnessFollowsCloseAdmissionAndDoesNotLeakOpenAdmission() {
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Tensor input = input(arena);
            CompiledGraph graph = engine.compile(List.of(input.contiguous()));
            assertThrows(NullPointerException.class,
                    () -> engine.prepareTuned(graph, request(input, true), null));
            // The deliberate no-observer overload retains its ordinary result semantics.
            assertFallbackRuns(engine, input, engine.prepareTuned(graph, request(input, true)));
            engine.close();
            assertThrows(IllegalStateException.class, () -> engine.prepareTuned(null, null, null));
        }
    }

    @Test
    void producerUsesBoundedClosedCategoriesAndDoesNotMislabelTrialIoAsCacheIo() {
        var events = new ArrayList<TraceEvent<EngineTuningDiagnostic>>();
        var cache = new EngineTuningTraceProducer(events::add);
        cache.cacheIoFailure();
        cache.fallbackSucceeded();
        cache.requiredTuningRejected();
        assertEquals(1, events.size());
        assertEquals(EngineTuningDiagnostic.Reason.CACHE_IO, events.getFirst().payload().reason());

        var trial = new EngineTuningTraceProducer(events::add);
        trial.trialIoFailure();
        trial.cacheIoFailure();
        trial.requiredTuningRejected();
        assertEquals(EngineTuningDiagnostic.Reason.OTHER_TUNING_FAILURE,
                events.getLast().payload().reason());

        var correctness = new EngineTuningTraceProducer(events::add);
        correctness.completePlanPhase();
        correctness.correctnessMismatch();
        correctness.fallbackSucceeded();
        assertEquals(EngineTuningDiagnostic.TuningPhase.COMPLETE_PLAN,
                events.getLast().payload().tuningPhase());
        assertEquals(EngineTuningDiagnostic.Reason.CORRECTNESS_MISMATCH,
                events.getLast().payload().reason());
    }

    private static Stream<Throwable> observerFailures() {
        return Stream.of(new IllegalStateException("observer runtime"), new AssertionError("observer error"));
    }

    private static void throwUnchecked(Throwable failure) {
        if (failure instanceof RuntimeException runtime) throw runtime;
        throw (Error) failure;
    }

    private static Tensor input(Arena arena) {
        Shape shape = Shape.of(2);
        var descriptor = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var segment = arena.allocate(8, Float.BYTES);
        segment.setAtIndex(ValueLayout.JAVA_FLOAT, 0, 3.25f);
        segment.setAtIndex(ValueLayout.JAVA_FLOAT, 1, -7.5f);
        return TensorFactory.create(descriptor, Optional.empty(), Optional.of(
                new MemorySegmentStorage(DataType.FLOAT32, 2, segment)));
    }

    private static ModelAutotuningRequest request(Tensor input, boolean fallbackAllowed) {
        var config = new ModelAutotuningConfig(
                ModelAutotuningConfig.Objective.MIN_MEDIAN_ELAPSED_NANOS,
                new ModelAutotuningConfig.Budget(1, 4, 0, 1),
                new ModelAutotuningConfig.RepresentativeProfileIdentity(1, new byte[] {2}),
                fallbackAllowed ? ModelAutotuningConfig.FallbackPolicy.ALLOW_SAFE_HEURISTIC
                        : ModelAutotuningConfig.FallbackPolicy.REQUIRE_TUNED_RESULT,
                Path.of("unused-native-free-autotuning-cache.bin"),
                new ModelAutotuningConfig.CompletePlanBudget(1, 0, 1, 1L, 0L),
                Path.of("unused-model-plan-cache.bin"));
        return new ModelAutotuningRequest(config,
                new ModelAutotuningRequest.ModelIdentity(1, new byte[] {1}), List.of(input));
    }

    private static void assertFallbackRuns(
            Engine engine, Tensor input, ModelAutotuningPreparation preparation) {
        assertEquals(ModelAutotuningPreparation.Outcome.SAFE_HEURISTIC_FALLBACK,
                preparation.outcome());
        assertEquals(Optional.empty(), preparation.evidence());
        try (var prepared = preparation.preparedExecution(); var result = engine.run(prepared, List.of(input))) {
            assertEquals(1, result.resultCount());
            var bytes = result.materialize(result.publications().getFirst(), 8).bytes();
            assertArrayEquals(new float[] {3.25f, -7.5f}, new float[] {bytes.getFloat(), bytes.getFloat()});
        }
    }
}

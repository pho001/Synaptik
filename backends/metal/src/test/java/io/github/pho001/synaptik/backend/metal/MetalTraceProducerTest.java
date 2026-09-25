package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.trace.TraceEvent;
import io.github.pho001.synaptik.trace.TraceLevel;
import io.github.pho001.synaptik.trace.TracePayload;
import io.github.pho001.synaptik.trace.TracePhase;
import io.github.pho001.synaptik.trace.payload.BackendInvocationOutcome;
import io.github.pho001.synaptik.trace.payload.BackendPreparationOutcome;
import io.github.pho001.synaptik.trace.payload.TraceCacheStatus;
import io.github.pho001.synaptik.trace.payload.TraceNativeStatusKind;
import io.github.pho001.synaptik.trace.payload.TraceNumericalProfile;
import io.github.pho001.synaptik.trace.payload.TraceOutcomeStatus;
import io.github.pho001.synaptik.trace.payload.TraceRouteKind;
import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.WildcardType;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

class MetalTraceProducerTest {
    @Test
    void observerAndOpenOverloadHaveTheExactPublicShape() throws Exception {
        assertTrue(Modifier.isPublic(MetalTraceObserver.class.getModifiers()));
        assertTrue(MetalTraceObserver.class.isInterface());
        assertEquals(0, MetalTraceObserver.class.getInterfaces().length);
        assertEquals(0, MetalTraceObserver.class.getDeclaredFields().length);
        assertEquals(0, MetalTraceObserver.class.getDeclaredClasses().length);
        assertEquals(1, MetalTraceObserver.class.getDeclaredMethods().length);

        var callback = MetalTraceObserver.class.getDeclaredMethod("onEvent", TraceEvent.class);
        assertTrue(Modifier.isPublic(callback.getModifiers()));
        assertTrue(Modifier.isAbstract(callback.getModifiers()));
        assertEquals(void.class, callback.getReturnType());
        ParameterizedType eventType = (ParameterizedType) callback.getGenericParameterTypes()[0];
        assertEquals(TraceEvent.class, eventType.getRawType());
        WildcardType payload = (WildcardType) eventType.getActualTypeArguments()[0];
        assertArrayEquals(
                new java.lang.reflect.Type[] {TracePayload.class}, payload.getUpperBounds());
        assertEquals(0, payload.getLowerBounds().length);

        var ordinary = MetalBackendIntegration.class.getMethod(
                "open", MetalBackendConfiguration.class);
        var traced = MetalBackendIntegration.class.getMethod(
                "open", MetalBackendConfiguration.class, MetalTraceObserver.class);
        assertTrue(Modifier.isPublic(ordinary.getModifiers()));
        assertTrue(Modifier.isStatic(ordinary.getModifiers()));
        assertTrue(Modifier.isPublic(traced.getModifiers()));
        assertTrue(Modifier.isStatic(traced.getModifiers()));
        assertEquals(MetalBackendIntegration.class, ordinary.getReturnType());
        assertEquals(MetalBackendIntegration.class, traced.getReturnType());

        NullPointerException missingConfiguration = assertThrows(
                NullPointerException.class,
                () -> MetalBackendIntegration.open(null, null));
        assertEquals("configuration", missingConfiguration.getMessage());
        MetalBackendConfiguration configuration = new MetalBackendConfiguration(
                Path.of("").toAbsolutePath().resolve("bridge.dylib"));
        NullPointerException missingObserver = assertThrows(
                NullPointerException.class,
                () -> MetalBackendIntegration.open(configuration, null));
        assertEquals("observer", missingObserver.getMessage());
    }

    @Test
    void emitsExactPreparationAndInvocationFactsWithIndependentSequences() {
        List<TraceEvent<? extends TracePayload>> events = new CopyOnWriteArrayList<>();
        MetalTraceProducer producer = new MetalTraceProducer(events::add);
        MetalTraceProducer.PreparedUnit first = producer.prepareUnit(
                NumericalProfile.STRICT_IEEE,
                MetalNegPreparationPlan.Route.CUSTOM_SINGLE_NEG);
        MetalTraceProducer.PreparedUnit second = producer.prepareUnit(
                NumericalProfile.ACCELERATOR,
                MetalNegPreparationPlan.Route.MPSGRAPH);

        first.preparationSucceeded();
        var firstInvocation = first.beginInvocation();
        first.invocationSucceeded(firstInvocation);
        second.preparationFailed(new IllegalStateException("java-side failure"));

        assertEquals(3, events.size());
        assertEquals(List.of(0L, 1L, 2L), events.stream()
                .map(event -> event.id().value())
                .toList());

        BackendPreparationOutcome prepared =
                (BackendPreparationOutcome) events.get(0).payload();
        assertEquals(TracePhase.PREPARE, events.get(0).phase());
        assertEquals(TraceLevel.INFO, events.get(0).level());
        assertEquals(0L, prepared.backendId().value());
        assertEquals(0L, prepared.deviceId().value());
        assertEquals(0L, prepared.preparedUnitId().value());
        assertEquals(TraceOutcomeStatus.SUCCEEDED, prepared.status());
        assertEquals(TraceNumericalProfile.STRICT_IEEE, prepared.profile());
        assertEquals(TraceRouteKind.CUSTOM_KERNEL, prepared.route());
        assertEquals(TraceCacheStatus.NOT_QUERIED, prepared.cacheStatus());
        assertEquals(TraceNativeStatusKind.SUCCESS,
                prepared.nativeStatus().orElseThrow().kind());
        assertEquals(0, prepared.nativeStatus().orElseThrow().code());

        BackendInvocationOutcome invoked =
                (BackendInvocationOutcome) events.get(1).payload();
        assertEquals(TracePhase.RUN, events.get(1).phase());
        assertEquals(TraceLevel.INFO, events.get(1).level());
        assertEquals(0L, invoked.preparedUnitId().value());
        assertEquals(0L, invoked.invocationId().value());
        assertEquals(TraceOutcomeStatus.SUCCEEDED, invoked.status());
        assertEquals(TraceRouteKind.CUSTOM_KERNEL, invoked.route());

        BackendPreparationOutcome failed =
                (BackendPreparationOutcome) events.get(2).payload();
        assertEquals(1L, failed.preparedUnitId().value());
        assertEquals(TraceOutcomeStatus.FAILED, failed.status());
        assertEquals(TraceNumericalProfile.ACCELERATOR, failed.profile());
        assertEquals(TraceRouteKind.GRAPH_EXECUTABLE, failed.route());
        assertTrue(failed.nativeStatus().isEmpty());
    }

    @Test
    void mapsEveryExactNativeStatusWithoutPublishingBackendNames() {
        List<TraceEvent<? extends TracePayload>> events = new CopyOnWriteArrayList<>();
        MetalTraceProducer producer = new MetalTraceProducer(events::add);
        MetalTraceProducer.PreparedUnit unit = producer.prepareUnit(
                NumericalProfile.STRICT_IEEE,
                MetalNegPreparationPlan.Route.MPSGRAPH);
        int[] codes = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -91, 91};
        TraceNativeStatusKind[] kinds = {
            TraceNativeStatusKind.INVALID_ARGUMENT,
            TraceNativeStatusKind.DEVICE_UNAVAILABLE,
            TraceNativeStatusKind.COMMAND_QUEUE_UNAVAILABLE,
            TraceNativeStatusKind.ALLOCATION_FAILED,
            TraceNativeStatusKind.RANGE_OUT_OF_BOUNDS,
            TraceNativeStatusKind.COPY_FAILED,
            TraceNativeStatusKind.INTERNAL_ERROR,
            TraceNativeStatusKind.UNSUPPORTED_SHAPE,
            TraceNativeStatusKind.COMPILATION_FAILED,
            TraceNativeStatusKind.INCOMPATIBLE_RESOURCE,
            TraceNativeStatusKind.EXECUTION_FAILED,
            TraceNativeStatusKind.COMPILATION_FAILED,
            TraceNativeStatusKind.UNKNOWN,
            TraceNativeStatusKind.UNKNOWN
        };

        for (int index = 0; index < codes.length; index++) {
            var invocationId = unit.beginInvocation();
            unit.invocationFailed(
                    invocationId, new MetalNativeApi.NativeFailure("native-run", codes[index]));
        }

        assertEquals(codes.length, events.size());
        for (int index = 0; index < codes.length; index++) {
            TraceEvent<? extends TracePayload> event = events.get(index);
            BackendInvocationOutcome outcome = (BackendInvocationOutcome) event.payload();
            assertEquals(TraceLevel.ERROR, event.level());
            assertEquals(TraceOutcomeStatus.FAILED, outcome.status());
            assertEquals(index, outcome.invocationId().value());
            assertEquals(codes[index], outcome.nativeStatus().orElseThrow().code());
            assertEquals(kinds[index], outcome.nativeStatus().orElseThrow().kind());
        }
    }

    @Test
    void runtimeCallbackFailureDisablesLaterWorkButErrorPropagates() {
        AtomicInteger callbacks = new AtomicInteger();
        RuntimeException callbackFailure = new RuntimeException("observer");
        MetalTraceProducer contained = new MetalTraceProducer(event -> {
            callbacks.incrementAndGet();
            throw callbackFailure;
        });
        MetalTraceProducer.PreparedUnit unit = contained.prepareUnit(
                NumericalProfile.STRICT_IEEE,
                MetalNegPreparationPlan.Route.CUSTOM_SINGLE_NEG);

        unit.preparationSucceeded();
        assertFalse(contained.enabled());
        assertNull(unit.beginInvocation());
        unit.preparationSucceeded();
        assertEquals(1, callbacks.get());

        AssertionError fatal = new AssertionError("fatal observer");
        MetalTraceProducer propagating = new MetalTraceProducer(event -> {
            throw fatal;
        });
        MetalTraceProducer.PreparedUnit fatalUnit = propagating.prepareUnit(
                NumericalProfile.STRICT_IEEE,
                MetalNegPreparationPlan.Route.CUSTOM_SINGLE_NEG);
        assertSame(fatal, assertThrows(AssertionError.class, fatalUnit::preparationSucceeded));
        assertTrue(propagating.enabled());
    }

    @Test
    void failureReportingPreservesPrimaryAndRejectsSelfOrCyclicSuppression() {
        AssertionError observerFailure = new AssertionError("observer failure");
        MetalTraceProducer producer = new MetalTraceProducer(event -> {
            throw observerFailure;
        });
        MetalTraceProducer.PreparedUnit unit = producer.prepareUnit(
                NumericalProfile.STRICT_IEEE,
                MetalNegPreparationPlan.Route.CUSTOM_SINGLE_NEG);
        RuntimeException preparationFailure = new RuntimeException("preparation");
        RuntimeException invocationFailure = new RuntimeException("invocation");

        unit.preparationFailed(preparationFailure);
        Error reported = unit.invocationFailed(unit.beginInvocation(), invocationFailure);

        assertArrayEquals(new Throwable[] {observerFailure}, preparationFailure.getSuppressed());
        assertArrayEquals(new Throwable[] {observerFailure}, invocationFailure.getSuppressed());
        assertSame(observerFailure, reported);
        assertTrue(producer.enabled());

        AssertionError backendError = new AssertionError("backend error");
        AssertionError errorObserverFailure = new AssertionError("error observer");
        MetalTraceProducer errorProducer = new MetalTraceProducer(event -> {
            throw errorObserverFailure;
        });
        MetalTraceProducer.PreparedUnit errorUnit = errorProducer.prepareUnit(
                NumericalProfile.STRICT_IEEE,
                MetalNegPreparationPlan.Route.CUSTOM_SINGLE_NEG);
        errorUnit.preparationFailed(backendError);
        assertArrayEquals(
                new Throwable[] {errorObserverFailure}, backendError.getSuppressed());
        assertTrue(errorProducer.enabled());

        AssertionError self = new AssertionError("self");
        MetalTraceProducer selfProducer = new MetalTraceProducer(event -> {
            throw self;
        });
        MetalTraceProducer.PreparedUnit selfUnit = selfProducer.prepareUnit(
                NumericalProfile.STRICT_IEEE,
                MetalNegPreparationPlan.Route.CUSTOM_SINGLE_NEG);
        selfUnit.preparationFailed(self);
        assertEquals(0, self.getSuppressed().length);

        RuntimeException primary = new RuntimeException("primary");
        AssertionError cyclic = new AssertionError("cyclic");
        cyclic.addSuppressed(primary);
        MetalTraceProducer cyclicProducer = new MetalTraceProducer(event -> {
            throw cyclic;
        });
        MetalTraceProducer.PreparedUnit cyclicUnit = cyclicProducer.prepareUnit(
                NumericalProfile.STRICT_IEEE,
                MetalNegPreparationPlan.Route.CUSTOM_SINGLE_NEG);
        cyclicUnit.preparationFailed(primary);
        assertEquals(0, primary.getSuppressed().length);
        assertArrayEquals(new Throwable[] {primary}, cyclic.getSuppressed());
    }

    @Test
    void constructionFailureAndEachSequenceExhaustionDisableWithoutWraparound() throws Exception {
        MetalTraceProducer invalid = new MetalTraceProducer(event -> {});
        assertNull(invalid.prepareUnit(
                null, MetalNegPreparationPlan.Route.CUSTOM_SINGLE_NEG));
        assertFalse(invalid.enabled());

        List<TraceEvent<? extends TracePayload>> preparedEvents = new CopyOnWriteArrayList<>();
        MetalTraceProducer prepared = new MetalTraceProducer(preparedEvents::add);
        sequence(prepared, "nextPreparedUnitId").set(Long.MAX_VALUE);
        MetalTraceProducer.PreparedUnit lastPrepared = prepared.prepareUnit(
                NumericalProfile.STRICT_IEEE,
                MetalNegPreparationPlan.Route.CUSTOM_SINGLE_NEG);
        assertEquals(Long.MAX_VALUE, lastPrepared.preparedUnitId().value());
        assertNull(prepared.prepareUnit(
                NumericalProfile.STRICT_IEEE,
                MetalNegPreparationPlan.Route.CUSTOM_SINGLE_NEG));
        assertFalse(prepared.enabled());

        List<TraceEvent<? extends TracePayload>> invocationEvents = new CopyOnWriteArrayList<>();
        MetalTraceProducer invocation = new MetalTraceProducer(invocationEvents::add);
        MetalTraceProducer.PreparedUnit invocationUnit = invocation.prepareUnit(
                NumericalProfile.STRICT_IEEE,
                MetalNegPreparationPlan.Route.CUSTOM_SINGLE_NEG);
        sequence(invocation, "nextInvocationId").set(Long.MAX_VALUE);
        var lastInvocation = invocationUnit.beginInvocation();
        assertEquals(Long.MAX_VALUE, lastInvocation.value());
        invocationUnit.invocationSucceeded(lastInvocation);
        assertNull(invocationUnit.beginInvocation());
        assertFalse(invocation.enabled());

        List<TraceEvent<? extends TracePayload>> eventEvents = new CopyOnWriteArrayList<>();
        MetalTraceProducer event = new MetalTraceProducer(eventEvents::add);
        MetalTraceProducer.PreparedUnit eventUnit = event.prepareUnit(
                NumericalProfile.STRICT_IEEE,
                MetalNegPreparationPlan.Route.CUSTOM_SINGLE_NEG);
        sequence(event, "nextEventId").set(Long.MAX_VALUE);
        eventUnit.preparationSucceeded();
        eventUnit.invocationSucceeded(eventUnit.beginInvocation());
        assertEquals(1, eventEvents.size());
        assertEquals(Long.MAX_VALUE, eventEvents.getFirst().id().value());
        assertFalse(event.enabled());
    }

    @Test
    void concurrentCallbacksRetainUniqueEventAndInvocationIdsWithoutSerialization() throws Exception {
        Set<Long> eventIds = ConcurrentHashMap.newKeySet();
        Set<Long> invocationIds = ConcurrentHashMap.newKeySet();
        CountDownLatch callbacksEntered = new CountDownLatch(4);
        MetalTraceProducer producer = new MetalTraceProducer(event -> {
            callbacksEntered.countDown();
            try {
                assertTrue(callbacksEntered.await(10, TimeUnit.SECONDS));
            } catch (InterruptedException failure) {
                Thread.currentThread().interrupt();
                throw new AssertionError(failure);
            }
            eventIds.add(event.id().value());
            BackendInvocationOutcome outcome = (BackendInvocationOutcome) event.payload();
            invocationIds.add(outcome.invocationId().value());
        });
        MetalTraceProducer.PreparedUnit unit = producer.prepareUnit(
                NumericalProfile.ACCELERATOR,
                MetalNegPreparationPlan.Route.MPSGRAPH);

        int invocations = 32;
        try (var executor = Executors.newFixedThreadPool(4)) {
            var work = Arrays.stream(new int[invocations])
                    .mapToObj(ignored -> executor.submit(() -> {
                        var invocationId = unit.beginInvocation();
                        unit.invocationSucceeded(invocationId);
                    }))
                    .toList();
            for (var future : work) {
                future.get(10, TimeUnit.SECONDS);
            }
        }

        assertEquals(invocations, eventIds.size());
        assertEquals(invocations, invocationIds.size());
        assertEquals(
                Set.copyOf(java.util.stream.LongStream.range(0, invocations)
                        .boxed()
                        .toList()),
                eventIds);
        assertEquals(eventIds, invocationIds);
    }

    private static AtomicLong sequence(MetalTraceProducer producer, String fieldName)
            throws ReflectiveOperationException {
        var field = MetalTraceProducer.class.getDeclaredField(fieldName);
        field.setAccessible(true);
        return (AtomicLong) field.get(producer);
    }
}

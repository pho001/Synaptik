package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.trace.TraceEvent;
import io.github.pho001.synaptik.trace.TraceEventId;
import io.github.pho001.synaptik.trace.TraceLevel;
import io.github.pho001.synaptik.trace.TracePayload;
import io.github.pho001.synaptik.trace.TracePhase;
import io.github.pho001.synaptik.trace.id.TraceBackendId;
import io.github.pho001.synaptik.trace.id.TraceDeviceId;
import io.github.pho001.synaptik.trace.id.TraceInvocationId;
import io.github.pho001.synaptik.trace.id.TracePreparedUnitId;
import io.github.pho001.synaptik.trace.payload.BackendInvocationOutcome;
import io.github.pho001.synaptik.trace.payload.BackendPreparationOutcome;
import io.github.pho001.synaptik.trace.payload.TraceCacheStatus;
import io.github.pho001.synaptik.trace.payload.TraceNativeStatus;
import io.github.pho001.synaptik.trace.payload.TraceNativeStatusKind;
import io.github.pho001.synaptik.trace.payload.TraceNumericalProfile;
import io.github.pho001.synaptik.trace.payload.TraceOutcomeStatus;
import io.github.pho001.synaptik.trace.payload.TraceRouteKind;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.LongFunction;

/** Concurrent, failure-contained producer for one explicitly traced Metal integration. */
final class MetalTraceProducer {
    private static final TraceBackendId BACKEND_ID = new TraceBackendId(0L);
    private static final TraceDeviceId DEVICE_ID = new TraceDeviceId(0L);
    private static final Optional<TraceNativeStatus> NATIVE_SUCCESS = Optional.of(
            new TraceNativeStatus(TraceNativeStatusKind.SUCCESS, 0));

    private final MetalTraceObserver observer;
    private final AtomicBoolean enabled = new AtomicBoolean(true);
    private final AtomicLong nextEventId = new AtomicLong();
    private final AtomicLong nextPreparedUnitId = new AtomicLong();
    private final AtomicLong nextInvocationId = new AtomicLong();

    MetalTraceProducer(MetalTraceObserver observer) {
        this.observer = Objects.requireNonNull(observer, "observer");
    }

    /**
     * Allocates and maps one prepared unit after route selection is final.
     *
     * @return immutable trace facts, or {@code null} when tracing is disabled or exhausted
     */
    PreparedUnit prepareUnit(
            NumericalProfile numericalProfile, MetalNegPreparationPlan.Route route) {
        if (!enabled.get()) {
            return null;
        }
        try {
            TracePreparedUnitId id = next(
                    nextPreparedUnitId, TracePreparedUnitId::new);
            if (id == null) {
                return null;
            }
            return new PreparedUnit(
                    this,
                    id,
                    mapProfile(numericalProfile),
                    mapRoute(route),
                    TraceCacheStatus.NOT_QUERIED);
        } catch (RuntimeException failure) {
            disable();
            return null;
        }
    }

    /** @return the next invocation identity, or {@code null} once tracing is disabled */
    TraceInvocationId beginInvocation(PreparedUnit unit) {
        if (!enabled.get()) {
            return null;
        }
        Objects.requireNonNull(unit, "unit");
        if (unit.producer != this) {
            throw new IllegalArgumentException("prepared trace unit belongs to another producer");
        }
        try {
            return next(nextInvocationId, TraceInvocationId::new);
        } catch (RuntimeException failure) {
            disable();
            return null;
        }
    }

    void preparationSucceeded(PreparedUnit unit) {
        emitPreparation(unit, TraceOutcomeStatus.SUCCEEDED, NATIVE_SUCCESS);
    }

    void preparationFailed(PreparedUnit unit, RuntimeException failure) {
        Objects.requireNonNull(failure, "failure");
        emitPreparation(unit, TraceOutcomeStatus.FAILED, nativeFailure(failure));
    }

    void invocationSucceeded(PreparedUnit unit, TraceInvocationId invocationId) {
        if (invocationId != null) {
            emitInvocation(unit, invocationId, TraceOutcomeStatus.SUCCEEDED, NATIVE_SUCCESS);
        }
    }

    void invocationFailed(
            PreparedUnit unit, TraceInvocationId invocationId, RuntimeException failure) {
        Objects.requireNonNull(failure, "failure");
        if (invocationId != null) {
            emitInvocation(
                    unit, invocationId, TraceOutcomeStatus.FAILED, nativeFailure(failure));
        }
    }

    boolean enabled() {
        return enabled.get();
    }

    private void emitPreparation(
            PreparedUnit unit,
            TraceOutcomeStatus status,
            Optional<TraceNativeStatus> nativeStatus) {
        if (!enabled.get() || unit == null) {
            return;
        }
        try {
            emit(
                    TracePhase.PREPARE,
                    status == TraceOutcomeStatus.SUCCEEDED ? TraceLevel.INFO : TraceLevel.ERROR,
                    new BackendPreparationOutcome(
                            BACKEND_ID,
                            DEVICE_ID,
                            unit.preparedUnitId,
                            status,
                            unit.profile,
                            unit.route,
                            unit.cacheStatus,
                            nativeStatus));
        } catch (RuntimeException failure) {
            disable();
        }
    }

    private void emitInvocation(
            PreparedUnit unit,
            TraceInvocationId invocationId,
            TraceOutcomeStatus status,
            Optional<TraceNativeStatus> nativeStatus) {
        if (!enabled.get() || unit == null) {
            return;
        }
        try {
            emit(
                    TracePhase.RUN,
                    status == TraceOutcomeStatus.SUCCEEDED ? TraceLevel.INFO : TraceLevel.ERROR,
                    new BackendInvocationOutcome(
                            BACKEND_ID,
                            DEVICE_ID,
                            unit.preparedUnitId,
                            invocationId,
                            status,
                            unit.profile,
                            unit.route,
                            nativeStatus));
        } catch (RuntimeException failure) {
            disable();
        }
    }

    private void emit(TracePhase phase, TraceLevel level, TracePayload payload) {
        if (!enabled.get()) {
            return;
        }
        try {
            TraceEventId eventId = next(nextEventId, TraceEventId::new);
            if (eventId == null) {
                return;
            }
            observer.onEvent(new TraceEvent<>(
                    eventId, phase, level, System.nanoTime(), payload));
        } catch (RuntimeException failure) {
            disable();
        }
    }

    private <T> T next(AtomicLong sequence, LongFunction<T> factory) {
        while (enabled.get()) {
            long current = sequence.get();
            if (current < 0L) {
                disable();
                return null;
            }
            long following = current == Long.MAX_VALUE ? -1L : current + 1L;
            if (sequence.compareAndSet(current, following)) {
                return factory.apply(current);
            }
        }
        return null;
    }

    private static Optional<TraceNativeStatus> nativeFailure(RuntimeException failure) {
        if (!(failure instanceof MetalNativeApi.NativeFailure nativeFailure)) {
            return Optional.empty();
        }
        int code = nativeFailure.statusCode();
        return Optional.of(new TraceNativeStatus(mapNativeStatus(code), code));
    }

    private static TraceNativeStatusKind mapNativeStatus(int code) {
        return switch (code) {
            case 0 -> TraceNativeStatusKind.SUCCESS;
            case 1 -> TraceNativeStatusKind.INVALID_ARGUMENT;
            case 2 -> TraceNativeStatusKind.DEVICE_UNAVAILABLE;
            case 3 -> TraceNativeStatusKind.COMMAND_QUEUE_UNAVAILABLE;
            case 4 -> TraceNativeStatusKind.ALLOCATION_FAILED;
            case 5 -> TraceNativeStatusKind.RANGE_OUT_OF_BOUNDS;
            case 6 -> TraceNativeStatusKind.COPY_FAILED;
            case 7 -> TraceNativeStatusKind.INTERNAL_ERROR;
            case 8 -> TraceNativeStatusKind.UNSUPPORTED_SHAPE;
            case 9, 12 -> TraceNativeStatusKind.COMPILATION_FAILED;
            case 10 -> TraceNativeStatusKind.INCOMPATIBLE_RESOURCE;
            case 11 -> TraceNativeStatusKind.EXECUTION_FAILED;
            default -> TraceNativeStatusKind.UNKNOWN;
        };
    }

    private static TraceNumericalProfile mapProfile(NumericalProfile profile) {
        return switch (Objects.requireNonNull(profile, "numericalProfile")) {
            case STRICT_IEEE -> TraceNumericalProfile.STRICT_IEEE;
            case ACCELERATOR -> TraceNumericalProfile.ACCELERATOR;
        };
    }

    private static TraceRouteKind mapRoute(MetalNegPreparationPlan.Route route) {
        return switch (Objects.requireNonNull(route, "route")) {
            case CUSTOM_SINGLE_NEG -> TraceRouteKind.CUSTOM_KERNEL;
            case MPSGRAPH -> TraceRouteKind.GRAPH_EXECUTABLE;
        };
    }

    private void disable() {
        enabled.set(false);
    }

    /** Immutable trace facts fixed before native finalization. */
    static final class PreparedUnit {
        private final MetalTraceProducer producer;
        private final TracePreparedUnitId preparedUnitId;
        private final TraceNumericalProfile profile;
        private final TraceRouteKind route;
        private final TraceCacheStatus cacheStatus;

        private PreparedUnit(
                MetalTraceProducer producer,
                TracePreparedUnitId preparedUnitId,
                TraceNumericalProfile profile,
                TraceRouteKind route,
                TraceCacheStatus cacheStatus) {
            this.producer = producer;
            this.preparedUnitId = preparedUnitId;
            this.profile = profile;
            this.route = route;
            this.cacheStatus = cacheStatus;
        }

        void preparationSucceeded() {
            producer.preparationSucceeded(this);
        }

        void preparationFailed(RuntimeException failure) {
            producer.preparationFailed(this, failure);
        }

        TraceInvocationId beginInvocation() {
            return producer.beginInvocation(this);
        }

        void invocationSucceeded(TraceInvocationId invocationId) {
            producer.invocationSucceeded(this, invocationId);
        }

        void invocationFailed(
                TraceInvocationId invocationId, RuntimeException failure) {
            producer.invocationFailed(this, invocationId, failure);
        }

        TracePreparedUnitId preparedUnitId() {
            return preparedUnitId;
        }
    }
}

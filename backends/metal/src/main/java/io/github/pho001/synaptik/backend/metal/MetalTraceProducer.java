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
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HexFormat;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.LongFunction;

/** Concurrent Metal trace producer with advisory structure and contract-preserving outcomes. */
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
            NumericalProfile numericalProfile, MetalPreparedRoute route) {
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
                    TraceCacheStatus.NOT_QUERIED,
                    List.of(),
                    false);
        } catch (RuntimeException failure) {
            disable();
            return null;
        }
    }

    PreparedUnit prepareUnit(
            NumericalProfile numericalProfile,
            MetalPreparedRoute route,
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] feeds,
            int[] targets,
            int internalCount) {
        if (!enabled.get()) {
            return null;
        }
        try {
            TracePreparedUnitId id = next(
                    nextPreparedUnitId, TracePreparedUnitId::new);
            if (id == null) {
                return null;
            }
            PlanFacts facts = planFacts(
                    numericalProfile, route, program, values, feeds, targets, internalCount);
            var unit = new PreparedUnit(
                    this,
                    id,
                    mapProfile(numericalProfile),
                    mapRoute(route),
                    TraceCacheStatus.NOT_QUERIED,
                    facts.plannedCustomSteps(),
                    facts.plannedCustomStepsTruncated());
            emitPreparationStructure(unit, facts);
            return unit;
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
    /** Emits advisory pre-run structure; observer failures disable tracing and never escape. */
    void invocationPlanned(
            PreparedUnit unit,
            TraceInvocationId invocationId,
            int feedSlotCount,
            int targetSlotCount,
            int internalSlotCount,
            long aggregateInputBytes,
            long aggregateOutputBytes,
            long aggregateInternalBytes,
            long aggregateSplatBytes,
            long aggregateWorkspaceBytes,
            long aggregateRequiredBytes,
            int splatCount,
            int workspaceCount) {
        if (!enabled.get() || unit == null || invocationId == null) {
            return;
        }
        try {
            emitStructural(
                    TracePhase.RUN,
                    TraceLevel.INFO,
                    new MetalInvocationPlan(
                            unit.preparedUnitId,
                            invocationId,
                            unit.route,
                            feedSlotCount,
                            targetSlotCount,
                            internalSlotCount,
                            aggregateInputBytes,
                            aggregateOutputBytes,
                            aggregateInternalBytes,
                            aggregateSplatBytes,
                            aggregateWorkspaceBytes,
                            aggregateRequiredBytes,
                            splatCount,
                            workspaceCount,
                            unit.plannedCustomSteps,
                            unit.plannedCustomStepsTruncated));
        } catch (RuntimeException failure) {
            disable();
        }
    }

    /** Emits advisory PREPARE structure without letting observer failures reach finalization. */
    private void emitPreparationStructure(PreparedUnit unit, PlanFacts facts) {
        emitStructural(
                TracePhase.PREPARE,
                TraceLevel.INFO,
                new MetalPreparationStructure(
                        unit.preparedUnitId,
                        unit.profile,
                        unit.route,
                        facts.anchorFamily(),
                        facts.anchorDisposition(),
                        facts.scalarMultiply(),
                        facts.externalAdd(),
                        facts.epilogueOrder(),
                        facts.terminal(),
                        facts.anchorCount(),
                        facts.anchorMemberCount(),
                        MetalMpsGraphProgram.SCHEMA_VERSION,
                        facts.generatorVersion(),
                        facts.valueCount(),
                        facts.nodeCount(),
                        facts.feedCount(),
                        facts.targetCount(),
                        facts.internalCount(),
                        facts.stepCount(),
                        facts.instructionCount(),
                        facts.canonicalDigest(),
                        facts.plannedCustomSteps(),
                        facts.plannedCustomStepsTruncated()));
    }


    void preparationSucceeded(PreparedUnit unit) {
        emitPreparation(unit, TraceOutcomeStatus.SUCCEEDED, NATIVE_SUCCESS);
    }

    void preparationFailed(PreparedUnit unit, Throwable failure) {
        Objects.requireNonNull(failure, "failure");
        try {
            emitPreparation(unit, TraceOutcomeStatus.FAILED, nativeFailure(failure));
        } catch (Error observerFailure) {
            suppressObserverError(failure, observerFailure);
        }
    }

    void invocationSucceeded(PreparedUnit unit, TraceInvocationId invocationId) {
        if (invocationId != null) {
            emitInvocation(unit, invocationId, TraceOutcomeStatus.SUCCEEDED, NATIVE_SUCCESS);
        }
    }

    Error invocationFailed(
            PreparedUnit unit, TraceInvocationId invocationId, Throwable failure) {
        Objects.requireNonNull(failure, "failure");
        if (invocationId == null) {
            return null;
        }
        try {
            emitInvocation(
                    unit, invocationId, TraceOutcomeStatus.FAILED, nativeFailure(failure));
            return null;
        } catch (Error observerFailure) {
            suppressObserverError(failure, observerFailure);
            return observerFailure;
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
            emitOutcome(
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
            emitOutcome(
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

    private void emitStructural(TracePhase phase, TraceLevel level, TracePayload payload) {
        try {
            emitEvent(phase, level, payload);
        } catch (RuntimeException | Error observerFailure) {
            disable();
        }
    }

    private void emitOutcome(TracePhase phase, TraceLevel level, TracePayload payload) {
        try {
            emitEvent(phase, level, payload);
        } catch (RuntimeException observerFailure) {
            disable();
        }
    }

    private void emitEvent(TracePhase phase, TraceLevel level, TracePayload payload) {
        if (!enabled.get()) {
            return;
        }
        TraceEventId eventId = next(nextEventId, TraceEventId::new);
        if (eventId == null) {
            return;
        }
        observer.onEvent(new TraceEvent<>(
                eventId, phase, level, System.nanoTime(), payload));
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

    private static PlanFacts planFacts(
            NumericalProfile numericalProfile,
            MetalPreparedRoute route,
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] feeds,
            int[] targets,
            int internalCount) {
        Objects.requireNonNull(program, "program");
        Objects.requireNonNull(values, "values");
        Objects.requireNonNull(feeds, "feeds");
        Objects.requireNonNull(targets, "targets");
        List<AnchorFacts> sourceAnchors =
                sourceAnchors(program, values, targets);
        MetalPointwiseFusionPlan fusion = route == MetalPreparedRoute.CUSTOM_PROGRAM
                ? MetalPointwiseFusionPlanner.plan(
                        numericalProfile, program, values, feeds, targets, route)
                : null;
        int fusedAnchors = 0;
        int fusedMembers = 0;
        var summaries = new ArrayList<MetalPreparationStructure.CustomStepSummary>();
        boolean summariesTruncated = false;
        if (fusion != null) {
            for (int ordinal = 0; ordinal < fusion.steps().size(); ordinal++) {
                MetalPointwiseFusionPlan.Step step = fusion.steps().get(ordinal);
                if (step.kind() == MetalPointwiseFusionPlan.StepKind.ANCHOR_EPILOGUE) {
                    fusedAnchors++;
                    fusedMembers += step.memberCount();
                }
                if (summaries.size() < MetalPreparationStructure.MAX_PLANNED_CUSTOM_STEPS) {
                    MetalPreparationStructure.AnchorFamily family = switch (step.anchorKindWire()) {
                        case 1 -> MetalPreparationStructure.AnchorFamily.MATMUL;
                        case 2 -> MetalPreparationStructure.AnchorFamily.CONV2D;
                        default -> MetalPreparationStructure.AnchorFamily.NONE;
                    };
                    MetalPreparationStructure.CustomStepKind kind = switch (step.kind()) {
                        case FIXED_CUSTOM ->
                            MetalPreparationStructure.CustomStepKind.FIXED_CUSTOM;
                        case MPSGRAPH_BOUNDARY ->
                            MetalPreparationStructure.CustomStepKind.GRAPH_BOUNDARY;
                        case GENERATED_POINTWISE ->
                            MetalPreparationStructure.CustomStepKind.GENERATED_POINTWISE;
                        case ANCHOR_EPILOGUE ->
                            MetalPreparationStructure.CustomStepKind.ANCHOR_EPILOGUE;
                    };
                    summaries.add(new MetalPreparationStructure.CustomStepSummary(
                            ordinal,
                            kind,
                            family,
                            step.memberCount(),
                            step.instructionCount()));
                } else {
                    summariesTruncated = true;
                }
            }
        }
        MetalPreparationStructure.AnchorFamily family = aggregateFamily(sourceAnchors);
        MetalPreparationStructure.AnchorDisposition disposition;
        if (sourceAnchors.isEmpty()) {
            disposition = MetalPreparationStructure.AnchorDisposition.NONE;
        } else if (fusedAnchors == 0) {
            disposition = MetalPreparationStructure.AnchorDisposition.COMPOSED;
        } else if (fusedAnchors == sourceAnchors.size()) {
            disposition = MetalPreparationStructure.AnchorDisposition.FUSED;
        } else {
            disposition = MetalPreparationStructure.AnchorDisposition.MIXED;
        }
        boolean scalar = sourceAnchors.stream().anyMatch(AnchorFacts::scalarMultiply);
        boolean add = sourceAnchors.stream().anyMatch(AnchorFacts::externalAdd);
        MetalPreparationStructure.EpilogueOrder order = sourceAnchors.size() == 1
                ? sourceAnchors.getFirst().order()
                : sourceAnchors.isEmpty()
                        ? MetalPreparationStructure.EpilogueOrder.NONE
                        : MetalPreparationStructure.EpilogueOrder.MULTIPLE;
        MetalPreparationStructure.EpilogueTerminal terminal = sourceAnchors.size() == 1
                ? sourceAnchors.getFirst().terminal()
                : sourceAnchors.isEmpty()
                        ? MetalPreparationStructure.EpilogueTerminal.NONE
                        : MetalPreparationStructure.EpilogueTerminal.MULTIPLE;
        int anchorMembers = fusedAnchors == sourceAnchors.size()
                ? fusedMembers
                : sourceAnchors.stream().mapToInt(AnchorFacts::memberCount).sum();
        return new PlanFacts(
                family,
                disposition,
                scalar,
                add,
                order,
                terminal,
                sourceAnchors.size(),
                anchorMembers,
                fusion == null ? 0 : MetalPointwiseFusionPlan.GENERATOR_SCHEMA,
                values.size(),
                program.nodes().size(),
                feeds.length,
                targets.length,
                internalCount,
                fusion == null ? program.nodes().size() : fusion.steps().size(),
                fusion == null ? 0 : fusion.instructions().size(),
                canonicalDigest(numericalProfile, route, program, values, feeds, targets),
                List.copyOf(summaries),
                summariesTruncated);
    }

    private static List<AnchorFacts> sourceAnchors(
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] targets) {
        List<MetalAnchorEpilogue> recognized =
                MetalAnchorEpilogueRecognizer.recognize(
                        NumericalProfile.ACCELERATOR, program, values, targets);
        var result = new ArrayList<AnchorFacts>(recognized.size());
        for (MetalAnchorEpilogue anchor : recognized) {
            var opcodes = new ArrayList<Integer>(anchor.operations().size());
            boolean scalar = false;
            boolean add = false;
            for (MetalAnchorEpilogue.Operation operation : anchor.operations()) {
                int opcode = operation.opcode().wire();
                opcodes.add(opcode);
                scalar |= operation instanceof MetalAnchorEpilogue.ScalarMultiply;
                add |= operation instanceof MetalAnchorEpilogue.Add;
            }
            MetalPreparationStructure.AnchorFamily family =
                    anchor.anchorKind() == MetalAnchorEpilogue.AnchorKind.MATMUL
                            ? MetalPreparationStructure.AnchorFamily.MATMUL
                            : MetalPreparationStructure.AnchorFamily.CONV2D;
            result.add(new AnchorFacts(
                    family,
                    scalar,
                    add,
                    epilogueOrder(opcodes),
                    epilogueTerminal(opcodes),
                    anchor.memberCount()));
        }
        return List.copyOf(result);
    }

    private static MetalPreparationStructure.AnchorFamily aggregateFamily(
            List<AnchorFacts> anchors) {
        boolean matmul = anchors.stream()
                .anyMatch(anchor ->
                        anchor.family() == MetalPreparationStructure.AnchorFamily.MATMUL);
        boolean conv = anchors.stream()
                .anyMatch(anchor ->
                        anchor.family() == MetalPreparationStructure.AnchorFamily.CONV2D);
        if (matmul && conv) return MetalPreparationStructure.AnchorFamily.MIXED;
        if (matmul) return MetalPreparationStructure.AnchorFamily.MATMUL;
        if (conv) return MetalPreparationStructure.AnchorFamily.CONV2D;
        return MetalPreparationStructure.AnchorFamily.NONE;
    }

    private static MetalPreparationStructure.EpilogueOrder epilogueOrder(
            List<Integer> opcodes) {
        StringBuilder key = new StringBuilder(opcodes.size());
        for (int opcode : opcodes) key.append(opcode);
        return switch (key.toString()) {
            case "1" -> MetalPreparationStructure.EpilogueOrder.SCALAR;
            case "2" -> MetalPreparationStructure.EpilogueOrder.ADD;
            case "3" -> MetalPreparationStructure.EpilogueOrder.RELU;
            case "4" -> MetalPreparationStructure.EpilogueOrder.CLAMP;
            case "12" -> MetalPreparationStructure.EpilogueOrder.SCALAR_ADD;
            case "13" -> MetalPreparationStructure.EpilogueOrder.SCALAR_RELU;
            case "14" -> MetalPreparationStructure.EpilogueOrder.SCALAR_CLAMP;
            case "23" -> MetalPreparationStructure.EpilogueOrder.ADD_RELU;
            case "24" -> MetalPreparationStructure.EpilogueOrder.ADD_CLAMP;
            case "123" -> MetalPreparationStructure.EpilogueOrder.SCALAR_ADD_RELU;
            case "124" -> MetalPreparationStructure.EpilogueOrder.SCALAR_ADD_CLAMP;
            default -> MetalPreparationStructure.EpilogueOrder.MULTIPLE;
        };
    }

    private static MetalPreparationStructure.EpilogueTerminal epilogueTerminal(
            List<Integer> opcodes) {
        return switch (opcodes.getLast()) {
            case 3 -> MetalPreparationStructure.EpilogueTerminal.RELU;
            case 4 -> MetalPreparationStructure.EpilogueTerminal.CLAMP;
            default -> MetalPreparationStructure.EpilogueTerminal.NONE;
        };
    }

    private static String canonicalDigest(
            NumericalProfile numericalProfile,
            MetalPreparedRoute route,
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] feeds,
            int[] targets) {
        MessageDigest digest;
        try {
            digest = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 unavailable", impossible);
        }
        MetalPreparedRoute imageRoute = route == MetalPreparedRoute.CUSTOM_SINGLE_NEG
                ? MetalPreparedRoute.MPSGRAPH : route;
        program.updateDigest(digest, numericalProfile, values, feeds, targets, imageRoute);
        return HexFormat.of().formatHex(digest.digest());
    }

    private record AnchorFacts(
            MetalPreparationStructure.AnchorFamily family,
            boolean scalarMultiply,
            boolean externalAdd,
            MetalPreparationStructure.EpilogueOrder order,
            MetalPreparationStructure.EpilogueTerminal terminal,
            int memberCount) {}

    private record PlanFacts(
            MetalPreparationStructure.AnchorFamily anchorFamily,
            MetalPreparationStructure.AnchorDisposition anchorDisposition,
            boolean scalarMultiply,
            boolean externalAdd,
            MetalPreparationStructure.EpilogueOrder epilogueOrder,
            MetalPreparationStructure.EpilogueTerminal terminal,
            int anchorCount,
            int anchorMemberCount,
            int generatorVersion,
            int valueCount,
            int nodeCount,
            int feedCount,
            int targetCount,
            int internalCount,
            int stepCount,
            int instructionCount,
            String canonicalDigest,
            List<MetalPreparationStructure.CustomStepSummary> plannedCustomSteps,
            boolean plannedCustomStepsTruncated) {}

    private static Optional<TraceNativeStatus> nativeFailure(Throwable failure) {
        if (!(failure instanceof MetalNativeApi.NativeFailure nativeFailure)) {
            return Optional.empty();
        }
        int code = nativeFailure.statusCode();
        return Optional.of(new TraceNativeStatus(mapNativeStatus(code), code));
    }

    static void suppressObserverError(Throwable primary, Error observerFailure) {
        if (primary == observerFailure
                || contains(primary, observerFailure)
                || contains(observerFailure, primary)) {
            return;
        }
        primary.addSuppressed(observerFailure);
    }

    private static boolean contains(Throwable root, Throwable target) {
        var pending = new ArrayDeque<Throwable>();
        var seen = Collections.newSetFromMap(new IdentityHashMap<Throwable, Boolean>());
        pending.add(root);
        while (!pending.isEmpty()) {
            Throwable current = pending.removeLast();
            if (!seen.add(current)) {
                continue;
            }
            if (current == target) {
                return true;
            }
            Throwable cause = current.getCause();
            if (cause != null) {
                pending.add(cause);
            }
            for (Throwable suppressed : current.getSuppressed()) {
                pending.add(suppressed);
            }
        }
        return false;
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

    private static TraceRouteKind mapRoute(MetalPreparedRoute route) {
        return switch (Objects.requireNonNull(route, "route").family()) {
            case CUSTOM_KERNEL -> TraceRouteKind.CUSTOM_KERNEL;
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
        private final List<MetalPreparationStructure.CustomStepSummary> plannedCustomSteps;
        private final boolean plannedCustomStepsTruncated;

        private PreparedUnit(
                MetalTraceProducer producer,
                TracePreparedUnitId preparedUnitId,
                TraceNumericalProfile profile,
                TraceRouteKind route,
                TraceCacheStatus cacheStatus,
                List<MetalPreparationStructure.CustomStepSummary> plannedCustomSteps,
                boolean plannedCustomStepsTruncated) {
            this.producer = producer;
            this.preparedUnitId = preparedUnitId;
            this.profile = profile;
            this.route = route;
            this.cacheStatus = cacheStatus;
            this.plannedCustomSteps = plannedCustomSteps;
            this.plannedCustomStepsTruncated = plannedCustomStepsTruncated;
        }

        void preparationSucceeded() {
            producer.preparationSucceeded(this);
        }

        void preparationFailed(Throwable failure) {
            producer.preparationFailed(this, failure);
        }

        TraceInvocationId beginInvocation() {
            return producer.beginInvocation(this);
        }

        void invocationSucceeded(TraceInvocationId invocationId) {
            producer.invocationSucceeded(this, invocationId);
        }

        void invocationPlanned(
                TraceInvocationId invocationId,
                int feedSlotCount,
                int targetSlotCount,
                int internalSlotCount,
                long aggregateInputBytes,
                long aggregateOutputBytes,
                long aggregateInternalBytes,
                long aggregateSplatBytes,
                long aggregateWorkspaceBytes,
                long aggregateRequiredBytes,
                int splatCount,
                int workspaceCount) {
            producer.invocationPlanned(
                    this,
                    invocationId,
                    feedSlotCount,
                    targetSlotCount,
                    internalSlotCount,
                    aggregateInputBytes,
                    aggregateOutputBytes,
                    aggregateInternalBytes,
                    aggregateSplatBytes,
                    aggregateWorkspaceBytes,
                    aggregateRequiredBytes,
                    splatCount,
                    workspaceCount);
        }

        boolean enabled() {
            return producer.enabled();
        }

        Error invocationFailed(
                TraceInvocationId invocationId, Throwable failure) {
            return producer.invocationFailed(this, invocationId, failure);
        }

        TracePreparedUnitId preparedUnitId() {
            return preparedUnitId;
        }
    }
}

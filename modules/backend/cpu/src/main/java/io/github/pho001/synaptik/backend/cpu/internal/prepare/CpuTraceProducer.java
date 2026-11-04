package io.github.pho001.synaptik.backend.cpu.internal.prepare;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.cpu.CpuPreparationDiagnostics;
import io.github.pho001.synaptik.backend.cpu.CpuPreparationDiagnostics.*;
import io.github.pho001.synaptik.backend.cpu.CpuTraceObserver;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnKernelSpec;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnLibrary;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnRoutePlan;
import io.github.pho001.synaptik.prepare.*;
import io.github.pho001.synaptik.prepare.analysis.*;
import io.github.pho001.synaptik.trace.*;
import io.github.pho001.synaptik.trace.id.TracePreparedUnitId;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.OptionalInt;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Optional cold-only CPU diagnostic producer. Never constructed by the no-observer composition.
 * Analysis scopes are thread-local and nested/reentrant-safe; detached selected facts travel in a
 * private plan wrapper to finalization, without a registry retaining graphs or native resources.
 * Every diagnostic-only operation is isolated from business work, including observer Errors.
 */
public final class CpuTraceProducer {
    private final CpuTraceObserver observer;
    private final Api configuredApi;
    private final AtomicLong nextEvent = new AtomicLong();
    private final AtomicLong nextPreparation = new AtomicLong();
    private final ThreadLocal<Attempt> current = new ThreadLocal<>();
    private volatile boolean enabled = true;

    /**
     * Creates an opt-in producer; the observer remains caller-owned and is never closed.
     * @param observer non-null thread-safe caller sink
     * @param configuredApi non-null oneDNN policy, NONE when not configured
     */
    public CpuTraceProducer(CpuTraceObserver observer, Api configuredApi) {
        this.observer = Objects.requireNonNull(observer, "observer");
        this.configuredApi = Objects.requireNonNull(configuredApi, "configuredApi");
    }

    /**
     * Wraps only cold shared-Prepare collaborations; execution and resource ownership are unchanged.
     * @param inputs exact non-null CPU analysis inputs
     * @param preparer non-null authoritative ordinary/selected-plan analysis
     * @param finalizer non-null authoritative finalizer, retaining all rollback duties
     * @return a preparation whose detached trace identity survives analysis-to-finalization handoff
     */
    public PartitionPreparation<?, ?> preparation(CpuPartitionAnalysisInputs inputs,
            BackendPartitionPreparer<CpuPartitionAnalysisInputs, CpuPartitionPreparationPlan> preparer,
            BackendPartitionFinalizer<CpuPartitionPreparationPlan> finalizer) {
        if (!enabled) return new PartitionPreparation<>(inputs, preparer, finalizer);
        return new PartitionPreparation<>(inputs,
                (BackendPartitionPreparer<CpuPartitionAnalysisInputs, ObservedPlan>) context -> {
                    Attempt attempt = begin();
                    BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis;
                    try {
                        analysis = preparer.analyze(context);
                    } catch (RuntimeException | Error failure) {
                        finish(attempt);
                        report(attempt, null, Stage.ANALYSIS_FAILED, failure);
                        throw failure;
                    }
                    finish(attempt);
                    CpuPreparationDiagnostics selected = report(attempt, analysis.plan(), Stage.SELECTED, null);
                    return new BackendPartitionAnalysis<>(analysis.partition(),
                            new ObservedPlan(analysis, selected), analysis.requirements());
                }, new BackendPartitionFinalizer<ObservedPlan>() {
                    @Override public BackendId backendId() { return finalizer.backendId(); }
                    @Override public BackendPartitionFinalizationResult finalizePartition(
                            BackendPartitionFinalization<ObservedPlan> handoff) {
                        ObservedPlan observed = handoff.analysis().plan();
                        BackendPartitionFinalizationResult result;
                        try {
                            result = finalizer.finalizePartition(new BackendPartitionFinalization<>(
                                    observed.analysis(), handoff.memoryPlan(), handoff.assignments()));
                        } catch (RuntimeException | Error failure) {
                            outcome(observed.selected(), Stage.FINALIZATION_FAILED, failure);
                            throw failure;
                        }
                        outcome(observed.selected(), Stage.FINALIZED, null);
                        return result;
                    }
                });
    }

    /**
     * Records a bounded native-candidate rejection inside the current synchronous analysis only.
     * It never changes candidate admission or propagates a diagnostic failure.
     * @param position candidate unit position, or -1 for a whole-plan reason
     * @param spec candidate descriptor recipe, or null for a whole-plan reason; not retained
     * @param api actual queried API, or null if no query occurred
     * @param reason non-null observed reason category
     * @param detail non-null data-free explanation; truncated to 240 characters
     * @param status exact native status when known, otherwise null
     */
    public void rejected(int position, CpuOneDnnKernelSpec spec, CpuOneDnnRoutePlan.Api api,
            Reason reason, String detail, Integer status) {
        if (!enabled) return;
        try {
            Attempt attempt = current.get();
            if (attempt == null) return;
            attempt.compared = true;
            if (attempt.rejections.size() == CpuPreparationDiagnostics.MAX_SUMMARIES) {
                attempt.truncated = true;
                return;
            }
            attempt.rejections.add(new Rejection(position, family(spec), api(api), reason,
                    detail.substring(0, Math.min(240, detail.length())),
                    status == null ? OptionalInt.empty() : OptionalInt.of(status)));
        } catch (RuntimeException | Error diagnosticFailure) {
            enabled = false;
        }
    }

    /** Marks a real selector comparison even if every candidate is selected without rejection. */
    public void compared() {
        if (!enabled) return;
        Attempt attempt = current.get();
        if (attempt != null) attempt.compared = true;
    }

    private Attempt begin() {
        if (!enabled) return null;
        try {
            long id = nextPreparation.incrementAndGet();
            if (id <= 0) { enabled = false; return null; }
            Attempt attempt = new Attempt(new TracePreparedUnitId(id), current.get());
            current.set(attempt);
            return attempt;
        } catch (RuntimeException | Error diagnosticFailure) {
            enabled = false;
            return null;
        }
    }

    private void finish(Attempt attempt) {
        if (attempt == null) return;
        if (attempt.previous == null) current.remove();
        else current.set(attempt.previous);
    }

    private CpuPreparationDiagnostics report(Attempt attempt, CpuPartitionPreparationPlan plan,
            Stage stage, Throwable failure) {
        if (!enabled || attempt == null) return null;
        try {
            if (!attempt.compared) {
                attempt.rejections.add(new Rejection(-1, Family.OTHER, Api.NONE,
                        configuredApi == Api.NONE ? Reason.NOT_CONFIGURED : Reason.COMPARISON_SKIPPED,
                        configuredApi == Api.NONE ? "oneDNN is not configured"
                                : "native comparison not entered for this analysis policy or topology",
                        OptionalInt.empty()));
            }
            var units = new ArrayList<Unit>();
            if (plan != null) {
                for (int position = 0; position < Math.min(plan.units().size(),
                        CpuPreparationDiagnostics.MAX_SUMMARIES); position++) {
                    var unit = plan.units().get(position);
                    CpuOneDnnRoutePlan.Unit nativeUnit = null;
                    if (plan.oneDnnPlan().isPresent()) {
                        for (var candidate : plan.oneDnnPlan().orElseThrow().units()) {
                            if (candidate.unitPosition() == position) { nativeUnit = candidate; break; }
                        }
                    }
                    Family family = unit.matmulGeometry().isPresent() ? Family.CONTRACTION
                            : unit.portablePlan() != null
                                    && unit.portablePlan().portableKernelIr() instanceof CpuKernelIr
                                    ? Family.POINTWISE : Family.OTHER;
                    units.add(new Unit(position, nativeUnit == null ? family : family(nativeUnit.projection().spec()),
                            nativeUnit != null ? Route.ONEDNN : plan.route() == CpuPartitionPreparationPlan.Route.OPENBLAS
                                    ? Route.OPENBLAS : Route.PORTABLE,
                            nativeUnit == null ? Api.NONE : api(nativeUnit.api())));
                }
            }
            var payload = new CpuPreparationDiagnostics(attempt.id, stage, configuredApi,
                    plan == null ? Route.UNKNOWN : Route.valueOf(plan.route().name()), units,
                    plan != null && plan.units().size() > units.size(), attempt.rejections,
                    attempt.truncated, failure == null ? "" : failure.getClass().getName(), status(failure));
            emit(payload);
            return payload;
        } catch (RuntimeException | Error diagnosticFailure) {
            enabled = false;
            return null;
        }
    }

    private void outcome(CpuPreparationDiagnostics selected, Stage stage, Throwable failure) {
        if (!enabled || selected == null) return;
        try {
            emit(new CpuPreparationDiagnostics(selected.preparationId(), stage, configuredApi,
                    selected.selectedRoute(), selected.units(), selected.unitsTruncated(),
                    selected.rejections(), selected.rejectionsTruncated(),
                    failure == null ? "" : failure.getClass().getName(), status(failure)));
        } catch (RuntimeException | Error diagnosticFailure) {
            enabled = false;
        }
    }

    private void emit(CpuPreparationDiagnostics payload) {
        if (!enabled) return;
        long id = nextEvent.incrementAndGet();
        if (id <= 0) { enabled = false; return; }
        observer.onEvent(new TraceEvent<>(new TraceEventId(id), TracePhase.PREPARE,
                payload.failureType().isEmpty() ? TraceLevel.INFO : TraceLevel.ERROR,
                System.nanoTime(), payload));
    }

    private static OptionalInt status(Throwable failure) {
        return failure instanceof CpuOneDnnLibrary.NativeStatusException nativeFailure
                ? OptionalInt.of(nativeFailure.status()) : OptionalInt.empty();
    }

    private static Api api(CpuOneDnnRoutePlan.Api api) {
        return api == null ? Api.NONE : switch (api) {
            case AUTO -> Api.AUTO;
            case PRIMITIVE -> Api.PRIMITIVES;
            case GRAPH -> Api.GRAPH;
        };
    }

    private static Family family(CpuOneDnnKernelSpec spec) {
        if (spec == null) return Family.OTHER;
        return switch (spec) {
            case CpuOneDnnKernelSpec.Contraction ignored -> Family.CONTRACTION;
            case CpuOneDnnKernelSpec.Unary ignored -> Family.POINTWISE;
            case CpuOneDnnKernelSpec.Binary ignored -> Family.POINTWISE;
            case CpuOneDnnKernelSpec.Reduction ignored -> Family.REDUCTION;
            case CpuOneDnnKernelSpec.Reorder ignored -> Family.MOVEMENT;
            case CpuOneDnnKernelSpec.Pipeline ignored -> Family.PIPELINE;
            default -> Family.OTHER;
        };
    }

    private static final class Attempt {
        final TracePreparedUnitId id;
        final Attempt previous;
        final List<Rejection> rejections = new ArrayList<>();
        boolean compared;
        boolean truncated;
        Attempt(TracePreparedUnitId id, Attempt previous) { this.id = id; this.previous = previous; }
    }

    private record ObservedPlan(BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis,
            CpuPreparationDiagnostics selected) implements BackendPreparationPlan { }
}

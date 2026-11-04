package io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.CpuPreparationDiagnostics.Reason;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuTraceProducer;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/**
 * Cold, deterministic whole-plan comparison over common units. Costs are conservative dimensionless
 * heuristics, not measured nanoseconds: arithmetic, dispatch/control binding, preparation,
 * materialized boundaries, ingress copies and exact user scratch are all charged. API availability
 * is tested before selection; no probe or route decision is retained in the hot invocation.
 * Both native APIs use retained common units and canonical external stores; API preference is
 * not provider priority or forced native selection. A selected topology is never replaced here.
 */
public final class CpuOneDnnSelector {
    private static final long PORTABLE_DISPATCH = 1_000;
    private static final long PORTABLE_MAC = 100;
    private static final long PORTABLE_ELEMENT = 10;
    private static final long NATIVE_CONTROL_AND_DISPATCH = 100_000;
    private static final long PRIMITIVE_PREPARATION = 20_000;
    private static final long GRAPH_PREPARATION = 40_000;
    private final CpuOneDnnLibrary library;
    private final CpuOneDnnRoutePlan.Api preference;
    private final CpuTraceProducer trace;

    private record CopyKey(ValueId value, long offset, long byteSize) { }

    /**
     * Freezes one library session and API policy for cold complete-plan comparison.
     *
     * @param library non-null exact library borrowed from the integration owner
     * @param preference non-null AUTO/PRIMITIVE/GRAPH policy, not a command to choose native code
     * @throws NullPointerException if either argument is null
     */
    public CpuOneDnnSelector(CpuOneDnnLibrary library, CpuOneDnnRoutePlan.Api preference) {
        this(library, preference, null);
    }

    /**
     * Freezes the same policy with optional passive cold diagnostics.
     * @param library non-null borrowed exact library
     * @param preference non-null native API preference, never forced selection
     * @param trace optional caller-owned cold producer, or null for no diagnostic work
     */
    public CpuOneDnnSelector(CpuOneDnnLibrary library, CpuOneDnnRoutePlan.Api preference,
            CpuTraceProducer trace) {
        this.library = Objects.requireNonNull(library, "library");
        this.preference = Objects.requireNonNull(preference, "preference");
        this.trace = trace;
    }

    /**
     * Compares the explicit ordinary-contraction alternative with the exact portable/OpenBLAS peer.
     * Analysis copies descriptor facts only; temporary Graph compilation is closed before return.
     * Retained resources are declarations, not live handles or data, and inconclusive/overflowing
     * profitability leaves the baseline unchanged rather than deferring a decision to execution.
     *
     * @param context non-null complete partition analysis facts, borrowed without mutation
     * @param baseline non-null retained portable/OpenBLAS peer, returned unchanged when not displaced
     * @param contractionCandidate non-null ordinary common-unit alternative retaining saved/public
     *     stores; not an already-selected Phase-2 topology to replace
     * @return non-null baseline or complete selected analysis bound to this exact library session
     * @throws NullPointerException if an argument is null
     * @throws IllegalStateException if the library is closed or an unexpected native query fails
     * @throws IllegalArgumentException if retained numeric or resource declarations are inconsistent
     */
    public BackendPartitionAnalysis<CpuPartitionPreparationPlan> select(
            PrepareContext<CpuPartitionAnalysisInputs> context,
            BackendPartitionAnalysis<CpuPartitionPreparationPlan> baseline,
            BackendPartitionAnalysis<CpuPartitionPreparationPlan> contractionCandidate) {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(baseline, "baseline");
        Objects.requireNonNull(contractionCandidate, "contractionCandidate");
        library.requireOpen();
        if (trace != null) trace.compared();
        var plan = contractionCandidate.plan();
        if (context.requiresBatchSize() || plan.route() != CpuPartitionPreparationPlan.Route.PORTABLE
                || plan.openBlasTuningBatch().isPresent() || plan.materialization().isPresent()
                || !plan.materializations().isEmpty() || plan.partialReductionRecipe().isPresent()) {
            if (trace != null) trace.rejected(-1, null, null, Reason.COMPARISON_SKIPPED,
                    "candidate topology is not eligible for native comparison", null);
            return baseline;
        }
        try {
            return selectChecked(context, baseline, contractionCandidate);
        } catch (ArithmeticException unrepresentableCost) {
            // Checked Model geometry is still valid. An overflowing heuristic cannot qualify
            // a new representation or displace an already valid complete CPU plan.
            if (trace != null) trace.rejected(-1, null, null, Reason.COST_OVERFLOW,
                    "dimensionless native comparison cost is not representable", null);
            return baseline;
        }
    }

    private BackendPartitionAnalysis<CpuPartitionPreparationPlan> selectChecked(
            PrepareContext<CpuPartitionAnalysisInputs> context,
            BackendPartitionAnalysis<CpuPartitionPreparationPlan> baseline,
            BackendPartitionAnalysis<CpuPartitionPreparationPlan> candidate) {
        var plan = candidate.plan();
        long runs = context.backendInputs().materializationPolicy().expectedRunCount();
        long arithmetic = arithmeticCost(plan.units());
        long baselineCost = baseline.plan().openBlasPlan().isPresent()
                ? baseline.plan().openBlasPlan().orElseThrow().openBlasCost()
                : Math.addExact(Math.multiplyExact(arithmetic, runs), overhead(baseline, runs));
        long selectedCost = Math.addExact(Math.multiplyExact(arithmetic, runs), overhead(candidate, runs));
        Set<ValueId> nativeValues = new HashSet<>(context.constants().keySet());
        context.memoryRequirements().stream()
                .filter(requirement -> requirement.producerPartition().isPresent())
                .forEach(requirement -> nativeValues.add(requirement.valueId()));
        long nextWorkspace = plan.units().size();
        for (var requirement : candidate.requirements()) {
            if (requirement instanceof PreparationResourceRequirement.Workspace workspace) {
                nextWorkspace = Math.max(nextWorkspace, Math.incrementExact(workspace.requirementId()));
            }
        }
        var selected = new ArrayList<CpuOneDnnRoutePlan.Unit>();
        var sharedCopies = new java.util.LinkedHashMap<CopyKey, PreparationResourceRequirement.Workspace>();
        for (int position = 0; position < plan.units().size(); position++) {
            var unit = plan.units().get(position);
            var projection = CpuOneDnnProjection.from(unit);
            if (projection.isEmpty() || !CpuOneDnnCompiler.qualified(projection.orElseThrow().spec())) {
                if (trace != null) trace.rejected(position,
                        projection.isEmpty() ? null : projection.orElseThrow().spec(), null,
                        Reason.UNQUALIFIED, "ordinary unit has no qualified native realization", null);
                continue;
            }
            var projected = projection.orElseThrow();
            var copiedInputs = new java.util.LinkedHashMap<CopyKey, List<Integer>>();
            for (int input = 0; input < projected.inputBoundaryPositions().size(); input++) {
                ValueId value = unit.boundaryValues().get(projected.inputBoundaryPositions().get(input));
                if (!expectedNative(context, plan, value, nativeValues)) {
                    var key = new CopyKey(value, projected.byteOffsets().get(input),
                            projected.spec().inputs().get(input).byteSize());
                    copiedInputs.computeIfAbsent(key, ignored -> new ArrayList<>()).add(input);
                }
            }
            var copies = new ArrayList<CpuOneDnnRoutePlan.InputCopy>();
            var freshCopies = new java.util.LinkedHashMap<CopyKey, PreparationResourceRequirement.Workspace>();
            long copyBytes = 0;
            long workspaceId = nextWorkspace;
            for (var entry : copiedInputs.entrySet()) {
                var workspace = sharedCopies.get(entry.getKey());
                boolean refresh = workspace == null;
                if (refresh) {
                    workspace = new PreparationResourceRequirement.Workspace(workspaceId++,
                            entry.getKey().byteSize(), 64);
                    freshCopies.put(entry.getKey(), workspace);
                    copyBytes = Math.addExact(copyBytes, workspace.byteSize());
                }
                copies.add(new CpuOneDnnRoutePlan.InputCopy(entry.getValue(), workspace, refresh));
            }
            Optional<PreparationResourceRequirement.Workspace> outputStaging = projected.requiresOutputNarrowing()
                    ? Optional.of(new PreparationResourceRequirement.Workspace(workspaceId++,
                            projected.spec().output().byteSize(), 64)) : Optional.empty();
            CpuOneDnnRoutePlan.Unit best = null;
            long bestCost = Long.MAX_VALUE;
            for (var api : apis()) {
                Optional<CpuOneDnnKernelMetadata> metadata = describe(position, projected.spec(), api);
                if (metadata.isEmpty()) continue;
                var facts = metadata.orElseThrow();
                long steps = projected.spec() instanceof CpuOneDnnKernelSpec.Pipeline pipeline
                        ? pipeline.steps().size() : 1;
                long perRun = Math.addExact(Math.multiplyExact(NATIVE_CONTROL_AND_DISPATCH, steps),
                        Math.addExact(macs(projected.spec()), elements(projected.spec().output())));
                perRun = Math.addExact(perRun, Math.addExact(copyBytes, facts.scratchpadBytes()));
                if (outputStaging.isPresent()) {
                    perRun = Math.addExact(perRun, Math.addExact(outputStaging.orElseThrow().byteSize(),
                            Math.addExact(projected.logicalOutputByteSize(),
                                    Math.multiplyExact(PORTABLE_ELEMENT, elements(projected.spec().output())))));
                    perRun = Math.addExact(perRun, 1_000L);
                }
                // Each declared workspace includes allocation, native control creation and cleanup.
                perRun = Math.addExact(perRun, Math.multiplyExact(1_000L, freshCopies.size() + 1L));
                long cost = Math.addExact(Math.multiplyExact(perRun, runs),
                        api == CpuOneDnnRoutePlan.Api.PRIMITIVE ? PRIMITIVE_PREPARATION : GRAPH_PREPARATION);
                if (cost < bestCost) {
                    bestCost = cost;
                    best = new CpuOneDnnRoutePlan.Unit(position, projected, api,
                            facts.scratchpadBytes(), facts.implementation(), copies,
                            new PreparationResourceRequirement.Workspace(workspaceId,
                                    facts.scratchpadBytes(), 64), outputStaging);
                }
            }
            long portableUnit = Math.multiplyExact(
                    Math.addExact(arithmeticCost(unit), PORTABLE_DISPATCH), runs);
            if (best != null && bestCost < portableUnit) {
                selected.add(best);
                sharedCopies.putAll(freshCopies);
                selectedCost = Math.addExact(Math.subtractExact(selectedCost, portableUnit), bestCost);
                nextWorkspace = Math.incrementExact(workspaceId);
            }
            else if (best != null && trace != null) {
                trace.rejected(position, projected.spec(), best.api(), Reason.UNIT_COST,
                        "native unit heuristic does not beat the ordinary unit", null);
            }
        }
        if (selected.isEmpty()) return baseline;
        // Common recipes remain available for semantic authentication, but their unused ordinary
        // accumulator state is not another native allocation. Only selected owners are removed.
        var selectedPositions = new HashSet<Integer>();
        var replacedWorkspaces = new HashSet<PreparationResourceRequirement.Workspace>();
        for (var unit : selected) {
            selectedPositions.add(unit.unitPosition());
            plan.units().get(unit.unitPosition()).runtimeFacts().workspaceDeclaration()
                    .ifPresent(replacedWorkspaces::add);
        }
        for (int position = 0; position < plan.units().size(); position++) {
            if (!selectedPositions.contains(position)) {
                plan.units().get(position).runtimeFacts().workspaceDeclaration()
                        .ifPresent(replacedWorkspaces::remove);
            }
        }
        var requirements = new ArrayList<PreparationResourceRequirement>();
        for (var requirement : candidate.requirements()) {
            if (requirement instanceof PreparationResourceRequirement.Workspace workspace
                    && replacedWorkspaces.contains(workspace)) {
                selectedCost = Math.subtractExact(selectedCost, Math.multiplyExact(workspace.byteSize(), runs));
            } else requirements.add(requirement);
        }
        if (selectedCost >= baselineCost) {
            if (trace != null) trace.rejected(-1, null, null, Reason.PLAN_COST,
                    "complete native candidate heuristic does not beat the retained baseline", null);
            return baseline;
        }
        var route = new CpuOneDnnRoutePlan(library.sessionIdentity(), selected, baselineCost, selectedCost);
        requirements.addAll(route.workspaceRequirements());
        return new BackendPartitionAnalysis<>(candidate.partition(), plan.withOneDnn(route), requirements);
    }

    private Optional<CpuOneDnnKernelMetadata> describe(int position, CpuOneDnnKernelSpec spec,
            CpuOneDnnRoutePlan.Api api) {
        try {
            return Optional.of(CpuOneDnnCompiler.describe(library, spec, api));
        } catch (CpuOneDnnCompiler.UnsupportedRealizationException unsupported) {
            if (trace != null) trace.rejected(position, spec, api, Reason.UNSUPPORTED_REALIZATION,
                    unsupported.getMessage(), null);
            return Optional.empty();
        } catch (CpuOneDnnLibrary.NativeStatusException failure) {
            if (failure.status() == 2 || failure.status() == 3) {
                if (trace != null) trace.rejected(position, spec, api, Reason.NATIVE_UNSUPPORTED,
                        failure.getMessage(), failure.status());
                return Optional.empty();
            }
            throw failure;
        }
    }

    private List<CpuOneDnnRoutePlan.Api> apis() {
        return preference == CpuOneDnnRoutePlan.Api.AUTO
                ? List.of(CpuOneDnnRoutePlan.Api.PRIMITIVE, CpuOneDnnRoutePlan.Api.GRAPH)
                : List.of(preference);
    }

    private static boolean expectedNative(PrepareContext<CpuPartitionAnalysisInputs> context,
            CpuPartitionPreparationPlan plan, ValueId value, Set<ValueId> producedOrConstant) {
        var facts = context.backendInputs().boundaryStorageFacts();
        if (!facts.isEmpty()) {
            int position = plan.boundaryValues().indexOf(value);
            if (facts.size() == plan.boundaryValues().size() && position >= 0) {
                return facts.get(position).nativeSegment();
            }
        }
        return producedOrConstant.contains(value);
    }

    private static long overhead(BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis,
            long runs) {
        long bytes = 0;
        for (var requirement : analysis.requirements()) {
            bytes = Math.addExact(bytes, switch (requirement) {
                case PreparationResourceRequirement.Buffer buffer -> buffer.byteSize();
                case PreparationResourceRequirement.Workspace workspace -> workspace.byteSize();
            });
        }
        return Math.multiplyExact(Math.addExact(bytes,
                Math.multiplyExact(PORTABLE_DISPATCH, analysis.plan().units().size())), runs);
    }

    private static long arithmeticCost(List<CpuPartitionPreparationPlan.ExecutionUnitPlan> units) {
        long total = 0;
        for (var unit : units) total = Math.addExact(total, arithmeticCost(unit));
        return total;
    }

    private static long arithmeticCost(CpuPartitionPreparationPlan.ExecutionUnitPlan unit) {
        if (unit.matmulGeometry().isPresent()) {
            var geometry = unit.matmulGeometry().orElseThrow();
            return Math.addExact(Math.multiplyExact(PORTABLE_MAC, Math.multiplyExact(
                    geometry.outputCount(), geometry.k())),
                    Math.multiplyExact(PORTABLE_ELEMENT, geometry.outputCount()));
        }
        var projected = CpuOneDnnProjection.from(unit);
        if (projected.isPresent()) return Math.addExact(
                Math.multiplyExact(projected.orElseThrow().spec() instanceof CpuOneDnnKernelSpec.Contraction
                        ? PORTABLE_MAC : PORTABLE_ELEMENT, macs(projected.orElseThrow().spec())),
                Math.multiplyExact(PORTABLE_ELEMENT, elements(projected.orElseThrow().spec().output())));
        long operations = unit.portablePlan() != null
                && unit.portablePlan().portableKernelIr() instanceof CpuKernelIr ir
                ? Math.max(1, ir.instructions().size()) : 1;
        return Math.multiplyExact(PORTABLE_ELEMENT, Math.multiplyExact(unit.elementCount(), operations));
    }

    private static long macs(CpuOneDnnKernelSpec spec) {
        if (spec instanceof CpuOneDnnKernelSpec.Pipeline pipeline) {
            long result = 0;
            for (var step : pipeline.steps()) result = Math.addExact(result, macs(step.spec()));
            return result;
        }
        if (spec instanceof CpuOneDnnKernelSpec.Contraction contraction) {
            long contributors = contraction.inputs().getFirst().dimensions().getLast();
            if (contraction.operation() == CpuOneDnnKernelSpec.Operation.CONVOLUTION) {
                contributors = 1;
                var dimensions = contraction.inputs().get(1).dimensions();
                for (int axis = 1; axis < dimensions.size(); axis++) {
                    contributors = Math.multiplyExact(contributors, dimensions.get(axis));
                }
            }
            return Math.multiplyExact(elements(spec.output()), contributors);
        }
        if (spec instanceof CpuOneDnnKernelSpec.Reduction reduction) return elements(reduction.input());
        if (spec instanceof CpuOneDnnKernelSpec.Pooling pooling) {
            long contributors = 1;
            for (long extent : pooling.kernel()) contributors = Math.multiplyExact(contributors, extent);
            return Math.multiplyExact(elements(pooling.output()), contributors);
        }
        return elements(spec.output());
    }

    private static long elements(CpuOneDnnKernelSpec.Tensor tensor) {
        long result = 1;
        for (long dimension : tensor.dimensions()) result = Math.multiplyExact(result, dimension);
        return result;
    }

}

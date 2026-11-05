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
    private static final int MAX_FINITE_ALTERNATIVES = 6_560;
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
                || plan.localTuningBatch().isPresent() || plan.materialization().isPresent()
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

    private record Choice(int position, CpuOneDnnProjection projection,
            CpuOneDnnRoutePlan.Api api, CpuOneDnnKernelMetadata metadata) { }

    /**
     * Enumerates every qualified native-containing assignment of this retained ordinary topology.
     * Each unit independently retains its ordinary realization or uses an available concrete API
     * allowed by the integration policy. Encounter order is ordinary, primitive, then Graph at each
     * topological position; the all-ordinary assignment is supplied by the caller, not repeated.
     * For unit API counts {@code a[i]}, this returns {@code product(1 + a[i]) - 1} analyses.
     * Neither per-unit nor complete-plan heuristic costs prune this explicit tuning domain.
     * Queries retain no handles; every analysis declares its exact ingress, staging and scratch.
     * At most 6,560 native-containing assignments are admitted (the full eight-unit AUTO domain).
     * A larger domain is rejected before constructing any alternatives, never silently prefixed.
     *
     * @param context non-null borrowed partition facts
     * @param portable non-null ordinary common-unit analysis without preselected native routes,
     *     materializations, partial reductions or local OpenBLAS batch
     * @return immutable complete native-containing alternatives, empty for ineligible topology
     * @throws NullPointerException if an argument is null
     * @throws IllegalStateException if the library is closed or an unexpected native query fails
     * @throws IllegalArgumentException if descriptors/declarations are inconsistent or the complete
     *     supported assignment domain exceeds 6,560 alternatives
     * @throws ArithmeticException if exact resource geometry is unrepresentable
     */
    public List<BackendPartitionAnalysis<CpuPartitionPreparationPlan>> alternatives(
            PrepareContext<CpuPartitionAnalysisInputs> context,
            BackendPartitionAnalysis<CpuPartitionPreparationPlan> portable) {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(portable, "portable");
        library.requireOpen();
        if (!enumerable(context, portable.plan())) return List.of();
        var domain = qualifiedDomain(portable.plan());
        var result = new ArrayList<BackendPartitionAnalysis<CpuPartitionPreparationPlan>>(domain.count());
        enumerate(context, portable, domain.choices(), 0, new ArrayList<>(), result);
        return List.copyOf(result);
    }

    /**
     * Preflights the exact supported assignment cardinality without constructing realization plans.
     * This performs the same qualified native metadata queries as enumeration and retains no native
     * handles. Callers can sum complete topology domains before allocating multiplied candidate
     * lists. The all-ordinary assignment is not counted.
     *
     * @param context non-null borrowed partition facts
     * @param portable non-null clean ordinary topology analysis, with enumeration's eligibility rules
     * @return exact native-containing count in 0..6,560
     * @throws NullPointerException if an argument is null
     * @throws IllegalStateException if the library is closed or an unexpected native query fails
     * @throws IllegalArgumentException if the full domain exceeds 6,560 or descriptor facts disagree
     * @throws ArithmeticException if exact descriptor geometry is unrepresentable
     */
    public long alternativeCount(PrepareContext<CpuPartitionAnalysisInputs> context,
            BackendPartitionAnalysis<CpuPartitionPreparationPlan> portable) {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(portable, "portable");
        library.requireOpen();
        return enumerable(context, portable.plan()) ? qualifiedDomain(portable.plan()).count() : 0;
    }

    /**
     * Counts qualified available API choices at every retained common-unit position, without
     * constructing assignment realizations. Ordinary implementations are deliberately not counted,
     * so the CPU caller can preflight {@code product(portableChoices[i] + nativeChoices[i])}
     * without multiplying portable forms that a native assignment would replace.
     *
     * @param context non-null borrowed partition facts
     * @param portable non-null clean ordinary topology analysis
     * @return immutable list matching the common-unit count; entries are 0..2 (0..1 for a concrete
     *     API policy), all zero when the ordinary topology is ineligible
     * @throws NullPointerException if an argument is null
     * @throws IllegalStateException if the library is closed or an unexpected native query fails
     * @throws IllegalArgumentException if the full native domain exceeds 6,560 or descriptors disagree
     * @throws ArithmeticException if exact descriptor geometry is unrepresentable
     */
    public List<Integer> alternativeCountsByUnit(
            PrepareContext<CpuPartitionAnalysisInputs> context,
            BackendPartitionAnalysis<CpuPartitionPreparationPlan> portable) {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(portable, "portable");
        library.requireOpen();
        var counts = new ArrayList<>(java.util.Collections.nCopies(portable.plan().units().size(), 0));
        if (enumerable(context, portable.plan())) {
            for (var choices : qualifiedDomain(portable.plan()).choices()) {
                counts.set(choices.getFirst().position(), choices.size());
            }
        }
        return List.copyOf(counts);
    }

    private record Domain(List<List<Choice>> choices, int count) { }

    private Domain qualifiedDomain(CpuPartitionPreparationPlan plan) {
        var domains = new ArrayList<List<Choice>>();
        int assignmentCount = 1;
        for (int position = 0; position < plan.units().size(); position++) {
            var choices = new ArrayList<Choice>();
            var projection = CpuOneDnnProjection.from(plan.units().get(position));
            if (projection.isPresent() && CpuOneDnnCompiler.qualified(projection.orElseThrow().spec())) {
                var projected = projection.orElseThrow();
                for (var api : apis()) {
                    var metadata = describe(position, projected.spec(), api);
                    if (metadata.isPresent()) {
                        choices.add(new Choice(position, projected, api, metadata.orElseThrow()));
                    }
                }
            }
            if (!choices.isEmpty()) {
                int factor = choices.size() + 1;
                if (assignmentCount > (MAX_FINITE_ALTERNATIVES + 1) / factor) {
                    throw new IllegalArgumentException("complete oneDNN assignment domain exceeds "
                            + MAX_FINITE_ALTERNATIVES + " alternatives");
                }
                assignmentCount *= factor;
                domains.add(List.copyOf(choices));
            }
        }
        return new Domain(List.copyOf(domains), assignmentCount - 1);
    }

    /**
     * Reauthenticates a Phase-1 assignment without searching APIs or changing ordinary/native units.
     * Only the selected concrete APIs are queried. A disappeared implementation, changed descriptor,
     * ingress declaration, scratch size, session or policy is an error, never heuristic substitution.
     * Finalization still compiles and checks the retained metadata under the existing protocol.
     *
     * @param context non-null fresh borrowed analysis facts
     * @param portable non-null fresh ordinary analysis of the selected common topology
     * @param selectedNativePlan non-null Phase-1 native realization from this selector's session
     * @return fresh complete analysis with the identical frozen native assignment and declarations
     * @throws NullPointerException if an argument is null
     * @throws IllegalArgumentException if topology, session, policy or exact resources differ
     * @throws IllegalStateException if a frozen realization is unavailable or a native query fails
     * @throws ArithmeticException if exact resource geometry is unrepresentable
     */
    public BackendPartitionAnalysis<CpuPartitionPreparationPlan> freeze(
            PrepareContext<CpuPartitionAnalysisInputs> context,
            BackendPartitionAnalysis<CpuPartitionPreparationPlan> portable,
            CpuPartitionPreparationPlan selectedNativePlan) {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(portable, "portable");
        Objects.requireNonNull(selectedNativePlan, "selectedNativePlan");
        library.requireOpen();
        var selected = selectedNativePlan.oneDnnPlan().orElseThrow(
                () -> new IllegalArgumentException("frozen plan has no oneDNN assignment"));
        if (!enumerable(context, portable.plan())
                || !library.sessionIdentity().equals(selected.sessionIdentity())) {
            throw new IllegalArgumentException("frozen oneDNN topology or library session differs");
        }
        selected.validate(portable.plan().units());
        var choices = new ArrayList<Choice>(selected.units().size());
        for (var unit : selected.units()) {
            if (!apis().contains(unit.api())
                    || !CpuOneDnnCompiler.qualified(unit.projection().spec())) {
                throw new IllegalArgumentException("frozen oneDNN API or qualification differs");
            }
            var metadata = describe(unit.unitPosition(), unit.projection().spec(), unit.api())
                    .orElseThrow(() -> new IllegalStateException("frozen oneDNN realization is unavailable"));
            choices.add(new Choice(unit.unitPosition(), unit.projection(), unit.api(), metadata));
        }
        var fresh = realize(context, portable, choices);
        if (!fresh.plan().oneDnnPlan().orElseThrow().units().equals(selected.units())) {
            throw new IllegalArgumentException("frozen oneDNN descriptors or resource declarations differ");
        }
        return fresh;
    }

    /**
     * Identifies an exact native assignment for the CPU-owned session-scoped tuning envelope.
     * Immutable descriptor records include literal constants, origins, boundary roles, logical low
     * result types, implementation identities, shared-copy refresh owners and exact workspaces.
     * Heuristic costs are deliberately absent: explicit alternatives are unscored (zero/zero).
     * Typed fields are streamed into a bounded canonical SHA-256 encoding, without descriptor
     * strings or an aggregate byte payload. The result is always 87 ASCII bytes; even the maximal
     * domain contributes only 570,720 fingerprint bytes before outer envelope overhead.
     *
     * @param plan non-null plan containing a native assignment owned by this selector
     * @return fixed 87-byte ASCII identity including library session and API policy
     * @throws NullPointerException if plan is null
     * @throws IllegalArgumentException if the plan is ordinary, belongs to another library session,
     *     or its canonical identity source exceeds one MiB
     */
    public String candidateFingerprint(CpuPartitionPreparationPlan plan) {
        Objects.requireNonNull(plan, "plan");
        var route = plan.oneDnnPlan().orElseThrow(
                () -> new IllegalArgumentException("candidate has no oneDNN assignment"));
        if (!library.sessionIdentity().equals(route.sessionIdentity())) {
            throw new IllegalArgumentException("candidate belongs to another oneDNN library session");
        }
        return CpuOneDnnCandidateIdentity.fingerprint(preference, route);
    }

    private static boolean enumerable(PrepareContext<CpuPartitionAnalysisInputs> context,
            CpuPartitionPreparationPlan plan) {
        return !context.requiresBatchSize() && plan.route() == CpuPartitionPreparationPlan.Route.PORTABLE
                && plan.localTuningBatch().isEmpty() && plan.materialization().isEmpty()
                && plan.materializations().isEmpty() && plan.partialReductionRecipe().isEmpty();
    }

    private void enumerate(PrepareContext<CpuPartitionAnalysisInputs> context,
            BackendPartitionAnalysis<CpuPartitionPreparationPlan> portable,
            List<List<Choice>> domains, int position, ArrayList<Choice> selected,
            ArrayList<BackendPartitionAnalysis<CpuPartitionPreparationPlan>> result) {
        if (position == domains.size()) {
            if (!selected.isEmpty()) result.add(realize(context, portable, selected));
            return;
        }
        enumerate(context, portable, domains, position + 1, selected, result);
        for (var choice : domains.get(position)) {
            selected.add(choice);
            enumerate(context, portable, domains, position + 1, selected, result);
            selected.removeLast();
        }
    }

    private BackendPartitionAnalysis<CpuPartitionPreparationPlan> realize(
            PrepareContext<CpuPartitionAnalysisInputs> context,
            BackendPartitionAnalysis<CpuPartitionPreparationPlan> portable, List<Choice> choices) {
        var plan = portable.plan();
        Set<ValueId> nativeValues = new HashSet<>(context.constants().keySet());
        context.memoryRequirements().stream()
                .filter(requirement -> requirement.producerPartition().isPresent())
                .forEach(requirement -> nativeValues.add(requirement.valueId()));
        long nextWorkspace = plan.units().size();
        for (var requirement : portable.requirements()) {
            if (requirement instanceof PreparationResourceRequirement.Workspace workspace) {
                nextWorkspace = Math.max(nextWorkspace, Math.incrementExact(workspace.requirementId()));
            }
        }
        var selected = new ArrayList<CpuOneDnnRoutePlan.Unit>(choices.size());
        var sharedCopies = new java.util.LinkedHashMap<CopyKey, PreparationResourceRequirement.Workspace>();
        var replacedWorkspaces = new HashSet<PreparationResourceRequirement.Workspace>();
        var selectedPositions = new HashSet<Integer>();
        for (var choice : choices) {
            var projected = choice.projection();
            var unit = plan.units().get(choice.position());
            var copiedInputs = new java.util.LinkedHashMap<CopyKey, List<Integer>>();
            for (int input = 0; input < projected.inputBoundaryPositions().size(); input++) {
                ValueId value = unit.boundaryValues().get(projected.inputBoundaryPositions().get(input));
                if (!expectedNative(context, plan, value, nativeValues)) {
                    var key = new CopyKey(value, projected.byteOffsets().get(input),
                            projected.spec().inputs().get(input).byteSize());
                    copiedInputs.computeIfAbsent(key, ignored -> new ArrayList<>()).add(input);
                }
            }
            var copies = new ArrayList<CpuOneDnnRoutePlan.InputCopy>(copiedInputs.size());
            for (var entry : copiedInputs.entrySet()) {
                var workspace = sharedCopies.get(entry.getKey());
                boolean refresh = workspace == null;
                if (refresh) {
                    workspace = new PreparationResourceRequirement.Workspace(nextWorkspace,
                            entry.getKey().byteSize(), 64);
                    nextWorkspace = Math.incrementExact(nextWorkspace);
                    sharedCopies.put(entry.getKey(), workspace);
                }
                copies.add(new CpuOneDnnRoutePlan.InputCopy(entry.getValue(), workspace, refresh));
            }
            Optional<PreparationResourceRequirement.Workspace> outputStaging = Optional.empty();
            if (projected.requiresOutputNarrowing()) {
                outputStaging = Optional.of(new PreparationResourceRequirement.Workspace(nextWorkspace,
                        projected.spec().output().byteSize(), 64));
                nextWorkspace = Math.incrementExact(nextWorkspace);
            }
            var invocation = new PreparationResourceRequirement.Workspace(nextWorkspace,
                    choice.metadata().scratchpadBytes(), 64);
            nextWorkspace = Math.incrementExact(nextWorkspace);
            selected.add(new CpuOneDnnRoutePlan.Unit(choice.position(), projected, choice.api(),
                    choice.metadata().scratchpadBytes(), choice.metadata().implementation(),
                    copies, invocation, outputStaging));
            selectedPositions.add(choice.position());
            unit.runtimeFacts().workspaceDeclaration().ifPresent(replacedWorkspaces::add);
        }
        for (int position = 0; position < plan.units().size(); position++) {
            if (!selectedPositions.contains(position)) {
                plan.units().get(position).runtimeFacts().workspaceDeclaration()
                        .ifPresent(replacedWorkspaces::remove);
            }
        }
        var requirements = new ArrayList<PreparationResourceRequirement>();
        for (var requirement : portable.requirements()) {
            if (!(requirement instanceof PreparationResourceRequirement.Workspace workspace)
                    || !replacedWorkspaces.contains(workspace)) requirements.add(requirement);
        }
        var route = new CpuOneDnnRoutePlan(library.sessionIdentity(), selected, 0, 0);
        requirements.addAll(route.workspaceRequirements());
        return new BackendPartitionAnalysis<>(portable.partition(), plan.withOneDnn(route), requirements);
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

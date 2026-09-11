package io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMaterializationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRepresentationPlanner;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.provider.openblas.OpenBlasLibrary;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Field-free cold producer and selector for exact/default FLOAT32/FLOAT64 OpenBLAS MATMUL
 * candidates. It derives one batch from common lowering and route facts, validates an optional
 * exact decision, and otherwise applies the unchanged safe heuristic. It performs no provider
 * query, host discovery, measurement, objective comparison, serialization, or cache I/O.
 */
public final class CpuOpenBlasRouteSelector {
    /** Creates a stateless selector with no resources or lifecycle. */
    public CpuOpenBlasRouteSelector() { }

    /**
     * Complete eligible cold outcome.
     * @param batch freshly generated candidate batch
     * @param selected selected member after decision or heuristic handling
     */
    public record Result(CpuOpenBlasTuningBatch batch,
            CpuOpenBlasTuningBatch.Candidate selected) {
        /**
         * Validates exact batch membership. The immutable batch and selected member are retained
         * without mutation or resource ownership.
         *
         * @throws NullPointerException if either component is {@code null}
         * @throws IllegalArgumentException if {@code selected} is not a member of {@code batch}
         */
        public Result {
            Objects.requireNonNull(batch, "batch");
            Objects.requireNonNull(selected, "selected");
            if (batch.find(selected.identity()).isEmpty()) {
                throw new IllegalArgumentException("selected candidate is outside batch");
            }
        }
        /**
         * Returns the selected native plan view without changing the selection.
         *
         * @return non-null optional containing the retained OpenBLAS plan, or empty for portable
         *     selection
         */
        public Optional<CpuOpenBlasRoutePlan> selectedOpenBlasPlan() {
            return selected.openBlasPlan();
        }
    }

    /**
     * Produces all valid candidates and applies one compatible decision or the safe heuristic.
     * @param context non-null complete CPU analysis projection
     * @param portablePlan non-null already-selected portable plan
     * @param representationPlanner non-null existing representation owner
     * @return complete outcome, or empty when the workload/configuration is ineligible
     * @throws NullPointerException if an argument is {@code null}
     */
    public Optional<Result> evaluate(PrepareContext<CpuPartitionAnalysisInputs> context,
            CpuPartitionPreparationPlan portablePlan,
            CpuRepresentationPlanner representationPlanner) {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(portablePlan, "portablePlan");
        Objects.requireNonNull(representationPlanner, "representationPlanner");
        try {
            return evaluateChecked(context, portablePlan, representationPlanner);
        } catch (ArithmeticException | IllegalArgumentException uncertain) {
            return Optional.empty();
        }
    }

    /**
     * Compatibility view of only the selected native plan.
     * @param context non-null complete CPU analysis projection; retained nowhere
     * @param portablePlan non-null already-selected portable plan; not mutated
     * @param representationPlanner non-null existing representation owner; not retained
     * @return non-null optional containing the selected immutable native plan, or empty when the
     *     workload is ineligible or portable is selected
     * @throws NullPointerException if an argument is {@code null}
     */
    public Optional<CpuOpenBlasRoutePlan> select(PrepareContext<CpuPartitionAnalysisInputs> context,
            CpuPartitionPreparationPlan portablePlan,
            CpuRepresentationPlanner representationPlanner) {
        return evaluate(context, portablePlan, representationPlanner)
                .flatMap(Result::selectedOpenBlasPlan);
    }

    private static Optional<Result> evaluateChecked(
            PrepareContext<CpuPartitionAnalysisInputs> context,
            CpuPartitionPreparationPlan plan, CpuRepresentationPlanner representationPlanner) {
        var config = context.backendInputs().openBlasRoute();
        if (!config.complete() || plan.route() != CpuPartitionPreparationPlan.Route.PORTABLE
                || context.nodes().size() != 1 || plan.units().size() != 1
                || context.nodes().getFirst().operation().kind()
                    != io.github.pho001.synaptik.model.operation.linalg.MatmulKind.MATMUL
                || plan.bufferDeclarations().size() != 3 || !plan.materializations().isEmpty()
                || plan.partialReductionRecipe().isPresent()) return Optional.empty();
        var unit = plan.units().getFirst();
        Optional<CpuMatmulIr> maybeIr = unit.portablePlan().specialization().matmulIr();
        if (maybeIr.isEmpty() || unit.matmulGeometry().isEmpty()
                || !(unit.portablePlan().portableKernelIr()
                    instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr)
                || unit.outputCount() != 1 || unit.boundaryValues().size() != 3) {
            return Optional.empty();
        }
        CpuMatmulIr ir = maybeIr.orElseThrow();
        var geometry = unit.matmulGeometry().orElseThrow();
        DataType type = ir.resultType();
        if (!ir.epilogue().equals(CpuMatmulIr.Epilogue.none()) || ir.leftType() != type
                || ir.rightType() != type
                || type != DataType.FLOAT32 && type != DataType.FLOAT64
                || unit.portablePlan().specialization().numericalMode()
                    != CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT
                || geometry.batchExtents().length != 0 || geometry.batchCount() != 1
                || geometry.removedM() || geometry.removedN() || geometry.m() <= 0
                || geometry.n() <= 0 || geometry.k() <= 0 || geometry.m() > Integer.MAX_VALUE
                || geometry.n() > Integer.MAX_VALUE || geometry.k() > Integer.MAX_VALUE) {
            return Optional.empty();
        }
        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        for (GraphValue value : context.values()) values.put(value.id(), value);
        for (ValueId boundary : unit.boundaryValues()) {
            GraphValue value = values.get(boundary);
            if (value == null || value.descriptor().shape().rank() != 2
                    || value.descriptor().layout().isEmpty()) return Optional.empty();
        }
        int m = Math.toIntExact(geometry.m()), n = Math.toIntExact(geometry.n());
        int k = Math.toIntExact(geometry.k());
        if ((!canonical(unit.accessBindings().get(0), m, k, CpuAccessPlan.AccessKind.READ)
                && !copyable(unit.accessBindings().get(0), m, k))
                || (!canonical(unit.accessBindings().get(1), k, n, CpuAccessPlan.AccessKind.READ)
                    && !copyable(unit.accessBindings().get(1), k, n))
                || (!canonical(unit.accessBindings().get(2), m, n, CpuAccessPlan.AccessKind.WRITE)
                    && !copyableOutput(unit.accessBindings().get(2), m, n))) return Optional.empty();
        List<CpuPartitionAnalysisInputs.BoundaryStorageFact> storage =
                context.backendInputs().boundaryStorageFacts();
        if (storage.size() != 3) return Optional.empty();
        boolean leftDirect = canonical(unit.accessBindings().get(0), m, k,
                CpuAccessPlan.AccessKind.READ) && directBoundary(unit, storage, 0, type);
        boolean rightDirect = canonical(unit.accessBindings().get(1), k, n,
                CpuAccessPlan.AccessKind.READ) && directBoundary(unit, storage, 1, type);
        boolean outputDirect = canonical(unit.accessBindings().get(2), m, n,
                CpuAccessPlan.AccessKind.WRITE) && directBoundary(unit, storage, 2, type);
        long outputs = Math.multiplyExact((long) m, n);
        long macs = Math.multiplyExact(outputs, k);
        long runs = context.backendInputs().materializationPolicy().expectedRunCount();
        long portableCost = total(config.portableCosts(), outputs, macs, runs);
        int capacity = context.backendInputs().portableExecution().availableParallelism();
        var candidates = new ArrayList<CpuOpenBlasTuningBatch.Candidate>();
        candidates.add(new CpuOpenBlasTuningBatch.Candidate(
                CpuOpenBlasTuningBatch.RouteKind.PORTABLE,
                new CpuOpenBlasTuningBatch.PortableIdentity(plan.executionStrategy(),
                        plan.selectedRangeCount(), plan.minimumElementsPerWorker(),
                        plan.vectorSpeciesBitSize(),
                        unit.portablePlan().specialization().classIdentitySchema(),
                        unit.portablePlan().portableKernelIr().structuralKey(),
                        plan.bufferDeclarations().stream().map(value ->
                                new CpuOpenBlasTuningBatch.ResourceIdentity(value.byteSize(),
                                        value.byteAlignment())).toList()), Optional.empty()));
        for (var representation : CpuOpenBlasRoutePlan.Representation.values()) {
            if (!representation.copiesLeft() && !leftDirect
                    || !representation.copiesRight() && !rightDirect
                    || !representation.copiesOutput() && !outputDirect) continue;
            Optional<CpuMaterializationPlan> left = representation.copiesLeft()
                    ? representationPlanner.openBlasInputMaterialization(plan, 0,
                            context.backendInputs().materializationPolicy(), 8) : Optional.empty();
            int next = 8 + (representation.copiesLeft() ? 1 : 0);
            Optional<CpuMaterializationPlan> right = representation.copiesRight()
                    ? representationPlanner.openBlasInputMaterialization(plan, 1,
                            context.backendInputs().materializationPolicy(), next++) : Optional.empty();
            Optional<CpuOpenBlasOutputCopyPlan> output = representation.copiesOutput()
                    ? representationPlanner.openBlasOutputCopy(plan, next) : Optional.empty();
            if (representation.copiesLeft() != left.isPresent()
                    || representation.copiesRight() != right.isPresent()
                    || representation.copiesOutput() != output.isPresent()) continue;
            append(candidates, representation, left, right, output, type, m, n, k, config,
                    outputs, macs, portableCost, runs,
                    context.backendInputs().materializationPolicy().maximumAdditionalBytes(),
                    capacity);
        }
        if (candidates.size() == 1) return Optional.empty();
        var batch = new CpuOpenBlasTuningBatch(CpuOpenBlasTuningBatch.SCHEMA_VERSION,
                workload(context, plan, values, config, type, storage, capacity), candidates);
        var decided = context.backendInputs().openBlasTuningDecision()
                .flatMap(decision -> decision.match(batch));
        return Optional.of(new Result(batch, decided.orElseGet(() -> heuristic(batch, config))));
    }

    private static CpuOpenBlasTuningBatch.WorkloadSignature workload(
            PrepareContext<CpuPartitionAnalysisInputs> context,
            CpuPartitionPreparationPlan plan, Map<ValueId, GraphValue> values,
            CpuPartitionAnalysisInputs.OpenBlasRouteConfig config, DataType type,
            List<CpuPartitionAnalysisInputs.BoundaryStorageFact> storage, int capacity) {
        var unit = plan.units().getFirst();
        var boundaries = new ArrayList<CpuOpenBlasTuningBatch.BoundarySignature>(3);
        for (int index = 0; index < 3; index++) {
            var descriptor = values.get(unit.boundaryValues().get(index)).descriptor();
            boundaries.add(new CpuOpenBlasTuningBatch.BoundarySignature(type, descriptor.shape(),
                    descriptor.layout().orElseThrow(), unit.carrierPattern().get(index),
                    storage.get(index), unit.accessBindings().get(index)));
        }
        return new CpuOpenBlasTuningBatch.WorkloadSignature(
                CpuOpenBlasTuningBatch.OperationKind.MATMUL,
                CpuOpenBlasTuningBatch.OperationAttributes.NONE, type, type, type,
                type, boundaries, CpuOpenBlasTuningBatch.NumericalMode.EXACT_DEFAULT,
                CpuOpenBlasTuningBatch.DeterminismMode.DEFAULT,
                new CpuOpenBlasTuningBatch.QualificationScope(config.qualification().orElseThrow()),
                context.backendInputs().cpuHardwareIdentity(), capacity,
                context.backendInputs().portableExecution(), plan.executionStrategy(),
                plan.selectedRangeCount(), plan.vectorSpeciesBitSize(),
                config.threadCandidates().stream().map(
                        CpuPartitionAnalysisInputs.OpenBlasRouteConfig.ThreadCandidate::threadCount)
                        .toList(), context.backendInputs().workloadCohort(),
                context.backendInputs().materializationPolicy(), config.portableCosts(),
                config.representationCosts(), config.threadCandidates(),
                config.minimumNetBenefitCostUnits().orElseThrow(),
                config.minimumBenefitBasisPoints().orElseThrow(),
                unit.portablePlan().specialization().classIdentitySchema(),
                CpuOpenBlasTuningBatch.ROUTE_POLICY_VERSION,
                CpuOpenBlasTuningBatch.COST_POLICY_VERSION);
    }

    private static void append(List<CpuOpenBlasTuningBatch.Candidate> candidates,
            CpuOpenBlasRoutePlan.Representation representation,
            Optional<CpuMaterializationPlan> left, Optional<CpuMaterializationPlan> right,
            Optional<CpuOpenBlasOutputCopyPlan> output, DataType type, int m, int n, int k,
            CpuPartitionAnalysisInputs.OpenBlasRouteConfig config, long outputs, long macs,
            long portableCost, long runs, long byteCeiling, int capacity) {
        var requirements = new ArrayList<PreparationResourceRequirement.Workspace>();
        left.map(CpuOpenBlasRouteSelector::workspace).ifPresent(requirements::add);
        right.map(CpuOpenBlasRouteSelector::workspace).ifPresent(requirements::add);
        output.map(CpuOpenBlasRouteSelector::workspace).ifPresent(requirements::add);
        long bytes = requirements.stream().mapToLong(
                PreparationResourceRequirement.Workspace::byteSize).reduce(0, Math::addExact);
        if (bytes > byteCeiling) return;
        long inputElements = Math.addExact(left.map(CpuMaterializationPlan::elementCount).orElse(0L),
                right.map(CpuMaterializationPlan::elementCount).orElse(0L));
        long outputElements = output.map(CpuOpenBlasOutputCopyPlan::elementCount).orElse(0L);
        long representationCost = representationCost(config.representationCosts(),
                requirements.size(), bytes, (left.isPresent() ? 1 : 0) + (right.isPresent() ? 1 : 0),
                inputElements, output.isPresent() ? 1 : 0, outputElements);
        for (int order = 0; order < config.threadCandidates().size(); order++) {
            var thread = config.threadCandidates().get(order);
            if (thread.threadCount() > capacity) continue;
            long nativeCost = Math.multiplyExact(runs, Math.addExact(
                    perRun(thread.openBlasCosts(), outputs, macs), representationCost));
            long benefit = Math.subtractExact(portableCost, nativeCost);
            int basis = portableCost == 0 ? 0 : Math.toIntExact(Math.floorDiv(
                    Math.multiplyExact(10_000L, benefit), portableCost));
            var routePlan = new CpuOpenBlasRoutePlan(type, m, n, k, 0, 1, 2, thread,
                    thread.threadCount(), thread.threadCount(), capacity, order, representation,
                    left, right, output, requirements, runs, bytes, inputElements, outputElements,
                    portableCost, nativeCost, benefit, basis, config.qualification());
            var identity = new CpuOpenBlasTuningBatch.OpenBlasIdentity(representation,
                    thread.threadCount(), thread.threadCount(), order,
                    requirements.stream().map(
                            PreparationResourceRequirement.Workspace::requirementId).toList(),
                    bytes, inputElements, outputElements);
            candidates.add(new CpuOpenBlasTuningBatch.Candidate(
                    CpuOpenBlasTuningBatch.RouteKind.OPENBLAS, identity, Optional.of(routePlan)));
        }
    }

    private static CpuOpenBlasTuningBatch.Candidate heuristic(CpuOpenBlasTuningBatch batch,
            CpuPartitionAnalysisInputs.OpenBlasRouteConfig config) {
        CpuOpenBlasTuningBatch.Candidate selected = batch.candidates().getFirst();
        CpuOpenBlasRoutePlan best = null;
        for (var candidate : batch.candidates()) {
            if (candidate.route() != CpuOpenBlasTuningBatch.RouteKind.OPENBLAS) continue;
            CpuOpenBlasRoutePlan plan = candidate.openBlasPlan().orElseThrow();
            if (plan.openBlasCost() >= plan.portableCost()
                    || plan.netBenefit() < config.minimumNetBenefitCostUnits().orElseThrow()
                    || plan.benefitBasisPoints()
                            < config.minimumBenefitBasisPoints().orElseThrow()) continue;
            if (best == null || plan.openBlasCost() < best.openBlasCost()
                    || plan.openBlasCost() == best.openBlasCost()
                        && plan.threadCount() < best.threadCount()
                    || plan.openBlasCost() == best.openBlasCost()
                        && plan.threadCount() == best.threadCount()
                        && plan.candidateOrder() < best.candidateOrder()) {
                best = plan;
                selected = candidate;
            }
        }
        return selected;
    }

    private static PreparationResourceRequirement.Workspace workspace(CpuMaterializationPlan copy) {
        return new PreparationResourceRequirement.Workspace(copy.workspaceRequirementId(),
                copy.byteCount(), copy.byteAlignment());
    }
    private static PreparationResourceRequirement.Workspace workspace(CpuOpenBlasOutputCopyPlan copy) {
        return new PreparationResourceRequirement.Workspace(copy.workspaceRequirementId(),
                copy.byteCount(), copy.byteAlignment());
    }
    private static long total(CpuPartitionAnalysisInputs.CostTerms costs, long outputs,
            long macs, long runs) { return Math.multiplyExact(runs, perRun(costs, outputs, macs)); }
    private static long perRun(CpuPartitionAnalysisInputs.CostTerms costs, long outputs, long macs) {
        return Math.addExact(Math.addExact(costs.fixedCostUnits().orElseThrow(),
                Math.multiplyExact(costs.costUnitsPerOutput().orElseThrow(), outputs)),
                Math.multiplyExact(costs.costUnitsPerMac().orElseThrow(), macs));
    }
    private static long representationCost(CpuPartitionAnalysisInputs.RepresentationCostTerms c,
            int workspaceCount, long bytes, int inputCopies, long inputElements,
            int outputCopies, long outputElements) {
        long cost = Math.multiplyExact(c.workspaceAllocationAndBindingFixed().orElseThrow(), workspaceCount);
        cost = Math.addExact(cost, Math.multiplyExact(c.workspaceCostPerByte().orElseThrow(), bytes));
        cost = Math.addExact(cost, Math.multiplyExact(c.copyInFixed().orElseThrow(), inputCopies));
        cost = Math.addExact(cost, Math.multiplyExact(c.copyInPerElement().orElseThrow(), inputElements));
        cost = Math.addExact(cost, Math.multiplyExact(c.copyOutFixed().orElseThrow(), outputCopies));
        return Math.addExact(cost, Math.multiplyExact(c.copyOutPerElement().orElseThrow(), outputElements));
    }
    private static boolean directBoundary(CpuPartitionPreparationPlan.ExecutionUnitPlan unit,
            List<CpuPartitionAnalysisInputs.BoundaryStorageFact> storage, int boundary, DataType type) {
        var fact = storage.get(boundary);
        return unit.carrierPattern().get(boundary) == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT
                && fact.nativeSegment() && fact.byteAlignment() >= type.byteWidth();
    }
    private static boolean canonical(CpuAccessPlan.Binding binding, long rows, long columns,
            CpuAccessPlan.AccessKind kind) {
        long count = Math.multiplyExact(rows, columns);
        return binding.plan().accessKind() == kind
                && binding.plan().regime() == CpuAccessPlan.Regime.DENSE_LINEAR
                && binding.plan().iterationRank() == 2 && binding.baseElementOffset() == 0
                && binding.extents().equals(List.of(rows, columns))
                && binding.effectiveStrides().equals(List.of(columns, 1L))
                && binding.elementCount() == count && binding.start() == 0
                && binding.end() == count && binding.referencedElementSpan() == count;
    }
    private static boolean copyable(CpuAccessPlan.Binding binding, long rows, long columns) {
        long count = Math.multiplyExact(rows, columns);
        return binding.plan().accessKind() == CpuAccessPlan.AccessKind.READ
                && binding.plan().iterationRank() == 2
                && binding.extents().equals(List.of(rows, columns))
                && binding.elementCount() == count && binding.start() == 0 && binding.end() == count;
    }
    private static boolean copyableOutput(CpuAccessPlan.Binding binding, long rows, long columns) {
        long count = Math.multiplyExact(rows, columns);
        return binding.plan().accessKind() == CpuAccessPlan.AccessKind.WRITE
                && binding.plan().iterationRank() == 2
                && binding.extents().equals(List.of(rows, columns))
                && binding.elementCount() == count && binding.start() == 0 && binding.end() == count;
    }

    /**
     * Creates the finalization-only adapter for a caller-owned provider handle.
     * @param library non-null provider handle to borrow
     * @return provider-free invocation view that retains the handle
     */
    static CpuOpenBlasInvocation borrowedInvocation(OpenBlasLibrary library) {
        Objects.requireNonNull(library, "library");
        return new CpuOpenBlasInvocation() {
            @Override public boolean isOpen() { return library.isOpen(); }
            @Override public int threadCount() { return library.threadCount(); }
            @Override public void setThreadCount(int count) { library.setThreadCount(count); }
            @Override public void sgemm(int m, int n, int k, float alpha, MemorySegment a,
                    MemorySegment b, float beta, MemorySegment c) {
                library.sgemm(m, n, k, alpha, a, b, beta, c);
            }
            @Override public void dgemm(int m, int n, int k, double alpha, MemorySegment a,
                    MemorySegment b, double beta, MemorySegment c) {
                library.dgemm(m, n, k, alpha, a, b, beta, c);
            }
        };
    }
}

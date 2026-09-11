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
 * Field-free cold selector for the exact bounded OpenBLAS MATMUL route.
 * It consumes only common lowering, representation, expected storage, qualification, and cost
 * facts. An incomplete configuration, ineligible route, tie with portable cost,
 * insufficient benefit, or arithmetic overflow retains the already-valid portable plan. Among
 * eligible native candidates, lower complete cost wins, followed by lower thread count and then
 * stable input order. The three direct/copy decisions are determined by the proved boundary facts,
 * so selection creates exactly the required one of eight masks rather than speculative extra
 * copies.
 */
public final class CpuOpenBlasRouteSelector {
    /** Creates a stateless selector with no provider, cache, or measurement state. */
    public CpuOpenBlasRouteSelector() { }

    /**
     * Selects an OpenBLAS plan only when exact eligibility, the combined workspace ceiling, and
     * both benefit thresholds hold. Checked-arithmetic or malformed-candidate failure is treated
     * as uncertainty and returns empty; the method performs no provider query or allocation.
     *
     * @param context non-null complete CPU analysis projection
     * @param portablePlan non-null already-selected common portable partition plan
     * @param representationPlanner non-null existing affine-copy planning owner
     * @return the selected immutable native plan, or empty to retain portable execution
     * @throws NullPointerException if an argument is {@code null}
     */
    public Optional<CpuOpenBlasRoutePlan> select(
            PrepareContext<CpuPartitionAnalysisInputs> context,
            CpuPartitionPreparationPlan portablePlan,
            CpuRepresentationPlanner representationPlanner) {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(portablePlan, "portablePlan");
        Objects.requireNonNull(representationPlanner, "representationPlanner");
        try {
            return selectChecked(context, portablePlan, representationPlanner);
        } catch (ArithmeticException | IllegalArgumentException uncertain) {
            return Optional.empty();
        }
    }

    private static Optional<CpuOpenBlasRoutePlan> selectChecked(
            PrepareContext<CpuPartitionAnalysisInputs> context,
            CpuPartitionPreparationPlan plan, CpuRepresentationPlanner representationPlanner) {
        var config = context.backendInputs().openBlasRoute();
        if (!config.complete() || plan.route() != CpuPartitionPreparationPlan.Route.PORTABLE
                || context.nodes().size() != 1 || plan.units().size() != 1
                || context.nodes().getFirst().operation().kind()
                    != io.github.pho001.synaptik.model.operation.linalg.MatmulKind.MATMUL
                || plan.bufferDeclarations().size() != 3 || !plan.materializations().isEmpty()
                || plan.partialReductionRecipe().isPresent()) {
            return Optional.empty();
        }
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
        if (!ir.epilogue().equals(CpuMatmulIr.Epilogue.none())
                || ir.leftType() != type || ir.rightType() != type
                || type != DataType.FLOAT32 && type != DataType.FLOAT64
                || unit.portablePlan().specialization().numericalMode()
                    != CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT
                || geometry.batchExtents().length != 0 || geometry.batchCount() != 1
                || geometry.removedM() || geometry.removedN()
                || geometry.m() <= 0 || geometry.n() <= 0 || geometry.k() <= 0
                || geometry.m() > Integer.MAX_VALUE || geometry.n() > Integer.MAX_VALUE
                || geometry.k() > Integer.MAX_VALUE) {
            return Optional.empty();
        }
        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        for (GraphValue value : context.values()) values.put(value.id(), value);
        for (ValueId boundary : unit.boundaryValues()) {
            GraphValue value = values.get(boundary);
            if (value == null || value.descriptor().shape().rank() != 2) return Optional.empty();
        }
        int m = Math.toIntExact(geometry.m());
        int n = Math.toIntExact(geometry.n());
        int k = Math.toIntExact(geometry.k());
        if ((!canonical(unit.accessBindings().get(0), m, k, CpuAccessPlan.AccessKind.READ)
                && !copyable(unit.accessBindings().get(0), m, k))
                || (!canonical(unit.accessBindings().get(1), k, n, CpuAccessPlan.AccessKind.READ)
                    && !copyable(unit.accessBindings().get(1), k, n))
                || (!canonical(unit.accessBindings().get(2), m, n,
                        CpuAccessPlan.AccessKind.WRITE)
                    && !copyableOutput(unit.accessBindings().get(2), m, n))) {
            return Optional.empty();
        }
        List<CpuPartitionAnalysisInputs.BoundaryStorageFact> storage =
                context.backendInputs().boundaryStorageFacts();
        if (storage.size() != 3) return Optional.empty();
        boolean leftDirect = canonical(unit.accessBindings().get(0), m, k,
                CpuAccessPlan.AccessKind.READ) && directBoundary(unit, storage, 0, type);
        boolean rightDirect = canonical(unit.accessBindings().get(1), k, n,
                CpuAccessPlan.AccessKind.READ) && directBoundary(unit, storage, 1, type);
        boolean outputDirect = canonical(unit.accessBindings().get(2), m, n,
                CpuAccessPlan.AccessKind.WRITE) && directBoundary(unit, storage, 2, type);

        long outputElements = Math.multiplyExact((long) m, n);
        long multiplyAccumulates = Math.multiplyExact(outputElements, k);
        long runs = context.backendInputs().materializationPolicy().expectedRunCount();
        long portableCost = total(config.portableCosts(), outputElements,
                multiplyAccumulates, runs);
        var policy = context.backendInputs().materializationPolicy();
        int nextWorkspace = 8;
        Optional<CpuMaterializationPlan> left = leftDirect ? Optional.empty()
                : representationPlanner.openBlasInputMaterialization(plan, 0, policy,
                        nextWorkspace++);
        if (!leftDirect && left.isEmpty()) return Optional.empty();
        Optional<CpuMaterializationPlan> right = rightDirect ? Optional.empty()
                : representationPlanner.openBlasInputMaterialization(plan, 1, policy,
                        nextWorkspace++);
        if (!rightDirect && right.isEmpty()) return Optional.empty();
        Optional<CpuOpenBlasOutputCopyPlan> output = outputDirect ? Optional.empty()
                : representationPlanner.openBlasOutputCopy(plan, nextWorkspace);
        if (!outputDirect && output.isEmpty()) return Optional.empty();
        CpuOpenBlasRoutePlan.Representation representation = representation(
                left.isPresent(), right.isPresent(), output.isPresent());
        return candidate(representation, left, right, output, type, m, n, k, config,
                outputElements, multiplyAccumulates, portableCost, runs,
                policy.maximumAdditionalBytes(),
                context.backendInputs().portableExecution().availableParallelism());
    }

    private static Optional<CpuOpenBlasRoutePlan> candidate(
            CpuOpenBlasRoutePlan.Representation representation,
            Optional<CpuMaterializationPlan> left,
            Optional<CpuMaterializationPlan> right,
            Optional<CpuOpenBlasOutputCopyPlan> output, DataType type,
            int m, int n, int k,
            CpuPartitionAnalysisInputs.OpenBlasRouteConfig config,
            long outputElements, long multiplyAccumulates, long portableCost,
            long runs, long maximumAdditionalBytes, int analysisCapacity) {
        var requirements = new ArrayList<PreparationResourceRequirement.Workspace>();
        left.map(CpuOpenBlasRouteSelector::workspace).ifPresent(requirements::add);
        right.map(CpuOpenBlasRouteSelector::workspace).ifPresent(requirements::add);
        output.map(CpuOpenBlasRouteSelector::workspace).ifPresent(requirements::add);
        long workspaceBytes = requirements.stream().mapToLong(
                PreparationResourceRequirement.Workspace::byteSize).reduce(0, Math::addExact);
        if (workspaceBytes > maximumAdditionalBytes) return Optional.empty();
        long inputElements = Math.addExact(left.map(CpuMaterializationPlan::elementCount).orElse(0L),
                right.map(CpuMaterializationPlan::elementCount).orElse(0L));
        long outputCopiedElements = output.map(CpuOpenBlasOutputCopyPlan::elementCount).orElse(0L);
        long representationCost = representationCost(config.representationCosts(),
                requirements.size(), workspaceBytes,
                (left.isPresent() ? 1 : 0) + (right.isPresent() ? 1 : 0),
                inputElements, output.isPresent() ? 1 : 0, outputCopiedElements);
        CpuOpenBlasRoutePlan selected = null;
        for (int order = 0; order < config.threadCandidates().size(); order++) {
            var thread = config.threadCandidates().get(order);
            if (thread.threadCount() > analysisCapacity) continue;
            long openBlasCost = Math.multiplyExact(runs, Math.addExact(
                    perRun(thread.openBlasCosts(), outputElements, multiplyAccumulates),
                    representationCost));
            if (openBlasCost >= portableCost) continue;
            long benefit = Math.subtractExact(portableCost, openBlasCost);
            if (benefit < config.minimumNetBenefitCostUnits().orElseThrow()) continue;
            int threshold = config.minimumBenefitBasisPoints().orElseThrow();
            if (portableCost == 0 && threshold != 0) continue;
            int basis = portableCost == 0 ? 0 : Math.toIntExact(Math.floorDiv(
                    Math.multiplyExact(10_000L, benefit), portableCost));
            if (basis < threshold) continue;
            var candidate = new CpuOpenBlasRoutePlan(type, m, n, k, 0, 1, 2,
                    thread, thread.threadCount(), thread.threadCount(), analysisCapacity, order,
                    representation, left, right, output, requirements, runs, workspaceBytes,
                    inputElements, outputCopiedElements, portableCost, openBlasCost, benefit,
                    basis);
            if (selected == null || candidate.openBlasCost() < selected.openBlasCost()
                    || candidate.openBlasCost() == selected.openBlasCost()
                        && candidate.threadCount() < selected.threadCount()
                    || candidate.openBlasCost() == selected.openBlasCost()
                        && candidate.threadCount() == selected.threadCount()
                        && candidate.candidateOrder() < selected.candidateOrder()) {
                selected = candidate;
            }
        }
        return Optional.ofNullable(selected);
    }

    private static PreparationResourceRequirement.Workspace workspace(CpuMaterializationPlan copy) {
        return new PreparationResourceRequirement.Workspace(copy.workspaceRequirementId(),
                copy.byteCount(), copy.byteAlignment());
    }

    private static PreparationResourceRequirement.Workspace workspace(
            CpuOpenBlasOutputCopyPlan copy) {
        return new PreparationResourceRequirement.Workspace(copy.workspaceRequirementId(),
                copy.byteCount(), copy.byteAlignment());
    }

    private static long total(CpuPartitionAnalysisInputs.CostTerms costs, long outputElements,
            long multiplyAccumulates, long runs) {
        return Math.multiplyExact(runs, perRun(costs, outputElements, multiplyAccumulates));
    }

    private static long perRun(CpuPartitionAnalysisInputs.CostTerms costs, long outputElements,
            long multiplyAccumulates) {
        long perRun = Math.addExact(costs.fixedCostUnits().orElseThrow(),
                Math.multiplyExact(costs.costUnitsPerOutput().orElseThrow(), outputElements));
        perRun = Math.addExact(perRun,
                Math.multiplyExact(costs.costUnitsPerMac().orElseThrow(), multiplyAccumulates));
        return perRun;
    }

    private static long representationCost(CpuPartitionAnalysisInputs.RepresentationCostTerms c,
            int workspaceCount, long workspaceBytes, int inputCopyCount, long inputElements,
            int outputCopyCount, long outputElements) {
        long cost = Math.multiplyExact(c.workspaceAllocationAndBindingFixed().orElseThrow(),
                workspaceCount);
        cost = Math.addExact(cost, Math.multiplyExact(c.workspaceCostPerByte().orElseThrow(),
                workspaceBytes));
        cost = Math.addExact(cost, Math.multiplyExact(c.copyInFixed().orElseThrow(),
                inputCopyCount));
        cost = Math.addExact(cost, Math.multiplyExact(c.copyInPerElement().orElseThrow(),
                inputElements));
        cost = Math.addExact(cost, Math.multiplyExact(c.copyOutFixed().orElseThrow(),
                outputCopyCount));
        return Math.addExact(cost, Math.multiplyExact(c.copyOutPerElement().orElseThrow(),
                outputElements));
    }

    private static boolean directBoundary(
            CpuPartitionPreparationPlan.ExecutionUnitPlan unit,
            List<CpuPartitionAnalysisInputs.BoundaryStorageFact> storage,
            int boundary, DataType type) {
        var fact = storage.get(boundary);
        return unit.carrierPattern().get(boundary)
                    == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT
                && fact.nativeSegment() && fact.byteAlignment() >= type.byteWidth();
    }

    private static boolean canonical(CpuAccessPlan.Binding binding, long rows, long columns,
            CpuAccessPlan.AccessKind accessKind) {
        long count = Math.multiplyExact(rows, columns);
        return binding.plan().accessKind() == accessKind
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
                && binding.elementCount() == count && binding.start() == 0
                && binding.end() == count;
    }

    private static boolean copyableOutput(CpuAccessPlan.Binding binding, long rows, long columns) {
        long count = Math.multiplyExact(rows, columns);
        return binding.plan().accessKind() == CpuAccessPlan.AccessKind.WRITE
                && binding.plan().iterationRank() == 2
                && binding.extents().equals(List.of(rows, columns))
                && binding.elementCount() == count && binding.start() == 0
                && binding.end() == count;
    }

    private static CpuOpenBlasRoutePlan.Representation representation(boolean left,
            boolean right, boolean output) {
        if (left && right && output) return CpuOpenBlasRoutePlan.Representation.COPY_LEFT_RIGHT_OUTPUT;
        if (left && right) return CpuOpenBlasRoutePlan.Representation.COPY_LEFT_RIGHT;
        if (left && output) return CpuOpenBlasRoutePlan.Representation.COPY_LEFT_OUTPUT;
        if (right && output) return CpuOpenBlasRoutePlan.Representation.COPY_RIGHT_OUTPUT;
        if (left) return CpuOpenBlasRoutePlan.Representation.COPY_LEFT;
        if (right) return CpuOpenBlasRoutePlan.Representation.COPY_RIGHT;
        if (output) return CpuOpenBlasRoutePlan.Representation.COPY_OUTPUT;
        return CpuOpenBlasRoutePlan.Representation.DIRECT;
    }

    /**
     * Creates the finalization-only production adapter for a caller-owned provider handle.
     * The adapter strongly retains the handle, forwards lifecycle queries and typed calls, and
     * never loads, configures, restores, or closes it.
     *
     * @param library non-null open or closed caller-owned provider handle to borrow
     * @return a non-null provider-free invocation view that retains {@code library}
     * @throws NullPointerException if {@code library} is {@code null}
     */
    static CpuOpenBlasInvocation borrowedInvocation(OpenBlasLibrary library) {
        Objects.requireNonNull(library, "library");
        return new CpuOpenBlasInvocation() {
            @Override public boolean isOpen() { return library.isOpen(); }
            @Override public int threadCount() { return library.threadCount(); }
            @Override public void setThreadCount(int threadCount) {
                library.setThreadCount(threadCount);
            }
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

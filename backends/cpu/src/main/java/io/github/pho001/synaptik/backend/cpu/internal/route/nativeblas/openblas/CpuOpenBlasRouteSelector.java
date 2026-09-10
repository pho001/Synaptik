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
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Field-free cold selector for the exact first OpenBLAS MATMUL route.
 * It consumes only common lowering, representation, expected storage, qualification, and cost
 * facts. Any incomplete, ineligible, tied, insufficient-benefit, or overflowing candidate is
 * rejected without changing the already-valid portable plan.
 */
public final class CpuOpenBlasRouteSelector {
    /** Creates a stateless selector with no provider, cache, or measurement state. */
    public CpuOpenBlasRouteSelector() { }

    /**
     * Selects an OpenBLAS plan only when exact eligibility and both benefit thresholds hold.
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
                || !canonical(unit.accessBindings().get(2), m, n,
                        CpuAccessPlan.AccessKind.WRITE)) {
            return Optional.empty();
        }
        List<CpuPartitionAnalysisInputs.BoundaryStorageFact> storage =
                context.backendInputs().boundaryStorageFacts();
        if (storage.size() != 3 || !directBoundary(unit, storage, 2, type)) {
            return Optional.empty();
        }
        boolean leftDirect = canonical(unit.accessBindings().get(0), m, k,
                CpuAccessPlan.AccessKind.READ) && directBoundary(unit, storage, 0, type);
        boolean rightDirect = canonical(unit.accessBindings().get(1), k, n,
                CpuAccessPlan.AccessKind.READ) && directBoundary(unit, storage, 1, type);

        long outputElements = Math.multiplyExact((long) m, n);
        long multiplyAccumulates = Math.multiplyExact(outputElements, k);
        long runs = context.backendInputs().materializationPolicy().expectedRunCount();
        long portableCost = total(config.portableCosts(), outputElements,
                multiplyAccumulates, 0, runs);
        var candidates = new ArrayList<Candidate>();
        if (leftDirect && rightDirect) {
            addCandidate(candidates, CpuOpenBlasRoutePlan.Representation.DIRECT,
                    Optional.empty(), type, m, n, k, config, outputElements,
                    multiplyAccumulates, portableCost, runs, 0);
        }
        var policy = context.backendInputs().materializationPolicy();
        if (rightDirect) {
            representationPlanner.openBlasInputMaterialization(plan, 0, policy).ifPresent(copy ->
                    addCandidate(candidates, CpuOpenBlasRoutePlan.Representation.COPY_LEFT,
                            Optional.of(copy), type, m, n, k, config, outputElements,
                            multiplyAccumulates, portableCost, runs, 1));
        }
        if (leftDirect) {
            representationPlanner.openBlasInputMaterialization(plan, 1, policy).ifPresent(copy ->
                    addCandidate(candidates, CpuOpenBlasRoutePlan.Representation.COPY_RIGHT,
                            Optional.of(copy), type, m, n, k, config, outputElements,
                            multiplyAccumulates, portableCost, runs, 2));
        }
        return candidates.stream().min(Candidate.ORDER).map(Candidate::plan);
    }

    private static void addCandidate(List<Candidate> candidates,
            CpuOpenBlasRoutePlan.Representation representation,
            Optional<CpuMaterializationPlan> materialization, DataType type,
            int m, int n, int k,
            CpuPartitionAnalysisInputs.OpenBlasRouteConfig config,
            long outputElements, long multiplyAccumulates, long portableCost,
            long runs, int stableOrder) {
        long copyCost = materialization.map(CpuMaterializationPlan::copyCost).orElse(0L);
        long openBlasCost = total(config.openBlasCosts(), outputElements,
                multiplyAccumulates, copyCost, runs);
        if (openBlasCost >= portableCost) return;
        long benefit = Math.subtractExact(portableCost, openBlasCost);
        if (benefit < config.minimumNetBenefitCostUnits().orElseThrow()) return;
        int threshold = config.minimumBenefitBasisPoints().orElseThrow();
        if (portableCost == 0 && threshold != 0) return;
        int basis = portableCost == 0 ? 0 : Math.toIntExact(Math.floorDiv(
                Math.multiplyExact(10_000L, benefit), portableCost));
        if (basis < threshold) return;
        Optional<PreparationResourceRequirement.Workspace> workspace = materialization.map(copy ->
                new PreparationResourceRequirement.Workspace(copy.workspaceRequirementId(),
                        copy.byteCount(), copy.byteAlignment()));
        var routePlan = new CpuOpenBlasRoutePlan(type, m, n, k, 0, 1, 2,
                config.threadConfiguration().orElseThrow(), representation, materialization,
                workspace, portableCost, openBlasCost, benefit, basis);
        candidates.add(new Candidate(routePlan, materialization.isPresent() ? 1 : 0,
                materialization.map(CpuMaterializationPlan::byteCount).orElse(0L), stableOrder));
    }

    private static long total(CpuPartitionAnalysisInputs.CostTerms costs, long outputElements,
            long multiplyAccumulates, long extraPerRun, long runs) {
        long perRun = Math.addExact(costs.fixedCostUnits().orElseThrow(),
                Math.multiplyExact(costs.costUnitsPerOutput().orElseThrow(), outputElements));
        perRun = Math.addExact(perRun,
                Math.multiplyExact(costs.costUnitsPerMac().orElseThrow(), multiplyAccumulates));
        perRun = Math.addExact(perRun, extraPerRun);
        return Math.multiplyExact(runs, perRun);
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
                && binding.end() == count && binding.referencedElementSpan() >= count;
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

    private record Candidate(CpuOpenBlasRoutePlan plan, int copyCount, long workspaceBytes,
            int stableOrder) {
        private static final Comparator<Candidate> ORDER = Comparator
                .comparingLong((Candidate value) -> value.plan().openBlasCost())
                .thenComparingInt(Candidate::copyCount)
                .thenComparingLong(Candidate::workspaceBytes)
                .thenComparingInt(Candidate::stableOrder);
    }
}

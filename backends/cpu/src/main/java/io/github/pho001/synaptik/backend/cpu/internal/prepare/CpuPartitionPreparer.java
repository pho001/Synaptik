package io.github.pho001.synaptik.backend.cpu.internal.prepare;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuSpecializationBudget;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMaterializationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionPreparer;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import jdk.incubator.vector.DoubleVector;

/**
 * Whole-partition CPU analysis entry for the current bounded static pointwise family.
 * Analysis deterministically compares direct access with at most three one-input contiguous-copy
 * candidates, then selects scalar or preferred-species vector compute and single-thread or
 * bounded parallel orchestration before shared resource assignment. It measures nothing and
 * performs no artifact or persistence access.
 */
public final class CpuPartitionPreparer implements BackendPartitionPreparer<
        CpuPartitionAnalysisInputs, CpuPartitionPreparationPlan> {
    private final CpuPartitionLowering lowering;

    /** Creates a preparer with the permanent common lowering owner. */
    public CpuPartitionPreparer() { this(new CpuPartitionLowering()); }

    /**
     * Creates a preparer with an explicit lowering collaborator.
     * @param lowering non-null whole-partition lowering retained by the preparer
     * @throws NullPointerException if {@code lowering} is {@code null}
     */
    public CpuPartitionPreparer(CpuPartitionLowering lowering) {
        this.lowering = Objects.requireNonNull(lowering, "lowering");
    }

    /**
     * Lowers, fuses, selects one bounded complete plan, and declares exact post-fusion resources.
     * @param context non-null complete CPU analysis context
     * @return one immutable analysis with one unit, a cold-selected portable strategy, one exact
     *     declaration per derived boundary, and at most one appended workspace declaration;
     *     never {@code null}
     * @throws NullPointerException if {@code context} is {@code null}
     * @throws IllegalArgumentException if complete-partition lowering rejects the occurrence or
     *     declared resource geometry is invalid
     * @throws ArithmeticException if exact byte geometry overflows {@code long}
     */
    @Override public BackendPartitionAnalysis<CpuPartitionPreparationPlan> analyze(
            PrepareContext<CpuPartitionAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        var lowered = lowering.lower(context);
        var requestedCarriers = context.backendInputs().carrierPattern().isEmpty()
                ? java.util.Collections.nCopies(lowered.boundaryValues().size(),
                        CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)
                : context.backendInputs().carrierPattern();
        if (requestedCarriers.size() != lowered.boundaryValues().size()) {
            throw new IllegalArgumentException("carrier pattern count must match boundary count");
        }
        var budget = new CpuSpecializationBudget(4, 1, 0, 0);
        Optional<CpuMaterializationPlan> materialization = selectMaterialization(lowered,
                context.backendInputs().materializationPolicy());
        var declarations = new ArrayList<PreparationResourceRequirement.Buffer>(lowered.boundaryValues().size());
        for (int i = 0; i < lowered.boundaryValues().size(); i++) declarations.add(
                new PreparationResourceRequirement.Buffer(lowered.boundaryValues().get(i),
                        Math.multiplyExact(lowered.referencedElementSpans().get(i),
                                lowered.boundaryDataTypes().get(i).byteWidth()),
                        lowered.boundaryDataTypes().get(i).byteWidth()));
        var bindings = new ArrayList<>(lowered.accessBindings());
        var carriers = new ArrayList<>(requestedCarriers);
        CpuKernelIr kernelIr = lowered.kernelIr();
        if (materialization.isPresent()) {
            var selected = materialization.orElseThrow();
            bindings.set(selected.sourceBoundaryIndex(), selected.consumerBinding());
            carriers.set(selected.sourceBoundaryIndex(), CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT);
            kernelIr = adjustedIr(kernelIr, selected.sourceBoundaryIndex(),
                    selected.consumerBinding().plan());
        }
        var config = context.backendInputs().portableExecution();
        int lanes = DoubleVector.SPECIES_PREFERRED.length();
        int speciesBits = DoubleVector.SPECIES_PREFERRED.vectorBitSize();
        boolean vectorEligible = config.computePreference()
                        == CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE
                && lanes > 1 && lowered.elementCount() >= lanes
                && lowered.kernelIr().values().stream().allMatch(value -> value.dataType() == DataType.FLOAT64)
                && lowered.kernelIr().instructions().stream().allMatch(instruction ->
                        instruction.opcode().vectorEligible())
                && bindings.stream().allMatch(binding -> vectorEligible(binding, lanes));
        int usableParallelism = Math.min(config.configuredMaximumParallelism(),
                config.availableParallelism());
        int selectedRangeCount = lowered.elementCount() == 0 ? 1 : Math.min(usableParallelism,
                Math.toIntExact(Math.min(Integer.MAX_VALUE,
                        ceilDiv(lowered.elementCount(), config.minimumElementsPerWorker()))));
        boolean parallel = selectedRangeCount >= 2;
        var strategy = vectorEligible
                ? (parallel ? CpuPartitionPreparationPlan.ExecutionStrategy.PARALLEL_VECTOR
                        : CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR)
                : (parallel ? CpuPartitionPreparationPlan.ExecutionStrategy.PARALLEL_SCALAR
                        : CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR);
        var artifactStrategy = vectorEligible
                ? CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR
                : CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR;
        var specialization = new CpuKernelSpecialization(
                CpuLoweringFingerprint.fromHex(kernelIr.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                artifactStrategy, lowered.boundaryDataTypes(), carriers,
                vectorEligible ? speciesBits : 0,
                materialization.map(CpuMaterializationPlan::sourceBoundaryIndex).orElse(-1));
        var routePlan = new CpuPortableRoutePlan(kernelIr, specialization);
        String manifest = "unit=0;fusion=" + lowered.fusionReason()
                + ";access=" + bindings.stream()
                        .map(binding -> binding.plan().regime().name()).toList()
                + ";carriers=" + carriers
                + ";route=PORTABLE;strategy=" + strategy + ";speciesBits="
                + (vectorEligible ? speciesBits : 0) + ";key="
                + specialization.structuralKey() + ";buffers=" + lowered.boundaryValues();
        var plan = new CpuPartitionPreparationPlan(
                List.of(new CpuPartitionPreparationPlan.ExecutionUnitPlan(
                        routePlan, lowered.fusionReason())),
                CpuPartitionPreparationPlan.Route.PORTABLE,
                strategy,
                declarations, lowered.boundaryValues(), bindings,
                requestedCarriers, carriers,
                lowered.extents(), lowered.elementCount(),
                selectedRangeCount, config.minimumElementsPerWorker(),
                vectorEligible ? speciesBits : 0,
                context.backendInputs().loweringManifestEnabled() ? manifest : "",
                materialization, materialization.map(copy ->
                    new PreparationResourceRequirement.Workspace(copy.workspaceRequirementId(),
                            copy.byteCount(), copy.byteAlignment())), budget);
        var requirements = new ArrayList<PreparationResourceRequirement>(declarations);
        plan.workspaceDeclaration().ifPresent(requirements::add);
        return new BackendPartitionAnalysis<>(context.partition(), plan, requirements);
    }

    private static Optional<CpuMaterializationPlan> selectMaterialization(
            CpuPartitionLowering.LoweredPartition lowered,
            CpuPartitionAnalysisInputs.MaterializationPolicy policy) {
        if (!policy.enabled()) return Optional.empty();
        long elements = lowered.elementCount();
        long bytes = Math.multiplyExact(elements, Double.BYTES);
        if (bytes > policy.maximumAdditionalBytes()) return Optional.empty();
        CpuMaterializationPlan best = null;
        int considered = 0;
        for (int index = 0; index < lowered.boundaryValues().size() - 1 && considered < 3; index++) {
            int sourceOrdinal = index;
            CpuAccessPlan.Binding source = lowered.accessBindings().get(index);
            if (lowered.boundaryDataTypes().get(index) != DataType.FLOAT64) continue;
            if (source.plan().regime() == CpuAccessPlan.Regime.SCALAR_ALL_ZERO
                    || source.plan().regime() == CpuAccessPlan.Regime.DENSE_LINEAR) continue;
            considered++;
            long useCount = lowered.kernelIr().instructions().stream()
                    .flatMap(instruction -> instruction.inputs().stream())
                    .filter(ordinal -> ordinal == sourceOrdinal)
                    .count();
            if (useCount == 0) continue;
            long directKernel = Math.multiplyExact(elements,
                    policy.directKernelCostUnitsPerElement());
            long contiguousKernel = Math.multiplyExact(elements,
                    policy.contiguousKernelCostUnitsPerElement());
            long copy = Math.addExact(policy.copyFixedCostUnits(), Math.multiplyExact(elements,
                    policy.copyCostUnitsPerElement()));
            long direct = Math.multiplyExact(policy.expectedRunCount(),
                    Math.multiplyExact(useCount, directKernel));
            long copied = Math.multiplyExact(policy.expectedRunCount(),
                    Math.addExact(copy, Math.multiplyExact(useCount, contiguousKernel)));
            if (direct == 0 || copied >= direct) continue;
            long benefit = Math.subtractExact(direct, copied);
            int basisPoints = Math.toIntExact(Math.floorDiv(
                    Math.multiplyExact(10_000L, benefit), direct));
            if (benefit < policy.minimumNetBenefitCostUnits()
                    || basisPoints < policy.minimumBenefitBasisPoints()) continue;
            CpuAccessPlan.Binding dense = denseBinding(lowered.extents(), elements);
            var candidate = new CpuMaterializationPlan(index, lowered.boundaryValues().get(index),
                    source, dense, elements, bytes, 0, Double.BYTES, useCount,
                    policy.expectedRunCount(), direct, copy, contiguousKernel, copied, benefit,
                    basisPoints, "selected: lower estimated total cost after all hard gates");
            if (best == null || copied < best.copiedTotalCost()) best = candidate;
        }
        return Optional.ofNullable(best);
    }

    private static CpuAccessPlan.Binding denseBinding(long[] extents, long elementCount) {
        var roles = new ArrayList<CpuAccessPlan.AxisRole>(extents.length);
        for (int i = 0; i < extents.length; i++) roles.add(CpuAccessPlan.AxisRole.CONTIGUOUS);
        var plan = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,
                CpuAccessPlan.Regime.DENSE_LINEAR, extents.length, roles, extents.length);
        long[] strides = new long[extents.length];
        long stride = 1;
        for (int axis = extents.length - 1; axis >= 0; axis--) {
            strides[axis] = stride;
            stride = Math.multiplyExact(stride, Math.max(1, extents[axis]));
        }
        return CpuAccessPlan.Binding.create(plan, extents, 0, strides, elementCount,
                0, elementCount, elementCount);
    }

    private static CpuKernelIr adjustedIr(CpuKernelIr source, int boundaryIndex,
            CpuAccessPlan plan) {
        var values = new ArrayList<>(source.values());
        CpuKernelIr.Value old = values.get(boundaryIndex);
        values.set(boundaryIndex, new CpuKernelIr.Value(old.ordinal(), old.dataType(), old.kind(), plan));
        return new CpuKernelIr(values, source.instructions(), source.loop(), source.stores());
    }

    private static boolean vectorEligible(
            io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Binding binding,
            int lanes) {
        return switch (binding.plan().regime()) {
            case GENERAL_ODOMETER -> false;
            case SCALAR_ALL_ZERO -> binding.plan().accessKind()
                    == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.AccessKind.READ;
            case DENSE_LINEAR -> true;
            case LAST_AXIS_BIAS, BLOCK_OUTER -> contiguousRun(binding) >= lanes;
        };
    }

    private static long contiguousRun(
            io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Binding binding) {
        long result = 1;
        int start = binding.extents().size() - binding.plan().contiguousSuffix();
        for (int axis = start; axis < binding.extents().size(); axis++) {
            result = Math.multiplyExact(result, binding.extents().get(axis));
        }
        return result;
    }

    private static long ceilDiv(long value, long divisor) {
        return value == 0 ? 0 : 1 + (value - 1) / divisor;
    }
}

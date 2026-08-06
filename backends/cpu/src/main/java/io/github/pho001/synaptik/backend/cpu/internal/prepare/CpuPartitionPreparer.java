package io.github.pho001.synaptik.backend.cpu.internal.prepare;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionLowering;
import io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionPreparer;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import jdk.incubator.vector.DoubleVector;

/**
 * Whole-partition CPU analysis entry for the current static broadcast/access proving slice.
 * Analysis deterministically selects scalar or preferred-species vector compute and single-thread
 * or bounded parallel orchestration before shared resource assignment.
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
     * Lowers, fuses, and declares exact post-fusion boundary resources.
     * @param context non-null complete CPU analysis context
     * @return one immutable analysis with one unit, a cold-selected portable strategy, and four
     *     exact buffer declarations
     * @throws NullPointerException if {@code context} is {@code null}
     * @throws IllegalArgumentException if complete-partition lowering rejects the occurrence or
     *     declared resource geometry is invalid
     * @throws ArithmeticException if exact byte geometry overflows {@code long}
     */
    @Override public BackendPartitionAnalysis<CpuPartitionPreparationPlan> analyze(
            PrepareContext<CpuPartitionAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        var lowered = lowering.lower(context);
        if (context.backendInputs().carrierPattern().size() != lowered.boundaryValues().size()) {
            throw new IllegalArgumentException("carrier pattern count must match boundary count");
        }
        var declarations = new ArrayList<PreparationResourceRequirement.Buffer>(4);
        for (int i = 0; i < lowered.boundaryValues().size(); i++) declarations.add(
                new PreparationResourceRequirement.Buffer(lowered.boundaryValues().get(i),
                        Math.multiplyExact(lowered.referencedElementSpans().get(i),
                                DataType.FLOAT64.byteWidth()), Double.BYTES));
        var config = context.backendInputs().portableExecution();
        int lanes = DoubleVector.SPECIES_PREFERRED.length();
        int speciesBits = DoubleVector.SPECIES_PREFERRED.vectorBitSize();
        boolean vectorEligible = config.computePreference()
                        == CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE
                && lanes > 1 && lowered.elementCount() >= lanes
                && lowered.accessBindings().stream().allMatch(binding -> vectorEligible(binding, lanes));
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
                CpuLoweringFingerprint.fromHex(lowered.kernelIr().structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                artifactStrategy, context.backendInputs().carrierPattern(),
                vectorEligible ? speciesBits : 0);
        var routePlan = new CpuPortableRoutePlan(lowered.kernelIr(), specialization);
        String manifest = "unit=0;fusion=" + lowered.fusionReason()
                + ";access=" + lowered.accessBindings().stream()
                        .map(binding -> binding.plan().regime().name()).toList()
                + ";carriers=" + context.backendInputs().carrierPattern()
                + ";route=PORTABLE;strategy=" + strategy + ";speciesBits="
                + (vectorEligible ? speciesBits : 0) + ";key="
                + specialization.structuralKey() + ";buffers=" + lowered.boundaryValues();
        var plan = new CpuPartitionPreparationPlan(
                List.of(new CpuPartitionPreparationPlan.ExecutionUnitPlan(
                        routePlan, lowered.fusionReason())),
                CpuPartitionPreparationPlan.Route.PORTABLE,
                strategy,
                declarations, lowered.boundaryValues(), lowered.accessBindings(),
                context.backendInputs().carrierPattern(), lowered.extents(), lowered.elementCount(),
                selectedRangeCount, config.minimumElementsPerWorker(),
                vectorEligible ? speciesBits : 0,
                context.backendInputs().loweringManifestEnabled() ? manifest : "");
        return new BackendPartitionAnalysis<>(context.partition(), plan,
                new ArrayList<>(declarations));
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

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

/** Whole-partition CPU analysis entry for the exact task-0005A proving slice. */
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
     * @return one immutable analysis with one unit and four exact buffer declarations
     * @throws NullPointerException if {@code context} is {@code null}
     * @throws IllegalArgumentException if complete-partition lowering rejects the occurrence or
     *     declared resource geometry is invalid
     * @throws ArithmeticException if exact byte geometry overflows {@code long}
     */
    @Override public BackendPartitionAnalysis<CpuPartitionPreparationPlan> analyze(
            PrepareContext<CpuPartitionAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        var lowered = lowering.lower(context);
        long bytes = Math.multiplyExact(lowered.elementCount(), DataType.FLOAT64.byteWidth());
        var declarations = new ArrayList<PreparationResourceRequirement.Buffer>(4);
        for (var value : lowered.boundaryValues()) declarations.add(
                new PreparationResourceRequirement.Buffer(value, bytes, Double.BYTES));
        var specialization = new CpuKernelSpecialization(
                CpuLoweringFingerprint.fromHex(lowered.kernelIr().structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR);
        var routePlan = new CpuPortableRoutePlan(lowered.kernelIr(), specialization);
        String manifest = "unit=0;fusion=" + lowered.fusionReason()
                + ";access=CANONICAL_DENSE;route=PORTABLE;strategy=scalar;key="
                + specialization.structuralKey() + ";buffers=" + lowered.boundaryValues();
        var plan = new CpuPartitionPreparationPlan(
                List.of(new CpuPartitionPreparationPlan.ExecutionUnitPlan(
                        routePlan, lowered.fusionReason())),
                CpuPartitionPreparationPlan.Route.PORTABLE,
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                declarations, lowered.boundaryValues(), lowered.extents(), lowered.elementCount(),
                context.backendInputs().loweringManifestEnabled() ? manifest : "");
        return new BackendPartitionAnalysis<>(context.partition(), plan,
                new ArrayList<>(declarations));
    }
}

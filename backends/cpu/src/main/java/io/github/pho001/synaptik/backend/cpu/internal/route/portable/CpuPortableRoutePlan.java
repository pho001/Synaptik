package io.github.pho001.synaptik.backend.cpu.internal.route.portable;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import java.util.Objects;

/**
 * Immutable portable Class-File realization facts for one already-lowered CPU execution unit.
 * This unsupported internal plan performs no graph interpretation or native-route selection.
 * A selected contiguous copy is already reflected in the supplied canonical consumer IR and
 * specialization; the plan does not retain source/workspace objects or execute the copy.
 *
 * @param kernelIr non-null route-independent canonical IR
 * @param specialization non-null selected exact/default scalar or preferred-species vector
 *     Class-File specialization; parallel orchestration, if selected, remains outside the artifact
 */
public record CpuPortableRoutePlan(CpuKernelIr kernelIr, CpuKernelSpecialization specialization) {
    /**
     * Validates one portable realization plan.
     *
     * @param kernelIr non-null route-independent canonical IR
     * @param specialization non-null matching scalar or vector generated specialization
     * @throws NullPointerException if either component is {@code null}
     * @throws IllegalArgumentException if the specialization does not match the canonical IR
     */
    public CpuPortableRoutePlan {
        Objects.requireNonNull(kernelIr, "kernelIr");
        Objects.requireNonNull(specialization, "specialization");
        if (!kernelIr.structuralKey().equals(specialization.loweringFingerprint().hex())) {
            throw new IllegalArgumentException("specialization must match canonical IR");
        }
        var realizations = kernelIr.instructions().stream()
                .filter(instruction -> instruction.opcode()
                        == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode.SCALAR_POW)
                .map(CpuKernelIr.Instruction::powerRealization).toList();
        if (!realizations.equals(specialization.scalarPowerRealizations())) {
            throw new IllegalArgumentException("specialization power realizations must match canonical IR");
        }
    }
}

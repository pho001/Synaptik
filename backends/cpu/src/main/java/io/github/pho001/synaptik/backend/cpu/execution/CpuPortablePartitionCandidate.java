package io.github.pho001.synaptik.backend.cpu.execution;

import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Immutable complete portable candidate for one ordered CPU-owned partition.
 *
 * <p>The candidate retains one non-empty kernel sequence in partition-node order and one exact
 * shared declaration set. Kernel declarations must reference this set by identity. Construction
 * performs no slot assignment, artifact access, binding, or execution.</p>
 */
final class CpuPortablePartitionCandidate {
    private final List<PreparationResourceRequirement> requirements;
    private final List<CpuPortableKernelCandidate> kernels;

    /**
     * Validates and snapshots one complete ordered partition candidate.
     *
     * @param requirements non-null ordered shared declarations
     * @param kernels non-null non-empty ordered node-kernel candidates
     * @throws NullPointerException if an aggregate or indexed entry is {@code null}
     * @throws IllegalArgumentException if the kernel sequence is empty, declarations duplicate
     *     an identity, or a kernel refers to a declaration outside the shared set
     */
    CpuPortablePartitionCandidate(
            List<PreparationResourceRequirement> requirements,
            List<CpuPortableKernelCandidate> kernels) {
        this.requirements = checkedCopy(requirements, "requirements");
        this.kernels = checkedCopy(kernels, "kernels");
        if (this.kernels.isEmpty()) {
            throw new IllegalArgumentException("kernels must not be empty");
        }
        Set<PreparationResourceRequirement> declared =
                Collections.newSetFromMap(new IdentityHashMap<>());
        for (int index = 0; index < this.requirements.size(); index++) {
            if (!declared.add(this.requirements.get(index))) {
                throw new IllegalArgumentException(
                        "requirements[" + index + "] duplicates declaration identity");
            }
        }
        Set<PreparationResourceRequirement> used =
                Collections.newSetFromMap(new IdentityHashMap<>());
        for (int kernelIndex = 0; kernelIndex < this.kernels.size(); kernelIndex++) {
            var kernel = this.kernels.get(kernelIndex);
            for (int requirementIndex = 0;
                    requirementIndex < kernel.requirements().size();
                    requirementIndex++) {
                var requirement = kernel.requirements().get(requirementIndex);
                if (!declared.contains(requirement)) {
                    throw new IllegalArgumentException(
                            "kernels[" + kernelIndex + "].requirements[" + requirementIndex
                                    + "] is not a shared declaration");
                }
                used.add(requirement);
            }
        }
        for (int index = 0; index < this.requirements.size(); index++) {
            if (!used.contains(this.requirements.get(index))) {
                throw new IllegalArgumentException("requirements[" + index + "] is unused");
            }
        }
    }

    private static <T> List<T> checkedCopy(List<T> values, String name) {
        Objects.requireNonNull(values, name);
        for (int index = 0; index < values.size(); index++) {
            Objects.requireNonNull(values.get(index), name + "[" + index + "]");
        }
        return List.copyOf(values);
    }

    /**
     * Returns the shared declarations retained in deterministic encounter order.
     *
     * @return immutable ordered declaration snapshot; never {@code null}
     */
    List<PreparationResourceRequirement> requirements() { return requirements; }

    /**
     * Returns the complete generated-kernel candidates in partition-node order.
     *
     * @return immutable non-empty kernel snapshot; never {@code null}
     */
    List<CpuPortableKernelCandidate> kernels() { return kernels; }

    /**
     * Returns the sole specialization for the CPU-0004 one-kernel compatibility path.
     *
     * @return exact non-null sole kernel specialization
     * @throws IllegalStateException if this partition contains more than one kernel
     */
    CpuKernelSpecialization specialization() {
        if (kernels.size() != 1) throw new IllegalStateException(
                "partition candidate does not contain exactly one kernel");
        return kernels.getFirst().specialization();
    }
}

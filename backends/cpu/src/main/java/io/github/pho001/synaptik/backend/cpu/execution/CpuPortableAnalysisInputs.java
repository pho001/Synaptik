package io.github.pho001.synaptik.backend.cpu.execution;

import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;

/**
 * Immutable explicit target facts consumed by portable CPU partition analysis.
 *
 * <p>The Vector-shape list is snapshotted in caller order. The parallel configuration is retained
 * by exact reference and owns no worker. This value contains no discovery result that may change
 * after construction, executable state, physical resource, or tuning measurement.</p>
 *
 * @param supportedVectorShapes ordered exact Java Vector API species supported by the target
 * @param parallelConfiguration non-null prepared worker/range configuration
 */
record CpuPortableAnalysisInputs(
        List<CpuKernelSpecialization.VectorShape> supportedVectorShapes,
        CpuPreparedParallelConfiguration parallelConfiguration)
        implements BackendAnalysisInputs {
    /**
     * Validates entries in order and snapshots the supplied species list.
     *
     * @param supportedVectorShapes non-null ordered supported-shape list; entries must be non-null
     *     and unique by value
     * @param parallelConfiguration non-null immutable prepared worker/range configuration,
     *     retained by exact reference
     * @throws NullPointerException if either component or an indexed shape is {@code null}
     * @throws IllegalArgumentException if a later shape duplicates an earlier shape
     */
    CpuPortableAnalysisInputs {
        Objects.requireNonNull(supportedVectorShapes, "supportedVectorShapes");
        var observed = new HashSet<CpuKernelSpecialization.VectorShape>();
        for (int index = 0; index < supportedVectorShapes.size(); index++) {
            var shape = Objects.requireNonNull(
                    supportedVectorShapes.get(index), "supportedVectorShapes[" + index + "]");
            if (!observed.add(shape)) {
                throw new IllegalArgumentException(
                        "supportedVectorShapes[" + index + "] duplicates " + shape);
            }
        }
        supportedVectorShapes = List.copyOf(supportedVectorShapes);
        Objects.requireNonNull(parallelConfiguration, "parallelConfiguration");
    }
}

package io.github.pho001.synaptik.prepare;

import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import java.util.Objects;

/**
 * Associates one exact planned partition with its finalized executable recipe.
 *
 * <p>Backend ownership remains authoritative in the partition, and prepared memory remains
 * authoritative in the executable. This immutable value duplicates neither association and owns
 * no physical or per-run resource.</p>
 *
 * @param partition exact non-null immutable planned-partition reference
 * @param executable exact non-null immutable executable reference
 */
public record PreparedPartition(PlannedPartition partition, PreparedExecutable executable) {
    /**
     * Creates one exact partition-to-executable association.
     *
     * @param partition exact non-null partition reference to retain
     * @param executable exact non-null executable reference to retain
     * @throws NullPointerException if either component is null, in declaration order
     */
    public PreparedPartition {
        Objects.requireNonNull(partition, "partition");
        Objects.requireNonNull(executable, "executable");
    }
}

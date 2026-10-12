package io.github.pho001.synaptik.training;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;

/**
 * Immutable SGD configuration for an ordered list of recursive Module parameter paths.
 *
 * <p>Paths use the keys returned by {@code Module.parametersRecursively()}. Their order is part
 * of this group's identity and therefore of training-state restore compatibility, even when a
 * different order would produce the same arithmetic. The group owns a defensive snapshot of its
 * path list; it does not own parameters or optimizer slots. A session validates that paths exist
 * and that no other group claims them.</p>
 */
public final class ParameterGroup {
    private final List<String> paths;
    private final Sgd optimizer;

    /**
     * Defines one explicit parameter group without changing the Module's gradient targets.
     *
     * @param paths non-null, nonempty ordered list of distinct, non-null, non-blank recursive
     *     parameter paths; copied before retention
     * @param optimizer non-null immutable optimizer configuration; currently only {@link Sgd}
     * @throws NullPointerException if either argument or a path is null
     * @throws IllegalArgumentException if paths is empty or contains a blank or duplicate path
     */
    public ParameterGroup(List<String> paths, Optimizer optimizer) {
        Objects.requireNonNull(paths, "paths");
        if (paths.isEmpty()) {
            throw new IllegalArgumentException("parameter group paths must not be empty");
        }
        var copy = new ArrayList<String>(paths.size());
        var seen = new HashSet<String>();
        for (int index = 0; index < paths.size(); index++) {
            String path = Objects.requireNonNull(paths.get(index), "paths[" + index + "]");
            if (path.isBlank() || !seen.add(path)) {
                throw new IllegalArgumentException("blank or duplicate parameter group path: " + path);
            }
            copy.add(path);
        }
        this.paths = List.copyOf(copy);
        this.optimizer = switch (Objects.requireNonNull(optimizer, "optimizer")) {
            case Sgd sgd -> sgd;
        };
    }

    /**
     * Returns the exact ordered recursive paths that identify this group.
     *
     * @return non-null immutable, nonempty ordered path list
     */
    public List<String> paths() {
        return paths;
    }

    /**
     * Returns the immutable SGD configuration assigned to this group's paths.
     *
     * @return non-null optimizer retained by exact reference
     */
    public Optimizer optimizer() {
        return optimizer;
    }

    /**
     * Compares the ordered path list and SGD configuration that identify this group.
     *
     * @param other the candidate group, or {@code null}
     * @return {@code true} exactly when {@code other} has the same paths in the same order and
     *     an equal optimizer configuration
     */
    @Override
    public boolean equals(Object other) {
        return this == other || other instanceof ParameterGroup group
                && paths.equals(group.paths) && optimizer.equals(group.optimizer);
    }

    /**
     * Hashes the ordered paths and optimizer configuration used by {@link #equals(Object)}.
     *
     * @return a hash code consistent with group equality
     */
    @Override
    public int hashCode() {
        return 31 * paths.hashCode() + optimizer.hashCode();
    }

    /**
     * Describes this group's ordered paths and optimizer configuration for diagnostics.
     *
     * @return a non-null diagnostic description
     */
    @Override
    public String toString() {
        return "ParameterGroup[paths=" + paths + ", optimizer=" + optimizer + ']';
    }
}

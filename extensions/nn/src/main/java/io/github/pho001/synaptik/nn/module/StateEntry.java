package io.github.pho001.synaptik.nn.module;

import io.github.pho001.synaptik.model.tensor.Tensor;
import java.util.Objects;

/**
 * Immutable snapshot of one qualified module-state binding.
 *
 * <p>The path is relative to the module that exported or will load the entry. Dot-separated path
 * segments identify child modules and the final local state name. The value is the exact Tensor
 * reference supplied at construction; this record performs no copying, evaluation, inspection,
 * storage access, or mutation. It retains no module, wrapper, mode, optimizer, random state, or
 * execution state.</p>
 *
 * <p>Record equality and hashing use the three components. The generated diagnostic string and
 * the {@link StateKind} enum name are not checkpoint-format or wire contracts.</p>
 *
 * @param path non-null, non-blank relative path containing non-blank segments separated by single
 *     {@code .} characters; accepted text is retained without trimming or normalization
 * @param kind non-null exact parameter-or-buffer role returned by {@link #kind()}
 * @param value non-null exact Tensor reference returned by {@link #value()} without copying,
 *     evaluation, or ownership transfer
 */
public record StateEntry(String path, StateKind kind, Tensor value) {
    /**
     * Creates one validated in-memory state entry.
     *
     * @param path non-null, non-blank relative path containing non-blank dot-separated segments;
     *     accepted text is retained exactly
     * @param kind non-null exact binding role to retain
     * @param value non-null exact Tensor reference to retain without copying or evaluation
     * @throws NullPointerException if {@code path}, {@code kind}, or {@code value} is null, checked
     *     in that order after path grammar validation
     * @throws IllegalArgumentException if {@code path} is blank or contains a blank segment
     */
    public StateEntry {
        Objects.requireNonNull(path, "path");
        String[] segments = path.split("\\.", -1);
        for (String segment : segments) {
            if (segment.isBlank()) {
                throw new IllegalArgumentException(
                        "state path must contain only non-blank segments: " + path);
            }
        }
        Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(value, "value");
    }
}

package io.github.pho001.synaptik.training;

import io.github.pho001.synaptik.model.tensor.Tensor;
import java.util.List;
import java.util.Objects;

/**
 * Caller-owned live admission boundary for an ordered set of trainable Tensor identities.
 *
 * <p>{@link #entries()} exposes immutable membership, not a detached value snapshot. Implementations
 * retain their live owner and {@link #validateUnchanged()} must read it again, rejecting changed
 * count, order, paths or exact Tensor bindings (and owner-specific wrapper identities). Comparing
 * only the frozen entries is not live validation. Membership validation must not freeze storage
 * associations or bytes: an explicit pre-session native copy intentionally replaces associations
 * without changing membership. Training separately validates descriptors, storage and values.</p>
 *
 * <p>The caller excludes conflicting owner and storage mutation during copy and for the complete
 * session lifetime. This boundary adds no lock, close lifecycle, automatic transfer or NN
 * dependency. Implementations must allow validation from every thread operating on the session
 * under that exclusion, and must not publish side effects when validation fails.</p>
 */
public interface TrainingParameters {
    /**
     * Returns the fixed membership of this borrow in caller encounter order, never sorted.
     *
     * @return non-null immutable ordered list of non-null entries with unique nonblank paths;
     *     Training rejects an empty list and snapshots the list at admission
     */
    List<Entry> entries();

    /**
     * Re-reads the live owner and verifies exact ordered membership against this borrow.
     *
     * <p>Called at capture, before execution/state/restore, and immediately before accumulation,
     * update or restore commit. A failure rejects that operation without publishing Training state.
     * Storage replacement and byte mutation are deliberately not membership changes.</p>
     *
     * @throws IllegalStateException if live count, order, path, Tensor or owner-specific wrapper
     *     identity differs, or the complete live membership can no longer be discovered
     */
    void validateUnchanged();

    /**
     * One stable diagnostic path and exact caller-owned Tensor reference; no value is copied.
     *
     * @param path non-null nonblank path, unique within its source; dots have no special meaning
     *     to Training
     * @param tensor non-null exact Tensor targeted by the objective; Training validates its
     *     supported descriptor, storage and finite values during admission
     */
    record Entry(String path, Tensor tensor) {
        /**
         * Retains a valid path and Tensor without inspecting or mutating storage.
         *
         * @param path non-null nonblank caller-owned diagnostic path
         * @param tensor non-null exact Tensor reference
         * @throws NullPointerException if either input is null
         * @throws IllegalArgumentException if the path is blank
         */
        public Entry {
            Objects.requireNonNull(path, "path");
            if (path.isBlank()) {
                throw new IllegalArgumentException("parameter path must not be blank");
            }
            Objects.requireNonNull(tensor, "tensor");
        }
    }
}

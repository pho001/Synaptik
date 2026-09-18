package io.github.pho001.synaptik.engine;

import io.github.pho001.synaptik.config.tuning.ModelAutotuningConfig;
import io.github.pho001.synaptik.model.tensor.Tensor;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Immutable public inputs for one synchronous Engine-owned model-autotuning preparation.
 *
 * <p>The request snapshots only its list structure and identity bytes. It retains the exact
 * configuration, identity, and {@link Tensor} references. The caller continues to own every
 * Tensor and its storage and must keep that storage live, accessible, and unmodified until
 * {@link Engine#prepareTuned(CompiledGraph, ModelAutotuningRequest)} returns or fails.</p>
 */
public final class ModelAutotuningRequest {
    private final ModelAutotuningConfig config;
    private final ModelIdentity modelIdentity;
    private final List<Tensor> representativeInputs;

    /**
     * Snapshots the representative-input container while retaining its caller-owned Tensors.
     *
     * @param config non-null declarative tuning configuration, retained exactly
     * @param modelIdentity non-null caller-defined model evidence identity, retained exactly
     * @param representativeInputs non-null list of non-null caller-owned representative Tensors
     * @throws NullPointerException if an argument or list element is null
     */
    public ModelAutotuningRequest(
            ModelAutotuningConfig config,
            ModelIdentity modelIdentity,
            List<Tensor> representativeInputs) {
        this.config = Objects.requireNonNull(config, "config");
        this.modelIdentity = Objects.requireNonNull(modelIdentity, "modelIdentity");
        this.representativeInputs = List.copyOf(
                Objects.requireNonNull(representativeInputs, "representativeInputs"));
    }

    /** Returns the request policy retained at construction.
     * @return the exact non-null configuration
     */
    public ModelAutotuningConfig config() { return config; }
    /** Returns the caller's evidence label.
     * @return the exact non-null retained model identity
     */
    public ModelIdentity modelIdentity() { return modelIdentity; }
    /** Returns the representative Tensor container snapshot.
     *
     * @return the same immutable list snapshot on every call; its non-null Tensor elements and
     *     their caller-owned storage are not copied
     */
    public List<Tensor> representativeInputs() { return representativeInputs; }

    /**
     * Immutable caller-defined versioned model evidence identity.
     *
     * <p>The caller owns schema meaning, uniqueness, collision resistance, and revision policy.
     * Engine faithfully labels evidence with this value but neither verifies it nor uses it as a
     * workload-cache compatibility key or candidate-selection authority.</p>
     */
    public static final class ModelIdentity {
        private final int schemaVersion;
        private final byte[] bytes;

        /** Creates a caller-defined evidence identity.
         *
         * @param schemaVersion positive caller-owned schema version
         * @param bytes non-null non-empty opaque bytes, defensively copied
         * @throws NullPointerException if {@code bytes} is null
         * @throws IllegalArgumentException if the version is not positive or bytes are empty
         */
        public ModelIdentity(int schemaVersion, byte[] bytes) {
            if (schemaVersion <= 0) throw new IllegalArgumentException("schemaVersion must be positive");
            Objects.requireNonNull(bytes, "bytes");
            if (bytes.length == 0) throw new IllegalArgumentException("bytes must not be empty");
            this.schemaVersion = schemaVersion;
            this.bytes = bytes.clone();
        }

        /** Returns the caller's identity schema.
         * @return the positive schema version
         */
        public int schemaVersion() { return schemaVersion; }
        /** Copies the opaque identity payload.
         * @return a fresh non-empty byte array
         */
        public byte[] bytes() { return bytes.clone(); }

        @Override public boolean equals(Object other) {
            return this == other || other instanceof ModelIdentity that
                    && schemaVersion == that.schemaVersion && Arrays.equals(bytes, that.bytes);
        }
        @Override public int hashCode() { return 31 * schemaVersion + Arrays.hashCode(bytes); }
        @Override public String toString() {
            return "ModelIdentity[schemaVersion=" + schemaVersion + ", byteLength=" + bytes.length + "]";
        }
    }
}

package io.github.pho001.synaptik.nn.checkpoint;

import io.github.pho001.synaptik.checkpoint.artifact.ArtifactEnvelope;
import java.util.List;
import java.util.Objects;

/**
 * Fully validated, detached model artifact candidate. It contains canonical metadata and opaque
 * bytes, but no live Tensor, NN binding, Engine handle, or state-installation authority.
 */
final class ModelArtifactCandidate {
    private final ModelArtifactCodec.Metadata metadata;
    private final ArtifactEnvelope envelope;

    /**
     * Retains previously validated model metadata and its matching immutable envelope. This
     * internal constructor does not revalidate their correspondence.
     *
     * @param metadata canonical model metadata, never null
     * @param envelope matching detached opaque envelope, never null
     * @throws NullPointerException if metadata or envelope is null
     */
    ModelArtifactCandidate(ModelArtifactCodec.Metadata metadata, ArtifactEnvelope envelope) {
        this.metadata = Objects.requireNonNull(metadata, "metadata");
        this.envelope = Objects.requireNonNull(envelope, "envelope");
    }

    /**
     * Returns the versioned opaque artifact image for a future publication facade.
     *
     * @return immutable detached envelope; never null
     */
    ArtifactEnvelope envelope() { return envelope; }

    /**
     * Returns the ordered path, kind, type, Shape, and length descriptors.
     *
     * @return immutable metadata list without live Tensor references
     */
    List<ModelArtifactCodec.State> states() { return metadata.states(); }

    /**
     * Returns the SHA-256 fingerprint of exact canonical model metadata.
     *
     * @return fresh 32-byte digest, including both caller fingerprints and all ordered entries
     */
    byte[] schemaFingerprint() { return ModelArtifactCodec.fingerprint(metadata); }

    /**
     * Returns the caller-provided configuration fingerprint.
     *
     * @return fresh 32-byte SHA-256 digest
     */
    byte[] configDigest() { return metadata.configDigest(); }

    /**
     * Returns the optional caller-provided input artifact fingerprint.
     *
     * @return fresh 32-byte SHA-256 digest, or null if absent
     */
    byte[] inputDigest() { return metadata.inputDigest(); }
}

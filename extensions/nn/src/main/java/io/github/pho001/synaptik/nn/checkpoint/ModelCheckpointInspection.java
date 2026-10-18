package io.github.pho001.synaptik.nn.checkpoint;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.nn.module.StateKind;
import java.util.List;
import java.util.Objects;

/**
 * Detached, typed description of a completely validated version-one NN model checkpoint. It owns
 * no file, Tensor, module, Engine, or closeable resource. A later file load rereads its path;
 * envelope byte access and load independently validate their supplied image. None trusts this
 * snapshot as a validation credential. The digests identify caller-defined configuration and
 * optional input artifact bytes; Checkpoint does not interpret their meaning. State descriptors
 * contain no value bytes; use {@link ModelCheckpoint#canonicalStateBytes} for independently
 * validated raw logical bytes keyed by their exact paths.
 */
public final class ModelCheckpointInspection {
    private final byte[] configDigest;
    private final byte[] inputDigest;
    private final byte[] schemaFingerprint;
    private final List<State> states;

    /**
     * Creates an immutable inspection snapshot from validated metadata.
     *
     * @param configDigest non-null 32-byte configuration digest, copied defensively
     * @param inputDigest optional 32-byte input digest, or null for absent, copied when present
     * @param schemaFingerprint non-null 32-byte digest of exact canonical model metadata, copied
     * @param states non-null ordered detached descriptors, copied into an immutable list
     * @throws NullPointerException if a required argument or descriptor is null
     * @throws IllegalArgumentException if a digest has a length other than 32 bytes
     */
    ModelCheckpointInspection(byte[] configDigest, byte[] inputDigest,
                              byte[] schemaFingerprint, List<State> states) {
        this.configDigest = digest(configDigest);
        this.inputDigest = inputDigest == null ? null : digest(inputDigest);
        this.schemaFingerprint = digest(schemaFingerprint);
        this.states = List.copyOf(states);
    }

    /**
     * Returns the caller-defined versioned configuration fingerprint without exposing this
     * inspection's retained bytes for mutation.
     *
     * @return a fresh 32-byte configuration digest, never null
     */
    public byte[] configDigest() { return configDigest.clone(); }

    /**
     * Returns the optional caller-defined input-artifact fingerprint. Absence is distinct from
     * every present digest, including all zero bytes.
     *
     * @return a fresh 32-byte input digest, or null when absent
     */
    public byte[] inputDigest() { return inputDigest == null ? null : inputDigest.clone(); }

    /**
     * Returns the fingerprint binding both caller digests and every ordered state descriptor.
     * It describes this inspected image, not a promise about another image or a file's later contents.
     *
     * @return a fresh 32-byte digest of exact canonical model metadata, never null
     */
    public byte[] schemaFingerprint() { return schemaFingerprint.clone(); }

    /**
     * Returns the inspected image's state descriptors in metadata order. They do not carry payload bytes,
     * Tensor identity, or a live NN binding.
     *
     * @return immutable ordered descriptors, with no payload or live state references
     */
    public List<State> states() { return states; }

    private static byte[] digest(byte[] value) {
        Objects.requireNonNull(value, "digest");
        if (value.length != 32) throw new IllegalArgumentException("SHA-256 digest must be 32 bytes");
        return value.clone();
    }

    /**
     * One state path and its persisted logical descriptor. Instances returned by
     * either {@code ModelCheckpoint.inspect} overload have validated NN paths;
     * direct construction checks only that the path is non-null. The value is not exposed;
     * byte length counts canonical dense payload bytes, including zero for empty Shape.
     *
     * @param path non-null exact NN relative path
     * @param kind non-null parameter or buffer role
     * @param dataType non-null logical element type
     * @param shape non-null immutable static logical Shape
     * @param byteLength exact checked Shape element count times type width, in bytes
     */
    public record State(String path, StateKind kind, DataType dataType, Shape shape,
                        long byteLength) {
        /**
         * Retains immutable descriptor components; no Tensor or payload is retained. Direct
         * construction does not check the NN path grammar.
         *
         * @param path non-null exact relative NN path
         * @param kind non-null state role
         * @param dataType non-null logical element type
         * @param shape non-null static logical Shape
         * @param byteLength exact checked Shape element count times type width, in bytes
         * @throws NullPointerException if a reference component is null
         * @throws IllegalArgumentException if Shape is dynamic or byteLength differs from its
         *     checked element-count-to-width product
         * @throws ArithmeticException if the Shape element count or canonical byte size overflows
         */
        public State {
            Objects.requireNonNull(path, "path");
            Objects.requireNonNull(kind, "kind");
            Objects.requireNonNull(dataType, "dataType");
            Objects.requireNonNull(shape, "shape");
            if (!shape.isFullyStatic() || byteLength < 0
                    || Math.multiplyExact(shape.knownElementCount().orElseThrow(),
                            dataType.byteWidth()) != byteLength)
                throw new IllegalArgumentException("invalid canonical state byte length");
        }
    }
}

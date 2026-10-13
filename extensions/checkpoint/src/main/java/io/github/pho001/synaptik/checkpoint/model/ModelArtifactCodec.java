package io.github.pho001.synaptik.checkpoint.model;

import io.github.pho001.synaptik.checkpoint.artifact.ArtifactEnvelope;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactFormatException;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.nn.module.StateKind;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.StandardCharsets;
import java.nio.charset.CodingErrorAction;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;

/**
 * Version-one model metadata and its bounded mapping to opaque artifact entries. The wire IDs,
 * framing, and entry order here are independent of Java enum order and diagnostic names.
 */
final class ModelArtifactCodec {
    private static final byte[] MAGIC = "SYNMOD01".getBytes(StandardCharsets.US_ASCII);
    private static final int VERSION = 1;
    private static final int ARRAY_CEILING = Integer.MAX_VALUE - 8;
    private static final String KIND = "model.v1";
    private static final String SCHEMA = "schema";
    private static final int MAX_ORDINAL = 99_999_999;

    private ModelArtifactCodec() {
    }

    /**
     * One exact NN state descriptor, excluding its live Tensor and detached payload.
     *
     * @param path non-null exact NN path; encoding later validates its nonblank dot-segment
     *     grammar and Unicode
     * @param kind parameter or buffer role
     * @param type exact logical data type
     * @param shape non-null logical shape; encoding later requires it to be fully static
     * @param length non-negative expected canonical byte length, checked against shape and type
     *     during encoding
     * @throws NullPointerException if a descriptor component other than length is null
     * @throws IllegalArgumentException if length is negative
     */
    record State(String path, StateKind kind, DataType type, Shape shape, int length) {
        State {
            Objects.requireNonNull(path, "path");
            Objects.requireNonNull(kind, "kind");
            Objects.requireNonNull(type, "type");
            Objects.requireNonNull(shape, "shape");
            if (length < 0) {
                throw new IllegalArgumentException("Negative state length");
            }
        }
    }

    /**
     * Immutable canonical metadata bytes and parsed state descriptors. Digest access is copied.
     *
     * @param bytes non-null canonical encoded metadata, copied on construction and access
     * @param states non-null ordered descriptors, copied into an immutable list
     * @param configDigest non-null caller's exact 32-byte configuration SHA-256, copied on
     *     construction and access
     * @param inputDigest optional exact 32-byte input-artifact SHA-256, copied when present
     * @throws NullPointerException if bytes, states, configDigest, or a state is null
     */
    record Metadata(byte[] bytes, List<State> states, byte[] configDigest, byte[] inputDigest) {
        Metadata {
            bytes = bytes.clone();
            states = List.copyOf(states);
            configDigest = configDigest.clone();
            inputDigest = inputDigest == null ? null : inputDigest.clone();
        }

        @Override public byte[] bytes() { return bytes.clone(); }
        @Override public byte[] configDigest() { return configDigest.clone(); }
        @Override public byte[] inputDigest() { return inputDigest == null ? null : inputDigest.clone(); }
    }

    /**
     * Encodes complete metadata after checking every artifact and canonical-payload bound.
     * Running bounds reject oversized inputs before allocating an entry-proportional path set or
     * metadata copy; path uniqueness is then checked before encoding any bytes.
     *
     * @param states ordered descriptors with unique exact paths
     * @param configDigest exact 32-byte caller configuration fingerprint
     * @param inputDigest optional exact 32-byte input fingerprint, or null
     * @param limits caller's artifact and output-byte limits
     * @return detached canonical metadata; never null
     * @throws IllegalArgumentException if metadata, Unicode, length, or a bound is invalid
     * @throws ArithmeticException if checked size arithmetic overflows
     * @throws NullPointerException if a required argument, digest, or state is null
     */
    static Metadata encode(List<State> states, byte[] configDigest, byte[] inputDigest,
                           ArtifactLimits limits) {
        Objects.requireNonNull(states, "states");
        Objects.requireNonNull(limits, "limits");
        checkDigest(configDigest);
        if (inputDigest != null) checkDigest(inputDigest);
        byte[] stableConfig = configDigest.clone();
        byte[] stableInput = inputDigest == null ? null : inputDigest.clone();
        if (states.size() > MAX_ORDINAL + 1L) throw new IllegalArgumentException("Too many states");
        long size = 49L + (stableInput == null ? 0 : 32);
        long payload = 0;
        preflight(states.size(), size, payload, limits);
        for (State state : states) {
            Objects.requireNonNull(state, "state");
            int pathBytes = pathLength(state.path());
            checkLength(state.type(), state.shape(), state.length());
            if (state.length() > limits.maxPayloadBytes())
                throw new IllegalArgumentException("State payload exceeds limits");
            size = Math.addExact(size, Math.addExact(18L + pathBytes,
                    Math.multiplyExact(8L, state.shape().rank())));
            payload = Math.addExact(payload, state.length());
            preflight(states.size(), size, payload, limits);
        }
        var paths = new HashSet<String>();
        for (State state : states) {
            if (!paths.add(state.path())) throw new IllegalArgumentException("Duplicate state path");
        }
        ByteBuffer out = ByteBuffer.allocate((int) size).order(ByteOrder.BIG_ENDIAN);
        out.put(MAGIC).putInt(VERSION).put(stableConfig).put((byte) (stableInput == null ? 0 : 1));
        if (stableInput != null) out.put(stableInput);
        out.putInt(states.size());
        for (State state : states) {
            byte[] path = state.path().getBytes(StandardCharsets.UTF_8);
            out.putInt(path.length).put(path).put(kindId(state.kind())).put(typeId(state.type()));
            out.putInt(state.shape().rank());
            for (long extent : state.shape().toLongArray()) out.putLong(extent);
            out.putLong(state.length());
        }
        return new Metadata(out.array(), states, stableConfig, stableInput);
    }

    /**
     * Validates an already detached artifact's exact schema and one-to-one ordinal payloads.
     *
     * @param envelope non-null opaque image to inspect without modifying it
     * @param limits caller-selected bounds applied to schema, payloads, manifest, and file size
     * @return validated detached model candidate
     * @throws ArtifactFormatException if any schema or payload correspondence is corrupt
     * @throws NullPointerException if the envelope or limits is null
     */
    static ModelArtifactCandidate decode(ArtifactEnvelope envelope, ArtifactLimits limits)
            throws ArtifactFormatException {
        Objects.requireNonNull(envelope, "envelope");
        Objects.requireNonNull(limits, "limits");
        if (!KIND.equals(envelope.kind()) || envelope.entries().isEmpty()
                || !SCHEMA.equals(envelope.entries().getFirst().key())) {
            throw new ArtifactFormatException("Not a model.v1 artifact with schema");
        }
        try {
            long stateBytes = 0;
            for (int i = 1; i < envelope.entries().size(); i++) {
                int size = envelope.entries().get(i).size();
                if (size > limits.maxPayloadBytes()) throw invalid();
                stateBytes = Math.addExact(stateBytes, size);
            }
            preflight(envelope.entries().size() - 1, envelope.entries().getFirst().size(),
                    stateBytes, limits);
            byte[] bytes = bytes(envelope.entries().getFirst());
            ByteBuffer in = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN);
            require(in, 45);
            byte[] magic = new byte[8];
            in.get(magic);
            if (!Arrays.equals(magic, MAGIC) || in.getInt() != VERSION) throw invalid();
            byte[] config = new byte[32];
            in.get(config);
            int present = Byte.toUnsignedInt(in.get());
            if (present > 1) throw invalid();
            byte[] input = null;
            if (present == 1) {
                require(in, 32);
                input = new byte[32];
                in.get(input);
            }
            require(in, 4);
            int count = in.getInt();
            if (count < 0 || count > MAX_ORDINAL + 1L || count + 1L > limits.maxEntries()
                    || count != envelope.entries().size() - 1
                    || count > in.remaining() / 18) throw invalid();
            var states = new ArrayList<State>(count);
            var paths = new HashSet<String>();
            for (int i = 0; i < count; i++) {
                require(in, 4);
                int pathSize = in.getInt();
                if (pathSize < 1 || pathSize > in.remaining() - 14) throw invalid();
                byte[] pathBytes = new byte[pathSize];
                in.get(pathBytes);
                String path = StandardCharsets.UTF_8.newDecoder()
                        .onMalformedInput(CodingErrorAction.REPORT)
                        .onUnmappableCharacter(CodingErrorAction.REPORT)
                        .decode(ByteBuffer.wrap(pathBytes)).toString();
                if (pathLength(path) != pathSize || !paths.add(path)) throw invalid();
                StateKind kind = kind(in.get());
                DataType type = type(in.get());
                int rank = in.getInt();
                if (rank < 0 || rank > in.remaining() / 8 - 1) throw invalid();
                long[] extents = new long[rank];
                for (int axis = 0; axis < rank; axis++) {
                    extents[axis] = in.getLong();
                    if (extents[axis] < 0) throw invalid();
                }
                long length = in.getLong();
                if (length < 0 || length > ARRAY_CEILING) throw invalid();
                Shape shape = Shape.of(extents);
                checkLength(type, shape, length);
                states.add(new State(path, kind, type, shape, (int) length));
                ArtifactEnvelope.Entry payload = envelope.entries().get(i + 1);
                if (!ordinal(i).equals(payload.key()) || payload.size() != length) throw invalid();
            }
            if (in.hasRemaining()) throw invalid();
            Metadata metadata = encode(states, config, input, limits);
            if (!Arrays.equals(bytes, metadata.bytes())) throw invalid();
            return new ModelArtifactCandidate(metadata, envelope);
        } catch (CharacterCodingException | IllegalArgumentException | ArithmeticException
                 | java.nio.BufferUnderflowException ex) {
            throw new ArtifactFormatException("Invalid model metadata or bounds", ex);
        }
    }

    /**
     * Names one payload by its zero-based encounter ordinal, independent of the NN path.
     *
     * @param index zero-based ordinal at most 99,999,999
     * @return fixed-width ASCII artifact key
     * @throws IllegalArgumentException if index is outside the supported ordinal range
     */
    static String ordinal(int index) {
        if (index < 0 || index > MAX_ORDINAL) throw new IllegalArgumentException("Invalid ordinal");
        return "state/" + String.format(java.util.Locale.ROOT, "%08d", index);
    }

    /**
     * Checks final opaque artifact limits before any Engine call or payload copy.
     *
     * @param count number of state payloads
     * @param schemaBytes canonical metadata length in bytes
     * @param stateBytes sum of all state payload lengths, counting aliases
     * @param limits caller-selected artifact bounds
     * @throws IllegalArgumentException if any model envelope bound is exceeded
     * @throws ArithmeticException if aggregate, manifest, or final file arithmetic overflows
     */
    static void preflight(int count, long schemaBytes, long stateBytes, ArtifactLimits limits) {
        long aggregate = Math.addExact(schemaBytes, stateBytes);
        long manifest = Math.addExact(66L, Math.multiplyExact(58L, count));
        long file = Math.addExact(Math.addExact(48L, manifest), aggregate);
        if (count > MAX_ORDINAL + 1L || count + 1L > limits.maxEntries()
                || limits.maxKeyBytes() < 14 && count > 0 || limits.maxKeyBytes() < 8
                || schemaBytes > ARRAY_CEILING || schemaBytes > limits.maxPayloadBytes()
                || stateBytes < 0 || aggregate > limits.maxAggregatePayloadBytes()
                || manifest > limits.maxManifestBytes() || file > limits.maxFileBytes()) {
            throw new IllegalArgumentException("Model artifact exceeds caller limits");
        }
    }

    /**
     * Computes the SHA-256 fingerprint of canonical metadata bytes.
     *
     * @param metadata validated canonical metadata
     * @return fresh 32-byte digest
     * @throws NullPointerException if metadata is null
     */
    static byte[] fingerprint(Metadata metadata) {
        try {
            return MessageDigest.getInstance("SHA-256").digest(metadata.bytes());
        } catch (NoSuchAlgorithmException ex) {
            throw new AssertionError(ex);
        }
    }

    private static void checkDigest(byte[] digest) {
        Objects.requireNonNull(digest, "digest");
        if (digest.length != 32) throw new IllegalArgumentException("SHA-256 digest must be 32 bytes");
    }

    private static int pathLength(String path) {
        if (path.isEmpty()) throw new IllegalArgumentException("Empty state path");
        long bytes = 0;
        boolean nonBlankSegment = false;
        for (int i = 0; i < path.length(); i++) {
            char ch = path.charAt(i);
            if (ch == '.') {
                if (!nonBlankSegment) throw new IllegalArgumentException("Blank state segment");
                nonBlankSegment = false;
                bytes++;
                if (bytes > ARRAY_CEILING) throw new IllegalArgumentException("State path too long");
                continue;
            }
            int codePoint = ch;
            if (ch < 0x80) bytes++;
            else if (ch < 0x800) bytes += 2;
            else if (Character.isHighSurrogate(ch)) {
                if (i + 1 >= path.length() || !Character.isLowSurrogate(path.charAt(i + 1)))
                    throw new IllegalArgumentException("Malformed Unicode state path");
                codePoint = Character.toCodePoint(ch, path.charAt(i + 1));
                bytes += 4;
                i++;
            } else if (Character.isLowSurrogate(ch)) {
                throw new IllegalArgumentException("Malformed Unicode state path");
            } else bytes += 3;
            if (!Character.isWhitespace(codePoint)) nonBlankSegment = true;
            if (bytes > ARRAY_CEILING) throw new IllegalArgumentException("State path too long");
        }
        if (!nonBlankSegment) throw new IllegalArgumentException("Blank state segment");
        return (int) bytes;
    }

    private static void checkLength(DataType type, Shape shape, long length) {
        if (!shape.isFullyStatic()) throw new IllegalArgumentException("Dynamic state shape");
        long expected = Math.multiplyExact(shape.knownElementCount().orElseThrow(), type.byteWidth());
        if (expected > ARRAY_CEILING || length != expected)
            throw new IllegalArgumentException("Noncanonical state byte length");
    }

    private static byte kindId(StateKind kind) {
        return switch (kind) { case PARAMETER -> 1; case BUFFER -> 2; };
    }

    private static StateKind kind(byte id) {
        return switch (id) {
            case 1 -> StateKind.PARAMETER;
            case 2 -> StateKind.BUFFER;
            default -> throw new IllegalArgumentException("Unknown state kind ID");
        };
    }

    private static byte typeId(DataType type) {
        return switch (type) {
            case FLOAT64 -> 1; case FLOAT32 -> 2; case BFLOAT16 -> 3;
            case INT32 -> 4; case INT64 -> 5; case BOOL -> 6; case FLOAT16 -> 7;
        };
    }

    private static DataType type(byte id) {
        return switch (id) {
            case 1 -> DataType.FLOAT64; case 2 -> DataType.FLOAT32;
            case 3 -> DataType.BFLOAT16; case 4 -> DataType.INT32;
            case 5 -> DataType.INT64; case 6 -> DataType.BOOL;
            case 7 -> DataType.FLOAT16;
            default -> throw new IllegalArgumentException("Unknown data type ID");
        };
    }

    private static void require(ByteBuffer buffer, int bytes) {
        if (bytes < 0 || buffer.remaining() < bytes) throw new IllegalArgumentException("Truncated metadata");
    }

    private static ArtifactFormatException invalid() { return new ArtifactFormatException("Invalid model metadata"); }

    private static byte[] bytes(ArtifactEnvelope.Entry entry) {
        byte[] copy = new byte[entry.size()];
        entry.payload().get(copy);
        return copy;
    }
}

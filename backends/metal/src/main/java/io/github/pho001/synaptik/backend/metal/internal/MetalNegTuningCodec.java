package io.github.pho001.synaptik.backend.metal.internal;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Arrays;
import java.util.Objects;
import java.util.Optional;
import java.util.zip.CRC32;

/**
 * Canonical bounded session codec for Metal NEG compatibility, candidates, and decisions.
 *
 * <p>The format is a backend-local defensive transport, not the tools-owned workload-cache
 * artifact. It contains no executable, native handle, file path, objective, sample, or timing.
 * Decoding is fail-closed and performs no I/O or native work.</p>
 */
final class MetalNegTuningCodec {
    static final int MAX_WORKLOAD_BYTES = 64;
    static final int MAX_COMPATIBILITY_BYTES = 128;
    static final int MAX_CANDIDATE_BYTES = 32;
    static final int MAX_DECISION_BYTES = 256;

    private static final int COMPATIBILITY_MAGIC = 0x4d4e434d; // MNCM
    private static final int CANDIDATE_MAGIC = 0x4d4e4341; // MNCA
    private static final int DECISION_MAGIC = 0x4d4e4443; // MNDC
    private static final int CODEC_VERSION = 1;
    private static final int SESSION_SCOPE = 1;

    /**
     * Encodes exact versioned workload and live-session target compatibility.
     *
     * @param compatibility non-null compatibility to encode
     * @return fresh canonical bounded bytes; never {@code null}
     * @throws NullPointerException if {@code compatibility} is {@code null}
     * @throws IllegalArgumentException if the result exceeds its defensive bound
     */
    byte[] encodeCompatibility(MetalNegTuningBatch.Compatibility compatibility) {
        Objects.requireNonNull(compatibility, "compatibility");
        return write(MAX_COMPATIBILITY_BYTES, output -> {
            output.writeInt(COMPATIBILITY_MAGIC);
            output.writeInt(CODEC_VERSION);
            output.writeInt(SESSION_SCOPE);
            output.writeInt(compatibility.schemaVersion());
            output.writeInt(compatibility.candidateSchemaVersion());
            output.writeInt(compatibility.routePolicyVersion());
            output.writeInt(compatibility.target().abiVersion());
            output.writeLong(compatibility.target().sessionNonce().highBits());
            output.writeLong(compatibility.target().sessionNonce().lowBits());
            byte[] workload = compatibility.workload().bytes();
            output.writeInt(workload.length);
            output.write(workload);
        });
    }

    /**
     * Encodes one schema-local complete candidate identity.
     *
     * @param candidate non-null candidate to encode
     * @return fresh canonical bounded bytes; never {@code null}
     * @throws NullPointerException if {@code candidate} is {@code null}
     */
    byte[] encodeCandidate(MetalNegTuningBatch.Candidate candidate) {
        Objects.requireNonNull(candidate, "candidate");
        return write(MAX_CANDIDATE_BYTES, output -> {
            output.writeInt(CANDIDATE_MAGIC);
            output.writeInt(CODEC_VERSION);
            output.writeInt(MetalNegTuningBatch.CANDIDATE_SCHEMA_VERSION);
            output.writeInt(candidate.wireIdentity());
        });
    }

    /**
     * Encodes one structurally valid Metal decision with a trailing checksum.
     *
     * @param decision non-null decision to encode
     * @return fresh canonical bounded decision bytes; never {@code null}
     * @throws NullPointerException if {@code decision} is {@code null}
     * @throws IllegalArgumentException if its schema is not the current Metal schema
     */
    byte[] encodeDecision(MetalNegTuningDecision decision) {
        Objects.requireNonNull(decision, "decision");
        if (decision.candidateSchemaVersion()
                != MetalNegTuningBatch.CANDIDATE_SCHEMA_VERSION) {
            throw new IllegalArgumentException("unsupported Metal NEG decision schema");
        }
        byte[] compatibility = encodeCompatibility(decision.compatibility());
        byte[] candidate = encodeCandidate(decision.selectedCandidate());
        byte[] payload = write(MAX_DECISION_BYTES - Integer.BYTES, output -> {
            output.writeInt(DECISION_MAGIC);
            output.writeInt(CODEC_VERSION);
            output.writeInt(SESSION_SCOPE);
            output.writeInt(compatibility.length);
            output.write(compatibility);
            output.writeInt(candidate.length);
            output.write(candidate);
        });
        CRC32 checksum = new CRC32();
        checksum.update(payload);
        return write(MAX_DECISION_BYTES, output -> {
            output.write(payload);
            output.writeInt((int) checksum.getValue());
        });
    }

    /**
     * Decodes and authenticates bytes against one freshly generated current batch.
     *
     * @param encoded non-null untrusted bytes; the array is never retained or mutated
     * @param freshBatch non-null batch regenerated from current live analysis facts
     * @return the current compatible decision, or empty for wrong magic, version, scope,
     *     malformed/truncated/trailing/corrupt bytes, changed workload/session, or an unknown or
     *     pruned candidate
     * @throws NullPointerException if an argument is {@code null}
     */
    Optional<MetalNegTuningDecision> decodeDecision(
            byte[] encoded, MetalNegTuningBatch freshBatch) {
        Objects.requireNonNull(encoded, "encoded");
        Objects.requireNonNull(freshBatch, "freshBatch");
        if (encoded.length < 7 * Integer.BYTES || encoded.length > MAX_DECISION_BYTES) {
            return Optional.empty();
        }
        try {
            int payloadLength = encoded.length - Integer.BYTES;
            CRC32 checksum = new CRC32();
            checksum.update(encoded, 0, payloadLength);
            int storedChecksum = readInt(encoded, payloadLength);
            if ((int) checksum.getValue() != storedChecksum) return Optional.empty();

            var input = new DataInputStream(
                    new ByteArrayInputStream(encoded, 0, payloadLength));
            if (input.readInt() != DECISION_MAGIC
                    || input.readInt() != CODEC_VERSION
                    || input.readInt() != SESSION_SCOPE) {
                return Optional.empty();
            }
            int compatibilityLength = input.readInt();
            if (compatibilityLength <= 0
                    || compatibilityLength > MAX_COMPATIBILITY_BYTES
                    || compatibilityLength > input.available() - Integer.BYTES) {
                return Optional.empty();
            }
            byte[] compatibility = input.readNBytes(compatibilityLength);
            int candidateLength = input.readInt();
            if (candidateLength <= 0 || candidateLength > MAX_CANDIDATE_BYTES
                    || candidateLength != input.available()) {
                return Optional.empty();
            }
            byte[] candidate = input.readNBytes(candidateLength);
            if (input.available() != 0
                    || !Arrays.equals(compatibility,
                            encodeCompatibility(freshBatch.compatibility()))) {
                return Optional.empty();
            }
            Optional<MetalNegTuningBatch.Candidate> selected = decodeCandidate(candidate);
            if (selected.isEmpty() || freshBatch.find(selected.orElseThrow()).isEmpty()) {
                return Optional.empty();
            }
            return Optional.of(new MetalNegTuningDecision(
                    MetalNegTuningBatch.CANDIDATE_SCHEMA_VERSION,
                    freshBatch.compatibility(), selected.orElseThrow()));
        } catch (IOException | IllegalArgumentException failure) {
            return Optional.empty();
        }
    }

    private Optional<MetalNegTuningBatch.Candidate> decodeCandidate(byte[] encoded)
            throws IOException {
        if (encoded.length != 4 * Integer.BYTES) return Optional.empty();
        var input = new DataInputStream(new ByteArrayInputStream(encoded));
        if (input.readInt() != CANDIDATE_MAGIC
                || input.readInt() != CODEC_VERSION
                || input.readInt() != MetalNegTuningBatch.CANDIDATE_SCHEMA_VERSION) {
            return Optional.empty();
        }
        return MetalNegTuningBatch.Candidate.fromWireIdentity(input.readInt());
    }

    private static int readInt(byte[] bytes, int offset) {
        return (bytes[offset] & 0xff) << 24
                | (bytes[offset + 1] & 0xff) << 16
                | (bytes[offset + 2] & 0xff) << 8
                | bytes[offset + 3] & 0xff;
    }

    private static byte[] write(int maximumBytes, IoWriter writer) {
        try {
            var bytes = new ByteArrayOutputStream();
            try (var output = new DataOutputStream(bytes)) {
                writer.write(output);
            }
            byte[] encoded = bytes.toByteArray();
            if (encoded.length > maximumBytes) {
                throw new IllegalArgumentException("Metal NEG codec bound exceeded");
            }
            return encoded;
        } catch (IOException impossible) {
            throw new UncheckedIOException(impossible);
        }
    }

    @FunctionalInterface
    private interface IoWriter {
        void write(DataOutputStream output) throws IOException;
    }
}

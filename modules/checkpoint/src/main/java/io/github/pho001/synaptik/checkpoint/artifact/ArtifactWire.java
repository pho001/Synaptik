package io.github.pho001.synaptik.checkpoint.artifact;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Encodes the canonical version-one big-endian header, manifest, and contiguous opaque payloads.
 * Unkeyed SHA-256 digests detect accidental changes to the manifest and payload bytes; they do
 * not authenticate a writer. Reading checks each encoded size against caller limits and refuses
 * an incomplete or noncanonical image before returning it.
 */
final class ArtifactWire {
    private static final byte[] MAGIC = "SYNART01".getBytes(StandardCharsets.US_ASCII);
    private static final int VERSION = 1;
    private static final int HEADER_BYTES = 48;
    private static final int HASH_BYTES = 32;

    private ArtifactWire() {
    }

    static void validateName(String value, boolean path) {
        if (value.isEmpty()) {
            throw new IllegalArgumentException("Empty artifact name");
        }
        String[] segments = value.split("/", -1);
        if (!path && segments.length != 1) {
            throw new IllegalArgumentException("Artifact kind must be one segment");
        }
        for (String segment : segments) {
            if (segment.isEmpty() || segment.equals(".") || segment.equals("..")) {
                throw new IllegalArgumentException("Unsafe artifact name segment");
            }
            for (int i = 0; i < segment.length(); i++) {
                char ch = segment.charAt(i);
                boolean allowed = (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z')
                        || (ch >= '0' && ch <= '9') || ch == '_' || ch == '-' || ch == '.';
                if (!allowed) {
                    throw new IllegalArgumentException("Noncanonical artifact name");
                }
            }
        }
    }

    static void write(FileChannel channel, ArtifactEnvelope image, ArtifactLimits limits)
            throws IOException {
        byte[] manifest = manifest(image, limits);
        DataOutputStream out = new DataOutputStream(Channels.newOutputStream(channel));
        out.write(MAGIC);
        out.writeInt(VERSION);
        out.writeInt(manifest.length);
        out.write(hash(manifest));
        out.write(manifest);
        for (ArtifactEnvelope.Entry entry : image.entries()) {
            out.write(entry.bytes());
        }
        out.flush();
    }

    static ArtifactEnvelope read(Path path, ArtifactLimits limits) throws IOException {
        return inspect(path, limits, null);
    }

    static void validate(Path path, ArtifactLimits limits, ArtifactEnvelope expected)
            throws IOException {
        inspect(path, limits, expected);
    }

    private static ArtifactEnvelope inspect(Path path, ArtifactLimits limits,
                                            ArtifactEnvelope expected) throws IOException {
        if (!Files.isRegularFile(path, LinkOption.NOFOLLOW_LINKS)) {
            throw new ArtifactFormatException("Artifact must be a regular file, not a symlink");
        }
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ,
                LinkOption.NOFOLLOW_LINKS)) {
            long fileSize = channel.size();
            if (fileSize < HEADER_BYTES || fileSize > limits.maxFileBytes()) {
                throw new ArtifactFormatException("Artifact file length exceeds bounds");
            }
            DataInputStream in = new DataInputStream(Channels.newInputStream(channel));
            byte[] magic = new byte[MAGIC.length];
            in.readFully(magic);
            if (!Arrays.equals(magic, MAGIC)) {
                throw new ArtifactFormatException("Invalid artifact magic");
            }
            if (in.readInt() != VERSION) {
                throw new ArtifactFormatException("Unsupported artifact version");
            }
            int manifestLength = in.readInt();
            if (manifestLength < 8 || manifestLength > limits.maxManifestBytes()
                    || manifestLength > fileSize - HEADER_BYTES) {
                throw new ArtifactFormatException("Invalid manifest length");
            }
            byte[] expectedManifestHash = new byte[HASH_BYTES];
            in.readFully(expectedManifestHash);
            byte[] manifest = new byte[manifestLength];
            in.readFully(manifest);
            if (!MessageDigest.isEqual(expectedManifestHash, hash(manifest))) {
                throw new ArtifactFormatException("Manifest checksum mismatch");
            }
            Parsed parsed = parseManifest(manifest, limits);
            long exactLength = checkedLength(manifestLength, parsed.aggregateBytes);
            if (exactLength != fileSize) {
                throw new ArtifactFormatException("Artifact length does not match manifest");
            }
            if (expected != null && (!expected.kind().equals(parsed.kind)
                    || expected.entries().size() != parsed.entries.size())) {
                throw new ArtifactFormatException("Temporary artifact differs from input image");
            }
            List<ArtifactEnvelope.Entry> entries = expected == null
                    ? new ArrayList<>(parsed.entries.size()) : null;
            byte[] scratch = expected == null ? null : new byte[8192];
            for (int i = 0; i < parsed.entries.size(); i++) {
                Descriptor descriptor = parsed.entries.get(i);
                if (expected == null) {
                    byte[] payload = new byte[descriptor.length];
                    in.readFully(payload);
                    if (!MessageDigest.isEqual(descriptor.hash, hash(payload))) {
                        throw new ArtifactFormatException("Payload checksum mismatch: " + descriptor.key);
                    }
                    entries.add(ArtifactEnvelope.Entry.fromOwnedBytes(descriptor.key, payload));
                } else {
                    ArtifactEnvelope.Entry original = expected.entries().get(i);
                    if (!original.key().equals(descriptor.key) || original.size() != descriptor.length) {
                        throw new ArtifactFormatException("Temporary artifact differs from input image");
                    }
                    MessageDigest digest = sha256();
                    int offset = 0;
                    while (offset < descriptor.length) {
                        int chunk = Math.min(scratch.length, descriptor.length - offset);
                        in.readFully(scratch, 0, chunk);
                        digest.update(scratch, 0, chunk);
                        if (Arrays.mismatch(scratch, 0, chunk, original.bytes(), offset,
                                offset + chunk) >= 0) {
                            throw new ArtifactFormatException("Temporary payload differs from input image");
                        }
                        offset += chunk;
                    }
                    if (!MessageDigest.isEqual(descriptor.hash, digest.digest())) {
                        throw new ArtifactFormatException("Payload checksum mismatch: " + descriptor.key);
                    }
                }
            }
            if (in.read() != -1) {
                throw new ArtifactFormatException("Trailing artifact bytes");
            }
            return expected == null ? new ArtifactEnvelope(parsed.kind, entries) : null;
        } catch (EOFException ex) {
            throw new ArtifactFormatException("Truncated artifact", ex);
        }
    }

    private static byte[] manifest(ArtifactEnvelope image, ArtifactLimits limits) throws IOException {
        byte[] kind = image.kind().getBytes(StandardCharsets.US_ASCII);
        if (kind.length > limits.maxKeyBytes() || image.entries().size() > limits.maxEntries()) {
            throw new ArtifactFormatException("Artifact kind or entry count exceeds limits");
        }
        long manifestSize = 8L + kind.length;
        long aggregate = 0;
        for (ArtifactEnvelope.Entry entry : image.entries()) {
            byte[] key = entry.key().getBytes(StandardCharsets.US_ASCII);
            if (key.length > limits.maxKeyBytes() || entry.size() > limits.maxPayloadBytes()) {
                throw new ArtifactFormatException("Artifact key or payload exceeds limits");
            }
            manifestSize = checkedAdd(manifestSize, 44L + key.length);
            aggregate = checkedAdd(aggregate, entry.size());
        }
        if (manifestSize > limits.maxManifestBytes() || aggregate > limits.maxAggregatePayloadBytes()
                || checkedLength(manifestSize, aggregate) > limits.maxFileBytes()) {
            throw new ArtifactFormatException("Artifact exceeds configured limits");
        }
        ByteArrayOutputStream bytes = new ByteArrayOutputStream((int) manifestSize);
        DataOutputStream out = new DataOutputStream(bytes);
        out.writeInt(kind.length);
        out.write(kind);
        out.writeInt(image.entries().size());
        for (ArtifactEnvelope.Entry entry : image.entries()) {
            byte[] key = entry.key().getBytes(StandardCharsets.US_ASCII);
            out.writeInt(key.length);
            out.write(key);
            out.writeLong(entry.size());
            out.write(hash(entry.bytes()));
        }
        out.flush();
        return bytes.toByteArray();
    }

    private static Parsed parseManifest(byte[] manifest, ArtifactLimits limits) throws IOException {
        DataInputStream in = new DataInputStream(new ByteArrayInputStream(manifest));
        try {
            String kind = readName(in, limits.maxKeyBytes(), false);
            int count = in.readInt();
            if (count < 0 || count > limits.maxEntries() || count > in.available() / 45) {
                throw new ArtifactFormatException("Entry count exceeds limits");
            }
            List<Descriptor> entries = new ArrayList<>(count);
            String previous = null;
            long aggregate = 0;
            for (int i = 0; i < count; i++) {
                String key = readName(in, limits.maxKeyBytes(), true);
                if (previous != null && previous.compareTo(key) >= 0) {
                    throw new ArtifactFormatException("Entries not strictly sorted or duplicate");
                }
                long length = in.readLong();
                if (length < 0 || length > limits.maxPayloadBytes()
                        || length > Integer.MAX_VALUE - 8) {
                    throw new ArtifactFormatException("Payload length exceeds limits");
                }
                aggregate = checkedAdd(aggregate, length);
                if (aggregate > limits.maxAggregatePayloadBytes()) {
                    throw new ArtifactFormatException("Aggregate payload exceeds limits");
                }
                byte[] digest = new byte[HASH_BYTES];
                in.readFully(digest);
                entries.add(new Descriptor(key, (int) length, digest));
                previous = key;
            }
            if (in.read() != -1) {
                throw new ArtifactFormatException("Trailing manifest bytes");
            }
            return new Parsed(kind, entries, aggregate);
        } catch (EOFException ex) {
            throw new ArtifactFormatException("Truncated manifest", ex);
        }
    }

    private static String readName(DataInputStream in, int maxBytes, boolean path) throws IOException {
        int length = in.readInt();
        if (length < 1 || length > maxBytes || length > in.available()) {
            throw new ArtifactFormatException("Invalid artifact name length");
        }
        byte[] bytes = new byte[length];
        in.readFully(bytes);
        for (byte value : bytes) {
            if (value < 0) {
                throw new ArtifactFormatException("Non-ASCII artifact name");
            }
        }
        String name = new String(bytes, StandardCharsets.US_ASCII);
        try {
            validateName(name, path);
        } catch (IllegalArgumentException ex) {
            throw new ArtifactFormatException("Invalid artifact name", ex);
        }
        return name;
    }

    private static long checkedLength(long manifestSize, long aggregate) throws ArtifactFormatException {
        return checkedAdd(checkedAdd(HEADER_BYTES, manifestSize), aggregate);
    }

    private static long checkedAdd(long left, long right) throws ArtifactFormatException {
        try {
            return Math.addExact(left, right);
        } catch (ArithmeticException ex) {
            throw new ArtifactFormatException("Artifact length overflow", ex);
        }
    }

    private static byte[] hash(byte[] bytes) {
        return sha256().digest(bytes);
    }

    private static MessageDigest sha256() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException ex) {
            throw new AssertionError("SHA-256 unavailable", ex);
        }
    }

    private record Descriptor(String key, int length, byte[] hash) {
    }

    private record Parsed(String kind, List<Descriptor> entries, long aggregateBytes) {
    }
}

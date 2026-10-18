package io.github.pho001.synaptik.checkpoint.artifact;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ReadOnlyBufferException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ArtifactWireTest {
    private static final ArtifactLimits LIMITS = new ArtifactLimits(4096, 2048, 8, 64, 1024, 2048);

    @TempDir Path directory;

    @Test
    void roundtripIsCanonicalDetachedAndSupportsEmptyPayloads() throws IOException {
        byte[] mutable = {1, 2, -1};
        ArtifactEnvelope image = new ArtifactEnvelope("model.v1", List.of(
                new ArtifactEnvelope.Entry("z/weights", mutable),
                new ArtifactEnvelope.Entry("a", new byte[0])));
        mutable[0] = 99;
        assertEquals(List.of("a", "z/weights"), image.entries().stream().map(
                ArtifactEnvelope.Entry::key).toList());
        assertArrayEquals(new byte[] {1, 2, -1}, bytes(image.entries().get(1)));
        assertThrows(ReadOnlyBufferException.class, () -> image.entries().get(1).payload().put((byte) 0));
        assertThrows(UnsupportedOperationException.class, () -> image.entries().clear());

        Path first = directory.resolve("first.art");
        Path second = directory.resolve("second.art");
        ArtifactFile.save(first, image, LIMITS);
        ArtifactFile.save(second, new ArtifactEnvelope("model.v1", List.of(
                new ArtifactEnvelope.Entry("a", new byte[0]),
                new ArtifactEnvelope.Entry("z/weights", new byte[] {1, 2, -1}))), LIMITS);
        assertArrayEquals(Files.readAllBytes(first), Files.readAllBytes(second));
        ArtifactEnvelope decoded = ArtifactFile.read(first, LIMITS);
        assertEquals("model.v1", decoded.kind());
        assertEquals(0, decoded.entries().get(0).size());
        assertArrayEquals(new byte[] {1, 2, -1}, bytes(decoded.entries().get(1)));
    }

    @Test
    void rejectsUnsafeNamesAndDuplicateKeysBeforeSave() {
        for (String name : List.of("", ".", "..", "a//b", "a/", "/a", "a/../b",
                "a/./b", "a\\b", "a b", "é", "a\u0000b")) {
            assertThrows(IllegalArgumentException.class, () -> new ArtifactEnvelope.Entry(name, new byte[0]));
        }
        assertThrows(IllegalArgumentException.class, () -> new ArtifactEnvelope("a/b", List.of()));
        assertThrows(IllegalArgumentException.class, () -> new ArtifactEnvelope("a", List.of(
                new ArtifactEnvelope.Entry("x", new byte[0]),
                new ArtifactEnvelope.Entry("x", new byte[0]))));
    }

    @Test
    void checksWriterAndReaderLimitsAndExactLengths() throws IOException {
        ArtifactEnvelope image = new ArtifactEnvelope("k", List.of(
                new ArtifactEnvelope.Entry("x", new byte[] {4, 5}),
                new ArtifactEnvelope.Entry("y", new byte[] {6})));
        Path file = directory.resolve("bounded.art");
        ArtifactFile.save(file, image, LIMITS);
        byte[] valid = Files.readAllBytes(file);
        int manifestLength = ByteBuffer.wrap(valid).getInt(12);
        assertEquals(48L + manifestLength + 3, valid.length);
        assertThrows(ArtifactFormatException.class, () -> ArtifactFile.save(directory.resolve("small.art"),
                image, limits(valid.length - 1, 2048, 8, 64, 1024, 2048)));
        assertThrows(ArtifactFormatException.class, () -> ArtifactFile.read(file,
                limits(valid.length - 1, 2048, 8, 64, 1024, 2048)));
        assertThrows(ArtifactFormatException.class, () -> ArtifactFile.read(file,
                limits(4096, manifestLength - 1, 8, 64, 1024, 2048)));
        assertThrows(ArtifactFormatException.class, () -> ArtifactFile.read(file,
                limits(4096, 2048, 1, 64, 1024, 2048)));
        assertThrows(ArtifactFormatException.class, () -> ArtifactFile.read(file,
                limits(4096, 2048, 8, 64, 1, 2048)));
        assertThrows(ArtifactFormatException.class, () -> ArtifactFile.read(file,
                limits(4096, 2048, 8, 64, 1024, 2)));
        ArtifactEnvelope longKey = new ArtifactEnvelope("kind", List.of(
                new ArtifactEnvelope.Entry("long/key", new byte[0])));
        assertThrows(ArtifactFormatException.class, () -> ArtifactFile.save(directory.resolve("key.art"),
                longKey, limits(4096, 2048, 8, 3, 1024, 2048)));
        assertThrows(IllegalArgumentException.class, () -> limits(4096, 2048, 8, 64,
                Integer.MAX_VALUE, 2048));
        assertThrows(IllegalArgumentException.class, () -> limits(Long.MAX_VALUE, 8, -1, 1, 0, 0));
        assertFalse(Files.exists(directory.resolve("small.art")));
    }

    @Test
    void rejectsHeaderManifestPayloadAndFileCorruption() throws Exception {
        byte[] valid = validBytes();
        int manifestLength = ByteBuffer.wrap(valid).getInt(12);
        List<byte[]> corrupt = new ArrayList<>();
        corrupt.add(mutate(valid, 0, (byte) 'X')); // magic
        corrupt.add(setInt(valid, 8, 2)); // version
        corrupt.add(setInt(valid, 12, -1)); // manifest length
        corrupt.add(setInt(valid, 12, Integer.MAX_VALUE));
        corrupt.add(flip(valid, 16)); // manifest checksum
        corrupt.add(flip(valid, 48)); // manifest without resealing
        corrupt.add(mutate(valid, 48 + manifestLength, (byte) 99)); // payload checksum
        corrupt.add(Arrays.copyOf(valid, valid.length - 1));
        corrupt.add(Arrays.copyOf(valid, valid.length + 1));
        corrupt.add(Arrays.copyOf(valid, 20));
        for (int i = 0; i < corrupt.size(); i++) {
            Path file = directory.resolve("corrupt-" + i);
            Files.write(file, corrupt.get(i));
            assertThrows(ArtifactFormatException.class, () -> ArtifactFile.read(file, LIMITS),
                    "corruption case " + i);
        }
        assertTrue(Files.size(directory.resolve("corrupt-0")) > 0);
    }

    @Test
    void rejectsIntegrityCheckedNoncanonicalManifestFieldsAndOverflow() throws Exception {
        byte[] valid = validBytes(); // kind k, entries a and b, each one byte
        int firstKeyLengthOffset = 48 + 4 + 1 + 4;
        int firstKeyOffset = firstKeyLengthOffset + 4;
        int secondKeyLengthOffset = firstKeyOffset + 1 + 8 + 32;
        int secondKeyOffset = secondKeyLengthOffset + 4;
        int firstLengthOffset = firstKeyOffset + 1;
        List<byte[]> invalid = new ArrayList<>();
        invalid.add(reseal(setInt(valid, 48, 0))); // empty kind
        invalid.add(reseal(mutate(valid, 52, (byte) '/'))); // invalid kind
        invalid.add(reseal(setInt(valid, 53, Integer.MAX_VALUE))); // impossible entry count
        invalid.add(reseal(setInt(valid, firstKeyLengthOffset, -1)));
        invalid.add(reseal(mutate(valid, firstKeyOffset, (byte) '.')));
        invalid.add(reseal(mutate(valid, secondKeyOffset, (byte) 'a'))); // duplicate
        invalid.add(reseal(mutate(valid, firstKeyOffset, (byte) 'z'))); // unsorted
        invalid.add(reseal(mutate(valid, firstKeyOffset, (byte) 0x80))); // non-ASCII
        invalid.add(reseal(setLong(valid, firstLengthOffset, Long.MAX_VALUE)));
        invalid.add(reseal(setLong(valid, firstLengthOffset, -1)));
        invalid.add(reseal(setInt(valid, 53, -1)));
        invalid.add(reseal(setInt(valid, 53, 0))); // trailing manifest data
        byte[] truncatedManifest = setInt(valid, 12, ByteBuffer.wrap(valid).getInt(12) - 1);
        invalid.add(truncatedManifest);
        for (int i = 0; i < invalid.size(); i++) {
            Path file = directory.resolve("manifest-" + i);
            Files.write(file, invalid.get(i));
            assertThrows(ArtifactFormatException.class, () -> ArtifactFile.read(file, LIMITS),
                    "manifest case " + i);
        }
    }

    @Test
    void rejectsIntegrityCheckedUnsafeKeySegments() throws Exception {
        Path file = directory.resolve("segments.art");
        ArtifactFile.save(file, new ArtifactEnvelope("k", List.of(
                new ArtifactEnvelope.Entry("a/xx/b", new byte[] {7}))), LIMITS);
        byte[] valid = Files.readAllBytes(file);
        int keyOffset = 48 + 4 + 1 + 4 + 4;
        byte[] emptySegment = mutate(valid, keyOffset + 2, (byte) '/');
        byte[] dotSegment = mutate(valid, keyOffset + 3, (byte) '/');
        byte[] parentSegment = valid.clone();
        parentSegment[keyOffset + 2] = '.';
        parentSegment[keyOffset + 3] = '.';
        for (byte[] malformed : List.of(emptySegment, dotSegment, parentSegment)) {
            Path corrupt = directory.resolve("unsafe-" + malformed[keyOffset + 2]);
            Files.write(corrupt, reseal(malformed));
            assertThrows(ArtifactFormatException.class, () -> ArtifactFile.read(corrupt, LIMITS));
        }
    }

    private byte[] validBytes() throws IOException {
        Path path = directory.resolve("valid.art");
        ArtifactFile.save(path, new ArtifactEnvelope("k", List.of(
                new ArtifactEnvelope.Entry("a", new byte[] {1}),
                new ArtifactEnvelope.Entry("b", new byte[] {2}))), LIMITS);
        return Files.readAllBytes(path);
    }

    private static ArtifactLimits limits(long file, int manifest, int count, int key, int payload,
                                         long aggregate) {
        return new ArtifactLimits(file, manifest, count, key, payload, aggregate);
    }

    private static byte[] bytes(ArtifactEnvelope.Entry entry) {
        ByteBuffer buffer = entry.payload();
        byte[] result = new byte[buffer.remaining()];
        buffer.get(result);
        return result;
    }

    private static byte[] mutate(byte[] source, int offset, byte value) {
        byte[] result = source.clone();
        result[offset] = value;
        return result;
    }

    private static byte[] flip(byte[] source, int offset) {
        byte[] result = source.clone();
        result[offset] ^= 1;
        return result;
    }

    private static byte[] setInt(byte[] source, int offset, int value) {
        byte[] result = source.clone();
        ByteBuffer.wrap(result).putInt(offset, value);
        return result;
    }

    private static byte[] setLong(byte[] source, int offset, long value) {
        byte[] result = source.clone();
        ByteBuffer.wrap(result).putLong(offset, value);
        return result;
    }

    private static byte[] reseal(byte[] source) throws NoSuchAlgorithmException {
        int manifestLength = ByteBuffer.wrap(source).getInt(12);
        byte[] result = source.clone();
        byte[] hash = MessageDigest.getInstance("SHA-256").digest(
                Arrays.copyOfRange(result, 48, 48 + manifestLength));
        System.arraycopy(hash, 0, result, 16, hash.length);
        return result;
    }
}

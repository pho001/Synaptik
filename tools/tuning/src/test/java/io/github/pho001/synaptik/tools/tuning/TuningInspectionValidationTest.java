package io.github.pho001.synaptik.tools.tuning;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.FileTime;
import java.nio.file.attribute.PosixFileAttributeView;
import java.nio.file.attribute.PosixFilePermission;
import java.security.MessageDigest;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

final class TuningInspectionValidationTest {
    @TempDir Path temporary;

    @Test
    void missingPathsAreReportedWithoutCreatingAnything() throws Exception {
        Path workload = temporary.resolve("absent-workload");
        Path plan = temporary.resolve("absent-plan");
        FileTime before = Files.getLastModifiedTime(temporary);
        assertEquals(TuningInspection.ArtifactStatus.MISSING,
                TuningInspection.inspectWorkloadCache(workload).status());
        assertEquals(TuningInspection.ArtifactStatus.MISSING,
                TuningInspection.inspectModelPlanCache(plan).status());
        assertFalse(Files.exists(workload));
        assertFalse(Files.exists(plan));
        assertEquals(before, Files.getLastModifiedTime(temporary));
        try (var entries = Files.list(temporary)) {
            assertEquals(0, entries.count());
        }
    }

    @Test
    void byteBoundsAreClassifiedBeforeCopyOrParsing() {
        assertInvalidWorkload(new byte[0], TuningInspection.InvalidReason.SIZE_TOO_SMALL);
        assertInvalidPlan(new byte[47], TuningInspection.InvalidReason.SIZE_TOO_SMALL);
        assertInvalidWorkload(new byte[WorkloadCacheFile.MAX_FILE_BYTES + 1],
                TuningInspection.InvalidReason.OVERSIZE);
        assertInvalidPlan(new byte[ModelPlanCacheFile.MAX_FILE_BYTES + 1],
                TuningInspection.InvalidReason.OVERSIZE);
    }

    @Test
    void deterministicPathSnapshotSeamClassifiesEarlyEndAndOneByteGrowth() throws Exception {
        byte[] valid = workloadBytes();
        TuningInspection.ReadChannel earlyEnd = new TuningInspection.ReadChannel() {
            private int calls;

            @Override
            public long size() {
                return valid.length;
            }

            @Override
            public int read(ByteBuffer target) {
                if (calls++ != 0) return -1;
                int count = valid.length / 2;
                target.put(valid, 0, count);
                return count;
            }
        };
        var truncated = TuningInspection.inspectWorkloadCache(earlyEnd);
        assertEquals(TuningInspection.ArtifactStatus.INVALID, truncated.status());
        assertEquals(TuningInspection.ArtifactSource.PATH, truncated.source());
        assertEquals(TuningInspection.InvalidReason.TRUNCATED, truncated.invalidReason());
        assertEquals(valid.length, truncated.observedByteCount());

        TuningInspection.ReadChannel growth = new TuningInspection.ReadChannel() {
            private boolean snapshotRead;

            @Override
            public long size() {
                return valid.length;
            }

            @Override
            public int read(ByteBuffer target) {
                if (!snapshotRead) {
                    snapshotRead = true;
                    target.put(valid);
                    return valid.length;
                }
                target.put((byte) 1);
                return 1;
            }
        };
        var grown = TuningInspection.inspectWorkloadCache(growth);
        assertEquals(TuningInspection.ArtifactStatus.INVALID, grown.status());
        assertEquals(TuningInspection.ArtifactSource.PATH, grown.source());
        assertEquals(TuningInspection.InvalidReason.GROWTH_DURING_READ, grown.invalidReason());
        assertEquals(valid.length, grown.observedByteCount());
    }

    @Test
    void workloadMalformedCategoriesAreTypedAndOperationalLoaderStillThrows() throws Exception {
        byte[] valid = workloadBytes();
        byte[] checksum = valid.clone();
        checksum[20] ^= 1;
        assertInvalidWorkload(checksum, TuningInspection.InvalidReason.CHECKSUM_MISMATCH);

        byte[] magic = valid.clone();
        magic[0] ^= 1;
        resign(magic);
        assertInvalidWorkload(magic, TuningInspection.InvalidReason.INVALID_MAGIC);

        byte[] schema = valid.clone();
        putInt(schema, 8, 99);
        resign(schema);
        assertInvalidWorkload(schema, TuningInspection.InvalidReason.UNSUPPORTED_SCHEMA);

        byte[] negativeSchema = valid.clone();
        putInt(negativeSchema, 8, -1);
        resign(negativeSchema);
        var negativeSchemaReport = TuningInspection.inspectWorkloadCache(negativeSchema);
        assertEquals(TuningInspection.ArtifactStatus.INVALID, negativeSchemaReport.status());
        assertEquals(TuningInspection.InvalidReason.UNSUPPORTED_SCHEMA,
                negativeSchemaReport.invalidReason());
        assertEquals(0, negativeSchemaReport.artifactSchema());

        byte[] count = valid.clone();
        putInt(count, 12, WorkloadCacheFile.MAX_ENTRIES + 1);
        resign(count);
        assertInvalidWorkload(count, TuningInspection.InvalidReason.INVALID_ENTRY_COUNT);

        byte[] length = valid.clone();
        putInt(length, 20, 0);
        resign(length);
        assertInvalidWorkload(length, TuningInspection.InvalidReason.INVALID_LENGTH);

        byte[] truncated = valid.clone();
        putInt(truncated, 20, WorkloadCacheFile.MAX_OPAQUE_BYTES);
        resign(truncated);
        assertInvalidWorkload(truncated, TuningInspection.InvalidReason.TRUNCATED);

        int compatibilityLength = intAt(valid, 20);
        int objectiveOffset = 24 + compatibilityLength;
        byte[] structure = valid.clone();
        putInt(structure, objectiveOffset, 0);
        resign(structure);
        assertInvalidWorkload(structure,
                TuningInspection.InvalidReason.INVALID_STRUCTURE_OR_SUMMARY);

        byte[] trailing = insertPayloadByte(valid);
        assertInvalidWorkload(trailing, TuningInspection.InvalidReason.TRAILING_PAYLOAD);

        byte[] duplicate = duplicateOnlyEntry(valid);
        assertInvalidWorkload(duplicate,
                TuningInspection.InvalidReason.DUPLICATE_OR_NONCANONICAL_ORDER);

        Path damaged = temporary.resolve("damaged-workload");
        Files.write(damaged, checksum);
        assertThrows(IOException.class, () -> new WorkloadCacheFile().load(damaged));
    }

    @Test
    void modelPlanMalformedCategoriesAreTypedAndOperationalLoaderStillThrows() throws Exception {
        byte[] valid = modelPlanBytes();
        byte[] checksum = valid.clone();
        checksum[24] ^= 1;
        assertInvalidPlan(checksum, TuningInspection.InvalidReason.CHECKSUM_MISMATCH);

        byte[] magic = valid.clone();
        magic[0] ^= 1;
        resign(magic);
        assertInvalidPlan(magic, TuningInspection.InvalidReason.INVALID_MAGIC);

        byte[] schema = valid.clone();
        putInt(schema, 8, 2);
        resign(schema);
        assertInvalidPlan(schema, TuningInspection.InvalidReason.UNSUPPORTED_SCHEMA);

        byte[] negativeSchema = valid.clone();
        putInt(negativeSchema, 8, -1);
        resign(negativeSchema);
        var negativeSchemaReport = TuningInspection.inspectModelPlanCache(negativeSchema);
        assertEquals(TuningInspection.ArtifactStatus.INVALID, negativeSchemaReport.status());
        assertEquals(TuningInspection.InvalidReason.UNSUPPORTED_SCHEMA,
                negativeSchemaReport.invalidReason());
        assertEquals(0, negativeSchemaReport.artifactSchema());

        byte[] count = valid.clone();
        putInt(count, 12, -1);
        resign(count);
        assertInvalidPlan(count, TuningInspection.InvalidReason.INVALID_ENTRY_COUNT);

        byte[] length = valid.clone();
        putInt(length, 20, 0);
        resign(length);
        assertInvalidPlan(length, TuningInspection.InvalidReason.INVALID_LENGTH);

        byte[] truncated = valid.clone();
        putInt(truncated, 20, ModelPlanCacheFile.MAX_OPAQUE_BYTES);
        resign(truncated);
        assertInvalidPlan(truncated, TuningInspection.InvalidReason.TRUNCATED);

        byte[] structure = valid.clone();
        putInt(structure, modelObjectiveOffset(structure), 0);
        resign(structure);
        assertInvalidPlan(structure,
                TuningInspection.InvalidReason.INVALID_STRUCTURE_OR_SUMMARY);

        byte[] trailing = insertPayloadByte(valid);
        assertInvalidPlan(trailing, TuningInspection.InvalidReason.TRAILING_PAYLOAD);

        byte[] duplicate = duplicateOnlyEntry(valid);
        assertInvalidPlan(duplicate,
                TuningInspection.InvalidReason.DUPLICATE_OR_NONCANONICAL_ORDER);

        Path damaged = temporary.resolve("damaged-plan");
        Files.write(damaged, checksum);
        assertThrows(IOException.class, () -> new ModelPlanCacheFile().load(damaged));
    }

    @Test
    void pathInspectionDoesNotChangeBytesOrTimestamp() throws Exception {
        byte[] workload = workloadBytes();
        Path path = temporary.resolve("read-only-cache");
        Files.write(path, workload);
        FileTime timestamp = FileTime.fromMillis(1_000_000);
        Files.setLastModifiedTime(path, timestamp);
        TuningInspection.inspectWorkloadCache(path);
        assertArrayEquals(workload, Files.readAllBytes(path));
        assertEquals(timestamp, Files.getLastModifiedTime(path));
    }

    @Test
    void readOnlyPosixDirectoryAndAbsentTargetRemainUnchanged() throws Exception {
        Path directory = temporary.resolve("read-only");
        Files.createDirectory(directory);
        Assumptions.assumeTrue(Files.getFileStore(directory)
                .supportsFileAttributeView(PosixFileAttributeView.class));
        Path cache = directory.resolve("workload.cache");
        Path absent = directory.resolve("absent.cache");
        byte[] original = workloadBytes();
        Files.write(cache, original);
        FileTime cacheTime = FileTime.fromMillis(2_000_000);
        FileTime directoryTime = FileTime.fromMillis(3_000_000);
        Files.setLastModifiedTime(cache, cacheTime);
        Files.setLastModifiedTime(directory, directoryTime);
        var originalPermissions = Files.getPosixFilePermissions(directory);
        List<String> originalEntries;
        try (var stream = Files.list(directory)) {
            originalEntries = stream.map(path -> path.getFileName().toString()).sorted().toList();
        }
        try {
            Files.setPosixFilePermissions(directory, EnumSet.of(
                    PosixFilePermission.OWNER_READ, PosixFilePermission.OWNER_EXECUTE));
            assertEquals(TuningInspection.ArtifactStatus.VALID,
                    TuningInspection.inspectWorkloadCache(cache).status());
            assertEquals(TuningInspection.ArtifactStatus.MISSING,
                    TuningInspection.inspectWorkloadCache(absent).status());
            assertArrayEquals(original, Files.readAllBytes(cache));
            assertEquals(cacheTime, Files.getLastModifiedTime(cache));
            assertEquals(directoryTime, Files.getLastModifiedTime(directory));
            try (var stream = Files.list(directory)) {
                assertEquals(originalEntries,
                        stream.map(path -> path.getFileName().toString()).sorted().toList());
            }
            assertFalse(Files.exists(absent));
        } finally {
            Files.setPosixFilePermissions(directory, originalPermissions);
        }
    }

    private byte[] workloadBytes() throws Exception {
        WorkloadCacheFile.Key key = new WorkloadCacheFile.Key(1, bytes("compat"), 1, 0, 3);
        return new WorkloadCacheFile().encode(Map.of(key,
                new WorkloadCacheFile.Entry(key, bytes("decision"), bytes("winner"),
                        new WorkloadTuningResult.SampleSummary(1, 2, 3, 3))));
    }

    private byte[] modelPlanBytes() throws Exception {
        ModelPlanCacheFile.Key key = new ModelPlanCacheFile.Key(
                1, bytes("producer"), 1, bytes("codec"), 1, bytes("model"),
                1, bytes("profile"), 1, bytes("target"), 1, bytes("compat"),
                1, 1, 1, bytes("policy"), 0, 3);
        Path path = temporary.resolve("encoded-plan");
        new ModelPlanCacheFile().publish(path, Map.of(key,
                new ModelPlanCacheFile.Entry(key, bytes("decision"), bytes("winner"),
                        new CompletePlanTuningResult.SampleSummary(1, 2, 3, 3))));
        return Files.readAllBytes(path);
    }

    private static byte[] insertPayloadByte(byte[] original) throws Exception {
        int payloadLength = original.length - 32;
        byte[] expanded = new byte[original.length + 1];
        System.arraycopy(original, 0, expanded, 0, payloadLength);
        expanded[payloadLength] = 42;
        resign(expanded);
        return expanded;
    }

    private static byte[] duplicateOnlyEntry(byte[] original) throws Exception {
        int payloadLength = original.length - 32;
        int entryLength = payloadLength - 16;
        byte[] duplicate = new byte[16 + entryLength * 2 + 32];
        System.arraycopy(original, 0, duplicate, 0, 16);
        putInt(duplicate, 12, 2);
        System.arraycopy(original, 16, duplicate, 16, entryLength);
        System.arraycopy(original, 16, duplicate, 16 + entryLength, entryLength);
        resign(duplicate);
        return duplicate;
    }

    private static void resign(byte[] bytes) throws Exception {
        int payloadLength = bytes.length - 32;
        byte[] digest = MessageDigest.getInstance("SHA-256").digest(
                java.util.Arrays.copyOf(bytes, payloadLength));
        System.arraycopy(digest, 0, bytes, payloadLength, digest.length);
    }

    private static void putInt(byte[] bytes, int offset, int value) {
        ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN).putInt(offset, value);
    }

    private static int intAt(byte[] bytes, int offset) {
        return ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN).getInt(offset);
    }

    private static int modelObjectiveOffset(byte[] bytes) {
        ByteBuffer input = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN);
        input.position(16);
        for (int field = 0; field < 6; field++) {
            input.getInt();
            int length = input.getInt();
            input.position(input.position() + length);
        }
        return input.position();
    }

    private static void assertInvalidWorkload(byte[] bytes, TuningInspection.InvalidReason reason) {
        var report = TuningInspection.inspectWorkloadCache(bytes);
        assertEquals(TuningInspection.ArtifactStatus.INVALID, report.status());
        assertEquals(reason, report.invalidReason());
        assertEquals(0, report.entries().size());
    }

    private static void assertInvalidPlan(byte[] bytes, TuningInspection.InvalidReason reason) {
        var report = TuningInspection.inspectModelPlanCache(bytes);
        assertEquals(TuningInspection.ArtifactStatus.INVALID, report.status());
        assertEquals(reason, report.invalidReason());
        assertEquals(0, report.entries().size());
    }

    private static byte[] bytes(String value) {
        return value.getBytes(java.nio.charset.StandardCharsets.UTF_8);
    }
}

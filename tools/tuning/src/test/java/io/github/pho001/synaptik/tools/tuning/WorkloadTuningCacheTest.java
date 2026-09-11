package io.github.pho001.synaptik.tools.tuning;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class WorkloadTuningCacheTest {
    @TempDir Path directory;

    @Test
    void persistentMissPublishesAndDifferentModelReusesWithoutWriteOrMeasurement() throws Exception {
        Path cache = directory.resolve("workloads.bin");
        var batch = new WorkloadTuningTest.Batch("alpha");
        var firstBackend = new WorkloadTuningTest.Backend(Map.of(
                "alpha", List.of(new WorkloadTuningTest.Candidate("winner"))));
        var firstMeasurement = new WorkloadTuningTest.CountingMeasurement();
        WorkloadTuning.tune(
                WorkloadTuningTest.request(
                        cache, batch, WorkloadTuningTest.scopePersistent(), 0, 0, 1),
                firstBackend,
                firstMeasurement,
                WorkloadTuningTest.clock(0, 4));
        byte[] firstBytes = Files.readAllBytes(cache);
        long firstModified = Files.getLastModifiedTime(cache).toMillis();

        var secondBackend = new WorkloadTuningTest.Backend(Map.of(
                "alpha", List.of(new WorkloadTuningTest.Candidate("winner"))));
        var secondMeasurement = new WorkloadTuningTest.CountingMeasurement();
        var base = WorkloadTuningTest.request(
                cache, batch, WorkloadTuningTest.scopePersistent(), 0, 0, 1);
        var differentModel = new WorkloadTuningRequest<>(
                new WorkloadTuningRequest.ModelFingerprint(9, WorkloadTuningTest.bytes("other-model")),
                base.profileFingerprint(),
                base.occurrences(),
                base.objective(),
                base.budget(),
                cache);
        WorkloadTuningResult<WorkloadTuningTest.Batch, WorkloadTuningTest.Decision> reused =
                WorkloadTuning.tune(
                        differentModel,
                        secondBackend,
                        secondMeasurement,
                        () -> { throw new AssertionError("clock must not be read on a hit"); });

        assertAll(
                () -> assertEquals(1, firstMeasurement.executions),
                () -> assertEquals(1, firstBackend.encodeCalls),
                () -> assertEquals(1, secondBackend.decodeCalls),
                () -> assertEquals(0, secondBackend.candidatesCalls),
                () -> assertEquals(0, secondBackend.selectedDecisionCalls),
                () -> assertEquals(0, secondMeasurement.executions),
                () -> assertEquals(WorkloadTuningResult.Source.CACHE_HIT,
                        reused.evidence().workloads().getFirst().source()),
                () -> assertTrue(reused.evidence().workloads().getFirst().candidates().isEmpty()),
                () -> assertArrayEquals(firstBytes, Files.readAllBytes(cache)),
                () -> assertEquals(firstModified, Files.getLastModifiedTime(cache).toMillis()));
    }

    @Test
    void backendRejectedEntryIsMeasuredAsAMissAndUnrelatedEntryIsRetained() throws Exception {
        Path cache = directory.resolve("merge.bin");
        WorkloadCacheFile file = new WorkloadCacheFile();
        WorkloadCacheFile.Entry unrelated = entry("unrelated", "other", 9);
        file.publish(cache, Map.of(unrelated.key(), unrelated));
        var batch = new WorkloadTuningTest.Batch("alpha");
        var backend = new WorkloadTuningTest.Backend(Map.of(
                "alpha", List.of(new WorkloadTuningTest.Candidate("fresh")))) {
            private boolean rejectedCacheEntry;

            @Override
            public java.util.Optional<WorkloadTuningTest.Decision> decodeCompatibleDecision(
                    WorkloadTuningTest.Batch batch, byte[] encodedDecision) {
                decodeCalls++;
                if (!rejectedCacheEntry) {
                    rejectedCacheEntry = true;
                    return java.util.Optional.empty();
                }
                return java.util.Optional.of(new WorkloadTuningTest.Decision("fresh"));
            }
        };
        var request = WorkloadTuningTest.request(
                cache, batch, new WorkloadTuningRequest.WorkloadCompatibility(
                        1, WorkloadTuningTest.bytes("unrelated"),
                        WorkloadTuningRequest.ReuseScope.PERSISTENT), 0, 0, 1);
        var measurement = new WorkloadTuningTest.CountingMeasurement();

        WorkloadTuningResult<WorkloadTuningTest.Batch, WorkloadTuningTest.Decision> result =
                WorkloadTuning.tune(
                        request, backend, measurement, WorkloadTuningTest.clock(0, 2));

        assertAll(
                () -> assertEquals(1, measurement.executions),
                () -> assertEquals(2, backend.decodeCalls),
                () -> assertEquals(WorkloadTuningResult.Source.MEASURED,
                        result.evidence().workloads().getFirst().source()),
                () -> assertEquals(1, file.load(cache).size()),
                () -> assertArrayEquals(
                        WorkloadTuningTest.bytes("fresh"),
                        file.load(cache).get(unrelated.key()).encodedDecision()));
    }

    @Test
    void newPersistentMissMergesWithAWellFormedUnrelatedEntry() throws Exception {
        Path cache = directory.resolve("retain.bin");
        WorkloadCacheFile file = new WorkloadCacheFile();
        WorkloadCacheFile.Entry unrelated = entry("a-unrelated", "other", 9);
        file.publish(cache, Map.of(unrelated.key(), unrelated));
        var batch = new WorkloadTuningTest.Batch("alpha");
        WorkloadTuning.tune(
                WorkloadTuningTest.request(
                        cache,
                        batch,
                        new WorkloadTuningRequest.WorkloadCompatibility(
                                1, WorkloadTuningTest.bytes("z-new"),
                                WorkloadTuningRequest.ReuseScope.PERSISTENT),
                        0,
                        0,
                        1),
                new WorkloadTuningTest.Backend(Map.of(
                        "alpha", List.of(new WorkloadTuningTest.Candidate("fresh")))),
                new WorkloadTuningTest.CountingMeasurement(),
                WorkloadTuningTest.clock(0, 2));

        Map<WorkloadCacheFile.Key, WorkloadCacheFile.Entry> merged = file.load(cache);
        assertAll(
                () -> assertEquals(2, merged.size()),
                () -> assertArrayEquals(
                        WorkloadTuningTest.bytes("other"),
                        merged.get(unrelated.key()).encodedDecision()));
    }

    @Test
    void encodingIsDeterministicAndLoadRetainsEveryEntry() throws Exception {
        WorkloadCacheFile file = new WorkloadCacheFile();
        WorkloadCacheFile.Entry first = entry("z", "one", 3);
        WorkloadCacheFile.Entry second = entry("a", "two", 5);
        Map<WorkloadCacheFile.Key, WorkloadCacheFile.Entry> forward = new LinkedHashMap<>();
        forward.put(first.key(), first);
        forward.put(second.key(), second);
        Map<WorkloadCacheFile.Key, WorkloadCacheFile.Entry> reverse = new LinkedHashMap<>();
        reverse.put(second.key(), second);
        reverse.put(first.key(), first);

        byte[] expected = file.encode(forward);
        byte[] actual = file.encode(reverse);
        Path target = directory.resolve("stable.bin");
        file.publish(target, reverse);

        assertAll(
                () -> assertArrayEquals(expected, actual),
                () -> assertArrayEquals(expected, Files.readAllBytes(target)),
                () -> assertEquals(2, file.load(target).size()),
                () -> assertArrayEquals(
                        WorkloadTuningTest.bytes("one"),
                        file.load(target).get(first.key()).encodedDecision()));
    }

    @Test
    void checksumTruncationTrailingAndDuplicateKeysAreRejected() throws Exception {
        WorkloadCacheFile file = new WorkloadCacheFile();
        WorkloadCacheFile.Entry entry = entry("key", "decision", 2);
        byte[] valid = file.encode(Map.of(entry.key(), entry));

        Path checksum = directory.resolve("checksum.bin");
        byte[] damaged = valid.clone();
        damaged[20] ^= 1;
        Files.write(checksum, damaged);
        Path truncated = directory.resolve("truncated.bin");
        Files.write(truncated, java.util.Arrays.copyOf(valid, valid.length - 1));
        Path trailing = directory.resolve("trailing.bin");
        byte[] withTrailing = java.util.Arrays.copyOf(valid, valid.length + 1);
        Files.write(trailing, withTrailing);

        assertAll(
                () -> assertThrows(IOException.class, () -> file.load(checksum)),
                () -> assertThrows(IOException.class, () -> file.load(truncated)),
                () -> assertThrows(IOException.class, () -> file.load(trailing)),
                () -> assertThrows(IOException.class,
                        () -> file.encode(new DuplicateKeyMap(entry))));
    }

    @Test
    void wellChecksummedInvalidSchemaLengthsDuplicatesSummariesAndTrailingBytesAreRejected()
            throws Exception {
        WorkloadCacheFile file = new WorkloadCacheFile();
        WorkloadCacheFile.Entry first = entry("a", "one", 3);
        WorkloadCacheFile.Entry second = entry("b", "two", 5);
        byte[] twoEntries = file.encode(Map.of(first.key(), first, second.key(), second));

        byte[] schema = twoEntries.clone();
        ByteBuffer.wrap(schema).order(ByteOrder.BIG_ENDIAN).putInt(8, 99);
        refreshChecksum(schema);

        byte[] oversized = twoEntries.clone();
        ByteBuffer.wrap(oversized).order(ByteOrder.BIG_ENDIAN)
                .putInt(20, WorkloadCacheFile.MAX_OPAQUE_BYTES + 1);
        refreshChecksum(oversized);

        int firstEntryBytes = encodedEntryBytes(first);
        int secondStart = 16 + firstEntryBytes;
        byte[] duplicate = twoEntries.clone();
        duplicate[secondStart + 8] = duplicate[16 + 8];
        refreshChecksum(duplicate);

        byte[] invalidSummary = file.encode(Map.of(first.key(), first));
        int minimumOffset = 16 + 4 + 4 + first.key().compatibility().length
                + 4 + 4 + 4
                + 4 + first.encodedDecision().length
                + 4 + first.winnerIdentity().length;
        ByteBuffer.wrap(invalidSummary).order(ByteOrder.BIG_ENDIAN).putLong(minimumOffset, 10);
        refreshChecksum(invalidSummary);

        byte[] trailing = new byte[twoEntries.length + 1];
        System.arraycopy(twoEntries, 0, trailing, 0, twoEntries.length - 32);
        trailing[twoEntries.length - 32] = 42;
        refreshChecksum(trailing);

        Path schemaPath = write("schema.bin", schema);
        Path oversizedPath = write("oversized.bin", oversized);
        Path duplicatePath = write("duplicate.bin", duplicate);
        Path summaryPath = write("summary.bin", invalidSummary);
        Path trailingPath = write("well-checksummed-trailing.bin", trailing);
        assertAll(
                () -> assertThrows(IOException.class, () -> file.load(schemaPath)),
                () -> assertThrows(IOException.class, () -> file.load(oversizedPath)),
                () -> assertThrows(IOException.class, () -> file.load(duplicatePath)),
                () -> assertThrows(IOException.class, () -> file.load(summaryPath)),
                () -> assertThrows(IOException.class, () -> file.load(trailingPath)));
    }

    @Test
    void oversizedFileIsRejectedBeforeContentAllocation() throws Exception {
        Path oversized = directory.resolve("too-large.bin");
        try (var channel = java.nio.channels.FileChannel.open(
                oversized,
                java.nio.file.StandardOpenOption.CREATE_NEW,
                java.nio.file.StandardOpenOption.WRITE)) {
            channel.position(WorkloadCacheFile.MAX_FILE_BYTES);
            channel.write(ByteBuffer.wrap(new byte[] {1}));
        }
        assertThrows(IOException.class, () -> new WorkloadCacheFile().load(oversized));
    }

    @Test
    void publicationFailuresPreservePriorBytesAndCleanTemporaryFiles() throws Exception {
        Path target = directory.resolve("atomic.bin");
        byte[] prior = WorkloadTuningTest.bytes("prior");
        Files.write(target, prior);
        WorkloadCacheFile.Entry entry = entry("key", "decision", 2);
        Map<WorkloadCacheFile.Key, WorkloadCacheFile.Entry> entries = Map.of(entry.key(), entry);
        WorkloadCacheFile beforeMoveFailure = new WorkloadCacheFile(
                (temporary, ignored) -> { throw new IOException("before move"); },
                (temporary, ignored) -> { throw new AssertionError("move must not run"); });

        assertThrows(IOException.class, () -> beforeMoveFailure.publish(target, entries));
        assertArrayEquals(prior, Files.readAllBytes(target));
        assertNoTemporaryFiles();

        WorkloadCacheFile moveFailure = new WorkloadCacheFile(
                (temporary, ignored) -> { },
                (temporary, ignored) -> { throw new IOException("move"); });
        assertThrows(IOException.class, () -> moveFailure.publish(target, entries));
        assertAll(
                () -> assertArrayEquals(prior, Files.readAllBytes(target)),
                this::assertNoTemporaryFiles);
    }

    private void assertNoTemporaryFiles() throws IOException {
        try (var files = Files.list(directory)) {
            assertFalse(files.anyMatch(path -> path.getFileName().toString().endsWith(".tmp")));
        }
    }

    private static WorkloadCacheFile.Entry entry(String key, String decision, long median) {
        WorkloadCacheFile.Key cacheKey = new WorkloadCacheFile.Key(
                1, WorkloadTuningTest.bytes(key), 1, 0, 1);
        return new WorkloadCacheFile.Entry(
                cacheKey,
                WorkloadTuningTest.bytes(decision),
                WorkloadTuningTest.bytes("winner-" + key),
                new WorkloadTuningResult.SampleSummary(median, median, median, 1));
    }

    private Path write(String name, byte[] bytes) throws IOException {
        Path path = directory.resolve(name);
        Files.write(path, bytes);
        return path;
    }

    private static int encodedEntryBytes(WorkloadCacheFile.Entry entry) {
        return 8 * Integer.BYTES + 3 * Long.BYTES
                + entry.key().compatibility().length
                + entry.encodedDecision().length
                + entry.winnerIdentity().length;
    }

    private static void refreshChecksum(byte[] bytes) throws Exception {
        int payloadLength = bytes.length - 32;
        byte[] checksum = MessageDigest.getInstance("SHA-256")
                .digest(java.util.Arrays.copyOf(bytes, payloadLength));
        System.arraycopy(checksum, 0, bytes, payloadLength, checksum.length);
    }

    private static final class DuplicateKeyMap
            extends java.util.AbstractMap<WorkloadCacheFile.Key, WorkloadCacheFile.Entry> {
        private final WorkloadCacheFile.Entry entry;

        DuplicateKeyMap(WorkloadCacheFile.Entry entry) {
            this.entry = entry;
        }

        @Override
        public int size() { return 2; }

        @Override
        public java.util.Set<Entry<WorkloadCacheFile.Key, WorkloadCacheFile.Entry>> entrySet() {
            return new java.util.AbstractSet<>() {
                @Override
                public java.util.Iterator<Entry<WorkloadCacheFile.Key, WorkloadCacheFile.Entry>>
                        iterator() {
                    return List.of(Map.entry(entry.key(), entry), Map.entry(entry.key(), entry))
                            .iterator();
                }

                @Override
                public int size() { return 2; }
            };
        }
    }
}

package io.github.pho001.synaptik.tools.tuning;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class CompletePlanTuningCacheTest {
    private static final int CHECKSUM_BYTES = 32;
    private static final int HEADER_BYTES = 16;

    @TempDir Path directory;

    @Test
    void persistentMissPublishesAndExactHitPerformsZeroCandidateOrExecutionActions()
            throws Exception {
        Path cache = directory.resolve("plans.bin");
        CompletePlanTuningTest.Backend backend = persistentBackend();
        AtomicInteger measurements = new AtomicInteger();
        CompletePlanTuningTest.Correctness firstCorrectness =
                new CompletePlanTuningTest.Correctness(new ArrayList<>());

        CompletePlanTuningResult<CompletePlanTuningTest.Batch, CompletePlanTuningTest.Decision>
                first =
                        CompletePlanTuning.tune(
                                CompletePlanTuningTest.request(cache, 2, 0, 1, 4),
                                backend,
                                firstCorrectness,
                                (batch, candidate) -> measurements.incrementAndGet(),
                                CompletePlanTuningTest.clock(0, 4, 10, 12));
        byte[] stable = Files.readAllBytes(cache);
        int enumerations = backend.candidatesCalls;
        int selected = backend.selectedCalls;
        int encoded = backend.encodeCalls;
        int measured = measurements.get();

        CompletePlanTuningResult<CompletePlanTuningTest.Batch, CompletePlanTuningTest.Decision>
                second =
                        CompletePlanTuning.tune(
                                CompletePlanTuningTest.request(cache, 2, 0, 1, 4),
                                backend,
                                new CompletePlanCorrectness<
                                        CompletePlanTuningTest.Batch,
                                        CompletePlanTuningTest.Candidate,
                                        String>() {
                                    @Override
                                    public String capture(
                                            CompletePlanTuningTest.Batch batch,
                                            CompletePlanTuningTest.Candidate candidate,
                                            long maximumBytes) {
                                        return fail("cache hit must not capture correctness");
                                    }

                                    @Override
                                    public Outcome compare(
                                            String reference,
                                            CompletePlanTuningTest.Batch batch,
                                            CompletePlanTuningTest.Candidate candidate,
                                            long maximumBytes) {
                                        return fail("cache hit must not compare correctness");
                                    }
                                },
                                (batch, candidate) -> fail("cache hit must not execute"));

        assertAll(
                () -> assertEquals(CompletePlanTuningResult.Source.MEASURED, first.evidence().source()),
                () -> assertEquals(CompletePlanTuningResult.Source.CACHE_HIT, second.evidence().source()),
                () -> assertTrue(second.evidence().candidates().isEmpty()),
                () -> assertEquals(enumerations, backend.candidatesCalls),
                () -> assertEquals(selected, backend.selectedCalls),
                () -> assertEquals(encoded, backend.encodeCalls),
                () -> assertEquals(measured, measurements.get()),
                () -> assertEquals(2, backend.decodeCalls),
                () -> assertArrayEquals(stable, Files.readAllBytes(cache)),
                () ->
                        assertEquals(
                                new CompletePlanTuningTest.Decision("b"),
                                second.selectedHandoff().selectedDecision().orElseThrow()));
    }

    @Test
    void decoderRejectedHitIsAMissAndRunsFreshCorrectnessAndMeasurement() throws Exception {
        Path cache = directory.resolve("plans.bin");
        CompletePlanTuning.tune(
                CompletePlanTuningTest.request(cache, 2, 0, 1, 4),
                persistentBackend(),
                new CompletePlanTuningTest.Correctness(new ArrayList<>()),
                (batch, candidate) -> {},
                CompletePlanTuningTest.clock(0, 1, 2, 3));

        AtomicInteger decodeSequence = new AtomicInteger();
        CompletePlanTuningTest.Backend rejecting =
                new CompletePlanTuningTest.Backend(
                        CompletePlanTuningTest.scope(
                                CompletePlanTuningRequest.ReuseScope.PERSISTENT)) {
                    @Override
                    public Optional<CompletePlanTuningTest.Decision> decodeCompatibleDecision(
                            CompletePlanTuningTest.Batch batch, byte[] encoded) {
                        decodeCalls++;
                        if (decodeSequence.getAndIncrement() == 0) {
                            return Optional.empty();
                        }
                        String id = new String(encoded, java.nio.charset.StandardCharsets.UTF_8);
                        return Optional.of(new CompletePlanTuningTest.Decision(id));
                    }
                };
        List<String> events = new ArrayList<>();

        CompletePlanTuningResult<CompletePlanTuningTest.Batch, CompletePlanTuningTest.Decision>
                result =
                        CompletePlanTuning.tune(
                                CompletePlanTuningTest.request(cache, 2, 0, 1, 4),
                                rejecting,
                                new CompletePlanTuningTest.Correctness(events),
                                (batch, candidate) -> events.add("measure:" + candidate.id()),
                                CompletePlanTuningTest.clock(0, 2, 3, 4));

        assertAll(
                () -> assertEquals(CompletePlanTuningResult.Source.MEASURED, result.evidence().source()),
                () -> assertEquals(1, rejecting.candidatesCalls),
                () -> assertEquals(1, rejecting.selectedCalls),
                () -> assertEquals(1, rejecting.encodeCalls),
                () -> assertEquals(2, rejecting.decodeCalls),
                () ->
                        assertEquals(
                                List.of("capture:a", "compare:b", "measure:a", "measure:b"),
                                events));
    }

    @Test
    void invalidDecisionEncodingsAndCodecRoundTripsNeverPublish() {
        List<CompletePlanTuningTest.Backend> backends =
                List.of(
                        encodingBackend(new byte[0], Optional.of(new CompletePlanTuningTest.Decision("a"))),
                        encodingBackend(
                                new byte[ModelPlanCacheFile.MAX_OPAQUE_BYTES + 1],
                                Optional.of(new CompletePlanTuningTest.Decision("a"))),
                        encodingBackend(
                                CompletePlanTuningTest.bytes("a"),
                                Optional.empty()),
                        encodingBackend(
                                CompletePlanTuningTest.bytes("a"),
                                Optional.of(new CompletePlanTuningTest.Decision("different"))));

        for (int index = 0; index < backends.size(); index++) {
            Path cache = directory.resolve("invalid-" + index + ".bin");
            CompletePlanTuningTest.Backend backend = backends.get(index);

            assertThrows(
                    RuntimeException.class,
                    () ->
                            CompletePlanTuning.tune(
                                    CompletePlanTuningTest.request(cache, 2, 0, 1, 4),
                                    backend,
                                    new CompletePlanTuningTest.Correctness(new ArrayList<>()),
                                    (batch, candidate) -> {},
                                    CompletePlanTuningTest.clock(0, 1, 2, 3)));
            assertAll(
                    () -> assertFalse(Files.exists(cache)),
                    () -> assertEquals(1, backend.selectedCalls),
                    () -> assertEquals(1, backend.encodeCalls));
        }
    }

    @Test
    void malformedCachesFailBeforeEnumerationOrExecutionAndRemainUnchanged() throws Exception {
        byte[] oneEntry = validCacheBytes(Map.of(key("a"), entry(key("a"), "a")));
        byte[] twoEntries =
                validCacheBytes(
                        orderedEntries(
                                entry(key("a"), "a"),
                                entry(key("b"), "b")));
        List<byte[]> corruptions = new ArrayList<>();

        corruptions.add(withIntAndChecksum(oneEntry, 8, 2));
        corruptions.add(withIntAndChecksum(oneEntry, 20, ModelPlanCacheFile.MAX_OPAQUE_BYTES + 1));
        corruptions.add(truncatedWithValidChecksum(oneEntry));
        byte[] checksumFailure = oneEntry.clone();
        checksumFailure[HEADER_BYTES] ^= 1;
        corruptions.add(checksumFailure);
        corruptions.add(duplicateSecondKey(twoEntries));
        corruptions.add(nonCanonicalEntries(twoEntries));
        corruptions.add(invalidSummary(oneEntry));
        corruptions.add(withTrailingPayloadByte(oneEntry));

        for (int index = 0; index < corruptions.size(); index++) {
            Path cache = directory.resolve("corrupt-" + index + ".bin");
            Files.write(cache, corruptions.get(index));
            assertRejectedBeforeExecution(cache);
            assertArrayEquals(corruptions.get(index), Files.readAllBytes(cache));
        }
    }

    @Test
    void oversizedCacheFailsBeforeEnumerationOrExecution() throws Exception {
        Path cache = directory.resolve("oversized.bin");
        try (FileChannel channel =
                FileChannel.open(
                        cache,
                        StandardOpenOption.CREATE_NEW,
                        StandardOpenOption.WRITE)) {
            channel.position(ModelPlanCacheFile.MAX_FILE_BYTES);
            channel.write(ByteBuffer.wrap(new byte[] {1}));
        }

        assertRejectedBeforeExecution(cache);
        assertEquals(ModelPlanCacheFile.MAX_FILE_BYTES + 1L, Files.size(cache));
    }

    @Test
    void canonicalEncodingIsDeterministicAcrossMapInsertionOrder() throws Exception {
        ModelPlanCacheFile.Key a = key("a");
        ModelPlanCacheFile.Key b = key("b");
        ModelPlanCacheFile.Entry entryA = entry(a, "a");
        ModelPlanCacheFile.Entry entryB = entry(b, "b");
        LinkedHashMap<ModelPlanCacheFile.Key, ModelPlanCacheFile.Entry> forward =
                orderedEntries(entryA, entryB);
        LinkedHashMap<ModelPlanCacheFile.Key, ModelPlanCacheFile.Entry> reverse =
                orderedEntries(entryB, entryA);
        Path first = directory.resolve("first.bin");
        Path second = directory.resolve("second.bin");

        new ModelPlanCacheFile().publish(first, forward);
        new ModelPlanCacheFile().publish(second, reverse);
        Map<ModelPlanCacheFile.Key, ModelPlanCacheFile.Entry> loadedFirst =
                new ModelPlanCacheFile().load(first);
        Map<ModelPlanCacheFile.Key, ModelPlanCacheFile.Entry> loadedSecond =
                new ModelPlanCacheFile().load(second);

        assertAll(
                () -> assertArrayEquals(Files.readAllBytes(first), Files.readAllBytes(second)),
                () -> assertEquals(forward.keySet(), loadedFirst.keySet()),
                () -> assertEquals(forward.keySet(), loadedSecond.keySet()),
                () -> assertArrayEquals(entryA.decision(), loadedFirst.get(a).decision()),
                () -> assertArrayEquals(entryB.winner(), loadedSecond.get(b).winner()),
                () -> assertEquals(entryA.summary(), loadedFirst.get(a).summary()),
                () -> assertEquals(entryB.summary(), loadedSecond.get(b).summary()));
    }

    @Test
    void beforeMoveAndMoveFailuresPreservePriorBytesAndCleanTemporaryFiles()
            throws Exception {
        Path cache = directory.resolve("atomic.bin");
        ModelPlanCacheFile.Key firstKey = key("a");
        ModelPlanCacheFile.Entry firstEntry = entry(firstKey, "a");
        new ModelPlanCacheFile().publish(cache, Map.of(firstKey, firstEntry));
        byte[] prior = Files.readAllBytes(cache);
        ModelPlanCacheFile.Key replacementKey = key("b");
        Map<ModelPlanCacheFile.Key, ModelPlanCacheFile.Entry> replacement =
                Map.of(replacementKey, entry(replacementKey, "b"));

        ModelPlanCacheFile beforeMoveFailure =
                new ModelPlanCacheFile(
                        (temporary, target) -> {
                            throw new IOException("before move");
                        },
                        (temporary, target) -> fail("move must not run"));
        assertThrows(IOException.class, () -> beforeMoveFailure.publish(cache, replacement));
        assertAtomicFailureState(cache, prior);

        ModelPlanCacheFile moveFailure =
                new ModelPlanCacheFile(
                        (temporary, target) -> {},
                        (temporary, target) -> {
                            throw new IOException("move failed");
                        });
        assertThrows(IOException.class, () -> moveFailure.publish(cache, replacement));
        assertAtomicFailureState(cache, prior);
    }

    private static CompletePlanTuningTest.Backend persistentBackend() {
        return new CompletePlanTuningTest.Backend(
                CompletePlanTuningTest.scope(
                        CompletePlanTuningRequest.ReuseScope.PERSISTENT));
    }

    private static CompletePlanTuningTest.Backend encodingBackend(
            byte[] encoded, Optional<CompletePlanTuningTest.Decision> decoded) {
        return new CompletePlanTuningTest.Backend(
                CompletePlanTuningTest.scope(
                        CompletePlanTuningRequest.ReuseScope.PERSISTENT)) {
            @Override
            public byte[] encodeDecision(CompletePlanTuningTest.Decision decision) {
                encodeCalls++;
                return encoded;
            }

            @Override
            public Optional<CompletePlanTuningTest.Decision> decodeCompatibleDecision(
                    CompletePlanTuningTest.Batch batch, byte[] ignored) {
                decodeCalls++;
                return decoded;
            }
        };
    }

    private void assertRejectedBeforeExecution(Path cache) {
        CompletePlanTuningTest.Backend backend = persistentBackend();
        AtomicInteger actions = new AtomicInteger();
        assertThrows(
                IOException.class,
                () ->
                        CompletePlanTuning.tune(
                                CompletePlanTuningTest.request(cache, 2, 0, 1, 4),
                                backend,
                                new CompletePlanCorrectness<
                                        CompletePlanTuningTest.Batch,
                                        CompletePlanTuningTest.Candidate,
                                        String>() {
                                    @Override
                                    public String capture(
                                            CompletePlanTuningTest.Batch batch,
                                            CompletePlanTuningTest.Candidate candidate,
                                            long maximumBytes) {
                                        actions.incrementAndGet();
                                        return "reference";
                                    }

                                    @Override
                                    public Outcome compare(
                                            String reference,
                                            CompletePlanTuningTest.Batch batch,
                                            CompletePlanTuningTest.Candidate candidate,
                                            long maximumBytes) {
                                        actions.incrementAndGet();
                                        return Outcome.MATCH;
                                    }
                                },
                                (batch, candidate) -> actions.incrementAndGet()));
        assertAll(
                () -> assertEquals(0, backend.candidatesCalls),
                () -> assertEquals(0, backend.decodeCalls),
                () -> assertEquals(0, actions.get()));
    }

    private void assertAtomicFailureState(Path cache, byte[] prior) throws IOException {
        assertArrayEquals(prior, Files.readAllBytes(cache));
        try (var files = Files.list(directory)) {
            assertEquals(
                    List.of(cache.getFileName()),
                    files.map(Path::getFileName).sorted().toList());
        }
    }

    private byte[] validCacheBytes(
            Map<ModelPlanCacheFile.Key, ModelPlanCacheFile.Entry> entries) throws IOException {
        Path cache = directory.resolve("valid-" + System.nanoTime() + ".bin");
        new ModelPlanCacheFile().publish(cache, entries);
        byte[] bytes = Files.readAllBytes(cache);
        Files.delete(cache);
        return bytes;
    }

    private static LinkedHashMap<ModelPlanCacheFile.Key, ModelPlanCacheFile.Entry>
            orderedEntries(ModelPlanCacheFile.Entry... entries) {
        LinkedHashMap<ModelPlanCacheFile.Key, ModelPlanCacheFile.Entry> result =
                new LinkedHashMap<>();
        for (ModelPlanCacheFile.Entry entry : entries) {
            result.put(entry.key(), entry);
        }
        return result;
    }

    private static ModelPlanCacheFile.Key key(String value) {
        byte[] bytes = CompletePlanTuningTest.bytes(value);
        return new ModelPlanCacheFile.Key(
                1,
                bytes,
                1,
                bytes,
                1,
                bytes,
                1,
                bytes,
                1,
                bytes,
                1,
                bytes,
                1,
                1,
                1,
                bytes,
                0,
                3);
    }

    private static ModelPlanCacheFile.Entry entry(
            ModelPlanCacheFile.Key key, String value) {
        return new ModelPlanCacheFile.Entry(
                key,
                CompletePlanTuningTest.bytes("decision-" + value),
                CompletePlanTuningTest.bytes("winner-" + value),
                new CompletePlanTuningResult.SampleSummary(1, 2, 3, 3));
    }

    private static byte[] withIntAndChecksum(byte[] source, int offset, int value)
            throws Exception {
        byte[] changed = source.clone();
        ByteBuffer.wrap(changed).order(ByteOrder.BIG_ENDIAN).putInt(offset, value);
        return replaceChecksum(changed);
    }

    private static byte[] truncatedWithValidChecksum(byte[] source) throws Exception {
        int payloadLength = source.length - CHECKSUM_BYTES;
        byte[] truncated = new byte[source.length - 1];
        System.arraycopy(source, 0, truncated, 0, payloadLength - 1);
        return replaceChecksum(truncated);
    }

    private static byte[] duplicateSecondKey(byte[] source) throws Exception {
        byte[] changed = source.clone();
        List<EntryRange> ranges = entryRanges(changed);
        EntryRange first = ranges.get(0);
        EntryRange second = ranges.get(1);
        int keyLength = first.keyEnd - first.start;
        assertEquals(keyLength, second.keyEnd - second.start);
        System.arraycopy(changed, first.start, changed, second.start, keyLength);
        return replaceChecksum(changed);
    }

    private static byte[] nonCanonicalEntries(byte[] source) throws Exception {
        byte[] changed = source.clone();
        List<EntryRange> ranges = entryRanges(changed);
        EntryRange first = ranges.get(0);
        EntryRange second = ranges.get(1);
        int length = first.end - first.start;
        assertEquals(length, second.end - second.start);
        byte[] firstBytes = Arrays.copyOfRange(changed, first.start, first.end);
        byte[] secondBytes = Arrays.copyOfRange(changed, second.start, second.end);
        System.arraycopy(secondBytes, 0, changed, first.start, length);
        System.arraycopy(firstBytes, 0, changed, second.start, length);
        return replaceChecksum(changed);
    }

    private static byte[] invalidSummary(byte[] source) throws Exception {
        byte[] changed = source.clone();
        EntryRange range = entryRanges(changed).getFirst();
        ByteBuffer.wrap(changed)
                .order(ByteOrder.BIG_ENDIAN)
                .putLong(range.end - 28, 10L);
        return replaceChecksum(changed);
    }

    private static byte[] withTrailingPayloadByte(byte[] source) throws Exception {
        int payloadLength = source.length - CHECKSUM_BYTES;
        byte[] changed = new byte[source.length + 1];
        System.arraycopy(source, 0, changed, 0, payloadLength);
        changed[payloadLength] = 7;
        return replaceChecksum(changed);
    }

    private static byte[] replaceChecksum(byte[] bytes) throws Exception {
        int payloadLength = bytes.length - CHECKSUM_BYTES;
        byte[] checksum =
                MessageDigest.getInstance("SHA-256")
                        .digest(Arrays.copyOf(bytes, payloadLength));
        System.arraycopy(checksum, 0, bytes, payloadLength, CHECKSUM_BYTES);
        return bytes;
    }

    private static List<EntryRange> entryRanges(byte[] bytes) {
        ByteBuffer input =
                ByteBuffer.wrap(bytes, 0, bytes.length - CHECKSUM_BYTES)
                        .order(ByteOrder.BIG_ENDIAN);
        input.position(12);
        int count = input.getInt();
        List<EntryRange> ranges = new ArrayList<>();
        for (int index = 0; index < count; index++) {
            int start = input.position();
            for (int field = 0; field < 6; field++) {
                input.getInt();
                skipBytes(input);
            }
            input.getInt();
            input.getInt();
            input.getInt();
            skipBytes(input);
            input.getInt();
            input.getInt();
            int keyEnd = input.position();
            skipBytes(input);
            skipBytes(input);
            input.position(input.position() + Long.BYTES * 3 + Integer.BYTES);
            ranges.add(new EntryRange(start, keyEnd, input.position()));
        }
        return ranges;
    }

    private static void skipBytes(ByteBuffer input) {
        int length = input.getInt();
        input.position(input.position() + length);
    }

    private record EntryRange(int start, int keyEnd, int end) {}
}

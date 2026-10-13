package io.github.pho001.synaptik.tools.tuning;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ModelPlanCacheFileTest {
    // Independent schema-1 fixture: distinct schemas, unequal opaque lengths, and SHA-256 trailer.
    private static final byte[] SCHEMA_ONE_BYTES = HexFormat.of().parseHex(
            "53594e504c414e310000000100000001"
                    + "000000010000000110"
                    + "00000002000000022021"
                    + "0000000300000003303132"
                    + "000000040000000440414243"
                    + "00000005000000055051525354"
                    + "0000000600000006606162636465"
                    + "0000000800000009000000070000000770717273747576"
                    + "0000000a00000003"
                    + "00000008808182838485868700000009909192939495969798"
                    + "000000000000000b000000000000000c000000000000000d00000003"
                    + "935b44667a19d04d5080ea8c4ceba4780f912330257575becd47e6a39e6c6589");
    private static final CompletePlanTuningResult.SampleSummary SUMMARY =
            new CompletePlanTuningResult.SampleSummary(11, 12, 13, 3);

    @TempDir Path directory;

    @Test
    void acceptedBytesMatchSchemaOneFixtureAndRemainLoadableAndInspectable() throws Exception {
        ModelPlanCacheFile.Entry entry = fixtureEntry(8);
        Path cache = directory.resolve("fixture.bin");
        ModelPlanCacheFile file = new ModelPlanCacheFile();

        file.publish(cache, Map.of(entry.key(), entry));
        byte[] encoded = Files.readAllBytes(cache);

        assertEquals(201, ModelPlanCacheFile.encodedSize(List.of(entry)));
        assertArrayEquals(SCHEMA_ONE_BYTES, encoded);
        ModelPlanCacheFile.Parsed parsed = ModelPlanCacheFile.parse(encoded);
        assertEquals(1, parsed.artifactSchema());
        assertEquals(1, parsed.entries().size());
        assertEntryEquals(entry, parsed.entries().getFirst());
        assertEntryEquals(entry, file.load(cache).get(entry.key()));
        var inspected = TuningInspection.inspectModelPlanCache(encoded);
        assertEquals(TuningInspection.ArtifactStatus.VALID, inspected.status());
        assertEquals(1, inspected.entries().size());
        assertEquals(encoded.length, inspected.observedByteCount());
    }

    @Test
    void emptyEncodingCountsHeaderAndChecksumAndRemainsValid() throws Exception {
        Path cache = directory.resolve("empty.bin");
        ModelPlanCacheFile file = new ModelPlanCacheFile();

        file.publish(cache, Map.of());

        assertEquals(48, ModelPlanCacheFile.encodedSize(List.of()));
        assertEquals(48, Files.size(cache));
        assertTrue(file.load(cache).isEmpty());
        assertEquals(TuningInspection.ArtifactStatus.VALID,
                TuningInspection.inspectModelPlanCache(cache).status());
    }

    @Test
    void sizePreflightIncludesEveryFieldAndAcceptsOnlyTheInclusiveFileBound() throws Exception {
        ModelPlanCacheFile.Entry repeated = fixtureEntry(256);
        // 11 key ints + 9 lengths + 3 summary longs + 1 summary int, then all opaque bytes.
        int entryBytes = 108 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 256 + 9;
        int count = (ModelPlanCacheFile.MAX_FILE_BYTES - 48) / entryBytes;
        int remainder = (ModelPlanCacheFile.MAX_FILE_BYTES - 48) % entryBytes;
        // Repeated immutable entries exercise counting without allocating a complete payload.
        List<ModelPlanCacheFile.Entry> entries =
                new ArrayList<>(Collections.nCopies(count, repeated));

        entries.set(count - 1, fixtureEntry(256 + remainder - 1));
        assertEquals(ModelPlanCacheFile.MAX_FILE_BYTES - 1, ModelPlanCacheFile.encodedSize(entries));
        entries.set(count - 1, fixtureEntry(256 + remainder));
        assertEquals(ModelPlanCacheFile.MAX_FILE_BYTES, ModelPlanCacheFile.encodedSize(entries));
        entries.set(count - 1, fixtureEntry(256 + remainder + 1));
        assertSizeRejected(entries);
    }

    @Test
    void sizePreflightRejectsLogicalTotalsBeyondIntegerRangeWithoutAllocatingThem() {
        ModelPlanCacheFile.Entry entry = fixtureEntry(ModelPlanCacheFile.MAX_OPAQUE_BYTES);
        long decisionBytes = (long) ModelPlanCacheFile.MAX_ENTRIES * ModelPlanCacheFile.MAX_OPAQUE_BYTES;
        assertTrue(decisionBytes > Integer.MAX_VALUE);

        // About 64 GiB of logical decisions, backed by just one bounded entry and a repeated view.
        assertSizeRejected(Collections.nCopies(ModelPlanCacheFile.MAX_ENTRIES, entry));
        IOException countFailure = assertThrows(IOException.class,
                () -> ModelPlanCacheFile.encodedSize(
                        Collections.nCopies(ModelPlanCacheFile.MAX_ENTRIES + 1, entry)));
        assertEquals("too many model-plan cache entries", countFailure.getMessage());
    }

    @Test
    void exactLimitPublishesButOneByteMorePreservesTargetAndCreatesNoArtifacts() throws Exception {
        Map<ModelPlanCacheFile.Key, ModelPlanCacheFile.Entry> entries = new LinkedHashMap<>();
        byte[] maximumDecision = new byte[ModelPlanCacheFile.MAX_OPAQUE_BYTES];
        byte[] opaque = {1};
        int entryOverhead = 108 + 7 + 1;
        int lastDecisionBytes = ModelPlanCacheFile.MAX_OPAQUE_BYTES - 48 - 16 * entryOverhead;
        ModelPlanCacheFile.Key lastKey = null;
        for (int index = 0; index < 16; index++) {
            ModelPlanCacheFile.Key key = new ModelPlanCacheFile.Key(
                    1, opaque, 1, opaque, 1, opaque, 1, opaque, 1, opaque, 1, opaque,
                    1, 1, 1, opaque, index, 3);
            entries.put(key, new ModelPlanCacheFile.Entry(key,
                    index == 15 ? new byte[lastDecisionBytes] : maximumDecision, opaque, SUMMARY));
            lastKey = key;
        }
        Path cache = directory.resolve("boundary.bin");
        ModelPlanCacheFile file = new ModelPlanCacheFile();

        file.publish(cache, entries);

        byte[] prior = Files.readAllBytes(cache);
        assertEquals(ModelPlanCacheFile.MAX_FILE_BYTES, prior.length);
        assertEquals(entries.keySet(), file.load(cache).keySet());
        entries.put(lastKey, new ModelPlanCacheFile.Entry(lastKey,
                new byte[lastDecisionBytes + 1], opaque, SUMMARY));
        ModelPlanCacheFile rejecting = new ModelPlanCacheFile(
                (temporary, target) -> fail("oversize must fail before publication"),
                (temporary, target) -> fail("oversize must not replace the target"));

        IOException failure = assertThrows(IOException.class, () -> rejecting.publish(cache, entries));
        assertEquals("model-plan cache exceeds size bound", failure.getMessage());
        assertArrayEquals(prior, Files.readAllBytes(cache));
        Path absent = directory.resolve("absent.bin");
        IOException absentFailure = assertThrows(IOException.class,
                () -> rejecting.publish(absent, entries));
        assertEquals("model-plan cache exceeds size bound", absentFailure.getMessage());
        assertFalse(Files.exists(absent));
        try (var files = Files.list(directory)) {
            assertEquals(List.of(cache), files.toList());
        }
    }

    private static void assertSizeRejected(List<ModelPlanCacheFile.Entry> entries) {
        IOException failure = assertThrows(IOException.class,
                () -> ModelPlanCacheFile.encodedSize(entries));
        assertEquals("model-plan cache exceeds size bound", failure.getMessage());
    }

    private static ModelPlanCacheFile.Entry fixtureEntry(int decisionBytes) {
        ModelPlanCacheFile.Key key = new ModelPlanCacheFile.Key(
                1, sequence(0x10, 1), 2, sequence(0x20, 2), 3, sequence(0x30, 3),
                4, sequence(0x40, 4), 5, sequence(0x50, 5), 6, sequence(0x60, 6),
                8, 9, 7, sequence(0x70, 7), 10, 3);
        return new ModelPlanCacheFile.Entry(
                key, sequence(0x80, decisionBytes), sequence(0x90, 9), SUMMARY);
    }

    private static byte[] sequence(int first, int length) {
        byte[] result = new byte[length];
        for (int index = 0; index < length; index++) {
            result[index] = (byte) (first + index);
        }
        return result;
    }

    private static void assertEntryEquals(
            ModelPlanCacheFile.Entry expected, ModelPlanCacheFile.Entry actual) {
        assertEquals(expected.key(), actual.key());
        assertArrayEquals(expected.decision(), actual.decision());
        assertArrayEquals(expected.winner(), actual.winner());
        assertEquals(expected.summary(), actual.summary());
    }
}

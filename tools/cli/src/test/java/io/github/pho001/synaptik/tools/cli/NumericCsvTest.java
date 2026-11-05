package io.github.pho001.synaptik.tools.cli;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Real decoder geometry, exact traversal and independently borrowed reusable native stores. */
final class NumericCsvTest {
    @TempDir Path directory;

    @Test
    void finiteDecimalRowsIgnoreBlanksPreserveOrderAndRefillDistinctNativeStores() throws Exception {
        Path path = Files.writeString(directory.resolve("values.csv"),
                "\n 1, -2.5, 3e1 \r\n\r\n4,+5.,.6\n7,8,9\n10,11,12\n");
        NumericCsv data = NumericCsv.scan(path, 3);
        assertEquals(4, data.rows());
        assertEquals(3, data.columns());
        data.requireBatches(2);
        var signature = CliModel.signature(CliModel.f32(false, 2, 2), CliModel.f32(true, 2, 1));
        try (NativeBatch batch = new NativeBatch(signature, true); var cursor = data.open()) {
            var input = batch.input().hostStorage().orElseThrow().segment();
            var target = batch.target().hostStorage().orElseThrow().segment();
            assertTrue(input.isNative());
            assertTrue(target.isNative());
            batch.fill(cursor);
            assertArrayEquals(new float[] {1, -2.5f, 4, 5}, input.toArray(ValueLayout.JAVA_FLOAT), 0);
            assertArrayEquals(new float[] {30, .6f}, target.toArray(ValueLayout.JAVA_FLOAT), 0);
            target.set(ValueLayout.JAVA_FLOAT, 0, -123f);
            assertEquals(1, input.get(ValueLayout.JAVA_FLOAT, 0));
            batch.fill(cursor);
            assertSame(input, batch.input().hostStorage().orElseThrow().segment());
            assertSame(target, batch.target().hostStorage().orElseThrow().segment());
            assertArrayEquals(new float[] {7, 8, 10, 11}, input.toArray(ValueLayout.JAVA_FLOAT), 0);
            assertArrayEquals(new float[] {9, 12}, target.toArray(ValueLayout.JAVA_FLOAT), 0);
            assertFalse(cursor.next());
        }
    }

    @Test
    void changesWithUnchangedShapeAreDetectedAtEndOfTraversal() throws Exception {
        Path path = Files.writeString(directory.resolve("stable.csv"), "1,2\n3,4\n");
        NumericCsv data = NumericCsv.scan(path, 2);
        Files.writeString(path, "1,2\n3,5\n");
        try (var cursor = data.open()) {
            assertTrue(cursor.next());
            assertTrue(cursor.next());
            assertEquals(5, cursor.value(1));
            assertThrows(IllegalArgumentException.class, cursor::next);
        }
    }

    @Test
    void cardinalityChangesCannotBePaddedRepeatedOrSilentlyIgnored() throws Exception {
        Path path = Files.writeString(directory.resolve("count.csv"), "1\n2\n");
        NumericCsv data = NumericCsv.scan(path, 1);
        Files.writeString(path, "1\n");
        try (var cursor = data.open()) {
            assertTrue(cursor.next());
            assertThrows(IllegalArgumentException.class, cursor::next);
        }
        Files.writeString(path, "1\n2\n3\n");
        try (var cursor = data.open()) {
            assertTrue(cursor.next());
            assertTrue(cursor.next());
            assertThrows(IllegalArgumentException.class, cursor::next);
        }
    }

    @Test
    void nativeBatchArenaIsOwnedByTheCallerRatherThanTheTensorWrapper() throws Exception {
        var signature = CliModel.signature(CliModel.f32(false, 1, 1), CliModel.f32(true, 1, 1));
        NativeBatch batch = new NativeBatch(signature, true);
        var input = batch.input();
        var target = batch.target();
        assertTrue(input.hostStorage().orElseThrow().isAlive());
        batch.close();
        assertFalse(input.hostStorage().orElseThrow().isAlive());
        assertFalse(target.hostStorage().orElseThrow().isAlive());
    }
}

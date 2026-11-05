package io.github.pho001.synaptik.tools.cli;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.regex.Pattern;

/**
 * Finite decimal FLOAT32 CSV without headers, quoting or empty fields. Blank rows are ignored.
 * A complete first traversal validates geometry; each subsequent traversal also authenticates
 * the original bytes before a successful result can be saved or published. The user must keep
 * the file stable while it is consumed; this is change detection, not a filesystem snapshot.
 */
final class NumericCsv {
    private static final Pattern DECIMAL = Pattern.compile(
            "[+-]?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)(?:[eE][+-]?[0-9]+)?");
    private final Path path;
    private final int columns;
    private final long rows;
    private final byte[] digest;

    private NumericCsv(Path path, int columns, long rows, byte[] digest) {
        this.path = path;
        this.columns = columns;
        this.rows = rows;
        this.digest = digest;
    }

    /**
     * Validates the entire data file before any execution or update.
     * @param path existing UTF-8 numeric file, kept stable by its owner
     * @param expectedColumns required width, or zero to infer from the first nonblank row
     * @return immutable traversal geometry and content fingerprint
     * @throws IOException for file/decoding failures
     * @throws IllegalArgumentException for empty, malformed or inconsistent data
     */
    static NumericCsv scan(Path path, int expectedColumns) throws IOException {
        try (Cursor cursor = new Cursor(path, expectedColumns, null, -1)) {
            while (cursor.next()) { /* Validate all rows before exposing geometry. */ }
            if (cursor.rowsRead == 0) throw new IllegalArgumentException("CSV has no examples: " + path);
            return new NumericCsv(path, cursor.columns, cursor.rowsRead, cursor.finalDigest);
        }
    }

    int columns() { return columns; }
    long rows() { return rows; }

    void requireBatches(int batchSize) {
        if (rows % batchSize != 0) {
            throw new IllegalArgumentException("CSV has " + rows + " rows, not divisible by static batch "
                    + batchSize + "; no rows may be dropped, repeated or padded");
        }
    }

    Cursor open() throws IOException { return new Cursor(path, columns, digest, rows); }

    /** One streaming traversal with a single reusable decoded row, never retained by execution. */
    static final class Cursor implements AutoCloseable {
        private final Path path;
        private final BufferedReader reader;
        private final MessageDigest digest;
        private final byte[] expectedDigest;
        private final long expectedRows;
        private int columns;
        private float[] row;
        private long lineNumber;
        private long rowsRead;
        private byte[] finalDigest;

        private Cursor(Path path, int columns, byte[] expectedDigest, long expectedRows) throws IOException {
            this.path = path;
            this.columns = columns;
            this.expectedDigest = expectedDigest;
            this.expectedRows = expectedRows;
            this.digest = sha256();
            this.reader = new BufferedReader(new InputStreamReader(
                    new DigestInputStream(Files.newInputStream(path), digest),
                    StandardCharsets.UTF_8.newDecoder()));
        }

        /**
         * Decodes the next nonblank row and verifies the complete original bytes at EOF.
         * @return true while a current reusable row is available
         * @throws IOException for file or decoding failure
         * @throws IllegalArgumentException for data errors or a changed file
         */
        boolean next() throws IOException {
            if (finalDigest != null) return false;
            String line;
            while ((line = reader.readLine()) != null) {
                lineNumber++;
                if (line.isBlank()) continue;
                int width = 1;
                for (int i = 0; i < line.length(); i++) if (line.charAt(i) == ',') width++;
                if (columns == 0) columns = width;
                if (width != columns) throw invalid("expected " + columns + " columns, found " + width);
                if (row == null) row = new float[columns];
                int start = 0;
                for (int column = 0; column < columns; column++) {
                    int comma = line.indexOf(',', start);
                    int end = comma < 0 ? line.length() : comma;
                    String field = line.substring(start, end).strip();
                    if (!DECIMAL.matcher(field).matches()) {
                        throw invalid("column " + (column + 1) + " is not a decimal number");
                    }
                    float value;
                    try {
                        value = Float.parseFloat(field);
                    } catch (NumberFormatException failure) {
                        throw invalid("column " + (column + 1) + " is not FLOAT32");
                    }
                    if (!Float.isFinite(value)) throw invalid("column " + (column + 1) + " is not finite FLOAT32");
                    row[column] = value;
                    start = end + 1;
                }
                rowsRead = Math.incrementExact(rowsRead);
                if (expectedRows >= 0 && rowsRead > expectedRows) throw invalid("file changed after validation");
                return true;
            }
            finalDigest = digest.digest();
            if (expectedDigest != null && (rowsRead != expectedRows || !Arrays.equals(expectedDigest, finalDigest))) {
                throw invalid("file changed after validation; no result will be published");
            }
            return false;
        }

        float value(int column) { return row[column]; }

        private IllegalArgumentException invalid(String detail) {
            return new IllegalArgumentException(path + ":" + lineNumber + ": " + detail);
        }

        @Override
        public void close() throws IOException { reader.close(); }
    }

    private static MessageDigest sha256() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("JDK SHA-256 is unavailable", impossible);
        }
    }
}

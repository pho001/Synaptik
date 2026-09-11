package io.github.pho001.synaptik.tools.tuning;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;

/**
 * Bounded deterministic binary persistence for compact compatible workload results.
 *
 * <p>The file contains no executable payload or rich raw samples. A complete file is validated
 * before any entry is returned, and publication uses a forced same-directory temporary file plus
 * atomic replacement. Instances own no open channel or other persistent resource.
 */
final class WorkloadCacheFile {
    static final int MAX_FILE_BYTES = 16 * 1024 * 1024;
    static final int MAX_ENTRIES = 65_536;
    static final int MAX_OPAQUE_BYTES = 1024 * 1024;

    private static final byte[] MAGIC = {
        0x53, 0x59, 0x4e, 0x54, 0x55, 0x4e, 0x45, 0x31
    };
    private static final int ARTIFACT_SCHEMA = 1;
    private static final int CHECKSUM_BYTES = 32;
    private static final int HEADER_BYTES = MAGIC.length + Integer.BYTES + Integer.BYTES;
    private static final int ENTRY_FIXED_BYTES = Integer.BYTES * 8 + Long.BYTES * 3;

    @FunctionalInterface
    interface BeforeMove {
        /**
         * Runs after the temporary file is forced and before atomic replacement.
         *
         * @param temporary non-null same-directory temporary file
         * @param target non-null absolute target path
         * @throws IOException if publication must stop before replacement
         */
        void run(Path temporary, Path target) throws IOException;
    }

    @FunctionalInterface
    interface AtomicMover {
        /**
         * Atomically replaces the target with the completed temporary file.
         *
         * @param temporary non-null same-directory temporary file
         * @param target non-null absolute target path
         * @throws IOException if atomic replacement fails
         */
        void move(Path temporary, Path target) throws IOException;
    }

    /** Immutable cache lookup key with snapshotted compatibility bytes. */
    record Key(
            int compatibilitySchema,
            byte[] compatibility,
            int objective,
            int warmupCount,
            int timedSampleCount) {
        /**
         * Validates the cache-key fields and snapshots compatibility bytes.
         *
         * @throws NullPointerException if {@code compatibility} is {@code null}
         * @throws IllegalArgumentException if schemas/objective/sampling fields or byte bounds are
         *     invalid
         */
        Key {
            if (compatibilitySchema <= 0) {
                throw new IllegalArgumentException("compatibilitySchema must be positive");
            }
            compatibility = boundedCopy(compatibility, "compatibility");
            if (objective <= 0) {
                throw new IllegalArgumentException("objective must be positive");
            }
            if (warmupCount < 0) {
                throw new IllegalArgumentException("warmupCount must be non-negative");
            }
            if (timedSampleCount <= 0 || (timedSampleCount & 1) == 0) {
                throw new IllegalArgumentException("timedSampleCount must be positive and odd");
            }
        }

        /** @return a fresh caller-owned copy of the non-empty compatibility bytes */
        @Override
        public byte[] compatibility() {
            return compatibility.clone();
        }

        @Override
        public boolean equals(java.lang.Object other) {
            return this == other || other instanceof Key that
                    && compatibilitySchema == that.compatibilitySchema
                    && objective == that.objective
                    && warmupCount == that.warmupCount
                    && timedSampleCount == that.timedSampleCount
                    && Arrays.equals(compatibility, that.compatibility);
        }

        @Override
        public int hashCode() {
            int result = Integer.hashCode(compatibilitySchema);
            result = 31 * result + Arrays.hashCode(compatibility);
            result = 31 * result + Integer.hashCode(objective);
            result = 31 * result + Integer.hashCode(warmupCount);
            return 31 * result + Integer.hashCode(timedSampleCount);
        }
    }

    /** Immutable compact cache entry with snapshotted decision and winner bytes. */
    record Entry(
            Key key,
            byte[] encodedDecision,
            byte[] winnerIdentity,
            WorkloadTuningResult.SampleSummary summary) {
        /**
         * Validates and snapshots one compact entry.
         *
         * @throws NullPointerException if any component is {@code null}
         * @throws IllegalArgumentException if opaque byte bounds are invalid or summary count does
         *     not equal the key's timed sample count
         */
        Entry {
            Objects.requireNonNull(key, "key");
            encodedDecision = boundedCopy(encodedDecision, "encodedDecision");
            winnerIdentity = boundedCopy(winnerIdentity, "winnerIdentity");
            Objects.requireNonNull(summary, "summary");
            if (summary.sampleCount() != key.timedSampleCount()) {
                throw new IllegalArgumentException("summary sample count does not match key");
            }
        }

        /** @return a fresh caller-owned copy of the non-empty encoded decision */
        @Override
        public byte[] encodedDecision() {
            return encodedDecision.clone();
        }

        /** @return a fresh caller-owned copy of the non-empty winner identity */
        @Override
        public byte[] winnerIdentity() {
            return winnerIdentity.clone();
        }
    }

    private static final Comparator<Key> KEY_ORDER = (left, right) -> {
        int compared = Arrays.compareUnsigned(left.compatibility, right.compatibility);
        if (compared != 0) return compared;
        compared = Integer.compare(left.compatibilitySchema, right.compatibilitySchema);
        if (compared != 0) return compared;
        compared = Integer.compare(left.objective, right.objective);
        if (compared != 0) return compared;
        compared = Integer.compare(left.warmupCount, right.warmupCount);
        if (compared != 0) return compared;
        return Integer.compare(left.timedSampleCount, right.timedSampleCount);
    };

    private final BeforeMove beforeMove;
    private final AtomicMover mover;

    /** Creates a cache codec using the production atomic replacement operation. */
    WorkloadCacheFile() {
        this((temporary, target) -> { }, (temporary, target) -> Files.move(
                temporary,
                target,
                StandardCopyOption.ATOMIC_MOVE,
                StandardCopyOption.REPLACE_EXISTING));
    }

    /**
     * Creates a cache codec with publication seams used to verify failure atomicity.
     *
     * @param beforeMove non-null action run after forcing the temporary file
     * @param mover non-null atomic replacement action
     * @throws NullPointerException if either action is {@code null}
     */
    WorkloadCacheFile(BeforeMove beforeMove, AtomicMover mover) {
        this.beforeMove = Objects.requireNonNull(beforeMove, "beforeMove");
        this.mover = Objects.requireNonNull(mover, "mover");
    }

    /**
     * Loads and validates the complete cache, or returns an empty map when the file is absent.
     *
     * @param path non-null explicit cache path
     * @return an immutable map of validated compact entries; never {@code null}
     * @throws NullPointerException if {@code path} is {@code null}
     * @throws IOException if the file cannot be read or violates any size, checksum, schema,
     *     ordering, uniqueness, or structural invariant
     */
    Map<Key, Entry> load(Path path) throws IOException {
        Objects.requireNonNull(path, "path");
        if (!Files.exists(path)) {
            return Map.of();
        }
        long size = Files.size(path);
        if (size < HEADER_BYTES + CHECKSUM_BYTES || size > MAX_FILE_BYTES) {
            throw new IOException("workload cache size is invalid");
        }
        byte[] complete = new byte[Math.toIntExact(size)];
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            ByteBuffer bounded = ByteBuffer.wrap(complete);
            while (bounded.hasRemaining()) {
                if (channel.read(bounded) < 0) {
                    throw new IOException("workload cache was truncated while loading");
                }
            }
            ByteBuffer extra = ByteBuffer.allocate(1);
            if (channel.read(extra) >= 0) {
                throw new IOException("workload cache grew while loading");
            }
        }
        int payloadLength = complete.length - CHECKSUM_BYTES;
        byte[] actualChecksum = Arrays.copyOfRange(complete, payloadLength, complete.length);
        byte[] expectedChecksum = digest(complete, 0, payloadLength);
        if (!MessageDigest.isEqual(expectedChecksum, actualChecksum)) {
            throw new IOException("workload cache checksum is invalid");
        }

        try {
            ByteBuffer input = ByteBuffer.wrap(complete, 0, payloadLength).order(ByteOrder.BIG_ENDIAN);
            byte[] magic = readBytes(input, MAGIC.length, MAGIC.length, "magic");
            if (!Arrays.equals(MAGIC, magic)) {
                throw new IOException("workload cache magic is invalid");
            }
            int artifactSchema = readInt(input, "artifact schema");
            if (artifactSchema != ARTIFACT_SCHEMA) {
                throw new IOException("unsupported workload cache artifact schema");
            }
            int entryCount = readInt(input, "entry count");
            if (entryCount < 0 || entryCount > MAX_ENTRIES) {
                throw new IOException("workload cache entry count is invalid");
            }
            Map<Key, Entry> entries = new LinkedHashMap<>(capacity(entryCount));
            Key previousKey = null;
            for (int index = 0; index < entryCount; index++) {
                int compatibilitySchema = readInt(input, "compatibility schema");
                byte[] compatibility = readLengthPrefixed(input, "compatibility");
                int objective = readInt(input, "objective");
                int warmupCount = readInt(input, "warmup count");
                int timedSampleCount = readInt(input, "timed sample count");
                byte[] decision = readLengthPrefixed(input, "decision");
                byte[] winner = readLengthPrefixed(input, "winner identity");
                long minimum = readLong(input, "minimum");
                long median = readLong(input, "median");
                long maximum = readLong(input, "maximum");
                int sampleCount = readInt(input, "sample count");
                Key key = new Key(
                        compatibilitySchema,
                        compatibility,
                        objective,
                        warmupCount,
                        timedSampleCount);
                Entry entry = new Entry(
                        key,
                        decision,
                        winner,
                        new WorkloadTuningResult.SampleSummary(
                                minimum, median, maximum, sampleCount));
                if (previousKey != null && KEY_ORDER.compare(previousKey, key) >= 0) {
                    throw new IOException("workload cache keys are not in strict canonical order");
                }
                if (entries.putIfAbsent(key, entry) != null) {
                    throw new IOException("workload cache contains a duplicate key");
                }
                previousKey = key;
            }
            if (input.hasRemaining()) {
                throw new IOException("workload cache contains trailing bytes");
            }
            return Map.copyOf(entries);
        } catch (IOException exception) {
            throw exception;
        } catch (IllegalArgumentException | ArithmeticException exception) {
            throw new IOException("workload cache structure is invalid", exception);
        }
    }

    /**
     * Deterministically encodes and atomically replaces one cache file.
     *
     * @param target non-null explicit cache path whose parent directory already exists
     * @param entries non-null complete entry map; keys and values must be non-null and agree
     * @throws NullPointerException if an argument, key, or entry is {@code null}
     * @throws IOException if encoding, temporary-file I/O, forcing, or atomic replacement fails
     */
    void publish(Path target, Map<Key, Entry> entries) throws IOException {
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(entries, "entries");
        byte[] encoded = encode(entries);
        Path absoluteTarget = target.toAbsolutePath();
        Path parent = absoluteTarget.getParent();
        if (parent == null || !Files.isDirectory(parent)) {
            throw new IOException("workload cache parent directory does not exist");
        }
        Path temporary = parent.resolve("." + absoluteTarget.getFileName()
                + "." + UUID.randomUUID() + ".tmp");
        try {
            try (FileChannel channel = FileChannel.open(
                    temporary, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
                ByteBuffer output = ByteBuffer.wrap(encoded);
                while (output.hasRemaining()) {
                    channel.write(output);
                }
                channel.force(true);
            }
            beforeMove.run(temporary, absoluteTarget);
            mover.move(temporary, absoluteTarget);
            forceDirectory(parent);
        } catch (AtomicMoveNotSupportedException exception) {
            throw exception;
        } finally {
            Files.deleteIfExists(temporary);
        }
    }

    /**
     * Produces canonical bounded big-endian file bytes including the whole-payload checksum.
     *
     * @param suppliedEntries non-null complete entry map; keys and values must be non-null and
     *     agree
     * @return a new caller-owned byte array in deterministic key order
     * @throws NullPointerException if the map, a key, or an entry is {@code null}
     * @throws IOException if entry count, structure, or encoded file size violates its bound
     * @throws ArithmeticException if checked encoded-size arithmetic overflows
     */
    byte[] encode(Map<Key, Entry> suppliedEntries) throws IOException {
        Objects.requireNonNull(suppliedEntries, "suppliedEntries");
        if (suppliedEntries.size() > MAX_ENTRIES) {
            throw new IOException("too many workload cache entries");
        }
        List<Entry> entries = new ArrayList<>(suppliedEntries.size());
        java.util.Set<Key> uniqueKeys = new java.util.HashSet<>();
        for (Map.Entry<Key, Entry> supplied : suppliedEntries.entrySet()) {
            Key key = Objects.requireNonNull(supplied.getKey(), "cache key");
            Entry entry = Objects.requireNonNull(supplied.getValue(), "cache entry");
            if (!key.equals(entry.key())) {
                throw new IOException("cache map key does not match entry key");
            }
            if (!uniqueKeys.add(key)) {
                throw new IOException("cache map contains a duplicate key");
            }
            entries.add(entry);
        }
        if (entries.size() != suppliedEntries.size()) {
            throw new IOException("cache map size is structurally inconsistent");
        }
        entries.sort((left, right) -> KEY_ORDER.compare(left.key(), right.key()));

        long payloadSize = HEADER_BYTES;
        for (Entry entry : entries) {
            payloadSize = Math.addExact(payloadSize, ENTRY_FIXED_BYTES);
            payloadSize = Math.addExact(payloadSize, entry.key().compatibility.length);
            payloadSize = Math.addExact(payloadSize, entry.encodedDecision.length);
            payloadSize = Math.addExact(payloadSize, entry.winnerIdentity.length);
        }
        long totalSize = Math.addExact(payloadSize, CHECKSUM_BYTES);
        if (totalSize > MAX_FILE_BYTES) {
            throw new IOException("encoded workload cache exceeds its bound");
        }
        ByteBuffer output = ByteBuffer.allocate(Math.toIntExact(totalSize)).order(ByteOrder.BIG_ENDIAN);
        output.put(MAGIC);
        output.putInt(ARTIFACT_SCHEMA);
        output.putInt(entries.size());
        for (Entry entry : entries) {
            Key key = entry.key();
            output.putInt(key.compatibilitySchema);
            putLengthPrefixed(output, key.compatibility);
            output.putInt(key.objective);
            output.putInt(key.warmupCount);
            output.putInt(key.timedSampleCount);
            putLengthPrefixed(output, entry.encodedDecision);
            putLengthPrefixed(output, entry.winnerIdentity);
            output.putLong(entry.summary().minimumNanos());
            output.putLong(entry.summary().medianNanos());
            output.putLong(entry.summary().maximumNanos());
            output.putInt(entry.summary().sampleCount());
        }
        int payloadEnd = output.position();
        output.put(digest(output.array(), 0, payloadEnd));
        return output.array();
    }

    private static byte[] readLengthPrefixed(ByteBuffer input, String field) throws IOException {
        int length = readInt(input, field + " length");
        return readBytes(input, length, MAX_OPAQUE_BYTES, field);
    }

    private static byte[] readBytes(ByteBuffer input, int length, int maximum, String field)
            throws IOException {
        if (length <= 0 || length > maximum || length > input.remaining()) {
            throw new IOException("invalid " + field + " length");
        }
        byte[] bytes = new byte[length];
        input.get(bytes);
        return bytes;
    }

    private static int readInt(ByteBuffer input, String field) throws IOException {
        if (input.remaining() < Integer.BYTES) {
            throw new IOException("truncated " + field);
        }
        return input.getInt();
    }

    private static long readLong(ByteBuffer input, String field) throws IOException {
        if (input.remaining() < Long.BYTES) {
            throw new IOException("truncated " + field);
        }
        return input.getLong();
    }

    private static void putLengthPrefixed(ByteBuffer output, byte[] bytes) {
        output.putInt(bytes.length);
        output.put(bytes);
    }

    private static byte[] boundedCopy(byte[] bytes, String field) {
        Objects.requireNonNull(bytes, field);
        if (bytes.length == 0 || bytes.length > MAX_OPAQUE_BYTES) {
            throw new IllegalArgumentException(field + " length is invalid");
        }
        return bytes.clone();
    }

    private static byte[] digest(byte[] bytes, int offset, int length) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            digest.update(bytes, offset, length);
            return digest.digest();
        } catch (NoSuchAlgorithmException exception) {
            throw new AssertionError("SHA-256 is required by the JDK", exception);
        }
    }

    private static int capacity(int count) {
        if (count < 3) return count + 1;
        return Math.min(MAX_ENTRIES, (int) Math.ceil(count / 0.75d));
    }

    private static void forceDirectory(Path directory) {
        try (FileChannel channel = FileChannel.open(directory, StandardOpenOption.READ)) {
            channel.force(true);
        } catch (IOException | UnsupportedOperationException ignored) {
            // Best effort only: some supported file systems do not expose directory channels.
        }
    }
}

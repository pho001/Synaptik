package io.github.pho001.synaptik.tools.tuning;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
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
 * Bounded canonical persistence for compact selected complete-plan decisions.
 *
 * <p>The package-private file format is schema-versioned, big-endian, deterministically ordered,
 * length-bounded, and protected by a whole-file SHA-256 checksum. Loading validates the complete
 * file before returning any entry. Publication writes and forces a unique same-directory
 * temporary file, performs atomic replacement only, attempts to force the directory, and removes
 * the temporary file on every exit. Entries contain compatibility-key material, an opaque
 * encoded decision, an opaque winner identity, and a compact timing summary; raw samples and
 * correctness evidence are never persisted here.
 *
 * <p>This class stores data only. The surrounding complete-plan transaction decides whether
 * persistence is permitted and requires the producer-owned decoder to authenticate a hit.
 */
final class ModelPlanCacheFile {
    static final int MAX_FILE_BYTES = 16 * 1024 * 1024;
    static final int MAX_ENTRIES = 65_536;
    static final int MAX_OPAQUE_BYTES = 1024 * 1024;
    private static final byte[] MAGIC = {0x53, 0x59, 0x4e, 0x50, 0x4c, 0x41, 0x4e, 0x31};
    private static final int SCHEMA = 1;
    private static final int CHECKSUM_BYTES = 32;

    /**
     * Fully validated schema and canonical entries produced by the authoritative parser.
     *
     * @param artifactSchema positive supported artifact schema
     * @param entries non-null canonical entries; defensively copied
     */
    record Parsed(int artifactSchema, List<Entry> entries) {
        /**
         * Snapshots the parsed entries.
         *
         * @param artifactSchema positive supported artifact schema
         * @param entries non-null canonical entries; defensively copied
         * @throws NullPointerException if {@code entries} or an entry is null
         */
        Parsed {
            entries = List.copyOf(entries);
        }

        /**
         * Projects the already validated unique entries for operational cache lookup.
         *
         * @return immutable key-to-entry map; never null
         */
        Map<Key, Entry> asMap() {
            Map<Key, Entry> result = new LinkedHashMap<>();
            for (Entry entry : entries) {
                result.put(entry.key(), entry);
            }
            return Map.copyOf(result);
        }
    }

    /** Typed internal malformed-content diagnostic shared by loading and inspection. */
    static final class FormatException extends Exception {
        private final TuningInspection.InvalidReason reason;

        /**
         * Creates a diagnostic without an underlying validation failure.
         *
         * @param reason non-null stable invalid-artifact category
         * @param message non-null detail for operational loader failures
         */
        FormatException(TuningInspection.InvalidReason reason, String message) {
            super(message);
            this.reason = reason;
        }

        /**
         * Creates a diagnostic retaining an underlying validation failure.
         *
         * @param reason non-null stable invalid-artifact category
         * @param message non-null detail for operational loader failures
         * @param cause non-null underlying validation failure
         */
        FormatException(TuningInspection.InvalidReason reason, String message, Throwable cause) {
            super(message, cause);
            this.reason = reason;
        }

        /**
         * Returns the stable public inspection category.
         *
         * @return non-null invalid-artifact reason
         */
        TuningInspection.InvalidReason reason() {
            return reason;
        }
    }

    @FunctionalInterface
    interface BeforeMove {
        void run(Path temporary, Path target) throws IOException;
    }

    @FunctionalInterface
    interface AtomicMover {
        void move(Path temporary, Path target) throws IOException;
    }

    record Key(int producerSchema, byte[] producer, int codecSchema, byte[] codec,
            int modelSchema, byte[] model, int profileSchema, byte[] profile,
            int targetSchema, byte[] target, int compatibilitySchema, byte[] compatibility,
            int objective, int correctnessPolicy, int policySchema, byte[] policy,
            int warmupCount, int timedSampleCount) {
        Key {
            if (producerSchema <= 0 || codecSchema <= 0 || modelSchema <= 0 || profileSchema <= 0
                    || targetSchema <= 0 || compatibilitySchema <= 0 || policySchema <= 0
                    || objective <= 0 || correctnessPolicy <= 0 || warmupCount < 0
                    || timedSampleCount <= 0 || (timedSampleCount & 1) == 0) {
                throw new IllegalArgumentException("invalid model-plan cache key");
            }
            producer = bounded(producer, "producer");
            codec = bounded(codec, "codec");
            model = bounded(model, "model");
            profile = bounded(profile, "profile");
            target = bounded(target, "target");
            compatibility = bounded(compatibility, "compatibility");
            policy = bounded(policy, "policy");
        }

        @Override
        public byte[] producer() {
            return producer.clone();
        }

        @Override
        public byte[] codec() {
            return codec.clone();
        }

        @Override
        public byte[] model() {
            return model.clone();
        }

        @Override
        public byte[] profile() {
            return profile.clone();
        }

        @Override
        public byte[] target() {
            return target.clone();
        }

        @Override
        public byte[] compatibility() {
            return compatibility.clone();
        }

        @Override
        public byte[] policy() {
            return policy.clone();
        }

        @Override
        public boolean equals(java.lang.Object other) {
            if (this == other) {
                return true;
            }
            if (!(other instanceof Key k)) {
                return false;
            }
            return producerSchema == k.producerSchema && codecSchema == k.codecSchema
                    && modelSchema == k.modelSchema && profileSchema == k.profileSchema
                    && targetSchema == k.targetSchema && compatibilitySchema == k.compatibilitySchema
                    && objective == k.objective && correctnessPolicy == k.correctnessPolicy
                    && policySchema == k.policySchema && warmupCount == k.warmupCount
                    && timedSampleCount == k.timedSampleCount && Arrays.equals(producer, k.producer)
                    && Arrays.equals(codec, k.codec) && Arrays.equals(model, k.model)
                    && Arrays.equals(profile, k.profile) && Arrays.equals(target, k.target)
                    && Arrays.equals(compatibility, k.compatibility) && Arrays.equals(policy, k.policy);
        }
        @Override
        public int hashCode() {
            int result = Objects.hash(producerSchema, codecSchema, modelSchema, profileSchema,
                    targetSchema, compatibilitySchema, objective, correctnessPolicy, policySchema,
                    warmupCount, timedSampleCount);
            for (byte[] value :
                    List.of(producer, codec, model, profile, target, compatibility, policy)) {
                result = 31 * result + Arrays.hashCode(value);
            }
            return result;
        }
    }

    record Entry(Key key, byte[] decision, byte[] winner,
            CompletePlanTuningResult.SampleSummary summary) {
        Entry {
            Objects.requireNonNull(key, "key");
            decision = bounded(decision, "decision");
            winner = bounded(winner, "winner");
            Objects.requireNonNull(summary, "summary");
            if (summary.sampleCount() != key.timedSampleCount()) {
                throw new IllegalArgumentException("summary count does not match key");
            }
        }

        @Override
        public byte[] decision() {
            return decision.clone();
        }

        @Override
        public byte[] winner() {
            return winner.clone();
        }
    }

    private static final Comparator<Key> ORDER = (a, b) -> {
        int c;
        for (byte[][] pair : List.of(
                new byte[][] {a.producer, b.producer},
                new byte[][] {a.codec, b.codec},
                new byte[][] {a.model, b.model},
                new byte[][] {a.profile, b.profile},
                new byte[][] {a.target, b.target},
                new byte[][] {a.compatibility, b.compatibility},
                new byte[][] {a.policy, b.policy})) {
            c = Arrays.compareUnsigned(pair[0], pair[1]);
            if (c != 0) {
                return c;
            }
        }
        c = Integer.compare(a.producerSchema, b.producerSchema);
        if (c != 0) return c;
        c = Integer.compare(a.codecSchema, b.codecSchema);
        if (c != 0) return c;
        c = Integer.compare(a.modelSchema, b.modelSchema);
        if (c != 0) return c;
        c = Integer.compare(a.profileSchema, b.profileSchema);
        if (c != 0) return c;
        c = Integer.compare(a.targetSchema, b.targetSchema);
        if (c != 0) return c;
        c = Integer.compare(a.compatibilitySchema, b.compatibilitySchema);
        if (c != 0) return c;
        c = Integer.compare(a.objective, b.objective);
        if (c != 0) return c;
        c = Integer.compare(a.correctnessPolicy, b.correctnessPolicy);
        if (c != 0) return c;
        c = Integer.compare(a.policySchema, b.policySchema);
        if (c != 0) return c;
        c = Integer.compare(a.warmupCount, b.warmupCount);
        if (c != 0) return c;
        return Integer.compare(a.timedSampleCount, b.timedSampleCount);
    };

    private final BeforeMove beforeMove;
    private final AtomicMover mover;
    ModelPlanCacheFile() {
        this(
                (temporary, target) -> {},
                (temporary, target) -> Files.move(
                        temporary,
                        target,
                        StandardCopyOption.ATOMIC_MOVE,
                        StandardCopyOption.REPLACE_EXISTING));
    }

    ModelPlanCacheFile(BeforeMove beforeMove, AtomicMover mover) {
        this.beforeMove = Objects.requireNonNull(beforeMove, "beforeMove");
        this.mover = Objects.requireNonNull(mover, "mover");
    }

    Map<Key, Entry> load(Path path) throws IOException {
        Objects.requireNonNull(path, "path");
        if (!Files.exists(path)) {
            return Map.of();
        }
        long size = Files.size(path);
        if (size < MAGIC.length + 8L + CHECKSUM_BYTES || size > MAX_FILE_BYTES) {
            throw new IOException("model-plan cache size is invalid");
        }
        byte[] bytes = new byte[Math.toIntExact(size)];
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            ByteBuffer bounded = ByteBuffer.wrap(bytes);
            while (bounded.hasRemaining()) {
                if (channel.read(bounded) < 0) {
                    throw new IOException("model-plan cache was truncated while loading");
                }
            }
            if (channel.read(ByteBuffer.allocate(1)) >= 0) {
                throw new IOException("model-plan cache grew while loading");
            }
        }
        try {
            return parse(bytes).asMap();
        } catch (FormatException exception) {
            throw new IOException(exception.getMessage(), exception);
        }
    }

    /**
     * Parses one complete snapshotted artifact using the format's 16 MiB file, 65,536-entry, and
     * 1 MiB opaque-value bounds without retaining the supplied array.
     *
     * @param bytes non-null complete artifact bytes owned by the caller and not mutated
     * @return validated schema and entries in strict canonical file order; never null
     * @throws NullPointerException if {@code bytes} is null
     * @throws FormatException if any bounded structural, checksum, ordering, or summary rule fails
     */
    static Parsed parse(byte[] bytes) throws FormatException {
        if (bytes.length < MAGIC.length + 8 + CHECKSUM_BYTES) {
            throw new FormatException(
                    TuningInspection.InvalidReason.SIZE_TOO_SMALL,
                    "model-plan cache size is invalid");
        }
        if (bytes.length > MAX_FILE_BYTES) {
            throw new FormatException(
                    TuningInspection.InvalidReason.OVERSIZE,
                    "model-plan cache exceeds its size bound");
        }
        int payloadLength = bytes.length - CHECKSUM_BYTES;
        byte[] expectedChecksum = Arrays.copyOfRange(bytes, payloadLength, bytes.length);
        if (!MessageDigest.isEqual(digest(bytes, 0, payloadLength), expectedChecksum)) {
            throw new FormatException(
                    TuningInspection.InvalidReason.CHECKSUM_MISMATCH,
                    "model-plan cache checksum is invalid");
        }
        ByteBuffer in = ByteBuffer.wrap(bytes, 0, payloadLength).order(ByteOrder.BIG_ENDIAN);
        if (!Arrays.equals(readFixedForInspection(in, MAGIC.length, "magic"), MAGIC)) {
            throw new FormatException(
                    TuningInspection.InvalidReason.INVALID_MAGIC,
                    "model-plan cache magic is invalid");
        }
        int artifactSchema = readIntForInspection(in, "artifact schema");
        if (artifactSchema != SCHEMA) {
            throw new FormatException(
                    TuningInspection.InvalidReason.UNSUPPORTED_SCHEMA,
                    "unsupported model-plan cache schema");
        }
        int count = readIntForInspection(in, "entry count");
        if (count < 0 || count > MAX_ENTRIES) {
            throw new FormatException(
                    TuningInspection.InvalidReason.INVALID_ENTRY_COUNT,
                    "model-plan cache entry count is invalid");
        }
        List<Entry> result = new ArrayList<>(count);
        Key previous = null;
        for (int i = 0; i < count; i++) {
            Key key;
            Entry entry;
            try {
                key = new Key(
                        readIntForInspection(in, "producer schema"),
                        readBytesForInspection(in, "producer"),
                        readIntForInspection(in, "codec schema"),
                        readBytesForInspection(in, "codec"),
                        readIntForInspection(in, "model schema"),
                        readBytesForInspection(in, "model"),
                        readIntForInspection(in, "profile schema"),
                        readBytesForInspection(in, "profile"),
                        readIntForInspection(in, "target schema"),
                        readBytesForInspection(in, "target"),
                        readIntForInspection(in, "compatibility schema"),
                        readBytesForInspection(in, "compatibility"),
                        readIntForInspection(in, "objective"),
                        readIntForInspection(in, "correctness policy"),
                        readIntForInspection(in, "policy schema"),
                        readBytesForInspection(in, "policy"),
                        readIntForInspection(in, "warmup count"),
                        readIntForInspection(in, "timed sample count"));
                entry = new Entry(
                        key,
                        readBytesForInspection(in, "decision"),
                        readBytesForInspection(in, "winner"),
                        new CompletePlanTuningResult.SampleSummary(
                                readLongForInspection(in, "minimum"),
                                readLongForInspection(in, "median"),
                                readLongForInspection(in, "maximum"),
                                readIntForInspection(in, "sample count")));
            } catch (FormatException exception) {
                throw exception;
            } catch (IllegalArgumentException | ArithmeticException exception) {
                throw new FormatException(
                        TuningInspection.InvalidReason.INVALID_STRUCTURE_OR_SUMMARY,
                        "model-plan cache structure is invalid",
                        exception);
            }
            if (previous != null && ORDER.compare(previous, key) >= 0) {
                throw new FormatException(
                        TuningInspection.InvalidReason.DUPLICATE_OR_NONCANONICAL_ORDER,
                        "model-plan cache order or uniqueness is invalid");
            }
            result.add(entry);
            previous = key;
        }
        if (in.hasRemaining()) {
            throw new FormatException(
                    TuningInspection.InvalidReason.TRAILING_PAYLOAD,
                    "model-plan cache has trailing bytes");
        }
        return new Parsed(artifactSchema, result);
    }

    private static byte[] readBytesForInspection(ByteBuffer in, String field)
            throws FormatException {
        int length = readIntForInspection(in, field + " length");
        if (length <= 0 || length > MAX_OPAQUE_BYTES) {
            throw new FormatException(
                    TuningInspection.InvalidReason.INVALID_LENGTH,
                    "invalid " + field + " length");
        }
        return readFixedForInspection(in, length, field);
    }

    private static byte[] readFixedForInspection(ByteBuffer in, int length, String field)
            throws FormatException {
        if (length > in.remaining()) {
            throw new FormatException(
                    TuningInspection.InvalidReason.TRUNCATED,
                    "truncated " + field);
        }
        byte[] bytes = new byte[length];
        in.get(bytes);
        return bytes;
    }

    private static int readIntForInspection(ByteBuffer in, String field)
            throws FormatException {
        if (in.remaining() < Integer.BYTES) {
            throw new FormatException(
                    TuningInspection.InvalidReason.TRUNCATED,
                    "truncated " + field);
        }
        return in.getInt();
    }

    private static long readLongForInspection(ByteBuffer in, String field)
            throws FormatException {
        if (in.remaining() < Long.BYTES) {
            throw new FormatException(
                    TuningInspection.InvalidReason.TRUNCATED,
                    "truncated " + field);
        }
        return in.getLong();
    }

    void publish(Path path, Map<Key, Entry> entries) throws IOException {
        Objects.requireNonNull(path, "path");
        Objects.requireNonNull(entries, "entries");
        if (entries.size() > MAX_ENTRIES) {
            throw new IOException("too many model-plan cache entries");
        }
        List<Entry> ordered = new ArrayList<>(entries.values());
        for (Map.Entry<Key, Entry> mapped : entries.entrySet()) {
            if (!mapped.getKey().equals(mapped.getValue().key())) {
                throw new IOException("model-plan cache map key is inconsistent");
            }
        }
        ordered.sort(Comparator.comparing(Entry::key, ORDER));
        if (ordered.size() != entries.size()) {
            throw new IOException("model-plan cache entry set is inconsistent");
        }
        byte[] encoded = encode(ordered);
        Path target = path.toAbsolutePath();
        Path parent = target.getParent();
        if (parent == null) {
            throw new IOException("model-plan cache path has no parent");
        }
        Path temporary =
                parent.resolve("." + target.getFileName() + "." + UUID.randomUUID() + ".tmp");
        try {
            try (FileChannel channel = FileChannel.open(temporary, StandardOpenOption.CREATE_NEW,
                    StandardOpenOption.WRITE)) {
                ByteBuffer out = ByteBuffer.wrap(encoded);
                while (out.hasRemaining()) {
                    channel.write(out);
                }
                channel.force(true);
            }
            beforeMove.run(temporary, target);
            mover.move(temporary, target);
            forceDirectory(parent);
        } finally {
            Files.deleteIfExists(temporary);
        }
    }

    private static byte[] encode(List<Entry> entries) throws IOException {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        try (DataOutputStream out = new DataOutputStream(buffer)) {
            out.write(MAGIC);
            out.writeInt(SCHEMA);
            out.writeInt(entries.size());
            Key previous = null;
            for (Entry entry : entries) {
                if (previous != null && ORDER.compare(previous, entry.key) >= 0) {
                    throw new IOException("duplicate model-plan cache key");
                }
                Key key = entry.key;
                out.writeInt(key.producerSchema);
                writeBytes(out, key.producer);
                out.writeInt(key.codecSchema);
                writeBytes(out, key.codec);
                out.writeInt(key.modelSchema);
                writeBytes(out, key.model);
                out.writeInt(key.profileSchema);
                writeBytes(out, key.profile);
                out.writeInt(key.targetSchema);
                writeBytes(out, key.target);
                out.writeInt(key.compatibilitySchema);
                writeBytes(out, key.compatibility);
                out.writeInt(key.objective);
                out.writeInt(key.correctnessPolicy);
                out.writeInt(key.policySchema);
                writeBytes(out, key.policy);
                out.writeInt(key.warmupCount);
                out.writeInt(key.timedSampleCount);
                writeBytes(out, entry.decision);
                writeBytes(out, entry.winner);
                out.writeLong(entry.summary.minimumNanos());
                out.writeLong(entry.summary.medianNanos());
                out.writeLong(entry.summary.maximumNanos());
                out.writeInt(entry.summary.sampleCount());
                previous = key;
            }
        }
        byte[] payload = buffer.toByteArray();
        byte[] checksum = digest(payload, 0, payload.length);
        if ((long) payload.length + checksum.length > MAX_FILE_BYTES) {
            throw new IOException("model-plan cache exceeds size bound");
        }
        byte[] complete = Arrays.copyOf(payload, payload.length + checksum.length);
        System.arraycopy(checksum, 0, complete, payload.length, checksum.length);
        return complete;
    }

    private static void writeBytes(DataOutputStream out, byte[] bytes) throws IOException {
        out.writeInt(bytes.length);
        out.write(bytes);
    }

    private static byte[] bounded(byte[] bytes, String name) {
        Objects.requireNonNull(bytes, name);
        if (bytes.length == 0 || bytes.length > MAX_OPAQUE_BYTES) {
            throw new IllegalArgumentException(name + " length is invalid");
        }
        return bytes.clone();
    }

    private static byte[] digest(byte[] bytes, int offset, int length) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            digest.update(bytes, offset, length);
            return digest.digest();
        } catch (NoSuchAlgorithmException exception) {
            throw new AssertionError(exception);
        }
    }

    private static void forceDirectory(Path parent) {
        try (FileChannel channel = FileChannel.open(parent, StandardOpenOption.READ)) {
            channel.force(true);
        } catch (IOException | UnsupportedOperationException ignored) {
            // The file itself has already been forced and atomically moved.
        }
    }
}

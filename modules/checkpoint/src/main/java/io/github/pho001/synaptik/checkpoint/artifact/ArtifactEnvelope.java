package io.github.pho001.synaptik.checkpoint.artifact;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

/**
 * Detached immutable collection of opaque named payloads and a logical kind. It carries no
 * model, execution, or training semantics. Entries are stored in canonical ASCII key order;
 * caller arrays are copied on entry and never exposed for mutation.
 */
public final class ArtifactEnvelope {
    private final String kind;
    private final List<Entry> entries;

    /**
     * Captures and canonically orders entries for a logical artifact kind.
     *
     * @param kind non-null strict ASCII single-segment kind, not equal to {@code .} or {@code ..}
     * @param entries non-null entries with unique strict ASCII slash-separated keys; neither
     *     list nor elements may be null, and caller payload arrays are already copied by Entry
     * @throws NullPointerException if an input or entry is null
     * @throws IllegalArgumentException if kind or keys are invalid or keys repeat
     */
    public ArtifactEnvelope(String kind, List<Entry> entries) {
        this.kind = Objects.requireNonNull(kind, "kind");
        ArtifactWire.validateName(kind, false);
        Objects.requireNonNull(entries, "entries");
        ArrayList<Entry> sorted = new ArrayList<>(entries.size());
        for (Entry entry : entries) {
            sorted.add(Objects.requireNonNull(entry, "entry"));
        }
        sorted.sort(Comparator.comparing(Entry::key));
        for (int i = 1; i < sorted.size(); i++) {
            if (sorted.get(i - 1).key.equals(sorted.get(i).key)) {
                throw new IllegalArgumentException("Duplicate artifact key");
            }
        }
        this.entries = List.copyOf(sorted);
    }

    /**
     * Returns the strict ASCII logical kind, without implying a semantic payload schema.
     *
     * @return non-null immutable kind string
     */
    public String kind() {
        return kind;
    }

    /**
     * Returns entries in canonical key order. Entries and their read-only payload views cannot
     * mutate this image.
     *
     * @return non-null immutable list of detached entries
     */
    public List<Entry> entries() {
        return entries;
    }

    /**
     * Immutable key and payload snapshot. A view is read-only and has independent position;
     * retaining or moving it cannot change this entry.
     */
    public static final class Entry {
        private final String key;
        private final byte[] payload;

        /**
         * Captures one opaque payload under a strict ASCII slash-separated key.
         *
         * @param key non-null key with nonempty segments, excluding {@code .} and {@code ..}
         * @param payload non-null caller-owned array; contents are copied immediately
         * @throws NullPointerException if key or payload is null
         * @throws IllegalArgumentException if the key is noncanonical
         */
        public Entry(String key, byte[] payload) {
            this(key, payload, false);
        }

        private Entry(String key, byte[] payload, boolean owned) {
            this.key = Objects.requireNonNull(key, "key");
            ArtifactWire.validateName(key, true);
            byte[] source = Objects.requireNonNull(payload, "payload");
            this.payload = owned ? source : source.clone();
        }

        static Entry fromOwnedBytes(String key, byte[] payload) {
            return new Entry(key, payload, true);
        }

        /**
         * Returns this entry's canonical logical key.
         *
         * @return non-null immutable key string
         */
        public String key() {
            return key;
        }

        /**
         * Returns a read-only view of the entire detached payload. Each call supplies an
         * independent position and limit; the bytes remain owned by this entry.
         *
         * @return non-null read-only buffer positioned at zero
         */
        public ByteBuffer payload() {
            return ByteBuffer.wrap(payload).asReadOnlyBuffer();
        }

        /**
         * Returns the exact payload length, including zero for an empty payload.
         *
         * @return payload size in bytes
         */
        public int size() {
            return payload.length;
        }

        byte[] bytes() {
            return payload;
        }
    }
}

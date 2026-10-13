package io.github.pho001.synaptik.checkpoint.artifact;

/**
 * Caller-selected upper bounds for one opaque artifact. All byte counts include their stated
 * unit. The reader bounds encoded lengths before allocating the manifest or payloads and checks
 * aggregate arithmetic before payload allocation; the writer checks its image against these
 * limits before encoding the file. The total file limit includes the fixed header, manifest,
 * and payloads.
 *
 * @param maxFileBytes maximum complete file size in bytes, at least 48
 * @param maxManifestBytes maximum manifest size in bytes, between 8 and the conservative JVM
 *     byte-array ceiling {@code Integer.MAX_VALUE - 8}
 * @param maxEntries maximum number of payload entries, possibly zero
 * @param maxKeyBytes maximum ASCII byte length of the kind and of each key, at least one
 * @param maxPayloadBytes maximum byte length of one payload, at most the conservative JVM
 *     byte-array ceiling {@code Integer.MAX_VALUE - 8}
 * @param maxAggregatePayloadBytes maximum sum of payload lengths in bytes
 */
public record ArtifactLimits(long maxFileBytes, int maxManifestBytes, int maxEntries,
                             int maxKeyBytes, int maxPayloadBytes, long maxAggregatePayloadBytes) {
    /**
     * Checks that each configured bound is usable and allocation-safe.
     *
     * @param maxFileBytes maximum complete file size in bytes, at least 48
     * @param maxManifestBytes maximum manifest size in bytes, between 8 and the conservative
     *     JVM byte-array ceiling {@code Integer.MAX_VALUE - 8}
     * @param maxEntries maximum entry count, possibly zero
     * @param maxKeyBytes maximum ASCII bytes in the kind or one key, at least one
     * @param maxPayloadBytes maximum bytes in one payload, between zero and the conservative
     *     JVM byte-array ceiling {@code Integer.MAX_VALUE - 8}
     * @param maxAggregatePayloadBytes maximum sum of payload bytes, at least zero
     * @throws IllegalArgumentException if a limit is negative, a required minimum is missing,
     *     or the manifest or single-payload limit exceeds the conservative JVM array ceiling
     */
    public ArtifactLimits {
        if (maxFileBytes < 48 || maxManifestBytes < 8
                || maxManifestBytes > Integer.MAX_VALUE - 8 || maxEntries < 0 || maxKeyBytes < 1
                || maxPayloadBytes < 0 || maxPayloadBytes > Integer.MAX_VALUE - 8
                || maxAggregatePayloadBytes < 0) {
            throw new IllegalArgumentException("Invalid artifact limits");
        }
    }
}

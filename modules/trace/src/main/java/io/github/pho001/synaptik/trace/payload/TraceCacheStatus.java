package io.github.pho001.synaptik.trace.payload;

/** Classifies the cache fact known to the producer for one preparation outcome. */
public enum TraceCacheStatus {
    /** The producer performed no cache lookup; this is neither a hit nor a miss. */
    NOT_QUERIED
}

package io.github.pho001.synaptik.trace;

import java.util.Objects;

/**
 * Immutable diagnostic envelope that combines common event metadata with a typed payload.
 *
 * <p>The producer supplies both identity and time. {@code monotonicNanos} is a monotonic-clock
 * reading expressed in nanoseconds, not an epoch timestamp. Only differences interpreted within
 * the producer's documented clock domain are meaningful; this record neither reads a clock nor
 * enforces ordering across events.</p>
 *
 * <p>The record is shallowly immutable: construction retains each component reference without
 * copying it, and the open {@link TracePayload} marker cannot enforce payload immutability. A
 * payload implementation is responsible for honoring the immutable diagnostic DTO contract.
 * Equality, hashing, and text use ordinary record component semantics. The envelope does not
 * emit, store, serialize, filter, or otherwise act on the event.</p>
 *
 * @param id non-null producer-supplied identity for this event
 * @param phase non-null lifecycle phase in which the diagnostic fact occurs
 * @param level non-null diagnostic detail or severity classification
 * @param monotonicNanos producer-supplied monotonic-clock reading in nanoseconds, retained exactly;
 *     every {@code long} value is accepted
 * @param payload non-null typed diagnostic payload retained by exact reference; its implementation
 *     is responsible for immutability
 * @param <T> concrete trace payload type; implementations must describe producer facts in
 *     trace-owned terms and honor the immutable DTO contract
 */
public record TraceEvent<T extends TracePayload>(
        TraceEventId id,
        TracePhase phase,
        TraceLevel level,
        long monotonicNanos,
        T payload) {
    /**
     * Creates an immutable diagnostic event from producer-supplied components.
     *
     * @param id non-null producer-supplied identity retained by exact reference
     * @param phase non-null lifecycle phase retained by exact reference
     * @param level non-null diagnostic classification retained by exact reference
     * @param monotonicNanos producer-supplied monotonic-clock reading in nanoseconds, retained
     *     unchanged; negative values are valid
     * @param payload non-null typed diagnostic payload retained by exact reference; its
     *     implementation is responsible for immutability
     * @throws NullPointerException if {@code id}, {@code phase}, {@code level}, or {@code payload}
     *     is {@code null}; validation follows that order and the message names the null component
     */
    public TraceEvent(
            TraceEventId id,
            TracePhase phase,
            TraceLevel level,
            long monotonicNanos,
            T payload) {
        this.id = Objects.requireNonNull(id, "id");
        this.phase = Objects.requireNonNull(phase, "phase");
        this.level = Objects.requireNonNull(level, "level");
        this.monotonicNanos = monotonicNanos;
        this.payload = Objects.requireNonNull(payload, "payload");
    }

    /**
     * Returns the producer-supplied event identity.
     *
     * @return the non-null identity retained by exact reference
     */
    public TraceEventId id() {
        return id;
    }

    /**
     * Returns the lifecycle phase in which the diagnostic fact occurred.
     *
     * @return the non-null phase retained by exact reference
     */
    public TracePhase phase() {
        return phase;
    }

    /**
     * Returns the diagnostic detail or severity classification.
     *
     * @return the non-null level retained by exact reference
     */
    public TraceLevel level() {
        return level;
    }

    /**
     * Returns the producer-supplied monotonic-clock reading.
     *
     * <p>The value is expressed in nanoseconds and retained unchanged. It is not an epoch
     * timestamp, and only differences within the producer's documented clock domain are
     * meaningful.</p>
     *
     * @return the exact caller-supplied {@code long} value
     */
    public long monotonicNanos() {
        return monotonicNanos;
    }

    /**
     * Returns the producer-supplied typed diagnostic payload.
     *
     * @return the non-null payload retained by exact reference; the envelope neither copies nor
     *     mutates it
     */
    public T payload() {
        return payload;
    }
}

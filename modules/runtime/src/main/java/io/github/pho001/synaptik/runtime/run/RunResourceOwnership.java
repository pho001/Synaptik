package io.github.pho001.synaptik.runtime.run;

/**
 * Describes who is responsible for closing a buffer representation during one run.
 *
 * <p>The value applies to one exact representation in one {@link BufferRepresentationBinding}.
 * It does not describe slot validity, residency, physical aliasing, publication, or ownership by
 * a prepared execution.
 */
public enum RunResourceOwnership {
    /**
     * The caller retains ownership and the run must never close the representation.
     *
     * <p>The caller must keep the representation alive and provide any required external
     * synchronization for the complete run.
     */
    BORROWED,

    /**
     * A successfully constructed run owns cleanup of the representation.
     *
     * <p>Construction failure transfers no ownership. After successful construction,
     * {@link RunState#close()} attempts cleanup once unless a later contract explicitly transfers
     * ownership away from the run; no such transition is defined by this foundation.
     */
    RUN_OWNED
}

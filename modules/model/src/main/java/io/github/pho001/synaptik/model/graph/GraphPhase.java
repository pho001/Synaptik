package io.github.pho001.synaptik.model.graph;

/**
 * Classifies a compiled node's role in forward or backward compile-time graph work.
 *
 * <p>A graph phase describes where node work originated in the compiled computation. It is not a
 * compile mode, a runtime schedule, or a prepared execution boundary. The current vocabulary is
 * limited to forward and backward work because optimizer-update graphs remain a future
 * architecture decision. Declaration order is not a serialization contract.</p>
 */
public enum GraphPhase {
    /**
     * Work originating in the captured forward computation.
     */
    FORWARD,

    /**
     * Gradient work introduced for backward computation.
     */
    BACKWARD
}

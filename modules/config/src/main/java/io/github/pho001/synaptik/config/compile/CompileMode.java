package io.github.pho001.synaptik.config.compile;

/**
 * Identifies the requested graph scope for a later graph-compilation operation.
 *
 * <p>A compile mode is immutable declarative configuration. It does not capture or transform a
 * graph, construct gradients, bind publications, select a runtime schedule, prepare backend
 * executables, or run a computation. The compiler and later lifecycle layers will interpret the
 * selected mode when their APIs are introduced.</p>
 */
public enum CompileMode {
    /**
     * Requests forward graph construction and only the requested forward publications.
     */
    FORWARD_ONLY,

    /**
     * Requests later compiler autograd expansion and combined forward and backward compile-time
     * graph work.
     */
    FORWARD_AND_BACKWARD,

    /**
     * Records the architecture's training-step graph-scope direction.
     *
     * <p>This mode does not itself add an optimizer, optimizer-update graph, training session,
     * schedule, or execution behavior.</p>
     */
    TRAINING_STEP
}

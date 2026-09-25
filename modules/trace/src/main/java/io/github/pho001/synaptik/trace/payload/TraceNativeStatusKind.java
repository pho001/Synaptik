package io.github.pho001.synaptik.trace.payload;

/** Classifies the backend-neutral meaning assigned to one returned native status code. */
public enum TraceNativeStatusKind {
    /** Native work completed successfully. */
    SUCCESS,
    /** A native argument was invalid. */
    INVALID_ARGUMENT,
    /** The required device was unavailable. */
    DEVICE_UNAVAILABLE,
    /** The required command queue was unavailable. */
    COMMAND_QUEUE_UNAVAILABLE,
    /** Native allocation failed. */
    ALLOCATION_FAILED,
    /** An index or byte range was outside its valid extent. */
    RANGE_OUT_OF_BOUNDS,
    /** A native copy failed. */
    COPY_FAILED,
    /** Native code reported an internal failure. */
    INTERNAL_ERROR,
    /** The requested shape was unsupported. */
    UNSUPPORTED_SHAPE,
    /** Native graph or kernel compilation failed. */
    COMPILATION_FAILED,
    /** A native resource was incompatible with the operation. */
    INCOMPATIBLE_RESOURCE,
    /** Native execution failed. */
    EXECUTION_FAILED,
    /** The producer did not recognize the exact retained native status code. */
    UNKNOWN
}

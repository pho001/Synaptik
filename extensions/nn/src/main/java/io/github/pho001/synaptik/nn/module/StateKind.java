package io.github.pho001.synaptik.nn.module;

/**
 * Identifies the role of one binding in a {@link StateDictionary}.
 *
 * <p>A parameter is trainable module state discovered by downstream optimizers, while a buffer is
 * persistent module state excluded from parameter discovery. The enum names are in-memory Java
 * identities and are not checkpoint-format or wire tokens.</p>
 */
public enum StateKind {
    /** A trainable {@link Parameter} binding. */
    PARAMETER,

    /** A persistent optimizer-excluded {@link Buffer} binding. */
    BUFFER
}

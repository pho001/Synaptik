package backend.metal.buffer;

/**
 * Access intent for a Metal buffer binding.
 */
public enum MetalBufferAccess {
    /**
     * The native executable reads this buffer but does not write it.
     */
    READ,

    /**
     * The native executable writes this buffer but does not read its prior contents.
     */
    WRITE,

    /**
     * The native executable may read and write this buffer.
     */
    READ_WRITE
}

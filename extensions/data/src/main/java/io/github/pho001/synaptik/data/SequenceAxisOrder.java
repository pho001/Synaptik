package io.github.pho001.synaptik.data;

/**
 * Declares the logical time and batch axes of already ordered sequence values.
 *
 * <p>This is not a physical layout or contiguity declaration. Selecting an order neither
 * transposes values nor changes their original row order or storage geometry.</p>
 */
public enum SequenceAxisOrder {
    /** Leading axes are time then original batch row: {@code [T, B, ...]}. */
    TIME_MAJOR,
    /** Leading axes are original batch row then time: {@code [B, T, ...]}. */
    BATCH_MAJOR
}

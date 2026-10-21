package io.github.pho001.synaptik.data;

/** Selects padding position independently of logical sequence-axis order or physical layout. */
public enum PaddingSide {
    /** Padding precedes valid positions {@code [T-L,T)} at time extent T and count L. */
    LEFT,
    /** Padding follows valid positions {@code [0,L)} at time extent T and count L. */
    RIGHT
}

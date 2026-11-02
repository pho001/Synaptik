package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.model.operation.layout.Window3dAttrs;
import java.util.Arrays;

/**
 * Cold checked spatial-window facts shared by volumetric extraction and its overlap adjoint.
 * Each axis occupies five cells: kernel, stride, padding, dilation, column-grid extent.
 * No instance or array is consulted by generated element loops.
 */
public final class CpuSpatialWindowGeometry {
    private final long[] cells;

    private CpuSpatialWindowGeometry(long[] cells) { this.cells = cells; }

    /**
     * Derives checked depth/height/width windows from an NCDHW shape.
     * @param shape exact rank-five image shape
     * @param window non-null validated Model window
     * @return immutable checked cold facts
     * @throws IllegalArgumentException if the rank or effective kernel fit is invalid
     * @throws ArithmeticException if window products or coordinates overflow
     */
    public static CpuSpatialWindowGeometry of(long[] shape, Window3dAttrs window) {
        if (shape.length != 5) throw new IllegalArgumentException("image window requires NCDHW");
        long[] k = {window.kernelDepth(), window.kernelHeight(), window.kernelWidth()};
        long[] s = {window.strideDepth(), window.strideHeight(), window.strideWidth()};
        long[] p = {window.paddingDepth(), window.paddingHeight(), window.paddingWidth()};
        long[] d = {window.dilationDepth(), window.dilationHeight(), window.dilationWidth()};
        long[] cells = new long[15];
        for (int a = 0; a < 3; a++) {
            long effective = Math.addExact(Math.multiplyExact(k[a] - 1, d[a]), 1);
            long numerator = Math.subtractExact(Math.addExact(shape[a + 2],
                    Math.multiplyExact(2, p[a])), effective);
            if (shape[a + 2] < 0 || numerator < 0)
                throw new IllegalArgumentException("effective window exceeds padded image");
            long grid = Math.addExact(numerator / s[a], 1);
            if (window.ceilMode() && numerator % s[a] != 0) grid = Math.addExact(grid, 1);
            Math.addExact(Math.subtractExact(Math.multiplyExact(grid - 1, s[a]), p[a]),
                    Math.multiplyExact(k[a] - 1, d[a]));
            // Inverse gather evaluates image coordinate + padding - kernel*dilation.
            Math.addExact(Math.max(0, shape[a + 2] - 1), p[a]);
            cells[5*a] = k[a]; cells[5*a+1] = s[a]; cells[5*a+2] = p[a];
            cells[5*a+3] = d[a]; cells[5*a+4] = grid;
        }
        return new CpuSpatialWindowGeometry(cells);
    }

    /**
     * Restores already checked cold facts retained by a named-batch recipe.
     * @param cells fifteen axis-major numeric cells
     * @return immutable snapshot
     * @throws IllegalArgumentException if a cell count or sign is invalid
     */
    public static CpuSpatialWindowGeometry fromCells(long[] cells) {
        if (cells.length != 15) throw new IllegalArgumentException("invalid spatial cell count");
        for (int a = 0; a < 3; a++) if (cells[5*a] <= 0 || cells[5*a+1] <= 0
                || cells[5*a+2] < 0 || cells[5*a+3] <= 0 || cells[5*a+4] <= 0)
            throw new IllegalArgumentException("invalid spatial cells");
        return new CpuSpatialWindowGeometry(cells.clone());
    }

    /** @return defensive copy of axis-major numeric window facts */
    public long[] cells() { return cells.clone(); }
    /** @return checked product of spatial kernel extents */
    public long kernelCount() { return product(0); }
    /** @return checked product of spatial column-grid extents */
    public long columnCount() { return product(4); }
    private long product(int offset) {
        return Math.multiplyExact(Math.multiplyExact(cells[offset], cells[5+offset]), cells[10+offset]);
    }
    /**
     * Checks the columns equation shared by extraction and folding.
     * @param image exact rank-five NCDHW image shape
     * @param columns exact rank-three columns shape
     * @throws IllegalArgumentException if ranks, batch, channels or positions disagree
     * @throws ArithmeticException if the channel/kernel product overflows
     */
    public void requireColumns(long[] image, long[] columns) {
        if (image.length != 5 || columns.length != 3 || columns[0] != image[0]
                || columns[1] != Math.multiplyExact(image[1], kernelCount())
                || columns[2] != columnCount())
            throw new IllegalArgumentException("spatial columns equation disagrees");
    }
    @Override public String toString() { return Arrays.toString(cells); }
}

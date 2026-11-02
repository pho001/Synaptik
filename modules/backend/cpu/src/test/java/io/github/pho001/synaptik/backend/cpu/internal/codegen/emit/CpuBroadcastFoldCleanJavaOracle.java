package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

/**
 * Independent bounded dense Java inverse gather with prebound carriers and geometry.
 * Callers prove array capacity, disjoint storage and every int arithmetic bound before invocation;
 * the reference deliberately performs no extra hot-loop validation. Neither overload retains
 * carriers or geometry, allocates numeric storage, or changes the stored input values.
 */
final class CpuBroadcastFoldCleanJavaOracle {
    /** Prevents construction of this stateless, test-only reference owner. */
    private CpuBroadcastFoldCleanJavaOracle() { }

    /**
     * Writes the complete dense F32 result with positive-zero initialization and sequential F32
     * additions, visiting ky before kx. Input and output are borrowed, nonoverlapping arrays;
     * only the first N*C*H*W output cells are overwritten, and the input is never mutated.
     *
     * <p>Unchecked bounded-domain preconditions: all dimensions/counts and their products,
     * flattened indices, loop increments, kernel/dilation products, and inverse numerator sums
     * and differences fit signed int. OH/OW are the exact window-grid counts for the supplied
     * restored H/W and window. Both arrays cover their stated dense domains. Invalid inputs are
     * outside this reference contract; no overflow, shape, capacity or alias checks are added.</p>
     *
     * @param input non-null stored rounded per-position [N,C,OH*OW] cells, borrowed read-only
     * @param output non-null writable [N,C,H,W] storage, disjoint from input and exclusively owned for the call
     * @param nCount nonnegative batch count N in the proved int domain
     * @param channels nonnegative channel count C in the proved int domain
     * @param height nonnegative restored height H, in elements
     * @param width nonnegative restored width W, in elements
     * @param kh positive kernel height, in kernel positions
     * @param kw positive kernel width, in kernel positions
     * @param sh positive height stride, in restored elements
     * @param sw positive width stride, in restored elements
     * @param ph nonnegative symmetric height padding, in restored elements
     * @param pw nonnegative symmetric width padding, in restored elements
     * @param dh positive kernel height dilation, in restored elements
     * @param dw positive kernel width dilation, in restored elements
     * @param oh nonnegative exact pooled-grid height
     * @param ow nonnegative exact pooled-grid width
     */
    static void fold(float[] input, float[] output, int nCount, int channels, int height, int width,
            int kh, int kw, int sh, int sw, int ph, int pw, int dh, int dw, int oh, int ow) {
        int positions = oh * ow;
        for (int nc = 0; nc < nCount * channels; nc++) {
            int sourceBase = nc * positions, targetBase = nc * height * width;
            for (int y = 0; y < height; y++) for (int x = 0; x < width; x++) {
                float sum = 0;
                for (int ky = 0; ky < kh; ky++) {
                    int a = y + ph - ky * dh;
                    if (a < 0 || a % sh != 0) continue;
                    int oy = a / sh;
                    if (oy >= oh) continue;
                    int row = sourceBase + oy * ow;
                    for (int kx = 0; kx < kw; kx++) {
                        int b = x + pw - kx * dw;
                        if (b < 0 || b % sw != 0) continue;
                        int ox = b / sw;
                        if (ox < ow) sum += input[row + ox];
                    }
                }
                output[targetBase + y * width + x] = sum;
            }
        }
    }

    /**
     * Writes the complete dense F64 result with positive-zero initialization and sequential F64
     * additions, visiting ky before kx. Input and output are borrowed, nonoverlapping arrays;
     * only the first N*C*H*W output cells are overwritten, and the input is never mutated.
     *
     * <p>Unchecked bounded-domain preconditions: all dimensions/counts and their products,
     * flattened indices, loop increments, kernel/dilation products, and inverse numerator sums
     * and differences fit signed int. OH/OW are the exact window-grid counts for the supplied
     * restored H/W and window. Both arrays cover their stated dense domains. Invalid inputs are
     * outside this reference contract; no overflow, shape, capacity or alias checks are added.</p>
     *
     * @param input non-null stored rounded per-position [N,C,OH*OW] cells, borrowed read-only
     * @param output non-null writable [N,C,H,W] storage, disjoint from input and exclusively owned for the call
     * @param nCount nonnegative batch count N in the proved int domain
     * @param channels nonnegative channel count C in the proved int domain
     * @param height nonnegative restored height H, in elements
     * @param width nonnegative restored width W, in elements
     * @param kh positive kernel height, in kernel positions
     * @param kw positive kernel width, in kernel positions
     * @param sh positive height stride, in restored elements
     * @param sw positive width stride, in restored elements
     * @param ph nonnegative symmetric height padding, in restored elements
     * @param pw nonnegative symmetric width padding, in restored elements
     * @param dh positive kernel height dilation, in restored elements
     * @param dw positive kernel width dilation, in restored elements
     * @param oh nonnegative exact pooled-grid height
     * @param ow nonnegative exact pooled-grid width
     */
    static void fold(double[] input, double[] output, int nCount, int channels, int height, int width,
            int kh, int kw, int sh, int sw, int ph, int pw, int dh, int dw, int oh, int ow) {
        int positions = oh * ow;
        for (int nc = 0; nc < nCount * channels; nc++) {
            int sourceBase = nc * positions, targetBase = nc * height * width;
            for (int y = 0; y < height; y++) for (int x = 0; x < width; x++) {
                double sum = 0;
                for (int ky = 0; ky < kh; ky++) {
                    int a = y + ph - ky * dh;
                    if (a < 0 || a % sh != 0) continue;
                    int oy = a / sh;
                    if (oy >= oh) continue;
                    int row = sourceBase + oy * ow;
                    for (int kx = 0; kx < kw; kx++) {
                        int b = x + pw - kx * dw;
                        if (b < 0 || b % sw != 0) continue;
                        int ox = b / sw;
                        if (ox < ow) sum += input[row + ox];
                    }
                }
                output[targetBase + y * width + x] = sum;
            }
        }
    }
}

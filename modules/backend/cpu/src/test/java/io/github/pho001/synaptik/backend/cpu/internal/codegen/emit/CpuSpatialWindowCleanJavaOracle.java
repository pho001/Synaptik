package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

/**
 * Primitive clean-Java design oracle for generated volumetric extraction and inverse gathering.
 * Only cold caller-owned facts are supplied: canonical image extents and axis-major window
 * cells. Kernel coordinates advance with primitive carries; no expanded columns are scanned,
 * and no contributor or output loop allocates or dispatches metadata. The generated owner
 * additionally hoists the window cells into locals and emits physical stride addressing.
 */
final class CpuSpatialWindowCleanJavaOracle {
    private CpuSpatialWindowCleanJavaOracle() { }

    static void unfold(float[] input, float[] output, int channels, int depth, int height, int width,
            long[] w, float padding, int start, int end) {
        long kd = w[0], kh = w[5], kw = w[10], sd = w[1], sh = w[6], sw = w[11];
        long pd = w[2], ph = w[7], pw = w[12], dd = w[3], dh = w[8], dw = w[13];
        long od = w[4], oh = w[9], ow = w[14], kernels = kd * kh * kw, positions = od * oh * ow;
        long p = start % positions, q = start / positions % (channels * kernels);
        long batch = start / positions / (channels * kernels), channel = q / kernels;
        long kz = q / (kh * kw) % kd, ky = q / kw % kh, kx = q % kw;
        long oz = p / (oh * ow), oy = p / ow % oh, ox = p % ow;
        for (int logical = start; logical < end; logical++) {
            long z = oz * sd - pd + kz * dd, y = oy * sh - ph + ky * dh, x = ox * sw - pw + kx * dw;
            output[logical] = z < 0 || z >= depth || y < 0 || y >= height || x < 0 || x >= width
                    ? padding : input[(int) ((((batch * channels + channel) * depth + z) * height + y) * width + x)];
            if (++ox == ow) {
                ox = 0;
                if (++oy == oh) {
                    oy = 0;
                    if (++oz == od) {
                        oz = 0;
                        if (++kx == kw) {
                            kx = 0;
                            if (++ky == kh) {
                                ky = 0;
                                if (++kz == kd) {
                                    kz = 0;
                                    if (++channel == channels) { channel = 0; batch++; }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    static void unfold(double[] input, double[] output, int channels, int depth, int height, int width,
            long[] w, double padding, int start, int end) {
        long kd = w[0], kh = w[5], kw = w[10], sd = w[1], sh = w[6], sw = w[11];
        long pd = w[2], ph = w[7], pw = w[12], dd = w[3], dh = w[8], dw = w[13];
        long od = w[4], oh = w[9], ow = w[14], kernels = kd * kh * kw, positions = od * oh * ow;
        long p = start % positions, q = start / positions % (channels * kernels);
        long batch = start / positions / (channels * kernels), channel = q / kernels;
        long kz = q / (kh * kw) % kd, ky = q / kw % kh, kx = q % kw;
        long oz = p / (oh * ow), oy = p / ow % oh, ox = p % ow;
        for (int logical = start; logical < end; logical++) {
            long z = oz * sd - pd + kz * dd, y = oy * sh - ph + ky * dh, x = ox * sw - pw + kx * dw;
            output[logical] = z < 0 || z >= depth || y < 0 || y >= height || x < 0 || x >= width
                    ? padding : input[(int) ((((batch * channels + channel) * depth + z) * height + y) * width + x)];
            if (++ox == ow) {
                ox = 0;
                if (++oy == oh) {
                    oy = 0;
                    if (++oz == od) {
                        oz = 0;
                        if (++kx == kw) {
                            kx = 0;
                            if (++ky == kh) {
                                ky = 0;
                                if (++kz == kd) {
                                    kz = 0;
                                    if (++channel == channels) { channel = 0; batch++; }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    static void fold(float[] input, float[] output, int channels, int depth, int height, int width,
            long[] w, int start, int end) {
        long kd = w[0], kh = w[5], kw = w[10];
        long sd = w[1], sh = w[6], sw = w[11];
        long pd = w[2], ph = w[7], pw = w[12];
        long dd = w[3], dh = w[8], dw = w[13];
        long od = w[4], oh = w[9], ow = w[14];
        long kernels = kd * kh * kw, positions = od * oh * ow;
        for (int logical = start; logical < end; logical++) {
            int rest = logical, x = rest % width; rest /= width;
            int y = rest % height; rest /= height;
            int z = rest % depth; rest /= depth;
            int channel = rest % channels, batch = rest / channels;
            float sum = +0.0f;
            long kz = 0, ky = 0, kx = 0;
            for (long kernel = 0; kernel < kernels; kernel++) {
                long iz = z + pd - kz * dd, iy = y + ph - ky * dh, ix = x + pw - kx * dw;
                if (iz >= 0 && iy >= 0 && ix >= 0 && iz % sd == 0 && iy % sh == 0 && ix % sw == 0) {
                    iz /= sd; iy /= sh; ix /= sw;
                    if (iz < od && iy < oh && ix < ow) {
                        long column = (iz * oh + iy) * ow + ix;
                        sum += input[(int) (((batch * channels + channel) * kernels + kernel) * positions + column)];
                    }
                }
                if (++kx == kw) { kx = 0; if (++ky == kh) { ky = 0; kz++; } }
            }
            output[logical] = sum;
        }
    }

    static void fold(double[] input, double[] output, int channels, int depth, int height, int width,
            long[] w, int start, int end) {
        long kd = w[0], kh = w[5], kw = w[10];
        long sd = w[1], sh = w[6], sw = w[11];
        long pd = w[2], ph = w[7], pw = w[12];
        long dd = w[3], dh = w[8], dw = w[13];
        long od = w[4], oh = w[9], ow = w[14];
        long kernels = kd * kh * kw, positions = od * oh * ow;
        for (int logical = start; logical < end; logical++) {
            int rest = logical, x = rest % width; rest /= width;
            int y = rest % height; rest /= height;
            int z = rest % depth; rest /= depth;
            int channel = rest % channels, batch = rest / channels;
            double sum = +0.0;
            long kz = 0, ky = 0, kx = 0;
            for (long kernel = 0; kernel < kernels; kernel++) {
                long iz = z + pd - kz * dd, iy = y + ph - ky * dh, ix = x + pw - kx * dw;
                if (iz >= 0 && iy >= 0 && ix >= 0 && iz % sd == 0 && iy % sh == 0 && ix % sw == 0) {
                    iz /= sd; iy /= sh; ix /= sw;
                    if (iz < od && iy < oh && ix < ow) {
                        long column = (iz * oh + iy) * ow + ix;
                        sum += input[(int) (((batch * channels + channel) * kernels + kernel) * positions + column)];
                    }
                }
                if (++kx == kw) { kx = 0; if (++ky == kh) { ky = 0; kz++; } }
            }
            output[logical] = sum;
        }
    }
}

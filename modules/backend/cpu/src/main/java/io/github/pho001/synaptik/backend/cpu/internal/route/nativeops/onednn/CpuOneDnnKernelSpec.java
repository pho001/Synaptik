package io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn;

import java.util.List;
import java.util.Objects;

/**
 * Immutable numeric F32 projection of common CPU IR, not a provider-owned graph IR.
 * Wider convolution/bias/activation descriptors support diagnostics only; production eligibility
 * is separately restricted by {@link CpuOneDnnCompiler#qualified(CpuOneDnnKernelSpec)}.
 *
 * @param operation non-null contraction kind
 * @param inputs non-null descriptor-order operands, defensively copied; no tensor data retained
 * @param output non-null external output geometry
 * @param groups positive group count, exactly one for MATMUL
 * @param strides non-null spatial step sizes, empty for MATMUL; copied
 * @param dilations non-null spatial dilation factors, one means undilated; copied
 * @param paddingLeft non-null non-negative leading spatial padding, empty for MATMUL; copied
 * @param paddingRight non-null non-negative trailing spatial padding, empty for MATMUL; copied
 * @param bias non-null literal bias role, not permission to fold an external ADD
 * @param activation non-null literal terminal activation, not numerical qualification
 */
public record CpuOneDnnKernelSpec(Operation operation, List<Tensor> inputs, Tensor output,
        int groups, List<Long> strides, List<Long> dilations, List<Long> paddingLeft,
        List<Long> paddingRight, Bias bias, Activation activation) {
    /** Contraction descriptor kinds; only qualified MATMUL is production-native. */
    public enum Operation { MATMUL, CONVOLUTION }
    /** Distinguishes intrinsic contraction bias from the exact order of a separate external ADD. */
    public enum Bias { NONE, INTRINSIC, EXTERNAL_LEFT, EXTERNAL_RIGHT }
    /** Diagnostic post-operation vocabulary, not a production capability declaration. */
    public enum Activation { NONE, RELU, TANH, SIGMOID }

    /**
     * Descriptor relative to the bound segment; original origins are applied once outside.
     *
     * @param dimensions non-null positive extents of rank 1..12, defensively copied
     * @param strides non-null matching non-negative element strides, defensively copied
     */
    public record Tensor(List<Long> dimensions, List<Long> strides) {
        /**
         * Validates descriptor rank and checked referenced F32 span without allocating data.
         *
         * @param dimensions positive descriptor-relative extents, copied
         * @param strides matching non-negative element strides, copied
         * @throws NullPointerException if a list or its elements are null
         * @throws IllegalArgumentException if rank, extents or strides are invalid
         * @throws ArithmeticException if referenced span bytes overflow
         */
        public Tensor {
            dimensions = List.copyOf(dimensions);
            strides = List.copyOf(strides);
            if (dimensions.isEmpty() || dimensions.size() > 12 || dimensions.size() != strides.size()
                    || dimensions.stream().anyMatch(value -> value <= 0)
                    || strides.stream().anyMatch(value -> value < 0)) {
                throw new IllegalArgumentException("invalid oneDNN tensor descriptor");
            }
            long span = 1;
            for (int axis = 0; axis < dimensions.size(); axis++) {
                span = Math.addExact(span, Math.multiplyExact(dimensions.get(axis) - 1,
                        strides.get(axis)));
            }
            Math.multiplyExact(span, Float.BYTES);
        }
        /**
         * Computes the referenced span, not an origin prefix or logical element count.
         *
         * @return positive span in bytes from relative element zero through the last referenced F32
         * @throws ArithmeticException if span calculation overflows
         */
        public long byteSize() {
            long span = 1;
            for (int axis = 0; axis < dimensions.size(); axis++) {
                span = Math.addExact(span, Math.multiplyExact(dimensions.get(axis) - 1,
                        strides.get(axis)));
            }
            return Math.multiplyExact(span, Float.BYTES);
        }
    }

    /**
     * Validates literal contraction attributes and snapshots all descriptor lists.
     *
     * @param operation non-null contraction kind
     * @param inputs non-null two operands plus a third exactly when bias is present; copied
     * @param output non-null result descriptor
     * @param groups positive groups; one for MATMUL
     * @param strides non-null positive spatial steps, or empty for MATMUL; copied
     * @param dilations non-null positive spatial dilation factors, or empty for MATMUL; copied
     * @param paddingLeft non-null non-negative leading padding, or empty for MATMUL; copied
     * @param paddingRight non-null non-negative trailing padding, or empty for MATMUL; copied
     * @param bias non-null literal bias role
     * @param activation non-null literal activation kind
     * @throws NullPointerException if a required object or list element is null
     * @throws IllegalArgumentException if operand counts or contraction attributes disagree
     */
    public CpuOneDnnKernelSpec {
        Objects.requireNonNull(operation, "operation");
        inputs = List.copyOf(inputs);
        Objects.requireNonNull(output, "output");
        strides = List.copyOf(strides);
        dilations = List.copyOf(dilations);
        paddingLeft = List.copyOf(paddingLeft);
        paddingRight = List.copyOf(paddingRight);
        Objects.requireNonNull(bias, "bias");
        Objects.requireNonNull(activation, "activation");
        if (groups <= 0 || inputs.size() != (bias == Bias.NONE ? 2 : 3)) {
            throw new IllegalArgumentException("oneDNN contraction input roles disagree");
        }
        int spatial = output.dimensions().size() - 2;
        if (operation == Operation.MATMUL) {
            if (groups != 1 || !strides.isEmpty() || !dilations.isEmpty()
                    || !paddingLeft.isEmpty() || !paddingRight.isEmpty() || bias == Bias.INTRINSIC) {
                throw new IllegalArgumentException("oneDNN MATMUL attributes disagree");
            }
        } else if ((spatial != 2 && spatial != 3) || strides.size() != spatial
                || dilations.size() != spatial || paddingLeft.size() != spatial
                || paddingRight.size() != spatial || strides.stream().anyMatch(value -> value <= 0)
                || dilations.stream().anyMatch(value -> value <= 0)
                || paddingLeft.stream().anyMatch(value -> value < 0)
                || paddingRight.stream().anyMatch(value -> value < 0)) {
            throw new IllegalArgumentException("oneDNN convolution attributes disagree");
        }
    }
}

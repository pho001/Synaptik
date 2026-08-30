package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.operation.pooling.AveragePool1dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool1dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool2dAttrs;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.Objects;

/**
 * Constructs NCW maximum and average pooling as exact three-producer Tensor compositions.
 *
 * <p>This package-private field-free owner validates floating rank-three input and exact width
 * geometry before allocating an intermediate identity. It then expands axis two, delegates to the
 * existing matching NCHW Pool2d expression with singleton-height geometry, and squeezes axis two.
 * It owns no operation kind, graph capture, gradient rule, storage, lowering, backend behavior, or
 * execution.</p>
 */
final class TensorPool1dExpressions {
    /** Prevents instantiation because construction is stateless. */
    private TensorPool1dExpressions() {
    }

    /**
     * Creates an NCW maximum-pooling composition.
     *
     * @param input non-null floating rank-three NCW input retained through the composition
     * @param attrs non-null maximum-pooling width geometry translated to fresh Pool2d attributes
     * @return non-null fresh canonical squeeze output of the exact three-producer composition
     * @throws NullPointerException if {@code input} or {@code attrs} is null, checked in that order
     * @throws IllegalArgumentException if input type, rank, or static width geometry is invalid
     * @throws ArithmeticException if checked width geometry overflows {@code long}
     * @throws IllegalStateException if Tensor identifier space is exhausted; exhaustion after
     *     composition starts may leave earlier intermediate identifiers consumed
     */
    static Tensor apply(Tensor input, MaxPool1dAttrs attrs) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(attrs, "attrs");
        validate(input, attrs.kernelWidth(), attrs.strideWidth(), attrs.paddingWidth(),
                attrs.dilationWidth(), attrs.ceilMode(), "maxPool1d");

        MaxPool2dAttrs mappedAttrs = new MaxPool2dAttrs(
                1, attrs.kernelWidth(),
                1, attrs.strideWidth(),
                0, attrs.paddingWidth(),
                1, attrs.dilationWidth(),
                attrs.ceilMode());
        return input.expandDims(2).maxPool2d(mappedAttrs).squeeze(2);
    }

    /**
     * Creates an NCW fixed-count average-pooling composition.
     *
     * @param input non-null floating rank-three NCW input retained through the composition
     * @param attrs non-null average-pooling width geometry translated to fresh Pool2d attributes
     * @return non-null fresh canonical squeeze output of the exact three-producer composition
     * @throws NullPointerException if {@code input} or {@code attrs} is null, checked in that order
     * @throws IllegalArgumentException if input type, rank, or static width geometry is invalid
     * @throws ArithmeticException if checked width geometry overflows {@code long}
     * @throws IllegalStateException if Tensor identifier space is exhausted; exhaustion after
     *     composition starts may leave earlier intermediate identifiers consumed
     */
    static Tensor apply(Tensor input, AveragePool1dAttrs attrs) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(attrs, "attrs");
        validate(input, attrs.kernelWidth(), attrs.strideWidth(), attrs.paddingWidth(),
                attrs.dilationWidth(), attrs.ceilMode(), "averagePool1d");

        AveragePool2dAttrs mappedAttrs = new AveragePool2dAttrs(
                1, attrs.kernelWidth(),
                1, attrs.strideWidth(),
                0, attrs.paddingWidth(),
                1, attrs.dilationWidth(),
                attrs.ceilMode());
        return input.expandDims(2).averagePool2d(mappedAttrs).squeeze(2);
    }

    /**
     * Validates the complete input-aware Pool1d contract before composition starts.
     *
     * @param input non-null candidate NCW input
     * @param kernelWidth positive width kernel sample or position count
     * @param strideWidth positive width output stride
     * @param paddingWidth non-negative symmetric width padding per side
     * @param dilationWidth positive width kernel dilation
     * @param ceilMode whether output width uses literal ceiling division
     * @param operationName non-null public operation name used in failure messages
     * @throws IllegalArgumentException if input is not floating rank three or static width
     *     geometry cannot fit the effective kernel
     * @throws ArithmeticException if checked width geometry overflows {@code long}
     */
    private static void validate(
            Tensor input,
            long kernelWidth,
            long strideWidth,
            long paddingWidth,
            long dilationWidth,
            boolean ceilMode,
            String operationName) {
        if (!input.descriptor().dataType().isFloating()) {
            throw new IllegalArgumentException(
                    operationName + " input must have a floating data type, but was "
                            + input.descriptor().dataType());
        }

        Shape inputShape = input.descriptor().shape();
        if (inputShape.rank() != 3) {
            throw new IllegalArgumentException(
                    operationName + " input rank must be 3: " + inputShape.rank());
        }

        outputWidth(inputShape.dimension(2), kernelWidth, paddingWidth, dilationWidth,
                strideWidth, ceilMode, operationName);
    }

    /**
     * Derives and validates the exact output-width expression without allocating a Tensor.
     *
     * @param input non-null input-width Dimension
     * @param kernel positive kernel sample or position count
     * @param padding non-negative symmetric padding per side
     * @param dilation positive spacing between kernel samples or positions
     * @param stride positive output-window step
     * @param ceilMode whether the quotient uses literal ceiling division
     * @param operationName non-null public operation name used in a static-geometry failure
     * @return non-null exact static or symbolic output-width Dimension
     * @throws IllegalArgumentException if a static padded width cannot fit the effective kernel
     * @throws ArithmeticException if checked geometry arithmetic overflows {@code long}
     */
    private static Dimension outputWidth(
            Dimension input,
            long kernel,
            long padding,
            long dilation,
            long stride,
            boolean ceilMode,
            String operationName) {
        long effectiveKernel = Math.addExact(
                Math.multiplyExact(dilation, Math.subtractExact(kernel, 1)), 1);
        long doublePadding = Math.multiplyExact(2, padding);
        if (input instanceof StaticDimension staticInput) {
            long paddedInput = Math.addExact(staticInput.size(), doublePadding);
            long numerator = Math.subtractExact(paddedInput, effectiveKernel);
            if (numerator < 0) {
                throw new IllegalArgumentException(
                        operationName + " effective kernel does not fit padded width: input="
                                + input + ", effectiveKernel=" + effectiveKernel
                                + ", padding=" + padding);
            }
            long quotient = numerator / stride;
            if (ceilMode && numerator % stride != 0) {
                quotient = Math.addExact(quotient, 1);
            }
            return new StaticDimension(Math.addExact(quotient, 1));
        }

        long offset = Math.subtractExact(doublePadding, effectiveKernel);
        Dimension numerator = DimensionExpressions.addConstant(input, offset);
        Dimension quotient = ceilMode
                ? DimensionExpressions.ceilingDivide(numerator, stride)
                : DimensionExpressions.floorDivide(numerator, stride);
        return DimensionExpressions.addConstant(quotient, 1);
    }
}

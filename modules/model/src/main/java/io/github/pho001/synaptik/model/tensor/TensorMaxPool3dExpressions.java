package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.Pool3dKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated, storage-free NCDHW maximum-pooling expressions.
 *
 * <p>This package-private field-free boundary owns floating eligibility, rank, checked spatial
 * Shape derivation, descriptor construction, and provenance. It retains unresolved spatial
 * binding obligations in exact dimension expressions. It reads no values, creates no indices or
 * gradient rules, captures no graph, chooses no algorithm or backend, allocates no storage, and
 * does not execute.</p>
 */
final class TensorMaxPool3dExpressions {
    /** Prevents instantiation because maximum-pooling construction is stateless. */
    private TensorMaxPool3dExpressions() {
    }

    /**
     * Creates one maximum-pooling expression for a floating rank-five NCDHW input.
     *
     * <p>For each spatial input extent {@code X}, kernel sample count {@code k}, padding per side
     * {@code p}, dilation {@code d}, and stride {@code s}, output geometry is floor or literal
     * ceiling division of {@code X + 2 * p - (d * (k - 1) + 1)} by {@code s}, plus one. Literal
     * ceiling mode retains terminal all-padding windows.</p>
     *
     * @param input non-null floating rank-five NCDHW input retained as the sole producer input
     * @param attrs non-null exact kernel, stride, padding, dilation, and ceil-mode semantics
     * @return non-null fresh canonical output retaining input type, exact batch/channel
     *     Dimensions and gradient request, with derived spatial Shape, unresolved layout, and
     *     output-index-zero provenance
     * @throws NullPointerException if {@code input} or {@code attrs} is null, checked in that order
     * @throws IllegalArgumentException if input type, rank, or static spatial geometry is invalid
     * @throws ArithmeticException if checked geometry arithmetic overflows {@code long}
     * @throws IllegalStateException if Tensor identifier space is exhausted
     */
    static Tensor apply(Tensor input, MaxPool3dAttrs attrs) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(attrs, "attrs");

        DataType dataType = input.descriptor().dataType();
        if (!dataType.isFloating()) {
            throw new IllegalArgumentException(
                    "maxPool3d input must have a floating data type, but was " + dataType);
        }

        Shape inputShape = input.descriptor().shape();
        if (inputShape.rank() != 5) {
            throw new IllegalArgumentException(
                    "maxPool3d input rank must be 5: " + inputShape.rank());
        }

        Dimension outputDepth = outputDimension(
                inputShape.dimension(2), attrs.kernelDepth(), attrs.paddingDepth(),
                attrs.dilationDepth(), attrs.strideDepth(), attrs.ceilMode(), "depth");
        Dimension outputHeight = outputDimension(
                inputShape.dimension(3), attrs.kernelHeight(), attrs.paddingHeight(),
                attrs.dilationHeight(), attrs.strideHeight(), attrs.ceilMode(), "height");
        Dimension outputWidth = outputDimension(
                inputShape.dimension(4), attrs.kernelWidth(), attrs.paddingWidth(),
                attrs.dilationWidth(), attrs.strideWidth(), attrs.ceilMode(), "width");
        Shape outputShape = Shape.ofDimensions(
                inputShape.dimension(0), inputShape.dimension(1),
                outputDepth, outputHeight, outputWidth);
        TensorDescriptor descriptor = new TensorDescriptor(
                dataType, outputShape, Optional.empty(), input.descriptor().requiresGrad());
        Operation operation = new Operation(Pool3dKind.MAX_POOL3D, attrs);
        return TensorFactory.createDerived(
                descriptor, Optional.empty(), operation, List.of(input));
    }

    private static Dimension outputDimension(
            Dimension input,
            long kernel,
            long padding,
            long dilation,
            long stride,
            boolean ceilMode,
            String axis) {
        long effectiveKernel = Math.addExact(
                Math.multiplyExact(dilation, Math.subtractExact(kernel, 1)), 1);
        long doublePadding = Math.multiplyExact(2, padding);
        if (input instanceof StaticDimension staticInput) {
            long paddedInput = Math.addExact(staticInput.size(), doublePadding);
            long numerator = Math.subtractExact(paddedInput, effectiveKernel);
            if (numerator < 0) {
                throw new IllegalArgumentException(
                        "maxPool3d effective kernel does not fit padded " + axis + ": input="
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

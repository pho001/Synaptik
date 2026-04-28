package tensor.ops.layout;

import operations.Operation;
import operations.layout.contiguous;
import operations.layout.expand;
import operations.layout.expandDims;
import operations.layout.permute;
import operations.layout.reshape;
import operations.layout.squeeze;
import tensor.Tensor;
import tensor.TensorBroadcastOps;
import tensor.TensorInternalAccess;
import tensor.TensorLayoutTransform;
import tensor.TensorMetadata;
import tensor.TensorPrimitiveBuilder;

import java.util.List;

/**
 * Layout and shape transformation operations.
 *
 * <p>Most methods return storage views when the requested transformation can be
 * represented with shape, stride, and storage-offset metadata. They do not copy
 * unless the transformation requires a contiguous physical layout. Returned
 * views share mutable storage with their input.</p>
 */
public final class TensorLayoutOps {
    private TensorLayoutOps() {
    }

    /**
     * Materializes a tensor in contiguous row-major storage order.
     *
     * @param input tensor to copy or normalize; must be non-null
     * @return tensor with the same shape and dtype in contiguous layout
     */
    public static Tensor contiguous(Tensor input) {
        Operation op = new contiguous();
        return TensorPrimitiveBuilder.unary(input, input.getShape(), op, "contiguous", input.getDataType());
    }

    /**
     * Reshapes a tensor without changing logical element order.
     *
     * <p>One requested dimension may be {@code -1} to infer its size. Contiguous
     * inputs produce a view; non-contiguous inputs produce a tensor operation that
     * materializes the requested layout during execution.</p>
     *
     * @param input tensor to reshape; must be non-null
     * @param requestedShape target shape; product must match input flat size
     * @return tensor with the requested shape
     * @throws IllegalArgumentException if the shape is null, invalid, or changes
     *                                  the number of logical elements
     */
    public static Tensor reshape(Tensor input, int[] requestedShape) {
        int[] newShape = TensorLayoutTransform.inferReshape(input.getShape(), requestedShape);
        Operation op = new reshape(newShape);
        Tensor out = input.isContiguous()
                ? TensorPrimitiveBuilder.unaryView(
                        input,
                        newShape,
                        TensorMetadata.computeStrides(newShape),
                        input.getStorageOffsetUnsafe(),
                        op,
                        "reshape",
                        input.getDataType()
                )
                : TensorPrimitiveBuilder.unary(input, newShape, op, "reshape", input.getDataType());
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            LayoutSupport.accumulateGradient(input, outGrad.reshape(input.getShape()));
        });
        return out;
    }

    /**
     * Broadcasts singleton dimensions to a larger shape using zero strides.
     *
     * @param input tensor to broadcast; must be non-null
     * @param requestedShape target shape, with broadcast-compatible dimensions
     * @return read-only broadcast view sharing storage with {@code input}
     * @throws IllegalArgumentException if dimensions are not broadcast-compatible
     */
    public static Tensor expand(Tensor input, int[] requestedShape) {
        int[] targetShape = TensorLayoutTransform.inferExpandShape(input.getShape(), requestedShape);
        int[] targetStrides = LayoutSupport.buildExpandedStrides(input.getShapeUnsafe(), input.getStridesUnsafe(), targetShape);
        Operation op = new expand(targetShape);
        Tensor out = TensorPrimitiveBuilder.unaryView(
                input,
                targetShape,
                targetStrides,
                input.getStorageOffsetUnsafe(),
                op,
                "expand",
                input.getDataType()
        );
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            LayoutSupport.accumulateGradient(input, TensorBroadcastOps.sumToShape(outGrad, input.getShape()));
        });
        return out;
    }

    /**
     * Reorders axes by returning a strided view.
     *
     * @param input tensor to permute; must be non-null
     * @param axes permutation of all axes; negative axes are normalized
     * @return view with shape and strides reordered by {@code axes}
     * @throws IllegalArgumentException if axes are missing, duplicated, or out of range
     */
    public static Tensor permute(Tensor input, int[] axes) {
        int rank = input.getShape().length;
        int[] normalizedAxes = TensorLayoutTransform.normalizeAxes(rank, axes);
        int[] inShape = input.getShape();
        int[] inStrides = input.getStrides();
        int[] outShape = new int[rank];
        int[] outStrides = new int[rank];
        for (int i = 0; i < rank; i++) {
            outShape[i] = inShape[normalizedAxes[i]];
            outStrides[i] = inStrides[normalizedAxes[i]];
        }

        Operation op = new permute(normalizedAxes);
        Tensor out = TensorPrimitiveBuilder.unaryView(
                input,
                outShape,
                outStrides,
                input.getStorageOffsetUnsafe(),
                op,
                "permute",
                input.getDataType()
        );
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            int[] inverse = TensorLayoutTransform.inverseAxes(normalizedAxes);
            LayoutSupport.accumulateGradient(input, outGrad.permute(inverse));
        });
        return out;
    }

    /**
     * Inserts a size-1 dimension.
     *
     * @param input source tensor; must be non-null
     * @param axis insertion position in {@code [0, rank]}; negative axes are normalized
     * @return view with rank increased by one
     * @throws IllegalArgumentException if {@code axis} is outside the valid insertion range
     */
    public static Tensor expandDims(Tensor input, int axis) {
        int rank = input.getShape().length;
        int normalizedAxis = TensorLayoutTransform.normalizeInsertAxis(axis, rank);
        int[] inShape = input.getShape();
        int[] inStrides = input.getStridesUnsafe();
        int[] outShape = new int[rank + 1];
        int[] outStrides = new int[rank + 1];
        for (int i = 0, j = 0; i < outShape.length; i++) {
            if (i == normalizedAxis) {
                outShape[i] = 1;
                outStrides[i] = LayoutSupport.insertedAxisStride(inShape, inStrides, normalizedAxis);
            } else {
                outShape[i] = inShape[j];
                outStrides[i] = inStrides[j];
                j++;
            }
        }
        Operation op = new expandDims(normalizedAxis);
        Tensor out = TensorPrimitiveBuilder.unaryView(
                input,
                outShape,
                outStrides,
                input.getStorageOffsetUnsafe(),
                op,
                "expandDims",
                input.getDataType()
        );
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            LayoutSupport.accumulateGradient(input, outGrad.squeeze(normalizedAxis));
        });
        return out;
    }

    /**
     * Removes a size-1 dimension.
     *
     * @param input source tensor; must be non-null
     * @param axis axis to remove; negative axes are normalized
     * @return view with rank decreased by one
     * @throws IllegalArgumentException if {@code axis} is invalid or the selected dimension is not 1
     */
    public static Tensor squeeze(Tensor input, int axis) {
        int rank = input.getShape().length;
        int normalizedAxis = TensorLayoutTransform.normalizeAxis(axis, rank);
        if (input.getShape()[normalizedAxis] != 1) {
            throw new IllegalArgumentException("Cannot squeeze dimension " + normalizedAxis + " with size " + input.getShape()[normalizedAxis]);
        }
        int[] inShape = input.getShape();
        int[] inStrides = input.getStridesUnsafe();
        int[] outShape = new int[rank - 1];
        int[] outStrides = new int[rank - 1];
        for (int i = 0, j = 0; i < inShape.length; i++) {
            if (i != normalizedAxis) {
                outShape[j] = inShape[i];
                outStrides[j] = inStrides[i];
                j++;
            }
        }
        Operation op = new squeeze(normalizedAxis);
        Tensor out = TensorPrimitiveBuilder.unaryView(
                input,
                outShape,
                outStrides,
                input.getStorageOffsetUnsafe(),
                op,
                "squeeze",
                input.getDataType()
        );
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            LayoutSupport.accumulateGradient(input, outGrad.expandDims(normalizedAxis));
        });
        return out;
    }
}

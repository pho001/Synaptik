package backend.cpu1.kernels.layout.unfold;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.kernels.layout.Cpu1LayoutKernelSupport;
import backend.cpu1.prepare.Cpu1PreparedLayoutUnit;
import backend.runtime.ExecutionContext;
import operations.layout.fold2d;
import operations.layout.unfold2d;
import operations.layout.unfoldAxis;
import tensor.options.Window2dOptions;

import java.util.Arrays;

public final class Cpu1UnfoldLayoutLoops {
    private Cpu1UnfoldLayoutLoops() {
    }

    public static void unfoldAxisScalar(Cpu1PreparedLayoutUnit unit, ExecutionContext context) {
        Cpu1LayoutKernelSupport support = new Cpu1LayoutKernelSupport(unit, context);
        Cpu1LayoutKernelSupport.LayoutCall call = support.bindMaterializingCall();
        Cpu1TensorView input = call.inputs().getFirst();
        Cpu1TensorView output = call.output();
        unfoldAxis unfoldOp = requireUnfoldAxis(support);
        int axis = unfoldOp.getAxis();
        int size = unfoldOp.getSize();
        int step = unfoldOp.getStep();
        int[] prefixShape = input.shape();
        prefixShape[axis] = output.shape(axis);
        int[] prefixDense = Cpu1LayoutKernelSupport.denseStrides(prefixShape);
        support.launchRange(output.elementCount(), (start, end) -> {
            for (int logical = start; logical < end; logical++) {
                int windowOffset = logical % size;
                int prefixLogical = logical / size;
                int remaining = prefixLogical;
                int inputOffset = input.storageOffset();
                for (int dim = 0; dim < input.rank(); dim++) {
                    int coordinate = remaining / prefixDense[dim];
                    remaining %= prefixDense[dim];
                    int inputCoordinate = dim == axis ? coordinate * step + windowOffset : coordinate;
                    inputOffset += inputCoordinate * input.stride(dim);
                }
                int outputOffset = output.storageOffset() + Cpu1LayoutKernelSupport.logicalOffset(
                        logical,
                        output.shape(),
                        output.strides(),
                        Cpu1LayoutKernelSupport.denseStrides(output.shape())
                );
                support.writeElement(output, outputOffset, support.readElement(input, inputOffset));
            }
        });
        support.markOutputWritten(call);
    }

    public static void unfoldAxisLastAxisBlockScalar(Cpu1PreparedLayoutUnit unit, ExecutionContext context) {
        unfoldAxisLastAxisBlock(unit, context, false);
    }

    public static void unfoldAxisLastAxisBlockVector(Cpu1PreparedLayoutUnit unit, ExecutionContext context) {
        unfoldAxisLastAxisBlock(unit, context, true);
    }

    private static void unfoldAxisLastAxisBlock(
            Cpu1PreparedLayoutUnit unit,
            ExecutionContext context,
            boolean vectorized
    ) {
        Cpu1LayoutKernelSupport support = new Cpu1LayoutKernelSupport(unit, context);
        Cpu1LayoutKernelSupport.LayoutCall call = support.bindMaterializingCall();
        Cpu1TensorView input = call.inputs().getFirst();
        Cpu1TensorView output = call.output();
        unfoldAxis unfoldOp = requireUnfoldAxis(support);
        int size = unfoldOp.getSize();
        int step = unfoldOp.getStep();
        int windows = output.shape(input.rank() - 1);
        int rows = input.elementCount() / input.shape(input.rank() - 1);
        support.launchRange(rows * windows, (start, end) -> {
            for (int logical = start; logical < end; logical++) {
                int row = logical / windows;
                int window = logical % windows;
                int inputOffset = input.storageOffset() + row * input.shape(input.rank() - 1) + window * step;
                int outputOffset = output.storageOffset() + logical * size;
                if (vectorized) {
                    support.copyDenseBlockVector(input, inputOffset, output, outputOffset, size);
                } else {
                    support.copyDenseBlockScalar(input, inputOffset, output, outputOffset, size);
                }
            }
        });
        support.markOutputWritten(call);
    }

    public static void unfold2dScalar(Cpu1PreparedLayoutUnit unit, ExecutionContext context) {
        Cpu1LayoutKernelSupport support = new Cpu1LayoutKernelSupport(unit, context);
        Cpu1LayoutKernelSupport.LayoutCall call = support.bindMaterializingCall();
        Cpu1TensorView input = call.inputs().getFirst();
        Cpu1TensorView output = call.output();
        unfold2d unfoldOp = requireUnfold2d(support);
        Window2dPlan plan = Window2dPlan.forUnfold(unfoldOp.getOptions(), input, output);
        support.launchRange(output.elementCount(), (start, end) -> {
            for (int logical = start; logical < end; logical++) {
                Window2dPlan.UnfoldCoordinate coordinate = plan.unfoldCoordinate(logical);
                double value = plan.inBounds(coordinate.imageY(), coordinate.imageX())
                        ? support.readElement(input, plan.inputImageOffset(
                                coordinate.batch(),
                                coordinate.channel(),
                                coordinate.imageY(),
                                coordinate.imageX()
                        ))
                        : 0.0d;
                support.writeElement(output, plan.outputColumnOffset(
                        coordinate.batch(),
                        coordinate.columnChannel(),
                        coordinate.windowIndex()
                ), value);
            }
        });
        support.markOutputWritten(call);
    }

    public static void fold2dScalar(Cpu1PreparedLayoutUnit unit, ExecutionContext context) {
        Cpu1LayoutKernelSupport support = new Cpu1LayoutKernelSupport(unit, context);
        Cpu1LayoutKernelSupport.LayoutCall call = support.bindMaterializingCall();
        Cpu1TensorView input = call.inputs().getFirst();
        Cpu1TensorView output = call.output();
        fold2d foldOp = requireFold2d(support);
        Window2dPlan plan = Window2dPlan.forFold(foldOp.getOptions(), input, output);
        int outputElements = output.elementCount();
        int slotCount = support.rangeSlotCount(input.elementCount());
        double[] acc = support.foldAccumulator(outputElements, slotCount);
        Arrays.fill(acc, 0, Math.multiplyExact(outputElements, slotCount), 0.0d);
        support.launchRangeWithSlot(input.elementCount(), (slotIndex, start, end) -> {
            int base = slotIndex * outputElements;
            for (int logical = start; logical < end; logical++) {
                Window2dPlan.UnfoldCoordinate coordinate = plan.unfoldCoordinate(logical);
                if (plan.inBounds(coordinate.imageY(), coordinate.imageX())) {
                    acc[base + plan.outputImageLogical(
                            coordinate.batch(),
                            coordinate.channel(),
                            coordinate.imageY(),
                            coordinate.imageX()
                    )] += support.readElement(input, plan.inputColumnOffset(
                            coordinate.batch(),
                            coordinate.columnChannel(),
                            coordinate.windowIndex()
                    ));
                }
            }
        });
        int[] outputShape = output.shape();
        int[] outputDense = Cpu1LayoutKernelSupport.denseStrides(outputShape);
        support.launchRange(outputElements, (start, end) -> {
            for (int logical = start; logical < end; logical++) {
                double value = 0.0d;
                for (int slot = 0; slot < slotCount; slot++) {
                    value += acc[slot * outputElements + logical];
                }
                int outputOffset = output.storageOffset()
                        + Cpu1LayoutKernelSupport.logicalOffset(logical, outputShape, output.strides(), outputDense);
                support.writeElement(output, outputOffset, value);
            }
        });
        support.markOutputWritten(call);
    }

    public static void fold2dNonOverlapDirectScalar(Cpu1PreparedLayoutUnit unit, ExecutionContext context) {
        Cpu1LayoutKernelSupport support = new Cpu1LayoutKernelSupport(unit, context);
        Cpu1LayoutKernelSupport.LayoutCall call = support.bindMaterializingCall();
        Cpu1TensorView input = call.inputs().getFirst();
        Cpu1TensorView output = call.output();
        fold2d foldOp = requireFold2d(support);
        Window2dPlan plan = Window2dPlan.forFold(foldOp.getOptions(), input, output);
        support.fillOutputScalar(output, 0.0d);
        support.launchRange(input.elementCount(), (start, end) -> {
            for (int logical = start; logical < end; logical++) {
                Window2dPlan.UnfoldCoordinate coordinate = plan.unfoldCoordinate(logical);
                if (plan.inBounds(coordinate.imageY(), coordinate.imageX())) {
                    support.writeElement(
                            output,
                            plan.inputImageOffset(
                                    coordinate.batch(),
                                    coordinate.channel(),
                                    coordinate.imageY(),
                                    coordinate.imageX()
                            ),
                            support.readElement(input, plan.inputColumnOffset(
                                    coordinate.batch(),
                                    coordinate.columnChannel(),
                                    coordinate.windowIndex()
                            ))
                    );
                }
            }
        });
        support.markOutputWritten(call);
    }

    private static unfoldAxis requireUnfoldAxis(Cpu1LayoutKernelSupport support) {
        if (support.context().runtimeTensorForNodeId(support.unit().nodeId()).getOperation()
                instanceof unfoldAxis unfoldOp) {
            return unfoldOp;
        }
        throw new IllegalArgumentException("cpu1 UNFOLD_AXIS requires operations.layout.unfoldAxis.");
    }

    private static unfold2d requireUnfold2d(Cpu1LayoutKernelSupport support) {
        if (support.context().runtimeTensorForNodeId(support.unit().nodeId()).getOperation()
                instanceof unfold2d unfoldOp) {
            return unfoldOp;
        }
        throw new IllegalArgumentException("cpu1 UNFOLD2D requires operations.layout.unfold2d.");
    }

    private static fold2d requireFold2d(Cpu1LayoutKernelSupport support) {
        if (support.context().runtimeTensorForNodeId(support.unit().nodeId()).getOperation()
                instanceof fold2d foldOp) {
            return foldOp;
        }
        throw new IllegalArgumentException("cpu1 FOLD2D requires operations.layout.fold2d.");
    }

    private static final class Window2dPlan {
        private final Window2dOptions options;
        private final int channels;
        private final int height;
        private final int width;
        private final int kernelW;
        private final int kernelArea;
        private final int outH;
        private final int outW;
        private final int[] imageStrides;
        private final int imageBaseOffset;
        private final int[] columnStrides;
        private final int columnBaseOffset;

        private Window2dPlan(
                Window2dOptions options,
                int channels,
                int height,
                int width,
                int kernelH,
                int kernelW,
                int outH,
                int outW,
                int[] imageStrides,
                int imageBaseOffset,
                int[] columnStrides,
                int columnBaseOffset
        ) {
            this.options = options;
            this.channels = channels;
            this.height = height;
            this.width = width;
            this.kernelW = kernelW;
            this.kernelArea = kernelH * kernelW;
            this.outH = outH;
            this.outW = outW;
            this.imageStrides = imageStrides;
            this.imageBaseOffset = imageBaseOffset;
            this.columnStrides = columnStrides;
            this.columnBaseOffset = columnBaseOffset;
        }

        private static Window2dPlan forUnfold(Window2dOptions options, Cpu1TensorView input, Cpu1TensorView output) {
            if (input.rank() != 4 || output.rank() != 3) {
                throw new IllegalArgumentException("cpu1 UNFOLD2D expects NCHW rank-4 input and rank-3 output.");
            }
            int[] inputShape = input.shape();
            int[] outputShape = output.shape();
            int outW = inferOutW(inputShape[3], options);
            int outH = outputShape[2] / outW;
            return new Window2dPlan(
                    options,
                    inputShape[1],
                    inputShape[2],
                    inputShape[3],
                    options.kernelH(),
                    options.kernelW(),
                    outH,
                    outW,
                    input.strides(),
                    input.storageOffset(),
                    output.strides(),
                    output.storageOffset()
            );
        }

        private static Window2dPlan forFold(Window2dOptions options, Cpu1TensorView input, Cpu1TensorView output) {
            if (input.rank() != 3 || output.rank() != 4) {
                throw new IllegalArgumentException("cpu1 FOLD2D expects rank-3 input and NCHW rank-4 output.");
            }
            int[] inputShape = input.shape();
            int[] outputShape = output.shape();
            int outW = inferOutW(outputShape[3], options);
            int outH = inputShape[2] / outW;
            return new Window2dPlan(
                    options,
                    outputShape[1],
                    outputShape[2],
                    outputShape[3],
                    options.kernelH(),
                    options.kernelW(),
                    outH,
                    outW,
                    output.strides(),
                    output.storageOffset(),
                    input.strides(),
                    input.storageOffset()
            );
        }

        private boolean inBounds(int y, int x) {
            return y >= 0 && y < height && x >= 0 && x < width;
        }

        private int inputImageOffset(int n, int c, int y, int x) {
            return imageBaseOffset
                    + n * imageStrides[0]
                    + c * imageStrides[1]
                    + y * imageStrides[2]
                    + x * imageStrides[3];
        }

        private int outputColumnOffset(int n, int columnChannel, int windowIndex) {
            return columnBaseOffset
                    + n * columnStrides[0]
                    + columnChannel * columnStrides[1]
                    + windowIndex * columnStrides[2];
        }

        private int inputColumnOffset(int n, int columnChannel, int windowIndex) {
            return outputColumnOffset(n, columnChannel, windowIndex);
        }

        private int outputImageLogical(int n, int c, int y, int x) {
            return ((n * channels + c) * height + y) * width + x;
        }

        private UnfoldCoordinate unfoldCoordinate(int logical) {
            int windowIndex = logical % (outH * outW);
            int columnChannel = (logical / (outH * outW)) % (channels * kernelArea);
            int n = logical / ((channels * kernelArea) * outH * outW);
            int c = columnChannel / kernelArea;
            int kernelOffset = columnChannel % kernelArea;
            int kh = kernelOffset / kernelW;
            int kw = kernelOffset % kernelW;
            int oy = windowIndex / outW;
            int ox = windowIndex % outW;
            int iy = oy * options.strideH() - options.padH() + kh * options.dilationH();
            int ix = ox * options.strideW() - options.padW() + kw * options.dilationW();
            return new UnfoldCoordinate(n, c, columnChannel, windowIndex, iy, ix);
        }

        private record UnfoldCoordinate(
                int batch,
                int channel,
                int columnChannel,
                int windowIndex,
                int imageY,
                int imageX
        ) {
        }

        private static int inferOutW(int width, Window2dOptions options) {
            int effectiveKernelW = options.dilationW() * (options.kernelW() - 1) + 1;
            int numerator = width + 2 * options.padW() - effectiveKernelW;
            return (options.ceilMode()
                    ? (numerator + options.strideW() - 1) / options.strideW()
                    : numerator / options.strideW()) + 1;
        }
    }
}

import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import tensor.DataType;
import tensor.Tensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class BroadcastContractMatrixTest {
    @ParameterizedTest
    @EnumSource(value = DataType.class, names = {"FLOAT64", "FLOAT32", "BFLOAT16"})
    void allBroadcastAwareOpsSupportRightAlignedLowerRankRightOperand(DataType dataType) {
        Tensor left = tensor(new double[]{
                1, 2, 3, 4,
                5, 6, 7, 8
        }, new int[]{2, 1, 4}, "left", dataType);
        Tensor right = tensor(new double[]{
                10, 20, 30, 40,
                50, 60, 70, 80,
                90, 100, 110, 120
        }, new int[]{3, 4}, "right", dataType);

        assertForward("add", left, right, dataType);
        assertForward("sub", left, right, dataType);
        assertForward("mul", left, right, dataType);
        assertForward("div", left, right, dataType);
        assertForward("min", left, right, dataType);
        assertForward("max", left, right, dataType);
    }

    @ParameterizedTest
    @EnumSource(value = DataType.class, names = {"FLOAT64", "FLOAT32", "BFLOAT16"})
    void allBroadcastAwareOpsSupportRightAlignedLowerRankLeftOperand(DataType dataType) {
        Tensor left = tensor(new double[]{
                10, 20, 30, 40,
                50, 60, 70, 80,
                90, 100, 110, 120
        }, new int[]{3, 4}, "left", dataType);
        Tensor right = tensor(new double[]{
                1, 2, 3, 4,
                5, 6, 7, 8
        }, new int[]{2, 1, 4}, "right", dataType);

        assertForward("add", left, right, dataType);
        assertForward("sub", left, right, dataType);
        assertForward("mul", left, right, dataType);
        assertForward("div", left, right, dataType);
        assertForward("min", left, right, dataType);
        assertForward("max", left, right, dataType);
    }

    @ParameterizedTest
    @EnumSource(value = DataType.class, names = {"FLOAT64", "FLOAT32", "BFLOAT16"})
    void addSubMulDivReduceGradientsCorrectlyForRankMismatchBroadcast(DataType dataType) {
        assertBackward("add", dataType);
        assertBackward("sub", dataType);
        assertBackward("mul", dataType);
        assertBackward("div", dataType);
    }

    @ParameterizedTest
    @EnumSource(value = DataType.class, names = {"FLOAT64", "FLOAT32", "BFLOAT16"})
    void addSubMulDivReduceGradientsCorrectlyWhenBothOperandsBroadcastAcrossDifferentAxes(DataType dataType) {
        assertBackward(
                "add",
                tensor(new double[]{1, 2, 3}, new int[]{1, 3, 1}, "left", dataType),
                tensor(new double[]{10, 20, 30, 40, 50, 60, 70, 80}, new int[]{2, 1, 4}, "right", dataType),
                dataType
        );
        assertBackward(
                "sub",
                tensor(new double[]{1, 2, 3}, new int[]{1, 3, 1}, "left", dataType),
                tensor(new double[]{10, 20, 30, 40, 50, 60, 70, 80}, new int[]{2, 1, 4}, "right", dataType),
                dataType
        );
        assertBackward(
                "mul",
                tensor(new double[]{1, 2, 3}, new int[]{1, 3, 1}, "left", dataType),
                tensor(new double[]{10, 20, 30, 40, 50, 60, 70, 80}, new int[]{2, 1, 4}, "right", dataType),
                dataType
        );
        assertBackward(
                "div",
                tensor(new double[]{1, 2, 3}, new int[]{1, 3, 1}, "left", dataType),
                tensor(new double[]{10, 20, 30, 40, 50, 60, 70, 80}, new int[]{2, 1, 4}, "right", dataType),
                dataType
        );
    }

    private static void assertForward(String op, Tensor left, Tensor right, DataType dataType) {
        Tensor out = switch (op) {
            case "add" -> left.add(right);
            case "sub" -> left.sub(right);
            case "mul" -> left.mul(right);
            case "div" -> left.div(right);
            case "min" -> left.min(right);
            case "max" -> left.max(right);
            default -> throw new IllegalArgumentException("Unsupported op: " + op);
        };

        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        double[] expected = expectedForward(op, left.toDoubleArrayCopy(), left.getShape(), right.toDoubleArrayCopy(), right.getShape(), dataType);
        assertArrayEquals(expected, out.toDoubleArrayCopy(), epsFor(dataType), "forward mismatch for op=" + op + ", dtype=" + dataType);
        assertArrayEquals(new int[]{2, 3, 4}, out.getShape(), "shape mismatch for op=" + op);
    }

    private static void assertBackward(String op, DataType dataType) {
        Tensor left = tensor(new double[]{
                        1, 2, 3, 4,
                        5, 6, 7, 8
                }, new int[]{2, 1, 4}, "left", dataType);
        Tensor right = tensor(new double[]{
                        10, 20, 30, 40,
                        50, 60, 70, 80,
                        90, 100, 110, 120
                }, new int[]{3, 4}, "right", dataType);
        assertBackward(op, left, right, dataType);
    }

    private static void assertBackward(String op, Tensor left, Tensor right, DataType dataType) {
        left.setRequiresGrad(true);
        right.setRequiresGrad(true);

        Tensor out = switch (op) {
            case "add" -> left.add(right);
            case "sub" -> left.sub(right);
            case "mul" -> left.mul(right);
            case "div" -> left.div(right);
            default -> throw new IllegalArgumentException("Unsupported op: " + op);
        };

        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        GradientPair expected = expectedBackward(op, left.toDoubleArrayCopy(), left.getShape(), right.toDoubleArrayCopy(), right.getShape());
        assertArrayEquals(expected.left(), left.getGradient().toDoubleArrayCopy(), epsFor(dataType), "left grad mismatch for op=" + op + ", dtype=" + dataType);
        assertArrayEquals(expected.right(), right.getGradient().toDoubleArrayCopy(), epsFor(dataType), "right grad mismatch for op=" + op + ", dtype=" + dataType);
    }

    private static double[] expectedForward(String op, double[] left, int[] leftShape, double[] right, int[] rightShape, DataType dataType) {
        int[] outShape = broadcastShape(leftShape, rightShape);
        int[] outStrides = denseStrides(outShape);
        int[] leftStrides = denseStrides(leftShape);
        int[] rightStrides = denseStrides(rightShape);
        double[] out = new double[sizeOf(outShape)];
        int leftOffset = outShape.length - leftShape.length;
        int rightOffset = outShape.length - rightShape.length;

        for (int i = 0; i < out.length; i++) {
            int[] coords = coordsFor(i, outShape, outStrides);
            double a = left[offsetFor(coords, leftShape, leftStrides, leftOffset)];
            double b = right[offsetFor(coords, rightShape, rightStrides, rightOffset)];
            double value = switch (op) {
                case "add" -> a + b;
                case "sub" -> a - b;
                case "mul" -> a * b;
                case "div" -> a / b;
                case "min" -> Math.min(a, b);
                case "max" -> Math.max(a, b);
                default -> throw new IllegalArgumentException("Unsupported op: " + op);
            };
            out[i] = dataType == DataType.BFLOAT16
                    ? backend.cpu.kernels.CpuDTypeOps.fromBFloat16Bits(backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits((float) value))
                    : value;
        }
        return out;
    }

    private static GradientPair expectedBackward(String op, double[] left, int[] leftShape, double[] right, int[] rightShape) {
        int[] outShape = broadcastShape(leftShape, rightShape);
        int[] outStrides = denseStrides(outShape);
        int[] leftStrides = denseStrides(leftShape);
        int[] rightStrides = denseStrides(rightShape);
        double[] leftGrad = new double[sizeOf(leftShape)];
        double[] rightGrad = new double[sizeOf(rightShape)];
        int leftOffset = outShape.length - leftShape.length;
        int rightOffset = outShape.length - rightShape.length;

        for (int i = 0; i < sizeOf(outShape); i++) {
            int[] coords = coordsFor(i, outShape, outStrides);
            int leftIndex = offsetFor(coords, leftShape, leftStrides, leftOffset);
            int rightIndex = offsetFor(coords, rightShape, rightStrides, rightOffset);
            double a = left[leftIndex];
            double b = right[rightIndex];
            switch (op) {
                case "add" -> {
                    leftGrad[leftIndex] += 1.0;
                    rightGrad[rightIndex] += 1.0;
                }
                case "sub" -> {
                    leftGrad[leftIndex] += 1.0;
                    rightGrad[rightIndex] -= 1.0;
                }
                case "mul" -> {
                    leftGrad[leftIndex] += b;
                    rightGrad[rightIndex] += a;
                }
                case "div" -> {
                    leftGrad[leftIndex] += 1.0 / b;
                    rightGrad[rightIndex] += -a / (b * b);
                }
                default -> throw new IllegalArgumentException("Unsupported op: " + op);
            }
        }

        return new GradientPair(leftGrad, rightGrad);
    }

    private static int[] broadcastShape(int[] leftShape, int[] rightShape) {
        int outRank = Math.max(leftShape.length, rightShape.length);
        int[] out = new int[outRank];
        for (int d = 0; d < outRank; d++) {
            int leftDimIndex = d - (outRank - leftShape.length);
            int rightDimIndex = d - (outRank - rightShape.length);
            int leftDim = leftDimIndex < 0 ? 1 : leftShape[leftDimIndex];
            int rightDim = rightDimIndex < 0 ? 1 : rightShape[rightDimIndex];
            out[d] = Math.max(leftDim, rightDim);
        }
        return out;
    }

    private static int[] denseStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    private static int sizeOf(int[] shape) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        return size;
    }

    private static int[] coordsFor(int flatIndex, int[] shape, int[] strides) {
        int[] coords = new int[shape.length];
        int rem = flatIndex;
        for (int i = 0; i < shape.length; i++) {
            coords[i] = rem / strides[i];
            rem %= strides[i];
        }
        return coords;
    }

    private static int offsetFor(int[] outCoords, int[] shape, int[] strides, int rankOffset) {
        int offset = 0;
        for (int i = 0; i < shape.length; i++) {
            int coord = outCoords[i + rankOffset];
            if (shape[i] == 1) {
                coord = 0;
            }
            offset += coord * strides[i];
        }
        return offset;
    }

    private static Tensor tensor(double[] values, int[] shape, String label, DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> new Tensor(Arrays.copyOf(values, values.length), shape, null, label, DataType.FLOAT64);
            case FLOAT32 -> {
                float[] out = new float[values.length];
                for (int i = 0; i < values.length; i++) {
                    out[i] = (float) values[i];
                }
                yield new Tensor(out, shape, null, label, DataType.FLOAT32);
            }
            case BFLOAT16 -> {
                short[] out = new short[values.length];
                for (int i = 0; i < values.length; i++) {
                    out[i] = backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits((float) values[i]);
                }
                yield new Tensor(out, shape, null, label, DataType.BFLOAT16);
            }
            case INT32, BOOL -> throw new UnsupportedOperationException("INT32/BOOL are not part of BroadcastContractMatrixTest.");
        };
    }

    private static double epsFor(DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> 1e-9;
            case FLOAT32 -> 1e-5;
            case BFLOAT16 -> 2e-2;
            case INT32, BOOL -> throw new UnsupportedOperationException("INT32/BOOL are not part of BroadcastContractMatrixTest.");
        };
    }

    private record GradientPair(double[] left, double[] right) {
    }
}

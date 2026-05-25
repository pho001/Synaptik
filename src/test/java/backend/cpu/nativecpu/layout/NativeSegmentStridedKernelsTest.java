package backend.cpu.nativecpu.layout;

import tensor.dtype.TensorDTypeOps;
import backend.cpu.kernels.elementwise.where.CpuWhereKernel;
import backend.cpu.nativecpu.NativeCpuStorageFactory;
import operations.elementwise.compare.greaterThan;
import operations.elementwise.binary.add;
import operations.elementwise.binary.max;
import operations.elementwise.binary.min;
import operations.elementwise.binary.mul;
import operations.elementwise.binary.powTensor;
import operations.elementwise.logical.logicalAnd;
import operations.elementwise.logical.logicalNot;
import operations.elementwise.unary.abs;
import operations.elementwise.unary.clampMax;
import operations.elementwise.unary.clampMin;
import operations.elementwise.unary.ceil;
import operations.elementwise.unary.floor;
import operations.elementwise.unary.mulScalar;
import operations.elementwise.unary.neg;
import operations.elementwise.unary.pow;
import operations.elementwise.unary.relu;
import operations.elementwise.unary.sign;
import operations.reduction.reduceAll;
import operations.reduction.reduceAny;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.storage.NativeBFloat16Storage;
import tensor.storage.NativeBoolStorage;
import tensor.storage.NativeFloat32Storage;
import tensor.storage.NativeFloat64Storage;
import tensor.storage.NativeTensorStorage;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NativeSegmentStridedKernelsTest {
    private final NativeCpuStorageFactory storageFactory = new NativeCpuStorageFactory();

    @Test
    void runsF32UnaryFromDenseAndOffsetSegmentViews() {
        NativeFloat32Storage input = f32Storage(8);
        NativeFloat32Storage denseOut = f32Storage(4);
        NativeFloat32Storage offsetOut = f32Storage(3);
        try {
            write(input, -2f, -1f, 0f, 1f, 2f, 3f, 4f, 5f);

            NativeSegmentStridedKernels.runUnary(
                    new relu(),
                    view(input, DataType.FLOAT32, new int[]{4}, new int[]{1}, 0),
                    view(denseOut, DataType.FLOAT32, new int[]{4}, new int[]{1}, 0),
                    false,
                    false
            );
            assertArrayEquals(new float[]{0f, 0f, 0f, 1f}, read(denseOut, 4), 0.0f);

            NativeSegmentStridedKernels.runUnary(
                    new neg(),
                    view(input, DataType.FLOAT32, new int[]{3}, new int[]{1}, 4),
                    view(offsetOut, DataType.FLOAT32, new int[]{3}, new int[]{1}, 0),
                    false,
                    false
            );
            assertArrayEquals(new float[]{-2f, -3f, -4f}, read(offsetOut, 3), 0.0f);
        } finally {
            input.close();
            denseOut.close();
            offsetOut.close();
        }
    }

    @Test
    void runsF64UnaryFromRank2TransposeReadIntoDenseOutput() {
        NativeFloat64Storage input = f64Storage(6);
        NativeFloat64Storage output = f64Storage(6);
        try {
            write(input, 1.0d, 2.0d, 3.0d, 4.0d, 5.0d, 6.0d);

            NativeSegmentStridedKernels.runUnary(
                    new mulScalar(2.0d),
                    view(input, DataType.FLOAT64, new int[]{3, 2}, new int[]{1, 3}, 0),
                    view(output, DataType.FLOAT64, new int[]{3, 2}, new int[]{2, 1}, 0),
                    false,
                    false
            );

            assertArrayEquals(new double[]{2.0d, 8.0d, 4.0d, 10.0d, 6.0d, 12.0d}, read(output, 6), 0.0d);
        } finally {
            input.close();
            output.close();
        }
    }

    @Test
    void runsF32UnaryFromGeneralStridedReadIntoDenseOutput() {
        NativeFloat32Storage input = f32Storage(8);
        NativeFloat32Storage output = f32Storage(4);
        try {
            write(input, -1f, 2f, 99f, -3f, 4f, 99f, -5f, 6f);

            NativeSegmentStridedKernels.runUnary(
                    new abs(),
                    view(input, DataType.FLOAT32, new int[]{2, 2}, new int[]{3, 1}, 0),
                    view(output, DataType.FLOAT32, new int[]{2, 2}, new int[]{2, 1}, 0),
                    false,
                    false
            );

            assertArrayEquals(new float[]{1f, 2f, 3f, 4f}, read(output, 4), 0.0f);
        } finally {
            input.close();
            output.close();
        }
    }

    @Test
    void runsF32F64FloorCeilSignFromStridedViews() {
        NativeFloat32Storage f32Input = f32Storage(8);
        NativeFloat32Storage f32FloorOut = f32Storage(4);
        NativeFloat32Storage f32SignOut = f32Storage(4);
        NativeFloat64Storage f64Input = f64Storage(6);
        NativeFloat64Storage f64CeilOut = f64Storage(6);
        try {
            write(f32Input, -1.7f, 0.0f, 99f, 2.3f, -0.2f, 99f, 4.9f, -5.1f);
            NativeSegmentView f32View = view(f32Input, DataType.FLOAT32, new int[]{2, 2}, new int[]{3, 1}, 0);

            NativeSegmentStridedKernels.runUnary(
                    new floor(),
                    f32View,
                    view(f32FloorOut, DataType.FLOAT32, new int[]{2, 2}, new int[]{2, 1}, 0),
                    false,
                    false
            );
            assertArrayEquals(new float[]{-2.0f, 0.0f, 2.0f, -1.0f}, read(f32FloorOut, 4), 0.0f);

            NativeSegmentStridedKernels.runUnary(
                    new sign(),
                    f32View,
                    view(f32SignOut, DataType.FLOAT32, new int[]{2, 2}, new int[]{2, 1}, 0),
                    false,
                    false
            );
            assertArrayEquals(new float[]{-1.0f, 0.0f, 1.0f, -1.0f}, read(f32SignOut, 4), 0.0f);

            write(f64Input, -1.7d, 0.0d, 2.3d, -0.2d, 4.9d, -5.1d);
            NativeSegmentStridedKernels.runUnary(
                    new ceil(),
                    view(f64Input, DataType.FLOAT64, new int[]{3, 2}, new int[]{1, 3}, 0),
                    view(f64CeilOut, DataType.FLOAT64, new int[]{3, 2}, new int[]{2, 1}, 0),
                    false,
                    false
            );
            assertArrayEquals(new double[]{-1.0d, -0.0d, 0.0d, 5.0d, 3.0d, -5.0d}, read(f64CeilOut, 6), 0.0d);
        } finally {
            f32Input.close();
            f32FloorOut.close();
            f32SignOut.close();
            f64Input.close();
            f64CeilOut.close();
        }
    }

    @Test
    void copiesBf16RawBitsFromTransposeReadIntoDenseOutput() {
        NativeBFloat16Storage input = bf16Storage(6);
        NativeBFloat16Storage output = bf16Storage(6);
        try {
            short[] bits = new short[]{
                    TensorDTypeOps.toBFloat16Bits(1.0f),
                    TensorDTypeOps.toBFloat16Bits(2.0f),
                    TensorDTypeOps.toBFloat16Bits(3.0f),
                    TensorDTypeOps.toBFloat16Bits(4.0f),
                    TensorDTypeOps.toBFloat16Bits(5.0f),
                    TensorDTypeOps.toBFloat16Bits(6.0f)
            };
            write(input, bits);

            NativeSegmentStridedKernels.copyRaw(
                    view(input, DataType.BFLOAT16, new int[]{3, 2}, new int[]{1, 3}, 0),
                    view(output, DataType.BFLOAT16, new int[]{3, 2}, new int[]{2, 1}, 0)
            );

            assertArrayEquals(new short[]{bits[0], bits[3], bits[1], bits[4], bits[2], bits[5]}, readBits(output, 6));
        } finally {
            input.close();
            output.close();
        }
    }

    @Test
    void runsF32BinaryWithZeroStrideLastDimBroadcastRead() {
        NativeFloat32Storage activations = f32Storage(6);
        NativeFloat32Storage bias = f32Storage(3);
        NativeFloat32Storage output = f32Storage(6);
        try {
            write(activations, 1f, 2f, 3f, 4f, 5f, 6f);
            write(bias, 10f, 20f, 30f);

            NativeSegmentStridedKernels.runBinary(
                    new add(),
                    view(activations, DataType.FLOAT32, new int[]{2, 3}, new int[]{3, 1}, 0),
                    view(bias, DataType.FLOAT32, new int[]{2, 3}, new int[]{0, 1}, 0),
                    view(output, DataType.FLOAT32, new int[]{2, 3}, new int[]{3, 1}, 0)
            );

            assertArrayEquals(new float[]{11f, 22f, 33f, 14f, 25f, 36f}, read(output, 6), 0.0f);
        } finally {
            activations.close();
            bias.close();
            output.close();
        }
    }

    @Test
    void runsF64BinaryWithTransposedAndBroadcastReads() {
        NativeFloat64Storage transposedInput = f64Storage(6);
        NativeFloat64Storage columnScale = f64Storage(2);
        NativeFloat64Storage output = f64Storage(6);
        try {
            write(transposedInput, 1.0d, 2.0d, 3.0d, 4.0d, 5.0d, 6.0d);
            write(columnScale, 10.0d, 100.0d);

            NativeSegmentStridedKernels.runBinary(
                    new mul(),
                    view(transposedInput, DataType.FLOAT64, new int[]{3, 2}, new int[]{1, 3}, 0),
                    view(columnScale, DataType.FLOAT64, new int[]{3, 2}, new int[]{0, 1}, 0),
                    view(output, DataType.FLOAT64, new int[]{3, 2}, new int[]{2, 1}, 0)
            );

            assertArrayEquals(new double[]{10.0d, 400.0d, 20.0d, 500.0d, 30.0d, 600.0d}, read(output, 6), 0.0d);
        } finally {
            transposedInput.close();
            columnScale.close();
            output.close();
        }
    }

    @Test
    void runsBf16BinaryWithPromotedComputeAndRawOutputBits() {
        NativeBFloat16Storage left = bf16Storage(4);
        NativeBFloat16Storage right = bf16Storage(2);
        NativeBFloat16Storage output = bf16Storage(4);
        try {
            write(left,
                    TensorDTypeOps.toBFloat16Bits(1.0f),
                    TensorDTypeOps.toBFloat16Bits(2.0f),
                    TensorDTypeOps.toBFloat16Bits(3.0f),
                    TensorDTypeOps.toBFloat16Bits(4.0f));
            write(right,
                    TensorDTypeOps.toBFloat16Bits(10.0f),
                    TensorDTypeOps.toBFloat16Bits(100.0f));

            NativeSegmentStridedKernels.runBinary(
                    new add(),
                    view(left, DataType.BFLOAT16, new int[]{2, 2}, new int[]{2, 1}, 0),
                    view(right, DataType.BFLOAT16, new int[]{2, 2}, new int[]{0, 1}, 0),
                    view(output, DataType.BFLOAT16, new int[]{2, 2}, new int[]{2, 1}, 0)
            );

            assertArrayEquals(new short[]{
                    TensorDTypeOps.toBFloat16Bits(11.0f),
                    TensorDTypeOps.toBFloat16Bits(102.0f),
                    TensorDTypeOps.toBFloat16Bits(13.0f),
                    TensorDTypeOps.toBFloat16Bits(104.0f)
            }, readBits(output, 4));
        } finally {
            left.close();
            right.close();
            output.close();
        }
    }

    @Test
    void runsMinMaxAndClampWithStridedBroadcastAndPromotedBf16Views() {
        NativeFloat32Storage f32Left = f32Storage(6);
        NativeFloat32Storage f32Right = f32Storage(3);
        NativeFloat32Storage f32MinOut = f32Storage(6);
        NativeFloat32Storage f32ClampOut = f32Storage(4);
        NativeFloat64Storage f64Left = f64Storage(6);
        NativeFloat64Storage f64Right = f64Storage(2);
        NativeFloat64Storage f64MaxOut = f64Storage(6);
        NativeFloat64Storage f64ClampOut = f64Storage(4);
        NativeBFloat16Storage bf16Left = bf16Storage(4);
        NativeBFloat16Storage bf16Right = bf16Storage(2);
        NativeBFloat16Storage bf16MaxOut = bf16Storage(4);
        NativeBFloat16Storage bf16ClampOut = bf16Storage(4);
        try {
            write(f32Left, 1f, 5f, 3f, 7f, 2f, 9f);
            write(f32Right, 4f, 4f, 4f);
            NativeSegmentStridedKernels.runBinary(
                    new min(),
                    view(f32Left, DataType.FLOAT32, new int[]{2, 3}, new int[]{3, 1}, 0),
                    view(f32Right, DataType.FLOAT32, new int[]{2, 3}, new int[]{0, 1}, 0),
                    view(f32MinOut, DataType.FLOAT32, new int[]{2, 3}, new int[]{3, 1}, 0)
            );
            assertArrayEquals(new float[]{1f, 4f, 3f, 4f, 2f, 4f}, read(f32MinOut, 6), 0.0f);

            NativeSegmentStridedKernels.runUnary(
                    new clampMin(3.0f),
                    view(f32Left, DataType.FLOAT32, new int[]{2, 2}, new int[]{3, 1}, 0),
                    view(f32ClampOut, DataType.FLOAT32, new int[]{2, 2}, new int[]{2, 1}, 0),
                    false,
                    false
            );
            assertArrayEquals(new float[]{3f, 5f, 7f, 3f}, read(f32ClampOut, 4), 0.0f);

            write(f64Left, 1.0d, 2.0d, 3.0d, 4.0d, 5.0d, 6.0d);
            write(f64Right, 10.0d, 100.0d);
            NativeSegmentStridedKernels.runBinary(
                    new max(),
                    view(f64Left, DataType.FLOAT64, new int[]{3, 2}, new int[]{1, 3}, 0),
                    view(f64Right, DataType.FLOAT64, new int[]{3, 2}, new int[]{0, 1}, 0),
                    view(f64MaxOut, DataType.FLOAT64, new int[]{3, 2}, new int[]{2, 1}, 0)
            );
            assertArrayEquals(new double[]{10.0d, 100.0d, 10.0d, 100.0d, 10.0d, 100.0d}, read(f64MaxOut, 6), 0.0d);

            NativeSegmentStridedKernels.runUnary(
                    new clampMax(4.0d),
                    view(f64Left, DataType.FLOAT64, new int[]{2, 2}, new int[]{1, 3}, 0),
                    view(f64ClampOut, DataType.FLOAT64, new int[]{2, 2}, new int[]{2, 1}, 0),
                    false,
                    false
            );
            assertArrayEquals(new double[]{1.0d, 4.0d, 2.0d, 4.0d}, read(f64ClampOut, 4), 0.0d);

            write(bf16Left,
                    TensorDTypeOps.toBFloat16Bits(1.0f),
                    TensorDTypeOps.toBFloat16Bits(5.0f),
                    TensorDTypeOps.toBFloat16Bits(3.0f),
                    TensorDTypeOps.toBFloat16Bits(7.0f));
            write(bf16Right,
                    TensorDTypeOps.toBFloat16Bits(4.0f),
                    TensorDTypeOps.toBFloat16Bits(6.0f));
            NativeSegmentStridedKernels.runBinary(
                    new max(),
                    view(bf16Left, DataType.BFLOAT16, new int[]{2, 2}, new int[]{2, 1}, 0),
                    view(bf16Right, DataType.BFLOAT16, new int[]{2, 2}, new int[]{0, 1}, 0),
                    view(bf16MaxOut, DataType.BFLOAT16, new int[]{2, 2}, new int[]{2, 1}, 0)
            );
            assertArrayEquals(new short[]{
                    TensorDTypeOps.toBFloat16Bits(4.0f),
                    TensorDTypeOps.toBFloat16Bits(6.0f),
                    TensorDTypeOps.toBFloat16Bits(4.0f),
                    TensorDTypeOps.toBFloat16Bits(7.0f)
            }, readBits(bf16MaxOut, 4));

            NativeSegmentStridedKernels.runUnary(
                    new clampMax(4.0f),
                    view(bf16Left, DataType.BFLOAT16, new int[]{2, 2}, new int[]{2, 1}, 0),
                    view(bf16ClampOut, DataType.BFLOAT16, new int[]{2, 2}, new int[]{2, 1}, 0),
                    false,
                    false
            );
            assertArrayEquals(new short[]{
                    TensorDTypeOps.toBFloat16Bits(1.0f),
                    TensorDTypeOps.toBFloat16Bits(4.0f),
                    TensorDTypeOps.toBFloat16Bits(3.0f),
                    TensorDTypeOps.toBFloat16Bits(4.0f)
            }, readBits(bf16ClampOut, 4));
        } finally {
            f32Left.close();
            f32Right.close();
            f32MinOut.close();
            f32ClampOut.close();
            f64Left.close();
            f64Right.close();
            f64MaxOut.close();
            f64ClampOut.close();
            bf16Left.close();
            bf16Right.close();
            bf16MaxOut.close();
            bf16ClampOut.close();
        }
    }

    @Test
    void runsF32F64PowAndPowTensorButRejectsBf16PowTensor() {
        NativeFloat32Storage f32Input = f32Storage(8);
        NativeFloat32Storage f32Out = f32Storage(4);
        NativeFloat64Storage f64Base = f64Storage(6);
        NativeFloat64Storage f64Exponent = f64Storage(2);
        NativeFloat64Storage f64Out = f64Storage(6);
        NativeBFloat16Storage bf16Base = bf16Storage(4);
        NativeBFloat16Storage bf16Exponent = bf16Storage(4);
        NativeBFloat16Storage bf16Out = bf16Storage(4);
        try {
            write(f32Input, 1f, 2f, 99f, 3f, 4f, 99f, 5f, 6f);
            NativeSegmentStridedKernels.runUnary(
                    new pow(2.0f),
                    view(f32Input, DataType.FLOAT32, new int[]{2, 2}, new int[]{3, 1}, 0),
                    view(f32Out, DataType.FLOAT32, new int[]{2, 2}, new int[]{2, 1}, 0),
                    false,
                    false
            );
            assertArrayEquals(new float[]{1f, 4f, 9f, 16f}, read(f32Out, 4), 0.0f);

            write(f64Base, 1.0d, 2.0d, 3.0d, 4.0d, 5.0d, 6.0d);
            write(f64Exponent, 2.0d, 0.5d);
            NativeSegmentStridedKernels.runBinary(
                    new powTensor(),
                    view(f64Base, DataType.FLOAT64, new int[]{3, 2}, new int[]{1, 3}, 0),
                    view(f64Exponent, DataType.FLOAT64, new int[]{3, 2}, new int[]{0, 1}, 0),
                    view(f64Out, DataType.FLOAT64, new int[]{3, 2}, new int[]{2, 1}, 0)
            );
            assertArrayEquals(new double[]{
                    1.0d,
                    2.0d,
                    4.0d,
                    Math.sqrt(5.0d),
                    9.0d,
                    Math.sqrt(6.0d)
            }, read(f64Out, 6), 1.0e-12d);

            write(bf16Base,
                    TensorDTypeOps.toBFloat16Bits(1.0f),
                    TensorDTypeOps.toBFloat16Bits(2.0f),
                    TensorDTypeOps.toBFloat16Bits(3.0f),
                    TensorDTypeOps.toBFloat16Bits(4.0f));
            write(bf16Exponent,
                    TensorDTypeOps.toBFloat16Bits(2.0f),
                    TensorDTypeOps.toBFloat16Bits(2.0f),
                    TensorDTypeOps.toBFloat16Bits(2.0f),
                    TensorDTypeOps.toBFloat16Bits(2.0f));
            assertThrows(UnsupportedOperationException.class, () ->
                    NativeSegmentStridedKernels.runBinary(
                            new powTensor(),
                            view(bf16Base, DataType.BFLOAT16, new int[]{2, 2}, new int[]{2, 1}, 0),
                            view(bf16Exponent, DataType.BFLOAT16, new int[]{2, 2}, new int[]{2, 1}, 0),
                            view(bf16Out, DataType.BFLOAT16, new int[]{2, 2}, new int[]{2, 1}, 0)
                    )
            );
        } finally {
            f32Input.close();
            f32Out.close();
            f64Base.close();
            f64Exponent.close();
            f64Out.close();
            bf16Base.close();
            bf16Exponent.close();
            bf16Out.close();
        }
    }

    @Test
    void runsF32CompareWithTransposedAndBroadcastReadsIntoCpuBoolOutput() {
        NativeFloat32Storage left = f32Storage(6);
        NativeFloat32Storage right = f32Storage(2);
        byte[] output = new byte[6];
        try {
            write(left, 1f, 2f, 3f, 4f, 5f, 6f);
            write(right, 2f, 4f);

            NativeSegmentStridedKernels.runCompare(
                    new greaterThan(null),
                    view(left, DataType.FLOAT32, new int[]{3, 2}, new int[]{1, 3}, 0),
                    view(right, DataType.FLOAT32, new int[]{3, 2}, new int[]{0, 1}, 0),
                    output,
                    cpuArrayView(DataType.BOOL, new int[]{3, 2}, new int[]{2, 1}, 0)
            );

            assertArrayEquals(new byte[]{0, 0, 0, 1, 1, 1}, output);
        } finally {
            left.close();
            right.close();
        }
    }

    @Test
    void runsBf16CompareWithPromotedSegmentReadsIntoCpuBoolOutput() {
        NativeBFloat16Storage left = bf16Storage(4);
        NativeBFloat16Storage right = bf16Storage(2);
        byte[] output = new byte[4];
        try {
            write(left,
                    TensorDTypeOps.toBFloat16Bits(1.0f),
                    TensorDTypeOps.toBFloat16Bits(4.0f),
                    TensorDTypeOps.toBFloat16Bits(3.0f),
                    TensorDTypeOps.toBFloat16Bits(2.0f));
            write(right,
                    TensorDTypeOps.toBFloat16Bits(2.0f),
                    TensorDTypeOps.toBFloat16Bits(3.0f));

            NativeSegmentStridedKernels.runCompare(
                    new greaterThan(null),
                    view(left, DataType.BFLOAT16, new int[]{2, 2}, new int[]{2, 1}, 0),
                    view(right, DataType.BFLOAT16, new int[]{2, 2}, new int[]{0, 1}, 0),
                    output,
                    cpuArrayView(DataType.BOOL, new int[]{2, 2}, new int[]{2, 1}, 0)
            );

            assertArrayEquals(new byte[]{0, 1, 1, 0}, output);
        } finally {
            left.close();
            right.close();
        }
    }

    @Test
    void runsF32CompareWithBroadcastReadsIntoNativeBoolMask() {
        NativeFloat32Storage left = f32Storage(6);
        NativeFloat32Storage right = f32Storage(2);
        NativeBoolStorage output = boolStorage(6);
        try {
            write(left, 1f, 2f, 3f, 4f, 5f, 6f);
            write(right, 2f, 4f);

            NativeSegmentStridedKernels.runCompare(
                    new greaterThan(null),
                    view(left, DataType.FLOAT32, new int[]{3, 2}, new int[]{1, 3}, 0),
                    view(right, DataType.FLOAT32, new int[]{3, 2}, new int[]{0, 1}, 0),
                    view(output, DataType.BOOL, new int[]{3, 2}, new int[]{2, 1}, 0)
            );

            assertArrayEquals(new byte[]{0, 0, 0, 1, 1, 1}, readBool(output, 6));
        } finally {
            left.close();
            right.close();
            output.close();
        }
    }

    @Test
    void runsBoolLogicalOpsFromNativeMaskViews() {
        NativeBoolStorage left = boolStorage(6);
        NativeBoolStorage right = boolStorage(2);
        NativeBoolStorage andOut = boolStorage(6);
        NativeBoolStorage notOut = boolStorage(6);
        try {
            write(left, (byte) 1, (byte) 0, (byte) 1, (byte) 1, (byte) 0, (byte) 1);
            write(right, (byte) 1, (byte) 0);

            NativeSegmentStridedKernels.runBinary(
                    new logicalAnd(null),
                    view(left, DataType.BOOL, new int[]{3, 2}, new int[]{1, 3}, 0),
                    view(right, DataType.BOOL, new int[]{3, 2}, new int[]{0, 1}, 0),
                    view(andOut, DataType.BOOL, new int[]{3, 2}, new int[]{2, 1}, 0)
            );
            assertArrayEquals(new byte[]{1, 0, 0, 0, 1, 0}, readBool(andOut, 6));

            NativeSegmentStridedKernels.runUnary(
                    new logicalNot(),
                    view(andOut, DataType.BOOL, new int[]{3, 2}, new int[]{2, 1}, 0),
                    view(notOut, DataType.BOOL, new int[]{3, 2}, new int[]{2, 1}, 0),
                    false,
                    false
            );
            assertArrayEquals(new byte[]{0, 1, 1, 1, 0, 1}, readBool(notOut, 6));
        } finally {
            left.close();
            right.close();
            andOut.close();
            notOut.close();
        }
    }

    @Test
    void runsF32WhereWithStridedCpuArrayConditionAndSegmentBranches() {
        byte[] condition = new byte[]{1, 0, 9, 0, 1};
        NativeFloat32Storage ifTrue = f32Storage(5);
        NativeFloat32Storage ifFalse = f32Storage(2);
        NativeFloat32Storage output = f32Storage(4);
        try {
            write(ifTrue, 1f, 2f, 99f, 3f, 4f);
            write(ifFalse, 10f, 20f);

            NativeSegmentStridedKernels.runWhere(
                    new CpuWhereKernel(),
                    condition,
                    cpuArrayView(DataType.BOOL, new int[]{2, 2}, new int[]{3, 1}, 0),
                    view(ifTrue, DataType.FLOAT32, new int[]{2, 2}, new int[]{3, 1}, 0),
                    view(ifFalse, DataType.FLOAT32, new int[]{2, 2}, new int[]{0, 1}, 0),
                    view(output, DataType.FLOAT32, new int[]{2, 2}, new int[]{2, 1}, 0)
            );

            assertArrayEquals(new float[]{1f, 20f, 10f, 4f}, read(output, 4), 0.0f);
        } finally {
            ifTrue.close();
            ifFalse.close();
            output.close();
        }
    }

    @Test
    void runsBf16WhereWithNativeBoolMaskAndBroadcastBranches() {
        NativeBoolStorage condition = boolStorage(2);
        NativeBFloat16Storage ifTrue = bf16Storage(6);
        NativeBFloat16Storage ifFalse = bf16Storage(2);
        NativeBFloat16Storage output = bf16Storage(6);
        try {
            write(condition, (byte) 0, (byte) 1);
            write(ifTrue,
                    TensorDTypeOps.toBFloat16Bits(1.0f),
                    TensorDTypeOps.toBFloat16Bits(2.0f),
                    TensorDTypeOps.toBFloat16Bits(3.0f),
                    TensorDTypeOps.toBFloat16Bits(4.0f),
                    TensorDTypeOps.toBFloat16Bits(5.0f),
                    TensorDTypeOps.toBFloat16Bits(6.0f));
            write(ifFalse,
                    TensorDTypeOps.toBFloat16Bits(-1.0f),
                    TensorDTypeOps.toBFloat16Bits(-2.0f));

            NativeSegmentStridedKernels.runWhere(
                    new CpuWhereKernel(),
                    view(condition, DataType.BOOL, new int[]{3, 2}, new int[]{0, 1}, 0),
                    view(ifTrue, DataType.BFLOAT16, new int[]{3, 2}, new int[]{2, 1}, 0),
                    view(ifFalse, DataType.BFLOAT16, new int[]{3, 2}, new int[]{0, 1}, 0),
                    view(output, DataType.BFLOAT16, new int[]{3, 2}, new int[]{2, 1}, 0)
            );

            assertArrayEquals(new short[]{
                    TensorDTypeOps.toBFloat16Bits(-1.0f),
                    TensorDTypeOps.toBFloat16Bits(2.0f),
                    TensorDTypeOps.toBFloat16Bits(-1.0f),
                    TensorDTypeOps.toBFloat16Bits(4.0f),
                    TensorDTypeOps.toBFloat16Bits(-1.0f),
                    TensorDTypeOps.toBFloat16Bits(6.0f)
            }, readBits(output, 6));
        } finally {
            condition.close();
            ifTrue.close();
            ifFalse.close();
            output.close();
        }
    }

    @Test
    void runsF64WhereWithZeroStrideConditionAndTransposedBranches() {
        byte[] condition = new byte[]{0, 1};
        NativeFloat64Storage ifTrue = f64Storage(6);
        NativeFloat64Storage ifFalse = f64Storage(6);
        NativeFloat64Storage output = f64Storage(6);
        try {
            write(ifTrue, 1.0d, 2.0d, 3.0d, 4.0d, 5.0d, 6.0d);
            write(ifFalse, -1.0d, -2.0d, -3.0d, -4.0d, -5.0d, -6.0d);

            NativeSegmentStridedKernels.runWhere(
                    new CpuWhereKernel(),
                    condition,
                    cpuArrayView(DataType.BOOL, new int[]{3, 2}, new int[]{0, 1}, 0),
                    view(ifTrue, DataType.FLOAT64, new int[]{3, 2}, new int[]{1, 3}, 0),
                    view(ifFalse, DataType.FLOAT64, new int[]{3, 2}, new int[]{1, 3}, 0),
                    view(output, DataType.FLOAT64, new int[]{3, 2}, new int[]{2, 1}, 0)
            );

            assertArrayEquals(new double[]{-1.0d, 4.0d, -2.0d, 5.0d, -3.0d, 6.0d}, read(output, 6), 0.0d);
        } finally {
            ifTrue.close();
            ifFalse.close();
            output.close();
        }
    }

    @Test
    void runsF32AllReductionFromGeneralStridedRead() {
        NativeFloat32Storage input = f32Storage(8);
        NativeFloat32Storage sumOut = f32Storage(1);
        NativeFloat32Storage meanOut = f32Storage(1);
        try {
            write(input, 1f, 2f, 99f, 3f, 4f, 99f, 5f, 6f);
            NativeSegmentView inputView = view(input, DataType.FLOAT32, new int[]{2, 2}, new int[]{3, 1}, 0);

            NativeSegmentStridedKernels.runReduction(
                    operations.Operation.OpType.SUM,
                    inputView,
                    view(sumOut, DataType.FLOAT32, new int[]{1}, new int[]{1}, 0),
                    -1
            );
            NativeSegmentStridedKernels.runReduction(
                    operations.Operation.OpType.MEAN,
                    inputView,
                    view(meanOut, DataType.FLOAT32, new int[]{1}, new int[]{1}, 0),
                    -1
            );

            assertArrayEquals(new float[]{10f}, read(sumOut, 1), 0.0f);
            assertArrayEquals(new float[]{2.5f}, read(meanOut, 1), 0.0f);
        } finally {
            input.close();
            sumOut.close();
            meanOut.close();
        }
    }

    @Test
    void runsF64AxisReductionFromTransposeRead() {
        NativeFloat64Storage input = f64Storage(6);
        NativeFloat64Storage sumOut = f64Storage(3);
        NativeFloat64Storage maxOut = f64Storage(3);
        try {
            write(input, 1.0d, 2.0d, 3.0d, 4.0d, 5.0d, 6.0d);

            NativeSegmentStridedKernels.runReduction(
                    operations.Operation.OpType.SUM,
                    view(input, DataType.FLOAT64, new int[]{3, 2}, new int[]{1, 3}, 0),
                    view(sumOut, DataType.FLOAT64, new int[]{3}, new int[]{1}, 0),
                    1
            );
            NativeSegmentStridedKernels.runReduction(
                    operations.Operation.OpType.REDUCE_MAX,
                    view(input, DataType.FLOAT64, new int[]{3, 2}, new int[]{1, 3}, 0),
                    view(maxOut, DataType.FLOAT64, new int[]{3}, new int[]{1}, 0),
                    1
            );

            assertArrayEquals(new double[]{5.0d, 7.0d, 9.0d}, read(sumOut, 3), 0.0d);
            assertArrayEquals(new double[]{4.0d, 5.0d, 6.0d}, read(maxOut, 3), 0.0d);
        } finally {
            input.close();
            sumOut.close();
            maxOut.close();
        }
    }

    @Test
    void runsF32ReduceMinFromGeneralStridedRead() {
        NativeFloat32Storage input = f32Storage(8);
        NativeFloat32Storage output = f32Storage(1);
        try {
            write(input, 4f, -2f, 99f, 3f, 8f, 99f, -5f, 6f);

            NativeSegmentStridedKernels.runReduction(
                    operations.Operation.OpType.REDUCE_MIN,
                    view(input, DataType.FLOAT32, new int[]{2, 2}, new int[]{3, 1}, 3),
                    view(output, DataType.FLOAT32, new int[]{1}, new int[]{1}, 0),
                    -1
            );

            assertArrayEquals(new float[]{-5f}, read(output, 1), 0.0f);
        } finally {
            input.close();
            output.close();
        }
    }

    @Test
    void runsF32AxisMeanWithKeepDimsOutputShape() {
        NativeFloat32Storage input = f32Storage(6);
        NativeFloat32Storage output = f32Storage(3);
        try {
            write(input, 1f, 2f, 3f, 4f, 5f, 6f);

            NativeSegmentStridedKernels.runReduction(
                    operations.Operation.OpType.MEAN,
                    view(input, DataType.FLOAT32, new int[]{2, 3}, new int[]{3, 1}, 0),
                    view(output, DataType.FLOAT32, new int[]{1, 3}, new int[]{3, 1}, 0),
                    0
            );

            assertArrayEquals(new float[]{2.5f, 3.5f, 4.5f}, read(output, 3), 0.0f);
        } finally {
            input.close();
            output.close();
        }
    }

    @Test
    void runsBF16MeanWithF32PromotedAccumulate() {
        NativeBFloat16Storage input = bf16Storage(6);
        NativeBFloat16Storage output = bf16Storage(2);
        try {
            writeBF16(input, 1.0f, 2.0f, 3.0f, 5.0f, 7.0f, 9.0f);

            NativeSegmentStridedKernels.runReduction(
                    operations.Operation.OpType.MEAN,
                    view(input, DataType.BFLOAT16, new int[]{2, 3}, new int[]{3, 1}, 0),
                    view(output, DataType.BFLOAT16, new int[]{2}, new int[]{1}, 0),
                    1
            );

            assertArrayEquals(new float[]{2.0f, 7.0f}, readBF16AsF32(output, 2), 0.0f);
        } finally {
            input.close();
            output.close();
        }
    }

    @Test
    void runsBoolAllAnyReductionsFromNativeMaskViews() {
        NativeBoolStorage input = boolStorage(6);
        NativeBoolStorage allOut = boolStorage(3);
        NativeBoolStorage anyOut = boolStorage(1);
        try {
            write(input, (byte) 1, (byte) 0, (byte) 1, (byte) 1, (byte) 0, (byte) 0);

            NativeSegmentStridedKernels.runReduction(
                    new reduceAll(1).opType(),
                    view(input, DataType.BOOL, new int[]{3, 2}, new int[]{2, 1}, 0),
                    view(allOut, DataType.BOOL, new int[]{3}, new int[]{1}, 0),
                    1
            );
            assertArrayEquals(new byte[]{0, 1, 0}, readBool(allOut, 3));

            NativeSegmentStridedKernels.runReduction(
                    new reduceAny(-1).opType(),
                    view(input, DataType.BOOL, new int[]{3, 2}, new int[]{2, 1}, 0),
                    view(anyOut, DataType.BOOL, new int[]{1}, new int[]{1}, 0),
                    -1
            );
            assertArrayEquals(new byte[]{1}, readBool(anyOut, 1));
        } finally {
            input.close();
            allOut.close();
            anyOut.close();
        }
    }

    @Test
    void dispatchesParallelUnaryForEligibleF32StridedInput() {
        NativeFloat32Storage input = f32Storage(8);
        NativeFloat32Storage output = f32Storage(4);
        try {
            write(input, -1f, 2f, 99f, -3f, 4f, 99f, -5f, 6f);

            NativeSegmentDispatchResult result = NativeSegmentStridedKernels.runUnaryWithDispatch(
                    new abs(),
                    view(input, DataType.FLOAT32, new int[]{2, 2}, new int[]{3, 1}, 0),
                    view(output, DataType.FLOAT32, new int[]{2, 2}, new int[]{2, 1}, 0),
                    false,
                    false,
                    NativeSegmentDispatchConfig.parallel(2, 1, 1)
            );

            assertEquals(NativeSegmentKernelFamily.SEGMENT_PARALLEL, result.family());
            assertTrue(result.parallel());
            assertEquals(4, result.logicalSize());
            assertEquals(4, result.chunks());
            assertArrayEquals(new float[]{1f, 2f, 3f, 4f}, read(output, 4), 0.0f);
        } finally {
            input.close();
            output.close();
        }
    }

    @Test
    void dispatchesParallelBinaryForEligibleF64BroadcastInput() {
        NativeFloat64Storage left = f64Storage(6);
        NativeFloat64Storage right = f64Storage(2);
        NativeFloat64Storage output = f64Storage(6);
        try {
            write(left, 1.0d, 2.0d, 3.0d, 4.0d, 5.0d, 6.0d);
            write(right, 10.0d, 100.0d);

            NativeSegmentDispatchResult result = NativeSegmentStridedKernels.runBinaryWithDispatch(
                    new mul(),
                    view(left, DataType.FLOAT64, new int[]{3, 2}, new int[]{1, 3}, 0),
                    view(right, DataType.FLOAT64, new int[]{3, 2}, new int[]{0, 1}, 0),
                    view(output, DataType.FLOAT64, new int[]{3, 2}, new int[]{2, 1}, 0),
                    NativeSegmentDispatchConfig.parallel(3, 2, 1)
            );

            assertEquals(NativeSegmentKernelFamily.SEGMENT_PARALLEL, result.family());
            assertTrue(result.parallel());
            assertEquals(3, result.chunks());
            assertArrayEquals(new double[]{10.0d, 400.0d, 20.0d, 500.0d, 30.0d, 600.0d}, read(output, 6), 0.0d);
        } finally {
            left.close();
            right.close();
            output.close();
        }
    }

    @Test
    void dispatchesParallelReductionForEligibleF32AxisMean() {
        NativeFloat32Storage input = f32Storage(6);
        NativeFloat32Storage output = f32Storage(3);
        try {
            write(input, 1f, 2f, 3f, 4f, 5f, 6f);

            NativeSegmentDispatchResult result = NativeSegmentStridedKernels.runReductionWithDispatch(
                    operations.Operation.OpType.MEAN,
                    view(input, DataType.FLOAT32, new int[]{2, 3}, new int[]{3, 1}, 0),
                    view(output, DataType.FLOAT32, new int[]{1, 3}, new int[]{3, 1}, 0),
                    0,
                    NativeSegmentDispatchConfig.parallel(2, 1, 1)
            );

            assertEquals(NativeSegmentKernelFamily.SEGMENT_PARALLEL, result.family());
            assertTrue(result.parallel());
            assertEquals(3, result.logicalSize());
            assertEquals(3, result.chunks());
            assertArrayEquals(new float[]{2.5f, 3.5f, 4.5f}, read(output, 3), 0.0f);
        } finally {
            input.close();
            output.close();
        }
    }

    @Test
    void runsFusedF32BiasAddReluInSingleSegmentKernel() {
        NativeFloat32Storage activations = f32Storage(6);
        NativeFloat32Storage bias = f32Storage(3);
        NativeFloat32Storage output = f32Storage(6);
        try {
            write(activations, -20f, 2f, 3f, 4f, -50f, 6f);
            write(bias, 10f, 20f, -30f);

            NativeSegmentDispatchResult result = NativeSegmentStridedKernels.runFusedBinaryUnary(
                    new add(),
                    new relu(),
                    view(activations, DataType.FLOAT32, new int[]{2, 3}, new int[]{3, 1}, 0),
                    view(bias, DataType.FLOAT32, new int[]{2, 3}, new int[]{0, 1}, 0),
                    view(output, DataType.FLOAT32, new int[]{2, 3}, new int[]{3, 1}, 0),
                    false,
                    false
            );

            assertEquals(NativeSegmentKernelFamily.SEGMENT_FUSED, result.family());
            assertEquals("segment-fused-binary-unary", result.reason());
            assertArrayEquals(new float[]{0f, 22f, 0f, 14f, 0f, 0f}, read(output, 6), 0.0f);
        } finally {
            activations.close();
            bias.close();
            output.close();
        }
    }

    @Test
    void runsFusedF64MultiplyThenNegateWithTransposeRead() {
        NativeFloat64Storage left = f64Storage(6);
        NativeFloat64Storage right = f64Storage(2);
        NativeFloat64Storage output = f64Storage(6);
        try {
            write(left, 1.0d, 2.0d, 3.0d, 4.0d, 5.0d, 6.0d);
            write(right, 10.0d, 100.0d);

            NativeSegmentDispatchResult result = NativeSegmentStridedKernels.runFusedBinaryUnary(
                    new mul(),
                    new neg(),
                    view(left, DataType.FLOAT64, new int[]{3, 2}, new int[]{1, 3}, 0),
                    view(right, DataType.FLOAT64, new int[]{3, 2}, new int[]{0, 1}, 0),
                    view(output, DataType.FLOAT64, new int[]{3, 2}, new int[]{2, 1}, 0),
                    false,
                    false
            );

            assertEquals(NativeSegmentKernelFamily.SEGMENT_FUSED, result.family());
            assertArrayEquals(new double[]{-10.0d, -400.0d, -20.0d, -500.0d, -30.0d, -600.0d}, read(output, 6), 0.0d);
        } finally {
            left.close();
            right.close();
            output.close();
        }
    }

    @Test
    void dispatchFallsBackToScalarWhenBelowParallelThresholdOrUnsupportedDtype() {
        NativeFloat32Storage f32Input = f32Storage(4);
        NativeFloat32Storage f32Output = f32Storage(4);
        NativeBFloat16Storage bf16Input = bf16Storage(4);
        NativeBFloat16Storage bf16Output = bf16Storage(4);
        try {
            write(f32Input, -1f, -2f, 3f, 4f);
            write(bf16Input,
                    TensorDTypeOps.toBFloat16Bits(-1.0f),
                    TensorDTypeOps.toBFloat16Bits(-2.0f),
                    TensorDTypeOps.toBFloat16Bits(3.0f),
                    TensorDTypeOps.toBFloat16Bits(4.0f));

            NativeSegmentDispatchResult threshold = NativeSegmentStridedKernels.runUnaryWithDispatch(
                    new abs(),
                    view(f32Input, DataType.FLOAT32, new int[]{4}, new int[]{1}, 0),
                    view(f32Output, DataType.FLOAT32, new int[]{4}, new int[]{1}, 0),
                    false,
                    false,
                    NativeSegmentDispatchConfig.parallel(2, 2, 8)
            );
            assertEquals(NativeSegmentKernelFamily.SEGMENT_DENSE_SCALAR, threshold.family());
            assertTrue(threshold.reason().contains("below-parallel-threshold"));
            assertArrayEquals(new float[]{1f, 2f, 3f, 4f}, read(f32Output, 4), 0.0f);

            NativeSegmentDispatchResult unsupported = NativeSegmentStridedKernels.runUnaryWithDispatch(
                    new abs(),
                    view(bf16Input, DataType.BFLOAT16, new int[]{4}, new int[]{1}, 0),
                    view(bf16Output, DataType.BFLOAT16, new int[]{4}, new int[]{1}, 0),
                    false,
                    false,
                    NativeSegmentDispatchConfig.parallel(2, 1, 1)
            );
            assertEquals(NativeSegmentKernelFamily.SEGMENT_DENSE_SCALAR, unsupported.family());
            assertTrue(unsupported.reason().contains("unsupported-parallel-family"));
            assertArrayEquals(new short[]{
                    TensorDTypeOps.toBFloat16Bits(1.0f),
                    TensorDTypeOps.toBFloat16Bits(2.0f),
                    TensorDTypeOps.toBFloat16Bits(3.0f),
                    TensorDTypeOps.toBFloat16Bits(4.0f)
            }, readBits(bf16Output, 4));
        } finally {
            f32Input.close();
            f32Output.close();
            bf16Input.close();
            bf16Output.close();
        }
    }

    @Test
    void validatesDenseOutputAndUnarySupport() {
        NativeFloat32Storage input = f32Storage(4);
        NativeFloat32Storage stridedOutput = f32Storage(6);
        NativeFloat32Storage denseOutput = f32Storage(4);
        NativeFloat64Storage wrongDTypeOutput = f64Storage(4);
        try {
            NativeSegmentView inView = view(input, DataType.FLOAT32, new int[]{2, 2}, new int[]{2, 1}, 0);

            IllegalArgumentException stridedWrite = assertThrows(IllegalArgumentException.class, () ->
                    NativeSegmentStridedKernels.runUnary(
                            new relu(),
                            inView,
                            view(stridedOutput, DataType.FLOAT32, new int[]{2, 2}, new int[]{3, 1}, 0),
                            false,
                            false
                    )
            );
            assertTrue(stridedWrite.getMessage().contains("dense contiguous output"));

            assertThrows(IllegalArgumentException.class, () ->
                    NativeSegmentStridedKernels.copyRaw(inView, view(wrongDTypeOutput, DataType.FLOAT64, new int[]{2, 2}, new int[]{2, 1}, 0))
            );

            assertThrows(UnsupportedOperationException.class, () ->
                    NativeSegmentStridedKernels.runUnary(
                            new operations.elementwise.unary.erf(),
                            inView,
                            view(denseOutput, DataType.FLOAT32, new int[]{2, 2}, new int[]{2, 1}, 0),
                            false,
                            false
                    )
            );
        } finally {
            input.close();
            stridedOutput.close();
            denseOutput.close();
            wrongDTypeOutput.close();
        }
    }

    @Test
    void validatesBinaryAndWhereContracts() {
        NativeFloat32Storage left = f32Storage(4);
        NativeFloat32Storage right = f32Storage(3);
        NativeFloat32Storage output = f32Storage(4);
        try {
            NativeSegmentView leftView = view(left, DataType.FLOAT32, new int[]{2, 2}, new int[]{2, 1}, 0);
            NativeSegmentView outView = view(output, DataType.FLOAT32, new int[]{2, 2}, new int[]{2, 1}, 0);

            assertThrows(IllegalArgumentException.class, () ->
                    NativeSegmentStridedKernels.runBinary(
                            new add(),
                            leftView,
                            view(right, DataType.FLOAT32, new int[]{3}, new int[]{1}, 0),
                            outView
                    )
            );

            assertThrows(IllegalArgumentException.class, () ->
                    NativeSegmentStridedKernels.runWhere(
                            new CpuWhereKernel(),
                            new byte[]{1, 0, 1, 0},
                            cpuArrayView(DataType.BOOL, new int[]{2, 2}, new int[]{2, 1}, 0),
                            leftView,
                            leftView,
                            view(output, DataType.FLOAT32, new int[]{2, 2}, new int[]{3, 1}, 0)
                    )
            );

            assertThrows(IllegalArgumentException.class, () ->
                    NativeSegmentStridedKernels.runWhere(
                            new CpuWhereKernel(),
                            new byte[]{1, 0},
                            cpuArrayView(DataType.BOOL, new int[]{2, 2}, new int[]{2, 1}, 0),
                            leftView,
                            leftView,
                            outView
                    )
            );
        } finally {
            left.close();
            right.close();
            output.close();
        }
    }

    @Test
    void validatesReductionContracts() {
        NativeFloat32Storage input = f32Storage(4);
        NativeFloat32Storage output = f32Storage(2);
        NativeBFloat16Storage bf16 = bf16Storage(4);
        try {
            NativeSegmentView inputView = view(input, DataType.FLOAT32, new int[]{2, 2}, new int[]{2, 1}, 0);

            assertThrows(IllegalArgumentException.class, () ->
                    NativeSegmentStridedKernels.runReduction(
                            operations.Operation.OpType.SUM,
                            inputView,
                            view(output, DataType.FLOAT32, new int[]{2}, new int[]{2}, 0),
                            1
                    )
            );

            assertThrows(IllegalArgumentException.class, () ->
                    NativeSegmentStridedKernels.runReduction(
                            operations.Operation.OpType.SUM,
                            inputView,
                            view(output, DataType.FLOAT32, new int[]{2}, new int[]{1}, 0),
                            2
                    )
            );

            assertThrows(UnsupportedOperationException.class, () ->
                    NativeSegmentStridedKernels.runReduction(
                            operations.Operation.OpType.REDUCE_MIN,
                            view(bf16, DataType.BFLOAT16, new int[]{2, 2}, new int[]{2, 1}, 0),
                            view(bf16, DataType.BFLOAT16, new int[]{1}, new int[]{1}, 0),
                            -1
                    )
            );
        } finally {
            input.close();
            output.close();
            bf16.close();
        }
    }

    private NativeFloat32Storage f32Storage(int elements) {
        return (NativeFloat32Storage) storageFactory.allocate(DataType.FLOAT32, elements, "test-f32");
    }

    private NativeFloat64Storage f64Storage(int elements) {
        return (NativeFloat64Storage) storageFactory.allocate(DataType.FLOAT64, elements, "test-f64");
    }

    private NativeBFloat16Storage bf16Storage(int elements) {
        return (NativeBFloat16Storage) storageFactory.allocate(DataType.BFLOAT16, elements, "test-bf16");
    }

    private NativeBoolStorage boolStorage(int elements) {
        return (NativeBoolStorage) storageFactory.allocate(DataType.BOOL, elements, "test-bool");
    }

    private static NativeSegmentView view(
            NativeTensorStorage storage,
            DataType dataType,
            int[] shape,
            int[] strides,
            int storageOffset
    ) {
        return NativeSegmentView.from(
                TensorPhysicalView.of(1, dataType, shape, strides, storageOffset, NativeCpuStorageFamily.CPU_NATIVE),
                storage
        );
    }

    private static TensorPhysicalView cpuArrayView(DataType dataType, int[] shape, int[] strides, int storageOffset) {
        return TensorPhysicalView.of(2, dataType, shape, strides, storageOffset, NativeCpuStorageFamily.CPU_ARRAY);
    }

    private static void write(NativeFloat32Storage storage, float... values) {
        for (int i = 0; i < values.length; i++) {
            storage.setFloat32At(i, values[i]);
        }
    }

    private static void write(NativeFloat64Storage storage, double... values) {
        for (int i = 0; i < values.length; i++) {
            storage.setFloat64At(i, values[i]);
        }
    }

    private static void write(NativeBFloat16Storage storage, short... values) {
        for (int i = 0; i < values.length; i++) {
            storage.setBFloat16BitsAt(i, values[i]);
        }
    }

    private static void writeBF16(NativeBFloat16Storage storage, float... values) {
        for (int i = 0; i < values.length; i++) {
            storage.setBFloat16BitsAt(i, TensorDTypeOps.toBFloat16Bits(values[i]));
        }
    }

    private static void write(NativeBoolStorage storage, byte... values) {
        for (int i = 0; i < values.length; i++) {
            storage.setBoolAt(i, values[i]);
        }
    }

    private static float[] read(NativeFloat32Storage storage, int elements) {
        float[] out = new float[elements];
        for (int i = 0; i < elements; i++) {
            out[i] = storage.getFloat32At(i);
        }
        return out;
    }

    private static double[] read(NativeFloat64Storage storage, int elements) {
        double[] out = new double[elements];
        for (int i = 0; i < elements; i++) {
            out[i] = storage.getFloat64At(i);
        }
        return out;
    }

    private static short[] readBits(NativeBFloat16Storage storage, int elements) {
        short[] out = new short[elements];
        for (int i = 0; i < elements; i++) {
            out[i] = storage.getBFloat16BitsAt(i);
        }
        return out;
    }

    private static float[] readBF16AsF32(NativeBFloat16Storage storage, int elements) {
        float[] out = new float[elements];
        for (int i = 0; i < elements; i++) {
            out[i] = TensorDTypeOps.fromBFloat16Bits(storage.getBFloat16BitsAt(i));
        }
        return out;
    }

    private static byte[] readBool(NativeBoolStorage storage, int elements) {
        byte[] out = new byte[elements];
        for (int i = 0; i < elements; i++) {
            out[i] = storage.getBoolAt(i);
        }
        return out;
    }
}

package backend.kernels.cpu.elementwise.strided;

import backend.kernels.cpu.elementwise.unary.support.CpuPowSupport;
import operations.Operation;
import operations.elementwise.unary.clampMax;
import operations.elementwise.unary.clampMin;
import operations.elementwise.unary.mulScalar;
import operations.elementwise.unary.pow;
import utils.FastExp;

final class StridedRank2Loops {
    private StridedRank2Loops() {
    }

    static boolean tryForwardF64(
            Operation op,
            double[] a,
            double[] b,
            int[] aStrides,
            int[] bStrides,
            int aBaseOffset,
            int bBaseOffset,
            double[] out,
            int rows,
            int cols,
            int[] outStrides,
            int outBaseOffset,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        if (out == null || outStrides == null || outStrides.length != 2) {
            return false;
        }
        return switch (op.opType()) {
            case ADD, SUB, MUL, DIV, MIN, MAX ->
                    tryBinaryF64(op, a, b, aStrides, bStrides, aBaseOffset, bBaseOffset, out, rows, cols, outStrides, outBaseOffset);
            case NEG, INV, LOG, EXP, FAST_EXP, TANH, FAST_TANH, POW, SQRT, ABS, MUL_SCALAR, RELU, CLAMP_MIN, CLAMP_MAX, SIGMOID ->
                    tryUnaryF64(op, a, aStrides, aBaseOffset, out, rows, cols, outStrides, outBaseOffset, useFastExpApprox, useFastTanhApprox);
            default -> false;
        };
    }

    static boolean tryForwardF32(
            Operation op,
            float[] a,
            float[] b,
            int[] aStrides,
            int[] bStrides,
            int aBaseOffset,
            int bBaseOffset,
            float[] out,
            int rows,
            int cols,
            int[] outStrides,
            int outBaseOffset,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        if (out == null || outStrides == null || outStrides.length != 2) {
            return false;
        }
        return switch (op.opType()) {
            case ADD, SUB, MUL, DIV, MIN, MAX ->
                    tryBinaryF32(op, a, b, aStrides, bStrides, aBaseOffset, bBaseOffset, out, rows, cols, outStrides, outBaseOffset);
            case NEG, INV, LOG, EXP, FAST_EXP, TANH, FAST_TANH, POW, SQRT, ABS, MUL_SCALAR, RELU, CLAMP_MIN, CLAMP_MAX, SIGMOID ->
                    tryUnaryF32(op, a, aStrides, aBaseOffset, out, rows, cols, outStrides, outBaseOffset, useFastExpApprox, useFastTanhApprox);
            default -> false;
        };
    }

    private static boolean tryBinaryF64(
            Operation op,
            double[] a,
            double[] b,
            int[] aStrides,
            int[] bStrides,
            int aBaseOffset,
            int bBaseOffset,
            double[] out,
            int rows,
            int cols,
            int[] outStrides,
            int outBaseOffset
    ) {
        if (a == null || b == null || aStrides == null || bStrides == null || aStrides.length != 2 || bStrides.length != 2) {
            return false;
        }

        int outRowStride = outStrides[0];
        int outColStride = outStrides[1];
        int aRowStride = aStrides[0];
        int aColStride = aStrides[1];
        int bRowStride = bStrides[0];
        int bColStride = bStrides[1];
        StridedElementWiseSemantics.BinaryKind kind = StridedElementWiseSemantics.resolveBinaryKind(op);

        if (outColStride == 1 && aColStride == 1 && bColStride == 0) {
            for (int row = 0; row < rows; row++) {
                int outRowBase = outBaseOffset + row * outRowStride;
                int aRowBase = aBaseOffset + row * aRowStride;
                double right = b[bBaseOffset + row * bRowStride];
                StridedVectorSupport.binaryBroadcastRightF64(kind, a, aRowBase, right, out, outRowBase, cols);
            }
            return true;
        }

        if (outColStride == 1 && aColStride == 0 && bColStride == 1) {
            for (int row = 0; row < rows; row++) {
                int outRowBase = outBaseOffset + row * outRowStride;
                int bRowBase = bBaseOffset + row * bRowStride;
                double left = a[aBaseOffset + row * aRowStride];
                StridedVectorSupport.binaryBroadcastLeftF64(kind, left, b, bRowBase, out, outRowBase, cols);
            }
            return true;
        }

        for (int row = 0; row < rows; row++) {
            int outRowBase = outBaseOffset + row * outRowStride;
            int aRowBase = aBaseOffset + row * aRowStride;
            int bRowBase = bBaseOffset + row * bRowStride;
            for (int col = 0; col < cols; col++) {
                out[outRowBase + col * outColStride] = switch (kind) {
                    case ADD -> a[aRowBase + col * aColStride] + b[bRowBase + col * bColStride];
                    case SUB -> a[aRowBase + col * aColStride] - b[bRowBase + col * bColStride];
                    case MUL -> a[aRowBase + col * aColStride] * b[bRowBase + col * bColStride];
                    case DIV -> a[aRowBase + col * aColStride] / b[bRowBase + col * bColStride];
                    case MIN -> Math.min(a[aRowBase + col * aColStride], b[bRowBase + col * bColStride]);
                    case MAX -> Math.max(a[aRowBase + col * aColStride], b[bRowBase + col * bColStride]);
                };
            }
        }
        return true;
    }

    private static boolean tryBinaryF32(
            Operation op,
            float[] a,
            float[] b,
            int[] aStrides,
            int[] bStrides,
            int aBaseOffset,
            int bBaseOffset,
            float[] out,
            int rows,
            int cols,
            int[] outStrides,
            int outBaseOffset
    ) {
        if (a == null || b == null || aStrides == null || bStrides == null || aStrides.length != 2 || bStrides.length != 2) {
            return false;
        }

        int outRowStride = outStrides[0];
        int outColStride = outStrides[1];
        int aRowStride = aStrides[0];
        int aColStride = aStrides[1];
        int bRowStride = bStrides[0];
        int bColStride = bStrides[1];
        StridedElementWiseSemantics.BinaryKind kind = StridedElementWiseSemantics.resolveBinaryKind(op);

        if (outColStride == 1 && aColStride == 1 && bColStride == 0) {
            for (int row = 0; row < rows; row++) {
                int outRowBase = outBaseOffset + row * outRowStride;
                int aRowBase = aBaseOffset + row * aRowStride;
                float right = b[bBaseOffset + row * bRowStride];
                StridedVectorSupport.binaryBroadcastRightF32(kind, a, aRowBase, right, out, outRowBase, cols);
            }
            return true;
        }

        if (outColStride == 1 && aColStride == 0 && bColStride == 1) {
            for (int row = 0; row < rows; row++) {
                int outRowBase = outBaseOffset + row * outRowStride;
                int bRowBase = bBaseOffset + row * bRowStride;
                float left = a[aBaseOffset + row * aRowStride];
                StridedVectorSupport.binaryBroadcastLeftF32(kind, left, b, bRowBase, out, outRowBase, cols);
            }
            return true;
        }

        for (int row = 0; row < rows; row++) {
            int outRowBase = outBaseOffset + row * outRowStride;
            int aRowBase = aBaseOffset + row * aRowStride;
            int bRowBase = bBaseOffset + row * bRowStride;
            for (int col = 0; col < cols; col++) {
                out[outRowBase + col * outColStride] = switch (kind) {
                    case ADD -> a[aRowBase + col * aColStride] + b[bRowBase + col * bColStride];
                    case SUB -> a[aRowBase + col * aColStride] - b[bRowBase + col * bColStride];
                    case MUL -> a[aRowBase + col * aColStride] * b[bRowBase + col * bColStride];
                    case DIV -> a[aRowBase + col * aColStride] / b[bRowBase + col * bColStride];
                    case MIN -> Math.min(a[aRowBase + col * aColStride], b[bRowBase + col * bColStride]);
                    case MAX -> Math.max(a[aRowBase + col * aColStride], b[bRowBase + col * bColStride]);
                };
            }
        }
        return true;
    }

    private static boolean tryUnaryF64(
            Operation op,
            double[] a,
            int[] aStrides,
            int aBaseOffset,
            double[] out,
            int rows,
            int cols,
            int[] outStrides,
            int outBaseOffset,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        if (a == null || aStrides == null || aStrides.length != 2) {
            return false;
        }

        int outRowStride = outStrides[0];
        int outColStride = outStrides[1];
        int aRowStride = aStrides[0];
        int aColStride = aStrides[1];

        switch (op.opType()) {
            case NEG -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) StridedVectorSupport.fillRowF64(out, outBaseOffset + row * outRowStride, cols, -a[aBaseOffset + row * aRowStride]);
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = -a[aRowBase + col * aColStride];
                    }
                }
            }
            case INV -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) StridedVectorSupport.fillRowF64(out, outBaseOffset + row * outRowStride, cols, 1.0 / a[aBaseOffset + row * aRowStride]);
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = 1.0 / a[aRowBase + col * aColStride];
                    }
                }
            }
            case LOG -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) StridedVectorSupport.fillRowF64(out, outBaseOffset + row * outRowStride, cols, Math.log(a[aBaseOffset + row * aRowStride]));
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = Math.log(a[aRowBase + col * aColStride]);
                    }
                }
            }
            case EXP -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) {
                        double value = a[aBaseOffset + row * aRowStride];
                        StridedVectorSupport.fillRowF64(out, outBaseOffset + row * outRowStride, cols, useFastExpApprox ? FastExp.fastExpF64(value) : Math.exp(value));
                    }
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) {
                            double value = a[aRowBase + col * aColStride];
                            out[outRowBase + col * outColStride] = useFastExpApprox ? FastExp.fastExpF64(value) : Math.exp(value);
                        }
                    }
                }
            }
            case FAST_EXP -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) StridedVectorSupport.fillRowF64(out, outBaseOffset + row * outRowStride, cols, FastExp.fastExpF64(a[aBaseOffset + row * aRowStride]));
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = FastExp.fastExpF64(a[aRowBase + col * aColStride]);
                    }
                }
            }
            case TANH -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) {
                        double value = a[aBaseOffset + row * aRowStride];
                        StridedVectorSupport.fillRowF64(out, outBaseOffset + row * outRowStride, cols, useFastTanhApprox ? FastExp.fastTanhF64(value) : Math.tanh(value));
                    }
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) {
                            double value = a[aRowBase + col * aColStride];
                            out[outRowBase + col * outColStride] = useFastTanhApprox ? FastExp.fastTanhF64(value) : Math.tanh(value);
                        }
                    }
                }
            }
            case FAST_TANH -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) StridedVectorSupport.fillRowF64(out, outBaseOffset + row * outRowStride, cols, FastExp.fastTanhF64(a[aBaseOffset + row * aRowStride]));
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = FastExp.fastTanhF64(a[aRowBase + col * aColStride]);
                    }
                }
            }
            case SQRT -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) StridedVectorSupport.fillRowF64(out, outBaseOffset + row * outRowStride, cols, Math.sqrt(a[aBaseOffset + row * aRowStride]));
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = Math.sqrt(a[aRowBase + col * aColStride]);
                    }
                }
            }
            case ABS -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) StridedVectorSupport.fillRowF64(out, outBaseOffset + row * outRowStride, cols, Math.abs(a[aBaseOffset + row * aRowStride]));
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = Math.abs(a[aRowBase + col * aColStride]);
                    }
                }
            }
            case MUL_SCALAR -> {
                double scalar = ((mulScalar) op).getScalar();
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) StridedVectorSupport.fillRowF64(out, outBaseOffset + row * outRowStride, cols, a[aBaseOffset + row * aRowStride] * scalar);
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = a[aRowBase + col * aColStride] * scalar;
                    }
                }
            }
            case RELU -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) StridedVectorSupport.fillRowF64(out, outBaseOffset + row * outRowStride, cols, Math.max(0.0, a[aBaseOffset + row * aRowStride]));
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = Math.max(0.0, a[aRowBase + col * aColStride]);
                    }
                }
            }
            case CLAMP_MIN -> {
                double minValue = ((clampMin) op).getMinValue();
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) StridedVectorSupport.fillRowF64(out, outBaseOffset + row * outRowStride, cols, Math.max(minValue, a[aBaseOffset + row * aRowStride]));
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = Math.max(minValue, a[aRowBase + col * aColStride]);
                    }
                }
            }
            case CLAMP_MAX -> {
                double maxValue = ((clampMax) op).getMaxValue();
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) StridedVectorSupport.fillRowF64(out, outBaseOffset + row * outRowStride, cols, Math.min(maxValue, a[aBaseOffset + row * aRowStride]));
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = Math.min(maxValue, a[aRowBase + col * aColStride]);
                    }
                }
            }
            case SIGMOID -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) {
                        double value = a[aBaseOffset + row * aRowStride];
                        StridedVectorSupport.fillRowF64(out, outBaseOffset + row * outRowStride, cols, 1.0 / (1.0 + Math.exp(-value)));
                    }
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) {
                            double value = a[aRowBase + col * aColStride];
                            out[outRowBase + col * outColStride] = 1.0 / (1.0 + Math.exp(-value));
                        }
                    }
                }
            }
            case POW -> {
                double exponent = ((pow) op).getExponent();
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) {
                        double value = a[aBaseOffset + row * aRowStride];
                        StridedVectorSupport.fillRowF64(out, outBaseOffset + row * outRowStride, cols, CpuPowSupport.applyF64(value, exponent));
                    }
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) {
                            double value = a[aRowBase + col * aColStride];
                            out[outRowBase + col * outColStride] = CpuPowSupport.applyF64(value, exponent);
                        }
                    }
                }
            }
            default -> throw new UnsupportedOperationException("Unsupported rank-2 F64 unary op: " + op.opType());
        }
        return true;
    }

    private static boolean tryUnaryF32(
            Operation op,
            float[] a,
            int[] aStrides,
            int aBaseOffset,
            float[] out,
            int rows,
            int cols,
            int[] outStrides,
            int outBaseOffset,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        if (a == null || aStrides == null || aStrides.length != 2) {
            return false;
        }

        int outRowStride = outStrides[0];
        int outColStride = outStrides[1];
        int aRowStride = aStrides[0];
        int aColStride = aStrides[1];

        switch (op.opType()) {
            case NEG -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) StridedVectorSupport.fillRowF32(out, outBaseOffset + row * outRowStride, cols, -a[aBaseOffset + row * aRowStride]);
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = -a[aRowBase + col * aColStride];
                    }
                }
            }
            case INV -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) StridedVectorSupport.fillRowF32(out, outBaseOffset + row * outRowStride, cols, 1.0f / a[aBaseOffset + row * aRowStride]);
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = 1.0f / a[aRowBase + col * aColStride];
                    }
                }
            }
            case LOG -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) StridedVectorSupport.fillRowF32(out, outBaseOffset + row * outRowStride, cols, (float) Math.log(a[aBaseOffset + row * aRowStride]));
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = (float) Math.log(a[aRowBase + col * aColStride]);
                    }
                }
            }
            case EXP -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) {
                        float value = a[aBaseOffset + row * aRowStride];
                        StridedVectorSupport.fillRowF32(out, outBaseOffset + row * outRowStride, cols, useFastExpApprox ? FastExp.fastExpF32(value) : (float) Math.exp(value));
                    }
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) {
                            float value = a[aRowBase + col * aColStride];
                            out[outRowBase + col * outColStride] = useFastExpApprox ? FastExp.fastExpF32(value) : (float) Math.exp(value);
                        }
                    }
                }
            }
            case FAST_EXP -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) StridedVectorSupport.fillRowF32(out, outBaseOffset + row * outRowStride, cols, FastExp.fastExpF32(a[aBaseOffset + row * aRowStride]));
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = FastExp.fastExpF32(a[aRowBase + col * aColStride]);
                    }
                }
            }
            case TANH -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) {
                        float value = a[aBaseOffset + row * aRowStride];
                        StridedVectorSupport.fillRowF32(out, outBaseOffset + row * outRowStride, cols, useFastTanhApprox ? FastExp.fastTanhF32(value) : (float) Math.tanh(value));
                    }
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) {
                            float value = a[aRowBase + col * aColStride];
                            out[outRowBase + col * outColStride] = useFastTanhApprox ? FastExp.fastTanhF32(value) : (float) Math.tanh(value);
                        }
                    }
                }
            }
            case FAST_TANH -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) StridedVectorSupport.fillRowF32(out, outBaseOffset + row * outRowStride, cols, FastExp.fastTanhF32(a[aBaseOffset + row * aRowStride]));
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = FastExp.fastTanhF32(a[aRowBase + col * aColStride]);
                    }
                }
            }
            case SQRT -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) StridedVectorSupport.fillRowF32(out, outBaseOffset + row * outRowStride, cols, (float) Math.sqrt(a[aBaseOffset + row * aRowStride]));
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = (float) Math.sqrt(a[aRowBase + col * aColStride]);
                    }
                }
            }
            case ABS -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) StridedVectorSupport.fillRowF32(out, outBaseOffset + row * outRowStride, cols, Math.abs(a[aBaseOffset + row * aRowStride]));
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = Math.abs(a[aRowBase + col * aColStride]);
                    }
                }
            }
            case MUL_SCALAR -> {
                float scalar = ((mulScalar) op).getScalarF32();
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) StridedVectorSupport.fillRowF32(out, outBaseOffset + row * outRowStride, cols, a[aBaseOffset + row * aRowStride] * scalar);
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = a[aRowBase + col * aColStride] * scalar;
                    }
                }
            }
            case RELU -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) StridedVectorSupport.fillRowF32(out, outBaseOffset + row * outRowStride, cols, Math.max(0.0f, a[aBaseOffset + row * aRowStride]));
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = Math.max(0.0f, a[aRowBase + col * aColStride]);
                    }
                }
            }
            case CLAMP_MIN -> {
                float minValue = ((clampMin) op).getMinValueF32();
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) StridedVectorSupport.fillRowF32(out, outBaseOffset + row * outRowStride, cols, Math.max(minValue, a[aBaseOffset + row * aRowStride]));
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = Math.max(minValue, a[aRowBase + col * aColStride]);
                    }
                }
            }
            case CLAMP_MAX -> {
                float maxValue = ((clampMax) op).getMaxValueF32();
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) StridedVectorSupport.fillRowF32(out, outBaseOffset + row * outRowStride, cols, Math.min(maxValue, a[aBaseOffset + row * aRowStride]));
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = Math.min(maxValue, a[aRowBase + col * aColStride]);
                    }
                }
            }
            case SIGMOID -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) {
                        float value = a[aBaseOffset + row * aRowStride];
                        StridedVectorSupport.fillRowF32(out, outBaseOffset + row * outRowStride, cols, (float) (1.0 / (1.0 + Math.exp(-value))));
                    }
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) {
                            float value = a[aRowBase + col * aColStride];
                            out[outRowBase + col * outColStride] = (float) (1.0 / (1.0 + Math.exp(-value)));
                        }
                    }
                }
            }
            case POW -> {
                float exponent = ((pow) op).getExponentF32();
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) {
                        float value = a[aBaseOffset + row * aRowStride];
                        StridedVectorSupport.fillRowF32(out, outBaseOffset + row * outRowStride, cols, CpuPowSupport.applyF32(value, exponent));
                    }
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) {
                            float value = a[aRowBase + col * aColStride];
                            out[outRowBase + col * outColStride] = CpuPowSupport.applyF32(value, exponent);
                        }
                    }
                }
            }
            default -> throw new UnsupportedOperationException("Unsupported rank-2 F32 unary op: " + op.opType());
        }
        return true;
    }
}

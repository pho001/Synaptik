package backend.kernels.cpu.elementwise.strided;

import backend.kernels.cpu.CpuDTypeOps;
import backend.kernels.cpu.elementwise.unary.support.CpuPowSupport;
import operations.Operation;
import operations.elementwise.unary.clampMax;
import operations.elementwise.unary.clampMin;
import operations.elementwise.unary.mulScalar;
import operations.elementwise.unary.pow;
import utils.FastExp;

final class StridedElementWiseSemantics {
    private static final BoolUnaryOp BOOL_NOT = value -> value == 0 ? (byte) 1 : (byte) 0;

    private StridedElementWiseSemantics() {
    }

    enum BinaryKind {
        ADD, SUB, MUL, DIV, MIN, MAX
    }

    @FunctionalInterface
    interface F64UnaryOp {
        double apply(double value);
    }

    @FunctionalInterface
    interface F32UnaryOp {
        float apply(float value);
    }

    @FunctionalInterface
    interface BF16UnaryOp {
        short apply(short valueBits);
    }

    @FunctionalInterface
    interface F64BinaryOp {
        double apply(double left, double right);
    }

    @FunctionalInterface
    interface F32BinaryOp {
        float apply(float left, float right);
    }

    @FunctionalInterface
    interface BF16BinaryOp {
        short apply(short leftBits, short rightBits);
    }

    @FunctionalInterface
    interface F64CompareOp {
        byte apply(double left, double right);
    }

    @FunctionalInterface
    interface F32CompareOp {
        byte apply(float left, float right);
    }

    @FunctionalInterface
    interface BF16CompareOp {
        byte apply(short leftBits, short rightBits);
    }

    @FunctionalInterface
    interface BoolUnaryOp {
        byte apply(byte value);
    }

    @FunctionalInterface
    interface BoolBinaryOp {
        byte apply(byte left, byte right);
    }

    static boolean supports(Operation op) {
        if (op == null) {
            return false;
        }
        return switch (op.opType()) {
            case ADD, SUB, MUL, DIV, MIN, MAX, GT, GE, LT, LE, EQ, NE, WHERE,
                    LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT,
                    NEG, INV, LOG, EXP, FAST_EXP, TANH, FAST_TANH, POW, SQRT, ABS, MUL_SCALAR, RELU, CLAMP_MIN, CLAMP_MAX, SIGMOID -> true;
            default -> false;
        };
    }

    static BinaryKind resolveBinaryKind(Operation op) {
        if (op == null) {
            return null;
        }
        return switch (op.opType()) {
            case ADD -> BinaryKind.ADD;
            case SUB -> BinaryKind.SUB;
            case MUL -> BinaryKind.MUL;
            case DIV -> BinaryKind.DIV;
            case MIN -> BinaryKind.MIN;
            case MAX -> BinaryKind.MAX;
            default -> null;
        };
    }

    static F64BinaryOp resolveF64Binary(Operation op) {
        BinaryKind kind = resolveBinaryKind(op);
        if (kind == null) {
            return null;
        }
        return switch (kind) {
            case ADD -> (left, right) -> left + right;
            case SUB -> (left, right) -> left - right;
            case MUL -> (left, right) -> left * right;
            case DIV -> (left, right) -> left / right;
            case MIN -> Math::min;
            case MAX -> Math::max;
        };
    }

    static F32BinaryOp resolveF32Binary(Operation op) {
        BinaryKind kind = resolveBinaryKind(op);
        if (kind == null) {
            return null;
        }
        return switch (kind) {
            case ADD -> (left, right) -> left + right;
            case SUB -> (left, right) -> left - right;
            case MUL -> (left, right) -> left * right;
            case DIV -> (left, right) -> left / right;
            case MIN -> Math::min;
            case MAX -> Math::max;
        };
    }

    static BF16BinaryOp resolveBF16Binary(Operation op) {
        BinaryKind kind = resolveBinaryKind(op);
        if (kind == null) {
            return null;
        }
        return switch (kind) {
            case ADD -> (left, right) -> CpuDTypeOps.toBFloat16Bits(
                    CpuDTypeOps.fromBFloat16Bits(left) + CpuDTypeOps.fromBFloat16Bits(right)
            );
            case SUB -> (left, right) -> CpuDTypeOps.toBFloat16Bits(
                    CpuDTypeOps.fromBFloat16Bits(left) - CpuDTypeOps.fromBFloat16Bits(right)
            );
            case MUL -> (left, right) -> CpuDTypeOps.toBFloat16Bits(
                    CpuDTypeOps.fromBFloat16Bits(left) * CpuDTypeOps.fromBFloat16Bits(right)
            );
            case DIV -> (left, right) -> CpuDTypeOps.toBFloat16Bits(
                    CpuDTypeOps.fromBFloat16Bits(left) / CpuDTypeOps.fromBFloat16Bits(right)
            );
            case MIN -> (left, right) -> CpuDTypeOps.toBFloat16Bits(
                    Math.min(CpuDTypeOps.fromBFloat16Bits(left), CpuDTypeOps.fromBFloat16Bits(right))
            );
            case MAX -> (left, right) -> CpuDTypeOps.toBFloat16Bits(
                    Math.max(CpuDTypeOps.fromBFloat16Bits(left), CpuDTypeOps.fromBFloat16Bits(right))
            );
        };
    }

    static F64UnaryOp resolveF64Unary(Operation op, boolean useFastExpApprox, boolean useFastTanhApprox) {
        if (op == null) {
            return null;
        }
        return switch (op.opType()) {
            case NEG -> value -> -value;
            case INV -> value -> 1.0 / value;
            case LOG -> Math::log;
            case EXP -> useFastExpApprox ? FastExp::fastExpF64 : Math::exp;
            case FAST_EXP -> FastExp::fastExpF64;
            case TANH -> useFastTanhApprox ? FastExp::fastTanhF64 : Math::tanh;
            case FAST_TANH -> FastExp::fastTanhF64;
            case SQRT -> Math::sqrt;
            case ABS -> Math::abs;
            case RELU -> value -> Math.max(0.0, value);
            case SIGMOID -> value -> 1.0 / (1.0 + Math.exp(-value));
            case CLAMP_MIN -> {
                double minValue = ((clampMin) op).getMinValue();
                yield value -> Math.max(minValue, value);
            }
            case CLAMP_MAX -> {
                double maxValue = ((clampMax) op).getMaxValue();
                yield value -> Math.min(maxValue, value);
            }
            case MUL_SCALAR -> {
                double scalar = ((mulScalar) op).getScalar();
                yield value -> value * scalar;
            }
            case POW -> {
                double exponent = ((pow) op).getExponent();
                yield value -> CpuPowSupport.applyF64(value, exponent);
            }
            default -> null;
        };
    }

    static F32UnaryOp resolveF32Unary(Operation op, boolean useFastExpApprox, boolean useFastTanhApprox) {
        if (op == null) {
            return null;
        }
        return switch (op.opType()) {
            case NEG -> value -> -value;
            case INV -> value -> 1.0f / value;
            case LOG -> value -> (float) Math.log(value);
            case EXP -> useFastExpApprox ? FastExp::fastExpF32 : value -> (float) Math.exp(value);
            case FAST_EXP -> FastExp::fastExpF32;
            case TANH -> useFastTanhApprox ? FastExp::fastTanhF32 : value -> (float) Math.tanh(value);
            case FAST_TANH -> FastExp::fastTanhF32;
            case SQRT -> value -> (float) Math.sqrt(value);
            case ABS -> Math::abs;
            case RELU -> value -> Math.max(0.0f, value);
            case SIGMOID -> value -> (float) (1.0 / (1.0 + Math.exp(-value)));
            case CLAMP_MIN -> {
                float minValue = ((clampMin) op).getMinValueF32();
                yield value -> Math.max(minValue, value);
            }
            case CLAMP_MAX -> {
                float maxValue = ((clampMax) op).getMaxValueF32();
                yield value -> Math.min(maxValue, value);
            }
            case MUL_SCALAR -> {
                float scalar = ((mulScalar) op).getScalarF32();
                yield value -> value * scalar;
            }
            case POW -> {
                float exponent = ((pow) op).getExponentF32();
                yield value -> CpuPowSupport.applyF32(value, exponent);
            }
            default -> null;
        };
    }

    static BF16UnaryOp resolveBF16Unary(Operation op, boolean useFastExpApprox, boolean useFastTanhApprox) {
        if (op == null) {
            return null;
        }
        return switch (op.opType()) {
            case NEG -> valueBits -> CpuDTypeOps.toBFloat16Bits(-CpuDTypeOps.fromBFloat16Bits(valueBits));
            case INV -> valueBits -> CpuDTypeOps.toBFloat16Bits(1.0f / CpuDTypeOps.fromBFloat16Bits(valueBits));
            case LOG -> valueBits -> CpuDTypeOps.toBFloat16Bits((float) Math.log(CpuDTypeOps.fromBFloat16Bits(valueBits)));
            case EXP -> valueBits -> {
                float value = CpuDTypeOps.fromBFloat16Bits(valueBits);
                return CpuDTypeOps.toBFloat16Bits(useFastExpApprox ? FastExp.fastExpF32(value) : (float) Math.exp(value));
            };
            case FAST_EXP -> valueBits -> CpuDTypeOps.toBFloat16Bits(FastExp.fastExpF32(CpuDTypeOps.fromBFloat16Bits(valueBits)));
            case TANH -> valueBits -> {
                float value = CpuDTypeOps.fromBFloat16Bits(valueBits);
                return CpuDTypeOps.toBFloat16Bits(useFastTanhApprox ? FastExp.fastTanhF32(value) : (float) Math.tanh(value));
            };
            case FAST_TANH -> valueBits -> CpuDTypeOps.toBFloat16Bits(FastExp.fastTanhF32(CpuDTypeOps.fromBFloat16Bits(valueBits)));
            case SQRT -> valueBits -> CpuDTypeOps.toBFloat16Bits((float) Math.sqrt(CpuDTypeOps.fromBFloat16Bits(valueBits)));
            case ABS -> valueBits -> CpuDTypeOps.toBFloat16Bits(Math.abs(CpuDTypeOps.fromBFloat16Bits(valueBits)));
            case RELU -> valueBits -> CpuDTypeOps.toBFloat16Bits(Math.max(0.0f, CpuDTypeOps.fromBFloat16Bits(valueBits)));
            case SIGMOID -> valueBits -> {
                float value = CpuDTypeOps.fromBFloat16Bits(valueBits);
                return CpuDTypeOps.toBFloat16Bits((float) (1.0 / (1.0 + Math.exp(-value))));
            };
            case CLAMP_MIN -> {
                float minValue = ((clampMin) op).getMinValueF32();
                yield valueBits -> CpuDTypeOps.toBFloat16Bits(Math.max(minValue, CpuDTypeOps.fromBFloat16Bits(valueBits)));
            }
            case CLAMP_MAX -> {
                float maxValue = ((clampMax) op).getMaxValueF32();
                yield valueBits -> CpuDTypeOps.toBFloat16Bits(Math.min(maxValue, CpuDTypeOps.fromBFloat16Bits(valueBits)));
            }
            case MUL_SCALAR -> {
                float scalar = ((mulScalar) op).getScalarF32();
                yield valueBits -> CpuDTypeOps.toBFloat16Bits(CpuDTypeOps.fromBFloat16Bits(valueBits) * scalar);
            }
            case POW -> {
                float exponent = ((pow) op).getExponentF32();
                yield valueBits -> CpuDTypeOps.toBFloat16Bits(CpuPowSupport.applyF32(CpuDTypeOps.fromBFloat16Bits(valueBits), exponent));
            }
            default -> null;
        };
    }

    static F64CompareOp resolveF64Compare(Operation op) {
        if (op == null) {
            return null;
        }
        return switch (op.opType()) {
            case GT -> (left, right) -> left > right ? (byte) 1 : (byte) 0;
            case GE -> (left, right) -> left >= right ? (byte) 1 : (byte) 0;
            case LT -> (left, right) -> left < right ? (byte) 1 : (byte) 0;
            case LE -> (left, right) -> left <= right ? (byte) 1 : (byte) 0;
            case EQ -> (left, right) -> left == right ? (byte) 1 : (byte) 0;
            case NE -> (left, right) -> left != right ? (byte) 1 : (byte) 0;
            default -> null;
        };
    }

    static F32CompareOp resolveF32Compare(Operation op) {
        if (op == null) {
            return null;
        }
        return switch (op.opType()) {
            case GT -> (left, right) -> left > right ? (byte) 1 : (byte) 0;
            case GE -> (left, right) -> left >= right ? (byte) 1 : (byte) 0;
            case LT -> (left, right) -> left < right ? (byte) 1 : (byte) 0;
            case LE -> (left, right) -> left <= right ? (byte) 1 : (byte) 0;
            case EQ -> (left, right) -> left == right ? (byte) 1 : (byte) 0;
            case NE -> (left, right) -> left != right ? (byte) 1 : (byte) 0;
            default -> null;
        };
    }

    static BF16CompareOp resolveBF16Compare(Operation op) {
        if (op == null) {
            return null;
        }
        return switch (op.opType()) {
            case GT -> (leftBits, rightBits) -> CpuDTypeOps.fromBFloat16Bits(leftBits) > CpuDTypeOps.fromBFloat16Bits(rightBits) ? (byte) 1 : (byte) 0;
            case GE -> (leftBits, rightBits) -> CpuDTypeOps.fromBFloat16Bits(leftBits) >= CpuDTypeOps.fromBFloat16Bits(rightBits) ? (byte) 1 : (byte) 0;
            case LT -> (leftBits, rightBits) -> CpuDTypeOps.fromBFloat16Bits(leftBits) < CpuDTypeOps.fromBFloat16Bits(rightBits) ? (byte) 1 : (byte) 0;
            case LE -> (leftBits, rightBits) -> CpuDTypeOps.fromBFloat16Bits(leftBits) <= CpuDTypeOps.fromBFloat16Bits(rightBits) ? (byte) 1 : (byte) 0;
            case EQ -> (leftBits, rightBits) -> CpuDTypeOps.fromBFloat16Bits(leftBits) == CpuDTypeOps.fromBFloat16Bits(rightBits) ? (byte) 1 : (byte) 0;
            case NE -> (leftBits, rightBits) -> CpuDTypeOps.fromBFloat16Bits(leftBits) != CpuDTypeOps.fromBFloat16Bits(rightBits) ? (byte) 1 : (byte) 0;
            default -> null;
        };
    }

    static BoolUnaryOp resolveBoolUnary(Operation op) {
        if (op == null || op.opType() != Operation.OpType.LOGICAL_NOT) {
            return null;
        }
        return BOOL_NOT;
    }

    static BoolUnaryOp boolNot() {
        return BOOL_NOT;
    }

    static BoolBinaryOp resolveBoolBinary(Operation op) {
        if (op == null) {
            return null;
        }
        return switch (op.opType()) {
            case LOGICAL_AND -> (left, right) -> (left != 0 && right != 0) ? (byte) 1 : (byte) 0;
            case LOGICAL_OR -> (left, right) -> (left != 0 || right != 0) ? (byte) 1 : (byte) 0;
            default -> null;
        };
    }
}

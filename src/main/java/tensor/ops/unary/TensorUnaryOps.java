package tensor.ops.unary;

import operations.Operation;
import operations.abs;
import operations.clampMax;
import operations.clampMin;
import operations.exp;
import operations.fastExp;
import operations.fastTanh;
import operations.inv;
import operations.log;
import operations.mulScalar;
import operations.neg;
import operations.pow;
import operations.relu;
import operations.sigmoid;
import operations.sqrt;
import operations.tanh;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorDataTypeUtil;
import tensor.TensorPrimitiveBuilder;

public final class TensorUnaryOps {
    private TensorUnaryOps() {
    }

    public static Tensor neg(Tensor input) {
        Operation op = new neg();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "neg", TensorDataTypeUtil.unary(input));
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            UnarySupport.accumulateGradient(input, outGrad.neg());
        });
        return out;
    }

    public static Tensor abs(Tensor input) {
        UnarySupport.requireNumeric(input, "abs");

        Operation op = new abs();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "abs", TensorDataTypeUtil.unary(input));
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }

            Tensor zero = Tensor.scalar(0.0, input.getDataType());
            Tensor sign = Tensor.where(
                    input.greaterThan(zero),
                    Tensor.onesLike(input),
                    Tensor.where(
                            input.lessThan(zero),
                            Tensor.onesLike(input).mul(-1.0),
                            Tensor.zerosLike(input)
                    )
            );
            UnarySupport.accumulateGradient(input, outGrad.mul(sign));
        });
        return out;
    }

    public static Tensor log(Tensor input) {
        Operation op = new log();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "log", TensorDataTypeUtil.unary(input));
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            UnarySupport.accumulateGradient(input, outGrad.div(input));
        });
        return out;
    }

    public static Tensor exp(Tensor input) {
        Operation op = new exp();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "exp", TensorDataTypeUtil.unary(input));
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            UnarySupport.accumulateGradient(input, outGrad.mul(out));
        });
        return out;
    }

    public static Tensor fastExp(Tensor input) {
        Operation op = new fastExp();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "fastExp", TensorDataTypeUtil.unary(input));
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            UnarySupport.accumulateGradient(input, outGrad.mul(out));
        });
        return out;
    }

    public static Tensor pow(Tensor input, double exponent) {
        boolean isF32 = input.getDataType() == DataType.FLOAT32;
        double exponentForGrad = isF32 ? (float) exponent : exponent;
        Operation op = isF32 ? new pow((float) exponent) : new pow(exponent);
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "pow", TensorDataTypeUtil.unary(input));

        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            Tensor gradForInput = outGrad
                    .mul(exponentForGrad)
                    .mul(input.pow(exponentForGrad - 1.0));
            UnarySupport.accumulateGradient(input, gradForInput);
        });

        return out;
    }

    public static Tensor mulScalar(Tensor input, double scalar) {
        boolean isF32 = input.getDataType() == DataType.FLOAT32;
        double scalarForGrad = isF32 ? (float) scalar : scalar;
        Operation op = isF32 ? new mulScalar((float) scalar) : new mulScalar(scalar);
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "* constant", TensorDataTypeUtil.unary(input));

        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            UnarySupport.accumulateGradient(input, outGrad.mul(scalarForGrad));
        });

        return out;
    }

    public static Tensor inv(Tensor input) {
        Operation op = new inv();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "inv", TensorDataTypeUtil.unary(input));

        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            UnarySupport.accumulateGradient(input, outGrad.neg().mul(out.mul(out)));
        });
        return out;
    }

    public static Tensor sqrt(Tensor input) {
        Operation op = new sqrt();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "sqrt", TensorDataTypeUtil.unary(input));

        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            UnarySupport.accumulateGradient(input, outGrad.mul(0.5).mul(out.inv()));
        });
        return out;
    }

    public static Tensor sigmoid(Tensor input) {
        Operation op = new sigmoid();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "sigmoid", TensorDataTypeUtil.unary(input));
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            UnarySupport.accumulateGradient(input, outGrad.mul(out).mul(Tensor.onesLike(out).sub(out)));
        });
        return out;
    }

    public static Tensor tanh(Tensor input) {
        Operation op = new tanh();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "tanh", TensorDataTypeUtil.unary(input));
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            UnarySupport.accumulateGradient(input, outGrad.mul(Tensor.onesLike(out).sub(out.mul(out))));
        });
        return out;
    }

    public static Tensor fastTanh(Tensor input) {
        Operation op = new fastTanh();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "fastTanh", TensorDataTypeUtil.unary(input));
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            UnarySupport.accumulateGradient(input, outGrad.mul(Tensor.onesLike(out).sub(out.mul(out))));
        });
        return out;
    }

    public static Tensor relu(Tensor input) {
        UnarySupport.requireNumeric(input, "relu");

        Operation op = new relu();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "relu", TensorDataTypeUtil.unary(input));
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }

            Tensor zero = Tensor.scalar(0.0, input.getDataType());
            Tensor gradForInput = Tensor.where(input.greaterThan(zero), outGrad, Tensor.zerosLike(outGrad));
            UnarySupport.accumulateGradient(input, gradForInput);
        });
        return out;
    }

    public static Tensor clamp(Tensor input, double minValue, double maxValue) {
        UnarySupport.requireNumeric(input, "clamp");
        if (minValue > maxValue) {
            throw new IllegalArgumentException("clamp requires minValue <= maxValue.");
        }
        return input.clampMax(maxValue).clampMin(minValue);
    }

    public static Tensor clampMin(Tensor input, double minValue) {
        UnarySupport.requireNumeric(input, "clampMin");
        boolean isF32 = input.getDataType() == DataType.FLOAT32;
        Operation op = isF32 ? new clampMin((float) minValue) : new clampMin(minValue);
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "clampMin", TensorDataTypeUtil.unary(input));
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }

            Tensor lower = Tensor.scalar(minValue, input.getDataType());
            Tensor gradForInput = Tensor.where(input.greaterOrEqual(lower), outGrad, Tensor.zerosLike(outGrad));
            UnarySupport.accumulateGradient(input, gradForInput);
        });
        return out;
    }

    public static Tensor clampMax(Tensor input, double maxValue) {
        UnarySupport.requireNumeric(input, "clampMax");
        boolean isF32 = input.getDataType() == DataType.FLOAT32;
        Operation op = isF32 ? new clampMax((float) maxValue) : new clampMax(maxValue);
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "clampMax", TensorDataTypeUtil.unary(input));
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }

            Tensor upper = Tensor.scalar(maxValue, input.getDataType());
            Tensor gradForInput = Tensor.where(input.lessOrEqual(upper), outGrad, Tensor.zerosLike(outGrad));
            UnarySupport.accumulateGradient(input, gradForInput);
        });
        return out;
    }
}

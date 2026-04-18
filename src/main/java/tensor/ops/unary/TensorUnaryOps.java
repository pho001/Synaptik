package tensor.ops.unary;

import operations.Operation;
import operations.elementwise.unary.abs;
import operations.elementwise.unary.clampMax;
import operations.elementwise.unary.clampMin;
import operations.elementwise.unary.exp;
import operations.elementwise.unary.fastExp;
import operations.elementwise.unary.fastTanh;
import operations.elementwise.unary.inv;
import operations.elementwise.unary.log;
import operations.elementwise.unary.mulScalar;
import operations.elementwise.unary.neg;
import operations.elementwise.unary.pow;
import operations.elementwise.unary.relu;
import operations.elementwise.unary.sigmoid;
import operations.elementwise.unary.sqrt;
import operations.elementwise.unary.tanh;
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
        UnarySupport.requireNumeric(input, "pow");

        boolean isF32 = input.getDataType() == DataType.FLOAT32;
        double exponentForGrad = isF32 ? (float) exponent : exponent;
        if (Double.compare(exponentForGrad, 0.0d) == 0) {
            return Tensor.onesLike(input);
        }
        if (Double.compare(exponentForGrad, 1.0d) == 0) {
            return input;
        }
        if (Double.compare(exponentForGrad, -1.0d) == 0) {
            return input.inv();
        }
        if (Double.compare(exponentForGrad, 2.0d) == 0) {
            return input.mul(input);
        }

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
        UnarySupport.requireNumeric(input, "mulScalar");

        boolean isF32 = input.getDataType() == DataType.FLOAT32;
        double scalarForGrad = isF32 ? (float) scalar : scalar;
        if (Double.compare(scalarForGrad, 0.0d) == 0) {
            return Tensor.zerosLike(input);
        }
        if (Double.compare(scalarForGrad, 1.0d) == 0) {
            return input;
        }
        if (Double.compare(scalarForGrad, -1.0d) == 0) {
            return input.neg();
        }

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
        if (input.getOperation() != null && input.getOperation().opType() == Operation.OpType.INV) {
            return input.getPrevTensors().get(0);
        }

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
        if (minValue == Double.NEGATIVE_INFINITY) {
            return input;
        }
        if (input.getOperation() instanceof clampMin inner) {
            return input.getPrevTensors().get(0).clampMin(Math.max(inner.getMinValue(), minValue));
        }
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
        if (maxValue == Double.POSITIVE_INFINITY) {
            return input;
        }
        if (input.getOperation() instanceof clampMax inner) {
            return input.getPrevTensors().get(0).clampMax(Math.min(inner.getMaxValue(), maxValue));
        }
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

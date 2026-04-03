package tensor;

import operations.Operation;
import operations.exp;
import operations.fastExp;
import operations.fastTanh;
import operations.inv;
import operations.log;
import operations.mulScalar;
import operations.neg;
import operations.pow;
import operations.sigmoid;
import operations.sqrt;
import operations.tanh;

import java.util.List;

final class TensorUnaryOps {
    private TensorUnaryOps() {}

    static Tensor neg(Tensor input) {
        Operation op = new neg();
        Tensor out = new Tensor(input.getShape(), List.of(input), op, "neg");
        out.setDataType(TensorDataTypeUtil.unary(input));
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (input.getRequiresGrad()) {
                Tensor gradForInput = outGrad.neg();
                if (input.getGradient() == null) {
                    input.setGradient(gradForInput);
                } else {
                    input.setGradient(input.getGradient().add(gradForInput));
                }
            }
        });
        return out;
    }

    static Tensor log(Tensor input) {
        Operation op = new log();
        Tensor out = new Tensor(input.getShape(), List.of(input), op, "log");
        out.setDataType(TensorDataTypeUtil.unary(input));
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (input.getRequiresGrad()) {
                Tensor gradForInput = outGrad.div(input);
                if (input.getGradient() == null) {
                    input.setGradient(gradForInput);
                } else {
                    input.setGradient(input.getGradient().add(gradForInput));
                }
            }
        });
        return out;
    }

    static Tensor exp(Tensor input) {
        Operation op = new exp();
        Tensor out = new Tensor(input.getShape(), List.of(input), op, "exp");
        out.setDataType(TensorDataTypeUtil.unary(input));
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (input.getRequiresGrad()) {
                Tensor gradForInput = outGrad.mul(out);
                if (input.getGradient() == null) {
                    input.setGradient(gradForInput);
                } else {
                    input.setGradient(input.getGradient().add(gradForInput));
                }
            }
        });
        return out;
    }

    static Tensor fastExp(Tensor input) {
        Operation op = new fastExp();
        Tensor out = new Tensor(input.getShape(), List.of(input), op, "fastExp");
        out.setDataType(TensorDataTypeUtil.unary(input));
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (input.getRequiresGrad()) {
                Tensor gradForInput = outGrad.mul(out);
                if (input.getGradient() == null) {
                    input.setGradient(gradForInput);
                } else {
                    input.setGradient(input.getGradient().add(gradForInput));
                }
            }
        });
        return out;
    }

    static Tensor pow(Tensor input, double exponent) {
        boolean isF32 = input.getDataType() == DataType.FLOAT32;
        final double exponentForGrad = isF32 ? (float) exponent : exponent;
        final Operation op = isF32 ? new pow((float) exponent) : new pow(exponent);
        Tensor out = new Tensor(input.getShape(), List.of(input), op, "pow");
        out.setDataType(TensorDataTypeUtil.unary(input));

        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (input.getRequiresGrad()) {
                Tensor gradForInput = outGrad
                        .mul(exponentForGrad)
                        .mul(input.pow(exponentForGrad - 1.0));

                if (input.getGradient() == null) {
                    input.setGradient(gradForInput);
                } else {
                    input.setGradient(input.getGradient().add(gradForInput));
                }
            }
        });

        return out;
    }

    static Tensor mulScalar(Tensor input, double scalar) {
        boolean isF32 = input.getDataType() == DataType.FLOAT32;
        final double scalarForGrad = isF32 ? (float) scalar : scalar;
        final Operation op = isF32 ? new mulScalar((float) scalar) : new mulScalar(scalar);
        Tensor out = new Tensor(input.getShape(), List.of(input), op, "* constant");
        out.setDataType(TensorDataTypeUtil.unary(input));

        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (input.getRequiresGrad()) {
                Tensor gradForInput = outGrad.mul(scalarForGrad);
                if (input.getGradient() == null) {
                    input.setGradient(gradForInput);
                } else {
                    input.setGradient(input.getGradient().add(gradForInput));
                }
            }
        });

        return out;
    }

    static Tensor inv(Tensor input) {
        Operation op = new inv();
        Tensor out = new Tensor(input.getShape(), List.of(input), op, "inv");
        out.setDataType(TensorDataTypeUtil.unary(input));

        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (input.getRequiresGrad()) {
                Tensor gradForInput = outGrad.neg().mul(out.mul(out));
                if (input.getGradient() == null) {
                    input.setGradient(gradForInput);
                } else {
                    input.setGradient(input.getGradient().add(gradForInput));
                }
            }
        });
        return out;
    }

    static Tensor sqrt(Tensor input) {
        Operation op = new sqrt();
        Tensor out = new Tensor(input.getShape(), List.of(input), op, "sqrt");
        out.setDataType(TensorDataTypeUtil.unary(input));

        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (input.getRequiresGrad()) {
                Tensor gradForInput = outGrad.mul(0.5).mul(out.inv());
                if (input.getGradient() == null) {
                    input.setGradient(gradForInput);
                } else {
                    input.setGradient(input.getGradient().add(gradForInput));
                }
            }
        });
        return out;
    }

    static Tensor sigmoid(Tensor input) {
        Operation op = new sigmoid();
        Tensor out = new Tensor(input.getShape(), List.of(input), op, "sigmoid");
        out.setDataType(TensorDataTypeUtil.unary(input));

        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (input.getRequiresGrad()) {
                Tensor gradForInput = outGrad.mul(out).mul(Tensor.onesLike(out).sub(out));
                if (input.getGradient() == null) {
                    input.setGradient(gradForInput);
                } else {
                    input.setGradient(input.getGradient().add(gradForInput));
                }
            }
        });
        return out;
    }

    static Tensor tanh(Tensor input) {
        Operation op = new tanh();
        Tensor out = new Tensor(input.getShape(), List.of(input), op, "tanh");
        out.setDataType(TensorDataTypeUtil.unary(input));

        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (input.getRequiresGrad()) {
                Tensor gradForInput = outGrad.mul(Tensor.onesLike(out).sub(out.mul(out)));
                if (input.getGradient() == null) {
                    input.setGradient(gradForInput);
                } else {
                    input.setGradient(input.getGradient().add(gradForInput));
                }
            }
        });
        return out;
    }

    static Tensor fastTanh(Tensor input) {
        Operation op = new fastTanh();
        Tensor out = new Tensor(input.getShape(), List.of(input), op, "fastTanh");
        out.setDataType(TensorDataTypeUtil.unary(input));

        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (input.getRequiresGrad()) {
                Tensor gradForInput = outGrad.mul(Tensor.onesLike(out).sub(out.mul(out)));
                if (input.getGradient() == null) {
                    input.setGradient(gradForInput);
                } else {
                    input.setGradient(input.getGradient().add(gradForInput));
                }
            }
        });
        return out;
    }

    static Tensor clamp(Tensor input, double minValue, double maxValue) {
        if (input == null) {
            throw new IllegalArgumentException("clamp input cannot be null");
        }
        if (input.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("clamp requires numeric input.");
        }
        if (minValue > maxValue) {
            throw new IllegalArgumentException("clamp requires minValue <= maxValue.");
        }

        Tensor lower = Tensor.scalar(minValue, input.getDataType());
        Tensor upper = Tensor.scalar(maxValue, input.getDataType());
        Tensor upperClamped = Tensor.where(input.greaterThan(upper), upper, input);
        return Tensor.where(upperClamped.lessThan(lower), lower, upperClamped);
    }

    static Tensor clampMin(Tensor input, double minValue) {
        if (input == null) {
            throw new IllegalArgumentException("clampMin input cannot be null");
        }
        if (input.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("clampMin requires numeric input.");
        }
        Tensor lower = Tensor.scalar(minValue, input.getDataType());
        return Tensor.where(input.lessThan(lower), lower, input);
    }

    static Tensor clampMax(Tensor input, double maxValue) {
        if (input == null) {
            throw new IllegalArgumentException("clampMax input cannot be null");
        }
        if (input.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("clampMax requires numeric input.");
        }
        Tensor upper = Tensor.scalar(maxValue, input.getDataType());
        return Tensor.where(input.greaterThan(upper), upper, input);
    }
}

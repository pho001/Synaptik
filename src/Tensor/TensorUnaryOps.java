package Tensor;

import Operations.Operation;
import Operations.exp;
import Operations.fastExp;
import Operations.fastTanh;
import Operations.inv;
import Operations.log;
import Operations.mulScalar;
import Operations.neg;
import Operations.pow;
import Operations.sigmoid;
import Operations.sqrt;
import Operations.tanh;

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
        Operation op = new pow(exponent);
        Tensor out = new Tensor(input.getShape(), List.of(input), op, "pow");
        out.setDataType(TensorDataTypeUtil.unary(input));

        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (input.getRequiresGrad()) {
                Tensor gradForInput = outGrad
                        .mul(exponent)
                        .mul(input.pow(exponent - 1.0));

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
        Operation op = new mulScalar(scalar);
        Tensor out = new Tensor(input.getShape(), List.of(input), op, "* constant");
        out.setDataType(TensorDataTypeUtil.unary(input));

        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (input.getRequiresGrad()) {
                Tensor gradForInput = outGrad.mul(scalar);
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
}

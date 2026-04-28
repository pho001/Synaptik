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
import tensor.TensorInternalAccess;
import tensor.TensorPrimitiveBuilder;

/**
 * Differentiable elementwise unary operations for floating tensors.
 *
 * <p>Public methods require non-null floating numeric inputs unless noted
 * otherwise. Methods return graph tensors and do not mutate input storage.
 * Algebraic simplifications may return the input tensor or a constant-like
 * tensor when the result is shape-preserving and exact.</p>
 */
public final class TensorUnaryOps {
    private TensorUnaryOps() {
    }

    /**
     * Negates every element.
     *
     * @param input floating tensor; must be non-null
     * @return tensor representing {@code -input}
     * @throws NullPointerException if {@code input} is null
     * @throws IllegalArgumentException if {@code input} is non-floating
     */
    public static Tensor neg(Tensor input) {
        Operation op = new neg();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "neg", TensorDataTypeUtil.unary(input));
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            UnarySupport.accumulateGradient(input, outGrad.neg());
        });
        return out;
    }

    /**
     * Computes the absolute value of every element.
     *
     * @param input floating tensor; must be non-null
     * @return shape-preserving absolute value tensor
     * @throws NullPointerException if {@code input} is null
     * @throws IllegalArgumentException if {@code input} is non-floating
     */
    public static Tensor abs(Tensor input) {
        UnarySupport.requireNumeric(input, "abs");

        Operation op = new abs();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "abs", TensorDataTypeUtil.unary(input));
        TensorInternalAccess.setBackwardFunction(out, () -> {
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

    /**
     * Computes the natural logarithm elementwise.
     *
     * @param input floating tensor; must be non-null
     * @return shape-preserving logarithm tensor
     * @throws NullPointerException if {@code input} is null
     * @throws IllegalArgumentException if {@code input} is non-floating
     */
    public static Tensor log(Tensor input) {
        Operation op = new log();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "log", TensorDataTypeUtil.unary(input));
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            UnarySupport.accumulateGradient(input, outGrad.div(input));
        });
        return out;
    }

    /**
     * Computes the exponential function elementwise.
     *
     * @param input floating tensor; must be non-null
     * @return shape-preserving exponential tensor
     * @throws NullPointerException if {@code input} is null
     * @throws IllegalArgumentException if {@code input} is non-floating
     */
    public static Tensor exp(Tensor input) {
        Operation op = new exp();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "exp", TensorDataTypeUtil.unary(input));
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            UnarySupport.accumulateGradient(input, outGrad.mul(out));
        });
        return out;
    }

    /**
     * Computes an implementation-specific fast exponential approximation.
     *
     * @param input floating tensor; must be non-null
     * @return shape-preserving approximate exponential tensor
     * @throws NullPointerException if {@code input} is null
     * @throws IllegalArgumentException if {@code input} is non-floating
     */
    public static Tensor fastExp(Tensor input) {
        Operation op = new fastExp();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "fastExp", TensorDataTypeUtil.unary(input));
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            UnarySupport.accumulateGradient(input, outGrad.mul(out));
        });
        return out;
    }

    /**
     * Raises each element to a scalar exponent.
     *
     * @param input floating tensor; must be non-null
     * @param exponent scalar exponent
     * @return shape-preserving power tensor
     * @throws NullPointerException if {@code input} is null
     * @throws IllegalArgumentException if {@code input} is non-floating
     */
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

        TensorInternalAccess.setBackwardFunction(out, () -> {
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

    /**
     * Multiplies each element by a scalar.
     *
     * @param input floating tensor; must be non-null
     * @param scalar scalar multiplier
     * @return shape-preserving scaled tensor
     * @throws NullPointerException if {@code input} is null
     * @throws IllegalArgumentException if {@code input} is non-floating
     */
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

        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            UnarySupport.accumulateGradient(input, outGrad.mul(scalarForGrad));
        });

        return out;
    }

    /**
     * Computes the reciprocal of each element.
     *
     * @param input floating tensor; must be non-null
     * @return shape-preserving reciprocal tensor
     * @throws NullPointerException if {@code input} is null
     * @throws IllegalArgumentException if {@code input} is non-floating
     */
    public static Tensor inv(Tensor input) {
        if (input.getOperation() != null && input.getOperation().opType() == Operation.OpType.INV) {
            return input.getPrevTensors().get(0);
        }

        Operation op = new inv();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "inv", TensorDataTypeUtil.unary(input));

        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            UnarySupport.accumulateGradient(input, outGrad.neg().mul(out.mul(out)));
        });
        return out;
    }

    /**
     * Computes the square root of each element.
     *
     * @param input floating tensor; must be non-null
     * @return shape-preserving square-root tensor
     * @throws NullPointerException if {@code input} is null
     * @throws IllegalArgumentException if {@code input} is non-floating
     */
    public static Tensor sqrt(Tensor input) {
        Operation op = new sqrt();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "sqrt", TensorDataTypeUtil.unary(input));

        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            UnarySupport.accumulateGradient(input, outGrad.mul(0.5).mul(out.inv()));
        });
        return out;
    }

    /**
     * Applies the logistic sigmoid elementwise.
     *
     * @param input floating tensor; must be non-null
     * @return shape-preserving sigmoid tensor
     * @throws NullPointerException if {@code input} is null
     * @throws IllegalArgumentException if {@code input} is non-floating
     */
    public static Tensor sigmoid(Tensor input) {
        Operation op = new sigmoid();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "sigmoid", TensorDataTypeUtil.unary(input));
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            UnarySupport.accumulateGradient(input, outGrad.mul(out).mul(Tensor.onesLike(out).sub(out)));
        });
        return out;
    }

    /**
     * Applies hyperbolic tangent elementwise.
     *
     * @param input floating tensor; must be non-null
     * @return shape-preserving tanh tensor
     * @throws NullPointerException if {@code input} is null
     * @throws IllegalArgumentException if {@code input} is non-floating
     */
    public static Tensor tanh(Tensor input) {
        Operation op = new tanh();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "tanh", TensorDataTypeUtil.unary(input));
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            UnarySupport.accumulateGradient(input, outGrad.mul(Tensor.onesLike(out).sub(out.mul(out))));
        });
        return out;
    }

    /**
     * Applies an implementation-specific fast tanh approximation.
     *
     * @param input floating tensor; must be non-null
     * @return shape-preserving approximate tanh tensor
     * @throws NullPointerException if {@code input} is null
     * @throws IllegalArgumentException if {@code input} is non-floating
     */
    public static Tensor fastTanh(Tensor input) {
        Operation op = new fastTanh();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "fastTanh", TensorDataTypeUtil.unary(input));
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            UnarySupport.accumulateGradient(input, outGrad.mul(Tensor.onesLike(out).sub(out.mul(out))));
        });
        return out;
    }

    /**
     * Applies rectified linear activation, returning {@code max(input, 0)}.
     *
     * @param input floating tensor; must be non-null
     * @return shape-preserving ReLU tensor
     * @throws NullPointerException if {@code input} is null
     * @throws IllegalArgumentException if {@code input} is non-floating
     */
    public static Tensor relu(Tensor input) {
        UnarySupport.requireNumeric(input, "relu");

        Operation op = new relu();
        Tensor out = TensorPrimitiveBuilder.unary(input, op, "relu", TensorDataTypeUtil.unary(input));
        TensorInternalAccess.setBackwardFunction(out, () -> {
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

    /**
     * Clamps each element to the inclusive range {@code [minValue, maxValue]}.
     *
     * @param input floating tensor; must be non-null
     * @param minValue inclusive lower bound
     * @param maxValue inclusive upper bound
     * @return shape-preserving clamped tensor
     * @throws NullPointerException if {@code input} is null
     * @throws IllegalArgumentException if input is non-floating or {@code minValue > maxValue}
     */
    public static Tensor clamp(Tensor input, double minValue, double maxValue) {
        UnarySupport.requireNumeric(input, "clamp");
        if (minValue > maxValue) {
            throw new IllegalArgumentException("clamp requires minValue <= maxValue.");
        }
        return input.clampMax(maxValue).clampMin(minValue);
    }

    /**
     * Clamps each element to be at least {@code minValue}.
     *
     * @param input floating tensor; must be non-null
     * @param minValue inclusive lower bound
     * @return shape-preserving clamped tensor, or {@code input} for negative infinity
     * @throws NullPointerException if {@code input} is null
     * @throws IllegalArgumentException if {@code input} is non-floating
     */
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
        TensorInternalAccess.setBackwardFunction(out, () -> {
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

    /**
     * Clamps each element to be at most {@code maxValue}.
     *
     * @param input floating tensor; must be non-null
     * @param maxValue inclusive upper bound
     * @return shape-preserving clamped tensor, or {@code input} for positive infinity
     * @throws NullPointerException if {@code input} is null
     * @throws IllegalArgumentException if {@code input} is non-floating
     */
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
        TensorInternalAccess.setBackwardFunction(out, () -> {
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

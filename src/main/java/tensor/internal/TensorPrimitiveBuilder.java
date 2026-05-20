package tensor.internal;

import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.autograd.GradientRule;

import operations.Operation;

import java.util.List;

public final class TensorPrimitiveBuilder {
    private TensorPrimitiveBuilder() {
    }

    public static Tensor unary(Tensor input, Operation op, String label, DataType dataType) {
        return unary(input, input.getShape(), op, label, dataType, null);
    }

    public static Tensor unary(Tensor input, Operation op, String label, DataType dataType, GradientRule backward) {
        return unary(input, input.getShape(), op, label, dataType, backward);
    }

    public static Tensor unary(Tensor input, int[] outShape, Operation op, String label, DataType dataType) {
        return unary(input, outShape, op, label, dataType, null);
    }

    public static Tensor unary(Tensor input, int[] outShape, Operation op, String label, DataType dataType, GradientRule backward) {
        return build(outShape, List.of(input), op, label, dataType, backward, false);
    }

    public static Tensor binary(Tensor first, Tensor second, int[] outShape, Operation op, String label, DataType dataType) {
        return binary(first, second, outShape, op, label, dataType, null);
    }

    public static Tensor binary(Tensor first, Tensor second, int[] outShape, Operation op, String label, DataType dataType, GradientRule backward) {
        return build(outShape, List.of(first, second), op, label, dataType, backward, false);
    }

    public static Tensor ternary(
            Tensor first,
            Tensor second,
            Tensor third,
            int[] outShape,
            Operation op,
            String label,
            DataType dataType
    ) {
        return ternary(first, second, third, outShape, op, label, dataType, null);
    }

    public static Tensor ternary(
            Tensor first,
            Tensor second,
            Tensor third,
            int[] outShape,
            Operation op,
            String label,
            DataType dataType,
            GradientRule backward
    ) {
        return build(outShape, List.of(first, second, third), op, label, dataType, backward, false);
    }

    public static Tensor nary(int[] outShape, List<Tensor> inputs, Operation op, String label, DataType dataType) {
        return nary(outShape, inputs, op, label, dataType, null);
    }

    public static Tensor nary(int[] outShape, List<Tensor> inputs, Operation op, String label, DataType dataType, GradientRule backward) {
        return build(outShape, inputs, op, label, dataType, backward, false);
    }

    public static Tensor ternaryNoGrad(
            Tensor first,
            Tensor second,
            Tensor third,
            int[] outShape,
            Operation op,
            String label,
            DataType dataType
    ) {
        return build(outShape, List.of(first, second, third), op, label, dataType, null, true);
    }

    public static Tensor naryNoGrad(int[] outShape, List<Tensor> inputs, Operation op, String label, DataType dataType) {
        return build(outShape, inputs, op, label, dataType, null, true);
    }

    public static Tensor unaryNoGrad(Tensor input, int[] outShape, Operation op, String label, DataType dataType) {
        return build(outShape, List.of(input), op, label, dataType, null, true);
    }

    public static Tensor binaryNoGrad(Tensor first, Tensor second, int[] outShape, Operation op, String label, DataType dataType) {
        return build(outShape, List.of(first, second), op, label, dataType, null, true);
    }

    public static Tensor unaryView(
            Tensor input,
            int[] outShape,
            int[] outStrides,
            int storageOffset,
            Operation op,
            String label,
            DataType dataType
    ) {
        return unaryView(input, outShape, outStrides, storageOffset, op, label, dataType, null);
    }

    public static Tensor unaryView(
            Tensor input,
            int[] outShape,
            int[] outStrides,
            int storageOffset,
            Operation op,
            String label,
            DataType dataType,
            GradientRule backward
    ) {
        Tensor out = new Tensor(outShape, outStrides, storageOffset, List.of(input), op, label, dataType);
        TensorInternalAccess.aliasRuntimeFrom(out, input);
        if (backward != null) {
            TensorInternalAccess.setGradientRule(out, backward);
        }
        return out;
    }

    private static Tensor build(
            int[] outShape,
            List<Tensor> inputs,
            Operation op,
            String label,
            DataType dataType,
            GradientRule backward,
            boolean forceNoGrad
    ) {
        Tensor out = new Tensor(outShape, inputs, op, label, dataType);
        if (forceNoGrad) {
            out.setRequiresGrad(false);
        }
        if (backward != null) {
            TensorInternalAccess.setGradientRule(out, backward);
        }
        return out;
    }
}

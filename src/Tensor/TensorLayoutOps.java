package Tensor;

import Operations.Operation;
import Operations.contiguous;
import Operations.expandDims;
import Operations.permute;
import Operations.reshape;
import Operations.squeeze;

import java.util.List;

final class TensorLayoutOps {
    private TensorLayoutOps() {}

    static Tensor contiguous(Tensor input) {
        Operation op = new contiguous();
        Tensor out = new Tensor(input.getShape(), List.of(input), op, "contiguous");
        out.setDataType(input.getDataType());
        return out;
    }

    static Tensor reshape(Tensor input, int[] requestedShape) {
        int[] newShape = TensorLayoutTransform.inferReshape(input.getShape(), requestedShape);
        Operation op = new reshape(newShape);
        Tensor out = new Tensor(newShape, List.of(input), op, "reshape");
        out.setDataType(input.getDataType());
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) return;
            if (!input.getRequiresGrad()) return;
            Tensor grad = outGrad.reshape(input.getShape());
            if (input.getGradient() == null) input.setGradient(grad);
            else input.setGradient(input.getGradient().add(grad));
        });
        return out;
    }

    static Tensor permute(Tensor input, int[] axes) {
        int rank = input.getShape().length;
        int[] normalizedAxes = TensorLayoutTransform.normalizeAxes(rank, axes);
        int[] inShape = input.getShape();
        int[] inStrides = input.getStrides();
        int[] outShape = new int[rank];
        int[] outStrides = new int[rank];
        for (int i = 0; i < rank; i++) {
            outShape[i] = inShape[normalizedAxes[i]];
            outStrides[i] = inStrides[normalizedAxes[i]];
        }

        Operation op = new permute(normalizedAxes);
        Tensor out = new Tensor(new double[input.getFlatDataSize()], outShape, outStrides, List.of(input), "permute", input.getDataType());
        out.setOperation(op);
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) return;
            if (!input.getRequiresGrad()) return;
            int[] inverse = TensorLayoutTransform.inverseAxes(normalizedAxes);
            Tensor grad = outGrad.permute(inverse);
            if (input.getGradient() == null) input.setGradient(grad);
            else input.setGradient(input.getGradient().add(grad));
        });
        return out;
    }

    static Tensor expandDims(Tensor input, int axis) {
        int rank = input.getShape().length;
        int normalizedAxis = TensorLayoutTransform.normalizeInsertAxis(axis, rank);
        int[] inShape = input.getShape();
        int[] outShape = new int[rank + 1];
        for (int i = 0, j = 0; i < outShape.length; i++) {
            if (i == normalizedAxis) outShape[i] = 1;
            else outShape[i] = inShape[j++];
        }
        Operation op = new expandDims(normalizedAxis);
        Tensor out = new Tensor(outShape, List.of(input), op, "expandDims");
        out.setDataType(input.getDataType());
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) return;
            if (!input.getRequiresGrad()) return;
            Tensor grad = outGrad.squeeze(normalizedAxis);
            if (input.getGradient() == null) input.setGradient(grad);
            else input.setGradient(input.getGradient().add(grad));
        });
        return out;
    }

    static Tensor squeeze(Tensor input, int axis) {
        int rank = input.getShape().length;
        int normalizedAxis = TensorLayoutTransform.normalizeAxis(axis, rank);
        if (input.getShape()[normalizedAxis] != 1) {
            throw new IllegalArgumentException("Cannot squeeze dimension " + normalizedAxis + " with size " + input.getShape()[normalizedAxis]);
        }
        int[] inShape = input.getShape();
        int[] outShape = new int[rank - 1];
        for (int i = 0, j = 0; i < inShape.length; i++) {
            if (i != normalizedAxis) outShape[j++] = inShape[i];
        }
        Operation op = new squeeze(normalizedAxis);
        Tensor out = new Tensor(outShape, List.of(input), op, "squeeze");
        out.setDataType(input.getDataType());
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) return;
            if (!input.getRequiresGrad()) return;
            Tensor grad = outGrad.expandDims(normalizedAxis);
            if (input.getGradient() == null) input.setGradient(grad);
            else input.setGradient(input.getGradient().add(grad));
        });
        return out;
    }
}

package tensor.ops.conv;

import tensor.DataType;
import tensor.Tensor;

final class ConvSupport {
    private ConvSupport() {
    }

    static int inferOutputSize(int inputSize, int kernelSize, int pad, int stride, int dilation, String axisName) {
        int effectiveKernel = dilation * (kernelSize - 1) + 1;
        int numerator = inputSize + 2 * pad - effectiveKernel;
        if (numerator < 0) {
            throw new IllegalArgumentException("conv2d effective kernel does not fit input " + axisName + ".");
        }
        return numerator / stride + 1;
    }

    static void validateFloatingTensor(Tensor tensor, String name) {
        if (tensor.getDataType() == DataType.BOOL || tensor.getDataType() == DataType.INT32) {
            throw new IllegalArgumentException(name + " must use a floating dtype.");
        }
    }

    static void accumulateGradient(Tensor input, Tensor gradientDelta) {
        if (input.getGradient() == null) {
            input.setGradient(gradientDelta);
        } else {
            input.setGradient(input.getGradient().add(gradientDelta));
        }
    }
}

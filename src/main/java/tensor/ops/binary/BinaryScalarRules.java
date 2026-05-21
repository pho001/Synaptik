package tensor.ops.binary;

import tensor.Tensor;

final class BinaryScalarRules {
    private BinaryScalarRules() {
    }

    static boolean isScalarConstant(Tensor tensor, double expected) {
        return tensor.getOperation() == null
                && !tensor.getRequiresGrad()
                && tensor.getFlatDataSize() == 1
                && Math.abs(tensor.scalarAsDouble() - expected) < 1e-12;
    }

    static boolean isNonZeroScalarConstant(Tensor tensor) {
        return tensor.getOperation() == null
                && !tensor.getRequiresGrad()
                && tensor.getFlatDataSize() == 1
                && Double.compare(tensor.scalarAsDouble(), 0.0d) != 0;
    }

    static Tensor minMaxElementwiseGrad(Tensor first, Tensor second, Tensor outGrad, boolean isMax, boolean forFirst) {
        Tensor tie = first.equalTo(second);
        Tensor wins = forFirst
                ? (isMax ? first.greaterThan(second) : first.lessThan(second))
                : (isMax ? second.greaterThan(first) : second.lessThan(first));
        Tensor zero = Tensor.zerosLike(outGrad);
        Tensor half = outGrad.mul(0.5d);
        return Tensor.where(tie, half, Tensor.where(wins, outGrad, zero));
    }
}

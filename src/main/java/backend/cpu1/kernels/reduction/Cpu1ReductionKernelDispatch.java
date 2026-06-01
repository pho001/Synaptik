package backend.cpu1.kernels.reduction;

import java.util.Objects;

/**
 * Resolves prepared reduction kernel ids to concrete loop runners.
 */
public final class Cpu1ReductionKernelDispatch {
    private Cpu1ReductionKernelDispatch() {
    }

    public static Cpu1ReductionKernel runnerFor(Cpu1ReductionKernelId kernelId) {
        Objects.requireNonNull(kernelId, "kernelId cannot be null");
        return switch (kernelId) {
            case SUM_F32_DENSE_SCALAR -> Cpu1SumMeanReductionLoops::sumF32DenseScalar;
            case SUM_F64_DENSE_SCALAR -> Cpu1SumMeanReductionLoops::sumF64DenseScalar;
            case SUM_BF16_DENSE_SCALAR -> Cpu1SumMeanReductionLoops::sumBf16DenseScalar;
            case MEAN_F32_DENSE_SCALAR -> Cpu1SumMeanReductionLoops::meanF32DenseScalar;
            case MEAN_F64_DENSE_SCALAR -> Cpu1SumMeanReductionLoops::meanF64DenseScalar;
            case MEAN_BF16_DENSE_SCALAR -> Cpu1SumMeanReductionLoops::meanBf16DenseScalar;
            case MIN_F32_DENSE_SCALAR -> Cpu1MinMaxProdReductionLoops::minF32DenseScalar;
            case MIN_F64_DENSE_SCALAR -> Cpu1MinMaxProdReductionLoops::minF64DenseScalar;
            case MIN_BF16_DENSE_SCALAR -> Cpu1MinMaxProdReductionLoops::minBf16DenseScalar;
            case MAX_F32_DENSE_SCALAR -> Cpu1MinMaxProdReductionLoops::maxF32DenseScalar;
            case MAX_F64_DENSE_SCALAR -> Cpu1MinMaxProdReductionLoops::maxF64DenseScalar;
            case MAX_BF16_DENSE_SCALAR -> Cpu1MinMaxProdReductionLoops::maxBf16DenseScalar;
            case PROD_F32_DENSE_SCALAR -> Cpu1MinMaxProdReductionLoops::prodF32DenseScalar;
            case PROD_F64_DENSE_SCALAR -> Cpu1MinMaxProdReductionLoops::prodF64DenseScalar;
            case PROD_BF16_DENSE_SCALAR -> Cpu1MinMaxProdReductionLoops::prodBf16DenseScalar;
            case ALL_BOOL_DENSE_SCALAR -> Cpu1BoolReductionLoops::allBoolDenseScalar;
            case ANY_BOOL_DENSE_SCALAR -> Cpu1BoolReductionLoops::anyBoolDenseScalar;
            case ARGMAX_F32_TO_I64_DENSE_SCALAR -> Cpu1ArgMaxReductionLoops::argMaxF32ToI64DenseScalar;
            case ARGMAX_F64_TO_I64_DENSE_SCALAR -> Cpu1ArgMaxReductionLoops::argMaxF64ToI64DenseScalar;
            case ARGMAX_BF16_TO_I64_DENSE_SCALAR -> Cpu1ArgMaxReductionLoops::argMaxBf16ToI64DenseScalar;
            case ARGMAX_I32_TO_I64_DENSE_SCALAR -> Cpu1ArgMaxReductionLoops::argMaxI32ToI64DenseScalar;
            case ARGMAX_I64_TO_I64_DENSE_SCALAR -> Cpu1ArgMaxReductionLoops::argMaxI64ToI64DenseScalar;
            case CUMSUM_F32_DENSE_SCALAR -> Cpu1CumSumReductionLoops::cumSumF32DenseScalar;
            case CUMSUM_F64_DENSE_SCALAR -> Cpu1CumSumReductionLoops::cumSumF64DenseScalar;
            case CUMSUM_BF16_DENSE_SCALAR -> Cpu1CumSumReductionLoops::cumSumBf16DenseScalar;
            case CUMSUM_I32_DENSE_SCALAR -> Cpu1CumSumReductionLoops::cumSumI32DenseScalar;
            case CUMSUM_I64_DENSE_SCALAR -> Cpu1CumSumReductionLoops::cumSumI64DenseScalar;
            case SOFTMAX_F32_DENSE_SCALAR -> Cpu1SoftmaxReductionLoops::softmaxF32DenseScalar;
            case SOFTMAX_F64_DENSE_SCALAR -> Cpu1SoftmaxReductionLoops::softmaxF64DenseScalar;
            case SOFTMAX_BF16_DENSE_SCALAR -> Cpu1SoftmaxReductionLoops::softmaxBf16DenseScalar;
            case LOG_SOFTMAX_F32_DENSE_SCALAR -> Cpu1SoftmaxReductionLoops::logSoftmaxF32DenseScalar;
            case LOG_SOFTMAX_F64_DENSE_SCALAR -> Cpu1SoftmaxReductionLoops::logSoftmaxF64DenseScalar;
            case LOG_SOFTMAX_BF16_DENSE_SCALAR -> Cpu1SoftmaxReductionLoops::logSoftmaxBf16DenseScalar;
        };
    }
}

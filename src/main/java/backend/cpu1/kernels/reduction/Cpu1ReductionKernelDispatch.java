package backend.cpu1.kernels.reduction;

import backend.cpu1.storage.Cpu1StorageKind;

/**
 * Resolves prepared reduction kernel ids to concrete reduction kernels.
 */
public final class Cpu1ReductionKernelDispatch {
    private Cpu1ReductionKernelDispatch() {
    }

    public static Cpu1ReductionKernel kernelFor(Cpu1ReductionKernelId kernelId, Cpu1StorageKind storageKind) {
        if (kernelId == null) {
            throw new IllegalArgumentException("kernelId cannot be null");
        }
        if (storageKind == null) {
            throw new IllegalArgumentException("storageKind cannot be null");
        }
        return switch (storageKind) {
            case JAVA_ARRAY -> arrayKernelFor(kernelId);
            case MEMORY_SEGMENT -> segmentKernelFor(kernelId);
        };
    }

    private static Cpu1ReductionKernel arrayKernelFor(Cpu1ReductionKernelId kernelId) {
        return switch (kernelId) {
            case SUM_F32_DENSE_SCALAR -> Cpu1SumMeanReductionLoops::sumF32DenseScalar;
            case SUM_F64_DENSE_SCALAR -> Cpu1SumMeanReductionLoops::sumF64DenseScalar;
            case SUM_BF16_DENSE_SCALAR -> Cpu1SumMeanReductionLoops::sumBf16DenseScalar;
            case SUM_F32_STRIDED_SCALAR -> Cpu1SumMeanReductionLoops::sumF32StridedScalar;
            case SUM_F64_STRIDED_SCALAR -> Cpu1SumMeanReductionLoops::sumF64StridedScalar;
            case MEAN_F32_DENSE_SCALAR -> Cpu1SumMeanReductionLoops::meanF32DenseScalar;
            case MEAN_F64_DENSE_SCALAR -> Cpu1SumMeanReductionLoops::meanF64DenseScalar;
            case MEAN_BF16_DENSE_SCALAR -> Cpu1SumMeanReductionLoops::meanBf16DenseScalar;
            case MEAN_F32_STRIDED_SCALAR -> Cpu1SumMeanReductionLoops::meanF32StridedScalar;
            case MEAN_F64_STRIDED_SCALAR -> Cpu1SumMeanReductionLoops::meanF64StridedScalar;
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

    private static Cpu1ReductionKernel segmentKernelFor(Cpu1ReductionKernelId kernelId) {
        return switch (kernelId) {
            case SUM_F32_DENSE_SCALAR -> Cpu1SumMeanReductionLoops::sumF32DenseScalarSegment;
            case SUM_F64_DENSE_SCALAR -> Cpu1SumMeanReductionLoops::sumF64DenseScalarSegment;
            case SUM_BF16_DENSE_SCALAR -> Cpu1SumMeanReductionLoops::sumBf16DenseScalarSegment;
            case SUM_F32_STRIDED_SCALAR -> Cpu1SumMeanReductionLoops::sumF32StridedScalarSegment;
            case SUM_F64_STRIDED_SCALAR -> Cpu1SumMeanReductionLoops::sumF64StridedScalarSegment;
            case MEAN_F32_DENSE_SCALAR -> Cpu1SumMeanReductionLoops::meanF32DenseScalarSegment;
            case MEAN_F64_DENSE_SCALAR -> Cpu1SumMeanReductionLoops::meanF64DenseScalarSegment;
            case MEAN_BF16_DENSE_SCALAR -> Cpu1SumMeanReductionLoops::meanBf16DenseScalarSegment;
            case MEAN_F32_STRIDED_SCALAR -> Cpu1SumMeanReductionLoops::meanF32StridedScalarSegment;
            case MEAN_F64_STRIDED_SCALAR -> Cpu1SumMeanReductionLoops::meanF64StridedScalarSegment;
            case MIN_F32_DENSE_SCALAR -> Cpu1MinMaxProdReductionLoops::minF32DenseScalarSegment;
            case MIN_F64_DENSE_SCALAR -> Cpu1MinMaxProdReductionLoops::minF64DenseScalarSegment;
            case MIN_BF16_DENSE_SCALAR -> Cpu1MinMaxProdReductionLoops::minBf16DenseScalarSegment;
            case MAX_F32_DENSE_SCALAR -> Cpu1MinMaxProdReductionLoops::maxF32DenseScalarSegment;
            case MAX_F64_DENSE_SCALAR -> Cpu1MinMaxProdReductionLoops::maxF64DenseScalarSegment;
            case MAX_BF16_DENSE_SCALAR -> Cpu1MinMaxProdReductionLoops::maxBf16DenseScalarSegment;
            case PROD_F32_DENSE_SCALAR -> Cpu1MinMaxProdReductionLoops::prodF32DenseScalarSegment;
            case PROD_F64_DENSE_SCALAR -> Cpu1MinMaxProdReductionLoops::prodF64DenseScalarSegment;
            case PROD_BF16_DENSE_SCALAR -> Cpu1MinMaxProdReductionLoops::prodBf16DenseScalarSegment;
            case ALL_BOOL_DENSE_SCALAR -> Cpu1BoolReductionLoops::allBoolDenseScalarSegment;
            case ANY_BOOL_DENSE_SCALAR -> Cpu1BoolReductionLoops::anyBoolDenseScalarSegment;
            case ARGMAX_F32_TO_I64_DENSE_SCALAR -> Cpu1ArgMaxReductionLoops::argMaxF32ToI64DenseScalarSegment;
            case ARGMAX_F64_TO_I64_DENSE_SCALAR -> Cpu1ArgMaxReductionLoops::argMaxF64ToI64DenseScalarSegment;
            case ARGMAX_BF16_TO_I64_DENSE_SCALAR -> Cpu1ArgMaxReductionLoops::argMaxBf16ToI64DenseScalarSegment;
            case ARGMAX_I32_TO_I64_DENSE_SCALAR -> Cpu1ArgMaxReductionLoops::argMaxI32ToI64DenseScalarSegment;
            case ARGMAX_I64_TO_I64_DENSE_SCALAR -> Cpu1ArgMaxReductionLoops::argMaxI64ToI64DenseScalarSegment;
            case CUMSUM_F32_DENSE_SCALAR -> Cpu1CumSumReductionLoops::cumSumF32DenseScalarSegment;
            case CUMSUM_F64_DENSE_SCALAR -> Cpu1CumSumReductionLoops::cumSumF64DenseScalarSegment;
            case CUMSUM_BF16_DENSE_SCALAR -> Cpu1CumSumReductionLoops::cumSumBf16DenseScalarSegment;
            case CUMSUM_I32_DENSE_SCALAR -> Cpu1CumSumReductionLoops::cumSumI32DenseScalarSegment;
            case CUMSUM_I64_DENSE_SCALAR -> Cpu1CumSumReductionLoops::cumSumI64DenseScalarSegment;
            case SOFTMAX_F32_DENSE_SCALAR -> Cpu1SoftmaxReductionLoops::softmaxF32DenseScalarSegment;
            case SOFTMAX_F64_DENSE_SCALAR -> Cpu1SoftmaxReductionLoops::softmaxF64DenseScalarSegment;
            case SOFTMAX_BF16_DENSE_SCALAR -> Cpu1SoftmaxReductionLoops::softmaxBf16DenseScalarSegment;
            case LOG_SOFTMAX_F32_DENSE_SCALAR -> Cpu1SoftmaxReductionLoops::logSoftmaxF32DenseScalarSegment;
            case LOG_SOFTMAX_F64_DENSE_SCALAR -> Cpu1SoftmaxReductionLoops::logSoftmaxF64DenseScalarSegment;
            case LOG_SOFTMAX_BF16_DENSE_SCALAR -> Cpu1SoftmaxReductionLoops::logSoftmaxBf16DenseScalarSegment;
        };
    }
}

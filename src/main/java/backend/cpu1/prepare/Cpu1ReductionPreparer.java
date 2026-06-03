package backend.cpu1.prepare;

import backend.cpu1.exec.Cpu1WorkspaceSpec;
import backend.cpu1.kernels.reduction.Cpu1ReductionKernelId;
import backend.cpu1.storage.Cpu1StorageKind;
import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptor;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import operations.Operation;
import operations.reduction.ArgMaxTiePolicy;
import operations.reduction.argMax;
import operations.reduction.cumSum;
import operations.reduction.mean;
import operations.reduction.reduceAll;
import operations.reduction.reduceAny;
import operations.reduction.reduceMax;
import operations.reduction.reduceMin;
import operations.reduction.reduceProd;
import operations.reduction.logSoftmax;
import operations.reduction.softmax;
import operations.reduction.sum;
import tensor.DataType;

import java.util.Arrays;
import java.util.Objects;

/**
 * Prepares the initial dense scalar reduction subset for cpu1.
 */
public final class Cpu1ReductionPreparer {
    public Cpu1PreparedArtifact prepare(
            CompiledNode node,
            CompiledTensorDescriptorIndex descriptorIndex,
            Cpu1PrepareConfig config
    ) {
        Objects.requireNonNull(node, "node cannot be null");
        Objects.requireNonNull(config, "config cannot be null");
        Operation operation = Objects.requireNonNull(node.operation(), "node operation cannot be null");
        Operation.OpType opType = operation.opType();
        if (!isReductionOp(opType)) {
            throw new UnsupportedOperationException("cpu1 reduction preparer does not support " + opType);
        }
        if (node.inputIds().size() != 1) {
            throw new UnsupportedOperationException("cpu1 " + opType + " expects 1 input, got "
                    + node.inputIds().size());
        }
        CompiledTensorDescriptor input = descriptorIndex == null
                ? null
                : descriptorIndex.byNodeId(node.inputIds().getFirst());
        if (input == null && opType == Operation.OpType.ARGMAX) {
            throw new UnsupportedOperationException("cpu1 ARGMAX requires descriptors to resolve input dtype.");
        }
        DataType inputDataType = input == null ? node.dataType() : input.dataType();
        if (!isSupportedDType(opType, node.dataType(), inputDataType)) {
            throw new UnsupportedOperationException("cpu1 reduction preparer does not support input dtype "
                    + inputDataType + " and output dtype " + node.dataType() + " for " + opType);
        }
        if (!isSupportedStorage(opType, node.dataType(), inputDataType, config.storageKind())) {
            throw new UnsupportedOperationException("cpu1 reduction preparer does not support storage "
                    + config.storageKind() + " for input dtype " + inputDataType + ", output dtype "
                    + node.dataType() + ", op " + opType);
        }
        if (input != null) {
            requireInputContract(opType, node, input);
        }
        int[] inputShape = input == null ? inferInputShape(node, operation) : input.shape();
        int axis = normalizedAxis(axis(operation), inputShape.length);
        boolean keepDims = keepDims(operation);
        int[] expectedOutputShape = opType == Operation.OpType.CUMSUM
                || opType == Operation.OpType.SOFTMAX
                || opType == Operation.OpType.LOG_SOFTMAX
                ? inputShape
                : reducedShape(inputShape, axis, keepDims);
        if (!Arrays.equals(expectedOutputShape, node.shape())) {
            throw new UnsupportedOperationException("cpu1 " + opType + " output shape mismatch. expected="
                    + Arrays.toString(expectedOutputShape) + ", actual=" + Arrays.toString(node.shape()));
        }
        int axisSize = inputShape[axis];
        int innerSize = product(inputShape, axis + 1, inputShape.length);
        int outerSize = product(inputShape, 0, axis);
        Cpu1PreparedReductionUnit unit = new Cpu1PreparedReductionUnit(
                node.id(),
                node.inputIds().getFirst(),
                opType,
                node.dataType(),
                config.storageKind(),
                kernelId(opType, node.dataType(), inputDataType),
                axis,
                axisSize,
                innerSize,
                outerSize,
                node.flatDataSize(),
                keepDims,
                argMaxLastIndexWins(operation),
                cumSumExclusive(operation),
                cumSumReverse(operation),
                Cpu1WorkspaceSpec.none()
        );
        return new Cpu1PreparedArtifact(unit);
    }

    public static boolean isReductionOp(Operation.OpType opType) {
        return opType == Operation.OpType.SUM
                || opType == Operation.OpType.MEAN
                || opType == Operation.OpType.REDUCE_MIN
                || opType == Operation.OpType.REDUCE_MAX
                || opType == Operation.OpType.REDUCE_PROD
                || opType == Operation.OpType.REDUCE_ALL
                || opType == Operation.OpType.REDUCE_ANY
                || opType == Operation.OpType.ARGMAX
                || opType == Operation.OpType.CUMSUM
                || opType == Operation.OpType.SOFTMAX
                || opType == Operation.OpType.LOG_SOFTMAX;
    }

    private static void requireInputContract(
            Operation.OpType opType,
            CompiledNode node,
            CompiledTensorDescriptor input
    ) {
        if (opType == Operation.OpType.ARGMAX) {
            if (node.dataType() != DataType.INT64) {
                throw new UnsupportedOperationException("cpu1 ARGMAX requires INT64 output, got "
                        + node.dataType());
            }
            if (!isArgMaxInputDType(input.dataType())) {
                throw new UnsupportedOperationException("cpu1 ARGMAX requires numeric input, got "
                        + input.dataType());
            }
        } else if (opType == Operation.OpType.CUMSUM) {
            if (!isCumSumDType(input.dataType()) || input.dataType() != node.dataType()) {
                throw new UnsupportedOperationException("cpu1 CUMSUM requires matching numeric input/output dtype. input="
                        + input.dataType() + ", output=" + node.dataType());
            }
        } else if (opType == Operation.OpType.SOFTMAX || opType == Operation.OpType.LOG_SOFTMAX) {
            if (!input.dataType().isFloating() || input.dataType() != node.dataType()) {
                throw new UnsupportedOperationException("cpu1 " + opType
                        + " requires matching floating input/output dtype. input="
                        + input.dataType() + ", output=" + node.dataType());
            }
        } else if (input.dataType() != node.dataType()) {
            throw new UnsupportedOperationException("cpu1 " + opType + " requires matching input/output dtype. input="
                    + input.dataType() + ", output=" + node.dataType());
        }
        if (!input.denseContiguousWithoutOffset()) {
            throw new UnsupportedOperationException("cpu1 initial " + opType
                    + " supports only dense contiguous input without storage offset.");
        }
        if (!node.contiguous() || node.hasStorageOffset()) {
            throw new UnsupportedOperationException("cpu1 initial " + opType
                    + " supports only dense contiguous output without storage offset.");
        }
    }

    private static int[] inferInputShape(CompiledNode node, Operation operation) {
        int axis = axis(operation);
        if (axis < 0) {
            throw new UnsupportedOperationException("cpu1 reductions require descriptors when axis is negative.");
        }
        int[] outputShape = node.shape();
        boolean keepDims = keepDims(operation);
        if (keepDims) {
            return outputShape;
        }
        throw new UnsupportedOperationException("cpu1 reductions require descriptors when keepDims=false.");
    }

    private static int axis(Operation operation) {
        if (operation instanceof sum sumOp) {
            return sumOp.getDimension();
        }
        if (operation instanceof mean meanOp) {
            return meanOp.getDimension();
        }
        if (operation instanceof reduceMin minOp) {
            return minOp.getDimension();
        }
        if (operation instanceof reduceMax maxOp) {
            return maxOp.getDimension();
        }
        if (operation instanceof reduceProd prodOp) {
            return prodOp.getDimension();
        }
        if (operation instanceof reduceAll allOp) {
            return allOp.getDimension();
        }
        if (operation instanceof reduceAny anyOp) {
            return anyOp.getDimension();
        }
        if (operation instanceof argMax argMaxOp) {
            return argMaxOp.getDimension();
        }
        if (operation instanceof cumSum cumSumOp) {
            return cumSumOp.getAxis();
        }
        if (operation instanceof softmax softmaxOp) {
            return softmaxOp.getDimension();
        }
        if (operation instanceof logSoftmax logSoftmaxOp) {
            return logSoftmaxOp.getDimension();
        }
        throw new IllegalArgumentException("cpu1 reduction operation must be SUM, MEAN, REDUCE_MIN, REDUCE_MAX, "
                + "REDUCE_PROD, REDUCE_ALL, REDUCE_ANY, ARGMAX, CUMSUM, SOFTMAX, or LOG_SOFTMAX.");
    }

    private static boolean keepDims(Operation operation) {
        if (operation instanceof sum sumOp) {
            return sumOp.keepDims();
        }
        if (operation instanceof mean meanOp) {
            return meanOp.keepDims();
        }
        if (operation instanceof reduceMin minOp) {
            return minOp.keepDims();
        }
        if (operation instanceof reduceMax maxOp) {
            return maxOp.keepDims();
        }
        if (operation instanceof reduceProd prodOp) {
            return prodOp.keepDims();
        }
        if (operation instanceof reduceAll allOp) {
            return allOp.keepDims();
        }
        if (operation instanceof reduceAny anyOp) {
            return anyOp.keepDims();
        }
        if (operation instanceof argMax argMaxOp) {
            return argMaxOp.keepDims();
        }
        if (operation instanceof cumSum) {
            return true;
        }
        if (operation instanceof softmax || operation instanceof logSoftmax) {
            return true;
        }
        throw new IllegalArgumentException("cpu1 reduction operation must be SUM, MEAN, REDUCE_MIN, REDUCE_MAX, "
                + "REDUCE_PROD, REDUCE_ALL, REDUCE_ANY, ARGMAX, CUMSUM, SOFTMAX, or LOG_SOFTMAX.");
    }

    private static int normalizedAxis(int axis, int rank) {
        if (rank <= 0) {
            throw new UnsupportedOperationException("cpu1 reductions require rank > 0 input.");
        }
        int normalized = axis < 0 ? axis + rank : axis;
        if (normalized < 0 || normalized >= rank) {
            throw new UnsupportedOperationException("cpu1 reduction axis out of bounds: axis="
                    + axis + ", rank=" + rank);
        }
        return normalized;
    }

    private static int[] reducedShape(int[] inputShape, int axis, boolean keepDims) {
        if (keepDims) {
            int[] outputShape = inputShape.clone();
            outputShape[axis] = 1;
            return outputShape;
        }
        int[] outputShape = new int[inputShape.length - 1];
        for (int inputDim = 0, outputDim = 0; inputDim < inputShape.length; inputDim++) {
            if (inputDim != axis) {
                outputShape[outputDim++] = inputShape[inputDim];
            }
        }
        return outputShape;
    }

    private static Cpu1ReductionKernelId kernelId(Operation.OpType opType, DataType outputDataType, DataType inputDataType) {
        return switch (opType) {
            case SUM -> switch (outputDataType) {
                case FLOAT32 -> Cpu1ReductionKernelId.SUM_F32_DENSE_SCALAR;
                case FLOAT64 -> Cpu1ReductionKernelId.SUM_F64_DENSE_SCALAR;
                case BFLOAT16 -> Cpu1ReductionKernelId.SUM_BF16_DENSE_SCALAR;
                case BOOL, INT32, INT64 -> throw unsupportedDType(opType, outputDataType);
            };
            case MEAN -> switch (outputDataType) {
                case FLOAT32 -> Cpu1ReductionKernelId.MEAN_F32_DENSE_SCALAR;
                case FLOAT64 -> Cpu1ReductionKernelId.MEAN_F64_DENSE_SCALAR;
                case BFLOAT16 -> Cpu1ReductionKernelId.MEAN_BF16_DENSE_SCALAR;
                case BOOL, INT32, INT64 -> throw unsupportedDType(opType, outputDataType);
            };
            case REDUCE_MIN -> switch (outputDataType) {
                case FLOAT32 -> Cpu1ReductionKernelId.MIN_F32_DENSE_SCALAR;
                case FLOAT64 -> Cpu1ReductionKernelId.MIN_F64_DENSE_SCALAR;
                case BFLOAT16 -> Cpu1ReductionKernelId.MIN_BF16_DENSE_SCALAR;
                case BOOL, INT32, INT64 -> throw unsupportedDType(opType, outputDataType);
            };
            case REDUCE_MAX -> switch (outputDataType) {
                case FLOAT32 -> Cpu1ReductionKernelId.MAX_F32_DENSE_SCALAR;
                case FLOAT64 -> Cpu1ReductionKernelId.MAX_F64_DENSE_SCALAR;
                case BFLOAT16 -> Cpu1ReductionKernelId.MAX_BF16_DENSE_SCALAR;
                case BOOL, INT32, INT64 -> throw unsupportedDType(opType, outputDataType);
            };
            case REDUCE_PROD -> switch (outputDataType) {
                case FLOAT32 -> Cpu1ReductionKernelId.PROD_F32_DENSE_SCALAR;
                case FLOAT64 -> Cpu1ReductionKernelId.PROD_F64_DENSE_SCALAR;
                case BFLOAT16 -> Cpu1ReductionKernelId.PROD_BF16_DENSE_SCALAR;
                case BOOL, INT32, INT64 -> throw unsupportedDType(opType, outputDataType);
            };
            case REDUCE_ALL -> switch (outputDataType) {
                case BOOL -> Cpu1ReductionKernelId.ALL_BOOL_DENSE_SCALAR;
                case FLOAT32, FLOAT64, BFLOAT16, INT32, INT64 -> throw unsupportedDType(opType, outputDataType);
            };
            case REDUCE_ANY -> switch (outputDataType) {
                case BOOL -> Cpu1ReductionKernelId.ANY_BOOL_DENSE_SCALAR;
                case FLOAT32, FLOAT64, BFLOAT16, INT32, INT64 -> throw unsupportedDType(opType, outputDataType);
            };
            case ARGMAX -> switch (inputDataType) {
                case FLOAT32 -> Cpu1ReductionKernelId.ARGMAX_F32_TO_I64_DENSE_SCALAR;
                case FLOAT64 -> Cpu1ReductionKernelId.ARGMAX_F64_TO_I64_DENSE_SCALAR;
                case BFLOAT16 -> Cpu1ReductionKernelId.ARGMAX_BF16_TO_I64_DENSE_SCALAR;
                case INT32 -> Cpu1ReductionKernelId.ARGMAX_I32_TO_I64_DENSE_SCALAR;
                case INT64 -> Cpu1ReductionKernelId.ARGMAX_I64_TO_I64_DENSE_SCALAR;
                case BOOL -> throw unsupportedDType(opType, inputDataType);
            };
            case CUMSUM -> switch (inputDataType) {
                case FLOAT32 -> Cpu1ReductionKernelId.CUMSUM_F32_DENSE_SCALAR;
                case FLOAT64 -> Cpu1ReductionKernelId.CUMSUM_F64_DENSE_SCALAR;
                case BFLOAT16 -> Cpu1ReductionKernelId.CUMSUM_BF16_DENSE_SCALAR;
                case INT32 -> Cpu1ReductionKernelId.CUMSUM_I32_DENSE_SCALAR;
                case INT64 -> Cpu1ReductionKernelId.CUMSUM_I64_DENSE_SCALAR;
                case BOOL -> throw unsupportedDType(opType, inputDataType);
            };
            case SOFTMAX -> switch (inputDataType) {
                case FLOAT32 -> Cpu1ReductionKernelId.SOFTMAX_F32_DENSE_SCALAR;
                case FLOAT64 -> Cpu1ReductionKernelId.SOFTMAX_F64_DENSE_SCALAR;
                case BFLOAT16 -> Cpu1ReductionKernelId.SOFTMAX_BF16_DENSE_SCALAR;
                case INT32, INT64, BOOL -> throw unsupportedDType(opType, inputDataType);
            };
            case LOG_SOFTMAX -> switch (inputDataType) {
                case FLOAT32 -> Cpu1ReductionKernelId.LOG_SOFTMAX_F32_DENSE_SCALAR;
                case FLOAT64 -> Cpu1ReductionKernelId.LOG_SOFTMAX_F64_DENSE_SCALAR;
                case BFLOAT16 -> Cpu1ReductionKernelId.LOG_SOFTMAX_BF16_DENSE_SCALAR;
                case INT32, INT64, BOOL -> throw unsupportedDType(opType, inputDataType);
            };
            default -> throw new UnsupportedOperationException("cpu1 reduction preparer does not support " + opType);
        };
    }

    private static UnsupportedOperationException unsupportedDType(Operation.OpType opType, DataType dataType) {
        return new UnsupportedOperationException("cpu1 reduction preparer does not support dtype "
                + dataType + " for " + opType);
    }

    private static boolean isSupportedDType(Operation.OpType opType, DataType outputDataType, DataType inputDataType) {
        if (opType == Operation.OpType.REDUCE_ALL || opType == Operation.OpType.REDUCE_ANY) {
            return inputDataType == DataType.BOOL && outputDataType == DataType.BOOL;
        }
        if (opType == Operation.OpType.ARGMAX) {
            return outputDataType == DataType.INT64 && isArgMaxInputDType(inputDataType);
        }
        if (opType == Operation.OpType.CUMSUM) {
            return outputDataType == inputDataType && isCumSumDType(inputDataType);
        }
        if (opType == Operation.OpType.SOFTMAX || opType == Operation.OpType.LOG_SOFTMAX) {
            return outputDataType == inputDataType && inputDataType.isFloating();
        }
        return outputDataType == inputDataType
                && (outputDataType == DataType.FLOAT32
                || outputDataType == DataType.FLOAT64
                || outputDataType == DataType.BFLOAT16);
    }

    private static boolean isSupportedStorage(
            Operation.OpType opType,
            DataType outputDataType,
            DataType inputDataType,
            Cpu1StorageKind storageKind
    ) {
        if (storageKind == Cpu1StorageKind.JAVA_ARRAY) {
            return true;
        }
        return storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                && (opType == Operation.OpType.SUM || opType == Operation.OpType.MEAN)
                && inputDataType == outputDataType
                && (outputDataType == DataType.FLOAT32 || outputDataType == DataType.FLOAT64);
    }

    private static boolean isArgMaxInputDType(DataType inputDataType) {
        return inputDataType == DataType.FLOAT32
                || inputDataType == DataType.FLOAT64
                || inputDataType == DataType.BFLOAT16
                || inputDataType == DataType.INT32
                || inputDataType == DataType.INT64;
    }

    private static boolean isCumSumDType(DataType inputDataType) {
        return inputDataType == DataType.FLOAT32
                || inputDataType == DataType.FLOAT64
                || inputDataType == DataType.BFLOAT16
                || inputDataType == DataType.INT32
                || inputDataType == DataType.INT64;
    }

    private static boolean argMaxLastIndexWins(Operation operation) {
        return operation instanceof argMax argMaxOp
                && argMaxOp.tiePolicy() == ArgMaxTiePolicy.LAST_INDEX;
    }

    private static boolean cumSumExclusive(Operation operation) {
        return operation instanceof cumSum cumSumOp && cumSumOp.isExclusive();
    }

    private static boolean cumSumReverse(Operation operation) {
        return operation instanceof cumSum cumSumOp && cumSumOp.isReverse();
    }

    private static int product(int[] shape, int startInclusive, int endExclusive) {
        int product = 1;
        for (int i = startInclusive; i < endExclusive; i++) {
            product = Math.multiplyExact(product, shape[i]);
        }
        return product;
    }

}

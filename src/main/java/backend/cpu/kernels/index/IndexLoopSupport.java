package backend.cpu.kernels.index;

import operations.index.ScatterReduction;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.TensorMetadata;
import tensor.dtype.TensorDTypeOps;

import java.util.Arrays;

final class IndexLoopSupport {
    private IndexLoopSupport() {
    }

    static int[] denseStrides(int[] shape) {
        return TensorMetadata.computeStrides(shape);
    }

    static int offsetForLogical(int logical, int[] shape, int[] dense, int[] strides, int baseOffset) {
        int rem = logical;
        int offset = baseOffset;
        for (int d = 0; d < shape.length; d++) {
            int coord = rem / dense[d];
            rem %= dense[d];
            offset += coord * strides[d];
        }
        return offset;
    }

    static IndexReader indexReader(Tensor indices) {
        return new IndexReader(indices);
    }

    static double reduce(double current, double update, ScatterReduction reduction) {
        return switch (reduction) {
            case NONE -> update;
            case ADD -> current + update;
            case MUL -> current * update;
            case MAX -> Math.max(current, update);
            case MIN -> Math.min(current, update);
        };
    }

    static int reduceInt(int current, int update, ScatterReduction reduction) {
        return switch (reduction) {
            case NONE -> update;
            case ADD -> Math.addExact(current, update);
            case MUL -> Math.multiplyExact(current, update);
            case MAX -> Math.max(current, update);
            case MIN -> Math.min(current, update);
        };
    }

    static long reduceLong(long current, long update, ScatterReduction reduction) {
        return switch (reduction) {
            case NONE -> update;
            case ADD -> Math.addExact(current, update);
            case MUL -> Math.multiplyExact(current, update);
            case MAX -> Math.max(current, update);
            case MIN -> Math.min(current, update);
        };
    }

    static short zeroBf16() {
        return TensorDTypeOps.toBFloat16Bits(0.0f);
    }

    static void fillZeroF64(Tensor tensor) {
        Arrays.fill(TensorInternalAccess.float64Data(tensor), 0.0d);
    }

    static void fillZeroF32(Tensor tensor) {
        Arrays.fill(TensorInternalAccess.float32Data(tensor), 0.0f);
    }

    static void fillZeroBF16(Tensor tensor) {
        Arrays.fill(TensorInternalAccess.bfloat16Data(tensor), zeroBf16());
    }

    static DuplicateState duplicateState(Tensor out, ScatterReduction reduction, String operationName) {
        if (reduction == ScatterReduction.NONE) {
            return new DuplicateState(new boolean[out.getFlatDataSize()], operationName);
        }
        return DuplicateState.NOOP;
    }

    static final class DuplicateState {
        static final DuplicateState NOOP = new DuplicateState(null, "scatter");

        private final boolean[] seen;
        private final String operationName;

        private DuplicateState(boolean[] seen, String operationName) {
            this.seen = seen;
            this.operationName = operationName;
        }

        void mark(int targetLogical) {
            if (seen == null) {
                return;
            }
            if (seen[targetLogical]) {
                throw new IllegalArgumentException(operationName + " NONE reduction does not allow duplicate target indices.");
            }
            seen[targetLogical] = true;
        }
    }

    static final class IndexReader {
        private final DataType dataType;
        private final int[] shape;
        private final int[] dense;
        private final int[] strides;
        private final int baseOffset;
        private final double[] f64;
        private final float[] f32;
        private final short[] bf16;
        private final int[] i32;
        private final long[] i64;

        private IndexReader(Tensor indices) {
            this.dataType = indices.getDataType();
            this.shape = indices.getShapeUnsafe();
            this.dense = denseStrides(shape);
            this.strides = indices.getStridesUnsafe();
            this.baseOffset = indices.getStorageOffsetUnsafe();
            this.f64 = dataType == DataType.FLOAT64 ? TensorInternalAccess.float64Data(indices) : null;
            this.f32 = dataType == DataType.FLOAT32 ? TensorInternalAccess.float32Data(indices) : null;
            this.bf16 = dataType == DataType.BFLOAT16 ? TensorInternalAccess.bfloat16Data(indices) : null;
            this.i32 = dataType == DataType.INT32 ? TensorInternalAccess.int32Data(indices) : null;
            this.i64 = dataType == DataType.INT64 ? TensorInternalAccess.int64Data(indices) : null;
        }

        int readAxisIndex(int logicalIndex, int axisSize) {
            long integral;
            if (dataType == DataType.INT32) {
                integral = i32[offset(logicalIndex)];
            } else if (dataType == DataType.INT64) {
                integral = i64[offset(logicalIndex)];
            } else {
                double raw = readFloating(logicalIndex);
                if (!Double.isFinite(raw)) {
                    throw new IllegalArgumentException("Gather index must be finite.");
                }
                integral = Math.round(raw);
                if (Math.abs(raw - integral) > 1e-9) {
                    throw new IllegalArgumentException("Gather index must be an integer value. got=" + raw);
                }
            }
            if (integral < 0 || integral >= axisSize) {
                throw new IllegalArgumentException("Gather index out of bounds: " + integral + " for axis size " + axisSize);
            }
            return (int) integral;
        }

        int readAxisIndexAllowNegative(int logicalIndex, int axisSize) {
            long integral;
            double rawDouble = 0.0d;
            boolean floating = false;
            if (dataType == DataType.INT32) {
                integral = i32[offset(logicalIndex)];
            } else if (dataType == DataType.INT64) {
                integral = i64[offset(logicalIndex)];
            } else {
                rawDouble = readFloating(logicalIndex);
                floating = true;
                if (!Double.isFinite(rawDouble)) {
                    throw new IllegalArgumentException("Gather index must be finite.");
                }
                integral = Math.round(rawDouble);
                if (Math.abs(rawDouble - integral) > 1e-9) {
                    throw new IllegalArgumentException("Gather index must be an integer value. got=" + rawDouble);
                }
            }
            long rawIntegral = integral;
            if (integral < 0) {
                integral += axisSize;
            }
            if (integral < 0 || integral >= axisSize) {
                String raw = floating ? Double.toString(rawDouble) : Long.toString(rawIntegral);
                throw new IllegalArgumentException("Gather index out of bounds: " + raw + " for axis size " + axisSize);
            }
            return (int) integral;
        }

        private int offset(int logicalIndex) {
            return offsetForLogical(logicalIndex, shape, dense, strides, baseOffset);
        }

        private double readFloating(int logicalIndex) {
            int offset = offset(logicalIndex);
            return switch (dataType) {
                case FLOAT64 -> f64[offset];
                case FLOAT32 -> f32[offset];
                case BFLOAT16 -> TensorDTypeOps.fromBFloat16Bits(bf16[offset]);
                case INT32, INT64, BOOL -> throw new IllegalArgumentException("Gather indices must be numeric integral values.");
            };
        }
    }
}

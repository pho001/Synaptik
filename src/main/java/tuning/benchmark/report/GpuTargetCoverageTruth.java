package tuning.benchmark.report;

import backend.ComputeBackend;
import backend.accelerator.lowering.GpuLoweringCoverageEntry;
import backend.accelerator.lowering.GpuLoweringCoverageMatrix;
import backend.accelerator.lowering.GpuLoweringCoverageStatus;
import backend.accelerator.lowering.GpuLoweringOperationFamily;
import operations.Operation;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Source-of-truth classification for target GPU coverage families.
 *
 * <p>The lowering matrix describes planner/lowering intent. This class describes whether a
 * target operation family is actually native-executable for a backend. New execution
 * work should only promote a target operation to {@link GpuTargetExecutionStatus#NATIVE_EXECUTABLE}
 * after backend execution, trace evidence, and parity tests exist.</p>
 */
public final class GpuTargetCoverageTruth {
    private static final List<Operation.OpType> TARGET_OPS = List.of(
            Operation.OpType.SUM,
            Operation.OpType.MEAN,
            Operation.OpType.REDUCE_MIN,
            Operation.OpType.REDUCE_MAX,
            Operation.OpType.LAYER_NORM,
            Operation.OpType.RMS_NORM,
            Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION,
            Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS,
            Operation.OpType.NLL_LOSS,
            Operation.OpType.CROSS_ENTROPY_LOSS,
            Operation.OpType.CROSS_ENTROPY_LOSS_INDICES,
            Operation.OpType.CROSS_ENTROPY_LOSS_INDICES_GRAD,
            Operation.OpType.CONV2D,
            Operation.OpType.MAX_POOL2D,
            Operation.OpType.AVG_POOL2D,
            Operation.OpType.GATHER,
            Operation.OpType.GATHER_ND,
            Operation.OpType.GATHER_GRAD,
            Operation.OpType.TAKE_ALONG_AXIS,
            Operation.OpType.TAKE_ALONG_AXIS_GRAD,
            Operation.OpType.SCATTER_ADD,
            Operation.OpType.GT,
            Operation.OpType.GE,
            Operation.OpType.LT,
            Operation.OpType.LE,
            Operation.OpType.EQ,
            Operation.OpType.NE,
            Operation.OpType.LOGICAL_AND,
            Operation.OpType.LOGICAL_OR,
            Operation.OpType.LOGICAL_NOT,
            Operation.OpType.REDUCE_ALL,
            Operation.OpType.REDUCE_ANY,
            Operation.OpType.SOFTMAX_GRAD,
            Operation.OpType.LOG_SOFTMAX_GRAD,
            Operation.OpType.REDUCE_MIN_GRAD,
            Operation.OpType.REDUCE_MAX_GRAD,
            Operation.OpType.MIN_GRAD,
            Operation.OpType.MAX_GRAD,
            Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_BACKWARD
    );

    private static final Map<ComputeBackend, Set<Operation.OpType>> NATIVE_EXECUTABLE_TARGETS =
            nativeExecutableTargets();

    private GpuTargetCoverageTruth() {
    }

    /**
     * Returns truth rows for all targeted operation families on one backend.
     */
    public static List<Row> rowsFor(ComputeBackend backend) {
        if (backend == null || (backend != ComputeBackend.GPU_METAL && backend != ComputeBackend.GPU_CUDA)) {
            return List.of();
        }
        ArrayList<Row> rows = new ArrayList<>(TARGET_OPS.size());
        for (Operation.OpType opType : TARGET_OPS) {
            GpuLoweringCoverageEntry entry = GpuLoweringCoverageMatrix.entryFor(backend, opType);
            rows.add(new Row(
                    backend,
                    opType,
                    entry.family(),
                    entry.status(),
                    classify(backend, entry),
                    detail(backend, entry)
            ));
        }
        return List.copyOf(rows);
    }

    /**
     * Returns truth rows for Metal and CUDA.
     */
    public static Map<ComputeBackend, List<Row>> rowsByBackend() {
        return Map.of(
                ComputeBackend.GPU_METAL, rowsFor(ComputeBackend.GPU_METAL),
                ComputeBackend.GPU_CUDA, rowsFor(ComputeBackend.GPU_CUDA)
        );
    }

    private static GpuTargetExecutionStatus classify(ComputeBackend backend, GpuLoweringCoverageEntry entry) {
        if (entry.status() == GpuLoweringCoverageStatus.SUPPORTED) {
            return NATIVE_EXECUTABLE_TARGETS.getOrDefault(backend, Set.of()).contains(entry.opType())
                    ? GpuTargetExecutionStatus.NATIVE_EXECUTABLE
                    : GpuTargetExecutionStatus.MATRIX_SUPPORTED_ONLY;
        }
        if (entry.status() == GpuLoweringCoverageStatus.FALLBACK) {
            return GpuTargetExecutionStatus.EXPLICIT_CPU_FALLBACK;
        }
        return GpuTargetExecutionStatus.UNSUPPORTED_REJECTION;
    }

    private static String detail(ComputeBackend backend, GpuLoweringCoverageEntry entry) {
        return switch (classify(backend, entry)) {
            case NATIVE_EXECUTABLE -> "backend native execution is verified for this v1.4 target";
            case MATRIX_SUPPORTED_ONLY -> "matrix row is supported, but v1.4 target native execution proof is not registered";
            case EXPLICIT_CPU_FALLBACK -> entry.reason().name() + ": " + entry.note();
            case UNSUPPORTED_REJECTION -> entry.reason().name() + ": " + entry.note();
        };
    }

    private static Map<ComputeBackend, Set<Operation.OpType>> nativeExecutableTargets() {
        EnumMap<ComputeBackend, Set<Operation.OpType>> out = new EnumMap<>(ComputeBackend.class);
        EnumSet<Operation.OpType> reductions = EnumSet.of(
                Operation.OpType.SUM,
                Operation.OpType.MEAN,
                Operation.OpType.REDUCE_MIN,
                Operation.OpType.REDUCE_MAX,
                Operation.OpType.LAYER_NORM,
                Operation.OpType.RMS_NORM
        );
        EnumSet<Operation.OpType> metalTargets = EnumSet.copyOf(reductions);
        metalTargets.add(Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION);
        metalTargets.add(Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS);
        metalTargets.add(Operation.OpType.NLL_LOSS);
        metalTargets.add(Operation.OpType.CROSS_ENTROPY_LOSS);
        metalTargets.add(Operation.OpType.CROSS_ENTROPY_LOSS_INDICES);
        metalTargets.add(Operation.OpType.CROSS_ENTROPY_LOSS_INDICES_GRAD);
        metalTargets.add(Operation.OpType.GT);
        metalTargets.add(Operation.OpType.GE);
        metalTargets.add(Operation.OpType.LT);
        metalTargets.add(Operation.OpType.LE);
        metalTargets.add(Operation.OpType.EQ);
        metalTargets.add(Operation.OpType.NE);
        metalTargets.add(Operation.OpType.LOGICAL_AND);
        metalTargets.add(Operation.OpType.LOGICAL_OR);
        metalTargets.add(Operation.OpType.LOGICAL_NOT);
        metalTargets.add(Operation.OpType.REDUCE_ALL);
        metalTargets.add(Operation.OpType.REDUCE_ANY);
        metalTargets.add(Operation.OpType.SOFTMAX_GRAD);
        metalTargets.add(Operation.OpType.LOG_SOFTMAX_GRAD);
        metalTargets.add(Operation.OpType.REDUCE_MIN_GRAD);
        metalTargets.add(Operation.OpType.REDUCE_MAX_GRAD);
        metalTargets.add(Operation.OpType.MIN_GRAD);
        metalTargets.add(Operation.OpType.MAX_GRAD);
        metalTargets.add(Operation.OpType.GATHER);
        metalTargets.add(Operation.OpType.GATHER_ND);
        metalTargets.add(Operation.OpType.GATHER_GRAD);
        metalTargets.add(Operation.OpType.TAKE_ALONG_AXIS);
        metalTargets.add(Operation.OpType.TAKE_ALONG_AXIS_GRAD);
        metalTargets.add(Operation.OpType.SCATTER_ADD);
        metalTargets.add(Operation.OpType.CONV2D);
        metalTargets.add(Operation.OpType.MAX_POOL2D);
        metalTargets.add(Operation.OpType.AVG_POOL2D);
        out.put(ComputeBackend.GPU_METAL, metalTargets);
        out.put(ComputeBackend.GPU_CUDA, EnumSet.copyOf(reductions));
        return Map.copyOf(out);
    }

    /**
     * Single target coverage truth row.
     */
    public record Row(
            ComputeBackend backend,
            Operation.OpType opType,
            GpuLoweringOperationFamily family,
            GpuLoweringCoverageStatus matrixStatus,
            GpuTargetExecutionStatus executionStatus,
            String detail
    ) {
        public Row {
            if (backend == null) {
                throw new IllegalArgumentException("backend cannot be null");
            }
            if (opType == null) {
                throw new IllegalArgumentException("opType cannot be null");
            }
            if (family == null) {
                throw new IllegalArgumentException("family cannot be null");
            }
            if (matrixStatus == null) {
                throw new IllegalArgumentException("matrixStatus cannot be null");
            }
            if (executionStatus == null) {
                throw new IllegalArgumentException("executionStatus cannot be null");
            }
            detail = detail == null ? "" : detail.strip();
        }
    }
}

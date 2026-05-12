package onnx;

import backend.ComputeBackend;
import backend.accelerator.lowering.GpuLoweringCoverageMatrix;
import backend.accelerator.lowering.GpuLoweringCoverageStatus;
import operations.Operation;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Source-of-truth coverage table for the ONNX interchange boundary.
 *
 * <p>The matrix intentionally separates ONNX import/export coverage from CPU
 * executability and native accelerator coverage. A supported ONNX mapping does
 * not imply that Metal or CUDA will execute the mapped graph natively.</p>
 */
public final class OnnxCoverageMatrix {
    private static final List<Entry> ENTRIES = buildEntries();

    private OnnxCoverageMatrix() {
    }

    public enum CoverageStatus {
        SUPPORTED,
        PARTIAL,
        UNSUPPORTED
    }

    public record Entry(
            String onnxOp,
            String synaptikMapping,
            CoverageStatus importStatus,
            CoverageStatus exportStatus,
            CoverageStatus cpuStatus,
            CoverageStatus metalStatus,
            CoverageStatus cudaStatus,
            String limitations,
            List<Operation.OpType> mappedOpTypes
    ) {
        public Entry {
            Objects.requireNonNull(onnxOp, "onnxOp cannot be null");
            Objects.requireNonNull(synaptikMapping, "synaptikMapping cannot be null");
            Objects.requireNonNull(importStatus, "importStatus cannot be null");
            Objects.requireNonNull(exportStatus, "exportStatus cannot be null");
            Objects.requireNonNull(cpuStatus, "cpuStatus cannot be null");
            Objects.requireNonNull(metalStatus, "metalStatus cannot be null");
            Objects.requireNonNull(cudaStatus, "cudaStatus cannot be null");
            limitations = limitations == null ? "" : limitations;
            mappedOpTypes = List.copyOf(mappedOpTypes == null ? List.of() : mappedOpTypes);
        }
    }

    public static List<Entry> entries() {
        return ENTRIES;
    }

    public static Entry entryFor(String onnxOp) {
        for (Entry entry : ENTRIES) {
            if (entry.onnxOp().equals(onnxOp)) {
                return entry;
            }
        }
        throw new IllegalArgumentException("No ONNX coverage row for op: " + onnxOp);
    }

    private static List<Entry> buildEntries() {
        ArrayList<Entry> out = new ArrayList<>();
        addBinary(out, "Add", Operation.OpType.ADD);
        addBinary(out, "Sub", Operation.OpType.SUB);
        addBinary(out, "Mul", Operation.OpType.MUL);
        addBinary(out, "Div", Operation.OpType.DIV);
        addBinary(out, "Min", Operation.OpType.MIN, "binary form only");
        addBinary(out, "Max", Operation.OpType.MAX, "binary form only");
        add(out, "Pow", "scalar-exponent pow", CoverageStatus.SUPPORTED, CoverageStatus.SUPPORTED,
                CoverageStatus.SUPPORTED, gpu(ComputeBackend.GPU_METAL, Operation.OpType.POW),
                gpu(ComputeBackend.GPU_CUDA, Operation.OpType.POW), "tensor exponent is unsupported", Operation.OpType.POW);
        addUnary(out, "Neg", Operation.OpType.NEG);
        addUnary(out, "Abs", Operation.OpType.ABS);
        addUnary(out, "Relu", Operation.OpType.RELU);
        addUnary(out, "Tanh", Operation.OpType.TANH);
        addUnary(out, "Sigmoid", Operation.OpType.SIGMOID);
        addUnary(out, "Exp", Operation.OpType.EXP);
        addUnary(out, "Log", Operation.OpType.LOG);
        addUnary(out, "Sqrt", Operation.OpType.SQRT);
        addBinary(out, "Equal", Operation.OpType.EQ);
        addBinary(out, "Greater", Operation.OpType.GT);
        addBinary(out, "GreaterOrEqual", Operation.OpType.GE);
        addBinary(out, "Less", Operation.OpType.LT);
        addBinary(out, "LessOrEqual", Operation.OpType.LE);
        addUnary(out, "Not", Operation.OpType.LOGICAL_NOT);
        addBinary(out, "And", Operation.OpType.LOGICAL_AND);
        addBinary(out, "Or", Operation.OpType.LOGICAL_OR);
        add(out, "Where", "where", CoverageStatus.SUPPORTED, CoverageStatus.SUPPORTED,
                CoverageStatus.SUPPORTED, gpu(ComputeBackend.GPU_METAL, Operation.OpType.WHERE),
                gpu(ComputeBackend.GPU_CUDA, Operation.OpType.WHERE), "", Operation.OpType.WHERE);
        add(out, "Identity", "pass-through", CoverageStatus.SUPPORTED, CoverageStatus.UNSUPPORTED,
                CoverageStatus.SUPPORTED, CoverageStatus.PARTIAL, CoverageStatus.PARTIAL,
                "import-only pass-through; export preserves the producer op instead", Operation.OpType.NOOP);
        add(out, "Clip", "clampMin/clampMax", CoverageStatus.SUPPORTED, CoverageStatus.SUPPORTED,
                CoverageStatus.SUPPORTED, CoverageStatus.SUPPORTED, CoverageStatus.SUPPORTED,
                "scalar bounds only", Operation.OpType.CLAMP_MIN, Operation.OpType.CLAMP_MAX);
        add(out, "Cast", "cast", CoverageStatus.SUPPORTED, CoverageStatus.SUPPORTED,
                CoverageStatus.SUPPORTED, CoverageStatus.UNSUPPORTED, CoverageStatus.UNSUPPORTED,
                "runtime INT64 is unsupported", Operation.OpType.CAST);
        add(out, "MatMul", "matmul", CoverageStatus.SUPPORTED, CoverageStatus.SUPPORTED,
                CoverageStatus.SUPPORTED, gpu(ComputeBackend.GPU_METAL, Operation.OpType.MATMUL),
                gpu(ComputeBackend.GPU_CUDA, Operation.OpType.MATMUL), "", Operation.OpType.MATMUL);
        add(out, "Gemm", "matmul plus optional bias/scale", CoverageStatus.SUPPORTED, CoverageStatus.SUPPORTED,
                CoverageStatus.SUPPORTED, CoverageStatus.SUPPORTED, CoverageStatus.SUPPORTED,
                "rank-2 transposition flags and scalar alpha/beta only", Operation.OpType.MATMUL, Operation.OpType.ADD);
        add(out, "Conv", "conv2d", CoverageStatus.SUPPORTED, CoverageStatus.SUPPORTED,
                CoverageStatus.SUPPORTED, gpu(ComputeBackend.GPU_METAL, Operation.OpType.CONV2D),
                gpu(ComputeBackend.GPU_CUDA, Operation.OpType.CONV2D),
                "rank-4 NCHW/OIHW, symmetric spatial pads, static attributes", Operation.OpType.CONV2D);
        add(out, "MaxPool", "maxPool2d", CoverageStatus.SUPPORTED, CoverageStatus.SUPPORTED,
                CoverageStatus.SUPPORTED, gpu(ComputeBackend.GPU_METAL, Operation.OpType.MAX_POOL2D),
                gpu(ComputeBackend.GPU_CUDA, Operation.OpType.MAX_POOL2D),
                "rank-4 NCHW, static attributes, ceil_mode=0", Operation.OpType.MAX_POOL2D);
        add(out, "AveragePool", "avgPool2d", CoverageStatus.SUPPORTED, CoverageStatus.SUPPORTED,
                CoverageStatus.SUPPORTED, gpu(ComputeBackend.GPU_METAL, Operation.OpType.AVG_POOL2D),
                gpu(ComputeBackend.GPU_CUDA, Operation.OpType.AVG_POOL2D),
                "rank-4 NCHW, static attributes, ceil_mode=0; Metal native row is scoped to count_include_pad=false", Operation.OpType.AVG_POOL2D);
        add(out, "LayerNormalization", "layerNorm", CoverageStatus.SUPPORTED, CoverageStatus.SUPPORTED,
                CoverageStatus.SUPPORTED, gpu(ComputeBackend.GPU_METAL, Operation.OpType.LAYER_NORM),
                gpu(ComputeBackend.GPU_CUDA, Operation.OpType.LAYER_NORM),
                "single output; axis must select trailing normalized dimensions", Operation.OpType.LAYER_NORM);
        add(out, "BatchNormalization", "batchNorm with external statistics", CoverageStatus.SUPPORTED, CoverageStatus.UNSUPPORTED,
                CoverageStatus.SUPPORTED, CoverageStatus.PARTIAL, CoverageStatus.PARTIAL,
                "single-output inference form only; export has no first-class batchNorm descriptor", Operation.OpType.SUB, Operation.OpType.DIV, Operation.OpType.MUL, Operation.OpType.ADD);
        addLayout(out, "Transpose", Operation.OpType.PERMUTE);
        addLayout(out, "Reshape", Operation.OpType.RESHAPE, "constant target shape");
        addLayout(out, "Flatten", Operation.OpType.RESHAPE, "static axis reshape");
        addLayout(out, "Expand", Operation.OpType.EXPAND, "constant target shape");
        addLayout(out, "Pad", Operation.OpType.PAD, "constant mode, static non-negative pads, scalar constant value");
        addLayout(out, "Tile", Operation.OpType.TILE, "constant positive repeats");
        addLayout(out, "Squeeze", Operation.OpType.SQUEEZE, "constant axes");
        addLayout(out, "Unsqueeze", Operation.OpType.EXPAND_DIMS, "constant axes");
        addLayout(out, "Slice", Operation.OpType.SLICE, "constant positive-step slice parameters");
        addLayout(out, "Concat", Operation.OpType.CONCAT, "runtime tensors or shape-only axis-0 constants");
        add(out, "Split", "slice per output", CoverageStatus.SUPPORTED, CoverageStatus.UNSUPPORTED,
                CoverageStatus.SUPPORTED, gpu(ComputeBackend.GPU_METAL, Operation.OpType.SLICE),
                gpu(ComputeBackend.GPU_CUDA, Operation.OpType.SLICE),
                "import-only multi-output lowering with static split sizes", Operation.OpType.SLICE);
        add(out, "Shape", "shape constant", CoverageStatus.SUPPORTED, CoverageStatus.UNSUPPORTED,
                CoverageStatus.SUPPORTED, CoverageStatus.UNSUPPORTED, CoverageStatus.UNSUPPORTED,
                "import-time static shape plumbing only");
        add(out, "Size", "size constant", CoverageStatus.SUPPORTED, CoverageStatus.UNSUPPORTED,
                CoverageStatus.SUPPORTED, CoverageStatus.UNSUPPORTED, CoverageStatus.UNSUPPORTED,
                "import-time static shape plumbing only");
        add(out, "Gather", "gatherAxis or shape gather", CoverageStatus.SUPPORTED, CoverageStatus.SUPPORTED,
                CoverageStatus.SUPPORTED, CoverageStatus.UNSUPPORTED, CoverageStatus.UNSUPPORTED,
                "runtime mapping uses GATHER_AXIS; shape-only mapping is axis-0 only", Operation.OpType.GATHER_AXIS);
        add(out, "GatherElements", "takeAlongAxis", CoverageStatus.SUPPORTED, CoverageStatus.SUPPORTED,
                CoverageStatus.SUPPORTED, gpu(ComputeBackend.GPU_METAL, Operation.OpType.TAKE_ALONG_AXIS),
                gpu(ComputeBackend.GPU_CUDA, Operation.OpType.TAKE_ALONG_AXIS),
                "runtime indices are INT32", Operation.OpType.TAKE_ALONG_AXIS);
        add(out, "GatherND", "gatherNd", CoverageStatus.SUPPORTED, CoverageStatus.SUPPORTED,
                CoverageStatus.SUPPORTED, gpu(ComputeBackend.GPU_METAL, Operation.OpType.GATHER_ND),
                gpu(ComputeBackend.GPU_CUDA, Operation.OpType.GATHER_ND),
                "runtime indices are INT32; batch_dims supported", Operation.OpType.GATHER_ND);
        add(out, "ScatterElements", "scatterElements", CoverageStatus.SUPPORTED, CoverageStatus.SUPPORTED,
                CoverageStatus.SUPPORTED, gpu(ComputeBackend.GPU_METAL, Operation.OpType.SCATTER_ELEMENTS),
                gpu(ComputeBackend.GPU_CUDA, Operation.OpType.SCATTER_ELEMENTS),
                "forward reductions none/add/mul/max/min; backward only none/add", Operation.OpType.SCATTER_ELEMENTS);
        add(out, "ScatterND", "scatterNd", CoverageStatus.SUPPORTED, CoverageStatus.SUPPORTED,
                CoverageStatus.SUPPORTED, gpu(ComputeBackend.GPU_METAL, Operation.OpType.SCATTER_ND),
                gpu(ComputeBackend.GPU_CUDA, Operation.OpType.SCATTER_ND),
                "forward reductions none/add/mul/max/min; backward only none/add", Operation.OpType.SCATTER_ND);
        addReduction(out, "ReduceSum", Operation.OpType.SUM);
        addReduction(out, "ReduceMean", Operation.OpType.MEAN);
        addReduction(out, "ReduceMax", Operation.OpType.REDUCE_MAX);
        addReduction(out, "ReduceMin", Operation.OpType.REDUCE_MIN);
        addReduction(out, "ReduceProd", Operation.OpType.REDUCE_PROD);
        add(out, "ArgMax", "argMax", CoverageStatus.SUPPORTED, CoverageStatus.SUPPORTED,
                CoverageStatus.SUPPORTED, gpu(ComputeBackend.GPU_METAL, Operation.OpType.ARGMAX),
                gpu(ComputeBackend.GPU_CUDA, Operation.OpType.ARGMAX),
                "output is INT32 because runtime INT64 tensors are unsupported; select_last_index=0 only", Operation.OpType.ARGMAX);
        add(out, "GlobalAveragePool", "repeated mean over spatial axes", CoverageStatus.SUPPORTED, CoverageStatus.UNSUPPORTED,
                CoverageStatus.SUPPORTED, gpu(ComputeBackend.GPU_METAL, Operation.OpType.MEAN),
                gpu(ComputeBackend.GPU_CUDA, Operation.OpType.MEAN),
                "import-only static rank >= 3 lowering to MEAN", Operation.OpType.MEAN);
        add(out, "Softmax", "softmax", CoverageStatus.SUPPORTED, CoverageStatus.SUPPORTED,
                CoverageStatus.SUPPORTED, gpu(ComputeBackend.GPU_METAL, Operation.OpType.SOFTMAX),
                gpu(ComputeBackend.GPU_CUDA, Operation.OpType.SOFTMAX), "", Operation.OpType.SOFTMAX);
        add(out, "LogSoftmax", "logSoftmax", CoverageStatus.SUPPORTED, CoverageStatus.SUPPORTED,
                CoverageStatus.SUPPORTED, gpu(ComputeBackend.GPU_METAL, Operation.OpType.LOG_SOFTMAX),
                gpu(ComputeBackend.GPU_CUDA, Operation.OpType.LOG_SOFTMAX), "", Operation.OpType.LOG_SOFTMAX);
        add(out, "Constant", "initializer tensor or shape constant", CoverageStatus.SUPPORTED, CoverageStatus.PARTIAL,
                CoverageStatus.SUPPORTED, CoverageStatus.PARTIAL, CoverageStatus.PARTIAL,
                "export usually serializes leaves as graph inputs or initializers rather than Constant nodes");
        return List.copyOf(out);
    }

    private static void addBinary(ArrayList<Entry> out, String onnxOp, Operation.OpType opType) {
        addBinary(out, onnxOp, opType, "");
    }

    private static void addBinary(ArrayList<Entry> out, String onnxOp, Operation.OpType opType, String limitation) {
        add(out, onnxOp, opType.name().toLowerCase(), CoverageStatus.SUPPORTED, CoverageStatus.SUPPORTED,
                CoverageStatus.SUPPORTED, gpu(ComputeBackend.GPU_METAL, opType), gpu(ComputeBackend.GPU_CUDA, opType),
                limitation, opType);
    }

    private static void addUnary(ArrayList<Entry> out, String onnxOp, Operation.OpType opType) {
        add(out, onnxOp, opType.name().toLowerCase(), CoverageStatus.SUPPORTED, CoverageStatus.SUPPORTED,
                CoverageStatus.SUPPORTED, gpu(ComputeBackend.GPU_METAL, opType), gpu(ComputeBackend.GPU_CUDA, opType),
                "", opType);
    }

    private static void addLayout(ArrayList<Entry> out, String onnxOp, Operation.OpType opType) {
        addLayout(out, onnxOp, opType, "");
    }

    private static void addLayout(ArrayList<Entry> out, String onnxOp, Operation.OpType opType, String limitation) {
        add(out, onnxOp, opType.name().toLowerCase(), CoverageStatus.SUPPORTED, CoverageStatus.SUPPORTED,
                CoverageStatus.SUPPORTED, gpu(ComputeBackend.GPU_METAL, opType), gpu(ComputeBackend.GPU_CUDA, opType),
                limitation, opType);
    }

    private static void addReduction(ArrayList<Entry> out, String onnxOp, Operation.OpType opType) {
        add(out, onnxOp, opType.name().toLowerCase(), CoverageStatus.SUPPORTED, CoverageStatus.SUPPORTED,
                CoverageStatus.SUPPORTED, gpu(ComputeBackend.GPU_METAL, opType), gpu(ComputeBackend.GPU_CUDA, opType),
                "multi-axis reductions are imported as repeated single-axis reductions", opType);
    }

    private static void add(
            ArrayList<Entry> out,
            String onnxOp,
            String mapping,
            CoverageStatus importStatus,
            CoverageStatus exportStatus,
            CoverageStatus cpuStatus,
            CoverageStatus metalStatus,
            CoverageStatus cudaStatus,
            String limitations,
            Operation.OpType... mappedOps
    ) {
        out.add(new Entry(onnxOp, mapping, importStatus, exportStatus, cpuStatus, metalStatus, cudaStatus, limitations, List.of(mappedOps)));
    }

    private static CoverageStatus gpu(ComputeBackend backend, Operation.OpType opType) {
        GpuLoweringCoverageStatus status = GpuLoweringCoverageMatrix.entryFor(backend, opType).status();
        return switch (status) {
            case SUPPORTED -> CoverageStatus.SUPPORTED;
            case FALLBACK -> CoverageStatus.PARTIAL;
            case UNSUPPORTED -> CoverageStatus.UNSUPPORTED;
        };
    }
}

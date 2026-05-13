package backend.metal.lowering;

import backend.ComputeBackend;
import backend.accelerator.lowering.GpuLoweringCoverageEntry;
import backend.accelerator.lowering.GpuLoweringCoverageMatrix;
import backend.accelerator.lowering.GpuLoweringCoverageStatus;
import backend.cpu.registry.CpuKernelResolver;
import operations.Operation;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;

/**
 * Code-generated Metal operation parity matrix used by docs and tests.
 */
public final class MetalOperationParityMatrix {
    private static final Set<Operation.OpType> NATIVE_MPSGRAPH_MAPPED = EnumSet.of(
            Operation.OpType.MATMUL,
            Operation.OpType.LINEAR,
            Operation.OpType.ADD,
            Operation.OpType.SUB,
            Operation.OpType.MUL,
            Operation.OpType.DIV,
            Operation.OpType.MIN,
            Operation.OpType.MAX,
            Operation.OpType.RELU,
            Operation.OpType.TANH,
            Operation.OpType.FAST_TANH,
            Operation.OpType.SIGMOID,
            Operation.OpType.ABS,
            Operation.OpType.EXP,
            Operation.OpType.FAST_EXP,
            Operation.OpType.LOG,
            Operation.OpType.NEG,
            Operation.OpType.SQRT,
            Operation.OpType.INV,
            Operation.OpType.CLAMP_MIN,
            Operation.OpType.CLAMP_MAX,
            Operation.OpType.MUL_SCALAR,
            Operation.OpType.POW,
            Operation.OpType.WHERE,
            Operation.OpType.SOFTMAX,
            Operation.OpType.LOG_SOFTMAX,
            Operation.OpType.SUM,
            Operation.OpType.MEAN,
            Operation.OpType.REDUCE_MIN,
            Operation.OpType.REDUCE_MAX,
            Operation.OpType.REDUCE_PROD,
            Operation.OpType.ARGMAX,
            Operation.OpType.CUMSUM,
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
            Operation.OpType.GATHER,
            Operation.OpType.GATHER_AXIS,
            Operation.OpType.TAKE_ALONG_AXIS,
            Operation.OpType.GATHER_GRAD,
            Operation.OpType.GATHER_AXIS_GRAD,
            Operation.OpType.TAKE_ALONG_AXIS_GRAD,
            Operation.OpType.SCATTER_ADD,
            Operation.OpType.CONV2D,
            Operation.OpType.CONV2D_GEMM,
            Operation.OpType.CONV2D_BACKWARD_INPUT,
            Operation.OpType.CONV2D_BACKWARD_WEIGHT,
            Operation.OpType.CONV2D_BACKWARD_INPUT_GEMM,
            Operation.OpType.CONV2D_BACKWARD_WEIGHT_GEMM,
            Operation.OpType.MAX_POOL2D,
            Operation.OpType.AVG_POOL2D,
            Operation.OpType.MAX_POOL2D_BACKWARD_INPUT,
            Operation.OpType.AVG_POOL2D_BACKWARD_INPUT,
            Operation.OpType.SOFTMAX_GRAD,
            Operation.OpType.LOG_SOFTMAX_GRAD,
            Operation.OpType.REDUCE_MIN_GRAD,
            Operation.OpType.REDUCE_MAX_GRAD,
            Operation.OpType.MIN_GRAD,
            Operation.OpType.MAX_GRAD,
            Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION,
            Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS,
            Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_BACKWARD,
            Operation.OpType.NLL_LOSS,
            Operation.OpType.CROSS_ENTROPY_LOSS,
            Operation.OpType.CROSS_ENTROPY_LOSS_INDICES,
            Operation.OpType.CROSS_ENTROPY_LOSS_INDICES_GRAD,
            Operation.OpType.RESHAPE,
            Operation.OpType.CONTIGUOUS,
            Operation.OpType.NOOP,
            Operation.OpType.PERMUTE,
            Operation.OpType.EXPAND,
            Operation.OpType.SELECT,
            Operation.OpType.SLICE,
            Operation.OpType.CONCAT,
            Operation.OpType.PAD,
            Operation.OpType.TILE,
            Operation.OpType.CAST,
            Operation.OpType.EXPAND_DIMS,
            Operation.OpType.SQUEEZE
    );

    private static final Set<Operation.OpType> CUSTOM_ROUTE_ELIGIBLE = EnumSet.of(
            Operation.OpType.RELU
    );

    private MetalOperationParityMatrix() {
    }

    public static List<Row> rows() {
        List<Row> rows = new ArrayList<>();
        for (Operation.OpType opType : Operation.OpType.values()) {
            if (opType == Operation.OpType.UNKNOWN) {
                continue;
            }
            GpuLoweringCoverageEntry coverage = GpuLoweringCoverageMatrix.entryFor(ComputeBackend.GPU_METAL, opType);
            boolean coverageSupported = coverage.status() == GpuLoweringCoverageStatus.SUPPORTED;
            boolean nativeMapped = NATIVE_MPSGRAPH_MAPPED.contains(opType);
            boolean cpuKernel = cpuKernelAvailable(opType);
            boolean customEligible = CUSTOM_ROUTE_ELIGIBLE.contains(opType);
            rows.add(new Row(
                    opType,
                    cpuKernel,
                    opType.isFusable(),
                    coverage.status().name(),
                    coverage.reason().name(),
                    coverageSupported,
                    coverageSupported,
                    nativeMapped,
                    coverageSupported,
                    customEligible,
                    !coverageSupported,
                    coverage.note()
            ));
        }
        return List.copyOf(rows);
    }

    public static String renderMarkdown() {
        StringBuilder out = new StringBuilder();
        out.append("# Metal Operation Parity Matrix\n\n");
        out.append("Generated from `MetalOperationParityMatrix`; do not hand-edit status rows.\n\n");
        out.append("| Operation | CPU kernel | CPU fusable | Metal coverage | Planner supported | DAG lowerable | Native MPSGraph mapped | Buffer executable | Custom route eligible | CPU fallback only | Reason | Note |\n");
        out.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|\n");
        for (Row row : rows()) {
            out.append("| ")
                    .append(row.opType().name())
                    .append(" | ")
                    .append(mark(row.cpuKernelAvailable()))
                    .append(" | ")
                    .append(mark(row.cpuFusable()))
                    .append(" | ")
                    .append(row.metalCoverageStatus().toLowerCase(Locale.ROOT))
                    .append(" | ")
                    .append(mark(row.plannerSupported()))
                    .append(" | ")
                    .append(mark(row.dagLowerable()))
                    .append(" | ")
                    .append(mark(row.nativeMpsGraphMapped()))
                    .append(" | ")
                    .append(mark(row.bufferExecutable()))
                    .append(" | ")
                    .append(mark(row.customRouteEligible()))
                    .append(" | ")
                    .append(mark(row.cpuFallbackOnly()))
                    .append(" | ")
                    .append(row.metalReason())
                    .append(" | ")
                    .append(escape(row.note()))
                    .append(" |\n");
        }
        return out.toString();
    }

    public static void write(Path path) {
        Objects.requireNonNull(path, "path cannot be null");
        try {
            Path parent = path.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            Files.writeString(path, renderMarkdown());
        } catch (IOException e) {
            throw new IllegalStateException("Failed to write Metal operation parity report to " + path + ".", e);
        }
    }

    public static void main(String[] args) {
        if (args.length == 0) {
            System.out.print(renderMarkdown());
            return;
        }
        if (args.length != 1) {
            throw new IllegalArgumentException("Usage: MetalOperationParityMatrix [output-path]");
        }
        write(Path.of(args[0]));
    }

    private static boolean cpuKernelAvailable(Operation.OpType opType) {
        try {
            CpuKernelResolver.resolve(opType);
            return true;
        } catch (RuntimeException ex) {
            return false;
        }
    }

    private static String mark(boolean value) {
        return value ? "yes" : "no";
    }

    private static String escape(String text) {
        return (text == null ? "" : text).replace("|", "\\|");
    }

    public record Row(
            Operation.OpType opType,
            boolean cpuKernelAvailable,
            boolean cpuFusable,
            String metalCoverageStatus,
            String metalReason,
            boolean plannerSupported,
            boolean dagLowerable,
            boolean nativeMpsGraphMapped,
            boolean bufferExecutable,
            boolean customRouteEligible,
            boolean cpuFallbackOnly,
            String note
    ) {
    }
}

package backend.cpu1.trace;

import backend.blas.OpenBlasRuntime;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.prepare.Cpu1PreparedLayoutUnit;
import backend.cpu1.prepare.Cpu1PreparedMatmulUnit;
import backend.cpu1.prepare.Cpu1PreparedReductionUnit;
import backend.cpu1.provider.matmul.Cpu1MatmulRoute;
import graph.CompiledNode;
import graph.execution.trace.DispatchTraceMetadata;
import graph.execution.trace.LayoutTraceMetadata;
import graph.execution.trace.MatMulTraceMetadata;
import graph.execution.trace.ReductionTraceMetadata;
import graph.execution.trace.StepTraceContribution;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.ShortVector;
import tensor.DataType;

import java.util.LinkedHashMap;

/**
 * cpu1-local trace metadata construction for prepared layout, reduction, and matmul units.
 */
public final class Cpu1TraceContributor {
    private Cpu1TraceContributor() {
    }

    public static StepTraceContribution traceContribution(
            CompiledNode node,
            Cpu1PreparedLayoutUnit preparedLayoutUnit,
            Cpu1PreparedReductionUnit preparedReductionUnit,
            Cpu1PreparedMatmulUnit preparedMatmulUnit
    ) {
        if (preparedLayoutUnit != null) {
            return layoutTrace(node, preparedLayoutUnit);
        }
        if (preparedReductionUnit != null) {
            return reductionTrace(preparedReductionUnit);
        }
        if (preparedMatmulUnit != null) {
            return matmulTrace(preparedMatmulUnit);
        }
        return StepTraceContribution.empty();
    }

    private static StepTraceContribution layoutTrace(CompiledNode node, Cpu1PreparedLayoutUnit unit) {
        LinkedHashMap<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("cpu1KernelId", unit.kernelId().name());
        attrs.put("cpu1LayoutKernelId", unit.kernelId().name());
        attrs.put("cpu1StorageKind", unit.storageKind().name());
        attrs.put("cpu1VectorizationKind", unit.vectorizationKind().name());
        attrs.put("cpu1LaunchWorkers", unit.launchConfig().workerCount());
        attrs.put("cpu1LaunchChunkSize", unit.launchConfig().chunkSize());
        attrs.put("cpu1MaterializeThreshold", unit.materializeThreshold());
        LayoutTraceMetadata layout = new LayoutTraceMetadata(
                node.storageOffset(),
                node.contiguous(),
                false,
                unit.kernelId().name()
        );
        DispatchTraceMetadata dispatch = new DispatchTraceMetadata(
                unit.vectorizationKind().name(),
                layoutVectorWidth(unit),
                unit.launchConfig().workerCount(),
                unit.launchConfig().chunkSize(),
                unit.launchConfig().chunkSize()
        );
        return new StepTraceContribution(
                unit.kernelId().name(),
                attrs,
                null,
                layout,
                dispatch,
                null,
                null,
                null,
                null
        );
    }

    private static int layoutVectorWidth(Cpu1PreparedLayoutUnit unit) {
        return switch (unit.vectorizationKind()) {
            case SCALAR -> 1;
            case VECTOR -> vectorWidth(unit.dataType());
        };
    }

    private static StepTraceContribution reductionTrace(Cpu1PreparedReductionUnit unit) {
        LinkedHashMap<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("cpu1KernelId", unit.kernelId().name());
        attrs.put("cpu1ReductionKernelId", unit.kernelId().name());
        attrs.put("cpu1StorageKind", unit.storageKind().name());
        attrs.put("cpu1ReductionAxis", unit.axis());
        attrs.put("cpu1ReductionAxisSize", unit.axisSize());
        attrs.put("cpu1ReductionInnerSize", unit.innerSize());
        attrs.put("cpu1ReductionOuterSize", unit.outerSize());
        attrs.put("cpu1ReductionOutputElements", unit.outputElementCount());
        attrs.put("cpu1ReductionKeepDims", unit.keepDims());
        ReductionTraceMetadata reduction = new ReductionTraceMetadata(
                unit.opType().name(),
                1,
                unit.outputElementCount(),
                1,
                unit.dataType() == DataType.BFLOAT16 ? "F32_ACCUMULATE" : unit.dataType().name()
        );
        return new StepTraceContribution(
                unit.kernelId().name(),
                attrs,
                null,
                null,
                null,
                reduction,
                null,
                null,
                null
        );
    }

    private static StepTraceContribution matmulTrace(Cpu1PreparedMatmulUnit unit) {
        LinkedHashMap<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("cpu1KernelId", unit.kernelId().name());
        attrs.put("cpu1MatmulKernelId", unit.kernelId().name());
        attrs.put("cpu1MatmulRoute", unit.route().name());
        attrs.put("cpu1StorageKind", unit.storageKind().name());
        attrs.put("cpu1MatmulVectorizationKind", unit.vectorizationKind().name());
        attrs.put("cpu1MatmulVectorWidth", matmulVectorWidth(unit));
        attrs.put("cpu1MatmulBatchCount", unit.batchCount());
        attrs.put("cpu1MatmulM", unit.m());
        attrs.put("cpu1MatmulN", unit.n());
        attrs.put("cpu1MatmulK", unit.k());
        attrs.put("cpu1MatmulWork", unit.work());
        attrs.put("cpu1MatmulLaunchWorkers", unit.launchConfig().workerCount());
        attrs.put("cpu1MatmulLaunchChunkSize", unit.launchConfig().chunkSize());
        addMatmulBlasAttrs(attrs, unit);
        MatMulTraceMetadata matMul = new MatMulTraceMetadata(
                matmulUsesBlas(unit),
                matmulUsesBatchedBlas(unit),
                matmulBlasProvider(unit),
                matmulBlasSymbol(unit),
                matmulBlasRoute(unit),
                unit.route().name(),
                unit.storageKind().name(),
                "",
                unit.storageKind().name(),
                unit.storageKind().name(),
                "",
                matmulUsesBlas(unit) && OpenBlasRuntime.isFloat32GemmAvailable(),
                matmulUsesBlas(unit) && OpenBlasRuntime.isFloat64GemmAvailable(),
                matmulUsesBlas(unit) && OpenBlasRuntime.isBFloat16ToFloatGemmAvailable(),
                matmulUsesBlas(unit) && OpenBlasRuntime.isBFloat16OutputGemmAvailable(),
                unit.dataType() == DataType.BFLOAT16 ? "JAVA" : "",
                unit.dataType() == DataType.BFLOAT16 ? "JAVA" : "",
                unit.dataType() == DataType.BFLOAT16 ? "F32_PROMOTED" : unit.dataType().name(),
                unit.dataType().name(),
                matmulCopyInBytes(unit),
                matmulCopyOutBytes(unit),
                matmulUsesBlas(unit) ? -1L : 0L,
                matmulUsesBlas(unit) ? OpenBlasRuntime.threadPolicy() : "SINGLE_THREAD",
                "",
                false,
                unit.m(),
                unit.n(),
                unit.k(),
                1,
                unit.work(),
                matmulImplementation(unit)
        );
        return new StepTraceContribution(
                unit.kernelId().name(),
                attrs,
                null,
                null,
                null,
                null,
                matMul,
                null,
                null
        );
    }

    private static int matmulVectorWidth(Cpu1PreparedMatmulUnit unit) {
        if (unit.vectorizationKind() == Cpu1VectorizationKind.SCALAR) {
            return 1;
        }
        return vectorWidth(unit.dataType());
    }

    private static String matmulImplementation(Cpu1PreparedMatmulUnit unit) {
        if (unit.route() == Cpu1MatmulRoute.OPENBLAS_ARRAY_COPYING) {
            return "OPENBLAS_ARRAY_COPYING";
        }
        return unit.vectorizationKind() == Cpu1VectorizationKind.VECTOR
                ? "JAVA_VECTOR_PACKED_B"
                : "JAVA_SCALAR";
    }

    private static void addMatmulBlasAttrs(LinkedHashMap<String, Object> attrs, Cpu1PreparedMatmulUnit unit) {
        if (!matmulUsesBlas(unit)) {
            return;
        }
        attrs.put("blasProvider", matmulBlasProvider(unit));
        attrs.put("blasSymbol", matmulBlasSymbol(unit));
        attrs.put("blasRoute", matmulBlasRoute(unit));
        attrs.put("openblasSgemmAvailable", OpenBlasRuntime.isFloat32GemmAvailable());
        attrs.put("openblasDgemmAvailable", OpenBlasRuntime.isFloat64GemmAvailable());
        attrs.put("openblasSbgemmAvailable", OpenBlasRuntime.isBFloat16ToFloatGemmAvailable());
        attrs.put("openblasBgemmAvailable", OpenBlasRuntime.isBFloat16OutputGemmAvailable());
        attrs.put("openblasLookupSource", OpenBlasRuntime.lookupSource());
        attrs.put("matMulCopyInBytes", matmulCopyInBytes(unit));
        attrs.put("matMulCopyOutBytes", matmulCopyOutBytes(unit));
        attrs.put("matMulNativeTempBytes", -1L);
        attrs.put("blasThreadPolicy", OpenBlasRuntime.threadPolicy());
    }

    private static boolean matmulUsesBlas(Cpu1PreparedMatmulUnit unit) {
        return unit.route() == Cpu1MatmulRoute.OPENBLAS_ARRAY_COPYING;
    }

    private static boolean matmulUsesBatchedBlas(Cpu1PreparedMatmulUnit unit) {
        return matmulUsesBlas(unit) && unit.batchCount() > 1;
    }

    private static String matmulBlasProvider(Cpu1PreparedMatmulUnit unit) {
        return matmulUsesBlas(unit) ? "OPENBLAS_FFM" : "";
    }

    private static String matmulBlasSymbol(Cpu1PreparedMatmulUnit unit) {
        if (!matmulUsesBlas(unit)) {
            return "";
        }
        return switch (unit.dataType()) {
            case FLOAT32 -> "cblas_sgemm";
            case FLOAT64 -> "cblas_dgemm";
            default -> "";
        };
    }

    private static String matmulBlasRoute(Cpu1PreparedMatmulUnit unit) {
        return matmulUsesBlas(unit) ? Cpu1MatmulRoute.OPENBLAS_ARRAY_COPYING.name() : "";
    }

    private static long matmulCopyInBytes(Cpu1PreparedMatmulUnit unit) {
        if (!matmulUsesBlas(unit)) {
            return 0L;
        }
        long leftElements = Math.multiplyExact(Math.multiplyExact((long) unit.batchCount(), unit.m()), unit.k());
        long rightElements = Math.multiplyExact(Math.multiplyExact((long) unit.batchCount(), unit.k()), unit.n());
        return Math.multiplyExact(Math.addExact(leftElements, rightElements), elementBytes(unit.dataType()));
    }

    private static long matmulCopyOutBytes(Cpu1PreparedMatmulUnit unit) {
        if (!matmulUsesBlas(unit)) {
            return 0L;
        }
        long outputElements = Math.multiplyExact(Math.multiplyExact((long) unit.batchCount(), unit.m()), unit.n());
        return Math.multiplyExact(outputElements, elementBytes(unit.dataType()));
    }

    private static int elementBytes(DataType dataType) {
        return switch (dataType) {
            case FLOAT32 -> Float.BYTES;
            case INT32 -> Integer.BYTES;
            case FLOAT64 -> Double.BYTES;
            case INT64 -> Long.BYTES;
            case BFLOAT16 -> Short.BYTES;
            case BOOL -> Byte.BYTES;
        };
    }

    private static int vectorWidth(DataType dataType) {
        return switch (dataType) {
            case FLOAT32 -> FloatVector.SPECIES_PREFERRED.length();
            case FLOAT64 -> DoubleVector.SPECIES_PREFERRED.length();
            case BFLOAT16 -> ShortVector.SPECIES_PREFERRED.length();
            case BOOL -> ByteVector.SPECIES_PREFERRED.length();
            case INT32, INT64 -> 1;
        };
    }
}

package backend.cpu1.trace;

import backend.blas.OpenBlasRuntime;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.prepare.Cpu1PreparedDTypeUnit;
import backend.cpu1.prepare.Cpu1PreparedLayoutUnit;
import backend.cpu1.prepare.Cpu1PreparedFusedElementwiseUnit;
import backend.cpu1.prepare.Cpu1PreparedMatmulUnit;
import backend.cpu1.prepare.Cpu1PreparedMseLossUnit;
import backend.cpu1.prepare.Cpu1PreparedReductionUnit;
import backend.cpu1.prepare.dispatch.Cpu1CostClass;
import backend.cpu1.provider.matmul.Cpu1MatmulRoute;
import graph.CompiledNode;
import graph.execution.trace.DispatchTraceMetadata;
import graph.execution.trace.FusedTraceMetadata;
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
            Cpu1PreparedDTypeUnit preparedDTypeUnit,
            Cpu1PreparedReductionUnit preparedReductionUnit,
            Cpu1PreparedMatmulUnit preparedMatmulUnit,
            Cpu1PreparedMseLossUnit preparedMseLossUnit,
            Cpu1PreparedFusedElementwiseUnit preparedFusedElementwiseUnit
    ) {
        if (preparedLayoutUnit != null) {
            return layoutTrace(node, preparedLayoutUnit);
        }
        if (preparedDTypeUnit != null) {
            return dtypeTrace(preparedDTypeUnit);
        }
        if (preparedReductionUnit != null) {
            return reductionTrace(preparedReductionUnit);
        }
        if (preparedMatmulUnit != null) {
            return matmulTrace(preparedMatmulUnit);
        }
        if (preparedMseLossUnit != null) {
            return mseLossTrace(preparedMseLossUnit);
        }
        if (preparedFusedElementwiseUnit != null) {
            return fusedElementwiseTrace(preparedFusedElementwiseUnit);
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

    private static StepTraceContribution dtypeTrace(Cpu1PreparedDTypeUnit unit) {
        LinkedHashMap<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("cpu1KernelId", unit.kernelId().name());
        attrs.put("cpu1DTypeKernelId", unit.kernelId().name());
        attrs.put("cpu1StorageKind", unit.storageKind().name());
        attrs.put("cpu1LayoutKind", unit.layoutKind().name());
        attrs.put("cpu1InputDType", unit.inputDataType().name());
        attrs.put("cpu1OutputDType", unit.outputDataType().name());
        attrs.put("cpu1ElementCount", unit.elementCount());
        attrs.put("cpu1LaunchWorkers", unit.launchConfig().workerCount());
        attrs.put("cpu1LaunchChunkSize", unit.launchConfig().chunkSize());
        DispatchTraceMetadata dispatch = new DispatchTraceMetadata(
                Cpu1VectorizationKind.SCALAR.name(),
                1,
                unit.launchConfig().workerCount(),
                unit.launchConfig().chunkSize(),
                unit.launchConfig().chunkSize()
        );
        return new StepTraceContribution(
                unit.kernelId().name(),
                attrs,
                null,
                null,
                dispatch,
                null,
                null,
                null,
                null
        );
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
        attrs.put("cpu1MatmulPostOp", unit.postOp().name());
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
                matmulUsesBlas(unit) ? OpenBlasRuntime.threadPolicy(unit.openBlasThreads()) : "SINGLE_THREAD",
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

    private static StepTraceContribution mseLossTrace(Cpu1PreparedMseLossUnit unit) {
        LinkedHashMap<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("cpu1KernelId", unit.kernelId().name());
        attrs.put("cpu1MseLossKernelId", unit.kernelId().name());
        attrs.put("cpu1SpecializationKind", "MSE_LOSS");
        attrs.put("cpu1StorageKind", unit.storageKind().name());
        attrs.put("cpu1MseLossReduction", unit.reductionOpType().name());
        attrs.put("cpu1MseLossElementCount", unit.elementCount());
        attrs.put("cpu1MseLossReductionDivisor", unit.reductionDivisor());
        attrs.put("cpu1MseLossReductionNodeCount", unit.orderedNodeIds().size() - 2);
        attrs.put("cpu1MseLossLaunchWorkers", unit.launchConfig().workerCount());
        attrs.put("cpu1MseLossLaunchChunkSize", unit.launchConfig().chunkSize());
        ReductionTraceMetadata reduction = new ReductionTraceMetadata(
                unit.reductionOpType().name(),
                1,
                1,
                1,
                unit.dataType() == DataType.BFLOAT16 ? "F32_ACCUMULATE" : "F64_ACCUMULATE"
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

    private static StepTraceContribution fusedElementwiseTrace(Cpu1PreparedFusedElementwiseUnit unit) {
        LinkedHashMap<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("cpu1KernelId", "CPU1_FUSED_ELEMENTWISE");
        attrs.put("cpu1FusedNodeCount", unit.plan().nodeCount());
        attrs.put("cpu1FusedInputCount", unit.plan().inputCount());
        attrs.put("cpu1StorageKind", unit.storageKind().name());
        attrs.put("cpu1LayoutKind", unit.layoutKind().name());
        attrs.put("cpu1FusedOutputNodeId", unit.outputNodeId());
        attrs.put("cpu1FusedElementCount", unit.elementCount());
        attrs.put("cpu1FusedLaunchWorkers", unit.launchConfig().workerCount());
        attrs.put("cpu1FusedLaunchChunkSize", unit.launchConfig().chunkSize());
        attrs.put("cpu1FusedCostClass", unit.dispatchDecision().costClass().name());
        attrs.put("cpu1FusedRequestedVectorization", unit.dispatchDecision().requestedVectorizationKind().name());
        attrs.put("cpu1FusedApproxExp", unit.approximateExp());
        attrs.put("cpu1FusedApproxTanh", unit.approximateTanh());
        attrs.put("cpu1FusedCodegenRejectionReason", unit.codegenRejectionReason().name());
        attrs.put("cpu1FusedClassSignature", unit.generatedKernel().classSignature().canonicalSignature());
        attrs.put("cpu1FusedGeneratedClassName", unit.generatedKernel().generatedClassName());
        DispatchTraceMetadata dispatch = new DispatchTraceMetadata(
                unit.dispatchDecision().requestedVectorizationKind().name(),
                unit.dispatchDecision().requestedVectorizationKind() == Cpu1VectorizationKind.VECTOR
                        ? vectorWidth(unit.outputDataType())
                        : 1,
                unit.launchConfig().workerCount(),
                unit.launchConfig().chunkSize(),
                unit.launchConfig().chunkSize()
        );
        FusedTraceMetadata fused = new FusedTraceMetadata(
                unit.storageKind().name() + ":" + unit.outputDataType().name(),
                unit.dispatchDecision().costClass() == Cpu1CostClass.CHEAP_ELEMENTWISE,
                "CPU1_FUSED_ELEMENTWISE",
                unit.generatedKernel().classSignature().canonicalSignature(),
                "CPU1",
                unit.plan().nodeCount(),
                unit.plan().inputCount(),
                "NONE"
        );
        return new StepTraceContribution(
                "CPU1_FUSED_ELEMENTWISE",
                attrs,
                null,
                null,
                dispatch,
                null,
                null,
                null,
                fused
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
        if (unit.route() == Cpu1MatmulRoute.OPENBLAS_NATIVE_SEGMENT) {
            return "OPENBLAS_NATIVE_SEGMENT";
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
        attrs.put("blasThreadPolicy", OpenBlasRuntime.threadPolicy(unit.openBlasThreads()));
    }

    private static boolean matmulUsesBlas(Cpu1PreparedMatmulUnit unit) {
        return unit.route() == Cpu1MatmulRoute.OPENBLAS_ARRAY_COPYING
                || unit.route() == Cpu1MatmulRoute.OPENBLAS_NATIVE_SEGMENT;
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
        return matmulUsesBlas(unit) ? unit.route().name() : "";
    }

    private static long matmulCopyInBytes(Cpu1PreparedMatmulUnit unit) {
        if (!matmulUsesBlas(unit) || unit.route() == Cpu1MatmulRoute.OPENBLAS_NATIVE_SEGMENT) {
            return 0L;
        }
        long leftElements = Math.multiplyExact(Math.multiplyExact((long) unit.batchCount(), unit.m()), unit.k());
        long rightElements = Math.multiplyExact(Math.multiplyExact((long) unit.batchCount(), unit.k()), unit.n());
        return Math.multiplyExact(Math.addExact(leftElements, rightElements), elementBytes(unit.dataType()));
    }

    private static long matmulCopyOutBytes(Cpu1PreparedMatmulUnit unit) {
        if (!matmulUsesBlas(unit) || unit.route() == Cpu1MatmulRoute.OPENBLAS_NATIVE_SEGMENT) {
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

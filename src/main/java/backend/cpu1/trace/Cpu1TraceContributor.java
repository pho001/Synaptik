package backend.cpu1.trace;

import backend.blas.OpenBlasRuntime;
import backend.cpu1.exec.Cpu1ScratchBufferSpec;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.prepare.Cpu1PreparedDTypeUnit;
import backend.cpu1.prepare.Cpu1PreparedLayoutUnit;
import backend.cpu1.prepare.Cpu1PreparedConv2dUnit;
import backend.cpu1.prepare.Cpu1PreparedFusedElementwiseUnit;
import backend.cpu1.prepare.Cpu1PreparedCrossEntropyLossUnit;
import backend.cpu1.prepare.Cpu1PreparedDenseCrossEntropyLossUnit;
import backend.cpu1.prepare.Cpu1PreparedMatmulUnit;
import backend.cpu1.prepare.Cpu1PreparedMseLossUnit;
import backend.cpu1.prepare.Cpu1PreparedNllLossUnit;
import backend.cpu1.prepare.Cpu1PreparedIndexUnit;
import backend.cpu1.prepare.Cpu1PreparedLayerNormUnit;
import backend.cpu1.prepare.Cpu1PreparedAvgPool2dUnit;
import backend.cpu1.prepare.Cpu1PreparedMaxPool2dUnit;
import backend.cpu1.prepare.Cpu1PreparedReductionUnit;
import backend.cpu1.prepare.Cpu1PreparedRmsNormUnit;
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
            Cpu1PreparedCrossEntropyLossUnit preparedCrossEntropyLossUnit,
            Cpu1PreparedDenseCrossEntropyLossUnit preparedDenseCrossEntropyLossUnit,
            Cpu1PreparedNllLossUnit preparedNllLossUnit,
            Cpu1PreparedFusedElementwiseUnit preparedFusedElementwiseUnit,
            Cpu1PreparedIndexUnit preparedIndexUnit,
            Cpu1PreparedLayerNormUnit preparedLayerNormUnit,
            Cpu1PreparedRmsNormUnit preparedRmsNormUnit,
            Cpu1PreparedMaxPool2dUnit preparedMaxPool2dUnit,
            Cpu1PreparedAvgPool2dUnit preparedAvgPool2dUnit,
            Cpu1PreparedConv2dUnit preparedConv2dUnit
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
        if (preparedCrossEntropyLossUnit != null) {
            return crossEntropyLossTrace(preparedCrossEntropyLossUnit);
        }
        if (preparedDenseCrossEntropyLossUnit != null) {
            return denseCrossEntropyLossTrace(preparedDenseCrossEntropyLossUnit);
        }
        if (preparedNllLossUnit != null) {
            return nllLossTrace(preparedNllLossUnit);
        }
        if (preparedFusedElementwiseUnit != null) {
            return fusedElementwiseTrace(preparedFusedElementwiseUnit);
        }
        if (preparedIndexUnit != null) {
            return indexTrace(preparedIndexUnit);
        }
        if (preparedLayerNormUnit != null) {
            return layerNormTrace(preparedLayerNormUnit);
        }
        if (preparedRmsNormUnit != null) {
            return rmsNormTrace(preparedRmsNormUnit);
        }
        if (preparedMaxPool2dUnit != null) {
            return maxPool2dTrace(preparedMaxPool2dUnit);
        }
        if (preparedAvgPool2dUnit != null) {
            return avgPool2dTrace(preparedAvgPool2dUnit);
        }
        if (preparedConv2dUnit != null) {
            return conv2dTrace(preparedConv2dUnit);
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
        attrs.put("cpu1ReductionInputAccessKind", unit.inputAccessPlan().kind().name());
        attrs.put("cpu1ReductionOutputAccessKind", unit.outputAccessPlan().kind().name());
        attrs.put("cpu1ReductionAxis", unit.axis());
        attrs.put("cpu1ReductionAxisSize", unit.axisSize());
        attrs.put("cpu1ReductionInnerSize", unit.innerSize());
        attrs.put("cpu1ReductionOuterSize", unit.outerSize());
        attrs.put("cpu1ReductionOutputElements", unit.outputElementCount());
        attrs.put("cpu1ReductionKeepDims", unit.keepDims());
        attrs.put("cpu1ReductionLaunchWorkers", unit.launchConfig().workerCount());
        attrs.put("cpu1ReductionLaunchChunkSize", unit.launchConfig().chunkSize());
        Cpu1ScratchBufferSpec scratch = unit.scratchBufferSpec();
        attrs.put("cpu1ReductionScratchF32", scratch.f32ArrayElements());
        attrs.put("cpu1ReductionScratchF64", scratch.f64ArrayElements());
        attrs.put("cpu1ReductionScratchI32", scratch.i32ArrayElements());
        ReductionTraceMetadata reduction = new ReductionTraceMetadata(
                unit.opType().name(),
                unit.launchConfig().workerCount(),
                unit.launchConfig().chunkSize(),
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

    private static StepTraceContribution crossEntropyLossTrace(Cpu1PreparedCrossEntropyLossUnit unit) {
        LinkedHashMap<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("cpu1KernelId", unit.kernelId().name());
        attrs.put("cpu1CrossEntropyLossKernelId", unit.kernelId().name());
        attrs.put("cpu1LossOpType", unit.opType().name());
        attrs.put("cpu1StorageKind", unit.storageKind().name());
        attrs.put("cpu1LossLogitsDType", unit.logitsDataType().name());
        attrs.put("cpu1LossTargetDType", unit.targetDataType().name());
        attrs.put("cpu1LossReduction", unit.reduction().name());
        attrs.put("cpu1LossIgnoreIndex", unit.ignoreIndex());
        attrs.put("cpu1LossClassAxis", unit.classAxis());
        attrs.put("cpu1LossAxisSize", unit.axisSize());
        attrs.put("cpu1LossAxisStride", unit.axisStride());
        attrs.put("cpu1LossGroupCount", unit.groupCount());
        attrs.put("cpu1LossLaunchWorkers", unit.launchConfig().workerCount());
        attrs.put("cpu1LossLaunchChunkSize", unit.launchConfig().chunkSize());
        Cpu1ScratchBufferSpec scratch = unit.scratchBufferSpec();
        attrs.put("cpu1LossScratchF64", scratch.f64ArrayElements());
        attrs.put("cpu1LossScratchI32", scratch.i32ArrayElements());
        ReductionTraceMetadata reduction = new ReductionTraceMetadata(
                unit.reduction().name(),
                unit.launchConfig().workerCount(),
                unit.launchConfig().chunkSize(),
                1,
                unit.logitsDataType() == DataType.BFLOAT16 ? "F32_ACCUMULATE" : "F64_ACCUMULATE"
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

    private static StepTraceContribution nllLossTrace(Cpu1PreparedNllLossUnit unit) {
        LinkedHashMap<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("cpu1KernelId", unit.kernelId().name());
        attrs.put("cpu1NllLossKernelId", unit.kernelId().name());
        attrs.put("cpu1LossOpType", unit.opType().name());
        attrs.put("cpu1StorageKind", unit.storageKind().name());
        attrs.put("cpu1LossDType", unit.dataType().name());
        attrs.put("cpu1LossReduction", "MEAN");
        attrs.put("cpu1LossClassAxis", unit.classAxis());
        attrs.put("cpu1LossAxisSize", unit.axisSize());
        attrs.put("cpu1LossAxisStride", unit.axisStride());
        attrs.put("cpu1LossGroupCount", unit.groupCount());
        attrs.put("cpu1LossLaunchWorkers", unit.launchConfig().workerCount());
        attrs.put("cpu1LossLaunchChunkSize", unit.launchConfig().chunkSize());
        Cpu1ScratchBufferSpec scratch = unit.scratchBufferSpec();
        attrs.put("cpu1LossScratchF64", scratch.f64ArrayElements());
        attrs.put("cpu1LossScratchI32", scratch.i32ArrayElements());
        ReductionTraceMetadata reduction = new ReductionTraceMetadata(
                unit.opType().name(),
                unit.launchConfig().workerCount(),
                unit.launchConfig().chunkSize(),
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

    private static StepTraceContribution denseCrossEntropyLossTrace(Cpu1PreparedDenseCrossEntropyLossUnit unit) {
        LinkedHashMap<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("cpu1KernelId", unit.kernelId().name());
        attrs.put("cpu1DenseCrossEntropyLossKernelId", unit.kernelId().name());
        attrs.put("cpu1LossOpType", unit.opType().name());
        attrs.put("cpu1StorageKind", unit.storageKind().name());
        attrs.put("cpu1LossDType", unit.dataType().name());
        attrs.put("cpu1LossReduction", "MEAN");
        attrs.put("cpu1LossClassAxis", unit.classAxis());
        attrs.put("cpu1LossAxisSize", unit.axisSize());
        attrs.put("cpu1LossAxisStride", unit.axisStride());
        attrs.put("cpu1LossGroupCount", unit.groupCount());
        attrs.put("cpu1LossLaunchWorkers", unit.launchConfig().workerCount());
        attrs.put("cpu1LossLaunchChunkSize", unit.launchConfig().chunkSize());
        Cpu1ScratchBufferSpec scratch = unit.scratchBufferSpec();
        attrs.put("cpu1LossScratchF64", scratch.f64ArrayElements());
        attrs.put("cpu1LossScratchI32", scratch.i32ArrayElements());
        ReductionTraceMetadata reduction = new ReductionTraceMetadata(
                unit.opType().name(),
                unit.launchConfig().workerCount(),
                unit.launchConfig().chunkSize(),
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

    private static StepTraceContribution indexTrace(Cpu1PreparedIndexUnit unit) {
        LinkedHashMap<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("cpu1KernelId", unit.kernelId().name());
        attrs.put("cpu1IndexKernelId", unit.kernelId().name());
        attrs.put("cpu1IndexOpType", unit.opType().name());
        attrs.put("cpu1IndexReduction", unit.reduction().name());
        attrs.put("cpu1StorageKind", unit.storageKind().name());
        attrs.put("cpu1IndexValueDType", unit.valueDataType().name());
        attrs.put("cpu1IndexDType", unit.indexDataType().name());
        attrs.put("cpu1IndexDimension", unit.dimension());
        attrs.put("cpu1IndexAxisSize", unit.axisSize());
        attrs.put("cpu1IndexInnerSize", unit.innerSize());
        attrs.put("cpu1IndexOuterSize", unit.outerSize());
        attrs.put("cpu1IndexElementCount", unit.indexElementCount());
        attrs.put("cpu1IndexAxisIndexCount", unit.indexAxisSize());
        attrs.put("cpu1IndexBatchDims", unit.batchDims());
        attrs.put("cpu1IndexTupleRank", unit.tupleRank());
        attrs.put("cpu1IndexPrefixRank", unit.prefixRank());
        attrs.put("cpu1IndexHasUpdates", unit.hasUpdateInput());
        attrs.put("cpu1IndexUpdateElements", unit.updateElementCount());
        attrs.put("cpu1IndexOutputElements", unit.outputElementCount());
        attrs.put("cpu1IndexLaunchWorkers", unit.launchConfig().workerCount());
        attrs.put("cpu1IndexLaunchChunkSize", unit.launchConfig().chunkSize());
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

    private static StepTraceContribution layerNormTrace(Cpu1PreparedLayerNormUnit unit) {
        LinkedHashMap<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("cpu1KernelId", unit.kernelId().name());
        attrs.put("cpu1LayerNormKernelId", unit.kernelId().name());
        attrs.put("cpu1NormalizationOpType", unit.opType().name());
        attrs.put("cpu1StorageKind", unit.storageKind().name());
        attrs.put("cpu1NormalizationDType", unit.dataType().name());
        attrs.put("cpu1LayerNormNormalizedRank", unit.normalizedRank());
        attrs.put("cpu1LayerNormNormalizedSize", unit.normalizedSize());
        attrs.put("cpu1LayerNormGroupCount", unit.groupCount());
        attrs.put("cpu1LayerNormOutputElements", unit.outputElementCount());
        attrs.put("cpu1LayerNormEpsilon", unit.epsilon());
        attrs.put("cpu1LayerNormInputAccessKind", unit.inputAccessPlan().kind().name());
        attrs.put("cpu1LayerNormGammaAccessKind", unit.gammaAccessPlan().kind().name());
        attrs.put("cpu1LayerNormBetaAccessKind", unit.betaAccessPlan().kind().name());
        attrs.put("cpu1LayerNormOutputAccessKind", unit.outputAccessPlan().kind().name());
        attrs.put("cpu1LayerNormLaunchWorkers", unit.launchConfig().workerCount());
        attrs.put("cpu1LayerNormLaunchChunkSize", unit.launchConfig().chunkSize());
        DispatchTraceMetadata dispatch = new DispatchTraceMetadata(
                Cpu1VectorizationKind.SCALAR.name(),
                1,
                unit.launchConfig().workerCount(),
                unit.launchConfig().chunkSize(),
                unit.launchConfig().chunkSize()
        );
        ReductionTraceMetadata reduction = new ReductionTraceMetadata(
                unit.opType().name(),
                unit.launchConfig().workerCount(),
                unit.launchConfig().chunkSize(),
                1,
                unit.dataType() == DataType.BFLOAT16 ? "F32_ACCUMULATE" : unit.dataType().name()
        );
        return new StepTraceContribution(
                unit.kernelId().name(),
                attrs,
                null,
                null,
                dispatch,
                reduction,
                null,
                null,
                null
        );
    }

    private static StepTraceContribution rmsNormTrace(Cpu1PreparedRmsNormUnit unit) {
        LinkedHashMap<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("cpu1KernelId", unit.kernelId().name());
        attrs.put("cpu1RmsNormKernelId", unit.kernelId().name());
        attrs.put("cpu1NormalizationOpType", unit.opType().name());
        attrs.put("cpu1StorageKind", unit.storageKind().name());
        attrs.put("cpu1NormalizationDType", unit.dataType().name());
        attrs.put("cpu1RmsNormNormalizedRank", unit.normalizedRank());
        attrs.put("cpu1RmsNormNormalizedSize", unit.normalizedSize());
        attrs.put("cpu1RmsNormGroupCount", unit.groupCount());
        attrs.put("cpu1RmsNormOutputElements", unit.outputElementCount());
        attrs.put("cpu1RmsNormEpsilon", unit.epsilon());
        attrs.put("cpu1RmsNormInputAccessKind", unit.inputAccessPlan().kind().name());
        attrs.put("cpu1RmsNormGammaAccessKind", unit.gammaAccessPlan().kind().name());
        attrs.put("cpu1RmsNormOutputAccessKind", unit.outputAccessPlan().kind().name());
        attrs.put("cpu1RmsNormLaunchWorkers", unit.launchConfig().workerCount());
        attrs.put("cpu1RmsNormLaunchChunkSize", unit.launchConfig().chunkSize());
        DispatchTraceMetadata dispatch = new DispatchTraceMetadata(
                Cpu1VectorizationKind.SCALAR.name(),
                1,
                unit.launchConfig().workerCount(),
                unit.launchConfig().chunkSize(),
                unit.launchConfig().chunkSize()
        );
        ReductionTraceMetadata reduction = new ReductionTraceMetadata(
                unit.opType().name(),
                unit.launchConfig().workerCount(),
                unit.launchConfig().chunkSize(),
                1,
                unit.dataType() == DataType.BFLOAT16 ? "F32_ACCUMULATE" : unit.dataType().name()
        );
        return new StepTraceContribution(
                unit.kernelId().name(),
                attrs,
                null,
                null,
                dispatch,
                reduction,
                null,
                null,
                null
        );
    }

    private static StepTraceContribution maxPool2dTrace(Cpu1PreparedMaxPool2dUnit unit) {
        LinkedHashMap<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("cpu1KernelId", unit.kernelId().name());
        attrs.put("cpu1MaxPool2dKernelId", unit.kernelId().name());
        attrs.put("cpu1Pool2dOpType", unit.opType().name());
        attrs.put("cpu1StorageKind", unit.storageKind().name());
        attrs.put("cpu1Pool2dDType", unit.dataType().name());
        attrs.put("cpu1Pool2dBatchCount", unit.batchCount());
        attrs.put("cpu1Pool2dChannels", unit.channels());
        attrs.put("cpu1Pool2dInputH", unit.inputH());
        attrs.put("cpu1Pool2dInputW", unit.inputW());
        attrs.put("cpu1Pool2dOutputH", unit.outputH());
        attrs.put("cpu1Pool2dOutputW", unit.outputW());
        attrs.put("cpu1Pool2dKernelH", unit.kernelH());
        attrs.put("cpu1Pool2dKernelW", unit.kernelW());
        attrs.put("cpu1Pool2dStrideH", unit.strideH());
        attrs.put("cpu1Pool2dStrideW", unit.strideW());
        attrs.put("cpu1Pool2dPadH", unit.padH());
        attrs.put("cpu1Pool2dPadW", unit.padW());
        attrs.put("cpu1Pool2dCeilMode", unit.options().ceilMode());
        attrs.put("cpu1Pool2dOutputElements", unit.outputElementCount());
        attrs.put("cpu1Pool2dInputAccessKind", unit.inputAccessPlan().kind().name());
        attrs.put("cpu1Pool2dOutputAccessKind", unit.outputAccessPlan().kind().name());
        attrs.put("cpu1Pool2dLaunchWorkers", unit.launchConfig().workerCount());
        attrs.put("cpu1Pool2dLaunchChunkSize", unit.launchConfig().chunkSize());
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

    private static StepTraceContribution avgPool2dTrace(Cpu1PreparedAvgPool2dUnit unit) {
        LinkedHashMap<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("cpu1KernelId", unit.kernelId().name());
        attrs.put("cpu1AvgPool2dKernelId", unit.kernelId().name());
        attrs.put("cpu1Pool2dOpType", unit.opType().name());
        attrs.put("cpu1StorageKind", unit.storageKind().name());
        attrs.put("cpu1Pool2dDType", unit.dataType().name());
        attrs.put("cpu1Pool2dBatchCount", unit.batchCount());
        attrs.put("cpu1Pool2dChannels", unit.channels());
        attrs.put("cpu1Pool2dInputH", unit.inputH());
        attrs.put("cpu1Pool2dInputW", unit.inputW());
        attrs.put("cpu1Pool2dOutputH", unit.outputH());
        attrs.put("cpu1Pool2dOutputW", unit.outputW());
        attrs.put("cpu1Pool2dKernelH", unit.kernelH());
        attrs.put("cpu1Pool2dKernelW", unit.kernelW());
        attrs.put("cpu1Pool2dStrideH", unit.strideH());
        attrs.put("cpu1Pool2dStrideW", unit.strideW());
        attrs.put("cpu1Pool2dPadH", unit.padH());
        attrs.put("cpu1Pool2dPadW", unit.padW());
        attrs.put("cpu1Pool2dCountIncludePad", unit.countIncludePad());
        attrs.put("cpu1Pool2dCeilMode", unit.options().ceilMode());
        attrs.put("cpu1Pool2dOutputElements", unit.outputElementCount());
        attrs.put("cpu1Pool2dInputAccessKind", unit.inputAccessPlan().kind().name());
        attrs.put("cpu1Pool2dOutputAccessKind", unit.outputAccessPlan().kind().name());
        attrs.put("cpu1Pool2dLaunchWorkers", unit.launchConfig().workerCount());
        attrs.put("cpu1Pool2dLaunchChunkSize", unit.launchConfig().chunkSize());
        DispatchTraceMetadata dispatch = new DispatchTraceMetadata(
                Cpu1VectorizationKind.SCALAR.name(),
                1,
                unit.launchConfig().workerCount(),
                unit.launchConfig().chunkSize(),
                unit.launchConfig().chunkSize()
        );
        ReductionTraceMetadata reduction = new ReductionTraceMetadata(
                unit.opType().name(),
                unit.launchConfig().workerCount(),
                unit.launchConfig().chunkSize(),
                1,
                unit.dataType() == DataType.BFLOAT16 ? "F32_ACCUMULATE" : unit.dataType().name()
        );
        return new StepTraceContribution(
                unit.kernelId().name(),
                attrs,
                null,
                null,
                dispatch,
                reduction,
                null,
                null,
                null
        );
    }

    private static StepTraceContribution conv2dTrace(Cpu1PreparedConv2dUnit unit) {
        LinkedHashMap<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("cpu1KernelId", unit.kernelId().name());
        attrs.put("cpu1Conv2dKernelId", unit.kernelId().name());
        attrs.put("cpu1Conv2dOpType", unit.opType().name());
        attrs.put("cpu1StorageKind", unit.storageKind().name());
        attrs.put("cpu1Conv2dDType", unit.dataType().name());
        attrs.put("cpu1Conv2dHasBias", unit.hasBias());
        attrs.put("cpu1Conv2dBatchCount", unit.batchCount());
        attrs.put("cpu1Conv2dInputChannels", unit.inChannels());
        attrs.put("cpu1Conv2dOutputChannels", unit.outChannels());
        attrs.put("cpu1Conv2dChannelsPerGroup", unit.channelsPerGroup());
        attrs.put("cpu1Conv2dOutputChannelsPerGroup", unit.outChannelsPerGroup());
        attrs.put("cpu1Conv2dGroups", unit.groups());
        attrs.put("cpu1Conv2dInputH", unit.inputH());
        attrs.put("cpu1Conv2dInputW", unit.inputW());
        attrs.put("cpu1Conv2dOutputH", unit.outputH());
        attrs.put("cpu1Conv2dOutputW", unit.outputW());
        attrs.put("cpu1Conv2dKernelH", unit.kernelH());
        attrs.put("cpu1Conv2dKernelW", unit.kernelW());
        attrs.put("cpu1Conv2dStrideH", unit.strideH());
        attrs.put("cpu1Conv2dStrideW", unit.strideW());
        attrs.put("cpu1Conv2dPadH", unit.padH());
        attrs.put("cpu1Conv2dPadW", unit.padW());
        attrs.put("cpu1Conv2dDilationH", unit.dilationH());
        attrs.put("cpu1Conv2dDilationW", unit.dilationW());
        attrs.put("cpu1Conv2dWork", unit.work());
        attrs.put("cpu1Conv2dOutputElements", unit.outputElementCount());
        attrs.put("cpu1Conv2dInputAccessKind", unit.inputAccessPlan().kind().name());
        attrs.put("cpu1Conv2dWeightAccessKind", unit.weightAccessPlan().kind().name());
        attrs.put("cpu1Conv2dBiasAccessKind", unit.biasAccessPlan() == null
                ? "NONE"
                : unit.biasAccessPlan().kind().name());
        attrs.put("cpu1Conv2dOutputAccessKind", unit.outputAccessPlan().kind().name());
        attrs.put("cpu1Conv2dLaunchWorkers", unit.launchConfig().workerCount());
        attrs.put("cpu1Conv2dLaunchChunkSize", unit.launchConfig().chunkSize());
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

package backend.metal.lowering;

import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;

import backend.ComputeBackend;
import backend.accelerator.dag.AcceleratorDagNodeType;
import backend.accelerator.dag.AcceleratorSubgraphOp;
import backend.accelerator.dag.AcceleratorSubgraphSpec;
import backend.accelerator.lowering.AcceleratorSubgraphLowerer;
import backend.accelerator.lowering.AcceleratorSubgraphLoweringResult;
import backend.accelerator.lowering.GpuCompoundLoweringArtifact;
import backend.accelerator.lowering.GpuCompoundPatternType;
import backend.accelerator.lowering.GpuLoweringCoverageMatrix;
import backend.lowering.BackendCapabilities;
import backend.lowering.LoweringContext;
import backend.lowering.LoweringRequest;
import backend.lowering.LoweringResult;
import config.optimizer.FuseConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledNode;
import graph.execution.trace.PartitionDecisionTrace;
import graph.optimizer.memory.MemoryPlanner;
import graph.optimizer.partition.Partition;
import graph.optimizer.partition.PartitionBoundaryReason;
import graph.optimizer.partition.PartitionCandidate;
import graph.optimizer.partition.PartitionEdge;
import graph.optimizer.partition.PartitionPlannerStrategy;
import graph.optimizer.partition.PartitionPlanningContext;
import graph.optimizer.partition.PartitionTarget;
import graph.optimizer.partition.PartitionValue;
import graph.optimizer.partition.PartitionValueRef;
import graph.optimizer.region.DefaultRegionOptimizer;
import graph.optimizer.region.ExecutionUnitKind;
import graph.optimizer.region.OptimizedRegion;
import graph.optimizer.region.RegionOptimizationContext;
import operations.Operation;
import operations.elementwise.unary.sign;
import operations.elementwise.where.where;
import operations.index.gatherAxisGrad;
import operations.index.gatherGrad;
import operations.index.takeAlongAxisGrad;
import operations.layout.sliceGrad;
import operations.nn.conv.conv2d;
import operations.nn.conv.conv2dGemm;
import operations.nn.pool.maxPool2d;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.TensorPrimitiveBuilder;
import tensor.options.AttentionOptions;
import tensor.options.Conv2dOptions;
import tensor.options.Pool2dOptions;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MetalRegionLowererTest {
    @Test
    void metalLowersLinearBiasReluAsLinearBiasActivationCompoundRegion() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "metalLinearInput", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{
                1f, 0f, 0f, 1f,
                0f, 1f, 1f, 0f,
                1f, 1f, 0f, 0f
        }, new int[]{3, 4}, null, "metalLinearWeight", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.5f, -0.5f, 1f, -1f}, new int[]{4}, null, "metalLinearBias", DataType.FLOAT32);
        Tensor linear = input.linear(weight, bias);
        Tensor out = linear.relu();
        TensorInternalAccess.setBackend(linear, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(graph);
        PartitionPlanningContext context = new PartitionPlanningContext(
                RuntimeConfig.inferenceDefaults(),
                false,
                compiledNodes,
                CompiledTensorDescriptorBuilder.build(compiledNodes),
                consumers(compiledNodes)
        );
        int linearNodeId = nodeId(context, Operation.OpType.LINEAR);
        int reluNodeId = nodeId(context, Operation.OpType.RELU);
        MetalRegionLegalityAdapter adapter = new MetalRegionLegalityAdapter();
        PartitionCandidate candidate = adapter.tryCreateStructuralCandidate(
                Set.of(linearNodeId, reluNodeId),
                context,
                Set.of(PartitionValueRef.ofNode(reluNodeId))
        );
        assertNotNull(candidate);
        MetalPartitionPlan attachedPlan = (MetalPartitionPlan) adapter.tryCreatePlan(candidate, context);

        assertNotNull(attachedPlan);
        assertEquals(GpuCompoundPatternType.LINEAR_BIAS_ACTIVATION, attachedPlan.lowering().compoundSummary().patternType());
        assertTrue(attachedPlan.lowering().compoundSummary().supported());
        assertTrue(attachedPlan.nodeIds().containsAll(List.of(linearNodeId, reluNodeId)));
        assertEquals(ComputeBackend.GPU_METAL, attachedPlan.manifest().backend());
        assertTrue(attachedPlan.manifest().selectedRegionLength() >= attachedPlan.nodeIds().size());
        assertTrue(attachedPlan.lowering().dagSpec().nodes().stream().anyMatch(node -> node.type() == AcceleratorDagNodeType.LINEAR));
        assertTrue(attachedPlan.lowering().dagSpec().nodes().stream().anyMatch(node -> node.type() == AcceleratorDagNodeType.RELU));
        assertTrue(attachedPlan.matMulSpec().biasInputNodeId() >= 0);
        assertTrue(attachedPlan.matMulSpec().postOps().stream()
                .anyMatch(postOp -> postOp.type() == backend.accelerator.dag.AcceleratorPostOpType.RELU));

        Partition partition = new Partition(
                "metal-linear-bias-activation",
                PartitionTarget.GPU_METAL,
                candidate.orderedNodeIds(),
                candidate.orderedNodeIds().stream().map(id -> new PartitionValue(PartitionValueRef.ofNode(id), id)).toList(),
                List.of(new PartitionEdge(linearNodeId, reluNodeId)),
                candidate.externalInputIds(),
                candidate.outputNodeIds().stream().map(PartitionValueRef::ofNode).toList(),
                candidate.anchorNodeId(),
                List.of(PartitionValueRef.ofNode(reluNodeId)),
                List.of(),
                List.of(PartitionBoundaryReason.NONE),
                attachedPlan.estimatedWork(),
                new graph.optimizer.partition.cost.AcceleratorPartitionScoreModel.CandidateMetrics(candidate.orderedNodeIds().size(), 1, candidate.externalInputIds().size(), 0, 1),
                PartitionPlannerStrategy.GREEDY_MAX_REGION,
                new PartitionDecisionTrace(
                        PartitionPlannerStrategy.GREEDY_MAX_REGION,
                        PartitionTarget.GPU_METAL,
                        candidate.anchorNodeId(),
                        true,
                        "test",
                        candidate.orderedNodeIds(),
                        candidate.orderedNodeIds(),
                        List.of("LINEAR", "RELU"),
                        attachedPlan.estimatedWork(),
                        0.0d,
                        0.0d,
                        0,
                        false,
                        -1
                )
        );
        OptimizedRegion region = new DefaultRegionOptimizer().optimize(partition, new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults()));

        LoweringResult result = new MetalRegionLowerer().lower(new LoweringRequest(
                region,
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.GPU_METAL)),
                new LoweringContext(RuntimeConfig.inferenceDefaults(), compiledNodes, CompiledTensorDescriptorBuilder.build(compiledNodes), java.util.Map.of(partition.partitionId(), attachedPlan))
        ));

        assertNotNull(result);
        GpuCompoundLoweringArtifact artifact = result.loweredRegion().units().getFirst().requireArtifact(GpuCompoundLoweringArtifact.class);
        assertEquals(GpuCompoundPatternType.LINEAR_BIAS_ACTIVATION, artifact.summary().patternType());
        assertTrue(artifact.summary().orderedNodeIds().containsAll(List.of(linearNodeId, reluNodeId)));
        assertTrue(artifact.units().stream().anyMatch(unit ->
                unit.kind() == ExecutionUnitKind.MATMUL_EPILOGUE
                        && unit.orderedNodeIds().containsAll(List.of(linearNodeId, reluNodeId))));
        assertTrue(artifact.units().stream()
                .flatMap(unit -> unit.traceEvents().stream())
                .anyMatch(event -> event.contains("lowering-decision:")));
    }

    @Test
    void phaseNineteenMetalLowererKeepsMultiOpRegionAsSingleGraphUnit() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "phase19MetalA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 0f, 0f, 1f, 1f, 1f}, new int[]{3, 2}, null, "phase19MetalB", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor relu = matmul.relu();
        Tensor out = relu.exp();
        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(relu, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);
        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(graph);
        PartitionPlanningContext planningContext = new PartitionPlanningContext(
                RuntimeConfig.inferenceDefaults(),
                false,
                compiledNodes,
                CompiledTensorDescriptorBuilder.build(compiledNodes),
                consumers(compiledNodes)
        );
        int matmulNodeId = nodeId(planningContext, Operation.OpType.MATMUL);
        int reluNodeId = nodeId(planningContext, Operation.OpType.RELU);
        int expNodeId = nodeId(planningContext, Operation.OpType.EXP);
        List<Integer> selectedNodeIds = List.of(matmulNodeId, reluNodeId, expNodeId);
        MetalRegionLegalityAdapter adapter = new MetalRegionLegalityAdapter();
        PartitionCandidate candidate = adapter.tryCreateStructuralCandidate(
                Set.copyOf(selectedNodeIds),
                planningContext,
                Set.of(PartitionValueRef.ofNode(expNodeId))
        );
        assertNotNull(candidate);
        MetalPartitionPlan attachedPlan = (MetalPartitionPlan) adapter.tryCreatePlan(candidate, planningContext);
        assertNotNull(attachedPlan);
        Partition partition = partition(
                "phase19-metal-multi-op",
                PartitionTarget.GPU_METAL,
                candidate.orderedNodeIds(),
                candidate.externalInputIds(),
                candidate.outputNodeIds().stream().map(PartitionValueRef::ofNode).toList(),
                List.of(PartitionValueRef.ofNode(expNodeId))
        );
        OptimizedRegion region = new DefaultRegionOptimizer().optimize(
                partition,
                new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults())
        );

        LoweringResult result = new MetalRegionLowerer().lower(new LoweringRequest(
                region,
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.GPU_METAL)),
                new LoweringContext(RuntimeConfig.inferenceDefaults(), compiledNodes, CompiledTensorDescriptorBuilder.build(compiledNodes), java.util.Map.of(partition.partitionId(), attachedPlan))
        ));

        assertNotNull(result);
        assertEquals(1, result.loweredRegion().units().size());
        assertEquals(backend.lowering.LoweringFamily.METAL_GRAPH_REGION, result.loweredRegion().units().getFirst().loweringFamily());
        assertTrue(attachedPlan.manifest().selectedRegionLength() > 1);
        assertTrue(attachedPlan.manifest().loweredPrimitives().size() > 1);
        assertTrue(result.loweredRegion().units().getFirst().orderedNodeIds().containsAll(selectedNodeIds));
        GpuCompoundLoweringArtifact artifact = result.loweredRegion().units().getFirst().requireArtifact(GpuCompoundLoweringArtifact.class);
        assertEquals(GpuCompoundPatternType.NONE, artifact.summary().patternType());
        assertTrue(artifact.units().stream().anyMatch(unit -> unit.kind() == ExecutionUnitKind.UNIT_KERNEL));
        assertTrue(artifact.units().stream().anyMatch(unit -> unit.kind() == ExecutionUnitKind.FUSED_ELEMENTWISE));
        assertTrue(artifact.units().stream()
                .flatMap(unit -> unit.traceEvents().stream())
                .anyMatch(event -> event.contains("KEEP_AS_BACKEND_PRIMITIVE")));
    }

    @Test
    void metalPlannerSupportMatchesSharedCoverageMatrixForForwardOps() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "metalMatrixA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "metalMatrixB", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor relu = matmul.relu();
        Tensor out = relu.exp();
        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(relu, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        for (Operation.OpType opType : List.of(Operation.OpType.MATMUL, Operation.OpType.RELU, Operation.OpType.EXP)) {
            assertTrue(GpuLoweringCoverageMatrix.isSupported(ComputeBackend.GPU_METAL, opType));
            assertEquals("", MetalPartitionSupport.plannerUnsupportedReason(context.compiledNode(nodeId(context, opType)), context));
        }
    }

    @Test
    void metalReductionIsPlannerSupported() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "metalReductionInput", DataType.FLOAT32);
        Tensor out = input.sum(1);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        String reason = MetalPartitionSupport.plannerUnsupportedReason(context.compiledNode(nodeId(context, Operation.OpType.SUM)), context);

        assertEquals("", reason);
    }

    @Test
    void metalReductionScanParityOpsArePlannerSupportedAndDagLowerable() {
        Tensor input = new Tensor(new float[]{1f, 5f, 5f, 4f, 2f, 3f}, new int[]{2, 3}, null, "metalReductionScanInput", DataType.FLOAT32);

        assertPlannerSupportedAndLowered(input.prod(1, true), Operation.OpType.REDUCE_PROD, AcceleratorDagNodeType.REDUCE_PROD);
        assertPlannerSupportedAndLowered(input.argMax(1, true), Operation.OpType.ARGMAX, AcceleratorDagNodeType.ARGMAX);
        assertPlannerSupportedAndLowered(input.cumSum(1, true, true), Operation.OpType.CUMSUM, AcceleratorDagNodeType.CUMSUM);
    }

    @Test
    void metalReductionScanParityOpsRejectUnsupportedDtypes() {
        Tensor intInput = new Tensor(new int[]{1, 2, 3, 4}, new int[]{2, 2}, null, "metalIntReductionScanInput", DataType.INT32);
        Tensor intCumSum = intInput.cumSum(1);
        TensorInternalAccess.setBackend(intCumSum, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(intCumSum);
        String reason = MetalPartitionSupport.plannerUnsupportedReason(context.compiledNode(nodeId(context, Operation.OpType.CUMSUM)), context);

        assertTrue(reason.contains("UNSUPPORTED_DTYPE"));
        assertTrue(reason.contains("operation CUMSUM cannot produce native INT32 output"));
    }

    @Test
    void metalCastSupportsScopedFloatBfloat16Pairs() {
        Tensor f32 = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "metalCastF32", DataType.FLOAT32);
        Tensor bf16Cast = f32.cast(DataType.BFLOAT16);
        assertPlannerSupportedAndLowered(bf16Cast, Operation.OpType.CAST, AcceleratorDagNodeType.CAST);

        Tensor bf16 = new Tensor(new double[]{1.0, 2.0, 3.0, 4.0}, new int[]{2, 2}, null, "metalCastBf16", DataType.BFLOAT16);
        Tensor f32Cast = bf16.cast(DataType.FLOAT32);
        assertPlannerSupportedAndLowered(f32Cast, Operation.OpType.CAST, AcceleratorDagNodeType.CAST);
    }

    @Test
    void metalCastRejectsUnsupportedPairsWithCastPolicyReason() {
        Tensor f32 = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "metalCastRejectF32", DataType.FLOAT32);
        Tensor intCast = f32.cast(DataType.INT32);
        TensorInternalAccess.setBackend(intCast, ComputeBackend.GPU_METAL);
        PartitionPlanningContext intContext = planningContext(intCast);
        String intReason = MetalPartitionSupport.plannerUnsupportedReason(
                intContext.compiledNode(nodeId(intContext, Operation.OpType.CAST)),
                intContext
        );
        assertTrue(intReason.contains("UNSUPPORTED_CAST_PAIR"));

        Tensor f64 = new Tensor(new double[]{1.0, 2.0, 3.0, 4.0}, new int[]{2, 2}, null, "metalCastRejectF64", DataType.FLOAT64);
        Tensor f64ToF32 = f64.cast(DataType.FLOAT32);
        TensorInternalAccess.setBackend(f64ToF32, ComputeBackend.GPU_METAL);
        PartitionPlanningContext f64Context = planningContext(f64ToF32);
        String f64Reason = MetalPartitionSupport.plannerUnsupportedReason(
                f64Context.compiledNode(nodeId(f64Context, Operation.OpType.CAST)),
                f64Context
        );
        assertTrue(f64Reason.contains("FLOAT64_UNSUPPORTED"));

        Tensor f32ToF64 = f32.cast(DataType.FLOAT64);
        TensorInternalAccess.setBackend(f32ToF64, ComputeBackend.GPU_METAL);
        PartitionPlanningContext f32ToF64Context = planningContext(f32ToF64);
        String f32ToF64Reason = MetalPartitionSupport.plannerUnsupportedReason(
                f32ToF64Context.compiledNode(nodeId(f32ToF64Context, Operation.OpType.CAST)),
                f32ToF64Context
        );
        assertTrue(f32ToF64Reason.contains("FLOAT64_UNSUPPORTED"));
    }

    @Test
    void metalUnaryMathParityOpsSupportScopedFloatingSubset() {
        Tensor input = new Tensor(new float[]{-1.25f, -0.0f, 0.25f, 2.75f}, new int[]{2, 2}, null, "metal70UnaryInput", DataType.FLOAT32);

        assertPlannerSupportedAndLowered(input.erf(), Operation.OpType.ERF, AcceleratorDagNodeType.ERF);
        assertPlannerSupportedAndLowered(input.floor(), Operation.OpType.FLOOR, AcceleratorDagNodeType.FLOOR);
        assertPlannerSupportedAndLowered(input.ceil(), Operation.OpType.CEIL, AcceleratorDagNodeType.CEIL);
        assertPlannerSupportedAndLowered(input.sign(), Operation.OpType.SIGN, AcceleratorDagNodeType.SIGN);

        Tensor bf16 = new Tensor(new double[]{-1.25, -0.0, 0.25, 2.75}, new int[]{2, 2}, null, "metal70UnaryBf16Input", DataType.BFLOAT16);
        assertPlannerSupportedAndLowered(bf16.erf(), Operation.OpType.ERF, AcceleratorDagNodeType.ERF);
        assertPlannerSupportedAndLowered(bf16.floor(), Operation.OpType.FLOOR, AcceleratorDagNodeType.FLOOR);
        assertPlannerSupportedAndLowered(bf16.ceil(), Operation.OpType.CEIL, AcceleratorDagNodeType.CEIL);
        assertPlannerSupportedAndLowered(bf16.sign(), Operation.OpType.SIGN, AcceleratorDagNodeType.SIGN);
    }

    @Test
    void metalUnaryMathParityOpsRejectUnsupportedDtypeAndShapeMismatch() {
        Tensor f64 = new Tensor(new double[]{-1.25, 0.25}, new int[]{2}, null, "metal70UnaryF64Input", DataType.FLOAT64);
        Tensor f64Sign = f64.sign();
        TensorInternalAccess.setBackend(f64Sign, ComputeBackend.GPU_METAL);
        PartitionPlanningContext f64Context = planningContext(f64Sign);
        assertContainsAll(
                MetalPartitionSupport.plannerUnsupportedReason(
                        f64Context.compiledNode(nodeId(f64Context, Operation.OpType.SIGN)),
                        f64Context
                ),
                "UNSUPPORTED_DTYPE",
                "FLOAT64"
        );

        Tensor intInput = new Tensor(new int[]{-1, 0, 2}, new int[]{3}, null, "metal70UnaryIntInput", DataType.INT32);
        Tensor intSign = TensorPrimitiveBuilder.unaryNoGrad(
                intInput,
                intInput.getShape(),
                new sign(),
                "metal70UnaryIntSign",
                DataType.INT32
        );
        TensorInternalAccess.setBackend(intSign, ComputeBackend.GPU_METAL);
        PartitionPlanningContext intContext = planningContext(intSign);
        assertContainsAll(
                MetalPartitionSupport.plannerUnsupportedReason(
                        intContext.compiledNode(nodeId(intContext, Operation.OpType.SIGN)),
                        intContext
                ),
                "UNSUPPORTED_DTYPE",
                "SIGN"
        );

        Tensor shapeMismatch = TensorPrimitiveBuilder.unaryNoGrad(
                new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "metal70UnaryShapeInput", DataType.FLOAT32),
                new int[]{2, 2},
                new sign(),
                "metal70UnaryShapeMismatch",
                DataType.FLOAT32
        );
        TensorInternalAccess.setBackend(shapeMismatch, ComputeBackend.GPU_METAL);
        PartitionPlanningContext shapeContext = planningContext(shapeMismatch);
        assertContainsAll(
                MetalPartitionSupport.plannerUnsupportedReason(
                        shapeContext.compiledNode(nodeId(shapeContext, Operation.OpType.SIGN)),
                        shapeContext
                ),
                "UNSUPPORTED_RANK_OR_SHAPE",
                "must preserve input shape"
        );
    }

    @Test
    void metalSupportedNormalizationUsesSharedCoverageReason() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "metalNormInput", DataType.FLOAT32);
        Tensor gamma = new Tensor(new float[]{1f, 1f}, new int[]{2}, null, "metalNormGamma", DataType.FLOAT32);
        Tensor beta = new Tensor(new float[]{0f, 0f}, new int[]{2}, null, "metalNormBeta", DataType.FLOAT32);
        Tensor out = input.layerNorm(gamma, beta, 1.0e-5);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        String reason = MetalPartitionSupport.plannerUnsupportedReason(context.compiledNode(nodeId(context, Operation.OpType.LAYER_NORM)), context);

        assertEquals("", reason);
    }

    private void assertPlannerSupportedAndLowered(Tensor out, Operation.OpType opType, AcceleratorDagNodeType dagType) {
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);
        PartitionPlanningContext context = planningContext(out);
        CompiledNode node = context.compiledNode(nodeId(context, opType));

        assertEquals("", MetalPartitionSupport.plannerUnsupportedReason(node, context));
        AcceleratorSubgraphSpec subgraph = new AcceleratorSubgraphSpec(
                node.id(),
                List.of(node.id()),
                List.of(new AcceleratorSubgraphOp(node.id(), opType)),
                node.inputIds(),
                List.of(node.id())
        );
        var lowering = new AcceleratorSubgraphLowerer().tryLower(ComputeBackend.GPU_METAL, subgraph, context);

        assertNotNull(lowering);
        assertEquals(dagType, lowering.dagSpec().nodes().getFirst().type());
        assertEquals(node.dataType(), lowering.dagSpec().nodes().getFirst().outputDataType());
    }

    @Test
    void metalGpuFusedOpTypeRejectsWithStableCompoundReason() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "metalFusedInput", DataType.FLOAT32);
        Tensor out = TensorPrimitiveBuilder.unary(input, new SyntheticFusedOperation(), "metalCpuFusedOp", DataType.FLOAT32);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        String reason = MetalPartitionSupport.plannerUnsupportedReason(context.compiledNode(nodeId(context, Operation.OpType.FUSED)), context);

        assertTrue(reason.contains("CPU_FUSED_OPERATION_UNSUPPORTED"));
        assertTrue(reason.contains("operation FUSED is not supported by GPU_METAL lowering"));
    }

    @Test
    void metalIndexTargetLossIsPlannerSupported() {
        Tensor logits = new Tensor(new float[]{1f, 2f, 3f, 1f, 0f, -1f}, new int[]{2, 3}, null, "metalLossLogits", DataType.FLOAT32);
        Tensor targetIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "metalLossTargets", DataType.INT32);
        Tensor out = logits.crossEntropyLossFromIndices(targetIndices, 1);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        String reason = MetalPartitionSupport.plannerUnsupportedReason(context.compiledNode(nodeId(context, Operation.OpType.CROSS_ENTROPY_LOSS_INDICES)), context);

        assertEquals("", reason);
    }

    @Test
    void metalPhaseSeventeenReductionAndNormReasonsIncludeMatrixDetail() {
        Tensor reductionInput = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "metalPhase17ReductionInput", DataType.FLOAT32);
        Tensor sum = reductionInput.sum(1);
        Tensor mean = reductionInput.mean(1);
        TensorInternalAccess.setBackend(sum, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(mean, ComputeBackend.GPU_METAL);
        PartitionPlanningContext sumContext = planningContext(sum);
        PartitionPlanningContext meanContext = planningContext(mean);

        assertEquals("", MetalPartitionSupport.plannerUnsupportedReason(sumContext.compiledNode(nodeId(sumContext, Operation.OpType.SUM)), sumContext));
        assertEquals("", MetalPartitionSupport.plannerUnsupportedReason(meanContext.compiledNode(nodeId(meanContext, Operation.OpType.MEAN)), meanContext));

        Tensor normInput = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "metalPhase17NormInput", DataType.FLOAT32);
        Tensor gamma = new Tensor(new float[]{1f, 1f}, new int[]{2}, null, "metalPhase17NormGamma", DataType.FLOAT32);
        Tensor beta = new Tensor(new float[]{0f, 0f}, new int[]{2}, null, "metalPhase17NormBeta", DataType.FLOAT32);
        Tensor layerNorm = normInput.layerNorm(gamma, beta, 1.0e-5);
        Tensor rmsNorm = normInput.rmsNorm(gamma, 1.0e-5);
        TensorInternalAccess.setBackend(layerNorm, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(rmsNorm, ComputeBackend.GPU_METAL);
        PartitionPlanningContext layerNormContext = planningContext(layerNorm);
        PartitionPlanningContext rmsNormContext = planningContext(rmsNorm);

        String layerNormReason = MetalPartitionSupport.plannerUnsupportedReason(layerNormContext.compiledNode(nodeId(layerNormContext, Operation.OpType.LAYER_NORM)), layerNormContext);
        String rmsNormReason = MetalPartitionSupport.plannerUnsupportedReason(rmsNormContext.compiledNode(nodeId(rmsNormContext, Operation.OpType.RMS_NORM)), rmsNormContext);

        assertEquals("", layerNormReason);
        assertEquals("", rmsNormReason);
    }

    @Test
    void metalPhaseSeventeenConvAndLossReasonsIncludeMatrixDetail() {
        Tensor input = new Tensor(
                new float[]{
                        1f, 2f, 3f,
                        4f, 5f, 6f,
                        7f, 8f, 9f
                },
                new int[]{1, 1, 3, 3},
                null,
                "metalPhase17ConvInput",
                DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 1, 2, 2}, null, "metalPhase17ConvWeight", DataType.FLOAT32);
        Tensor conv = input.conv2d(weight, Conv2dOptions.defaults());
        TensorInternalAccess.setBackend(conv, ComputeBackend.GPU_METAL);

        PartitionPlanningContext convContext = planningContext(conv);
        String convReason = MetalPartitionSupport.plannerUnsupportedReason(convContext.compiledNode(nodeId(convContext, Operation.OpType.CONV2D)), convContext);

        assertEquals("", convReason);

        Tensor logits = new Tensor(new float[]{1f, 2f, 3f, 1f, 0f, -1f}, new int[]{2, 3}, null, "metalPhase17LossLogits", DataType.FLOAT32);
        Tensor targetIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "metalPhase17LossTargets", DataType.INT32);
        Tensor loss = logits.crossEntropyLossFromIndices(targetIndices, 1);
        TensorInternalAccess.setBackend(loss, ComputeBackend.GPU_METAL);

        PartitionPlanningContext lossContext = planningContext(loss);
        String lossReason = MetalPartitionSupport.plannerUnsupportedReason(lossContext.compiledNode(nodeId(lossContext, Operation.OpType.CROSS_ENTROPY_LOSS_INDICES)), lossContext);

        assertEquals("", lossReason);
    }

    @Test
    void acceptsLogSoftmaxAsSoftmaxLikeGpuRegion() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "metalLogSoftmaxInput", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f}, new int[]{3, 3}, null, "metalLogSoftmaxWeight", DataType.FLOAT32);
        Tensor matmul = input.matmul(weight);
        Tensor out = specialLogSoftmax(matmul, 1);
        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        int matmulNodeId = nodeId(context, Operation.OpType.MATMUL);
        int logSoftmaxNodeId = nodeId(context, Operation.OpType.LOG_SOFTMAX);
        MetalRegionLegalityAdapter adapter = new MetalRegionLegalityAdapter();
        PartitionCandidate candidate = adapter.tryCreateStructuralCandidate(
                Set.of(matmulNodeId, logSoftmaxNodeId),
                context,
                Set.of(PartitionValueRef.ofNode(logSoftmaxNodeId))
        );
        assertNotNull(candidate);

        MetalPartitionPlan plan = (MetalPartitionPlan) adapter.tryCreatePlan(candidate, context);

        assertNotNull(plan);
        List<AcceleratorDagNodeType> types = plan.lowering().dagSpec().nodes().stream()
                .map(backend.accelerator.dag.AcceleratorDagNode::type)
                .toList();
        assertTrue(types.contains(AcceleratorDagNodeType.SOFTMAX));
        assertTrue(types.contains(AcceleratorDagNodeType.LOG));
    }

    @Test
    void supportsSumReductionWithStableCoverageReason() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "metalStableReductionInput", DataType.FLOAT32);
        Tensor out = input.sum(1);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        String reason = MetalPartitionSupport.plannerUnsupportedReason(context.compiledNode(nodeId(context, Operation.OpType.SUM)), context);

        assertEquals("", reason);
    }

    @Test
    void acceptsLayerNormWithStableCoverageReason() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "metalStableNormInput", DataType.FLOAT32);
        Tensor gamma = new Tensor(new float[]{1f, 1f}, new int[]{2}, null, "metalStableNormGamma", DataType.FLOAT32);
        Tensor beta = new Tensor(new float[]{0f, 0f}, new int[]{2}, null, "metalStableNormBeta", DataType.FLOAT32);
        Tensor out = input.layerNorm(gamma, beta, 1.0e-5);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        String reason = MetalPartitionSupport.plannerUnsupportedReason(context.compiledNode(nodeId(context, Operation.OpType.LAYER_NORM)), context);

        assertEquals("", reason);
    }

    @Test
    void acceptsCrossEntropyLossWithIndexTargets() {
        Tensor logits = new Tensor(new float[]{1f, 2f, 3f, 1f, 0f, -1f}, new int[]{2, 3}, null, "metalStableLossLogits", DataType.FLOAT32);
        Tensor targetIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "metalStableLossTargets", DataType.INT32);
        Tensor out = logits.crossEntropyLossFromIndices(targetIndices, 1);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        String reason = MetalPartitionSupport.plannerUnsupportedReason(context.compiledNode(nodeId(context, Operation.OpType.CROSS_ENTROPY_LOSS_INDICES)), context);

        assertEquals("", reason);
    }

    @Test
    void phaseThirtySevenDenseLossContractAdmitsScopedDenseLosses() {
        Tensor logProbs = new Tensor(new float[]{
                -0.1f, -2.0f, -3.0f,
                -1.5f, -0.3f, -2.5f
        }, new int[]{2, 3}, null, "metalDenseNllLogProbs", DataType.FLOAT32);
        Tensor nllTargets = new Tensor(new float[]{
                1f, 0f, 0f,
                0f, 1f, 0f
        }, new int[]{2, 3}, null, "metalDenseNllTargets", DataType.FLOAT32);
        Tensor nll = logProbs.nllLoss(nllTargets, 1);
        TensorInternalAccess.setBackend(nll, ComputeBackend.GPU_METAL);
        PartitionPlanningContext nllContext = planningContext(nll);
        String nllReason = MetalPartitionSupport.plannerUnsupportedReason(
                nllContext.compiledNode(nodeId(nllContext, Operation.OpType.NLL_LOSS)),
                nllContext
        );

        Tensor logits = new Tensor(new float[]{
                1f, 2f, 3f,
                1f, 0f, -1f
        }, new int[]{2, 3}, null, "metalDenseCeLogits", DataType.FLOAT32);
        Tensor ceTargets = new Tensor(new float[]{
                0f, 0f, 1f,
                1f, 0f, 0f
        }, new int[]{2, 3}, null, "metalDenseCeTargets", DataType.FLOAT32);
        Tensor ce = logits.crossEntropyLoss(ceTargets, 1);
        TensorInternalAccess.setBackend(ce, ComputeBackend.GPU_METAL);
        PartitionPlanningContext ceContext = planningContext(ce);
        String ceReason = MetalPartitionSupport.plannerUnsupportedReason(
                ceContext.compiledNode(nodeId(ceContext, Operation.OpType.CROSS_ENTROPY_LOSS)),
                ceContext
        );

        assertEquals("", nllReason);
        assertEquals("", ceReason);
        List<AcceleratorDagNodeType> nllTypes = planFor(nll, Operation.OpType.NLL_LOSS).lowering().dagSpec().nodes().stream()
                .map(backend.accelerator.dag.AcceleratorDagNode::type)
                .toList();
        List<AcceleratorDagNodeType> ceTypes = planFor(ce, Operation.OpType.CROSS_ENTROPY_LOSS).lowering().dagSpec().nodes().stream()
                .map(backend.accelerator.dag.AcceleratorDagNode::type)
                .toList();
        assertTrue(nllTypes.contains(AcceleratorDagNodeType.MUL));
        assertTrue(nllTypes.contains(AcceleratorDagNodeType.SUM));
        assertTrue(nllTypes.contains(AcceleratorDagNodeType.MUL_SCALAR));
        assertTrue(ceTypes.contains(AcceleratorDagNodeType.SOFTMAX));
        assertTrue(ceTypes.contains(AcceleratorDagNodeType.LOG));
        assertTrue(ceTypes.contains(AcceleratorDagNodeType.MUL));
        assertTrue(ceTypes.contains(AcceleratorDagNodeType.SUM));
        assertTrue(ceTypes.contains(AcceleratorDagNodeType.MUL_SCALAR));

        Tensor bf16Logits = new Tensor(new double[]{
                1d, 2d, 3d,
                1d, 0d, -1d
        }, new int[]{2, 3}, null, "metalDenseBf16CeLogits", DataType.BFLOAT16);
        Tensor bf16Targets = new Tensor(new double[]{
                0d, 0d, 1d,
                1d, 0d, 0d
        }, new int[]{2, 3}, null, "metalDenseBf16CeTargets", DataType.BFLOAT16);
        Tensor bf16Ce = bf16Logits.crossEntropyLoss(bf16Targets, 1);
        TensorInternalAccess.setBackend(bf16Ce, ComputeBackend.GPU_METAL);
        PartitionPlanningContext bf16CeContext = planningContext(bf16Ce);
        assertEquals(
                "",
                MetalPartitionSupport.plannerUnsupportedReason(
                        bf16CeContext.compiledNode(nodeId(bf16CeContext, Operation.OpType.CROSS_ENTROPY_LOSS)),
                        bf16CeContext
                )
        );
        assertTrue(planFor(bf16Ce, Operation.OpType.CROSS_ENTROPY_LOSS).lowering().dagSpec().nodes().stream()
                .allMatch(node -> node.outputDataType() == DataType.BFLOAT16));
    }

    @Test
    void phaseThirtySevenDenseLossContractRejectsNonDenseInputsBeforePendingExecution() {
        Tensor logits = new Tensor(new float[]{
                1f, 2f, 3f,
                1f, 0f, -1f
        }, new int[]{2, 3}, null, "metalDenseLossLayoutLogits", DataType.FLOAT32);
        Tensor targets = new Tensor(new float[]{
                0f, 0f, 1f,
                1f, 0f, 0f
        }, new int[]{2, 3}, null, "metalDenseLossLayoutTargets", DataType.FLOAT32);
        Tensor out = logits.select(0, 1).crossEntropyLoss(targets.select(0, 1), 0);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        String reason = MetalPartitionSupport.plannerUnsupportedReason(
                context.compiledNode(nodeId(context, Operation.OpType.CROSS_ENTROPY_LOSS)),
                context
        );

        assertEquals("UNSUPPORTED_LAYOUT: GPU_METAL dense CROSS_ENTROPY_LOSS inputs require dense zero-offset layout", reason);
    }

    @Test
    void phaseThirtyTwoIndexFamilySupportsForwardGatherTakeAndScatterAdd() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "metalPhase26IndexInput", DataType.FLOAT32);
        Tensor gatherIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "metalPhase26GatherIndices", DataType.INT32);
        Tensor gather = input.gather(gatherIndices, 1);
        TensorInternalAccess.setBackend(gather, ComputeBackend.GPU_METAL);
        PartitionPlanningContext gatherContext = planningContext(gather);
        String gatherReason = MetalPartitionSupport.plannerUnsupportedReason(gatherContext.compiledNode(nodeId(gatherContext, Operation.OpType.GATHER)), gatherContext);

        Tensor takeIndices = new Tensor(new int[]{2, 1, 0, 0}, new int[]{2, 2}, null, "metalPhase26TakeIndices", DataType.INT32);
        Tensor take = input.takeAlongAxis(takeIndices, 1);
        TensorInternalAccess.setBackend(take, ComputeBackend.GPU_METAL);
        PartitionPlanningContext takeContext = planningContext(take);
        String takeReason = MetalPartitionSupport.plannerUnsupportedReason(takeContext.compiledNode(nodeId(takeContext, Operation.OpType.TAKE_ALONG_AXIS)), takeContext);

        Tensor base = new Tensor(new float[]{10f, 20f, 30f, 40f, 50f, 60f}, new int[]{2, 3}, null, "metalPhase26ScatterBase", DataType.FLOAT32);
        Tensor scatterIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "metalPhase26ScatterIndices", DataType.INT32);
        Tensor src = new Tensor(new float[]{1f, 5f}, new int[]{2}, null, "metalPhase26ScatterSrc", DataType.FLOAT32);
        Tensor scatter = base.scatterAdd(scatterIndices, src, 1);
        TensorInternalAccess.setBackend(scatter, ComputeBackend.GPU_METAL);
        PartitionPlanningContext scatterContext = planningContext(scatter);
        String scatterReason = MetalPartitionSupport.plannerUnsupportedReason(scatterContext.compiledNode(nodeId(scatterContext, Operation.OpType.SCATTER_ADD)), scatterContext);

        assertEquals("", gatherReason);
        assertEquals("", takeReason);
        assertTrue(planFor(gather, Operation.OpType.GATHER).lowering().dagSpec().nodes().stream()
                .anyMatch(node -> node.type() == AcceleratorDagNodeType.GATHER && node.scalarValueBits() == 1));
        assertTrue(planFor(take, Operation.OpType.TAKE_ALONG_AXIS).lowering().dagSpec().nodes().stream()
                .anyMatch(node -> node.type() == AcceleratorDagNodeType.TAKE_ALONG_AXIS && node.scalarValueBits() == 1));
        assertEquals("", scatterReason);
        assertTrue(planFor(scatter, Operation.OpType.SCATTER_ADD).lowering().dagSpec().nodes().stream()
                .anyMatch(node -> node.type() == AcceleratorDagNodeType.SCATTER_ADD && node.scalarValueBits() == 1));

        Tensor bf16Input = new Tensor(new double[]{1d, 2d, 3d, 4d, 5d, 6d}, new int[]{2, 3}, null, "metalPhase32Bf16IndexInput", DataType.BFLOAT16);
        Tensor bf16Gather = bf16Input.gather(gatherIndices, 1);
        TensorInternalAccess.setBackend(bf16Gather, ComputeBackend.GPU_METAL);
        PartitionPlanningContext bf16GatherContext = planningContext(bf16Gather);
        assertEquals(
                "",
                MetalPartitionSupport.plannerUnsupportedReason(
                        bf16GatherContext.compiledNode(nodeId(bf16GatherContext, Operation.OpType.GATHER)),
                        bf16GatherContext
                )
        );
        assertTrue(planFor(bf16Gather, Operation.OpType.GATHER).lowering().dagSpec().nodes().stream()
                .anyMatch(node -> node.type() == AcceleratorDagNodeType.GATHER && node.outputDataType() == DataType.BFLOAT16));
    }

    @Test
    void metalLayoutIndexParityWaveSupportsGatherAxisAndStaticLayoutOps() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "metal68GatherAxisInput", DataType.FLOAT32);
        Tensor indices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "metal68GatherAxisIndices", DataType.INT32);
        Tensor gatherAxis = input.gatherAxis(indices, 1);
        TensorInternalAccess.setBackend(gatherAxis, ComputeBackend.GPU_METAL);
        PartitionPlanningContext gatherAxisContext = planningContext(gatherAxis);

        assertEquals("", MetalPartitionSupport.plannerUnsupportedReason(
                gatherAxisContext.compiledNode(nodeId(gatherAxisContext, Operation.OpType.GATHER_AXIS)),
                gatherAxisContext
        ));
        assertTrue(planFor(gatherAxis, Operation.OpType.GATHER_AXIS).lowering().dagSpec().nodes().stream()
                .anyMatch(node -> node.type() == AcceleratorDagNodeType.GATHER_AXIS && node.scalarValueBits() == 1));

        Tensor outGrad = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "metal68GatherAxisOutGrad", DataType.FLOAT32);
        Tensor gatherAxisGrad = TensorPrimitiveBuilder.binary(
                indices,
                outGrad,
                new int[]{2, 3},
                new gatherAxisGrad(1, new int[]{2, 3}),
                "metal68GatherAxisGrad",
                DataType.FLOAT32
        );
        TensorInternalAccess.setBackend(gatherAxisGrad, ComputeBackend.GPU_METAL);
        PartitionPlanningContext gatherAxisGradContext = planningContext(gatherAxisGrad);
        assertEquals("", MetalPartitionSupport.plannerUnsupportedReason(
                gatherAxisGradContext.compiledNode(nodeId(gatherAxisGradContext, Operation.OpType.GATHER_AXIS_GRAD)),
                gatherAxisGradContext
        ));
        assertTrue(planFor(gatherAxisGrad, Operation.OpType.GATHER_AXIS_GRAD).lowering().dagSpec().nodes().stream()
                .anyMatch(node -> node.type() == AcceleratorDagNodeType.GATHER_AXIS_GRAD && node.scalarValueBits() == 1));

        Tensor slice = input.slice(new int[]{0, 1}, new int[]{2, 3}, new int[]{0, 1}, new int[]{1, 1});
        TensorInternalAccess.setBackend(slice, ComputeBackend.GPU_METAL);
        PartitionPlanningContext sliceContext = planningContext(slice);
        assertEquals("", MetalPartitionSupport.plannerUnsupportedReason(
                sliceContext.compiledNode(nodeId(sliceContext, Operation.OpType.SLICE)),
                sliceContext
        ));
        assertTrue(planFor(slice, Operation.OpType.SLICE).lowering().dagSpec().nodes().stream()
                .anyMatch(node -> node.type() == AcceleratorDagNodeType.SLICE && node.attribute1() == 1));

        Tensor pad = input.pad(new int[]{1, 0}, new int[]{0, 1}, -1.0);
        TensorInternalAccess.setBackend(pad, ComputeBackend.GPU_METAL);
        PartitionPlanningContext padContext = planningContext(pad);
        assertEquals("", MetalPartitionSupport.plannerUnsupportedReason(
                padContext.compiledNode(nodeId(padContext, Operation.OpType.PAD)),
                padContext
        ));
        assertTrue(planFor(pad, Operation.OpType.PAD).lowering().dagSpec().nodes().stream()
                .anyMatch(node -> node.type() == AcceleratorDagNodeType.PAD
                        && node.attribute0() == 1
                        && node.attribute5() == 1
                        && Float.intBitsToFloat(node.scalarValueBits()) == -1.0f));

        Tensor tile = input.tile(2, 1);
        TensorInternalAccess.setBackend(tile, ComputeBackend.GPU_METAL);
        PartitionPlanningContext tileContext = planningContext(tile);
        assertEquals("", MetalPartitionSupport.plannerUnsupportedReason(
                tileContext.compiledNode(nodeId(tileContext, Operation.OpType.TILE)),
                tileContext
        ));
        assertTrue(planFor(tile, Operation.OpType.TILE).lowering().dagSpec().nodes().stream()
                .anyMatch(node -> node.type() == AcceleratorDagNodeType.TILE && node.attribute0() == 2 && node.attribute1() == 1));

        Tensor concatLeft = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "metal68ConcatLeft", DataType.FLOAT32);
        Tensor concatRight = new Tensor(new float[]{7f, 8f, 9f, 10f}, new int[]{2, 2}, null, "metal68ConcatRight", DataType.FLOAT32);
        Tensor concat = Tensor.concat(1, concatLeft, concatRight);
        TensorInternalAccess.setBackend(concat, ComputeBackend.GPU_METAL);
        PartitionPlanningContext concatContext = planningContext(concat);
        assertEquals("", MetalPartitionSupport.plannerUnsupportedReason(
                concatContext.compiledNode(nodeId(concatContext, Operation.OpType.CONCAT)),
                concatContext
        ));
        assertTrue(planFor(concat, Operation.OpType.CONCAT).lowering().dagSpec().nodes().stream()
                .anyMatch(node -> node.type() == AcceleratorDagNodeType.CONCAT && node.scalarValueBits() == 1));
    }

    @Test
    void metalSliceGradSupportsStaticStepOnePadBasedSubset() {
        Tensor outGrad = new Tensor(new float[]{10f, 20f, 30f, 40f}, new int[]{2, 2}, null, "metal72SliceGradOutGrad", DataType.FLOAT32);
        Tensor grad = TensorPrimitiveBuilder.unaryNoGrad(
                outGrad,
                new int[]{2, 4},
                new sliceGrad(new int[]{0, 1}, new int[]{0, 1}, new int[]{1, 1}, new int[]{2, 4}),
                "metal72SliceGrad",
                DataType.FLOAT32
        );
        TensorInternalAccess.setBackend(grad, ComputeBackend.GPU_METAL);
        PartitionPlanningContext context = planningContext(grad);

        assertEquals("", MetalPartitionSupport.plannerUnsupportedReason(
                context.compiledNode(nodeId(context, Operation.OpType.SLICE_GRAD)),
                context
        ));
        assertTrue(planFor(grad, Operation.OpType.SLICE_GRAD).lowering().dagSpec().nodes().stream()
                .anyMatch(node -> node.type() == AcceleratorDagNodeType.SLICE_GRAD
                        && node.attribute0() == 0
                        && node.attribute1() == 1
                        && node.attribute4() == 0
                        && node.attribute5() == 1));
    }

    @Test
    void metalSliceGradCanLowerAfterElementwiseProducerWithoutCpuBoundary() {
        Tensor outGrad = new Tensor(new float[]{-1f, 20f, 30f, -4f}, new int[]{2, 2}, null, "metal72SliceGradChainOutGrad", DataType.FLOAT32);
        Tensor relu = outGrad.relu();
        Tensor grad = TensorPrimitiveBuilder.unaryNoGrad(
                relu,
                new int[]{2, 4},
                new sliceGrad(new int[]{0, 1}, new int[]{0, 1}, new int[]{1, 1}, new int[]{2, 4}),
                "metal72SliceGradChain",
                DataType.FLOAT32
        );
        PartitionPlanningContext context = planningContext(grad);
        int reluNodeId = nodeId(context, Operation.OpType.RELU);
        int sliceGradNodeId = nodeId(context, Operation.OpType.SLICE_GRAD);
        List<Integer> selectedNodeIds = List.of(reluNodeId, sliceGradNodeId);

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(
                ComputeBackend.GPU_METAL,
                new AcceleratorSubgraphSpec(
                        reluNodeId,
                        selectedNodeIds,
                        List.of(
                                new AcceleratorSubgraphOp(reluNodeId, Operation.OpType.RELU),
                                new AcceleratorSubgraphOp(sliceGradNodeId, Operation.OpType.SLICE_GRAD)
                        ),
                        externalInputNodeIds(context, selectedNodeIds),
                        List.of(sliceGradNodeId)
                ),
                context
        );

        assertNotNull(result);
        assertTrue(result.dagSpec().nodes().stream().anyMatch(node -> node.type() == AcceleratorDagNodeType.RELU));
        assertTrue(result.dagSpec().nodes().stream().anyMatch(node -> node.type() == AcceleratorDagNodeType.SLICE_GRAD));
        assertEquals(List.of(sliceGradNodeId), result.dagSpec().outputNodeIds());
    }

    @Test
    void metalSliceGradRejectsStridedAndUnsupportedDtypeSubsets() {
        Tensor outGrad = new Tensor(new float[]{10f, 20f}, new int[]{2}, null, "metal72SliceGradRejectOutGrad", DataType.FLOAT32);
        Tensor stridedGrad = TensorPrimitiveBuilder.unaryNoGrad(
                outGrad,
                new int[]{5},
                new sliceGrad(new int[]{1}, new int[]{0}, new int[]{2}, new int[]{5}),
                "metal72SliceGradRejectStep",
                DataType.FLOAT32
        );
        TensorInternalAccess.setBackend(stridedGrad, ComputeBackend.GPU_METAL);
        PartitionPlanningContext stepContext = planningContext(stridedGrad);
        assertContainsAll(
                MetalPartitionSupport.plannerUnsupportedReason(
                        stepContext.compiledNode(nodeId(stepContext, Operation.OpType.SLICE_GRAD)),
                        stepContext
                ),
                "UNSUPPORTED_RANK_OR_SHAPE",
                "SLICE_GRAD supports step=1 only"
        );

        Tensor intOutGrad = new Tensor(new int[]{1, 2}, new int[]{2}, null, "metal72SliceGradRejectIntOutGrad", DataType.INT32);
        Tensor intGrad = TensorPrimitiveBuilder.unaryNoGrad(
                intOutGrad,
                new int[]{4},
                new sliceGrad(new int[]{1}, new int[]{0}, new int[]{1}, new int[]{4}),
                "metal72SliceGradRejectInt",
                DataType.INT32
        );
        TensorInternalAccess.setBackend(intGrad, ComputeBackend.GPU_METAL);
        PartitionPlanningContext dtypeContext = planningContext(intGrad);
        assertContainsAll(
                MetalPartitionSupport.plannerUnsupportedReason(
                        dtypeContext.compiledNode(nodeId(dtypeContext, Operation.OpType.SLICE_GRAD)),
                        dtypeContext
                ),
                "UNSUPPORTED_DTYPE",
                "SLICE_GRAD"
        );
    }

    @Test
    void metalLayoutIndexParityWaveRejectsUnsupportedLayoutSubsetsExplicitly() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "metal68LayoutRejectInput", DataType.FLOAT32);

        Tensor steppedSlice = input.slice(new int[]{0, 0}, new int[]{2, 3}, new int[]{0, 1}, new int[]{1, 2});
        TensorInternalAccess.setBackend(steppedSlice, ComputeBackend.GPU_METAL);
        PartitionPlanningContext steppedContext = planningContext(steppedSlice);
        assertContainsAll(
                MetalPartitionSupport.plannerUnsupportedReason(
                        steppedContext.compiledNode(nodeId(steppedContext, Operation.OpType.SLICE)),
                        steppedContext
                ),
                "UNSUPPORTED_RANK_OR_SHAPE",
                "SLICE supports step=1 only"
        );

        Tensor nonDense = input.permute(1, 0);
        Tensor layoutPad = nonDense.pad(new int[]{0, 0}, new int[]{1, 1}, 0.0);
        TensorInternalAccess.setBackend(layoutPad, ComputeBackend.GPU_METAL);
        PartitionPlanningContext layoutPadContext = planningContext(layoutPad);
        assertContainsAll(
                MetalPartitionSupport.plannerUnsupportedReason(
                        layoutPadContext.compiledNode(nodeId(layoutPadContext, Operation.OpType.PAD)),
                        layoutPadContext
                ),
                "UNSUPPORTED_LAYOUT",
                "PAD input requires dense layout"
        );
    }

    @Test
    void phaseThirtySixIndexGradientOpsLowerToMetalScatterPrimitives() {
        Tensor indices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "metalPhase36GradIndices", DataType.INT32);
        Tensor outGrad = new Tensor(new float[]{1f, 2f}, new int[]{2}, null, "metalPhase36GatherOutGrad", DataType.FLOAT32);
        Tensor gatherGrad = TensorPrimitiveBuilder.binary(
                indices,
                outGrad,
                new int[]{2, 3},
                new gatherGrad(1),
                "metalPhase36GatherGrad",
                DataType.FLOAT32
        );
        TensorInternalAccess.setBackend(gatherGrad, ComputeBackend.GPU_METAL);
        PartitionPlanningContext gatherGradContext = planningContext(gatherGrad);

        Tensor takeIndices = new Tensor(new int[]{2, 2, 0, 0}, new int[]{2, 2}, null, "metalPhase36TakeGradIndices", DataType.INT32);
        Tensor takeOutGrad = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "metalPhase36TakeOutGrad", DataType.FLOAT32);
        Tensor takeGrad = TensorPrimitiveBuilder.binary(
                takeIndices,
                takeOutGrad,
                new int[]{2, 3},
                new takeAlongAxisGrad(1),
                "metalPhase36TakeGrad",
                DataType.FLOAT32
        );
        TensorInternalAccess.setBackend(takeGrad, ComputeBackend.GPU_METAL);
        PartitionPlanningContext takeGradContext = planningContext(takeGrad);

        assertEquals("", MetalPartitionSupport.plannerUnsupportedReason(
                gatherGradContext.compiledNode(nodeId(gatherGradContext, Operation.OpType.GATHER_GRAD)),
                gatherGradContext
        ));
        assertEquals("", MetalPartitionSupport.plannerUnsupportedReason(
                takeGradContext.compiledNode(nodeId(takeGradContext, Operation.OpType.TAKE_ALONG_AXIS_GRAD)),
                takeGradContext
        ));
        assertTrue(planFor(gatherGrad, Operation.OpType.GATHER_GRAD).lowering().dagSpec().nodes().stream()
                .anyMatch(node -> node.type() == AcceleratorDagNodeType.GATHER_GRAD && node.scalarValueBits() == 1));
        assertTrue(planFor(takeGrad, Operation.OpType.TAKE_ALONG_AXIS_GRAD).lowering().dagSpec().nodes().stream()
                .anyMatch(node -> node.type() == AcceleratorDagNodeType.TAKE_ALONG_AXIS_GRAD && node.scalarValueBits() == 1));

        Tensor bf16OutGrad = new Tensor(new double[]{1d, 2d}, new int[]{2}, null, "metalPhase36Bf16GatherOutGrad", DataType.BFLOAT16);
        Tensor bf16GatherGrad = TensorPrimitiveBuilder.binary(
                indices,
                bf16OutGrad,
                new int[]{2, 3},
                new gatherGrad(1),
                "metalPhase36Bf16GatherGrad",
                DataType.BFLOAT16
        );
        TensorInternalAccess.setBackend(bf16GatherGrad, ComputeBackend.GPU_METAL);
        PartitionPlanningContext bf16GatherGradContext = planningContext(bf16GatherGrad);
        assertEquals(
                "",
                MetalPartitionSupport.plannerUnsupportedReason(
                        bf16GatherGradContext.compiledNode(nodeId(bf16GatherGradContext, Operation.OpType.GATHER_GRAD)),
                        bf16GatherGradContext
                )
        );
        assertTrue(planFor(bf16GatherGrad, Operation.OpType.GATHER_GRAD).lowering().dagSpec().nodes().stream()
                .anyMatch(node -> node.type() == AcceleratorDagNodeType.GATHER_GRAD && node.outputDataType() == DataType.BFLOAT16));
    }

    @Test
    void phaseThirtySixIndexWriteRejectsDtypeLayoutAndBoundsBeforeDuplicateBlocker() {
        Tensor base = new Tensor(new float[]{10f, 20f, 30f, 40f, 50f, 60f}, new int[]{2, 3}, null, "metalPhase36ScatterBase", DataType.FLOAT32);
        Tensor src = new Tensor(new float[]{1f, 5f}, new int[]{2}, null, "metalPhase36ScatterSrc", DataType.FLOAT32);

        Tensor floatIndices = new Tensor(new float[]{2f, 0f}, new int[]{2}, null, "metalPhase36ScatterFloatIndices", DataType.FLOAT32);
        Tensor dtypeScatter = base.scatterAdd(floatIndices, src, 1);
        TensorInternalAccess.setBackend(dtypeScatter, ComputeBackend.GPU_METAL);
        PartitionPlanningContext dtypeContext = planningContext(dtypeScatter);
        assertContainsAll(
                MetalPartitionSupport.plannerUnsupportedReason(
                        dtypeContext.compiledNode(nodeId(dtypeContext, Operation.OpType.SCATTER_ADD)),
                        dtypeContext
                ),
                "UNSUPPORTED_DTYPE",
                "SCATTER_ADD index input requires INT32"
        );

        Tensor oobIndices = new Tensor(new int[]{3, 0}, new int[]{2}, null, "metalPhase36ScatterOobIndices", DataType.INT32);
        Tensor oobScatter = base.scatterAdd(oobIndices, src, 1);
        TensorInternalAccess.setBackend(oobScatter, ComputeBackend.GPU_METAL);
        PartitionPlanningContext oobContext = planningContext(oobScatter);
        assertContainsAll(
                MetalPartitionSupport.plannerUnsupportedReason(
                        oobContext.compiledNode(nodeId(oobContext, Operation.OpType.SCATTER_ADD)),
                        oobContext
                ),
                "UNSUPPORTED_BOUNDS_CHECK",
                "index 3 is outside axis size 3"
        );

        Tensor layoutBase = new Tensor(new float[]{10f, 40f, 20f, 50f, 30f, 60f}, new int[]{3, 2}, null, "metalPhase36ScatterLayoutBase", DataType.FLOAT32);
        Tensor nonDenseBase = layoutBase.permute(1, 0);
        Tensor intIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "metalPhase36ScatterIntIndices", DataType.INT32);
        Tensor layoutScatter = nonDenseBase.scatterAdd(intIndices, src, 1);
        TensorInternalAccess.setBackend(layoutScatter, ComputeBackend.GPU_METAL);
        PartitionPlanningContext layoutContext = planningContext(layoutScatter);
        assertContainsAll(
                MetalPartitionSupport.plannerUnsupportedReason(
                        layoutContext.compiledNode(nodeId(layoutContext, Operation.OpType.SCATTER_ADD)),
                        layoutContext
                ),
                "UNSUPPORTED_LAYOUT",
                "SCATTER_ADD inputs require dense layout"
        );
    }

    @Test
    void phaseThirtySixIndexGradientRejectsBoundsAndUnprovenIndexBeforeDuplicateBlocker() {
        Tensor gatherOobIndices = new Tensor(new int[]{3, 0}, new int[]{2}, null, "metalPhase36GatherOobIndices", DataType.INT32);
        Tensor gatherOutGrad = new Tensor(new float[]{1f, 2f}, new int[]{2}, null, "metalPhase36GatherBoundsOutGrad", DataType.FLOAT32);
        Tensor gatherGradOut = TensorPrimitiveBuilder.binary(
                gatherOobIndices,
                gatherOutGrad,
                new int[]{2, 3},
                new gatherGrad(1),
                "metalPhase36GatherBoundsGrad",
                DataType.FLOAT32
        );
        TensorInternalAccess.setBackend(gatherGradOut, ComputeBackend.GPU_METAL);
        PartitionPlanningContext gatherBoundsContext = planningContext(gatherGradOut);
        assertContainsAll(
                MetalPartitionSupport.plannerUnsupportedReason(
                        gatherBoundsContext.compiledNode(nodeId(gatherBoundsContext, Operation.OpType.GATHER_GRAD)),
                        gatherBoundsContext
                ),
                "UNSUPPORTED_BOUNDS_CHECK",
                "index 3 is outside axis size 3"
        );

        Tensor takeOobIndices = new Tensor(new int[]{2, 3, 0, 0}, new int[]{2, 2}, null, "metalPhase36TakeOobIndices", DataType.INT32);
        Tensor takeOutGrad = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "metalPhase36TakeBoundsOutGrad", DataType.FLOAT32);
        Tensor takeGradOut = TensorPrimitiveBuilder.binary(
                takeOobIndices,
                takeOutGrad,
                new int[]{2, 3},
                new takeAlongAxisGrad(1),
                "metalPhase36TakeBoundsGrad",
                DataType.FLOAT32
        );
        TensorInternalAccess.setBackend(takeGradOut, ComputeBackend.GPU_METAL);
        PartitionPlanningContext takeBoundsContext = planningContext(takeGradOut);
        assertContainsAll(
                MetalPartitionSupport.plannerUnsupportedReason(
                        takeBoundsContext.compiledNode(nodeId(takeBoundsContext, Operation.OpType.TAKE_ALONG_AXIS_GRAD)),
                        takeBoundsContext
                ),
                "UNSUPPORTED_BOUNDS_CHECK",
                "index 3 is outside axis size 3"
        );

        Tensor dynamicGatherIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "metalPhase36DynamicGatherIndices", DataType.INT32)
                .reshape(2);
        Tensor dynamicGatherGrad = TensorPrimitiveBuilder.binary(
                dynamicGatherIndices,
                gatherOutGrad,
                new int[]{2, 3},
                new gatherGrad(1),
                "metalPhase36DynamicGatherGrad",
                DataType.FLOAT32
        );
        TensorInternalAccess.setBackend(dynamicGatherGrad, ComputeBackend.GPU_METAL);
        PartitionPlanningContext dynamicGatherContext = planningContext(dynamicGatherGrad);
        assertContainsAll(
                MetalPartitionSupport.plannerUnsupportedReason(
                        dynamicGatherContext.compiledNode(nodeId(dynamicGatherContext, Operation.OpType.GATHER_GRAD)),
                        dynamicGatherContext
                ),
                "UNSUPPORTED_BOUNDS_CHECK",
                "index bounds require a static INT32 leaf tensor"
        );
    }

    @Test
    void phaseThirtyTwoIndexFamilyRejectsUnprovenBoundsDtypeAndLayout() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "metalPhase32RejectInput", DataType.FLOAT32);

        Tensor oobIndices = new Tensor(new int[]{3, 0}, new int[]{2}, null, "metalPhase32OobIndices", DataType.INT32);
        Tensor oobGather = input.gather(oobIndices, 1);
        TensorInternalAccess.setBackend(oobGather, ComputeBackend.GPU_METAL);
        PartitionPlanningContext oobContext = planningContext(oobGather);
        assertContainsAll(
                MetalPartitionSupport.plannerUnsupportedReason(oobContext.compiledNode(nodeId(oobContext, Operation.OpType.GATHER)), oobContext),
                "UNSUPPORTED_BOUNDS_CHECK",
                "outside axis size 3"
        );

        Tensor f32Indices = new Tensor(new float[]{2f, 0f}, new int[]{2}, null, "metalPhase32F32Indices", DataType.FLOAT32);
        Tensor dtypeGather = input.gather(f32Indices, 1);
        TensorInternalAccess.setBackend(dtypeGather, ComputeBackend.GPU_METAL);
        PartitionPlanningContext dtypeContext = planningContext(dtypeGather);
        assertContainsAll(
                MetalPartitionSupport.plannerUnsupportedReason(dtypeContext.compiledNode(nodeId(dtypeContext, Operation.OpType.GATHER)), dtypeContext),
                "UNSUPPORTED_DTYPE",
                "index input requires INT32"
        );

        Tensor nonDenseValue = input.permute(1, 0);
        Tensor valueLayoutIndices = new Tensor(new int[]{1, 0}, new int[]{2}, null, "metalPhase32ValueLayoutIndices", DataType.INT32);
        Tensor valueLayoutGather = nonDenseValue.gather(valueLayoutIndices, 0);
        TensorInternalAccess.setBackend(valueLayoutGather, ComputeBackend.GPU_METAL);
        PartitionPlanningContext valueLayoutContext = planningContext(valueLayoutGather);
        assertContainsAll(
                MetalPartitionSupport.plannerUnsupportedReason(valueLayoutContext.compiledNode(nodeId(valueLayoutContext, Operation.OpType.GATHER)), valueLayoutContext),
                "UNSUPPORTED_LAYOUT",
                "inputs require dense layout"
        );

        Tensor denseInput = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "metalPhase32DenseInput", DataType.FLOAT32);
        Tensor indexBase = new Tensor(new int[]{1, 0, 0, 1}, new int[]{2, 2}, null, "metalPhase32IndexBase", DataType.INT32);
        Tensor nonDenseIndices = indexBase.permute(1, 0);
        Tensor indexLayoutTake = denseInput.takeAlongAxis(nonDenseIndices, 1);
        TensorInternalAccess.setBackend(indexLayoutTake, ComputeBackend.GPU_METAL);
        PartitionPlanningContext indexLayoutContext = planningContext(indexLayoutTake);
        assertContainsAll(
                MetalPartitionSupport.plannerUnsupportedReason(indexLayoutContext.compiledNode(nodeId(indexLayoutContext, Operation.OpType.TAKE_ALONG_AXIS)), indexLayoutContext),
                "UNSUPPORTED_LAYOUT",
                "inputs require dense layout"
        );
    }

    @Test
    void metalPhaseTwentySevenBoolCompareAndPoolUseStableCoverageReasons() {
        Tensor left = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "metalPhase27CompareLeft", DataType.FLOAT32);
        Tensor right = new Tensor(new float[]{2f, 2f, 2f, 2f}, new int[]{2, 2}, null, "metalPhase27CompareRight", DataType.FLOAT32);
        Tensor compare = left.greaterOrEqual(right);
        TensorInternalAccess.setBackend(compare, ComputeBackend.GPU_METAL);
        PartitionPlanningContext compareContext = planningContext(compare);
        String compareReason = MetalPartitionSupport.plannerUnsupportedReason(
                compareContext.compiledNode(nodeId(compareContext, Operation.OpType.GE)),
                compareContext
        );

        Tensor poolInput = new Tensor(new float[]{
                1f, 2f, 3f, 4f,
                5f, 6f, 7f, 8f,
                9f, 10f, 11f, 12f,
                13f, 14f, 15f, 16f
        }, new int[]{1, 1, 4, 4}, null, "metalPhase27PoolInput", DataType.FLOAT32);
        Tensor maxPool = poolInput.maxPool2d(Pool2dOptions.square(2));
        Tensor avgPool = poolInput.avgPool2d(Pool2dOptions.square(2));
        TensorInternalAccess.setBackend(maxPool, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(avgPool, ComputeBackend.GPU_METAL);
        PartitionPlanningContext maxPoolContext = planningContext(maxPool);
        PartitionPlanningContext avgPoolContext = planningContext(avgPool);

        String maxPoolReason = MetalPartitionSupport.plannerUnsupportedReason(
                maxPoolContext.compiledNode(nodeId(maxPoolContext, Operation.OpType.MAX_POOL2D)),
                maxPoolContext
        );
        String avgPoolReason = MetalPartitionSupport.plannerUnsupportedReason(
                avgPoolContext.compiledNode(nodeId(avgPoolContext, Operation.OpType.AVG_POOL2D)),
                avgPoolContext
        );

        assertEquals("", compareReason);
        assertEquals("", maxPoolReason);
        assertEquals("", avgPoolReason);
    }

    @Test
    void phaseThirtyFiveConvAndPoolLowerLegalForwardCases() {
        assertEquals(54, AcceleratorDagNodeType.CONV2D.abiCode());
        assertEquals(55, AcceleratorDagNodeType.MAX_POOL2D.abiCode());
        assertEquals(56, AcceleratorDagNodeType.AVG_POOL2D.abiCode());

        Tensor input = new Tensor(
                new float[]{
                        1f, 2f, 3f,
                        4f, 5f, 6f,
                        7f, 8f, 9f
                },
                new int[]{1, 1, 3, 3},
                null,
                "metalPhase35ConvInput",
                DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 1, 2, 2}, null, "metalPhase35ConvWeight", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.5f}, new int[]{1}, null, "metalPhase35ConvBias", DataType.FLOAT32);
        Tensor conv = input.conv2d(weight, bias, Conv2dOptions.defaults());
        TensorInternalAccess.setBackend(conv, ComputeBackend.GPU_METAL);
        PartitionPlanningContext convContext = planningContext(conv);

        String convReason = MetalPartitionSupport.plannerUnsupportedReason(
                convContext.compiledNode(nodeId(convContext, Operation.OpType.CONV2D)),
                convContext
        );

        Tensor gemm = TensorPrimitiveBuilder.binary(
                input,
                weight,
                conv.getShape(),
                new conv2dGemm(Conv2dOptions.defaults(), false),
                "metalPhase35ConvGemm",
                DataType.FLOAT32
        );
        TensorInternalAccess.setBackend(gemm, ComputeBackend.GPU_METAL);
        PartitionPlanningContext gemmContext = planningContext(gemm);
        String gemmReason = MetalPartitionSupport.plannerUnsupportedReason(
                gemmContext.compiledNode(nodeId(gemmContext, Operation.OpType.CONV2D_GEMM)),
                gemmContext
        );

        Tensor poolInput = new Tensor(new float[]{
                1f, 2f, 3f, 4f,
                5f, 6f, 7f, 8f,
                9f, 10f, 11f, 12f,
                13f, 14f, 15f, 16f
        }, new int[]{1, 1, 4, 4}, null, "metalPhase35PoolInput", DataType.FLOAT32);
        Tensor maxPool = poolInput.maxPool2d(Pool2dOptions.square(2));
        Tensor avgPool = poolInput.avgPool2d(Pool2dOptions.square(2));
        TensorInternalAccess.setBackend(maxPool, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(avgPool, ComputeBackend.GPU_METAL);
        PartitionPlanningContext maxPoolContext = planningContext(maxPool);
        PartitionPlanningContext avgPoolContext = planningContext(avgPool);

        String maxPoolReason = MetalPartitionSupport.plannerUnsupportedReason(
                maxPoolContext.compiledNode(nodeId(maxPoolContext, Operation.OpType.MAX_POOL2D)),
                maxPoolContext
        );
        String avgPoolReason = MetalPartitionSupport.plannerUnsupportedReason(
                avgPoolContext.compiledNode(nodeId(avgPoolContext, Operation.OpType.AVG_POOL2D)),
                avgPoolContext
        );

        assertEquals("", convReason);
        assertEquals("", gemmReason);
        assertEquals("", maxPoolReason);
        assertEquals("", avgPoolReason);
        MetalRegionLegalityAdapter adapter = new MetalRegionLegalityAdapter();
        PartitionCandidate candidate = adapter.tryCreateStructuralCandidate(
                Set.of(nodeId(convContext, Operation.OpType.CONV2D)),
                convContext,
                Set.of(PartitionValueRef.ofNode(nodeId(convContext, Operation.OpType.CONV2D)))
        );
        assertNotNull(candidate);
        MetalPartitionPlan plan = (MetalPartitionPlan) adapter.tryCreatePlan(candidate, convContext);
        assertNotNull(plan);
        assertTrue(plan.lowering().dagSpec().nodes().stream()
                .anyMatch(node -> node.type() == AcceleratorDagNodeType.CONV2D));
        assertTrue(planFor(gemm, Operation.OpType.CONV2D_GEMM).lowering().dagSpec().nodes().stream()
                .anyMatch(node -> node.type() == AcceleratorDagNodeType.CONV2D));
        assertTrue(planFor(maxPool, Operation.OpType.MAX_POOL2D).lowering().dagSpec().nodes().stream()
                .anyMatch(node -> node.type() == AcceleratorDagNodeType.MAX_POOL2D));
        assertTrue(planFor(avgPool, Operation.OpType.AVG_POOL2D).lowering().dagSpec().nodes().stream()
                .anyMatch(node -> node.type() == AcceleratorDagNodeType.AVG_POOL2D));
    }

    @Test
    void phaseThirtyFiveConvRejectsDtypeLayoutRankGroupsAndDilationPrecisely() {
        Tensor bf16Input = new Tensor(new double[]{
                1d, 2d, 3d,
                4d, 5d, 6d,
                7d, 8d, 9d
        }, new int[]{1, 1, 3, 3}, null, "metalPhase35Bf16ConvInput", DataType.BFLOAT16);
        Tensor bf16Weight = new Tensor(new double[]{1d, 0d, 0d, 1d}, new int[]{1, 1, 2, 2}, null, "metalPhase35Bf16ConvWeight", DataType.BFLOAT16);
        Tensor bf16Conv = bf16Input.conv2d(bf16Weight, Conv2dOptions.defaults());
        TensorInternalAccess.setBackend(bf16Conv, ComputeBackend.GPU_METAL);
        PartitionPlanningContext bf16Context = planningContext(bf16Conv);
        assertEquals(
                "",
                MetalPartitionSupport.plannerUnsupportedReason(bf16Context.compiledNode(nodeId(bf16Context, Operation.OpType.CONV2D)), bf16Context)
        );
        assertTrue(planFor(bf16Conv, Operation.OpType.CONV2D).lowering().dagSpec().nodes().stream()
                .anyMatch(node -> node.type() == AcceleratorDagNodeType.CONV2D && node.outputDataType() == DataType.BFLOAT16));

        Tensor f32Weight = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 1, 2, 2}, null, "metalPhase35F32ConvWeight", DataType.FLOAT32);
        Tensor dtypeConv = bf16Input.conv2d(f32Weight, Conv2dOptions.defaults());
        TensorInternalAccess.setBackend(dtypeConv, ComputeBackend.GPU_METAL);
        PartitionPlanningContext dtypeContext = planningContext(dtypeConv);
        assertContainsAll(
                MetalPartitionSupport.plannerUnsupportedReason(dtypeContext.compiledNode(nodeId(dtypeContext, Operation.OpType.CONV2D)), dtypeContext),
                "UNSUPPORTED_DTYPE",
                "CONV2D inputs and output must use the same FLOAT32/BFLOAT16 dtype"
        );

        Tensor layoutBase = new Tensor(new float[]{
                1f, 4f, 7f,
                2f, 5f, 8f,
                3f, 6f, 9f
        }, new int[]{1, 3, 3, 1}, null, "metalPhase35LayoutConvBase", DataType.FLOAT32);
        Tensor nonDenseInput = layoutBase.permute(0, 3, 1, 2);
        Tensor layoutConv = nonDenseInput.conv2d(f32Weight, Conv2dOptions.defaults());
        TensorInternalAccess.setBackend(layoutConv, ComputeBackend.GPU_METAL);
        PartitionPlanningContext layoutContext = planningContext(layoutConv);
        assertContainsAll(
                MetalPartitionSupport.plannerUnsupportedReason(layoutContext.compiledNode(nodeId(layoutContext, Operation.OpType.CONV2D)), layoutContext),
                "UNSUPPORTED_LAYOUT",
                "CONV2D inputs require dense layout"
        );

        Tensor rankInput = new Tensor(new float[9], new int[]{1, 3, 3}, null, "metalPhase35RankConvInput", DataType.FLOAT32);
        Tensor rankConv = TensorPrimitiveBuilder.binary(
                rankInput,
                f32Weight,
                new int[]{1, 1, 2, 2},
                new conv2d(Conv2dOptions.defaults(), false),
                "metalPhase35RankConv",
                DataType.FLOAT32
        );
        TensorInternalAccess.setBackend(rankConv, ComputeBackend.GPU_METAL);
        PartitionPlanningContext rankContext = planningContext(rankConv);
        assertContainsAll(
                MetalPartitionSupport.plannerUnsupportedReason(rankContext.compiledNode(nodeId(rankContext, Operation.OpType.CONV2D)), rankContext),
                "UNSUPPORTED_RANK_OR_SHAPE",
                "rank-4 NCHW input/output and OIHW weight"
        );

        Tensor groupedInput = new Tensor(new float[1 * 2 * 3 * 3], new int[]{1, 2, 3, 3}, null, "metalPhase35GroupedConvInput", DataType.FLOAT32);
        Tensor groupedWeight = new Tensor(new float[2 * 1 * 2 * 2], new int[]{2, 1, 2, 2}, null, "metalPhase35GroupedConvWeight", DataType.FLOAT32);
        Tensor groupedConv = groupedInput.conv2d(groupedWeight, Conv2dOptions.defaults().withGroups(2));
        TensorInternalAccess.setBackend(groupedConv, ComputeBackend.GPU_METAL);
        PartitionPlanningContext groupedContext = planningContext(groupedConv);
        assertContainsAll(
                MetalPartitionSupport.plannerUnsupportedReason(groupedContext.compiledNode(nodeId(groupedContext, Operation.OpType.CONV2D)), groupedContext),
                "CAPABILITY_MISSING",
                "grouped/depthwise native execution is not implemented"
        );

        Tensor dilationInput = new Tensor(new float[1 * 1 * 4 * 4], new int[]{1, 1, 4, 4}, null, "metalPhase35DilationConvInput", DataType.FLOAT32);
        Tensor dilationConv = dilationInput.conv2d(f32Weight, Conv2dOptions.defaults().withDilation(2, 2));
        TensorInternalAccess.setBackend(dilationConv, ComputeBackend.GPU_METAL);
        PartitionPlanningContext dilationContext = planningContext(dilationConv);
        assertContainsAll(
                MetalPartitionSupport.plannerUnsupportedReason(dilationContext.compiledNode(nodeId(dilationContext, Operation.OpType.CONV2D)), dilationContext),
                "CAPABILITY_MISSING",
                "dilation native execution is not implemented"
        );
    }

    @Test
    void phaseThirtyFivePoolRejectsDtypeLayoutRankAndAvgCountIncludePadPrecisely() {
        Tensor bf16Input = new Tensor(new double[1 * 1 * 4 * 4], new int[]{1, 1, 4, 4}, null, "metalPhase35Bf16PoolInput", DataType.BFLOAT16);
        Tensor bf16Pool = bf16Input.maxPool2d(Pool2dOptions.square(2));
        TensorInternalAccess.setBackend(bf16Pool, ComputeBackend.GPU_METAL);
        PartitionPlanningContext bf16Context = planningContext(bf16Pool);
        assertEquals(
                "",
                MetalPartitionSupport.plannerUnsupportedReason(bf16Context.compiledNode(nodeId(bf16Context, Operation.OpType.MAX_POOL2D)), bf16Context)
        );
        assertTrue(planFor(bf16Pool, Operation.OpType.MAX_POOL2D).lowering().dagSpec().nodes().stream()
                .anyMatch(node -> node.type() == AcceleratorDagNodeType.MAX_POOL2D && node.outputDataType() == DataType.BFLOAT16));

        Tensor dtypePool = TensorPrimitiveBuilder.unary(
                bf16Input,
                new int[]{1, 1, 2, 2},
                new maxPool2d(Pool2dOptions.square(2)),
                "metalPhase35DtypePool",
                DataType.FLOAT32
        );
        TensorInternalAccess.setBackend(dtypePool, ComputeBackend.GPU_METAL);
        PartitionPlanningContext dtypeContext = planningContext(dtypePool);
        assertContainsAll(
                MetalPartitionSupport.plannerUnsupportedReason(dtypeContext.compiledNode(nodeId(dtypeContext, Operation.OpType.MAX_POOL2D)), dtypeContext),
                "UNSUPPORTED_DTYPE",
                "MAX_POOL2D inputs and output must use the same FLOAT32/BFLOAT16 dtype"
        );

        Tensor layoutBase = new Tensor(new float[1 * 4 * 4 * 1], new int[]{1, 4, 4, 1}, null, "metalPhase35LayoutPoolBase", DataType.FLOAT32);
        Tensor nonDenseInput = layoutBase.permute(0, 3, 1, 2);
        Tensor layoutPool = nonDenseInput.maxPool2d(Pool2dOptions.square(2));
        TensorInternalAccess.setBackend(layoutPool, ComputeBackend.GPU_METAL);
        PartitionPlanningContext layoutContext = planningContext(layoutPool);
        assertContainsAll(
                MetalPartitionSupport.plannerUnsupportedReason(layoutContext.compiledNode(nodeId(layoutContext, Operation.OpType.MAX_POOL2D)), layoutContext),
                "UNSUPPORTED_LAYOUT",
                "MAX_POOL2D inputs require dense layout"
        );

        Tensor rankInput = new Tensor(new float[16], new int[]{4, 4}, null, "metalPhase35RankPoolInput", DataType.FLOAT32);
        Tensor rankPool = TensorPrimitiveBuilder.unary(
                rankInput,
                new int[]{2, 2},
                new maxPool2d(Pool2dOptions.square(2)),
                "metalPhase35RankPool",
                DataType.FLOAT32
        );
        TensorInternalAccess.setBackend(rankPool, ComputeBackend.GPU_METAL);
        PartitionPlanningContext rankContext = planningContext(rankPool);
        assertContainsAll(
                MetalPartitionSupport.plannerUnsupportedReason(rankContext.compiledNode(nodeId(rankContext, Operation.OpType.MAX_POOL2D)), rankContext),
                "UNSUPPORTED_RANK_OR_SHAPE",
                "MAX_POOL2D requires rank-4 NCHW input/output"
        );

        Tensor avgInput = new Tensor(new float[1 * 1 * 4 * 4], new int[]{1, 1, 4, 4}, null, "metalPhase35AvgPoolInput", DataType.FLOAT32);
        Tensor avgWithPad = avgInput.avgPool2d(Pool2dOptions.square(2).withPadding(1, 1).withCountIncludePad(true));
        TensorInternalAccess.setBackend(avgWithPad, ComputeBackend.GPU_METAL);
        PartitionPlanningContext avgContext = planningContext(avgWithPad);
        assertContainsAll(
                MetalPartitionSupport.plannerUnsupportedReason(avgContext.compiledNode(nodeId(avgContext, Operation.OpType.AVG_POOL2D)), avgContext),
                "CAPABILITY_MISSING",
                "AVG_POOL2D countIncludePad=true native divisor semantics are not implemented"
        );
    }

    @Test
    void boolCompareDagContractLowersWithPlannerAdmission() {
        assertEquals(41, AcceleratorDagNodeType.GT.abiCode());
        assertEquals(42, AcceleratorDagNodeType.GE.abiCode());
        assertEquals(43, AcceleratorDagNodeType.LT.abiCode());
        assertEquals(44, AcceleratorDagNodeType.LE.abiCode());
        assertEquals(45, AcceleratorDagNodeType.EQ.abiCode());
        assertEquals(46, AcceleratorDagNodeType.NE.abiCode());
        assertEquals(47, AcceleratorDagNodeType.LOGICAL_AND.abiCode());
        assertEquals(48, AcceleratorDagNodeType.LOGICAL_OR.abiCode());
        assertEquals(49, AcceleratorDagNodeType.LOGICAL_NOT.abiCode());
        assertEquals(50, AcceleratorDagNodeType.REDUCE_ALL.abiCode());
        assertEquals(51, AcceleratorDagNodeType.REDUCE_ANY.abiCode());

        Tensor left = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "metalPhase31BoolLeft", DataType.FLOAT32);
        Tensor right = new Tensor(new float[]{2f, 2f, 2f, 2f}, new int[]{2, 2}, null, "metalPhase31BoolRight", DataType.FLOAT32);
        Tensor compare = left.greaterOrEqual(right);
        TensorInternalAccess.setBackend(compare, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(compare);
        int compareNodeId = nodeId(context, Operation.OpType.GE);
        CompiledNode compareNode = context.compiledNode(compareNodeId);
        var lowered = new AcceleratorSubgraphLowerer().tryLower(
                ComputeBackend.GPU_METAL,
                new AcceleratorSubgraphSpec(
                        compareNodeId,
                        List.of(compareNodeId),
                        List.of(new backend.accelerator.dag.AcceleratorSubgraphOp(compareNodeId, Operation.OpType.GE)),
                        compareNode.inputIds(),
                        List.of(compareNodeId)
                ),
                context
        );

        assertNotNull(lowered);
        assertEquals(AcceleratorDagNodeType.GE, lowered.dagSpec().nodes().getFirst().type());
        assertEquals(DataType.BOOL, lowered.dagSpec().nodes().getFirst().outputDataType());
        assertTrue(lowered.dagSpec().externalInputs().stream().allMatch(input -> input.dataType() == DataType.FLOAT32));
        assertEquals(
                "SUPPORTED",
                GpuLoweringCoverageMatrix.entryFor(ComputeBackend.GPU_METAL, Operation.OpType.GE).reason().name()
        );
        assertNotNull(new MetalRegionLegalityAdapter().tryCreateStructuralCandidate(
                Set.of(compareNodeId),
                context,
                Set.of(PartitionValueRef.ofNode(compareNodeId))
        ));
    }

    @Test
    void boolLogicalAndReductionDagContractsLowerForMetal() {
        Tensor left = new Tensor(new byte[]{1, 0, 1, 0}, new int[]{2, 2}, null, "metalPhase31BoolLeft", DataType.BOOL);
        Tensor right = new Tensor(new byte[]{1, 1, 0, 0}, new int[]{2, 2}, null, "metalPhase31BoolRight", DataType.BOOL);
        Tensor logical = left.logicalAnd(right);
        Tensor reduced = logical.any(1, true);
        TensorInternalAccess.setBackend(logical, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(reduced, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(reduced);
        int logicalNodeId = nodeId(context, Operation.OpType.LOGICAL_AND);
        int reduceNodeId = nodeId(context, Operation.OpType.REDUCE_ANY);
        MetalRegionLegalityAdapter adapter = new MetalRegionLegalityAdapter();
        PartitionCandidate candidate = adapter.tryCreateStructuralCandidate(
                Set.of(logicalNodeId, reduceNodeId),
                context,
                Set.of(PartitionValueRef.ofNode(reduceNodeId))
        );

        assertNotNull(candidate);
        MetalPartitionPlan plan = (MetalPartitionPlan) adapter.tryCreatePlan(candidate, context);
        assertNotNull(plan);
        List<AcceleratorDagNodeType> types = plan.lowering().dagSpec().nodes().stream()
                .map(backend.accelerator.dag.AcceleratorDagNode::type)
                .toList();
        assertTrue(types.contains(AcceleratorDagNodeType.LOGICAL_AND));
        assertTrue(types.contains(AcceleratorDagNodeType.REDUCE_ANY));
        assertTrue(plan.lowering().dagSpec().nodes().stream().allMatch(node -> node.outputDataType() == DataType.BOOL));
    }

    @Test
    void lowersGpuMetalRegionToMetalGraphRegion() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor out = matmul.relu();
        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(graph);
        graph.optimizer.partition.PartitionPlanningContext planningContext = new graph.optimizer.partition.PartitionPlanningContext(
                RuntimeConfig.inferenceDefaults(),
                false,
                compiledNodes,
                CompiledTensorDescriptorBuilder.build(compiledNodes),
                consumers(compiledNodes)
        );
        MetalRegionLegalityAdapter adapter = new MetalRegionLegalityAdapter();
        PartitionCandidate candidate = adapter.tryCreateStructuralCandidate(
                Set.of(2, 3),
                planningContext,
                Set.of(PartitionValueRef.ofNode(3))
        );
        var attachedPlan = adapter.tryCreatePlan(candidate, planningContext);
        Partition partition = new Partition(
                "metal-partition",
                PartitionTarget.GPU_METAL,
                candidate.orderedNodeIds(),
                candidate.orderedNodeIds().stream().map(id -> new PartitionValue(PartitionValueRef.ofNode(id), id)).toList(),
                List.of(new PartitionEdge(2, 3)),
                candidate.externalInputIds(),
                candidate.outputNodeIds().stream().map(PartitionValueRef::ofNode).toList(),
                candidate.anchorNodeId(),
                List.of(PartitionValueRef.ofNode(3)),
                List.of(),
                List.of(PartitionBoundaryReason.NONE),
                attachedPlan.estimatedWork(),
                new graph.optimizer.partition.cost.AcceleratorPartitionScoreModel.CandidateMetrics(candidate.orderedNodeIds().size(), 1, candidate.externalInputIds().size(), 0, 1),
                PartitionPlannerStrategy.GREEDY_MAX_REGION,
                new PartitionDecisionTrace(
                        PartitionPlannerStrategy.GREEDY_MAX_REGION,
                        PartitionTarget.GPU_METAL,
                        candidate.anchorNodeId(),
                        true,
                        "test",
                        candidate.orderedNodeIds(),
                        candidate.orderedNodeIds(),
                        List.of("MATMUL", "RELU"),
                        attachedPlan.estimatedWork(),
                        0.0d,
                        0.0d,
                        0,
                        false,
                        -1
                )
        );
        OptimizedRegion region = new DefaultRegionOptimizer().optimize(partition, new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults()));

        MetalRegionLowerer lowerer = new MetalRegionLowerer();
        LoweringResult result = lowerer.lower(new LoweringRequest(
                region,
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.GPU_METAL)),
                new LoweringContext(RuntimeConfig.inferenceDefaults(), compiledNodes, CompiledTensorDescriptorBuilder.build(compiledNodes), java.util.Map.of(partition.partitionId(), attachedPlan))
        ));

        assertNotNull(result);
        assertNotNull(result.loweredRegion());
        assertEquals(backend.lowering.LoweringFamily.METAL_GRAPH_REGION, result.loweredRegion().units().getFirst().loweringFamily());
    }

    @Test
    void lowersPureElementwiseGpuMetalRegionToMpsGraphDagWithFusionMetadata() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor add = a.add(b);
        Tensor out = add.relu().exp();
        TensorInternalAccess.setBackend(add, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out.getPrevTensors().getFirst(), ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(graph);
        graph.optimizer.partition.PartitionPlanningContext planningContext = new graph.optimizer.partition.PartitionPlanningContext(
                RuntimeConfig.inferenceDefaults(),
                false,
                compiledNodes,
                CompiledTensorDescriptorBuilder.build(compiledNodes),
                consumers(compiledNodes)
        );
        MetalRegionLegalityAdapter adapter = new MetalRegionLegalityAdapter();
        PartitionCandidate candidate = adapter.tryCreateStructuralCandidate(
                Set.of(2, 3, 4),
                planningContext,
                Set.of(PartitionValueRef.ofNode(4))
        );
        assertNotNull(candidate);
        var attachedPlan = adapter.tryCreatePlan(candidate, planningContext);
        assertNotNull(attachedPlan);

        Partition partition = new Partition(
                "metal-elementwise-partition",
                PartitionTarget.GPU_METAL,
                candidate.orderedNodeIds(),
                candidate.orderedNodeIds().stream().map(id -> new PartitionValue(PartitionValueRef.ofNode(id), id)).toList(),
                List.of(new PartitionEdge(2, 3), new PartitionEdge(3, 4)),
                candidate.externalInputIds(),
                candidate.outputNodeIds().stream().map(PartitionValueRef::ofNode).toList(),
                candidate.anchorNodeId(),
                List.of(PartitionValueRef.ofNode(4)),
                List.of(),
                List.of(PartitionBoundaryReason.NONE),
                attachedPlan.estimatedWork(),
                new graph.optimizer.partition.cost.AcceleratorPartitionScoreModel.CandidateMetrics(candidate.orderedNodeIds().size(), 2, candidate.externalInputIds().size(), 0, 2),
                PartitionPlannerStrategy.GREEDY_MAX_REGION,
                new PartitionDecisionTrace(
                        PartitionPlannerStrategy.GREEDY_MAX_REGION,
                        PartitionTarget.GPU_METAL,
                        candidate.anchorNodeId(),
                        true,
                        "test",
                        candidate.orderedNodeIds(),
                        candidate.orderedNodeIds(),
                        List.of("ADD", "RELU", "EXP"),
                        attachedPlan.estimatedWork(),
                        0.0d,
                        0.0d,
                        0,
                        false,
                        -1
                )
        );
        OptimizedRegion region = new DefaultRegionOptimizer().optimize(
                partition,
                new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults())
        );

        assertEquals(1, region.executionUnits().size());
        assertEquals(ExecutionUnitKind.FUSED_ELEMENTWISE, region.executionUnits().getFirst().kind());

        MetalRegionLowerer lowerer = new MetalRegionLowerer();
        LoweringResult result = lowerer.lower(new LoweringRequest(
                region,
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.GPU_METAL)),
                new LoweringContext(RuntimeConfig.inferenceDefaults(), compiledNodes, CompiledTensorDescriptorBuilder.build(compiledNodes), java.util.Map.of(partition.partitionId(), attachedPlan))
        ));

        assertNotNull(result);
        assertNotNull(result.loweredRegion());
        assertEquals(backend.lowering.LoweringFamily.METAL_GRAPH_REGION, result.loweredRegion().units().getFirst().loweringFamily());
        GpuCompoundLoweringArtifact artifact = result.loweredRegion().units().getFirst()
                .requireArtifact(GpuCompoundLoweringArtifact.class);
        assertTrue(artifact.units().stream()
                .anyMatch(unit -> unit.kind() == ExecutionUnitKind.FUSED_ELEMENTWISE));
    }

    @Test
    void metalLowersElementwiseSubchainAsRegionInternalFusion() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "metalSubchainA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 0f, 0f, 1f, 1f, 1f}, new int[]{3, 2}, null, "metalSubchainB", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.25f, -0.5f}, new int[]{2}, null, "metalSubchainBias", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor add = matmul.add(bias);
        Tensor relu = add.relu();
        Tensor out = relu.exp();
        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(add, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(relu, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(graph);
        PartitionPlanningContext planningContext = new PartitionPlanningContext(
                RuntimeConfig.inferenceDefaults(),
                false,
                compiledNodes,
                CompiledTensorDescriptorBuilder.build(compiledNodes),
                consumers(compiledNodes)
        );
        int matmulNodeId = nodeId(planningContext, Operation.OpType.MATMUL);
        int addNodeId = nodeId(planningContext, Operation.OpType.ADD);
        int reluNodeId = nodeId(planningContext, Operation.OpType.RELU);
        int expNodeId = nodeId(planningContext, Operation.OpType.EXP);
        List<Integer> selectedNodeIds = List.of(matmulNodeId, addNodeId, reluNodeId, expNodeId);
        MetalRegionLegalityAdapter adapter = new MetalRegionLegalityAdapter();
        PartitionCandidate candidate = adapter.tryCreateStructuralCandidate(
                Set.copyOf(selectedNodeIds),
                planningContext,
                Set.of(PartitionValueRef.ofNode(expNodeId))
        );
        assertNotNull(candidate);
        MetalPartitionPlan attachedPlan = (MetalPartitionPlan) adapter.tryCreatePlan(candidate, planningContext);
        assertNotNull(attachedPlan);

        Partition partition = partition(
                "metal-elementwise-subchain",
                PartitionTarget.GPU_METAL,
                candidate.orderedNodeIds(),
                candidate.externalInputIds(),
                candidate.outputNodeIds().stream().map(PartitionValueRef::ofNode).toList(),
                List.of(PartitionValueRef.ofNode(expNodeId))
        );
        OptimizedRegion region = new DefaultRegionOptimizer().optimize(
                partition,
                new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults())
        );
        LoweringResult result = new MetalRegionLowerer().lower(new LoweringRequest(
                region,
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.GPU_METAL)),
                new LoweringContext(RuntimeConfig.inferenceDefaults(), compiledNodes, CompiledTensorDescriptorBuilder.build(compiledNodes), java.util.Map.of(partition.partitionId(), attachedPlan))
        ));

        assertNotNull(result);
        assertEquals(backend.lowering.LoweringFamily.METAL_GRAPH_REGION, result.loweredRegion().units().getFirst().loweringFamily());
        assertTrue(result.loweredRegion().units().getFirst().orderedNodeIds().containsAll(selectedNodeIds));
        var subpattern = attachedPlan.manifest().fusedSubpatterns().stream()
                .filter(candidateSubpattern -> candidateSubpattern.patternType() == GpuCompoundPatternType.ELEMENTWISE_CHAIN)
                .findFirst()
                .orElseThrow();
        assertEquals(List.of(addNodeId, reluNodeId, expNodeId), subpattern.originalOperationNodeIds());
        assertEquals(3, subpattern.loweredPrimitiveCount());
    }

    @Test
    void metalLowersMatmulBiasActivationAsEpilogueSubpattern() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "metalEpilogueA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 0f, 0f, 1f, 1f, 1f}, new int[]{3, 2}, null, "metalEpilogueB", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.25f, -0.5f}, new int[]{2}, null, "metalEpilogueBias", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor add = matmul.add(bias);
        Tensor out = add.relu();
        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(add, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(graph);
        PartitionPlanningContext planningContext = new PartitionPlanningContext(
                RuntimeConfig.inferenceDefaults(),
                false,
                compiledNodes,
                CompiledTensorDescriptorBuilder.build(compiledNodes),
                consumers(compiledNodes)
        );
        int matmulNodeId = nodeId(planningContext, Operation.OpType.MATMUL);
        int addNodeId = nodeId(planningContext, Operation.OpType.ADD);
        int reluNodeId = nodeId(planningContext, Operation.OpType.RELU);
        List<Integer> selectedNodeIds = List.of(matmulNodeId, addNodeId, reluNodeId);
        MetalRegionLegalityAdapter adapter = new MetalRegionLegalityAdapter();
        PartitionCandidate candidate = adapter.tryCreateStructuralCandidate(
                Set.copyOf(selectedNodeIds),
                planningContext,
                Set.of(PartitionValueRef.ofNode(reluNodeId))
        );
        assertNotNull(candidate);
        MetalPartitionPlan plan = (MetalPartitionPlan) adapter.tryCreatePlan(candidate, planningContext);
        assertNotNull(plan);
        var epilogue = plan.manifest().fusedSubpatterns().stream()
                .filter(candidateSubpattern -> candidateSubpattern.patternType() == GpuCompoundPatternType.LINEAR_BIAS_ACTIVATION)
                .findFirst()
                .orElseThrow();

        assertEquals(selectedNodeIds, epilogue.originalOperationNodeIds());
        assertTrue(epilogue.detail().contains("epilogue"));
        assertTrue(epilogue.loweredPrimitiveCount() >= 2);
    }

    @Test
    void metalRejectsIllegalEpilogueLayoutWithStableReason() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "metalBadEpilogueA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 0f, 0f, 1f, 1f, 1f}, new int[]{3, 2}, null, "metalBadEpilogueB", DataType.FLOAT32);
        Tensor biasBase = new Tensor(new float[]{0.25f, -0.5f, 0.75f, 1f}, new int[]{2, 2}, null, "metalBadEpilogueBiasBase", DataType.FLOAT32);
        Tensor bias = biasBase.select(0, 1);
        Tensor matmul = a.matmul(b);
        Tensor add = matmul.add(bias);
        Tensor out = add.relu();
        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(add, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        String reason = MetalPartitionSupport.plannerUnsupportedReason(context.compiledNode(nodeId(context, Operation.OpType.ADD)), context);

        assertContainsAll(reason,
                "UNSUPPORTED_LAYOUT",
                "family=MATMUL_LINEAR",
                "LINEAR_BIAS_ACTIVATION",
                "GPU_METAL");
    }

    @Test
    void rejectsFloat64MetalCandidateBeforeLowering() {
        Tensor a = new Tensor(new double[]{1d, 2d, 3d, 4d, 5d, 6d}, new int[]{2, 3}, null, "a64", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{1d, 2d, 3d, 4d, 5d, 6d}, new int[]{3, 2}, null, "b64", DataType.FLOAT64);
        Tensor matmul = a.matmul(b);
        Tensor out = matmul.relu();
        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        assertNull(new MetalRegionLegalityAdapter().tryCreateStructuralCandidate(
                operationNodeIds(context),
                context,
                Set.of(PartitionValueRef.ofNode(nodeId(context, Operation.OpType.RELU)))
        ));
    }

    @Test
    void acceptsScopedBfloat16MetalCandidateBeforeLowering() {
        Tensor a = new Tensor(new double[]{1d, 2d, 3d, 4d, 5d, 6d}, new int[]{2, 3}, null, "abf16", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[]{1d, 2d, 3d, 4d, 5d, 6d}, new int[]{3, 2}, null, "bbf16", DataType.BFLOAT16);
        Tensor matmul = a.matmul(b);
        Tensor out = matmul.relu();
        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        MetalRegionLegalityAdapter adapter = new MetalRegionLegalityAdapter();
        PartitionCandidate candidate = adapter.tryCreateStructuralCandidate(
                operationNodeIds(context),
                context,
                Set.of(PartitionValueRef.ofNode(nodeId(context, Operation.OpType.RELU)))
        );

        assertNotNull(candidate);
        MetalPartitionPlan plan = (MetalPartitionPlan) adapter.tryCreatePlan(candidate, context);
        assertNotNull(plan);
        assertTrue(plan.lowering().dagSpec().externalInputs().stream().allMatch(input -> input.dataType() == DataType.BFLOAT16));
        assertTrue(plan.lowering().dagSpec().nodes().stream().allMatch(node -> node.outputDataType() == DataType.BFLOAT16));
    }

    @Test
    void rejectsInt32MetalLayoutCandidateBeforeLowering() {
        Tensor index = new Tensor(new int[]{1, 2, 3, 4}, new int[]{4}, null, "index", DataType.INT32);
        Tensor out = index.contiguous();
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        assertNull(new MetalRegionLegalityAdapter().tryCreateStructuralCandidate(
                operationNodeIds(context),
                context,
                Set.of(PartitionValueRef.ofNode(nodeId(context, Operation.OpType.CONTIGUOUS)))
        ));
    }

    @Test
    void acceptsWhereWithBoolPredicateAndFloat32Branches() {
        Tensor mask = new Tensor(new byte[]{1, 0}, new int[]{2}, null, "mask", DataType.BOOL);
        Tensor x = new Tensor(new float[]{1f, 2f}, new int[]{2}, null, "x", DataType.FLOAT32);
        Tensor y = new Tensor(new float[]{3f, 4f}, new int[]{2}, null, "y", DataType.FLOAT32);
        Tensor out = Tensor.where(mask, x, y);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        MetalRegionLegalityAdapter adapter = new MetalRegionLegalityAdapter();
        PartitionCandidate candidate = adapter.tryCreateStructuralCandidate(
                operationNodeIds(context),
                context,
                Set.of(PartitionValueRef.ofNode(nodeId(context, Operation.OpType.WHERE)))
        );

        assertNotNull(candidate);
        assertNotNull(adapter.tryCreatePlan(candidate, context));
    }

    @Test
    void rejectsWhereWithBoolValueBranch() {
        Tensor mask = new Tensor(new byte[]{1, 0}, new int[]{2}, null, "mask", DataType.BOOL);
        Tensor boolValue = new Tensor(new byte[]{1, 1}, new int[]{2}, null, "boolValue", DataType.BOOL);
        Tensor floatValue = new Tensor(new float[]{3f, 4f}, new int[]{2}, null, "floatValue", DataType.FLOAT32);
        Tensor out = TensorPrimitiveBuilder.ternary(mask, boolValue, floatValue, new int[]{2}, new where(), "invalidWhere", DataType.FLOAT32);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        assertNull(new MetalRegionLegalityAdapter().tryCreateStructuralCandidate(
                operationNodeIds(context),
                context,
                Set.of(PartitionValueRef.ofNode(nodeId(context, Operation.OpType.WHERE)))
        ));
    }

    @Test
    void metalForwardSdpaRegionPublishesInternalContextInputsForBackwardRegions() {
        Tensor qBase = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "qBase", DataType.FLOAT32);
        Tensor kBase = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "kBase", DataType.FLOAT32);
        Tensor vBase = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "vBase", DataType.FLOAT32);
        Tensor q = qBase.relu();
        Tensor k = kBase.relu();
        Tensor v = vBase.relu();
        Tensor out = specialSdpa(q, k, v, null, 0.5d);
        TensorInternalAccess.setBackend(q, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(k, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(v, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        int sdpaNodeId = nodeId(context, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION);
        List<Integer> selectedNodeIds = context.compiledNodes().stream()
                .filter(node -> node.operation() != null)
                .map(CompiledNode::id)
                .toList();
        PartitionCandidate candidate = new MetalRegionLegalityAdapter().tryCreateStructuralCandidate(
                Set.copyOf(selectedNodeIds),
                context,
                Set.of(PartitionValueRef.ofNode(sdpaNodeId))
        );

        assertNotNull(candidate);
        CompiledNode sdpaNode = context.compiledNode(sdpaNodeId);
        assertTrue(candidate.outputNodeIds().containsAll(sdpaNode.inputIds()));
        MetalPartitionPlan plan = (MetalPartitionPlan) new MetalRegionLegalityAdapter().tryCreatePlan(candidate, context);
        assertNotNull(plan);
        assertTrue(plan.producedOutputNodeIds().containsAll(sdpaNode.inputIds()));
        assertTrue(plan.lowering().dagSpec().outputNodeIds().containsAll(sdpaNode.inputIds()));
    }

    @Test
    void admitsDirectUnmaskedFloat32SdpaAfterNativeScaleParityVerification() {
        Tensor q = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "q", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "k", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "v", DataType.FLOAT32);
        Tensor out = specialSdpa(q, k, v, null, 0.5d);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        int sdpaNodeId = nodeId(context, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION);
        assertEquals("", MetalPartitionSupport.plannerUnsupportedReason(context.compiledNode(sdpaNodeId), context));
        assertNotNull(new MetalRegionLegalityAdapter().tryCreateStructuralCandidate(
                Set.of(sdpaNodeId),
                context,
                Set.of(PartitionValueRef.ofNode(sdpaNodeId))
        ));
    }

    @Test
    void directSdpaLowererEncodesScaleForFutureNativeEnablement() {
        Tensor q = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "q", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "k", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "v", DataType.FLOAT32);
        Tensor out = specialSdpa(q, k, v, null, 0.5d);
        PartitionPlanningContext context = planningContext(out);
        int sdpaNodeId = nodeId(context, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION);
        CompiledNode sdpa = context.compiledNode(sdpaNodeId);

        var lowered = new AcceleratorSubgraphLowerer().tryLower(
                new AcceleratorSubgraphSpec(
                        sdpaNodeId,
                        List.of(sdpaNodeId),
                        List.of(new backend.accelerator.dag.AcceleratorSubgraphOp(sdpaNodeId, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION)),
                        sdpa.inputIds(),
                        List.of(sdpaNodeId)
                ),
                context
        );

        assertNotNull(lowered);
        assertEquals(backend.accelerator.dag.AcceleratorDagNodeType.SDPA, lowered.dagSpec().nodes().getFirst().type());
        assertEquals(0.5f, Float.intBitsToFloat(lowered.dagSpec().nodes().getFirst().scalarValueBits()), 0.0f);
    }

    @Test
    void supportsDirectExternalBoolMaskedSdpaForMetal() {
        Tensor q = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "q", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "k", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "v", DataType.FLOAT32);
        Tensor mask = new Tensor(new byte[]{1, 0, 1, 1}, new int[]{1, 2, 2}, null, "mask", DataType.BOOL);
        Tensor out = specialSdpa(q, k, v, mask, 0.5d);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        int sdpaNodeId = nodeId(context, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION);
        assertEquals(
                "",
                MetalPartitionSupport.plannerUnsupportedReason(context.compiledNode(sdpaNodeId), context)
        );
        PartitionCandidate candidate = new MetalRegionLegalityAdapter().tryCreateStructuralCandidate(
                Set.of(sdpaNodeId),
                context,
                Set.of(PartitionValueRef.ofNode(sdpaNodeId))
        );
        assertNotNull(candidate);

        CompiledNode sdpa = context.compiledNode(sdpaNodeId);
        var lowered = new AcceleratorSubgraphLowerer().tryLower(
                new AcceleratorSubgraphSpec(
                        sdpaNodeId,
                        List.of(sdpaNodeId),
                        List.of(new backend.accelerator.dag.AcceleratorSubgraphOp(sdpaNodeId, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION)),
                        sdpa.inputIds(),
                        List.of(sdpaNodeId)
                ),
                context
        );
        assertNotNull(lowered);
        assertEquals(backend.accelerator.dag.AcceleratorDagNodeType.SDPA, lowered.dagSpec().nodes().getFirst().type());
        assertEquals(backend.accelerator.dag.AcceleratorDagValueRefKind.EXTERNAL_INPUT, lowered.dagSpec().nodes().getFirst().input3().kind());
    }

    @Test
    void supportsDirectBfloat16SdpaForMetal() {
        Tensor q = new Tensor(new double[]{1d, 0d, 0d, 1d}, new int[]{1, 2, 2}, null, "bf16SdpaQ", DataType.BFLOAT16);
        Tensor k = new Tensor(new double[]{1d, 0d, 0d, 1d}, new int[]{1, 2, 2}, null, "bf16SdpaK", DataType.BFLOAT16);
        Tensor v = new Tensor(new double[]{10d, 1d, 1d, 10d}, new int[]{1, 2, 2}, null, "bf16SdpaV", DataType.BFLOAT16);
        Tensor out = specialSdpa(q, k, v, null, 0.5d);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        int sdpaNodeId = nodeId(context, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION);
        assertEquals(
                "",
                MetalPartitionSupport.plannerUnsupportedReason(context.compiledNode(sdpaNodeId), context)
        );
        PartitionCandidate candidate = new MetalRegionLegalityAdapter().tryCreateStructuralCandidate(
                Set.of(sdpaNodeId),
                context,
                Set.of(PartitionValueRef.ofNode(sdpaNodeId))
        );
        assertNotNull(candidate);

        CompiledNode sdpa = context.compiledNode(sdpaNodeId);
        var lowered = new AcceleratorSubgraphLowerer().tryLower(
                new AcceleratorSubgraphSpec(
                        sdpaNodeId,
                        List.of(sdpaNodeId),
                        List.of(new backend.accelerator.dag.AcceleratorSubgraphOp(sdpaNodeId, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION)),
                        sdpa.inputIds(),
                        List.of(sdpaNodeId)
                ),
                context
        );
        assertNotNull(lowered);
        assertEquals(DataType.BFLOAT16, lowered.dagSpec().nodes().getFirst().outputDataType());
        assertTrue(lowered.dagSpec().externalInputs().stream().allMatch(input -> input.dataType() == DataType.BFLOAT16));
        assertTrue(lowered.manifest().loweredPrimitives().stream().allMatch(primitive -> primitive.dataType() == DataType.BFLOAT16));
    }

    @Test
    void supportsDirectCausalSdpaViaEffectiveBoolMask() {
        Tensor q = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "q", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "k", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "v", DataType.FLOAT32);
        Tensor out = specialSdpa(q, k, v, causalMask(), 0.5d);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        int sdpaNodeId = nodeId(context, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION);
        assertEquals(
                "",
                MetalPartitionSupport.plannerUnsupportedReason(context.compiledNode(sdpaNodeId), context)
        );
    }

    @Test
    void supportsDirectExternalAndCausalSdpaViaEffectiveBoolMask() {
        Tensor q = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "q", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "k", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "v", DataType.FLOAT32);
        Tensor mask = new Tensor(new byte[]{1, 0, 1, 1}, new int[]{1, 2, 2}, null, "mask", DataType.BOOL);
        Tensor out = specialSdpa(q, k, v, mask.logicalAnd(causalMask()), 0.5d);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        int sdpaNodeId = nodeId(context, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION);
        assertEquals(
                "",
                MetalPartitionSupport.plannerUnsupportedReason(context.compiledNode(sdpaNodeId), context)
        );
    }

    @Test
    void rejectsDirectBroadcastMaskSdpaWithMaskLayoutReason() {
        Tensor q = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "q", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "k", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "v", DataType.FLOAT32);
        Tensor mask = new Tensor(new byte[]{1, 0}, new int[]{1, 1, 2}, null, "mask", DataType.BOOL);
        Tensor out = specialSdpa(q, k, v, mask.expand(1, 2, 2), 0.5d);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        int sdpaNodeId = nodeId(context, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION);
        assertEquals(
                "UNSUPPORTED_LAYOUT: GPU_METAL SDPA mask input requires dense BOOL layout",
                MetalPartitionSupport.plannerUnsupportedReason(context.compiledNode(sdpaNodeId), context)
        );
    }

    @Test
    void rejectsDirectFloat64SdpaForMetal() {
        Tensor q = new Tensor(new double[]{1d, 0d, 0d, 1d}, new int[]{1, 2, 2}, null, "q64", DataType.FLOAT64);
        Tensor k = new Tensor(new double[]{1d, 0d, 0d, 1d}, new int[]{1, 2, 2}, null, "k64", DataType.FLOAT64);
        Tensor v = new Tensor(new double[]{10d, 1d, 1d, 10d}, new int[]{1, 2, 2}, null, "v64", DataType.FLOAT64);
        Tensor out = specialSdpa(q, k, v, null, 0.5d);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        int sdpaNodeId = nodeId(context, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION);
        assertEquals(
                backend.metal.MetalMpsCapabilities.unsupportedDTypeMessage(DataType.FLOAT64),
                MetalPartitionSupport.plannerUnsupportedReason(context.compiledNode(sdpaNodeId), context)
        );
        assertNull(new MetalRegionLegalityAdapter().tryCreateStructuralCandidate(
                Set.of(sdpaNodeId),
                context,
                Set.of(PartitionValueRef.ofNode(sdpaNodeId))
        ));
    }

    private static Partition partition(
            String id,
            PartitionTarget target,
            List<Integer> orderedNodeIds,
            List<Integer> externalInputNodeIds,
            List<PartitionValueRef> outputValueRefs,
            List<PartitionValueRef> requiredMaterialized
    ) {
        List<PartitionValue> values = orderedNodeIds.stream()
                .map(nodeId -> new PartitionValue(PartitionValueRef.ofNode(nodeId), nodeId))
                .toList();
        List<PartitionEdge> internalEdges = orderedNodeIds.size() < 2
                ? List.of()
                : java.util.stream.IntStream.range(0, orderedNodeIds.size() - 1)
                .mapToObj(i -> new PartitionEdge(orderedNodeIds.get(i), orderedNodeIds.get(i + 1)))
                .toList();
        return new Partition(
                id,
                target,
                orderedNodeIds,
                values,
                internalEdges,
                externalInputNodeIds,
                outputValueRefs,
                orderedNodeIds.getLast(),
                requiredMaterialized,
                List.of(),
                List.of(PartitionBoundaryReason.NONE),
                orderedNodeIds.size(),
                new graph.optimizer.partition.cost.AcceleratorPartitionScoreModel.CandidateMetrics(orderedNodeIds.size(), internalEdges.size(), externalInputNodeIds.size(), 0, Math.max(0, orderedNodeIds.size() - 1)),
                PartitionPlannerStrategy.GREEDY_MAX_REGION,
                new PartitionDecisionTrace(
                        PartitionPlannerStrategy.GREEDY_MAX_REGION,
                        target,
                        orderedNodeIds.getLast(),
                        true,
                        "test",
                        orderedNodeIds,
                        orderedNodeIds,
                        List.of(),
                        orderedNodeIds.size(),
                        0.0d,
                        0.0d,
                        0,
                        false,
                        -1
                )
        );
    }

    private static java.util.Map<Integer, java.util.List<CompiledNode>> consumers(List<CompiledNode> graph) {
        java.util.Map<Integer, java.util.List<CompiledNode>> consumers = new java.util.HashMap<>();
        for (CompiledNode node : graph) {
            consumers.computeIfAbsent(node.id(), ignored -> new java.util.ArrayList<>());
        }
        for (CompiledNode node : graph) {
            for (int inputId : node.inputIds()) {
                consumers.computeIfAbsent(inputId, ignored -> new java.util.ArrayList<>()).add(node);
            }
        }
        return consumers;
    }

    private static PartitionPlanningContext planningContext(Tensor out) {
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(out.topologicalSort());
        return new PartitionPlanningContext(
                RuntimeConfig.inferenceDefaults(),
                false,
                compiledNodes,
                CompiledTensorDescriptorBuilder.build(compiledNodes),
                consumers(compiledNodes)
        );
    }

    private static Set<Integer> operationNodeIds(PartitionPlanningContext context) {
        return context.compiledNodes().stream()
                .filter(node -> node.operation() != null)
                .map(CompiledNode::id)
                .collect(java.util.stream.Collectors.toSet());
    }

    private static MetalPartitionPlan planFor(Tensor out, Operation.OpType opType) {
        PartitionPlanningContext context = planningContext(out);
        int nodeId = nodeId(context, opType);
        MetalRegionLegalityAdapter adapter = new MetalRegionLegalityAdapter();
        PartitionCandidate candidate = adapter.tryCreateStructuralCandidate(
                Set.of(nodeId),
                context,
                Set.of(PartitionValueRef.ofNode(nodeId))
        );
        assertNotNull(candidate);
        MetalPartitionPlan plan = (MetalPartitionPlan) adapter.tryCreatePlan(candidate, context);
        assertNotNull(plan);
        return plan;
    }

    private static List<Integer> externalInputNodeIds(PartitionPlanningContext context, List<Integer> selectedNodeIds) {
        Set<Integer> selected = Set.copyOf(selectedNodeIds);
        java.util.LinkedHashSet<Integer> out = new java.util.LinkedHashSet<>();
        for (int nodeId : selectedNodeIds) {
            CompiledNode node = context.compiledNode(nodeId);
            if (node != null) {
                node.inputIds().stream().filter(inputId -> !selected.contains(inputId)).forEach(out::add);
            }
        }
        return List.copyOf(out);
    }

    private static Tensor specialLogSoftmax(Tensor input, int dimension) {
        return TensorPrimitiveBuilder.unary(
                input,
                input.getShapeUnsafe().clone(),
                new operations.reduction.logSoftmax(dimension),
                "legacyLogSoftmax",
                input.getDataType()
        );
    }

    private static Tensor specialSdpa(Tensor query, Tensor key, Tensor value, Tensor mask, double scale) {
        int[] outShape = query.getShapeUnsafe().clone();
        outShape[outShape.length - 1] = value.getShapeUnsafe()[value.getShapeUnsafe().length - 1];
        java.util.ArrayList<Tensor> inputs = new java.util.ArrayList<>();
        inputs.add(query);
        inputs.add(key);
        inputs.add(value);
        if (mask != null) {
            inputs.add(mask);
        }
        return TensorPrimitiveBuilder.nary(
                outShape,
                inputs,
                new operations.linalg.scaledDotProductAttention(scale, mask != null),
                "legacyScaledDotProductAttention",
                query.getDataType()
        );
    }

    private static Tensor causalMask() {
        return new Tensor(new byte[]{
                1, 0,
                1, 1
        }, new int[]{1, 2, 2}, null, "causal_mask", DataType.BOOL);
    }

    private static int nodeId(PartitionPlanningContext context, Operation.OpType opType) {
        return context.compiledNodes().stream()
                .filter(node -> node.operation() != null && node.operation().opType() == opType)
                .map(CompiledNode::id)
                .findFirst()
                .orElseThrow();
    }

    private static void assertContainsAll(String actual, String... expectedSubstrings) {
        for (String expected : expectedSubstrings) {
            assertTrue(actual.contains(expected), () -> "Expected '" + actual + "' to contain '" + expected + "'");
        }
    }

    private record SyntheticFusedOperation() implements Operation {
        @Override
        public OpType opType() {
            return OpType.FUSED;
        }

        @Override
        public String getExpression() {
            return "synthetic_fused";
        }
    }
}

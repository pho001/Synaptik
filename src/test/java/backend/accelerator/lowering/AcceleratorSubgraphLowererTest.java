package backend.accelerator.lowering;

import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.compile.intent.BackendIntentPlan;

import backend.ComputeBackend;
import backend.accelerator.dag.AcceleratorDagNodeType;
import backend.accelerator.dag.AcceleratorDagValueRefKind;
import backend.accelerator.dag.AcceleratorPostOpType;
import backend.accelerator.dag.AcceleratorSubgraphOp;
import backend.accelerator.dag.AcceleratorSubgraphSpec;
import config.runtime.RuntimeConfig;
import graph.CompiledNode;
import graph.compile.planning.partition.PartitionPlanningContext;
import operations.Operation;
import operations.layout.sliceGrad;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.internal.TensorPrimitiveBuilder;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AcceleratorSubgraphLowererTest {
    @Test
    void linearBiasReluProducesLinearBiasActivationSummary() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "linearInput", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{
                1f, 0f, 0f, 1f,
                0f, 1f, 1f, 0f,
                1f, 1f, 0f, 0f
        }, new int[]{3, 4}, null, "linearWeight", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.5f, -0.5f, 1f, -1f}, new int[]{4}, null, "linearBias", DataType.FLOAT32);
        Tensor linear = input.linear(weight, bias);
        Tensor out = linear.relu();
        PartitionPlanningContext context = planningContext(out);
        int linearNodeId = nodeId(context, Operation.OpType.LINEAR);
        int reluNodeId = nodeId(context, Operation.OpType.RELU);
        CompiledNode linearNode = context.compiledNode(linearNodeId);

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(
                ComputeBackend.GPU_METAL,
                new AcceleratorSubgraphSpec(
                        linearNodeId,
                        List.of(linearNodeId, reluNodeId),
                        List.of(
                                new AcceleratorSubgraphOp(linearNodeId, Operation.OpType.LINEAR),
                                new AcceleratorSubgraphOp(reluNodeId, Operation.OpType.RELU)
                        ),
                        linearNode.inputIds(),
                        List.of(reluNodeId)
                ),
                context
        );

        assertNotNull(result);
        assertEquals(GpuCompoundPatternType.LINEAR_BIAS_ACTIVATION, result.compoundSummary().patternType());
        assertTrue(result.compoundSummary().supported());
        assertTrue(result.dagSpec().nodes().stream().anyMatch(node -> node.type() == AcceleratorDagNodeType.LINEAR));
        assertTrue(result.dagSpec().nodes().stream().anyMatch(node -> node.type() == AcceleratorDagNodeType.RELU));
        assertNotNull(result.matMulSpec());
        assertTrue(result.matMulSpec().biasInputNodeId() >= 0);
        assertTrue(result.matMulSpec().postOps().stream().anyMatch(postOp -> postOp.type() == AcceleratorPostOpType.RELU));
    }

    @Test
    void logSoftmaxLowersAsSoftmaxThenLog() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "logSoftmaxInput", DataType.FLOAT32);
        Tensor out = specialLogSoftmax(input, 1);
        PartitionPlanningContext context = planningContext(out);
        CompiledNode node = context.compiledNode(nodeId(context, Operation.OpType.LOG_SOFTMAX));

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(spec(node), context);

        assertEquals(2, result.dagSpec().nodes().size());
        assertEquals(AcceleratorDagNodeType.SOFTMAX, result.dagSpec().nodes().get(0).type());
        assertEquals(AcceleratorDagNodeType.LOG, result.dagSpec().nodes().get(1).type());
        assertEquals(List.of(node.id()), result.dagSpec().outputNodeIds());
    }

    @Test
    void logSoftmaxKeepsOriginalCompiledNodeAsOutput() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "logSoftmaxOutputInput", DataType.FLOAT32);
        Tensor out = specialLogSoftmax(input, 1);
        PartitionPlanningContext context = planningContext(out);
        CompiledNode node = context.compiledNode(nodeId(context, Operation.OpType.LOG_SOFTMAX));

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(spec(node), context);

        assertEquals(node.id(), result.computeNodeId());
        assertEquals(node.id(), result.dagSpec().nodes().get(1).nodeId());
        assertEquals(List.of(1), result.dagSpec().outputNodeIndices());
        assertEquals(List.of(node.id()), result.dagSpec().outputNodeIds());
    }

    @Test
    void castLowersWithTargetDTypeAttribute() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "castLowerInput", DataType.FLOAT32);
        Tensor out = input.cast(DataType.BFLOAT16);
        PartitionPlanningContext context = planningContext(out);
        CompiledNode node = context.compiledNode(nodeId(context, Operation.OpType.CAST));

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(
                ComputeBackend.GPU_METAL,
                spec(node),
                context
        );

        assertNotNull(result);
        assertEquals(AcceleratorDagNodeType.CAST, result.dagSpec().nodes().getFirst().type());
        assertEquals(DataType.BFLOAT16, result.dagSpec().nodes().getFirst().outputDataType());
        assertEquals(3, result.dagSpec().nodes().getFirst().attribute0());
        assertEquals(List.of(node.id()), result.dagSpec().outputNodeIds());
    }

    @Test
    void windowLayoutPrimitivesLowerToDedicatedDagNodesWithGeometryAttributes() {
        Tensor axisInput = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "axisLowerInput", DataType.FLOAT32);
        Tensor axisOut = axisInput.unfold(0, 2, 1);
        PartitionPlanningContext axisContext = planningContext(axisOut);
        CompiledNode axisNode = axisContext.compiledNode(nodeId(axisContext, Operation.OpType.UNFOLD_AXIS));
        AcceleratorSubgraphLoweringResult axisResult = new AcceleratorSubgraphLowerer().tryLower(
                ComputeBackend.GPU_METAL,
                spec(axisNode),
                axisContext
        );

        assertNotNull(axisResult);
        assertEquals(AcceleratorDagNodeType.UNFOLD_AXIS, axisResult.dagSpec().nodes().getFirst().type());
        assertEquals(0, axisResult.dagSpec().nodes().getFirst().attribute0());
        assertEquals(2, axisResult.dagSpec().nodes().getFirst().attribute1());
        assertEquals(1, axisResult.dagSpec().nodes().getFirst().attribute2());

        Tensor image = new Tensor(new float[]{
                1f, 2f, 3f,
                4f, 5f, 6f,
                7f, 8f, 9f
        }, new int[]{1, 1, 3, 3}, null, "unfold2dLowerInput", DataType.FLOAT32);
        tensor.options.Window2dOptions options = tensor.options.Window2dOptions.of(2, 2);
        Tensor columns = image.unfold2d(options);
        PartitionPlanningContext unfoldContext = planningContext(columns);
        CompiledNode unfoldNode = unfoldContext.compiledNode(nodeId(unfoldContext, Operation.OpType.UNFOLD2D));
        AcceleratorSubgraphLoweringResult unfoldResult = new AcceleratorSubgraphLowerer().tryLower(
                ComputeBackend.GPU_METAL,
                spec(unfoldNode),
                unfoldContext
        );

        assertNotNull(unfoldResult);
        assertEquals(AcceleratorDagNodeType.UNFOLD2D, unfoldResult.dagSpec().nodes().getFirst().type());
        assertEquals(2, unfoldResult.dagSpec().nodes().getFirst().attribute0());
        assertEquals(2, unfoldResult.dagSpec().nodes().getFirst().attribute1());
        assertEquals(1, unfoldResult.dagSpec().nodes().getFirst().attribute2());
        assertEquals(1, unfoldResult.dagSpec().nodes().getFirst().attribute3());

        Tensor folded = columns.fold2d(new int[]{1, 1, 3, 3}, options);
        PartitionPlanningContext foldContext = planningContext(folded);
        CompiledNode foldNode = foldContext.compiledNode(nodeId(foldContext, Operation.OpType.FOLD2D));
        AcceleratorSubgraphLoweringResult foldResult = new AcceleratorSubgraphLowerer().tryLower(
                ComputeBackend.GPU_METAL,
                spec(foldNode),
                foldContext
        );

        assertNotNull(foldResult);
        assertEquals(AcceleratorDagNodeType.FOLD2D, foldResult.dagSpec().nodes().getFirst().type());
        assertEquals(2, foldResult.dagSpec().nodes().getFirst().attribute0());
        assertEquals(2, foldResult.dagSpec().nodes().getFirst().attribute1());
    }

    @Test
    void sliceGradLowersWithPadAttributes() {
        Tensor outGrad = new Tensor(new float[]{10f, 20f, 30f, 40f}, new int[]{2, 2}, null, "sliceGradLowerInput", DataType.FLOAT32);
        Tensor grad = TensorPrimitiveBuilder.unaryNoGrad(
                outGrad,
                new int[]{2, 4},
                new sliceGrad(new int[]{0, 1}, new int[]{0, 1}, new int[]{1, 1}, new int[]{2, 4}),
                "sliceGradLower",
                DataType.FLOAT32
        );
        PartitionPlanningContext context = planningContext(grad);
        CompiledNode node = context.compiledNode(nodeId(context, Operation.OpType.SLICE_GRAD));

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(
                ComputeBackend.GPU_METAL,
                spec(node),
                context
        );

        assertNotNull(result);
        assertEquals(AcceleratorDagNodeType.SLICE_GRAD, result.dagSpec().nodes().getFirst().type());
        assertEquals(0, result.dagSpec().nodes().getFirst().attribute0());
        assertEquals(1, result.dagSpec().nodes().getFirst().attribute1());
        assertEquals(0, result.dagSpec().nodes().getFirst().attribute4());
        assertEquals(1, result.dagSpec().nodes().getFirst().attribute5());
        assertEquals(List.of(node.id()), result.dagSpec().outputNodeIds());
    }

    @Test
    void unaryMathParityOpsLowerToDedicatedDagNodes() {
        Tensor input = new Tensor(new float[]{-1.25f, 0.0f, 2.75f}, new int[]{3}, null, "unaryMathLowerInput", DataType.FLOAT32);

        assertSingleUnaryMathLowering(input.erf(), Operation.OpType.ERF, AcceleratorDagNodeType.ERF);
        assertSingleUnaryMathLowering(input.floor(), Operation.OpType.FLOOR, AcceleratorDagNodeType.FLOOR);
        assertSingleUnaryMathLowering(input.ceil(), Operation.OpType.CEIL, AcceleratorDagNodeType.CEIL);
        assertSingleUnaryMathLowering(input.sign(), Operation.OpType.SIGN, AcceleratorDagNodeType.SIGN);
    }

    @Test
    void logSoftmaxManifestMapsOneOriginalOpToTwoPrimitives() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "logSoftmaxManifestInput", DataType.FLOAT32);
        Tensor out = specialLogSoftmax(input, 1);
        PartitionPlanningContext context = planningContext(out);
        CompiledNode node = context.compiledNode(nodeId(context, Operation.OpType.LOG_SOFTMAX));

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(ComputeBackend.GPU_METAL, spec(node), context);

        assertNotNull(result);
        assertEquals(ComputeBackend.GPU_METAL, result.manifest().backend());
        assertEquals("LOG_SOFTMAX", result.manifest().originalOps().getFirst().opType());
        assertEquals(List.of("p0", "p1"), result.manifest().originalOps().getFirst().loweredPrimitiveIds());
        List<String> primitiveTypes = result.manifest().loweredPrimitives().stream()
                .map(GpuLoweredPrimitiveManifest::primitiveType)
                .toList();
        assertTrue(primitiveTypes.contains("SOFTMAX"));
        assertTrue(primitiveTypes.contains("LOG"));
        assertTrue(result.manifest().loweredPrimitives().stream()
                .allMatch(primitive -> primitive.sourceOriginalNodeIds().contains(node.id())));
    }

    @Test
    void linearBiasReluManifestIncludesFusedSummaryAndAssumptions() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "linearManifestInput", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{
                1f, 0f, 0f, 1f,
                0f, 1f, 1f, 0f,
                1f, 1f, 0f, 0f
        }, new int[]{3, 4}, null, "linearManifestWeight", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.5f, -0.5f, 1f, -1f}, new int[]{4}, null, "linearManifestBias", DataType.FLOAT32);
        Tensor linear = input.linear(weight, bias);
        Tensor out = linear.relu();
        PartitionPlanningContext context = planningContext(out);
        int linearNodeId = nodeId(context, Operation.OpType.LINEAR);
        int reluNodeId = nodeId(context, Operation.OpType.RELU);
        CompiledNode linearNode = context.compiledNode(linearNodeId);

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(
                ComputeBackend.GPU_METAL,
                new AcceleratorSubgraphSpec(
                        linearNodeId,
                        List.of(linearNodeId, reluNodeId),
                        List.of(
                                new AcceleratorSubgraphOp(linearNodeId, Operation.OpType.LINEAR),
                                new AcceleratorSubgraphOp(reluNodeId, Operation.OpType.RELU)
                        ),
                        linearNode.inputIds(),
                        List.of(reluNodeId)
                ),
                context
        );

        assertNotNull(result);
        assertEquals(GpuCompoundPatternType.LINEAR_BIAS_ACTIVATION, result.manifest().fusedSummary().patternType());
        assertTrue(!result.manifest().inputAssumptions().isEmpty());
        assertTrue(!result.manifest().outputAssumptions().isEmpty());
        assertTrue(result.manifest().inputAssumptions().stream()
                .anyMatch(assumption -> "CONTIGUOUS".equals(assumption.layout())));
    }

    @Test
    void manifestRegionIdBackendAndLengthAreStable() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "stableManifestInput", DataType.FLOAT32);
        Tensor out = specialLogSoftmax(input, 1);
        PartitionPlanningContext context = planningContext(out);
        CompiledNode node = context.compiledNode(nodeId(context, Operation.OpType.LOG_SOFTMAX));

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(ComputeBackend.GPU_CUDA, spec(node), context);

        assertNotNull(result);
        assertEquals("gpu-gpu_cuda-region-" + node.id(), result.manifest().regionId());
        assertEquals(ComputeBackend.GPU_CUDA, result.manifest().backend());
        assertEquals(1, result.manifest().selectedRegionLength());
        assertEquals("2", result.manifest().backendExtensions().get("dagNodeCount"));
    }

    @Test
    void manifestRecordsDTypeResidencyAssumptions() {
        Tensor condition = new Tensor(new byte[]{1, 0, 1, 0}, new int[]{4}, null, "dtypeResidencyMask", DataType.BOOL);
        Tensor trueBranch = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "dtypeResidencyTrue", DataType.FLOAT32);
        Tensor falseBranch = new Tensor(new float[]{-1f, -2f, -3f, -4f}, new int[]{4}, null, "dtypeResidencyFalse", DataType.FLOAT32);
        Tensor where = Tensor.where(condition, trueBranch, falseBranch);
        Tensor bf16 = new Tensor(new float[]{1f, -2f, 3f, -4f}, new int[]{4}, null, "dtypeResidencyBf16", DataType.FLOAT32);
        bf16.setDataType(DataType.BFLOAT16);
        Tensor relu = bf16.relu();
        PartitionPlanningContext boolContext = planningContext(where);
        PartitionPlanningContext bf16Context = planningContext(relu);
        CompiledNode whereNode = boolContext.compiledNode(nodeId(boolContext, Operation.OpType.WHERE));
        CompiledNode reluNode = bf16Context.compiledNode(nodeId(bf16Context, Operation.OpType.RELU));

        AcceleratorSubgraphLoweringResult boolResult = new AcceleratorSubgraphLowerer().tryLower(
                ComputeBackend.GPU_METAL,
                spec(whereNode),
                boolContext
        );
        AcceleratorSubgraphLoweringResult bf16Result = new AcceleratorSubgraphLowerer().tryLower(
                ComputeBackend.GPU_CUDA,
                spec(reluNode),
                bf16Context
        );

        assertNotNull(boolResult);
        assertNotNull(bf16Result);
        String boolEvidence = String.join("\n", boolResult.manifest().backendExtensions().values());
        String bf16Rendered = GpuLoweredRegionManifestRenderer.renderCompact(bf16Result.manifest());
        assertTrue(boolResult.manifest().backendExtensions().keySet().stream()
                .anyMatch(key -> key.startsWith("dtypeResidency.")));
        assertTrue(boolEvidence.contains("backend=GPU_METAL"));
        assertTrue(boolEvidence.contains("dtype=BOOL"));
        assertTrue(bf16Rendered.contains("dtypeResidency"));
        assertTrue(bf16Rendered.contains("UNSUPPORTED_DTYPE"));
        assertTrue(bf16Rendered.contains("backend=GPU_CUDA"));
        assertTrue(bf16Rendered.contains("dtype=BFLOAT16"));
    }

    @Test
    void lowererRecordsElementwiseFusionSubpatternPrimitiveIds() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "subpatternA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 0f, 0f, 1f, 1f, 1f}, new int[]{3, 2}, null, "subpatternB", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.25f, -0.5f}, new int[]{2}, null, "subpatternBias", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor add = matmul.add(bias);
        Tensor relu = add.relu();
        Tensor out = relu.exp();
        PartitionPlanningContext context = planningContext(out);
        int matmulNodeId = nodeId(context, Operation.OpType.MATMUL);
        int addNodeId = nodeId(context, Operation.OpType.ADD);
        int reluNodeId = nodeId(context, Operation.OpType.RELU);
        int expNodeId = nodeId(context, Operation.OpType.EXP);
        List<Integer> selectedNodeIds = List.of(matmulNodeId, addNodeId, reluNodeId, expNodeId);

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(
                ComputeBackend.GPU_CUDA,
                new AcceleratorSubgraphSpec(
                        matmulNodeId,
                        selectedNodeIds,
                        List.of(
                                new AcceleratorSubgraphOp(matmulNodeId, Operation.OpType.MATMUL),
                                new AcceleratorSubgraphOp(addNodeId, Operation.OpType.ADD),
                                new AcceleratorSubgraphOp(reluNodeId, Operation.OpType.RELU),
                                new AcceleratorSubgraphOp(expNodeId, Operation.OpType.EXP)
                        ),
                        externalInputNodeIds(context, selectedNodeIds),
                        List.of(expNodeId)
                ),
                context
        );

        assertNotNull(result);
        GpuFusionSubpatternSummary subpattern = result.manifest().fusedSubpatterns().stream()
                .filter(candidate -> candidate.patternType() == GpuCompoundPatternType.ELEMENTWISE_CHAIN)
                .findFirst()
                .orElseThrow();
        assertEquals(List.of(addNodeId, reluNodeId, expNodeId), subpattern.originalOperationNodeIds());
        assertEquals(List.of("p1", "p2", "p3"), subpattern.loweredPrimitiveIds());
        assertEquals(3, subpattern.loweredPrimitiveCount());
    }

    @Test
    void lowererRecordsMatmulBiasActivationEpilogueSubpattern() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "epilogueA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 0f, 0f, 1f, 1f, 1f}, new int[]{3, 2}, null, "epilogueB", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.25f, -0.5f}, new int[]{2}, null, "epilogueBias", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor add = matmul.add(bias);
        Tensor out = add.relu();
        PartitionPlanningContext context = planningContext(out);
        int matmulNodeId = nodeId(context, Operation.OpType.MATMUL);
        int addNodeId = nodeId(context, Operation.OpType.ADD);
        int reluNodeId = nodeId(context, Operation.OpType.RELU);
        List<Integer> selectedNodeIds = List.of(matmulNodeId, addNodeId, reluNodeId);

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(
                ComputeBackend.GPU_METAL,
                new AcceleratorSubgraphSpec(
                        matmulNodeId,
                        selectedNodeIds,
                        List.of(
                                new AcceleratorSubgraphOp(matmulNodeId, Operation.OpType.MATMUL),
                                new AcceleratorSubgraphOp(addNodeId, Operation.OpType.ADD),
                                new AcceleratorSubgraphOp(reluNodeId, Operation.OpType.RELU)
                        ),
                        externalInputNodeIds(context, selectedNodeIds),
                        List.of(reluNodeId)
                ),
                context
        );

        assertNotNull(result);
        assertNotNull(result.matMulSpec());
        assertTrue(result.matMulSpec().biasInputNodeId() >= 0);
        assertTrue(result.matMulSpec().postOps().stream()
                .anyMatch(postOp -> postOp.type() == AcceleratorPostOpType.RELU));
        GpuFusionSubpatternSummary epilogue = result.manifest().fusedSubpatterns().stream()
                .filter(candidate -> candidate.patternType() == GpuCompoundPatternType.LINEAR_BIAS_ACTIVATION)
                .findFirst()
                .orElseThrow();
        assertEquals(selectedNodeIds, epilogue.originalOperationNodeIds());
        assertTrue(epilogue.loweredPrimitiveCount() >= 2);
        assertTrue(epilogue.detail().contains("epilogue"));
    }

    @Test
    void batchedMatMulLowersToDagWithoutLegacyMatrixSpec() {
        Tensor a = new Tensor(new float[12], new int[]{2, 2, 3}, null, "batchedMatMulDagA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[24], new int[]{2, 3, 4}, null, "batchedMatMulDagB", DataType.FLOAT32);
        Tensor out = a.matmul(b);
        PartitionPlanningContext context = planningContext(out);
        CompiledNode matmulNode = context.compiledNode(nodeId(context, Operation.OpType.MATMUL));

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(
                ComputeBackend.GPU_METAL,
                spec(matmulNode),
                context
        );

        assertNotNull(result);
        assertNull(result.matMulSpec());
        assertEquals(List.of(AcceleratorDagNodeType.MATMUL), result.dagSpec().nodes().stream()
                .map(nodeSpec -> nodeSpec.type())
                .toList());
        assertEquals(List.of(matmulNode.id()), result.dagSpec().outputNodeIds());
    }

    @Test
    void metalElementwiseParityGapOpsLowerToMpsGraphDagPrimitives() {
        Tensor a = new Tensor(new float[]{1f, 4f, 9f, 16f}, new int[]{4}, null, "metalParityA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{2f, 3f, 10f, 8f}, new int[]{4}, null, "metalParityB", DataType.FLOAT32);
        Tensor c = new Tensor(new float[]{3f, 3f, 7f, 12f}, new int[]{4}, null, "metalParityC", DataType.FLOAT32);
        Tensor min = a.min(b);
        Tensor max = min.max(c);
        Tensor out = max.pow(1.5);
        PartitionPlanningContext context = planningContext(out);
        int minNodeId = nodeId(context, Operation.OpType.MIN);
        int maxNodeId = nodeId(context, Operation.OpType.MAX);
        int powNodeId = nodeId(context, Operation.OpType.POW);
        List<Integer> selectedNodeIds = List.of(minNodeId, maxNodeId, powNodeId);

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(
                ComputeBackend.GPU_METAL,
                new AcceleratorSubgraphSpec(
                        minNodeId,
                        selectedNodeIds,
                        List.of(
                                new AcceleratorSubgraphOp(minNodeId, Operation.OpType.MIN),
                                new AcceleratorSubgraphOp(maxNodeId, Operation.OpType.MAX),
                                new AcceleratorSubgraphOp(powNodeId, Operation.OpType.POW)
                        ),
                        externalInputNodeIds(context, selectedNodeIds),
                        List.of(powNodeId)
                ),
                context
        );

        assertNotNull(result);
        List<AcceleratorDagNodeType> types = result.dagSpec().nodes().stream()
                .map(nodeSpec -> nodeSpec.type())
                .toList();
        assertEquals(List.of(
                AcceleratorDagNodeType.MIN,
                AcceleratorDagNodeType.MAX,
                AcceleratorDagNodeType.POW_SCALAR
        ), types);
        assertEquals(Float.floatToIntBits(1.5f), result.dagSpec().nodes().getLast().scalarValueBits());
        assertEquals(GpuCompoundPatternType.ELEMENTWISE_CHAIN, result.manifest().fusedSummary().patternType());
        assertTrue(result.manifest().fusedSummary().supported());
        assertEquals(List.of(powNodeId), result.dagSpec().outputNodeIds());
    }

    @Test
    void epilogueFusionRejectsUnsupportedActivationWithStableReason() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "badEpilogueA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 0f, 0f, 1f, 1f, 1f}, new int[]{3, 2}, null, "badEpilogueB", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.25f, -0.5f}, new int[]{2}, null, "badEpilogueBias", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor add = matmul.add(bias);
        Tensor out = specialSoftmax(add, 1);
        PartitionPlanningContext context = planningContext(out);
        int matmulNodeId = nodeId(context, Operation.OpType.MATMUL);
        int addNodeId = nodeId(context, Operation.OpType.ADD);
        int softmaxNodeId = nodeId(context, Operation.OpType.SOFTMAX);
        List<Integer> selectedNodeIds = List.of(matmulNodeId, addNodeId, softmaxNodeId);

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(
                ComputeBackend.GPU_METAL,
                new AcceleratorSubgraphSpec(
                        matmulNodeId,
                        selectedNodeIds,
                        List.of(
                                new AcceleratorSubgraphOp(matmulNodeId, Operation.OpType.MATMUL),
                                new AcceleratorSubgraphOp(addNodeId, Operation.OpType.ADD),
                                new AcceleratorSubgraphOp(softmaxNodeId, Operation.OpType.SOFTMAX)
                        ),
                        externalInputNodeIds(context, selectedNodeIds),
                        List.of(softmaxNodeId)
                ),
                context
        );

        assertNotNull(result);
        assertNull(result.matMulSpec());
        assertTrue(result.manifest().fusedSubpatterns().stream()
                .noneMatch(subpattern -> subpattern.patternType() == GpuCompoundPatternType.LINEAR_BIAS_ACTIVATION));
    }

    @Test
    void phaseNineteenLowersMultiOpRegionWithLayoutElementwiseAndSoftmaxPrimitives() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "phase19LayoutInput", DataType.FLOAT32);
        Tensor permuted = input.permute(1, 0);
        Tensor relu = permuted.relu();
        Tensor out = specialLogSoftmax(relu, 1);
        PartitionPlanningContext context = planningContext(out);
        int permuteNodeId = nodeId(context, Operation.OpType.PERMUTE);
        int reluNodeId = nodeId(context, Operation.OpType.RELU);
        int logSoftmaxNodeId = nodeId(context, Operation.OpType.LOG_SOFTMAX);
        List<Integer> selectedNodeIds = List.of(permuteNodeId, reluNodeId, logSoftmaxNodeId);

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(
                ComputeBackend.GPU_METAL,
                new AcceleratorSubgraphSpec(
                        permuteNodeId,
                        selectedNodeIds,
                        List.of(
                                new AcceleratorSubgraphOp(permuteNodeId, Operation.OpType.PERMUTE),
                                new AcceleratorSubgraphOp(reluNodeId, Operation.OpType.RELU),
                                new AcceleratorSubgraphOp(logSoftmaxNodeId, Operation.OpType.LOG_SOFTMAX)
                        ),
                        externalInputNodeIds(context, selectedNodeIds),
                        List.of(logSoftmaxNodeId)
                ),
                context
        );

        assertNotNull(result);
        assertTrue(result.manifest().selectedRegionLength() > 1);
        assertTrue(result.manifest().loweredPrimitives().size() > 1);
        assertEquals("4", result.manifest().backendExtensions().get("dagNodeCount"));
        List<String> primitiveTypes = result.manifest().loweredPrimitives().stream()
                .map(GpuLoweredPrimitiveManifest::primitiveType)
                .toList();
        assertTrue(primitiveTypes.contains("PERMUTE"));
        assertTrue(primitiveTypes.contains("RELU"));
        assertTrue(primitiveTypes.contains("SOFTMAX"));
        assertTrue(primitiveTypes.contains("LOG"));
    }

    @Test
    void metalLayoutExpandAndSelectLowerToNativeDagShapePrimitives() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f}, new int[]{1, 3}, null, "metalLayoutShapeInput", DataType.FLOAT32);
        Tensor expanded = input.expand(2, 3);
        Tensor selected = expanded.select(0, 1);
        Tensor out = selected.relu();
        PartitionPlanningContext context = planningContext(out);
        int expandNodeId = nodeId(context, Operation.OpType.EXPAND);
        int selectNodeId = nodeId(context, Operation.OpType.SELECT);
        int reluNodeId = nodeId(context, Operation.OpType.RELU);
        List<Integer> selectedNodeIds = List.of(expandNodeId, selectNodeId, reluNodeId);

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(
                ComputeBackend.GPU_METAL,
                new AcceleratorSubgraphSpec(
                        expandNodeId,
                        selectedNodeIds,
                        List.of(
                                new AcceleratorSubgraphOp(expandNodeId, Operation.OpType.EXPAND),
                                new AcceleratorSubgraphOp(selectNodeId, Operation.OpType.SELECT),
                                new AcceleratorSubgraphOp(reluNodeId, Operation.OpType.RELU)
                        ),
                        externalInputNodeIds(context, selectedNodeIds),
                        List.of(reluNodeId)
                ),
                context
        );

        assertNotNull(result);
        List<AcceleratorDagNodeType> types = result.dagSpec().nodes().stream()
                .map(nodeSpec -> nodeSpec.type())
                .toList();
        assertEquals(List.of(
                AcceleratorDagNodeType.EXPAND,
                AcceleratorDagNodeType.SELECT,
                AcceleratorDagNodeType.RELU
        ), types);
        assertEquals((0 & 0xFFFF) | (1 << 16), result.dagSpec().nodes().get(1).scalarValueBits());
        assertEquals(List.of(reluNodeId), result.dagSpec().outputNodeIds());
        assertTrue(result.manifest().loweredPrimitives().stream()
                .anyMatch(primitive -> "EXPAND".equals(primitive.primitiveType())));
        assertTrue(result.manifest().loweredPrimitives().stream()
                .anyMatch(primitive -> "SELECT".equals(primitive.primitiveType())));
    }

    @Test
    void specializedSdpaDoesNotLiftInternalRegionValuesToExternalInputs() {
        Tensor qSource = new Tensor(new float[]{
                1f, 0f,
                0f, 1f
        }, new int[]{1, 1, 2, 2}, null, "sdpaInternalQSource", DataType.FLOAT32);
        Tensor kSource = new Tensor(new float[]{
                1f, 2f,
                3f, 4f
        }, new int[]{1, 1, 2, 2}, null, "sdpaInternalKSource", DataType.FLOAT32);
        Tensor vSource = new Tensor(new float[]{
                5f, 6f,
                7f, 8f
        }, new int[]{1, 1, 2, 2}, null, "sdpaInternalVSource", DataType.FLOAT32);
        Tensor query = qSource.relu();
        Tensor key = kSource.relu();
        Tensor value = vSource.relu();
        Tensor keyTransposed = key.permute(0, 1, 3, 2);
        Tensor scores = query.matmul(keyTransposed);
        Tensor scaled = scores.mul(0.5);
        Tensor weights = scaled.softmax(3);
        Tensor out = weights.matmul(value);
        PartitionPlanningContext context = planningContext(out);
        List<Integer> selectedNodeIds = context.compiledNodes().stream()
                .filter(node -> node.operation() != null)
                .map(CompiledNode::id)
                .toList();
        java.util.Set<Integer> selected = java.util.Set.copyOf(selectedNodeIds);

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(
                ComputeBackend.GPU_METAL,
                new AcceleratorSubgraphSpec(
                        selectedNodeIds.getFirst(),
                        selectedNodeIds,
                        selectedNodeIds.stream()
                                .map(nodeId -> new AcceleratorSubgraphOp(
                                        nodeId,
                                        context.compiledNode(nodeId).operation().opType()
                                ))
                                .toList(),
                        externalInputNodeIds(context, selectedNodeIds),
                        List.of(selectedNodeIds.getLast())
                ),
                context
        );

        assertNotNull(result);
        assertTrue(result.dagSpec().externalInputs().stream()
                .noneMatch(input -> selected.contains(input.nodeId())));
        List<AcceleratorDagNodeType> types = result.dagSpec().nodes().stream()
                .map(nodeSpec -> nodeSpec.type())
                .toList();
        assertTrue(types.contains(AcceleratorDagNodeType.RELU));
        assertTrue(types.contains(AcceleratorDagNodeType.MATMUL));
        assertTrue(types.contains(AcceleratorDagNodeType.EXP));
        assertTrue(types.contains(AcceleratorDagNodeType.SUM));
    }

    @Test
    void metalSdpaWeightsPublicationLowersFromProducerAttentionDescriptor() {
        Tensor query = new Tensor(new float[]{
                1f, 0f,
                0f, 1f
        }, new int[]{1, 2, 2}, null, "sdpaWeightsQuery", DataType.FLOAT32);
        Tensor key = new Tensor(new float[]{
                1f, 0f,
                0f, 1f
        }, new int[]{1, 2, 2}, null, "sdpaWeightsKey", DataType.FLOAT32);
        Tensor value = new Tensor(new float[]{
                10f, 1f,
                1f, 10f
        }, new int[]{1, 2, 2}, null, "sdpaWeightsValue", DataType.FLOAT32);
        Tensor attention = specialSdpa(query, key, value, null, 1.0d);
        Tensor weights = TensorPrimitiveBuilder.unaryNoGrad(
                attention,
                new int[]{1, 2, 2},
                new operations.linalg.scaledDotProductAttentionWeights(),
                "sdpaWeightsPublication",
                DataType.FLOAT32
        );
        PartitionPlanningContext context = planningContext(weights);
        int weightsNodeId = nodeId(context, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS);

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(
                ComputeBackend.GPU_METAL,
                spec(context.compiledNode(weightsNodeId)),
                context
        );

        assertNotNull(result);
        assertEquals(List.of(AcceleratorDagNodeType.SDPA_WEIGHTS), result.dagSpec().nodes().stream()
                .map(nodeSpec -> nodeSpec.type())
                .toList());
        assertEquals(List.of(weightsNodeId), result.dagSpec().outputNodeIds());
        assertEquals(2, result.dagSpec().externalInputs().size());
        CompiledNode attentionNode = context.compiledNode(nodeId(context, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION));
        assertEquals(attentionNode.inputIds().subList(0, 2), result.dagSpec().externalInputs().stream()
                .map(input -> input.nodeId())
                .toList());
        assertTrue(result.manifest().loweredPrimitives().stream()
                .anyMatch(primitive -> "SDPA_WEIGHTS".equals(primitive.primitiveType())));
    }

    @Test
    void maskedSdpaBackwardLowersMaskAsFifthDagInput() {
        Tensor query = new Tensor(new float[]{
                1f, 0f,
                0f, 1f
        }, new int[]{1, 2, 2}, null, "maskedBackwardQuery", DataType.FLOAT32);
        Tensor key = new Tensor(new float[]{
                1f, 0f,
                0f, 1f
        }, new int[]{1, 2, 2}, null, "maskedBackwardKey", DataType.FLOAT32);
        Tensor value = new Tensor(new float[]{
                10f, 1f,
                1f, 10f
        }, new int[]{1, 2, 2}, null, "maskedBackwardValue", DataType.FLOAT32);
        Tensor mask = new Tensor(new byte[]{1, 0, 1, 1}, new int[]{1, 2, 2}, null, "maskedBackwardMask", DataType.BOOL);
        Tensor outGrad = new Tensor(new float[]{
                1f, 2f,
                3f, 4f
        }, new int[]{1, 2, 2}, null, "maskedBackwardOutGrad", DataType.FLOAT32);
        Tensor attention = specialSdpa(query, key, value, mask, 1.0d);
        Tensor queryGrad = TensorPrimitiveBuilder.binaryNoGrad(
                attention,
                outGrad,
                new int[]{1, 2, 2},
                new operations.linalg.scaledDotProductAttentionBackward(
                        operations.linalg.scaledDotProductAttentionBackward.OutputKind.QUERY
                ),
                "maskedBackwardQueryGrad",
                DataType.FLOAT32
        );
        PartitionPlanningContext context = planningContext(queryGrad);
        int backwardNodeId = nodeId(context, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_BACKWARD);
        CompiledNode backwardNode = context.compiledNode(backwardNodeId);
        CompiledNode attentionNode = context.compiledNode(nodeId(context, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION));

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(
                ComputeBackend.GPU_METAL,
                spec(backwardNode),
                context
        );

        assertNotNull(result);
        assertEquals(List.of(AcceleratorDagNodeType.SDPA_BACKWARD_QUERY), result.dagSpec().nodes().stream()
                .map(nodeSpec -> nodeSpec.type())
                .toList());
        assertEquals(5, result.dagSpec().externalInputs().size());
        assertEquals(attentionNode.inputIds().get(3), result.dagSpec().externalInputs().get(4).nodeId());
        assertEquals(AcceleratorDagValueRefKind.EXTERNAL_INPUT, result.dagSpec().nodes().getFirst().input4().kind());
        assertEquals(4, result.dagSpec().nodes().getFirst().input4().index());
    }

    @Test
    void valueSdpaBackwardLowersToRegionInternalWeightsDag() {
        Tensor query = new Tensor(new float[]{
                1f, 0f,
                0f, 1f
        }, new int[]{1, 2, 2}, null, "valueBackwardQuery", DataType.FLOAT32);
        Tensor key = new Tensor(new float[]{
                1f, 0f,
                0f, 1f
        }, new int[]{1, 2, 2}, null, "valueBackwardKey", DataType.FLOAT32);
        Tensor value = new Tensor(new float[]{
                10f, 1f,
                1f, 10f
        }, new int[]{1, 2, 2}, null, "valueBackwardValue", DataType.FLOAT32);
        Tensor outGrad = new Tensor(new float[]{
                1f, 2f,
                3f, 4f
        }, new int[]{1, 2, 2}, null, "valueBackwardOutGrad", DataType.FLOAT32);
        Tensor attention = specialSdpa(query, key, value, null, 1.0d);
        Tensor valueGrad = TensorPrimitiveBuilder.binaryNoGrad(
                attention,
                outGrad,
                new int[]{1, 2, 2},
                new operations.linalg.scaledDotProductAttentionBackward(
                        operations.linalg.scaledDotProductAttentionBackward.OutputKind.VALUE
                ),
                "valueBackwardGrad",
                DataType.FLOAT32
        );
        PartitionPlanningContext context = planningContext(valueGrad);
        CompiledNode backwardNode = context.compiledNode(nodeId(context, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_BACKWARD));

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(
                ComputeBackend.GPU_METAL,
                spec(backwardNode),
                context
        );

        assertNotNull(result);
        assertEquals(List.of(
                        AcceleratorDagNodeType.SDPA_WEIGHTS,
                        AcceleratorDagNodeType.PERMUTE,
                        AcceleratorDagNodeType.MATMUL
                ),
                result.dagSpec().nodes().stream().map(nodeSpec -> nodeSpec.type()).toList());
        assertEquals(List.of(backwardNode.id()), result.dagSpec().outputNodeIds());
        assertEquals(List.of(2), result.dagSpec().outputNodeIndices());
    }

    @Test
    void sdpaWeightsCanLowerInsideMultiNodeValueBackwardDag() {
        Tensor query = new Tensor(new float[]{
                1f, 0f,
                0f, 1f
        }, new int[]{1, 2, 2}, null, "valueDagQuery", DataType.FLOAT32);
        Tensor key = new Tensor(new float[]{
                1f, 0f,
                0f, 1f
        }, new int[]{1, 2, 2}, null, "valueDagKey", DataType.FLOAT32);
        Tensor value = new Tensor(new float[]{
                10f, 1f,
                1f, 10f
        }, new int[]{1, 2, 2}, null, "valueDagValue", DataType.FLOAT32);
        Tensor outGrad = new Tensor(new float[]{
                1f, 2f,
                3f, 4f
        }, new int[]{1, 2, 2}, null, "valueDagOutGrad", DataType.FLOAT32);
        Tensor attention = specialSdpa(query, key, value, null, 1.0d);
        Tensor weights = TensorPrimitiveBuilder.unaryNoGrad(
                attention,
                new int[]{1, 2, 2},
                new operations.linalg.scaledDotProductAttentionWeights(),
                "valueDagWeights",
                DataType.FLOAT32
        );
        Tensor weightsT = weights.permute(0, 2, 1);
        Tensor valueGrad = weightsT.matmul(outGrad);
        PartitionPlanningContext context = planningContext(valueGrad);
        int weightsNodeId = nodeId(context, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS);
        int permuteNodeId = nodeId(context, Operation.OpType.PERMUTE);
        int matmulNodeId = nodeId(context, Operation.OpType.MATMUL);
        List<Integer> selectedNodeIds = List.of(weightsNodeId, permuteNodeId, matmulNodeId);

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(
                ComputeBackend.GPU_METAL,
                new AcceleratorSubgraphSpec(
                        weightsNodeId,
                        selectedNodeIds,
                        selectedNodeIds.stream()
                                .map(nodeId -> new AcceleratorSubgraphOp(
                                        nodeId,
                                        context.compiledNode(nodeId).operation().opType()
                                ))
                                .toList(),
                        externalInputNodeIds(context, selectedNodeIds),
                        List.of(matmulNodeId)
                ),
                context
        );

        assertNotNull(result);
        assertEquals(List.of(
                        AcceleratorDagNodeType.SDPA_WEIGHTS,
                        AcceleratorDagNodeType.PERMUTE,
                        AcceleratorDagNodeType.MATMUL
                ),
                result.dagSpec().nodes().stream().map(nodeSpec -> nodeSpec.type()).toList());
        assertEquals(List.of(matmulNodeId), result.dagSpec().outputNodeIds());
        assertTrue(result.dagSpec().externalInputs().stream()
                .anyMatch(input -> input.nodeId() == context.compiledNode(nodeId(context, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION)).inputIds().get(0)));
        assertTrue(result.manifest().loweredPrimitives().stream()
                .anyMatch(primitive -> "SDPA_WEIGHTS".equals(primitive.primitiveType())));
    }

    @Test
    void metalIndexWriteAndGradientOpsLowerToScatterDagPrimitives() {
        Tensor base = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "scatterBase", DataType.FLOAT32);
        Tensor indices = new Tensor(new int[]{2, 1}, new int[]{2}, null, "scatterIndices", DataType.INT32);
        Tensor src = new Tensor(new float[]{10f, 20f}, new int[]{2}, null, "scatterSrc", DataType.FLOAT32);
        Tensor scatter = base.scatterAdd(indices, src, 1);
        Tensor gatherGrad = TensorPrimitiveBuilder.binaryNoGrad(
                indices,
                src,
                new int[]{2, 3},
                new operations.index.gatherGrad(1),
                "gatherGrad",
                DataType.FLOAT32
        );
        Tensor takeIndices = new Tensor(new int[]{2, 1, 0, 2}, new int[]{2, 2}, null, "takeGradIndices", DataType.INT32);
        Tensor takeOutGrad = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "takeGradOut", DataType.FLOAT32);
        Tensor takeGrad = TensorPrimitiveBuilder.binaryNoGrad(
                takeIndices,
                takeOutGrad,
                new int[]{2, 3},
                new operations.index.takeAlongAxisGrad(1),
                "takeAlongAxisGrad",
                DataType.FLOAT32
        );

        assertSingleNodeLowering(scatter, Operation.OpType.SCATTER_ADD, AcceleratorDagNodeType.SCATTER_ADD);
        assertSingleNodeLowering(gatherGrad, Operation.OpType.GATHER_GRAD, AcceleratorDagNodeType.GATHER_GRAD);
        assertSingleNodeLowering(takeGrad, Operation.OpType.TAKE_ALONG_AXIS_GRAD, AcceleratorDagNodeType.TAKE_ALONG_AXIS_GRAD);

        Tensor scatterElementsIndices = new Tensor(new int[]{2, 0, 1, 2}, new int[]{2, 2}, null, "scatterElementsIndices", DataType.INT32);
        Tensor scatterElementsUpdates = new Tensor(new float[]{10f, 20f, 30f, 40f}, new int[]{2, 2}, null, "scatterElementsUpdates", DataType.FLOAT32);
        Tensor scatterElements = base.scatterElements(scatterElementsIndices, scatterElementsUpdates, 1, operations.index.ScatterReduction.MAX);
        PartitionPlanningContext elementsContext = planningContext(scatterElements);
        CompiledNode elementsNode = elementsContext.compiledNode(nodeId(elementsContext, Operation.OpType.SCATTER_ELEMENTS));
        AcceleratorSubgraphLoweringResult elementsResult = new AcceleratorSubgraphLowerer().tryLower(
                ComputeBackend.GPU_METAL,
                spec(elementsNode),
                elementsContext
        );
        assertNotNull(elementsResult);
        assertEquals(AcceleratorDagNodeType.SCATTER_ELEMENTS, elementsResult.dagSpec().nodes().getFirst().type());
        assertEquals(1, elementsResult.dagSpec().nodes().getFirst().scalarValueBits());
        assertEquals(operations.index.ScatterReduction.MAX.ordinal(), elementsResult.dagSpec().nodes().getFirst().attribute0());

        Tensor scatterNdIndices = new Tensor(new int[]{0, 1, 1, 0}, new int[]{2, 2}, null, "scatterNdIndices", DataType.INT32);
        Tensor scatterNdUpdates = new Tensor(new float[]{10f, 20f}, new int[]{2}, null, "scatterNdUpdates", DataType.FLOAT32);
        Tensor scatterNd = base.scatterNd(scatterNdIndices, scatterNdUpdates, operations.index.ScatterReduction.ADD);
        PartitionPlanningContext ndContext = planningContext(scatterNd);
        CompiledNode ndNode = ndContext.compiledNode(nodeId(ndContext, Operation.OpType.SCATTER_ND));
        AcceleratorSubgraphLoweringResult ndResult = new AcceleratorSubgraphLowerer().tryLower(
                ComputeBackend.GPU_METAL,
                spec(ndNode),
                ndContext
        );
        assertNotNull(ndResult);
        assertEquals(AcceleratorDagNodeType.SCATTER_ND, ndResult.dagSpec().nodes().getFirst().type());
        assertEquals(0, ndResult.dagSpec().nodes().getFirst().scalarValueBits());
        assertEquals(operations.index.ScatterReduction.ADD.ordinal(), ndResult.dagSpec().nodes().getFirst().attribute0());
    }

    @Test
    void gatherNdLowersToDedicatedDagNodeWithBatchDims() {
        Tensor data = new Tensor(new float[]{
                1f, 2f, 3f,
                4f, 5f, 6f,
                7f, 8f, 9f,
                10f, 11f, 12f
        }, new int[]{2, 2, 3}, null, "gatherNdLowerData", DataType.FLOAT32);
        Tensor indices = new Tensor(new int[]{1, 0}, new int[]{2, 1, 1}, null, "gatherNdLowerIndices", DataType.INT32);
        Tensor out = data.gatherNd(indices, 1);
        PartitionPlanningContext context = planningContext(out);
        CompiledNode node = context.compiledNode(nodeId(context, Operation.OpType.GATHER_ND));

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(
                ComputeBackend.GPU_METAL,
                spec(node),
                context
        );

        assertNotNull(result);
        assertEquals(AcceleratorDagNodeType.GATHER_ND, result.dagSpec().nodes().getFirst().type());
        assertEquals(1, result.dagSpec().nodes().getFirst().scalarValueBits());
        assertEquals(List.of(node.id()), result.dagSpec().outputNodeIds());
    }

    @Test
    void phaseNineteenUnsupportedInternalPrimitiveRecordsCandidateShortening() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "phase19ShortenInput", DataType.FLOAT32);
        Tensor gamma = new Tensor(new float[]{1f, 1f}, new int[]{2}, null, "phase19ShortenGamma", DataType.FLOAT32);
        Tensor beta = new Tensor(new float[]{0f, 0f}, new int[]{2}, null, "phase19ShortenBeta", DataType.FLOAT32);
        Tensor relu = input.relu();
        Tensor out = relu.layerNorm(gamma, beta, 1.0e-5);
        PartitionPlanningContext context = planningContext(out);
        int reluNodeId = nodeId(context, Operation.OpType.RELU);
        int layerNormNodeId = nodeId(context, Operation.OpType.LAYER_NORM);
        List<Integer> selectedNodeIds = List.of(reluNodeId, layerNormNodeId);

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLowerShortenedCandidate(
                ComputeBackend.GPU_CUDA,
                new AcceleratorSubgraphSpec(
                        reluNodeId,
                        selectedNodeIds,
                        List.of(
                                new AcceleratorSubgraphOp(reluNodeId, Operation.OpType.RELU),
                                new AcceleratorSubgraphOp(layerNormNodeId, Operation.OpType.LAYER_NORM)
                        ),
                        externalInputNodeIds(context, selectedNodeIds),
                        List.of(layerNormNodeId)
                ),
                context
        );

        assertNotNull(result);
        assertEquals(List.of(reluNodeId), result.manifest().orderedNodeIds());
        assertEquals(GpuLoweringUnsupportedReason.DAG_CANDIDATE_SHORTENED, result.manifest().candidateSpan().reason());
        assertEquals(selectedNodeIds, result.manifest().candidateSpan().originalCandidateNodeIds());
        assertEquals(List.of(reluNodeId), result.manifest().candidateSpan().acceptedNodeIds());
        assertEquals(layerNormNodeId, result.manifest().candidateSpan().rejectedOriginalNodeId());
        assertTrue(result.manifest().rejections().stream()
                .anyMatch(rejection -> rejection.reason() == GpuLoweringUnsupportedReason.DAG_CANDIDATE_SHORTENED));
    }

    @Test
    void forwardReductionLowersWithAxisAndKeepDimsMetadata() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "sumInput", DataType.FLOAT32);
        Tensor out = input.sum(1, true);
        PartitionPlanningContext context = planningContext(out);
        CompiledNode node = context.compiledNode(nodeId(context, Operation.OpType.SUM));

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(spec(node), context);

        assertNotNull(result);
        assertEquals(AcceleratorDagNodeType.SUM, result.dagSpec().nodes().getFirst().type());
        int scalar = result.dagSpec().nodes().getFirst().scalarValueBits();
        assertEquals(1, scalar & 0xFFFF);
        assertTrue((scalar & (1 << 16)) != 0);
    }

    @Test
    void layerNormLowersToReductionAndElementwiseDagWithEpsilonMetadata() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "layerNormDagInput", DataType.FLOAT32);
        Tensor gamma = new Tensor(new float[]{1f, 1f, 1f}, new int[]{3}, null, "layerNormDagGamma", DataType.FLOAT32);
        Tensor beta = new Tensor(new float[]{0f, 0f, 0f}, new int[]{3}, null, "layerNormDagBeta", DataType.FLOAT32);
        Tensor out = input.layerNorm(gamma, beta, 1.0e-5);
        PartitionPlanningContext context = planningContext(out);
        CompiledNode node = context.compiledNode(nodeId(context, Operation.OpType.LAYER_NORM));

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(ComputeBackend.GPU_METAL, spec(node), context);

        assertNotNull(result);
        List<AcceleratorDagNodeType> types = result.dagSpec().nodes().stream()
                .map(nodeSpec -> nodeSpec.type())
                .toList();
        assertTrue(types.contains(AcceleratorDagNodeType.MEAN));
        assertTrue(types.contains(AcceleratorDagNodeType.SUB));
        assertTrue(types.contains(AcceleratorDagNodeType.MUL));
        assertTrue(types.contains(AcceleratorDagNodeType.ADD_SCALAR));
        assertTrue(types.contains(AcceleratorDagNodeType.SQRT));
        assertTrue(types.contains(AcceleratorDagNodeType.INV));
        assertTrue(types.contains(AcceleratorDagNodeType.ADD));
        assertEquals(2, types.stream().filter(type -> type == AcceleratorDagNodeType.MEAN).count());
        assertEquals(Float.floatToIntBits(1.0e-5f), result.dagSpec().nodes().stream()
                .filter(nodeSpec -> nodeSpec.type() == AcceleratorDagNodeType.ADD_SCALAR)
                .findFirst()
                .orElseThrow()
                .scalarValueBits());
        assertEquals(node.inputIds(), result.dagSpec().externalInputs().stream()
                .map(inputSpec -> inputSpec.nodeId())
                .toList());
        assertEquals(GpuCompoundPatternType.NORMALIZATION, result.manifest().fusedSummary().patternType());
        assertTrue(result.manifest().fusedSummary().supported());
        assertEquals(1, result.manifest().originalOps().size());
        assertTrue(result.manifest().originalOps().getFirst().loweredPrimitiveIds().size() > 1);
    }

    @Test
    void rmsNormLowersToMeanEpsilonSqrtInvAndGammaScale() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "rmsNormDagInput", DataType.FLOAT32);
        Tensor gamma = new Tensor(new float[]{1f, 1f, 1f}, new int[]{3}, null, "rmsNormDagGamma", DataType.FLOAT32);
        Tensor out = input.rmsNorm(gamma, 1.0e-4);
        PartitionPlanningContext context = planningContext(out);
        CompiledNode node = context.compiledNode(nodeId(context, Operation.OpType.RMS_NORM));

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(ComputeBackend.GPU_CUDA, spec(node), context);

        assertNotNull(result);
        List<AcceleratorDagNodeType> types = result.dagSpec().nodes().stream()
                .map(nodeSpec -> nodeSpec.type())
                .toList();
        assertTrue(types.contains(AcceleratorDagNodeType.MEAN));
        assertTrue(types.contains(AcceleratorDagNodeType.MUL));
        assertTrue(types.contains(AcceleratorDagNodeType.ADD_SCALAR));
        assertTrue(types.contains(AcceleratorDagNodeType.SQRT));
        assertTrue(types.contains(AcceleratorDagNodeType.INV));
        assertEquals(AcceleratorDagNodeType.MUL, result.dagSpec().nodes().getLast().type());
        assertEquals(Float.floatToIntBits(1.0e-4f), result.dagSpec().nodes().stream()
                .filter(nodeSpec -> nodeSpec.type() == AcceleratorDagNodeType.ADD_SCALAR)
                .findFirst()
                .orElseThrow()
                .scalarValueBits());
        assertEquals(node.inputIds(), result.dagSpec().externalInputs().stream()
                .map(inputSpec -> inputSpec.nodeId())
                .toList());
        assertEquals(GpuCompoundPatternType.NORMALIZATION, result.manifest().fusedSummary().patternType());
    }

    @Test
    void layerNormMultiAxisUsesRepeatedKeepDimsMeansFromLastAxisDown() {
        Tensor input = new Tensor(new float[64], new int[]{2, 4, 8, 1}, null, "layerNormMultiAxisInput", DataType.FLOAT32);
        Tensor gamma = new Tensor(new float[8], new int[]{8, 1}, null, "layerNormMultiAxisGamma", DataType.FLOAT32);
        Tensor beta = new Tensor(new float[8], new int[]{8, 1}, null, "layerNormMultiAxisBeta", DataType.FLOAT32);
        Tensor out = input.layerNorm(gamma, beta, 1.0e-5);
        PartitionPlanningContext context = planningContext(out);
        CompiledNode node = context.compiledNode(nodeId(context, Operation.OpType.LAYER_NORM));

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(spec(node), context);

        assertNotNull(result);
        List<Integer> meanModes = result.dagSpec().nodes().stream()
                .filter(nodeSpec -> nodeSpec.type() == AcceleratorDagNodeType.MEAN)
                .map(nodeSpec -> nodeSpec.scalarValueBits())
                .toList();
        assertEquals(List.of(
                (3 & 0xFFFF) | (1 << 16),
                (2 & 0xFFFF) | (1 << 16),
                (3 & 0xFFFF) | (1 << 16),
                (2 & 0xFFFF) | (1 << 16)
        ), meanModes);
        assertTrue(result.dagSpec().nodes().stream()
                .filter(nodeSpec -> nodeSpec.type() == AcceleratorDagNodeType.MEAN)
                .allMatch(nodeSpec -> nodeSpec.outputRank() == 4));
    }

    @Test
    void normalizationRejectsUnsupportedParameterShapeBeforeBackendAdmission() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "badNormInput", DataType.FLOAT32);
        Tensor gamma = new Tensor(new float[]{1f, 1f}, new int[]{2}, null, "badNormGamma", DataType.FLOAT32);
        Tensor beta = new Tensor(new float[]{0f, 0f}, new int[]{2}, null, "badNormBeta", DataType.FLOAT32);
        Tensor out = TensorPrimitiveBuilder.ternary(
                input,
                gamma,
                beta,
                input.getShape().clone(),
                new operations.normalization.layerNorm(1, 1.0e-5),
                "badLayerNorm",
                DataType.FLOAT32
        );
        PartitionPlanningContext context = planningContext(out);
        CompiledNode node = context.compiledNode(nodeId(context, Operation.OpType.LAYER_NORM));

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(spec(node), context);

        assertNull(result);
    }

    private static AcceleratorSubgraphSpec spec(CompiledNode node) {
        return new AcceleratorSubgraphSpec(
                node.id(),
                List.of(node.id()),
                List.of(new AcceleratorSubgraphOp(node.id(), node.operation().opType())),
                node.inputIds(),
                List.of(node.id())
        );
    }

    private static void assertSingleNodeLowering(Tensor out, Operation.OpType opType, AcceleratorDagNodeType expectedType) {
        PartitionPlanningContext context = planningContext(out);
        CompiledNode node = context.compiledNode(nodeId(context, opType));

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(
                ComputeBackend.GPU_METAL,
                spec(node),
                context
        );

        assertNotNull(result);
        assertEquals(expectedType, result.dagSpec().nodes().getFirst().type());
        assertEquals(List.of(node.id()), result.dagSpec().outputNodeIds());
        assertEquals(1, result.dagSpec().nodes().getFirst().scalarValueBits());
    }

    private static void assertSingleUnaryMathLowering(Tensor out, Operation.OpType opType, AcceleratorDagNodeType expectedType) {
        PartitionPlanningContext context = planningContext(out);
        CompiledNode node = context.compiledNode(nodeId(context, opType));

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(
                ComputeBackend.GPU_METAL,
                spec(node),
                context
        );

        assertNotNull(result);
        assertEquals(expectedType, result.dagSpec().nodes().getFirst().type());
        assertEquals(0, result.dagSpec().nodes().getFirst().scalarValueBits());
        assertEquals(DataType.FLOAT32, result.dagSpec().nodes().getFirst().outputDataType());
        assertEquals(List.of(node.id()), result.dagSpec().outputNodeIds());
    }

    private static PartitionPlanningContext planningContext(Tensor out) {
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
        return new PartitionPlanningContext(
                false,
                compiledNodes,
                CompiledTensorDescriptorBuilder.build(compiledNodes),
                consumers(compiledNodes)
        );
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

    private static Tensor specialSoftmax(Tensor input, int dimension) {
        return TensorPrimitiveBuilder.unary(
                input,
                input.getShapeUnsafe().clone(),
                new operations.reduction.softmax(dimension),
                "legacySoftmax",
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

    private static int nodeId(PartitionPlanningContext context, Operation.OpType opType) {
        return context.compiledNodes().stream()
                .filter(node -> node.operation() != null && node.operation().opType() == opType)
                .map(CompiledNode::id)
                .findFirst()
                .orElseThrow();
    }

    private static List<Integer> externalInputNodeIds(PartitionPlanningContext context, List<Integer> selectedNodeIds) {
        java.util.Set<Integer> selected = java.util.Set.copyOf(selectedNodeIds);
        java.util.LinkedHashSet<Integer> out = new java.util.LinkedHashSet<>();
        for (int nodeId : selectedNodeIds) {
            CompiledNode node = context.compiledNode(nodeId);
            if (node != null) {
                node.inputIds().stream().filter(inputId -> !selected.contains(inputId)).forEach(out::add);
            }
        }
        return List.copyOf(out);
    }
}

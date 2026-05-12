package backend.accelerator.lowering;

import backend.ComputeBackend;
import graph.optimizer.partition.PartitionPlanningContext;
import graph.CompiledNode;
import backend.accelerator.dag.AcceleratorDagInput;
import backend.accelerator.dag.AcceleratorDagNode;
import backend.accelerator.dag.AcceleratorDagNodeType;
import backend.accelerator.dag.AcceleratorDagSpec;
import backend.accelerator.dag.AcceleratorSubgraphSpec;
import backend.accelerator.dag.AcceleratorDagValueRef;
import backend.accelerator.dag.AcceleratorDagValueRefKind;
import backend.accelerator.dag.AcceleratorPostOp;
import backend.accelerator.dag.AcceleratorPostOpType;
import backend.accelerator.dag.AcceleratorSubgraphOp;
import backend.accelerator.residency.AcceleratorDTypeResidencyDecision;
import backend.accelerator.residency.AcceleratorDTypeResidencyPolicy;
import operations.Operation;
import operations.elementwise.unary.clampMax;
import operations.elementwise.unary.clampMin;
import operations.elementwise.unary.mulScalar;
import operations.elementwise.unary.pow;
import operations.elementwise.binary.maxGrad;
import operations.elementwise.binary.minGrad;
import operations.index.gather;
import operations.index.takeAlongAxis;
import operations.layout.expandDims;
import operations.layout.permute;
import operations.layout.select;
import operations.layout.squeeze;
import operations.nn.conv.conv2d;
import operations.nn.conv.conv2dBackwardInput;
import operations.nn.conv.conv2dBackwardInputGemm;
import operations.nn.conv.conv2dBackwardWeight;
import operations.nn.conv.conv2dBackwardWeightGemm;
import operations.nn.conv.conv2dGemm;
import operations.nn.pool.avgPool2d;
import operations.nn.pool.avgPool2dBackwardInput;
import operations.nn.pool.maxPool2d;
import operations.nn.pool.maxPool2dBackwardInput;
import operations.normalization.layerNorm;
import operations.normalization.rmsNorm;
import operations.reduction.reduceMaxGrad;
import operations.reduction.reduceMinGrad;
import operations.reduction.reduceMax;
import operations.reduction.reduceMin;
import operations.reduction.reduceAll;
import operations.reduction.reduceAny;
import operations.reduction.logSoftmax;
import operations.reduction.logSoftmaxGrad;
import operations.reduction.mean;
import operations.reduction.softmax;
import operations.reduction.softmaxGrad;
import operations.reduction.sum;
import operations.linalg.scaledDotProductAttention;
import operations.linalg.scaledDotProductAttentionBackward;
import operations.linalg.scaledDotProductAttentionWeights;
import operations.loss.crossEntropyLoss;
import operations.loss.crossEntropyLossIndices;
import operations.loss.crossEntropyLossIndicesGrad;
import operations.loss.nllLoss;
import tensor.DataType;
import tensor.loss.LossReduction;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Lowers a partition-planner accelerator candidate into the backend-neutral DAG ABI.
 *
 * <p>This is internal lowering SPI shared by CUDA and Metal. It returns {@code null}
 * when a candidate cannot be represented by the native graph bridge contract.</p>
 */
public final class AcceleratorSubgraphLowerer {
    /**
     * Attempts to lower a candidate subgraph into an accelerator DAG.
     *
     * @param subgraph candidate partition produced by accelerator legality planning
     * @param context compiled graph lookup used to inspect shapes, dtypes, and inputs
     * @return lowered DAG result, or {@code null} when the candidate is unsupported
     */
    public AcceleratorSubgraphLoweringResult tryLower(AcceleratorSubgraphSpec subgraph, PartitionPlanningContext context) {
        return tryLower(null, subgraph, context);
    }

    /**
     * Attempts to lower a candidate subgraph into an accelerator DAG for a concrete backend.
     *
     * @param backend backend that owns the selected accelerator region
     * @param subgraph candidate partition produced by accelerator legality planning
     * @param context compiled graph lookup used to inspect shapes, dtypes, and inputs
     * @return lowered DAG result, or {@code null} when the candidate is unsupported
     */
    public AcceleratorSubgraphLoweringResult tryLower(
            ComputeBackend backend,
            AcceleratorSubgraphSpec subgraph,
            PartitionPlanningContext context
    ) {
        if (subgraph == null || context == null || subgraph.orderedNodeIds().isEmpty()) {
            return null;
        }

        CompiledNode compute = context.compiledNode(subgraph.computeNodeId());
        if (compute == null || compute.operation() == null) {
            return null;
        }
        AcceleratorDagSpec dagSpec = buildDagSpec(subgraph, context);
        if (dagSpec == null) {
            return null;
        }
        AcceleratorMatMulSpec matMulSpec = tryBuildLegacyMatMulSpec(subgraph, context, compute);
        long estimatedWork = estimateWork(subgraph, context, matMulSpec, dagSpec);
        GpuCompoundRegionSummary compoundSummary = GpuCompoundPatternDetector.detect(
                backend,
                subgraph,
                context,
                dagSpec,
                matMulSpec
        );

        return new AcceleratorSubgraphLoweringResult(
                compute.id(),
                matMulSpec,
                dagSpec,
                estimatedWork,
                compoundSummary,
                buildManifest(backend, subgraph, context, dagSpec, compoundSummary)
        );
    }

    /**
     * Attempts to lower the full candidate and, when an unsupported internal primitive blocks the tail,
     * returns the longest supported prefix with explicit candidate-shortening evidence.
     */
    public AcceleratorSubgraphLoweringResult tryLowerShortenedCandidate(
            ComputeBackend backend,
            AcceleratorSubgraphSpec subgraph,
            PartitionPlanningContext context
    ) {
        AcceleratorSubgraphLoweringResult full = tryLower(backend, subgraph, context);
        if (full != null || subgraph == null || context == null || subgraph.orderedNodeIds().size() <= 1) {
            return full;
        }
        List<Integer> originalNodeIds = subgraph.orderedNodeIds();
        for (int acceptedLength = originalNodeIds.size() - 1; acceptedLength >= 1; acceptedLength--) {
            List<Integer> acceptedNodeIds = List.copyOf(originalNodeIds.subList(0, acceptedLength));
            AcceleratorSubgraphSpec shortened = shortenedSpec(subgraph, context, acceptedNodeIds);
            AcceleratorSubgraphLoweringResult shortenedResult = tryLower(backend, shortened, context);
            if (shortenedResult != null) {
                return withCandidateShortening(
                        shortenedResult,
                        originalNodeIds,
                        acceptedNodeIds,
                        originalNodeIds.get(acceptedLength)
                );
            }
        }
        return null;
    }

    private AcceleratorSubgraphSpec shortenedSpec(
            AcceleratorSubgraphSpec subgraph,
            PartitionPlanningContext context,
            List<Integer> acceptedNodeIds
    ) {
        List<AcceleratorSubgraphOp> acceptedOps = subgraph.ops().stream()
                .filter(op -> acceptedNodeIds.contains(op.nodeId()))
                .toList();
        return new AcceleratorSubgraphSpec(
                acceptedNodeIds.getFirst(),
                acceptedNodeIds,
                acceptedOps,
                externalInputNodeIds(context, acceptedNodeIds),
                List.of(acceptedNodeIds.getLast())
        );
    }

    private AcceleratorSubgraphLoweringResult withCandidateShortening(
            AcceleratorSubgraphLoweringResult result,
            List<Integer> originalNodeIds,
            List<Integer> acceptedNodeIds,
            int rejectedNodeId
    ) {
        GpuLoweredRegionManifest manifest = result.manifest();
        GpuLoweredRegionCandidateSpan candidateSpan = new GpuLoweredRegionCandidateSpan(
                originalNodeIds,
                acceptedNodeIds,
                rejectedNodeId,
                "p" + acceptedNodeIds.size(),
                GpuLoweringUnsupportedReason.DAG_CANDIDATE_SHORTENED
        );
        List<GpuLoweredRegionRejection> rejections = new ArrayList<>(manifest.rejections());
        rejections.add(new GpuLoweredRegionRejection(
                "candidate_span",
                rejectedNodeId,
                candidateSpan.rejectedPrimitiveId(),
                "",
                GpuLoweringUnsupportedReason.DAG_CANDIDATE_SHORTENED,
                "candidate shortened before execution because node " + rejectedNodeId + " is not lowerable inside the GPU DAG"
        ));
        GpuLoweredRegionManifest shortenedManifest = new GpuLoweredRegionManifest(
                manifest.regionId(),
                manifest.backend(),
                manifest.anchorNodeId(),
                manifest.orderedNodeIds(),
                manifest.externalInputNodeIds(),
                manifest.outputNodeIds(),
                acceptedNodeIds.size(),
                manifest.originalOps(),
                manifest.loweredPrimitives(),
                manifest.inputAssumptions(),
                manifest.outputAssumptions(),
                manifest.fusedSummary(),
                manifest.fusedSubpatterns(),
                rejections,
                candidateSpan,
                manifest.backendExtensions()
        );
        return new AcceleratorSubgraphLoweringResult(
                result.computeNodeId(),
                result.matMulSpec(),
                result.dagSpec(),
                result.estimatedWork(),
                result.compoundSummary(),
                shortenedManifest
        );
    }

    private List<Integer> externalInputNodeIds(PartitionPlanningContext context, List<Integer> selectedNodeIds) {
        List<Integer> out = new ArrayList<>();
        for (int nodeId : selectedNodeIds) {
            CompiledNode node = context.compiledNode(nodeId);
            if (node == null) {
                continue;
            }
            for (int inputId : node.inputIds()) {
                if (!selectedNodeIds.contains(inputId) && !out.contains(inputId)) {
                    out.add(inputId);
                }
            }
        }
        return List.copyOf(out);
    }

    private GpuLoweredRegionManifest buildManifest(
            ComputeBackend backend,
            AcceleratorSubgraphSpec subgraph,
            PartitionPlanningContext context,
            AcceleratorDagSpec dagSpec,
            GpuCompoundRegionSummary compoundSummary
    ) {
        ComputeBackend selectedBackend = backend == null ? ComputeBackend.CPU : backend;
        List<GpuLoweredRegionRejection> rejections = new ArrayList<>();
        Map<String, String> backendExtensions = buildDTypeResidencyEvidence(
                selectedBackend,
                subgraph,
                context,
                rejections
        );
        backendExtensions.put("dagNodeCount", Integer.toString(dagSpec.nodes().size()));
        return new GpuLoweredRegionManifest(
                "gpu-" + selectedBackend.name().toLowerCase(Locale.ROOT) + "-region-" + subgraph.computeNodeId(),
                selectedBackend,
                subgraph.computeNodeId(),
                subgraph.orderedNodeIds(),
                subgraph.externalInputNodeIds(),
                subgraph.outputNodeIds(),
                subgraph.orderedNodeIds().size(),
                buildOriginalOps(subgraph, context, dagSpec),
                buildLoweredPrimitives(dagSpec),
                buildInputAssumptions(subgraph, context),
                buildOutputAssumptions(subgraph, context),
                compoundSummary,
                buildFusionSubpatterns(subgraph, dagSpec, compoundSummary),
                rejections,
                GpuLoweredRegionCandidateSpan.none(subgraph.orderedNodeIds()),
                backendExtensions
        );
    }

    private List<GpuFusionSubpatternSummary> buildFusionSubpatterns(
            AcceleratorSubgraphSpec subgraph,
            AcceleratorDagSpec dagSpec,
            GpuCompoundRegionSummary compoundSummary
    ) {
        if (subgraph == null) {
            return List.of();
        }
        List<GpuFusionSubpatternSummary> out = new ArrayList<>();
        if (compoundSummary != null
                && compoundSummary.patternType() != GpuCompoundPatternType.NONE
                && !compoundSummary.supported()) {
            return List.of(GpuFusionSubpatternSummary.unsupported(
                    compoundSummary.patternType(),
                    compoundSummary.reason(),
                    subgraph.orderedNodeIds(),
                    compoundSummary.detail()
            ));
        }
        if (compoundSummary != null && compoundSummary.patternType() != GpuCompoundPatternType.NONE) {
            out.add(GpuFusionSubpatternSummary.supported(
                    compoundSummary.patternType(),
                    compoundSummary.orderedNodeIds().isEmpty() ? subgraph.orderedNodeIds() : compoundSummary.orderedNodeIds(),
                    primitiveIdsFor(compoundSummary.orderedNodeIds(), subgraph, dagSpec),
                    compoundSummary.patternType() == GpuCompoundPatternType.LINEAR_BIAS_ACTIVATION
                            ? "epilogue " + compoundSummary.detail()
                            : compoundSummary.detail()
            ));
        }
        for (List<Integer> chain : GpuCompoundPatternDetector.detectElementwiseSubchains(dagSpec)) {
            if (hasSubpattern(out, GpuCompoundPatternType.ELEMENTWISE_CHAIN, chain)) {
                continue;
            }
            out.add(GpuFusionSubpatternSummary.supported(
                    GpuCompoundPatternType.ELEMENTWISE_CHAIN,
                    chain,
                    primitiveIdsFor(chain, subgraph, dagSpec),
                    "region-internal elementwise subchain lowered through accelerator DAG"
            ));
        }
        return List.copyOf(out);
    }

    private boolean hasSubpattern(
            List<GpuFusionSubpatternSummary> subpatterns,
            GpuCompoundPatternType patternType,
            List<Integer> originalNodeIds
    ) {
        for (GpuFusionSubpatternSummary subpattern : subpatterns) {
            if (subpattern.patternType() == patternType && subpattern.originalOperationNodeIds().equals(originalNodeIds)) {
                return true;
            }
        }
        return false;
    }

    private List<String> primitiveIdsFor(
            List<Integer> originalNodeIds,
            AcceleratorSubgraphSpec subgraph,
            AcceleratorDagSpec dagSpec
    ) {
        List<Integer> span = originalNodeIds == null || originalNodeIds.isEmpty()
                ? subgraph.orderedNodeIds()
                : originalNodeIds;
        List<String> out = new ArrayList<>();
        for (int i = 0; i < dagSpec.nodes().size(); i++) {
            if (span.contains(dagSpec.nodes().get(i).nodeId())) {
                out.add("p" + i);
            }
        }
        return List.copyOf(out);
    }

    private Map<String, String> buildDTypeResidencyEvidence(
            ComputeBackend backend,
            AcceleratorSubgraphSpec subgraph,
            PartitionPlanningContext context,
            List<GpuLoweredRegionRejection> rejections
    ) {
        LinkedHashMap<String, String> out = new LinkedHashMap<>();
        for (int nodeId : subgraph.externalInputNodeIds()) {
            CompiledNode node = context.compiledNode(nodeId);
            if (node != null) {
                addDTypeResidencyDecision(
                        out,
                        rejections,
                        "input",
                        nodeId,
                        "",
                        AcceleratorDTypeResidencyPolicy.forExternalInput(backend, node.dataType())
                );
            }
        }
        for (int nodeId : subgraph.orderedNodeIds()) {
            CompiledNode node = context.compiledNode(nodeId);
            if (node != null) {
                addDTypeResidencyDecision(
                        out,
                        rejections,
                        "compute",
                        nodeId,
                        primitiveIdFor(nodeId, subgraph),
                        AcceleratorDTypeResidencyPolicy.forCompute(backend, node.dataType())
                );
                if (!subgraph.outputNodeIds().contains(nodeId)) {
                    addDTypeResidencyDecision(
                            out,
                            rejections,
                            "internalValue",
                            nodeId,
                            primitiveIdFor(nodeId, subgraph),
                            AcceleratorDTypeResidencyPolicy.forInternalValue(backend, node.dataType())
                    );
                }
            }
        }
        for (int nodeId : subgraph.outputNodeIds()) {
            CompiledNode node = context.compiledNode(nodeId);
            if (node != null) {
                addDTypeResidencyDecision(
                        out,
                        rejections,
                        "output",
                        nodeId,
                        primitiveIdFor(nodeId, subgraph),
                        AcceleratorDTypeResidencyPolicy.forOutput(backend, node.dataType())
                );
            }
        }
        return out;
    }

    private void addDTypeResidencyDecision(
            Map<String, String> backendExtensions,
            List<GpuLoweredRegionRejection> rejections,
            String role,
            int nodeId,
            String primitiveId,
            AcceleratorDTypeResidencyDecision decision
    ) {
        String key = "dtypeResidency." + role + "." + nodeId;
        backendExtensions.put(key, decision.detail());
        if (!decision.rejected()) {
            return;
        }
        rejections.add(new GpuLoweredRegionRejection(
                "dtype_residency." + role,
                nodeId,
                primitiveId,
                "",
                decision.reason(),
                "dtypeResidency " + decision.detail()
        ));
    }

    private String primitiveIdFor(int nodeId, AcceleratorSubgraphSpec subgraph) {
        int index = subgraph.orderedNodeIds().indexOf(nodeId);
        return index < 0 ? "" : "p" + index;
    }

    private List<GpuLoweredRegionOriginalOp> buildOriginalOps(
            AcceleratorSubgraphSpec subgraph,
            PartitionPlanningContext context,
            AcceleratorDagSpec dagSpec
    ) {
        List<GpuLoweredPrimitiveManifest> primitives = buildLoweredPrimitives(dagSpec);
        List<GpuLoweredRegionOriginalOp> out = new ArrayList<>(subgraph.orderedNodeIds().size());
        for (int nodeId : subgraph.orderedNodeIds()) {
            CompiledNode node = context.compiledNode(nodeId);
            if (node == null || node.operation() == null) {
                continue;
            }
            List<String> primitiveIds = primitives.stream()
                    .filter(primitive -> primitive.sourceOriginalNodeIds().contains(nodeId))
                    .map(GpuLoweredPrimitiveManifest::primitiveId)
                    .toList();
            out.add(new GpuLoweredRegionOriginalOp(
                    nodeId,
                    node.operation().opType().name(),
                    node.inputIds(),
                    subgraph.outputNodeIds().contains(nodeId) ? List.of(nodeId) : List.of(),
                    node.dataType(),
                    shapeList(node.shape()),
                    primitiveIds,
                    List.of()
            ));
        }
        return List.copyOf(out);
    }

    private List<GpuLoweredPrimitiveManifest> buildLoweredPrimitives(AcceleratorDagSpec dagSpec) {
        List<GpuLoweredPrimitiveManifest> out = new ArrayList<>(dagSpec.nodes().size());
        for (int i = 0; i < dagSpec.nodes().size(); i++) {
            AcceleratorDagNode node = dagSpec.nodes().get(i);
            out.add(new GpuLoweredPrimitiveManifest(
                    "p" + i,
                    node.type().name(),
                    List.of(node.nodeId()),
                    inputRefs(node),
                    "node:" + i,
                    node.outputDataType(),
                    outputShape(node),
                    List.of()
            ));
        }
        return List.copyOf(out);
    }

    private List<GpuLoweredRegionValueAssumption> buildInputAssumptions(
            AcceleratorSubgraphSpec subgraph,
            PartitionPlanningContext context
    ) {
        List<GpuLoweredRegionValueAssumption> out = new ArrayList<>(subgraph.externalInputNodeIds().size());
        for (int nodeId : subgraph.externalInputNodeIds()) {
            CompiledNode node = context.compiledNode(nodeId);
            if (node != null) {
                out.add(valueAssumption(node, "input"));
            }
        }
        return List.copyOf(out);
    }

    private List<GpuLoweredRegionValueAssumption> buildOutputAssumptions(
            AcceleratorSubgraphSpec subgraph,
            PartitionPlanningContext context
    ) {
        List<GpuLoweredRegionValueAssumption> out = new ArrayList<>(subgraph.outputNodeIds().size());
        for (int nodeId : subgraph.outputNodeIds()) {
            CompiledNode node = context.compiledNode(nodeId);
            if (node != null) {
                out.add(valueAssumption(node, "output"));
            }
        }
        return List.copyOf(out);
    }

    private GpuLoweredRegionValueAssumption valueAssumption(CompiledNode node, String role) {
        int[] shape = node.shape();
        return new GpuLoweredRegionValueAssumption(
                node.id(),
                role,
                node.dataType(),
                shape.length,
                shapeList(shape),
                layoutFor(node),
                node.contiguous(),
                node.hasStorageOffset(),
                node.storageOffset()
        );
    }

    private String layoutFor(CompiledNode node) {
        if (node == null) {
            return "UNKNOWN";
        }
        if (node.hasStorageOffset()) {
            return "STORAGE_OFFSET_VIEW";
        }
        if (node.contiguous()) {
            return "CONTIGUOUS";
        }
        return "STRIDED_VIEW";
    }

    private List<String> inputRefs(AcceleratorDagNode node) {
        List<String> refs = new ArrayList<>(5);
        addRef(refs, node.input0());
        addRef(refs, node.input1());
        addRef(refs, node.input2());
        addRef(refs, node.input3());
        addRef(refs, node.input4());
        return List.copyOf(refs);
    }

    private void addRef(List<String> refs, AcceleratorDagValueRef ref) {
        if (ref == null || ref.kind() == AcceleratorDagValueRefKind.NONE) {
            return;
        }
        refs.add(ref.kind().name().toLowerCase(Locale.ROOT) + ":" + ref.index());
    }

    private List<Integer> outputShape(AcceleratorDagNode node) {
        return switch (node.outputRank()) {
            case 1 -> List.of(node.outputDim0());
            case 2 -> List.of(node.outputDim0(), node.outputDim1());
            case 3 -> List.of(node.outputDim0(), node.outputDim1(), node.outputDim2());
            default -> List.of(node.outputDim0(), node.outputDim1(), node.outputDim2(), node.outputDim3());
        };
    }

    private List<Integer> shapeList(int[] shape) {
        return Arrays.stream(shape == null ? new int[0] : shape).boxed().toList();
    }

    private boolean isMetalFloatingDType(DataType dataType) {
        return dataType == DataType.FLOAT32 || dataType == DataType.BFLOAT16;
    }

    private AcceleratorMatMulSpec tryBuildLegacyMatMulSpec(
            AcceleratorSubgraphSpec subgraph,
            PartitionPlanningContext context,
            CompiledNode compute
    ) {
        if (compute == null || compute.operation() == null) {
            return null;
        }
        if (subgraph.outputNodeIds().size() > 1) {
            return null;
        }
        boolean linear = compute.operation().opType() == Operation.OpType.LINEAR;
        if (!linear && compute.operation().opType() != Operation.OpType.MATMUL) {
            return null;
        }
        if ((!linear && compute.inputIds().size() != 2) || (linear && (compute.inputIds().size() != 2 && compute.inputIds().size() != 3))) {
            return null;
        }
        CompiledNode left = context.compiledNode(compute.inputIds().getFirst());
        CompiledNode right = context.compiledNode(compute.inputIds().get(1));
        if (left == null || right == null) {
            return null;
        }
        int[] leftShape = left.shape();
        int[] rightShape = right.shape();
        int[] outShape = compute.shape();
        if (linear) {
            if (leftShape.length != 2 || rightShape.length != 2 || outShape.length != 2) {
                return null;
            }
        } else if (leftShape.length < 2 || leftShape.length > 4
                || rightShape.length < 2 || rightShape.length > 4
                || outShape.length < 2 || outShape.length > 4) {
            return null;
        }
        int biasInputNodeId = -1;
        boolean biasVector = false;
        ArrayList<AcceleratorPostOp> postOps = new ArrayList<>();
        int outputNodeId = compute.id();

        if (linear && compute.inputIds().size() == 3) {
            CompiledNode bias = context.compiledNode(compute.inputIds().get(2));
            if (bias == null || !bias.contiguous() || bias.hasStorageOffset()) {
                return null;
            }
            int[] biasShape = bias.shape();
            if (biasShape.length != 1 || biasShape[0] != rightShape[1]) {
                return null;
            }
            biasInputNodeId = bias.id();
            biasVector = true;
        }

        List<Integer> nodeIds = subgraph.orderedNodeIds();
        if (nodeIds.size() >= 2) {
            CompiledNode second = context.compiledNode(nodeIds.get(1));
            if (second == null || second.operation() == null) {
                return null;
            }
            if (!linear && second.operation().opType() == Operation.OpType.ADD) {
                if (second.inputIds().size() != 2 || !second.inputIds().contains(compute.id())) {
                    return null;
                }
                int otherInputId = second.inputIds().getFirst() == compute.id()
                        ? second.inputIds().get(1)
                        : second.inputIds().getFirst();
                CompiledNode bias = context.compiledNode(otherInputId);
                if (bias == null || bias.dataType() != compute.dataType() || !bias.contiguous() || bias.hasStorageOffset()) {
                    return null;
                }
                int[] biasShape = bias.shape();
                if (biasShape.length == 1 && biasShape[0] == rightShape[1]) {
                    biasVector = true;
                } else if (!(biasShape.length == 2 && biasShape[0] == outShape[0] && biasShape[1] == rightShape[1])) {
                    return null;
                }
                biasInputNodeId = otherInputId;
                outputNodeId = second.id();
            }
        }
        int postOpStart = (!linear && nodeIds.size() >= 2 && context.compiledNode(nodeIds.get(1)) != null
                && context.compiledNode(nodeIds.get(1)).operation() != null
                && context.compiledNode(nodeIds.get(1)).operation().opType() == Operation.OpType.ADD
                && outputNodeId == nodeIds.get(1))
                ? 2
                : 1;
        for (int i = postOpStart; i < nodeIds.size(); i++) {
            CompiledNode node = context.compiledNode(nodeIds.get(i));
            if (node == null || node.operation() == null) {
                return null;
            }
            AcceleratorPostOp resolved = resolvePostOp(node, outputNodeId, context, compute);
            if (resolved == null) {
                return null;
            }
            postOps.add(resolved);
            outputNodeId = node.id();
        }
        return new AcceleratorMatMulSpec(
                compute.inputIds().getFirst(),
                compute.inputIds().get(1),
                biasInputNodeId,
                outputNodeId,
                leftShape[0],
                rightShape[1],
                leftShape[1],
                biasVector,
                postOps
        );
    }

    private long estimateWork(
            AcceleratorSubgraphSpec subgraph,
            PartitionPlanningContext context,
            AcceleratorMatMulSpec matMulSpec,
            AcceleratorDagSpec dagSpec
    ) {
        if (matMulSpec != null) {
            return (long) matMulSpec.m() * matMulSpec.n() * matMulSpec.k()
                    + (long) matMulSpec.m() * matMulSpec.n() * Math.max(0, dagSpec.nodes().size() - 1);
        }
        long estimatedWork = 0L;
        for (int nodeId : subgraph.orderedNodeIds()) {
            CompiledNode node = context.compiledNode(nodeId);
            if (node != null) {
                estimatedWork += Math.max(1, node.flatDataSize());
            }
        }
        return Math.max(1L, estimatedWork);
    }

    private AcceleratorDagSpec buildDagSpec(AcceleratorSubgraphSpec subgraph, PartitionPlanningContext context) {
        AcceleratorDagSpec specializedDenseLoss = tryBuildDenseLossDagSpec(subgraph, context);
        if (specializedDenseLoss != null) {
            return specializedDenseLoss;
        }
        AcceleratorDagSpec specializedNormalization = tryBuildNormalizationDagSpec(subgraph, context);
        if (specializedNormalization != null) {
            return specializedNormalization;
        }
        AcceleratorDagSpec specializedLogSoftmax = tryBuildLogSoftmaxDagSpec(subgraph, context);
        if (specializedLogSoftmax != null) {
            return specializedLogSoftmax;
        }
        AcceleratorDagSpec specializedSdpaBackward = tryBuildSpecializedSdpaBackwardDagSpec(subgraph, context);
        if (specializedSdpaBackward != null) {
            return specializedSdpaBackward;
        }
        AcceleratorDagSpec specializedSdpaWeights = tryBuildSpecializedSdpaWeightsDagSpec(subgraph, context);
        if (specializedSdpaWeights != null) {
            return specializedSdpaWeights;
        }
        AcceleratorDagSpec canonicalSdpaPrimitive = tryBuildCanonicalBfloat16SdpaPrimitiveDagSpec(subgraph, context);
        if (canonicalSdpaPrimitive != null) {
            return canonicalSdpaPrimitive;
        }
        AcceleratorDagSpec specializedSdpa = tryBuildSpecializedSdpaDagSpec(subgraph, context);
        if (specializedSdpa != null) {
            return specializedSdpa;
        }
        Map<Integer, Integer> externalInputIndex = new HashMap<>();
        List<AcceleratorDagInput> externalInputs = new ArrayList<>(subgraph.externalInputNodeIds().size());
        for (int i = 0; i < subgraph.externalInputNodeIds().size(); i++) {
            int nodeId = subgraph.externalInputNodeIds().get(i);
            CompiledNode node = context.compiledNode(nodeId);
            if (node == null) {
                return null;
            }
            int[] shape = node.shape();
            if (shape.length < 1 || shape.length > 4) {
                return null;
            }
            externalInputIndex.put(nodeId, i);
            externalInputs.add(new AcceleratorDagInput(nodeId, java.util.Arrays.stream(shape).boxed().toList(), node.dataType()));
        }

        Map<Integer, Integer> loweredNodeIndex = new HashMap<>();
        List<AcceleratorDagNode> nodes = new ArrayList<>(subgraph.orderedNodeIds().size());
        for (int nodeId : subgraph.orderedNodeIds()) {
            CompiledNode node = context.compiledNode(nodeId);
            if (node == null || node.operation() == null) {
                return null;
            }
            AcceleratorDagValueRef input0 = resolveDagValueRef(node.inputIds(), 0, externalInputIndex, loweredNodeIndex);
            AcceleratorDagValueRef input1 = resolveDagValueRef(node.inputIds(), 1, externalInputIndex, loweredNodeIndex);
            AcceleratorDagValueRef input2 = resolveDagValueRef(node.inputIds(), 2, externalInputIndex, loweredNodeIndex);
            AcceleratorDagValueRef input3 = resolveDagValueRef(node.inputIds(), 3, externalInputIndex, loweredNodeIndex);
            AcceleratorDagValueRef input4 = resolveDagValueRef(node.inputIds(), 4, externalInputIndex, loweredNodeIndex);
            if ((node.inputIds().size() >= 1 && input0.kind() == AcceleratorDagValueRefKind.NONE)
                    || (node.inputIds().size() >= 2 && input1.kind() == AcceleratorDagValueRefKind.NONE)
                    || (node.inputIds().size() >= 3 && input2.kind() == AcceleratorDagValueRefKind.NONE)
                    || (node.inputIds().size() >= 4 && input3.kind() == AcceleratorDagValueRefKind.NONE)
                    || (node.inputIds().size() >= 5 && input4.kind() == AcceleratorDagValueRefKind.NONE)) {
                return null;
            }
            int[] shape = node.shape();
            if (shape.length < 1 || shape.length > 4) {
                return null;
            }
            if (node.operation().opType() == Operation.OpType.LOG_SOFTMAX) {
                if (!(node.operation() instanceof logSoftmax op) || node.inputIds().size() != 1) {
                    return null;
                }
                nodes.add(new AcceleratorDagNode(
                        nodeId,
                        AcceleratorDagNodeType.SOFTMAX,
                        input0,
                        AcceleratorDagValueRef.none(),
                        AcceleratorDagValueRef.none(),
                        AcceleratorDagValueRef.none(),
                        op.getDimension(),
                        shape.length,
                        shape[0],
                        shape.length >= 2 ? shape[1] : 1,
                        shape.length >= 3 ? shape[2] : 1,
                        shape.length >= 4 ? shape[3] : 1,
                        node.dataType()
                ));
                nodes.add(new AcceleratorDagNode(
                        nodeId,
                        AcceleratorDagNodeType.LOG,
                        AcceleratorDagValueRef.nodeOutput(nodes.size() - 1),
                        AcceleratorDagValueRef.none(),
                        AcceleratorDagValueRef.none(),
                        AcceleratorDagValueRef.none(),
                        0,
                        shape.length,
                        shape[0],
                        shape.length >= 2 ? shape[1] : 1,
                        shape.length >= 3 ? shape[2] : 1,
                        shape.length >= 4 ? shape[3] : 1,
                        node.dataType()
                ));
                loweredNodeIndex.put(nodeId, nodes.size() - 1);
                continue;
            }
            if (node.operation().opType() == Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_BACKWARD) {
                if (!(node.operation() instanceof scaledDotProductAttentionBackward backwardOp)) {
                    return null;
                }
                AcceleratorDagNode backwardDagNode = buildGenericSdpaBackwardNode(
                        node,
                        backwardOp,
                        context,
                        externalInputIndex,
                        externalInputs,
                        loweredNodeIndex
                );
                if (backwardDagNode == null) {
                    return null;
                }
                nodes.add(backwardDagNode);
                loweredNodeIndex.put(nodeId, nodes.size() - 1);
                continue;
            }
            if (node.operation().opType() == Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS) {
                if (!(node.operation() instanceof scaledDotProductAttentionWeights)) {
                    return null;
                }
                AcceleratorDagNode weightsDagNode = buildGenericSdpaWeightsNode(
                        node,
                        context,
                        externalInputIndex,
                        externalInputs,
                        loweredNodeIndex
                );
                if (weightsDagNode == null) {
                    return null;
                }
                nodes.add(weightsDagNode);
                loweredNodeIndex.put(nodeId, nodes.size() - 1);
                continue;
            }
            AcceleratorDagNodeType type = resolveDagNodeType(node.operation().opType());
            if (type == null) {
                return null;
            }
            int scalarValueBits = resolveScalarValueBits(node, context);
            if (type == AcceleratorDagNodeType.PERMUTE && scalarValueBits == Integer.MIN_VALUE) {
                return null;
            }
            if (type == AcceleratorDagNodeType.SELECT && scalarValueBits == Integer.MIN_VALUE) {
                return null;
            }
            if (isReduction(type) && scalarValueBits == Integer.MIN_VALUE) {
                return null;
            }
            if (isIndexGather(type) && scalarValueBits == Integer.MIN_VALUE) {
                return null;
            }
            if (isIndexWriteOrGradient(type) && scalarValueBits == Integer.MIN_VALUE) {
                return null;
            }
            if ((type == AcceleratorDagNodeType.CONV2D
                    || type == AcceleratorDagNodeType.CONV2D_BACKWARD_INPUT
                    || type == AcceleratorDagNodeType.CONV2D_BACKWARD_WEIGHT)
                    && scalarValueBits == Integer.MIN_VALUE) {
                return null;
            }
            if ((type == AcceleratorDagNodeType.MAX_POOL2D
                    || type == AcceleratorDagNodeType.AVG_POOL2D
                    || type == AcceleratorDagNodeType.AVG_POOL2D_BACKWARD_INPUT
                    || type == AcceleratorDagNodeType.MAX_POOL2D_BACKWARD_INPUT)
                    && scalarValueBits == Integer.MIN_VALUE) {
                return null;
            }
            nodes.add(new AcceleratorDagNode(
                    nodeId,
                    type,
                    input0,
                    input1,
                    input2,
                    input3,
                    input4,
                    scalarValueBits,
                    shape.length,
                    shape[0],
                    shape.length >= 2 ? shape[1] : 1,
                    shape.length >= 3 ? shape[2] : 1,
                    shape.length >= 4 ? shape[3] : 1,
                    node.dataType()
            ));
            loweredNodeIndex.put(nodeId, nodes.size() - 1);
        }
        List<Integer> outputNodeIds = subgraph.outputNodeIds().isEmpty()
                ? List.of(subgraph.orderedNodeIds().getLast())
                : subgraph.outputNodeIds();
        List<Integer> outputNodeIndexes = new ArrayList<>(outputNodeIds.size());
        for (int outputNodeId : outputNodeIds) {
            Integer outputNodeIndex = loweredNodeIndex.get(outputNodeId);
            if (outputNodeIndex == null) {
                return null;
            }
            outputNodeIndexes.add(outputNodeIndex);
        }
        return new AcceleratorDagSpec(externalInputs, nodes, outputNodeIndexes, outputNodeIds);
    }

    private AcceleratorDagNode buildGenericSdpaBackwardNode(
            CompiledNode outputNode,
            scaledDotProductAttentionBackward backwardOp,
            PartitionPlanningContext context,
            Map<Integer, Integer> externalInputIndex,
            List<AcceleratorDagInput> externalInputs,
            Map<Integer, Integer> loweredNodeIndex
    ) {
        if (outputNode == null || outputNode.inputIds().size() != 2) {
            return null;
        }
        CompiledNode attentionOutNode = context.compiledNode(outputNode.inputIds().getFirst());
        CompiledNode outGradNode = context.compiledNode(outputNode.inputIds().get(1));
        if (attentionOutNode == null
                || attentionOutNode.operation() == null
                || attentionOutNode.operation().opType() != Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION
                || !(attentionOutNode.operation() instanceof scaledDotProductAttention attentionOp)
                || outGradNode == null) {
            return null;
        }
        if (attentionOutNode.inputIds().size() != 3 && attentionOutNode.inputIds().size() != 4) {
            return null;
        }
        CompiledNode queryNode = context.compiledNode(attentionOutNode.inputIds().get(0));
        CompiledNode keyNode = context.compiledNode(attentionOutNode.inputIds().get(1));
        CompiledNode valueNode = context.compiledNode(attentionOutNode.inputIds().get(2));
        CompiledNode maskNode = attentionOutNode.inputIds().size() == 4
                ? context.compiledNode(attentionOutNode.inputIds().get(3))
                : null;
        if (queryNode == null || keyNode == null || valueNode == null) {
            return null;
        }
        DataType dtype = outputNode.dataType();
        if (!isMetalFloatingDType(dtype)
                || queryNode.dataType() != dtype
                || keyNode.dataType() != dtype
                || valueNode.dataType() != dtype
                || outGradNode.dataType() != dtype) {
            return null;
        }
        if (maskNode != null && maskNode.dataType() != DataType.BOOL) {
            return null;
        }
        AcceleratorDagValueRef queryRef = resolveOrAppendDagInput(queryNode, externalInputIndex, externalInputs, loweredNodeIndex);
        AcceleratorDagValueRef keyRef = resolveOrAppendDagInput(keyNode, externalInputIndex, externalInputs, loweredNodeIndex);
        AcceleratorDagValueRef valueRef = resolveOrAppendDagInput(valueNode, externalInputIndex, externalInputs, loweredNodeIndex);
        AcceleratorDagValueRef outGradRef = resolveOrAppendDagInput(outGradNode, externalInputIndex, externalInputs, loweredNodeIndex);
        AcceleratorDagValueRef maskRef = maskNode == null
                ? AcceleratorDagValueRef.none()
                : resolveOrAppendDagInput(maskNode, externalInputIndex, externalInputs, loweredNodeIndex);
        if (queryRef.kind() == AcceleratorDagValueRefKind.NONE
                || keyRef.kind() == AcceleratorDagValueRefKind.NONE
                || valueRef.kind() == AcceleratorDagValueRefKind.NONE
                || outGradRef.kind() == AcceleratorDagValueRefKind.NONE
                || (maskNode != null && maskRef.kind() == AcceleratorDagValueRefKind.NONE)) {
            return null;
        }
        AcceleratorDagNodeType nodeType = switch (backwardOp.getOutputKind()) {
            case QUERY -> AcceleratorDagNodeType.SDPA_BACKWARD_QUERY;
            case KEY -> AcceleratorDagNodeType.SDPA_BACKWARD_KEY;
            case VALUE -> AcceleratorDagNodeType.SDPA_BACKWARD_VALUE;
        };
        int[] outputShape = outputNode.shape();
        if (outputShape.length < 1 || outputShape.length > 4) {
            return null;
        }
        return new AcceleratorDagNode(
                outputNode.id(),
                nodeType,
                queryRef,
                keyRef,
                valueRef,
                outGradRef,
                maskRef,
                Float.floatToIntBits((float) attentionOp.getScale()),
                outputShape.length,
                outputShape[0],
                outputShape.length >= 2 ? outputShape[1] : 1,
                outputShape.length >= 3 ? outputShape[2] : 1,
                outputShape.length >= 4 ? outputShape[3] : 1,
                outputNode.dataType()
        );
    }

    private AcceleratorDagNode buildGenericSdpaWeightsNode(
            CompiledNode weightsNode,
            PartitionPlanningContext context,
            Map<Integer, Integer> externalInputIndex,
            List<AcceleratorDagInput> externalInputs,
            Map<Integer, Integer> loweredNodeIndex
    ) {
        if (weightsNode == null || weightsNode.inputIds().size() != 1) {
            return null;
        }
        CompiledNode attentionOutNode = context.compiledNode(weightsNode.inputIds().getFirst());
        if (attentionOutNode == null
                || attentionOutNode.operation() == null
                || attentionOutNode.operation().opType() != Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION
                || !(attentionOutNode.operation() instanceof scaledDotProductAttention attentionOp)
                || (attentionOutNode.inputIds().size() != 3 && attentionOutNode.inputIds().size() != 4)) {
            return null;
        }
        CompiledNode queryNode = context.compiledNode(attentionOutNode.inputIds().get(0));
        CompiledNode keyNode = context.compiledNode(attentionOutNode.inputIds().get(1));
        CompiledNode maskNode = attentionOutNode.inputIds().size() == 4
                ? context.compiledNode(attentionOutNode.inputIds().get(3))
                : null;
        if (queryNode == null || keyNode == null) {
            return null;
        }
        DataType dtype = weightsNode.dataType();
        if (!isMetalFloatingDType(dtype)
                || queryNode.dataType() != dtype
                || keyNode.dataType() != dtype) {
            return null;
        }
        if (maskNode != null && maskNode.dataType() != DataType.BOOL) {
            return null;
        }
        AcceleratorDagValueRef queryRef = resolveOrAppendDagInput(queryNode, externalInputIndex, externalInputs, loweredNodeIndex);
        AcceleratorDagValueRef keyRef = resolveOrAppendDagInput(keyNode, externalInputIndex, externalInputs, loweredNodeIndex);
        AcceleratorDagValueRef maskRef = maskNode == null
                ? AcceleratorDagValueRef.none()
                : resolveOrAppendDagInput(maskNode, externalInputIndex, externalInputs, loweredNodeIndex);
        if (queryRef.kind() == AcceleratorDagValueRefKind.NONE
                || keyRef.kind() == AcceleratorDagValueRefKind.NONE
                || (maskNode != null && maskRef.kind() == AcceleratorDagValueRefKind.NONE)) {
            return null;
        }
        int[] outputShape = weightsNode.shape();
        if (outputShape.length < 1 || outputShape.length > 4) {
            return null;
        }
        return new AcceleratorDagNode(
                weightsNode.id(),
                AcceleratorDagNodeType.SDPA_WEIGHTS,
                queryRef,
                keyRef,
                maskRef,
                AcceleratorDagValueRef.none(),
                Float.floatToIntBits((float) attentionOp.getScale()),
                outputShape.length,
                outputShape[0],
                outputShape.length >= 2 ? outputShape[1] : 1,
                outputShape.length >= 3 ? outputShape[2] : 1,
                outputShape.length >= 4 ? outputShape[3] : 1,
                weightsNode.dataType()
        );
    }

    private AcceleratorDagValueRef resolveOrAppendDagInput(
            CompiledNode node,
            Map<Integer, Integer> externalInputIndex,
            List<AcceleratorDagInput> externalInputs,
            Map<Integer, Integer> loweredNodeIndex
    ) {
        if (node == null) {
            return AcceleratorDagValueRef.none();
        }
        Integer loweredIndex = loweredNodeIndex.get(node.id());
        if (loweredIndex != null) {
            return AcceleratorDagValueRef.nodeOutput(loweredIndex);
        }
        Integer externalIndex = externalInputIndex.get(node.id());
        if (externalIndex != null) {
            return AcceleratorDagValueRef.externalInput(externalIndex);
        }
        int[] shape = node.shape();
        if (shape.length < 1 || shape.length > 4) {
            return AcceleratorDagValueRef.none();
        }
        int index = externalInputs.size();
        externalInputIndex.put(node.id(), index);
        externalInputs.add(new AcceleratorDagInput(node.id(), Arrays.stream(shape).boxed().toList(), node.dataType()));
        return AcceleratorDagValueRef.externalInput(index);
    }

    private AcceleratorDagSpec tryBuildDenseLossDagSpec(AcceleratorSubgraphSpec subgraph, PartitionPlanningContext context) {
        if (subgraph == null || context == null || subgraph.orderedNodeIds().isEmpty()) {
            return null;
        }
        int lossNodeId = subgraph.outputNodeIds().isEmpty()
                ? subgraph.orderedNodeIds().getLast()
                : subgraph.outputNodeIds().getLast();
        CompiledNode lossNode = context.compiledNode(lossNodeId);
        if (lossNode == null || lossNode.operation() == null || lossNode.inputIds().size() != 2) {
            return null;
        }
        Operation.OpType lossOpType = lossNode.operation().opType();
        boolean nll = lossOpType == Operation.OpType.NLL_LOSS;
        boolean ce = lossOpType == Operation.OpType.CROSS_ENTROPY_LOSS;
        if (!nll && !ce) {
            return null;
        }
        int classAxis = denseLossClassAxis(lossNode);
        if (classAxis < 0) {
            return null;
        }
        CompiledNode scoreInput = context.compiledNode(lossNode.inputIds().getFirst());
        CompiledNode targets = context.compiledNode(lossNode.inputIds().get(1));
        DataType lossDType = lossNode.dataType();
        if (!isMetalFloatingDType(lossDType)
                || !isSupportedDenseLossValue(scoreInput)
                || !isSupportedDenseLossValue(targets)
                || scoreInput.dataType() != lossDType
                || targets.dataType() != lossDType) {
            return null;
        }

        CompiledNode externalScores = scoreInput;
        boolean logSoftmaxInputIsInternal = false;
        int logSoftmaxNodeId = scoreInput.id();
        if (nll
                && subgraph.orderedNodeIds().contains(scoreInput.id())
                && scoreInput.operation() != null
                && scoreInput.operation().opType() == Operation.OpType.LOG_SOFTMAX
                && scoreInput.operation() instanceof logSoftmax logSoftmaxOp
                && scoreInput.inputIds().size() == 1
                && logSoftmaxOp.getDimension() == classAxis) {
            externalScores = context.compiledNode(scoreInput.inputIds().getFirst());
            logSoftmaxInputIsInternal = true;
        }
        if (!isSupportedDenseLossValue(externalScores) || externalScores.dataType() != lossDType) {
            return null;
        }
        int[] scoreShape = scoreInput.shape();
        if (!Arrays.equals(scoreShape, targets.shape())
                || !Arrays.equals(lossNode.shape(), new int[]{1})
                || classAxis >= scoreShape.length
                || scoreShape.length < 1
                || scoreShape.length > 4) {
            return null;
        }
        if (logSoftmaxInputIsInternal && !subgraph.orderedNodeIds().equals(List.of(logSoftmaxNodeId, lossNode.id()))) {
            return null;
        }
        if (!logSoftmaxInputIsInternal && subgraph.orderedNodeIds().size() != 1) {
            return null;
        }

        List<AcceleratorDagInput> externalInputs = List.of(
                new AcceleratorDagInput(externalScores.id(), shapeList(externalScores.shape()), externalScores.dataType()),
                new AcceleratorDagInput(targets.id(), shapeList(targets.shape()), targets.dataType())
        );
        List<AcceleratorDagNode> nodes = new ArrayList<>();
        AcceleratorDagValueRef scoresRef = AcceleratorDagValueRef.externalInput(0);
        if (ce || logSoftmaxInputIsInternal) {
            int softmaxSourceNodeId = logSoftmaxInputIsInternal ? logSoftmaxNodeId : lossNode.id();
            scoresRef = addNode(nodes, softmaxSourceNodeId, AcceleratorDagNodeType.SOFTMAX, scoresRef, AcceleratorDagValueRef.none(), classAxis, scoreShape, lossDType);
            scoresRef = addNode(nodes, softmaxSourceNodeId, AcceleratorDagNodeType.LOG, scoresRef, AcceleratorDagValueRef.none(), 0, scoreShape, lossDType);
        }
        AcceleratorDagValueRef weighted = addNode(
                nodes,
                lossNode.id(),
                AcceleratorDagNodeType.MUL,
                scoresRef,
                AcceleratorDagValueRef.externalInput(1),
                0,
                scoreShape,
                lossDType
        );
        AcceleratorDagValueRef reduced = addAllAxesSumNodes(nodes, lossNode.id(), weighted, scoreShape, lossDType);
        float scale = -1.0f / denseLossSampleCount(scoreShape, classAxis);
        addNode(
                nodes,
                lossNode.id(),
                AcceleratorDagNodeType.MUL_SCALAR,
                reduced,
                AcceleratorDagValueRef.none(),
                Float.floatToIntBits(scale),
                new int[]{1},
                lossDType
        );
        return new AcceleratorDagSpec(externalInputs, nodes, List.of(nodes.size() - 1), List.of(lossNode.id()));
    }

    private int denseLossClassAxis(CompiledNode lossNode) {
        Operation operation = lossNode.operation();
        return switch (operation.opType()) {
            case NLL_LOSS -> operation instanceof nllLoss op ? op.getClassDimension() : -1;
            case CROSS_ENTROPY_LOSS -> operation instanceof crossEntropyLoss op ? op.getClassDimension() : -1;
            default -> -1;
        };
    }

    private boolean isSupportedDenseLossValue(CompiledNode node) {
        return node != null
                && isMetalFloatingDType(node.dataType())
                && node.shape().length >= 1
                && node.shape().length <= 4
                && node.contiguous()
                && !node.hasStorageOffset();
    }

    private AcceleratorDagValueRef addAllAxesSumNodes(
            List<AcceleratorDagNode> nodes,
            int nodeId,
            AcceleratorDagValueRef inputRef,
            int[] inputShape,
            DataType dataType
    ) {
        AcceleratorDagValueRef current = inputRef;
        int[] currentShape = inputShape.clone();
        for (int axis = inputShape.length - 1; axis >= 0; axis--) {
            currentShape = removeAxisForReduction(currentShape, axis);
            current = addNode(
                    nodes,
                    nodeId,
                    AcceleratorDagNodeType.SUM,
                    current,
                    AcceleratorDagValueRef.none(),
                    encodeReductionMode(axis, false),
                    currentShape,
                    dataType
            );
        }
        return current;
    }

    private int[] removeAxisForReduction(int[] shape, int axis) {
        if (shape.length == 1) {
            return new int[]{1};
        }
        int[] out = new int[shape.length - 1];
        for (int i = 0, j = 0; i < shape.length; i++) {
            if (i != axis) {
                out[j++] = shape[i];
            }
        }
        return out;
    }

    private int denseLossSampleCount(int[] shape, int classAxis) {
        int count = 1;
        for (int i = 0; i < shape.length; i++) {
            if (i != classAxis) {
                count *= shape[i];
            }
        }
        return Math.max(1, count);
    }

    private AcceleratorDagSpec tryBuildNormalizationDagSpec(AcceleratorSubgraphSpec subgraph, PartitionPlanningContext context) {
        if (subgraph == null || context == null || subgraph.orderedNodeIds().size() != 1) {
            return null;
        }
        int nodeId = subgraph.orderedNodeIds().getFirst();
        CompiledNode node = context.compiledNode(nodeId);
        if (node == null || node.operation() == null) {
            return null;
        }
        Operation.OpType opType = node.operation().opType();
        boolean layerNormOp = opType == Operation.OpType.LAYER_NORM;
        boolean rmsNormOp = opType == Operation.OpType.RMS_NORM;
        if (!layerNormOp && !rmsNormOp) {
            return null;
        }
        int normalizedRank;
        float epsilon;
        if (layerNormOp && node.operation() instanceof layerNorm op) {
            normalizedRank = op.getNormalizedRank();
            epsilon = (float) op.getEpsilon();
        } else if (rmsNormOp && node.operation() instanceof rmsNorm op) {
            normalizedRank = op.getNormalizedRank();
            epsilon = (float) op.getEpsilon();
        } else {
            return null;
        }
        if ((layerNormOp && node.inputIds().size() != 3) || (rmsNormOp && node.inputIds().size() != 2)) {
            return null;
        }
        CompiledNode input = context.compiledNode(node.inputIds().get(0));
        CompiledNode gamma = context.compiledNode(node.inputIds().get(1));
        CompiledNode beta = layerNormOp ? context.compiledNode(node.inputIds().get(2)) : null;
        if (!isSupportedNormalizationValue(input)
                || !isSupportedNormalizationValue(gamma)
                || (layerNormOp && !isSupportedNormalizationValue(beta))) {
            return null;
        }
        int[] inputShape = input.shape();
        int[] outputShape = node.shape();
        int[] gammaShape = gamma.shape();
        int[] betaShape = beta == null ? null : beta.shape();
        if (!isValidNormalizationShape(inputShape, outputShape, gammaShape, betaShape, normalizedRank)) {
            return null;
        }

        List<AcceleratorDagInput> externalInputs = new ArrayList<>(layerNormOp ? 3 : 2);
        externalInputs.add(new AcceleratorDagInput(input.id(), shapeList(inputShape), input.dataType()));
        externalInputs.add(new AcceleratorDagInput(gamma.id(), shapeList(gammaShape), gamma.dataType()));
        if (layerNormOp) {
            externalInputs.add(new AcceleratorDagInput(beta.id(), shapeList(betaShape), beta.dataType()));
        }

        List<AcceleratorDagNode> nodes = new ArrayList<>();
        AcceleratorDagValueRef inputRef = AcceleratorDagValueRef.externalInput(0);
        AcceleratorDagValueRef gammaRef = AcceleratorDagValueRef.externalInput(1);
        AcceleratorDagValueRef betaRef = layerNormOp ? AcceleratorDagValueRef.externalInput(2) : AcceleratorDagValueRef.none();

        DataType normalizationDType = node.dataType();
        AcceleratorDagValueRef valueForVariance;
        AcceleratorDagValueRef scaledInput;
        if (layerNormOp) {
            AcceleratorDagValueRef mean = addTrailingMeanNodes(nodes, node.id(), inputRef, inputShape, normalizedRank, normalizationDType);
            AcceleratorDagValueRef centered = addNode(
                    nodes,
                    node.id(),
                    AcceleratorDagNodeType.SUB,
                    inputRef,
                    mean,
                    0,
                    inputShape,
                    normalizationDType
            );
            valueForVariance = centered;
            AcceleratorDagValueRef squared = addNode(
                    nodes,
                    node.id(),
                    AcceleratorDagNodeType.MUL,
                    centered,
                    centered,
                    0,
                    inputShape,
                    normalizationDType
            );
            AcceleratorDagValueRef variance = addTrailingMeanNodes(nodes, node.id(), squared, inputShape, normalizedRank, normalizationDType);
            AcceleratorDagValueRef invStd = addEpsilonSqrtInv(nodes, node.id(), variance, epsilon, reducedKeepDimsShape(inputShape, normalizedRank), normalizationDType);
            scaledInput = addNode(nodes, node.id(), AcceleratorDagNodeType.MUL, valueForVariance, invStd, 0, inputShape, normalizationDType);
        } else {
            AcceleratorDagValueRef squared = addNode(
                    nodes,
                    node.id(),
                    AcceleratorDagNodeType.MUL,
                    inputRef,
                    inputRef,
                    0,
                    inputShape,
                    normalizationDType
            );
            AcceleratorDagValueRef meanSquares = addTrailingMeanNodes(nodes, node.id(), squared, inputShape, normalizedRank, normalizationDType);
            AcceleratorDagValueRef invRms = addEpsilonSqrtInv(nodes, node.id(), meanSquares, epsilon, reducedKeepDimsShape(inputShape, normalizedRank), normalizationDType);
            scaledInput = addNode(nodes, node.id(), AcceleratorDagNodeType.MUL, inputRef, invRms, 0, inputShape, normalizationDType);
        }

        AcceleratorDagValueRef scaled = addNode(nodes, node.id(), AcceleratorDagNodeType.MUL, scaledInput, gammaRef, 0, inputShape, normalizationDType);
        if (layerNormOp) {
            addNode(nodes, node.id(), AcceleratorDagNodeType.ADD, scaled, betaRef, 0, outputShape, normalizationDType);
        }
        return new AcceleratorDagSpec(
                externalInputs,
                nodes,
                List.of(nodes.size() - 1),
                List.of(node.id())
        );
    }

    private boolean isSupportedNormalizationValue(CompiledNode node) {
        return node != null
                && (node.dataType() == DataType.FLOAT32 || node.dataType() == DataType.BFLOAT16)
                && node.shape().length >= 1
                && node.shape().length <= 4
                && node.contiguous()
                && !node.hasStorageOffset();
    }

    private boolean isValidNormalizationShape(
            int[] inputShape,
            int[] outputShape,
            int[] gammaShape,
            int[] betaShape,
            int normalizedRank
    ) {
        if (inputShape == null || outputShape == null || gammaShape == null) {
            return false;
        }
        if (inputShape.length < 1 || inputShape.length > 4 || normalizedRank < 1 || normalizedRank > inputShape.length) {
            return false;
        }
        if (!Arrays.equals(inputShape, outputShape) || gammaShape.length != normalizedRank) {
            return false;
        }
        if (betaShape != null && !Arrays.equals(gammaShape, betaShape)) {
            return false;
        }
        int tailStart = inputShape.length - normalizedRank;
        for (int i = 0; i < normalizedRank; i++) {
            if (gammaShape[i] != inputShape[tailStart + i]) {
                return false;
            }
        }
        return true;
    }

    private AcceleratorDagValueRef addTrailingMeanNodes(
            List<AcceleratorDagNode> nodes,
            int nodeId,
            AcceleratorDagValueRef inputRef,
            int[] inputShape,
            int normalizedRank,
            DataType dataType
    ) {
        AcceleratorDagValueRef current = inputRef;
        int[] currentShape = inputShape.clone();
        int firstAxis = inputShape.length - normalizedRank;
        for (int axis = inputShape.length - 1; axis >= firstAxis; axis--) {
            currentShape[axis] = 1;
            current = addNode(
                    nodes,
                    nodeId,
                    AcceleratorDagNodeType.MEAN,
                    current,
                    AcceleratorDagValueRef.none(),
                    encodeReductionMode(axis, true),
                    currentShape,
                    dataType
            );
        }
        return current;
    }

    private AcceleratorDagValueRef addEpsilonSqrtInv(
            List<AcceleratorDagNode> nodes,
            int nodeId,
            AcceleratorDagValueRef inputRef,
            float epsilon,
            int[] shape,
            DataType dataType
    ) {
        AcceleratorDagValueRef withEpsilon = addNode(
                nodes,
                nodeId,
                AcceleratorDagNodeType.ADD_SCALAR,
                inputRef,
                AcceleratorDagValueRef.none(),
                Float.floatToIntBits(epsilon),
                shape,
                dataType
        );
        AcceleratorDagValueRef sqrt = addNode(
                nodes,
                nodeId,
                AcceleratorDagNodeType.SQRT,
                withEpsilon,
                AcceleratorDagValueRef.none(),
                0,
                shape,
                dataType
        );
        return addNode(
                nodes,
                nodeId,
                AcceleratorDagNodeType.INV,
                sqrt,
                AcceleratorDagValueRef.none(),
                0,
                shape,
                dataType
        );
    }

    private AcceleratorDagValueRef addNode(
            List<AcceleratorDagNode> nodes,
            int nodeId,
            AcceleratorDagNodeType type,
            AcceleratorDagValueRef input0,
            AcceleratorDagValueRef input1,
            int scalarValueBits,
            int[] outputShape
    ) {
        return addNode(nodes, nodeId, type, input0, input1, scalarValueBits, outputShape, DataType.FLOAT32);
    }

    private AcceleratorDagValueRef addNode(
            List<AcceleratorDagNode> nodes,
            int nodeId,
            AcceleratorDagNodeType type,
            AcceleratorDagValueRef input0,
            AcceleratorDagValueRef input1,
            int scalarValueBits,
            int[] outputShape,
            DataType dataType
    ) {
        nodes.add(new AcceleratorDagNode(
                nodeId,
                type,
                input0,
                input1,
                AcceleratorDagValueRef.none(),
                AcceleratorDagValueRef.none(),
                scalarValueBits,
                outputShape.length,
                outputShape[0],
                outputShape.length >= 2 ? outputShape[1] : 1,
                outputShape.length >= 3 ? outputShape[2] : 1,
                outputShape.length >= 4 ? outputShape[3] : 1,
                dataType
        ));
        return AcceleratorDagValueRef.nodeOutput(nodes.size() - 1);
    }

    private int[] reducedKeepDimsShape(int[] inputShape, int normalizedRank) {
        int[] shape = inputShape.clone();
        int firstAxis = inputShape.length - normalizedRank;
        for (int axis = firstAxis; axis < inputShape.length; axis++) {
            shape[axis] = 1;
        }
        return shape;
    }

    private boolean isReduction(AcceleratorDagNodeType type) {
        return type == AcceleratorDagNodeType.SUM
                || type == AcceleratorDagNodeType.MEAN
                || type == AcceleratorDagNodeType.REDUCE_MIN
                || type == AcceleratorDagNodeType.REDUCE_MAX
                || type == AcceleratorDagNodeType.REDUCE_ALL
                || type == AcceleratorDagNodeType.REDUCE_ANY;
    }

    private boolean isIndexGather(AcceleratorDagNodeType type) {
        return type == AcceleratorDagNodeType.GATHER
                || type == AcceleratorDagNodeType.TAKE_ALONG_AXIS;
    }

    private boolean isIndexWriteOrGradient(AcceleratorDagNodeType type) {
        return type == AcceleratorDagNodeType.SCATTER_ADD
                || type == AcceleratorDagNodeType.GATHER_GRAD
                || type == AcceleratorDagNodeType.TAKE_ALONG_AXIS_GRAD;
    }

    private AcceleratorDagSpec tryBuildLogSoftmaxDagSpec(AcceleratorSubgraphSpec subgraph, PartitionPlanningContext context) {
        if (subgraph == null || context == null || subgraph.orderedNodeIds().size() != 1) {
            return null;
        }
        int nodeId = subgraph.orderedNodeIds().getFirst();
        CompiledNode logSoftmaxNode = context.compiledNode(nodeId);
        if (logSoftmaxNode == null
                || logSoftmaxNode.operation() == null
                || logSoftmaxNode.operation().opType() != Operation.OpType.LOG_SOFTMAX
                || !(logSoftmaxNode.operation() instanceof logSoftmax op)
                || logSoftmaxNode.inputIds().size() != 1) {
            return null;
        }
        CompiledNode inputNode = context.compiledNode(logSoftmaxNode.inputIds().getFirst());
        if (inputNode == null) {
            return null;
        }
        int[] inputShape = inputNode.shape();
        int[] outputShape = logSoftmaxNode.shape();
        if (inputShape.length < 1 || inputShape.length > 4 || outputShape.length < 1 || outputShape.length > 4) {
            return null;
        }
        AcceleratorDagInput input = new AcceleratorDagInput(
                inputNode.id(),
                java.util.Arrays.stream(inputShape).boxed().toList(),
                inputNode.dataType()
        );
        AcceleratorDagNode softmax = new AcceleratorDagNode(
                logSoftmaxNode.id(),
                AcceleratorDagNodeType.SOFTMAX,
                AcceleratorDagValueRef.externalInput(0),
                AcceleratorDagValueRef.none(),
                AcceleratorDagValueRef.none(),
                AcceleratorDagValueRef.none(),
                op.getDimension(),
                outputShape.length,
                outputShape[0],
                outputShape.length >= 2 ? outputShape[1] : 1,
                outputShape.length >= 3 ? outputShape[2] : 1,
                outputShape.length >= 4 ? outputShape[3] : 1,
                logSoftmaxNode.dataType()
        );
        AcceleratorDagNode log = new AcceleratorDagNode(
                logSoftmaxNode.id(),
                AcceleratorDagNodeType.LOG,
                AcceleratorDagValueRef.nodeOutput(0),
                AcceleratorDagValueRef.none(),
                AcceleratorDagValueRef.none(),
                AcceleratorDagValueRef.none(),
                0,
                outputShape.length,
                outputShape[0],
                outputShape.length >= 2 ? outputShape[1] : 1,
                outputShape.length >= 3 ? outputShape[2] : 1,
                outputShape.length >= 4 ? outputShape[3] : 1,
                logSoftmaxNode.dataType()
        );
        return new AcceleratorDagSpec(List.of(input), List.of(softmax, log), List.of(1), List.of(logSoftmaxNode.id()));
    }

    private AcceleratorDagSpec tryBuildSpecializedSdpaDagSpec(AcceleratorSubgraphSpec subgraph, PartitionPlanningContext context) {
        if (subgraph == null || context == null || subgraph.orderedNodeIds().isEmpty()) {
            return null;
        }
        if (!hasSingleRegionOutput(subgraph)) {
            return null;
        }
        List<Integer> nodeIds = subgraph.orderedNodeIds();
        int outputNodeId = subgraph.outputNodeIds().isEmpty() ? nodeIds.getLast() : subgraph.outputNodeIds().getFirst();
        CompiledNode outputNode = context.compiledNode(outputNodeId);
        if (outputNode == null || outputNode.operation() == null || outputNode.operation().opType() != Operation.OpType.MATMUL) {
            return null;
        }
        if (outputNode.inputIds().size() != 2) {
            return null;
        }
        CompiledNode softmaxNode = context.compiledNode(outputNode.inputIds().getFirst());
        CompiledNode valueNode = context.compiledNode(outputNode.inputIds().get(1));
        if (softmaxNode == null
                || softmaxNode.operation() == null
                || softmaxNode.operation().opType() != Operation.OpType.SOFTMAX
                || !(softmaxNode.operation() instanceof softmax softmaxOp)
                || valueNode == null) {
            return null;
        }
        int[] outputShape = outputNode.shape();
        if (outputShape.length < 3 || outputShape.length > 4 || softmaxOp.getDimension() != outputShape.length - 1) {
            return null;
        }

        CompiledNode maskedOrScaled = context.compiledNode(softmaxNode.inputIds().getFirst());
        CompiledNode scoreNode = maskedOrScaled;
        if (maskedOrScaled != null
                && maskedOrScaled.operation() != null
                && maskedOrScaled.operation().opType() == Operation.OpType.WHERE) {
            // MPSGraph SDPA expects a floating mask tensor on verified macOS runtimes, while
            // Synaptik attention masks are BOOL. Keep masked attention as a generic DAG.
            return null;
        }
        if (scoreNode == null || scoreNode.operation() == null || scoreNode.operation().opType() != Operation.OpType.MUL_SCALAR) {
            return null;
        }
        if (!(scoreNode.operation() instanceof mulScalar mulScalarOp) || scoreNode.inputIds().size() != 1) {
            return null;
        }
        CompiledNode qkNode = context.compiledNode(scoreNode.inputIds().getFirst());
        if (qkNode == null || qkNode.operation() == null || qkNode.operation().opType() != Operation.OpType.MATMUL || qkNode.inputIds().size() != 2) {
            return null;
        }
        CompiledNode queryNode = context.compiledNode(qkNode.inputIds().getFirst());
        CompiledNode permutedKeyNode = context.compiledNode(qkNode.inputIds().get(1));
        if (queryNode == null || permutedKeyNode == null || permutedKeyNode.operation() == null || permutedKeyNode.operation().opType() != Operation.OpType.PERMUTE) {
            return null;
        }
        int permuteBits = resolveScalarValueBits(permutedKeyNode, context);
        if (permuteBits == Integer.MIN_VALUE) {
            return null;
        }
        CompiledNode keyNode = context.compiledNode(permutedKeyNode.inputIds().getFirst());
        DataType dtype = outputNode.dataType();
        if (keyNode == null
                || !isMetalFloatingDType(dtype)
                || queryNode.dataType() != dtype
                || keyNode.dataType() != dtype
                || valueNode.dataType() != dtype
                || valueNode.shape().length < 3
                || valueNode.shape().length > 4) {
            return null;
        }
        if (!allSpecializedInputsAreExternal(subgraph, queryNode, keyNode, valueNode)) {
            return null;
        }

        List<AcceleratorDagInput> externalInputs = new ArrayList<>();
        externalInputs.add(new AcceleratorDagInput(queryNode.id(), java.util.Arrays.stream(queryNode.shape()).boxed().toList(), queryNode.dataType()));
        externalInputs.add(new AcceleratorDagInput(keyNode.id(), java.util.Arrays.stream(keyNode.shape()).boxed().toList(), keyNode.dataType()));
        externalInputs.add(new AcceleratorDagInput(valueNode.id(), java.util.Arrays.stream(valueNode.shape()).boxed().toList(), valueNode.dataType()));

        AcceleratorDagNode sdpaNode = new AcceleratorDagNode(
                outputNode.id(),
                AcceleratorDagNodeType.SDPA,
                AcceleratorDagValueRef.externalInput(0),
                AcceleratorDagValueRef.externalInput(1),
                AcceleratorDagValueRef.externalInput(2),
                AcceleratorDagValueRef.none(),
                Float.floatToIntBits(mulScalarOp.getScalarF32()),
                outputShape.length,
                outputShape[0],
                outputShape.length >= 2 ? outputShape[1] : 1,
                outputShape.length >= 3 ? outputShape[2] : 1,
                outputShape.length >= 4 ? outputShape[3] : 1,
                outputNode.dataType()
        );
        return new AcceleratorDagSpec(externalInputs, List.of(sdpaNode), List.of(0), List.of(outputNode.id()));
    }

    private AcceleratorDagSpec tryBuildCanonicalBfloat16SdpaPrimitiveDagSpec(
            AcceleratorSubgraphSpec subgraph,
            PartitionPlanningContext context
    ) {
        if (subgraph == null || context == null || subgraph.orderedNodeIds().size() != 1 || !hasSingleRegionOutput(subgraph)) {
            return null;
        }
        int outputNodeId = subgraph.outputNodeIds().isEmpty()
                ? subgraph.orderedNodeIds().getFirst()
                : subgraph.outputNodeIds().getFirst();
        CompiledNode outputNode = context.compiledNode(outputNodeId);
        if (outputNode == null
                || outputNode.dataType() != DataType.BFLOAT16
                || outputNode.operation() == null
                || outputNode.operation().opType() != Operation.OpType.MATMUL
                || outputNode.inputIds().size() != 2) {
            return null;
        }
        CompiledNode attentionInputNode = context.compiledNode(outputNode.inputIds().getFirst());
        CompiledNode valueNode = context.compiledNode(outputNode.inputIds().get(1));
        if (attentionInputNode == null
                || attentionInputNode.operation() == null
                || valueNode == null
                || valueNode.dataType() != DataType.BFLOAT16) {
            return null;
        }
        boolean decomposedSoftmax = attentionInputNode.operation().opType() == Operation.OpType.DIV;
        CompiledNode reduceMaxNode = null;
        CompiledNode subNode = null;
        CompiledNode expNode = null;
        CompiledNode sumNode = null;
        CompiledNode softmaxNode = null;
        CompiledNode scoreNode;
        if (decomposedSoftmax) {
            if (attentionInputNode.inputIds().size() != 2) {
                return null;
            }
            expNode = context.compiledNode(attentionInputNode.inputIds().getFirst());
            sumNode = context.compiledNode(attentionInputNode.inputIds().get(1));
            if (expNode == null
                    || expNode.operation() == null
                    || expNode.operation().opType() != Operation.OpType.EXP
                    || expNode.inputIds().size() != 1
                    || sumNode == null
                    || sumNode.operation() == null
                    || sumNode.operation().opType() != Operation.OpType.SUM
                    || sumNode.inputIds().size() != 1
                    || sumNode.inputIds().getFirst() != expNode.id()) {
                return null;
            }
            subNode = context.compiledNode(expNode.inputIds().getFirst());
            if (subNode == null
                    || subNode.operation() == null
                    || subNode.operation().opType() != Operation.OpType.SUB
                    || subNode.inputIds().size() != 2) {
                return null;
            }
            scoreNode = context.compiledNode(subNode.inputIds().getFirst());
            reduceMaxNode = context.compiledNode(subNode.inputIds().get(1));
            if (reduceMaxNode == null
                    || reduceMaxNode.operation() == null
                    || reduceMaxNode.operation().opType() != Operation.OpType.REDUCE_MAX
                    || reduceMaxNode.inputIds().size() != 1
                    || scoreNode == null
                    || reduceMaxNode.inputIds().getFirst() != scoreNode.id()) {
                return null;
            }
        } else if (attentionInputNode.operation().opType() == Operation.OpType.SOFTMAX) {
            softmaxNode = attentionInputNode;
            scoreNode = context.compiledNode(softmaxNode.inputIds().isEmpty() ? -1 : softmaxNode.inputIds().getFirst());
        } else {
            return null;
        }
        if (scoreNode == null
                || scoreNode.operation() == null
                || scoreNode.operation().opType() != Operation.OpType.MUL_SCALAR
                || scoreNode.inputIds().size() != 1) {
            return null;
        }
        CompiledNode qkNode = context.compiledNode(scoreNode.inputIds().getFirst());
        if (qkNode == null
                || qkNode.operation() == null
                || qkNode.operation().opType() != Operation.OpType.MATMUL
                || qkNode.inputIds().size() != 2) {
            return null;
        }
        CompiledNode queryNode = context.compiledNode(qkNode.inputIds().getFirst());
        CompiledNode permutedKeyNode = context.compiledNode(qkNode.inputIds().get(1));
        if (queryNode == null
                || queryNode.dataType() != DataType.BFLOAT16
                || permutedKeyNode == null
                || permutedKeyNode.operation() == null
                || permutedKeyNode.operation().opType() != Operation.OpType.PERMUTE
                || permutedKeyNode.inputIds().size() != 1) {
            return null;
        }
        CompiledNode keyNode = context.compiledNode(permutedKeyNode.inputIds().getFirst());
        if (keyNode == null || keyNode.dataType() != DataType.BFLOAT16) {
            return null;
        }
        int permuteBits = resolveScalarValueBits(permutedKeyNode, context);
        int scaleBits = resolveScalarValueBits(scoreNode, context);
        int attentionBits = decomposedSoftmax ? 0 : resolveScalarValueBits(softmaxNode, context);
        int reduceMaxBits = reduceMaxNode == null ? 0 : resolveScalarValueBits(reduceMaxNode, context);
        int sumBits = sumNode == null ? 0 : resolveScalarValueBits(sumNode, context);
        if (permuteBits == Integer.MIN_VALUE
                || scaleBits == Integer.MIN_VALUE
                || attentionBits == Integer.MIN_VALUE
                || reduceMaxBits == Integer.MIN_VALUE
                || sumBits == Integer.MIN_VALUE) {
            return null;
        }
        if (!validDagShape(queryNode)
                || !validDagShape(keyNode)
                || !validDagShape(valueNode)
                || !validDagShape(permutedKeyNode)
                || !validDagShape(qkNode)
                || !validDagShape(scoreNode)
                || (!decomposedSoftmax && !validDagShape(softmaxNode))
                || (decomposedSoftmax && (!validDagShape(reduceMaxNode)
                || !validDagShape(subNode)
                || !validDagShape(expNode)
                || !validDagShape(sumNode)
                || !validDagShape(attentionInputNode)))
                || !validDagShape(outputNode)) {
            return null;
        }
        List<AcceleratorDagInput> externalInputs = List.of(
                new AcceleratorDagInput(queryNode.id(), shapeList(queryNode.shape()), queryNode.dataType()),
                new AcceleratorDagInput(keyNode.id(), shapeList(keyNode.shape()), keyNode.dataType()),
                new AcceleratorDagInput(valueNode.id(), shapeList(valueNode.shape()), valueNode.dataType())
        );
        List<AcceleratorDagNode> nodes = new ArrayList<>(decomposedSoftmax ? 9 : 5);
        nodes.add(dagNode(
                permutedKeyNode,
                AcceleratorDagNodeType.PERMUTE,
                AcceleratorDagValueRef.externalInput(1),
                AcceleratorDagValueRef.none(),
                AcceleratorDagValueRef.none(),
                AcceleratorDagValueRef.none(),
                permuteBits
        ));
        nodes.add(dagNode(
                qkNode,
                AcceleratorDagNodeType.MATMUL,
                AcceleratorDagValueRef.externalInput(0),
                AcceleratorDagValueRef.nodeOutput(0),
                AcceleratorDagValueRef.none(),
                AcceleratorDagValueRef.none(),
                0
        ));
        nodes.add(dagNode(
                scoreNode,
                AcceleratorDagNodeType.MUL_SCALAR,
                AcceleratorDagValueRef.nodeOutput(1),
                AcceleratorDagValueRef.none(),
                AcceleratorDagValueRef.none(),
                AcceleratorDagValueRef.none(),
                scaleBits
        ));
        int attentionOutputIndex;
        if (decomposedSoftmax) {
            nodes.add(dagNode(
                    reduceMaxNode,
                    AcceleratorDagNodeType.REDUCE_MAX,
                    AcceleratorDagValueRef.nodeOutput(2),
                    AcceleratorDagValueRef.none(),
                    AcceleratorDagValueRef.none(),
                    AcceleratorDagValueRef.none(),
                    reduceMaxBits
            ));
            nodes.add(dagNode(
                    subNode,
                    AcceleratorDagNodeType.SUB,
                    AcceleratorDagValueRef.nodeOutput(2),
                    AcceleratorDagValueRef.nodeOutput(3),
                    AcceleratorDagValueRef.none(),
                    AcceleratorDagValueRef.none(),
                    0
            ));
            nodes.add(dagNode(
                    expNode,
                    AcceleratorDagNodeType.EXP,
                    AcceleratorDagValueRef.nodeOutput(4),
                    AcceleratorDagValueRef.none(),
                    AcceleratorDagValueRef.none(),
                    AcceleratorDagValueRef.none(),
                    0
            ));
            nodes.add(dagNode(
                    sumNode,
                    AcceleratorDagNodeType.SUM,
                    AcceleratorDagValueRef.nodeOutput(5),
                    AcceleratorDagValueRef.none(),
                    AcceleratorDagValueRef.none(),
                    AcceleratorDagValueRef.none(),
                    sumBits
            ));
            nodes.add(dagNode(
                    attentionInputNode,
                    AcceleratorDagNodeType.DIV,
                    AcceleratorDagValueRef.nodeOutput(5),
                    AcceleratorDagValueRef.nodeOutput(6),
                    AcceleratorDagValueRef.none(),
                    AcceleratorDagValueRef.none(),
                    0
            ));
            attentionOutputIndex = 7;
        } else {
            nodes.add(dagNode(
                    softmaxNode,
                    AcceleratorDagNodeType.SOFTMAX,
                    AcceleratorDagValueRef.nodeOutput(2),
                    AcceleratorDagValueRef.none(),
                    AcceleratorDagValueRef.none(),
                    AcceleratorDagValueRef.none(),
                    attentionBits
            ));
            attentionOutputIndex = 3;
        }
        nodes.add(dagNode(
                outputNode,
                AcceleratorDagNodeType.MATMUL,
                AcceleratorDagValueRef.nodeOutput(attentionOutputIndex),
                AcceleratorDagValueRef.externalInput(2),
                AcceleratorDagValueRef.none(),
                AcceleratorDagValueRef.none(),
                0
        ));
        return new AcceleratorDagSpec(externalInputs, nodes, List.of(nodes.size() - 1), List.of(outputNode.id()));
    }

    private AcceleratorDagSpec tryBuildSpecializedSdpaWeightsDagSpec(AcceleratorSubgraphSpec subgraph, PartitionPlanningContext context) {
        if (subgraph == null || context == null || subgraph.orderedNodeIds().isEmpty()) {
            return null;
        }
        if (!hasSingleRegionOutput(subgraph)) {
            return null;
        }
        int outputNodeId = subgraph.outputNodeIds().isEmpty()
                ? subgraph.orderedNodeIds().getLast()
                : subgraph.outputNodeIds().getFirst();
        CompiledNode weightsNode = context.compiledNode(outputNodeId);
        if (weightsNode == null
                || weightsNode.operation() == null
                || weightsNode.operation().opType() != Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS
                || weightsNode.inputIds().size() != 1) {
            return null;
        }
        CompiledNode attentionNode = context.compiledNode(weightsNode.inputIds().getFirst());
        if (attentionNode == null
                || attentionNode.operation() == null
                || attentionNode.operation().opType() != Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION
                || !(attentionNode.operation() instanceof scaledDotProductAttention attentionOp)
                || (attentionNode.inputIds().size() != 3 && attentionNode.inputIds().size() != 4)) {
            return null;
        }
        CompiledNode queryNode = context.compiledNode(attentionNode.inputIds().get(0));
        CompiledNode keyNode = context.compiledNode(attentionNode.inputIds().get(1));
        CompiledNode maskNode = attentionNode.inputIds().size() == 4
                ? context.compiledNode(attentionNode.inputIds().get(3))
                : null;
        if (queryNode == null || keyNode == null) {
            return null;
        }
        DataType dtype = weightsNode.dataType();
        if (!isMetalFloatingDType(dtype)
                || queryNode.dataType() != dtype
                || keyNode.dataType() != dtype
                || weightsNode.shape().length < 3
                || weightsNode.shape().length > 4) {
            return null;
        }
        if (maskNode != null && maskNode.dataType() != DataType.BOOL) {
            return null;
        }
        if (!allSpecializedInputsAreExternal(subgraph, queryNode, keyNode, maskNode)) {
            return null;
        }

        List<AcceleratorDagInput> externalInputs = new ArrayList<>();
        externalInputs.add(new AcceleratorDagInput(queryNode.id(), shapeList(queryNode.shape()), queryNode.dataType()));
        externalInputs.add(new AcceleratorDagInput(keyNode.id(), shapeList(keyNode.shape()), keyNode.dataType()));
        AcceleratorDagValueRef maskRef = AcceleratorDagValueRef.none();
        if (maskNode != null) {
            externalInputs.add(new AcceleratorDagInput(maskNode.id(), shapeList(maskNode.shape()), maskNode.dataType()));
            maskRef = AcceleratorDagValueRef.externalInput(2);
        }

        int[] outputShape = weightsNode.shape();
        AcceleratorDagNode weightsDagNode = new AcceleratorDagNode(
                weightsNode.id(),
                AcceleratorDagNodeType.SDPA_WEIGHTS,
                AcceleratorDagValueRef.externalInput(0),
                AcceleratorDagValueRef.externalInput(1),
                maskRef,
                AcceleratorDagValueRef.none(),
                Float.floatToIntBits((float) attentionOp.getScale()),
                outputShape.length,
                outputShape[0],
                outputShape.length >= 2 ? outputShape[1] : 1,
                outputShape.length >= 3 ? outputShape[2] : 1,
                outputShape.length >= 4 ? outputShape[3] : 1,
                weightsNode.dataType()
        );
        return new AcceleratorDagSpec(externalInputs, List.of(weightsDagNode), List.of(0), List.of(weightsNode.id()));
    }

    private AcceleratorDagSpec tryBuildSpecializedSdpaBackwardDagSpec(AcceleratorSubgraphSpec subgraph, PartitionPlanningContext context) {
        if (subgraph == null || context == null || subgraph.orderedNodeIds().size() != 1) {
            return null;
        }
        if (!hasSingleRegionOutput(subgraph)) {
            return null;
        }
        int outputNodeId = subgraph.outputNodeIds().isEmpty() ? subgraph.orderedNodeIds().getFirst() : subgraph.outputNodeIds().getFirst();
        CompiledNode outputNode = context.compiledNode(outputNodeId);
        if (outputNode == null
                || outputNode.operation() == null
                || outputNode.operation().opType() != Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_BACKWARD
                || !(outputNode.operation() instanceof scaledDotProductAttentionBackward backwardOp)
                || outputNode.inputIds().size() != 2) {
            return null;
        }
        CompiledNode attentionOutNode = context.compiledNode(outputNode.inputIds().getFirst());
        CompiledNode outGradNode = context.compiledNode(outputNode.inputIds().get(1));
        if (attentionOutNode == null
                || attentionOutNode.operation() == null
                || attentionOutNode.operation().opType() != Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION
                || !(attentionOutNode.operation() instanceof scaledDotProductAttention attentionOp)
                || outGradNode == null) {
            return null;
        }
        if (attentionOutNode.inputIds().size() != 3 && attentionOutNode.inputIds().size() != 4) {
            return null;
        }
        CompiledNode queryNode = context.compiledNode(attentionOutNode.inputIds().get(0));
        CompiledNode keyNode = context.compiledNode(attentionOutNode.inputIds().get(1));
        CompiledNode valueNode = context.compiledNode(attentionOutNode.inputIds().get(2));
        CompiledNode maskNode = attentionOutNode.inputIds().size() == 4
                ? context.compiledNode(attentionOutNode.inputIds().get(3))
                : null;
        if (queryNode == null || keyNode == null || valueNode == null) {
            return null;
        }
        if (maskNode != null && maskNode.dataType() != DataType.BOOL) {
            return null;
        }
        DataType dtype = outputNode.dataType();
        if (!isMetalFloatingDType(dtype)
                || queryNode.dataType() != dtype
                || keyNode.dataType() != dtype
                || valueNode.dataType() != dtype
                || outGradNode.dataType() != dtype) {
            return null;
        }
        if (backwardOp.getOutputKind() == scaledDotProductAttentionBackward.OutputKind.VALUE) {
            if (!allSpecializedInputsAreExternal(subgraph, queryNode, keyNode, outGradNode, maskNode)) {
                return null;
            }
            List<AcceleratorDagInput> externalInputs = new ArrayList<>(maskNode == null ? 3 : 4);
            externalInputs.add(new AcceleratorDagInput(queryNode.id(), Arrays.stream(queryNode.shape()).boxed().toList(), queryNode.dataType()));
            externalInputs.add(new AcceleratorDagInput(keyNode.id(), Arrays.stream(keyNode.shape()).boxed().toList(), keyNode.dataType()));
            externalInputs.add(new AcceleratorDagInput(outGradNode.id(), Arrays.stream(outGradNode.shape()).boxed().toList(), outGradNode.dataType()));
            AcceleratorDagValueRef maskRef = AcceleratorDagValueRef.none();
            if (maskNode != null) {
                externalInputs.add(new AcceleratorDagInput(maskNode.id(), Arrays.stream(maskNode.shape()).boxed().toList(), maskNode.dataType()));
                maskRef = AcceleratorDagValueRef.externalInput(3);
            }
            int[] weightsShape = expectedScoresShape(queryNode.shape(), keyNode.shape());
            if (weightsShape.length < 1 || weightsShape.length > 4) {
                return null;
            }
            int[] weightsTransposedShape = transposeLastTwoShape(weightsShape);
            int[] outputShape = outputNode.shape();
            int permuteMode = encodePermuteAxes(lastTwoAxesSwap(weightsShape.length));
            if (permuteMode == Integer.MIN_VALUE) {
                return null;
            }
            List<AcceleratorDagNode> nodes = new ArrayList<>(3);
            nodes.add(new AcceleratorDagNode(
                    outputNode.id(),
                    AcceleratorDagNodeType.SDPA_WEIGHTS,
                    AcceleratorDagValueRef.externalInput(0),
                    AcceleratorDagValueRef.externalInput(1),
                    maskRef,
                    AcceleratorDagValueRef.none(),
                    Float.floatToIntBits((float) attentionOp.getScale()),
                    weightsShape.length,
                    weightsShape[0],
                    weightsShape.length >= 2 ? weightsShape[1] : 1,
                    weightsShape.length >= 3 ? weightsShape[2] : 1,
                    weightsShape.length >= 4 ? weightsShape[3] : 1,
                    outputNode.dataType()
            ));
            nodes.add(new AcceleratorDagNode(
                    outputNode.id(),
                    AcceleratorDagNodeType.PERMUTE,
                    AcceleratorDagValueRef.nodeOutput(0),
                    AcceleratorDagValueRef.none(),
                    AcceleratorDagValueRef.none(),
                    AcceleratorDagValueRef.none(),
                    permuteMode,
                    weightsTransposedShape.length,
                    weightsTransposedShape[0],
                    weightsTransposedShape.length >= 2 ? weightsTransposedShape[1] : 1,
                    weightsTransposedShape.length >= 3 ? weightsTransposedShape[2] : 1,
                    weightsTransposedShape.length >= 4 ? weightsTransposedShape[3] : 1,
                    outputNode.dataType()
            ));
            nodes.add(new AcceleratorDagNode(
                    outputNode.id(),
                    AcceleratorDagNodeType.MATMUL,
                    AcceleratorDagValueRef.nodeOutput(1),
                    AcceleratorDagValueRef.externalInput(2),
                    AcceleratorDagValueRef.none(),
                    AcceleratorDagValueRef.none(),
                    0,
                    outputShape.length,
                    outputShape[0],
                    outputShape.length >= 2 ? outputShape[1] : 1,
                    outputShape.length >= 3 ? outputShape[2] : 1,
                    outputShape.length >= 4 ? outputShape[3] : 1,
                    outputNode.dataType()
            ));
            return new AcceleratorDagSpec(externalInputs, nodes, List.of(2), List.of(outputNode.id()));
        }
        if (!allSpecializedInputsAreExternal(subgraph, queryNode, keyNode, valueNode, outGradNode, maskNode)) {
            return null;
        }
        List<AcceleratorDagInput> externalInputs = new ArrayList<>(maskNode == null ? 4 : 5);
        externalInputs.add(new AcceleratorDagInput(queryNode.id(), Arrays.stream(queryNode.shape()).boxed().toList(), queryNode.dataType()));
        externalInputs.add(new AcceleratorDagInput(keyNode.id(), Arrays.stream(keyNode.shape()).boxed().toList(), keyNode.dataType()));
        externalInputs.add(new AcceleratorDagInput(valueNode.id(), Arrays.stream(valueNode.shape()).boxed().toList(), valueNode.dataType()));
        externalInputs.add(new AcceleratorDagInput(outGradNode.id(), Arrays.stream(outGradNode.shape()).boxed().toList(), outGradNode.dataType()));
        if (maskNode != null) {
            externalInputs.add(new AcceleratorDagInput(maskNode.id(), Arrays.stream(maskNode.shape()).boxed().toList(), maskNode.dataType()));
        }
        AcceleratorDagNodeType nodeType = switch (backwardOp.getOutputKind()) {
            case QUERY -> AcceleratorDagNodeType.SDPA_BACKWARD_QUERY;
            case KEY -> AcceleratorDagNodeType.SDPA_BACKWARD_KEY;
            case VALUE -> AcceleratorDagNodeType.SDPA_BACKWARD_VALUE;
        };
        int[] outputShape = outputNode.shape();
        AcceleratorDagValueRef maskRef = maskNode == null ? AcceleratorDagValueRef.none() : AcceleratorDagValueRef.externalInput(4);
        AcceleratorDagNode backwardNode = new AcceleratorDagNode(
                outputNode.id(),
                nodeType,
                AcceleratorDagValueRef.externalInput(0),
                AcceleratorDagValueRef.externalInput(1),
                AcceleratorDagValueRef.externalInput(2),
                AcceleratorDagValueRef.externalInput(3),
                maskRef,
                Float.floatToIntBits((float) attentionOp.getScale()),
                outputShape.length,
                outputShape[0],
                outputShape.length >= 2 ? outputShape[1] : 1,
                outputShape.length >= 3 ? outputShape[2] : 1,
                outputShape.length >= 4 ? outputShape[3] : 1,
                outputNode.dataType()
        );
        return new AcceleratorDagSpec(externalInputs, List.of(backwardNode), List.of(0), List.of(outputNode.id()));
    }

    private boolean hasSingleRegionOutput(AcceleratorSubgraphSpec subgraph) {
        return subgraph.outputNodeIds().isEmpty() || subgraph.outputNodeIds().size() == 1;
    }

    private boolean validDagShape(CompiledNode node) {
        if (node == null || node.shape() == null) {
            return false;
        }
        int rank = node.shape().length;
        return rank >= 1 && rank <= 4;
    }

    private AcceleratorDagNode dagNode(
            CompiledNode node,
            AcceleratorDagNodeType type,
            AcceleratorDagValueRef input0,
            AcceleratorDagValueRef input1,
            AcceleratorDagValueRef input2,
            AcceleratorDagValueRef input3,
            int scalarValueBits
    ) {
        int[] shape = node.shape();
        return new AcceleratorDagNode(
                node.id(),
                type,
                input0,
                input1,
                input2,
                input3,
                scalarValueBits,
                shape.length,
                shape[0],
                shape.length >= 2 ? shape[1] : 1,
                shape.length >= 3 ? shape[2] : 1,
                shape.length >= 4 ? shape[3] : 1,
                node.dataType()
        );
    }

    private int[] expectedScoresShape(int[] queryShape, int[] keyShape) {
        if (queryShape == null || keyShape == null || queryShape.length < 2 || keyShape.length < 2) {
            return new int[0];
        }
        int[] qBatch = Arrays.copyOf(queryShape, queryShape.length - 2);
        int[] kBatch = Arrays.copyOf(keyShape, keyShape.length - 2);
        int[] batch = broadcastBatchShape(qBatch, kBatch);
        if (batch == null) {
            return new int[0];
        }
        int[] out = Arrays.copyOf(batch, batch.length + 2);
        out[out.length - 2] = queryShape[queryShape.length - 2];
        out[out.length - 1] = keyShape[keyShape.length - 2];
        return out;
    }

    private int[] broadcastBatchShape(int[] left, int[] right) {
        int rank = Math.max(left.length, right.length);
        int[] out = new int[rank];
        for (int i = 0; i < rank; i++) {
            int leftIndex = left.length - 1 - i;
            int rightIndex = right.length - 1 - i;
            int l = leftIndex >= 0 ? left[leftIndex] : 1;
            int r = rightIndex >= 0 ? right[rightIndex] : 1;
            if (l != r && l != 1 && r != 1) {
                return null;
            }
            out[rank - 1 - i] = Math.max(l, r);
        }
        return out;
    }

    private int[] transposeLastTwoShape(int[] shape) {
        int[] out = shape.clone();
        if (out.length >= 2) {
            int last = out.length - 1;
            int tmp = out[last];
            out[last] = out[last - 1];
            out[last - 1] = tmp;
        }
        return out;
    }

    private int[] lastTwoAxesSwap(int rank) {
        int[] axes = new int[rank];
        for (int i = 0; i < rank; i++) {
            axes[i] = i;
        }
        if (rank >= 2) {
            axes[rank - 2] = rank - 1;
            axes[rank - 1] = rank - 2;
        }
        return axes;
    }

    private int encodePermuteAxes(int[] axes) {
        if (axes == null || axes.length < 1 || axes.length > 4) {
            return Integer.MIN_VALUE;
        }
        int encoded = axes.length & 0xFF;
        boolean[] seen = new boolean[axes.length];
        for (int i = 0; i < axes.length; i++) {
            int axis = axes[i];
            if (axis < 0 || axis >= axes.length || seen[axis]) {
                return Integer.MIN_VALUE;
            }
            seen[axis] = true;
            encoded |= (axis & 0xF) << (8 + i * 4);
        }
        return encoded;
    }

    private boolean allSpecializedInputsAreExternal(AcceleratorSubgraphSpec subgraph, CompiledNode... inputs) {
        if (subgraph == null || inputs == null) {
            return false;
        }
        java.util.Set<Integer> selected = java.util.Set.copyOf(subgraph.orderedNodeIds());
        for (CompiledNode input : inputs) {
            if (input != null && selected.contains(input.id())) {
                return false;
            }
        }
        return true;
    }

    private AcceleratorDagValueRef resolveDagValueRef(
            List<Integer> inputIds,
            int inputIndex,
            Map<Integer, Integer> externalInputIndex,
            Map<Integer, Integer> loweredNodeIndex
    ) {
        if (inputIds == null || inputIndex >= inputIds.size()) {
            return AcceleratorDagValueRef.none();
        }
        int inputNodeId = inputIds.get(inputIndex);
        Integer external = externalInputIndex.get(inputNodeId);
        if (external != null) {
            return AcceleratorDagValueRef.externalInput(external);
        }
        Integer nodeOutput = loweredNodeIndex.get(inputNodeId);
        return nodeOutput == null ? AcceleratorDagValueRef.none() : AcceleratorDagValueRef.nodeOutput(nodeOutput);
    }

    private AcceleratorDagNodeType resolveDagNodeType(Operation.OpType opType) {
        if (opType == null) {
            return null;
        }
        return switch (opType) {
            case MATMUL -> AcceleratorDagNodeType.MATMUL;
            case LINEAR -> AcceleratorDagNodeType.LINEAR;
            case ADD -> AcceleratorDagNodeType.ADD;
            case SUB -> AcceleratorDagNodeType.SUB;
            case MUL -> AcceleratorDagNodeType.MUL;
            case DIV -> AcceleratorDagNodeType.DIV;
            case MIN -> AcceleratorDagNodeType.MIN;
            case MAX -> AcceleratorDagNodeType.MAX;
            case RELU -> AcceleratorDagNodeType.RELU;
            case TANH, FAST_TANH -> AcceleratorDagNodeType.TANH;
            case SIGMOID -> AcceleratorDagNodeType.SIGMOID;
            case ABS -> AcceleratorDagNodeType.ABS;
            case EXP, FAST_EXP -> AcceleratorDagNodeType.EXP;
            case LOG -> AcceleratorDagNodeType.LOG;
            case NEG -> AcceleratorDagNodeType.NEG;
            case SQRT -> AcceleratorDagNodeType.SQRT;
            case INV -> AcceleratorDagNodeType.INV;
            case CLAMP_MIN -> AcceleratorDagNodeType.CLAMP_MIN;
            case CLAMP_MAX -> AcceleratorDagNodeType.CLAMP_MAX;
            case RESHAPE -> AcceleratorDagNodeType.RESHAPE;
            case CONTIGUOUS, NOOP -> AcceleratorDagNodeType.CONTIGUOUS;
            case PERMUTE -> AcceleratorDagNodeType.PERMUTE;
            case EXPAND -> AcceleratorDagNodeType.EXPAND;
            case SELECT -> AcceleratorDagNodeType.SELECT;
            case EXPAND_DIMS -> AcceleratorDagNodeType.EXPAND_DIMS;
            case SQUEEZE -> AcceleratorDagNodeType.SQUEEZE;
            case MUL_SCALAR -> AcceleratorDagNodeType.MUL_SCALAR;
            case POW -> AcceleratorDagNodeType.POW_SCALAR;
            case WHERE -> AcceleratorDagNodeType.WHERE;
            case SOFTMAX -> AcceleratorDagNodeType.SOFTMAX;
            case SUM -> AcceleratorDagNodeType.SUM;
            case MEAN -> AcceleratorDagNodeType.MEAN;
            case REDUCE_MIN -> AcceleratorDagNodeType.REDUCE_MIN;
            case REDUCE_MAX -> AcceleratorDagNodeType.REDUCE_MAX;
            case GT -> AcceleratorDagNodeType.GT;
            case GE -> AcceleratorDagNodeType.GE;
            case LT -> AcceleratorDagNodeType.LT;
            case LE -> AcceleratorDagNodeType.LE;
            case EQ -> AcceleratorDagNodeType.EQ;
            case NE -> AcceleratorDagNodeType.NE;
            case LOGICAL_AND -> AcceleratorDagNodeType.LOGICAL_AND;
            case LOGICAL_OR -> AcceleratorDagNodeType.LOGICAL_OR;
            case LOGICAL_NOT -> AcceleratorDagNodeType.LOGICAL_NOT;
            case REDUCE_ALL -> AcceleratorDagNodeType.REDUCE_ALL;
            case REDUCE_ANY -> AcceleratorDagNodeType.REDUCE_ANY;
            case GATHER -> AcceleratorDagNodeType.GATHER;
            case TAKE_ALONG_AXIS -> AcceleratorDagNodeType.TAKE_ALONG_AXIS;
            case SCATTER_ADD -> AcceleratorDagNodeType.SCATTER_ADD;
            case GATHER_GRAD -> AcceleratorDagNodeType.GATHER_GRAD;
            case TAKE_ALONG_AXIS_GRAD -> AcceleratorDagNodeType.TAKE_ALONG_AXIS_GRAD;
            case CONV2D, CONV2D_GEMM -> AcceleratorDagNodeType.CONV2D;
            case CONV2D_BACKWARD_INPUT, CONV2D_BACKWARD_INPUT_GEMM -> AcceleratorDagNodeType.CONV2D_BACKWARD_INPUT;
            case CONV2D_BACKWARD_WEIGHT, CONV2D_BACKWARD_WEIGHT_GEMM -> AcceleratorDagNodeType.CONV2D_BACKWARD_WEIGHT;
            case MAX_POOL2D -> AcceleratorDagNodeType.MAX_POOL2D;
            case AVG_POOL2D -> AcceleratorDagNodeType.AVG_POOL2D;
            case AVG_POOL2D_BACKWARD_INPUT -> AcceleratorDagNodeType.AVG_POOL2D_BACKWARD_INPUT;
            case MAX_POOL2D_BACKWARD_INPUT -> AcceleratorDagNodeType.MAX_POOL2D_BACKWARD_INPUT;
            case CROSS_ENTROPY_LOSS_INDICES -> AcceleratorDagNodeType.CROSS_ENTROPY_LOSS_INDICES;
            case CROSS_ENTROPY_LOSS_INDICES_GRAD -> AcceleratorDagNodeType.CROSS_ENTROPY_LOSS_INDICES_GRAD;
            case SOFTMAX_GRAD -> AcceleratorDagNodeType.SOFTMAX_GRAD;
            case LOG_SOFTMAX_GRAD -> AcceleratorDagNodeType.LOG_SOFTMAX_GRAD;
            case REDUCE_MIN_GRAD -> AcceleratorDagNodeType.REDUCE_MIN_GRAD;
            case REDUCE_MAX_GRAD -> AcceleratorDagNodeType.REDUCE_MAX_GRAD;
            case MIN_GRAD -> AcceleratorDagNodeType.MIN_GRAD;
            case MAX_GRAD -> AcceleratorDagNodeType.MAX_GRAD;
            case SCALED_DOT_PRODUCT_ATTENTION -> AcceleratorDagNodeType.SDPA;
            default -> null;
        };
    }

    private int resolveScalarValueBits(CompiledNode node) {
        return resolveScalarValueBits(node, null);
    }

    private int resolveScalarValueBits(CompiledNode node, PartitionPlanningContext context) {
        if (node == null || node.operation() == null) {
            return 0;
        }
        return switch (node.operation().opType()) {
            case CLAMP_MIN -> node.operation() instanceof clampMin clamp ? Float.floatToIntBits(clamp.getMinValueF32()) : 0;
            case CLAMP_MAX -> node.operation() instanceof clampMax clamp ? Float.floatToIntBits(clamp.getMaxValueF32()) : 0;
            case MUL_SCALAR -> node.operation() instanceof mulScalar op ? Float.floatToIntBits(op.getScalarF32()) : Integer.MIN_VALUE;
            case POW -> node.operation() instanceof pow op ? Float.floatToIntBits(op.getExponentF32()) : Integer.MIN_VALUE;
            case SOFTMAX -> node.operation() instanceof softmax op ? op.getDimension() : Integer.MIN_VALUE;
            case SUM -> node.operation() instanceof sum op ? encodeReductionMode(op.getDimension(), op.keepDims()) : Integer.MIN_VALUE;
            case MEAN -> node.operation() instanceof mean op ? encodeReductionMode(op.getDimension(), op.keepDims()) : Integer.MIN_VALUE;
            case REDUCE_MIN -> node.operation() instanceof reduceMin op ? encodeReductionMode(op.getDimension(), op.keepDims()) : Integer.MIN_VALUE;
            case REDUCE_MAX -> node.operation() instanceof reduceMax op ? encodeReductionMode(op.getDimension(), op.keepDims()) : Integer.MIN_VALUE;
            case REDUCE_ALL -> node.operation() instanceof reduceAll op ? encodeReductionMode(op.getDimension(), op.keepDims()) : Integer.MIN_VALUE;
            case REDUCE_ANY -> node.operation() instanceof reduceAny op ? encodeReductionMode(op.getDimension(), op.keepDims()) : Integer.MIN_VALUE;
            case GATHER -> node.operation() instanceof gather op ? op.getDimension() : Integer.MIN_VALUE;
            case TAKE_ALONG_AXIS -> node.operation() instanceof takeAlongAxis op ? op.getDimension() : Integer.MIN_VALUE;
            case SCATTER_ADD -> node.operation() instanceof operations.index.scatterAdd op ? op.getDimension() : Integer.MIN_VALUE;
            case GATHER_GRAD -> node.operation() instanceof operations.index.gatherGrad op ? op.getDimension() : Integer.MIN_VALUE;
            case TAKE_ALONG_AXIS_GRAD -> node.operation() instanceof operations.index.takeAlongAxisGrad op ? op.getDimension() : Integer.MIN_VALUE;
            case CONV2D -> node.operation() instanceof conv2d op ? encodeConv2dMode(op) : Integer.MIN_VALUE;
            case CONV2D_GEMM -> node.operation() instanceof conv2dGemm op ? encodeConv2dMode(op) : Integer.MIN_VALUE;
            case CONV2D_BACKWARD_INPUT -> node.operation() instanceof conv2dBackwardInput op ? encodeConv2dMode(op.getOptions()) : Integer.MIN_VALUE;
            case CONV2D_BACKWARD_INPUT_GEMM -> node.operation() instanceof conv2dBackwardInputGemm op ? encodeConv2dMode(op.getOptions()) : Integer.MIN_VALUE;
            case CONV2D_BACKWARD_WEIGHT -> node.operation() instanceof conv2dBackwardWeight op ? encodeConv2dMode(op.getOptions()) : Integer.MIN_VALUE;
            case CONV2D_BACKWARD_WEIGHT_GEMM -> node.operation() instanceof conv2dBackwardWeightGemm op ? encodeConv2dMode(op.getOptions()) : Integer.MIN_VALUE;
            case MAX_POOL2D -> node.operation() instanceof maxPool2d op ? encodePool2dMode(op.getOptions()) : Integer.MIN_VALUE;
            case AVG_POOL2D -> node.operation() instanceof avgPool2d op ? encodePool2dMode(op.getOptions()) : Integer.MIN_VALUE;
            case AVG_POOL2D_BACKWARD_INPUT -> node.operation() instanceof avgPool2dBackwardInput op ? encodePool2dMode(op.getOptions()) : Integer.MIN_VALUE;
            case MAX_POOL2D_BACKWARD_INPUT -> node.operation() instanceof maxPool2dBackwardInput op ? encodePool2dMode(op.getOptions()) : Integer.MIN_VALUE;
            case CROSS_ENTROPY_LOSS_INDICES -> node.operation() instanceof crossEntropyLossIndices op ? encodeCrossEntropyLossIndicesMode(op) : Integer.MIN_VALUE;
            case CROSS_ENTROPY_LOSS_INDICES_GRAD -> node.operation() instanceof crossEntropyLossIndicesGrad op ? encodeAxisMode(op.getClassDimension()) : Integer.MIN_VALUE;
            case SOFTMAX_GRAD -> node.operation() instanceof softmaxGrad op ? op.getDimension() : Integer.MIN_VALUE;
            case LOG_SOFTMAX_GRAD -> node.operation() instanceof logSoftmaxGrad op ? op.getDimension() : Integer.MIN_VALUE;
            case REDUCE_MIN_GRAD -> node.operation() instanceof reduceMinGrad op ? op.getDimension() : Integer.MIN_VALUE;
            case REDUCE_MAX_GRAD -> node.operation() instanceof reduceMaxGrad op ? op.getDimension() : Integer.MIN_VALUE;
            case MIN_GRAD -> node.operation() instanceof minGrad op ? (op.isForFirstInput() ? 1 : 0) : Integer.MIN_VALUE;
            case MAX_GRAD -> node.operation() instanceof maxGrad op ? (op.isForFirstInput() ? 1 : 0) : Integer.MIN_VALUE;
            case SCALED_DOT_PRODUCT_ATTENTION -> node.operation() instanceof scaledDotProductAttention op ? Float.floatToIntBits((float) op.getScale()) : Integer.MIN_VALUE;
            case PERMUTE -> encodePermuteMode(node);
            case SELECT -> encodeSelectMode(node, context);
            case EXPAND_DIMS -> node.operation() instanceof expandDims op ? op.getAxis() : Integer.MIN_VALUE;
            case SQUEEZE -> node.operation() instanceof squeeze op ? op.getAxis() : Integer.MIN_VALUE;
            default -> 0;
        };
    }

    private int encodeSelectMode(CompiledNode node, PartitionPlanningContext context) {
        if (!(node.operation() instanceof select selectOp)) {
            return Integer.MIN_VALUE;
        }
        CompiledNode input = context == null || node.inputIds().isEmpty()
                ? null
                : context.compiledNode(node.inputIds().getFirst());
        int[] inputShape = input == null ? null : input.shape();
        int axis = selectOp.getDimension();
        int index = selectOp.getIndex();
        if (axis < 0 || axis > 0xFFFF || index < 0 || index > 0xFFFF) {
            return Integer.MIN_VALUE;
        }
        if (inputShape == null || inputShape.length < 1 || inputShape.length > 4) {
            return Integer.MIN_VALUE;
        }
        if (axis >= inputShape.length || index >= inputShape[axis]) {
            return Integer.MIN_VALUE;
        }
        int[] outputShape = node.shape();
        int expectedOutputRank = inputShape.length == 1 ? 1 : inputShape.length - 1;
        if (outputShape.length != expectedOutputRank) {
            return Integer.MIN_VALUE;
        }
        return (axis & 0xFFFF) | ((index & 0xFFFF) << 16);
    }

    private int encodeReductionMode(int axis, boolean keepDims) {
        if (axis < Short.MIN_VALUE || axis > Short.MAX_VALUE) {
            return Integer.MIN_VALUE;
        }
        return (axis & 0xFFFF) | (keepDims ? 1 << 16 : 0);
    }

    private int encodeConv2dMode(conv2d op) {
        return encodeConv2dMode(op.getOptions());
    }

    private int encodeConv2dMode(tensor.options.Conv2dOptions options) {
        int strideH = options.strideH();
        int strideW = options.strideW();
        int padH = options.padH();
        int padW = options.padW();
        if (strideH < 1 || strideH > 255 || strideW < 1 || strideW > 255 || padH < 0 || padH > 255 || padW < 0 || padW > 255) {
            return Integer.MIN_VALUE;
        }
        return (strideH & 0xFF)
                | ((strideW & 0xFF) << 8)
                | ((padH & 0xFF) << 16)
                | ((padW & 0xFF) << 24);
    }

    private int encodeConv2dMode(conv2dGemm op) {
        return encodeConv2dMode(op.getOptions());
    }

    private int encodePool2dMode(tensor.options.Pool2dOptions options) {
        int kernelH = options.kernelH();
        int kernelW = options.kernelW();
        int strideH = options.strideH();
        int strideW = options.strideW();
        int padH = options.padH();
        int padW = options.padW();
        if (kernelH < 1 || kernelH > 15
                || kernelW < 1 || kernelW > 15
                || strideH < 1 || strideH > 15
                || strideW < 1 || strideW > 15
                || padH < 0 || padH > 15
                || padW < 0 || padW > 15) {
            return Integer.MIN_VALUE;
        }
        return (kernelH & 0xF)
                | ((kernelW & 0xF) << 4)
                | ((strideH & 0xF) << 8)
                | ((strideW & 0xF) << 12)
                | ((padH & 0xF) << 16)
                | ((padW & 0xF) << 20)
                | (options.countIncludePad() ? 1 << 24 : 0);
    }

    private int encodeCrossEntropyLossIndicesMode(crossEntropyLossIndices op) {
        int axis = op.getClassDimension();
        if (axis < 0 || axis > 0xFF) {
            return Integer.MIN_VALUE;
        }
        int reductionCode = switch (op.getReduction()) {
            case NONE -> 0;
            case SUM -> 1;
            case MEAN -> 2;
        };
        int encoded = axis | (reductionCode << 8);
        if (op.hasIgnoreIndex()) {
            int ignoreIndex = op.getIgnoreIndex();
            if (ignoreIndex < Short.MIN_VALUE || ignoreIndex > Short.MAX_VALUE) {
                return Integer.MIN_VALUE;
            }
            encoded |= 1 << 10;
            encoded |= (ignoreIndex & 0xFFFF) << 16;
        }
        return encoded;
    }

    private int encodeAxisMode(int axis) {
        return axis < 0 || axis > 0xFF ? Integer.MIN_VALUE : axis;
    }

    private int encodePermuteMode(CompiledNode node) {
        if (!(node.operation() instanceof permute permuteOp)) {
            return 0;
        }
        int[] axes = permuteOp.getAxes();
        int[] shape = node.shape();
        if (shape.length < 1 || shape.length > 4 || axes.length != shape.length) {
            return Integer.MIN_VALUE;
        }
        int encoded = shape.length & 0xFF;
        boolean[] seen = new boolean[shape.length];
        for (int i = 0; i < axes.length; i++) {
            int axis = axes[i];
            if (axis < 0 || axis >= shape.length || seen[axis]) {
                return Integer.MIN_VALUE;
            }
            seen[axis] = true;
            encoded |= (axis & 0xF) << (8 + i * 4);
        }
        return encoded;
    }

    private AcceleratorPostOp resolvePostOp(
            CompiledNode node,
            int currentOutputNodeId,
            PartitionPlanningContext context,
            CompiledNode computeNode
    ) {
        if (node.inputIds().size() == 1 && node.inputIds().getFirst() == currentOutputNodeId) {
            AcceleratorPostOpType unaryType = resolveUnaryPostOpType(node.operation().opType());
            if (unaryType == null) {
                return null;
            }
            if (unaryType == AcceleratorPostOpType.CLAMP_MIN) {
                if (!(node.operation() instanceof clampMin clamp)) {
                    return null;
                }
                return AcceleratorPostOp.scalarUnary(unaryType, clamp.getMinValueF32());
            }
            if (unaryType == AcceleratorPostOpType.CLAMP_MAX) {
                if (!(node.operation() instanceof clampMax clamp)) {
                    return null;
                }
                return AcceleratorPostOp.scalarUnary(unaryType, clamp.getMaxValueF32());
            }
            if (unaryType == AcceleratorPostOpType.MUL_SCALAR) {
                if (!(node.operation() instanceof mulScalar mul)) {
                    return null;
                }
                return AcceleratorPostOp.scalarUnary(unaryType, mul.getScalarF32());
            }
            return AcceleratorPostOp.unary(unaryType);
        }
        if (node.inputIds().size() != 2 || !node.inputIds().contains(currentOutputNodeId)) {
            return null;
        }
        AcceleratorPostOpType binaryType = resolveBinaryPostOpType(node.operation().opType());
        if (binaryType == null) {
            return null;
        }
        int otherInputNodeId = node.inputIds().getFirst() == currentOutputNodeId
                ? node.inputIds().get(1)
                : node.inputIds().getFirst();
        CompiledNode other = context.compiledNode(otherInputNodeId);
        if (other == null
                || other.dataType() != computeNode.dataType()
                || !other.contiguous()
                || other.hasStorageOffset()) {
            return null;
        }
        int[] otherShape = other.shape();
        int[] outputShape = computeNode.shape();
        boolean inputVector;
        if (otherShape.length == 1 && otherShape[0] == outputShape[1]) {
            inputVector = true;
        } else if (otherShape.length == 2 && otherShape[0] == outputShape[0] && otherShape[1] == outputShape[1]) {
            inputVector = false;
        } else {
            return null;
        }
        return AcceleratorPostOp.binary(binaryType, otherInputNodeId, inputVector);
    }

    private AcceleratorPostOpType resolveUnaryPostOpType(Operation.OpType opType) {
        if (opType == null) {
            return null;
        }
        return switch (opType) {
            case RELU -> AcceleratorPostOpType.RELU;
            case TANH, FAST_TANH -> AcceleratorPostOpType.TANH;
            case SIGMOID -> AcceleratorPostOpType.SIGMOID;
            case ABS -> AcceleratorPostOpType.ABS;
            case EXP, FAST_EXP -> AcceleratorPostOpType.EXP;
            case LOG -> AcceleratorPostOpType.LOG;
            case NEG -> AcceleratorPostOpType.NEG;
            case SQRT -> AcceleratorPostOpType.SQRT;
            case INV -> AcceleratorPostOpType.INV;
            case CLAMP_MIN -> AcceleratorPostOpType.CLAMP_MIN;
            case CLAMP_MAX -> AcceleratorPostOpType.CLAMP_MAX;
            case MUL_SCALAR -> AcceleratorPostOpType.MUL_SCALAR;
            default -> null;
        };
    }

    private AcceleratorPostOpType resolveBinaryPostOpType(Operation.OpType opType) {
        if (opType == null) {
            return null;
        }
        return switch (opType) {
            case ADD -> AcceleratorPostOpType.ADD;
            case MUL -> AcceleratorPostOpType.MUL;
            case DIV -> AcceleratorPostOpType.DIV;
            case SUB -> AcceleratorPostOpType.SUB;
            default -> null;
        };
    }
}

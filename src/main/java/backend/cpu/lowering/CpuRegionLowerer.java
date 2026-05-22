package backend.cpu.lowering;

import backend.ComputeBackend;
import backend.blas.BlasProvider;
import backend.cpu.fused.plan.FusedOperationBuilder;
import backend.cpu.kernels.linalg.matmul.plan.MatMulExecutionRoute;
import backend.cpu.kernels.linalg.matmul.plan.ResolvedMatMulHints;
import backend.cpu.kernels.plan.CpuExecutionPlanner;
import backend.cpu.nativecpu.NativeCpuCoverageEntry;
import backend.cpu.nativecpu.NativeCpuCoverageMatrix;
import backend.cpu.nativecpu.NativeCpuKernelPerformanceStatus;
import backend.cpu.nativecpu.NativeCpuParityMatrix;
import backend.cpu.nativecpu.layout.NativeCpuLayoutClass;
import backend.cpu.nativecpu.layout.NativeCpuStorageFamily;
import backend.cpu.nativecpu.layout.NativeSegmentKernelFamily;
import backend.cpu.nativecpu.layout.NativeSegmentStridedKernels;
import backend.cpu.nativecpu.layout.TensorPhysicalView;
import backend.lowering.BackendWorkspaceRequirement;
import backend.lowering.LoweredExecutionUnit;
import backend.lowering.LoweredUnitArtifact;
import backend.lowering.LoweredRegion;
import backend.lowering.LoweringFamily;
import backend.lowering.LoweringRequest;
import backend.lowering.LoweringResult;
import backend.lowering.RegionLowerer;
import backend.lowering.region.CpuFusedRegionPayload;
import backend.lowering.region.CpuNativeRegionPayload;
import backend.lowering.region.EmptyRegionPayload;
import backend.lowering.region.RegionBackendPayload;
import backend.lowering.region.RegionCost;
import backend.lowering.region.RegionDecision;
import backend.lowering.region.RegionExecutionGroup;
import backend.lowering.region.RegionExecutionKind;
import backend.lowering.region.RegionFallbackPlan;
import backend.lowering.region.RegionExecutionPlan;
import backend.lowering.region.RegionLegalityStatus;
import backend.lowering.region.RegionNodePlan;
import backend.lowering.region.RegionRole;
import backend.lowering.region.RegionStorageContract;
import config.runtime.CpuStorageProfile;
import config.runtime.RuntimeConfig;
import graph.AliasViewPolicy;
import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptor;
import graph.compile.planning.region.ExecutionUnit;
import graph.compile.planning.region.ExecutionUnitKind;
import graph.compile.planning.region.RegionOptimizationTrace;
import graph.compile.planning.value.GraphValueRef;
import operations.Operation;
import tensor.DataType;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public final class CpuRegionLowerer implements RegionLowerer {
    @Override
    public LoweringResult lower(LoweringRequest request) {
        if (request == null || request.region().target() != graph.compile.planning.partition.PartitionTarget.CPU) {
            return null;
        }
        if (!request.capabilities().supports(ComputeBackend.CPU)) {
            return null;
        }
        LoweredExecutionUnit nativeRegion = tryLowerNativeRegion(request);
        if (nativeRegion != null) {
            return new LoweringResult(
                    new LoweredRegion(request.region().regionId(), request.region().target(), List.of(nativeRegion)),
                    List.of()
            );
        }
        List<LoweredExecutionUnit> nativeSubregions = tryLowerNativeSubregions(request);
        if (!nativeSubregions.isEmpty()) {
            return new LoweringResult(
                    new LoweredRegion(request.region().regionId(), request.region().target(),
                            mergeNativeSubregionsWithCpuUnits(request, nativeSubregions)),
                    List.of()
            );
        }
        List<LoweredExecutionUnit> loweredUnits = new ArrayList<>(request.region().executionUnits().size());
        for (ExecutionUnit unit : request.region().executionUnits()) {
            loweredUnits.add(lowerUnit(unit, request));
        }
        return new LoweringResult(
                new LoweredRegion(request.region().regionId(), request.region().target(), loweredUnits),
                List.of()
        );
    }

    private LoweredExecutionUnit tryLowerNativeRegion(LoweringRequest request) {
        RuntimeConfig runtimeConfig = request.context().runtimeConfig();
        if (runtimeConfig == null || runtimeConfig.cpuStorageProfile() == CpuStorageProfile.CPU_ARRAY) {
            return null;
        }
        List<Integer> orderedNodeIds = request.region().sourcePartition().orderedNodeIds();
        if (orderedNodeIds == null || orderedNodeIds.size() < 2) {
            return null;
        }
        NativeRegionLegality legality = nativeRegionLegality(request, orderedNodeIds, runtimeConfig);
        if (!legality.selected()) {
            return null;
        }
        List<Integer> externalInputNodeIds = request.region().sourcePartition().externalInputNodeIds();
        List<Integer> boundaryOutputNodeIds = request.region().sourcePartition().outputValueRefs().stream()
                .map(CpuRegionLowerer::nodeIdFromPartitionRef)
                .filter(id -> id >= 0)
                .distinct()
                .toList();
        int anchorNodeId = boundaryOutputNodeIds.isEmpty()
                ? orderedNodeIds.getLast()
                : boundaryOutputNodeIds.getLast();
        if (!orderedNodeIds.contains(anchorNodeId)) {
            anchorNodeId = orderedNodeIds.getLast();
        }
        RegionExecutionPlan regionPlan = nativeRegionPlan(
                request,
                orderedNodeIds,
                externalInputNodeIds,
                boundaryOutputNodeIds.isEmpty() ? List.of(anchorNodeId) : boundaryOutputNodeIds,
                anchorNodeId,
                legality,
                request.region().regionId() + "-cpu-native"
        );
        return new LoweredExecutionUnit(
                request.region().regionId() + "-cpu-native",
                LoweringFamily.CPU_NATIVE_REGION,
                orderedNodeIds,
                externalInputNodeIds,
                regionPlan
        );
    }

    private List<LoweredExecutionUnit> tryLowerNativeSubregions(LoweringRequest request) {
        RuntimeConfig runtimeConfig = request.context().runtimeConfig();
        if (runtimeConfig == null || runtimeConfig.cpuStorageProfile() == CpuStorageProfile.CPU_ARRAY) {
            return List.of();
        }
        List<Integer> orderedNodeIds = request.region().sourcePartition().orderedNodeIds();
        if (orderedNodeIds == null || orderedNodeIds.size() < 2) {
            return List.of();
        }
        ArrayList<LoweredExecutionUnit> out = new ArrayList<>();
        int index = 0;
        while (index < orderedNodeIds.size()) {
            LoweredExecutionUnit selected = null;
            int selectedEndExclusive = -1;
            for (int end = orderedNodeIds.size(); end >= index + 2; end--) {
                List<Integer> candidateNodeIds = List.copyOf(orderedNodeIds.subList(index, end));
                NativeRegionLegality legality = nativeRegionLegality(request, candidateNodeIds, runtimeConfig);
                if (!legality.selected()) {
                    continue;
                }
                selected = nativeSubregionUnit(request, candidateNodeIds, legality);
                selectedEndExclusive = end;
                break;
            }
            if (selected == null) {
                index++;
                continue;
            }
            out.add(selected);
            index = selectedEndExclusive;
        }
        return List.copyOf(out);
    }

    private LoweredExecutionUnit nativeSubregionUnit(
            LoweringRequest request,
            List<Integer> orderedNodeIds,
            NativeRegionLegality legality
    ) {
        List<Integer> externalInputNodeIds = externalInputNodeIdsForNativeSubregion(request, orderedNodeIds);
        List<Integer> boundaryOutputNodeIds = boundaryOutputNodeIdsForNativeSubregion(request, orderedNodeIds);
        int anchorNodeId = boundaryOutputNodeIds.isEmpty()
                ? orderedNodeIds.getLast()
                : boundaryOutputNodeIds.getLast();
        String unitId = request.region().regionId()
                + "-cpu-native-" + orderedNodeIds.getFirst() + "-" + orderedNodeIds.getLast();
        RegionExecutionPlan regionPlan = nativeRegionPlan(
                request,
                orderedNodeIds,
                externalInputNodeIds,
                boundaryOutputNodeIds.isEmpty() ? List.of(anchorNodeId) : boundaryOutputNodeIds,
                anchorNodeId,
                legality,
                unitId
        );
        return new LoweredExecutionUnit(
                unitId,
                LoweringFamily.CPU_NATIVE_REGION,
                orderedNodeIds,
                externalInputNodeIds,
                regionPlan
        );
    }

    private List<LoweredExecutionUnit> mergeNativeSubregionsWithCpuUnits(
            LoweringRequest request,
            List<LoweredExecutionUnit> nativeSubregions
    ) {
        Map<Integer, LoweredExecutionUnit> nativeByFirstNode = nativeSubregions.stream()
                .collect(Collectors.toMap(unit -> unit.orderedNodeIds().getFirst(), unit -> unit));
        Set<Integer> nativeCoveredNodeIds = nativeSubregions.stream()
                .flatMap(unit -> unit.orderedNodeIds().stream())
                .collect(Collectors.toCollection(HashSet::new));
        Map<Integer, ExecutionUnit> executionUnitByFirstNode = request.region().executionUnits().stream()
                .filter(unit -> !unit.orderedNodeIds().isEmpty())
                .collect(Collectors.toMap(unit -> unit.orderedNodeIds().getFirst(), unit -> unit, (left, ignored) -> left));
        Set<String> emittedCpuUnitIds = new HashSet<>();
        ArrayList<LoweredExecutionUnit> out = new ArrayList<>();
        List<Integer> orderedNodeIds = request.region().sourcePartition().orderedNodeIds();
        int index = 0;
        while (index < orderedNodeIds.size()) {
            int nodeId = orderedNodeIds.get(index);
            LoweredExecutionUnit nativeUnit = nativeByFirstNode.get(nodeId);
            if (nativeUnit != null) {
                out.add(nativeUnit);
                index += nativeUnit.orderedNodeIds().size();
                continue;
            }
            ExecutionUnit cpuUnit = executionUnitByFirstNode.get(nodeId);
            if (cpuUnit != null
                    && emittedCpuUnitIds.add(cpuUnit.unitId())
                    && java.util.Collections.disjoint(cpuUnit.orderedNodeIds(), nativeCoveredNodeIds)) {
                out.add(lowerUnit(cpuUnit, request));
                index += cpuUnit.orderedNodeIds().size();
                continue;
            }
            if (!nativeCoveredNodeIds.contains(nodeId)) {
                out.add(lowerUnit(singleNodeUnit(request, nodeId), request));
            }
            index++;
        }
        return List.copyOf(out);
    }

    private ExecutionUnit singleNodeUnit(LoweringRequest request, int nodeId) {
        CompiledNode node = request.context().compiledNode(nodeId);
        List<GraphValueRef> inputRefs = node == null
                ? List.of()
                : node.inputIds().stream().map(GraphValueRef::node).toList();
        return new ExecutionUnit(
                request.region().regionId() + "-unit-" + nodeId + "-native-split",
                ExecutionUnitKind.UNIT_KERNEL,
                request.region().target(),
                inputRefs,
                List.of(GraphValueRef.node(nodeId)),
                List.of(),
                List.of(),
                List.of(nodeId),
                node == null ? 1L : Math.max(1L, node.flatDataSize()),
                node == null ? List.of() : node.inputIds(),
                new RegionOptimizationTrace(List.of("cpu-native-split-single-op:" + nodeId))
        );
    }

    private List<Integer> externalInputNodeIdsForNativeSubregion(
            LoweringRequest request,
            List<Integer> orderedNodeIds
    ) {
        Set<Integer> selected = Set.copyOf(orderedNodeIds);
        LinkedHashSet<Integer> externalInputs = new LinkedHashSet<>();
        for (int nodeId : orderedNodeIds) {
            CompiledNode node = request.context().compiledNode(nodeId);
            if (node == null) {
                continue;
            }
            for (int inputId : node.inputIds()) {
                if (!selected.contains(inputId)) {
                    externalInputs.add(inputId);
                }
            }
        }
        return List.copyOf(externalInputs);
    }

    private List<Integer> boundaryOutputNodeIdsForNativeSubregion(
            LoweringRequest request,
            List<Integer> orderedNodeIds
    ) {
        Set<Integer> selected = Set.copyOf(orderedNodeIds);
        LinkedHashSet<Integer> outputs = new LinkedHashSet<>();
        for (int nodeId : orderedNodeIds) {
            List<CompiledNode> consumers = consumersFor(request, nodeId);
            boolean hasSelectedConsumer = false;
            boolean hasExternalConsumer = false;
            for (CompiledNode consumer : consumers) {
                if (consumer != null && selected.contains(consumer.id())) {
                    hasSelectedConsumer = true;
                } else if (consumer != null) {
                    hasExternalConsumer = true;
                }
            }
            if (!hasSelectedConsumer || hasExternalConsumer) {
                outputs.add(nodeId);
            }
        }
        if (outputs.isEmpty()) {
            outputs.add(orderedNodeIds.getLast());
        }
        return List.copyOf(outputs);
    }

    private List<CompiledNode> consumersFor(LoweringRequest request, int producerNodeId) {
        ArrayList<CompiledNode> consumers = new ArrayList<>();
        for (CompiledNode candidate : request.context().compiledNodes()) {
            if (candidate != null && candidate.inputIds().contains(producerNodeId)) {
                consumers.add(candidate);
            }
        }
        return List.copyOf(consumers);
    }

    private NativeRegionLegality nativeRegionLegality(
            LoweringRequest request,
            List<Integer> orderedNodeIds,
            RuntimeConfig runtimeConfig
    ) {
        boolean provider = false;
        ArrayList<Integer> providerNodeIds = new ArrayList<>();
        ArrayList<Integer> localKernelNodeIds = new ArrayList<>();
        String rejection = "";
        for (int nodeId : orderedNodeIds) {
            CompiledNode node = request.context().compiledNode(nodeId);
            if (node == null || node.operation() == null) {
                return NativeRegionLegality.rejected("native-cpu-region-unsupported:missing-node");
            }
            NativeCpuCoverageEntry coverage = NativeCpuCoverageMatrix.entryFor(node.operation().opType(), node.dataType());
            if (!coverage.nativeSupported()) {
                return NativeRegionLegality.rejected(nonBlank(
                        coverage.fallbackReason(),
                        "native-cpu-region-unsupported:" + node.operation().opType().name().toLowerCase()
                ));
            }
            NativeNodeLayoutPlan layoutPlan = nativeNodeLayoutPlan(nodeId, request, coverage);
            if (!layoutPlan.selected()) {
                return NativeRegionLegality.rejected(layoutPlan.rejectionReason());
            }
            if (runtimeConfig.cpuStorageProfile() == CpuStorageProfile.AUTO
                    && !NativeCpuParityMatrix.isAutoEligible(node.operation().opType(), node.dataType())) {
                return NativeRegionLegality.rejected("native-cpu-region-auto-rejected-slow-op:"
                        + node.operation().opType().name().toLowerCase());
            }
            if (coverage.status() == NativeCpuKernelPerformanceStatus.LIBRARY_PROVIDER) {
                if (!nativeProviderEligible(node, request, runtimeConfig)) {
                    rejection = "native-cpu-region-provider-unavailable:" + node.operation().opType().name().toLowerCase();
                    return NativeRegionLegality.rejected(rejection);
                }
                provider = true;
                providerNodeIds.add(nodeId);
            } else {
                localKernelNodeIds.add(nodeId);
            }
        }
        if (!provider) {
            return NativeRegionLegality.rejected("native-cpu-region-rejected:no-provider-kernel");
        }
        return NativeRegionLegality.selected(providerNodeIds, localKernelNodeIds, "provider-backed-native-cpu-region");
    }

    private boolean nativeProviderEligible(CompiledNode node, LoweringRequest request, RuntimeConfig runtimeConfig) {
        if (node == null || node.operation() == null
                || (node.operation().opType() != Operation.OpType.MATMUL
                && node.operation().opType() != Operation.OpType.LINEAR)) {
            return false;
        }
        if (runtimeConfig.blas().provider() != BlasProvider.OPENBLAS_FFM) {
            return false;
        }
        if (node.inputIds().size() < 2) {
            return false;
        }
        CpuExecutionPlanner planner = CpuExecutionPlanner.from(runtimeConfig.cpuKernelConfig());
        ResolvedMatMulHints hints = planner.resolveMatMulHints(
                request.context().descriptor(node.inputIds().get(0)),
                request.context().descriptor(node.inputIds().get(1)),
                request.context().descriptor(node.id()),
                runtimeConfig.blas(),
                runtimeConfig.cpuStorageProfile(),
                false
        );
        return hints.route() == MatMulExecutionRoute.OPENBLAS_NATIVE_SEGMENT;
    }

    private RegionExecutionPlan nativeRegionPlan(
            LoweringRequest request,
            List<Integer> orderedNodeIds,
            List<Integer> externalInputNodeIds,
            List<Integer> boundaryOutputNodeIds,
            int anchorNodeId,
            NativeRegionLegality legality,
            String regionKey
    ) {
        List<RegionNodePlan> nodePlans = orderedNodeIds.stream()
                .map(nodeId -> nativeNodePlan(nodeId, request, boundaryOutputNodeIds))
                .toList();
        List<RegionExecutionGroup> executionGroups = nativeExecutionGroups(
                request,
                orderedNodeIds,
                externalInputNodeIds,
                boundaryOutputNodeIds,
                regionKey
        );
        CpuNativeRegionPayload payload = new CpuNativeRegionPayload(
                "OPENBLAS_FFM",
                legality.providerNodeIds(),
                legality.localKernelNodeIds(),
                List.of(),
                List.of(new RegionFallbackPlan(
                        orderedNodeIds,
                        boundaryOutputNodeIds,
                        "CPU_ARRAY",
                        "native-cpu-region-array-fallback",
                        externalInputNodeIds
                ))
        );
        return new RegionExecutionPlan(
                regionKey + "/plan",
                graph.compile.planning.partition.PartitionTarget.CPU,
                LoweringFamily.CPU_NATIVE_REGION,
                anchorNodeId,
                orderedNodeIds,
                externalInputNodeIds,
                boundaryOutputNodeIds,
                nodePlans,
                executionGroups,
                RegionCost.ofWork(request.region().sourcePartition().estimatedWork()),
                RegionDecision.selected(LoweringFamily.CPU_NATIVE_REGION.id(), legality.reason()),
                payload
        );
    }

    private List<RegionExecutionGroup> nativeExecutionGroups(
            LoweringRequest request,
            List<Integer> orderedNodeIds,
            List<Integer> externalInputNodeIds,
            List<Integer> boundaryOutputNodeIds,
            String regionKey
    ) {
        ArrayList<RegionExecutionGroup> groups = new ArrayList<>();
        ArrayList<Integer> currentLocalKernels = new ArrayList<>();
        int groupIndex = 0;
        for (int nodeId : orderedNodeIds) {
            RegionExecutionKind kind = nativeExecutionKind(nodeId, request);
            if (kind == RegionExecutionKind.PROVIDER_CALL || kind == RegionExecutionKind.VIEW) {
                if (!currentLocalKernels.isEmpty()) {
                    groups.add(nativeExecutionGroup(
                            request,
                            regionKey,
                            groupIndex++,
                            currentLocalKernels,
                            RegionExecutionKind.DIRECT_KERNEL,
                            "SEGMENT_SCALAR",
                            externalInputNodeIds,
                            boundaryOutputNodeIds,
                            "native-cpu-local-kernel"
                    ));
                    currentLocalKernels = new ArrayList<>();
                }
                groups.add(nativeExecutionGroup(
                        request,
                        regionKey,
                        groupIndex++,
                        List.of(nodeId),
                        kind,
                        nativePhysicalKernel(nodeId, request),
                        externalInputNodeIds,
                        boundaryOutputNodeIds,
                        kind == RegionExecutionKind.PROVIDER_CALL ? "native-cpu-provider" : "native-cpu-view"
                ));
                continue;
            }
            currentLocalKernels.add(nodeId);
        }
        if (!currentLocalKernels.isEmpty()) {
            groups.add(nativeExecutionGroup(
                    request,
                    regionKey,
                    groupIndex,
                    currentLocalKernels,
                    RegionExecutionKind.DIRECT_KERNEL,
                    "SEGMENT_SCALAR",
                    externalInputNodeIds,
                    boundaryOutputNodeIds,
                    "native-cpu-local-kernel"
            ));
        }
        return List.copyOf(groups);
    }

    private RegionExecutionGroup nativeExecutionGroup(
            LoweringRequest request,
            String regionKey,
            int groupIndex,
            List<Integer> groupNodeIds,
            RegionExecutionKind executionKind,
            String physicalKernel,
            List<Integer> externalInputNodeIds,
            List<Integer> boundaryOutputNodeIds,
            String reason
    ) {
        java.util.LinkedHashSet<Integer> groupSet = new java.util.LinkedHashSet<>(groupNodeIds);
        java.util.LinkedHashSet<Integer> inputs = new java.util.LinkedHashSet<>();
        for (int nodeId : groupNodeIds) {
            CompiledNode node = request.context().compiledNode(nodeId);
            if (node == null) {
                continue;
            }
            for (int inputId : node.inputIds()) {
                if (!groupSet.contains(inputId)) {
                    inputs.add(inputId);
                }
            }
        }
        java.util.LinkedHashSet<Integer> outputs = new java.util.LinkedHashSet<>();
        for (int i = 0; i < groupNodeIds.size(); i++) {
            int nodeId = groupNodeIds.get(i);
            boolean lastInGroup = i == groupNodeIds.size() - 1;
            if (lastInGroup || boundaryOutputNodeIds.contains(nodeId)) {
                outputs.add(nodeId);
            }
        }
        return new RegionExecutionGroup(
                regionKey + "-group-" + groupIndex,
                groupNodeIds,
                executionKind,
                physicalKernel,
                inputs.isEmpty() ? externalInputNodeIds : List.copyOf(inputs),
                List.copyOf(outputs),
                List.of(),
                nativeGroupStorageContract(request, groupNodeIds, executionKind),
                reason
        );
    }

    private RegionStorageContract nativeGroupStorageContract(
            LoweringRequest request,
            List<Integer> groupNodeIds,
            RegionExecutionKind executionKind
    ) {
        if (executionKind == RegionExecutionKind.VIEW) {
            return RegionStorageContract.VIEW_ALIAS;
        }
        boolean hasCpuArrayOutput = groupNodeIds.stream()
                .map(nodeId -> request.context().compiledNode(nodeId))
                .anyMatch(node -> {
                    Operation op = node == null ? null : node.operation();
                    NativeCpuCoverageEntry coverage = NativeCpuCoverageMatrix.entryFor(
                            op == null ? Operation.OpType.UNKNOWN : op.opType(),
                            node == null ? tensor.DataType.FLOAT64 : node.dataType()
                    );
                    return !coverage.preservesNativeStorage();
                });
        return hasCpuArrayOutput ? RegionStorageContract.MIXED_BOUNDARY : RegionStorageContract.CPU_NATIVE;
    }

    private RegionExecutionKind nativeExecutionKind(int nodeId, LoweringRequest request) {
        CompiledNode node = request.context().compiledNode(nodeId);
        Operation op = node == null ? null : node.operation();
        NativeCpuCoverageEntry coverage = NativeCpuCoverageMatrix.entryFor(
                op == null ? Operation.OpType.UNKNOWN : op.opType(),
                node == null ? tensor.DataType.FLOAT64 : node.dataType()
        );
        return switch (coverage.status()) {
            case LIBRARY_PROVIDER -> RegionExecutionKind.PROVIDER_CALL;
            case VIEW_ONLY -> RegionExecutionKind.VIEW;
            default -> RegionExecutionKind.DIRECT_KERNEL;
        };
    }

    private String nativePhysicalKernel(int nodeId, LoweringRequest request) {
        CompiledNode node = request.context().compiledNode(nodeId);
        Operation op = node == null ? null : node.operation();
        NativeCpuCoverageEntry coverage = NativeCpuCoverageMatrix.entryFor(
                op == null ? Operation.OpType.UNKNOWN : op.opType(),
                node == null ? tensor.DataType.FLOAT64 : node.dataType()
        );
        return coverage.family().name();
    }

    private RegionNodePlan nativeNodePlan(
            int nodeId,
            LoweringRequest request,
            List<Integer> boundaryOutputNodeIds
    ) {
        CompiledNode node = request.context().compiledNode(nodeId);
        Operation op = node == null ? null : node.operation();
        NativeCpuCoverageEntry coverage = NativeCpuCoverageMatrix.entryFor(
                op == null ? Operation.OpType.UNKNOWN : op.opType(),
                node == null ? tensor.DataType.FLOAT64 : node.dataType()
        );
        NativeNodeLayoutPlan layoutPlan = nativeNodeLayoutPlan(nodeId, request, coverage);
        RegionExecutionKind executionKind = switch (coverage.status()) {
            case LIBRARY_PROVIDER -> RegionExecutionKind.PROVIDER_CALL;
            case VIEW_ONLY -> RegionExecutionKind.VIEW;
            default -> RegionExecutionKind.DIRECT_KERNEL;
        };
        RegionRole role = boundaryOutputNodeIds.contains(nodeId)
                ? RegionRole.BOUNDARY_OUTPUT
                : coverage.status() == NativeCpuKernelPerformanceStatus.LIBRARY_PROVIDER
                        ? RegionRole.PROVIDER
                        : !coverage.preservesNativeStorage()
                                ? RegionRole.CONTROL
                                : coverage.status() == NativeCpuKernelPerformanceStatus.VIEW_ONLY
                                ? RegionRole.VIEW_ALIAS
                                : RegionRole.LOCAL_KERNEL;
        RegionStorageContract storageContract = !coverage.preservesNativeStorage()
                ? RegionStorageContract.CPU_ARRAY
                : coverage.status() == NativeCpuKernelPerformanceStatus.VIEW_ONLY
                        ? RegionStorageContract.VIEW_ALIAS
                        : RegionStorageContract.CPU_NATIVE;
        return new RegionNodePlan(
                nodeId,
                op == null ? Operation.OpType.UNKNOWN : op.opType(),
                node == null ? tensor.DataType.FLOAT64 : node.dataType(),
                role,
                executionKind,
                coverage.family().name(),
                layoutPlan.segmentKernelFamily(),
                layoutPlan.layoutClass(),
                layoutPlan.inputLayoutClasses(),
                layoutPlan.outputLayoutClass(),
                layoutPlan.materializationReason(),
                storageContract,
                node == null ? List.of() : node.inputIds(),
                List.of(nodeId),
                RegionLegalityStatus.SELECTED,
                coverage.status().name().toLowerCase()
        );
    }

    private NativeNodeLayoutPlan nativeNodeLayoutPlan(
            int nodeId,
            LoweringRequest request,
            NativeCpuCoverageEntry coverage
    ) {
        CompiledNode node = request.context().compiledNode(nodeId);
        Operation op = node == null ? null : node.operation();
        List<Integer> inputNodeIds = node == null ? List.of() : node.inputIds();
        ArrayList<String> inputLayoutClasses = new ArrayList<>();
        ArrayList<NativeCpuLayoutClass> inputLayouts = new ArrayList<>();
        for (int inputNodeId : inputNodeIds) {
            NativeCpuLayoutClass inputLayout = nativeLayoutClass(inputNodeId, request);
            inputLayouts.add(inputLayout);
            inputLayoutClasses.add(inputLayout.name());
            if (inputLayout == NativeCpuLayoutClass.UNSUPPORTED_LAYOUT) {
                return NativeNodeLayoutPlan.rejected(
                        NativeCpuLayoutClass.UNSUPPORTED_LAYOUT.name(),
                        inputLayoutClasses,
                        nativeLayoutClass(nodeId, request).name(),
                        "",
                        "native-layout-unsupported:input-node-" + inputNodeId
                );
            }
        }
        NativeCpuLayoutClass outputLayout = nativeLayoutClass(nodeId, request);
        if (outputLayout == NativeCpuLayoutClass.UNSUPPORTED_LAYOUT) {
            return NativeNodeLayoutPlan.rejected(
                    NativeCpuLayoutClass.UNSUPPORTED_LAYOUT.name(),
                    inputLayoutClasses,
                    outputLayout.name(),
                    "",
                    "native-layout-unsupported:node-" + nodeId
            );
        }
        NativeCpuLayoutClass accessLayout = nativeAccessLayout(op, coverage, inputLayouts, outputLayout);
        String materializationReason = nativeLayoutMaterializationReason(request, node, op, coverage, inputLayouts, outputLayout);
        if (!materializationReason.isBlank()) {
            return NativeNodeLayoutPlan.rejected(
                    accessLayout.name(),
                    inputLayoutClasses,
                    outputLayout.name(),
                    materializationReason,
                    materializationReason
            );
        }
        return NativeNodeLayoutPlan.selected(
                accessLayout.name(),
                inputLayoutClasses,
                outputLayout.name(),
                "",
                nativeSegmentKernelFamily(coverage, accessLayout).name()
        );
    }

    private NativeCpuLayoutClass nativeLayoutClass(int nodeId, LoweringRequest request) {
        try {
            CompiledTensorDescriptor descriptor = request.context().descriptor(nodeId);
            return TensorPhysicalView.fromDescriptor(descriptor, NativeCpuStorageFamily.CPU_NATIVE).layoutClass();
        } catch (RuntimeException ignored) {
            return NativeCpuLayoutClass.UNSUPPORTED_LAYOUT;
        }
    }

    private NativeCpuLayoutClass nativeAccessLayout(
            Operation op,
            NativeCpuCoverageEntry coverage,
            List<NativeCpuLayoutClass> inputLayouts,
            NativeCpuLayoutClass outputLayout
    ) {
        if (coverage.status() == NativeCpuKernelPerformanceStatus.VIEW_ONLY) {
            return NativeCpuLayoutClass.VIEW_ALIAS_ONLY;
        }
        for (NativeCpuLayoutClass candidate : List.of(
                NativeCpuLayoutClass.GENERAL_STRIDED_READ_STRIDED_WRITE,
                NativeCpuLayoutClass.GENERAL_STRIDED_READ_DENSE_WRITE,
                NativeCpuLayoutClass.TRANSPOSE_2D_READ_DENSE_WRITE,
                NativeCpuLayoutClass.LAST_DIM_BIAS_BROADCAST,
                NativeCpuLayoutClass.BROADCAST_READ_DENSE_WRITE,
                NativeCpuLayoutClass.OFFSET_CONTIGUOUS
        )) {
            if (inputLayouts.contains(candidate)) {
                return candidate;
            }
        }
        if (outputLayout != NativeCpuLayoutClass.DENSE_CONTIGUOUS) {
            return outputLayout;
        }
        return op == null ? outputLayout : NativeCpuLayoutClass.DENSE_CONTIGUOUS;
    }

    private String nativeLayoutMaterializationReason(
            LoweringRequest request,
            CompiledNode node,
            Operation op,
            NativeCpuCoverageEntry coverage,
            List<NativeCpuLayoutClass> inputLayouts,
            NativeCpuLayoutClass outputLayout
    ) {
        if (coverage.status() == NativeCpuKernelPerformanceStatus.VIEW_ONLY) {
            return "";
        }
        Operation.OpType opType = op == null ? Operation.OpType.UNKNOWN : op.opType();
        for (NativeCpuLayoutClass inputLayout : inputLayouts) {
            if (inputLayout == NativeCpuLayoutClass.DENSE_CONTIGUOUS) {
                continue;
            }
            if (coverage.status() == NativeCpuKernelPerformanceStatus.LIBRARY_PROVIDER) {
                return "native-layout-materialization-required:provider-dense-input";
            }
            if (nativeSegmentLayoutEligible(node, request, op, coverage)) {
                continue;
            }
            if (inputLayout == NativeCpuLayoutClass.OFFSET_CONTIGUOUS) {
                return "native-layout-materialization-required:offset-input:" + opType.name().toLowerCase();
            }
            if (inputLayout == NativeCpuLayoutClass.BROADCAST_READ_DENSE_WRITE
                    || inputLayout == NativeCpuLayoutClass.LAST_DIM_BIAS_BROADCAST) {
                return "native-layout-materialization-required:broadcast-input:" + opType.name().toLowerCase();
            }
            return "native-layout-unsupported:strided-input:" + opType.name().toLowerCase();
        }
        if (outputLayout != NativeCpuLayoutClass.DENSE_CONTIGUOUS
                && coverage.status() != NativeCpuKernelPerformanceStatus.VIEW_ONLY) {
            return "native-layout-unsupported:strided-output:" + opType.name().toLowerCase();
        }
        return "";
    }

    private boolean nativeSegmentLayoutEligible(
            CompiledNode node,
            LoweringRequest request,
            Operation op,
            NativeCpuCoverageEntry coverage
    ) {
        if (op == null || coverage == null) {
            return false;
        }
        DataType dataType = coverage.dataType();
        Operation.OpType opType = op.opType();
        if (NativeSegmentStridedKernels.supportsUnary(op, dataType)) {
            return true;
        }
        if (NativeSegmentStridedKernels.supportsBinary(op, dataType)) {
            return true;
        }
        if (NativeSegmentStridedKernels.supportsReduction(opType, dataType)) {
            return true;
        }
        if (isCompareOp(opType) && node != null && node.inputIds().size() >= 2) {
            DataType leftDataType = request.context().descriptor(node.inputIds().get(0)).dataType();
            DataType rightDataType = request.context().descriptor(node.inputIds().get(1)).dataType();
            return leftDataType == rightDataType && NativeSegmentStridedKernels.supportsCompare(op, leftDataType);
        }
        return opType == Operation.OpType.WHERE
                && (dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64 || dataType == DataType.BFLOAT16);
    }

    private static boolean isCompareOp(Operation.OpType opType) {
        return opType == Operation.OpType.GT
                || opType == Operation.OpType.GE
                || opType == Operation.OpType.LT
                || opType == Operation.OpType.LE
                || opType == Operation.OpType.EQ
                || opType == Operation.OpType.NE;
    }

    private NativeSegmentKernelFamily nativeSegmentKernelFamily(
            NativeCpuCoverageEntry coverage,
            NativeCpuLayoutClass accessLayout
    ) {
        if (coverage.status() == NativeCpuKernelPerformanceStatus.LIBRARY_PROVIDER) {
            return NativeSegmentKernelFamily.PROVIDER;
        }
        if (coverage.status() == NativeCpuKernelPerformanceStatus.VIEW_ONLY) {
            return NativeSegmentKernelFamily.VIEW_ALIAS;
        }
        if (accessLayout == NativeCpuLayoutClass.DENSE_CONTIGUOUS) {
            return NativeSegmentKernelFamily.SEGMENT_DENSE_SCALAR;
        }
        return NativeSegmentKernelFamily.SEGMENT_STRIDED_SCALAR;
    }

    private LoweringFamily chooseSingleOpFamily(ExecutionUnit unit, LoweringRequest request) {
        CompiledNode node = unit.orderedNodeIds().isEmpty() ? null : request.context().compiledNode(unit.orderedNodeIds().getFirst());
        if (node == null || node.operation() == null) {
            return LoweringFamily.DIRECT_KERNEL;
        }
        boolean blasEnabled = request.context().runtimeConfig() != null
                && request.context().runtimeConfig().blas().provider() != BlasProvider.NONE;
        long blasMinWork = request.context().runtimeConfig() == null
                ? Long.MAX_VALUE
                : request.context().runtimeConfig().blas().matmulMinWork();
        boolean matmulFamily = node.operation().opType() == operations.Operation.OpType.MATMUL
                || node.operation().opType() == operations.Operation.OpType.LINEAR;
        if (blasEnabled && matmulFamily && unit.estimatedWork() >= blasMinWork) {
            return LoweringFamily.BLAS;
        }
        return LoweringFamily.DIRECT_KERNEL;
    }

    private LoweredExecutionUnit lowerUnit(ExecutionUnit unit, LoweringRequest request) {
        LoweringFamily family;
        LoweredUnitArtifact legacyArtifact = null;
        if (unit.kind() == ExecutionUnitKind.FUSED_ELEMENTWISE) {
            family = LoweringFamily.FUSED_NATIVE;
            legacyArtifact = FusedOperationBuilder.build(
                    unit.orderedNodeIds(),
                    request.context()::compiledNode,
                    request.context().descriptorIndex()
            );
        } else {
            family = chooseSingleOpFamily(unit, request);
        }
        RegionExecutionPlan regionPlan = regionPlan(unit, request, family, legacyArtifact);
        return new LoweredExecutionUnit(
                unit.unitId(),
                family,
                unit.orderedNodeIds(),
                unit.inputValueRefs().stream()
                        .map(CpuRegionLowerer::nodeIdFromRef)
                        .map(nodeId -> resolveExecutionInputNodeId(nodeId, request))
                        .filter(id -> id >= 0)
                        .distinct()
                        .toList(),
                regionPlan
        );
    }

    private RegionExecutionPlan regionPlan(
            ExecutionUnit unit,
            LoweringRequest request,
            LoweringFamily family,
            LoweredUnitArtifact legacyArtifact
    ) {
        List<Integer> orderedNodeIds = unit.orderedNodeIds();
        int anchorNodeId = orderedNodeIds.getLast();
        List<Integer> externalInputNodeIds = unit.inputValueRefs().stream()
                .map(CpuRegionLowerer::nodeIdFromRef)
                .map(nodeId -> resolveExecutionInputNodeId(nodeId, request))
                .filter(id -> id >= 0)
                .distinct()
                .toList();
        List<Integer> boundaryOutputNodeIds = unit.outputValueRefs().stream()
                .map(CpuRegionLowerer::nodeIdFromRef)
                .filter(id -> id >= 0)
                .distinct()
                .toList();
        RegionBackendPayload payload = legacyArtifact instanceof backend.cpu.fused.plan.FusedOperationPreparation fused
                ? new CpuFusedRegionPayload(fused)
                : EmptyRegionPayload.INSTANCE;
        RegionExecutionKind executionKind = switch (family) {
            case FUSED_NATIVE -> RegionExecutionKind.FUSED_KERNEL;
            case BLAS -> RegionExecutionKind.PROVIDER_CALL;
            default -> RegionExecutionKind.DIRECT_KERNEL;
        };
        RegionStorageContract storageContract = RegionStorageContract.CPU_ARRAY;
        String physicalKernel = family.id();
        List<RegionNodePlan> nodePlans = orderedNodeIds.stream()
                .map(nodeId -> nodePlan(nodeId, request, family, executionKind, physicalKernel, storageContract, boundaryOutputNodeIds))
                .toList();
        RegionExecutionGroup group = new RegionExecutionGroup(
                unit.unitId() + "-group-0",
                orderedNodeIds,
                executionKind,
                physicalKernel,
                externalInputNodeIds,
                boundaryOutputNodeIds.isEmpty() ? List.of(anchorNodeId) : boundaryOutputNodeIds,
                List.of(),
                storageContract,
                "cpu-lowered-unit"
        );
        return new RegionExecutionPlan(
                request.region().regionId() + "/" + unit.unitId(),
                graph.compile.planning.partition.PartitionTarget.CPU,
                family,
                anchorNodeId,
                orderedNodeIds,
                externalInputNodeIds,
                boundaryOutputNodeIds.isEmpty() ? List.of(anchorNodeId) : boundaryOutputNodeIds,
                nodePlans,
                List.of(group),
                RegionCost.ofWork(unit.estimatedWork()),
                RegionDecision.selected(family.id(), "cpu-lowered-unit"),
                payload
        );
    }

    private RegionNodePlan nodePlan(
            int nodeId,
            LoweringRequest request,
            LoweringFamily family,
            RegionExecutionKind executionKind,
            String physicalKernel,
            RegionStorageContract storageContract,
            List<Integer> boundaryOutputNodeIds
    ) {
        CompiledNode node = request.context().compiledNode(nodeId);
        Operation op = node == null ? null : node.operation();
        RegionRole role = boundaryOutputNodeIds.contains(nodeId)
                ? RegionRole.BOUNDARY_OUTPUT
                : family == LoweringFamily.BLAS ? RegionRole.PROVIDER : RegionRole.LOCAL_KERNEL;
        return new RegionNodePlan(
                nodeId,
                op == null ? Operation.OpType.UNKNOWN : op.opType(),
                node == null ? tensor.DataType.FLOAT64 : node.dataType(),
                role,
                executionKind,
                physicalKernel,
                storageContract,
                node == null ? List.of() : node.inputIds(),
                List.of(nodeId),
                RegionLegalityStatus.SELECTED,
                "cpu-lowered-unit"
        );
    }

    private static int nodeIdFromRef(GraphValueRef ref) {
        return ref == null ? -1 : ref.nodeId();
    }

    private static int nodeIdFromPartitionRef(GraphValueRef ref) {
        return nodeIdFromRef(ref);
    }

    private static String nonBlank(String value, String fallback) {
        return value == null || value.isBlank() ? fallback : value;
    }

    private int resolveExecutionInputNodeId(int nodeId, LoweringRequest request) {
        int current = nodeId;
        while (current >= 0) {
            CompiledNode node = request.context().compiledNode(current);
            if (node == null || node.operation() == null || node.inputIds().isEmpty()) {
                return current;
            }
            if (!AliasViewPolicy.aliasesInput0AtRuntime(node, request.context().descriptorIndex())) {
                return current;
            }
            current = node.inputIds().getFirst();
        }
        return nodeId;
    }

    private record NativeRegionLegality(
            boolean selected,
            List<Integer> providerNodeIds,
            List<Integer> localKernelNodeIds,
            String reason
    ) {
        private NativeRegionLegality {
            providerNodeIds = List.copyOf(providerNodeIds == null ? List.of() : providerNodeIds);
            localKernelNodeIds = List.copyOf(localKernelNodeIds == null ? List.of() : localKernelNodeIds);
            reason = reason == null ? "" : reason;
        }

        static NativeRegionLegality selected(List<Integer> providerNodeIds, List<Integer> localKernelNodeIds, String reason) {
            return new NativeRegionLegality(true, providerNodeIds, localKernelNodeIds, reason);
        }

        static NativeRegionLegality rejected(String reason) {
            return new NativeRegionLegality(false, List.of(), List.of(), reason);
        }
    }

    private record NativeNodeLayoutPlan(
            boolean selected,
            String layoutClass,
            List<String> inputLayoutClasses,
            String outputLayoutClass,
            String materializationReason,
            String segmentKernelFamily,
            String rejectionReason
    ) {
        private NativeNodeLayoutPlan {
            layoutClass = layoutClass == null ? "" : layoutClass;
            inputLayoutClasses = List.copyOf(inputLayoutClasses == null ? List.of() : inputLayoutClasses);
            outputLayoutClass = outputLayoutClass == null ? "" : outputLayoutClass;
            materializationReason = materializationReason == null ? "" : materializationReason;
            segmentKernelFamily = segmentKernelFamily == null ? "" : segmentKernelFamily;
            rejectionReason = rejectionReason == null ? "" : rejectionReason;
        }

        static NativeNodeLayoutPlan selected(
                String layoutClass,
                List<String> inputLayoutClasses,
                String outputLayoutClass,
                String materializationReason,
                String segmentKernelFamily
        ) {
            return new NativeNodeLayoutPlan(
                    true,
                    layoutClass,
                    inputLayoutClasses,
                    outputLayoutClass,
                    materializationReason,
                    segmentKernelFamily,
                    ""
            );
        }

        static NativeNodeLayoutPlan rejected(
                String layoutClass,
                List<String> inputLayoutClasses,
                String outputLayoutClass,
                String materializationReason,
                String rejectionReason
        ) {
            return new NativeNodeLayoutPlan(
                    false,
                    layoutClass,
                    inputLayoutClasses,
                    outputLayoutClass,
                    materializationReason,
                    "",
                    rejectionReason
            );
        }
    }
}

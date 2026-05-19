package graph.execution.trace.contrib;

import backend.cpu.nativecpu.NativeCpuParityMatrix;
import backend.cpu.nativecpu.layout.NativeCpuLayoutClass;
import backend.cpu.nativecpu.layout.NativeCpuStorageFamily;
import backend.cpu.nativecpu.layout.TensorPhysicalView;
import backend.lowering.region.RegionExecutionPlan;
import backend.lowering.region.RegionNodePlan;
import graph.CompiledNode;
import graph.optimizer.cost.CostComponent;
import tensor.Tensor;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

final class BackendTraceSupport {
    private BackendTraceSupport() {
    }

    static void addRegionPlanAttrs(LinkedHashMap<String, Object> attrs, RegionExecutionPlan regionPlan) {
        attrs.put("regionId", regionPlan.regionId());
        attrs.put("regionTarget", regionPlan.target().name());
        attrs.put("loweringFamily", regionPlan.loweringFamily().name());
        attrs.put("anchorNodeId", regionPlan.anchorNodeId());
        attrs.put("orderedNodeIds", regionPlan.orderedNodeIds());
        attrs.put("boundaryOutputNodeIds", regionPlan.boundaryOutputNodeIds());
        attrs.put("regionNodeCount", regionPlan.orderedNodeIds().size());
        attrs.put("regionDecision", regionPlan.decision().selected() ? "SELECTED" : "REJECTED");
        attrs.put("regionReason", regionPlan.decision().reason());
        attrs.put("regionExecutionKindSummary", regionPlan.executionGroups().stream()
                .map(group -> group.executionKind().name())
                .distinct()
                .toList());
        attrs.put("regionStorageContractSummary", regionPlan.executionGroups().stream()
                .map(group -> group.storageContract().name())
                .distinct()
                .toList());
    }

    static Map<String, Integer> stringCounts(List<String> values) {
        LinkedHashMap<String, Integer> counts = new LinkedHashMap<>();
        if (values == null) {
            return counts;
        }
        for (String value : values) {
            String key = value == null || value.isBlank() ? "UNKNOWN" : value;
            counts.merge(key, 1, Integer::sum);
        }
        return counts;
    }

    static long nativeCpuStridedNodeCount(List<RegionNodePlan> nodePlans) {
        if (nodePlans == null) {
            return 0L;
        }
        return nodePlans.stream()
                .filter(nodePlan -> nodePlan != null && !isDenseOrViewLayout(nodePlan.layoutClass()))
                .count();
    }

    static boolean isSegmentScalarNodePlan(RegionNodePlan nodePlan) {
        if (nodePlan == null) {
            return false;
        }
        String physicalKernel = nodePlan.physicalKernel();
        String segmentKernelFamily = nodePlan.segmentKernelFamily();
        return "SEGMENT_SCALAR".equals(physicalKernel)
                || "SEGMENT_DENSE_SCALAR".equals(segmentKernelFamily)
                || "SEGMENT_STRIDED_SCALAR".equals(segmentKernelFamily);
    }

    static List<String> nativeCpuParityStoragePaths(RegionNodePlan nodePlan) {
        if (nodePlan == null) {
            return List.of();
        }
        return NativeCpuParityMatrix.entryFor(nodePlan.opType(), nodePlan.dataType()).storagePaths().stream()
                .map(Enum::name)
                .sorted()
                .toList();
    }

    static List<String> nativeCpuParityLayoutCapabilities(RegionNodePlan nodePlan) {
        if (nodePlan == null) {
            return List.of();
        }
        return NativeCpuParityMatrix.entryFor(nodePlan.opType(), nodePlan.dataType()).layoutCapabilities().stream()
                .map(Enum::name)
                .sorted()
                .toList();
    }

    static List<String> nativeCpuParityResultResidencies(RegionNodePlan nodePlan) {
        if (nodePlan == null) {
            return List.of();
        }
        return NativeCpuParityMatrix.entryFor(nodePlan.opType(), nodePlan.dataType()).resultResidencies().stream()
                .map(Enum::name)
                .sorted()
                .toList();
    }

    static boolean isBf16PromotedRegionNodePlan(RegionNodePlan nodePlan) {
        if (nodePlan == null || nodePlan.dataType() != tensor.DataType.BFLOAT16) {
            return false;
        }
        return switch (nodePlan.opType()) {
            case ADD, SUB, MUL, DIV, MIN, MAX, MUL_SCALAR, NEG, RELU, ABS, CLAMP_MIN, CLAMP_MAX, WHERE, SUM, MEAN -> true;
            default -> false;
        };
    }

    static boolean isDenseOrViewLayout(String layoutClass) {
        return "DENSE_CONTIGUOUS".equals(layoutClass) || "VIEW_ALIAS_ONLY".equals(layoutClass);
    }

    static String nodeLayoutClassName(CompiledNode node) {
        if (node == null) {
            return NativeCpuLayoutClass.UNSUPPORTED_LAYOUT.name();
        }
        return layoutClassName(
                node.id(),
                node.dataType(),
                node.shape(),
                node.strides(),
                node.storageOffset()
        );
    }

    static String tensorLayoutClassName(Tensor tensor) {
        if (tensor == null) {
            return NativeCpuLayoutClass.UNSUPPORTED_LAYOUT.name();
        }
        return layoutClassName(
                0,
                tensor.getDataType(),
                tensor.getShapeUnsafe(),
                tensor.getStridesUnsafe(),
                tensor.getStorageOffsetUnsafe()
        );
    }

    static String layoutClassName(
            int nodeId,
            tensor.DataType dataType,
            int[] shape,
            int[] strides,
            int storageOffset
    ) {
        try {
            return TensorPhysicalView.of(
                    Math.max(0, nodeId),
                    dataType,
                    shape,
                    strides,
                    storageOffset,
                    NativeCpuStorageFamily.CPU_NATIVE
            ).layoutClass().name();
        } catch (RuntimeException ignored) {
            return NativeCpuLayoutClass.UNSUPPORTED_LAYOUT.name();
        }
    }

    static String costComponentSummary(CostComponent component) {
        if (component == null) {
            return "";
        }
        return component.name()
                + "=" + String.format(Locale.US, "%.6f", component.value())
                + " " + component.direction().name()
                + " (" + component.reason() + ")";
    }

}

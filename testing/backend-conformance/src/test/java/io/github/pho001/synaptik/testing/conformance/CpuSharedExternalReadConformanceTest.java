package io.github.pho001.synaptik.testing.conformance;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFusionDecision;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Conformance of repeated producerless reads against CPU's retained preparation topology. */
final class CpuSharedExternalReadConformanceTest {
    @Test
    void repeatedWeightIsExternalWhileProducedIntermediateCrossesUnits() {
        for (boolean relu : List.of(false, true)) {
            var plan = new CpuPartitionPreparer().analyze(context(relu)).plan();
            var selection = (CpuFusionDecision.Selection) plan.fusionDecisions().getLast();
            int weight = plan.boundaryValues().indexOf(new ValueId(1));
            int intermediate = plan.boundaryValues().indexOf(new ValueId(2));
            assertEquals(2, occurrences(selection, weight,
                    CpuFusionDecision.BoundaryRole.EXTERNAL_READ));
            assertEquals(2, occurrences(selection, intermediate,
                    CpuFusionDecision.BoundaryRole.CROSS_UNIT));
            assertEquals(2, plan.specializedSubgraphs().size());
            assertFalse(plan.bufferDeclarations().isEmpty());
        }
    }

    private static long occurrences(CpuFusionDecision.Selection selection, int relative,
            CpuFusionDecision.BoundaryRole role) {
        return selection.compatibilityBaseline().units().stream()
                .flatMap(unit -> unit.boundaries().stream())
                .filter(boundary -> boundary.relativeBoundaryPosition() == relative
                        && boundary.role() == role).count();
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> context(boolean relu) {
        Shape shape = Shape.of(2, 2);
        var descriptor = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var nodes = new ArrayList<CompiledNode>();
        nodes.add(new CompiledNode(new NodeId(0),
                new Operation(MatmulKind.MATMUL, NoOperationAttrs.INSTANCE),
                List.of(new ValueId(0), new ValueId(1)), List.of(new ValueId(2))));
        nodes.add(new CompiledNode(new NodeId(1),
                new Operation(MatmulKind.MATMUL, NoOperationAttrs.INSTANCE),
                List.of(new ValueId(2), new ValueId(1)), List.of(new ValueId(3))));
        if (relu) nodes.add(new CompiledNode(new NodeId(2),
                new Operation(UnaryElementwiseKind.RELU, NoOperationAttrs.INSTANCE),
                List.of(new ValueId(3)), List.of(new ValueId(4))));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        var values = new ArrayList<GraphValue>();
        var requirements = new ArrayList<LogicalMemoryRequirement>();
        for (int i = 0; i < (relu ? 5 : 4); i++) {
            var id = new ValueId(i);
            values.add(new GraphValue(id, descriptor));
            boolean produced = nodes.stream().anyMatch(node -> node.outputs().contains(id));
            boolean consumed = nodes.stream().anyMatch(node -> node.inputs().contains(id));
            requirements.add(new LogicalMemoryRequirement(id, descriptor,
                    produced ? Optional.of(partition) : Optional.empty(),
                    consumed ? List.of(partition) : List.of(),
                    nodes.getLast().outputs().contains(id)));
        }
        return new PrepareContext<>(partition, nodes, values,
                requirements, Map.of(), new CpuPartitionAnalysisInputs(false, List.of(),
                new PortableExecutionConfig(ComputePreference.SCALAR, 1, 1, 1)));
    }
}

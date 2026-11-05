package io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuClassFileKernelGenerator;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMatmulLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
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
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Exercises actual generated complete realizations, without a local topology search or natives. */
final class CpuFiniteCompleteRealizationTest {
    @Test void allIndependentMatmulUnitChoicesExecuteWithoutALocalPhase() throws Throwable {
        for (var type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            var context = chain(type, 1);
            var preparer = new CpuPartitionPreparer();
            assertTrue(preparer.localTuningBatch(context).isEmpty());
            var complete = preparer.completePlanCandidates(context);
            assertTrue(complete.complete());
            assertEquals(16, complete.identities().size(), "four implemented forms per ordinary unit");
            var observed = new HashSet<List<String>>();
            for (var identity : complete.identities()) {
                var plan = preparer.analyzeSelected(context,
                        new CpuPartitionPreparer.SelectedCompletePlan(5, identity, List.of(0))).plan();
                observed.add(plan.units().stream().map(unit -> unit.portablePlan().specialization()
                        .matmulIr().orElseThrow().realization().name()).toList());
                var storage = new HashMap<ValueId, Object>();
                storage.put(new ValueId(0), values(type, 2, 3));
                storage.put(new ValueId(1), values(type, 1, 2, 3, 4));
                storage.put(new ValueId(2), values(type, 0, 0));
                storage.put(new ValueId(3), values(type, 5, 6));
                storage.put(new ValueId(4), values(type, 0));
                for (var unit : plan.units()) execute(unit, storage);
                assertEquals(151, first(storage.get(new ValueId(4))), 0.0);
            }
            assertEquals(16, observed.size(), "no overwritten unit configuration or duplicate physical choice");
        }
    }

    @Test void completeCombinationLimitRejectsBeforeAnyPartialDomainIsReturned() {
        var preparer = new CpuPartitionPreparer();
        assertThrows(IllegalArgumentException.class, () -> preparer.completePlanCandidates(chain(DataType.FLOAT32, 128))));
        assertEquals(16, preparer.completePlanCandidates(chain(DataType.FLOAT32, 1)).identities().size());
    }

    @Test void fusedEpilogueChoicesRemainCompletePhaseChoicesAndActuallyExecute() throws Throwable {
        for (var type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            var source = CpuMatmulLoweringTest.linear(type, 1, 2, 1, UnaryElementwiseKind.TANH, 1);
            var context = new PrepareContext<>(source.partitionDag(), source.values(), source.memoryRequirements(),
                    source.constants(), inputs(1), source.scanActivityPlans());
            var preparer = new CpuPartitionPreparer();
            assertTrue(preparer.localTuningBatch(context).isEmpty(), "fusion must not move into Phase 1");
            var complete = preparer.completePlanCandidates(context);
            assertTrue(complete.complete());
            for (var topology : complete.identities().stream().map(identity -> identity.topology()).distinct().toList())
                assertTrue(complete.identities().stream().anyMatch(identity -> identity.topology().equals(topology)
                        && identity.units().isEmpty() && identity.nativeAssignment().isEmpty()
                        && identity.openBlasAssignment().isEmpty()), "ordinary variant is retained");
            var observed = new HashSet<String>();
            for (var identity : complete.identities()) {
                if (identity.topology().topology().units().size() != 1
                        || !identity.topology().materializations().isEmpty()) continue;
                var plan = preparer.analyzeSelected(context,
                        new CpuPartitionPreparer.SelectedCompletePlan(5, identity, List.of(0))).plan();
                var unit = plan.units().getFirst();
                observed.add(unit.portablePlan().specialization().matmulIr().orElseThrow().realization().name());
                var storage = new HashMap<ValueId, Object>();
                storage.put(new ValueId(0), values(type, 2, 3));
                storage.put(new ValueId(1), values(type, 4, 5));
                storage.put(new ValueId(3), values(type, -23));
                storage.put(new ValueId(5), values(type, 42));
                execute(unit, storage);
                assertEquals(0L, Double.doubleToRawLongBits(first(storage.get(new ValueId(5)))));
            }
            assertEquals(4, observed.size(), "each supported fused fixed form executes, including tails");
        }
    }

    private static void execute(CpuPartitionPreparationPlan.ExecutionUnitPlan unit,
            Map<ValueId, Object> storage) throws Throwable {
        var route = unit.portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        var generated = generator.defineClassBytes(route.specialization(), route.kernelIr(),
                generator.generateClassBytes(route.specialization(), route.kernelIr()));
        var arguments = new ArrayList<Object>();
        for (var value : unit.boundaryValues()) arguments.add(segment(storage.get(value)));
        var geometry = unit.matmulGeometry().orElseThrow();
        arguments.add(route.specialization().matmulIr().orElseThrow().epilogue().hasBias()
                ? geometry.pack(0, 0, 0, 0, unit.accessBindings().get(2).effectiveStrides().getLast())
                : geometry.pack(0, 0, 0));
        arguments.add(0L); arguments.add(unit.elementCount());
        generated.entryPoint().invokeWithArguments(arguments);
    }

    private static Object values(DataType type, int... values) {
        if (type == DataType.FLOAT32) {
            float[] result = new float[values.length];
            for (int index = 0; index < values.length; index++) result[index] = values[index];
            return result;
        }
        double[] result = new double[values.length];
        for (int index = 0; index < values.length; index++) result[index] = values[index];
        return result;
    }

    private static MemorySegment segment(Object array) {
        return array instanceof float[] values ? MemorySegment.ofArray(values)
                : MemorySegment.ofArray((double[]) array);
    }

    private static double first(Object array) {
        return array instanceof float[] values ? values[0] : ((double[]) array)[0];
    }

    private static CpuPartitionAnalysisInputs inputs(int parallelism) {
        return new CpuPartitionAnalysisInputs(false, List.of(),
                new CpuPartitionAnalysisInputs.PortableExecutionConfig(
                        CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR,
                        parallelism, parallelism, 1),
                new CpuPartitionAnalysisInputs.MaterializationPolicy(true, 0, 1, 3, 1, 1,
                        Long.MAX_VALUE, 0, 0));
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> chain(DataType type, int parallelism) {
        var nodes = List.of(new CompiledNode(new NodeId(0), new Operation(MatmulKind.MATMUL,
                        NoOperationAttrs.INSTANCE), List.of(new ValueId(0), new ValueId(1)), List.of(new ValueId(2))),
                new CompiledNode(new NodeId(1), new Operation(MatmulKind.MATMUL,
                        NoOperationAttrs.INSTANCE), List.of(new ValueId(2), new ValueId(3)), List.of(new ValueId(4))));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID, nodes.stream().map(CompiledNode::id).toList());
        long rows = parallelism == 1 ? 1 : 2L * parallelism;
        var shapes = List.of(Shape.of(rows, 2), Shape.of(2, 2), Shape.of(rows, 2),
                Shape.of(2, 1), Shape.of(rows, 1));
        var values = new ArrayList<GraphValue>();
        var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int index = 0; index < shapes.size(); index++) {
            var id = new ValueId(index);
            var descriptor = new TensorDescriptor(type, shapes.get(index),
                    Optional.of(LayoutDescriptor.contiguous(shapes.get(index))), false);
            values.add(new GraphValue(id, descriptor));
            boolean produced = index == 2 || index == 4;
            memory.add(new LogicalMemoryRequirement(id, descriptor, produced ? Optional.of(partition) : Optional.empty(),
                    index != 4 ? List.of(partition) : List.of(), index == 4));
        }
        return new PrepareContext<>(partition, nodes, values, memory, Map.of(), inputs(parallelism));
    }
}

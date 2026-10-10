package io.github.pho001.synaptik.testing.conformance;

import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr.Realization;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Portable MATMUL realization and prepared compute strategy must describe one work domain. */
final class CpuMatmulStrategyConformanceTest {
    @Test void eligibleDirectAndTiledFormsHonorColdPreferenceForFourTypes() {
        for (var type : List.of(DataType.FLOAT32, DataType.FLOAT64, DataType.INT32,
                DataType.INT64)) {
            check(type, 2, 3, 4, ComputePreference.VECTOR_IF_ELIGIBLE,
                    Realization.DIRECT_SCALAR, 8, 0);
            check(type, 2, 63, 128, ComputePreference.SCALAR,
                    Realization.DIRECT_SCALAR, 256, 0);
            check(type, 2, 63, 128, ComputePreference.VECTOR_IF_ELIGIBLE,
                    Realization.DIRECT_N_VECTOR, 2, bits(type));
            check(type, 32, 127, 256, ComputePreference.SCALAR,
                    Realization.TILED_SCALAR_2X2, 2_048, 0);
            int lanes = bits(type) / type.bitWidth();
            check(type, 32, 127, 256, ComputePreference.VECTOR_IF_ELIGIBLE,
                    Realization.TILED_N_VECTOR_2X2,
                    16L * ((256L + 2L * lanes - 1) / (2L * lanes)), bits(type));
        }
    }

    @Test void nonUnitRightStrideKeepsScalarTiledFormDespiteVectorPreference() {
        var context = context(DataType.INT32, 32, 127, 256,
                ComputePreference.VECTOR_IF_ELIGIBLE, true);
        var unit = new CpuPartitionPreparer().analyze(context).plan().units().getFirst();
        assertEquals(Realization.TILED_SCALAR_2X2,
                unit.portablePlan().specialization().matmulIr().orElseThrow().realization());
        assertEquals(0, unit.portablePlan().specialization().vectorSpeciesBitSize());
    }

    @Test void parallelOrchestrationPreservesSelectedComputeForm() {
        for (var preference : ComputePreference.values()) {
            var base = context(DataType.INT32, 2, 63, 128, preference, false);
            var inputs = new CpuPartitionAnalysisInputs(false, List.of(),
                    new PortableExecutionConfig(preference, 4, 4, 1));
            var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                    base.memoryRequirements(), base.constants(), inputs);
            var unit = new CpuPartitionPreparer().analyze(context).plan().units().getFirst();
            assertEquals(preference == ComputePreference.SCALAR
                    ? ExecutionStrategy.PARALLEL_SCALAR : ExecutionStrategy.PARALLEL_VECTOR,
                    unit.executionStrategy());
            assertEquals(preference == ComputePreference.SCALAR
                    ? Realization.DIRECT_SCALAR : Realization.DIRECT_N_VECTOR,
                    unit.portablePlan().specialization().matmulIr().orElseThrow().realization());
        }
    }

    private static void check(DataType type, int m, int k, int n,
            ComputePreference preference, Realization expected, long work, int speciesBits) {
        var unit = new CpuPartitionPreparer().analyze(context(type, m, k, n, preference, false))
                .plan().units().getFirst();
        var specialization = unit.portablePlan().specialization();
        assertEquals(expected, specialization.matmulIr().orElseThrow().realization(), type.toString());
        assertEquals(speciesBits, specialization.vectorSpeciesBitSize());
        assertEquals(speciesBits,
                specialization.matmulIr().orElseThrow().preferredSpeciesBitSize());
        assertEquals(work, unit.elementCount());
    }

    private static int bits(DataType type) {
        return switch (type) {
            case FLOAT32 -> jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.vectorBitSize();
            case FLOAT64 -> jdk.incubator.vector.DoubleVector.SPECIES_PREFERRED.vectorBitSize();
            case INT32 -> jdk.incubator.vector.IntVector.SPECIES_PREFERRED.vectorBitSize();
            case INT64 -> jdk.incubator.vector.LongVector.SPECIES_PREFERRED.vectorBitSize();
            default -> throw new AssertionError(type);
        };
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> context(DataType type, int m,
            int k, int n, ComputePreference preference, boolean stridedRight) {
        Shape a = Shape.of(m, k), b = Shape.of(k, n), c = Shape.of(m, n);
        var rightLayout = stridedRight
                ? LayoutDescriptor.of(b, new long[] {2L * n, 2}, 0, true)
                : LayoutDescriptor.contiguous(b);
        var descriptors = List.of(new TensorDescriptor(type, a,
                        Optional.of(LayoutDescriptor.contiguous(a)), false),
                new TensorDescriptor(type, b, Optional.of(rightLayout), false),
                new TensorDescriptor(type, c,
                        Optional.of(LayoutDescriptor.contiguous(c)), false));
        var node = new CompiledNode(new NodeId(0),
                new Operation(MatmulKind.MATMUL, NoOperationAttrs.INSTANCE),
                List.of(new ValueId(0), new ValueId(1)), List.of(new ValueId(2)));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                List.of(node.id()));
        var values = List.of(new GraphValue(new ValueId(0), descriptors.get(0)),
                new GraphValue(new ValueId(1), descriptors.get(1)),
                new GraphValue(new ValueId(2), descriptors.get(2)));
        var requirements = List.of(
                new LogicalMemoryRequirement(new ValueId(0), descriptors.get(0),
                        Optional.empty(), List.of(partition), false),
                new LogicalMemoryRequirement(new ValueId(1), descriptors.get(1),
                        Optional.empty(), List.of(partition), false),
                new LogicalMemoryRequirement(new ValueId(2), descriptors.get(2),
                        Optional.of(partition), List.of(), true));
        var inputs = new CpuPartitionAnalysisInputs(false, List.of(),
                new PortableExecutionConfig(preference, 1, 1, 1));
        return new PrepareContext<>(partition, List.of(node), values, requirements,
                Map.of(), inputs);
    }
}

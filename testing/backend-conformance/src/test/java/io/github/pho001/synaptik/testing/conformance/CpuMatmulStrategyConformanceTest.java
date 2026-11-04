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
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Portable MATMUL realization and prepared compute strategy must describe one work domain. */
final class CpuMatmulStrategyConformanceTest {
    @Test void ordinaryTanhUsesTheSameVectorAndParallelWorkDomainsWithOrderedBias() {
        for (var type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            for (boolean tiled : List.of(false, true)) for (int biasOrder : new int[] {-1, 0, 1}) {
                for (int threads : new int[] {1, 4}) {
                    int m = tiled ? 33 : 2, k = tiled ? 127 : 63, n = tiled ? 257 : 129;
                    var base = context(type, m, k, n, ComputePreference.VECTOR_IF_ELIGIBLE, false);
                    var nodes = new ArrayList<>(base.nodes());
                    var values = new ArrayList<>(base.values());
                    ValueId current = new ValueId(2);
                    if (biasOrder >= 0) {
                        var bias = new TensorDescriptor(type, Shape.of(n),
                                Optional.of(LayoutDescriptor.contiguous(Shape.of(n))), false);
                        values.add(new GraphValue(new ValueId(3), bias));
                        values.add(new GraphValue(new ValueId(4), values.get(2).descriptor()));
                        nodes.add(new CompiledNode(new NodeId(1), new Operation(
                                io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind.ADD,
                                NoOperationAttrs.INSTANCE), biasOrder == 0
                                    ? List.of(current, new ValueId(3)) : List.of(new ValueId(3), current),
                                List.of(new ValueId(4))));
                        current = new ValueId(4);
                    }
                    ValueId output = new ValueId(values.size());
                    values.add(new GraphValue(output, values.get(2).descriptor()));
                    nodes.add(new CompiledNode(new NodeId(nodes.size()), new Operation(
                            io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind.TANH,
                            NoOperationAttrs.INSTANCE), List.of(current), List.of(output)));
                    var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                            nodes.stream().map(CompiledNode::id).toList());
                    var memory = values.stream().map(value -> new LogicalMemoryRequirement(value.id(),
                            value.descriptor(), nodes.stream().anyMatch(node -> node.outputs().contains(value.id()))
                                ? Optional.of(partition) : Optional.empty(),
                            value.id().equals(output) ? List.of() : List.of(partition),
                            value.id().equals(output))).toList();
                    var inputs = new CpuPartitionAnalysisInputs(false, List.of(),
                            new PortableExecutionConfig(ComputePreference.VECTOR_IF_ELIGIBLE, threads, threads, 1));
                    var plan = new CpuPartitionPreparer().analyze(new PrepareContext<>(partition,
                            nodes, values, memory, Map.of(), inputs)).plan();
                    assertEquals(1, plan.units().size());
                    var unit = plan.units().getFirst();
                    var ir = unit.portablePlan().specialization().matmulIr().orElseThrow();
                    assertEquals(tiled ? Realization.TILED_N_VECTOR_2X2 : Realization.DIRECT_N_VECTOR,
                            ir.realization());
                    assertEquals(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr
                            .Epilogue.Terminal.TANH, ir.epilogue().terminal());
                    assertEquals(biasOrder < 0 ? "NONE" : biasOrder == 0 ? "MATMUL_LEFT" : "MATMUL_RIGHT",
                            ir.epilogue().addInputOrder().name());
                    assertEquals(threads == 1 ? ExecutionStrategy.VECTOR : ExecutionStrategy.PARALLEL_VECTOR,
                            unit.executionStrategy());
                    int lanes = bits(type) / type.bitWidth();
                    assertEquals(tiled ? ((m + 1L) / 2) * ((n + 2L * lanes - 1) / (2L * lanes)) : m,
                            unit.elementCount());
                }
            }
        }
    }

    @Test void privateTransposesExecuteTheSameProductAcrossTypesPreferencesAndWorkerCounts() {
        for (var type : List.of(DataType.FLOAT32, DataType.FLOAT64, DataType.INT32, DataType.INT64,
                DataType.BFLOAT16, DataType.FLOAT16, DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2)) {
            for (var preference : ComputePreference.values()) for (int threads : new int[] {1, 4}) {
                for (boolean right : List.of(false, true)) {
                    var context = transposedContext(type, preference, threads, right);
                    var data = new java.util.HashMap<ValueId, java.lang.foreign.MemorySegment>();
                    for (var value : context.values()) {
                        if (value.id().equals(new ValueId(2)) || value.id().equals(new ValueId(right ? 1 : 0))) continue;
                        int count = Math.toIntExact(value.descriptor().shape().knownElementCount().orElseThrow());
                        var segment = java.lang.foreign.MemorySegment.ofArray(new byte[count * type.byteWidth()]);
                        for (int i = 0; i < count; i++) write(type, segment, i, i % 5 - 2);
                        data.put(value.id(), segment);
                    }
                    byte[] result = CpuLowPrecisionConformanceTest.execute(context, data).get(new ValueId(2));
                    var output = java.lang.foreign.MemorySegment.ofArray(result);
                    var left = data.get(new ValueId(right ? 0 : 3));
                    var rhs = data.get(new ValueId(right ? 3 : 1));
                    for (int row = 0; row < 2; row++) for (int column = 0; column < 128; column++) {
                        float sum = 0;
                        for (int k = 0; k < 63; k++) sum += (float) read(type, left, right ? row * 63 + k : k * 2 + row)
                                * (float) read(type, rhs, right ? column * 63 + k : k * 128 + column);
                        double expected = LowPrecisionValues.TYPES.contains(type)
                                ? LowPrecisionValues.decode(type, LowPrecisionValues.encode(type, sum)) : sum;
                        assertEquals(expected, read(type, output, row * 128 + column),
                                type + " right=" + right + " preference=" + preference + " threads=" + threads);
                    }
                }
            }
        }
    }

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
            var context = new PrepareContext<>(base.partitionDag(), base.values(),
                    base.memoryRequirements(), base.constants(), inputs, base.scanActivityPlans());
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

    private static void write(DataType type, java.lang.foreign.MemorySegment segment, int index, int value) {
        switch (type) {
            case FLOAT32 -> segment.set(java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,index*4L,(float)value);
            case FLOAT64 -> segment.set(java.lang.foreign.ValueLayout.JAVA_DOUBLE_UNALIGNED,index*8L,(double)value);
            case INT32 -> segment.set(java.lang.foreign.ValueLayout.JAVA_INT_UNALIGNED,index*4L,value);
            case INT64 -> segment.set(java.lang.foreign.ValueLayout.JAVA_LONG_UNALIGNED,index*8L,(long)value);
            default -> {
                int raw=LowPrecisionValues.encode(type,value);
                if(type.byteWidth()==1)segment.set(java.lang.foreign.ValueLayout.JAVA_BYTE,index,(byte)raw);
                else segment.set(java.lang.foreign.ValueLayout.JAVA_SHORT_UNALIGNED,index*2L,(short)raw);
            }
        }
    }

    private static double read(DataType type, java.lang.foreign.MemorySegment segment, int index) {
        return switch (type) {
            case FLOAT32 -> segment.get(java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,index*4L);
            case FLOAT64 -> segment.get(java.lang.foreign.ValueLayout.JAVA_DOUBLE_UNALIGNED,index*8L);
            case INT32 -> segment.get(java.lang.foreign.ValueLayout.JAVA_INT_UNALIGNED,index*4L);
            case INT64 -> segment.get(java.lang.foreign.ValueLayout.JAVA_LONG_UNALIGNED,index*8L);
            default -> LowPrecisionValues.decode(type,type.byteWidth()==1
                    ?segment.get(java.lang.foreign.ValueLayout.JAVA_BYTE,index)&255
                    :segment.get(java.lang.foreign.ValueLayout.JAVA_SHORT_UNALIGNED,index*2L)&65535);
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

    /**
     * Replaces one ordinary matrix input with a private permutation of a canonical backing.
     *
     * @param type common numeric matrix type
     * @param preference requested existing scalar/vector policy
     * @param threads positive available worker count
     * @param right whether to transpose the right rather than left input
     * @return complete immutable preparation projection with only the product published
     */
    private static PrepareContext<CpuPartitionAnalysisInputs> transposedContext(DataType type,
            ComputePreference preference, int threads, boolean right) {
        var base = context(type, 2, 63, 128, preference, false);
        int target = right ? 1 : 0;
        Shape sourceShape = right ? Shape.of(128, 63) : Shape.of(63, 2);
        Shape viewShape = base.values().get(target).descriptor().shape();
        var source = new TensorDescriptor(type, sourceShape,
                Optional.of(LayoutDescriptor.contiguous(sourceShape)), false);
        var view = new TensorDescriptor(type, viewShape, Optional.of(LayoutDescriptor.of(viewShape,
                new long[] {1, right ? 63 : 2}, 0, true)), false);
        var nodes = List.of(new CompiledNode(new NodeId(1),
                        new Operation(AxisTransformKind.PERMUTE, new PermutationAttrs(List.of(1, 0))),
                        List.of(new ValueId(3)), List.of(new ValueId(target))), base.nodes().getFirst());
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        var values = new ArrayList<>(base.values());
        values.set(target, new GraphValue(new ValueId(target), view));
        values.add(new GraphValue(new ValueId(3), source));
        var memory = values.stream().map(value -> new LogicalMemoryRequirement(value.id(),
                value.descriptor(), value.id().equals(new ValueId(target)) || value.id().equals(new ValueId(2))
                        ? Optional.of(partition) : Optional.empty(),
                value.id().equals(new ValueId(2)) ? List.of() : List.of(partition),
                value.id().equals(new ValueId(2)))).toList();
        var inputs = new CpuPartitionAnalysisInputs(false, List.of(),
                new PortableExecutionConfig(preference, threads, threads, 1));
        return new PrepareContext<>(partition, nodes, values, memory, Map.of(), inputs);
    }
}

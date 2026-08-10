package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ClampRangeAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.*;
import org.junit.jupiter.api.Test;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuNativeBuffer;
import io.github.pho001.synaptik.runtime.run.*;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

class CpuPointwisePartitionLoweringTest {
    @Test void lowersOneThroughEightOccurrencesWithDerivedBoundaries() {
        for (int count = 1; count <= 8; count++) {
            int expectedCount = count;
            var lowered = new CpuPartitionLowering().lower(chain(count));
            assertAll(
                    () -> assertEquals(expectedCount, lowered.kernelIr().instructions().size()),
                    () -> assertEquals(2, lowered.boundaryValues().size()),
                    () -> assertEquals(expectedCount - 1, lowered.virtualValues().size()),
                    () -> assertEquals(expectedCount == 1 ? 0 : expectedCount - 1,
                            lowered.kernelIr().values().stream()
                                    .filter(value -> value.kind() == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr.Value.Kind.VIRTUAL)
                                    .count()));
        }
    }

    @Test void rejectsOverBudgetAndDisconnectedPartitionsBeforeDeclarations() {
        assertThrows(IllegalArgumentException.class, () -> new CpuPartitionLowering().lower(chain(9)));
        var context = chain(2);
        var nodes = new ArrayList<>(context.nodes());
        CompiledNode second = nodes.get(1);
        nodes.set(1, new CompiledNode(second.id(), second.operation(),
                List.of(new ValueId(0)), second.outputs()));
        var disconnected = new PrepareContext<>(context.partition(), nodes, context.values(),
                context.memoryRequirements(), Map.of(), context.backendInputs());
        assertThrows(IllegalArgumentException.class,
                () -> new CpuPartitionLowering().lower(disconnected));
    }

    @Test void preparationDerivesDefaultSegmentPatternAndScalarFallback() {
        var plan = new CpuPartitionPreparer().analyze(chain(8)).plan();
        var executable = CpuPartitionFinalizerTest.finalizeExecutable(
                new CpuPartitionPreparer().analyze(chain(8)), Optional.empty());
        assertAll(
                () -> assertEquals(2, plan.bufferDeclarations().size()),
                () -> assertEquals(List.of(
                        io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT,
                        io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT),
                        plan.carrierPattern()),
                () -> assertEquals("scalar", plan.executionStrategy().toString()),
                () -> assertEquals(List.of(DataType.INT32, DataType.INT32),
                        plan.units().getFirst().portablePlan().specialization().boundaryDataTypes()),
                () -> assertEquals(2, executable.bufferSelectionCount()),
                () -> assertNotNull(executable.artifact().hiddenClass()));
    }

    @Test void finalizedEightInstructionExecutableRunsThroughOneBoundInvocation() {
        var analysis = new CpuPartitionPreparer().analyze(chain(8));
        var executable = CpuPartitionFinalizerTest.finalizeExecutable(analysis, Optional.empty());
        var input = CpuNativeBuffer.allocate(DataType.INT32, 20, 4);
        var output = CpuNativeBuffer.allocate(DataType.INT32, 20, 4);
        var state = new RunState(executable.memoryPlan(), List.of(
                List.of(new BufferRepresentationBinding(input, RunResourceOwnership.RUN_OWNED)),
                List.of(new BufferRepresentationBinding(output, RunResourceOwnership.RUN_OWNED))), List.of());
        var layout = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.nativeOrder());
        try {
            for (int i = 0; i < 5; i++) input.segment().set(layout, i * 4L, Integer.MAX_VALUE - i);
            executable.bind(state).execute();
            for (int i = 0; i < 5; i++) assertEquals(Integer.MAX_VALUE - i + 36,
                    output.segment().get(layout, i * 4L));
        } finally { state.close(); }
    }

    @Test void coldBindingRejectsNonCanonicalWhereCondition() {
        var analysis = new CpuPartitionPreparer().analyze(where());
        var executable = CpuPartitionFinalizerTest.finalizeExecutable(analysis, Optional.empty());
        var condition = CpuNativeBuffer.allocate(DataType.BOOL, 5, 1);
        var whenTrue = CpuNativeBuffer.allocate(DataType.FLOAT32, 20, 4);
        var whenFalse = CpuNativeBuffer.allocate(DataType.FLOAT32, 20, 4);
        var output = CpuNativeBuffer.allocate(DataType.FLOAT32, 20, 4);
        condition.segment().set(ValueLayout.JAVA_BYTE, 0, (byte) 2);
        var state = new RunState(executable.memoryPlan(), List.of(condition, whenTrue, whenFalse, output)
                .stream().map(buffer -> List.of(new BufferRepresentationBinding(buffer,
                        RunResourceOwnership.RUN_OWNED))).toList(), List.of());
        try {
            assertThrows(IllegalArgumentException.class, () -> executable.bind(state));
        } finally { state.close(); }
    }

    @Test void coldBindingRejectsNonCanonicalLogicalInput() {
        var context = logical();
        var analysis = new CpuPartitionPreparer().analyze(context);
        var executable = CpuPartitionFinalizerTest.finalizeExecutable(analysis, Optional.empty());
        var left = CpuNativeBuffer.allocate(DataType.BOOL, 5, 1);
        var right = CpuNativeBuffer.allocate(DataType.BOOL, 5, 1);
        var output = CpuNativeBuffer.allocate(DataType.BOOL, 5, 1);
        left.segment().set(ValueLayout.JAVA_BYTE, 0, (byte) 2);
        var state = new RunState(executable.memoryPlan(), List.of(left, right, output).stream()
                .map(buffer -> List.of(new BufferRepresentationBinding(buffer,
                        RunResourceOwnership.RUN_OWNED))).toList(), List.of());
        try {
            assertThrows(IllegalArgumentException.class, () -> executable.bind(state));
        } finally { state.close(); }
    }

    @Test void lowersBroadcastDivisionAndEveryScalarPowerPlanAsOrdinaryInstructions() {
        Shape output = Shape.of(2, 3);
        var f32Left = descriptor(DataType.FLOAT32, Shape.of(2, 1));
        var f32Right = descriptor(DataType.FLOAT32, Shape.of(3));
        var f32Output = descriptor(DataType.FLOAT32, output);
        var division = new CpuPartitionLowering().lower(single(
                new Operation(BinaryArithmeticKind.DIV, NoOperationAttrs.INSTANCE),
                List.of(f32Left, f32Right), f32Output, CpuPartitionAnalysisInputs.DEFAULT));
        assertAll(
                () -> assertEquals(CpuPointwiseOpcode.DIV,
                        division.kernelIr().instructions().getFirst().opcode()),
                () -> assertEquals(3, division.boundaryValues().size()),
                () -> assertEquals(CpuKernelIr.PowerRealization.POSITIVE_ONE,
                        power(ScalarValue.float64(-0.0d)).powerRealization()),
                () -> assertEquals(CpuKernelIr.PowerRealization.IDENTITY,
                        power(ScalarValue.float64(1.0d)).powerRealization()),
                () -> assertEquals(CpuKernelIr.PowerRealization.SQUARE,
                        power(ScalarValue.float32(2.0f)).powerRealization()),
                () -> assertEquals(CpuKernelIr.PowerRealization.RECIPROCAL,
                        power(ScalarValue.float32(-1.0f)).powerRealization()),
                () -> assertEquals(CpuKernelIr.PowerRealization.DIRECT,
                        power(ScalarValue.float64(0.5d)).powerRealization()));
    }

    @Test void directPowerForcesScalarComputeWhileSpecialPowerRetainsVectorEligibility() {
        int count = jdk.incubator.vector.DoubleVector.SPECIES_PREFERRED.length() * 2;
        var config = new CpuPartitionAnalysisInputs.PortableExecutionConfig(
                CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE,
                4, 2, 1);
        var inputs = new CpuPartitionAnalysisInputs(true, List.of(), config);
        var direct = new CpuPartitionPreparer().analyze(singlePower(ScalarValue.float64(0.5d),
                Shape.of(count), inputs)).plan();
        var square = new CpuPartitionPreparer().analyze(singlePower(ScalarValue.float64(2.0d),
                Shape.of(count), inputs)).plan();
        assertAll(
                () -> assertEquals("parallel-scalar", direct.executionStrategy().toString()),
                () -> assertEquals(0, direct.vectorSpeciesBitSize()),
                () -> assertEquals("parallel-vector", square.executionStrategy().toString()),
                () -> assertEquals(List.of(CpuKernelIr.PowerRealization.DIRECT),
                        direct.units().getFirst().portablePlan().specialization()
                                .scalarPowerRealizations()),
                () -> assertTrue(direct.loweringManifest().contains("power=[DIRECT]")),
                () -> assertTrue(square.loweringManifest().contains("power=[SQUARE]")));
    }

    @Test void lowersExtremaClampTensorPowerAndLogicalOccurrencesWithoutDecomposition() {
        Shape shape = Shape.of(3);
        var f64 = descriptor(DataType.FLOAT64, shape);
        var bool = descriptor(DataType.BOOL, shape);
        var clamp = new CpuPartitionLowering().lower(single(new Operation(
                ScalarElementwiseKind.CLAMP, new ClampRangeAttrs(
                        ScalarValue.float64(-0.0d), ScalarValue.float64(+0.0d))),
                List.of(f64), f64, CpuPartitionAnalysisInputs.DEFAULT));
        var tensorPower = new CpuPartitionPreparer().analyze(single(new Operation(
                BinaryArithmeticKind.POW, NoOperationAttrs.INSTANCE), List.of(f64, f64), f64,
                vectorInputs())).plan();
        var logical = new CpuPartitionPreparer().analyze(single(new Operation(
                BooleanLogicalKind.AND, NoOperationAttrs.INSTANCE), List.of(bool, bool), bool,
                vectorInputs())).plan();
        assertAll(
                () -> assertEquals(1, clamp.kernelIr().instructions().size()),
                () -> assertEquals(CpuPointwiseOpcode.SCALAR_CLAMP,
                        clamp.kernelIr().instructions().getFirst().opcode()),
                () -> assertEquals(Double.doubleToRawLongBits(-0.0d), clamp.kernelIr()
                        .instructions().getFirst().clampImmediate().lower().bits()),
                () -> assertEquals(Double.doubleToRawLongBits(+0.0d), clamp.kernelIr()
                        .instructions().getFirst().clampImmediate().upper().bits()),
                () -> assertEquals("parallel-scalar", tensorPower.executionStrategy().toString()),
                () -> assertEquals("parallel-scalar", logical.executionStrategy().toString()),
                () -> assertEquals(0, tensorPower.vectorSpeciesBitSize()),
                () -> assertEquals(0, logical.vectorSpeciesBitSize()));
    }

    @Test void lowersEveryUnaryKindOnceAndSelectsTheDeclaredComputeEligibility() {
        int count = jdk.incubator.vector.DoubleVector.SPECIES_PREFERRED.length() * 2;
        Shape shape = Shape.of(count);
        var inputs = vectorInputs();
        for (UnaryElementwiseKind kind : UnaryElementwiseKind.values()) {
            CpuPointwiseOpcode expected = kind == UnaryElementwiseKind.GELU
                    ? CpuPointwiseOpcode.GELU_EXACT : CpuPointwiseOpcode.valueOf(kind.name());
            for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32)) {
                var descriptor = descriptor(type, shape);
                var plan = new CpuPartitionPreparer().analyze(single(new Operation(kind,
                        NoOperationAttrs.INSTANCE), List.of(descriptor), descriptor, inputs)).plan();
                assertAll(kind + " " + type,
                        () -> assertEquals(1, plan.units().getFirst().portablePlan().kernelIr()
                                .instructions().size()),
                        () -> assertEquals(expected, plan.units().getFirst().portablePlan().kernelIr()
                                .instructions().getFirst().opcode()),
                        () -> assertEquals(expected.vectorEligible()
                                        ? "parallel-vector" : "parallel-scalar",
                                plan.executionStrategy().toString()));
            }
        }
    }

    private static CpuPartitionAnalysisInputs vectorInputs() {
        return new CpuPartitionAnalysisInputs(false, List.of(),
                new CpuPartitionAnalysisInputs.PortableExecutionConfig(
                        CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference
                                .VECTOR_IF_ELIGIBLE, 2, 2, 1));
    }

    static PrepareContext<CpuPartitionAnalysisInputs> chain(int count) {
        Shape shape = Shape.of(5);
        var descriptor = new TensorDescriptor(DataType.INT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var nodes = new ArrayList<CompiledNode>();
        ValueId input = new ValueId(0);
        ValueId previous = input;
        for (int i = 0; i < count; i++) {
            ValueId output = new ValueId(i + 1L);
            nodes.add(new CompiledNode(new NodeId(i), new Operation(ScalarElementwiseKind.ADD,
                    new ScalarValueAttrs(ScalarValue.int32(i + 1))), List.of(previous), List.of(output)));
            previous = output;
        }
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        var values = new ArrayList<GraphValue>();
        var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int i = 0; i <= count; i++) {
            ValueId id = new ValueId(i);
            values.add(new GraphValue(id, descriptor));
            memory.add(new LogicalMemoryRequirement(id, descriptor,
                    i == 0 ? Optional.empty() : Optional.of(partition),
                    i == count ? List.of() : List.of(partition), i == count));
        }
        return new PrepareContext<>(partition, nodes, values, memory, Map.of(),
                CpuPartitionAnalysisInputs.DEFAULT);
    }

    private static CpuKernelIr.Instruction power(ScalarValue exponent) {
        return new CpuPartitionLowering().lower(singlePower(exponent, Shape.of(3),
                CpuPartitionAnalysisInputs.DEFAULT)).kernelIr().instructions().getFirst();
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> singlePower(ScalarValue exponent,
            Shape shape, CpuPartitionAnalysisInputs inputs) {
        var descriptor = descriptor(exponent.dataType(), shape);
        return single(new Operation(ScalarElementwiseKind.POW, new ScalarValueAttrs(exponent)),
                List.of(descriptor), descriptor, inputs);
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape) {
        return new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> single(Operation operation,
            List<TensorDescriptor> inputs, TensorDescriptor output,
            CpuPartitionAnalysisInputs analysisInputs) {
        var inputIds = java.util.stream.IntStream.range(0, inputs.size())
                .mapToObj(index -> new ValueId(index)).toList();
        ValueId outputId = new ValueId(inputs.size());
        var node = new CompiledNode(new NodeId(0), operation, inputIds, List.of(outputId));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID, List.of(node.id()));
        var values = new ArrayList<GraphValue>();
        var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int index = 0; index < inputs.size(); index++) {
            values.add(new GraphValue(inputIds.get(index), inputs.get(index)));
            memory.add(new LogicalMemoryRequirement(inputIds.get(index), inputs.get(index),
                    Optional.empty(), List.of(partition), false));
        }
        values.add(new GraphValue(outputId, output));
        memory.add(new LogicalMemoryRequirement(outputId, output, Optional.of(partition),
                List.of(), true));
        return new PrepareContext<>(partition, List.of(node), values, memory, Map.of(),
                analysisInputs);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> where() {
        Shape shape = Shape.of(5);
        var bool = new TensorDescriptor(DataType.BOOL, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var f32 = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var ids = List.of(new ValueId(0), new ValueId(1), new ValueId(2), new ValueId(3));
        var node = new CompiledNode(new NodeId(0),
                new Operation(WhereSelectionKind.WHERE, NoOperationAttrs.INSTANCE),
                ids.subList(0, 3), List.of(ids.get(3)));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID, List.of(node.id()));
        var descriptors = List.of(bool, f32, f32, f32);
        var values = new ArrayList<GraphValue>();
        var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int i = 0; i < 4; i++) {
            values.add(new GraphValue(ids.get(i), descriptors.get(i)));
            memory.add(new LogicalMemoryRequirement(ids.get(i), descriptors.get(i),
                    i == 3 ? Optional.of(partition) : Optional.empty(),
                    i == 3 ? List.of() : List.of(partition), i == 3));
        }
        return new PrepareContext<>(partition, List.of(node), values, memory, Map.of(),
                CpuPartitionAnalysisInputs.DEFAULT);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> logical() {
        Shape shape = Shape.of(5);
        var bool = descriptor(DataType.BOOL, shape);
        return single(new Operation(BooleanLogicalKind.OR, NoOperationAttrs.INSTANCE),
                List.of(bool, bool), bool, CpuPartitionAnalysisInputs.DEFAULT);
    }
}

package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAggregateLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuIndexingLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.backend.cpu.internal.reference.CpuScalarReferenceKernel;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.SumToShapeAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Scan;
import io.github.pho001.synaptik.model.tensor.ScanBody;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;
import java.lang.classfile.ClassFile;
import java.lang.classfile.Instruction;
import java.lang.classfile.Opcode;
import java.lang.classfile.instruction.InvokeInstruction;
import java.lang.classfile.instruction.BranchInstruction;
import java.lang.classfile.instruction.LabelTarget;
import java.lang.classfile.constantpool.DynamicConstantPoolEntry;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.classfile.constantpool.MethodHandleEntry;
import java.lang.reflect.AccessFlag;

class CpuAggregateGeneratedKernelTest {
    @Test void controlledTerminalSumSkipsPoisonBeforeLoadsAndKeepsRepresentedRounding()
            throws Throwable {
        var base = CpuAggregateLoweringTest.context(AggregateReductionKind.SUM, DataType.FLOAT32,
                Shape.of(3, 3), NoOperationAttrs.INSTANCE, Shape.scalar());
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false,
                        List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.FLOAT_ARRAY)));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        var row = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanControl.Row(0, 3);
        var shared = io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanControl.Row.shared();
        var facts = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanControl(
                List.of(row, shared), List.of(row, shared));
        float[] input = {1e20f, 1f, -1e20f, Float.NaN, Float.NaN, Float.NaN, 3f, 4f, 5f};
        float[] output = {-19f};
        invokeControlledAggregate(route, facts, input, output,
                plan.aggregateGeometry().orElseThrow().pack(new long[2]), true, true, false, true);
        assertArrayEquals(new float[]{12f}, output,
                "inactive poison is never a factor; active factors retain sequential rounding");
        output[0] = -19f;
        invokeControlledAggregate(route, facts, input, output,
                plan.aggregateGeometry().orElseThrow().pack(new long[2]), false, true, true, true);
        assertArrayEquals(new float[]{-19f}, output, "owner absence dominates the reduction");
    }

    @Test void controlledNonleadingRowOutputSkipsEagerReadsButRetainsOrdinaryExactSum()
            throws Throwable {
        var base = CpuAggregateLoweringTest.context(AggregateReductionKind.SUM, DataType.FLOAT32,
                Shape.of(2, 3, 3), new AxisReductionAttrs(2, false), Shape.of(2, 3));
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false,
                        List.of(CarrierAccess.MEMORY_SEGMENT, CarrierAccess.FLOAT_ARRAY)));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var inputRow = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanControl.Row(1, 3);
        var outputRow = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanControl.Row(1, 1);
        var facts = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanControl(
                List.of(inputRow, outputRow), List.of(inputRow, outputRow));
        float[] input = {1e20f, 1f, -1e20f, Float.NaN, Float.NaN, Float.NaN, 2, 3, 4,
                1e20f, 1f, -1e20f, Float.NaN, Float.NaN, Float.NaN, 5, 6, 7};
        float[] output = {-19, -19, -19, -19, -19, -19};
        invokeControlledAggregate(plan.units().getFirst().portablePlan(), facts,
                MemorySegment.ofArray(input), output,
                plan.aggregateGeometry().orElseThrow().pack(new long[2]), true, true, false, true);
        assertArrayEquals(new float[]{1, -19, 9, 1, -19, 18}, output);
    }

    @Test void controlledSymbolicNonleadingTerminalUsesActualRowsAndCanonicalOrder()
            throws Throwable {
        Shape source = Shape.ofDimensions(new io.github.pho001.synaptik.model.shape.StaticDimension(2),
                new io.github.pho001.synaptik.model.shape.DynamicDimension("B"),
                new io.github.pho001.synaptik.model.shape.StaticDimension(2));
        var route = CpuOrdinaryDynamicBatchTest.aggregateTyped(DataType.FLOAT64,
                new Operation(AggregateReductionKind.SUM,
                        new MultiAxisReductionAttrs(List.of(0, 1), false)), source, Shape.of(2));
        var row = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanControl.Row(1, 2);
        var shared = io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanControl.Row.shared();
        var facts = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanControl(
                List.of(row, shared), List.of(row, shared));
        double[] input = {1e100, 2, Double.NaN, Double.NaN, 1, 3,
                -1e100, 4, Double.NaN, Double.NaN, 7, 5};
        double[] output = {-19, -19};
        invokeControlledAggregate(route, facts, MemorySegment.ofArray(input),
                MemorySegment.ofArray(output), route.batchAggregate().orElseThrow().bind(3)
                        .pack(new long[2]), true, true, false, true);
        assertArrayEquals(new double[]{7, 14}, output);
    }

    @Test void controlledEmptyOccurrenceAndSingletonCopyDoNotInferPresenceFromContributors()
            throws Throwable {
        var emptyBase = CpuAggregateLoweringTest.context(AggregateReductionKind.SUM, DataType.FLOAT32,
                Shape.of(0, 3), NoOperationAttrs.INSTANCE, Shape.scalar());
        var emptyPlan = new CpuPartitionPreparer().analyze(emptyBase).plan();
        var row = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanControl.Row(0, 3);
        var shared = io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanControl.Row.shared();
        var emptyFacts = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanControl(
                List.of(row, shared), List.of(row, shared));
        float[] output = {-19};
        var emptyRoute = emptyPlan.units().getFirst().portablePlan();
        Object emptyInput = emptyRoute.specialization().carrierPattern().getFirst()
                == CarrierAccess.MEMORY_SEGMENT ? MemorySegment.ofArray(new float[0]) : new float[0];
        Object emptyOutput = emptyRoute.specialization().carrierPattern().getLast()
                == CarrierAccess.MEMORY_SEGMENT ? MemorySegment.ofArray(output) : output;
        invokeControlledAggregate(emptyRoute, emptyFacts, emptyInput, emptyOutput,
                emptyPlan.aggregateGeometry().orElseThrow().pack(new long[2]), true);
        assertEquals(0, Float.floatToRawIntBits(output[0]),
                "a present all-valid empty occurrence still completes positive zero");

        var route = CpuOrdinaryDynamicBatchTest.aggregateTyped(DataType.FLOAT64,
                new Operation(AggregateReductionKind.SUM, new SumToShapeAttrs(Shape.of(1, 3))),
                CpuOrdinaryDynamicBatchTest.symbolic(3), Shape.of(1, 3));
        var singletonFacts = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanControl(
                List.of(row, shared), List.of(row, shared));
        double[] input = {-0d, Double.longBitsToDouble(0x7ff8123456789abcL), Double.MIN_VALUE};
        double[] copied = new double[3];
        invokeControlledAggregate(route, singletonFacts, MemorySegment.ofArray(input),
                MemorySegment.ofArray(copied), route.batchAggregate().orElseThrow().bind(1)
                        .pack(new long[2]), true, true);
        assertEquals(-1, MemorySegment.ofArray(input).mismatch(MemorySegment.ofArray(copied)));
    }

    /**
     * Emits and invokes the real controlled ordinary helper with a separate occurrence header.
     * @param route selected ordinary route
     * @param facts proved actual boundary provenance
     * @param input input with the selected carrier
     * @param output output with the selected carrier
     * @param geometry invocation-owned ordinary pack
     * @param present complete-occurrence presence independent of the row-bit count
     * @param active original-row activity bits
     * @throws Throwable if generation, linkage, or generated numerical execution fails
     */
    private static void invokeControlledAggregate(
            io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan route,
            io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanControl facts,
            Object input, Object output, long[] geometry, boolean present, boolean... active)
            throws Throwable {
        invokeControlledAggregateBoundaries(route, facts, List.of(input, output), geometry, present, active);
    }

    /**
     * Invokes a generated aggregate or aggregate-fusion helper with its actual boundary inventory.
     * @param route selected ordinary route
     * @param facts proved actual boundary and removed-producing-value provenance
     * @param boundaries selected carriers in exact ABI order, including output
     * @param geometry invocation-owned ordinary or fusion pack
     * @param present complete-occurrence presence independent of contributor count
     * @param active original-row activity bits
     * @throws Throwable if generation, linkage, or generated numerical execution fails
     */
    static void invokeControlledAggregateBoundaries(
            io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan route,
            io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanControl facts,
            List<?> boundaries, long[] geometry, boolean present, boolean... active) throws Throwable {
        var owner = java.lang.constant.ClassDesc.of(
                "io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.ControlledAggregateFixture");
        var inventory = new ArrayList<CpuClassFileKernelGenerator.GeneratedMethod>();
        byte[] bytes = ClassFile.of().build(owner, builder -> {
            builder.withVersion(ClassFile.JAVA_26_VERSION, 0).withFlags(AccessFlag.FINAL);
            inventory.addAll(CpuClassFileKernelGenerator.emitMethods(builder, owner, "",
                    route.specialization(), route.kernelIr(), facts));
        });
        var lookup = java.lang.invoke.MethodHandles.lookup().defineHiddenClass(bytes, true);
        var declaration = inventory.getLast();
        var methodType = java.lang.invoke.MethodType.fromMethodDescriptorString(
                declaration.type().descriptorString(), lookup.lookupClass().getClassLoader());
        var entry = lookup.findStatic(lookup.lookupClass(), declaration.name(), methodType);
        try (var arena = Arena.ofConfined()) {
            var control = arena.allocate(16L + active.length, 8);
            control.set(ValueLayout.JAVA_BYTE, 0, (byte) (present ? 1 : 0));
            control.set(ValueLayout.JAVA_LONG_UNALIGNED, 8, active.length);
            for (int i = 0; i < active.length; i++)
                control.set(ValueLayout.JAVA_BYTE, 16L + i, (byte) (active[i] ? 1 : 0));
            var arguments = new ArrayList<Object>(boundaries);
            if (route.specialization().scratchParameter()) arguments.add(arena.allocate(4096, 8));
            arguments.add(geometry); arguments.add(0L); arguments.add(geometry[6]); arguments.add(control);
            entry.invokeWithArguments(arguments);
        }
    }

    @Test void sumToShapeGeneratedBodiesCoverFiveTypesAndRepresentedCopy() throws Throwable {
        Shape source = Shape.of(2,3,4), target = Shape.of(3,1);
        var attrs = new SumToShapeAttrs(target);
        compare(AggregateReductionKind.SUM, DataType.FLOAT64, source, attrs, target,
                new double[]{1,-1,2,-2,3,-3,4,-4,5,-5,6,-6,7,-7,8,-8,9,-9,10,-10,11,-11,12,-12});
        compare(AggregateReductionKind.SUM, DataType.FLOAT32, source, attrs, target,
                new float[]{1,-1,2,-2,3,-3,4,-4,5,-5,6,-6,7,-7,8,-8,9,-9,10,-10,11,-11,12,-12});
        short[] bfloat = new short[24];
        for (int i = 0; i < bfloat.length; i++) bfloat[i] = (short) ((i & 1) == 0 ? 0x3f80 : 0xbf80);
        compare(AggregateReductionKind.SUM, DataType.BFLOAT16, source, attrs, target, bfloat);
        compare(AggregateReductionKind.SUM, DataType.INT32, source, attrs, target,
                java.util.stream.IntStream.range(0,24).map(i -> i * 1_000_000_003).toArray());
        long[] longs = new long[24];
        for (int i = 0; i < longs.length; i++) longs[i] = Long.MAX_VALUE - i;
        compare(AggregateReductionKind.SUM, DataType.INT64, source, attrs, target, longs);

        float[] represented = {Float.intBitsToFloat(0x7fa12345), -0.0f,
                Float.intBitsToFloat(0xffc54321), Float.MIN_VALUE};
        compare(AggregateReductionKind.SUM, DataType.FLOAT32, Shape.of(4),
                new SumToShapeAttrs(Shape.of(4)), Shape.of(4), represented);
    }

    @Test void sumToShapeGeneralMixedCarriersLayoutsAndRangesMatchIndependentReference()
            throws Throwable {
        Shape source = Shape.of(2,3,4), target = Shape.of(3,1);
        var inputDescriptor = CpuIndexingLoweringTest.descriptor(DataType.BFLOAT16, source,
                LayoutDescriptor.of(source, new long[]{16,5,1}, 2, true));
        var outputDescriptor = CpuIndexingLoweringTest.descriptor(DataType.BFLOAT16, target,
                LayoutDescriptor.of(target, new long[]{3,0}, 1, true));
        var operation = new Operation(AggregateReductionKind.SUM, new SumToShapeAttrs(target));
        var base = CpuScatterLoweringTest.context(operation, List.of(0),
                List.of(inputDescriptor), outputDescriptor);
        PrepareContext<CpuPartitionAnalysisInputs> context = new PrepareContext<>(base.partition(), base.nodes(), base.values(), base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false, List.of(CarrierAccess.MEMORY_SEGMENT,
                CarrierAccess.SHORT_ARRAY)));
        
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        byte[] firstBytes = generator.generateClassBytes(route.specialization(), route.kernelIr());
        byte[] secondBytes = generator.generateClassBytes(route.specialization(), route.kernelIr());
        var artifact = generator.defineClassBytes(route.specialization(), route.kernelIr(), firstBytes);
        short[] physical = new short[32];
        for (int i = 0; i < physical.length; i++) physical[i] = (short) ((i & 1) == 0
                ? 0x3f80 : 0xbf80);
        MemorySegment input = MemorySegment.ofArray(physical);
        short[] expected = new short[10], actual = new short[10];
        java.util.Arrays.fill(expected, (short) 0x55aa);
        java.util.Arrays.fill(actual, (short) 0x55aa);
        CpuScalarReferenceKernel.execute((CpuAggregateIr) plan.units().getFirst().portablePlan()
                .portableKernelIr(), plan.aggregateGeometry().orElseThrow(), List.of(
                    new CpuBufferArgument.Segment(DataType.BFLOAT16, input, input.byteSize(), true),
                    new CpuBufferArgument.Shorts(expected, 0, expected.length * 2L, false)));
        try (var arena = Arena.ofConfined()) {
            var scratch = arena.allocate(plan.aggregateGeometry().orElseThrow().workspaceBytes(1), 8);
            long[] geometry = plan.aggregateGeometry().orElseThrow().pack(new long[2]);
            artifact.entryPoint().invokeWithArguments(input, actual, scratch, geometry, 0L, 1L);
            artifact.entryPoint().invokeWithArguments(input, actual, scratch, geometry, 1L, 1L);
            artifact.entryPoint().invokeWithArguments(input, actual, scratch, geometry, 1L, 3L);
        }
        assertAll(() -> assertArrayEquals(expected, actual),
                () -> assertArrayEquals(firstBytes, secondBytes),
                () -> assertEquals("(Ljava/lang/foreign/MemorySegment;[SLjava/lang/foreign/MemorySegment;[JJJ)V",
                        route.specialization().entryType().descriptorString()));
    }

    @Test void sumToShapeGeneralCopyPreservesRawFloatingBitsWithOffsetsAndRanges()
            throws Throwable {
        Shape shape = Shape.of(4);
        var inputDescriptor = CpuIndexingLoweringTest.descriptor(DataType.FLOAT32, shape,
                LayoutDescriptor.of(shape, new long[]{2}, 1, true));
        var outputDescriptor = CpuIndexingLoweringTest.descriptor(DataType.FLOAT32, shape,
                LayoutDescriptor.of(shape, new long[]{2}, 2, true));
        var operation = new Operation(AggregateReductionKind.SUM, new SumToShapeAttrs(shape));
        var base = CpuScatterLoweringTest.context(operation, List.of(0),
                List.of(inputDescriptor), outputDescriptor);
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(), base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false,
                List.of(CarrierAccess.MEMORY_SEGMENT, CarrierAccess.FLOAT_ARRAY)));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(), route.kernelIr(), generator.generateClassBytes(route.specialization(), route.kernelIr()));
        float[] physical = new float[9];
        int[] bits = {0x7fa12345, 0x80000000, 0xffa54321, 0x00000001};
        for (int i = 0; i < bits.length; i++) physical[1 + 2 * i] = Float.intBitsToFloat(bits[i]);
        float[] output = new float[11];
        java.util.Arrays.fill(output, Float.intBitsToFloat(0x7fc00001));
        long[] geometry = plan.aggregateGeometry().orElseThrow().pack(new long[2]);
        artifact.entryPoint().invokeWithArguments(MemorySegment.ofArray(physical), output,
                geometry, 0L, 2L);
        artifact.entryPoint().invokeWithArguments(MemorySegment.ofArray(physical), output,
                geometry, 2L, 2L);
        artifact.entryPoint().invokeWithArguments(MemorySegment.ofArray(physical), output,
                geometry, 2L, 4L);
        for (int i = 0; i < bits.length; i++)
            assertEquals(bits[i], Float.floatToRawIntBits(output[2 + 2 * i]));
    }


    @Test void sumToShapeEvidenceMatrixHasTypedDeterministicSelfContainedClassFiles() throws Throwable {
        var f64 = sumToShapeRoute(DataType.FLOAT64, Shape.of(64,128,256), Shape.of(128,1),
                List.of(CarrierAccess.DOUBLE_ARRAY, CarrierAccess.DOUBLE_ARRAY));
        assertThrows(IllegalArgumentException.class, () -> sumToShapeRoute(DataType.BFLOAT16,
                Shape.of(2,3,4), Shape.of(3,1),
                List.of(CarrierAccess.MEMORY_SEGMENT, CarrierAccess.SHORT_ARRAY)));
        var i64 = sumToShapeRoute(DataType.INT64, Shape.of(8,16), Shape.of(8,1),
                List.of(CarrierAccess.LONG_ARRAY, CarrierAccess.LONG_ARRAY));
        var copy = sumToShapeRoute(DataType.FLOAT32, Shape.of(32,32), Shape.of(32,32),
                List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.FLOAT_ARRAY));
        var generator = new CpuClassFileKernelGenerator();
        var routes = List.of(f64, i64, copy);
        var descriptors = List.of("([D[DLjava/lang/foreign/MemorySegment;[JJJ)V",
                "([J[J[JJJ)V", "([F[F[JJJ)V");
        for (int index = 0; index < routes.size(); index++) {
            int current = index;
            var route = routes.get(index);
            byte[] first = generator.generateClassBytes(route.specialization(), route.kernelIr());
            byte[] second = generator.generateClassBytes(route.specialization(), route.kernelIr());
            var model = ClassFile.of().parse(first);
            var method = model.methods().getFirst();
            List<MemberRefEntry> members = java.util.stream.StreamSupport.stream(
                    model.constantPool().spliterator(), false).filter(MemberRefEntry.class::isInstance)
                    .map(MemberRefEntry.class::cast).toList();
            assertAll(
                    () -> assertArrayEquals(first, second),
                    () -> assertEquals(descriptors.get(current),
                            method.methodTypeSymbol().descriptorString()),
                    () -> assertEquals(0, model.fields().size()),
                    () -> assertEquals(1, model.methods().size()),
                    () -> assertTrue(method.flags().has(AccessFlag.STATIC)),
                    () -> assertTrue(members.stream().noneMatch(entry -> entry.owner()
                            .asInternalName().startsWith("io/github/pho001/synaptik"))),
                    () -> assertTrue(members.stream().noneMatch(entry -> entry.type().stringValue()
                            .contains("Ljava/lang/Object;") || entry.owner().asInternalName()
                                    .startsWith("java/lang/reflect/") || entry.owner()
                                    .asInternalName().startsWith("java/util/"))),
                    () -> assertTrue(model.constantPool().bootstrapMethodCount() == 0),
                    () -> assertTrue(method.code().orElseThrow().elementStream().noneMatch(
                            java.lang.classfile.instruction.NewObjectInstruction.class::isInstance)));
        }
        assertAll(() -> assertNotEquals(f64.specialization().structuralKey(),
                        copy.specialization().structuralKey()),
                () -> assertTrue(f64.specialization().scratchParameter()),
                () -> assertFalse(i64.specialization().scratchParameter()),
                () -> assertFalse(copy.specialization().scratchParameter()));
    }

    private static io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan
            sumToShapeRoute(DataType type, Shape source, Shape target,
                    List<CarrierAccess> carriers) {
        var base = CpuAggregateLoweringTest.context(AggregateReductionKind.SUM, type, source,
                new SumToShapeAttrs(target), target);
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(), base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false, carriers));
        return new CpuPartitionPreparer().analyze(context).plan().units().getFirst().portablePlan();
    }



    @Test void frozenMultiAxisMinimumPreservesRawSemanticsRangesAndGuardedFallback()
            throws Throwable {
        Shape inputShape = Shape.of(64, 64, 64), outputShape = Shape.of(1, 64, 1);
        var inputDescriptor = CpuIndexingLoweringTest.descriptor(DataType.BFLOAT16, inputShape,
                LayoutDescriptor.contiguous(inputShape));
        var outputDescriptor = CpuIndexingLoweringTest.descriptor(DataType.BFLOAT16, outputShape,
                LayoutDescriptor.of(outputShape, new long[]{128, 2, 2}, 3, true));
        var base = CpuScatterLoweringTest.context(new Operation(AggregateReductionKind.MIN,
                new MultiAxisReductionAttrs(List.of(0, 2), true)), List.of(0),
                List.of(inputDescriptor), outputDescriptor);
        PrepareContext<CpuPartitionAnalysisInputs> context = new PrepareContext<>(base.partition(), base.nodes(), base.values(), base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false, List.of(CarrierAccess.MEMORY_SEGMENT,
                CarrierAccess.SHORT_ARRAY)));
        
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        byte[] bytes = generator.generateClassBytes(route.specialization(), route.kernelIr());
        var artifact = generator.defineClassBytes(route.specialization(), route.kernelIr(), bytes);

        short[] physical = new short[64 * 64 * 64];
        java.util.Arrays.fill(physical, (short) 0x4000);
        physical[multiAxisIndex(0, 0, 0)] = (short) 0x7fc1;
        physical[multiAxisIndex(2, 0, 5)] = (short) 0xffc2;
        physical[multiAxisIndex(0, 1, 0)] = 0;
        physical[multiAxisIndex(1, 1, 7)] = (short) 0x8000;
        physical[multiAxisIndex(0, 2, 0)] = (short) 0x4040;
        physical[multiAxisIndex(63, 2, 63)] = (short) 0xbf80;
        short[] immutableInput = physical.clone();
        short[] expected = new short[64 * 2 + 7];
        short[] full = new short[expected.length], ranged = new short[expected.length];
        short[] fallback = new short[expected.length];
        java.util.Arrays.fill(expected, (short) 0x55aa);
        java.util.Arrays.fill(full, (short) 0x55aa);
        java.util.Arrays.fill(ranged, (short) 0x55aa);
        java.util.Arrays.fill(fallback, (short) 0x55aa);
        for (int cell = 0; cell < 64; cell++) {
            int accumulator = Short.toUnsignedInt(physical[multiAxisIndex(0, cell, 0)]);
            for (int outer = 0; outer < 64; outer++) for (int inner = 0; inner < 64; inner++) {
                if (outer != 0 || inner != 0) accumulator = CpuAggregateEmitter.minBfloat(
                        accumulator, Short.toUnsignedInt(physical[multiAxisIndex(
                                outer, cell, inner)]));
            }
            expected[3 + 2 * cell] = (short) accumulator;
        }
        long[] geometry = plan.aggregateGeometry().orElseThrow().pack(new long[2]);
        MemorySegment input = MemorySegment.ofArray(physical);
        artifact.entryPoint().invokeWithArguments(input, full, geometry, 0L, 64L);
        long[] provedGeometry = geometry.clone();
        artifact.entryPoint().invokeWithArguments(input, ranged, geometry, 0L, 0L);
        artifact.entryPoint().invokeWithArguments(input, ranged, geometry, 0L, 7L);
        artifact.entryPoint().invokeWithArguments(input, ranged, geometry, 7L, 39L);
        artifact.entryPoint().invokeWithArguments(input, ranged, geometry, 39L, 64L);
        long[] failedSentinelProof = geometry.clone();
        failedSentinelProof[14] = 1;
        artifact.entryPoint().invokeWithArguments(input, fallback, failedSentinelProof, 0L, 64L);

        assertAll(
                () -> assertArrayEquals(expected, full),
                () -> assertArrayEquals(expected, ranged),
                () -> assertArrayEquals(expected, fallback),
                () -> assertArrayEquals(immutableInput, physical),
                () -> assertArrayEquals(new long[6], java.util.Arrays.copyOfRange(
                        provedGeometry, 14, 20)),
                () -> assertEquals((short) 0x7fc1, full[3]),
                () -> assertEquals((short) 0x8000, full[5]),
                () -> assertEquals((short) 0xbf80, full[7]));
    }

    @Test void frozenMultiAxisMinimumArtifactHasDirectPrimitiveLoopsAndTypedFallback() throws Throwable {
        Shape inputShape = Shape.of(64, 64, 64), outputShape = Shape.of(1, 64, 1);
        var inputDescriptor = CpuIndexingLoweringTest.descriptor(DataType.BFLOAT16, inputShape,
                LayoutDescriptor.contiguous(inputShape));
        var outputDescriptor = CpuIndexingLoweringTest.descriptor(DataType.BFLOAT16, outputShape,
                LayoutDescriptor.of(outputShape, new long[]{128, 2, 2}, 3, true));
        var base = CpuScatterLoweringTest.context(new Operation(AggregateReductionKind.MIN,
                new MultiAxisReductionAttrs(List.of(0, 2), true)), List.of(0),
                List.of(inputDescriptor), outputDescriptor);
        PrepareContext<CpuPartitionAnalysisInputs> context = new PrepareContext<>(base.partition(), base.nodes(), base.values(), base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false, List.of(CarrierAccess.MEMORY_SEGMENT,
                CarrierAccess.SHORT_ARRAY)));
        
        var route = new CpuPartitionPreparer().analyze(context).plan().units().getFirst()
                .portablePlan();
        var model = ClassFile.of().parse(new CpuClassFileKernelGenerator().generateClassBytes(
                route.specialization(), route.kernelIr()));
        var method = model.methods().getFirst();
        var code = method.code().orElseThrow();
        List<MemberRefEntry> members = java.util.stream.StreamSupport.stream(
                model.constantPool().spliterator(), false).filter(MemberRefEntry.class::isInstance)
                .map(MemberRefEntry.class::cast).toList();
        assertAll(
                () -> assertTrue(model.flags().has(AccessFlag.FINAL)),
                () -> assertEquals(0, model.fields().size()),
                () -> assertEquals(1, model.methods().size()),
                () -> assertEquals("invoke", method.methodName().stringValue()),
                () -> assertTrue(method.flags().has(AccessFlag.STATIC)),
                () -> assertEquals("(Ljava/lang/foreign/MemorySegment;[S[JJJ)V",
                        method.methodTypeSymbol().descriptorString()),
                () -> assertTrue(hasBoundedBfloatMinimumLoop(code)),
                () -> assertTrue(opcodeCount(code, Opcode.LALOAD) > 0),
                () -> assertTrue(opcodeCount(code, Opcode.LDIV) > 0),
                () -> assertTrue(opcodeCount(code, Opcode.LREM) > 0),
                () -> assertTrue(members.stream().noneMatch(entry -> entry.owner().asInternalName()
                        .startsWith("io/github/pho001/synaptik"))),
                () -> assertTrue(members.stream().noneMatch(entry -> entry.type().stringValue()
                        .contains("Ljava/lang/Object;") || entry.owner().asInternalName()
                                .startsWith("java/lang/reflect/") || entry.owner().asInternalName()
                                .startsWith("java/util/"))),
                () -> assertEquals(0, model.constantPool().bootstrapMethodCount()),
                () -> assertTrue(java.util.stream.StreamSupport.stream(
                        model.constantPool().spliterator(), false)
                        .noneMatch(MethodHandleEntry.class::isInstance)),
                () -> assertTrue(java.util.stream.StreamSupport.stream(
                        model.constantPool().spliterator(), false)
                        .noneMatch(DynamicConstantPoolEntry.class::isInstance)),
                () -> assertTrue(code.elementStream().noneMatch(
                        java.lang.classfile.instruction.NewObjectInstruction.class::isInstance)));
    }

    private static int multiAxisIndex(int outer, int cell, int inner) {
        return (outer * 64 + cell) * 64 + inner;
    }

    @Test void frozenNumericalResidualShapesPreserveExactSemanticsAndDisjointRanges()
            throws Throwable {
        Shape meanInputShape = Shape.of(128, 2048), meanOutputShape = Shape.of(128);
        var meanInputDescriptor = CpuIndexingLoweringTest.descriptor(DataType.FLOAT32,
                meanInputShape, LayoutDescriptor.of(meanInputShape,
                        new long[]{4096, 2}, 3, true));
        var meanOutputDescriptor = CpuIndexingLoweringTest.descriptor(DataType.FLOAT32,
                meanOutputShape, LayoutDescriptor.of(meanOutputShape, new long[]{2}, 3, true));
        var meanBase = CpuScatterLoweringTest.context(new Operation(AggregateReductionKind.MEAN,
                new AxisReductionAttrs(1, false)), List.of(0), List.of(meanInputDescriptor),
                meanOutputDescriptor);
        PrepareContext<CpuPartitionAnalysisInputs> meanContext = new PrepareContext<>(meanBase.partition(), meanBase.nodes(), meanBase.values(), meanBase.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false,
                List.of(CarrierAccess.MEMORY_SEGMENT, CarrierAccess.FLOAT_ARRAY)));
        var meanPlan = new CpuPartitionPreparer().analyze(meanContext).plan();
        var meanRoute = meanPlan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        var meanArtifact = generator.defineClassBytes(meanRoute.specialization(), meanRoute.kernelIr(), generator.generateClassBytes(meanRoute.specialization(), meanRoute.kernelIr()));


        try (var arena = Arena.ofConfined()) {
            int meanPhysicalCount = 3 + 2 * 128 * 2048;
            MemorySegment meanInput = arena.allocate((long) meanPhysicalCount * Float.BYTES,
                    Float.BYTES);
            for (int cell = 0; cell < 128; cell++) for (int factor = 0; factor < 2048; factor++) {
                int logical = cell * 2048 + factor;
                float value = 1.0f + ((logical % 17) - 8) * 0x1p-12f;
                if (cell == 1) value = -0.0f;
                else if (cell == 2 && factor == 0) value = Float.POSITIVE_INFINITY;
                else if (cell == 3 && factor < 2)
                    value = factor == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
                else if (cell == 4 && factor == 0) value = Float.intBitsToFloat(0xff80_0042);
                meanInput.set(ValueLayout.JAVA_FLOAT_UNALIGNED,
                        (long) (3 + 2 * logical) * Float.BYTES, value);
            }
            float[] meanExpected = new float[3 + 2 * 128], meanActual = new float[3 + 2 * 128];
            java.util.Arrays.fill(meanExpected, Float.intBitsToFloat(0x7fc0_005a));
            java.util.Arrays.fill(meanActual, Float.intBitsToFloat(0x7fc0_005a));
            CpuScalarReferenceKernel.execute((CpuAggregateIr) meanRoute.portableKernelIr(),
                    meanPlan.aggregateGeometry().orElseThrow(), List.of(
                            new CpuBufferArgument.Segment(DataType.FLOAT32, meanInput,
                                    meanInput.byteSize(), true),
                            new CpuBufferArgument.Floats(meanExpected, 0,
                                    (long) meanExpected.length * Float.BYTES, false)));
            long meanBytes = meanPlan.aggregateGeometry().orElseThrow().workspaceBytes(1);
            MemorySegment meanOwner = arena.allocate(meanBytes + 16, 8).fill((byte) 0x5a);
            MemorySegment meanScratch = meanOwner.asSlice(8, meanBytes);
            long[] meanGeometry = meanPlan.aggregateGeometry().orElseThrow().pack(new long[2]);
            meanArtifact.entryPoint().invokeWithArguments(meanInput, meanActual, meanScratch,
                    meanGeometry, 0L, 0L);
            meanArtifact.entryPoint().invokeWithArguments(meanInput, meanActual, meanScratch,
                    meanGeometry, 0L, 29L);
            meanArtifact.entryPoint().invokeWithArguments(meanInput, meanActual, meanScratch,
                    meanGeometry, 29L, 91L);
            meanArtifact.entryPoint().invokeWithArguments(meanInput, meanActual, meanScratch,
                    meanGeometry, 91L, 128L);


            int[] expectedMeanBits = new int[meanExpected.length];
            int[] actualMeanBits = new int[meanActual.length];
            for (int index = 0; index < meanExpected.length; index++) {
                expectedMeanBits[index] = Float.floatToRawIntBits(meanExpected[index]);
                actualMeanBits[index] = Float.floatToRawIntBits(meanActual[index]);
            }
            assertAll(
                    () -> assertArrayEquals(expectedMeanBits, actualMeanBits),
                    () -> assertEquals((byte) 0x5a,
                            meanOwner.get(ValueLayout.JAVA_BYTE, 0)),
                    () -> assertEquals((byte) 0x5a,
                            meanOwner.get(ValueLayout.JAVA_BYTE, meanBytes + 15)));
        }
    }

    @Test void frozenNumericalResidualArtifactsHaveProvedPrimitiveLoopsAndTypedFallbacks() throws Throwable {
        Shape meanInputShape = Shape.of(128, 2048), meanOutputShape = Shape.of(128);
        var meanInputDescriptor = CpuIndexingLoweringTest.descriptor(DataType.FLOAT32,
                meanInputShape, LayoutDescriptor.of(meanInputShape,
                        new long[]{4096, 2}, 3, true));
        var meanOutputDescriptor = CpuIndexingLoweringTest.descriptor(DataType.FLOAT32,
                meanOutputShape, LayoutDescriptor.of(meanOutputShape, new long[]{2}, 3, true));
        var meanBase = CpuScatterLoweringTest.context(new Operation(AggregateReductionKind.MEAN,
                new AxisReductionAttrs(1, false)), List.of(0), List.of(meanInputDescriptor),
                meanOutputDescriptor);
        PrepareContext<CpuPartitionAnalysisInputs> meanContext = new PrepareContext<>(meanBase.partition(), meanBase.nodes(), meanBase.values(), meanBase.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false,
                List.of(CarrierAccess.MEMORY_SEGMENT, CarrierAccess.FLOAT_ARRAY)));
        var meanRoute = new CpuPartitionPreparer().analyze(meanContext).plan().units().getFirst()
                .portablePlan();


        var generator = new CpuClassFileKernelGenerator();
        var meanModel = ClassFile.of().parse(generator.generateClassBytes(
                meanRoute.specialization(), meanRoute.kernelIr()));
        assertBoundedNumericalArtifact(meanModel,
                "(Ljava/lang/foreign/MemorySegment;[FLjava/lang/foreign/MemorySegment;[JJJ)V",
                true);
    }

    @Test void zeroStrideAnyVisitsTheFullDomainInOneDirectPrimitiveLoopAndRetainsFallback()
            throws Throwable {
        Shape inputShape = Shape.of(6, 5), outputShape = Shape.of(6);
        var inputDescriptor = CpuIndexingLoweringTest.descriptor(DataType.BOOL, inputShape,
                LayoutDescriptor.of(inputShape, new long[]{1, 0}, 0, true));
        var outputDescriptor = CpuIndexingLoweringTest.descriptor(DataType.BOOL, outputShape,
                LayoutDescriptor.of(outputShape, new long[]{2}, 1, true));
        var base = CpuScatterLoweringTest.context(new Operation(AggregateReductionKind.ANY,
                new AxisReductionAttrs(1, false)), List.of(0), List.of(inputDescriptor),
                outputDescriptor);
        PrepareContext<CpuPartitionAnalysisInputs> context = new PrepareContext<>(base.partition(), base.nodes(), base.values(), base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false, List.of(CarrierAccess.MEMORY_SEGMENT,
                CarrierAccess.BYTE_ARRAY)));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        byte[] bytes = generator.generateClassBytes(route.specialization(), route.kernelIr());
        var model = ClassFile.of().parse(bytes);
        var code = model.methods().getFirst().code().orElseThrow();
        var artifact = generator.defineClassBytes(route.specialization(), route.kernelIr(), bytes);
        byte[] physical = {1, 0, 1, 0, 0, 1};
        byte[] full = new byte[13], ranged = new byte[13];
        java.util.Arrays.fill(full, (byte) -1);
        java.util.Arrays.fill(ranged, (byte) -1);
        long[] geometry = plan.aggregateGeometry().orElseThrow().pack(new long[2]);
        artifact.entryPoint().invokeWithArguments(MemorySegment.ofArray(physical), full,
                geometry, 0L, 6L);
        artifact.entryPoint().invokeWithArguments(MemorySegment.ofArray(physical), ranged,
                geometry, 0L, 0L);
        artifact.entryPoint().invokeWithArguments(MemorySegment.ofArray(physical), ranged,
                geometry, 0L, 2L);
        artifact.entryPoint().invokeWithArguments(MemorySegment.ofArray(physical), ranged,
                geometry, 2L, 5L);
        artifact.entryPoint().invokeWithArguments(MemorySegment.ofArray(physical), ranged,
                geometry, 5L, 6L);
        assertAll(
                () -> assertArrayEquals(new byte[]{
                        -1, 1, -1, 0, -1, 1, -1, 0, -1, 0, -1, 1, -1}, full),
                () -> assertArrayEquals(full, ranged),
                () -> assertTrue(hasDirectFullVisitLoop(code)),
                () -> assertTrue(opcodeCount(code, Opcode.LALOAD) > 0),
                () -> assertEquals(0, model.fields().size()),
                () -> assertEquals(1, model.methods().size()));
    }

    @Test void extremaAndBooleanArtifactsAreSelfContainedTypedAndFreeOfDynamicConstructs() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16,
                DataType.INT32, DataType.INT64)) {
            for (AggregateReductionKind kind : List.of(AggregateReductionKind.MIN,
                    AggregateReductionKind.MAX)) assertStrictArtifact(kind, type);
        }
        assertStrictArtifact(AggregateReductionKind.ALL, DataType.BOOL);
        assertStrictArtifact(AggregateReductionKind.ANY, DataType.BOOL);
    }

    @Test void numericalArtifactsHaveTypedScratchDescriptorsAndDirectAllowedMembers() throws Throwable {
        var base = CpuAggregateLoweringTest.context(AggregateReductionKind.SUM, DataType.FLOAT64,
                Shape.of(3), NoOperationAttrs.INSTANCE, Shape.scalar());
        PrepareContext<CpuPartitionAnalysisInputs> context = new PrepareContext<>(base.partition(), base.nodes(), base.values(), base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false,
                List.of(CarrierAccess.DOUBLE_ARRAY, CarrierAccess.DOUBLE_ARRAY)));
        var route = new CpuPartitionPreparer().analyze(context).plan().units().getFirst().portablePlan();
        var method = ClassFile.of().parse(new CpuClassFileKernelGenerator().generateClassBytes(
                route.specialization(), route.kernelIr())).methods().getFirst();
        var invokes = method.code().orElseThrow().elementStream()
                .filter(InvokeInstruction.class::isInstance).map(InvokeInstruction.class::cast)
                .toList();
        assertAll(
                () -> assertEquals("([D[DLjava/lang/foreign/MemorySegment;[JJJ)V",
                        method.methodTypeSymbol().descriptorString()),
                () -> assertTrue(route.specialization().scratchParameter()),
                () -> assertTrue(invokes.stream().noneMatch(i -> i.owner().asInternalName()
                        .startsWith("io/github/pho001/synaptik"))),
                () -> assertTrue(invokes.stream().allMatch(i -> {
                    String owner = i.owner().asInternalName();
                    return owner.equals("java/lang/Double") || owner.equals("java/lang/Long")
                            || owner.equals("java/lang/foreign/MemorySegment");
                })));
    }
    @Test void generatedClassContainsTypedAggregateFoldWithoutGenericDispatchBridge() throws Throwable {
        var base = CpuAggregateLoweringTest.context(AggregateReductionKind.MIN, DataType.FLOAT64,
                Shape.of(1024), NoOperationAttrs.INSTANCE, Shape.scalar());
        PrepareContext<CpuPartitionAnalysisInputs> context = new PrepareContext<>(base.partition(), base.nodes(), base.values(), base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false,
                List.of(CarrierAccess.DOUBLE_ARRAY, CarrierAccess.DOUBLE_ARRAY)));
        var route = new CpuPartitionPreparer().analyze(context).plan().units().getFirst().portablePlan();
        var code = ClassFile.of().parse(new CpuClassFileKernelGenerator().generateClassBytes(
                route.specialization(), route.kernelIr())).methods().getFirst().code().orElseThrow();
        var invokes = code.elementStream().filter(InvokeInstruction.class::isInstance)
                .map(InvokeInstruction.class::cast).toList();
        assertAll(() -> assertTrue(invokes.stream().noneMatch(i -> i.name().stringValue().equals("execute"))),
                () -> assertTrue(invokes.stream().noneMatch(i -> i.type().stringValue().contains("Ljava/lang/Object;"))),
                () -> assertTrue(invokes.stream().noneMatch(i -> i.owner().asInternalName()
                        .startsWith("io/github/pho001/synaptik"))),
                () -> assertTrue(code.elementStream().filter(Instruction.class::isInstance)
                        .map(Instruction.class::cast).anyMatch(i -> i.opcode() == Opcode.DALOAD)),
                () -> assertTrue(code.elementStream().filter(Instruction.class::isInstance)
                        .map(Instruction.class::cast).anyMatch(i -> i.opcode() == Opcode.DASTORE)),
                () -> assertTrue(code.elementStream().noneMatch(element -> element instanceof java.lang.classfile.instruction.NewObjectInstruction)));
    }

    @Test void executesAllKindsAndRepresentedTypesWithExactIdentities() throws Throwable {
        assertArrayEquals(new double[]{2,5}, (double[]) invoke(AggregateReductionKind.SUM,
                DataType.FLOAT64, Shape.of(2,3), new AxisReductionAttrs(1,false), Shape.of(2),
                new double[]{1,-2,3,4,5,-4}));
        assertArrayEquals(new float[]{2f/3f,5f/3f}, (float[]) invoke(AggregateReductionKind.MEAN,
                DataType.FLOAT32, Shape.of(2,3), new AxisReductionAttrs(1,false), Shape.of(2),
                new float[]{1,-2,3,4,5,-4}));
        assertArrayEquals(new double[]{-6,-80}, (double[]) invoke(AggregateReductionKind.PROD,
                DataType.FLOAT64, Shape.of(2,3), new AxisReductionAttrs(1,false), Shape.of(2),
                new double[]{1,-2,3,4,5,-4}));
        assertArrayEquals(new int[]{2,5}, (int[]) invoke(AggregateReductionKind.SUM,
                DataType.INT32, Shape.of(2,3), new AxisReductionAttrs(1,false), Shape.of(2),
                new int[]{1,-2,3,4,5,-4}));
        assertArrayEquals(new long[]{-6,-80}, (long[]) invoke(AggregateReductionKind.PROD,
                DataType.INT64, Shape.of(2,3), new AxisReductionAttrs(1,false), Shape.of(2),
                new long[]{1,-2,3,4,5,-4}));
        assertArrayEquals(new double[]{-2,-4}, (double[]) invoke(AggregateReductionKind.MIN,
                DataType.FLOAT64, Shape.of(2,3), new AxisReductionAttrs(1,false), Shape.of(2),
                new double[]{1,-2,3,4,5,-4}));
        assertArrayEquals(new float[]{3,5}, (float[]) invoke(AggregateReductionKind.MAX,
                DataType.FLOAT32, Shape.of(2,3), new AxisReductionAttrs(1,false), Shape.of(2),
                new float[]{1,-2,3,4,5,-4}));
        assertArrayEquals(new short[]{(short)0xc000,(short)0xc080}, (short[]) invoke(
                AggregateReductionKind.MIN, DataType.BFLOAT16, Shape.of(2,3),
                new AxisReductionAttrs(1,false), Shape.of(2),
                new short[]{(short)0x3f80,(short)0xc000,(short)0x4040,
                    (short)0x4080,(short)0x40a0,(short)0xc080}));
        assertArrayEquals(new int[]{3,9}, (int[]) invoke(AggregateReductionKind.MAX, DataType.INT32,
                Shape.of(2,3), new AxisReductionAttrs(1,false), Shape.of(2),
                new int[]{-8,3,2,9,-4,1}));
        assertArrayEquals(new long[]{Long.MIN_VALUE}, (long[]) invoke(AggregateReductionKind.MIN,
                DataType.INT64, Shape.of(3), NoOperationAttrs.INSTANCE, Shape.scalar(),
                new long[]{7,Long.MIN_VALUE,9}));
        assertArrayEquals(new byte[]{0,1}, (byte[]) invoke(AggregateReductionKind.ALL, DataType.BOOL,
                Shape.of(2,2), new AxisReductionAttrs(1,false), Shape.of(2),
                new byte[]{1,0,1,1}));
        assertArrayEquals(new byte[]{1,0}, (byte[]) invoke(AggregateReductionKind.ANY, DataType.BOOL,
                Shape.of(2,2), new AxisReductionAttrs(1,false), Shape.of(2),
                new byte[]{0,1,0,0}));
        assertEquals(Double.POSITIVE_INFINITY, ((double[]) invoke(AggregateReductionKind.MIN,
                DataType.FLOAT64, Shape.of(0), NoOperationAttrs.INSTANCE, Shape.scalar(),
                new double[0]))[0]);
        assertArrayEquals(new byte[]{1,1}, (byte[]) invoke(AggregateReductionKind.ALL, DataType.BOOL,
                Shape.of(2,0), new AxisReductionAttrs(1,false), Shape.of(2), new byte[0]));
    }

    @Test void preservesFirstNaNBitsSignedZeroInfinitiesAndMultiAxisMembership() throws Throwable {
        long first = 0x7ff8000000000042L, second = 0x7ff8000000000099L;
        double[] nan = (double[]) invoke(AggregateReductionKind.MIN, DataType.FLOAT64, Shape.of(4),
                NoOperationAttrs.INSTANCE, Shape.scalar(), new double[]{3,
                    Double.longBitsToDouble(first), Double.longBitsToDouble(second), -1});
        assertEquals(first, Double.doubleToRawLongBits(nan[0]));
        float[] minimum = (float[]) invoke(AggregateReductionKind.MIN, DataType.FLOAT32, Shape.of(2),
                NoOperationAttrs.INSTANCE, Shape.scalar(), new float[]{0.0f,-0.0f});
        float[] maximum = (float[]) invoke(AggregateReductionKind.MAX, DataType.FLOAT32, Shape.of(2),
                NoOperationAttrs.INSTANCE, Shape.scalar(), new float[]{-0.0f,0.0f});
        assertEquals(Float.floatToRawIntBits(-0.0f), Float.floatToRawIntBits(minimum[0]));
        assertEquals(Float.floatToRawIntBits(0.0f), Float.floatToRawIntBits(maximum[0]));
        int[] values = {1,9,3,4,5,6,7,8,2,0,11,10};
        Object ordered = invoke(AggregateReductionKind.MAX, DataType.INT32, Shape.of(2,2,3),
                new MultiAxisReductionAttrs(List.of(2,0),false), Shape.of(2), values);
        Object reversed = invoke(AggregateReductionKind.MAX, DataType.INT32, Shape.of(2,2,3),
                new MultiAxisReductionAttrs(List.of(0,2),false), Shape.of(2), values);
        assertArrayEquals((int[]) ordered, (int[]) reversed);
    }

    @Test void extremaPreserveRawBitsAcrossAllRepresentedTypesAndBothKinds() throws Throwable {
        for (AggregateReductionKind kind : List.of(AggregateReductionKind.MIN,
                AggregateReductionKind.MAX)) {
            compare(kind, DataType.FLOAT64, Shape.of(8), NoOperationAttrs.INSTANCE, Shape.scalar(),
                    new double[]{Double.longBitsToDouble(0x7ff0000000000042L),
                            Double.longBitsToDouble(0xfff8000000000099L), -0.0, +0.0,
                            Double.MIN_VALUE, -Double.MIN_VALUE, Double.MAX_VALUE,
                            -Double.MAX_VALUE});
            compare(kind, DataType.FLOAT32, Shape.of(8), NoOperationAttrs.INSTANCE, Shape.scalar(),
                    new float[]{Float.intBitsToFloat(0x7f800042), Float.intBitsToFloat(0xffc00099),
                            -0.0f, +0.0f, Float.MIN_VALUE, -Float.MIN_VALUE, Float.MAX_VALUE,
                            -Float.MAX_VALUE});
            compare(kind, DataType.BFLOAT16, Shape.of(8), NoOperationAttrs.INSTANCE, Shape.scalar(),
                    new short[]{(short) 0x7f81, (short) 0xffc2, (short) 0x8000, 0,
                            1, (short) 0x8001, (short) 0x7f7f, (short) 0xff7f});
            compare(kind, DataType.INT32, Shape.of(5), NoOperationAttrs.INSTANCE, Shape.scalar(),
                    new int[]{Integer.MIN_VALUE, -1, 0, 1, Integer.MAX_VALUE});
            compare(kind, DataType.INT64, Shape.of(5), NoOperationAttrs.INSTANCE, Shape.scalar(),
                    new long[]{Long.MIN_VALUE, -1, 0, 1, Long.MAX_VALUE});
        }
        compare(AggregateReductionKind.ALL, DataType.BOOL, Shape.of(4), NoOperationAttrs.INSTANCE,
                Shape.scalar(), new byte[]{1, 1, 0, 1});
        compare(AggregateReductionKind.ANY, DataType.BOOL, Shape.of(4), NoOperationAttrs.INSTANCE,
                Shape.scalar(), new byte[]{0, 0, 1, 0});
    }

    /** Shared aggregate ownership must not substitute exact ordinary math for Scan's represented fold. */
    @Test void commonOwnerRetainsExactOrdinaryAndSequentialScanCancellationAlgorithms() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64)) {
            Object values = type == DataType.FLOAT32 ? new float[]{1e20f,1f,-1e20f} : new double[]{1e20,1,-1e20};
            Object exact = invoke(AggregateReductionKind.SUM,type,Shape.of(3),NoOperationAttrs.INSTANCE,Shape.scalar(),values);
            assertEquals(1d,type == DataType.FLOAT32 ? ((float[])exact)[0] : ((double[])exact)[0]);
            var h = CpuGeneralScanLoweringTest.tensor(type,false,3);
            var x = CpuGeneralScanLoweringTest.tensor(type,false,3);
            var scan = Scan.apply(List.of(
                    CpuGeneralScanLoweringTest.tensor(type,false,3)),
                    List.of(CpuGeneralScanLoweringTest.tensor(type,false,1,3)),
                    List.of(),new ScanBody(List.of(h),List.of(x),List.of(),List.of(h),List.of(x.sum())),false);
            var facts = CpuGeneralScanLowering.analyze(
                    CpuGeneralScanLoweringTest.query(scan.outputs()));
            var generated = CpuGeneralScanGeneratedKernelTest.generated(facts);
            try (var arena = Arena.ofConfined()) {
                var args = new ArrayList<Object>();
                for (long span : facts.spans()) args.add(arena.allocate(span*type.byteWidth(),type.byteWidth()));
                MemorySegment sequence = (MemorySegment)args.get(facts.ir().actualReads().get(1));
                for (int i = 0; i < 3; i++) {
                    if (type == DataType.FLOAT32) sequence.setAtIndex(ValueLayout.JAVA_FLOAT,i,((float[])values)[i]);
                    else sequence.setAtIndex(ValueLayout.JAVA_DOUBLE,i,((double[])values)[i]);
                }
                MemorySegment output = (MemorySegment)args.get(facts.ir().scalarReads().size());
                if (generated.specialization().scratchParameter()) args.add(arena.allocate(facts.geometry().scratchBytes(),type.byteWidth()));
                args.add(facts.pack(3)); args.add(0L); args.add(3L);
                generated.entryPoint().invokeWithArguments(args);
                assertEquals(0d,type == DataType.FLOAT32 ? output.getAtIndex(ValueLayout.JAVA_FLOAT,0) : output.getAtIndex(ValueLayout.JAVA_DOUBLE,0));
            }
        }
    }

    @Test void numericalAggregatesRoundExactFiniteValuesAndCanonicalizeSpecialValues()
            throws Throwable {
        double[] cancellation = (double[]) invoke(AggregateReductionKind.SUM, DataType.FLOAT64,
                Shape.of(3), NoOperationAttrs.INSTANCE, Shape.scalar(),
                new double[]{0x1p53, 1.0, -0x1p53});
        assertEquals(Double.doubleToRawLongBits(1.0), Double.doubleToRawLongBits(cancellation[0]));
        double[] mean = (double[]) invoke(AggregateReductionKind.MEAN, DataType.FLOAT64,
                Shape.of(3), NoOperationAttrs.INSTANCE, Shape.scalar(),
                new double[]{1.0, 0.0, 0.0});
        assertEquals(Double.doubleToRawLongBits(1.0 / 3.0), Double.doubleToRawLongBits(mean[0]));
        double[] subnormal = (double[]) invoke(AggregateReductionKind.SUM, DataType.FLOAT64,
                Shape.of(2), NoOperationAttrs.INSTANCE, Shape.scalar(),
                new double[]{Double.MIN_VALUE, Double.MIN_VALUE});
        assertEquals(2L, Double.doubleToRawLongBits(subnormal[0]));
        double[] nan = (double[]) invoke(AggregateReductionKind.SUM, DataType.FLOAT64,
                Shape.of(2), NoOperationAttrs.INSTANCE, Shape.scalar(),
                new double[]{Double.longBitsToDouble(0xfff0000000000042L), 1});
        assertEquals(0x7ff8000000000000L, Double.doubleToRawLongBits(nan[0]));
        double[] negativeZero = (double[]) invoke(AggregateReductionKind.SUM, DataType.FLOAT64,
                Shape.of(2), NoOperationAttrs.INSTANCE, Shape.scalar(),
                new double[]{-0.0, -0.0});
        assertEquals(Long.MIN_VALUE, Double.doubleToRawLongBits(negativeZero[0]));
        double[] productNan = (double[]) invoke(AggregateReductionKind.PROD, DataType.FLOAT64,
                Shape.of(2), NoOperationAttrs.INSTANCE, Shape.scalar(),
                new double[]{-0.0, Double.POSITIVE_INFINITY});
        assertEquals(0x7ff8000000000000L, Double.doubleToRawLongBits(productNan[0]));
        double[] emptyProduct = (double[]) invoke(AggregateReductionKind.PROD, DataType.FLOAT64,
                Shape.of(0), NoOperationAttrs.INSTANCE, Shape.scalar(), new double[0]);
        assertEquals(Double.doubleToRawLongBits(1.0), Double.doubleToRawLongBits(emptyProduct[0]));
    }

    @Test void exactRoundingCoversBothHalfwayParitiesOverflowBoundariesAndSignedUnderflow()
            throws Throwable {
        double[] even = (double[]) invoke(AggregateReductionKind.SUM, DataType.FLOAT64,
                Shape.of(2), NoOperationAttrs.INSTANCE, Shape.scalar(),
                new double[]{1.0, 0x1p-53});
        double[] odd = (double[]) invoke(AggregateReductionKind.SUM, DataType.FLOAT64,
                Shape.of(2), NoOperationAttrs.INSTANCE, Shape.scalar(),
                new double[]{Math.nextUp(1.0), 0x1p-53});
        double[] overflow = (double[]) invoke(AggregateReductionKind.SUM, DataType.FLOAT64,
                Shape.of(2), NoOperationAttrs.INSTANCE, Shape.scalar(),
                new double[]{Double.MAX_VALUE, Double.MAX_VALUE});
        double[] largestSubnormal = (double[]) invoke(AggregateReductionKind.SUM,
                DataType.FLOAT64, Shape.of(2), NoOperationAttrs.INSTANCE, Shape.scalar(),
                new double[]{Double.MIN_NORMAL, -Double.MIN_VALUE});
        double[] negativeUnderflow = (double[]) invoke(AggregateReductionKind.MEAN,
                DataType.FLOAT64, Shape.of(3), NoOperationAttrs.INSTANCE, Shape.scalar(),
                new double[]{-Double.MIN_VALUE, -0.0, -0.0});
        assertAll(
                () -> assertEquals(Double.doubleToRawLongBits(1.0),
                        Double.doubleToRawLongBits(even[0])),
                () -> assertEquals(Double.doubleToRawLongBits(Math.nextUp(Math.nextUp(1.0))),
                        Double.doubleToRawLongBits(odd[0])),
                () -> assertEquals(Double.doubleToRawLongBits(Double.POSITIVE_INFINITY),
                        Double.doubleToRawLongBits(overflow[0])),
                () -> assertEquals(0x000fffffffffffffL,
                        Double.doubleToRawLongBits(largestSubnormal[0])),
                () -> assertEquals(Long.MIN_VALUE,
                        Double.doubleToRawLongBits(negativeUnderflow[0])));
    }

    @Test void numericalInventoryMatchesIndependentOracleAcrossTypesAndForms() throws Throwable {
        compare(AggregateReductionKind.SUM, DataType.FLOAT32, Shape.of(2,2,2),
                new MultiAxisReductionAttrs(List.of(2,0), true), Shape.of(1,2,1),
                new float[]{1,-2,3,4,Float.MAX_VALUE,Float.MAX_VALUE,Float.MIN_VALUE,-0.0f});
        compare(AggregateReductionKind.SUM, DataType.BFLOAT16, Shape.of(2,3),
                new AxisReductionAttrs(1, false), Shape.of(2),
                new short[]{(short)0x3f80,(short)0xbf80,(short)0x0001,
                        (short)0x7f7f,(short)0x7f7f,(short)0x8000});
        compare(AggregateReductionKind.MEAN, DataType.FLOAT64, Shape.of(2,2,2),
                new MultiAxisReductionAttrs(List.of(0,2), false), Shape.of(2),
                new double[]{1,0,3,0,-1,0,-3,Double.MIN_VALUE});
        compare(AggregateReductionKind.MEAN, DataType.BFLOAT16, Shape.of(3),
                NoOperationAttrs.INSTANCE, Shape.scalar(),
                new short[]{(short)0x3f80,(short)0,(short)0});
        compare(AggregateReductionKind.PROD, DataType.FLOAT32, Shape.of(2,3),
                new AxisReductionAttrs(1, true), Shape.of(2,1),
                new float[]{Float.MAX_VALUE,2,0.5f,Float.MIN_VALUE,0.5f,-2});
        compare(AggregateReductionKind.PROD, DataType.FLOAT64, Shape.of(4),
                NoOperationAttrs.INSTANCE, Shape.scalar(),
                new double[]{-0.0,Double.NEGATIVE_INFINITY,2,Double.longBitsToDouble(1)});
        compare(AggregateReductionKind.SUM, DataType.INT64, Shape.of(3),
                NoOperationAttrs.INSTANCE, Shape.scalar(),
                new long[]{Long.MAX_VALUE,Long.MAX_VALUE,2});
        compare(AggregateReductionKind.PROD, DataType.INT32, Shape.of(3),
                NoOperationAttrs.INSTANCE, Shape.scalar(),
                new int[]{Integer.MAX_VALUE,Integer.MAX_VALUE,3});
    }

    @Test void arbitraryFloatingDomainCountsMatchIndependentRationalOracle() throws Throwable {
        int[] counts = {0, 1, 2, 3, 5, 7, 9, 17};
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16)) {
            for (AggregateReductionKind kind : List.of(AggregateReductionKind.SUM,
                    AggregateReductionKind.MEAN, AggregateReductionKind.PROD)) {
                for (int count : counts) {
                    compare(kind, type, Shape.of(count), NoOperationAttrs.INSTANCE, Shape.scalar(),
                            adversarialFloatingValues(type, count));
                }
            }
        }
    }

    @Test void sumToShapeAdversarialDomainsReuseExactOracleAndIntegralModularRules()
            throws Throwable {
        int[] counts = {0, 1, 2, 3, 5, 7, 9, 17};
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16)) {
            for (int count : counts) {
                Shape source = Shape.of(1, count), target = Shape.of(1, 1);
                compare(AggregateReductionKind.SUM, type, source,
                        new SumToShapeAttrs(target), target,
                        adversarialFloatingValues(type, count));
            }
        }
        compare(AggregateReductionKind.SUM, DataType.INT32, Shape.of(1,3),
                new SumToShapeAttrs(Shape.of(1,1)), Shape.of(1,1),
                new int[]{Integer.MAX_VALUE, Integer.MAX_VALUE, 2});
        compare(AggregateReductionKind.SUM, DataType.INT64, Shape.of(1,3),
                new SumToShapeAttrs(Shape.of(1,1)), Shape.of(1,1),
                new long[]{Long.MAX_VALUE, Long.MAX_VALUE, 2});
        compare(AggregateReductionKind.SUM, DataType.INT64, Shape.of(1,0),
                new SumToShapeAttrs(Shape.of(1,1)), Shape.of(1,1), new long[0]);
    }


    @Test void generatedResultsMatchIndependentReferenceIncludingEmptyAxisPointForm() throws Throwable {
        compare(AggregateReductionKind.SUM, DataType.FLOAT64, Shape.of(2,3),
                new AxisReductionAttrs(1,false), Shape.of(2),
                new double[]{0x1p53,1,-0x1p53,Double.MIN_VALUE,Double.MIN_VALUE,-0.0});
        compare(AggregateReductionKind.MEAN, DataType.FLOAT32, Shape.of(2,3),
                new AxisReductionAttrs(1,false), Shape.of(2), new float[]{1,0,0,-1,0,0});
        compare(AggregateReductionKind.PROD, DataType.BFLOAT16, Shape.of(2,2),
                new AxisReductionAttrs(1,false), Shape.of(2),
                new short[]{(short)0x3f80,(short)0x4000,(short)0x8000,(short)0x7f80});
        compare(AggregateReductionKind.MIN, DataType.FLOAT64, Shape.of(2,3),
                new AxisReductionAttrs(1,false), Shape.of(2),
                new double[]{1,-2,Double.NaN,4,-0.0,0.0});
        compare(AggregateReductionKind.MAX, DataType.INT64, Shape.of(2,2,2),
                new MultiAxisReductionAttrs(List.of(2,0),true), Shape.of(1,2,1),
                new long[]{1,-9,3,4,5,6,Long.MAX_VALUE,-8});
        compare(AggregateReductionKind.ANY, DataType.BOOL, Shape.of(2,3),
                new MultiAxisReductionAttrs(List.of(),true), Shape.of(2,3),
                new byte[]{0,1,0,1,1,0});
    }

    @Test void supportsTransposedNativeInputAndOffsetInterleavedHeapOutput() throws Throwable {
        Shape inputShape = Shape.of(2,3), outputShape = Shape.of(2);
        var inputDescriptor = CpuIndexingLoweringTest.descriptor(DataType.FLOAT32, inputShape,
                LayoutDescriptor.of(inputShape, new long[]{1,2},0,true));
        var outputDescriptor = CpuIndexingLoweringTest.descriptor(DataType.FLOAT32, outputShape,
                LayoutDescriptor.of(outputShape,new long[]{2},1,true));
        var base = CpuScatterLoweringTest.context(new Operation(AggregateReductionKind.MAX,
                new AxisReductionAttrs(1,false)), List.of(0), List.of(inputDescriptor), outputDescriptor);
        PrepareContext<CpuPartitionAnalysisInputs> context = new PrepareContext<>(base.partition(), base.nodes(), base.values(), base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false,List.of(CarrierAccess.MEMORY_SEGMENT,
                CarrierAccess.FLOAT_ARRAY)));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator(); var artifact = generator.defineClassBytes(route.specialization(), route.kernelIr(), generator.generateClassBytes(route.specialization(), route.kernelIr()));
        try (var arena = Arena.ofConfined()) {
            var segment = arena.allocate(6L*Float.BYTES,Float.BYTES);
            float[] physical={1,4,2,5,3,6};
            for(int i=0;i<physical.length;i++) segment.set(ValueLayout.JAVA_FLOAT,i*4L,physical[i]);
            float[] output={-7,-7,-7,-7};
            artifact.entryPoint().invokeWithArguments(segment,output,
                    plan.aggregateGeometry().orElseThrow().pack(new long[2]),0L,2L);
            assertArrayEquals(new float[]{-7,3,-7,6},output);
        }
    }

    @Test void numericalGeneralSegmentInputAndOffsetHeapOutputUseLongAddressBody() throws Throwable {
        Shape inputShape = Shape.of(2,3), outputShape = Shape.of(2);
        var inputDescriptor = CpuIndexingLoweringTest.descriptor(DataType.FLOAT32, inputShape,
                LayoutDescriptor.of(inputShape, new long[]{1,2},0,true));
        var outputDescriptor = CpuIndexingLoweringTest.descriptor(DataType.FLOAT32, outputShape,
                LayoutDescriptor.of(outputShape,new long[]{2},1,true));
        var base = CpuScatterLoweringTest.context(new Operation(AggregateReductionKind.SUM,
                new AxisReductionAttrs(1,false)), List.of(0), List.of(inputDescriptor),
                outputDescriptor);
        PrepareContext<CpuPartitionAnalysisInputs> context = new PrepareContext<>(base.partition(), base.nodes(), base.values(), base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false,List.of(CarrierAccess.MEMORY_SEGMENT,
                CarrierAccess.FLOAT_ARRAY)));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator(); var artifact = generator.defineClassBytes(route.specialization(), route.kernelIr(), generator.generateClassBytes(route.specialization(), route.kernelIr()));
        try (var arena = Arena.ofConfined()) {
            var segment = arena.allocate(6L*Float.BYTES,Float.BYTES);
            float[] physical={1,4,2,5,3,6};
            for(int i=0;i<physical.length;i++) segment.set(ValueLayout.JAVA_FLOAT,i*4L,physical[i]);
            float[] output={-7,-7,-7,-7};
            long bytes = plan.aggregateGeometry().orElseThrow().workspaceBytes(1);
            artifact.entryPoint().invokeWithArguments(segment,output,arena.allocate(bytes,8),
                    plan.aggregateGeometry().orElseThrow().pack(new long[2]),0L,2L);
            assertArrayEquals(new float[]{-7,6,-7,15},output);
        }
    }

    @Test void numericalGeneratedBodyHonorsPartialAndEmptyOutputCellRanges() throws Throwable {
        var base = CpuAggregateLoweringTest.context(AggregateReductionKind.SUM, DataType.FLOAT64,
                Shape.of(4,3), new AxisReductionAttrs(1,false), Shape.of(4));
        PrepareContext<CpuPartitionAnalysisInputs> context = new PrepareContext<>(base.partition(), base.nodes(), base.values(), base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false,List.of(CarrierAccess.DOUBLE_ARRAY,
                CarrierAccess.DOUBLE_ARRAY)));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan(); var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(), route.kernelIr(), generator.generateClassBytes(route.specialization(), route.kernelIr()));
        double[] input={1,2,3,4,5,6,7,8,9,10,11,12};
        double[] output={-7,-7,-7,-7};
        try (var arena = Arena.ofConfined()) {
            var scratch=arena.allocate(plan.aggregateGeometry().orElseThrow().workspaceBytes(1),8);
            long[] geometry=plan.aggregateGeometry().orElseThrow().pack(new long[2]);
            artifact.entryPoint().invokeWithArguments(input,output,scratch,geometry,1L,3L);
            artifact.entryPoint().invokeWithArguments(input,output,scratch,geometry,2L,2L);
        }
        assertArrayEquals(new double[]{-7,15,24,-7},output);
    }

    @Test void zeroStrideReadsHaveBitwiseFullAndDisjointRangeParity() throws Throwable {
        Shape inputShape=Shape.of(8,2), outputShape=Shape.of(8);
        var inputDescriptor=CpuIndexingLoweringTest.descriptor(DataType.FLOAT32,inputShape,
                LayoutDescriptor.of(inputShape,new long[]{1,0},0,true));
        var outputDescriptor=CpuScatterLoweringTest.desc(DataType.FLOAT32,outputShape);
        var base=CpuScatterLoweringTest.context(new Operation(AggregateReductionKind.MIN,
                new AxisReductionAttrs(1,false)),List.of(0),List.of(inputDescriptor),outputDescriptor);
        PrepareContext<CpuPartitionAnalysisInputs> context=new PrepareContext<>(base.partition(), base.nodes(), base.values(), base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false,List.of(CarrierAccess.FLOAT_ARRAY,
                CarrierAccess.FLOAT_ARRAY)));
        var plan=new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator=new CpuClassFileKernelGenerator(); var artifact=generator.defineClassBytes(route.specialization(), route.kernelIr(), generator.generateClassBytes(route.specialization(), route.kernelIr()));
        float[] input={-0.0f,0.0f,Float.intBitsToFloat(0x7fc00042),-3,4,-5,
                Float.POSITIVE_INFINITY,Float.NEGATIVE_INFINITY};
        float[] full=new float[8], ranged=new float[8];
        artifact.entryPoint().invokeWithArguments(input,full,
                plan.aggregateGeometry().orElseThrow().pack(new long[2]),0L,8L);
        for(long start=0;start<8;start+=2) artifact.entryPoint().invokeWithArguments(input,ranged,
                plan.aggregateGeometry().orElseThrow().pack(new long[2]),start,start+2);
        assertArrayEquals(java.util.stream.IntStream.range(0,8)
                .map(i->Float.floatToRawIntBits(full[i])).toArray(),
                java.util.stream.IntStream.range(0,8)
                .map(i->Float.floatToRawIntBits(ranged[i])).toArray());
    }

    @Test void everyLowFormatUsesItsOwnRawNumericAggregateBoundary() throws Throwable {
        for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT16,
                DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2)) {
            Object values = lowValues(type, 2, 4);
            for (AggregateReductionKind kind : List.of(AggregateReductionKind.SUM,
                    AggregateReductionKind.MEAN, AggregateReductionKind.PROD)) {
                Object result = invoke(kind, type, Shape.of(2), NoOperationAttrs.INSTANCE,
                        Shape.scalar(), values);
                Object expected = lowValues(type, kind == AggregateReductionKind.SUM ? 6
                        : kind == AggregateReductionKind.MEAN ? 3 : 8);
                if (result instanceof byte[] bytes) assertArrayEquals((byte[]) expected, bytes);
                else assertArrayEquals((short[]) expected, (short[]) result);
            }
        }
    }

    @Test void e4m3ExtremaDoNotUseNanAsAnInfinityWinnerAndKeepRawSelections() throws Throwable {
        for (AggregateReductionKind kind : List.of(AggregateReductionKind.MIN, AggregateReductionKind.MAX)) {
            byte[] result = (byte[]) invoke(kind, DataType.FLOAT8_E4M3, Shape.of(3),
                    NoOperationAttrs.INSTANCE, Shape.scalar(), new byte[] {0x7e, (byte) 0xfe, 0x38});
            assertArrayEquals(new byte[] {kind == AggregateReductionKind.MIN ? (byte) 0xfe : (byte) 0x7e}, result);
            assertArrayEquals(new byte[] {(byte) 0xff}, (byte[]) invoke(kind, DataType.FLOAT8_E4M3,
                    Shape.of(3), NoOperationAttrs.INSTANCE, Shape.scalar(),
                    new byte[] {0x38, (byte) 0xff, 0x7f}));
            assertArrayEquals(new byte[] {0x7f}, (byte[]) invoke(kind, DataType.FLOAT8_E4M3,
                    Shape.of(0), NoOperationAttrs.INSTANCE, Shape.scalar(), new byte[0]));
            assertArrayEquals(new byte[] {kind == AggregateReductionKind.MIN ? (byte) 0x80 : 0},
                    (byte[]) invoke(kind, DataType.FLOAT8_E4M3, Shape.of(2),
                            NoOperationAttrs.INSTANCE, Shape.scalar(), new byte[] {0, (byte) 0x80}));
        }
    }

    @Test void fp8ExactSumsAndProductsClassifyTopFiniteBinsAndOverflowAtFinalRounding() throws Throwable {
        assertArrayEquals(new byte[] {0x7e}, (byte[]) invoke(AggregateReductionKind.SUM,
                DataType.FLOAT8_E4M3, Shape.of(2), NoOperationAttrs.INSTANCE, Shape.scalar(),
                lowValues(DataType.FLOAT8_E4M3, 448, 16)));
        assertArrayEquals(new byte[] {0x7f}, (byte[]) invoke(AggregateReductionKind.SUM,
                DataType.FLOAT8_E4M3, Shape.of(2), NoOperationAttrs.INSTANCE, Shape.scalar(),
                lowValues(DataType.FLOAT8_E4M3, 448, 32)));
        assertArrayEquals(new byte[] {0x7e}, (byte[]) invoke(AggregateReductionKind.PROD,
                DataType.FLOAT8_E4M3, Shape.of(2), NoOperationAttrs.INSTANCE, Shape.scalar(),
                lowValues(DataType.FLOAT8_E4M3, 448, 1)));
        assertArrayEquals(new byte[] {0x7c}, (byte[]) invoke(AggregateReductionKind.PROD,
                DataType.FLOAT8_E5M2, Shape.of(2), NoOperationAttrs.INSTANCE, Shape.scalar(),
                lowValues(DataType.FLOAT8_E5M2, 57344, 2)));
    }

    private static Object lowValues(DataType type, float... values) {
        if (type == DataType.FLOAT8_E4M3 || type == DataType.FLOAT8_E5M2) {
            byte[] result = new byte[values.length];
            for (int i = 0; i < values.length; i++) result[i] = type == DataType.FLOAT8_E4M3
                    ? io.github.pho001.synaptik.model.datatype.Float8E4M3Bits.fromFloat(values[i])
                    : io.github.pho001.synaptik.model.datatype.Float8E5M2Bits.fromFloat(values[i]);
            return result;
        }
        short[] result = new short[values.length];
        for (int i = 0; i < values.length; i++) result[i] = type == DataType.FLOAT16
                ? Float.floatToFloat16(values[i]) : ScalarValue.bfloat16(values[i]).bfloat16Bits();
        return result;
    }

    private static void compare(AggregateReductionKind kind, DataType type, Shape inputShape,
            OperationAttrs attrs, Shape outputShape, Object input) throws Throwable {
        { Object actual = invoke(kind, type, inputShape, attrs, outputShape, input);
        Object expected = java.lang.reflect.Array.newInstance(input.getClass().componentType(),
                Math.toIntExact(outputShape.knownElementCount().orElseThrow()));
        var lowered = CpuAggregateLoweringTest.lower(kind, type, inputShape, attrs, outputShape);
        CpuScalarReferenceKernel.execute((CpuAggregateIr) lowered.portableKernelIr(),
                lowered.aggregateGeometry().orElseThrow(), List.of(argument(type,input,true),
                    argument(type,expected,false)));
        if (expected instanceof double[] a) assertArrayEquals(
                java.util.Arrays.stream(a).mapToLong(Double::doubleToRawLongBits).toArray(),
                java.util.Arrays.stream((double[]) actual).mapToLong(Double::doubleToRawLongBits)
                        .toArray());
        else if (expected instanceof float[] a) {
            int[] expectedBits = new int[a.length], actualBits = new int[a.length];
            for (int index = 0; index < a.length; index++) {
                expectedBits[index] = Float.floatToRawIntBits(a[index]);
                actualBits[index] = Float.floatToRawIntBits(((float[]) actual)[index]);
            }
            assertArrayEquals(expectedBits, actualBits);
        }
        else if (expected instanceof short[] a) assertArrayEquals(a,(short[])actual);
        else if (expected instanceof int[] a) assertArrayEquals(a,(int[])actual);
        else if (expected instanceof long[] a) assertArrayEquals(a,(long[])actual);
        else assertArrayEquals((byte[])expected,(byte[])actual); }
    }

    private static Object invoke(AggregateReductionKind kind, DataType type, Shape inputShape,
            OperationAttrs attrs, Shape outputShape, Object input) throws Throwable {
        var base = CpuAggregateLoweringTest.context(kind,type,inputShape,attrs,outputShape);
        CarrierAccess carrier = carrier(type);
        PrepareContext<CpuPartitionAnalysisInputs> context = new PrepareContext<>(base.partition(), base.nodes(), base.values(), base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false,List.of(carrier,carrier)));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan(); var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(), route.kernelIr(), generator.generateClassBytes(route.specialization(), route.kernelIr()));
        Object output = java.lang.reflect.Array.newInstance(input.getClass().componentType(),
                Math.toIntExact(outputShape.knownElementCount().orElseThrow()));
        long[] packed = plan.aggregateGeometry().orElseThrow().pack(new long[2]);
        if (route.specialization().scratchParameter()) {
            try (var arena = Arena.ofConfined()) {
                long bytes = plan.aggregateGeometry().orElseThrow().workspaceBytes(1);
                artifact.entryPoint().invokeWithArguments(input,output,arena.allocate(bytes,8),
                        packed,0L,plan.elementCount());
            }
        } else artifact.entryPoint().invokeWithArguments(input,output,packed,0L,plan.elementCount());
        return output;
    }

    private static CarrierAccess carrier(DataType type) { return switch(type) {
        case FLOAT64 -> CarrierAccess.DOUBLE_ARRAY; case FLOAT32 -> CarrierAccess.FLOAT_ARRAY;
        case BFLOAT16, FLOAT16 -> CarrierAccess.SHORT_ARRAY; case INT32 -> CarrierAccess.INT_ARRAY;
        case INT64 -> CarrierAccess.LONG_ARRAY; case BOOL, FLOAT8_E4M3, FLOAT8_E5M2 -> CarrierAccess.BYTE_ARRAY;
        }; }
    private static CpuBufferArgument argument(DataType type,Object value,boolean readOnly) {
        return switch(type) {
            case FLOAT64 -> new CpuBufferArgument.Doubles((double[])value,0,((double[])value).length*8L,readOnly);
            case FLOAT32 -> new CpuBufferArgument.Floats((float[])value,0,((float[])value).length*4L,readOnly);
            case BFLOAT16, FLOAT16 -> new CpuBufferArgument.Shorts((short[])value,0,((short[])value).length*2L,readOnly);
            case INT32 -> new CpuBufferArgument.Ints((int[])value,0,((int[])value).length*4L,readOnly);
            case INT64 -> new CpuBufferArgument.Longs((long[])value,0,((long[])value).length*8L,readOnly);
            case BOOL, FLOAT8_E4M3, FLOAT8_E5M2 -> new CpuBufferArgument.Bytes((byte[])value,0,((byte[])value).length,readOnly);
        };
    }

    private static Object adversarialFloatingValues(DataType type, int count) {
        long[] doubleBits = {
                0x0000000000000001L, 0x8000000000000001L,
                0x0010000000000000L, 0x8010000000000000L,
                0x3ff0000000000000L, 0xbff0000000000000L,
                0x3ca0000000000000L, 0x7fefffffffffffffL,
                0xffefffffffffffffL, 0x0000000000000000L,
                0x8000000000000000L, 0x7ff0000000000000L,
                0xfff0000000000000L, 0x7ff0000000000042L,
                0xfff8000000000042L, 0x4008000000000000L,
                0xbfe0000000000000L
        };
        int[] floatBits = {
                0x00000001, 0x80000001, 0x00800000, 0x80800000,
                0x3f800000, 0xbf800000, 0x34000000, 0x7f7fffff,
                0xff7fffff, 0x00000000, 0x80000000, 0x7f800000,
                0xff800000, 0x7f800042, 0xffc00042, 0x40400000,
                0xbf000000
        };
        int[] bfloatBits = {
                0x0001, 0x8001, 0x0080, 0x8080, 0x3f80, 0xbf80,
                0x3a00, 0x7f7f, 0xff7f, 0x0000, 0x8000, 0x7f80,
                0xff80, 0x7f81, 0xffc2, 0x4040, 0xbf00
        };
        if (type == DataType.FLOAT64) {
            double[] values = new double[count];
            for (int index = 0; index < count; index++)
                values[index] = Double.longBitsToDouble(doubleBits[index % doubleBits.length]);
            return values;
        }
        if (type == DataType.FLOAT32) {
            float[] values = new float[count];
            for (int index = 0; index < count; index++)
                values[index] = Float.intBitsToFloat(floatBits[index % floatBits.length]);
            return values;
        }
        if (type == DataType.FLOAT8_E4M3 || type == DataType.FLOAT8_E5M2) {
            byte[] values = new byte[count];
            for (int index = 0; index < count; index++) values[index] = (byte) index;
            return values;
        }
        int[] halfBits = {0x0001, 0x8001, 0x0400, 0x8400, 0x3c00, 0xbc00,
                0x1000, 0x7bff, 0xfbff, 0x0000, 0x8000, 0x7c00, 0xfc00, 0x7c01, 0xfe42};
        int[] selectedBits = type == DataType.FLOAT16 ? halfBits : bfloatBits;
        short[] values = new short[count];
        for (int index = 0; index < count; index++)
            values[index] = (short) selectedBits[index % selectedBits.length];
        return values;
    }

    private static void assertStrictArtifact(AggregateReductionKind kind, DataType type) {
        var base = CpuAggregateLoweringTest.context(kind, type, Shape.of(8),
                NoOperationAttrs.INSTANCE, Shape.scalar());
        CarrierAccess carrier = carrier(type);
        PrepareContext<CpuPartitionAnalysisInputs> context = new PrepareContext<>(base.partition(), base.nodes(), base.values(), base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false, List.of(carrier, carrier)));
        
        var route = new CpuPartitionPreparer().analyze(context).plan().units().getFirst().portablePlan();
        var model = ClassFile.of().parse(new CpuClassFileKernelGenerator().generateClassBytes(
                route.specialization(), route.kernelIr()));
        List<MemberRefEntry> members = java.util.stream.StreamSupport.stream(
                model.constantPool().spliterator(), false).filter(MemberRefEntry.class::isInstance)
                .map(MemberRefEntry.class::cast).toList();
        String carrierDescriptor = switch (type) {
            case FLOAT64 -> "[D"; case FLOAT32 -> "[F"; case BFLOAT16, FLOAT16 -> "[S";
            case INT32 -> "[I"; case INT64 -> "[J"; case BOOL, FLOAT8_E4M3, FLOAT8_E5M2 -> "[B";
        };
        assertAll(kind + " " + type,
                () -> assertEquals("(" + carrierDescriptor + carrierDescriptor + "[JJJ)V",
                        model.methods().getFirst().methodTypeSymbol().descriptorString()),
                () -> assertTrue(members.stream().noneMatch(entry -> entry.owner().asInternalName()
                        .startsWith("io/github/pho001/synaptik"))),
                () -> assertTrue(members.stream().noneMatch(entry -> entry.type().stringValue()
                        .contains("Ljava/lang/Object;") || entry.owner().asInternalName()
                                .startsWith("java/lang/reflect/") || entry.owner().asInternalName()
                                .startsWith("java/util/"))),
                () -> assertEquals(0, model.constantPool().bootstrapMethodCount()),
                () -> assertTrue(java.util.stream.StreamSupport.stream(
                        model.constantPool().spliterator(), false)
                        .noneMatch(MethodHandleEntry.class::isInstance)),
                () -> assertTrue(java.util.stream.StreamSupport.stream(
                        model.constantPool().spliterator(), false)
                        .noneMatch(DynamicConstantPoolEntry.class::isInstance)),
                () -> assertTrue(model.methods().stream().flatMap(method -> method.code().stream())
                        .flatMap(code -> code.elementStream()).noneMatch(
                                java.lang.classfile.instruction.NewObjectInstruction.class::isInstance)));
    }

    private static long opcodeCount(java.lang.classfile.CodeModel code, Opcode opcode) {
        return code.elementStream().filter(Instruction.class::isInstance)
                .map(Instruction.class::cast).filter(instruction -> instruction.opcode() == opcode)
                .count();
    }

    private static void assertBoundedNumericalArtifact(java.lang.classfile.ClassModel model,
            String descriptor, boolean segmentInput) {
        var method = model.methods().getFirst();
        var code = method.code().orElseThrow();
        List<MemberRefEntry> members = java.util.stream.StreamSupport.stream(
                model.constantPool().spliterator(), false).filter(MemberRefEntry.class::isInstance)
                .map(MemberRefEntry.class::cast).toList();
        assertAll(
                () -> assertEquals(0, model.fields().size()),
                () -> assertEquals(1, model.methods().size()),
                () -> assertEquals(descriptor, method.methodTypeSymbol().descriptorString()),
                () -> assertTrue(hasBoundedInputLoop(code, segmentInput)),
                () -> assertTrue(opcodeCount(code, Opcode.LALOAD) > 0),
                () -> assertTrue(opcodeCount(code, Opcode.LDIV) > 0),
                () -> assertTrue(opcodeCount(code, Opcode.LREM) > 0),
                () -> assertTrue(members.stream().noneMatch(entry -> entry.owner().asInternalName()
                        .startsWith("io/github/pho001/synaptik"))),
                () -> assertTrue(members.stream().noneMatch(entry -> entry.type().stringValue()
                        .contains("Ljava/lang/Object;") || entry.owner().asInternalName()
                                .startsWith("java/lang/reflect/") || entry.owner().asInternalName()
                                .startsWith("java/util/"))),
                () -> assertEquals(0, model.constantPool().bootstrapMethodCount()),
                () -> assertTrue(java.util.stream.StreamSupport.stream(
                        model.constantPool().spliterator(), false)
                        .noneMatch(MethodHandleEntry.class::isInstance)),
                () -> assertTrue(java.util.stream.StreamSupport.stream(
                        model.constantPool().spliterator(), false)
                        .noneMatch(DynamicConstantPoolEntry.class::isInstance)),
                () -> assertTrue(code.elementStream().noneMatch(
                        java.lang.classfile.instruction.NewObjectInstruction.class::isInstance)));
    }

    private static boolean hasBoundedInputLoop(java.lang.classfile.CodeModel code,
            boolean segmentInput) {
        var elements = code.elementStream().toList();
        var labels = new java.util.IdentityHashMap<java.lang.classfile.Label, Integer>();
        for (int index = 0; index < elements.size(); index++) {
            if (elements.get(index) instanceof LabelTarget target) labels.put(target.label(), index);
        }
        for (int index = 0; index < elements.size(); index++) {
            if (!(elements.get(index) instanceof BranchInstruction branch)) continue;
            Integer target = labels.get(branch.target());
            if (target == null || target >= index) continue;
            boolean inputLoad = false, genericAddressing = false;
            for (int body = target; body <= index; body++) {
                Object element = elements.get(body);
                if (element instanceof InvokeInstruction invoke) {
                    inputLoad |= segmentInput
                            && invoke.owner().asInternalName().equals(
                                    "java/lang/foreign/MemorySegment")
                            && invoke.name().stringValue().equals("get")
                            && invoke.type().stringValue().endsWith(")F");
                }
                if (element instanceof Instruction instruction) {
                    inputLoad |= !segmentInput && instruction.opcode() == Opcode.SALOAD;
                    genericAddressing |= instruction.opcode() == Opcode.LALOAD
                            || instruction.opcode() == Opcode.LDIV
                            || instruction.opcode() == Opcode.LREM;
                }
            }
            if (inputLoad && !genericAddressing) return true;
        }
        return false;
    }

    private static boolean hasBoundedBfloatMinimumLoop(java.lang.classfile.CodeModel code) {
        var elements = code.elementStream().toList();
        var labels = new java.util.IdentityHashMap<java.lang.classfile.Label, Integer>();
        for (int index = 0; index < elements.size(); index++) {
            if (elements.get(index) instanceof LabelTarget target) labels.put(target.label(), index);
        }
        for (int index = 0; index < elements.size(); index++) {
            if (!(elements.get(index) instanceof BranchInstruction branch)) continue;
            Integer target = labels.get(branch.target());
            if (target == null || target >= index) continue;
            boolean inputLoad = false, comparison = false, genericAddressing = false;
            for (int body = target; body <= index; body++) {
                Object element = elements.get(body);
                if (element instanceof InvokeInstruction invoke) {
                    inputLoad |= invoke.owner().asInternalName().equals(
                            "java/lang/foreign/MemorySegment")
                            && invoke.name().stringValue().equals("get")
                            && invoke.type().stringValue().endsWith(")S");
                }
                if (element instanceof Instruction instruction) {
                    comparison |= instruction.opcode() == Opcode.FCMPG;
                    genericAddressing |= instruction.opcode() == Opcode.LALOAD
                            || instruction.opcode() == Opcode.LDIV
                            || instruction.opcode() == Opcode.LREM;
                }
            }
            if (inputLoad && comparison && !genericAddressing) return true;
        }
        return false;
    }

    private static boolean hasDirectFullVisitLoop(java.lang.classfile.CodeModel code) {
        var elements = code.elementStream().toList();
        var labels = new java.util.IdentityHashMap<java.lang.classfile.Label, Integer>();
        for (int index = 0; index < elements.size(); index++) {
            if (elements.get(index) instanceof LabelTarget target) labels.put(target.label(), index);
        }
        for (int index = 0; index < elements.size(); index++) {
            if (!(elements.get(index) instanceof BranchInstruction branch)) continue;
            Integer target = labels.get(branch.target());
            if (target == null || target >= index) continue;
            boolean get = false, or = false, geometryLoad = false, accumulatorExit = false;
            for (int body = target; body <= index; body++) {
                Object element = elements.get(body);
                if (element instanceof InvokeInstruction invoke) {
                    get |= invoke.owner().asInternalName().equals("java/lang/foreign/MemorySegment")
                            && invoke.name().stringValue().equals("get");
                }
                if (element instanceof Instruction instruction) {
                    or |= instruction.opcode() == Opcode.IOR;
                    geometryLoad |= instruction.opcode() == Opcode.LALOAD;
                    accumulatorExit |= instruction.opcode() == Opcode.IFNE
                            || instruction.opcode() == Opcode.IFEQ;
                }
            }
            if (get && or && !geometryLoad && !accumulatorExit) return true;
        }
        return false;
    }
}

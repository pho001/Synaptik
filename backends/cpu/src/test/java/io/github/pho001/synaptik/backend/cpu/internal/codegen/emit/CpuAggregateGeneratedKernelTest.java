package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAggregateLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuIndexingLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.backend.cpu.internal.reference.CpuScalarReferenceKernel;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.List;
import java.util.Map;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;
import java.lang.classfile.ClassFile;
import java.lang.classfile.Instruction;
import java.lang.classfile.Opcode;
import java.lang.classfile.instruction.InvokeInstruction;

class CpuAggregateGeneratedKernelTest {
    @Test void generatedClassContainsTypedAggregateFoldWithoutGenericDispatchBridge() {
        var base = CpuAggregateLoweringTest.context(AggregateReductionKind.MIN, DataType.FLOAT64,
                Shape.of(1024), NoOperationAttrs.INSTANCE, Shape.scalar());
        PrepareContext<CpuPartitionAnalysisInputs> context = new PrepareContext<>(base.partition(),
                base.nodes(), base.values(), base.memoryRequirements(), Map.of(),
                new CpuPartitionAnalysisInputs(false,
                        List.of(CarrierAccess.DOUBLE_ARRAY, CarrierAccess.DOUBLE_ARRAY)));
        var route = new CpuPartitionPreparer().analyze(context).plan().units().getFirst().portablePlan();
        var code = ClassFile.of().parse(new CpuClassFileKernelGenerator().generateClassBytes(
                route.specialization(), route.kernelIr())).methods().getFirst().code().orElseThrow();
        var invokes = code.elementStream().filter(InvokeInstruction.class::isInstance)
                .map(InvokeInstruction.class::cast).toList();
        assertAll(() -> assertTrue(invokes.stream().noneMatch(i -> i.name().stringValue().equals("execute"))),
                () -> assertTrue(invokes.stream().noneMatch(i -> i.type().stringValue().contains("Ljava/lang/Object;"))),
                () -> assertTrue(invokes.stream().anyMatch(i -> i.name().stringValue().equals("minDouble"))),
                () -> assertTrue(code.elementStream().filter(Instruction.class::isInstance)
                        .map(Instruction.class::cast).anyMatch(i -> i.opcode() == Opcode.DALOAD)),
                () -> assertTrue(code.elementStream().filter(Instruction.class::isInstance)
                        .map(Instruction.class::cast).anyMatch(i -> i.opcode() == Opcode.DASTORE)),
                () -> assertTrue(code.elementStream().noneMatch(element -> element instanceof java.lang.classfile.instruction.NewObjectInstruction)));
    }

    @Test void executesAllKindsAndRepresentedTypesWithExactIdentities() throws Throwable {
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

    @Test void generatedResultsMatchIndependentReferenceIncludingEmptyAxisPointForm() throws Throwable {
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
        PrepareContext<CpuPartitionAnalysisInputs> context = new PrepareContext<>(base.partition(),
                base.nodes(),base.values(),base.memoryRequirements(),Map.of(),
                new CpuPartitionAnalysisInputs(false,List.of(CarrierAccess.MEMORY_SEGMENT,
                        CarrierAccess.FLOAT_ARRAY)));
        var plan = new CpuPartitionPreparer().analyze(context).plan(); var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator(); var artifact = generator.defineClassBytes(
                route.specialization(),generator.generateClassBytes(route.specialization(),route.kernelIr()));
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

    @Test void zeroStrideReadsHaveBitwiseFullAndDisjointRangeParity() throws Throwable {
        Shape inputShape=Shape.of(8,2), outputShape=Shape.of(8);
        var inputDescriptor=CpuIndexingLoweringTest.descriptor(DataType.FLOAT32,inputShape,
                LayoutDescriptor.of(inputShape,new long[]{1,0},0,true));
        var outputDescriptor=CpuScatterLoweringTest.desc(DataType.FLOAT32,outputShape);
        var base=CpuScatterLoweringTest.context(new Operation(AggregateReductionKind.MIN,
                new AxisReductionAttrs(1,false)),List.of(0),List.of(inputDescriptor),outputDescriptor);
        PrepareContext<CpuPartitionAnalysisInputs> context=new PrepareContext<>(base.partition(),
                base.nodes(),base.values(),base.memoryRequirements(),Map.of(),
                new CpuPartitionAnalysisInputs(false,List.of(CarrierAccess.FLOAT_ARRAY,
                        CarrierAccess.FLOAT_ARRAY)));
        var plan=new CpuPartitionPreparer().analyze(context).plan(); var route=plan.units().getFirst().portablePlan();
        var generator=new CpuClassFileKernelGenerator(); var artifact=generator.defineClassBytes(
                route.specialization(),generator.generateClassBytes(route.specialization(),route.kernelIr()));
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

    private static void compare(AggregateReductionKind kind, DataType type, Shape inputShape,
            OperationAttrs attrs, Shape outputShape, Object input) throws Throwable {
        Object actual = invoke(kind, type, inputShape, attrs, outputShape, input);
        Object expected = java.lang.reflect.Array.newInstance(input.getClass().componentType(),
                Math.toIntExact(outputShape.knownElementCount().orElseThrow()));
        var lowered = CpuAggregateLoweringTest.lower(kind, type, inputShape, attrs, outputShape);
        CpuScalarReferenceKernel.execute((CpuAggregateIr) lowered.portableKernelIr(),
                lowered.aggregateGeometry().orElseThrow(), List.of(argument(type,input,true),
                    argument(type,expected,false)));
        if (expected instanceof double[] a) assertArrayEquals(a,(double[])actual);
        else if (expected instanceof float[] a) assertArrayEquals(a,(float[])actual);
        else if (expected instanceof short[] a) assertArrayEquals(a,(short[])actual);
        else if (expected instanceof int[] a) assertArrayEquals(a,(int[])actual);
        else if (expected instanceof long[] a) assertArrayEquals(a,(long[])actual);
        else assertArrayEquals((byte[])expected,(byte[])actual);
    }

    private static Object invoke(AggregateReductionKind kind, DataType type, Shape inputShape,
            OperationAttrs attrs, Shape outputShape, Object input) throws Throwable {
        var base = CpuAggregateLoweringTest.context(kind,type,inputShape,attrs,outputShape);
        CarrierAccess carrier = carrier(type);
        PrepareContext<CpuPartitionAnalysisInputs> context = new PrepareContext<>(base.partition(),
                base.nodes(), base.values(), base.memoryRequirements(), Map.of(),
                new CpuPartitionAnalysisInputs(false,List.of(carrier,carrier)));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan(); var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(),
                generator.generateClassBytes(route.specialization(),route.kernelIr()));
        Object output = java.lang.reflect.Array.newInstance(input.getClass().componentType(),
                Math.toIntExact(outputShape.knownElementCount().orElseThrow()));
        long[] packed = plan.aggregateGeometry().orElseThrow().pack(new long[2]);
        artifact.entryPoint().invokeWithArguments(input,output,packed,0L,plan.elementCount());
        return output;
    }

    private static CarrierAccess carrier(DataType type) { return switch(type) {
        case FLOAT64 -> CarrierAccess.DOUBLE_ARRAY; case FLOAT32 -> CarrierAccess.FLOAT_ARRAY;
        case BFLOAT16 -> CarrierAccess.SHORT_ARRAY; case INT32 -> CarrierAccess.INT_ARRAY;
        case INT64 -> CarrierAccess.LONG_ARRAY; case BOOL -> CarrierAccess.BYTE_ARRAY; }; }
    private static CpuBufferArgument argument(DataType type,Object value,boolean readOnly) {
        return switch(type) {
            case FLOAT64 -> new CpuBufferArgument.Doubles((double[])value,0,((double[])value).length*8L,readOnly);
            case FLOAT32 -> new CpuBufferArgument.Floats((float[])value,0,((float[])value).length*4L,readOnly);
            case BFLOAT16 -> new CpuBufferArgument.Shorts((short[])value,0,((short[])value).length*2L,readOnly);
            case INT32 -> new CpuBufferArgument.Ints((int[])value,0,((int[])value).length*4L,readOnly);
            case INT64 -> new CpuBufferArgument.Longs((long[])value,0,((long[])value).length*8L,readOnly);
            case BOOL -> new CpuBufferArgument.Bytes((byte[])value,0,((byte[])value).length,readOnly);
        };
    }
}

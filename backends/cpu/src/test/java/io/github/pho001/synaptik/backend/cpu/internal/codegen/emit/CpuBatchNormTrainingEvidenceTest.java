package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratorSchema;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuBatchNormTrainingLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.classfile.ClassFile;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Frozen specialization, member, and retained Class-File evidence for CPU 0007F2. */
public final class CpuBatchNormTrainingEvidenceTest {
    private record Target(String name, List<DataType> types, Shape shape, int axis,
            List<Integer> occurrences, List<CarrierAccess> carriers,
            List<LayoutDescriptor> layouts, int parallelism) { }

    @Test void frozenEightTargetInventoryHasClosedGeneratedShapeAndMembers() throws Exception {
        assertEquals(51, CpuGeneratorSchema.CURRENT_VERSION);
        List<Target> targets = targets();
        assertEquals(List.of("BNT-BF16-A1", "BNT-F32-A1", "BNT-F64-A1", "BNT-F32-A0",
                "BNT-F32-A2", "BNT-MIX-F64", "BNT-MIX-F32", "BNT-REPEAT"),
                targets.stream().map(Target::name).toList());
        for (Target target : targets) inspect(target);
    }

    private static void inspect(Target target) throws Exception {
        var base = CpuBatchNormTrainingLoweringTest.context(target.types, target.shape,
                target.axis, target.occurrences, target.layouts);
        var config = new CpuPartitionAnalysisInputs.PortableExecutionConfig(
                CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR,
                target.parallelism, target.parallelism, 4096);
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), base.constants(),
                new CpuPartitionAnalysisInputs(false, target.carriers, config));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        byte[] first = generator.generateClassBytes(route.specialization(), route.kernelIr());
        byte[] second = generator.generateClassBytes(route.specialization(), route.kernelIr());
        assertArrayEquals(first, second, target.name);
        var model = ClassFile.of().parse(first);
        assertEquals(1, model.methods().size(), target.name);
        assertEquals(0, model.fields().size(), target.name);
        var method = model.methods().getFirst();
        assertEquals("invoke", method.methodName().stringValue(), target.name);
        assertTrue(method.flags().has(java.lang.reflect.AccessFlag.PUBLIC));
        assertTrue(method.flags().has(java.lang.reflect.AccessFlag.STATIC));
        StringBuilder members = new StringBuilder();
        java.util.stream.StreamSupport.stream(model.constantPool().spliterator(), false)
                .filter(MemberRefEntry.class::isInstance).map(MemberRefEntry.class::cast)
                .forEach(member -> members.append(member.owner().asInternalName()).append('.')
                        .append(member.name().stringValue()).append(member.type().stringValue())
                        .append('\n'));
        String references = members.toString();
        assertTrue(references.contains("java/lang/Math.sqrt"), target.name);
        assertFalse(references.contains("synaptik"), target.name);
        assertFalse(references.contains("java/lang/Object"), target.name);
        assertFalse(references.contains("java/util/"), target.name);
        assertFalse(references.contains("java/lang/reflect"), target.name);
        assertFalse(references.contains("java/lang/invoke"), target.name);
        assertFalse(references.contains("java/nio/ByteOrder"), target.name);
        assertFalse(references.contains("ValueLayout.withOrder"), target.name);
        retain(target.name, first, route.specialization().compatibilityBytes(),
                route.specialization().toString(), method.methodTypeSymbol().displayDescriptor(),
                references, plan.selectedRangeCount());
    }

    private static List<Target> targets() {
        var result = new ArrayList<Target>();
        result.add(dense("BNT-BF16-A1", DataType.BFLOAT16, Shape.of(32,64,256),1,4));
        result.add(dense("BNT-F32-A1", DataType.FLOAT32, Shape.of(32,64,256),1,4));
        result.add(dense("BNT-F64-A1", DataType.FLOAT64, Shape.of(32,64,256),1,1));
        result.add(dense("BNT-F32-A0", DataType.FLOAT32, Shape.of(128,4096),0,4));
        result.add(dense("BNT-F32-A2", DataType.FLOAT32, Shape.of(32,256,64),2,4));
        result.add(mixed("BNT-MIX-F64", true));
        result.add(mixed("BNT-MIX-F32", false));
        Shape repeatShape=Shape.of(32,64,128), vector=Shape.of(64);
        var repeatLayouts=layouts(repeatShape,vector,new long[]{8192,128,1},0,new long[]{8192,128,1},0,
                new long[]{2,3,4,5},new long[]{2,2,2,2});
        result.add(new Target("BNT-REPEAT",Collections.nCopies(5,DataType.FLOAT32),repeatShape,1,
                List.of(0,1,1,1,1),List.of(CarrierAccess.FLOAT_ARRAY,CarrierAccess.FLOAT_ARRAY,
                CarrierAccess.FLOAT_ARRAY,CarrierAccess.MEMORY_SEGMENT,CarrierAccess.FLOAT_ARRAY,
                CarrierAccess.MEMORY_SEGMENT,CarrierAccess.FLOAT_ARRAY),repeatLayouts,4));
        return result;
    }

    private static Target dense(String name, DataType type, Shape shape, int axis, int parallelism) {
        Shape vector=Shape.of(shape.toLongArray()[axis]);var layouts=new ArrayList<LayoutDescriptor>();
        layouts.add(LayoutDescriptor.contiguous(shape));for(int i=1;i<5;i++)layouts.add(LayoutDescriptor.contiguous(vector));
        layouts.add(LayoutDescriptor.contiguous(shape));for(int i=1;i<5;i++)layouts.add(LayoutDescriptor.contiguous(vector));
        return new Target(name,Collections.nCopies(5,type),shape,axis,List.of(0,1,2,3,4),
                Collections.nCopies(10,array(type)),layouts,parallelism);
    }

    private static Target mixed(String name, boolean f64) {
        Shape shape=Shape.of(16,32,64),vector=Shape.of(32);
        List<DataType> types=f64?List.of(DataType.FLOAT32,DataType.FLOAT64,DataType.BFLOAT16,DataType.FLOAT32,DataType.FLOAT64)
                :List.of(DataType.BFLOAT16,DataType.FLOAT32,DataType.BFLOAT16,DataType.FLOAT32,DataType.BFLOAT16);
        var layouts=layouts(shape,vector,new long[]{5000,137,2},11,new long[]{6000,151,2},13,
                new long[]{3,5,7,11},new long[]{2,1,3,1});
        List<CarrierAccess> carriers=f64?List.of(CarrierAccess.FLOAT_ARRAY,CarrierAccess.MEMORY_SEGMENT,
                CarrierAccess.SHORT_ARRAY,CarrierAccess.MEMORY_SEGMENT,CarrierAccess.DOUBLE_ARRAY,
                CarrierAccess.MEMORY_SEGMENT,CarrierAccess.DOUBLE_ARRAY,CarrierAccess.MEMORY_SEGMENT,
                CarrierAccess.DOUBLE_ARRAY,CarrierAccess.MEMORY_SEGMENT)
                :List.of(CarrierAccess.MEMORY_SEGMENT,CarrierAccess.FLOAT_ARRAY,CarrierAccess.MEMORY_SEGMENT,
                CarrierAccess.FLOAT_ARRAY,CarrierAccess.SHORT_ARRAY,CarrierAccess.FLOAT_ARRAY,
                CarrierAccess.MEMORY_SEGMENT,CarrierAccess.FLOAT_ARRAY,CarrierAccess.MEMORY_SEGMENT,
                CarrierAccess.FLOAT_ARRAY);
        return new Target(name,types,shape,1,List.of(0,1,2,3,4),carriers,layouts,4);
    }

    private static List<LayoutDescriptor> layouts(Shape shape,Shape vector,long[] inputStrides,
            long inputOffset,long[] outputStrides,long outputOffset,long[] vectorOffsets,long[] vectorStrides){
        var result=new ArrayList<LayoutDescriptor>();result.add(LayoutDescriptor.of(shape,inputStrides,inputOffset,true));
        for(int i=0;i<4;i++)result.add(LayoutDescriptor.of(vector,new long[]{vectorStrides[i]},vectorOffsets[i],true));
        result.add(LayoutDescriptor.of(shape,outputStrides,outputOffset,true));
        for(int i=0;i<4;i++)result.add(LayoutDescriptor.of(vector,new long[]{vectorStrides[i]+1},vectorOffsets[i]+1,true));
        return result;
    }

    private static CarrierAccess array(DataType type){return switch(type){case BFLOAT16->CarrierAccess.SHORT_ARRAY;
        case FLOAT32->CarrierAccess.FLOAT_ARRAY;case FLOAT64->CarrierAccess.DOUBLE_ARRAY;default->throw new IllegalArgumentException();};}

    private static void retain(String name,byte[] bytes,byte[] compatibility,String specialization,
            String descriptor,String members,int ranges)throws Exception{
        String root=System.getProperty("synaptik.cpu.batchnorm.training.evidence");if(root==null)root=System.getenv("SYNAPTIK_CPU_BATCHNORM_TRAINING_EVIDENCE");if(root==null)return;
        Path generated=Path.of(root,"generated");Files.createDirectories(generated);
        Files.write(generated.resolve(name+".class"),bytes);Files.write(generated.resolve(name+".compatibility"),compatibility);
        Files.writeString(generated.resolve(name+".specialization"),specialization+'\n');Files.writeString(generated.resolve(name+".descriptor"),descriptor+'\n');
        Files.writeString(generated.resolve(name+".members"),members);Files.writeString(generated.resolve(name+".ranges"),Integer.toString(ranges)+'\n');
    }
}

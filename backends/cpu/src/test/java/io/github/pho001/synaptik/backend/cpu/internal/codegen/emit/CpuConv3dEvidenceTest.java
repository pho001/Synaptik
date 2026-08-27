package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratorSchema;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuConv3dLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.io.InputStream;
import java.lang.classfile.ClassFile;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Frozen generated-form, deterministic Class-File, and compatibility evidence for CPU 0008A. */
public final class CpuConv3dEvidenceTest {
    private static final Path ROOT=Path.of("/private/tmp/synaptik-cpu-0008a-retained-evidence-20260826");
    private static final Path GENERATED=ROOT.resolve("generated");
    private record Target(String name,PrepareContext<CpuPartitionAnalysisInputs> context){}

    @Test void retainsClosedRepresentativeGeneratedInventory()throws Exception{
        assertEquals(53,CpuGeneratorSchema.CURRENT_VERSION);Files.createDirectories(GENERATED);
        try(InputStream input=getClass().getResourceAsStream("/cpu-0008a-generated-form-ledger.csv")){assertNotNull(input);Files.copy(input,ROOT.resolve("generated-form-ledger.csv"),java.nio.file.StandardCopyOption.REPLACE_EXISTING);}
        var targets=targets();assertEquals(List.of("CONV3D-DENSE-F32","CONV3D-GROUPED-BIASED-F32","CONV3D-GROUPED-BIASED-F64","CONV3D-DEPTHWISE-BF16","CONV3D-ALL-SEGMENT-F32","CONV3D-GENERAL-MIXED","CONV3D-PARALLEL-F32"),targets.stream().map(Target::name).toList());
        for(Target target:targets)inspect(target);
    }

    private static void inspect(Target target)throws Exception{
        var unit=new CpuPartitionPreparer().analyze(target.context).plan().units().getFirst();var route=unit.portablePlan();var generator=new CpuClassFileKernelGenerator();byte[]bytes=generator.generateClassBytes(route.specialization(),route.kernelIr());assertArrayEquals(bytes,generator.generateClassBytes(route.specialization(),route.kernelIr()),target.name);
        var model=ClassFile.of().parse(bytes);StringBuilder members=new StringBuilder();java.util.stream.StreamSupport.stream(model.constantPool().spliterator(),false).filter(MemberRefEntry.class::isInstance).map(MemberRefEntry.class::cast).forEach(member->members.append(member.owner().asInternalName()).append('.').append(member.name().stringValue()).append(member.type().stringValue()).append('\n'));String references=members.toString();
        assertAll(target.name,()->assertTrue(model.flags().has(java.lang.reflect.AccessFlag.FINAL)),()->assertTrue(model.fields().isEmpty()),()->assertEquals(1,model.methods().size()),()->assertEquals("invoke",model.methods().getFirst().methodName().stringValue()),()->assertTrue(model.methods().getFirst().flags().has(java.lang.reflect.AccessFlag.STATIC)),()->assertFalse(references.contains("synaptik")),()->assertFalse(references.contains("java/lang/reflect")),()->assertFalse(references.contains("java/lang/invoke")),()->assertFalse(references.contains("java/util/")),()->assertFalse(references.contains(".valueOf")));
        retain(target.name,bytes,route.specialization().compatibilityBytes(),route.specialization().toString(),model.methods().getFirst().methodTypeSymbol().displayDescriptor(),references);
    }

    private static List<Target>targets(){var result=new ArrayList<Target>();result.add(target("CONV3D-DENSE-F32",List.of(DataType.FLOAT32,DataType.FLOAT32),Shape.of(1,4,10,10,10),Shape.of(8,4,3,3,3),Shape.of(1,8,8,8,8),Conv3dAttrs.defaults(),null,Collections.nCopies(3,CarrierAccess.FLOAT_ARRAY),1));result.add(target("CONV3D-GROUPED-BIASED-F32",Collections.nCopies(3,DataType.FLOAT32),Shape.of(1,8,9,9,9),Shape.of(8,2,3,3,3),Shape.of(1,8,9,9,9),new Conv3dAttrs(1,1,1,1,1,1,1,1,1,4),null,Collections.nCopies(4,CarrierAccess.FLOAT_ARRAY),1));result.add(target("CONV3D-GROUPED-BIASED-F64",Collections.nCopies(3,DataType.FLOAT64),Shape.of(1,4,9,9,9),Shape.of(6,2,3,3,3),Shape.of(1,6,9,9,9),new Conv3dAttrs(1,1,1,1,1,1,1,1,1,2),null,Collections.nCopies(4,CarrierAccess.DOUBLE_ARRAY),1));result.add(target("CONV3D-DEPTHWISE-BF16",Collections.nCopies(2,DataType.BFLOAT16),Shape.of(1,8,8,8,8),Shape.of(8,1,3,3,3),Shape.of(1,8,8,8,8),new Conv3dAttrs(1,1,1,1,1,1,1,1,1,8),null,Collections.nCopies(3,CarrierAccess.SHORT_ARRAY),1));result.add(target("CONV3D-ALL-SEGMENT-F32",List.of(DataType.FLOAT32,DataType.FLOAT32),Shape.of(1,2,8,8,8),Shape.of(3,2,3,3,3),Shape.of(1,3,6,6,6),Conv3dAttrs.defaults(),null,Collections.nCopies(3,CarrierAccess.MEMORY_SEGMENT),1));
        Shape x=Shape.of(1,4,7,8,9),w=Shape.of(6,2,2,2,2),y=Shape.of(1,6,6,7,8);var layouts=List.of(LayoutDescriptor.of(x,new long[]{2600,641,89,10,1},7,true),LayoutDescriptor.of(w,new long[]{41,19,9,4,2},3,true),LayoutDescriptor.of(y,new long[]{3600,587,83,9,1},5,true));result.add(target("CONV3D-GENERAL-MIXED",List.of(DataType.BFLOAT16,DataType.FLOAT64),x,w,y,new Conv3dAttrs(1,1,1,0,0,0,1,1,1,2),layouts,List.of(CarrierAccess.MEMORY_SEGMENT,CarrierAccess.DOUBLE_ARRAY,CarrierAccess.MEMORY_SEGMENT),1));result.add(target("CONV3D-PARALLEL-F32",List.of(DataType.FLOAT32,DataType.FLOAT32),Shape.of(2,8,10,10,10),Shape.of(16,8,3,3,3),Shape.of(2,16,8,8,8),Conv3dAttrs.defaults(),null,Collections.nCopies(3,CarrierAccess.FLOAT_ARRAY),4));return List.copyOf(result);}

    private static Target target(String name,List<DataType>types,Shape x,Shape w,Shape y,Conv3dAttrs attrs,List<LayoutDescriptor>layouts,List<CarrierAccess>carriers,int ranges){var base=CpuConv3dLoweringTest.context(types,x,w,y,attrs,layouts);var config=new CpuPartitionAnalysisInputs.PortableExecutionConfig(CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR,ranges,ranges,1);return new Target(name,new PrepareContext<>(base.partition(),base.nodes(),base.values(),base.memoryRequirements(),base.constants(),new CpuPartitionAnalysisInputs(false,carriers,config)));}
    private static void retain(String name,byte[]bytes,byte[]compatibility,String specialization,String descriptor,String members)throws Exception{Files.write(GENERATED.resolve(name+".class"),bytes);Files.write(GENERATED.resolve(name+".compatibility"),compatibility);Files.writeString(GENERATED.resolve(name+".sha256"),hex(bytes)+"\n");Files.writeString(GENERATED.resolve(name+".specialization"),specialization+"\n");Files.writeString(GENERATED.resolve(name+".descriptor"),descriptor+"\n");Files.writeString(GENERATED.resolve(name+".members"),members);}
    private static String hex(byte[]bytes)throws Exception{return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));}
}

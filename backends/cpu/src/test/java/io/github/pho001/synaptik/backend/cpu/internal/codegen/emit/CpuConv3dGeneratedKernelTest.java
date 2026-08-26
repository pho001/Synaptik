package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuConv3dLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.backend.cpu.internal.reference.CpuConv3dReferenceKernel;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.classfile.ClassFile;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.reflect.AccessFlag;
import java.util.List;
import org.junit.jupiter.api.Test;

class CpuConv3dGeneratedKernelTest {
    @Test void emitsDeterministicGroupedBiasedFloatBody() throws Throwable {
        var base=CpuConv3dLoweringTest.context(List.of(DataType.FLOAT32,DataType.FLOAT32,DataType.FLOAT32),Shape.of(1,4,2,2,2),Shape.of(4,2,2,2,2),Shape.of(1,4,3,3,3),new Conv3dAttrs(1,1,1,1,1,1,1,1,1,2),null);
        var context=new PrepareContext<>(base.partition(),base.nodes(),base.values(),base.memoryRequirements(),base.constants(),new CpuPartitionAnalysisInputs(false,java.util.Collections.nCopies(4,CarrierAccess.FLOAT_ARRAY)));
        var plan=new CpuPartitionPreparer().analyze(context).plan();var unit=plan.units().getFirst();var route=unit.portablePlan();var generator=new CpuClassFileKernelGenerator();byte[] bytes=generator.generateClassBytes(route.specialization(),route.kernelIr());assertArrayEquals(bytes,generator.generateClassBytes(route.specialization(),route.kernelIr()));
        var model=ClassFile.of().parse(bytes);StringBuilder members=new StringBuilder();java.util.stream.StreamSupport.stream(model.constantPool().spliterator(),false).filter(MemberRefEntry.class::isInstance).map(MemberRefEntry.class::cast).forEach(m->members.append(m.owner().asInternalName()).append('.').append(m.name().stringValue()).append('\n'));
        assertAll(()->assertTrue(model.flags().has(AccessFlag.FINAL)),()->assertTrue(model.fields().isEmpty()),()->assertEquals(1,model.methods().size()),()->assertFalse(members.toString().contains("synaptik")),()->assertFalse(members.toString().contains("java/lang/reflect")));
        float[] input=new float[32],weight=new float[64],bias={.25f,-.5f,1f,-2f},output=new float[108];for(int i=0;i<input.length;i++)input[i]=i*.125f-2f;for(int i=0;i<weight.length;i++)weight[i]=(i%7-3)*.25f;
        var handle=generator.defineClassBytes(route.specialization(),bytes).entryPoint();handle.invokeExact(input,weight,bias,output,unit.conv3dGeometry().orElseThrow().pack(new long[4]),0L,108L);
        double[] expected=CpuConv3dReferenceKernel.evaluate(List.of(DataType.FLOAT32,DataType.FLOAT32,DataType.FLOAT32),DataType.FLOAT32,new double[][]{doubles(input),doubles(weight),doubles(bias)},new long[][]{{1,4,2,2,2},{4,2,2,2,2},{4}},new long[]{0,0,0},new long[][]{{32,8,4,2,1},{16,8,4,2,1},{1}},new long[]{1,4,3,3,3},new Conv3dAttrs(1,1,1,1,1,1,1,1,1,2));
        for(int i=0;i<output.length;i++)assertEquals(Float.floatToRawIntBits((float)expected[i]),Float.floatToRawIntBits(output[i]),"cell "+i);assertAll(()->assertTrue(plan.workspaceDeclaration().isEmpty()),()->assertTrue(plan.materialization().isEmpty()));
    }

    @Test void conceptualPaddingMultipliesZeroByInfinity() throws Throwable {
        var base=CpuConv3dLoweringTest.context(List.of(DataType.FLOAT64,DataType.FLOAT64),Shape.of(1,1,1,1,1),Shape.of(1,1,1,1,1),Shape.of(1,1,3,3,3),new Conv3dAttrs(1,1,1,1,1,1,1,1,1,1),null);
        var context=new PrepareContext<>(base.partition(),base.nodes(),base.values(),base.memoryRequirements(),base.constants(),new CpuPartitionAnalysisInputs(false,java.util.Collections.nCopies(3,CarrierAccess.DOUBLE_ARRAY)));var plan=new CpuPartitionPreparer().analyze(context).plan();var unit=plan.units().getFirst();var route=unit.portablePlan();var generator=new CpuClassFileKernelGenerator();var handle=generator.defineClassBytes(route.specialization(),generator.generateClassBytes(route.specialization(),route.kernelIr())).entryPoint();double[] output=new double[27];handle.invokeExact(new double[]{2},new double[]{Double.POSITIVE_INFINITY},output,unit.conv3dGeometry().orElseThrow().pack(new long[3]),0L,27L);assertAll(()->assertTrue(Double.isNaN(output[0])),()->assertEquals(Double.POSITIVE_INFINITY,output[13]),()->assertTrue(Double.isNaN(output[26])));
    }

    private static double[] doubles(float[] values){double[] result=new double[values.length];for(int i=0;i<values.length;i++)result[i]=values[i];return result;}
}

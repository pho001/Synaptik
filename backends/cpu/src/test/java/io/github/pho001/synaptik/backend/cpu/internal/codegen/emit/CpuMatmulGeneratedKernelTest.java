package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.*;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.ir.*;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ClampRangeAttrs;
import io.github.pho001.synaptik.backend.cpu.internal.reference.CpuScalarReferenceKernel;
import java.util.List;
import org.junit.jupiter.api.Test;

final class CpuMatmulGeneratedKernelTest {
    @Test void generatedScalarBodyExecutesFullKForFloatingAndIntegralRows() throws Throwable {
        run(DataType.FLOAT32, new float[] {1,2,3,4,5,6}, new float[] {7,8,9,10,11,12},
                new float[4], new float[] {58,64,139,154});
        run(DataType.INT32, new int[] {1,2,3,4,5,6}, new int[] {7,8,9,10,11,12},
                new int[4], new int[] {58,64,139,154});
    }

    @Test void generatedDirectVectorOwnsWholeRowsAndExecutesScalarTail() throws Throwable {
        int lanes = jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.length();
        int n = lanes + 3, k = 3, m = 2;
        float[] left = {1, 2, 3, 4, 5, 6};
        float[] right = new float[k * n];
        for (int index = 0; index < right.length; index++) right[index] = index % 7 - 2;
        float[] output = new float[m * n], expected = new float[m * n];
        for (int row = 0; row < m; row++) for (int column = 0; column < n; column++)
            for (int inner = 0; inner < k; inner++)
                expected[row * n + column] += left[row * k + inner] * right[inner * n + column];
        var read = plan(CpuAccessPlan.AccessKind.READ);
        var write = plan(CpuAccessPlan.AccessKind.WRITE);
        int bits = jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.vectorBitSize();
        var ir = new CpuMatmulIr(DataType.FLOAT32, DataType.FLOAT32, DataType.FLOAT32,
                CpuMatmulIr.Realization.DIRECT_N_VECTOR, CpuMatmulIr.Epilogue.none(), bits,
                CpuMatmulIr.NumericalForm.SEQUENTIAL, List.of(read, read), write);
        var specialization = new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(
                ir.structuralKey()), CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR,
                List.of(DataType.FLOAT32, DataType.FLOAT32, DataType.FLOAT32),
                List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.FLOAT_ARRAY,
                        CarrierAccess.FLOAT_ARRAY), bits, -1, List.of(), false, 54,
                java.util.Optional.of(ir));
        var generator = new CpuClassFileKernelGenerator();
        byte[] bytes=generator.generateClassBytes(specialization, ir.encodedKernelIr());
        var generated = generator.defineClassBytes(specialization,bytes);
        var evidence=java.nio.file.Path.of("/tmp/synaptik-cpu-0008f-evidence/focused");
        java.nio.file.Files.createDirectories(evidence);
        java.nio.file.Files.write(evidence.resolve("direct-n-vector-f32.class"),bytes);
        String constants=new String(bytes,java.nio.charset.StandardCharsets.ISO_8859_1);
        var references=java.util.stream.StreamSupport.stream(java.lang.classfile.ClassFile.of()
                        .parse(bytes).constantPool().spliterator(),false)
                .filter(java.lang.classfile.constantpool.MemberRefEntry.class::isInstance)
                .map(java.lang.classfile.constantpool.MemberRefEntry.class::cast).toList();
        assertAll(()->assertTrue(constants.contains("jdk/incubator/vector/FloatVector")),
                ()->assertFalse(constants.contains("io/github/pho001/synaptik/backend/cpu/internal/reference")),
                ()->assertFalse(constants.contains("java/lang/reflect")),
                ()->assertFalse(constants.contains("java/util/Map")),
                ()->assertTrue(references.stream().anyMatch(value->value.name().stringValue()
                        .equals("mul")&&value.type().stringValue()
                        .equals("(F)Ljdk/incubator/vector/FloatVector;"))),
                ()->assertTrue(references.stream().noneMatch(value->value.name().stringValue()
                        .equals("broadcast"))));
        long[] geometry = {0,0,0,1,m,k,n,k,1,n,1,n,1,0,0,0,(long)m*n,1};
        generated.entryPoint().invokeWithArguments(left,right,output,geometry,0L,(long)m);
        assertArrayEquals(expected, output);
    }

    @Test void generatedDirectVectorAddsContiguousBiasToFullSpeciesAndScalarTail() throws Throwable {
        int lanes=jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.length(),n=lanes+3;
        float[] right=new float[n],bias=new float[n],output=new float[n],expected=new float[n];
        for(int i=0;i<n;i++){right[i]=i-2;bias[i]=i*.25f;expected[i]=2*right[i]+bias[i];}
        var read=plan(CpuAccessPlan.AccessKind.READ);var write=plan(CpuAccessPlan.AccessKind.WRITE);
        var biasPlan=new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,CpuAccessPlan.Regime.LAST_AXIS_BIAS,
                1,List.of(CpuAccessPlan.AxisRole.CONTIGUOUS),1);
        int bits=jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.vectorBitSize();
        var ir=new CpuMatmulIr(DataType.FLOAT32,DataType.FLOAT32,DataType.FLOAT32,
                CpuMatmulIr.Realization.DIRECT_N_VECTOR,new CpuMatmulIr.Epilogue(
                CpuMatmulIr.Epilogue.AddInputOrder.MATMUL_LEFT,CpuMatmulIr.Epilogue.Terminal.NONE,null),
                bits,CpuMatmulIr.NumericalForm.SEQUENTIAL,List.of(read,read,biasPlan),write);
        var specialization=new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR,
                java.util.Collections.nCopies(4,DataType.FLOAT32),
                java.util.Collections.nCopies(4,CarrierAccess.FLOAT_ARRAY),bits,-1,List.of(),false,54,
                java.util.Optional.of(ir));var generator=new CpuClassFileKernelGenerator();
        byte[] bytes=generator.generateClassBytes(specialization,ir.encodedKernelIr());
        var instructions=java.lang.classfile.ClassFile.of().parse(bytes).methods().getFirst()
                .code().orElseThrow().elementStream().filter(java.lang.classfile.Instruction.class::isInstance)
                .map(java.lang.classfile.Instruction.class::cast).toList();
        assertAll(()->assertTrue(instructions.stream().noneMatch(value->value.opcode()
                        ==java.lang.classfile.Opcode.F2D)),
                ()->assertTrue(instructions.stream().noneMatch(value->value.opcode()
                        ==java.lang.classfile.Opcode.D2F)));
        var generated=generator.defineClassBytes(specialization,bytes);
        long[] geometry={0,0,0,1,1,1,n,1,1,n,1,n,1,0,0,1,n,1};
        generated.entryPoint().invokeWithArguments(new float[]{2},right,bias,output,geometry,0L,1L);
        assertArrayEquals(expected,output);
    }

    @Test void generatedScalarBodyCoversAllThirteenOrderedNumericPairs() throws Throwable {
        DataType[] floating = {DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64};
        for (DataType leftType : floating) for (DataType rightType : floating)
            scalarPair(leftType, rightType);
        DataType[] integral = {DataType.INT32, DataType.INT64};
        for (DataType leftType : integral) for (DataType rightType : integral)
            scalarPair(leftType, rightType);
    }

    @Test void generatedScalarBodyUsesDirectFloatingAndIntegralSegmentAccess() throws Throwable {
        segmentScalar(DataType.FLOAT32, new float[]{2,3}, new float[]{4,5}, new float[1]);
        segmentScalar(DataType.INT64, new long[]{2,3}, new long[]{4,5}, new long[1]);
    }

    @Test void generatedScalarTileOwnsTwoByTwoCellsAndExplicitMNtails() throws Throwable {
        int m=3,k=2,n=3;float[] left={1,2,3,4,5,6},right={1,2,3,4,5,6};
        float[] output=new float[m*n],expected=new float[m*n];
        for(int row=0;row<m;row++)for(int column=0;column<n;column++)for(int inner=0;inner<k;inner++)
            expected[row*n+column]+=left[row*k+inner]*right[inner*n+column];
        var read=plan(CpuAccessPlan.AccessKind.READ);var write=plan(CpuAccessPlan.AccessKind.WRITE);
        var ir=new CpuMatmulIr(DataType.FLOAT32,DataType.FLOAT32,DataType.FLOAT32,
                CpuMatmulIr.Realization.TILED_SCALAR_2X2,CpuMatmulIr.Epilogue.none(),0,
                CpuMatmulIr.NumericalForm.SEQUENTIAL,List.of(read,read),write);
        var specialization=new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(
                ir.structuralKey()),CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                List.of(DataType.FLOAT32,DataType.FLOAT32,DataType.FLOAT32),
                List.of(CarrierAccess.FLOAT_ARRAY,CarrierAccess.FLOAT_ARRAY,
                        CarrierAccess.FLOAT_ARRAY),0,-1,List.of(),false,54,
                java.util.Optional.of(ir));
        var generator=new CpuClassFileKernelGenerator();
        byte[] bytes=generator.generateClassBytes(specialization,ir.encodedKernelIr());
        var instructions=java.lang.classfile.ClassFile.of().parse(bytes).methods().getFirst()
                .code().orElseThrow().elementStream()
                .filter(java.lang.classfile.Instruction.class::isInstance)
                .map(java.lang.classfile.Instruction.class::cast).toList();
        assertNoPerAccessLongConversion(instructions);
        assertTrue(instructions.stream().filter(value->value.opcode()
                ==java.lang.classfile.Opcode.LALOAD).count()<=32);
        var generated=generator.defineClassBytes(specialization,bytes);
        long[] geometry={0,0,0,1,m,k,n,k,1,n,1,n,1,0,0,0,(long)m*n,1};
        generated.entryPoint().invokeWithArguments(left,right,output,geometry,0L,4L);
        assertArrayEquals(expected,output);
        float[] ranged=new float[m*n];
        generated.entryPoint().invokeWithArguments(left,right,ranged,geometry,2L,4L);
        generated.entryPoint().invokeWithArguments(left,right,ranged,geometry,0L,2L);
        assertArrayEquals(expected,ranged);
    }

    @Test void generatedVectorTileOwnsTwoRowsTwoSpeciesAndExplicitMNtails() throws Throwable {
        int lanes=jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.length();
        int m=3,k=5,n=2*lanes+3;float[] left=new float[m*k],right=new float[k*n];
        for(int index=0;index<left.length;index++)left[index]=index%5-2;
        for(int index=0;index<right.length;index++)right[index]=index%7-3;
        float[] output=new float[m*n],expected=new float[m*n];
        for(int row=0;row<m;row++)for(int column=0;column<n;column++)for(int inner=0;inner<k;inner++)
            expected[row*n+column]+=left[row*k+inner]*right[inner*n+column];
        var read=plan(CpuAccessPlan.AccessKind.READ);var write=plan(CpuAccessPlan.AccessKind.WRITE);
        int bits=jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.vectorBitSize();
        var ir=new CpuMatmulIr(DataType.FLOAT32,DataType.FLOAT32,DataType.FLOAT32,
                CpuMatmulIr.Realization.TILED_N_VECTOR_2X2,CpuMatmulIr.Epilogue.none(),bits,
                CpuMatmulIr.NumericalForm.SEQUENTIAL,List.of(read,read),write);
        var specialization=new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(
                ir.structuralKey()),CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR,
                List.of(DataType.FLOAT32,DataType.FLOAT32,DataType.FLOAT32),
                List.of(CarrierAccess.FLOAT_ARRAY,CarrierAccess.FLOAT_ARRAY,
                        CarrierAccess.FLOAT_ARRAY),bits,-1,List.of(),false,54,
                java.util.Optional.of(ir));
        var generator=new CpuClassFileKernelGenerator();
        byte[] bytes=generator.generateClassBytes(specialization,ir.encodedKernelIr());
        var members=java.lang.classfile.ClassFile.of().parse(bytes).constantPool().spliterator();
        var references=java.util.stream.StreamSupport.stream(members,false)
                .filter(java.lang.classfile.constantpool.MemberRefEntry.class::isInstance)
                .map(java.lang.classfile.constantpool.MemberRefEntry.class::cast).toList();
        var invokes=java.lang.classfile.ClassFile.of().parse(bytes).methods().getFirst().code()
                .orElseThrow().elementStream()
                .filter(java.lang.classfile.instruction.InvokeInstruction.class::isInstance)
                .map(java.lang.classfile.instruction.InvokeInstruction.class::cast).toList();
        assertAll(()->assertTrue(references.stream().anyMatch(value->value.name().stringValue()
                        .equals("mul")&&value.type().stringValue().equals("(F)Ljdk/incubator/vector/FloatVector;"))),
                ()->assertTrue(references.stream().noneMatch(value->value.name().stringValue()
                        .equals("broadcast"))),
                ()->assertEquals(1,invokes.stream().filter(value->value.name().stringValue()
                        .equals("zero")&&value.owner().asInternalName()
                        .equals("jdk/incubator/vector/FloatVector")).count()));
        var generated=generator.defineClassBytes(specialization,bytes);
        long[] geometry={0,0,0,1,m,k,n,k,1,n,1,n,1,0,0,0,(long)m*n,1};
        long work=(long)((m+1)/2)*((n+2L*lanes-1)/(2L*lanes));
        generated.entryPoint().invokeWithArguments(left,right,output,geometry,0L,work);
        assertArrayEquals(expected,output);
        float[] ranged=new float[m*n];long split=work/2;
        generated.entryPoint().invokeWithArguments(left,right,ranged,geometry,split,work);
        generated.entryPoint().invokeWithArguments(left,right,ranged,geometry,0L,split);
        assertArrayEquals(expected,ranged);
    }

    @Test void generatedScalarBiasAndEveryRecognizedTerminalExecuteInOneBody() throws Throwable {
        for(var terminal:CpuMatmulIr.Epilogue.Terminal.values()) {
            if(terminal==CpuMatmulIr.Epilogue.Terminal.NONE)continue;
            ClampRangeAttrs clamp=terminal==CpuMatmulIr.Epilogue.Terminal.CLAMP
                    ?new ClampRangeAttrs(ScalarValue.float64(-1),ScalarValue.float64(1)):null;
            var epilogue=new CpuMatmulIr.Epilogue(
                    CpuMatmulIr.Epilogue.AddInputOrder.MATMUL_RIGHT,terminal,clamp);
            double actual=epilogue(epilogue);
            double value=6.5;
            double expected=switch(terminal) {
                case RELU -> Math.max(value,+0.0);case SIGMOID -> CpuScalarReferenceKernel.sigmoid(value);
                case TANH -> StrictMath.tanh(value);case GELU -> CpuScalarReferenceKernel.gelu(value);
                case GELU_TANH_APPROXIMATION -> CpuScalarReferenceKernel.geluTanhApproximation(value);
                case SILU -> CpuScalarReferenceKernel.silu(value);case CLAMP -> 1.0;
                case NONE -> throw new AssertionError();
            };
            assertEquals(expected,actual,0.0,terminal.toString());
        }
        assertEquals(6.5,epilogue(new CpuMatmulIr.Epilogue(
                CpuMatmulIr.Epilogue.AddInputOrder.MATMUL_LEFT,
                CpuMatmulIr.Epilogue.Terminal.NONE,null)),0.0);
        assertEquals(StrictMath.tanh(6.0),epilogue(new CpuMatmulIr.Epilogue(
                CpuMatmulIr.Epilogue.AddInputOrder.NONE,
                CpuMatmulIr.Epilogue.Terminal.TANH,null)),0.0);
    }

    private static double epilogue(CpuMatmulIr.Epilogue epilogue)throws Throwable {
        var read=plan(CpuAccessPlan.AccessKind.READ);var write=plan(CpuAccessPlan.AccessKind.WRITE);
        var biasPlan=new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,
                CpuAccessPlan.Regime.LAST_AXIS_BIAS,1,List.of(CpuAccessPlan.AxisRole.CONTIGUOUS),1);
        var accesses=epilogue.hasBias()?List.of(read,read,biasPlan):List.of(read,read);
        var ir=new CpuMatmulIr(DataType.FLOAT64,DataType.FLOAT64,DataType.FLOAT64,
                CpuMatmulIr.Realization.DIRECT_SCALAR,epilogue,0,
                CpuMatmulIr.NumericalForm.SEQUENTIAL,accesses,write);
        var types=java.util.Collections.nCopies(epilogue.hasBias()?4:3,DataType.FLOAT64);
        var carriers=java.util.Collections.nCopies(epilogue.hasBias()?4:3,CarrierAccess.DOUBLE_ARRAY);
        var specialization=new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                types,carriers,0,-1,List.of(),false,
                54,java.util.Optional.of(ir));
        var generator=new CpuClassFileKernelGenerator();var generated=generator.defineClassBytes(
                specialization,generator.generateClassBytes(specialization,ir.encodedKernelIr()));
        double[] output={0};long[] geometry={0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1};
        if(epilogue.hasBias())generated.entryPoint().invokeWithArguments(new double[]{2},
                new double[]{3},new double[]{0.5},output,geometry,0L,1L);
        else generated.entryPoint().invokeWithArguments(new double[]{2},new double[]{3},output,
                geometry,0L,1L);
        return output[0];
    }

    private static void segmentScalar(DataType type, Object leftArray, Object rightArray,
            Object outputArray) throws Throwable {
        var read=plan(CpuAccessPlan.AccessKind.READ);var write=plan(CpuAccessPlan.AccessKind.WRITE);
        var ir=new CpuMatmulIr(type,type,type,CpuMatmulIr.Realization.DIRECT_SCALAR,
                CpuMatmulIr.Epilogue.none(),0,CpuMatmulIr.NumericalForm.SEQUENTIAL,
                List.of(read,read),write);
        var specialization=new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(
                ir.structuralKey()),CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,List.of(type,type,type),
                List.of(CarrierAccess.MEMORY_SEGMENT,CarrierAccess.MEMORY_SEGMENT,
                        CarrierAccess.MEMORY_SEGMENT),0,-1,List.of(),false,54,
                java.util.Optional.of(ir));
        var generator=new CpuClassFileKernelGenerator();
        var generated=generator.defineClassBytes(specialization,
                generator.generateClassBytes(specialization,ir.encodedKernelIr()));
        java.lang.foreign.MemorySegment left=segment(leftArray),right=segment(rightArray),
                output=segment(outputArray);
        long[] geometry={0,0,0,1,1,2,1,2,1,1,1,1,1,0,0,0,1,1};
        generated.entryPoint().invokeWithArguments(left,right,output,geometry,0L,1L);
        assertEquals(23.0,value(type,outputArray),0.0);
    }

    private static void scalarPair(DataType leftType, DataType rightType) throws Throwable {
        DataType resultType = io.github.pho001.synaptik.model.datatype.DataTypePromotion
                .promoteNumeric(leftType, rightType);
        var read = plan(CpuAccessPlan.AccessKind.READ);
        var write = plan(CpuAccessPlan.AccessKind.WRITE);
        var ir = new CpuMatmulIr(leftType, rightType, resultType,
                CpuMatmulIr.Realization.DIRECT_SCALAR, CpuMatmulIr.Epilogue.none(), 0,
                CpuMatmulIr.NumericalForm.SEQUENTIAL, List.of(read, read), write);
        var carriers = List.of(carrier(leftType), carrier(rightType), carrier(resultType));
        var specialization = new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(
                ir.structuralKey()), CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                List.of(leftType, rightType, resultType), carriers, 0, -1, List.of(), false,
                54, java.util.Optional.of(ir));
        var generator = new CpuClassFileKernelGenerator();
        var generated = generator.defineClassBytes(specialization,
                generator.generateClassBytes(specialization, ir.encodedKernelIr()));
        Object left = values(leftType, 2, 3), right = values(rightType, 4, 5);
        Object output = values(resultType, 0);
        long[] geometry = {0,0,0,1,1,2,1,2,1,1,1,1,1,0,0,0,1,1};
        generated.entryPoint().invokeWithArguments(left,right,output,geometry,0L,1L);
        assertEquals(23.0, value(resultType, output), 0.0, leftType + " x " + rightType);
    }

    private static void run(DataType type, Object left, Object right, Object output,
            Object expected) throws Throwable {
        var read = plan(CpuAccessPlan.AccessKind.READ);
        var write = plan(CpuAccessPlan.AccessKind.WRITE);
        var ir = new CpuMatmulIr(type, type, type, CpuMatmulIr.Realization.DIRECT_SCALAR,
                CpuMatmulIr.Epilogue.none(), 0, CpuMatmulIr.NumericalForm.SEQUENTIAL,
                List.of(read, read), write);
        CarrierAccess carrier = type == DataType.FLOAT32 ? CarrierAccess.FLOAT_ARRAY
                : CarrierAccess.INT_ARRAY;
        var specialization = new CpuKernelSpecialization(
                CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                List.of(type,type,type), List.of(carrier,carrier,carrier), 0, -1,
                List.of(), false, 54, java.util.Optional.of(ir));
        var generator = new CpuClassFileKernelGenerator();
        var generated = generator.defineClassBytes(specialization,
                generator.generateClassBytes(specialization, ir.encodedKernelIr()));
        long[] geometry = {0,0,0,1,2,3,2,3,1,2,1,2,1,0,0,0,4,1};
        generated.entryPoint().invokeWithArguments(left,right,output,geometry,0L,4L);
        if (type == DataType.FLOAT32) assertArrayEquals((float[]) expected, (float[]) output);
        else assertArrayEquals((int[]) expected, (int[]) output);
    }

    private static CpuAccessPlan plan(CpuAccessPlan.AccessKind kind) {
        return new CpuAccessPlan(kind, CpuAccessPlan.Regime.DENSE_LINEAR, 2,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS,
                        CpuAccessPlan.AxisRole.CONTIGUOUS), 2);
    }

    private static CarrierAccess carrier(DataType type) {
        return switch (type) {
            case BFLOAT16 -> CarrierAccess.SHORT_ARRAY; case FLOAT32 -> CarrierAccess.FLOAT_ARRAY;
            case FLOAT64 -> CarrierAccess.DOUBLE_ARRAY; case INT32 -> CarrierAccess.INT_ARRAY;
            case INT64 -> CarrierAccess.LONG_ARRAY; case BOOL -> throw new AssertionError();
        };
    }

    private static Object values(DataType type, int... values) {
        return switch (type) {
            case BFLOAT16 -> { short[] result=new short[values.length]; for(int i=0;i<values.length;i++)
                result[i]=io.github.pho001.synaptik.model.datatype.BFloat16Bits.fromFloat(values[i]); yield result; }
            case FLOAT32 -> { float[] result=new float[values.length]; for(int i=0;i<values.length;i++)result[i]=values[i]; yield result; }
            case FLOAT64 -> { double[] result=new double[values.length]; for(int i=0;i<values.length;i++)result[i]=values[i]; yield result; }
            case INT32 -> values.clone();
            case INT64 -> { long[] result=new long[values.length]; for(int i=0;i<values.length;i++)result[i]=values[i]; yield result; }
            case BOOL -> throw new AssertionError();
        };
    }

    private static double value(DataType type, Object values) {
        return switch (type) {
            case BFLOAT16 -> io.github.pho001.synaptik.model.datatype.BFloat16Bits.toFloat(((short[])values)[0]);
            case FLOAT32 -> ((float[])values)[0]; case FLOAT64 -> ((double[])values)[0];
            case INT32 -> ((int[])values)[0]; case INT64 -> ((long[])values)[0];
            case BOOL -> throw new AssertionError();
        };
    }

    private static java.lang.foreign.MemorySegment segment(Object array) {
        if(array instanceof float[] values)return java.lang.foreign.MemorySegment.ofArray(values);
        if(array instanceof long[] values)return java.lang.foreign.MemorySegment.ofArray(values);
        throw new AssertionError(array.getClass());
    }

    private static void assertNoPerAccessLongConversion(
            List<java.lang.classfile.Instruction> instructions) {
        for(int index=1;index<instructions.size();index++) {
            var opcode=instructions.get(index).opcode();
            if(opcode==java.lang.classfile.Opcode.FALOAD||opcode==java.lang.classfile.Opcode.FASTORE
                    ||opcode==java.lang.classfile.Opcode.SALOAD
                    ||opcode==java.lang.classfile.Opcode.SASTORE)
                assertNotEquals(java.lang.classfile.Opcode.L2I,instructions.get(index-1).opcode());
        }
    }
}

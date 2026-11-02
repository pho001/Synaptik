package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.*;
import java.lang.classfile.instruction.*;
import java.lang.constant.*;
import java.lang.invoke.*;
import java.lang.reflect.AccessFlag;
import java.util.*;
import java.util.function.Consumer;
import org.junit.jupiter.api.Test;

/** Permanent common-contraction/control/setup checks independent of private evidence paths. */
final class CpuMatmulBindingTest {
    /**
     * Checks both represented dtypes and length modes without warmup or measurement. The
     * prepared production consumer is compared with independent specialized Java numerical
     * outputs, including poisoned padding and nonuniform public seeds.
     * @throws Throwable if preparation, execution or a numerical comparison fails
     */
    @Test void conditionalRowsAreSemanticallyQualifiedWithoutTiming() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean length : List.of(false,true))
            try (var fixture = new CpuMatmulBindingFixture(type,length)) { fixture.verifyLatest(); }
    }
    /**
     * Exercises cursor/indexed binding independently from Scan, including poisoned inactive
     * operands and null zero-trip inputs, then compares typed arithmetic with clean Java.
     * @throws Throwable if generated linkage or a represented result differs
     */
    @Test void sharedContractionsGateBothLoadsPreserveFullKAndTypedZero() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean indexed : List.of(false,true))
            for (TypeKind kKind : indexed ? List.of(TypeKind.INT) : List.of(TypeKind.INT,TypeKind.LONG)) {
                byte[] bytes = contraction(type,indexed,kKind);
                assertTrue(ClassFile.of().verify(bytes).isEmpty());
                var lookup = MethodHandles.lookup().defineHiddenClass(bytes,true);
                var primitive = type == DataType.FLOAT64 ? double.class : float.class;
                var array = type == DataType.FLOAT64 ? double[].class : float[].class;
                var entry = lookup.findStatic(lookup.lookupClass(),"dot",MethodType.methodType(primitive,array,array,int.class,boolean.class));
                Object left = type == DataType.FLOAT64 ? new double[]{2,Double.NaN,3} : new float[]{2,Float.NaN,3};
                Object right = type == DataType.FLOAT64 ? new double[]{5,Double.POSITIVE_INFINITY,7} : new float[]{5,Float.POSITIVE_INFINITY,7};
                Number active = (Number)entry.invokeWithArguments(left,right,3,true);
                Number all = (Number)entry.invokeWithArguments(left,right,3,false);
                Number empty = (Number)entry.invokeWithArguments(null,null,0,true);
                assertEquals(31,active.doubleValue(),0); assertTrue(Double.isNaN(all.doubleValue()));
                assertEquals(0,Double.doubleToRawLongBits(empty.doubleValue()),"K=0 typed +0 without either load");
                if (type == DataType.FLOAT64) assertEquals(cleanDouble((double[])left,(double[])right,3,true),active.doubleValue(),0);
                else assertEquals(cleanFloat((float[])left,(float[])right,3,true),active.floatValue(),0);
                var code = ClassFile.of().parse(bytes).methods().getFirst().code().orElseThrow();
                var instructions = code.elementStream().filter(Instruction.class::isInstance).map(Instruction.class::cast).toList();
                assertEquals(1,instructions.stream().filter(i -> i.opcode() == (type == DataType.FLOAT64 ? Opcode.DMUL : Opcode.FMUL)).count());
                assertEquals(1,instructions.stream().filter(i -> i.opcode() == (type == DataType.FLOAT64 ? Opcode.DADD : Opcode.FADD)).count());
                assertTrue(instructions.stream().noneMatch(i -> i instanceof InvokeInstruction || i instanceof NewObjectInstruction
                        || i instanceof NewPrimitiveArrayInstruction || i instanceof NewReferenceArrayInstruction));
                // Both loads occur after the active-coordinate branch and before its forward join.
                int loads = 0;
                for (int i = 0; i < instructions.size(); i++) if (instructions.get(i).opcode() == (type == DataType.FLOAT64 ? Opcode.DALOAD : Opcode.FALOAD)) {
                    loads++;
                    assertTrue(instructions.subList(0,i).stream().anyMatch(x -> x.opcode() == (kKind == TypeKind.INT ? Opcode.IF_ICMPEQ : Opcode.IFEQ)));
                }
                assertEquals(2,loads);
            }
    }


    /**
     * Builds a scalar bound contraction without entry geometry or any Scan semantic owner.
     * @param type FLOAT32 or FLOAT64 accumulator type
     * @param indexed whether loaders use original K directly rather than operand-value locals
     * @param kKind independent INT/LONG induction proof
     * @return verified-class candidate containing one static primitive-returning dot method
     */
    private static byte[] contraction(DataType type,boolean indexed,TypeKind kKind) {
        var primitive = type == DataType.FLOAT64 ? ConstantDescs.CD_double : ConstantDescs.CD_float;
        var array = primitive.arrayType(); var valueKind = type == DataType.FLOAT64 ? TypeKind.DOUBLE : TypeKind.FLOAT;
        return ClassFile.of().build(ClassDesc.of(CpuMatmulBindingTest.class.getPackageName()+".BoundDot"),builder -> builder
                .withMethod("dot",MethodTypeDesc.of(primitive,array,array,ConstantDescs.CD_int,ConstantDescs.CD_boolean),AccessFlag.STATIC.mask(),method -> method.withCode(code -> {
                    int k = code.allocateLocal(kKind), sum = code.allocateLocal(valueKind);
                    int left = code.allocateLocal(valueKind), right = code.allocateLocal(valueKind);
                    if (type == DataType.FLOAT64) code.loadConstant(0d); else code.loadConstant(0f);
                    code.storeLocal(valueKind,sum);
                    Runnable bound = () -> { code.iload(2); if (kKind == TypeKind.LONG) code.i2l(); };
                    java.util.function.IntConsumer leftAt = coordinate -> { code.aload(0).loadLocal(kKind,coordinate); if (kKind == TypeKind.LONG) code.l2i(); if (type == DataType.FLOAT64) code.daload(); else code.faload(); };
                    java.util.function.IntConsumer rightAt = coordinate -> { code.aload(1).loadLocal(kKind,coordinate); if (kKind == TypeKind.LONG) code.l2i(); if (type == DataType.FLOAT64) code.daload(); else code.faload(); };
                    Consumer<Runnable> gate = body -> {
                        var unguarded = code.newLabel(); var inactive = code.newLabel();
                        code.iload(3).branch(Opcode.IFEQ,unguarded).loadLocal(kKind,k);
                        if (kKind == TypeKind.INT) code.loadConstant(1).branch(Opcode.IF_ICMPEQ,inactive);
                        else code.loadConstant(1L).lcmp().branch(Opcode.IFEQ,inactive);
                        code.labelBinding(unguarded); body.run(); code.labelBinding(inactive);
                    };
                    if (indexed) CpuMatmulEmitter.emitIndexedContraction(code,type,k,sum,bound,leftAt,rightAt,gate);
                    else CpuMatmulEmitter.emitCursorContraction(code,type,kKind,k,sum,left,right,bound,
                            () -> { leftAt.accept(k); code.storeLocal(valueKind,left); },
                            () -> { rightAt.accept(k); code.storeLocal(valueKind,right); },() -> {},gate);
                    code.loadLocal(valueKind,sum).return_(valueKind);
                })));
    }

    /**
     * Optimal specialized Java oracle: increasing K, branch before both reads, separate math.
     * @param left borrowed F32 operands
     * @param right borrowed F32 operands
     * @param bound full contributor extent
     * @param gate whether contributor one is inactive
     * @return represented F32 ordered sum, positive zero for an empty domain
     */
    private static float cleanFloat(float[] left,float[] right,int bound,boolean gate) {
        float sum = 0f; for (int k = 0; k < bound; k++) if (!gate || k != 1) sum += left[k]*right[k]; return sum;
    }

    /**
     * F64 counterpart with no float narrowing or altered contributor domain.
     * @param left borrowed F64 operands
     * @param right borrowed F64 operands
     * @param bound full contributor extent
     * @param gate whether contributor one is inactive
     * @return represented F64 ordered sum, positive zero for an empty domain
     */
    private static double cleanDouble(double[] left,double[] right,int bound,boolean gate) {
        double sum = 0d; for (int k = 0; k < bound; k++) if (!gate || k != 1) sum += left[k]*right[k]; return sum;
    }
}

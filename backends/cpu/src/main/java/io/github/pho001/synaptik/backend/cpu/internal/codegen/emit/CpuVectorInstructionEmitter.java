package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;
import java.lang.constant.MethodTypeDesc;
import java.util.Locale;

/**
 * Emits the closed pointwise opcode family's already-admitted vector instructions into one
 * generated method body.
 *
 * <p>The emitter owns operation-level bytecode selection for one homogeneous FLOAT32 or FLOAT64
 * body. It writes only to preallocated topology-local vector locals and delegates pure
 * multi-instruction formulas to {@link CpuVectorMath}; class structure, loops, carriers, stores,
 * tails, and route selection remain outside this type.</p>
 */
final class CpuVectorInstructionEmitter {
    private static final ClassDesc VECTOR_BASE = ClassDesc.of("jdk.incubator.vector.Vector");
    private static final ClassDesc MATH = ClassDesc.of(CpuVectorMath.class.getName());
    private final CodeBuilder code;
    private final DataType dataType;
    private final ClassDesc vector;
    private final ClassDesc primitive;

    /**
     * Creates an instruction emitter for one homogeneous floating lane type.
     *
     * @param code non-null Class-File API builder retained for generation only
     * @param dataType exact lane type, either {@link DataType#FLOAT32} or {@link DataType#FLOAT64}
     * @throws IllegalArgumentException if {@code dataType} is not a supported vector lane type
     */
    CpuVectorInstructionEmitter(CodeBuilder code, DataType dataType) {
        this.code = code;
        this.dataType = dataType;
        this.vector = switch (dataType) {
            case FLOAT32 -> ClassDesc.of("jdk.incubator.vector.FloatVector");
            case FLOAT64 -> ClassDesc.of("jdk.incubator.vector.DoubleVector");
            default -> throw new IllegalArgumentException("unsupported vector data type");
        };
        this.primitive = dataType == DataType.FLOAT32 ? ConstantDescs.CD_float : ConstantDescs.CD_double;
    }

    /**
     * Emits one vector instruction and stores its result in the output local.
     *
     * @param instruction non-null admitted instruction with typed immediates already validated
     * @param locals non-null value-ordinal-to-reference-local mapping for this generated body
     * @throws IllegalArgumentException for an opcode or power realization outside the closed set
     */
    void emit(CpuKernelIr.Instruction instruction, int[] locals) {
        if (instruction.opcode() != CpuPointwiseOpcode.SCALAR_POW
                || instruction.powerRealization() != CpuKernelIr.PowerRealization.POSITIVE_ONE) {
            code.aload(locals[instruction.inputs().getFirst()]);
        }
        switch (instruction.opcode()) {
            case ADD, SUB, MUL, DIV -> binary(instruction, locals, switch (instruction.opcode()) {
                case ADD -> "add"; case SUB -> "sub"; case DIV -> "div"; default -> "mul";
            });
            case SCALAR_ADD, SCALAR_SUB, SCALAR_MUL, SCALAR_DIV -> {
                loadImmediate(instruction.scalarImmediate().bits());
                code.invokevirtual(vector, switch (instruction.opcode()) {
                    case SCALAR_ADD -> "add"; case SCALAR_SUB -> "sub";
                    case SCALAR_DIV -> "div"; default -> "mul";
                }, MethodTypeDesc.of(vector, primitive));
            }
            case SCALAR_POW -> emitPower(instruction.powerRealization());
            case NEG -> code.invokevirtual(vector, "neg", MethodTypeDesc.of(vector));
            case ABS, RECIPROCAL, LOG, LOG1P, EXP, EXPM1, ERF, SQRT, RSQRT, TANH -> math(
                    instruction.opcode().name().toLowerCase(Locale.ROOT));
            case GELU_EXACT -> math("gelu");
            default -> throw new IllegalArgumentException("unsupported vector opcode");
        }
        code.astore(locals[instruction.output()]);
    }

    private void binary(CpuKernelIr.Instruction instruction, int[] locals, String method) {
        code.aload(locals[instruction.inputs().get(1)]);
        code.invokevirtual(vector, method, MethodTypeDesc.of(vector, VECTOR_BASE));
    }

    private void emitPower(CpuKernelIr.PowerRealization realization) {
        switch (realization) {
            case POSITIVE_ONE -> code.invokestatic(MATH,
                    dataType == DataType.FLOAT32 ? "positiveOneFloat" : "positiveOne",
                    MethodTypeDesc.of(vector));
            case IDENTITY -> { }
            case SQUARE -> code.dup().invokevirtual(vector, "mul",
                    MethodTypeDesc.of(vector, VECTOR_BASE));
            case RECIPROCAL -> math("reciprocal");
            case DIRECT -> throw new IllegalArgumentException("direct scalar power is not vector eligible");
        }
    }

    private void math(String method) {
        code.invokestatic(MATH, method, MethodTypeDesc.of(vector, vector));
    }

    private void loadImmediate(long bits) {
        if (dataType == DataType.FLOAT32) code.loadConstant(Float.intBitsToFloat((int) bits));
        else code.loadConstant(Double.longBitsToDouble(bits));
    }
}

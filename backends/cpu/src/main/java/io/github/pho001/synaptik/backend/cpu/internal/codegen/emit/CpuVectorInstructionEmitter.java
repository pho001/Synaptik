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
 * Emits one already-admitted typed value-vector or virtual-mask pointwise instruction.
 *
 * <p>The emitter consumes validated CPU IR and owns only the closed operation-level bytecode
 * choice. Ordinary values use the exact preferred FLOAT32, FLOAT64, INT32, INT64, or canonical
 * BOOL vector type. Eligible floating predicates use unit-private {@code VectorMask} values that
 * may be combined logically and consumed by floating {@code WHERE}; they are never materialized
 * or exposed as a boundary representation. Loop, carrier, tail, route, and fallback decisions
 * remain outside this type.</p>
 */
final class CpuVectorInstructionEmitter {
    private static final ClassDesc VECTOR_BASE = ClassDesc.of("jdk.incubator.vector.Vector");
    private static final ClassDesc VECTOR_MASK = ClassDesc.of("jdk.incubator.vector.VectorMask");
    private static final ClassDesc VECTOR_OPERATORS = ClassDesc.of("jdk.incubator.vector.VectorOperators");
    private static final ClassDesc COMPARISON =
            ClassDesc.of("jdk.incubator.vector.VectorOperators$Comparison");
    private static final ClassDesc TEST = ClassDesc.of("jdk.incubator.vector.VectorOperators$Test");
    private static final ClassDesc MATH = ClassDesc.of(CpuVectorMath.class.getName());
    private final CodeBuilder code;
    private final CpuKernelIr ir;
    private final DataType laneType;
    private final ClassDesc vector;
    private final ClassDesc primitive;

    /**
     * Creates an emitter for one validated lane type and topology.
     *
     * @param code non-null Class-File API builder retained for generation only
     * @param ir non-null typed pointwise IR
     * @param laneType exact FLOAT32, FLOAT64, INT32, INT64, or canonical BOOL lane type
     * @throws IllegalArgumentException if {@code laneType} is unsupported
     */
    CpuVectorInstructionEmitter(CodeBuilder code, CpuKernelIr ir, DataType laneType) {
        this.code = code;
        this.ir = ir;
        this.laneType = laneType;
        this.vector = switch (laneType) {
            case FLOAT32 -> ClassDesc.of("jdk.incubator.vector.FloatVector");
            case FLOAT64 -> ClassDesc.of("jdk.incubator.vector.DoubleVector");
            case INT32 -> ClassDesc.of("jdk.incubator.vector.IntVector");
            case INT64 -> ClassDesc.of("jdk.incubator.vector.LongVector");
            case BOOL -> ClassDesc.of("jdk.incubator.vector.ByteVector");
            default -> throw new IllegalArgumentException("unsupported vector data type");
        };
        this.primitive = switch (laneType) {
            case FLOAT32 -> ConstantDescs.CD_float;
            case FLOAT64 -> ConstantDescs.CD_double;
            case INT32 -> ConstantDescs.CD_int;
            case INT64 -> ConstantDescs.CD_long;
            case BOOL -> ConstantDescs.CD_byte;
            default -> throw new IllegalArgumentException("unsupported vector data type");
        };
    }

    /**
     * Emits one instruction and stores its vector or mask result in its output local.
     *
     * @param instruction non-null admitted typed instruction
     * @param locals non-null value-ordinal-to-reference-local mapping
     * @throws IllegalArgumentException if the instruction is outside the selected vector form
     */
    void emit(CpuKernelIr.Instruction instruction, int[] locals) {
        switch (instruction.opcode().vectorForm()) {
            case VALUE -> emitValue(instruction, locals);
            case VALUE_OR_MASK -> emitLogical(instruction, locals);
            case MASK_PRODUCER -> emitMaskProducer(instruction, locals);
            case MASK_CONSUMER -> emitWhere(instruction, locals);
            case NONE -> throw new IllegalArgumentException("unsupported vector opcode");
        }
        code.astore(locals[instruction.output()]);
    }

    private void emitValue(CpuKernelIr.Instruction instruction, int[] locals) {
        CpuPointwiseOpcode opcode = instruction.opcode();
        if (opcode == CpuPointwiseOpcode.SCALAR_POW
                && instruction.powerRealization() == CpuKernelIr.PowerRealization.POSITIVE_ONE) {
            emitPower(instruction.powerRealization());
            return;
        }
        code.aload(locals[instruction.inputs().getFirst()]);
        switch (opcode) {
            case ADD, SUB, MUL, DIV, MIN, MAX -> binary(instruction, locals, switch (opcode) {
                case ADD -> "add"; case SUB -> "sub"; case DIV -> "div";
                case MIN -> "min"; case MAX -> "max"; default -> "mul";
            });
            case SCALAR_ADD, SCALAR_SUB, SCALAR_MUL, SCALAR_DIV,
                    SCALAR_MIN, SCALAR_MAX -> {
                loadImmediate(instruction.scalarImmediate().bits());
                code.invokevirtual(vector, switch (opcode) {
                    case SCALAR_ADD -> "add"; case SCALAR_SUB -> "sub";
                    case SCALAR_DIV -> "div"; case SCALAR_MIN -> "min";
                    case SCALAR_MAX -> "max"; default -> "mul";
                }, MethodTypeDesc.of(vector, primitive));
            }
            case SCALAR_CLAMP -> {
                loadImmediate(instruction.clampImmediate().lower().bits());
                code.invokevirtual(vector, "max", MethodTypeDesc.of(vector, primitive));
                loadImmediate(instruction.clampImmediate().upper().bits());
                code.invokevirtual(vector, "min", MethodTypeDesc.of(vector, primitive));
            }
            case SCALAR_POW -> emitPower(instruction.powerRealization());
            case NEG -> code.invokevirtual(vector, "neg", MethodTypeDesc.of(vector));
            case ABS, RECIPROCAL, LOG, LOG1P, EXP, EXPM1, ERF, SQRT, RSQRT, SIGN, TANH ->
                    math(opcode.name().toLowerCase(Locale.ROOT));
            case RELU -> {
                loadZero();
                code.invokevirtual(vector, "max", MethodTypeDesc.of(vector, primitive));
            }
            case GELU_EXACT -> math("gelu");
            case CAST -> { }
            default -> throw new IllegalArgumentException("unsupported value-vector opcode");
        }
    }

    private void emitLogical(CpuKernelIr.Instruction instruction, int[] locals) {
        String method = switch (instruction.opcode()) {
            case LOGICAL_AND -> "and"; case LOGICAL_OR -> "or";
            case LOGICAL_NOT -> "not"; default -> throw new IllegalArgumentException(
                    "unsupported logical vector opcode");
        };
        code.aload(locals[instruction.inputs().getFirst()]);
        if (instruction.inputs().size() == 2) code.aload(locals[instruction.inputs().get(1)]);
        boolean mask = maskValue(instruction.output());
        ClassDesc owner = mask ? VECTOR_MASK : vector;
        MethodTypeDesc type = instruction.inputs().size() == 1
                ? MethodTypeDesc.of(owner) : MethodTypeDesc.of(owner,
                        mask ? VECTOR_MASK : VECTOR_BASE);
        code.invokevirtual(owner, method, type);
        if (instruction.opcode() == CpuPointwiseOpcode.LOGICAL_NOT && !mask) {
            code.loadConstant(1);
            code.i2b();
            code.invokevirtual(vector, "and", MethodTypeDesc.of(vector, ConstantDescs.CD_byte));
        }
    }

    private void emitMaskProducer(CpuKernelIr.Instruction instruction, int[] locals) {
        code.aload(locals[instruction.inputs().getFirst()]);
        if (instruction.opcode().family() == CpuPointwiseOpcode.Family.CLASSIFICATION) {
            String token = switch (instruction.opcode()) {
                case IS_FINITE -> "IS_FINITE"; case IS_NAN -> "IS_NAN";
                case IS_INF -> "IS_INFINITE"; default -> throw new AssertionError();
            };
            code.getstatic(VECTOR_OPERATORS, token, TEST);
            code.invokevirtual(vector, "test", MethodTypeDesc.of(VECTOR_MASK, TEST));
            return;
        }
        code.getstatic(VECTOR_OPERATORS, switch (instruction.opcode()) {
            case GREATER_THAN -> "GT"; case GREATER_OR_EQUAL -> "GE";
            case LESS_THAN -> "LT"; case LESS_OR_EQUAL -> "LE";
            case EQUAL -> "EQ"; case NOT_EQUAL -> "NE"; default -> throw new AssertionError();
        }, COMPARISON);
        code.aload(locals[instruction.inputs().get(1)]);
        code.invokevirtual(vector, "compare", MethodTypeDesc.of(VECTOR_MASK,
                COMPARISON, VECTOR_BASE));
    }

    private void emitWhere(CpuKernelIr.Instruction instruction, int[] locals) {
        code.aload(locals[instruction.inputs().get(2)]);
        code.aload(locals[instruction.inputs().get(1)]);
        code.aload(locals[instruction.inputs().get(0)]);
        code.invokevirtual(vector, "blend", MethodTypeDesc.of(vector, VECTOR_BASE, VECTOR_MASK));
    }

    private boolean maskValue(int ordinal) {
        return laneType != DataType.BOOL && ir.values().get(ordinal).dataType() == DataType.BOOL;
    }

    private void binary(CpuKernelIr.Instruction instruction, int[] locals, String method) {
        code.aload(locals[instruction.inputs().get(1)]);
        code.invokevirtual(vector, method, MethodTypeDesc.of(vector, VECTOR_BASE));
    }

    private void emitPower(CpuKernelIr.PowerRealization realization) {
        switch (realization) {
            case POSITIVE_ONE -> code.invokestatic(MATH,
                    laneType == DataType.FLOAT32 ? "positiveOneFloat" : "positiveOne",
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

    private void loadZero() {
        if (laneType == DataType.FLOAT32) code.loadConstant(+0.0f);
        else if (laneType == DataType.FLOAT64) code.loadConstant(+0.0d);
        else throw new IllegalArgumentException("floating zero requires floating vector type");
    }

    private void loadImmediate(long bits) {
        switch (laneType) {
            case FLOAT32 -> code.loadConstant(Float.intBitsToFloat((int) bits));
            case FLOAT64 -> code.loadConstant(Double.longBitsToDouble(bits));
            case INT32 -> code.loadConstant((int) bits);
            case INT64 -> code.loadConstant(bits);
            default -> throw new IllegalArgumentException("BOOL has no scalar arithmetic immediate");
        }
    }
}

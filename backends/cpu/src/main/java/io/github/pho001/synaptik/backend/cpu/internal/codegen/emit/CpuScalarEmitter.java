package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import io.github.pho001.synaptik.backend.cpu.internal.reference.CpuScalarReferenceKernel;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;
import java.lang.constant.MethodTypeDesc;

/**
 * Package-private family-grouped exact/default scalar semantic emitter.
 *
 * <p>Unary FLOAT64 emission uses primitive operations, {@link Math}, {@link StrictMath}, or the
 * shared pure activation helpers according to the selected opcode. FLOAT32 widens one represented
 * value to {@code double} where the JDK lacks a float overload and narrows the result once. The
 * FLOAT32 reciprocal-square-root path performs both the square root and reciprocal after that
 * widening, then narrows only their combined result. The emitter consumes typed IR only and makes
 * no capability, route, or numerical-policy decision.</p>
 */
final class CpuScalarEmitter {
    private static final ClassDesc REFERENCE = ClassDesc.of(CpuScalarReferenceKernel.class.getName());
    private static final ClassDesc STRICT_MATH = ClassDesc.of(StrictMath.class.getName());
    private static final ClassDesc MATH = ClassDesc.of(Math.class.getName());
    private static final ClassDesc INTEGER = ClassDesc.of(Integer.class.getName());
    private static final ClassDesc LONG = ClassDesc.of(Long.class.getName());
    private static final ClassDesc DOUBLE = ClassDesc.of(Double.class.getName());
    private static final ClassDesc FLOAT = ClassDesc.of(Float.class.getName());
    private final CodeBuilder code;

    /**
     * Creates an emitter bound to one non-null generated method body.
     *
     * @param code non-null Class-File API code builder retained for generation only
     */
    CpuScalarEmitter(CodeBuilder code) { this.code = code; }

    /**
     * Emits one already-typed instruction into its preallocated topology-local output.
     *
     * @param ir non-null typed canonical IR owning the instruction values
     * @param instruction non-null instruction already validated by {@code ir}
     * @param locals non-null value-ordinal-to-local-slot mapping
     */
    void emit(CpuKernelIr ir, CpuKernelIr.Instruction instruction, int[] locals) {
        CpuPointwiseOpcode opcode = instruction.opcode();
        DataType inputType = ir.values().get(instruction.inputs().getFirst()).dataType();
        DataType outputType = ir.values().get(instruction.output()).dataType();
        if (opcode == CpuPointwiseOpcode.WHERE) {
            emitWhere(ir, instruction, locals, outputType);
            return;
        }
        if (opcode == CpuPointwiseOpcode.SCALAR_POW) {
            emitPower(instruction, locals, inputType);
            store(outputType, locals[instruction.output()]);
            return;
        }
        if (opcode == CpuPointwiseOpcode.POW) {
            emitTensorPower(instruction, locals, inputType);
            store(outputType, locals[instruction.output()]);
            return;
        }
        if (opcode == CpuPointwiseOpcode.SCALAR_CLAMP) {
            load(inputType, locals[instruction.inputs().getFirst()]);
            loadImmediate(instruction.clampImmediate().lower());
            extrema(false, inputType);
            loadImmediate(instruction.clampImmediate().upper());
            extrema(true, inputType);
            store(outputType, locals[instruction.output()]);
            return;
        }
        load(inputType, locals[instruction.inputs().getFirst()]);
        if (instruction.inputs().size() == 2) {
            load(inputType, locals[instruction.inputs().get(1)]);
        }
        if (opcode.carriesScalarImmediate()) loadImmediate(instruction.scalarImmediate());
        switch (opcode.family()) {
            case BINARY_ARITHMETIC, SCALAR_ARITHMETIC -> arithmetic(opcode, inputType);
            case UNARY -> unary(opcode, inputType);
            case CLASSIFICATION -> classification(opcode, inputType);
            case COMPARISON -> comparison(opcode, inputType);
            case LOGICAL -> logical(opcode);
            case CAST -> { }
            case SELECTION, SCALAR_RANGE -> throw new AssertionError(opcode);
        }
        store(outputType, locals[instruction.output()]);
    }

    private void arithmetic(CpuPointwiseOpcode opcode, DataType type) {
        boolean add = opcode == CpuPointwiseOpcode.ADD || opcode == CpuPointwiseOpcode.SCALAR_ADD;
        boolean sub = opcode == CpuPointwiseOpcode.SUB || opcode == CpuPointwiseOpcode.SCALAR_SUB;
        boolean div = opcode == CpuPointwiseOpcode.DIV || opcode == CpuPointwiseOpcode.SCALAR_DIV;
        boolean min = opcode == CpuPointwiseOpcode.MIN || opcode == CpuPointwiseOpcode.SCALAR_MIN;
        boolean max = opcode == CpuPointwiseOpcode.MAX || opcode == CpuPointwiseOpcode.SCALAR_MAX;
        if (min || max) {
            extrema(min, type);
            return;
        }
        switch (type) {
            case FLOAT64 -> { if (add) code.dadd(); else if (sub) code.dsub();
                else if (div) code.ddiv(); else code.dmul(); }
            case FLOAT32 -> { if (add) code.fadd(); else if (sub) code.fsub();
                else if (div) code.fdiv(); else code.fmul(); }
            case INT32 -> { if (div) throw new IllegalArgumentException("integral division unsupported");
                if (add) code.iadd(); else if (sub) code.isub(); else code.imul(); }
            case INT64 -> { if (div) throw new IllegalArgumentException("integral division unsupported");
                if (add) code.ladd(); else if (sub) code.lsub(); else code.lmul(); }
            default -> throw new IllegalArgumentException("unsupported arithmetic type");
        }
    }

    private void emitTensorPower(CpuKernelIr.Instruction instruction, int[] locals, DataType type) {
        load(type, locals[instruction.inputs().get(0)]);
        if (type == DataType.FLOAT32) code.f2d();
        load(type, locals[instruction.inputs().get(1)]);
        if (type == DataType.FLOAT32) code.f2d();
        code.invokestatic(STRICT_MATH, "pow", MethodTypeDesc.of(ConstantDescs.CD_double,
                ConstantDescs.CD_double, ConstantDescs.CD_double));
        if (type == DataType.FLOAT32) code.d2f();
    }

    private void extrema(boolean minimum, DataType type) {
        String method = minimum ? "min" : "max";
        ClassDesc owner;
        ClassDesc primitive;
        switch (type) {
            case FLOAT64 -> { owner = MATH; primitive = ConstantDescs.CD_double; }
            case FLOAT32 -> { owner = MATH; primitive = ConstantDescs.CD_float; }
            case INT32 -> { owner = INTEGER; primitive = ConstantDescs.CD_int; }
            case INT64 -> { owner = LONG; primitive = ConstantDescs.CD_long; }
            default -> throw new IllegalArgumentException("unsupported extrema type");
        }
        code.invokestatic(owner, method, MethodTypeDesc.of(primitive, primitive, primitive));
    }

    private void logical(CpuPointwiseOpcode opcode) {
        switch (opcode) {
            case LOGICAL_AND -> code.iand();
            case LOGICAL_OR -> code.ior();
            case LOGICAL_NOT -> code.loadConstant(1).ixor();
            default -> throw new AssertionError(opcode);
        }
    }

    private void emitPower(CpuKernelIr.Instruction instruction, int[] locals, DataType type) {
        int base = locals[instruction.inputs().getFirst()];
        switch (instruction.powerRealization()) {
            case POSITIVE_ONE -> loadPositiveOne(type);
            case IDENTITY -> load(type, base);
            case SQUARE -> {
                load(type, base);
                load(type, base);
                if (type == DataType.FLOAT64) code.dmul(); else code.fmul();
            }
            case RECIPROCAL -> {
                loadPositiveOne(type);
                load(type, base);
                if (type == DataType.FLOAT64) code.ddiv(); else code.fdiv();
            }
            case DIRECT -> {
                load(type, base);
                if (type == DataType.FLOAT32) code.f2d();
                loadImmediate(instruction.scalarImmediate());
                if (type == DataType.FLOAT32) code.f2d();
                code.invokestatic(STRICT_MATH, "pow", MethodTypeDesc.of(ConstantDescs.CD_double,
                        ConstantDescs.CD_double, ConstantDescs.CD_double));
                if (type == DataType.FLOAT32) code.d2f();
            }
        }
    }

    private void loadPositiveOne(DataType type) {
        if (type == DataType.FLOAT64) code.loadConstant(1.0d);
        else if (type == DataType.FLOAT32) code.loadConstant(1.0f);
        else throw new IllegalArgumentException("scalar power requires floating type");
    }

    private void unary(CpuPointwiseOpcode opcode, DataType type) {
        if (opcode == CpuPointwiseOpcode.NEG) {
            if (type == DataType.FLOAT64) code.dneg(); else if (type == DataType.FLOAT32) code.fneg();
            else throw new IllegalArgumentException("unsupported unary type");
            return;
        }
        if (opcode == CpuPointwiseOpcode.RSQRT && type == DataType.FLOAT32) {
            code.f2d();
            code.invokestatic(STRICT_MATH, "sqrt", MethodTypeDesc.of(ConstantDescs.CD_double,
                    ConstantDescs.CD_double));
            int root = code.allocateLocal(TypeKind.DOUBLE);
            code.dstore(root);
            code.loadConstant(1.0d);
            code.dload(root);
            code.ddiv();
            code.d2f();
            return;
        }
        if (opcode == CpuPointwiseOpcode.RECIPROCAL || opcode == CpuPointwiseOpcode.RSQRT) {
            int value = code.allocateLocal(type == DataType.FLOAT64 ? TypeKind.DOUBLE : TypeKind.FLOAT);
            store(type, value);
            loadPositiveOne(type);
            load(type, value);
            if (opcode == CpuPointwiseOpcode.RSQRT) invokeDoubleUnary(type, STRICT_MATH, "sqrt");
            if (type == DataType.FLOAT64) code.ddiv(); else code.fdiv();
            return;
        }
        if (opcode == CpuPointwiseOpcode.ABS || opcode == CpuPointwiseOpcode.SIGN
                || opcode == CpuPointwiseOpcode.RELU) {
            ClassDesc primitive = type == DataType.FLOAT64 ? ConstantDescs.CD_double
                    : ConstantDescs.CD_float;
            if (opcode == CpuPointwiseOpcode.RELU) {
                if (type == DataType.FLOAT64) code.loadConstant(+0.0d); else code.loadConstant(+0.0f);
                code.invokestatic(MATH, "max", MethodTypeDesc.of(primitive, primitive, primitive));
            } else code.invokestatic(MATH, opcode == CpuPointwiseOpcode.ABS ? "abs" : "signum",
                    MethodTypeDesc.of(primitive, primitive));
            return;
        }
        String helper = switch (opcode) {
            case ERF -> "erf"; case SIGMOID -> "sigmoid"; case GELU_EXACT -> "gelu";
            case GELU_TANH_APPROXIMATION -> "geluTanhApproximation"; case SILU -> "silu";
            default -> null;
        };
        if (helper != null) {
            invokeDoubleUnary(type, REFERENCE, helper);
            return;
        }
        String method = switch (opcode) {
            case LOG -> "log"; case LOG1P -> "log1p"; case EXP -> "exp"; case EXPM1 -> "expm1";
            case SQRT -> "sqrt"; case FLOOR -> "floor"; case CEIL -> "ceil"; case TANH -> "tanh";
            default -> throw new AssertionError(opcode);
        };
        invokeDoubleUnary(type, STRICT_MATH, method);
    }

    private void invokeDoubleUnary(DataType type, ClassDesc owner, String method) {
        if (type == DataType.FLOAT32) code.f2d();
        code.invokestatic(owner, method, MethodTypeDesc.of(ConstantDescs.CD_double,
                ConstantDescs.CD_double));
        if (type == DataType.FLOAT32) code.d2f();
    }

    private void classification(CpuPointwiseOpcode opcode, DataType type) {
        String method = switch (opcode) {
            case IS_FINITE -> "isFinite"; case IS_NAN -> "isNaN"; case IS_INF -> "isInfinite";
            default -> throw new AssertionError(opcode);
        };
        ClassDesc owner = type == DataType.FLOAT64 ? DOUBLE : FLOAT;
        code.invokestatic(owner, method, MethodTypeDesc.of(ConstantDescs.CD_boolean,
                type == DataType.FLOAT64 ? ConstantDescs.CD_double : ConstantDescs.CD_float));
    }

    private void comparison(CpuPointwiseOpcode opcode, DataType type) {
        if (type == DataType.FLOAT64) {
            if (opcode == CpuPointwiseOpcode.GREATER_THAN
                    || opcode == CpuPointwiseOpcode.GREATER_OR_EQUAL) code.dcmpl(); else code.dcmpg();
        } else if (type == DataType.FLOAT32) {
            if (opcode == CpuPointwiseOpcode.GREATER_THAN
                    || opcode == CpuPointwiseOpcode.GREATER_OR_EQUAL) code.fcmpl(); else code.fcmpg();
        } else if (type == DataType.INT64) code.lcmp();
        Opcode branch = type == DataType.INT32 ? switch (opcode) {
            case GREATER_THAN -> Opcode.IF_ICMPGT; case GREATER_OR_EQUAL -> Opcode.IF_ICMPGE;
            case LESS_THAN -> Opcode.IF_ICMPLT; case LESS_OR_EQUAL -> Opcode.IF_ICMPLE;
            case EQUAL -> Opcode.IF_ICMPEQ; case NOT_EQUAL -> Opcode.IF_ICMPNE;
            default -> throw new AssertionError(opcode);
        } : switch (opcode) {
            case GREATER_THAN -> Opcode.IFGT; case GREATER_OR_EQUAL -> Opcode.IFGE;
            case LESS_THAN -> Opcode.IFLT; case LESS_OR_EQUAL -> Opcode.IFLE;
            case EQUAL -> Opcode.IFEQ; case NOT_EQUAL -> Opcode.IFNE;
            default -> throw new AssertionError(opcode);
        };
        var yes = code.newLabel();
        var done = code.newLabel();
        code.branch(branch, yes);
        code.loadConstant(0).branch(Opcode.GOTO, done);
        code.labelBinding(yes).loadConstant(1);
        code.labelBinding(done);
    }

    private void emitWhere(CpuKernelIr ir, CpuKernelIr.Instruction instruction, int[] locals,
            DataType resultType) {
        var whenFalse = code.newLabel();
        var done = code.newLabel();
        code.iload(locals[instruction.inputs().get(0)]).branch(Opcode.IFEQ, whenFalse);
        load(resultType, locals[instruction.inputs().get(1)]);
        store(resultType, locals[instruction.output()]);
        code.branch(Opcode.GOTO, done).labelBinding(whenFalse);
        load(resultType, locals[instruction.inputs().get(2)]);
        store(resultType, locals[instruction.output()]);
        code.labelBinding(done);
    }

    private void loadImmediate(CpuKernelIr.ScalarImmediate immediate) {
        switch (immediate.dataType()) {
            case FLOAT64 -> code.loadConstant(Double.longBitsToDouble(immediate.bits()));
            case FLOAT32 -> code.loadConstant(Float.intBitsToFloat((int) immediate.bits()));
            case INT32 -> code.loadConstant((int) immediate.bits());
            case INT64 -> code.loadConstant(immediate.bits());
            default -> throw new IllegalArgumentException("unsupported scalar immediate");
        }
    }

    private void load(DataType type, int local) {
        switch (type) {
            case FLOAT64 -> code.dload(local); case FLOAT32 -> code.fload(local);
            case INT32, BOOL -> code.iload(local); case INT64 -> code.lload(local);
            default -> throw new IllegalArgumentException("unsupported value type");
        }
    }

    private void store(DataType type, int local) {
        switch (type) {
            case FLOAT64 -> code.dstore(local); case FLOAT32 -> code.fstore(local);
            case INT32, BOOL -> code.istore(local); case INT64 -> code.lstore(local);
            default -> throw new IllegalArgumentException("unsupported value type");
        }
    }
}

package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;
import java.lang.constant.MethodTypeDesc;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Emits typed ordinary aggregate folds directly into generated CPU entries.
 *
 * <p>The static body reduces complete output cells only. It traverses every selected domain in
 * logical input row-major order, selects the first represented NaN, applies explicit signed-zero
 * extrema rules, and allocates no per-cell or per-element object. Full dense heap-array
 * reductions use one typed integer-address fold; other forms embed a typed long-address fallback
 * that decodes output and selected-domain coordinates without runtime type, kind, form, or
 * carrier dispatch.</p>
 */
public final class CpuAggregateEmitter {
    private static final ClassDesc OWNER = ClassDesc.of(CpuAggregateEmitter.class.getName());
    private static final DataType[] TYPES = DataType.values();
    /** Creates a stateless typed-body emitter. */
    public CpuAggregateEmitter() { }

    /**
     * Emits a typed aggregate body whose hot loops contain no runtime semantic dispatch.
     *
     * @param code non-null Class-File method builder mutated with the generated fold
     * @param specialization non-null two-boundary, scratch-free carrier/type specialization
     * @param ir non-null canonical aggregate IR supplying kind, form, axes, rank, and access facts
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if the specialization has another boundary shape
     */
    public void emit(CodeBuilder code, CpuKernelSpecialization specialization, CpuKernelIr ir) {
        if (specialization.carrierPattern().size() != 2 || specialization.scratchParameter())
            throw new IllegalArgumentException("aggregate requires two boundaries and no scratch");
        String identity = ir.familyIdentity();
        DataType type = specialization.boundaryDataTypes().getFirst();
        int kind = identity.startsWith("aggregate:MIN:") ? 0
                : identity.startsWith("aggregate:MAX:") ? 1
                : identity.startsWith("aggregate:ALL:") ? 2 : 3;
        int inRank = ir.values().getFirst().accessPlan().iterationRank();
        int outRank = ir.values().getLast().accessPlan().iterationRank();
        boolean fullDenseArrays = identity.contains(":FULL:")
                && specialization.carrierPattern().stream().noneMatch(
                    access -> access == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)
                && ir.values().getFirst().accessPlan().regime() == CpuAccessPlan.Regime.DENSE_LINEAR;
        if (fullDenseArrays) emitFullDense(code, specialization, type, kind, inRank);
        else emitGeneral(code, specialization, type, kind, identity, inRank, outRank);
    }

    private static void emitFullDense(CodeBuilder code, CpuKernelSpecialization specialization,
            DataType type, int kind, int inRank) {
        var done = code.newLabel();
        code.lload(3).lload(5).lcmp().branch(Opcode.IFGE, done);
        int inputLayout = 8 + 2 * inRank;
        int outputLayout = inputLayout + 2 + 2 * inRank;
        int input = code.allocateLocal(TypeKind.INT), output = code.allocateLocal(TypeKind.INT);
        code.aload(2).loadConstant(inputLayout + 1).laload().l2i().istore(input);
        code.aload(2).loadConstant(outputLayout + 1).laload().l2i().istore(output);
        int accumulator = code.allocateLocal(localKind(type));
        emitIdentity(code, type, kind); store(code, type, accumulator);
        int domain = code.allocateLocal(TypeKind.INT); code.loadConstant(0).istore(domain);
        var store = code.newLabel();
        code.aload(2).loadConstant(7).laload().loadConstant(0L).lcmp().branch(Opcode.IFEQ, store);
        var loop = code.newLabel(); code.labelBinding(loop);
        int address = code.allocateLocal(TypeKind.INT);
        code.iload(input).iload(domain).iadd().istore(address);
        int value = code.allocateLocal(localKind(type));
        var carriers = new CpuCarrierEmitter(code);
        carriers.load(type, specialization.carrierPattern().getFirst(), 0, address, true);
        store(code, type, value); emitApply(code, type, kind, accumulator, value);
        if (type == DataType.BOOL) {
            load(code, type, accumulator);
            code.branch(kind == 2 ? Opcode.IFEQ : Opcode.IFNE, store);
        }
        code.iinc(domain, 1); code.iload(domain).aload(2).loadConstant(7).laload().l2i()
                .branch(Opcode.IF_ICMPLT, loop);
        code.labelBinding(store);
        carriers.store(type, specialization.carrierPattern().getLast(), 1, output, accumulator, true);
        code.labelBinding(done);
    }

    private static void emitGeneral(CodeBuilder code, CpuKernelSpecialization specialization,
            DataType type, int kind, String identity, int inRank, int outRank) {
        boolean keep = identity.contains(":keep=true:");
        boolean[] selectedAxes = selectedAxes(identity, inRank);
        int selected = 8, inputCoordinates = selected + inRank;
        int outputCoordinates = inputCoordinates + inRank;
        int inputLayout = outputCoordinates + outRank;
        int outputLayout = inputLayout + 2 + 2 * inRank;
        int cell = code.allocateLocal(TypeKind.LONG);
        code.lload(3).lstore(cell);
        var done = code.newLabel();
        code.lload(cell).lload(5).lcmp().branch(Opcode.IFGE, done);
        var cells = code.newLabel(); code.labelBinding(cells);
        decode(code, cell, outputCoordinates, outputLayout, outRank);
        int outAxis = 0;
        for (int axis = 0; axis < inRank; axis++) {
            code.aload(2).loadConstant(inputCoordinates + axis);
            if (selectedAxes[axis]) code.loadConstant(0L);
            else code.aload(2).loadConstant(outputCoordinates + (keep ? axis : outAxis++)).laload();
            code.lastore();
        }
        int accumulator = code.allocateLocal(localKind(type));
        emitIdentity(code, type, kind); store(code, type, accumulator);
        int domain = code.allocateLocal(TypeKind.LONG); code.loadConstant(0L).lstore(domain);
        var write = code.newLabel();
        code.aload(2).loadConstant(7).laload().loadConstant(0L).lcmp().branch(Opcode.IFEQ, write);
        var domains = code.newLabel(); code.labelBinding(domains);
        int remaining = code.allocateLocal(TypeKind.LONG); code.lload(domain).lstore(remaining);
        for (int axis = inRank - 1; axis >= 0; axis--) if (selectedAxes[axis]) {
            code.aload(2).loadConstant(inputCoordinates + axis).lload(remaining)
                    .aload(2).loadConstant(inputLayout + 2 + axis).laload().lrem().lastore();
            code.lload(remaining).aload(2).loadConstant(inputLayout + 2 + axis).laload()
                    .ldiv().lstore(remaining);
        }
        int inputAddress = address(code, inRank, inputCoordinates, inputLayout);
        int value = code.allocateLocal(localKind(type));
        var carriers = new CpuCarrierEmitter(code);
        carriers.load(type, specialization.carrierPattern().getFirst(), 0, inputAddress);
        store(code, type, value); emitApply(code, type, kind, accumulator, value);
        if (type == DataType.BOOL) {
            load(code, type, accumulator);
            code.branch(kind == 2 ? Opcode.IFEQ : Opcode.IFNE, write);
        }
        code.lload(domain).loadConstant(1L).ladd().lstore(domain);
        code.lload(domain).aload(2).loadConstant(7).laload().lcmp().branch(Opcode.IFLT, domains);
        code.labelBinding(write);
        int outputAddress = address(code, outRank, outputCoordinates, outputLayout);
        carriers.store(type, specialization.carrierPattern().getLast(), 1, outputAddress, accumulator);
        code.lload(cell).loadConstant(1L).ladd().lstore(cell);
        code.lload(cell).lload(5).lcmp().branch(Opcode.IFLT, cells);
        code.labelBinding(done);
    }

    private static boolean[] selectedAxes(String identity, int rank) {
        boolean[] result = new boolean[rank];
        int start = identity.indexOf(":axes=[") + 7, end = identity.indexOf(']', start);
        String body = identity.substring(start, end).trim();
        if (!body.isEmpty()) for (String axis : body.split(", ")) result[Integer.parseInt(axis)] = true;
        return result;
    }

    private static void decode(CodeBuilder code, int logical, int coordinates, int layout, int rank) {
        int remaining = code.allocateLocal(TypeKind.LONG); code.lload(logical).lstore(remaining);
        for (int axis = rank - 1; axis >= 0; axis--) {
            code.aload(2).loadConstant(coordinates + axis).lload(remaining)
                    .aload(2).loadConstant(layout + 2 + axis).laload().lrem().lastore();
            code.lload(remaining).aload(2).loadConstant(layout + 2 + axis).laload()
                    .ldiv().lstore(remaining);
        }
    }

    private static int address(CodeBuilder code, int rank, int coordinates, int layout) {
        int address = code.allocateLocal(TypeKind.LONG);
        code.aload(2).loadConstant(layout + 1).laload().lstore(address);
        for (int axis = 0; axis < rank; axis++) code.lload(address)
                .aload(2).loadConstant(coordinates + axis).laload()
                .aload(2).loadConstant(layout + 2 + rank + axis).laload().lmul().ladd().lstore(address);
        return address;
    }

    private static TypeKind localKind(DataType type) { return switch (type) {
        case FLOAT64 -> TypeKind.DOUBLE; case FLOAT32 -> TypeKind.FLOAT;
        case INT64 -> TypeKind.LONG; case BFLOAT16, INT32, BOOL -> TypeKind.INT;
    }; }
    private static void store(CodeBuilder code, DataType type, int local) { switch (type) {
        case FLOAT64 -> code.dstore(local); case FLOAT32 -> code.fstore(local);
        case INT64 -> code.lstore(local); case BFLOAT16, INT32, BOOL -> code.istore(local);
    } }
    private static void load(CodeBuilder code, DataType type, int local) { switch (type) {
        case FLOAT64 -> code.dload(local); case FLOAT32 -> code.fload(local);
        case INT64 -> code.lload(local); case BFLOAT16, INT32, BOOL -> code.iload(local);
    } }
    private static void emitIdentity(CodeBuilder code, DataType type, int kind) { switch (type) {
        case FLOAT64 -> code.loadConstant(kind == 0 ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY);
        case FLOAT32 -> code.loadConstant(kind == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY);
        case BFLOAT16 -> code.loadConstant(kind == 0 ? 0x7f80 : 0xff80);
        case INT32 -> code.loadConstant(kind == 0 ? Integer.MAX_VALUE : Integer.MIN_VALUE);
        case INT64 -> code.loadConstant(kind == 0 ? Long.MAX_VALUE : Long.MIN_VALUE);
        case BOOL -> code.loadConstant(kind == 2 ? 1 : 0);
    } }
    private static void emitApply(CodeBuilder code, DataType type, int kind,
            int accumulator, int value) {
        load(code, type, accumulator); load(code, type, value);
        String name = switch (type) {
            case FLOAT64 -> kind == 0 ? "minDouble" : "maxDouble";
            case FLOAT32 -> kind == 0 ? "minFloat" : "maxFloat";
            case BFLOAT16 -> kind == 0 ? "minBfloat" : "maxBfloat";
            case INT32 -> kind == 0 ? "minInt" : "maxInt";
            case INT64 -> kind == 0 ? "minLong" : "maxLong";
            case BOOL -> kind == 2 ? "all" : "any";
        };
        ClassDesc primitive = switch (type) {
            case FLOAT64 -> ConstantDescs.CD_double; case FLOAT32 -> ConstantDescs.CD_float;
            case INT64 -> ConstantDescs.CD_long; default -> ConstantDescs.CD_int;
        };
        code.invokestatic(OWNER, name, MethodTypeDesc.of(primitive, primitive, primitive));
        store(code, type, accumulator);
    }

    /**
     * Selects the FLOAT64 minimum using CPU aggregate NaN and signed-zero rules.
     *
     * @param left first represented value
     * @param right second represented value
     * @return the selected represented value
     */
    static double minDouble(double left, double right) { return selectDouble(left, right, true); }

    /**
     * Selects the FLOAT64 maximum using CPU aggregate NaN and signed-zero rules.
     *
     * @param left first represented value
     * @param right second represented value
     * @return the selected represented value
     */
    static double maxDouble(double left, double right) { return selectDouble(left, right, false); }
    private static double selectDouble(double left, double right, boolean minimum) {
        if (Double.isNaN(left)) return left; if (Double.isNaN(right)) return right;
        if (left == 0.0 && right == 0.0) return minimum
                ? (Double.doubleToRawLongBits(left) < 0 ? left : right)
                : (Double.doubleToRawLongBits(left) >= 0 ? left : right);
        return minimum ? (right < left ? right : left) : (right > left ? right : left);
    }
    /**
     * Selects the FLOAT32 minimum using CPU aggregate NaN and signed-zero rules.
     *
     * @param left first represented value
     * @param right second represented value
     * @return the selected represented value
     */
    static float minFloat(float left, float right) { return selectFloat(left, right, true); }

    /**
     * Selects the FLOAT32 maximum using CPU aggregate NaN and signed-zero rules.
     *
     * @param left first represented value
     * @param right second represented value
     * @return the selected represented value
     */
    static float maxFloat(float left, float right) { return selectFloat(left, right, false); }
    private static float selectFloat(float left, float right, boolean minimum) {
        if (Float.isNaN(left)) return left; if (Float.isNaN(right)) return right;
        if (left == 0.0f && right == 0.0f) return minimum
                ? (Float.floatToRawIntBits(left) < 0 ? left : right)
                : (Float.floatToRawIntBits(left) >= 0 ? left : right);
        return minimum ? (right < left ? right : left) : (right > left ? right : left);
    }
    /**
     * Selects the BFLOAT16 minimum using unsigned represented-bit inputs and aggregate
     * NaN/signed-zero rules.
     *
     * @param left unsigned 16-bit BFLOAT16 bits in the low half of the integer
     * @param right unsigned 16-bit BFLOAT16 bits in the low half of the integer
     * @return unsigned 16-bit BFLOAT16 bits for the selected represented value
     */
    static int minBfloat(int left, int right) { return selectBfloat(left, right, true); }

    /**
     * Selects the BFLOAT16 maximum using unsigned represented-bit inputs and aggregate
     * NaN/signed-zero rules.
     *
     * @param left unsigned 16-bit BFLOAT16 bits in the low half of the integer
     * @param right unsigned 16-bit BFLOAT16 bits in the low half of the integer
     * @return unsigned 16-bit BFLOAT16 bits for the selected represented value
     */
    static int maxBfloat(int left, int right) { return selectBfloat(left, right, false); }
    private static int selectBfloat(int left, int right, boolean minimum) {
        int le = left & 0xffff, ri = right & 0xffff;
        if ((le & 0x7f80) == 0x7f80 && (le & 0x7f) != 0) return le;
        if ((ri & 0x7f80) == 0x7f80 && (ri & 0x7f) != 0) return ri;
        float l = Float.intBitsToFloat(le << 16), r = Float.intBitsToFloat(ri << 16);
        if (l == 0.0f && r == 0.0f) return minimum ? ((short) le < 0 ? le : ri)
                : ((short) le >= 0 ? le : ri);
        return minimum ? (r < l ? ri : le) : (r > l ? ri : le);
    }
    /**
     * Selects the smaller INT32 value.
     *
     * @param left first represented value
     * @param right second represented value
     * @return the smaller represented value
     */
    static int minInt(int left, int right) { return Math.min(left, right); }

    /**
     * Selects the larger INT32 value.
     *
     * @param left first represented value
     * @param right second represented value
     * @return the larger represented value
     */
    static int maxInt(int left, int right) { return Math.max(left, right); }

    /**
     * Selects the smaller INT64 value.
     *
     * @param left first represented value
     * @param right second represented value
     * @return the smaller represented value
     */
    static long minLong(long left, long right) { return Math.min(left, right); }

    /**
     * Selects the larger INT64 value.
     *
     * @param left first represented value
     * @param right second represented value
     * @return the larger represented value
     */
    static long maxLong(long left, long right) { return Math.max(left, right); }

    /**
     * Applies represented BOOL conjunction.
     *
     * @param left first represented boolean, where zero is false and non-zero is true
     * @param right second represented boolean, where zero is false and non-zero is true
     * @return {@code 1} when both inputs are true, otherwise {@code 0}
     */
    static int all(int left, int right) { return left != 0 && right != 0 ? 1 : 0; }

    /**
     * Applies represented BOOL disjunction.
     *
     * @param left first represented boolean, where zero is false and non-zero is true
     * @param right second represented boolean, where zero is false and non-zero is true
     * @return {@code 1} when either input is true, otherwise {@code 0}
     */
    static int any(int left, int right) { return left != 0 || right != 0 ? 1 : 0; }

    /**
     * Reduces complete flattened output cells in {@code [start,end)}.
     * @param input non-null readable primitive-array or native-segment carrier
     * @param output non-null writable non-overlapping carrier
     * @param packed non-null invocation-owned geometry and mutable coordinate state
     * @param start non-negative inclusive output-cell ordinal
     * @param end exclusive output-cell ordinal no greater than the output count
     * @throws NullPointerException if a carrier or {@code packed} is {@code null}
     * @throws ClassCastException if a carrier does not match the represented type selected during
     *     cold specialization
     * @throws ArithmeticException if an array address cannot be represented as {@code int}
     * @throws IndexOutOfBoundsException if a carrier does not cover a packed address
     * @throws IllegalStateException if a supplied memory segment is inaccessible
     */
    public static void execute(Object input, Object output, long[] packed, long start, long end) {
        int kind = (int) packed[0]; DataType type = TYPES[(int) packed[1]];
        boolean keep = packed[3] != 0; int inRank = (int) packed[4], outRank = (int) packed[5];
        long domainCount = packed[7]; int selected = 8;
        int inputCoordinates = selected + inRank;
        int outputCoordinates = inputCoordinates + inRank;
        int inputLayout = outputCoordinates + outRank;
        int outputLayout = inputLayout + 2 + 2 * inRank;
        for (long cell = start; cell < end; cell++) {
            decode(cell, packed, outputCoordinates, packed, outputLayout);
            int outAxis = 0;
            for (int axis = 0; axis < inRank; axis++) {
                if (packed[selected + axis] != 0) packed[inputCoordinates + axis] = 0;
                else packed[inputCoordinates + axis] = packed[outputCoordinates
                        + (keep ? axis : outAxis++)];
            }
            long accumulator = identity(kind, type);
            for (long domain = 0; domain < domainCount; domain++) {
                long remaining = domain;
                for (int axis = inRank - 1; axis >= 0; axis--) if (packed[selected + axis] != 0) {
                    long extent = packed[inputLayout + 2 + axis];
                    packed[inputCoordinates + axis] = remaining % extent; remaining /= extent;
                }
                long value = readBits(input, address(packed, inputLayout, inputCoordinates), type);
                accumulator = apply(kind, accumulator, value, type);
            }
            writeBits(output, address(packed, outputLayout, outputCoordinates), type, accumulator);
        }
    }

    private static void decode(long logical, long[] coordinatesOwner, int coordinates,
            long[] layoutOwner, int layout) {
        int rank = (int) layoutOwner[layout];
        for (int axis = rank - 1; axis >= 0; axis--) {
            long extent = layoutOwner[layout + 2 + axis];
            coordinatesOwner[coordinates + axis] = logical % extent; logical /= extent;
        }
    }
    private static long identity(int kind, DataType type) {
        return switch (type) {
            case FLOAT64 -> Double.doubleToRawLongBits(kind == 0
                    ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY);
            case FLOAT32 -> Integer.toUnsignedLong(Float.floatToRawIntBits(kind == 0
                    ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY));
            case BFLOAT16 -> kind == 0 ? 0x7f80L : 0xff80L;
            case INT32 -> kind == 0 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            case INT64 -> kind == 0 ? Long.MAX_VALUE : Long.MIN_VALUE;
            case BOOL -> kind == 2 ? 1 : 0;
        };
    }
    private static long apply(int kind, long left, long right, DataType type) {
        if (type == DataType.BOOL) return kind == 2
                ? ((left != 0 && right != 0) ? 1 : 0) : ((left != 0 || right != 0) ? 1 : 0);
        if (type == DataType.INT32) return kind == 0
                ? Math.min((int) left, (int) right) : Math.max((int) left, (int) right);
        if (type == DataType.INT64) return kind == 0 ? Math.min(left, right) : Math.max(left, right);
        if (isNaN(left, type)) return left;
        if (isNaN(right, type)) return right;
        double l = floatingValue(left, type), r = floatingValue(right, type);
        if (l == 0.0 && r == 0.0) {
            boolean leftNegative = negative(left, type), rightNegative = negative(right, type);
            if (kind == 0) return leftNegative ? left : rightNegative ? right : left;
            return !leftNegative ? left : !rightNegative ? right : left;
        }
        return kind == 0 ? (r < l ? right : left) : (r > l ? right : left);
    }
    private static boolean isNaN(long bits, DataType type) {
        return switch (type) {
            case FLOAT64 -> Double.isNaN(Double.longBitsToDouble(bits));
            case FLOAT32 -> Float.isNaN(Float.intBitsToFloat((int) bits));
            case BFLOAT16 -> ((bits & 0x7f80L) == 0x7f80L) && (bits & 0x7fL) != 0;
            default -> false;
        };
    }
    private static double floatingValue(long bits, DataType type) {
        return switch (type) {
            case FLOAT64 -> Double.longBitsToDouble(bits);
            case FLOAT32 -> Float.intBitsToFloat((int) bits);
            case BFLOAT16 -> Float.intBitsToFloat((int) bits << 16);
            default -> throw new AssertionError("non-floating aggregate type");
        };
    }
    private static boolean negative(long bits, DataType type) {
        return switch (type) {
            case FLOAT64 -> bits < 0;
            case FLOAT32 -> ((int) bits) < 0;
            case BFLOAT16 -> ((short) bits) < 0;
            default -> false;
        };
    }
    private static long address(long[] p, int layout, int coordinates) {
        long result = p[layout + 1]; int rank = (int) p[layout];
        for (int axis = 0; axis < rank; axis++)
            result += p[coordinates + axis] * p[layout + 2 + rank + axis];
        return result;
    }
    private static long readBits(Object carrier, long address, DataType type) {
        return switch (type) {
            case FLOAT64 -> Double.doubleToRawLongBits(carrier instanceof double[] a ? a[Math.toIntExact(address)] : ((MemorySegment) carrier).get(ValueLayout.JAVA_DOUBLE, address * 8));
            case FLOAT32 -> Integer.toUnsignedLong(Float.floatToRawIntBits(carrier instanceof float[] a ? a[Math.toIntExact(address)] : ((MemorySegment) carrier).get(ValueLayout.JAVA_FLOAT, address * 4)));
            case BFLOAT16 -> Short.toUnsignedLong(carrier instanceof short[] a ? a[Math.toIntExact(address)] : ((MemorySegment) carrier).get(ValueLayout.JAVA_SHORT, address * 2));
            case INT32 -> carrier instanceof int[] a ? a[Math.toIntExact(address)] : ((MemorySegment) carrier).get(ValueLayout.JAVA_INT, address * 4);
            case INT64 -> carrier instanceof long[] a ? a[Math.toIntExact(address)] : ((MemorySegment) carrier).get(ValueLayout.JAVA_LONG, address * 8);
            case BOOL -> carrier instanceof byte[] a ? Byte.toUnsignedLong(a[Math.toIntExact(address)]) : Byte.toUnsignedLong(((MemorySegment) carrier).get(ValueLayout.JAVA_BYTE, address));
        };
    }
    private static void writeBits(Object carrier, long address, DataType type, long bits) {
        switch (type) {
            case FLOAT64 -> { double v = Double.longBitsToDouble(bits); if (carrier instanceof double[] a) a[Math.toIntExact(address)] = v; else ((MemorySegment) carrier).set(ValueLayout.JAVA_DOUBLE, address * 8, v); }
            case FLOAT32 -> { float v = Float.intBitsToFloat((int) bits); if (carrier instanceof float[] a) a[Math.toIntExact(address)] = v; else ((MemorySegment) carrier).set(ValueLayout.JAVA_FLOAT, address * 4, v); }
            case BFLOAT16 -> { short v = (short) bits; if (carrier instanceof short[] a) a[Math.toIntExact(address)] = v; else ((MemorySegment) carrier).set(ValueLayout.JAVA_SHORT, address * 2, v); }
            case INT32 -> { int v = (int) bits; if (carrier instanceof int[] a) a[Math.toIntExact(address)] = v; else ((MemorySegment) carrier).set(ValueLayout.JAVA_INT, address * 4, v); }
            case INT64 -> { if (carrier instanceof long[] a) a[Math.toIntExact(address)] = bits; else ((MemorySegment) carrier).set(ValueLayout.JAVA_LONG, address * 8, bits); }
            case BOOL -> { byte v = (byte) (bits == 0 ? 0 : 1); if (carrier instanceof byte[] a) a[Math.toIntExact(address)] = v; else ((MemorySegment) carrier).set(ValueLayout.JAVA_BYTE, address, v); }
        }
    }
}

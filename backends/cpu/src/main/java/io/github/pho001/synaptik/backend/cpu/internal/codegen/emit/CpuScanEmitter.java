package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
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
 * Emits CPU-owned typed scan loops for independent slice ranges directly into generated entries.
 *
 * <p>Each range contains whole logical slices only. Within a slice, execution visits the selected
 * axis sequentially in forward or reverse order and applies inclusive or exclusive placement.
 * FLOAT64 and FLOAT32 retain same-format results, BFLOAT16 directly emits widening to FLOAT32,
 * one typed operation, and round-to-nearest-ties-to-even conversion after every value, and
 * INT32/INT64 wrap at their represented width. The generated BFLOAT16 body calls no Synaptik
 * runtime helper. The body allocates no
 * per-slice or per-element object and owns no worker, workspace, or persistent accumulator.
 * Dense rank-one heap-array scans use one-time integer base narrowing and a direct typed loop.
 * One completely guarded fixed {@code [1024,1024]} axis-one exclusive reverse INT64 product form
 * over two {@link MemorySegment} carriers uses direct {@link ValueLayout#JAVA_LONG_UNALIGNED}
 * access and descending invocation-local element cursors for arbitrary legal complete-slice
 * subranges. Every unproved geometry uses the typed general body, which decodes non-axis
 * coordinates once per slice. Neither form requests workspace or materialization.</p>
 */
public final class CpuScanEmitter {
    private static final ClassDesc FLOAT = ClassDesc.of(Float.class.getName());
    private static final ClassDesc SEGMENT = ClassDesc.of(MemorySegment.class.getName());
    private static final ClassDesc VALUE_LAYOUT = ClassDesc.of(ValueLayout.class.getName());
    private static final ClassDesc LONG_LAYOUT =
            ClassDesc.of("java.lang.foreign.ValueLayout$OfLong");
    private static final DataType[] TYPES = DataType.values();
    /** Creates a stateless emitter with no retained specialization or invocation state. */
    public CpuScanEmitter() { }

    /**
     * Emits one typed two-boundary scan body selected entirely from structural IR facts.
     *
     * @param code non-null Class-File method builder mutated with the generated hot body
     * @param specialization non-null scalar, two-boundary, workspace-free specialization; read
     *     during emission and not retained
     * @param ir non-null canonical scan IR whose kind, axis, modes, rank, and access plans are
     *     consumed during generation and never retained by the generated class
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if the specialization has another boundary or scratch
     *     shape
     */
    public void emit(CodeBuilder code, CpuKernelSpecialization specialization, CpuKernelIr ir) {
        if (specialization.carrierPattern().size() != 2 || specialization.scratchParameter())
            throw new IllegalArgumentException("scan requires two boundaries and no scratch");
        DataType type = specialization.boundaryDataTypes().getFirst();
        String identity = ir.familyIdentity();
        boolean product = identity.startsWith("scan:CUM_PROD:");
        boolean exclusive = identity.contains(":exclusive=true:");
        boolean reverse = identity.contains(":reverse=true:");
        int axis = integerAfter(identity, ":axis=");
        int rank = ir.values().getFirst().accessPlan().iterationRank();
        boolean denseRankOneArrays = rank == 1 && axis == 0
                && specialization.carrierPattern().stream().noneMatch(
                    access -> access == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)
                && ir.values().stream().allMatch(value -> value.accessPlan().regime()
                    == CpuAccessPlan.Regime.DENSE_LINEAR);
        if (denseRankOneArrays) emitDenseRankOne(code, specialization, type, product,
                exclusive, reverse);
        else if (provedReverseInt64ProductSegments(specialization, type, product, exclusive,
                reverse, rank, axis)) emitGuardedReverseInt64ProductSegments(code, specialization,
                        type, product, exclusive, reverse, rank, axis);
        else emitGeneral(code, specialization, type, product, exclusive, reverse, rank, axis);
    }

    private static boolean provedReverseInt64ProductSegments(
            CpuKernelSpecialization specialization, DataType type, boolean product,
            boolean exclusive, boolean reverse, int rank, int axis) {
        return type == DataType.INT64 && product && exclusive && reverse && rank == 2 && axis == 1
                && specialization.carrierPattern().stream().allMatch(
                        access -> access == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT);
    }

    private static void emitGuardedReverseInt64ProductSegments(CodeBuilder code,
            CpuKernelSpecialization specialization, DataType type, boolean product,
            boolean exclusive, boolean reverse, int rank, int axis) {
        var fallback = code.newLabel();
        var done = code.newLabel();
        guardPackedLong(code, fallback, 0, 1); // CUM_PROD
        guardPackedLong(code, fallback, 1, DataType.INT64.ordinal());
        guardPackedLong(code, fallback, 2, 2);
        guardPackedLong(code, fallback, 3, 1);
        guardPackedLong(code, fallback, 4, 1);
        guardPackedLong(code, fallback, 5, 1);
        guardPackedLong(code, fallback, 6, 1024);
        guardPackedLong(code, fallback, 7, 1024);
        guardPackedLong(code, fallback, 10, 2);
        guardPackedLong(code, fallback, 12, 1024);
        guardPackedLong(code, fallback, 13, 1024);
        guardPackedLong(code, fallback, 14, 2048);
        guardPackedLong(code, fallback, 15, 2);
        guardPackedLong(code, fallback, 16, 2);
        guardPackedLong(code, fallback, 18, 1024);
        guardPackedLong(code, fallback, 19, 1024);
        guardPackedLong(code, fallback, 20, 2048);
        guardPackedLong(code, fallback, 21, 2);
        code.lload(3).loadConstant(0L).lcmp().branch(Opcode.IFLT, fallback);
        code.lload(5).loadConstant(1024L).lcmp().branch(Opcode.IFGT, fallback);
        code.lload(3).lload(5).lcmp().branch(Opcode.IFGT, fallback);
        emitReverseInt64ProductSegmentCursors(code, done);
        code.branch(Opcode.GOTO, done);
        code.labelBinding(fallback);
        emitGeneral(code, specialization, type, product, exclusive, reverse, rank, axis);
        code.labelBinding(done);
    }

    private static void guardPackedLong(CodeBuilder code, java.lang.classfile.Label fallback,
            int index, long expected) {
        code.aload(2).loadConstant(index).laload().loadConstant(expected).lcmp()
                .branch(Opcode.IFNE, fallback);
    }

    private static void emitReverseInt64ProductSegmentCursors(CodeBuilder code,
            java.lang.classfile.Label done) {
        int slice = code.allocateLocal(TypeKind.LONG);
        code.lload(3).lstore(slice);
        code.lload(slice).lload(5).lcmp().branch(Opcode.IFGE, done);
        int inputAddress = code.allocateLocal(TypeKind.LONG);
        int outputAddress = code.allocateLocal(TypeKind.LONG);
        int accumulator = code.allocateLocal(TypeKind.LONG);
        int value = code.allocateLocal(TypeKind.LONG);
        int remaining = code.allocateLocal(TypeKind.INT);
        var slices = code.newLabel();
        var elements = code.newLabel();
        code.labelBinding(slices);
        code.aload(2).loadConstant(11).laload().lload(slice).loadConstant(2048L).lmul()
                .ladd().loadConstant(2046L).ladd().lstore(inputAddress);
        code.aload(2).loadConstant(17).laload().lload(slice).loadConstant(2048L).lmul()
                .ladd().loadConstant(2046L).ladd().lstore(outputAddress);
        code.loadConstant(1L).lstore(accumulator);
        code.loadConstant(1024).istore(remaining);
        code.labelBinding(elements);
        code.aload(1).getstatic(VALUE_LAYOUT, "JAVA_LONG_UNALIGNED", LONG_LAYOUT)
                .lload(outputAddress).loadConstant(8L).lmul().lload(accumulator)
                .invokeinterface(SEGMENT, "set", MethodTypeDesc.of(ConstantDescs.CD_void,
                        LONG_LAYOUT, ConstantDescs.CD_long, ConstantDescs.CD_long));
        code.aload(0).getstatic(VALUE_LAYOUT, "JAVA_LONG_UNALIGNED", LONG_LAYOUT)
                .lload(inputAddress).loadConstant(8L).lmul()
                .invokeinterface(SEGMENT, "get", MethodTypeDesc.of(ConstantDescs.CD_long,
                        LONG_LAYOUT, ConstantDescs.CD_long));
        code.lstore(value);
        code.lload(accumulator).lload(value).lmul().lstore(accumulator);
        code.lload(inputAddress).loadConstant(2L).lsub().lstore(inputAddress);
        code.lload(outputAddress).loadConstant(2L).lsub().lstore(outputAddress);
        code.iinc(remaining, -1).iload(remaining).branch(Opcode.IFGT, elements);
        code.lload(slice).loadConstant(1L).ladd().lstore(slice);
        code.lload(slice).lload(5).lcmp().branch(Opcode.IFLT, slices);
    }

    private static int integerAfter(String value, String marker) {
        int start = value.indexOf(marker) + marker.length(), end = value.indexOf(':', start);
        return Integer.parseInt(value.substring(start, end));
    }

    private static void emitDenseRankOne(CodeBuilder code, CpuKernelSpecialization specialization,
            DataType type, boolean product, boolean exclusive, boolean reverse) {
        var done = code.newLabel();
        code.lload(3).lload(5).lcmp().branch(Opcode.IFGE, done);
        int extent = code.allocateLocal(TypeKind.INT);
        code.aload(2).loadConstant(7).laload().l2i().istore(extent);
        code.iload(extent).branch(Opcode.IFEQ, done);
        int inputBase = code.allocateLocal(TypeKind.INT), outputBase = code.allocateLocal(TypeKind.INT);
        code.aload(2).loadConstant(10).laload().l2i().istore(inputBase);
        code.aload(2).loadConstant(14).laload().l2i().istore(outputBase);
        int accumulator = code.allocateLocal(localKind(type));
        emitIdentity(code, type, product); store(code, type, accumulator);
        int step = code.allocateLocal(TypeKind.INT); code.loadConstant(0).istore(step);
        int address = code.allocateLocal(TypeKind.INT);
        var loop = code.newLabel(); code.labelBinding(loop);
        if (reverse) code.iload(extent).loadConstant(1).isub().iload(step).isub();
        else code.iload(step);
        code.istore(address);
        int value = code.allocateLocal(localKind(type));
        var carriers = new CpuCarrierEmitter(code);
        code.iload(inputBase).iload(address).iadd().istore(address);
        carriers.load(type, specialization.carrierPattern().getFirst(), 0, address, true);
        store(code, type, value);
        if (!exclusive) emitApply(code, type, product, accumulator, value);
        code.iload(outputBase).iload(address).iload(inputBase).isub().iadd().istore(address);
        carriers.store(type, specialization.carrierPattern().getLast(), 1, address, accumulator, true);
        if (exclusive) emitApply(code, type, product, accumulator, value);
        code.iinc(step, 1); code.iload(step).iload(extent).branch(Opcode.IF_ICMPLT, loop);
        code.labelBinding(done);
    }

    private static void emitGeneral(CodeBuilder code, CpuKernelSpecialization specialization,
            DataType type, boolean product, boolean exclusive, boolean reverse, int rank, int axis) {
        int coordinates = 8, inputLayout = coordinates + rank;
        int outputLayout = inputLayout + 2 + 2 * rank;
        int slice = code.allocateLocal(TypeKind.LONG), remaining = code.allocateLocal(TypeKind.LONG);
        code.lload(3).lstore(slice);
        var done = code.newLabel();
        var slices = code.newLabel();
        code.lload(slice).lload(5).lcmp().branch(Opcode.IFGE, done); code.labelBinding(slices);
        code.lload(slice).lstore(remaining);
        for (int current = rank - 1; current >= 0; current--) {
            if (current == axis) { storePackedLong(code, coordinates + current, 0); continue; }
            code.aload(2).loadConstant(coordinates + current).lload(remaining)
                    .aload(2).loadConstant(inputLayout + 2 + current).laload().lrem().lastore();
            code.lload(remaining).aload(2).loadConstant(inputLayout + 2 + current).laload()
                    .ldiv().lstore(remaining);
        }
        int accumulator = code.allocateLocal(localKind(type));
        emitIdentity(code, type, product); store(code, type, accumulator);
        int step = code.allocateLocal(TypeKind.LONG); code.loadConstant(0L).lstore(step);
        var emptyAxis = code.newLabel();
        code.aload(2).loadConstant(7).laload().loadConstant(0L).lcmp().branch(Opcode.IFEQ, emptyAxis);
        var elements = code.newLabel(); code.labelBinding(elements);
        code.aload(2).loadConstant(coordinates + axis);
        if (reverse) code.aload(2).loadConstant(7).laload().loadConstant(1L).lsub().lload(step).lsub();
        else code.lload(step);
        code.lastore();
        int inputAddress = address(code, rank, coordinates, inputLayout);
        int value = code.allocateLocal(localKind(type));
        var carriers = new CpuCarrierEmitter(code);
        carriers.load(type, specialization.carrierPattern().getFirst(), 0, inputAddress);
        store(code, type, value);
        if (!exclusive) emitApply(code, type, product, accumulator, value);
        int outputAddress = address(code, rank, coordinates, outputLayout);
        carriers.store(type, specialization.carrierPattern().getLast(), 1, outputAddress, accumulator);
        if (exclusive) emitApply(code, type, product, accumulator, value);
        code.lload(step).loadConstant(1L).ladd().lstore(step);
        code.lload(step).aload(2).loadConstant(7).laload().lcmp().branch(Opcode.IFLT, elements);
        code.labelBinding(emptyAxis);
        code.lload(slice).loadConstant(1L).ladd().lstore(slice);
        code.lload(slice).lload(5).lcmp().branch(Opcode.IFLT, slices);
        code.labelBinding(done);
    }

    private static int address(CodeBuilder code, int rank, int coordinates, int layout) {
        int address = code.allocateLocal(TypeKind.LONG);
        code.aload(2).loadConstant(layout + 1).laload().lstore(address);
        for (int axis = 0; axis < rank; axis++) code.lload(address)
                .aload(2).loadConstant(coordinates + axis).laload()
                .aload(2).loadConstant(layout + 2 + rank + axis).laload().lmul().ladd().lstore(address);
        return address;
    }

    private static void storePackedLong(CodeBuilder code, int index, long value) {
        code.aload(2).loadConstant(index).loadConstant(value).lastore();
    }

    private static TypeKind localKind(DataType type) { return switch (type) {
        case FLOAT64 -> TypeKind.DOUBLE; case FLOAT32 -> TypeKind.FLOAT;
        case INT64 -> TypeKind.LONG; case BFLOAT16, INT32 -> TypeKind.INT;
        case BOOL -> throw new AssertionError();
    }; }
    private static void store(CodeBuilder code, DataType type, int local) { switch (type) {
        case FLOAT64 -> code.dstore(local); case FLOAT32 -> code.fstore(local);
        case INT64 -> code.lstore(local); case BFLOAT16, INT32 -> code.istore(local);
        case BOOL -> throw new AssertionError();
    } }
    private static void load(CodeBuilder code, DataType type, int local) { switch (type) {
        case FLOAT64 -> code.dload(local); case FLOAT32 -> code.fload(local);
        case INT64 -> code.lload(local); case BFLOAT16, INT32 -> code.iload(local);
        case BOOL -> throw new AssertionError();
    } }
    private static void emitIdentity(CodeBuilder code, DataType type, boolean product) { switch (type) {
        case FLOAT64 -> code.loadConstant(product ? 1.0d : 0.0d);
        case FLOAT32 -> code.loadConstant(product ? 1.0f : 0.0f);
        case INT64 -> code.loadConstant(product ? 1L : 0L);
        case INT32 -> code.loadConstant(product ? 1 : 0);
        case BFLOAT16 -> code.loadConstant(product ? 0x3f80 : 0);
        case BOOL -> throw new AssertionError();
    } }
    private static void emitApply(CodeBuilder code, DataType type, boolean product,
            int accumulator, int value) {
        load(code, type, accumulator); load(code, type, value);
        switch (type) {
            case FLOAT64 -> { if (product) code.dmul(); else code.dadd(); }
            case FLOAT32 -> { if (product) code.fmul(); else code.fadd(); }
            case INT64 -> { if (product) code.lmul(); else code.ladd(); }
            case INT32 -> { if (product) code.imul(); else code.iadd(); }
            case BFLOAT16 -> emitBfloatApply(code, product);
            case BOOL -> throw new AssertionError();
        }
        store(code, type, accumulator);
    }

    private static void emitBfloatApply(CodeBuilder code, boolean product) {
        int right = code.allocateLocal(TypeKind.INT), left = code.allocateLocal(TypeKind.INT);
        code.istore(right).istore(left);
        code.iload(left).loadConstant(16).ishl().invokestatic(FLOAT, "intBitsToFloat",
                MethodTypeDesc.of(ConstantDescs.CD_float, ConstantDescs.CD_int));
        code.iload(right).loadConstant(16).ishl().invokestatic(FLOAT, "intBitsToFloat",
                MethodTypeDesc.of(ConstantDescs.CD_float, ConstantDescs.CD_int));
        if (product) code.fmul(); else code.fadd();
        code.invokestatic(FLOAT, "floatToRawIntBits", MethodTypeDesc.of(ConstantDescs.CD_int,
                ConstantDescs.CD_float));
        int bits = code.allocateLocal(TypeKind.INT); code.istore(bits);
        var ordinary = code.newLabel(); var noRound = code.newLabel(); var round = code.newLabel();
        var done = code.newLabel();
        code.iload(bits).loadConstant(0x7f800000).iand().loadConstant(0x7f800000)
                .branch(Opcode.IF_ICMPNE, ordinary);
        code.iload(bits).loadConstant(0x7fffff).iand().branch(Opcode.IFEQ, ordinary);
        code.iload(bits).loadConstant(16).iushr().loadConstant(0x40).ior()
                .branch(Opcode.GOTO, done);
        code.labelBinding(ordinary);
        int upper = code.allocateLocal(TypeKind.INT), lower = code.allocateLocal(TypeKind.INT);
        code.iload(bits).loadConstant(16).iushr().istore(upper);
        code.iload(bits).loadConstant(0xffff).iand().istore(lower);
        code.iload(lower).loadConstant(0x8000).branch(Opcode.IF_ICMPGT, round);
        code.iload(lower).loadConstant(0x8000).branch(Opcode.IF_ICMPNE, noRound);
        code.iload(upper).loadConstant(1).iand().branch(Opcode.IFEQ, noRound);
        code.labelBinding(round).iinc(upper, 1);
        code.labelBinding(noRound).iload(upper);
        code.labelBinding(done).loadConstant(0xffff).iand();
    }

    /**
     * Adds two represented BFLOAT16 values by widening to FLOAT32 and rounding back to BFLOAT16.
     *
     * @param left unsigned 16-bit BFLOAT16 bits in the low half of the integer
     * @param right unsigned 16-bit BFLOAT16 bits in the low half of the integer
     * @return unsigned 16-bit BFLOAT16 bits for the rounded sum
     */
    static int addBfloat(int left, int right) {
        return Short.toUnsignedInt(floatToBfloat(bfloatToFloat((short) left)
                + bfloatToFloat((short) right)));
    }

    /**
     * Multiplies two represented BFLOAT16 values by widening to FLOAT32 and rounding back to
     * BFLOAT16.
     *
     * @param left unsigned 16-bit BFLOAT16 bits in the low half of the integer
     * @param right unsigned 16-bit BFLOAT16 bits in the low half of the integer
     * @return unsigned 16-bit BFLOAT16 bits for the rounded product
     */
    static int multiplyBfloat(int left, int right) {
        return Short.toUnsignedInt(floatToBfloat(bfloatToFloat((short) left)
                * bfloatToFloat((short) right)));
    }

    /**
     * Executes complete scan slices in {@code [start,end)} without splitting a slice.
     *
     * <p>The packed geometry is invocation-owned mutable coordinate state. Input and output are
     * borrowed exact primitive arrays or accessible native-order memory segments; this method
     * neither retains nor closes them.</p>
     *
     * @param input non-null readable carrier matching the packed data type
     * @param output non-null writable, non-overlapping carrier matching the packed data type
     * @param packed non-null invocation-owned scan geometry and coordinate state; mutated during
     *     execution and not retained
     * @param start non-negative inclusive independent-slice ordinal
     * @param end exclusive independent-slice ordinal no greater than the packed slice count
     * @throws NullPointerException if a required carrier or {@code packed} is {@code null}
     * @throws ArithmeticException if an array address is outside the Java index range
     * @throws IndexOutOfBoundsException if a carrier cannot cover a packed address
     * @throws IllegalStateException if a supplied memory segment is inaccessible
     */
    public static void execute(Object input, Object output, long[] packed, long start, long end) {
        int kind = (int) packed[0]; DataType type = TYPES[(int) packed[1]];
        int rank = (int) packed[2], axis = (int) packed[3];
        boolean exclusive = packed[4] != 0, reverse = packed[5] != 0;
        long axisExtent = packed[7]; int coordinates = 8;
        int inputLayout = coordinates + rank;
        int outputLayout = inputLayout + 2 + 2 * rank;
        for (long slice = start; slice < end; slice++) {
            long remaining = slice;
            for (int current = rank - 1; current >= 0; current--) {
                if (current == axis) { packed[coordinates + current] = 0; continue; }
                long extent = packed[inputLayout + 2 + current];
                packed[coordinates + current] = remaining % extent;
                remaining /= extent;
            }
            long accumulator = identity(kind, type);
            for (long step = 0; step < axisExtent; step++) {
                long coordinate = reverse ? axisExtent - 1 - step : step;
                packed[coordinates + axis] = coordinate;
                long value = readBits(input, address(packed, inputLayout, coordinates), type);
                if (!exclusive) accumulator = apply(kind, accumulator, value, type);
                writeBits(output, address(packed, outputLayout, coordinates), type, accumulator);
                if (exclusive) accumulator = apply(kind, accumulator, value, type);
            }
        }
    }

    private static long identity(int kind, DataType type) {
        if (kind == 0) return 0;
        return switch (type) {
            case FLOAT64 -> Double.doubleToRawLongBits(1.0d);
            case FLOAT32 -> Integer.toUnsignedLong(Float.floatToRawIntBits(1.0f));
            case BFLOAT16 -> 0x3f80L;
            case INT32, INT64 -> 1;
            case BOOL -> throw new AssertionError("BOOL scan is unsupported");
        };
    }
    private static long apply(int kind, long left, long right, DataType type) {
        return switch (type) {
            case INT32 -> kind == 0 ? (int) left + (int) right : (int) left * (int) right;
            case INT64 -> kind == 0 ? left + right : left * right;
            case FLOAT64 -> Double.doubleToRawLongBits(kind == 0
                    ? Double.longBitsToDouble(left) + Double.longBitsToDouble(right)
                    : Double.longBitsToDouble(left) * Double.longBitsToDouble(right));
            case FLOAT32 -> Integer.toUnsignedLong(Float.floatToRawIntBits(kind == 0
                    ? Float.intBitsToFloat((int) left) + Float.intBitsToFloat((int) right)
                    : Float.intBitsToFloat((int) left) * Float.intBitsToFloat((int) right)));
            case BFLOAT16 -> Short.toUnsignedLong(floatToBfloat(kind == 0
                    ? bfloatToFloat((short) left) + bfloatToFloat((short) right)
                    : bfloatToFloat((short) left) * bfloatToFloat((short) right)));
            case BOOL -> throw new AssertionError("BOOL scan is unsupported");
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
            case BOOL -> throw new AssertionError("BOOL scan is unsupported");
        };
    }
    private static void writeBits(Object carrier, long address, DataType type, long bits) {
        switch (type) {
            case FLOAT64 -> { double v = Double.longBitsToDouble(bits); if (carrier instanceof double[] a) a[Math.toIntExact(address)] = v; else ((MemorySegment) carrier).set(ValueLayout.JAVA_DOUBLE, address * 8, v); }
            case FLOAT32 -> { float v = Float.intBitsToFloat((int) bits); if (carrier instanceof float[] a) a[Math.toIntExact(address)] = v; else ((MemorySegment) carrier).set(ValueLayout.JAVA_FLOAT, address * 4, v); }
            case BFLOAT16 -> { short v = (short) bits; if (carrier instanceof short[] a) a[Math.toIntExact(address)] = v; else ((MemorySegment) carrier).set(ValueLayout.JAVA_SHORT, address * 2, v); }
            case INT32 -> { int v = (int) bits; if (carrier instanceof int[] a) a[Math.toIntExact(address)] = v; else ((MemorySegment) carrier).set(ValueLayout.JAVA_INT, address * 4, v); }
            case INT64 -> { if (carrier instanceof long[] a) a[Math.toIntExact(address)] = bits; else ((MemorySegment) carrier).set(ValueLayout.JAVA_LONG, address * 8, bits); }
            case BOOL -> throw new AssertionError("BOOL scan is unsupported");
        }
    }
    private static float bfloatToFloat(short bits) { return Float.intBitsToFloat(Short.toUnsignedInt(bits) << 16); }
    private static short floatToBfloat(float value) {
        int bits = Float.floatToRawIntBits(value);
        if ((bits & 0x7f800000) == 0x7f800000 && (bits & 0x7fffff) != 0)
            return (short) ((bits >>> 16) | 0x40);
        int upper = bits >>> 16, lower = bits & 0xffff;
        if (lower > 0x8000 || lower == 0x8000 && (upper & 1) != 0) upper++;
        return (short) upper;
    }
}

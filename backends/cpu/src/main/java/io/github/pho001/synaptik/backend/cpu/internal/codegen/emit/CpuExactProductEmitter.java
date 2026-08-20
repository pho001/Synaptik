package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

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
 * Emits a complete type-specialized exact floating product into a generated CPU entry.
 *
 * <p>This generation-time owner is shared by scatter multiplication and ordinary aggregate
 * products. Generated execution uses a prevalidated primitive-limb {@link MemorySegment} state;
 * it contains no dependency on this emitter or on a reference kernel. Ordinary aggregates clear
 * the complete declared slice. Scatter's grouped-product path may instead keep header state in
 * generated locals and initialize only the reachable limb prefix, because one output's state is
 * completed before the same range reuses the slice for another output.</p>
 */
final class CpuExactProductEmitter {
    private static final ClassDesc SEGMENT = ClassDesc.of(MemorySegment.class.getName());
    private static final ClassDesc VALUE_LAYOUT = ClassDesc.of(ValueLayout.class.getName());
    private static final ClassDesc LONG_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout$OfLong");
    private static final ClassDesc LONG_CLASS = ClassDesc.of(Long.class.getName());
    private static final ClassDesc DOUBLE_CLASS = ClassDesc.of(Double.class.getName());
    private static final ClassDesc FLOAT_CLASS = ClassDesc.of(Float.class.getName());
    private static final ClassDesc MATH_CLASS = ClassDesc.of(Math.class.getName());
    private static final long NAN = 1, ZERO = 2, INFINITY = 4, SIGN = 8;
    private final CodeBuilder code;
    private final DataType type;
    private final int scratch;
    private final int geometry;
    private final boolean localState;
    private final int offset;
    private final int bytes;
    private final int flags;
    private final int bits;
    private final int fraction;
    private final int exponentField;
    private final int significand;
    private final int exponent;
    private final int totalExponent;
    private final int used;
    private final int limb;
    private final int factor;
    private final int carry;
    private final int word;
    private final int low;
    private final int high;
    private final int sum;
    private final int position;
    private final int top;
    private final int bitLength;
    private final int unbiased;
    private final int shift;
    private final int quotient;
    private final int normal;
    private final int guard;
    private final int sticky;
    private final int bitOffset;
    private final int fullLimbs;
    private final int remainingBits;
    private final int result;

    /**
     * Creates scatter-product emission using the established scatter geometry indices.
     *
     * @param code Class-File method builder mutated by subsequent emission; not {@code null}
     * @param type exact floating represented type
     * @param scratch local-variable slot containing the exact-product state segment
     * @param geometry local-variable slot containing packed scatter geometry
     * @throws NullPointerException if {@code code} is {@code null}
     * @throws IllegalArgumentException if {@code type} is not supported
     */
    CpuExactProductEmitter(CodeBuilder code, DataType type, int scratch, int geometry) {
        this(code, type, scratch, geometry, 13, 14, false);
    }

    /**
     * Creates exact-product emission using explicit packed state-location indices.
     *
     * @param code Class-File method builder mutated by subsequent emission; not {@code null}
     * @param type exact floating represented type
     * @param scratch local-variable slot containing the exact-product state segment
     * @param geometry local-variable slot containing packed family geometry
     * @param offsetIndex packed geometry index of the state-slice byte offset
     * @param bytesIndex packed geometry index of the state-slice byte size
     * @throws NullPointerException if {@code code} is {@code null}
     * @throws IllegalArgumentException if {@code type} is not supported
     */
    CpuExactProductEmitter(CodeBuilder code, DataType type, int scratch, int geometry,
            int offsetIndex, int bytesIndex) {
        this(code, type, scratch, geometry, offsetIndex, bytesIndex, false);
    }

    /**
     * Creates exact-product emission with explicit state layout and header-locality policy.
     *
     * @param code Class-File method builder mutated by subsequent emission; not {@code null}
     * @param type exact floating represented type
     * @param scratch local-variable slot containing the exact-product state segment
     * @param geometry local-variable slot containing packed family geometry
     * @param offsetIndex packed geometry index of the state-slice byte offset
     * @param bytesIndex packed geometry index of the state-slice byte size
     * @param localState whether flags, exponent, and used-limb count remain generated locals until
     *     finalization instead of being reloaded for independently encountered factors; use only
     *     when one complete factor group is emitted and finished before state reuse
     * @throws NullPointerException if {@code code} is {@code null}
     * @throws IllegalArgumentException if {@code type} is not supported
     */
    CpuExactProductEmitter(CodeBuilder code, DataType type, int scratch, int geometry,
            int offsetIndex, int bytesIndex, boolean localState) {
        if (type != DataType.FLOAT64 && type != DataType.FLOAT32 && type != DataType.BFLOAT16)
            throw new IllegalArgumentException("exact product requires a floating type");
        this.code = code;
        this.type = type;
        this.scratch = scratch;
        this.geometry = geometry;
        this.localState = localState;
        offset = code.allocateLocal(TypeKind.LONG);
        bytes = code.allocateLocal(TypeKind.LONG);
        flags = code.allocateLocal(TypeKind.LONG);
        bits = code.allocateLocal(TypeKind.LONG);
        fraction = code.allocateLocal(TypeKind.LONG);
        exponentField = code.allocateLocal(TypeKind.LONG);
        significand = code.allocateLocal(TypeKind.LONG);
        exponent = code.allocateLocal(TypeKind.LONG);
        totalExponent = code.allocateLocal(TypeKind.LONG);
        used = code.allocateLocal(TypeKind.INT);
        limb = code.allocateLocal(TypeKind.INT);
        factor = code.allocateLocal(TypeKind.LONG);
        carry = code.allocateLocal(TypeKind.LONG);
        word = code.allocateLocal(TypeKind.LONG);
        low = code.allocateLocal(TypeKind.LONG);
        high = code.allocateLocal(TypeKind.LONG);
        sum = code.allocateLocal(TypeKind.LONG);
        position = code.allocateLocal(TypeKind.LONG);
        top = code.allocateLocal(TypeKind.LONG);
        bitLength = code.allocateLocal(TypeKind.INT);
        unbiased = code.allocateLocal(TypeKind.LONG);
        shift = code.allocateLocal(TypeKind.LONG);
        quotient = code.allocateLocal(TypeKind.LONG);
        normal = code.allocateLocal(TypeKind.INT);
        guard = code.allocateLocal(TypeKind.INT);
        sticky = code.allocateLocal(TypeKind.INT);
        bitOffset = code.allocateLocal(TypeKind.INT);
        fullLimbs = code.allocateLocal(TypeKind.INT);
        remainingBits = code.allocateLocal(TypeKind.INT);
        result = code.allocateLocal(TypeKind.LONG);
        geometry(code, geometry, offsetIndex).lstore(offset);
        geometry(code, geometry, bytesIndex).lstore(bytes);
    }

    /**
     * Emits scatter's initial base/update pair handling without changing existing semantics.
     *
     * @param left local-variable slot containing the base represented value
     * @param right local-variable slot containing the update represented value
     * @param found local-variable slot indicating whether an earlier update initialized state
     */
    void emitFactors(int left, int right, int found) {
        emitFactors(left, right, found, true);
    }

    /**
     * Emits scatter's base and current update while leaving unreachable unused limb capacity
     * untouched. The caller must finish one target group before reusing the scratch slice.
     *
     * @param left local-variable slot containing the base represented value
     * @param right local-variable slot containing the update represented value
     * @param found local-variable slot indicating whether an earlier update initialized state
     */
    void emitScatterFactors(int left, int right, int found) {
        emitFactors(left, right, found, false);
    }

    private void emitFactors(int left, int right, int found, boolean clearUnused) {
        var initialized = code.newLabel();
        code.iload(found).branch(Opcode.IFNE, initialized);
        emitReset(clearUnused);
        emitFactor(left);
        code.labelBinding(initialized);
        emitFactor(right);
    }

    /** Emits deterministic multiplicative-identity initialization for one exact state slice. */
    void emitReset() {
        emitReset(true);
    }

    /**
     * Emits grouped-scatter initialization without stores to limbs outside the initial used
     * prefix. Subsequent factor emission overwrites every limb that becomes reachable.
     */
    void emitScatterReset() {
        emitReset(false);
    }

    private void emitReset(boolean clearUnused) {
        code.loadConstant(0L).lstore(flags);
        code.loadConstant(0L).lstore(totalExponent);
        code.loadConstant(1).istore(used);
        set(0, () -> code.loadConstant(0L));
        set(8, () -> code.loadConstant(0L));
        set(16, () -> code.loadConstant(1L));
        if (clearUnused) {
            code.lload(offset).loadConstant(24L).ladd().lstore(position);
            var loop = code.newLabel();
            var done = code.newLabel();
            code.labelBinding(loop);
            code.lload(position).lload(offset).lload(bytes).ladd().lcmp().branch(Opcode.IFGE, done);
            setAt(position, () -> code.loadConstant(0L));
            code.lload(position)
                    .loadConstant(8L)
                    .ladd()
                    .lstore(position)
                    .branch(Opcode.GOTO, loop)
                    .labelBinding(done);
        }
        set(24, () -> code.loadConstant(1L));
    }

    /**
     * Emits special-value classification and exact unsigned-limb multiplication for one factor.
     *
     * @param value local-variable slot containing a represented floating factor
     */
    void emitFactor(int value) {
        loadBits(value);
        code.lstore(bits);
        if (!localState) get(0).lstore(flags);
        code.lload(bits).loadConstant(signMask()).land().loadConstant(0L).lcmp();
        var positive = code.newLabel();
        code.branch(Opcode.IFEQ, positive);
        code.lload(flags).loadConstant(SIGN).lxor().lstore(flags).labelBinding(positive);
        code.lload(bits).loadConstant(fractionMask()).land().lstore(fraction);
        code.lload(bits)
                .loadConstant(fractionBits())
                .lushr()
                .loadConstant(exponentMask())
                .land()
                .lstore(exponentField);
        var finite = code.newLabel();
        code.lload(exponentField)
                .loadConstant(exponentMask())
                .lcmp()
                .branch(Opcode.IFNE, finite);
        var infinity = code.newLabel();
        code.lload(fraction).loadConstant(0L).lcmp().branch(Opcode.IFEQ, infinity);
        code.lload(flags).loadConstant(NAN).lor().lstore(flags);
        var classified = code.newLabel();
        code.branch(Opcode.GOTO, classified).labelBinding(infinity);
        code.lload(flags)
                .loadConstant(INFINITY)
                .lor()
                .lstore(flags)
                .branch(Opcode.GOTO, classified)
                .labelBinding(finite);
        var nonzero = code.newLabel();
        code.lload(exponentField).loadConstant(0L).lcmp().branch(Opcode.IFNE, nonzero);
        code.lload(fraction).loadConstant(0L).lcmp().branch(Opcode.IFNE, nonzero);
        code.lload(flags)
                .loadConstant(ZERO)
                .lor()
                .lstore(flags)
                .branch(Opcode.GOTO, classified)
                .labelBinding(nonzero);
        var normalFactor = code.newLabel();
        var factorReady = code.newLabel();
        code.lload(exponentField).loadConstant(0L).lcmp().branch(Opcode.IFNE, normalFactor);
        code.lload(fraction).lstore(significand);
        code.loadConstant((long) (1 - bias() - fractionBits()))
                .lstore(exponent)
                .branch(Opcode.GOTO, factorReady)
                .labelBinding(normalFactor);
        code.loadConstant(1L << fractionBits()).lload(fraction).lor().lstore(significand);
        code.lload(exponentField)
                .loadConstant((long) (bias() + fractionBits()))
                .lsub()
                .lstore(exponent)
                .labelBinding(factorReady);
        if (localState) code.lload(totalExponent); else get(8);
        code.lload(exponent).ladd().lstore(totalExponent);
        if (!localState) set(8, () -> code.lload(totalExponent));
        code.lload(significand).lstore(factor);
        emitMultiply();
        code.labelBinding(classified);
        if (!localState) set(0, () -> code.lload(flags));
    }

    private void emitMultiply() {
        if (!localState) get(16).l2i().istore(used);
        code.loadConstant(0L).lstore(carry);
        code.loadConstant(0).istore(limb);
        var loop = code.newLabel();
        var done = code.newLabel();
        code.labelBinding(loop).iload(limb).iload(used).branch(Opcode.IF_ICMPGE, done);
        limbOffset(limb);
        code.lstore(position);
        getAt(position).lstore(word);
        code.lload(word).lload(factor).lmul().lstore(low);
        code.lload(word)
                .lload(factor)
                .invokestatic(
                        MATH_CLASS,
                        "unsignedMultiplyHigh",
                        MethodTypeDesc.of(
                                ConstantDescs.CD_long,
                                ConstantDescs.CD_long,
                                ConstantDescs.CD_long))
                .lstore(high);
        code.lload(low).lload(carry).ladd().lstore(sum);
        code.lload(sum)
                .lload(low)
                .invokestatic(
                        LONG_CLASS,
                        "compareUnsigned",
                        MethodTypeDesc.of(
                                ConstantDescs.CD_int,
                                ConstantDescs.CD_long,
                                ConstantDescs.CD_long));
        var noOverflow = code.newLabel();
        code.branch(Opcode.IFGE, noOverflow);
        code.lload(high).loadConstant(1L).ladd().lstore(high).labelBinding(noOverflow);
        setAt(position, () -> code.lload(sum));
        code.lload(high).lstore(carry);
        code.iinc(limb, 1).branch(Opcode.GOTO, loop).labelBinding(done);
        var noCarry = code.newLabel();
        code.lload(carry).loadConstant(0L).lcmp().branch(Opcode.IFEQ, noCarry);
        limbOffset(used);
        code.lstore(position);
        setAt(position, () -> code.lload(carry));
            code.iinc(used, 1);
            if (!localState) set(16, () -> code.iload(used).i2l());
        code.labelBinding(noCarry);
    }

    /**
     * Emits special-value resolution and one ties-to-even conversion into a result local.
     *
     * @param value local-variable slot receiving the represented result
     * @param found local-variable slot indicating whether a scatter destination was encountered;
     *     ordinary aggregates pass a constant true local
     */
    void emitFinish(int value, int found) {
        var absent = code.newLabel();
        var store = code.newLabel();
        code.iload(found).branch(Opcode.IFEQ, absent);
        if (!localState) get(0).lstore(flags);
        else {
            set(0, () -> code.lload(flags));
            set(8, () -> code.lload(totalExponent));
            set(16, () -> code.iload(used).i2l());
        }
        code.loadConstant(0L).lstore(result);
        code.lload(flags).loadConstant(SIGN).land().loadConstant(0L).lcmp();
        var positive = code.newLabel();
        code.branch(Opcode.IFEQ, positive);
        code.loadConstant(signBit()).lstore(result).labelBinding(positive);
        code.lload(flags).loadConstant(NAN).land().loadConstant(0L).lcmp();
        var checkZeroInfinity = code.newLabel();
        code.branch(Opcode.IFEQ, checkZeroInfinity);
        code.loadConstant(canonicalNan()).lstore(result)
                .branch(Opcode.GOTO, store)
                .labelBinding(checkZeroInfinity);
        code.lload(flags)
                .loadConstant(ZERO | INFINITY)
                .land()
                .loadConstant(ZERO | INFINITY)
                .lcmp();
        var checkInfinity = code.newLabel();
        code.branch(Opcode.IFNE, checkInfinity);
        code.loadConstant(canonicalNan()).lstore(result)
                .branch(Opcode.GOTO, store)
                .labelBinding(checkInfinity);
        code.lload(flags).loadConstant(INFINITY).land().loadConstant(0L).lcmp();
        var checkZero = code.newLabel();
        code.branch(Opcode.IFEQ, checkZero);
        code.lload(result)
                .loadConstant(positiveInfinity())
                .lor()
                .lstore(result)
                .branch(Opcode.GOTO, store)
                .labelBinding(checkZero);
        code.lload(flags)
                .loadConstant(ZERO)
                .land()
                .loadConstant(0L)
                .lcmp()
                .branch(Opcode.IFNE, store);
        emitFiniteFinish(store);
        code.labelBinding(store);
        storeResult(value);
        code.labelBinding(absent);
    }

    private void emitFiniteFinish(java.lang.classfile.Label store) {
        if (!localState) get(16).l2i().istore(used);
        limbOffsetMinusOne();
        code.lstore(position);
        getAt(position).lstore(top);
        code.iload(used)
                .loadConstant(1)
                .isub()
                .loadConstant(64)
                .imul()
                .loadConstant(64)
                .lload(top)
                .invokestatic(
                        LONG_CLASS,
                        "numberOfLeadingZeros",
                        MethodTypeDesc.of(ConstantDescs.CD_int, ConstantDescs.CD_long))
                .isub()
                .iadd()
                .istore(bitLength);
        if (localState) code.lload(totalExponent); else get(8);
        code.iload(bitLength).i2l().ladd().loadConstant(1L).lsub().lstore(unbiased);
        code.lload(unbiased).loadConstant((long) maxExponent()).lcmp();
        var notOverflow = code.newLabel();
        code.branch(Opcode.IFLE, notOverflow);
        code.lload(result)
                .loadConstant(positiveInfinity())
                .lor()
                .lstore(result)
                .branch(Opcode.GOTO, store)
                .labelBinding(notOverflow);
        code.lload(unbiased).loadConstant((long) minimumNormal()).lcmp();
        var subnormal = code.newLabel();
        var round = code.newLabel();
        code.branch(Opcode.IFLT, subnormal);
        code.iload(bitLength).loadConstant(precision()).isub().i2l().lstore(shift);
        code.loadConstant(1).istore(normal).branch(Opcode.GOTO, round).labelBinding(subnormal);
        code.loadConstant((long) (minimumNormal() - (precision() - 1)));
        if (localState) code.lload(totalExponent); else get(8);
        code.lsub().lstore(shift);
        code.loadConstant(0).istore(normal).labelBinding(round);
        emitRounded();
        var finishSubnormal = code.newLabel();
        code.iload(normal).branch(Opcode.IFEQ, finishSubnormal);
        code.lload(quotient).loadConstant(1L << precision()).lcmp();
        var normalWidth = code.newLabel();
        code.branch(Opcode.IFNE, normalWidth);
        code.lload(quotient).loadConstant(1).lushr().lstore(quotient);
        code.lload(unbiased).loadConstant(1L).ladd().lstore(unbiased);
        code.lload(unbiased).loadConstant((long) maxExponent()).lcmp();
        var roundedNormal = code.newLabel();
        code.branch(Opcode.IFLE, roundedNormal);
        code.lload(result)
                .loadConstant(positiveInfinity())
                .lor()
                .lstore(result)
                .branch(Opcode.GOTO, store)
                .labelBinding(roundedNormal)
                .labelBinding(normalWidth);
        code.lload(unbiased).loadConstant((long) bias()).ladd().lstore(exponentField);
        code.lload(result)
                .lload(exponentField)
                .loadConstant(fractionBits())
                .lshl()
                .lload(quotient)
                .loadConstant(fractionMask())
                .land()
                .lor()
                .lor()
                .lstore(result)
                .branch(Opcode.GOTO, store)
                .labelBinding(finishSubnormal);
        code.lload(quotient).loadConstant(0L).lcmp().branch(Opcode.IFEQ, store);
        code.lload(quotient).loadConstant(1L << fractionBits()).lcmp();
        var trueSubnormal = code.newLabel();
        code.branch(Opcode.IFLT, trueSubnormal);
        code.lload(result)
                .loadConstant(1L << fractionBits())
                .lor()
                .lstore(result)
                .branch(Opcode.GOTO, store)
                .labelBinding(trueSubnormal);
        code.lload(result).lload(quotient).lor().lstore(result).branch(Opcode.GOTO, store);
    }

    private void emitRounded() {
        var positiveShift = code.newLabel();
        var withinProduct = code.newLabel();
        var quotientReady = code.newLabel();
        var afterShiftedLow = code.newLabel();
        code.lload(shift).loadConstant(0L).lcmp().branch(Opcode.IFGT, positiveShift);
        get(24).lload(shift)
                .lneg()
                .l2i()
                .lshl()
                .lstore(quotient)
                .branch(Opcode.GOTO, quotientReady)
                .labelBinding(positiveShift);
        code.lload(shift).iload(bitLength).i2l().lcmp().branch(Opcode.IFLE, withinProduct);
        code.loadConstant(0L)
                .lstore(quotient)
                .branch(Opcode.GOTO, quotientReady)
                .labelBinding(withinProduct);
        code.lload(shift).iload(bitLength).i2l().lcmp();
        var belowLength = code.newLabel();
        code.branch(Opcode.IFLT, belowLength);
        code.loadConstant(0L)
                .lstore(quotient)
                .branch(Opcode.GOTO, afterShiftedLow)
                .labelBinding(belowLength);
        code.lload(shift).loadConstant(6).lushr().l2i().istore(limb);
        code.lload(shift).loadConstant(63L).land().l2i().istore(bitOffset);
        limbOffset(limb);
        code.lstore(position);
        getAt(position).iload(bitOffset).lushr().lstore(quotient);
        code.iload(bitOffset);
        var noNextWord = code.newLabel();
        code.branch(Opcode.IFEQ, noNextWord);
        code.iload(limb)
                .loadConstant(1)
                .iadd()
                .iload(used)
                .branch(Opcode.IF_ICMPGE, noNextWord);
        code.iinc(limb, 1);
        limbOffset(limb);
        code.lstore(position);
        getAt(position).loadConstant(64).iload(bitOffset).isub().lshl();
        code.lload(quotient)
                .lor()
                .lstore(quotient)
                .labelBinding(noNextWord)
                .labelBinding(afterShiftedLow)
                .labelBinding(quotientReady);
        code.loadConstant(0).istore(guard);
        code.loadConstant(0).istore(sticky);
        code.lload(shift).loadConstant(1L).lsub().lstore(position);
        code.lload(position).loadConstant(0L).lcmp();
        var guardDone = code.newLabel();
        code.branch(Opcode.IFLT, guardDone);
        code.lload(position).loadConstant(6).lushr().l2i().istore(limb);
        code.iload(limb).iload(used).branch(Opcode.IF_ICMPGE, guardDone);
        code.lload(position).loadConstant(63L).land().l2i().istore(bitOffset);
        limbOffset(limb);
        code.lstore(position);
        getAt(position)
                .iload(bitOffset)
                .lushr()
                .loadConstant(1L)
                .land()
                .l2i()
                .istore(guard)
                .labelBinding(guardDone);
        code.lload(shift).loadConstant(1L).lsub().lstore(position);
        code.lload(position).loadConstant(0L).lcmp();
        var stickyDone = code.newLabel();
        code.branch(Opcode.IFLE, stickyDone);
        code.lload(position).loadConstant(6).lushr().l2i().istore(fullLimbs);
        code.lload(position).loadConstant(63L).land().l2i().istore(remainingBits);
        code.loadConstant(0).istore(limb);
        var stickyLoop = code.newLabel();
        var afterFull = code.newLabel();
        code.labelBinding(stickyLoop)
                .iload(limb)
                .iload(fullLimbs)
                .branch(Opcode.IF_ICMPGE, afterFull);
        code.iload(limb).iload(used).branch(Opcode.IF_ICMPGE, afterFull);
        limbOffset(limb);
        code.lstore(position);
        getAt(position).loadConstant(0L).lcmp();
        var next = code.newLabel();
        code.branch(Opcode.IFEQ, next);
        code.loadConstant(1)
                .istore(sticky)
                .branch(Opcode.GOTO, stickyDone)
                .labelBinding(next)
                .iinc(limb, 1)
                .branch(Opcode.GOTO, stickyLoop)
                .labelBinding(afterFull);
        code.iload(remainingBits).branch(Opcode.IFEQ, stickyDone);
        code.iload(fullLimbs).iload(used).branch(Opcode.IF_ICMPGE, stickyDone);
        limbOffset(fullLimbs);
        code.lstore(position);
        getAt(position)
                .loadConstant(1L)
                .iload(remainingBits)
                .lshl()
                .loadConstant(1L)
                .lsub()
                .land()
                .loadConstant(0L)
                .lcmp();
        code.branch(Opcode.IFEQ, stickyDone);
        code.loadConstant(1).istore(sticky).labelBinding(stickyDone);
        code.iload(guard);
        var rounded = code.newLabel();
        code.branch(Opcode.IFEQ, rounded);
        code.iload(sticky);
        var roundUp = code.newLabel();
        code.branch(Opcode.IFNE, roundUp);
        code.lload(quotient)
                .loadConstant(1L)
                .land()
                .loadConstant(0L)
                .lcmp()
                .branch(Opcode.IFEQ, rounded)
                .labelBinding(roundUp);
        code.lload(quotient).loadConstant(1L).ladd().lstore(quotient).labelBinding(rounded);
    }

    private void loadBits(int value) {
        switch (type) {
            case FLOAT64 ->
                    code.dload(value)
                            .invokestatic(
                                    DOUBLE_CLASS,
                                    "doubleToRawLongBits",
                                    MethodTypeDesc.of(
                                            ConstantDescs.CD_long, ConstantDescs.CD_double));
            case FLOAT32 ->
                    code.fload(value)
                            .invokestatic(
                                    FLOAT_CLASS,
                                    "floatToRawIntBits",
                                    MethodTypeDesc.of(
                                            ConstantDescs.CD_int, ConstantDescs.CD_float))
                            .i2l()
                            .loadConstant(0xffffffffL)
                            .land();
            case BFLOAT16 -> code.iload(value).i2l().loadConstant(0xffffL).land();
            default -> throw new IllegalArgumentException("not a floating product");
        }
    }

    private void storeResult(int value) {
        switch (type) {
            case FLOAT64 ->
                    code.lload(result)
                            .invokestatic(
                                    DOUBLE_CLASS,
                                    "longBitsToDouble",
                                    MethodTypeDesc.of(
                                            ConstantDescs.CD_double, ConstantDescs.CD_long))
                            .dstore(value);
            case FLOAT32 ->
                    code.lload(result)
                            .l2i()
                            .invokestatic(
                                    FLOAT_CLASS,
                                    "intBitsToFloat",
                                    MethodTypeDesc.of(
                                            ConstantDescs.CD_float, ConstantDescs.CD_int))
                            .fstore(value);
            case BFLOAT16 -> code.lload(result).l2i().istore(value);
            default -> throw new IllegalArgumentException("not a floating product");
        }
    }

    private CodeBuilder get(int delta) {
        segmentOffset(delta);
        return code.invokeinterface(
                SEGMENT,
                "get",
                MethodTypeDesc.of(ConstantDescs.CD_long, LONG_LAYOUT, ConstantDescs.CD_long));
    }

    private CodeBuilder getAt(int address) {
        return code.aload(scratch)
                .getstatic(VALUE_LAYOUT, "JAVA_LONG", LONG_LAYOUT)
                .lload(address)
                .invokeinterface(
                        SEGMENT,
                        "get",
                        MethodTypeDesc.of(
                                ConstantDescs.CD_long, LONG_LAYOUT, ConstantDescs.CD_long));
    }

    private void set(int delta, Runnable value) {
        segmentOffset(delta);
        value.run();
        code.invokeinterface(
                SEGMENT,
                "set",
                MethodTypeDesc.of(
                        ConstantDescs.CD_void,
                        LONG_LAYOUT,
                        ConstantDescs.CD_long,
                        ConstantDescs.CD_long));
    }

    private void setAt(int address, Runnable value) {
        code.aload(scratch).getstatic(VALUE_LAYOUT, "JAVA_LONG", LONG_LAYOUT).lload(address);
        value.run();
        code.invokeinterface(
                SEGMENT,
                "set",
                MethodTypeDesc.of(
                        ConstantDescs.CD_void,
                        LONG_LAYOUT,
                        ConstantDescs.CD_long,
                        ConstantDescs.CD_long));
    }

    private void segmentOffset(int delta) {
        code.aload(scratch).getstatic(VALUE_LAYOUT, "JAVA_LONG", LONG_LAYOUT).lload(offset);
        if (delta != 0) code.loadConstant((long) delta).ladd();
    }

    private void limbOffset(int index) {
        code.lload(offset)
                .loadConstant(24L)
                .ladd()
                .iload(index)
                .i2l()
                .loadConstant(8L)
                .lmul()
                .ladd();
    }

    private void limbOffsetMinusOne() {
        code.lload(offset)
                .loadConstant(24L)
                .ladd()
                .iload(used)
                .loadConstant(1)
                .isub()
                .i2l()
                .loadConstant(8L)
                .lmul()
                .ladd();
    }

    private long signMask() {
        return type == DataType.FLOAT64
                ? 1L << 63
                : type == DataType.FLOAT32 ? 1L << 31 : 1L << 15;
    }

    private int fractionBits() {
        return type == DataType.FLOAT64 ? 52 : type == DataType.FLOAT32 ? 23 : 7;
    }

    private int bias() {
        return type == DataType.FLOAT64 ? 1023 : 127;
    }

    private int precision() {
        return fractionBits() + 1;
    }

    private int maxExponent() {
        return type == DataType.FLOAT64 ? 1023 : 127;
    }

    private int minimumNormal() {
        return 1 - bias();
    }

    private long fractionMask() {
        return (1L << fractionBits()) - 1;
    }

    private long exponentMask() {
        return type == DataType.FLOAT64 ? 0x7ffL : 0xffL;
    }

    private long signBit() {
        return type == DataType.FLOAT64
                ? 1L << 63
                : type == DataType.FLOAT32 ? 1L << 31 : 1L << 15;
    }

    private long canonicalNan() {
        return type == DataType.FLOAT64
                ? 0x7ff8000000000000L
                : type == DataType.FLOAT32 ? 0x7fc00000L : 0x7fc0L;
    }

    private long positiveInfinity() {
        return type == DataType.FLOAT64
                ? 0x7ff0000000000000L
                : type == DataType.FLOAT32 ? 0x7f800000L : 0x7f80L;
    }

    private static CodeBuilder geometry(CodeBuilder code, int slot, int index) {
        return code.aload(slot).loadConstant(index).laload();
    }
}

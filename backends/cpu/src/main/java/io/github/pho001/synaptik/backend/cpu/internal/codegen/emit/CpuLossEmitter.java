package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratorSchema;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy.Compute;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.loss.LossKind;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.ClassBuilder;
import java.lang.classfile.Label;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;
import java.lang.reflect.AccessFlag;

/**
 * Emits field-free, direct scalar Class-File bodies for the CPU loss family.
 *
 * <p>The generated entry receives only typed carriers, a cold primitive geometry array, and a
 * range. It decodes ordered semantic input roles from the loss identity, so equal MSE or dense
 * roles may share one physical read boundary without changing their semantics. Geometry contains
 * ranks, extents, bases, and strides only; none of those facts participates in generated-class
 * identity.</p>
 *
 * <p>MSE visits increasing elements. Categorical forms visit increasing non-class samples and
 * then increasing classes, use the frozen max/shifted-exponential/log-sum-exp traversal, and do
 * not call an external Synaptik helper, reference kernel, bridge, or dispatcher from generated
 * code.
 * BFLOAT16 and FLOAT32 bodies use binary32 accumulator locals; FLOAT64 bodies use binary64.
 * Index targets are loaded and compared to a present ignore value before any logits load. The
 * prepared executable's cold pre-write validator proves every non-ignored target is in range
 * before this entry can run, so the generated hot loop neither allocates an exception nor repeats
 * whole-invocation validation.</p>
 */
public final class CpuLossEmitter {
    /** Name of the private body selected when cold geometry proves contiguous int addresses. */
    static final String CONTIGUOUS_INT_NAME = "lossContiguousInt";

    /** Name of the private body selected for the general affine-address geometry. */
    static final String GENERIC_AFFINE_NAME = "lossGenericAffine";
    private static final ClassDesc STRICT_MATH = ClassDesc.of(StrictMath.class.getName());
    private static final ClassDesc FLOAT = ClassDesc.of(Float.class.getName());

    /** Creates a stateless generation-time loss emitter. */
    public CpuLossEmitter() { }

    /**
     * Emits one complete typed loss entry body.
     *
     * @param classBuilder non-null Class-File builder receiving the generated body
     * @param owner non-null generated class that declares the direct entry and private bodies
     * @param type non-null typed descriptor shared by the generated entry and both private bodies
     * @param specialization non-null schema-58 scalar carrier specialization
     * @param ir non-null canonical loss identity with no executable instructions
     * @throws NullPointerException if any parameter is {@code null}
     * @throws IllegalArgumentException if the supplied facts do not describe one direct loss form
     */
    public void emit(ClassBuilder classBuilder, ClassDesc owner, MethodTypeDesc type,
            CpuKernelSpecialization specialization, CpuKernelIr ir) {
        classBuilder.withMethod(CpuGeneratorSchema.ENTRY_NAME, type,
                AccessFlag.PUBLIC.mask() | AccessFlag.STATIC.mask(),
                method -> method.withCode(code -> emitDispatch(code, owner, type, specialization, ir)));
        classBuilder.withMethod(CONTIGUOUS_INT_NAME, type,
                AccessFlag.PRIVATE.mask() | AccessFlag.STATIC.mask(), method -> method.withCode(code -> {
                    emitContiguousIntBody(code, specialization, ir); code.return_();
                }));
        classBuilder.withMethod(GENERIC_AFFINE_NAME, type,
                AccessFlag.PRIVATE.mask() | AccessFlag.STATIC.mask(), method -> method.withCode(code -> {
                    emitGenericAffine(code, specialization, ir); code.return_();
                }));
    }

    private static void emitDispatch(CodeBuilder code, ClassDesc owner, MethodTypeDesc type,
            CpuKernelSpecialization specialization, CpuKernelIr ir) {
        int geometry = specialization.boundaryDataTypes().size();
        int rank = code.allocateLocal(TypeKind.INT), axis = code.allocateLocal(TypeKind.INT);
        int targetRank = code.allocateLocal(TypeKind.INT), outputRank = code.allocateLocal(TypeKind.INT);
        geometry(code, geometry, 0).l2i().istore(rank);
        geometry(code, geometry, 1).l2i().istore(axis);
        geometry(code, geometry, 2).l2i().istore(targetRank);
        geometry(code, geometry, 3).l2i().istore(outputRank);
        Label generic = code.newLabel();
        LossKind kind = LossKind.valueOf(field(ir.familyIdentity(), "kind="));
        emitIntGeometryGuard(code, geometry, generic);
        if (kind == LossKind.MEAN_SQUARED_ERROR) {
            emitContiguousMseGuard(code, geometry, rank, targetRank, outputRank, generic);
        } else {
            int outer = code.allocateLocal(TypeKind.LONG), inner = code.allocateLocal(TypeKind.LONG);
            emitContiguousCategoricalGuard(code, geometry, rank, axis, targetRank, outputRank,
                    geometry + 1, geometry + 3,
                    LossReduction.valueOf(field(ir.familyIdentity(), "reduction=")), generic, outer, inner);
        }
        invokeHelper(code, owner, CONTIGUOUS_INT_NAME, type, geometry); code.return_();
        code.labelBinding(generic);
        invokeHelper(code, owner, GENERIC_AFFINE_NAME, type, geometry); code.return_();
    }

    private static void invokeHelper(CodeBuilder code, ClassDesc owner, String name,
            MethodTypeDesc type, int geometry) {
        for (int parameter = 0; parameter <= geometry; parameter++) code.aload(parameter);
        code.lload(geometry + 1).lload(geometry + 3).invokestatic(owner, name, type);
    }

    private static void emitContiguousIntBody(CodeBuilder code, CpuKernelSpecialization specialization,
            CpuKernelIr ir) {
        LossKind kind = LossKind.valueOf(field(ir.familyIdentity(), "kind="));
        LossReduction reduction = LossReduction.valueOf(field(ir.familyIdentity(), "reduction="));
        DataType predictionType = DataType.valueOf(field(ir.familyIdentity(), "prediction="));
        DataType targetType = DataType.valueOf(field(ir.familyIdentity(), "target="));
        DataType resultType = DataType.valueOf(field(ir.familyIdentity(), "result="));
        boolean indexIgnorePresent = Boolean.parseBoolean(field(ir.familyIdentity(), "indexIgnore="));
        int[] roles = roles(ir.familyIdentity());
        int geometry = specialization.boundaryDataTypes().size();
        boolean f64 = resultType == DataType.FLOAT64;
        if (kind == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS
                && reduction == LossReduction.NONE
                && exactContiguousIndexNone(specialization, predictionType, targetType, resultType)) {
            emitExactArrayIndexNone(code, specialization, indexIgnorePresent, predictionType,
                    targetType, resultType, f64);
            return;
        }
        if (kind == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS
                && (reduction == LossReduction.SUM || reduction == LossReduction.MEAN)
                && exactArrayIndexReduced(specialization, predictionType, targetType, resultType)) {
            emitExactArrayIndexReduced(code, specialization, indexIgnorePresent, predictionType,
                    targetType, resultType, f64, reduction);
            return;
        }
        var carriers = new CpuCarrierEmitter(code);
        if (kind == LossKind.MEAN_SQUARED_ERROR) {
            int predictionAddress = code.allocateLocal(TypeKind.INT), targetAddress = code.allocateLocal(TypeKind.INT);
            int outputAddress = code.allocateLocal(TypeKind.INT);
            /*
             * The direct NONE oracle initializes its three bases before its range ordinal and
             * keeps one prediction address for the array load.  Preserve that lifetime graph:
             * it avoids a synthetic range-limit local and makes the selected Class-File's
             * source/output address formation match the ordinary Java body for every carrier.
             */
            int ordinal = code.allocateLocal(TypeKind.INT);
            int predictionElementAddress = reduction == LossReduction.NONE
                    ? code.allocateLocal(TypeKind.INT) : -1;
            int limit = reduction == LossReduction.NONE ? -1 : code.allocateLocal(TypeKind.INT);
            int loss = code.allocateLocal(arithmeticKind(f64));
            int reduced = reduction == LossReduction.NONE ? -1 : code.allocateLocal(arithmeticKind(f64));
            int included = reduction == LossReduction.MEAN ? code.allocateLocal(TypeKind.INT) : -1;
            if (reduction == LossReduction.NONE) {
                geometry(code, geometry, 4).l2i().istore(predictionAddress);
                geometry(code, geometry, 5).l2i().istore(targetAddress);
                geometry(code, geometry, 6).l2i().istore(outputAddress);
                code.lload(geometry + 1).l2i().istore(ordinal);
            }
            else { code.loadConstant(0).istore(ordinal); geometry(code, geometry, 9).l2i().istore(limit); }
            emitContiguousMseInt(code, carriers, specialization, predictionType, targetType, resultType,
                    roles[0], roles[1], geometry - 1, ordinal, limit, predictionAddress,
                    targetAddress, outputAddress, predictionElementAddress, loss, reduced, included, f64,
                    reduction);
        } else {
            /*
             * Keep the categorical entry's live-local footprint aligned with its frozen
             * ordinary-Java oracle.  In particular, index loss has no floating target decode or
             * MSE ordinal state.  Reserving those dead locals here causes avoidable register
             * pressure in the class traversal even though the emitted arithmetic is identical.
             */
            int predictionAddress = code.allocateLocal(TypeKind.INT), targetAddress = code.allocateLocal(TypeKind.INT);
            int classes = code.allocateLocal(TypeKind.INT), clazz = code.allocateLocal(TypeKind.INT);
            /* A NONE body owns a per-sample output address.  A reduced body has no such live
             * state: it reuses clazz only after its class loops have ended for the final scalar
             * store.  This is the same lifetime boundary javac derives from the frozen oracle. */
            int outputAddress = reduction == LossReduction.NONE ? code.allocateLocal(TypeKind.INT) : clazz;
            int index = kind == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS
                    ? code.allocateLocal(TypeKind.LONG) : -1;
            // Dense contribution loads form their cold-proved class address directly on the
            // operand stack.  A reusable address local would add a store/reload to both loads
            // in every class and, unlike the frozen Java oracle, keep it live across the
            // zero-weight branch.
            int inputAddress = -1;
            int predictionRepresented = kind == LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS
                    && (predictionType == DataType.BFLOAT16 || predictionType != resultType)
                            ? code.allocateLocal(representedKind(predictionType)) : -1;
            int targetRepresented = kind == LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS
                    ? code.allocateLocal(representedKind(targetType)) : -1;
            int value = code.allocateLocal(arithmeticKind(f64));
            int targetValue = kind == LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS
                    ? code.allocateLocal(arithmeticKind(f64)) : -1;
            int maximum = code.allocateLocal(arithmeticKind(f64)), sum = code.allocateLocal(arithmeticKind(f64));
            int loss = code.allocateLocal(arithmeticKind(f64));
            int reduced = reduction == LossReduction.NONE ? -1 : code.allocateLocal(arithmeticKind(f64));
            // SUM never observes an index included-count. Keeping it in emitted bytecode leaves
            // a dead per-sample increment that an ordinary Java compiler can eliminate.
            int included = reduction == LossReduction.MEAN ? code.allocateLocal(TypeKind.INT) : -1;
            int axis = code.allocateLocal(TypeKind.INT), outer = code.allocateLocal(TypeKind.INT), inner = code.allocateLocal(TypeKind.INT);
            geometry(code, geometry, 1).l2i().istore(axis);
            /*
             * The direct index form owns the class count in this cold prelude.  Its selected
             * LSE uses the index local as scratch-address state. This follows the frozen
             * ordinary-Java oracle's forward
             * coordinate order without making either cold extent a class-shaping fact.
             */
            Label countLoop = code.newLabel(), countDone = code.newLabel(), afterAxis = code.newLabel();
            int rank = code.allocateLocal(TypeKind.INT);
            geometry(code, geometry, 0).l2i().istore(rank);
            extent(code, geometry, axis).l2i().istore(classes);
            code.loadConstant(1).istore(outer); code.loadConstant(1).istore(inner);
            code.loadConstant(0).istore(clazz);
            code.labelBinding(countLoop).iload(clazz).iload(rank)
                    .branch(Opcode.IF_ICMPGE, countDone);
            Label notBeforeAxis = code.newLabel();
            code.iload(clazz).iload(axis).branch(Opcode.IF_ICMPGE, notBeforeAxis);
            // Match ordinary Java's compound-assignment dataflow here: the cold extent remains
            // a long through multiplication and is narrowed only when it is assigned back to
            // the int traversal count.  The contiguous guard makes this semantically identical
            // to an int product while retaining javac's loop-prelude CFG and local lifetime.
            code.iload(outer).i2l(); extent(code, geometry, clazz).lmul().l2i().istore(outer)
                    .branch(Opcode.GOTO, afterAxis);
            code.labelBinding(notBeforeAxis).iload(clazz).iload(axis)
                    .branch(Opcode.IF_ICMPLE, afterAxis);
            code.iload(inner).i2l(); extent(code, geometry, clazz).lmul().l2i().istore(inner);
            code.labelBinding(afterAxis).iinc(clazz, 1).branch(Opcode.GOTO, countLoop);
            code.labelBinding(countDone);
            emitContiguousCategoricalInt(code, carriers, specialization, kind, indexIgnorePresent, predictionType, targetType,
                    resultType, roles[0], roles[1], geometry - 1, reduction, predictionAddress,
                    targetAddress, outputAddress, classes, clazz, inputAddress, index,
                    predictionRepresented, targetRepresented, value, targetValue, maximum, sum, loss,
                    reduced,
                    included, outer, inner, axis, f64);
        }
    }

    /*
     * This is intentionally not expressed through the shared categorical local prelude.  The
     * frozen Java oracle has a narrower lifetime graph: coordinate/predictionBase, maxClazz/sum,
     * and sumClazz/lse share slots.  Keeping this direct array path in source order lets the
     * Class-File retain that graph (25 F32 locals, 29 F64 locals) instead of reserving dead
     * categorical state for the duration of every sample.  The output may be either its direct
     * heap carrier or a segment: output representation does not alter the classification loop.
     */
    private static boolean exactContiguousIndexNone(CpuKernelSpecialization specialization,
            DataType predictionType, DataType targetType, DataType resultType) {
        var carriers = specialization.carrierPattern();
        return predictionType == resultType
                && (predictionType == DataType.FLOAT32 || predictionType == DataType.FLOAT64)
                && carriers.size() == 3
                && (carriers.get(0) == (predictionType == DataType.FLOAT64
                        ? CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY
                        : CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY)
                        || carriers.get(0) == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)
                && (carriers.get(1) == (targetType == DataType.INT64
                        ? CpuKernelSpecialization.CarrierAccess.LONG_ARRAY
                        : CpuKernelSpecialization.CarrierAccess.INT_ARRAY)
                        || carriers.get(1) == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)
                && (carriers.get(2) == (resultType == DataType.FLOAT64
                        ? CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY
                        : CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY)
                        || carriers.get(2) == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT);
    }

    /**
     * Identifies the array-input reduced-index form that needs the frozen oracle's lifetime graph.
     *
     * <p>This is deliberately a carrier/reduction identity fact rather than a geometry fact:
     * rank, axis, extents, bases, ignore value, and range remain cold invocation data.</p>
     */
    private static boolean exactArrayIndexReduced(CpuKernelSpecialization specialization,
            DataType predictionType, DataType targetType, DataType resultType) {
        var carriers = specialization.carrierPattern();
        return predictionType == DataType.FLOAT32 && resultType == DataType.FLOAT32
                && carriers.size() == 3
                && carriers.get(0) == CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY
                && carriers.get(1) == (targetType == DataType.INT64
                        ? CpuKernelSpecialization.CarrierAccess.LONG_ARRAY
                        : CpuKernelSpecialization.CarrierAccess.INT_ARRAY);
    }

    /*
     * Exact array-input reduced INDEX body. Keep initialization and local lifetimes in the same
     * order as {@code CpuLossPerformanceOracle.categorical()}: total, an optional MEAN count,
     * cold geometry, the two advancing bases, samples, and one final output store. In particular
     * no output base is live through a sample; the scalar address is loaded only after the
     * complete ordered reduction. The output carrier is deliberately not a shape distinction:
     * its direct final store uses this same lifetime graph for every typed carrier.
     */
    private static void emitExactArrayIndexReduced(CodeBuilder code,
            CpuKernelSpecialization specialization, boolean indexIgnorePresent,
            DataType predictionType, DataType targetType, DataType resultType, boolean f64,
            LossReduction reduction) {
        var carriers = new CpuCarrierEmitter(code);
        int total = code.allocateLocal(arithmeticKind(f64));
        zero(code, f64, total);
        int count = reduction == LossReduction.MEAN ? code.allocateLocal(TypeKind.INT) : -1;
        if (count >= 0) code.loadConstant(0).istore(count);

        int axis = code.allocateLocal(TypeKind.INT);
        int rank = code.allocateLocal(TypeKind.INT);
        int classes = code.allocateLocal(TypeKind.INT);
        int outer = code.allocateLocal(TypeKind.INT);
        int inner = code.allocateLocal(TypeKind.INT);
        geometry(code, 3, 1).l2i().istore(axis);
        geometry(code, 3, 0).l2i().istore(rank);
        code.aload(3).loadConstant(10).iload(axis).iadd().laload().l2i().istore(classes);
        code.loadConstant(1).istore(outer);
        code.loadConstant(1).istore(inner);
        int coordinate = code.allocateLocal(TypeKind.INT);
        code.loadConstant(0).istore(coordinate);
        Label coordinates = code.newLabel(), coordinatesDone = code.newLabel(), afterCoordinate = code.newLabel();
        code.labelBinding(coordinates).iload(coordinate).iload(rank)
                .branch(Opcode.IF_ICMPGE, coordinatesDone);
        Label notBeforeAxis = code.newLabel();
        code.iload(coordinate).iload(axis).branch(Opcode.IF_ICMPGE, notBeforeAxis);
        code.iload(outer).i2l(); extent(code, 3, coordinate).lmul().l2i().istore(outer)
                .branch(Opcode.GOTO, afterCoordinate);
        code.labelBinding(notBeforeAxis).iload(coordinate).iload(axis)
                .branch(Opcode.IF_ICMPLE, afterCoordinate);
        code.iload(inner).i2l(); extent(code, 3, coordinate).lmul().l2i().istore(inner);
        code.labelBinding(afterCoordinate).iinc(coordinate, 1).branch(Opcode.GOTO, coordinates)
                .labelBinding(coordinatesDone);

        // The coordinate loop is dead after the cold prelude. Reusing that slot for the first
        // advancing base retains javac's oracle lifetime boundary rather than reserving another
        // traversal local through every sample.
        int predictionBase = coordinate;
        int targetBase = code.allocateLocal(TypeKind.INT);
        geometry(code, 3, 4).l2i().istore(predictionBase);
        geometry(code, 3, 5).l2i().istore(targetBase);
        int outerIndex = code.allocateLocal(TypeKind.INT);
        int sample = code.allocateLocal(TypeKind.INT);
        int selected = code.allocateLocal(TypeKind.LONG);
        int base = code.allocateLocal(TypeKind.INT);
        int maximum = code.allocateLocal(arithmeticKind(f64));
        int clazzOrSum = code.allocateLocal(TypeKind.INT);
        int valueOrLse = code.allocateLocal(TypeKind.INT);
        int loss = code.allocateLocal(arithmeticKind(f64));
        code.loadConstant(0).istore(outerIndex);
        Label outerLoop = code.newLabel(), outerDone = code.newLabel();
        code.labelBinding(outerLoop).iload(outerIndex).iload(outer).branch(Opcode.IF_ICMPGE, outerDone);
        code.loadConstant(0).istore(sample);
        Label sampleLoop = code.newLabel(), sampleDone = code.newLabel();
        code.labelBinding(sampleLoop).iload(sample).iload(inner).branch(Opcode.IF_ICMPGE, sampleDone);
        loadIndex(code, carriers, specialization, targetType, 1, targetBase, selected, true);
        if (indexIgnorePresent) {
            Label active = code.newLabel();
            geometry(code, 3, 7).loadConstant(0L).lcmp().branch(Opcode.IFEQ, active);
            code.lload(selected); geometry(code, 3, 8).lcmp().branch(Opcode.IFNE, active);
            code.iinc(predictionBase, 1).iinc(targetBase, 1).iinc(sample, 1)
                    .branch(Opcode.GOTO, sampleLoop);
            code.labelBinding(active);
        }
        code.iload(predictionBase).istore(base);
        code.loadConstant(Float.NEGATIVE_INFINITY).fstore(maximum);
        code.loadConstant(0).istore(clazzOrSum);
        Label maximumLoop = code.newLabel(), maximumDone = code.newLabel();
        code.labelBinding(maximumLoop).iload(clazzOrSum).iload(classes)
                .branch(Opcode.IF_ICMPGE, maximumDone);
        code.aload(0).iload(base).iload(clazzOrSum).iload(inner).imul().iadd().faload()
                .fstore(valueOrLse);
        Label retainMaximum = code.newLabel();
        code.fload(valueOrLse).fload(maximum).fcmpl().branch(Opcode.IFLE, retainMaximum);
        code.fload(valueOrLse).fstore(maximum);
        code.labelBinding(retainMaximum).iinc(clazzOrSum, 1).branch(Opcode.GOTO, maximumLoop)
                .labelBinding(maximumDone);
        code.loadConstant(0.0f).fstore(clazzOrSum);
        code.loadConstant(0).istore(valueOrLse);
        Label sumLoop = code.newLabel(), sumDone = code.newLabel();
        code.labelBinding(sumLoop).iload(valueOrLse).iload(classes).branch(Opcode.IF_ICMPGE, sumDone);
        code.fload(clazzOrSum).aload(0).iload(base).iload(valueOrLse).iload(inner).imul()
                .iadd().faload().fload(maximum).fsub().f2d().invokestatic(STRICT_MATH, "exp",
                        doubleUnary()).d2f().fadd().fstore(clazzOrSum);
        code.iinc(valueOrLse, 1).branch(Opcode.GOTO, sumLoop).labelBinding(sumDone);
        code.fload(maximum).fload(clazzOrSum).f2d().invokestatic(STRICT_MATH, "log", doubleUnary())
                .d2f().fadd().fstore(valueOrLse);
        code.fload(valueOrLse).aload(0).iload(base).lload(selected).l2i().iload(inner).imul()
                .iadd().faload().fsub().fstore(loss);
        add(code, f64, total, loss, total);
        if (count >= 0) code.iinc(count, 1);
        code.iinc(predictionBase, 1).iinc(targetBase, 1).iinc(sample, 1)
                .branch(Opcode.GOTO, sampleLoop).labelBinding(sampleDone);
        code.iload(predictionBase).iload(classes).loadConstant(1).isub().iload(inner).imul()
                .iadd().istore(predictionBase);
        code.iinc(outerIndex, 1).branch(Opcode.GOTO, outerLoop).labelBinding(outerDone);
        if (resultType == DataType.FLOAT32 || resultType == DataType.FLOAT64) {
            carriers.beginFrozenStoreAtStackIntAddress(resultType,
                    specialization.carrierPattern().get(2), 2);
            geometry(code, 3, 6).l2i();
            carriers.endFrozenStoreAtStackIntAddress(resultType,
                    specialization.carrierPattern().get(2));
            if (reduction == LossReduction.MEAN) {
                if (f64) code.dload(total).iload(count).i2d().ddiv();
                else code.fload(total).iload(count).i2f().fdiv();
            } else if (f64) code.dload(total); else code.fload(total);
            carriers.endFrozenStoreAtStackValue(resultType,
                    specialization.carrierPattern().get(2));
        } else {
            if (reduction == LossReduction.MEAN) divideByIntCount(code, f64, total, count);
            int outputAddress = code.allocateLocal(TypeKind.INT);
            geometry(code, 3, 6).l2i().istore(outputAddress);
            store(code, carriers, specialization, resultType, 2, outputAddress, total, true);
        }
    }

    private static void emitExactArrayIndexNone(CodeBuilder code,
            CpuKernelSpecialization specialization, boolean indexIgnorePresent,
            DataType predictionType, DataType targetType, DataType resultType, boolean f64) {
        var carriers = new CpuCarrierEmitter(code);
        // Parameters occupy 0..7: left, right, output, geometry, start, end.
        // Register the manually aliased span with the Class-File builder.  The stores below still
        // establish each verifier type at its real lifetime boundary; this only reserves the
        // highest slot so automatic frame construction retains the complete local domain.
        for (int local = 0; local < (f64 ? 21 : 17); local++) code.allocateLocal(TypeKind.INT);
        final int axis = 8, rank = 9, classes = 10, outer = 11, inner = 12;
        final int coordinateAndPredictionBase = 13, targetBase = 14, outputBase = 15;
        final int outerIndex = 16, sample = 17, selected = 18, base = 20;
        /*
         * The binary64 maximum pass retains javac's ordinary candidate-value local at 24..25.
         * Although sum's class cursor later reuses slot 25, the Class-File builder constructs
         * valid frames because the candidate is established before the maximum-loop backedge and
         * is dead before that cursor starts.  This preserves the clean Java dataflow without a
         * stack-only DUP2/POP2 branch, while retaining the same 29-slot lifetime domain:
         * maximum 21..22 / max clazz 23 / candidate 24..25, then sum 23..24 / sum clazz 25,
         * then lse 25..26 and loss 27..28.  F32 already has the equivalent value-local form.
         */
        final int maximum = 21, maxClazzOrSum = f64 ? 23 : 22;
        final int valueOrSumClazz = f64 ? 25 : 23;
        final int lseOrLoss = f64 ? 25 : 24, loss = f64 ? 27 : 19;

        geometry(code, 3, 1).l2i().istore(axis);
        geometry(code, 3, 0).l2i().istore(rank);
        code.aload(3).loadConstant(10).iload(axis).iadd().laload().l2i().istore(classes);
        code.loadConstant(1).istore(outer); code.loadConstant(1).istore(inner);
        code.loadConstant(0).istore(coordinateAndPredictionBase);
        Label count = code.newLabel(), countDone = code.newLabel(), afterCoordinate = code.newLabel();
        code.labelBinding(count).iload(coordinateAndPredictionBase).iload(rank)
                .branch(Opcode.IF_ICMPGE, countDone);
        Label notBeforeAxis = code.newLabel();
        code.iload(coordinateAndPredictionBase).iload(axis).branch(Opcode.IF_ICMPGE, notBeforeAxis);
        code.iload(outer).i2l().aload(3).loadConstant(10).iload(coordinateAndPredictionBase)
                .iadd().laload().lmul().l2i().istore(outer).branch(Opcode.GOTO, afterCoordinate);
        code.labelBinding(notBeforeAxis).iload(coordinateAndPredictionBase).iload(axis)
                .branch(Opcode.IF_ICMPLE, afterCoordinate);
        code.iload(inner).i2l().aload(3).loadConstant(10).iload(coordinateAndPredictionBase)
                .iadd().laload().lmul().l2i().istore(inner);
        code.labelBinding(afterCoordinate).iinc(coordinateAndPredictionBase, 1)
                .branch(Opcode.GOTO, count).labelBinding(countDone);

        geometry(code, 3, 4).l2i().istore(coordinateAndPredictionBase);
        geometry(code, 3, 5).l2i().istore(targetBase);
        geometry(code, 3, 6).l2i().istore(outputBase);
        code.loadConstant(0).istore(outerIndex);
        Label outerLoop = code.newLabel(), outerDone = code.newLabel();
        code.labelBinding(outerLoop).iload(outerIndex).iload(outer).branch(Opcode.IF_ICMPGE, outerDone);
        code.loadConstant(0).istore(sample);
        Label sampleLoop = code.newLabel(), sampleDone = code.newLabel(), advanceSample = code.newLabel();
        code.labelBinding(sampleLoop).iload(sample).iload(inner).branch(Opcode.IF_ICMPGE, sampleDone);
        carriers.beginFrozenLoadAtStackIntAddress(targetType,
                specialization.carrierPattern().get(1), 1);
        code.iload(targetBase);
        carriers.endFrozenLoadAtStackIntAddress(targetType,
                specialization.carrierPattern().get(1));
        if (targetType != DataType.INT64) code.i2l();
        code.lstore(selected);
        if (indexIgnorePresent) {
            Label active = code.newLabel();
            geometry(code, 3, 7).loadConstant(0L).lcmp().branch(Opcode.IFEQ, active);
            code.lload(selected); geometry(code, 3, 8).lcmp().branch(Opcode.IFNE, active);
            if (specialization.carrierPattern().get(2)
                    == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT) {
                zero(code, f64, loss);
                store(code, carriers, specialization, resultType, 2, outputBase, loss, true);
            } else {
                code.aload(2).iload(outputBase);
                if (f64) code.loadConstant(0.0d).dastore(); else code.loadConstant(0.0f).fastore();
            }
            code.iinc(coordinateAndPredictionBase, 1).iinc(targetBase, 1).iinc(outputBase, 1)
                    .branch(Opcode.GOTO, advanceSample);
            code.labelBinding(active);
        }
        code.iload(coordinateAndPredictionBase).istore(base);
        if (f64) code.loadConstant(Double.NEGATIVE_INFINITY).dstore(maximum);
        else code.loadConstant(Float.NEGATIVE_INFINITY).fstore(maximum);
        code.loadConstant(0).istore(maxClazzOrSum);
        Label maxLoop = code.newLabel(), maxDone = code.newLabel(), keep = code.newLabel();
        code.labelBinding(maxLoop).iload(maxClazzOrSum).iload(classes).branch(Opcode.IF_ICMPGE, maxDone);
        carriers.beginFrozenLoadAtStackIntAddress(predictionType,
                specialization.carrierPattern().get(0), 0);
        code.iload(base).iload(maxClazzOrSum).iload(inner).imul().iadd();
        if (f64) {
            Label maximumAdvanced = code.newLabel();
            carriers.endFrozenLoadAtStackIntAddress(predictionType,
                    specialization.carrierPattern().get(0));
            code.dstore(24).dload(24).dload(maximum).dcmpg().branch(Opcode.IFLE, keep);
            code.dload(24).dstore(maximum).branch(Opcode.GOTO, maximumAdvanced);
            code.labelBinding(keep);
            code.labelBinding(maximumAdvanced);
        } else {
            carriers.endFrozenLoadAtStackIntAddress(predictionType,
                    specialization.carrierPattern().get(0));
            code.fstore(valueOrSumClazz);
            code.fload(valueOrSumClazz).fload(maximum).fcmpg().branch(Opcode.IFLE, keep);
            code.fload(valueOrSumClazz).fstore(maximum);
            code.labelBinding(keep);
        }
        code.iinc(maxClazzOrSum, 1).branch(Opcode.GOTO, maxLoop).labelBinding(maxDone);

        if (f64) code.loadConstant(0.0d).dstore(maxClazzOrSum); else code.loadConstant(0.0f).fstore(maxClazzOrSum);
        code.loadConstant(0).istore(valueOrSumClazz);
        Label sumLoop = code.newLabel(), sumDone = code.newLabel();
        code.labelBinding(sumLoop).iload(valueOrSumClazz).iload(classes).branch(Opcode.IF_ICMPGE, sumDone);
        if (f64) code.dload(maxClazzOrSum); else code.fload(maxClazzOrSum);
        carriers.beginFrozenLoadAtStackIntAddress(predictionType,
                specialization.carrierPattern().get(0), 0);
        code.iload(base).iload(valueOrSumClazz).iload(inner).imul().iadd();
        carriers.endFrozenLoadAtStackIntAddress(predictionType,
                specialization.carrierPattern().get(0));
        if (f64) code.dload(maximum).dsub().invokestatic(STRICT_MATH, "exp", doubleUnary()).dadd().dstore(maxClazzOrSum);
        else code.fload(maximum).fsub().f2d().invokestatic(STRICT_MATH, "exp", doubleUnary()).d2f().fadd().fstore(maxClazzOrSum);
        code.iinc(valueOrSumClazz, 1).branch(Opcode.GOTO, sumLoop).labelBinding(sumDone);
        if (f64) code.dload(maximum).dload(maxClazzOrSum).invokestatic(STRICT_MATH, "log", doubleUnary()).dadd().dstore(lseOrLoss);
        else code.fload(maximum).fload(maxClazzOrSum).f2d().invokestatic(STRICT_MATH, "log", doubleUnary()).d2f().fadd().fstore(lseOrLoss);
        if (f64) code.dload(lseOrLoss); else code.fload(lseOrLoss);
        carriers.beginFrozenLoadAtStackIntAddress(predictionType,
                specialization.carrierPattern().get(0), 0);
        code.iload(base).lload(selected).l2i().iload(inner).imul().iadd();
        carriers.endFrozenLoadAtStackIntAddress(predictionType,
                specialization.carrierPattern().get(0));
        if (f64) code.dsub().dstore(loss); else code.fsub().fstore(loss);
        storeExactArrayIndexNoneOutput(code, carriers, specialization, resultType, outputBase,
                loss, f64);
        code.iinc(coordinateAndPredictionBase, 1).iinc(targetBase, 1).iinc(outputBase, 1);
        code.labelBinding(advanceSample).iinc(sample, 1).branch(Opcode.GOTO, sampleLoop)
                .labelBinding(sampleDone);
        code.iload(coordinateAndPredictionBase).iload(classes).loadConstant(1).isub().iload(inner)
                .imul().iadd().istore(coordinateAndPredictionBase);
        code.iinc(outerIndex, 1).branch(Opcode.GOTO, outerLoop).labelBinding(outerDone);
    }

    /** Stores the exact index-NONE result without perturbing its heap-array oracle body. */
    private static void storeExactArrayIndexNoneOutput(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType resultType, int outputAddress,
            int result, boolean f64) {
        if (specialization.carrierPattern().get(2)
                == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT) {
            store(code, carriers, specialization, resultType, 2, outputAddress, result, true);
            return;
        }
        code.aload(2).iload(outputAddress);
        if (f64) code.dload(result).dastore(); else code.fload(result).fastore();
    }

    private static void emitGenericAffine(CodeBuilder code, CpuKernelSpecialization specialization, CpuKernelIr ir) {
        LossKind kind = LossKind.valueOf(field(ir.familyIdentity(), "kind="));
        LossReduction reduction = LossReduction.valueOf(field(ir.familyIdentity(), "reduction="));
        DataType predictionType = DataType.valueOf(field(ir.familyIdentity(), "prediction="));
        DataType targetType = DataType.valueOf(field(ir.familyIdentity(), "target="));
        DataType resultType = DataType.valueOf(field(ir.familyIdentity(), "result="));
        boolean indexIgnorePresent = Boolean.parseBoolean(field(ir.familyIdentity(), "indexIgnore="));
        int[] roles = roles(ir.familyIdentity());
        int boundaryCount = specialization.boundaryDataTypes().size();
        if (specialization.classIdentitySchema() != 58
                || specialization.executionStrategy().compute() != Compute.SCALAR
                || specialization.scratchParameter()
                || boundaryCount < 2 || boundaryCount > 3
                || roles.length != 2 || roles[0] < 0 || roles[1] < 0
                || roles[0] >= boundaryCount - 1 || roles[1] >= boundaryCount - 1
                || specialization.boundaryDataTypes().get(roles[0]) != predictionType
                || specialization.boundaryDataTypes().get(roles[1]) != targetType
                || specialization.boundaryDataTypes().getLast() != resultType
                || !floating(predictionType) || !floating(resultType)
                || kind == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS
                        != integral(targetType)) {
            throw new IllegalArgumentException("loss generated facts disagree");
        }

        boolean f64 = resultType == DataType.FLOAT64;
        int geometry = boundaryCount;
        int start = geometry + 1;
        int end = geometry + 3;
        int predictionBoundary = roles[0];
        int targetBoundary = roles[1];
        int outputBoundary = boundaryCount - 1;
        var carriers = new CpuCarrierEmitter(code);

        int rank = code.allocateLocal(TypeKind.INT);
        int axis = code.allocateLocal(TypeKind.INT);
        int targetRank = code.allocateLocal(TypeKind.INT);
        int outputRank = code.allocateLocal(TypeKind.INT);
        int ordinal = code.allocateLocal(TypeKind.LONG);
        int limit = code.allocateLocal(TypeKind.LONG);
        int remainder = code.allocateLocal(TypeKind.LONG);
        int coordinate = code.allocateLocal(TypeKind.LONG);
        int logicalAxis = code.allocateLocal(TypeKind.INT);
        int targetAxis = code.allocateLocal(TypeKind.INT);
        int predictionAddress = code.allocateLocal(TypeKind.LONG);
        int targetAddress = code.allocateLocal(TypeKind.LONG);
        int outputAddress = code.allocateLocal(TypeKind.LONG);
        int classStride = code.allocateLocal(TypeKind.LONG);
        int targetClassStride = code.allocateLocal(TypeKind.LONG);
        int classes = code.allocateLocal(TypeKind.LONG);
        int clazz = code.allocateLocal(TypeKind.LONG);
        int inputAddress = code.allocateLocal(TypeKind.LONG);
        int index = code.allocateLocal(TypeKind.LONG);
        int predictionRepresented = code.allocateLocal(representedKind(predictionType));
        int targetRepresented = code.allocateLocal(representedKind(targetType));
        int value = code.allocateLocal(arithmeticKind(f64));
        int targetValue = code.allocateLocal(arithmeticKind(f64));
        int maximum = code.allocateLocal(arithmeticKind(f64));
        int sum = code.allocateLocal(arithmeticKind(f64));
        int loss = code.allocateLocal(arithmeticKind(f64));
        int reduced = code.allocateLocal(arithmeticKind(f64));
        int temporary = code.allocateLocal(arithmeticKind(f64));
        int included = reduction == LossReduction.MEAN ? code.allocateLocal(TypeKind.LONG) : -1;

        geometry(code, geometry, 0).l2i().istore(rank);
        geometry(code, geometry, 1).l2i().istore(axis);
        geometry(code, geometry, 2).l2i().istore(targetRank);
        geometry(code, geometry, 3).l2i().istore(outputRank);
        zero(code, f64, reduced);
        if (included >= 0) code.loadConstant(0L).lstore(included);
        if (reduction == LossReduction.NONE) {
            code.lload(start).lstore(ordinal);
            code.lload(end).lstore(limit);
        } else {
            code.loadConstant(0L).lstore(ordinal);
            geometry(code, geometry, 9).lstore(limit);
        }

        Label loop = code.newLabel();
        Label done = code.newLabel();
        code.labelBinding(loop).lload(ordinal).lload(limit).lcmp().branch(Opcode.IFGE, done);
        addresses(code, geometry, kind, reduction, rank, axis, targetRank, outputRank, ordinal,
                remainder, coordinate, logicalAxis, targetAxis, predictionAddress, targetAddress,
                outputAddress);
        Label next = code.newLabel();
        if (kind == LossKind.MEAN_SQUARED_ERROR) {
            loadFloating(code, carriers, specialization, predictionType, predictionBoundary,
                    predictionAddress, predictionRepresented, value, f64);
            loadFloating(code, carriers, specialization, targetType, targetBoundary, targetAddress,
                    targetRepresented, targetValue, f64);
            subtract(code, f64, value, targetValue, loss);
            multiply(code, f64, loss, loss, loss);
        } else {
            extent(code, geometry, axis).lstore(classes);
            predictionStride(code, geometry, rank, axis).lstore(classStride);
            if (kind == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS) {
                loadIndex(code, carriers, specialization, targetType, targetBoundary, targetAddress,
                        index);
                if (indexIgnorePresent) {
                Label notIgnored = code.newLabel();
                geometry(code, geometry, 7).loadConstant(0L).lcmp().branch(Opcode.IFEQ,
                        notIgnored);
                code.lload(index); geometry(code, geometry, 8).lcmp().branch(Opcode.IFNE,
                        notIgnored);
                zero(code, f64, loss);
                if (reduction == LossReduction.NONE) {
                    store(code, carriers, specialization, resultType, outputBoundary, outputAddress,
                            loss);
                }
                code.branch(Opcode.GOTO, next);
                code.labelBinding(notIgnored);
                }
            }
            emitLse(code, carriers, specialization, predictionType, predictionBoundary,
                    predictionAddress, classStride, classes, clazz, inputAddress,
                    predictionRepresented, value, maximum, sum, temporary, f64);
            if (kind == LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS) {
                zero(code, f64, loss);
                targetStride(code, geometry, rank, targetRank, axis).lstore(targetClassStride);
                code.loadConstant(0L).lstore(clazz);
                Label weights = code.newLabel();
                Label weightsDone = code.newLabel();
                code.labelBinding(weights).lload(clazz).lload(classes).lcmp()
                        .branch(Opcode.IFGE, weightsDone);
                code.lload(targetAddress).lload(clazz).lload(targetClassStride).lmul().ladd()
                        .lstore(inputAddress);
                loadFloating(code, carriers, specialization, targetType, targetBoundary, inputAddress,
                        targetRepresented, targetValue, f64);
                Label zeroWeight = code.newLabel();
                compareZero(code, f64, targetValue).branch(Opcode.IFEQ, zeroWeight);
                code.lload(predictionAddress).lload(clazz).lload(classStride).lmul().ladd()
                        .lstore(inputAddress);
                loadFloating(code, carriers, specialization, predictionType, predictionBoundary,
                        inputAddress, predictionRepresented, value, f64);
                subtract(code, f64, sum, value, temporary);
                multiply(code, f64, targetValue, temporary, temporary);
                add(code, f64, loss, temporary, loss);
                code.labelBinding(zeroWeight).lload(clazz).loadConstant(1L).ladd().lstore(clazz)
                        .branch(Opcode.GOTO, weights).labelBinding(weightsDone);
            } else {
                code.lload(predictionAddress).lload(index).lload(classStride).lmul().ladd()
                        .lstore(inputAddress);
                loadFloating(code, carriers, specialization, predictionType, predictionBoundary,
                        inputAddress, predictionRepresented, value, f64);
                subtract(code, f64, sum, value, loss);
            }
        }

        if (reduction == LossReduction.NONE) {
            store(code, carriers, specialization, resultType, outputBoundary, outputAddress, loss);
        } else {
            add(code, f64, reduced, loss, reduced);
            if (kind == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS && included >= 0) {
                code.lload(included).loadConstant(1L).ladd().lstore(included);
            }
        }
        code.labelBinding(next).lload(ordinal).loadConstant(1L).ladd().lstore(ordinal)
                .branch(Opcode.GOTO, loop).labelBinding(done);

        if (reduction != LossReduction.NONE) {
            if (reduction == LossReduction.MEAN) {
                if (kind == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS) {
                    divideByCount(code, f64, reduced, included);
                } else {
                    geometry(code, geometry, 9).lstore(included);
                    divideByCount(code, f64, reduced, included);
                }
            }
            geometry(code, geometry, 6).lstore(outputAddress);
            store(code, carriers, specialization, resultType, outputBoundary, outputAddress, reduced);
        }
    }

    /** Emits a cold proof that all three MSE layouts are dense row-major views. */
    private static void emitContiguousMseGuard(CodeBuilder code, int geometry, int rank,
            int targetRank, int outputRank, Label generic) {
        int axis = code.allocateLocal(TypeKind.INT);
        int expected = code.allocateLocal(TypeKind.LONG);
        code.iload(rank).iload(targetRank).branch(Opcode.IF_ICMPNE, generic);
        // A NONE result has the input rank; SUM and MEAN have the required scalar result.
        Label outputRankValid = code.newLabel();
        code.iload(rank).iload(outputRank).branch(Opcode.IF_ICMPEQ, outputRankValid);
        code.iload(outputRank).branch(Opcode.IFNE, generic);
        code.labelBinding(outputRankValid);
        code.loadConstant(1L).lstore(expected);
        code.iload(rank).loadConstant(1).isub().istore(axis);
        Label loop = code.newLabel();
        Label done = code.newLabel();
        code.labelBinding(loop).iload(axis).branch(Opcode.IFLT, done);
        // prediction strides start at 10 + rank; target and output follow their rank-sized blocks.
        code.aload(geometry).loadConstant(10).iload(rank).iadd().iload(axis).iadd().laload()
                .lload(expected).lcmp().branch(Opcode.IFNE, generic);
        code.aload(geometry).loadConstant(10).iload(rank).iadd().iload(rank).iadd()
                .iload(axis).iadd().laload().lload(expected).lcmp().branch(Opcode.IFNE, generic);
        // Reduced MSE has a scalar output and therefore no output-stride block to inspect.
        Label scalarOutputStride = code.newLabel();
        code.iload(outputRank).branch(Opcode.IFEQ, scalarOutputStride);
        code.aload(geometry).loadConstant(10).iload(rank).iadd().iload(rank).iadd()
                .iload(targetRank).iadd().iload(axis).iadd().laload().lload(expected).lcmp()
                .branch(Opcode.IFNE, generic);
        code.labelBinding(scalarOutputStride);
        code.lload(expected); extent(code, geometry, axis).lmul().lstore(expected);
        code.iinc(axis, -1).branch(Opcode.GOTO, loop);
        code.labelBinding(done);
        emitBasePlusCountIntGuard(code, geometry, 4, expected, generic);
        emitBasePlusCountIntGuard(code, geometry, 5, expected, generic);
        Label scalarOutput = code.newLabel();
        code.iload(outputRank).branch(Opcode.IFEQ, scalarOutput);
        emitBasePlusCountIntGuard(code, geometry, 6, expected, generic);
        code.labelBinding(scalarOutput);
    }

    /** Rejects a fast-path invocation unless every cold geometry/address fact fits Java int. */
    private static void emitIntGeometryGuard(CodeBuilder code, int geometry, Label generic) {
        int slot = code.allocateLocal(TypeKind.INT);
        int value = code.allocateLocal(TypeKind.LONG);
        code.lload(geometry + 1).loadConstant(0L).lcmp().branch(Opcode.IFLT, generic);
        code.lload(geometry + 3).loadConstant(0L).lcmp().branch(Opcode.IFLT, generic);
        code.lload(geometry + 1).loadConstant((long) Integer.MAX_VALUE).lcmp().branch(Opcode.IFGT, generic);
        code.lload(geometry + 3).loadConstant((long) Integer.MAX_VALUE).lcmp().branch(Opcode.IFGT, generic);
        code.lload(geometry + 1).lload(geometry + 3).lcmp().branch(Opcode.IFGT, generic);
        // Axis -1 denotes MSE and the optional index-ignore value is an exact signed long.
        // They are control data, not int-address facts.  Start at target rank and skip slots 7/8.
        code.loadConstant(2).istore(slot);
        Label loop = code.newLabel(), done = code.newLabel();
        code.labelBinding(loop).iload(slot).aload(geometry).arraylength().branch(Opcode.IF_ICMPGE, done);
        Label skip = code.newLabel();
        code.iload(slot).loadConstant(7).branch(Opcode.IF_ICMPEQ, skip);
        code.iload(slot).loadConstant(8).branch(Opcode.IF_ICMPEQ, skip);
        code.aload(geometry).iload(slot).laload().lstore(value);
        code.lload(value).loadConstant(0L).lcmp().branch(Opcode.IFLT, generic);
        code.lload(value).loadConstant((long) Integer.MAX_VALUE).lcmp().branch(Opcode.IFGT, generic);
        code.labelBinding(skip).iinc(slot, 1).branch(Opcode.GOTO, loop);
        code.labelBinding(done);
    }

    /** Rejects an int loop when a contiguous role's base plus its complete element count overflows. */
    private static void emitBasePlusCountIntGuard(CodeBuilder code, int geometry, int base,
            int count, Label generic) {
        geometry(code, geometry, base).lload(count).ladd().loadConstant((long) Integer.MAX_VALUE)
                .lcmp().branch(Opcode.IFGT, generic);
    }

    /** Emits the proved int-index MSE loop; segment calls widen only at byte-offset formation. */
    private static void emitContiguousMseInt(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType predictionType, DataType targetType,
            DataType resultType, int predictionBoundary, int targetBoundary, int outputBoundary,
            int ordinal, int limit, int predictionAddress, int targetAddress, int outputAddress,
            int predictionElementAddress, int loss, int reduced, int included, boolean f64,
            LossReduction reduction) {
        int geometry = specialization.boundaryDataTypes().size();
        if (reduction != LossReduction.NONE) {
            geometry(code, geometry, 4).l2i().istore(predictionAddress);
            geometry(code, geometry, 5).l2i().istore(targetAddress);
        }
        if (reduction != LossReduction.NONE) zero(code, f64, reduced);
        Label loop = code.newLabel(), done = code.newLabel();
        code.labelBinding(loop).iload(ordinal);
        if (reduction == LossReduction.NONE) code.lload(geometry + 3).l2i(); else code.iload(limit);
        code.branch(Opcode.IF_ICMPGE, done);
        if (reduction == LossReduction.NONE)
            code.iload(predictionAddress).iload(ordinal).iadd().istore(predictionElementAddress);
        carriers.beginFrozenLoadAtStackIntAddress(predictionType,
                specialization.carrierPattern().get(predictionBoundary), predictionBoundary);
        if (reduction == LossReduction.NONE) code.iload(predictionElementAddress);
        else code.iload(predictionAddress).iload(ordinal).iadd();
        finishFloatingLoadAtStackIntAddress(code, carriers, specialization, predictionType,
                predictionBoundary, f64);
        carriers.beginFrozenLoadAtStackIntAddress(targetType,
                specialization.carrierPattern().get(targetBoundary), targetBoundary);
        code.iload(targetAddress).iload(ordinal).iadd();
        finishFloatingLoadAtStackIntAddress(code, carriers, specialization, targetType,
                targetBoundary, f64);
        if (f64) code.dsub().dstore(loss); else code.fsub().fstore(loss);
        if (reduction == LossReduction.NONE) {
            carriers.beginFrozenStoreAtStackIntAddress(resultType,
                    specialization.carrierPattern().get(outputBoundary), outputBoundary);
            code.iload(outputAddress).iload(ordinal).iadd();
            carriers.endFrozenStoreAtStackIntAddress(resultType,
                    specialization.carrierPattern().get(outputBoundary));
            if (resultType == DataType.BFLOAT16) {
                multiply(code, f64, loss, loss, loss);
                emitBfloat16StoreValue(code, loss);
            } else if (f64) code.dload(loss).dload(loss).dmul();
            else code.fload(loss).fload(loss).fmul();
            carriers.endFrozenStoreAtStackValue(resultType,
                    specialization.carrierPattern().get(outputBoundary));
        } else {
            multiply(code, f64, loss, loss, loss);
            add(code, f64, reduced, loss, reduced);
        }
        code.iinc(ordinal, 1)
                .branch(Opcode.GOTO, loop).labelBinding(done);
        if (reduction != LossReduction.NONE) {
            if (reduction == LossReduction.MEAN) {
                geometry(code, geometry, 9).l2i().istore(included); divideByIntCount(code, f64, reduced, included);
            }
            geometry(code, geometry, 6).l2i().istore(outputAddress);
            store(code, carriers, specialization, resultType, outputBoundary, outputAddress, reduced, true);
        }
    }

    /** Emits the allocation-free linear MSE loop selected by the cold contiguous proof. */
    private static void emitContiguousMse(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType predictionType, DataType targetType,
            DataType resultType, int predictionBoundary, int targetBoundary, int outputBoundary,
            int ordinal, int limit, int predictionAddress, int targetAddress, int outputAddress,
            int predictionRepresented, int targetRepresented, int value, int targetValue, int loss,
            int reduced, int included, boolean f64, LossReduction reduction) {
        geometry(code, specialization.boundaryDataTypes().size(), 4).lstore(predictionAddress);
        geometry(code, specialization.boundaryDataTypes().size(), 5).lstore(targetAddress);
        geometry(code, specialization.boundaryDataTypes().size(), 6).lstore(outputAddress);
        if (reduction == LossReduction.NONE) {
            code.lload(predictionAddress).lload(ordinal).ladd().lstore(predictionAddress);
            code.lload(targetAddress).lload(ordinal).ladd().lstore(targetAddress);
            code.lload(outputAddress).lload(ordinal).ladd().lstore(outputAddress);
        }
        zero(code, f64, reduced);
        Label loop = code.newLabel(); Label done = code.newLabel();
        code.labelBinding(loop).lload(ordinal).lload(limit).lcmp().branch(Opcode.IFGE, done);
        loadFloating(code, carriers, specialization, predictionType, predictionBoundary,
                predictionAddress, predictionRepresented, value, f64);
        loadFloating(code, carriers, specialization, targetType, targetBoundary, targetAddress,
                targetRepresented, targetValue, f64);
        subtract(code, f64, value, targetValue, loss);
        multiply(code, f64, loss, loss, loss);
        if (reduction == LossReduction.NONE) {
            store(code, carriers, specialization, resultType, outputBoundary, outputAddress, loss);
            code.lload(outputAddress).loadConstant(1L).ladd().lstore(outputAddress);
        } else add(code, f64, reduced, loss, reduced);
        code.lload(predictionAddress).loadConstant(1L).ladd().lstore(predictionAddress);
        code.lload(targetAddress).loadConstant(1L).ladd().lstore(targetAddress);
        code.lload(ordinal).loadConstant(1L).ladd().lstore(ordinal).branch(Opcode.GOTO, loop);
        code.labelBinding(done);
        if (reduction != LossReduction.NONE) {
            if (reduction == LossReduction.MEAN) {
                geometry(code, specialization.boundaryDataTypes().size(), 9).lstore(included);
                divideByCount(code, f64, reduced, included);
            }
            geometry(code, specialization.boundaryDataTypes().size(), 6).lstore(outputAddress);
            store(code, carriers, specialization, resultType, outputBoundary, outputAddress, reduced);
        }
    }

    /**
     * Proves the categorical row form cold.  The two resulting counters deliberately avoid
     * converting a sample ordinal back to coordinates: every accepted class row is reached by
     * direct base advancement, including an axis that is not the final logits axis.
     */
    private static void emitContiguousCategoricalGuard(CodeBuilder code, int geometry, int rank,
            int axis, int targetRank, int outputRank, int ordinal, int limit,
            LossReduction reduction, Label generic, int outer, int inner) {
        int coordinate = code.allocateLocal(TypeKind.INT);
        int targetCoordinate = code.allocateLocal(TypeKind.INT);
        int expectedLogits = code.allocateLocal(TypeKind.LONG);
        int expectedSample = code.allocateLocal(TypeKind.LONG);
        int expectedTarget = code.allocateLocal(TypeKind.LONG);
        // Dense targets retain the logits rank; index targets omit only the class coordinate.
        Label targetRankValid = code.newLabel();
        code.iload(targetRank).iload(rank).branch(Opcode.IF_ICMPEQ, targetRankValid);
        code.iload(rank).loadConstant(1).isub().iload(targetRank).branch(Opcode.IF_ICMPNE, generic);
        code.labelBinding(targetRankValid);
        if (reduction == LossReduction.NONE) {
            code.iload(rank).loadConstant(1).isub().iload(outputRank)
                    .branch(Opcode.IF_ICMPNE, generic);
            code.lload(ordinal).loadConstant(0L).lcmp().branch(Opcode.IFNE, generic);
            code.lload(limit); geometry(code, geometry, 9).lcmp().branch(Opcode.IFNE, generic);
        }
        code.loadConstant(1L).lstore(expectedLogits);
        code.loadConstant(1L).lstore(expectedSample);
        code.loadConstant(1L).lstore(outer);
        code.loadConstant(1L).lstore(inner);
        code.iload(rank).loadConstant(1).isub().istore(coordinate);
        Label loop = code.newLabel();
        Label done = code.newLabel();
        code.labelBinding(loop).iload(coordinate).branch(Opcode.IFLT, done);
        predictionStride(code, geometry, rank, coordinate).lload(expectedLogits).lcmp()
                .branch(Opcode.IFNE, generic);
        Label classAxis = code.newLabel();
        code.iload(coordinate).iload(axis).branch(Opcode.IF_ICMPEQ, classAxis);
        Label denseTarget = code.newLabel();
        Label targetMapped = code.newLabel();
        code.iload(targetRank).iload(rank).branch(Opcode.IF_ICMPEQ, denseTarget);
        Label beforeClass = code.newLabel();
        code.iload(coordinate).iload(axis).branch(Opcode.IF_ICMPLT, beforeClass);
        code.iload(coordinate).loadConstant(1).isub().istore(targetCoordinate)
                .branch(Opcode.GOTO, targetMapped);
        code.labelBinding(beforeClass).iload(coordinate).istore(targetCoordinate)
                .branch(Opcode.GOTO, targetMapped);
        code.labelBinding(denseTarget).iload(coordinate).istore(targetCoordinate);
        code.labelBinding(targetMapped);
        Label targetExpectedMapped = code.newLabel();
        Label targetIsDense = code.newLabel();
        code.iload(targetRank).iload(rank).branch(Opcode.IF_ICMPEQ, targetIsDense);
        code.lload(expectedSample).lstore(expectedTarget).branch(Opcode.GOTO, targetExpectedMapped);
        code.labelBinding(targetIsDense).lload(expectedLogits).lstore(expectedTarget);
        code.labelBinding(targetExpectedMapped);
        targetStride(code, geometry, rank, targetRank, targetCoordinate).lload(expectedTarget).lcmp()
                .branch(Opcode.IFNE, generic);
        if (reduction == LossReduction.NONE) {
            int outputCoordinate = code.allocateLocal(TypeKind.INT);
            Label before = code.newLabel();
            Label outputMapped = code.newLabel();
            code.iload(coordinate).iload(axis).branch(Opcode.IF_ICMPLT, before);
            code.iload(coordinate).loadConstant(1).isub().istore(outputCoordinate)
                    .branch(Opcode.GOTO, outputMapped);
            code.labelBinding(before).iload(coordinate).istore(outputCoordinate);
            code.labelBinding(outputMapped);
            outputStride(code, geometry, rank, targetRank, outputRank, outputCoordinate)
                    .lload(expectedSample).lcmp().branch(Opcode.IFNE, generic);
        }
        Label afterCounts = code.newLabel();
        code.labelBinding(classAxis);
        Label innerCoordinate = code.newLabel();
        code.iload(coordinate).iload(axis).branch(Opcode.IF_ICMPGT, innerCoordinate);
        code.iload(coordinate).iload(axis).branch(Opcode.IF_ICMPEQ, afterCounts);
        code.lload(outer); extent(code, geometry, coordinate).lmul().lstore(outer)
                .branch(Opcode.GOTO, afterCounts);
        code.labelBinding(innerCoordinate).lload(inner); extent(code, geometry, coordinate).lmul()
                .lstore(inner).labelBinding(afterCounts);
        code.lload(expectedLogits); extent(code, geometry, coordinate).lmul().lstore(expectedLogits);
        Label skipSample = code.newLabel();
        code.iload(coordinate).iload(axis).branch(Opcode.IF_ICMPEQ, skipSample);
        code.lload(expectedSample); extent(code, geometry, coordinate).lmul().lstore(expectedSample);
        code.labelBinding(skipSample).iinc(coordinate, -1).branch(Opcode.GOTO, loop)
                .labelBinding(done);
        emitBasePlusCountIntGuard(code, geometry, 4, expectedLogits, generic);
        Label indexTarget = code.newLabel();
        code.iload(targetRank).iload(rank).branch(Opcode.IF_ICMPNE, indexTarget);
        emitBasePlusCountIntGuard(code, geometry, 5, expectedLogits, generic);
        Label targetChecked = code.newLabel();
        code.branch(Opcode.GOTO, targetChecked);
        code.labelBinding(indexTarget);
        emitBasePlusCountIntGuard(code, geometry, 5, expectedSample, generic);
        code.labelBinding(targetChecked);
        Label scalarOutput = code.newLabel();
        code.iload(outputRank).branch(Opcode.IFEQ, scalarOutput);
        emitBasePlusCountIntGuard(code, geometry, 6, expectedSample, generic);
        code.labelBinding(scalarOutput);
    }

    /**
     * Emits the proved dense categorical row loops with no sample-coordinate reconstruction.
     *
     * <p>The {@code axis} argument is the allocated local holding the cold normalized class-axis
     * value. It must obtain the class extent: a carrier parameter ordinal is not an integer
     * local, even when it has the same numeric value as a geometry slot.</p>
     */
    private static void emitContiguousCategorical(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, LossKind kind, boolean indexIgnorePresent, DataType predictionType,
            DataType targetType, DataType resultType, int predictionBoundary, int targetBoundary,
            int outputBoundary, LossReduction reduction, int predictionAddress, int targetAddress,
            int outputAddress, int classStride, int classes, int clazz, int inputAddress, int index,
            int predictionRepresented, int targetRepresented, int value, int targetValue,
            int maximum, int sum, int loss, int reduced, int temporary, int included, int outer,
            int inner, int axis, boolean f64) {
        int outerIndex = code.allocateLocal(TypeKind.LONG);
        int sample = code.allocateLocal(TypeKind.LONG);
        geometry(code, specialization.boundaryDataTypes().size(), 4).lstore(predictionAddress);
        geometry(code, specialization.boundaryDataTypes().size(), 5).lstore(targetAddress);
        geometry(code, specialization.boundaryDataTypes().size(), 6).lstore(outputAddress);
        extent(code, specialization.boundaryDataTypes().size(), axis).lstore(classes);
        code.lload(inner).lstore(classStride);
        zero(code, f64, reduced);
        if (included >= 0) code.loadConstant(0L).lstore(included);
        code.loadConstant(0L).lstore(outerIndex);
        Label outerLoop = code.newLabel();
        Label outerDone = code.newLabel();
        code.labelBinding(outerLoop).lload(outerIndex).lload(outer).lcmp()
                .branch(Opcode.IFGE, outerDone);
        code.loadConstant(0L).lstore(sample);
        Label sampleLoop = code.newLabel();
        Label sampleDone = code.newLabel();
        code.labelBinding(sampleLoop).lload(sample).lload(inner).lcmp()
                .branch(Opcode.IFGE, sampleDone);
        Label next = code.newLabel();
        if (kind == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS) {
            loadIndex(code, carriers, specialization, targetType, targetBoundary, targetAddress, index);
            if (indexIgnorePresent) {
            Label notIgnored = code.newLabel();
            geometry(code, specialization.boundaryDataTypes().size(), 7).loadConstant(0L).lcmp()
                    .branch(Opcode.IFEQ, notIgnored);
            code.lload(index); geometry(code, specialization.boundaryDataTypes().size(), 8).lcmp()
                    .branch(Opcode.IFNE, notIgnored);
            zero(code, f64, loss);
            if (reduction == LossReduction.NONE) {
                store(code, carriers, specialization, resultType, outputBoundary, outputAddress, loss);
            }
            code.branch(Opcode.GOTO, next);
            code.labelBinding(notIgnored);
            }
        }
        emitLse(code, carriers, specialization, predictionType, predictionBoundary,
                predictionAddress, classStride, classes, clazz, inputAddress, predictionRepresented,
                value, maximum, sum, temporary, f64);
        if (kind == LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS) {
            zero(code, f64, loss);
            code.loadConstant(0L).lstore(clazz);
            Label weights = code.newLabel();
            Label weightsDone = code.newLabel();
            code.labelBinding(weights).lload(clazz).lload(classes).lcmp()
                    .branch(Opcode.IFGE, weightsDone);
            code.lload(targetAddress).lload(clazz).lload(classStride).lmul().ladd()
                    .lstore(inputAddress);
            loadFloating(code, carriers, specialization, targetType, targetBoundary, inputAddress,
                    targetRepresented, targetValue, f64);
            Label zeroWeight = code.newLabel();
            compareZero(code, f64, targetValue).branch(Opcode.IFEQ, zeroWeight);
            code.lload(predictionAddress).lload(clazz).lload(classStride).lmul().ladd()
                    .lstore(inputAddress);
            loadFloating(code, carriers, specialization, predictionType, predictionBoundary,
                    inputAddress, predictionRepresented, value, f64);
            subtract(code, f64, sum, value, temporary);
            multiply(code, f64, targetValue, temporary, temporary);
            add(code, f64, loss, temporary, loss);
            code.labelBinding(zeroWeight).lload(clazz).loadConstant(1L).ladd().lstore(clazz)
                    .branch(Opcode.GOTO, weights).labelBinding(weightsDone);
        } else {
            code.lload(predictionAddress).lload(index).lload(classStride).lmul().ladd()
                    .lstore(inputAddress);
            loadFloating(code, carriers, specialization, predictionType, predictionBoundary,
                    inputAddress, predictionRepresented, value, f64);
            subtract(code, f64, sum, value, loss);
        }
        if (reduction == LossReduction.NONE) {
            store(code, carriers, specialization, resultType, outputBoundary, outputAddress, loss);
        } else {
            add(code, f64, reduced, loss, reduced);
            if (kind == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS && included >= 0) {
                code.lload(included).loadConstant(1L).ladd().lstore(included);
            }
        }
        code.labelBinding(next).lload(predictionAddress).loadConstant(1L).ladd()
                .lstore(predictionAddress).lload(targetAddress).loadConstant(1L).ladd()
                .lstore(targetAddress);
        if (reduction == LossReduction.NONE) {
            code.lload(outputAddress).loadConstant(1L).ladd().lstore(outputAddress);
        }
        code.lload(sample).loadConstant(1L).ladd().lstore(sample).branch(Opcode.GOTO, sampleLoop);
        code.labelBinding(sampleDone);
        // Move from the last inner sample to the next outer class-row block without deriving an
        // address from an ordinal.  Target/output sample rows are already adjacent.
        code.lload(predictionAddress).lload(classes).loadConstant(1L).lsub().lload(classStride)
                .lmul().ladd().lstore(predictionAddress);
        code.lload(outerIndex).loadConstant(1L).ladd().lstore(outerIndex)
                .branch(Opcode.GOTO, outerLoop).labelBinding(outerDone);
        if (reduction != LossReduction.NONE) {
            if (reduction == LossReduction.MEAN) {
                if (kind == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS) {
                    divideByCount(code, f64, reduced, included);
                } else {
                    geometry(code, specialization.boundaryDataTypes().size(), 9).lstore(included);
                    divideByCount(code, f64, reduced, included);
                }
            }
            geometry(code, specialization.boundaryDataTypes().size(), 6).lstore(outputAddress);
            store(code, carriers, specialization, resultType, outputBoundary, outputAddress, reduced);
        }
    }

    /**
     * Emits the categorical row traversal with int counters and element addresses.
     *
     * <p>The proved contiguous {@code inner} extent is used directly as the class-address
     * stride. Keeping that one primitive local, rather than copying it into a second stride
     * local, preserves the frozen clean-Java oracle's address dataflow while retaining the same
     * cold-geometry and int-range proof.</p>
     */
    private static void emitContiguousCategoricalInt(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, LossKind kind, boolean indexIgnorePresent, DataType predictionType,
            DataType targetType, DataType resultType, int predictionBoundary, int targetBoundary,
            int outputBoundary, LossReduction reduction, int predictionAddress, int targetAddress,
            int outputAddress, int classes, int clazz, int inputAddress, int index,
            int predictionRepresented, int targetRepresented, int value, int targetValue,
            int maximum, int sum, int loss, int reduced, int included, int outer,
            int inner, int axis, boolean f64) {
        if (kind == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS) {
            emitContiguousIndexCategoricalInt(code, carriers, specialization, indexIgnorePresent,
                    predictionType, targetType, resultType, predictionBoundary, targetBoundary,
                    outputBoundary, reduction, predictionAddress, targetAddress, outputAddress,
                    classes, clazz, index, predictionRepresented, value, maximum,
                    sum, loss, reduced, included, outer, inner, axis, f64);
            return;
        }
        int outerIndex = code.allocateLocal(TypeKind.INT), sample = code.allocateLocal(TypeKind.INT);
        int geometry = specialization.boundaryDataTypes().size();
        geometry(code, geometry, 4).l2i().istore(predictionAddress);
        geometry(code, geometry, 5).l2i().istore(targetAddress);
        // A reduced index form writes one scalar only after its complete ordered traversal.
        // Keep this cold base out of the class-loop live-local set; NONE needs it per sample.
        if (reduction == LossReduction.NONE) geometry(code, geometry, 6).l2i().istore(outputAddress);
        if (reduction != LossReduction.NONE) zero(code, f64, reduced);
        if (included >= 0) code.loadConstant(0).istore(included);
        code.loadConstant(0).istore(outerIndex);
        Label outerLoop = code.newLabel(), outerDone = code.newLabel();
        code.labelBinding(outerLoop).iload(outerIndex).iload(outer).branch(Opcode.IF_ICMPGE, outerDone);
        code.loadConstant(0).istore(sample);
        Label sampleLoop = code.newLabel(), sampleDone = code.newLabel();
        code.labelBinding(sampleLoop).iload(sample).iload(inner).branch(Opcode.IF_ICMPGE, sampleDone);
        Label next = code.newLabel();
        if (kind == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS) {
            loadIndex(code, carriers, specialization, targetType, targetBoundary, targetAddress, index, true);
            if (indexIgnorePresent) {
            Label notIgnored = code.newLabel();
            geometry(code, geometry, 7).loadConstant(0L).lcmp().branch(Opcode.IFEQ, notIgnored);
            code.lload(index); geometry(code, geometry, 8).lcmp().branch(Opcode.IFNE, notIgnored);
            zero(code, f64, loss);
            if (reduction == LossReduction.NONE) store(code, carriers, specialization, resultType,
                    outputBoundary, outputAddress, loss, true);
            code.branch(Opcode.GOTO, next); code.labelBinding(notIgnored);
            }
        }
            emitLseInt(code, carriers, specialization, predictionType, predictionBoundary,
                    predictionAddress, inner, classes, clazz, index, predictionRepresented,
                value, maximum, sum, f64);
        if (kind == LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS) {
            zero(code, f64, loss); code.loadConstant(0).istore(clazz);
            Label weights = code.newLabel(), weightsDone = code.newLabel();
            code.labelBinding(weights).iload(clazz).iload(classes).branch(Opcode.IF_ICMPGE, weightsDone);
            carriers.beginFrozenLoadAtStackIntAddress(targetType,
                    specialization.carrierPattern().get(targetBoundary), targetBoundary);
            code.iload(targetAddress).iload(clazz).iload(inner).imul().iadd();
            finishFloatingLoadAtStackIntAddress(code, carriers, specialization, targetType,
                    targetBoundary, f64);
            if (f64) code.dstore(targetValue); else code.fstore(targetValue);
            Label zeroWeight = code.newLabel(); compareZero(code, f64, targetValue).branch(Opcode.IFEQ, zeroWeight);
            // Match the frozen ordinary-Java contribution exactly: retain loss and weight,
            // then consume the direct logit load as lse - logit on the operand stack.  A
            // temporary subtraction/product local adds register pressure to every target-segment
            // categorical sample without preserving a semantic or carrier fact.
            if (f64) code.dload(loss).dload(targetValue).dload(sum);
            else code.fload(loss).fload(targetValue).fload(sum);
            carriers.beginFrozenLoadAtStackIntAddress(predictionType,
                    specialization.carrierPattern().get(predictionBoundary), predictionBoundary);
            code.iload(predictionAddress).iload(clazz).iload(inner).imul().iadd();
            finishFloatingLoadAtStackIntAddress(code, carriers, specialization, predictionType,
                    predictionBoundary, f64);
            if (f64) code.dsub().dmul().dadd().dstore(loss);
            else code.fsub().fmul().fadd().fstore(loss);
            code.labelBinding(zeroWeight).iinc(clazz, 1)
                    .branch(Opcode.GOTO, weights).labelBinding(weightsDone);
        } else {
            // Selected class indexes remain long semantically, but the cold validator guarantees
            // they are within the int class extent before this generated body is entered.
            if (f64) code.dload(sum); else code.fload(sum);
            carriers.beginFrozenLoadAtStackIntAddress(predictionType,
                    specialization.carrierPattern().get(predictionBoundary), predictionBoundary);
            code.iload(predictionAddress).lload(index).l2i().iload(inner).imul().iadd();
            finishFloatingLoadAtStackIntAddress(code, carriers, specialization, predictionType,
                    predictionBoundary, f64);
            if (f64) code.dsub().dstore(loss); else code.fsub().fstore(loss);
        }
        if (reduction == LossReduction.NONE) store(code, carriers, specialization, resultType,
                outputBoundary, outputAddress, loss, true);
        else {
            add(code, f64, reduced, loss, reduced);
            // The frozen clean-Java categorical oracle counts every dense sample as well as
            // every non-ignored index sample.  Dense has a statically equivalent total-domain
            // denominator, but replacing this loop-carried count with a late cold load changes
            // the selected loop's SSA/dataflow shape and can make C2 compile it differently.
            if (included >= 0) {
                code.iinc(included, 1);
            }
        }
        code.labelBinding(next).iinc(predictionAddress, 1).iinc(targetAddress, 1);
        if (reduction == LossReduction.NONE) code.iinc(outputAddress, 1);
        code.iinc(sample, 1).branch(Opcode.GOTO, sampleLoop).labelBinding(sampleDone);
        code.iload(predictionAddress).iload(classes).loadConstant(1).isub().iload(inner).imul()
                .iadd().istore(predictionAddress);
        // Dense targets retain the class coordinate, so after the compact inner samples we must
        // skip the remaining class entries exactly as logits do.  Index targets omit it.
        if (kind == LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS) {
            code.iload(targetAddress).iload(classes).loadConstant(1).isub().iload(inner)
                    .imul().iadd().istore(targetAddress);
        }
        code.iinc(outerIndex, 1).branch(Opcode.GOTO, outerLoop).labelBinding(outerDone);
        if (reduction != LossReduction.NONE) {
            if (reduction == LossReduction.MEAN) {
                divideByIntCount(code, f64, reduced, included);
            }
            /* clazz is dead after the class loops and is the frozen oracle's final output
             * address slot for a reduced categorical result. */
            geometry(code, geometry, 6).l2i().istore(outputAddress);
            store(code, carriers, specialization, resultType, outputBoundary, outputAddress, reduced, true);
        }
    }

    /*
     * The index form deliberately owns its complete contiguous body.  It has no dense-target
     * load, zero-weight branch, or target row adjustment: a target is one direct integer per
     * sample.  Keeping this body separate makes those facts visible in the generated Class-File
     * and prevents future dense changes from silently changing index's selected hot loop.
     */
    private static void emitContiguousIndexCategoricalInt(CodeBuilder code,
            CpuCarrierEmitter carriers, CpuKernelSpecialization specialization,
            boolean indexIgnorePresent, DataType predictionType, DataType targetType,
            DataType resultType, int predictionBoundary, int targetBoundary, int outputBoundary,
            LossReduction reduction, int predictionAddress, int targetAddress, int outputAddress,
            int classes, int clazz, int index, int predictionRepresented,
            int value, int maximum, int sum, int loss, int reduced, int included,
            int outer, int inner, int axis, boolean f64) {
        int outerIndex = code.allocateLocal(TypeKind.INT);
        int sample = code.allocateLocal(TypeKind.INT);
        int geometry = specialization.boundaryDataTypes().size();
        geometry(code, geometry, 4).l2i().istore(predictionAddress);
        geometry(code, geometry, 5).l2i().istore(targetAddress);
        // NONE writes once per sample.  A reduced result has no output address until its one
        // final store, matching the frozen clean-Java lifetime graph.
        if (reduction == LossReduction.NONE) geometry(code, geometry, 6).l2i().istore(outputAddress);
        // The caller's cold prelude already loaded the index class extent.  The direct index
        // body never reads a reduced accumulator for NONE, matching the clean Java oracle.
        if (reduction != LossReduction.NONE) zero(code, f64, reduced);
        if (included >= 0) code.loadConstant(0).istore(included);
        code.loadConstant(0).istore(outerIndex);
        Label outerLoop = code.newLabel(), outerDone = code.newLabel();
        code.labelBinding(outerLoop).iload(outerIndex).iload(outer).branch(Opcode.IF_ICMPGE, outerDone);
        code.loadConstant(0).istore(sample);
        Label sampleLoop = code.newLabel(), sampleDone = code.newLabel();
        code.labelBinding(sampleLoop).iload(sample).iload(inner).branch(Opcode.IF_ICMPGE, sampleDone);
        loadIndex(code, carriers, specialization, targetType, targetBoundary, targetAddress, index, true);
        if (indexIgnorePresent) {
            Label notIgnored = code.newLabel();
            geometry(code, geometry, 7).loadConstant(0L).lcmp().branch(Opcode.IFEQ, notIgnored);
            code.lload(index); geometry(code, geometry, 8).lcmp().branch(Opcode.IFNE, notIgnored);
            zero(code, f64, loss);
            if (reduction == LossReduction.NONE) {
                store(code, carriers, specialization, resultType, outputBoundary, outputAddress,
                        loss, true);
            }
            // Exact clean-Java continue flow: advance each sample carrier then re-test the loop.
            code.iinc(predictionAddress, 1).iinc(targetAddress, 1);
            if (reduction == LossReduction.NONE) code.iinc(outputAddress, 1);
            code.iinc(sample, 1).branch(Opcode.GOTO, sampleLoop);
            code.labelBinding(notIgnored);
        }
        emitLseInt(code, carriers, specialization, predictionType, predictionBoundary,
                predictionAddress, inner, classes, clazz, index, predictionRepresented,
                value, maximum, sum, f64);
        if (f64) code.dload(sum); else code.fload(sum);
        carriers.beginFrozenLoadAtStackIntAddress(predictionType,
                specialization.carrierPattern().get(predictionBoundary), predictionBoundary);
        code.iload(predictionAddress).lload(index).l2i().iload(inner).imul().iadd();
        finishFloatingLoadAtStackIntAddress(code, carriers, specialization, predictionType,
                predictionBoundary, f64);
        if (f64) code.dsub().dstore(loss); else code.fsub().fstore(loss);
        if (reduction == LossReduction.NONE) {
            store(code, carriers, specialization, resultType, outputBoundary, outputAddress,
                    loss, true);
        } else {
            add(code, f64, reduced, loss, reduced);
            if (included >= 0) code.iinc(included, 1);
        }
        code.iinc(predictionAddress, 1).iinc(targetAddress, 1);
        if (reduction == LossReduction.NONE) code.iinc(outputAddress, 1);
        code.iinc(sample, 1).branch(Opcode.GOTO, sampleLoop).labelBinding(sampleDone);
        code.iload(predictionAddress).iload(classes).loadConstant(1).isub().iload(inner).imul()
                .iadd().istore(predictionAddress);
        code.iinc(outerIndex, 1).branch(Opcode.GOTO, outerLoop).labelBinding(outerDone);
        if (reduction != LossReduction.NONE) {
            if (reduction == LossReduction.MEAN) divideByIntCount(code, f64, reduced, included);
            geometry(code, geometry, 6).l2i().istore(outputAddress);
            store(code, carriers, specialization, resultType, outputBoundary, outputAddress,
                    reduced, true);
        }
    }

    private static void emitLseInt(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType predictionType, int predictionBoundary,
            int predictionBase, int classStride, int classes, int clazz, int address,
            int represented, int value, int maximum, int sum, boolean f64) {
        if (f64) code.loadConstant(Double.NEGATIVE_INFINITY).dstore(maximum);
        else code.loadConstant(Float.NEGATIVE_INFINITY).fstore(maximum);
        code.loadConstant(0).istore(clazz);
        Label maximumLoop = code.newLabel(), maximumDone = code.newLabel();
        code.labelBinding(maximumLoop).iload(clazz).iload(classes).branch(Opcode.IF_ICMPGE, maximumDone);
        /* Match the frozen clean-Java max pass: the class address is consumed by its one load.
           Retaining it in inputAddress forces a store/reload on every class and leaves avoidable
           register pressure in the selected contiguous body. */
        carriers.beginFrozenLoadAtStackIntAddress(predictionType,
                specialization.carrierPattern().get(predictionBoundary), predictionBoundary);
        code.iload(predictionBase).iload(clazz).iload(classStride).imul().iadd();
        finishFloatingLoadAtStackIntAddress(code, carriers, specialization, predictionType,
                predictionBoundary, f64);
        if (f64) code.dstore(value); else code.fstore(value);
        Label keep = code.newLabel(); compare(code, f64, value, maximum).branch(Opcode.IFLE, keep);
        copy(code, f64, value, maximum); code.labelBinding(keep).iinc(clazz, 1).branch(Opcode.GOTO, maximumLoop).labelBinding(maximumDone);
        zero(code, f64, sum); code.loadConstant(0).istore(clazz);
        Label sumLoop = code.newLabel(), sumDone = code.newLabel();
        code.labelBinding(sumLoop).iload(clazz).iload(classes).branch(Opcode.IF_ICMPGE, sumDone);
        /* Keep the second logit pass on the operand stack, matching the frozen ordinary-Java
           oracle.  Unlike the classification pass, its value is consumed exactly once; spilling
           it to the reusable value local adds a store/reload to every class without preserving
           any semantic or carrier fact. */
        if (f64) code.dload(sum); else code.fload(sum);
        carriers.beginFrozenLoadAtStackIntAddress(predictionType,
                specialization.carrierPattern().get(predictionBoundary), predictionBoundary);
        code.iload(predictionBase).iload(clazz).iload(classStride).imul().iadd();
        finishFloatingLoadAtStackIntAddress(code, carriers, specialization, predictionType,
                predictionBoundary, f64);
        if (f64) code.dload(maximum).dsub().invokestatic(STRICT_MATH, "exp", doubleUnary())
                .dadd().dstore(sum);
        else code.fload(maximum).fsub().f2d().invokestatic(STRICT_MATH, "exp", doubleUnary())
                .d2f().fadd().fstore(sum);
        code.iinc(clazz, 1).branch(Opcode.GOTO, sumLoop).labelBinding(sumDone);
        // As in ordinary Java's `lse = maximum + log(sum)`, compose the log result directly
        // with maximum.  Index loss does not reuse a log temporary, so a spill/reload here is
        // pure overhead in every sample and disproportionately harms the binary32 traversal.
        if (f64) code.dload(maximum).dload(sum).invokestatic(STRICT_MATH, "log", doubleUnary())
                .dadd().dstore(sum);
        else code.fload(maximum).fload(sum).f2d().invokestatic(STRICT_MATH, "log", doubleUnary())
                .d2f().fadd().fstore(sum);
    }

    /**
     * Finishes one direct typed carrier load after its proved {@code int} element address has
     * been supplied on the operand stack, preserving the accumulator representation without a
     * temporary local.
     *
     * @param code non-null Class-File builder with the carrier and int element address on its
     *     operand stack
     * @param carriers non-null carrier emitter that began the matching frozen load
     * @param specialization non-null selected typed carrier specialization
     * @param type non-null floating source type represented by the pending load
     * @param boundary non-negative selected source-carrier parameter index
     * @param f64 whether the caller's accumulator is binary64 rather than binary32
     */
    private static void finishFloatingLoadAtStackIntAddress(CodeBuilder code,
            CpuCarrierEmitter carriers, CpuKernelSpecialization specialization, DataType type,
            int boundary, boolean f64) {
        carriers.endFrozenLoadAtStackIntAddress(type,
                specialization.carrierPattern().get(boundary));
        if (type == DataType.FLOAT64) return;
        if (type == DataType.FLOAT32) {
            if (f64) code.f2d();
            return;
        }
        if (type != DataType.BFLOAT16) throw new IllegalArgumentException("loss floating type");
        code.loadConstant(16).ishl().invokestatic(FLOAT, "intBitsToFloat", floatFromBits());
        if (f64) code.f2d();
    }

    private static void emitLse(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType predictionType, int predictionBoundary,
            int predictionBase, int classStride, int classes, int clazz, int address,
            int represented, int value, int maximum, int sum, int temporary, boolean f64) {
        if (f64) {
            code.loadConstant(Double.NEGATIVE_INFINITY).dstore(maximum);
        } else {
            code.loadConstant(Float.NEGATIVE_INFINITY).fstore(maximum);
        }
        code.loadConstant(0L).lstore(clazz);
        Label maximumLoop = code.newLabel();
        Label maximumDone = code.newLabel();
        code.labelBinding(maximumLoop).lload(clazz).lload(classes).lcmp()
                .branch(Opcode.IFGE, maximumDone);
        code.lload(predictionBase).lload(clazz).lload(classStride).lmul().ladd().lstore(address);
        loadFloating(code, carriers, specialization, predictionType, predictionBoundary, address,
                represented, value, f64);
        Label keep = code.newLabel();
        compare(code, f64, value, maximum).branch(Opcode.IFLE, keep);
        copy(code, f64, value, maximum);
        code.labelBinding(keep).lload(clazz).loadConstant(1L).ladd().lstore(clazz)
                .branch(Opcode.GOTO, maximumLoop).labelBinding(maximumDone);

        zero(code, f64, sum);
        code.loadConstant(0L).lstore(clazz);
        Label sumLoop = code.newLabel();
        Label sumDone = code.newLabel();
        code.labelBinding(sumLoop).lload(clazz).lload(classes).lcmp().branch(Opcode.IFGE, sumDone);
        code.lload(predictionBase).lload(clazz).lload(classStride).lmul().ladd().lstore(address);
        loadFloating(code, carriers, specialization, predictionType, predictionBoundary, address,
                represented, value, f64);
        subtract(code, f64, value, maximum, temporary);
        if (f64) {
            code.dload(temporary).invokestatic(STRICT_MATH, "exp", doubleUnary()).dstore(temporary);
        } else {
            code.fload(temporary).f2d().invokestatic(STRICT_MATH, "exp", doubleUnary())
                    .d2f().fstore(temporary);
        }
        add(code, f64, sum, temporary, sum);
        code.lload(clazz).loadConstant(1L).ladd().lstore(clazz).branch(Opcode.GOTO, sumLoop)
                .labelBinding(sumDone);
        if (f64) {
            code.dload(sum).invokestatic(STRICT_MATH, "log", doubleUnary()).dstore(temporary);
        } else {
            code.fload(sum).f2d().invokestatic(STRICT_MATH, "log", doubleUnary())
                    .d2f().fstore(temporary);
        }
        add(code, f64, maximum, temporary, sum);
    }

    private static void addresses(CodeBuilder code, int geometry, LossKind kind,
            LossReduction reduction, int rank, int axis, int targetRank, int outputRank,
            int ordinal, int remainder, int coordinate, int logicalAxis, int targetAxis,
            int prediction, int target, int output) {
        geometry(code, geometry, 4).lstore(prediction);
        geometry(code, geometry, 5).lstore(target);
        geometry(code, geometry, 6).lstore(output);
        code.lload(ordinal).lstore(remainder);
        code.iload(rank).loadConstant(1).isub().istore(logicalAxis);
        Label coordinates = code.newLabel();
        Label coordinatesDone = code.newLabel();
        code.labelBinding(coordinates).iload(logicalAxis).branch(Opcode.IFLT, coordinatesDone);
        if (kind != LossKind.MEAN_SQUARED_ERROR) {
            Label notClass = code.newLabel();
            code.iload(logicalAxis).iload(axis).branch(Opcode.IF_ICMPNE, notClass);
            code.iinc(logicalAxis, -1).branch(Opcode.GOTO, coordinates);
            code.labelBinding(notClass);
        }
        code.lload(remainder); extent(code, geometry, logicalAxis).lrem().lstore(coordinate);
        code.lload(remainder); extent(code, geometry, logicalAxis).ldiv().lstore(remainder);
        code.lload(prediction).lload(coordinate); predictionStride(code, geometry, rank, logicalAxis)
                .lmul().ladd().lstore(prediction);
        if (kind == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS) {
            Label beforeClass = code.newLabel();
            Label mapped = code.newLabel();
            code.iload(logicalAxis).iload(axis).branch(Opcode.IF_ICMPLT, beforeClass);
            code.iload(logicalAxis).loadConstant(1).isub().istore(targetAxis)
                    .branch(Opcode.GOTO, mapped);
            code.labelBinding(beforeClass).iload(logicalAxis).istore(targetAxis)
                    .labelBinding(mapped);
        } else {
            code.iload(logicalAxis).istore(targetAxis);
        }
        code.lload(target).lload(coordinate); targetStride(code, geometry, rank, targetRank, targetAxis)
                .lmul().ladd().lstore(target);
        if (reduction == LossReduction.NONE) {
            if (kind != LossKind.MEAN_SQUARED_ERROR) {
                Label beforeClass = code.newLabel();
                Label mapped = code.newLabel();
                code.iload(logicalAxis).iload(axis).branch(Opcode.IF_ICMPLT, beforeClass);
                code.iload(logicalAxis).loadConstant(1).isub().istore(targetAxis)
                        .branch(Opcode.GOTO, mapped);
                code.labelBinding(beforeClass).iload(logicalAxis).istore(targetAxis)
                        .labelBinding(mapped);
            }
            code.lload(output).lload(coordinate);
            outputStride(code, geometry, rank, targetRank, outputRank, targetAxis).lmul().ladd()
                    .lstore(output);
        }
        code.iinc(logicalAxis, -1).branch(Opcode.GOTO, coordinates)
                .labelBinding(coordinatesDone);
    }

    private static void loadFloating(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType type, int boundary, int address,
            int represented, int destination, boolean f64) {
        loadFloating(code, carriers, specialization, type, boundary, address, represented, destination, f64, false);
    }

    private static void loadFloating(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType type, int boundary, int address,
            int represented, int destination, boolean f64, boolean intAddress) {
        carriers.loadFrozen(type, specialization.carrierPattern().get(boundary), boundary, address,
                intAddress);
        // Same-domain FLOAT32/FLOAT64 loads already have the exact accumulator representation.
        // Do not round-trip them through a represented local: the frozen Java oracle consumes the
        // direct array/segment load in the arithmetic local as well.
        if (f64 && type == DataType.FLOAT64) {
            code.dstore(destination);
            return;
        }
        if (!f64 && type == DataType.FLOAT32) {
            code.fstore(destination);
            return;
        }
        storeRepresented(code, type, represented);
        if (f64) {
            if (type == DataType.FLOAT64) code.dload(represented).dstore(destination);
            else if (type == DataType.FLOAT32) code.fload(represented).f2d().dstore(destination);
            else code.iload(represented).loadConstant(16).ishl().invokestatic(FLOAT,
                    "intBitsToFloat", floatFromBits()).f2d().dstore(destination);
        } else if (type == DataType.FLOAT32) {
            code.fload(represented).fstore(destination);
        } else if (type == DataType.BFLOAT16) {
            code.iload(represented).loadConstant(16).ishl().invokestatic(FLOAT, "intBitsToFloat",
                    floatFromBits()).fstore(destination);
        } else {
            throw new IllegalArgumentException("binary32 loss cannot decode FLOAT64 input");
        }
    }

    private static void loadIndex(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType type, int boundary, int address,
            int destination) {
        loadIndex(code, carriers, specialization, type, boundary, address, destination, false);
    }

    private static void loadIndex(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType type, int boundary, int address,
            int destination, boolean intAddress) {
        carriers.loadFrozen(type, specialization.carrierPattern().get(boundary), boundary, address,
                intAddress);
        if (type == DataType.INT32) code.i2l().lstore(destination);
        else if (type == DataType.INT64) code.lstore(destination);
        else throw new IllegalArgumentException("loss index type");
    }

    private static void store(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType type, int boundary, int address,
            int result) {
        store(code, carriers, specialization, type, boundary, address, result, false);
    }

    private static void store(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType type, int boundary, int address,
            int result, boolean intAddress) {
        int represented = result;
        if (type == DataType.BFLOAT16) {
            represented = code.allocateLocal(TypeKind.INT);
            int bits = code.allocateLocal(TypeKind.INT);
            code.fload(result).invokestatic(FLOAT, "floatToRawIntBits", bitsFromFloat()).istore(bits);
            Label finite = code.newLabel();
            Label rounded = code.newLabel();
            code.iload(bits).loadConstant(0x7fffffff).iand().loadConstant(0x7f800000)
                    .branch(Opcode.IF_ICMPLE, finite).loadConstant(0x7fc0).istore(represented)
                    .branch(Opcode.GOTO, rounded).labelBinding(finite).iload(bits)
                    .loadConstant(0x7fff).iadd().iload(bits).loadConstant(16).iushr().loadConstant(1)
                    .iand().iadd().loadConstant(16).iushr().istore(represented)
                    .labelBinding(rounded);
        }
        carriers.storeFrozen(type, specialization.carrierPattern().get(boundary), boundary, address,
                represented, intAddress);
    }

    /**
     * Narrows the floating loss local to canonical BFLOAT16 bits on the operand stack.
     *
     * <p>The carrier and proved output address remain below that value.  Keeping the conversion
     * stack-local lets the contiguous MSE {@code NONE} loop retain the clean Java body's direct
     * address/load/store dataflow instead of allocating a per-element rounding temporary.</p>
     */
    private static void emitBfloat16StoreValue(CodeBuilder code, int result) {
        int bits = code.allocateLocal(TypeKind.INT);
        int represented = code.allocateLocal(TypeKind.INT);
        code.fload(result).invokestatic(FLOAT, "floatToRawIntBits", bitsFromFloat()).istore(bits);
        Label finite = code.newLabel();
        Label rounded = code.newLabel();
        code.iload(bits).loadConstant(0x7fffffff).iand().loadConstant(0x7f800000)
                .branch(Opcode.IF_ICMPLE, finite).loadConstant(0x7fc0).istore(represented)
                .branch(Opcode.GOTO, rounded).labelBinding(finite).iload(bits)
                .loadConstant(0x7fff).iadd().iload(bits).loadConstant(16).iushr().loadConstant(1)
                .iand().iadd().loadConstant(16).iushr().istore(represented)
                .labelBinding(rounded).iload(represented);
    }

    private static void divideByCount(CodeBuilder code, boolean f64, int result, int count) {
        if (f64) code.dload(result).lload(count).l2d().ddiv().dstore(result);
        else code.fload(result).lload(count).l2f().fdiv().fstore(result);
    }

    private static void divideByIntCount(CodeBuilder code, boolean f64, int result, int count) {
        if (f64) code.dload(result).iload(count).i2d().ddiv().dstore(result);
        else code.fload(result).iload(count).i2f().fdiv().fstore(result);
    }

    private static void zero(CodeBuilder code, boolean f64, int local) {
        if (f64) code.loadConstant(0.0d).dstore(local);
        else code.loadConstant(0.0f).fstore(local);
    }

    private static void copy(CodeBuilder code, boolean f64, int source, int destination) {
        if (f64) code.dload(source).dstore(destination); else code.fload(source).fstore(destination);
    }

    private static void add(CodeBuilder code, boolean f64, int left, int right, int destination) {
        if (f64) code.dload(left).dload(right).dadd().dstore(destination);
        else code.fload(left).fload(right).fadd().fstore(destination);
    }

    private static void subtract(CodeBuilder code, boolean f64, int left, int right,
            int destination) {
        if (f64) code.dload(left).dload(right).dsub().dstore(destination);
        else code.fload(left).fload(right).fsub().fstore(destination);
    }

    private static void multiply(CodeBuilder code, boolean f64, int left, int right,
            int destination) {
        if (f64) code.dload(left).dload(right).dmul().dstore(destination);
        else code.fload(left).fload(right).fmul().fstore(destination);
    }

    private static CodeBuilder compare(CodeBuilder code, boolean f64, int left, int right) {
        return f64 ? code.dload(left).dload(right).dcmpl()
                : code.fload(left).fload(right).fcmpl();
    }

    private static CodeBuilder compareZero(CodeBuilder code, boolean f64, int value) {
        return f64 ? code.dload(value).loadConstant(0.0d).dcmpl()
                : code.fload(value).loadConstant(0.0f).fcmpl();
    }

    private static CodeBuilder geometry(CodeBuilder code, int local, int index) {
        return code.aload(local).loadConstant(index).laload();
    }

    private static CodeBuilder extent(CodeBuilder code, int geometry, int logicalAxis) {
        return code.aload(geometry).loadConstant(10).iload(logicalAxis).iadd().laload();
    }

    private static CodeBuilder predictionStride(CodeBuilder code, int geometry, int rank,
            int logicalAxis) {
        return code.aload(geometry).loadConstant(10).iload(rank).iadd().iload(logicalAxis).iadd()
                .laload();
    }

    private static CodeBuilder targetStride(CodeBuilder code, int geometry, int rank,
            int targetRank, int targetAxis) {
        return code.aload(geometry).loadConstant(10).iload(rank).iadd().iload(rank).iadd()
                .iload(targetAxis).iadd().laload();
    }

    private static CodeBuilder outputStride(CodeBuilder code, int geometry, int rank,
            int targetRank, int outputRank, int outputAxis) {
        return code.aload(geometry).loadConstant(10).iload(rank).iadd().iload(rank).iadd()
                .iload(targetRank).iadd().iload(outputAxis).iadd().laload();
    }

    private static void storeRepresented(CodeBuilder code, DataType type, int local) {
        switch (type) {
            case FLOAT64 -> code.dstore(local);
            case FLOAT32 -> code.fstore(local);
            case BFLOAT16, INT32 -> code.istore(local);
            case INT64 -> code.lstore(local);
            default -> throw new IllegalArgumentException("loss type");
        }
    }

    private static TypeKind representedKind(DataType type) {
        return switch (type) {
            case FLOAT64 -> TypeKind.DOUBLE;
            case FLOAT32 -> TypeKind.FLOAT;
            case BFLOAT16, INT32 -> TypeKind.INT;
            case INT64 -> TypeKind.LONG;
            default -> throw new IllegalArgumentException("loss type");
        };
    }

    private static TypeKind arithmeticKind(boolean f64) {
        return f64 ? TypeKind.DOUBLE : TypeKind.FLOAT;
    }

    private static boolean floating(DataType type) {
        return type == DataType.BFLOAT16 || type == DataType.FLOAT32 || type == DataType.FLOAT64;
    }

    private static boolean integral(DataType type) {
        return type == DataType.INT32 || type == DataType.INT64;
    }

    private static MethodTypeDesc doubleUnary() {
        return MethodTypeDesc.of(TypeKind.DOUBLE.upperBound(), TypeKind.DOUBLE.upperBound());
    }

    private static MethodTypeDesc floatFromBits() {
        return MethodTypeDesc.of(TypeKind.FLOAT.upperBound(), TypeKind.INT.upperBound());
    }

    private static MethodTypeDesc bitsFromFloat() {
        return MethodTypeDesc.of(TypeKind.INT.upperBound(), TypeKind.FLOAT.upperBound());
    }

    private static int[] roles(String identity) {
        String text = field(identity, "roles=");
        if (text.length() < 5 || text.charAt(0) != '[' || text.charAt(text.length() - 1) != ']') {
            throw new IllegalArgumentException("loss identity roles");
        }
        String[] values = text.substring(1, text.length() - 1).split(", ");
        if (values.length != 2) throw new IllegalArgumentException("loss identity roles");
        return new int[] {Integer.parseInt(values[0]), Integer.parseInt(values[1])};
    }

    private static String field(String identity, String key) {
        int start = identity.indexOf(key);
        if (start < 0) throw new IllegalArgumentException("loss identity missing " + key);
        start += key.length();
        int end = identity.indexOf(':', start);
        return end < 0 ? identity.substring(start) : identity.substring(start, end);
    }
}

package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;
import java.lang.constant.ConstantDescs;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.IntConsumer;

/**
 * Emits allocation-free full-K direct-scalar, direct-N-vector, scalar-2x2, and
 * vector-2x2 portable MATMUL bodies. Common bound scalar contractions own increasing full-K
 * control and represented math, independently of entry or temporal binding. Generation-only
 * loaders/activity callbacks inline bytecode; they never become runtime calls or allocate per
 * step. Binding owners retain accumulator-zero/address initialization, output gates and stores.
 * Controlled scalar bindings gate original ROW outputs before any eager operand load and
 * gate ROW-contracting contributors before both operand loaders. Only authenticated boundary
 * axes identify K as the original row; inactive contributors retain unconditional cursor
 * advancement and never enter represented arithmetic.
 * Floating vector TANH is applied after the full-K sum and optional ordered bias ADD, before
 * the sole output store, using the same typed mathematics as pointwise vectors. Scalar N tails
 * retain the scalar epilogue; no terminal is applied inside the contraction loop.
 * Homogeneous low contractions use FLOAT32 throughout full K and narrow exactly once at the
 * output store; unlike low operands use the Model-promoted FLOAT32 or FLOAT64 result working type.
 * Realizations without an exact emitted loop shape fail before class generation; this type never
 * relabels the scalar fallback as a tiled or vector-tiled artifact.
 */
public final class CpuMatmulEmitter {
    /** Creates a stateless MATMUL emitter. */
    public CpuMatmulEmitter() { }

    /**
     * Initializes a represented scalar accumulator without selecting contributor traversal.
     * @param code nonnull generation builder
     * @param dataType represented accumulator dtype accepted by the ordinary math owner
     * @param accumulator allocated typed accumulator slot
     */
    static void emitInit(CodeBuilder code, DataType dataType, int accumulator) {
        zero(code,dataType,accumulator);
    }

    /**
     * Emits one typed product, assigning it for a singleton contraction or adding it otherwise.
     * No FMA is introduced. Both loaders must be enclosed by the caller's activity gate.
     * @param code nonnull generation builder
     * @param dataType represented working dtype
     * @param accumulator initialized typed accumulator slot, updated here
     * @param directProduct whether the complete logical contraction has exactly one contributor
     * @param left generation action leaving the ordered left typed operand on stack
     * @param right generation action leaving the ordered right typed operand on stack
     * @throws IllegalArgumentException if the dtype is not supported by this scalar seam
     */
    static void emitContributorStep(CodeBuilder code, DataType dataType, int accumulator,
            boolean directProduct, Runnable left, Runnable right) {
        var kind = switch (dataType) {
            case FLOAT64 -> TypeKind.DOUBLE;
            case FLOAT32, BFLOAT16, FLOAT16, FLOAT8_E4M3, FLOAT8_E5M2 -> TypeKind.FLOAT;
            case INT32 -> TypeKind.INT;
            case INT64 -> TypeKind.LONG;
            case BOOL -> throw new IllegalArgumentException("BOOL contraction");
        };
        if (!directProduct) code.loadLocal(kind,accumulator);
        left.run(); right.run();
        switch (kind) {
            case DOUBLE -> { code.dmul(); if (!directProduct) code.dadd(); }
            case FLOAT -> { code.fmul(); if (!directProduct) code.fadd(); }
            case INT -> { code.imul(); if (!directProduct) code.iadd(); }
            case LONG -> { code.lmul(); if (!directProduct) code.ladd(); }
            default -> throw new AssertionError(kind);
        }
        code.storeLocal(kind,accumulator);
    }

    /**
     * Publishes only the completed accumulator; no extra arithmetic or rounding is introduced.
     * @param code nonnull generation builder owning the accumulator
     * @param dataType represented accumulator dtype
     * @param accumulator initialized typed completed-result slot
     * @param sink generation-only result consumer receiving that exact slot
     */
    static void emitFinish(CodeBuilder code, DataType dataType, int accumulator, IntConsumer sink) {
        sink.accept(accumulator);
    }

    /**
     * Initializes a feature-vector accumulator to typed positive zero, without reducing lanes.
     * @param code builder owning the reference local
     * @param type homogeneous FLOAT32 or FLOAT64 preferred-species dtype
     * @param accumulator allocated reference local
     * @throws IllegalArgumentException if the dtype is not a supported floating vector
     */
    static void emitVectorInit(CodeBuilder code, DataType type, int accumulator) {
        if (type != DataType.FLOAT32 && type != DataType.FLOAT64)
            throw new IllegalArgumentException("row vector requires FLOAT32 or FLOAT64");
        ClassDesc vector = vectorClass(type), species = ClassDesc.of("jdk.incubator.vector.VectorSpecies");
        code.getstatic(vector,"SPECIES_PREFERRED",species)
                .invokestatic(vector,"zero",MethodTypeDesc.of(vector,species)).astore(accumulator);
    }

    /**
     * Emits a lane-wise product, assigning it for a singleton or adding it otherwise, never FMA
     * or horizontal reduction. The caller's activity gate must dominate both operand loaders.
     * @param code builder owning the initialized reference local
     * @param type homogeneous FLOAT32 or FLOAT64 lane type
     * @param accumulator initialized matching preferred-species vector local, replaced here
     * @param directProduct whether the complete logical contraction has exactly one contributor
     * @param left generation-only loader leaving a matching left vector
     * @param right generation-only loader leaving a matching right vector
     * @throws IllegalArgumentException if the dtype is unsupported
     */
    static void emitVectorContributorStep(CodeBuilder code, DataType type, int accumulator,
            boolean directProduct, Runnable left, Runnable right) {
        if (type != DataType.FLOAT32 && type != DataType.FLOAT64)
            throw new IllegalArgumentException("row vector requires FLOAT32 or FLOAT64");
        ClassDesc vector = vectorClass(type), operand = ClassDesc.of("jdk.incubator.vector.Vector");
        if (!directProduct) code.aload(accumulator);
        left.run(); right.run();
        code.invokevirtual(vector,"mul",MethodTypeDesc.of(vector,operand));
        if (!directProduct) code.invokevirtual(vector,"add",MethodTypeDesc.of(vector,operand));
        code.astore(accumulator);
    }

    /**
     * Publishes a completed vector without extra arithmetic or a horizontal reduction.
     * @param code builder owning the reference local
     * @param type selected FLOAT32 or FLOAT64 lane type
     * @param accumulator completed matching vector local
     * @param sink generation-only consumer of that exact local
     */
    static void emitVectorFinish(CodeBuilder code, DataType type, int accumulator, IntConsumer sink) {
        sink.accept(accumulator);
    }

    /**
     * Emits the common cursor contraction over already bound operands and an initialized sum.
     * An active-contributor wrapper must gate both loaders and dependent math; cursor advances
     * remain unconditional so inactive contributors do not compact or reorder K. All callback
     * invocations occur during generation only. The zero-trip edge preserves the caller's
     * represented positive zero and no operand local escapes that edge.
     * @param code non-null entry builder; sole owner of JVM-local allocation
     * @param type exact represented accumulator type
     * @param kKind INT or LONG proved K induction kind, independent of address width
     * @param k allocated induction local, initialized here
     * @param sum initialized typed accumulator local
     * @param leftValue allocated typed left working-value local
     * @param rightValue allocated typed right working-value local
     * @param loadKBound emits a bound of kKind on the stack without runtime binding work
     * @param loadLeft fills leftValue from the selected carrier/address
     * @param loadRight fills rightValue from the selected carrier/address
     * @param advanceOperands emits unconditional cursor advances, or no code for indexed bases
     * @param activeContributor emits the supplied math only within the proved contributor gate;
     *     uncontrolled callers use Runnable::run
     * @throws IllegalArgumentException if kKind is not INT or LONG
     */
    static void emitCursorContraction(CodeBuilder code, DataType type, TypeKind kKind,
            int k, int sum, int leftValue, int rightValue, Runnable loadKBound,
            Runnable loadLeft, Runnable loadRight, Runnable advanceOperands,
            Consumer<Runnable> activeContributor) {
        emitBoundContraction(code,k,kKind,loadKBound,directProduct ->
                activeContributor.accept(() -> {
                    loadLeft.run(); loadRight.run();
                    accumulate(code,type,sum,leftValue,rightValue,directProduct);
                }),advanceOperands);
    }

    /**
     * Emits direct indexed scalar K control/math with operands left on the stack by bound
     * loaders. The contributor gate precedes both addresses/loads and multiplication; there
     * are no operand cursors, operand-value copies or hidden helper calls in emitted code.
     * @param code non-null method builder
     * @param type admitted FLOAT32 or FLOAT64 represented type
     * @param k allocated proved INT induction local
     * @param sum initialized typed accumulator local, including empty-K positive zero
     * @param loadKBound emits the checked INT contraction bound
     * @param loadLeftAtK emits the selected left value for the supplied original K local
     * @param loadRightAtK emits the selected right value for the supplied original K local
     * @param activeContributor generation-only activity wrapper, or Runnable::run if unguarded
     */
    static void emitIndexedContraction(CodeBuilder code, DataType type, int k, int sum,
            Runnable loadKBound, IntConsumer loadLeftAtK, IntConsumer loadRightAtK,
            Consumer<Runnable> activeContributor) {
        emitBoundContraction(code,k,TypeKind.INT,loadKBound,directProduct ->
                activeContributor.accept(() -> emitContributorStep(code,type,sum,directProduct,
                        () -> loadLeftAtK.accept(k),() -> loadRightAtK.accept(k))),() -> {});
    }

    /**
     * Emits geometry-only singleton selection before contraction traversal. Each branch owns
     * its generation state; callers must rebuild mutable load caches before emitting a branch.
     * @param code builder owning the branch labels
     * @param kind INT or LONG bound representation
     * @param bound generation action leaving the complete logical contraction extent
     * @param body emits the singleton assignment body or the unchanged ordinary body
     */
    static void emitSingletonChoice(CodeBuilder code, TypeKind kind, Runnable bound,
            Consumer<Boolean> body) {
        boolean ints = contractionKind(kind);
        var ordinary = code.newLabel(); var done = code.newLabel();
        bound.run();
        if (ints) code.loadConstant(1).branch(Opcode.IF_ICMPNE,ordinary);
        else code.loadConstant(1L).lcmp().branch(Opcode.IFNE,ordinary);
        body.accept(true);
        code.branch(Opcode.GOTO,done).labelBinding(ordinary);
        body.accept(false);
        code.labelBinding(done);
    }

    /**
     * Binds singleton selection and full increasing-K control without a per-contributor mode
     * branch. Empty contractions execute neither operand loader; cursor advances remain
     * unconditional after every physical contributor, including inactive singleton lanes.
     * @param code builder owning locals and labels
     * @param k allocated induction local, initialized here
     * @param kind INT or LONG induction and bound representation
     * @param bound generation action leaving the complete logical K extent
     * @param contributor emits typed product assignment or ordered MUL then ADD
     * @param advance generation action advancing both operand cursors unconditionally
     */
    private static void emitBoundContraction(CodeBuilder code, int k, TypeKind kind,
            Runnable bound, Consumer<Boolean> contributor, Runnable advance) {
        boolean ints = contractionKind(kind);
        zeroIndex(code,k,ints);
        emitSingletonChoice(code,kind,bound,directProduct -> {
            if (directProduct) {
                contributor.accept(true); advance.run(); incrementIndex(code,k,ints);
            } else {
                var loop = code.newLabel(); var done = code.newLabel();
                code.labelBinding(loop).loadLocal(kind,k); bound.run();
                if (ints) code.branch(Opcode.IF_ICMPGE,done);
                else code.lcmp().branch(Opcode.IFGE,done);
                contributor.accept(false); advance.run(); incrementIndex(code,k,ints);
                code.branch(Opcode.GOTO,loop).labelBinding(done);
            }
        });
    }

    /**
     * Validates the independent contraction induction proof without inspecting runtime extents.
     * @param kind non-null proposed integral induction kind
     * @return true for INT and false for LONG
     * @throws IllegalArgumentException if kind is neither admitted induction kind
     */
    private static boolean contractionKind(TypeKind kind) {
        if (kind != TypeKind.INT && kind != TypeKind.LONG)
            throw new IllegalArgumentException("MATMUL K must be INT or LONG");
        return kind == TypeKind.INT;
    }

    /**
     * Wraps represented scalar work only when an authenticated operand ROW axis contracts.
     * The ordinary input plans use effective MATMUL operand coordinates; controlled lowering
     * must retain that same coordinate domain in its boundary provenance. The K coordinate is
     * an original row only for the left final axis or the right penultimate axis (axis zero
     * for a vector), independently of extents, strides, or the other operand's shape.
     * @param code current scalar method builder
     * @param ir exact effective operand access ranks
     * @param k original increasing contraction-coordinate local
     * @param kind INT or LONG contraction-coordinate representation
     * @return generation-only wrapper gating both operand loads and all contributor arithmetic,
     *     or the unchanged direct invocation when no ROW axis contracts
     */
    private static Consumer<Runnable> activeContributor(CodeBuilder code, CpuMatmulIr ir,
            int k, TypeKind kind) {
        var control = CpuCarrierEmitter.controlFacts();
        if (control.isEmpty()) return Runnable::run;
        var rows = control.orElseThrow().boundaryRows();
        int leftRank = ir.inputAccesses().get(0).iterationRank();
        int rightRank = ir.inputAccesses().get(1).iterationRank();
        boolean leftContracts = rows.get(0).axis() >= 0
                && rows.get(0).axis() == leftRank - 1;
        boolean rightContracts = rows.get(1).axis() >= 0
                && rows.get(1).axis() == Math.max(0, rightRank - 2);
        if (!leftContracts && !rightContracts) return Runnable::run;
        return contributor -> {
            var inactive = code.newLabel();
            CpuCarrierEmitter.emitActiveRowGuard(code, k, kind, inactive);
            contributor.run();
            code.labelBinding(inactive);
        };
    }

    /**
     * Gates one rank-two ordinary-batch result cell using the proved output ROW axis.
     * This gate precedes all eager loads, including SHARED operands and any output store.
     * @param code current ordinary-batch method builder
     * @param row original M coordinate in an INT local
     * @param column original N coordinate in an INT local
     * @param inactive target retaining normal column advancement
     */
    private static void emitBatchOutputGuard(CodeBuilder code, int row, int column,
            java.lang.classfile.Label inactive) {
        var control = CpuCarrierEmitter.controlFacts();
        if (control.isEmpty()) return;
        var output = control.orElseThrow().boundaryRows().get(2);
        if (output.axis() < 0) return;
        if (output.axis() > 1)
            throw new IllegalArgumentException("ordinary batch MATMUL output ROW rank disagrees");
        CpuCarrierEmitter.emitActiveRowGuard(code, output.axis() == 0 ? row : column,
                TypeKind.INT, inactive);
    }

    /**
     * Emits the selected realization over a half-open range of flattened work units. A scalar
     * work unit owns one output cell, a direct-vector work unit owns one complete M row, and a
     * tiled work unit owns one disjoint M/N microtile. Every owned accumulator traverses the full
     * K extent before its optional epilogue and sole store.
     *
     * @param code non-null method builder
     * @param specialization non-null exact carriers and selected compute identity
     * @param ir non-null canonical MATMUL identity
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if family, carrier, or boundary facts disagree, or a
     *     controlled binding requests a realization other than direct scalar
     */
    public void emit(CodeBuilder code, CpuKernelSpecialization specialization, CpuMatmulIr ir) {
        int expectedCarriers = ir.epilogue().hasBias() ? 4 : 3;
        if (specialization.carrierPattern().size() != expectedCarriers)
            throw new IllegalArgumentException("MATMUL generated facts disagree");
        if (CpuCarrierEmitter.controlFacts().isPresent()
                && ir.realization() != CpuMatmulIr.Realization.DIRECT_SCALAR)
            throw new IllegalArgumentException("controlled MATMUL requires direct scalar realization");
        if (specialization.ordinaryBatch()) {
            emitOrdinaryBatch(code, specialization, ir);
            return;
        }
        if (ir.realization() == CpuMatmulIr.Realization.DIRECT_N_VECTOR) {
            emitDirectVector(code, specialization, ir);
            return;
        }
        if (ir.realization() == CpuMatmulIr.Realization.TILED_SCALAR_2X2) {
            emitTiledScalar(code, specialization, ir);
            return;
        }
        if (ir.realization() == CpuMatmulIr.Realization.TILED_N_VECTOR_2X2) {
            emitTiledVector(code, specialization, ir);
            return;
        }
        if (ir.realization() != CpuMatmulIr.Realization.DIRECT_SCALAR)
            throw new IllegalArgumentException("MATMUL realization is not implemented");
        DataType leftType = specialization.boundaryDataTypes().get(0);
        DataType rightType = specialization.boundaryDataTypes().get(1);
        DataType resultType = specialization.boundaryDataTypes().get(2);
        int geometry = specialization.carrierPattern().size(), start = geometry + 1,
                end = geometry + 3;
        boolean intAddress = heapArrays(specialization);
        var carriers = new CpuCarrierEmitter(code);
        int ordinal = code.allocateLocal(TypeKind.LONG), remaining = code.allocateLocal(TypeKind.LONG);
        int n = code.allocateLocal(TypeKind.LONG), m = code.allocateLocal(TypeKind.LONG);
        int batch = code.allocateLocal(TypeKind.LONG), axis = code.allocateLocal(TypeKind.INT);
        int coordinate = code.allocateLocal(TypeKind.LONG);
        int leftBase = code.allocateLocal(TypeKind.LONG), rightBase = code.allocateLocal(TypeKind.LONG);
        int outputBase = code.allocateLocal(TypeKind.LONG), leftAddress = code.allocateLocal(TypeKind.LONG);
        int rightAddress = code.allocateLocal(TypeKind.LONG), outputAddress = code.allocateLocal(TypeKind.LONG);
        if (intAddress) {
            leftAddress = code.allocateLocal(TypeKind.INT);
            rightAddress = code.allocateLocal(TypeKind.INT);
            outputAddress = code.allocateLocal(TypeKind.INT);
        }
        int kExtent = code.allocateLocal(intAddress ? TypeKind.INT : TypeKind.LONG);
        int leftKStride = code.allocateLocal(intAddress ? TypeKind.INT : TypeKind.LONG);
        int rightKStride = code.allocateLocal(intAddress ? TypeKind.INT : TypeKind.LONG);
        int k = code.allocateLocal(intAddress ? TypeKind.INT : TypeKind.LONG);
        TypeKind accumulatorKind = resultType == DataType.FLOAT64 ? TypeKind.DOUBLE
                : resultType == DataType.INT64 ? TypeKind.LONG
                : resultType == DataType.INT32 ? TypeKind.INT : TypeKind.FLOAT;
        int accumulator = code.allocateLocal(accumulatorKind);
        int leftRepresented = code.allocateLocal(representedKind(leftType));
        int rightRepresented = code.allocateLocal(representedKind(rightType));
        int leftValue = code.allocateLocal(accumulatorKind), rightValue = code.allocateLocal(accumulatorKind);
        int biasBase=-1,biasStride=-1;
        if(ir.epilogue().hasBias()) {
            biasBase=code.allocateLocal(TypeKind.LONG);biasStride=code.allocateLocal(TypeKind.LONG);
            geometry(code,geometry,14).lstore(biasBase);geometry(code,geometry,15).lstore(biasStride);
        }
        geometry(code, geometry, 5); storeIndex(code,kExtent,intAddress);
        geometry(code, geometry, 8); storeIndex(code, leftKStride, intAddress);
        geometry(code, geometry, 9); storeIndex(code, rightKStride, intAddress);
        code.lload(start).lstore(ordinal);
        var cells = code.newLabel(); var done = code.newLabel();
        code.labelBinding(cells).lload(ordinal).lload(end).lcmp().branch(Opcode.IFGE, done);
        var nextCell = code.newLabel();
        CpuLoopEmitter.emitActiveOrdinalGuard(code, expectedCarriers - 1, ordinal,
                TypeKind.LONG, nextCell);
        code.lload(ordinal).lstore(remaining);
        code.lload(remaining); geometry(code, geometry, 6).lrem().lstore(n);
        code.lload(remaining); geometry(code, geometry, 6).ldiv().lstore(remaining);
        code.lload(remaining); geometry(code, geometry, 4).lrem().lstore(m);
        code.lload(remaining); geometry(code, geometry, 4).ldiv().lstore(batch);
        geometry(code, geometry, 0).lstore(leftBase); geometry(code, geometry, 1).lstore(rightBase);
        geometry(code, geometry, 2).lstore(outputBase);
        geometry(code, geometry, 13).l2i().loadConstant(1).isub().istore(axis);
        var batches = code.newLabel(); var batchesDone = code.newLabel();
        code.labelBinding(batches).iload(axis).branch(Opcode.IFLT, batchesDone);
        code.lload(batch).aload(geometry).iload(axis).loadConstant(18).iadd().laload().lrem()
                .lstore(coordinate);
        code.lload(batch).aload(geometry).iload(axis).loadConstant(18).iadd().laload().ldiv()
                .lstore(batch);
        addBatch(code, geometry, axis, coordinate, leftBase, 18);
        addBatch(code, geometry, axis, coordinate, rightBase, 18, 2);
        addBatch(code, geometry, axis, coordinate, outputBase, 18, 3);
        code.iinc(axis, -1).branch(Opcode.GOTO, batches).labelBinding(batchesDone);
        code.lload(leftBase).lload(m); geometry(code, geometry, 7).lmul().ladd();
        storeIndex(code, leftAddress, intAddress);
        code.lload(rightBase).lload(n); geometry(code, geometry, 10).lmul().ladd();
        storeIndex(code, rightAddress, intAddress);
        code.lload(outputBase).lload(m); geometry(code, geometry, 11).lmul().ladd()
                .lload(n); geometry(code, geometry, 12).lmul().ladd();
        storeIndex(code, outputAddress, intAddress);
        zero(code, resultType, accumulator);
        int boundLeftAddress = leftAddress, boundRightAddress = rightAddress;
        emitCursorContraction(code,resultType,intAddress ? TypeKind.INT : TypeKind.LONG,
                k,accumulator,leftValue,rightValue,() -> code.loadLocal(intAddress ? TypeKind.INT : TypeKind.LONG,kExtent),
                () -> load(code,carriers,specialization,leftType,resultType,0,boundLeftAddress,leftRepresented,leftValue,intAddress),
                () -> load(code,carriers,specialization,rightType,resultType,1,boundRightAddress,rightRepresented,rightValue,intAddress),
                () -> { addIndex(code,boundLeftAddress,leftKStride,intAddress); addIndex(code,boundRightAddress,rightKStride,intAddress); },
                activeContributor(code,ir,k,intAddress ? TypeKind.INT : TypeKind.LONG));
        applyScalarEpilogue(code,carriers,specialization,ir,geometry,n,accumulator,
                biasBase,biasStride);
        int outputBoundary=specialization.carrierPattern().size()-1;
        if (resultType == DataType.INT32 || resultType == DataType.INT64) {
            storeIntegral(code, carriers, specialization, resultType, outputBoundary,
                    outputAddress, accumulator, intAddress);
        } else if(resultType.isFloating() && resultType.byteWidth() <= 2) {
            int raw = code.allocateLocal(TypeKind.INT);
            CpuCastEmitter.emitFloatToLow(code, resultType, accumulator);
            code.istore(raw);
            carriers.storeFrozen(resultType, specialization.carrierPattern().get(outputBoundary),
                    outputBoundary, outputAddress, raw, intAddress);
        } else {
            carriers.storeFrozen(resultType,specialization.carrierPattern().get(outputBoundary),
                    outputBoundary,outputAddress,accumulator,intAddress);
        }
        code.labelBinding(nextCell);
        code.lload(ordinal).loadConstant(1L).ladd().lstore(ordinal)
                .branch(Opcode.GOTO, cells).labelBinding(done);
    }

    /**
     * Emits rank-two row/column/full-K loops with fixed F/H and constant native-order layouts.
     * Frozen carrier access avoids a per-invocation layout object obscuring FFM constant folding.
     * The ordinary binder supplies the complete output domain: an entry guard selects its
     * canonical loops without per-row boundary selection or range division. A separate
     * subrange body preserves the generated entry's half-open flattened-cell ABI.
     * @param code non-null current method builder, mutated by emission
     * @param specialization non-null exact scalar FLOAT32 carrier specialization
     * @param ir non-null matrix facts with prepared B-only coefficients
     */
    private static void emitOrdinaryBatch(CodeBuilder code, CpuKernelSpecialization specialization, CpuMatmulIr ir) {
        var recipe = java.util.Objects.requireNonNull(ir.batchGeometry(), "batchGeometry");
        int geometrySlot = 3, startSlot = 4, endSlot = 6;
        int m = code.allocateLocal(TypeKind.INT), n = code.allocateLocal(TypeKind.INT), kExtent = code.allocateLocal(TypeKind.INT);
        int aBase = code.allocateLocal(TypeKind.INT), bBase = code.allocateLocal(TypeKind.INT), cBase = code.allocateLocal(TypeKind.INT);
        int am = code.allocateLocal(TypeKind.INT), ak = code.allocateLocal(TypeKind.INT);
        int bk = code.allocateLocal(TypeKind.INT), bn = code.allocateLocal(TypeKind.INT);
        int cm = code.allocateLocal(TypeKind.INT), cn = code.allocateLocal(TypeKind.INT);
        batchFact(code, geometrySlot, 4, recipe.left().extentConstants().get(0), recipe.left().extentBatchFactors().get(0)).istore(m);
        batchFact(code, geometrySlot, 6, recipe.right().extentConstants().get(1), recipe.right().extentBatchFactors().get(1)).istore(n);
        batchFact(code, geometrySlot, 5, recipe.left().extentConstants().get(1), recipe.left().extentBatchFactors().get(1)).istore(kExtent);
        geometry(code, geometrySlot, 0).l2i().istore(aBase); geometry(code, geometrySlot, 1).l2i().istore(bBase);
        geometry(code, geometrySlot, 2).l2i().istore(cBase);
        batchFact(code, geometrySlot, 7, recipe.left().strideConstants().get(0), recipe.left().strideBatchFactors().get(0)).istore(am);
        batchFact(code, geometrySlot, 8, recipe.left().strideConstants().get(1), recipe.left().strideBatchFactors().get(1)).istore(ak);
        batchFact(code, geometrySlot, 9, recipe.right().strideConstants().get(0), recipe.right().strideBatchFactors().get(0)).istore(bk);
        batchFact(code, geometrySlot, 10, recipe.right().strideConstants().get(1), recipe.right().strideBatchFactors().get(1)).istore(bn);
        batchFact(code, geometrySlot, 11, recipe.result().strideConstants().get(0), recipe.result().strideBatchFactors().get(0)).istore(cm);
        batchFact(code, geometrySlot, 12, recipe.result().strideConstants().get(1), recipe.result().strideBatchFactors().get(1)).istore(cn);
        var done = code.newLabel();
        code.lload(startSlot).lload(endSlot).lcmp().branch(Opcode.IFGE, done);
        int firstRow = code.allocateLocal(TypeKind.INT), lastRow = code.allocateLocal(TypeKind.INT);
        int firstColumn = code.allocateLocal(TypeKind.INT), lastColumn = code.allocateLocal(TypeKind.INT);
        int row = code.allocateLocal(TypeKind.INT), column = code.allocateLocal(TypeKind.INT), stop = code.allocateLocal(TypeKind.INT);
        int leftRow = code.allocateLocal(TypeKind.INT), outputRow = code.allocateLocal(TypeKind.INT);
        int leftAddress = code.allocateLocal(TypeKind.INT), rightAddress = code.allocateLocal(TypeKind.INT), outputAddress = code.allocateLocal(TypeKind.INT);
        int k = code.allocateLocal(TypeKind.INT), accumulator = code.allocateLocal(TypeKind.FLOAT);
        int leftValue = code.allocateLocal(TypeKind.FLOAT), rightValue = code.allocateLocal(TypeKind.FLOAT);
        var subrange = code.newLabel();
        code.lload(startSlot).loadConstant(0L).lcmp().branch(Opcode.IFNE, subrange);
        code.lload(endSlot).iload(m).i2l().iload(n).i2l().lmul().lcmp().branch(Opcode.IFNE, subrange);
        // Emit two bodies, not a run-time mode branch inside the canonical row/column/K loops.
        for (boolean fullRange : new boolean[] {true, false}) {
            if (fullRange) {
                code.loadConstant(0).istore(row);
            } else {
                code.labelBinding(subrange);
                code.lload(startSlot).l2i().iload(n).idiv().istore(firstRow);
                code.lload(startSlot).l2i().iload(n).irem().istore(firstColumn);
                code.lload(endSlot).loadConstant(1L).lsub().l2i().iload(n).idiv().istore(lastRow);
                code.lload(endSlot).loadConstant(1L).lsub().l2i().iload(n).irem().loadConstant(1).iadd().istore(lastColumn);
                code.iload(firstRow).istore(row);
            }
            var rows = code.newLabel(); var columns = code.newLabel(); var nextRow = code.newLabel();
            code.labelBinding(rows).iload(row);
            if (fullRange) code.iload(m).branch(Opcode.IF_ICMPGE, done);
            else code.iload(lastRow).branch(Opcode.IF_ICMPGT, done);
            code.iload(aBase).iload(row).iload(am).imul().iadd().istore(leftRow);
            code.iload(cBase).iload(row).iload(cm).imul().iadd().istore(outputRow);
            if (fullRange) {
                code.loadConstant(0).istore(column).iload(n).istore(stop);
            } else {
                var notFirst = code.newLabel(); var startReady = code.newLabel();
                code.iload(row).iload(firstRow).branch(Opcode.IF_ICMPNE, notFirst);
                code.iload(firstColumn).branch(Opcode.GOTO, startReady).labelBinding(notFirst).loadConstant(0).labelBinding(startReady).istore(column);
                var notLast = code.newLabel(); var stopReady = code.newLabel();
                code.iload(row).iload(lastRow).branch(Opcode.IF_ICMPNE, notLast);
                code.iload(lastColumn).branch(Opcode.GOTO, stopReady).labelBinding(notLast).iload(n).labelBinding(stopReady).istore(stop);
            }
            code.labelBinding(columns).iload(column).iload(stop).branch(Opcode.IF_ICMPGE, nextRow);
            var nextColumn = code.newLabel();
            emitBatchOutputGuard(code,row,column,nextColumn);
            code.iload(leftRow).istore(leftAddress).iload(bBase).iload(column).iload(bn).imul().iadd().istore(rightAddress);
            code.loadConstant(0.0f).fstore(accumulator);
            var carriers = new CpuCarrierEmitter(code);
            emitCursorContraction(code,DataType.FLOAT32,TypeKind.INT,k,accumulator,leftValue,rightValue,
                    () -> code.iload(kExtent),
                    () -> { carriers.loadFrozen(DataType.FLOAT32,specialization.carrierPattern().get(0),0,leftAddress,true); code.fstore(leftValue); },
                    () -> { carriers.loadFrozen(DataType.FLOAT32,specialization.carrierPattern().get(1),1,rightAddress,true); code.fstore(rightValue); },
                    () -> { code.iload(leftAddress).iload(ak).iadd().istore(leftAddress); code.iload(rightAddress).iload(bk).iadd().istore(rightAddress); },
                    activeContributor(code,ir,k,TypeKind.INT));
            code.iload(outputRow).iload(column).iload(cn).imul().iadd().istore(outputAddress);
            carriers.storeFrozen(DataType.FLOAT32, specialization.carrierPattern().get(2), 2, outputAddress, accumulator, true);
            code.labelBinding(nextColumn);
            code.iinc(column, 1).branch(Opcode.GOTO, columns).labelBinding(nextRow).iinc(row, 1).branch(Opcode.GOTO, rows);
        }
        code.labelBinding(done);
    }

    private static CodeBuilder batchFact(CodeBuilder code, int geometry, int index, long constant, long factor) {
        if (factor == 0) return code.loadConstant(Math.toIntExact(constant));
        return geometry(code, geometry, index).l2i();
    }

    private static void emitTiledScalar(CodeBuilder code,
            CpuKernelSpecialization specialization, CpuMatmulIr ir) {
        DataType type=ir.resultType();int geometry=specialization.carrierPattern().size();
        boolean intAddress=heapArrays(specialization);
        int start=geometry+1,end=geometry+3;
        var carriers=new CpuCarrierEmitter(code);
        int ordinal=code.allocateLocal(TypeKind.LONG),remaining=code.allocateLocal(TypeKind.LONG);
        int nTiles=code.allocateLocal(TypeKind.LONG),mTiles=code.allocateLocal(TypeKind.LONG);
        int mt=code.allocateLocal(TypeKind.LONG),nt=code.allocateLocal(TypeKind.LONG);
        int m0=code.allocateLocal(TypeKind.LONG),n0=code.allocateLocal(TypeKind.LONG);
        int batch=code.allocateLocal(TypeKind.LONG),batchRemaining=code.allocateLocal(TypeKind.LONG);
        int axis=code.allocateLocal(TypeKind.INT);
        int coordinate=code.allocateLocal(TypeKind.LONG),m1=code.allocateLocal(TypeKind.INT);
        int n1=code.allocateLocal(TypeKind.INT),leftBase=code.allocateLocal(TypeKind.LONG);
        int rightBase=code.allocateLocal(TypeKind.LONG),outputBase=code.allocateLocal(TypeKind.LONG);
        int a0=code.allocateLocal(intAddress?TypeKind.INT:TypeKind.LONG);
        int a1=code.allocateLocal(intAddress?TypeKind.INT:TypeKind.LONG);
        int b0=code.allocateLocal(intAddress?TypeKind.INT:TypeKind.LONG);
        int b1=code.allocateLocal(intAddress?TypeKind.INT:TypeKind.LONG);
        int kExtent=code.allocateLocal(intAddress?TypeKind.INT:TypeKind.LONG);
        int leftKStride=code.allocateLocal(intAddress?TypeKind.INT:TypeKind.LONG);
        int rightKStride=code.allocateLocal(intAddress?TypeKind.INT:TypeKind.LONG);
        int k=code.allocateLocal(intAddress?TypeKind.INT:TypeKind.LONG);
        TypeKind kind=type==DataType.FLOAT64?TypeKind.DOUBLE:type==DataType.INT64?TypeKind.LONG
                :type==DataType.INT32?TypeKind.INT:TypeKind.FLOAT;
        int[] sums={code.allocateLocal(kind),code.allocateLocal(kind),code.allocateLocal(kind),
                code.allocateLocal(kind)};
        int lv0=code.allocateLocal(kind),lv1=code.allocateLocal(kind);
        int rv0=code.allocateLocal(kind),rv1=code.allocateLocal(kind);
        TypeKind leftRepresentedKind=representedKind(ir.leftType());
        TypeKind rightRepresentedKind=representedKind(ir.rightType());
        int lr0=leftRepresentedKind==kind?lv0:code.allocateLocal(leftRepresentedKind);
        int lr1=leftRepresentedKind==kind?lv1:code.allocateLocal(leftRepresentedKind);
        int rr0=rightRepresentedKind==kind?rv0:code.allocateLocal(rightRepresentedKind);
        int rr1=rightRepresentedKind==kind?rv1:code.allocateLocal(rightRepresentedKind);
        geometry(code,geometry,5);storeIndex(code,kExtent,intAddress);
        geometry(code,geometry,8);storeIndex(code,leftKStride,intAddress);
        geometry(code,geometry,9);storeIndex(code,rightKStride,intAddress);
        geometry(code,geometry,6).loadConstant(1L).ladd().loadConstant(2L).ldiv().lstore(nTiles);
        geometry(code,geometry,4).loadConstant(1L).ladd().loadConstant(2L).ldiv().lstore(mTiles);
        code.lload(start).lstore(ordinal).lload(start).lstore(remaining);
        code.lload(remaining).lload(nTiles).lrem().lstore(nt);
        code.lload(remaining).lload(nTiles).ldiv().lstore(remaining);
        code.lload(remaining).lload(mTiles).lrem().lstore(mt);
        code.lload(remaining).lload(mTiles).ldiv().lstore(batch);
        var prepareBatch=code.newLabel();var tiles=code.newLabel();var done=code.newLabel();
        code.labelBinding(prepareBatch);
        resolveBatchBases(code,geometry,batch,batchRemaining,axis,coordinate,
                leftBase,rightBase,outputBase);
        code.branch(Opcode.GOTO,tiles).labelBinding(tiles).lload(ordinal).lload(end).lcmp()
                .branch(Opcode.IFGE,done);
        code.lload(mt).loadConstant(2L).lmul().lstore(m0);
        code.lload(nt).loadConstant(2L).lmul().lstore(n0);
        lessThanNext(code,m0,geometry,4,m1);lessThanNext(code,n0,geometry,6,n1);
        code.lload(leftBase).lload(m0);geometry(code,geometry,7).lmul().ladd();
        storeIndex(code,a0,intAddress);
        loadIndexAsLong(code,a0,intAddress);geometry(code,geometry,7).ladd();
        storeIndex(code,a1,intAddress);
        code.lload(rightBase).lload(n0);geometry(code,geometry,10).lmul().ladd();
        storeIndex(code,b0,intAddress);
        loadIndexAsLong(code,b0,intAddress);geometry(code,geometry,10).ladd();
        storeIndex(code,b1,intAddress);
        for(int sum:sums)zero(code,type,sum);
        zero(code,type,lv1);zero(code,type,rv1);
        emitBoundContraction(code,k,intAddress ? TypeKind.INT : TypeKind.LONG,
                () -> code.loadLocal(intAddress ? TypeKind.INT : TypeKind.LONG,kExtent),directProduct -> {
        load(code,carriers,specialization,ir.leftType(),type,0,a0,lr0,lv0,intAddress);
        load(code,carriers,specialization,ir.rightType(),type,1,b0,rr0,rv0,intAddress);
        accumulate(code,type,sums[0],lv0,rv0,directProduct);
        var skipN=code.newLabel();code.iload(n1).branch(Opcode.IFEQ,skipN);
        load(code,carriers,specialization,ir.rightType(),type,1,b1,rr1,rv1,intAddress);
        accumulate(code,type,sums[1],lv0,rv1,directProduct);code.labelBinding(skipN);
        var skipM=code.newLabel();code.iload(m1).branch(Opcode.IFEQ,skipM);
        load(code,carriers,specialization,ir.leftType(),type,0,a1,lr1,lv1,intAddress);
        accumulate(code,type,sums[2],lv1,rv0,directProduct);
        var skipBoth=code.newLabel();code.iload(n1).branch(Opcode.IFEQ,skipBoth);
        accumulate(code,type,sums[3],lv1,rv1,directProduct);code.labelBinding(skipBoth).labelBinding(skipM);
        },() -> {
        addIndex(code,a0,leftKStride,intAddress);addIndex(code,a1,leftKStride,intAddress);
        addIndex(code,b0,rightKStride,intAddress);addIndex(code,b1,rightKStride,intAddress);
        });
        int nextN=code.allocateLocal(TypeKind.LONG),nextM=code.allocateLocal(TypeKind.LONG);
        code.lload(n0).loadConstant(1L).ladd().lstore(nextN);
        code.lload(m0).loadConstant(1L).ladd().lstore(nextM);
        storeTileValue(code,carriers,specialization,ir,geometry,outputBase,m0,n0,sums[0],intAddress);
        var noNStore=code.newLabel();code.iload(n1).branch(Opcode.IFEQ,noNStore);
        storeTileValue(code,carriers,specialization,ir,geometry,outputBase,m0,nextN,sums[1],intAddress);
        code.labelBinding(noNStore);var noMStore=code.newLabel();code.iload(m1).branch(Opcode.IFEQ,noMStore);
        storeTileValue(code,carriers,specialization,ir,geometry,outputBase,nextM,n0,sums[2],intAddress);
        var noBothStore=code.newLabel();code.iload(n1).branch(Opcode.IFEQ,noBothStore);
        storeTileValue(code,carriers,specialization,ir,geometry,outputBase,nextM,nextN,sums[3],intAddress);
        code.labelBinding(noBothStore).labelBinding(noMStore);
        code.lload(ordinal).loadConstant(1L).ladd().lstore(ordinal);
        code.lload(nt).loadConstant(1L).ladd().lstore(nt);
        var advanceM=code.newLabel();code.lload(nt).lload(nTiles).lcmp().branch(Opcode.IFGE,advanceM)
                .branch(Opcode.GOTO,tiles).labelBinding(advanceM).loadConstant(0L).lstore(nt)
                .lload(mt).loadConstant(1L).ladd().lstore(mt);
        var nextBatch=code.newLabel();code.lload(mt).lload(mTiles).lcmp()
                .branch(Opcode.IFGE,nextBatch).branch(Opcode.GOTO,tiles).labelBinding(nextBatch)
                .loadConstant(0L).lstore(mt).lload(batch).loadConstant(1L).ladd().lstore(batch)
                .branch(Opcode.GOTO,prepareBatch).labelBinding(done);
    }

    private static void lessThanNext(CodeBuilder code,int coordinate,int geometry,int extentIndex,
            int target) {
        var yes=code.newLabel();var done=code.newLabel();
        code.lload(coordinate).loadConstant(1L).ladd();geometry(code,geometry,extentIndex).lcmp()
                .branch(Opcode.IFLT,yes).loadConstant(0).istore(target).branch(Opcode.GOTO,done)
                .labelBinding(yes).loadConstant(1).istore(target).labelBinding(done);
    }

    private static void storeTileValue(CodeBuilder code,CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization,CpuMatmulIr ir,int geometry,int outputBase,
            int m,int n,int value,boolean intAddress) {
        DataType type=ir.resultType();
        int address=code.allocateLocal(TypeKind.LONG);
        code.lload(outputBase).lload(m);geometry(code,geometry,11).lmul().ladd().lload(n);
        geometry(code,geometry,12).lmul().ladd();
        if(intAddress)address=code.allocateLocal(TypeKind.INT);
        storeIndex(code,address,intAddress);
        applyScalarEpilogue(code,carriers,specialization,ir,geometry,n,value);
        int outputBoundary=specialization.carrierPattern().size()-1;
        if(type==DataType.INT32||type==DataType.INT64)
            storeIntegral(code,carriers,specialization,type,outputBoundary,address,value,intAddress);
        else if(type.isFloating() && type.byteWidth() <= 2) {
            int raw = code.allocateLocal(TypeKind.INT);
            CpuCastEmitter.emitFloatToLow(code, type, value);
            code.istore(raw);
            carriers.storeFrozen(type, specialization.carrierPattern().get(outputBoundary),
                    outputBoundary, address, raw, intAddress);
        } else carriers.storeFrozen(type,specialization.carrierPattern().get(outputBoundary),
                outputBoundary,address,value,intAddress);
    }

    private static void emitDirectVector(CodeBuilder code,
            CpuKernelSpecialization specialization, CpuMatmulIr ir) {
        DataType type = ir.resultType();
        int lanes = specialization.vectorSpeciesBitSize() / type.bitWidth();
        if (lanes <= 1 || ir.leftType() != type || ir.rightType() != type)
            throw new IllegalArgumentException("MATMUL vector facts disagree");
        ClassDesc vector = vectorClass(type);
        boolean intAddress=heapArrays(specialization);
        int geometry=specialization.carrierPattern().size(),start=geometry+1,end=geometry+3;
        var carriers = new CpuCarrierEmitter(code);
        carriers.prepareVectorSpecies(type);
        int ordinal=code.allocateLocal(TypeKind.LONG),remaining=code.allocateLocal(TypeKind.LONG);
        int m=code.allocateLocal(intAddress?TypeKind.INT:TypeKind.LONG);
        int batch=code.allocateLocal(TypeKind.LONG);
        int batchRemaining=code.allocateLocal(TypeKind.LONG);
        int axis=code.allocateLocal(TypeKind.INT),coordinate=code.allocateLocal(TypeKind.LONG);
        int leftBase=code.allocateLocal(TypeKind.LONG),rightBase=code.allocateLocal(TypeKind.LONG);
        int outputBase=code.allocateLocal(TypeKind.LONG);
        int n=code.allocateLocal(intAddress?TypeKind.INT:TypeKind.LONG);
        int leftRowBase=code.allocateLocal(TypeKind.LONG);
        int outputRowBase=code.allocateLocal(TypeKind.LONG);
        int leftAddress=code.allocateLocal(intAddress?TypeKind.INT:TypeKind.LONG);
        int rightAddress=code.allocateLocal(intAddress?TypeKind.INT:TypeKind.LONG);
        int outputAddress=code.allocateLocal(intAddress?TypeKind.INT:TypeKind.LONG);
        int rightVectorBase=code.allocateLocal(TypeKind.INT);
        int outputVectorBase=code.allocateLocal(TypeKind.INT);
        int rightNStride=code.allocateLocal(TypeKind.INT);
        int outputNStride=code.allocateLocal(TypeKind.INT);
        int rightVectorStep=code.allocateLocal(TypeKind.INT);
        int outputVectorStep=code.allocateLocal(TypeKind.INT);
        int k=code.allocateLocal(intAddress?TypeKind.INT:TypeKind.LONG);
        int kExtent=code.allocateLocal(intAddress?TypeKind.INT:TypeKind.LONG);
        int leftKStride=code.allocateLocal(intAddress?TypeKind.INT:TypeKind.LONG);
        int rightKStride=code.allocateLocal(intAddress?TypeKind.INT:TypeKind.LONG);
        int accumulator=code.allocateLocal(TypeKind.REFERENCE);
        int leftScalar=code.allocateLocal(type==DataType.FLOAT64?TypeKind.DOUBLE
                :type==DataType.FLOAT32?TypeKind.FLOAT:type==DataType.INT64?TypeKind.LONG
                :TypeKind.INT);
        geometry(code,geometry,5);storeIndex(code,kExtent,intAddress);
        geometry(code,geometry,8);storeIndex(code,leftKStride,intAddress);
        geometry(code,geometry,9);storeIndex(code,rightKStride,intAddress);
        if(intAddress){geometry(code,geometry,10).l2i().istore(rightNStride);
            geometry(code,geometry,12).l2i().istore(outputNStride);
            code.iload(rightNStride).loadConstant(lanes).imul().istore(rightVectorStep);
            code.iload(outputNStride).loadConstant(lanes).imul().istore(outputVectorStep);}
        code.lload(start).lstore(ordinal).lload(start).lstore(remaining);
        code.lload(remaining);geometry(code,geometry,4).lrem();storeIndex(code,m,intAddress);
        code.lload(remaining);geometry(code,geometry,4).ldiv().lstore(batch);
        var prepareBatch=code.newLabel();var rows=code.newLabel();var done=code.newLabel();
        code.labelBinding(prepareBatch);
        resolveBatchBases(code,geometry,batch,batchRemaining,axis,coordinate,
                leftBase,rightBase,outputBase);
        code.branch(Opcode.GOTO,rows).labelBinding(rows).lload(ordinal).lload(end).lcmp()
                .branch(Opcode.IFGE,done);
        code.lload(leftBase);loadIndexAsLong(code,m,intAddress);geometry(code,geometry,7).lmul()
                .ladd().lstore(leftRowBase);
        code.lload(outputBase);loadIndexAsLong(code,m,intAddress);geometry(code,geometry,11).lmul()
                .ladd().lstore(outputRowBase);
        if(intAddress){code.lload(rightBase).l2i().istore(rightVectorBase);
            code.lload(outputRowBase).l2i().istore(outputVectorBase);}
        zeroIndex(code,n,intAddress);
        var vectors=code.newLabel();var vectorDone=code.newLabel();
        code.labelBinding(vectors);
        if(intAddress)code.iload(n).loadConstant(lanes).iadd().aload(geometry).loadConstant(6)
                .laload().l2i().branch(Opcode.IF_ICMPGT,vectorDone);
        else code.lload(n).loadConstant((long)lanes).ladd().aload(geometry).loadConstant(6)
                .laload().lcmp().branch(Opcode.IFGT,vectorDone);
        code.lload(leftRowBase);storeIndex(code,leftAddress,intAddress);
        if(intAddress){code.iload(rightVectorBase).istore(rightAddress);
            code.iload(outputVectorBase).istore(outputAddress);}
        else {code.lload(rightBase).lload(n);geometry(code,geometry,10).lmul().ladd()
                    .lstore(rightAddress);
            code.lload(outputRowBase).lload(n);geometry(code,geometry,12).lmul().ladd()
                    .lstore(outputAddress);}
        code.getstatic(vector,"SPECIES_PREFERRED",ClassDesc.of("jdk.incubator.vector.VectorSpecies"));
        code.invokestatic(vector,"zero",MethodTypeDesc.of(vector,
                ClassDesc.of("jdk.incubator.vector.VectorSpecies"))).astore(accumulator);
        emitBoundContraction(code,k,intAddress ? TypeKind.INT : TypeKind.LONG,
                () -> code.loadLocal(intAddress ? TypeKind.INT : TypeKind.LONG,kExtent),directProduct -> {
        loadSameTypeScalar(code,carriers,specialization,type,0,leftAddress,leftScalar,intAddress);
        if (!directProduct) code.aload(accumulator);
        carriers.vectorLoad(type,specialization.carrierPattern().get(1),1,rightAddress,false,intAddress);
        loadPrimitive(code,type,leftScalar);
        code.invokevirtual(vector,"mul",MethodTypeDesc.of(vector,primitive(type)));
        if (!directProduct) code.invokevirtual(vector,"add",
                MethodTypeDesc.of(vector,ClassDesc.of("jdk.incubator.vector.Vector")));
        code.astore(accumulator);
        },() -> {
        addIndex(code,leftAddress,leftKStride,intAddress);
        addIndex(code,rightAddress,rightKStride,intAddress);
        });
        int epilogueN=n;
        if(ir.epilogue().hasBias()&&intAddress){epilogueN=code.allocateLocal(TypeKind.LONG);
            code.iload(n).i2l().lstore(epilogueN);}
        applyVectorEpilogue(code,carriers,specialization,ir,geometry,epilogueN,accumulator);
        int outputBoundary=specialization.carrierPattern().size()-1;
        carriers.vectorStore(type,specialization.carrierPattern().get(outputBoundary),outputBoundary,outputAddress,
                accumulator,intAddress);
        if(intAddress){code.iinc(n,lanes);
            code.iload(rightVectorBase).iload(rightVectorStep).iadd().istore(rightVectorBase);
            code.iload(outputVectorBase).iload(outputVectorStep).iadd().istore(outputVectorBase);
        } else code.lload(n).loadConstant((long)lanes).ladd().lstore(n);
        code.branch(Opcode.GOTO,vectors).labelBinding(vectorDone);
        if(intAddress)emitScalarTailHeap(code,carriers,specialization,ir,geometry,n,leftRowBase,
                rightBase,outputRowBase,kExtent,leftKStride,rightKStride);
        else emitScalarTail(code,carriers,specialization,ir,geometry,n,leftRowBase,rightBase,
                outputRowBase);
        code.lload(ordinal).loadConstant(1L).ladd().lstore(ordinal);
        incrementIndex(code,m,intAddress);
        var nextBatch=code.newLabel();
        if(intAddress)code.iload(m).aload(geometry).loadConstant(4).laload().l2i()
                .branch(Opcode.IF_ICMPGE,nextBatch);
        else code.lload(m).aload(geometry).loadConstant(4).laload().lcmp()
                .branch(Opcode.IFGE,nextBatch);
        code.branch(Opcode.GOTO,rows).labelBinding(nextBatch);
        zeroIndex(code,m,intAddress);code.lload(batch).loadConstant(1L).ladd().lstore(batch)
                .branch(Opcode.GOTO,prepareBatch).labelBinding(done);
    }

    private static void emitTiledVector(CodeBuilder code,
            CpuKernelSpecialization specialization, CpuMatmulIr ir) {
        DataType type = ir.resultType();
        int lanes = specialization.vectorSpeciesBitSize() / type.bitWidth();
        if (lanes <= 1 || ir.leftType() != type || ir.rightType() != type)
            throw new IllegalArgumentException("MATMUL vector-tile facts disagree");
        ClassDesc vector = vectorClass(type);
        boolean intAddress=heapArrays(specialization);
        int geometry=specialization.carrierPattern().size(),start=geometry+1,end=geometry+3;
        var carriers=new CpuCarrierEmitter(code);carriers.prepareVectorSpecies(type);
        int ordinal=code.allocateLocal(TypeKind.LONG),remaining=code.allocateLocal(TypeKind.LONG);
        int nTiles=code.allocateLocal(TypeKind.LONG),mTiles=code.allocateLocal(TypeKind.LONG);
        int mt=code.allocateLocal(TypeKind.LONG),nt=code.allocateLocal(TypeKind.LONG);
        int m0=code.allocateLocal(TypeKind.LONG),mNext=code.allocateLocal(TypeKind.LONG);
        int n0=code.allocateLocal(TypeKind.LONG),nSecond=code.allocateLocal(TypeKind.LONG);
        int tileEnd=code.allocateLocal(TypeKind.LONG),tailStart=code.allocateLocal(TypeKind.LONG);
        int batch=code.allocateLocal(TypeKind.LONG),batchRemaining=code.allocateLocal(TypeKind.LONG);
        int axis=code.allocateLocal(TypeKind.INT);
        int coordinate=code.allocateLocal(TypeKind.LONG),row1=code.allocateLocal(TypeKind.INT);
        int full0=code.allocateLocal(TypeKind.INT),full1=code.allocateLocal(TypeKind.INT);
        int leftBatchBase=code.allocateLocal(TypeKind.LONG);
        int outputBatchBase=code.allocateLocal(TypeKind.LONG);
        int leftBase=code.allocateLocal(TypeKind.LONG),left1Base=code.allocateLocal(TypeKind.LONG);
        int rightBase=code.allocateLocal(TypeKind.LONG),outputBase=code.allocateLocal(TypeKind.LONG);
        int output1Base=code.allocateLocal(TypeKind.LONG);
        int a0=code.allocateLocal(intAddress?TypeKind.INT:TypeKind.LONG);
        int a1=code.allocateLocal(intAddress?TypeKind.INT:TypeKind.LONG);
        int b0=code.allocateLocal(intAddress?TypeKind.INT:TypeKind.LONG);
        int b1=code.allocateLocal(intAddress?TypeKind.INT:TypeKind.LONG);
        int k=code.allocateLocal(intAddress?TypeKind.INT:TypeKind.LONG);
        int kExtent=code.allocateLocal(intAddress?TypeKind.INT:TypeKind.LONG);
        int leftKStride=code.allocateLocal(intAddress?TypeKind.INT:TypeKind.LONG);
        int rightKStride=code.allocateLocal(intAddress?TypeKind.INT:TypeKind.LONG);
        TypeKind scalarKind=type==DataType.FLOAT64?TypeKind.DOUBLE:type==DataType.INT64
                ?TypeKind.LONG:type==DataType.INT32?TypeKind.INT:TypeKind.FLOAT;
        int left0=code.allocateLocal(scalarKind),left1=code.allocateLocal(scalarKind);
        int right0=code.allocateLocal(TypeKind.REFERENCE),right1=code.allocateLocal(TypeKind.REFERENCE);
        int[] sums={code.allocateLocal(TypeKind.REFERENCE),code.allocateLocal(TypeKind.REFERENCE),
                code.allocateLocal(TypeKind.REFERENCE),code.allocateLocal(TypeKind.REFERENCE)};
        geometry(code,geometry,5);storeIndex(code,kExtent,intAddress);
        geometry(code,geometry,8);storeIndex(code,leftKStride,intAddress);
        geometry(code,geometry,9);storeIndex(code,rightKStride,intAddress);
        geometry(code,geometry,6).loadConstant((long)(2*lanes-1)).ladd()
                .loadConstant((long)(2*lanes)).ldiv().lstore(nTiles);
        geometry(code,geometry,4).loadConstant(1L).ladd().loadConstant(2L).ldiv().lstore(mTiles);
        code.lload(start).lstore(ordinal).lload(start).lstore(remaining);
        code.lload(remaining).lload(nTiles).lrem().lstore(nt);
        code.lload(remaining).lload(nTiles).ldiv().lstore(remaining);
        code.lload(remaining).lload(mTiles).lrem().lstore(mt);
        code.lload(remaining).lload(mTiles).ldiv().lstore(batch);
        var prepareBatch=code.newLabel();var tiles=code.newLabel();var done=code.newLabel();
        code.labelBinding(prepareBatch);
        resolveBatchBases(code,geometry,batch,batchRemaining,axis,coordinate,
                leftBatchBase,rightBase,outputBatchBase);
        code.branch(Opcode.GOTO,tiles).labelBinding(tiles).lload(ordinal).lload(end).lcmp()
                .branch(Opcode.IFGE,done);
        code.lload(mt).loadConstant(2L).lmul().lstore(m0);
        code.lload(m0).loadConstant(1L).ladd().lstore(mNext);
        code.lload(nt).loadConstant((long)(2*lanes)).lmul().lstore(n0);
        code.lload(n0).loadConstant((long)lanes).ladd().lstore(nSecond);
        lessThan(code,mNext,geometry,4,row1);
        lessThanOrEqual(code,nSecond,geometry,6,full0);
        int twoLanesEnd=code.allocateLocal(TypeKind.LONG);
        code.lload(n0).loadConstant((long)(2*lanes)).ladd().lstore(twoLanesEnd);
        lessThanOrEqual(code,twoLanesEnd,geometry,6,full1);
        code.lload(twoLanesEnd).lstore(tileEnd);
        var tileEndOk=code.newLabel();code.lload(tileEnd);geometry(code,geometry,6).lcmp()
                .branch(Opcode.IFLE,tileEndOk);geometry(code,geometry,6).lstore(tileEnd)
                .labelBinding(tileEndOk);
        code.lload(n0).lstore(tailStart);
        var noFirstTail=code.newLabel();code.iload(full0).branch(Opcode.IFEQ,noFirstTail);
        code.lload(nSecond).lstore(tailStart);code.labelBinding(noFirstTail);
        var noSecondTail=code.newLabel();code.iload(full1).branch(Opcode.IFEQ,noSecondTail);
        code.lload(twoLanesEnd).lstore(tailStart);code.labelBinding(noSecondTail);
        code.lload(leftBatchBase).lload(m0);geometry(code,geometry,7).lmul().ladd().lstore(leftBase);
        code.lload(leftBase);geometry(code,geometry,7).ladd().lstore(left1Base);
        code.lload(outputBatchBase).lload(m0);geometry(code,geometry,11).lmul().ladd()
                .lstore(outputBase);
        code.lload(outputBase);geometry(code,geometry,11).ladd().lstore(output1Base);
        code.lload(leftBase);storeIndex(code,a0,intAddress);
        code.lload(left1Base);storeIndex(code,a1,intAddress);
        code.lload(rightBase).lload(n0);geometry(code,geometry,10).lmul().ladd();
        storeIndex(code,b0,intAddress);
        code.lload(rightBase).lload(nSecond);geometry(code,geometry,10).lmul().ladd();
        storeIndex(code,b1,intAddress);
        code.getstatic(vector,"SPECIES_PREFERRED",ClassDesc.of("jdk.incubator.vector.VectorSpecies"));
        code.invokestatic(vector,"zero",MethodTypeDesc.of(vector,
                ClassDesc.of("jdk.incubator.vector.VectorSpecies"))).astore(sums[0]);
        for(int index=1;index<sums.length;index++)code.aload(sums[0]).astore(sums[index]);
        zero(code,type,left1);code.aload(sums[0]).astore(right0);code.aload(sums[0]).astore(right1);
        emitBoundContraction(code,k,intAddress ? TypeKind.INT : TypeKind.LONG,
                () -> code.loadLocal(intAddress ? TypeKind.INT : TypeKind.LONG,kExtent),directProduct -> {
        loadSameTypeScalar(code,carriers,specialization,type,0,a0,left0,intAddress);
        var skipLeft1=code.newLabel();code.iload(row1).branch(Opcode.IFEQ,skipLeft1);
        loadSameTypeScalar(code,carriers,specialization,type,0,a1,left1,intAddress);
        code.labelBinding(skipLeft1);
        var skipFirst=code.newLabel();code.iload(full0).branch(Opcode.IFEQ,skipFirst);
        carriers.vectorLoad(type,specialization.carrierPattern().get(1),1,b0,false,intAddress);
        code.astore(right0);vectorScalarAccumulate(code,vector,type,sums[0],right0,left0,directProduct);
        var skipFirstRow1=code.newLabel();code.iload(row1).branch(Opcode.IFEQ,skipFirstRow1);
        vectorScalarAccumulate(code,vector,type,sums[2],right0,left1,directProduct);code.labelBinding(skipFirstRow1)
                .labelBinding(skipFirst);
        var skipSecond=code.newLabel();code.iload(full1).branch(Opcode.IFEQ,skipSecond);
        carriers.vectorLoad(type,specialization.carrierPattern().get(1),1,b1,false,intAddress);
        code.astore(right1);vectorScalarAccumulate(code,vector,type,sums[1],right1,left0,directProduct);
        var skipSecondRow1=code.newLabel();code.iload(row1).branch(Opcode.IFEQ,skipSecondRow1);
        vectorScalarAccumulate(code,vector,type,sums[3],right1,left1,directProduct);code.labelBinding(skipSecondRow1)
                .labelBinding(skipSecond);
        },() -> {
        addIndex(code,a0,leftKStride,intAddress);addIndex(code,a1,leftKStride,intAddress);
        addIndex(code,b0,rightKStride,intAddress);addIndex(code,b1,rightKStride,intAddress);
        });
        var noFirstStore=code.newLabel();code.iload(full0).branch(Opcode.IFEQ,noFirstStore);
        storeVectorTile(code,carriers,specialization,ir,geometry,outputBase,n0,sums[0],intAddress);
        var noFirstRow1Store=code.newLabel();code.iload(row1).branch(Opcode.IFEQ,noFirstRow1Store);
        storeVectorTile(code,carriers,specialization,ir,geometry,output1Base,n0,sums[2],intAddress);
        code.labelBinding(noFirstRow1Store).labelBinding(noFirstStore);
        var noSecondStore=code.newLabel();code.iload(full1).branch(Opcode.IFEQ,noSecondStore);
        storeVectorTile(code,carriers,specialization,ir,geometry,outputBase,nSecond,sums[1],intAddress);
        var noSecondRow1Store=code.newLabel();code.iload(row1).branch(Opcode.IFEQ,noSecondRow1Store);
        storeVectorTile(code,carriers,specialization,ir,geometry,output1Base,nSecond,sums[3],intAddress);
        code.labelBinding(noSecondRow1Store).labelBinding(noSecondStore);
        int row1Tail=code.allocateLocal(TypeKind.LONG);code.lload(tailStart).lstore(row1Tail);
        emitScalarRange(code,carriers,specialization,ir,geometry,tailStart,tileEnd,leftBase,
                rightBase,outputBase);
        var noRow1Tail=code.newLabel();code.iload(row1).branch(Opcode.IFEQ,noRow1Tail);
        emitScalarRange(code,carriers,specialization,ir,geometry,row1Tail,tileEnd,left1Base,
                rightBase,output1Base);code.labelBinding(noRow1Tail);
        code.lload(ordinal).loadConstant(1L).ladd().lstore(ordinal);
        code.lload(nt).loadConstant(1L).ladd().lstore(nt);
        var advanceM=code.newLabel();code.lload(nt).lload(nTiles).lcmp()
                .branch(Opcode.IFGE,advanceM).branch(Opcode.GOTO,tiles).labelBinding(advanceM)
                .loadConstant(0L).lstore(nt).lload(mt).loadConstant(1L).ladd().lstore(mt);
        var advanceBatch=code.newLabel();code.lload(mt).lload(mTiles).lcmp()
                .branch(Opcode.IFGE,advanceBatch).branch(Opcode.GOTO,tiles)
                .labelBinding(advanceBatch).loadConstant(0L).lstore(mt).lload(batch)
                .loadConstant(1L).ladd().lstore(batch).branch(Opcode.GOTO,prepareBatch)
                .labelBinding(done);
    }

    private static void lessThan(CodeBuilder code,int coordinate,int geometry,int extentIndex,
            int target) {
        var yes=code.newLabel();var done=code.newLabel();code.lload(coordinate);
        geometry(code,geometry,extentIndex).lcmp().branch(Opcode.IFLT,yes).loadConstant(0)
                .istore(target).branch(Opcode.GOTO,done).labelBinding(yes).loadConstant(1)
                .istore(target).labelBinding(done);
    }

    private static void lessThanOrEqual(CodeBuilder code,int coordinate,int geometry,
            int extentIndex,int target) {
        var yes=code.newLabel();var done=code.newLabel();code.lload(coordinate);
        geometry(code,geometry,extentIndex).lcmp().branch(Opcode.IFLE,yes).loadConstant(0)
                .istore(target).branch(Opcode.GOTO,done).labelBinding(yes).loadConstant(1)
                .istore(target).labelBinding(done);
    }

    private static void vectorAccumulate(CodeBuilder code,ClassDesc vector,int left,int right,
            int sum) {
        code.aload(left).aload(right).invokevirtual(vector,"mul",
                MethodTypeDesc.of(vector,ClassDesc.of("jdk.incubator.vector.Vector")))
                .aload(sum).invokevirtual(vector,"add",
                        MethodTypeDesc.of(vector,ClassDesc.of("jdk.incubator.vector.Vector")))
                .astore(sum);
    }

    private static void vectorScalarAccumulate(CodeBuilder code,ClassDesc vector,DataType type,
            int sum,int right,int scalar,boolean directProduct) {
        if (!directProduct) code.aload(sum);
        code.aload(right);
        loadPrimitive(code,type,scalar);
        code.invokevirtual(vector,"mul",MethodTypeDesc.of(vector,primitive(type)));
        if (!directProduct) code.invokevirtual(vector,"add",MethodTypeDesc.of(vector,
                ClassDesc.of("jdk.incubator.vector.Vector")));
        code.astore(sum);
    }

    private static void storeVectorTile(CodeBuilder code,CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization,CpuMatmulIr ir,int geometry,int outputBase,
            int n,int value,boolean intAddress) {
        DataType type=ir.resultType();
        int address=code.allocateLocal(intAddress?TypeKind.INT:TypeKind.LONG);
        code.lload(outputBase).lload(n);geometry(code,geometry,12).lmul().ladd();
        storeIndex(code,address,intAddress);
        applyVectorEpilogue(code,carriers,specialization,ir,geometry,n,value);
        int outputBoundary=specialization.carrierPattern().size()-1;
        carriers.vectorStore(type,specialization.carrierPattern().get(outputBoundary),
                outputBoundary,address,value,intAddress);
    }

    /**
     * Emits the common ordinary-vector suffix, preserving ordered ADD before typed TANH.
     * @param code generation-only method builder
     * @param carriers generation-only typed carrier emitter
     * @param specialization exact entry carrier signature
     * @param ir validated NONE/TANH vector matrix identity
     * @param geometry local containing packed invocation geometry
     * @param n local containing the first logical output column; long when bias reads use it,
     *     otherwise unused by the suffix
     * @param value reference local containing the completed vector sum, replaced by the suffix
     */
    private static void applyVectorEpilogue(CodeBuilder code,CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization,CpuMatmulIr ir,int geometry,int n,int value) {
        applyVectorBias(code,carriers,specialization,ir,geometry,n,value);
        if(ir.epilogue().terminal()==CpuMatmulIr.Epilogue.Terminal.TANH) {
            ClassDesc vector=vectorClass(ir.resultType());
            code.aload(value).invokestatic(ClassDesc.of(CpuVectorMath.class.getName()),"tanh",
                    MethodTypeDesc.of(vector,vector)).astore(value);
        }
    }

    private static void applyVectorBias(CodeBuilder code,CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization,CpuMatmulIr ir,int geometry,int n,int value) {
        if(!ir.epilogue().hasBias())return;
        DataType type=ir.resultType();ClassDesc vector=vectorClass(type);
        int address=code.allocateLocal(TypeKind.LONG),bias=code.allocateLocal(TypeKind.REFERENCE);
        code.aload(geometry).loadConstant(14).laload().lload(n).aload(geometry)
                .loadConstant(15).laload().lmul().ladd().lstore(address);
        carriers.vectorLoad(type,specialization.carrierPattern().get(2),2,address,false,false);
        code.astore(bias);
        if(ir.epilogue().addInputOrder()==CpuMatmulIr.Epilogue.AddInputOrder.MATMUL_LEFT)
            code.aload(value).aload(bias);
        else code.aload(bias).aload(value);
        code.invokevirtual(vector,"add",
                MethodTypeDesc.of(vector,ClassDesc.of("jdk.incubator.vector.Vector"))).astore(value);
    }

    private static void emitScalarTailHeap(CodeBuilder code,CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization,CpuMatmulIr ir,int geometry,int n,
            int leftBase,int rightBase,int outputBase,int kExtent,int leftKStride,
            int rightKStride) {
        DataType type=ir.resultType();
        int limit=code.allocateLocal(TypeKind.INT),rightNStride=code.allocateLocal(TypeKind.INT);
        int outputNStride=code.allocateLocal(TypeKind.INT);
        geometry(code,geometry,6).l2i().istore(limit);
        geometry(code,geometry,10).l2i().istore(rightNStride);
        geometry(code,geometry,12).l2i().istore(outputNStride);
        int leftAddress=code.allocateLocal(TypeKind.INT),rightAddress=code.allocateLocal(TypeKind.INT);
        int outputAddress=code.allocateLocal(TypeKind.INT),k=code.allocateLocal(TypeKind.INT);
        TypeKind kind=type==DataType.FLOAT64?TypeKind.DOUBLE:type==DataType.INT64?TypeKind.LONG
                :type==DataType.INT32?TypeKind.INT:TypeKind.FLOAT;
        int sum=code.allocateLocal(kind),left=code.allocateLocal(kind),right=code.allocateLocal(kind);
        var loop=code.newLabel();var done=code.newLabel();
        code.labelBinding(loop).iload(n).iload(limit).branch(Opcode.IF_ICMPGE,done);
        code.lload(leftBase).l2i().istore(leftAddress);
        code.lload(rightBase).l2i().iload(n).iload(rightNStride).imul().iadd()
                .istore(rightAddress);
        code.lload(outputBase).l2i().iload(n).iload(outputNStride).imul().iadd()
                .istore(outputAddress);
        zero(code,type,sum);
        emitCursorContraction(code,type,TypeKind.INT,k,sum,left,right,() -> code.iload(kExtent),
                () -> load(code,carriers,specialization,type,type,0,leftAddress,left,left,true),
                () -> load(code,carriers,specialization,type,type,1,rightAddress,right,right,true),
                () -> {
                    code.iload(leftAddress).iload(leftKStride).iadd().istore(leftAddress);
                    code.iload(rightAddress).iload(rightKStride).iadd().istore(rightAddress);
                },Runnable::run);
        int epilogueN=code.allocateLocal(TypeKind.LONG);code.iload(n).i2l().lstore(epilogueN);
        applyScalarEpilogue(code,carriers,specialization,ir,geometry,epilogueN,sum);
        int outputBoundary=specialization.carrierPattern().size()-1;
        if(type==DataType.INT32||type==DataType.INT64)
            storeIntegral(code,carriers,specialization,type,outputBoundary,outputAddress,sum,true);
        else carriers.storeFrozen(type,specialization.carrierPattern().get(outputBoundary),
                outputBoundary,outputAddress,sum,true);
        code.iinc(n,1).branch(Opcode.GOTO,loop).labelBinding(done);
    }

    private static void emitScalarTail(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, CpuMatmulIr ir, int geometry, int n,
            int leftBase, int rightBase, int outputBase) {
        int limit=code.allocateLocal(TypeKind.LONG);geometry(code,geometry,6).lstore(limit);
        emitScalarRange(code,carriers,specialization,ir,geometry,n,limit,leftBase,rightBase,
                outputBase);
    }

    private static void emitScalarRange(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, CpuMatmulIr ir, int geometry, int n, int limit,
            int leftBase, int rightBase, int outputBase) {
        DataType type=ir.resultType();
        int leftAddress=code.allocateLocal(TypeKind.LONG),rightAddress=code.allocateLocal(TypeKind.LONG);
        int outputAddress=code.allocateLocal(TypeKind.LONG),k=code.allocateLocal(TypeKind.LONG);
        TypeKind kind=type==DataType.FLOAT64?TypeKind.DOUBLE:type==DataType.INT64?TypeKind.LONG
                :type==DataType.INT32?TypeKind.INT:TypeKind.FLOAT;
        int sum=code.allocateLocal(kind),left=code.allocateLocal(kind),right=code.allocateLocal(kind);
        var loop=code.newLabel();var done=code.newLabel();
        code.labelBinding(loop).lload(n).lload(limit).lcmp().branch(Opcode.IFGE,done);
        code.lload(leftBase).lstore(leftAddress);
        code.lload(rightBase).lload(n);geometry(code,geometry,10).lmul().ladd().lstore(rightAddress);
        code.lload(outputBase).lload(n);geometry(code,geometry,12).lmul().ladd().lstore(outputAddress);
        zero(code,type,sum);
        emitCursorContraction(code,type,TypeKind.LONG,k,sum,left,right,() -> geometry(code,geometry,5),
                () -> load(code,carriers,specialization,type,type,0,leftAddress,left,left),
                () -> load(code,carriers,specialization,type,type,1,rightAddress,right,right),
                () -> {
                    code.lload(leftAddress);geometry(code,geometry,8).ladd().lstore(leftAddress);
                    code.lload(rightAddress);geometry(code,geometry,9).ladd().lstore(rightAddress);
                },Runnable::run);
        applyScalarEpilogue(code,carriers,specialization,ir,geometry,n,sum);
        int outputBoundary=specialization.carrierPattern().size()-1;
        if (type == DataType.INT32 || type == DataType.INT64)
            storeIntegral(code,carriers,specialization,type,outputBoundary,outputAddress,sum);
        else carriers.storeFrozen(type,specialization.carrierPattern().get(outputBoundary),
                outputBoundary,outputAddress,sum,false);
        code.lload(n).loadConstant(1L).ladd().lstore(n).branch(Opcode.GOTO,loop).labelBinding(done);
    }

    private static void applyScalarEpilogue(CodeBuilder code,CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization,CpuMatmulIr ir,int geometry,int n,int value) {
        applyScalarEpilogue(code,carriers,specialization,ir,geometry,n,value,-1,-1);
    }

    private static void applyScalarEpilogue(CodeBuilder code,CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization,CpuMatmulIr ir,int geometry,int n,int value,
            int retainedBiasBase,int retainedBiasStride) {
        DataType type=ir.resultType();TypeKind kind=type==DataType.FLOAT64?TypeKind.DOUBLE:TypeKind.FLOAT;
        if(ir.epilogue().hasBias()) {
            int biasAddress=code.allocateLocal(TypeKind.LONG),bias=code.allocateLocal(kind);
            if(retainedBiasBase>=0)code.lload(retainedBiasBase).lload(n)
                    .lload(retainedBiasStride).lmul().ladd().lstore(biasAddress);
            else code.aload(geometry).loadConstant(14).laload().lload(n).aload(geometry)
                    .loadConstant(15).laload().lmul().ladd().lstore(biasAddress);
            load(code,carriers,specialization,type,type,2,biasAddress,bias,bias);
            if(ir.epilogue().addInputOrder()==CpuMatmulIr.Epilogue.AddInputOrder.MATMUL_LEFT) {
                if(type==DataType.FLOAT64)code.dload(value).dload(bias).dadd().dstore(value);
                else code.fload(value).fload(bias).fadd().fstore(value);
            } else {
                if(type==DataType.FLOAT64)code.dload(bias).dload(value).dadd().dstore(value);
                else code.fload(bias).fload(value).fadd().fstore(value);
            }
        }
        if(!ir.epilogue().hasTerminal())return;
        CpuPointwiseOpcode opcode=switch(ir.epilogue().terminal()) {
            case RELU -> CpuPointwiseOpcode.RELU;case SIGMOID -> CpuPointwiseOpcode.SIGMOID;
            case TANH -> CpuPointwiseOpcode.TANH;case GELU -> CpuPointwiseOpcode.GELU_EXACT;
            case GELU_TANH_APPROXIMATION -> CpuPointwiseOpcode.GELU_TANH_APPROXIMATION;
            case SILU -> CpuPointwiseOpcode.SILU;case CLAMP -> CpuPointwiseOpcode.SCALAR_CLAMP;
            case NONE -> throw new AssertionError();
        };
        CpuKernelIr.Instruction instruction;
        if(opcode==CpuPointwiseOpcode.SCALAR_CLAMP) {
            var range=ir.epilogue().clampRange();
            instruction=new CpuKernelIr.Instruction(opcode,List.of(0),1,
                    new CpuKernelIr.ClampImmediate(immediate(range.minValue(),type),
                            immediate(range.maxValue(),type)));
        } else instruction=new CpuKernelIr.Instruction(opcode,List.of(0),1,null,null);
        var values=List.of(new CpuKernelIr.Value(0,type,CpuKernelIr.Value.Kind.INPUT,
                        ir.outputAccess()),
                new CpuKernelIr.Value(1,type,CpuKernelIr.Value.Kind.VIRTUAL,ir.outputAccess()));
        var scalarIr=new CpuKernelIr(values,List.of(instruction),
                new CpuKernelIr.Loop("start","end"),List.of(),"matmul-epilogue");
        new CpuScalarEmitter(code).emit(scalarIr,instruction,new int[]{value,value});
    }

    private static CpuKernelIr.ScalarImmediate immediate(
            io.github.pho001.synaptik.model.datatype.ScalarValue value,DataType type) {
        if(value.dataType()!=type)throw new IllegalArgumentException("MATMUL clamp type disagrees");
        long bits=type==DataType.FLOAT64?Double.doubleToRawLongBits(value.float64Value())
                :Float.floatToRawIntBits(value.float32Value())&0xffff_ffffL;
        return new CpuKernelIr.ScalarImmediate(type,bits);
    }

    private static ClassDesc vectorClass(DataType type) {
        return ClassDesc.of("jdk.incubator.vector." + switch(type) {
            case FLOAT32 -> "FloatVector"; case FLOAT64 -> "DoubleVector";
            case INT32 -> "IntVector"; case INT64 -> "LongVector";
            default -> throw new IllegalArgumentException("unsupported MATMUL vector type");
        });
    }

    private static void resolveBatchBases(CodeBuilder code,int geometry,int batch,
            int batchRemaining,int axis,int coordinate,int leftBase,int rightBase,int outputBase) {
        geometry(code,geometry,0).lstore(leftBase);
        geometry(code,geometry,1).lstore(rightBase);
        geometry(code,geometry,2).lstore(outputBase);
        code.lload(batch).lstore(batchRemaining);
        geometry(code,geometry,13).l2i().loadConstant(1).isub().istore(axis);
        var batches=code.newLabel();var batchesDone=code.newLabel();
        code.labelBinding(batches).iload(axis).branch(Opcode.IFLT,batchesDone);
        code.lload(batchRemaining).aload(geometry).iload(axis).loadConstant(18).iadd().laload()
                .lrem().lstore(coordinate);
        code.lload(batchRemaining).aload(geometry).iload(axis).loadConstant(18).iadd().laload()
                .ldiv().lstore(batchRemaining);
        addBatch(code,geometry,axis,coordinate,leftBase,18);
        addBatch(code,geometry,axis,coordinate,rightBase,18,2);
        addBatch(code,geometry,axis,coordinate,outputBase,18,3);
        code.iinc(axis,-1).branch(Opcode.GOTO,batches).labelBinding(batchesDone);
    }

    private static void addBatch(CodeBuilder code, int geometry, int axis, int coordinate,
            int base, int arrays) { addBatch(code, geometry, axis, coordinate, base, arrays, 1); }
    private static void addBatch(CodeBuilder code, int geometry, int axis, int coordinate,
            int base, int arrays, int strideArray) {
        code.lload(base).lload(coordinate).aload(geometry).iload(axis);
        geometry(code, geometry, 13).l2i().loadConstant(strideArray).imul().iadd()
                .loadConstant(arrays).iadd().laload().lmul().ladd().lstore(base);
    }
    private static CodeBuilder geometry(CodeBuilder code, int geometry, int index) {
        return code.aload(geometry).loadConstant(index).laload();
    }
    private static boolean heapArrays(CpuKernelSpecialization specialization) {
        return specialization.carrierPattern().stream()
                .noneMatch(CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT::equals);
    }
    private static void storeIndex(CodeBuilder code,int target,boolean intAddress) {
        if(intAddress)code.l2i().istore(target);else code.lstore(target);
    }
    private static void loadIndexAsLong(CodeBuilder code,int source,boolean intAddress) {
        if(intAddress)code.iload(source).i2l();else code.lload(source);
    }
    private static void addIndex(CodeBuilder code,int address,int stride,boolean intAddress) {
        if(intAddress)code.iload(address).iload(stride).iadd().istore(address);
        else code.lload(address).lload(stride).ladd().lstore(address);
    }
    private static void zeroIndex(CodeBuilder code,int index,boolean intAddress) {
        if(intAddress)code.loadConstant(0).istore(index);else code.loadConstant(0L).lstore(index);
    }
    private static void incrementIndex(CodeBuilder code,int index,boolean intAddress) {
        if(intAddress)code.iinc(index,1);
        else code.lload(index).loadConstant(1L).ladd().lstore(index);
    }
    private static void branchIndexGreaterOrEqual(CodeBuilder code,int index,int limit,
            boolean intAddress,java.lang.classfile.Label target) {
        if(intAddress)code.iload(index).iload(limit).branch(Opcode.IF_ICMPGE,target);
        else code.lload(index).lload(limit).lcmp().branch(Opcode.IFGE,target);
    }
    private static TypeKind representedKind(DataType type) {
        return type == DataType.FLOAT64 ? TypeKind.DOUBLE : type == DataType.FLOAT32
                ? TypeKind.FLOAT : type == DataType.INT64 ? TypeKind.LONG : TypeKind.INT;
    }
    private static ClassDesc primitive(DataType type) {
        return switch(type) {
            case FLOAT64 -> ConstantDescs.CD_double;case FLOAT32 -> ConstantDescs.CD_float;
            case INT32 -> ConstantDescs.CD_int;case INT64 -> ConstantDescs.CD_long;
            default -> throw new IllegalArgumentException("unsupported MATMUL vector scalar type");
        };
    }
    private static void loadPrimitive(CodeBuilder code,DataType type,int local) {
        switch(type) {
            case FLOAT64 -> code.dload(local);case FLOAT32 -> code.fload(local);
            case INT32 -> code.iload(local);case INT64 -> code.lload(local);
            default -> throw new IllegalArgumentException("unsupported MATMUL vector scalar type");
        }
    }
    private static void loadSameTypeScalar(CodeBuilder code,CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization,DataType type,int boundary,int address,
            int target,boolean intAddress) {
        if(type==DataType.FLOAT32||type==DataType.FLOAT64)
            carriers.loadFrozen(type,specialization.carrierPattern().get(boundary),boundary,
                    address,intAddress);
        else carriers.load(type,specialization.carrierPattern().get(boundary),boundary,address,
                intAddress);
        switch(type) {
            case FLOAT64 -> code.dstore(target);case FLOAT32 -> code.fstore(target);
            case INT32 -> code.istore(target);case INT64 -> code.lstore(target);
            default -> throw new IllegalArgumentException("unsupported MATMUL vector scalar type");
        }
    }
    private static void zero(CodeBuilder code, DataType type, int target) {
        switch (type) {
            case FLOAT64 -> code.loadConstant(0.0).dstore(target);
            case FLOAT32, BFLOAT16, FLOAT16, FLOAT8_E4M3, FLOAT8_E5M2 -> code.loadConstant(0.0f).fstore(target);
            case INT32 -> code.loadConstant(0).istore(target);
            case INT64 -> code.loadConstant(0L).lstore(target);
            case BOOL -> throw new AssertionError();
        }
    }
    private static void load(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType source, DataType result,
            int boundary, int address, int represented, int value) {
        load(code,carriers,specialization,source,result,boundary,address,represented,value,false);
    }
    private static void load(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType source, DataType result,
            int boundary, int address, int represented, int value,boolean intAddress) {
        if (source == DataType.INT32 || source == DataType.INT64)
            carriers.load(source, specialization.carrierPattern().get(boundary), boundary,
                    address, intAddress);
        else carriers.loadFrozen(source, specialization.carrierPattern().get(boundary), boundary,
                    address, intAddress);
        switch (source) {
            case FLOAT64 -> code.dstore(represented);
            case FLOAT32 -> code.fstore(represented);
            case BFLOAT16, FLOAT16, FLOAT8_E4M3, FLOAT8_E5M2, INT32 -> code.istore(represented);
            case INT64 -> code.lstore(represented);
            case BOOL -> throw new AssertionError();
        }
        if (result == DataType.FLOAT64) {
            if (source == DataType.FLOAT64) {
                if(represented!=value)code.dload(represented).dstore(value);
            } else if (source == DataType.FLOAT32) code.fload(represented).f2d().dstore(value);
            else {
                CpuCastEmitter.emitLowToFloat(code, source, represented);
                code.f2d().dstore(value);
            }
        } else if (result.isFloating()) {
            if (source == DataType.FLOAT32) {
                if(represented!=value)code.fload(represented).fstore(value);
            } else {
                CpuCastEmitter.emitLowToFloat(code, source, represented);
                code.fstore(value);
            }
        } else if (result == DataType.INT64) {
            if (source == DataType.INT64) {
                if(represented!=value)code.lload(represented).lstore(value);
            }
            else code.iload(represented).i2l().lstore(value);
        } else if(represented!=value)code.iload(represented).istore(value);
    }
    /**
     * Composes the ordinary scalar contraction primitive over explicitly bound typed locals.
     * Scan generation reuses this site without entry-slot, batch or geometry-array assumptions.
     * Low represented loads have already decoded into FLOAT working locals; they must
     * not be reloaded with their integral storage kind. MUL and ADD remain separate sites.
     * @param code generation-only builder
     * @param type admitted accumulator dtype
     * @param sum accumulator local, updated with one represented product
     * @param left represented left value local
     * @param right represented right value local
     * @param directProduct whether to assign the sole logical product without an identity ADD
     * @throws IllegalArgumentException if the accumulator dtype is unsupported
     */
    static void accumulate(CodeBuilder code, DataType type, int sum, int left, int right,
            boolean directProduct) {
        var workingKind = switch (type) {
            case FLOAT64 -> TypeKind.DOUBLE;
            case FLOAT32, BFLOAT16, FLOAT16, FLOAT8_E4M3, FLOAT8_E5M2 -> TypeKind.FLOAT;
            case INT32 -> TypeKind.INT;
            case INT64 -> TypeKind.LONG;
            case BOOL -> throw new AssertionError();
        };
        emitContributorStep(code,type,sum,directProduct,() -> code.loadLocal(workingKind,left),
                () -> code.loadLocal(workingKind,right));
    }

    private static void storeIntegral(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType type, int boundary,
            int address, int value) {
        storeIntegral(code,carriers,specialization,type,boundary,address,value,false);
    }
    private static void storeIntegral(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType type, int boundary,
            int address, int value,boolean intAddress) {
        carriers.store(type, specialization.carrierPattern().get(boundary), boundary,
                address, value, intAddress);
    }
}

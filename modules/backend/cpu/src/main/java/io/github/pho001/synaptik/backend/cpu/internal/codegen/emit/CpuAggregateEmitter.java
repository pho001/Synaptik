package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.constant.ConstantDescs;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Emits typed ordinary aggregate folds directly into generated CPU entries.
 *
 * <p>The static body reduces complete output cells only. It traverses every selected domain in
 * logical input row-major order. Numerical floating rows update an exact primitive-limb state and
 * finalize at working and storage boundaries; extrema retain first-NaN and signed-zero
 * selection, with a distinct unseen
 * E4M3 seed because that format has no represented infinity. Extrema and Boolean combination
 * are emitted directly, so the generated per-element body calls no Synaptik runtime helper. The
 * body allocates no per-cell or per-element object. Dense heap-array
 * reductions use one typed integer-address fold; other forms embed a typed long-address fallback
 * that decodes output and selected-domain coordinates without runtime type, kind, form, or
 * carrier dispatch. A cold-proved rank-two canonical-BOOL ANY form with one zero-stride selected
 * axis uses direct primitive cell and domain loops while retaining every logical selected-domain
 * visit and canonicalizing the final byte; unproved geometry retains the typed long-address
 * body. Two runtime-guarded numerical forms cover the frozen FLOAT32 axis-one MEAN with domain
 * {@code 2048} and BFLOAT16 axes-zero-and-two PROD geometries. The MEAN form keeps its exact
 * five-limb state in primitive locals, writes the identical final state to the assigned run-owned
 * workspace slice, and implements division by {@code 2048 == 2^11} through equivalent exact
 * extraction and ties-to-even rounding. The PROD form preserves the exact-product working state
 * transitions while replacing general coordinate reconstruction with primitive geometry cursors.
 * Both forms preserve logical factor order and arbitrary legal output-cell subranges; every
 * unproved geometry retains the typed long-address body. A third completely guarded form covers
 * the frozen BFLOAT16 {@code [64,64,64]} axes-zero-and-two MIN geometry with kept output
 * {@code [1,64,1]}, a {@link MemorySegment} input, and an offset, two-strided {@code short[]}
 * output. Complete geometry, carrier, ordered-range, start-address, and sentinel guards precede
 * every specialized write. For each arbitrary legal complete-output-cell subrange, the body
 * initializes from the first represented factor, then visits axis 0 outside axis 2 in canonical
 * selected-domain order using the direct clean-Java-equivalent primitive traversal. It retains
 * the first raw BFLOAT16 NaN, chooses negative zero when both zero signs occur, otherwise keeps
 * the smaller represented value, and writes the selected raw 16-bit result once. This guarded
 * body uses no workspace, materialization, partial/combine state, or selected-domain buffer;
 * every failed proof enters the unchanged typed general-long body. Binding-aware SUM-to-Shape
 * adds a direct represented-bit copy when no axis reduces, a typed right-aligned cursor fallback,
 * and a guarded primitive nested loop for the dense {@code [64,128,256] -> [128,1]} evidence
 * geometry. Floating reductions continue to delegate emitted state transitions and final
 * conversion solely to {@link CpuExactSumEmitter}.</p>
 * <p>The package-local represented SUM entry composes sequential FLOAT32/FLOAT64 sites under
 * caller-bound traversal/activity and stores. Controlled ordinary entries use the complete
 * canonical coordinate traversal for static and symbolic geometry, guarding ROW outputs before
 * eager reads and proved ROW contributors before operand access. A controlled terminal ROW SUM
 * reuses represented producing sites; other ordinary folds retain their exact-state algorithm.
 * Occurrence presence remains independent of the number of visited contributors, including an
 * empty all-valid reduction. Uncontrolled entry selection and exact-state guards are unchanged.</p>
 */
public final class CpuAggregateEmitter {
    private static final DataType[] TYPES = DataType.values();
    /** Creates a stateless typed-body emitter. */
    public CpuAggregateEmitter() { }

    /**
     * Composes the current ordinary exact-state algorithm with caller-bound typed sites.
     * Unlike represented Scan SUM, each contribution enters exact limbs and finish rounds the
     * result before the sink can execute an epilogue. All actions run only during generation.
     * @param code nonnull method builder
     * @param aggregate existing static SUM/MEAN IR with domain-sized limbs
     * @param scratch reference slot containing run-private exact state
     * @param geometry reference slot containing the unchanged aggregate header
     * @param contribution represented contributor local
     * @param result represented completed-result local
     * @param outputCells generation-only complete-output-cell traversal
     * @param contributors generation-only canonical selected-domain traversal
     * @param producer generation-only action initializing the typed contribution local
     * @param sink generation-only consumer of the completed rounded result
     * @throws NullPointerException if a required argument is null
     * @throws IllegalArgumentException if the represented dtype or exact-state width is unsupported
     */
    static void emitExactSum(CodeBuilder code, CpuAggregateIr aggregate, int scratch, int geometry,
            int contribution, int result, java.util.function.Consumer<Runnable> outputCells,
            java.util.function.Consumer<Runnable> contributors, Runnable producer, Runnable sink) {
        var exact = new CpuExactSumEmitter(code,aggregate.dataType(),aggregate.kind() == CpuAggregateIr.Kind.MEAN,
                scratch,geometry,aggregate.stateLimbCount());
        outputCells.accept(() -> {
            exact.emitReset();
            contributors.accept(() -> { producer.run(); exact.emitFactor(contribution); });
            exact.emitFinish(result);
            sink.run();
        });
    }

    /**
     * Initializes a represented Scan sum, retaining the historical omitted singleton reset.
     * @param code nonnull generation builder
     * @param dataType homogeneous FLOAT32 or FLOAT64
     * @param accumulator allocated typed accumulator slot
     * @param directSingleton whether geometry proves raw movement and no reset is required
     * @throws IllegalArgumentException if the represented dtype is unsupported
     */
    static void emitRepresentedInit(CodeBuilder code, DataType dataType, int accumulator,
            boolean directSingleton) {
        if (dataType != DataType.FLOAT32 && dataType != DataType.FLOAT64)
            throw new IllegalArgumentException("represented SUM requires FLOAT32 or FLOAT64");
        if (!directSingleton) {
            if (dataType == DataType.FLOAT64) code.loadConstant(0d); else code.loadConstant(0f);
            store(code,dataType,accumulator);
        }
    }

    /**
     * Emits one represented contributor or raw singleton movement under caller-owned activity.
     * Singleton is a geometry fact, never inferred from how many rows are active.
     * @param code nonnull generation builder
     * @param dataType homogeneous FLOAT32 or FLOAT64
     * @param accumulator initialized typed slot unless raw movement supplies its first value
     * @param rawSingleton whether this producing site requires raw movement instead of ADD
     * @param load generation action leaving one represented typed contributor on the stack
     */
    static void emitRepresentedContributorStep(CodeBuilder code, DataType dataType, int accumulator,
            boolean rawSingleton, Runnable load) {
        if (!rawSingleton) load(code,dataType,accumulator);
        load.run();
        if (!rawSingleton) { if (dataType == DataType.FLOAT64) code.dadd(); else code.fadd(); }
        store(code,dataType,accumulator);
    }

    /**
     * Publishes a completed represented sum without another arithmetic site.
     * @param code nonnull generation builder owning the result
     * @param dataType homogeneous represented dtype
     * @param accumulator initialized completed-result slot
     * @param sink generation-only result consumer receiving the exact accumulator slot
     */
    static void emitRepresentedFinish(CodeBuilder code, DataType dataType, int accumulator,
            java.util.function.IntConsumer sink) {
        sink.accept(accumulator);
    }

    /**
     * Initializes a represented feature-vector SUM, independently of ordinary exact SUM state.
     * @param code builder owning the accumulator
     * @param type homogeneous FLOAT32 or FLOAT64
     * @param accumulator reference local for the matching preferred species
     * @param directSingleton whether geometry requires raw assignment without reset
     * @throws IllegalArgumentException if the dtype is unsupported
     */
    static void emitRepresentedVectorInit(CodeBuilder code, DataType type, int accumulator,
            boolean directSingleton) {
        if (type != DataType.FLOAT32 && type != DataType.FLOAT64)
            throw new IllegalArgumentException("represented vector SUM requires FLOAT32 or FLOAT64");
        if (!directSingleton) CpuMatmulEmitter.emitVectorInit(code,type,accumulator);
    }

    /**
     * Adds one represented vector or assigns a geometry-singleton raw vector. Unused lanes
     * never enter another cell's reduction; the caller owns activity and tail masks.
     * @param code builder owning the accumulator
     * @param type homogeneous FLOAT32 or FLOAT64
     * @param accumulator matching reference local, initialized unless assigned raw here
     * @param rawSingleton geometry-proved raw movement, never inferred from active count
     * @param load generation-only loader leaving a matching preferred-species vector
     * @throws IllegalArgumentException if the dtype is unsupported
     */
    static void emitRepresentedVectorContributorStep(CodeBuilder code, DataType type,
            int accumulator, boolean rawSingleton, Runnable load) {
        if (type != DataType.FLOAT32 && type != DataType.FLOAT64)
            throw new IllegalArgumentException("represented vector SUM requires FLOAT32 or FLOAT64");
        if (!rawSingleton) code.aload(accumulator);
        load.run();
        if (!rawSingleton) {
            var vector = java.lang.constant.ClassDesc.of("jdk.incubator.vector."
                    +(type == DataType.FLOAT64 ? "DoubleVector" : "FloatVector"));
            code.invokevirtual(vector,"add",java.lang.constant.MethodTypeDesc.of(vector,
                    java.lang.constant.ClassDesc.of("jdk.incubator.vector.Vector")));
        }
        code.astore(accumulator);
    }

    /**
     * Passes a completed represented vector to its sink without another arithmetic site.
     * @param code builder owning the local
     * @param type homogeneous FLOAT32 or FLOAT64
     * @param accumulator completed preferred-species reference local
     * @param sink generation-only consumer of that exact local
     */
    static void emitRepresentedVectorFinish(CodeBuilder code, DataType type, int accumulator,
            java.util.function.IntConsumer sink) {
        sink.accept(accumulator);
    }

    /**
     * Emits represented sequential SUM, separately from the ordinary exact-state entry.
     * Traversal adapters supply active output/contributor scopes, never SUM arithmetic.
     * Producer and result sinks retain their represented sites; the tail sees only the
     * completed result. A raw singleton copies without addition; the direct-singleton proof
     * additionally omits reset, while legacy raw paths retain their unused reset.
     * All callbacks execute at generation time and introduce no runtime object or dispatch.
     * @param code non-null method builder
     * @param type admitted FLOAT32 or FLOAT64 represented type
     * @param accumulator reserved typed accumulator local, including legacy unused reservations
     * @param rawSingleton whether contributors are raw movement rather than typed addition
     * @param directSingleton whether the existing proof also omits accumulator reset
     * @param outputCells non-null output traversal invoking its action inside activity guards
     * @param contributors non-null contributor traversal invoking its action after coordinate/activity binding
     * @param preload non-null legacy invariant operand emission, or no-op for length paths
     * @param producer non-null contributor's producing-site emission with fresh availability
     * @param loadContributor non-null operand load emission leaving one typed value on stack
     * @param resultSink non-null completed-result emission receiving the accumulator local
     * @param epilogue non-null same-output-cell tail emission
     * @throws IllegalArgumentException if the represented type is not FLOAT32 or FLOAT64
     */
    static void emitRepresentedSum(CodeBuilder code, DataType type, int accumulator,
            boolean rawSingleton, boolean directSingleton,
            java.util.function.Consumer<Runnable> outputCells,
            java.util.function.Consumer<Runnable> contributors, Runnable preload, Runnable producer,
            Runnable loadContributor, java.util.function.IntConsumer resultSink, Runnable epilogue) {
        if (type != DataType.FLOAT32 && type != DataType.FLOAT64)
            throw new IllegalArgumentException("represented SUM requires FLOAT32 or FLOAT64");
        TypeKind kind = type == DataType.FLOAT64 ? TypeKind.DOUBLE : TypeKind.FLOAT;
        outputCells.accept(() -> {
            emitRepresentedInit(code,type,accumulator,directSingleton);
            preload.run();
            contributors.accept(() -> {
                producer.run();
                emitRepresentedContributorStep(code,type,accumulator,rawSingleton,loadContributor);
            });
            emitRepresentedFinish(code,type,accumulator,resultSink);
            epilogue.run();
        });
    }

    /**
     * Emits a typed aggregate body whose hot loops contain no runtime semantic dispatch.
     *
     * @param code non-null Class-File method builder mutated with the generated fold
     * @param specialization non-null two-boundary carrier/type specialization, scratch-bearing
     *     exactly for floating numerical aggregates; the emitted entry retains typed general
     *     fallback behavior when a guarded geometry does not match
     * @param ir non-null canonical aggregate IR supplying kind, form, axes, rank, and access facts
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if the specialization has another boundary shape
     */
    public void emit(CodeBuilder code, CpuKernelSpecialization specialization, CpuKernelIr ir) {
        if (specialization.carrierPattern().size() != 2)
            throw new IllegalArgumentException("aggregate requires two boundaries");
        String identity = ir.familyIdentity();
        DataType type = specialization.boundaryDataTypes().getFirst();
        int kind = switch (CpuAggregateIr.encodedKind(ir)) {
            case MIN -> 0; case MAX -> 1; case ALL -> 2; case ANY -> 3;
            case SUM -> 4; case MEAN -> 5; case PROD -> 6;
        };
        boolean sumToShapeCopy = !specialization.ordinaryBatch() && identity.contains(":SUM_TO_SHAPE:axes=[]:");
        boolean exactFloating = !sumToShapeCopy && kind >= 4 && (type == DataType.FLOAT64
                || type == DataType.FLOAT32 || type == DataType.BFLOAT16
                || type == DataType.FLOAT16 || type == DataType.FLOAT8_E4M3 || type == DataType.FLOAT8_E5M2);
        int exactLimbs = exactFloating ? identityNumber(identity, ":limbs=") : 0;
        if (specialization.scratchParameter() != exactFloating)
            throw new IllegalArgumentException("aggregate scratch shape disagrees with numerical kind");
        int geometrySlot = exactFloating ? 3 : 2;
        int startSlot = exactFloating ? 4 : 3;
        int endSlot = exactFloating ? 6 : 5;
        int inRank = ir.values().getFirst().accessPlan().iterationRank();
        int outRank = ir.values().getLast().accessPlan().iterationRank();
        if (CpuCarrierEmitter.controlFacts().isPresent()) {
            // The existing general traversal covers every specialized geometry as well as
            // symbolic packs, without relying on physical addresses to recover logical rows.
            emitGeneral(code, specialization, type, kind, identity, inRank, outRank,
                    exactFloating, exactLimbs, geometrySlot, startSlot, endSlot);
            return;
        }
        if (specialization.ordinaryBatch()) {
            emitOrdinaryBatch(code, specialization, ir, kind, exactLimbs, inRank, outRank,
                    geometrySlot, startSlot, endSlot);
            return;
        }
        boolean fullDenseArrays = identity.contains(":FULL:")
                && specialization.carrierPattern().stream().noneMatch(
                    access -> access == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)
                && ir.values().getFirst().accessPlan().regime() == CpuAccessPlan.Regime.DENSE_LINEAR;
        boolean denseTrailingArrays = !identity.contains(":FULL:")
                && specialization.carrierPattern().stream().noneMatch(
                    access -> access == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)
                && ir.values().getFirst().accessPlan().regime() == CpuAccessPlan.Regime.DENSE_LINEAR
                && ir.values().getLast().accessPlan().regime() == CpuAccessPlan.Regime.DENSE_LINEAR
                && selectedSuffix(identity, inRank);
        boolean zeroStrideAny = kind == 3 && type == DataType.BOOL && inRank == 2 && outRank == 1
                && java.util.Arrays.equals(selectedAxes(identity, inRank),
                        new boolean[]{false, true})
                && specialization.carrierPattern().getFirst()
                        == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT
                && specialization.carrierPattern().getLast()
                        == CpuKernelSpecialization.CarrierAccess.BYTE_ARRAY;
        boolean boundedMean = kind == 5 && type == DataType.FLOAT32 && inRank == 2 && outRank == 1
                && exactLimbs == 5 && identity.contains(":SINGLE_AXIS:axes=[1]:keep=false:")
                && identity.contains(":domain=2048:limbs=5:slice=48:")
                && specialization.carrierPattern().getFirst()
                        == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT
                && specialization.carrierPattern().getLast()
                        == CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY;
        boolean boundedProduct = kind == 6 && type == DataType.BFLOAT16
                && inRank == 3 && outRank == 3 && exactLimbs == 6
                && identity.contains(":MULTI_AXIS:axes=[0, 2]:keep=true:")
                && identity.contains(":domain=16:limbs=6:slice=72:")
                && specialization.carrierPattern().getFirst()
                        == CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY
                && specialization.carrierPattern().getLast()
                        == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT;
        boolean boundedMinimum = kind == 0 && type == DataType.BFLOAT16
                && inRank == 3 && outRank == 3
                && identity.contains(":MULTI_AXIS:axes=[0, 2]:keep=true:")
                && identity.contains(":domain=4096:limbs=0:slice=0:")
                && specialization.carrierPattern().getFirst()
                        == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT
                && specialization.carrierPattern().getLast()
                        == CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY;
        boolean boundedSumToShape = identity.contains(":SUM_TO_SHAPE:axes=[0, 2]:keep=false:")
                && identity.contains(":source=[64, 128, 256]:target=[128, 1]:domain=16384:")
                && inRank == 3 && outRank == 2
                && specialization.carrierPattern().stream().noneMatch(
                    access -> access == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT);
        if (sumToShapeCopy) emitSumToShapeCopy(code, specialization, type, ir, inRank, outRank,
                geometrySlot, startSlot, endSlot);
        else if (boundedSumToShape) emitBoundedSumToShape(code, specialization, type,
                exactFloating, exactLimbs, geometrySlot, startSlot, endSlot, 19, 27);
        else if (zeroStrideAny) emitZeroStrideAnyOrGeneral(code, specialization, identity, inRank,
                outRank, geometrySlot, startSlot, endSlot);
        else if (boundedMean) emitBoundedMeanOrGeneral(code, specialization, identity,
                geometrySlot, startSlot, endSlot);
        else if (boundedProduct) emitBoundedProductOrGeneral(code, specialization, identity,
                geometrySlot, startSlot, endSlot);
        else if (boundedMinimum) emitBoundedMinimumOrGeneral(code, specialization, identity,
                geometrySlot, startSlot, endSlot);
        else if (fullDenseArrays) emitFullDense(code, specialization, type, kind, inRank,
                exactFloating, exactLimbs, geometrySlot, startSlot, endSlot);
        else if (denseTrailingArrays) emitDenseCells(code, specialization, type, kind, inRank,
                outRank, exactFloating, exactLimbs, geometrySlot, startSlot, endSlot);
        else emitGeneral(code, specialization, type, kind, identity, inRank, outRank,
                exactFloating, exactLimbs, geometrySlot, startSlot, endSlot);
    }

    /**
     * Emits bounded canonical/scalar typed symbolic folds with entry-only geometry and exact
     * five-limb F32 or 34-limb F64 state; no float intermediate is used for F64.
     * Rank-at-most-two loops retain their established form; rank-three Scan outer closure uses
     * direct selected-axis cursors, not per-contributor mapping or a new aggregate algorithm.
     *
     * @param code non-null generation builder
     * @param specialization non-null prepared FLOAT32/FLOAT64 scalar carrier selection
     * @param ir non-null prepared axis membership and access categories
     * @param kind SUM or MEAN wire
     * @param limbs exact typed accumulator count, five or 34
     * @param inRank prepared input rank, at most three
     * @param outRank prepared output rank
     * @param geometrySlot existing long-array geometry parameter slot
     * @param startSlot inclusive output-cell range parameter slot
     * @param endSlot exclusive output-cell range parameter slot
     * @throws IllegalArgumentException if carrier, reduction kind or limb count is unsupported
     */
    private static void emitOrdinaryBatch(CodeBuilder code, CpuKernelSpecialization specialization,
            CpuKernelIr ir, int kind, int limbs, int inRank, int outRank,
            int geometrySlot, int startSlot, int endSlot) {
        if (inRank == 3) {
            emitRankThreeBatch(code,specialization,ir,kind,limbs,outRank,geometrySlot,startSlot,endSlot);
            return;
        }
        DataType type = specialization.boundaryDataTypes().getFirst();
        TypeKind numeric = localKind(type);
        boolean scalar = ir.values().getFirst().accessPlan().regime() == CpuAccessPlan.Regime.SCALAR_ALL_ZERO;
        if (!scalar && ir.values().getFirst().accessPlan().regime() != CpuAccessPlan.Regime.DENSE_LINEAR
                || limbs != (type == DataType.FLOAT64 ? 34 : 5) || kind != 4 && kind != 5) throw new IllegalArgumentException("unsupported batch aggregate loop");
        boolean sumTo = ir.familyIdentity().contains(":SUM_TO_SHAPE:");
        boolean[] axes = selectedAxes(ir.familyIdentity(), inRank);
        boolean columns = inRank == 2 && axes[0] && !axes[1];
        int inputLayout = 11 + 2 * inRank + outRank, outputLayout = inputLayout + 2 + 2 * inRank;
        int cell = code.allocateLocal(TypeKind.INT), end = code.allocateLocal(TypeKind.INT);
        int inputBase = code.allocateLocal(TypeKind.INT), outputBase = code.allocateLocal(TypeKind.INT);
        int domain = code.allocateLocal(TypeKind.INT), step = code.allocateLocal(TypeKind.INT);
        geometry(code, geometrySlot, inputLayout + 1).l2i().istore(inputBase);
        geometry(code, geometrySlot, outputLayout + 1).l2i().istore(outputBase);
        geometry(code, geometrySlot, 7).l2i().istore(domain);
        if (scalar) code.loadConstant(0);
        else if (columns) geometry(code, geometrySlot, inputLayout + 3).l2i();
        else code.loadConstant(1);
        code.istore(step).lload(startSlot).l2i().istore(cell).lload(endSlot).l2i().istore(end);
        var done = code.newLabel();
        if (sumTo) {
            var reduce = code.newLabel();
            code.iload(domain).loadConstant(1).branch(Opcode.IF_ICMPNE, reduce);
            var copies = code.newLabel(); code.labelBinding(copies);
            code.iload(cell).iload(end).branch(Opcode.IF_ICMPGE, done);
            int source = code.allocateLocal(TypeKind.INT), target = code.allocateLocal(TypeKind.INT);
            int value = code.allocateLocal(numeric);
            code.iload(inputBase); if (!scalar) code.iload(cell).iadd(); code.istore(source);
            code.iload(outputBase).iload(cell).iadd().istore(target);
            var carriers = new CpuCarrierEmitter(code);
            carriers.load(type, specialization.carrierPattern().getFirst(), 0, source, true);
            code.storeLocal(numeric,value); carriers.store(type, specialization.carrierPattern().getLast(), 1, target, value, true);
            code.iinc(cell, 1).branch(Opcode.GOTO, copies).labelBinding(reduce);
        }
        int address = code.allocateLocal(TypeKind.INT), target = code.allocateLocal(TypeKind.INT);
        int term = code.allocateLocal(TypeKind.INT), value = code.allocateLocal(numeric);
        int accumulator = code.allocateLocal(numeric);
        CpuExactSumEmitter exact = new CpuExactSumEmitter(code, type, kind == 5, 2, geometrySlot, limbs);
        var cells = code.newLabel(); code.labelBinding(cells);
        code.iload(cell).iload(end).branch(Opcode.IF_ICMPGE, done);
        exact.emitReset();
        code.iload(inputBase);
        if (!scalar) { code.iload(cell); if (!columns) code.iload(domain).imul(); code.iadd(); }
        code.istore(address).loadConstant(0).istore(term);
        var terms = code.newLabel(); var finish = code.newLabel(); code.labelBinding(terms);
        code.iload(term).iload(domain).branch(Opcode.IF_ICMPGE, finish);
        var carriers = new CpuCarrierEmitter(code);
        carriers.load(type, specialization.carrierPattern().getFirst(), 0, address, true);
        code.storeLocal(numeric,value); exact.emitFactor(value);
        code.iload(address).iload(step).iadd().istore(address).iinc(term, 1).branch(Opcode.GOTO, terms).labelBinding(finish);
        exact.emitFinish(accumulator);
        code.iload(outputBase).iload(cell).iadd().istore(target);
        carriers.store(type, specialization.carrierPattern().getLast(), 1, target, accumulator, true);
        code.iinc(cell, 1).branch(Opcode.GOTO, cells).labelBinding(done);
    }

    /**
     * Emits the bounded three-axis outer Scan-AD folds with direct canonical cursors. Prefix
     * reductions advance by retained width; a middle-axis reduction carries its feature cursor
     * across rows without per-cell division; selecting time and features uses two contributor
     * loops. Tail/full reductions retain one contiguous contributor loop. All geometry loads
     * precede cells, exact state retains five F32 or 34 F64 limbs, and SUM-to-Shape domain one
     * is raw copy.
     * Rank-at-most-two emission never enters this method and remains byte-identical.
     * @param code generation builder
     * @param specialization prepared scalar FLOAT32/FLOAT64 carrier selection
     * @param ir prepared axis membership and access category
     * @param kind SUM or MEAN wire
     * @param limbs fixed typed state count, five for F32 or 34 for F64
     * @param outRank prepared result rank
     * @param geometrySlot existing packed-geometry parameter
     * @param startSlot inclusive output-cell range parameter
     * @param endSlot exclusive output-cell range parameter
     * @throws IllegalArgumentException if the admitted carrier or exact-state contract disagrees
     */
    private static void emitRankThreeBatch(CodeBuilder code,CpuKernelSpecialization specialization,
            CpuKernelIr ir,int kind,int limbs,int outRank,int geometrySlot,int startSlot,int endSlot) {
        DataType type = specialization.boundaryDataTypes().getFirst();
        TypeKind numeric = localKind(type);
        boolean scalar = ir.values().getFirst().accessPlan().regime() == CpuAccessPlan.Regime.SCALAR_ALL_ZERO;
        if (!scalar && ir.values().getFirst().accessPlan().regime() != CpuAccessPlan.Regime.DENSE_LINEAR
                || limbs != (type == DataType.FLOAT64 ? 34 : 5) || kind != 4 && kind != 5) throw new IllegalArgumentException("unsupported rank-three batch aggregate");
        boolean[] axes = selectedAxes(ir.familyIdentity(),3);
        boolean middle = !axes[0] && axes[1] && !axes[2];
        boolean separated = axes[0] && !axes[1] && axes[2];
        boolean prefix = axes[0] && !axes[2];
        int inputLayout = 17+outRank, outputLayout = inputLayout+8;
        int inputBase = code.allocateLocal(TypeKind.INT), outputBase = code.allocateLocal(TypeKind.INT);
        int domain = code.allocateLocal(TypeKind.INT), rows = code.allocateLocal(TypeKind.INT), features = code.allocateLocal(TypeKind.INT);
        int time = code.allocateLocal(TypeKind.INT), frame = code.allocateLocal(TypeKind.INT), outputCount = code.allocateLocal(TypeKind.INT);
        int cell = code.allocateLocal(TypeKind.INT), end = code.allocateLocal(TypeKind.INT);
        geometry(code,geometrySlot,inputLayout+1).l2i().istore(inputBase);
        geometry(code,geometrySlot,outputLayout+1).l2i().istore(outputBase);
        geometry(code,geometrySlot,7).l2i().istore(domain);
        geometry(code,geometrySlot,inputLayout+2).l2i().istore(time);
        geometry(code,geometrySlot,inputLayout+3).l2i().istore(rows);
        geometry(code,geometrySlot,inputLayout+4).l2i().istore(features);
        geometry(code,geometrySlot,6).l2i().istore(outputCount);
        code.iload(rows).iload(features).imul().istore(frame);
        code.lload(startSlot).l2i().istore(cell).lload(endSlot).l2i().istore(end);
        var done = code.newLabel(); code.iload(cell).iload(end).branch(Opcode.IF_ICMPGE,done);
        var carriers = new CpuCarrierEmitter(code);
        if (ir.familyIdentity().contains(":SUM_TO_SHAPE:")) {
            var reduce = code.newLabel(); var copies = code.newLabel();
            code.iload(domain).loadConstant(1).branch(Opcode.IF_ICMPNE,reduce).labelBinding(copies);
            code.iload(cell).iload(end).branch(Opcode.IF_ICMPGE,done);
            int source = code.allocateLocal(TypeKind.INT), target = code.allocateLocal(TypeKind.INT), value = code.allocateLocal(numeric);
            code.iload(inputBase); if (!scalar) code.iload(cell).iadd(); code.istore(source);
            code.iload(outputBase).iload(cell).iadd().istore(target);
            carriers.load(type,specialization.carrierPattern().getFirst(),0,source,true); code.storeLocal(numeric,value);
            carriers.store(type,specialization.carrierPattern().getLast(),1,target,value,true);
            code.iinc(cell,1).branch(Opcode.GOTO,copies).labelBinding(reduce);
        }
        int sourceBase = code.allocateLocal(TypeKind.INT), step = code.allocateLocal(TypeKind.INT), column = code.allocateLocal(TypeKind.INT);
        code.iload(inputBase);
        if (!scalar) {
            if (middle) code.iload(cell).iload(features).idiv().iload(frame).imul()
                    .iload(cell).iload(features).irem().iadd().iadd();
            else if (separated) code.iload(cell).iload(features).imul().iadd();
            else if (prefix) code.iload(cell).iadd();
            else code.iload(cell).iload(domain).imul().iadd();
        }
        code.istore(sourceBase);
        if (middle && !scalar) code.iload(cell).iload(features).irem(); else code.loadConstant(0);
        code.istore(column);
        if (scalar) code.loadConstant(0);
        else if (middle) code.iload(features);
        else if (prefix) code.iload(outputCount);
        else code.loadConstant(1);
        code.istore(step);
        int address = code.allocateLocal(TypeKind.INT), term = code.allocateLocal(TypeKind.INT);
        int value = code.allocateLocal(numeric), accumulator = code.allocateLocal(numeric), target = code.allocateLocal(TypeKind.INT);
        var exact = new CpuExactSumEmitter(code,type,kind == 5,2,geometrySlot,limbs);
        var cells = code.newLabel(); code.labelBinding(cells);
        code.iload(cell).iload(end).branch(Opcode.IF_ICMPGE,done); exact.emitReset();
        code.iload(sourceBase).istore(address).loadConstant(0).istore(term);
        if (separated) {
            int outer = code.allocateLocal(TypeKind.INT), rowBase = code.allocateLocal(TypeKind.INT);
            var outerLoop = code.newLabel(); var outerDone = code.newLabel(); var innerLoop = code.newLabel(); var innerDone = code.newLabel();
            code.loadConstant(0).istore(outer).iload(sourceBase).istore(rowBase).labelBinding(outerLoop);
            code.iload(outer).iload(time).branch(Opcode.IF_ICMPGE,outerDone);
            code.iload(rowBase).istore(address).loadConstant(0).istore(term).labelBinding(innerLoop);
            code.iload(term).iload(features).branch(Opcode.IF_ICMPGE,innerDone);
            carriers.load(type,specialization.carrierPattern().getFirst(),0,address,true); code.storeLocal(numeric,value); exact.emitFactor(value);
            if (!scalar) code.iinc(address,1);
            code.iinc(term,1).branch(Opcode.GOTO,innerLoop).labelBinding(innerDone);
            if (!scalar) code.iload(rowBase).iload(frame).iadd().istore(rowBase);
            code.iinc(outer,1).branch(Opcode.GOTO,outerLoop).labelBinding(outerDone);
        } else {
            var terms = code.newLabel(); var finish = code.newLabel(); code.labelBinding(terms);
            code.iload(term).iload(domain).branch(Opcode.IF_ICMPGE,finish);
            carriers.load(type,specialization.carrierPattern().getFirst(),0,address,true); code.storeLocal(numeric,value); exact.emitFactor(value);
            code.iload(address).iload(step).iadd().istore(address).iinc(term,1).branch(Opcode.GOTO,terms).labelBinding(finish);
        }
        exact.emitFinish(accumulator); code.iload(outputBase).iload(cell).iadd().istore(target);
        carriers.store(type,specialization.carrierPattern().getLast(),1,target,accumulator,true);
        if (!scalar) {
            if (middle) {
                code.iinc(sourceBase,1).iinc(column,1); var next = code.newLabel();
                code.iload(column).iload(features).branch(Opcode.IF_ICMPLT,next);
                code.iload(sourceBase).iload(frame).iload(features).isub().iadd().istore(sourceBase).loadConstant(0).istore(column).labelBinding(next);
            } else if (separated) code.iload(sourceBase).iload(features).iadd().istore(sourceBase);
            else if (prefix) code.iinc(sourceBase,1);
            else code.iload(sourceBase).iload(domain).iadd().istore(sourceBase);
        }
        code.iinc(cell,1).branch(Opcode.GOTO,cells).labelBinding(done);
    }

    private static void emitBoundedSumToShape(CodeBuilder code,
            CpuKernelSpecialization specialization, DataType type, boolean exactFloating,
            int exactLimbs, int geometrySlot, int startSlot, int endSlot,
            int inputLayout, int outputLayout) {
        if (!exactFloating) {
            emitBoundedIntegralSumToShape(code, specialization, type, geometrySlot,
                    startSlot, endSlot, inputLayout, outputLayout);
            return;
        }
        int cell = code.allocateLocal(TypeKind.INT), end = code.allocateLocal(TypeKind.INT);
        int inputBase = code.allocateLocal(TypeKind.INT), outputBase = code.allocateLocal(TypeKind.INT);
        int rowBase = code.allocateLocal(TypeKind.INT);
        int outer = code.allocateLocal(TypeKind.INT), inner = code.allocateLocal(TypeKind.INT);
        int accumulator = code.allocateLocal(localKind(type));
        int value = code.allocateLocal(localKind(type));
        code.lload(startSlot).l2i().istore(cell).lload(endSlot).l2i().istore(end);
        geometry(code, geometrySlot, inputLayout + 1).l2i().istore(inputBase);
        geometry(code, geometrySlot, outputLayout + 1).l2i().istore(outputBase);
        CpuExactSumEmitter exact = exactFloating
                ? new CpuExactSumEmitter(code, type, false, 2, geometrySlot, exactLimbs) : null;
        var carriers = new CpuCarrierEmitter(code);
        var cells = code.newLabel(); var done = code.newLabel();
        var outerLoop = code.newLabel(); var innerLoop = code.newLabel();
        code.labelBinding(cells).iload(cell).iload(end).branch(Opcode.IF_ICMPGE, done);
        if (exact != null) {
            emitFloatingZero(code, type); store(code, type, accumulator); exact.emitReset();
        } else {
            emitIdentity(code, type, 4); store(code, type, accumulator);
        }
        code.loadConstant(0).istore(outer).labelBinding(outerLoop);
        code.iload(inputBase).iload(outer).loadConstant(32_768).imul()
                .iload(cell).loadConstant(256).imul().iadd().iadd().istore(rowBase);
        code.loadConstant(0).istore(inner).labelBinding(innerLoop);
        emitDenseArrayLoad(code, type, rowBase, inner, value);
        if (exact != null) exact.emitFactor(value);
        else emitApply(code, type, 4, accumulator, value);
        code.iinc(inner, 1).iload(inner).loadConstant(256)
                .branch(Opcode.IF_ICMPLT, innerLoop);
        code.iinc(outer, 1).iload(outer).loadConstant(64)
                .branch(Opcode.IF_ICMPLT, outerLoop);
        if (exact != null) exact.emitFinish(accumulator);
        int output = code.allocateLocal(TypeKind.INT);
        code.iload(outputBase).iload(cell).iadd().istore(output);
        carriers.store(type, specialization.carrierPattern().getLast(), 1, output, accumulator,
                true);
        code.iinc(cell, 1).branch(Opcode.GOTO, cells).labelBinding(done);
    }

    private static void emitBoundedIntegralSumToShape(CodeBuilder code,
            CpuKernelSpecialization specialization, DataType type, int geometrySlot,
            int startSlot, int endSlot, int inputLayout, int outputLayout) {
        int cell = code.allocateLocal(TypeKind.INT), end = code.allocateLocal(TypeKind.INT);
        int inputBase = code.allocateLocal(TypeKind.INT), outputBase = code.allocateLocal(TypeKind.INT);
        int rowBase = code.allocateLocal(TypeKind.INT), outer = code.allocateLocal(TypeKind.INT);
        int inner = code.allocateLocal(TypeKind.INT);
        int accumulator = code.allocateLocal(localKind(type));
        code.lload(startSlot).l2i().istore(cell).lload(endSlot).l2i().istore(end);
        geometry(code, geometrySlot, inputLayout + 1).l2i().istore(inputBase);
        geometry(code, geometrySlot, outputLayout + 1).l2i().istore(outputBase);
        var cells = code.newLabel(); var done = code.newLabel();
        var outerLoop = code.newLabel(); var outerDone = code.newLabel();
        var innerLoop = code.newLabel(); var innerDone = code.newLabel();
        code.labelBinding(cells).iload(cell).iload(end).branch(Opcode.IF_ICMPGE, done);
        emitIdentity(code, type, 4); store(code, type, accumulator);
        code.loadConstant(0).istore(outer).labelBinding(outerLoop);
        code.iload(outer).loadConstant(64).branch(Opcode.IF_ICMPGE, outerDone);
        code.iload(inputBase).iload(outer).loadConstant(32_768).imul()
                .iload(cell).loadConstant(256).imul().iadd().iadd().istore(rowBase);
        code.loadConstant(0).istore(inner).labelBinding(innerLoop);
        code.iload(inner).loadConstant(256).branch(Opcode.IF_ICMPGE, innerDone);
        emitDenseArrayAdd(code, type, rowBase, inner, 0, accumulator);
        code.iinc(inner, 1).branch(Opcode.GOTO, innerLoop).labelBinding(innerDone);
        code.iinc(outer, 1).branch(Opcode.GOTO, outerLoop).labelBinding(outerDone);
        int output = code.allocateLocal(TypeKind.INT);
        code.iload(outputBase).iload(cell).iadd().istore(output);
        new CpuCarrierEmitter(code).store(type, specialization.carrierPattern().getLast(), 1,
                output, accumulator, true);
        code.iinc(cell, 1).branch(Opcode.GOTO, cells).labelBinding(done);
    }

    private static void emitDenseArrayAdd(CodeBuilder code, DataType type, int base, int index,
            int displacement, int accumulator) {
        load(code, type, accumulator);
        code.aload(0).iload(base).iload(index).iadd();
        if (displacement != 0) code.loadConstant(displacement).iadd();
        if (type == DataType.INT32) code.iaload().iadd().istore(accumulator);
        else if (type == DataType.INT64) code.laload().ladd().lstore(accumulator);
        else throw new IllegalArgumentException("integral SUM-to-Shape type required");
    }

    private static void emitDenseArrayLoad(CodeBuilder code, DataType type, int base, int index,
            int value) {
        emitDenseArrayLoad(code, type, base, index, 0, value);
    }

    private static void emitDenseArrayLoad(CodeBuilder code, DataType type, int base, int index,
            int displacement, int value) {
        code.aload(0).iload(base).iload(index).iadd();
        if (displacement != 0) code.loadConstant(displacement).iadd();
        switch (type) {
            case FLOAT64 -> code.daload().dstore(value);
            case FLOAT32 -> code.faload().fstore(value);
            case BFLOAT16, FLOAT16 -> code.saload().istore(value);
            case FLOAT8_E4M3, FLOAT8_E5M2 -> code.baload().istore(value);
            case INT32 -> code.iaload().istore(value);
            case INT64 -> code.laload().lstore(value);
            default -> throw new IllegalArgumentException("unsupported SUM-to-Shape data type");
        }
    }

    private static void emitSumToShapeCopy(CodeBuilder code,
            CpuKernelSpecialization specialization, DataType type, CpuKernelIr ir,
            int inRank, int outRank, int geometrySlot, int startSlot, int endSlot) {
        boolean denseArrays = specialization.carrierPattern().stream().noneMatch(
                access -> access == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)
                && ir.values().stream().allMatch(value -> value.accessPlan().regime()
                    == CpuAccessPlan.Regime.DENSE_LINEAR);
        int inputCoordinates = 11 + inRank;
        int outputCoordinates = inputCoordinates + inRank;
        int inputLayout = outputCoordinates + outRank;
        int outputLayout = inputLayout + 2 + 2 * inRank;
        int cell = code.allocateLocal(denseArrays ? TypeKind.INT : TypeKind.LONG);
        if (denseArrays) code.lload(startSlot).l2i().istore(cell);
        else code.lload(startSlot).lstore(cell);
        var done = code.newLabel(); var cells = code.newLabel();
        if (denseArrays) code.iload(cell).lload(endSlot).l2i().branch(Opcode.IF_ICMPGE, done);
        else code.lload(cell).lload(endSlot).lcmp().branch(Opcode.IFGE, done);
        code.labelBinding(cells);
        int inputAddress;
        int outputAddress;
        if (denseArrays) {
            inputAddress = code.allocateLocal(TypeKind.INT);
            outputAddress = code.allocateLocal(TypeKind.INT);
            code.aload(geometrySlot).loadConstant(inputLayout + 1).laload().l2i()
                    .iload(cell).iadd().istore(inputAddress);
            code.aload(geometrySlot).loadConstant(outputLayout + 1).laload().l2i()
                    .iload(cell).iadd().istore(outputAddress);
        } else {
            decode(code, geometrySlot, cell, outputCoordinates, outputLayout, outRank);
            int leading = inRank - outRank;
            for (int axis = 0; axis < inRank; axis++) {
                code.aload(geometrySlot).loadConstant(inputCoordinates + axis);
                if (axis < leading) code.loadConstant(0L);
                else code.aload(geometrySlot).loadConstant(outputCoordinates + axis - leading)
                        .laload();
                code.lastore();
            }
            inputAddress = address(code, geometrySlot, inRank, inputCoordinates, inputLayout);
            outputAddress = address(code, geometrySlot, outRank, outputCoordinates, outputLayout);
        }
        int value = code.allocateLocal(localKind(type));
        var carriers = new CpuCarrierEmitter(code);
        carriers.load(type, specialization.carrierPattern().getFirst(), 0, inputAddress,
                denseArrays);
        store(code, type, value);
        carriers.store(type, specialization.carrierPattern().getLast(), 1, outputAddress, value,
                denseArrays);
        if (denseArrays) {
            code.iinc(cell, 1).iload(cell).lload(endSlot).l2i()
                    .branch(Opcode.IF_ICMPLT, cells);
        } else {
            code.lload(cell).loadConstant(1L).ladd().lstore(cell);
            code.lload(cell).lload(endSlot).lcmp().branch(Opcode.IFLT, cells);
        }
        code.labelBinding(done);
    }

    private static void emitBoundedMinimumOrGeneral(CodeBuilder code,
            CpuKernelSpecialization specialization, String identity, int geometrySlot,
            int startSlot, int endSlot) {
        int inputCoordinates = 14, outputCoordinates = 17;
        int inputLayout = 20, outputLayout = 28;
        var fallback = code.newLabel(); var complete = code.newLabel();
        requireRange(code, startSlot, endSlot, 64, fallback);
        requireGeometrySequence(code, geometrySlot, 0, new long[]{
                CpuAggregateIr.Kind.MIN.ordinal(), DataType.BFLOAT16.ordinal(),
                CpuAggregateIr.Form.MULTI_AXIS.ordinal()}, fallback);
        requireGeometrySequence(code, geometrySlot, 3,
                new long[]{1, 3, 3, 64, 4096, 0, 0, 0}, fallback);
        requireGeometrySequence(code, geometrySlot, 11, new long[]{1, 0, 1}, fallback);
        requireGeometrySequence(code, geometrySlot, inputCoordinates,
                new long[]{0, 0, 0}, fallback);
        requireGeometrySequence(code, geometrySlot, outputCoordinates,
                new long[]{0, 0, 0}, fallback);
        requireGeometry(code, geometrySlot, inputLayout, 3, fallback);
        requireGeometrySequence(code, geometrySlot, inputLayout + 1,
                new long[]{0, 64, 64, 64, 4096, 64, 1}, fallback);
        requireGeometry(code, geometrySlot, outputLayout, 3, fallback);
        requireGeometrySequence(code, geometrySlot, outputLayout + 1,
                new long[]{3, 1, 64, 1, 128, 2, 2}, fallback);
        emitBoundedMinimum(code, specialization, startSlot, endSlot);
        code.branch(Opcode.GOTO, complete).labelBinding(fallback);
        emitGeneral(code, specialization, DataType.BFLOAT16, 0, identity, 3, 3,
                false, 0, geometrySlot, startSlot, endSlot);
        code.labelBinding(complete);
    }

    private static void emitBoundedMinimum(CodeBuilder code,
            CpuKernelSpecialization specialization, int startSlot, int endSlot) {
        int cell = code.allocateLocal(TypeKind.INT), end = code.allocateLocal(TypeKind.INT);
        int rowBase = code.allocateLocal(TypeKind.LONG);
        int inputAddress = code.allocateLocal(TypeKind.LONG);
        int outputAddress = code.allocateLocal(TypeKind.INT);
        int outer = code.allocateLocal(TypeKind.INT), inner = code.allocateLocal(TypeKind.INT);
        int accumulator = code.allocateLocal(TypeKind.INT), value = code.allocateLocal(TypeKind.INT);
        int leftValue = code.allocateLocal(TypeKind.FLOAT);
        int rightValue = code.allocateLocal(TypeKind.FLOAT);
        code.lload(startSlot).l2i().istore(cell).lload(endSlot).l2i().istore(end);
        var carriers = new CpuCarrierEmitter(code);
        var cells = code.newLabel(); var done = code.newLabel();
        var factors = code.newLabel(); var write = code.newLabel();
        code.labelBinding(cells).iload(cell).iload(end).branch(Opcode.IF_ICMPGE, done);
        code.iload(cell).i2l().loadConstant(64L).lmul().lstore(rowBase);
        code.lload(rowBase).lstore(inputAddress);
        emitDirectBfloatSegmentLoad(code, inputAddress, accumulator);
        code.lload(inputAddress).loadConstant(1L).ladd().lstore(inputAddress);
        code.loadConstant(0).istore(outer).loadConstant(1).istore(inner);
        code.labelBinding(factors);
        emitDirectBfloatSegmentLoad(code, inputAddress, value);
        emitDirectBfloatMinimum(code, accumulator, value, leftValue, rightValue);
        code.lload(inputAddress).loadConstant(1L).ladd().lstore(inputAddress);
        code.iinc(inner, 1).iload(inner).loadConstant(64)
                .branch(Opcode.IF_ICMPLT, factors);
        code.iinc(outer, 1);
        code.iload(outer).loadConstant(64).branch(Opcode.IF_ICMPGE, write);
        code.lload(rowBase).iload(outer).i2l().loadConstant(4096L).lmul().ladd()
                .lstore(inputAddress);
        code.loadConstant(0).istore(inner).branch(Opcode.GOTO, factors);
        code.labelBinding(write);
        code.loadConstant(3).iload(cell).loadConstant(2).imul().iadd().istore(outputAddress);
        carriers.store(DataType.BFLOAT16, specialization.carrierPattern().getLast(), 1,
                outputAddress, accumulator, true);
        code.iinc(cell, 1).branch(Opcode.GOTO, cells);
        code.labelBinding(done);
    }

    private static void emitDirectBfloatSegmentLoad(CodeBuilder code, int address, int result) {
        code.aload(0).getstatic(ClassDescHolder.VALUE_LAYOUT, "JAVA_SHORT_UNALIGNED",
                ClassDescHolder.SHORT_LAYOUT).lload(address).loadConstant(2L).lmul()
                .invokeinterface(ClassDescHolder.SEGMENT, "get",
                        java.lang.constant.MethodTypeDesc.of(ConstantDescs.CD_short,
                                ClassDescHolder.SHORT_LAYOUT, ConstantDescs.CD_long))
                .istore(result);
    }

    private static void emitDirectBfloatMinimum(CodeBuilder code, int left, int right,
            int leftValue, int rightValue) {
        code.iload(left).loadConstant(0xffff).iand().loadConstant(16).ishl()
                .invokestatic(ClassDescHolder.FLOAT, "intBitsToFloat",
                        java.lang.constant.MethodTypeDesc.of(ConstantDescs.CD_float,
                                ConstantDescs.CD_int))
                .fstore(leftValue);
        code.iload(right).loadConstant(0xffff).iand().loadConstant(16).ishl()
                .invokestatic(ClassDescHolder.FLOAT, "intBitsToFloat",
                        java.lang.constant.MethodTypeDesc.of(ConstantDescs.CD_float,
                                ConstantDescs.CD_int))
                .fstore(rightValue);
        var leftNumber = code.newLabel(); var rightNumber = code.newLabel();
        var ordinary = code.newLabel(); var chooseRight = code.newLabel(); var done = code.newLabel();
        code.fload(leftValue).invokestatic(ClassDescHolder.FLOAT, "isNaN",
                java.lang.constant.MethodTypeDesc.of(ConstantDescs.CD_boolean,
                        ConstantDescs.CD_float)).branch(Opcode.IFEQ, leftNumber)
                .branch(Opcode.GOTO, done).labelBinding(leftNumber);
        code.fload(rightValue).invokestatic(ClassDescHolder.FLOAT, "isNaN",
                java.lang.constant.MethodTypeDesc.of(ConstantDescs.CD_boolean,
                        ConstantDescs.CD_float)).branch(Opcode.IFEQ, rightNumber);
        code.iload(right).istore(left).branch(Opcode.GOTO, done).labelBinding(rightNumber);
        code.fload(leftValue).loadConstant(0.0f).fcmpl().branch(Opcode.IFNE, ordinary);
        code.fload(rightValue).loadConstant(0.0f).fcmpl().branch(Opcode.IFNE, ordinary);
        code.iload(left).iload(right).ior().i2s().istore(left).branch(Opcode.GOTO, done)
                .labelBinding(ordinary);
        code.fload(rightValue).fload(leftValue).fcmpg().branch(Opcode.IFLT, chooseRight)
                .branch(Opcode.GOTO, done).labelBinding(chooseRight);
        code.iload(right).istore(left).labelBinding(done);
    }

    private static void emitBoundedMeanOrGeneral(CodeBuilder code,
            CpuKernelSpecialization specialization, String identity, int geometrySlot,
            int startSlot, int endSlot) {
        int inputLayout = 16, outputLayout = 22;
        var fallback = code.newLabel(); var complete = code.newLabel();
        requireRange(code, startSlot, endSlot, 128, fallback);
        requireGeometrySequence(code, geometrySlot, 3,
                new long[]{0, 2, 1, 128, 2048, 5, 48}, fallback);
        requireGeometrySequence(code, geometrySlot, 11, new long[]{0, 1}, fallback);
        requireGeometry(code, geometrySlot, inputLayout, 2, fallback);
        requireGeometrySequence(code, geometrySlot, inputLayout + 2,
                new long[]{128, 2048, 4096, 2}, fallback);
        requireGeometry(code, geometrySlot, outputLayout, 1, fallback);
        requireGeometrySequence(code, geometrySlot, outputLayout + 2,
                new long[]{128, 2}, fallback);
        requireBoundedBase(code, geometrySlot, inputLayout + 1, 524_286, fallback);
        requireBoundedBase(code, geometrySlot, outputLayout + 1, 254, fallback);
        requireWorkspaceOffset(code, geometrySlot, 48, fallback);
        emitBoundedMean(code, specialization, geometrySlot, startSlot, endSlot,
                inputLayout, outputLayout);
        code.branch(Opcode.GOTO, complete).labelBinding(fallback);
        emitGeneral(code, specialization, DataType.FLOAT32, 5, identity, 2, 1,
                true, 5, geometrySlot, startSlot, endSlot);
        code.labelBinding(complete);
    }

    private static void emitBoundedMean(CodeBuilder code,
            CpuKernelSpecialization specialization, int geometrySlot, int startSlot, int endSlot,
            int inputLayout, int outputLayout) {
        int cell = code.allocateLocal(TypeKind.INT), end = code.allocateLocal(TypeKind.INT);
        int inputBase = code.allocateLocal(TypeKind.LONG);
        int outputBase = code.allocateLocal(TypeKind.INT);
        int inputAddress = code.allocateLocal(TypeKind.LONG);
        int outputAddress = code.allocateLocal(TypeKind.INT);
        int domain = code.allocateLocal(TypeKind.INT);
        int value = code.allocateLocal(TypeKind.FLOAT);
        int result = code.allocateLocal(TypeKind.FLOAT);
        code.lload(startSlot).l2i().istore(cell).lload(endSlot).l2i().istore(end);
        geometry(code, geometrySlot, inputLayout + 1).lstore(inputBase);
        geometry(code, geometrySlot, outputLayout + 1).l2i().istore(outputBase);
        int[] exact = boundedMeanLocals(code, geometrySlot);
        var carriers = new CpuCarrierEmitter(code);
        var cells = code.newLabel(); var done = code.newLabel();
        var factors = code.newLabel();
        code.labelBinding(cells).iload(cell).iload(end).branch(Opcode.IF_ICMPGE, done);
        code.loadConstant(0.0f).fstore(result);
        emitBoundedMeanReset(code, exact);
        code.lload(inputBase).iload(cell).i2l().loadConstant(4096L).lmul().ladd()
                .lstore(inputAddress);
        code.loadConstant(0).istore(domain).labelBinding(factors);
        carriers.load(DataType.FLOAT32, specialization.carrierPattern().getFirst(), 0,
                inputAddress);
        code.fstore(value); emitBoundedMeanFactor(code, exact, value);
        code.lload(inputAddress).loadConstant(2L).ladd().lstore(inputAddress);
        code.iinc(domain, 1).iload(domain).loadConstant(2048)
                .branch(Opcode.IF_ICMPLT, factors);
        emitBoundedMeanFinish(code, exact, result);
        code.iload(outputBase).iload(cell).loadConstant(2).imul().iadd()
                .istore(outputAddress);
        carriers.store(DataType.FLOAT32, specialization.carrierPattern().getLast(), 1,
                outputAddress, result, true);
        code.iinc(cell, 1).branch(Opcode.GOTO, cells).labelBinding(done);
    }

    private static final int M_OFFSET = 0, M_FLAGS = 1, M_BITS = 2, M_FRACTION = 3;
    private static final int M_EXPONENT = 4, M_SIGNIFICAND = 5, M_SHIFT = 6, M_WORD = 7;
    private static final int M_BIT_OFFSET = 8, M_LIMB = 9, M_ADDEND = 10, M_CARRY = 11;
    private static final int M_OLD = 12, M_SUM = 13, M_POSITION = 14, M_NEGATIVE = 15;
    private static final int M_TOP = 16, M_BIT_LENGTH = 17, M_UNBIASED = 18, M_QUOTIENT = 19;
    private static final int M_GUARD = 20, M_STICKY = 21, M_RESULT = 22;
    private static final int M_L0 = 23, M_L1 = 24, M_L2 = 25, M_L3 = 26, M_L4 = 27;

    private static int[] boundedMeanLocals(CodeBuilder code, int geometrySlot) {
        int[] local = new int[28];
        for (int index : new int[]{M_OFFSET, M_FLAGS, M_BITS, M_FRACTION, M_EXPONENT,
                M_SIGNIFICAND, M_ADDEND, M_CARRY, M_OLD, M_SUM, M_POSITION, M_TOP,
                M_UNBIASED, M_QUOTIENT, M_RESULT, M_L0, M_L1, M_L2, M_L3, M_L4})
            local[index] = code.allocateLocal(TypeKind.LONG);
        for (int index : new int[]{M_SHIFT, M_WORD, M_BIT_OFFSET, M_LIMB, M_NEGATIVE,
                M_BIT_LENGTH, M_GUARD, M_STICKY})
            local[index] = code.allocateLocal(TypeKind.INT);
        geometry(code, geometrySlot, 10).lstore(local[M_OFFSET]);
        return local;
    }

    private static void emitBoundedMeanReset(CodeBuilder code, int[] m) {
        code.loadConstant(0L).lstore(m[M_FLAGS]);
        for (int limb = 0; limb < 5; limb++)
            code.loadConstant(0L).lstore(m[M_L0 + limb]);
    }

    private static void emitBoundedMeanFactor(CodeBuilder code, int[] m, int value) {
        code.fload(value).invokestatic(ClassDescHolder.FLOAT, "floatToRawIntBits",
                java.lang.constant.MethodTypeDesc.of(ConstantDescs.CD_int, ConstantDescs.CD_float))
                .i2l().loadConstant(0xffff_ffffL).land().lstore(m[M_BITS]);
        code.lload(m[M_BITS]).loadConstant(0x8000_0000L).land().loadConstant(0L).lcmp();
        var positive = code.newLabel(); var signReady = code.newLabel();
        code.branch(Opcode.IFEQ, positive).loadConstant(1).branch(Opcode.GOTO, signReady)
                .labelBinding(positive).loadConstant(0).labelBinding(signReady)
                .istore(m[M_NEGATIVE]);
        code.lload(m[M_BITS]).loadConstant(0x7f_ffffL).land().lstore(m[M_FRACTION]);
        code.lload(m[M_BITS]).loadConstant(23).lushr().loadConstant(0xffL).land()
                .lstore(m[M_EXPONENT]);
        var finite = code.newLabel(); var classified = code.newLabel();
        code.lload(m[M_EXPONENT]).loadConstant(0xffL).lcmp().branch(Opcode.IFNE, finite);
        var infinity = code.newLabel();
        code.lload(m[M_FRACTION]).loadConstant(0L).lcmp().branch(Opcode.IFEQ, infinity);
        code.lload(m[M_FLAGS]).loadConstant(1L).lor().lstore(m[M_FLAGS])
                .branch(Opcode.GOTO, classified).labelBinding(infinity);
        code.lload(m[M_FLAGS]).iload(m[M_NEGATIVE]);
        var positiveInfinity = code.newLabel(); var infinityReady = code.newLabel();
        code.branch(Opcode.IFEQ, positiveInfinity).loadConstant(4L)
                .branch(Opcode.GOTO, infinityReady).labelBinding(positiveInfinity)
                .loadConstant(2L).labelBinding(infinityReady).lor().lstore(m[M_FLAGS])
                .branch(Opcode.GOTO, classified).labelBinding(finite);
        var nonzero = code.newLabel();
        code.lload(m[M_EXPONENT]).loadConstant(0L).lcmp().branch(Opcode.IFNE, nonzero);
        code.lload(m[M_FRACTION]).loadConstant(0L).lcmp().branch(Opcode.IFNE, nonzero);
        code.lload(m[M_FLAGS]).iload(m[M_NEGATIVE]);
        var positiveZero = code.newLabel(); var zeroReady = code.newLabel();
        code.branch(Opcode.IFEQ, positiveZero).loadConstant(16L)
                .branch(Opcode.GOTO, zeroReady).labelBinding(positiveZero)
                .loadConstant(8L).labelBinding(zeroReady).lor().lstore(m[M_FLAGS])
                .branch(Opcode.GOTO, classified).labelBinding(nonzero);
        code.lload(m[M_FLAGS]).loadConstant(32L).lor().lstore(m[M_FLAGS]);
        var normal = code.newLabel(); var factorReady = code.newLabel();
        code.lload(m[M_EXPONENT]).loadConstant(0L).lcmp().branch(Opcode.IFNE, normal);
        code.lload(m[M_FRACTION]).lstore(m[M_SIGNIFICAND]);
        code.loadConstant(0).istore(m[M_SHIFT]).branch(Opcode.GOTO, factorReady)
                .labelBinding(normal);
        code.loadConstant(1L << 23).lload(m[M_FRACTION]).lor().lstore(m[M_SIGNIFICAND]);
        code.lload(m[M_EXPONENT]).loadConstant(1L).lsub().l2i().istore(m[M_SHIFT])
                .labelBinding(factorReady);
        code.iload(m[M_SHIFT]).loadConstant(6).iushr().istore(m[M_WORD]);
        code.iload(m[M_SHIFT]).loadConstant(63).iand().istore(m[M_BIT_OFFSET]);
        var positiveAdd = code.newLabel(); var generalPositiveAdd = code.newLabel();
        var added = code.newLabel();
        code.iload(m[M_NEGATIVE]).branch(Opcode.IFEQ, positiveAdd);
        emitBoundedMeanAddLoop(code, m, true);
        code.branch(Opcode.GOTO, added).labelBinding(positiveAdd);
        code.lload(m[M_EXPONENT]).loadConstant(126L).lcmp()
                .branch(Opcode.IFLT, generalPositiveAdd);
        code.lload(m[M_EXPONENT]).loadConstant(127L).lcmp()
                .branch(Opcode.IFGT, generalPositiveAdd);
        emitBoundedMeanPositiveNearOne(code, m);
        code.branch(Opcode.GOTO, added).labelBinding(generalPositiveAdd);
        emitBoundedMeanPositiveAdd(code, m);
        code.labelBinding(added).labelBinding(classified);
    }

    private static void emitBoundedMeanPositiveNearOne(CodeBuilder code, int[] m) {
        code.lload(m[M_SIGNIFICAND]).iload(m[M_BIT_OFFSET]).lshl().lstore(m[M_ADDEND]);
        emitBoundedMeanAddToLocalLimb(code, m, 1, false);
        code.lload(m[M_SIGNIFICAND]).loadConstant(64).iload(m[M_BIT_OFFSET]).isub().lushr()
                .lstore(m[M_ADDEND]);
        emitBoundedMeanAddToLocalLimb(code, m, 2, true);
        var done = code.newLabel();
        code.lload(m[M_CARRY]).loadConstant(0L).lcmp().branch(Opcode.IFEQ, done);
        code.loadConstant(1L).lstore(m[M_ADDEND]);
        emitBoundedMeanAddToLocalLimb(code, m, 3, false);
        code.lload(m[M_CARRY]).loadConstant(0L).lcmp().branch(Opcode.IFEQ, done);
        code.loadConstant(1L).lstore(m[M_ADDEND]);
        emitBoundedMeanAddToLocalLimb(code, m, 4, false);
        code.labelBinding(done);
    }

    private static void emitBoundedMeanAddToLocalLimb(CodeBuilder code, int[] m, int limb,
            boolean includeCarry) {
        if (includeCarry) {
            code.lload(m[M_ADDEND]).lload(m[M_CARRY]).ladd().lstore(m[M_SUM]);
            code.lload(m[M_SUM]).lload(m[M_ADDEND]).invokestatic(ClassDescHolder.LONG,
                    "compareUnsigned", java.lang.constant.MethodTypeDesc.of(ConstantDescs.CD_int,
                            ConstantDescs.CD_long, ConstantDescs.CD_long));
            var noFirstCarry = code.newLabel(); var firstCarryReady = code.newLabel();
            code.branch(Opcode.IFGE, noFirstCarry).loadConstant(1L)
                    .branch(Opcode.GOTO, firstCarryReady).labelBinding(noFirstCarry)
                    .loadConstant(0L).labelBinding(firstCarryReady).lstore(m[M_POSITION]);
        } else {
            code.lload(m[M_ADDEND]).lstore(m[M_SUM]);
            code.loadConstant(0L).lstore(m[M_POSITION]);
        }
        code.lload(m[M_L0 + limb]).lstore(m[M_OLD]);
        code.lload(m[M_OLD]).lload(m[M_SUM]).ladd().lstore(m[M_ADDEND]);
        code.lload(m[M_ADDEND]).lload(m[M_OLD]).invokestatic(ClassDescHolder.LONG,
                "compareUnsigned", java.lang.constant.MethodTypeDesc.of(ConstantDescs.CD_int,
                        ConstantDescs.CD_long, ConstantDescs.CD_long));
        var noCarry = code.newLabel();
        code.branch(Opcode.IFGE, noCarry).lload(m[M_POSITION]).loadConstant(1L).lor()
                .lstore(m[M_POSITION]).labelBinding(noCarry);
        code.lload(m[M_POSITION]).lstore(m[M_CARRY]);
        code.lload(m[M_ADDEND]).lstore(m[M_L0 + limb]);
    }

    private static void emitBoundedMeanAddLoop(CodeBuilder code, int[] m, boolean complement) {
        code.loadConstant(complement ? 1L : 0L).lstore(m[M_CARRY]);
        for (int limb = 0; limb < 5; limb++) {
            code.loadConstant(0L).lstore(m[M_ADDEND]);
            var nextPart = code.newLabel(); var partReady = code.newLabel();
            code.iload(m[M_WORD]).loadConstant(limb).branch(Opcode.IF_ICMPNE, nextPart);
            code.lload(m[M_SIGNIFICAND]).iload(m[M_BIT_OFFSET]).lshl().lstore(m[M_ADDEND])
                    .branch(Opcode.GOTO, partReady).labelBinding(nextPart);
            code.iload(m[M_BIT_OFFSET]).branch(Opcode.IFEQ, partReady);
            code.iload(m[M_WORD]).loadConstant(limb - 1).branch(Opcode.IF_ICMPNE, partReady);
            code.lload(m[M_SIGNIFICAND]).loadConstant(64).iload(m[M_BIT_OFFSET]).isub().lushr()
                    .lstore(m[M_ADDEND]).labelBinding(partReady);
            if (complement)
                code.lload(m[M_ADDEND]).loadConstant(-1L).lxor().lstore(m[M_ADDEND]);
            emitBoundedMeanAddToLocalLimb(code, m, limb, true);
        }
    }

    private static void emitBoundedMeanPositiveAdd(CodeBuilder code, int[] m) {
        code.loadConstant(0L).lstore(m[M_CARRY]);
        for (int limb = 0; limb < 5; limb++) {
            code.loadConstant(0L).lstore(m[M_ADDEND]);
            var nextPart = code.newLabel(); var partReady = code.newLabel();
            code.iload(m[M_WORD]).loadConstant(limb).branch(Opcode.IF_ICMPNE, nextPart);
            code.lload(m[M_SIGNIFICAND]).iload(m[M_BIT_OFFSET]).lshl().lstore(m[M_ADDEND])
                    .branch(Opcode.GOTO, partReady).labelBinding(nextPart);
            code.iload(m[M_BIT_OFFSET]).branch(Opcode.IFEQ, partReady);
            code.iload(m[M_WORD]).loadConstant(limb - 1).branch(Opcode.IF_ICMPNE, partReady);
            code.lload(m[M_SIGNIFICAND]).loadConstant(64).iload(m[M_BIT_OFFSET]).isub().lushr()
                    .lstore(m[M_ADDEND]).labelBinding(partReady);
            code.lload(m[M_ADDEND]).lload(m[M_CARRY]).ladd().lstore(m[M_SUM]);
            code.lload(m[M_SUM]).lload(m[M_ADDEND]).invokestatic(ClassDescHolder.LONG,
                    "compareUnsigned", java.lang.constant.MethodTypeDesc.of(ConstantDescs.CD_int,
                            ConstantDescs.CD_long, ConstantDescs.CD_long));
            var noFirstCarry = code.newLabel(); var firstCarryReady = code.newLabel();
            code.branch(Opcode.IFGE, noFirstCarry).loadConstant(1L)
                    .branch(Opcode.GOTO, firstCarryReady).labelBinding(noFirstCarry)
                    .loadConstant(0L).labelBinding(firstCarryReady).lstore(m[M_CARRY]);
            code.lload(m[M_L0 + limb]).lstore(m[M_OLD]);
            code.lload(m[M_OLD]).lload(m[M_SUM]).ladd().lstore(m[M_ADDEND]);
            code.lload(m[M_ADDEND]).lload(m[M_OLD]).invokestatic(ClassDescHolder.LONG,
                    "compareUnsigned", java.lang.constant.MethodTypeDesc.of(ConstantDescs.CD_int,
                            ConstantDescs.CD_long, ConstantDescs.CD_long));
            var noSecondCarry = code.newLabel();
            code.branch(Opcode.IFGE, noSecondCarry).loadConstant(1L).lload(m[M_CARRY]).lor()
                    .lstore(m[M_CARRY]).labelBinding(noSecondCarry);
            code.lload(m[M_ADDEND]).lstore(m[M_L0 + limb]);
        }
    }

    private static void emitBoundedMeanFinish(CodeBuilder code, int[] m, int value) {
        code.loadConstant(0L).lstore(m[M_RESULT]);
        var finite = code.newLabel(); var store = code.newLabel();
        code.lload(m[M_FLAGS]).loadConstant(1L).land().loadConstant(0L).lcmp()
                .branch(Opcode.IFEQ, finite);
        code.loadConstant(0x7fc0_0000L).lstore(m[M_RESULT]).branch(Opcode.GOTO, store)
                .labelBinding(finite);
        code.lload(m[M_FLAGS]).loadConstant(6L).land().loadConstant(6L).lcmp();
        var oneInfinity = code.newLabel();
        code.branch(Opcode.IFNE, oneInfinity).loadConstant(0x7fc0_0000L)
                .lstore(m[M_RESULT]).branch(Opcode.GOTO, store).labelBinding(oneInfinity);
        code.lload(m[M_FLAGS]).loadConstant(2L).land().loadConstant(0L).lcmp();
        var negativeInfinity = code.newLabel();
        code.branch(Opcode.IFEQ, negativeInfinity).loadConstant(0x7f80_0000L)
                .lstore(m[M_RESULT]).branch(Opcode.GOTO, store).labelBinding(negativeInfinity);
        code.lload(m[M_FLAGS]).loadConstant(4L).land().loadConstant(0L).lcmp();
        var magnitude = code.newLabel();
        code.branch(Opcode.IFEQ, magnitude).loadConstant(0xff80_0000L)
                .lstore(m[M_RESULT]).branch(Opcode.GOTO, store).labelBinding(magnitude);
        code.lload(m[M_L4]).loadConstant(0L).lcmp();
        var positive = code.newLabel(); var magnitudeReady = code.newLabel();
        code.branch(Opcode.IFGE, positive).loadConstant(1).istore(m[M_NEGATIVE]);
        code.loadConstant(1L).lstore(m[M_CARRY]);
        for (int limb = 0; limb < 5; limb++) {
            code.lload(m[M_L0 + limb]).loadConstant(-1L).lxor().lstore(m[M_OLD]);
            code.lload(m[M_OLD]).lload(m[M_CARRY]).ladd().lstore(m[M_SUM]);
            code.lload(m[M_SUM]).lload(m[M_OLD]).invokestatic(ClassDescHolder.LONG,
                    "compareUnsigned", java.lang.constant.MethodTypeDesc.of(ConstantDescs.CD_int,
                            ConstantDescs.CD_long, ConstantDescs.CD_long));
            var noCarry = code.newLabel();
            code.branch(Opcode.IFLT, noCarry).loadConstant(0L).lstore(m[M_CARRY])
                    .labelBinding(noCarry);
            code.lload(m[M_SUM]).lstore(m[M_L0 + limb]);
        }
        code.branch(Opcode.GOTO, magnitudeReady).labelBinding(positive);
        code.loadConstant(0).istore(m[M_NEGATIVE]).labelBinding(magnitudeReady);
        var foundTop = code.newLabel();
        for (int limb = 4; limb >= 0; limb--) {
            code.loadConstant(limb).istore(m[M_LIMB]);
            code.lload(m[M_L0 + limb]).lstore(m[M_TOP]);
            if (limb != 0)
                code.lload(m[M_TOP]).loadConstant(0L).lcmp().branch(Opcode.IFNE, foundTop);
        }
        code.labelBinding(foundTop);
        code.lload(m[M_TOP]).loadConstant(0L).lcmp(); var nonzero = code.newLabel();
        code.branch(Opcode.IFNE, nonzero);
        code.lload(m[M_FLAGS]).loadConstant(16L).lcmp(); var positiveZero = code.newLabel();
        code.branch(Opcode.IFNE, positiveZero).loadConstant(0x8000_0000L)
                .lstore(m[M_RESULT]).branch(Opcode.GOTO, store).labelBinding(positiveZero);
        code.loadConstant(0L).lstore(m[M_RESULT]).branch(Opcode.GOTO, store)
                .labelBinding(nonzero);
        code.iload(m[M_LIMB]).loadConstant(64).imul().loadConstant(64)
                .lload(m[M_TOP]).invokestatic(ClassDescHolder.LONG, "numberOfLeadingZeros",
                        java.lang.constant.MethodTypeDesc.of(ConstantDescs.CD_int,
                                ConstantDescs.CD_long))
                .isub().iadd().istore(m[M_BIT_LENGTH]);
        code.loadConstant(-161L).iload(m[M_BIT_LENGTH]).i2l().ladd()
                .lstore(m[M_UNBIASED]);
        code.lload(m[M_UNBIASED]).loadConstant(-126L).lcmp();
        var subnormal = code.newLabel(); var shiftReady = code.newLabel();
        code.branch(Opcode.IFLT, subnormal).iload(m[M_BIT_LENGTH]).loadConstant(24).isub()
                .istore(m[M_SHIFT]).branch(Opcode.GOTO, shiftReady).labelBinding(subnormal);
        code.loadConstant(11).istore(m[M_SHIFT]).labelBinding(shiftReady);
        emitBoundedMeanExtractAndRound(code, m);
        code.lload(m[M_UNBIASED]).loadConstant(-126L).lcmp();
        var finishSubnormal = code.newLabel();
        code.branch(Opcode.IFLT, finishSubnormal);
        code.lload(m[M_QUOTIENT]).loadConstant(1L << 24).lcmp();
        var widthReady = code.newLabel();
        code.branch(Opcode.IFNE, widthReady).lload(m[M_QUOTIENT]).loadConstant(1).lushr()
                .lstore(m[M_QUOTIENT]);
        code.lload(m[M_UNBIASED]).loadConstant(1L).ladd().lstore(m[M_UNBIASED])
                .labelBinding(widthReady);
        code.lload(m[M_UNBIASED]).loadConstant(127L).lcmp(); var notOverflow = code.newLabel();
        code.branch(Opcode.IFLE, notOverflow); emitBoundedMeanSign(code, m);
        code.loadConstant(0x7f80_0000L).lor().lstore(m[M_RESULT])
                .branch(Opcode.GOTO, store).labelBinding(notOverflow);
        emitBoundedMeanSign(code, m);
        code.lload(m[M_UNBIASED]).loadConstant(127L).ladd().loadConstant(23).lshl()
                .lload(m[M_QUOTIENT]).loadConstant(0x7f_ffffL).land().lor().lor()
                .lstore(m[M_RESULT]).branch(Opcode.GOTO, store)
                .labelBinding(finishSubnormal);
        emitBoundedMeanSign(code, m); code.lstore(m[M_RESULT]);
        code.lload(m[M_QUOTIENT]).loadConstant(1L << 23).lcmp();
        var trueSubnormal = code.newLabel();
        code.branch(Opcode.IFLT, trueSubnormal).lload(m[M_RESULT]).loadConstant(1L << 23).lor()
                .lstore(m[M_RESULT]).branch(Opcode.GOTO, store).labelBinding(trueSubnormal);
        code.lload(m[M_RESULT]).lload(m[M_QUOTIENT]).lor().lstore(m[M_RESULT])
                .labelBinding(store);
        emitBoundedMeanStoreState(code, m);
        code.lload(m[M_RESULT]).l2i().invokestatic(ClassDescHolder.FLOAT, "intBitsToFloat",
                java.lang.constant.MethodTypeDesc.of(ConstantDescs.CD_float, ConstantDescs.CD_int))
                .fstore(value);
    }

    private static void emitBoundedMeanStoreState(CodeBuilder code, int[] m) {
        meanSet(code, m, 0, () -> code.lload(m[M_FLAGS]));
        for (int limb = 0; limb < 5; limb++) {
            int index = M_L0 + limb;
            meanSet(code, m, 8 + 8 * limb, () -> code.lload(m[index]));
        }
    }

    private static void emitBoundedMeanExtractAndRound(CodeBuilder code, int[] m) {
        code.iload(m[M_SHIFT]).loadConstant(6).iushr().istore(m[M_WORD]);
        code.iload(m[M_SHIFT]).loadConstant(63).iand().istore(m[M_BIT_OFFSET]);
        meanLoadSelectedLimb(code, m, m[M_WORD]);
        code.iload(m[M_BIT_OFFSET]).lushr()
                .lstore(m[M_QUOTIENT]);
        var noNext = code.newLabel();
        code.iload(m[M_BIT_OFFSET]).branch(Opcode.IFEQ, noNext);
        code.iload(m[M_WORD]).loadConstant(1).iadd().loadConstant(5)
                .branch(Opcode.IF_ICMPGE, noNext);
        code.iinc(m[M_WORD], 1); meanLoadSelectedLimb(code, m, m[M_WORD]);
        code.loadConstant(64).iload(m[M_BIT_OFFSET]).isub().lshl()
                .lload(m[M_QUOTIENT]).lor().lstore(m[M_QUOTIENT]).labelBinding(noNext);
        code.loadConstant(0).istore(m[M_GUARD]).loadConstant(0).istore(m[M_STICKY]);
        code.iload(m[M_SHIFT]).loadConstant(1).isub().istore(m[M_WORD]);
        code.iload(m[M_WORD]).loadConstant(6).iushr().istore(m[M_LIMB]);
        code.iload(m[M_WORD]).loadConstant(63).iand().istore(m[M_BIT_OFFSET]);
        meanLoadSelectedLimb(code, m, m[M_LIMB]);
        code.iload(m[M_BIT_OFFSET]).lushr().loadConstant(1L).land()
                .l2i().istore(m[M_GUARD]);
        code.loadConstant(0).istore(m[M_LIMB]);
        var stickyLoop = code.newLabel(); var stickyDone = code.newLabel();
        code.labelBinding(stickyLoop).iload(m[M_LIMB]).iload(m[M_WORD]).loadConstant(6).iushr()
                .branch(Opcode.IF_ICMPGE, stickyDone);
        meanLoadSelectedLimb(code, m, m[M_LIMB]);
        code.loadConstant(0L).lcmp(); var next = code.newLabel();
        code.branch(Opcode.IFEQ, next).loadConstant(1).istore(m[M_STICKY])
                .branch(Opcode.GOTO, stickyDone).labelBinding(next);
        code.iinc(m[M_LIMB], 1).branch(Opcode.GOTO, stickyLoop).labelBinding(stickyDone);
        code.iload(m[M_STICKY]); var round = code.newLabel();
        code.branch(Opcode.IFNE, round);
        code.iload(m[M_BIT_OFFSET]).branch(Opcode.IFEQ, round);
        meanLoadSelectedLimb(code, m, m[M_LIMB]);
        code.loadConstant(1L).iload(m[M_BIT_OFFSET]).lshl()
                .loadConstant(1L).lsub().land().loadConstant(0L).lcmp();
        var noLow = code.newLabel();
        code.branch(Opcode.IFEQ, noLow).loadConstant(1).istore(m[M_STICKY])
                .labelBinding(noLow).labelBinding(round);
        code.iload(m[M_GUARD]); var rounded = code.newLabel();
        code.branch(Opcode.IFEQ, rounded).iload(m[M_STICKY]); var up = code.newLabel();
        code.branch(Opcode.IFNE, up).lload(m[M_QUOTIENT]).loadConstant(1L).land()
                .loadConstant(0L).lcmp().branch(Opcode.IFEQ, rounded).labelBinding(up);
        code.lload(m[M_QUOTIENT]).loadConstant(1L).ladd().lstore(m[M_QUOTIENT])
                .labelBinding(rounded);
    }

    private static void emitBoundedMeanSign(CodeBuilder code, int[] m) {
        var positive = code.newLabel(); var done = code.newLabel();
        code.iload(m[M_NEGATIVE]).branch(Opcode.IFEQ, positive).loadConstant(0x8000_0000L)
                .branch(Opcode.GOTO, done).labelBinding(positive).loadConstant(0L)
                .labelBinding(done);
    }

    private static void meanLoadSelectedLimb(CodeBuilder code, int[] m, int selected) {
        var done = code.newLabel();
        for (int limb = 0; limb < 4; limb++) {
            var next = code.newLabel();
            code.iload(selected).loadConstant(limb).branch(Opcode.IF_ICMPNE, next);
            code.lload(m[M_L0 + limb]).branch(Opcode.GOTO, done).labelBinding(next);
        }
        code.lload(m[M_L4]).labelBinding(done);
    }

    private static void meanSet(CodeBuilder code, int[] m, int delta, Runnable value) {
        code.aload(2).getstatic(ClassDescHolder.VALUE_LAYOUT, "JAVA_LONG",
                ClassDescHolder.LONG_LAYOUT).lload(m[M_OFFSET]);
        if (delta != 0) code.loadConstant((long) delta).ladd();
        value.run();
        code.invokeinterface(ClassDescHolder.SEGMENT, "set",
                java.lang.constant.MethodTypeDesc.of(ConstantDescs.CD_void,
                        ClassDescHolder.LONG_LAYOUT, ConstantDescs.CD_long, ConstantDescs.CD_long));
    }

    private static void emitBoundedProductOrGeneral(CodeBuilder code,
            CpuKernelSpecialization specialization, String identity, int geometrySlot,
            int startSlot, int endSlot) {
        int inputLayout = 20, outputLayout = 28;
        var fallback = code.newLabel(); var complete = code.newLabel();
        requireRange(code, startSlot, endSlot, 16_384, fallback);
        requireGeometrySequence(code, geometrySlot, 3,
                new long[]{1, 3, 3, 16_384, 16, 6, 72}, fallback);
        requireGeometrySequence(code, geometrySlot, 11, new long[]{1, 0, 1}, fallback);
        requireGeometry(code, geometrySlot, inputLayout, 3, fallback);
        requireGeometrySequence(code, geometrySlot, inputLayout + 2,
                new long[]{4, 16_384, 4, 65_536, 4, 1}, fallback);
        requireGeometry(code, geometrySlot, outputLayout, 3, fallback);
        requireGeometrySequence(code, geometrySlot, outputLayout + 2,
                new long[]{1, 16_384, 1, 32_768, 2, 2}, fallback);
        requireBoundedBase(code, geometrySlot, inputLayout + 1, 262_143, fallback);
        requireBoundedBase(code, geometrySlot, outputLayout + 1, 32_766, fallback);
        requireWorkspaceOffset(code, geometrySlot, 72, fallback);
        emitBoundedProduct(code, specialization, geometrySlot, startSlot, endSlot,
                inputLayout, outputLayout);
        code.branch(Opcode.GOTO, complete).labelBinding(fallback);
        emitGeneral(code, specialization, DataType.BFLOAT16, 6, identity, 3, 3,
                true, 6, geometrySlot, startSlot, endSlot);
        code.labelBinding(complete);
    }

    private static void emitBoundedProduct(CodeBuilder code,
            CpuKernelSpecialization specialization, int geometrySlot, int startSlot, int endSlot,
            int inputLayout, int outputLayout) {
        int cell = code.allocateLocal(TypeKind.INT), end = code.allocateLocal(TypeKind.INT);
        int inputBase = code.allocateLocal(TypeKind.INT);
        int outputBase = code.allocateLocal(TypeKind.LONG);
        int rowBase = code.allocateLocal(TypeKind.INT);
        int inputAddress = code.allocateLocal(TypeKind.INT);
        int outputAddress = code.allocateLocal(TypeKind.LONG);
        int outer = code.allocateLocal(TypeKind.INT), inner = code.allocateLocal(TypeKind.INT);
        int value = code.allocateLocal(TypeKind.INT), result = code.allocateLocal(TypeKind.INT);
        code.lload(startSlot).l2i().istore(cell).lload(endSlot).l2i().istore(end);
        geometry(code, geometrySlot, inputLayout + 1).l2i().istore(inputBase);
        geometry(code, geometrySlot, outputLayout + 1).lstore(outputBase);
        var exact = new CpuExactProductEmitter(code, DataType.BFLOAT16,
                2, geometrySlot, 10, 9, true);
        var carriers = new CpuCarrierEmitter(code);
        var cells = code.newLabel(); var done = code.newLabel();
        var outerLoop = code.newLabel(); var innerLoop = code.newLabel(); var nextOuter = code.newLabel();
        code.labelBinding(cells).iload(cell).iload(end).branch(Opcode.IF_ICMPGE, done);
        code.loadConstant(0).istore(result); exact.emitReset();
        code.iload(inputBase).iload(cell).loadConstant(4).imul().iadd().istore(rowBase);
        code.loadConstant(0).istore(outer).labelBinding(outerLoop);
        code.iload(outer).loadConstant(4).branch(Opcode.IF_ICMPGE, nextOuter);
        code.iload(rowBase).iload(outer).loadConstant(65_536).imul().iadd()
                .istore(inputAddress);
        code.loadConstant(0).istore(inner).labelBinding(innerLoop);
        carriers.load(DataType.BFLOAT16, specialization.carrierPattern().getFirst(), 0,
                inputAddress, true);
        code.istore(value); exact.emitFactor(value);
        code.iinc(inputAddress, 1).iinc(inner, 1).iload(inner).loadConstant(4)
                .branch(Opcode.IF_ICMPLT, innerLoop);
        code.iinc(outer, 1).iload(outer).loadConstant(4)
                .branch(Opcode.IF_ICMPLT, outerLoop);
        code.labelBinding(nextOuter);
        int found = code.allocateLocal(TypeKind.INT); code.loadConstant(1).istore(found);
        exact.emitFinish(result, found);
        code.lload(outputBase).iload(cell).i2l().loadConstant(2L).lmul().ladd()
                .lstore(outputAddress);
        carriers.store(DataType.BFLOAT16, specialization.carrierPattern().getLast(), 1,
                outputAddress, result);
        code.iinc(cell, 1).branch(Opcode.GOTO, cells).labelBinding(done);
    }

    private static void requireRange(CodeBuilder code, int startSlot, int endSlot, long maximum,
            java.lang.classfile.Label fallback) {
        code.lload(startSlot).loadConstant(0L).lcmp().branch(Opcode.IFLT, fallback);
        code.lload(endSlot).lload(startSlot).lcmp().branch(Opcode.IFLT, fallback);
        code.lload(endSlot).loadConstant(maximum).lcmp().branch(Opcode.IFGT, fallback);
    }

    private static void requireGeometrySequence(CodeBuilder code, int geometrySlot, int first,
            long[] expected, java.lang.classfile.Label fallback) {
        for (int index = 0; index < expected.length; index++)
            requireGeometry(code, geometrySlot, first + index, expected[index], fallback);
    }

    private static void requireGeometry(CodeBuilder code, int geometrySlot, int index,
            long expected, java.lang.classfile.Label fallback) {
        geometry(code, geometrySlot, index).loadConstant(expected).lcmp()
                .branch(Opcode.IFNE, fallback);
    }

    private static void requireBoundedBase(CodeBuilder code, int geometrySlot, int index,
            long span, java.lang.classfile.Label fallback) {
        geometry(code, geometrySlot, index).loadConstant(0L).lcmp().branch(Opcode.IFLT, fallback);
        geometry(code, geometrySlot, index).loadConstant((long) Integer.MAX_VALUE - span).lcmp()
                .branch(Opcode.IFGT, fallback);
    }

    private static void requireWorkspaceOffset(CodeBuilder code, int geometrySlot, long bytes,
            java.lang.classfile.Label fallback) {
        geometry(code, geometrySlot, 10).loadConstant(0L).lcmp().branch(Opcode.IFLT, fallback);
        geometry(code, geometrySlot, 10).loadConstant(Long.MAX_VALUE - bytes).lcmp()
                .branch(Opcode.IFGT, fallback);
    }

    private static CodeBuilder geometry(CodeBuilder code, int geometrySlot, int index) {
        return code.aload(geometrySlot).loadConstant(index).laload();
    }

    private static void emitZeroStrideAnyOrGeneral(CodeBuilder code,
            CpuKernelSpecialization specialization, String identity, int inRank, int outRank,
            int geometrySlot, int startSlot, int endSlot) {
        int inputLayout = 11 + 2 * inRank + outRank;
        int outputLayout = inputLayout + 2 + 2 * inRank;
        var fallback = code.newLabel();
        var done = code.newLabel();
        code.lload(startSlot).loadConstant(0L).lcmp().branch(Opcode.IFLT, fallback);
        code.lload(startSlot).loadConstant((long) Integer.MAX_VALUE).lcmp()
                .branch(Opcode.IFGT, fallback);
        code.lload(endSlot).lload(startSlot).lcmp().branch(Opcode.IFLT, fallback);
        code.lload(endSlot).loadConstant((long) Integer.MAX_VALUE).lcmp()
                .branch(Opcode.IFGT, fallback);
        code.aload(geometrySlot).loadConstant(7).laload().loadConstant(0L).lcmp()
                .branch(Opcode.IFLT, fallback);
        code.aload(geometrySlot).loadConstant(7).laload()
                .loadConstant((long) Integer.MAX_VALUE).lcmp().branch(Opcode.IFGT, fallback);
        code.aload(geometrySlot).loadConstant(inputLayout).laload().loadConstant(2L).lcmp()
                .branch(Opcode.IFNE, fallback);
        code.aload(geometrySlot).loadConstant(outputLayout).laload().loadConstant(1L).lcmp()
                .branch(Opcode.IFNE, fallback);
        code.aload(geometrySlot).loadConstant(inputLayout + 3).laload()
                .aload(geometrySlot).loadConstant(7).laload().lcmp()
                .branch(Opcode.IFNE, fallback);
        code.aload(geometrySlot).loadConstant(inputLayout + 2).laload()
                .aload(geometrySlot).loadConstant(outputLayout + 2).laload().lcmp()
                .branch(Opcode.IFNE, fallback);
        code.aload(geometrySlot).loadConstant(inputLayout + 5).laload().loadConstant(0L).lcmp()
                .branch(Opcode.IFNE, fallback);
        for (int index : new int[]{inputLayout + 1, inputLayout + 2, inputLayout + 4,
                outputLayout + 1, outputLayout + 2, outputLayout + 3}) {
            code.aload(geometrySlot).loadConstant(index).laload().loadConstant(0L).lcmp()
                    .branch(Opcode.IFLT, fallback);
            code.aload(geometrySlot).loadConstant(index).laload()
                    .loadConstant((long) Integer.MAX_VALUE).lcmp().branch(Opcode.IFGT, fallback);
        }
        code.aload(geometrySlot).loadConstant(inputLayout + 1).laload()
                .aload(geometrySlot).loadConstant(inputLayout + 2).laload()
                .aload(geometrySlot).loadConstant(inputLayout + 4).laload().lmul().ladd()
                .loadConstant(1L << 31).lcmp().branch(Opcode.IFGT, fallback);
        code.aload(geometrySlot).loadConstant(outputLayout + 1).laload()
                .aload(geometrySlot).loadConstant(outputLayout + 2).laload()
                .aload(geometrySlot).loadConstant(outputLayout + 3).laload().lmul().ladd()
                .loadConstant(1L << 31).lcmp().branch(Opcode.IFGT, fallback);
        emitZeroStrideAny(code, specialization, geometrySlot, startSlot, endSlot,
                inputLayout, outputLayout);
        code.branch(Opcode.GOTO, done);
        code.labelBinding(fallback);
        emitGeneral(code, specialization, DataType.BOOL, 3, identity, inRank, outRank,
                false, 0, geometrySlot, startSlot, endSlot);
        code.labelBinding(done);
    }

    private static void emitZeroStrideAny(CodeBuilder code,
            CpuKernelSpecialization specialization, int geometrySlot, int startSlot, int endSlot,
            int inputLayout, int outputLayout) {
        int inputBase = code.allocateLocal(TypeKind.INT);
        int inputStride = code.allocateLocal(TypeKind.INT);
        int outputBase = code.allocateLocal(TypeKind.INT);
        int outputStride = code.allocateLocal(TypeKind.INT);
        int domainCount = code.allocateLocal(TypeKind.INT);
        int cell = code.allocateLocal(TypeKind.INT);
        int end = code.allocateLocal(TypeKind.INT);
        code.aload(geometrySlot).loadConstant(inputLayout + 1).laload().l2i().istore(inputBase);
        code.aload(geometrySlot).loadConstant(inputLayout + 4).laload().l2i().istore(inputStride);
        code.aload(geometrySlot).loadConstant(outputLayout + 1).laload().l2i().istore(outputBase);
        code.aload(geometrySlot).loadConstant(outputLayout + 3).laload().l2i().istore(outputStride);
        code.aload(geometrySlot).loadConstant(7).laload().l2i().istore(domainCount);
        code.lload(startSlot).l2i().istore(cell);
        code.lload(endSlot).l2i().istore(end);
        var carriers = new CpuCarrierEmitter(code);
        var cells = code.newLabel();
        var finished = code.newLabel();
        code.labelBinding(cells);
        code.iload(cell).iload(end).branch(Opcode.IF_ICMPGE, finished);
        int inputAddress = code.allocateLocal(TypeKind.LONG);
        int outputAddress = code.allocateLocal(TypeKind.INT);
        code.iload(inputBase).i2l().iload(cell).i2l().iload(inputStride).i2l().lmul().ladd()
                .lstore(inputAddress);
        code.iload(outputBase).iload(cell).iload(outputStride).imul().iadd()
                .istore(outputAddress);
        int accumulator = code.allocateLocal(TypeKind.INT);
        int domain = code.allocateLocal(TypeKind.INT);
        code.loadConstant(0).istore(accumulator).loadConstant(0).istore(domain);
        var domains = code.newLabel();
        var write = code.newLabel();
        code.iload(domainCount).branch(Opcode.IFEQ, write);
        code.labelBinding(domains);
        carriers.load(DataType.BOOL, specialization.carrierPattern().getFirst(), 0,
                inputAddress);
        code.iload(accumulator).ior().istore(accumulator);
        code.iinc(domain, 1).iload(domain).iload(domainCount)
                .branch(Opcode.IF_ICMPLT, domains);
        code.labelBinding(write);
        var canonical = code.newLabel();
        var stored = code.newLabel();
        code.iload(accumulator).branch(Opcode.IFNE, canonical);
        code.loadConstant(0).istore(accumulator).branch(Opcode.GOTO, stored);
        code.labelBinding(canonical).loadConstant(1).istore(accumulator);
        code.labelBinding(stored);
        carriers.store(DataType.BOOL, specialization.carrierPattern().getLast(), 1,
                outputAddress, accumulator, true);
        code.iinc(cell, 1).branch(Opcode.GOTO, cells);
        code.labelBinding(finished);
    }

    private static boolean selectedSuffix(String identity, int rank) {
        boolean[] axes = selectedAxes(identity, rank); boolean selected = false;
        for (boolean axis : axes) { if (axis) selected = true; else if (selected) return false; }
        return selected;
    }

    private static void emitDenseCells(CodeBuilder code, CpuKernelSpecialization specialization,
            DataType type, int kind, int inRank, int outRank, boolean exactFloating,
            int exactLimbs, int geometrySlot, int startSlot, int endSlot) {
        int inputLayout = 11 + 2 * inRank + outRank;
        int outputLayout = inputLayout + 2 + 2 * inRank;
        int inputBase = code.allocateLocal(TypeKind.INT), outputBase = code.allocateLocal(TypeKind.INT);
        code.aload(geometrySlot).loadConstant(inputLayout + 1).laload().l2i().istore(inputBase);
        code.aload(geometrySlot).loadConstant(outputLayout + 1).laload().l2i().istore(outputBase);
        int cell = code.allocateLocal(TypeKind.INT); code.lload(startSlot).l2i().istore(cell);
        var done = code.newLabel(); var cells = code.newLabel();
        code.iload(cell).lload(endSlot).l2i().branch(Opcode.IF_ICMPGE, done).labelBinding(cells);
        int accumulator = code.allocateLocal(localKind(type));
        CpuExactSumEmitter exactSum = exactFloating && kind != 6
                ? new CpuExactSumEmitter(code, type, kind == 5, 2, geometrySlot, exactLimbs) : null;
        CpuExactProductEmitter exactProduct = exactFloating && kind == 6
                ? new CpuExactProductEmitter(code, type, 2, geometrySlot, 10, 9, true) : null;
        if (exactFloating) { emitFloatingZero(code, type); store(code, type, accumulator); }
        if (exactSum != null) exactSum.emitReset();
        else if (exactProduct != null) exactProduct.emitReset();
        else { emitIdentity(code, type, kind); store(code, type, accumulator); }
        int domain = code.allocateLocal(TypeKind.INT); code.loadConstant(0).istore(domain);
        var write = code.newLabel();
        code.aload(geometrySlot).loadConstant(7).laload().loadConstant(0L).lcmp()
                .branch(Opcode.IFEQ, write);
        var domains = code.newLabel(); code.labelBinding(domains);
        int address = code.allocateLocal(TypeKind.INT);
        code.iload(inputBase).iload(cell).aload(geometrySlot).loadConstant(7).laload().l2i()
                .imul().iadd().iload(domain).iadd().istore(address);
        int value = code.allocateLocal(localKind(type)); var carriers = new CpuCarrierEmitter(code);
        carriers.load(type, specialization.carrierPattern().getFirst(), 0, address, true);
        store(code, type, value);
        if (exactSum != null) exactSum.emitFactor(value);
        else if (exactProduct != null) exactProduct.emitFactor(value);
        else emitApply(code, type, kind, accumulator, value);
        if (type == DataType.BOOL) { load(code, type, accumulator);
            code.branch(kind == 2 ? Opcode.IFEQ : Opcode.IFNE, write); }
        code.iinc(domain, 1); code.iload(domain).aload(geometrySlot).loadConstant(7).laload().l2i()
                .branch(Opcode.IF_ICMPLT, domains).labelBinding(write);
        if (exactSum != null) exactSum.emitFinish(accumulator);
        else if (exactProduct != null) { int found = code.allocateLocal(TypeKind.INT);
            code.loadConstant(1).istore(found); exactProduct.emitFinish(accumulator, found); }
        int output = code.allocateLocal(TypeKind.INT); code.iload(outputBase).iload(cell).iadd()
                .istore(output);
        carriers.store(type, specialization.carrierPattern().getLast(), 1, output, accumulator, true);
        code.iinc(cell, 1); code.iload(cell).lload(endSlot).l2i().branch(Opcode.IF_ICMPLT, cells)
                .labelBinding(done);
    }

    private static void emitFullDense(CodeBuilder code, CpuKernelSpecialization specialization,
            DataType type, int kind, int inRank, boolean exactFloating, int exactLimbs,
            int geometrySlot, int startSlot, int endSlot) {
        var done = code.newLabel();
        code.lload(startSlot).lload(endSlot).lcmp().branch(Opcode.IFGE, done);
        int inputLayout = 11 + 2 * inRank;
        int outputLayout = inputLayout + 2 + 2 * inRank;
        int input = code.allocateLocal(TypeKind.INT), output = code.allocateLocal(TypeKind.INT);
        code.aload(geometrySlot).loadConstant(inputLayout + 1).laload().l2i().istore(input);
        code.aload(geometrySlot).loadConstant(outputLayout + 1).laload().l2i().istore(output);
        int accumulator = code.allocateLocal(localKind(type));
        CpuExactSumEmitter exactSum = exactFloating && kind != 6
                ? new CpuExactSumEmitter(code, type, kind == 5, 2, geometrySlot, exactLimbs) : null;
        CpuExactProductEmitter exactProduct = exactFloating && kind == 6
                ? new CpuExactProductEmitter(code, type, 2, geometrySlot, 10, 9, true) : null;
        if (exactFloating) { emitFloatingZero(code, type); store(code, type, accumulator); }
        if (exactSum != null) exactSum.emitReset();
        else if (exactProduct != null) exactProduct.emitReset();
        else { emitIdentity(code, type, kind); store(code, type, accumulator); }
        int domain = code.allocateLocal(TypeKind.INT); code.loadConstant(0).istore(domain);
        var store = code.newLabel();
        code.aload(geometrySlot).loadConstant(7).laload().loadConstant(0L).lcmp()
                .branch(Opcode.IFEQ, store);
        var loop = code.newLabel(); code.labelBinding(loop);
        int address = code.allocateLocal(TypeKind.INT);
        code.iload(input).iload(domain).iadd().istore(address);
        int value = code.allocateLocal(localKind(type));
        var carriers = new CpuCarrierEmitter(code);
        carriers.load(type, specialization.carrierPattern().getFirst(), 0, address, true);
        store(code, type, value);
        if (exactSum != null) exactSum.emitFactor(value);
        else if (exactProduct != null) exactProduct.emitFactor(value);
        else emitApply(code, type, kind, accumulator, value);
        if (type == DataType.BOOL) {
            load(code, type, accumulator);
            code.branch(kind == 2 ? Opcode.IFEQ : Opcode.IFNE, store);
        }
        code.iinc(domain, 1); code.iload(domain).aload(geometrySlot).loadConstant(7).laload().l2i()
                .branch(Opcode.IF_ICMPLT, loop);
        code.labelBinding(store);
        if (exactSum != null) exactSum.emitFinish(accumulator);
        else if (exactProduct != null) {
            int found = code.allocateLocal(TypeKind.INT); code.loadConstant(1).istore(found);
            exactProduct.emitFinish(accumulator, found);
        }
        carriers.store(type, specialization.carrierPattern().getLast(), 1, output, accumulator, true);
        code.labelBinding(done);
    }

    private static void emitGeneral(CodeBuilder code, CpuKernelSpecialization specialization,
            DataType type, int kind, String identity, int inRank, int outRank,
            boolean exactFloating, int exactLimbs, int geometrySlot, int startSlot, int endSlot) {
        boolean keep = identity.contains(":keep=true:");
        boolean sumToShape = identity.contains(":SUM_TO_SHAPE:");
        boolean[] selectedAxes = selectedAxes(identity, inRank);
        boolean controlled = CpuCarrierEmitter.controlFacts().isPresent();
        boolean represented = kind == 4 && (type == DataType.FLOAT32 || type == DataType.FLOAT64)
                && controlledTerminalRowSum(selectedAxes);
        int selected = 11, inputCoordinates = selected + inRank;
        int outputCoordinates = inputCoordinates + inRank;
        int inputLayout = outputCoordinates + outRank;
        int outputLayout = inputLayout + 2 + 2 * inRank;
        int cell = code.allocateLocal(TypeKind.LONG);
        code.lload(startSlot).lstore(cell);
        var done = code.newLabel();
        code.lload(cell).lload(endSlot).lcmp().branch(Opcode.IFGE, done);
        if (sumToShape) decode(code, geometrySlot, cell, outputCoordinates, outputLayout, outRank);
        var cells = code.newLabel(); code.labelBinding(cells);
        if (!sumToShape) decode(code, geometrySlot, cell, outputCoordinates, outputLayout, outRank);
        var nextCell = controlled ? code.newLabel() : null;
        if (controlled)
            CpuLoopEmitter.emitActiveOrdinalGuard(code, 1, cell, TypeKind.LONG, nextCell);
        int outAxis = 0;
        int leading = inRank - outRank;
        for (int axis = 0; axis < inRank; axis++) {
            code.aload(geometrySlot).loadConstant(inputCoordinates + axis);
            if (selectedAxes[axis]) code.loadConstant(0L);
            else if (sumToShape) code.aload(geometrySlot).loadConstant(
                    outputCoordinates + axis - leading).laload();
            else code.aload(geometrySlot).loadConstant(
                    outputCoordinates + (keep ? axis : outAxis++)).laload();
            code.lastore();
        }
        int accumulator = code.allocateLocal(localKind(type));
        CpuExactSumEmitter exactSum = exactFloating && kind != 6 && !represented
                ? new CpuExactSumEmitter(code, type, kind == 5, 2, geometrySlot, exactLimbs) : null;
        CpuExactProductEmitter exactProduct = exactFloating && kind == 6
                ? new CpuExactProductEmitter(code, type, 2, geometrySlot, 10, 9, true) : null;
        if (controlled && sumToShape) {
            // Domain one is an ordinary geometry fact, not an active-contributor count.
            var fold = code.newLabel();
            geometry(code, geometrySlot, 7).loadConstant(1L).lcmp().branch(Opcode.IFNE, fold);
            emitControlledContributorGuard(code, geometrySlot, inputCoordinates, nextCell);
            int source = address(code, geometrySlot, inRank, inputCoordinates, inputLayout);
            int target = address(code, geometrySlot, outRank, outputCoordinates, outputLayout);
            var copy = new CpuCarrierEmitter(code);
            copy.load(type, specialization.carrierPattern().getFirst(), 0, source);
            store(code, type, accumulator);
            copy.store(type, specialization.carrierPattern().getLast(), 1, target, accumulator);
            code.branch(Opcode.GOTO, nextCell).labelBinding(fold);
        }
        if (exactFloating && !represented) { emitFloatingZero(code, type); store(code, type, accumulator); }
        if (represented) emitRepresentedInit(code, type, accumulator, false);
        else if (exactSum != null) exactSum.emitReset();
        else if (exactProduct != null) exactProduct.emitReset();
        else { emitIdentity(code, type, kind); store(code, type, accumulator); }
        int domain = code.allocateLocal(TypeKind.LONG); code.loadConstant(0L).lstore(domain);
        var write = code.newLabel();
        code.aload(geometrySlot).loadConstant(7).laload().loadConstant(0L).lcmp()
                .branch(Opcode.IFEQ, write);
        var domains = code.newLabel(); code.labelBinding(domains);
        if (!sumToShape) {
            int remaining = code.allocateLocal(TypeKind.LONG); code.lload(domain).lstore(remaining);
            for (int axis = inRank - 1; axis >= 0; axis--) if (selectedAxes[axis]) {
                code.aload(geometrySlot).loadConstant(inputCoordinates + axis).lload(remaining)
                        .aload(geometrySlot).loadConstant(inputLayout + 2 + axis).laload().lrem()
                        .lastore();
                code.lload(remaining).aload(geometrySlot).loadConstant(inputLayout + 2 + axis)
                        .laload().ldiv().lstore(remaining);
            }
        }
        var nextContributor = controlled ? code.newLabel() : null;
        if (controlled)
            emitControlledContributorGuard(code, geometrySlot, inputCoordinates, nextContributor);
        int inputAddress = address(code, geometrySlot, inRank, inputCoordinates, inputLayout);
        int value = code.allocateLocal(localKind(type));
        var carriers = new CpuCarrierEmitter(code);
        if (represented) {
            emitRepresentedContributorStep(code, type, accumulator, false,
                    () -> carriers.load(type, specialization.carrierPattern().getFirst(), 0, inputAddress));
        } else {
            carriers.load(type, specialization.carrierPattern().getFirst(), 0, inputAddress);
            store(code, type, value);
            if (exactSum != null) exactSum.emitFactor(value);
            else if (exactProduct != null) exactProduct.emitFactor(value);
            else emitApply(code, type, kind, accumulator, value);
        }
        if (type == DataType.BOOL) {
            load(code, type, accumulator);
            code.branch(kind == 2 ? Opcode.IFEQ : Opcode.IFNE, write);
        }
        if (controlled) code.labelBinding(nextContributor);
        code.lload(domain).loadConstant(1L).ladd().lstore(domain);
        if (sumToShape) {
            var advance = code.newLabel();
            code.lload(domain).aload(geometrySlot).loadConstant(7).laload().lcmp()
                    .branch(Opcode.IFLT, advance).branch(Opcode.GOTO, write)
                    .labelBinding(advance);
            incrementCoordinates(code, geometrySlot, inputCoordinates, inputLayout, inRank,
                    selectedAxes);
            code.branch(Opcode.GOTO, domains);
        } else {
            code.lload(domain).aload(geometrySlot).loadConstant(7).laload().lcmp()
                    .branch(Opcode.IFLT, domains);
        }
        code.labelBinding(write);
        if (exactSum != null) exactSum.emitFinish(accumulator);
        else if (exactProduct != null) {
            int found = code.allocateLocal(TypeKind.INT); code.loadConstant(1).istore(found);
            exactProduct.emitFinish(accumulator, found);
        }
        int outputAddress = address(code, geometrySlot, outRank, outputCoordinates, outputLayout);
        carriers.store(type, specialization.carrierPattern().getLast(), 1, outputAddress, accumulator);
        if (controlled) code.labelBinding(nextCell);
        code.lload(cell).loadConstant(1L).ladd().lstore(cell);
        if (sumToShape) {
            code.lload(cell).lload(endSlot).lcmp().branch(Opcode.IFGE, done);
            boolean[] everyOutputAxis = new boolean[outRank];
            java.util.Arrays.fill(everyOutputAxis, true);
            incrementCoordinates(code, geometrySlot, outputCoordinates, outputLayout, outRank,
                    everyOutputAxis);
            code.branch(Opcode.GOTO, cells);
        } else code.lload(cell).lload(endSlot).lcmp().branch(Opcode.IFLT, cells);
        code.labelBinding(done);
    }

    /**
     * Recognizes a terminal SUM from exact boundary provenance and selected axes, never extents.
     * @param selected selected ordinary input axes
     * @return whether the input's proved ROW contracts into a SHARED output
     */
    private static boolean controlledTerminalRowSum(boolean[] selected) {
        var facts = CpuCarrierEmitter.controlFacts();
        if (facts.isEmpty()) return false;
        var input = facts.orElseThrow().boundaryRows().getFirst();
        return input.rowIndexed() && selected[input.axis()]
                && !facts.orElseThrow().boundaryRows().getLast().rowIndexed();
    }

    /**
     * Guards the current complete ordinary input coordinate before addresses or loads.
     * SHARED values emit no guard, and empty domains never enter this contributor site.
     * @param code current controlled generation builder
     * @param geometry existing aggregate numeric pack
     * @param coordinates first input-coordinate cell in that pack
     * @param inactive target preceding canonical contributor advancement
     */
    private static void emitControlledContributorGuard(CodeBuilder code, int geometry,
            int coordinates, java.lang.classfile.Label inactive) {
        var row = CpuCarrierEmitter.controlFacts().orElseThrow().boundaryRows().getFirst();
        if (!row.rowIndexed()) return;
        int originalRow = code.allocateLocal(TypeKind.LONG);
        geometry(code, geometry, coordinates + row.axis()).lstore(originalRow);
        CpuCarrierEmitter.emitActiveRowGuard(code, originalRow, TypeKind.LONG, inactive);
    }

    private static void incrementCoordinates(CodeBuilder code, int geometrySlot,
            int coordinates, int layout, int rank, boolean[] selectedAxes) {
        var complete = code.newLabel();
        for (int axis = rank - 1; axis >= 0; axis--) if (selectedAxes[axis]) {
            code.aload(geometrySlot).loadConstant(coordinates + axis)
                    .aload(geometrySlot).loadConstant(coordinates + axis).laload()
                    .loadConstant(1L).ladd().lastore();
            code.aload(geometrySlot).loadConstant(coordinates + axis).laload()
                    .aload(geometrySlot).loadConstant(layout + 2 + axis).laload().lcmp()
                    .branch(Opcode.IFLT, complete);
            code.aload(geometrySlot).loadConstant(coordinates + axis).loadConstant(0L).lastore();
        }
        code.labelBinding(complete);
    }

    private static boolean[] selectedAxes(String identity, int rank) {
        boolean[] result = new boolean[rank];
        int start = identity.indexOf(":axes=[") + 7, end = identity.indexOf(']', start);
        String body = identity.substring(start, end).trim();
        if (!body.isEmpty()) for (String axis : body.split(", ")) result[Integer.parseInt(axis)] = true;
        return result;
    }

    private static int identityNumber(String identity, String marker) {
        int start = identity.indexOf(marker) + marker.length();
        int end = identity.indexOf(':', start);
        return Integer.parseInt(identity.substring(start, end));
    }

    private static void decode(CodeBuilder code, int geometry, int logical, int coordinates,
            int layout, int rank) {
        int remaining = code.allocateLocal(TypeKind.LONG); code.lload(logical).lstore(remaining);
        for (int axis = rank - 1; axis >= 0; axis--) {
            code.aload(geometry).loadConstant(coordinates + axis).lload(remaining)
                    .aload(geometry).loadConstant(layout + 2 + axis).laload().lrem().lastore();
            code.lload(remaining).aload(geometry).loadConstant(layout + 2 + axis).laload()
                    .ldiv().lstore(remaining);
        }
    }

    private static int address(CodeBuilder code, int geometry, int rank, int coordinates, int layout) {
        int address = code.allocateLocal(TypeKind.LONG);
        code.aload(geometry).loadConstant(layout + 1).laload().lstore(address);
        for (int axis = 0; axis < rank; axis++) code.lload(address)
                .aload(geometry).loadConstant(coordinates + axis).laload()
                .aload(geometry).loadConstant(layout + 2 + rank + axis).laload().lmul().ladd()
                .lstore(address);
        return address;
    }

    private static TypeKind localKind(DataType type) { return switch (type) {
        case FLOAT64 -> TypeKind.DOUBLE; case FLOAT32 -> TypeKind.FLOAT;
        case INT64 -> TypeKind.LONG; case BFLOAT16, FLOAT16, FLOAT8_E4M3, FLOAT8_E5M2, INT32, BOOL -> TypeKind.INT;
    }; }
    private static void emitFloatingZero(CodeBuilder code, DataType type) { switch (type) {
        case FLOAT64 -> code.loadConstant(0.0d); case FLOAT32 -> code.loadConstant(0.0f);
        case BFLOAT16, FLOAT16, FLOAT8_E4M3, FLOAT8_E5M2 -> code.loadConstant(0);
        default -> throw new IllegalArgumentException();
    } }
    private static void store(CodeBuilder code, DataType type, int local) { switch (type) {
        case FLOAT64 -> code.dstore(local); case FLOAT32 -> code.fstore(local);
        case INT64 -> code.lstore(local); case BFLOAT16, FLOAT16, FLOAT8_E4M3, FLOAT8_E5M2, INT32, BOOL -> code.istore(local);
    } }
    private static void load(CodeBuilder code, DataType type, int local) { switch (type) {
        case FLOAT64 -> code.dload(local); case FLOAT32 -> code.fload(local);
        case INT64 -> code.lload(local); case BFLOAT16, FLOAT16, FLOAT8_E4M3, FLOAT8_E5M2, INT32, BOOL -> code.iload(local);
    } }
    private static void emitIdentity(CodeBuilder code, DataType type, int kind) { switch (type) {
        case FLOAT64 -> code.loadConstant(kind == 0 ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY);
        case FLOAT32 -> code.loadConstant(kind == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY);
        case BFLOAT16 -> code.loadConstant(kind == 0 ? 0x7f80 : 0xff80);
        case FLOAT16 -> code.loadConstant(kind == 0 ? 0x7c00 : 0xfc00);
        case FLOAT8_E4M3 -> code.loadConstant(0x17f); // Unseen sentinel; empty output narrows to NaN.
        case FLOAT8_E5M2 -> code.loadConstant(kind == 0 ? 0x7c : 0xfc);
        case INT32 -> code.loadConstant(kind == 4 ? 0 : kind == 6 ? 1
                : kind == 0 ? Integer.MAX_VALUE : Integer.MIN_VALUE);
        case INT64 -> code.loadConstant(kind == 4 ? 0L : kind == 6 ? 1L
                : kind == 0 ? Long.MAX_VALUE : Long.MIN_VALUE);
        case BOOL -> code.loadConstant(kind == 2 ? 1 : 0);
    } }
    private static void emitApply(CodeBuilder code, DataType type, int kind,
            int accumulator, int value) {
        if ((type == DataType.INT32 || type == DataType.INT64) && (kind == 4 || kind == 6)) {
            load(code, type, accumulator); load(code, type, value);
            if (type == DataType.INT32) {
                if (kind == 4) code.iadd(); else code.imul();
            } else if (kind == 4) code.ladd(); else code.lmul();
            store(code, type, accumulator);
            return;
        }
        switch (type) {
            case FLOAT64 -> emitFloatingSelection(code, true, kind == 0, accumulator, value);
            case FLOAT32 -> emitFloatingSelection(code, false, kind == 0, accumulator, value);
            case BFLOAT16, FLOAT16, FLOAT8_E4M3, FLOAT8_E5M2 -> emitShortFloatSelection(code, type, kind == 0,
                    accumulator, value);
            case INT32 -> emitIntSelection(code, kind == 0, accumulator, value);
            case INT64 -> emitLongSelection(code, kind == 0, accumulator, value);
            case BOOL -> emitBooleanCombination(code, kind == 2, accumulator, value);
        }
    }

    private static void emitIntSelection(CodeBuilder code, boolean minimum, int left, int right) {
        code.iload(left).iload(right).invokestatic(ClassDescHolder.MATH,
                minimum ? "min" : "max", java.lang.constant.MethodTypeDesc.of(
                        ConstantDescs.CD_int, ConstantDescs.CD_int, ConstantDescs.CD_int))
                .istore(left);
    }

    private static void emitLongSelection(CodeBuilder code, boolean minimum, int left, int right) {
        code.lload(left).lload(right).invokestatic(ClassDescHolder.MATH,
                minimum ? "min" : "max", java.lang.constant.MethodTypeDesc.of(
                        ConstantDescs.CD_long, ConstantDescs.CD_long, ConstantDescs.CD_long))
                .lstore(left);
    }

    private static void emitBooleanCombination(CodeBuilder code, boolean all, int left, int right) {
        var falseResult = code.newLabel(); var trueResult = code.newLabel(); var done = code.newLabel();
        if (all) {
            code.iload(left).branch(Opcode.IFEQ, falseResult);
            code.iload(right).branch(Opcode.IFEQ, falseResult);
            code.labelBinding(trueResult).loadConstant(1).branch(Opcode.GOTO, done);
            code.labelBinding(falseResult).loadConstant(0);
        } else {
            code.iload(left).branch(Opcode.IFNE, trueResult);
            code.iload(right).branch(Opcode.IFNE, trueResult);
            code.loadConstant(0).branch(Opcode.GOTO, done);
            code.labelBinding(trueResult).loadConstant(1);
        }
        code.labelBinding(done).istore(left);
    }

    private static void emitFloatingSelection(CodeBuilder code, boolean binary64,
            boolean minimum, int left, int right) {
        var leftNumber = code.newLabel(); var rightNumber = code.newLabel();
        var ordinary = code.newLabel(); var chooseRight = code.newLabel(); var chooseLeft = code.newLabel();
        var done = code.newLabel();
        if (binary64) {
            code.dload(left).invokestatic(ClassDescHolder.DOUBLE, "isNaN",
                    java.lang.constant.MethodTypeDesc.of(ConstantDescs.CD_boolean,
                            ConstantDescs.CD_double)).branch(Opcode.IFEQ, leftNumber);
            code.dload(left).branch(Opcode.GOTO, done).labelBinding(leftNumber);
            code.dload(right).invokestatic(ClassDescHolder.DOUBLE, "isNaN",
                    java.lang.constant.MethodTypeDesc.of(ConstantDescs.CD_boolean,
                            ConstantDescs.CD_double)).branch(Opcode.IFEQ, rightNumber);
            code.dload(right).branch(Opcode.GOTO, done).labelBinding(rightNumber);
            code.dload(left).loadConstant(0.0d).dcmpl().branch(Opcode.IFNE, ordinary);
            code.dload(right).loadConstant(0.0d).dcmpl().branch(Opcode.IFNE, ordinary);
            code.dload(left).invokestatic(ClassDescHolder.DOUBLE, "doubleToRawLongBits",
                    java.lang.constant.MethodTypeDesc.of(ConstantDescs.CD_long,
                            ConstantDescs.CD_double)).loadConstant(0L).lcmp()
                    .branch(minimum ? Opcode.IFLT : Opcode.IFGE, chooseLeft);
            code.branch(Opcode.GOTO, chooseRight).labelBinding(ordinary);
            code.dload(right).dload(left).dcmpg().branch(
                    minimum ? Opcode.IFLT : Opcode.IFGT, chooseRight);
            code.labelBinding(chooseLeft).dload(left).branch(Opcode.GOTO, done);
            code.labelBinding(chooseRight).dload(right);
            code.labelBinding(done).dstore(left);
        } else {
            code.fload(left).invokestatic(ClassDescHolder.FLOAT, "isNaN",
                    java.lang.constant.MethodTypeDesc.of(ConstantDescs.CD_boolean,
                            ConstantDescs.CD_float)).branch(Opcode.IFEQ, leftNumber);
            code.fload(left).branch(Opcode.GOTO, done).labelBinding(leftNumber);
            code.fload(right).invokestatic(ClassDescHolder.FLOAT, "isNaN",
                    java.lang.constant.MethodTypeDesc.of(ConstantDescs.CD_boolean,
                            ConstantDescs.CD_float)).branch(Opcode.IFEQ, rightNumber);
            code.fload(right).branch(Opcode.GOTO, done).labelBinding(rightNumber);
            code.fload(left).loadConstant(0.0f).fcmpl().branch(Opcode.IFNE, ordinary);
            code.fload(right).loadConstant(0.0f).fcmpl().branch(Opcode.IFNE, ordinary);
            code.fload(left).invokestatic(ClassDescHolder.FLOAT, "floatToRawIntBits",
                    java.lang.constant.MethodTypeDesc.of(ConstantDescs.CD_int,
                            ConstantDescs.CD_float)).branch(minimum ? Opcode.IFLT : Opcode.IFGE,
                            chooseLeft);
            code.branch(Opcode.GOTO, chooseRight).labelBinding(ordinary);
            code.fload(right).fload(left).fcmpg().branch(
                    minimum ? Opcode.IFLT : Opcode.IFGT, chooseRight);
            code.labelBinding(chooseLeft).fload(left).branch(Opcode.GOTO, done);
            code.labelBinding(chooseRight).fload(right);
            code.labelBinding(done).fstore(left);
        }
    }

    /** Emits exact represented FP8 winner selection without an encoded E4M3 infinity seed. */
    private static void emitFp8Selection(CodeBuilder code, DataType type,
            boolean minimum, int left, int right) {
        int leftValue = code.allocateLocal(TypeKind.FLOAT), rightValue = code.allocateLocal(TypeKind.FLOAT);
        var chooseRight = code.newLabel(); var done = code.newLabel(); var ordinary = code.newLabel();
        if (type == DataType.FLOAT8_E4M3)
            code.iload(left).loadConstant(0x17f).branch(Opcode.IF_ICMPEQ, chooseRight);
        CpuCastEmitter.emitLowToFloat(code, type, left); code.fstore(leftValue);
        CpuCastEmitter.emitLowToFloat(code, type, right); code.fstore(rightValue);
        code.fload(leftValue).fload(leftValue).fcmpl().branch(Opcode.IFNE, done);
        code.fload(rightValue).fload(rightValue).fcmpl().branch(Opcode.IFNE, chooseRight);
        code.iload(left).loadConstant(0x7f).iand().branch(Opcode.IFNE, ordinary);
        code.iload(right).loadConstant(0x7f).iand().branch(Opcode.IFNE, ordinary);
        code.iload(left).loadConstant(0x80).iand().branch(minimum ? Opcode.IFNE : Opcode.IFEQ, done);
        code.branch(Opcode.GOTO, chooseRight).labelBinding(ordinary);
        code.fload(rightValue).fload(leftValue).fcmpg()
                .branch(minimum ? Opcode.IFLT : Opcode.IFGT, chooseRight);
        code.branch(Opcode.GOTO, done).labelBinding(chooseRight);
        code.iload(right).istore(left).labelBinding(done);
    }

    /**
     * Selects one exact raw low-format winner, retaining first NaN and signed-zero preferences.
     * @param code nonnull method builder
     * @param type one of the four low storage formats
     * @param minimum whether to select the smaller represented value
     * @param left int raw-bit accumulator local, replaced only when the right operand wins
     * @param right int raw-bit contributor local
     */
    static void emitShortFloatSelection(CodeBuilder code, DataType type,
            boolean minimum, int left, int right) {
        if (type == DataType.FLOAT8_E4M3 || type == DataType.FLOAT8_E5M2) {
            emitFp8Selection(code, type, minimum, left, right);
            return;
        }
        int exponentMask = type == DataType.FLOAT16 ? 0x7c00 : 0x7f80;
        int fractionMask = type == DataType.FLOAT16 ? 0x03ff : 0x007f;
        var leftNumber = code.newLabel(); var rightNumber = code.newLabel();
        var ordinary = code.newLabel(); var chooseRight = code.newLabel(); var chooseLeft = code.newLabel();
        var done = code.newLabel();
        code.iload(left).loadConstant(0xffff).iand().istore(left);
        code.iload(right).loadConstant(0xffff).iand().istore(right);
        code.iload(left).loadConstant(exponentMask).iand().loadConstant(exponentMask)
                .branch(Opcode.IF_ICMPNE, leftNumber);
        code.iload(left).loadConstant(fractionMask).iand().branch(Opcode.IFEQ, leftNumber);
        code.iload(left).branch(Opcode.GOTO, done).labelBinding(leftNumber);
        code.iload(right).loadConstant(exponentMask).iand().loadConstant(exponentMask)
                .branch(Opcode.IF_ICMPNE, rightNumber);
        code.iload(right).loadConstant(fractionMask).iand().branch(Opcode.IFEQ, rightNumber);
        code.iload(right).branch(Opcode.GOTO, done).labelBinding(rightNumber);
        code.iload(left).loadConstant(0x7fff).iand().branch(Opcode.IFNE, ordinary);
        code.iload(right).loadConstant(0x7fff).iand().branch(Opcode.IFNE, ordinary);
        code.iload(left).i2s().branch(minimum ? Opcode.IFLT : Opcode.IFGE, chooseLeft);
        code.branch(Opcode.GOTO, chooseRight).labelBinding(ordinary);
        emitShortFloatDecode(code, type, right);
        emitShortFloatDecode(code, type, left);
        code.fcmpg().branch(minimum ? Opcode.IFLT : Opcode.IFGT, chooseRight);
        code.labelBinding(chooseLeft).iload(left).branch(Opcode.GOTO, done);
        code.labelBinding(chooseRight).iload(right);
        code.labelBinding(done).istore(left);
    }

    private static void emitShortFloatDecode(CodeBuilder code, DataType type, int local) {
        if (type == DataType.FLOAT16) {
            code.iload(local).i2s().invokestatic(ClassDescHolder.FLOAT, "float16ToFloat",
                    java.lang.constant.MethodTypeDesc.of(ConstantDescs.CD_float,
                            ConstantDescs.CD_short));
        } else {
            code.iload(local).loadConstant(16).ishl().invokestatic(ClassDescHolder.FLOAT,
                    "intBitsToFloat", java.lang.constant.MethodTypeDesc.of(
                            ConstantDescs.CD_float, ConstantDescs.CD_int));
        }
    }

    private static final class ClassDescHolder {
        private static final java.lang.constant.ClassDesc DOUBLE =
                java.lang.constant.ClassDesc.of(Double.class.getName());
        private static final java.lang.constant.ClassDesc FLOAT =
                java.lang.constant.ClassDesc.of(Float.class.getName());
        private static final java.lang.constant.ClassDesc MATH =
                java.lang.constant.ClassDesc.of(Math.class.getName());
        private static final java.lang.constant.ClassDesc LONG =
                java.lang.constant.ClassDesc.of(Long.class.getName());
        private static final java.lang.constant.ClassDesc SEGMENT =
                java.lang.constant.ClassDesc.of(MemorySegment.class.getName());
        private static final java.lang.constant.ClassDesc VALUE_LAYOUT =
                java.lang.constant.ClassDesc.of(ValueLayout.class.getName());
        private static final java.lang.constant.ClassDesc LONG_LAYOUT =
                java.lang.constant.ClassDesc.of("java.lang.foreign.ValueLayout$OfLong");
        private static final java.lang.constant.ClassDesc SHORT_LAYOUT =
                java.lang.constant.ClassDesc.of("java.lang.foreign.ValueLayout$OfShort");
        private ClassDescHolder() { }
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
        long domainCount = packed[7]; int selected = 11;
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
            case FLOAT16 -> kind == 0 ? 0x7c00L : 0xfc00L;
            case FLOAT8_E4M3 -> 0x17fL;
            case FLOAT8_E5M2 -> kind == 0 ? 0x7cL : 0xfcL;
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
        if (type == DataType.FLOAT8_E4M3 && left == 0x17fL) return right;
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
            case FLOAT16 -> ((bits & 0x7c00L) == 0x7c00L) && (bits & 0x3ffL) != 0;
            case FLOAT8_E4M3 -> (bits & 0x7fL) == 0x7fL;
            case FLOAT8_E5M2 -> (bits & 0x7cL) == 0x7cL && (bits & 3L) != 0;
            default -> false;
        };
    }
    private static double floatingValue(long bits, DataType type) {
        return switch (type) {
            case FLOAT64 -> Double.longBitsToDouble(bits);
            case FLOAT32 -> Float.intBitsToFloat((int) bits);
            case BFLOAT16 -> Float.intBitsToFloat((int) bits << 16);
            case FLOAT16 -> Float.float16ToFloat((short) bits);
            case FLOAT8_E4M3 -> io.github.pho001.synaptik.model.datatype.Float8E4M3Bits.toFloat((byte) bits);
            case FLOAT8_E5M2 -> io.github.pho001.synaptik.model.datatype.Float8E5M2Bits.toFloat((byte) bits);
            default -> throw new AssertionError("non-floating aggregate type");
        };
    }
    private static boolean negative(long bits, DataType type) {
        return switch (type) {
            case FLOAT64 -> bits < 0;
            case FLOAT32 -> ((int) bits) < 0;
            case BFLOAT16, FLOAT16 -> ((short) bits) < 0;
            case FLOAT8_E4M3, FLOAT8_E5M2 -> ((byte) bits) < 0;
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
            case BFLOAT16, FLOAT16 -> Short.toUnsignedLong(carrier instanceof short[] a ? a[Math.toIntExact(address)] : ((MemorySegment) carrier).get(ValueLayout.JAVA_SHORT, address * 2));
            case INT32 -> carrier instanceof int[] a ? a[Math.toIntExact(address)] : ((MemorySegment) carrier).get(ValueLayout.JAVA_INT, address * 4);
            case INT64 -> carrier instanceof long[] a ? a[Math.toIntExact(address)] : ((MemorySegment) carrier).get(ValueLayout.JAVA_LONG, address * 8);
            case BOOL, FLOAT8_E4M3, FLOAT8_E5M2 -> carrier instanceof byte[] a ? Byte.toUnsignedLong(a[Math.toIntExact(address)]) : Byte.toUnsignedLong(((MemorySegment) carrier).get(ValueLayout.JAVA_BYTE, address));
        };
    }
    private static void writeBits(Object carrier, long address, DataType type, long bits) {
        switch (type) {
            case FLOAT64 -> { double v = Double.longBitsToDouble(bits); if (carrier instanceof double[] a) a[Math.toIntExact(address)] = v; else ((MemorySegment) carrier).set(ValueLayout.JAVA_DOUBLE, address * 8, v); }
            case FLOAT32 -> { float v = Float.intBitsToFloat((int) bits); if (carrier instanceof float[] a) a[Math.toIntExact(address)] = v; else ((MemorySegment) carrier).set(ValueLayout.JAVA_FLOAT, address * 4, v); }
            case BFLOAT16, FLOAT16 -> { short v = (short) bits; if (carrier instanceof short[] a) a[Math.toIntExact(address)] = v; else ((MemorySegment) carrier).set(ValueLayout.JAVA_SHORT, address * 2, v); }
            case INT32 -> { int v = (int) bits; if (carrier instanceof int[] a) a[Math.toIntExact(address)] = v; else ((MemorySegment) carrier).set(ValueLayout.JAVA_INT, address * 4, v); }
            case INT64 -> { if (carrier instanceof long[] a) a[Math.toIntExact(address)] = bits; else ((MemorySegment) carrier).set(ValueLayout.JAVA_LONG, address * 8, bits); }
            case FLOAT8_E4M3, FLOAT8_E5M2 -> { byte v = (byte) bits; if (carrier instanceof byte[] a) a[Math.toIntExact(address)] = v; else ((MemorySegment) carrier).set(ValueLayout.JAVA_BYTE, address, v); }
            case BOOL -> { byte v = (byte) (bits == 0 ? 0 : 1); if (carrier instanceof byte[] a) a[Math.toIntExact(address)] = v; else ((MemorySegment) carrier).set(ValueLayout.JAVA_BYTE, address, v); }
        }
    }
}

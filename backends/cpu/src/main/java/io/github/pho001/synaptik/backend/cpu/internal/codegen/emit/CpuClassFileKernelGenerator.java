package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratorSchema;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.ClassFile;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.AccessFlag;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Stateless Java 26 Class-File generator for one typed portable CPU unit.
 *
 * <p>It realizes the already-selected scalar body or eligible preferred-species
 * FLOAT32/FLOAT64, signed-integral, canonical-BOOL, or narrowly virtual floating-mask vector
 * body, and retains the scalar body for tails. It does not choose capability, numerical
 * semantics, access structure, strategy, or fallback.</p>
 * Instruction-free affine, movement, indexing, scatter, fold, ordering, explicit-state random,
 * cumulative-scan, and ordinary aggregate forms delegate to their focused emitters after
 * structural specialization checks. Dense heap-array pointwise, affine-copy, movement, and
 * indexing bodies use one-time integer narrowing and hoisted geometry; indexing, functional
 * scatter, overlap-fold, ordering, scan, and aggregate emitters embed typed family hot loops.
 * Functional-scatter classes additionally embed their selected reduction and optional
 * exact-product scratch state. Fold classes embed coordinate mapping and sequential represented
 * addition without a generic carrier bridge. General layouts and segment or mixed carriers retain
 * typed long-address fallbacks. Ordering classes additionally embed stable merge, comparison,
 * selected-pair ordering, and represented value/index stores over assigned scratch. Random
 * classes embed their typed state prologue, counter mapping, threshold, represented value,
 * canonical mask, and dense-integer or general-long element loop. One completely guarded
 * pointwise form replaces the frozen {@code [512,512]} FLOAT32 mixed-carrier
 * {@code DIV -> SIGMOID -> MUL} general odometer with direct ordinal-derived row/column
 * addresses. Its scalar body preserves FLOAT32 division and multiplication around the stable
 * sign-branch sigmoid, performs {@link StrictMath#exp(double)} work in binary64, and narrows the
 * sigmoid result to FLOAT32 exactly once before multiplication. Complete topology, carrier,
 * geometry, ordered-range, start-address, and sentinel guards precede the narrowed loop; every
 * failed proof enters the unchanged typed general-long state machine.
 */
public final class CpuClassFileKernelGenerator {
    private static final ClassDesc STRICT_MATH = ClassDesc.of(StrictMath.class.getName());
    private static final ClassDesc SEGMENT = ClassDesc.of("java.lang.foreign.MemorySegment");
    private static final ClassDesc VALUE_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout");
    private static final ClassDesc FLOAT_LAYOUT =
            ClassDesc.of("java.lang.foreign.ValueLayout$OfFloat");
    /** Creates a stateless generator with no retained route or specialization state. */
    public CpuClassFileKernelGenerator() { }

    /**
     * Emits deterministic verified bytes for one exact structural specialization.
     *
     * @param specialization non-null typed carrier, strategy, and compatibility facts
     * @param kernelIr non-null canonical typed CPU kernel IR matching the specialization
     * @return a new deterministic verified class-byte array
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if specialization and IR facts disagree
     */
    public byte[] generateClassBytes(CpuKernelSpecialization specialization, CpuKernelIr kernelIr) {
        validate(specialization, kernelIr);
        ClassDesc owner = ClassDesc.of(CpuGeneratorSchema.generatedBinaryName(specialization));
        MethodTypeDesc type = MethodTypeDesc.ofDescriptor(specialization.entryType().descriptorString());
        int entryFlags = AccessFlag.STATIC.mask()
                | (kernelIr.familyIdentity().startsWith("arg-extrema:")
                    || kernelIr.familyIdentity().startsWith("masked-reduction:")
                    || kernelIr.familyIdentity().startsWith("advanced-reduction:")
                    || kernelIr.familyIdentity().startsWith("softmax:")
                    || kernelIr.familyIdentity().startsWith("trailing-normalization:")
                        ? AccessFlag.PUBLIC.mask() : 0);
        byte[] bytes = ClassFile.of().build(owner, classBuilder -> classBuilder
                .withVersion(ClassFile.JAVA_26_VERSION, 0).withFlags(AccessFlag.FINAL)
                .withMethod(CpuGeneratorSchema.ENTRY_NAME, type, entryFlags, method ->
                        method.withCode(code -> {
                            if (usesSharedCarrierLayouts(kernelIr)) {
                                CpuCarrierEmitter.prepareSegmentLayouts(code,
                                        specialization.boundaryDataTypes(),
                                        specialization.carrierPattern());
                            }
                            if (kernelIr.instructions().isEmpty()) {
                                if (kernelIr.familyIdentity().startsWith("trailing-normalization:LAYER:")) {
                                    new CpuLayerNormEmitter().emit(code, specialization, kernelIr);
                                } else if (kernelIr.familyIdentity().startsWith("trailing-normalization:RMS:")) {
                                    new CpuRmsNormEmitter().emit(code, specialization, kernelIr);
                                } else if (kernelIr.familyIdentity().startsWith("softmax:")) {
                                    new CpuSoftmaxEmitter().emit(code, specialization, kernelIr);
                                } else if (kernelIr.familyIdentity().startsWith("advanced-reduction:LOG_SUM_EXP:")) {
                                    new CpuLogSumExpEmitter().emit(code, specialization, kernelIr);
                                } else if (kernelIr.familyIdentity().startsWith("advanced-reduction:VARIANCE:")
                                        || kernelIr.familyIdentity().startsWith("advanced-reduction:STANDARD_DEVIATION:")) {
                                    new CpuStatisticalReductionEmitter().emit(code, specialization, kernelIr);
                                } else if (kernelIr.familyIdentity().startsWith("advanced-reduction:L1_NORM:")
                                        || kernelIr.familyIdentity().startsWith("advanced-reduction:L2_NORM:")) {
                                    new CpuNormEmitter().emit(code, specialization, kernelIr);
                                } else if (kernelIr.familyIdentity().startsWith("masked-reduction:")) {
                                    new CpuMaskedReductionEmitter().emit(code, specialization, kernelIr);
                                } else if (kernelIr.familyIdentity().startsWith("arg-extrema:")) {
                                    new CpuArgExtremaEmitter().emit(code, specialization, kernelIr);
                                } else if (kernelIr.familyIdentity().startsWith("aggregate:")) {
                                    new CpuAggregateEmitter().emit(code, specialization, kernelIr);
                                } else if (kernelIr.familyIdentity().startsWith("scan:")) {
                                    new CpuScanEmitter().emit(code, specialization, kernelIr);
                                } else if (kernelIr.familyIdentity().startsWith("random:")) {
                                    new CpuRandomEmitter().emit(code, specialization, kernelIr);
                                } else if (kernelIr.familyIdentity().startsWith("ordering:")) {
                                    new CpuOrderingEmitter().emit(code, specialization, kernelIr);
                                } else if (kernelIr.familyIdentity().startsWith("fold:")) {
                                    new CpuFoldEmitter().emit(code, specialization, kernelIr);
                                } else if (kernelIr.familyIdentity().startsWith("scatter:")) {
                                    new CpuScatterEmitter().emit(code, specialization, kernelIr);
                                } else if (kernelIr.familyIdentity().startsWith("indexing:")) {
                                    new CpuIndexingEmitter().emit(code, specialization, kernelIr);
                                } else if (kernelIr.familyIdentity().startsWith("movement:")) {
                                    new CpuDataMovementEmitter().emit(code, specialization, kernelIr);
                                } else {
                                    new CpuAffineCopyEmitter().emit(code, specialization, kernelIr);
                                }
                                code.return_();
                                return;
                            }
                            var carriers = new CpuCarrierEmitter(code);
                            var scalar = new CpuScalarEmitter(code);
                            var loops = new CpuLoopEmitter(code);
                            List<CpuKernelIr.Value> boundaries = kernelIr.values().stream()
                                    .filter(value -> value.kind() != CpuKernelIr.Value.Kind.VIRTUAL)
                                    .toList();
                            List<io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan> plans =
                                    boundaries.stream().map(CpuKernelIr.Value::accessPlan).toList();
                            int[] scalarLocals = allocateScalarLocals(code, kernelIr);
                            java.util.function.Consumer<CpuLoopEmitter.State> scalarBody = state -> {
                                for (int boundary = 0; boundary < boundaries.size(); boundary++) {
                                    CpuKernelIr.Value value = boundaries.get(boundary);
                                    if (value.kind() != CpuKernelIr.Value.Kind.INPUT
                                            || !requiresInputLoad(kernelIr, value.ordinal())) continue;
                                    carriers.load(value.dataType(), specialization.carrierPattern().get(boundary),
                                            boundary, state.addresses()[boundary], state.intAddresses());
                                    store(code, value.dataType(), scalarLocals[value.ordinal()]);
                                }
                                kernelIr.instructions().forEach(instruction ->
                                        scalar.emit(kernelIr, instruction, scalarLocals));
                                for (CpuKernelIr.Store store : kernelIr.stores()) {
                                    CpuKernelIr.Value value = kernelIr.values().get(store.value());
                                    int boundary = boundaryIndex(boundaries, value.ordinal());
                                    carriers.store(value.dataType(), specialization.carrierPattern().get(boundary),
                                            boundary, state.addresses()[boundary], scalarLocals[value.ordinal()],
                                            state.intAddresses());
                                }
                            };
                            if (specialization.executionStrategy().compute()
                                    == io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy.Compute.VECTOR) {
                                DataType vectorType = vectorDataType(kernelIr);
                                carriers.prepareVectorSpecies(vectorType);
                                int[] vectorLocals = allocateVectorLocals(code, kernelIr);
                                var vectorInstructions = new CpuVectorInstructionEmitter(
                                        code, kernelIr, vectorType);
                                int lanes = specialization.vectorSpeciesBitSize() / vectorType.bitWidth();
                                if (specialization.loopAddressing(kernelIr)
                                        == CpuKernelSpecialization.LoopAddressing.DENSE_HEAP_ARRAY_INT)
                                    loops.emitDenseArrayIntVector(plans, lanes,
                                            state -> emitVectorBody(code, carriers, vectorInstructions,
                                                    specialization, kernelIr, boundaries, state,
                                                    vectorLocals, vectorType), scalarBody);
                                else loops.emitVector(plans, lanes,
                                            state -> emitVectorBody(code, carriers, vectorInstructions,
                                                    specialization, kernelIr, boundaries, state,
                                                    vectorLocals, vectorType), scalarBody);
                            } else if (specialization.loopAddressing(kernelIr)
                                    == CpuKernelSpecialization.LoopAddressing.DENSE_HEAP_ARRAY_INT)
                                loops.emitDenseArrayInt(plans, scalarBody);
                            else if (isGuardedScalarGeneral(specialization, kernelIr))
                                loops.emitGuardedScalarGeneral(plans,
                                        state -> emitGuardedScalarGeneralBody(code, carriers,
                                                state, scalarLocals), scalarBody);
                            else loops.emit(plans, scalarBody);
                            code.return_();
                        })));
        verify(specialization, bytes);
        return bytes;
    }

    private static boolean usesSharedCarrierLayouts(CpuKernelIr kernelIr) {
        if (!kernelIr.instructions().isEmpty()) return true;
        String family = kernelIr.familyIdentity();
        return !family.startsWith("fold:") && !family.startsWith("ordering:")
                && !family.startsWith("random:") && !family.startsWith("arg-extrema:")
                && !family.startsWith("trailing-normalization:");
    }

    /**
     * Verifies, defines, and resolves one exact generated entry.
     *
     * @param specialization non-null specialization expected by the bytes
     * @param classBytes non-null complete generated class bytes; copied into the returned artifact
     * @return a new strongly owning generated-kernel artifact with the exact direct handle
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if verification, hidden-class definition, or exact handle
     *     resolution fails
     */
    public CpuGeneratedKernel defineClassBytes(CpuKernelSpecialization specialization,
            byte[] classBytes) {
        Objects.requireNonNull(specialization, "specialization");
        Objects.requireNonNull(classBytes, "classBytes");
        verify(specialization, classBytes);
        try {
            MethodHandles.Lookup lookup = MethodHandles.lookup().defineHiddenClass(
                    classBytes, false, MethodHandles.Lookup.ClassOption.NESTMATE);
            var handle = lookup.findStatic(lookup.lookupClass(), CpuGeneratorSchema.ENTRY_NAME,
                    specialization.entryType());
            return new CpuGeneratedKernel(specialization, lookup, handle, classBytes);
        } catch (ReflectiveOperationException | IllegalArgumentException failure) {
            throw new IllegalArgumentException("generated kernel definition failed", failure);
        }
    }

    private static int[] allocateScalarLocals(java.lang.classfile.CodeBuilder code, CpuKernelIr ir) {
        int[] result = new int[ir.values().size()];
        for (CpuKernelIr.Value value : ir.values()) result[value.ordinal()] = code.allocateLocal(
                switch (value.dataType()) {
                    case FLOAT64 -> TypeKind.DOUBLE; case FLOAT32 -> TypeKind.FLOAT;
                    case INT32, BOOL -> TypeKind.INT; case INT64 -> TypeKind.LONG;
                    default -> throw new IllegalArgumentException("unsupported generated type");
                });
        return result;
    }

    private static int[] allocateVectorLocals(java.lang.classfile.CodeBuilder code, CpuKernelIr ir) {
        int[] result = new int[ir.values().size()];
        for (CpuKernelIr.Value value : ir.values()) result[value.ordinal()] =
                code.allocateLocal(TypeKind.REFERENCE);
        return result;
    }

    private static void emitVectorBody(java.lang.classfile.CodeBuilder code,
            CpuCarrierEmitter carriers, CpuVectorInstructionEmitter instructions,
            CpuKernelSpecialization specialization, CpuKernelIr ir,
            List<CpuKernelIr.Value> boundaries, CpuLoopEmitter.State state, int[] locals,
            DataType vectorType) {
        for (int boundary = 0; boundary < boundaries.size(); boundary++) {
            CpuKernelIr.Value value = boundaries.get(boundary);
            if (value.kind() != CpuKernelIr.Value.Kind.INPUT
                    || !requiresInputLoad(ir, value.ordinal())) continue;
            if (value.dataType() == DataType.BOOL && vectorType != DataType.BOOL) {
                carriers.scalarBoolMaskLoad(vectorType,
                        specialization.carrierPattern().get(boundary), boundary,
                        state.addresses()[boundary], state.intAddresses());
            } else carriers.vectorLoad(vectorType,
                    specialization.carrierPattern().get(boundary), boundary,
                    state.addresses()[boundary], value.accessPlan().regime()
                            == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Regime.SCALAR_ALL_ZERO,
                    state.intAddresses());
            code.astore(locals[value.ordinal()]);
        }
        ir.instructions().forEach(instruction -> instructions.emit(instruction, locals));
        for (CpuKernelIr.Store store : ir.stores()) {
            CpuKernelIr.Value value = ir.values().get(store.value());
            int boundary = boundaryIndex(boundaries, value.ordinal());
            carriers.vectorStore(vectorType, specialization.carrierPattern().get(boundary), boundary,
                    state.addresses()[boundary], locals[value.ordinal()], state.intAddresses());
        }
    }

    private static int boundaryIndex(List<CpuKernelIr.Value> boundaries, int ordinal) {
        for (int i = 0; i < boundaries.size(); i++) if (boundaries.get(i).ordinal() == ordinal) return i;
        throw new IllegalArgumentException("stored value is not a materialized boundary");
    }

    private static boolean requiresInputLoad(CpuKernelIr ir, int ordinal) {
        return ir.instructions().stream().anyMatch(instruction -> {
            for (int input = 0; input < instruction.inputs().size(); input++) {
                if (instruction.inputs().get(input) != ordinal) continue;
                if (input == 0 && instruction.opcode() == CpuPointwiseOpcode.SCALAR_POW
                        && instruction.powerRealization()
                            == CpuKernelIr.PowerRealization.POSITIVE_ONE) continue;
                return true;
            }
            return false;
        });
    }

    private static boolean isGuardedScalarGeneral(CpuKernelSpecialization specialization,
            CpuKernelIr ir) {
        if (specialization.executionStrategy().compute()
                    != io.github.pho001.synaptik.backend.cpu.internal.prepare
                            .CpuPartitionPreparationPlan.ExecutionStrategy.Compute.SCALAR
                || specialization.loopAddressing(ir)
                    != CpuKernelSpecialization.LoopAddressing.GENERAL_LONG
                || !specialization.carrierPattern().equals(List.of(
                        CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT,
                        CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY,
                        CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT,
                        CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY))
                || ir.values().size() != 6 || ir.instructions().size() != 3
                || ir.stores().size() != 1) return false;
        CpuAccessPlan denseRead = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,
                CpuAccessPlan.Regime.DENSE_LINEAR, 2,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS,
                        CpuAccessPlan.AxisRole.CONTIGUOUS), 2);
        CpuAccessPlan biasRead = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,
                CpuAccessPlan.Regime.LAST_AXIS_BIAS, 2,
                List.of(CpuAccessPlan.AxisRole.BROADCAST,
                        CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
        CpuAccessPlan generalWrite = new CpuAccessPlan(CpuAccessPlan.AccessKind.WRITE,
                CpuAccessPlan.Regime.GENERAL_ODOMETER, 2,
                List.of(CpuAccessPlan.AxisRole.STRIDED,
                        CpuAccessPlan.AxisRole.STRIDED), 0);
        List<CpuKernelIr.Value> expectedValues = List.of(
                new CpuKernelIr.Value(0, DataType.FLOAT32, CpuKernelIr.Value.Kind.INPUT, denseRead),
                new CpuKernelIr.Value(1, DataType.FLOAT32, CpuKernelIr.Value.Kind.INPUT, biasRead),
                new CpuKernelIr.Value(2, DataType.FLOAT32, CpuKernelIr.Value.Kind.INPUT, denseRead),
                new CpuKernelIr.Value(3, DataType.FLOAT32, CpuKernelIr.Value.Kind.VIRTUAL,
                        generalWrite),
                new CpuKernelIr.Value(4, DataType.FLOAT32, CpuKernelIr.Value.Kind.VIRTUAL,
                        generalWrite),
                new CpuKernelIr.Value(5, DataType.FLOAT32, CpuKernelIr.Value.Kind.OUTPUT,
                        generalWrite));
        return ir.familyIdentity().equals("pointwise") && ir.values().equals(expectedValues)
                && ir.instructions().equals(List.of(
                        new CpuKernelIr.Instruction(CpuPointwiseOpcode.DIV, List.of(0, 1), 3),
                        new CpuKernelIr.Instruction(CpuPointwiseOpcode.SIGMOID, List.of(3), 4),
                        new CpuKernelIr.Instruction(CpuPointwiseOpcode.MUL, List.of(4, 2), 5)))
                && ir.stores().equals(List.of(new CpuKernelIr.Store(5, 0)));
    }

    private static void emitGuardedScalarGeneralBody(java.lang.classfile.CodeBuilder code,
            CpuCarrierEmitter carriers, CpuLoopEmitter.State state, int[] locals) {
        emitGuardedFloatSegmentLoad(code, 0, state.addresses()[0]);
        carriers.load(DataType.FLOAT32, CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY,
                1, state.addresses()[1]);
        code.fdiv().fstore(locals[3]);
        var negative = code.newLabel();
        var sigmoid = code.newLabel();
        code.fload(locals[3]).loadConstant(0.0f).fcmpl().branch(
                java.lang.classfile.Opcode.IFLT, negative);
        code.loadConstant(1.0d).loadConstant(1.0d).fload(locals[3]).f2d().dneg()
                .invokestatic(STRICT_MATH, "exp", MethodTypeDesc.of(
                        TypeKind.DOUBLE.upperBound(), TypeKind.DOUBLE.upperBound()))
                .dadd().ddiv().branch(java.lang.classfile.Opcode.GOTO, sigmoid);
        code.labelBinding(negative);
        int exponential = code.allocateLocal(TypeKind.DOUBLE);
        code.fload(locals[3]).f2d().invokestatic(STRICT_MATH, "exp", MethodTypeDesc.of(
                TypeKind.DOUBLE.upperBound(), TypeKind.DOUBLE.upperBound()))
                .dstore(exponential);
        code.dload(exponential).loadConstant(1.0d).dload(exponential).dadd().ddiv();
        code.labelBinding(sigmoid);
        code.d2f().fstore(locals[4]);
        code.aload(3).lload(state.addresses()[3]).l2i().fload(locals[4]);
        emitGuardedFloatSegmentLoad(code, 2, state.addresses()[2]);
        code.fmul().fastore();
    }

    private static void emitGuardedFloatSegmentLoad(java.lang.classfile.CodeBuilder code,
            int parameter, int address) {
        code.aload(parameter).getstatic(VALUE_LAYOUT, "JAVA_FLOAT_UNALIGNED", FLOAT_LAYOUT)
                .lload(address).loadConstant(4L).lmul()
                .invokeinterface(SEGMENT, "get", MethodTypeDesc.of(
                        TypeKind.FLOAT.upperBound(), FLOAT_LAYOUT, TypeKind.LONG.upperBound()));
    }

    private static void store(java.lang.classfile.CodeBuilder code, DataType type, int local) {
        switch (type) {
            case FLOAT64 -> code.dstore(local); case FLOAT32 -> code.fstore(local);
            case INT32, BOOL -> code.istore(local); case INT64 -> code.lstore(local);
            default -> throw new IllegalArgumentException("unsupported generated type");
        }
    }

    private static void validate(CpuKernelSpecialization specialization, CpuKernelIr kernelIr) {
        Objects.requireNonNull(specialization, "specialization");
        Objects.requireNonNull(kernelIr, "kernelIr");
        if (!specialization.loweringFingerprint().hex().equals(kernelIr.structuralKey())) {
            throw new IllegalArgumentException("kernel IR does not match specialization");
        }
        List<DataType> boundaryTypes = kernelIr.values().stream()
                .filter(value -> value.kind() != CpuKernelIr.Value.Kind.VIRTUAL)
                .map(CpuKernelIr.Value::dataType).toList();
        if (!boundaryTypes.equals(specialization.boundaryDataTypes())
                || kernelIr.instructions().size() > 8
                || kernelIr.stores().size() != 1) {
            throw new IllegalArgumentException("unsupported canonical pointwise IR");
        }
        if (kernelIr.instructions().isEmpty()) {
            boolean indexing = kernelIr.familyIdentity().startsWith("indexing:");
            boolean random = kernelIr.familyIdentity().startsWith("random:");
            boolean scan = kernelIr.familyIdentity().startsWith("scan:");
            boolean aggregate = kernelIr.familyIdentity().startsWith("aggregate:");
            boolean argExtrema = kernelIr.familyIdentity().startsWith("arg-extrema:");
            boolean maskedReduction = kernelIr.familyIdentity().startsWith("masked-reduction:");
            boolean advancedReduction = kernelIr.familyIdentity().startsWith("advanced-reduction:");
            boolean softmax = kernelIr.familyIdentity().startsWith("softmax:");
            boolean trailingNormalization = kernelIr.familyIdentity()
                    .startsWith("trailing-normalization:");
            boolean scatter = kernelIr.familyIdentity().startsWith("scatter:");
            boolean fold = kernelIr.familyIdentity().startsWith("fold:");
            boolean ordering = kernelIr.familyIdentity().startsWith("ordering:");
            boolean movement = kernelIr.familyIdentity().startsWith("movement:");
            boolean affine = kernelIr.familyIdentity().startsWith("affine:");
            if ((!movement && !affine && !indexing && !scatter && !fold && !ordering && !random && !scan && !aggregate && !argExtrema && !maskedReduction && !advancedReduction && !softmax && !trailingNormalization)
                    || affine && kernelIr.values().size() != 2
                    || movement && (kernelIr.values().size() < 2 || kernelIr.values().size() > 17)
                    || !ordering && !random && !scan && !aggregate && !argExtrema && !maskedReduction && !advancedReduction && !softmax && !trailingNormalization && kernelIr.values().subList(0, kernelIr.values().size() - 1).stream()
                        .anyMatch(value -> value.kind() != CpuKernelIr.Value.Kind.INPUT)
                    || kernelIr.values().getLast().kind() != CpuKernelIr.Value.Kind.OUTPUT
                    || !indexing && !scatter && !fold && !ordering && !random && !scan && !aggregate && !argExtrema && !maskedReduction && !advancedReduction && !softmax && !trailingNormalization
                        && kernelIr.values().stream().map(CpuKernelIr.Value::dataType).distinct().count() != 1
                    || random && kernelIr.values().size() != 1 && kernelIr.values().size() != 5
                    || specialization.executionStrategy().compute()
                        != io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy.Compute.SCALAR
                    || !specialization.scalarPowerRealizations().isEmpty()) {
                throw new IllegalArgumentException("unsupported canonical affine copy IR");
            }
            return;
        }
        if (!kernelIr.familyIdentity().equals("pointwise")
                || kernelIr.values().stream().anyMatch(value -> value.dataType()
                        == io.github.pho001.synaptik.model.datatype.DataType.BFLOAT16)
                || specialization.carrierPattern().contains(
                        CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY)) {
            throw new IllegalArgumentException("unsupported canonical pointwise IR");
        }
        List<CpuKernelIr.PowerRealization> realizations = kernelIr.instructions().stream()
                .filter(instruction -> instruction.opcode() == CpuPointwiseOpcode.SCALAR_POW)
                .map(CpuKernelIr.Instruction::powerRealization).toList();
        if (!realizations.equals(specialization.scalarPowerRealizations())) {
            throw new IllegalArgumentException("specialization power realizations do not match IR");
        }
        if (specialization.executionStrategy().compute()
                == io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy.Compute.VECTOR
                && vectorDataTypeOrNull(kernelIr) == null) {
            throw new IllegalArgumentException(
                    "vector specialization requires one supported typed value or virtual-mask topology");
        }
    }

    private static DataType vectorDataType(CpuKernelIr kernelIr) {
        DataType result = vectorDataTypeOrNull(kernelIr);
        if (result == null) throw new IllegalArgumentException(
                "vector specialization requires one supported typed value or virtual-mask topology");
        return result;
    }

    private static DataType vectorDataTypeOrNull(CpuKernelIr kernelIr) {
        var numeric = kernelIr.values().stream().map(CpuKernelIr.Value::dataType)
                .filter(type -> type != DataType.BOOL).distinct().toList();
        DataType result;
        if (numeric.isEmpty()) result = DataType.BOOL;
        else if (numeric.size() == 1) result = numeric.getFirst();
        else return null;
        if (result != DataType.FLOAT32 && result != DataType.FLOAT64
                && result != DataType.INT32 && result != DataType.INT64
                && result != DataType.BOOL) return null;
        return vectorTopologyEligible(kernelIr, result) ? result : null;
    }

    private static boolean vectorTopologyEligible(CpuKernelIr ir, DataType laneType) {
        boolean mixedMasks = (laneType == DataType.FLOAT32 || laneType == DataType.FLOAT64)
                && ir.values().stream().anyMatch(value -> value.dataType() == DataType.BOOL);
        for (CpuKernelIr.Value value : ir.values()) {
            if (value.dataType() == DataType.BOOL && mixedMasks
                    && value.kind() != CpuKernelIr.Value.Kind.VIRTUAL
                    && !(value.kind() == CpuKernelIr.Value.Kind.INPUT
                        && value.accessPlan().regime()
                            == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Regime.SCALAR_ALL_ZERO)) {
                return false;
            }
        }
        for (CpuKernelIr.Instruction instruction : ir.instructions()) {
            if (!instruction.opcode().vectorEligible()
                    || instruction.opcode() == CpuPointwiseOpcode.SCALAR_POW
                        && instruction.powerRealization() == CpuKernelIr.PowerRealization.DIRECT) {
                return false;
            }
            switch (instruction.opcode().vectorForm()) {
                case VALUE -> {
                    if (!valueOpcodeEligible(instruction.opcode(), laneType)) return false;
                }
                case MASK_PRODUCER -> {
                    if (!mixedMasks || ir.values().get(instruction.output()).kind()
                            != CpuKernelIr.Value.Kind.VIRTUAL) return false;
                }
                case VALUE_OR_MASK -> {
                    boolean byteValues = laneType == DataType.BOOL;
                    boolean virtualMasks = mixedMasks
                            && instruction.inputs().stream().allMatch(input ->
                                ir.values().get(input).kind() == CpuKernelIr.Value.Kind.VIRTUAL)
                            && ir.values().get(instruction.output()).kind()
                                == CpuKernelIr.Value.Kind.VIRTUAL;
                    if (!byteValues && !virtualMasks) return false;
                }
                case MASK_CONSUMER -> {
                    if (!mixedMasks) return false;
                    CpuKernelIr.Value condition = ir.values().get(instruction.inputs().getFirst());
                    if (condition.kind() != CpuKernelIr.Value.Kind.VIRTUAL
                            && !(condition.kind() == CpuKernelIr.Value.Kind.INPUT
                                && condition.accessPlan().regime()
                                    == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Regime.SCALAR_ALL_ZERO)) {
                        return false;
                    }
                }
                case NONE -> { return false; }
            }
        }
        return true;
    }

    private static boolean valueOpcodeEligible(CpuPointwiseOpcode opcode, DataType laneType) {
        if (laneType == DataType.BOOL) return opcode == CpuPointwiseOpcode.CAST;
        if (laneType == DataType.INT32 || laneType == DataType.INT64) return switch (opcode) {
            case ADD, SUB, MUL, MIN, MAX, SCALAR_ADD, SCALAR_SUB, SCALAR_MUL,
                    SCALAR_MIN, SCALAR_MAX, CAST -> true;
            default -> false;
        };
        return laneType == DataType.FLOAT32 || laneType == DataType.FLOAT64;
    }

    private static void verify(CpuKernelSpecialization specialization, byte[] bytes) {
        try {
            var errors = ClassFile.of().verify(bytes);
            if (!errors.isEmpty()) throw new IllegalArgumentException(
                    "generated class verification failed: " + errors);
            var model = ClassFile.of().parse(bytes);
            String expectedName = CpuGeneratorSchema.generatedBinaryName(specialization).replace('.', '/');
            boolean method = model.methods().size() == 1
                    && model.methods().getFirst().methodName().stringValue().equals(CpuGeneratorSchema.ENTRY_NAME)
                    && model.methods().getFirst().methodType().stringValue()
                    .equals(specialization.entryType().descriptorString());
            if (model.majorVersion() != ClassFile.JAVA_26_VERSION
                    || !model.thisClass().asInternalName().equals(expectedName)
                    || !model.interfaces().isEmpty() || !model.fields().isEmpty() || !method) {
                throw new IllegalArgumentException("generated class shape does not match specialization");
            }
        } catch (IllegalArgumentException failure) { throw failure; }
        catch (RuntimeException failure) {
            throw new IllegalArgumentException("generated class verification failed", failure);
        }
    }
}

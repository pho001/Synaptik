package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratorSchema;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import java.lang.classfile.ClassBuilder;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Opcode;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPartialReductionIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.ClassFile;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;
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
 * FLOAT32/FLOAT64, signed-integral, canonical-BOOL, or bounded dense floating-mask vector body,
 * and retains the scalar body for tails. Unit-private masks remain virtual, while required dense
 * BOOL boundaries use a fixed covering byte species. All four low floating types use scalar
 * pointwise work with raw represented locals, F32 numerical work, and one final RNE per logical
 * result. OFP8 carriers remain bytes rather than Boolean masks.
 * A pointwise topology containing cross-type CAST is likewise scalar-only: each CAST emits its
 * Model-defined conversion directly into the target-typed local, preserving explicit
 * intermediate conversion boundaries without a generated Synaptik helper call.
 * Eligible direct dense same-typed FLOAT32/FLOAT64 Conv2d and Conv3d schema-63 specializations
 * instead emit output-width vector chunks with ordered scalar borders, range fragments, and tails;
 * width stride and dilation alone must be one. Ineligible convolution retains its historical
 * schema-52 scalar class projection. Proved vector scalar-power bodies prepare reusable constants
 * and, only for an accessed segment vector, its byte order before execution. They choose a dense
 * loop from the boundaries the body actually accesses, compute its vector bound once, and omit
 * positive-one base-address work; arbitrary ranges and scalar tails remain explicit. The
 * generator does not choose capability, numerical
 * semantics, access structure, strategy, or fallback.</p>
 * <p>A typed general-scan specialization instead emits one serial time loop and typed FLOAT32/FLOAT64
 * pointwise, shaped/reduction or rank-two MATMUL body. Its instruction-free boundary envelope
 * must exactly match the typed body, role, demand and temporal-bank facts. Layout setup is
 * reserved once before emission. Combined symbolic-length entries keep the same slot block
 * without initializing unused layout locals, since all their checked byte accesses use
 * constant predefined layouts; no
 * generated runtime graph or per-operation invocation is introduced. Only the selected row-region
 * exception below permits prebound synchronous range dispatch inside time.</p>
 * <p>Selected row-map/multiple-terminal exec-v1 regions retain the scalar v1 projection when
 * unselected. Their FLOAT32/FLOAT64 feature vectors reuse the common arithmetic emitters,
 * stage only proved transpose lifetimes, retain invariant coefficient chunks before each row
 * loop, and publish disjoint range partials with validity flags. Fixed-index merge moves the
 * first valid partial raw. Selected Scan entries keep the emitted serial time loop and a
 * distinct carry ADD at each active time; only their prebound row region may dispatch.</p>
 * Proved static count-write envelopes emit typed literals and output-only stores directly.
 * Fixed softmax derivative envelopes dispatch exclusively to their two-pass emitter after
 * matching typed sites and baked geometry; no forward finite-input validator is inherited.
 * Instruction-free affine, movement, indexing, scatter, fold, ordering, explicit-state random,
 * cumulative-scan, ordinary aggregate, and first-class loss forms delegate to their focused
 * emitters after
 * structural specialization checks. Dense heap-array pointwise, affine-copy, movement, and
 * indexing bodies use one-time integer narrowing and hoisted geometry; indexing, functional
 * scatter, overlap-fold, ordering, scan, and aggregate emitters embed typed family hot loops.
 * Functional-scatter classes additionally embed their selected reduction and optional
 * exact-product scratch state. Fold classes embed coordinate mapping and sequential represented
 * addition without a generic carrier bridge. General layouts and segment or mixed carriers retain
 * typed long-address fallbacks. Ordering classes additionally embed stable merge, comparison,
 * selected-pair ordering, and represented value/index stores over assigned scratch. Loss bodies
 * decode only their cold primitive geometry and emit direct typed MSE or stable categorical loops
 * with no generated Synaptik helper bridge. Random
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

    /**
     * Enumerates straight-line scalar loads in emission order, with one binding per distinct
     * pointwise operand. Singleton SUM alternatives are excluded from caching.
     * @param ir complete pre-resource origin/coefficient proof
     * @return external-cell keys in exact load order
     */
    private static java.util.List<java.util.List<Long>> regionScalarTrace(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr ir) {
        var trace = new java.util.ArrayList<java.util.List<Long>>(); int epoch = 0;
        var products = new java.util.HashMap<Integer,io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr.Contraction>();
        ir.contractions().forEach(c -> products.put(c.output(),c));
        var instructions = new java.util.HashMap<Integer,io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr.Instruction>();
        ir.arithmetic().instructions().forEach(i -> instructions.put(i.output(),i));
        for (int out : ir.rowSites()) for (int cell = 0; cell < regionCells(ir,out,true); cell++) {
            int[] logical = regionCoordinates(ir,out,cell,true);
            var product = products.get(out);
            if (product != null) {
                int k = Math.toIntExact(ir.accesses().get(product.left()).extentConstants().get(1));
                for (int f = 0; f < k; f++) {
                    regionTraceLoad(ir,trace,epoch,product.left(),new int[]{-1,f});
                    regionTraceLoad(ir,trace,epoch,product.right(),new int[]{f,logical[1]});
                }
            } else for (int in : new java.util.LinkedHashSet<>(instructions.get(out).inputs()))
                regionTraceLoad(ir,trace,epoch,in,logical);
            if (ir.retainedStores().contains(out)) epoch++;
        }
        for (int out : ir.terminalOutputs()) {
            var product = products.get(out); if (product == null) continue;
            for (int cell = 0; cell < regionCells(ir,out,false); cell++) {
                int[] logical = regionCoordinates(ir,out,cell,false);
                regionTraceLoad(ir,trace,epoch,product.left(),new int[]{logical[0],-1});
                regionTraceLoad(ir,trace,epoch,product.right(),new int[]{-1,logical[1]});
            }
        }
        return trace;
    }

    /**
     * Appends only storage-backed operands, never completed producing locals.
     * @param ir authenticated origin and mapping inventory
     * @param trace ordered generation-only load requests
     * @param epoch interval after the last retained store
     * @param v operand/view ordinal
     * @param logical consuming coordinates, minus one for the original row
     */
    private static void regionTraceLoad(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr ir,
            java.util.List<java.util.List<Long>> trace,int epoch,int v,int[] logical) {
        if (!ir.rowSites().contains(ir.viewOrigins().get(v))) {
            var key = new java.util.ArrayList<>(regionScalarKey(ir,v,logical)); key.add((long)epoch);
            trace.add(java.util.List.copyOf(key));
        }
    }

    /**
     * Normalizes the complete nominal affine cell polynomial for every admitted B. Temporal
     * binding and storage epoch are fixed by the active-row scope; each binder separately
     * excludes roots that could be overwritten or change effective mapping within that scope.
     * @param ir authenticated root and nominal coefficient proof
     * @param v operand/view ordinal
     * @param logical right-aligned feature coordinates, minus one for the row
     * @return root, fixed constant/B and row constant/B coefficients
     * @throws ArithmeticException if checked cold coefficient arithmetic overflows
     */
    private static java.util.List<Long> regionScalarKey(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr ir,int v,int[] logical) {
        var a = ir.accesses().get(v); long c = a.baseElementOffset(), f = 0, rc = 0, rf = 0;
        int shift = logical.length-a.plan().iterationRank();
        for (int axis = 0; axis < a.plan().iterationRank(); axis++) {
            int x = a.extentConstants().get(axis) == 1 && a.extentBatchFactors().get(axis) == 0 ? 0 : logical[shift+axis];
            long sc = a.strideConstants().get(axis), sf = a.strideBatchFactors().get(axis);
            if (x < 0) { rc = Math.addExact(rc,sc); rf = Math.addExact(rf,sf); }
            else { c = Math.addExact(c,Math.multiplyExact(x,sc)); f = Math.addExact(f,Math.multiplyExact(x,sf)); }
        }
        return java.util.List.of((long)ir.viewOrigins().get(v),c,f,rc,rf);
    }
    private static final String PARTIAL_REDUCTION_SCHEMA = "partial-reduction-v1";
    private static final String PARTIAL_BODY_NAME = "partialBody";
    private static final String ORDERED_COMBINE_NAME = "orderedCombine";
    private static final ClassDesc STRICT_MATH = ClassDesc.of(StrictMath.class.getName());
    private static final ClassDesc SEGMENT = ClassDesc.of("java.lang.foreign.MemorySegment");
    private static final ClassDesc VALUE_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout");
    private static final ClassDesc FLOAT_LAYOUT =
            ClassDesc.of("java.lang.foreign.ValueLayout$OfFloat");
    /** Creates a stateless generator with no retained route or specialization state. */
    public CpuClassFileKernelGenerator() { }

    /**
     * Authenticates a cold recipe before any generated-class cache lookup, including weak hits.
     *
     * @param specialization non-null typed carrier and selective compatibility facts
     * @param kernelIr non-null canonical body envelope
     * @throws NullPointerException if either argument is null
     * @throws IllegalArgumentException if family facts, carriers, ABI or compatibility disagree
     */
    public static void validateSpecialization(CpuKernelSpecialization specialization, CpuKernelIr kernelIr) {
        validate(specialization, kernelIr);
    }

    /**
     * Binds the two iteration domains of an ordinary aggregate composition. Static coordinate
     * cursors and cell bases dominate contributor arithmetic; controlled ROW guards precede
     * producing-site reads and final-output epilogues. Removed producer values use their own
     * proved value-inventory provenance, never that of an unrelated backing boundary. Terminal
     * ROW SUM reuses represented sites; other folds retain exact completion before the epilogue.
     * No removed value has a carrier and no runtime traversal callback is retained.
     * @param code nonnull entry builder
     * @param specialization matching static scalar fusion and physical carrier facts
     */
    private static void emitAggregateFusion(CodeBuilder code, CpuKernelSpecialization specialization) {
        var fusion = specialization.aggregateFusionIr().orElseThrow();
        var aggregate = fusion.aggregate(); var type = aggregate.dataType();
        TypeKind numeric = type == DataType.FLOAT64 ? TypeKind.DOUBLE : TypeKind.FLOAT;
        int count = fusion.boundaries().size(), geometry = count+1, start = count+2, end = count+4;
        int inRank = fusion.inputExtents().size();
        var output = fusion.boundaries().getLast(); int outRank = output.extents().size();
        boolean integer = specialization.carrierPattern().stream().noneMatch(c -> c == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)
                && java.util.stream.Stream.of(fusion.boundaries(),fusion.producerAccesses(),fusion.epilogueAccesses())
                    .flatMap(List::stream).allMatch(b -> b.elementCount() <= Integer.MAX_VALUE
                        && b.referencedElementSpan() <= Integer.MAX_VALUE && b.baseElementOffset() <= Integer.MAX_VALUE
                        && b.extents().stream().allMatch(e -> e <= Integer.MAX_VALUE)
                        && b.effectiveStrides().stream().allMatch(s -> s <= Integer.MAX_VALUE));
        TypeKind coordinate = integer ? TypeKind.INT : TypeKind.LONG;
        int headerSize = 15+4*inRank+3*outRank;
        int[] bases = new int[count];
        for (int i = 0; i < count; i++) {
            bases[i] = code.allocateLocal(coordinate);
            code.aload(geometry).loadConstant(headerSize+i).laload(); if (integer) code.l2i(); code.storeLocal(coordinate,bases[i]);
        }
        int cell = code.allocateLocal(coordinate), remainder = code.allocateLocal(coordinate), limit = code.allocateLocal(coordinate);
        code.lload(end); if (integer) code.l2i(); code.storeLocal(coordinate,limit);
        int[] out = new int[outRank], in = new int[inRank];
        for (int i = 0; i < outRank; i++) out[i] = code.allocateLocal(coordinate);
        boolean[] selected = new boolean[inRank]; for (int axis : aggregate.selectedAxes()) selected[axis] = true;
        int o = 0;
        for (int axis = 0; axis < inRank; axis++) {
            if (selected[axis]) { in[axis] = code.allocateLocal(coordinate); if (aggregate.keepDimensions()) o++; }
            else in[axis] = out[o++];
        }
        var producer = fusion.producer().orElse(null); var tail = fusion.epilogue().orElse(null);
        var control = CpuCarrierEmitter.controlFacts().orElse(null);
        var contributionRow = control == null ? null : producer == null
                ? control.boundaryRows().get(fusion.aggregateInput())
                : control.valueRows().get(count + producer.stores().getFirst().value());
        boolean represented = aggregate.kind() == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateIr.Kind.SUM
                && contributionRow != null && contributionRow.rowIndexed()
                && selected[contributionRow.axis()] && !control.boundaryRows().getLast().rowIndexed();
        int[] producerLocals = aggregateSiteLocals(code,producer,numeric);
        int[] tailLocals = aggregateSiteLocals(code,tail,numeric);
        int contribution = producer == null ? code.allocateLocal(numeric)
                : producerLocals[producer.stores().getFirst().value()];
        int result = code.allocateLocal(numeric), address = code.allocateLocal(coordinate);
        var reads = producer == null ? java.util.List.of(fusion.boundaries().get(fusion.aggregateInput()))
                : fusion.producerAccesses().subList(0,fusion.producerInputs().size());
        var positions = producer == null ? java.util.List.of(fusion.aggregateInput()) : fusion.producerInputs();
        int[] cellBases = new int[reads.size()];
        for (int i = 0; i < cellBases.length; i++) cellBases[i] = code.allocateLocal(coordinate);
        var scalar = new CpuScalarEmitter(code); var carrier = new CpuCarrierEmitter(code);
        java.util.function.Consumer<Runnable> cells = body -> {
            var loop = code.newLabel(); var done = code.newLabel();
            var advance = control == null ? null : code.newLabel();
            code.lload(start); if (integer) code.l2i(); code.storeLocal(coordinate,cell).labelBinding(loop);
            code.loadLocal(coordinate,cell).loadLocal(coordinate,limit);
            if (integer) code.branch(Opcode.IF_ICMPGE,done); else code.lcmp().branch(Opcode.IFGE,done);
            if (control != null)
                CpuLoopEmitter.emitActiveOrdinalGuard(code, count - 1, cell, coordinate, advance);
            code.loadLocal(coordinate,cell).storeLocal(coordinate,remainder);
            for (int axis = outRank-1; axis >= 0; axis--) {
                long extent = Math.max(1,output.extents().get(axis));
                if (extent == 1) aggregateConstant(code,integer,0).storeLocal(coordinate,out[axis]);
                else if (axis == 0) code.loadLocal(coordinate,remainder).storeLocal(coordinate,out[axis]);
                else {
                    code.loadLocal(coordinate,remainder); aggregateConstant(code,integer,extent);
                    if (integer) code.irem(); else code.lrem(); code.storeLocal(coordinate,out[axis]);
                    if (axis != 0) {
                        code.loadLocal(coordinate,remainder); aggregateConstant(code,integer,extent);
                        if (integer) code.idiv(); else code.ldiv(); code.storeLocal(coordinate,remainder);
                    }
                }
            }
            for (int i = 0; i < reads.size(); i++) {
                var access = reads.get(i);
                code.loadLocal(coordinate,bases[positions.get(i)]); aggregateConstant(code,integer,access.baseElementOffset());
                aggregateAdd(code,integer); aggregateAddressTerms(code,access,in,selected,false,integer);
                code.storeLocal(coordinate,cellBases[i]);
            }
            body.run();
            if (control != null) code.labelBinding(advance);
            if (integer) code.iinc(cell,1); else code.lload(cell).loadConstant(1L).ladd().lstore(cell);
            code.branch(Opcode.GOTO,loop).labelBinding(done);
        };
        java.util.function.Consumer<Runnable> contributors = body -> {
            if (aggregate.domainCount() != 0)
                emitAggregateSelected(code,aggregate.selectedAxes(),fusion.inputExtents(),in,0,integer,() -> {
                    if (contributionRow == null || !contributionRow.rowIndexed()) body.run();
                    else {
                        var next = code.newLabel();
                        CpuCarrierEmitter.emitActiveRowGuard(code, in[contributionRow.axis()], coordinate, next);
                        body.run();
                        code.labelBinding(next);
                    }
                });
        };
        Runnable producingSite = () -> {
            Runnable loads = () -> {
                for (int i = 0; i < reads.size(); i++) {
                    code.loadLocal(coordinate,cellBases[i]); aggregateAddressTerms(code,reads.get(i),in,selected,true,integer); code.storeLocal(coordinate,address);
                    carrier.load(type,specialization.carrierPattern().get(positions.get(i)),positions.get(i),address,integer);
                    code.storeLocal(numeric,producer == null ? contribution : producerLocals[i]);
                }
            };
            if (producer == null) loads.run();
            else emitBoundPointwise(producer.instructions(),loads,i -> {},i -> scalar.emit(producer,i,producerLocals),i -> {},() -> {});
        };
        Runnable sink = () -> {
            int value = result;
            if (tail != null) {
                emitBoundPointwise(tail.instructions(),() -> {
                    for (int i = 0; i < fusion.epilogueInputs().size(); i++) {
                        int position = fusion.epilogueInputs().get(i);
                        if (position == -1) code.loadLocal(numeric,result);
                        else {
                            var access = fusion.epilogueAccesses().get(i);
                            code.loadLocal(coordinate,bases[position]); aggregateConstant(code,integer,access.baseElementOffset()); aggregateAdd(code,integer);
                            aggregateAddressTerms(code,access,out,new boolean[outRank],false,integer); code.storeLocal(coordinate,address);
                            carrier.load(type,specialization.carrierPattern().get(position),position,address,integer);
                        }
                        code.storeLocal(numeric,tailLocals[i]);
                    }
                },i -> {},i -> scalar.emit(tail,i,tailLocals),i -> {},() -> {});
                value = tailLocals[tail.stores().getFirst().value()];
            }
            code.loadLocal(coordinate,bases[count-1]); aggregateConstant(code,integer,output.baseElementOffset()); aggregateAdd(code,integer);
            aggregateAddressTerms(code,output,out,new boolean[outRank],false,integer); code.storeLocal(coordinate,address);
            carrier.store(type,specialization.carrierPattern().getLast(),count-1,address,value,integer);
        };
        if (represented)
            CpuAggregateEmitter.emitRepresentedSum(code,type,result,false,false,cells,contributors,
                    () -> {},producingSite,() -> code.loadLocal(numeric,contribution),
                    ignored -> sink.run(),() -> {});
        else CpuAggregateEmitter.emitExactSum(code,aggregate,count,geometry,contribution,result,
                cells,contributors,producingSite,sink);
    }

    /**
     * Reserves typed producing-site locals once while generating the enclosing entry.
     * @param code nonnull builder
     * @param sites optional body, null when absent
     * @param numeric exact FLOAT32/FLOAT64 local kind
     * @return new generation-only local table, empty without sites
     */
    private static int[] aggregateSiteLocals(CodeBuilder code, CpuKernelIr sites, TypeKind numeric) {
        int[] result = new int[sites == null ? 0 : sites.values().size()];
        for (int i = 0; i < result.length; i++) result[i] = code.allocateLocal(numeric);
        return result;
    }

    /**
     * Emits canonical nested static contribution cursors with no runtime recipe or callback.
     * @param code nonnull builder
     * @param axes increasing selected input axes
     * @param extents complete static input extents
     * @param coordinates allocated primitive coordinate slots indexed by input axis
     * @param depth current selected-axis depth, initially zero
     * @param integer whether complete heap geometry proves int coordinates
     * @param body generation-only action at each complete contribution coordinate
     */
    private static void emitAggregateSelected(CodeBuilder code,int[] axes,List<Long> extents,
            int[] coordinates,int depth,boolean integer,Runnable body) {
        if (depth == axes.length) { body.run(); return; }
        int axis = axes[depth], coordinate = coordinates[axis];
        var loop = code.newLabel(); var done = code.newLabel();
        if (integer) {
            code.loadConstant(0).istore(coordinate).labelBinding(loop);
            code.iload(coordinate).loadConstant(Math.toIntExact(extents.get(axis))).branch(Opcode.IF_ICMPGE,done);
        } else {
            code.loadConstant(0L).lstore(coordinate).labelBinding(loop);
            code.lload(coordinate).loadConstant(extents.get(axis)).lcmp().branch(Opcode.IFGE,done);
        }
        emitAggregateSelected(code,axes,extents,coordinates,depth+1,integer,body);
        if (integer) code.iinc(coordinate,1);
        else code.lload(coordinate).loadConstant(1L).ladd().lstore(coordinate);
        code.branch(Opcode.GOTO,loop).labelBinding(done);
    }

    /**
     * Appends only varying coordinate terms to an element address already on the stack.
     * Unit/zero strides are folded at generation, with cell terms hoisted outside contributors.
     * @param code nonnull builder
     * @param access normalized static access
     * @param coordinates primitive long coordinate slots
     * @param selected selected-axis membership
     * @param contribution whether to append selected rather than retained-axis terms
     * @param integer whether addresses and coordinates have a complete int-bound heap proof
     */
    private static void aggregateAddressTerms(CodeBuilder code,
            io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Binding access,
            int[] coordinates, boolean[] selected, boolean contribution, boolean integer) {
        for (int axis = 0; axis < coordinates.length; axis++) {
            long stride = access.effectiveStrides().get(axis);
            if (selected[axis] != contribution || stride == 0 || access.extents().get(axis) <= 1) continue;
            code.loadLocal(integer ? TypeKind.INT : TypeKind.LONG,coordinates[axis]);
            if (stride != 1) { aggregateConstant(code,integer,stride); if (integer) code.imul(); else code.lmul(); }
            aggregateAdd(code,integer);
        }
    }

    /** Emits a proved coordinate-width constant during generation.
     * @param code nonnull builder
     * @param integer whether the coordinate has an int-bound proof
     * @param value static coordinate or stride
     * @return the same builder for following primitive emission
     * @throws ArithmeticException if an asserted int proof is inconsistent */
    private static CodeBuilder aggregateConstant(CodeBuilder code, boolean integer, long value) {
        return integer ? code.loadConstant(Math.toIntExact(value)) : code.loadConstant(value);
    }

    /** Appends the coordinate-width primitive ADD, not a represented arithmetic site.
     * @param code nonnull builder
     * @param integer whether coordinates use int rather than long */
    private static void aggregateAdd(CodeBuilder code, boolean integer) { if (integer) code.iadd(); else code.ladd(); }

    /**
     * Composes already-bound pointwise sites without choosing a traversal or access policy.
     * Ordinary entries bind boundary loads/stores at entry/exit; step adapters bind operands
     * on demand and publish represented results immediately. Every action runs during code
     * generation only, preserving its caller's instruction and local-allocation order.
     * @param instructions non-null ordered sites, borrowed without mutation
     * @param entryLoads non-null dominating load emission
     * @param operandLoads non-null per-site operand binding emission
     * @param emitInstruction non-null existing scalar or vector arithmetic emission
     * @param resultStores non-null exact producing-site publication emission
     * @param exitStores non-null completed-body publication emission
     * @throws NullPointerException if the instruction list, a site or an action is null
     */
    static void emitBoundPointwise(List<CpuKernelIr.Instruction> instructions, Runnable entryLoads,
            java.util.function.Consumer<CpuKernelIr.Instruction> operandLoads,
            java.util.function.Consumer<CpuKernelIr.Instruction> emitInstruction,
            java.util.function.Consumer<CpuKernelIr.Instruction> resultStores, Runnable exitStores) {
        entryLoads.run();
        for (var instruction : instructions) {
            operandLoads.accept(instruction);
            emitInstruction.accept(instruction);
            resultStores.accept(instruction);
        }
        exitStores.run();
    }

    /**
     * Emits deterministic verified bytes for one exact structural specialization.
     * Current CPU numerical admission is authenticated independently before emission.
     *
     * @param specialization non-null typed carrier, strategy, and compatibility facts
     * @param kernelIr non-null canonical typed CPU kernel IR matching the specialization
     * @return a new deterministic verified class-byte array
     *     with constant predefined native-order layouts for the ordinary B-only path, avoiding
     *     per-invocation layout reconstruction while all static branches remain unchanged
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if specialization and IR facts disagree, including a
     *     cross-type CAST paired with a non-scalar strategy or a class-identity schema other than
     *     {@code 60}, or a vector Conv2d/Conv3d specialization without class-identity schema
     *     {@code 63}, a regional/Scan envelope differing from its typed facts, or a selected
     *     region method exceeding its cold-proved 48000-byte ceiling; size-proof failure is a
     *     defect, never permission to change topology/resources during finalization; selected
     *     exec-v1 artifacts also reject missing/mistyped helpers or absent typed vector math,
     *     rather than relabeling a scalar body as vector execution
     */
    public byte[] generateClassBytes(CpuKernelSpecialization specialization, CpuKernelIr kernelIr) {
        ClassDesc owner = ClassDesc.of(CpuGeneratorSchema.generatedBinaryName(specialization));
        byte[] bytes = ClassFile.of().build(owner, classBuilder -> {
            classBuilder.withVersion(ClassFile.JAVA_26_VERSION, 0).withFlags(AccessFlag.FINAL);
            emitMethods(classBuilder, owner, "", specialization, kernelIr);
        });
        verify(specialization, kernelIr, bytes);
        return bytes;
    }

    /**
     * Describes one method actually declared by a complete generation-time emission.
     * The inventory contains declaration order, not a second family registry or runtime binding.
     *
     * @param name exact declared name, including the caller's namespace
     * @param type exact ordinary descriptor, including a control trailer when selected
     * @param accessFlags exact Class-File access mask used for the declaration
     */
    record GeneratedMethod(String name, MethodTypeDesc type, int accessFlags) { }

    /**
     * Emits one already-selected unit directly into its declaring class, with no wrapper call.
     * Numerical admission and the complete specialization/IR agreement are checked independently
     * before declaring any method. Every emitted method retains its ordinary carrier, scratch,
     * geometry and range ABI; descriptor-derived generated locals follow all parameters.
     *
     * <p>The empty prefix preserves standalone declaration order and names, including a selected
     * Scan's range helper before its entry. A nonempty prefix namespaces every ordinary declaration
     * and intra-unit direct call. The controlled overload adds one invocation-owned segment
     * without changing existing parameter slots. General Scan is not composable through this
     * seam. The caller owns unique JVM-valid names and whole-class verification.</p>
     *
     * @param classBuilder non-null destination, retaining ownership of the complete class
     * @param owner non-null descriptor of that declaring class, used for direct static calls
     * @param prefix non-null deterministic namespace, empty for standalone production generation
     * @param specialization non-null selected carrier, strategy and compatibility facts
     * @param kernelIr non-null canonical typed envelope matching the specialization
     * @return immutable actual declaration inventory in emission order
     * @throws NullPointerException if an argument is null
     * @throws IllegalArgumentException if current CPU numerical admission or specialization/IR
     *     validation fails, or a nonempty prefix requests a nested general Scan
     */
    static List<GeneratedMethod> emitMethods(ClassBuilder classBuilder, ClassDesc owner, String prefix,
            CpuKernelSpecialization specialization, CpuKernelIr kernelIr) {
        return emitMethods(classBuilder, owner, prefix, specialization, kernelIr, false);
    }

    /**
     * Emits an ordinary unit with an optional invocation-owned control trailer.
     * Every declaration and subsidiary call appends exactly one MemorySegment when controlled;
     * the original carrier, scratch, geometry and range parameter slots do not move.
     *
     * @param classBuilder non-null destination class
     * @param owner non-null declaring class
     * @param prefix non-null unique generation-time namespace
     * @param specialization non-null selected ordinary specialization
     * @param kernelIr non-null matching ordinary family envelope
     * @param controlled whether to append the invocation-owned control segment
     * @return immutable complete actual declaration inventory
     * @throws IllegalArgumentException if the unit is invalid or is a controlled nested Scan
     */
    static List<GeneratedMethod> emitMethods(ClassBuilder classBuilder, ClassDesc owner, String prefix,
            CpuKernelSpecialization specialization, CpuKernelIr kernelIr, boolean controlled) {
        Objects.requireNonNull(classBuilder, "classBuilder");
        return declareMethods(classBuilder, owner, prefix, specialization, kernelIr, controlled);
    }

    /**
     * Emits a controlled ordinary helper with independently proved original-row mappings.
     * @param classBuilder non-null destination class
     * @param owner non-null declaring class
     * @param prefix non-null unique helper namespace
     * @param specialization non-null ordinary specialization
     * @param kernelIr non-null matching family envelope
     * @param control non-null proved boundary/value row mappings
     * @return immutable complete declaration inventory
     */
    static List<GeneratedMethod> emitMethods(ClassBuilder classBuilder, ClassDesc owner, String prefix,
            CpuKernelSpecialization specialization, CpuKernelIr kernelIr,
            io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanControl control) {
        return CpuCarrierEmitter.withControlFacts(Objects.requireNonNull(control, "control"),
                () -> emitMethods(classBuilder, owner, prefix, specialization, kernelIr, true));
    }

    /**
     * Obtains declarations without generating numerical code, through the emission owner itself.
     * @param owner non-null declaring class
     * @param prefix non-null deterministic namespace
     * @param specialization non-null selected specialization
     * @param kernelIr non-null matching typed envelope
     * @param controlled whether every ordinary declaration has a control trailer
     * @return immutable declaration inventory, including all subsidiary methods
     */
    static List<GeneratedMethod> methodInventory(ClassDesc owner, String prefix,
            CpuKernelSpecialization specialization, CpuKernelIr kernelIr, boolean controlled) {
        return declareMethods(null, owner, prefix, specialization, kernelIr, controlled);
    }

    private static List<GeneratedMethod> declareMethods(ClassBuilder classBuilder, ClassDesc owner,
            String prefix, CpuKernelSpecialization specialization, CpuKernelIr kernelIr,
            boolean controlled) {
        Objects.requireNonNull(owner, "owner");
        Objects.requireNonNull(prefix, "prefix");
        validate(specialization, kernelIr);
        if ((!prefix.isEmpty() || controlled) && specialization.generalScanIr().isPresent())
            throw new IllegalArgumentException("nested general Scan emission is not supported");
        if (specialization.generalScanIr().filter(scan -> scan.composition().isPresent()).isPresent())
            return new CpuGeneralScanBodyEmitter().emitMethods(classBuilder, owner, prefix,
                    specialization, specialization.generalScanIr().orElseThrow());
        MethodTypeDesc ordinaryType = MethodTypeDesc.ofDescriptor(specialization.entryType().descriptorString());
        // Every controlled composition boundary is already normalized into its owning arena.
        if (controlled && specialization.movementOperandTable())
            ordinaryType = ordinaryType.changeParameterType(0, SEGMENT);
        MethodTypeDesc type = controlled
                ? ordinaryType.insertParameterTypes(ordinaryType.parameterCount(), SEGMENT) : ordinaryType;
        int entryFlags = AccessFlag.STATIC.mask()
                | (kernelIr.familyIdentity().startsWith("loss:")
                    || specialization.generalScanIr().isPresent()
                    || kernelIr.familyIdentity().startsWith("arg-extrema:")
                    || kernelIr.familyIdentity().startsWith("masked-reduction:")
                    || kernelIr.familyIdentity().startsWith("advanced-reduction:")
                    || kernelIr.familyIdentity().startsWith("softmax:")
                    || kernelIr.familyIdentity().startsWith("trailing-normalization:")
                    || kernelIr.familyIdentity().startsWith("batch-normalization-inference:")
                    || kernelIr.familyIdentity().startsWith("batch-normalization-training:")
                    || kernelIr.familyIdentity().startsWith("conv2d:")
                    || kernelIr.familyIdentity().startsWith("conv3d:")
                    || kernelIr.familyIdentity().startsWith("matmul:")
                    || kernelIr.familyIdentity().startsWith("pool2d:")
                    || kernelIr.familyIdentity().startsWith("pool3d:")
                    || kernelIr.familyIdentity().startsWith("attention:")
                        ? AccessFlag.PUBLIC.mask() : 0);
        if (kernelIr.familyIdentity().startsWith("loss:")) {
            return new CpuLossEmitter().emit(classBuilder, owner, prefix, type, specialization, kernelIr);
        }
        var methods = new ArrayList<GeneratedMethod>();
        if (specialization.rowMapReductionIr().isPresent() && specialization.rowFusionEmission().isPresent()) {
            for (int phase = 0; phase < 3; phase++) {
                int selectedPhase = phase;
                String name = prefix + (phase == 0 ? CpuGeneratorSchema.ENTRY_NAME
                        : phase == 1 ? "ordinaryRowRange" : "ordinaryRowMerge");
                declareMethod(classBuilder, name, type, AccessFlag.STATIC.mask(), controlled, code -> {
                    var types = new ArrayList<>(specialization.boundaryDataTypes()); types.add(DataType.INT64);
                    var patterns = new ArrayList<>(specialization.carrierPattern());
                    patterns.add(CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT);
                    // Scalar partials are checked native cells even when every ordinary
                    // boundary is a heap array. Their typed layout must dominate both
                    // range publication and merge reads, independently of boundary layouts.
                    if (selectedPhase != 0 && specialization.executionStrategy().compute()
                            == io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy.Compute.SCALAR) {
                        types.add(specialization.boundaryDataTypes().getFirst());
                        patterns.add(CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT);
                    }
                    CpuCarrierEmitter.prepareSegmentLayouts(code,types,patterns,true,false);
                    emitRowRegion(code,specialization,selectedPhase);
                    code.return_();
                });
                methods.add(new GeneratedMethod(name, type, AccessFlag.STATIC.mask()));
            }
            return List.copyOf(methods);
        }
        if (specialization.generalScanIr().isPresent() && specialization.rowFusionEmission().isPresent()
                && specialization.rowFusionEmission().orElseThrow().maximumPartialCount() > 1) {
            var helperType = MethodTypeDesc.of(ConstantDescs.CD_void,
                    ClassDesc.of("java.lang.foreign.MemorySegment").arrayType(),ConstantDescs.CD_long.arrayType(),
                    ConstantDescs.CD_long,ConstantDescs.CD_long,ConstantDescs.CD_long);
            declareMethod(classBuilder, "scanRowRange", helperType, AccessFlag.STATIC.mask(), false, code -> {
                CpuCarrierEmitter.prepareSegmentLayouts(code,List.of(specialization.generalScanIr().orElseThrow().dataType(),DataType.INT64),
                        List.of(CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT,CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT),true,false);
                new CpuGeneralScanEmitter().emitRowRange(code,specialization,specialization.generalScanIr().orElseThrow());
                code.return_();
            });
            methods.add(new GeneratedMethod("scanRowRange", helperType, AccessFlag.STATIC.mask()));
        }
        String entryName = prefix + CpuGeneratorSchema.ENTRY_NAME;
        declareMethod(classBuilder, entryName, type, entryFlags, controlled, code -> {
                        if (specialization.movementOperandTable()) {
                            new CpuDataMovementEmitter().emit(code, specialization, kernelIr);
                            code.return_(); return;
                        }
                        if (specialization.softmaxBackwardIr().isPresent()) {
                            CpuCarrierEmitter.prepareSegmentLayouts(code,specialization.boundaryDataTypes(),
                                    specialization.carrierPattern(),true,false);
                            new CpuSoftmaxBackwardEmitter().emit(code,specialization);
                            code.return_(); return;
                        }
                        if (specialization.aggregateFusionIr().isPresent()) {
                            CpuCarrierEmitter.prepareSegmentLayouts(code,specialization.boundaryDataTypes(),
                                    specialization.carrierPattern(),true,false);
                            emitAggregateFusion(code,specialization);
                            code.return_(); return;
                        }
                        if (specialization.rowMapReductionIr().isPresent()) {
                            CpuCarrierEmitter.prepareSegmentLayouts(code,specialization.boundaryDataTypes(),
                                    specialization.carrierPattern(),true,false);
                            emitRowRegion(code,specialization);
                            code.return_(); return;
                        }
                        if (specialization.generalScanIr().isPresent()) {
                            var scanTypes = new ArrayList<>(specialization.boundaryDataTypes());
                            var scanCarriers = new ArrayList<>(specialization.carrierPattern());
                            if (specialization.rowFusionEmission().isPresent()) {
                                scanTypes.add(DataType.INT64); scanCarriers.add(CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT);
                            }
                            CpuCarrierEmitter.prepareSegmentLayouts(code, scanTypes,
                                    scanCarriers, true,
                                    specialization.generalScanIr().orElseThrow().symbolic()
                                            && specialization.generalScanIr().orElseThrow().lengthAware());
                            new CpuGeneralScanEmitter().emit(code, specialization,
                                    specialization.generalScanIr().orElseThrow());
                            code.return_();
                            return;
                        }
                        if (usesSharedCarrierLayouts(kernelIr)
                                && !CpuAffineCopyEmitter
                                        .ownsGeneralLongDenseResultCarrierAccess(
                                                specialization, kernelIr)
                                && !kernelIr.familyIdentity().startsWith("conv2d:")
                                && !kernelIr.familyIdentity().startsWith("conv3d:")
                                && !kernelIr.familyIdentity().startsWith("pool2d:")
                                && !kernelIr.familyIdentity().startsWith("pool3d:")
                                && !kernelIr.familyIdentity().startsWith("attention:")) {
                            CpuCarrierEmitter.prepareSegmentLayouts(code,
                                    specialization.boundaryDataTypes(),
                                    hasProvedVectorScalarPower(specialization, kernelIr)
                                            ? accessedCarrierPattern(kernelIr, specialization)
                                            : specialization.carrierPattern(),
                                    controlled || kernelIr.familyIdentity().startsWith("aggregate:")
                                        || kernelIr.familyIdentity().startsWith("constant-write-v1:") || specialization.ordinaryBatch());
                        }
                        if (kernelIr.instructions().isEmpty()) {
                            if (kernelIr.familyIdentity().startsWith("constant-write-v1:")) {
                                emitConstantWrite(code, specialization, kernelIr);
                            } else if (kernelIr.familyIdentity().startsWith("attention:")) {
                                new CpuAttentionEmitter().emit(code, specialization, kernelIr);
                            } else if (kernelIr.familyIdentity().startsWith("pool2d:")) {
                                new CpuPool2dEmitter().emit(code, specialization, kernelIr);
                            } else if (kernelIr.familyIdentity().startsWith("pool3d:")) {
                                new CpuPool3dEmitter().emit(code, specialization, kernelIr);
                            } else if (kernelIr.familyIdentity().startsWith("matmul:")) {
                                new CpuMatmulEmitter().emit(code, specialization,
                                        specialization.matmulIr().orElseThrow(() ->
                                            new IllegalArgumentException(
                                                "MATMUL specialization lacks typed facts")));
                            } else if (kernelIr.familyIdentity().startsWith("conv3d:")) {
                                new CpuConv3dEmitter().emit(code, specialization, kernelIr);
                            } else if (kernelIr.familyIdentity().startsWith("conv2d:")) {
                                new CpuConv2dEmitter().emit(code, specialization, kernelIr);
                            } else if (kernelIr.familyIdentity().startsWith("batch-normalization-training:")) {
                                new CpuBatchNormTrainingEmitter().emit(code, specialization, kernelIr);
                            } else if (kernelIr.familyIdentity().startsWith("batch-normalization-inference:")) {
                                new CpuBatchNormInferenceEmitter().emit(code, specialization, kernelIr);
                            } else if (kernelIr.familyIdentity().startsWith("trailing-normalization:LAYER:")) {
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
                            emitBoundPointwise(kernelIr.instructions(), () -> {
                            for (int boundary = 0; boundary < boundaries.size(); boundary++) {
                                CpuKernelIr.Value value = boundaries.get(boundary);
                                if (value.kind() != CpuKernelIr.Value.Kind.INPUT
                                        || !requiresInputLoad(kernelIr, value.ordinal())) continue;
                                carriers.load(value.dataType(), specialization.carrierPattern().get(boundary),
                                        boundary, state.addresses()[boundary], state.intAddresses());
                                store(code, value.dataType(), scalarLocals[value.ordinal()]);
                            }
                            }, instruction -> {}, instruction -> scalar.emit(kernelIr, instruction, scalarLocals),
                            instruction -> {}, () -> {
                            for (CpuKernelIr.Store store : kernelIr.stores()) {
                                CpuKernelIr.Value value = kernelIr.values().get(store.value());
                                int boundary = boundaryIndex(boundaries, value.ordinal());
                                carriers.store(value.dataType(), specialization.carrierPattern().get(boundary),
                                        boundary, state.addresses()[boundary], scalarLocals[value.ordinal()],
                                        state.intAddresses());
                            }
                            });
                        };
                        if (specialization.executionStrategy().compute()
                                == io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy.Compute.VECTOR) {
                            DataType vectorType = vectorDataType(kernelIr);
                            carriers.prepareVectorSpecies(vectorType);
                            boolean provedScalarPower = hasProvedVectorScalarPower(
                                    specialization, kernelIr);
                            if (provedScalarPower && requiresVectorByteOrder(kernelIr,
                                    boundaries, specialization.carrierPattern())) {
                                carriers.prepareVectorByteOrder();
                            }
                            if (usesDenseFloatingMaskBoundary(kernelIr, vectorType)) {
                                carriers.prepareMaskByteBridge(specialization.vectorSpeciesBitSize()
                                        / vectorType.bitWidth());
                            }
                            int[] vectorLocals = allocateVectorLocals(code, kernelIr);
                            var vectorInstructions = new CpuVectorInstructionEmitter(
                                    code, kernelIr, vectorType);
                            int lanes = specialization.vectorSpeciesBitSize() / vectorType.bitWidth();
                            boolean[] accessed = provedScalarPower
                                    ? accessedBoundaries(kernelIr, boundaries) : null;
                            boolean accessedDense = provedScalarPower
                                    && allAccessedBoundariesAreDense(plans, accessed);
                            boolean denseArrayInt = provedScalarPower
                                    ? accessedDense && usesOnlyArrayVectorCarriers(kernelIr,
                                            boundaries, specialization.carrierPattern())
                                    : specialization.loopAddressing(kernelIr)
                                            == CpuKernelSpecialization.LoopAddressing
                                                    .DENSE_HEAP_ARRAY_INT;
                            if (denseArrayInt && provedScalarPower)
                                loops.emitDenseArrayIntVector(plans, accessed, lanes,
                                        state -> emitVectorBody(code, carriers, vectorInstructions,
                                                specialization, kernelIr, boundaries, state,
                                                vectorLocals, vectorType), scalarBody);
                            else if (denseArrayInt)
                                loops.emitDenseArrayIntVector(plans, lanes,
                                        state -> emitVectorBody(code, carriers, vectorInstructions,
                                                specialization, kernelIr, boundaries, state,
                                                vectorLocals, vectorType), scalarBody);
                            else if (provedScalarPower && accessedDense)
                                loops.emitDenseLongVector(plans, accessed, lanes,
                                        state -> emitVectorBody(code, carriers, vectorInstructions,
                                                specialization, kernelIr, boundaries, state,
                                                vectorLocals, vectorType), scalarBody);
                            else if (provedScalarPower)
                                loops.emitVector(plans, accessed, lanes,
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
                    });
        methods.add(new GeneratedMethod(entryName, type, entryFlags));
        return List.copyOf(methods);
    }

    /**
     * Declares one method, or visits only its declaration when collecting an inventory.
     * Descriptor scope is cold generation state only and is restored even if emission fails.
     * @param builder destination, or null for declaration-only collection
     * @param name exact method name
     * @param type complete JVM descriptor
     * @param flags exact access mask
     * @param controlled whether the last parameter is an invocation-owned control segment
     * @param body numerical bytecode emitter, not called during inventory collection
     */
    static void declareMethod(ClassBuilder builder, String name, MethodTypeDesc type, int flags,
            boolean controlled, java.util.function.Consumer<CodeBuilder> body) {
        if (builder != null) builder.withMethod(name, type, flags, method ->
                method.withCode(code -> CpuCarrierEmitter.withDescriptor(type, controlled,
                        () -> {
                            CpuCarrierEmitter.emitOccurrenceGuard(code);
                            body.accept(code);
                        })));
    }

    /**
     * Binds an ordinary full-body region to the established typed entry ABI. Feature cells are
     * generation-specialized, rows are not unrolled; every operand load remains inside the row
     * scope. The numeric entry pack contains invocation B followed by boundary element bases.
     * @param code method builder owning every local
     * @param specialization already proved ordinary region, scalar v1 or selected scalar/vector exec-v1
     */
    private static void emitRowRegion(java.lang.classfile.CodeBuilder code,CpuKernelSpecialization specialization) {
        emitRowRegion(code,specialization,0);
    }

    /**
     * Emits a complete scalar/vector region or its selected private range/ordered-merge helper. Phase
     * is a generation fact only. Every helper uses the same authenticated carrier/access facts;
     * ranges write disjoint private terminal slots and validity, never shared floating results.
     * @param code builder owning all locals
     * @param specialization exact row-region compute strategy and selected helper layout
     * @param phase zero for full unit, one for a private row range, two for caller-side merge
     */
    private static void emitRowRegion(java.lang.classfile.CodeBuilder code,CpuKernelSpecialization specialization,int phase) {
        if (phase == 0) {
            int geometry = specialization.carrierPattern().size()+(specialization.scratchParameter() ? 1 : 0);
            int start = geometry+1, end = start+2;
            var invalid = code.newLabel(); var empty = code.newLabel(); var valid = code.newLabel();
            code.lload(start).loadConstant(0L).lcmp().branch(java.lang.classfile.Opcode.IFNE,invalid);
            code.lload(end).loadConstant(0L).lcmp().branch(java.lang.classfile.Opcode.IFEQ,empty);
            code.lload(end).loadConstant(1L).lcmp().branch(java.lang.classfile.Opcode.IFEQ,valid);
            code.labelBinding(invalid).new_(ClassDesc.of(IllegalArgumentException.class.getName())).dup()
                    .invokespecial(ClassDesc.of(IllegalArgumentException.class.getName()),"<init>",MethodTypeDesc.of(ConstantDescs.CD_void)).athrow();
            code.labelBinding(empty).return_().labelBinding(valid);
        }
        var ir = specialization.rowMapReductionIr().orElseThrow();
        boolean terminalProduct = ir.contractions().stream().anyMatch(c -> ir.terminalOutputs().contains(c.output()));
        if (phase == 2 || !terminalProduct) {
            emitRowRegionBody(code,specialization,phase,false);
        } else if (!specialization.ordinaryBatch()) {
            int anchor = ir.rowSites().getFirst();
            emitRowRegionBody(code,specialization,phase,
                    ir.accesses().get(anchor).extentConstants().get(ir.rowAxes().get(anchor)) == 1);
        } else {
            int geometry = specialization.carrierPattern().size()+(specialization.scratchParameter() ? 1 : 0);
            CpuMatmulEmitter.emitSingletonChoice(code,TypeKind.INT,
                    () -> code.aload(geometry).loadConstant(0).laload().l2i(),
                    directProduct -> emitRowRegionBody(code,specialization,phase,directProduct));
        }
    }

    /**
     * Emits one selected numerical body with fresh generation-local load caches.
     * @param code method builder owning all branch-local slots
     * @param specialization checked ordinary region
     * @param phase full unit, private range, or merge phase
     * @param directProduct whether the complete logical row contraction is singleton
     */
    private static void emitRowRegionBody(java.lang.classfile.CodeBuilder code,
            CpuKernelSpecialization specialization,int phase,boolean directProduct) {
        if (specialization.executionStrategy().compute()
                == io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy.Compute.VECTOR) {
            if (phase == 2) emitVectorRowMerge(code,specialization); else emitVectorRowRegion(code,specialization,phase,directProduct);
            return;
        }
        var ir = specialization.rowMapReductionIr().orElseThrow();
        DataType type = ir.arithmetic().values().getFirst().dataType();
        var kind = type == DataType.FLOAT64 ? java.lang.classfile.TypeKind.DOUBLE : java.lang.classfile.TypeKind.FLOAT;
        var carriers = new CpuCarrierEmitter(code); var scalar = new CpuScalarEmitter(code);
        int count = specialization.carrierPattern().size(), geometry = count+(specialization.scratchParameter() ? 1 : 0), start = geometry+1, end = start+2;
        if (phase == 2) {
        var invalid = code.newLabel(); var empty = code.newLabel(); var valid = code.newLabel();
        code.lload(start).loadConstant(0L).lcmp().branch(java.lang.classfile.Opcode.IFNE,invalid);
        code.lload(end).loadConstant(0L).lcmp().branch(java.lang.classfile.Opcode.IFEQ,empty);
        code.lload(end).loadConstant(1L).lcmp().branch(java.lang.classfile.Opcode.IFEQ,valid);
        code.labelBinding(invalid).new_(java.lang.constant.ClassDesc.of(IllegalArgumentException.class.getName()))
                .dup().invokespecial(java.lang.constant.ClassDesc.of(IllegalArgumentException.class.getName()),"<init>",
                        java.lang.constant.MethodTypeDesc.of(java.lang.constant.ConstantDescs.CD_void)).athrow();
        code.labelBinding(empty).return_().labelBinding(valid);
        }
        int row = code.allocateLocal(java.lang.classfile.TypeKind.INT), limit = code.allocateLocal(java.lang.classfile.TypeKind.INT);
        int ref = ir.rowSites().getFirst(), axis = ir.rowAxes().get(ref);
        if (specialization.ordinaryBatch()) code.aload(geometry).loadConstant(0).laload().l2i();
        else code.loadConstant(Math.toIntExact(ir.accesses().get(ref).extentConstants().get(axis)));
        code.istore(limit);
        int[][] locals = new int[ir.arithmetic().values().size()][], sums = new int[ir.terminalOutputs().size()][];
        int[] boundary = new int[locals.length]; java.util.Arrays.fill(boundary,-1);
        int next = 0;
        for (var value : ir.arithmetic().values()) if (value.kind() == CpuKernelIr.Value.Kind.INPUT) boundary[value.ordinal()] = next++;
        for (var value : ir.arithmetic().values()) if (value.kind() == CpuKernelIr.Value.Kind.OUTPUT) boundary[value.ordinal()] = next++;
        for (var value : ir.arithmetic().values()) {
            int cells = regionCells(ir,value.ordinal(),ir.rowAxes().get(value.ordinal()) >= 0);
            locals[value.ordinal()] = new int[cells]; java.util.Arrays.fill(locals[value.ordinal()],-1);
            if (ir.rowSites().contains(value.ordinal())) for (int j = 0; j < cells; j++) locals[value.ordinal()][j] = code.allocateLocal(kind);
        }
        for (int t = 0; t < sums.length; t++) {
            sums[t] = new int[regionCells(ir,ir.terminalOutputs().get(t),false)];
            for (int j = 0; j < sums[t].length; j++) sums[t][j] = code.allocateLocal(kind);
        }
        for (int v = 0; v < locals.length; v++) if (ir.viewOrigins().get(v) != v
                && ir.rowSites().contains(ir.viewOrigins().get(v))) for (int cell = 0; cell < locals[v].length; cell++)
            locals[v][cell] = regionValueLocal(ir,locals,v,regionCoordinates(ir,v,cell,ir.rowAxes().get(v) >= 0));
        int[] operands = new int[]{code.allocateLocal(kind),code.allocateLocal(kind),code.allocateLocal(kind)};
        int address = code.allocateLocal(java.lang.classfile.TypeKind.LONG);
        var sourceOrigins = ir.arithmetic().values().stream().filter(v -> v.kind() == CpuKernelIr.Value.Kind.INPUT
                && ir.viewOrigins().get(v.ordinal()) == v.ordinal()).map(v -> v.ordinal()).toList();
        var staging = phase == 2 ? List.<Integer>of()
                : io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRowMapReductionLowering
                    .invariantLoadStaging(ir,sourceOrigins,List.of(),112,116);
        var staged = new java.util.LinkedHashMap<List<Long>,Integer>();
        for (int v : staging) for (int cell = 0; cell < regionCells(ir,v,false); cell++) {
            var key = regionScalarKey(ir,v,regionCoordinates(ir,v,cell,false));
            if (!staged.containsKey(key)) {
                int slot = code.allocateLocal(kind); staged.put(key,slot);
                if (type == DataType.FLOAT64) code.loadConstant(0d); else code.loadConstant(0f);
                code.storeLocal(kind,slot);
            }
        }
        int cacheSlots = Math.max(0,regionScalarBudget(ir)-(phase == 0 ? 0 : 4));
        int rowOffset = cacheSlots < 2 ? -1 : code.allocateLocal(TypeKind.LONG);
        int cacheCapacity = Math.min(Math.max(0,cacheSlots-2)/(type == DataType.FLOAT64 ? 2 : 1),
                staged.isEmpty() ? Integer.MAX_VALUE : Math.max(0,128-(phase == 0 ? 112 : 116)
                    -1-staged.size()*(type == DataType.FLOAT64 ? 2 : 1))/(type == DataType.FLOAT64 ? 2 : 1));
        long[] rowRecipe = {Long.MIN_VALUE,Long.MIN_VALUE};
        var remaining = new java.util.HashMap<List<Long>,Integer>();
        regionScalarTrace(ir).forEach(k -> remaining.merge(k,1,Integer::sum));
        var cached = new java.util.HashMap<List<Long>,Integer>();
        var free = new java.util.ArrayDeque<Integer>();
        int[] cacheEpoch = {0};
        int[] bases = new int[count];
        for (int b = 0; b < count; b++) {
            bases[b] = code.allocateLocal(java.lang.classfile.TypeKind.LONG);
            code.aload(geometry).loadConstant(1+b).laload().lstore(bases[b]);
        }
        int[][] strides = new int[locals.length][];
        var strideFacts = new java.util.HashMap<java.util.List<Long>,Integer>();
        for (int v = 0; v < strides.length; v++) {
            var access = ir.accesses().get(v); strides[v] = new int[access.plan().iterationRank()];
            java.util.Arrays.fill(strides[v],-1);
            for (int a = 0; a < strides[v].length; a++) {
                long sc = access.strideConstants().get(a), sf = access.strideBatchFactors().get(a);
                if (sf == 0) continue;
                var key = java.util.List.of(sc,sf); Integer fact = strideFacts.get(key);
                if (fact == null) {
                    fact = code.allocateLocal(java.lang.classfile.TypeKind.LONG);
                    code.iload(limit).i2l(); if (sf != 1) code.loadConstant(sf).lmul();
                    if (sc != 0) code.loadConstant(sc).ladd(); code.lstore(fact); strideFacts.put(key,fact);
                }
                strides[v][a] = fact;
            }
        }
        java.util.function.BiConsumer<Integer,int[]> load = (v,coordinates) -> regionLoad(code,specialization,ir,
                carriers,locals,boundary,bases,strides,row,address,rowOffset,rowRecipe,remaining,cached,free,cacheEpoch,cacheCapacity,staged,v,coordinates);
        if (!staged.isEmpty()) {
            var noConsumer = code.newLabel();
            code.iload(limit).branch(java.lang.classfile.Opcode.IFLE,noConsumer);
            if (phase == 1) code.lload(start).lload(end).lcmp().branch(java.lang.classfile.Opcode.IFGE,noConsumer);
            var loaded = new java.util.HashSet<List<Long>>();
            for (int v : staging) for (int cell = 0; cell < regionCells(ir,v,false); cell++) {
                int[] coordinates = regionCoordinates(ir,v,cell,false); var key = regionScalarKey(ir,v,coordinates);
                if (!loaded.add(key)) continue;
                int root = ir.viewOrigins().get(v);
                regionAddress(code,ir.accesses().get(v),bases[boundary[root]],strides[v],row,address,coordinates);
                carriers.load(type,specialization.carrierPattern().get(boundary[root]),boundary[root],address,false);
                code.storeLocal(kind,staged.get(key));
            }
            code.labelBinding(noConsumer);
        }
        var sites = new java.util.HashMap<Integer,CpuKernelIr.Instruction>();
        ir.arithmetic().instructions().forEach(i -> sites.put(i.output(),i));
        var products = new java.util.HashMap<Integer,io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr.Contraction>();
        ir.contractions().forEach(c -> products.put(c.output(),c));
        Runnable initialize = () -> { for (var terminal : sums) for (int sum : terminal) CpuMatmulEmitter.emitInit(code,type,sum); };
        Runnable publish = () -> {
            for (int t = 0; t < sums.length; t++) for (int cell = 0; cell < sums[t].length; cell++) {
                int out = ir.terminalOutputs().get(t); int[] coordinates = regionCoordinates(ir,out,cell,false);
                CpuMatmulEmitter.emitFinish(code,type,sums[t][cell],local -> regionStore(code,specialization,ir,
                        carriers,boundary,bases,strides,row,address,out,coordinates,local));
            }
        };
        int partialBase = phase == 0 ? -1 : code.allocateLocal(TypeKind.LONG);
        int validPartial = phase == 0 ? -1 : code.allocateLocal(TypeKind.LONG);
        if (phase == 2) {
            initialize.run();
            int seen = code.allocateLocal(TypeKind.INT);
            int partials = code.allocateLocal(TypeKind.INT);
            code.loadConstant(0).istore(seen).loadConstant(0).istore(row);
            code.aload(geometry).loadConstant(count+1).laload().l2i().istore(partials);
            long cells = java.util.Arrays.stream(sums).mapToLong(a -> a.length).sum();
            long slice = Math.addExact(8,Math.multiplyExact((Math.multiplyExact(cells,type.byteWidth())+7)/8,8));
            long stageCells = 0;
            var shape = specialization.rowFusionEmission().orElseThrow();
            for (var list : List.of(shape.invariantTransposeValues(),shape.temporalTransposeValues()))
                for (int v : list) stageCells += regionCells(ir,v,false);
            long stageBytes = (stageCells*type.byteWidth()+7)/8*8;
            var loop = code.newLabel(); var done = code.newLabel(); var skip = code.newLabel();
            code.labelBinding(loop).iload(row).iload(partials).branch(java.lang.classfile.Opcode.IF_ICMPGE,done)
                    .iload(row).i2l().loadConstant(slice).lmul();
            if (stageBytes != 0) code.loadConstant(stageBytes).ladd();
            code.lstore(partialBase);
            carriers.loadBytes(DataType.INT64,count,partialBase);
            code.loadConstant(0L).lcmp().branch(java.lang.classfile.Opcode.IFEQ,skip);
            long offset = 8;
            for (var terminal : sums) for (int sum : terminal) {
                long cellOffset = offset;
                Runnable loadPartial = () -> {
                    code.lload(partialBase).loadConstant(cellOffset).ladd().lstore(address);
                    carriers.loadBytes(type,count,address);
                };
                var add = code.newLabel(); var finishedCell = code.newLabel();
                code.iload(seen).branch(java.lang.classfile.Opcode.IFNE,add);
                CpuAggregateEmitter.emitRepresentedContributorStep(code,type,sum,true,loadPartial);
                code.branch(java.lang.classfile.Opcode.GOTO,finishedCell).labelBinding(add);
                CpuAggregateEmitter.emitRepresentedContributorStep(code,type,sum,false,loadPartial);
                code.labelBinding(finishedCell);
                offset += type.byteWidth();
            }
            code.loadConstant(1).istore(seen).labelBinding(skip).iinc(row,1)
                    .branch(java.lang.classfile.Opcode.GOTO,loop).labelBinding(done);
            publish.run(); return;
        }
        java.util.function.IntConsumer emitSite = out -> {
                    for (int cell = 0; cell < locals[out].length; cell++) {
                        int target = locals[out][cell]; int[] coordinates = regionCoordinates(ir,out,cell,true);
                        if (products.containsKey(out)) {
                            var c = products.get(out); CpuMatmulEmitter.emitInit(code,type,target);
                            int k = Math.toIntExact(ir.accesses().get(c.left()).extentConstants().get(1));
                            for (int feature = 0; feature < k; feature++) {
                                int f = feature;
                                CpuMatmulEmitter.emitContributorStep(code,type,target,k == 1,
                                        () -> load.accept(c.left(),new int[]{-1,f}),
                                        () -> load.accept(c.right(),new int[]{f,coordinates[1]}));
                            }
                        } else {
                            var instruction = sites.get(out); int[] siteLocals = new int[locals.length];
                            java.util.Arrays.fill(siteLocals,-1); siteLocals[out] = target;
                            CpuClassFileKernelGenerator.emitBoundPointwise(java.util.List.of(instruction),() -> {}, i -> {
                                for (int p = 0; p < i.inputs().size(); p++) {
                                    int in = i.inputs().get(p);
                                    if (siteLocals[in] >= 0) continue;
                                    int direct = regionValueLocal(ir,locals,in,coordinates);
                                    if (direct >= 0) siteLocals[in] = direct;
                                    else {
                                        load.accept(in,coordinates); code.storeLocal(kind,operands[p]); siteLocals[in] = operands[p];
                                    }
                                }
                            },i -> scalar.emit(ir.arithmetic(),i,siteLocals),i -> {},() -> {});
                        }
                        if (ir.retainedStores().contains(out)) regionStore(code,specialization,ir,carriers,boundary,
                                bases,strides,row,address,out,coordinates,target);
                        if (ir.retainedStores().contains(out)) { free.addAll(cached.values()); cached.clear(); cacheEpoch[0]++; }
                    }
                };
        java.util.function.IntConsumer emitTerminal = out -> {
                    var c = products.get(out); int t = ir.terminalOutputs().indexOf(out);
                    for (int cell = 0; cell < sums[t].length; cell++) {
                        int[] coordinates = regionCoordinates(ir,out,cell,false);
                        CpuMatmulEmitter.emitContributorStep(code,type,sums[t][cell],directProduct,
                                () -> load.accept(c.left(),new int[]{coordinates[0],-1}),
                                () -> load.accept(c.right(),new int[]{-1,coordinates[1]}));
                    }
                };
        if (phase == 0) CpuRowMapReductionEmitter.emitBound(code,ir,row,limit,locals,sums,initialize,
                action -> { rowRecipe[0] = Long.MIN_VALUE; rowRecipe[1] = Long.MIN_VALUE; action.run(); },emitSite,emitTerminal,publish);
        else {
            int begin = code.allocateLocal(TypeKind.INT), finish = code.allocateLocal(TypeKind.INT);
            code.lload(start).l2i().istore(begin).lload(end).l2i().istore(finish);
            code.aload(geometry).loadConstant(count+2).laload().lstore(partialBase);
            CpuRowMapReductionEmitter.emitRangeBound(code,ir,row,begin,finish,locals,sums,() -> {
                initialize.run(); code.loadConstant(0L).lstore(validPartial);
                carriers.storeBytes(DataType.INT64,count,partialBase,validPartial);
            }, action -> { code.loadConstant(1L).lstore(validPartial); rowRecipe[0] = Long.MIN_VALUE; rowRecipe[1] = Long.MIN_VALUE; action.run(); },emitSite,emitTerminal,() -> {
                long offset = 8;
                for (var terminal : sums) for (int sum : terminal) {
                    code.lload(partialBase).loadConstant(offset).ladd().lstore(address);
                    carriers.storeBytes(type,count,address,sum); offset += type.byteWidth();
                }
                carriers.storeBytes(DataType.INT64,count,partialBase,validPartial);
            });
        }
    }

    /**
     * Emits fixed feature chunks within one row. Scalar K features are extracted once from an
     * already computed vector and broadcast in original K order; no strided withLane assembly,
     * horizontal reduction, FMA, row-vector mixing or pointwise opcode copy is used.
     * Staged immutable coefficients are loaded once before the row loop into dedicated vector
     * locals, including checked mask-one loads for [K,1] transposes. Empty ranges initialize
     * references only and never read an unstaged bank.
     * @param code builder owning all handles
     * @param specialization cold-proved ordinary vector region and exact staged layout
     * @param phase zero for a complete inline unit, one for a private bounded range
     * @param directProduct whether the complete logical row contraction is singleton
     */
    private static void emitVectorRowRegion(java.lang.classfile.CodeBuilder code,CpuKernelSpecialization specialization,int phase,boolean directProduct) {
        var ir = specialization.rowMapReductionIr().orElseThrow(); var type = ir.arithmetic().values().getFirst().dataType();
        var kind = type == DataType.FLOAT64 ? TypeKind.DOUBLE : TypeKind.FLOAT;
        var vector = ClassDesc.of("jdk.incubator.vector."+(type == DataType.FLOAT64 ? "DoubleVector" : "FloatVector"));
        var species = ClassDesc.of("jdk.incubator.vector.VectorSpecies");
        int lanes = specialization.vectorSpeciesBitSize()/type.bitWidth();
        int count = specialization.carrierPattern().size(), geometry = count+1, start = geometry+1, end = start+2;
        int row = code.allocateLocal(TypeKind.INT), limit = code.allocateLocal(TypeKind.INT);
        int begin = code.allocateLocal(TypeKind.INT), finish = code.allocateLocal(TypeKind.INT);
        int anchor = ir.rowSites().getFirst();
        if (specialization.ordinaryBatch()) code.aload(geometry).loadConstant(0).laload().l2i();
        else code.loadConstant(Math.toIntExact(ir.accesses().get(anchor).extentConstants().get(ir.rowAxes().get(anchor))));
        code.istore(limit);
        if (phase == 0) code.loadConstant(0).istore(begin).iload(limit).istore(finish);
        else code.lload(start).l2i().istore(begin).lload(end).l2i().istore(finish);
        var carrier = new CpuCarrierEmitter(code); carrier.prepareVectorSpecies(type); carrier.prepareVectorByteOrder();
        var arithmetic = new CpuVectorInstructionEmitter(code,ir.arithmetic(),type);
        int address = code.allocateLocal(TypeKind.LONG);
        int partial = code.allocateLocal(TypeKind.LONG), validity = code.allocateLocal(TypeKind.LONG);
        if (phase == 1) code.aload(geometry).loadConstant(count+2).laload().lstore(partial);
        int[] boundary = new int[ir.arithmetic().values().size()]; java.util.Arrays.fill(boundary,-1);
        int next = 0;
        for (var v : ir.arithmetic().values()) if (v.kind() == CpuKernelIr.Value.Kind.INPUT) boundary[v.ordinal()] = next++;
        for (var v : ir.arithmetic().values()) if (v.kind() == CpuKernelIr.Value.Kind.OUTPUT) boundary[v.ordinal()] = next++;
        int[] bases = new int[count];
        for (int b = 0; b < count; b++) {
            bases[b] = code.allocateLocal(TypeKind.LONG);
            code.aload(geometry).loadConstant(1+b).laload().lstore(bases[b]);
        }
        int[][] strides = new int[boundary.length][];
        var strideFacts = new java.util.HashMap<List<Long>,Integer>();
        for (int v = 0; v < strides.length; v++) {
            var a = ir.accesses().get(v); strides[v] = new int[a.plan().iterationRank()]; java.util.Arrays.fill(strides[v],-1);
            for (int axis = 0; axis < strides[v].length; axis++) if (a.strideBatchFactors().get(axis) != 0) {
                var key = List.of(a.strideConstants().get(axis),a.strideBatchFactors().get(axis));
                Integer fact = strideFacts.get(key);
                if (fact == null) {
                    fact = code.allocateLocal(TypeKind.LONG); code.iload(limit).i2l();
                    if (key.get(1) != 1) code.loadConstant(key.get(1)).lmul();
                    if (key.get(0) != 0) code.loadConstant(key.get(0)).ladd();
                    code.lstore(fact); strideFacts.put(key,fact);
                }
                strides[v][axis] = fact;
            }
        }
        var masks = new java.util.HashMap<Integer,Integer>();
        for (int v : java.util.stream.Stream.concat(ir.rowSites().stream(),ir.terminalOutputs().stream()).toList()) {
            int n = Math.toIntExact(ir.accesses().get(v).extentConstants().getLast());
            for (int column = 0; column < n; column += lanes) {
                int used = Math.min(lanes,n-column);
                if (masks.containsKey(used)) continue;
                int mask = code.allocateLocal(TypeKind.REFERENCE); masks.put(used,mask);
                var maskType = ClassDesc.of("jdk.incubator.vector.VectorMask");
                code.getstatic(vector,"SPECIES_PREFERRED",species).loadConstant(used == 64 ? -1L : (1L<<used)-1)
                        .invokestatic(maskType,"fromLong",MethodTypeDesc.of(maskType,species,ConstantDescs.CD_long)).astore(mask);
            }
        }
        int[][] values = new int[boundary.length][], sums = new int[ir.terminalOutputs().size()][];
        for (int v = 0; v < values.length; v++) {
            values[v] = new int[regionVectorBlocks(ir,v,ir.rowAxes().get(v) >= 0,lanes)]; java.util.Arrays.fill(values[v],-1);
            if (ir.rowSites().contains(v)) for (int i = 0; i < values[v].length; i++) values[v][i] = code.allocateLocal(TypeKind.REFERENCE);
        }
        for (int t = 0; t < sums.length; t++) {
            sums[t] = new int[regionVectorBlocks(ir,ir.terminalOutputs().get(t),false,lanes)];
            for (int i = 0; i < sums[t].length; i++) sums[t][i] = code.allocateLocal(TypeKind.REFERENCE);
        }
        int[] operands = {code.allocateLocal(TypeKind.REFERENCE),code.allocateLocal(TypeKind.REFERENCE)};
        var extracted = new java.util.HashMap<List<Integer>,Integer>();
        var retainedInputs = new java.util.HashMap<List<Integer>,Integer>();
        java.util.function.BiConsumer<Integer,int[]> scalarLoad = (v,logical) -> {
            int root = ir.viewOrigins().get(v), offset = regionFeatureOffset(ir,v,logical);
            Integer retained = retainedInputs.get(List.of(root,offset/lanes));
            if (ir.rowSites().contains(root) || retained != null) {
                var key = List.of(root,offset); Integer cell = extracted.get(key);
                if (cell == null) {
                    cell = code.allocateLocal(kind); extracted.put(key,cell);
                    code.aload(retained == null ? values[root][offset/lanes] : retained).loadConstant(offset%lanes)
                            .invokevirtual(vector,"lane",MethodTypeDesc.of(kind.upperBound(),ConstantDescs.CD_int)).storeLocal(kind,cell);
                }
                code.loadLocal(kind,cell);
            } else {
                var a = ir.accesses().get(v); int rank = a.plan().iterationRank();
                int[] coordinates = java.util.Arrays.copyOfRange(logical,logical.length-rank,logical.length);
                for (int axis = 0; axis < rank; axis++) if (a.extentConstants().get(axis) == 1 && a.extentBatchFactors().get(axis) == 0) coordinates[axis] = 0;
                regionVectorAddress(code,a,bases[boundary[root]],strides[v],row,address,coordinates);
                carrier.load(type,specialization.carrierPattern().get(boundary[root]),boundary[root],address,false);
            }
        };
        var staging = new java.util.HashMap<Integer,Long>(); long stagedCells = 0;
        var layout = specialization.rowFusionEmission().orElseThrow();
        for (var list : List.of(layout.invariantTransposeValues(),layout.temporalTransposeValues())) for (int v : list) {
            staging.put(v,stagedCells); stagedCells += regionCells(ir,v,false);
        }
        // Proven immutable W/U chunks are range invariants, not per-row loads. Empty ranges
        // initialize only references: they must not read unstaged scratch on the R=0 path.
        // A [K,1] transpose still needs its checked mask-one vector, not a scalar row reload.
        var stagedVectors = new java.util.HashMap<List<Integer>,Integer>();
        for (int v : staging.keySet()) {
            int n = Math.toIntExact(ir.accesses().get(v).extentConstants().getLast());
            if (n == 0) continue;
            for (int cell = 0; cell < regionCells(ir,v,false); cell += n)
                for (int column = 0; column < n; column += lanes) {
                    int local = code.allocateLocal(TypeKind.REFERENCE);
                    stagedVectors.put(List.of(v,cell/n,column),local);
                    code.aconst_null().astore(local);
                }
        }
        var noCoefficients = code.newLabel(); code.iload(begin).iload(finish).branch(java.lang.classfile.Opcode.IF_ICMPGE,noCoefficients);
        for (var entry : stagedVectors.entrySet()) {
            var key = entry.getKey(); int v = key.get(0), column = key.get(2);
            int n = Math.toIntExact(ir.accesses().get(v).extentConstants().getLast());
            code.loadConstant((staging.get(v)+key.get(1)*n+column)*type.byteWidth()).lstore(address);
            carrier.vectorLoadBytes(type,count,address,masks.get(Math.min(lanes,n-column))); code.astore(entry.getValue());
        }
        code.labelBinding(noCoefficients);
        var invariantVectors = new java.util.HashMap<List<Integer>,Integer>();
        java.util.function.BiConsumer<Integer,int[]> vectorLoad = (v,logical) -> {
            var invariantKey = rowInvariantVectorKey(ir,v,logical);
            if (invariantKey != null && invariantVectors.containsKey(invariantKey)) { code.aload(invariantVectors.get(invariantKey)); return; }
            var a = ir.accesses().get(v); int rank = a.plan().iterationRank(), root = ir.viewOrigins().get(v);
            if (rank == 2) {
                Integer staged = stagedVectors.get(List.of(v,logical[logical.length-2],logical[logical.length-1]));
                if (staged != null) { code.aload(staged); return; }
            }
            boolean broadcast = rank == 0 || a.extentConstants().getLast() == 1 && a.extentBatchFactors().getLast() == 0
                    || a.strideConstants().getLast() == 0 && a.strideBatchFactors().getLast() == 0;
            if (broadcast) {
                code.getstatic(vector,"SPECIES_PREFERRED",species); scalarLoad.accept(v,logical);
                code.invokestatic(vector,"broadcast",MethodTypeDesc.of(vector,species,kind.upperBound()));
            } else if (ir.rowSites().contains(root)) {
                int offset = regionFeatureOffset(ir,v,logical);
                if (offset%lanes != 0) throw new IllegalArgumentException("unaligned regional feature mapping");
                code.aload(values[root][offset/lanes]);
            } else {
                int[] coordinates = java.util.Arrays.copyOfRange(logical,logical.length-rank,logical.length);
                for (int axis = 0; axis < rank; axis++) if (a.extentConstants().get(axis) == 1 && a.extentBatchFactors().get(axis) == 0) coordinates[axis] = 0;
                int mask = masks.get(Math.min(lanes,Math.toIntExact(a.extentConstants().getLast())-coordinates[rank-1]));
                var rootAccess = ir.accesses().get(root);
                boolean retain = invariantKey == null && ir.rowAxes().get(v) == 0 && ir.rowAxes().get(root) == 0 && rank == 2
                        && rootAccess.plan().iterationRank() == 2 && a.extentConstants().getLast().equals(rootAccess.extentConstants().getLast())
                        && a.strideConstants().getLast() == 1 && a.strideBatchFactors().getLast() == 0;
                var key = List.of(root,retain ? regionFeatureOffset(ir,v,logical)/lanes : 0);
                if (retain && retainedInputs.containsKey(key)) { code.aload(retainedInputs.get(key)); return; }
                if (staging.containsKey(v)) {
                    code.loadConstant((staging.get(v)+coordinates[0]*a.extentConstants().get(1)+coordinates[1])*type.byteWidth()).lstore(address);
                    carrier.vectorLoadBytes(type,count,address,mask);
                } else {
                    regionVectorAddress(code,a,bases[boundary[root]],strides[v],row,address,coordinates);
                    carrier.vectorLoad(type,specialization.carrierPattern().get(boundary[root]),boundary[root],address,false,false,mask);
                }
                if (retain) {
                    int retained = code.allocateLocal(TypeKind.REFERENCE); retainedInputs.put(key,retained);
                    code.dup().astore(retained);
                }
            }
        };
        // Cold requests prove independence from the runtime row, not from sampled dimensions.
        // Publish each local only after its dominating load has been emitted, avoiding branch
        // cache reuse before initialization. Empty ranges leave nulls that no row can consume.
        var invariantRequests = new java.util.LinkedHashSet<List<Integer>>();
        for (var instruction : ir.arithmetic().instructions()) if (ir.rowSites().contains(instruction.output())) {
            int out = instruction.output();
            for (int block = 0; block < regionVectorBlocks(ir,out,true,lanes); block++) for (int input : instruction.inputs()) {
                var key = rowInvariantVectorKey(ir,input,regionVectorCoordinates(ir,out,block,true,lanes));
                if (key != null) invariantRequests.add(key);
            }
        }
        for (var contraction : ir.contractions()) {
            int out = contraction.output();
            boolean rowOutput = ir.rowSites().contains(out);
            int vectorBlocks = regionVectorBlocks(ir,out,rowOutput,lanes);
            for (int block = 0; block < vectorBlocks; block++) {
                int[] logical = regionVectorCoordinates(ir,out,block,rowOutput,lanes);
                int kCount = rowOutput ? Math.toIntExact(ir.accesses().get(contraction.left()).extentConstants().get(1)) : 1;
                for (int k = 0; k < kCount; k++) {
                    int[] input = {rowOutput ? k : -1,logical[logical.length-1]};
                    var key = rowInvariantVectorKey(ir,contraction.right(),input);
                    if (key != null && !staging.containsKey(contraction.right())) invariantRequests.add(key);
                }
            }
        }
        var invariantLocals = new java.util.LinkedHashMap<List<Integer>,Integer>();
        for (var key : invariantRequests) { int local = code.allocateLocal(TypeKind.REFERENCE); invariantLocals.put(key,local); code.aconst_null().astore(local); }
        var noInvariantReads = code.newLabel(); code.iload(begin).iload(finish).branch(java.lang.classfile.Opcode.IF_ICMPGE,noInvariantReads);
        for (var entry : invariantLocals.entrySet()) {
            var key = entry.getKey(); int[] logical = key.subList(1,key.size()).stream().mapToInt(Integer::intValue).toArray();
            int v = key.getFirst(), root = ir.viewOrigins().get(v); var a = ir.accesses().get(v); var source = ir.accesses().get(root);
            vectorLoad.accept(v,logical); code.astore(entry.getValue()); invariantVectors.put(key,entry.getValue());
            if (ir.rowAxes().get(v) == 0 && ir.rowAxes().get(root) == 0 && a.plan().iterationRank() == 2
                    && source.plan().iterationRank() == 2 && a.extentConstants().getLast().equals(source.extentConstants().getLast())
                    && a.strideConstants().getLast() == 1 && a.strideBatchFactors().getLast() == 0)
                retainedInputs.put(List.of(root,regionFeatureOffset(ir,v,logical)/lanes),entry.getValue());
        }
        code.labelBinding(noInvariantReads);
        var instructions = new java.util.HashMap<Integer,CpuKernelIr.Instruction>(); ir.arithmetic().instructions().forEach(i -> instructions.put(i.output(),i));
        var products = new java.util.HashMap<Integer,io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr.Contraction>();
        ir.contractions().forEach(c -> products.put(c.output(),c));
        CpuRowMapReductionEmitter.emitRangeBound(code,ir,row,begin,finish,values,sums,() -> {
            for (var terminal : sums) for (int sum : terminal) CpuMatmulEmitter.emitVectorInit(code,type,sum);
            if (phase == 1) { code.loadConstant(0L).lstore(validity); carrier.storeBytes(DataType.INT64,count,partial,validity); }
        },action -> { if (phase == 1) code.loadConstant(1L).lstore(validity); action.run(); },out -> {
            for (int block = 0; block < values[out].length; block++) {
                int result = values[out][block]; int[] logical = regionVectorCoordinates(ir,out,block,true,lanes);
                if (products.containsKey(out)) {
                    var c = products.get(out); CpuMatmulEmitter.emitVectorInit(code,type,result);
                    int k = Math.toIntExact(ir.accesses().get(c.left()).extentConstants().get(1));
                    for (int f = 0; f < k; f++) {
                        int feature = f;
                        CpuMatmulEmitter.emitVectorContributorStep(code,type,result,k == 1,() -> {
                            code.getstatic(vector,"SPECIES_PREFERRED",species); scalarLoad.accept(c.left(),new int[]{-1,feature});
                            code.invokestatic(vector,"broadcast",MethodTypeDesc.of(vector,species,kind.upperBound()));
                        },() -> vectorLoad.accept(c.right(),new int[]{feature,logical[1]}));
                    }
                } else {
                    var i = instructions.get(out); int[] handles = new int[values.length]; java.util.Arrays.fill(handles,-1); handles[out] = result;
                    for (int p = 0; p < i.inputs().size(); p++) {
                        int in = i.inputs().get(p); if (handles[in] >= 0) continue;
                        vectorLoad.accept(in,logical); code.astore(operands[p]); handles[in] = operands[p];
                    }
                    arithmetic.emit(i,handles);
                }
                if (ir.retainedStores().contains(out)) {
                    regionVectorAddress(code,ir.accesses().get(out),bases[boundary[out]],strides[out],row,address,logical);
                    carrier.vectorStore(type,specialization.carrierPattern().get(boundary[out]),boundary[out],address,result,false,
                            masks.get(Math.min(lanes,Math.toIntExact(ir.accesses().get(out).extentConstants().getLast())-logical[logical.length-1])));
                }
            }
        },out -> {
            var c = products.get(out); int t = ir.terminalOutputs().indexOf(out);
            for (int block = 0; block < sums[t].length; block++) {
                int[] logical = regionVectorCoordinates(ir,out,block,false,lanes);
                CpuMatmulEmitter.emitVectorContributorStep(code,type,sums[t][block],directProduct,() -> {
                    code.getstatic(vector,"SPECIES_PREFERRED",species); scalarLoad.accept(c.left(),new int[]{logical[0],-1});
                    code.invokestatic(vector,"broadcast",MethodTypeDesc.of(vector,species,kind.upperBound()));
                },() -> vectorLoad.accept(c.right(),new int[]{-1,logical[1]}));
            }
        },() -> {
            long offset = 8;
            for (int t = 0; t < sums.length; t++) {
                int out = ir.terminalOutputs().get(t), n = Math.toIntExact(ir.accesses().get(out).extentConstants().getLast());
                for (int block = 0; block < sums[t].length; block++) {
                    int[] logical = regionVectorCoordinates(ir,out,block,false,lanes);
                    int mask = masks.get(Math.min(lanes,n-logical[logical.length-1]));
                    if (phase == 0) {
                        regionVectorAddress(code,ir.accesses().get(out),bases[boundary[out]],strides[out],row,address,logical);
                        carrier.vectorStore(type,specialization.carrierPattern().get(boundary[out]),boundary[out],address,sums[t][block],false,mask);
                    } else {
                        int cell = logical.length == 1 ? logical[0] : logical[0]*n+logical[1];
                        code.lload(partial).loadConstant(offset+(long)cell*type.byteWidth()).ladd().lstore(address);
                        carrier.vectorStoreBytes(type,count,address,sums[t][block],mask);
                    }
                }
                offset += (long)regionCells(ir,out,false)*type.byteWidth();
            }
            if (phase == 1) carrier.storeBytes(DataType.INT64,count,partial,validity);
        });
    }

    /**
     * Combines private terminal feature vectors in fixed range order. The first valid partial
     * moves raw; invalid slots are never read. Boundary addresses retain their element/array
     * convention while private scratch uses checked byte offsets directly.
     * @param code merge helper builder owning every local
     * @param specialization authenticated ordinary selected-vector layout
     */
    private static void emitVectorRowMerge(java.lang.classfile.CodeBuilder code,CpuKernelSpecialization specialization) {
        var ir = specialization.rowMapReductionIr().orElseThrow(); var type = ir.arithmetic().values().getFirst().dataType();
        int lanes = specialization.vectorSpeciesBitSize()/type.bitWidth(), count = specialization.carrierPattern().size(), geometry = count+1;
        int start = geometry+1, end = start+2;
        var invalid = code.newLabel(); var empty = code.newLabel(); var valid = code.newLabel();
        code.lload(start).loadConstant(0L).lcmp().branch(java.lang.classfile.Opcode.IFNE,invalid)
                .lload(end).loadConstant(0L).lcmp().branch(java.lang.classfile.Opcode.IFEQ,empty)
                .lload(end).loadConstant(1L).lcmp().branch(java.lang.classfile.Opcode.IFEQ,valid)
                .labelBinding(invalid).new_(ClassDesc.of(IllegalArgumentException.class.getName())).dup()
                .invokespecial(ClassDesc.of(IllegalArgumentException.class.getName()),"<init>",MethodTypeDesc.of(ConstantDescs.CD_void)).athrow()
                .labelBinding(empty).return_().labelBinding(valid);
        var carrier = new CpuCarrierEmitter(code); carrier.prepareVectorSpecies(type); carrier.prepareVectorByteOrder();
        var vector = ClassDesc.of("jdk.incubator.vector."+(type == DataType.FLOAT64 ? "DoubleVector" : "FloatVector"));
        var species = ClassDesc.of("jdk.incubator.vector.VectorSpecies"); var maskType = ClassDesc.of("jdk.incubator.vector.VectorMask");
        int[][] sums = new int[ir.terminalOutputs().size()][]; long cells = 0;
        var masks = new java.util.HashMap<Integer,Integer>();
        for (int t = 0; t < sums.length; t++) {
            int out = ir.terminalOutputs().get(t), n = Math.toIntExact(ir.accesses().get(out).extentConstants().getLast());
            cells += regionCells(ir,out,false); sums[t] = new int[regionVectorBlocks(ir,out,false,lanes)];
            for (int b = 0; b < sums[t].length; b++) { sums[t][b] = code.allocateLocal(TypeKind.REFERENCE); CpuAggregateEmitter.emitRepresentedVectorInit(code,type,sums[t][b],false); }
            for (int column = 0; column < n; column += lanes) {
                int used = Math.min(lanes,n-column); if (masks.containsKey(used)) continue;
                int mask = code.allocateLocal(TypeKind.REFERENCE); masks.put(used,mask);
                code.getstatic(vector,"SPECIES_PREFERRED",species).loadConstant((1L<<used)-1)
                        .invokestatic(maskType,"fromLong",MethodTypeDesc.of(maskType,species,ConstantDescs.CD_long)).astore(mask);
            }
        }
        long stageCells = 0; var emission = specialization.rowFusionEmission().orElseThrow();
        for (var list : List.of(emission.invariantTransposeValues(),emission.temporalTransposeValues())) for (int v : list) stageCells += regionCells(ir,v,false);
        long stageBytes = (stageCells*type.byteWidth()+7)&~7L, slice = 8+((cells*type.byteWidth()+7)&~7L);
        int range = code.allocateLocal(TypeKind.INT), ranges = code.allocateLocal(TypeKind.INT), seen = code.allocateLocal(TypeKind.INT);
        int partial = code.allocateLocal(TypeKind.LONG), address = code.allocateLocal(TypeKind.LONG);
        code.aload(geometry).loadConstant(count+1).laload().l2i().istore(ranges).loadConstant(0).istore(range).loadConstant(0).istore(seen);
        var loop = code.newLabel(); var skip = code.newLabel(); var done = code.newLabel(); var add = code.newLabel(); var merged = code.newLabel();
        code.labelBinding(loop).iload(range).iload(ranges).branch(java.lang.classfile.Opcode.IF_ICMPGE,done)
                .iload(range).i2l().loadConstant(slice).lmul().loadConstant(stageBytes).ladd().lstore(partial);
        carrier.loadBytes(DataType.INT64,count,partial); code.loadConstant(0L).lcmp().branch(java.lang.classfile.Opcode.IFEQ,skip);
        code.iload(seen).branch(java.lang.classfile.Opcode.IFNE,add);
        for (boolean accumulate : List.of(false,true)) {
            if (accumulate) code.labelBinding(add);
            long offset = 8;
            for (int t = 0; t < sums.length; t++) {
                int out = ir.terminalOutputs().get(t), n = Math.toIntExact(ir.accesses().get(out).extentConstants().getLast());
                for (int block = 0; block < sums[t].length; block++) {
                    int[] logical = regionVectorCoordinates(ir,out,block,false,lanes);
                    int column = logical[logical.length-1], cell = block/((n+lanes-1)/lanes)*n+column, mask = masks.get(Math.min(lanes,n-column));
                    long cellOffset = offset+(long)cell*type.byteWidth();
                    CpuAggregateEmitter.emitRepresentedVectorContributorStep(code,type,sums[t][block],!accumulate,() -> {
                        code.lload(partial).loadConstant(cellOffset).ladd().lstore(address); carrier.vectorLoadBytes(type,count,address,mask);
                    });
                }
                offset += (long)regionCells(ir,out,false)*type.byteWidth();
            }
            if (!accumulate) code.branch(java.lang.classfile.Opcode.GOTO,merged);
        }
        code.labelBinding(merged).loadConstant(1).istore(seen).labelBinding(skip).iinc(range,1).branch(java.lang.classfile.Opcode.GOTO,loop).labelBinding(done);
        int[] boundaries = new int[ir.arithmetic().values().size()]; int next = 0;
        for (var value : ir.arithmetic().values()) if (value.kind() == CpuKernelIr.Value.Kind.INPUT) boundaries[value.ordinal()] = next++;
        for (var value : ir.arithmetic().values()) if (value.kind() == CpuKernelIr.Value.Kind.OUTPUT) boundaries[value.ordinal()] = next++;
        int base = code.allocateLocal(TypeKind.LONG);
        for (int t = 0; t < sums.length; t++) {
            int out = ir.terminalOutputs().get(t), boundary = boundaries[out], n = Math.toIntExact(ir.accesses().get(out).extentConstants().getLast());
            code.aload(geometry).loadConstant(boundary+1).laload().lstore(base);
            for (int block = 0; block < sums[t].length; block++) {
                int[] logical = regionVectorCoordinates(ir,out,block,false,lanes);
                int column = logical[logical.length-1];
                int[] strides = new int[logical.length]; java.util.Arrays.fill(strides,-1);
                regionVectorAddress(code,ir.accesses().get(out),base,strides,0,address,logical);
                carrier.vectorStore(type,specialization.carrierPattern().get(boundary),boundary,address,sums[t][block],false,masks.get(Math.min(lanes,n-column)));
            }
        }
    }

    /**
     * Maps a proved alias to its producing row feature, independent of the current row.
     * @param ir complete origin/access proof
     * @param v source ordinal
     * @param enclosing right-aligned consuming coordinates, minus one for R
     * @return exact flattened fixed feature-cell offset
     */
    private static int regionFeatureOffset(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr ir,int v,int[] enclosing) {
        var a = ir.accesses().get(v); long offset = 0; int shift = enclosing.length-a.plan().iterationRank();
        for (int axis = 0; axis < a.plan().iterationRank(); axis++) if (axis != ir.rowAxes().get(v)) {
            int coordinate = a.extentConstants().get(axis) == 1 && a.extentBatchFactors().get(axis) == 0 ? 0 : enclosing[shift+axis];
            offset += coordinate*a.strideConstants().get(axis);
        }
        return Math.toIntExact(offset);
    }

    /**
     * Counts vector blocks along the last fixed feature dimension, never across R.
     * @param ir proved region
     * @param v value ordinal
     * @param row whether to omit the row dimension
     * @param lanes matching preferred typed lane count
     * @return exact nonnegative feature-block count
     */
    private static int regionVectorBlocks(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr ir,int v,boolean row,int lanes) {
        var a = ir.accesses().get(v); if (a.plan().iterationRank() == 0) return 1;
        int n = Math.toIntExact(a.extentConstants().getLast());
        if (ir.rowAxes().get(v) == a.plan().iterationRank()-1 && row) return regionCells(ir,v,true);
        return n == 0 ? 0 : regionCells(ir,v,row)/n*((n+lanes-1)/lanes);
    }

    /**
     * Binds one fixed feature block to existing logical coordinates.
     * @param ir proved region
     * @param v producing ordinal
     * @param block flattened block index
     * @param row whether to insert the runtime R coordinate
     * @param lanes matching preferred typed lane count
     * @return generation-only coordinates of the block's first real lane
     */
    private static int[] regionVectorCoordinates(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr ir,int v,int block,boolean row,int lanes) {
        int n = Math.toIntExact(ir.accesses().get(v).extentConstants().getLast()), chunks = (n+lanes-1)/lanes;
        return regionCoordinates(ir,v,block/chunks*n+block%chunks*lanes,row);
    }

    /**
     * Sizes a generation-only typed feature table; rows never contribute to local count.
     * @param ir cold-proved region
     * @param v value ordinal
     * @param row whether to omit its row axis
     * @return fixed nonnegative table length
     * @throws ArithmeticException if an unproved caller supplies an overflowing product
     */
    private static int regionCells(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr ir,int v,boolean row) {
        int cells = 1; var access = ir.accesses().get(v);
        for (int axis = 0; axis < access.plan().iterationRank(); axis++) if (!row || axis != ir.rowAxes().get(v))
            cells = Math.multiplyExact(cells,Math.toIntExact(access.extentConstants().get(axis)));
        return cells;
    }

    /**
     * Expands one fixed feature cell; minus one denotes the single runtime row coordinate.
     * @param ir proved geometry
     * @param v value ordinal
     * @param cell fixed flattened feature position
     * @param row whether to insert a runtime-row marker
     * @return fresh generation-only coordinate vector, never invocation storage
     */
    private static int[] regionCoordinates(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr ir,
            int v,int cell,boolean row) {
        var access = ir.accesses().get(v); int[] coordinates = new int[access.plan().iterationRank()];
        for (int axis = coordinates.length-1; axis >= 0; axis--) {
            if (row && axis == ir.rowAxes().get(v)) { coordinates[axis] = -1; continue; }
            int extent = Math.toIntExact(access.extentConstants().get(axis));
            coordinates[axis] = cell%extent; cell /= extent;
        }
        return coordinates;
    }

    /**
     * Loads a completed producer directly, or a checked external cell. Identical authenticated
     * root/affine/epoch requests share a typed local only until their last use, with a cold-bounded
     * free pool. Every retained write ends that pool's epoch; lack of a slot retains a checked
     * read. The shared cold proof's separate read-only staged cells span the complete region
     * lifetime under its nonempty entry/range guard; actual binding rejects writable overlap.
     * @param code owning builder
     * @param specialization exact typed carrier pattern
     * @param ir proved origin/access inventory
     * @param carrier existing typed carrier emitter
     * @param locals shared per-value feature handles
     * @param boundary root-to-boundary positions, minus one only for private values
     * @param bases invocation-bound element-base handles
     * @param strides invocation-bound symbolic stride handles
     * @param row read-only INT induction local
     * @param address reusable LONG element-address local
     * @param rowOffset single recycled LONG row-product local, or minus one without capacity
     * @param rowRecipe last demanded constant/B row-stride pair in this straight-line row
     * @param remaining exact generation-only remaining requests by root/affine/store epoch
     * @param cached live typed handles for repeated external cells
     * @param free type-compatible locals available after last use or an epoch-ending write
     * @param epoch generation-only retained-store epoch, never invocation mutable state
     * @param capacity cold-proved maximum simultaneously cached cells
     * @param staged generation-only handles for the shared proof's guarded immutable cell set
     * @param v operand ordinal
     * @param enclosing consumer coordinates, right-aligned for broadcast
     */
    private static void regionLoad(java.lang.classfile.CodeBuilder code,CpuKernelSpecialization specialization,
            io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr ir,CpuCarrierEmitter carrier,
            int[][] locals,int[] boundary,int[] bases,int[][] strides,int row,int address,
            int rowOffset,long[] rowRecipe,java.util.Map<List<Long>,Integer> remaining,
            java.util.Map<List<Long>,Integer> cached,java.util.ArrayDeque<Integer> free,int[] epoch,int capacity,
            java.util.Map<List<Long>,Integer> staged,int v,int[] enclosing) {
        var access = ir.accesses().get(v); int root = ir.viewOrigins().get(v);
        int[] coordinates = java.util.Arrays.copyOfRange(enclosing,enclosing.length-access.plan().iterationRank(),enclosing.length);
        for (int axis = 0; axis < coordinates.length; axis++) if (access.extentConstants().get(axis) == 1
                && access.extentBatchFactors().get(axis) == 0) coordinates[axis] = 0;
        DataType type = ir.arithmetic().values().get(v).dataType();
        int direct = regionValueLocal(ir,locals,v,enclosing);
        if (direct >= 0) {
            code.loadLocal(type == DataType.FLOAT64 ? java.lang.classfile.TypeKind.DOUBLE : java.lang.classfile.TypeKind.FLOAT,
                    direct); return;
        }
        Integer invariant = staged.get(regionScalarKey(ir,v,enclosing));
        if (invariant != null) {
            code.loadLocal(type == DataType.FLOAT64 ? TypeKind.DOUBLE : TypeKind.FLOAT,invariant); return;
        }
        var key = new java.util.ArrayList<>(regionScalarKey(ir,v,enclosing)); key.add((long)epoch[0]);
        int after = remaining.merge(key,-1,Integer::sum);
        Integer saved = cached.get(key);
        if (saved != null) {
            code.loadLocal(type == DataType.FLOAT64 ? TypeKind.DOUBLE : TypeKind.FLOAT,saved);
            if (after == 0) { cached.remove(key); free.addLast(saved); }
            return;
        }
        regionScalarAddress(code,access,bases[boundary[root]],strides[v],row,address,rowOffset,rowRecipe,coordinates);
        carrier.load(type,specialization.carrierPattern().get(boundary[root]),boundary[root],address,false);
        if (after > 0 && cached.size() < capacity) {
            int slot = free.isEmpty() ? code.allocateLocal(type == DataType.FLOAT64 ? TypeKind.DOUBLE : TypeKind.FLOAT) : free.removeFirst();
            if (type == DataType.FLOAT64) code.dup2(); else code.dup();
            code.storeLocal(type == DataType.FLOAT64 ? TypeKind.DOUBLE : TypeKind.FLOAT,slot);
            cached.put(key,slot);
        }
    }

    /**
     * Publishes only a proved retained row site or a completed terminal boundary.
     * @param code owning builder
     * @param specialization exact typed carrier pattern
     * @param ir proved retained stores and geometry
     * @param carrier existing typed carrier emitter
     * @param boundary exact output boundary positions
     * @param bases prebound element bases
     * @param strides prebound symbolic strides
     * @param row read-only INT row local
     * @param address reusable LONG address local
     * @param v producing ordinal
     * @param coordinates exact producing coordinates
     * @param local completed represented value handle
     */
    private static void regionStore(java.lang.classfile.CodeBuilder code,CpuKernelSpecialization specialization,
            io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr ir,CpuCarrierEmitter carrier,
            int[] boundary,int[] bases,int[][] strides,int row,int address,int v,int[] coordinates,int local) {
        regionAddress(code,ir.accesses().get(v),bases[boundary[v]],strides[v],row,address,coordinates);
        carrier.store(ir.arithmetic().values().get(v).dataType(),specialization.carrierPattern().get(boundary[v]),
                boundary[v],address,local,false);
    }

    /**
     * Proves that the requested external feature address is independent of the row.
     * Only structural zero strides erase a runtime row coordinate; changing nominal
     * ROW strides never qualify merely because an initial scalar happens to broadcast.
     * @param region complete origin/access proof
     * @param value requested external ordinal
     * @param logical right-aligned first-lane coordinates; -1 denotes runtime R
     * @return immutable value/coordinate cache key, or null when a row read is required
     */
    private static List<Integer> rowInvariantVectorKey(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr region,int value,int[] logical) {
        int root = region.viewOrigins().get(value);
        if (region.rowSites().contains(root) || region.terminalOutputs().contains(root)) return null;
        var access = region.accesses().get(value); int rank = access.plan().iterationRank();
        var key = new java.util.ArrayList<Integer>(); key.add(value);
        for (int axis = 0; axis < rank; axis++) {
            int coordinate = logical[logical.length-rank+axis];
            if (access.extentConstants().get(axis) == 1 && access.extentBatchFactors().get(axis) == 0
                    || access.strideConstants().get(axis) == 0 && access.strideBatchFactors().get(axis) == 0) coordinate = 0;
            if (coordinate < 0) return null;
            key.add(coordinate);
        }
        return List.copyOf(key);
    }
    /**
     * Folds cold-proved selected-vector affine integer terms. This remains independent of
     * scalar reuse so SIMD literal controls do not change. No floating arithmetic is folded.
     * @param code owning selected builder
     * @param access exact element recipe
     * @param base dominating LONG boundary base
     * @param strides bound symbolic LONG handles, minus one for constant axes
     * @param row runtime INT row, consumed only for nonzero row strides
     * @param address reusable LONG destination
     * @param coordinates fixed features and minus-one row marker
     */
    private static void regionVectorAddress(java.lang.classfile.CodeBuilder code,
            io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.BatchBinding access,int base,
            int[] strides,int row,int address,int[] coordinates) {
        code.lload(base); long fixed = access.baseElementOffset();
        for (int axis = 0; axis < coordinates.length; axis++) {
            long sc = access.strideConstants().get(axis), sf = access.strideBatchFactors().get(axis);
            int coordinate = coordinates[axis]; if (coordinate == 0 || sc == 0 && sf == 0) continue;
            if (coordinate >= 0 && sf == 0) { fixed += coordinate*sc; continue; }
            if (coordinate < 0) code.iload(row).i2l(); else code.loadConstant((long)coordinate);
            if (sf != 0) code.lload(strides[axis]).lmul(); else if (sc != 1) code.loadConstant(sc).lmul();
            code.ladd();
        }
        if (fixed != 0) code.loadConstant(fixed).ladd(); code.lstore(address);
    }

    /**
     * Folds fixed scalar feature-address terms, retaining checked runtime row/B terms.
     * No load or floating arithmetic moves across an activity or producing-store boundary.
     * @param code owning builder
     * @param access exact element recipe
     * @param base dominating LONG boundary base
     * @param strides bound symbolic LONG handles, minus one for constant axes
     * @param row runtime INT row
     * @param address reusable LONG destination
     * @param coordinates fixed features and minus-one row marker
     */
    private static void regionAddress(java.lang.classfile.CodeBuilder code,
            io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.BatchBinding access,int base,
            int[] strides,int row,int address,int[] coordinates) {
        code.lload(base); long fixed = access.baseElementOffset();
        for (int axis = 0; axis < coordinates.length; axis++) {
            long sc = access.strideConstants().get(axis), sf = access.strideBatchFactors().get(axis);
            if (coordinates[axis] == 0 || sc == 0 && sf == 0) continue;
            if (coordinates[axis] >= 0 && sf == 0) { fixed = Math.addExact(fixed,Math.multiplyExact(coordinates[axis],sc)); continue; }
            if (coordinates[axis] == -1) code.iload(row).i2l(); else code.loadConstant((long)coordinates[axis]);
            if (sf != 0) code.lload(strides[axis]).lmul();
            else if (sc != 1) code.loadConstant(sc).lmul();
            code.ladd();
        }
        if (fixed != 0) code.loadConstant(fixed).ladd();
        code.lstore(address);
    }

    /**
     * Shares the last demanded row-stride product in one recycled LONG slot and folds fixed
     * features. It executes only at the original checked load, never above the row guard.
     * @param code owning builder
     * @param access complete affine element mapping
     * @param base immutable boundary base within this row epoch
     * @param strides bound symbolic stride handles
     * @param row original INT row
     * @param address recycled address destination
     * @param offset recycled row-product local
     * @param recipe generation-only last constant/B row coefficient pair
     * @param coordinates fixed features and minus-one row marker
     */
    private static void regionScalarAddress(java.lang.classfile.CodeBuilder code,CpuAccessPlan.BatchBinding access,
            int base,int[] strides,int row,int address,int offset,long[] recipe,int[] coordinates) {
        if (offset < 0) { regionAddress(code,access,base,strides,row,address,coordinates); return; }
        int rowAxis = -1;
        for (int a = 0; a < coordinates.length; a++) if (coordinates[a] < 0
                && (access.strideConstants().get(a) != 0 || access.strideBatchFactors().get(a) != 0)) rowAxis = a;
        if (rowAxis < 0) { regionAddress(code,access,base,strides,row,address,coordinates); return; }
        long c = access.strideConstants().get(rowAxis), f = access.strideBatchFactors().get(rowAxis);
        if (recipe[0] != c || recipe[1] != f) {
            code.iload(row).i2l();
            if (f != 0) code.lload(strides[rowAxis]).lmul(); else if (c != 1) code.loadConstant(c).lmul();
            code.lstore(offset); recipe[0] = c; recipe[1] = f;
        }
        int[] fixed = coordinates.clone(); fixed[rowAxis] = 0;
        regionAddress(code,access,base,strides,row,address,fixed);
        code.lload(address).lload(offset).ladd().lstore(address);
    }

    /**
     * Repeats the cold proof's remaining scalar budget without changing admission or resources.
     * A partial pool is legal; any request without a slot remains a checked direct read.
     * @param ir selected complete region
     * @return remaining JVM slots before the affine offset and recycled typed cache
     */
    private static int regionScalarBudget(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr ir) {
        int width = ir.arithmetic().values().getFirst().dataType() == DataType.FLOAT64 ? 2 : 1;
        long cells = 0;
        for (int v : ir.rowSites()) cells += regionCells(ir,v,true);
        for (int v : ir.terminalOutputs()) cells += regionCells(ir,v,false);
        long boundaries = ir.arithmetic().values().stream().filter(v -> v.kind() != CpuKernelIr.Value.Kind.VIRTUAL).count();
        return Math.toIntExact(Math.max(0,128-(7+3*boundaries+(cells+6)*width)));
    }

    /**
     * Directly binds an existing producer handle through proved feature/transpose views.
     * No operand copy or replacement local is emitted for an already computed row value.
     * @param ir complete proved origin/coefficient inventory
     * @param locals caller-allocated producing handles
     * @param v operand or alias ordinal
     * @param enclosing right-aligned consumer coordinates
     * @return exact typed feature handle, or minus one only for direct-storage operands
     */
    private static int regionValueLocal(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr ir,
            int[][] locals,int v,int[] enclosing) {
        int root = ir.viewOrigins().get(v);
        if (locals[root].length == 0 || locals[root][0] < 0) return -1;
        var access = ir.accesses().get(v); long offset = 0;
        var origin = ir.accesses().get(root);
        long unitC = ir.rowAxes().get(root) == 1 ? origin.strideConstants().get(0) : 1;
        long unitF = ir.rowAxes().get(root) == 1 ? origin.strideBatchFactors().get(0) : 0;
        int shift = enclosing.length-access.plan().iterationRank();
        for (int axis = 0; axis < access.plan().iterationRank(); axis++) if (axis != ir.rowAxes().get(v)) {
            int coordinate = access.extentConstants().get(axis) == 1 && access.extentBatchFactors().get(axis) == 0
                    ? 0 : enclosing[shift+axis];
            long stride = unitC != 0 ? access.strideConstants().get(axis)/unitC
                    : unitF != 0 ? access.strideBatchFactors().get(axis)/unitF : 0;
            offset += coordinate*stride;
        }
        return locals[root][Math.toIntExact(offset)];
    }

    private static boolean usesSharedCarrierLayouts(CpuKernelIr kernelIr) {
        if (!kernelIr.instructions().isEmpty()) return true;
        String family = kernelIr.familyIdentity();
        return !family.startsWith("fold:") && !family.startsWith("ordering:")
                && !family.startsWith("random:") && !family.startsWith("arg-extrema:")
                && !family.startsWith("trailing-normalization:")
                && !family.startsWith("batch-normalization-inference:")
                && !family.startsWith("batch-normalization-training:");

    }

    /**
     * Emits the clean-Java constant-store loop: load one typed literal before the output loop,
     * then store it through the existing output-address induction. No input or reduction state
     * exists. Parsing the versioned envelope happens only during generation, never in the entry.
     * @param code non-null entry builder after shared carrier-layout setup
     * @param specialization non-null scalar output-carrier specialization
     * @param ir non-null validated output-only constant envelope including exact literal bits
     * @throws IllegalArgumentException if the envelope's versioned literal is malformed
     */
    private static void emitConstantWrite(java.lang.classfile.CodeBuilder code,
            CpuKernelSpecialization specialization, CpuKernelIr ir) {
        var type = ir.values().getFirst().dataType();
        long bits = Long.parseUnsignedLong(ir.familyIdentity().substring("constant-write-v1:".length()), 16);
        var carriers = new CpuCarrierEmitter(code);
        int literal = code.allocateLocal(switch (type) {
            case FLOAT32 -> java.lang.classfile.TypeKind.FLOAT;
            case FLOAT64 -> java.lang.classfile.TypeKind.DOUBLE;
            case INT64 -> java.lang.classfile.TypeKind.LONG;
            default -> java.lang.classfile.TypeKind.INT;
        });
        switch (type) {
            case FLOAT32 -> code.loadConstant(Float.intBitsToFloat((int) bits)).fstore(literal);
            case FLOAT64 -> code.loadConstant(Double.longBitsToDouble(bits)).dstore(literal);
            case INT64 -> code.loadConstant(bits).lstore(literal);
            default -> code.loadConstant((int) bits).istore(literal);
        }
        var loops = new CpuLoopEmitter(code);
        var plans = List.of(ir.values().getFirst().accessPlan());
        java.util.function.Consumer<CpuLoopEmitter.State> body = state ->
                carriers.store(type, specialization.carrierPattern().getFirst(), 0,
                        state.addresses()[0], literal, state.intAddresses());
        if (specialization.loopAddressing(ir) == CpuKernelSpecialization.LoopAddressing.DENSE_HEAP_ARRAY_INT)
            loops.emitDenseArrayInt(plans, body);
        else loops.emit(plans, body);
    }

    /**
     * Verifies, defines, and resolves one exact generated entry.
     *
     * @param specialization non-null specialization expected by the bytes
     * @param kernelIr non-null current exact IR, authenticated before class definition
     * @param classBytes non-null complete generated class bytes; copied into the returned artifact
     * @return a new strongly owning generated-kernel artifact with the exact direct handle
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if current CPU numerical admission, IR authentication,
     *     verification, hidden-class definition, or exact handle resolution fails
     */
    public CpuGeneratedKernel defineClassBytes(CpuKernelSpecialization specialization,
            CpuKernelIr kernelIr, byte[] classBytes) {
        Objects.requireNonNull(specialization, "specialization");
        Objects.requireNonNull(classBytes, "classBytes");
        validate(specialization, kernelIr);
        verify(specialization, kernelIr, classBytes);
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

    /**
     * Generates, verifies, defines, and cold-binds both methods of one admitted partial route.
     *
     * <p>This deliberately has no {@link CpuKernelSpecialization}: a partial state body is not a
     * complete-output-cell kernel.  Its deterministic v1 identity includes every code-shaping
     * partial IR fact.  The returned handles are resolved once from the retained hidden lookup;
     * invocation therefore performs no reflective resolution or route selection.</p>
     *
     * @param ir non-null admitted partial-reduction identity
     * @return non-null artifact retaining exact partial and ordered-combine handles
     * @throws IllegalArgumentException if generated bytes cannot be verified or defined
     */
    public CpuGeneratedKernel.PartialReductionArtifact generatePartialReduction(
            CpuPartialReductionIr ir) {
        Objects.requireNonNull(ir, "ir");
        ClassDesc primitiveArray = ir.dataType() == DataType.INT32
                ? ClassDesc.ofDescriptor("[I") : ClassDesc.ofDescriptor("[J");
        MethodTypeDesc partialType = MethodTypeDesc.of(ConstantDescs.CD_void, primitiveArray,
                ConstantDescs.CD_int, ConstantDescs.CD_int,
                ClassDesc.of("java.lang.foreign.MemorySegment"), ConstantDescs.CD_long);
        MethodTypeDesc combineType = MethodTypeDesc.of(ConstantDescs.CD_void,
                ClassDesc.of("java.lang.foreign.MemorySegment"), ConstantDescs.CD_int, ConstantDescs.CD_int,
                primitiveArray, ConstantDescs.CD_int);
        ClassDesc owner = ClassDesc.of(partialReductionBinaryName(ir));
        byte[] bytes = ClassFile.of().build(owner, classBuilder -> {
            classBuilder.withVersion(ClassFile.JAVA_26_VERSION, 0).withFlags(AccessFlag.FINAL);
            var emitter = new CpuPartialReductionEmitter();
            classBuilder.withMethod(PARTIAL_BODY_NAME, partialType, AccessFlag.STATIC.mask(),
                    method -> method.withCode(code -> emitter.emitPartial(code, ir)));
            classBuilder.withMethod(ORDERED_COMBINE_NAME, combineType, AccessFlag.STATIC.mask(),
                    method -> method.withCode(code -> emitter.emitCombine(code, ir)));
        });
        verifyPartialReduction(ir, bytes);
        try {
            MethodHandles.Lookup lookup = MethodHandles.lookup().defineHiddenClass(bytes, false,
                    MethodHandles.Lookup.ClassOption.NESTMATE);
            Class<?> primitive = ir.dataType() == DataType.INT32 ? int[].class : long[].class;
            return new CpuGeneratedKernel.PartialReductionArtifact(ir, lookup,
                    lookup.findStatic(lookup.lookupClass(), PARTIAL_BODY_NAME,
                            java.lang.invoke.MethodType.methodType(void.class, primitive,
                                    int.class, int.class, java.lang.foreign.MemorySegment.class,
                                    long.class)),
                    lookup.findStatic(lookup.lookupClass(), ORDERED_COMBINE_NAME,
                            java.lang.invoke.MethodType.methodType(void.class,
                                    java.lang.foreign.MemorySegment.class,
                                    int.class, int.class, primitive, int.class)), bytes);
        } catch (ReflectiveOperationException | IllegalArgumentException failure) {
            throw new IllegalArgumentException("partial-reduction generated class definition failed",
                    failure);
        }
    }

    private static String partialReductionBinaryName(CpuPartialReductionIr ir) {
        return "io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.GeneratedPartial_"
                + PARTIAL_REDUCTION_SCHEMA + '_' + ir.dataType() + '_' + ir.kind() + '_'
                + ir.form() + '_' + ir.outputCount() + '_' + ir.domainCount() + '_'
                + ir.partialCount();
    }

    private static void verifyPartialReduction(CpuPartialReductionIr ir, byte[] bytes) {
        try {
            var errors = ClassFile.of().verify(bytes);
            if (!errors.isEmpty()) throw new IllegalArgumentException("partial generated class verification failed: " + errors);
            var model = ClassFile.of().parse(bytes);
            String primitiveArray = ir.dataType() == DataType.INT32 ? "[I" : "[J";
            String partialDescriptor = '(' + primitiveArray + "IILjava/lang/foreign/MemorySegment;J)V";
            String combineDescriptor = "(Ljava/lang/foreign/MemorySegment;II" + primitiveArray + "I)V";
            boolean methods = model.methods().size() == 2 && model.methods().stream().anyMatch(method ->
                    method.methodName().stringValue().equals(PARTIAL_BODY_NAME)
                            && method.methodTypeSymbol().descriptorString().equals(partialDescriptor))
                    && model.methods().stream().anyMatch(method -> method.methodName().stringValue()
                            .equals(ORDERED_COMBINE_NAME) && method.methodTypeSymbol()
                                    .descriptorString().equals(combineDescriptor));
            if (model.majorVersion() != ClassFile.JAVA_26_VERSION || !model.fields().isEmpty()
                    || !model.interfaces().isEmpty() || !methods || !model.thisClass().asInternalName()
                            .equals(partialReductionBinaryName(ir).replace('.', '/'))) {
                throw new IllegalArgumentException("partial generated class shape does not match identity");
            }
        } catch (IllegalArgumentException failure) { throw failure; }
        catch (RuntimeException failure) {
            throw new IllegalArgumentException("partial generated class verification failed", failure);
        }
    }

    private static int[] allocateScalarLocals(java.lang.classfile.CodeBuilder code, CpuKernelIr ir) {
        int[] result = new int[ir.values().size()];
        for (CpuKernelIr.Value value : ir.values()) result[value.ordinal()] = code.allocateLocal(
                switch (value.dataType()) {
                    case FLOAT64 -> TypeKind.DOUBLE; case FLOAT32 -> TypeKind.FLOAT;
                    case BFLOAT16, FLOAT16, FLOAT8_E4M3, FLOAT8_E5M2, INT32, BOOL -> TypeKind.INT;
                    case INT64 -> TypeKind.LONG;
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
        emitBoundPointwise(ir.instructions(), () -> {
        for (int boundary = 0; boundary < boundaries.size(); boundary++) {
            CpuKernelIr.Value value = boundaries.get(boundary);
            if (value.kind() != CpuKernelIr.Value.Kind.INPUT
                    || !requiresInputLoad(ir, value.ordinal())) continue;
            if (value.dataType() == DataType.BOOL && vectorType != DataType.BOOL) {
                if (value.accessPlan().regime() == CpuAccessPlan.Regime.DENSE_LINEAR) {
                    carriers.denseBoolMaskLoad(vectorType, specialization.carrierPattern().get(boundary),
                            boundary, state.addresses()[boundary], state.intAddresses());
                } else carriers.scalarBoolMaskLoad(vectorType,
                        specialization.carrierPattern().get(boundary), boundary,
                        state.addresses()[boundary], state.intAddresses());
            } else carriers.vectorLoad(vectorType,
                    specialization.carrierPattern().get(boundary), boundary,
                    state.addresses()[boundary], value.accessPlan().regime()
                            == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Regime.SCALAR_ALL_ZERO,
                    state.intAddresses());
            code.astore(locals[value.ordinal()]);
        }
        }, instruction -> {}, instruction -> instructions.emit(instruction, locals), instruction -> {}, () -> {
        for (CpuKernelIr.Store store : ir.stores()) {
            CpuKernelIr.Value value = ir.values().get(store.value());
            int boundary = boundaryIndex(boundaries, value.ordinal());
            if (value.dataType() == DataType.BOOL && vectorType != DataType.BOOL) {
                carriers.denseBoolMaskStore(vectorType, specialization.carrierPattern().get(boundary),
                        boundary, state.addresses()[boundary], locals[value.ordinal()], state.intAddresses());
            } else carriers.vectorStore(vectorType, specialization.carrierPattern().get(boundary), boundary,
                    state.addresses()[boundary], locals[value.ordinal()], state.intAddresses());
        }
        });
    }

    private static boolean usesDenseFloatingMaskBoundary(CpuKernelIr ir, DataType vectorType) {
        return (vectorType == DataType.FLOAT32 || vectorType == DataType.FLOAT64)
                && ir.values().stream().anyMatch(value -> value.dataType() == DataType.BOOL
                    && value.kind() != CpuKernelIr.Value.Kind.VIRTUAL
                    && value.accessPlan().regime() == CpuAccessPlan.Regime.DENSE_LINEAR);
    }

    private static boolean hasProvedVectorScalarPower(CpuKernelSpecialization specialization,
            CpuKernelIr ir) {
        if (specialization.executionStrategy().compute()
                != io.github.pho001.synaptik.backend.cpu.internal.prepare
                    .CpuPartitionPreparationPlan.ExecutionStrategy.Compute.VECTOR) return false;
        DataType vectorType = vectorDataTypeOrNull(ir);
        return (vectorType == DataType.FLOAT32 || vectorType == DataType.FLOAT64)
                && ir.instructions().stream().anyMatch(instruction ->
                instruction.opcode() == CpuPointwiseOpcode.SCALAR_POW
                && instruction.powerRealization() != CpuKernelIr.PowerRealization.DIRECT);
    }

    private static boolean usesOnlyArrayVectorCarriers(CpuKernelIr ir,
            List<CpuKernelIr.Value> boundaries,
            List<CpuKernelSpecialization.CarrierAccess> carriers) {
        for (int boundary = 0; boundary < boundaries.size(); boundary++) {
            CpuKernelIr.Value value = boundaries.get(boundary);
            boolean accessed = value.kind() == CpuKernelIr.Value.Kind.INPUT
                    ? requiresInputLoad(ir, value.ordinal())
                    : ir.stores().stream().anyMatch(store -> store.value() == value.ordinal());
            if (accessed && carriers.get(boundary)
                    == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT) return false;
        }
        return true;
    }

    private static boolean requiresVectorByteOrder(CpuKernelIr ir,
            List<CpuKernelIr.Value> boundaries,
            List<CpuKernelSpecialization.CarrierAccess> carriers) {
        for (int boundary = 0; boundary < boundaries.size(); boundary++) {
            if (carriers.get(boundary)
                    != CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT) continue;
            CpuKernelIr.Value value = boundaries.get(boundary);
            if (value.dataType() == DataType.BOOL) continue;
            if (value.kind() == CpuKernelIr.Value.Kind.INPUT) {
                if (requiresInputLoad(ir, value.ordinal())
                        && value.accessPlan().regime() != CpuAccessPlan.Regime.SCALAR_ALL_ZERO) {
                    return true;
                }
            } else if (ir.stores().stream().anyMatch(
                    store -> store.value() == value.ordinal())) return true;
        }
        return false;
    }

    private static boolean allAccessedBoundariesAreDense(List<CpuAccessPlan> plans,
            boolean[] accessed) {
        for (int boundary = 0; boundary < plans.size(); boundary++) {
            if (!accessed[boundary]) continue;
            CpuAccessPlan.Regime regime = plans.get(boundary).regime();
            if (regime != CpuAccessPlan.Regime.DENSE_LINEAR
                    && regime != CpuAccessPlan.Regime.SCALAR_ALL_ZERO) return false;
        }
        return true;
    }

    private static boolean[] accessedBoundaries(CpuKernelIr ir,
            List<CpuKernelIr.Value> boundaries) {
        boolean[] result = new boolean[boundaries.size()];
        for (int boundary = 0; boundary < boundaries.size(); boundary++) {
            CpuKernelIr.Value value = boundaries.get(boundary);
            result[boundary] = value.kind() == CpuKernelIr.Value.Kind.INPUT
                    ? requiresInputLoad(ir, value.ordinal())
                    : ir.stores().stream().anyMatch(store -> store.value() == value.ordinal());
        }
        return result;
    }

    private static List<CpuKernelSpecialization.CarrierAccess> accessedCarrierPattern(
            CpuKernelIr ir, CpuKernelSpecialization specialization) {
        List<CpuKernelIr.Value> boundaries = ir.values().stream()
                .filter(value -> value.kind() != CpuKernelIr.Value.Kind.VIRTUAL).toList();
        var result = new java.util.ArrayList<>(specialization.carrierPattern());
        for (int boundary = 0; boundary < boundaries.size(); boundary++) {
            CpuKernelIr.Value value = boundaries.get(boundary);
            boolean accessed = value.kind() == CpuKernelIr.Value.Kind.INPUT
                    ? requiresInputLoad(ir, value.ordinal())
                    : ir.stores().stream().anyMatch(store -> store.value() == value.ordinal());
            if (!accessed && result.get(boundary)
                    == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT) {
                result.set(boundary, switch (value.dataType()) {
                    case FLOAT32 -> CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY;
                    case FLOAT64 -> CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY;
                    default -> throw new IllegalArgumentException(
                            "proved scalar power requires a floating boundary");
                });
            }
        }
        return List.copyOf(result);
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
            case BFLOAT16, FLOAT16, FLOAT8_E4M3, FLOAT8_E5M2, INT32, BOOL -> code.istore(local);
            case INT64 -> code.lstore(local);
            default -> throw new IllegalArgumentException("unsupported generated type");
        }
    }

    /**
     * Checks that canonical generated meaning and the selected entry specialization agree.
     * Constant-write envelopes require one output, scalar computation, no scratch and
     * a parseable baked literal; this validates code shape, not graph constant provenance.
     * @param specialization non-null immutable typed carrier, strategy and compatibility facts
     * @param kernelIr non-null canonical generated envelope, not mutated
     * @throws NullPointerException if either argument is {@code null}
     * @throws IllegalArgumentException if identities, types, entry shape or family facts disagree
     */
    private static void validate(CpuKernelSpecialization specialization, CpuKernelIr kernelIr) {
        Objects.requireNonNull(specialization, "specialization");
        Objects.requireNonNull(kernelIr, "kernelIr");
        if (!specialization.loweringFingerprint().hex().equals(kernelIr.structuralKey())) {
            throw new IllegalArgumentException("kernel IR does not match specialization");
        }
        if (specialization.matmulIr().isPresent()
                && !specialization.matmulIr().orElseThrow().structuralKey().equals(kernelIr.structuralKey()))
            throw new IllegalArgumentException("matrix envelope differs from typed facts");
        boolean lowSchema = CpuKernelSpecialization.requiresLowPrecisionArithmeticSchema(kernelIr)
                && !specialization.movementOperandTable();
        if (lowSchema != (specialization.classIdentitySchema()
                == CpuKernelSpecialization.LOW_PRECISION_ARITHMETIC_SCHEMA))
            throw new IllegalArgumentException("low numerical compatibility projection disagrees");
        boolean exactMseCount = !lowSchema && kernelIr.familyIdentity().startsWith("loss:kind=MEAN_SQUARED_ERROR:")
                && kernelIr.familyIdentity().contains(":result=FLOAT32:reduction=MEAN:");
        if (exactMseCount != (specialization.classIdentitySchema()
                == CpuKernelSpecialization.FLOAT32_MSE_EXACT_COUNT_SCHEMA))
            throw new IllegalArgumentException("MSE exact-count compatibility projection disagrees");
        if (specialization.softmaxBackwardIr().isPresent()) {
            if (!kernelIr.equals(specialization.softmaxBackwardIr().orElseThrow().encodedKernelIr()))
                throw new IllegalArgumentException("softmax backward envelope differs from typed facts");
            return;
        }
        if (kernelIr.familyIdentity().startsWith("ordinary-softmax-backward-v1:"))
            throw new IllegalArgumentException("softmax backward requires typed facts");
        if (specialization.aggregateFusionIr().isPresent()) {
            if (!kernelIr.equals(specialization.aggregateFusionIr().orElseThrow().encodedKernelIr()))
                throw new IllegalArgumentException("ordinary aggregate fusion envelope differs from typed facts");
            return;
        }
        if (kernelIr.familyIdentity().startsWith("ordinary-aggregate-fusion-v1:"))
            throw new IllegalArgumentException("ordinary aggregate fusion requires typed facts");
        if (specialization.generalScanIr().isPresent()) {
            if (!kernelIr.equals(specialization.generalScanIr().orElseThrow().encodedKernelIr()))
                throw new IllegalArgumentException("scan envelope must match typed scan facts");
            return;
        }
        if (specialization.rowMapReductionIr().isPresent()) {
            if (!kernelIr.equals(specialization.rowMapReductionIr().orElseThrow().encodedKernelIr()))
                throw new IllegalArgumentException("row-region envelope must match typed region facts");
            return;
        }
        List<DataType> boundaryTypes = kernelIr.values().stream()
                .filter(value -> value.kind() != CpuKernelIr.Value.Kind.VIRTUAL)
                .map(CpuKernelIr.Value::dataType).toList();
        boolean typedBatch = kernelIr.familyIdentity().endsWith(":ordinary-batch-typed-v1");
        if (specialization.ordinaryBatch() != (kernelIr.familyIdentity().endsWith(":ordinary-batch-v1") || typedBatch)
                || specialization.typedOrdinaryBatch() != typedBatch) {
            throw new IllegalArgumentException("ordinary batch artifact meaning disagrees");
        }
        if (typedBatch) validateTypedOrdinaryBatch(kernelIr);
        if (specialization.movementOperandTable() != CpuDataMovementEmitter.usesOperandTable(kernelIr))
            throw new IllegalArgumentException("movement operand-table projection disagrees");
        boolean pointwise = kernelIr.familyIdentity().equals("pointwise")
                || specialization.ordinaryBatch() && (kernelIr.familyIdentity().equals("pointwise:ordinary-batch-v1")
                    || typedBatch && kernelIr.familyIdentity().startsWith("pointwise:mapping="));
        boolean denseArrayAffine = kernelIr.familyIdentity().startsWith("affine:")
                && specialization.loopAddressing(kernelIr)
                    == CpuKernelSpecialization.LoopAddressing.DENSE_HEAP_ARRAY_INT;
        if (specialization.classIdentitySchema() != CpuKernelSpecialization.LOW_PRECISION_ARITHMETIC_SCHEMA && denseArrayAffine != (specialization.classIdentitySchema()
                == CpuKernelSpecialization.DENSE_ARRAY_AFFINE_EMPTY_RANGE_SCHEMA)) {
            throw new IllegalArgumentException("affine empty-range compatibility projection disagrees");
        }
        boolean attentionFamily = kernelIr.familyIdentity().startsWith("attention:");
        if (specialization.classIdentitySchema() != CpuKernelSpecialization.LOW_PRECISION_ARITHMETIC_SCHEMA && attentionFamily != (specialization.classIdentitySchema()
                == CpuKernelSpecialization.ATTENTION_LITERAL_EXP_SCHEMA)) {
            throw new IllegalArgumentException("literal attention compatibility projection disagrees");
        }
        boolean bfloatPointwise = pointwise && kernelIr.values().stream().anyMatch(value ->
                value.dataType() == io.github.pho001.synaptik.model.datatype.DataType.BFLOAT16);
        boolean float16Pointwise = pointwise && kernelIr.values().stream().anyMatch(value ->
                value.dataType() == io.github.pho001.synaptik.model.datatype.DataType.FLOAT16
                        || value.dataType() == DataType.FLOAT8_E4M3 || value.dataType() == DataType.FLOAT8_E5M2);
        boolean crossTypeCast = pointwise && kernelIr.instructions().stream().anyMatch(instruction ->
                instruction.opcode() == CpuPointwiseOpcode.CAST
                        && kernelIr.values().get(instruction.inputs().getFirst()).dataType()
                                != kernelIr.values().get(instruction.output()).dataType());
        int expectedPointwiseSchema = !pointwise ? 0
                : lowSchema ? CpuKernelSpecialization.LOW_PRECISION_ARITHMETIC_SCHEMA
                : CpuKernelSpecialization.hasFloat32RsqrtSites(kernelIr)
                    ? CpuKernelSpecialization.FLOAT32_RSQRT_SITES_SCHEMA
                : float16Pointwise ? 68 : crossTypeCast ? 60 : bfloatPointwise ? 59
                : pointwiseSchema(specialization, kernelIr);
        int expectedStores = kernelIr.familyIdentity()
                .startsWith("batch-normalization-training:") ? 5
                : kernelIr.familyIdentity().contains(":outputs=2:") ? 2 : 1;
        if (!boundaryTypes.equals(specialization.boundaryDataTypes())
                || kernelIr.instructions().size() > 8
                || pointwise && specialization.classIdentitySchema() != expectedPointwiseSchema
                || (pointwise ? kernelIr.stores().isEmpty()
                    : kernelIr.stores().size() != expectedStores)) {
            throw new IllegalArgumentException("unsupported canonical pointwise IR");
        }
        if (kernelIr.instructions().isEmpty()) {
            if (kernelIr.familyIdentity().startsWith("constant-write-v1:")) {
                if (kernelIr.values().size() != 1 || kernelIr.values().getFirst().kind() != CpuKernelIr.Value.Kind.OUTPUT
                        || kernelIr.values().getFirst().accessPlan().accessKind()
                            != io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.AccessKind.WRITE
                        || specialization.scratchParameter()
                        || specialization.executionStrategy().compute()
                            != io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy.Compute.SCALAR)
                    throw new IllegalArgumentException("invalid constant-write envelope");
                Long.parseUnsignedLong(kernelIr.familyIdentity().substring("constant-write-v1:".length()), 16);
                return;
            }
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
            boolean batchNormalization = kernelIr.familyIdentity()
                    .startsWith("batch-normalization-inference:");
            boolean batchTraining = kernelIr.familyIdentity()
                    .startsWith("batch-normalization-training:");
            boolean conv2d = kernelIr.familyIdentity().startsWith("conv2d:");
            boolean conv3d = kernelIr.familyIdentity().startsWith("conv3d:");
            boolean matmul = kernelIr.familyIdentity().startsWith("matmul:");
            boolean pool2d = kernelIr.familyIdentity().startsWith("pool2d:");
            boolean pool3d = kernelIr.familyIdentity().startsWith("pool3d:");
            boolean attention = kernelIr.familyIdentity().startsWith("attention:");
            boolean loss = kernelIr.familyIdentity().startsWith("loss:");
            boolean scatter = kernelIr.familyIdentity().startsWith("scatter:");
            boolean fold = kernelIr.familyIdentity().startsWith("fold:");
            boolean ordering = kernelIr.familyIdentity().startsWith("ordering:");
            boolean movement = kernelIr.familyIdentity().startsWith("movement:");
            boolean affine = kernelIr.familyIdentity().startsWith("affine:");
            if ((!movement && !affine && !indexing && !scatter && !fold && !ordering && !random && !scan && !aggregate && !argExtrema && !maskedReduction && !advancedReduction && !softmax && !trailingNormalization && !batchNormalization && !batchTraining && !conv2d && !conv3d && !matmul && !pool2d && !pool3d && !attention && !loss)
                    || affine && kernelIr.values().size() != 2
                    || movement && (kernelIr.values().size() < 2
                        || !specialization.movementOperandTable() && kernelIr.values().size() + 6 > 254)
                    || !ordering && !random && !scan && !aggregate && !argExtrema && !maskedReduction && !advancedReduction && !softmax && !trailingNormalization && !batchNormalization && !batchTraining && !conv2d && !conv3d && !matmul && !pool2d && !pool3d && !attention && kernelIr.values().subList(0, kernelIr.values().size() - 1).stream()
                        .anyMatch(value -> value.kind() != CpuKernelIr.Value.Kind.INPUT)
                    || kernelIr.values().getLast().kind() != CpuKernelIr.Value.Kind.OUTPUT
                    || !indexing && !scatter && !fold && !ordering && !random && !scan && !aggregate && !argExtrema && !maskedReduction && !advancedReduction && !softmax && !trailingNormalization && !batchNormalization && !batchTraining && !conv2d && !conv3d && !matmul && !pool2d && !pool3d && !attention && !loss
                        && kernelIr.values().stream().map(CpuKernelIr.Value::dataType).distinct().count() != 1
                    || conv2d && (kernelIr.values().size() < 3 || kernelIr.values().size() > 5)
                    || conv3d && (kernelIr.values().size() < 3 || kernelIr.values().size() > 4)
                    || pool2d && kernelIr.values().size() != 2
                    || pool3d && kernelIr.values().size() != 2
                    || attention && !specialization.scratchParameter()
                    || matmul && (specialization.matmulIr().isEmpty()
                        || kernelIr.values().size()
                            != (specialization.matmulIr().orElseThrow().epilogue().hasBias()?4:3)
                        || specialization.matmulIr().orElseThrow().realization()
                            != io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr.Realization.DIRECT_SCALAR
                        && specialization.matmulIr().orElseThrow().realization()
                            != io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr.Realization.DIRECT_N_VECTOR
                        && specialization.matmulIr().orElseThrow().realization()
                            != io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr.Realization.TILED_SCALAR_2X2
                        && specialization.matmulIr().orElseThrow().realization()
                            != io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr.Realization.TILED_N_VECTOR_2X2)
                    || random && kernelIr.values().size() != 1 && kernelIr.values().size() != 5
                    || !matmul && !loss && specialization.executionStrategy().compute()
                        != io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy.Compute.SCALAR
                        && !(specialization.classIdentitySchema() == 63 && (conv2d || conv3d))
                    || !specialization.scalarPowerRealizations().isEmpty()) {
                throw new IllegalArgumentException("unsupported canonical affine copy IR");
            }
            return;
        }
        if (!pointwise
                || (kernelIr.values().stream().anyMatch(value ->
                        value.dataType() == io.github.pho001.synaptik.model.datatype.DataType.BFLOAT16
                        || value.dataType()
                            == io.github.pho001.synaptik.model.datatype.DataType.FLOAT16
                        || value.dataType() == DataType.FLOAT8_E4M3 || value.dataType() == DataType.FLOAT8_E5M2)
                    || crossTypeCast)
                    && specialization.executionStrategy().compute()
                        != io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy.Compute.SCALAR) {
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
        DataType vectorType = vectorDataTypeOrNull(kernelIr);
        if (specialization.executionStrategy().compute()
                    == io.github.pho001.synaptik.backend.cpu.internal.prepare
                        .CpuPartitionPreparationPlan.ExecutionStrategy.Compute.VECTOR
                && usesDenseFloatingMaskBoundary(kernelIr, vectorType)
                && specialization.vectorSpeciesBitSize() / vectorType.bitWidth() > Long.SIZE) {
            throw new IllegalArgumentException(
                    "floating mask bridge requires a preferred species of at most 64 lanes");
        }
    }

    /**
     * Independently checks the additive typed namespace at validated generation. The namespace
     * does not license arbitrary F64 algebra, mixed arithmetic, integral operations or MATMUL.
     * Numeric mapping coefficients are authenticated by the canonical fingerprint; actual B,
     * bytes and storage remain invocation facts.
     * @param ir non-null typed ordinary envelope with a matching specialization
     * @throws IllegalArgumentException if types, access, exact-state policy or scalar closure disagree
     */
    private static void validateTypedOrdinaryBatch(CpuKernelIr ir) {
        var values = ir.values(); var type = values.getFirst().dataType();
        if (type != DataType.FLOAT32 && type != DataType.FLOAT64 || values.size() < 2 || values.size() > 3
                || values.stream().anyMatch(v -> v.accessPlan().iterationRank() > 3)
                || values.subList(0,values.size()-1).stream().anyMatch(v -> v.dataType() != type || v.kind() != CpuKernelIr.Value.Kind.INPUT)
                || values.getLast().kind() != CpuKernelIr.Value.Kind.OUTPUT)
            throw new IllegalArgumentException("unsupported typed ordinary boundaries");
        boolean control = values.getLast().dataType() == DataType.INT64;
        if (control) {
            if (values.size() != 2 || values.stream().anyMatch(v -> v.accessPlan().iterationRank() != 1)
                    || ir.instructions().size() != 1 || ir.instructions().getFirst().opcode() != CpuPointwiseOpcode.CAST)
                throw new IllegalArgumentException("typed control is only floating [B] to INT64 [B] CAST");
        } else if (values.getLast().dataType() != type) throw new IllegalArgumentException("typed ordinary arithmetic is homogeneous");
        if (ir.familyIdentity().startsWith("aggregate:")) {
            int limbs = type == DataType.FLOAT64 ? 34 : 5, slice = type == DataType.FLOAT64 ? 280 : 48;
            if (control || !ir.instructions().isEmpty() || values.size() != 2
                    || !(ir.familyIdentity().startsWith("aggregate:SUM:") || ir.familyIdentity().startsWith("aggregate:MEAN:"))
                    || !ir.familyIdentity().contains(":limbs="+limbs+":slice="+slice+":floating=1:range=1:workspace=1:"))
                throw new IllegalArgumentException("typed aggregate exact-state contract disagrees");
        } else {
            if (!ir.familyIdentity().startsWith("pointwise:mapping=") || ir.instructions().size() != 1)
                throw new IllegalArgumentException("unsupported typed ordinary family");
            var opcode = ir.instructions().getFirst().opcode();
            if (opcode != CpuPointwiseOpcode.CAST && opcode != CpuPointwiseOpcode.ADD && opcode != CpuPointwiseOpcode.MUL
                    && opcode != CpuPointwiseOpcode.DIV && opcode != CpuPointwiseOpcode.SCALAR_ADD
                    && opcode != CpuPointwiseOpcode.SCALAR_MUL && opcode != CpuPointwiseOpcode.SCALAR_DIV)
                throw new IllegalArgumentException("unsupported typed ordinary scalar closure");
        }
    }

    private static int pointwiseSchema(CpuKernelSpecialization specialization,
            CpuKernelIr kernelIr) {
        return specialization.executionStrategy().compute()
                    == io.github.pho001.synaptik.backend.cpu.internal.prepare
                        .CpuPartitionPreparationPlan.ExecutionStrategy.Compute.VECTOR
                && usesDenseFloatingMaskBoundary(kernelIr, vectorDataTypeOrNull(kernelIr))
                ? 61 : 52;
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
        if (ir.instructions().stream().anyMatch(instruction ->
                instruction.opcode() == CpuPointwiseOpcode.CAST
                        && ir.values().get(instruction.inputs().getFirst()).dataType()
                                != ir.values().get(instruction.output()).dataType())) return false;
        boolean mixedMasks = (laneType == DataType.FLOAT32 || laneType == DataType.FLOAT64)
                && ir.values().stream().anyMatch(value -> value.dataType() == DataType.BOOL);
        for (CpuKernelIr.Value value : ir.values()) {
            if (value.dataType() == DataType.BOOL && mixedMasks
                    && value.kind() != CpuKernelIr.Value.Kind.VIRTUAL
                    && !(value.kind() == CpuKernelIr.Value.Kind.INPUT
                        && value.accessPlan().regime()
                            == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Regime.SCALAR_ALL_ZERO)) {
                if (value.accessPlan().regime() != CpuAccessPlan.Regime.DENSE_LINEAR) return false;
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
                    if (!mixedMasks || !isProducedFloatingMask(
                            ir.values().get(instruction.output()))) return false;
                }
                case VALUE_OR_MASK -> {
                    boolean byteValues = laneType == DataType.BOOL;
                    boolean floatingMasks = mixedMasks
                            && instruction.inputs().stream().allMatch(input ->
                                isProducedFloatingMask(ir.values().get(input)))
                            && isProducedFloatingMask(ir.values().get(instruction.output()));
                    if (!byteValues && !floatingMasks) return false;
                }
                case MASK_CONSUMER -> {
                    if (!mixedMasks) return false;
                    CpuKernelIr.Value condition = ir.values().get(instruction.inputs().getFirst());
                    if (!isProducedFloatingMask(condition)
                            && !(condition.kind() == CpuKernelIr.Value.Kind.INPUT
                                && condition.accessPlan().regime()
                                    == CpuAccessPlan.Regime.SCALAR_ALL_ZERO)
                            && !(condition.kind() == CpuKernelIr.Value.Kind.INPUT
                                && condition.accessPlan().regime()
                                    == CpuAccessPlan.Regime.DENSE_LINEAR)) {
                        return false;
                    }
                }
                case NONE -> { return false; }
            }
        }
        return true;
    }

    private static boolean isProducedFloatingMask(CpuKernelIr.Value value) {
        return value.dataType() == DataType.BOOL
                && (value.kind() == CpuKernelIr.Value.Kind.VIRTUAL
                    || value.kind() == CpuKernelIr.Value.Kind.OUTPUT
                        && value.accessPlan().regime() == CpuAccessPlan.Regime.DENSE_LINEAR);
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

    /**
     * Authenticates exact declarations and JVM structural limits for fresh or cached bytes.
     * @param bytes complete generated class bytes
     * @param inventory complete emission-owner inventory, including composite subsidiary methods
     * @throws IllegalArgumentException if verification, limits or declarations disagree
     */
    static void verifyMethodInventory(byte[] bytes, List<GeneratedMethod> inventory) {
        var errors = ClassFile.of().verify(bytes);
        if (!errors.isEmpty()) throw new IllegalArgumentException("generated class verification failed: " + errors);
        var model = ClassFile.of().parse(bytes);
        if (model.methods().size() != inventory.size())
            throw new IllegalArgumentException("generated method inventory size differs");
        for (int index = 0; index < inventory.size(); index++) {
            var expected = inventory.get(index);
            var actual = model.methods().get(index);
            int slots = 0;
            for (ClassDesc parameter : expected.type().parameterList()) {
                String descriptor = parameter.descriptorString();
                slots += descriptor.equals("J") || descriptor.equals("D") ? 2 : 1;
            }
            if (slots > 255 || !actual.methodName().equalsString(expected.name())
                    || !actual.methodType().equalsString(expected.type().descriptorString())
                    || actual.flags().flagsMask() != expected.accessFlags()
                    || actual.code().isEmpty())
                throw new IllegalArgumentException("generated method declaration differs: " + expected.name());
        }
    }

    private static void verify(CpuKernelSpecialization specialization, CpuKernelIr kernelIr, byte[] bytes) {
        try {
            verifyMethodInventory(bytes, methodInventory(
                    ClassDesc.of(CpuGeneratorSchema.generatedBinaryName(specialization)),
                    "", specialization, kernelIr, false));
            var model = ClassFile.of().parse(bytes);
            String expectedName = CpuGeneratorSchema.generatedBinaryName(specialization).replace('.', '/');
            boolean regional = specialization.rowMapReductionIr().isPresent()
                    || specialization.generalScanIr().flatMap(ir -> ir.stages())
                            .flatMap(stages -> stages.rowMapReductionIr()).isPresent();
            if (regional && model.methods().stream().flatMap(m -> m.code().stream())
                    .map(java.lang.classfile.attribute.CodeAttribute.class::cast)
                    .anyMatch(code -> code.codeLength() > 48000))
                throw new IllegalArgumentException("cold row-region method-size proof was insufficient");
            if (specialization.rowFusionEmission().isPresent()) {
                // Declaration shape is authenticated by the emission owner's inventory above.
                if (specialization.executionStrategy().compute()
                        == io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy.Compute.VECTOR) {
                    DataType laneType = specialization.generalScanIr().map(s -> s.dataType())
                            .orElse(specialization.boundaryDataTypes().getFirst());
                    String vector = "jdk/incubator/vector/"+(laneType == DataType.FLOAT64
                            ? "DoubleVector" : "FloatVector");
                    // Structural stale/scalar-artifact rejection only, never an execution proof.
                    // NEG + a raw geometry-singleton represented SUM has no MUL/ADD site.
                    var carrierOnly = java.util.Set.of("zero","broadcast","fromArray","fromMemorySegment",
                            "intoArray","intoMemorySegment","lane","withLane","species","length");
                    boolean vectorMath = model.methods().stream().flatMap(m -> m.code().stream())
                            .flatMap(c -> c.elementStream()).anyMatch(e -> e instanceof java.lang.classfile.instruction.InvokeInstruction call
                                && (call.owner().asInternalName().equals(vector)
                                    || call.owner().asInternalName().equals(CpuVectorMath.class.getName().replace('.','/')))
                                && call.typeSymbol().descriptorString().contains("L"+vector+";")
                                && !carrierOnly.contains(call.name().stringValue()));
                    if (!vectorMath) throw new IllegalArgumentException("selected row artifact contains no typed vector arithmetic");
                }
            }
            if (model.majorVersion() != ClassFile.JAVA_26_VERSION
                    || !model.thisClass().asInternalName().equals(expectedName)
                    || !model.interfaces().isEmpty() || !model.fields().isEmpty()) {
                throw new IllegalArgumentException("generated class shape does not match specialization");
            }
        } catch (IllegalArgumentException failure) { throw failure; }
        catch (RuntimeException failure) {
            throw new IllegalArgumentException("generated class verification failed", failure);
        }
    }
}

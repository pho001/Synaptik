package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;

/**
 * Emits only the bounded two-pass ordinary softmax derivative physical form.
 * All callbacks compose code during generation; generated execution uses direct slice/feature
 * loops, typed locals and the unchanged exact SUM state, with no intermediate tensor store.
 * A proved zero feature stride permits one cotangent load per nonempty row, shared by both
 * passes. Identical physical input positions likewise share a cell load, not an arithmetic
 * result. Neither proof removes a logical SUM contribution or a captured primitive site.
 * Controlled entries additionally guard each complete slice before the hoisted cotangent load
 * and exact-sum reset using explicitly proved boundary ROW, not physical strides.
 */
public final class CpuSoftmaxBackwardEmitter {
    /** Creates a stateless generation-only emitter. */
    public CpuSoftmaxBackwardEmitter() { }

    /**
     * Emits a complete-slice scalar subrange using independent typed carrier boundaries.
     * Hoisted invocation bases and row-local cursors avoid coordinate decoding. The exact SUM
     * finishes and rounds before any second-pass primitive executes; width one retains every
     * captured site. An empty slice range performs no state or tensor access. A zero-width
     * slice performs no tensor access, although exact-state reset and finish may remain.
     * @param code nonnull builder to mutate, after typed segment-layout reservation
     * @param specialization nonnull validated exclusive backward family and carrier tuple;
     *     retained only during generation, with run bases and state supplied by each invocation
     * @throws NullPointerException if either argument is null
     * @throws java.util.NoSuchElementException if backward physical facts are absent
     */
    public void emit(CodeBuilder code, CpuKernelSpecialization specialization) {
        var ir = specialization.softmaxBackwardIr().orElseThrow();
        var aggregate = ir.aggregate(); var type = aggregate.dataType();
        var numeric = type == DataType.FLOAT64 ? TypeKind.DOUBLE : TypeKind.FLOAT;
        int count = ir.boundaries().size(), geometry = count+1, start = count+2, end = count+4;
        int rank = ir.boundaries().getLast().extents().size();
        int width = Math.toIntExact(aggregate.domainCount());
        int row = code.allocateLocal(TypeKind.INT), limit = code.allocateLocal(TypeKind.INT);
        int feature = code.allocateLocal(TypeKind.INT);
        int[] bases = new int[count], rowBases = new int[count], cursors = new int[count];
        for (int i = 0; i < count; i++) {
            bases[i] = code.allocateLocal(TypeKind.INT);
            rowBases[i] = code.allocateLocal(TypeKind.INT);
            cursors[i] = code.allocateLocal(TypeKind.INT);
            code.aload(geometry).loadConstant(15+7*rank+i).laload().l2i()
                    .loadConstant(Math.toIntExact(ir.boundaries().get(i).baseElementOffset())).iadd().istore(bases[i]);
        }
        code.lload(end).l2i().istore(limit);
        var sites = ir.sites(); int[] locals = new int[sites.values().size()];
        for (int i = 0; i < locals.length; i++) locals[i] = code.allocateLocal(numeric);
        var scalar = new CpuScalarEmitter(code); var carrier = new CpuCarrierEmitter(code);
        boolean constantGradient = ir.boundaries().get(ir.gradientInput()).effectiveStrides().getLast() == 0;
        Runnable loadGradient = () -> {
            if (constantGradient) return; // One immutable scalar cotangent load dominates both passes.
            int p = ir.gradientInput();
            carrier.load(type,specialization.carrierPattern().get(p),p,cursors[p],true);
            code.storeLocal(numeric,locals[1]);
        };
        Runnable loadSaved = () -> {
            if (ir.savedInput() == ir.gradientInput()) {
                code.loadLocal(numeric,locals[1]).storeLocal(numeric,locals[0]);
                return;
            }
            int p = ir.savedInput();
            carrier.load(type,specialization.carrierPattern().get(p),p,cursors[p],true);
            code.storeLocal(numeric,locals[0]);
        };
        CpuAggregateEmitter.emitExactSum(code,aggregate,count,geometry,
                ir.logarithmic() ? locals[1] : locals[3],locals[2],body -> {
            var loop = code.newLabel(); var done = code.newLabel();
            code.lload(start).l2i().istore(row).labelBinding(loop);
            code.iload(row).iload(limit).branch(Opcode.IF_ICMPGE,done);
            var advance = code.newLabel();
            CpuCarrierEmitter.controlFacts().ifPresent(control -> {
                var provenance = control.boundaryRows().get(count - 1);
                if (provenance.rowIndexed()) {
                    if (rank != 2 || provenance.axis() != 0)
                        throw new IllegalArgumentException("backward slice removes its proved ROW axis");
                    CpuCarrierEmitter.emitActiveRowGuard(code, row, TypeKind.INT, advance);
                }
            });
            for (int i = 0; i < count; i++) {
                code.iload(bases[i]);
                if (rank == 2) code.iload(row).loadConstant(Math.toIntExact(ir.boundaries().get(i).effectiveStrides().getFirst()))
                        .imul().iadd();
                code.istore(rowBases[i]);
            }
            if (constantGradient && width != 0) {
                int p = ir.gradientInput();
                carrier.load(type,specialization.carrierPattern().get(p),p,rowBases[p],true);
                code.storeLocal(numeric,locals[1]);
            }
            body.run();
            code.labelBinding(advance);
            code.iinc(row,1).branch(Opcode.GOTO,loop).labelBinding(done);
        },body -> featureLoop(code,ir.boundaries(),rowBases,cursors,feature,width,
                java.util.stream.IntStream.range(0,count-1).filter(p ->
                    (p == ir.gradientInput() && !constantGradient || p == ir.savedInput() && !ir.logarithmic()
                        && p != ir.gradientInput())).toArray(),body),() -> {
            loadGradient.run();
            if (!ir.logarithmic()) { loadSaved.run(); scalar.emit(sites,sites.instructions().getFirst(),locals); }
        },() -> featureLoop(code,ir.boundaries(),rowBases,cursors,feature,width,
                java.util.stream.IntStream.range(0,count).filter(p -> !constantGradient || p != ir.gradientInput()).toArray(),() -> {
            loadGradient.run(); loadSaved.run();
            for (int i = ir.logarithmic() ? 0 : 1; i < sites.instructions().size(); i++)
                scalar.emit(sites,sites.instructions().get(i),locals);
            carrier.store(type,specialization.carrierPattern().getLast(),count-1,cursors[count-1],locals[5],true);
        }));
    }

    /**
     * Emits one top-tested full logical feature pass and direct affine cursor advances.
     * The logical slice still has {@code width} contributor visits even when its cotangent
     * was loaded once at row entry; no smaller source-domain reduction is substituted.
     * @param code active method builder
     * @param bindings baked normalized physical boundaries
     * @param rowBases dominating row-origin locals
     * @param cursors reusable int element-address locals authenticated before invocation
     * @param feature int induction local
     * @param width nonnegative int-bounded logical slice width
     * @param positions physical roles actually visited by this pass
     * @param body generation-only typed element action
     */
    private static void featureLoop(CodeBuilder code,
            java.util.List<io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Binding> bindings,
            int[] rowBases, int[] cursors, int feature, int width, int[] positions, Runnable body) {
        if (width == 0) return;
        for (int i : positions) code.iload(rowBases[i]).istore(cursors[i]);
        var loop = code.newLabel(); var done = code.newLabel();
        code.loadConstant(0).istore(feature).labelBinding(loop);
        code.iload(feature).loadConstant(width).branch(Opcode.IF_ICMPGE,done);
        body.run();
        for (int i : positions) {
            long stride = bindings.get(i).effectiveStrides().getLast();
            if (stride != 0) code.iinc(cursors[i],Math.toIntExact(stride));
        }
        code.iinc(feature,1).branch(Opcode.GOTO,loop).labelBinding(done);
    }
}

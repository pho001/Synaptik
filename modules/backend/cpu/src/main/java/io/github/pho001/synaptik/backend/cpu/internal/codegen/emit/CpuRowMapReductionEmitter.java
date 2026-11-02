package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Opcode;
import java.util.*;
import java.util.function.*;

/** Generation-only owner of one increasing-row traversal and completed-terminal lifetimes. */
public final class CpuRowMapReductionEmitter {
    private CpuRowMapReductionEmitter() { }

    /**
     * Emits one row region using caller-allocated typed handles, without allocating runtime
     * objects or emitting arithmetic itself. Initialization and finish execute once, also at
     * zero rows; the activity wrapper dominates every row site and terminal contributor.
     * The same terminal value may name its exact accumulator handles in both indexed tables;
     * no distinct values may overlap. Callbacks must not change handle tables, replace
     * accumulators or write rowIndex/rowLimit.
     * The Scan caller encloses this entire region in its existing active-time scope.
     * @param code nonnull builder that allocated all supplied local slots
     * @param ir nonnull cold-proved immutable region
     * @param rowIndex distinct allocated INT induction local, initialized here
     * @param rowLimit distinct prebound nonnegative INT row-count local
     * @param valueLocals total ordinal/feature-cell table; minus one means direct storage only
     * @param terminalAccumulatorLocals terminal-position/cell table of typed accumulator slots
     * @param init nonnull generation action initializing all terminal accumulators
     * @param activeRow nonnull generation wrapper guarding all row work and operand loads
     * @param emitRowSite nonnull generation action completing the supplied producing ordinal
     * @param emitTerminalStep nonnull generation action contributing to the supplied terminal
     * @param finish nonnull completed-result publication and distinct carry-ADD generation action
     * @throws NullPointerException if a reference or a local-table row is null
     * @throws IllegalArgumentException if local coverage, widths or disjointness disagree
     * @throws IllegalStateException if a callback mutates the handle tables
     */
    static void emitBound(CodeBuilder code, CpuRowMapReductionIr ir,
            int rowIndex, int rowLimit, int[][] valueLocals, int[][] terminalAccumulatorLocals,
            Runnable init, Consumer<Runnable> activeRow, IntConsumer emitRowSite,
            IntConsumer emitTerminalStep, Runnable finish) {
        Objects.requireNonNull(code); Objects.requireNonNull(ir); Objects.requireNonNull(init);
        Objects.requireNonNull(activeRow); Objects.requireNonNull(emitRowSite);
        Objects.requireNonNull(emitTerminalStep); Objects.requireNonNull(finish);
        if (rowIndex < 0 || rowLimit < 0 || rowIndex == rowLimit
                || valueLocals.length != ir.arithmetic().values().size()
                || terminalAccumulatorLocals.length != ir.terminalOutputs().size())
            throw new IllegalArgumentException("region local-table coverage disagrees");
        var occupied = new HashSet<Integer>(); occupied.add(rowIndex); occupied.add(rowLimit);
        int width = ir.arithmetic().values().getFirst().dataType().byteWidth()/4;
        for (int v = 0; v < valueLocals.length; v++) {
            if (Objects.requireNonNull(valueLocals[v]).length != cells(ir,v,ir.rowAxes().get(v) >= 0))
                throw new IllegalArgumentException("incomplete region value feature coverage");
            int root = ir.viewOrigins().get(v);
            boolean direct = ir.arithmetic().values().get(root).kind()
                    == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr.Value.Kind.INPUT
                    || ir.retainedStores().contains(root) && !ir.rowSites().contains(root);
            if (root != v) {
                for (int cell = 0; cell < valueLocals[v].length; cell++) {
                    int slot = valueLocals[v][cell];
                    if (slot == -1 && direct) continue;
                    if (slot < 0 || slot != aliasLocal(ir,valueLocals,v,cell))
                        throw new IllegalArgumentException("view handles must name their exact producing feature cells");
                }
            } else {
                int terminal = ir.terminalOutputs().indexOf(v);
                if (terminal < 0 || !Arrays.equals(valueLocals[v],terminalAccumulatorLocals[terminal]))
                    check(new int[][]{valueLocals[v]},occupied,width,direct);
            }
        }
        for (int t = 0; t < terminalAccumulatorLocals.length; t++) {
            if (Objects.requireNonNull(terminalAccumulatorLocals[t]).length != cells(ir,ir.terminalOutputs().get(t),false))
                throw new IllegalArgumentException("incomplete region terminal feature coverage");
            check(new int[][]{terminalAccumulatorLocals[t]},occupied,width,false);
        }
        int[][] valuesBefore = Arrays.stream(valueLocals).map(int[]::clone).toArray(int[][]::new);
        int[][] terminalsBefore = Arrays.stream(terminalAccumulatorLocals).map(int[]::clone).toArray(int[][]::new);
        init.run();
        var loop = code.newLabel(); var done = code.newLabel();
        code.loadConstant(0).istore(rowIndex).labelBinding(loop).iload(rowIndex).iload(rowLimit)
                .branch(Opcode.IF_ICMPGE,done);
        activeRow.accept(() -> {
            ir.rowSites().forEach(emitRowSite::accept);
            ir.terminalOutputs().forEach(emitTerminalStep::accept);
        });
        code.iinc(rowIndex,1).branch(Opcode.GOTO,loop).labelBinding(done);
        finish.run();
        if (!Arrays.deepEquals(valueLocals,valuesBefore) || !Arrays.deepEquals(terminalAccumulatorLocals,terminalsBefore))
            throw new IllegalStateException("region callbacks changed allocated handles");
    }

    /**
     * Emits one cold-bound row range, preserving its caller-selected scalar-cell or vector-chunk
     * table convention. The binding owner validates typed coverage, aliases and load dominance
     * before this control seam; callbacks never mutate those tables or the range locals.
     * Initialization and finish execute once even for an empty range. Activity dominates every
     * producer and terminal operand load; finish belongs after the complete range, not each row.
     * @param code builder that allocated all locals
     * @param ir immutable proved region
     * @param rowIndex allocated distinct INT induction local initialized here
     * @param rowBegin distinct nonnegative INT local-slot index holding the inclusive row bound
     * @param rowEnd distinct nonnegative INT local-slot index holding the exclusive row bound;
     *     the binding owner proves its value is at least the value held in rowBegin
     * @param valueLocals total immutable-by-convention ordinal/cell-or-chunk table
     * @param terminalAccumulatorLocals terminal-position/cell-or-chunk table
     * @param init generation action initializing terminal accumulators and validity
     * @param activeRow generation wrapper gating the complete row
     * @param emitRowSite generation action completing a row producer
     * @param emitTerminalStep generation action contributing to a terminal
     * @param finish generation action publishing complete private partials or inline results
     * @throws NullPointerException if a reference or table row is null
     * @throws IllegalArgumentException if induction slots or total table coverage disagree
     * @throws IllegalStateException if a callback mutates either table
     */
    static void emitRangeBound(CodeBuilder code, CpuRowMapReductionIr ir,
            int rowIndex, int rowBegin, int rowEnd, int[][] valueLocals,
            int[][] terminalAccumulatorLocals, Runnable init, Consumer<Runnable> activeRow,
            IntConsumer emitRowSite, IntConsumer emitTerminalStep, Runnable finish) {
        Objects.requireNonNull(code); Objects.requireNonNull(ir); Objects.requireNonNull(init);
        Objects.requireNonNull(activeRow); Objects.requireNonNull(emitRowSite);
        Objects.requireNonNull(emitTerminalStep); Objects.requireNonNull(finish);
        if (rowIndex < 0 || rowBegin < 0 || rowEnd < 0 || rowIndex == rowBegin
                || rowIndex == rowEnd || rowBegin == rowEnd
                || valueLocals.length != ir.arithmetic().values().size()
                || terminalAccumulatorLocals.length != ir.terminalOutputs().size())
            throw new IllegalArgumentException("region range/table coverage disagrees");
        int[][] valuesBefore = Arrays.stream(valueLocals).map(r -> Objects.requireNonNull(r).clone()).toArray(int[][]::new);
        int[][] terminalsBefore = Arrays.stream(terminalAccumulatorLocals).map(r -> Objects.requireNonNull(r).clone()).toArray(int[][]::new);
        init.run();
        var loop = code.newLabel(); var done = code.newLabel();
        code.iload(rowBegin).istore(rowIndex).labelBinding(loop).iload(rowIndex).iload(rowEnd)
                .branch(Opcode.IF_ICMPGE,done);
        activeRow.accept(() -> {
            ir.rowSites().forEach(emitRowSite::accept);
            ir.terminalOutputs().forEach(emitTerminalStep::accept);
        });
        code.iinc(rowIndex,1).branch(Opcode.GOTO,loop).labelBinding(done);
        finish.run();
        if (!Arrays.deepEquals(valueLocals,valuesBefore) || !Arrays.deepEquals(terminalAccumulatorLocals,terminalsBefore))
            throw new IllegalStateException("region callbacks changed allocated handles");
    }

    /**
     * Reserves complete typed widths without conflating storage sentinels with live locals.
     * @param table complete feature rows to reserve
     * @param occupied already reserved slots, mutated only during this validation
     * @param width one slot for F32, two for F64
     * @param directAllowed whether minus one denotes an existing storage-backed value
     * @throws IllegalArgumentException if a sentinel is unbacked or distinct handles overlap
     */
    private static void check(int[][] table,Set<Integer> occupied,int width,boolean directAllowed) {
        for (var row : table) for (int slot : Objects.requireNonNull(row)) {
            if (directAllowed && slot == -1) continue;
            if (slot < 0) throw new IllegalArgumentException("invalid region local");
            for (int part = 0; part < width; part++) if (!occupied.add(slot+part))
                throw new IllegalArgumentException("overlapping region local widths");
        }
    }

    /**
     * Computes exact fixed feature coverage without substituting an invocation B.
     * @param ir proved region
     * @param value covered value ordinal
     * @param row whether the authenticated row axis is omitted
     * @return exact nonnegative feature-cell count
     * @throws IllegalArgumentException if a remaining feature is symbolic
     * @throws ArithmeticException if cell-count arithmetic overflows
     */
    private static int cells(CpuRowMapReductionIr ir,int value,boolean row) {
        int cells = 1; var access = ir.accesses().get(value);
        for (int axis = 0; axis < access.plan().iterationRank(); axis++) {
            if (row && axis == ir.rowAxes().get(value)) continue;
            if (access.extentBatchFactors().get(axis) != 0)
                throw new IllegalArgumentException("unbounded region feature table");
            cells = Math.multiplyExact(cells,Math.toIntExact(access.extentConstants().get(axis)));
        }
        return cells;
    }

    /**
     * Resolves an already proved view to the producing feature's exact typed handle.
     * @param ir proved complete origin and coefficient inventory
     * @param handles caller-allocated per-value handles
     * @param value derived view ordinal
     * @param cell fixed flattened feature position
     * @return producer handle, never a newly allocated local
     */
    private static int aliasLocal(CpuRowMapReductionIr ir,int[][] handles,int value,int cell) {
        int root = ir.viewOrigins().get(value); var access = ir.accesses().get(value);
        var origin = ir.accesses().get(root); long offset = 0;
        long unitC = ir.rowAxes().get(root) == 1 ? origin.strideConstants().get(0) : 1;
        long unitF = ir.rowAxes().get(root) == 1 ? origin.strideBatchFactors().get(0) : 0;
        for (int axis = access.plan().iterationRank()-1; axis >= 0; axis--) if (axis != ir.rowAxes().get(value)) {
            int extent = Math.toIntExact(access.extentConstants().get(axis)); int coordinate = cell%extent; cell /= extent;
            long stride = unitC != 0 ? access.strideConstants().get(axis)/unitC
                    : unitF != 0 ? access.strideBatchFactors().get(axis)/unitF : 0;
            offset += coordinate*stride;
        }
        return handles[root][Math.toIntExact(offset)];
    }
}

package io.github.pho001.synaptik.backend.cpu.execution;

import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Label;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.util.Objects;

/**
 * Emits deterministic half-open range, non-empty tile, and selected Vector-tail control flow for
 * the entry signature fixed by a specialization. Parallel ranges are invocation inputs; this
 * helper creates no workers and owns no scheduling or family semantics.
 */
final class CpuLoopEmitter {
    /** Callback receiving a local slot containing the current long element index. */
    @FunctionalInterface interface IndexedBody {
        /** Emits one indexed body.
         * @param elementIndexLocal local slot containing the current long element index */
        void emit(int elementIndexLocal);
    }
    /** Callback receiving long tile bounds and an int tile index in local slots. */
    @FunctionalInterface interface TiledBody {
        /** Emits one bounded tile body.
         *
         * @param tileStartLocal local slot containing inclusive long tile start
         * @param tileEndLocal local slot containing exclusive long tile end
         * @param tileIndexLocal local slot containing the zero-based int tile index
         */
        void emit(int tileStartLocal, int tileEndLocal, int tileIndexLocal);
    }

    private final CodeBuilder code;
    private final CpuKernelSpecialization specialization;

    /**
     * Creates a control-flow helper for one generated entry body.
     *
     * @param code method builder; must not be {@code null}
     * @param specialization complete immutable specialization; must not be {@code null}
     * @throws NullPointerException if either argument is {@code null}
     */
    CpuLoopEmitter(CodeBuilder code, CpuKernelSpecialization specialization) {
        this.code = Objects.requireNonNull(code, "code");
        this.specialization = Objects.requireNonNull(specialization, "specialization");
    }
    /** Returns the current entry-body builder.
     * @return the exact non-null code builder supplied at construction */ CodeBuilder code() { return code; }
    /** Returns the specialization governing emitted control flow.
     * @return the exact non-null immutable specialization supplied at construction */ CpuKernelSpecialization specialization() { return specialization; }

    /**
     * Emits one counted loop over the complete or externally assigned range.
     * @param body non-null callback invoked once at generation time to emit the loop body
     * @throws NullPointerException if {@code body} is {@code null}
     */
    void emitRange(IndexedBody body) {
        Objects.requireNonNull(body, "body"); int start = code.allocateLocal(TypeKind.LONG);
        int end = code.allocateLocal(TypeKind.LONG); emitBounds(start, end); emitIndexLoop(start, end, body);
    }

    /**
     * Emits non-empty tiles of the baked size over the current invocation range.
     * @param body non-null callback that emits one tile body
     * @throws NullPointerException if {@code body} is {@code null}
     */
    void emitTiles(TiledBody body) {
        Objects.requireNonNull(body, "body"); int start = code.allocateLocal(TypeKind.LONG);
        int end = code.allocateLocal(TypeKind.LONG); emitBounds(start, end);
        int tileStart = code.allocateLocal(TypeKind.LONG); int tileEnd = code.allocateLocal(TypeKind.LONG);
        int tileIndex = code.allocateLocal(TypeKind.INT); code.lload(start); code.lstore(tileStart);
        code.loadConstant(0); code.istore(tileIndex); Label loop = code.newLabel(); Label done = code.newLabel();
        code.labelBinding(loop); code.lload(tileStart); code.lload(end); code.lcmp(); code.branch(Opcode.IFGE, done);
        code.lload(tileStart); code.loadConstant(specialization.tileElementCount()); code.ladd(); code.lstore(tileEnd);
        Label bounded = code.newLabel(); code.lload(tileEnd); code.lload(end); code.lcmp();
        code.branch(Opcode.IFLE, bounded); code.lload(end); code.lstore(tileEnd); code.labelBinding(bounded);
        body.emit(tileStart, tileEnd, tileIndex); code.lload(tileEnd); code.lstore(tileStart);
        code.iinc(tileIndex, 1); code.branch(Opcode.GOTO, loop); code.labelBinding(done);
    }

    /**
     * Emits only the selected tail interval after the largest complete vector prefix.
     * @param scalarBody non-null scalar-tail callback
     * @param maskedVectorBody non-null masked-vector-tail callback
     * @throws NullPointerException if either callback is {@code null}
     */
    void emitTail(IndexedBody scalarBody, IndexedBody maskedVectorBody) {
        Objects.requireNonNull(scalarBody, "scalarBody");
        Objects.requireNonNull(maskedVectorBody, "maskedVectorBody");
        if (specialization.tail() == CpuKernelSpecialization.Tail.NONE) return;
        int start = code.allocateLocal(TypeKind.LONG); int end = code.allocateLocal(TypeKind.LONG);
        emitBounds(start, end); int tailStart = code.allocateLocal(TypeKind.LONG);
        int lanes = specialization.vectorShape().orElseThrow().laneCount();
        code.lload(end); code.lload(end); code.lload(start); code.lsub(); code.loadConstant((long) lanes);
        code.lrem(); code.lsub(); code.lstore(tailStart);
        if (specialization.tail() == CpuKernelSpecialization.Tail.SCALAR) {
            emitIndexLoop(tailStart, end, scalarBody);
        } else {
            Label done = code.newLabel(); code.lload(tailStart); code.lload(end); code.lcmp();
            code.branch(Opcode.IFGE, done); maskedVectorBody.emit(tailStart); code.labelBinding(done);
        }
    }

    private void emitIndexLoop(int start, int end, IndexedBody body) {
        int index = code.allocateLocal(TypeKind.LONG); code.lload(start); code.lstore(index);
        Label loop = code.newLabel(); Label done = code.newLabel(); code.labelBinding(loop);
        code.lload(index); code.lload(end); code.lcmp(); code.branch(Opcode.IFGE, done);
        body.emit(index); code.lload(index); code.loadConstant(1L); code.ladd(); code.lstore(index);
        code.branch(Opcode.GOTO, loop); code.labelBinding(done);
    }
    private void emitBounds(int start, int end) {
        int control = controlParameterSlot();
        if (specialization.executionMode().parallel()) {
            code.lload(control); code.lstore(start); code.lload(control + 2); code.lstore(end);
        } else { code.loadConstant(0L); code.lstore(start); code.lload(control); code.lstore(end); }
    }
    private int controlParameterSlot() {
        int slot = 0; int controlParameter = specialization.entryType().parameterCount()
                - (specialization.executionMode().parallel() ? 3 : 1);
        for (int index = 0; index < controlParameter; index++) slot +=
                specialization.entryType().parameterType(index) == long.class ? 2 : 1;
        return slot;
    }
}

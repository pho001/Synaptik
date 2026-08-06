package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.util.List;
import java.util.function.Consumer;

/** Emits generation-time-selected primitive state machines for normalized access regimes. */
final class CpuLoopEmitter {
    /**
     * Generation-local element-address slots supplied to the fused body emitter.
     *
     * @param addresses non-null ordered local-variable slots for the four boundary addresses
     */
    record State(int[] addresses) { }
    private record PlanState(CpuAccessPlan plan, int address, int[] outerCoordinates,
            int innerPosition) { }
    private final CodeBuilder code;
    /**
     * Creates an emitter bound to one non-null generated method body.
     *
     * @param code non-null Class-File API code builder retained for generation only
     */
    CpuLoopEmitter(CodeBuilder code) { this.code = code; }

    /**
     * Emits one universal half-open loop and regime-specific address state for four boundaries.
     *
     * @param plans non-null ordered structural plans sharing one iteration rank
     * @param body non-null generation callback receiving the four address-local slots
     */
    void emit(List<CpuAccessPlan> plans, Consumer<State> body) {
        int rank = plans.getFirst().iterationRank();
        var done = code.newLabel();
        code.lload(5).lload(7).lcmp().branch(Opcode.IFGE, done);
        int[] addresses = new int[4];
        PlanState[] states = new PlanState[4];
        for (int value = 0; value < 4; value++) {
            CpuAccessPlan plan = plans.get(value);
            addresses[value] = code.allocateLocal(TypeKind.LONG);
            loadGeometry(2 * rank + value).lstore(addresses[value]);
            int coordinateCount = switch (plan.regime()) {
                case GENERAL_ODOMETER -> rank;
                case BLOCK_OUTER -> rank - plan.contiguousSuffix();
                default -> 0;
            };
            int[] coordinates = new int[coordinateCount];
            for (int axis = 0; axis < coordinateCount; axis++) {
                coordinates[axis] = code.allocateLocal(TypeKind.LONG);
                loadGeometry(rank + axis).lstore(coordinates[axis]);
            }
            int inner = -1;
            if (plan.regime() == CpuAccessPlan.Regime.LAST_AXIS_BIAS
                    || plan.regime() == CpuAccessPlan.Regime.BLOCK_OUTER) {
                inner = code.allocateLocal(TypeKind.LONG);
                loadGeometry(2 * rank + 4 + 4 * rank + value).lstore(inner);
            }
            states[value] = new PlanState(plan, addresses[value], coordinates, inner);
        }
        int index = code.allocateLocal(TypeKind.LONG);
        code.lload(5).lstore(index);
        var loop = code.newLabel();
        code.labelBinding(loop);
        body.accept(new State(addresses));
        code.lload(index).loadConstant(1L).ladd().lstore(index);
        code.lload(index).lload(7).lcmp().branch(Opcode.IFGE, done);
        for (int value = 0; value < 4; value++) emitAdvance(states[value], value, rank);
        code.branch(Opcode.GOTO, loop);
        code.labelBinding(done);
    }

    /**
     * Emits unmasked full vectors inside every participating contiguous run and delegates each
     * remaining element to the scalar body before advancing the same primitive address state.
     *
     * @param plans non-null ordered structural plans sharing one iteration rank; no plan may use
     *     the general-odometer regime
     * @param lanes positive lane count of the already-selected exact species
     * @param vectorBody non-null generation callback for one complete vector
     * @param scalarBody non-null generation callback for one scalar remainder element
     * @throws IllegalArgumentException if vector advancement encounters a general-odometer plan
     */
    void emitVector(List<CpuAccessPlan> plans, int lanes, Consumer<State> vectorBody,
            Consumer<State> scalarBody) {
        int rank = plans.getFirst().iterationRank();
        var done = code.newLabel();
        code.lload(5).lload(7).lcmp().branch(Opcode.IFGE, done);
        int[] addresses = new int[4];
        PlanState[] states = new PlanState[4];
        for (int value = 0; value < 4; value++) {
            CpuAccessPlan plan = plans.get(value);
            addresses[value] = code.allocateLocal(TypeKind.LONG);
            loadGeometry(2 * rank + value).lstore(addresses[value]);
            int coordinateCount = switch (plan.regime()) {
                case GENERAL_ODOMETER -> rank;
                case BLOCK_OUTER -> rank - plan.contiguousSuffix();
                default -> 0;
            };
            int[] coordinates = new int[coordinateCount];
            for (int axis = 0; axis < coordinateCount; axis++) {
                coordinates[axis] = code.allocateLocal(TypeKind.LONG);
                loadGeometry(rank + axis).lstore(coordinates[axis]);
            }
            int inner = -1;
            if (plan.regime() == CpuAccessPlan.Regime.LAST_AXIS_BIAS
                    || plan.regime() == CpuAccessPlan.Regime.BLOCK_OUTER) {
                inner = code.allocateLocal(TypeKind.LONG);
                loadGeometry(2 * rank + 4 + 4 * rank + value).lstore(inner);
            }
            states[value] = new PlanState(plan, addresses[value], coordinates, inner);
        }
        int index = code.allocateLocal(TypeKind.LONG);
        int available = code.allocateLocal(TypeKind.LONG);
        code.lload(5).lstore(index);
        var loop = code.newLabel();
        var scalar = code.newLabel();
        code.labelBinding(loop);
        code.lload(index).lload(7).lcmp().branch(Opcode.IFGE, done);
        code.lload(7).lload(index).lsub().lstore(available);
        for (int value = 0; value < 4; value++) {
            PlanState state = states[value];
            if (state.plan().regime() != CpuAccessPlan.Regime.LAST_AXIS_BIAS
                    && state.plan().regime() != CpuAccessPlan.Regime.BLOCK_OUTER) continue;
            int sizeIndex = state.plan().regime() == CpuAccessPlan.Regime.LAST_AXIS_BIAS
                    ? rank - 1 : 2 * rank + 4 + 4 * rank + 4 + value;
            int candidate = code.allocateLocal(TypeKind.LONG);
            loadGeometry(sizeIndex).lload(state.innerPosition()).lsub().lstore(candidate);
            var keep = code.newLabel();
            code.lload(candidate).lload(available).lcmp().branch(Opcode.IFGE, keep);
            code.lload(candidate).lstore(available);
            code.labelBinding(keep);
        }
        code.lload(available).loadConstant((long) lanes).lcmp().branch(Opcode.IFLT, scalar);
        vectorBody.accept(new State(addresses));
        code.lload(index).loadConstant((long) lanes).ladd().lstore(index);
        for (int value = 0; value < 4; value++) emitVectorAdvance(
                states[value], value, rank, lanes);
        code.branch(Opcode.GOTO, loop);
        code.labelBinding(scalar);
        scalarBody.accept(new State(addresses));
        code.lload(index).loadConstant(1L).ladd().lstore(index);
        code.lload(index).lload(7).lcmp().branch(Opcode.IFGE, done);
        for (int value = 0; value < 4; value++) emitAdvance(states[value], value, rank);
        code.branch(Opcode.GOTO, loop);
        code.labelBinding(done);
    }

    private void emitVectorAdvance(PlanState state, int value, int rank, int lanes) {
        switch (state.plan().regime()) {
            case DENSE_LINEAR -> increment(state.address(), lanes);
            case SCALAR_ALL_ZERO -> { }
            case LAST_AXIS_BIAS -> {
                increment(state.address(), lanes); increment(state.innerPosition(), lanes);
                var finished = code.newLabel();
                code.lload(state.innerPosition()); loadGeometry(rank - 1).lcmp()
                        .branch(Opcode.IFLT, finished);
                code.loadConstant(0L).lstore(state.innerPosition());
                code.lload(state.address()); loadGeometry(rank - 1).lsub()
                        .lstore(state.address());
                code.labelBinding(finished);
            }
            case BLOCK_OUTER -> {
                int innerSizeIndex = 2 * rank + 4 + 4 * rank + 4 + value;
                increment(state.address(), lanes); increment(state.innerPosition(), lanes);
                var finished = code.newLabel();
                code.lload(state.innerPosition()); loadGeometry(innerSizeIndex).lcmp()
                        .branch(Opcode.IFLT, finished);
                code.loadConstant(0L).lstore(state.innerPosition());
                code.lload(state.address()); loadGeometry(innerSizeIndex).lsub()
                        .lstore(state.address());
                emitOuterCarry(state, value, rank, finished);
                code.labelBinding(finished);
            }
            case GENERAL_ODOMETER -> throw new IllegalArgumentException(
                    "general odometer cannot use vector emission");
        }
    }

    private void emitAdvance(PlanState state, int value, int rank) {
        switch (state.plan().regime()) {
            case DENSE_LINEAR -> increment(state.address(), 1);
            case SCALAR_ALL_ZERO -> { }
            case LAST_AXIS_BIAS -> emitLastAxis(state, rank);
            case BLOCK_OUTER -> emitBlockOuter(state, value, rank);
            case GENERAL_ODOMETER -> emitGeneral(state, value, rank);
        }
    }

    private void emitLastAxis(PlanState state, int rank) {
        increment(state.address(), 1);
        increment(state.innerPosition(), 1);
        var finished = code.newLabel();
        code.lload(state.innerPosition()); loadGeometry(rank - 1).lcmp()
                .branch(Opcode.IFLT, finished);
        code.loadConstant(0L).lstore(state.innerPosition());
        code.lload(state.address()); loadGeometry(rank - 1).lsub().lstore(state.address());
        code.labelBinding(finished);
    }

    private void emitBlockOuter(PlanState state, int value, int rank) {
        increment(state.address(), 1);
        increment(state.innerPosition(), 1);
        int innerSizeIndex = 2 * rank + 4 + 4 * rank + 4 + value;
        var finished = code.newLabel();
        code.lload(state.innerPosition()); loadGeometry(innerSizeIndex).lcmp()
                .branch(Opcode.IFLT, finished);
        code.loadConstant(0L).lstore(state.innerPosition());
        code.lload(state.address()); loadGeometry(innerSizeIndex).lsub().lstore(state.address());
        emitOuterCarry(state, value, rank, finished);
        code.labelBinding(finished);
    }

    private void emitGeneral(PlanState state, int value, int rank) {
        var finished = code.newLabel();
        emitOuterCarry(state, value, rank, finished);
        code.labelBinding(finished);
    }

    private void emitOuterCarry(PlanState state, int value, int rank,
            java.lang.classfile.Label finished) {
        for (int axis = state.outerCoordinates().length - 1; axis >= 0; axis--) {
            increment(state.outerCoordinates()[axis], 1);
            code.lload(state.address()); stride(value, rank, axis).ladd().lstore(state.address());
            code.lload(state.outerCoordinates()[axis]); loadGeometry(axis).lcmp()
                    .branch(Opcode.IFLT, finished);
            code.loadConstant(0L).lstore(state.outerCoordinates()[axis]);
            code.lload(state.address()); stride(value, rank, axis);
            loadGeometry(axis).lmul().lsub().lstore(state.address());
        }
    }

    private void increment(int local, long amount) {
        code.lload(local).loadConstant(amount).ladd().lstore(local);
    }

    private CodeBuilder stride(int value, int rank, int axis) {
        return loadGeometry(2 * rank + 4 + value * rank + axis);
    }

    private CodeBuilder loadGeometry(int index) {
        return code.aload(4).loadConstant(index).laload();
    }
}

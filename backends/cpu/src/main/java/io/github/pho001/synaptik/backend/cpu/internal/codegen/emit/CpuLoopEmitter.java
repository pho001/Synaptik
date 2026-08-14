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
     * @param addresses non-null ordered local-variable slots for all derived boundary addresses
     * @param intAddresses whether every slot contains a proved Java array index rather than a
     *     general element address represented as {@code long}
     */
    record State(int[] addresses, boolean intAddresses) { }
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
     * Emits one universal half-open loop and regime-specific address state for all boundaries.
     *
     * @param plans non-null ordered structural plans sharing one iteration rank
     * @param body non-null generation callback receiving all ordered address-local slots
     */
    void emit(List<CpuAccessPlan> plans, Consumer<State> body) {
        int rank = plans.getFirst().iterationRank();
        int boundaryCount = plans.size();
        int geometrySlot = boundaryCount;
        int startSlot = boundaryCount + 1;
        int endSlot = boundaryCount + 3;
        var done = code.newLabel();
        code.lload(startSlot).lload(endSlot).lcmp().branch(Opcode.IFGE, done);
        int[] addresses = new int[boundaryCount];
        PlanState[] states = new PlanState[boundaryCount];
        for (int value = 0; value < boundaryCount; value++) {
            CpuAccessPlan plan = plans.get(value);
            addresses[value] = code.allocateLocal(TypeKind.LONG);
            loadGeometry(geometrySlot, 2 * rank + value).lstore(addresses[value]);
            int coordinateCount = switch (plan.regime()) {
                case GENERAL_ODOMETER -> rank;
                case BLOCK_OUTER -> rank - plan.contiguousSuffix();
                default -> 0;
            };
            int[] coordinates = new int[coordinateCount];
            for (int axis = 0; axis < coordinateCount; axis++) {
                coordinates[axis] = code.allocateLocal(TypeKind.LONG);
                loadGeometry(geometrySlot, rank + axis).lstore(coordinates[axis]);
            }
            int inner = -1;
            if (plan.regime() == CpuAccessPlan.Regime.LAST_AXIS_BIAS
                    || plan.regime() == CpuAccessPlan.Regime.BLOCK_OUTER) {
                inner = code.allocateLocal(TypeKind.LONG);
                loadGeometry(geometrySlot, 2 * rank + boundaryCount
                        + boundaryCount * rank + value).lstore(inner);
            }
            states[value] = new PlanState(plan, addresses[value], coordinates, inner);
        }
        int index = code.allocateLocal(TypeKind.LONG);
        code.lload(startSlot).lstore(index);
        var loop = code.newLabel();
        code.labelBinding(loop);
        body.accept(new State(addresses, false));
        code.lload(index).loadConstant(1L).ladd().lstore(index);
        code.lload(index).lload(endSlot).lcmp().branch(Opcode.IFGE, done);
        for (int value = 0; value < boundaryCount; value++) emitAdvance(
                states[value], value, rank, boundaryCount, geometrySlot);
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
        int boundaryCount = plans.size();
        int geometrySlot = boundaryCount;
        int startSlot = boundaryCount + 1;
        int endSlot = boundaryCount + 3;
        var done = code.newLabel();
        code.lload(startSlot).lload(endSlot).lcmp().branch(Opcode.IFGE, done);
        int[] addresses = new int[boundaryCount];
        PlanState[] states = new PlanState[boundaryCount];
        for (int value = 0; value < boundaryCount; value++) {
            CpuAccessPlan plan = plans.get(value);
            addresses[value] = code.allocateLocal(TypeKind.LONG);
            loadGeometry(geometrySlot, 2 * rank + value).lstore(addresses[value]);
            int coordinateCount = switch (plan.regime()) {
                case GENERAL_ODOMETER -> rank;
                case BLOCK_OUTER -> rank - plan.contiguousSuffix();
                default -> 0;
            };
            int[] coordinates = new int[coordinateCount];
            for (int axis = 0; axis < coordinateCount; axis++) {
                coordinates[axis] = code.allocateLocal(TypeKind.LONG);
                loadGeometry(geometrySlot, rank + axis).lstore(coordinates[axis]);
            }
            int inner = -1;
            if (plan.regime() == CpuAccessPlan.Regime.LAST_AXIS_BIAS
                    || plan.regime() == CpuAccessPlan.Regime.BLOCK_OUTER) {
                inner = code.allocateLocal(TypeKind.LONG);
                loadGeometry(geometrySlot, 2 * rank + boundaryCount
                        + boundaryCount * rank + value).lstore(inner);
            }
            states[value] = new PlanState(plan, addresses[value], coordinates, inner);
        }
        int index = code.allocateLocal(TypeKind.LONG);
        int available = code.allocateLocal(TypeKind.LONG);
        code.lload(startSlot).lstore(index);
        var loop = code.newLabel();
        var scalar = code.newLabel();
        code.labelBinding(loop);
        code.lload(index).lload(endSlot).lcmp().branch(Opcode.IFGE, done);
        code.lload(endSlot).lload(index).lsub().lstore(available);
        for (int value = 0; value < boundaryCount; value++) {
            PlanState state = states[value];
            if (state.plan().regime() != CpuAccessPlan.Regime.LAST_AXIS_BIAS
                    && state.plan().regime() != CpuAccessPlan.Regime.BLOCK_OUTER) continue;
            int sizeIndex = state.plan().regime() == CpuAccessPlan.Regime.LAST_AXIS_BIAS
                    ? rank - 1 : 2 * rank + boundaryCount + boundaryCount * rank
                            + boundaryCount + value;
            int candidate = code.allocateLocal(TypeKind.LONG);
            loadGeometry(geometrySlot, sizeIndex).lload(state.innerPosition()).lsub().lstore(candidate);
            var keep = code.newLabel();
            code.lload(candidate).lload(available).lcmp().branch(Opcode.IFGE, keep);
            code.lload(candidate).lstore(available);
            code.labelBinding(keep);
        }
        code.lload(available).loadConstant((long) lanes).lcmp().branch(Opcode.IFLT, scalar);
        vectorBody.accept(new State(addresses, false));
        code.lload(index).loadConstant((long) lanes).ladd().lstore(index);
        for (int value = 0; value < boundaryCount; value++) emitVectorAdvance(
                states[value], value, rank, lanes, boundaryCount, geometrySlot);
        code.branch(Opcode.GOTO, loop);
        code.labelBinding(scalar);
        scalarBody.accept(new State(addresses, false));
        code.lload(index).loadConstant(1L).ladd().lstore(index);
        code.lload(index).lload(endSlot).lcmp().branch(Opcode.IFGE, done);
        for (int value = 0; value < boundaryCount; value++) emitAdvance(
                states[value], value, rank, boundaryCount, geometrySlot);
        code.branch(Opcode.GOTO, loop);
        code.labelBinding(done);
    }

    /**
     * Emits a proved dense heap-array scalar loop after one-time narrowing at entry.
     * @param plans non-null dense-linear or all-zero structural plans
     * @param body non-null scalar body receiving integer address locals
     */
    void emitDenseArrayInt(List<CpuAccessPlan> plans, Consumer<State> body) {
        int boundaryCount = plans.size(), geometrySlot = boundaryCount;
        int startSlot = boundaryCount + 1, endSlot = boundaryCount + 3;
        var done = code.newLabel();
        code.lload(startSlot).lload(endSlot).lcmp().branch(Opcode.IFGE, done);
        int end = code.allocateLocal(TypeKind.INT);
        code.lload(endSlot).l2i().istore(end);
        int[] addresses = denseIntAddresses(plans, geometrySlot);
        int index = code.allocateLocal(TypeKind.INT);
        code.lload(startSlot).l2i().istore(index);
        var loop = code.newLabel();
        code.labelBinding(loop);
        body.accept(new State(addresses, true));
        incrementDenseAddresses(plans, addresses, 1);
        code.iinc(index, 1);
        code.iload(index).iload(end).branch(Opcode.IF_ICMPLT, loop);
        code.labelBinding(done);
    }

    /**
     * Emits one dense heap-array vector loop followed by one scalar tail.
     * @param plans non-null dense-linear or all-zero structural plans
     * @param lanes positive preferred-species lane count
     * @param vectorBody non-null body for one complete unmasked vector
     * @param scalarBody non-null body for one scalar tail element
     */
    void emitDenseArrayIntVector(List<CpuAccessPlan> plans, int lanes,
            Consumer<State> vectorBody, Consumer<State> scalarBody) {
        int boundaryCount = plans.size(), geometrySlot = boundaryCount;
        int startSlot = boundaryCount + 1, endSlot = boundaryCount + 3;
        var done = code.newLabel();
        code.lload(startSlot).lload(endSlot).lcmp().branch(Opcode.IFGE, done);
        int start = code.allocateLocal(TypeKind.INT), end = code.allocateLocal(TypeKind.INT);
        code.lload(startSlot).l2i().istore(start);
        code.lload(endSlot).l2i().istore(end);
        int bound = code.allocateLocal(TypeKind.INT);
        code.iload(end).iload(end).iload(start).isub().loadConstant(lanes).irem()
                .isub().istore(bound);
        int[] addresses = denseIntAddresses(plans, geometrySlot);
        int index = code.allocateLocal(TypeKind.INT);
        code.iload(start).istore(index);
        var scalar = code.newLabel();
        code.iload(index).iload(bound).branch(Opcode.IF_ICMPGE, scalar);
        var vector = code.newLabel();
        code.labelBinding(vector);
        vectorBody.accept(new State(addresses, true));
        incrementDenseAddresses(plans, addresses, lanes);
        code.iinc(index, lanes);
        code.iload(index).iload(bound).branch(Opcode.IF_ICMPLT, vector);
        code.labelBinding(scalar);
        code.iload(index).iload(end).branch(Opcode.IF_ICMPGE, done);
        var tail = code.newLabel();
        code.labelBinding(tail);
        scalarBody.accept(new State(addresses, true));
        incrementDenseAddresses(plans, addresses, 1);
        code.iinc(index, 1);
        code.iload(index).iload(end).branch(Opcode.IF_ICMPLT, tail);
        code.labelBinding(done);
    }

    private int[] denseIntAddresses(List<CpuAccessPlan> plans, int geometrySlot) {
        int rank = plans.getFirst().iterationRank();
        int[] addresses = new int[plans.size()];
        for (int value = 0; value < plans.size(); value++) {
            addresses[value] = code.allocateLocal(TypeKind.INT);
            loadGeometry(geometrySlot, 2 * rank + value).l2i().istore(addresses[value]);
        }
        return addresses;
    }

    private void incrementDenseAddresses(List<CpuAccessPlan> plans, int[] addresses, int amount) {
        for (int value = 0; value < plans.size(); value++)
            if (plans.get(value).regime() == CpuAccessPlan.Regime.DENSE_LINEAR)
                code.iinc(addresses[value], amount);
    }

    private void emitVectorAdvance(PlanState state, int value, int rank, int lanes,
            int boundaryCount, int geometrySlot) {
        switch (state.plan().regime()) {
            case DENSE_LINEAR -> increment(state.address(), lanes);
            case SCALAR_ALL_ZERO -> { }
            case LAST_AXIS_BIAS -> {
                increment(state.address(), lanes); increment(state.innerPosition(), lanes);
                var finished = code.newLabel();
                code.lload(state.innerPosition()); loadGeometry(geometrySlot, rank - 1).lcmp()
                        .branch(Opcode.IFLT, finished);
                code.loadConstant(0L).lstore(state.innerPosition());
                code.lload(state.address()); loadGeometry(geometrySlot, rank - 1).lsub()
                        .lstore(state.address());
                code.labelBinding(finished);
            }
            case BLOCK_OUTER -> {
                int innerSizeIndex = 2 * rank + boundaryCount + boundaryCount * rank
                        + boundaryCount + value;
                increment(state.address(), lanes); increment(state.innerPosition(), lanes);
                var finished = code.newLabel();
                code.lload(state.innerPosition()); loadGeometry(geometrySlot, innerSizeIndex).lcmp()
                        .branch(Opcode.IFLT, finished);
                code.loadConstant(0L).lstore(state.innerPosition());
                code.lload(state.address()); loadGeometry(geometrySlot, innerSizeIndex).lsub()
                        .lstore(state.address());
                emitOuterCarry(state, value, rank, boundaryCount, geometrySlot, finished);
                code.labelBinding(finished);
            }
            case GENERAL_ODOMETER -> throw new IllegalArgumentException(
                    "general odometer cannot use vector emission");
        }
    }

    private void emitAdvance(PlanState state, int value, int rank, int boundaryCount,
            int geometrySlot) {
        switch (state.plan().regime()) {
            case DENSE_LINEAR -> increment(state.address(), 1);
            case SCALAR_ALL_ZERO -> { }
            case LAST_AXIS_BIAS -> emitLastAxis(state, rank, geometrySlot);
            case BLOCK_OUTER -> emitBlockOuter(state, value, rank, boundaryCount, geometrySlot);
            case GENERAL_ODOMETER -> emitGeneral(state, value, rank, boundaryCount, geometrySlot);
        }
    }

    private void emitLastAxis(PlanState state, int rank, int geometrySlot) {
        increment(state.address(), 1);
        increment(state.innerPosition(), 1);
        var finished = code.newLabel();
        code.lload(state.innerPosition()); loadGeometry(geometrySlot, rank - 1).lcmp()
                .branch(Opcode.IFLT, finished);
        code.loadConstant(0L).lstore(state.innerPosition());
        code.lload(state.address()); loadGeometry(geometrySlot, rank - 1).lsub().lstore(state.address());
        code.labelBinding(finished);
    }

    private void emitBlockOuter(PlanState state, int value, int rank, int boundaryCount,
            int geometrySlot) {
        increment(state.address(), 1);
        increment(state.innerPosition(), 1);
        int innerSizeIndex = 2 * rank + boundaryCount + boundaryCount * rank
                + boundaryCount + value;
        var finished = code.newLabel();
        code.lload(state.innerPosition()); loadGeometry(geometrySlot, innerSizeIndex).lcmp()
                .branch(Opcode.IFLT, finished);
        code.loadConstant(0L).lstore(state.innerPosition());
        code.lload(state.address()); loadGeometry(geometrySlot, innerSizeIndex).lsub().lstore(state.address());
        emitOuterCarry(state, value, rank, boundaryCount, geometrySlot, finished);
        code.labelBinding(finished);
    }

    private void emitGeneral(PlanState state, int value, int rank, int boundaryCount,
            int geometrySlot) {
        var finished = code.newLabel();
        emitOuterCarry(state, value, rank, boundaryCount, geometrySlot, finished);
        code.labelBinding(finished);
    }

    private void emitOuterCarry(PlanState state, int value, int rank, int boundaryCount,
            int geometrySlot,
            java.lang.classfile.Label finished) {
        for (int axis = state.outerCoordinates().length - 1; axis >= 0; axis--) {
            increment(state.outerCoordinates()[axis], 1);
            code.lload(state.address()); stride(geometrySlot, value, rank, boundaryCount, axis)
                    .ladd().lstore(state.address());
            code.lload(state.outerCoordinates()[axis]); loadGeometry(geometrySlot, axis).lcmp()
                    .branch(Opcode.IFLT, finished);
            code.loadConstant(0L).lstore(state.outerCoordinates()[axis]);
            code.lload(state.address()); stride(geometrySlot, value, rank, boundaryCount, axis);
            loadGeometry(geometrySlot, axis).lmul().lsub().lstore(state.address());
        }
    }

    private void increment(int local, long amount) {
        code.lload(local).loadConstant(amount).ladd().lstore(local);
    }

    private CodeBuilder stride(int geometrySlot, int value, int rank, int boundaryCount, int axis) {
        return loadGeometry(geometrySlot, 2 * rank + boundaryCount + value * rank + axis);
    }

    private CodeBuilder loadGeometry(int geometrySlot, int index) {
        return code.aload(geometrySlot).loadConstant(index).laload();
    }
}

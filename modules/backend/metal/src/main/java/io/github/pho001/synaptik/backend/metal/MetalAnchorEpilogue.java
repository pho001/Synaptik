package io.github.pho001.synaptik.backend.metal;

import java.util.List;
import java.util.Objects;

/** Immutable source-ordered v1 epilogue attached to one physical Metal anchor dispatch. */
final class MetalAnchorEpilogue {
    enum AnchorKind {
        MATMUL(1), CONV2D(2);

        private final int wire;

        AnchorKind(int wire) {
            this.wire = wire;
        }

        int wire() {
            return wire;
        }
    }

    enum Opcode {
        SCALAR_MUL(1), ADD(2), RELU(3), CLAMP(4);

        private final int wire;

        Opcode(int wire) {
            this.wire = wire;
        }

        int wire() {
            return wire;
        }
    }

    sealed interface Operation permits ScalarMultiply, Add, Relu, Clamp {
        int nodePosition();
        int inputValue();
        int outputValue();
        Opcode opcode();
    }

    record ScalarMultiply(int nodePosition, int inputValue, int outputValue, int rawBits)
            implements Operation {
        ScalarMultiply {
            requireIndices(nodePosition, inputValue, outputValue);
        }

        @Override
        public Opcode opcode() {
            return Opcode.SCALAR_MUL;
        }
    }

    record Add(
            int nodePosition,
            int inputValue,
            int outputValue,
            int leftValue,
            int rightValue,
            int externalValue,
            boolean externalOnLeft)
            implements Operation {
        Add {
            requireIndices(nodePosition, inputValue, outputValue);
            if (leftValue < 0 || rightValue < 0 || externalValue < 0
                    || (externalOnLeft ? rightValue : leftValue) != inputValue
                    || (externalOnLeft ? leftValue : rightValue) != externalValue
                    || externalValue == inputValue) {
                throw new IllegalArgumentException("malformed Metal anchor ADD");
            }
        }

        @Override
        public Opcode opcode() {
            return Opcode.ADD;
        }
    }

    record Relu(int nodePosition, int inputValue, int outputValue) implements Operation {
        Relu {
            requireIndices(nodePosition, inputValue, outputValue);
        }

        @Override
        public Opcode opcode() {
            return Opcode.RELU;
        }
    }

    record Clamp(
            int nodePosition,
            int inputValue,
            int outputValue,
            int lowerRawBits,
            int upperRawBits)
            implements Operation {
        Clamp {
            requireIndices(nodePosition, inputValue, outputValue);
        }

        @Override
        public Opcode opcode() {
            return Opcode.CLAMP;
        }
    }

    private final AnchorKind anchorKind;
    private final int anchorNodePosition;
    private final int anchorOutputValue;
    private final List<Operation> operations;

    MetalAnchorEpilogue(
            AnchorKind anchorKind,
            int anchorNodePosition,
            int anchorOutputValue,
            List<? extends Operation> operations) {
        this.anchorKind = Objects.requireNonNull(anchorKind, "anchorKind");
        if (anchorNodePosition < 0 || anchorOutputValue < 0) {
            throw new IllegalArgumentException("malformed Metal anchor position");
        }
        this.anchorNodePosition = anchorNodePosition;
        this.anchorOutputValue = anchorOutputValue;
        this.operations = List.copyOf(operations);
        validate();
    }

    private void validate() {
        if (operations.isEmpty() || operations.size() > 3) {
            throw new IllegalArgumentException("Metal anchor epilogue must contain one to three nodes");
        }
        boolean scalar = false;
        boolean add = false;
        boolean terminal = false;
        int priorValue = anchorOutputValue;
        for (int index = 0; index < operations.size(); index++) {
            Operation operation = Objects.requireNonNull(operations.get(index), "operation");
            if (operation.nodePosition() != anchorNodePosition + index + 1
                    || operation.inputValue() != priorValue) {
                throw new IllegalArgumentException("Metal anchor epilogue is not a linear source-order suffix");
            }
            switch (operation.opcode()) {
                case SCALAR_MUL -> {
                    if (scalar || add || terminal || index != 0) throw malformedOrder();
                    scalar = true;
                }
                case ADD -> {
                    if (add || terminal) throw malformedOrder();
                    add = true;
                }
                case RELU, CLAMP -> {
                    if (terminal || index != operations.size() - 1) throw malformedOrder();
                    terminal = true;
                }
            }
            priorValue = operation.outputValue();
        }
    }

    private static IllegalArgumentException malformedOrder() {
        return new IllegalArgumentException("Metal anchor epilogue order is not v1");
    }

    private static void requireIndices(int nodePosition, int inputValue, int outputValue) {
        if (nodePosition < 0 || inputValue < 0 || outputValue < 0 || inputValue == outputValue) {
            throw new IllegalArgumentException("malformed Metal anchor epilogue operation");
        }
    }

    AnchorKind anchorKind() {
        return anchorKind;
    }

    int anchorNodePosition() {
        return anchorNodePosition;
    }

    int anchorOutputValue() {
        return anchorOutputValue;
    }

    List<Operation> operations() {
        return operations;
    }

    int finalOutputValue() {
        return operations.getLast().outputValue();
    }

    int memberCount() {
        return operations.size() + 1;
    }

    int externalAddValue() {
        for (Operation operation : operations) {
            if (operation instanceof Add add) return add.externalValue();
        }
        return -1;
    }
}

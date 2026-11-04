package io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Immutable, handle-free native ABI recipes for one existing common CPU unit. A pipeline fixes
 * literal arithmetic stores at preparation; it is not an alternative semantic graph or executor.
 * Representability is separate from {@link CpuOneDnnCompiler#qualified(CpuOneDnnKernelSpec)}.
 */
public sealed interface CpuOneDnnKernelSpec permits CpuOneDnnKernelSpec.Atomic, CpuOneDnnKernelSpec.Pipeline {
    /** @return immutable descriptor-order borrowed input geometries */
    List<Tensor> inputs();
    /** @return exact external output geometry */
    Tensor output();

    /** One native operation, never a recursively nested pipeline. */
    sealed interface Atomic extends CpuOneDnnKernelSpec permits Contraction, Unary, Binary, Reduction, Reorder, Pooling { }
    /** Contraction descriptor kinds, including unqualified diagnostic convolution. */
    enum Operation { MATMUL, CONVOLUTION }
    /** Literal contraction bias role; external ADD order must not be silently changed. */
    enum Bias { NONE, INTRINSIC, EXTERNAL_LEFT, EXTERNAL_RIGHT }
    /** Diagnostic contraction post-operations; these do not confer qualification. */
    enum Activation { NONE, RELU, TANH, SIGMOID }
    /** Standalone named elementary sites. */
    enum UnaryAlgorithm { ABS, SQRT, TANH, EXP }
    /** Stored-value predicates produce canonical one-byte BOOL; GT/GE reverse LT/LE operands. */
    enum BinaryAlgorithm { ADD, SUB, MUL, DIV, MIN, MAX, LT, LE, EQ, NE }
    /** All-contributor reductions; MEAN requires a separate literal DIV site. */
    enum ReductionAlgorithm { SUM, PROD }

    /**
     * Descriptor relative to its bound segment, with no origin prefix.
     * @param dataType non-null represented element type
     * @param dimensions positive extents of rank 1..12; scalars use [1]
     * @param strides matching nonnegative element strides; copied
     */
    record Tensor(DataType dataType, List<Long> dimensions, List<Long> strides) {
        public Tensor {
            Objects.requireNonNull(dataType, "dataType");
            dimensions = List.copyOf(dimensions);
            strides = List.copyOf(strides);
            if (dimensions.isEmpty() || dimensions.size() > 12 || dimensions.size() != strides.size()
                    || dimensions.stream().anyMatch(value -> value <= 0)
                    || strides.stream().anyMatch(value -> value < 0)) {
                throw new IllegalArgumentException("invalid oneDNN tensor descriptor");
            }
            spanBytes(dataType, dimensions, strides);
        }
        /** @return checked referenced byte span, not logical count or origin prefix */
        public long byteSize() { return spanBytes(dataType, dimensions, strides); }
    }

    /**
     * Literal diagnostic contraction recipe. All collections are immutable snapshots.
     * @param operation contraction kind
     * @param inputs two operands and a third exactly when bias is present
     * @param output result descriptor
     * @param groups positive group count, one for MATMUL
     * @param strides positive spatial steps, empty for MATMUL
     * @param dilations positive spatial factors, empty for MATMUL
     * @param paddingLeft nonnegative leading padding, empty for MATMUL
     * @param paddingRight nonnegative trailing padding, empty for MATMUL
     * @param bias literal bias role
     * @param activation literal post-operation
     */
    record Contraction(Operation operation, List<Tensor> inputs, Tensor output, int groups,
            List<Long> strides, List<Long> dilations, List<Long> paddingLeft,
            List<Long> paddingRight, Bias bias, Activation activation) implements Atomic {
        public Contraction {
            Objects.requireNonNull(operation, "operation");
            inputs = List.copyOf(inputs);
            Objects.requireNonNull(output, "output");
            strides = List.copyOf(strides);
            dilations = List.copyOf(dilations);
            paddingLeft = List.copyOf(paddingLeft);
            paddingRight = List.copyOf(paddingRight);
            Objects.requireNonNull(bias, "bias");
            Objects.requireNonNull(activation, "activation");
            if (groups <= 0 || inputs.size() != (bias == Bias.NONE ? 2 : 3)) {
                throw new IllegalArgumentException("oneDNN contraction input roles disagree");
            }
            if (operation == Operation.MATMUL) {
                if (groups != 1 || !strides.isEmpty() || !dilations.isEmpty()
                        || !paddingLeft.isEmpty() || !paddingRight.isEmpty() || bias == Bias.INTRINSIC) {
                    throw new IllegalArgumentException("oneDNN MATMUL attributes disagree");
                }
            } else {
                spatial(output, strides, dilations, paddingLeft, paddingRight);
            }
        }
    }

    /**
     * A separately materialized elementary arithmetic site.
     * @param algorithm non-null named elementary function
     * @param input non-null operand, possibly a same-span broadcast view
     * @param output non-null same-shape result
     */
    record Unary(UnaryAlgorithm algorithm, Tensor input, Tensor output) implements Atomic {
        public Unary {
            Objects.requireNonNull(algorithm, "algorithm");
            sameShape(input, output);
        }
        @Override public List<Tensor> inputs() { return List.of(input); }
    }
    /**
     * One separately materialized arithmetic or stored-value predicate site.
     * @param algorithm non-null arithmetic or predicate kind
     * @param left non-null left operand
     * @param right non-null right operand
     * @param output non-null exact NumPy-broadcast result geometry
     */
    record Binary(BinaryAlgorithm algorithm, Tensor left, Tensor right, Tensor output) implements Atomic {
        public Binary {
            Objects.requireNonNull(algorithm, "algorithm");
            Objects.requireNonNull(left, "left");
            Objects.requireNonNull(right, "right");
            Objects.requireNonNull(output, "output");
            broadcast(left, right, output);
        }
        @Override public List<Tensor> inputs() { return List.of(left, right); }
    }
    /**
     * An all-contributor reduction, never an implicit reciprocal-multiply mean.
     * @param algorithm non-null aggregate kind
     * @param input non-null contributor geometry
     * @param output non-null geometry retaining reduced axes as extent one
     */
    record Reduction(ReductionAlgorithm algorithm, Tensor input, Tensor output) implements Atomic {
        public Reduction {
            Objects.requireNonNull(algorithm, "algorithm");
            Objects.requireNonNull(input, "input");
            Objects.requireNonNull(output, "output");
            if (input.dimensions().size() != output.dimensions().size()) {
                throw new IllegalArgumentException("reduction descriptor must retain singleton axes");
            }
            for (int axis = 0; axis < input.dimensions().size(); axis++) {
                long extent = output.dimensions().get(axis);
                if (extent != 1 && extent != input.dimensions().get(axis)) {
                    throw new IllegalArgumentException("reduction dimensions disagree");
                }
            }
        }
        @Override public List<Tensor> inputs() { return List.of(input); }
    }
    /**
     * Raw coordinate-preserving physical movement, not a cast or promotion.
     * @param input non-null original F32 coordinates
     * @param output non-null same coordinates in another physical layout
     */
    record Reorder(Tensor input, Tensor output) implements Atomic {
        public Reorder { sameShape(input, output); }
        @Override public List<Tensor> inputs() { return List.of(input); }
    }
    /**
     * Average-include-padding descriptor; qualification separately checks every window and divisor.
     * @param input non-null NCX operand
     * @param output non-null NCX result
     * @param kernel positive spatial window, copied
     * @param strides positive spatial steps, copied
     * @param dilations positive spatial factors, copied
     * @param paddingLeft nonnegative leading padding, copied
     * @param paddingRight nonnegative trailing padding, copied
     */
    record Pooling(Tensor input, Tensor output, List<Long> kernel, List<Long> strides,
            List<Long> dilations, List<Long> paddingLeft, List<Long> paddingRight) implements Atomic {
        public Pooling {
            Objects.requireNonNull(input, "input");
            Objects.requireNonNull(output, "output");
            kernel = List.copyOf(kernel);
            strides = List.copyOf(strides);
            dilations = List.copyOf(dilations);
            paddingLeft = List.copyOf(paddingLeft);
            paddingRight = List.copyOf(paddingRight);
            spatial(output, strides, dilations, paddingLeft, paddingRight);
            if (input.dimensions().size() != output.dimensions().size()
                    || kernel.size() != strides.size() || kernel.stream().anyMatch(value -> value <= 0)
                    || !input.dimensions().subList(0, 2).equals(output.dimensions().subList(0, 2))) {
                throw new IllegalArgumentException("pooling geometry disagrees");
            }
        }
        @Override public List<Tensor> inputs() { return List.of(input); }
    }
    /**
     * One represented compile-time literal; no live input/weight snapshot is permitted.
     * @param tensor non-null scalar F32 geometry
     * @param rawWords one represented literal word, copied
     */
    record Constant(Tensor tensor, List<Integer> rawWords) {
        public Constant {
            Objects.requireNonNull(tensor, "tensor");
            rawWords = List.copyOf(rawWords);
            if (tensor.dataType() != DataType.FLOAT32 || tensor.byteSize() != Float.BYTES
                    || rawWords.size() != 1 || tensor.dimensions().stream().anyMatch(value -> value != 1)) {
                throw new IllegalArgumentException("only scalar F32 native literals are supported");
            }
        }
    }
    /**
     * One atomic ABI operation with fixed, descriptor-order slot positions.
     * @param spec non-null atomic recipe, never a pipeline
     * @param inputSlots descriptor-order readable slots, copied
     * @param outputSlot nonnegative writable result slot
     */
    record Step(CpuOneDnnKernelSpec spec, List<Integer> inputSlots, int outputSlot) {
        public Step {
            Objects.requireNonNull(spec, "spec");
            inputSlots = List.copyOf(inputSlots);
            if (!(spec instanceof Atomic) || inputSlots.size() != spec.inputs().size() || outputSlot < 0
                    || inputSlots.stream().anyMatch(slot -> slot < 0)) {
                throw new IllegalArgumentException("invalid native pipeline step");
            }
        }
    }
    /**
     * Cold-fixed ABI sequence preserving every literal store of one existing common unit.
     * Slots are inputs, final output, intermediates, constants, in that order. A view may reshape
     * a slot only while preserving dtype and exact byte span. Each writable slot has one writer.
     * @param inputs borrowed external slot geometries, copied
     * @param output non-null final result slot
     * @param intermediates owned temporary slot geometries, copied
     * @param constants immutable literal slots, copied
     * @param steps ordered atomic operations; no read-before-write or external-input writes
     */
    record Pipeline(List<Tensor> inputs, Tensor output, List<Tensor> intermediates,
            List<Constant> constants, List<Step> steps) implements CpuOneDnnKernelSpec {
        public Pipeline {
            inputs = List.copyOf(inputs);
            Objects.requireNonNull(output, "output");
            intermediates = List.copyOf(intermediates);
            constants = List.copyOf(constants);
            steps = List.copyOf(steps);
            var slots = new ArrayList<>(inputs);
            slots.add(output);
            slots.addAll(intermediates);
            for (Constant constant : constants) slots.add(constant.tensor());
            boolean[] written = new boolean[slots.size()];
            for (int slot = 0; slot < inputs.size(); slot++) written[slot] = true;
            int constantStart = inputs.size() + 1 + intermediates.size();
            for (int slot = constantStart; slot < slots.size(); slot++) written[slot] = true;
            for (Step step : steps) {
                for (int operand = 0; operand < step.inputSlots().size(); operand++) {
                    int slot = step.inputSlots().get(operand);
                    if (slot >= slots.size() || !written[slot]) {
                        throw new IllegalArgumentException("native pipeline reads an unwritten slot");
                    }
                    view(slots.get(slot), step.spec().inputs().get(operand));
                }
                int slot = step.outputSlot();
                if (slot < inputs.size() || slot >= constantStart || written[slot]) {
                    throw new IllegalArgumentException("native pipeline slot has an illegal writer");
                }
                view(slots.get(slot), step.spec().output());
                written[slot] = true;
            }
            for (int slot = inputs.size(); slot < constantStart; slot++) {
                if (!written[slot]) throw new IllegalArgumentException("native pipeline leaves an unwritten result");
            }
        }
    }

    private static long spanBytes(DataType type, List<Long> dimensions, List<Long> strides) {
        long span = 1;
        for (int axis = 0; axis < dimensions.size(); axis++) {
            span = Math.addExact(span, Math.multiplyExact(dimensions.get(axis) - 1, strides.get(axis)));
        }
        return Math.multiplyExact(span, type.byteWidth());
    }
    private static void view(Tensor slot, Tensor descriptor) {
        if (slot.dataType() != descriptor.dataType() || slot.byteSize() != descriptor.byteSize()) {
            throw new IllegalArgumentException("native pipeline view changes dtype or referenced byte span");
        }
    }
    private static void sameShape(Tensor input, Tensor output) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(output, "output");
        if (!input.dimensions().equals(output.dimensions())) throw new IllegalArgumentException("tensor shapes disagree");
    }
    private static void broadcast(Tensor left, Tensor right, Tensor output) {
        int rank = output.dimensions().size();
        if (left.dimensions().size() > rank || right.dimensions().size() > rank) {
            throw new IllegalArgumentException("broadcast input rank exceeds output");
        }
        for (int axis = 0; axis < rank; axis++) {
            int a = axis - rank + left.dimensions().size(), b = axis - rank + right.dimensions().size();
            long x = a < 0 ? 1 : left.dimensions().get(a), y = b < 0 ? 1 : right.dimensions().get(b);
            long z = output.dimensions().get(axis);
            if ((x != 1 && x != z) || (y != 1 && y != z) || Math.max(x, y) != z) {
                throw new IllegalArgumentException("broadcast dimensions disagree");
            }
        }
    }
    private static void spatial(Tensor output, List<Long> strides, List<Long> dilations,
            List<Long> left, List<Long> right) {
        int spatial = output.dimensions().size() - 2;
        if ((spatial != 2 && spatial != 3) || strides.size() != spatial || dilations.size() != spatial
                || left.size() != spatial || right.size() != spatial
                || strides.stream().anyMatch(value -> value <= 0) || dilations.stream().anyMatch(value -> value <= 0)
                || left.stream().anyMatch(value -> value < 0) || right.stream().anyMatch(value -> value < 0)) {
            throw new IllegalArgumentException("invalid native spatial geometry");
        }
    }
}

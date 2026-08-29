package io.github.pho001.synaptik.backend.cpu.internal.ir;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ClampRangeAttrs;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Immutable code-shaping identity for one portable MATMUL body.
 *
 * <p>Concrete extents, offsets, strides, batch count, tile count, worker count, and range bounds
 * are invocation geometry. This identity retains only facts that can change generated code, so
 * compatible Shapes share one generated class.</p>
 *
 * @param leftType represented left operand type
 * @param rightType represented right operand type
 * @param resultType exact Model promotion result and accumulator width
 * @param realization one of the four bounded full-K loop forms
 * @param epilogue exact bounded suffix emitted after full-K accumulation
 * @param preferredSpeciesBitSize preferred species bits, or zero for scalar forms
 * @param numericalForm exact generated/direct floating multiply-add decision
 * @param inputAccesses exact read access plans in carrier order
 * @param outputAccess exact result write access plan
 */
public record CpuMatmulIr(DataType leftType, DataType rightType, DataType resultType,
        Realization realization, Epilogue epilogue,
        int preferredSpeciesBitSize, NumericalForm numericalForm,
        List<CpuAccessPlan> inputAccesses,
        CpuAccessPlan outputAccess) {

    /** Closed portable realization vocabulary. */
    public enum Realization {
        /** One output accumulator and one full-K traversal. */ DIRECT_SCALAR,
        /** One M row with preferred-species N accumulators and a scalar tail. */ DIRECT_N_VECTOR,
        /** Fixed two-by-two scalar M/N microtiles and explicit tails. */ TILED_SCALAR_2X2,
        /** Fixed two-row by two-species N microtiles and scalar tails. */ TILED_N_VECTOR_2X2
    }

    /** Exact floating multiply-add decision shared with the direct oracle. */
    public enum NumericalForm { /** Separate multiply then add. */ SEQUENTIAL,
        /** Explicit fused multiply-add. */ FUSED_MULTIPLY_ADD }

    /**
     * Exact external ADD order and optional terminal suffix.
     *
     * @param addInputOrder preserved ordered ADD relationship, or {@code NONE}
     * @param terminal optional exact terminal operation
     * @param clampRange exact typed bounds only when {@code terminal} is {@code CLAMP}
     */
    public record Epilogue(AddInputOrder addInputOrder, Terminal terminal,
            ClampRangeAttrs clampRange) {
        /** Position of the MATMUL value in the preserved ordered ADD. */
        public enum AddInputOrder { /** No bias. */ NONE, /** MATMUL is left. */ MATMUL_LEFT,
            /** MATMUL is right. */ MATMUL_RIGHT }
        /** Closed CPU 0008C terminal vocabulary. */
        public enum Terminal { /** No terminal. */ NONE, /** ReLU. */ RELU,
            /** Sigmoid. */ SIGMOID, /** Hyperbolic tangent. */ TANH, /** Exact GELU. */ GELU,
            /** Tanh-approximation GELU. */ GELU_TANH_APPROXIMATION, /** SiLU. */ SILU,
            /** Typed clamp. */ CLAMP }
        /** Validates exact bias/terminal facts. */
        public Epilogue {
            Objects.requireNonNull(addInputOrder, "addInputOrder");
            Objects.requireNonNull(terminal, "terminal");
            if ((terminal == Terminal.CLAMP) != (clampRange != null))
                throw new IllegalArgumentException("MATMUL epilogue facts disagree");
        }
        /** Creates the no-bias, no-terminal suffix.
         * @return an empty non-null epilogue */
        public static Epilogue none() {
            return new Epilogue(AddInputOrder.NONE, Terminal.NONE, null);
        }
        /** Reports whether an exact external bias participates.
         * @return whether this suffix contains an external bias ADD */
        public boolean hasBias() { return addInputOrder != AddInputOrder.NONE; }
        /** Reports whether an exact terminal follows the ADD.
         * @return whether this suffix contains a terminal operation */
        public boolean hasTerminal() { return terminal != Terminal.NONE; }
    }

    /** Validates one bounded, Shape-polymorphic MATMUL class identity. */
    public CpuMatmulIr {
        Objects.requireNonNull(leftType, "leftType");
        Objects.requireNonNull(rightType, "rightType");
        Objects.requireNonNull(resultType, "resultType");
        Objects.requireNonNull(realization, "realization");
        Objects.requireNonNull(epilogue, "epilogue");
        Objects.requireNonNull(numericalForm, "numericalForm");
        inputAccesses = List.copyOf(inputAccesses);
        Objects.requireNonNull(outputAccess, "outputAccess");
        DataType promoted;
        try { promoted = DataTypePromotion.promoteNumeric(leftType, rightType); }
        catch (IllegalArgumentException exception) {
            throw new IllegalArgumentException("MATMUL requires one numeric category", exception);
        }
        int expectedInputs = epilogue.hasBias() ? 3 : 2;
        boolean vector = realization == Realization.DIRECT_N_VECTOR
                || realization == Realization.TILED_N_VECTOR_2X2;
        boolean sameVectorType = leftType == rightType && rightType == resultType
                && (resultType == DataType.FLOAT32 || resultType == DataType.FLOAT64
                    || resultType == DataType.INT32 || resultType == DataType.INT64);
        if (promoted != resultType || inputAccesses.size() != expectedInputs
                || inputAccesses.stream().anyMatch(access ->
                    access.accessKind() != CpuAccessPlan.AccessKind.READ)
                || outputAccess.accessKind() != CpuAccessPlan.AccessKind.WRITE
                || preferredSpeciesBitSize < 0
                || vector != (preferredSpeciesBitSize > 0)
                || numericalForm == NumericalForm.FUSED_MULTIPLY_ADD && !resultType.isFloating()
                || epilogue.hasBias()
                    && resultType != DataType.FLOAT32 && resultType != DataType.FLOAT64
                || epilogue.clampRange() != null
                    && epilogue.clampRange().minValue().dataType() != resultType
                || vector && (!sameVectorType || epilogue.hasTerminal())) {
            throw new IllegalArgumentException("MATMUL IR facts disagree");
        }
    }

    /**
     * Encodes the family facts into the existing generated-artifact identity seam.
     *
     * @return a fresh canonical instruction-free kernel identity
     */
    public CpuKernelIr encodedKernelIr() {
        var values = new ArrayList<CpuKernelIr.Value>();
        for (int i = 0; i < inputAccesses.size(); i++) {
            DataType type = i == 0 ? leftType : i == 1 ? rightType : resultType;
            values.add(new CpuKernelIr.Value(i, type, CpuKernelIr.Value.Kind.INPUT,
                    inputAccesses.get(i)));
        }
        values.add(new CpuKernelIr.Value(values.size(), resultType,
                CpuKernelIr.Value.Kind.OUTPUT, outputAccess));
        String family = "matmul:realization=" + realization + ":left=" + leftType
                + ":right=" + rightType + ":result=" + resultType + ":epilogue=" + epilogue
                + ":species=" + preferredSpeciesBitSize + ":numerical=" + numericalForm;
        return new CpuKernelIr(values, List.of(), new CpuKernelIr.Loop("start", "end"),
                List.of(new CpuKernelIr.Store(values.size() - 1, 0)), family);
    }

    /**
     * Returns the same code shape with one exact represented input access transition.
     *
     * @param boundaryPosition zero-based left, right, or bias input position
     * @param access non-null replacement read access selected by CPU 0008E
     * @return a new immutable MATMUL identity
     * @throws IndexOutOfBoundsException if the position is not an input
     * @throws IllegalArgumentException if {@code access} is not a read
     */
    public CpuMatmulIr withInputAccess(int boundaryPosition, CpuAccessPlan access) {
        Objects.requireNonNull(access, "access");
        if (boundaryPosition < 0 || boundaryPosition >= inputAccesses.size())
            throw new IndexOutOfBoundsException("MATMUL input boundary is invalid");
        var adjusted = new ArrayList<>(inputAccesses);
        adjusted.set(boundaryPosition, access);
        return new CpuMatmulIr(leftType, rightType, resultType, realization, epilogue,
                preferredSpeciesBitSize, numericalForm, adjusted, outputAccess);
    }

    /** Returns the deterministic encoded structural key.
     * @return non-null lowercase hexadecimal structural key
     */
    public String structuralKey() { return encodedKernelIr().structuralKey(); }
}

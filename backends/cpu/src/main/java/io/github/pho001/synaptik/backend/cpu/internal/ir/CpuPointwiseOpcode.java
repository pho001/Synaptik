package io.github.pho001.synaptik.backend.cpu.internal.ir;

/**
 * CPU-private, family-oriented pointwise opcode vocabulary with exactly forty-eight opcodes.
 *
 * <p>Each opcode is selected once while lowering a Model operation occurrence. Generated and
 * reference execution consume this typed value and never inspect Model operations or strings.</p>
 */
public enum CpuPointwiseOpcode {
    /** Same-type binary addition. */
    ADD(Family.BINARY_ARITHMETIC, 2, ResultCategory.INPUT_TYPE, false, VectorForm.VALUE),
    /** Same-type binary subtraction. */
    SUB(Family.BINARY_ARITHMETIC, 2, ResultCategory.INPUT_TYPE, false, VectorForm.VALUE),
    /** Same-type binary multiplication. */
    MUL(Family.BINARY_ARITHMETIC, 2, ResultCategory.INPUT_TYPE, false, VectorForm.VALUE),
    /** Same-type floating binary division in left-divided-by-right order. */
    DIV(Family.BINARY_ARITHMETIC, 2, ResultCategory.INPUT_TYPE, false, VectorForm.VALUE),
    /** Same-type represented-value minimum. */
    MIN(Family.BINARY_ARITHMETIC, 2, ResultCategory.INPUT_TYPE, false, VectorForm.VALUE),
    /** Same-type represented-value maximum. */
    MAX(Family.BINARY_ARITHMETIC, 2, ResultCategory.INPUT_TYPE, false, VectorForm.VALUE),
    /** Same-type floating Tensor base and exponent power. */
    POW(Family.BINARY_ARITHMETIC, 2, ResultCategory.INPUT_TYPE, false, VectorForm.NONE),
    /** Addition of one exact typed scalar immediate. */
    SCALAR_ADD(Family.SCALAR_ARITHMETIC, 1, ResultCategory.INPUT_TYPE, true, VectorForm.VALUE),
    /** Subtraction of one exact typed scalar immediate. */
    SCALAR_SUB(Family.SCALAR_ARITHMETIC, 1, ResultCategory.INPUT_TYPE, true, VectorForm.VALUE),
    /** Multiplication by one exact typed scalar immediate. */
    SCALAR_MUL(Family.SCALAR_ARITHMETIC, 1, ResultCategory.INPUT_TYPE, true, VectorForm.VALUE),
    /** Division by one exact same-typed floating scalar immediate. */
    SCALAR_DIV(Family.SCALAR_ARITHMETIC, 1, ResultCategory.INPUT_TYPE, true, VectorForm.VALUE),
    /** Scalar power with one exact exponent and one selected realization fact. */
    SCALAR_POW(Family.SCALAR_ARITHMETIC, 1, ResultCategory.INPUT_TYPE, true, VectorForm.VALUE),
    /** Minimum with one exact same-typed scalar candidate. */
    SCALAR_MIN(Family.SCALAR_ARITHMETIC, 1, ResultCategory.INPUT_TYPE, true, VectorForm.VALUE),
    /** Maximum with one exact same-typed scalar candidate. */
    SCALAR_MAX(Family.SCALAR_ARITHMETIC, 1, ResultCategory.INPUT_TYPE, true, VectorForm.VALUE),
    /** One first-class floating range clamp with two exact bounds. */
    SCALAR_CLAMP(Family.SCALAR_RANGE, 1, ResultCategory.INPUT_TYPE, false, VectorForm.VALUE),
    /** Floating primitive negation. */
    NEG(Family.UNARY, 1, ResultCategory.INPUT_TYPE, false, VectorForm.VALUE),
    /** Floating absolute magnitude. */
    ABS(Family.UNARY, 1, ResultCategory.INPUT_TYPE, false, VectorForm.VALUE),
    /** Floating reciprocal. */
    RECIPROCAL(Family.UNARY, 1, ResultCategory.INPUT_TYPE, false, VectorForm.VALUE),
    /** Floating natural logarithm. */
    LOG(Family.UNARY, 1, ResultCategory.INPUT_TYPE, false, VectorForm.VALUE),
    /** Floating natural logarithm of one plus the input. */
    LOG1P(Family.UNARY, 1, ResultCategory.INPUT_TYPE, false, VectorForm.VALUE),
    /** Floating natural exponential. */
    EXP(Family.UNARY, 1, ResultCategory.INPUT_TYPE, false, VectorForm.VALUE),
    /** Floating natural exponential minus one. */
    EXPM1(Family.UNARY, 1, ResultCategory.INPUT_TYPE, false, VectorForm.VALUE),
    /** Floating Gaussian error function. */
    ERF(Family.UNARY, 1, ResultCategory.INPUT_TYPE, false, VectorForm.VALUE),
    /** Floating principal square root. */
    SQRT(Family.UNARY, 1, ResultCategory.INPUT_TYPE, false, VectorForm.VALUE),
    /** Floating reciprocal principal square root. */
    RSQRT(Family.UNARY, 1, ResultCategory.INPUT_TYPE, false, VectorForm.VALUE),
    /** Floating floor. */
    FLOOR(Family.UNARY, 1, ResultCategory.INPUT_TYPE, false, VectorForm.NONE),
    /** Floating ceiling. */
    CEIL(Family.UNARY, 1, ResultCategory.INPUT_TYPE, false, VectorForm.NONE),
    /** Floating sign classification as a floating result. */
    SIGN(Family.UNARY, 1, ResultCategory.INPUT_TYPE, false, VectorForm.VALUE),
    /** Rectified linear unit. */
    RELU(Family.UNARY, 1, ResultCategory.INPUT_TYPE, false, VectorForm.VALUE),
    /** Stable logistic sigmoid. */
    SIGMOID(Family.UNARY, 1, ResultCategory.INPUT_TYPE, false, VectorForm.NONE),
    /** Floating hyperbolic tangent. */
    TANH(Family.UNARY, 1, ResultCategory.INPUT_TYPE, false, VectorForm.VALUE),
    /** Exact/default floating Gaussian error linear unit. */
    GELU_EXACT(Family.UNARY, 1, ResultCategory.INPUT_TYPE, false, VectorForm.VALUE),
    /** Fixed hyperbolic-tangent GELU approximation. */
    GELU_TANH_APPROXIMATION(Family.UNARY, 1, ResultCategory.INPUT_TYPE, false, VectorForm.NONE),
    /** Stable sigmoid linear unit. */
    SILU(Family.UNARY, 1, ResultCategory.INPUT_TYPE, false, VectorForm.NONE),
    /** Floating finite-value classification. */
    IS_FINITE(Family.CLASSIFICATION, 1, ResultCategory.BOOL, false, VectorForm.MASK_PRODUCER),
    /** Floating not-a-number classification. */
    IS_NAN(Family.CLASSIFICATION, 1, ResultCategory.BOOL, false, VectorForm.MASK_PRODUCER),
    /** Floating infinity classification. */
    IS_INF(Family.CLASSIFICATION, 1, ResultCategory.BOOL, false, VectorForm.MASK_PRODUCER),
    /** Same-type numeric greater-than comparison. */
    GREATER_THAN(Family.COMPARISON, 2, ResultCategory.BOOL, false, VectorForm.MASK_PRODUCER),
    /** Same-type numeric greater-than-or-equal comparison. */
    GREATER_OR_EQUAL(Family.COMPARISON, 2, ResultCategory.BOOL, false, VectorForm.MASK_PRODUCER),
    /** Same-type numeric less-than comparison. */
    LESS_THAN(Family.COMPARISON, 2, ResultCategory.BOOL, false, VectorForm.MASK_PRODUCER),
    /** Same-type numeric less-than-or-equal comparison. */
    LESS_OR_EQUAL(Family.COMPARISON, 2, ResultCategory.BOOL, false, VectorForm.MASK_PRODUCER),
    /** Same-type numeric equality comparison. */
    EQUAL(Family.COMPARISON, 2, ResultCategory.BOOL, false, VectorForm.MASK_PRODUCER),
    /** Same-type numeric inequality comparison. */
    NOT_EQUAL(Family.COMPARISON, 2, ResultCategory.BOOL, false, VectorForm.MASK_PRODUCER),
    /** Canonical-BOOL conjunction. */
    LOGICAL_AND(Family.LOGICAL, 2, ResultCategory.BOOL, false, VectorForm.VALUE_OR_MASK),
    /** Canonical-BOOL disjunction. */
    LOGICAL_OR(Family.LOGICAL, 2, ResultCategory.BOOL, false, VectorForm.VALUE_OR_MASK),
    /** Canonical-BOOL complement. */
    LOGICAL_NOT(Family.LOGICAL, 1, ResultCategory.BOOL, false, VectorForm.VALUE_OR_MASK),
    /** Canonical-BOOL selection between same-type floating branches. */
    WHERE(Family.SELECTION, 3, ResultCategory.BRANCH_TYPE, false, VectorForm.MASK_CONSUMER),
    /** Represented-value-preserving same-type explicit cast. */
    CAST(Family.CAST, 1, ResultCategory.INPUT_TYPE, false, VectorForm.VALUE);

    /** Semantic family used for grouped validation and emission. */
    public enum Family {
        /** Two-value same-type arithmetic. */ BINARY_ARITHMETIC,
        /** One value plus an exact typed scalar immediate. */ SCALAR_ARITHMETIC,
        /** One value plus two exact ordered scalar bounds. */ SCALAR_RANGE,
        /** One-value floating computation. */ UNARY,
        /** Floating predicate producing BOOL. */ CLASSIFICATION,
        /** Two-value numeric predicate producing BOOL. */ COMPARISON,
        /** Canonical-BOOL logic producing canonical BOOL. */ LOGICAL,
        /** BOOL-conditioned branch selection. */ SELECTION,
        /** Same-type represented-value identity. */ CAST
    }

    /** Rule used to validate an instruction result type. */
    public enum ResultCategory {
        /** Result has the exact primary input type. */ INPUT_TYPE,
        /** Result has canonical BOOL type. */ BOOL,
        /** Result has the exact common branch type. */ BRANCH_TYPE
    }

    /** Generated vector representation required by an opcode. */
    public enum VectorForm {
        /** No vector realization is currently admitted. */ NONE,
        /** Consume and produce ordinary typed value vectors. */ VALUE,
        /** Consume and produce either canonical byte vectors or virtual typed masks. */ VALUE_OR_MASK,
        /** Produce a virtual typed mask from floating value vectors. */ MASK_PRODUCER,
        /** Consume a virtual or scalar-broadcast mask to select floating value vectors. */ MASK_CONSUMER
    }

    private final Family family;
    private final int arity;
    private final ResultCategory resultCategory;
    private final boolean scalarImmediate;
    private final VectorForm vectorForm;

    CpuPointwiseOpcode(Family family, int arity, ResultCategory resultCategory,
            boolean scalarImmediate, VectorForm vectorForm) {
        this.family = family;
        this.arity = arity;
        this.resultCategory = resultCategory;
        this.scalarImmediate = scalarImmediate;
        this.vectorForm = vectorForm;
    }

    /**
     * Returns the group that owns this opcode's type validation and emission.
     *
     * @return the non-null semantic family
     */
    public Family family() { return family; }

    /**
     * Returns the number of topology-local value inputs consumed by one instruction.
     *
     * @return the exact positive input arity
     */
    public int arity() { return arity; }

    /**
     * Returns the rule used to relate input and output data types.
     *
     * @return the non-null result-type rule
     */
    public ResultCategory resultCategory() { return resultCategory; }

    /**
     * Reports whether one exact typed scalar immediate is part of the instruction.
     *
     * @return {@code true} for scalar arithmetic and {@code false} otherwise
     */
    public boolean carriesScalarImmediate() { return scalarImmediate; }

    /**
     * Reports opcode eligibility for some current exact typed vector form.
     *
     * <p>This flag is necessary but not sufficient: analysis also validates lane data type,
     * immediate realization, BOOL boundary or virtual-mask role, and every external access plan's
     * vector-run eligibility.</p>
     *
     * @return {@code true} only for an opcode the current typed Vector emitter can realize
     */
    public boolean vectorEligible() { return vectorForm != VectorForm.NONE; }

    /**
     * Returns the generated representation role required by this opcode.
     *
     * @return the non-null closed vector form
     */
    public VectorForm vectorForm() { return vectorForm; }
}

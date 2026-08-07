package io.github.pho001.synaptik.backend.cpu.internal.ir;

/**
 * CPU-private, family-oriented pointwise opcode vocabulary with exactly thirty-one opcodes.
 *
 * <p>Each opcode is selected once while lowering a Model operation occurrence. Generated and
 * reference execution consume this typed value and never inspect Model operations or strings.</p>
 */
public enum CpuPointwiseOpcode {
    /** Same-type binary addition. */
    ADD(Family.BINARY_ARITHMETIC, 2, ResultCategory.INPUT_TYPE, false, true),
    /** Same-type binary subtraction. */
    SUB(Family.BINARY_ARITHMETIC, 2, ResultCategory.INPUT_TYPE, false, true),
    /** Same-type binary multiplication. */
    MUL(Family.BINARY_ARITHMETIC, 2, ResultCategory.INPUT_TYPE, false, true),
    /** Same-type floating binary division in left-divided-by-right order. */
    DIV(Family.BINARY_ARITHMETIC, 2, ResultCategory.INPUT_TYPE, false, true),
    /** Same-type represented-value minimum. */
    MIN(Family.BINARY_ARITHMETIC, 2, ResultCategory.INPUT_TYPE, false, false),
    /** Same-type represented-value maximum. */
    MAX(Family.BINARY_ARITHMETIC, 2, ResultCategory.INPUT_TYPE, false, false),
    /** Same-type floating Tensor base and exponent power. */
    POW(Family.BINARY_ARITHMETIC, 2, ResultCategory.INPUT_TYPE, false, false),
    /** Addition of one exact typed scalar immediate. */
    SCALAR_ADD(Family.SCALAR_ARITHMETIC, 1, ResultCategory.INPUT_TYPE, true, true),
    /** Subtraction of one exact typed scalar immediate. */
    SCALAR_SUB(Family.SCALAR_ARITHMETIC, 1, ResultCategory.INPUT_TYPE, true, true),
    /** Multiplication by one exact typed scalar immediate. */
    SCALAR_MUL(Family.SCALAR_ARITHMETIC, 1, ResultCategory.INPUT_TYPE, true, true),
    /** Division by one exact same-typed floating scalar immediate. */
    SCALAR_DIV(Family.SCALAR_ARITHMETIC, 1, ResultCategory.INPUT_TYPE, true, true),
    /** Scalar power with one exact exponent and one selected realization fact. */
    SCALAR_POW(Family.SCALAR_ARITHMETIC, 1, ResultCategory.INPUT_TYPE, true, true),
    /** Minimum with one exact same-typed scalar candidate. */
    SCALAR_MIN(Family.SCALAR_ARITHMETIC, 1, ResultCategory.INPUT_TYPE, true, false),
    /** Maximum with one exact same-typed scalar candidate. */
    SCALAR_MAX(Family.SCALAR_ARITHMETIC, 1, ResultCategory.INPUT_TYPE, true, false),
    /** One first-class floating range clamp with two exact bounds. */
    SCALAR_CLAMP(Family.SCALAR_RANGE, 1, ResultCategory.INPUT_TYPE, false, false),
    /** Floating primitive negation. */
    NEG(Family.UNARY, 1, ResultCategory.INPUT_TYPE, false, true),
    /** Exact/default FLOAT64 Gaussian error linear unit. */
    GELU_EXACT(Family.UNARY, 1, ResultCategory.INPUT_TYPE, false, true),
    /** Floating finite-value classification. */
    IS_FINITE(Family.CLASSIFICATION, 1, ResultCategory.BOOL, false, false),
    /** Floating not-a-number classification. */
    IS_NAN(Family.CLASSIFICATION, 1, ResultCategory.BOOL, false, false),
    /** Floating infinity classification. */
    IS_INF(Family.CLASSIFICATION, 1, ResultCategory.BOOL, false, false),
    /** Same-type numeric greater-than comparison. */
    GREATER_THAN(Family.COMPARISON, 2, ResultCategory.BOOL, false, false),
    /** Same-type numeric greater-than-or-equal comparison. */
    GREATER_OR_EQUAL(Family.COMPARISON, 2, ResultCategory.BOOL, false, false),
    /** Same-type numeric less-than comparison. */
    LESS_THAN(Family.COMPARISON, 2, ResultCategory.BOOL, false, false),
    /** Same-type numeric less-than-or-equal comparison. */
    LESS_OR_EQUAL(Family.COMPARISON, 2, ResultCategory.BOOL, false, false),
    /** Same-type numeric equality comparison. */
    EQUAL(Family.COMPARISON, 2, ResultCategory.BOOL, false, false),
    /** Same-type numeric inequality comparison. */
    NOT_EQUAL(Family.COMPARISON, 2, ResultCategory.BOOL, false, false),
    /** Canonical-BOOL conjunction. */
    LOGICAL_AND(Family.LOGICAL, 2, ResultCategory.BOOL, false, false),
    /** Canonical-BOOL disjunction. */
    LOGICAL_OR(Family.LOGICAL, 2, ResultCategory.BOOL, false, false),
    /** Canonical-BOOL complement. */
    LOGICAL_NOT(Family.LOGICAL, 1, ResultCategory.BOOL, false, false),
    /** Canonical-BOOL selection between same-type floating branches. */
    WHERE(Family.SELECTION, 3, ResultCategory.BRANCH_TYPE, false, false),
    /** Represented-value-preserving same-type explicit cast. */
    CAST(Family.CAST, 1, ResultCategory.INPUT_TYPE, false, false);

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

    private final Family family;
    private final int arity;
    private final ResultCategory resultCategory;
    private final boolean scalarImmediate;
    private final boolean vectorEligible;

    CpuPointwiseOpcode(Family family, int arity, ResultCategory resultCategory,
            boolean scalarImmediate, boolean vectorEligible) {
        this.family = family;
        this.arity = arity;
        this.resultCategory = resultCategory;
        this.scalarImmediate = scalarImmediate;
        this.vectorEligible = vectorEligible;
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
     * Reports eligibility for the current exact FLOAT64 Vector emitter.
     *
     * <p>This flag is necessary but not sufficient: analysis also requires every IR value to be
     * {@code FLOAT64} and every external access plan to satisfy the vector-run rules.</p>
     *
     * @return {@code true} only for an opcode the current FLOAT64 Vector emitter can realize
     */
    public boolean vectorEligible() { return vectorEligible; }
}

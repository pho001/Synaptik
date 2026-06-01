package graph.optimizer.rewrite.algebraic;

/**
 * System-property switches for isolating algebraic rewrite transforms.
 */
final class AlgebraicRewriteSwitches {
    static final boolean DISABLE_ALL_TRANSFORMS =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableAllTransforms", "false"));
    static final boolean DISABLE_REBUILD_TOPO_CLOSURE =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableRebuildTopologicalClosure", "false"));
    static final boolean DISABLE_POW2_TO_MUL =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disablePow2ToMul", "false"));
    static final boolean DISABLE_POW_NEG2_TO_MUL_INV =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disablePowNeg2ToMulInv", "false"));
    static final boolean DISABLE_ADD_SELF_TO_MUL2 =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableAddSelfToMul2", "false"));
    static final boolean DISABLE_ADD_NEG_TO_ZERO =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableAddNegToZero", "false"));
    static final boolean DISABLE_ADD_NEGNEG_TO_NEGADD =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableAddNegNegToNegAdd", "false"));
    static final boolean DISABLE_ADD_LOGLOG_TO_LOGMUL =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableAddLogLogToLogMul", "false"));
    static final boolean DISABLE_SUB_NEG_TO_ADD =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableSubNegToAdd", "false"));
    static final boolean DISABLE_DIV_CONST_TO_MULRECIP =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableDivConstToMulRecip", "false"));
    static final boolean DISABLE_DIV_MULSCALAR_BY_CONST =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableDivMulScalarByConst", "false"));
    static final boolean DISABLE_DIV_INV_TO_MUL =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableDivInvToMul", "false"));
    static final boolean DISABLE_DIV_ONE_TO_INV =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableDivOneToInv", "false"));
    static final boolean DISABLE_MULSCALAR_ASSOC =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableMulScalarAssoc", "false"));
    static final boolean DISABLE_MULSCALAR_NEG_PUSH =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableMulScalarNegPush", "false"));
    static final boolean DISABLE_MULSCALAR_CONST_FOLD =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableMulScalarConstFold", "false"));
    static final boolean DISABLE_ADD_SUB_FACTORIZE =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableAddSubFactorize", "false"));
    static final boolean DISABLE_MUL_INV_TO_ONE =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableMulInvToOne", "false"));
    static final boolean DISABLE_MUL_NEGNEG_TO_MUL =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableMulNegNegToMul", "false"));
    static final boolean DISABLE_MUL_EXPEXP_TO_EXPADD =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableMulExpExpToExpAdd", "false"));
    static final boolean DISABLE_NEG_SUB_SWAP =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableNegSubSwap", "false"));
    static final boolean DISABLE_NEG_MULSCALAR_PUSH =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableNegMulScalarPush", "false"));
    static final boolean DISABLE_POW_POW_FLATTEN =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disablePowPowFlatten", "false"));
    static final boolean DISABLE_POW_INV_TO_NEGEXP =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disablePowInvToNegExp", "false"));
    static final boolean DISABLE_LOG_POW_TO_MULLOG =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableLogPowToMulLog", "false"));
    static final boolean DISABLE_LOG_INV_TO_NEGLOG =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableLogInvToNegLog", "false"));
    static final boolean DISABLE_LOG_SQRT_TO_HALFLOG =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableLogSqrtToHalfLog", "false"));
    static final boolean DISABLE_EXP_LOG_CANCEL =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableExpLogCancel", "false"));
    static final boolean DISABLE_INV_SIGMOID_PATTERN =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableInvSigmoidPattern", "false"));
    static final boolean DISABLE_INV_POW_TO_NEGEXP =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableInvPowToNegExp", "false"));
    static final boolean DISABLE_INV_EXP_TO_EXPNEG =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableInvExpToExpNeg", "false"));
    static final boolean DISABLE_INV_NEG_PUSH =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableInvNegPush", "false"));
    static final boolean DISABLE_CLAMPMIN_IDENTITY =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableClampMinIdentity", "false"));
    static final boolean DISABLE_CLAMPMIN_FLATTEN =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableClampMinFlatten", "false"));
    static final boolean DISABLE_CLAMPMAX_IDENTITY =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableClampMaxIdentity", "false"));
    static final boolean DISABLE_CLAMPMAX_FLATTEN =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableClampMaxFlatten", "false"));

    private AlgebraicRewriteSwitches() {
    }
}

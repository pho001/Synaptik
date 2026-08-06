package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Evaluates the current exact/default FLOAT64 GELU formula for vector-specialized generated
 * entries. The helper preserves the scalar polynomial branches and coefficient order and uses
 * only the Java 26 preferred FLOAT64 species selected during CPU analysis.
 */
final class CpuVectorEmitter {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final double[] ERF_T = {9.60497373987051638749E0, 9.00260197203842689217E1,
            2.23200534594684319226E3, 7.00332514112805075473E3, 5.55923013010394962768E4};
    private static final double[] ERF_U = {3.35617141647503099647E1, 5.21357949780152679795E2,
            4.59432382970980127987E3, 2.26290000613890934246E4, 4.92673942608635921086E4};
    private static final double[] ERFC_P = {2.46196981473530512524E-10, 5.64189564831068821977E-1,
            7.46321056442269912687E0, 4.86371970985681366614E1, 1.96520832956077098242E2,
            5.26445194995477358631E2, 9.34528527171957607540E2, 1.02755188689515710272E3,
            5.57535335369399327526E2};
    private static final double[] ERFC_Q = {1.32281951154744992508E1, 8.67072140885989742329E1,
            3.54937778887819891062E2, 9.75708501743205489753E2, 1.82390916687909736289E3,
            2.24633760818710981792E3, 1.65666309194161350182E3, 5.57535340817727675546E2};
    private static final double[] ERFC_R = {5.64189583547755073984E-1, 1.27536670759978104416E0,
            5.01905042251180477414E0, 6.16021097993053585195E0, 7.40974269950448939160E0,
            2.97886665372100240670E0};
    private static final double[] ERFC_S = {2.26052863220117276590E0, 9.39603524938001434673E0,
            1.20489539808096656605E1, 1.70814450747565897222E1, 9.60896809063285878198E0,
            3.36907645100081516050E0};

    private CpuVectorEmitter() { }

    /**
     * Evaluates exact/default GELU for one preferred-species vector in fixed coefficient order.
     *
     * @param value non-null preferred-species vector of pre-activation values
     * @return a non-null preferred-species vector containing the GELU results
     */
    static DoubleVector gelu(DoubleVector value) {
        DoubleVector erfInput = value.div(Math.sqrt(2.0d));
        DoubleVector x = erfInput.abs();
        DoubleVector z = x.mul(x);
        DoubleVector small = polevl(z, ERF_T).div(p1evl(z, ERF_U)).mul(x);
        DoubleVector near = polevl(x, ERFC_P).div(p1evl(x, ERFC_Q));
        DoubleVector far = polevl(x, ERFC_R).div(p1evl(x, ERFC_S));
        DoubleVector ratio = far.blend(near, x.lt(8.0d));
        DoubleVector large = DoubleVector.broadcast(SPECIES, 1.0d)
                .sub(z.neg().lanewise(VectorOperators.EXP).mul(ratio));
        DoubleVector erf = large.blend(small, x.compare(VectorOperators.LE, 1.0d));
        erf = erf.blend(1.0d, x.eq(Double.POSITIVE_INFINITY));
        erf = erf.blend(erf.neg(), erfInput.lt(0.0d));
        return value.mul(0.5d).mul(erf.add(1.0d));
    }

    private static DoubleVector polevl(DoubleVector x, double[] coefficients) {
        DoubleVector result = DoubleVector.broadcast(SPECIES, coefficients[0]);
        for (int i = 1; i < coefficients.length; i++) result = result.mul(x).add(coefficients[i]);
        return result;
    }

    private static DoubleVector p1evl(DoubleVector x, double[] coefficients) {
        DoubleVector result = x.add(coefficients[0]);
        for (int i = 1; i < coefficients.length; i++) result = result.mul(x).add(coefficients[i]);
        return result;
    }
}

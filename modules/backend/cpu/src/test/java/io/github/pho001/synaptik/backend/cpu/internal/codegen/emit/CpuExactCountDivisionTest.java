package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.lang.classfile.ClassFile;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;
import java.lang.constant.MethodTypeDesc;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.AccessFlag;
import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import org.junit.jupiter.api.Test;

/** Numerical regression for exact-cardinality binary32 mean finalization. */
class CpuExactCountDivisionTest {
    @Test void largeExactCountsAndSubnormalTiesRoundOnlyTheQuotient() throws Throwable {
        var owner = ClassDesc.of(getClass().getPackageName() + ".ExactCountDivision");
        var descriptor = MethodTypeDesc.of(ConstantDescs.CD_float,
                ConstantDescs.CD_float, ConstantDescs.CD_long);
        byte[] bytes = ClassFile.of().build(owner, builder -> {
            builder.withVersion(ClassFile.JAVA_26_VERSION, 0).withFlags(AccessFlag.FINAL);
            builder.withMethodBody("divide", descriptor,
                    ClassFile.ACC_PUBLIC | ClassFile.ACC_STATIC, code -> {
                        CpuExactSumEmitter.emitFloatDivideByCount(code, 0, 1);
                        code.fload(0).freturn();
                    });
        });
        var lookup = MethodHandles.lookup().defineHiddenClass(bytes, true);
        var divide = lookup.findStatic(lookup.lookupClass(), "divide",
                MethodType.methodType(float.class, float.class, long.class));
        float[] numerators = {0f, -0f, 1f, -1f, 3f, Float.MIN_VALUE,
                3 * Float.MIN_VALUE, Float.MIN_NORMAL, Float.MAX_VALUE,
                Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NaN};
        long[] counts = {0, 1, 2, 3, (1L << 24) - 1, 1L << 24, (1L << 24) + 1,
                (1L << 53) + 1, Long.MAX_VALUE};
        var precision = new MathContext(256, RoundingMode.HALF_EVEN);
        for (float numerator : numerators) for (long count : counts) {
            float expected = !Float.isFinite(numerator) || numerator == 0f || count == 0
                    ? numerator / (float) count
                    : new BigDecimal((double) numerator).divide(BigDecimal.valueOf(count), precision)
                            .floatValue();
            float actual = (float) divide.invokeExact(numerator, count);
            assertEquals(Float.floatToIntBits(expected), Float.floatToIntBits(actual),
                    () -> numerator + "/" + count);
        }
        float largeCountResult = (float) divide.invokeExact(1f, (1L << 24) + 1);
        assertEquals(Float.floatToRawIntBits(Math.nextDown(0x1p-24f)),
                Float.floatToRawIntBits(largeCountResult));
    }
}

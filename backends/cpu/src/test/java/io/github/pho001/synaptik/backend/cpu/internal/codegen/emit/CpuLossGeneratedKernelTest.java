package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuLossIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.backend.cpu.internal.reference.CpuLossReferenceKernel;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.loss.DenseCategoricalCrossEntropyWithLogitsAttrs;
import io.github.pho001.synaptik.model.operation.loss.IndexCategoricalCrossEntropyWithLogitsAttrs;
import io.github.pho001.synaptik.model.operation.loss.LossKind;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.model.operation.loss.MeanSquaredErrorAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.classfile.ClassFile;
import java.lang.classfile.Instruction;
import java.lang.classfile.Opcode;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.classfile.instruction.InvokeInstruction;
import java.lang.classfile.instruction.StoreInstruction;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.AccessFlag;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/**
 * Verifies direct schema-58 generated loss bodies against the frozen typed clean-Java oracle.
 *
 * <p>The representative cases cover the three loss families and reductions, binary64/binary32/
 * BFLOAT16 accumulator domains, both index widths, shared input roles, non-contiguous cold
 * layouts, and a segment carrier. They invoke the generated artifact directly so successful
 * definition proves Class-File verification independently of later lifecycle work.</p>
 */
class CpuLossGeneratedKernelTest {
    @Test
    void contiguousIndexF32AndF64BodiesRetainTheFrozenOracleStructuralFacts() throws Exception {
        Shape logits = Shape.of(2, 32, 64);
        var f32 = plan(new Operation(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                new IndexCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.NONE,
                        Optional.of(ScalarValue.int32(-1)))), List.of(0, 1),
                List.of(desc(DataType.FLOAT32, logits), desc(DataType.INT32, Shape.of(2, 64))),
                desc(DataType.FLOAT32, Shape.of(2, 64)), List.of(CarrierAccess.FLOAT_ARRAY,
                        CarrierAccess.INT_ARRAY, CarrierAccess.FLOAT_ARRAY));
        var f64 = plan(new Operation(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                new IndexCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.NONE,
                        Optional.of(ScalarValue.int64(-1)))), List.of(0, 1),
                List.of(desc(DataType.FLOAT64, logits), desc(DataType.INT64, Shape.of(2, 64))),
                desc(DataType.FLOAT64, Shape.of(2, 64)), List.of(CarrierAccess.DOUBLE_ARRAY,
                        CarrierAccess.LONG_ARRAY, CarrierAccess.DOUBLE_ARRAY));
        var generator = new CpuClassFileKernelGenerator();
        String f32Body = normalizedContiguousBody(generator.generateClassBytes(
                f32.units().getFirst().portablePlan().specialization(),
                f32.units().getFirst().portablePlan().kernelIr()));
        String f64Body = normalizedContiguousBody(generator.generateClassBytes(
                f64.units().getFirst().portablePlan().specialization(),
                f64.units().getFirst().portablePlan().kernelIr()));
        String oracle = CpuLossPerformanceOracle.compile(List.of(
                new CpuLossPerformanceOracle.Spec("f32", CpuLossPerformanceOracle.Family.INDEX,
                        CpuLossPerformanceOracle.Floating.F32, CpuLossPerformanceOracle.Floating.F32,
                        CpuLossPerformanceOracle.Reduction.NONE, true,
                        CpuLossPerformanceOracle.Carrier.FLOAT_ARRAY,
                        CpuLossPerformanceOracle.Carrier.INT_ARRAY,
                        CpuLossPerformanceOracle.Carrier.FLOAT_ARRAY,
                        CpuLossPerformanceOracle.Index.I32),
                new CpuLossPerformanceOracle.Spec("f64", CpuLossPerformanceOracle.Family.INDEX,
                        CpuLossPerformanceOracle.Floating.F64, CpuLossPerformanceOracle.Floating.F32,
                        CpuLossPerformanceOracle.Reduction.NONE, true,
                        CpuLossPerformanceOracle.Carrier.DOUBLE_ARRAY,
                        CpuLossPerformanceOracle.Carrier.LONG_ARRAY,
                        CpuLossPerformanceOracle.Carrier.DOUBLE_ARRAY,
                        CpuLossPerformanceOracle.Index.I64))).source();
        assertAll(() -> assertTrue(oracle.contains("for(int coordinate=0;coordinate<rank;coordinate++)")),
                () -> assertTrue(oracle.contains("selected==geometry[8]")),
                () -> assertTrue(f32Body.contains("FALOAD") && f32Body.contains("FASTORE"), f32Body),
                () -> assertTrue(f64Body.contains("DALOAD") && f64Body.contains("DASTORE"), f64Body),
                () -> assertTrue(f32Body.contains("StrictMath.exp") && f32Body.contains("StrictMath.log"), f32Body),
                () -> assertTrue(f64Body.contains("StrictMath.exp") && f64Body.contains("StrictMath.log"), f64Body));
    }

    @Test
    void contiguousFloat64IndexLsePassKeepsDirectLogitLoadOnStack() {
        Shape logits = Shape.of(2, 32, 64);
        var plan = plan(new Operation(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                new IndexCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.SUM,
                        Optional.empty())), List.of(0, 1),
                List.of(desc(DataType.FLOAT64, logits), desc(DataType.INT64, Shape.of(2, 64))),
                desc(DataType.FLOAT64, Shape.scalar()), List.of(CarrierAccess.DOUBLE_ARRAY,
                        CarrierAccess.LONG_ARRAY, CarrierAccess.MEMORY_SEGMENT));
        var route = plan.units().getFirst().portablePlan();
        var model = ClassFile.of().parse(new CpuClassFileKernelGenerator().generateClassBytes(
                route.specialization(), route.kernelIr()));
        List<Instruction> instructions = model.methods().stream().filter(method -> method
                .methodName().stringValue().equals(CpuLossEmitter.CONTIGUOUS_INT_NAME)).findFirst()
                .orElseThrow().code().orElseThrow().elementStream()
                .filter(Instruction.class::isInstance).map(Instruction.class::cast).toList();
        int exp = java.util.stream.IntStream.range(0, instructions.size()).filter(index ->
                instructions.get(index) instanceof InvokeInstruction invoke
                        && invoke.owner().asInternalName().equals("java/lang/StrictMath")
                        && invoke.name().stringValue().equals("exp")).findFirst().orElseThrow();
        assertEquals(java.lang.classfile.Opcode.DSUB, instructions.get(exp - 1).opcode());
        assertEquals(java.lang.classfile.Opcode.DALOAD, instructions.get(exp - 3).opcode(),
                "the direct FLOAT64 logit load must feed subtraction without a local spill/reload");
    }

    @Test
    void contiguousIndexMaximumPassKeepsItsClassAddressOnStack() {
        Shape logits = Shape.of(2, 32, 64);
        var plan = plan(new Operation(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                new IndexCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.SUM,
                        Optional.empty())), List.of(0, 1),
                List.of(desc(DataType.FLOAT32, logits), desc(DataType.INT32, Shape.of(2, 64))),
                desc(DataType.FLOAT32, Shape.scalar()), List.of(CarrierAccess.FLOAT_ARRAY,
                        CarrierAccess.INT_ARRAY, CarrierAccess.MEMORY_SEGMENT));
        var route = plan.units().getFirst().portablePlan();
        var model = ClassFile.of().parse(new CpuClassFileKernelGenerator().generateClassBytes(
                route.specialization(), route.kernelIr()));
        List<Instruction> instructions = model.methods().stream().filter(method -> method
                .methodName().stringValue().equals(CpuLossEmitter.CONTIGUOUS_INT_NAME)).findFirst()
                .orElseThrow().code().orElseThrow().elementStream()
                .filter(Instruction.class::isInstance).map(Instruction.class::cast).toList();
        int firstLoad = java.util.stream.IntStream.range(0, instructions.size()).filter(index ->
                instructions.get(index).opcode() == java.lang.classfile.Opcode.FALOAD)
                .findFirst().orElseThrow();
        assertEquals(java.lang.classfile.Opcode.IADD, instructions.get(firstLoad - 1).opcode(),
                "the maximum-pass address must feed the direct logit load without a local spill");
        int log = java.util.stream.IntStream.range(0, instructions.size()).filter(index ->
                instructions.get(index) instanceof InvokeInstruction invoke
                        && invoke.owner().asInternalName().equals("java/lang/StrictMath")
                        && invoke.name().stringValue().equals("log")).findFirst().orElseThrow();
        assertAll(() -> assertEquals(java.lang.classfile.Opcode.D2F,
                        instructions.get(log + 1).opcode()),
                () -> assertEquals(java.lang.classfile.Opcode.FADD,
                        instructions.get(log + 2).opcode(),
                        "the log result must feed LSE addition without a local spill/reload"));
    }

    @Test
    void contiguousDenseSegmentTargetAccumulatesItsContributionWithoutATemporarySpill() {
        Shape logits = Shape.of(2, 32, 64);
        var plan = plan(new Operation(LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                new DenseCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.MEAN)),
                List.of(0, 1), List.of(desc(DataType.BFLOAT16, logits),
                        desc(DataType.BFLOAT16, logits)), desc(DataType.BFLOAT16, Shape.scalar()),
                List.of(CarrierAccess.SHORT_ARRAY, CarrierAccess.MEMORY_SEGMENT,
                        CarrierAccess.SHORT_ARRAY));
        List<Opcode> opcodes = contiguousIntCode(plan).elementStream()
                .filter(Instruction.class::isInstance).map(Instruction.class::cast)
                .map(Instruction::opcode).toList();
        int multiply = java.util.stream.IntStream.range(0, opcodes.size()).filter(index ->
                opcodes.get(index) == Opcode.FMUL).findFirst().orElseThrow();

        assertAll(
                () -> assertEquals(Opcode.FADD, opcodes.get(multiply + 1),
                        "loss + weight * (lse - logit) must remain on the operand stack"),
                () -> assertEquals(Opcode.FSTORE, opcodes.get(multiply + 2),
                        "only the accumulated loss may be stored after the contribution"));
    }

    @Test
    void contiguousIndexMeanSegmentTargetDefersItsScalarOutputAddressUntilTheFinalStore() {
        Shape logits = Shape.of(2, 32, 64);
        var plan = plan(new Operation(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                new IndexCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.MEAN,
                        Optional.empty())), List.of(0, 1), List.of(desc(DataType.FLOAT32, logits),
                        desc(DataType.INT64, Shape.of(2, 64))), desc(DataType.FLOAT32, Shape.scalar()),
                List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.MEMORY_SEGMENT,
                        CarrierAccess.FLOAT_ARRAY));
        var code = contiguousIntCode(plan);
        List<InvokeInstruction> calls = code.elementStream().filter(InvokeInstruction.class::isInstance)
                .map(InvokeInstruction.class::cast).toList();

        assertAll(
                () -> assertEquals(26, code.maxLocals(),
                        "reduced index traversal must not retain a per-sample output address"),
                () -> assertTrue(calls.stream().anyMatch(call -> call.owner().asInternalName()
                        .equals("java/lang/foreign/MemorySegment")
                        && call.name().stringValue().equals("get")),
                        "the target remains a direct segment load"),
                () -> assertFalse(calls.stream().anyMatch(call -> call.owner().asInternalName()
                        .startsWith("io/github/pho001/synaptik")),
                        "the general target-segment body has no helper bridge"));
    }

    @Test
    void generatedMseNoneVerifiesAndExecutesACompleteIndependentRange() throws Throwable {
        Shape shape = Shape.of(3);
        var plan = plan(new Operation(LossKind.MEAN_SQUARED_ERROR,
                new MeanSquaredErrorAttrs(LossReduction.NONE)), List.of(0, 1),
                List.of(desc(DataType.FLOAT64, shape), desc(DataType.FLOAT64, shape)),
                desc(DataType.FLOAT64, shape), List.of(CarrierAccess.DOUBLE_ARRAY,
                        CarrierAccess.DOUBLE_ARRAY, CarrierAccess.DOUBLE_ARRAY));
        double[] prediction = {3.0, -2.0, 0.5};
        double[] target = {1.0, 2.0, -1.5};
        double[] output = {-17.0, 0.0, 0.0};
        invoke(artifact(plan), plan, prediction, target, output, 1L, 3L);
        assertAll(() -> assertEquals(-17.0, output[0]),
                () -> assertArrayEquals(new double[] {16.0, 4.0},
                        new double[] {output[1], output[2]}),
                () -> assertEquals(CpuLossIr.RangeForm.INDEPENDENT_DOMAIN,
                        loss(plan).rangeForm()));
    }

    @Test
    void generatedMseSumUsesFloatSegmentCarriersAndSharedBfloatMeanRole() throws Throwable {
        Shape shape = Shape.of(3);
        var sumPlan = plan(new Operation(LossKind.MEAN_SQUARED_ERROR,
                new MeanSquaredErrorAttrs(LossReduction.SUM)), List.of(0, 1),
                List.of(desc(DataType.FLOAT32, shape), desc(DataType.FLOAT32, shape)),
                desc(DataType.FLOAT32, Shape.scalar()), List.of(CarrierAccess.MEMORY_SEGMENT,
                        CarrierAccess.FLOAT_ARRAY, CarrierAccess.MEMORY_SEGMENT));
        float[] prediction = {2.5f, -3.0f, .25f};
        float[] target = {-1.0f, 4.0f, -.75f};
        float[] sum = new float[1];
        invoke(artifact(sumPlan), sumPlan, MemorySegment.ofArray(prediction), target,
                MemorySegment.ofArray(sum), 0L, 1L);
        assertAll(() -> assertEquals(Float.floatToRawIntBits(
                        CpuLossReferenceKernel.meanSquaredError(prediction, target,
                                LossReduction.SUM)[0]), Float.floatToRawIntBits(sum[0])),
                () -> assertEquals(CpuLossIr.RangeForm.COMPLETE_REDUCTION,
                        loss(sumPlan).rangeForm()));

        var meanPlan = plan(new Operation(LossKind.MEAN_SQUARED_ERROR,
                new MeanSquaredErrorAttrs(LossReduction.MEAN)), List.of(0, 0),
                List.of(desc(DataType.BFLOAT16, shape)), desc(DataType.BFLOAT16, Shape.scalar()),
                List.of(CarrierAccess.SHORT_ARRAY, CarrierAccess.SHORT_ARRAY));
        short[] shared = {bf16(2.0f), bf16(-.5f), bf16(7.0f)};
        short[] mean = new short[1];
        invoke(artifact(meanPlan), meanPlan, shared, mean, 0L, 1L);
        assertAll(() -> assertEquals(List.of(0, 0), loss(meanPlan).roleBoundaryPositions()),
                () -> assertEquals(2, loss(meanPlan).boundaryTypes().size()),
                () -> assertEquals(bf16(0.0f), mean[0]));
    }

    @Test
    void generatedDenseNoneUsesLogitsRankTargetAndColdOffsetStrideGeometry() throws Throwable {
        Shape logitsShape = Shape.of(2, 3);
        LayoutDescriptor logitsLayout = LayoutDescriptor.of(logitsShape, new long[] {4, 1}, 1,
                true);
        LayoutDescriptor targetLayout = LayoutDescriptor.of(logitsShape, new long[] {5, 1}, 2,
                true);
        Shape outputShape = Shape.of(2);
        LayoutDescriptor outputLayout = LayoutDescriptor.of(outputShape, new long[] {2}, 1, true);
        var plan = plan(new Operation(LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                new DenseCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.NONE)),
                List.of(0, 1), List.of(desc(DataType.FLOAT64, logitsShape, logitsLayout),
                        desc(DataType.FLOAT64, logitsShape, targetLayout)),
                desc(DataType.FLOAT64, outputShape, outputLayout), List.of(CarrierAccess.DOUBLE_ARRAY,
                        CarrierAccess.DOUBLE_ARRAY, CarrierAccess.DOUBLE_ARRAY));
        double[] logits = new double[8];
        logits[1] = 1.0; logits[2] = 2.0; logits[3] = 3.0;
        logits[5] = -3.0; logits[6] = -2.0; logits[7] = -1.0;
        double[] target = new double[10];
        target[2] = 0.0; target[3] = 0.0; target[4] = 1.0;
        target[7] = 1.0; target[8] = 0.0; target[9] = 0.0;
        double[] output = new double[4];
        invoke(artifact(plan), plan, logits, target, output, 0L, 2L);
        double[] expected = CpuLossReferenceKernel.denseCategoricalCrossEntropy(
                new double[] {1, 2, 3, -3, -2, -1}, new double[] {0, 0, 1, 1, 0, 0}, 2,
                3, LossReduction.NONE);
        assertAll(() -> assertEquals(2, loss(plan).geometry().targetRank()),
                () -> assertEquals(1, loss(plan).geometry().outputRank()),
                () -> assertArrayEquals(new long[] {1L, 2L, 1L}, java.util.Arrays.copyOfRange(
                        loss(plan).geometry().pack(new long[] {0L, 0L, 0L}), 4, 7)),
                () -> assertEquals(expected[0], output[1], 0.0d),
                () -> assertEquals(expected[1], output[3], 0.0d));
    }

    @Test
    void generatedDenseSumSkipsExactZeroTargetsBeforeNonFiniteContribution() throws Throwable {
        Shape logitsShape = Shape.of(1, 2);
        var plan = plan(new Operation(LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                new DenseCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.SUM)),
                List.of(0, 1), List.of(desc(DataType.FLOAT32, logitsShape),
                        desc(DataType.FLOAT32, logitsShape)), desc(DataType.FLOAT32, Shape.scalar()),
                List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.FLOAT_ARRAY,
                        CarrierAccess.FLOAT_ARRAY));
        float[] logits = {Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY};
        float[] target = {0.0f, -0.0f};
        float[] output = new float[1];
        invoke(artifact(plan), plan, logits, target, output, 0L, 1L);
        assertEquals(Float.floatToRawIntBits(0.0f), Float.floatToRawIntBits(output[0]));
    }

    @Test
    void generatedDenseBfloatMeanSupportsSharedInputCarrierRole() throws Throwable {
        Shape logitsShape = Shape.of(2, 2);
        var plan = plan(new Operation(LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                new DenseCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.MEAN)),
                List.of(0, 0), List.of(desc(DataType.BFLOAT16, logitsShape)),
                desc(DataType.BFLOAT16, Shape.scalar()), List.of(CarrierAccess.SHORT_ARRAY,
                        CarrierAccess.SHORT_ARRAY));
        short[] shared = {bf16(1.0f), bf16(0.0f), bf16(0.0f), bf16(1.0f)};
        short[] output = new short[1];
        invoke(artifact(plan), plan, shared, output, 0L, 1L);
        float[] decoded = decode(shared);
        float expected = CpuLossReferenceKernel.denseCategoricalCrossEntropy(decoded, decoded,
                2, 2, LossReduction.MEAN)[0];
        assertEquals(bf16(expected), output[0]);
    }

    @Test
    void generatedDenseNoneUsesContiguousRowsWhenTheClassAxisHasInnerSamples() throws Throwable {
        Shape logitsShape = Shape.of(2, 3, 2);
        Shape outputShape = Shape.of(2, 2);
        var plan = plan(new Operation(LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                new DenseCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.NONE)),
                List.of(0, 1), List.of(desc(DataType.FLOAT64, logitsShape),
                        desc(DataType.FLOAT64, logitsShape)), desc(DataType.FLOAT64, outputShape),
                List.of(CarrierAccess.DOUBLE_ARRAY, CarrierAccess.DOUBLE_ARRAY,
                        CarrierAccess.DOUBLE_ARRAY));
        double[] logits = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        double[] target = {0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0};
        double[] output = new double[4];
        invoke(artifact(plan), plan, logits, target, output, 0L, 4L);
        double[] expected = {
                CpuLossReferenceKernel.denseCategoricalCrossEntropy(
                        new double[] {1, 3, 5}, new double[] {0, 1, 0}, 1, 3,
                        LossReduction.NONE)[0],
                CpuLossReferenceKernel.denseCategoricalCrossEntropy(
                        new double[] {2, 4, 6}, new double[] {0, 0, 1}, 1, 3,
                        LossReduction.NONE)[0],
                CpuLossReferenceKernel.denseCategoricalCrossEntropy(
                        new double[] {7, 9, 11}, new double[] {1, 0, 0}, 1, 3,
                        LossReduction.NONE)[0],
                CpuLossReferenceKernel.denseCategoricalCrossEntropy(
                        new double[] {8, 10, 12}, new double[] {0, 1, 0}, 1, 3,
                        LossReduction.NONE)[0]
        };
        assertArrayEquals(expected, output);
    }

    @Test
    void contiguousDispatchGeometryRetainsOnlyControlValuesThatFormerlyRejectedDenseRows()
            throws Throwable {
        Shape logits = Shape.of(2, 3, 2);
        var mse = plan(new Operation(LossKind.MEAN_SQUARED_ERROR,
                new MeanSquaredErrorAttrs(LossReduction.MEAN)), List.of(0, 1),
                List.of(desc(DataType.FLOAT32, logits), desc(DataType.FLOAT32, logits)),
                desc(DataType.FLOAT32, Shape.scalar()), List.of(CarrierAccess.FLOAT_ARRAY,
                        CarrierAccess.MEMORY_SEGMENT, CarrierAccess.FLOAT_ARRAY));
        var dense = plan(new Operation(LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                new DenseCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.NONE)),
                List.of(0, 1), List.of(desc(DataType.FLOAT32, logits), desc(DataType.FLOAT32, logits)),
                desc(DataType.FLOAT32, Shape.of(2, 2)), List.of(CarrierAccess.FLOAT_ARRAY,
                        CarrierAccess.FLOAT_ARRAY, CarrierAccess.MEMORY_SEGMENT));
        var indexed = plan(new Operation(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                new IndexCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.MEAN,
                        Optional.of(ScalarValue.int64(Long.MIN_VALUE)))), List.of(0, 1),
                List.of(desc(DataType.FLOAT32, logits), desc(DataType.INT64, Shape.of(2, 2))),
                desc(DataType.FLOAT32, Shape.scalar()), List.of(CarrierAccess.MEMORY_SEGMENT,
                        CarrierAccess.LONG_ARRAY, CarrierAccess.FLOAT_ARRAY));
        assertAll(
                () -> assertEquals(-1L, loss(mse).geometry().pack(new long[] {0, 0, 0})[1]),
                () -> assertEquals(0, loss(mse).geometry().outputRank()),
                () -> assertEquals(3, loss(dense).geometry().targetRank()),
                () -> assertEquals(2, loss(dense).geometry().outputRank()),
                () -> assertEquals(Long.MIN_VALUE,
                        loss(indexed).geometry().pack(new long[] {0, 0, 0})[8]),
                () -> assertEquals(2, loss(indexed).geometry().targetRank()));
    }

    @Test
    void generatedIndexFormsCoverBothWidthsIgnoreOrderingAndAllAccumulatorDomains()
            throws Throwable {
        Shape logitsShape = Shape.of(2, 3);
        Shape targetShape = Shape.of(2);
        var f64None = plan(new Operation(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                new IndexCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.NONE,
                        Optional.empty())), List.of(0, 1), List.of(desc(DataType.FLOAT64, logitsShape),
                        desc(DataType.INT32, targetShape)), desc(DataType.FLOAT64, targetShape),
                List.of(CarrierAccess.DOUBLE_ARRAY, CarrierAccess.INT_ARRAY,
                        CarrierAccess.DOUBLE_ARRAY));
        double[] f64Logits = {1, 2, 3, -3, -2, -1};
        int[] int32 = {2, 0};
        double[] f64Output = new double[2];
        invoke(artifact(f64None), f64None, f64Logits, int32, f64Output, 0L, 2L);
        assertArrayEquals(CpuLossReferenceKernel.indexCategoricalCrossEntropy(f64Logits,
                new long[] {2, 0}, 3, null, LossReduction.NONE), f64Output);

        var f32Sum = plan(new Operation(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                new IndexCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.SUM,
                        Optional.of(ScalarValue.int64(Long.MIN_VALUE)))), List.of(0, 1),
                List.of(desc(DataType.FLOAT32, logitsShape), desc(DataType.INT64, targetShape)),
                desc(DataType.FLOAT32, Shape.scalar()), List.of(CarrierAccess.FLOAT_ARRAY,
                        CarrierAccess.LONG_ARRAY, CarrierAccess.FLOAT_ARRAY));
        float[] f32Logits = {1f, 2f, 3f, -3f, -2f, -1f};
        long[] int64 = {Long.MIN_VALUE, 1L};
        float[] f32Output = new float[1];
        invoke(artifact(f32Sum), f32Sum, f32Logits, int64, f32Output, 0L, 1L);
        assertEquals(Float.floatToRawIntBits(CpuLossReferenceKernel.indexCategoricalCrossEntropy(
                f32Logits, int64, 3, Long.MIN_VALUE, LossReduction.SUM)[0]),
                Float.floatToRawIntBits(f32Output[0]));

        var bfMean = plan(new Operation(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                new IndexCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.MEAN,
                        Optional.of(ScalarValue.int32(-1)))), List.of(0, 1),
                List.of(desc(DataType.BFLOAT16, logitsShape), desc(DataType.INT32, targetShape)),
                desc(DataType.BFLOAT16, Shape.scalar()), List.of(CarrierAccess.SHORT_ARRAY,
                        CarrierAccess.INT_ARRAY, CarrierAccess.SHORT_ARRAY));
        short[] bfLogits = {bf16(1f), bf16(2f), bf16(3f), bf16(-3f), bf16(-2f), bf16(-1f)};
        int[] bfIndices = {-1, 1};
        short[] bfOutput = new short[1];
        invoke(artifact(bfMean), bfMean, bfLogits, bfIndices, bfOutput, 0L, 1L);
        float expected = CpuLossReferenceKernel.indexCategoricalCrossEntropy(decode(bfLogits),
                new long[] {-1, 1}, 3, -1L, LossReduction.MEAN)[0];
        assertAll(() -> assertEquals(1, loss(f64None).geometry().targetRank()),
                () -> assertEquals(1, loss(f64None).geometry().outputRank()),
                () -> assertEquals(bf16(expected), bfOutput[0]));
    }

    @Test
    void generalizedCategoricalSegmentCarrierFormsKeepTheFrozenDirectSemantics() throws Throwable {
        Shape logits = Shape.of(2, 3);
        var dense = plan(new Operation(LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                new DenseCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.MEAN)),
                List.of(0, 1), List.of(desc(DataType.BFLOAT16, logits),
                        desc(DataType.FLOAT32, logits)), desc(DataType.FLOAT32, Shape.scalar()),
                List.of(CarrierAccess.SHORT_ARRAY, CarrierAccess.MEMORY_SEGMENT,
                        CarrierAccess.MEMORY_SEGMENT));
        short[] denseLogits = {bf16(1f), bf16(2f), bf16(3f), bf16(-3f), bf16(-2f), bf16(-1f)};
        float[] denseTarget = {0f, 0f, 1f, 1f, 0f, 0f};
        float[] denseOutput = new float[1];
        invoke(artifact(dense), dense, denseLogits, MemorySegment.ofArray(denseTarget),
                MemorySegment.ofArray(denseOutput), 0L, 1L);

        var indexed = plan(new Operation(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                new IndexCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.NONE,
                        Optional.of(ScalarValue.int64(-1)))), List.of(0, 1),
                List.of(desc(DataType.FLOAT64, logits), desc(DataType.INT64, Shape.of(2))),
                desc(DataType.FLOAT64, Shape.of(2)), List.of(CarrierAccess.DOUBLE_ARRAY,
                        CarrierAccess.LONG_ARRAY, CarrierAccess.MEMORY_SEGMENT));
        double[] indexLogits = {1, 2, 3, -3, -2, -1};
        long[] indexTarget = {-1, 1};
        double[] indexOutput = {-7, -7};
        invoke(artifact(indexed), indexed, indexLogits, indexTarget,
                MemorySegment.ofArray(indexOutput), 0L, 2L);

        assertAll(
                () -> assertEquals(Float.floatToRawIntBits(CpuLossReferenceKernel
                        .denseCategoricalCrossEntropy(decode(denseLogits), denseTarget, 2, 3,
                                LossReduction.MEAN)[0]), Float.floatToRawIntBits(denseOutput[0])),
                () -> assertArrayEquals(CpuLossReferenceKernel.indexCategoricalCrossEntropy(
                        indexLogits, indexTarget, 3, -1L, LossReduction.NONE), indexOutput));
    }

    @Test
    void contiguousIndexBodyDoesNotReserveMseOrDenseOnlyLocals() {
        Shape logitsShape = Shape.of(2, 32, 64);
        var plan = plan(new Operation(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                new IndexCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.MEAN,
                        Optional.empty())), List.of(0, 1),
                List.of(desc(DataType.FLOAT32, logitsShape),
                        desc(DataType.INT64, Shape.of(2, 64))),
                desc(DataType.FLOAT32, Shape.scalar()), List.of(CarrierAccess.FLOAT_ARRAY,
                        CarrierAccess.LONG_ARRAY, CarrierAccess.MEMORY_SEGMENT));
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        var model = ClassFile.of().parse(generator.generateClassBytes(route.specialization(),
                route.kernelIr()));
        int maxLocals = model.methods().stream().filter(method -> method.methodName().stringValue()
                .equals(CpuLossEmitter.CONTIGUOUS_INT_NAME)).findFirst().orElseThrow().attributes()
                .stream().filter(java.lang.classfile.attribute.CodeAttribute.class::isInstance)
                .map(java.lang.classfile.attribute.CodeAttribute.class::cast).findFirst().orElseThrow()
                .maxLocals();

        // The frozen FLOAT32 index oracle needs only index/category state.  Keep this body below
        // the former all-family reservation, which needlessly carried MSE and dense locals into
        // the hot traversal and regressed the same array/segment specialization family.
        assertEquals(26, maxLocals, "index contiguous maxLocals");

        var segmentPlan = plan(new Operation(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                new IndexCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.NONE,
                        Optional.of(ScalarValue.int32(-1)))), List.of(0, 1),
                List.of(desc(DataType.FLOAT32, logitsShape),
                        desc(DataType.INT32, Shape.of(2, 64))),
                desc(DataType.FLOAT32, Shape.of(2, 64)), List.of(CarrierAccess.MEMORY_SEGMENT,
                        CarrierAccess.MEMORY_SEGMENT, CarrierAccess.FLOAT_ARRAY));
        assertEquals(25, contiguousIntMaxLocals(segmentPlan),
                "segment-backed index NONE retains the direct oracle lifetime graph");
    }

    @Test
    void contiguousIndexSumDoesNotReserveMeanOnlyIncludedCount() {
        Shape logitsShape = Shape.of(2, 32, 64);
        var sum = plan(new Operation(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                new IndexCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.SUM,
                        Optional.empty())), List.of(0, 1),
                List.of(desc(DataType.FLOAT32, logitsShape),
                        desc(DataType.INT32, Shape.of(2, 64))),
                desc(DataType.FLOAT32, Shape.scalar()), List.of(CarrierAccess.FLOAT_ARRAY,
                        CarrierAccess.INT_ARRAY, CarrierAccess.MEMORY_SEGMENT));
        var mean = plan(new Operation(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                new IndexCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.MEAN,
                        Optional.empty())), List.of(0, 1),
                List.of(desc(DataType.FLOAT32, logitsShape),
                        desc(DataType.INT32, Shape.of(2, 64))),
                desc(DataType.FLOAT32, Shape.scalar()), List.of(CarrierAccess.FLOAT_ARRAY,
                        CarrierAccess.INT_ARRAY, CarrierAccess.MEMORY_SEGMENT));

        assertEquals(25, contiguousIntMaxLocals(sum),
                "SUM has neither an included-count nor a reduced-output-address local");
        assertEquals(26, contiguousIntMaxLocals(mean),
                "MEAN retains its denominator count without an extra output-address local");
    }

    @Test
    void exactArrayIndexMeanSegmentUsesOneFinalDirectSegmentStore() {
        Shape logits = Shape.of(2, 32, 64);
        var plan = plan(new Operation(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                new IndexCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.MEAN,
                        Optional.of(ScalarValue.int32(-1)))), List.of(0, 1),
                List.of(desc(DataType.BFLOAT16, logits), desc(DataType.INT32, Shape.of(2, 64))),
                desc(DataType.BFLOAT16, Shape.scalar()), List.of(CarrierAccess.SHORT_ARRAY,
                        CarrierAccess.INT_ARRAY, CarrierAccess.MEMORY_SEGMENT));
        var route = plan.units().getFirst().portablePlan();
        var model = ClassFile.of().parse(new CpuClassFileKernelGenerator().generateClassBytes(
                route.specialization(), route.kernelIr()));
        List<InvokeInstruction> calls = model.methods().stream().filter(method -> method
                .methodName().stringValue().equals(CpuLossEmitter.CONTIGUOUS_INT_NAME)).findFirst()
                .orElseThrow().code().orElseThrow().elementStream()
                .filter(InvokeInstruction.class::isInstance).map(InvokeInstruction.class::cast).toList();
        long stores = calls.stream().filter(call -> call.owner().asInternalName()
                .equals("java/lang/foreign/MemorySegment") && call.name().stringValue().equals("set")).count();
        assertAll(
                () -> assertEquals(1L, stores, "reduced output has one final segment store"),
                () -> assertTrue(calls.stream().anyMatch(call -> call.owner().asInternalName()
                        .equals("java/lang/StrictMath") && call.name().stringValue().equals("exp"))),
                () -> assertTrue(calls.stream().anyMatch(call -> call.owner().asInternalName()
                        .equals("java/lang/Float") && call.name().stringValue().equals("intBitsToFloat"))),
                () -> assertFalse(calls.stream().anyMatch(call -> call.owner().asInternalName()
                        .startsWith("io/github/pho001/synaptik")), "no generated helper/dispatch call"));
    }

    @Test
    void exactArrayIndexMeanKeepsTheFrozenF32ArrayOracleLifetimeGraph() {
        Shape logits = Shape.of(2, 32, 64);
        var plan = plan(new Operation(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                new IndexCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.MEAN,
                        Optional.empty())), List.of(0, 1),
                List.of(desc(DataType.FLOAT32, logits), desc(DataType.INT32, Shape.of(2, 64))),
                desc(DataType.FLOAT32, Shape.scalar()), List.of(CarrierAccess.FLOAT_ARRAY,
                        CarrierAccess.INT_ARRAY, CarrierAccess.FLOAT_ARRAY));

        var code = contiguousIntCode(plan);
        assertAll(
                () -> assertEquals(306, code.codeLength(),
                        "F32 array INDEX/MEAN must retain the frozen oracle control-flow body"),
                () -> assertEquals(26, code.maxLocals(),
                        "F32 array INDEX/MEAN must retain javac's total/count lifetime graph"));
    }

    @Test
    void contiguousMixedMseNoneKeepsTargetLoadAndOutputAddressOnTheDirectOracleStackPath() {
        Shape shape = Shape.of(2, 32, 64);
        var plan = plan(new Operation(LossKind.MEAN_SQUARED_ERROR,
                new MeanSquaredErrorAttrs(LossReduction.NONE)), List.of(0, 1),
                List.of(desc(DataType.FLOAT64, shape), desc(DataType.FLOAT32, shape)),
                desc(DataType.FLOAT64, shape), List.of(CarrierAccess.DOUBLE_ARRAY,
                        CarrierAccess.FLOAT_ARRAY, CarrierAccess.DOUBLE_ARRAY));
        var code = contiguousIntCode(plan);
        List<Opcode> opcodes = code.elementStream().filter(Instruction.class::isInstance)
                .map(Instruction.class::cast).map(Instruction::opcode).toList();
        int targetLoad = opcodes.indexOf(Opcode.FALOAD);
        int outputStore = opcodes.lastIndexOf(Opcode.DASTORE);

        assertAll(
                () -> assertEquals(15, code.maxLocals(),
                        "mixed MSE retains only bases, ordinal, and its difference local"),
                () -> assertTrue(targetLoad >= 0),
                () -> assertEquals(List.of(Opcode.FALOAD, Opcode.F2D, Opcode.DSUB,
                        Opcode.DSTORE), opcodes.subList(targetLoad, targetLoad + 4),
                        "the target load must feed subtraction without address/value spills"),
                () -> assertEquals(List.of(Opcode.ILOAD, Opcode.ILOAD, Opcode.IADD,
                        Opcode.DLOAD, Opcode.DLOAD, Opcode.DMUL, Opcode.DASTORE),
                        opcodes.subList(outputStore - 6, outputStore + 1),
                        "the output base and difference square feed the direct array store"));
    }

    @Test
    void contiguousMixedMseNoneRetainsTheOracleBaseOrdinalAndSourceAddressLifetimes() {
        Shape shape = Shape.of(2, 32, 64);
        var plan = plan(new Operation(LossKind.MEAN_SQUARED_ERROR,
                new MeanSquaredErrorAttrs(LossReduction.NONE)), List.of(0, 1),
                List.of(desc(DataType.FLOAT64, shape), desc(DataType.FLOAT32, shape)),
                desc(DataType.FLOAT64, shape), List.of(CarrierAccess.DOUBLE_ARRAY,
                        CarrierAccess.FLOAT_ARRAY, CarrierAccess.DOUBLE_ARRAY));
        var generated = contiguousIntCode(plan);
        var oracle = CpuLossPerformanceOracle.compile(List.of(new CpuLossPerformanceOracle.Spec(
                "mse", CpuLossPerformanceOracle.Family.MSE,
                CpuLossPerformanceOracle.Floating.F64, CpuLossPerformanceOracle.Floating.F32,
                CpuLossPerformanceOracle.Reduction.NONE, false,
                CpuLossPerformanceOracle.Carrier.DOUBLE_ARRAY,
                CpuLossPerformanceOracle.Carrier.FLOAT_ARRAY,
                CpuLossPerformanceOracle.Carrier.DOUBLE_ARRAY,
                CpuLossPerformanceOracle.Index.UNUSED)));
        var oracleCode = ClassFile.of().parse(oracle.classBytes()).methods().stream()
                .filter(method -> method.methodName().stringValue().equals("mse")).findFirst()
                .orElseThrow().attributes().stream()
                .filter(java.lang.classfile.attribute.CodeAttribute.class::isInstance)
                .map(java.lang.classfile.attribute.CodeAttribute.class::cast).findFirst().orElseThrow();
        List<Integer> generatedInitialStores = generated.elementStream()
                .filter(StoreInstruction.class::isInstance).map(StoreInstruction.class::cast)
                .map(StoreInstruction::slot).limit(6).toList();
        List<Integer> oracleInitialStores = oracleCode.elementStream()
                .filter(StoreInstruction.class::isInstance).map(StoreInstruction.class::cast)
                .map(StoreInstruction::slot).limit(6).toList();

        assertAll(
                () -> assertEquals(oracleCode.maxLocals(), generated.maxLocals(),
                        "the selected helper must retain the oracle local domain"),
                () -> assertEquals(oracleCode.maxStack(), generated.maxStack(),
                        "the selected helper must retain the oracle operand-stack bound"),
                () -> assertEquals(oracleInitialStores, generatedInitialStores,
                        "bases, range ordinal, source address, and difference must begin in the "
                                + "same lifetime order as the clean Java oracle"));
    }

    @Test
    void contiguousMixedMseNoneBindingSelectsTheDirectOracleEquivalentHelper() throws Exception {
        Shape shape = Shape.of(2, 32, 64);
        var plan = plan(new Operation(LossKind.MEAN_SQUARED_ERROR,
                new MeanSquaredErrorAttrs(LossReduction.NONE)), List.of(0, 1),
                List.of(desc(DataType.FLOAT64, shape), desc(DataType.FLOAT32, shape)),
                desc(DataType.FLOAT64, shape), List.of(CarrierAccess.DOUBLE_ARRAY,
                        CarrierAccess.FLOAT_ARRAY, CarrierAccess.DOUBLE_ARRAY));
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        byte[] bytes = generator.generateClassBytes(route.specialization(), route.kernelIr());
        var artifact = generator.defineClassBytes(route.specialization(), bytes);
        var generated = contiguousIntCode(plan);
        var oracle = CpuLossPerformanceOracle.compile(List.of(new CpuLossPerformanceOracle.Spec(
                "mse", CpuLossPerformanceOracle.Family.MSE,
                CpuLossPerformanceOracle.Floating.F64, CpuLossPerformanceOracle.Floating.F32,
                CpuLossPerformanceOracle.Reduction.NONE, false,
                CpuLossPerformanceOracle.Carrier.DOUBLE_ARRAY,
                CpuLossPerformanceOracle.Carrier.FLOAT_ARRAY,
                CpuLossPerformanceOracle.Carrier.DOUBLE_ARRAY,
                CpuLossPerformanceOracle.Index.UNUSED)));
        var oracleMethod = ClassFile.of().parse(oracle.classBytes()).methods().stream()
                .filter(method -> method.methodName().stringValue().equals("mse")).findFirst()
                .orElseThrow();
        var oracleCode = oracleMethod.code().orElseThrow();
        var oracleAttribute = oracleMethod.attributes().stream()
                .filter(java.lang.classfile.attribute.CodeAttribute.class::isInstance)
                .map(java.lang.classfile.attribute.CodeAttribute.class::cast).findFirst().orElseThrow();
        var model = ClassFile.of().parse(bytes);
        var publicEntry = model.methods().stream().filter(method -> method.flags()
                .has(AccessFlag.PUBLIC)).findFirst().orElseThrow().code().orElseThrow();
        var privateMethod = model.methods().stream().filter(method -> method.methodName()
                .stringValue().equals(CpuLossEmitter.CONTIGUOUS_INT_NAME)).findFirst().orElseThrow();
        var privateBody = privateMethod.code().orElseThrow();
        var privateAttribute = privateMethod.attributes().stream()
                .filter(java.lang.classfile.attribute.CodeAttribute.class::isInstance)
                .map(java.lang.classfile.attribute.CodeAttribute.class::cast).findFirst().orElseThrow();
        List<Opcode> oracleOpcodes = oracleCode.elementStream().filter(Instruction.class::isInstance)
                .map(Instruction.class::cast).map(Instruction::opcode).toList();
        List<Opcode> generatedOpcodes = generated.elementStream().filter(Instruction.class::isInstance)
                .map(Instruction.class::cast).map(Instruction::opcode).toList();

        assertAll(
                () -> assertEquals(route.specialization().entryType(),
                        artifact.lossEntryPointFor(true).type(),
                        "cold contiguous proof must bind the private typed body directly"),
                () -> assertEquals(oracleAttribute.codeLength(), privateAttribute.codeLength(),
                        "the selected private body must retain the direct oracle byte length"),
                () -> assertEquals(oracleOpcodes, generatedOpcodes,
                        "the selected private body must retain the direct oracle instruction path"),
                () -> assertTrue(publicEntry.elementStream().filter(InvokeInstruction.class::isInstance)
                        .map(InvokeInstruction.class::cast).anyMatch(invoke -> invoke.name()
                                .stringValue().equals(CpuLossEmitter.CONTIGUOUS_INT_NAME)),
                        "the public entry remains the cold unproved-geometry dispatcher"),
                () -> assertFalse(privateBody.elementStream().filter(InvokeInstruction.class::isInstance)
                        .map(InvokeInstruction.class::cast).anyMatch(invoke -> invoke.name()
                                .stringValue().equals(CpuLossEmitter.CONTIGUOUS_INT_NAME)),
                        "the measured selected body must not re-enter the public dispatcher"));
    }

    @Test
    void exactArrayIndexSumKeepsTheFrozenF32ArrayOracleLifetimeGraph() {
        Shape logits = Shape.of(2, 32, 64);
        var plan = plan(new Operation(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                new IndexCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.SUM,
                        Optional.empty())), List.of(0, 1),
                List.of(desc(DataType.FLOAT32, logits), desc(DataType.INT32, Shape.of(2, 64))),
                desc(DataType.FLOAT32, Shape.scalar()), List.of(CarrierAccess.FLOAT_ARRAY,
                        CarrierAccess.INT_ARRAY, CarrierAccess.FLOAT_ARRAY));

        var code = contiguousIntCode(plan);
        assertAll(
                () -> assertEquals(296, code.codeLength(),
                        "F32 array INDEX/SUM must retain the frozen oracle control-flow body"),
                () -> assertEquals(25, code.maxLocals(),
                        "F32 array INDEX/SUM must retain javac's total-only lifetime graph"));
    }

    @Test
    void exactHeapIndexNoneKeepsTheFrozenF32AndF64ValueLocalShapes() {
        Shape logits = Shape.of(2, 32, 64);
        Shape target = Shape.of(2, 64);
        var f32 = plan(new Operation(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                new IndexCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.NONE,
                        Optional.of(ScalarValue.int32(-1)))), List.of(0, 1),
                List.of(desc(DataType.FLOAT32, logits), desc(DataType.INT32, target)),
                desc(DataType.FLOAT32, target), List.of(CarrierAccess.FLOAT_ARRAY,
                        CarrierAccess.INT_ARRAY, CarrierAccess.FLOAT_ARRAY));
        var f64 = plan(new Operation(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                new IndexCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.NONE,
                        Optional.of(ScalarValue.int32(-1)))), List.of(0, 1),
                List.of(desc(DataType.FLOAT64, logits), desc(DataType.INT32, target)),
                desc(DataType.FLOAT64, target), List.of(CarrierAccess.DOUBLE_ARRAY,
                        CarrierAccess.INT_ARRAY, CarrierAccess.DOUBLE_ARRAY));

        var f32Code = contiguousIntCode(f32);
        assertEquals(329, f32Code.codeLength(), "frozen javac-equivalent F32 body length");
        assertEquals(25, f32Code.maxLocals(), "frozen javac-equivalent F32 local domain");

        var f64Code = contiguousIntCode(f64);
        List<Opcode> f64Opcodes = f64Code.elementStream().filter(Instruction.class::isInstance)
                .map(Instruction.class::cast).map(Instruction::opcode).toList();
        assertEquals(29, f64Code.maxLocals(), "F64 direct lifetime local domain");
        assertEquals(0, f64Opcodes.stream().filter(opcode -> opcode == Opcode.DUP2).count(),
                "F64 maximum uses the frozen ordinary-Java value local, not stack duplication");
        assertEquals(0, f64Opcodes.stream().filter(opcode -> opcode == Opcode.POP2).count(),
                "F64 maximum has no stack-only non-greater edge");
        assertTrue(f64Opcodes.contains(Opcode.DSTORE),
                "F64 maximum retains the direct logit in its verifier-safe candidate local");
    }

    @Test
    void contiguousArrayInputIndexNoneRetainsItsOracleLoopWhenOutputIsASegment() {
        Shape logits = Shape.of(2, 32, 64);
        var plan = plan(new Operation(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                new IndexCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.NONE,
                        Optional.of(ScalarValue.int64(-1)))), List.of(0, 1),
                List.of(desc(DataType.FLOAT32, logits), desc(DataType.INT64, Shape.of(2, 64))),
                desc(DataType.FLOAT32, Shape.of(2, 64)), List.of(CarrierAccess.FLOAT_ARRAY,
                        CarrierAccess.LONG_ARRAY, CarrierAccess.MEMORY_SEGMENT));

        var code = contiguousIntCode(plan);
        List<InvokeInstruction> calls = code.elementStream().filter(InvokeInstruction.class::isInstance)
                .map(InvokeInstruction.class::cast).toList();
        assertAll(
                () -> assertEquals(25, code.maxLocals(),
                        "a segment store does not expand the array-input classification lifetime"),
                () -> assertTrue(calls.stream().anyMatch(call -> call.owner().asInternalName()
                        .equals("java/lang/foreign/MemorySegment")
                        && call.name().stringValue().equals("set")),
                        "the output carrier changes only the direct final/per-ignored store"),
                () -> assertFalse(calls.stream().anyMatch(call -> call.owner().asInternalName()
                        .startsWith("io/github/pho001/synaptik")),
                        "the carrier-specific body has no Synaptik helper bridge"));
    }

    private static java.lang.classfile.attribute.CodeAttribute contiguousIntCode(
            io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan plan) {
        var route = plan.units().getFirst().portablePlan();
        var model = ClassFile.of().parse(new CpuClassFileKernelGenerator().generateClassBytes(
                route.specialization(), route.kernelIr()));
        return model.methods().stream().filter(method -> method.methodName().stringValue()
                .equals(CpuLossEmitter.CONTIGUOUS_INT_NAME)).findFirst().orElseThrow().attributes()
                .stream().filter(java.lang.classfile.attribute.CodeAttribute.class::isInstance)
                .map(java.lang.classfile.attribute.CodeAttribute.class::cast).findFirst().orElseThrow();
    }

    private static int contiguousIntMaxLocals(
            io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan plan) {
        return contiguousIntCode(plan).maxLocals();
    }

    @Test
    void generatedIndexLossLeavesInvalidTargetRejectionToTheColdPreWriteValidator()
            throws Throwable {
        Shape logitsShape = Shape.of(1, 2);
        Shape targetShape = Shape.of(1);
        var nonePlan = plan(new Operation(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                new IndexCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.NONE,
                        Optional.of(ScalarValue.int32(-1)))), List.of(0, 1),
                List.of(desc(DataType.FLOAT64, logitsShape), desc(DataType.INT32, targetShape)),
                desc(DataType.FLOAT64, targetShape), List.of(CarrierAccess.DOUBLE_ARRAY,
                        CarrierAccess.INT_ARRAY, CarrierAccess.DOUBLE_ARRAY));
        double[] noneOutput = {71.0d};
        byte[] noneBytes = new CpuClassFileKernelGenerator().generateClassBytes(
                nonePlan.units().getFirst().portablePlan().specialization(),
                nonePlan.units().getFirst().portablePlan().kernelIr());
        boolean allocatesException = java.util.stream.StreamSupport.stream(
                ClassFile.of().parse(noneBytes).constantPool().spliterator(), false)
                .filter(MemberRefEntry.class::isInstance).map(MemberRefEntry.class::cast)
                .anyMatch(member -> member.owner().asInternalName()
                        .equals("java/lang/IllegalArgumentException")
                        && member.name().stringValue().equals("<init>"));
        assertAll(() -> assertFalse(allocatesException),
                () -> assertEquals(71.0d, noneOutput[0]));
    }

    @Test
    void generatedLossClassIsPublicFieldFreeAndCallsNoSynaptikReferenceOrHelper() {
        Shape shape = Shape.of(1);
        var plan = plan(new Operation(LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                new DenseCategoricalCrossEntropyWithLogitsAttrs(0, LossReduction.NONE)),
                List.of(0, 1), List.of(desc(DataType.FLOAT32, shape),
                        desc(DataType.FLOAT32, shape)), desc(DataType.FLOAT32, Shape.scalar()),
                List.of(CarrierAccess.FLOAT_ARRAY,
                        CarrierAccess.FLOAT_ARRAY, CarrierAccess.FLOAT_ARRAY));
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        byte[] bytes = generator.generateClassBytes(route.specialization(), route.kernelIr());
        var model = ClassFile.of().parse(bytes);
        StringBuilder members = new StringBuilder();
        java.util.stream.StreamSupport.stream(model.constantPool().spliterator(), false)
                .filter(MemberRefEntry.class::isInstance).map(MemberRefEntry.class::cast)
                .forEach(member -> members.append(member.owner().asInternalName()).append('.')
                        .append(member.name().stringValue()).append('\n'));
        assertAll(() -> assertTrue(model.flags().has(AccessFlag.FINAL)),
                () -> assertTrue(model.fields().isEmpty()),
                () -> assertEquals(3, model.methods().size()),
                () -> assertEquals(1, model.methods().stream().filter(method ->
                        method.flags().has(AccessFlag.PUBLIC)).count()),
                () -> assertTrue(model.methods().stream().anyMatch(method ->
                        method.methodName().stringValue().equals(CpuLossEmitter.CONTIGUOUS_INT_NAME)
                                && method.flags().has(AccessFlag.PRIVATE))),
                () -> assertTrue(model.methods().stream().anyMatch(method ->
                        method.methodName().stringValue().equals(
                                io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratorSchema.ENTRY_NAME)
                                && method.flags().has(AccessFlag.PUBLIC))),
                () -> assertTrue(model.methods().stream().anyMatch(method ->
                        method.methodName().stringValue().equals(CpuLossEmitter.GENERIC_AFFINE_NAME)
                                && method.flags().has(AccessFlag.PRIVATE))),
                () -> assertFalse(members.toString().contains("CpuLossReferenceKernel")),
                () -> assertFalse(members.toString().contains("CpuLossReferenceKernel")),
                () -> assertFalse(members.toString().contains("CpuLossEmitter")),
                () -> assertFalse(members.toString().contains("java/lang/reflect")),
                () -> assertFalse(members.toString().contains("java/lang/invoke")),
                () -> assertTrue(members.toString().contains("java/lang/StrictMath.exp")));
    }

    private static io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedKernel
            artifact(io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan plan) {
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        return generator.defineClassBytes(route.specialization(),
                generator.generateClassBytes(route.specialization(), route.kernelIr()));
    }

    private static void invoke(
            io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedKernel artifact,
            io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan plan,
            Object sharedInput, Object output, long start, long end) throws Throwable {
        artifact.entryPoint().invokeWithArguments(sharedInput, output,
                loss(plan).geometry().pack(new long[] {0L, 0L, 0L}), start, end);
    }

    private static void invoke(
            io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedKernel artifact,
            io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan plan,
            Object prediction, Object targetOrOutput, Object outputOrStart, long startOrEnd,
            long end) throws Throwable {
        CpuLossIr loss = loss(plan);
        if (loss.boundaryTypes().size() == 2) {
            artifact.entryPoint().invokeWithArguments(prediction, targetOrOutput,
                    loss.geometry().pack(new long[] {0L, 0L, 0L}), startOrEnd, end);
        } else {
            artifact.entryPoint().invokeWithArguments(prediction, targetOrOutput, outputOrStart,
                    loss.geometry().pack(new long[] {0L, 0L, 0L}), startOrEnd, end);
        }
    }

    private static io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan
            plan(Operation operation, List<Integer> roles, List<TensorDescriptor> inputs,
                    TensorDescriptor output, List<CarrierAccess> carriers) {
        PrepareContext<CpuPartitionAnalysisInputs> base = CpuScatterLoweringTest.context(operation,
                roles, inputs, output);
        return new CpuPartitionPreparer().analyze(new PrepareContext<>(base.partition(),
                base.nodes(), base.values(), base.memoryRequirements(), Map.of(),
                new CpuPartitionAnalysisInputs(false, carriers))).plan();
    }

    private static CpuLossIr loss(
            io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan plan) {
        return (CpuLossIr) plan.units().getFirst().portablePlan().portableKernelIr();
    }

    private static TensorDescriptor desc(DataType type, Shape shape) {
        return CpuScatterLoweringTest.desc(type, shape);
    }

    private static TensorDescriptor desc(DataType type, Shape shape, LayoutDescriptor layout) {
        return new TensorDescriptor(type, shape, Optional.of(layout), false);
    }

    private static String normalizedContiguousBody(byte[] bytes) {
        var model = ClassFile.of().parse(bytes);
        return model.methods().stream().filter(method -> method.methodName().stringValue()
                .equals(CpuLossEmitter.CONTIGUOUS_INT_NAME)).findFirst().orElseThrow().code()
                .orElseThrow().elementStream().filter(Instruction.class::isInstance)
                .map(Instruction.class::cast).map(instruction -> instruction instanceof InvokeInstruction invoke
                        ? invoke.owner().asInternalName() + '.' + invoke.name().stringValue()
                        : instruction.opcode().name()).collect(java.util.stream.Collectors.joining("\n"));
    }

    private static short bf16(float value) {
        return ScalarValue.bfloat16(value).bfloat16Bits();
    }

    private static float[] decode(short[] values) {
        float[] result = new float[values.length];
        for (int index = 0; index < values.length; index++) {
            result[index] = Float.intBitsToFloat((values[index] & 0xffff) << 16);
        }
        return result;
    }
}

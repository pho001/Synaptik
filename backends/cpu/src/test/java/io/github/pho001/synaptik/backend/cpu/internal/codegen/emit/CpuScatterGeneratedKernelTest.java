package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScatterIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.backend.cpu.internal.reference.CpuScalarReferenceKernel;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.index.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;

import org.junit.jupiter.api.Test;

import java.lang.classfile.ClassFile;
import java.lang.classfile.ClassModel;
import java.lang.classfile.Instruction;
import java.lang.classfile.Opcode;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.classfile.instruction.InvokeInstruction;
import java.lang.classfile.instruction.NewMultiArrayInstruction;
import java.lang.classfile.instruction.NewObjectInstruction;
import java.lang.classfile.instruction.NewPrimitiveArrayInstruction;
import java.lang.classfile.instruction.NewReferenceArrayInstruction;
import java.lang.classfile.instruction.TypeCheckInstruction;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.*;

class CpuScatterGeneratedKernelTest {
    @Test
    void everyGeneratedScatterClassHasAStableDirectTypedShape() {
        var generated = new ArrayList<GeneratedScatterClass>();
        for (DataType dataType : DataType.values()) {
            var reductions =
                    dataType == DataType.BOOL
                            ? List.of(ScatterReduction.NONE)
                            : List.of(
                                    ScatterReduction.NONE,
                                    ScatterReduction.ADD,
                                    ScatterReduction.MUL,
                                    ScatterReduction.MIN,
                                    ScatterReduction.MAX);
            for (DataType indexType : List.of(DataType.INT32, DataType.INT64)) {
                for (boolean segments : List.of(false, true)) {
                    for (ScatterReduction reduction : reductions) {
                        generated.add(
                                generatedClass(
                                        "SCATTER_ELEMENTS",
                                        reduction,
                                        dataType,
                                        indexType,
                                        segments));
                        generated.add(
                                generatedClass(
                                        "SCATTER_ND", reduction, dataType, indexType, segments));
                    }
                    if (dataType != DataType.BOOL) {
                        generated.add(
                                generatedClass(
                                        "SCATTER_ADD",
                                        ScatterReduction.ADD,
                                        dataType,
                                        indexType,
                                        segments));
                    }
                }
            }
        }

        assertEquals(228, generated.size());
        for (GeneratedScatterClass generatedClass : generated) {
            assertDirectGeneratedShape(generatedClass);
        }
    }

    @Test
    void generatedClassesEmbedTypedScatterLoopsWithoutObjectExecutionBridges() {
        var none =
                code(
                        context(
                                new Operation(
                                        AxisScatterKind.SCATTER_ELEMENTS,
                                        new ScatterElementsAttrs(0, ScatterReduction.NONE)),
                                List.of(
                                        desc(DataType.FLOAT32, Shape.of(8)),
                                        desc(DataType.INT32, Shape.of(8)),
                                        desc(DataType.FLOAT32, Shape.of(8))),
                                desc(DataType.FLOAT32, Shape.of(8))));
        var add =
                code(
                        context(
                                new Operation(AxisScatterKind.SCATTER_ADD, new IndexAxisAttrs(0)),
                                List.of(
                                        desc(DataType.FLOAT32, Shape.of(8)),
                                        desc(DataType.INT32, Shape.of(8)),
                                        desc(DataType.FLOAT32, Shape.of(8))),
                                desc(DataType.FLOAT32, Shape.of(8))));
        var nd =
                code(
                        context(
                                new Operation(
                                        ScatterNdKind.SCATTER_ND,
                                        new ScatterNdAttrs(0, ScatterReduction.MAX)),
                                List.of(
                                        desc(DataType.INT64, Shape.of(2, 3)),
                                        desc(DataType.INT64, Shape.of(2, 1)),
                                        desc(DataType.INT64, Shape.of(2, 3))),
                                desc(DataType.INT64, Shape.of(2, 3))));
        var products =
                List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16).stream()
                        .map(
                                type ->
                                        code(
                                                context(
                                                        new Operation(
                                                                AxisScatterKind.SCATTER_ELEMENTS,
                                                                new ScatterElementsAttrs(
                                                                        0, ScatterReduction.MUL)),
                                                        List.of(
                                                                desc(type, Shape.of(3)),
                                                                desc(DataType.INT32, Shape.of(4)),
                                                                desc(type, Shape.of(4))),
                                                        desc(type, Shape.of(3)))))
                        .toList();
        assertAll(
                () -> assertTrue(opcodeCount(none, Opcode.FALOAD) >= 2),
                () -> assertTrue(opcodeCount(none, Opcode.IALOAD) > 0),
                () -> assertTrue(opcodeCount(none, Opcode.FASTORE) > 0),
                () -> assertTrue(opcodeCount(add, Opcode.FADD) > 0),
                () -> assertTrue(opcodeCount(nd, Opcode.LALOAD) > 0),
                () -> assertTrue(opcodeCount(nd, Opcode.LASTORE) > 0),
                () ->
                        assertTrue(
                                products.stream()
                                        .allMatch(
                                                body ->
                                                        opcodeCount(body, Opcode.LMUL) > 0
                                                                && opcodeCount(body, Opcode.LUSHR)
                                                                        > 0
                                                                && opcodeCount(body, Opcode.LSHL)
                                                                        > 0)),
                () ->
                        assertTrue(
                                products.stream()
                                        .flatMap(body -> invokes(body).stream())
                                        .anyMatch(
                                                call ->
                                                        call.owner()
                                                                        .asInternalName()
                                                                        .equals("java/lang/Math")
                                                                && call.name()
                                                                        .stringValue()
                                                                        .equals(
                                                                                "unsignedMultiplyHigh"))),
                () ->
                        assertTrue(
                                products.stream()
                                        .flatMap(body -> invokes(body).stream())
                                        .filter(
                                                call ->
                                                        call.owner()
                                                                .asInternalName()
                                                                .equals(
                                                                        "java/lang/foreign/MemorySegment"))
                                        .allMatch(
                                                call ->
                                                        Set.of("get", "set")
                                                                .contains(
                                                                        call.name()
                                                                                .stringValue()))),
                () ->
                        assertTrue(
                                java.util.stream.Stream.concat(
                                                java.util.stream.Stream.of(none, add, nd),
                                                products.stream())
                                        .flatMap(body -> invokes(body).stream())
                                        .noneMatch(
                                                call ->
                                                        call.name()
                                                                        .stringValue()
                                                                        .startsWith("execute")
                                                                || call.owner()
                                                                        .asInternalName()
                                                                        .equals(
                                                                                CpuScatterEmitter
                                                                                        .class
                                                                                        .getName()
                                                                                        .replace(
                                                                                                '.',
                                                                                                '/'))
                                                                || call.type()
                                                                        .stringValue()
                                                                        .contains(
                                                                                "Ljava/lang/Object;")
                                                                || call.owner()
                                                                        .asInternalName()
                                                                        .startsWith(
                                                                                "java/lang/reflect/")
                                                                || call.owner()
                                                                        .asInternalName()
                                                                        .equals("java/util/Map")
                                                                || call.owner()
                                                                        .asInternalName()
                                                                        .startsWith(
                                                                                "io/github/pho001/synaptik/runtime/"))),
                () ->
                        assertTrue(
                                java.util.stream.Stream.concat(
                                                java.util.stream.Stream.of(none, add, nd),
                                                products.stream())
                                        .allMatch(
                                                body ->
                                                        body.elementStream()
                                                                .noneMatch(
                                                                        element ->
                                                                                element
                                                                                        instanceof
                                                                                        java.lang
                                                                                                .classfile
                                                                                                .instruction
                                                                                                .NewObjectInstruction))));
    }

    @Test
    void generatedShapeCopiesBeforeOneUpdateBodyAndKeepsDenseExactProductGrouped() {
        var addition =
                code(
                        context(
                                new Operation(
                                        AxisScatterKind.SCATTER_ELEMENTS,
                                        new ScatterElementsAttrs(0, ScatterReduction.ADD)),
                                List.of(
                                        desc(DataType.FLOAT32, Shape.of(64)),
                                        desc(DataType.INT32, Shape.of(8)),
                                        desc(DataType.FLOAT32, Shape.of(8))),
                                desc(DataType.FLOAT32, Shape.of(64))));
        var product =
                code(
                        context(
                                new Operation(
                                        AxisScatterKind.SCATTER_ELEMENTS,
                                        new ScatterElementsAttrs(0, ScatterReduction.MUL)),
                                List.of(
                                        desc(DataType.FLOAT32, Shape.of(64)),
                                        desc(DataType.INT32, Shape.of(8)),
                                        desc(DataType.FLOAT32, Shape.of(8))),
                                desc(DataType.FLOAT32, Shape.of(64))));
        List<Opcode> additionOpcodes = opcodes(addition);
        List<Opcode> productOpcodes = opcodes(product);
        assertAll(
                () ->
                        assertTrue(
                                additionOpcodes.indexOf(Opcode.FALOAD)
                                        < additionOpcodes.indexOf(Opcode.FASTORE)),
                () ->
                        assertTrue(
                                additionOpcodes.indexOf(Opcode.FASTORE)
                                        < additionOpcodes.indexOf(Opcode.IALOAD)),
                () -> assertEquals(1, opcodeCount(product, Opcode.IALOAD)),
                () -> assertEquals(0, opcodeCount(product, Opcode.LASTORE)),
                () -> assertTrue(opcodeCount(product, Opcode.FASTORE) > 0),
                () ->
                        assertTrue(
                                invokes(product).stream()
                                        .filter(
                                                call ->
                                                        call.owner()
                                                                .asInternalName()
                                                                .equals(
                                                                        "java/lang/foreign/MemorySegment"))
                                        .allMatch(
                                                call ->
                                                        Set.of("get", "set")
                                                                .contains(
                                                                        call.name()
                                                                                .stringValue()))));
    }

    @Test
    void executesElementsReplacementFixedAddAndNdReduction() throws Throwable {
        long[] replacement = {-1, -1, -1, -1, -1, -1};
        invoke(
                context(
                        new Operation(
                                AxisScatterKind.SCATTER_ELEMENTS,
                                new ScatterElementsAttrs(1, ScatterReduction.NONE)),
                        List.of(
                                desc(DataType.INT64, Shape.of(2, 3)),
                                desc(DataType.INT32, Shape.of(2, 2)),
                                desc(DataType.INT64, Shape.of(2, 2))),
                        desc(DataType.INT64, Shape.of(2, 3))),
                List.of(
                        new long[] {10, 11, 12, 20, 21, 22},
                        new int[] {2, 0, 1, 2},
                        new long[] {90, 91, 80, 81},
                        replacement));
        int[] addition = new int[6];
        invoke(
                context(
                        new Operation(AxisScatterKind.SCATTER_ADD, new IndexAxisAttrs(1)),
                        List.of(
                                desc(DataType.INT32, Shape.of(2, 3)),
                                desc(DataType.INT64, Shape.of(2)),
                                desc(DataType.INT32, Shape.of(2, 2))),
                        desc(DataType.INT32, Shape.of(2, 3))),
                List.of(
                        new int[] {1, 2, 3, 4, 5, 6},
                        new long[] {2, 0},
                        new int[] {10, 20, 30, 40},
                        addition));
        float[] nd = new float[6];
        invoke(
                context(
                        new Operation(
                                ScatterNdKind.SCATTER_ND,
                                new ScatterNdAttrs(0, ScatterReduction.MAX)),
                        List.of(
                                desc(DataType.FLOAT32, Shape.of(2, 3)),
                                desc(DataType.INT32, Shape.of(3, 1)),
                                desc(DataType.FLOAT32, Shape.of(3, 3))),
                        desc(DataType.FLOAT32, Shape.of(2, 3))),
                List.of(
                        new float[] {1, 2, 3, 4, 5, 6},
                        new int[] {1, 0, 1},
                        new float[] {10, 1, 8, 7, 9, 0, 11, 4, 2},
                        nd));
        assertAll(
                () -> assertArrayEquals(new long[] {91, 11, 90, 20, 80, 81}, replacement),
                () -> assertArrayEquals(new int[] {21, 2, 13, 44, 5, 36}, addition),
                () -> assertArrayEquals(new float[] {7, 9, 3, 11, 5, 8}, nd));
    }

    @Test
    void coversAllRepresentedReplacementTypesAndBothIndexCarriers() throws Throwable {
        for (DataType type : DataType.values())
            for (DataType indexType : List.of(DataType.INT32, DataType.INT64)) {
                Object output = values(type, 0, 0, 0);
                Object indices = indexType == DataType.INT32 ? new int[] {2} : new long[] {2};
                invoke(
                        context(
                                new Operation(
                                        AxisScatterKind.SCATTER_ELEMENTS,
                                        new ScatterElementsAttrs(0, ScatterReduction.NONE)),
                                List.of(
                                        desc(type, Shape.of(3)),
                                        desc(indexType, Shape.of(1)),
                                        desc(type, Shape.of(1))),
                                desc(type, Shape.of(3))),
                        List.of(values(type, 1, 2, 3), indices, values(type, 9), output));
                assertCarrier(values(type, 1, 2, 9), output, type + "/" + indexType);
            }
    }

    @Test
    void coversEveryNumericReductionRowForBothIndexTypesAndBothReducingFamilies() throws Throwable {
        for (DataType type :
                List.of(
                        DataType.FLOAT64,
                        DataType.FLOAT32,
                        DataType.BFLOAT16,
                        DataType.INT32,
                        DataType.INT64)) {
            for (DataType indexType : List.of(DataType.INT32, DataType.INT64)) {
                Object indices = indexType == DataType.INT32 ? new int[] {1, 1} : new long[] {1, 1};
                for (ScatterReduction reduction :
                        List.of(
                                ScatterReduction.ADD,
                                ScatterReduction.MUL,
                                ScatterReduction.MIN,
                                ScatterReduction.MAX)) {
                    int middle =
                            switch (reduction) {
                                case ADD -> 15;
                                case MUL -> 105;
                                case MIN -> 3;
                                case MAX -> 7;
                                default -> throw new AssertionError(reduction);
                            };
                    Object elementsOutput = values(type, 0, 0, 0);
                    invoke(
                            context(
                                    new Operation(
                                            AxisScatterKind.SCATTER_ELEMENTS,
                                            new ScatterElementsAttrs(0, reduction)),
                                    List.of(
                                            desc(type, Shape.of(3)),
                                            desc(indexType, Shape.of(2)),
                                            desc(type, Shape.of(2))),
                                    desc(type, Shape.of(3))),
                            List.of(
                                    values(type, 2, 3, 4),
                                    indices,
                                    values(type, 5, 7),
                                    elementsOutput));
                    assertCarrier(
                            values(type, 2, middle, 4),
                            elementsOutput,
                            "SCATTER_ELEMENTS/" + type + "/" + indexType + "/" + reduction);

                    Object ndOutput = values(type, 0, 0, 0);
                    invoke(
                            context(
                                    new Operation(
                                            ScatterNdKind.SCATTER_ND,
                                            new ScatterNdAttrs(0, reduction)),
                                    List.of(
                                            desc(type, Shape.of(3)),
                                            desc(indexType, Shape.of(2, 1)),
                                            desc(type, Shape.of(2))),
                                    desc(type, Shape.of(3))),
                            List.of(values(type, 2, 3, 4), indices, values(type, 5, 7), ndOutput));
                    assertCarrier(
                            values(type, 2, middle, 4),
                            ndOutput,
                            "SCATTER_ND/" + type + "/" + indexType + "/" + reduction);
                }

                Object addOutput = values(type, 0, 0, 0);
                invoke(
                        context(
                                new Operation(AxisScatterKind.SCATTER_ADD, new IndexAxisAttrs(0)),
                                List.of(
                                        desc(type, Shape.of(3)),
                                        desc(indexType, Shape.of(2)),
                                        desc(type, Shape.of(2))),
                                desc(type, Shape.of(3))),
                        List.of(values(type, 2, 3, 4), indices, values(type, 5, 7), addOutput));
                assertCarrier(
                        values(type, 2, 15, 4), addOutput, "SCATTER_ADD/" + type + "/" + indexType);
            }
        }
    }

    @Test
    void exactFloatingProductUsesDeclaredScratchAndHandlesSpecialValues() throws Throwable {
        double[] output = new double[3];
        var plan =
                invoke(
                        context(
                                new Operation(
                                        AxisScatterKind.SCATTER_ELEMENTS,
                                        new ScatterElementsAttrs(0, ScatterReduction.MUL)),
                                List.of(
                                        desc(DataType.FLOAT64, Shape.of(3)),
                                        desc(DataType.INT32, Shape.of(4)),
                                        desc(DataType.FLOAT64, Shape.of(4))),
                                desc(DataType.FLOAT64, Shape.of(3))),
                        List.of(
                                new double[] {0.5, Double.POSITIVE_INFINITY, -0.0},
                                new int[] {0, 0, 1, 2},
                                new double[] {0.25, 8.0, 0.0, -2.0},
                                output));
        assertAll(
                () -> assertEquals(1.0, output[0]),
                () -> assertTrue(Double.isNaN(output[1])),
                () ->
                        assertEquals(
                                Double.doubleToRawLongBits(0.0),
                                Double.doubleToRawLongBits(output[2])),
                () ->
                        assertEquals(
                                CpuPartitionPreparationPlan.WorkspaceUse.SCATTER_PRODUCT,
                                plan.workspaceUse()),
                () ->
                        assertTrue(
                                plan.units()
                                        .getFirst()
                                        .portablePlan()
                                        .specialization()
                                        .scratchParameter()));
    }

    @Test
    void exactFloatingProductPreservesLargeOpposingExponentCancellation() throws Throwable {
        int updateCount = 2_001;
        int[] indices = new int[updateCount];
        double[] updates = new double[updateCount];
        Arrays.fill(updates, 0, 1_000, Math.scalb(1.0, 1000));
        Arrays.fill(updates, 1_000, updateCount, Math.scalb(1.0, -1000));
        double[] output = {Double.NaN};
        invoke(
                context(
                        new Operation(
                                AxisScatterKind.SCATTER_ELEMENTS,
                                new ScatterElementsAttrs(0, ScatterReduction.MUL)),
                        List.of(
                                desc(DataType.FLOAT64, Shape.of(1)),
                                desc(DataType.INT32, Shape.of(updateCount)),
                                desc(DataType.FLOAT64, Shape.of(updateCount))),
                        desc(DataType.FLOAT64, Shape.of(1))),
                List.of(new double[] {Math.scalb(1.0, 1000)}, indices, updates, output));
        assertEquals(Double.doubleToRawLongBits(1.0), Double.doubleToRawLongBits(output[0]));
    }

    @Test
    void exactFloatingProductMatchesIndependentSpecialValueAndRoundingOracle() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16)) {
            double minimumNormal =
                    type == DataType.FLOAT64
                            ? Double.MIN_NORMAL
                            : type == DataType.FLOAT32 ? Float.MIN_NORMAL : Math.scalb(1.0, -126);
            double minimumSubnormal =
                    type == DataType.FLOAT64
                            ? Double.MIN_VALUE
                            : type == DataType.FLOAT32 ? Float.MIN_VALUE : Math.scalb(1.0, -133);
            double maximumFinite =
                    type == DataType.FLOAT64
                            ? Double.MAX_VALUE
                            : type == DataType.FLOAT32
                                    ? Float.MAX_VALUE
                                    : Float.intBitsToFloat(0x7f7f0000);
            Object base =
                    floatingValues(
                            type,
                            maximumFinite,
                            minimumNormal,
                            minimumSubnormal,
                            minimumSubnormal,
                            -0.0,
                            Double.POSITIVE_INFINITY,
                            0.0,
                            Double.NaN,
                            -2.0);
            int[] indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 8};
            Object updates =
                    floatingValues(
                            type,
                            2.0,
                            0.5,
                            0.5,
                            1.5,
                            -2.0,
                            -2.0,
                            Double.POSITIVE_INFINITY,
                            2.0,
                            -3.0,
                            -4.0);
            assertGeneratedMatchesReference(type, ScatterReduction.MUL, base, indices, updates);
        }
    }

    @Test
    void floatingMinMaxNaNAndSignedZeroMatchIndependentRepresentedOracle() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16)) {
            Object base = floatingValues(type, 0.0, -0.0, Double.NaN, 1.0);
            int[] indices = {0, 1, 2, 3};
            Object updates = floatingValues(type, -0.0, 0.0, 2.0, Double.NaN);
            assertGeneratedMatchesReference(type, ScatterReduction.MIN, base, indices, updates);
            assertGeneratedMatchesReference(type, ScatterReduction.MAX, base, indices, updates);
        }
    }

    @Test
    void additionUsesLogicalContributionOrderAndRoundsBfloat16AfterEveryStep() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32)) {
            double large = type == DataType.FLOAT64 ? 1e300 : 1e30;
            Object output = emptyCarrier(type, 1);
            invoke(
                    context(
                            new Operation(
                                    AxisScatterKind.SCATTER_ELEMENTS,
                                    new ScatterElementsAttrs(0, ScatterReduction.ADD)),
                            List.of(
                                    desc(type, Shape.of(1)),
                                    desc(DataType.INT32, Shape.of(2)),
                                    desc(type, Shape.of(2))),
                            desc(type, Shape.of(1))),
                    List.of(
                            floatingValues(type, large),
                            new int[] {0, 0},
                            floatingValues(type, -large, 3.0),
                            output));
            assertRawCarrier(floatingValues(type, 3.0), output, type.name());
        }
        short[] bfloatOutput = new short[1];
        invoke(
                context(
                        new Operation(
                                AxisScatterKind.SCATTER_ELEMENTS,
                                new ScatterElementsAttrs(0, ScatterReduction.ADD)),
                        List.of(
                                desc(DataType.BFLOAT16, Shape.of(1)),
                                desc(DataType.INT32, Shape.of(3)),
                                desc(DataType.BFLOAT16, Shape.of(3))),
                        desc(DataType.BFLOAT16, Shape.of(1))),
                List.of(
                        floatingValues(DataType.BFLOAT16, 1.0),
                        new int[] {0, 0, 0},
                        floatingValues(
                                DataType.BFLOAT16,
                                Math.scalb(1.0, -8),
                                Math.scalb(1.0, -8),
                                Math.scalb(1.0, -7)),
                        bfloatOutput));
        assertArrayEquals(new short[] {(short) 0x3f81}, bfloatOutput);
    }

    @Test
    void integerAdditionAndMultiplicationUseModularWidth() throws Throwable {
        int[] intAdd = new int[1], intMultiply = new int[1];
        invoke(
                context(
                        new Operation(
                                AxisScatterKind.SCATTER_ELEMENTS,
                                new ScatterElementsAttrs(0, ScatterReduction.ADD)),
                        List.of(
                                desc(DataType.INT32, Shape.of(1)),
                                desc(DataType.INT32, Shape.of(1)),
                                desc(DataType.INT32, Shape.of(1))),
                        desc(DataType.INT32, Shape.of(1))),
                List.of(new int[] {Integer.MAX_VALUE}, new int[] {0}, new int[] {1}, intAdd));
        invoke(
                context(
                        new Operation(
                                AxisScatterKind.SCATTER_ELEMENTS,
                                new ScatterElementsAttrs(0, ScatterReduction.MUL)),
                        List.of(
                                desc(DataType.INT32, Shape.of(1)),
                                desc(DataType.INT32, Shape.of(1)),
                                desc(DataType.INT32, Shape.of(1))),
                        desc(DataType.INT32, Shape.of(1))),
                List.of(new int[] {Integer.MAX_VALUE}, new int[] {0}, new int[] {2}, intMultiply));
        long[] longAdd = new long[1], longMultiply = new long[1];
        invoke(
                context(
                        new Operation(
                                AxisScatterKind.SCATTER_ELEMENTS,
                                new ScatterElementsAttrs(0, ScatterReduction.ADD)),
                        List.of(
                                desc(DataType.INT64, Shape.of(1)),
                                desc(DataType.INT64, Shape.of(1)),
                                desc(DataType.INT64, Shape.of(1))),
                        desc(DataType.INT64, Shape.of(1))),
                List.of(new long[] {Long.MAX_VALUE}, new long[] {0}, new long[] {1}, longAdd));
        invoke(
                context(
                        new Operation(
                                AxisScatterKind.SCATTER_ELEMENTS,
                                new ScatterElementsAttrs(0, ScatterReduction.MUL)),
                        List.of(
                                desc(DataType.INT64, Shape.of(1)),
                                desc(DataType.INT64, Shape.of(1)),
                                desc(DataType.INT64, Shape.of(1))),
                        desc(DataType.INT64, Shape.of(1))),
                List.of(new long[] {Long.MAX_VALUE}, new long[] {0}, new long[] {2}, longMultiply));
        assertAll(
                () -> assertArrayEquals(new int[] {Integer.MIN_VALUE}, intAdd),
                () -> assertArrayEquals(new int[] {-2}, intMultiply),
                () -> assertArrayEquals(new long[] {Long.MIN_VALUE}, longAdd),
                () -> assertArrayEquals(new long[] {-2}, longMultiply));
    }

    @Test
    void packedGeometryCanBeReusedForTheSamePartialRange() throws Throwable {
        var operation =
                new Operation(
                        AxisScatterKind.SCATTER_ELEMENTS,
                        new ScatterElementsAttrs(0, ScatterReduction.ADD));
        var inputs =
                List.of(
                        desc(DataType.INT32, Shape.of(4)),
                        desc(DataType.INT32, Shape.of(2)),
                        desc(DataType.INT32, Shape.of(2)));
        var plan =
                new CpuPartitionPreparer()
                        .analyze(context(operation, inputs, desc(DataType.INT32, Shape.of(4))))
                        .plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        var artifact =
                generator.defineClassBytes(
                        route.specialization(),
                        generator.generateClassBytes(route.specialization(), route.kernelIr()));
        int[] output = {-7, -7, -7, -7};
        var arguments =
                new ArrayList<Object>(
                        List.of(
                                new int[] {1, 2, 3, 4},
                                new int[] {1, 2},
                                new int[] {10, 20},
                                output));
        long[] geometry = plan.scatterGeometry().orElseThrow().pack(new long[4], 1, 3, 0);
        arguments.add(geometry);
        arguments.add(1L);
        arguments.add(3L);
        artifact.entryPoint().invokeWithArguments(arguments);
        artifact.entryPoint().invokeWithArguments(arguments);
        assertArrayEquals(new int[] {-7, 12, 23, -7}, output);
    }

    @Test
    void rawReplacementEntryIsLastWinsButWritesOnlyItsOwnedRange() throws Throwable {
        var operation =
                new Operation(
                        AxisScatterKind.SCATTER_ELEMENTS,
                        new ScatterElementsAttrs(0, ScatterReduction.NONE));
        var inputs =
                List.of(
                        desc(DataType.INT32, Shape.of(4)),
                        desc(DataType.INT32, Shape.of(3)),
                        desc(DataType.INT32, Shape.of(3)));
        var plan =
                new CpuPartitionPreparer()
                        .analyze(context(operation, inputs, desc(DataType.INT32, Shape.of(4))))
                        .plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        var artifact =
                generator.defineClassBytes(
                        route.specialization(),
                        generator.generateClassBytes(route.specialization(), route.kernelIr()));
        int[] output = {-7, -7, -7, -7};
        var arguments =
                new ArrayList<Object>(
                        List.of(
                                new int[] {1, 2, 3, 4},
                                new int[] {1, 1, 3},
                                new int[] {10, 20, 30},
                                output));
        arguments.add(plan.scatterGeometry().orElseThrow().pack(new long[4], 1, 3, 0));
        arguments.add(1L);
        arguments.add(3L);
        artifact.entryPoint().invokeWithArguments(arguments);
        assertArrayEquals(new int[] {-7, 20, 3, -7}, output);
    }

    @Test
    void executesArbitraryResolvedLayoutsAcrossMixedHeapAndNativeCarriers() throws Throwable {
        Shape dataShape = Shape.of(2, 3), updateShape = Shape.of(2, 2);
        var data = desc(DataType.INT64, dataShape, new long[] {5, 1}, 2);
        var indices = desc(DataType.INT32, updateShape, new long[] {3, 1}, 1);
        var updates = desc(DataType.INT64, updateShape, new long[] {4, 1}, 2);
        var outputDescriptor = desc(DataType.INT64, dataShape, new long[] {6, 1}, 3);
        var base =
                CpuScatterLoweringTest.context(
                        new Operation(
                                AxisScatterKind.SCATTER_ELEMENTS,
                                new ScatterElementsAttrs(1, ScatterReduction.NONE)),
                        List.of(0, 1, 2),
                        List.of(data, indices, updates),
                        outputDescriptor);
        var context =
                new PrepareContext<>(
                        base.partition(),
                        base.nodes(),
                        base.values(),
                        base.memoryRequirements(),
                        Map.of(),
                        new CpuPartitionAnalysisInputs(
                                false,
                                List.of(
                                        CarrierAccess.MEMORY_SEGMENT, CarrierAccess.INT_ARRAY,
                                        CarrierAccess.MEMORY_SEGMENT, CarrierAccess.LONG_ARRAY)));
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment dataSegment = arena.allocate(10L * Long.BYTES, Long.BYTES);
            MemorySegment updateSegment = arena.allocate(8L * Long.BYTES, Long.BYTES);
            long[] dataAddresses = {2, 3, 4, 7, 8, 9};
            long[] dataValues = {10, 11, 12, 20, 21, 22};
            for (int i = 0; i < dataValues.length; i++)
                dataSegment.set(
                        ValueLayout.JAVA_LONG, dataAddresses[i] * Long.BYTES, dataValues[i]);
            long[] updateAddresses = {2, 3, 6, 7};
            long[] updateValues = {90, 91, 80, 81};
            for (int i = 0; i < updateValues.length; i++)
                updateSegment.set(
                        ValueLayout.JAVA_LONG, updateAddresses[i] * Long.BYTES, updateValues[i]);
            int[] indexCarrier = {-9, 2, 0, -9, 1, 2};
            long[] output = new long[12];
            Arrays.fill(output, -7);
            invoke(context, List.of(dataSegment, indexCarrier, updateSegment, output));
            assertArrayEquals(new long[] {-7, -7, -7, 91, 11, 90, -7, -7, -7, 20, 80, 81}, output);
        }
    }

    @Test
    void executesLegalScalarIndexShapeAndZeroDomains() throws Throwable {
        int[] scalarOutput = new int[3];
        invoke(
                context(
                        new Operation(AxisScatterKind.SCATTER_ADD, new IndexAxisAttrs(0)),
                        List.of(
                                desc(DataType.INT32, Shape.of(3)),
                                desc(DataType.INT64, Shape.scalar()),
                                desc(DataType.INT32, Shape.scalar())),
                        desc(DataType.INT32, Shape.of(3))),
                List.of(new int[] {1, 2, 3}, new long[] {1}, new int[] {5}, scalarOutput));
        assertArrayEquals(new int[] {1, 7, 3}, scalarOutput);

        var zeroPlan =
                invoke(
                        context(
                                new Operation(
                                        AxisScatterKind.SCATTER_ELEMENTS,
                                        new ScatterElementsAttrs(0, ScatterReduction.MUL)),
                                List.of(
                                        desc(DataType.FLOAT32, Shape.of(0)),
                                        desc(DataType.INT32, Shape.of(0)),
                                        desc(DataType.FLOAT32, Shape.of(0))),
                                desc(DataType.FLOAT32, Shape.of(0))),
                        List.of(new float[0], new int[0], new float[0], new float[0]));
        assertAll(
                () -> assertEquals(0, zeroPlan.elementCount()),
                () -> assertTrue(zeroPlan.workspaceDeclaration().isEmpty()));
    }

    private static CpuPartitionPreparationPlan invoke(
            PrepareContext<CpuPartitionAnalysisInputs> context, List<Object> carriers)
            throws Throwable {
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        var artifact =
                generator.defineClassBytes(
                        route.specialization(),
                        generator.generateClassBytes(route.specialization(), route.kernelIr()));
        long[] bases = new long[carriers.size()];
        long[] geometry =
                plan.scatterGeometry().orElseThrow().pack(bases, 0, plan.elementCount(), 0);
        var args = new ArrayList<Object>(carriers);
        try (Arena arena = Arena.ofConfined()) {
            if (plan.workspaceDeclaration().isPresent())
                args.add(arena.allocate(plan.workspaceDeclaration().orElseThrow().byteSize(), 8));
            args.add(geometry);
            args.add(0L);
            args.add(plan.elementCount());
            artifact.entryPoint().invokeWithArguments(args);
        }
        return plan;
    }

    private static java.lang.classfile.CodeModel code(
            PrepareContext<CpuPartitionAnalysisInputs> context) {
        return generatedModel(context).methods().getFirst().code().orElseThrow();
    }

    private static ClassModel generatedModel(PrepareContext<CpuPartitionAnalysisInputs> context) {
        var route =
                new CpuPartitionPreparer()
                        .analyze(context)
                        .plan()
                        .units()
                        .getFirst()
                        .portablePlan();
        return ClassFile.of()
                .parse(
                        new CpuClassFileKernelGenerator()
                                .generateClassBytes(route.specialization(), route.kernelIr()));
    }

    private static GeneratedScatterClass generatedClass(
            String family,
            ScatterReduction reduction,
            DataType dataType,
            DataType indexType,
            boolean segments) {
        Operation operation =
                switch (family) {
                    case "SCATTER_ELEMENTS" ->
                            new Operation(
                                    AxisScatterKind.SCATTER_ELEMENTS,
                                    new ScatterElementsAttrs(0, reduction));
                    case "SCATTER_ADD" ->
                            new Operation(AxisScatterKind.SCATTER_ADD, new IndexAxisAttrs(0));
                    case "SCATTER_ND" ->
                            new Operation(
                                    ScatterNdKind.SCATTER_ND, new ScatterNdAttrs(0, reduction));
                    default -> throw new AssertionError(family);
                };
        Shape dataShape = Shape.of(3);
        Shape indexShape = family.equals("SCATTER_ND") ? Shape.of(2, 1) : Shape.of(2);
        Shape updateShape = Shape.of(2);
        var context =
                context(
                        operation,
                        List.of(
                                desc(dataType, dataShape),
                                desc(indexType, indexShape),
                                desc(dataType, updateShape)),
                        desc(dataType, dataShape),
                        segments);
        return new GeneratedScatterClass(
                family, reduction, dataType, indexType, segments, generatedModel(context));
    }

    private static void assertDirectGeneratedShape(GeneratedScatterClass generated) {
        String label = generated.label();
        var method = generated.model().methods().getFirst();
        var code = method.code().orElseThrow();
        var invokes = invokes(code);
        assertFalse(
                method.methodTypeSymbol().descriptorString().contains("Ljava/lang/Object;"), label);
        var forbiddenInstructions =
                code.elementStream()
                        .filter(
                                element ->
                                        element instanceof TypeCheckInstruction check
                                                        && (check.type()
                                                                        .asInternalName()
                                                                        .equals("java/lang/Object")
                                                                || check.type()
                                                                        .asInternalName()
                                                                        .startsWith("["))
                                                || element instanceof NewObjectInstruction
                                                || element instanceof NewPrimitiveArrayInstruction
                                                || element instanceof NewReferenceArrayInstruction
                                                || element instanceof NewMultiArrayInstruction)
                        .toList();
        assertTrue(forbiddenInstructions.isEmpty(), label + ": " + forbiddenInstructions);
        assertTrue(
                java.util.stream.StreamSupport.stream(
                                generated.model().constantPool().spliterator(), false)
                        .filter(MemberRefEntry.class::isInstance)
                        .map(MemberRefEntry.class::cast)
                        .noneMatch(
                                reference ->
                                        reference
                                                .owner()
                                                .asInternalName()
                                                .equals(
                                                        CpuScatterEmitter.class
                                                                .getName()
                                                                .replace('.', '/'))),
                label);
        assertTrue(
                invokes.stream()
                        .noneMatch(
                                call ->
                                        call.type().stringValue().contains("Ljava/lang/Object;")
                                                || call.owner()
                                                        .asInternalName()
                                                        .startsWith("java/lang/reflect/")
                                                || call.owner()
                                                        .asInternalName()
                                                        .startsWith("java/util/Map")
                                                || call.owner()
                                                        .asInternalName()
                                                        .startsWith(
                                                                "io/github/pho001/synaptik/runtime/")
                                                || call.owner()
                                                        .asInternalName()
                                                        .startsWith(
                                                                "io/github/pho001/synaptik/backend/cpu/internal/cache/")),
                label);

        if (generated.segments()) {
            assertTrue(
                    invokes.stream()
                            .anyMatch(
                                    call ->
                                            call.owner()
                                                            .asInternalName()
                                                            .equals(
                                                                    "java/lang/foreign/MemorySegment")
                                                    && call.name().stringValue().equals("get")),
                    label);
            assertTrue(
                    invokes.stream()
                            .anyMatch(
                                    call ->
                                            call.owner()
                                                            .asInternalName()
                                                            .equals(
                                                                    "java/lang/foreign/MemorySegment")
                                                    && call.name().stringValue().equals("set")),
                    label);
        } else {
            assertTrue(opcodeCount(code, loadOpcode(generated.dataType())) > 0, label);
            assertTrue(opcodeCount(code, storeOpcode(generated.dataType())) > 0, label);
            assertTrue(
                    opcodeCount(
                                    code,
                                    generated.indexType() == DataType.INT32
                                            ? Opcode.IALOAD
                                            : Opcode.LALOAD)
                            > 0,
                    label);
        }
        if (generated.reduction() == ScatterReduction.ADD) {
            assertTrue(
                    opcodeCount(
                                    code,
                                    switch (generated.dataType()) {
                                        case FLOAT64 -> Opcode.DADD;
                                        case FLOAT32, BFLOAT16 -> Opcode.FADD;
                                        case INT32 -> Opcode.IADD;
                                        case INT64 -> Opcode.LADD;
                                        case BOOL ->
                                                throw new AssertionError("BOOL ADD is unsupported");
                                    })
                            > 0,
                    label);
        }
        if (generated.dataType() == DataType.BFLOAT16
                && generated.reduction() != ScatterReduction.NONE
                && generated.reduction() != ScatterReduction.MUL) {
            assertTrue(
                    invokes.stream()
                            .anyMatch(
                                    call ->
                                            call.owner().asInternalName().equals("java/lang/Float")
                                                    && call.name()
                                                            .stringValue()
                                                            .equals("floatToRawIntBits")),
                    label);
            assertTrue(
                    invokes.stream()
                            .noneMatch(
                                    call ->
                                            call.name().stringValue().equals("values")
                                                    || call.name().stringValue().equals("ordinal")),
                    label);
        }
    }

    private static Opcode loadOpcode(DataType type) {
        return switch (type) {
            case FLOAT64 -> Opcode.DALOAD;
            case FLOAT32 -> Opcode.FALOAD;
            case BFLOAT16 -> Opcode.SALOAD;
            case INT32 -> Opcode.IALOAD;
            case INT64 -> Opcode.LALOAD;
            case BOOL -> Opcode.BALOAD;
        };
    }

    private static Opcode storeOpcode(DataType type) {
        return switch (type) {
            case FLOAT64 -> Opcode.DASTORE;
            case FLOAT32 -> Opcode.FASTORE;
            case BFLOAT16 -> Opcode.SASTORE;
            case INT32 -> Opcode.IASTORE;
            case INT64 -> Opcode.LASTORE;
            case BOOL -> Opcode.BASTORE;
        };
    }

    private static List<InvokeInstruction> invokes(java.lang.classfile.CodeModel code) {
        return code.elementStream()
                .filter(InvokeInstruction.class::isInstance)
                .map(InvokeInstruction.class::cast)
                .toList();
    }

    private static long opcodeCount(java.lang.classfile.CodeModel code, Opcode opcode) {
        return code.elementStream()
                .filter(Instruction.class::isInstance)
                .map(Instruction.class::cast)
                .filter(value -> value.opcode() == opcode)
                .count();
    }

    private static List<Opcode> opcodes(java.lang.classfile.CodeModel code) {
        return code.elementStream()
                .filter(Instruction.class::isInstance)
                .map(Instruction.class::cast)
                .map(Instruction::opcode)
                .toList();
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> context(
            Operation operation,
            List<io.github.pho001.synaptik.model.tensor.TensorDescriptor> inputs,
            io.github.pho001.synaptik.model.tensor.TensorDescriptor output) {
        return context(operation, inputs, output, false);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> context(
            Operation operation,
            List<io.github.pho001.synaptik.model.tensor.TensorDescriptor> inputs,
            io.github.pho001.synaptik.model.tensor.TensorDescriptor output,
            boolean segments) {
        var base = CpuScatterLoweringTest.context(operation, List.of(0, 1, 2), inputs, output);
        var carriers = new ArrayList<CarrierAccess>();
        for (var input : inputs)
            carriers.add(segments ? CarrierAccess.MEMORY_SEGMENT : heap(input.dataType()));
        carriers.add(segments ? CarrierAccess.MEMORY_SEGMENT : heap(output.dataType()));
        return new PrepareContext<>(
                base.partition(),
                base.nodes(),
                base.values(),
                base.memoryRequirements(),
                Map.of(),
                new CpuPartitionAnalysisInputs(false, carriers));
    }

    private static CarrierAccess heap(DataType t) {
        return switch (t) {
            case FLOAT64 -> CarrierAccess.DOUBLE_ARRAY;
            case FLOAT32 -> CarrierAccess.FLOAT_ARRAY;
            case BFLOAT16 -> CarrierAccess.SHORT_ARRAY;
            case INT32 -> CarrierAccess.INT_ARRAY;
            case INT64 -> CarrierAccess.LONG_ARRAY;
            case BOOL -> CarrierAccess.BYTE_ARRAY;
        };
    }

    private static io.github.pho001.synaptik.model.tensor.TensorDescriptor desc(
            DataType t, Shape s) {
        return CpuScatterLoweringTest.desc(t, s);
    }

    private static TensorDescriptor desc(DataType type, Shape shape, long[] strides, long offset) {
        return new TensorDescriptor(
                type, shape, Optional.of(LayoutDescriptor.of(shape, strides, offset, true)), false);
    }

    private static Object values(DataType t, int... v) {
        return switch (t) {
            case FLOAT64 -> Arrays.stream(v).asDoubleStream().toArray();
            case FLOAT32 -> {
                float[] x = new float[v.length];
                for (int i = 0; i < v.length; i++) x[i] = v[i];
                yield x;
            }
            case BFLOAT16 -> {
                short[] x = new short[v.length];
                for (int i = 0; i < v.length; i++)
                    x[i] = (short) (Float.floatToRawIntBits(v[i]) >>> 16);
                yield x;
            }
            case INT32 -> v.clone();
            case INT64 -> Arrays.stream(v).asLongStream().toArray();
            case BOOL -> {
                byte[] x = new byte[v.length];
                for (int i = 0; i < v.length; i++) x[i] = (byte) (v[i] & 1);
                yield x;
            }
        };
    }

    private static void assertCarrier(Object expected, Object actual, String message) {
        if (expected instanceof double[] x) assertArrayEquals(x, (double[]) actual, message);
        else if (expected instanceof float[] x) assertArrayEquals(x, (float[]) actual, message);
        else if (expected instanceof short[] x) assertArrayEquals(x, (short[]) actual, message);
        else if (expected instanceof int[] x) assertArrayEquals(x, (int[]) actual, message);
        else if (expected instanceof long[] x) assertArrayEquals(x, (long[]) actual, message);
        else assertArrayEquals((byte[]) expected, (byte[]) actual, message);
    }

    private static void assertGeneratedMatchesReference(
            DataType type, ScatterReduction reduction, Object base, int[] indices, Object updates)
            throws Throwable {
        int outputCount = java.lang.reflect.Array.getLength(base);
        int updateCount = indices.length;
        var operation =
                new Operation(
                        AxisScatterKind.SCATTER_ELEMENTS, new ScatterElementsAttrs(0, reduction));
        var inputs =
                List.of(
                        desc(type, Shape.of(outputCount)),
                        desc(DataType.INT32, Shape.of(updateCount)),
                        desc(type, Shape.of(updateCount)));
        var prepare = context(operation, inputs, desc(type, Shape.of(outputCount)));
        var lowered = new CpuPartitionLowering().lower(prepare);
        Object expected = emptyCarrier(type, outputCount);
        CpuScalarReferenceKernel.execute(
                (CpuScatterIr) lowered.portableKernelIr(),
                lowered.scatterGeometry().orElseThrow(),
                List.of(
                        argument(type, base, true),
                        argument(DataType.INT32, indices, true),
                        argument(type, updates, true),
                        argument(type, expected, false)),
                0,
                outputCount);
        Object actual = emptyCarrier(type, outputCount);
        invoke(prepare, List.of(base, indices, updates, actual));
        assertRawCarrier(expected, actual, type.name());
    }

    private static Object floatingValues(DataType type, double... values) {
        return switch (type) {
            case FLOAT64 -> values.clone();
            case FLOAT32 -> {
                float[] result = new float[values.length];
                for (int i = 0; i < values.length; i++) result[i] = (float) values[i];
                yield result;
            }
            case BFLOAT16 -> {
                short[] result = new short[values.length];
                for (int i = 0; i < values.length; i++) {
                    int bits = Float.floatToRawIntBits((float) values[i]);
                    int upper = bits >>> 16, lower = bits & 0xffff;
                    if ((bits & 0x7f800000) == 0x7f800000 && (bits & 0x7fffff) != 0) upper |= 0x40;
                    else if (lower > 0x8000 || lower == 0x8000 && (upper & 1) != 0) upper++;
                    result[i] = (short) upper;
                }
                yield result;
            }
            default -> throw new IllegalArgumentException("not floating: " + type);
        };
    }

    private static Object emptyCarrier(DataType type, int count) {
        return switch (type) {
            case FLOAT64 -> new double[count];
            case FLOAT32 -> new float[count];
            case BFLOAT16 -> new short[count];
            default -> throw new IllegalArgumentException("not floating: " + type);
        };
    }

    private static CpuBufferArgument argument(DataType type, Object carrier, boolean readOnly) {
        long bytes =
                Math.multiplyExact(
                        (long) java.lang.reflect.Array.getLength(carrier), type.byteWidth());
        return switch (type) {
            case FLOAT64 -> new CpuBufferArgument.Doubles((double[]) carrier, 0, bytes, readOnly);
            case FLOAT32 -> new CpuBufferArgument.Floats((float[]) carrier, 0, bytes, readOnly);
            case BFLOAT16 -> new CpuBufferArgument.Shorts((short[]) carrier, 0, bytes, readOnly);
            case INT32 -> new CpuBufferArgument.Ints((int[]) carrier, 0, bytes, readOnly);
            default -> throw new IllegalArgumentException("unsupported: " + type);
        };
    }

    private static void assertRawCarrier(Object expected, Object actual, String message) {
        if (expected instanceof double[] x) {
            double[] y = (double[]) actual;
            for (int i = 0; i < x.length; i++)
                assertEquals(
                        Double.doubleToRawLongBits(x[i]),
                        Double.doubleToRawLongBits(y[i]),
                        message + "[" + i + "]");
        } else if (expected instanceof float[] x) {
            float[] y = (float[]) actual;
            for (int i = 0; i < x.length; i++)
                assertEquals(
                        Float.floatToRawIntBits(x[i]),
                        Float.floatToRawIntBits(y[i]),
                        message + "[" + i + "]");
        } else assertArrayEquals((short[]) expected, (short[]) actual, message);
    }

    private record GeneratedScatterClass(
            String family,
            ScatterReduction reduction,
            DataType dataType,
            DataType indexType,
            boolean segments,
            ClassModel model) {
        String label() {
            return family
                    + "/"
                    + reduction
                    + "/"
                    + dataType
                    + "/"
                    + indexType
                    + "/"
                    + (segments ? "segment" : "heap");
        }
    }
}

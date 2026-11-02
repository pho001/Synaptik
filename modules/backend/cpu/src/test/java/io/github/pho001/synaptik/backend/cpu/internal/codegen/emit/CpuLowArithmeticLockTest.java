package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStore;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionLowering;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.io.ByteArrayInputStream;
import java.lang.invoke.MethodHandles;
import java.lang.ref.WeakReference;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.zip.GZIPInputStream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

/** Current admission rejects authentic pre-lock executable payloads, never regenerated stand-ins. */
class CpuLowArithmeticLockTest {
    @TempDir Path root;

    @Test void promotedMatrixChecksEachOperandAndAuthenticatesTypedEnvelope() {
        var shape = io.github.pho001.synaptik.model.shape.Shape.of(2, 2);
        var plan = new io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer()
                .analyze(io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMatmulLoweringTest
                        .context(DataType.FLOAT32, shape, shape, shape)).plan();
        var route = plan.units().getFirst().portablePlan();
        var original = route.specialization().matmulIr().orElseThrow();
        var generator = new CpuClassFileKernelGenerator();
        byte[] allowedBytes = generator.generateClassBytes(route.specialization(), route.kernelIr());
        for (var type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
            for (boolean lowLeft : List.of(false, true)) {
                var matrix = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr(
                        lowLeft ? type : DataType.FLOAT32, lowLeft ? DataType.FLOAT32 : type,
                        DataType.FLOAT32, original.realization(), original.epilogue(),
                        original.preferredSpeciesBitSize(), original.numericalForm(),
                        original.inputAccesses(), original.outputAccess());
                var ir = matrix.encodedKernelIr();
                var spec = new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                        CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                        List.of(matrix.leftType(), matrix.rightType(), DataType.FLOAT32),
                        Collections.nCopies(3, CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT),
                        0, -1, List.of(), false, 54, Optional.of(matrix));
                assertThrows(IllegalArgumentException.class,
                        () -> CpuPartitionLowering.requirePermittedLowPrecisionUse(spec, ir));
                assertThrows(IllegalArgumentException.class, () -> generator.generateClassBytes(spec, ir));
                assertThrows(IllegalArgumentException.class,
                        () -> generator.defineClassBytes(spec, ir, allowedBytes));
                assertThrows(IllegalArgumentException.class,
                        () -> new CpuGeneratedKernelArtifactStore(Optional.empty()).loadOrGenerate(spec, ir));
            }
        }
        var altered = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr(
                original.leftType(), original.rightType(), original.resultType(), original.realization(),
                new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr.Epilogue(
                        io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr.Epilogue.AddInputOrder.NONE,
                        io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr.Epilogue.Terminal.RELU, null),
                original.preferredSpeciesBitSize(), original.numericalForm(),
                original.inputAccesses(), original.outputAccess());
        var mismatched = new CpuKernelSpecialization(route.specialization().loweringFingerprint(),
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR, List.of(DataType.FLOAT32, DataType.FLOAT32, DataType.FLOAT32),
                route.specialization().carrierPattern(), 0, -1, List.of(), false, 54, Optional.of(altered));
        assertThrows(IllegalArgumentException.class,
                () -> generator.generateClassBytes(mismatched, route.kernelIr()),
                "ordinary F32 also requires exact typed/envelope agreement");
    }

    /**
     * The captured classes were emitted and defined by the accepted pre-CPU0078 source at
     * 07a281edd3086efde2853211b71b48288db027e1 plus the CPU0077 source manifest.
     * BF16's successful capture preceded a failed first F16 projection; the separate corrected
     * F16 capture uses projection 68. No current production admission bypass creates these bytes.
     */
    @Test void authenticOldAddRejectsBeforeFreshDefinitionWeakInternAndTrustedRootReuse() throws Exception {
        for (var type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
            var ir = binary(type, CpuPointwiseOpcode.ADD);
            var spec = specialization(type, ir);
            byte[] bytes = payload(type, "class");
            assertArrayEquals(payload(type, "identity"), spec.classIdentityBytes());
            assertArrayEquals(payload(type, "compatibility"), spec.compatibilityBytes());
            assertEquals(new String(payload(type, "key"), StandardCharsets.US_ASCII).trim(), spec.structuralKey());
            var generator = new CpuClassFileKernelGenerator();
            // Public Class-File validation is structural only; execution admission stays current.
            assertTrue(java.lang.classfile.ClassFile.of().verify(bytes).isEmpty());
            assertThrows(IllegalArgumentException.class, () -> generator.generateClassBytes(spec, ir));
            assertThrows(IllegalArgumentException.class, () -> generator.defineClassBytes(spec, ir, bytes));

            var envelope = CpuGeneratedKernelArtifactStore.class.getDeclaredMethod("envelope", String.class, byte[].class, byte[].class);
            envelope.setAccessible(true);
            byte[] persisted = (byte[]) envelope.invoke(null, spec.structuralKey(), payload(type, "compatibility"), bytes);
            Path path = root.resolve(spec.structuralKey() + ".artifact");
            Files.write(path, persisted);
            var store = new CpuGeneratedKernelArtifactStore(Optional.of(root));
            assertThrows(IllegalArgumentException.class, () -> store.loadOrGenerate(spec, ir));
            assertArrayEquals(persisted, Files.readAllBytes(path), "policy rejection never overwrites or regenerates");

            // Model a genuinely live artifact loaded before the policy cutover. This is test-only
            // hidden definition of the captured payload, not a selectable production generator.
            var lookup = MethodHandles.lookup().defineHiddenClass(bytes, false, MethodHandles.Lookup.ClassOption.NESTMATE);
            var old = new CpuGeneratedKernel(spec, lookup,
                    lookup.findStatic(lookup.lookupClass(), "invoke", spec.entryType()), bytes);
            var field = CpuGeneratedKernelArtifactStore.class.getDeclaredField("LOADED");
            field.setAccessible(true);
            @SuppressWarnings("unchecked")
            var loaded = (Map<String, WeakReference<CpuGeneratedKernel>>) field.get(null);
            var reference = new WeakReference<>(old);
            var previous = loaded.put(spec.structuralKey(), reference);
            try {
                assertThrows(IllegalArgumentException.class, () -> store.loadOrGenerate(spec, ir));
                assertSame(old, reference.get(), "rejection occurs even while the genuine old entry is live");
                assertArrayEquals(persisted, Files.readAllBytes(path));
            } finally {
                loaded.remove(spec.structuralKey(), reference);
                if (previous != null) loaded.put(spec.structuralKey(), previous);
            }
        }
    }

    @Test void everyNumericalOpcodeRejectsLowInputsWhileExactOpcodesRetainAdmission() {
        for (var type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
            for (var opcode : List.of(CpuPointwiseOpcode.ADD, CpuPointwiseOpcode.SUB,
                    CpuPointwiseOpcode.MUL, CpuPointwiseOpcode.DIV, CpuPointwiseOpcode.POW)) {
                assertThrows(IllegalArgumentException.class,
                        () -> CpuPartitionLowering.requirePermittedLowPrecisionUse(binary(type, opcode)), opcode.toString());
            }
            for (var opcode : List.of(CpuPointwiseOpcode.MIN, CpuPointwiseOpcode.MAX))
                assertDoesNotThrow(() -> CpuPartitionLowering.requirePermittedLowPrecisionUse(binary(type, opcode)));
        }
    }

    private static CpuKernelIr binary(DataType type, CpuPointwiseOpcode opcode) {
        var read = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ, CpuAccessPlan.Regime.DENSE_LINEAR,
                1, List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
        var write = new CpuAccessPlan(CpuAccessPlan.AccessKind.WRITE, CpuAccessPlan.Regime.DENSE_LINEAR,
                1, List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
        return new CpuKernelIr(List.of(new CpuKernelIr.Value(0, type, CpuKernelIr.Value.Kind.INPUT, read),
                new CpuKernelIr.Value(1, type, CpuKernelIr.Value.Kind.INPUT, read),
                new CpuKernelIr.Value(2, type, CpuKernelIr.Value.Kind.OUTPUT, write)),
                List.of(new CpuKernelIr.Instruction(opcode, List.of(0, 1), 2)),
                new CpuKernelIr.Loop("start", "end"), List.of(new CpuKernelIr.Store(2, 0)));
    }
    @Test void sumDerivedCountWritesRejectBothTypedAndInstructionFreeAdmission() {
        var write = new CpuAccessPlan(CpuAccessPlan.AccessKind.WRITE, CpuAccessPlan.Regime.DENSE_LINEAR,
                1, List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
        for (var type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
            var count = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConstantWriteIr(
                    new CpuKernelIr.ScalarImmediate(type, type == DataType.BFLOAT16 ? 0x3f80 : 0x3c00), write);
            var envelope = count.encodedKernelIr();
            var spec = new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(envelope.structuralKey()),
                    CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR, List.of(type),
                    List.of(CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY), 0, -1, List.of(), false, 52);
            assertThrows(IllegalArgumentException.class, () -> CpuPartitionLowering.requirePermittedLowPrecisionUse(count));
            assertThrows(IllegalArgumentException.class, () -> CpuPartitionLowering.requirePermittedLowPrecisionUse(spec, envelope));
            assertThrows(IllegalArgumentException.class, () -> new CpuClassFileKernelGenerator().generateClassBytes(spec, envelope));
            assertThrows(IllegalArgumentException.class, () -> new CpuGeneratedKernelArtifactStore().loadOrGenerate(spec, envelope));
        }
    }


    private static CpuKernelSpecialization specialization(DataType type, CpuKernelIr ir) {
        return new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR, Collections.nCopies(3, type),
                Collections.nCopies(3, CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY),
                0, -1, List.of(), false, type == DataType.BFLOAT16 ? 59 : 68);
    }

    private static byte[] payload(DataType type, String extension) throws Exception {
        try (var stream = CpuLowArithmeticLockTest.class.getResourceAsStream("low-lock-before-" + type + "." + extension + ".gz.b64")) {
            assertNotNull(stream);
            try (var gzip = new GZIPInputStream(new ByteArrayInputStream(Base64.getMimeDecoder().decode(stream.readAllBytes())))) {
                return gzip.readAllBytes();
            }
        }
    }
}

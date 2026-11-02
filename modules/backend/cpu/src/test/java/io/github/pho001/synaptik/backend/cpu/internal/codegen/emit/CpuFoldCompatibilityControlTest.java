package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuFoldLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.nio.file.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Literal accepted-dirty-before controls; low formats are compatibility controls, not oracles. */
class CpuFoldCompatibilityControlTest {
    @Test void ordinaryFoldLiteralControls() throws Exception {
        String root = System.getenv("SYNAPTIK_CPU0050_CONTROLS");
        org.junit.jupiter.api.Assumptions.assumeTrue(root != null);
        boolean capture = "true".equals(System.getenv("SYNAPTIK_CPU0050_CAPTURE"));
        int retained = 0, changed = 0;
        for (var type : List.of(DataType.FLOAT32, DataType.FLOAT64,
                DataType.BFLOAT16, DataType.FLOAT16, DataType.INT32, DataType.INT64)) {
            for (boolean image : List.of(false, true)) {
                if (image && (type == DataType.INT32 || type == DataType.INT64)) continue;
                Shape output = image ? Shape.of(2, 3, 5, 7) : Shape.of(5);
                Shape input = image ? Shape.of(2, 18, 35) : Shape.of(3, 3);
                Operation op = image ? new Operation(WindowTransformKind.FOLD2D,
                        new Fold2dAttrs(output, new Window2dAttrs(2, 3, 1, 1, 1, 2, 2, 2, false)))
                        : new Operation(WindowTransformKind.FOLD_AXIS, new FoldAxisAttrs(0, 5, 1));
                var base = CpuFoldLoweringTest.context(op, type, input, output);
                for (int carriers = 0; carriers < 4; carriers++) {
                    CarrierAccess heap = type == DataType.FLOAT32 ? CarrierAccess.FLOAT_ARRAY
                            : type == DataType.FLOAT64 ? CarrierAccess.DOUBLE_ARRAY
                            : type == DataType.INT32 ? CarrierAccess.INT_ARRAY
                            : type == DataType.INT64 ? CarrierAccess.LONG_ARRAY : CarrierAccess.SHORT_ARRAY;
                    var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                            base.memoryRequirements(), base.constants(), new CpuPartitionAnalysisInputs(false,
                                    List.of((carriers & 1) == 0 ? heap : CarrierAccess.MEMORY_SEGMENT,
                                            (carriers & 2) == 0 ? heap : CarrierAccess.MEMORY_SEGMENT)));
                    if (io.github.pho001.synaptik.backend.cpu.CpuGeneratedCoverageUnsupportedBoundaryTest.assertLowUnsupported(context)) continue;
                    var plan = new CpuPartitionPreparer().analyze(context).plan();
                    var route = plan.units().getFirst().portablePlan();
                    var spec = route.specialization();
                    String name = type + "-" + (image ? "2d" : "axis") + "-" + carriers;
                    var payloads = Map.of("compatibility", spec.compatibilityBytes(),
                            "identity", spec.classIdentityBytes(),
                            "key", spec.structuralKey().getBytes(StandardCharsets.US_ASCII),
                            "class", new CpuClassFileKernelGenerator().generateClassBytes(spec, route.kernelIr()),
                            "pack", Arrays.toString(plan.foldGeometry().orElseThrow()
                                    .pack(new long[]{3, 5}, 1, 3)).getBytes(StandardCharsets.US_ASCII));
                    String selectedRoot = type == DataType.INT32 || type == DataType.INT64
                            ? System.getenv("SYNAPTIK_CPU0050A_CONTROLS") : root;
                    assertNotNull(selectedRoot, "missing pre-edit integral controls");
                    Files.createDirectories(Path.of(selectedRoot));
                    boolean low = type == DataType.BFLOAT16 || type == DataType.FLOAT16;
                    for (var entry : payloads.entrySet()) {
                        Path file = Path.of(selectedRoot, name + "." + entry.getKey());
                        if (capture) { assertFalse(Files.exists(file)); Files.write(file, entry.getValue()); }
                        else if (low && !entry.getKey().equals("pack")) {
                            assertFalse(Arrays.equals(Files.readAllBytes(file), entry.getValue()),
                                    "must invalidate authentic low " + file);
                        }
                        else assertArrayEquals(Files.readAllBytes(file), entry.getValue(), file.toString());
                    }
                    if (low) changed++; else retained++;
                }
            }
        }
        assertEquals(16, changed, "both low families/all carriers");
        assertEquals(24, retained, "16 original floating and 8 integral forms remain literal");
    }

    @Test void compactFoldLiteralControls() throws Exception {
        String root = System.getenv("SYNAPTIK_CPU0050A_CONTROLS");
        org.junit.jupiter.api.Assumptions.assumeTrue(root != null);
        boolean capture = "true".equals(System.getenv("SYNAPTIK_CPU0050_CAPTURE"));
        for (var type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            var base = CpuBroadcastFoldGeneratedKernelTest.context(type, Shape.of(2, 3, 5, 7),
                    new Window2dAttrs(2, 3, 1, 1, 0, 1, 1, 1, false), 0, false);
            for (int carriers = 0; carriers < 4; carriers++) {
                CarrierAccess heap = type == DataType.FLOAT32 ? CarrierAccess.FLOAT_ARRAY : CarrierAccess.DOUBLE_ARRAY;
                var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                        base.memoryRequirements(), base.constants(), new CpuPartitionAnalysisInputs(false,
                                List.of((carriers & 1) == 0 ? heap : CarrierAccess.MEMORY_SEGMENT,
                                        (carriers & 2) == 0 ? heap : CarrierAccess.MEMORY_SEGMENT)));
                var plan = new CpuPartitionPreparer().analyze(context).plan();
                var route = plan.units().getFirst().portablePlan();
                var spec = route.specialization();
                String name = type + "-compact-" + carriers;
                var payloads = Map.of("compatibility", spec.compatibilityBytes(),
                        "identity", spec.classIdentityBytes(),
                        "key", spec.structuralKey().getBytes(StandardCharsets.US_ASCII),
                        "class", new CpuClassFileKernelGenerator().generateClassBytes(spec, route.kernelIr()),
                        "pack", Arrays.toString(plan.foldGeometry().orElseThrow()
                                .pack(new long[]{3, 5}, 1, 3)).getBytes(StandardCharsets.US_ASCII));
                Files.createDirectories(Path.of(root));
                for (var entry : payloads.entrySet()) {
                    Path file = Path.of(root, name + "." + entry.getKey());
                    if (capture) { assertFalse(Files.exists(file)); Files.write(file, entry.getValue()); }
                    else assertArrayEquals(Files.readAllBytes(file), entry.getValue(), file.toString());
                }
            }
        }
    }
}

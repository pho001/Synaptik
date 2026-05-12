package onnx;

import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.nio.file.Files;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class OnnxExportImportTest {

    @Test
    void exportSimpleAddAsGraphInputs() {
        Tensor a = new Tensor(new float[]{1f, 2f}, new int[]{2}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{3f, 4f}, new int[]{2}, null, "b", DataType.FLOAT32);
        Tensor out = a.add(b);
        out.setLabel("out");

        OnnxModel model = Onnx.exportModel(out, OnnxExportOptions.defaults().withLeafTensorPolicy(OnnxLeafTensorPolicy.INPUTS));

        OnnxProto.GraphProto graph = model.proto().getGraph();
        assertEquals(2, graph.getInputCount());
        assertEquals(0, graph.getInitializerCount());
        assertEquals(1, graph.getNodeCount());
        assertEquals("Add", graph.getNode(0).getOpType());
        assertEquals("out", graph.getOutput(0).getName());
    }

    @Test
    void defaultExportTreatsLeavesAsInputs() {
        Tensor a = new Tensor(new float[]{1f, 2f}, new int[]{2}, null, "defaultA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{3f, 4f}, new int[]{2}, null, "defaultB", DataType.FLOAT32);
        Tensor out = a.add(b);
        out.setLabel("defaultOut");

        OnnxModel model = Onnx.exportModel(out);

        assertEquals(2, model.proto().getGraph().getInputCount());
        assertEquals(0, model.proto().getGraph().getInitializerCount());
    }

    @Test
    void exportedInputModelImportsAndExecutes() {
        Tensor a = new Tensor(new float[]{0f, 0f}, new int[]{2}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{0f, 0f}, new int[]{2}, null, "b", DataType.FLOAT32);
        Tensor out = a.add(b).mul(a);
        out.setLabel("out");

        OnnxModel exported = Onnx.exportModel(out, OnnxExportOptions.defaults().withLeafTensorPolicy(OnnxLeafTensorPolicy.INPUTS));
        ImportedOnnxModel imported = Onnx.importModel(exported.proto());
        imported.input("a").setData(new float[]{2f, 3f});
        imported.input("b").setData(new float[]{5f, 7f});

        imported.compile(CompileConfig.inference()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{14.0, 30.0}, imported.output("out").toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void matmulWithTransposeRoundTripsThroughOnnx() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{7f, 8f, 9f, 10f, 11f, 12f}, new int[]{2, 3}, null, "b", DataType.FLOAT32);
        Tensor out = a.matmul(b.transpose());
        out.setLabel("scores");

        OnnxModel exported = Onnx.exportModel(out, OnnxExportOptions.defaults().withLeafTensorPolicy(OnnxLeafTensorPolicy.INPUTS));
        ImportedOnnxModel imported = Onnx.importModel(exported.proto());
        imported.input("a").setData(new float[]{1f, 2f, 3f, 4f, 5f, 6f});
        imported.input("b").setData(new float[]{7f, 8f, 9f, 10f, 11f, 12f});

        imported.compile("scores", CompileConfig.inference()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{50.0, 68.0, 122.0, 167.0}, imported.output("scores").toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void exportInitializersPreservesSupportedDtypes() {
        Tensor f64 = new Tensor(new double[]{1.25, 2.5}, new int[]{2}, null, "f64", DataType.FLOAT64);
        Tensor f64b = new Tensor(new double[]{3.0, 4.0}, new int[]{2}, null, "f64b", DataType.FLOAT64);
        Tensor f64Out = f64.add(f64b);
        f64Out.setLabel("f64_out");

        OnnxModel f64Model = Onnx.exportModel(f64Out, OnnxExportOptions.defaults().withLeafTensorPolicy(OnnxLeafTensorPolicy.INITIALIZERS));

        assertEquals(2, f64Model.proto().getGraph().getInitializerCount());
        assertTrue(f64Model.proto().getGraph().getInitializerList().stream()
                .anyMatch(t -> t.getName().equals("f64")
                        && t.getDataType() == OnnxProto.TensorProto.DataType.DOUBLE.getNumber()));

        Tensor bf16 = new Tensor(new short[]{(short) 0x3f80, (short) 0x4000}, new int[]{2}, null, "bf16", DataType.BFLOAT16);
        Tensor bf16b = new Tensor(new short[]{(short) 0x4040, (short) 0x4080}, new int[]{2}, null, "bf16b", DataType.BFLOAT16);
        Tensor bf16Out = bf16.add(bf16b);
        bf16Out.setLabel("bf16_out");
        OnnxModel bf16Model = Onnx.exportModel(bf16Out, OnnxExportOptions.defaults().withLeafTensorPolicy(OnnxLeafTensorPolicy.INITIALIZERS));

        assertTrue(bf16Model.proto().getGraph().getInitializerList().stream()
                .anyMatch(t -> t.getName().equals("bf16")
                        && t.getDataType() == OnnxProto.TensorProto.DataType.BFLOAT16.getNumber()
                        && t.getInt32Data(0) == 0x3f80));
    }

    @Test
    void importRejectsExternalDataByDefault() {
        OnnxProto.TensorProto external = OnnxProto.TensorProto.newBuilder()
                .setName("weight")
                .setDataType(OnnxProto.TensorProto.DataType.FLOAT.getNumber())
                .addDims(1)
                .setDataLocation(OnnxProto.TensorProto.DataLocation.EXTERNAL)
                .addExternalData(OnnxProto.StringStringEntryProto.newBuilder()
                        .setKey("location")
                        .setValue("../weight.bin"))
                .build();
        OnnxProto.ModelProto model = OnnxProto.ModelProto.newBuilder()
                .setIrVersion(OnnxProto.Version.IR_VERSION_2023_5_5.getNumber())
                .addOpsetImport(OnnxProto.OperatorSetIdProto.newBuilder().setDomain("").setVersion(OnnxExportOptions.DEFAULT_OPSET))
                .setGraph(OnnxProto.GraphProto.newBuilder()
                        .setName("external")
                        .addInitializer(external))
                .build();

        OnnxUnsupportedException ex = assertThrows(OnnxUnsupportedException.class, () -> Onnx.importModel(model));
        assertTrue(ex.getMessage().contains("external tensor data"));
    }

    @Test
    void importNormalizesNegativeReductionAxesBeforeRankChanges() {
        OnnxProto.ModelProto model = OnnxProto.ModelProto.newBuilder()
                .setIrVersion(OnnxProto.Version.IR_VERSION_2023_5_5.getNumber())
                .addOpsetImport(OnnxProto.OperatorSetIdProto.newBuilder().setDomain("").setVersion(OnnxExportOptions.DEFAULT_OPSET))
                .setGraph(OnnxProto.GraphProto.newBuilder()
                        .setName("negative_reduce_axes")
                        .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{2, 3, 4}))
                        .addInitializer(OnnxTensorProtoUtil.int64Initializer("axes", new long[]{-1, -2}))
                        .addNode(OnnxProto.NodeProto.newBuilder()
                                .setName("reduce")
                                .setOpType("ReduceSum")
                                .addInput("x")
                                .addInput("axes")
                                .addOutput("y")
                                .addAttribute(OnnxProto.AttributeProto.newBuilder().setName("keepdims").setI(0)))
                        .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2})))
                .build();

        ImportedOnnxModel imported = Onnx.importModel(model);
        imported.input("x").setData(new float[]{
                1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f,
                2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f
        });

        imported.compile(CompileConfig.inference()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{12.0, 24.0}, imported.output("y").toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void importNormalizesNegativeSqueezeAndUnsqueezeAxes() {
        OnnxProto.ModelProto model = OnnxProto.ModelProto.newBuilder()
                .setIrVersion(OnnxProto.Version.IR_VERSION_2023_5_5.getNumber())
                .addOpsetImport(OnnxProto.OperatorSetIdProto.newBuilder().setDomain("").setVersion(OnnxExportOptions.DEFAULT_OPSET))
                .setGraph(OnnxProto.GraphProto.newBuilder()
                        .setName("negative_layout_axes")
                        .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{2, 1, 3}))
                        .addInitializer(OnnxTensorProtoUtil.int64Initializer("squeeze_axes", new long[]{-2}))
                        .addInitializer(OnnxTensorProtoUtil.int64Initializer("unsqueeze_axes", new long[]{-1}))
                        .addNode(OnnxProto.NodeProto.newBuilder()
                                .setName("squeeze")
                                .setOpType("Squeeze")
                                .addInput("x")
                                .addInput("squeeze_axes")
                                .addOutput("squeezed"))
                        .addNode(OnnxProto.NodeProto.newBuilder()
                                .setName("unsqueeze")
                                .setOpType("Unsqueeze")
                                .addInput("squeezed")
                                .addInput("unsqueeze_axes")
                                .addOutput("y"))
                        .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2, 3, 1})))
                .build();

        ImportedOnnxModel imported = Onnx.importModel(model);

        assertArrayEquals(new int[]{2, 3, 1}, imported.output("y").getShape());
    }

    @Test
    void exportTensorAlgebraSubsetUsesOnnxOperatorNames() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f}, new int[]{3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{3f, 2f, 1f}, new int[]{3}, null, "b", DataType.FLOAT32);
        Tensor p = new Tensor(new byte[]{1, 0, 1}, new int[]{3}, null, "p", DataType.BOOL);
        Tensor q = new Tensor(new byte[]{1, 1, 0}, new int[]{3}, null, "q", DataType.BOOL);

        assertEquals("Min", singleNode(a.min(b)).getOpType());
        assertEquals("Max", singleNode(a.max(b)).getOpType());
        assertEquals("Equal", singleNode(a.equalTo(b)).getOpType());
        assertEquals("Greater", singleNode(a.greaterThan(b)).getOpType());
        assertEquals("GreaterOrEqual", singleNode(a.greaterOrEqual(b)).getOpType());
        assertEquals("Less", singleNode(a.lessThan(b)).getOpType());
        assertEquals("LessOrEqual", singleNode(a.lessOrEqual(b)).getOpType());
        assertEquals("And", singleNode(p.logicalAnd(q)).getOpType());
        assertEquals("Or", singleNode(p.logicalOr(q)).getOpType());
        assertEquals("Not", singleNode(p.logicalNot()).getOpType());
        assertEquals("Where", singleNode(Tensor.where(p, a, b)).getOpType());

        OnnxModel pow = exportInputs(a.pow(3.0));
        assertEquals("Pow", pow.proto().getGraph().getNode(0).getOpType());
        assertEquals(2, pow.proto().getGraph().getNode(0).getInputCount());
        assertEquals(1, pow.proto().getGraph().getInitializerCount());

        OnnxModel clampMin = exportInputs(a.clampMin(0.25));
        assertEquals("Clip", clampMin.proto().getGraph().getNode(0).getOpType());
        assertEquals(2, clampMin.proto().getGraph().getNode(0).getInputCount());

        OnnxModel clampMax = exportInputs(a.clampMax(2.5));
        OnnxProto.NodeProto clampMaxNode = clampMax.proto().getGraph().getNode(0);
        assertEquals("Clip", clampMaxNode.getOpType());
        assertEquals(3, clampMaxNode.getInputCount());
        assertEquals("", clampMaxNode.getInput(1));
    }

    @Test
    void importTensorAlgebraSubsetExecutes() {
        OnnxProto.ModelProto model = model("tensor_algebra", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{4}))
                .addInput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{4}))
                .addInput(OnnxTensorProtoUtil.valueInfo("cond", DataType.BOOL, new int[]{4}))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("exponent", Tensor.scalar(3.0, DataType.FLOAT32)))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("min_bound", Tensor.scalar(0.0, DataType.FLOAT32)))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("max_bound", Tensor.scalar(10.0, DataType.FLOAT32)))
                .addNode(node("min", "Min", "min", "x", "y"))
                .addNode(node("max", "Max", "max", "x", "y"))
                .addNode(node("pow", "Pow", "pow", "min", "exponent"))
                .addNode(node("clip", "Clip", "clipped", "pow", "min_bound", "max_bound"))
                .addNode(node("where", "Where", "out", "cond", "clipped", "max"))
                .addOutput(OnnxTensorProtoUtil.valueInfo("max", DataType.FLOAT32, new int[]{4}))
                .addOutput(OnnxTensorProtoUtil.valueInfo("out", DataType.FLOAT32, new int[]{4})));

        ImportedOnnxModel imported = Onnx.importModel(model);
        imported.input("x").setData(new float[]{-2f, 2f, 4f, 6f});
        imported.input("y").setData(new float[]{1f, 3f, 2f, 8f});
        imported.input("cond").setData(new byte[]{1, 0, 1, 0});

        execute(imported, "max");
        assertArrayEquals(new double[]{1.0, 3.0, 4.0, 8.0}, imported.output("max").toDoubleArrayCopy(), 1e-6);

        execute(imported, "out");
        assertArrayEquals(new double[]{0.0, 3.0, 8.0, 8.0}, imported.output("out").toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void importCompareAndLogicalSubsetProducesBoolOutputs() {
        OnnxProto.ModelProto model = model("compare_logical", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{3}))
                .addInput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{3}))
                .addInput(OnnxTensorProtoUtil.valueInfo("p", DataType.BOOL, new int[]{3}))
                .addInput(OnnxTensorProtoUtil.valueInfo("q", DataType.BOOL, new int[]{3}))
                .addNode(node("eq", "Equal", "eq", "x", "y"))
                .addNode(node("gt", "Greater", "gt", "x", "y"))
                .addNode(node("ge", "GreaterOrEqual", "ge", "x", "y"))
                .addNode(node("lt", "Less", "lt", "x", "y"))
                .addNode(node("le", "LessOrEqual", "le", "x", "y"))
                .addNode(node("and", "And", "and", "p", "q"))
                .addNode(node("or", "Or", "or", "p", "q"))
                .addNode(node("not", "Not", "not", "p"))
                .addOutput(OnnxTensorProtoUtil.valueInfo("eq", DataType.BOOL, new int[]{3}))
                .addOutput(OnnxTensorProtoUtil.valueInfo("gt", DataType.BOOL, new int[]{3}))
                .addOutput(OnnxTensorProtoUtil.valueInfo("ge", DataType.BOOL, new int[]{3}))
                .addOutput(OnnxTensorProtoUtil.valueInfo("lt", DataType.BOOL, new int[]{3}))
                .addOutput(OnnxTensorProtoUtil.valueInfo("le", DataType.BOOL, new int[]{3}))
                .addOutput(OnnxTensorProtoUtil.valueInfo("and", DataType.BOOL, new int[]{3}))
                .addOutput(OnnxTensorProtoUtil.valueInfo("or", DataType.BOOL, new int[]{3}))
                .addOutput(OnnxTensorProtoUtil.valueInfo("not", DataType.BOOL, new int[]{3})));

        ImportedOnnxModel imported = Onnx.importModel(model);
        imported.input("x").setData(new float[]{1f, 2f, 3f});
        imported.input("y").setData(new float[]{1f, 1f, 4f});
        imported.input("p").setData(new byte[]{1, 0, 1});
        imported.input("q").setData(new byte[]{1, 1, 0});

        assertBoolOutput(imported, "eq", true, false, false);
        assertBoolOutput(imported, "gt", false, true, false);
        assertBoolOutput(imported, "ge", true, true, false);
        assertBoolOutput(imported, "lt", false, false, true);
        assertBoolOutput(imported, "le", true, false, true);
        assertBoolOutput(imported, "and", true, false, false);
        assertBoolOutput(imported, "or", true, true, true);
        assertBoolOutput(imported, "not", false, true, false);
    }

    @Test
    void importIdentityAndMaxOnlyClip() {
        OnnxProto.ModelProto model = model("identity_clip", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{4}))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("max_bound", Tensor.scalar(2.5, DataType.FLOAT32)))
                .addNode(node("identity", "Identity", "id", "x"))
                .addNode(node("clip", "Clip", "y", "id", "", "max_bound"))
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{4})));

        ImportedOnnxModel imported = Onnx.importModel(model);
        imported.input("x").setData(new float[]{-1f, 2f, 3f, 4f});

        execute(imported, "y");

        assertArrayEquals(new double[]{-1.0, 2.0, 2.5, 2.5}, imported.output("y").toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void importRejectsNonConstantPowExponent() {
        OnnxProto.ModelProto model = model("dynamic_pow", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{2}))
                .addInput(OnnxTensorProtoUtil.valueInfo("exponent", DataType.FLOAT32, new int[]{2}))
                .addNode(node("pow", "Pow", "y", "x", "exponent"))
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2})));

        OnnxUnsupportedException ex = assertThrows(OnnxUnsupportedException.class, () -> Onnx.importModel(model));
        assertTrue(ex.getMessage().contains("scalar initializer or Constant node"));
    }

    @Test
    void importRejectsCastUntilGraphCastExists() {
        OnnxProto.ModelProto model = model("cast", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{2}))
                .addNode(OnnxProto.NodeProto.newBuilder()
                        .setName("cast")
                        .setOpType("Cast")
                        .addInput("x")
                        .addOutput("y")
                        .addAttribute(OnnxProto.AttributeProto.newBuilder()
                                .setName("to")
                                .setI(OnnxProto.TensorProto.DataType.DOUBLE.getNumber())))
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT64, new int[]{2})));

        OnnxUnsupportedException ex = assertThrows(OnnxUnsupportedException.class, () -> Onnx.importModel(model));
        assertTrue(ex.getMessage().contains("unsupported op type"));
    }

    @Test
    void modelCanBeWrittenAndReadBack() throws Exception {
        Tensor a = new Tensor(new float[]{1f}, new int[]{1}, null, "a", DataType.FLOAT32);
        Tensor out = a.exp();
        out.setLabel("out");
        OnnxModel model = Onnx.exportModel(out, OnnxExportOptions.defaults().withLeafTensorPolicy(OnnxLeafTensorPolicy.INPUTS));

        java.nio.file.Path path = Files.createTempFile("synaptik-onnx-", ".onnx");
        model.write(path);

        ImportedOnnxModel imported = Onnx.read(path);
        assertEquals("a", imported.inputs().keySet().iterator().next());
        assertEquals("out", imported.outputs().keySet().iterator().next());
    }

    private static OnnxModel exportInputs(Tensor output) {
        return Onnx.exportModel(output, OnnxExportOptions.defaults().withLeafTensorPolicy(OnnxLeafTensorPolicy.INPUTS));
    }

    private static OnnxProto.NodeProto singleNode(Tensor output) {
        OnnxProto.GraphProto graph = exportInputs(output).proto().getGraph();
        assertEquals(1, graph.getNodeCount());
        return graph.getNode(0);
    }

    private static OnnxProto.ModelProto model(String graphName, Consumer<OnnxProto.GraphProto.Builder> configure) {
        OnnxProto.GraphProto.Builder graph = OnnxProto.GraphProto.newBuilder().setName(graphName);
        configure.accept(graph);
        return OnnxProto.ModelProto.newBuilder()
                .setIrVersion(OnnxProto.Version.IR_VERSION_2023_5_5.getNumber())
                .addOpsetImport(OnnxProto.OperatorSetIdProto.newBuilder()
                        .setDomain("")
                        .setVersion(OnnxExportOptions.DEFAULT_OPSET))
                .setGraph(graph)
                .build();
    }

    private static OnnxProto.NodeProto node(String name, String opType, String output, String... inputs) {
        OnnxProto.NodeProto.Builder node = OnnxProto.NodeProto.newBuilder()
                .setName(name)
                .setOpType(opType);
        for (String input : inputs) {
            node.addInput(input);
        }
        return node.addOutput(output).build();
    }

    private static void execute(ImportedOnnxModel imported, String outputName) {
        imported.compile(outputName, CompileConfig.inference()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
    }

    private static void assertBoolOutput(ImportedOnnxModel imported, String outputName, boolean... expected) {
        execute(imported, outputName);
        assertEquals(DataType.BOOL, imported.output(outputName).getDataType());
        assertArrayEquals(expected, imported.output(outputName).toBooleanArrayCopy());
    }
}

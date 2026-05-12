package onnx;

import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import operations.index.ScatterReduction;
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
    void importCastSliceConcatExpandAndFlattenExecute() {
        OnnxProto.ModelProto model = model("shape_index_values", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{2, 3}))
                .addInput(OnnxTensorProtoUtil.valueInfo("row", DataType.FLOAT32, new int[]{1, 2}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("starts", new long[]{0, 1}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("ends", new long[]{2, 3}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("axes", new long[]{0, 1}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("steps", new long[]{1, 1}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("expand_shape", new long[]{3, 2}))
                .addNode(node("slice", "Slice", "sliced", "x", "starts", "ends", "axes", "steps"))
                .addNode(axisNode("concat", "Concat", "joined", 0, "sliced", "sliced"))
                .addNode(castNode("cast", "joined", "casted", OnnxProto.TensorProto.DataType.DOUBLE))
                .addNode(node("expand", "Expand", "expanded", "row", "expand_shape"))
                .addNode(axisNode("flatten", "Flatten", "flat", 1, "expanded"))
                .addOutput(OnnxTensorProtoUtil.valueInfo("casted", DataType.FLOAT64, new int[]{4, 2}))
                .addOutput(OnnxTensorProtoUtil.valueInfo("flat", DataType.FLOAT32, new int[]{3, 2})));

        ImportedOnnxModel imported = Onnx.importModel(model);
        imported.input("x").setData(new float[]{1f, 2f, 3f, 4f, 5f, 6f});
        imported.input("row").setData(new float[]{7f, 8f});

        execute(imported, "casted");
        assertEquals(DataType.FLOAT64, imported.output("casted").getDataType());
        assertArrayEquals(new double[]{2.0, 3.0, 5.0, 6.0, 2.0, 3.0, 5.0, 6.0},
                imported.output("casted").toDoubleArrayCopy(), 1e-9);

        execute(imported, "flat");
        assertArrayEquals(new double[]{7.0, 8.0, 7.0, 8.0, 7.0, 8.0},
                imported.output("flat").toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void importPyTorchStyleShapeSubgraphFeedsReshape() {
        OnnxProto.ModelProto model = model("shape_subgraph", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{2, 3, 4}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("idx0", new long[]{0}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("axes0", new long[]{0}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("tail", new long[]{12}))
                .addNode(node("shape", "Shape", "shape", "x"))
                .addNode(node("gather", "Gather", "dim0_scalar", "shape", "idx0"))
                .addNode(node("unsqueeze", "Unsqueeze", "dim0", "dim0_scalar", "axes0"))
                .addNode(axisNode("concat", "Concat", "target_shape", 0, "dim0", "tail"))
                .addNode(node("reshape", "Reshape", "y", "x", "target_shape"))
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2, 12})));

        ImportedOnnxModel imported = Onnx.importModel(model);
        imported.input("x").setData(new float[]{
                1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f,
                13f, 14f, 15f, 16f, 17f, 18f, 19f, 20f, 21f, 22f, 23f, 24f
        });

        execute(imported, "y");

        assertArrayEquals(new int[]{2, 12}, imported.output("y").getShape());
        assertArrayEquals(new double[]{
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
        }, imported.output("y").toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void importShapeSliceAndShapeStartEndFeedReshape() {
        OnnxProto.ModelProto model = model("shape_slice_subgraph", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{2, 3, 4}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("starts", new long[]{1}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("ends", new long[]{Long.MAX_VALUE}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("axes", new long[]{0}))
                .addNode(node("shape", "Shape", "shape", "x"))
                .addNode(nodeBuilder("batch_shape", "Shape", "batch_shape", "x")
                        .addAttribute(OnnxProto.AttributeProto.newBuilder().setName("start").setI(0))
                        .addAttribute(OnnxProto.AttributeProto.newBuilder().setName("end").setI(1))
                        .build())
                .addNode(node("tail_slice", "Slice", "tail_shape", "shape", "starts", "ends", "axes"))
                .addNode(axisNode("concat", "Concat", "target_shape", 0, "batch_shape", "tail_shape"))
                .addNode(node("reshape", "Reshape", "y", "x", "target_shape"))
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2, 3, 4})));

        ImportedOnnxModel imported = Onnx.importModel(model);
        imported.input("x").setData(new float[]{
                1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f,
                13f, 14f, 15f, 16f, 17f, 18f, 19f, 20f, 21f, 22f, 23f, 24f
        });

        execute(imported, "y");

        assertArrayEquals(new int[]{2, 3, 4}, imported.output("y").getShape());
        assertArrayEquals(new double[]{
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
        }, imported.output("y").toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void exportShapeIndexSubsetUsesOnnxOperatorNames() {
        Tensor x = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "x", DataType.FLOAT32);
        Tensor y = new Tensor(new float[]{10f, 20f, 30f, 40f, 50f, 60f}, new int[]{2, 3}, null, "y", DataType.FLOAT32);

        assertEquals("Cast", singleNode(x.cast(DataType.FLOAT64)).getOpType());
        assertEquals("Slice", singleNode(x.slice(new int[]{0, 1}, new int[]{2, 3}, new int[]{0, 1}, new int[]{1, 1})).getOpType());
        assertEquals("Concat", singleNode(Tensor.concat(0, x, y)).getOpType());
        assertEquals("Expand", singleNode(new Tensor(new float[]{1f, 2f, 3f}, new int[]{1, 3}, null, "row", DataType.FLOAT32)
                .expand(2, 3)).getOpType());
        assertEquals("Gather", singleNode(x.gatherAxis(new Tensor(new int[]{2, 0}, new int[]{2}, null, "idx", DataType.INT32), 1)).getOpType());
        OnnxProto.NodeProto gatherElements = singleNode(x.takeAlongAxis(new Tensor(new int[]{2, 1, 0, 0}, new int[]{2, 2}, null, "take_idx", DataType.INT32), 1));
        assertEquals("GatherElements", gatherElements.getOpType());
        assertEquals(1, gatherElements.getAttribute(0).getI());
        OnnxProto.NodeProto scatterElements = singleNode(x.scatterElements(
                new Tensor(new int[]{2, 0, 0, 2}, new int[]{2, 2}, null, "scatter_idx", DataType.INT32),
                new Tensor(new float[]{1f, 5f, 7f, 9f}, new int[]{2, 2}, null, "updates", DataType.FLOAT32),
                1,
                ScatterReduction.ADD));
        assertEquals("ScatterElements", scatterElements.getOpType());
        assertEquals(1, scatterElements.getAttribute(0).getI());
        assertEquals("add", scatterElements.getAttribute(1).getS().toStringUtf8());
        OnnxProto.NodeProto scatterNd = singleNode(x.scatterNd(
                new Tensor(new int[]{0, 2, 1, 0}, new int[]{2, 2}, null, "scatter_nd_idx", DataType.INT32),
                new Tensor(new float[]{1f, 7f}, new int[]{2}, null, "scatter_nd_updates", DataType.FLOAT32),
                ScatterReduction.ADD));
        assertEquals("ScatterND", scatterNd.getOpType());
        assertEquals("add", scatterNd.getAttribute(0).getS().toStringUtf8());
        assertThrows(OnnxUnsupportedException.class,
                () -> exportInputs(x.gather(new Tensor(new int[]{2, 0}, new int[]{2}, null, "old_idx", DataType.INT32), 1)));
    }

    @Test
    void importRuntimeGatherExecutesOnnxShapeSemantics() {
        OnnxProto.ModelProto model = model("runtime_gather", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{2, 3}))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("idx",
                        new Tensor(new int[]{2, 0, -1, 1}, new int[]{2, 2}, null, "idx", DataType.INT32)))
                .addNode(axisNode("gather", "Gather", "y", 1, "x", "idx"))
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2, 2, 2})));

        ImportedOnnxModel imported = Onnx.importModel(model);
        imported.input("x").setData(new float[]{1f, 2f, 3f, 4f, 5f, 6f});

        execute(imported, "y");

        assertArrayEquals(new int[]{2, 2, 2}, imported.output("y").getShape());
        assertArrayEquals(new double[]{
                3.0, 1.0, 3.0, 2.0,
                6.0, 4.0, 6.0, 5.0
        }, imported.output("y").toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void importGatherElementsMapsToTakeAlongAxis() {
        OnnxProto.ModelProto model = model("runtime_gather_elements", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{2, 3}))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("idx",
                        new Tensor(new int[]{2, 1, 0, 0}, new int[]{2, 2}, null, "idx", DataType.INT32)))
                .addNode(axisNode("gather_elements", "GatherElements", "y", 1, "x", "idx"))
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2, 2})));

        ImportedOnnxModel imported = Onnx.importModel(model);
        imported.input("x").setData(new float[]{1f, 2f, 3f, 4f, 5f, 6f});

        execute(imported, "y");

        assertArrayEquals(new int[]{2, 2}, imported.output("y").getShape());
        assertArrayEquals(new double[]{3.0, 2.0, 4.0, 4.0}, imported.output("y").toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void importGatherElementsNormalizesNegativeAxisAndIndices() {
        OnnxProto.ModelProto model = model("runtime_gather_elements_negative_axis", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{2, 3}))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("idx",
                        new Tensor(new int[]{2, -1, 0, 1}, new int[]{2, 2}, null, "idx", DataType.INT32)))
                .addNode(axisNode("gather_elements", "GatherElements", "y", -1, "x", "idx"))
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2, 2})));

        ImportedOnnxModel imported = Onnx.importModel(model);
        imported.input("x").setData(new float[]{1f, 2f, 3f, 4f, 5f, 6f});

        execute(imported, "y");

        assertArrayEquals(new double[]{3.0, 3.0, 4.0, 5.0}, imported.output("y").toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void takeAlongAxisRoundTripsThroughGatherElements() {
        Tensor x = new Tensor(new float[6], new int[]{2, 3}, null, "x", DataType.FLOAT32);
        Tensor idx = new Tensor(new int[4], new int[]{2, 2}, null, "idx", DataType.INT32);
        Tensor out = x.takeAlongAxis(idx, 1);
        out.setLabel("taken");

        OnnxModel exported = Onnx.exportModel(out, OnnxExportOptions.defaults().withLeafTensorPolicy(OnnxLeafTensorPolicy.INPUTS));
        ImportedOnnxModel imported = Onnx.importModel(exported.proto());
        imported.input("x").setData(new float[]{1f, 2f, 3f, 4f, 5f, 6f});
        imported.input("idx").setData(new int[]{2, 1, 0, 0});

        execute(imported, "taken");

        assertArrayEquals(new int[]{2, 2}, imported.output("taken").getShape());
        assertArrayEquals(new double[]{3.0, 2.0, 4.0, 4.0}, imported.output("taken").toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void importGatherElementsRejectsInt64ShapeConstantIndices() {
        OnnxProto.ModelProto model = model("gather_elements_int64_indices", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{2, 3}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("idx", new long[]{2, 1, 0, 0}))
                .addNode(axisNode("gather_elements", "GatherElements", "y", 1, "x", "idx"))
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2, 2})));

        OnnxUnsupportedException ex = assertThrows(OnnxUnsupportedException.class, () -> Onnx.importModel(model));
        assertTrue(ex.getMessage().contains("GatherElements requires runtime INT32 indices"));
    }

    @Test
    void importScatterElementsExecutesNoneReduction() {
        OnnxProto.ModelProto model = model("scatter_elements", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("data", DataType.FLOAT32, new int[]{2, 3}))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("indices",
                        new Tensor(new int[]{2, 0, 0, 2}, new int[]{2, 2}, null, "indices", DataType.INT32)))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("updates",
                        new Tensor(new float[]{1f, 5f, 7f, 9f}, new int[]{2, 2}, null, "updates", DataType.FLOAT32)))
                .addNode(axisNode("scatter_elements", "ScatterElements", "y", 1, "data", "indices", "updates"))
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2, 3})));

        ImportedOnnxModel imported = Onnx.importModel(model);
        imported.input("data").setData(new float[]{10f, 20f, 30f, 40f, 50f, 60f});

        execute(imported, "y");

        assertArrayEquals(new double[]{
                5.0, 20.0, 1.0,
                7.0, 50.0, 9.0
        }, imported.output("y").toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void importScatterElementsExecutesAddReduction() {
        OnnxProto.ModelProto model = model("scatter_elements_add", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("data", DataType.FLOAT32, new int[]{2, 3}))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("indices",
                        new Tensor(new int[]{1, 1, 0, 2}, new int[]{2, 2}, null, "indices", DataType.INT32)))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("updates",
                        new Tensor(new float[]{1f, 5f, 7f, 9f}, new int[]{2, 2}, null, "updates", DataType.FLOAT32)))
                .addNode(nodeBuilder("scatter_elements", "ScatterElements", "y", "data", "indices", "updates")
                        .addAttribute(OnnxProto.AttributeProto.newBuilder().setName("axis").setI(1))
                        .addAttribute(OnnxProto.AttributeProto.newBuilder()
                                .setName("reduction")
                                .setS(com.google.protobuf.ByteString.copyFromUtf8("add")))
                        .build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2, 3})));

        ImportedOnnxModel imported = Onnx.importModel(model);
        imported.input("data").setData(new float[]{10f, 20f, 30f, 40f, 50f, 60f});

        execute(imported, "y");

        assertArrayEquals(new double[]{
                10.0, 26.0, 30.0,
                47.0, 50.0, 69.0
        }, imported.output("y").toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void scatterElementsRoundTripsThroughOnnx() {
        Tensor data = new Tensor(new float[6], new int[]{2, 3}, null, "data", DataType.FLOAT32);
        Tensor indices = new Tensor(new int[4], new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor updates = new Tensor(new float[4], new int[]{2, 2}, null, "updates", DataType.FLOAT32);
        Tensor out = data.scatterElements(indices, updates, 1, ScatterReduction.ADD);
        out.setLabel("scattered");

        OnnxModel exported = Onnx.exportModel(out, OnnxExportOptions.defaults().withLeafTensorPolicy(OnnxLeafTensorPolicy.INPUTS));
        ImportedOnnxModel imported = Onnx.importModel(exported.proto());
        imported.input("data").setData(new float[]{10f, 20f, 30f, 40f, 50f, 60f});
        imported.input("indices").setData(new int[]{1, 1, 0, 2});
        imported.input("updates").setData(new float[]{1f, 5f, 7f, 9f});

        execute(imported, "scattered");

        assertArrayEquals(new double[]{
                10.0, 26.0, 30.0,
                47.0, 50.0, 69.0
        }, imported.output("scattered").toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void importScatterNdExecutesTupleIndexedElementsAndSlices() {
        OnnxProto.ModelProto elementModel = model("scatter_nd_elements", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("data", DataType.FLOAT32, new int[]{2, 3}))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("indices",
                        new Tensor(new int[]{0, 2, 1, 0}, new int[]{2, 2}, null, "indices", DataType.INT32)))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("updates",
                        new Tensor(new float[]{1f, 7f}, new int[]{2}, null, "updates", DataType.FLOAT32)))
                .addNode(node("scatter_nd", "ScatterND", "y", "data", "indices", "updates"))
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2, 3})));
        OnnxProto.ModelProto sliceModel = model("scatter_nd_slices", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("data", DataType.FLOAT32, new int[]{2, 3}))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("indices",
                        new Tensor(new int[]{1}, new int[]{1, 1}, null, "indices", DataType.INT32)))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("updates",
                        new Tensor(new float[]{40f, 50f, 60f}, new int[]{1, 3}, null, "updates", DataType.FLOAT32)))
                .addNode(node("scatter_nd", "ScatterND", "y", "data", "indices", "updates"))
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2, 3})));

        ImportedOnnxModel elements = Onnx.importModel(elementModel);
        elements.input("data").setData(new float[]{10f, 20f, 30f, 40f, 50f, 60f});
        execute(elements, "y");

        assertArrayEquals(new double[]{
                10.0, 20.0, 1.0,
                7.0, 50.0, 60.0
        }, elements.output("y").toDoubleArrayCopy(), 1e-6);

        ImportedOnnxModel slices = Onnx.importModel(sliceModel);
        slices.input("data").setData(new float[]{1f, 2f, 3f, 4f, 5f, 6f});
        execute(slices, "y");

        assertArrayEquals(new double[]{
                1.0, 2.0, 3.0,
                40.0, 50.0, 60.0
        }, slices.output("y").toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void scatterNdRoundTripsThroughOnnx() {
        Tensor data = new Tensor(new float[6], new int[]{2, 3}, null, "data", DataType.FLOAT32);
        Tensor indices = new Tensor(new int[4], new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor updates = new Tensor(new float[2], new int[]{2}, null, "updates", DataType.FLOAT32);
        Tensor out = data.scatterNd(indices, updates, ScatterReduction.ADD);
        out.setLabel("scatter_nd_out");

        OnnxModel exported = Onnx.exportModel(out, OnnxExportOptions.defaults().withLeafTensorPolicy(OnnxLeafTensorPolicy.INPUTS));
        ImportedOnnxModel imported = Onnx.importModel(exported.proto());
        imported.input("data").setData(new float[]{10f, 20f, 30f, 40f, 50f, 60f});
        imported.input("indices").setData(new int[]{0, 1, 0, 1});
        imported.input("updates").setData(new float[]{5f, 7f});

        execute(imported, "scatter_nd_out");

        assertArrayEquals(new double[]{
                10.0, 32.0, 30.0,
                40.0, 50.0, 60.0
        }, imported.output("scatter_nd_out").toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void importRejectsScatterElementsAndScatterNdInvalidCases() {
        OnnxProto.ModelProto int64ScatterElements = model("scatter_elements_int64_indices", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("data", DataType.FLOAT32, new int[]{2, 3}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("indices", new long[]{2, 0, 0, 2}))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("updates",
                        new Tensor(new float[]{1f, 5f, 7f, 9f}, new int[]{2, 2}, null, "updates", DataType.FLOAT32)))
                .addNode(axisNode("scatter_elements", "ScatterElements", "y", 1, "data", "indices", "updates"))
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2, 3})));
        OnnxProto.ModelProto duplicateNone = model("scatter_elements_duplicate_none", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("data", DataType.FLOAT32, new int[]{2, 3}))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("indices",
                        new Tensor(new int[]{1, 1, 0, 2}, new int[]{2, 2}, null, "indices", DataType.INT32)))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("updates",
                        new Tensor(new float[]{1f, 5f, 7f, 9f}, new int[]{2, 2}, null, "updates", DataType.FLOAT32)))
                .addNode(axisNode("scatter_elements", "ScatterElements", "y", 1, "data", "indices", "updates"))
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2, 3})));
        OnnxProto.ModelProto int64ScatterNd = model("scatter_nd_int64_indices", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("data", DataType.FLOAT32, new int[]{2, 3}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("indices", new long[]{0, 1}))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("updates",
                        new Tensor(new float[]{9f}, new int[]{1}, null, "updates", DataType.FLOAT32)))
                .addNode(node("scatter_nd", "ScatterND", "y", "data", "indices", "updates"))
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2, 3})));

        OnnxUnsupportedException elementsEx = assertThrows(OnnxUnsupportedException.class, () -> Onnx.importModel(int64ScatterElements));
        OnnxUnsupportedException ndEx = assertThrows(OnnxUnsupportedException.class, () -> Onnx.importModel(int64ScatterNd));
        ImportedOnnxModel duplicateImported = Onnx.importModel(duplicateNone);
        duplicateImported.input("data").setData(new float[]{10f, 20f, 30f, 40f, 50f, 60f});

        assertTrue(elementsEx.getMessage().contains("ScatterElements requires runtime INT32 indices"));
        assertTrue(ndEx.getMessage().contains("ScatterND requires runtime INT32 indices"));
        assertThrows(IllegalArgumentException.class, () -> execute(duplicateImported, "y"));
    }

    @Test
    void importCastToBoolProducesBoolTensor() {
        OnnxProto.ModelProto model = model("cast", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{2}))
                .addNode(OnnxProto.NodeProto.newBuilder()
                        .setName("cast")
                        .setOpType("Cast")
                        .addInput("x")
                        .addOutput("y")
                        .addAttribute(OnnxProto.AttributeProto.newBuilder()
                                .setName("to")
                                .setI(OnnxProto.TensorProto.DataType.BOOL.getNumber())))
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.BOOL, new int[]{2})));

        ImportedOnnxModel imported = Onnx.importModel(model);
        imported.input("x").setData(new float[]{0f, -2f});

        execute(imported, "y");

        assertArrayEquals(new boolean[]{false, true}, imported.output("y").toBooleanArrayCopy());
    }

    @Test
    void importRejectsNonConstantSliceParameters() {
        OnnxProto.ModelProto dynamicSlice = model("dynamic_slice", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{4}))
                .addInput(OnnxTensorProtoUtil.valueInfo("starts", DataType.INT32, new int[]{1}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("ends", new long[]{3}))
                .addNode(node("slice", "Slice", "y", "x", "starts", "ends"))
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{3})));

        OnnxUnsupportedException sliceEx = assertThrows(OnnxUnsupportedException.class, () -> Onnx.importModel(dynamicSlice));
        assertTrue(sliceEx.getMessage().contains("constant initializer or Constant node"));
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

    private static OnnxProto.NodeProto axisNode(String name, String opType, String output, int axis, String... inputs) {
        return nodeBuilder(name, opType, output, inputs)
                .addAttribute(OnnxProto.AttributeProto.newBuilder().setName("axis").setI(axis))
                .build();
    }

    private static OnnxProto.NodeProto castNode(String name, String input, String output, OnnxProto.TensorProto.DataType target) {
        return nodeBuilder(name, "Cast", output, input)
                .addAttribute(OnnxProto.AttributeProto.newBuilder().setName("to").setI(target.getNumber()))
                .build();
    }

    private static OnnxProto.NodeProto.Builder nodeBuilder(String name, String opType, String output, String... inputs) {
        OnnxProto.NodeProto.Builder node = OnnxProto.NodeProto.newBuilder()
                .setName(name)
                .setOpType(opType);
        for (String input : inputs) {
            node.addInput(input);
        }
        return node.addOutput(output);
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

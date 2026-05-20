package debug;

import backend.ComputeBackend;
import backend.metal.lowering.MetalPartitionPlan;
import backend.runtime.ExecutionMode;
import config.optimizer.CpuFusionConfig;
import config.optimizer.CpuRegionConfig;
import config.compile.CompileConfig;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import config.runtime.AcceleratorBufferConfig;
import config.runtime.AcceleratorBufferBindingMode;
import config.runtime.AcceleratorConfig;
import config.runtime.AcceleratorBackendConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.CompiledNode;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.CompileMode;
import tensor.Tensor;
import tensor.factory.TensorDataFactory;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;
import tensor.options.AttentionOptions;
import tuning.workload.StandardWorkloads;
import tuning.workload.WorkloadEnvironment;
import tuning.workload.WorkloadInstance;

import operations.linalg.scaledDotProductAttentionBackward;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

final class TransformerMetalPartitionAnalysisTest {
    @Test
    void printGreedyTransformerMetalPartitions() {
        var optimizer = CompileConfig.training()
                .withBackendPlanning(config.compile.BackendPlanningConfig.autoAccelerator().withCpuRegions(CpuRegionConfig.defaults()))
                .withRegionOptimization(CompileConfig.training().regionOptimization().withCpuFusion(CpuFusionConfig.defaults()));
        var profile = new ExecutionProfile(
                "debug-transformer-metal",
                "debug-transformer-metal",
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                optimizer,
                RuntimeConfig.trainingDefaults(),
                WorkloadProfile.transformerHotPathMedium()
        );
        var workload = StandardWorkloads.transformerBlockHotPath(
                "debug_transformer_block",
                WorkloadProfile.transformerHotPathMedium()
        ).instantiate(new WorkloadEnvironment(profile));
        CompiledGraph compiled = CompiledGraph.compile(workload.root(), optimizer, CompileMode.TRAINING);
        var prepared = compiled.prepare(profile.runtime());

        System.out.println("NODES");
        for (CompiledNode node : compiled.compileArtifacts().compiledNodes()) {
            System.out.println(node.id()
                    + " " + (node.backwardNode() ? "B" : "F")
                    + " " + (node.operation() == null ? "LEAF" : node.operation().opType())
                    + " " + node.label()
                    + " inputs=" + node.inputIds()
                    + " backend=" + node.backend());
        }

        System.out.println("SELECTED");
        prepared.prepareTrace().backendSelection().decisions().stream()
                .filter(decision -> decision.selected() && decision.selectedBackend() == ComputeBackend.GPU_METAL)
                .forEach(decision -> System.out.println("anchor=" + decision.anchorNodeId()
                        + " nodes=" + decision.nodeIds()
                        + " reason=" + decision.reason()
                        + " outputs=" + (decision.gpuLoweredRegionManifest() == null
                        ? "n/a"
                        : decision.gpuLoweredRegionManifest().outputNodeIds())
                        + " external=" + (decision.gpuLoweredRegionManifest() == null
                        ? "n/a"
                        : decision.gpuLoweredRegionManifest().externalInputNodeIds())
                        + " primitives=" + (decision.gpuLoweredRegionManifest() == null
                        ? "n/a"
                        : decision.gpuLoweredRegionManifest().loweredPrimitives().stream()
                                .map(primitive -> primitive.primitiveType() + primitive.sourceOriginalNodeIds())
                                .toList())));

        System.out.println("PLANS");
        compiled.compileArtifacts().plannedPartitions().stream()
                .filter(partition -> partition.plan() instanceof MetalPartitionPlan)
                .map(partition -> (MetalPartitionPlan) partition.plan())
                .forEach(plan -> System.out.println("anchor=" + plan.anchorNodeId()
                        + " nodes=" + plan.nodeIds()
                        + " subgraphExternal=" + plan.externalInputNodeIds()
                        + " subgraphOutputs=" + plan.producedOutputNodeIds()
                        + " dagExternal=" + plan.lowering().dagSpec().externalInputs().stream()
                                .map(input -> input.nodeId())
                                .toList()
                        + " dagOutputs=" + plan.lowering().dagSpec().outputNodeIds()));
    }

    @Test
    void printGreedyTransformerMetalRuntimeTrace() {
        var optimizer = CompileConfig.training()
                .withBackendPlanning(config.compile.BackendPlanningConfig.autoAccelerator().withCpuRegions(CpuRegionConfig.defaults()))
                .withRegionOptimization(CompileConfig.training().regionOptimization().withCpuFusion(CpuFusionConfig.defaults()));
        var profile = new ExecutionProfile(
                "debug-transformer-metal-runtime",
                "debug-transformer-metal-runtime",
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                optimizer,
                RuntimeConfig.trainingDefaults(),
                WorkloadProfile.transformerHotPathMedium()
        );
        var workload = StandardWorkloads.transformerBlockHotPath(
                "debug_transformer_block",
                WorkloadProfile.transformerHotPathMedium()
        ).instantiate(new WorkloadEnvironment(profile));
        CompiledGraph compiled = CompiledGraph.compile(workload.root(), optimizer, CompileMode.TRAINING);
        var prepared = compiled.prepare(profile.runtime());
        var trace = prepared.executeTraced(profile.mode());

        long gpuSelected = prepared.prepareTrace().backendSelection().decisions().stream()
                .filter(decision -> decision.selected() && decision.selectedBackend() == ComputeBackend.GPU_METAL)
                .count();
        long gpuSteps = trace.steps().stream()
                .filter(step -> "GPU_METAL".equals(step.backend()))
                .count();
        long cpuSteps = trace.steps().stream()
                .filter(step -> "CPU".equals(step.backend()))
                .count();

        System.out.println("RUNTIME TRACE SUMMARY");
        System.out.println("selectedGpuRegions=" + gpuSelected
                + " totalSteps=" + trace.steps().size()
                + " gpuSteps=" + gpuSteps
                + " cpuSteps=" + cpuSteps
                + " cpuMaterializations=" + trace.cpuMaterializations().size());

        System.out.println("RUNTIME GPU STEPS");
        trace.steps().stream()
                .filter(step -> "GPU_METAL".equals(step.backend()))
                .forEach(step -> {
                    Map<String, Object> attrs = step.metadata().attributes();
                    System.out.println("index=" + step.index()
                            + " label=" + step.label()
                            + " op=" + step.opType()
                            + " kernel=" + step.kernel()
                            + " durationMs=" + String.format("%.6f", step.durationNs() / 1_000_000.0)
                            + " metalExecutionPath=" + attrs.get("metalExecutionPath")
                            + " metalUsedCpuFallback=" + attrs.get("metalUsedCpuFallback")
                            + " metalRoute=" + attrs.get("metalExecutionRoute")
                            + " metalRouteReasonCode=" + attrs.get("metalRouteReasonCode")
                            + " acceleratorBufferPath=" + attrs.get("acceleratorBufferExecutionPath")
                            + " acceleratorBufferReason=" + attrs.get("acceleratorBufferReasonCode")
                            + " nodeCount=" + attrs.get("metalSubgraphNodeCount")
                            + " externalInputs=" + attrs.get("metalExternalInputCount")
                            + " outputs=" + attrs.get("metalOutputCount")
                            + " inputBytes=" + attrs.get("metalInputBytes")
                            + " outputBytes=" + attrs.get("metalOutputBytes")
                            + " javaToNativeMs=" + ms(attrs.get("metalJavaToNativeCopyNs"))
                            + " nativeExecuteMs=" + ms(attrs.get("metalNativeExecuteNs"))
                            + " nativeDeviceCopyMs=" + ms(attrs.get("metalNativeDeviceCopyNs"))
                            + " nativeToJavaMs=" + ms(attrs.get("metalNativeToJavaCopyNs"))
                            + " bridgeTotalMs=" + ms(attrs.get("metalBridgeTotalNs"))
                            + " outputBufferWriteProven=" + attrs.get("metalOutputBufferWriteProven")
                            + " nativeCopyStrategy=" + attrs.get("metalNativeCopyStrategy"));
                    System.out.println("  metalSubgraphOps=" + attrs.get("metalSubgraphOps"));
                });

        System.out.println("RUNTIME NON-GPU STEPS");
        trace.steps().stream()
                .filter(step -> !"GPU_METAL".equals(step.backend()))
                .forEach(step -> System.out.println("index=" + step.index()
                        + " label=" + step.label()
                        + " op=" + step.opType()
                        + " backend=" + step.backend()
                        + " kernel=" + step.kernel()
                        + " durationMs=" + String.format("%.6f", step.durationNs() / 1_000_000.0)
                        + " attributes=" + step.metadata().attributes()));

        System.out.println("CPU MATERIALIZATIONS");
        trace.cpuMaterializations().forEach(materialization -> System.out.println(
                "node=" + materialization.nodeId()
                        + " reason=" + materialization.reason()
                        + " from=" + materialization.materializedFrom()
                        + " bytes=" + materialization.bytes()
                        + " durationMs=" + String.format("%.6f", materialization.durationNs() / 1_000_000.0)
                        + " completed=" + materialization.completed()
        ));
    }

    @Test
    void printGreedyTransformerGradientDiffs() {
        WorkloadProfile workloadProfile = WorkloadProfile.transformerHotPathMedium();
        ExecutionProfile cpuProfile = new ExecutionProfile(
                "debug-transformer-cpu",
                "debug-transformer-cpu",
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                CompileConfig.noGraphOptimizationBaseline(),
                RuntimeConfig.trainingDefaults(),
                workloadProfile
        );
        ExecutionProfile metalProfile = new ExecutionProfile(
                "debug-transformer-metal",
                "debug-transformer-metal",
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                CompileConfig.training()
                        .withBackendPlanning(config.compile.BackendPlanningConfig.autoAccelerator().withCpuRegions(CpuRegionConfig.defaults()))
                        .withRegionOptimization(CompileConfig.training().regionOptimization().withCpuFusion(CpuFusionConfig.defaults())),
                RuntimeConfig.trainingDefaults(),
                workloadProfile
        );

        WorkloadInstance cpu = StandardWorkloads.transformerBlockHotPath("debug_transformer_block", workloadProfile)
                .instantiate(new WorkloadEnvironment(cpuProfile));
        WorkloadInstance metal = StandardWorkloads.transformerBlockHotPath("debug_transformer_block", workloadProfile)
                .instantiate(new WorkloadEnvironment(metalProfile));

        CompiledGraph.compile(cpu.root(), cpuProfile.compile(), CompileMode.TRAINING)
                .prepare(cpuProfile.runtime())
                .execute(cpuProfile.mode());
        CompiledGraph.compile(metal.root(), metalProfile.compile(), CompileMode.TRAINING)
                .prepare(metalProfile.runtime())
                .execute(metalProfile.mode());

        Map<String, Tensor> cpuByLabel = tensorsByLabel(cpu.root());
        Map<String, Tensor> metalByLabel = tensorsByLabel(metal.root());
        for (String label : List.of(
                "TBLOCK_X",
                "TBLOCK_WQ", "TBLOCK_BQ",
                "TBLOCK_WK", "TBLOCK_BK",
                "TBLOCK_WV", "TBLOCK_BV",
                "TBLOCK_WO", "TBLOCK_BO",
                "TBLOCK_W1", "TBLOCK_B1",
                "TBLOCK_W2", "TBLOCK_B2"
        )) {
            Tensor cpuGrad = cpuByLabel.get(label).getGradient();
            Tensor metalGrad = metalByLabel.get(label).getGradient();
            Diff diff = diff(cpuGrad.toDoubleArrayCopy(), metalGrad.toDoubleArrayCopy());
            System.out.printf(
                    "%s maxAbs=%.12g maxRel=%.12g index=%d cpu=%.12g metal=%.12g%n",
                    label,
                    diff.maxAbs,
                    diff.maxRel,
                    diff.index,
                    diff.expected,
                    diff.actual
            );
        }
    }

    @Test
    void printCpuOptimizedTransformerGradientDiffs() {
        WorkloadProfile workloadProfile = WorkloadProfile.transformerHotPathMedium();
        ExecutionProfile baselineProfile = new ExecutionProfile(
                "debug-transformer-cpu-baseline",
                "debug-transformer-cpu-baseline",
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                CompileConfig.noGraphOptimizationBaseline(),
                RuntimeConfig.noOptNoVecNoPar(),
                workloadProfile
        );
        ExecutionProfile optimizedProfile = new ExecutionProfile(
                "debug-transformer-cpu-optimized",
                "debug-transformer-cpu-optimized",
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                CompileConfig.training()
                        .withBackendPlanning(config.compile.BackendPlanningConfig.cpuOnly().withCpuRegions(CpuRegionConfig.defaults()))
                        .withRegionOptimization(CompileConfig.training().regionOptimization().withCpuFusion(CpuFusionConfig.defaults())),
                RuntimeConfig.trainingDefaults(),
                workloadProfile
        );

        WorkloadInstance baseline = StandardWorkloads.transformerBlockHotPath("debug_transformer_block", workloadProfile)
                .instantiate(new WorkloadEnvironment(baselineProfile));
        WorkloadInstance optimized = StandardWorkloads.transformerBlockHotPath("debug_transformer_block", workloadProfile)
                .instantiate(new WorkloadEnvironment(optimizedProfile));

        CompiledGraph.compile(baseline.root(), baselineProfile.compile(), CompileMode.TRAINING)
                .prepare(baselineProfile.runtime())
                .execute(baselineProfile.mode());
        CompiledGraph.compile(optimized.root(), optimizedProfile.compile(), CompileMode.TRAINING)
                .prepare(optimizedProfile.runtime())
                .execute(optimizedProfile.mode());

        Map<String, Tensor> baselineByLabel = tensorsByLabel(baseline.root());
        Map<String, Tensor> optimizedByLabel = tensorsByLabel(optimized.root());
        for (String label : List.of(
                "TBLOCK_X",
                "TBLOCK_WQ", "TBLOCK_BQ",
                "TBLOCK_WK", "TBLOCK_BK",
                "TBLOCK_WV", "TBLOCK_BV",
                "TBLOCK_WO", "TBLOCK_BO",
                "TBLOCK_W1", "TBLOCK_B1",
                "TBLOCK_W2", "TBLOCK_B2"
        )) {
            Tensor baselineGrad = baselineByLabel.get(label).getGradient();
            Tensor optimizedGrad = optimizedByLabel.get(label).getGradient();
            Diff diff = diff(baselineGrad.toDoubleArrayCopy(), optimizedGrad.toDoubleArrayCopy());
            System.out.printf(
                    "CPU_OPT %s maxAbs=%.12g maxRel=%.12g index=%d baseline=%.12g optimized=%.12g%n",
                    label,
                    diff.maxAbs,
                    diff.maxRel,
                    diff.index,
                    diff.expected,
                    diff.actual
            );
        }
    }

    @Test
    void printLabeledTransformerForwardAndGradientDiffs() {
        LabeledTransformerGraph cpu = labeledTransformerGraph("cpu", false);
        LabeledTransformerGraph metal = labeledTransformerGraph("metal", true);

        ExecutionProfile cpuProfile = new ExecutionProfile(
                "debug-labeled-transformer-cpu",
                "debug-labeled-transformer-cpu",
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                CompileConfig.noGraphOptimizationBaseline(),
                RuntimeConfig.trainingDefaults(),
                WorkloadProfile.transformerHotPathMedium()
        );
        ExecutionProfile metalProfile = new ExecutionProfile(
                "debug-labeled-transformer-metal",
                "debug-labeled-transformer-metal",
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                CompileConfig.training()
                        .withBackendPlanning(config.compile.BackendPlanningConfig.autoAccelerator().withCpuRegions(CpuRegionConfig.defaults()))
                        .withRegionOptimization(CompileConfig.training().regionOptimization().withCpuFusion(CpuFusionConfig.defaults())),
                RuntimeConfig.trainingDefaults(),
                WorkloadProfile.transformerHotPathMedium()
        );

        CompiledGraph.compile(cpu.loss(), cpuProfile.compile(), CompileMode.TRAINING)
                .prepare(cpuProfile.runtime())
                .execute(cpuProfile.mode());
        var metalPrepared = CompiledGraph.compile(metal.loss(), metalProfile.compile(), CompileMode.TRAINING)
                .prepare(metalProfile.runtime());
        System.out.println("LABELED SELECTED DECISIONS");
        metalPrepared.prepareTrace().backendSelection().decisions().stream()
                .filter(decision -> decision.selected())
                .forEach(decision -> System.out.println(
                        "anchor=" + decision.anchorNodeId()
                                + " backend=" + decision.selectedBackend()
                                + " nodes=" + decision.nodeIds()
                                + " reason=" + decision.reason()
                                + " outputs=" + (decision.gpuLoweredRegionManifest() == null
                                ? "n/a"
                                : decision.gpuLoweredRegionManifest().outputNodeIds())
                ));
        metalPrepared.execute(metalProfile.mode());

        for (String label : List.of(
                "x2d",
                "qLinear", "qReshape", "q",
                "kLinear", "kReshape", "k",
                "vLinear", "vReshape", "v",
                "attention", "attentionMerged", "projected", "residual1",
                "ff1", "tanh", "geluLike", "ff2", "output", "loss"
        )) {
            Diff forward = diff(cpu.tensors().get(label).toDoubleArrayCopy(), metal.tensors().get(label).toDoubleArrayCopy());
            System.out.printf(
                    "FORWARD %s maxAbs=%.12g maxRel=%.12g index=%d cpu=%.12g metal=%.12g%n",
                    label,
                    forward.maxAbs,
                    forward.maxRel,
                    forward.index,
                    forward.expected,
                    forward.actual
            );
        }
        for (String label : List.of(
                "TBLOCK_X",
                "TBLOCK_WQ", "TBLOCK_BQ",
                "TBLOCK_WK", "TBLOCK_BK",
                "TBLOCK_WV", "TBLOCK_BV",
                "TBLOCK_WO", "TBLOCK_BO",
                "TBLOCK_W1", "TBLOCK_B1",
                "TBLOCK_W2", "TBLOCK_B2",
                "attention", "v", "vLinear", "projected", "residual1", "ff1", "geluLike"
        )) {
            Tensor cpuGrad = cpu.tensors().get(label).getGradient();
            Tensor metalGrad = metal.tensors().get(label).getGradient();
            if (cpuGrad == null || metalGrad == null) {
                System.out.println("GRAD " + label + " missing cpu=" + (cpuGrad == null) + " metal=" + (metalGrad == null));
                continue;
            }
            Diff gradient = diff(cpuGrad.toDoubleArrayCopy(), metalGrad.toDoubleArrayCopy());
            System.out.printf(
                    "GRAD %s maxAbs=%.12g maxRel=%.12g index=%d cpu=%.12g metal=%.12g%n",
                    label,
                    gradient.maxAbs,
                    gradient.maxRel,
                    gradient.index,
                    gradient.expected,
                    gradient.actual
            );
        }
    }

    @Test
    void printDirectSdpaBackwardValueDiff() {
        int[] qkvShape = new int[]{8, 8, 128, 64};
        int[] scoreShape = new int[]{8, 8, 128, 128};
        Tensor cpuQ = f32("cpuQ", qkvShape, 0.017f);
        Tensor cpuK = f32("cpuK", qkvShape, 0.019f);
        Tensor cpuV = f32("cpuV", qkvShape, 0.023f);
        Tensor cpuOutGrad = f32("cpuOutGrad", qkvShape, 0.011f);
        cpuQ.setRequiresGrad(true);
        cpuK.setRequiresGrad(true);
        cpuV.setRequiresGrad(true);
        Tensor cpuMask = causalMask(scoreShape);
        Tensor cpuAttention = cpuQ.scaledDotProductAttention(cpuK, cpuV, cpuMask, AttentionOptions.defaults().withScale(0.125));
        cpuAttention.setRequiresGrad(true);
        Tensor cpuLoss = cpuAttention.mul(cpuOutGrad);

        Tensor metalQ = f32("metalQ", qkvShape, 0.017f);
        Tensor metalK = f32("metalK", qkvShape, 0.019f);
        Tensor metalV = f32("metalV", qkvShape, 0.023f);
        Tensor metalOutGrad = f32("metalOutGrad", qkvShape, 0.011f);
        metalQ.setRequiresGrad(true);
        metalK.setRequiresGrad(true);
        metalV.setRequiresGrad(true);
        Tensor metalMask = causalMask(scoreShape);
        Tensor metalAttention = metalQ.scaledDotProductAttention(metalK, metalV, metalMask, AttentionOptions.defaults().withScale(0.125));
        metalAttention.setRequiresGrad(true);
        TensorInternalAccess.setBackendIntent(metalAttention, ComputeBackend.GPU_METAL);
        Tensor metalLoss = metalAttention.mul(metalOutGrad);

        ExecutionProfile cpuProfile = new ExecutionProfile(
                "debug-sdpa-backward-cpu",
                "debug-sdpa-backward-cpu",
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                CompileConfig.noGraphOptimizationBaseline(),
                RuntimeConfig.trainingDefaults()
        );
        ExecutionProfile metalProfile = new ExecutionProfile(
                "debug-sdpa-backward-metal",
                "debug-sdpa-backward-metal",
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                CompileConfig.training().withBackendPlanning(config.compile.BackendPlanningConfig.autoAccelerator()),
                requireMetalBuffers(RuntimeConfig.trainingDefaults())
        );

        CompiledGraph.compile(cpuLoss, cpuProfile.compile(), CompileMode.TRAINING)
                .prepare(cpuProfile.runtime())
                .execute(cpuProfile.mode());
        var metalPrepared = CompiledGraph.compile(metalLoss, metalProfile.compile(), CompileMode.TRAINING)
                .prepare(metalProfile.runtime())
        ;
        long metalBackwardSdpa = metalPrepared.backwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .filter(step -> step.compiledNode().operation() != null
                        && step.compiledNode().operation().opType() == operations.Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_BACKWARD)
                .count();
        System.out.println("METAL BACKWARD STEPS");
        metalPrepared.backwardSteps().forEach(step -> System.out.println(
                "node=" + step.compiledNode().label()
                        + " op=" + (step.compiledNode().operation() == null ? "LEAF" : step.compiledNode().operation().opType())
                        + " backend=" + step.metadata().backend()
                        + " role=" + step.metadata().partitionRole()
        ));
        System.out.println("METAL SELECTED DECISIONS");
        metalPrepared.prepareTrace().backendSelection().decisions().stream()
                .filter(decision -> decision.selected())
                .forEach(decision -> System.out.println(
                        "anchor=" + decision.anchorNodeId()
                                + " backend=" + decision.selectedBackend()
                                + " nodes=" + decision.nodeIds()
                                + " reason=" + decision.reason()
                                + " manifest=" + decision.gpuLoweredRegionManifest()
                ));
        metalPrepared.execute(metalProfile.mode());

        Diff diff = diff(cpuV.getGradient().toDoubleArrayCopy(), metalV.getGradient().toDoubleArrayCopy());
        System.out.printf(
                "SDPA_BACKWARD_VALUE_GRAD_V metalSteps=%d maxAbs=%.12g maxRel=%.12g index=%d cpu=%.12g metal=%.12g%n",
                metalBackwardSdpa,
                diff.maxAbs,
                diff.maxRel,
                diff.index,
                diff.expected,
                diff.actual
        );
    }

    @Test
    void printValueProjectionBackwardDiff() {
        int batch = 8;
        int seqLen = 128;
        int heads = 8;
        int valueDim = 64;
        int modelDim = 512;
        int tokenCount = batch * seqLen;
        int valueProjectionDim = heads * valueDim;

        Tensor cpuX = f32("cpuX", new int[]{batch, seqLen, modelDim}, 0.012f);
        Tensor cpuWv = f32("cpuWv", new int[]{modelDim, valueProjectionDim}, 0.035f);
        Tensor cpuBv = f32("cpuBv", new int[]{valueProjectionDim}, 0.010f);
        Tensor cpuDV = f32("cpuDV", new int[]{batch, heads, seqLen, valueDim}, 0.011f);
        cpuX.setRequiresGrad(true);
        cpuWv.setRequiresGrad(true);
        cpuBv.setRequiresGrad(true);
        Tensor cpuValue = cpuX.reshape(tokenCount, modelDim)
                .linear(cpuWv, cpuBv)
                .reshape(batch, seqLen, heads, valueDim)
                .permute(0, 2, 1, 3);
        Tensor cpuLoss = cpuValue.mul(cpuDV);

        Tensor metalX = f32("metalX", new int[]{batch, seqLen, modelDim}, 0.012f);
        Tensor metalWv = f32("metalWv", new int[]{modelDim, valueProjectionDim}, 0.035f);
        Tensor metalBv = f32("metalBv", new int[]{valueProjectionDim}, 0.010f);
        Tensor metalDV = f32("metalDV", new int[]{batch, heads, seqLen, valueDim}, 0.011f);
        metalX.setRequiresGrad(true);
        metalWv.setRequiresGrad(true);
        metalBv.setRequiresGrad(true);
        Tensor metalValue = metalX.reshape(tokenCount, modelDim)
                .linear(metalWv, metalBv)
                .reshape(batch, seqLen, heads, valueDim)
                .permute(0, 2, 1, 3);
        TensorInternalAccess.setBackendIntent(metalValue, ComputeBackend.GPU_METAL);
        Tensor metalLoss = metalValue.mul(metalDV);

        RuntimeConfig metalRuntime = requireMetalBuffers(RuntimeConfig.trainingDefaults());
        CompiledGraph.compile(cpuLoss, CompileConfig.noGraphOptimizationBaseline(), CompileMode.TRAINING)
                .prepare(RuntimeConfig.trainingDefaults())
                .execute(ExecutionMode.FORWARD_BACKWARD);
        var metalPrepared = CompiledGraph.compile(
                        metalLoss,
                        CompileConfig.training().withBackendPlanning(config.compile.BackendPlanningConfig.autoAccelerator()),
                        CompileMode.TRAINING
                )
                .prepare(metalRuntime);
        System.out.println("VALUE PROJECTION SELECTED DECISIONS");
        metalPrepared.prepareTrace().backendSelection().decisions().stream()
                .filter(decision -> decision.selected())
                .forEach(decision -> System.out.println(
                        "anchor=" + decision.anchorNodeId()
                                + " backend=" + decision.selectedBackend()
                                + " nodes=" + decision.nodeIds()
                                + " reason=" + decision.reason()
                                + " outputs=" + (decision.gpuLoweredRegionManifest() == null
                                ? "n/a"
                                : decision.gpuLoweredRegionManifest().outputNodeIds())
                ));
        metalPrepared.execute(ExecutionMode.FORWARD_BACKWARD);

        Diff wvDiff = diff(cpuWv.getGradient().toDoubleArrayCopy(), metalWv.getGradient().toDoubleArrayCopy());
        Diff bvDiff = diff(cpuBv.getGradient().toDoubleArrayCopy(), metalBv.getGradient().toDoubleArrayCopy());
        Diff xDiff = diff(cpuX.getGradient().toDoubleArrayCopy(), metalX.getGradient().toDoubleArrayCopy());
        System.out.printf(
                "VALUE_PROJECTION_WV maxAbs=%.12g maxRel=%.12g index=%d cpu=%.12g metal=%.12g%n",
                wvDiff.maxAbs,
                wvDiff.maxRel,
                wvDiff.index,
                wvDiff.expected,
                wvDiff.actual
        );
        System.out.printf(
                "VALUE_PROJECTION_BV maxAbs=%.12g maxRel=%.12g index=%d cpu=%.12g metal=%.12g%n",
                bvDiff.maxAbs,
                bvDiff.maxRel,
                bvDiff.index,
                bvDiff.expected,
                bvDiff.actual
        );
        System.out.printf(
                "VALUE_PROJECTION_X maxAbs=%.12g maxRel=%.12g index=%d cpu=%.12g metal=%.12g%n",
                xDiff.maxAbs,
                xDiff.maxRel,
                xDiff.index,
                xDiff.expected,
                xDiff.actual
        );
    }

    @Test
    void printSdpaToValueProjectionBackwardDiff() {
        int batch = 8;
        int seqLen = 128;
        int heads = 8;
        int valueDim = 64;
        int modelDim = 512;
        int tokenCount = batch * seqLen;
        int valueProjectionDim = heads * valueDim;
        int[] qkvShape = new int[]{batch, heads, seqLen, valueDim};
        int[] scoreShape = new int[]{batch, heads, seqLen, seqLen};

        Tensor cpuX = f32("cpuX", new int[]{batch, seqLen, modelDim}, 0.012f);
        Tensor cpuWv = f32("cpuWv", new int[]{modelDim, valueProjectionDim}, 0.035f);
        Tensor cpuBv = f32("cpuBv", new int[]{valueProjectionDim}, 0.010f);
        Tensor cpuQ = f32("cpuQ", qkvShape, 0.017f);
        Tensor cpuK = f32("cpuK", qkvShape, 0.019f);
        Tensor cpuOutGrad = f32("cpuOutGrad", qkvShape, 0.011f);
        Tensor cpuMask = causalMask(scoreShape);
        cpuX.setRequiresGrad(true);
        cpuWv.setRequiresGrad(true);
        cpuBv.setRequiresGrad(true);
        Tensor cpuV = cpuX.reshape(tokenCount, modelDim)
                .linear(cpuWv, cpuBv)
                .reshape(batch, seqLen, heads, valueDim)
                .permute(0, 2, 1, 3);
        Tensor cpuAttention = cpuQ.scaledDotProductAttention(cpuK, cpuV, cpuMask, AttentionOptions.defaults().withScale(0.125));
        Tensor cpuLoss = cpuAttention.mul(cpuOutGrad);

        Tensor metalX = f32("metalX", new int[]{batch, seqLen, modelDim}, 0.012f);
        Tensor metalWv = f32("metalWv", new int[]{modelDim, valueProjectionDim}, 0.035f);
        Tensor metalBv = f32("metalBv", new int[]{valueProjectionDim}, 0.010f);
        Tensor metalQ = f32("metalQ", qkvShape, 0.017f);
        Tensor metalK = f32("metalK", qkvShape, 0.019f);
        Tensor metalOutGrad = f32("metalOutGrad", qkvShape, 0.011f);
        Tensor metalMask = causalMask(scoreShape);
        metalX.setRequiresGrad(true);
        metalWv.setRequiresGrad(true);
        metalBv.setRequiresGrad(true);
        Tensor metalV = metalX.reshape(tokenCount, modelDim)
                .linear(metalWv, metalBv)
                .reshape(batch, seqLen, heads, valueDim)
                .permute(0, 2, 1, 3);
        Tensor metalAttention = metalQ.scaledDotProductAttention(metalK, metalV, metalMask, AttentionOptions.defaults().withScale(0.125));
        TensorInternalAccess.setBackendIntent(metalAttention, ComputeBackend.GPU_METAL);
        Tensor metalLoss = metalAttention.mul(metalOutGrad);

        RuntimeConfig metalRuntime = requireMetalBuffers(RuntimeConfig.trainingDefaults());
        CompiledGraph.compile(cpuLoss, CompileConfig.noGraphOptimizationBaseline(), CompileMode.TRAINING)
                .prepare(RuntimeConfig.trainingDefaults())
                .execute(ExecutionMode.FORWARD_BACKWARD);
        var metalPrepared = CompiledGraph.compile(
                        metalLoss,
                        CompileConfig.training().withBackendPlanning(config.compile.BackendPlanningConfig.autoAccelerator()),
                        CompileMode.TRAINING
                )
                .prepare(metalRuntime);
        System.out.println("SDPA VALUE PROJECTION SELECTED DECISIONS");
        metalPrepared.prepareTrace().backendSelection().decisions().stream()
                .filter(decision -> decision.selected())
                .forEach(decision -> System.out.println(
                        "anchor=" + decision.anchorNodeId()
                                + " backend=" + decision.selectedBackend()
                                + " nodes=" + decision.nodeIds()
                                + " reason=" + decision.reason()
                                + " outputs=" + (decision.gpuLoweredRegionManifest() == null
                                ? "n/a"
                                : decision.gpuLoweredRegionManifest().outputNodeIds())
                ));
        metalPrepared.execute(ExecutionMode.FORWARD_BACKWARD);

        Diff wvDiff = diff(cpuWv.getGradient().toDoubleArrayCopy(), metalWv.getGradient().toDoubleArrayCopy());
        Diff bvDiff = diff(cpuBv.getGradient().toDoubleArrayCopy(), metalBv.getGradient().toDoubleArrayCopy());
        Diff xDiff = diff(cpuX.getGradient().toDoubleArrayCopy(), metalX.getGradient().toDoubleArrayCopy());
        System.out.printf(
                "SDPA_VALUE_PROJECTION_WV maxAbs=%.12g maxRel=%.12g index=%d cpu=%.12g metal=%.12g%n",
                wvDiff.maxAbs,
                wvDiff.maxRel,
                wvDiff.index,
                wvDiff.expected,
                wvDiff.actual
        );
        System.out.printf(
                "SDPA_VALUE_PROJECTION_BV maxAbs=%.12g maxRel=%.12g index=%d cpu=%.12g metal=%.12g%n",
                bvDiff.maxAbs,
                bvDiff.maxRel,
                bvDiff.index,
                bvDiff.expected,
                bvDiff.actual
        );
        System.out.printf(
                "SDPA_VALUE_PROJECTION_X maxAbs=%.12g maxRel=%.12g index=%d cpu=%.12g metal=%.12g%n",
                xDiff.maxAbs,
                xDiff.maxRel,
                xDiff.index,
                xDiff.expected,
                xDiff.actual
        );
    }

    @Test
    void printTransformerTailAttentionGradientDiff() {
        int batch = 8;
        int seqLen = 128;
        int heads = 8;
        int valueDim = 64;
        int modelDim = 512;
        int ffHiddenDim = 2048;
        int tokenCount = batch * seqLen;
        int valueProjectionDim = heads * valueDim;
        int[] attentionShape = new int[]{batch, heads, seqLen, valueDim};

        Tensor cpuX = f32("cpuX", new int[]{batch, seqLen, modelDim}, 0.012f);
        Tensor cpuAttention = f32("cpuAttention", attentionShape, 0.023f);
        Tensor cpuWo = f32("cpuWo", new int[]{valueProjectionDim, modelDim}, 0.035f);
        Tensor cpuBo = f32("cpuBo", new int[]{modelDim}, 0.010f);
        Tensor cpuW1 = f32("cpuW1", new int[]{modelDim, ffHiddenDim}, 0.030f);
        Tensor cpuB1 = f32("cpuB1", new int[]{ffHiddenDim}, 0.010f);
        Tensor cpuW2 = f32("cpuW2", new int[]{ffHiddenDim, modelDim}, 0.030f);
        Tensor cpuB2 = f32("cpuB2", new int[]{modelDim}, 0.010f);
        cpuX.setRequiresGrad(true);
        cpuAttention.setRequiresGrad(true);
        cpuWo.setRequiresGrad(true);
        cpuBo.setRequiresGrad(true);
        cpuW1.setRequiresGrad(true);
        cpuB1.setRequiresGrad(true);
        cpuW2.setRequiresGrad(true);
        cpuB2.setRequiresGrad(true);
        Tensor cpuTailLoss = transformerTailLoss(
                cpuX, cpuAttention, cpuWo, cpuBo, cpuW1, cpuB1, cpuW2, cpuB2,
                batch, seqLen, heads, valueDim, modelDim, valueProjectionDim, tokenCount
        );

        Tensor metalX = f32("metalX", new int[]{batch, seqLen, modelDim}, 0.012f);
        Tensor metalAttention = f32("metalAttention", attentionShape, 0.023f);
        Tensor metalWo = f32("metalWo", new int[]{valueProjectionDim, modelDim}, 0.035f);
        Tensor metalBo = f32("metalBo", new int[]{modelDim}, 0.010f);
        Tensor metalW1 = f32("metalW1", new int[]{modelDim, ffHiddenDim}, 0.030f);
        Tensor metalB1 = f32("metalB1", new int[]{ffHiddenDim}, 0.010f);
        Tensor metalW2 = f32("metalW2", new int[]{ffHiddenDim, modelDim}, 0.030f);
        Tensor metalB2 = f32("metalB2", new int[]{modelDim}, 0.010f);
        metalX.setRequiresGrad(true);
        metalAttention.setRequiresGrad(true);
        metalWo.setRequiresGrad(true);
        metalBo.setRequiresGrad(true);
        metalW1.setRequiresGrad(true);
        metalB1.setRequiresGrad(true);
        metalW2.setRequiresGrad(true);
        metalB2.setRequiresGrad(true);
        Tensor metalTailLoss = transformerTailLoss(
                metalX, metalAttention, metalWo, metalBo, metalW1, metalB1, metalW2, metalB2,
                batch, seqLen, heads, valueDim, modelDim, valueProjectionDim, tokenCount
        );
        TensorInternalAccess.setBackendIntent(metalTailLoss, ComputeBackend.GPU_METAL);

        RuntimeConfig metalRuntime = requireMetalBuffers(RuntimeConfig.trainingDefaults());
        CompiledGraph.compile(cpuTailLoss, CompileConfig.noGraphOptimizationBaseline(), CompileMode.TRAINING)
                .prepare(RuntimeConfig.trainingDefaults())
                .execute(ExecutionMode.FORWARD_BACKWARD);
        var metalPrepared = CompiledGraph.compile(
                        metalTailLoss,
                        CompileConfig.training().withBackendPlanning(config.compile.BackendPlanningConfig.autoAccelerator()),
                        CompileMode.TRAINING
                )
                .prepare(metalRuntime);
        System.out.println("TAIL SELECTED DECISIONS");
        metalPrepared.prepareTrace().backendSelection().decisions().stream()
                .filter(decision -> decision.selected())
                .forEach(decision -> System.out.println(
                        "anchor=" + decision.anchorNodeId()
                                + " backend=" + decision.selectedBackend()
                                + " nodes=" + decision.nodeIds()
                                + " reason=" + decision.reason()
                                + " outputs=" + (decision.gpuLoweredRegionManifest() == null
                                ? "n/a"
                                : decision.gpuLoweredRegionManifest().outputNodeIds())
                ));
        metalPrepared.execute(ExecutionMode.FORWARD_BACKWARD);

        Diff attentionDiff = diff(cpuAttention.getGradient().toDoubleArrayCopy(), metalAttention.getGradient().toDoubleArrayCopy());
        Diff woDiff = diff(cpuWo.getGradient().toDoubleArrayCopy(), metalWo.getGradient().toDoubleArrayCopy());
        System.out.printf(
                "TAIL_ATTENTION_GRAD maxAbs=%.12g maxRel=%.12g index=%d cpu=%.12g metal=%.12g%n",
                attentionDiff.maxAbs,
                attentionDiff.maxRel,
                attentionDiff.index,
                attentionDiff.expected,
                attentionDiff.actual
        );
        System.out.printf(
                "TAIL_WO_GRAD maxAbs=%.12g maxRel=%.12g index=%d cpu=%.12g metal=%.12g%n",
                woDiff.maxAbs,
                woDiff.maxRel,
                woDiff.index,
                woDiff.expected,
                woDiff.actual
        );
    }

    private static Tensor transformerTailLoss(
            Tensor x,
            Tensor attention,
            Tensor wo,
            Tensor bo,
            Tensor w1,
            Tensor b1,
            Tensor w2,
            Tensor b2,
            int batch,
            int seqLen,
            int heads,
            int valueDim,
            int modelDim,
            int valueProjectionDim,
            int tokenCount
    ) {
        Tensor x2d = x.reshape(tokenCount, modelDim);
        Tensor attentionMerged = attention.permute(0, 2, 1, 3).reshape(tokenCount, valueProjectionDim);
        Tensor projected = attentionMerged.linear(wo, bo);
        Tensor residual1 = x2d.add(projected);
        Tensor ff1 = residual1.linear(w1, b1);
        Tensor geluLike = ff1.mul(0.5).mul(ff1.tanh().add(Tensor.scalar(1.0, x.getDataType())));
        Tensor ff2 = geluLike.linear(w2, b2);
        Tensor output = residual1.add(ff2);
        return output.mul(output).mean();
    }

    private static LabeledTransformerGraph labeledTransformerGraph(String prefix, boolean preferMetal) {
        int batch = 8;
        int seqLen = 128;
        int heads = 8;
        int headDim = 64;
        int valueDim = 64;
        int modelDim = 512;
        int ffHiddenDim = 2048;
        int tokenCount = batch * seqLen;
        int qkProjectionDim = heads * headDim;
        int valueProjectionDim = heads * valueDim;
        Map<String, Tensor> tensors = new java.util.LinkedHashMap<>();

        Tensor x = workloadTensor("TBLOCK_X", 901, true, 0.12, batch, seqLen, modelDim);
        Tensor wq = workloadTensor("TBLOCK_WQ", 902, true, 0.035, modelDim, qkProjectionDim);
        Tensor bq = workloadTensor("TBLOCK_BQ", 903, true, 0.010, qkProjectionDim);
        Tensor wk = workloadTensor("TBLOCK_WK", 904, true, 0.035, modelDim, qkProjectionDim);
        Tensor bk = workloadTensor("TBLOCK_BK", 905, true, 0.010, qkProjectionDim);
        Tensor wv = workloadTensor("TBLOCK_WV", 906, true, 0.035, modelDim, valueProjectionDim);
        Tensor bv = workloadTensor("TBLOCK_BV", 907, true, 0.010, valueProjectionDim);
        Tensor wo = workloadTensor("TBLOCK_WO", 908, true, 0.035, valueProjectionDim, modelDim);
        Tensor bo = workloadTensor("TBLOCK_BO", 909, true, 0.010, modelDim);
        Tensor w1 = workloadTensor("TBLOCK_W1", 910, true, 0.030, modelDim, ffHiddenDim);
        Tensor b1 = workloadTensor("TBLOCK_B1", 911, true, 0.010, ffHiddenDim);
        Tensor w2 = workloadTensor("TBLOCK_W2", 912, true, 0.030, ffHiddenDim, modelDim);
        Tensor b2 = workloadTensor("TBLOCK_B2", 913, true, 0.010, modelDim);
        for (Tensor leaf : List.of(x, wq, bq, wk, bk, wv, bv, wo, bo, w1, b1, w2, b2)) {
            tensors.put(leaf.getLabel(), leaf);
        }

        Tensor x2d = labeled(tensors, "x2d", x.reshape(tokenCount, modelDim));

        Tensor qLinear = labeled(tensors, "qLinear", x2d.linear(wq, bq));
        Tensor qReshape = labeled(tensors, "qReshape", qLinear.reshape(batch, seqLen, heads, headDim));
        Tensor q = labeled(tensors, "q", qReshape.permute(0, 2, 1, 3));

        Tensor kLinear = labeled(tensors, "kLinear", x2d.linear(wk, bk));
        Tensor kReshape = labeled(tensors, "kReshape", kLinear.reshape(batch, seqLen, heads, headDim));
        Tensor k = labeled(tensors, "k", kReshape.permute(0, 2, 1, 3));

        Tensor vLinear = labeled(tensors, "vLinear", x2d.linear(wv, bv));
        Tensor vReshape = labeled(tensors, "vReshape", vLinear.reshape(batch, seqLen, heads, valueDim));
        Tensor v = labeled(tensors, "v", vReshape.permute(0, 2, 1, 3));

        Tensor attention = labeled(tensors, "attention", q.scaledDotProductAttention(k, v, new AttentionOptions(true, null)));
        Tensor attentionMerged = labeled(tensors, "attentionMerged", attention.permute(0, 2, 1, 3).reshape(tokenCount, valueProjectionDim));
        Tensor projected = labeled(tensors, "projected", attentionMerged.linear(wo, bo));
        Tensor residual1 = labeled(tensors, "residual1", x2d.add(projected));
        Tensor ff1 = labeled(tensors, "ff1", residual1.linear(w1, b1));
        Tensor tanh = labeled(tensors, "tanh", ff1.tanh());
        Tensor geluLike = labeled(tensors, "geluLike", ff1.mul(0.5).mul(tanh.add(Tensor.scalar(1.0, DataType.FLOAT32))));
        Tensor ff2 = labeled(tensors, "ff2", geluLike.linear(w2, b2));
        Tensor output = labeled(tensors, "output", residual1.add(ff2));
        Tensor loss = labeled(tensors, "loss", output.mul(output).mean());
        if (preferMetal) {
            TensorInternalAccess.setBackendIntent(loss, ComputeBackend.GPU_METAL);
        }
        return new LabeledTransformerGraph(prefix, loss, tensors);
    }

    private static Tensor labeled(Map<String, Tensor> tensors, String label, Tensor tensor) {
        tensor.setLabel(label);
        tensors.put(label, tensor);
        return tensor;
    }

    private static Tensor workloadTensor(String label, int seed, boolean requiresGrad, double scale, int... shape) {
        Tensor tensor = TensorDataFactory.shapedTensor(
                label,
                workloadRandomData(flatSize(shape), seed, scale),
                requiresGrad,
                DataType.FLOAT32,
                shape
        );
        return tensor;
    }

    private static int flatSize(int[] shape) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        return size;
    }

    private static double[] workloadRandomData(int size, int seed, double scale) {
        java.util.Random random = new java.util.Random(seed);
        double[] out = new double[size];
        for (int i = 0; i < size; i++) {
            double wave = Math.sin(i * 0.013 + seed * 0.017) + Math.cos(i * 0.007 + seed * 0.031);
            out[i] = scale * (wave + (random.nextDouble() - 0.5) * 0.2);
        }
        return out;
    }

    private record LabeledTransformerGraph(String prefix, Tensor loss, Map<String, Tensor> tensors) {
    }

    private static Tensor f32(String label, int[] shape, float seed) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        float[] values = new float[size];
        for (int i = 0; i < values.length; i++) {
            values[i] = (float) (Math.sin(seed * (i + 1)) * 0.02 + Math.cos(seed * 0.37 * (i + 3)) * 0.01);
        }
        return new Tensor(values, shape.clone(), null, label, DataType.FLOAT32);
    }

    private static RuntimeConfig requireMetalBuffers(RuntimeConfig runtime) {
        AcceleratorBackendConfig requiredMetal = runtime.accelerator().metal().withBuffer(
                new AcceleratorBufferConfig(AcceleratorBufferBindingMode.REQUIRE, true, 0)
        );
        AcceleratorConfig accelerator = runtime.accelerator().withMetal(requiredMetal);
        return runtime.withAccelerator(accelerator);
    }

    private static Tensor causalMask(int[] scoreShape) {
        byte[] values = new byte[scoreShape[0] * scoreShape[1] * scoreShape[2] * scoreShape[3]];
        int queryLen = scoreShape[2];
        int keyLen = scoreShape[3];
        for (int b = 0; b < scoreShape[0]; b++) {
            for (int h = 0; h < scoreShape[1]; h++) {
                int base = ((b * scoreShape[1]) + h) * queryLen * keyLen;
                for (int q = 0; q < queryLen; q++) {
                    for (int k = 0; k < keyLen; k++) {
                        values[base + q * keyLen + k] = (byte) (k <= q ? 1 : 0);
                    }
                }
            }
        }
        return new Tensor(values, scoreShape.clone(), null, "causalMask", DataType.BOOL);
    }

    private static Map<String, Tensor> tensorsByLabel(Tensor root) {
        return root.topologicalSort().stream()
                .filter(tensor -> tensor.getLabel() != null && !tensor.getLabel().isBlank())
                .collect(Collectors.toMap(Tensor::getLabel, tensor -> tensor, (a, b) -> a));
    }

    private static Diff diff(double[] expected, double[] actual) {
        double maxAbs = 0.0d;
        double maxRel = 0.0d;
        int index = -1;
        for (int i = 0; i < expected.length; i++) {
            double abs = Math.abs(actual[i] - expected[i]);
            double rel = abs / Math.max(1.0e-30d, Math.abs(expected[i]));
            if (abs > maxAbs) {
                maxAbs = abs;
                maxRel = rel;
                index = i;
            }
        }
        return new Diff(maxAbs, maxRel, index, index < 0 ? 0.0d : expected[index], index < 0 ? 0.0d : actual[index]);
    }

    private static String ms(Object nanos) {
        if (!(nanos instanceof Number number)) {
            return "n/a";
        }
        return String.format("%.6f", number.longValue() / 1_000_000.0);
    }

    private record Diff(double maxAbs, double maxRel, int index, double expected, double actual) {
    }
}

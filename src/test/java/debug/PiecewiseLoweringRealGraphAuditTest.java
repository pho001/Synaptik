package debug;

import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.optimizer.PiecewiseLoweringConfig;
import graph.CompiledGraph;
import graph.model.CompiledNode;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tuning.workload.StandardWorkloads;
import tuning.workload.WorkloadEnvironment;
import tuning.workload.WorkloadInstance;
import tuning.workload.WorkloadSpec;

import java.util.EnumMap;
import java.util.List;

final class PiecewiseLoweringRealGraphAuditTest {
    @Test
    void auditRealForwardBackwardWorkloadsForPiecewiseLoweringHits() {
        List<WorkloadSpec> workloads = List.of(
                StandardWorkloads.transformerHotPath("audit_transformer_hot_path"),
                StandardWorkloads.mlpClassification("audit_mlp_classifier", 16, 32, 48, 24, 6, tensor.loss.LossReduction.MEAN),
                StandardWorkloads.indexedLoss("audit_cross_entropy_indices", tuning.workload.LossWorkloadSpec.LossKind.CROSS_ENTROPY_FROM_INDICES, 8, 16, tensor.loss.LossReduction.MEAN),
                StandardWorkloads.normalization("audit_layer_norm", tuning.workload.NormalizationWorkloadSpec.NormalizationKind.LAYER_NORM, 4, 64, 8, 1, 1e-5)
        );

        CompileConfig baseline = CompileConfig.training();
        CompileConfig piecewise = baseline.withGraphOptimization(
                baseline.graphOptimization().withRewrite(
                        baseline.graphOptimization().rewrite().withPiecewiseLowering(PiecewiseLoweringConfig.aggressiveDefaults())
                )
        );

        for (WorkloadSpec spec : workloads) {
            var profile = new config.profile.ExecutionProfile(
                    spec.name(),
                    spec.name(),
                    DataType.FLOAT32,
                    ExecutionMode.FORWARD_BACKWARD,
                    baseline,
                    config.runtime.RuntimeConfig.trainingDefaults(),
                    config.profile.WorkloadProfile.none()
            );
            System.out.println("PIECEWISE_AUDIT workload=" + spec.name());
            try {
                WorkloadInstance baselineInstance = spec.instantiate(new WorkloadEnvironment(profile));
                WorkloadInstance piecewiseInstance = spec.instantiate(new WorkloadEnvironment(profile));
                CompiledGraph baselineGraph = CompiledGraph.compile(baselineInstance.root(), baseline);
                CompiledGraph piecewiseGraph = CompiledGraph.compile(piecewiseInstance.root(), piecewise);

                EnumMap<Operation.OpType, Integer> baselineCounts = countInterestingOps(baselineGraph);
                EnumMap<Operation.OpType, Integer> piecewiseCounts = countInterestingOps(piecewiseGraph);

                System.out.println("baselineNodes=" + baselineGraph.program().compiledNodes().size());
                System.out.println("piecewiseNodes=" + piecewiseGraph.program().compiledNodes().size());
                System.out.println("baselineInteresting=" + baselineCounts);
                System.out.println("piecewiseInteresting=" + piecewiseCounts);
                System.out.println("deltaNodes=" + (piecewiseGraph.program().compiledNodes().size() - baselineGraph.program().compiledNodes().size()));
            } catch (Exception e) {
                System.out.println("compileError=" + e.getClass().getSimpleName() + ": " + e.getMessage());
            }
            System.out.println();
        }
    }

    private static EnumMap<Operation.OpType, Integer> countInterestingOps(CompiledGraph graph) {
        EnumMap<Operation.OpType, Integer> counts = new EnumMap<>(Operation.OpType.class);
        for (CompiledNode tensor : graph.program().compiledNodes()) {
            Operation op = tensor.operation();
            if (op == null) {
                continue;
            }
            Operation.OpType type = op.opType();
            if (type == Operation.OpType.WHERE
                    || type == Operation.OpType.RELU
                    || type == Operation.OpType.SIGMOID
                    || type == Operation.OpType.CLAMP_MIN
                    || type == Operation.OpType.CLAMP_MAX
                    || type == Operation.OpType.EXP
                    || type == Operation.OpType.INV
                    || type == Operation.OpType.GT
                    || type == Operation.OpType.GE
                    || type == Operation.OpType.LT
                    || type == Operation.OpType.LE) {
                counts.merge(type, 1, Integer::sum);
            }
        }
        return counts;
    }
}

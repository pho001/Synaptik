package graph.execution.runner;

import backend.ComputeEngine;
import runtime.contract.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import graph.execution.PreparedExecutionStep;
import graph.execution.device.DeviceLayoutViewPropagator;
import graph.execution.plan.InputResidencyRequirement;
import graph.execution.plan.OutputResidencyEffect;
import trace.execution.ExecutionStepTrace;
import runtime.runner.StepExecutionTracer;

import java.util.List;

/**
 * Executes prepared node steps against one run-scoped execution context.
 */
public final class PreparedExecutionRunner {
    private PreparedExecutionRunner() {
    }

    public static void executeSteps(
            List<PreparedExecutionStep> steps,
            ExecutionContext context,
            boolean captureTrace,
            List<ExecutionStepTrace> traces,
            int startIndex
    ) {
        for (int i = 0; i < steps.size(); i++) {
            PreparedExecutionStep step = steps.get(i);
            long t0 = captureTrace ? System.nanoTime() : 0L;
            if (DeviceLayoutViewPropagator.tryPropagate(step, context)) {
                if (captureTrace) {
                    traces.add(StepExecutionTracer.toStepTrace(startIndex + i, step, System.nanoTime() - t0, context));
                }
                continue;
            }
            requireCpuReadableInputs(step, context);
            ComputeEngine.compute(step.compiledNode(), step.metadata(), context);
            markResidencyAfterStep(step, context);
            if (captureTrace) {
                traces.add(StepExecutionTracer.toStepTrace(startIndex + i, step, System.nanoTime() - t0, context));
            }
        }
    }

    private static void markResidencyAfterStep(PreparedExecutionStep step, ExecutionContext context) {
        OutputResidencyEffect effect = step.metadata().outputResidencyEffect();
        if (effect.mode() == OutputResidencyEffect.Mode.NONE) {
            return;
        }
        for (int nodeId : step.boundaryOutputNodeIds()) {
            markResidencyForNode(nodeId, effect, context);
        }
    }

    private static void markResidencyForNode(
            int nodeId,
            OutputResidencyEffect effect,
            ExecutionContext context
    ) {
        if (effect.mode() == OutputResidencyEffect.Mode.CPU_CURRENT_PRESERVE_NATIVE) {
            var residency = context.residencyForNodeId(nodeId);
            if (residency != null && residency.nativeCurrent()) {
                return;
            }
            context.markCpuCurrent(nodeId, effect.reason());
            return;
        }
        var residency = context.residencyForNodeId(nodeId);
        if (residency == null || (!residency.cpuCurrent() && !residency.deviceCurrent())) {
            context.markCpuCurrent(nodeId, effect.reason());
        }
    }

    private static void requireCpuReadableInputs(PreparedExecutionStep step, ExecutionContext context) {
        InputResidencyRequirement requirement = step.metadata().inputResidencyRequirement();
        List<Integer> inputIds = inputIds(step);
        switch (requirement.mode()) {
            case NONE -> {
            }
            case CPU_READABLE_FIRST -> {
                if (!inputIds.isEmpty()) {
                    context.requireCpuReadable(inputIds.get(0), CpuMaterializationReason.CPU_CONSUMER);
                }
            }
            case CPU_READABLE_ALL -> {
                for (int inputId : inputIds) {
                    context.requireCpuReadable(inputId, CpuMaterializationReason.CPU_CONSUMER);
                }
            }
        }
    }

    private static List<Integer> inputIds(PreparedExecutionStep step) {
        return step.metadata().executionInputNodeIds().isEmpty()
                ? step.compiledNode().inputIds()
                : step.metadata().executionInputNodeIds();
    }

}

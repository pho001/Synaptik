package graph.execution.runner;

import backend.ComputeBackend;
import backend.ComputeEngine;
import backend.accelerator.exec.PreparedAcceleratorExecutable;
import backend.cpu.nativecpu.PreparedNativeCpuInputPolicy;
import backend.cpu.nativecpu.PreparedNativeCpuPlan;
import backend.memory.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import graph.execution.PreparedNodeExecution;
import graph.execution.device.DeviceLayoutViewPropagator;
import graph.execution.trace.ExecutionStepTrace;
import graph.execution.trace.contrib.StepExecutionTracer;

import java.util.List;

/**
 * Executes prepared node steps against one run-scoped execution context.
 */
public final class PreparedExecutionRunner {
    private PreparedExecutionRunner() {
    }

    public static void executeSteps(
            List<PreparedNodeExecution> steps,
            ExecutionContext context,
            boolean captureTrace,
            List<ExecutionStepTrace> traces,
            int startIndex
    ) {
        for (int i = 0; i < steps.size(); i++) {
            PreparedNodeExecution step = steps.get(i);
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

    private static void markResidencyAfterStep(PreparedNodeExecution step, ExecutionContext context) {
        int nodeId = step.compiledNode().id();
        if (step.metadata().backend() == ComputeBackend.CPU) {
            var residency = context.residencyForNodeId(nodeId);
            if (residency != null && residency.nativeCurrent()) {
                return;
            }
            context.markCpuCurrent(nodeId, residencyReason(step));
            return;
        }
        var residency = context.residencyForNodeId(nodeId);
        if (residency == null || (!residency.cpuCurrent() && !residency.deviceCurrent())) {
            context.markCpuCurrent(nodeId, residencyReason(step));
        }
    }

    private static void requireCpuReadableInputs(PreparedNodeExecution step, ExecutionContext context) {
        if (step.metadata().backend() != ComputeBackend.CPU) {
            return;
        }
        if (step.metadata().cpuRegionExecutable() != null) {
            return;
        }
        PreparedNativeCpuPlan nativeCpuPlan = step.metadata().cpuPlan() == null
                ? null
                : step.metadata().cpuPlan().nativeCpuPlan();
        if (nativeCpuPlan != null) {
            if (nativeCpuPlan.inputPolicy() == PreparedNativeCpuInputPolicy.ALL_NATIVE) {
                return;
            }
            if (nativeCpuPlan.inputPolicy() == PreparedNativeCpuInputPolicy.CONDITION_CPU_VALUES_NATIVE) {
                List<Integer> inputIds = inputIds(step);
                if (!inputIds.isEmpty()) {
                    context.requireCpuReadable(inputIds.get(0), CpuMaterializationReason.CPU_CONSUMER);
                }
                return;
            }
        }
        List<Integer> inputIds = inputIds(step);
        for (int inputId : inputIds) {
            context.requireCpuReadable(inputId, CpuMaterializationReason.CPU_CONSUMER);
        }
    }

    private static List<Integer> inputIds(PreparedNodeExecution step) {
        return step.metadata().executionInputNodeIds().isEmpty()
                ? step.compiledNode().inputIds()
                : step.metadata().executionInputNodeIds();
    }

    private static String residencyReason(PreparedNodeExecution step) {
        PreparedAcceleratorExecutable executable = step == null ? null : step.metadata().acceleratorExecutable();
        if (executable != null) {
            return executable.outputResidencyReason();
        }
        return "backend wrote CPU array";
    }
}

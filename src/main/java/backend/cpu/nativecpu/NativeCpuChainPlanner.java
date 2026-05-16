package backend.cpu.nativecpu;

import backend.ComputeBackend;
import backend.cpu.kernels.CpuNodeExecutionPlan;
import config.runtime.CpuStorageProfile;
import config.runtime.RuntimeConfig;
import graph.execution.CompiledNodeExecutionMetadata;
import graph.execution.PreparedNodeExecution;
import operations.Operation;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

/**
 * Annotates prepare-time native CPU routes with chain-level decisions.
 */
public final class NativeCpuChainPlanner {
    private NativeCpuChainPlanner() {
    }

    public static List<PreparedNodeExecution> annotate(List<PreparedNodeExecution> steps, RuntimeConfig runtimeConfig) {
        List<PreparedNodeExecution> safeSteps = steps == null ? List.of() : steps;
        CpuStorageProfile profile = runtimeConfig == null || runtimeConfig.cpuStorageProfile() == null
                ? CpuStorageProfile.CPU_ARRAY
                : runtimeConfig.cpuStorageProfile();
        if (profile == CpuStorageProfile.CPU_ARRAY || safeSteps.isEmpty()) {
            return List.copyOf(safeSteps);
        }
        return switch (profile) {
            case CPU_NATIVE -> annotateRequiredNative(safeSteps);
            case AUTO -> annotateAuto(safeSteps);
            case CPU_ARRAY -> List.copyOf(safeSteps);
        };
    }

    private static List<PreparedNodeExecution> annotateRequiredNative(List<PreparedNodeExecution> steps) {
        ArrayList<PreparedNodeExecution> out = new ArrayList<>(steps.size());
        int nextSegmentId = 0;
        int currentSegmentId = -1;
        for (PreparedNodeExecution step : steps) {
            PreparedNativeCpuPlan plan = nativePlan(step);
            if (isChainPreservingNative(plan)) {
                if (currentSegmentId < 0) {
                    currentSegmentId = nextSegmentId++;
                }
                out.add(withNativePlan(step, plan.withChain(
                        currentSegmentId,
                        NativeCpuChainDecision.REQUIRED_NATIVE,
                        "cpu-native-required"
                )));
                continue;
            }
            currentSegmentId = -1;
            if (plan != null && plan.route() == PreparedNativeCpuRoute.FALLBACK_ONLY) {
                out.add(withNativePlan(step, plan.withChain(
                        -1,
                        NativeCpuChainDecision.UNSUPPORTED_OP,
                        nonBlank(plan.fallbackReason(), "native-kernel-unsupported:" + opLabel(step))
                )));
            } else {
                out.add(step);
            }
        }
        return List.copyOf(out);
    }

    private static List<PreparedNodeExecution> annotateAuto(List<PreparedNodeExecution> steps) {
        ArrayList<PreparedNodeExecution> out = new ArrayList<>(steps.size());
        HashSet<Integer> activeNativeOutputs = new HashSet<>();
        int nextSegmentId = 0;
        int currentSegmentId = -1;
        for (PreparedNodeExecution step : steps) {
            PreparedNativeCpuPlan plan = nativePlan(step);
            NativeCpuCoverageEntry coverage = coverage(step, plan);
            if (coverage == null || step.metadata().backend() != ComputeBackend.CPU || step.metadata().cpuPlan() == null) {
                activeNativeOutputs.clear();
                currentSegmentId = -1;
                out.add(step);
                continue;
            }
            if (!coverage.nativeSupported()) {
                activeNativeOutputs.clear();
                currentSegmentId = -1;
                out.add(withNativePlan(step, autoPlan(step, plan, coverage).withChain(
                        -1,
                        NativeCpuChainDecision.UNSUPPORTED_OP,
                        nonBlank(coverage.fallbackReason(), "native-kernel-unsupported:" + opLabel(step))
                )));
                continue;
            }
            if (coverage.status() == NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW) {
                activeNativeOutputs.clear();
                currentSegmentId = -1;
                out.add(withNativePlan(step, autoPlan(step, plan, coverage).withChain(
                        -1,
                        NativeCpuChainDecision.AUTO_REJECTED_SLOW_OP,
                        "auto-rejected-slow-op:" + opLabel(step)
                )));
                continue;
            }
            if (!autoFastEligible(step, coverage)) {
                activeNativeOutputs.clear();
                currentSegmentId = -1;
                out.add(withNativePlan(step, autoPlan(step, plan, coverage).withChain(
                        -1,
                        NativeCpuChainDecision.MATERIALIZATION_BOUNDARY,
                        "auto-materialization-boundary:" + opLabel(step)
                )));
                continue;
            }

            PreparedNativeCpuPlan nativePlan = autoNativePlan(step, plan, coverage);
            if (nativePlan == null) {
                activeNativeOutputs.clear();
                currentSegmentId = -1;
                out.add(step);
                continue;
            }
            boolean startsProvider = coverage.status() == NativeCpuKernelPerformanceStatus.LIBRARY_PROVIDER;
            boolean joinsCurrent = hasInputInActiveSegment(step, activeNativeOutputs);
            if (coverage.status() == NativeCpuKernelPerformanceStatus.VIEW_ONLY && !joinsCurrent) {
                activeNativeOutputs.clear();
                currentSegmentId = -1;
                out.add(withNativePlan(step, autoPlan(step, plan, coverage).withChain(
                        -1,
                        NativeCpuChainDecision.MATERIALIZATION_BOUNDARY,
                        "auto-materialization-boundary:" + opLabel(step)
                )));
                continue;
            }
            if (!startsProvider && !joinsCurrent && currentSegmentId >= 0) {
                activeNativeOutputs.clear();
                currentSegmentId = -1;
            }
            if (currentSegmentId < 0) {
                currentSegmentId = nextSegmentId++;
            }
            activeNativeOutputs.add(step.compiledNode().id());
            out.add(withNativePlan(step, nativePlan.withChain(
                    currentSegmentId,
                    NativeCpuChainDecision.AUTO_FAST_NATIVE,
                    "auto-fast-native"
            )));
        }
        return List.copyOf(out);
    }

    private static boolean autoFastEligible(PreparedNodeExecution step, NativeCpuCoverageEntry coverage) {
        if (coverage.status() == NativeCpuKernelPerformanceStatus.LIBRARY_PROVIDER) {
            return step.metadata().cpuPlan().matMulExecutable() != null
                    && step.metadata().cpuPlan().matMulExecutable().acceptsNativeInputs();
        }
        return coverage.status() == NativeCpuKernelPerformanceStatus.VIEW_ONLY
                || coverage.status() == NativeCpuKernelPerformanceStatus.NATIVE_FAST;
    }

    private static PreparedNativeCpuPlan autoNativePlan(
            PreparedNodeExecution step,
            PreparedNativeCpuPlan plan,
            NativeCpuCoverageEntry coverage
    ) {
        PreparedNativeCpuPlan base = autoPlan(step, plan, coverage);
        if (coverage.status() == NativeCpuKernelPerformanceStatus.VIEW_ONLY) {
            return base.withRoute(PreparedNativeCpuRoute.VIEW_ALIAS, PreparedNativeCpuInputPolicy.ALL_NATIVE, coverage, "");
        }
        if (coverage.status() == NativeCpuKernelPerformanceStatus.LIBRARY_PROVIDER
                || coverage.status() == NativeCpuKernelPerformanceStatus.NATIVE_FAST) {
            return base.withRoute(PreparedNativeCpuRoute.NATIVE_EXECUTABLE, PreparedNativeCpuInputPolicy.ALL_NATIVE, coverage, "");
        }
        return null;
    }

    private static PreparedNativeCpuPlan autoPlan(
            PreparedNodeExecution step,
            PreparedNativeCpuPlan plan,
            NativeCpuCoverageEntry coverage
    ) {
        if (plan != null) {
            return plan;
        }
        return PreparedNativeCpuPlan.none(CpuStorageProfile.AUTO, "cpu-storage-profile-not-native:auto")
                .withRoute(PreparedNativeCpuRoute.NONE, PreparedNativeCpuInputPolicy.ALL_CPU, coverage, "cpu-storage-profile-not-native:auto");
    }

    private static boolean isChainPreservingNative(PreparedNativeCpuPlan plan) {
        return plan != null
                && plan.allowsNativeInputs()
                && plan.coverageEntry() != null
                && plan.coverageEntry().preservesNativeStorage();
    }

    private static PreparedNativeCpuPlan nativePlan(PreparedNodeExecution step) {
        if (step == null || step.metadata().cpuPlan() == null) {
            return null;
        }
        return step.metadata().cpuPlan().nativeCpuPlan();
    }

    private static NativeCpuCoverageEntry coverage(PreparedNodeExecution step, PreparedNativeCpuPlan plan) {
        if (plan != null && plan.coverageEntry() != null) {
            return plan.coverageEntry();
        }
        Operation op = step == null ? null : step.executionOperation();
        Operation.OpType opType = op == null ? Operation.OpType.UNKNOWN : op.opType();
        return NativeCpuCoverageMatrix.entryFor(opType, step == null ? null : step.compiledNode().dataType());
    }

    private static PreparedNodeExecution withNativePlan(PreparedNodeExecution step, PreparedNativeCpuPlan nativePlan) {
        CpuNodeExecutionPlan cpuPlan = step.metadata().cpuPlan().withNativeCpuPlan(nativePlan);
        CompiledNodeExecutionMetadata metadata = step.metadata();
        return new PreparedNodeExecution(
                step.compiledNode(),
                new CompiledNodeExecutionMetadata(
                        metadata.backend(),
                        metadata.cpuKernel(),
                        cpuPlan,
                        metadata.fusedExecutable(),
                        metadata.cpuWorkspace(),
                        metadata.acceleratorExecutable(),
                        metadata.executionOperation(),
                        metadata.executionInputNodeIds(),
                        metadata.partitionRole()
                )
        );
    }

    private static boolean hasInputInActiveSegment(PreparedNodeExecution step, HashSet<Integer> activeNativeOutputs) {
        for (int inputId : inputIds(step)) {
            if (activeNativeOutputs.contains(inputId)) {
                return true;
            }
        }
        return false;
    }

    private static List<Integer> inputIds(PreparedNodeExecution step) {
        return step.metadata().executionInputNodeIds().isEmpty()
                ? step.compiledNode().inputIds()
                : step.metadata().executionInputNodeIds();
    }

    private static String opLabel(PreparedNodeExecution step) {
        Operation op = step == null ? null : step.executionOperation();
        return op == null ? "unknown" : op.opType().name().toLowerCase();
    }

    private static String nonBlank(String value, String fallback) {
        return value == null || value.isBlank() ? fallback : value;
    }
}

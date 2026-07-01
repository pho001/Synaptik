package backend.cpu1;

import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuKernelRegistry;
import operations.Operation.OpType;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;

import static java.util.Map.entry;

/**
 * Central cpu1 parity inventory against old CPU direct kernels.
 */
public final class Cpu1CoverageReport {
    private static final Map<OpType, String> CPU1_PREPARED_FAMILY_ROUTES = map(
            entry(OpType.ADD, "elementwise"),
            entry(OpType.SUB, "elementwise"),
            entry(OpType.MUL, "elementwise"),
            entry(OpType.DIV, "elementwise"),
            entry(OpType.MIN, "elementwise"),
            entry(OpType.MAX, "elementwise"),
            entry(OpType.POW_TENSOR, "elementwise"),
            entry(OpType.GT, "elementwise"),
            entry(OpType.GE, "elementwise"),
            entry(OpType.LT, "elementwise"),
            entry(OpType.LE, "elementwise"),
            entry(OpType.EQ, "elementwise"),
            entry(OpType.NE, "elementwise"),
            entry(OpType.LOGICAL_AND, "elementwise"),
            entry(OpType.LOGICAL_OR, "elementwise"),
            entry(OpType.LOGICAL_NOT, "elementwise"),
            entry(OpType.WHERE, "elementwise"),
            entry(OpType.NEG, "elementwise"),
            entry(OpType.INV, "elementwise"),
            entry(OpType.LOG, "elementwise"),
            entry(OpType.EXP, "elementwise"),
            entry(OpType.FAST_EXP, "elementwise"),
            entry(OpType.ERF, "elementwise"),
            entry(OpType.TANH, "elementwise"),
            entry(OpType.FAST_TANH, "elementwise"),
            entry(OpType.POW, "elementwise"),
            entry(OpType.SQRT, "elementwise"),
            entry(OpType.ABS, "elementwise"),
            entry(OpType.FLOOR, "elementwise"),
            entry(OpType.CEIL, "elementwise"),
            entry(OpType.SIGN, "elementwise"),
            entry(OpType.MUL_SCALAR, "elementwise"),
            entry(OpType.RELU, "elementwise"),
            entry(OpType.CLAMP_MIN, "elementwise"),
            entry(OpType.CLAMP_MAX, "elementwise"),
            entry(OpType.SIGMOID, "elementwise"),
            entry(OpType.GATHER, "index"),
            entry(OpType.GATHER_AXIS, "index"),
            entry(OpType.GATHER_ND, "index"),
            entry(OpType.TAKE_ALONG_AXIS, "index"),
            entry(OpType.SCATTER_ADD, "index"),
            entry(OpType.SCATTER_AXIS_ADD, "index"),
            entry(OpType.SCATTER_ELEMENTS, "index"),
            entry(OpType.SCATTER_ND, "index"),
            entry(OpType.SUM, "reduction"),
            entry(OpType.MEAN, "reduction"),
            entry(OpType.REDUCE_MIN, "reduction"),
            entry(OpType.REDUCE_MAX, "reduction"),
            entry(OpType.REDUCE_PROD, "reduction"),
            entry(OpType.REDUCE_ALL, "reduction"),
            entry(OpType.REDUCE_ANY, "reduction"),
            entry(OpType.ARGMAX, "reduction"),
            entry(OpType.CUMSUM, "reduction"),
            entry(OpType.SOFTMAX, "reduction"),
            entry(OpType.LOG_SOFTMAX, "reduction"),
            entry(OpType.MATMUL, "matmul"),
            entry(OpType.LINEAR, "matmul"),
            entry(OpType.SCALED_DOT_PRODUCT_ATTENTION, "attention"),
            entry(OpType.SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS, "attention"),
            entry(OpType.NOOP, "layout"),
            entry(OpType.RESHAPE, "layout"),
            entry(OpType.EXPAND, "layout"),
            entry(OpType.SELECT, "layout"),
            entry(OpType.SLICE, "layout"),
            entry(OpType.PERMUTE, "layout"),
            entry(OpType.EXPAND_DIMS, "layout"),
            entry(OpType.SQUEEZE, "layout"),
            entry(OpType.CONTIGUOUS, "layout"),
            entry(OpType.CONCAT, "layout"),
            entry(OpType.PAD, "layout"),
            entry(OpType.TILE, "layout"),
            entry(OpType.UNFOLD_AXIS, "layout"),
            entry(OpType.UNFOLD2D, "layout"),
            entry(OpType.FOLD2D, "layout"),
            entry(OpType.SLICE_BACKWARD, "layout"),
            entry(OpType.CAST, "dtype"),
            entry(OpType.NLL_LOSS, "loss"),
            entry(OpType.CROSS_ENTROPY_LOSS, "loss"),
            entry(OpType.CROSS_ENTROPY_LOSS_INDICES, "loss"),
            entry(OpType.LAYER_NORM, "normalization"),
            entry(OpType.RMS_NORM, "normalization"),
            entry(OpType.MAX_POOL2D, "pool2d"),
            entry(OpType.AVG_POOL2D, "pool2d"),
            entry(OpType.CONV2D, "conv2d")
    );

    private static final Map<OpType, String> ALLOWED_MISSING_OR_DEFERRED_OLD_CPU_DIRECT_OPS = map();

    private static final Map<OpType, String> INTENTIONALLY_GRAPH_LOWERED_OR_NOT_DIRECT = map(
            entry(OpType.FUSED, "old CPU fused kernel is represented in cpu1 by lowered fused elementwise prepared units")
    );

    private static final Map<OpType, String> LEGACY_OR_SPECIAL_WITHOUT_OLD_CPU_DIRECT_KERNEL = map(
            entry(OpType.MIN_GRAD, "legacy backward op; old CPU registry intentionally has no direct kernel"),
            entry(OpType.MAX_GRAD, "legacy backward op; old CPU registry intentionally has no direct kernel"),
            entry(OpType.REDUCE_MIN_GRAD, "legacy backward op; old CPU registry intentionally has no direct kernel"),
            entry(OpType.REDUCE_MAX_GRAD, "legacy backward op; old CPU registry intentionally has no direct kernel"),
            entry(OpType.SOFTMAX_GRAD, "legacy backward op; old CPU registry intentionally has no direct kernel"),
            entry(OpType.LOG_SOFTMAX_GRAD, "legacy backward op; old CPU registry intentionally has no direct kernel"),
            entry(OpType.GATHER_GRAD, "legacy backward op; old CPU registry intentionally has no direct kernel"),
            entry(OpType.GATHER_AXIS_GRAD, "legacy backward op; old CPU registry intentionally has no direct kernel"),
            entry(OpType.GATHER_ND_GRAD, "legacy backward op; old CPU registry intentionally has no direct kernel"),
            entry(OpType.TAKE_ALONG_AXIS_GRAD, "legacy backward op; old CPU registry intentionally has no direct kernel"),
            entry(OpType.CROSS_ENTROPY_LOSS_INDICES_GRAD, "legacy backward op; old CPU registry intentionally has no direct kernel"),
            entry(OpType.SCALED_DOT_PRODUCT_ATTENTION_BACKWARD, "legacy backward op; old CPU registry intentionally has no direct kernel"),
            entry(OpType.CONST_SCALAR, "internal fused-plan scalar, not a standalone old CPU kernel"),
            entry(OpType.UNKNOWN, "invalid sentinel, no parity expectation")
    );

    private final Map<OpType, String> oldCpuDirectKernelClasses;
    private final Map<OpType, String> cpu1PreparedFamilyRoutes;
    private final Map<OpType, String> allowedMissingOrDeferredOps;
    private final Map<OpType, String> intentionallyGraphLoweredOrNotDirectOps;
    private final Map<OpType, String> legacyOrSpecialWithoutOldCpuDirectKernelOps;
    private final List<OpType> missingRequiredOps;
    private final List<OpType> unclassifiedNonOldCpuDirectOps;

    private Cpu1CoverageReport(
            Map<OpType, String> oldCpuDirectKernelClasses,
            Map<OpType, String> cpu1PreparedFamilyRoutes,
            Map<OpType, String> allowedMissingOrDeferredOps,
            Map<OpType, String> intentionallyGraphLoweredOrNotDirectOps,
            Map<OpType, String> legacyOrSpecialWithoutOldCpuDirectKernelOps,
            Collection<OpType> missingRequiredOps,
            Collection<OpType> unclassifiedNonOldCpuDirectOps
    ) {
        this.oldCpuDirectKernelClasses = copyMap(oldCpuDirectKernelClasses);
        this.cpu1PreparedFamilyRoutes = copyMap(cpu1PreparedFamilyRoutes);
        this.allowedMissingOrDeferredOps = copyMap(allowedMissingOrDeferredOps);
        this.intentionallyGraphLoweredOrNotDirectOps = copyMap(intentionallyGraphLoweredOrNotDirectOps);
        this.legacyOrSpecialWithoutOldCpuDirectKernelOps = copyMap(legacyOrSpecialWithoutOldCpuDirectKernelOps);
        this.missingRequiredOps = sortedList(missingRequiredOps);
        this.unclassifiedNonOldCpuDirectOps = sortedList(unclassifiedNonOldCpuDirectOps);
    }

    public static Cpu1CoverageReport current() {
        Map<OpType, String> oldCpuDirectKernelClasses = loadOldCpuDirectKernelClasses();
        EnumSet<OpType> oldCpuDirectOps = copyOf(oldCpuDirectKernelClasses.keySet());

        EnumSet<OpType> missingRequiredOps = copyOf(oldCpuDirectOps);
        missingRequiredOps.removeAll(CPU1_PREPARED_FAMILY_ROUTES.keySet());
        missingRequiredOps.removeAll(ALLOWED_MISSING_OR_DEFERRED_OLD_CPU_DIRECT_OPS.keySet());
        missingRequiredOps.removeAll(INTENTIONALLY_GRAPH_LOWERED_OR_NOT_DIRECT.keySet());

        EnumSet<OpType> unclassifiedNonOldCpuDirectOps = EnumSet.allOf(OpType.class);
        unclassifiedNonOldCpuDirectOps.removeAll(oldCpuDirectOps);
        unclassifiedNonOldCpuDirectOps.removeAll(LEGACY_OR_SPECIAL_WITHOUT_OLD_CPU_DIRECT_KERNEL.keySet());

        return new Cpu1CoverageReport(
                oldCpuDirectKernelClasses,
                CPU1_PREPARED_FAMILY_ROUTES,
                ALLOWED_MISSING_OR_DEFERRED_OLD_CPU_DIRECT_OPS,
                INTENTIONALLY_GRAPH_LOWERED_OR_NOT_DIRECT,
                LEGACY_OR_SPECIAL_WITHOUT_OLD_CPU_DIRECT_KERNEL,
                missingRequiredOps,
                unclassifiedNonOldCpuDirectOps
        );
    }

    public Map<OpType, String> oldCpuDirectKernelClasses() {
        return oldCpuDirectKernelClasses;
    }

    public Map<OpType, String> cpu1PreparedFamilyRoutes() {
        return cpu1PreparedFamilyRoutes;
    }

    public Map<OpType, String> allowedMissingOrDeferredOps() {
        return allowedMissingOrDeferredOps;
    }

    public Map<OpType, String> intentionallyGraphLoweredOrNotDirectOps() {
        return intentionallyGraphLoweredOrNotDirectOps;
    }

    public Map<OpType, String> legacyOrSpecialWithoutOldCpuDirectKernelOps() {
        return legacyOrSpecialWithoutOldCpuDirectKernelOps;
    }

    public List<OpType> missingRequiredOps() {
        return missingRequiredOps;
    }

    public List<OpType> unclassifiedNonOldCpuDirectOps() {
        return unclassifiedNonOldCpuDirectOps;
    }

    public String gateReport() {
        return String.join("\n",
                "cpu1 coverage gate:",
                "  oldCpuDirectOps=" + oldCpuDirectKernelClasses.size() + " " + format(oldCpuDirectKernelClasses),
                "  cpu1PreparedFamilyRoutes=" + cpu1PreparedFamilyRoutes.size() + " " + format(cpu1PreparedFamilyRoutes),
                "  missingRequiredOps=" + format(missingRequiredOps),
                "  allowedMissingOrDeferredOps=" + format(allowedMissingOrDeferredOps),
                "  intentionallyGraphLoweredOrNotDirectOps=" + format(intentionallyGraphLoweredOrNotDirectOps),
                "  legacyOrSpecialWithoutOldCpuDirectKernelOps=" + format(legacyOrSpecialWithoutOldCpuDirectKernelOps),
                "  unclassifiedNonOldCpuDirectOps=" + format(unclassifiedNonOldCpuDirectOps)
        );
    }

    private static Map<OpType, String> loadOldCpuDirectKernelClasses() {
        EnumMap<OpType, String> direct = new EnumMap<>(OpType.class);
        for (OpType opType : OpType.values()) {
            CpuKernel kernel = oldCpuDirectKernel(opType);
            if (kernel != null) {
                direct.put(opType, oldCpuKernelName(kernel));
            }
        }
        return Map.copyOf(direct);
    }

    private static CpuKernel oldCpuDirectKernel(OpType opType) {
        try {
            return CpuKernelRegistry.resolve(opType);
        } catch (IllegalStateException ex) {
            return null;
        }
    }

    private static String oldCpuKernelName(CpuKernel kernel) {
        String name = kernel.getClass().getName();
        String prefix = "backend.cpu.kernels.";
        return name.startsWith(prefix) ? name.substring(prefix.length()) : name;
    }

    private static EnumSet<OpType> copyOf(Collection<OpType> opTypes) {
        EnumSet<OpType> copy = EnumSet.noneOf(OpType.class);
        copy.addAll(opTypes);
        return copy;
    }

    private static List<OpType> sortedList(Collection<OpType> opTypes) {
        return opTypes.stream()
                .sorted(Comparator.comparingInt(Enum::ordinal))
                .toList();
    }

    private static Map<OpType, String> copyMap(Map<OpType, String> source) {
        EnumMap<OpType, String> copy = new EnumMap<>(OpType.class);
        copy.putAll(source);
        return Map.copyOf(copy);
    }

    private static String format(Collection<OpType> opTypes) {
        return sortedList(opTypes).toString();
    }

    private static String format(Map<OpType, String> opTypes) {
        List<String> entries = new ArrayList<>();
        opTypes.entrySet().stream()
                .sorted(Map.Entry.comparingByKey(Comparator.comparingInt(Enum::ordinal)))
                .forEach(entry -> entries.add(entry.getKey() + "=" + entry.getValue()));
        return entries.toString();
    }

    @SafeVarargs
    private static Map<OpType, String> map(Map.Entry<OpType, String>... entries) {
        EnumMap<OpType, String> map = new EnumMap<>(OpType.class);
        for (Map.Entry<OpType, String> entry : entries) {
            String previous = map.put(entry.getKey(), entry.getValue());
            if (previous != null) {
                throw new IllegalArgumentException("Duplicate cpu1 coverage entry for " + entry.getKey());
            }
        }
        return Map.copyOf(map);
    }
}

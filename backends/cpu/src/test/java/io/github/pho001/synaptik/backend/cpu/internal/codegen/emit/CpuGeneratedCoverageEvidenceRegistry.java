package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.*;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.pooling.Pool2dKind;
import io.github.pho001.synaptik.model.operation.index.SelectKind;
import io.github.pho001.synaptik.model.operation.layout.ContiguousKind;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.SliceKind;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.security.MessageDigest;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Test-side, execution-produced facts for generated-coverage owners.
 *
 * <p>This deliberately records values after the real provider, preparer, and generator have run;
 * a resource label alone is not evidence and cannot manufacture an owner record. The checked
 * inventory is frozen support accounting, while this registry is a process-local live witness
 * used to prove that an inventory owner still resolves to its exact descriptor, class bytes, and
 * preparation facts. Duplicate owners are accepted only when every recorded fact is identical;
 * conflicting observations, absent bytes, and a request to relabel a generated unit as metadata
 * fail closed.</p>
 */
final class CpuGeneratedCoverageEvidenceRegistry {
    private static final Map<String, Record> RECORDS = new LinkedHashMap<>();
    /*
     * Kept only for the lifetime of a checkpoint execution.  The inventory deliberately stores
     * hashes rather than class files; the structural-oracle catalog needs the execution-produced
     * bytes while it validates one representative of each material category.
     */
    private static final Map<String, byte[]> CLASS_BYTES = new LinkedHashMap<>();

    private CpuGeneratedCoverageEvidenceRegistry() { }

    /**
     * Exact provider-query boundary fixture, retained as a rejection rather than a generated row.
     *
     * @param id stable fixture identifier; never blank
     * @param operationForm inspected operation form; never blank
     * @param operation exact occurrence queried from the provider; never {@code null}
     * @param inputs ordered input descriptors owned by the fixture; never {@code null}
     * @param outputs ordered output descriptors owned by the fixture; never {@code null}
     * @param reason explicit unsupported boundary; never blank
     */
    record ProviderRejectedFixture(String id, String operationForm, Operation operation,
            List<io.github.pho001.synaptik.model.tensor.TensorDescriptor> inputs,
            List<io.github.pho001.synaptik.model.tensor.TensorDescriptor> outputs, String reason) { }

    /**
     * Exact preparation boundary after provider admission, never evidence of a generated unit.
     *
     * @param id stable fixture identifier; never blank
     * @param operationForm inspected operation form; never blank
     * @param context real prepare context that must be rejected; never {@code null}
     * @param reason explicit preparation boundary; never blank
     */
    record PreparerRejectedFixture(String id, String operationForm,
            PrepareContext<CpuPartitionAnalysisInputs> context, String reason) { }

    static synchronized Record generated(String occurrenceId,
            PrepareContext<CpuPartitionAnalysisInputs> context, CpuPartitionPreparationPlan plan) {
        var node = context.nodes().getFirst();
        var inputs = descriptors(context, node.inputs());
        var outputs = descriptors(context, node.outputs());
        boolean supported = new CpuCapabilityProvider().supports(
                new OperationCapabilityQuery(node.operation(), inputs, outputs));
        var unit = plan.units().getFirst();
        var route = unit.portablePlan();
        byte[] bytes = new CpuClassFileKernelGenerator().generateClassBytes(
                route.specialization(), route.kernelIr());
        String inventoryForm = classify(route.portableKernelIr(), context, plan);
        var record = new Record(occurrenceId, inventoryForm,
                route.portableKernelIr().getClass().getSimpleName(),
                descriptorRoles(context, node.inputs(), node.outputs()), boundaryIds(unit.boundaryValues()),
                virtualIds(context, unit.boundaryValues()), plan.units().size(),
                plan.affineAddressPairs().length / 2,
                unit.portablePlan().specialization().carrierPattern().stream().map(Enum::name).toList().toString(),
                unit.accessBindings().stream().map(binding -> binding.plan().regime().name()).toList().toString(),
                executionRequest(context), plan.executionStrategy().toString(),
                plan.representationDecisions().stream().anyMatch(decision -> decision instanceof
                        io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRepresentationDecision.Variant variant
                        && !variant.identity().materializations().isEmpty()),
                plan.materializations().size(), supported, true,
                sha256(route.kernelIr().structuralKey().getBytes(StandardCharsets.UTF_8)),
                route.specialization().entryType().descriptorString(), sha256(bytes),
                sha256(route.specialization().structuralKey().getBytes(StandardCharsets.UTF_8)),
                "GENERATED", "NONE", ordinarySelection(plan), fallbackDisposition(plan));
        Record previous = RECORDS.putIfAbsent(occurrenceId, record);
        if (previous != null && !previous.equals(record)) {
            throw new AssertionError("evidence occurrence id is not deterministic: " + occurrenceId);
        }
        CLASS_BYTES.putIfAbsent(occurrenceId, bytes.clone());
        return record;
    }

    /**
     * Records an affine occurrence that the real preparer proves owns no generated unit.
     *
     * <p>This path intentionally never constructs a class-file generator. A prepared unit is
     * evidence of a standalone generated realization and must be recorded through
     * {@link #generated(String, PrepareContext, CpuPartitionPreparationPlan)} instead.</p>
     */
    static synchronized Record metadataOnly(String occurrenceId,
            PrepareContext<CpuPartitionAnalysisInputs> context, CpuPartitionPreparationPlan plan) {
        String form = affineForm(context).orElseThrow(() -> new AssertionError(
                "metadata-only evidence requires one exact affine occurrence"));
        if (!plan.units().isEmpty()) throw new AssertionError(
                "affine occurrence owns a generated unit and cannot be relabeled metadata-only: " + form);
        var node = context.nodes().getFirst();
        var inputs = descriptors(context, node.inputs());
        var outputs = descriptors(context, node.outputs());
        boolean supported = new CpuCapabilityProvider().supports(
                new OperationCapabilityQuery(node.operation(), inputs, outputs));
        var record = new Record(occurrenceId, form, "NO_GENERATED_UNIT",
                descriptorRoles(context, node.inputs(), node.outputs()), "[]", virtualIds(context, List.of()),
                0, 0, "[]", "[]",
                context.backendInputs().portableExecution().toString(), plan.executionStrategy().toString(),
                false, plan.materializations().size(), supported, true, "NO_GENERATED_UNIT", "",
                "", "", "METADATA_NO_GENERATED_HOT_LOOP",
                "the real preparer selected no execution unit, so no standalone generated class is owned",
                "NO_ORDINARY_GENERATED_SELECTION", "METADATA_NO_GENERATED_HOT_LOOP");
        Record previous = RECORDS.putIfAbsent(occurrenceId, record);
        if (previous != null && !previous.equals(record)) {
            throw new AssertionError("evidence occurrence id is not deterministic: " + occurrenceId);
        }
        return record;
    }

    /**
     * Records a real provider rejection.  No preparer or generator is reached after a provider
     * says no: the N/A facts are deliberate evidence of that boundary, not missing observations.
     */
    static synchronized Record rejectedByProvider(String occurrenceId, String operationForm,
            Operation operation, List<io.github.pho001.synaptik.model.tensor.TensorDescriptor> inputs,
            List<io.github.pho001.synaptik.model.tensor.TensorDescriptor> outputs, String reason) {
        boolean supported = new CpuCapabilityProvider().supports(new OperationCapabilityQuery(operation, inputs, outputs));
        if (supported) throw new AssertionError("provider admitted rejected fixture: " + occurrenceId);
        var record = new Record(occurrenceId, operationForm, "NO_PREPARED_IR",
                queryDescriptorRoles(inputs, outputs), "N/A_PROVIDER_REJECTED", "N/A_PROVIDER_REJECTED",
                0, 0, "N/A_PROVIDER_REJECTED", queryLayouts(inputs, outputs),
                "N/A_PROVIDER_REJECTED", "N/A_PROVIDER_REJECTED", false, 0, false, false,
                "N/A_PROVIDER_REJECTED", "N/A_PROVIDER_REJECTED", "N/A_PROVIDER_REJECTED",
                "N/A_PROVIDER_REJECTED", "REJECTED_PROVIDER", reason,
                "PROVIDER_REJECTION", "PROVIDER_REJECTION:" + reason);
        return put(record);
    }

    /**
     * Records a provider-supported composition rejected by the real partition preparer.  The
     * method intentionally has no generator reference: no unit was admitted for generation.
     */
    static synchronized Record rejectedByPreparer(String occurrenceId, String operationForm,
            PrepareContext<CpuPartitionAnalysisInputs> context, String reason) {
        var mapped = context.nodes().get(context.nodes().size() == 3 ? 1 : 2);
        boolean supported = new CpuCapabilityProvider().supports(new OperationCapabilityQuery(mapped.operation(),
                descriptors(context, mapped.inputs()), descriptors(context, mapped.outputs())));
        if (!supported) throw new AssertionError("provider did not admit preparer-rejected fixture: " + occurrenceId);
        IllegalArgumentException failure;
        try {
            new io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer().analyze(context);
            throw new AssertionError("preparer admitted rejected fixture: " + occurrenceId);
        } catch (IllegalArgumentException expected) {
            failure = expected;
        }
        String exactReason = reason + ":" + failure.getMessage() + ":"
                + (failure.getCause() == null ? "NO_CAUSE" : failure.getCause().getMessage());
        var record = new Record(occurrenceId, operationForm, "NO_PREPARED_IR",
                descriptorRoles(context, mapped.inputs(), mapped.outputs()), "N/A_PREPARER_REJECTED",
                virtualIds(context, List.of()), 0, 0, requestedCarriers(context), requestedLayouts(context),
                executionRequest(context), "N/A_PREPARER_REJECTED",
                context.backendInputs().materializationPolicy().enabled(), 0, true, false,
                "N/A_PREPARER_REJECTED", "N/A_PREPARER_REJECTED", "N/A_PREPARER_REJECTED",
                "N/A_PREPARER_REJECTED", "REJECTED_PREPARER", exactReason,
                "PREPARER_REJECTION", "PREPARER_REJECTION:" + exactReason);
        return put(record);
    }

    private static Record put(Record record) {
        Record previous = RECORDS.putIfAbsent(record.occurrenceId(), record);
        if (previous != null && !previous.equals(record)) {
            throw new AssertionError("evidence occurrence id is not deterministic: " + record.occurrenceId());
        }
        return record;
    }

    private static String ordinarySelection(CpuPartitionPreparationPlan plan) {
        var selection = plan.representationDecisions().stream()
                .filter(CpuRepresentationDecision.Selection.class::isInstance)
                .map(CpuRepresentationDecision.Selection.class::cast).reduce((left, right) -> right)
                .orElse(null);
        if (selection == null) return "SELECTED_GENERATED_DIRECT_NO_REPRESENTATION_DECISION";
        boolean candidates = plan.representationDecisions().stream()
                .filter(CpuRepresentationDecision.Variant.class::isInstance)
                .map(CpuRepresentationDecision.Variant.class::cast)
                .anyMatch(variant -> !variant.identity().materializations().isEmpty());
        if (!selection.selected().materializations().isEmpty())
            return "SELECTED_GENERATED_AFFINE_COPY_MATERIALIZED_BOUNDARY";
        return candidates ? "DIRECT_RETAINED_WHILE_MATERIALIZATION_CANDIDATES_EXIST"
                : "SELECTED_GENERATED_DIRECT";
    }

    private static String fallbackDisposition(CpuPartitionPreparationPlan plan) {
        var selection = plan.representationDecisions().stream()
                .filter(CpuRepresentationDecision.Selection.class::isInstance)
                .map(CpuRepresentationDecision.Selection.class::cast).reduce((left, right) -> right);
        if (selection.isEmpty()) return "NO_SAFE_FALLBACK_REQUIRED";
        var value = selection.orElseThrow();
        return value.selected().materializations().isEmpty()
                ? "SAFE_DIRECT_FALLBACK:" + value.reason().name()
                : "SELECTED_MATERIALIZED_BOUNDARY:" + value.reason().name();
    }

    static synchronized Record require(String occurrenceId) {
        Record record = RECORDS.get(occurrenceId);
        if (record == null) throw new AssertionError("no execution-produced evidence for owner " + occurrenceId);
        return record;
    }

    /**
     * Requires the full, execution-derived combination identity for an owner.
     *
     * <p>The identity deliberately contains every fact that makes a generated witness
     * materially different.  A form name is consequently not a join key: changing a type role,
     * carrier order, layout, requested or selected strategy, materialization result, prepared IR,
     * generated body, or outcome changes this value and fails the join.</p>
     */
    static synchronized Record requireExact(String occurrenceId, String evidenceKey) {
        Record record = require(occurrenceId);
        if (!record.evidenceKey().equals(evidenceKey)) {
            throw new AssertionError("execution evidence does not match owner " + occurrenceId);
        }
        return record;
    }

    /**
     * Captures the current immutable owner-to-fact view without exposing mutable registry state.
     *
     * @return an immutable snapshot of all recorded owners; never {@code null}
     */
    static synchronized Map<String, Record> snapshot() { return Map.copyOf(RECORDS); }

    /** Returns the generated bytes observed for this exact execution owner. */
    static synchronized byte[] classBytes(String occurrenceId) {
        byte[] bytes = CLASS_BYTES.get(occurrenceId);
        if (bytes == null) throw new AssertionError("no generated class bytes for owner " + occurrenceId);
        return bytes.clone();
    }

    /** Clears process-local observations so a checkpoint creates all of its own evidence. */
    static synchronized void clear() {
        RECORDS.clear();
        CLASS_BYTES.clear();
    }

    /**
     * Converts only actual prepared CPU IR into one inventory form.  In particular this never
     * consults an inventory row or an operation-name label: ordinary extrema are identified by
     * {@link CpuAggregateIr}, while pointwise extrema remain a {@link CpuKernelIr} opcode.
     */
    private static String classify(CpuPortableKernelIr ir, PrepareContext<?> context,
            CpuPartitionPreparationPlan plan) {
        var affine = affineForm(context);
        if (affine.isPresent()) {
            if (ir instanceof CpuAffineCopyIr copy && copy.mappingSteps().size() == 1
                    && copy.mappingSteps().getFirst().kind().name().equals(affine.orElseThrow())) {
                return affine.orElseThrow();
            }
            if (ir instanceof CpuKernelIr kernel && kernel.instructions().isEmpty()
                    && kernel.familyIdentity().startsWith("affine:")) {
                return affine.orElseThrow();
            }
            {
                throw unknown(ir, "exact affine occurrence did not lower to an instruction-free affine copy");
            }
        }
        if (ir instanceof CpuKernelIr kernel) {
            if (kernel.instructions().isEmpty()
                    && plan.units().getFirst().portablePlan().specialization().matmulIr().isPresent()) {
                return "MATMUL";
            }
            if (kernel.instructions().size() != 1) throw unknown(ir, "pointwise instruction count");
            return kernel.instructions().getFirst().opcode().name();
        }
        if (ir instanceof CpuDataMovementIr movement) return switch (movement.plan()) {
            case CpuDataMovementIr.PadPlan ignored -> "PAD";
            case CpuDataMovementIr.TilePlan ignored -> "TILE";
            case CpuDataMovementIr.ConcatPlan ignored -> "CONCAT";
            case CpuDataMovementIr.StackPlan ignored -> "STACK";
            case CpuDataMovementIr.UnfoldAxisPlan ignored -> "UNFOLD_AXIS";
            case CpuDataMovementIr.Unfold2dPlan ignored -> "UNFOLD2D";
            case CpuDataMovementIr.SliceUpdatePlan ignored -> "SLICE_UPDATE";
        };
        if (ir instanceof CpuIndexingIr indexing) return indexing.family().name();
        if (ir instanceof CpuScatterIr scatter) return scatter.family().name();
        if (ir instanceof CpuFoldIr fold) return fold.family().name();
        if (ir instanceof CpuOrderingIr ordering) return ordering.family().name();
        if (ir instanceof CpuRandomIr random) return random.family().name();
        if (ir instanceof CpuScanIr scan) return scan.kind().name();
        if (ir instanceof CpuAggregateIr aggregate) {
            if (aggregate.form() == CpuAggregateIr.Form.SUM_TO_SHAPE) return "SUM_TO_SHAPE";
            return switch (aggregate.kind()) {
                case MIN -> "AGGREGATE_MIN";
                case MAX -> "AGGREGATE_MAX";
                default -> aggregate.kind().name();
            };
        }
        if (ir instanceof CpuArgExtremaIr arg) return arg.kind().name();
        if (ir instanceof CpuMaskedReductionIr masked) return "MASKED_" + masked.kind().name();
        if (ir instanceof CpuAdvancedReductionIr advanced) return advanced.kind().name();
        if (ir instanceof CpuSoftmaxIr softmax) return softmax.kind().name();
        if (ir instanceof CpuTrailingNormalizationIr normalization) return normalization.form().name();
        if (ir instanceof CpuBatchNormInferenceIr) return "BATCH_NORM_INFERENCE";
        if (ir instanceof CpuBatchNormTrainingIr) return "BATCH_NORM_TRAINING";
        if (ir instanceof CpuConv2dIr) return conv1dTopology(context) ? "CONV1D_COMPOSITION" : "CONV2D";
        if (ir instanceof CpuConv3dIr) return "CONV3D";
        if (ir instanceof CpuPool2dIr pool) return pool1dTopology(context) ? "POOL1D_COMPOSITION"
                : pool.kind() == CpuPool2dIr.Kind.MAX ? "MAX_POOL2D" : "AVERAGE_POOL2D";
        if (ir instanceof CpuPool3dIr pool) return pool.kind() == CpuPool3dIr.Kind.MAX
                ? "MAX_POOL3D" : "AVERAGE_POOL3D";
        if (ir instanceof CpuAttentionIr) return "SCALED_DOT_PRODUCT_ATTENTION";
        if (ir instanceof CpuLossIr loss) return loss.kind().name();
        throw unknown(ir, "unmapped prepared IR");
    }

    private static java.util.Optional<String> affineForm(PrepareContext<?> context) {
        if (context.nodes().size() != 1) return java.util.Optional.empty();
        var node = context.nodes().getFirst();
        if (node.inputs().size() != 1 || node.outputs().size() != 1) return java.util.Optional.empty();
        Object kind = node.operation().kind();
        if (kind == ContiguousKind.CONTIGUOUS) return java.util.Optional.of("CONTIGUOUS");
        if (kind == ShapeTransformKind.RESHAPE) return java.util.Optional.of("RESHAPE");
        if (kind == ShapeTransformKind.EXPAND) return java.util.Optional.of("EXPAND");
        if (kind == AxisTransformKind.PERMUTE) return java.util.Optional.of("PERMUTE");
        if (kind == AxisTransformKind.EXPAND_DIMS) return java.util.Optional.of("EXPAND_DIMS");
        if (kind == AxisTransformKind.SQUEEZE) return java.util.Optional.of("SQUEEZE");
        if (kind == SelectKind.SELECT) return java.util.Optional.of("SELECT");
        if (kind == SliceKind.SLICE) return java.util.Optional.of("SLICE");
        return java.util.Optional.empty();
    }

    private static AssertionError unknown(CpuPortableKernelIr ir, String detail) {
        return new AssertionError("unknown or ambiguous prepared IR form: "
                + ir.getClass().getName() + " (" + detail + ')');
    }

    private static boolean conv1dTopology(PrepareContext<?> context) {
        if (context.nodes().size() != 4) return false;
        var first = context.nodes().get(0); var second = context.nodes().get(1);
        var conv = context.nodes().get(2); var squeeze = context.nodes().get(3);
        return expansionAt(first, 2) && expansionAt(second, 2)
                && conv.operation().kind() == Conv2dKind.CONV2D
                && conv.inputs().size() >= 2 && conv.inputs().size() <= 3
                && conv.inputs().subList(0, 2).containsAll(List.of(first.outputs().getFirst(), second.outputs().getFirst()))
                && squeezeAt(squeeze, 2, conv.outputs().getFirst());
    }

    private static boolean pool1dTopology(PrepareContext<?> context) {
        if (context.nodes().size() != 3) return false;
        var expand = context.nodes().get(0); var pool = context.nodes().get(1);
        var squeeze = context.nodes().get(2);
        return expansionAt(expand, 2) && (pool.operation().kind() == Pool2dKind.MAX_POOL2D
                || pool.operation().kind() == Pool2dKind.AVERAGE_POOL2D)
                && pool.inputs().equals(List.of(expand.outputs().getFirst()))
                && squeezeAt(squeeze, 2, pool.outputs().getFirst());
    }

    private static boolean expansionAt(io.github.pho001.synaptik.model.graph.CompiledNode node, int axis) {
        return node.operation().kind() == AxisTransformKind.EXPAND_DIMS
                && node.operation().attrs() instanceof AxisTransformAttrs attrs
                && attrs.axis() == axis && node.inputs().size() == 1 && node.outputs().size() == 1;
    }

    private static boolean squeezeAt(io.github.pho001.synaptik.model.graph.CompiledNode node, int axis,
            ValueId input) {
        return node.operation().kind() == AxisTransformKind.SQUEEZE
                && node.operation().attrs() instanceof AxisTransformAttrs attrs
                && attrs.axis() == axis && node.inputs().equals(List.of(input)) && node.outputs().size() == 1;
    }

    private static String descriptorRoles(PrepareContext<?> context, List<ValueId> inputs,
            List<ValueId> outputs) {
        return "inputs=" + typedIds(context, inputs) + ";outputs=" + typedIds(context, outputs);
    }

    private static String queryDescriptorRoles(List<io.github.pho001.synaptik.model.tensor.TensorDescriptor> inputs,
            List<io.github.pho001.synaptik.model.tensor.TensorDescriptor> outputs) {
        return "inputs=" + inputs.stream().map(CpuGeneratedCoverageEvidenceRegistry::queryDescriptor).toList()
                + ";outputs=" + outputs.stream().map(CpuGeneratedCoverageEvidenceRegistry::queryDescriptor).toList();
    }

    private static String queryLayouts(List<io.github.pho001.synaptik.model.tensor.TensorDescriptor> inputs,
            List<io.github.pho001.synaptik.model.tensor.TensorDescriptor> outputs) {
        return java.util.stream.Stream.concat(inputs.stream(), outputs.stream()).map(descriptor -> descriptor.layout()
                .map(layout -> layout.isContiguous() ? "CONTIGUOUS" : "NONCONTIGUOUS").orElse("NO_LAYOUT"))
                .toList().toString();
    }

    private static String queryDescriptor(io.github.pho001.synaptik.model.tensor.TensorDescriptor descriptor) {
        return descriptor.dataType() + ":" + descriptor.shape() + ":" + descriptor.layout()
                .map(layout -> layout.isContiguous() ? "CONTIGUOUS" : "NONCONTIGUOUS").orElse("NO_LAYOUT");
    }

    private static String requestedCarriers(PrepareContext<CpuPartitionAnalysisInputs> context) {
        return context.backendInputs().carrierPattern().stream().map(Enum::name).toList().toString();
    }

    private static String requestedLayouts(PrepareContext<?> context) {
        return context.values().stream().map(GraphValue::descriptor).map(descriptor -> descriptor.layout()
                .map(layout -> layout.isContiguous() ? "CONTIGUOUS" : "NONCONTIGUOUS").orElse("NO_LAYOUT"))
                .toList().toString();
    }

    private static String typedIds(PrepareContext<?> context, List<ValueId> ids) {
        return ids.stream().map(id -> id + ":" + descriptors(context, List.of(id)).getFirst().dataType())
                .toList().toString();
    }

    private static String executionRequest(PrepareContext<CpuPartitionAnalysisInputs> context) {
        var config = context.backendInputs().portableExecution();
        return config.computePreference() + ":configured=" + config.configuredMaximumParallelism()
                + ":available=" + config.availableParallelism() + ":minimum=" + config.minimumElementsPerWorker();
    }

    private static String boundaryIds(List<ValueId> ids) {
        return ids.stream().distinct().map(ValueId::toString).toList().toString();
    }

    private static String virtualIds(PrepareContext<?> context, List<ValueId> boundaries) {
        return context.nodes().stream().flatMap(node -> node.outputs().stream())
                .filter(id -> !boundaries.contains(id)).map(ValueId::toString).toList().toString();
    }

    private static List<io.github.pho001.synaptik.model.tensor.TensorDescriptor> descriptors(
            PrepareContext<?> context, List<ValueId> ids) {
        return ids.stream().map(id -> context.values().stream().filter(value -> value.id().equals(id))
                .findFirst().map(GraphValue::descriptor).orElseThrow()).toList();
    }

    private static String sha256(byte[] bytes) {
        try { return java.util.HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes)); }
        catch (java.security.NoSuchAlgorithmException impossible) { throw new AssertionError(impossible); }
    }

    record Record(String occurrenceId, String operationForm, String loweredForm, String descriptorRoles,
                  String boundaryIds, String virtualValues, int unitCount, int affineAddressPairCount,
                  String orderedCarrierTypes, String layoutAccessRegime, String requestedStrategy,
                  String selectedStrategy, boolean materializationCandidate,
                  int materializationsSelected, boolean providerSupported, boolean preparerAdmitted,
                  String preparedIrStructuralKey, String generatedEntryDescriptor, String classHash,
                  String normalizedBodyKey, String outcome, String reason, String ordinarySelection,
                  String fallbackDisposition) {
        /** Stable digest of the actual generated-combination facts, excluding its owner name. */
        String evidenceKey() {
            String facts = String.join("\u001f", operationForm, loweredForm, descriptorRoles, boundaryIds,
                    virtualValues, Integer.toString(unitCount), Integer.toString(affineAddressPairCount),
                    orderedCarrierTypes, layoutAccessRegime, requestedStrategy, selectedStrategy,
                    Boolean.toString(materializationCandidate), Integer.toString(materializationsSelected),
                    Boolean.toString(providerSupported), Boolean.toString(preparerAdmitted),
                    preparedIrStructuralKey, generatedEntryDescriptor, classHash, normalizedBodyKey, outcome, reason,
                    ordinarySelection, fallbackDisposition);
            return sha256(facts.getBytes(StandardCharsets.UTF_8));
        }

        Record mutate(String field) {
            return switch (field) {
                case "form" -> copy("MUTATED_FORM", descriptorRoles, orderedCarrierTypes, layoutAccessRegime,
                        requestedStrategy, selectedStrategy, materializationCandidate, preparedIrStructuralKey,
                        classHash, normalizedBodyKey, outcome, reason);
                case "type-or-alias-roles" -> copy(operationForm, "MUTATED_ROLES", orderedCarrierTypes,
                        layoutAccessRegime, requestedStrategy, selectedStrategy, materializationCandidate,
                        preparedIrStructuralKey, classHash, normalizedBodyKey, outcome, reason);
                case "carrier-order" -> copy(operationForm, descriptorRoles, "MUTATED_CARRIERS", layoutAccessRegime,
                        requestedStrategy, selectedStrategy, materializationCandidate, preparedIrStructuralKey,
                        classHash, normalizedBodyKey, outcome, reason);
                case "layout" -> copy(operationForm, descriptorRoles, orderedCarrierTypes, "MUTATED_LAYOUT",
                        requestedStrategy, selectedStrategy, materializationCandidate, preparedIrStructuralKey,
                        classHash, normalizedBodyKey, outcome, reason);
                case "requested-strategy" -> copy(operationForm, descriptorRoles, orderedCarrierTypes,
                        layoutAccessRegime, "MUTATED_REQUEST", selectedStrategy, materializationCandidate,
                        preparedIrStructuralKey, classHash, normalizedBodyKey, outcome, reason);
                case "selected-strategy" -> copy(operationForm, descriptorRoles, orderedCarrierTypes,
                        layoutAccessRegime, requestedStrategy, "MUTATED_SELECTED", materializationCandidate,
                        preparedIrStructuralKey, classHash, normalizedBodyKey, outcome, reason);
                case "materialization" -> copy(operationForm, descriptorRoles, orderedCarrierTypes,
                        layoutAccessRegime, requestedStrategy, selectedStrategy, !materializationCandidate,
                        preparedIrStructuralKey, classHash, normalizedBodyKey, outcome, reason);
                case "ir-family" -> copy(operationForm, descriptorRoles, orderedCarrierTypes, layoutAccessRegime,
                        requestedStrategy, selectedStrategy, materializationCandidate, "MUTATED_IR", classHash,
                        normalizedBodyKey, outcome, reason);
                case "hash-or-body" -> copy(operationForm, descriptorRoles, orderedCarrierTypes, layoutAccessRegime,
                        requestedStrategy, selectedStrategy, materializationCandidate, preparedIrStructuralKey,
                        "0".repeat(64), "MUTATED_BODY", outcome, reason);
                case "oracle-or-performance-disposition" -> copy(operationForm, descriptorRoles, orderedCarrierTypes,
                        layoutAccessRegime, requestedStrategy, selectedStrategy, materializationCandidate,
                        preparedIrStructuralKey, classHash, normalizedBodyKey, "MUTATED_DISPOSITION", reason);
                case "rejection-boundary-or-reason" -> copy(operationForm, descriptorRoles, orderedCarrierTypes,
                        layoutAccessRegime, requestedStrategy, selectedStrategy, materializationCandidate,
                        preparedIrStructuralKey, classHash, normalizedBodyKey, "MUTATED_REJECTION_BOUNDARY",
                        "MUTATED_REJECTION_REASON");
                case "ordinary-selection" -> new Record(occurrenceId, operationForm, loweredForm,
                        descriptorRoles, boundaryIds, virtualValues, unitCount, affineAddressPairCount,
                        orderedCarrierTypes, layoutAccessRegime, requestedStrategy, selectedStrategy,
                        materializationCandidate, materializationsSelected, providerSupported, preparerAdmitted,
                        preparedIrStructuralKey, generatedEntryDescriptor, classHash, normalizedBodyKey, outcome,
                        reason, "MUTATED_ORDINARY_SELECTION", fallbackDisposition);
                case "fallback-disposition" -> new Record(occurrenceId, operationForm, loweredForm,
                        descriptorRoles, boundaryIds, virtualValues, unitCount, affineAddressPairCount,
                        orderedCarrierTypes, layoutAccessRegime, requestedStrategy, selectedStrategy,
                        materializationCandidate, materializationsSelected, providerSupported, preparerAdmitted,
                        preparedIrStructuralKey, generatedEntryDescriptor, classHash, normalizedBodyKey, outcome,
                        reason, ordinarySelection, "MUTATED_FALLBACK_DISPOSITION");
                default -> throw new IllegalArgumentException("unknown fact mutation: " + field);
            };
        }

        private Record copy(String form, String roles, String carriers, String layout, String requested,
                String selected, boolean materialized, String ir, String hash, String body, String result,
                String detail) {
            return new Record(occurrenceId, form, loweredForm, roles, boundaryIds, virtualValues, unitCount,
                    affineAddressPairCount, carriers, layout, requested, selected, materialized,
                    materializationsSelected, providerSupported, preparerAdmitted, ir, generatedEntryDescriptor,
                    hash, body, result, detail, ordinarySelection, fallbackDisposition);
        }
    }
}

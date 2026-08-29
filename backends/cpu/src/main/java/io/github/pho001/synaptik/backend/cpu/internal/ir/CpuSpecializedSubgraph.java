package io.github.pho001.synaptik.backend.cpu.internal.ir;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ClampRangeAttrs;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Immutable CPU-private description of one bounded semantic subgraph recognized during cold
 * preparation. A fact describes topology and eligibility only: it is neither generated-kernel IR
 * nor an artifact key, and cannot authorize execution that the independently built baseline does
 * not already provide. Its baseline snapshot is exact and fail-closed: disagreement in portable
 * IR, specialization, ranges, materialized boundaries, resource topology, or executable geometry
 * rejects the fact instead of changing the selected plan.
 */
public sealed interface CpuSpecializedSubgraph permits CpuSpecializedSubgraph.MatmulEpilogue,
        CpuSpecializedSubgraph.ConvolutionEpilogue,
        CpuSpecializedSubgraph.ReductionEpilogue,
        CpuSpecializedSubgraph.ExplicitSemanticKernel {
    /** Returns the recognized graph members.
     * @return stable partition-node ordinals in semantic member order */
    List<Integer> memberNodeOrdinals();
    /** Returns the associated unchanged execution units.
     * @return stable baseline-unit indices covering all members exactly */
    List<Integer> baselineUnitIndices();
    /** Returns the recognized anchor's input types.
     * @return ordered semantic input data types */
    List<DataType> inputDataTypes();
    /** Returns the recognized form's published result types.
     * @return ordered semantic result data types */
    List<DataType> resultDataTypes();
    /** Returns the recognition eligibility access projection.
     * @return immutable resolved access facts for inputs followed by results */
    List<AccessFact> accessFacts();
    /** Returns the literal suffix represented by this fact.
     * @return the literal admitted external suffix */
    Epilogue epilogue();
    /** Returns the relationship to current execution.
     * @return how the unchanged baseline realizes the recognized form */
    ExecutionDisposition disposition();
    /** Returns the stable cold recognition identity.
     * @return graph-identity-free typed recognition identity */
    StructuralIdentity structuralIdentity();

    /** Closed recognized anchor family. */
    enum Family {
        /** Matrix multiplication anchor. */ MATMUL,
        /** Rank-specific convolution anchor. */ CONVOLUTION,
        /** Floating numerical-reduction anchor. */ REDUCTION,
        /** Existing first-class semantic kernel. */ EXPLICIT_SEMANTIC_KERNEL
    }
    /** Closed recognized semantic form. */
    enum Form {
        /** Matrix multiplication. */ MATMUL,
        /** Visible NCW Conv1d composition. */ CONV1D_COMPOSITION,
        /** NCHW Conv2d. */ CONV2D,
        /** NCDHW Conv3d. */ CONV3D,
        /** Sum reduction. */ SUM,
        /** Mean reduction. */ MEAN,
        /** Product reduction. */ PROD,
        /** Minimum reduction. */ MIN,
        /** Maximum reduction. */ MAX,
        /** Log-sum-exp reduction. */ LOG_SUM_EXP,
        /** Variance reduction. */ VARIANCE,
        /** Standard-deviation reduction. */ STANDARD_DEVIATION,
        /** L1-norm reduction. */ L1_NORM,
        /** L2-norm reduction. */ L2_NORM,
        /** First-class softmax. */ SOFTMAX,
        /** First-class log-softmax. */ LOG_SOFTMAX,
        /** First-class Layer normalization. */ LAYER_NORM,
        /** First-class RMS normalization. */ RMS_NORM,
        /** First-class batch-normalization inference. */ BATCH_NORM_INFERENCE,
        /** First-class batch-normalization training. */ BATCH_NORM_TRAINING
    }
    /** Relationship between recognition and already implemented execution. */
    enum ExecutionDisposition {
        /** Existing identical specialized execution. */ EXISTING_SPECIALIZED,
        /** Existing ordinary materialized split execution. */ ORDINARY_SPLIT,
        /** Canonical executable MATMUL split plus one exact complete fused alternative. */
        EXECUTABLE_ALTERNATIVES,
        /** Recognized diagnostic anchor without CPU execution. */ UNSUPPORTED_ANCHOR
    }
    /** MATMUL right-input topology retained without claiming linear execution. */
    enum MatmulInputForm {
        /** Ordinary right input. */ ORDINARY,
        /** Single-use rank-two transposed weight. */ TRANSPOSED_WEIGHT
    }
    /** Position of the preceding result in an admitted binary ADD. */
    enum AddInputOrder {
        /** No external ADD. */ NONE,
        /** Preceding result is the left ADD operand. */ PRECEDING_LEFT,
        /** Preceding result is the right ADD operand. */ PRECEDING_RIGHT
    }
    /** Closed terminal suffix vocabulary. */
    enum Terminal {
        /** No terminal. */ NONE,
        /** ReLU terminal. */ RELU,
        /** Sigmoid terminal. */ SIGMOID,
        /** Hyperbolic-tangent terminal. */ TANH,
        /** Exact GELU terminal. */ GELU,
        /** Tanh-approximation GELU terminal. */ GELU_TANH_APPROXIMATION,
        /** SiLU terminal. */ SILU,
        /** Typed clamp terminal. */ CLAMP
    }
    /** Closed reduction-attribute projection. */
    enum ReductionForm {
        /** Full reduction. */ FULL,
        /** Single-axis reduction. */ SINGLE_AXIS,
        /** Multi-axis reduction. */ MULTI_AXIS,
        /** Statistical reduction with correction. */ STATISTICAL
    }
    /** Closed explicit-kernel attribute projection. */
    enum ExplicitForm {
        /** Softmax. */ SOFTMAX,
        /** Log-softmax. */ LOG_SOFTMAX,
        /** Unaffined Layer normalization. */ LAYER,
        /** Affine Layer normalization. */ LAYER_AFFINE,
        /** Unscaled RMS normalization. */ RMS,
        /** Scaled RMS normalization. */ RMS_SCALED,
        /** Batch-normalization inference. */ BATCH_INFERENCE,
        /** Batch-normalization training. */ BATCH_TRAINING
    }
    /** Closed workspace-resource role retained by the baseline topology proof. */
    enum WorkspaceRole {
        /** No workspace. */ NONE,
        /** Contiguous-copy materialization. */ MATERIALIZATION,
        /** Scatter product state. */ SCATTER_PRODUCT,
        /** Ordering merge indices. */ ORDERING_INDICES,
        /** Exact aggregate state. */ AGGREGATE_EXACT_STATE
    }
    /** Closed route retained by every current recognition baseline. */
    enum BaselineRoute { /** Portable generated route. */ PORTABLE }
    /** Generated compute form retained independently of orchestration. */
    enum BaselineCompute { /** Scalar generated body. */ SCALAR, /** Vector generated body. */ VECTOR }
    /** Invocation orchestration retained independently of generated compute. */
    enum BaselineOrchestration {
        /** One invocation range. */ SINGLE_THREAD,
        /** Multiple disjoint invocation ranges. */ PARALLEL
    }
    /** Closed executable geometry owner used by the selected baseline unit. */
    enum RuntimeTopology {
        /** Pointwise unit. */ POINTWISE,
        /** Conv2d geometry. */ CONV2D,
        /** Conv3d geometry. */ CONV3D,
        /** Full-K portable MATMUL geometry. */ MATMUL,
        /** Ordinary aggregate geometry. */ AGGREGATE,
        /** Advanced-reduction geometry. */ ADVANCED_REDUCTION,
        /** Softmax geometry. */ SOFTMAX,
        /** Layer/RMS trailing geometry. */ TRAILING_NORMALIZATION,
        /** Batch-inference geometry. */ BATCH_NORM_INFERENCE,
        /** Batch-training geometry. */ BATCH_NORM_TRAINING
    }

    /**
     * Graph-identity-free projection of one materialized unit boundary.
     *
     * @param dataType represented logical type
     * @param role materialized input or output role
     * @param accessPlan structural access plan
     * @param extents exact cold iteration extents
     * @param baseElementOffset resolved element origin
     * @param effectiveStrides exact right-aligned element strides
     * @param elementCount exact logical range size
     * @param start inclusive bound represented by the retained binding
     * @param end exclusive bound represented by the retained binding
     * @param referencedElementSpan exact referenced storage span in elements
     * @param startCoordinates exact initial odometer coordinates
     * @param startAddress exact initial physical element address
     * @param accessedElementStart inclusive accessed element bound
     * @param accessedElementEnd exclusive accessed element bound
     * @param carrier requested carrier form
     * @param generatedCarrier generated-entry carrier form
     */
    record BoundaryResourceFact(DataType dataType, CpuKernelIr.Value.Kind role,
            CpuAccessPlan accessPlan, List<Long> extents, long baseElementOffset,
            List<Long> effectiveStrides, long elementCount, long start, long end,
            long referencedElementSpan, List<Long> startCoordinates, long startAddress,
            long accessedElementStart, long accessedElementEnd, CarrierAccess carrier,
            CarrierAccess generatedCarrier) {
        /** Validates and snapshots one stable materialized boundary projection. */
        public BoundaryResourceFact {
            Objects.requireNonNull(dataType, "dataType");
            Objects.requireNonNull(role, "role");
            Objects.requireNonNull(accessPlan, "accessPlan");
            extents = List.copyOf(extents);
            effectiveStrides = List.copyOf(effectiveStrides);
            startCoordinates = List.copyOf(startCoordinates);
            Objects.requireNonNull(carrier, "carrier");
            Objects.requireNonNull(generatedCarrier, "generatedCarrier");
            if (role == CpuKernelIr.Value.Kind.VIRTUAL || baseElementOffset < 0
                    || elementCount < 0 || start < 0 || end < start || end > elementCount
                    || referencedElementSpan < 0 || startAddress < 0
                    || accessedElementStart < 0 || accessedElementEnd < accessedElementStart
                    || accessedElementEnd > referencedElementSpan
                    || startCoordinates.size() != extents.size()) {
                throw new IllegalArgumentException("baseline boundary resource is invalid");
            }
        }
    }

    /**
     * Stable materialization decision with graph and requirement identities removed.
     *
     * @param sourceBoundaryIndex copied input position
     * @param sourceBinding exact original source binding
     * @param consumerBinding exact dense consumer binding
     * @param elementCount copied logical elements
     * @param byteCount exact copied byte count
     * @param byteAlignment required workspace alignment
     * @param useCount exact lowered-unit use count
     * @param expectedRunCount cold repeated-run estimate
     * @param directCost estimated direct cost
     * @param copyCost estimated copy cost
     * @param contiguousCost estimated contiguous-consumer cost
     * @param copiedTotalCost estimated complete copied cost
     * @param netBenefit selected positive benefit
     * @param benefitBasisPoints selected benefit in basis points
     * @param selectionReason stable cold diagnostic reason
     */
    record MaterializationFact(int sourceBoundaryIndex, CpuAccessPlan.Binding sourceBinding,
            CpuAccessPlan.Binding consumerBinding, long elementCount, long byteCount,
            long byteAlignment, long useCount, long expectedRunCount, long directCost,
            long copyCost, long contiguousCost, long copiedTotalCost, long netBenefit,
            int benefitBasisPoints, String selectionReason) {
        /** Validates one graph-identity-free materialization projection. */
        public MaterializationFact {
            Objects.requireNonNull(sourceBinding, "sourceBinding");
            Objects.requireNonNull(consumerBinding, "consumerBinding");
            Objects.requireNonNull(selectionReason, "selectionReason");
            if (sourceBoundaryIndex < 0 || elementCount < 0 || byteCount < 0
                    || byteAlignment <= 0 || useCount <= 0 || expectedRunCount <= 0
                    || directCost < 0 || copyCost < 0 || contiguousCost < 0
                    || copiedTotalCost < 0 || netBenefit <= 0 || benefitBasisPoints < 0
                    || benefitBasisPoints > 10_000) {
                throw new IllegalArgumentException("baseline materialization fact is invalid");
            }
        }
    }

    /**
     * Exact stable route, specialization, range, and executable-geometry facts for one unit.
     *
     * @param route selected route
     * @param specialization complete existing generated specialization
     * @param compute generated compute form
     * @param orchestration invocation orchestration
     * @param extents exact unit range extents
     * @param elementCount exact logical work count
     * @param selectedRangeCount selected invocation range count
     * @param minimumElementsPerWorker minimum submitted work
     * @param vectorSpeciesBitSize selected vector width, or zero
     * @param affineAddressPairs exact affine executable geometry, otherwise empty
     * @param materialization stable materialization parameters, when selected
     * @param runtimeTopology executable geometry owner
     * @param packedGeometry exact zero-base geometry consumed by the generated entry
     * @param fusionReason stable selected-unit diagnostic
     */
    record BaselineExecutionFact(BaselineRoute route, CpuKernelSpecialization specialization,
            BaselineCompute compute, BaselineOrchestration orchestration, List<Long> extents,
            long elementCount, int selectedRangeCount, long minimumElementsPerWorker,
            int vectorSpeciesBitSize, List<Long> affineAddressPairs,
            Optional<MaterializationFact> materialization, RuntimeTopology runtimeTopology,
            List<Long> packedGeometry, String fusionReason) {
        /** Validates and snapshots the complete stable execution projection. */
        public BaselineExecutionFact {
            Objects.requireNonNull(route, "route");
            Objects.requireNonNull(specialization, "specialization");
            Objects.requireNonNull(compute, "compute");
            Objects.requireNonNull(orchestration, "orchestration");
            extents = List.copyOf(extents);
            affineAddressPairs = List.copyOf(affineAddressPairs);
            materialization = Objects.requireNonNull(materialization, "materialization");
            Objects.requireNonNull(runtimeTopology, "runtimeTopology");
            packedGeometry = List.copyOf(packedGeometry);
            Objects.requireNonNull(fusionReason, "fusionReason");
            boolean vector = compute == BaselineCompute.VECTOR;
            boolean parallel = orchestration == BaselineOrchestration.PARALLEL;
            if (elementCount < 0 || selectedRangeCount <= 0 || minimumElementsPerWorker <= 0
                    || vectorSpeciesBitSize < 0 || vector != (vectorSpeciesBitSize > 0)
                    || parallel != (selectedRangeCount >= 2)
                    || specialization.vectorSpeciesBitSize() != vectorSpeciesBitSize
                    || specialization.executionStrategy().compute().name().equals(compute.name())
                        == false) {
                throw new IllegalArgumentException("baseline execution fact is invalid");
            }
        }
    }

    /**
     * Exact graph-identity-free workspace topology of one associated baseline unit.
     *
     * @param role closed workspace purpose
     * @param byteSize exact declared byte size, zero when absent
     * @param byteAlignment exact declared alignment, zero when absent
     */
    record WorkspaceResourceFact(WorkspaceRole role, long byteSize, long byteAlignment) {
        /** Validates one workspace-resource projection. */
        public WorkspaceResourceFact {
            Objects.requireNonNull(role, "role");
            boolean absent = role == WorkspaceRole.NONE;
            if (absent != (byteSize == 0 && byteAlignment == 0)
                    || (!absent && (byteSize < 0 || byteAlignment <= 0))) {
                throw new IllegalArgumentException("baseline workspace resource is invalid");
            }
        }
    }

    /**
     * Stable exact IR and resource topology of one associated CPU 0008B baseline unit.
     *
     * @param structuralKey existing portable-IR structural key
     * @param execution exact stable route, specialization, range, and executable facts
     * @param dependencies stable direct producer-unit indices
     * @param boundaries ordered materialized boundary/resource facts
     * @param outputCount number of trailing output boundaries
     * @param workspace exact workspace topology
     */
    record BaselineUnitFact(String structuralKey, BaselineExecutionFact execution,
            List<Integer> dependencies,
            List<BoundaryResourceFact> boundaries, int outputCount,
            WorkspaceResourceFact workspace) {
        /**
         * Validates one associated unit without graph, slot, run, loader, generated-class, or
         * artifact-store identity.
         */
        public BaselineUnitFact {
            Objects.requireNonNull(structuralKey, "structuralKey");
            Objects.requireNonNull(execution, "execution");
            dependencies = List.copyOf(dependencies);
            boundaries = List.copyOf(boundaries);
            Objects.requireNonNull(workspace, "workspace");
            List<DataType> boundaryTypes = boundaries.stream()
                    .map(BoundaryResourceFact::dataType).toList();
            List<CarrierAccess> generatedCarriers = boundaries.stream()
                    .map(BoundaryResourceFact::generatedCarrier).toList();
            boolean materialized = execution.materialization().isPresent();
            if (structuralKey.isBlank()
                    || !structuralKey.equals(execution.specialization()
                        .loweringFingerprint().hex())
                    || boundaries.isEmpty() || outputCount <= 0
                    || outputCount > boundaries.size()
                    || !boundaryTypes.equals(execution.specialization().boundaryDataTypes())
                    || !generatedCarriers.equals(execution.specialization().carrierPattern())
                    || materialized != (execution.specialization()
                        .materializedSourcePosition() >= 0)
                    || materialized && execution.materialization().orElseThrow()
                        .sourceBoundaryIndex() != execution.specialization()
                            .materializedSourcePosition()
                    || materialized != (workspace.role() == WorkspaceRole.MATERIALIZATION)
                    || (execution.runtimeTopology() == RuntimeTopology.POINTWISE)
                        != execution.packedGeometry().isEmpty()
                    || boundaries.subList(boundaries.size() - outputCount, boundaries.size())
                        .stream().anyMatch(value -> value.role() != CpuKernelIr.Value.Kind.OUTPUT)
                    || boundaries.subList(0, boundaries.size() - outputCount).stream()
                        .anyMatch(value -> value.role() != CpuKernelIr.Value.Kind.INPUT)
                    || dependencies.stream().anyMatch(value -> value == null || value < 0)) {
                throw new IllegalArgumentException("baseline unit fact is invalid");
            }
        }
    }

    /**
     * Literal external epilogue topology.
     *
     * @param addInputOrder exact ADD operand order, or {@link AddInputOrder#NONE}
     * @param terminal exact terminal operation
     * @param clampRange exact typed bounds, present only for {@link Terminal#CLAMP}
     */
    record Epilogue(AddInputOrder addInputOrder, Terminal terminal,
            Optional<ClampRangeAttrs> clampRange) {
        /** Validates the closed suffix representation. */
        public Epilogue {
            Objects.requireNonNull(addInputOrder, "addInputOrder");
            Objects.requireNonNull(terminal, "terminal");
            clampRange = Objects.requireNonNull(clampRange, "clampRange");
            if (clampRange.isPresent() != (terminal == Terminal.CLAMP)) {
                throw new IllegalArgumentException("CLAMP terminal and bounds must agree");
            }
        }
        /** Creates the empty external epilogue.
         * @return the empty external epilogue */
        public static Epilogue none() {
            return new Epilogue(AddInputOrder.NONE, Terminal.NONE, Optional.empty());
        }
        /** Counts the literal suffix operations.
         * @return number of represented external epilogue operations */
        public int operationCount() {
            return (addInputOrder == AddInputOrder.NONE ? 0 : 1)
                    + (terminal == Terminal.NONE ? 0 : 1);
        }
    }

    /**
     * Exact resolved descriptor/access projection used by recognition eligibility.
     *
     * @param dataType exact logical type
     * @param shape exact static Shape
     * @param storageOffset non-negative element offset
     * @param strides non-negative element strides in axis order
     * @param regime normalized CPU access regime
     * @param injective whether logical coordinates select distinct addresses
     */
    record AccessFact(DataType dataType, Shape shape, long storageOffset, List<Long> strides,
            CpuAccessPlan.Regime regime, boolean injective) {
        /** Validates and snapshots one resolved descriptor projection. */
        public AccessFact {
            Objects.requireNonNull(dataType, "dataType");
            Objects.requireNonNull(shape, "shape");
            strides = List.copyOf(strides);
            Objects.requireNonNull(regime, "regime");
            if (storageOffset < 0 || strides.size() != shape.rank()
                    || strides.stream().anyMatch(value -> value == null || value < 0)) {
                throw new IllegalArgumentException("resolved access fact is invalid");
            }
        }
    }

    /** Closed typed anchor-attribute projection used in structural recognition identity. */
    sealed interface AnchorAttributes permits MatmulAttributes, ConvolutionAttributes,
            ReductionAttributes, ExplicitAttributes { }

    /** Exact MATMUL input-topology projection.
     * @param inputForm exact right-input topology */
    record MatmulAttributes(MatmulInputForm inputForm) implements AnchorAttributes {
        /** Validates the MATMUL projection. */
        public MatmulAttributes { Objects.requireNonNull(inputForm, "inputForm"); }
    }

    /**
     * Rank-specific convolution attribute projection.
     * @param dimensions spatial rank
     * @param strides ordered spatial strides
     * @param paddings ordered spatial symmetric paddings
     * @param dilations ordered spatial dilations
     * @param groups positive group count
     * @param intrinsicBias whether the anchor has its intrinsic rank-one bias input
     */
    record ConvolutionAttributes(int dimensions, List<Long> strides, List<Long> paddings,
            List<Long> dilations, long groups, boolean intrinsicBias) implements AnchorAttributes {
        /** Validates and snapshots the rank-specific geometry. */
        public ConvolutionAttributes {
            strides = List.copyOf(strides); paddings = List.copyOf(paddings);
            dilations = List.copyOf(dilations);
            if (dimensions < 1 || dimensions > 3 || strides.size() != dimensions
                    || paddings.size() != dimensions || dilations.size() != dimensions
                    || groups <= 0 || strides.stream().anyMatch(v -> v == null || v <= 0)
                    || paddings.stream().anyMatch(v -> v == null || v < 0)
                    || dilations.stream().anyMatch(v -> v == null || v <= 0)) {
                throw new IllegalArgumentException("convolution attributes are invalid");
            }
        }
    }

    /**
     * Exact admitted reduction attribute projection.
     * @param kind exact first-class reduction kind
     * @param form exact attribute form
     * @param axes normalized ordered selected axes
     * @param keepDimensions exact rank-retention flag
     * @param correction statistical divisor correction, otherwise zero
     */
    record ReductionAttributes(AggregateReductionKind kind, ReductionForm form,
            List<Integer> axes, boolean keepDimensions, long correction)
            implements AnchorAttributes {
        /** Validates and snapshots the reduction projection. */
        public ReductionAttributes {
            Objects.requireNonNull(kind, "kind"); Objects.requireNonNull(form, "form");
            axes = List.copyOf(axes);
            if (axes.stream().anyMatch(axis -> axis == null || axis < 0) || correction < 0) {
                throw new IllegalArgumentException("reduction attributes are invalid");
            }
        }
    }

    /**
     * Exact explicit-kernel attribute projection.
     * @param form exact first-class form
     * @param axis selected axis, or minus one when not axis-based
     * @param normalizedShape trailing normalized Shape, or scalar Shape when not applicable
     * @param firstScalar first exact scalar attribute, empty when absent
     * @param secondScalar second exact scalar attribute, empty when absent
     */
    record ExplicitAttributes(ExplicitForm form, int axis, Shape normalizedShape,
            Optional<ScalarValue> firstScalar, Optional<ScalarValue> secondScalar)
            implements AnchorAttributes {
        /** Validates the explicit semantic projection. */
        public ExplicitAttributes {
            Objects.requireNonNull(form, "form");
            Objects.requireNonNull(normalizedShape, "normalizedShape");
            firstScalar = Objects.requireNonNull(firstScalar, "firstScalar");
            secondScalar = Objects.requireNonNull(secondScalar, "secondScalar");
            if (axis < -1) throw new IllegalArgumentException("explicit semantic axis is invalid");
        }
    }

    /**
     * Stable typed candidate identity containing no graph-value, slot, run, loader,
     * generated-class, or artifact-store identity. Associated supported facts retain the exact
     * existing carrier, specialization, range, and executable-geometry facts solely to validate
     * equality with the independently selected baseline; they do not become artifact identity.
     * @param family closed family
     * @param form closed form
     * @param inputDataTypes ordered input types
     * @param resultDataTypes ordered result types
     * @param accessFacts ordered descriptor/access projections
     * @param attributes exact typed anchor attributes
     * @param epilogue literal suffix
     * @param baselineUnits exact associated baseline IR/resource topology; empty only for an
     *     unsupported MATMUL diagnostic fact
     */
    record StructuralIdentity(Family family, Form form, List<DataType> inputDataTypes,
            List<DataType> resultDataTypes, List<AccessFact> accessFacts,
            AnchorAttributes attributes, Epilogue epilogue,
            List<BaselineUnitFact> baselineUnits) {
        /** Validates and snapshots one graph-identity-free recognition identity. */
        public StructuralIdentity {
            Objects.requireNonNull(family, "family"); Objects.requireNonNull(form, "form");
            inputDataTypes = List.copyOf(inputDataTypes);
            resultDataTypes = List.copyOf(resultDataTypes);
            accessFacts = List.copyOf(accessFacts);
            baselineUnits = List.copyOf(baselineUnits);
            Objects.requireNonNull(attributes, "attributes"); Objects.requireNonNull(epilogue, "epilogue");
            if (accessFacts.size() > 10) {
                throw new IllegalArgumentException("recognition identity references at most ten positions");
            }
            if (baselineUnits.size() > 2) {
                throw new IllegalArgumentException("recognition identity associates at most two units");
            }
        }
    }

    /**
     * MATMUL recognition fact associated with its canonical executable baseline units.
     * @param memberNodeOrdinals stable semantic member ordinals
     * @param baselineUnitIndices exact one- or two-unit canonical executable split
     * @param inputDataTypes ordered semantic input types
     * @param resultDataTypes ordered semantic result types
     * @param accessFacts immutable input/result access projections
     * @param epilogue literal recognized suffix
     * @param structuralIdentity graph-identity-free recognition identity
     */
    record MatmulEpilogue(List<Integer> memberNodeOrdinals, List<Integer> baselineUnitIndices,
            List<DataType> inputDataTypes, List<DataType> resultDataTypes,
            List<AccessFact> accessFacts, Epilogue epilogue, StructuralIdentity structuralIdentity)
            implements CpuSpecializedSubgraph {
        /** Validates and snapshots an executable-alternatives fact. */
        public MatmulEpilogue {
            memberNodeOrdinals = members(memberNodeOrdinals); baselineUnitIndices = units(baselineUnitIndices);
            inputDataTypes = List.copyOf(inputDataTypes); resultDataTypes = List.copyOf(resultDataTypes);
            accessFacts = List.copyOf(accessFacts); Objects.requireNonNull(epilogue, "epilogue");
            requireIdentity(structuralIdentity, Family.MATMUL, Form.MATMUL, epilogue,
                    inputDataTypes, resultDataTypes, accessFacts, baselineUnitIndices);
            if (baselineUnitIndices.size() > 2
                    || baselineUnitIndices.size() != structuralIdentity.baselineUnits().size())
                throw new IllegalArgumentException(
                        "MATMUL alternatives require the exact one- or two-unit baseline");
        }
        /** Returns executable alternatives for associated facts and the legacy diagnostic
         * unsupported disposition only for an explicitly unassociated focused query.
         * @return the exact association disposition */
        @Override public ExecutionDisposition disposition() {
            return baselineUnitIndices.isEmpty() ? ExecutionDisposition.UNSUPPORTED_ANCHOR
                    : ExecutionDisposition.EXECUTABLE_ALTERNATIVES;
        }
    }

    /**
     * Rank-specific convolution recognition fact.
     * @param form exact Conv1d-composition, Conv2d, or Conv3d form
     * @param memberNodeOrdinals stable semantic member ordinals
     * @param baselineUnitIndices exact associated unchanged unit indices
     * @param inputDataTypes ordered semantic input types
     * @param resultDataTypes ordered semantic result types
     * @param accessFacts immutable input/result access projections
     * @param epilogue literal recognized suffix
     * @param disposition existing-specialized or ordinary-split execution relationship
     * @param structuralIdentity graph-identity-free recognition identity
     */
    record ConvolutionEpilogue(Form form, List<Integer> memberNodeOrdinals,
            List<Integer> baselineUnitIndices, List<DataType> inputDataTypes,
            List<DataType> resultDataTypes, List<AccessFact> accessFacts, Epilogue epilogue,
            ExecutionDisposition disposition, StructuralIdentity structuralIdentity)
            implements CpuSpecializedSubgraph {
        /** Validates and snapshots a convolution fact. */
        public ConvolutionEpilogue {
            if (form != Form.CONV1D_COMPOSITION && form != Form.CONV2D && form != Form.CONV3D)
                throw new IllegalArgumentException("convolution form is invalid");
            memberNodeOrdinals = members(memberNodeOrdinals); baselineUnitIndices = units(baselineUnitIndices);
            inputDataTypes = List.copyOf(inputDataTypes); resultDataTypes = List.copyOf(resultDataTypes);
            accessFacts = List.copyOf(accessFacts); Objects.requireNonNull(epilogue, "epilogue");
            Objects.requireNonNull(disposition, "disposition");
            if (disposition == ExecutionDisposition.UNSUPPORTED_ANCHOR)
                throw new IllegalArgumentException("only MATMUL may be unsupported");
            requireIdentity(structuralIdentity, Family.CONVOLUTION, form, epilogue,
                    inputDataTypes, resultDataTypes, accessFacts, baselineUnitIndices);
        }
    }

    /**
     * Floating-reduction recognition fact retaining ordinary split execution.
     * @param form exact admitted floating-reduction form
     * @param memberNodeOrdinals stable semantic member ordinals
     * @param baselineUnitIndices exact associated unchanged unit indices
     * @param inputDataTypes ordered semantic input types
     * @param resultDataTypes ordered semantic result types
     * @param accessFacts immutable input/result access projections
     * @param epilogue literal recognized suffix
     * @param structuralIdentity graph-identity-free recognition identity
     */
    record ReductionEpilogue(Form form, List<Integer> memberNodeOrdinals,
            List<Integer> baselineUnitIndices, List<DataType> inputDataTypes,
            List<DataType> resultDataTypes, List<AccessFact> accessFacts, Epilogue epilogue,
            StructuralIdentity structuralIdentity) implements CpuSpecializedSubgraph {
        /** Validates and snapshots a reduction fact. */
        public ReductionEpilogue {
            memberNodeOrdinals = members(memberNodeOrdinals); baselineUnitIndices = units(baselineUnitIndices);
            inputDataTypes = List.copyOf(inputDataTypes); resultDataTypes = List.copyOf(resultDataTypes);
            accessFacts = List.copyOf(accessFacts); Objects.requireNonNull(epilogue, "epilogue");
            requireIdentity(structuralIdentity, Family.REDUCTION, form, epilogue,
                    inputDataTypes, resultDataTypes, accessFacts, baselineUnitIndices);
        }
        /** Returns the unchanged split disposition.
         * @return always {@link ExecutionDisposition#ORDINARY_SPLIT} */
        @Override public ExecutionDisposition disposition() { return ExecutionDisposition.ORDINARY_SPLIT; }
    }

    /**
     * Exact first-class semantic-kernel fact retaining its established one-unit execution.
     * @param form exact first-class softmax or normalization form
     * @param memberNodeOrdinals the single stable semantic member ordinal
     * @param baselineUnitIndices the single associated unchanged unit index
     * @param inputDataTypes ordered semantic input types
     * @param resultDataTypes ordered semantic result types
     * @param accessFacts immutable input/result access projections
     * @param structuralIdentity graph-identity-free recognition identity
     */
    record ExplicitSemanticKernel(Form form, List<Integer> memberNodeOrdinals,
            List<Integer> baselineUnitIndices, List<DataType> inputDataTypes,
            List<DataType> resultDataTypes, List<AccessFact> accessFacts,
            StructuralIdentity structuralIdentity) implements CpuSpecializedSubgraph {
        /** Validates and snapshots a one-node explicit semantic fact. */
        public ExplicitSemanticKernel {
            memberNodeOrdinals = members(memberNodeOrdinals); baselineUnitIndices = units(baselineUnitIndices);
            inputDataTypes = List.copyOf(inputDataTypes); resultDataTypes = List.copyOf(resultDataTypes);
            accessFacts = List.copyOf(accessFacts);
            if (memberNodeOrdinals.size() != 1 || baselineUnitIndices.size() != 1)
                throw new IllegalArgumentException("explicit semantic fact requires one node and unit");
            requireIdentity(structuralIdentity, Family.EXPLICIT_SEMANTIC_KERNEL, form,
                    Epilogue.none(), inputDataTypes, resultDataTypes, accessFacts,
                    baselineUnitIndices);
        }
        /** @return explicit kernels have no external epilogue */
        @Override public Epilogue epilogue() { return Epilogue.none(); }
        /** @return always {@link ExecutionDisposition#EXISTING_SPECIALIZED} */
        @Override public ExecutionDisposition disposition() { return ExecutionDisposition.EXISTING_SPECIALIZED; }
    }

    private static List<Integer> members(List<Integer> source) {
        var result = List.copyOf(source);
        if (result.isEmpty() || result.size() > 6 || result.stream().anyMatch(v -> v == null || v < 0)
                || result.stream().distinct().count() != result.size())
            throw new IllegalArgumentException("recognized member ordinals are invalid");
        return result;
    }
    private static List<Integer> units(List<Integer> source) {
        var result = List.copyOf(source);
        if (result.size() > 2 || result.stream().anyMatch(v -> v == null || v < 0)
                || result.stream().distinct().count() != result.size())
            throw new IllegalArgumentException("baseline unit association is invalid");
        return result;
    }
    private static void requireIdentity(StructuralIdentity identity, Family family, Form form,
            Epilogue epilogue, List<DataType> inputDataTypes, List<DataType> resultDataTypes,
            List<AccessFact> accessFacts, List<Integer> baselineUnitIndices) {
        Objects.requireNonNull(identity, "structuralIdentity");
        if (identity.family() != family || identity.form() != form
                || !identity.epilogue().equals(epilogue)
                || !identity.inputDataTypes().equals(inputDataTypes)
                || !identity.resultDataTypes().equals(resultDataTypes)
                || !identity.accessFacts().equals(accessFacts)
                || identity.baselineUnits().size() != baselineUnitIndices.size())
            throw new IllegalArgumentException("recognition identity disagrees with fact");
    }
}

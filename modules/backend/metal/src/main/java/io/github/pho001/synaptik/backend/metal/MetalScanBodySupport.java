package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.tensor.ScanBody;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.ArrayList;
import java.util.Optional;
import java.util.stream.IntStream;

/**
 * Qualifies genuine empty ordinary occurrences inside the bounded Scan binder. Original dimensions,
 * ordinary family equations and trailing feature products remain intact; no sample batch is
 * substituted. Physical body materializations use the same canonical ordinary native-ABI validator
 * as nonempty units, scoped explicitly to Scan. Neither empty cardinality nor child support grants
 * ordinary public empty/symbolic admission, low arithmetic or missing body control.
 */
final class MetalScanBodySupport {
    private MetalScanBodySupport() {}

    /**
     * Qualifies actual empty metadata through Model inference and the ordinary physical owner.
     * @param query ordinary local occurrence with at least one empty input or output domain
     * @return whether its complete equations, carriers, layouts and native bounds are eligible
     */
    static boolean supports(OperationCapabilityQuery query) {
        try {
            if (query.scanBody().isPresent() || query.scanActivityPlan().isPresent()) return false;
            var descriptors = new ArrayList<TensorDescriptor>(query.inputs()); descriptors.addAll(query.outputs());
            boolean empty = false;
            var physical = new ArrayList<MetalMpsGraphProgram.ValueDescriptor>();
            for (var descriptor : descriptors) {
                DataType type = descriptor.dataType();
                if (!descriptor.shape().isFullyStatic() || descriptor.shape().rank() > 16 || descriptor.layout().isEmpty()
                        || type != DataType.FLOAT32 && type != DataType.BOOL && type != DataType.INT32 && type != DataType.INT64) return false;
                long trailing = 1;
                for (int a = descriptor.shape().rank() - 1; a >= 0; a--) {
                    long extent = descriptor.shape().dimension(a).staticSize().orElseThrow(); empty |= extent == 0;
                    trailing = Math.multiplyExact(trailing, extent);
                }
                Math.multiplyExact(trailing, type.byteWidth());
                physical.add(MetalMpsGraphProgram.ValueDescriptor.from(new TensorDescriptor(type, descriptor.shape(),
                        Optional.of(LayoutDescriptor.contiguous(descriptor.shape())), descriptor.requiresGrad())));
            }
            if (!empty) return false;
            var inferred = ScanBody.describeOrdinary(query.operation(), query.inputs(), query.outputs().size(), Optional.empty());
            for (int p = 0; p < inferred.size(); p++) {
                var actual = query.outputs().get(p); var expected = inferred.get(p);
                if (actual.dataType() != expected.dataType() || !actual.shape().equals(expected.shape())
                        || actual.requiresGrad() != expected.requiresGrad()) return false;
            }
            var node = MetalNegPartitionPreparer.lowerScanBody(query.operation(), IntStream.range(0, query.inputs().size()).toArray(),
                    IntStream.range(query.inputs().size(), descriptors.size()).toArray(), query.inputs(), query.outputs());
            MetalNativeApi.ProgramExecutableAbi.validateScanBody(node, physical);
            return true;
        } catch (IllegalArgumentException | ArithmeticException | java.util.NoSuchElementException unsupported) {
            return false;
        }
    }
}

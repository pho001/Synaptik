package backend.accelerator.lowering;

import tensor.DataType;

import java.util.List;

/**
 * One lowered backend primitive inside a selected GPU region manifest.
 *
 * @param primitiveId stable primitive id inside the manifest
 * @param primitiveType lowered backend primitive type
 * @param sourceOriginalNodeIds original compiled node ids that produced this primitive
 * @param inputRefs primitive input references in manifest/debug form
 * @param outputRef primitive output reference in manifest/debug form
 * @param dataType primitive output dtype
 * @param shape primitive output shape
 * @param reasons stable primitive-level rejection or support reasons
 */
public record GpuLoweredPrimitiveManifest(
        String primitiveId,
        String primitiveType,
        List<Integer> sourceOriginalNodeIds,
        List<String> inputRefs,
        String outputRef,
        DataType dataType,
        List<Integer> shape,
        List<GpuLoweringUnsupportedReason> reasons
) {
    public GpuLoweredPrimitiveManifest {
        primitiveId = primitiveId == null ? "" : primitiveId;
        primitiveType = primitiveType == null ? "UNKNOWN" : primitiveType;
        sourceOriginalNodeIds = List.copyOf(sourceOriginalNodeIds == null ? List.of() : sourceOriginalNodeIds);
        inputRefs = List.copyOf(inputRefs == null ? List.of() : inputRefs);
        outputRef = outputRef == null ? "" : outputRef;
        dataType = dataType == null ? DataType.FLOAT32 : dataType;
        shape = List.copyOf(shape == null ? List.of() : shape);
        reasons = List.copyOf(reasons == null ? List.of() : reasons);
    }
}

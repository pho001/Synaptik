package backend.metal;

import tensor.DataType;

/**
 * Metal cast-pair legality for explicit graph CAST nodes.
 *
 * <p>This policy is deliberately narrower than storage residency. INT32 and BOOL can live in
 * Metal buffers for index/predicate roles, but that does not make arbitrary numeric casts legal.
 * The only value-changing cast pairs supported today are FLOAT32 <-> BFLOAT16.</p>
 */
public final class MetalCastPolicy {
    private MetalCastPolicy() {
    }

    public static Decision decide(DataType source, DataType target) {
        if (source == null || target == null) {
            return unsupported(source, target, MetalDTypeReasonCode.UNSUPPORTED_CAST_PAIR,
                    "CAST requires source and target dtype metadata");
        }
        if (source == DataType.FLOAT64 || target == DataType.FLOAT64) {
            return unsupported(source, target, MetalDTypeReasonCode.FLOAT64_UNSUPPORTED,
                    "Metal CAST does not support FLOAT64 source or target tensors");
        }
        if (source == target) {
            return switch (source) {
                case FLOAT32, BFLOAT16, BOOL, INT32 -> new Decision(
                        source,
                        target,
                        true,
                        MetalDTypeReasonCode.SUPPORTED,
                        "identity CAST is metadata-only for " + source
                );
                case FLOAT64 -> unsupported(source, target, MetalDTypeReasonCode.FLOAT64_UNSUPPORTED,
                        "Metal CAST does not support FLOAT64 identity tensors");
            };
        }
        if ((source == DataType.FLOAT32 && target == DataType.BFLOAT16)
                || (source == DataType.BFLOAT16 && target == DataType.FLOAT32)) {
            return new Decision(
                    source,
                    target,
                    true,
                    MetalDTypeReasonCode.SUPPORTED,
                    "Metal CAST supports FLOAT32 <-> BFLOAT16 conversion"
            );
        }
        return unsupported(source, target, MetalDTypeReasonCode.UNSUPPORTED_CAST_PAIR,
                "Metal CAST supports only identity casts and FLOAT32 <-> BFLOAT16 conversion; got "
                        + source + " -> " + target);
    }

    private static Decision unsupported(
            DataType source,
            DataType target,
            MetalDTypeReasonCode reasonCode,
            String detail
    ) {
        return new Decision(source, target, false, reasonCode, detail);
    }

    public record Decision(
            DataType source,
            DataType target,
            boolean supported,
            MetalDTypeReasonCode reasonCode,
            String detail
    ) {
        public Decision {
            reasonCode = reasonCode == null ? MetalDTypeReasonCode.UNSUPPORTED_CAST_PAIR : reasonCode;
            detail = detail == null ? "" : detail;
        }
    }
}

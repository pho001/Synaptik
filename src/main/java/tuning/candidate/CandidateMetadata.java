package tuning.candidate;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

public record CandidateMetadata(
        String candidateSpaceId,
        String candidateSpaceVersion,
        String parameterFamily,
        String parameterVariant,
        String graphAutotuneMode,
        boolean runtimeFrozen,
        boolean graphPolicyMutated,
        boolean productionEligible,
        Map<String, String> attributes
) {
    public CandidateMetadata {
        candidateSpaceId = normalize(candidateSpaceId, "unknown");
        candidateSpaceVersion = normalize(candidateSpaceVersion, "1");
        parameterFamily = normalize(parameterFamily, "unknown");
        parameterVariant = normalize(parameterVariant, "unknown");
        graphAutotuneMode = normalize(graphAutotuneMode, "");
        attributes = attributes == null ? Map.of() : Map.copyOf(attributes);
    }

    public static CandidateMetadata generic() {
        return new CandidateMetadata(
                "generic",
                "1",
                "generic",
                "candidate",
                "",
                false,
                false,
                true,
                Map.of()
        );
    }

    public static CandidateMetadata graphStandard(String variant) {
        return new CandidateMetadata(
                "graph-autotune",
                "1",
                "graphPolicy",
                normalize(variant, "current"),
                "STANDARD",
                true,
                false,
                true,
                Map.of()
        );
    }

    public static CandidateMetadata graphResearch(
            String parameterFamily,
            String parameterVariant,
            boolean graphPolicyMutated
    ) {
        return new CandidateMetadata(
                "graph-autotune",
                "1",
                parameterFamily,
                parameterVariant,
                "RESEARCH",
                true,
                graphPolicyMutated,
                false,
                Map.of()
        );
    }

    public static CandidateMetadata fromMap(Map<String, String> values) {
        if (values == null || values.isEmpty()) {
            return generic();
        }
        LinkedHashMap<String, String> attributes = new LinkedHashMap<>();
        for (Map.Entry<String, String> entry : values.entrySet()) {
            if (!RESERVED_KEYS.contains(entry.getKey())) {
                attributes.put(entry.getKey(), entry.getValue());
            }
        }
        return new CandidateMetadata(
                values.get("candidateSpaceId"),
                values.get("candidateSpaceVersion"),
                values.get("parameterFamily"),
                values.get("parameterVariant"),
                values.get("graphAutotuneMode"),
                parseBoolean(values.get("runtimeFrozen"), false),
                parseBoolean(values.get("graphPolicyMutated"), false),
                parseBoolean(values.get("productionEligible"), true),
                attributes
        );
    }

    public Map<String, String> toMap() {
        LinkedHashMap<String, String> out = new LinkedHashMap<>();
        out.put("candidateSpaceId", candidateSpaceId);
        out.put("candidateSpaceVersion", candidateSpaceVersion);
        out.put("parameterFamily", parameterFamily);
        out.put("parameterVariant", parameterVariant);
        out.put("graphAutotuneMode", graphAutotuneMode);
        out.put("runtimeFrozen", Boolean.toString(runtimeFrozen));
        out.put("graphPolicyMutated", Boolean.toString(graphPolicyMutated));
        out.put("productionEligible", Boolean.toString(productionEligible));
        out.putAll(attributes);
        return Map.copyOf(out);
    }

    public CandidateMetadata withAttribute(String key, String value) {
        LinkedHashMap<String, String> next = new LinkedHashMap<>(attributes);
        next.put(normalize(key, "attribute"), normalize(value, ""));
        return new CandidateMetadata(
                candidateSpaceId,
                candidateSpaceVersion,
                parameterFamily,
                parameterVariant,
                graphAutotuneMode,
                runtimeFrozen,
                graphPolicyMutated,
                productionEligible,
                next
        );
    }

    private static String normalize(String value, String fallback) {
        return value == null || value.isBlank() ? fallback : value;
    }

    private static boolean parseBoolean(String value, boolean fallback) {
        if (value == null || value.isBlank()) {
            return fallback;
        }
        return Boolean.parseBoolean(value);
    }

    private static final Set<String> RESERVED_KEYS = Set.of(
            "candidateSpaceId",
            "candidateSpaceVersion",
            "parameterFamily",
            "parameterVariant",
            "graphAutotuneMode",
            "runtimeFrozen",
            "graphPolicyMutated",
            "productionEligible"
    );
}

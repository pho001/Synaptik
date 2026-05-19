package graph.execution;

import java.util.LinkedHashMap;

interface BackendRunTraceContributor {
    void contribute(BackendRunTraceContext context, LinkedHashMap<String, Object> attrs);
}

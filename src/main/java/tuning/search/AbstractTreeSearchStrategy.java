package tuning.search;

import tuning.candidate.Candidate;
import tuning.candidate.ExecutableProfileFingerprint;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

abstract class AbstractTreeSearchStrategy implements SearchStrategy, SearchTreeInspectable {
    protected final SearchStrategy seedStrategy;
    protected final Map<String, SearchTreeNode> nodesByFingerprint = new LinkedHashMap<>();
    protected List<String> frontierFingerprints = List.of();

    protected AbstractTreeSearchStrategy(SearchStrategy seedStrategy) {
        this.seedStrategy = java.util.Objects.requireNonNull(seedStrategy, "seedStrategy cannot be null");
    }

    @Override
    public SearchResult search(SearchContext context) {
        SearchResult seed = seedStrategy.search(context);
        initializeTree(seed.selectedCandidates());
        return seed;
    }

    @Override
    public boolean supportsRefinement() {
        return true;
    }

    @Override
    public SearchTreeSnapshot snapshot() {
        return new SearchTreeSnapshot(List.copyOf(nodesByFingerprint.values()), frontierFingerprints);
    }

    @Override
    public SearchTreeReport report() {
        return SearchTreeInspectable.super.report();
    }

    protected void initializeTree(List<Candidate> seeds) {
        nodesByFingerprint.clear();
        List<String> frontier = new ArrayList<>(seeds == null ? 0 : seeds.size());
        if (seeds != null) {
            for (Candidate candidate : seeds) {
                String fp = ExecutableProfileFingerprint.of(candidate);
                frontier.add(fp);
                nodesByFingerprint.put(fp, new SearchTreeNode(
                        fp,
                        candidate.name(),
                        null,
                        0,
                        0
                ));
            }
        }
        frontierFingerprints = List.copyOf(frontier);
    }

    protected void registerChild(Candidate child, String parentFingerprint, int round) {
        String fp = ExecutableProfileFingerprint.of(child);
        SearchTreeNode parent = nodesByFingerprint.get(parentFingerprint);
        nodesByFingerprint.putIfAbsent(fp, new SearchTreeNode(
                fp,
                child.name(),
                parentFingerprint,
                parent == null ? round : parent.depth() + 1,
                round
        ));
    }
}

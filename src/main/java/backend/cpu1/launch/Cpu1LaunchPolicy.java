package backend.cpu1.launch;

/**
 * Launch policy for a prepared cpu1 kernel.
 */
public interface Cpu1LaunchPolicy {
    /**
     * Runs a prepared range task over its logical element range.
     *
     * @param elementCount logical element count
     * @param task concrete range task
     */
    void launch(int elementCount, Cpu1RangeTask task);
}

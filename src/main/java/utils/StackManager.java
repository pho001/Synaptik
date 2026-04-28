package utils;

/**
 * Internal bytecode-generation helper that tracks simulated operand-stack depth.
 */
public class StackManager {
    int stackDepth = 0;

    /**
     * Creates a stack-depth tracker with zero depth.
     */
    public StackManager() {
        stackDepth = 0;
    }

    /**
     * Returns the currently tracked stack depth.
     *
     * @return simulated operand-stack depth
     */
    public int getStackDepth() {
        return stackDepth;
    }

    /**
     * Replaces the currently tracked stack depth.
     *
     * @param stackDepth simulated operand-stack depth
     */
    public void setStackDepth(int stackDepth) {
        this.stackDepth = stackDepth;
    }

    /**
     * Records one value being pushed onto the simulated operand stack.
     */
    public void push() {
        stackDepth++;
    }

    /**
     * Records one value being popped from the simulated operand stack.
     */
    public void pop() {
        stackDepth--;
    }

}

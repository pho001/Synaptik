package utils;

public class StackManager {
    int stackDepth = 0;

    public StackManager() {
        stackDepth = 0;
    }

    public int getStackDepth() {
        return stackDepth;
    }
    public void setStackDepth(int stackDepth) {
        this.stackDepth = stackDepth;
    }

    public void push() {
        stackDepth++;
    }

    public void pop() {
        stackDepth--;
    }

}

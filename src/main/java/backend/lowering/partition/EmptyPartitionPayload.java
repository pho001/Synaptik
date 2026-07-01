package backend.lowering.partition;

public record EmptyPartitionPayload() implements PartitionBackendPayload {
    public static final EmptyPartitionPayload INSTANCE = new EmptyPartitionPayload();
}

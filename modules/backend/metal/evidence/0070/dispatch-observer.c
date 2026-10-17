#include <stdint.h>
#include <string.h>

#define SYNAPTIK_OBSERVER_EXPORT __attribute__((visibility("default")))
#define SYNAPTIK_OBSERVER_CAPACITY 256U

typedef struct {
    uint32_t step_ordinal;
    uint32_t step_kind;
    uint8_t manifest_digest[32];
    uint64_t grid_width;
    uint64_t grid_height;
    uint64_t grid_depth;
    uint64_t threadgroup_width;
    uint64_t metadata_bytes;
    uint64_t element_count;
    uint64_t point_grid_width;
    uint64_t point_grid_height;
    uint32_t scalar;
    uint32_t reserved;
} SynaptikDispatchRecord;

static SynaptikDispatchRecord records[SYNAPTIK_OBSERVER_CAPACITY];
static uint32_t record_count;

SYNAPTIK_OBSERVER_EXPORT void synaptik_metal_test_observe_dispatch(
        uint32_t step_ordinal,
        uint32_t step_kind,
        const uint8_t manifest_digest[32],
        uint64_t grid_width,
        uint64_t grid_height,
        uint64_t grid_depth,
        uint64_t threadgroup_width,
        uint64_t metadata_bytes,
        uint64_t element_count,
        uint64_t point_grid_width,
        uint64_t point_grid_height,
        uint32_t scalar,
        uint32_t reserved) {
    if (record_count >= SYNAPTIK_OBSERVER_CAPACITY || manifest_digest == NULL) return;
    SynaptikDispatchRecord *record = &records[record_count++];
    record->step_ordinal = step_ordinal;
    record->step_kind = step_kind;
    memcpy(record->manifest_digest, manifest_digest, sizeof(record->manifest_digest));
    record->grid_width = grid_width;
    record->grid_height = grid_height;
    record->grid_depth = grid_depth;
    record->threadgroup_width = threadgroup_width;
    record->metadata_bytes = metadata_bytes;
    record->element_count = element_count;
    record->point_grid_width = point_grid_width;
    record->point_grid_height = point_grid_height;
    record->scalar = scalar;
    record->reserved = reserved;
}

SYNAPTIK_OBSERVER_EXPORT void synaptik_metal_test_observer_reset(void) {
    memset(records, 0, sizeof(records));
    record_count = 0U;
}

SYNAPTIK_OBSERVER_EXPORT uint32_t synaptik_metal_test_observer_count(void) {
    return record_count;
}

SYNAPTIK_OBSERVER_EXPORT uint64_t synaptik_metal_test_observer_field(
        uint32_t record_index, uint32_t field) {
    if (record_index >= record_count) return UINT64_MAX;
    const SynaptikDispatchRecord *record = &records[record_index];
    switch (field) {
        case 0U: return record->step_ordinal;
        case 1U: return record->step_kind;
        case 2U: return record->grid_width;
        case 3U: return record->grid_height;
        case 4U: return record->grid_depth;
        case 5U: return record->threadgroup_width;
        case 6U: return record->metadata_bytes;
        case 7U: return record->element_count;
        case 8U: return record->point_grid_width;
        case 9U: return record->point_grid_height;
        case 10U: return record->scalar;
        case 11U: return record->reserved;
        default: return UINT64_MAX;
    }
}

SYNAPTIK_OBSERVER_EXPORT uint32_t synaptik_metal_test_observer_digest_byte(
        uint32_t record_index, uint32_t byte_index) {
    if (record_index >= record_count || byte_index >= 32U) return UINT32_MAX;
    return records[record_index].manifest_digest[byte_index];
}

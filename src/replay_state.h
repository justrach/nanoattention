// Optional CPU-encoded indirect replay. Included only by the headless bridge.
#pragma once
#include "replay_limits.h"
typedef struct {
    id<MTLComputePipelineState> pipeline; // borrowed from context pipeline cache
    void *buffers[8];
    size_t count, threads, group;
    int valid;
} FMReplaySlot;
typedef struct {
    id<MTLIndirectCommandBuffer> commands;
    id<MTLBuffer> parameters;
    // Retain all indirectly bound buffers until explicit model teardown. The
    // model clears these BEFORE releasing/unmapping any no-copy checkpoint.
    NSMutableArray *resources;
    NSMutableSet *seen;
    NSMutableSet *read_only; // explicit immutable checkpoint residency declarations
    FMReplaySlot slots[FM_REPLAY_COMMANDS];
    size_t cursor;
    int active;
} FMReplay;

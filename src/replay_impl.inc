// Included after FMContext/fail and the encoder_for forward declaration.
// No asynchronous submissions: the ordinary finish still waits for completion.
static void replay_destroy(FMReplay *r) {
    if (!r) return;
    [r->commands resetWithRange:NSMakeRange(0, FM_REPLAY_COMMANDS)];
    [r->commands release]; [r->parameters release];
    [r->resources release]; [r->seen release]; [r->read_only release]; free(r);
}

int fm_replay_prepare(void *opaque) {
    FMContext *c = opaque;
    if (c->profile || c->command || c->encoder) {
        fail(@"Indirect replay requires an idle, unprofiled context"); return 0;
    }
    if (c->replay) return 1;
    if (c->pipelines.count != 0) {
        fail(@"Prepare indirect replay before compiling any pipelines"); return 0;
    }
    FMReplay *r = calloc(1, sizeof(*r));
    if (!r) { fail(@"Cannot allocate replay state"); return 0; }
    MTLIndirectCommandBufferDescriptor *d = [[MTLIndirectCommandBufferDescriptor alloc] init];
    d.commandTypes = MTLIndirectCommandTypeConcurrentDispatchThreads;
    d.inheritPipelineState = NO;
    d.inheritBuffers = NO;
    d.maxKernelBufferBindCount = 9;
    r->commands = [c->device newIndirectCommandBufferWithDescriptor:d
        maxCommandCount:FM_REPLAY_COMMANDS options:MTLResourceStorageModeShared];
    [d release];
    r->parameters = [c->device newBufferWithLength:FM_REPLAY_COMMANDS * FM_REPLAY_PARAM_STRIDE
        options:MTLResourceStorageModeShared];
    r->resources = [[NSMutableArray alloc] init];
    r->seen = [[NSMutableSet alloc] init];
    r->read_only = [[NSMutableSet alloc] init];
    if (!r->commands || !r->parameters) {
        replay_destroy(r); fail(@"Device cannot allocate indirect replay buffers"); return 0;
    }
    c->replay = r;
    return 1;
}

int fm_replay_enable(void *opaque, int enabled) {
    FMContext *c = opaque;
    if (!c->replay || c->profile || c->command || c->encoder || c->replay->cursor) {
        fail(@"Replay mode changes require a prepared, idle context"); return 0;
    }
    c->replay->active = enabled != 0;
    return 1;
}

int fm_replay_clear(void *opaque) {
    FMContext *c = opaque;
    if (!c->replay) return 1;
    if (c->command || c->encoder || c->replay->cursor) {
        fail(@"Clear replay only after finishing the current batch"); return 0;
    }
    FMReplay *r = c->replay;
    [r->commands resetWithRange:NSMakeRange(0, FM_REPLAY_COMMANDS)];
    memset(r->slots, 0, sizeof(r->slots));
    [r->resources removeAllObjects]; [r->seen removeAllObjects]; [r->read_only removeAllObjects];
    r->active = 0;
    return 1;
}

static int replay_record(FMContext *c, id<MTLComputePipelineState> pipeline,
        void *const *buffers, size_t count, const void *params, size_t param_size,
        size_t threads, size_t group) {
    FMReplay *r = c->replay;
    if (r->cursor >= FM_REPLAY_COMMANDS || count > 8 || param_size > FM_REPLAY_PARAM_STRIDE ||
        threads == 0 || group == 0 || !pipeline.supportIndirectCommandBuffers) {
        fail(@"Invalid or unsupported indirect replay dispatch"); return 0;
    }
    size_t index = r->cursor;
    FMReplaySlot *slot = &r->slots[index];
    unsigned char *destination = (unsigned char *)r->parameters.contents + index * FM_REPLAY_PARAM_STRIDE;
    memset(destination, 0, FM_REPLAY_PARAM_STRIDE);
    memcpy(destination, params, param_size);
    if (!slot->valid || slot->pipeline != pipeline || slot->count != count ||
        slot->threads != threads || slot->group != group ||
        memcmp(slot->buffers, buffers, count * sizeof(void *)) != 0) {
        id<MTLIndirectComputeCommand> command = [r->commands indirectComputeCommandAtIndex:index];
        [command reset];
        [command setComputePipelineState:pipeline];
        for (size_t i = 0; i < count; ++i) {
            id<MTLBuffer> buffer = (id<MTLBuffer>)buffers[i];
            if (!buffer) { fail(@"Nil replay buffer"); return 0; }
            [command setKernelBuffer:buffer offset:0 atIndex:i];
            NSValue *key = [NSValue valueWithPointer:buffer];
            if (![r->seen containsObject:key]) {
                [r->seen addObject:key]; [r->resources addObject:buffer];
            }
        }
        [command setKernelBuffer:r->parameters offset:index * FM_REPLAY_PARAM_STRIDE atIndex:8];
        // Concurrent ICB commands must explicitly preserve every dependency.
        [command setBarrier];
        [command concurrentDispatchThreads:MTLSizeMake(threads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
        slot->pipeline = pipeline; slot->count = count;
        slot->threads = threads; slot->group = group;
        memcpy(slot->buffers, buffers, count * sizeof(void *));
        slot->valid = 1;
    }
    if (c->has_dispatch) ++c->barriers;
    c->has_dispatch = 1;
    ++c->dispatches; ++r->cursor;
    return 1;
}

static int replay_encode(FMContext *c) {
    FMReplay *r = c->replay;
    if (!r || !r->active || !r->cursor) return 1;
    id<MTLComputeCommandEncoder> encoder = encoder_for(c, nil, 0);
    if (!encoder) { r->cursor = 0; return 0; }
    [encoder useResource:r->commands usage:MTLResourceUsageRead];
    [encoder useResource:r->parameters usage:MTLResourceUsageRead];
    // Conservative explicit residency for buffers bound indirectly by the ICB.
    // GPU-address checkpoint subresources retain their existing declarations.
    for (id<MTLResource> resource in r->resources) {
        MTLResourceUsage usage = [r->read_only containsObject:[NSValue valueWithPointer:resource]] ?
            MTLResourceUsageRead : (MTLResourceUsageRead | MTLResourceUsageWrite);
        [encoder useResource:resource usage:usage];
    }
    [encoder executeCommandsInBuffer:r->commands withRange:NSMakeRange(0, r->cursor)];
    r->cursor = 0;
    return 1;
}

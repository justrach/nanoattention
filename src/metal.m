// Narrow, headless Objective-C ABI. Model scheduling and checkpoint logic live in Zig.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mach/mach_time.h>

// Opt-in diagnostics only: FLINT_PROFILE=1. These timings/throughputs are NOT
// benchmarks. Auto selection prefers dispatch counters, then encoder-boundary
// counters (one encoder/dispatch, same command buffer), then command GPU times
// (one synchronous command/dispatch). FLINT_PROFILE_METHOD=dispatch|stage|command
// forces a method for diagnostics; unsupported forced counters fail cleanly.
enum { FM_PROFILE_SAMPLES = 2048 };
typedef enum { FMProfileDispatch, FMProfileStage, FMProfileCommand } FMProfileMethod;
typedef struct {
    id<MTLCounterSet> counter_set;
    NSMutableArray *sample_buffers;
    NSMutableArray *names;
    NSMutableDictionary *totals;
    mach_timebase_info_data_t timebase;
    FMProfileMethod method;
    // Retained only through the logical batch (fm_finish), including across the
    // synchronous per-dispatch commands in the last-resort timing mode.
    NSMutableArray *resident_resources;
    NSMutableDictionary *resident_usage;
    uint64_t invalid_dispatches, failed_dispatches, profiled_submissions;
} FMProfile;

enum { FM_PIPELINE_CACHE_SIZE = 128 }; // Power of two; bounded open-addressed cache.
typedef struct {
    char *name;
    uint32_t hash;
    // Borrowed from pipelines, which owns every cached pipeline until fm_close.
    id<MTLComputePipelineState> pipeline;
} FMPipelineCacheEntry;

typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    NSMutableDictionary *pipelines;
    FMPipelineCacheEntry pipeline_cache[FM_PIPELINE_CACHE_SIZE];
    id<MTLCommandBuffer> command;
    id<MTLComputeCommandEncoder> encoder;
    uint64_t submissions, dispatches, barriers;
    double gpu_seconds;
    int has_dispatch;
    NSAutoreleasePool *pool;
    NSAutoreleasePool *batch_pool;
    FMProfile *profile;
} FMContext;
static _Thread_local char fm_error_text[2048];
static void fail(NSString *message) {
    snprintf(fm_error_text, sizeof(fm_error_text), "%s", message.UTF8String ?: "Metal failure");
}
const char *fm_error(void) { return fm_error_text; }

static void profile_destroy(FMProfile *p) {
    if (!p) return;
    [p->counter_set release]; [p->sample_buffers release];
    [p->names release]; [p->totals release];
    [p->resident_resources release]; [p->resident_usage release]; free(p);
}

static int profile_add_buffer(FMContext *c) {
    FMProfile *p = c->profile;
    MTLCounterSampleBufferDescriptor *d = [[MTLCounterSampleBufferDescriptor alloc] init];
    d.counterSet = p->counter_set;
    d.storageMode = MTLStorageModeShared;
    d.sampleCount = FM_PROFILE_SAMPLES;
    d.label = @"Flint per-dispatch timestamps";
    NSError *error = nil;
    id<MTLCounterSampleBuffer> b = [c->device newCounterSampleBufferWithDescriptor:d error:&error];
    [d release];
    if (!b) {
        fail(error.localizedDescription ?: @"FLINT_PROFILE: cannot allocate timestamp samples");
        return 0;
    }
    [p->sample_buffers addObject:b]; [b release];
    return 1;
}

static int profile_open(FMContext *c) {
    const char *requested = getenv("FLINT_PROFILE_METHOD");
    int automatic = !requested || strcmp(requested, "auto") == 0;
    int force_dispatch = requested && strcmp(requested, "dispatch") == 0;
    int force_stage = requested && strcmp(requested, "stage") == 0;
    int force_command = requested && strcmp(requested, "command") == 0;
    if (!automatic && !force_dispatch && !force_stage && !force_command) {
        fail(@"FLINT_PROFILE_METHOD must be auto, dispatch, stage, or command");
        return 0;
    }
    BOOL dispatch_supported = NO, stage_supported = NO;
    id<MTLCounterSet> timestamp_set = nil;
    if (@available(macOS 11.0, *)) {
        dispatch_supported = [c->device supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary];
        stage_supported = [c->device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary];
        for (id<MTLCounterSet> set in c->device.counterSets) {
            if (![set.name isEqualToString:MTLCommonCounterSetTimestamp]) continue;
            for (id<MTLCounter> counter in set.counters) {
                if ([counter.name isEqualToString:MTLCommonCounterTimestamp]) timestamp_set = set;
            }
        }
    }
    FMProfileMethod method = FMProfileCommand;
    if ((automatic || force_dispatch) && dispatch_supported && timestamp_set) method = FMProfileDispatch;
    else if ((automatic || force_stage) && stage_supported && timestamp_set) method = FMProfileStage;
    else if (force_dispatch || force_stage) {
        fail(@"FLINT_PROFILE: requested timestamp counter method is unsupported; use auto or command, or unset FLINT_PROFILE");
        return 0;
    }
    if (method == FMProfileCommand) {
        if (@available(macOS 10.15, *)) { /* GPUStartTime/GPUEndTime available. */ }
        else { fail(@"FLINT_PROFILE: no supported GPU timestamp method"); return 0; }
    }
    FMProfile *p = calloc(1, sizeof(*p));
    if (!p) { fail(@"FLINT_PROFILE: cannot allocate profiler"); return 0; }
    c->profile = p;
    p->method = method;
    p->counter_set = method == FMProfileCommand ? nil : [timestamp_set retain];
    p->sample_buffers = [[NSMutableArray alloc] init];
    p->names = [[NSMutableArray alloc] init];
    p->totals = [[NSMutableDictionary alloc] init];
    if (method != FMProfileDispatch) {
        p->resident_resources = [[NSMutableArray alloc] init];
        p->resident_usage = [[NSMutableDictionary alloc] init];
    }
    if (method != FMProfileCommand &&
        (mach_timebase_info(&p->timebase) != KERN_SUCCESS || !p->timebase.denom)) {
        fail(@"FLINT_PROFILE: cannot calibrate timestamp clock");
        return 0;
    }
    return method == FMProfileCommand || profile_add_buffer(c);
}

static void profile_record(FMProfile *p, NSString *name, double us) {
    NSMutableDictionary *row = p->totals[name];
    if (!row) {
        row = [NSMutableDictionary dictionaryWithDictionary:@{
            @"kernel": name, @"count": @0, @"total_us": @0.0,
            @"min_us": @(us), @"max_us": @(us)
        }];
        p->totals[name] = row;
    }
    row[@"count"] = @([row[@"count"] unsignedLongLongValue] + 1);
    row[@"total_us"] = @([row[@"total_us"] doubleValue] + us);
    row[@"min_us"] = @(MIN([row[@"min_us"] doubleValue], us));
    row[@"max_us"] = @(MAX([row[@"max_us"] doubleValue], us));
}

static void profile_resolve(FMContext *c, int ok, double ns_per_tick) {
    FMProfile *p = c->profile;
    NSUInteger count = p->names.count;
    if (!ok) { p->failed_dispatches += count; [p->names removeAllObjects]; return; }
    ++p->profiled_submissions;
    for (NSUInteger base = 0; base < count; base += FM_PROFILE_SAMPLES / 2) {
        NSUInteger n = MIN(count - base, FM_PROFILE_SAMPLES / 2);
        id<MTLCounterSampleBuffer> b = p->sample_buffers[base / (FM_PROFILE_SAMPLES / 2)];
        // Shared counter samples are CPU-resolved only after command completion.
        NSData *data = [b resolveCounterRange:NSMakeRange(0, n * 2)];
        if (!(ns_per_tick > 0) || data.length != n * 2 * sizeof(MTLCounterResultTimestamp)) {
            p->invalid_dispatches += n;
            continue;
        }
        const MTLCounterResultTimestamp *samples = data.bytes;
        for (NSUInteger i = 0; i < n; ++i) {
            uint64_t begin = samples[i * 2].timestamp, end = samples[i * 2 + 1].timestamp;
            if (begin == MTLCounterErrorValue || end == MTLCounterErrorValue || end < begin) {
                ++p->invalid_dispatches;
                continue;
            }
            double us = (double)(end - begin) * ns_per_tick / 1000.0;
            profile_record(p, p->names[base + i], us);
        }
    }
    [p->names removeAllObjects];
}

static void profile_report(FMContext *c) {
    FMProfile *p = c->profile;
    NSArray *rows = [[p->totals allValues] sortedArrayUsingDescriptors:@[
        [NSSortDescriptor sortDescriptorWithKey:@"total_us" ascending:NO],
        [NSSortDescriptor sortDescriptorWithKey:@"kernel" ascending:YES]
    ]];
    uint64_t count = 0;
    double total_us = 0;
    for (NSMutableDictionary *row in rows) {
        uint64_t n = [row[@"count"] unsignedLongLongValue];
        row[@"mean_us"] = @([row[@"total_us"] doubleValue] / n);
        count += n;
        total_us += [row[@"total_us"] doubleValue];
    }
    NSDictionary *report = @{
        @"type": @"flint_kernel_profile", @"version": @1,
        @"profiling_overhead": @YES, @"benchmark": @NO,
        @"timestamp_method": p->method == FMProfileDispatch ? @"dispatch_boundary_counters" :
            (p->method == FMProfileStage ? @"encoder_boundary_counters" : @"command_gpu_start_end"),
        @"timestamp_barriers_per_dispatch": @(p->method == FMProfileDispatch ? 2 : 0),
        @"encoder_splitting": @(p->method != FMProfileDispatch),
        @"command_buffer_splitting": @(p->method == FMProfileCommand),
        @"cpu_wait_per_dispatch": @(p->method == FMProfileCommand),
        @"timing_scope": p->method == FMProfileDispatch ? @"dispatch plus sampling barriers" :
            (p->method == FMProfileStage ? @"whole compute encoder, including residency and bindings" :
             @"whole command buffer; synchronous CPU submission/wait overhead perturbs execution"),
        @"clock": p->method == FMProfileCommand ? @"Metal command GPUStartTime/GPUEndTime seconds" :
            @"GPU ticks calibrated against CPU nanosecond timestamps per submission",
        @"profiled_submissions": @(p->profiled_submissions),
        @"dispatches": @(count), @"total_us": @(total_us),
        @"encoded_dispatches": @(c->dispatches),
        @"command_gpu_us": @(c->gpu_seconds * 1000000.0),
        @"invalid_dispatches": @(p->invalid_dispatches),
        @"failed_dispatches": @(p->failed_dispatches), @"kernels": rows
    };
    NSError *error = nil;
    NSData *json = [NSJSONSerialization dataWithJSONObject:report options:0 error:&error];
    if (json) { fwrite(json.bytes, 1, json.length, stderr); fputc('\n', stderr); }
    else fprintf(stderr, "FLINT_PROFILE: could not serialize timestamp report\n");
}

int fm_is_m3_ultra(void *context) {
    FMContext *c = context;
    return [c->device.name isEqualToString:@"Apple M3 Ultra"];
}

void *fm_open(const char *source) {
    FMContext *c = calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->pool = [[NSAutoreleasePool alloc] init];
    c->device = MTLCreateSystemDefaultDevice();
    if (!c->device || !c->device.hasUnifiedMemory) {
        fail(@"A unified-memory Apple GPU is required");
        goto bad;
    }
    c->queue = [c->device newCommandQueue];
    NSError *error = nil;
    MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
    // Precise arithmetic makes the baseline easier to compare with a reference.
    options.mathMode = MTLMathModeSafe;
    c->library = [c->device newLibraryWithSource:[NSString stringWithUTF8String:source]
                                       options:options error:&error];
    [options release];
    if (!c->library || !c->queue) {
        fail(error.localizedDescription ?: @"Cannot create Metal library/queue");
        goto bad;
    }
    const char *profile = getenv("FLINT_PROFILE");
    if (profile && strcmp(profile, "1") == 0 && !profile_open(c)) goto bad;
    c->pipelines = [[NSMutableDictionary alloc] init];
    return c;
bad:
    profile_destroy(c->profile);
    [c->library release]; [c->queue release]; [c->device release];
    [c->pool drain]; free(c); return NULL;
}

static id<MTLComputeCommandEncoder> encoder_for(FMContext *c,
        id<MTLCounterSampleBuffer> samples, NSUInteger sample_index) {
    if (!c->batch_pool) c->batch_pool = [[NSAutoreleasePool alloc] init];
    if (!c->command) c->command = [[c->queue commandBuffer] retain];
    if (!c->encoder) {
        if (c->profile && c->profile->method == FMProfileStage) {
            if (@available(macOS 11.0, *)) {
                MTLComputePassDescriptor *d = [MTLComputePassDescriptor computePassDescriptor];
                d.dispatchType = MTLDispatchTypeSerial;
                MTLComputePassSampleBufferAttachmentDescriptor *a = d.sampleBufferAttachments[0];
                a.sampleBuffer = samples;
                a.startOfEncoderSampleIndex = sample_index;
                a.endOfEncoderSampleIndex = sample_index + 1;
                c->encoder = [[c->command computeCommandEncoderWithDescriptor:d] retain];
            }
        } else c->encoder = [[c->command computeCommandEncoder] retain];
        // GPU-address resources are not implicitly tracked by direct bindings.
        // Every split encoder needs the entire logical batch's declarations.
        if (c->profile) {
            for (id<MTLResource> resource in c->profile->resident_resources) {
                NSValue *key = [NSValue valueWithPointer:resource];
                MTLResourceUsage usage = [c->profile->resident_usage[key] unsignedIntegerValue];
                [c->encoder useResource:resource usage:usage];
            }
        }
    }
    if (!c->encoder) fail(@"Cannot create compute encoder");
    return c->encoder;
}
static int resident(FMContext *c, id<MTLBuffer> buffer, MTLResourceUsage usage) {
    if (c->profile && c->profile->method != FMProfileDispatch) {
        if (!buffer) { fail(@"Cannot declare a nil resident resource"); return 0; }
        NSValue *key = [NSValue valueWithPointer:buffer];
        NSNumber *previous = c->profile->resident_usage[key];
        if (!previous) [c->profile->resident_resources addObject:buffer];
        usage |= previous.unsignedIntegerValue;
        c->profile->resident_usage[key] = @(usage);
        // Residency calls do not create an empty sampled encoder. Dispatch will
        // declare the union of read/write usage on its newly created encoder.
        if (c->encoder) [c->encoder useResource:buffer usage:usage];
        return 1;
    }
    id<MTLComputeCommandEncoder> encoder = encoder_for(c, nil, 0);
    if (!encoder) return 0;
    [encoder useResource:buffer usage:usage];
    return 1;
}
uint64_t fm_address(void *buffer) { return [(id<MTLBuffer>)buffer gpuAddress]; }
int fm_resident(void *opaque, void *buffer) {
    FMContext *c = opaque;
    return resident(c, (id<MTLBuffer>)buffer, MTLResourceUsageRead);
}
int fm_resident_write(void *opaque, void *buffer) {
    FMContext *c = opaque;
    return resident(c, (id<MTLBuffer>)buffer, MTLResourceUsageRead | MTLResourceUsageWrite);
}
typedef struct { uint64_t submissions, dispatches, barriers; double gpu_seconds; } FMStats;
void fm_stats(void *opaque, FMStats *out) {
    FMContext *c = opaque;
    *out = (FMStats){c->submissions, c->dispatches, c->barriers, c->gpu_seconds};
}

static int submit_command(FMContext *c) {
    if (!c->command) {
        [c->batch_pool drain]; c->batch_pool = nil;
        return 1;
    }
    [c->encoder endEncoding]; [c->encoder release]; c->encoder = nil;
    c->has_dispatch = 0;
    ++c->submissions;
    MTLTimestamp cpu_begin = 0, gpu_begin = 0;
    if (c->profile && c->profile->method != FMProfileCommand)
        [c->device sampleTimestamps:&cpu_begin gpuTimestamp:&gpu_begin];
    [c->command commit];
    [c->command waitUntilCompleted];
    int ok = c->command.status == MTLCommandBufferStatusCompleted;
    if (!ok) fail(c->command.error.localizedDescription ?: @"Metal command failed");
    double begin = c->command.GPUStartTime, end = c->command.GPUEndTime;
    if (ok && end >= begin) c->gpu_seconds += end - begin;
    if (c->profile && c->profile->method == FMProfileCommand) {
        FMProfile *p = c->profile;
        if (!ok) p->failed_dispatches += p->names.count;
        else {
            ++p->profiled_submissions;
            if (p->names.count == 1 && begin > 0 && end >= begin)
                profile_record(p, p->names[0], (end - begin) * 1000000.0);
            else p->invalid_dispatches += p->names.count;
        }
        [p->names removeAllObjects];
    } else if (c->profile) {
        MTLTimestamp cpu_end = 0, gpu_end = 0;
        [c->device sampleTimestamps:&cpu_end gpuTimestamp:&gpu_end];
        // sampleTimestamps returns CPU nanoseconds, not raw mach absolute ticks.
        // Calibrate GPU counter ticks against that clock once per batch.
        double ns_per_tick = 0;
        if (cpu_end > cpu_begin && gpu_end > gpu_begin)
            ns_per_tick = (double)(cpu_end - cpu_begin) / (double)(gpu_end - gpu_begin);
        profile_resolve(c, ok, ns_per_tick);
    }
    [c->command release]; c->command = nil;
    [c->batch_pool drain]; c->batch_pool = nil;
    return ok;
}
int fm_finish(void *opaque) {
    FMContext *c = opaque;
    int ok = submit_command(c);
    // End of the logical command/batch: do not retain no-copy mmap buffers into
    // the next model. Command fallback keeps these only between its dispatches.
    if (c->profile) {
        [c->profile->resident_resources removeAllObjects];
        [c->profile->resident_usage removeAllObjects];
    }
    return ok;
}
void fm_close(void *opaque) {
    FMContext *c = opaque;
    if (!c) return;
    fm_finish(c);
    if (c->profile) { profile_report(c); profile_destroy(c->profile); }
    for (size_t i = 0; i < FM_PIPELINE_CACHE_SIZE; ++i) free(c->pipeline_cache[i].name);
    [c->pipelines release]; [c->library release];
    [c->queue release]; [c->device release];
    [c->pool drain]; free(c);
}
void *fm_alloc(void *opaque, size_t length) {
    FMContext *c = opaque;
    id<MTLBuffer> b = [c->device newBufferWithLength:MAX(length, 4) options:MTLResourceStorageModeShared];
    if (!b) fail(@"Metal buffer allocation failed (size or memory limit)");
    return b;
}
void *fm_wrap(void *opaque, void *bytes, size_t length) {
    FMContext *c = opaque;
    id<MTLBuffer> b = [c->device newBufferWithBytesNoCopy:bytes length:length
                                               options:MTLResourceStorageModeShared deallocator:nil];
    if (!b) fail(@"Cannot wrap checkpoint mapping in a Metal buffer");
    return b;
}
void *fm_contents(void *buffer) { return [(id<MTLBuffer>)buffer contents]; }
void fm_release(void *buffer) { [(id<MTLBuffer>)buffer release]; }

int fm_dispatch(void *opaque, const char *name, void *const *buffers, size_t count,
                const void *params, size_t param_size, size_t threads, size_t group) {
    FMContext *c = opaque;
    if (!c->batch_pool) c->batch_pool = [[NSAutoreleasePool alloc] init];
    uint32_t hash = 2166136261u;
    for (const unsigned char *p = (const unsigned char *)name; *p; ++p)
        hash = (hash ^ *p) * 16777619u;
    FMPipelineCacheEntry *entry = NULL;
    id<MTLComputePipelineState> pipeline = nil;
    for (size_t i = 0; i < FM_PIPELINE_CACHE_SIZE; ++i) {
        FMPipelineCacheEntry *candidate = &c->pipeline_cache[(hash + i) & (FM_PIPELINE_CACHE_SIZE - 1)];
        if (!candidate->name) { entry = candidate; break; }
        if (candidate->hash == hash && strcmp(candidate->name, name) == 0) {
            pipeline = candidate->pipeline;
            break;
        }
    }
    NSString *key = nil;
    if (!pipeline) {
        key = [NSString stringWithUTF8String:name];
        pipeline = c->pipelines[key];
        if (!pipeline) {
            id<MTLFunction> f = [c->library newFunctionWithName:key];
            if (!f) { fail([@"Missing Metal kernel: " stringByAppendingString:key]); return 0; }
            NSError *error = nil;
            pipeline = [c->device newComputePipelineStateWithFunction:f error:&error];
            [f release];
            if (!pipeline) { fail(error.localizedDescription); return 0; }
            c->pipelines[key] = pipeline;
            [pipeline release];
        }
        // Caller names may be temporary. Full cache/allocation failure simply
        // falls back to the owning dictionary without changing dispatch errors.
        if (entry && (entry->name = strdup(name))) {
            entry->hash = hash;
            entry->pipeline = pipeline;
        }
    }
    if (group > pipeline.maxTotalThreadsPerThreadgroup) { fail(@"Invalid threadgroup size"); return 0; }
    id<MTLCounterSampleBuffer> samples = nil;
    NSUInteger sample_index = 0;
    if (c->profile && c->profile->method != FMProfileCommand) {
        NSUInteger dispatch_index = c->profile->names.count;
        NSUInteger buffer_index = dispatch_index / (FM_PROFILE_SAMPLES / 2);
        if (buffer_index == c->profile->sample_buffers.count && !profile_add_buffer(c)) return 0;
        samples = c->profile->sample_buffers[buffer_index];
        sample_index = (dispatch_index % (FM_PROFILE_SAMPLES / 2)) * 2;
    }
    id<MTLComputeCommandEncoder> encoder = encoder_for(c, samples, sample_index);
    if (!encoder) return 0;
    if (c->profile) [c->profile->names addObject:key ?: [NSString stringWithUTF8String:name]];
    // Explicit buffer hazards preserve inter-dispatch dependencies within one encoder.
    if (c->has_dispatch) { [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers]; ++c->barriers; }
    [encoder setComputePipelineState:pipeline];
    for (size_t i = 0; i < count; ++i) [encoder setBuffer:(id<MTLBuffer>)buffers[i] offset:0 atIndex:i];
    [encoder setBytes:params length:param_size atIndex:8];
    if (samples && c->profile->method == FMProfileDispatch)
        [encoder sampleCountersInBuffer:samples atSampleIndex:sample_index withBarrier:YES];
    [encoder dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
    if (samples && c->profile->method == FMProfileDispatch)
        [encoder sampleCountersInBuffer:samples atSampleIndex:sample_index + 1 withBarrier:YES];
    c->has_dispatch = 1;
    ++c->dispatches;
    if (c->profile && c->profile->method != FMProfileDispatch) {
        [c->encoder endEncoding]; [c->encoder release]; c->encoder = nil;
        // Serial encoders on one command buffer preserve buffer hazards between
        // dispatches; command fallback additionally waits after every dispatch.
        c->has_dispatch = 0;
        if (c->profile->method == FMProfileCommand) {
            int ok = submit_command(c);
            if (!ok) {
                [c->profile->resident_resources removeAllObjects];
                [c->profile->resident_usage removeAllObjects];
            }
            return ok;
        }
    }
    return 1;
}

void *fm_map(const char *path, size_t *length, size_t *mapped_length) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { snprintf(fm_error_text, sizeof(fm_error_text), "Cannot open checkpoint file"); return NULL; }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { close(fd); return NULL; }
    size_t page = (size_t)getpagesize();
    *length = (size_t)st.st_size;
    if (*length > SIZE_MAX - page + 1) { close(fd); return NULL; }
    *mapped_length = (*length + page - 1) / page * page;
    // Private, writable VM is required for Metal's no-copy shared-memory buffer API.
    // Kernels never write weights; the underlying checkpoint is opened read-only.
    void *p = mmap(NULL, *mapped_length, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) { snprintf(fm_error_text, sizeof(fm_error_text), "Cannot map checkpoint file"); return NULL; }
    return p;
}
void fm_unmap(void *bytes, size_t length) { munmap(bytes, length); }

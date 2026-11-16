//! Greedy-only speculative generation. History drafts never see future answers;
//! every returned token has passed the target and is committed to native KV.
const std = @import("std");
const Model = @import("model.zig").Model;
const sampling = @import("sampling.zig");
const draft = @import("speculative.zig");

pub const Result = struct {
    count: usize = 0,
    eos: bool = false,
    logits: []const f32,
    batches: usize = 0,
    proposed: usize = 0,
    accepted: usize = 0,
};

pub fn generate(model: *Model, a: std.mem.Allocator, history: *std.ArrayList(u32), output: []u32, initial: []const f32) !Result {
    if (model.speculative_tokens > draft.max_draft) return error.InvalidSpeculativeLength;
    if (history.items.len != model.position) return error.HistoryPositionMismatch;
    if (@as(u64, model.position) + output.len > model.context) return error.ContextExhausted;
    try history.ensureUnusedCapacity(a, output.len + 1);
    var result: Result = .{ .logits = initial };
    while (result.count < output.len) {
        const first = try sampling.greedy(result.logits);
        if (model.speculative_tokens < 2 or isEos(model, first)) {
            result.logits = try model.forward(first);
            output[result.count] = first;
            history.appendAssumeCapacity(first);
            result.count += 1;
            result.eos = isEos(model, first);
            if (result.eos) break;
            continue;
        }
        var proposed: [draft.max_draft]u32 = undefined;
        proposed[0] = first;
        // The first input is already the exact target prediction. Match the
        // history suffix INCLUDING it to draft only its following tokens.
        history.appendAssumeCapacity(first);
        const room = @min(@as(usize, model.speculative_tokens), output.len - result.count);
        const n = 1 + draft.propose(history.items, proposed[1..room]);
        history.items.len -= 1;
        if (n == 1) {
            result.logits = try model.forward(first);
            output[result.count] = first;
            history.appendAssumeCapacity(first);
            result.count += 1;
            continue;
        }
        _ = try model.forwardBatch(proposed[0..n]);
        result.batches += 1;
        result.proposed += n;
        var prediction = first;
        var accepted: usize = 0;
        // A proposal is accepted iff it equals the target prediction after
        // exactly the committed prefix. Later rows after rejection are ignored.
        while (accepted < n and proposed[accepted] == prediction) {
            const token = proposed[accepted];
            prediction = try model.batchPrediction(accepted);
            accepted += 1;
            if (isEos(model, token)) {
                result.eos = true;
                break;
            }
        }
        result.logits = try model.finishBatch(accepted);
        @memcpy(output[result.count..][0..accepted], proposed[0..accepted]);
        history.appendSliceAssumeCapacity(proposed[0..accepted]);
        result.count += accepted;
        result.accepted += accepted;
        if (result.eos) break;
        // Correct a rejection, or commit the all-accepted bonus. It has not
        // been fed yet; forward it even at EOS/final budget for session parity.
        if (result.count < output.len) {
            result.logits = try model.forward(prediction);
            output[result.count] = prediction;
            history.appendAssumeCapacity(prediction);
            result.count += 1;
            result.eos = isEos(model, prediction);
            if (result.eos) break;
        }
    }
    return result;
}
fn isEos(model: *const Model, token: u32) bool {
    return std.mem.indexOfScalar(u32, model.cfg.eos_token_id, token) != null;
}

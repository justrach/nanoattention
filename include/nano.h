#ifndef NANO_H
#define NANO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NanoSession NanoSession;

typedef struct NanoConfig {
    uint32_t context;
    uint32_t batch;
    const char *preset;
    int32_t use_tokenizer;
} NanoConfig;

typedef struct NanoStats {
    uint32_t prompt_tokens;
    uint32_t generated_tokens;
    uint32_t position;
    double load_ms;
    double prefill_ms;
    double decode_ms;
    uint64_t dispatches;
} NanoStats;

typedef struct NanoImageInput {
    /* Local path, file://, data:...;base64,..., or http(s):// URL (NULL when using bytes). */
    const char *source;
    /* Encoded image bytes (PNG/JPEG/WebP/HEIC/etc.), NULL when using source. */
    const uint8_t *bytes;
    size_t bytes_len;
} NanoImageInput;

/* Return 0 to continue generation, non-zero to stop early. */
typedef int (*NanoTokenCallback)(
    uint32_t token_id,
    const char *utf8_piece,
    size_t piece_len,
    void *user_data);

const char *nano_version(void);
const char *nano_last_error(void);

NanoSession *nano_open(const char *checkpoint_dir, const NanoConfig *config);
void nano_close(NanoSession *session);

const char *nano_device_name(const NanoSession *session);
const char *nano_model_type(const NanoSession *session);
int nano_has_vision(const NanoSession *session);

int nano_reset(NanoSession *session);

int nano_generate_tokens(
    NanoSession *session,
    const uint32_t *prompt_tokens,
    size_t prompt_len,
    uint32_t max_tokens,
    NanoTokenCallback cb,
    void *user_data,
    NanoStats *out_stats);

int nano_generate_tokens_with_images(
    NanoSession *session,
    const uint32_t *prompt_tokens,
    size_t prompt_len,
    const NanoImageInput *images,
    size_t image_count,
    uint32_t max_image_tokens,
    uint32_t max_tokens,
    NanoTokenCallback cb,
    void *user_data,
    NanoStats *out_stats);

int nano_generate_text(
    NanoSession *session,
    const char *system_prompt,
    const char *user_prompt,
    int32_t raw_prompt,
    uint32_t max_tokens,
    NanoTokenCallback cb,
    void *user_data,
    NanoStats *out_stats);

int nano_generate_multimodal(
    NanoSession *session,
    const char *system_prompt,
    const char *user_prompt,
    const NanoImageInput *images,
    size_t image_count,
    uint32_t max_image_tokens,
    uint32_t max_tokens,
    NanoTokenCallback cb,
    void *user_data,
    NanoStats *out_stats);

#ifdef __cplusplus
}
#endif

#endif /* NANO_H */

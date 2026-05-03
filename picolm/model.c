#include "model.h"
#include "tensor.h"
#include "quant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

/* ---- GGUF metadata value types ---- */
enum {
    GGUF_META_UINT8   = 0,
    GGUF_META_INT8    = 1,
    GGUF_META_UINT16  = 2,
    GGUF_META_INT16   = 3,
    GGUF_META_UINT32  = 4,
    GGUF_META_INT32   = 5,
    GGUF_META_FLOAT32 = 6,
    GGUF_META_BOOL    = 7,
    GGUF_META_STRING  = 8,
    GGUF_META_ARRAY   = 9,
    GGUF_META_UINT64  = 10,
    GGUF_META_INT64   = 11,
    GGUF_META_FLOAT64 = 12,
};

/* ---- Helpers for reading GGUF binary format ---- */

typedef struct {
    const uint8_t *data;
    size_t pos;
    size_t size;
} reader_t;

static uint8_t read_u8(reader_t *r) {
    uint8_t v = r->data[r->pos];
    r->pos += 1;
    return v;
}

static uint16_t read_u16(reader_t *r) {
    uint16_t v;
    memcpy(&v, r->data + r->pos, 2);
    r->pos += 2;
    return v;
}

static uint32_t read_u32(reader_t *r) {
    uint32_t v;
    memcpy(&v, r->data + r->pos, 4);
    r->pos += 4;
    return v;
}

static int32_t read_i32(reader_t *r) {
    int32_t v;
    memcpy(&v, r->data + r->pos, 4);
    r->pos += 4;
    return v;
}

static uint64_t read_u64(reader_t *r) {
    uint64_t v;
    memcpy(&v, r->data + r->pos, 8);
    r->pos += 8;
    return v;
}

static float read_f32(reader_t *r) {
    float v;
    memcpy(&v, r->data + r->pos, 4);
    r->pos += 4;
    return v;
}

typedef struct { const char *str; uint64_t len; } gguf_str_t;

static gguf_str_t read_gguf_string(reader_t *r) {
    gguf_str_t s;
    s.len = read_u64(r);
    s.str = (const char *)(r->data + r->pos);
    r->pos += s.len;
    return s;
}

static int str_eq(gguf_str_t s, const char *lit) {
    size_t n = strlen(lit);
    return s.len == n && memcmp(s.str, lit, n) == 0;
}

static uint64_t skip_meta_value(reader_t *r, uint32_t vtype, int *is_numeric) {
    *is_numeric = 1;
    switch (vtype) {
        case GGUF_META_UINT8:   return read_u8(r);
        case GGUF_META_INT8:    return (uint64_t)(int64_t)(int8_t)read_u8(r);
        case GGUF_META_UINT16:  return read_u16(r);
        case GGUF_META_INT16:   return (uint64_t)(int64_t)(int16_t)read_u16(r);
        case GGUF_META_UINT32:  return read_u32(r);
        case GGUF_META_INT32:   return (uint64_t)(int64_t)read_i32(r);
        case GGUF_META_UINT64:  return read_u64(r);
        case GGUF_META_INT64:   return read_u64(r);
        case GGUF_META_FLOAT32: { read_f32(r); *is_numeric = 0; return 0; }
        case GGUF_META_FLOAT64: { r->pos += 8; *is_numeric = 0; return 0; }
        case GGUF_META_BOOL:    return read_u8(r);
        case GGUF_META_STRING:  { read_gguf_string(r); *is_numeric = 0; return 0; }
        case GGUF_META_ARRAY: {
            *is_numeric = 0;
            uint32_t arr_type = read_u32(r);
            uint64_t arr_len  = read_u64(r);
            int dummy;
            for (uint64_t i = 0; i < arr_len; i++) {
                skip_meta_value(r, arr_type, &dummy);
            }
            return 0;
        }
        default:
            fprintf(stderr, "Unknown GGUF metadata type: %u\n", vtype);
            exit(1);
    }
}

/* ---- mmap abstraction ---- */

static int mmap_file(model_t *m, const char *path) {
#ifdef _WIN32
    HANDLE fh = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (fh == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "Cannot open file: %s\n", path);
        return -1;
    }

    LARGE_INTEGER fsize;
    GetFileSizeEx(fh, &fsize);
    m->mmap_size = (size_t)fsize.QuadPart;

    HANDLE mh = CreateFileMappingA(fh, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mh) {
        fprintf(stderr, "CreateFileMapping failed\n");
        CloseHandle(fh);
        return -1;
    }

    void *addr = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (!addr) {
        fprintf(stderr, "MapViewOfFile failed\n");
        CloseHandle(mh);
        CloseHandle(fh);
        return -1;
    }

    m->mmap_addr  = addr;
    m->file_handle = fh;
    m->map_handle  = mh;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Cannot open file: %s\n", path);
        return -1;
    }

    struct stat st;
    fstat(fd, &st);
    m->mmap_size = (size_t)st.st_size;

    void *addr = mmap(NULL, m->mmap_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
        fprintf(stderr, "mmap failed\n");
        close(fd);
        return -1;
    }
    madvise(addr, m->mmap_size, MADV_SEQUENTIAL);

    m->mmap_addr = addr;
    m->fd = fd;
#endif
    return 0;
}

static void munmap_file(model_t *m) {
    if (!m->mmap_addr) return;
#ifdef _WIN32
    UnmapViewOfFile(m->mmap_addr);
    CloseHandle(m->map_handle);
    CloseHandle(m->file_handle);
#else
    munmap(m->mmap_addr, m->mmap_size);
    close(m->fd);
#endif
    m->mmap_addr = NULL;
}

/* ---- GGUF Parser ---- */

static int parse_gguf(model_t *m, int max_seq_len) {
    reader_t r = { .data = (const uint8_t *)m->mmap_addr, .pos = 0, .size = m->mmap_size };
    model_config_t *cfg = &m->config;

    uint32_t magic = read_u32(&r);
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "Invalid GGUF magic: 0x%08X\n", magic);
        return -1;
    }

    uint32_t version = read_u32(&r);
    if (version < 2 || version > 3) {
        fprintf(stderr, "Unsupported GGUF version: %u\n", version);
        return -1;
    }

    uint64_t n_tensors  = read_u64(&r);
    uint64_t n_metadata = read_u64(&r);

    cfg->alignment = 32;
    cfg->rope_freq_base = 10000.0f;
    cfg->max_seq_len = 2048;
    cfg->weight_type = GGUF_TYPE_F16;
    m->tok_bos_id = 1;
    m->tok_eos_id = 2;
    m->tok_model_type = 0; /* default: SentencePiece */

    for (uint64_t i = 0; i < n_metadata; i++) {
        gguf_str_t key = read_gguf_string(&r);
        uint32_t vtype = read_u32(&r);

        if (str_eq(key, "llama.embedding_length") || str_eq(key, "general.embedding_length") ||
            str_eq(key, "qwen2.embedding_length") || str_eq(key, "qwen3.embedding_length") ||
            str_eq(key, "qwen35.embedding_length") ||
            str_eq(key, "gemma4.embedding_length")) {
            int dummy; cfg->n_embd = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.feed_forward_length") || str_eq(key, "general.feed_forward_length") ||
                   str_eq(key, "qwen2.feed_forward_length") || str_eq(key, "qwen3.feed_forward_length") ||
                   str_eq(key, "qwen35.feed_forward_length") ||
                   str_eq(key, "gemma4.feed_forward_length")) {
            int dummy; cfg->n_ffn = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.attention.head_count") ||
                   str_eq(key, "qwen2.attention.head_count") || str_eq(key, "qwen3.attention.head_count") ||
                   str_eq(key, "qwen35.attention.head_count") ||
                   str_eq(key, "gemma4.attention.head_count")) {
            int dummy; cfg->n_heads = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.attention.head_count_kv") ||
                   str_eq(key, "qwen2.attention.head_count_kv") || str_eq(key, "qwen3.attention.head_count_kv") ||
                   str_eq(key, "qwen35.attention.head_count_kv") ||
                   str_eq(key, "gemma4.attention.head_count_kv")) {
            int dummy; cfg->n_kv_heads = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.block_count") ||
                   str_eq(key, "qwen2.block_count") || str_eq(key, "qwen3.block_count") ||
                   str_eq(key, "qwen35.block_count") ||
                   str_eq(key, "gemma4.block_count")) {
            int dummy; cfg->n_layers = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.context_length") ||
                   str_eq(key, "qwen2.context_length") || str_eq(key, "qwen3.context_length") ||
                   str_eq(key, "qwen35.context_length") ||
                   str_eq(key, "gemma4.context_length")) {
            int dummy; cfg->max_seq_len = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.rope.freq_base") ||
                   str_eq(key, "qwen2.rope.freq_base") || str_eq(key, "qwen3.rope.freq_base") ||
                   str_eq(key, "qwen35.rope.freq_base") ||
                   str_eq(key, "gemma4.rope.freq_base")) {
            if (vtype == GGUF_META_FLOAT32) {
                cfg->rope_freq_base = read_f32(&r);
            } else {
                int dummy; skip_meta_value(&r, vtype, &dummy);
            }
        } else if (str_eq(key, "general.alignment")) {
            int dummy; cfg->alignment = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.vocab_size") ||
                   str_eq(key, "qwen2.vocab_size") || str_eq(key, "qwen3.vocab_size") ||
                   str_eq(key, "qwen35.vocab_size") ||
                   str_eq(key, "gemma4.vocab_size")) {
            int dummy; cfg->vocab_size = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "qwen35.ssm.conv_kernel")) {
            int dummy; cfg->ssm_d_conv = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "qwen35.ssm.inner_size")) {
            int dummy; cfg->ssm_d_inner = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "qwen35.ssm.state_size")) {
            int dummy; cfg->ssm_d_state = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "qwen35.ssm.time_step_rank")) {
            int dummy; cfg->ssm_dt_rank = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "qwen35.ssm.group_count")) {
            int dummy; cfg->ssm_n_group = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "qwen35.attention.full_attention_interval")) {
            int dummy; cfg->full_attn_interval = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "qwen35.attention.layer_norm_rms_epsilon")) {
            if (vtype == GGUF_META_FLOAT32) {
                cfg->norm_rms_eps = read_f32(&r);
            } else {
                int dummy; skip_meta_value(&r, vtype, &dummy);
            }
        } else if (str_eq(key, "tokenizer.ggml.model")) {
            if (vtype == GGUF_META_STRING) {
                gguf_str_t tok_model = read_gguf_string(&r);
                /* "gpt2" tokenizer type means GPT2/tiktoken BPE (used by Qwen family) */
                if (str_eq(tok_model, "gpt2")) {
                    m->tok_model_type = 1;
                }
            } else {
                int dummy; skip_meta_value(&r, vtype, &dummy);
            }
        } else if (str_eq(key, "tokenizer.ggml.bos_token_id")) {
            int dummy; m->tok_bos_id = (uint32_t)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "tokenizer.ggml.eos_token_id")) {
            int dummy; m->tok_eos_id = (uint32_t)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "tokenizer.ggml.tokens")) {
            if (vtype != GGUF_META_ARRAY) {
                int dummy; skip_meta_value(&r, vtype, &dummy);
            } else {
                uint32_t arr_type = read_u32(&r);
                uint64_t arr_len  = read_u64(&r);
                m->tok_tokens_data = r.data + r.pos;
                m->tok_n_tokens = arr_len;
                int dummy;
                for (uint64_t j = 0; j < arr_len; j++) {
                    skip_meta_value(&r, arr_type, &dummy);
                }
            }
        } else if (str_eq(key, "tokenizer.ggml.scores")) {
            if (vtype != GGUF_META_ARRAY) {
                int dummy; skip_meta_value(&r, vtype, &dummy);
            } else {
                uint32_t arr_type = read_u32(&r);
                uint64_t arr_len  = read_u64(&r);
                (void)arr_type;
                m->tok_scores_data = r.data + r.pos;
                m->tok_n_scores = arr_len;
                r.pos += arr_len * 4;
            }
        } else {
            int dummy; skip_meta_value(&r, vtype, &dummy);
        }
    }

    if (max_seq_len > 0 && max_seq_len < cfg->max_seq_len) {
        cfg->max_seq_len = max_seq_len;
    }
    cfg->head_dim = cfg->n_embd / cfg->n_heads;

    /* For Qwen3.5: derive per-layer recurrent flag from full_attn_interval.
     * Layer i is a full-attention layer when (i+1) is divisible by the interval. */
    if (cfg->full_attn_interval > 0) {
        for (int i = 0; i < cfg->n_layers && i < MAX_LAYERS; i++) {
            cfg->is_recurrent[i] = (((i + 1) % cfg->full_attn_interval) != 0) ? 1 : 0;
        }
        /* Default L2 norm epsilon if not supplied by the GGUF */
        if (cfg->norm_rms_eps == 0.0f) cfg->norm_rms_eps = 1e-6f;
    }

    /* Parse tensor info entries */
    typedef struct {
        gguf_str_t name;
        uint32_t   n_dims;
        uint64_t   dims[4];
        uint32_t   type;
        uint64_t   offset;
    } tensor_info_t;

    tensor_info_t *tinfos = (tensor_info_t *)malloc(n_tensors * sizeof(tensor_info_t));
    if (!tinfos) { fprintf(stderr, "OOM allocating tensor info\n"); return -1; }

    for (uint64_t i = 0; i < n_tensors; i++) {
        tinfos[i].name   = read_gguf_string(&r);
        tinfos[i].n_dims = read_u32(&r);
        for (uint32_t d = 0; d < tinfos[i].n_dims; d++) {
            tinfos[i].dims[d] = read_u64(&r);
        }
        tinfos[i].type   = read_u32(&r);
        tinfos[i].offset = read_u64(&r);
    }

    size_t alignment = (size_t)cfg->alignment;
    size_t tensor_data_base = (r.pos + alignment - 1) & ~(alignment - 1);

    model_weights_t *w = &m->weights;
    memset(w, 0, sizeof(*w));

    for (uint64_t i = 0; i < n_tensors; i++) {
        const void *ptr = (const uint8_t *)m->mmap_addr + tensor_data_base + tinfos[i].offset;
        gguf_type_t qtype = (gguf_type_t)tinfos[i].type;

        if (str_eq(tinfos[i].name, "token_embd.weight")) {
            w->token_embd = ptr; w->type_token_embd = qtype;
        } else if (str_eq(tinfos[i].name, "output_norm.weight")) {
            w->output_norm = ptr; w->type_output_norm = qtype;
        } else if (str_eq(tinfos[i].name, "output.weight")) {
            w->output = ptr; w->type_output = qtype;
        } else {
            int layer = -1;
            char suffix[64] = {0};

            if (tinfos[i].name.len > 4 && memcmp(tinfos[i].name.str, "blk.", 4) == 0) {
                const char *p = tinfos[i].name.str + 4;
                const char *end = tinfos[i].name.str + tinfos[i].name.len;
                layer = 0;
                while (p < end && *p >= '0' && *p <= '9') {
                    layer = layer * 10 + (*p - '0');
                    p++;
                }
                if (p < end && *p == '.') {
                    p++;
                    size_t slen = (size_t)(end - p);
                    if (slen < sizeof(suffix)) {
                        memcpy(suffix, p, slen);
                        suffix[slen] = '\0';
                    }
                }
            }

            if (layer >= 0 && layer < MAX_LAYERS) {
                layer_weights_t *lw = &w->layers[layer];
                if (strcmp(suffix, "attn_norm.weight") == 0) {
                    lw->attn_norm = ptr; lw->type_attn_norm = qtype;
                } else if (strcmp(suffix, "attn_q.weight") == 0) {
                    lw->attn_q = ptr; lw->type_attn_q = qtype;
                } else if (strcmp(suffix, "attn_k.weight") == 0) {
                    lw->attn_k = ptr; lw->type_attn_k = qtype;
                } else if (strcmp(suffix, "attn_v.weight") == 0) {
                    lw->attn_v = ptr; lw->type_attn_v = qtype;
                } else if (strcmp(suffix, "attn_output.weight") == 0) {
                    lw->attn_output = ptr; lw->type_attn_output = qtype;
                } else if (strcmp(suffix, "ffn_norm.weight") == 0) {
                    lw->ffn_norm = ptr; lw->type_ffn_norm = qtype;
                } else if (strcmp(suffix, "ffn_gate.weight") == 0) {
                    lw->ffn_gate = ptr; lw->type_ffn_gate = qtype;
                } else if (strcmp(suffix, "ffn_down.weight") == 0) {
                    lw->ffn_down = ptr; lw->type_ffn_down = qtype;
                } else if (strcmp(suffix, "ffn_up.weight") == 0) {
                    lw->ffn_up = ptr; lw->type_ffn_up = qtype;
                } else if (strcmp(suffix, "post_attn_norm.weight") == 0) {
                    lw->post_attn_norm = ptr; lw->type_post_attn_norm = qtype;
                } else if (strcmp(suffix, "post_ffn_norm.weight") == 0) {
                    lw->post_ffn_norm = ptr; lw->type_post_ffn_norm = qtype;
                } else if (strcmp(suffix, "attn_q_norm.weight") == 0) {
                    lw->attn_q_norm = ptr; lw->type_attn_q_norm = qtype;
                } else if (strcmp(suffix, "attn_k_norm.weight") == 0) {
                    lw->attn_k_norm = ptr; lw->type_attn_k_norm = qtype;
                /* Qwen3.5 GDN (recurrent) layer tensors */
                } else if (strcmp(suffix, "attn_qkv.weight") == 0) {
                    lw->wqkv = ptr; lw->type_wqkv = qtype;
                } else if (strcmp(suffix, "attn_gate.weight") == 0) {
                    lw->wqkv_gate = ptr; lw->type_wqkv_gate = qtype;
                } else if (strcmp(suffix, "ssm_alpha.weight") == 0) {
                    lw->ssm_alpha = ptr; lw->type_ssm_alpha = qtype;
                } else if (strcmp(suffix, "ssm_beta.weight") == 0) {
                    lw->ssm_beta = ptr; lw->type_ssm_beta = qtype;
                } else if (strcmp(suffix, "ssm_dt.bias") == 0) {
                    lw->ssm_dt = ptr; lw->type_ssm_dt = qtype;
                } else if (strcmp(suffix, "ssm_a") == 0) {
                    lw->ssm_a = ptr; lw->type_ssm_a = qtype;
                } else if (strcmp(suffix, "ssm_conv1d.weight") == 0) {
                    lw->ssm_conv1d = ptr; lw->type_ssm_conv1d = qtype;
                } else if (strcmp(suffix, "ssm_norm.weight") == 0) {
                    lw->ssm_norm = ptr; lw->type_ssm_norm = qtype;
                } else if (strcmp(suffix, "ssm_out.weight") == 0) {
                    lw->ssm_out = ptr; lw->type_ssm_out = qtype;
                }
            }
        }
    }

    if (!w->output) {
        w->output = w->token_embd;
        w->type_output = w->type_token_embd;
    }

    if (cfg->vocab_size == 0) {
        for (uint64_t i = 0; i < n_tensors; i++) {
            if (str_eq(tinfos[i].name, "token_embd.weight")) {
                if (tinfos[i].n_dims >= 2) {
                    int d0 = (int)tinfos[i].dims[0];
                    int d1 = (int)tinfos[i].dims[1];
                    cfg->vocab_size = (d0 == cfg->n_embd) ? d1 : d0;
                }
                break;
            }
        }
    }
    if (cfg->vocab_size == 0 && m->tok_n_tokens > 0) {
        cfg->vocab_size = (int)m->tok_n_tokens;
    }

    cfg->weight_type = w->layers[0].type_attn_q;

    fprintf(stderr, "Model config:\n");
    fprintf(stderr, "  n_embd=%d, n_ffn=%d, n_heads=%d, n_kv_heads=%d\n",
            cfg->n_embd, cfg->n_ffn, cfg->n_heads, cfg->n_kv_heads);
    fprintf(stderr, "  n_layers=%d, vocab_size=%d, max_seq=%d\n",
            cfg->n_layers, cfg->vocab_size, cfg->max_seq_len);
    fprintf(stderr, "  head_dim=%d, rope_base=%.1f\n", cfg->head_dim, cfg->rope_freq_base);
    if (cfg->full_attn_interval > 0) {
        fprintf(stderr, "  ssm: d_state=%d d_inner=%d d_conv=%d dt_rank=%d n_group=%d full_attn_interval=%d\n",
                cfg->ssm_d_state, cfg->ssm_d_inner, cfg->ssm_d_conv,
                cfg->ssm_dt_rank, cfg->ssm_n_group, cfg->full_attn_interval);
    }

    free(tinfos);
    return 0;
}

/* ---- Pre-compute RoPE cos/sin lookup tables ---- */

static void init_rope_tables(run_state_t *s, const model_config_t *c) {
    int half_dim = c->head_dim / 2;
    for (int pos = 0; pos < c->max_seq_len; pos++) {
        float *cos_row = s->rope_cos + (size_t)pos * half_dim;
        float *sin_row = s->rope_sin + (size_t)pos * half_dim;
        for (int i = 0; i < half_dim; i++) {
            float theta = (float)pos / powf(c->rope_freq_base, (float)(2 * i) / (float)c->head_dim);
            cos_row[i] = cosf(theta);
            sin_row[i] = sinf(theta);
        }
    }
}

/* ---- Buffer allocation ---- */

static int allocate_run_state(model_t *m) {
    model_config_t *c = &m->config;
    run_state_t *s = &m->state;

    int kv_dim = c->n_kv_heads * c->head_dim;
    int half_dim = c->head_dim / 2;

    /* Calculate sizes for float buffers */
    size_t sz_x      = (size_t)c->n_embd * sizeof(float);
    size_t sz_xb     = (size_t)c->n_embd * sizeof(float);
    size_t sz_xb2    = (size_t)c->n_embd * sizeof(float);
    size_t sz_q      = (size_t)c->n_embd * sizeof(float);
    /* att buffer removed (flash attention) */
    size_t sz_hb     = (size_t)c->n_ffn * sizeof(float);
    size_t sz_hb2    = (size_t)c->n_ffn * sizeof(float);
    size_t sz_logits = (size_t)c->vocab_size * sizeof(float);

    int scratch_dim = c->n_embd > c->n_ffn ? c->n_embd : c->n_ffn;
    if (c->vocab_size > scratch_dim) scratch_dim = c->vocab_size;
    size_t sz_scratch = (size_t)scratch_dim * sizeof(float);

    /* RoPE tables: cos and sin for each (position, dim_pair) */
    size_t sz_rope = (size_t)c->max_seq_len * half_dim * sizeof(float) * 2;

    /* Norm weights: (n_layers * 2 + 1) * n_embd floats for pre-norms + output norm.
     * For Gemma 4, also include post_attn_norm, post_ffn_norm (n_embd each per layer)
     * and attn_q_norm, attn_k_norm (head_dim each per layer).
     * For Qwen3.5, attn_norm + post_attn_norm per layer (no separate ffn_norm). */
    int has_post_norm = (m->weights.layers[0].post_attn_norm != NULL);
    int has_qk_norm   = (m->weights.layers[0].attn_q_norm   != NULL);
    /* Count layers that have an ffn_norm (pure transformer layers) */
    int n_ffn_norm_layers = 0;
    for (int l = 0; l < c->n_layers; l++) {
        if (m->weights.layers[l].ffn_norm != NULL) n_ffn_norm_layers++;
    }
    size_t n_norm = (size_t)c->n_layers * (size_t)c->n_embd;  /* attn_norm per layer */
    n_norm += (size_t)n_ffn_norm_layers * (size_t)c->n_embd;   /* ffn_norm where present */
    n_norm += 1 * (size_t)c->n_embd;                           /* output_norm */
    if (has_post_norm) n_norm += (size_t)c->n_layers * 2 * (size_t)c->n_embd;
    if (has_qk_norm)   n_norm += (size_t)c->n_layers * 2 * (size_t)c->head_dim;
    size_t sz_norm = n_norm * sizeof(float);

    size_t total = sz_x + sz_xb + sz_xb2 + sz_q +
                   sz_hb + sz_hb2 + sz_logits +
                   sz_scratch + sz_rope + sz_norm;

    /* FP16 KV cache: separate allocation */
    size_t kv_elements = (size_t)c->n_layers * c->max_seq_len * kv_dim;
    size_t sz_kv = kv_elements * sizeof(uint16_t) * 2; /* key + val */

    /* SSM state allocation for Qwen3.5 hybrid models */
    size_t ssm_block_sz = 0;
    if (c->full_attn_interval > 0 && c->ssm_d_state > 0 && c->ssm_dt_rank > 0) {
        int head_dim_ssm = c->ssm_d_state;    /* head_k_dim == head_v_dim */
        int n_v_heads    = c->ssm_dt_rank;
        int n_k_heads    = c->ssm_n_group;
        int d_conv       = c->ssm_d_conv;
        int conv_dim     = head_dim_ssm * n_k_heads * 2 + head_dim_ssm * n_v_heads;
        /* ssm_states  : n_layers * n_v_heads * head_dim * head_dim */
        size_t sz_ssm_state  = (size_t)c->n_layers * n_v_heads * head_dim_ssm * head_dim_ssm;
        /* conv_states : n_layers * (d_conv-1) * conv_dim */
        size_t sz_conv_state = (size_t)c->n_layers * (d_conv > 1 ? d_conv - 1 : 1) * conv_dim;
        /* ssm_norm_w  : n_layers * head_dim */
        size_t sz_ssm_norm   = (size_t)c->n_layers * head_dim_ssm;
        /* ssm_dt_w    : n_layers * n_v_heads */
        size_t sz_ssm_dt     = (size_t)c->n_layers * n_v_heads;
        /* ssm_a_w     : n_layers * n_v_heads */
        size_t sz_ssm_a      = (size_t)c->n_layers * n_v_heads;
        /* ssm_conv1d_w: n_layers * conv_dim * d_conv */
        size_t sz_ssm_conv1d = (size_t)c->n_layers * conv_dim * d_conv;
        ssm_block_sz = (sz_ssm_state + sz_conv_state +
                        sz_ssm_norm + sz_ssm_dt + sz_ssm_a + sz_ssm_conv1d) * sizeof(float);
    }

    fprintf(stderr, "Allocating %.2f MB for runtime state (+ %.2f MB FP16 KV cache%s)\n",
            (double)total / (1024.0 * 1024.0),
            (double)sz_kv  / (1024.0 * 1024.0),
            ssm_block_sz > 0 ? " + SSM states" : "");

    s->mem_block = calloc(1, total);
    if (!s->mem_block) {
        fprintf(stderr, "OOM: cannot allocate %zu bytes\n", total);
        return -1;
    }
    s->mem_size = total + sz_kv;

    /* Allocate FP16 KV cache separately */
    s->kv_block = calloc(1, sz_kv);
    if (!s->kv_block) {
        fprintf(stderr, "OOM: cannot allocate %zu bytes for KV cache\n", sz_kv);
        free(s->mem_block);
        return -1;
    }
    s->kv_size = sz_kv;

    /* Allocate SSM block (zeroed — recurrent states must start as zero) */
    s->ssm_block      = NULL;
    s->ssm_block_size = ssm_block_sz;
    s->ssm_states     = NULL;
    s->conv_states    = NULL;
    s->ssm_norm_w     = NULL;
    s->ssm_dt_w       = NULL;
    s->ssm_a_w        = NULL;
    s->ssm_conv1d_w   = NULL;
    if (ssm_block_sz > 0) {
        s->ssm_block = calloc(1, ssm_block_sz);
        if (!s->ssm_block) {
            fprintf(stderr, "OOM: cannot allocate %zu bytes for SSM states\n", ssm_block_sz);
            free(s->kv_block);
            free(s->mem_block);
            return -1;
        }
    }

    /* Carve float pointers */
    float *p = (float *)s->mem_block;
    s->x      = p; p += c->n_embd;
    s->xb     = p; p += c->n_embd;
    s->xb2    = p; p += c->n_embd;
    s->q      = p; p += c->n_embd;
    s->hb     = p; p += c->n_ffn;
    s->hb2    = p; p += c->n_ffn;
    s->logits = p; p += c->vocab_size;
    s->dequant_scratch = p; p += scratch_dim;

    /* RoPE tables */
    s->rope_cos = p; p += (size_t)c->max_seq_len * half_dim;
    s->rope_sin = p; p += (size_t)c->max_seq_len * half_dim;

    /* Norm weights */
    s->norm_weights = p;

    /* FP16 KV cache pointers */
    uint16_t *kp = (uint16_t *)s->kv_block;
    s->key_cache = kp; kp += kv_elements;
    s->val_cache = kp;

    /* Pre-dequantize norm weights */
    float *nw = s->norm_weights;
    for (int l = 0; l < c->n_layers; l++) {
        s->attn_norm_w[l] = nw;
        dequantize_row(m->weights.layers[l].attn_norm, nw, c->n_embd,
                       m->weights.layers[l].type_attn_norm);
        nw += c->n_embd;

        if (m->weights.layers[l].ffn_norm) {
            s->ffn_norm_w[l] = nw;
            dequantize_row(m->weights.layers[l].ffn_norm, nw, c->n_embd,
                           m->weights.layers[l].type_ffn_norm);
            nw += c->n_embd;
        } else {
            s->ffn_norm_w[l] = NULL;
        }
    }
    s->output_norm_w = nw;
    dequantize_row(m->weights.output_norm, nw, c->n_embd,
                   m->weights.type_output_norm);
    nw += c->n_embd;

    /* Pre-dequantize Gemma 4 post-norm weights (post_attn_norm, post_ffn_norm) */
    for (int l = 0; l < c->n_layers; l++) {
        if (m->weights.layers[l].post_attn_norm) {
            s->post_attn_norm_w[l] = nw;
            dequantize_row(m->weights.layers[l].post_attn_norm, nw, c->n_embd,
                           m->weights.layers[l].type_post_attn_norm);
            nw += c->n_embd;
        } else {
            s->post_attn_norm_w[l] = NULL;
        }
        if (m->weights.layers[l].post_ffn_norm) {
            s->post_ffn_norm_w[l] = nw;
            dequantize_row(m->weights.layers[l].post_ffn_norm, nw, c->n_embd,
                           m->weights.layers[l].type_post_ffn_norm);
            nw += c->n_embd;
        } else {
            s->post_ffn_norm_w[l] = NULL;
        }
    }

    /* Pre-dequantize Gemma 4 QK-norm weights (attn_q_norm, attn_k_norm) */
    for (int l = 0; l < c->n_layers; l++) {
        if (m->weights.layers[l].attn_q_norm) {
            s->attn_q_norm_w[l] = nw;
            dequantize_row(m->weights.layers[l].attn_q_norm, nw, c->head_dim,
                           m->weights.layers[l].type_attn_q_norm);
            nw += c->head_dim;
        } else {
            s->attn_q_norm_w[l] = NULL;
        }
        if (m->weights.layers[l].attn_k_norm) {
            s->attn_k_norm_w[l] = nw;
            dequantize_row(m->weights.layers[l].attn_k_norm, nw, c->head_dim,
                           m->weights.layers[l].type_attn_k_norm);
            nw += c->head_dim;
        } else {
            s->attn_k_norm_w[l] = NULL;
        }
    }

    /* Init tensor scratch */
    tensor_init_scratch(s->dequant_scratch, scratch_dim);

    /* Pre-compute RoPE tables (eliminates powf/cosf/sinf from hot path) */
    init_rope_tables(s, c);

    /* Set up SSM state and weight pointers, and pre-dequantize small SSM tensors */
    if (s->ssm_block && c->ssm_d_state > 0 && c->ssm_dt_rank > 0) {
        int hd       = c->ssm_d_state;   /* head_dim for both K and V */
        int n_vh     = c->ssm_dt_rank;   /* num V heads */
        int n_kh     = c->ssm_n_group;   /* num K heads */
        int dc       = c->ssm_d_conv;
        int conv_dim = hd * n_kh * 2 + hd * n_vh;
        int dconv1   = (dc > 1) ? dc - 1 : 1; /* conv state depth */

        float *sp = (float *)s->ssm_block;

        /* ssm_states: n_layers * n_vh * hd * hd */
        s->ssm_states = sp;
        sp += (size_t)c->n_layers * n_vh * hd * hd;

        /* conv_states: n_layers * dconv1 * conv_dim */
        s->conv_states = sp;
        sp += (size_t)c->n_layers * dconv1 * conv_dim;

        /* ssm_norm_w: n_layers * hd */
        s->ssm_norm_w = sp;
        sp += (size_t)c->n_layers * hd;

        /* ssm_dt_w: n_layers * n_vh */
        s->ssm_dt_w = sp;
        sp += (size_t)c->n_layers * n_vh;

        /* ssm_a_w: n_layers * n_vh */
        s->ssm_a_w = sp;
        sp += (size_t)c->n_layers * n_vh;

        /* ssm_conv1d_w: n_layers * conv_dim * dc */
        s->ssm_conv1d_w = sp;
        /* sp += c->n_layers * conv_dim * dc; (end of block) */

        /* Dequantize per-layer SSM small tensors */
        for (int l = 0; l < c->n_layers; l++) {
            layer_weights_t *lw = &m->weights.layers[l];
            if (!c->is_recurrent[l]) continue; /* skip full-attention layers */

            /* ssm_norm weight: [hd] */
            if (lw->ssm_norm) {
                dequantize_row(lw->ssm_norm, s->ssm_norm_w + (size_t)l * hd,
                               hd, lw->type_ssm_norm);
            }
            /* ssm_dt bias: [n_vh] */
            if (lw->ssm_dt) {
                dequantize_row(lw->ssm_dt, s->ssm_dt_w + (size_t)l * n_vh,
                               n_vh, lw->type_ssm_dt);
            }
            /* ssm_a parameter: [n_vh] */
            if (lw->ssm_a) {
                dequantize_row(lw->ssm_a, s->ssm_a_w + (size_t)l * n_vh,
                               n_vh, lw->type_ssm_a);
            }
            /* ssm_conv1d kernel: [d_conv, conv_dim] stored as conv_dim rows of d_conv elems */
            if (lw->ssm_conv1d) {
                float *dst = s->ssm_conv1d_w + (size_t)l * conv_dim * dc;
                size_t row_bytes = gguf_type_row_size(lw->type_ssm_conv1d, dc);
                const uint8_t *src = (const uint8_t *)lw->ssm_conv1d;
                for (int ch = 0; ch < conv_dim; ch++) {
                    dequantize_row(src + (size_t)ch * row_bytes,
                                   dst + ch * dc, dc, lw->type_ssm_conv1d);
                }
            }
        }
    }

    return 0;
}

/* ---- Public API ---- */

int model_load(model_t *m, const char *path, int max_seq_len) {
    memset(m, 0, sizeof(*m));

    if (mmap_file(m, path) != 0) return -1;
    if (parse_gguf(m, max_seq_len) != 0) return -1;
    if (allocate_run_state(m) != 0) return -1;

    return 0;
}

/* ================================================================
 * Forward pass with:
 *   - FP16 KV cache (halves memory bandwidth in attention)
 *   - Flash attention / online softmax (single pass, no score buffer)
 *   - Pre-computed RoPE tables (table lookup instead of trig)
 *   - Qwen3.5 hybrid GDN (Gated Delta Network) layers
 * ================================================================ */

/* Upper bounds for on-stack GDN arrays — no heap allocation in the hot path.
 * A runtime check in gdn_forward() enforces these limits. */
#define MAX_SSM_HEADS   64   /* max n_vh (V-heads); Qwen3.5-0.8B uses 8 */
#define MAX_SSM_HEAD_DIM 256 /* max hd  (SSM head dim); Qwen3.5-0.8B uses 64 */

/* ---- Small math helpers used by the GDN forward pass ---- */

/* sigmoid(x) = 1 / (1 + exp(-x)) */
static float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* softplus(x) = log(1 + exp(x)), numerically stable */
static float softplusf(float x) {
    if (x > 20.0f) return x;
    return log1pf(expf(x));
}

/* In-place L2 normalisation: x[i] /= sqrt(sum(x^2) + eps^2) */
static void l2_norm_inplace(float *x, int n, float eps) {
    float sq = eps * eps;
    for (int i = 0; i < n; i++) sq += x[i] * x[i];
    float inv = 1.0f / sqrtf(sq);
    for (int i = 0; i < n; i++) x[i] *= inv;
}

/* ---- GDN (Gated Delta Network) single-token forward pass ----
 *
 * Implements the autoregressive Gated Delta Rule for one token:
 *   for each v-head h:
 *     state_h *= forget_h                         (decay)
 *     sk[j]    = sum_i state_h[i*hd+j] * k_h[i]  (state lookup)
 *     d[j]     = beta_h * (v_h[j] - sk[j])        (delta error)
 *     state_h[i*hd+j] += k_h[i] * d[j]            (rank-1 update)
 *     o_h[j]   = scale * sum_i state_h[i*hd+j]*q_h[i]  (output)
 *
 * Buffer contract:
 *   qkv_buf   : size >= conv_dim  (caller may use s->hb)
 *   z_buf     : size >= val_dim   (caller may use s->hb2)
 *   accum_buf : size >= val_dim   (may alias xb — safe because step-1 reads finish before
 *                                  steps 5-6 write; caller may pass s->xb)
 *   result    : size >= dim       (caller may use s->xb2)
 */
static void gdn_forward(
    float       *xb,            /* [dim] layer input; also used as accum below     */
    layer_weights_t *lw,
    const float *ssm_norm_w,    /* [hd]               pre-dequantized GDN out-norm */
    const float *ssm_dt_w,      /* [n_vh]             pre-dequantized DT bias      */
    const float *ssm_a_w,       /* [n_vh]             pre-dequantized A parameter  */
    const float *ssm_conv1d_w,  /* [conv_dim * d_conv] channel-major, pre-deq.     */
    float       *ssm_state,     /* [n_vh * hd * hd]   mutable recurrent state      */
    float       *conv_state,    /* [(d_conv-1)*conv_dim] mutable conv history      */
    float       *qkv_buf,       /* scratch >= conv_dim (e.g. s->hb)                */
    float       *z_buf,         /* scratch >= val_dim  (e.g. s->hb2)               */
    float       *accum_buf,     /* [>= val_dim] accumulation; may alias xb         */
    float       *result,        /* [dim] final output  (e.g. s->xb2)               */
    int dim, int hd, int n_vh, int n_kh, int d_conv, float l2_eps
) {
    int key_dim  = hd * n_kh;
    int val_dim  = hd * n_vh;
    int conv_dim = key_dim * 2 + val_dim;
    int dconv1   = (d_conv > 1) ? d_conv - 1 : 1;

    /* Runtime bound-checks for on-stack array sizes */
    if (n_vh > MAX_SSM_HEADS || hd > MAX_SSM_HEAD_DIM) {
        fprintf(stderr, "gdn_forward: n_vh=%d or hd=%d exceeds compiled limit "
                "(%d/%d); rebuild with larger MAX_SSM_HEADS/MAX_SSM_HEAD_DIM\n",
                n_vh, hd, MAX_SSM_HEADS, MAX_SSM_HEAD_DIM);
        return;
    }

    /* Step 1: input projections (xb is read-only in this block) */
    matmul(qkv_buf,   xb, lw->wqkv,      dim, conv_dim, lw->type_wqkv);
    matmul(z_buf,     xb, lw->wqkv_gate, dim, val_dim,  lw->type_wqkv_gate);
    float beta_raw[MAX_SSM_HEADS], alpha_raw[MAX_SSM_HEADS];
    matmul(beta_raw,  xb, lw->ssm_beta,  dim, n_vh, lw->type_ssm_beta);
    matmul(alpha_raw, xb, lw->ssm_alpha, dim, n_vh, lw->type_ssm_alpha);

    /* Step 2: depthwise 1-D conv (in-place in qkv_buf) + conv state update.
     *
     * For channel j, save qkv_buf[j] as `orig` first, compute conv output and
     * overwrite qkv_buf[j], then update conv_state using `orig` — all in one pass,
     * no extra allocation needed.
     */
    for (int j = 0; j < conv_dim; j++) {
        const float *k_j = ssm_conv1d_w + (size_t)j * d_conv;
        float       *cs_j = conv_state  + (size_t)j * dconv1;
        float orig = qkv_buf[j];                     /* current token input */
        float acc  = k_j[d_conv - 1] * orig;         /* current tap */
        for (int t = 0; t < dconv1; t++) acc += k_j[t] * cs_j[t];
        qkv_buf[j] = acc * (1.0f / (1.0f + expf(-acc))); /* SiLU in-place */
        /* shift history window: drop oldest tap, add current token */
        for (int t = 0; t < dconv1 - 1; t++) cs_j[t] = cs_j[t + 1];
        if (dconv1 > 0) cs_j[dconv1 - 1] = orig;
    }

    /* Step 3: split silu(conv_out) and L2-normalise Q/K per k-head */
    float *q_c = qkv_buf;               /* [key_dim] */
    float *k_c = qkv_buf + key_dim;     /* [key_dim] */
    float *v_c = qkv_buf + 2 * key_dim; /* [val_dim] */
    for (int h = 0; h < n_kh; h++) {
        l2_norm_inplace(q_c + h * hd, hd, l2_eps);
        l2_norm_inplace(k_c + h * hd, hd, l2_eps);
    }

    /* Step 4: per-head forget gate and beta */
    float forget_arr[MAX_SSM_HEADS], beta_arr[MAX_SSM_HEADS];
    for (int h = 0; h < n_vh; h++) {
        float alpha_sp = softplusf(alpha_raw[h] + ssm_dt_w[h]);
        forget_arr[h] = expf(alpha_sp * ssm_a_w[h]); /* ssm_a = -exp(A_log) < 0 */
        beta_arr[h]   = sigmoidf(beta_raw[h]);
    }

    /* Steps 5-6: delta rule + gated RMS-norm per V head.
     * Writes to accum_buf (may alias xb; safe because step-1 reads are done). */
    float scale = 1.0f / sqrtf((float)hd);
    for (int h = 0; h < n_vh; h++) {
        int kh = (n_vh > n_kh) ? (h * n_kh / n_vh) : h;
        float       *state_h = ssm_state + (size_t)h  * hd * hd;
        const float *k_h     = k_c       + kh * hd;
        const float *q_h     = q_c       + kh * hd;
        const float *v_h     = v_c       + h  * hd;
        float       *o_h     = accum_buf + h  * hd;

        /* decay */
        float fh = forget_arr[h];
        for (int ij = 0; ij < hd * hd; ij++) state_h[ij] *= fh;

        /* sk[j] = state_h^T @ k_h */
        float sk[MAX_SSM_HEAD_DIM];
        for (int j = 0; j < hd; j++) {
            float a = 0.0f;
            for (int i = 0; i < hd; i++) a += state_h[(size_t)i * hd + j] * k_h[i];
            sk[j] = a;
        }

        /* delta d = beta * (v - sk) */
        float d_arr[MAX_SSM_HEAD_DIM];
        float bh = beta_arr[h];
        for (int j = 0; j < hd; j++) d_arr[j] = bh * (v_h[j] - sk[j]);

        /* rank-1 state update */
        for (int i = 0; i < hd; i++) {
            float ki = k_h[i];
            float *row = state_h + (size_t)i * hd;
            for (int j = 0; j < hd; j++) row[j] += ki * d_arr[j];
        }

        /* output o_h = scale * state_h^T @ q_h */
        for (int j = 0; j < hd; j++) {
            float a = 0.0f;
            for (int i = 0; i < hd; i++) a += state_h[(size_t)i * hd + j] * q_h[i];
            o_h[j] = a * scale;
        }

        /* per-head gated RMS-norm + SiLU gate */
        {
            float ss = 0.0f;
            for (int j = 0; j < hd; j++) ss += o_h[j] * o_h[j];
            float inv_rms = 1.0f / sqrtf(ss / (float)hd + 1e-6f);
            const float *z_h = z_buf + (size_t)h * hd;
            for (int j = 0; j < hd; j++) {
                float normed = o_h[j] * inv_rms * ssm_norm_w[j];
                float z      = z_h[j];
                o_h[j] = normed * (z / (1.0f + expf(-z))); /* *= silu(z) */
            }
        }
    }

    /* Step 7: output projection result[dim] = ssm_out @ accum_buf[val_dim] */
    matmul(result, accum_buf, lw->ssm_out, val_dim, dim, lw->type_ssm_out);
}

float *model_forward(model_t *m, int token, int pos) {
    model_config_t *c = &m->config;
    model_weights_t *w = &m->weights;
    run_state_t *s = &m->state;

    int dim    = c->n_embd;
    int n_ffn  = c->n_ffn;
    int n_heads = c->n_heads;
    int n_kv_heads = c->n_kv_heads;
    int head_dim = c->head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int kv_mul = n_heads / n_kv_heads;
    int seq_len = c->max_seq_len;
    int half_dim = head_dim / 2;

    /* RoPE table pointers for this position */
    const float *cos_pos = s->rope_cos + (size_t)pos * half_dim;
    const float *sin_pos = s->rope_sin + (size_t)pos * half_dim;

    /* 1. Embedding lookup */
    {
        size_t row_bytes = gguf_type_row_size(w->type_token_embd, dim);
        const void *embd_row = (const uint8_t *)w->token_embd + (size_t)token * row_bytes;
        dequantize_row(embd_row, s->x, dim, w->type_token_embd);
    }

    /* 2. Transformer layers */
    for (int l = 0; l < c->n_layers; l++) {
        layer_weights_t *lw = &w->layers[l];

        /* Pre-attention RMSNorm (shared by all architectures) */
        rmsnorm(s->xb, s->x, s->attn_norm_w[l], dim);

        /* ================================================================
         * Qwen3.5 GDN (recurrent) layer
         * ================================================================ */
        if (c->is_recurrent[l]) {
            int hd    = c->ssm_d_state;
            int n_vh  = c->ssm_dt_rank;
            int n_kh  = c->ssm_n_group;
            int dc    = c->ssm_d_conv;
            int val_dim  = hd * n_vh;
            int key_dim  = hd * n_kh;
            int conv_dim = key_dim * 2 + val_dim;
            int dconv1   = (dc > 1) ? dc - 1 : 1;

            float *ssm_state_l  = s->ssm_states  + (size_t)l * n_vh * hd * hd;
            float *conv_state_l = s->conv_states  + (size_t)l * dconv1 * conv_dim;

            /* s->hb  : qkv_buf  (n_ffn >= conv_dim) */
            /* s->hb2 : z_buf    (n_ffn >= val_dim)  */
            /* s->xb  : accum_buf (n_embd >= val_dim; xb read-only in step-1, then reused) */
            /* s->xb2 : result   (n_embd = dim)      */
            gdn_forward(
                s->xb,
                lw,
                s->ssm_norm_w   + (size_t)l * hd,
                s->ssm_dt_w     + (size_t)l * n_vh,
                s->ssm_a_w      + (size_t)l * n_vh,
                s->ssm_conv1d_w + (size_t)l * conv_dim * dc,
                ssm_state_l, conv_state_l,
                s->hb, s->hb2, s->xb, s->xb2,
                dim, hd, n_vh, n_kh, dc, c->norm_rms_eps
            );

            /* Post-GDN normalization (Qwen3.5 uses post_attn_norm here) */
            if (s->post_attn_norm_w[l]) {
                rmsnorm(s->xb2, s->xb2, s->post_attn_norm_w[l], dim);
            }
            vec_add(s->x, s->xb2, dim);

            /* FFN: use post_attn_norm as pre-FFN norm for Qwen3.5 (no ffn_norm) */
            const float *ffn_pre_norm = s->ffn_norm_w[l]
                                        ? s->ffn_norm_w[l]
                                        : s->post_attn_norm_w[l];
            rmsnorm(s->xb, s->x, ffn_pre_norm, dim);

            matmul(s->hb,  s->xb, lw->ffn_gate, dim, n_ffn, lw->type_ffn_gate);
            matmul(s->hb2, s->xb, lw->ffn_up,   dim, n_ffn, lw->type_ffn_up);
            silu(s->hb, n_ffn);
            elemwise_mul(s->hb, s->hb, s->hb2, n_ffn);
            matmul(s->xb, s->hb, lw->ffn_down, n_ffn, dim, lw->type_ffn_down);
            if (s->post_ffn_norm_w[l]) {
                rmsnorm(s->xb, s->xb, s->post_ffn_norm_w[l], dim);
            }
            vec_add(s->x, s->xb, dim);
            continue; /* next layer */
        }

        /* ================================================================
         * Full-attention layer (standard transformer or Qwen3.5 full-attn)
         * ================================================================ */

        /* QKV projections.
         * For Qwen3.5 full-attention layers, attn_q.weight outputs Q+gate (dim*2),
         * stored into hb (n_ffn >= dim*2).  Pure transformer layers use the standard
         * path writing directly into s->q. */
        int is_qwen35 = (c->full_attn_interval > 0);
        float *gate_ptr = NULL; /* pointer to Q gate values (Qwen3.5 only) */

        if (is_qwen35 && lw->wqkv == NULL) {
            /* Qwen3.5 full-attention: attn_q produces Q+gate fused (dim*2 outputs) */
            matmul(s->hb, s->xb, lw->attn_q, dim, dim * 2, lw->type_attn_q);
            /* Copy Q into s->q; keep gate in hb[dim..2*dim-1] */
            memcpy(s->q, s->hb, (size_t)dim * sizeof(float));
            gate_ptr = s->hb + dim; /* gate lives in second half of hb */
        } else {
            matmul(s->q, s->xb, lw->attn_q, dim, dim, lw->type_attn_q);
        }

        /* K and V: project into float temp, then store as FP16 in cache */
        float *k_tmp = s->xb2; /* reuse xb2 as temp for K (kv_dim <= dim) */
        matmul(k_tmp, s->xb, lw->attn_k, dim, kv_dim, lw->type_attn_k);

        /* Store K as FP16 */
        uint16_t *kcache_layer = s->key_cache + (size_t)l * seq_len * kv_dim;
        uint16_t *vcache_layer = s->val_cache + (size_t)l * seq_len * kv_dim;
        uint16_t *key_pos_fp16 = kcache_layer + (size_t)pos * kv_dim;

        /* Per-head QK normalization:
         *   Gemma 4: applied AFTER RoPE
         *   Qwen3.5 full-attention: applied BEFORE RoPE */
        if (is_qwen35 && s->attn_q_norm_w[l]) {
            for (int h = 0; h < n_heads; h++) {
                rmsnorm(s->q + h * head_dim, s->q + h * head_dim,
                        s->attn_q_norm_w[l], head_dim);
            }
            for (int h = 0; h < n_kv_heads; h++) {
                rmsnorm(k_tmp + h * head_dim, k_tmp + h * head_dim,
                        s->attn_k_norm_w[l], head_dim);
            }
        }

        /* Apply RoPE to Q and K (using pre-computed tables) */
        rope(s->q, k_tmp, head_dim, n_heads, n_kv_heads, cos_pos, sin_pos);

        /* Per-head QK normalization (Gemma 4 style): applied AFTER RoPE */
        if (!is_qwen35 && s->attn_q_norm_w[l]) {
            for (int h = 0; h < n_heads; h++) {
                rmsnorm(s->q + h * head_dim, s->q + h * head_dim,
                        s->attn_q_norm_w[l], head_dim);
            }
            for (int h = 0; h < n_kv_heads; h++) {
                rmsnorm(k_tmp + h * head_dim, k_tmp + h * head_dim,
                        s->attn_k_norm_w[l], head_dim);
            }
        }

        /* Convert K to FP16 and store */
        for (int d = 0; d < kv_dim; d++) {
            key_pos_fp16[d] = fp32_to_fp16(k_tmp[d]);
        }

        /* V projection -> store directly as FP16 */
        float *v_tmp = s->xb2;
        matmul(v_tmp, s->xb, lw->attn_v, dim, kv_dim, lw->type_attn_v);
        uint16_t *val_pos_fp16 = vcache_layer + (size_t)pos * kv_dim;
        for (int d = 0; d < kv_dim; d++) {
            val_pos_fp16[d] = fp32_to_fp16(v_tmp[d]);
        }

        /* ---- Flash Attention (online softmax) ----
         *
         * Instead of materializing the full [n_heads * seq_len] score array,
         * compute attention in a single pass using the online softmax trick:
         *
         *   max_s = -inf, sum_exp = 0, acc[d] = 0
         *   for each cached position t:
         *     s = dot(Q_h, K_t) / sqrt(d)
         *     if s > max_s:
         *       correction = exp(max_s - s)
         *       acc *= correction, sum_exp *= correction
         *       sum_exp += 1, acc += V_t
         *       max_s = s
         *     else:
         *       w = exp(s - max_s)
         *       sum_exp += w, acc += w * V_t
         *   result = acc / sum_exp
         *
         * This saves memory (no att[] buffer) and is more cache-friendly.
         */
        for (int h = 0; h < n_heads; h++) {
            float *qh = s->q + h * head_dim;
            int kv_h = h / kv_mul;
            float *xbh = s->xb + h * head_dim;

            float max_score = -1e30f;
            float sum_exp = 0.0f;
            /* Accumulator for weighted V values */
            float acc[256]; /* head_dim is typically 64-128 */
            memset(acc, 0, (size_t)head_dim * sizeof(float));

            for (int t = 0; t <= pos; t++) {
                /* Compute score: dot(Q_h, K_t) / sqrt(head_dim) */
                const uint16_t *kt = kcache_layer + (size_t)t * kv_dim + kv_h * head_dim;
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += qh[d] * fp16_to_fp32(kt[d]);
                }
                score /= sqrtf((float)head_dim);

                /* Online softmax update */
                const uint16_t *vt = vcache_layer + (size_t)t * kv_dim + kv_h * head_dim;

                if (score > max_score) {
                    float correction = expf(max_score - score);
                    sum_exp = sum_exp * correction + 1.0f;
                    for (int d = 0; d < head_dim; d++) {
                        acc[d] = acc[d] * correction + fp16_to_fp32(vt[d]);
                    }
                    max_score = score;
                } else {
                    float w = expf(score - max_score);
                    sum_exp += w;
                    for (int d = 0; d < head_dim; d++) {
                        acc[d] += w * fp16_to_fp32(vt[d]);
                    }
                }
            }

            /* Normalize */
            float inv_sum = 1.0f / sum_exp;
            for (int d = 0; d < head_dim; d++) {
                xbh[d] = acc[d] * inv_sum;
            }
        }

        /* Apply sigmoid Q-gate (Qwen3.5 full-attention only) */
        if (gate_ptr) {
            for (int i = 0; i < dim; i++) {
                s->xb[i] *= sigmoidf(gate_ptr[i]);
            }
        }

        /* Output projection */
        matmul(s->xb2, s->xb, lw->attn_output, dim, dim, lw->type_attn_output);
        /* Post-attention normalization (Gemma 4 / Qwen3.5) */
        if (s->post_attn_norm_w[l]) {
            rmsnorm(s->xb2, s->xb2, s->post_attn_norm_w[l], dim);
        }
        vec_add(s->x, s->xb2, dim);

        /* ---- FFN (SwiGLU) ---- */
        /* For Qwen3.5 full-attention layers, use post_attn_norm as pre-FFN norm
         * when no separate ffn_norm exists. */
        {
            const float *ffn_pre_norm = s->ffn_norm_w[l]
                                        ? s->ffn_norm_w[l]
                                        : s->post_attn_norm_w[l];
            rmsnorm(s->xb, s->x, ffn_pre_norm, dim);
        }

        matmul(s->hb,  s->xb, lw->ffn_gate, dim, n_ffn, lw->type_ffn_gate);
        matmul(s->hb2, s->xb, lw->ffn_up,   dim, n_ffn, lw->type_ffn_up);

        silu(s->hb, n_ffn);
        elemwise_mul(s->hb, s->hb, s->hb2, n_ffn);

        matmul(s->xb, s->hb, lw->ffn_down, n_ffn, dim, lw->type_ffn_down);
        /* Post-FFN normalization (Gemma 4) */
        if (s->post_ffn_norm_w[l]) {
            rmsnorm(s->xb, s->xb, s->post_ffn_norm_w[l], dim);
        }
        vec_add(s->x, s->xb, dim);
    }

    /* 3. Final RMSNorm */
    rmsnorm(s->x, s->x, s->output_norm_w, dim);

    /* 4. Output projection -> logits */
    matmul(s->logits, s->x, w->output, dim, c->vocab_size, w->type_output);

    return s->logits;
}

void model_free(model_t *m) {
    if (m->state.mem_block) {
        free(m->state.mem_block);
        m->state.mem_block = NULL;
    }
    if (m->state.kv_block) {
        free(m->state.kv_block);
        m->state.kv_block = NULL;
    }
    if (m->state.ssm_block) {
        free(m->state.ssm_block);
        m->state.ssm_block = NULL;
    }
    munmap_file(m);
}

/* ================================================================
 * KV Cache Persistence — save/load KV state to skip prompt prefill
 *
 * File format:
 *   [4 bytes] magic: KVCACHE_MAGIC
 *   [4 bytes] n_pos: number of cached positions
 *   [4 bytes] n_layers
 *   [4 bytes] kv_dim (n_kv_heads * head_dim)
 *   [N bytes] key_cache FP16 data (n_layers * n_pos * kv_dim * sizeof(uint16_t))
 *   [N bytes] val_cache FP16 data (same size)
 * ================================================================ */

int kvcache_save(const model_t *m, const char *path, int n_pos) {
    const model_config_t *c = &m->config;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int seq_len = c->max_seq_len;

    if (n_pos <= 0 || n_pos > seq_len) return -1;

    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "kvcache_save: cannot open %s\n", path);
        return -1;
    }

    uint32_t header[4] = {
        KVCACHE_MAGIC,
        (uint32_t)n_pos,
        (uint32_t)c->n_layers,
        (uint32_t)kv_dim
    };
    fwrite(header, sizeof(uint32_t), 4, f);

    /* Write KV cache for each layer, only the first n_pos positions */
    size_t row_size = (size_t)kv_dim * sizeof(uint16_t);
    for (int l = 0; l < c->n_layers; l++) {
        const uint16_t *kcache_l = m->state.key_cache + (size_t)l * seq_len * kv_dim;
        for (int p = 0; p < n_pos; p++) {
            fwrite(kcache_l + (size_t)p * kv_dim, 1, row_size, f);
        }
    }
    for (int l = 0; l < c->n_layers; l++) {
        const uint16_t *vcache_l = m->state.val_cache + (size_t)l * seq_len * kv_dim;
        for (int p = 0; p < n_pos; p++) {
            fwrite(vcache_l + (size_t)p * kv_dim, 1, row_size, f);
        }
    }

    fclose(f);
    fprintf(stderr, "KV cache saved: %d positions to %s\n", n_pos, path);
    return 0;
}

int kvcache_load(model_t *m, const char *path) {
    const model_config_t *c = &m->config;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int seq_len = c->max_seq_len;

    FILE *f = fopen(path, "rb");
    if (!f) return 0;

    uint32_t header[4];
    if (fread(header, sizeof(uint32_t), 4, f) != 4) {
        fclose(f);
        return 0;
    }

    if (header[0] != KVCACHE_MAGIC) {
        fprintf(stderr, "kvcache_load: invalid magic\n");
        fclose(f);
        return 0;
    }

    int n_pos = (int)header[1];
    int file_layers = (int)header[2];
    int file_kv_dim = (int)header[3];

    if (file_layers != c->n_layers || file_kv_dim != kv_dim) {
        fprintf(stderr, "kvcache_load: model mismatch (layers=%d/%d, kv_dim=%d/%d)\n",
                file_layers, c->n_layers, file_kv_dim, kv_dim);
        fclose(f);
        return 0;
    }
    if (n_pos > seq_len) {
        fprintf(stderr, "kvcache_load: cached %d positions exceeds max_seq_len %d\n",
                n_pos, seq_len);
        fclose(f);
        return 0;
    }

    size_t row_size = (size_t)kv_dim * sizeof(uint16_t);
    for (int l = 0; l < c->n_layers; l++) {
        uint16_t *kcache_l = m->state.key_cache + (size_t)l * seq_len * kv_dim;
        for (int p = 0; p < n_pos; p++) {
            if (fread(kcache_l + (size_t)p * kv_dim, 1, row_size, f) != row_size) {
                fclose(f);
                return 0;
            }
        }
    }
    for (int l = 0; l < c->n_layers; l++) {
        uint16_t *vcache_l = m->state.val_cache + (size_t)l * seq_len * kv_dim;
        for (int p = 0; p < n_pos; p++) {
            if (fread(vcache_l + (size_t)p * kv_dim, 1, row_size, f) != row_size) {
                fclose(f);
                return 0;
            }
        }
    }

    fclose(f);
    fprintf(stderr, "KV cache loaded: %d positions from %s\n", n_pos, path);
    return n_pos;
}

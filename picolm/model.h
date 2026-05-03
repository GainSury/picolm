#ifndef MODEL_H
#define MODEL_H

#include "quant.h"
#include <stdint.h>
#include <stddef.h>

#define GGUF_MAGIC 0x46554747
#define MAX_LAYERS 64

/* Magic for KV cache files */
#define KVCACHE_MAGIC 0x4B564350  /* "KVCP" */

/* ---- Configuration ---- */

typedef struct {
    int n_embd;         /* embedding dimension (e.g. 2048) */
    int n_ffn;          /* feed-forward hidden size (e.g. 5632) */
    int n_heads;        /* number of attention heads (e.g. 32) */
    int n_kv_heads;     /* number of KV heads for GQA (e.g. 4) */
    int n_layers;       /* number of transformer layers (e.g. 22) */
    int vocab_size;     /* vocabulary size (e.g. 32000) */
    int max_seq_len;    /* maximum sequence length (e.g. 2048) */
    int head_dim;       /* = n_embd / n_heads */
    float rope_freq_base; /* RoPE theta base (e.g. 10000.0) */
    int alignment;      /* GGUF data alignment */
    gguf_type_t weight_type; /* default weight quantization type */

    /* SSM/GDN config for hybrid architectures (Qwen3.5, 0 = pure transformer) */
    int ssm_d_state;          /* head dimension for K and V (= ssm_d_state) */
    int ssm_d_inner;          /* SSM inner size (= n_v_heads * ssm_d_state) */
    int ssm_d_conv;           /* depthwise conv kernel size (typically 4) */
    int ssm_dt_rank;          /* number of V heads (ssm_time_step_rank) */
    int ssm_n_group;          /* number of K heads (ssm_group_count) */
    int full_attn_interval;   /* full attn every N layers, 0 = pure transformer */
    float norm_rms_eps;       /* L2 norm epsilon used in GDN Q/K normalisation */
    int is_recurrent[MAX_LAYERS]; /* 1 = GDN layer, 0 = full transformer attention */
} model_config_t;

/* ---- Per-layer weight pointers (into mmap) ---- */

typedef struct {
    const void *attn_norm;
    const void *attn_q;
    const void *attn_k;
    const void *attn_v;
    const void *attn_output;
    const void *ffn_norm;
    const void *ffn_gate;
    const void *ffn_down;
    const void *ffn_up;
    /* Optional Gemma 4 weights (NULL for other architectures) */
    const void *post_attn_norm;  /* post-attention normalization */
    const void *post_ffn_norm;   /* post-FFN normalization */
    const void *attn_q_norm;     /* per-head Q normalization (head_dim elements) */
    const void *attn_k_norm;     /* per-head K normalization (head_dim elements) */
    /* SSM/GDN weights for Qwen3.5 recurrent layers (NULL for other architectures) */
    const void *wqkv;       /* fused QKV input projection [n_embd → key_dim*2+value_dim] */
    const void *wqkv_gate;  /* gate z projection [n_embd → value_dim] */
    const void *ssm_alpha;  /* alpha weight [n_embd → n_v_heads] */
    const void *ssm_beta;   /* beta weight  [n_embd → n_v_heads] */
    const void *ssm_dt;     /* DT bias  [n_v_heads] */
    const void *ssm_a;      /* A parameter [n_v_heads] */
    const void *ssm_conv1d; /* depthwise conv kernel [d_conv, conv_dim] */
    const void *ssm_norm;   /* GDN output norm weight [head_v_dim] */
    const void *ssm_out;    /* GDN output projection [value_dim → n_embd] */
    /* Per-tensor quantization types */
    gguf_type_t type_attn_norm;
    gguf_type_t type_attn_q;
    gguf_type_t type_attn_k;
    gguf_type_t type_attn_v;
    gguf_type_t type_attn_output;
    gguf_type_t type_ffn_norm;
    gguf_type_t type_ffn_gate;
    gguf_type_t type_ffn_down;
    gguf_type_t type_ffn_up;
    gguf_type_t type_post_attn_norm;
    gguf_type_t type_post_ffn_norm;
    gguf_type_t type_attn_q_norm;
    gguf_type_t type_attn_k_norm;
    gguf_type_t type_wqkv;
    gguf_type_t type_wqkv_gate;
    gguf_type_t type_ssm_alpha;
    gguf_type_t type_ssm_beta;
    gguf_type_t type_ssm_dt;
    gguf_type_t type_ssm_a;
    gguf_type_t type_ssm_conv1d;
    gguf_type_t type_ssm_norm;
    gguf_type_t type_ssm_out;
} layer_weights_t;

typedef struct {
    const void *token_embd;
    gguf_type_t type_token_embd;
    const void *output_norm;
    gguf_type_t type_output_norm;
    const void *output;        /* final output projection (may alias token_embd) */
    gguf_type_t type_output;
    layer_weights_t layers[MAX_LAYERS];
} model_weights_t;

/* ---- Runtime state (pre-allocated buffers) ---- */

typedef struct {
    float *x;            /* current activation [n_embd] */
    float *xb;           /* buffer after norm / attention output [n_embd] */
    float *xb2;          /* second buffer [n_embd] */
    float *q;            /* query vector [n_embd] */
    /* att buffer REMOVED — flash attention uses online softmax */
    float *hb;           /* FFN hidden buffer [n_ffn] */
    float *hb2;          /* FFN hidden buffer 2 [n_ffn] */
    float *logits;       /* output logits [vocab_size] */

    /* KV cache stored as FP16 to halve memory (22 MB -> 11 MB for TinyLlama) */
    uint16_t *key_cache;    /* [n_layers * max_seq_len * n_kv_heads * head_dim] as FP16 */
    uint16_t *val_cache;    /* [n_layers * max_seq_len * n_kv_heads * head_dim] as FP16 */

    float *dequant_scratch; /* scratch for matmul dequant [max(n_embd, n_ffn)] */

    /* Pre-computed RoPE cos/sin tables [max_seq_len * head_dim/2] */
    float *rope_cos;
    float *rope_sin;

    /* Pre-dequantized norm weights (small, keep in RAM) */
    float *norm_weights;
    float *attn_norm_w[MAX_LAYERS];
    float *ffn_norm_w[MAX_LAYERS];
    float *output_norm_w;
    /* Optional Gemma 4 norm weights (NULL for other architectures) */
    float *post_attn_norm_w[MAX_LAYERS];
    float *post_ffn_norm_w[MAX_LAYERS];
    float *attn_q_norm_w[MAX_LAYERS];  /* head_dim elements, shared across heads */
    float *attn_k_norm_w[MAX_LAYERS];  /* head_dim elements, shared across KV heads */

    /* SSM recurrent states for Qwen3.5 (NULL if not a hybrid model).
     * Flat arrays; helper macros index by layer.
     *   ssm_states  : [n_layers * n_v_heads * head_dim * head_dim] (zero-init = clear state)
     *   conv_states : [n_layers * (d_conv-1) * conv_dim] */
    float *ssm_states;
    float *conv_states;
    /* Pre-dequantized small SSM weights per layer (flat, indexed by layer index):
     *   ssm_norm_w  : [n_layers * head_dim]
     *   ssm_dt_w    : [n_layers * n_v_heads]
     *   ssm_a_w     : [n_layers * n_v_heads]
     *   ssm_conv1d_w: [n_layers * conv_dim * d_conv] */
    float *ssm_norm_w;
    float *ssm_dt_w;
    float *ssm_a_w;
    float *ssm_conv1d_w;
    void  *ssm_block;    /* single calloc for all four SSM arrays above */
    size_t ssm_block_size;

    /* Single allocation base */
    void *mem_block;
    size_t mem_size;

    /* Separate allocation for FP16 KV cache */
    void *kv_block;
    size_t kv_size;
} run_state_t;

/* ---- Model ---- */

typedef struct {
    model_config_t  config;
    model_weights_t weights;
    run_state_t     state;

    /* mmap bookkeeping */
    void  *mmap_addr;
    size_t mmap_size;
#ifdef _WIN32
    void  *file_handle;
    void  *map_handle;
#else
    int    fd;
#endif

    /* Tokenizer data offsets (filled by GGUF parser, used by tokenizer_load) */
    const void *tok_tokens_data;
    uint64_t    tok_n_tokens;
    const void *tok_scores_data;
    uint64_t    tok_n_scores;
    uint32_t    tok_bos_id;
    uint32_t    tok_eos_id;
    /* Tokenizer type: 0 = SentencePiece/llama (▁), 1 = GPT2/tiktoken (Ġ) */
    int         tok_model_type;
} model_t;

/* Load a GGUF model file. Returns 0 on success. */
int model_load(model_t *m, const char *path, int max_seq_len);

/* Run one forward pass. Returns pointer to logits[vocab_size]. */
float *model_forward(model_t *m, int token, int pos);

/* Free all resources. */
void model_free(model_t *m);

/* ---- KV cache persistence ---- */

/* Save KV cache state for positions [0, n_pos) to a file.
 * Returns 0 on success. */
int kvcache_save(const model_t *m, const char *path, int n_pos);

/* Load KV cache state from a file. Returns the number of positions
 * loaded (0 on failure). Caller should start generation from this position. */
int kvcache_load(model_t *m, const char *path);

#endif /* MODEL_H */

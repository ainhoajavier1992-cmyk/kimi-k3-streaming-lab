#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define K3W_HEADER_BYTES 4096
#define K3W_MAGIC "K3STREAM_EXPERTS_V1\n"

typedef struct {
    int layers;
    int experts;
    int hidden;
    int inter;
    int topk;
    int qbits;
    int norm_topk;
    unsigned seed;
    float routed_scale;
    int64_t expert_bytes;
} K3Config;

typedef struct {
    const uint8_t *q;
    const uint8_t *scales;
    int rows;
    int cols;
    int row_bytes;
} Q4Matrix;

typedef struct {
    int eid;
    uint64_t used;
    uint8_t *slab;
    size_t slab_cap;
    Q4Matrix gate;
    Q4Matrix up;
    Q4Matrix down;
} ExpertSlot;

typedef struct {
    K3Config cfg;
    int fd;
    float *router_w;
    float *router_b;
    uint64_t *usage;
    ExpertSlot *slots;
    int *counts;
    int cache_cap;
    uint64_t clock;
    uint64_t requests;
    uint64_t hits;
    uint64_t misses;
    uint64_t bytes_read;
} Runtime;

typedef struct {
    uint64_t s;
} Rng;

static void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

static void die_errno(const char *msg) {
    fprintf(stderr, "%s: %s\n", msg, strerror(errno));
    exit(1);
}

static int64_t q4_matrix_bytes(int rows, int cols) {
    return (int64_t)rows * ((cols + 1) / 2) + (int64_t)rows * 4;
}

static int64_t expert_bytes_for(const K3Config *cfg) {
    return q4_matrix_bytes(cfg->inter, cfg->hidden) * 2 +
           q4_matrix_bytes(cfg->hidden, cfg->inter);
}

static uint64_t rng_u64(Rng *r) {
    uint64_t x = r->s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    r->s = x;
    return x * 2685821657736338717ULL;
}

static float rng_f32(Rng *r) {
    uint32_t v = (uint32_t)(rng_u64(r) >> 40);
    return ((float)v / (float)0xFFFFFFu) * 2.0f - 1.0f;
}

static float silu(float x) {
    return x / (1.0f + expf(-x));
}

static float sigmoidf_local(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    float z = expf(x);
    return z / (1.0f + z);
}

static void read_full_fd(int fd, void *buf, size_t n, off_t off, const char *tag) {
    uint8_t *p = (uint8_t *)buf;
    size_t got = 0;
    while (got < n) {
        ssize_t r = pread(fd, p + got, n - got, off + (off_t)got);
        if (r < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "%s: %s\n", tag, strerror(errno));
            exit(1);
        }
        if (r == 0) {
            fprintf(stderr, "%s: short read at EOF\n", tag);
            exit(1);
        }
        got += (size_t)r;
    }
}

static void write_full(FILE *f, const void *buf, size_t n, const char *tag) {
    if (n == 0) return;
    if (fwrite(buf, 1, n, f) != n) {
        fprintf(stderr, "%s: write failed\n", tag);
        exit(1);
    }
}

static void path_join(char *out, size_t cap, const char *a, const char *b) {
    int n = snprintf(out, cap, "%s/%s", a, b);
    if (n < 0 || (size_t)n >= cap) die("path too long");
}

static int parse_int_value(const char *s, int *out) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (end == s || *end != 0 || v < 0 || v > 1000000000L) return 0;
    *out = (int)v;
    return 1;
}

static int parse_float_value(const char *s, float *out) {
    char *end = NULL;
    float v = strtof(s, &end);
    if (end == s || *end != 0) return 0;
    *out = v;
    return 1;
}

static void cfg_defaults(K3Config *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->layers = 4;
    cfg->experts = 16;
    cfg->hidden = 32;
    cfg->inter = 64;
    cfg->topk = 2;
    cfg->qbits = 4;
    cfg->norm_topk = 1;
    cfg->seed = 1;
    cfg->routed_scale = 1.0f;
}

static void cfg_validate(K3Config *cfg) {
    if (cfg->layers < 1 || cfg->layers > 256) die("layers outside supported range");
    if (cfg->experts < 1 || cfg->experts > 8192) die("experts outside supported range");
    if (cfg->hidden < 1 || cfg->hidden > 262144) die("hidden outside supported range");
    if (cfg->inter < 1 || cfg->inter > 262144) die("inter outside supported range");
    if (cfg->topk < 1 || cfg->topk > cfg->experts || cfg->topk > 64) die("topk outside supported range");
    if (cfg->qbits != 4) die("only qbits=4 fixture/runtime is implemented now");
    cfg->expert_bytes = expert_bytes_for(cfg);
}

static void cfg_write(const char *dir, const K3Config *cfg) {
    char path[PATH_MAX];
    path_join(path, sizeof(path), dir, "k3stream.cfg");
    FILE *f = fopen(path, "wb");
    if (!f) die_errno(path);
    fprintf(f, "format=k3stream-fixture-v1\n");
    fprintf(f, "layers=%d\n", cfg->layers);
    fprintf(f, "experts=%d\n", cfg->experts);
    fprintf(f, "hidden=%d\n", cfg->hidden);
    fprintf(f, "inter=%d\n", cfg->inter);
    fprintf(f, "topk=%d\n", cfg->topk);
    fprintf(f, "qbits=%d\n", cfg->qbits);
    fprintf(f, "norm_topk=%d\n", cfg->norm_topk);
    fprintf(f, "seed=%u\n", cfg->seed);
    fprintf(f, "routed_scale=%.9g\n", cfg->routed_scale);
    fprintf(f, "expert_bytes=%lld\n", (long long)cfg->expert_bytes);
    fclose(f);
}

static void cfg_read(const char *dir, K3Config *cfg) {
    cfg_defaults(cfg);
    char path[PATH_MAX];
    path_join(path, sizeof(path), dir, "k3stream.cfg");
    FILE *f = fopen(path, "rb");
    if (!f) die_errno(path);
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = 0;
        char *eq = strchr(line, '=');
        if (!eq) continue;
        *eq = 0;
        const char *key = line;
        const char *val = eq + 1;
        int iv = 0;
        float fv = 0.0f;
        if (!strcmp(key, "layers") && parse_int_value(val, &iv)) cfg->layers = iv;
        else if (!strcmp(key, "experts") && parse_int_value(val, &iv)) cfg->experts = iv;
        else if (!strcmp(key, "hidden") && parse_int_value(val, &iv)) cfg->hidden = iv;
        else if (!strcmp(key, "inter") && parse_int_value(val, &iv)) cfg->inter = iv;
        else if (!strcmp(key, "topk") && parse_int_value(val, &iv)) cfg->topk = iv;
        else if (!strcmp(key, "qbits") && parse_int_value(val, &iv)) cfg->qbits = iv;
        else if (!strcmp(key, "norm_topk") && parse_int_value(val, &iv)) cfg->norm_topk = iv;
        else if (!strcmp(key, "seed") && parse_int_value(val, &iv)) cfg->seed = (unsigned)iv;
        else if (!strcmp(key, "routed_scale") && parse_float_value(val, &fv)) cfg->routed_scale = fv;
    }
    fclose(f);
    cfg_validate(cfg);
}

static void q4_write_matrix(FILE *f, int rows, int cols, Rng *rng) {
    int row_bytes = (cols + 1) / 2;
    uint8_t *packed = (uint8_t *)calloc((size_t)rows * (size_t)row_bytes, 1);
    float *scales = (float *)calloc((size_t)rows, sizeof(float));
    float *row = (float *)malloc((size_t)cols * sizeof(float));
    if (!packed || !scales || !row) die("OOM while generating fixture matrix");

    for (int r = 0; r < rows; r++) {
        float amax = 0.0f;
        for (int c = 0; c < cols; c++) {
            float w = rng_f32(rng) * 0.25f;
            row[c] = w;
            float aw = fabsf(w);
            if (aw > amax) amax = aw;
        }
        float s = amax > 1e-8f ? amax / 7.0f : 1e-8f;
        scales[r] = s;
        for (int c = 0; c < cols; c++) {
            int q = (int)lrintf(row[c] / s);
            if (q < -8) q = -8;
            if (q > 7) q = 7;
            uint8_t nib = (uint8_t)(q + 8);
            uint8_t *dst = &packed[(size_t)r * (size_t)row_bytes + (size_t)(c / 2)];
            if ((c & 1) == 0) *dst = (uint8_t)((*dst & 0xF0u) | nib);
            else *dst = (uint8_t)((*dst & 0x0Fu) | (uint8_t)(nib << 4));
        }
    }

    write_full(f, packed, (size_t)rows * (size_t)row_bytes, "expert q4 weights");
    write_full(f, scales, (size_t)rows * sizeof(float), "expert q4 scales");
    free(packed);
    free(scales);
    free(row);
}

static void write_router(const char *dir, const K3Config *cfg, Rng *rng) {
    char path[PATH_MAX];
    path_join(path, sizeof(path), dir, "router.bin");
    FILE *f = fopen(path, "wb");
    if (!f) die_errno(path);
    int64_t n_w = (int64_t)cfg->layers * cfg->experts * cfg->hidden;
    for (int64_t i = 0; i < n_w; i++) {
        float v = rng_f32(rng) * 0.20f;
        write_full(f, &v, sizeof(v), "router weights");
    }
    int64_t n_b = (int64_t)cfg->layers * cfg->experts;
    for (int64_t i = 0; i < n_b; i++) {
        float v = rng_f32(rng) * 0.01f;
        write_full(f, &v, sizeof(v), "router bias");
    }
    fclose(f);
}

static void write_experts(const char *dir, const K3Config *cfg, Rng *rng) {
    char path[PATH_MAX];
    path_join(path, sizeof(path), dir, "experts.k3w");
    FILE *f = fopen(path, "wb");
    if (!f) die_errno(path);
    uint8_t hdr[K3W_HEADER_BYTES];
    memset(hdr, 0, sizeof(hdr));
    memcpy(hdr, K3W_MAGIC, strlen(K3W_MAGIC));
    write_full(f, hdr, sizeof(hdr), "expert header");
    for (int l = 0; l < cfg->layers; l++) {
        for (int e = 0; e < cfg->experts; e++) {
            (void)l;
            (void)e;
            q4_write_matrix(f, cfg->inter, cfg->hidden, rng);
            q4_write_matrix(f, cfg->inter, cfg->hidden, rng);
            q4_write_matrix(f, cfg->hidden, cfg->inter, rng);
        }
    }
    fclose(f);
}

static const char *arg_value(int *i, int argc, char **argv, const char *name) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "%s needs a value\n", name);
        exit(2);
    }
    (*i)++;
    return argv[*i];
}

static int cmd_fixture(int argc, char **argv) {
    K3Config cfg;
    cfg_defaults(&cfg);
    const char *out = NULL;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--out")) out = arg_value(&i, argc, argv, "--out");
        else if (!strcmp(argv[i], "--layers")) cfg.layers = atoi(arg_value(&i, argc, argv, "--layers"));
        else if (!strcmp(argv[i], "--experts")) cfg.experts = atoi(arg_value(&i, argc, argv, "--experts"));
        else if (!strcmp(argv[i], "--hidden")) cfg.hidden = atoi(arg_value(&i, argc, argv, "--hidden"));
        else if (!strcmp(argv[i], "--inter")) cfg.inter = atoi(arg_value(&i, argc, argv, "--inter"));
        else if (!strcmp(argv[i], "--topk")) cfg.topk = atoi(arg_value(&i, argc, argv, "--topk"));
        else if (!strcmp(argv[i], "--seed")) cfg.seed = (unsigned)atoi(arg_value(&i, argc, argv, "--seed"));
        else {
            fprintf(stderr, "unknown fixture arg: %s\n", argv[i]);
            return 2;
        }
    }
    if (!out) die("fixture needs --out DIR");
    cfg_validate(&cfg);
    if (mkdir(out, 0775) != 0 && errno != EEXIST) die_errno(out);
    Rng rng = { cfg.seed ? cfg.seed : 1 };
    cfg_write(out, &cfg);
    write_router(out, &cfg, &rng);
    write_experts(out, &cfg, &rng);
    printf("fixture written: %s\n", out);
    printf("layers=%d experts=%d topk=%d hidden=%d inter=%d expert_bytes=%.2f KB\n",
           cfg.layers, cfg.experts, cfg.topk, cfg.hidden, cfg.inter, cfg.expert_bytes / 1024.0);
    return 0;
}

static void matrix_bind(Q4Matrix *m, const uint8_t **p, int rows, int cols) {
    int rb = (cols + 1) / 2;
    m->rows = rows;
    m->cols = cols;
    m->row_bytes = rb;
    m->q = *p;
    *p += (size_t)rows * (size_t)rb;
    m->scales = *p;
    *p += (size_t)rows * 4u;
}

static void expert_bind(Runtime *rt, ExpertSlot *slot) {
    const uint8_t *p = slot->slab;
    matrix_bind(&slot->gate, &p, rt->cfg.inter, rt->cfg.hidden);
    matrix_bind(&slot->up, &p, rt->cfg.inter, rt->cfg.hidden);
    matrix_bind(&slot->down, &p, rt->cfg.hidden, rt->cfg.inter);
}

static void runtime_open(Runtime *rt, const char *model, int cache_cap) {
    memset(rt, 0, sizeof(*rt));
    cfg_read(model, &rt->cfg);
    if (cache_cap < 1) cache_cap = rt->cfg.topk;
    rt->cache_cap = cache_cap;

    char path[PATH_MAX];
    path_join(path, sizeof(path), model, "experts.k3w");
    rt->fd = open(path, O_RDONLY);
    if (rt->fd < 0) die_errno(path);
    char magic[sizeof(K3W_MAGIC)];
    memset(magic, 0, sizeof(magic));
    read_full_fd(rt->fd, magic, strlen(K3W_MAGIC), 0, "expert header");
    if (memcmp(magic, K3W_MAGIC, strlen(K3W_MAGIC)) != 0) die("experts.k3w has bad magic");

    path_join(path, sizeof(path), model, "router.bin");
    FILE *rf = fopen(path, "rb");
    if (!rf) die_errno(path);
    int64_t nw = (int64_t)rt->cfg.layers * rt->cfg.experts * rt->cfg.hidden;
    int64_t nb = (int64_t)rt->cfg.layers * rt->cfg.experts;
    rt->router_w = (float *)malloc((size_t)nw * sizeof(float));
    rt->router_b = (float *)malloc((size_t)nb * sizeof(float));
    rt->usage = (uint64_t *)calloc((size_t)nb, sizeof(uint64_t));
    if (!rt->router_w || !rt->router_b) die("OOM loading router");
    if (!rt->usage) die("OOM allocating usage counters");
    if (fread(rt->router_w, sizeof(float), (size_t)nw, rf) != (size_t)nw) die("short router weight read");
    if (fread(rt->router_b, sizeof(float), (size_t)nb, rf) != (size_t)nb) die("short router bias read");
    fclose(rf);

    rt->slots = (ExpertSlot *)calloc((size_t)rt->cfg.layers * (size_t)cache_cap, sizeof(ExpertSlot));
    rt->counts = (int *)calloc((size_t)rt->cfg.layers, sizeof(int));
    if (!rt->slots || !rt->counts) die("OOM allocating cache");
    for (int i = 0; i < rt->cfg.layers * cache_cap; i++) rt->slots[i].eid = -1;
}

static void runtime_close(Runtime *rt) {
    if (rt->fd >= 0) close(rt->fd);
    int nslots = rt->cfg.layers * rt->cache_cap;
    for (int i = 0; i < nslots; i++) free(rt->slots[i].slab);
    free(rt->slots);
    free(rt->counts);
    free(rt->router_w);
    free(rt->router_b);
    free(rt->usage);
}

static ExpertSlot *load_expert(Runtime *rt, int layer, int eid) {
    ExpertSlot *base = &rt->slots[layer * rt->cache_cap];
    int *count = &rt->counts[layer];
    rt->requests++;
    for (int i = 0; i < *count; i++) {
        if (base[i].eid == eid) {
            rt->hits++;
            base[i].used = ++rt->clock;
            return &base[i];
        }
    }

    rt->misses++;
    int idx = 0;
    if (*count < rt->cache_cap) {
        idx = (*count)++;
    } else {
        for (int i = 1; i < *count; i++) {
            if (base[i].used < base[idx].used) idx = i;
        }
    }
    ExpertSlot *slot = &base[idx];
    if (!slot->slab || slot->slab_cap < (size_t)rt->cfg.expert_bytes) {
        free(slot->slab);
        slot->slab = (uint8_t *)malloc((size_t)rt->cfg.expert_bytes);
        if (!slot->slab) die("OOM expert slab");
        slot->slab_cap = (size_t)rt->cfg.expert_bytes;
    }
    int64_t ordinal = (int64_t)layer * rt->cfg.experts + eid;
    off_t off = (off_t)(K3W_HEADER_BYTES + ordinal * rt->cfg.expert_bytes);
    read_full_fd(rt->fd, slot->slab, (size_t)rt->cfg.expert_bytes, off, "expert load");
    rt->bytes_read += (uint64_t)rt->cfg.expert_bytes;
    slot->eid = eid;
    slot->used = ++rt->clock;
    expert_bind(rt, slot);
    return slot;
}

static float q4_scale_at(const Q4Matrix *m, int row) {
    float s;
    memcpy(&s, m->scales + (size_t)row * 4u, sizeof(float));
    return s;
}

static void q4_matvec(const Q4Matrix *m, const float *x, float *y) {
    for (int r = 0; r < m->rows; r++) {
        const uint8_t *row = m->q + (size_t)r * (size_t)m->row_bytes;
        float acc = 0.0f;
        for (int c = 0; c < m->cols; c++) {
            uint8_t byte = row[c >> 1];
            int q = ((c & 1) == 0) ? (int)(byte & 0x0Fu) : (int)(byte >> 4);
            q -= 8;
            acc += (float)q * x[c];
        }
        y[r] = acc * q4_scale_at(m, r);
    }
}

static void expert_forward(const Runtime *rt, const ExpertSlot *slot, const float *x,
                           float *tmp_g, float *tmp_u, float *tmp_h, float *out) {
    q4_matvec(&slot->gate, x, tmp_g);
    q4_matvec(&slot->up, x, tmp_u);
    for (int i = 0; i < rt->cfg.inter; i++) tmp_h[i] = silu(tmp_g[i]) * tmp_u[i];
    q4_matvec(&slot->down, tmp_h, out);
}

static void route_topk(Runtime *rt, int layer, const float *x, int *idx, float *w) {
    K3Config *cfg = &rt->cfg;
    float *scores = (float *)malloc((size_t)cfg->experts * sizeof(float));
    if (!scores) die("OOM route scores");
    const float *rw = rt->router_w + (int64_t)layer * cfg->experts * cfg->hidden;
    const float *rb = rt->router_b + (int64_t)layer * cfg->experts;
    for (int e = 0; e < cfg->experts; e++) {
        const float *we = rw + (int64_t)e * cfg->hidden;
        float dot = rb[e];
        for (int d = 0; d < cfg->hidden; d++) dot += we[d] * x[d];
        scores[e] = sigmoidf_local(dot);
    }
    for (int k = 0; k < cfg->topk; k++) {
        int best = -1;
        float bv = -1.0f;
        for (int e = 0; e < cfg->experts; e++) {
            int used = 0;
            for (int j = 0; j < k; j++) {
                if (idx[j] == e) {
                    used = 1;
                    break;
                }
            }
            if (!used && scores[e] > bv) {
                best = e;
                bv = scores[e];
            }
        }
        idx[k] = best;
        w[k] = bv;
        rt->usage[(int64_t)layer * cfg->experts + best]++;
    }
    if (cfg->norm_topk) {
        float sum = 1e-20f;
        for (int k = 0; k < cfg->topk; k++) sum += w[k];
        for (int k = 0; k < cfg->topk; k++) w[k] = (w[k] / sum) * cfg->routed_scale;
    } else {
        for (int k = 0; k < cfg->topk; k++) w[k] *= cfg->routed_scale;
    }
    free(scores);
}

static void usage_save(Runtime *rt, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) die_errno(path);
    for (int l = 0; l < rt->cfg.layers; l++) {
        for (int e = 0; e < rt->cfg.experts; e++) {
            uint64_t c = rt->usage[(int64_t)l * rt->cfg.experts + e];
            if (c) fprintf(f, "%d %d %llu\n", l, e, (unsigned long long)c);
        }
    }
    fclose(f);
}

static void pin_from_usage(Runtime *rt, const char *path, int per_layer) {
    if (per_layer <= 0) return;
    if (per_layer > rt->cache_cap) per_layer = rt->cache_cap;
    int layers = rt->cfg.layers;
    int experts = rt->cfg.experts;
    int *best_e = (int *)malloc((size_t)layers * (size_t)per_layer * sizeof(int));
    uint64_t *best_c = (uint64_t *)calloc((size_t)layers * (size_t)per_layer, sizeof(uint64_t));
    if (!best_e || !best_c) die("OOM pin tables");
    for (int i = 0; i < layers * per_layer; i++) best_e[i] = -1;

    FILE *f = fopen(path, "rb");
    if (!f) die_errno(path);
    int l = 0;
    int e = 0;
    unsigned long long c = 0;
    while (fscanf(f, "%d %d %llu", &l, &e, &c) == 3) {
        if (l < 0 || l >= layers || e < 0 || e >= experts || c == 0) continue;
        int base = l * per_layer;
        for (int i = 0; i < per_layer; i++) {
            if ((uint64_t)c > best_c[base + i]) {
                for (int j = per_layer - 1; j > i; j--) {
                    best_c[base + j] = best_c[base + j - 1];
                    best_e[base + j] = best_e[base + j - 1];
                }
                best_c[base + i] = (uint64_t)c;
                best_e[base + i] = e;
                break;
            }
        }
    }
    fclose(f);

    int loaded = 0;
    for (l = 0; l < layers; l++) {
        for (int i = 0; i < per_layer; i++) {
            e = best_e[l * per_layer + i];
            if (e >= 0) {
                (void)load_expert(rt, l, e);
                loaded++;
            }
        }
    }
    printf("pin loaded=%d from=%s per_layer=%d\n", loaded, path, per_layer);
    rt->requests = 0;
    rt->hits = 0;
    rt->misses = 0;
    rt->bytes_read = 0;
    free(best_e);
    free(best_c);
}

static void init_token_state(float *x, int hidden, int token) {
    for (int d = 0; d < hidden; d++) {
        float a = (float)((token + 1) * (d + 3));
        x[d] = sinf(a * 0.017f) * 0.5f + cosf(a * 0.011f) * 0.25f;
    }
}

static int cmd_run(int argc, char **argv) {
    const char *model = NULL;
    int tokens = 8;
    int cache = 0;
    int trace = 0;
    const char *stats_out = NULL;
    const char *pin_path = NULL;
    int pin_per_layer = 0;
    int batch = 1;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--model")) model = arg_value(&i, argc, argv, "--model");
        else if (!strcmp(argv[i], "--tokens")) tokens = atoi(arg_value(&i, argc, argv, "--tokens"));
        else if (!strcmp(argv[i], "--cache")) cache = atoi(arg_value(&i, argc, argv, "--cache"));
        else if (!strcmp(argv[i], "--batch")) batch = atoi(arg_value(&i, argc, argv, "--batch"));
        else if (!strcmp(argv[i], "--trace")) trace = 1;
        else if (!strcmp(argv[i], "--stats-out")) stats_out = arg_value(&i, argc, argv, "--stats-out");
        else if (!strcmp(argv[i], "--pin")) pin_path = arg_value(&i, argc, argv, "--pin");
        else if (!strcmp(argv[i], "--pin-per-layer")) pin_per_layer = atoi(arg_value(&i, argc, argv, "--pin-per-layer"));
        else {
            fprintf(stderr, "unknown run arg: %s\n", argv[i]);
            return 2;
        }
    }
    if (!model) die("run needs --model DIR");
    Runtime rt;
    runtime_open(&rt, model, cache);
    if (pin_path) pin_from_usage(&rt, pin_path, pin_per_layer ? pin_per_layer : rt.cfg.topk);
    K3Config *cfg = &rt.cfg;
    if (batch < 1) batch = 1;
    if (batch > 256) die("batch outside supported range");
    float *x = (float *)malloc((size_t)batch * (size_t)cfg->hidden * sizeof(float));
    float *out = (float *)malloc((size_t)cfg->hidden * sizeof(float));
    float *acc = (float *)malloc((size_t)batch * (size_t)cfg->hidden * sizeof(float));
    float *tmp_g = (float *)malloc((size_t)cfg->inter * sizeof(float));
    float *tmp_u = (float *)malloc((size_t)cfg->inter * sizeof(float));
    float *tmp_h = (float *)malloc((size_t)cfg->inter * sizeof(float));
    int *idx = (int *)malloc((size_t)batch * (size_t)cfg->topk * sizeof(int));
    float *w = (float *)malloc((size_t)batch * (size_t)cfg->topk * sizeof(float));
    int *uniq = (int *)malloc((size_t)cfg->experts * sizeof(int));
    uint8_t *seen = (uint8_t *)malloc((size_t)cfg->experts);
    ExpertSlot **resident = (ExpertSlot **)calloc((size_t)cfg->experts, sizeof(ExpertSlot *));
    if (!x || !out || !acc || !tmp_g || !tmp_u || !tmp_h || !idx || !w || !uniq || !seen || !resident) die("OOM run buffers");

    for (int base_tok = 0; base_tok < tokens; base_tok += batch) {
        int rows = tokens - base_tok;
        if (rows > batch) rows = batch;
        for (int s = 0; s < rows; s++) {
            init_token_state(x + (int64_t)s * cfg->hidden, cfg->hidden, base_tok + s);
        }
        for (int l = 0; l < cfg->layers; l++) {
            memset(acc, 0, (size_t)rows * (size_t)cfg->hidden * sizeof(float));
            memset(seen, 0, (size_t)cfg->experts);
            int nu = 0;
            for (int s = 0; s < rows; s++) {
                int *row_idx = idx + (int64_t)s * cfg->topk;
                float *row_w = w + (int64_t)s * cfg->topk;
                route_topk(&rt, l, x + (int64_t)s * cfg->hidden, row_idx, row_w);
                if (trace) {
                    printf("token=%d layer=%d route", base_tok + s, l);
                    for (int k = 0; k < cfg->topk; k++) printf(" %d:%.4f", row_idx[k], row_w[k]);
                    printf("\n");
                }
                for (int k = 0; k < cfg->topk; k++) {
                    int eid = row_idx[k];
                    if (!seen[eid]) {
                        seen[eid] = 1;
                        uniq[nu++] = eid;
                    }
                }
            }
            if (rt.cache_cap >= nu) {
                memset(resident, 0, (size_t)cfg->experts * sizeof(ExpertSlot *));
                for (int u = 0; u < nu; u++) resident[uniq[u]] = load_expert(&rt, l, uniq[u]);
                for (int s = 0; s < rows; s++) {
                    int *row_idx = idx + (int64_t)s * cfg->topk;
                    float *row_w = w + (int64_t)s * cfg->topk;
                    for (int k = 0; k < cfg->topk; k++) {
                        ExpertSlot *slot = resident[row_idx[k]];
                        expert_forward(&rt, slot, x + (int64_t)s * cfg->hidden, tmp_g, tmp_u, tmp_h, out);
                        for (int d = 0; d < cfg->hidden; d++) {
                            acc[(int64_t)s * cfg->hidden + d] += row_w[k] * out[d];
                        }
                    }
                }
            } else {
                for (int s = 0; s < rows; s++) {
                    int *row_idx = idx + (int64_t)s * cfg->topk;
                    float *row_w = w + (int64_t)s * cfg->topk;
                    for (int k = 0; k < cfg->topk; k++) {
                        ExpertSlot *slot = load_expert(&rt, l, row_idx[k]);
                        expert_forward(&rt, slot, x + (int64_t)s * cfg->hidden, tmp_g, tmp_u, tmp_h, out);
                        for (int d = 0; d < cfg->hidden; d++) {
                            acc[(int64_t)s * cfg->hidden + d] += row_w[k] * out[d];
                        }
                    }
                }
            }
            for (int s = 0; s < rows; s++) {
                for (int d = 0; d < cfg->hidden; d++) {
                    int64_t off = (int64_t)s * cfg->hidden + d;
                    x[off] = tanhf(x[off] + acc[off]);
                }
            }
        }
        for (int s = 0; s < rows; s++) {
            float checksum = 0.0f;
            for (int d = 0; d < cfg->hidden; d++) checksum += x[(int64_t)s * cfg->hidden + d] * (float)(d + 1);
            printf("token=%d checksum=%.8f\n", base_tok + s, checksum);
        }
    }

    printf("stats requests=%llu hits=%llu misses=%llu hit_rate=%.2f%% bytes_read=%.2f MB cache=%d/layer\n",
           (unsigned long long)rt.requests,
           (unsigned long long)rt.hits,
           (unsigned long long)rt.misses,
           rt.requests ? 100.0 * (double)rt.hits / (double)rt.requests : 0.0,
           (double)rt.bytes_read / 1000000.0,
           rt.cache_cap);
    if (stats_out) {
        usage_save(&rt, stats_out);
        printf("usage_stats=%s\n", stats_out);
    }

    free(x);
    free(out);
    free(acc);
    free(tmp_g);
    free(tmp_u);
    free(tmp_h);
    free(idx);
    free(w);
    free(uniq);
    free(seen);
    free(resident);
    runtime_close(&rt);
    return 0;
}

static int cmd_inspect(int argc, char **argv) {
    const char *model = NULL;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--model")) model = arg_value(&i, argc, argv, "--model");
        else {
            fprintf(stderr, "unknown inspect arg: %s\n", argv[i]);
            return 2;
        }
    }
    if (!model) die("inspect needs --model DIR");
    K3Config cfg;
    cfg_read(model, &cfg);
    int64_t total_experts = (int64_t)cfg.layers * cfg.experts;
    double total_bytes = (double)total_experts * (double)cfg.expert_bytes;
    printf("model=%s\n", model);
    printf("layers=%d experts/layer=%d total_experts=%lld topk=%d\n",
           cfg.layers, cfg.experts, (long long)total_experts, cfg.topk);
    printf("hidden=%d inter=%d qbits=%d expert_bytes=%.2f KB expert_store=%.2f MB\n",
           cfg.hidden, cfg.inter, cfg.qbits, cfg.expert_bytes / 1024.0, total_bytes / 1000000.0);
    printf("streaming_floor_per_token=layers*topk*expert_bytes=%.2f MB before cache hits\n",
           ((double)cfg.layers * cfg.topk * cfg.expert_bytes) / 1000000.0);
    return 0;
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage:\n"
            "  %s fixture --out DIR [--layers N --experts N --hidden N --inter N --topk N --seed N]\n"
            "  %s inspect --model DIR\n"
            "  %s run --model DIR [--tokens N --cache N --batch N --trace --stats-out FILE --pin FILE --pin-per-layer N]\n",
            argv0, argv0, argv0);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }
    if (!strcmp(argv[1], "fixture")) return cmd_fixture(argc, argv);
    if (!strcmp(argv[1], "run")) return cmd_run(argc, argv);
    if (!strcmp(argv[1], "inspect")) return cmd_inspect(argc, argv);
    usage(argv[0]);
    return 2;
}

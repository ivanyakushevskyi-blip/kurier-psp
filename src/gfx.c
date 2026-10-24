#include "gfx.h"
#include <math.h>
#include <string.h>

static uint32_t __attribute__((aligned(16))) clut[256];

#ifndef HOST
/* ================= PSP: графічний чип ================= */
#include <pspkernel.h>
#include <pspgu.h>

static unsigned int __attribute__((aligned(16))) dlist[256 * 1024 / 4];

typedef struct { float u, v; float x, y, z; } VT;
typedef struct { float u, v; uint32_t c; float x, y, z; } VCT;
typedef struct { uint32_t c; float x, y, z; } VC;

static int pow2(int n) { int p = 1; while (p < n) p <<= 1; return p; }

void gfx_init(const uint32_t *pal) {
    memcpy(clut, pal, sizeof clut);
    clut[0] = 0;                                         /* індекс 0 — прозорий */
    sceKernelDcacheWritebackRange(clut, sizeof clut);
    sceGuInit();
    sceGuStart(GU_DIRECT, dlist);
    sceGuDrawBuffer(GU_PSM_8888, (void *)0, 512);
    sceGuOffset(2048 - 240, 2048 - 136);
    sceGuViewport(2048, 2048, 480, 272);
    sceGuScissor(0, 0, 480, 272);
    sceGuEnable(GU_SCISSOR_TEST);
    sceGuDisable(GU_DEPTH_TEST);
    sceGuDepthMask(GU_TRUE);                             /* без запису глибини */
    sceGuDisable(GU_CULL_FACE);
    sceGuShadeModel(GU_SMOOTH);
    sceGuBlendFunc(GU_ADD, GU_SRC_ALPHA, GU_ONE_MINUS_SRC_ALPHA, 0, 0);
    sceGuTexWrap(GU_CLAMP, GU_CLAMP);
    sceGuFinish();
    sceGuSync(0, 0);
    /* показ кадрів лишається за перевіреною послідовністю в main.c (sceGuDisplay не викликаємо) */
}

void gfx_begin(void *target) {
    sceKernelDcacheWritebackAll();                        /* текстури, які щойно намалював процесор */
    sceGuStart(GU_DIRECT, dlist);
    sceGuDrawBufferList(GU_PSM_8888, target, 512);
    sceGuClearColor(0xFF000000);
    sceGuClear(GU_COLOR_BUFFER_BIT);
    sceGuClutMode(GU_PSM_8888, 0, 0xFF, 0);
    sceGuClutLoad(256 / 8, clut);
    sceGuTexFlush();
}

void gfx_layer8(const uint8_t *tex, int stride, int w, int h, float sx, float sy, float scale, int alpha, int linear) {
    sceGuEnable(GU_TEXTURE_2D);
    if (alpha) sceGuEnable(GU_BLEND); else sceGuDisable(GU_BLEND);
    sceGuTexMode(GU_PSM_T8, 0, 0, 0);
    sceGuTexFunc(GU_TFX_REPLACE, GU_TCC_RGBA);
    sceGuTexFilter(linear ? GU_LINEAR : GU_NEAREST, linear ? GU_LINEAR : GU_NEAREST);
    int th = pow2(h);
    /* смугами по 64 тексели (кеш текстур); вікно текстури 128, щоб згладжування на стику бачило сусіда */
    for (int x0 = 0; x0 < w; x0 += 64) {
        int sw = w - x0 < 64 ? w - x0 : 64;
        sceGuTexImage(0, 128, th, stride, tex + x0);
        sceGuTexFlush();
        VT *v = (VT *)sceGuGetMemory(2 * sizeof(VT));
        v[0].u = 0; v[0].v = 0; v[0].x = sx + x0 * scale; v[0].y = sy; v[0].z = 0;
        v[1].u = (float)sw; v[1].v = (float)h; v[1].x = sx + (x0 + sw) * scale; v[1].y = sy + h * scale; v[1].z = 0;
        sceGuDrawArray(GU_SPRITES, GU_TEXTURE_32BITF | GU_VERTEX_32BITF | GU_TRANSFORM_2D, 2, 0, v);
    }
}

void gfx_layer16(const uint16_t *tex, int stride, int w, int h, float sx, float sy, float scale, int fmt, int alpha, int linear) {
    sceGuEnable(GU_TEXTURE_2D);
    if (alpha) sceGuEnable(GU_BLEND); else sceGuDisable(GU_BLEND);
    sceGuTexMode(fmt ? GU_PSM_5551 : GU_PSM_5650, 0, 0, 0);
    sceGuTexFunc(GU_TFX_REPLACE, GU_TCC_RGBA);
    sceGuTexFilter(linear ? GU_LINEAR : GU_NEAREST, linear ? GU_LINEAR : GU_NEAREST);
    int th = pow2(h);
    for (int x0 = 0; x0 < w; x0 += 64) {
        int sw = w - x0 < 64 ? w - x0 : 64;
        sceGuTexImage(0, 128, th, stride, tex + x0);
        sceGuTexFlush();
        VT *v = (VT *)sceGuGetMemory(2 * sizeof(VT));
        v[0].u = 0; v[0].v = 0; v[0].x = sx + x0 * scale; v[0].y = sy; v[0].z = 0;
        v[1].u = (float)sw; v[1].v = (float)h; v[1].x = sx + (x0 + sw) * scale; v[1].y = sy + h * scale; v[1].z = 0;
        sceGuDrawArray(GU_SPRITES, GU_TEXTURE_32BITF | GU_VERTEX_32BITF | GU_TRANSFORM_2D, 2, 0, v);
    }
}

void gfx_quad(const uint32_t *tex, int tw, int th, float cx, float cy, float ang, float hx, float hy, uint32_t col) {
    sceGuEnable(GU_TEXTURE_2D);
    sceGuEnable(GU_BLEND);
    sceGuTexMode(GU_PSM_8888, 0, 0, 0);
    sceGuTexImage(0, tw, th, tw, tex);
    sceGuTexFunc(GU_TFX_MODULATE, GU_TCC_RGBA);
    sceGuTexFilter(GU_LINEAR, GU_LINEAR);
    sceGuTexFlush();
    float c = cosf(ang), s = sinf(ang);
    VCT *v = (VCT *)sceGuGetMemory(4 * sizeof(VCT));
    static const float K[4][2] = { { -1, -1 }, { 1, -1 }, { -1, 1 }, { 1, 1 } };
    for (int i = 0; i < 4; i++) {
        float lx = K[i][0] * hx, ly = K[i][1] * hy;
        v[i].u = K[i][0] > 0 ? (float)tw : 0; v[i].v = K[i][1] > 0 ? (float)th : 0; v[i].c = col;
        v[i].x = cx + lx * c - ly * s; v[i].y = cy + lx * s + ly * c; v[i].z = 0;
    }
    sceGuDrawArray(GU_TRIANGLE_STRIP, GU_TEXTURE_32BITF | GU_COLOR_8888 | GU_VERTEX_32BITF | GU_TRANSFORM_2D, 4, 0, v);
}

void gfx_rect(float x, float y, float w, float h, uint32_t col) {
    sceGuDisable(GU_TEXTURE_2D);
    sceGuEnable(GU_BLEND);
    VC *v = (VC *)sceGuGetMemory(2 * sizeof(VC));
    v[0].c = col; v[0].x = x; v[0].y = y; v[0].z = 0;
    v[1].c = col; v[1].x = x + w; v[1].y = y + h; v[1].z = 0;
    sceGuDrawArray(GU_SPRITES, GU_COLOR_8888 | GU_VERTEX_32BITF | GU_TRANSFORM_2D, 2, 0, v);
}

void gfx_end(void) {
    sceGuFinish();
    sceGuSync(0, 0);
}

#else
/* ================= ПК: програмна копія ================= */
static uint32_t *dst;

void gfx_init(const uint32_t *pal) { memcpy(clut, pal, sizeof clut); clut[0] = 0; }
void gfx_begin(void *target) { dst = (uint32_t *)target; for (int i = 0; i < 512 * 272; i++) dst[i] = 0xFF000000; }

static void blend(uint32_t *d, float r, float g, float b, float a) {
    if (a <= 0) return;
    if (a > 1) a = 1;
    uint32_t o = *d;
    float orr = o & 255, og = (o >> 8) & 255, ob = (o >> 16) & 255;
    int R = (int)(orr + (r - orr) * a + 0.5f), G = (int)(og + (g - og) * a + 0.5f), B = (int)(ob + (b - ob) * a + 0.5f);
    *d = 0xFF000000u | ((uint32_t)B << 16) | ((uint32_t)G << 8) | (uint32_t)R;
}
static void unpack(uint32_t c, float *p) { p[0] = c & 255; p[1] = (c >> 8) & 255; p[2] = (c >> 16) & 255; p[3] = c >> 24; }

static void fetch16(const uint16_t *t, int fmt, float *p) {
    uint16_t c = *t;
    if (fmt) { p[0] = (c & 31) * 255 / 31.0f; p[1] = ((c >> 5) & 31) * 255 / 31.0f; p[2] = ((c >> 10) & 31) * 255 / 31.0f; p[3] = (c >> 15) ? 255 : 0; }
    else { p[0] = (c & 31) * 255 / 31.0f; p[1] = ((c >> 5) & 63) * 255 / 63.0f; p[2] = (c >> 11) * 255 / 31.0f; p[3] = 255; }
}
void gfx_layer16(const uint16_t *tex, int stride, int w, int h, float sx, float sy, float scale, int fmt, int alpha, int linear) {
    int x0 = (int)floorf(sx), y0 = (int)floorf(sy), x1 = (int)ceilf(sx + w * scale), y1 = (int)ceilf(sy + h * scale);
    if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0; if (x1 > 480) x1 = 480; if (y1 > 272) y1 = 272;
    for (int y = y0; y < y1; y++) for (int x = x0; x < x1; x++) {
        float u = (x + 0.5f - sx) / scale, v = (y + 0.5f - sy) / scale, c[4];
        if (u < 0 || v < 0 || u >= w || v >= h) continue;
        if (linear) {
            float uu = u - 0.5f, vv = v - 0.5f; int iu = (int)floorf(uu), iv = (int)floorf(vv); float fu = uu - iu, fv = vv - iv;
            float acc[4] = { 0, 0, 0, 0 };
            for (int k = 0; k < 4; k++) {
                int tu = iu + (k & 1), tv = iv + (k >> 1);
                if (tu < 0) tu = 0; if (tv < 0) tv = 0; if (tu >= w) tu = w - 1; if (tv >= h) tv = h - 1;
                float wgt = ((k & 1) ? fu : 1 - fu) * ((k >> 1) ? fv : 1 - fv), p[4];
                fetch16(tex + tv * stride + tu, fmt, p);
                for (int j = 0; j < 4; j++) acc[j] += p[j] * wgt;
            }
            memcpy(c, acc, sizeof c);
        } else fetch16(tex + (int)v * stride + (int)u, fmt, c);
        blend(dst + y * 512 + x, c[0], c[1], c[2], alpha ? c[3] / 255.0f : 1);
    }
}
void gfx_layer8(const uint8_t *tex, int stride, int w, int h, float sx, float sy, float scale, int alpha, int linear) {
    int x0 = (int)floorf(sx), y0 = (int)floorf(sy), x1 = (int)ceilf(sx + w * scale), y1 = (int)ceilf(sy + h * scale);
    if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0; if (x1 > 480) x1 = 480; if (y1 > 272) y1 = 272;
    for (int y = y0; y < y1; y++) for (int x = x0; x < x1; x++) {
        float u = (x + 0.5f - sx) / scale, v = (y + 0.5f - sy) / scale;
        if (u < 0 || v < 0 || u >= w || v >= h) continue;
        float c[4];
        if (linear) {
            float uu = u - 0.5f, vv = v - 0.5f; int iu = (int)floorf(uu), iv = (int)floorf(vv); float fu = uu - iu, fv = vv - iv;
            float acc[4] = { 0, 0, 0, 0 };
            for (int k = 0; k < 4; k++) {
                int tu = iu + (k & 1), tv = iv + (k >> 1);
                if (tu < 0) tu = 0; if (tv < 0) tv = 0; if (tu >= w) tu = w - 1; if (tv >= h) tv = h - 1;
                float wgt = ((k & 1) ? fu : 1 - fu) * ((k >> 1) ? fv : 1 - fv), p[4];
                unpack(clut[tex[tv * stride + tu]], p);
                for (int j = 0; j < 4; j++) acc[j] += p[j] * wgt;
            }
            memcpy(c, acc, sizeof c);
        } else unpack(clut[tex[(int)v * stride + (int)u]], c);
        uint32_t *d = dst + y * 512 + x;
        if (alpha) blend(d, c[0], c[1], c[2], c[3] / 255.0f);
        else blend(d, c[0], c[1], c[2], 1);
    }
}

void gfx_quad(const uint32_t *tex, int tw, int th, float cx, float cy, float ang, float hx, float hy, uint32_t col) {
    float c = cosf(ang), s = sinf(ang), R = sqrtf(hx * hx + hy * hy) + 1, m[4];
    unpack(col, m);
    for (int y = (int)(cy - R); y <= (int)(cy + R); y++) for (int x = (int)(cx - R); x <= (int)(cx + R); x++) {
        if ((unsigned)x >= 480 || (unsigned)y >= 272) continue;
        float dx = x + 0.5f - cx, dy = y + 0.5f - cy, lx = dx * c + dy * s, ly = -dx * s + dy * c;
        if (lx < -hx || lx >= hx || ly < -hy || ly >= hy) continue;
        float u = (lx / hx * 0.5f + 0.5f) * tw - 0.5f, v = (ly / hy * 0.5f + 0.5f) * th - 0.5f;
        int iu = (int)floorf(u), iv = (int)floorf(v); float fu = u - iu, fv = v - iv, acc[4] = { 0, 0, 0, 0 };
        for (int k = 0; k < 4; k++) {
            int tu = iu + (k & 1), tv = iv + (k >> 1);
            if (tu < 0) tu = 0; if (tv < 0) tv = 0; if (tu >= tw) tu = tw - 1; if (tv >= th) tv = th - 1;
            float wgt = ((k & 1) ? fu : 1 - fu) * ((k >> 1) ? fv : 1 - fv), p[4];
            unpack(tex[tv * tw + tu], p);
            for (int j = 0; j < 4; j++) acc[j] += p[j] * wgt;
        }
        blend(dst + y * 512 + x, acc[0] * m[0] / 255, acc[1] * m[1] / 255, acc[2] * m[2] / 255, acc[3] * m[3] / 255 / 255);
    }
}

void gfx_rect(float x, float y, float w, float h, uint32_t col) {
    float p[4]; unpack(col, p);
    for (int j = (int)y; j < (int)(y + h); j++) for (int i = (int)x; i < (int)(x + w); i++)
        if ((unsigned)i < 480 && (unsigned)j < 272) blend(dst + j * 512 + i, p[0], p[1], p[2], p[3] / 255);
}

void gfx_end(void) {}
#endif

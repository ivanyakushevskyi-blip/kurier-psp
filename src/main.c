// КУР'ЄР: ХМЕЛЬНИЦЬКИЙ - PSP homebrew prototype
// Real Khmelnytskyi centre (OpenStreetMap) rendered to pixel art by tools/*.py
// Map lives in MAP.BIN next to EBOOT: zlib-compressed 64x64 chunks (pixels + collision bits),
// streamed from the memory stick into an LRU cache with prefetch around the camera.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <zlib.h>

#ifndef HOST
#include <pspkernel.h>
#include <pspdisplay.h>
#include <pspctrl.h>
#include <pspge.h>
PSP_MODULE_INFO("KurierKhm", 0, 1, 0);
PSP_MAIN_THREAD_ATTR(PSP_THREAD_ATTR_USER | PSP_THREAD_ATTR_VFPU);
PSP_HEAP_SIZE_KB(-1024);
#else
#define PSP_CTRL_SELECT   0x000001
#define PSP_CTRL_START    0x000008
#define PSP_CTRL_UP       0x000010
#define PSP_CTRL_RIGHT    0x000020
#define PSP_CTRL_DOWN     0x000040
#define PSP_CTRL_LEFT     0x000080
#define PSP_CTRL_LTRIGGER 0x000100
#define PSP_CTRL_RTRIGGER 0x000200
#define PSP_CTRL_TRIANGLE 0x001000
#define PSP_CTRL_CIRCLE   0x002000
#define PSP_CTRL_CROSS    0x004000
#define PSP_CTRL_SQUARE   0x008000
#endif

#define SW 480
#define SH 272
#define STRIDE 512
#define BUFSZ (STRIDE * SH * 4)
#define PI_F 3.14159265f
#define VMAX 1.31f                   /* макс. швидкість, пікс/кадр (≈141 км/год) */

typedef uint32_t u32;
typedef uint8_t u8;

#include "assets.h"
#include "pal_ids.h"

static u8 fb[SW * SH];
static int tick;

/* ================= map chunk cache (MAP.BIN) ================= */
#define NCHUNK (NCX * NCY)
#define CSB (CS * CS / 8)
#define NCACHE 200
static u8 cache[NCACHE][CS * CS + CSB];
static int cache_key[NCACHE];
static unsigned cache_used[NCACHE];
static int16_t slot_of[NCHUNK];
static unsigned ctick;
static uint32_t moff[NCHUNK + 1];
static FILE *mapf;
static char map_path[512] = "MAP.BIN";
static int map_err;                       /* 1 = немає файлу, 2 = не той файл */
static u8 zbuf[8192];
static int loads_frame;
static const u8 empty_chunk[CS * CS + CSB];

static int map_open(void) {
    mapf = fopen(map_path, "rb");
    if (!mapf) { map_err = 1; return 0; }
    char mg[4]; uint32_t h[3];
    if (fread(mg, 1, 4, mapf) != 4 || memcmp(mg, "KHM1", 4) || fread(h, 4, 3, mapf) != 3 || h[0] != NCX || h[1] != NCY || h[2] != CS ||
        fread(moff, 4, NCHUNK + 1, mapf) != NCHUNK + 1 || moff[NCHUNK] != MAPBIN_SIZE) { map_err = 2; fclose(mapf); mapf = NULL; return 0; }
    return 1;
}
static void cache_init(void) {
    for (int i = 0; i < NCHUNK; i++) slot_of[i] = -1;
    for (int i = 0; i < NCACHE; i++) { cache_key[i] = -1; cache_used[i] = 0; }
}
static int chunk_load(int k) {
    int best = 0;
    for (int i = 0; i < NCACHE; i++) {
        if (cache_key[i] < 0) { best = i; break; }
        if (cache_used[i] < cache_used[best]) best = i;
    }
    if (cache_key[best] >= 0) slot_of[cache_key[best]] = -1;
    uint32_t n = moff[k + 1] - moff[k];
    uLongf len = CS * CS + CSB;
    if (!mapf || n > sizeof zbuf || fseek(mapf, (long)moff[k], SEEK_SET) || fread(zbuf, 1, n, mapf) != n ||
        uncompress(cache[best], &len, zbuf, n) != Z_OK)
        memset(cache[best], 0, sizeof cache[best]);
    cache_key[best] = k; slot_of[k] = (int16_t)best; cache_used[best] = ++ctick;
    loads_frame++;
    return best;
}
static const u8 *chunk(int cx, int cy) {
    if (cx < 0 || cy < 0 || cx >= NCX || cy >= NCY) return empty_chunk;
    int k = cy * NCX + cx;
    int s = slot_of[k];
    if (s < 0) s = chunk_load(k);
    cache_used[s] = ++ctick;
    return cache[s];
}
/* підвантаження наперед: кільце чанків навколо екрана, не більше кількох за кадр */
static void prefetch(int ox, int oy, float dx, float dy) {
    int cx0 = ox / CS - 2, cx1 = (ox + SW) / CS + 2, cy0 = oy / CS - 2, cy1 = (oy + SH) / CS + 2;
    if (dx > 0.3f) cx1++; if (dx < -0.3f) cx0--; if (dy > 0.3f) cy1++; if (dy < -0.3f) cy0--;
    for (int cy = cy0; cy <= cy1 && loads_frame < 4; cy++) for (int cx = cx0; cx <= cx1 && loads_frame < 4; cx++) {
        if (cx < 0 || cy < 0 || cx >= NCX || cy >= NCY) continue;
        int k = cy * NCX + cx;
        if (slot_of[k] < 0) chunk_load(k);
    }
}
static u8 map_px(int x, int y) {
    if (x < 0 || y < 0 || x >= MAP_W || y >= MAP_H) return 0;
    return chunk(x / CS, y / CS)[(y % CS) * CS + (x % CS)];
}
static void draw_map(int ox, int oy) {
    int cy0 = oy / CS, cy1 = (oy + SH - 1) / CS, cx0 = ox / CS, cx1 = (ox + SW - 1) / CS;
    for (int cy = cy0; cy <= cy1; cy++) for (int cx = cx0; cx <= cx1; cx++) {
        if (cx < 0 || cy < 0 || cx >= NCX || cy >= NCY) continue;
        const u8 *c = chunk(cx, cy);
        int x0 = cx * CS - ox, y0 = cy * CS - oy;
        int sx0 = x0 < 0 ? 0 : x0, sx1 = x0 + CS > SW ? SW : x0 + CS;
        int sy0 = y0 < 0 ? 0 : y0, sy1 = y0 + CS > SH ? SH : y0 + CS;
        for (int sy = sy0; sy < sy1; sy++) memcpy(fb + sy * SW + sx0, c + (sy - y0) * CS + (sx0 - x0), (size_t)(sx1 - sx0));
    }
}

/* ================= collision + minimap ================= */
static u8 bigb[BIG_W * BIG_H];
static int is_solid(int x, int y) {
    if (x < 6 || y < 6 || x >= MAP_W - 6 || y >= MAP_H - 6) return 1;
    const u8 *c = chunk(x / CS, y / CS) + CS * CS;
    int lx = x % CS, ly = y % CS;
    return (c[ly * (CS / 8) + (lx >> 3)] >> (lx & 7)) & 1;
}

/* ================= drawing ================= */
static void hline(int x0, int x1, int y, u8 c) {
    if (y < 0 || y >= SH) return;
    if (x0 > x1) { int t = x0; x0 = x1; x1 = t; }
    if (x0 < 0) x0 = 0; if (x1 >= SW) x1 = SW - 1;
    if (x0 > x1) return;
    memset(fb + y * SW + x0, c, (size_t)(x1 - x0 + 1));
}
static void rect(int x, int y, int w, int h, u8 c) { for (int j = 0; j < h; j++) hline(x, x + w - 1, y + j, c); }
static void pset(int x, int y, u8 c) { if ((unsigned)x < SW && (unsigned)y < SH) fb[y * SW + x] = c; }
static void darken_rect(int x, int y, int w, int h, int times) {
    for (int j = y; j < y + h; j++) { if ((unsigned)j >= SH) continue;
        for (int i = x; i < x + w; i++) { if ((unsigned)i >= SW) continue;
            u8 *p = fb + j * SW + i; for (int t = 0; t < times; t++) *p = DARK[*p]; } }
}
static void shadow_ell(float cx, float cy, float rx, float ry) {
    for (int y = (int)(cy - ry); y <= (int)(cy + ry); y++) {
        if ((unsigned)y >= SH) continue;
        float dy = (y + 0.5f - cy) / ry; if (dy * dy >= 1) continue;
        float w = rx * sqrtf(1 - dy * dy);
        for (int x = (int)(cx - w); x <= (int)(cx + w); x++) if ((unsigned)x < SW) fb[y * SW + x] = DARK[fb[y * SW + x]];
    }
}
static void disc(float cx, float cy, float r, u8 c) {
    for (int y = (int)floorf(cy - r); y <= (int)ceilf(cy + r); y++) {
        float dy = (y + 0.5f - cy) / r; if (dy * dy >= 1) continue;
        float w = r * sqrtf(1 - dy * dy);
        hline((int)floorf(cx - w + 0.5f), (int)floorf(cx + w - 0.5f), y, c);
    }
}
static void ring(float cx, float cy, float r, u8 c, int th) {
    for (int i = 0; i < 64; i++) {
        float a = i * 2 * PI_F / 64;
        for (int t = 0; t < th; t++) pset((int)(cx + cosf(a) * (r - t)), (int)(cy + sinf(a) * (r - t)), c);
    }
}
static void tri(float x0, float y0, float x1, float y1, float x2, float y2, u8 c) {
    float ymin = fminf(y0, fminf(y1, y2)), ymax = fmaxf(y0, fmaxf(y1, y2));
    float P[6] = { x0, y0, x1, y1, x2, y2 };
    for (int y = (int)floorf(ymin); y <= (int)ceilf(ymax); y++) {
        float yc = y + 0.5f, xs[3]; int k = 0;
        for (int i = 0; i < 3; i++) {
            int j = (i + 1) % 3; float ya = P[2 * i + 1], yb = P[2 * j + 1];
            if ((ya <= yc && yb > yc) || (yb <= yc && ya > yc)) xs[k++] = P[2 * i] + (yc - ya) / (yb - ya) * (P[2 * j] - P[2 * i]);
        }
        if (k == 2) hline((int)ceilf(fminf(xs[0], xs[1]) - 0.5f), (int)floorf(fmaxf(xs[0], xs[1]) - 0.5f), y, c);
    }
}
static void blit(const u8 *src, int w, int h, int x, int y) {
    for (int j = 0; j < h; j++) {
        int yy = y + j; if ((unsigned)yy >= SH) continue;
        const u8 *r = src + j * w; u8 *d = fb + yy * SW;
        for (int i = 0; i < w; i++) { int xx = x + i; u8 v = r[i]; if (v && (unsigned)xx < SW) d[xx] = v; }
    }
}

/* ================= text ================= */
static int utf8next(const char **s) {
    const unsigned char *p = (const unsigned char *)*s; int c = *p;
    if (c < 0x80) { *s += 1; return c; }
    if ((c & 0xE0) == 0xC0) { int r = ((c & 0x1F) << 6) | (p[1] & 0x3F); *s += 2; return r; }
    if ((c & 0xF0) == 0xE0) { int r = ((c & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F); *s += 3; return r; }
    *s += 1; return '?';
}
static int ncp(const char *s) { int n = 0; while (*s) { utf8next(&s); n++; } return n; }
static const Glyph *glyph(int sc, int cp) {
    const FontTab *F = &FONTS[sc];
    int lo = 0, hi = F->n - 1;
    while (lo <= hi) { int m = (lo + hi) / 2; if (F->g[m].cp == cp) return &F->g[m]; if (F->g[m].cp < cp) lo = m + 1; else hi = m - 1; }
    return cp == '?' ? NULL : glyph(sc, '?');
}
static int fsc(int sc) { return sc < 1 ? 1 : (sc > 5 ? 5 : sc); }
static int tw8(const char *s, int sc) {
    sc = fsc(sc); int w = 0;
    while (*s) { const Glyph *g = glyph(sc, utf8next(&s)); if (g) w += g->adv; }
    return w;
}
/* пропорційний шрифт Rubik; y — верх великих літер (як було з 8x8) */
static void text8(int x, int y, int sc, u8 col, const char *s) {
    sc = fsc(sc); int h = FONTS[sc].h; y -= FONTS[sc].asc;
    while (*s) {
        const Glyph *g = glyph(sc, utf8next(&s));
        if (!g) continue;
        int st = (g->w + 7) >> 3; const u8 *b = FONT_DATA + g->off;
        for (int r = 0; r < h; r++) {
            int yy = y + r; if ((unsigned)yy >= SH) { b += st; continue; }
            u8 *d = fb + yy * SW;
            for (int c = 0; c < g->w; c++) if (b[c >> 3] & (128 >> (c & 7))) { int xx = x + g->xoff + c; if ((unsigned)xx < SW) d[xx] = col; }
            b += st;
        }
        x += g->adv;
    }
}
static void otext8(int x, int y, int sc, u8 col, u8 oc, const char *s) {
    int o = sc > 2 ? 2 : 1;
    for (int dy = -o; dy <= o; dy += o) for (int dx = -o; dx <= o; dx += o) if (dx || dy) text8(x + dx, y + dy, sc, oc, s);
    text8(x, y, sc, col, s);
}
static void otext8_c(int cx, int y, int sc, u8 col, u8 oc, const char *s) { otext8(cx - tw8(s, sc) / 2, y, sc, col, oc, s); }

/* ================= game state ================= */
enum { S_TITLE, S_PLAY, S_PAUSE, S_BIGMAP };
enum { O_NONE, O_PICK, O_LOAD, O_DROP };
static int state = S_TITLE;
static float px_, py_, ang, vel;           /* vel — швидкість уздовж кузова (пікс/кадр, зі знаком) */
static float vx_, vy_;                       /* вектор швидкості у світі (інерція, занос) */
static float health = 100;                   /* стан авто, % */
static float fuel = 100;                     /* пальне, % */
static int fuel_t, fuel_warned;
static int lane_assist;
static int menu_i;                      /* помічник утримання смуги (налаштування) */
static int braking, drifting;
enum { EV_NONE, EV_WRECK, EV_DROWN, EV_FUEL };
static int evac, evac_t, repair_t;
static int shake;                            /* сила струсу камери */
static float camx, camy;
static int freecam;                       /* L: огляд міста без машини */
static float fcx, fcy;
static int ostate = O_NONE, owait = 60, otimer, olimit, shop_i = -1, addr_i = -1, pay, load_t;
static int money, deliveries, late;
static char toast[48]; static int toast_t; static u8 toast_c;
static int street_i = -1, street_t;
static char save_path[512] = "save.dat";
/* пам'ятки: відкриті (біт = порядок у data/landmarks.json), анімація таблички */
static u8 lm_found[NLM > 0 ? NLM : 1], lm_lab[NLM > 0 ? NLM : 1];
static int lm_count;
#define LM_REWARD 50

static int bump;
static void puff(float x, float y, float dx, float dy, int life, u8 kind);
static void say(const char *t, u8 c, int life) { strncpy(toast, t, 47); toast[47] = 0; toast_t = life; toast_c = c; }
static float dist(float ax, float ay, float bx, float by) { float dx = ax - bx, dy = ay - by; return sqrtf(dx * dx + dy * dy); }

static void load_save(void) {
    FILE *f = fopen(save_path, "rb");
    if (f) {
        int a = 0, b = 0; char bits[130] = "";
        if (fscanf(f, "%d %d", &a, &b) == 2) { money = a; deliveries = b; }
        if (fscanf(f, "%129s", bits) == 1)
            for (int i = 0; i < NLM; i++) { int k = LMS[i].bit; if (k < (int)strlen(bits) && bits[k] == '1') { lm_found[i] = 1; lm_count++; } }
        int la = 0; if (fscanf(f, "%d", &la) == 1) lane_assist = la != 0;
        float fu, he; if (fscanf(f, "%f %f", &fu, &he) == 2) { fuel = fu; health = he > 5 ? he : 100; }
        fclose(f);
    }
}
static void write_save(void) {
    FILE *f = fopen(save_path, "wb");
    if (f) {
        char bits[130]; int n = 0;
        for (int i = 0; i < NLM; i++) if (LMS[i].bit + 1 > n) n = LMS[i].bit + 1;
        if (n > 128) n = 128;
        memset(bits, '0', (size_t)n); bits[n] = 0;
        for (int i = 0; i < NLM; i++) if (lm_found[i] && LMS[i].bit < n) bits[LMS[i].bit] = '1';
        fprintf(f, "%d %d %s %d %.1f %.1f\n", money, deliveries, n ? bits : "-", lane_assist, fuel, health); fclose(f);
    }
}

/* ---- замовлення: на вибір 3 пропозиції з різними умовами */
enum { T_NORMAL, T_URGENT, T_FAR, T_FRAGILE, T_DANGER, T_HEAVY, T_FOOD, NTYPE };
static const char *TNAME[NTYPE] = { "ЗВИЧАЙНЕ", "ТЕРМІНОВЕ", "ДАЛЕКЕ", "КРИХКЕ", "НЕБЕЗПЕЧНЕ", "ВАЖКЕ", "ГАРЯЧА ЇЖА" };
static const float TPAY[NTYPE] = { 1.0f, 1.6f, 1.3f, 1.5f, 2.2f, 1.4f, 1.3f };
static const float TTIME[NTYPE] = { 1.0f, 0.62f, 1.0f, 1.15f, 1.2f, 1.3f, 0.8f };
typedef struct { int type, shop, addr, pay, limit, ttl; } Offer;
static Offer offers[3]; static int n_offers, offer_sel, offer_t, otype;
static int note_open, note_sel;          /* нотатник висунутий; вибраний рядок/кнопка */
static float cargo = 100;                    /* стан крихкого вантажу, % */
static u8 tcol(int t) { return t == T_DANGER ? C_red_3 : (t == T_URGENT || t == T_FOOD ? C_yellow_3 : (t == T_FRAGILE ? C_white_2 : C_green_3)); }
static int is_food(int i) { const char *k = SHOPS[i].kind; return !strcmp(k, "КАВ'ЯРНЯ") || !strcmp(k, "РЕСТОРАН") || !strcmp(k, "ФАСТФУД") || !strcmp(k, "ПІЦА") || !strcmp(k, "ПЕКАРНЯ"); }
static int pick_shop(float maxd, int food) {
    int best = -1;
    for (int t = 0; t < 300; t++) {
        int i = rand() % NSHOPS;
        if (food && !is_food(i)) continue;
        float d = dist(px_, py_, SHOPS[i].x, SHOPS[i].y);
        if (d > 120 && d < maxd) return i;
        if (best < 0 || d < dist(px_, py_, SHOPS[best].x, SHOPS[best].y)) best = i;
    }
    return best < 0 ? rand() % NSHOPS : best;
}
static int pick_addr(int shop, float dmin, float dmax) {
    int ab = -1;
    float bd = 0;
    for (int t = 0; t < 600; t++) {
        int i = rand() % NADDRS;
        float d = dist(SHOPS[shop].x, SHOPS[shop].y, ADDRS[i].x, ADDRS[i].y);
        if (d > dmin && d < dmax) return i;
        if (d < dmax && d > bd) { bd = d; ab = i; }              /* інакше — найдальша з допустимих */
    }
    return ab < 0 ? rand() % NADDRS : ab;
}
/* відстані в пікселях (0,5 м): звичайне 1,5–3,5 км, термінове/їжа 1,2–2,5 км, далеке 4–6 км — з району в район */
static void make_offer(int k) {
    int used = 0;
    for (int j = 0; j < 3; j++) if (j != k && offers[j].ttl > 0) used |= 1 << offers[j].type;
    int t; do { t = rand() % NTYPE; } while (used & (1 << t));
    Offer *o = &offers[k]; o->type = t;
    o->shop = pick_shop(1600, t == T_FOOD);
    float dmin = 3000, dmax = 7000;                          /* 1,5–3,5 км */
    if (t == T_FAR) { dmin = 8000; dmax = 12500; }           /* 4–6 км: через усе місто */
    else if (t == T_URGENT || t == T_FOOD) { dmin = 2400; dmax = 5000; }
    o->addr = pick_addr(o->shop, dmin, dmax);
    float dm = dist(SHOPS[o->shop].x, SHOPS[o->shop].y, ADDRS[o->addr].x, ADDRS[o->addr].y) * MPP;
    o->pay = (int)((35 + dm * 0.09f) * TPAY[t]);
    o->limit = (int)((dm * 1.4f / 12.0f + 30.0f) * 60.0f * TTIME[t]);
    o->ttl = 60 * (50 + rand() % 50);                        /* пропозиція висить 50–100 с, потім її забирає інший кур'єр */
}
static void make_offers(void) {
    for (int k = 0; k < 3; k++) offers[k].ttl = 0;
    for (int k = 0; k < 3; k++) make_offer(k);
    n_offers = 3; offer_sel = 0;
    say("НОВІ ЗАМОВЛЕННЯ!", C_green_3, 100);
}
static void accept_offer(int k) {
    Offer *o = &offers[k];
    shop_i = o->shop; addr_i = o->addr; otype = o->type; pay = o->pay; olimit = o->limit; cargo = 100;
    n_offers = 0; ostate = O_PICK;
    char b[48]; snprintf(b, sizeof b, "ЗАМОВЛЕННЯ: %s", TNAME[otype]); say(b, tcol(otype), 110);
}
static void notebook_input(unsigned pressed) {
    if (!note_open) {
        if ((pressed & PSP_CTRL_TRIANGLE) && !(fuel <= 0 && fabsf(vel) < 0.05f)) { note_open = 1; note_sel = 0; }
        return;
    }
    if (pressed & PSP_CTRL_CIRCLE) { note_open = 0; return; }
    if (ostate != O_NONE) {                                  /* під час замовлення: СХОВАТИ / ВІДМОВИТИСЯ */
        if (pressed & (PSP_CTRL_UP | PSP_CTRL_DOWN)) note_sel ^= 1;
        if (pressed & PSP_CTRL_TRIANGLE) {
            if (note_sel) { ostate = O_NONE; n_offers = 0; owait = 300; say("ВІДМОВИВСЯ ВІД ЗАМОВЛЕННЯ", C_red_3, 120); }
            note_open = 0;
        }
    }
}
static void fail_order(const char *why) { ostate = O_NONE; owait = 200; late++; say(why, C_red_3, 160); }

static void update_orders(unsigned pressed) {
    if (ostate == O_NONE) {
        if (!n_offers) { if (--owait <= 0) make_offers(); return; }
        if (note_open) {
            if (pressed & PSP_CTRL_UP) offer_sel = (offer_sel + 2) % 3;
            if (pressed & PSP_CTRL_DOWN) offer_sel = (offer_sel + 1) % 3;
            if (pressed & PSP_CTRL_TRIANGLE) { accept_offer(offer_sel); note_open = 0; return; }
        }
        for (int k = 0; k < 3; k++) if (--offers[k].ttl <= 0) make_offer(k);     /* без «перемішування»: лише чекати */
        return;
    }
    if (ostate == O_PICK) {
        if (dist(px_, py_, SHOPS[shop_i].x, SHOPS[shop_i].y) < 24 && fabsf(vel) < 0.35f) { ostate = O_LOAD; load_t = 0; }
        return;
    }
    if (ostate == O_LOAD) {
        if (dist(px_, py_, SHOPS[shop_i].x, SHOPS[shop_i].y) > 34) { ostate = O_PICK; return; }
        if (++load_t >= 60) {
            otimer = olimit;
            ostate = O_DROP;
            say(otype == T_DANGER ? "ОБЕРЕЖНО! НЕ БИЙ МАШИНУ" : (otype == T_FRAGILE ? "КРИХКЕ! ЇДЬ АКУРАТНО" : "ЗАБРАВ! ВЕЗИ ДО КЛІЄНТА"), C_yellow_3, 110);
        }
        return;
    }
    if (ostate == O_DROP) {
        if (--otimer <= 0) { fail_order("ЗАПІЗНИВСЯ. ЗАМОВЛЕННЯ СКАСОВАНО"); return; }
        if (dist(px_, py_, ADDRS[addr_i].x, ADDRS[addr_i].y) < 24 && fabsf(vel) < 0.45f) {
            int bonus = otimer / 60 / 3;
            int got = pay + bonus;
            if (otype == T_FRAGILE) got = (int)(got * cargo / 100.0f);
            money += got; deliveries++;
            char b[48]; snprintf(b, sizeof b, "+%d ГРН%s", got, bonus > 5 ? "  ШВИДКО!" : "");
            say(b, C_green_3, 150);
            ostate = O_NONE; owait = 140;
            write_save();
        }
    }
}
/* удар під час доставки: крихке псується, небезпечне вибухає */
static void cargo_hit(float spd) {
    if (ostate != O_DROP) return;
    if (otype == T_FRAGILE) {
        cargo -= (spd - 0.2f) * 140.0f;
        if (cargo <= 0) { cargo = 0; fail_order("ВАНТАЖ РОЗБИТО!"); }
        else { char b[40]; snprintf(b, sizeof b, "ВАНТАЖ: %d%%", (int)cargo); say(b, C_red_3, 70); }
    } else if (otype == T_DANGER && spd > 0.34f) {
        health = 0; fail_order("ВИБУХ! ВАНТАЖ ЗНИЩЕНО");
        for (int k = 0; k < 24; k++) puff(px_ + rand() % 15 - 7, py_ + rand() % 15 - 7, (rand() % 11 - 5) * 0.12f, (rand() % 11 - 5) * 0.12f, 30 + rand() % 20, 1);
    }
}

static int hit(float x, float y, float a) {
    float c = cosf(a), s = sinf(a);
    static const float P[][2] = { { 0, 0 }, { 9, 0 }, { -9, 0 }, { 8, 3 }, { 8, -3 }, { -8, 3 }, { -8, -3 }, { 0, 3 }, { 0, -3 } };   /* Golf ~20x9 пікс, трохи менше за спрайт */
    for (int i = 0; i < 9; i++) {
        float lx = P[i][0], ly = P[i][1];
        if (is_solid((int)(x + lx * c - ly * s), (int)(y + lx * s + ly * c))) return 1;
    }
    return 0;
}

/* Аварійно: якщо кузов опинився в стіні — виштовхнути на найближче вільне місце (спіраль до ~25 м). */
static void unstick(void) {
    if (!hit(px_, py_, ang)) return;
    for (int r = 2; r <= 50; r += 2)
        for (int k = 0; k < 16; k++) {
            float a = k * (2 * PI_F / 16);
            float x = px_ + cosf(a) * r, y = py_ + sinf(a) * r;
            if (!hit(x, y, ang)) { px_ = x; py_ = y; return; }
            for (int t = 1; t < 8; t++) { float aa = ang + t * (PI_F / 8); if (!hit(x, y, aa)) { px_ = x; py_ = y; ang = aa; return; } }
        }
}

/* Якщо машина затиснута (газ/задня не рухають її ~1,5 с) — переставити на найближче відкрите місце. */
static int blocked_t;
static int open_spot(float x, float y) {
    for (int k = 0; k < 8; k++) { float a = k * (PI_F / 4); if (hit(x + cosf(a) * 8, y + sinf(a) * 8, a)) return 0; }
    return !hit(x, y, ang);
}
static void stop_car(void);
static void unwedge(void) {
    for (int r = 6; r <= 120; r += 3)
        for (int k = 0; k < 24; k++) {
            float a = k * (2 * PI_F / 24), x = px_ + cosf(a) * r, y = py_ + sinf(a) * r;
            if (open_spot(x, y)) { px_ = x; py_ = y; stop_car(); blocked_t = 0; return; }
        }
}

static void stop_car(void) { vx_ = vy_ = 0; vel = 0; }

/* ---- ефекти: дим з-під коліс, сліди шин, пара з-під капота, бульбашки */
#define NPART 64
static struct { float x, y, dx, dy; int life, max; u8 kind; } part[NPART];
static int part_i;
static void puff(float x, float y, float dx, float dy, int life, u8 kind) {
    part[part_i].x = x; part[part_i].y = y; part[part_i].dx = dx; part[part_i].dy = dy;
    part[part_i].life = part[part_i].max = life; part[part_i].kind = kind; part_i = (part_i + 1) % NPART;
}
#define NSKID 1200
static int16_t skid[NSKID][2]; static int skid_i, skid_n;
static void add_skid(float x, float y) {
    int16_t ix = (int16_t)x, iy = (int16_t)y; int pi = (skid_i + NSKID - 1) % NSKID;
    if (skid_n && skid[pi][0] == ix && skid[pi][1] == iy) return;
    skid[skid_i][0] = ix; skid[skid_i][1] = iy; skid_i = (skid_i + 1) % NSKID; if (skid_n < NSKID) skid_n++;
}

/* ---- помічник смуги: найближча вісь дороги (сітка LANE_G) -> кермо до правої смуги */
static float lane_steer(void) {
    if (NLANE == 0) return 0;
    int gx = (int)px_ / LANE_G, gy = (int)py_ / LANE_G;
    if (gx < 0 || gy < 0 || gx >= LANE_GW || gy >= LANE_GH) return 0;
    int c = gy * LANE_GW + gx;
    float fx = cosf(ang), fy = sinf(ang), best = 24.0f * 24.0f, out = 0; int found = 0;
    for (uint32_t k = LANE_CELL[c]; k < LANE_CELL[c + 1]; k++) {
        const int16_t *L = LANES[LANE_IDX[k]];
        float x0 = L[0], y0 = L[1], dx = L[2] - x0, dy = L[3] - y0, l2 = dx * dx + dy * dy;
        if (l2 < 1) continue;
        float t = ((px_ - x0) * dx + (py_ - y0) * dy) / l2; if (t < -0.05f || t > 1.05f) continue;
        float ln = sqrtf(l2), ux = dx / ln, uy = dy / ln;
        float dot = ux * fx + uy * fy;
        if (L[5] && dot < 0.2f) continue;                    /* одностороння: лише за напрямком */
        if (dot < 0) { ux = -ux; uy = -uy; dot = -dot; }     /* двостороння: у бік руху машини */
        if (dot < 0.64f) continue;                           /* перехрестя / поворот — не заважаємо */
        float nx = -uy, ny = ux;                             /* праворуч від напрямку (y донизу) */
        float cx = x0 + dx * t, cy = y0 + dy * t;
        float off = L[5] ? 0 : L[4];
        float lat = (px_ - cx) * nx + (py_ - cy) * ny - off;  /* >0 — правіше центру смуги */
        float d2 = (px_ - cx) * (px_ - cx) + (py_ - cy) * (py_ - cy);
        if (d2 < best) {
            best = d2; found = 1;
            float err = atan2f(uy, ux) - ang;
            while (err > PI_F) err -= 2 * PI_F; while (err < -PI_F) err += 2 * PI_F;
            out = err * 2.2f - lat * 0.035f;
        }
    }
    if (!found) return 0;
    if (out > 0.7f) out = 0.7f; if (out < -0.7f) out = -0.7f;
    return out;
}

static int on_water(float x, float y) { return WATERIDX[map_px((int)x, (int)y)]; }

static void update_player(unsigned btn, int lx) {
    float steer = 0;
    if (btn & PSP_CTRL_LEFT) steer = -1;
    if (btn & PSP_CTRL_RIGHT) steer = 1;
    if (abs(lx - 128) > 32) steer = (lx - 128) / 96.0f;
    if (steer > 1) steer = 1; if (steer < -1) steer = -1;
    int gas = (btn & PSP_CTRL_CROSS) != 0;
    int brake = (btn & PSP_CTRL_SQUARE) != 0;
    int hand = (btn & PSP_CTRL_RTRIGGER) != 0;
    static int free_t;
    if (fabsf(steer) > 0.08f) free_t = 0; else free_t++;
    if (lane_assist && free_t > 8 && !hand && vel > 0.12f) steer = lane_steer();

    float fx = cosf(ang), fy = sinf(ang), rx = -fy, ry = fx;
    float vf = vx_ * fx + vy_ * fy, vr = vx_ * rx + vy_ * ry;
    int off = OFFROAD[map_px((int)px_, (int)py_)];
    float vmax = off ? 0.50f : VMAX;                         /* 1,31 пікс/кадр ≈ 141 км/год; на спідометрі — 140 */
    int heavy = ostate == O_DROP && otype == T_HEAVY;
    if (heavy) vmax *= 0.7f;
    if (fuel <= 0) gas = 0;                                   /* пальне скінчилося */
    /* двигун / гальмо / задній хід */
    /* розгін: 0-100 ≈ 3 с, далі все важче, 140 ≈ 10 с — межа можливостей авто */
    if (gas && vf >= -0.02f) { float k = vf / vmax; if (k > 1) k = 1; vf += ((off ? 0.008f : 0.011f) * powf(1.0f - k, 1.6f) + 0.0004f) * (heavy ? 0.6f : 1.0f); }
    else if (gas) vf += 0.045f;
    braking = 0;
    /* гальма: ~1 g, 140→0 приблизно за 2,5 с; накатом машина котиться довго */
    if (brake) { if (vf > 0.03f) { vf -= 0.0085f + 0.004f * (vf / VMAX); braking = 1; } else if (vf > -0.34f) vf -= 0.011f; }
    if (!gas && !brake) { vf -= vf * 0.0025f + (vf > 0 ? 0.0009f : -0.0009f); if (fabsf(vf) < 0.004f) vf = 0; }
    if (vf > vmax) vf -= (vf - vmax) * 0.12f;
    if (vf < -0.34f) vf = -0.34f;
    /* зчеплення: гасить боковий рух; ручник — занос */
    float grip = off ? 0.10f : 0.24f;
    if (hand) { grip = 0.035f; vf -= vf * 0.006f + (vf > 0 ? 0.004f : -0.004f); braking = 1; }
    vr *= 1.0f - grip;
    /* поворот: залежить від швидкості; на великій — плавніше */
    float sp = fabsf(vf);
    float turn = sp / 0.22f; if (turn > 1) turn = 1;
    if ((gas || brake) && turn < 0.35f) turn = 0.35f;           /* аркадно: вивернути від стіни майже з місця */
    turn *= 1.0f - 0.38f * (sp / VMAX);
    if (hand) turn *= 1.45f;
    float na = ang + steer * 0.058f * turn * (vf < 0 ? -1 : 1);
    if (na > PI_F) na -= 2 * PI_F; if (na < -PI_F) na += 2 * PI_F;
    if (!hit(px_, py_, na)) ang = na;
    /* вектор швидкості у світі: інерція — повертається за кузовом через зчеплення */
    vx_ = fx * vf + rx * vr; vy_ = fy * vf + ry * vr;
    drifting = fabsf(vr) > 0.16f && sp > 0.3f;
    /* рух дрібними кроками з ковзанням уздовж стін */
    float ox_ = px_, oy_ = py_;
    float spd = sqrtf(vx_ * vx_ + vy_ * vy_);
    int steps = 1 + (int)(spd / 0.4f);
    float dx = vx_ / steps, dy = vy_ / steps;
    for (int i = 0; i < steps; i++) {
        if (!hit(px_ + dx, py_ + dy, ang)) { px_ += dx; py_ += dy; continue; }
        if (fabsf(dx) > 0.01f && !hit(px_ + dx, py_, ang)) { px_ += dx; vx_ *= 0.97f; vy_ *= 0.6f; continue; }
        if (fabsf(dy) > 0.01f && !hit(px_, py_ + dy, ang)) { py_ += dy; vy_ *= 0.97f; vx_ *= 0.6f; continue; }
        /* удар */
        if (spd > 0.28f) {
            float dmg = (spd - 0.28f) * 42.0f;
            health -= dmg; bump = 10; shake = (int)(2 + spd * 6);
            cargo_hit(spd);
            for (int k = 0; k < 4; k++) puff(px_ + fx * 9, py_ + fy * 9, (rand() % 7 - 3) * 0.08f, (rand() % 7 - 3) * 0.08f, 22, 1);
        }
        vx_ *= -0.22f; vy_ *= -0.22f;
        break;
    }
    vel = vx_ * fx + vy_ * fy;
    /* дим і сліди при заносі / пробуксовці */
    int burn = gas && sp < 0.35f && vf > 0.02f && !off && (tick & 1);
    if ((drifting || (hand && sp > 0.35f) || burn) && !off) {
        for (int sgn = -1; sgn <= 1; sgn += 2) {
            float wx = px_ - fx * 7 + rx * 4 * sgn, wy = py_ - fy * 7 + ry * 4 * sgn;
            add_skid(wx, wy);
            if ((tick + sgn) % 3 == 0) puff(wx, wy, -vx_ * 0.2f + (rand() % 5 - 2) * 0.05f, -vy_ * 0.2f - 0.05f, 34, 0);
        }
    }
    if (health < 35 && tick % 9 == 0) puff(px_ + fx * 8, py_ + fy * 8, 0.02f, -0.18f, 40, 2);   /* пара з-під капота */
    float moved = fabsf(px_ - ox_) + fabsf(py_ - oy_);
    fuel -= moved * 0.00156f * (heavy ? 1.4f : 1.0f) + (gas ? 0.0002f : 0);     /* повний бак ≈ 32 км */
    if (fuel < 0) fuel = 0;
    if (fuel < 20 && !fuel_warned) { fuel_warned = 1; say("МАЛО ПАЛЬНОГО! ЗАЇДЬ НА АЗС", C_yellow_3, 160); }
    if ((gas || brake) && moved < 0.02f) { if (++blocked_t > 90) unwedge(); } else blocked_t = 0;
    static int win_t; static float wx, wy;
    if (gas || brake) {
        if (++win_t >= 240) { if (fabsf(px_ - wx) + fabsf(py_ - wy) < 6) unwedge(); win_t = 0; wx = px_; wy = py_; }
    } else { win_t = 0; wx = px_; wy = py_; }
    unstick();
    /* вода: машина тоне; розбита: евакуація */
    if (on_water(px_, py_) && on_water(px_ + fx * 6, py_ + fy * 6)) { evac = EV_DROWN; evac_t = 0; say("МАШИНА ТОНЕ!", C_red_3, 120); }
    else if (health <= 0) { health = 0; evac = EV_WRECK; evac_t = 0; say("АВТО РОЗБИТЕ!", C_red_3, 120); }
}

/* ---- евакуація на найближче СТО і ремонт */
static int nearest_sto(float x, float y) {
    int b = 0; for (int i = 1; i < NSTO; i++) if (dist(x, y, STOS[i].x, STOS[i].y) < dist(x, y, STOS[b].x, STOS[b].y)) b = i;
    return b;
}
static void update_evac(void) {
    evac_t++; vel = 0;
    if (evac == EV_FUEL) { vx_ = vy_ = 0; }
    else if (evac == EV_DROWN) { vx_ *= 0.9f; vy_ *= 0.9f; px_ += vx_; py_ += vy_; if (tick % 4 == 0) puff(px_ + (rand() % 13 - 6), py_ + (rand() % 9 - 4), 0, -0.15f, 26, 3); }
    else { vx_ *= 0.8f; vy_ *= 0.8f; if (tick % 5 == 0) puff(px_ + (rand() % 9 - 4), py_, 0.02f, -0.2f, 44, 2); }
    if (evac_t < 110) return;
    if (evac == EV_FUEL) {
        int b = 0; for (int i = 1; i < NFUEL; i++) if (dist(px_, py_, FUELS[i].x, FUELS[i].y) < dist(px_, py_, FUELS[b].x, FUELS[b].y)) b = i;
        money -= 300; px_ = FUELS[b].x; py_ = FUELS[b].y; stop_car(); fuel = 15; fuel_warned = 0; evac = EV_NONE;
        unstick(); say("ЕВАКУАТОР ДО АЗС: -300 ГРН", C_red_3, 200); write_save(); return;
    }
    int k = nearest_sto(px_, py_);
    int cost = evac == EV_DROWN ? 700 : 450;
    money -= cost;
    if (ostate != O_NONE) { ostate = O_NONE; n_offers = 0; owait = 200; }
    px_ = STOS[k].x; py_ = STOS[k].y; stop_car(); health = 100; evac = EV_NONE;
    for (int r = 0; r < 200 && hit(px_, py_, ang); r += 2) px_ = STOS[k].x + r;
    unstick();
    char b[48]; snprintf(b, sizeof b, "ЕВАКУАЦІЯ НА СТО: -%d ГРН", cost);
    say(b, C_red_3, 200);
    write_save();
}
static void update_fuel(unsigned pressed) {
    if (fuel <= 0 && fabsf(vel) < 0.05f && (pressed & PSP_CTRL_CIRCLE)) { evac = EV_FUEL; evac_t = 0; say("ВИКЛИКАЮ ЕВАКУАТОР...", C_white_2, 110); return; }
    if (fuel >= 99.5f) { fuel_t = 0; return; }
    int b = 0; for (int i = 1; i < NFUEL; i++) if (dist(px_, py_, FUELS[i].x, FUELS[i].y) < dist(px_, py_, FUELS[b].x, FUELS[b].y)) b = i;
    if (dist(px_, py_, FUELS[b].x, FUELS[b].y) < 26 && fabsf(vel) < 0.3f) {
        if (++fuel_t >= 70) {
            int cost = (int)((100 - fuel) * 2 + 0.5f);
            money -= cost; fuel = 100; fuel_t = 0; fuel_warned = 0;
            char t[48]; snprintf(t, sizeof t, "ЗАПРАВКА: -%d ГРН", cost); say(t, C_yellow_3, 150); write_save();
        }
    } else fuel_t = 0;
}
static void update_repair(void) {
    if (health >= 99.5f) { repair_t = 0; return; }
    int k = nearest_sto(px_, py_);
    if (dist(px_, py_, STOS[k].x, STOS[k].y) < 26 && fabsf(vel) < 0.3f) {
        if (++repair_t >= 70) {
            int cost = (int)((100 - health) * 3 + 0.5f);
            money -= cost; health = 100; repair_t = 0;
            char b[48]; snprintf(b, sizeof b, "РЕМОНТ: -%d ГРН", cost); say(b, C_yellow_3, 150); write_save();
        }
    } else repair_t = 0;
}

static void update_camera(int snap) {
    /* динамічна: дивиться вперед за вектором швидкості, на швидкості доганяє швидше */
    float spd = sqrtf(vx_ * vx_ + vy_ * vy_), k = spd / VMAX; if (k > 1) k = 1;
    float tx = px_ + vx_ * (40 + 45 * k) - SW / 2, ty = py_ + vy_ * (34 + 36 * k) - SH / 2 + 12;
    if (snap) { camx = tx; camy = ty; } else { float a = 0.06f + 0.05f * k; camx += (tx - camx) * a; camy += (ty - camy) * a; }
    if (camx < 0) camx = 0; if (camy < 0) camy = 0;
    if (camx > MAP_W - SW) camx = MAP_W - SW; if (camy > MAP_H - SH) camy = MAP_H - SH;
}

static void update_freecam(unsigned btn, int lx, int ly) {
    float dx = 0, dy = 0;
    if (btn & PSP_CTRL_LEFT) dx = -1; if (btn & PSP_CTRL_RIGHT) dx = 1;
    if (btn & PSP_CTRL_UP) dy = -1; if (btn & PSP_CTRL_DOWN) dy = 1;
    if (abs(lx - 128) > 30) dx = (lx - 128) / 100.0f;
    if (abs(ly - 128) > 30) dy = (ly - 128) / 100.0f;
    float sp = (btn & PSP_CTRL_CROSS) ? 9.0f : 3.0f;
    fcx += dx * sp; fcy += dy * sp;
    if (fcx < SW / 2) fcx = SW / 2; if (fcy < SH / 2) fcy = SH / 2;
    if (fcx > MAP_W - SW / 2) fcx = MAP_W - SW / 2; if (fcy > MAP_H - SH / 2) fcy = MAP_H - SH / 2;
    camx = fcx - SW / 2; camy = fcy - SH / 2;
}

static void update_street(void) {
    if (++street_t < 15) return;
    street_t = 0;
    float bd = 40.0f * 40.0f; int bi = -1;
    float qx = freecam ? fcx : px_, qy = freecam ? fcy : py_;
    for (int i = 0; i < NSEGS; i++) {
        float x0 = SEGS[i][0], y0 = SEGS[i][1], x1 = SEGS[i][2], y1 = SEGS[i][3];
        if ((x0 < qx - 40 && x1 < qx - 40) || (x0 > qx + 40 && x1 > qx + 40) ||
            (y0 < qy - 40 && y1 < qy - 40) || (y0 > qy + 40 && y1 > qy + 40)) continue;     /* далекий відрізок */
        float dx = x1 - x0, dy = y1 - y0, L = dx * dx + dy * dy;
        float t = L > 0 ? ((qx - x0) * dx + (qy - y0) * dy) / L : 0; if (t < 0) t = 0; if (t > 1) t = 1;
        float ex = x0 + dx * t - qx, ey = y0 + dy * t - qy, d = ex * ex + ey * ey;
        if (d < bd) { bd = d; bi = SEGS[i][4]; }
    }
    if (bi >= 0) street_i = bi;
}

/* ================= пам'ятки ================= */
static void update_landmarks(void) {
    float qx = freecam ? fcx : px_, qy = freecam ? fcy : py_;
    for (int i = 0; i < NLM; i++) {
        float d = dist(qx, qy, LMS[i].x, LMS[i].y);
        int near = d < LMS[i].r;
        if (near && !freecam && !lm_found[i]) {
            lm_found[i] = 1; lm_count++; money += LM_REWARD;
            char b[48]; snprintf(b, sizeof b, "ПАМ'ЯТКА %d/%d  +%d ГРН", lm_count, NLM, LM_REWARD);
            say(b, C_yellow_3, 170);
            write_save();
        }
        if (near && (lm_found[i] || freecam)) { if (lm_lab[i] < 60) lm_lab[i]++; }
        else if (lm_lab[i] > 0) lm_lab[i] = lm_lab[i] > 14 ? 14 : lm_lab[i] - 1;
    }
}
static const u8 *lm_frame(int i) {
    int st = (LMS[i].w * LMS[i].h + 3) & ~3;
    int f = LMS[i].nfr > 1 ? (tick / 7 + i) % LMS[i].nfr : 0;
    return assets_bin + LMS[i].off + f * st;
}
static void draw_landmarks(int ox, int oy) {
    for (int i = 0; i < NLM; i++) {
        int sx = LMS[i].x - ox, sy = LMS[i].y - oy, w = LMS[i].w, h = LMS[i].h;
        if (sx < -w || sx > SW + w || sy < -4 || sy > SH + h) continue;
        int bob = (int)floorf(sinf(tick * 0.07f + i * 1.7f) * 1.5f + 0.5f);
        shadow_ell(sx + 1.5f, sy + 0.5f, w * 0.45f, 3);
        blit(lm_frame(i), w, h, sx - w / 2, sy - h + 1 + bob);
    }
}
/* табличка з назвою: виїжджає збоку від позначки, назва друкується по літерах */
static void draw_lm_labels(int ox, int oy) {
    char b[96];
    for (int i = 0; i < NLM; i++) {
        int t = lm_lab[i]; if (!t) continue;
        int sx = LMS[i].x - ox, sy = LMS[i].y - oy, w = LMS[i].w, h = LMS[i].h;
        int bob = (int)floorf(sinf(tick * 0.07f + i * 1.7f) * 1.5f + 0.5f);
        const char *nm = LMS[i].name;
        int full = tw8(nm, 1) + 12;
        float k = t >= 10 ? 1.0f : t / 10.0f;
        int pw = (int)(full * k), ph = 14;
        if (pw < 8) continue;
        int py0 = sy - h + bob;
        if (py0 < 40) py0 = 40; if (py0 > SH - 50) py0 = SH - 50;
        /* праворуч від позначки; якщо не влазить або наїжджає на мінікарту/гроші — ліворуч; інакше нижче мінікарти */
        int xr = sx + w / 2 + 4, xl = sx - w / 2 - 4 - full;
        #define LM_BAD(x) ((x) < 4 || (x) + full > SW - 4 || ((x) + full > 408 && py0 < 84) || ((x) < 80 && py0 < 34))
        int px0 = xr;
        if (LM_BAD(xr)) { if (!LM_BAD(xl)) px0 = xl; else { px0 = xr + full > SW - 4 ? SW - 4 - full : xr; if (px0 < 4) px0 = 4; if (px0 + full > 408 && py0 < 84) py0 = 84; } }
        if (px0 == xl) px0 = xl + full - pw;         /* ліворуч — виїжджає від позначки */
        rect(px0, py0, pw, ph, C_ink_0);
        if (pw > 2) { rect(px0 + 1, py0 + 1, pw - 2, ph - 2, C_panel_1); rect(px0 + 1, py0 + 1, pw - 2, 1, C_yellow_3); }
        int nch = t > 8 ? (t - 8) * 2 / 3 + 1 : 0, n = 0; const char *p = nm, *q = nm;
        while (*p && n < nch) { utf8next(&p); n++; }
        int len = (int)(p - q); if (len > (int)sizeof b - 1) len = sizeof b - 1;
        memcpy(b, q, (size_t)len); b[len] = 0;
        if (k >= 1.0f) text8(px0 + 6, py0 + 4, 1, C_white_2, b);
    }
}

/* ================= HUD ================= */
static void panel(int x, int y, int w, int h) {
    rect(x, y, w, h, C_ink_0);
    rect(x + 1, y + 1, w - 2, h - 2, C_panel_1);
    rect(x + 1, y + 1, w - 2, 1, C_ink_2);
}

static void draw_target(float tx, float ty, const u8 *icon, u8 col, u8 coldark) {
    float sx = tx - (int)camx, sy = ty - (int)camy;
    if (sx > 8 && sx < SW - 8 && sy > 20 && sy < SH - 44) {
        float r = 13 + sinf(tick * 0.15f) * 2.5f;
        ring(sx, sy, r + 1, C_ink_0, 1);
        ring(sx, sy, r, col, 2);
        ring(sx, sy, r - 2, coldark, 1);
        int bob = (int)(sinf(tick * 0.12f) * 3);
        tri(sx - 4, sy - 16 + bob, sx + 4, sy - 16 + bob, sx, sy - 10 + bob, C_ink_0);
        blit(icon, ICON_SZ, ICON_SZ, (int)sx - 8, (int)sy - 32 + bob);
        return;
    }
    /* за межами екрана напрямок показує радар */
}

/* текст, обрізаний по ширині (пропорційний шрифт) */
static void text_clip(int x, int y, int maxw, u8 col, const char *s) {
    char b[96]; int n = 0, w = 0; const char *p = s;
    while (*p && n < 90) { const char *q = p; const Glyph *g = glyph(1, utf8next(&p)); int aw = g ? g->adv : 0;
        if (w + aw > maxw) break; memcpy(b + n, q, (size_t)(p - q)); n += (int)(p - q); w += aw; }
    b[n] = 0; text8(x, y, 1, col, b);
}
#define DASH_H 38
/* світлодіодна шкала: 10 сегментів */
static void leds(int x, int y, float v, u8 on, u8 lowc) {
    int n = (int)(v / 10.0f + 0.5f);
    for (int i = 0; i < 10; i++) {
        u8 c = i < n ? (n <= 2 ? ((tick / 10) % 2 ? lowc : C_ink_2) : on) : C_ink_1;
        rect(x + i * 5, y, 4, 5, c);
    }
}
/* радар: північ угорі, як карта; ціль — стрілка на краю або точка; пам'ятка поруч — блакитна зірочка */
static void radar_blip(int cx, int cy, int R, float tx, float ty, float range, u8 col, int star) {
    float dx = tx - px_, dy = ty - py_, d = sqrtf(dx * dx + dy * dy);
    float r = d / range * (R - 3);
    if (d < 1) return;
    float ux = dx / d, uy = dy / d;
    if (r > R - 4) {
        float ex = cx + ux * (R - 2), ey = cy + uy * (R - 2), nx = -uy, ny = ux;
        tri(ex + ux * 3, ey + uy * 3, ex - ux * 5 + nx * 4, ey - uy * 5 + ny * 4, ex - ux * 5 - nx * 4, ey - uy * 5 - ny * 4, C_ink_0);
        tri(ex + ux * 2, ey + uy * 2, ex - ux * 4 + nx * 3, ey - uy * 4 + ny * 3, ex - ux * 4 - nx * 3, ey - uy * 4 - ny * 3, col);
    } else {
        int x = cx + (int)(ux * r), y = cy + (int)(uy * r);
        if (star) { rect(x - 2, y, 5, 1, col); rect(x, y - 2, 1, 5, col); pset(x - 1, y - 1, col); pset(x + 1, y + 1, col); pset(x + 1, y - 1, col); pset(x - 1, y + 1, col); }
        else { rect(x - 2, y - 2, 5, 5, C_ink_0); rect(x - 1, y - 1, 3, 3, col); }
    }
}
static void draw_radar(int cx, int cy, int R) {
    disc(cx + 0.5f, cy + 0.5f, R + 1, C_ink_0);
    disc(cx + 0.5f, cy + 0.5f, R, C_panel_0);
    ring(cx + 0.5f, cy + 0.5f, R, C_ink_2, 1);
    ring(cx + 0.5f, cy + 0.5f, R / 2, C_panel_1, 1);
    rect(cx, cy - R + 2, 1, 3, C_white_0);                                 /* північ */
    /* розгортка */
    float sw = tick * 0.06f;
    for (int i = 3; i < R - 1; i++) pset(cx + (int)(cosf(sw) * i), cy + (int)(sinf(sw) * i), C_blue_1);
    const float RANGE = 800;                                                /* 400 м до краю */
    /* пам'ятка поруч (ще не відкрита) */
    int li = -1; float ld = 900;
    for (int i = 0; i < NLM; i++) if (!lm_found[i]) { float d = dist(px_, py_, LMS[i].x, LMS[i].y); if (d < ld) { ld = d; li = i; } }
    if (li >= 0 && (tick / 14) % 3) radar_blip(cx, cy, R, LMS[li].x, LMS[li].y, RANGE, C_blue_3, 1);
    /* АЗС / СТО, коли треба */
    if (fuel < 25 || health < 50) {
        const Sto *L = fuel < 25 ? FUELS : STOS; int n = fuel < 25 ? NFUEL : NSTO, b = 0;
        for (int i = 1; i < n; i++) if (dist(px_, py_, L[i].x, L[i].y) < dist(px_, py_, L[b].x, L[b].y)) b = i;
        if ((tick / 12) % 2) radar_blip(cx, cy, R, L[b].x, L[b].y, RANGE, fuel < 25 ? C_green_2 : C_red_3, 0);
    }
    /* ціль */
    if (ostate == O_PICK || ostate == O_LOAD) radar_blip(cx, cy, R, SHOPS[shop_i].x, SHOPS[shop_i].y, RANGE, C_green_3, 0);
    else if (ostate == O_DROP) radar_blip(cx, cy, R, ADDRS[addr_i].x, ADDRS[addr_i].y, RANGE, C_yellow_3, 0);
    /* машина */
    rect(cx - 1, cy - 1, 3, 3, C_white_2);
    pset(cx + (int)roundf(cosf(ang) * 4), cy + (int)roundf(sinf(ang) * 4), C_white_2);
}
static void note_paper(int x, int y, int w, int h) {
    rect(x + 2, y + 2, w, h, C_ink_0);
    rect(x, y, w, h, C_box_4); rect(x, y, w, 3, C_box_3);
    for (int i = x + 5; i < x + w - 4; i += 3) pset(i, y + 1, C_box_2);              /* «пружинка» блокнота */
}
static void km_str(char *b, int n, float dm) {
    if (dm >= 1000) snprintf(b, n, "%d,%d КМ", (int)(dm / 1000), (int)(dm / 100) % 10); else snprintf(b, n, "%d М", (int)dm);
}
static void draw_hud(void) {
    char b[200], w[32];
    const int y0 = SH - DASH_H, NW = 298;
    u8 ink = C_ink_1;
    snprintf(w, sizeof w, "%d ГРН", money);
    u8 wc = money < 0 ? C_red_1 : C_green_0;
    /* вгорі: таймер ліворуч, вулиця по центру, радар праворуч, відстань під ним */
    if (ostate == O_DROP) {
        int s_ = otimer / 60; snprintf(b, sizeof b, "%d:%02d", s_ / 60, s_ % 60);
        otext8(8, 9, 2, s_ < 15 && (tick / 8) % 2 ? C_red_3 : C_white_2, C_ink_0, b);
    }
    if (street_i >= 0) otext8_c(SW / 2, 8, 1, C_white_2, C_ink_0, STREETS[street_i]);
    draw_radar(SW - 34, 36, 28);
    if (ostate != O_NONE) {
        float tx = ostate == O_DROP ? ADDRS[addr_i].x : SHOPS[shop_i].x, ty = ostate == O_DROP ? ADDRS[addr_i].y : SHOPS[shop_i].y;
        km_str(b, sizeof b, dist(px_, py_, tx, ty) * MPP);
        otext8_c(SW - 34, 70, 1, ostate == O_DROP ? C_yellow_3 : C_green_3, C_ink_0, b);
    }
    /* нижня смуга: ліворуч нотатник, праворуч прилади */
    rect(0, y0, SW, DASH_H, C_panel_0);
    rect(0, y0, SW, 1, C_ink_2); rect(NW + 1, y0 + 1, 1, DASH_H - 1, C_ink_2);
    if (!note_open) {
        note_paper(3, y0 + 3, NW - 6, DASH_H - 7);
        const int tx = 9, l1 = y0 + 9, l2 = y0 + 22;
        if (ostate == O_PICK || ostate == O_LOAD) {
            text8(tx, l1, 1, C_green_0, "ЗАБРАТИ:"); text_clip(tx + tw8("ЗАБРАТИ:", 1) + 4, l1, NW - 20 - tw8("ЗАБРАТИ:", 1), ink, SHOPS[shop_i].name);
            snprintf(b, sizeof b, "%s · %s", TNAME[otype], SHOPS[shop_i].kind); text_clip(tx, l2, NW - 30 - tw8(w, 1), C_ink_2, b);
        } else if (ostate == O_DROP) {
            text8(tx, l1, 1, C_red_1, "ВЕЗУ:"); text_clip(tx + tw8("ВЕЗУ:", 1) + 4, l1, NW - 20 - tw8("ВЕЗУ:", 1), ink, ADDRS[addr_i].name);
            snprintf(b, sizeof b, "ВІД: %s", SHOPS[shop_i].name); text_clip(tx, l2, NW - 30 - tw8(w, 1), C_ink_2, b);
        } else if (n_offers) {
            text8(tx, l1, 1, ink, "ВІЛЬНИЙ ЧАС");
            text8(tx, l2, 1, (tick / 20) % 2 ? C_red_1 : C_ink_2, "ТРИКУТНИК: 3 НОВІ ЗАМОВЛЕННЯ");
        } else {
            text8(tx, l1, 1, ink, "ВІЛЬНИЙ ЧАС"); text8(tx, l2, 1, C_ink_2, "ЧЕКАЮ НА ЗАМОВЛЕННЯ...");
        }
        text8(NW - 8 - tw8(w, 1), l2, 1, wc, w);
    } else {
        /* висунутий нотатник */
        const int H = ostate == O_NONE ? 116 : 104, ny = SH - H - 3, tx = 9;
        note_paper(3, ny, NW - 6, H);
        text8(NW - 8 - tw8(w, 1), ny + 7, 1, wc, w);
        if (ostate == O_NONE) {
            text8(tx, ny + 7, 1, ink, n_offers ? "НОВІ ЗАМОВЛЕННЯ" : "ЗАМОВЛЕНЬ ПОКИ НЕМАЄ");
            for (int k = 0; k < (n_offers ? 3 : 0); k++) {
                const Offer *o = &offers[k]; int y = ny + 21 + k * 25;
                if (k == offer_sel) rect(5, y - 3, NW - 12, 25, C_box_3);
                text8(tx, y, 1, k == offer_sel ? C_red_1 : C_ink_2, k == offer_sel ? ">" : " ");
                u8 tc_ = tcol(o->type) == C_red_3 ? C_red_1 : (tcol(o->type) == C_yellow_3 ? C_box_1 : C_green_0);
                text8(tx + 9, y, 1, tc_, TNAME[o->type]);
                text_clip(tx + 15 + tw8(TNAME[o->type], 1), y, NW - 40 - tw8(TNAME[o->type], 1), ink, SHOPS[o->shop].name);
                km_str(b, sizeof b, dist(SHOPS[o->shop].x, SHOPS[o->shop].y, ADDRS[o->addr].x, ADDRS[o->addr].y) * MPP);
                int s_ = o->limit / 60; char t2[64];
                snprintf(t2, sizeof t2, "%s   ·   %d ГРН   ·   %d:%02d", b, o->pay, s_ / 60, s_ % 60);
                text8(tx + 9, y + 11, 1, C_ink_2, t2);
            }
            text8(tx, ny + H - 15, 1, C_ink_2, "ТРИКУТНИК: ВЗЯТИ     КОЛО: СХОВАТИ");
        } else {
            snprintf(b, sizeof b, "ЗАМОВЛЕННЯ: %s", TNAME[otype]); text8(tx, ny + 7, 1, ink, b);
            snprintf(b, sizeof b, "ЗВІДКИ: %s (%s)", SHOPS[shop_i].name, SHOPS[shop_i].kind); text_clip(tx, ny + 22, NW - 20, C_ink_2, b);
            snprintf(b, sizeof b, "КУДИ: %s", ADDRS[addr_i].name); text_clip(tx, ny + 35, NW - 20, C_ink_2, b);
            if (otype == T_FRAGILE) snprintf(b, sizeof b, "ЗАРОБЛЮ: %d ГРН + БОНУС ЗА ЧАС   ВАНТАЖ %d%%", pay, (int)cargo);
            else snprintf(b, sizeof b, "ЗАРОБЛЮ: %d ГРН + БОНУС ЗА ЧАС", pay);
            text_clip(tx, ny + 48, NW - 20, C_green_0, b);
            const char *opt[2] = { "СХОВАТИ НОТАТНИК", "ВІДМОВИТИСЯ ВІД ЗАМОВЛЕННЯ" };
            for (int k = 0; k < 2; k++) {
                int y = ny + 66 + k * 15;
                if (k == note_sel) rect(5, y - 3, NW - 12, 14, C_box_3);
                text8(tx, y, 1, k == note_sel ? (k ? C_red_1 : ink) : C_ink_2, opt[k]);
            }
        }
    }
    /* прилади */
    float kmh = fabsf(vel) * MPP * 60 * 3.6f;
    int ik = (int)(kmh + 0.5f); if (ik > 140) ik = 140;               /* фізичний максимум ~141: показуємо рівно 140 */
    snprintf(b, sizeof b, "%d", ik);
    text8(NW + 66 - tw8(b, 2), y0 + 11, 2, ik >= 130 ? C_box_3 : C_white_2, b);
    text8(NW + 70, y0 + 19, 1, C_white_0, "КМ/Г");
    text8(NW + 106, y0 + 9, 1, C_white_0, "П");
    leds(NW + 118, y0 + 10, fuel, C_yellow_2, C_red_3);
    text8(NW + 106, y0 + 23, 1, C_white_0, "А");
    leds(NW + 118, y0 + 24, health, health > 60 ? C_green_2 : (health > 30 ? C_yellow_2 : C_red_2), C_red_3);
    /* прогрес: ремонт / заправка / завантаження */
    const char *prog = repair_t > 0 ? "РЕМОНТ НА СТО" : (fuel_t > 0 ? "ЗАПРАВКА" : (ostate == O_LOAD ? "ЗАБИРАЮ ЗАМОВЛЕННЯ" : NULL));
    if (prog) {
        int ww = 120, x = SW / 2 - ww / 2, y = 150, t = repair_t > 0 ? repair_t : (fuel_t > 0 ? fuel_t : load_t * 70 / 60);
        otext8_c(SW / 2, y - 15, 1, C_white_2, C_ink_0, prog);
        rect(x - 2, y - 2, ww + 4, 10, C_ink_0); rect(x, y, ww * t / 70, 6, C_yellow_3);
    }
    if (fuel <= 0 && fabsf(vel) < 0.05f && !evac) otext8_c(SW / 2, 172, 1, C_red_3, C_ink_0, "ПАЛЬНЕ СКІНЧИЛОСЯ. КОЛО: ЕВАКУАТОР ДО АЗС (300 ГРН)");
    if (toast_t > 0) {
        int sc = tw8(toast, 2) > 440 ? 1 : 2;
        otext8_c(SW / 2, 50, sc, toast_c, C_ink_0, toast);
    }
}

static void update_particles(void) {
    for (int i = 0; i < NPART; i++) if (part[i].life > 0) { part[i].x += part[i].dx; part[i].y += part[i].dy; part[i].dx *= 0.95f; part[i].dy *= 0.95f; part[i].life--; }
}
static void draw_particles(int ox, int oy) {
    for (int i = 0; i < NPART; i++) {
        if (part[i].life <= 0) continue;
        float t = 1.0f - (float)part[i].life / part[i].max;         /* 0 -> 1 */
        float x = part[i].x - ox, y = part[i].y - oy;
        if (part[i].kind == 3) { ring(x, y - t * 6, 1.5f + t * 2, C_white_2, 1); continue; }      /* бульбашки */
        float r = part[i].kind == 1 ? 1.5f + t * 2 : 2.0f + t * 4.5f;
        u8 c = part[i].kind == 2 ? (t < 0.5f ? C_ink_2 : C_white_0) : (t < 0.4f ? C_white_2 : (t < 0.75f ? C_white_1 : C_white_0));
        if (part[i].kind == 1) c = C_yellow_3;
        /* «прозорість» — розріджена шахівниця в кінці життя */
        int sparse = t > 0.7f;
        for (int yy = (int)(y - r); yy <= (int)(y + r); yy++) for (int xx = (int)(x - r); xx <= (int)(x + r); xx++) {
            if ((xx - x) * (xx - x) + (yy - y) * (yy - y) > r * r) continue;
            if (sparse && ((xx + yy + tick) & 1)) continue;
            pset(xx, yy, c);
        }
    }
}

static void draw_world(void) {
    int ox = (int)camx, oy = (int)camy;
    if (bump > 0) { int a = shake * bump / 10 + 1; ox += (rand() % (2 * a + 1)) - a; oy += (rand() % (2 * a + 1)) - a; if (ox < 0) ox = 0; if (oy < 0) oy = 0;
        if (ox > MAP_W - SW) ox = MAP_W - SW; if (oy > MAP_H - SH) oy = MAP_H - SH; }
    draw_map(ox, oy);
    for (int i = 0; i < skid_n; i++) {                           /* сліди шин */
        int x = skid[i][0] - ox, y = skid[i][1] - oy;
        if ((unsigned)x < SW && (unsigned)y < SH) { u8 *p = fb + y * SW + x; *p = DARK[DARK[*p]]; }
    }
    for (int i = 0; i < NFUEL; i++) {                             /* АЗС */
        int x = FUELS[i].x - ox, y = FUELS[i].y - oy;
        if (x > -16 && x < SW + 16 && y > -16 && y < SH + 16) blit(assets_bin + ICON_FUEL, ICON_SZ, ICON_SZ, x - 8, y - 8);
    }
    for (int i = 0; i < NSTO; i++) {                              /* СТО */
        int x = STOS[i].x - ox, y = STOS[i].y - oy;
        if (x > -16 && x < SW + 16 && y > -16 && y < SH + 16) blit(assets_bin + ICON_STO, ICON_SZ, ICON_SZ, x - 8, y - 8);
    }
    draw_landmarks(ox, oy);
    if (freecam) {
        draw_lm_labels(ox, oy);
        int cx = SW / 2, cy = SH / 2;
        rect(cx - 7, cy, 5, 1, C_white_2); rect(cx + 3, cy, 5, 1, C_white_2);
        rect(cx, cy - 7, 1, 5, C_white_2); rect(cx, cy + 3, 1, 5, C_white_2);
        return;
    }
    float sx = px_ - ox, sy = py_ - oy;
    int fi = (int)floorf(ang / (2 * PI_F) * SPR_N + 0.5f);
    fi = ((fi % SPR_N) + SPR_N) % SPR_N;
    const u8 *spr = assets_bin + ((braking || evac) ? SPR_LIT : SPR_OFF) + fi * SPR_SZ * SPR_SZ;
    if (evac == EV_DROWN) {
        /* тоне: машина темнішає й «осідає» — видно все менше рядків */
        int keep = SPR_SZ - evac_t * SPR_SZ / 110; if (keep < 0) keep = 0;
        u8 tmp[SPR_SZ * SPR_SZ];
        for (int i = 0; i < SPR_SZ * SPR_SZ; i++) { u8 v = spr[i]; if (v) { v = DARK[v]; if (evac_t > 40) v = DARK[v]; } tmp[i] = v; }
        int y0 = (SPR_SZ - keep) / 2;
        blit(tmp + y0 * SPR_SZ, SPR_SZ, keep, (int)sx - SPR_SZ / 2, (int)sy - SPR_SZ / 2 + y0);
        ring(sx, sy, 8 + (evac_t % 30) / 3.0f, C_white_1, 1);
    } else {
        shadow_ell(sx + 2, sy + 2, 11, 6);
        blit(spr, SPR_SZ, SPR_SZ, (int)sx - SPR_SZ / 2, (int)sy - SPR_SZ / 2);
    }
    /* арки: будинок над проїздом малюється поверх машини */
    for (int i = 0; i < NARCH; i++) {
        const Arch *A = &ARCHES[i];
        int x0 = (int)px_ - 16, x1 = (int)px_ + 16, y0 = (int)py_ - 16, y1 = (int)py_ + 16;
        if (x1 < A->x0 || x0 > A->x1 || y1 < A->y0 || y0 > A->y1) continue;
        for (int y = y0; y <= y1; y++) for (int x = x0; x <= x1; x++) {
            int in = 0;
            for (int a = 0, b = A->n - 1; a < A->n; b = a++) {
                int ya = A->p[a][1], yb = A->p[b][1];
                if ((ya > y) != (yb > y) && x < (A->p[b][0] - A->p[a][0]) * (y - ya) / (float)(yb - ya) + A->p[a][0]) in = !in;
            }
            if (in) pset(x - ox, y - oy, map_px(x, y));
        }
    }
    draw_particles(ox, oy);
    draw_lm_labels(ox, oy);
}

static void draw_bigmap(void) {
    memset(fb, C_ink_0, sizeof fb);
    int ox = (SW - BIG_W) / 2, oy = (SH - BIG_H) / 2;
    for (int y = 0; y < BIG_H; y++) memcpy(fb + (oy + y) * SW + ox, bigb + y * BIG_W, BIG_W);
    float qx = freecam ? fcx : px_, qy = freecam ? fcy : py_;
    int mx = ox + (int)(qx / BIG_S), my = oy + (int)(qy / BIG_S);
    if (ostate == O_PICK || ostate == O_LOAD) { int tx = ox + SHOPS[shop_i].x / BIG_S, ty = oy + SHOPS[shop_i].y / BIG_S; rect(tx - 2, ty - 2, 5, 5, C_ink_0); rect(tx - 1, ty - 1, 3, 3, C_green_3); }
    if (ostate == O_DROP) { int tx = ox + ADDRS[addr_i].x / BIG_S, ty = oy + ADDRS[addr_i].y / BIG_S; rect(tx - 2, ty - 2, 5, 5, C_ink_0); rect(tx - 1, ty - 1, 3, 3, C_yellow_3); }
    for (int i = 0; i < NLM; i++) if (lm_found[i]) {
        int tx = ox + LMS[i].x / BIG_S, ty = oy + LMS[i].y / BIG_S;
        rect(tx - 2, ty, 5, 1, C_ink_0); rect(tx, ty - 2, 1, 5, C_ink_0); rect(tx - 1, ty, 3, 1, C_yellow_3); rect(tx, ty - 1, 1, 3, C_yellow_3);
    }
    rect(mx - 3, my - 3, 7, 7, C_ink_0);
    rect(mx - 2, my - 2, 5, 5, (tick / 8) % 2 ? C_red_3 : C_white_2);
    text8(6, 6, 1, C_white_2, "КАРТА");
    { char b[40]; snprintf(b, sizeof b, "ПАМ'ЯТКИ: %d/%d", lm_count, NLM); text8(SW - 8 - tw8(b, 1), 6, 1, C_yellow_3, b); }
    text8(6, SH - 22, 1, C_white_0, "SELECT: НАЗАД");
    text8(6, SH - 10, 1, C_white_0, "(C) OSM, MICROSOFT BUILDING FOOTPRINTS");
}

static void draw_frame(void) {
    char b[64];
    if (state == S_TITLE) {
        camx = SPAWN_X - SW / 2 + sinf(tick * 0.004f) * 300; camy = SPAWN_Y - SH / 2 - 200 + cosf(tick * 0.003f) * 180;
        if (camx < 0) camx = 0; if (camy < 0) camy = 0;
        if (camx > MAP_W - SW) camx = MAP_W - SW; if (camy > MAP_H - SH) camy = MAP_H - SH;
        draw_map((int)camx, (int)camy);
        darken_rect(0, 30, SW, 150, 2);
        otext8_c(SW / 2, 46, 5, C_box_3, C_ink_0, "КУР'ЄР");
        otext8_c(SW / 2, 100, 3, C_white_2, C_ink_0, "ХМЕЛЬНИЦЬКИЙ");
        otext8_c(SW / 2, 140, 1, C_yellow_3, C_ink_0, "ЗАБЕРИ ЗАМОВЛЕННЯ - ДОВЕЗИ ВЧАСНО");
        if ((tick / 30) % 2) otext8_c(SW / 2, 196, 2, C_white_2, C_ink_0, "НАТИСНИ СТАРТ");
        otext8_c(SW / 2, 220, 1, C_white_1, C_ink_0, "X: ГАЗ   КВАДРАТ: ГАЛЬМО / НАЗАД   R: РУЧНИК");
        otext8_c(SW / 2, 234, 1, C_white_1, C_ink_0, "СТІК: КЕРМО   L: ОГЛЯД   SELECT: КАРТА   START: ПАУЗА");
        otext8_c(SW / 2, 249, 1, C_white_0, C_ink_0, "КАРТА: (C) OPENSTREETMAP CONTRIBUTORS");
        if (money) { snprintf(b, sizeof b, "ЗАРОБЛЕНО: %d ГРН  ДОСТАВОК: %d", money, deliveries); otext8_c(SW / 2, 172, 1, C_green_3, C_ink_0, b); }
        return;
    }
    if (map_err) {
        memset(fb, C_ink_0, sizeof fb);
        otext8_c(SW / 2, 90, 2, C_red_3, C_ink_0, map_err == 1 ? "НЕМАЄ ФАЙЛУ MAP.BIN" : "MAP.BIN НЕ ВІД ЦІЄЇ ВЕРСІЇ");
        otext8_c(SW / 2, 130, 1, C_white_2, C_ink_0, "ПОКЛАДИ MAP.BIN ПОРУЧ З EBOOT.PBP");
        otext8_c(SW / 2, 146, 1, C_white_1, C_ink_0, "PSP/GAME/KURIERKHM/");
        return;
    }
    if (state == S_BIGMAP) { draw_bigmap(); return; }
    draw_world();
    if (freecam) {
        if (street_i >= 0) otext8_c(244, 10, 1, C_white_2, C_ink_0, STREETS[street_i]);
        panel(0, SH - 18, SW, 18);   /* огляд: без приладів */
        text8(8, SH - 13, 1, C_yellow_3, "ОГЛЯД МІСТА");
        text8(118, SH - 13, 1, C_white_1, "СТІК: РУХ  X: ШВИДКО  L: ДО МАШИНИ");
        return;
    }
    if (ostate == O_PICK || ostate == O_LOAD) draw_target(SHOPS[shop_i].x, SHOPS[shop_i].y, assets_bin + ICON_SHOP, C_green_2, C_green_0);
    else if (ostate == O_DROP) draw_target(ADDRS[addr_i].x, ADDRS[addr_i].y, assets_bin + ICON_HOME, C_yellow_2, C_yellow_0);
    draw_hud();
    if (state == S_PAUSE) {
        darken_rect(0, 0, SW, SH - DASH_H, 2);
        otext8_c(SW / 2, 56, 4, C_white_2, C_ink_0, "ПАУЗА");
        snprintf(b, sizeof b, "ЗАРОБЛЕНО: %d ГРН", money); otext8_c(SW / 2, 100, 1, C_yellow_3, C_ink_0, b);
        snprintf(b, sizeof b, "ДОСТАВОК: %d   ЗАПІЗНЕНЬ: %d", deliveries, late); otext8_c(SW / 2, 114, 1, C_white_2, C_ink_0, b);
        snprintf(b, sizeof b, "ПАМ'ЯТКИ МІСТА: %d/%d", lm_count, NLM); otext8_c(SW / 2, 128, 1, C_yellow_3, C_ink_0, b);
        const char *items[2] = { "ГРАТИ ДАЛІ", lane_assist ? "ПОМІЧНИК СМУГИ: УВІМК" : "ПОМІЧНИК СМУГИ: ВИМК" };
        for (int i = 0; i < 2; i++) {
            int y = 156 + i * 18;
            if (i == menu_i) { snprintf(b, sizeof b, "> %s <", items[i]); otext8_c(SW / 2, y, 1, C_yellow_3, C_ink_0, b); }
            else otext8_c(SW / 2, y, 1, C_white_1, C_ink_0, items[i]);
        }
        otext8_c(SW / 2, 214, 1, C_white_0, C_ink_0, "ВГОРУ/ВНИЗ, X: ВИБРАТИ   R: РУЧНИК");
    }
}

static void game_init(void) {
    px_ = SPAWN_X; py_ = SPAWN_Y; ang = 0; stop_car();
    /* старт у випадковому місці міста: середина випадкового відрізка звичайної дороги */
    for (int t = 0; t < 400 && NLANE > 0; t++) {
        const int16_t *L = LANES[rand() % NLANE];
        if (L[4] < 3) continue;
        float x = (L[0] + L[2]) * 0.5f, y = (L[1] + L[3]) * 0.5f, a = atan2f(L[3] - L[1], L[2] - L[0]);
        if (!hit(x, y, a) && !on_water(x, y)) { px_ = x; py_ = y; ang = a; break; }
    }
    for (int r = 0; r < 200 && hit(px_, py_, ang); r += 2) { px_ += 2; }
    update_camera(1);
}

static void game_update(unsigned btn, unsigned pressed, int lx, int ly) {
    tick++;
    loads_frame = 0;
    if (toast_t > 0) toast_t--;
    if (bump > 0) bump--;
    if (map_err) return;
    if (state == S_TITLE) {
        if (pressed & (PSP_CTRL_START | PSP_CTRL_CROSS)) { state = S_PLAY; game_init(); owait = 60; }
        return;
    }
    if (state == S_BIGMAP) { if (pressed & (PSP_CTRL_SELECT | PSP_CTRL_START | PSP_CTRL_CIRCLE)) state = S_PLAY; return; }
    if (state == S_PAUSE) {
        if (pressed & (PSP_CTRL_UP | PSP_CTRL_DOWN)) menu_i ^= 1;
        if (pressed & PSP_CTRL_START) state = S_PLAY;
        if (pressed & PSP_CTRL_CROSS) {
            if (menu_i == 0) state = S_PLAY;
            else { lane_assist = !lane_assist; write_save(); }
        }
        return;
    }
    if (pressed & PSP_CTRL_START) { state = S_PAUSE; menu_i = 0; return; }
    if (pressed & PSP_CTRL_SELECT) { state = S_BIGMAP; return; }
    update_particles();
    if (evac) { update_evac(); update_camera(0); update_street(); prefetch((int)camx, (int)camy, 0, 0); return; }
    if (pressed & PSP_CTRL_LTRIGGER) {
        freecam = !freecam;
        if (freecam) { fcx = camx + SW / 2; fcy = camy + SH / 2; stop_car(); }
        else update_camera(1);
    }
    if (freecam) update_freecam(btn, lx, ly);
    else {
        int was_open = note_open;
        notebook_input(pressed);
        update_player(btn, lx); update_camera(0);
        update_orders(was_open ? pressed : (pressed & ~PSP_CTRL_TRIANGLE)); update_repair();
        update_fuel(note_open ? 0 : pressed);
    }
    update_street();
    update_landmarks();
    prefetch((int)camx, (int)camy, freecam ? 0 : cosf(ang) * vel, freecam ? 0 : sinf(ang) * vel);
}

static void assets_init(void) {
    cache_init();
    map_open();
    uLongf l = BIG_W * BIG_H; uncompress(bigb, &l, assets_bin + BIG_OFF, BIG_ZLEN);
}

/* ================= platform ================= */
static void present(u32 *dst) {
    const u8 *s = fb;
    for (int y = 0; y < SH; y++) {
        u32 *d = dst + y * STRIDE;
        for (int x = 0; x < SW; x += 4) { d[x] = PALETTE[s[x]]; d[x + 1] = PALETTE[s[x + 1]]; d[x + 2] = PALETTE[s[x + 2]]; d[x + 3] = PALETTE[s[x + 3]]; }
        s += SW;
    }
}

#ifndef HOST
static int exit_cb(int a, int b, void *c) { write_save(); sceKernelExitGame(); return 0; }
static int cb_thread(SceSize args, void *argp) {
    int id = sceKernelCreateCallback("exit", exit_cb, NULL);
    sceKernelRegisterExitCallback(id);
    sceKernelSleepThreadCB();
    return 0;
}
static u32 *vw[2];
static void *vp[2];
static int back = 1;
static void flip(void) {
    present(vw[back]);
    sceDisplayWaitVblankStart();
    sceDisplaySetFrameBuf(vp[back], STRIDE, PSP_DISPLAY_PIXEL_FORMAT_8888, PSP_DISPLAY_SETBUF_IMMEDIATE);
    back ^= 1;
}
int main(int argc, char **argv) {
    int t = sceKernelCreateThread("cb", cb_thread, 0x11, 0xFA0, 0, NULL);
    if (t >= 0) sceKernelStartThread(t, 0, NULL);
    /* display wake-up sequence proven on real hardware */
    sceDisplaySetMode(0, SW, SH);
    u8 *vbase = (u8 *)sceGeEdramGetAddr();
    vw[0] = (u32 *)((uintptr_t)vbase | 0x40000000); vw[1] = (u32 *)((uintptr_t)(vbase + BUFSZ) | 0x40000000);
    vp[0] = (void *)vbase; vp[1] = (void *)(vbase + BUFSZ);
    for (int i = 0; i < STRIDE * SH; i++) { vw[0][i] = 0xFF000000; vw[1][i] = 0xFF000000; }
    sceDisplaySetFrameBuf((void *)vw[0], STRIDE, PSP_DISPLAY_PIXEL_FORMAT_8888, PSP_DISPLAY_SETBUF_NEXTFRAME);
    sceDisplayWaitVblankStart(); sceDisplayWaitVblankStart(); sceDisplayWaitVblankStart();
    sceDisplaySetFrameBuf(vp[0], STRIDE, PSP_DISPLAY_PIXEL_FORMAT_8888, PSP_DISPLAY_SETBUF_IMMEDIATE);
    sceDisplayWaitVblankStart();
    sceDisplaySetFrameBuf((void *)vw[0], STRIDE, PSP_DISPLAY_PIXEL_FORMAT_8888, PSP_DISPLAY_SETBUF_NEXTFRAME);
    sceDisplayWaitVblankStart(); sceDisplayWaitVblankStart();

    if (argc > 0 && argv[0]) {
        const char *sl = strrchr(argv[0], '/');
        if (sl) { size_t n = (size_t)(sl - argv[0]) + 1; if (n < 480) { memcpy(save_path, argv[0], n); strcpy(save_path + n, "save.dat");
                                                               memcpy(map_path, argv[0], n); strcpy(map_path + n, "MAP.BIN"); } }
    }
    load_save();
    assets_init();
    sceCtrlSetSamplingCycle(0);
    sceCtrlSetSamplingMode(PSP_CTRL_MODE_ANALOG);
    srand(sceKernelGetSystemTimeLow());
    SceCtrlData pad; unsigned old = 0;
    while (1) {
        sceCtrlPeekBufferPositive(&pad, 1);
        unsigned pressed = pad.Buttons & ~old;
        old = pad.Buttons;
        game_update(pad.Buttons, pressed, pad.Lx, pad.Ly);
        draw_frame();
        flip();
    }
    return 0;
}
#else
static u32 hostfb[STRIDE * SH];
static void dump(const char *name) {
    present(hostfb);
    FILE *f = fopen(name, "wb"); fprintf(f, "P6\n%d %d\n255\n", SW, SH);
    for (int y = 0; y < SH; y++) for (int x = 0; x < SW; x++) {
        u32 c = hostfb[y * STRIDE + x]; fputc(c & 255, f); fputc((c >> 8) & 255, f); fputc((c >> 16) & 255, f);
    }
    fclose(f);
}
int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : ".";
    char nm[600];
    srand(5);
    snprintf(save_path, sizeof save_path, "%s/save_host.dat", dir);
    remove(save_path);
    snprintf(map_path, sizeof map_path, "%s", argc > 2 ? argv[2] : "build/MAP.BIN");
    assets_init();
    if (map_err) { printf("MAP.BIN error %d\n", map_err); return 1; }
    /* title */
    for (int i = 0; i < 40; i++) { game_update(0, 0, 128, 128); }
    draw_frame(); snprintf(nm, sizeof nm, "%s/f_title.ppm", dir); dump(nm);
    game_update(0, PSP_CTRL_START, 128, 128);
    /* simple autopilot toward the current target */
    int shots = 0, stuck = 0; float lastx = 0, lasty = 0;
    for (int f = 0; f < 9000; f++) {
        float tx = 0, ty = 0;
        if (ostate == O_PICK || ostate == O_LOAD) { tx = SHOPS[shop_i].x; ty = SHOPS[shop_i].y; }
        else if (ostate == O_DROP) { tx = ADDRS[addr_i].x; ty = ADDRS[addr_i].y; }
        unsigned btn = 0; int lx = 128;
        if (ostate == O_NONE && n_offers) {
            static int shot_off; if (!shot_off++) { draw_frame(); snprintf(nm, sizeof nm, "%s/f_offers.ppm", dir); dump(nm); }
            game_update(0, PSP_CTRL_TRIANGLE, 128, 128);
            if (shot_off == 1) { draw_frame(); snprintf(nm, sizeof nm, "%s/f_note.ppm", dir); dump(nm); }
            game_update(0, PSP_CTRL_TRIANGLE, 128, 128); fuel = 100; }
        if (ostate != O_NONE) {
            float want = atan2f(ty - py_, tx - px_), d = want - ang;
            while (d > PI_F) d -= 2 * PI_F; while (d < -PI_F) d += 2 * PI_F;
            lx = 128 + (int)(fmaxf(-1, fminf(1, d * 2)) * 127);
            float dd = dist(px_, py_, tx, ty);
            if (dd > 60) btn |= PSP_CTRL_CROSS; else if (vel > 0.2f) btn |= PSP_CTRL_SQUARE;
        }
        if (f % 120 == 0) { if (dist(px_, py_, lastx, lasty) < 6 && ostate != O_NONE) stuck++; lastx = px_; lasty = py_; }
        if (stuck > 1 && ostate != O_NONE) {     /* teleport the bot along when it gets stuck in the street maze */
            float d = dist(px_, py_, tx, ty), k = d > 200 ? 160 / d : 1;
            float nx = px_ + (tx - px_) * k, ny = py_ + (ty - py_) * k;
            if (!hit(nx, ny, ang)) { px_ = nx; py_ = ny; }
            else { px_ = tx; py_ = ty; }
            stop_car(); stuck = 0;
        }
        if (f == 400 && ostate != O_NONE) {
            game_update(0, PSP_CTRL_TRIANGLE, 128, 128); draw_frame(); snprintf(nm, sizeof nm, "%s/f_note_task.ppm", dir); dump(nm);
            game_update(0, PSP_CTRL_CIRCLE, 128, 128);
        }
        game_update(btn, 0, lx, 128);
        draw_frame();
        if ((f == 200 || f == 900 || (ostate == O_LOAD && load_t == 30) || (ostate == O_DROP && otimer == olimit - 300) || (toast_t == 140 && strstr(toast, "ГРН"))) && shots < 8) {
            snprintf(nm, sizeof nm, "%s/f_play%d.ppm", dir, shots++); dump(nm);
        }
    }
    /* стрес-тест зіткнень: «п'яний» водій 72 000 кадрів (20 хв гри), газ і випадкове кермо */
    {
        int stuck_ep = 0, still = 0, inwall = 0; float lx0 = px_, ly0 = py_;
        for (int f = 0; f < 72000; f++) {
            static int st = 128, rev = 0; if (f % 40 == 0) st = rand() % 256;
            if (still >= 1 && rev == 0) rev = 70;                       /* як гравець: уперся — здай назад */
            fuel = 100;
            game_update(rev > 0 ? PSP_CTRL_SQUARE : PSP_CTRL_CROSS, 0, rev > 0 ? 255 - st : st, 128); if (rev > 0) rev--;
            if (hit(px_, py_, ang)) inwall++;
            if (f % 60 == 0) { if (dist(px_, py_, lx0, ly0) < 3) { if (++still == 3) { stuck_ep++; printf("  stuck at %d %d ang %.2f\n", (int)px_, (int)py_, ang); } } else still = 0; lx0 = px_; ly0 = py_; }
        }
        draw_frame(); snprintf(nm, sizeof nm, "%s/f_stress.ppm", dir); dump(nm);
        printf("stress: money after drunk driving %d (evacuations included)\n", money);
        printf("stress: застрягань (3 с без руху, з газом і задньою) = %d, кадрів у стіні = %d\n", stuck_ep, inwall);
    }
    /* ---- етап 2: швидкість, занос, помічник смуги, вода, арка */
    {
        int bi = 0; float bl = 0;                                 /* найдовший прямий відрізок дороги */
        for (int i = 0; i < NLANE; i++) { float l = dist(LANES[i][0], LANES[i][1], LANES[i][2], LANES[i][3]); if (l > bl && LANES[i][4] >= 4) { bl = l; bi = i; } }
        const int16_t *L = LANES[bi];
        float ux = (L[2] - L[0]) / bl, uy = (L[3] - L[1]) / bl;
        evac = 0; health = 100; fuel = 100; ostate = O_NONE; n_offers = 0; owait = 100000;
        px_ = L[0] + ux * 10 - uy * L[4]; py_ = L[1] + uy * 10 + ux * L[4]; ang = atan2f(uy, ux); stop_car(); unstick(); update_camera(1);
        int kmin = 999, kmax = 0, t100 = -1, t140 = -1;
        for (int f = 0; f < 900 && dist(px_, py_, L[0], L[1]) < bl - 60; f++) {
            game_update(PSP_CTRL_CROSS, 0, 128, 128);
            int k = (int)(fabsf(vel) * MPP * 60 * 3.6f + 0.5f); if (k > 140) k = 140;
            if (k >= 100 && t100 < 0) t100 = f;
            if (k >= 140 && t140 < 0) t140 = f;
            if (t140 >= 0) { if (k < kmin) kmin = k; if (k > kmax) kmax = k;}
        }
        printf("speed: segment %.0f m, 100 km/h after %.1f s, 140 after %.1f s, shown after that %d..%d\n", bl * MPP, t100 / 60.0f, t140 / 60.0f, kmin, kmax);
        /* занос на ручнику */
        for (int f = 0; f < 24; f++) { game_update(PSP_CTRL_CROSS | PSP_CTRL_RTRIGGER, 0, 255, 128); draw_frame(); }
        snprintf(nm, sizeof nm, "%s/f_drift.ppm", dir); dump(nm);
        printf("drift: slip %d, skids %d\n", drifting, skid_n);
        /* помічник смуги: відхилити на 0,35 рад, керма немає */
        lane_assist = 1; px_ = L[0] + ux * 30 - uy * L[4]; py_ = L[1] + uy * 30 + ux * L[4]; ang = atan2f(uy, ux) + 0.35f; stop_car(); vx_ = cosf(ang) * 0.5f; vy_ = sinf(ang) * 0.5f;
        float e0 = 0.35f;
        for (int f = 0; f < 150; f++) game_update(PSP_CTRL_CROSS, 0, 128, 128);
        float e1 = ang - atan2f(uy, ux); while (e1 > PI_F) e1 -= 2 * PI_F; while (e1 < -PI_F) e1 += 2 * PI_F;
        float lat = (px_ - L[0]) * (-uy) + (py_ - L[1]) * ux;
        printf("lane assist: heading error %.2f -> %.2f rad, lateral %.1f px (lane centre %d), health %.0f\n", e0, e1, lat, L[4], health);
        lane_assist = 0;
    }
    {   /* вода: знайти водний піксель біля дороги й заїхати */
        int found = 0;
        for (int y = 200; y < MAP_H - 200 && !found; y += 16) for (int x = 200; x < MAP_W - 200 && !found; x += 16)
            if (WATERIDX[map_px(x, y)] && !is_solid(x - 40, y) && !WATERIDX[map_px(x - 40, y)] && WATERIDX[map_px(x + 12, y)] && WATERIDX[map_px(x + 30, y)]) {
                px_ = x - 40; py_ = y; ang = 0; stop_car(); unstick(); found = 1;
            }
        int m0 = money, drowned = 0;
        for (int f = 0; f < 300; f++) { game_update(evac ? 0 : PSP_CTRL_CROSS, 0, 128, 128); if (evac == EV_DROWN) { drowned = 1; if (evac_t == 60) { draw_frame(); snprintf(nm, sizeof nm, "%s/f_drown.ppm", dir); dump(nm); } } }
        printf("water: found %d, drowned %d, money %d -> %d\n", found, drowned, m0, money);
        int a = 0; for (int k = -3; k <= 3; k++) a += is_solid(6974 + k * 2, 7638);
        printf("arch on Podilska solid samples: %d/7\n", a);
        evac = 0; health = 100; px_ = 6974; py_ = 7638; stop_car(); update_camera(1); state = S_PLAY;
        for (int f = 0; f < 3; f++) game_update(0, 0, 128, 128);
        px_ = 6974; py_ = 7638; draw_frame(); snprintf(nm, sizeof nm, "%s/f_arch.ppm", dir); dump(nm);
    }
    /* пам'ятка: підвезти до першої (каланча), дочекатися таблички */
    if (NLM > 0) {
        int c0 = lm_count;
        px_ = LMS[0].x + 20; py_ = LMS[0].y + 40; stop_car(); unstick(); update_camera(1);
        for (int f = 0; f < 50; f++) { game_update(0, 0, 128, 128); }
        draw_frame(); snprintf(nm, sizeof nm, "%s/f_landmark.ppm", dir); dump(nm);
        printf("landmarks: %d total, found %d -> %d\n", NLM, c0, lm_count);
        int bytes = 0; for (int i = 0; i < NLM; i++) bytes += LMS[i].nfr * ((LMS[i].w * LMS[i].h + 3) & ~3);
        printf("landmark sprites: %d bytes\n", bytes);
    }
    game_update(0, PSP_CTRL_SELECT, 128, 128); draw_frame();
    snprintf(nm, sizeof nm, "%s/f_bigmap.ppm", dir); dump(nm);
    game_update(0, PSP_CTRL_SELECT, 128, 128);
    game_update(0, PSP_CTRL_LTRIGGER, 128, 128);
    int maxl = 0;
    for (int f = 0; f < 400; f++) { game_update(PSP_CTRL_CROSS, 0, 255, 60); draw_frame(); if (loads_frame > maxl) maxl = loads_frame; }
    snprintf(nm, sizeof nm, "%s/f_freecam.ppm", dir); dump(nm);
    printf("max chunk loads per frame in fast free camera: %d\n", maxl);
    game_update(0, PSP_CTRL_LTRIGGER, 128, 128);
    game_update(0, PSP_CTRL_START, 128, 128); draw_frame();
    snprintf(nm, sizeof nm, "%s/f_pause.ppm", dir); dump(nm);
    printf("money=%d deliveries=%d late=%d shots=%d cache=%d\n", money, deliveries, late, shots, NCACHE);
    return 0;
}
#endif

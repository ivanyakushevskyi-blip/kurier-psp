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
static u8 minib[MINI_W * MINI_H];
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
static int tw8(const char *s, int sc) { return ncp(s) * 8 * sc; }
static const u8 *glyph8(int cp) {
    int lo = 0, hi = NGLYPH - 1;
    while (lo <= hi) { int m = (lo + hi) / 2; if (FONT_CP[m] == cp) return FONT_BITS[m]; if (FONT_CP[m] < cp) lo = m + 1; else hi = m - 1; }
    return NULL;
}
static void text8(int x, int y, int sc, u8 col, const char *s) {
    while (*s) {
        const u8 *g = glyph8(utf8next(&s));
        if (g) for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) if (g[r] & (128 >> c)) rect(x + c * sc, y + r * sc, sc, sc, col);
        x += 8 * sc;
    }
}
static void otext8(int x, int y, int sc, u8 col, u8 oc, const char *s) {
    int o = sc;
    for (int dy = -o; dy <= o; dy += o) for (int dx = -o; dx <= o; dx += o) if (dx || dy) text8(x + dx, y + dy, sc, oc, s);
    text8(x, y, sc, col, s);
}
static void otext8_c(int cx, int y, int sc, u8 col, u8 oc, const char *s) { otext8(cx - tw8(s, sc) / 2, y, sc, col, oc, s); }

/* ================= game state ================= */
enum { S_TITLE, S_PLAY, S_PAUSE, S_BIGMAP };
enum { O_NONE, O_PICK, O_LOAD, O_DROP };
static int state = S_TITLE;
static float px_, py_, ang, vel;
static float camx, camy;
static int bump;
static int freecam;                       /* L: огляд міста без машини */
static float fcx, fcy;
static int ostate = O_NONE, owait = 90, otimer, olimit, shop_i = -1, addr_i = -1, pay, load_t;
static int money, deliveries, late;
static char toast[48]; static int toast_t; static u8 toast_c;
static int street_i = -1, street_t;
static char save_path[512] = "save.dat";

static void say(const char *t, u8 c, int life) { strncpy(toast, t, 47); toast[47] = 0; toast_t = life; toast_c = c; }
static float dist(float ax, float ay, float bx, float by) { float dx = ax - bx, dy = ay - by; return sqrtf(dx * dx + dy * dy); }

static void load_save(void) {
    FILE *f = fopen(save_path, "rb");
    if (f) { int a = 0, b = 0; if (fscanf(f, "%d %d", &a, &b) == 2) { money = a; deliveries = b; } fclose(f); }
}
static void write_save(void) {
    FILE *f = fopen(save_path, "wb");
    if (f) { fprintf(f, "%d %d\n", money, deliveries); fclose(f); }
}

static void new_order(void) {
    int best = -1;
    for (int t = 0; t < 80; t++) {
        int i = rand() % NSHOPS;
        float d = dist(px_, py_, SHOPS[i].x, SHOPS[i].y);
        if (d > 160 && d < 1100) { best = i; break; }
        if (best < 0 || d < dist(px_, py_, SHOPS[best].x, SHOPS[best].y)) best = i;
    }
    shop_i = best;
    int ab = -1;
    for (int t = 0; t < 120; t++) {
        int i = rand() % NADDRS;
        float d = dist(SHOPS[shop_i].x, SHOPS[shop_i].y, ADDRS[i].x, ADDRS[i].y);
        if (d > 500 && d < 1850) { ab = i; break; }
        if (ab < 0) ab = i;
    }
    addr_i = ab;
    ostate = O_PICK;
    say("НОВЕ ЗАМОВЛЕННЯ!", C_green_3, 110);
}

static void update_orders(void) {
    if (ostate == O_NONE) { if (--owait <= 0) new_order(); return; }
    if (ostate == O_PICK) {
        if (dist(px_, py_, SHOPS[shop_i].x, SHOPS[shop_i].y) < 24 && fabsf(vel) < 0.35f) { ostate = O_LOAD; load_t = 0; }
        return;
    }
    if (ostate == O_LOAD) {
        if (dist(px_, py_, SHOPS[shop_i].x, SHOPS[shop_i].y) > 34) { ostate = O_PICK; return; }
        if (++load_t >= 60) {
            float dm = dist(SHOPS[shop_i].x, SHOPS[shop_i].y, ADDRS[addr_i].x, ADDRS[addr_i].y) * MPP;
            olimit = (int)((dm * 1.5f / 10.0f + 25.0f) * 60.0f);
            otimer = olimit;
            pay = 35 + (int)(dm * 0.09f);
            ostate = O_DROP;
            say("ЗАБРАВ! ВЕЗИ ДО КЛІЄНТА", C_yellow_3, 100);
        }
        return;
    }
    if (ostate == O_DROP) {
        if (--otimer <= 0) {
            late++; ostate = O_NONE; owait = 150;
            say("ЗАПІЗНИВСЯ. ЗАМОВЛЕННЯ СКАСОВАНО", C_red_3, 150);
            return;
        }
        if (dist(px_, py_, ADDRS[addr_i].x, ADDRS[addr_i].y) < 24 && fabsf(vel) < 0.45f) {
            int bonus = otimer / 60 / 3;
            money += pay + bonus; deliveries++;
            char b[48]; snprintf(b, sizeof b, "+%d ГРН%s", pay + bonus, bonus > 5 ? "  ШВИДКО!" : "");
            say(b, C_green_3, 150);
            ostate = O_NONE; owait = 140;
            write_save();
        }
    }
}

static int hit(float x, float y, float a) {
    float c = cosf(a), s = sinf(a);
    static const float P[][2] = { { 0, 0 }, { 10, 0 }, { -10, 0 }, { 9, 4 }, { 9, -4 }, { -9, 4 }, { -9, -4 } };   /* Golf ~20x9 пікс */
    for (int i = 0; i < 7; i++) {
        float lx = P[i][0], ly = P[i][1];
        if (is_solid((int)(x + lx * c - ly * s), (int)(y + lx * s + ly * c))) return 1;
    }
    return 0;
}

static void update_player(unsigned btn, int lx) {
    float steer = 0;
    if (btn & PSP_CTRL_LEFT) steer = -1;
    if (btn & PSP_CTRL_RIGHT) steer = 1;
    if (abs(lx - 128) > 32) steer = (lx - 128) / 96.0f;
    if (steer > 1) steer = 1; if (steer < -1) steer = -1;
    int gas = (btn & PSP_CTRL_CROSS) != 0;
    int brake = (btn & PSP_CTRL_SQUARE) != 0;
    int off = OFFROAD[map_px((int)px_, (int)py_)];
    float vmax = off ? 0.4f : 0.75f;
    if (gas && vel < vmax) vel += vel < 0 ? 0.04f : 0.017f;
    if (brake) vel -= vel > 0.02f ? 0.045f : 0.012f;
    if (!gas && !brake) { vel *= 0.984f; if (fabsf(vel) < 0.006f) vel = 0; }
    if (vel > vmax) vel -= 0.025f;
    if (vel < -0.34f) vel = -0.34f;
    float grip = fabsf(vel) / 0.45f; if (grip > 1) grip = 1;
    ang += steer * 0.058f * grip * (vel < 0 ? -1 : 1);
    if (ang > PI_F) ang -= 2 * PI_F; if (ang < -PI_F) ang += 2 * PI_F;
    float nx = px_ + cosf(ang) * vel, ny = py_ + sinf(ang) * vel;
    if (!hit(nx, ny, ang)) { px_ = nx; py_ = ny; }
    else if (!hit(nx, py_, ang)) { px_ = nx; vel *= 0.9f; }
    else if (!hit(px_, ny, ang)) { py_ = ny; vel *= 0.9f; }
    else { if (fabsf(vel) > 0.5f) bump = 8; vel *= -0.25f; }
}

static void update_camera(int snap) {
    float tx = px_ + cosf(ang) * vel * 60 - SW / 2, ty = py_ + sinf(ang) * vel * 50 - SH / 2 + 12;
    if (snap) { camx = tx; camy = ty; } else { camx += (tx - camx) * 0.09f; camy += (ty - camy) * 0.09f; }
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
    for (int i = 0; i < NSEGS; i++) {
        float x0 = SEGS[i][0], y0 = SEGS[i][1], x1 = SEGS[i][2], y1 = SEGS[i][3];
        float dx = x1 - x0, dy = y1 - y0, L = dx * dx + dy * dy;
        float qx = freecam ? fcx : px_, qy = freecam ? fcy : py_;
        float t = L > 0 ? ((qx - x0) * dx + (qy - y0) * dy) / L : 0; if (t < 0) t = 0; if (t > 1) t = 1;
        float ex = x0 + dx * t - qx, ey = y0 + dy * t - qy, d = ex * ex + ey * ey;
        if (d < bd) { bd = d; bi = SEGS[i][4]; }
    }
    if (bi >= 0) street_i = bi;
}

/* ================= HUD ================= */
static void panel(int x, int y, int w, int h) {
    rect(x, y, w, h, C_ink_0);
    rect(x + 1, y + 1, w - 2, h - 2, C_panel_1);
    rect(x + 1, y + 1, w - 2, 1, C_ink_2);
}

static void draw_minimap(float tx, float ty, int have_t, u8 tcol) {
    const int MX = 364, MY = 6, MW = 110, MH = 82;
    rect(MX - 2, MY - 2, MW + 4, MH + 4, C_ink_0);
    rect(MX - 1, MY - 1, MW + 2, MH + 2, C_white_1);
    float qx = freecam ? fcx : px_, qy = freecam ? fcy : py_;
    int ox = (int)(qx / MINI_S) - MW / 2, oy = (int)(qy / MINI_S) - MH / 2;
    if (ox < 0) ox = 0; if (oy < 0) oy = 0;
    if (ox > MINI_W - MW) ox = MINI_W - MW; if (oy > MINI_H - MH) oy = MINI_H - MH;
    for (int y = 0; y < MH; y++) memcpy(fb + (MY + y) * SW + MX, minib + (oy + y) * MINI_W + ox, MW);
    if (have_t) {
        float mx = tx / MINI_S - ox, my = ty / MINI_S - oy;
        if (mx < 2) mx = 2; if (my < 2) my = 2; if (mx > MW - 3) mx = MW - 3; if (my > MH - 3) my = MH - 3;
        rect(MX + (int)mx - 2, MY + (int)my - 2, 5, 5, C_ink_0);
        rect(MX + (int)mx - 1, MY + (int)my - 1, 3, 3, tcol);
    }
    int pxm = MX + (int)(qx / MINI_S) - ox, pym = MY + (int)(qy / MINI_S) - oy;
    rect(pxm - 2, pym - 2, 5, 5, C_ink_0);
    rect(pxm - 1, pym - 1, 3, 3, (tick / 8) % 2 ? C_red_3 : C_white_2);
    pset(pxm + (int)roundf(cosf(ang) * 4), pym + (int)roundf(sinf(ang) * 4), C_white_2);
}

static void draw_target(float tx, float ty, const u8 *icon, u8 col, u8 coldark) {
    float sx = tx - (int)camx, sy = ty - (int)camy;
    if (sx > 8 && sx < SW - 8 && sy > 30 && sy < SH - 34) {
        float r = 13 + sinf(tick * 0.15f) * 2.5f;
        ring(sx, sy, r + 1, C_ink_0, 1);
        ring(sx, sy, r, col, 2);
        ring(sx, sy, r - 2, coldark, 1);
        int bob = (int)(sinf(tick * 0.12f) * 3);
        tri(sx - 4, sy - 16 + bob, sx + 4, sy - 16 + bob, sx, sy - 10 + bob, C_ink_0);
        blit(icon, ICON_SZ, ICON_SZ, (int)sx - 8, (int)sy - 32 + bob);
        return;
    }
    float cx = SW / 2, cy = (SH - 30) / 2 + 15, dx = sx - cx, dy = sy - cy;
    float L = sqrtf(dx * dx + dy * dy); if (L < 1) return;
    dx /= L; dy /= L;
    float kx = dx != 0 ? (dx > 0 ? (SW - 22 - cx) / dx : (22 - cx) / dx) : 1e9f;
    float ky = dy != 0 ? (dy > 0 ? (SH - 52 - cy) / dy : (40 - cy) / dy) : 1e9f;
    float k = fminf(kx, ky);
    float ax = cx + dx * k, ay = cy + dy * k;
    float nx = -dy, ny = dx;
    tri(ax + dx * 12, ay + dy * 12, ax - dx * 6 + nx * 10, ay - dy * 6 + ny * 10, ax - dx * 6 - nx * 10, ay - dy * 6 - ny * 10, C_ink_0);
    tri(ax + dx * 9, ay + dy * 9, ax - dx * 4 + nx * 7, ay - dy * 4 + ny * 7, ax - dx * 4 - nx * 7, ay - dy * 4 - ny * 7, col);
    char b[24]; snprintf(b, sizeof b, "%dМ", (int)(dist(px_, py_, tx, ty) * MPP));
    int lx = (int)(ax - dx * 22) - tw8(b, 1) / 2, ly = (int)(ay - dy * 22) - 4;
    otext8(lx, ly, 1, C_white_2, C_ink_0, b);
}

static void draw_hud(void) {
    char b[200];
    /* money */
    panel(6, 6, 124, 30);
    snprintf(b, sizeof b, "%d ГРН", money);
    text8(13, 11, 1, C_yellow_3, b);
    snprintf(b, sizeof b, "ДОСТАВОК: %d", deliveries);
    text8(13, 23, 1, C_white_1, b);
    /* street name */
    if (street_i >= 0) otext8_c(244, 10, 1, C_white_2, C_ink_0, STREETS[street_i]);
    /* order panel */
    panel(0, SH - 30, SW, 30);
    const u8 *ish = assets_bin + ICON_SHOP, *iho = assets_bin + ICON_HOME;
    float kmh = fabsf(vel) * MPP * 60 * 3.6f;
    snprintf(b, sizeof b, "%d КМ/Г", (int)(kmh + 0.5f));
    text8(SW - 8 - tw8(b, 1), SH - 12, 1, C_white_1, b);
    if (ostate == O_PICK || ostate == O_LOAD) {
        blit(ish, ICON_SZ, ICON_SZ, 6, SH - 26);
        snprintf(b, sizeof b, "ЗАБРАТИ: %s", SHOPS[shop_i].name);
        text8(26, SH - 25, 1, C_green_3, b);
        snprintf(b, sizeof b, "%s  %dМ", SHOPS[shop_i].kind, (int)(dist(px_, py_, SHOPS[shop_i].x, SHOPS[shop_i].y) * MPP));
        text8(26, SH - 12, 1, C_white_1, b);
    } else if (ostate == O_DROP) {
        blit(iho, ICON_SZ, ICON_SZ, 6, SH - 26);
        snprintf(b, sizeof b, "ДОСТАВИТИ: %s", ADDRS[addr_i].name);
        text8(26, SH - 25, 1, C_yellow_3, b);
        snprintf(b, sizeof b, "%dМ   ОПЛАТА %d ГРН", (int)(dist(px_, py_, ADDRS[addr_i].x, ADDRS[addr_i].y) * MPP), pay);
        text8(26, SH - 12, 1, C_white_1, b);
        int s = otimer / 60;
        snprintf(b, sizeof b, "%d:%02d", s / 60, s % 60);
        u8 tc = s < 15 ? ((tick / 8) % 2 ? C_red_3 : C_white_2) : C_white_2;
        text8(SW - 8 - tw8(b, 2), SH - 28, 2, tc, b);
    } else {
        text8(26, SH - 20, 1, C_white_1, "ОЧІКУЄМО ЗАМОВЛЕННЯ...");
    }
    if (ostate == O_LOAD) {
        int w = 120, x = SW / 2 - w / 2, y = 150;
        otext8_c(SW / 2, y - 14, 1, C_white_2, C_ink_0, "ЗАБИРАЮ ЗАМОВЛЕННЯ");
        rect(x - 2, y - 2, w + 4, 10, C_ink_0);
        rect(x, y, w * load_t / 60, 6, C_green_2);
    }
    if (toast_t > 0) {
        int sc = tw8(toast, 2) > 440 ? 1 : 2;
        otext8_c(SW / 2, 60, sc, toast_c, C_ink_0, toast);
    }
}

static void draw_world(void) {
    int ox = (int)camx, oy = (int)camy;
    if (bump > 0) { ox += (rand() % 5) - 2; oy += (rand() % 5) - 2; if (ox < 0) ox = 0; if (oy < 0) oy = 0;
        if (ox > MAP_W - SW) ox = MAP_W - SW; if (oy > MAP_H - SH) oy = MAP_H - SH; }
    draw_map(ox, oy);
    if (freecam) {
        int cx = SW / 2, cy = SH / 2;
        rect(cx - 7, cy, 5, 1, C_white_2); rect(cx + 3, cy, 5, 1, C_white_2);
        rect(cx, cy - 7, 1, 5, C_white_2); rect(cx, cy + 3, 1, 5, C_white_2);
        return;
    }
    float sx = px_ - ox, sy = py_ - oy;
    shadow_ell(sx + 2, sy + 2, 11, 6);
    int fi = (int)floorf(ang / (2 * PI_F) * SPR_N + 0.5f);
    fi = ((fi % SPR_N) + SPR_N) % SPR_N;
    blit(assets_bin + SPR_OFF + fi * SPR_SZ * SPR_SZ, SPR_SZ, SPR_SZ, (int)sx - SPR_SZ / 2, (int)sy - SPR_SZ / 2);
}

static void draw_bigmap(void) {
    memset(fb, C_ink_0, sizeof fb);
    int ox = (SW - BIG_W) / 2, oy = (SH - BIG_H) / 2;
    for (int y = 0; y < BIG_H; y++) memcpy(fb + (oy + y) * SW + ox, bigb + y * BIG_W, BIG_W);
    float qx = freecam ? fcx : px_, qy = freecam ? fcy : py_;
    int mx = ox + (int)(qx / BIG_S), my = oy + (int)(qy / BIG_S);
    if (ostate == O_PICK || ostate == O_LOAD) { int tx = ox + SHOPS[shop_i].x / BIG_S, ty = oy + SHOPS[shop_i].y / BIG_S; rect(tx - 2, ty - 2, 5, 5, C_ink_0); rect(tx - 1, ty - 1, 3, 3, C_green_3); }
    if (ostate == O_DROP) { int tx = ox + ADDRS[addr_i].x / BIG_S, ty = oy + ADDRS[addr_i].y / BIG_S; rect(tx - 2, ty - 2, 5, 5, C_ink_0); rect(tx - 1, ty - 1, 3, 3, C_yellow_3); }
    rect(mx - 3, my - 3, 7, 7, C_ink_0);
    rect(mx - 2, my - 2, 5, 5, (tick / 8) % 2 ? C_red_3 : C_white_2);
    text8(6, 6, 1, C_white_2, "КАРТА");
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
        otext8_c(SW / 2, 232, 1, C_white_1, C_ink_0, "X: ГАЗ   КВАДРАТ: ГАЛЬМО / НАЗАД");
        otext8_c(SW / 2, 246, 1, C_white_1, C_ink_0, "СТІК: КЕРМО  L: ОГЛЯД  SELECT: КАРТА");
        otext8_c(SW / 2, 262, 1, C_white_0, C_ink_0, "КАРТА: (C) OPENSTREETMAP CONTRIBUTORS");
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
        draw_minimap(0, 0, 0, 0);
        if (street_i >= 0) otext8_c(244, 10, 1, C_white_2, C_ink_0, STREETS[street_i]);
        panel(0, SH - 18, SW, 18);
        text8(8, SH - 13, 1, C_yellow_3, "ОГЛЯД МІСТА");
        text8(118, SH - 13, 1, C_white_1, "СТІК: РУХ  X: ШВИДКО  L: ДО МАШИНИ");
        return;
    }
    if (ostate == O_PICK || ostate == O_LOAD) draw_target(SHOPS[shop_i].x, SHOPS[shop_i].y, assets_bin + ICON_SHOP, C_green_2, C_green_0);
    else if (ostate == O_DROP) draw_target(ADDRS[addr_i].x, ADDRS[addr_i].y, assets_bin + ICON_HOME, C_yellow_2, C_yellow_0);
    if (ostate == O_PICK || ostate == O_LOAD) draw_minimap(SHOPS[shop_i].x, SHOPS[shop_i].y, 1, C_green_3);
    else if (ostate == O_DROP) draw_minimap(ADDRS[addr_i].x, ADDRS[addr_i].y, 1, C_yellow_3);
    else draw_minimap(0, 0, 0, 0);
    draw_hud();
    if (state == S_PAUSE) {
        darken_rect(0, 0, SW, SH - 30, 2);
        otext8_c(SW / 2, 70, 4, C_white_2, C_ink_0, "ПАУЗА");
        snprintf(b, sizeof b, "ЗАРОБЛЕНО: %d ГРН", money); otext8_c(SW / 2, 120, 1, C_yellow_3, C_ink_0, b);
        snprintf(b, sizeof b, "ДОСТАВОК: %d   ЗАПІЗНЕНЬ: %d", deliveries, late); otext8_c(SW / 2, 136, 1, C_white_2, C_ink_0, b);
        otext8_c(SW / 2, 170, 1, C_white_1, C_ink_0, "СТАРТ: ГРАТИ ДАЛІ");
        otext8_c(SW / 2, 184, 1, C_white_1, C_ink_0, "SELECT: ІНШЕ ЗАМОВЛЕННЯ");
    }
}

static void game_init(void) {
    px_ = SPAWN_X; py_ = SPAWN_Y; ang = 0; vel = 0;
    /* find a free spot near spawn */
    for (int r = 0; r < 200 && hit(px_, py_, ang); r += 2) { px_ = SPAWN_X + r; }
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
        if (pressed & PSP_CTRL_START) state = S_PLAY;
        if (pressed & PSP_CTRL_SELECT) { ostate = O_NONE; owait = 20; state = S_PLAY; }
        return;
    }
    if (pressed & PSP_CTRL_START) { state = S_PAUSE; return; }
    if (pressed & PSP_CTRL_SELECT) { state = S_BIGMAP; return; }
    if (pressed & PSP_CTRL_LTRIGGER) {
        freecam = !freecam;
        if (freecam) { fcx = camx + SW / 2; fcy = camy + SH / 2; vel = 0; }
        else update_camera(1);
    }
    if (freecam) update_freecam(btn, lx, ly);
    else { update_player(btn, lx); update_camera(0); update_orders(); }
    update_street();
    prefetch((int)camx, (int)camy, freecam ? 0 : cosf(ang) * vel, freecam ? 0 : sinf(ang) * vel);
}

static void assets_init(void) {
    cache_init();
    map_open();
    uLongf l = MINI_W * MINI_H; uncompress(minib, &l, assets_bin + MINI_OFF, MINI_ZLEN);
    l = BIG_W * BIG_H; uncompress(bigb, &l, assets_bin + BIG_OFF, BIG_ZLEN);
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
            vel = 0; stuck = 0;
        }
        game_update(btn, 0, lx, 128);
        draw_frame();
        if ((f == 200 || f == 900 || (ostate == O_LOAD && load_t == 30) || (ostate == O_DROP && otimer == olimit - 300) || (toast_t == 140 && strstr(toast, "ГРН"))) && shots < 8) {
            snprintf(nm, sizeof nm, "%s/f_play%d.ppm", dir, shots++); dump(nm);
        }
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

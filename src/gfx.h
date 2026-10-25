/* Графічний шар: на PSP — графічний чип (sceGu), на ПК (-DHOST) — програмна копія тих самих операцій
   для автотесту з дампом кадрів. Кольори — 0xAABBGGRR (як у PALETTE). */
#ifndef GFX_H
#define GFX_H
#include <stdint.h>

void gfx_init(const uint32_t *pal);      /* палітра 256 кольорів; індекс 0 у прозорих шарах — прозорий */
void gfx_begin(void *target);            /* PSP: зсув буфера кадру у VRAM; ПК: uint32_t[512*272] */
/* 8-бітний шар (індекси палітри): w*h текселів з кроком stride, лівий верхній кут на екрані (sx,sy), масштаб scale.
   alpha=1 — індекс 0 прозорий, змішування; linear=1 — білінійне згладжування. stride кратний 16, адреса вирівняна на 16. */
void gfx_layer8(const uint8_t *tex, int stride, int w, int h, float sx, float sy, float scale, int alpha, int linear);
/* 16-бітний шар: fmt 0 — 5650 (непрозорий), 1 — 5551 (біт 15 — непрозорість). stride кратний 8. */
void gfx_layer16(const uint16_t *tex, int stride, int w, int h, float sx, float sy, float scale, int fmt, int alpha, int linear);
/* RGBA-текстура tw*th (степені двійки), центр (cx,cy), поворот ang (рад), півширина/піввисота на екрані hx,hy,
   колір-множник col (0xAABBGGRR): затемнення й прозорість. */
void gfx_quad(const uint32_t *tex, int tw, int th, float cx, float cy, float ang, float hx, float hy, uint32_t col);
void gfx_rect(float x, float y, float w, float h, uint32_t col);
/* трикутники з 16-бітної текстури (fmt як у gfx_layer16), білінійно, зі змішуванням за альфою:
   uvxy — по 4 числа на вершину (u, v у текселях; x, y на екрані), nv кратне 3 */
void gfx_tris_flat(const float *xy, int nv, uint32_t col);   /* одноколірні трикутники (0xAABBGGRR), xy — по 2 числа */
void gfx_tris16(const uint16_t *tex, int stride, int w, int h, int fmt, const float *uvxy, int nv);   /* напівпрозорий прямокутник */
void gfx_end(void);                     /* PSP: кадр відданий чипу, не чекаємо */
void gfx_sync(void);                    /* дочекатися, поки чип домалює попередній кадр */

#endif

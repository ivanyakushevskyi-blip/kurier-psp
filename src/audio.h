/* Звук (01.11.2026): мікшер на 14 голосів, 44,1 кГц стерео; на PSP — окремий потік і апаратний аудіоканал,
   на ПК (-DHOST) — лише мікшування в буфер для автотесту. Звуки — int16 моно (SND_RATE), синтезовані tools/sounds.py. */
#ifndef AUDIO_H
#define AUDIO_H
#include <stdint.h>
enum { V_ENGINE, V_INTAKE, V_SKID, V_HORN, V_CITY, V_NIGHT, V_RAIN, V_TRAF, V_LOOPS, V_ONESHOT = V_LOOPS, V_TOTAL = V_LOOPS + 6 };
void audio_init(const int16_t *const *data, const uint32_t *len, int nsnd, int rate);
void audio_loop(int voice, int snd, float vol, float pitch, float pan);   /* постійний голос: vol 0 — тиша */
void audio_play(int snd, float vol, float pitch, float pan);               /* одноразовий звук */
void audio_mix(int16_t *out, int frames);                                 /* стерео int16 (для ПК-тесту) */
void audio_master(float v);
#endif

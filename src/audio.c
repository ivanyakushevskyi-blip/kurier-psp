#include "audio.h"
#include <math.h>
#include <string.h>

typedef struct { int snd; float pos, pitch, vol, pan, cvol, cpan; int on; } Voice;
static volatile Voice V[V_TOTAL];
static const int16_t *const *SD; static const uint32_t *SL; static int NS, RATE;
static float master = 1.5f;                      /* PSP-динаміки тихі */
static int next_one;

void audio_master(float v) { master = v; }
void audio_loop(int v, int snd, float vol, float pitch, float pan) {
    if (v < 0 || v >= V_LOOPS) return;
    if (V[v].snd != snd) { V[v].snd = snd; V[v].pos = 0; }
    V[v].vol = vol; V[v].pitch = pitch; V[v].pan = pan; V[v].on = 1;
}
void audio_play(int snd, float vol, float pitch, float pan) {
    int v = V_ONESHOT + next_one; next_one = (next_one + 1) % (V_TOTAL - V_ONESHOT);
    V[v].on = 0; V[v].snd = snd; V[v].pos = 0; V[v].pitch = pitch; V[v].vol = vol; V[v].cvol = vol; V[v].pan = pan; V[v].cpan = pan; V[v].on = 1;
}
void audio_mix(int16_t *out, int frames) {
    static float acc[2 * 2048];
    if (frames > 2048) frames = 2048;
    memset(acc, 0, sizeof(float) * 2 * frames);
    float step0 = (float)RATE / 44100.0f;
    for (int v = 0; v < V_TOTAL; v++) {
        if (!V[v].on) continue;
        int s = V[v].snd; if (s < 0 || s >= NS) continue;
        const int16_t *d = SD[s]; float n = (float)SL[s], pos = V[v].pos, st = step0 * V[v].pitch;
        float v0 = V[v].cvol, v1 = V[v].vol, p0 = V[v].cpan, p1 = V[v].pan;   /* плавно за буфер — без клацань */
        if (v0 < 0.0005f && v1 < 0.0005f) { if (v >= V_ONESHOT) V[v].on = 0; V[v].cvol = 0; continue; }
        int loop = v < V_LOOPS;
        for (int i = 0; i < frames; i++) {
            float k = (float)i / frames, vol = (v0 + (v1 - v0) * k) * master, pan = p0 + (p1 - p0) * k;
            int ip = (int)pos; float fr = pos - ip;
            int i2 = ip + 1; if (i2 >= (int)n) i2 = loop ? 0 : ip;
            float smp = (d[ip] * (1 - fr) + d[i2] * fr) * vol;
            acc[2 * i] += smp * (pan < 0 ? 1 : 1 - pan); acc[2 * i + 1] += smp * (pan > 0 ? 1 : 1 + pan);
            pos += st;
            if (pos >= n) { if (loop) pos -= n; else { V[v].on = 0; break; } }
        }
        V[v].pos = pos; V[v].cvol = v1; V[v].cpan = p1;
    }
    for (int i = 0; i < 2 * frames; i++) {                       /* м'яке обмеження гучних ударів, далі — жорстке */
        float x = acc[i], ax = x < 0 ? -x : x;
        if (ax > 22000) { ax = 22000 + (ax - 22000) * 0.25f; x = x < 0 ? -ax : ax; }
        if (x > 32767) x = 32767; if (x < -32768) x = -32768; out[i] = (int16_t)x;
    }
}

#ifndef HOST
#include <pspkernel.h>
#include <pspaudio.h>
static int16_t __attribute__((aligned(64))) abuf[2][1024 * 2];
static int audio_thread(SceSize a, void *p) {
    int ch = sceAudioChReserve(PSP_AUDIO_NEXT_CHANNEL, 1024, PSP_AUDIO_FORMAT_STEREO);
    if (ch < 0) return 0;
    int b = 0;
    while (1) { audio_mix(abuf[b], 1024); sceAudioOutputBlocking(ch, PSP_AUDIO_VOLUME_MAX, abuf[b]); b ^= 1; }
    return 0;
}
#endif

void audio_init(const int16_t *const *data, const uint32_t *len, int nsnd, int rate) {
    SD = data; SL = len; NS = nsnd; RATE = rate;
    for (int v = 0; v < V_TOTAL; v++) { V[v].on = 0; V[v].snd = -1; }
#ifndef HOST
    int th = sceKernelCreateThread("audio", audio_thread, 0x12, 0x4000, 0, NULL);
    if (th >= 0) sceKernelStartThread(th, 0, NULL);
#endif
}

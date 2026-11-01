"""Звуки гри, синтезовані кодом (01.11.2026): без чужих записів і ліцензій. 22050 Гц, моно, int16.
Петлі мають ціле число періодів / плавний шов, щоб грати по колу без клацань. sounds() -> {ім'я: (int16 масив, петля?)}"""
import numpy as np

SR = 22050
rng = np.random.default_rng(7)


def _norm(x, peak=0.9):
    m = np.abs(x).max() or 1
    return (x / m * peak * 32767).astype(np.int16)


def _lp(x, a):                                    # однополюсний фільтр нижніх частот (0 < a < 1, більше — темніше)
    y = np.empty_like(x); acc = 0.0
    for i, v in enumerate(x): acc = acc * a + v * (1 - a); y[i] = acc
    return y


def _bp_noise(n, lo, hi):                         # смуговий шум через FFT
    X = np.fft.rfft(rng.standard_normal(n)); f = np.fft.rfftfreq(n, 1 / SR)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, n)


def _loop_fade(x, k=600):                         # плавний шов петлі: кінець перетікає в початок
    y = x.copy(); w = np.linspace(0, 1, k)
    y[:k] = x[:k] * w + x[-k:] * (1 - w)
    return y[:-k]


def engine():
    """4 циліндри на 2000 об/хв: тактова частота 66,7 Гц; гармоніки з нерівномірністю спалахів + шум впуску.
    Рівно 40 періодів — петля без шва; у грі грається з кроком об/хв / 2000."""
    f0 = 2000 / 60 * 2; n = int(round(40 * SR / f0)); t = np.arange(n) / SR
    x = np.zeros(n)
    # 01.11.2026: басовитіше — сильна половинна гармоніка (нерівномірність циліндрів), слабкі верхні, темний фільтр
    for h, a in ((0.5, 0.8), (1, 1.0), (1.5, 0.25), (2, 0.4), (3, 0.14), (4, 0.05)):
        x += a * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6))
    pulse = (np.sin(2 * np.pi * f0 * t) > 0.6).astype(float)                    # глухий «бух» кожного спалаху
    x += 0.15 * _lp(pulse * rng.standard_normal(n), 0.9)
    x = _lp(np.tanh(x * 0.7), 0.55)
    return _norm(x, 0.8)


def intake():
    """шум впуску/вихлопу під навантаженням (додається з газом)"""
    x = _bp_noise(SR, 70, 600) + 0.15 * _bp_noise(SR, 600, 1500)        # темний гул, без шипіння
    return _norm(_loop_fade(x), 0.7)


def skid():
    """вереск шин: вузька смуга ~1,1–2,4 кГц з «тремтінням»"""
    n = int(SR * 1.2); t = np.arange(n) / SR
    x = _bp_noise(n, 1100, 2400) * (0.7 + 0.3 * np.sin(2 * np.pi * 11 * t))
    x += 0.25 * np.sin(2 * np.pi * (1650 + 60 * np.sin(2 * np.pi * 3 * t)) * t)
    return _norm(_loop_fade(x), 0.75)


def crash():
    """удар: глухий низ + металевий брязкіт + розсип"""
    n = int(SR * 0.7); t = np.arange(n) / SR
    thump = np.sin(2 * np.pi * (70 * np.exp(-t * 6)) * t) * np.exp(-t * 9)
    metal = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) * np.exp(-t * d) for f, d in ((523, 9), (877, 12), (1310, 15), (2190, 20)))
    noise = rng.standard_normal(n) * np.exp(-t * 14)
    return _norm(1.2 * thump + 0.35 * metal + 0.6 * noise, 0.95)


def bump():
    """бордюр: короткий глухий стук"""
    n = int(SR * 0.18); t = np.arange(n) / SR
    return _norm(np.sin(2 * np.pi * 95 * t) * np.exp(-t * 30) + 0.3 * _lp(rng.standard_normal(n), 0.8) * np.exp(-t * 40), 0.9)


def horn():
    """сигнал легковика: два тони (≈420 і 500 Гц), трохи «квакаючий»; петля"""
    n = int(round(SR / 20)) * 10; t = np.arange(n) / SR                      # 10 періодів 20 Гц — шов без клацання
    x = np.zeros(n)
    for f in (420, 500):
        for h, a in ((1, 1), (2, 0.5), (3, 0.35), (4, 0.2), (5, 0.12)):
            x += a * np.sign(np.sin(2 * np.pi * f * h * t)) * 0.3 + a * np.sin(2 * np.pi * f * h * t) * 0.7
    return _norm(np.tanh(x * 0.6), 0.7)


def tick():
    """поворотник / перемикання в меню"""
    n = int(SR * 0.03); t = np.arange(n) / SR
    return _norm(np.sin(2 * np.pi * 2400 * t) * np.exp(-t * 300) + 0.5 * rng.standard_normal(n) * np.exp(-t * 500), 0.6)


def chime(notes, dur=0.11, tail=0.35):
    """короткий дзвіночок (замовлення, доставка)"""
    out = []
    for k, f in enumerate(notes):
        n = int(SR * (dur if k < len(notes) - 1 else tail)); t = np.arange(n) / SR
        out.append((np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t)) * np.exp(-t * 7))
    return _norm(np.concatenate(out), 0.7)


def rain():
    """дощ: рожевий шум + краплі; петля 3 с (для погоди)"""
    n = SR * 3
    x = _bp_noise(n, 300, 7000) * 0.6
    for _ in range(900):
        p = rng.integers(0, n - 400); L = rng.integers(60, 300)
        x[p:p + L] += rng.uniform(0.3, 1.0) * np.exp(-np.arange(L) / (L / 4)) * rng.standard_normal(L)
    return _norm(_loop_fade(x, 1500), 0.6)


def city():
    """фон міста: далекий гул транспорту, вітер; петля 4 с"""
    n = SR * 4; t = np.arange(n) / SR
    x = _bp_noise(n, 40, 400) * (0.8 + 0.2 * np.sin(2 * np.pi * 0.25 * t)) + 0.25 * _bp_noise(n, 400, 2000)
    return _norm(_loop_fade(x, 2000), 0.5)


def night():
    """ніч на околицях: цвіркуни; петля 2 с"""
    n = SR * 2; t = np.arange(n) / SR
    x = np.zeros(n)
    for f, r, ph in ((4300, 15, 0.0), (4700, 13, 0.4), (3900, 17, 0.8)):
        env = (np.sin(2 * np.pi * r * t + ph) > 0.3) * (np.sin(2 * np.pi * 0.7 * t + ph) > -0.2)
        x += np.sin(2 * np.pi * f * t) * _lp(env.astype(float), 0.97)
    return _norm(_loop_fade(x + 0.05 * _bp_noise(n, 200, 1500), 1000), 0.4)


def sounds():
    return {'engine': (engine(), 1), 'intake': (intake(), 1), 'skid': (skid(), 1), 'crash': (crash(), 0), 'bump': (bump(), 0),
            'horn': (horn(), 1), 'tick': (tick(), 0), 'pick': (chime([660, 880]), 0), 'deliver': (chime([523, 659, 784, 1047]), 0),
            'fail': (chime([392, 311], 0.16, 0.4), 0), 'rain': (rain(), 1), 'city': (city(), 1)}       # цвіркуни прибрано (01.11.2026)


if __name__ == '__main__':
    import wave, sys
    for k, (a, lp) in sounds().items():
        with wave.open('build/snd_%s.wav' % k, 'wb') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(a.tobytes())
        print(k, len(a) / SR, 's', 'петля' if lp else '')

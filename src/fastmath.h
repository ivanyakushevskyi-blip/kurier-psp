/* 15.11.2026 (швидкодія): швидкі sinf/cosf для гри — поліном 9-го степеня після зведення до [-pi/2, pi/2],
   похибка < 4e-6. Бібліотечні на PSP помітно повільніші, а гра викликає їх тисячі разів за кадр. */
#ifndef FASTMATH_H
#define FASTMATH_H
#include <math.h>
static inline float fm_sin(float x) {
    const float TWO_PI = 6.28318531f, INV = 0.159154943f, HALF_PI = 1.57079633f, PI_ = 3.14159265f;
    x -= TWO_PI * floorf(x * INV + 0.5f);                 /* [-pi, pi] */
    if (x > HALF_PI) x = PI_ - x; else if (x < -HALF_PI) x = -PI_ - x;
    float x2 = x * x;
    return x * (1.0f + x2 * (-0.166666667f + x2 * (0.00833333333f + x2 * (-0.000198412698f + x2 * 2.75573192e-6f))));
}
static inline float fm_cos(float x) { return fm_sin(x + 1.57079633f); }
#define sinf(x) fm_sin(x)
#define cosf(x) fm_cos(x)
#endif

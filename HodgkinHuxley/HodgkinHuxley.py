import math


class HudgkinHuxley:

    def __init__(self, u_0, m_0, h_0, n_0, e_na, e_k, e_l, g_na, g_k, g_l, I, c):
        self.u = u_0
        self.m = m_0
        self.h = h_0
        self.n = n_0
        self.e_na = e_na
        self.e_k = e_k
        self.e_l = e_l
        self.g_na = g_na
        self.g_k = g_k
        self.g_l = g_l
        self.I = I
        self.c = c

    def _d_u_(self, t):
        return (1 / self.c) * (
                -1 * self.g_na * self.m ** 3 * self.h * (self.u - self.e_na) +
                -1 * self.g_k * self.n ** 4 * (self.u - self.e_k) +
                -1 * self.g_l * (self.u - self.e_l) +
                self.I[t]
        )

    def _d_m_(self):
        return self._alpha_m_() * (1 - self.m) - \
               self._beta_m_() * self.m

    def _d_h_(self):
        return self._alpha_h_() * (1 - self.h) - \
               self._beta_h_() * self.h

    def _d_n_(self):
        return self._alpha_n_() * (1 - self.n) - \
               self._beta_n_() * self.n

    def _alpha_m_(self):
        v = self.u + 40
        return (v / 10) / (1 - math.exp(v / -10))

    def _beta_m_(self):
        v = self.u + 65
        return 4 * math.exp(v / -18)

    def _alpha_h_(self):
        v = self.u + 65
        return 0.07 * math.exp(v / -20)

    def _beta_h_(self):
        v = self.u + 35
        return 1 / (1 + math.exp(v / -10))

    def _alpha_n_(self):
        v = self.u + 55
        return (v / 100) / (1 - math.exp(v / -10))

    def _beta_n_(self):
        v = self.u + 65
        return math.exp(v / 80) / 8


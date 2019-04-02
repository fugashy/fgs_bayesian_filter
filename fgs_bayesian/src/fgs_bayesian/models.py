# -*- coding: utf-8 -*-
import numpy as np


class Shopping(object):
    def __init__(self):
        # 事前確率
        # 買う確率, 買わない確率
        self.prior = np.array([0.2, 0.8])

        # 条件付き確率
        # 買う人が声をかける確率・買わない人が声をかける確率
        # 買う人が声をかけない確率・買わない人が声をかけない確率
        self.conditional = np.array([[0.9, 0.3], [0.1, 0.7]])

    def update(self):
        # 声をかけた世界
        # 声をかけない世界を消す
        self.conditional[1] = 0.

        # 更新
        posterior_raw = self.prior * self.conditional

        # 比例関係を保ったまま正規化条件を回復
        weight = np.max(posterior_raw[:, ::-1] + posterior_raw)
        posterior = posterior_raw / weight

        return posterior

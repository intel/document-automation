# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

from paddleocr import PaddleOCR

print('Downloading paddleocr models....')
pocr = PaddleOCR(use_angle_cls=True, lang='ch')

print('Completed downloading paddleocr models!')

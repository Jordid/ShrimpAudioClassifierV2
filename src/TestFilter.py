# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 19:15:06 2022

@author: frederich
"""

import MainFilters as jdMainFilter
file_name = '21_11_2022_test_21_11_2022 15_13_06 - 15_13_35.wav'
audio_duration_in_seconds = 30
from_in_hertz = 7000
to_in_hertz = 1500
audio_filtered_filename = jdMainFilter.apply_band_pass_filter(file_name, audio_duration_in_seconds, from_in_hertz, to_in_hertz)
print('audio_filtered_filename: ',audio_filtered_filename)

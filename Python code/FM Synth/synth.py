import random
import time
from tqdm import tqdm
import julius
import numpy as np
import torch
from args import cfg
from scipy.io.wavfile import write
from utils import get_unique_labels
import pandas as pd

from synth_utils import *


PI = 3.141592653589793
TWO_PI = 2 * PI

SAMPLE_RATE = 16384
SIGNAL_DURATION_SEC = 1.0

WAVE_TYPE_DIC = {"sine": 0,
                 "square": 1,
                 "sawtooth": 2}

WAVE_TYPE_DIC_INV = {v: k for k, v in WAVE_TYPE_DIC.items()}

FILTER_TYPE_DIC = {"low_pass": 0,
                   "high_pass": 1}
FILTER_TYPE_DIC_INV = {v: k for k, v in FILTER_TYPE_DIC.items()}

# build a list of possible frequencies
SEMITONES_MAX_OFFSET = 24
MIDDLE_C_FREQ = 261.6255653005985
semitones_list = [*range(-SEMITONES_MAX_OFFSET, SEMITONES_MAX_OFFSET + 1)]
OSC_FREQ_LIST = [MIDDLE_C_FREQ * (2 ** (1 / 12)) ** x for x in semitones_list]
for i in range(len(OSC_FREQ_LIST)):
    OSC_FREQ_LIST[i] = round(OSC_FREQ_LIST[i], 4)

OSC_FREQ_LIST = OSC_FREQ_LIST[33:45]
# OSC_FREQ_LIST = OSC_FREQ_LIST1[39:]
OSC_FREQ_DIC = {round(key, 4): value for value, key in enumerate(OSC_FREQ_LIST)}
OSC_FREQ_DIC_INV = {v: k for k, v in OSC_FREQ_DIC.items()}

MAX_AMP = 1
MAX_MOD_INDEX = 0.3
MAX_LFO_FREQ = 16
# MIN_FILTER_FREQ = 20
MIN_FILTER_FREQ = 1000
MAX_FILTER_FREQ = SAMPLE_RATE // 2 # Just for LPF

osc_amp_opts = np.linspace(0., 1., num=2)
osc_freq_opts = OSC_FREQ_LIST
osc_wave_opts = WAVE_TYPE_DIC
osc_mod_index_opts = np.linspace(0., MAX_MOD_INDEX, num=20)
for i in range(len(osc_mod_index_opts)):
    osc_mod_index_opts[i] = round(osc_mod_index_opts[i], 4)
lfo_freq_opts = np.linspace(1., MAX_LFO_FREQ, num=31)
lfo_wave_opts = ['sine', 'square', 'triangle', 'sawtooth']
filter_type_opts = FILTER_TYPE_DIC
filter_freq_opts = np.linspace(MIN_FILTER_FREQ, MAX_FILTER_FREQ, num=16)
for i in range(len(filter_freq_opts)):
    filter_freq_opts[i] = round(filter_freq_opts[i], 4)

am_mod_wave = ['none', 'sine', 'square', 'triangle', 'sawtooth']
am_mod_amount = np.linspace(0., 1., num=5)
# am_mod_amount = np.array([0.2, 0.4, 0.6, 0.8, 1])
am_mod_freq = np.linspace(1., 8., num=8)
for i in range(len(am_mod_freq)):
    am_mod_freq[i] = round(am_mod_freq[i], 4)

random.seed(2022)
num_sounds = 1000
params_dict = {}
params_dict['osc1_amp'] = random.choices(osc_amp_opts*0+1, k=num_sounds)
params_dict['osc1_freq'] = random.choices(osc_freq_opts, k=num_sounds)
params_dict['osc1_wave'] = random.choices(list(osc_wave_opts.keys()), k=num_sounds)
params_dict['osc1_mod_index'] = random.choices(osc_mod_index_opts, k=num_sounds)
params_dict['lfo1_wave'] = random.choices(lfo_wave_opts, k=num_sounds)
params_dict['lfo1_freq'] = random.choices(lfo_freq_opts, k=num_sounds)

params_dict['am_mod_wave'] = random.choices(am_mod_wave, k=num_sounds)
params_dict['am_mod_freq'] = random.choices(am_mod_freq, k=num_sounds)
params_dict['am_mod_amount'] = random.choices(am_mod_amount, k=num_sounds)

params_dict['filter_type'] = random.choices([list(filter_type_opts.keys())[0]], k=num_sounds)
params_dict['filter_freq'] = random.choices(filter_freq_opts, k=num_sounds)

zero_mod_index = (np.array(params_dict['osc1_mod_index']) == 0)
params_dict['lfo1_freq'] = np.array(params_dict['lfo1_freq'])
params_dict['lfo1_freq'][zero_mod_index] = 1
params_dict['lfo1_freq'] = list(params_dict['lfo1_freq'])

sine_carrier_index = (np.array(params_dict['osc1_wave']) == 'sine')
params_dict['filter_freq'] = np.array(params_dict['filter_freq'])
params_dict['filter_freq'][sine_carrier_index] = SAMPLE_RATE // 2
params_dict['filter_freq'] = list(params_dict['filter_freq'])

params_dict['lfo1_wave'] = np.array(params_dict['lfo1_wave'])
params_dict['lfo1_wave'][zero_mod_index] = 'sine'
# params_dict['lfo1_wave'][:] = 'sine'
params_dict['lfo1_wave'] = list(params_dict['lfo1_wave'])

none_am_index = (np.array(params_dict['am_mod_wave']) == 'none')
params_dict['am_mod_freq'] = np.array(params_dict['am_mod_freq'])
params_dict['am_mod_freq'][none_am_index] = 1
params_dict['am_mod_freq'] = list(params_dict['am_mod_freq'])
params_dict['am_mod_amount'] = np.array(params_dict['am_mod_amount'])
params_dict['am_mod_amount'][none_am_index] = 0
params_dict['am_mod_amount'] = list(params_dict['am_mod_amount'])

params_dict = remove_duplicates(params_dict)

num_sounds = len(params_dict['osc1_amp'])

params_dict['wav_id'] = list(range(num_sounds))
print('Writing CSV...')

dic2save = {'wav_id': params_dict['wav_id'],
            'osc1_wave': params_dict['osc1_wave'],
            'osc1_freq': params_dict['osc1_freq'],
            'osc1_mod_index': params_dict['osc1_mod_index'],
            'lfo1_freq': params_dict['lfo1_freq'],
            'lfo1_wave': params_dict['lfo1_wave'],
            'am_mod_wave': params_dict['am_mod_wave'],
            'am_mod_freq': params_dict['am_mod_freq'],
            'am_mod_amount': params_dict['am_mod_amount'],
            'filter_freq': params_dict['filter_freq'],
            }

subset = list(dic2save.keys())
fields2remove = ['wav_id', 'am_mod_wave', 'am_mod_freq', 'am_mod_amount']
for f in fields2remove:
    subset.remove(f)
df = pd.DataFrame.from_dict(dic2save).drop_duplicates(subset=subset)
if True:
    df.to_csv(r'data/for_test/Data_custom_synth.csv', index=False)

dic_unique = get_unique_labels(r'data/for_test/Data_custom_synth.csv')


print('Generating sounds...')
for i in tqdm(range(num_sounds)):
    file_name = params_dict['wav_id'][i]
    single_wav_dic = {}
    for p in params_dict:
        single_wav_dic[p] = params_dict[p][i]
    my_synth = SynthBasicFlow(file_name='unnamed_sound', parameters_dict=single_wav_dic, num_sounds=1)
    wavs = my_synth.my_generate_signal(1)
    s = np.squeeze(wavs.detach().cpu().numpy())
    write(r'data/for_test/Data_custom_synth/%d.wav' % file_name, SAMPLE_RATE, s)
    # Noy_Synth_Wavs


print('DONE')

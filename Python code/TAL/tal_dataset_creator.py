import dawdreamer as daw
import numpy as np
from scipy.io import wavfile
import pandas as pd
import os
from tqdm import tqdm

NUM_ACTIVE_OSC = [1, 2]

# random params
OSC_VOLUME_LIST = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
OSC_TUNE_LIST = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
OSC1WAVEFORM_LIST = [0, 0.5]
OSC2WAVEFORM_LIST = [0, 0.25, 0.5, 0.75]
LFO_WAVEFORM_LIST = [0, 0.2, 0.4, 0.6]
LFO1DESTINATION_LIST = [0, 0.285, 0.43, 1]
LFO2DESTINATION_LIST = [0, 0.285, 0.43, 0.715]
LFO_AMOUNT_LIST = [0.52, 0.552, 0.584, 0.616, 0.648, 0.68, 0.712, 0.744, 0.776, 0.808, 0.84,
                   0.872, 0.904, 0.936, 0.968, 1.0]
LFO_RATE_LIST = [0.06, 0.096, 0.132, 0.168, 0.204, 0.24, 0.276, 0.312, 0.348,
                 0.384, 0.42, 0.456, 0.492, 0.528, 0.564, 0.6]
FILTER_CUTOFF_LIST = [0.3, 0.346666667, 0.393333333, 0.44, 0.486666667, 0.533333333, 0.58, 0.626666667, 0.673333333,
                      0.72,
                      0.766666667, 0.813333333, 0.86, 0.906666667, 0.953333333, 1]
FILTER_RESONANCE_LIST = [0, 0.066666667, 0.133333333, 0.2, 0.266666667, 0.333333333, 0.4, 0.466666667, 0.533333333, 0.6,
                         0.666666667, 0.733333333, 0.8, 0.866666667, 0.933333333, 1]
AMP_ATTACK_LIST = [0, 0.07, 0.14, 0.21, 0.28, 0.35, 0.42, 0.5]
AMP_RELEASE_LIST = [0.2, 0.28, 0.37, 0.45, 0.54, 0.62, 0.71, 0.8]


class Config:
    dataset_size = 400000
    sample_rate = 16384
    buffer_size = 128  # Parameters will undergo automation at this block size. It can be as small as 1 sample.
    # synth_plugin = r"libTAL-NoiseMaker.so"
    # synth_plugin = r"Dexed.so"
    synth_plugin = r'/home/PycharmProjects/commercial_synth_dataset/libTAL-NoiseMaker.so'
    synth_preset = r"/home/PycharmProjects/commercial_synth_dataset/presets/80sBrass_-_01.fxp"
    duration = 1  # How many seconds we want to render.
    midi_note = 70  # 70 is middle A (A4 - 440Hz)
    velocity = 127
    start_sec = 0
    note_duration = 0.9
    disable_non_deterministic_parameters = True
    use_preset = False
    debug_sound_identity = False
    disable_effects = True
    fix_non_timbral_parameters = True
    mono = True
    csv_filename = '/home/PycharmProjects/Tal/full_parameters_osc_lfo_am_debug.csv'
    full_csv_filename = '/home/PycharmProjects/Tal/full_parameters_osc_lfo_am_debug.csv'
    save_folder = '/home/PycharmProjects/Tal/sounds_debug/'
    synth_version = 5  # Tal-NoiseMaker may be old version (4) or new version (5)


def print_synth_params(synth):
    param_list = synth.get_plugin_parameters_description()
    print("--------SYNTH PARAMETERS--------")
    for i in range(synth.get_plugin_parameter_size()):
        print('{:<5} {:<25s} {:<10}'.format(param_list[i]['index'], param_list[i]['name'], synth.get_parameter(i)))
        # print(f"{param_list[i]['name']} \t\t{param_list[i]['text']}")
    print("-------------END--------------\n")


def randomize_parameters(synth):
    for i in range(synth.get_plugin_parameter_size()):
        synth.set_parameter(i, np.random.rand())


def reset_all_params(synth):
    for i in range(synth.get_plugin_parameter_size()):
        synth.set_parameter(i, 0)


def disable_non_deterministic_params(synth, include_lfo: bool = False):
    synth.set_parameter(67, 0)  # Detune
    synth.set_parameter(60, 0)  # Reverb
    synth.set_parameter(65, 0)  # Osc Bitcrusher
    if include_lfo:
        synth.set_parameter(32, 0)  # Lfo 1 Destination
        synth.set_parameter(33, 0)  # Lfo 2 Destination


def fix_non_timbral_params(synth):
    synth.set_parameter(0, 1)  # No Named Param ("-")
    synth.set_parameter(1, 1)  # Master Volume
    synth.set_parameter(5, 0)  # Filter Keyfollow
    synth.set_parameter(25, 0)  # Osc Sync
    synth.set_parameter(45, 0)  # Lfo 1 Sync
    synth.set_parameter(46, 0)  # Lfo 1 Keytrigger
    synth.set_parameter(47, 0)  # Lfo 2 Sync
    synth.set_parameter(48, 0)  # Lfo 2 Keytrigger
    synth.set_parameter(49, 0)  # Portamento Amount
    synth.set_parameter(50, 0)  # Portamento Mode
    synth.set_parameter(51, 0)  # Voices
    synth.set_parameter(52, 0)  # Velocity Volume
    synth.set_parameter(53, 0)  # Velocity Contour
    synth.set_parameter(54, 0)  # Velocity Filter
    synth.set_parameter(69, 0)  # Panic
    synth.set_parameter(70, 0)  # MIDI LEARN
    synth.set_parameter(86, 0)  # MIDI Clear
    synth.set_parameter(87, 0)  # MIDI Lock
    synth.set_parameter(55, 0)  # Pitchwheel Cutoff
    synth.set_parameter(56, 0)  # Pitchwheel Pitch


def set_effects_off(synth):
    for i in range(60, 65):
        synth.set_parameter(i, 0)  # Reverb Parameters
    for i in range(78, 86):
        synth.set_parameter(i, 0)  # Delay Parameters
    synth.set_parameter(65, 0)  # Osc Bitcrusher
    synth.set_parameter(57, 0)  # Ringmodulation
    synth.set_parameter(58, 0)  # Chorus 1 Enable
    synth.set_parameter(59, 0)  # Chorus 2 Enable
    synth.set_parameter(77, 0)  # Filter Drive
    synth.set_parameter(66, 0)  # Master High Pass
    synth.set_parameter(68, 0)  # Vintage Noise


def init_dataset_dict(synth, dataset_dict):
    dataset_dict['wav_id'] = []
    for i in range(synth.get_plugin_parameter_size()):
        param_name = synth.get_parameter_name(i)
        dataset_dict[f"{i}.{param_name}"] = []


def insert_synth_params_to_dataset(synth, dataset_dict):
    for i in range(synth.get_plugin_parameter_size()):
        param_name = synth.get_parameter_name(i)
        param_value = synth.get_parameter(i)
        dataset_dict[f"{i}.{param_name}"].append(param_value)


def check_sound_identity(sounds: dict):
    for i in range(len(sounds) - 1):
        print(np.linalg.norm(sounds[i] - sounds[i + 1]))


def randomize_params(synth):
    synth.set_parameter(24, np.random.choice(OSC2WAVEFORM_LIST))  # Osc 2 Waveform
    synth.set_parameter(26, np.random.choice(LFO_WAVEFORM_LIST))  # Lfo 1 Waveform
    synth.set_parameter(32, np.random.choice(LFO1DESTINATION_LIST, p=[0.1, 0.9]))  # Lfo 1 Destination
    synth.set_parameter(30, np.random.choice(LFO_AMOUNT_LIST))  # Lfo 1 Amount
    synth.set_parameter(28, np.random.choice(LFO_RATE_LIST))  # Lfo 1 Rate
    synth.set_parameter(3, np.random.choice(FILTER_CUTOFF_LIST))  # Filter Cutoff
    synth.set_parameter(4, np.random.choice(FILTER_RESONANCE_LIST))  # Filter Resonance


def randomize_params_v1(synth):
    synth.set_parameter(3, np.random.choice(FILTER_CUTOFF_LIST))  # Filter Cutoff
    synth.set_parameter(4, np.random.choice(FILTER_RESONANCE_LIST))  # Filter Resonance
    synth.set_parameter(7, np.random.choice(AMP_ATTACK_LIST))  # Amp Attack
    synth.set_parameter(10, np.random.choice(AMP_RELEASE_LIST))  # Amp release

    num_osc = np.random.choice(NUM_ACTIVE_OSC)
    if num_osc == 1:
        synth.set_parameter(15, 0)  # Osc1 Volume
        synth.set_parameter(23, 0)  # Osc 1 Waveform
        synth.set_parameter(16, np.random.choice(OSC_VOLUME_LIST))  # Osc2 Volume
    elif num_osc == 2:
        synth.set_parameter(15, np.random.choice(OSC_VOLUME_LIST))  # Osc1 Volume
        synth.set_parameter(23, np.random.choice(OSC1WAVEFORM_LIST))  # Osc 1 Waveform
        synth.set_parameter(16, np.random.choice(OSC_VOLUME_LIST))  # Osc2 Volume

    synth.set_parameter(24, np.random.choice(OSC2WAVEFORM_LIST))  # Osc 2 Waveform

    synth.set_parameter(20, np.random.choice(OSC_TUNE_LIST))  # Osc 2 Tune
    synth.set_parameter(22, np.random.choice(OSC_TUNE_LIST))  # Osc 2 Fine Tune

    # LFO params
    lfo1rand = np.random.random()
    lfo2rand = np.random.random()
    if lfo1rand < 0.15:
        synth.set_parameter(26, 0)  # Lfo 1 Waveform
        synth.set_parameter(28, 0)  # Lfo 1 Rate
        synth.set_parameter(30, 0.5)  # Lfo 1 Amount
    else:
        synth.set_parameter(26, np.random.choice(LFO_WAVEFORM_LIST))  # Lfo 1 Waveform
        synth.set_parameter(28, np.random.choice(LFO_RATE_LIST))  # Lfo 1 Rate
        synth.set_parameter(30, np.random.choice(LFO_AMOUNT_LIST))  # Lfo 1 Amount

    if lfo2rand > 0.15:
        synth.set_parameter(27, 0)  # Lfo 2 Waveform
        synth.set_parameter(29, 0)  # Lfo 2 Rate
        synth.set_parameter(31, 0.5)  # Lfo 2 Amount
    else:
        synth.set_parameter(27, np.random.choice(LFO_WAVEFORM_LIST))  # Lfo 2 Waveform
        synth.set_parameter(29, np.random.choice(LFO_RATE_LIST))  # Lfo 2 Rate
        synth.set_parameter(31, np.random.choice(LFO_AMOUNT_LIST))  # Lfo 2 Amount


def my_randomize_params_v1(N):
    np.random.seed(2022)
    dic = {}
    dic['wav_id'] = np.arange(0, N)
    dic['3.Filter Cutoff'] = np.random.choice(FILTER_CUTOFF_LIST, size=N)
    dic['4.Filter Resonance'] = np.random.choice(FILTER_RESONANCE_LIST, size=N)
    dic['7.Amp Attack'] = np.random.choice(AMP_ATTACK_LIST, size=N)
    dic['10.Amp Release'] = np.random.choice(AMP_RELEASE_LIST, size=N)

    num_osc = np.random.choice(NUM_ACTIVE_OSC, size=N)

    osc1volume = np.random.choice(OSC_VOLUME_LIST, size=N)
    osc1volume[num_osc == 1] = 0
    dic['15.Osc1 Volume'] = osc1volume

    dic['16.Osc2 Volume'] = np.random.choice(OSC_VOLUME_LIST, size=N)

    osc1waveform = np.random.choice(OSC1WAVEFORM_LIST, size=N)
    osc1waveform[num_osc == 1] = 0
    dic['23.Osc 1 Waveform'] = osc1waveform
    # dic['23.Osc 1 Waveform'] = np.random.choice(OSC1WAVEFORM_LIST, size=N)

    dic['24.Osc 2 Waveform'] = np.random.choice(OSC2WAVEFORM_LIST, size=N)

    dic['20.Osc 2 Tune'] = np.random.choice(OSC_TUNE_LIST, size=N)
    dic['22.Osc 2 Fine Tune'] = np.random.choice(OSC_TUNE_LIST, size=N)

    lfo1rand = np.random.random(size=N)

    dic['26.Lfo 1 Waveform'] = np.random.choice(LFO_WAVEFORM_LIST, size=N)
    dic['28.Lfo 1 Rate'] = np.random.choice(LFO_RATE_LIST, size=N)
    dic['30.Lfo 1 Amount'] = np.random.choice(LFO_AMOUNT_LIST, size=N)

    dic['26.Lfo 1 Waveform'][lfo1rand < 0.15] = 0
    dic['28.Lfo 1 Rate'][lfo1rand < 0.15] = 0
    dic['30.Lfo 1 Amount'][lfo1rand < 0.15] = 0.5

    lfo2rand = np.random.random(size=N)
    dic['27.Lfo 2 Waveform'] = np.random.choice(LFO_WAVEFORM_LIST, size=N)
    dic['29.Lfo 2 Rate'] = np.random.choice(LFO_RATE_LIST, size=N)
    dic['31.Lfo 2 Amount'] = np.random.choice(LFO_AMOUNT_LIST, size=N)

    dic['27.Lfo 2 Waveform'][lfo2rand > 0.15] = 0
    dic['29.Lfo 2 Rate'][lfo2rand > 0.15] = 0
    dic['31.Lfo 2 Amount'][lfo2rand > 0.15] = 0.5

    return dic


def my_randomize_params_debug(N, opt='osc_only'):  # 'osc_only' / 'osc_with_lfo'
    np.random.seed(2022)
    dic = {}
    dic['wav_id'] = np.arange(0, N)

    if opt == 'osc_only':
        dic['3.Filter Cutoff'] = np.random.choice([1.], size=N)
        dic['4.Filter Resonance'] = np.random.choice([0.], size=N)
        dic['7.Amp Attack'] = np.random.choice([0.], size=N)
        dic['10.Amp Release'] = np.random.choice([0.], size=N)

        num_osc = np.random.choice([1], size=N)

        osc1volume = np.random.choice([0], size=N)
        # osc1volume[num_osc == 1] = 0
        dic['15.Osc1 Volume'] = osc1volume

        dic['16.Osc2 Volume'] = np.random.choice([1.], size=N)

        osc1waveform = np.random.choice([0.], size=N)
        # osc1waveform[num_osc == 1] = 0
        dic['23.Osc 1 Waveform'] = osc1waveform

        dic['24.Osc 2 Waveform'] = np.random.choice(OSC2WAVEFORM_LIST, size=N)

        dic['20.Osc 2 Tune'] = np.random.choice(OSC_TUNE_LIST, size=N)
        dic['22.Osc 2 Fine Tune'] = np.random.choice([0.5], size=N)

        lfo1rand = np.random.random(size=N)

        dic['26.Lfo 1 Waveform'] = np.random.choice([0.], size=N)
        dic['28.Lfo 1 Rate'] = np.random.choice([0.], size=N)
        dic['30.Lfo 1 Amount'] = np.random.choice([0.5], size=N)

        # dic['26.Lfo 1 Waveform'][lfo1rand < 0.15] = 0
        # dic['28.Lfo 1 Rate'][lfo1rand < 0.15] = 0
        # dic['30.Lfo 1 Amount'][lfo1rand < 0.15] = 0.5

        lfo2rand = np.random.random(size=N)
        dic['27.Lfo 2 Waveform'] = np.random.choice([0.], size=N)
        dic['29.Lfo 2 Rate'] = np.random.choice([0.], size=N)
        dic['31.Lfo 2 Amount'] = np.random.choice([0.5], size=N)

        # dic['27.Lfo 2 Waveform'][lfo2rand > 0.15] = 0
        # dic['29.Lfo 2 Rate'][lfo2rand > 0.15] = 0
        # dic['31.Lfo 2 Amount'][lfo2rand > 0.15] = 0.5

        variable_params = ['24.Osc 2 Waveform', '20.Osc 2 Tune']

    if opt == 'osc_with_lfo':
        dic['3.Filter Cutoff'] = np.random.choice([1.], size=N)
        dic['4.Filter Resonance'] = np.random.choice([0.], size=N)
        dic['7.Amp Attack'] = np.random.choice([0.], size=N)
        dic['10.Amp Release'] = np.random.choice([0.], size=N)

        num_osc = np.random.choice([1.], size=N)

        osc1volume = np.random.choice([0.], size=N)
        # osc1volume[num_osc == 1] = 0
        dic['15.Osc1 Volume'] = osc1volume

        dic['16.Osc2 Volume'] = np.random.choice([1.], size=N)

        osc1waveform = np.random.choice([0.], size=N)
        # osc1waveform[num_osc == 1] = 0
        dic['23.Osc 1 Waveform'] = osc1waveform
        # dic['23.Osc 1 Waveform'] = np.random.choice(OSC1WAVEFORM_LIST, size=N)

        dic['24.Osc 2 Waveform'] = np.random.choice(OSC2WAVEFORM_LIST, size=N)

        dic['20.Osc 2 Tune'] = np.random.choice(OSC_TUNE_LIST, size=N)
        dic['22.Osc 2 Fine Tune'] = np.random.choice([0.5], size=N)

        lfo1rand = np.random.random(size=N)

        dic['26.Lfo 1 Waveform'] = np.random.choice(LFO_WAVEFORM_LIST, size=N)
        dic['28.Lfo 1 Rate'] = np.random.choice(LFO_RATE_LIST, size=N)
        dic['30.Lfo 1 Amount'] = np.random.choice(LFO_AMOUNT_LIST, size=N)

        dic['26.Lfo 1 Waveform'][lfo1rand < 0.15] = 0
        dic['28.Lfo 1 Rate'][lfo1rand < 0.15] = 0
        dic['30.Lfo 1 Amount'][lfo1rand < 0.15] = 0.5

        # lfo2rand = np.random.random(size=N)
        dic['27.Lfo 2 Waveform'] = np.random.choice([0.], size=N)
        dic['29.Lfo 2 Rate'] = np.random.choice([0.], size=N)
        dic['31.Lfo 2 Amount'] = np.random.choice([0.5], size=N)

        # dic['27.Lfo 2 Waveform'][lfo2rand > 0.15] = 0
        # dic['29.Lfo 2 Rate'][lfo2rand > 0.15] = 0
        # dic['31.Lfo 2 Amount'][lfo2rand > 0.15] = 0.5

        variable_params = ['24.Osc 2 Waveform', '20.Osc 2 Tune', '26.Lfo 1 Waveform', '28.Lfo 1 Rate',
                           '30.Lfo 1 Amount']

    if opt == 'osc_lfo_am':
        dic['3.Filter Cutoff'] = np.random.choice(FILTER_CUTOFF_LIST, size=N)
        dic['4.Filter Resonance'] = np.random.choice([0.], size=N)
        dic['7.Amp Attack'] = np.random.choice([0.], size=N)
        dic['10.Amp Release'] = np.random.choice([0.], size=N)

        num_osc = np.random.choice([1.], size=N)

        osc1volume = np.random.choice([0.], size=N)
        # osc1volume[num_osc == 1] = 0
        dic['15.Osc1 Volume'] = osc1volume

        dic['16.Osc2 Volume'] = np.random.choice([1.], size=N)

        osc1waveform = np.random.choice([0.], size=N)
        # osc1waveform[num_osc == 1] = 0
        dic['23.Osc 1 Waveform'] = osc1waveform
        # dic['23.Osc 1 Waveform'] = np.random.choice(OSC1WAVEFORM_LIST, size=N)

        dic['24.Osc 2 Waveform'] = np.random.choice(OSC2WAVEFORM_LIST, size=N)

        dic['20.Osc 2 Tune'] = np.random.choice(OSC_TUNE_LIST, size=N)
        dic['22.Osc 2 Fine Tune'] = np.random.choice([0.5], size=N)

        lfo1rand = np.random.random(size=N)

        dic['26.Lfo 1 Waveform'] = np.random.choice(LFO_WAVEFORM_LIST, size=N)
        dic['28.Lfo 1 Rate'] = np.random.choice(LFO_RATE_LIST, size=N)
        dic['30.Lfo 1 Amount'] = np.random.choice(LFO_AMOUNT_LIST, size=N)

        dic['26.Lfo 1 Waveform'][lfo1rand < 0.15] = 0
        dic['28.Lfo 1 Rate'][lfo1rand < 0.15] = 0
        dic['30.Lfo 1 Amount'][lfo1rand < 0.15] = 0.5

        lfo2rand = np.random.random(size=N)
        dic['27.Lfo 2 Waveform'] = np.random.choice(LFO_WAVEFORM_LIST, size=N)
        dic['29.Lfo 2 Rate'] = np.random.choice(LFO_RATE_LIST, size=N)
        dic['31.Lfo 2 Amount'] = np.random.choice(LFO_AMOUNT_LIST, size=N)

        dic['27.Lfo 2 Waveform'][lfo2rand > 0.15] = 0
        dic['29.Lfo 2 Rate'][lfo2rand > 0.15] = 0
        dic['31.Lfo 2 Amount'][lfo2rand > 0.15] = 0.5

        variable_params = ['3.Filter Cutoff', '24.Osc 2 Waveform', '20.Osc 2 Tune', '26.Lfo 1 Waveform',
                           '28.Lfo 1 Rate', '30.Lfo 1 Amount', '27.Lfo 2 Waveform', '29.Lfo 2 Rate', '31.Lfo 2 Amount']

    return dic, variable_params


def set_params(synth,
                      osc2waveform,
                      lfo1waveform,
                      lfo1destination,
                      lfo1amount,
                      lfo1rate,
                      filter_cutoff,
                      filter_resonance):
    synth.set_parameter(24, osc2waveform)
    synth.set_parameter(26, lfo1waveform)
    synth.set_parameter(32, lfo1destination)
    synth.set_parameter(30, lfo1amount)
    synth.set_parameter(28, lfo1rate)
    synth.set_parameter(3, filter_cutoff)
    synth.set_parameter(4, filter_resonance)


def set_params0(synth, params):
    param_idx = 0
    for param, value in params.items():
        synth.set_parameter(param_idx, value)
        param_idx += 1


def set_params(synth, params):
    # param_idx = 0
    for param, value in params.items():
        param_idx = int(param.partition('.')[0])
        synth.set_parameter(param_idx, value)
        # param_idx += 1


def set_non_randomized_params(synth):
    synth.set_parameter(0, 0)  # -
    synth.set_parameter(1, 1)  # Master Volume
    synth.set_parameter(6, 0.5)  # Filter Contour
    synth.set_parameter(9, 1)  # Filter Sustain
    synth.set_parameter(13, 1)  # Amp Sustain
    synth.set_parameter(15, 0)  # Osc 1 Volume
    synth.set_parameter(16, 1)  # Osc 2 Volume
    synth.set_parameter(18, 0.5)  # Osc Mastertune
    synth.set_parameter(19, 0.5)  # Osc 1 Tune
    synth.set_parameter(20, 0.5)  # Osc 2 Tune
    synth.set_parameter(21, 0.5)  # Osc 1 Fine Tune
    synth.set_parameter(22, 0.5)  # Osc 2 Fine Tune
    synth.set_parameter(40, 0.5)  # Transpose
    synth.set_parameter(43, 0.5)  # Free Ad Amount
    synth.set_parameter(46, 1)  # Lfo 1 Keytrigger
    synth.set_parameter(48, 1)  # Lfo 1 Keytrigger
    synth.set_parameter(65, 1)  # Osc Bitcrusher


def set_non_randomized_params(synth):
    synth.set_parameter(0, 0)  # -
    synth.set_parameter(1, 1)  # Master Volume
    synth.set_parameter(6, 0.5)  # Filter Contour
    synth.set_parameter(9, 1)  # Filter Sustain
    synth.set_parameter(13, 1)  # Amp Sustain
    synth.set_parameter(18, 0.5)  # Osc Mastertune
    synth.set_parameter(19, 0.5)  # Osc 1 Tune
    synth.set_parameter(21, 0.5)  # Osc 1 Fine Tune
    synth.set_parameter(32, 1)  # Lfo 1 Destination - to Osc1&2
    synth.set_parameter(33, 0.43)  # Lfo 2 Destination - to Volume
    synth.set_parameter(40, 0.5)  # Transpose
    synth.set_parameter(43, 0.5)  # Free Ad Amount
    synth.set_parameter(46, 1)  # Lfo 1 Keytrigger
    synth.set_parameter(48, 1)  # Lfo 1 Keytrigger
    synth.set_parameter(65, 1)  # Osc Bitcrusher


def format_csv(data_frame):
    # "-1" stands for the prepended column "wav_id"
    params_indices = [-1, 24, 26, 32, 30, 28, 3, 4]
    return data_frame.iloc[:, list(np.asarray(params_indices) + 1)]


def create_csv(cfg, synth_plugin_path, dataset_dict):
    print(f"\nGenerating Dataset Using {cfg.synth_plugin}\n")
    for i in range(cfg.dataset_size):
        engine = daw.RenderEngine(cfg.sample_rate, cfg.buffer_size)
        engine.set_bpm(120.)  # default is 120.
        synth = engine.make_plugin_processor("my_synth", synth_plugin_path)
        assert synth.get_name() == "my_synth"
        graph = [
            (synth, [])
        ]
        engine.load_graph(graph)

        np.random.seed(i)
        reset_all_params(synth)
        randomize_params(synth)
        set_non_randomized_params(synth)
        # print_synth_params(synth)
        print(f"randomized parameters for sound {i + 1}/{cfg.dataset_size}")
        dataset_dict['wav_id'].append(i)
        insert_synth_params_to_dataset(synth, dataset_dict)

    data_frame = pd.DataFrame(dataset_dict)
    data_frame = format_csv(data_frame)
    data_frame.to_csv(cfg.csv_filename, index=False)
    data_frame.to_csv(cfg.full_csv_filename, index=False)


def create_csv(cfg, synth_plugin_path, dataset_dict):
    print(f"\nGenerating Dataset Using {cfg.synth_plugin}\n")
    for i in range(cfg.dataset_size):
        engine = daw.RenderEngine(cfg.sample_rate, cfg.buffer_size)
        engine.set_bpm(120.)  # default is 120.
        synth = engine.make_plugin_processor("my_synth", synth_plugin_path)
        assert synth.get_name() == "my_synth"
        graph = [
            (synth, [])
        ]
        engine.load_graph(graph)

        np.random.seed(i)
        reset_all_params(synth)
        randomize_params_v1(synth)
        set_non_randomized_params(synth)
        # print_synth_params(synth)
        print(f"randomized parameters for sound {i + 1}/{cfg.dataset_size}")
        dataset_dict['wav_id'].append(i)
        insert_synth_params_to_dataset(synth, dataset_dict)

    data_frame = pd.DataFrame(dataset_dict)
    data_frame.to_csv(cfg.full_csv_filename, index=False)


def my_create_csv(cfg, debug):
    print(f"\nGenerating Dataset Using {cfg.synth_plugin}\n")
    if debug == 'no':
        dic = my_randomize_params_v1(cfg.dataset_size)
    else:
        dic, variable_params = my_randomize_params_debug(cfg.dataset_size, opt=debug)

    subset = variable_params.copy()
    for f in subset.copy():
        index_param = int(f.partition('.')[0])
        if index_param == 27 or index_param == 29 or index_param == 31:
            print('YES')
            subset.remove(f)
    print(*subset)
    data_frame = pd.DataFrame(dic).drop_duplicates(subset=subset)
    data_frame.to_csv(os.path.join(cfg.save_folder, cfg.full_csv_filename), index=False)

    if not (debug == 'no'):
        for p in dic:
            if (not (p in variable_params)) and (p != 'wav_id'):
                data_frame.pop(p)
        data_frame.to_csv(f'short_params_{debug}.csv', index=False)


def read_csv_and_generate_audio(cfg, synth_plugin_path):
    dataframe = pd.read_csv(cfg.csv_filename)
    sounds = {}

    for _, row in dataframe.iterrows():
        engine = daw.RenderEngine(cfg.sample_rate, cfg.buffer_size)
        engine.set_bpm(120.)  # default is 120.
        synth = engine.make_plugin_processor("my_synth", synth_plugin_path)
        assert synth.get_name() == "my_synth"
        graph = [
            (synth, [])
        ]
        engine.load_graph(graph)

        if cfg.synth_version == 5:
            index = int(row['wav_id'])
            osc2waveform = row['24.Osc 1 Waveform']
            lfo1waveform = row['26.Lfo 1 Waveform']
            lfo1destination = row['32.Lfo 1 Destination']
            lfo1amount = row['30.Lfo 1 Amount']
            lfo1rate = row['28.Lfo 1 Rate']
            filter_cutoff = row['3.Filter Cutoff']
            filter_resonance = row['4.Filter Resonance']

        elif cfg.synth_version == 4:
            index = int(row['wav_id'])
            osc2waveform = row['24.osc2waveform']
            lfo1waveform = row['26.lfo1waveform']
            lfo1destination = row['32.lfo1destination']
            lfo1amount = row['30.lfo1amount']
            lfo1rate = row['28.lfo1rate']
            filter_cutoff = row['3.cutoff']
            filter_resonance = row['4.resonance']

        reset_all_params(synth)
        set_params(synth,
                          osc2waveform=osc2waveform,
                          lfo1waveform=lfo1waveform,
                          lfo1destination=lfo1destination,
                          lfo1amount=lfo1amount,
                          lfo1rate=lfo1rate,
                          filter_cutoff=filter_cutoff,
                          filter_resonance=filter_resonance)
        set_non_randomized_params(synth)
        print_synth_params(synth)
        synth.add_midi_note(cfg.midi_note, cfg.velocity, cfg.start_sec, cfg.note_duration)
        engine.render(cfg.duration)
        audio = engine.get_audio()
        if cfg.mono:
            audio = (audio[0] + audio[1]) / 2
        audio_path = os.path.join('audio_output', f'{index}.wav')
        wavfile.write(audio_path, cfg.sample_rate, audio.transpose())
        sounds[index] = audio
        print(f"generated sound {index + 1}/{dataframe.shape[0]}")
        synth.clear_midi()

    if cfg.debug_sound_identity:
        check_sound_identity(sounds)


def read_csv_and_generate_audio(cfg, synth_plugin_path):
    dataframe = pd.read_csv(cfg.full_csv_filename)
    sounds = {}

    engine = daw.RenderEngine(cfg.sample_rate, cfg.buffer_size)
    engine.set_bpm(120.)  # default is 120.
    synth = engine.make_plugin_processor("my_synth", synth_plugin_path)
    assert synth.get_name() == "my_synth"
    graph = [
        (synth, [])
    ]
    engine.load_graph(graph)

    for index, params in tqdm(dataframe.iterrows()):
        reset_all_params(synth)
        wav_name = int(params.iloc[0])
        set_params(synth, params.iloc[1:])
        set_non_randomized_params(synth)
        # print_synth_params(synth)
        synth.add_midi_note(cfg.midi_note, cfg.velocity, cfg.start_sec, cfg.note_duration)
        engine.render(cfg.duration)
        audio = engine.get_audio()
        if cfg.mono:
            audio = (audio[0] + audio[1]) / 2
        audio_path = os.path.join(cfg.save_folder, f'{wav_name}.wav')
        wavfile.write(audio_path, cfg.sample_rate, audio.transpose())
        sounds[index] = audio
        # print(f"generated sound {index + 1}/{dataframe.shape[0]}")
        synth.clear_midi()

    if cfg.debug_sound_identity:
        check_sound_identity(sounds)


def run():
    FLAG_DEBUG = 'osc_lfo_am'  # 'osc_with_lfo' / 'osc_only' / 'osc_lfo_am' / 'no'
    FLAG_CREATE_DATA = 0
    cfg = Config()

    if not (FLAG_DEBUG == 'no'):
        # cfg.dataset_size = 250000
        cfg.dataset_size = 1000
        # cfg.full_csv_filename = f'full_parameters_{FLAG_DEBUG}.csv'
        # cfg.save_folder = 'audio_output_osc_lfo_am'

    synth_plugin_path = os.path.abspath(cfg.synth_plugin)
    engine = daw.RenderEngine(cfg.sample_rate, cfg.buffer_size)
    # print('got here 0')
    synth = engine.make_plugin_processor("my_synth", synth_plugin_path)
    # print('got here!')
    # exit()
    if cfg.use_preset:
        synth.load_preset(cfg.synth_preset)
    dataset_dict = {}
    init_dataset_dict(synth, dataset_dict)
    # print_synth_params(synth)

    # create_csv(cfg, synth_plugin_path, dataset_dict)
    # read_csv_and_generate_audio(cfg, synth_plugin_path)
    if FLAG_CREATE_DATA:
        my_create_csv(cfg, FLAG_DEBUG)
    read_csv_and_generate_audio(cfg, synth_plugin_path)
    print('Done!')

    # create_csv(cfg, synth_plugin_path, dataset_dict)
    # read_csv_and_generate_audio(cfg, synth_plugin_path)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()

import random
import time
from tqdm import tqdm
import julius
import numpy as np
import torch
from args import cfg
from scipy.io.wavfile import write
# from utils import get_unique_labels
import pandas as pd

PI = 3.141592653589793
TWO_PI = 2 * PI

SAMPLE_RATE = 16384
SIGNAL_DURATION_SEC = 1.0

def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to")


def smoothed_square_wave(x, delta=0.01):
    return 2 * torch.arctan(torch.sin(TWO_PI * x) / delta) / PI

def smoothed_triangle_wave(x, delta=0.01):
    return 1 - 2 * torch.arccos((1 - delta) * torch.sin(2 * PI * x)) / PI

def smoothed_sawtooth_wave(x, delta=0.01):
    return 1 + smoothed_triangle_wave((2 * x - 1) / 4) * smoothed_square_wave(x / 2) / 2


class SynthModules:
    def __init__(self, num_sounds=1):
        self.sample_rate = SAMPLE_RATE
        self.sig_duration = SIGNAL_DURATION_SEC
        self.time_samples = torch.linspace(0, self.sig_duration, steps=int(self.sample_rate * self.sig_duration),
                                           requires_grad=True)
        self.modulation_time = torch.linspace(0, self.sig_duration, steps=self.sample_rate)
        self.modulation = 0
        self.signal = torch.zeros(size=(num_sounds, self.time_samples.shape[0]), dtype=torch.float32,
                                  requires_grad=True)

        self.device = cfg.device
        self.time_samples = move_to(self.time_samples, self.device)
        self.modulation_time = move_to(self.modulation_time, self.device)
        self.signal = move_to(self.signal, self.device)
        # self.room_impulse_responses = torch.load('rir_for_reverb_no_amp')

    def oscillator(self, amp, freq, waveform, phase=0, num_sounds=1):
        """Creates a basic oscillator.

            Retrieves a waveform shape and attributes, and construct the respected signal

            Args:
                self: Self object
                amp: Amplitude in range [0, 1]
                freq: Frequency in range [0, 22000]
                phase: Phase in range [0, 2pi], default is 0
                waveform: a string, one of ['sine', 'square', 'triangle', 'sawtooth'] or a probability vector
                num_sounds: number of sounds to process

            Returns:
                A torch with the constructed signal

            Raises:
                ValueError: Provided variables are out of range
                :rtype: object
            """

        self.signal_values_sanity_check(amp, freq, waveform)
        t = self.time_samples
        # oscillator = torch.zeros_like(t, requires_grad=True)

        oscillator_tensor = torch.tensor((), requires_grad=True).to(cfg.device)
        first_time = True
        for i in range(num_sounds):

            if num_sounds == 1:
                freq_float = freq
            else:
                freq_float = freq[i]

            sine_wave = amp * torch.sin(TWO_PI * freq_float * t + phase)
            square_wave = amp * torch.sign(torch.sin(TWO_PI * freq_float * t + phase))
            triangle_wave = (2 * amp / PI) * torch.arcsin(torch.sin((TWO_PI * freq_float * t + phase)))
            # triangle_wave = amp * torch.sin(TWO_PI * freq_float * t + phase)
            sawtooth_wave = 2 * (t * freq_float - torch.floor(0.5 + t * freq_float))  # Sawtooth closed form
            # Phase shift (by normalization to range [0,1] and modulo operation)
            sawtooth_wave = (((sawtooth_wave + 1) / 2) + phase / TWO_PI) % 1
            sawtooth_wave = amp * (sawtooth_wave * 2 - 1)  # re-normalization to range [-amp, amp]
            none_wave = torch.ones_like(sine_wave)

            # ## My version
            # square_wave = amp * smoothed_square_wave(freq_float * t)
            # triangle_wave = amp * smoothed_triangle_wave(freq_float * t)
            # sawtooth_wave = amp * smoothed_sawtooth_wave(freq_float * t)
            # ##$$$$$$$$$$$

            if isinstance(waveform, str):
                if waveform == 'sine':
                    oscillator = sine_wave
                elif waveform == 'square':
                    oscillator = square_wave
                elif waveform == 'triangle':
                    oscillator = triangle_wave
                elif waveform == 'sawtooth':
                    oscillator = sawtooth_wave
                elif waveform == 'none':
                    oscillator = none_wave

            else:
                waveform_probabilities = waveform[i]
                oscillator = waveform_probabilities[0] * sine_wave \
                             + waveform_probabilities[1] * square_wave \
                             + waveform_probabilities[2] * triangle_wave \
                             + waveform_probabilities[3] * sawtooth_wave

            if first_time:
                oscillator_tensor = torch.cat((oscillator_tensor, oscillator), dim=0).unsqueeze(dim=0)
                first_time = False
            else:
                oscillator = oscillator.unsqueeze(dim=0)
                oscillator_tensor = torch.cat((oscillator_tensor, oscillator), dim=0)

        return oscillator_tensor

    def mix_signal(self, new_signal, factor):
        """Signal superposition. factor balances the mix
        0 - original signal only, 1 - new signal only, 0.5 evenly balanced. """
        if factor < 0 or factor > 1:
            raise ValueError("Provided factor value is out of range [0, 1]")
        self.signal = factor * self.signal + (1 - factor) * new_signal

    def oscillator_fm(self, amp_c, freq_c, waveform, mod_index, modulator, num_sounds=1):
        """Basic oscillator with FM modulation

            Creates an oscillator and modulates its frequency by a given modulator

            Args:
                self: Self object
                amp_c: Amplitude in range [0, 1]
                freq_c: Frequency in range [0, 22000]
                waveform: One of [sine, square, triangle, sawtooth]
                mod_index: Modulation index, which affects the amount of modulation
                modulator: Modulator signal, to affect carrier frequency
                num_sounds: number of sounds to process

            Returns:
                A torch with the constructed FM signal

            Raises:
                ValueError: Provided variables are out of range
            """

        self.signal_values_sanity_check(amp_c, freq_c, waveform)
        t = self.time_samples
        oscillator_tensor = torch.tensor((), requires_grad=True).to(cfg.device)
        first_time = True
        for i in range(num_sounds):
            if num_sounds == 1:
                amp_float = amp_c
                mod_index_float = mod_index
                freq_float = freq_c
                input_signal_cur = modulator
            else:
                amp_float = amp_c[i]
                mod_index_float = mod_index[i]
                freq_float = freq_c[i]
                input_signal_cur = modulator[i]

            fm_sine_wave = amp_float * torch.sin(TWO_PI * freq_float * t + mod_index_float * torch.cumsum(input_signal_cur, dim=1))
            fm_square_wave = amp_float * torch.sign(torch.sin(TWO_PI * freq_float * t + mod_index_float * torch.cumsum(input_signal_cur, dim=1)))
            # fm_triangle_wave = (2 * amp_float / PI) * torch.arcsin(torch.sin((TWO_PI * freq_float * t + mod_index_float * input_signal_cur)))
            fm_triangle_wave = (2 * amp_float / PI) * torch.arcsin(
                torch.sin((TWO_PI * freq_float * t + mod_index_float * torch.cumsum(input_signal_cur, dim=1))))

            fm_sawtooth_wave = 2 * (t * freq_float - torch.floor(0.5 + t * freq_float))
            fm_sawtooth_wave = (((fm_sawtooth_wave + 1) / 2) + mod_index_float * torch.cumsum(input_signal_cur,dim=1) / TWO_PI) % 1
            fm_sawtooth_wave = amp_float * (fm_sawtooth_wave * 2 - 1)

            if isinstance(waveform, list):
                waveform = waveform[i]

            if isinstance(waveform, str):
                if waveform == 'sine':
                    oscillator = fm_sine_wave
                elif waveform == 'square':
                    oscillator = fm_square_wave
                elif waveform == 'triangle':
                    oscillator = fm_triangle_wave
                elif waveform == 'sawtooth':
                    oscillator = fm_sawtooth_wave

            else:
                if num_sounds == 1:
                    waveform_probabilities = waveform

                elif torch.is_tensor(waveform):
                    waveform_probabilities = waveform[i]

                oscillator = waveform_probabilities[0] * fm_sine_wave \
                             + waveform_probabilities[1] * fm_square_wave \
                             + waveform_probabilities[2] * fm_sawtooth_wave

            if first_time:
                if num_sounds == 1:
                    oscillator_tensor = oscillator
                else:
                    oscillator_tensor = torch.cat((oscillator_tensor, oscillator), dim=0).unsqueeze(dim=0)
                    first_time = False
            else:
                oscillator = oscillator.unsqueeze(dim=0)
                oscillator_tensor = torch.cat((oscillator_tensor, oscillator), dim=0)

        return oscillator_tensor

    def am_modulation(self, amp_c, freq_c, amp_m, freq_m, final_max_amp, waveform):
        """AM modulation

            Modulates the amplitude of a carrier signal with a sine modulator
            see https://en.wikipedia.org/wiki/Amplitude_modulation

            Args:
                self: Self object
                amp_c: Amplitude of carrier in range [0, 1]
                freq_c: Frequency of carrier in range [0, 22000]
                amp_m: Amplitude of modulator in range [0, 1]
                freq_m: Frequency of modulator in range [0, 22000]
                final_max_amp: The final maximum amplitude of the modulated signal
                waveform: One of [sine, square, triangle, sawtooth]

            Returns:
                A torch with the constructed AM signal

            Raises:
                ValueError: Provided variables are out of range
                ValueError: modulation index > 1. Amplitude values must obey amp_m < amp_c
                # todo add documentation for sensible frequency values
            """
        self.signal_values_sanity_check(amp_m, freq_m, waveform)
        self.signal_values_sanity_check(amp_c, freq_c, waveform)
        modulation_index = amp_m / amp_c
        if modulation_index > 1:
            raise ValueError("Provided amplitudes results modulation index > 1, and yields over-modulation ")
        if final_max_amp < 0 or final_max_amp > 1:
            raise ValueError("Provided final max amplitude is not in range [0, 1]")
        # todo: add restriction freq_c >> freq_m

        t = self.time_samples
        dc = 1
        carrier = SynthModules()
        carrier.oscillator(amp=amp_c, freq=freq_c, phase=0, waveform=waveform)
        modulator = amp_m * torch.sin(TWO_PI * freq_m * t)
        am_signal = (dc + modulator / amp_c) * carrier.signal
        normalized_am_signal = (final_max_amp / (amp_c + amp_m)) * am_signal
        self.signal = normalized_am_signal

    def am_modulation_by_input_signal(self, input_signal, modulation_factor, amp_c, freq_c, waveform):
        """AM modulation by an input signal

            Modulates the amplitude of a carrier signal with a provided input signal
            see https://en.wikipedia.org/wiki/Amplitude_modulation

            Args:
                self: Self object
                input_signal: Input signal to be used as modulator
                modulation_factor: factor to be multiplied by modulator, in range [0, 1]
                amp_c: Amplitude of carrier in range [0, 1]
                freq_c: Frequency of carrier in range [0, 22000]
                waveform: Waveform of carrier. One of [sine, square, triangle, sawtooth]

            Returns:
                A torch with the constructed AM signal

            Raises:
                ValueError: Provided variables are inappropriate
                ValueError: Resulted Amplitude is out of range [-1, 1]
            """
        self.signal_values_sanity_check(amp_c, freq_c, waveform)
        carrier = SynthModules()
        carrier.oscillator(amp=1, freq=freq_c, phase=0, waveform=waveform)
        modulated_amplitude = (amp_c + modulation_factor * input_signal)
        if torch.max(modulated_amplitude).item() > 1 or torch.min(modulated_amplitude).item() < -1:
            raise ValueError("AM modulation resulted amplitude out of range [-1, 1].")
        self.signal = modulated_amplitude * carrier.signal

    def my_am_modulation_by_input_signal(self, input_signal, modulation_factor, amp_c, freq_c, waveform, num_sounds=1):
        """AM modulation by an input signal

            Modulates the amplitude of a carrier signal with a provided input signal
            see https://en.wikipedia.org/wiki/Amplitude_modulation

            Args:
                self: Self object
                input_signal: Input signal to be used as modulator
                modulation_factor: factor to be multiplied by modulator, in range [0, 1]
                amp_c: Amplitude of carrier in range [0, 1]
                freq_c: Frequency of carrier in range [0, 22000]
                waveform: Waveform of carrier. One of [sine, square, triangle, sawtooth]

            Returns:
                A torch with the constructed AM signal

            Raises:
                ValueError: Provided variables are inappropriate
                ValueError: Resulted Amplitude is out of range [-1, 1]
            """
        # self.signal_values_sanity_check(amp_c, freq_c, waveform)
        carrier = SynthModules()
        oscillator = carrier.oscillator(amp=1, freq=freq_c, phase=0, waveform=waveform, num_sounds=num_sounds)
        modulated_amplitude = (amp_c + modulation_factor * input_signal)
        if torch.max(modulated_amplitude).item() > 1 or torch.min(modulated_amplitude).item() < -1:
            raise ValueError("AM modulation resulted amplitude out of range [-1, 1].")
        self.signal = modulated_amplitude * oscillator
        return self.signal

    def tremolo_for_input_signal(self, input_signal, amount, freq_m, waveform_m):
        """tremolo effect for an input signal

            This is a kind of AM modulation, where the signal is multiplied as a whole by a given modulator.
            The modulator is shifted such that it resides in range [start, 1], where start is <1 - amount>.
            so start is > 0, such that the original amplitude of the input audio is preserved and there is no phase
            shift due to multiplication by negative number.

            Args:
                self: Self object
                input_signal: Input signal to be used as carrier
                amount: amount of effect, in range [0, 1]
                freq_m: Frequency of modulator in range [0, 20]
                waveform_m: Waveform of modulator. One of [sine, square, triangle, sawtooth]

            Returns:
                A torch with the constructed AM signal

            Raises:
                ValueError: Provided variables are inappropriate
                ValueError: Amount is out of range [-1, 1]
            """
        self.signal_values_sanity_check(amp=1, freq=freq_m, waveform=waveform_m)
        if amount > 1 or amount < 0:
            ValueError("amount is out of range [0, 1]")
        modulator = SynthModules()
        modulator.signal = modulator.oscillator(amp=1, freq=freq_m, phase=0, waveform=waveform_m)
        modulator.signal = amount * (modulator.signal + 1) / 2 + (1 - amount)
        #
        # modulatorplot = modulator.signal[0]
        # modulatorplot = modulatorplot.detach().cpu().numpy()
        # plt.figure()
        # plt.plot(modulatorplot)
        # plt.ylabel("Amplitude")
        # plt.xlabel("Time")
        # plt.title("MODULATOR AM PLOT")
        # plt.show()

        am_signal = input_signal * modulator.signal

        # am_signalplot = am_signal[0]
        # am_signalplot = am_signalplot.detach().cpu().numpy()
        # plt.figure()
        # plt.plot(am_signalplot)
        # plt.ylabel("Amplitude")
        # plt.xlabel("Time")
        # plt.title("AM SIGNAL PLOT")
        # plt.show()

        return am_signal

    def adsr_envelope(self, input_signal, attack_t, decay_t, sustain_t, sustain_level, release_t, num_sounds=1):
        """Apply an ADSR envelope to the signal

            builds the ADSR shape and multiply by the signal

            Args:
                self: Self object
                input_signal: target signal to apply adsr
                attack_t: Length of attack in seconds. Time to go from 0 to 1 amplitude.
                decay_t: Length of decay in seconds. Time to go from 1 amplitude to sustain level.
                sustain_t: Length of sustain in seconds, with sustain level amplitude
                sustain_level: Sustain volume level
                release_t: Length of release in seconds. Time to go ftom sustain level to 0 amplitude
                num_sounds: number of sounds to process

            Raises:
                ValueError: Provided ADSR timings are not the same as the signal length
            """
        if num_sounds == 1:
            if attack_t + decay_t + sustain_t + release_t > self.sig_duration:
                raise ValueError("Provided ADSR durations exceeds signal duration")

        else:
            for i in range(num_sounds):
                if attack_t[i] + decay_t[i] + sustain_t[i] + release_t[i] > self.sig_duration:
                    raise ValueError("Provided ADSR durations exceeds signal duration")

        if num_sounds == 1:
            attack_num_samples = int(self.sample_rate * attack_t)
            decay_num_samples = int(self.sample_rate * decay_t)
            sustain_num_samples = int(self.sample_rate * sustain_t)
            release_num_samples = int(self.sample_rate * release_t)
        else:
            attack_num_samples = [torch.floor(torch.tensor(self.sample_rate * attack_t[k])) for k in range(num_sounds)]
            decay_num_samples = [torch.floor(torch.tensor(self.sample_rate * decay_t[k])) for k in range(num_sounds)]
            sustain_num_samples = [torch.floor(torch.tensor(self.sample_rate * sustain_t[k])) for k in
                                   range(num_sounds)]
            release_num_samples = [torch.floor(torch.tensor(self.sample_rate * release_t[k])) for k in
                                   range(num_sounds)]
            attack_num_samples = torch.stack(attack_num_samples)
            decay_num_samples = torch.stack(decay_num_samples)
            sustain_num_samples = torch.stack(sustain_num_samples)
            release_num_samples = torch.stack(release_num_samples)

        if num_sounds > 1:
            # todo: change the check with sustain_level[0]
            if torch.is_tensor(sustain_level[0]):
                sustain_level = [sustain_level[i] for i in range(num_sounds)]
                sustain_level = torch.stack(sustain_level)
            else:
                sustain_level = [sustain_level[i] for i in range(num_sounds)]

        enveloped_signal_tensor = torch.tensor((), requires_grad=True).to(cfg.device)
        first_time = True
        for i in range(num_sounds):

            if num_sounds == 1:
                attack = torch.linspace(0, 1, attack_num_samples)
                decay = torch.linspace(1, sustain_level, decay_num_samples)
                sustain = torch.full((sustain_num_samples,), sustain_level)
                release = torch.linspace(sustain_level, 0, release_num_samples)
            else:
                attack = torch.linspace(0, 1, int(attack_num_samples[i].item()), device=cfg.device)
                decay = torch.linspace(1, sustain_level[i], int(decay_num_samples[i]), device=cfg.device)
                sustain = torch.full((int(sustain_num_samples[i].item()),), sustain_level[i], device=cfg.device)
                release = torch.linspace(sustain_level[i], 0, int(release_num_samples[i].item()), device=cfg.device)

                # todo: make sure ADSR behavior is differentiable. linspace has to know to get tensors
                # attack_mod = helper.linspace(torch.tensor(0), torch.tensor(1), attack_num_samples[i])
                # decay_mod = helper.linspace(torch.tensor(1), sustain_level[i], decay_num_samples[i])
                # sustain_mod = torch.full((sustain_num_samples[i],), sustain_level[i])
                # release_mod = helper.linspace(sustain_num_samples[i], torch.tensor(0), release_num_samples[i])

            envelope = torch.cat((attack, decay, sustain, release))
            envelope = move_to(envelope, self.device)

            # envelope = torch.cat((attack_mod, decay_mod, sustain_mod, release_mod))

            envelope_len = envelope.shape[0]
            signal_len = self.time_samples.shape[0]
            if envelope_len <= signal_len:
                padding = torch.zeros((signal_len - envelope_len), device=cfg.device)
                envelope = torch.cat((envelope, padding))
            else:
                raise ValueError("Envelope length exceeds signal duration")

            if torch.is_tensor(input_signal) and num_sounds > 1:
                signal_to_shape = input_signal[i]
            else:
                signal_to_shape = input_signal

            enveloped_signal = signal_to_shape * envelope

            if first_time:
                if num_sounds == 1:
                    enveloped_signal_tensor = enveloped_signal
                else:
                    enveloped_signal_tensor = torch.cat((enveloped_signal_tensor, enveloped_signal), dim=0).unsqueeze(
                        dim=0)
                    first_time = False
            else:
                enveloped = enveloped_signal.unsqueeze(dim=0)
                enveloped_signal_tensor = torch.cat((enveloped_signal_tensor, enveloped), dim=0)

            # if DEBUG_MODE:
            #     plt.plot(envelope.cpu())
            #     plt.plot(self.signal.cpu())
            #     plt.show()

        return enveloped_signal_tensor

    def filter(self, input_signal, filter_freq, filter_type, num_sounds=1):
        """Apply an ADSR envelope to the signal

            builds the ADSR shape and multiply by the signal

            Args:
                self: Self object
                :param input_signal: 1D or 2D array or tensor to apply filter along rows
                :param filter_type: one of ['low_pass', 'high_pass', 'band_pass']
                :param filter_freq: corner or central frequency
                :param num_sounds: number of sounds in the input

            Raises:
                none

            """
        filtered_signal_tensor = torch.tensor((), requires_grad=True).to(cfg.device)
        first_time = True
        for i in range(num_sounds):
            if num_sounds == 1:
                filter_frequency = filter_freq
            elif num_sounds > 1:
                filter_frequency = filter_freq[i]

            if torch.is_tensor(filter_frequency):
                filter_frequency = move_to(filter_frequency, "cpu")
            high_pass_signal = self.high_pass(input_signal[i], cutoff_freq=filter_frequency, index=i)
            low_pass_signal = self.low_pass(input_signal[i], cutoff_freq=filter_frequency, index=i)

            if isinstance(filter_type, list):
                filter_type = filter_type[i]

            if isinstance(filter_type, str):
                if filter_type == 'high_pass':
                    filtered_signal = high_pass_signal
                elif filter_type == 'low_pass':
                    filtered_signal = low_pass_signal

            else:
                if num_sounds == 1:
                    filter_type_probabilities = filter_type
                else:
                    filter_type_probabilities = filter_type[i]

                filter_type_probabilities = filter_type_probabilities.cpu()
                filtered_signal = filter_type_probabilities[0] * high_pass_signal \
                                  + filter_type_probabilities[1] * low_pass_signal
                filtered_signal = filtered_signal.to(cfg.device)

            if first_time:
                if num_sounds == 1:
                    filtered_signal_tensor = filtered_signal
                else:
                    filtered_signal_tensor = torch.cat((filtered_signal_tensor, filtered_signal), dim=0).unsqueeze(
                        dim=0)
                    first_time = False
            else:
                filtered_signal = filtered_signal.unsqueeze(dim=0)
                filtered_signal_tensor = torch.cat((filtered_signal_tensor, filtered_signal), dim=0)

        return filtered_signal_tensor

    def low_pass(self, input_signal, cutoff_freq, q=0.707, index=0):
        # filtered_waveform = taF.lowpass_biquad(input_signal, self.sample_rate, cutoff_freq, q)
        # filtered_waveform = julius.lowpass_filter(input_signal, cutoff_freq.item()/44100)
        if cutoff_freq == 0:
            return input_signal
        else:
            filtered_waveform_new = julius.lowpass_filter_new(input_signal, cutoff_freq / SAMPLE_RATE)
            return filtered_waveform_new

    def high_pass(self, input_signal, cutoff_freq, q=0.707, index=0):
        # filtered_waveform = julius.lowpass_filter(input_signal, cutoff_freq.item()/44100)
        # filtered_waveform_new = julius.highpass_filter(input_signal, cutoff_freq/44100)
        if cutoff_freq == 0:
            return input_signal
        else:
            filtered_waveform_new = julius.highpass_filter_new(input_signal, cutoff_freq / SAMPLE_RATE)
            return filtered_waveform_new

    @staticmethod
    # todo: remove all except list instances
    def signal_values_sanity_check(amp, freq, waveform):
        """Check signal properties are reasonable."""
        if isinstance(freq, float):
            if freq < 0 or freq > 20000:
                raise ValueError("Provided frequency is not in range [0, 20000]")
        elif isinstance(freq, list):
            if any(element < 0 or element > 2000 for element in freq):
                raise ValueError("Provided frequency is not in range [0, 20000]")
        if isinstance(amp, int):
            if amp < 0 or amp > 1:
                raise ValueError("Provided amplitude is not in range [0, 1]")
        elif isinstance(amp, list):
            if any(element < 0 or element > 3 for element in amp): ############################ 1
                raise ValueError("Provided amplitude is not in range [0, 1]")
        if isinstance(waveform, str):
            if not any(x in waveform for x in ['sine', 'square', 'triangle', 'sawtooth', 'none']):
                raise ValueError("Unknown waveform provided")


class SynthBasicFlow:
    """A basic synthesizer signal flow architecture.
        The synth is based over common commercial software synthesizers.
        It has dual oscillators followed by FM module, summed together
        and passed in a frequency filter and envelope shaper

        [osc1] -> FM
                    \
                     + -> [frequency filter] -> [envelope shaper] -> output sound
                    /
        [osc2] -> FM

        Args:
            self: Self object
            file_name: name for sound
            parameters_dict(optional): parameters for the synth components to generate specific sounds
            num_sounds: number of sounds to generate.
        """

    def __init__(self, file_name='unnamed_sound', parameters_dict=None, num_sounds=1):
        self.file_name = file_name
        self.params_dict = {}
        # init parameters_dict
        if parameters_dict is None:
            self.init_random_synth_params(num_sounds)
        elif type(parameters_dict) is dict:
            self.params_dict = parameters_dict.copy()
        else:
            ValueError("Provided parameters are not provided as dictionary")

        # generate signal with basic signal flow
        # self.signal = self.generate_signal(num_sounds)

    # def init_random_synth_params(self, num_sounds):
    #     """init params_dict with lists of parameters"""
    #
    #     # todo: refactor: initializations by iterating/referencing synth.PARAM_LIST
    #     self.params_dict['osc1_amp'] = np.random.random_sample(size=num_sounds)
    #     self.params_dict['osc1_freq'] = random.choices(OSC_FREQ_LIST, k=num_sounds)
    #     self.params_dict['osc1_wave'] = random.choices(list(WAVE_TYPE_DIC), k=num_sounds)
    #     self.params_dict['osc1_mod_index'] = np.random.uniform(low=0, high=MAX_MOD_INDEX, size=num_sounds)
    #     self.params_dict['lfo1_freq'] = np.random.uniform(low=0, high=MAX_LFO_FREQ, size=num_sounds)
    #
    #     self.params_dict['osc2_amp'] = np.random.random_sample(size=num_sounds)
    #     self.params_dict['osc2_freq'] = random.choices(OSC_FREQ_LIST, k=num_sounds)
    #     self.params_dict['osc2_wave'] = random.choices(list(WAVE_TYPE_DIC), k=num_sounds)
    #     self.params_dict['osc2_mod_index'] = np.random.uniform(low=0, high=MAX_MOD_INDEX, size=num_sounds)
    #     self.params_dict['lfo2_freq'] = np.random.uniform(low=0, high=MAX_LFO_FREQ, size=num_sounds)
    #
    #     self.params_dict['filter_type'] = random.choices(list(FILTER_TYPE_DIC), k=num_sounds)
    #     self.params_dict['filter_freq'] = \
    #         np.random.uniform(low=MIN_FILTER_FREQ, high=MAX_FILTER_FREQ, size=num_sounds)
    #
    #     attack_t = np.random.random_sample(size=num_sounds)
    #     decay_t = np.random.random_sample(size=num_sounds)
    #     sustain_t = np.random.random_sample(size=num_sounds)
    #     release_t = np.random.random_sample(size=num_sounds)
    #     adsr_sum = attack_t + decay_t + sustain_t + release_t
    #     attack_t = attack_t / adsr_sum
    #     decay_t = decay_t / adsr_sum
    #     sustain_t = sustain_t / adsr_sum
    #     release_t = release_t / adsr_sum
    #
    #     # fixing a numerical issue in case the ADSR times exceeds signal length
    #     adsr_aggregated_time = attack_t + decay_t + sustain_t + release_t
    #     overflow_indices = [idx for idx, val in enumerate(adsr_aggregated_time) if val > SIGNAL_DURATION_SEC]
    #     attack_t[overflow_indices] -= 1e-6
    #     decay_t[overflow_indices] -= 1e-6
    #     sustain_t[overflow_indices] -= 1e-6
    #     release_t[overflow_indices] -= 1e-6
    #
    #     self.params_dict['attack_t'] = attack_t
    #     self.params_dict['decay_t'] = decay_t
    #     self.params_dict['sustain_t'] = sustain_t
    #     self.params_dict['release_t'] = release_t
    #     self.params_dict['sustain_level'] = np.random.random_sample(size=num_sounds)
    #
    #     for key, val in self.params_dict.items():
    #         if isinstance(val, np.ndarray):
    #             self.params_dict[key] = val.tolist()
    #
    #     if num_sounds == 1:
    #         for key, value in self.params_dict.items():
    #             self.params_dict[key] = value[0]

    def generate_signal(self, num_sounds):
        osc1_amp = self.params_dict['osc1_amp']
        osc1_freq = self.params_dict['osc1_freq']
        osc1_wave = self.params_dict['osc1_wave']
        osc1_mod_index = self.params_dict['osc1_mod_index']
        lfo1_freq = self.params_dict['lfo1_freq']

        osc2_amp = self.params_dict['osc2_amp']
        osc2_freq = self.params_dict['osc2_freq']
        osc2_wave = self.params_dict['osc2_wave']
        osc2_mod_index = self.params_dict['osc2_mod_index']
        lfo2_freq = self.params_dict['lfo2_freq']

        filter_type = self.params_dict['filter_type']
        filter_freq = self.params_dict['filter_freq']

        attack_t = self.params_dict['attack_t']
        decay_t = self.params_dict['decay_t']
        sustain_t = self.params_dict['sustain_t']
        release_t = self.params_dict['release_t']
        sustain_level = self.params_dict['sustain_level']

        synth = SynthModules(num_sounds)

        lfo1 = synth.oscillator(amp=1,
                                freq=lfo1_freq,
                                phase=0,
                                waveform='sine',
                                num_sounds=num_sounds)

        fm_osc1 = synth.oscillator_fm(amp_c=osc1_amp,
                                      freq_c=osc1_freq,
                                      waveform=osc1_wave,
                                      mod_index=osc1_mod_index,
                                      modulator=lfo1,
                                      num_sounds=num_sounds)

        lfo2 = synth.oscillator(amp=1,
                                freq=lfo2_freq,
                                phase=0,
                                waveform='sine',
                                num_sounds=num_sounds)

        fm_osc2 = synth.oscillator_fm(amp_c=osc2_amp,
                                      freq_c=osc2_freq,
                                      waveform=osc2_wave,
                                      mod_index=osc2_mod_index,
                                      modulator=lfo2,
                                      num_sounds=num_sounds)

        mixed_signal = (fm_osc1 + fm_osc2) / 2

        # mixed_signal = mixed_signal.cpu()

        filtered_signal = synth.filter(mixed_signal, filter_freq, filter_type, num_sounds)

        enveloped_signal = synth.adsr_envelope(filtered_signal,
                                               attack_t,
                                               decay_t,
                                               sustain_t,
                                               sustain_level,
                                               release_t,
                                               num_sounds)

        return enveloped_signal

    def my_generate_signal(self, num_sounds):
        # t = time.time()

        # osc1_amp = self.params_dict['osc1_amp']
        osc1_freq = self.params_dict['osc1_freq']
        osc1_amp = 1.
        osc1_wave = self.params_dict['osc1_wave']
        osc1_mod_index = self.params_dict['osc1_mod_index']
        lfo1_freq = self.params_dict['lfo1_freq']
        lfo1_wave = self.params_dict['lfo1_wave']
        # lfo1_wave = 'sine'

        # osc2_amp = self.params_dict['osc2_amp']
        # osc2_freq = self.params_dict['osc2_freq']
        # osc2_wave = self.params_dict['osc2_wave']
        # osc2_mod_index = self.params_dict['osc2_mod_index']
        # lfo2_freq = self.params_dict['lfo2_freq']

        # osc3_amp = self.params_dict['osc3_amp']
        # osc3_freq = self.params_dict['osc3_freq']
        # osc3_wave = self.params_dict['osc3_wave']
        # osc3_mod_index = self.params_dict['osc3_mod_index']
        # lfo3_freq = self.params_dict['lfo3_freq']

        am_mod_wave = self.params_dict['am_mod_wave']
        am_mod_freq = self.params_dict['am_mod_freq']
        am_mod_amount = self.params_dict['am_mod_amount']


        # filter_type = self.params_dict['filter_type']
        filter_type = 'low_pass'
        filter_freq = self.params_dict['filter_freq']

        # attack_t = self.params_dict['attack_t']
        # decay_t = self.params_dict['decay_t']
        # sustain_t = self.params_dict['sustain_t']
        # release_t = self.params_dict['release_t']
        # sustain_level = self.params_dict['sustain_level']

        synth = SynthModules(num_sounds)

        # if num_sounds > 1:
        #     num_osc_work = (torch.FloatTensor(osc1_amp) + \
        #                    torch.FloatTensor(osc2_amp) + \
        #                    torch.FloatTensor(osc3_amp)).unsqueeze(1).to(cfg.device)
        # else:
        #     num_osc_work = osc1_amp + osc2_amp + osc3_amp

        num_osc_work = 1.

        lfo1 = synth.oscillator(amp=1,
                                freq=lfo1_freq,
                                phase=0,
                                waveform=lfo1_wave,
                                num_sounds=num_sounds)

        fm_osc1 = synth.oscillator_fm(amp_c=osc1_amp,
                                      freq_c=osc1_freq,
                                      waveform=osc1_wave,
                                      mod_index=osc1_mod_index,
                                      modulator=lfo1,
                                      num_sounds=num_sounds)

        # lfo2 = synth.oscillator(amp=1,
        #                         freq=lfo2_freq,
        #                         phase=0,
        #                         waveform='sine',
        #                         num_sounds=num_sounds)
        #
        # fm_osc2 = synth.oscillator_fm(amp_c=osc2_amp,
        #                               freq_c=osc2_freq,
        #                               waveform=osc2_wave,
        #                               mod_index=osc2_mod_index,
        #                               modulator=lfo2,
        #                               num_sounds=num_sounds)
        #
        # lfo3 = synth.oscillator(amp=1,
        #                         freq=lfo3_freq,
        #                         phase=0,
        #                         waveform='sine',
        #                         num_sounds=num_sounds)
        #
        # fm_osc3 = synth.oscillator_fm(amp_c=osc3_amp,
        #                               freq_c=osc3_freq,
        #                               waveform=osc3_wave,
        #                               mod_index=osc3_mod_index,
        #                               modulator=lfo3,
        #                               num_sounds=num_sounds)

        # mixed_signal = torch.div((fm_osc1 + fm_osc2 + fm_osc3), num_osc_work)
        mixed_signal = fm_osc1

        am_modulated_signal = synth.tremolo_for_input_signal(mixed_signal, am_mod_amount, am_mod_freq, am_mod_wave)

        filtered_signal = synth.filter(am_modulated_signal, filter_freq, filter_type, num_sounds)

        normalized_signal = filtered_signal / torch.abs(filtered_signal).max()

        # am_modulated_signal = synth.my_am_modulation_by_input_signal(mixed_signal, 1., 0., am_mod_freq, am_mod_wave, num_sounds=num_sounds)

        # mixed_signal = mixed_signal.cpu()


        # elapsed = time.time() - t
        # print("Signals are generated within (sec): ", elapsed)
        return normalized_signal

        # enveloped_signal = synth.adsr_envelope(filtered_signal,
        #                                        attack_t,
        #                                        decay_t,
        #                                        sustain_t,
        #                                        sustain_level,
        #                                        release_t,
        #                                        num_sounds)
        #
        # return enveloped_signal


class SynthOscOnly:
    """A synthesizer that produces a single sine oscillator.
        Args:
            self: Self object
            file_name: name for sound
            parameters_dict(optional): parameters for the synth components to generate specific sounds
            num_sounds: number of sounds to generate.
        """

    def __init__(self, file_name='unnamed_sound', parameters_dict=None, num_sounds=1):
        self.file_name = file_name
        self.params_dict = {}
        # init parameters_dict
        if parameters_dict is None:
            self.init_random_synth_params(num_sounds)
        elif type(parameters_dict) is dict:
            self.params_dict = parameters_dict.copy()
        else:
            ValueError("Provided parameters are not provided as dictionary")

        # generate signal with basic signal flow
        # self.signal = self.generate_signal(num_sounds)

    # def init_random_synth_params(self, num_sounds):
    #     """init params_dict with lists of parameters"""
    #
    #     self.params_dict['osc1_freq'] = random.choices(OSC_FREQ_LIST, k=num_sounds)
    #
    #     for key, val in self.params_dict.items():
    #         if isinstance(val, np.ndarray):
    #             self.params_dict[key] = val.tolist()
    #
    #     if num_sounds == 1:
    #         for key, value in self.params_dict.items():
    #             self.params_dict[key] = value[0]

    def generate_signal(self, num_sounds):
        osc_freq = self.params_dict['osc1_freq']
        osc_wave = self.params_dict['osc1_wave']

        filter_type = self.params_dict['filter_type']
        filter_freq = self.params_dict['filter_freq']

        attack_t = self.params_dict['attack_t']
        decay_t = self.params_dict['decay_t']
        sustain_t = self.params_dict['sustain_t']
        release_t = self.params_dict['release_t']
        sustain_level = self.params_dict['sustain_level']

        synthesizer = SynthModules(num_sounds)

        osc = synthesizer.oscillator(amp=1,
                                     freq=osc_freq,
                                     phase=0,
                                     waveform=osc_wave,
                                     num_sounds=num_sounds)

        filtered_signal = synthesizer.filter(osc, filter_freq, filter_type, num_sounds)

        enveloped_signal = synthesizer.adsr_envelope(filtered_signal,
                                               attack_t,
                                               decay_t,
                                               sustain_t,
                                               sustain_level,
                                               release_t,
                                               num_sounds)

        return enveloped_signal
        # return osc


def remove_duplicates(d):
    df = pd.DataFrame.from_dict(d)
    df_clean = df.drop_duplicates()
    d_out = df_clean.to_dict()
    for p in d_out:
        d_out[p] = list(d_out[p].values())
    return d_out

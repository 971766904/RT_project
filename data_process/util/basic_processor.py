import math
from jddb.file_repo import FileRepo
from jddb.processor import Signal, Shot, ShotSet, BaseProcessor
from typing import Optional, List
from scipy.fftpack import fft
from copy import deepcopy
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
from scipy import signal as sig
from copy import deepcopy
import json

class ResampleProcessor(BaseProcessor):
    def __init__(self, sample_rate):
        super().__init__()
        self._sample_rate = sample_rate

    def transform(self, signal: Signal):
        new_signal = deepcopy(signal)

        start_time = signal.time[0]
        down_time = start_time + len(new_signal.data) / new_signal.attributes['SampleRate']
        down_time = round(down_time, 3)

        f = interp1d(np.linspace(start_time, down_time, len(new_signal.data)), new_signal.data)
        new_time = np.linspace(start_time, down_time,
                               int(math.ceil(down_time - start_time) * self._sample_rate)+1)
        new_signal.raw = f(new_time)
        new_signal.attributes['SampleRate'] = self._sample_rate
        return new_signal

class SliceProcessor(BaseProcessor):
    """
            input the point number of the window  and overlap rate of the given window ,
        then the sample rate is recalculated,  return a signal of time window sequence
    """

    def __init__(self, window_length: int, overlap: float):
        super().__init__()
        assert (0 <= overlap <= 1), "Overlap is not between 0 and 1."
        self.params.update({"WindowLength": window_length,
                            "Overlap": overlap})

    def transform(self, signal: Signal) -> Signal:
        window_length = self.params["WindowLength"]
        overlap = self.params["Overlap"]
        new_signal = deepcopy(signal)
        raw_sample_rate = new_signal.attributes["SampleRate"]
        step = round(window_length * (1 - overlap))

        down_time = new_signal.time[-1]

        down_time = round(down_time, 3)

        idx = len(signal.data)
        window = list()
        while (idx - window_length) >= 0:
            window.append(new_signal.data[idx - window_length:idx])
            idx -= step
        window.reverse()
        new_signal.attributes['SampleRate'] = raw_sample_rate * len(window) / (len(new_signal.data) - window_length + 1)
        new_signal.data = np.array(window)
        new_start_time = down_time - len(window) / new_signal.attributes['SampleRate']
        new_signal.attributes['StartTime'] = round(new_start_time, 3)
        new_signal.attributes['OriginalSampleRate'] = raw_sample_rate
        return new_signal


class FFTProcessor(BaseProcessor):
    """
        processing signal by Fast Fourier Transform , return the maximum amplitude and the corresponding frequency
    """

    def __init__(self):
        super().__init__()

        self.amp_signal = None
        self.signal_rate = None
        self.fre_signal = None

    def transform(self, signal: Signal):

        self.amp_signal = deepcopy(signal)
        self.signal_rate = signal.attributes['OriginalSampleRate']
        self.fre_signal = deepcopy(signal)
        self.fft()
        self.amp_max()

        return self.amp_signal, self.fre_signal

    def fft(self):
        if self.amp_signal.data.ndim == 1:
            N = len(self.amp_signal.data)
            fft_y = fft(self.amp_signal.data)
            abs_y = np.abs(fft_y)
            normed_abs_y = abs_y / (N / 2)
            self.amp_signal.data = normed_abs_y[:int(N / 2)]
        elif self.amp_signal.data.ndim == 2:
            N = self.amp_signal.data.shape[1]
            R = self.amp_signal.data.shape[0]
            raw_cover = np.empty(shape=[0, int(N / 2)], dtype=float)
            for i in range(R):
                fft_y = fft(self.amp_signal.data[i])
                abs_y = np.abs(fft_y)
                normed_abs_y = abs_y / (N / 2)
                raw_cover = np.append(raw_cover, [normed_abs_y[:int(N / 2)]], axis=0)
            self.amp_signal.data = raw_cover

    def amp_max(self):
        fs = self.signal_rate
        raw = self.amp_signal.data
        amp_cover = np.empty(shape=0, dtype=float)
        fre_cover = np.empty(shape=0, dtype=float)
        N = (raw.shape[1]) * 2
        f = (np.linspace(start=0, stop=N - 1, num=N) / N) * fs
        f = f[:int(N / 2)]
        for j in range(raw.shape[0]):
            list_max = (raw[j, :]).tolist()
            raw_max = max(list_max)
            max_index = list_max.index(max(list_max))
            f_rawmax = f[max_index]
            amp_cover = np.append(amp_cover, raw_max)
            fre_cover = np.append(fre_cover, f_rawmax)
        self.amp_signal.data = amp_cover
        self.fre_signal.data = fre_cover


class Mean(BaseProcessor):
    """
         Given a set of input signals, average each instant
    """

    def __init__(self):
        super().__init__()

    def transform(self, *signal: Signal) -> Signal:
        new_signal = Signal(np.row_stack([sign.data for sign in signal.__iter__()]).T, signal.__getitem__(0).attributes)
        new_signal.data = np.mean(np.array(new_signal.data, dtype=np.float32), axis=1)
        return new_signal


class Concatenate(BaseProcessor):
    """
        calculate the mean and standard deviation of the given signal
    """

    def __init__(self):
        super().__init__()

    def transform(self, *signal: Signal) -> Signal:
        new_signal = Signal(np.concatenate([sign.data for sign in signal.__iter__()], axis=0),
                            signal.__getitem__(0).attributes)

        return new_signal


class AlarmTag(BaseProcessor):
    """
            Give arbitrary signals, extract downtime, timeline,
        and generate actual warning time labels

    """

    def __init__(self, lead_time, disruption_label: str, downtime_label: str):
        super().__init__()
        self.lead_time = lead_time
        self._disruption_label = disruption_label
        self._downtime_label = downtime_label

    def transform(self, signal: Signal):
        copy_signal = deepcopy(signal)
        fs = copy_signal.attributes['SampleRate']
        start_time = copy_signal.attributes['StartTime']

        if self.params[self._disruption_label] == 1:
            undisrupt_number = int(fs * (self.params[self._downtime_label] - self.lead_time - start_time))
        else:
            undisrupt_number = len(copy_signal.data)
        if undisrupt_number < len(copy_signal.data):
            # new_data = np.zeros(shape=undisrupt_number, dtype=int)
            new_data = np.zeros(shape=1, dtype=int)
            for i in range(len(copy_signal.data) - 1):
                if i <= undisrupt_number - 1:
                    new_data = np.append(new_data, np.array(0))
                else:
                    new_data = np.append(new_data, np.array(1))
        else:
            new_data = np.zeros(shape=len(copy_signal.data), dtype=int)

        new_signal = Signal(data=new_data, attributes=dict())
        new_signal.attributes['SampleRate'] = fs
        new_signal.attributes['StartTime'] = start_time

        return new_signal


class StackProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()

    def transform(self, *signal: Signal) -> Signal:
        return Signal(np.row_stack([sig.data for sig in signal.__iter__()]).T, signal.__getitem__(0).attributes)


class CutProcessor(BaseProcessor):
    def __init__(self, pre_time, is_test=False):
        super().__init__()
        self.params.update({"PreTime": pre_time,
                            "IsTest": is_test})

    def transform(self, signal: Signal) -> Signal:
        pre_time = self.params["PreTime"]
        is_test = self.params["IsTest"]
        is_disrupt = self.params["IsDisrupt"]
        new_signal = deepcopy(signal)

        if is_disrupt:
            if not is_test:
                new_signal.data = new_signal.data[-pre_time:]
        sliced_signal = new_signal
        sliced_signal.attributes['PreTime'] = pre_time
        return sliced_signal


class BinaryLabelProcessor(BaseProcessor):
    def __init__(self, is_test: bool = False):
        super().__init__()
        self.params.update({"IsTest": is_test})

    def transform(self, signal: Signal) -> Signal:
        disruptive = self.params["IsDisrupt"]
        is_test = self.params["IsTest"]
        prop_dict = signal.attributes
        pre_time = signal.attributes['PreTime']
        if disruptive:
            if is_test:
                if len(signal.data) < pre_time:
                    label = Signal(attributes=prop_dict, data=np.ones(len(signal.data)))
                else:
                    label = Signal(attributes=prop_dict,
                                   data=np.concatenate((np.zeros(len(signal.data) - pre_time), np.ones(pre_time))))
            else:
                label = Signal(attributes=prop_dict, data=np.ones(len(signal.data)))
        else:
            label = Signal(attributes=prop_dict, data=np.zeros(len(signal.data)))
        return label


class RadiatedFraction(BaseProcessor):
    """
            Given the radiated power signal and input power
            signal to calculate the radiated fraction.

    """

    def __init__(self, ):
        super().__init__()

    def transform(self, radiated_power_signal: Signal, input_power_signal: Signal) -> Signal:
        """

        :param radiated_power_signal:
        :param input_power_signal:
        :return:
        """
        resampled_attributes = deepcopy(input_power_signal.attributes)
        new_data = radiated_power_signal.data / input_power_signal.data

        return Signal(data=new_data, attributes=resampled_attributes)


def find_tags(prefix, all_tags):
    """
        find tags that start with the prefix
    param:
        prefix: The first few strings of the tags users need to look for
        all_tags: a list of all the tags that needed to be filtered
    :return: matching tags as a list[sting]
    """
    return list(filter(lambda tag: tag.encode("utf-8").decode("utf-8", "ignore")[0:len(prefix)] == prefix, all_tags))


def get_machine_tags(machine_name, name_list, df_signal):
    """
    get machine tags' name by common name list name_list from the signal dataframe df_signal
    :param machine_name: The machine name like 'J-TEXT'
    :param name_list: common name list for 3 machine
    :param df_signal: signal data frame which contain machine tags' name
    :return: target tags list of machine
    """
    machine_tags = []
    for signal_name in name_list:
        target_row = df_signal.loc[df_signal.Diagnostics == signal_name]
        signal_tag = target_row['{} MDSplus Tag'.format(machine_name)].values[0]
        machine_tags.append(signal_tag)
    return machine_tags


def read_config(file_name: str):
    """"read config files"""
    with open(file_name, 'r', encoding='UTF-8') as f:
        config = json.load(f)
    return config

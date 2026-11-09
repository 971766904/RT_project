from jddb.processor import BaseProcessor, Signal
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from copy import deepcopy

class SliceProcessor(BaseProcessor):
    def __init__(self, window_length: int, overlap: float):
        super().__init__()
        assert 0 <= overlap <= 1, "Expected overlap between 0 and 1, got {}".format(overlap)
        self.params.update({
            "WindowLength": window_length,
            "Overlap": overlap
        })

    def transform(self, signal: Signal) -> Signal:
        window_length = self.params['WindowLength']
        overlap = self.params['Overlap']
        step = int(round(window_length * (1 - overlap)))
        sliced_data = sliding_window_view(signal.data, window_length, axis=0)[::step]
        sliced_sample_rate = signal.attributes['SampleRate'] * len(sliced_data) / (
                len(signal.data) // step * step - window_length + step)
        sliced_start_time = round((signal.attributes['StartTime'] + window_length / signal.attributes[
            'SampleRate']) * sliced_sample_rate) / sliced_sample_rate
        sliced_prop_dict = deepcopy(signal.attributes)
        sliced_prop_dict['SampleRate'] = sliced_sample_rate
        sliced_prop_dict['StartTime'] = sliced_start_time

        return Signal(data=sliced_data, attributes=sliced_prop_dict)

class StackProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()

    def transform(self, *signal: Signal) -> Signal:
        return Signal(np.row_stack([sig.data for sig in signal.__iter__()]).T, signal.__getitem__(0).attributes)

class PadProcessor(BaseProcessor):
    def __init__(self, total_length: int):
        super().__init__()
        self.params.update({"TotalLength": total_length})

    def transform(self, signal: Signal) -> Signal:
        total_length = self.params["TotalLength"]
        new_signal = deepcopy(signal)
        new_signal.data = np.pad(new_signal.data, ((0, 0), (0, total_length - new_signal.data.shape[-1])),
                                 mode='constant', constant_values=(0, 0))
        return new_signal

class ClassificationLabelProcessor(BaseProcessor):
    def __init__(self, pre_time, vague_time):
        super().__init__()
        self.pre_time = pre_time
        self.vague_time = vague_time

    def transform(self, signal: Signal) -> Signal:
        disruptive = self.params["IsDisrupt"]
        attributes = signal.attributes
        sig_len = len(signal.data)
        if disruptive:
            if sig_len < self.pre_time+self.vague_time:
                nan = np.full(sig_len-self.pre_time, np.nan)
                ones = np.ones(self.pre_time)
                label = Signal(attributes=attributes,
                               data=np.concatenate((nan, ones)))
            else:
                zeros = np.zeros(sig_len - self.pre_time - self.vague_time)
                nan = np.empty(self.vague_time)
                nan[:] = np.nan
                ones = np.ones(self.pre_time)
                label = Signal(attributes=attributes,
                               data=np.concatenate((zeros, nan, ones)))
        else:
            label = Signal(attributes=attributes,
                           data=np.zeros(sig_len))
        return label


class DomainLabelProcessor(BaseProcessor):
    def __init__(self, domain_label):
        super().__init__()
        self.domain_label = domain_label

    def transform(self, signal: Signal) -> Signal:
        return Signal(attributes=signal.attributes,
                      data=np.full(len(signal.data), self.domain_label, dtype=int))

class ConvStackProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()

    def transform(self, *signal: Signal) -> Signal:
        attributes = signal[0].attributes
        stacked_data = np.stack([sig.data for sig in signal.__iter__()]).transpose((1, 2, 0))
        return Signal(data=stacked_data, attributes=attributes)
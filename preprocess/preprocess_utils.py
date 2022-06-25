import numpy as np
import pretty_midi

# 4분음표 단위인지 확인
def check_time_sign(pm, num=4, denom=4):

    time_sign_list = pm.time_signature_changes

    if len(time_sign_list) == 0:
        return False

    for time_sign in time_sign_list:
        if time_sign.numerator != num or time_sign.denominator != denom:
            return False

    return True


# sampling rate 변환하기
def change_fs(beats, target_beats=16):

    quarter_length = beats[1] - beats[0]
    changed_length = quarter_length / (target_beats/4)
    changed_fs = 1 / changed_length

    return changed_fs


def get_comp():

    standard = {35: 'kick', 38: 'snare',
                46: 'open hi-hat', 42: 'closed hi-hat',
                50: 'high tom', 48: 'mid tom', 45: 'low tom',
                49: 'crash', 51: 'ride'}

    encoded = {'kick': 0, 'snare': 1,
               'open hi-hat': 2, 'closed hi-hat': 3,
               'high tom': 4, 'mid tom': 5, 'low tom': 6,
               'crash': 7, 'ride': 8}

    return standard, encoded


# 드럼 구성 요소 매핑
def map_unique_drum(note):   # pm.instruments[drum][kick, snare, hi-hat...]

    pitch = note.pitch
    standard, encoded = get_comp()

    map_to_standard = {36: 35,
                       37: 38, 39: 38, 40: 38,
                       44: 42,
                       41: 45, 43: 45,
                       47: 48,
                       55: 49, 57: 49,
                       59: 51}

    if pitch not in standard.keys():
        if pitch in map_to_standard.keys():
            note.pitch = map_to_standard[pitch]
        else:
            return False   # 매핑되는 드럼 요소가 없을 경우

    return True


# 퀀타이즈 -> 비트 간 부정확한 비율 처리
def quantize_drum(inst, fs, start_time, comp=9):

    # inst: pm.instruments[drum]
    # fs: sampling rate
    # start_time: pm.get_onsets()[0]

    fs_time = 1 / fs
    end_time = inst.get_end_time()

    standard, encoded = get_comp()

    quantize_time = np.arange(start_time, end_time+fs_time, fs_time)
    drum_roll = np.zeros((quantize_time.shape[0], comp))

    for i, note in enumerate(inst.notes):

        if map_unique_drum(note) == False:
            continue

        start_index = np.argmin(np.abs(quantize_time - note.start))
        end_index = np.argmin(np.abs(quantize_time - note.end))

        if start_index == end_index:
            end_index += 1

        range_index = np.arange(start_index, end_index)
        inst_index = encoded[standard[note.pitch]]

        for index in range_index:
            drum_roll[index, inst_index] = 1

    return drum_roll


# (seq, feat) -> (batch, window, feat)
def windowing(roll, window_size=64, bar=16, cut_ratio=0.9):

    # roll: (seq, feat)
    # bar: 마디 내 음표 단위

    new_roll = []
    num_windows = roll.shape[0] // window_size
    do_nothing = (np.sum((roll == 0), axis=1) == roll.shape[1])

    for i in range(0, num_windows):
        break_flag = False
        start_index = window_size * i
        end_index = window_size * (i + 1)

        check_vacant = do_nothing[start_index:end_index]
        for j in range(0, window_size, bar):
            if np.sum(check_vacant[j:j + bar]) > (bar * cut_ratio):     # 마디 안 90%가 비어있으면 패스
                break_flag = True
                break

        if break_flag: continue
        new_roll.append(np.expand_dims(roll[start_index:end_index], axis=0))

    return np.vstack(new_roll)


# 2진수 -> 10진수
def bin_to_dec(array):

    decimal = 0
    length = array.shape[0]

    for i, elem in enumerate(array):
        decimal += (np.power(2, length-i-1) * elem)

    return int(decimal)


# 원핫인코딩
def hot_encoding(roll):   # (batch, seq, feat)

    last_axis = len(roll.shape) - 1
    I = np.eye(np.power(2, roll.shape[-1]), dtype='bool')
    dec_index = np.apply_along_axis(bin_to_dec, last_axis, roll)

    return I[dec_index]


# 음악 재생
def drum_play(array, fs, comp=9):

    fs_time = 1 / fs
    standard, encoded = get_comp()
    reverse_standard = {v: k for k, v in standard.items()}
    reverse_encoded = {v: k for k, v in encoded.items()}

    decimal_idx = np.where(array == 1)[1]
    binary_idx = list(map(lambda x: np.binary_repr(x, comp), decimal_idx))

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=32, is_drum=True)
    pm.instruments.append(inst)

    for i, inst_in_click in enumerate(binary_idx):
        start_time = fs_time * i
        end_time = fs_time * (i + 1)

        for j in range(0, len(inst_in_click)):
            if inst_in_click[j] == '1':
                pitch = reverse_standard[reverse_encoded[j]]
                inst.notes.append(pretty_midi.Note(80, pitch, start_time, end_time))

    return pm


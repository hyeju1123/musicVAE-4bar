import pickle
import sparse
import pretty_midi
from glob import glob
from preprocess.preprocess_utils import *


data = []
count = 0
data_path = '/content/drive/MyDrive/musicvae'
save_path = './preprocessed-data.pkl'

for folder_path in glob(data_path+'*'):
    for file_path in glob(folder_path+'/*'):
        try:
            pm = pretty_midi.PrettyMIDI(file_path)

            if not check_time_sign(pm, num=4, denom=4):
                continue

            start_time = pm.get_onsets()[0]
            beats = pm.get_beats(start_time)
            tempo = pm.estimate_tempo()
            fs = change_fs(beats)

            for inst in pm.instruments:
                if inst.is_drum == True:
                    drum_roll = quantize_drum(inst, fs, start_time)
                    drum_roll = windowing(drum_roll)
                    drum_roll = hot_encoding(drum_roll)

                    for i in range(0, drum_roll.shape[0]):
                        data.append(sparse.COO(drum_roll[i]))
        except:
            continue

        count += 1
        if count % 100 == 0:
            print('Files iterations %d ' % count)


with open(save_path, 'wb') as f:
    print('File saved')
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

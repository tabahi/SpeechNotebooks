import numpy as np
#import scipy.io.wavfile
import os

import modules.SER_DB as SER_DB
import modules.Mel_spectrum as mel



# define name and directory addresses of databases:
DBs = [{'DB': "IEMOCAP", 'path' : "C:\\DB\\IEMOCAP_noVideo\\"},
        {'DB': "MSPIMPROV", 'path' : "C:\\DB\\MSP-IMPROV\\"},
        {'DB': "RAVDESS", 'path' : "C:\\DB\\RAVDESS_COPY2\\"}]




def example_mel_spectrum(wav_file_path):
    
    # Get mel spectrum of the first file in database
    mel_spec, duration = mel.Extract_mel_spec(wav_file_path, window_length=0.025, window_step=0.010, num_filt=128, amp_DB=1, emphasize_ratio=0.95, norm=0, f0_min=30, f0_max=4000, trim=1, fixed_size=200)

    print("Clip mel size: ", mel_spec.shape, wav_file_path)
    mel.plot_mel_spectrum(mel_spec[0])






def import_features_n_labels(list_of_clips):
    print("Creating mel spectrums")
    features_list = []
    labels_list = []

    for clip in list_of_clips:
        
        mel_spec, duration = mel.Extract_mel_spec(clip.filepath, window_length=0.025, window_step=0.010, num_filt=64, amp_DB=1, emphasize_ratio=0.95, norm=1, f0_min=30, f0_max=4000, trim=1, fixed_size=200)
        
        for s in range(mel_spec.shape[0]):
            features_list.append(mel_spec[s])
            labels_list.append(clip.speaker_id) # you can also use clip.emotion_cat or clip.sex etc as labels


    labels_list = np.array(labels_list)
    
    #lb = LabelEncoder()
    #class_labels = np_utils.to_categorical(lb.fit_transform(labels_list))

    return features_list, labels_list




def main():

    print("Start")

    # Read the database directory to get clip addresses and labels
    list_of_clips = SER_DB.create_DB_file_objects(DBs[2]['DB'], DBs[2]['path'], deselect=['F', 'D', 'U', 'C', 'R', 'E', 'X'])

    #example_mel_spectrum(list_of_clips[0].filepath) #show the mel-spectrum of file at index 0

    for clip in list_of_clips:
        mel_spec, duration = mel.Extract_mel_spec(clip.filepath, window_length=0.025, window_step=0.010, num_filt=64, amp_DB=1, emphasize_ratio=0.95, norm=0, f0_min=30, f0_max=4000, trim=1, fixed_size=400)

        filename = os.path.basename(clip.filepath)

        # if audio has more than 200 frames, it is divided into multiple segments s
        for s in range(mel_spec.shape[0]):
            mel.save_mel_spectrum(mel_spec[s], 'RAVDESS_spectra/' + filename + '~' + str(s) + '.png', 3, 2)
            

    #features, labels = import_features_n_labels(list_of_clips)
    # features is a list of 2D arrays, list of mel-spectrums
    # labels is a 1D array of labels of all clips


    print("End")



if __name__ == '__main__':
    main()
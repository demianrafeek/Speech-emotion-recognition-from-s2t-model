
import torch
from torch import torchaudio


class prepare_dataset():

    def __init__(self, data_path: str):
        
        self.model_path = data_path
 
        

    def speech_file_to_array_fn(path):
        speech_array, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
        speech = resampler(speech_array).squeeze().numpy()
        return speech



    def preprocess_function(examples):
        input_column = "name"
        output_column = "emotion"
        speech_list = [ speech_file_to_array_fn(path) for path in examples[input_column]]
        target_list = [ label_to_id(label, label_list) for label in examples[output_column]]

        result = processor(speech_list, sampling_rate=target_sampling_rate)
        result["labels"] = list(target_list)

        return result
    
        
    def detect_labels(time,sub_df):
    # times between which to extract the wave from
    duration = sub_df.iloc[-1,2]
    start = 0.0 # seconds
    end = time # seconds
    
    emotion_list = []
    idx=0
    start_time = sub_df['start_time'][idx]
    end_time = sub_df['end_time'][idx]
    
    while end < duration:
        if end>end_time and idx<sub_df.shape[0]-1:
        idx+=1
        start_time = sub_df['start_time'][idx]
        end_time = sub_df['end_time'][idx]

        if start>=start_time and end<=end_time:
        emotion_list.append(sub_df['main_emotion'][idx])
        else:
        emotion_list.append(8) # other 
        
        start = end
        end += time
    print(len(emotion_list),duration)
    return (''.join(map(str,emotion_list)))


    for i,path in enumerate(Iemocap_df['name']):
    record_name = path.split(sep='\\')[-1]
    sub_records = df[df['titre'].str.contains(record_name[:-4])]
    # print(sub_records.iloc[:,:]) # record duration
    sub_records = sub_records.sort_values('start_time',ignore_index=True)
    Iemocap_df['main_emotion'][i] = detect_labels(emo_every_ms,sub_records)
    # print(sub_records.iloc[-1,2],len(Iemocap_df['main_emotion'][i]))



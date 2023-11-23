from huggingsound import SpeechRecognitionModel
import torch 

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SpeechRecognitionModel("output/", device=device)

data = pd.read_csv(path+'/iemocap_40s_data_with_dublicats.csv')
data_path = path+'iemocap_40s_waves/'
data = data.iloc[int(len(data)*0.9):,:]

test_data = [data_path+name for name in data['name']]

transcriptions = model.transcribe(test_data)
print(transcriptions)




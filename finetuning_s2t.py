
from huggingsound import TrainingArguments, ModelArguments, SpeechRecognitionModel, TokenSet

model = SpeechRecognitionModel("facebook/wav2vec2-large-xlsr-53")
output_dir = "G:\Arete\projects\SER_project\\finetune_s2t_model\\output"

# first of all, you need to define your model's token set
# however, the token set is only needed for non-finetuned models
# if you pass a new token set for an already finetuned model, it'll be ignored during training

# # finetune the model with the iemocap dataset 
tokens = ['n', 'f', 'a', 's', 'h', 'e', 'z', 'r', 'o', 'd']
token_set = TokenSet(tokens)

# # define your custom train data
# data = pd.read_csv('/media/ires/DATA/Speech_task RA/archive/Speech-emotion-recognition-from-s2t-model/iemocap_short_waves_data.csv')
# path = '/media/ires/DATA/Speech_task RA/archive/Iemocap_audio/iemocap_audio/IEMOCAP_wav/'
data = pd.read_csv('/iemocap_40s_data_with_dublicats.csv')
path = '/media/ires/DATA/Speech_task RA/archive/Speech-emotion-recognition-from-s2t-model/iemocap_40s_waves/'
data = [{'path':path+name, 'transcription':script} for name,script in zip(data['name'],data['emo_script'])]
train_data = data[:int(len(data)*0.8)]
eval_data  = data[int(len(data)*0.8):int(len(data)*0.9)]

# and finally, fine-tune your model
model.finetune(
    output_dir, 
    train_data=train_data,
    eval_data=eval_data
    token_set=token_set,
)


######################################################





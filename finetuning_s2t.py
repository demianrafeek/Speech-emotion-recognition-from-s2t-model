
from huggingsound import TrainingArguments, ModelArguments, SpeechRecognitionModel, TokenSet

model = SpeechRecognitionModel("facebook/wav2vec2-large-xlsr-53")
output_dir = "G:\Arete\projects\SER_project\\finetune_s2t_model\\output"

# first of all, you need to define your model's token set
# however, the token set is only needed for non-finetuned models
# if you pass a new token set for an already finetuned model, it'll be ignored during training

tokens = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]#, "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]
# tokens = ["ا", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ","ر", "ز", "س", "ش", "ص", "ض", "ط", "ظ", "ع", "غ", "ف", "ق", "ك", "ل", "م", "ن", "ه", "و", "ي", "'"]
token_set = TokenSet(tokens)

# define your custom train data
train_data = [
    {"path": "G:\Arete\projects\SER_project\\finetune_s2t_model\\en_test1.wav", "transcription": "say the word bar"},
    {"path": "G:\Arete\projects\SER_project\\finetune_s2t_model\\en_test2.wav", "transcription": "say the word date"},
]

# train_data = [
#     {"path": "G:\Arete\projects\SER_project\\finetune_s2t_model\\ar_test1.wav", "transcription": " لا تفكر في ابسط ما يمكنك فعله بل فكر في اكبر ما يمكنك فعله"},
#     {"path": "G:\Arete\projects\SER_project\\finetune_s2t_model\\ar_test2.wav", "transcription": "اننا نتحدث بنفس الوتيره اللي نفكر ونفهم بها الامور"},
# ]

# and finally, fine-tune your model
model.finetune(
    output_dir, 
    train_data=train_data,
    token_set=token_set,
)


######################################################





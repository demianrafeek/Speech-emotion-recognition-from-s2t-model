from huggingsound import SpeechRecognitionModel
import torch 

device = "cuda" if torch.cuda.is_available() else "cpu"
# model = SpeechRecognitionModel("G:\Arete\projects\SER_project\\finetune_s2t_model\output\\pytorch_model.bin")
model = SpeechRecognitionModel("G:\Arete\projects\SER_project\\finetune_s2t_model\\output\\", device=device)

audio_paths = ["G:\Arete\projects\SER_project\\finetune_s2t_model\\en_test2.wav"]#, "G:\Arete\projects\SER_project\\finetune_s2t_model\\ar_test1.wav"]

transcriptions = model.transcribe(audio_paths)

print(transcriptions)




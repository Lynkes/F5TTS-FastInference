Modified F5TTS to dont wait for all inference to finish and merge into a single wave file and just play as soon as it generates. 

Install 
python.exe -m venv venv 
.\venv\Scripts\activate

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

!pip install mkv -q
!pip install -q torchaudio omegaconf
!pip install pydub -q
!pip install num2words -q
!pip install git+https://github.com/huggingface/transformers
!pip install sentencepiece -q
!pip install py7zr
!pip install scenedetect[opencv] --upgrade

!pip install openvino-dev[torch, onnx]




!rm -f '/usr/local/bin/mkvpropedit'
!rm -f '/usr/local/bin/mkvmerge'
!rm -f '/usr/local/bin/mkvextract'
!rm -f '/usr/local/bin/mkvinfo'

!sudo curl -L 'https://mkvtoolnix.download/appimage/MKVToolNix_GUI-75.0.0-x86_64.AppImage' -o /usr/local/bin/MKVToolNix_GUI.AppImage
!sudo chmod u+rx /usr/local/bin/MKVToolNix_GUI.AppImage
!sudo ln -s /usr/local/bin/MKVToolNix_GUI.AppImage /usr/local/bin/mkvpropedit
!sudo chmod a+rx /usr/local/bin/mkvpropedit
!sudo ln -s /usr/local/bin/MKVToolNix_GUI.AppImage /usr/local/bin/mkvmerge
!sudo chmod a+rx /usr/local/bin/mkvmerge
!sudo ln -s /usr/local/bin/MKVToolNix_GUI.AppImage /usr/local/bin/mkvextract
!sudo chmod a+rx /usr/local/bin/mkvextract
!sudo ln -s /usr/local/bin/MKVToolNix_GUI.AppImage /usr/local/bin/mkvinfo
!sudo chmod a+rx /usr/local/bin/mkvinfo
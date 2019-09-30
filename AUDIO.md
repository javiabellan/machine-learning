<h1 align="center">Audio</h1>

## Audio recognition 🎤
- [Audio classification with fast.ai](https://towardsdatascience.com/audio-classification-using-fastai-and-on-the-fly-frequency-transforms-4dbe1b540f89) (CNN)
- Speech recognition
  - **CTC**
  - **RNN-T**: RNN-Transducer, 2012 [paper](https://arxiv.org/abs/1211.3711)
  - **LAS**:  Listen, Attend and Spell. Attention-based, sequence-to-sequence model. 2015 [paper](https://arxiv.org/abs/1508.01211)

## Audio generation 🔊
- [Speech generation guide](https://www.kdnuggets.com/2019/09/2019-guide-speech-synthesis-deep-learning.html)
- Music generation
- WaveNet: A Generative Model for Raw Audio.

---


## Packages
- PyAudio
- [gTTS](https://github.com/pndurette/gTTS): Ptyoen interface of Google text to speech
- [SpeechRecognition](https://github.com/Uberi/speech_recognition)
- playsound

---

## CTC speech recognition

uno puede "samplear" la entrada en trozos de longitud fija, pero la salida esperada, una serie de fonemas o de letras, lleva un ritmo distinto.
CTC lo que hace es que la salida de la red (típicamente LTSM) sea "si hay cambio de fonema, y con qué probabilidad para cada posible opción", y luego esa señal se procesa, con o sin ayuda de un modelo del idioma del que se trate, para generar la secuencia más probable de fonemas o de palabras (el truco esta en como entrenar la red para que la salida sea esa). Antes de CTC lo tradicional era poner un HMM a la salida de la red para hacer básicamente lo mismo, pero bastante más complicado de entrenar, y sin salida probabilística.


## LAS speech recognition
LAS (listen, attend, spell) es otro mecanismo para hacer lo mismo, combinando un lstm, un modelo de atención y un diccionario del idioma en cuestión en una sola entidad que se entrena end to end. De hecho creo que hay más evoluciones de la misma idea.



# Reti Neurali Convoluzionali per riconoscimento di retinopatia diabetica all'interno di immagini di fundus oculi

Il seguente codice implementa modelli di reti convoluzionali per il riconoscimento di un grado di retinopatia diabetica all'interno di immagini appartenenti al seguente dataset disponibile sul portale Kaggle: 

https://www.kaggle.com/tanlikesmath/diabetic-retinopathy-resized

## 1. Introduzione

La versione di linguaggio python utilizzata è la 3.x (In particolare la 3.6). Le librerie utilizzati sono:

1) tensorflow==1.7.0
2) numpy==1.17.3
3) matplotlib==2.1.2
4) Keras_Preprocessing==1.1.0
5) pandas==0.22.0
6) opencv_contrib_python==3.4.0.12
7) Keras==2.2.4
8) scikit_learn==0.22

L'ambiente di sviluppo è stato prevalentemente in OS Unix (Ubuntu 16 e MacOs Mojave)

## 2. Configurazione

Per avviare una pipeline è necessario configurare il file `config.json`:

```json
{
	"dataset_dir": "/Users/loretto/Downloads/diabetic-retinopathy-resized/resized_train/resized_train",
	"labels_path": "/Users/loretto/Downloads/diabetic-retinopathy-resized/trainLabels_cropped.csv",
	"output_dir": "/Users/loretto/Desktop/output",
	"model": "standard",
	"epoch": 10,
	"size": 256,
	"batch": 32,
	"val_ratio": 20,
	"kfold": false,
	"fold": 3
}
```

I parametri configurabili sono:

1) **dataset_dir**: directory che punta al dataset 
2) **labels_path**: directory che punta al file csv delle label 
3) **output_dir**: è una directory che raccoglie in output i risultati dell'apprendimento, ossia history di apprendimento, matrici di confusioni, coefficienti di coehn...
4) **model**: è il modello di rete. Possibili valori "standard" e "fine"
5) **epoch**: numero di epoche di training 
6) **size**: dimensione di ridimensionamento delle immagini
7) **batch**: dimensione del batch di training
8) **val_ratio**: percentuale proporzione validation set rispetto al training set
9) **kfold**: se eseguire con cross validation
10) **fold**: numero fold per cross validation



Il file config presenta una configurazione di default. Per eseguire la pipeline di apprendimento: 


1) `python organizer.py`
2) `python augmentation.py` (opzionale)
3) `python kfold.py` (opzionale)
4) `python run.py`

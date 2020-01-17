# Reti Neurali Convoluzionali per riconoscimento di retinopatia diabetica all'interno di immagini di fundus oculi

Il seguente codice implementa modelli di reti convoluzionali per il riconoscimento di un grado di retinopatia diabetica all'interno di immagini appartenenti al seguente dataset disponibile sul portale Kaggle, attraverso tecniche di data augmentation e cross validazione: 

https://www.kaggle.com/tanlikesmath/diabetic-retinopathy-resized

## 1. Modello di rete per classificazione (tramite classificazione o regressione)

### 1.1 Introduzione

La versione di linguaggio python utilizzata è la 3.x (In particolare la 3.6), su macchina Linux (ubuntu 16).
 GPU utilizzata è stata GTX 1050 Ti Nvidia.
Le librerie utilizzate sono:

1) tensorflow==1.7.0
2) numpy==1.17.3
3) matplotlib==2.1.2
4) Keras_Preprocessing==1.1.0
5) pandas==0.22.0
6) opencv_contrib_python==3.4.0.12
7) Keras==2.2.4
8) scikit_learn==0.22

L'ambiente di sviluppo è stato prevalentemente in OS Unix (Ubuntu 16 e MacOs Mojave)

### 1.2 Configurazione

Attraverso il file `config.json` è possibile configurare la pipeline di apprendimento.
 Il `config.json` fornisce già una configurazione di default:

```json
{
   "dataset_dir": "/Users/vannucchi/Downloads/diabetic-retinopathy-resized/resized_train_cropped/resized_train_cropped",
   "labels_path": "/Users/vannucchi/Downloads/diabetic-retinopathy-resized/trainLabels_cropped.csv",
   "output_dir": "/Users/vannucchi/Desktop/output",
   "model": "regression",
   "epoch": 20,
   "size": 256,
   "batch": 32,
   "dense_level": 512,
   "val_ratio": 20,
   "kfold": false,
   "fold": 3
} 
```
é necessario sostituire i valori dei parametri `dataset_dir`, `labels_path`, `output_dir` con percorsi opportuni. Di seguito sono riportate le descrizioni dei vari campi.
I parametri configurabili sono:

1) **dataset_dir**: directory che punta ai file del dataset fornito da kaggle (**sostituire** con percorso valido)
2) **labels_path**: directory che punta al file csv delle labels (**sostituire** con percorso valido)
3) **output_dir**: è una directory che raccoglie in output i risultati dell'apprendimento, ossia history di apprendimento, matrici di confusione, coefficienti di cohen...(**sostituire** con percorso valido)
4) **model**: è il modello di rete. Possibili valori `standard`, `fine`, `regression` .`standard` fornisce una implementazione di una rete neurali convoluzionale (file `cnn1.py`). Per modificare i layer del modello è necessario operare direttamente sul file. `fine` fornisce una implementazione tramite tecnica di fine tuning.  `regression` fornisce un modello di regressione ( `cnn3.py`)
5) **epoch**: numero di epoche di training 
6) **size**: dimensione di ridimensionamento delle immagini
7) **dense_level**: dimensione del livello fully connected
8) **batch**: dimensione del batch di training
9) **val_ratio**: percentuale proporzione validation set rispetto al training set
10) **kfold**: se eseguire con cross validation
11) **fold**: numero fold per cross validation

### 1.3 Esecuzione

Il file config presenta una configurazione di default. Per eseguire la pipeline di apprendimento, spostarsi tramite terminale nella cartella del progetto ed eseguire in sequenza: 


1) `python organizer.py` (**obbligatorio**, riorganizza il dataset per compatibilità di keras spostandolo in una cartella "dataset" all'interno della cartella di **output** indicata nel config.json)
2) `python augmentation.py` (opzionale, esegue un data augmentation sulle classi 1, 2, 3 per bilanciare la distribuzione dei samples. La classe 0 è la più numerosa. Tale operazione potrebbe impiegare parecchi minuti causa impiego di operazioni su disco). 
3) `python kfold.py` (opzionale, genera una cartella **kfold** con all'interno le varie fold per cross-validation, tale operazione potrebbe impiegare parecchi minuti causa utilizzo di operazioni su disco)
4) `python run.py` (**obbligatorio**, lancia un apprendimento direttamente sul dataset senza cross validazione)


N.B: per eseguire apprendimenti sulle k fold generate, è necessario il seguente comando:

`python run.py -f 2`

dove in questo caso **2** è l'indice della fold. Per ogni fold deve essere eseguito il comando se si vogliono ottenere risultati cross validati.
Per eseguire un apprendimento senza cross validation è sufficiente `python run.py`.


Tutti i risultati degli apprendimenti vengono salvati nella cartella **results**. Ogni risultato fornisce una immagine dei modello utilizzato, matrice di confusione sul test set, accuracy e loss function per training e validation set, alcuni meta parametri e coefficiente Kappa di Cohen per valutare la concordanza sul test set.

### 1.4 Checkpoint

La versione del modello  `regression` implementa un meccanismo di checkpoint per cui è la pipeline salva automaticamente il modelllo addestrato per riutilizzarlo in addestramenti successivi. Ad ogni lancio viene controllata la presenza o meno del modello. Se presente viene caricato e sostituito alla fine del nuovo addestramento.

# Generating Word and Document Embeddings for Sentiment Analysis
This repo contains the source code of the paper --

[Generating Word and Document Embeddings for Sentiment Analysis](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiK8bSAotDsAhXyh4sKHSrZB-EQFjABegQIARAC&url=https%3A%2F%2Fwww.cmpe.boun.edu.tr%2F~gungort%2Fpapers%2FGenerating%2520Word%2520and%2520Document%2520Embeddings%2520for%2520Sentiment%2520Analysis.pdf&usg=AOvVaw3KaXHe7J0caSq1QdTsUcVU).
Cem Rıfkı Aydın, Tunga Güngör, Ali Erkan. CICLing 2019

This approach attempts to model word and document embeddings for the binary sentiment classification task in Turkish and English. This study can be adapted to other languages with minor changes.

## Requirements

- Python 2.x
- Python 3.7 or a newer version of Python
- gensim
- nltk
- numpy
- pandas
- scikit-fuzzy
- scikit_learn
- scipy
- turkish.deasciifier

In this study, we have made use of the official Turkish dictionary (TDK). On account of copyright issues, I am not allowed to share the whole of it online. In order to get a copy of the full dictionary to employ it, please send a request e-mail to `cemrifkiaydin@gmail.com`. 
We also utilised both Python 2.x and Python 3.7 versions. The reason we depended on the Python 2.x version as well is that external tools (i.e. Haşim Sak's morphological parser and disambiguator tools) cannot work with Python 3. The ".so" file built by the Python 2.x version we leveraged could not be converted to Python 3. Also, the use of Python 3.7 or a newer version is required. The command python3 must refer to the use of Python 3.7 or a newer version, and python should refer to Python 2.x. In the input folder, you can change the file you want to train or evaluate your model on. The path to it can also be specified by specific commands explained below. We rely on the use of a .csv file, and the columns are categorised (header) as "Text" and "Sentiment". 
We also thank Cumali Türkmenoğlu for providing us with the Turkish movie dataset. In our study, we used only a subset of the whole corpus.

## Execution

Execute the file `runner.py` to train word and document embeddings and evaluate the model.
The following are the command-line arguments:
- `--command`: set command, which can be `cross_validate`, `train_and_test_separately`, or `predict`
- `--language`: set language, which can be `english` or `turkish`
- `--embedding_type`: set embedding type, which can be `corpus_svd`, `lexical_svd`, `supervised`, `ensemble`, `clustering`, or `word2vec`
- `--embedding_size`: set embedding size
- `--cv_number`: set number of folds for cross-validation
- `--use_3_review_polarities`: set whether or not you use three polarities on a review-basis
- `--file_path`: set file path for the `cross-validation` and `prediction` cases
- `--training_path`: set training file path in case the test dataset is also specified in the below command
- `--test_path`: set test file path in case the training dataset is also specified in the above command
- `--model_path`: set path to the model trained previously. Be careful in the sense that the same model parameters should be chosen for both the training and test sets

#### Setup with virtual environment (Python 3):
-  python3 -m venv my_venv
-  source my_venv/bin/activate

Install the requirements:
-  pip3 install -r requirements.txt

If everything works well, you can run the example usage given below.

### Example Usage:
- The following guide shows an example usage of the model in generating word and document vectors and performing the evaluation.
- Instructions
      
      1. Change directory to the location of the source code
      2. Run the instructions in "Setup with virtual environment (Python 3)"
      3. Run the runner.py file with chosen command parameters. Some examples are given below

Examples:
```
python3 runner.py --command cross_validate --language turkish --embedding_type corpus_svd --embedding_size 50 --file_path input/Sentiment_dataset_turk.csv
python3 runner.py --command predict --language turkish --embedding_type ensemble --file_path input/Sentiment_dataset_turk.csv
python3 runner.py --command train_and_test_separately --language turkish --embedding_type ensemble --embedding_size 200 --training_path input/Turkish_twitter_train.csv --test_path input/Turkish_twitter_test.csv
python3 runner.py --command cross_validate --language english --embedding_type supervised --file_path input/Sentiment_dataset_eng.csv
```
## Citation
If you find this code useful, please cite the following in your work:
```
@InProceedings{sent-emb:19,
  author = 	"Aydin, Cem Rifki
		and Gungor, Tunga
		and Erkan, Ali",
  title = 	"Generating Word and Document Embeddings for Sentiment Analysis",
  booktitle = 	"20th International Conference on Intelligent Text Processing and Computational Linguistics (CICLing 2019), Ed. A.Gelbukh",
  year = 	"2019",
  month =       "April",
  location = 	"La Rochelle, France"
}
```
## Credits
Codes were written by Cem Rıfkı Aydın

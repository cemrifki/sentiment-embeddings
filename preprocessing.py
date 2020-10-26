#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Cem Rıfkı Aydın
"""

import os
import re
import subprocess
from collections import Counter

import numpy as np
import pandas as pd
from turkish.deasciifier import Deasciifier

import constants

try:
    __instance = str
except:
    __instance = bytes


class UnicodeTr(__instance):
    """
    This class handles the case where the lower case of the Turkish letter I is defined as ı.
    Also, the upper case of the Turkish letter i is defined as İ unlike in English.
    """

    CHARMAP = {
        "to_upper": {
            u"ı": u"I",
            u"i": u"İ",

        },
        "to_lower": {
            u"I": u"ı",
            u"İ": u"i",
        }
    }

    def lower(self):
        """
        The lowercasing operation handling the special case for the Turkish letters "I" and "İ.

        :return: Lowercased string.
        :rtype: str
        """
        for key, value in self.CHARMAP.get("to_lower").items():
            self = self.replace(key, value)

        return self.lower()

    def upper(self):
        """
        The uppercasing operation handling the special case for the Turkish letters "I" and "İ.

        :return: Uppercased string.
        :rtype: str
        """
        for key, value in self.CHARMAP.get("to_upper").items():
            self = self.replace(key, value)

        return self.upper()


class Preprocessing:
    """
    This class performs preprocessing operations separately for both Turkish and English.
    """
    TURKISH_CHARS = "ğüışöçĞÜİŞÖÇîÎûÛâÂ"

    TURKISH_SUFFIX_IN_FULL_FORM = {"yon": "yorsun", "yom": "yorum", "cam": "cağım", "cem": "cağım",
                                   "çam": "cağım", "caz": "cağız", "çaz": "cağız", "cez": "cağız"}
    TURKISH_NEGATORS = {"değil", "yok"}

    def __init(self):
        pass

    def common_tokenize(self, string):
        """
        This tokeniser is common to all languages, including the Turkish and English languages.

        :param string: Text to be tokenised.
        :type string: str
        :return: String in a tokenised form that is to be split later by other helper methods.
        :rtype: str
        """
        # This is added to reduce the same characters appearing consecutively more than twice
        # to the same two chars only.
        string = re.sub(r"(.)(\1)\1{2,}", r"\1\1\1", string)
        # Added, since mentions (e.g. @trump) do not contribute to sentiment.
        string = re.sub(r"@[a-zA-Z" + self.TURKISH_CHARS + "0-9()#,!?:=;\\-\\\'`./]+", r"", string)
        # Some extra chars are added to be taken into account.
        string = re.sub(r"[^A-Za-z" + self.TURKISH_CHARS + "0-9()#,!?:=;\\-\\\'`./]", " ", string)

        # Numeric forms and emoticons, such as 22:30, are not disrupted.
        string = re.sub(r"([^\d)(])([,.:;]+)([^\d()]|$)", r"\1 \2 \3", string)
        # The punctuation marks "?" and "!" can be indicative of expressing sentiment. These
        # are therefore not removed.
        string = re.sub(r"([!?]+)", r" \1 ", string)

        # The below four regex commands are implemented to put blank spaces before or after
        # parens without disrupting emoticons.

        string = re.sub(r"\(([A-Za-z" + self.TURKISH_CHARS + "0-9,!?\\-\\\'`])", r"( \1", string)
        string = re.sub(r"([A-Za-z" + self.TURKISH_CHARS + "0-9,!?\\-\\\'`])\\(", r"\1 (", string)

        string = re.sub(r"([A-Za-z" + self.TURKISH_CHARS + "0-9,!?\\-\\\'`])\\)", r"\1 )", string)
        string = re.sub(r"\)([A-Za-z" + self.TURKISH_CHARS + "0-9,!?\\-\\\'`])", r") \1", string)

        # "(?!)" and similar forms that likely indicate sarcasm are kept.
        string = re.sub(r"(\() +([?!]+) +(\))", r"\1\2\3", string)
        # Useless parens are removed.
        string = re.sub(r"(^|[ ])+([()]+)([ ]+|$)", r" ", string)
        # Other useless punctuations are also eliminated.
        string = re.sub(r"(^|[ ])+([.;,]+)([ ]+|$)", r" ", string)

        # Emoticons ":s" and ":D".
        string = re.sub(r"((\s|^)[:]+)[ ]+([dDsSpP]+(\s|$))", r"\1\3", string)
        # Emoticon handling.
        string = re.sub(r"([:;.]+)([()dDsSpP]+)", r" \1\2 ", string)

        string = re.sub(r"\s{2,}", " ", string)
        return string

    def english_tokenize(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        Updated to some extent for this study.
        Using nltk.TweetTokenizer could be another option instead of employing this method.

        :param string: Text that is already tokenised by common_tokenize and is to be tokenised
        by rules specific to English.
        :type string: str
        :return: List that contains the output of the tokenised text.
        :rtype: list
        """

        string = string.lower()
        string = re.sub(r"\'s", " is", string)
        string = re.sub(r"\'m", "  am", string)
        string = re.sub(r"\'ve", " have", string)
        string = re.sub(r"n\'t", " n't", string)
        string = re.sub(r"\'re", " are", string)
        string = re.sub(r"\'d", " had", string)
        string = re.sub(r"\'ll", " will", string)

        string = self.common_tokenize(string)
        return string.strip().split()

    def turkish_tokenize(self, string):
        """
        This helper method tokenises the text with respect to rules specific to the Turkish language.

        :param string: The text that is already tokenised by the common tokeniser.
        :type string: str
        :return: The tokenised output of the text.
        :rtype: list
        """

        string = UnicodeTr(string).lower()
        deasciifier = Deasciifier(string)
        string = deasciifier.convert_to_turkish()
        string = self.common_tokenize(string)
        return string.strip().split()

    def get_turkish_stopwords(self):
        """
        This helper method obtains the Turkish stopwords. Hard-coding is used for the path
        to the Turkish resource. This path could also be employed as an input to the method.

        :return: Turkish stop words.
        :rtype: set
        """

        with open(os.path.join("Lexical_Resources", "Turkish-Stop-Words.txt"), "r") as f:
            turkish_stopwords = set([word for word in f])
        return turkish_stopwords

    def extract_parsed_blocks(self, file_name):
        """
        This helper method creates blocks with respect to the parsed and disambiguated Turkish texts.
        In this case, this is used to extract multi-word expressions (MWEs) in Turkish.

        :param file_name: The file path of the disambiguated text.
        :type file_name: str
        :return: List of block words (e.g. MWEs)
        :rtype: list
        """

        block_separator = "sepx"
        blocks = []
        block = []
        suffix_morpho_symbols = {"-lh": "li", "without": "_", "neg": "_"}
        with open(file_name, "r") as f:
            for line in f:
                line = UnicodeTr(line).lower()
                if line[0:2] == "<s" or line[0:3] == "</s":
                    continue
                elif line[0:4] == block_separator:  # Separator used to distinguish between different idioms/proverbs

                    blocks.append(" ".join(block))
                    block = []
                else:
                    correct_parse = line.split(" ")[1]
                    parse_ind = correct_parse.index("[")
                    root, remaind = correct_parse[:parse_ind], correct_parse[parse_ind:]
                    if "(ı" in root:
                        root = root[:root.index("(ı")]
                    root += "".join([symbol for suffix, symbol in suffix_morpho_symbols.items()
                                     if suffix in remaind])
                    block.append(root)

        return blocks

    def get_turkish_idioms(self):
        """
        This method extract Turkish idioms and proverbs using a hard-coded technique.
        The path to the resource could also be used as an input to the method.
        The resource contains the root forms of words and negation morphemes only.
        The full forms can be obtained on "Lexical_Resources/Turkish-Idioms-and-Proverbs-Full"

        :return: Turkish idioms (e.g. "kafayı yemek" (losing one's mind)) and proverbs.
        :rtype: set
        """

        with open(os.path.join("Lexical_Resources", "Turkish-Idioms-and-Proverbs"), "r") as f:
            turkish_mwes = set([word.strip() for word in f])
        return turkish_mwes

    def replace_mwe(self, review, all_mwes):
        """
        This method performs MWE extraction on a review-basis.
        For example, two tokens ("kafayı", "yemek") are converted to one token ("kafayı_yemek").

        :param review: One review.
        :type review: str
        :param all_mwes: The set of all MWEs.
        :type all_mwes: set
        :return: Updated review with respect to MWEs.
        :rtype: list
        """

        review = review.split()

        rev_length = len(review)
        if rev_length <= 1:
            return review
        ngram_lengths = [4, 3, 2]  # Could be [5, 4, 3, 2] as well.
        ind = 0
        updated_review = []
        while ind < rev_length:
            mwe_found = False
            for ngram_length in ngram_lengths:
                if ind + ngram_length <= rev_length:
                    ngram = " ".join(review[ind:ind + ngram_length])
                    is_neg = False
                    if ngram[-1] == '_':
                        ngram = ngram[:-1]
                        is_neg = True
                    if ngram in all_mwes:
                        if is_neg:
                            ngram += "_"
                        ngram = ngram.replace(" ", "-")
                        updated_review.append(ngram)
                        mwe_found = True
                        ind += ngram_length
                        break
            if not mwe_found:
                updated_review.append(review[ind])
                ind += 1
        return updated_review

    def replace_mwes(self, reviews, all_mwes):
        """
        This method detects MWEs in the whole corpus (reviews) and converts them into one token if any.

        :param reviews: All corpus reviews.
        :type reviews: list
        :param all_mwes: All MWEs in a language.
        :type all_mwes: set
        :return: Updated corpus containing MWEs.
        :rtype: list
        """

        return [self.replace_mwe(review, all_mwes) for review in reviews]

    def suffix_expansion(self, word):
        """
        A basic Turkish normalisation process. For example, the word "yapıyom" is normalised as "yapıyorum".
        In this method, several rules are specified to detect incorrectly typed suffixes specific to Turkish.

        :param word: A Turkish word.
        :type word: str
        :return: Normalised Turkish word with regard to the shortened form of the suffix.
        :rtype: str
        """

        if len(word) < 4:
            return word
        last_chs = word[-3:]
        replaced = self.TURKISH_SUFFIX_IN_FULL_FORM[
            last_chs] if last_chs in self.TURKISH_SUFFIX_IN_FULL_FORM else last_chs
        return word[:-3] + replaced

    def capture_and_update_consec_negs(self, review):
        """
        This method captures negation specific to Turkish and handles it on a review-basis specific to Turkish.
        For example, the expression "güzel değil" (not beautiful) is changed as "güzel_".

        :param review: A single review.
        :type review: list
        :return: The output review updated according to negation handling.
        :rtype: list
        """

        if len(review) < 2:
            return review
        review = ["_" if word in self.TURKISH_NEGATORS else word for word in review]
        updated_review = ["yok"] if review[0] == "_" else [review[0]]
        for word in review[1:]:
            if word[-1] == "_":
                if len(word) >= 2:
                    updated_review.append(word)
                else:

                    prev_word = updated_review[-1]
                    updated_word = prev_word[:-1] if prev_word[-1] == "_" else prev_word + "_"
                    updated_review[-1] = updated_word
            else:
                updated_review.append(word)

        return updated_review

    def capture_and_update_all_consec_negs(self, reviews):
        """
        The negation handling process performed on all corpus reviews.

        :param reviews: All corpus reviews.
        :type reviews: list
        :return: The output reviews that are changed in accordance with negation handling.
        :rtype: list
        """

        return [self.capture_and_update_consec_negs(review) for review in reviews]

    def remove_noise(self, revs):
        """
        This helper method removes noise words that are too infrequent in the corpus.
        These words are considered to not be discriminative for the sentiment classification task.

        :param revs: All corpus reviews.
        :type revs: list
        :return: Corpus reviews stripped off noise words.
        :rtype: list
        """

        noise_threshold = (1 / 1000) * len(revs)  # Could also be defined as cnt=constants.NOISE_THRESHOLD_VAL.
        word_cnt = Counter()
        for rev in revs:
            rev = set(rev)
            for word in rev:
                word_cnt[word] += 1

        revs_without_noise = []
        for rev in revs:
            revs_without_noise.append([word for word in rev if word_cnt[word] > noise_threshold])
        return revs_without_noise

    def preprocess_before_parsing(self, reviews):
        """
        This method is specific to Turkish that performs several preprocessing operations.
        This is leveraged before the parsing/disambiguation processes.

        :param reviews: Corpus reviews in Turkish.
        :type reviews: list
        :return: Corpus reviews after some preprocessing operations are performed.
        :rtype: list
        """

        review_separator = "sepx"
        reviews = [self.turkish_tokenize(review) for review in reviews]

        updated_reviews = []
        for review in reviews:
            updated_reviews.append([self.suffix_expansion(word) for word in review])

        reviews_with_sep = [review + [review_separator] for review in reviews]
        return reviews_with_sep

    def parse_and_disamb(self, reviews):
        """
        This helper method parses and disambiguates the text in Turkish.
        In this respect, Haşim Sak's morphological analyser tools are leveraged.
        To employ these external tools, the use of Python 2 is required.
        Therefore, I had to mix Python 2 and Python 3 for this project.
        Apart from this helper method, all the other code works with Python 3.7 or a more recent version.

        :param reviews: Corpus reviews in Turkish.
        :type reviews: list
        :return: Parsed and disambiguated forms of the Turkish text.
            Only roots and discriminative morphemes (e.g. negation) are taken into account.
        :rtype: list
        """

        print("Parsing and disambiguation are being performed.")
        with open(os.path.join("input", "Turkish_Morpho", "MP", "data_file.txt"), "w") as file:
            for review in reviews:
                review_str = " ".join([word for word in review])
                file.write(review_str + "\n")

        # python parse_corpus.py test.txt > test.parse.txt
        python2_command = "python parse_corpus.py data_file.txt > " + \
                          os.path.join("..", "..", "..", "sentiment.parse.txt")

        subprocess.call(python2_command, shell=True, cwd=os.path.join("input", "Turkish_Morpho", "MP"))

        # perl md.pl -disamb model.txt.. / MP / test.parse.txt test.disamb.txt
        perl_command = "perl md.pl -disamb model.txt " + \
                       os.path.join("..", "..", "..", "sentiment.parse.txt") + " " + \
                       os.path.join("..", "..", "..", "sentiment.disamb.txt")

        subprocess.run(perl_command.split(), cwd=os.path.join("input", "Turkish_Morpho", "MD-2.0"))

        reviews = self.extract_parsed_blocks("sentiment.disamb.txt")
        return reviews

    def preprocess_after_disamb(self, reviews):
        """
        This method performs preprocessing operatipns specific to Turkish.
        This is employed after parsing and disambiguation are performed.

        :param reviews: The corpus reviews in Turkish.
        :type reviews: list
        :return: The output text of the corresponding preprocessing pipeline.
        :rtype: list
        """

        all_mwes = self.get_turkish_idioms()
        reviews = self.replace_mwes(reviews, all_mwes)
        reviews = self.capture_and_update_all_consec_negs(reviews)
        return reviews

    def preprocess_english(self, reviews):
        """
        This method performs preprocessing operations for the text in English.

        :param reviews: All corpus reviews in English.
        :type reviews: list
        :return: The output text of the preprocessing pipeline.
        :rtype: list
        """

        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        stopwords = set(stopwords.words("english"))
        # nltk.TweetTokenizer could also be used in lieu of the below tokeniser.
        reviews = [self.english_tokenize(review) for review in reviews]

        reviews_upd = []
        for review in reviews:
            review_upd = [lemmatizer.lemmatize(word) for word in review]
            review_upd = [word for word in review_upd if word not in stopwords]
            reviews_upd.append(review_upd)
        return reviews_upd

    def preprocess_turkish(self, reviews):
        """
        This method performs preprocessing operations, such as parsing and stop word removal.
        This is applicable to the Turkish text only.

        :param reviews: All corpus reviews in Turkish.
        :type reviews: list
        :return: The output Turkish text after the precprocessing operations are performed.
        :rtype: list
        """

        reviews = self.preprocess_before_parsing(reviews)
        reviews = self.parse_and_disamb(reviews)
        reviews = self.preprocess_after_disamb(reviews)
        stopwords = self.get_turkish_stopwords()
        reviews_upd = []
        for review in reviews:
            review_upd = [word for word in review if word not in stopwords]
            reviews_upd.append(review_upd)
        return reviews_upd

    def preprocess_lang(self, reviews):
        """
        This helper method is an interface that helps perform the preprocessing operations.
        The available languages for this method to work on are Turkish and English.

        :param reviews: All corpus reviews.
        :type reviews: list or pandas.core.series.Series or ndarray
        :return: Preprocessed texts in the corresponding language.
        :rtype: list
        """

        lang_specific_preprocessing = {"english": self.preprocess_english,
                                       "turkish": self.preprocess_turkish}
        preprocessed_reviews = lang_specific_preprocessing[constants.LANG](reviews)

        return preprocessed_reviews

    def get_data(self, data_file):
        """
        This method reads the data from the input file. The default path is the .csv file in the "input" folder.
        This path can be changed via the corresponding variable (DATASET_PATH) in the constants.py file.

        :param data_file: The path to the data file.
        :type data_file: str
        :return: Reviews and labels (positive or negative).
        :rtype: list, list
        """

        df = pd.read_csv(data_file)
        data = df.sample(frac=1)
        reviews, labels = data["Text"], data["Sentiment"]

        reviews = self.preprocess_lang(reviews)
        reviews = self.remove_noise(reviews)
        return reviews, labels

    def preprocess_one_line(self, string):
        """
        This helper method reads the input line from terminal and performs preprocessing
            operations on one review only.

        :param string: Review.
        :type string: str
        :return: The parsed and disambiguated form of the input string
        :rtype: str
        """

        review = self.preprocess_lang(np.array([[string]]))
        return review

    def get_all_context_words(self, all_revs, context_size=5):
        """
        This helper method obtains the context window words of each target word in the corpus.

        :param all_revs: Corpus reviews.
        :type all_revs: list
        :param context_size: The size of the context window.
        :type context_size: int
        :return: Target words and the corresponding context window words explicitly appearing with them.
        :rtype: dict
        """

        contexts_of_words = {}
        for rev in all_revs:

            for i in range(0, len(rev)):
                target_word = rev[i]
                start_index = max(int(i - context_size / 2), 0)
                end_index = min(int(i + context_size / 2), len(rev))

                left_hand_side_els = {}
                right_hand_side_els = {}

                if i > 0:
                    left_hand_side_els = set(rev[start_index:i])
                if i < len(rev) - 1:
                    right_hand_side_els = set(rev[i + 1:end_index])

                word_context_words = set([])
                word_context_words = word_context_words.union(left_hand_side_els).union(right_hand_side_els)
                if target_word in word_context_words:
                    word_context_words.remove(target_word)

                if target_word not in contexts_of_words:
                    contexts_of_words[target_word] = set()
                contexts_of_words[target_word] = contexts_of_words[target_word].union(word_context_words)

        return contexts_of_words


if __name__ == "__main__":
    pre = Preprocessing()
    print(pre.english_tokenize("@christina #BLM I'm 'soooooooooooooo' nervous(!!!!) :((((((("))
    print(pre.turkish_tokenize("bu kadar etkilenmezdim.((((("))
    print(pre.turkish_tokenize("işine yaradılar... :DDDDD"))
    print(pre.turkish_tokenize("Bu filme 10/10 puan verdim.."))
    print(pre.common_tokenize("It is 22:30!"))
    print(pre.capture_and_update_consec_negs(["güzel", "değil"]))

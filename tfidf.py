# flake8 100 symb
'''
string.punctuation: Module containing data on all kinds of punctuation
numpy.log: Module for calculating ln
'''
from string import punctuation
from numpy import log


class CountVectorizer():
    """
    A simple CountVectorizer implementation.

    Attributes:
    - word_matrix (dict): A dictionary to store word frequencies.
    - vocab (dict): A dictionary to map words to indices.

    Methods:
    - _preprocess(input_corpus: str | list[str]): Preprocess the input corpus.
    - fit_transform(text: str | list[str]): Fit the model and transform the text.
    - get_feature_names(): Get the feature names.
    """

    def __init__(self) -> None:
        self.word_matrix = {}
        self.vocab = {}

    def _preprocess(self, input_corpus: str | list[str]) -> list[list[str]]:
        """
        Preprocess the input corpus.

        Parameters:
        - input_corpus (str or list[str]): Input text or a list of texts.

        Returns:
        - list[list[str]]: Processed list of words for each document.
        """
        def del_punct(string_row: str) -> str:
            pattern = str.maketrans({ord(alpha): None for alpha in punctuation})
            return string_row.translate(pattern).lower()

        output_corpus = []

        if isinstance(input_corpus, str):
            input_corpus = del_punct(input_corpus)
            output_corpus.append(input_corpus)

        elif isinstance(input_corpus, list):
            for row in input_corpus:
                row = del_punct(row)
                output_corpus.append(row.split())

        return output_corpus

    def fit_transform(self, text: str | list[str]) -> list[list[int]]:
        """
        Fit the model and transform the text.

        Parameters:
        - text (str or list[str]): Input text or a list of texts.

        Returns:
        - list[list[int]]: Count matrix for the input text.
        """
        def fit(corpus: list[list[str]]) -> None:
            index = 0
            for word_list in corpus:
                for word in word_list:
                    if word in self.word_matrix:
                        self.word_matrix[word] += 1
                    else:
                        self.word_matrix[word] = 1
                        self.vocab[word] = index
                        index += 1

        def transform(corpus: list[list[str]]) -> list[list[int]]:
            count_matrix = [[0]*len(self.vocab)
                            for _ in range(len(corpus))]

            for row_index, word_list in enumerate(corpus):
                for word in word_list:
                    word_index = self.vocab[word]
                    count_matrix[row_index][word_index] += 1

            return count_matrix

        # function pipeline
        if self.word_matrix is not None:
            self.word_matrix = {}
            self.vocab = {}

        corpus = self._preprocess(text)
        fit(corpus)
        return transform(corpus)

    def get_feature_names(self) -> list[str]:
        """
        Get the feature names.

        Returns:
        - list[str]: List of feature names.
        """
        return list(self.vocab.keys()) if self.vocab else None


class TfidfTransformer():
    """
    A TfidfTransformer implementation.

    Methods:
    - tf_transform(matrix: list[list[int]]): Transform term frequency matrix.
    - idf_transform(matrix: list[list[int]]): Transform inverse document frequency matrix.
    - fit_transform(matrix: list[list[int]]): Fit and transform the matrix.
    """
    def tf_transform(self, matrix: list[list[int]]) -> list[list[float]]:
        """
        Transform the term frequency matrix.

        Parameters:
        - matrix (list[list[int]]): The input term frequency matrix.

        Returns:
        - list[list[float]]: The transformed term frequency matrix.
        """
        for row_index, row in enumerate(matrix):
            words = sum(row)
            for cell_index, cell in enumerate(row):
                matrix[row_index][cell_index] = round(cell / words, 3)

        return matrix

    def idf_transform(self, matrix: list[list[int]]) -> list[list[float]]:
        """
        Transform the inverse document frequency matrix.

        Parameters:
        - matrix (list[list[int]]): The input term frequency matrix.

        Returns:
        - list[list[float]]: The transformed inverse document frequency matrix.
        """

        num_docs = len(matrix)
        freq_doc_matrix = [0 for _ in range(len(matrix[0]))]

        for row in matrix:
            for cell_index, cell in enumerate(row):
                if cell > 0:
                    freq_doc_matrix[cell_index] += 1

        idf_matrix = [log((num_docs + 1) / (freq_doc_matrix[i] + 1)) + 1
                      for i in range(len(freq_doc_matrix))]

        return idf_matrix

    def fit_transform(self, matrix: list[list[int]]):
        """
        Fit and transform the matrix using TF-IDF.

        Parameters:
        - matrix (list[list[int]]): The input term frequency matrix.

        Returns:
        - list[list[float]]: The transformed TF-IDF matrix.
        """
        tf_matrix = self.tf_transform(matrix)
        idf_matrix = self.idf_transform(matrix)

        return [[round(tf_matrix[j][i]*idf_matrix[i], 3)
                for i in range(len(matrix[0]))]
                for j in range(len(matrix))]


class TfidfVectorizer(CountVectorizer):
    """
    A TfidfVectorizer implementation, extending CountVectorizer.

    Attributes:
    - matrix (list[list[int]]): The count matrix.
    - _vectorizer (TfidfTransformer): TfidfTransformer instance.

    Methods:
    - fit_transform(text: str | list[str]): Fit and transform the text.
    """
    def __init__(self):
        super().__init__()
        self.matrix = None
        self._vectorizer = TfidfTransformer()

    def fit_transform(self, text: str | list[str]) -> None:
        self.matrix = super().fit_transform(text)
        return self._vectorizer.fit_transform(self.matrix)


if __name__ == '__main__':
    test_corpus = [
        'Crock Pot Pasta Never boil pasta again',
        'Pasta Pomodoro Fresh ingredients Parmesan to taste'
    ]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(test_corpus)
    print(vectorizer.get_feature_names())
    print(tfidf_matrix)

    # testcase
    test1 = [[0.201, 0.201, 0.286, 0.201, 0.201,
              0.201, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.143, 0.0, 0.0, 0.0, 0.201,
              0.201, 0.201, 0.201, 0.201, 0.201]]

    assert tfidf_matrix == test1, 'matrix not equal test'
    print('All test passed!')

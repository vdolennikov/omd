from string import punctuation


class CountVectorizer:
    """
    Text to token count matrix converter.

    Attributes:
        vocab (dict): Vocabulary of words and their counts.

    Methods:
        processing_string(corpus): Preprocess input text by removing
        punctuation and converting to lowercase.

        fit_transform(corpus): Fit vectorizer to corpus and transform it into
        a matrix of token counts.

        get_feature_names(): Return a list of feature names (words) in the
        vectorizer's vocabulary.
    """

    def __init__(self) -> None:
        """
        Initialize an instance of CountVectorizer.

        The vocabulary (vocab) is initially an empty dictionary.
        """
        self.vocab = {}

    @staticmethod
    def processing_string(corpus: list[str]) -> list[str]:
        """
        Preprocesses a list of text documents by removing punctuation
        and converting to lowercase.

        Args:
            corpus (list): A list of text documents.

        Returns:
            list: A list of preprocessed text documents.
        """
        def del_punct(str_arg: str):
            return str_arg.translate(str.maketrans('', '', punctuation))

        for index_row in range(len(corpus)):
            corpus[index_row] = del_punct(corpus[index_row]).lower()

        return corpus

    def fit_transform(self, corpus: list[str]) -> list[list[int]] | None:
        """
        Fits the CountVectorizer to the given corpus and transforms
        it into a matrix of token counts.

        Args:
            corpus (list): A list of text documents.

        Returns:
            list: A list of lists where each inner list represents
            the token counts for each document.

            None if the vocabulary is empty.
        """
        # Preprocess the input corpus
        corpus = self.processing_string(corpus)

        # Populate the vocabulary with words from the corpus
        for row in corpus:
            for word in row.split():
                if word not in self.vocab:
                    self.vocab[word] = 0

        # Count the tokens in each document
        counter_matrix = []
        for row in corpus:
            tmp_vocab = self.vocab.copy()

            for word in row.split():
                tmp_vocab[word] += 1

            counter_matrix.append(list(tmp_vocab.values()))

        return counter_matrix if counter_matrix else None

    def get_feature_names(self) -> list[str] | None:
        """
        Returns a list of feature names (words) in the vectorizer's vocabulary.

        Returns:
            list: A list of feature names (words) in the vocabulary.
            None if the vocabulary is empty.
        """
        return list(self.vocab.keys()) if self.vocab else None


if __name__ == '__main__':
    corpus = ['Crock Pot Pasta Never boil pasta again',
              'Pasta Pomodoro Fresh ingredients Parmesan to taste']

    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus)
    print(vectorizer.get_feature_names())

    assert count_matrix == [[1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
    print('All tests passed successfully')

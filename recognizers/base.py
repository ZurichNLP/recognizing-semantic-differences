from typing import List

from tqdm import tqdm

from recognizers.utils import DifferenceSample


class DifferenceRecognizer:

    def __str__(self):
        raise NotImplemented

    def predict(self,
                a: str,
                b: str,
                **kwargs,
                ) -> DifferenceSample:
        raise NotImplemented

    def predict_all(self,
                    a: List[str],
                    b: List[str],
                    **kwargs,
                    ) -> List[DifferenceSample]:
        assert len(a) == len(b)
        predictions = []
        for i in tqdm(list(range(len(a)))):
            prediction = self.predict(a[i], b[i], **kwargs)
            predictions.append(prediction)
        return predictions

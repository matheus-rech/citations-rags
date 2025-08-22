from dataclasses import dataclass
from typing import List
import pandas as pd
import numpy as np
from openai import OpenAI


@dataclass
class EmbeddingConfig:
    model: str = "text-embedding-3-small"


class Embedder:
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed_text(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(
            model=self.model,
            input=text,
            encoding_format="float",
        )
        return resp.data[0].embedding

    def build_df(self, contents: List[str]) -> pd.DataFrame:
        df = pd.DataFrame(contents, columns=["content"])
        df["embeddings"] = df["content"].apply(lambda x: self.embed_text(x))
        return df

    @staticmethod
    def save_df(df: pd.DataFrame, path: str) -> None:
        df.to_csv(path, index=False)

    @staticmethod
    def load_df(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df["embeddings"] = df.embeddings.apply(lambda s: np.array(eval(s)))
        return df

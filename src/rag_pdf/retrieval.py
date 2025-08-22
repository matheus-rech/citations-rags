from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI


@dataclass
class RAGConfig:
    chat_model: str = "gpt-4o"


class RAG:
    def __init__(self, api_key: str, chat_model: str):
        self.client = OpenAI(api_key=api_key)
        self.chat_model = chat_model

    @staticmethod
    def search(df: pd.DataFrame, query: str, embedder) -> pd.DataFrame:
        q_emb = embedder.embed_text(query)
        df = df.copy()
        df["similarity"] = df.embeddings.apply(
            lambda x: cosine_similarity(np.array(x).reshape(1, -1), np.array(q_emb).reshape(1, -1))
        )
        return df.sort_values("similarity", ascending=False)

    def generate(self, prompt: str, similar_content: pd.DataFrame, threshold: float = 0.5) -> str:
        content = similar_content.iloc[0]["content"]
        if len(similar_content) > 1:
            for _, row in similar_content.iterrows():
                sim = row["similarity"]
                if isinstance(sim, np.ndarray):
                    sim = sim[0][0]
                if sim > threshold:
                    content += f"\n\n{row['content']}"
        sys_prompt = (
            "You will be provided with an input prompt and content as context that can be used to reply to the prompt.\n\n"
            "You will do 2 things:\n\n"
            "1. First, you will internally assess whether the content provided is relevant to reply to the input prompt.\n\n"
            "2a. If that is the case, answer directly using this content. If the content is relevant, use elements found in the content to craft a reply to the input prompt.\n\n"
            "2b. If the content is not relevant, use your own knowledge to reply or say that you don't know how to respond if your knowledge is not sufficient to answer.\n\n"
            "Stay concise with your answer, replying specifically to the input prompt without mentioning additional information provided in the context content."
        )
        user_prompt = f"INPUT PROMPT:\n{prompt}\n-------\nCONTENT:\n{content}"
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            temperature=0.5,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content or ""

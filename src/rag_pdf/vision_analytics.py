from typing import List
from openai import OpenAI
from .utils import image_to_data_uri

SYSTEM_PROMPT = (
    "You will be provided with an image of a PDF page or a slide. "
    "Your goal is to deliver a detailed and engaging presentation about the content you see, using clear and accessible language suitable for a 101-level audience.\n\n"
    "If there is an identifiable title, start by stating the title to provide context for your audience.\n\n"
    "Describe visual elements in detail:\n"
    "- Diagrams: Explain each component and how they interact. For example, \"The process begins with X, which then leads to Y and results in Z.\"\n"
    "- Tables: Break down the information logically. For instance, \"Product A costs X dollars, while Product B is priced at Y dollars.\"\n\n"
    "Focus on the content itself rather than the format:\n"
    "- DO NOT include terms referring to the content format.\n"
    "- DO NOT mention the content type. Instead, directly discuss the information presented.\n\n"
    "Keep your explanation comprehensive yet concise:\n"
    "- Be exhaustive in describing the content, as your audience cannot see the image.\n"
    "- Exclude irrelevant details such as page numbers or the position of elements on the image.\n\n"
    "Use clear and accessible language:\n"
    "- Explain technical terms or concepts in simple language appropriate for a 101-level audience.\n\n"
    "Engage with the content:\n"
    "- Interpret and analyze the information where appropriate, offering insights to help the audience understand its significance.\n\n"
    "If there is an identifiable title, present the output in the following format:\n\n"
    "{TITLE}\n\n{Content description}\n\n"
    "If there is no clear title, simply provide the content description."
)


class VisionAnalyzer:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def analyze_image(self, img) -> str:
        data_uri = image_to_data_uri(img)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri}
                        }
                    ]
                },
            ],
            max_tokens=500,
            temperature=0,
            top_p=0.1,
        )
        return resp.choices[0].message.content or ""

    def analyze_images(self, images: List) -> List[str]:
        return [self.analyze_image(img) for img in images]

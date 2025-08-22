import base64
import io
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Callable, List, Any


def image_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def run_concurrently(callables: Iterable[Callable[[], Any]], max_workers: int) -> List[Any]:
    results: List[Any] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fn) for fn in callables]
        for f in as_completed(futures):
            # Preserve original order by collecting later if needed
            pass
        # Collect results in original order
        results = [f.result() for f in futures]
    return results

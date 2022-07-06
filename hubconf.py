import importlib
import os
import sys

import requests


AUDIO_INPUTS = {
    "automatic-speech-recognition",
    "audio-to-audio",
    "speech-segmentation",
    "audio-classification",
}

IMAGE_INPUTS = {
    "image-classification",
    "image-segmentation",
    "image-to-text",
    "object-detection",
}

TEXT_INPUTS = {
    "conversational",
    "feature-extraction",
    "question-answering",
    "sentence-similarity",
    "fill-mask",
    "table-question-answering",
    "tabular-classification",
    "tabular-regression",
    "summarization",
    "text2text-generation",
    "text-classification",
    "text-to-image",
    "text-to-speech",
    "token-classification",
    "zero-shot-classification",
}

TASKS = set().union(AUDIO_INPUTS, IMAGE_INPUTS, TEXT_INPUTS)
PIPELINES_MAP = {
    k: "".join([x.title() for x in k.split("-")]) + "Pipeline" for k in TASKS
}


def pipeline(model_id, library_name=None, task=None):
    if library_name is None or task is None:
        r = requests.get(f"https://hf.co/api/models/{model_id}")
        r.raise_for_status()
        model_info = r.json()
        if library_name is None:
            library_name = model_info.get("library_name")
        if task is None:
            task = model_info.get("pipeline_tag")

    assert library_name is not None
    assert task is not None

    path = f"docker_images/{library_name}"
    assert os.path.exists(path)

    pipeline_class_name = PIPELINES_MAP.get(task)
    assert pipeline_class_name is not None

    sys.path.append(path)
    module = importlib.import_module(f"app.pipelines.{task.replace('-', '_')}")
    sys.path.remove(path)

    pipeline_class = getattr(module, pipeline_class_name)
    return pipeline_class(model_id)

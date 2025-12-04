import os

import pytest


def require_env(var_name: str) -> str:
    """Skip the test unless the required environment variable is provided."""
    value = os.environ.get(var_name)
    if not value:
        pytest.skip(f"Set {var_name} to run this integration snippet")
    return value


def require_flag(flag_name: str) -> None:
    """Skip unless a generic feature flag is set to a truthy value."""
    value = os.environ.get(flag_name, "").lower()
    if value not in {"1", "true", "yes", "on"}:
        pytest.skip(f"Enable {flag_name} to run this integration snippet")


def test_embedding_openai_basic() -> None:
    require_env("OPENAI_API_KEY")

    # --8<-- [start:embedding_openai_basic]
    import tempfile
    from pathlib import Path

    import lancedb
    from lancedb.embeddings import get_registry
    from lancedb.pydantic import LanceModel, Vector

    db_path = Path(tempfile.mkdtemp()) / "openai-embeddings"
    db = lancedb.connect(str(db_path))
    func = get_registry().get("openai").create(name="text-embedding-ada-002")

    class Words(LanceModel):
        text: str = func.SourceField()
        vector: Vector(func.ndims()) = func.VectorField()

    table = db.create_table("words", schema=Words, mode="overwrite")
    table.add(
        [
            {"text": "hello world"},
            {"text": "goodbye world"},
        ]
    )

    query = "greetings"
    actual = table.search(query).limit(1).to_pydantic(Words)[0]
    print(actual.text)
    # --8<-- [end:embedding_openai_basic]


def test_embedding_aws_usage() -> None:
    require_flag("RUN_AWS_BEDROCK_SNIPPETS")

    # --8<-- [start:embedding_aws_usage]
    import tempfile
    from pathlib import Path

    import lancedb
    import pandas as pd
    from lancedb.embeddings import get_registry
    from lancedb.pydantic import LanceModel, Vector

    model = get_registry().get("bedrock-text").create()

    class TextModel(LanceModel):
        text: str = model.SourceField()
        vector: Vector(model.ndims()) = model.VectorField()

    df = pd.DataFrame({"text": ["hello world", "goodbye world"]})
    db = lancedb.connect(str(Path(tempfile.mkdtemp()) / "bedrock-demo"))
    tbl = db.create_table("test", schema=TextModel, mode="overwrite")

    tbl.add(df)
    rs = tbl.search("hello").limit(1).to_pandas()
    print(rs.head())
    # --8<-- [end:embedding_aws_usage]


def test_embedding_cohere_usage() -> None:
    require_env("COHERE_API_KEY")

    # --8<-- [start:embedding_cohere_usage]
    import tempfile
    from pathlib import Path

    import lancedb
    from lancedb.embeddings import EmbeddingFunctionRegistry
    from lancedb.pydantic import LanceModel, Vector

    cohere = (
        EmbeddingFunctionRegistry.get_instance()
        .get("cohere")
        .create(name="embed-multilingual-v2.0")
    )

    class TextModel(LanceModel):
        text: str = cohere.SourceField()
        vector: Vector(cohere.ndims()) = cohere.VectorField()

    data = [{"text": "hello world"}, {"text": "goodbye world"}]

    db = lancedb.connect(str(Path(tempfile.mkdtemp()) / "cohere-demo"))
    tbl = db.create_table("test", schema=TextModel, mode="overwrite")
    tbl.add(data)
    # --8<-- [end:embedding_cohere_usage]


def test_embedding_gemini_usage() -> None:
    require_flag("RUN_GEMINI_SNIPPETS")

    # --8<-- [start:embedding_gemini_usage]
    import tempfile
    from pathlib import Path

    import lancedb
    import pandas as pd
    from lancedb.embeddings import get_registry
    from lancedb.pydantic import LanceModel, Vector

    model = get_registry().get("gemini-text").create()

    class TextModel(LanceModel):
        text: str = model.SourceField()
        vector: Vector(model.ndims()) = model.VectorField()

    df = pd.DataFrame({"text": ["hello world", "goodbye world"]})
    db = lancedb.connect(str(Path(tempfile.mkdtemp()) / "gemini-demo"))
    tbl = db.create_table("test", schema=TextModel, mode="overwrite")

    tbl.add(df)
    rs = tbl.search("hello").limit(1).to_pandas()
    print(rs.head())
    # --8<-- [end:embedding_gemini_usage]


def test_embedding_huggingface_usage() -> None:
    require_flag("RUN_HUGGINGFACE_SNIPPETS")

    # --8<-- [start:embedding_huggingface_usage]
    import tempfile
    from pathlib import Path

    import lancedb
    import pandas as pd
    from lancedb.embeddings import get_registry
    from lancedb.pydantic import LanceModel, Vector

    db = lancedb.connect(str(Path(tempfile.mkdtemp()) / "huggingface-demo"))
    model = get_registry().get("huggingface").create(name="facebook/bart-base")

    class Words(LanceModel):
        text: str = model.SourceField()
        vector: Vector(model.ndims()) = model.VectorField()

    df = pd.DataFrame({"text": ["hi hello sayonara", "goodbye world"]})
    table = db.create_table("greets", schema=Words)
    table.add(df)
    query = "old greeting"
    actual = table.search(query).limit(1).to_pydantic(Words)[0]
    print(actual.text)
    # --8<-- [end:embedding_huggingface_usage]


def test_embedding_ibm_usage() -> None:
    require_flag("RUN_IBM_WATSONX_SNIPPETS")

    # --8<-- [start:embedding_ibm_usage]
    import os
    import tempfile
    from pathlib import Path

    import lancedb
    from lancedb.embeddings import EmbeddingFunctionRegistry
    from lancedb.pydantic import LanceModel, Vector

    watsonx_embed = (
        EmbeddingFunctionRegistry.get_instance()
        .get("watsonx")
        .create(
            name="ibm/slate-125m-english-rtrvr",
            api_key=os.environ.get("WATSONX_API_KEY"),
            project_id=os.environ.get("WATSONX_PROJECT_ID"),
        )
    )

    class TextModel(LanceModel):
        text: str = watsonx_embed.SourceField()
        vector: Vector(watsonx_embed.ndims()) = watsonx_embed.VectorField()

    data = [
        {"text": "hello world"},
        {"text": "goodbye world"},
    ]

    db = lancedb.connect(str(Path(tempfile.mkdtemp()) / "watsonx-demo"))
    tbl = db.create_table("watsonx_test", schema=TextModel, mode="overwrite")
    tbl.add(data)

    rs = tbl.search("hello").limit(1).to_pandas()
    print(rs.head())
    # --8<-- [end:embedding_ibm_usage]


def test_embedding_imagebind_examples() -> None:
    require_flag("RUN_IMAGEBIND_SNIPPETS")
    pytest.importorskip("imagebind")

    # --8<-- [start:embedding_imagebind_setup]
    import lancedb
    from lancedb.embeddings import get_registry
    from lancedb.pydantic import LanceModel, Vector

    db = lancedb.connect("/tmp/imagebind-db")
    func = get_registry().get("imagebind").create()

    class ImageBindModel(LanceModel):
        text: str
        image_uri: str = func.SourceField()
        audio_path: str
        vector: Vector(func.ndims()) = func.VectorField()

    text_list = ["A dog.", "A car", "A bird"]
    image_paths = [
        "./assets/dog_image.jpg",
        "./assets/car_image.jpg",
        "./assets/bird_image.jpg",
    ]
    audio_paths = [
        "./assets/dog_audio.wav",
        "./assets/car_audio.wav",
        "./assets/bird_audio.wav",
    ]

    inputs = [
        {"text": a, "audio_path": b, "image_uri": c}
        for a, b, c in zip(text_list, audio_paths, image_paths)
    ]

    table = db.create_table("img_bind", schema=ImageBindModel)
    table.add(inputs)
    # --8<-- [end:embedding_imagebind_setup]

    # --8<-- [start:embedding_imagebind_image_search]
    query_image = "./assets/dog_image2.jpg"
    actual = table.search(query_image).limit(1).to_pydantic(ImageBindModel)[0]
    print(actual.text == "dog")
    # --8<-- [end:embedding_imagebind_image_search]

    # --8<-- [start:embedding_imagebind_audio_search]
    query_audio = "./assets/car_audio2.wav"
    actual = table.search(query_audio).limit(1).to_pydantic(ImageBindModel)[0]
    print(actual.text == "car")
    # --8<-- [end:embedding_imagebind_audio_search]

    # --8<-- [start:embedding_imagebind_text_search]
    query = "an animal which flies and tweets"
    actual = table.search(query).limit(1).to_pydantic(ImageBindModel)[0]
    print(actual.text == "bird")
    # --8<-- [end:embedding_imagebind_text_search]


def test_embedding_instructor_usage() -> None:
    require_flag("RUN_INSTRUCTOR_SNIPPETS")

    # --8<-- [start:embedding_instructor_usage]
    import tempfile
    from pathlib import Path

    import lancedb
    from lancedb.embeddings import get_registry
    from lancedb.pydantic import LanceModel, Vector

    instructor = (
        get_registry()
        .get("instructor")
        .create(
            source_instruction="represent the document for retrieval",
            query_instruction="represent the document for retrieving the most similar documents",
        )
    )

    class Schema(LanceModel):
        vector: Vector(instructor.ndims()) = instructor.VectorField()
        text: str = instructor.SourceField()

    db = lancedb.connect(str(Path(tempfile.mkdtemp()) / "instructor-demo"))
    tbl = db.create_table("test", schema=Schema, mode="overwrite")

    texts = [
        {
            "text": "Capitalism has been dominant in the Western world since the end of feudalism."
        },
        {
            "text": "The disparate impact theory is especially controversial under the Fair Housing Act."
        },
        {
            "text": "Disparate impact in United States labor law refers to practices in employment."
        },
    ]

    tbl.add(texts)
    # --8<-- [end:embedding_instructor_usage]


def test_embedding_jina_text() -> None:
    require_env("JINA_API_KEY")

    # --8<-- [start:embedding_jina_text]
    import os
    import tempfile
    from pathlib import Path

    import lancedb
    from lancedb.embeddings import EmbeddingFunctionRegistry
    from lancedb.pydantic import LanceModel, Vector

    os.environ["JINA_API_KEY"] = os.environ["JINA_API_KEY"]

    jina_embed = (
        EmbeddingFunctionRegistry.get_instance()
        .get("jina")
        .create(name="jina-embeddings-v2-base-en")
    )

    class TextModel(LanceModel):
        text: str = jina_embed.SourceField()
        vector: Vector(jina_embed.ndims()) = jina_embed.VectorField()

    data = [{"text": "hello world"}, {"text": "goodbye world"}]

    db = lancedb.connect(str(Path(tempfile.mkdtemp()) / "jina-text"))
    tbl = db.create_table("test", schema=TextModel, mode="overwrite")

    tbl.add(data)
    # --8<-- [end:embedding_jina_text]


def test_embedding_jina_multimodal() -> None:
    require_flag("RUN_JINA_MULTIMODAL_SNIPPETS")

    # --8<-- [start:embedding_jina_multimodal]
    import os
    import tempfile
    from pathlib import Path

    import lancedb
    import pandas as pd
    import requests
    from lancedb.embeddings import get_registry
    from lancedb.pydantic import LanceModel, Vector

    os.environ["JINA_API_KEY"] = os.environ.get("JINA_API_KEY", "jina_*")

    db = lancedb.connect(str(Path(tempfile.mkdtemp()) / "jina-images"))
    func = get_registry().get("jina").create()

    class Images(LanceModel):
        label: str
        image_uri: str = func.SourceField()
        image_bytes: bytes = func.SourceField()
        vector: Vector(func.ndims()) = func.VectorField()
        vec_from_bytes: Vector(func.ndims()) = func.VectorField()

    table = db.create_table("images", schema=Images)
    labels = ["cat", "cat", "dog", "dog", "horse", "horse"]
    uris = [
        "http://farm1.staticflickr.com/53/167798175_7c7845bbbd_z.jpg",
        "http://farm1.staticflickr.com/134/332220238_da527d8140_z.jpg",
        "http://farm9.staticflickr.com/8387/8602747737_2e5c2a45d4_z.jpg",
        "http://farm5.staticflickr.com/4092/5017326486_1f46057f5f_z.jpg",
        "http://farm9.staticflickr.com/8216/8434969557_d37882c42d_z.jpg",
        "http://farm6.staticflickr.com/5142/5835678453_4f3a4edb45_z.jpg",
    ]
    image_bytes = [requests.get(uri).content for uri in uris]
    table.add(
        pd.DataFrame({"label": labels, "image_uri": uris, "image_bytes": image_bytes})
    )
    # --8<-- [end:embedding_jina_multimodal]


def test_embedding_ollama_usage() -> None:
    require_flag("RUN_OLLAMA_SNIPPETS")

    # --8<-- [start:embedding_ollama_usage]
    import tempfile
    from pathlib import Path

    import lancedb
    from lancedb.embeddings import get_registry
    from lancedb.pydantic import LanceModel, Vector

    db = lancedb.connect(str(Path(tempfile.mkdtemp()) / "ollama-demo"))
    func = get_registry().get("ollama").create(name="nomic-embed-text")

    class Words(LanceModel):
        text: str = func.SourceField()
        vector: Vector(func.ndims()) = func.VectorField()

    table = db.create_table("words", schema=Words, mode="overwrite")
    table.add(
        [
            {"text": "hello world"},
            {"text": "goodbye world"},
        ]
    )

    query = "greetings"
    actual = table.search(query).limit(1).to_pydantic(Words)[0]
    print(actual.text)
    # --8<-- [end:embedding_ollama_usage]


def test_embedding_openclip_examples() -> None:
    require_flag("RUN_OPENCLIP_SNIPPETS")

    # --8<-- [start:embedding_openclip_setup]
    import tempfile
    from pathlib import Path

    import lancedb
    import pandas as pd
    import requests
    from lancedb.embeddings import get_registry
    from lancedb.pydantic import LanceModel, Vector

    db = lancedb.connect(str(Path(tempfile.mkdtemp()) / "openclip-demo"))
    func = get_registry().get("open-clip").create()

    class Images(LanceModel):
        label: str
        image_uri: str = func.SourceField()
        image_bytes: bytes = func.SourceField()
        vector: Vector(func.ndims()) = func.VectorField()
        vec_from_bytes: Vector(func.ndims()) = func.VectorField()

    table = db.create_table("images", schema=Images)
    labels = ["cat", "cat", "dog", "dog", "horse", "horse"]
    uris = [
        "http://farm1.staticflickr.com/53/167798175_7c7845bbbd_z.jpg",
        "http://farm1.staticflickr.com/134/332220238_da527d8140_z.jpg",
        "http://farm9.staticflickr.com/8387/8602747737_2e5c2a45d4_z.jpg",
        "http://farm5.staticflickr.com/4092/5017326486_1f46057f5f_z.jpg",
        "http://farm9.staticflickr.com/8216/8434969557_d37882c42d_z.jpg",
        "http://farm6.staticflickr.com/5142/5835678453_4f3a4edb45_z.jpg",
    ]
    image_bytes = [requests.get(uri).content for uri in uris]
    table.add(
        pd.DataFrame({"label": labels, "image_uri": uris, "image_bytes": image_bytes})
    )
    # --8<-- [end:embedding_openclip_setup]

    # --8<-- [start:embedding_openclip_text_search]
    actual = table.search("man's best friend").limit(1).to_pydantic(Images)[0]
    print(actual.label)

    frombytes = (
        table.search("man's best friend", vector_column_name="vec_from_bytes")
        .limit(1)
        .to_pydantic(Images)[0]
    )
    print(frombytes.label)
    # --8<-- [end:embedding_openclip_text_search]

    # --8<-- [start:embedding_openclip_image_search]
    import io

    from PIL import Image

    query_image_uri = "http://farm1.staticflickr.com/200/467715466_ed4a31801f_z.jpg"
    image_bytes = requests.get(query_image_uri).content
    query_image = Image.open(io.BytesIO(image_bytes))
    actual = table.search(query_image).limit(1).to_pydantic(Images)[0]
    print(actual.label == "dog")

    other = (
        table.search(query_image, vector_column_name="vec_from_bytes")
        .limit(1)
        .to_pydantic(Images)[0]
    )
    print(other.label)
    # --8<-- [end:embedding_openclip_image_search]


def test_embedding_sentence_transformers_baai() -> None:
    require_flag("RUN_SENTENCE_TRANSFORMERS_SNIPPETS")

    # --8<-- [start:embedding_sentence_transformers_baai]
    import tempfile
    from pathlib import Path

    import lancedb
    from lancedb.embeddings import get_registry
    from lancedb.pydantic import LanceModel, Vector

    db = lancedb.connect(str(Path(tempfile.mkdtemp()) / "sentence-transformers"))
    model = (
        get_registry()
        .get("sentence-transformers")
        .create(name="BAAI/bge-small-en-v1.5", device="cpu")
    )

    class Words(LanceModel):
        text: str = model.SourceField()
        vector: Vector(model.ndims()) = model.VectorField()

    table = db.create_table("words", schema=Words)
    table.add(
        [
            {"text": "hello world"},
            {"text": "goodbye world"},
        ]
    )

    query = "greetings"
    actual = table.search(query).limit(1).to_pydantic(Words)[0]
    print(actual.text)
    # --8<-- [end:embedding_sentence_transformers_baai]


def test_embedding_voyageai_usage() -> None:
    require_env("VOYAGE_API_KEY")

    # --8<-- [start:embedding_voyageai_usage]
    import tempfile
    from pathlib import Path

    import lancedb
    from lancedb.embeddings import EmbeddingFunctionRegistry
    from lancedb.pydantic import LanceModel, Vector

    voyageai = (
        EmbeddingFunctionRegistry.get_instance().get("voyageai").create(name="voyage-3")
    )

    class TextModel(LanceModel):
        text: str = voyageai.SourceField()
        vector: Vector(voyageai.ndims()) = voyageai.VectorField()

    data = [{"text": "hello world"}, {"text": "goodbye world"}]

    db = lancedb.connect(str(Path(tempfile.mkdtemp()) / "voyageai-demo"))
    tbl = db.create_table("test", schema=TextModel, mode="overwrite")

    tbl.add(data)
    # --8<-- [end:embedding_voyageai_usage]

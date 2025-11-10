from typing import Iterable

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Chroma(
    collection_name="conventions",
    embedding_function=embeddings_model,
    persist_directory="./chroma_db",
)


def load_conventions_pdf() -> list[Document]:
    conventions_files = ["sources/Kafka.pdf", "sources/API types.pdf", "sources/REST API.pdf"]
    all_documents = []
    for conventions_file in conventions_files:
        loader = PyPDFLoader(file_path=conventions_file)
        all_documents.extend(loader.load())
    return all_documents


def load_conventions_md() -> list[Document]:
    loader = DirectoryLoader("sources/", glob="**/*.md", loader_cls=TextLoader)
    return loader.load()


def split_by_length(documents: Iterable[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, add_start_index=True
    )
    return text_splitter.split_documents(documents)


def split_by_sections(documents: Iterable[Document]) -> list[Document]:
    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
        ("#", "convention-type"),
        ("##", "convention"),
    ], strip_headers=False)

    chunks = [
        chunk
        for doc in documents
        for chunk in text_splitter.split_text(doc.page_content)
    ]
    return chunks


def index_pdf_conventions():
    documents = load_conventions_pdf()
    print(f"{len(documents)} documents loaded")
    chunks = split_by_length(documents)
    print(f"Documents split in {len(chunks)} chunks")
    for chunk in enumerate(chunks):
        print(chunk)
    vector_store.reset_collection()
    vector_store.add_documents(chunks)


def index_md_conventions():
    documents = load_conventions_md()
    print(f"{len(documents)} documents loaded")
    chunks = split_by_sections(documents)
    print(f"Documents split in {len(chunks)} chunks")
    for chunk in enumerate(chunks):
        print(chunk)
    vector_store.reset_collection()
    vector_store.add_documents(chunks)


if __name__ == '__main__':
    index_md_conventions()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    results = retriever.invoke("How to name GET API endpoint?")
    print("Found documents:")
    for result in enumerate(results):
        print(result)

import os
from dotenv import load_dotenv, find_dotenv
import numpy as np
import nest_asyncio

nest_asyncio.apply()

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.query_engine import RetrieverQueryEngine
from trulens_eval import Feedback
from trulens.apps.llamaindex import TruLlama
from trulens.core import TruSession
from trulens_eval.feedback import GroundTruthAgreement
from trulens.apps.llamaindex import TruLlama
import numpy as np
import pandas as pd


def get_prebuilt_trulens_recorder(query_engine, app_id,provider,data):
    
    f_qa_relevance = Feedback(
        provider.relevance_with_cot_reasons,
        name="Answer Relevance"
    ).on_input_output()

    context_selection = TruLlama.select_source_nodes().node.text
    
    f_qs_relevance = (Feedback(provider.qs_relevance,name="Context Relevance")
    .on_input()
    .on(context_selection)
    .aggregate(np.mean)
    )
  

    df = pd.DataFrame(data)
    
    session = TruSession()
    session.reset_database()
    
    session.add_ground_truth_to_dataset(
        dataset_name="test_dataset_ir",
        ground_truth_df=df,
        dataset_metadata={"domain": "Random IR dataset"},
    )
    ground_truth_df = session.get_ground_truth("test_dataset_ir")

    grounded = GroundTruthAgreement(ground_truth_df,provider=provider)
    
    f_groundedness = (Feedback(grounded.agreement_measure,name="Groundedness")
    .on(context_selection)
    .on_output()
    # .aggregate(grounded.grounded_statements_aggregator)
    )
    
    tru_recorder = TruLlama(
    # sentence_window_engine,
    query_engine,
    app_id="App_1",
    feedbacks=[
        f_qa_relevance,
        f_qs_relevance,
        f_groundedness
    ]
)
    return tru_recorder

def build_sentence_window_index(
    documents,
    llm,
    embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    sentence_window_size=3,
    save_dir="sentence_index",
):
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    # sentence_context = ServiceContext.from_defaults(
    #     llm=llm,
    #     embed_model=embed_model,
    #     node_parser=node_parser,
    # )
    Settings.llm = llm
    Settings.embed_model = embed_model
    # Settings.node_parser = SentenceSplitter(chunk_size=512,chunk_overlap=20)
    Settings.node_parser = node_parser
    Settings.text_splitter = text_splitter
    Settings.num_output = 512
    Settings.context_window = 4096
    text_splitter = SentenceSplitter()
    
    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            documents, service_context=sentence_context
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=sentence_context,
        )

    return sentence_index

def get_automerging_query_engine(
    automerging_index,
    similarity_top_k=12,
    rerank_top_n=6,
):
    base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever, automerging_index.storage_context, verbose=True
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )
    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[rerank]
    )
    return auto_merging_engine
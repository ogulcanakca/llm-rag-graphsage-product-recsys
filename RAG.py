import streamlit as st
import json
import chromadb
from sentence_transformers import SentenceTransformer
import os
import warnings
import sqlite3
from config import DATA_FILE_NEW, RECOMMENDATIONS_FILE, PRODUCT_METADATA_DB, CHROMA_DB_PATH, COLLECTION_NAME, MODEL_NAME


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")

@st.cache_resource
def load_embedding_model():
    model = SentenceTransformer(MODEL_NAME)
    return model

@st.cache_resource
def get_chroma_client():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client

def load_data_from_new_json(file_path):
    documents = []
    metadatas = []
    ids = []
    if not os.path.exists(file_path):
        st.error(f"'{file_path}' not found. Please make sure the file is in the correct location.")
        return [], [], []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for asin, keywords in data.items():
            document_text = ". ".join(keywords) + "." if keywords else "No keywords."
            documents.append(document_text)
            metadatas.append({"asin": asin, "source": "new_json"})
            ids.append(asin)
    except json.JSONDecodeError:
        st.error(f"File '{file_path}' is not in a valid JSON format.")
        return [], [], []
    except Exception as e:
        st.error(f"An error occurred while reading '{file_path}': {e}")
        return [], [], []
    return documents, metadatas, ids

@st.cache_data
def load_product_recommendations(file_path):
    if not os.path.exists(file_path):
        st.warning(f"'{file_path}' not found. GraphSAGE recommendations will not be displayed.")
        return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        recommendations_map = {}
        for product_group in data.get("products", []):
            anchor_product_info = product_group.get("anchor_product", {})
            anchor_asin = anchor_product_info.get("asin")
            if anchor_asin:
                recommendations_map[anchor_asin] = {
                    "powered_desc": anchor_product_info.get("powered_desc"),
                    "similar_products": product_group.get("similar_products", [])
                }
        return recommendations_map
    except json.JSONDecodeError:
        st.error(f"File '{file_path}' is not in a valid JSON format. GraphSAGE recommendations could not be loaded.")
        return {}
    except Exception as e:
        st.error(f"An error occurred while reading '{file_path}': {e}. GraphSAGE recommendations could not be loaded.")
        return {}

def initialize_and_populate_chroma(client, model, documents, metadatas, ids_list):
    from chromadb.utils import embedding_functions
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )

    if collection.count() == 0 and documents:
        st.info(f"Collection '{COLLECTION_NAME}' is empty. Creating embeddings for {len(documents)} documents...")
        embeddings = model.encode(documents, show_progress_bar=True, normalize_embeddings=True).tolist()
        
        batch_size = 100
        for i in range(0, len(ids_list), batch_size):
            id_batch = ids_list[i:i+batch_size]
            doc_batch = documents[i:i+batch_size]
            meta_batch = metadatas[i:i+batch_size]
            embed_batch = embeddings[i:i+batch_size]
            
            collection.add(
                ids=id_batch,
                embeddings=embed_batch,
                documents=doc_batch,
                metadatas=meta_batch
            )
        st.success(f"{len(documents)} documents successfully added to collection '{COLLECTION_NAME}'.")
    elif not documents and collection.count() == 0:
        st.warning(f"Data could not be loaded from '{DATA_FILE_NEW}', so the collection will remain empty.")
    return collection

@st.cache_data
def get_product_image_details(asin_to_find):
    """Fetches image information for the specified ASIN from the SQLite database."""
    if not os.path.exists(PRODUCT_METADATA_DB):
        return None
    
    conn = None
    try:
        conn = sqlite3.connect(PRODUCT_METADATA_DB)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT first_image_url, all_image_urls_low_json, all_image_urls_high_json, title
            FROM products
            WHERE asin = ?
        """, (asin_to_find,))
        row = cursor.fetchone()
        if row:
            return {
                "first_image_url": row[0],
                "all_low_json": row[1],
                "all_high_json": row[2],
                "title": row[3]
            }
        return None
    except sqlite3.Error as e:
        st.error(f"SQLite query error (ASIN: {asin_to_find}): {e}")
        return None
    finally:
        if conn:
            conn.close()

def increment_image_index(session_key_for_idx, num_urls):
    current_val = st.session_state.get(session_key_for_idx, 0)
    st.session_state[session_key_for_idx] = (current_val + 1) % num_urls

def decrement_image_index(session_key_for_idx, num_urls):
    current_val = st.session_state.get(session_key_for_idx, 0)
    st.session_state[session_key_for_idx] = (current_val - 1 + num_urls) % num_urls

def display_image_carousel(product_asin, image_details, container):
    if not image_details:
        container.caption("Image details not found for this product.")
        return

    default_image_url = image_details.get("first_image_url")
    low_urls_json = image_details.get("all_low_json")
    high_urls_json = image_details.get("all_high_json")
    product_title = image_details.get("title", product_asin)

    if product_title and (product_title.strip().startswith("var ") or product_title.strip().startswith("<")):
        product_title = product_asin

    session_key_idx = f"image_idx_{product_asin}"
    session_key_urls_list = f"image_urls_list_{product_asin}"

    if session_key_urls_list not in st.session_state:
        try:
            low_list = json.loads(low_urls_json or '[]')
            high_list = json.loads(high_urls_json or '[]')
            combined_urls = list(dict.fromkeys(high_list + low_list))
            st.session_state[session_key_urls_list] = combined_urls
        except json.JSONDecodeError:
            st.session_state[session_key_urls_list] = []
            
    all_available_urls = st.session_state.get(session_key_urls_list, [])

    if session_key_idx not in st.session_state:
        st.session_state[session_key_idx] = 0
    
    current_idx = st.session_state[session_key_idx]

    image_to_display = default_image_url
    if all_available_urls:
        if not (0 <= current_idx < len(all_available_urls)):
            current_idx = 0
            st.session_state[session_key_idx] = 0 
        image_to_display = all_available_urls[current_idx]
    
    if image_to_display:
        container.image(image_to_display, caption=f"{product_title} (Image {current_idx + 1}/{len(all_available_urls) if all_available_urls else 1})", width=200) # Sabit genişlik için

    else:
        container.caption("Image not found.")

    if len(all_available_urls) > 1:
        container.button(
            "◀ Previous Image", 
            key=f"prev_{product_asin}", 
            use_container_width=True,
            on_click=decrement_image_index,
            args=(session_key_idx, len(all_available_urls))
        )

        container.markdown(
            f"<div style='text-align: center; font-size: 0.9em;'>{current_idx + 1} / {len(all_available_urls)}</div>", 
            unsafe_allow_html=True
        )
        
        container.button(
            "Next Image ▶", 
            key=f"next_{product_asin}", 
            use_container_width=True,
            on_click=increment_image_index,
            args=(session_key_idx, len(all_available_urls))
        )


st.set_page_config(layout="wide", page_title="TÜBİTAK 2209-A")
st.title("Ürün Yorumlarından Stil Çıkarımı ve LLM ile RAG Destekli Doğal Dil Tabanlı Öneri Sistemi Tasarımı")
st.markdown(f"""
🛍️ RAG & GraphSAGE Based Product Recommendation System
""")

col1, col2, col3 = st.columns([1,2,1]) 

image_file_name = "asset/logo.png" 

if os.path.exists(image_file_name):
    with col2: 
        st.image(image_file_name, use_container_width=True)
else:
    st.warning(f"Image file '{image_file_name}' not found. Please ensure the file is in the same directory as the code.")

embedding_model = load_embedding_model()

embedding_model = load_embedding_model()
chroma_client = get_chroma_client()
documents_new, metadatas_new, ids_list_new = load_data_from_new_json(DATA_FILE_NEW)

collection = None
if documents_new: 
    collection = initialize_and_populate_chroma(chroma_client, embedding_model, documents_new, metadatas_new, ids_list_new)
else:
    st.error(f"Since data could not be loaded from '{DATA_FILE_NEW}', the RAG search feature will not work.")

product_recommendations_data = load_product_recommendations(RECOMMENDATIONS_FILE)

st.sidebar.header("Search Parameters")
if 'search_query' not in st.session_state:
    st.session_state['search_query'] = "stylish durable leather"

query_text = st.sidebar.text_input(
    "Enter the product features you want to search for:", 
    key='search_query'
) 
top_n_results = st.sidebar.slider("How Many Products to Retrieve with RAG?", 1, 10, 3)

if st.sidebar.button("🔍 Find Products"):
    if not query_text:
        st.warning("Please enter a search query.")
        st.session_state['last_results'] = None
        st.session_state['last_query'] = None
    elif collection is None or collection.count() == 0:
        st.warning(f"There is no RAG data in the database.")
        st.session_state['last_results'] = None
        st.session_state['last_query'] = None
    else:
        st.info(f"Searching with RAG for '{query_text}'...")
        try:
            query_embedding = embedding_model.encode(query_text, normalize_embeddings=True).tolist()
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_n_results,
                include=['metadatas', 'documents', 'distances']
            )
            st.session_state['last_results'] = results
            st.session_state['last_query'] = query_text
            st.success("Search completed.")
        except Exception as e:
            st.error(f"An error occurred during the search: {e}")
            st.session_state['last_results'] = None
            st.session_state['last_query'] = query_text

if 'last_results' in st.session_state and st.session_state['last_results']:
    results_to_display = st.session_state['last_results']
    last_query_text = st.session_state.get('last_query', query_text)
    
    st.header(f"Search Results for '{last_query_text}':")

    if not results_to_display.get('ids') or not results_to_display['ids'][0]:
        st.info("No RAG results found matching your search.")
    else:
        for i in range(len(results_to_display['ids'][0])):
            rag_asin = results_to_display['metadatas'][0][i]['asin']
            rag_document_keywords = results_to_display['documents'][0][i]
            rag_distance = results_to_display['distances'][0][i]

            st.markdown("---")
            col_img_rag, col_details_rag = st.columns([1, 3]) 

            with col_img_rag:
                st.markdown(f"##### `{rag_asin}`") 
                rag_image_details = get_product_image_details(rag_asin)
                rag_carousel_id = f"rag_{rag_asin}_{i}" 
                display_image_carousel(rag_carousel_id, rag_image_details, col_img_rag)

            with col_details_rag:
                st.markdown(f"**RAG Similarity Score (Distance):** `{rag_distance:.4f}`")
                
                with st.expander("Keywords:"):
                    st.json(rag_document_keywords.split(". ") if rag_document_keywords else [])

                anchor_specific_data = product_recommendations_data.get(rag_asin)
                if anchor_specific_data:
                    powered_description = anchor_specific_data.get("powered_desc")
                    if powered_description:
                        st.markdown("**💬 Product Description (LLM):**")
                        st.container(height=100, border=False).caption(powered_description)

                    similar_products_from_graph = anchor_specific_data.get("similar_products", [])
                    if similar_products_from_graph:
                        st.markdown("**🔗 GraphSAGE Recommendations Specific to This Product:**")
                        for gs_idx, similar_prod in enumerate(similar_products_from_graph): 
                            sim_asin = similar_prod.get('asin')
                            sim_reason = similar_prod.get('recommendation_reason')
                            sim_keywords_gs = similar_prod.get('keywords', [])
                            
                            with st.container(border=True):
                                col_img_gs, col_details_gs = st.columns([1,3])
                                with col_img_gs:
                                    st.markdown(f"**`{sim_asin}`**")
                                    gs_image_details = get_product_image_details(sim_asin)
                                    unique_gs_carousel_id = f"gs_{rag_asin}_{sim_asin}_{gs_idx}" 
                                    display_image_carousel(unique_gs_carousel_id, gs_image_details, col_img_gs) 
                                
                                with col_details_gs:
                                    if sim_reason:
                                        st.caption(f"💬 {sim_reason}")
                                    if sim_keywords_gs:
                                        with st.expander(f"keywords for `{sim_asin}`:", expanded=False):
                                            st.json(sim_keywords_gs)
                                st.markdown("<br>", unsafe_allow_html=True) 
                else:
                    st.caption(f"No additional information found for `{rag_asin}` in `{RECOMMENDATIONS_FILE}`.")
            st.markdown("<hr style='border:2px solid gray'>", unsafe_allow_html=True)

elif 'last_results' not in st.session_state:
     st.info("Use the 'Find Products' button in the sidebar to perform a search.")

st.sidebar.markdown("---")
st.sidebar.info(f"Semantic Model: {MODEL_NAME}")
st.sidebar.info("Vector DB: ChromaDB")
st.sidebar.info("LLM Model: meta-llama/Meta-Llama-3-8B-Instruct")
st.sidebar.info("Recommendation Model: GraphSAGE")
st.sidebar.markdown("---")
st.sidebar.info("Proje Yürütücüsü: Oğulcan AKCA")
st.sidebar.info("Proje Danışmanı: Dr.Öğr.Üyesi Başak Köktürk Güzel ")



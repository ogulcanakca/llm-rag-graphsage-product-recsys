import os
import networkx as nx
import pickle
import sys
import json
import logging
this_dir = os.path.dirname(__file__)
if this_dir not in sys.path:
    sys.path.insert(0, this_dir)

from model import SemanticRecommendationSystem
from utils import list_json_files,data_count_json_files,download_file_from_google_drive, list_datasets,read_datasets,drive_downloader
from config import DATASETS_PATH, MATCHED_KEYWORDS_FILE_ID, MATCHED_KEYWORDS_OUTPUT, PRODUCT_GRAPH_FINAL_FILE_ID, PRODUCT_GRAPH_FINAL_OUTPUT, MODEL_FILE_ID, MODEL_OUTPUT, RECOMMENDED_ASINS, PLOT_PATH,MODEL_PATH, PARAMETER_COMBINATIONS, MODELS, LOSS_FUNCTIONS

def main() -> None:
    directory_path_af = DATASETS_PATH
    json_files_af = list_json_files(directory_path_af)
    data_count_json_files(json_files_af)
    list_datasets(json_files_af,row_count=2,mode=0)
    reviews,meta_product_review = read_datasets(json_files_af)

    matched_keywords_file_id = MATCHED_KEYWORDS_FILE_ID
    download_file_from_google_drive(matched_keywords_file_id, MATCHED_KEYWORDS_OUTPUT)

    with open(MATCHED_KEYWORDS_OUTPUT, 'r') as f:
        matched_keywords = json.load(f)

    product_graph_final_file_id = PRODUCT_GRAPH_FINAL_FILE_ID

    drive_downloader(product_graph_final_file_id, PRODUCT_GRAPH_FINAL_OUTPUT)

    with open(PRODUCT_GRAPH_FINAL_OUTPUT, 'rb') as f:
        G = pickle.load(f)

    model_file_id = MODEL_FILE_ID
    drive_downloader(model_file_id, MODEL_OUTPUT)
    
    selected_matched_keywords = {
    asin: keywords for asin, keywords in matched_keywords.items() if len(keywords) >= 7}
    asin_keywords = {}
    for asin, keywords in selected_matched_keywords.items():
        asin_keywords[asin] = list(keywords)
        
    nx.set_node_attributes(G, asin_keywords, name='keywords')
    
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    model_path = MODEL_PATH

    recommendation_system = SemanticRecommendationSystem(
        graph=G,
        reduced_keywords=matched_keywords,
        meta_product_review=meta_product_review
    )
    
    asin_keys = list(selected_matched_keywords.keys())
    total = len(asin_keys)

    for idx, asin in enumerate(asin_keys):
        logging.info(f"Processing product {idx + 1}/{total}: ASIN {asin}")
        print(f"Processing product {idx + 1}/{total}: ASIN {asin}")

        if not os.path.exists(model_path):
            recommendation_system.train_model(
                asin,
                model_name=MODELS[0],
                params=PARAMETER_COMBINATIONS[0],
                loss_functions=LOSS_FUNCTIONS
            )
            recommended_asins, plot_path = recommendation_system.get_recommendations(asin)
        else:
            recommendation_system.load_model(
                model_path,
                in_channels=384,
                params={
                    'hidden_channels': PARAMETER_COMBINATIONS[0]['hidden_channels'],
                    'num_layers': PARAMETER_COMBINATIONS[0]['num_layers'],
                    'dropout': PARAMETER_COMBINATIONS[0]['dropout']
                }
            )
            recommendation_system.setup_logging(
                  asin,
                  model_name=MODELS[0],
                  loss_function=LOSS_FUNCTIONS[list(LOSS_FUNCTIONS.keys())[0]]
              )
            recommended_asins, plot_path = recommendation_system.get_recommendations(asin)
if __name__ == "__main__":
    main()

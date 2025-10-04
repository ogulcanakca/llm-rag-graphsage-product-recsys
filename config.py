DATASETS_PATH = "data"
MATCHED_KEYWORDS_FILE_ID = '18iedcFVTBkNKSI4ahyQvWyVdT-Uu8sQF'
MATCHED_KEYWORDS_OUTPUT = 'matched_keywords.json'
PRODUCT_GRAPH_FINAL_FILE_ID = '1M3LYaSKc4-bp7S27TT9WwZZgtzYYZ2lr'
PRODUCT_GRAPH_FINAL_OUTPUT = 'product_graph_final.gpickle'
MODEL_FILE_ID = '1puHGSTynSlMtUrpGHM28q6EkGWEga3lD'
MODEL_OUTPUT = 'model.pt'
RECOMMENDED_ASINS = ''
PLOT_PATH = ''
MODEL_PATH = f"model.pt"
PARAMETER_COMBINATIONS = [
        {'hidden_channels': 1024, 'learning_rate': 0.001, 'num_epochs': 100, 'dropout': 0.3, 'num_layers': 10}
    ]
MODELS = ['paraphrase-MiniLM-L6-v2']
LOSS_FUNCTIONS = {'Unsupervised Laplacian Regularization Loss': 'ulrl'}

DATA_FILE_NEW = "data/new.json"
RECOMMENDATIONS_FILE = "data/product_recommendations.json"
PRODUCT_METADATA_DB = "data/product_metadata.db"

CHROMA_DB_PATH = "data/chroma_db_product_keywords"
COLLECTION_NAME = "product_keywords_collection"
MODEL_NAME = 'paraphrase-MiniLM-L6-v2'

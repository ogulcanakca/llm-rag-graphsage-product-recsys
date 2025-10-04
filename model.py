import os
import logging
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import networkx as nx
import requests
from PIL import Image
from io import BytesIO
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
from torch_geometric.nn import GraphSAGE, SAGEConv
from torch_geometric.data import Data
from torch_geometric.nn import BatchNorm
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from regularization import EarlyStopping

class SemanticRecommendationSystem:
    def __init__(self,
                 graph: nx.Graph,
                 reduced_keywords: Dict[str, List[str]],
                 meta_product_review: List[Dict[str, Any]],
                 model_name: str = "paraphrase-MiniLM-L6-v2",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.graph = graph
        self.reduced_keywords = reduced_keywords
        self.meta_product_review = meta_product_review
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.model = None
        self.node_embeddings = None
        self.device = device
        self.semantic_model = SentenceTransformer(model_name,device=self.device)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class UnsupervisedGraphSAGE(nn.Module):
        def __init__(self,
                    in_channels: int,
                    hidden_channels: int,
                    out_channels: int,
                    num_layers: int = 3,
                    dropout: float = 0.5,
                    activation_fn: nn.Module = nn.ReLU(),
                    aggregation: str = 'mean'):

            super().__init__()
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            self.dropout = nn.Dropout(dropout)
            self.activation = activation_fn

            self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggregation))
            self.norms.append(BatchNorm(hidden_channels))

            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggregation))
                self.norms.append(BatchNorm(hidden_channels))

            self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggregation))

        def forward(self, x, edge_index):
          for i, conv in enumerate(self.convs):
              x_in = x
              x = conv(x, edge_index)
              if i < len(self.convs) - 1:
                  x = self.norms[i](x)
                  x = self.activation(x)
                  x = self.dropout(x)
                  if x.shape == x_in.shape:
                      x = x + x_in
          return x

    def setup_logging(self, asin:str ,model_name: str, loss_function: str) -> str:

        log_dir = f'logs_{model_name}_{loss_function}'
        os.makedirs(log_dir, exist_ok=True)

        log_filename = os.path.join(log_dir, f'training_log_{model_name}_{loss_function}.txt')
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            force=True
        )
        return log_filename

    def cosine_similarity(self, embedding: torch.Tensor, other_embeddings: torch.Tensor) -> torch.Tensor:
      device = embedding.device
      other_embeddings = other_embeddings.to(device)

      embedding = F.normalize(embedding, p=2, dim=0)
      other_embeddings = F.normalize(other_embeddings, p=2, dim=1)

      return torch.mm(embedding.unsqueeze(0), other_embeddings.t()).squeeze()

    def find_image_url(self, product: Dict[str, Any]) -> Optional[str]:

        if 'imageURLHighRes' in product and product['imageURLHighRes']:
            return product['imageURLHighRes'][0]
        elif 'imageURL' in product and product['imageURL']:
            return product['imageURL'][0]
        return None

    def recommend_products(self,
                       node_id: str,
                       node_embeddings: Dict[str, np.ndarray],
                       model_name:str,
                       top_k: int = 4) -> List[str]:

        semantic_model = self.semantic_model
        if node_id not in node_embeddings:
            logging.info("---------------------------------------------------------")
            logging.error(f"Referans node {node_id} node_embeddings'de bulunamadı")
            return []

        embedding = torch.tensor(node_embeddings[node_id]).to(self.device)

        product_nodes = [
            n for n in self.graph.nodes()
            if n in self.reduced_keywords.keys()
            and n in node_embeddings
        ]

        product_embeddings = torch.tensor([node_embeddings[asin] for asin in product_nodes]).to(self.device)

        keyword_similarities = self.cosine_similarity(embedding, product_embeddings)

        anchor_title = next((prod.get('title', '') for prod in self.meta_product_review if prod.get('asin') == node_id), '')
        candidate_titles = [next((prod.get('title', '') for prod in self.meta_product_review if prod.get('asin') == asin), '') for asin in product_nodes]

        with torch.no_grad():
            anchor_title_embedding = torch.tensor(semantic_model.encode([anchor_title], show_progress_bar=False)[0]).to(self.device)
            candidate_title_embeddings = torch.tensor(semantic_model.encode(candidate_titles, show_progress_bar=False)).to(self.device)

        title_similarities = self.cosine_similarity(anchor_title_embedding, candidate_title_embeddings)

        combined_scores = 0.2 * keyword_similarities + 0.8 * title_similarities

        similarities_cpu = combined_scores.cpu()
        similar_node_indices = torch.argsort(similarities_cpu, descending=True)
        similar_node_indices = similar_node_indices.numpy()

        similar_node_indices = [
            i for i in similar_node_indices
            if product_nodes[i] != node_id
        ][:top_k]

        similar_asins = [product_nodes[i] for i in similar_node_indices]

        logging.info("---------------------------------------------------------")
        logging.info(f"Keywords of anchor product (ASIN: {node_id}): {self.reduced_keywords[node_id]}\n")
        for asin in similar_asins:
            logging.info("---------------------------------------------------------")
            logging.info(f"Keywords of similar product (ASIN: {asin}): {self.reduced_keywords[asin]}\n")

        return similar_asins

    def plot_recommendations(self,
                     node_id: str,
                     recommended_asins: List[str],
                     model_name: str,
                     loss_function: str,
                     hidden_channels: int = 128,
                     learning_rate: float = 0.001,
                     num_epochs: int = 20,
                     dropout: float = 0.5,
                     num_layers: int = 3,
                     top_k: int = 4) -> str:

      plots_dir = 'plots'
      os.makedirs(plots_dir, exist_ok=True)

      asin_list = [node_id] + recommended_asins
      image_data = []

      for asin in asin_list:
          for product in self.meta_product_review:
              if product.get('asin') == asin:
                  image_url = self.find_image_url(product)
                  if image_url:
                      try:
                          response = requests.get(image_url)
                          img = Image.open(BytesIO(response.content))
                          image_data.append((asin, img))
                      except Exception as e:
                          logging.error(f"Error loading image for ASIN {asin}: {e}")
                          image_data.append((asin, None))
                  break
          else:
              logging.error(f"ASIN {asin} not found in meta_product_review")
              image_data.append((asin, None))

      num_images = min(len(image_data), top_k * 2)
      if num_images == 0:
          logging.error("No images to display")
          return ""

      if num_images == 1:
          fig, axes = plt.subplots(1, 1, figsize=(20, 5))
          axes = [axes]
      else:
          fig, axes = plt.subplots(1, num_images, figsize=(20, 5))

      for ax, (asin, img) in zip(axes, image_data):
          if img is not None:
              ax.imshow(img)
              ax.set_title(f"ASIN: {asin}", pad=20)
          else:
              ax.set_title(f"ASIN: {asin}\nNo Image", pad=20)
          ax.axis('off')

      plt.subplots_adjust(wspace=0.7)
      plt.tight_layout()

      try:
          plot_name = f'recommendations_{node_id}_{model_name}_{hidden_channels}_{learning_rate}_{num_epochs}_{dropout}_{num_layers}.png'
          full_save_path = os.path.join(plots_dir, plot_name)
          plt.savefig(full_save_path, bbox_inches='tight')
          logging.info(f"Plot saved as {full_save_path}")
          plt.close()
          return full_save_path
      except Exception as e:
          logging.error(f"Error saving plot: {e}")
          plt.close()
          return ""

    def get_recommended_asins(self) -> List[str]:
        return self.recommended_asins

    def ulr_loss(self, node_embeddings, graph):
        node_embeddings = F.normalize(node_embeddings, p=2, dim=1)

        L = nx.laplacian_matrix(graph)
        indices = torch.LongTensor(np.vstack(L.nonzero()))
        values = torch.FloatTensor(L.data)

        L_sparse = torch.sparse_coo_tensor(
            indices, values,
            size=(graph.number_of_nodes(), graph.number_of_nodes())
        ).to(node_embeddings.device)

        loss = torch.sparse.mm(L_sparse, node_embeddings)
        loss = torch.sum(node_embeddings * loss)

        return loss

    def calculate_loss(self,loss_type, output, data=None):
      loss_functions = {
          'ulrl': lambda: self.ulr_loss(output, self.graph)
      }
      if loss_type not in loss_functions.keys():
          raise logging.info(f"Unknown loss type: {loss_type}")

      return loss_functions[loss_type]()

    def save_model(self, model_path: str):
        if self.model is None:
            raise ValueError("No model to save")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'node_embeddings': self.node_embeddings
        }, model_path)

    def load_model(self, model_path: str, in_channels: int, params: Dict[str, Any]):
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self.model = self.UnsupervisedGraphSAGE(
            in_channels,
            params['hidden_channels'],
            in_channels,
            num_layers=params['num_layers'],
            dropout=params['dropout']
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.node_embeddings = checkpoint['node_embeddings']
        self.model.eval()

    def train_model(self,
                asin: str,
                model_name: str,
                params: Dict[str, Any],
                loss_functions: Dict[str,str],
                node_embeddings: Optional[Dict[str, np.ndarray]] = None) -> None:

      for loss_function in loss_functions.keys():
          torch.cuda.empty_cache()
          self.setup_logging(asin,model_name,loss_functions[loss_function])

          logging.info("---------------------------------------------------------")
          logging.info(f"Semantic Search Model: {model_name}")
          logging.info(f"\nStarting training with parameters:")
          logging.info("---------------------------------------------------------")
          logging.info(f"Hidden Channels: {params['hidden_channels']}")
          logging.info(f"Learning Rate: {params['learning_rate']}")
          logging.info(f"Num Epochs: {params['num_epochs']}")
          logging.info(f"Dropout: {params['dropout']}")
          logging.info(f"Num Layers: {params['num_layers']}")
          logging.info("---------------------------------------------------------")

          if node_embeddings is None:
              self.node_embeddings = {}
              for node in self.graph.nodes():
                  keyword_embedding = self.semantic_model.encode([node], show_progress_bar=False)[0]
                  self.node_embeddings[node] = keyword_embedding
          else:
              self.node_embeddings = node_embeddings

          for node in self.graph.nodes():
              if node in self.node_embeddings:
                  self.graph.nodes[node]['embedding'] = self.node_embeddings[node]
              else:
                  keyword_embedding = self.semantic_model.encode([node], show_progress_bar=False)[0]
                  self.graph.nodes[node]['embedding'] = keyword_embedding

          node_features = torch.tensor(
              np.array([self.graph.nodes[node]['embedding'] for node in self.graph.nodes()]),
              dtype=torch.float
          )
          edge_index = torch.tensor(
              [[list(self.graph.nodes).index(u), list(self.graph.nodes).index(v)] for u, v in self.graph.edges],
              dtype=torch.long
          ).t().contiguous()

          data = Data(x=node_features, edge_index=edge_index).to(self.device)

          in_channels = node_features.shape[1]
          out_channels = in_channels
          self.model = self.UnsupervisedGraphSAGE(
              in_channels,
              params['hidden_channels'],
              out_channels,
              num_layers=params['num_layers'],
              dropout=params['dropout']
          ).to(self.device)
          optimizer = torch.optim.Adam(self.model.parameters(), lr=params['learning_rate'])

          self.model.train()
          early_stopping = EarlyStopping(patience=3, min_delta=0.01)

          for epoch in range(params['num_epochs']):
              optimizer.zero_grad()
              output = self.model(data.x, data.edge_index)
              loss = self.calculate_loss(loss_functions[loss_function], output, data)
              loss.backward()
              torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
              optimizer.step()

              logging.info(f"Epoch {epoch+1}, {loss_function}: {loss.item()}")
              print(f"Epoch {epoch+1}, {loss_function}: {loss.item()}")
              early_stopping(loss.item())
              if early_stopping.early_stop:
                  logging.info(f"Early stopping triggered at epoch {epoch+1}")
                  break

          model_path = f'model.pt'
          self.save_model(model_path)


    def get_recommendations(self, asin: str, top_k: int = 4,
                        model_name: str = "paraphrase-MiniLM-L6-v2",
                        loss_function: str = "ulrl") -> Tuple[List[str], str]:

      if self.model is None:
          raise ValueError("Model needs to be trained or loaded first")

      try:
          recommended_asins = self.recommend_products(
              node_id=asin,
              node_embeddings=self.node_embeddings,
              top_k=top_k,
              model_name = model_name
          )

          plot_path = self.plot_recommendations(
              node_id=asin,
              recommended_asins=recommended_asins,
              model_name=model_name,
              loss_function=loss_function,
              hidden_channels=1024,
              top_k=top_k
          )

          return recommended_asins, plot_path
      except Exception as e:
          logging.error(f"Error in get_recommendations: {e}")
          return [], ""

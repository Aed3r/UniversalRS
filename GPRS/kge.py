import os
from pykeen.pipeline import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from pykeen.datasets import DBpedia50
from pykeen.triples import TriplesFactory
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import torch

USE_SAVED = True
SAVE_PATH = './res'

def evaluate_recommendations(true_items, recommended_items):
    precision = precision_score(true_items, recommended_items, average='micro')
    recall = recall_score(true_items, recommended_items, average='micro')
    f1 = f1_score(true_items, recommended_items, average='micro')
    return precision, recall, f1

# find similar items using both entity and relationship embeddings
def find_similar_items(item_index, entity_embeddings, relation_embeddings, top_k=5):
    item_embedding = entity_embeddings[item_index]

    # Compute similarities based on entity embeddings
    entity_similarities = cosine_similarity(item_embedding.unsqueeze(0).detach().numpy(), entity_embeddings.detach().numpy())
    entity_similarities = entity_similarities.flatten()

    top_indices = entity_similarities.argsort()[-(top_k+1):][::-1]

    # Remove the item itself from the similar items
    top_indices = top_indices[top_indices != item_index.detach().numpy()]

    return top_indices

data = DBpedia50()

if not USE_SAVED:
    pipeline_result = pipeline(
        dataset=data,
        model='TransE',
        training_loop='sLCWA', # alternatives: 'sLCWA', 'LCWA'
        negative_sampler='basic', # alternatives: 'basic', 'bernoulli', 'corrupt'
        evaluator='RankBasedEvaluator', # alternatives: 'RankBasedEvaluator', 'AbsoluteRankBasedEvaluator', 'LCWAEvaluator'
    )

    # Save results
    pipeline_result.save_to_directory(SAVE_PATH)

    # Get embeddings
    model = pipeline_result.model
else:
    model_path = os.path.join(SAVE_PATH, "trained_model.pkl")
    model = torch.load(model_path)

entity_representation_modules = model.entity_representations
relation_representation_modules = model.relation_representations

entity_embeddings = entity_representation_modules[0]
relation_embeddings = relation_representation_modules[0]

entity_embedding_tensor = entity_embeddings()
relation_embedding_tensor = relation_embeddings()

#entity_embeddings_array = model.entity_representations[0](indices=None).detach().numpy()
#relation_embeddings_array = model.relation_representations[0](indices=None).detach().numpy()

if __name__ == '__main__':
    # Test
    #item_index = 0 
    #similar_items = find_similar_items(item_index, entity_embedding_tensor, relation_embedding_tensor)

    # Print the top 5 most similar items
    #for idx in similar_items:
        #print(f"Similar item: {idx}")

    true_items = data.testing.mapped_triples
    recommended_items = []
    for item_index, _, _ in true_items:
        similar_items = find_similar_items(item_index, entity_embedding_tensor, relation_embedding_tensor)
        recommended_items.append(similar_items)

    true_items = [item_index for item_index, _, _ in true_items]

    true_items = np.array(true_items)
    recommended_items = np.array(recommended_items)

    precision, recall, f1 = evaluate_recommendations(true_items, recommended_items)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
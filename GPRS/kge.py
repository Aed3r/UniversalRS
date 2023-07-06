from pykeen.pipeline import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from pykeen.datasets import DBpedia50

pipeline_result = pipeline(
    dataset=DBpedia50,
    model='TransE',
    training_loop='sLCWA', # alternatives: 'sLCWA', 'LCWA'
    negative_sampler='basic', # alternatives: 'basic', 'bernoulli', 'corrupt'
    evaluator='RankBasedEvaluator', # alternatives: 'RankBasedEvaluator', 'AbsoluteRankBasedEvaluator', 'LCWAEvaluator'
)

pipeline_result.save_to_directory('./res')

model = pipeline_result.model
entity_representation_modules = model.entity_representations
relation_representation_modules = model.relation_representations

entity_embeddings = entity_representation_modules[0]
relation_embeddings = relation_representation_modules[0]

entity_embedding_tensor = entity_embeddings()
relation_embedding_tensor = relation_embeddings()

#entity_embeddings_array = model.entity_representations[0](indices=None).detach().numpy()
#relation_embeddings_array = model.relation_representations[0](indices=None).detach().numpy()

# find similar items using both entity and relationship embeddings
def find_similar_items(item_index, entity_embeddings, relation_embeddings, top_k=5):
    item_embedding = entity_embeddings[item_index]

    # Compute similarities based on entity embeddings
    entity_similarities = cosine_similarity(item_embedding.unsqueeze(0).detach().numpy(), entity_embeddings.detach().numpy())
    entity_similarities = entity_similarities.flatten()

    top_indices = entity_similarities.argsort()[-top_k:][::-1]

    return top_indices

# Test
item_index = 0 
similar_items = find_similar_items(item_index, entity_embedding_tensor, relation_embedding_tensor)

# Print the top 5 most similar items
for idx in similar_items:
    print(f"Similar item: {idx}")
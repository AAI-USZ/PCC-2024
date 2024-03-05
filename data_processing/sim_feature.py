import os
import argparse
import re
from patch_util import read_patches
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from scipy import spatial


# Splits the input string by the given pattern
def splitter(s):
    tokens = []
    splitted = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', s)
    for newtoken in splitted:
        tokens.append(newtoken.lower())
    return tokens


in_path = "./data/patches_for_static/"
out_path = "./data"

patches = read_patches(in_path)
train_sentences = ["".join(p.source).replace("\n", " ") for p in patches]
train_sentences.extend(["".join(p.target).replace("\n", " ") for p in patches])

train_corpus = [TaggedDocument(words=splitter(s), tags=[str(i)]) for i, s in enumerate(train_sentences)]

epochs = 20
vec_size = 256
window_size = 10
min_count = 1

model = Doc2Vec(vector_size=vec_size,  min_count=min_count, epochs=epochs, workers=8, window=window_size)
model.build_vocab(train_corpus, keep_raw_vocab=True)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

print("Counting similarities...")
with open(f"{out_path}/sims.csv", "w", encoding="utf-8") as results:
    results.write("patch_name,cosine_distance,braycurtis_distanc},canberra_distance,chebyshev_distance,cityblock_distance,"
                  "euclidean_distance,jensenshannon_distance,minkowski_distance,seuclidean_distance,"
                  f"{',vec_dim_'.join(map(str, list(range(vec_size))))}\n")
    for patch in patches:
        tokenized_program = splitter("".join(patch.source).replace("\n", " "))
        tokenized_patched_program = splitter("".join(patch.target).replace("\n", " "))

        vec1 = model.infer_vector(tokenized_program)
        vec2 = model.infer_vector(tokenized_patched_program)

        cosine_distance = spatial.distance.cosine(vec1, vec2)
        braycurtis_distance = spatial.distance.braycurtis(vec1, vec2)
        canberra_distance = spatial.distance.canberra(vec1, vec2)
        chebyshev_distance = spatial.distance.chebyshev(vec1, vec2)
        cityblock_distance = spatial.distance.cityblock(vec1, vec2)
        euclidean_distance = spatial.distance.euclidean(vec1, vec2)
        jensenshannon_distance = spatial.distance.jensenshannon(vec1, vec2)
        minkowski_distance = spatial.distance.minkowski(vec1, vec2)
        seuclidean_distance = spatial.distance.seuclidean(vec1, vec2, [0.01] * vec_size)

        results.write(f"{patch.project},{cosine_distance},{braycurtis_distance},{canberra_distance},"
                      f"{chebyshev_distance},{cityblock_distance},{euclidean_distance},{jensenshannon_distance},"
                      f"{minkowski_distance},{seuclidean_distance},{','.join(map(str, vec2))}\n")

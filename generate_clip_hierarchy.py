import torch
import clip
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering


def generate_clip_clusters():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP (ViT-B/32) on {device}...")
    model, _ = clip.load("ViT-B/32", device=device)

    rel_classes = [
        "above",  # 0
        "across",  # 1
        "against",  # 2
        "along",  # 3
        "and",  # 4
        "at",  # 5
        "attached to",  # 6
        "behind",  # 7
        "belonging to",  # 8
        "between",  # 9
        "carrying",  # 10
        "covered in",  # 11
        "covering",  # 12
        "eating",  # 13
        "flying in",  # 14
        "for",  # 15
        "from",  # 16
        "growing on",  # 17
        "hanging from",  # 18
        "has",  # 19
        "holding",  # 20
        "in",  # 21
        "in front of",  # 22
        "laying on",  # 23
        "looking at",  # 24
        "lying on",  # 25
        "made of",  # 26
        "mounted on",  # 27
        "near",  # 28
        "of",  # 29
        "on",  # 30
        "on back of",  # 31
        "over",  # 32
        "painted on",  # 33
        "parked on",  # 34
        "part of",  # 35
        "playing",  # 36
        "riding",  # 37
        "says",  # 38
        "sitting on",  # 39
        "standing on",  # 40
        "to",  # 41
        "under",  # 42
        "using",  # 43
        "walking in",  # 44
        "walking on",  # 45
        "watching",  # 46
        "wearing",  # 47
        "wears",  # 48
        "with",  # 49
    ]

    templates = [
        lambda c: f"a photo of interaction where the subject is {c} the object",
    ]

    print(
        f"Extracting features for {len(rel_classes)} relations using {len(templates)} prompt templates..."
    )

    all_features = []

    with torch.no_grad():
        for rel in rel_classes:
            prompts = [t(rel) for t in templates]

            text_inputs = clip.tokenize(prompts).to(device)

            features = model.encode_text(text_inputs)

            features /= features.norm(dim=-1, keepdim=True)

            mean_feature = features.mean(dim=0)

            mean_feature /= mean_feature.norm()

            all_features.append(mean_feature.cpu().numpy())

    X = np.array(all_features)

    print("Clustering into 3 semantic families...")
    kmeans = KMeans(n_clusters=3, random_state=41, n_init=50)
    labels = kmeans.fit_predict(X)


    print("\n" + "=" * 40)
    print("GENERATED SUPER RELATION MAP")
    print("=" * 40)
    print(labels.tolist())
    print("\n" + "=" * 40)

    print("Cluster Interpretation (Sanity Check):")
    for i in range(3):
        print(f"\n--- Family {i} ---")
        cluster_members = [
            rel_classes[j] for j in range(len(rel_classes)) if labels[j] == i
        ]
        print(f"Count: {len(cluster_members)}")
        print(", ".join(cluster_members))


if __name__ == "__main__":
    generate_clip_clusters()

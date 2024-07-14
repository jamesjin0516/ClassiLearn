import argparse
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--features-file", type=str, required=True, help="Path to gz file of features")
    parser.add_argument("-c", "--characterization", type=str, choices=["PCA", "LDA"], help="Type of feature characterization to use")
    parser.add_argument("-t", "--feature-type", type=str, help="Name of feature stored; if not set, applies early fusion")
    return parser.parse_args()


def pca(charmat, maxk):
    covmat = np.cov(charmat)
    ei_vals, ei_vecs = np.linalg.eig(covmat)
    eival_ord = ei_vecs[:, np.argsort(ei_vals)[-maxk:]]
    return eival_ord.T @ charmat


def lda(charmat, clas):
    pass


def apply_characterization(charmat, labels):
    # result_pca = pca(charmat.T, 50)
    sci_pca = PCA(n_components=2)
    norm_charmat = StandardScaler().fit_transform(charmat)
    result_pca = sci_pca.fit_transform(norm_charmat)

    fig, axes_pca = plt.subplots(figsize=(17.5, 10))
    axes_pca.scatter(result_pca[labels == 0, 0], result_pca[labels == 0, 1], color="blue", label="low")
    axes_pca.scatter(result_pca[labels == 1, 0], result_pca[labels == 1, 1], color="red", label="high")
    # axes_pca[1].scatter(sci_pca.components_[0][np.where(labels == 0)], sci_pca.components_[1][np.where(labels == 0)], color="blue")
    # axes_pca[1].scatter(sci_pca.components_[0][np.where(labels == 1)], sci_pca.components_[1][np.where(labels == 1)], color="red")
    axes_pca.set_title("PCA", fontsize=20)
    axes_pca.legend(fontsize=20)
    axes_pca.tick_params(which="both", labelsize=20)
    fig.savefig(os.path.join("tutorial", "PCA_Characteristics.png"))
    fig.clear()

    sci_lda = LinearDiscriminantAnalysis()
    result_lda = sci_lda.fit_transform(charmat, labels)
    result_lda = np.concatenate((result_lda, np.ones((result_lda.shape[0], 1))), 1)
    plt.scatter(result_lda[labels == 0, 0], result_lda[labels == 0, 1], color="blue", label="low")
    plt.scatter(result_lda[labels == 1, 0], result_lda[labels == 1, 1], color="red", label="high")
    plt.legend(fontsize=20)
    plt.title("LDA", fontsize=20)
    plt.tick_params(which="both", labelsize=20)
    plt.savefig(os.path.join("tutorial", "LDA_Characteristics.png"))
    plt.clf()

    return result_pca, result_lda


if __name__ == "__main__":
    args = get_arguments()
    all_feats = joblib.load(args.features_file)
    feats = set(all_feats.keys()) - {"labels"}
    features = all_feats[args.feature_type] if args.feature_type else np.concatenate([all_feats[feat_type] for feat_type in feats], 1)
    if args.characterization:
        result_pca, result_lda = apply_characterization(features, all_feats["labels"])
        features = result_pca if args.characterization == "PCA" else result_lda
    joblib.dump({"features": features, "labels": all_feats["labels"]}, "simplified_characteristics.gz")

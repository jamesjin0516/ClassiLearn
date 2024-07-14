import argparse
from facemesh.facemesh_character import frame_level_features, frame_level_labels, frame_to_vid_probs, frame_to_vid_preds
import joblib
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from statistics import mean, mode, stdev


def get_arguments():
    parser = argparse.ArgumentParser()
    fusion_parser = parser.add_argument_group("Late Fusion", "Enable various ways to combine individual decisions")
    fusion_parser.add_argument("-cl", "--classifier-late-fusion", action="store_true", help="Whether to enable late fusion between classifier decisions")
    fusion_parser.add_argument("-fl", "--feature-late-fusion", action="store_true", help="Whether each item to classifier has multiple features, which undergo late fusion")
    input_parser = parser.add_argument_group("Input Control", "Customize sources of featuers to classify")
    input_parser.add_argument("-f", "--features-file", default="simplified_characteristics.gz", type=str, help="Path to gz file of features to classify")
    input_parser.add_argument("-t", "--feature-type", nargs="+", default=["features"], help="If set, applies probability fusion on specified feature types, \
                        else uses one feature type under key \"features\"")
    return parser.parse_args()


def getGridSearchInstances():
    return {"KNN": GridSearchCV(KNeighborsClassifier(n_jobs=-1), {"n_neighbors": [3, 4, 5, 6, 7]}, n_jobs=-1),
            "SVM": GridSearchCV(SVC(probability=True), [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                                        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}], n_jobs=-1),
            "RF": GridSearchCV(RandomForestClassifier(), {"n_estimators": [50, 75, 100, 125, 150]}, n_jobs=-1)}


def evaluate(exp_num, feats_train, feats_test, labels_train, labels_test, late_fusion):
    print(f"Experiment #{exp_num}. feats train: {feats_train.shape}; feats test: {feats_test.shape}")
    feature_scalar = StandardScaler()
    feats_train = feature_scalar.fit_transform(feats_train)
    feats_test = feature_scalar.transform(feats_test)

    # Visualize the training and testing groups
    if feats_train.shape == (labels_train.shape[0], 2) and feats_test.shape == (labels_test.shape[0], 2):
        fig, axes_pca = plt.subplots(1, 2, figsize=(20, 10))
        axes_pca[0].scatter(feats_train[labels_train == 0, 0], feats_train[labels_train == 0, 1], color="blue", label="control")
        axes_pca[0].scatter(feats_train[labels_train == 1, 0], feats_train[labels_train == 1, 1], color="red", label="test")
        axes_pca[1].scatter(feats_test[labels_test == 0, 0], feats_test[labels_test == 0, 1], color="blue", label="control")
        axes_pca[1].scatter(feats_test[labels_test == 1, 0], feats_test[labels_test == 1, 1], color="red", label="test")
        axes_pca[0].set_title("train data")
        axes_pca[1].set_title("test data")
        axes_pca[0].legend()
        axes_pca[1].legend()
        fig.savefig(f"training_data_#{exp_num}.png", bbox_inches="tight")
        fig.clear()
    else:
        print(f"For plotting train test groups, training features should be {labels_train.shape[0], 2} (now {feats_train.shape});"
              f" testing features should be {labels_test.shape[0], 2} (now {feats_test.shape})")

    # Same evaluation process for all classifiers
    methods = getGridSearchInstances()
    best_params = dict()
    train_probabilities, class_probabilities, class_predictions = dict(), dict(), dict()
    for name, method in methods.items():
        print(f"{name}......")
        # For each classifier, fit and predict with testing data
        method.fit(feats_train, labels_train)
        class_predictions[name] = method.predict(feats_test)
        class_probabilities[name] = method.predict_proba(feats_test)
        train_probabilities[name] = method.predict_proba(feats_train)
        best_params[name] = method.best_params_.copy()
    # Perform late fusion
    if late_fusion:
        class_predictions["late fusion (mean)"] = np.array([1 if mean([class_probabilities[name][samp][0] for name in class_probabilities]) < 0.5
                                                            else 0 for samp in range(feats_test.shape[0])])
        class_predictions["late fusion (certainty)"] = np.array([1 if max([class_probabilities[name][samp][0] for name in class_probabilities], key=lambda x: abs(x - 0.5)) < 0.5
                                                                 else 0 for samp in range(feats_test.shape[0])])
    return train_probabilities, class_probabilities, class_predictions, best_params


def calculate_performance(exp_num, class_probabilities, class_predictions, feats_test, labels_test, best_params, axe_prob):
    score = dict()
    if feats_test.shape == (labels_test.shape[0], 2): fig, axes_pred = plt.subplots(1, len(class_predictions), figsize=(len(class_predictions) * 10, 10))
    for ind, (name, predictions) in enumerate(class_predictions.items()):
        # Calculate parameters of correctness
        correct_control = predictions[labels_test == 0] == labels_test[labels_test == 0]
        correct_test = predictions[labels_test == 1] == labels_test[labels_test == 1]
        score[name] = dict(CCR=np.sum(predictions == labels_test) / labels_test.shape[0], S=np.sum(correct_control) / np.count_nonzero(labels_test == 0),
                      E=np.sum(correct_test) / np.count_nonzero(labels_test == 1))
        # Visualize predictions
        if feats_test.shape == (labels_test.shape[0], 2):
            axes_pred[ind].scatter(np.mean(feats_test[labels_test == 0, 0]), np.mean(feats_test[labels_test == 0, 1]), color="blue", label="control (errors hollow)",
                facecolors=np.where(correct_control, "blue", "none"))
            axes_pred[ind].scatter(np.mean(feats_test[labels_test == 1, 0]), np.mean(feats_test[labels_test == 1, 1]), color="red", label="test (errors hollow)",
                facecolors=np.where(correct_test, "red", "none"))
            if name in best_params:
                axes_pred[ind].text(0.95, 0.05, f"Best parameters: {json.dumps(best_params[name])}", horizontalalignment="right", verticalalignment="bottom",
                                    transform=axes_pred[ind].transAxes, fontsize=15)
            axes_pred[ind].legend(fontsize=15)
            axe_prob.tick_params(which="both", labelsize=20)
            axes_pred[ind].set_title(f"{name} results", fontsize=20)
    if feats_test.shape == (labels_test.shape[0], 2):
        print("Presenting final graphical results...")
        fig.savefig(f"evaluation_results_#{exp_num}.png", bbox_inches="tight")
        fig.clear()
    else:
        print(f"For plotting evaluation results, testing features should be {labels_test.shape[0], 2} (now {feats_test.shape})")
    # Graph the 3D point cloud from each sample's predicted probabilities by different classifiers
    axe_prob.scatter(*[class_probabilities[name][labels_test == 0, 1] for name in class_probabilities], s=200, color="blue", label="control (errors hollow)" if exp_num == 0 else None,
                     facecolors=np.where(correct_control, "blue", "none"))
    axe_prob.scatter(*[class_probabilities[name][labels_test == 1, 1] for name in class_probabilities], s=200, color="red", label="test (errors hollow)" if exp_num == 0 else None,
                     facecolors=np.where(correct_test, "red", "none"))
    axe_prob.set_xlabel(list(class_probabilities.keys())[0], fontsize=20, labelpad=10)
    axe_prob.set_ylabel(list(class_probabilities.keys())[1], fontsize=20, labelpad=10)
    axe_prob.set_zlabel(list(class_probabilities.keys())[2], fontsize=20, labelpad=10)
    return score, best_params


if __name__ == "__main__":
    args = get_arguments()
    simple_indexing, identity = lambda data, _, indexes: data[indexes], lambda any_data, *args: any_data
    get_features, get_labels = (frame_level_features, frame_level_labels) if args.feature_late_fusion else (simple_indexing, simple_indexing)
    postproc_probs, postproc_preds = (frame_to_vid_probs, frame_to_vid_preds) if args.feature_late_fusion else (identity, identity)
    # Restore PCA processed data, and split into training and testing groups
    all_data = joblib.load(args.features_file)

    features_collection = [all_data[feat_type] for feat_type in args.feature_type]
    labels = all_data["labels"]

    os.makedirs("classification_output", exist_ok=True)
    os.chdir("classification_output")
    # Cross validation with different partitions of the same data (default 5)
    cross_val = StratifiedKFold()
    fig_prob = plt.figure(figsize=(20, 20))
    axe_prob = fig_prob.add_subplot(projection="3d")
    exp_scores, comb_scores = [], dict()
    exp_params = []
    for exp_num, (train_ind, test_ind) in enumerate(cross_val.split(np.zeros(labels.shape[0]), labels)):
        prob_traindata, prob_testdata = [], []
        for features in features_collection:
            feats_train, feats_test, labels_train, labels_test = get_features(features, labels, train_ind), get_features(features, labels, test_ind), get_labels(labels, features, train_ind), get_labels(labels, features, test_ind)
            train_probabilities, class_probabilities, class_predictions, best_params = evaluate(exp_num, feats_train, feats_test, labels_train, labels_test, args.classifier_late_fusion)
            class_probabilities, class_predictions = postproc_probs(class_probabilities, features, test_ind), postproc_preds(class_predictions, features, test_ind)
            if args.feature_type[0] == "features": continue
            train_probabilities = postproc_probs(train_probabilities, features, train_ind)
            prob_train = np.concatenate([train_probabilities[classifier][:, 1][np.newaxis].T for classifier in train_probabilities], 1)
            prob_test = np.concatenate([class_probabilities[classifier][:, 1][np.newaxis].T for classifier in class_probabilities], 1)
            prob_traindata.append(prob_train)
            prob_testdata.append(prob_test)
        labels_train, labels_test = labels[train_ind], labels[test_ind]
        if args.feature_type[0] != "features":
            prob_train = np.concatenate([prob_train for prob_train in prob_traindata], 1) 
            prob_test = np.concatenate([prob_test for prob_test in prob_testdata], 1)
            train_probabilities, class_probabilities, class_predictions, best_params = evaluate(exp_num, prob_train, prob_test, labels_train, labels_test, args.classifier_late_fusion)
        score, best_params = calculate_performance(exp_num, class_probabilities, class_predictions, feats_test, labels_test, best_params, axe_prob)
        exp_scores.append(score)    # Remember each paritition's scores for now
        exp_params.append(best_params)
    axe_prob.legend(fontsize=25)
    axe_prob.tick_params(which="both", labelsize=20)
    axe_prob.set_title("Probabilities of data belonging to test group by classifiers", fontsize=25)
    fig_prob.savefig(f"Probabilities_by_classifiers.png", bbox_inches="tight")
    # Under each classifier's each criteria, collect results from all experiments
    for classifier in exp_scores[0].keys():
        if classifier not in comb_scores: comb_scores[classifier] = dict()
        for criteria in exp_scores[0][classifier]:
            if criteria not in comb_scores[classifier]: comb_scores[classifier][criteria] = []
            for exp_num in range(len(exp_scores)):
                comb_scores[classifier][criteria].append(exp_scores[exp_num][classifier][criteria].item())
    # Attach the best parameters found in each experiment
    for classifier in comb_scores:
        if classifier not in exp_params[0]: continue
        comb_scores[classifier]["best_params"] = []
        for exp_num in range(len(exp_params)):
            comb_scores[classifier]["best_params"].append({f"Experiment #{exp_num}": exp_params[exp_num][classifier]})
    with open("evaluation_all_scores.txt", "w") as all_file:
        json.dump(comb_scores, all_file, indent=4)
    # Find mean and standard average for each classifier's each criteria
    for classifier in comb_scores.keys():
        for criteria in comb_scores[classifier]:
            if criteria == "best_params":
                # Group best parameters for every experiment by their type
                collected_params = dict()
                for param_dict in comb_scores[classifier][criteria]:
                    for param, best_val in list(param_dict.values())[0].items():
                        if param in collected_params:
                            collected_params[param].append(best_val)
                        else:
                            collected_params[param] = [best_val]
                for param in collected_params: collected_params[param] = mode(collected_params[param])
                comb_scores[classifier][criteria] = collected_params
            else:
                scores = comb_scores[classifier][criteria]
                comb_scores[classifier][criteria] = dict(mean=mean(scores), stand_dev=stdev(scores))
    with open("evaluation_scores.txt", "w") as avg_file:
        json.dump(comb_scores, avg_file, indent=4)
    os.chdir("..")

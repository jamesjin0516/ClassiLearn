# ClassiLearn: Streamlined classical classification
This repository modularizes the familiarization with classical, non-ML/AI classifiers using user-provided data, accepting audio recordings of speech and videos of human faces.

## High-Level Usage
1. *Feature extraction*: choose 1 of 2 available data modalities and provide input files to their respective characterization (feature extraction) script.
2. *Optional dimensionality reduction*: apply [PCA](https://scikit-learn.org/stable/modules/decomposition.html#principal-component-analysis-pca) (principle component analysis) and/or [LDA](https://scikit-learn.org/stable/modules/lda_qda.html#lda-qda) (linear discriminant analysis) to the extracted features.
3. *Classification*: select reduced or original features and perform classification with [cross validation](https://scikit-learn.org/stable/modules/cross_validation.html) (CV) and [grid search](https://scikit-learn.org/stable/modules/grid_search.html), with various fusion strategies available.

### 1. Feature Extraction
- audio: [DisVoice](https://github.com/jcvasquezc/DisVoice); extracts multiple types of speech features from audio input. ([details & examples](disvoice/disvoice_features.md))
- video: [Face Mesh](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md) & HOG descriptors; locates facial landmarks and define facial region. ([details & examples](facemesh/facemesh_features.md))
### 2. Dimensionality Reduction
Handled by `dim_reduce.py`. For individual argument description, run `python dim_reduce.py --help`
- Input: manually specify the path to the outputted features of 1 of the modalities
- Configurations: option to choose 1 of PCA or LDA; specification of feature type to preserve from original extraction. option of [early fusion](fusion_strategies.md)
- Output: appropriately selected and simplified features. Whether any configurations are set, output matches default format of classification input.
### 3. Classification
Run `python classification.py --help` for details
- **Default** input: features file formatted by `dim_reduce.py`
- **Custom** input: manually select original features extracted by 1 modality (*omit step 2*)
- Configurations: option of [classifier late fusion](fusion_strategies.md), [feature late fusion](fusion_strategies.md), and [probability fusion](fusion_strategies.md)
- Output (stored and *overwritten* in `classification_output` directory):
    1. classification scores, as both mean & stdev and raw scores across CV folds
    2. graphic of class probabilities by classifiers
    3. graphic of train & test split and correctness within feature space, if 2-dimensional.


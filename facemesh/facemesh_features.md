### Input File Structure
Besides grouping input by class folders ([refer to DisVoice](../disvoice/disvoice_features.md)), Face Mesh and HOG feature extraction requires an additional auxiliary data-enclosing folder within each class folder. Example:
```
facemesh/
├── EXPR_AGITATED
│   └── facecap
│       ├── agitated_sample_1.mp4
│       ├── agitated_sample_2.mp4
│       |── agitated_sample_3.mp4
|       ...
├── EXPR_RELIEVED
│   └── facecap
│       ├── relieved_sample_1.mp4
│       ├── relieved_sample_2.mp4
│       |── relieved_sample_3.mp4
|       ...
├── facedetection.py
├── facemesh_character.py
├── facemesh_features.md
```

### Feature Extraction Usage
Face Mesh only provides the region of interest across frames. The HOG descriptor extract features from videos. Two strategies are implemented:
1. Dense HOG: features are extracted per frame and either averaged or becomes inputs for [feature late fusion](../fusion_strategies.md)
2. Sparse HOG: space-time interest points[^1][^2] across the video yields local features, pooled to video-level descriptors through [Bag of Words](https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html)

Use `python facemesh_character.py --help` for argument details
- Input: must specify class names and the nested auxiliary directory as well
- Configurations: option to use dense HOG over sparse HOG; variant selection for dense HOG; settings for sparse HOG.
- Outputs:video labels & 1 of the following depending on extraction configurations
    - **Video-level** features from dense HOG averaged across fraems
    - **Video-level** features from sparse HOG
    - **Frame-level** features from dense HOG without average (must classify with [feature late fusion](../fusion_strategies.md))

[^1]:Laptev, I. On Space-Time Interest Points. Int J Comput Vision 64, 107–123 (2005).
[^2]:https://github.com/theantimist/action-detection/

### Sparse HOG-Specific Dependencies
#### 1. Space-Time Interest Point Detector 
- Download source: https://www.di.ens.fr/~laptev/download/stip-2.0-linux.zip
- The binary executable needs to be visible within `PATH`: `stip-2.0-linux/bin/stipdet`
#### 2. FFmpeg 0.5.2
*Dependency of OpenCV 2.2.0*
- Download source: https://ffmpeg.org/releases/ffmpeg-0.5.2.tar.bz2
- `$ ./configure` with `--enable-swscale` and `--build-shared` options
#### 3. OpenCV 2.2.0
*Dependency of Space-Time Interest Point Detector*
- Download source: https://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.2/
- **Python=2.7** for compatibility with OpenCV 2.2.0
- Potentially needed OpenCV build ([CMake overview](https://internalpointers.com/post/modern-cmake-beginner-introduction)) adjustments in `CMakeLists.txt`:
    - Line 214: `set(OPENCV_BUILD_3RDPARTY_LIBS TRUE)`. Avoids backwards incompabtibility due to newer dependencies
    - Line 384: **+** `set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:[FFmpeg 0.5.2 install location]/lib/pkgconfig")`. Ensures desired FFmpeg installation will be used
    - Line 1275-6: `target_compile_options(opencv_core PRIVATE -Wno-narrowing)` &
    `target_compile_options(opencv_ts PRIVATE -Wno-narrowing)`. Prevents build failure due to the narrowing conversion compiler warning
- `$ cmake` with `-DWITH_EIGEN2=OFF`
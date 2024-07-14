import argparse
import cv2
import itertools
import joblib
from matplotlib import pyplot as plt
from mediapipe.python.solutions import face_mesh, drawing_utils, drawing_styles
from multiprocessing import Pool
import numpy as np
import os
from skimage.feature import hog
from sklearn.cluster import KMeans
from statistics import mean, mode
import subprocess as sp


def frame_level_features(vid_features, vid_labels, indexes):
    return np.concatenate([vid_features[ind] for ind in indexes], 0)

def frame_level_labels(vid_labels, vid_features, indexes):
    frame_labels = []
    for ind in indexes:
        frame_labels.extend([vid_labels[ind] for _ in range(vid_features[ind].shape[0])])
    return np.array(frame_labels)

def frame_to_vid_probs(class_probabilities, features, indexes):
    vid_class_probs = {}
    for classifier, probabilities in class_probabilities.items():
        vid_probs, prev_frames = [], 0
        for i, ind in enumerate(indexes):
            vid_probs.append(np.mean(probabilities[prev_frames: prev_frames + features[ind].shape[0]], 0))
            prev_frames += features[ind].shape[0]
        vid_class_probs[classifier] = np.stack(vid_probs, 0)
    return vid_class_probs

def frame_to_vid_preds(class_predictions, features, indexes):
    vid_class_preds = {}
    for classifier, predictions in class_predictions.items():
        vid_preds, prev_frames = np.empty(indexes.shape[0]), 0
        for i, ind in enumerate(indexes):
            vid_preds[i] = mode(list(predictions[prev_frames: prev_frames + features[ind].shape[0]]))
            prev_frames += features[ind].shape[0]
        vid_class_preds[classifier] = vid_preds
    return vid_class_preds


def get_arguments():
    parser = argparse.ArgumentParser()
    input_parser = parser.add_argument_group("Input Control", "Configure the data folders for characterization")
    input_parser.add_argument("-c", "--categories", nargs="+", required=True, help="Input videos' directories, each represent one category")
    input_parser.add_argument("-s", "--suffix-dir", type=str, required=True, help="The additional nested directory in categories to separate original videos")
    sparse_parser = parser.add_argument_group("Sparse HOG", "Adjust the settings for sparse HOG")
    sparse_parser.add_argument("-n", "--n_clusters", default=800, type=int, help="# of clusters in bag of features")
    dense_parser = parser.add_argument_group("Dense HOG", "Enable and select the mode of dense HOG")
    dense_parser.add_argument("-d", "--dense", action="store_true", help="Whether to use dense feature extraction instead of space-time interest points")
    dense_parser.add_argument("-f", "--frame-level", action="store_true", help="Whether to record dense features per frame instead of averaging across frames."
                        " Only has effect given --dense")
    sparse_parser.add_argument("-g", "--graph-bof", action="store_true", help="Whether to visualize 1 random BoF histogram from every category."
                        " Only has effect if --dense is not set")
    sparse_parser.add_argument("-l", "--stip-linker-path", default="~/.local/lib", type=str, help="The value for LD_LIBRARY_PATH when calling the stipdet program")
    return parser.parse_args()


def extract_face_video(video_name, video_dir, landmark_dir, save_dir, visualize=False):
    video = cv2.VideoCapture(os.path.join(video_dir, video_name))
    save_name = os.path.splitext(video_name)[0] + "_face.avi"
    save_path = os.path.join(save_dir, save_name)
    if os.path.exists(save_path):
        print(f"{video_name} skipped because already processed (stored at {save_path})")
        return
    framerate = video.get(cv2.CAP_PROP_FPS)

    bound_box_sizes = ([], [])
    cropped_frames, lmks_per_frame = [], []

    with face_mesh.FaceMesh(refine_landmarks=True) as mesh_extractor:
        frame_count = -1
        while video.isOpened():
            read_success, frame = video.read()
            frame_count += 1
            if not read_success: break
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mesh_extractor.process(frame)

            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                assert len(results.multi_face_landmarks) == 1, f"{len(results.multi_face_landmarks)} faces detected (instead of 1)."
                for face_landmarks in results.multi_face_landmarks:
                    if visualize:
                        frame_draw = frame.copy()
                        drawing_utils.draw_landmarks(
                            image=frame_draw,
                            landmark_list=face_landmarks,
                            connections=face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=drawing_styles
                            .get_default_face_mesh_tesselation_style())
                        drawing_utils.draw_landmarks(
                            image=frame_draw,
                            landmark_list=face_landmarks,
                            connections=face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=drawing_styles
                            .get_default_face_mesh_contours_style())
                        drawing_utils.draw_landmarks(
                            image=frame_draw,
                            landmark_list=face_landmarks,
                            connections=face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=drawing_styles
                            .get_default_face_mesh_iris_connections_style())
                        cv2.imshow(save_name, frame_draw)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            visualize = False
                            cv2.destroyWindow(save_name)
                    x_coords = [face_landmarks.landmark[ind].x for ind in range(len(face_landmarks.landmark))]
                    y_coords = [face_landmarks.landmark[ind].y for ind in range(len(face_landmarks.landmark))]
                    x_min, x_max, y_min, y_max = min(x_coords), max(x_coords), min(y_coords), max(y_coords)
                    top_bound, bot_bound = int(y_min * frame.shape[0]), int(y_max * frame.shape[0])
                    left_bound, right_bound = int(x_min * frame.shape[1]), int(x_max * frame.shape[1])
                    cropped_frames.append(frame[top_bound: bot_bound, left_bound: right_bound])
                    bound_box_sizes[0].append(right_bound - left_bound)
                    bound_box_sizes[1].append(bot_bound - top_bound)
                    landmark_coords = set()
                    for landmark_coord in face_landmarks.landmark:
                        coord = ((landmark_coord.x - x_min) / (x_max - x_min), (landmark_coord.y - y_min) / (y_max - y_min))
                        landmark_coords.add(coord)
                    lmks_per_frame.append(landmark_coords)

    video.release()
    print(f"{video_name}: Lost {frame_count + 1 - len(cropped_frames)} frame(s). # frames original: {frame_count + 1}; # frames cropped: {len(cropped_frames)}")

    joblib.dump(lmks_per_frame, os.path.join(landmark_dir, os.path.splitext(video_name)[0] + "_landmarks.gz"))
    avg_width, avg_height = int(mean(bound_box_sizes[0])), int(mean(bound_box_sizes[1]))
    for ind in range(len(cropped_frames)):
        cropped_frames[ind] = cv2.resize(cropped_frames[ind], (avg_width, avg_height))
    cropped_video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), min(framerate, 30), (avg_width, avg_height))
    for frame in cropped_frames:
        cropped_video.write(frame)
    cropped_video.release()


def resize_vid(vid_path, landmark_path, avg_width, avg_height):
    video_orig = cv2.VideoCapture(vid_path)
    framerate, width, height = video_orig.get(cv2.CAP_PROP_FPS), video_orig.get(cv2.CAP_PROP_FRAME_WIDTH), video_orig.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if width != avg_width or height != avg_height:
        resized_frames = []
        while video_orig.isOpened():
            read_success, frame = video_orig.read()
            if not read_success: break
            resized_frames.append(cv2.resize(frame, (avg_width, avg_height)))
        video_orig.release()
        norm_video = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"mp4v"), min(framerate, 30), (avg_width, avg_height))
        for resized_frame in resized_frames:
            norm_video.write(resized_frame)
        norm_video.release()
    else:
        print(f"Skiping resizing {os.path.basename(vid_path)} (already width: {width}; height: {height})")
        video_orig.release()
    lmks_per_frame = joblib.load(landmark_path)
    for ind in range(len(lmks_per_frame)):
        coords_remove, coords_add = set(), set()
        for coord in lmks_per_frame[ind]:
            assert type(coord[0]) == type(coord[1]), f"{landmark_path} at frame #{ind} has coordinate with mistmatched types ({coord[0]}: {type(coord[0])}, {coord[1]}: {type(coord[1])})"
            if isinstance(coord[0], float):
                assert 0 <= coord[0] <= 1 and 0 <= coord[0] <= 1, f"{landmark_path} at frame #{ind} has coordinate with invalid relative positions ({coord[0]}, {coord[1]})"
                coords_remove.add(coord)
                coords_add.add((int(coord[0] * avg_width), int(coord[1] * avg_height)))
        lmks_per_frame[ind].difference_update(coords_remove)
        lmks_per_frame[ind].update(coords_add)
    joblib.dump(lmks_per_frame, landmark_path)


def normalize_vid_size(video_files, landmark_files):
    avg_calced = False
    if os.path.exists("mean_resolutions.txt"):
        with open("mean_resolutions.txt", "r") as res_info_file:
            avg_width, avg_height = res_info_file.read().split()
        avg_width, avg_height = int(avg_width), int(avg_height)
        avg_calced = True
    resolutions, resize_args = ([], []), []
    for vid_path, landmark_path in zip(video_files, landmark_files):
        video = cv2.VideoCapture(vid_path)
        width, height = video.get(cv2.CAP_PROP_FRAME_WIDTH), video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if avg_calced:
            resize_args.append((vid_path, landmark_path, avg_width, avg_height))
        else:
            resolutions[0].append(width)
            resolutions[1].append(height)
            resize_args.append([vid_path, landmark_path])
        video.release()
    if not avg_calced:
        avg_width, avg_height = int(mean(resolutions[0])), int(mean(resolutions[1]))
        with open("mean_resolutions.txt", "w") as res_info_file:
            res_info_file.write(f"{avg_width} {avg_height}")
        for ind in range(len(resize_args)):
            resize_args[ind] = (*resize_args[ind], avg_width, avg_height)
    with Pool() as pool:
        pool.starmap(resize_vid, resize_args)
    os.remove("mean_resolutions.txt")


def video_level_hog(video_files, frame_level=False):
    video_hogs = []
    for vid_path in video_files:
        video = cv2.VideoCapture(vid_path)
        hog_per_frame = []
        while video.isOpened():
            read_success, frame = video.read()
            if not read_success: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hog_descriptor = hog(frame, pixels_per_cell=(32, 32), cells_per_block=(2, 2))
            hog_per_frame.append(hog_descriptor[np.newaxis, ...])
        video.release()
        hog_per_frame = np.concatenate(hog_per_frame, 0)
        if frame_level:
            video_hogs.append(hog_per_frame)
        else:
            hog_mean, hog_stdev = np.mean(hog_per_frame, 0), np.std(hog_per_frame, 0)
            video_hogs.append(np.concatenate((hog_mean, hog_stdev))[np.newaxis, ...])
    return video_hogs if frame_level else np.concatenate(video_hogs, 0)


def calc_stip_features(stip_linker_path, path_to_videos, path_to_output_file, filename, ext=None, start_frame=None, end_frame=None,
                       overwrite=False, visualize=False):
    """
    https://github.com/TheAntimist/action-detection
    Calculates the STIP features for a specified set of input files and stores them in the output location.
    :param path_to_videos: Directory containing the training dataset to be computed
    :param filename: Input filename of video
    :param path_to_output_file: Output file location containing the STIP features.
    :param ext: Extension of video, if None is provided, then "avi" is assumed
    :param start_frame: (Optional) Start frame of the video
    :param end_frame: (Optional) End frame of the video
    :return:
    """

    if not overwrite and os.path.isfile(path_to_output_file):
        # Skip if file already present
        print(f"Ignoring file {filename}, because overwrite is disabled")
        return

    if start_frame and end_frame:
        video_list_file = filename + ".temp." + start_frame + "-" + end_frame + ".txt"
    else:
        video_list_file = filename + ".temp.txt"

    with open(os.path.join(path_to_videos, video_list_file), "w") as video_list:
        if start_frame is not None and end_frame is not None:
            video_list.write(filename + " " + start_frame + " " + end_frame + "\n")
        else:
            video_list.write(filename + "\n")

    ext_str = " -ext " + ext if ext else ""

    args = f"stipdet -i {os.path.join(path_to_videos, video_list_file)} {ext_str} -vpath {path_to_videos if path_to_videos[-1] == '/' else path_to_videos + '/'} " + \
        f"-o {path_to_output_file} -dscr hog -vis {'yes' if visualize else 'no'} -stdout no"

    process = ["/bin/bash", "-c", args]

    with sp.Popen(process, env=dict(os.environ, LD_LIBRARY_PATH=stip_linker_path), stdout=sp.DEVNULL) as p:
        try:
            print(f"Running Stipdet on {filename}{(' with frames ' + start_frame + '-' + end_frame) if start_frame else ''}")
            retcode = p.wait()
            if retcode:
                cmd = "stipdet"
                raise sp.CalledProcessError(retcode, cmd)
        except:
            p.kill()
            p.wait()
        finally:
            os.remove(os.path.join(path_to_videos, video_list_file))


def filter_stips_with_landmarks(stip_path, landmark_path):
    def check_stip_valid(stip_info, all_landmarks):
        stip_coord, pt_param = stip_info[4:7].astype(int), stip_info[7:9]
        space_range = int(9 * (pt_param[0] ** 0.5))
        time_range = int(4 * (pt_param[1] ** 0.5))
        stip_region = set(itertools.product(range(stip_coord[2] - time_range, stip_coord[2] + time_range), range(stip_coord[1] - space_range, stip_coord[1] + space_range),
                                            range(stip_coord[0] - space_range, stip_coord[0] + space_range)))
        return True if len(all_landmarks.intersection(stip_region)) != 0 else False
    file_stips = np.genfromtxt(stip_path, comments="#")
    lmks_per_frame = joblib.load(landmark_path)
    all_landmarks = set((ind, coord[0], coord[1]) for ind in range(len(lmks_per_frame)) for coord in lmks_per_frame[ind])
    valid_stips = np.apply_along_axis(check_stip_valid, 1, file_stips, all_landmarks)
    return file_stips[valid_stips, 9:]


def cluster_kmeans(n_clusters, stip_files, landmark_files, labels, categories, graph_bof):
    feats_per_vid = []
    if os.path.exists("collected_stip_features.gz"):
        stip_feats = joblib.load("collected_stip_features.gz")
        if stip_feats["labels"] == labels:
            feats_per_vid = stip_feats["feats_per_vid"]
            print(f"Using existing collected_stip_features.gz")
    if len(feats_per_vid) == 0:
        print(f"Generating collected_stip_features.gz")
        filter_args = [(stip_path, landmark_path) for stip_path, landmark_path in zip(stip_files, landmark_files)]
        with Pool() as pool:
            for filtered_stips in pool.starmap(filter_stips_with_landmarks, filter_args):
                feats_per_vid.append(filtered_stips)
        joblib.dump({"feats_per_vid": feats_per_vid, "labels": labels}, "collected_stip_features.gz")

    cluster_feats = np.concatenate(feats_per_vid, 0)
    bof = KMeans(n_clusters).fit(cluster_feats)
    video_descriptors = np.zeros((len(feats_per_vid), n_clusters))
    for vid_ind in range(len(feats_per_vid)):
        feat_clusters = bof.predict(feats_per_vid[vid_ind])
        for cluster_ind in np.unique(feat_clusters):
            video_descriptors[vid_ind][cluster_ind] += np.sum(feat_clusters == cluster_ind)
    df = np.sum(video_descriptors > 0, axis=0)
    idf = np.log(len(feats_per_vid) / df)
    video_descriptors *= idf
    if graph_bof:
        unique_labels, inds_to_graph = set(), []
        for ind, label in enumerate(labels):
            if label not in unique_labels:
                unique_labels.add(label)
                inds_to_graph.append(ind)
        fig, axes = plt.subplots(1, len(unique_labels), figsize=(len(unique_labels) * 10, 10))
        for ax_ind, (data_label, data_ind) in enumerate(zip(unique_labels, inds_to_graph)):
            axes[ax_ind].bar(list(range(n_clusters)), video_descriptors[data_ind])
            axes[ax_ind].set_title(categories[data_label], fontsize=20)
            axes[ax_ind].tick_params(which="both", labelsize=17.5)
        fig.savefig("BoF_sample_histograms.png", transparent=True)
    return video_descriptors


if __name__ == "__main__":
    args = get_arguments()
    video_dirs = [os.path.join(category, args.suffix_dir) for category in args.categories]
    landmark_dirs = [video_dir + "_landmarks" for video_dir in video_dirs]
    face_dirs = [video_dir + "_faces" for video_dir in video_dirs]
    for face_dir in face_dirs: os.makedirs(face_dir, exist_ok=True)
    for landmark_dir in landmark_dirs: os.makedirs(landmark_dir, exist_ok=True)

    mesh_args, labels = [], []
    for ind in range(len(args.categories)):
        for video_name in os.listdir(video_dirs[ind]):
            mesh_args.append((video_name, video_dirs[ind], landmark_dirs[ind], face_dirs[ind]))
            labels.append(ind)
    print(f"Processing {len(mesh_args)} videos. Output: {face_dirs}; original: {video_dirs}")
    with Pool() as pool:
        pool.starmap(extract_face_video, mesh_args)
    
    video_files, landmark_files = [], []
    for arg_set, label in zip(mesh_args, labels):
        vid_name, _ = os.path.splitext(arg_set[0])
        video_files.append(os.path.join(face_dirs[label], vid_name + "_face.avi"))
        landmark_files.append(os.path.join(landmark_dirs[label], vid_name + "_landmarks.gz"))
    normalize_vid_size(video_files, landmark_files)

    if args.dense:
        video_descriptors = video_level_hog(video_files, args.frame_level)
        joblib.dump({"dense_hog": video_descriptors, "labels": np.array(labels)}, "combined_characteristics.gz")
        exit()
    stip_args = []
    feats_dirs = [video_dir + "_feats" for video_dir in video_dirs]
    for feats_dir in feats_dirs: os.makedirs(feats_dir, exist_ok=True)
    for arg_set, label in zip(mesh_args, labels):
        filename, _ = os.path.splitext(arg_set[0])
        stip_args.append((args.stip_linker_path, face_dirs[label], os.path.join(feats_dirs[label], filename + "_feats.txt"), filename + "_face"))
    with Pool() as pool:
        pool.starmap(calc_stip_features, stip_args)
    cv2.destroyAllWindows()

    stip_files = [arg_set[2] for arg_set in stip_args]
    video_descriptors = cluster_kmeans(args.n_clusters, stip_files, landmark_files, labels, args.categories, args.graph_bof)
    print(f"Zeros in video_descriptors: {np.sum(video_descriptors == 0)} ({100 * np.sum(video_descriptors == 0) / video_descriptors.size:.2f}% of {video_descriptors.size})")
    joblib.dump({"bof": video_descriptors, "labels": labels}, "combined_characteristics.gz")

import argparse
import subprocess
from collections import Counter
from pathlib import Path

import artistic as art
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from joblib import load
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, Dense, Flatten,
                                     Permute, Reshape)
from tensorflow.keras.models import Model, Sequential
from tqdm import tqdm


def CAE(input_shape=(256, 384, 1), filters=[16, 32, 64, 128, 40], strides=[4, 4, 2, 2], size_filters=[16, 8, 3, 3]):
    model = Sequential()

    model.add(Conv2D(filters[0], size_filters[0], strides=strides[0], padding='same', activation='relu', name='conv1', input_shape=input_shape))
    model.add(Conv2D(filters[1], size_filters[1], strides=strides[1], padding='same', activation='relu', name='conv2'))
    model.add(Conv2D(filters[2], size_filters[2], strides=strides[2], padding='same', activation='relu', name='conv3'))
    model.add(Conv2D(filters[3], size_filters[3], strides=strides[3], padding='same', activation='relu', name='conv4'))

    model.add(Flatten())
    model.add(Dense(units=filters[4], name='embedding'))
    reduction = np.prod(strides)
    model.add(Dense(units=filters[3]*int(input_shape[0]/reduction)*int(input_shape[1]/reduction), activation='relu'))
    model.add(Reshape((filters[3],int(input_shape[0]/reduction), int(input_shape[1]/reduction))))
    model.add(Permute((2,3,1)))
    model.add(Conv2DTranspose(filters[2], size_filters[3], strides=strides[3], padding='same', activation='relu', name='deconv4'))
    model.add(Conv2DTranspose(filters[1], size_filters[2], strides=strides[2], padding='same', activation='relu', name='deconv3'))
    model.add(Conv2DTranspose(filters[0], size_filters[1], strides=strides[1], padding='same', activation='relu', name='deconv2'))
    model.add(Conv2DTranspose(input_shape[-1], size_filters[0], strides=strides[0], padding='same', name='deconv1'))

    return model

@tf.function
def CAE_predictions(samples, model):
    print('^' * 80)
    print('Keras compute...\n')
    predictions = model(samples)
    print('\nEnd of Keras compute.')
    print('^' * 80)
    return predictions


def process_rr_npy(scratch_path, nb_batch_render, source, step, mode='decode'):
    print('\nProcessing rr_npy...')
    if mode not in {'latent', 'decode'}:
        raise ValueError(f"<mode> should be either 'latent' or 'decode'. You provided {mode}.")
    BATCH_SIZE_AE = 4096
    model = CAE()
    model.load_weights('/home/users/u101957/DE371_StyleGAN/gan_2_ae/Best_autoencodeur_weights.h5')
    print(model.summary())  # This shows the architecture of the model  
    dummy_input = tf.random.normal((1, 256, 384, 1))  # Replace input_dim with the actual input dimension
    model(dummy_input)  
    for layer in model.layers:
        print('LAYERS',layer.name,model.inputs)
    if mode == 'latent':
        model = Model(inputs=model.inputs, outputs=model.get_layer(name='embedding').output)
    for batch in tqdm(range(nb_batch_render), desc='Processing batches', total=nb_batch_render, leave=False, unit='batch', colour='green'):
        predictions = []
        rr_npy = np.load(scratch_path / Path(f"{f'step_{step}_' if source == 'GAN' else ''}batch_{batch + 1}.npy"))
        rr_npy = np.pad(rr_npy, ((0, 0), (0, 0), (0, 384 - 256)), mode='constant')
        
        rr_npy_mask = (rr_npy < 0.5)
        rr_npy[rr_npy_mask] = 0

        for sub_batch in range(0, 4096, BATCH_SIZE_AE):
            predictions.append(CAE_predictions(rr_npy[sub_batch:sub_batch + BATCH_SIZE_AE], model))
        if mode == 'decode':
            can = art.CanvasHolder('SE_GAN_extend', 384, 256, Path(__file__).parent.absolute())
            for idx, (source, prediction) in enumerate(zip(rr_npy, predictions)):
                can.plot_data_normal(source, scratch_path, Path(f'{idx}_source.jpg'))
                can.plot_data_normal(prediction[:, :, 0], scratch_path, Path(f'{idx}_prediction.jpg'))
        np.save(scratch_path / f'prediction_batch_{batch + 1}.npy', predictions)
    print('\nProcessed rr_npy.\n')


def load_clustering_model(model_path='/home/users/u101957/DE371_StyleGAN/gan_2_ae/KM.joblib'):
    """Load the clustering model."""
    return load(model_path)

def labels_Y(zone):
    """Get class labels for the given zone."""
    if zone == 'SE':
        return ['Rien signif', 'Espagne', 'Espagne_Roussillon', 'Pyrénées', 'PO_Aude_Hérault-S',
                'Centre Médit', 'Cevennes_Gard_Hérault-N', 'Var_PACA Ouest_Drôme-S',
                'Alpes-Mar_Golfe-G', 'Corse O', 'Corse S', 'Corse E', 'Massif-C Sud',
                'Massif-C Centre', 'Jura_Alpes_Drôme-N', 'Italie']

def new_assign_position(zone, cluster_num):
    """Assign a new position based on the zone and cluster number."""
    if zone == 'SE':
        new_num = [1., 3., 16., 6., 16., 11., 4., 2., 9., 13., 16., 12., 14., 16., 5., 8., 7., 12., 10., 15.]
        return new_num[cluster_num]

def process_batch(predictions, rr_maps, rr_map_moy, rr_map_squares):
    """Process a batch of predictions and update mean and square maps."""
    new_assign_vect = np.vectorize(new_assign_position)
    latest_classed = new_assign_vect('SE', predictions)

    for idx, classe_idx in enumerate(latest_classed):
        rr_map_moy[int(classe_idx) - 1] += rr_maps[idx]
        rr_map_squares[int(classe_idx) - 1] += rr_maps[idx] ** 2

    return rr_map_moy, rr_map_squares, latest_classed

def save_and_draw_min_max_samples(min_samples, min_distance_samples, max_samples, max_distance_samples, Y_labels, scratch_path):
    """Save best and worst samples for each cluster"""
    print('Saving best and worse samples...')
    can = art.CanvasHolder('SE_GAN_extend', 384, 256, Path(__file__).parent.absolute())
    save_path = scratch_path / 'picture' / 'best_and_worst'
    save_path.mkdir(parents=True, exist_ok=True)
    min_samples = np.pad(min_samples, ((0, 0), (0, 0), (0, 384 - 256)), mode='constant')
    max_samples = np.pad(max_samples, ((0, 0), (0, 0), (0, 384 - 256)), mode='constant')
    for idx, ((rr_map_best, best_distance), (rr_map_worst, worst_distance)) in enumerate(zip(zip(min_samples, min_distance_samples), zip(max_samples, max_distance_samples))):
        print(f"Drawing and saving for centroid {idx + 1}, {Y_labels[int(new_assign_position('SE', idx)) - 1]}...")
        can.plot_data_normal(rr_map_best, save_path, Path(f"{Y_labels[int(new_assign_position('SE', idx)) - 1]}_{idx}_best_{round(best_distance, 2)}.jpg"))
        can.plot_data_normal(rr_map_worst, save_path, Path(f"{Y_labels[int(new_assign_position('SE', idx)) - 1]}_{idx}_worst_{round(worst_distance, 2)}.jpg"))
        

def save_results(rr_map_moy, rr_map_squares, class_counter, Y_labels, scratch_path):
    """Save classification results."""
    print('Batches processed')
    for class_idx in range(len(rr_map_moy)):
        if class_counter[Y_labels[class_idx]] != 0:
            rr_map_moy[class_idx] /= class_counter[Y_labels[class_idx]]
            rr_map_squares[class_idx] = np.sqrt(class_counter[Y_labels[class_idx]] / (class_counter[Y_labels[class_idx]] - 1) * ((rr_map_squares[class_idx] / class_counter[Y_labels[class_idx]]) - rr_map_moy[class_idx] ** 2))
    print('Drawing...')
    can = art.CanvasHolder('SE_GAN_extend', 384, 256, Path(__file__).parent.absolute(), bound_num=2)
    rr_map_moy = np.pad(rr_map_moy, ((0, 0), (0, 0), (0, 384 - 256)), mode='constant')
    rr_map_squares = np.pad(rr_map_squares, ((0, 0), (0, 0), (0, 384 - 256)), mode='constant')

    save_path = scratch_path / 'picture'
    save_path.mkdir(parents=True, exist_ok=True)

    for idx, (rr_moy, rr_squares) in enumerate(zip(rr_map_moy, rr_map_squares)):
        can.plot_data_normal(rr_moy, save_path, Path(f'{Y_labels[idx]}_moy.jpg'))
        can.plot_data_normal(rr_squares, save_path, Path(f'{Y_labels[idx]}_std.jpg'))

    save_path = save_path / 'npys'
    save_path.mkdir(exist_ok=True)
    np.save(save_path / 'moy.npy', rr_map_moy)
    np.save(save_path / 'std.npy', rr_map_squares)

    print('\nClassification done.\n')

def classify(scratch_path, step, shape, nb_batch, source):
    """Classify and visualize weather data."""
    print('\nClassifying...\n')

    clustering_model = load_clustering_model()
    
    zone = 'SE'
    Y_labels = labels_Y(zone)
    rr_map_moy = np.zeros((len(Y_labels), shape[0], shape[1]))
    rr_map_squares = np.zeros((len(Y_labels), shape[0], shape[1]))
    classes = []
    pred_class_record = []

    min_samples = np.empty((len(clustering_model.cluster_centers_), shape[0], shape[1]))
    max_samples = np.empty((len(clustering_model.cluster_centers_), shape[0], shape[1]))
    min_distance_samples = np.full((len(clustering_model.cluster_centers_,)), np.inf)
    max_distance_samples = np.zeros((len(clustering_model.cluster_centers_)))

    for batch in tqdm(range(nb_batch), desc='Processing batches', total=nb_batch, leave=False, unit='batch', colour='green'):
        predictions = np.concatenate(np.load(scratch_path / f'prediction_batch_{batch + 1}.npy'))
        rr_maps = np.load(scratch_path / Path(f"{f'step_{step}_' if source == 'GAN' else ''}batch_{batch + 1}.npy"))

        predictions_classed = clustering_model.predict(predictions.astype(float))
        pred_class_record.append(predictions_classed)
        rr_map_moy, rr_map_squares, latest_classed = process_batch(predictions_classed, rr_maps, rr_map_moy, rr_map_squares)
        classes.append(latest_classed)
        
        distances = clustering_model.transform(predictions.astype(float))
        
        min_indices = np.argmin(distances, axis=0)
        min_distance = distances[min_indices, range(len(min_indices))]
        min_mask = min_distance < min_distance_samples
        min_samples[min_mask] = rr_maps[min_indices[min_mask]]
        min_distance_samples[min_mask] = min_distance[min_mask]

        for centroid in range(len(clustering_model.cluster_centers_)):
            cluster_mask = (predictions_classed == centroid)
            try:
                max_distance_within_cluster = np.max(distances[cluster_mask, centroid])
                if max_distance_within_cluster > max_distance_samples[centroid]:
                    max_sample_index_within_cluster = np.argmax(distances[cluster_mask, centroid])
                    max_samples[centroid] = rr_maps[max_sample_index_within_cluster]
                    max_distance_samples[centroid] = max_distance_within_cluster
            except ValueError as err:
                # print(f"Centroid {Y_labels[int(new_assign_position('SE', centroid)) - 1]} (centroid {centroid}) has no sample. Skipping.")
                pass


    pred_class_record = np.concatenate(pred_class_record)
    for centroid in range(len(clustering_model.cluster_centers_)):
        if centroid not in pred_class_record:
            min_samples[centroid] = np.full_like(min_samples[centroid], 1000)
            max_samples[centroid] = np.full_like(max_samples[centroid], 1000)

    save_and_draw_min_max_samples(min_samples, min_distance_samples, max_samples, max_distance_samples, Y_labels, scratch_path)
    classes = np.concatenate(classes)
    classes = [Y_labels[int(index) - 1] for index in classes]
    class_counter = Counter(classes)
    save_results(rr_map_moy, rr_map_squares, class_counter, Y_labels, scratch_path)

    return classes, class_counter

def name_space_to_slurm_arg(config):
    args = [f'--{key}={value}' for key, value in vars(config).items() if key != 'skip_pre_processing']
    return "|".join(args)

def copy_distant_files(folder_source, scratch_root_path):
    # scp_command = f'scp -r belenos:/scratch/work/gandonb/{folder_source} {scratch_root_path}/'
    scp_command = f'scp -r belenos:/scratch/work/gandonb/{folder_source} {scratch_root_path}/'

    try:
        print('Copying distant files...')
        subprocess.run(scp_command, shell=True, check=True)
        print('Copying distant files done.')
    except subprocess.CalledProcessError as err:
        print(f'Error while executing scp command: {err}')

def main(args):
    scratch_root_path = Path('/project/home/p200177/DE_371/datasets/dataset_Meteo_France_rr_u_v_t2m/')#data/IS_rr_debug_1_1.0_0_0_0_0_0_256_large_lt')
    folder_source = f'samples_AROME_for_AE_None' if args.source == 'AROME' else f'samples_detransformed_for_AE_{args.name}'
    
    scratch_root_path.mkdir(parents=True, exist_ok=True)
    
    if not args.skip_pre_processing:
        result = subprocess.run(['bash', 'launch_pre_proc_for_AE.sh', name_space_to_slurm_arg(args)], text=True)
        copy_distant_files(folder_source, scratch_root_path)

    scratch_root_path /= folder_source

    process_rr_npy(scratch_root_path, args.nb_batch_render, args.source, args.step, 'latent')
    classes, class_counter = classify(scratch_root_path, args.step, (256, 256), args.nb_batch_render, args.source)

    with open(scratch_root_path / 'log.txt', 'w') as logfile:
        keys = sorted(class_counter.keys())
        for key in keys:
            logfile.write(f'{key}: {class_counter[key]}\n')

    print(f'End of <{__file__}> execution.')

if __name__ == '__main__':
    print('°' * 80)
    get_hostname = "hostname"
    output = subprocess.run(get_hostname.split(), capture_output=True, text=True)
    print(f"<{__file__}> running on {output.stdout.strip()}")

    parser = argparse.ArgumentParser()

    main_features_args = parser.add_argument_group('Main features')
    main_features_args.add_argument('set_num', type=int, default = 0, help='Set number, 0 for AROME')
    main_features_args.add_argument('step', type=int, default = 0,help='Step to throw through pipeline, 0 for AROME')
    main_features_args.add_argument('name', type=int, default='scenarios',help='Suffix added to the folders')

    generation_characteristics_args = parser.add_argument_group('Generation characteristics for one step')
    generation_characteristics_args.add_argument('-n', '--nb_fake_samples', type=int, default=131072, help='Number of sample generated')
    generation_characteristics_args.add_argument('--nb_batch', type=int, default=1024, help='Number of batch')
    generation_characteristics_args.add_argument('--source', choices=['AROME', 'GAN'], default='AROME', type=str, help='Select the source of the samples')
    generation_characteristics_args.add_argument('--nb_batch_render', type=int, default=1, help='Number of render batch of nb_fake_samples files')

    specifics_for_main_predict_args = parser.add_argument_group('Arguments specific to this file')
    specifics_for_main_predict_args.add_argument('-s', '--skip_pre_processing', action='store_true', help='If used, skip the preprocessing part')

    args = parser.parse_args()
    main(args)
    print('°' * 80)

    
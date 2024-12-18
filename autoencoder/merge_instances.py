from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from math import sqrt
from pathlib import Path
import artistic as art
import numpy as np

SCRATCH_ROOT_PATH = Path('')
NB_EXP = 10

def get_labels_Y(zone):
    if zone == 'SE':
        return ['Rien signif', 'Espagne', 'Espagne_Roussillon', 'Pyrénées', 'PO_Aude_Hérault-S', 'Centre Médit', 'Cevennes_Gard_Hérault-N', 'Var_PACA Ouest_Drôme-S', 'Alpes-Mar_Golfe-G', 'Corse O', 'Corse S', 'Corse E', 'Massif-C Sud', 'Massif-C Centre', 'Jura_Alpes_Drôme-N', 'Italie']
    return []

def load_data(exp):
    folder_counter = SCRATCH_ROOT_PATH / f'samples_detransformed_for_AE_{exp + 1}'
    folder_npy = folder_counter / 'picture/npys'
    moy = np.load(folder_npy / 'moy.npy')
    std = np.load(folder_npy / 'std.npy')

    with open(folder_counter / 'log.txt', 'r') as logfile:
        file_content = logfile.read()

    counter_exp = Counter()
    for line in file_content.split('\n'):
        key, sep, value = line.partition(': ')
        if sep and value.isdigit():
            counter_exp[key] = int(value)

    return moy, std, counter_exp

def process_data(exp):
    moy, std, counter_exp = load_data(exp)
    return moy, std, counter_exp

def main():
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_data, range(NB_EXP)))

    rr_map_moy, rr_map_std, counter = zip(*results)
    counter_glob = sum(counter, Counter())
    Y_labels = get_labels_Y('SE')

    moy_glob = np.mean(rr_map_moy, axis=0)
    std_glob = np.sqrt(np.mean(np.array(rr_map_std) ** 2, axis=0))

    print('Drawing...')
    can = art.CanvasHolder('SE_GAN_extend', 384, 256, Path(__file__).parent.absolute(), bound_num=2)
    save_path = SCRATCH_ROOT_PATH / 'picture'
    save_path.mkdir(parents=True, exist_ok=True)

    for idx, (rr_moy, rr_squares) in enumerate(zip(moy_glob, std_glob)):
        can.plot_data_normal(rr_moy, save_path, Path(f'{Y_labels[idx]}_moy_n_samples_{counter_glob[Y_labels[idx]]}.jpg'))
        can.plot_data_normal(rr_squares, save_path, Path(f'{Y_labels[idx]}_std_n_samples_{counter_glob[Y_labels[idx]]}.jpg'))

if __name__ == "__main__":
    main()

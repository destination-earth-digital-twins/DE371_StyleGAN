from encoders.utils import common
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def log_images_diff(config, x, y_hat, iter=None):

        sample_data = []

        if isinstance(y_hat, dict):
            output = [
                [common.numpyfy(y_hat[0][iter_idx][0])]
                for iter_idx in range(len(y_hat[0]))
            ]
        else:
            output = [common.numpyfy(y_hat[0])]
            
        cur_sample_data = {
            'input': common.numpyfy(x[0]),
            'output': output,
        }

        sample_data.append(cur_sample_data)

        fig = common.vis_samples_diff(sample_data, config.n_vars, single=True if iter is None else False)
        if iter is not None :
            figname = config.output_dir + f"{config.date_index}_{config.lt_index}_{iter}_diff.png"
        else :
            figname = config.output_dir + f"{config.date_index}_{config.lt_index}_diff.png"
        fig.savefig(figname)
        plt.close(fig)


def log_images(config, x, y_hat):

        sample_data = []

        if isinstance(y_hat, dict):
            output = [
                [common.numpyfy(y_hat[0][iter_idx][0])]
                for iter_idx in range(len(y_hat[0]))
            ]
        else:
            output = [common.numpyfy(y_hat[0])]
            
        cur_sample_data = {
            'input': common.numpyfy(x[0]),
            'output': output,
        }

        sample_data.append(cur_sample_data)

        fig = common.vis_samples(sample_data, config.n_vars)
        figname = config.output_dir + f"{config.date_index}_{config.lt_index}_all_iter.png"
        fig.savefig(figname)
        plt.close(fig)
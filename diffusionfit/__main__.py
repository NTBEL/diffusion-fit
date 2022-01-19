"""Main run module for diffusion fitting.
"""
import sys
import glob
import os.path
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(context='paper', style='ticks', palette='colorblind',
        rc={'figure.dpi':100, 'savefig.dpi':300, 'figure.figsize': [4, 3.5]})
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterator, **kwargs):
        return iterator

from .diffusionfit import GaussianFit


parser = argparse.ArgumentParser()
parser.add_argument('timestep', metavar='timestep', type=float,
                    help='Set the time interval between frames in seconds.')
parser.add_argument('pixel_size', metavar='pixel_size', type=float,
                    help='Set the pixel size in microns.')
parser.add_argument('stim_frame', metavar='stim_frame', type=int,
                    default=0,
                    help='Set the frame where the stimulation takes place.')
parser.add_argument('d_stim', metavar='d_stim', type=float, default=0.,
                    help='Set the diameter of the stimulation zone in microns.')
parser.add_argument('-peak-to-tail', nargs='?', metavar='peak_to_tail', type=int, default=3,
                    help='Set the peak/tail threshold during step 1 fitting for terminating the fitting analysis.')
parser.add_argument('-center', nargs='?', metavar='center', type=str, default='image',
                    help='Set how the center of the diffusion cloud is determined. Options are: image - (default), use the center pixel location of the image. intensity - centroid of intensity after stimulation. y,x - a specific pixel location.')
parser.add_argument('--ignore-threshold', dest='apply_threshold',
                    action='store_false',
                    help='Ignore the thresholding for termination after step 1 fitting.')
parser.set_defaults(apply_threshold=True)
parser.add_argument('-end-frame', nargs='?', metavar='end_frame', type=int,
                    default=None, const=None,
                    help='Specify the maximum frame to include in the analysis. Should larger than stim_frame.')
parser.add_argument('--write-tif', dest='write_tif', action='store_true',
                    help='Write out an ImageJ compatible tiff image file of the step 1 fits.')
parser.set_defaults(write_tif=False)
parser.add_argument('--time-resolved', dest='time_resolved', action='store_true',
                    help='Compute estimate of time-resolved diffusion coefficient and output corresponding plot.')
parser.set_defaults(time_resolved=False)
parser.add_argument('--loss-rate', dest='loss_rate', action='store_true',
                    help='Compute estimate of the loss rate for the diffusing species.')
parser.set_defaults(loss_rate=False)
args = parser.parse_args()
# Get the current directory from which to read files.
current_path = os.path.abspath('./')
# get the location to dump the output
out_path = os.path.abspath('./diffusion_fitting')
if not os.path.isdir(out_path):
    os.mkdir(out_path)

files = glob.glob(current_path+'/*.tif')

Dstar_values = list()
# Clean up the center argument in case of non-default
center = ''.join(args.center.split())
# If a pixel positon is given convert to a list
if ',' in center:
    center_split = center.split(',')
    print(center_split)
    center = list([int(center_split[0]), int(center_split[1])])
end_frame = args.end_frame
if end_frame is not None:
    if end_frame < args.stim_frame:
        end_frame = None
for file in tqdm(files, desc='Samples: '):
    file_prefix = os.path.splitext(os.path.split(file)[1])[0]
    sample_name = os.path.splitext(os.path.split(os.path.basename(os.path.normpath(file)))[1])[0]
    gfit = GaussianFit(file, stimulation_frame=args.stim_frame,
                       timestep=args.timestep, pixel_width=args.pixel_size,
                       stimulation_radius=args.d_stim/2,
                       center=center)

    D = gfit.fit(end=end_frame, verbose=False,
                 apply_step1_threshold=args.apply_threshold,
                 step1_threshold=args.peak_to_tail)
    if np.isnan(D):
        D = None
        Dstar_values.append({'sample:':sample_name, 'D*(x10^-7 cm^2/s)':D})
        continue
    rmse_avg = gfit.step1_rmse.mean()
    rmse_std = gfit.step1_rmse.std()
    rsquared = gfit.step2_rsquared
    effective_time = gfit.effective_time
    if args.loss_rate:
        loss_rate = gfit.loss_rate
        Dstar_values.append({'sample:':sample_name, 'D*(x10^-7 cm^2/s)':D*1e7,
                             'RMSE-mean':rmse_avg, 'RMSE-std':rmse_std,
                             'R-squared':rsquared, 'EffectiveTime':effective_time,
                             'LossRate':loss_rate})
    else:
        Dstar_values.append({'sample:':sample_name, 'D*(x10^-7 cm^2/s)':D*1e7,
                             'RMSE-mean':rmse_avg, 'RMSE-std':rmse_std,
                             'R-squared':rsquared, 'EffectiveTime':effective_time})
    fn_step1 = file_prefix + "_step1.png"

    gfit.display_image_fits(saveas=os.path.join(out_path, fn_step1))
    #tp = [0., 6., 12., 18., 24., 30.]
    #gfit.display_image_fits_at_times(tp, saveas=os.path.join(out_path, fn_step1))
    plt.close()

    fn_step2 = file_prefix + "_step2.png"
    gfit.display_linear_fit(saveas=os.path.join(out_path, fn_step2))
    plt.close()

    if args.time_resolved:
        fn_step2_time_resolved = file_prefix + "_step2_TR-D.png"
        gfit.display_time_resolved_dc(saveas=os.path.join(out_path, fn_step2_time_resolved))
        plt.close()
    gfit.export_to_csv(os.path.join(out_path, file_prefix))
    if args.write_tif:
        tiff_name = file_prefix + "_step1_fits.tif"
        gfit.write_step1_fits_to_tiff(saveas=os.path.join(out_path, tiff_name))

    gfit.export_to_csv(os.path.join(out_path, file_prefix))
Dstar_values_df = pd.DataFrame(Dstar_values)
print(Dstar_values_df)

"""Main run module for diffusion fitting.
"""
import sys
import os.path
import argparse
import pandas as pd
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
parser.add_argument('stimulation', metavar='stim_frame', type=int,
                    default=0,
                    help='Set the frame where the stimulation takes place.')
parser.add_argument('d_stim', metavar='d_stim', type=float, default=0.,
                    help='Set the diameter of the stimulation zone in microns.')
parser.add_argument('signal-to-noise', metavar='ston', type=int, default=5,
                    help='Set the singal/noise threshold for terminating the fitting analysis.')
args = parser.parse_args()
# Get the current directory from which to read files.
current_path = os.abspath('./')
# get the location to dump the output
out_path = os.path.abspath('./diffusion_fitting')
if not os.isdir(out_path):
    os.mkdir(out_path)

files = glob.glob(current_path+'*.tif')
Dstar_values = list()
for file in tqdm(files, desc='Image files: '):
    file_prefix = os.path.splitext(os.path.split(file)[1])[0]
    gfit = GaussianFit(file, stimulation_frame=args.stim_frame,
                       timestep=args.timestep, pixel_width=args.pixel_size,
                       stimulation_radius=args.d_stim/2)

    D = gfit.fit(verbose=False, s_to_n=args.ston)
    if np.isnan(D):
        D = None
        Dstar_values.append({'sample:':file, 'D*(x10^-7 cm^2/s)':D})
        continue
    Dstar_values.append({'sample:':file, 'D*(x10^-7 cm^2/s)':D*1e7})
    fn_step1 = file_prefix + "_step1.png"
    gaussian.display_image_fits(saveas=os.path.join(outdir, fn_step1))
    plt.close()

    fn_step2 = file_prefix + "_step2.png"
    gaussian.display_linear_fit(saveas=os.path.join(outdir, fn_step2))
    plt.close()
Dstar_values_df = pd.DataFrame(Dstar_values)
print(Dstar_values_df)

"""Main run module for diffusion fitting.
"""
import sys
import os.path
import argparse
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

# get the location to dump the output
out_path = os.path.abspath('./diffusion_fitting')
if not os.isdir(out_path):
    os.mkdir(out_path)
# initialize the analyzer with the script
analyzer = BilayerAnalyzer(input_file=input_script)
# run the analyzer
analyzer.run_analysis()
    # dump the output
    print("dumping output to: " + out_path)
    analyzer.dump_data(path=out_path)

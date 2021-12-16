"""
"""

from abc import ABC, abstractmethod
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.stats import linregress
import skimage
from skimage import io as skio
import matplotlib.pyplot as plt
import seaborn as sns

class DiffusionFitBase(ABC):
    """Abstract base class for diffusion fitting."""

    def __init__(self, img_file, stimulation_frame=1, timestep=1,
                 pixel_width=1, stimulation_radius=0):
        self._img_file = img_file
        self.images = skio.imread(img_file)
        self.n_frames = len(self.images)
        # Assume input for indexing of frames starts at 1.
        self.stimulation_frame = stimulation_frame
        # Adjust the index to start at 0.
        self._idx_stimulation = stimulation_frame - 1
        # This is the frame index where the diffusion starts.
        self._idx_zero_time = stimulation_frame
        self.timestep = timestep
        self.pixel_width = pixel_width
        self._idx_img_center = (0.5 * np.array(self.images[0].shape)).astype(np.int)
        self.img_center = np.array(self._idx_img_center) * pixel_width - 0.5*pixel_width
        self.times = np.array(list(range(0, self.n_frames))) * timestep - (stimulation_frame) * timestep
        self.x_edges = np.linspace(0, pixel_width * self.images[0].shape[1],
                                   self.images[0].shape[1] + 1, endpoint=True)
        self.y_edges = np.linspace(0, pixel_width * self.images[0].shape[0],
                                   self.images[0].shape[0] + 1, endpoint=True)
        self.x_centers = self.x_edges[:-1] + 0.5 * (self.x_edges[1:] - self.x_edges[:-1])
        self.y_centers = self.y_edges[:-1] + 0.5 * (self.y_edges[1:] - self.y_edges[:-1])
        # Generate a meshgrid for the x and y positions of pixels
        yv, xv = np.meshgrid(self.y_centers, self.x_centers, indexing='ij')
        self.yv = yv
        self.xv = xv

        # Get the distance of each pixel from the image center
        self.r = np.sqrt((xv - self.img_center[1])**2 + (yv - self.img_center[0])**2)
        # Mask to include only points outside the stimulation zone.
        self.r_stim = stimulation_radius
        self.rmask_stim_out = self.r > self.r_stim
        self.r_max = np.max(self.r)
        if self._idx_stimulation > 2:
            self.background = self.images[:self._idx_stimulation-1].mean(axis=0)
            print("background ",self.background.shape)
        else:
            self.background = 0
            print("background ", self.background)
        #self.background = background

        self._idx_fitted_frames = None
        self._fitting_parameters = None
        self._linr_res = None
        self._Ds = None
        self._t0 = None
        self._n_params = None

        return

    def fit(self, start=None, end=None, interval=1, verbose=False):
        if start is None:
            start = self._idx_zero_time
        if end is None:
            end = self.n_frames
        self._set_n_params()
        self._idx_fitted_frames = list()
        self._fitted_times = list()
        self._fitting_parameters = list()
        idx_t = 0
        r_sig = self.r_stim + 2 * self.pixel_width
        r_noise = self.r_max - 2 * self.pixel_width
        for f in range(start, end, interval):
            img = self.images[f] - self.background
            # Estimate the signal and noise
            signal_mask = (self.r > self.r_stim) & (self.r < r_sig)
            signal = img[signal_mask].mean()
            noise_mask = self.r > r_noise
            noise = img[noise_mask].std()
            signal_to_noise = signal / noise
            if signal_to_noise < 3:
                if verbose:
                    print("stopping at frame {} time {} signal {} noise {} signal/noise {} < 3".format(f, self.times[f], signal, noise, signal_to_noise))
                break
            fit_parms = self._fit_step1(img, signal)
            if verbose:
                print("frame {} time {} signal {} noise {} fit_parms {}".format(f,self.times[f], signal, noise, fit_parms))
            self._idx_fitted_frames.append(f)
            self._fitting_parameters.append(fit_parms)
            idx_t += 1
        self._fitting_parameters = np.array(self._fitting_parameters)
        linr_res, Ds, t0 = self._fit_step2(self.times[self._idx_fitted_frames],
                            self._fitting_parameters[:,-1])
        self._linr_res = linr_res
        self._Ds = Ds * 1e-8 # 1e-8 converts from um^2/s to cm^2/s
        self._t0 = t0

        return self._Ds

    @staticmethod
    def linear_model(time, diff_coeff, t0):
        return 4 * diff_coeff * (time + t0)

    @abstractmethod
    def model(self, r, param):
        """The model for the intensity distribution.
        """
        pass

    @abstractmethod
    def _set_n_params(self):
        self._n_params = 2
        return

    def _fit_step1(self, image, signal):
        """Non-linear fit of the images."""
        rmask = self.rmask_stim_out
        def cost(theta):
            if (theta < 0).any():
                return np.inf
            I_fit = self.model(self.r, *theta)
            I_exp = image
            sse = np.sum((I_exp[rmask] - I_fit[rmask])**2)
            return sse
        initial_guess = list()
        initial_guess.append(signal)
        for i in range(1, self._n_params):
            initial_guess.append(100)
        initial_guess = np.array(initial_guess)
        opt_res = minimize(cost, initial_guess, method='Nelder-Mead')
        return opt_res.x

    def _fit_step2(self, times, gamma):
        """Linear fit of gamma^2 vs. time."""
        linr_res = linregress(times, gamma**2)
        # run it
        Ds_fit = linr_res.slope / 4
        t0_fit = linr_res.intercept / (4 * Ds_fit)
        return linr_res, Ds_fit, t0_fit


    def radial_average(self, intensities, delta_r=None):
        r_min = self.r_stim
        r_max = self.r_max
        if delta_r is None:
            delta_r = 15*self.pixel_width
        r_edges = np.linspace(r_min, r_max, 1+int((r_max-r_min)/delta_r), endpoint=True)
        r_centers = r_edges[:-1] + 0.5 * delta_r
        rad_avg = list()
        rad_std = list()
        for i in range(len(r_centers)):
            j = i + 1
            rad_mask = (self.r >= r_edges[i]) & ( self.r < r_edges[j])
            rad_avg.append(intensities[rad_mask].mean())
            rad_std.append(intensities[rad_mask].std())
        return r_centers, np.array(rad_avg), np.array(rad_std)


    def line_average(self, intensities):
        r_max = self.r_max
        idx_low = self._idx_img_center[0]-10
        idx_high = self._idx_img_center[0]+10
        I_line_x = intensities[idx_low:idx_high,:].mean(axis=0)
        #r_ex = self.x_centers-self.img_center[1]
        #r_ex_mask = (r_ex > -r_max) & (r_ex < r_max)
        I_line_y = intensities[:, idx_low:idx_high].mean(axis=1)
        #r_ey = y_centers-img_center[0]
        #r_ey_mask = (r_ey > -r_max) & (r_ey < r_max)
        # Assuming the images are square.
        I_line = 0.5*(I_line_x + I_line_y)
        return I_line

    @abstractmethod
    def display_image_fits(self, n_rows = 5, vmax = None, ring_roi_width=None, saveas=None):
        pass

    def display_linear_fit(self, saveas=None):
        
        t_v = self.times[self._idx_fitted_frames]
        gamma_vals = self._fitting_parameters[:,-1]
        R2_fit = self._linr_res.rvalue**2
        Ds_fit = self._Ds
        t0_fit = self._t0
        # Generate the plot for the gamma^2 linear fit - IOI step 2 fitting
        plt.plot(t_v, gamma_vals**2, marker='o', linestyle="", label=None,
                 color='grey')
        tspan = np.linspace(0, np.max(t_v) * 1.1, 500)
        plt.plot(tspan, self.linear_model(tspan, self._Ds * 1e8, self._t0),
                 linestyle='--', label='Fit', color='k')
        plt.ylabel(r'$\gamma^2$')
        plt.xlabel('Time (s)')
        plt.legend(loc=0, frameon=False)
        plt.title("Step 2 - linear fit of $\gamma^2$ vs. $t$ \n $R^2$={:.3f} | D={:.1f} x$10^{{-7}}$ cm$^2$/s | $t_0$={:.2f} s".format(R2_fit, Ds_fit*1e7, t0_fit), pad=20)
        plt.tight_layout()
        sns.despine()
        if saveas is not None:
            plt.savefig(saveas)

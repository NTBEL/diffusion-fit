"""Base class for diffusion fitting.
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
        self.n_pixels = np.prod(self.images[0].shape)
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
        x_max = np.max(self.x_centers - self.img_center[1])
        y_max = np.max(self.y_centers - self.img_center[0])
        self.r_max = np.min( [x_max, y_max] )
        self.rmask_rmax_in = self.r < self.r_max
        self.fitting_mask = self.rmask_stim_out & self.rmask_rmax_in
        self.n_fitted_pixels = np.prod(self.r[self.fitting_mask].shape)
        if y_max < x_max:
            self._line = self.y_centers - self.img_center[0]
            self._min_dim = 'y'
        else:
            self._line = self.x_centers - self.img_center[1]
            self._min_dim = 'x'
        if self._idx_stimulation > 2:
            self.background = self.images[:self._idx_stimulation-1].mean(axis=0)
            #print("background ",self.background.shape)
        else:
            self.background = 0
            #print("background ", self.background)
        #self.background = background

        self._idx_fitted_frames = None
        self._fitting_parameters = None
        self._fitting_scores = None
        self._linr_res = None
        self._Ds = None
        self._t0 = None
        self._n_params = None

        return



    def fit(self, start=None, end=None, interval=1, verbose=False, s_to_n=3):
        if start is None:
            start = self._idx_zero_time
        if end is None:
            end = self.n_frames
        self._set_n_params()
        self._idx_fitted_frames = list()
        self._fitted_times = list()
        self._fitting_parameters = list()
        self._fitting_scores = list()
        r_sig = self.r_stim + 5 * self.pixel_width
        x_line = self._line
        r_line = np.abs(x_line)
        r_noise = np.max(r_line) - 10 * self.pixel_width
        for f in range(start, end, interval):
            img = self.images[f] - self.background
            I_line = self.line_average(img)
            # Estimate the signal and noise
            signal_mask = (r_line > (self.r_stim + self.pixel_width)) & (r_line < r_sig)
            signal = I_line[signal_mask].mean()
            noise_mask = r_line > r_noise
            noise = np.abs(I_line[noise_mask]).mean()
            signal_to_noise = signal / noise
            if signal_to_noise < s_to_n:
                if verbose:
                    print("stopping at frame {} time {} signal {} noise {} signal/noise {} < {}".format(f, self.times[f], signal, noise, signal_to_noise, s_to_n))
                break
            fit_parms, sse, rmse = self._fit_step1(img, signal)
            rmse = np.sqrt(sse / self.n_pixels)
            #er = self.error_rate(img, fit_parms, rmse)
            rsse = self.rsse(img, fit_parms)
            if verbose:
                print("frame {} time {} signal {} noise {} fit_parms {} RMSE {} RSSE {:.1f}".format(f,self.times[f], signal, noise, fit_parms, rmse, rsse))
            self._idx_fitted_frames.append(f)
            self._fitting_parameters.append(fit_parms)
            self._fitting_scores.append([rmse, rsse])
        if len(self._fitting_parameters) == 0:
            return np.nan
        self._fitting_parameters = np.array(self._fitting_parameters)
        self._fitting_scores = np.array(self._fitting_scores)
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

    def error_rate(self, image, theta, rmse):
        rmask = self.fitting_mask
        I_fit = self.model(self.r[rmask], *theta)
        I_exp = image
        abs_error = np.abs(I_exp[rmask] - I_fit)
        ci_val = 3 * rmse
        err_rate = 100 * np.prod(abs_error[abs_error > ci_val].shape) / self.n_fitted_pixels
        return err_rate

    def rsse(self, image, theta):
        rmask = self.fitting_mask
        I_fit = self.model(self.r[rmask], *theta)
        I_exp = image
        sse = np.std((I_exp[rmask] - I_fit)**2)
        return np.sqrt(sse)

    def _fit_step1(self, image, signal):
        """Non-linear fit of the images."""
        rmask = self.fitting_mask
        def cost(theta):
            if (theta < 0).any():
                return np.inf
            I_fit = self.model(self.r[rmask], *theta)
            I_exp = image
            sse = np.sum((I_exp[rmask] - I_fit)**2)
            return sse
        initial_guess = list()
        initial_guess.append(signal)
        for i in range(1, self._n_params):
            initial_guess.append(100)
        initial_guess = np.array(initial_guess)
        opt_res = minimize(cost, initial_guess, method='Nelder-Mead')
        # Sum of squared error from the minimized cost fucntion.
        sse = opt_res.fun
        # Root mean squared error.
        rmse = np.sqrt(sse / self.n_fitted_pixels)
        return opt_res.x, sse, rmse

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
        if self._min_dim == 'x':
            idx_low = self._idx_img_center[0]-10
            idx_high = self._idx_img_center[0]+10
            I_line_x = intensities[idx_low:idx_high,:].mean(axis=0)
            return I_line_x
        else:
            idx_low = self._idx_img_center[1]-10
            idx_high = self._idx_img_center[1]+10
            I_line_y = intensities[:,idx_low:idx_high].mean(axis=1)
            return I_line_y

    @property
    def time_resolved_diffusion(self):
        return None

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

    @property
    @abstractmethod
    def fitting_parameters(self):
        pass

    def export_to_csv(self, prefix):
        fp_df = self.fitting_parameters
        fp_df.to_csv(prefix+'_step_1_fits.csv', index=False)
        fp_df_step2 = fp_df[['Time','Gamma^2']]
        lin_fit = self.linear_model(fp_df['Time'].values, self._Ds * 1e8, self._t0)
        fp_df_step2['Linear-Fit'] = lin_fit.tolist()
        fp_df_step2.to_csv(prefix+'_step_2_fits.csv', index=False)

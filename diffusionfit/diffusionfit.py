"""
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from .dfbase import DiffusionFitBase
from . import models
from . import measure

def _estimate_loss_rate(t, intensity, t0_max=10):

    # Fit with loss rate
    popt_k, pcov_k = curve_fit(models.log_intensity_withloss, t, np.log(intensity), bounds=[[0, -np.inf, 0], [1, np.inf, t0_max]])
    # Fit a fixed zero loss rate
    popt_k0, pcov_k0 = curve_fit(models.log_intensity_noloss, t, np.log(intensity), bounds=[[-np.inf, 0], [np.inf, t0_max]])
    # Compute the sum of squared error for the two fits.
    sse_k = measure.ss_error(intensity, np.exp(models.log_intensity_withloss(t, *popt_k)))
    sse_k0 = measure.ss_error(intensity, np.exp(models.log_intensity_noloss(t, *popt_k0)))
    # Now compute their Akaike information criterion for model comparison.
    aic_k = measure.akaike_ic(-sse_k, 3) # with loss
    aic_k0 = measure.akaike_ic(-sse_k0, 2) # no loss
    kprime = 0
    otherparm = popt_k0[1:]
    if aic_k < aic_k0:
        kprime = popt_k[0]
        otherparm = popt_k[1:]
    return kprime, popt_k, popt_k0, sse_k, sse_k0, aic_k, aic_k0

class GaussianFit(DiffusionFitBase):

    @staticmethod
    def intensity_model(r, E, gamma):
        """Gaussian diffusion function.
        """
        return models.gaussian(r, E, gamma)

    def _set_n_params(self):
        self._n_params = 2
        return

    def display_image_fits(self, n_rows = 5, vmax = None, ring_roi_width=None, saveas=None):
        t_v = self.times[self._idx_fitted_frames]
        ntimes = len(t_v)
        rows = n_rows
        interval = int(ntimes/n_rows) #+ 1
        if interval == 0:
            interval = 1
        columns = 4
        counter = 0
        #print(rows, columns)
        f_height = 4*rows
        f, axes = plt.subplots(rows, columns, figsize=(18, f_height), sharex=False, sharey=False)
        #x = np.linspace(-self.r_max, self.r_max, 201, endpoint=True)
        #r = np.abs(x)*1e-4
        row = 0
        lineROI_zero_max = None
        ringROI_zero_max = None
        if vmax is None:
            vmax = self._fitting_parameters[:, 0].max()
        xhi = 0.5 * self.x_edges.max()
        xlow = -xhi
        yhi = 0.5 * self.y_edges.max()
        ylow = -yhi
        extent = [xlow, xhi, ylow, yhi]
        r_ex = self._line
        for i in range(0, ntimes, interval):
            if row >= n_rows: break
            time = t_v[i]
            #tcol = time_col[t_idx]
            E = self._fitting_parameters[i][0]
            gamma = self._fitting_parameters[i][1]
            rmse = self._fitting_scores[i][0]
            rsse = self._fitting_scores[i][1]
            dF_sim = self.intensity_model(self.r, *self._fitting_parameters[i])
            image = self.images[self._idx_fitted_frames[i]] - self.background
            axes[row, 0].imshow(image, cmap='viridis', vmin=0, vmax=1.5*vmax, extent=extent)
            axes[row, 0].set_xlabel(r'x ($\mu$m)', fontsize=14)
            axes[row, 0].set_ylabel(r'y ($\mu$m)', fontsize=14)
            axes[row, 0].set_title("Exp. Image\nTime: {:.2f} s".format(time), fontdict={'fontsize':10}, pad=10)
            axes[row, 1].imshow(dF_sim, cmap='viridis', vmin=0, vmax=vmax, extent=extent)
            axes[row, 1].set_xlabel(r'x ($\mu$m)', fontsize=14)
            axes[row, 1].set_ylabel(r'y ($\mu$m)', fontsize=14)
            axes[row, 1].set_title("2D Gaussian Fit\nE: {:.1e} | $\gamma$: {:.1e} | RMSE: {:.1f} | RSSE: {:.1f}".format(E, gamma, rmse, rsse), fontdict={'fontsize':10}, pad=10)
            I_line_roi_exp = self.line_average(image)
            r_centers, I_ring_roi_exp, std_ring_roi_exp = self.radial_average(image)
            I_line_roi_fit = self.line_average(dF_sim)
            r_centers, I_ring_roi_fit, std_ring_roi_fit = self.radial_average(dF_sim)
            if i == 0:
                lineROI_zero_max = np.max(I_line_roi_exp)
                ringROI_zero_max = np.max(I_ring_roi_exp)
            #r_ex_mask2 = (r_ex[r_ex_mask] < r_stim) & (r_ex[r_ex_mask] > -r_stim)
            axes[row, 2].fill_between([-self.r_stim, self.r_stim],
                                      1.5*lineROI_zero_max, label='STIM',
                                      color='r', alpha=0.5)
            axes[row, 2].plot(r_ex, I_line_roi_exp, linewidth=2, label='Exp.', color='grey')
            axes[row, 2].plot(r_ex, I_line_roi_fit, linewidth=4, label='from Fit', color='k')
            axes[row, 2].set_title("Line ROI", fontdict={'fontsize':10}, pad=10)
            axes[row, 2].set_xlabel(r'Position ($\mu$m)', fontsize=14)
            axes[row, 2].set_ylabel(r'$\Delta F$', fontsize=14)
            axes[row, 2].set_ylim((0, 1.25*lineROI_zero_max))
            axes[row, 2].legend(loc=0, fontsize=10)
            # Radial averages (Ring ROI)
            axes[row, 3].fill_between([0, self.r_stim], 1.2*ringROI_zero_max,
                                       label='STIM', color='r', alpha=0.5)
            axes[row, 3].plot(r_centers, I_ring_roi_exp, linewidth=2,
                              label='Exp.', linestyle="", marker='8',
                              markersize=10, color='grey')
            #r_centers, fit_r_avg, fit_r_std = self.radial_average(r_img, dF_sim)
            axes[row, 3].plot(r_centers, I_ring_roi_fit, linewidth=4,
                              label='from Fit',  linestyle='--', color='k')
            axes[row, 3].set_title("Ring ROI", fontdict={'fontsize':10}, pad=10)
            axes[row, 3].set_xlabel(r'Distance ($\mu$m)', fontsize=14)
            axes[row, 3].set_ylabel(r'$\Delta F$', fontsize=14)
            axes[row, 3].set_ylim((0, 1.1*ringROI_zero_max))
            axes[row, 3].legend(loc=0, fontsize=10)
            row += 1
        f.add_subplot(111, frameon=False)
        f.subplots_adjust(wspace=0.35)
        f.subplots_adjust(hspace=0.55)
        # hide tick and tick label of the big axes
        plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off', length=0)
        plt.grid(False)
        plt.title("Step 1 - 2D Gaussian fits over time", pad=40)
        plt.tight_layout()
        if saveas is not None:
            plt.savefig(saveas)
        return

    def display_image_fits_at_times(self, time_points, vmax = None, ring_roi_width=None, saveas=None):
        t_v = self.times[self._idx_fitted_frames]
        ntimes = len(t_v)
        n_rows = len(time_points)
        rows = n_rows
        findex = list()
        for time in time_points:
            try:
                idx = np.where(t_v == time)[0][0]
                findex.append(idx)
            except:
                continue
        columns = 4
        counter = 0
        #print(rows, columns)
        f_height = 4*rows
        f, axes = plt.subplots(rows, columns, figsize=(18, f_height), sharex=False, sharey=False)
        #x = np.linspace(-self.r_max, self.r_max, 201, endpoint=True)
        #r = np.abs(x)*1e-4
        row = 0
        lineROI_zero_max = None
        ringROI_zero_max = None
        if vmax is None:
            vmax = self._fitting_parameters[:, 0].max()
        xhi = 0.5 * self.x_edges.max()
        xlow = -xhi
        yhi = 0.5 * self.y_edges.max()
        ylow = -yhi
        extent = [xlow, xhi, ylow, yhi]
        r_ex = self._line
        for i in findex:
            if row >= n_rows: break
            time = t_v[i]
            #tcol = time_col[t_idx]
            E = self._fitting_parameters[i][0]
            gamma = self._fitting_parameters[i][1]
            rmse = self._fitting_scores[i][0]
            rsse = self._fitting_scores[i][1]
            dF_sim = self.intensity_model(self.r, *self._fitting_parameters[i])
            image = self.images[self._idx_fitted_frames[i]] - self.background
            axes[row, 0].imshow(image, cmap='viridis', vmin=0, vmax=1.5*vmax, extent=extent)
            axes[row, 0].set_xlabel(r'x ($\mu$m)', fontsize=14)
            axes[row, 0].set_ylabel(r'y ($\mu$m)', fontsize=14)
            axes[row, 0].set_title("Exp. Image\nTime: {:.2f} s".format(time), fontdict={'fontsize':10}, pad=10)
            axes[row, 1].imshow(dF_sim, cmap='viridis', vmin=0, vmax=vmax, extent=extent)
            axes[row, 1].set_xlabel(r'x ($\mu$m)', fontsize=14)
            axes[row, 1].set_ylabel(r'y ($\mu$m)', fontsize=14)
            axes[row, 1].set_title("2D Gaussian Fit\nE: {:.1e} | $\gamma$: {:.1e} | RMSE: {:.1f} | RSSE: {:.1f}".format(E, gamma, rmse, rsse), fontdict={'fontsize':10}, pad=10)
            I_line_roi_exp = self.line_average(image)
            r_centers, I_ring_roi_exp, std_ring_roi_exp = self.radial_average(image)
            I_line_roi_fit = self.line_average(dF_sim)
            r_centers, I_ring_roi_fit, std_ring_roi_fit = self.radial_average(dF_sim)
            
            if i == 0:
                lineROI_zero_max = np.max(I_line_roi_exp)
                ringROI_zero_max = np.max(I_ring_roi_exp)
            #r_ex_mask2 = (r_ex[r_ex_mask] < r_stim) & (r_ex[r_ex_mask] > -r_stim)
            axes[row, 2].fill_between([-self.r_stim, self.r_stim],
                                      1.5*lineROI_zero_max, label='STIM',
                                      color='r', alpha=0.5)
            axes[row, 2].plot(r_ex, I_line_roi_exp, linewidth=2, label='Exp.', color='grey')
            axes[row, 2].plot(r_ex, I_line_roi_fit, linewidth=4, label='from Fit', color='k')
            axes[row, 2].set_title("Line ROI", fontdict={'fontsize':10}, pad=10)
            axes[row, 2].set_xlabel(r'Position ($\mu$m)', fontsize=14)
            axes[row, 2].set_ylabel(r'$\Delta F$', fontsize=14)
            axes[row, 2].set_ylim((0, 1.25*lineROI_zero_max))
            axes[row, 2].legend(loc=0, fontsize=10)
            # Radial averages (Ring ROI)
            axes[row, 3].fill_between([0, self.r_stim], 1.2*ringROI_zero_max,
                                       label='STIM', color='r', alpha=0.5)
            axes[row, 3].plot(r_centers, I_ring_roi_exp, linewidth=2,
                              label='Exp.', linestyle="", marker='8',
                              markersize=10, color='grey')
            #r_centers, fit_r_avg, fit_r_std = self.radial_average(r_img, dF_sim)
            axes[row, 3].plot(r_centers, I_ring_roi_fit, linewidth=4,
                              label='from Fit',  linestyle='--', color='k')
            axes[row, 3].set_title("Ring ROI", fontdict={'fontsize':10}, pad=10)
            axes[row, 3].set_xlabel(r'Distance ($\mu$m)', fontsize=14)
            axes[row, 3].set_ylabel(r'$\Delta F$', fontsize=14)
            axes[row, 3].set_ylim((0, 1.1*ringROI_zero_max))
            axes[row, 3].legend(loc=0, fontsize=10)
            row += 1
        f.add_subplot(111, frameon=False)
        f.subplots_adjust(wspace=0.35)
        f.subplots_adjust(hspace=0.55)
        # hide tick and tick label of the big axes
        plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off', length=0)
        plt.grid(False)
        plt.title("Step 1 - 2D Gaussian fits over time", pad=40)
        plt.tight_layout()
        if saveas is not None:
            plt.savefig(saveas)
        return

    @property
    def fitting_parameters(self):
        t_v = self.times[self._idx_fitted_frames]
        E_vals = self._fitting_parameters[:, 0]
        gamma_vals = self._fitting_parameters[:, -1]
        RMSE = self._fitting_scores[:, 0]
        RSSE = self._fitting_scores[:, 1]
        fp_vals = list()
        for i in range(len(t_v)):
            fp_vals.append({"Time":t_v[i], "E":E_vals[i], "Gamma":gamma_vals[i],
                            "Gamma^2":gamma_vals[i]**2, "RMSE":RMSE[i],
                            "RSSE":RSSE[i]})
        return pd.DataFrame(fp_vals)


    def estimate_loss_rate(self):
        t_v = self.times[self._idx_fitted_frames]
        E_vals = self._fitting_parameters[:, 0]
        loss_rate_data = _estimate_loss_rate(t_v, E_vals)
        self._loss_rate_data = loss_rate_data
        self._loss_rate = loss_rate_data[0]
        return self._loss_rate


class PointClarkFit(DiffusionFitBase):

    @staticmethod
    def intensity_model(r, Emax, beta, gamma):
        """Point-Clark diffusion distribution function for receptor-based sensors.
        """
        return models.point_clark(r, Emax, beta, gamma)

    def _set_n_params(self):
        self._n_params = 3
        return

    def display_image_fits(self, n_rows = 5, vmax = None, ring_roi_width=None, saveas=None):
        t_v = self.times[self._idx_fitted_frames]
        ntimes = len(t_v)
        rows = n_rows
        interval = int(ntimes/n_rows) #+ 1
        if interval == 0:
            interval = 1
        columns = 4
        counter = 0
        #print(rows, columns)
        f_height = 4*rows
        f, axes = plt.subplots(rows, columns, figsize=(18, f_height), sharex=False, sharey=False)
        #x = np.linspace(-self.r_max, self.r_max, 201, endpoint=True)
        #r = np.abs(x)*1e-4
        row = 0
        lineROI_zero_max = None
        ringROI_zero_max = None
        if vmax is None:
            vmax = self._fitting_parameters[:, 0].max()
        xhi = 0.5 * self.x_edges.max()
        xlow = -xhi
        yhi = 0.5 * self.y_edges.max()
        ylow = -yhi
        extent = [xlow, xhi, ylow, yhi]
        r_ex = self._line
        for i in range(0, ntimes, interval):
            if row >= n_rows: break
            time = t_v[i]
            #tcol = time_col[t_idx]
            Emax = self._fitting_parameters[i][0]
            beta = self.fitting_parameters[i][1]
            gamma = self._fitting_parameters[i][2]
            rmse = self._fitting_scores[i][0]
            rsse = self._fitting_scores[i][1]
            dF_sim = self.intensity_model(self.r, *self._fitting_parameters[i])
            image = self.images[self._idx_fitted_frames[i]] - self.background
            axes[row, 0].imshow(image, cmap='viridis', vmin=0, vmax=1.5*vmax, extent=extent)
            axes[row, 0].set_xlabel(r'x ($\mu$m)', fontsize=14)
            axes[row, 0].set_ylabel(r'y ($\mu$m)', fontsize=14)
            axes[row, 0].set_title("Exp. Image\nTime: {:.2f} s".format(time), fontdict={'fontsize':10}, pad=10)
            axes[row, 1].imshow(dF_sim, cmap='viridis', vmin=0, vmax=vmax, extent=extent)
            axes[row, 1].set_xlabel(r'x ($\mu$m)', fontsize=14)
            axes[row, 1].set_ylabel(r'y ($\mu$m)', fontsize=14)
            axes[row, 1].set_title("2D Point-Clark Fit\n$E_m$: {:.1e} | $\beta$: {:.1e} | $\gamma$: {:.1e} | RMSE: {:.1f} | RSSE: {:.1f}".format(Emax, beta, gamma, rmse, rsse), fontdict={'fontsize':10}, pad=10)
            I_line_roi_exp = self.line_average(image)
            r_centers, I_ring_roi_exp, std_ring_roi_exp = self.radial_average(image)
            I_line_roi_fit = self.line_average(dF_sim)
            r_centers, I_ring_roi_fit, std_ring_roi_fit = self.radial_average(dF_sim)
            if i == 0:
                lineROI_zero_max = np.max(I_line_roi_exp)
                ringROI_zero_max = np.max(I_ring_roi_exp)
            #r_ex_mask2 = (r_ex[r_ex_mask] < r_stim) & (r_ex[r_ex_mask] > -r_stim)
            axes[row, 2].fill_between([-self.r_stim, self.r_stim],
                                      1.5*lineROI_zero_max, label='STIM',
                                      color='r', alpha=0.5)
            axes[row, 2].plot(r_ex, I_line_roi_exp, linewidth=2, label='Exp.', color='grey')
            axes[row, 2].plot(r_ex, I_line_roi_fit, linewidth=4, label='from Fit', color='k')
            axes[row, 2].set_title("Line ROI", fontdict={'fontsize':10}, pad=10)
            axes[row, 2].set_xlabel(r'Position ($\mu$m)', fontsize=14)
            axes[row, 2].set_ylabel(r'$\Delta F$', fontsize=14)
            axes[row, 2].set_ylim((0, 1.25*lineROI_zero_max))
            axes[row, 2].legend(loc=0, fontsize=10)
            # Radial averages (Ring ROI)
            axes[row, 3].fill_between([0, self.r_stim], 1.2*ringROI_zero_max,
                                       label='STIM', color='r', alpha=0.5)
            axes[row, 3].plot(r_centers, I_ring_roi_exp, linewidth=2,
                              label='Exp.', linestyle="", marker='8',
                              markersize=10, color='grey')
            #r_centers, fit_r_avg, fit_r_std = self.radial_average(r_img, dF_sim)
            axes[row, 3].plot(r_centers, I_ring_roi_fit, linewidth=4,
                              label='from Fit',  linestyle='--', color='k')
            axes[row, 3].set_title("Ring ROI", fontdict={'fontsize':10}, pad=10)
            axes[row, 3].set_xlabel(r'Distance ($\mu$m)', fontsize=14)
            axes[row, 3].set_ylabel(r'$\Delta F$', fontsize=14)
            axes[row, 3].set_ylim((0, 1.1*ringROI_zero_max))
            axes[row, 3].legend(loc=0, fontsize=10)
            row += 1
        f.add_subplot(111, frameon=False)
        f.subplots_adjust(wspace=0.35)
        f.subplots_adjust(hspace=0.55)
        # hide tick and tick label of the big axes
        plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off', length=0)
        plt.grid(False)
        plt.title("Step 1 - 2D Point-Clark fits over time", pad=40)
        plt.tight_layout()
        if saveas is not None:
            plt.savefig(saveas)
        return

    def display_image_fits_at_times(self, time_points, vmax = None, ring_roi_width=None, saveas=None):
        t_v = self.times[self._idx_fitted_frames]
        ntimes = len(t_v)
        rows = len(time_points)
        findex = list()
        for time in time_points:
            idx = np.where(t_v == time)[0][0]
            findex.append(idx)
        columns = 4
        counter = 0
        #print(rows, columns)
        f_height = 4*rows
        f, axes = plt.subplots(rows, columns, figsize=(18, f_height), sharex=False, sharey=False)
        #x = np.linspace(-self.r_max, self.r_max, 201, endpoint=True)
        #r = np.abs(x)*1e-4
        row = 0
        lineROI_zero_max = None
        ringROI_zero_max = None
        if vmax is None:
            vmax = self._fitting_parameters[:, 0].max()
        xhi = 0.5 * self.x_edges.max()
        xlow = -xhi
        yhi = 0.5 * self.y_edges.max()
        ylow = -yhi
        extent = [xlow, xhi, ylow, yhi]
        r_ex = self._line
        for i in findex:
            if row >= n_rows: break
            time = t_v[i]
            #tcol = time_col[t_idx]
            Emax = self._fitting_parameters[i][0]
            beta = self.fitting_parameters[i][1]
            gamma = self._fitting_parameters[i][2]
            rmse = self._fitting_scores[i][0]
            rsse = self._fitting_scores[i][1]
            dF_sim = self.intensity_model(self.r, *self._fitting_parameters[i])
            image = self.images[self._idx_fitted_frames[i]] - self.background
            axes[row, 0].imshow(image, cmap='viridis', vmin=0, vmax=1.5*vmax, extent=extent)
            axes[row, 0].set_xlabel(r'x ($\mu$m)', fontsize=14)
            axes[row, 0].set_ylabel(r'y ($\mu$m)', fontsize=14)
            axes[row, 0].set_title("Exp. Image\nTime: {:.2f} s".format(time), fontdict={'fontsize':10}, pad=10)
            axes[row, 1].imshow(dF_sim, cmap='viridis', vmin=0, vmax=vmax, extent=extent)
            axes[row, 1].set_xlabel(r'x ($\mu$m)', fontsize=14)
            axes[row, 1].set_ylabel(r'y ($\mu$m)', fontsize=14)
            axes[row, 1].set_title("2D Point-Clark Fit\n$E_m$: {:.1e} | $\beta$: {:.1e} | $\gamma$: {:.1e} | RMSE: {:.1f} | RSSE: {:.1f}".format(Emax, beta, gamma, rmse, rsse), fontdict={'fontsize':10}, pad=10)
            I_line_roi_exp = self.line_average(image)
            r_centers, I_ring_roi_exp, std_ring_roi_exp = self.radial_average(image)
            I_line_roi_fit = self.line_average(dF_sim)
            r_centers, I_ring_roi_fit, std_ring_roi_fit = self.radial_average(dF_sim)
            if i == 0:
                lineROI_zero_max = np.max(I_line_roi_exp)
                ringROI_zero_max = np.max(I_ring_roi_exp)
            #r_ex_mask2 = (r_ex[r_ex_mask] < r_stim) & (r_ex[r_ex_mask] > -r_stim)
            axes[row, 2].fill_between([-self.r_stim, self.r_stim],
                                      1.5*lineROI_zero_max, label='STIM',
                                      color='r', alpha=0.5)
            axes[row, 2].plot(r_ex, I_line_roi_exp, linewidth=2, label='Exp.', color='grey')
            axes[row, 2].plot(r_ex, I_line_roi_fit, linewidth=4, label='from Fit', color='k')
            axes[row, 2].set_title("Line ROI", fontdict={'fontsize':10}, pad=10)
            axes[row, 2].set_xlabel(r'Position ($\mu$m)', fontsize=14)
            axes[row, 2].set_ylabel(r'$\Delta F$', fontsize=14)
            axes[row, 2].set_ylim((0, 1.25*lineROI_zero_max))
            axes[row, 2].legend(loc=0, fontsize=10)
            # Radial averages (Ring ROI)
            axes[row, 3].fill_between([0, self.r_stim], 1.2*ringROI_zero_max,
                                       label='STIM', color='r', alpha=0.5)
            axes[row, 3].plot(r_centers, I_ring_roi_exp, linewidth=2,
                              label='Exp.', linestyle="", marker='8',
                              markersize=10, color='grey')
            #r_centers, fit_r_avg, fit_r_std = self.radial_average(r_img, dF_sim)
            axes[row, 3].plot(r_centers, I_ring_roi_fit, linewidth=4,
                              label='from Fit',  linestyle='--', color='k')
            axes[row, 3].set_title("Ring ROI", fontdict={'fontsize':10}, pad=10)
            axes[row, 3].set_xlabel(r'Distance ($\mu$m)', fontsize=14)
            axes[row, 3].set_ylabel(r'$\Delta F$', fontsize=14)
            axes[row, 3].set_ylim((0, 1.1*ringROI_zero_max))
            axes[row, 3].legend(loc=0, fontsize=10)
            row += 1
        f.add_subplot(111, frameon=False)
        f.subplots_adjust(wspace=0.35)
        f.subplots_adjust(hspace=0.55)
        # hide tick and tick label of the big axes
        plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off', length=0)
        plt.grid(False)
        plt.title("Step 1 - 2D Point-Clark fits over time", pad=40)
        plt.tight_layout()
        if saveas is not None:
            plt.savefig(saveas)
        return

    @property
    def fitting_parameters(self):
        t_v = self.times[self._idx_fitted_frames]
        E_vals = self._fitting_parameters[:, 0]
        beta_vals = self._fitting_parameters[:, 1]
        gamma_vals = self._fitting_parameters[:, -1]
        RMSE = self._fitting_scores[:, 0]
        RSSE = self._fitting_scores[:, 1]
        fp_vals = list()
        for i in range(len(t_v)):
            fp_vals.append({"Time":t_v[i], "Emax":E_vals[i], "Beta":beta_vals[i],
                            "Gamma":gamma_vals[i], "Gamma^2":gamma_vals[i]**2,
                            "RMSE":RMSE[i], "RSSE":RSSE[i]})
        return pd.DataFrame(fp_vals)

    def estimate_loss_rate(self):
        t_v = self.times[self._idx_fitted_frames]
        beta_vals = self._fitting_parameters[:, 1]
        loss_rate_data = _estimate_loss_rate(t_v, beta_vals)
        self._loss_rate_data = loss_rate_data
        self._loss_rate = loss_rate_data[0]
        return self._loss_rate

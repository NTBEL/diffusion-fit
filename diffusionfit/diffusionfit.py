"""
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit, minimize
from scipy.ndimage import gaussian_filter
from diffusionfit.dfbase import DiffusionFitBase
from . import models
from . import measure


def _estimate_loss_rate(t, intensity, t0_max=10):

    # Fit with loss rate
    popt_k, pcov_k = curve_fit(
        models.log_intensity_withloss,
        t,
        np.log(intensity),
        bounds=[[0, -np.inf, 0], [1, np.inf, t0_max]],
    )
    # Fit a fixed zero loss rate
    popt_k0, pcov_k0 = curve_fit(
        models.log_intensity_noloss,
        t,
        np.log(intensity),
        bounds=[[-np.inf, 0], [np.inf, t0_max]],
    )
    # Compute the sum of squared error for the two fits.
    sse_k = measure.ss_error(
        intensity, np.exp(models.log_intensity_withloss(t, *popt_k))
    )
    sse_k0 = measure.ss_error(
        intensity, np.exp(models.log_intensity_noloss(t, *popt_k0))
    )
    # Now compute their Akaike information criterion for model comparison.
    aic_k = measure.akaike_ic(-sse_k, 3)  # with loss
    aic_k0 = measure.akaike_ic(-sse_k0, 2)  # no loss
    kprime = 0
    otherparm = popt_k0[1:]
    if aic_k < aic_k0:
        kprime = popt_k[0]
        otherparm = popt_k[1:]
    return kprime, popt_k, popt_k0, sse_k, sse_k0, aic_k, aic_k0


class GaussianFit(DiffusionFitBase):
    @staticmethod
    def intensity_model(r, E, gamma):
        """Gaussian diffusion function."""
        return models.gaussian(r, E, gamma)

    def _set_n_params(self):
        self._n_params = 2
        return

    def display_image_fits(self, n_rows=5, vmax=None, ring_roi_width=None, saveas=None):
        t_v = self.times[self._idx_fitted_frames]
        ntimes = len(t_v)
        rows = n_rows
        interval = int(ntimes / n_rows)  # + 1
        if interval == 0:
            interval = 1
        columns = 4
        counter = 0
        # print(rows, columns)
        f_height = 4 * rows
        f, axes = plt.subplots(
            rows, columns, figsize=(18, f_height), sharex=False, sharey=False
        )
        # x = np.linspace(-self.r_max, self.r_max, 201, endpoint=True)
        # r = np.abs(x)*1e-4
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
            if row >= n_rows:
                break
            time = t_v[i]
            # tcol = time_col[t_idx]
            E = self._fitting_parameters[i][0]
            gamma = self._fitting_parameters[i][1]
            rmse = self._fitting_scores[i][0]
            rsse = self._fitting_scores[i][1]
            dF_sim = self.intensity_model(self.r, *self._fitting_parameters[i])
            image = self.images[self._idx_fitted_frames[i]] - self.background
            axes[row, 0].imshow(
                image, cmap="viridis", vmin=0, vmax=1.5 * vmax, extent=extent
            )
            axes[row, 0].set_xlabel(r"x ($\mu$m)", fontsize=14)
            axes[row, 0].set_ylabel(r"y ($\mu$m)", fontsize=14)
            axes[row, 0].set_title(
                "Exp. Image\nTime: {:.2f} s".format(time),
                fontdict={"fontsize": 10},
                pad=10,
            )
            axes[row, 1].imshow(
                dF_sim, cmap="viridis", vmin=0, vmax=vmax, extent=extent
            )
            axes[row, 1].set_xlabel(r"x ($\mu$m)", fontsize=14)
            axes[row, 1].set_ylabel(r"y ($\mu$m)", fontsize=14)
            axes[row, 1].set_title(
                "2D Gaussian Fit\nE: {:.1e} | $\gamma$: {:.1e} | RMSE: {:.1f} | RSSE: {:.1f}".format(
                    E, gamma, rmse, rsse
                ),
                fontdict={"fontsize": 10},
                pad=10,
            )
            I_line_roi_exp = self.line_average(image)
            r_centers, I_ring_roi_exp, std_ring_roi_exp = self.radial_average(image)
            I_line_roi_fit = self.line_average(dF_sim)
            r_centers, I_ring_roi_fit, std_ring_roi_fit = self.radial_average(dF_sim)
            if i == 0:
                lineROI_zero_max = np.max(I_line_roi_exp)
                ringROI_zero_max = np.max(I_ring_roi_exp)
            # r_ex_mask2 = (r_ex[r_ex_mask] < r_stim) & (r_ex[r_ex_mask] > -r_stim)
            axes[row, 2].fill_between(
                [-self.r_stim, self.r_stim],
                1.5 * lineROI_zero_max,
                label="STIM",
                color="r",
                alpha=0.5,
            )
            axes[row, 2].plot(
                r_ex, I_line_roi_exp, linewidth=2, label="Exp.", color="grey"
            )
            axes[row, 2].plot(
                r_ex, I_line_roi_fit, linewidth=4, label="from Fit", color="k"
            )
            axes[row, 2].set_title("Line ROI", fontdict={"fontsize": 10}, pad=10)
            axes[row, 2].set_xlabel(r"Position ($\mu$m)", fontsize=14)
            axes[row, 2].set_ylabel(r"$\Delta F$", fontsize=14)
            axes[row, 2].set_ylim((0, 1.25 * lineROI_zero_max))
            axes[row, 2].legend(loc=0, fontsize=10)
            # Radial averages (Ring ROI)
            axes[row, 3].fill_between(
                [0, self.r_stim],
                1.2 * ringROI_zero_max,
                label="STIM",
                color="r",
                alpha=0.5,
            )
            axes[row, 3].plot(
                r_centers,
                I_ring_roi_exp,
                linewidth=2,
                label="Exp.",
                linestyle="",
                marker="8",
                markersize=10,
                color="grey",
            )
            # r_centers, fit_r_avg, fit_r_std = self.radial_average(r_img, dF_sim)
            axes[row, 3].plot(
                r_centers,
                I_ring_roi_fit,
                linewidth=4,
                label="from Fit",
                linestyle="--",
                color="k",
            )
            axes[row, 3].set_title("Ring ROI", fontdict={"fontsize": 10}, pad=10)
            axes[row, 3].set_xlabel(r"Distance ($\mu$m)", fontsize=14)
            axes[row, 3].set_ylabel(r"$\Delta F$", fontsize=14)
            axes[row, 3].set_ylim((0, 1.1 * ringROI_zero_max))
            axes[row, 3].legend(loc=0, fontsize=10)
            row += 1
        f.add_subplot(111, frameon=False)
        f.subplots_adjust(wspace=0.35)
        f.subplots_adjust(hspace=0.55)
        # hide tick and tick label of the big axes
        plt.tick_params(
            labelcolor="none",
            top="off",
            bottom="off",
            left="off",
            right="off",
            length=0,
        )
        plt.grid(False)
        plt.title("Step 1 - 2D Gaussian fits over time", pad=40)
        plt.tight_layout()
        if saveas is not None:
            plt.savefig(saveas)
        return

    def display_image_fits_at_times(
        self, time_points, vmax=None, ring_roi_width=None, saveas=None
    ):
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
        # print(rows, columns)
        f_height = 4 * rows
        f, axes = plt.subplots(
            rows, columns, figsize=(18, f_height), sharex=False, sharey=False
        )
        # x = np.linspace(-self.r_max, self.r_max, 201, endpoint=True)
        # r = np.abs(x)*1e-4
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
            if row >= n_rows:
                break
            time = t_v[i]
            # tcol = time_col[t_idx]
            E = self._fitting_parameters[i][0]
            gamma = self._fitting_parameters[i][1]
            rmse = self._fitting_scores[i][0]
            rsse = self._fitting_scores[i][1]
            dF_sim = self.intensity_model(self.r, *self._fitting_parameters[i])
            image = self.images[self._idx_fitted_frames[i]] - self.background
            axes[row, 0].imshow(
                image, cmap="viridis", vmin=0, vmax=1.5 * vmax, extent=extent
            )
            axes[row, 0].set_xlabel(r"x ($\mu$m)", fontsize=14)
            axes[row, 0].set_ylabel(r"y ($\mu$m)", fontsize=14)
            axes[row, 0].set_title(
                "Exp. Image\nTime: {:.2f} s".format(time),
                fontdict={"fontsize": 10},
                pad=10,
            )
            axes[row, 1].imshow(
                dF_sim, cmap="viridis", vmin=0, vmax=vmax, extent=extent
            )
            axes[row, 1].set_xlabel(r"x ($\mu$m)", fontsize=14)
            axes[row, 1].set_ylabel(r"y ($\mu$m)", fontsize=14)
            axes[row, 1].set_title(
                "2D Gaussian Fit\nE: {:.1e} | $\gamma$: {:.1e} | RMSE: {:.1f} | RSSE: {:.1f}".format(
                    E, gamma, rmse, rsse
                ),
                fontdict={"fontsize": 10},
                pad=10,
            )
            I_line_roi_exp = self.line_average(image)
            r_centers, I_ring_roi_exp, std_ring_roi_exp = self.radial_average(image)
            I_line_roi_fit = self.line_average(dF_sim)
            r_centers, I_ring_roi_fit, std_ring_roi_fit = self.radial_average(dF_sim)

            if i == 0:
                lineROI_zero_max = np.max(I_line_roi_exp)
                ringROI_zero_max = np.max(I_ring_roi_exp)
            # r_ex_mask2 = (r_ex[r_ex_mask] < r_stim) & (r_ex[r_ex_mask] > -r_stim)
            axes[row, 2].fill_between(
                [-self.r_stim, self.r_stim],
                1.5 * lineROI_zero_max,
                label="STIM",
                color="r",
                alpha=0.5,
            )
            axes[row, 2].plot(
                r_ex, I_line_roi_exp, linewidth=2, label="Exp.", color="grey"
            )
            axes[row, 2].plot(
                r_ex, I_line_roi_fit, linewidth=4, label="from Fit", color="k"
            )
            axes[row, 2].set_title("Line ROI", fontdict={"fontsize": 10}, pad=10)
            axes[row, 2].set_xlabel(r"Position ($\mu$m)", fontsize=14)
            axes[row, 2].set_ylabel(r"$\Delta F$", fontsize=14)
            axes[row, 2].set_ylim((0, 1.25 * lineROI_zero_max))
            axes[row, 2].legend(loc=0, fontsize=10)
            # Radial averages (Ring ROI)
            axes[row, 3].fill_between(
                [0, self.r_stim],
                1.2 * ringROI_zero_max,
                label="STIM",
                color="r",
                alpha=0.5,
            )
            axes[row, 3].plot(
                r_centers,
                I_ring_roi_exp,
                linewidth=2,
                label="Exp.",
                linestyle="",
                marker="8",
                markersize=10,
                color="grey",
            )
            # r_centers, fit_r_avg, fit_r_std = self.radial_average(r_img, dF_sim)
            axes[row, 3].plot(
                r_centers,
                I_ring_roi_fit,
                linewidth=4,
                label="from Fit",
                linestyle="--",
                color="k",
            )
            axes[row, 3].set_title("Ring ROI", fontdict={"fontsize": 10}, pad=10)
            axes[row, 3].set_xlabel(r"Distance ($\mu$m)", fontsize=14)
            axes[row, 3].set_ylabel(r"$\Delta F$", fontsize=14)
            axes[row, 3].set_ylim((0, 1.1 * ringROI_zero_max))
            axes[row, 3].legend(loc=0, fontsize=10)
            row += 1
        f.add_subplot(111, frameon=False)
        f.subplots_adjust(wspace=0.35)
        f.subplots_adjust(hspace=0.55)
        # hide tick and tick label of the big axes
        plt.tick_params(
            labelcolor="none",
            top="off",
            bottom="off",
            left="off",
            right="off",
            length=0,
        )
        plt.grid(False)
        plt.title("Step 1 - 2D Gaussian fits over time", pad=40)
        plt.tight_layout()
        if saveas is not None:
            plt.savefig(saveas)
        return

    @property
    def fitting_parameters(self):
        """The intensity model fitting paramaters from step 1."""
        t_v = self.times[self._idx_fitted_frames]
        E_vals = self._fitting_parameters[:, 0]
        gamma_vals = self._fitting_parameters[:, -1]
        RMSE = self._fitting_scores[:, 0]
        RSSE = self._fitting_scores[:, 1]
        fp_vals = list()
        for i in range(len(t_v)):
            fp_vals.append(
                {
                    "Time": t_v[i],
                    "Imax": E_vals[i],
                    "Gamma": gamma_vals[i],
                    "Gamma^2": gamma_vals[i] ** 2,
                    "RMSE": RMSE[i],
                    "RSSE": RSSE[i],
                }
            )
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
        """Point-Clark diffusion distribution function for receptor-based sensors."""
        return models.point_clark(r, Emax, beta, gamma)

    def _set_n_params(self):
        self._n_params = 3
        return

    def display_image_fits(self, n_rows=5, vmax=None, ring_roi_width=None, saveas=None):
        t_v = self.times[self._idx_fitted_frames]
        ntimes = len(t_v)
        rows = n_rows
        interval = int(ntimes / n_rows)  # + 1
        if interval == 0:
            interval = 1
        columns = 4
        counter = 0
        # print(rows, columns)
        f_height = 4 * rows
        f, axes = plt.subplots(
            rows, columns, figsize=(18, f_height), sharex=False, sharey=False
        )
        # x = np.linspace(-self.r_max, self.r_max, 201, endpoint=True)
        # r = np.abs(x)*1e-4
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
            if row >= n_rows:
                break
            time = t_v[i]
            # tcol = time_col[t_idx]
            Emax = self._fitting_parameters[i][0]
            beta = self._fitting_parameters[i][1]
            gamma = self._fitting_parameters[i][2]
            rmse = self._fitting_scores[i][0]
            rsse = self._fitting_scores[i][1]
            dF_sim = self.intensity_model(self.r, *self._fitting_parameters[i])
            image = self.images[self._idx_fitted_frames[i]] - self.background
            axes[row, 0].imshow(
                image, cmap="viridis", vmin=0, vmax=1.5 * vmax, extent=extent
            )
            axes[row, 0].set_xlabel(r"x ($\mu$m)", fontsize=14)
            axes[row, 0].set_ylabel(r"y ($\mu$m)", fontsize=14)
            axes[row, 0].set_title(
                "Exp. Image\nTime: {:.2f} s".format(time),
                fontdict={"fontsize": 10},
                pad=10,
            )
            axes[row, 1].imshow(
                dF_sim, cmap="viridis", vmin=0, vmax=vmax, extent=extent
            )
            axes[row, 1].set_xlabel(r"x ($\mu$m)", fontsize=14)
            axes[row, 1].set_ylabel(r"y ($\mu$m)", fontsize=14)
            axes[row, 1].set_title(
                "2D Point-Clark Fit\n$I_m$: {:.1e} | Beta: {:.1e} | $\gamma$: {:.1e} | RMSE: {:.1f}".format(
                    Emax, beta, gamma, rmse
                ),
                fontdict={"fontsize": 10},
                pad=10,
            )
            I_line_roi_exp = self.line_average(image)
            r_centers, I_ring_roi_exp, std_ring_roi_exp = self.radial_average(image)
            I_line_roi_fit = self.line_average(dF_sim)
            r_centers, I_ring_roi_fit, std_ring_roi_fit = self.radial_average(dF_sim)
            if i == 0:
                lineROI_zero_max = np.max(I_line_roi_exp)
                ringROI_zero_max = np.max(I_ring_roi_exp)
            # r_ex_mask2 = (r_ex[r_ex_mask] < r_stim) & (r_ex[r_ex_mask] > -r_stim)
            axes[row, 2].fill_between(
                [-self.r_stim, self.r_stim],
                1.5 * lineROI_zero_max,
                label="STIM",
                color="r",
                alpha=0.5,
            )
            axes[row, 2].plot(
                r_ex, I_line_roi_exp, linewidth=2, label="Exp.", color="grey"
            )
            axes[row, 2].plot(
                r_ex, I_line_roi_fit, linewidth=4, label="from Fit", color="k"
            )
            axes[row, 2].set_title("Line ROI", fontdict={"fontsize": 10}, pad=10)
            axes[row, 2].set_xlabel(r"Position ($\mu$m)", fontsize=14)
            axes[row, 2].set_ylabel(r"$\Delta F$", fontsize=14)
            axes[row, 2].set_ylim((0, 1.25 * lineROI_zero_max))
            axes[row, 2].legend(loc=0, fontsize=10)
            # Radial averages (Ring ROI)
            axes[row, 3].fill_between(
                [0, self.r_stim],
                1.2 * ringROI_zero_max,
                label="STIM",
                color="r",
                alpha=0.5,
            )
            axes[row, 3].plot(
                r_centers,
                I_ring_roi_exp,
                linewidth=2,
                label="Exp.",
                linestyle="",
                marker="8",
                markersize=10,
                color="grey",
            )
            # r_centers, fit_r_avg, fit_r_std = self.radial_average(r_img, dF_sim)
            axes[row, 3].plot(
                r_centers,
                I_ring_roi_fit,
                linewidth=4,
                label="from Fit",
                linestyle="--",
                color="k",
            )
            axes[row, 3].set_title("Ring ROI", fontdict={"fontsize": 10}, pad=10)
            axes[row, 3].set_xlabel(r"Distance ($\mu$m)", fontsize=14)
            axes[row, 3].set_ylabel(r"$\Delta F$", fontsize=14)
            axes[row, 3].set_ylim((0, 1.1 * ringROI_zero_max))
            axes[row, 3].legend(loc=0, fontsize=10)
            row += 1
        f.add_subplot(111, frameon=False)
        f.subplots_adjust(wspace=0.35)
        f.subplots_adjust(hspace=0.55)
        # hide tick and tick label of the big axes
        plt.tick_params(
            labelcolor="none",
            top="off",
            bottom="off",
            left="off",
            right="off",
            length=0,
        )
        plt.grid(False)
        plt.title("Step 1 - 2D Point-Clark fits over time", pad=40)
        plt.tight_layout()
        if saveas is not None:
            plt.savefig(saveas)
        return

    def display_image_fits_at_times(
        self, time_points, vmax=None, ring_roi_width=None, saveas=None
    ):
        t_v = self.times[self._idx_fitted_frames]
        ntimes = len(t_v)
        rows = len(time_points)
        findex = list()
        for time in time_points:
            idx = np.where(t_v == time)[0][0]
            findex.append(idx)
        columns = 4
        counter = 0
        # print(rows, columns)
        f_height = 4 * rows
        f, axes = plt.subplots(
            rows, columns, figsize=(18, f_height), sharex=False, sharey=False
        )
        # x = np.linspace(-self.r_max, self.r_max, 201, endpoint=True)
        # r = np.abs(x)*1e-4
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
            if row >= n_rows:
                break
            time = t_v[i]
            # tcol = time_col[t_idx]
            Emax = self._fitting_parameters[i][0]
            beta = self.fitting_parameters[i][1]
            gamma = self._fitting_parameters[i][2]
            rmse = self._fitting_scores[i][0]
            rsse = self._fitting_scores[i][1]
            dF_sim = self.intensity_model(self.r, *self._fitting_parameters[i])
            image = self.images[self._idx_fitted_frames[i]] - self.background
            axes[row, 0].imshow(
                image, cmap="viridis", vmin=0, vmax=1.5 * vmax, extent=extent
            )
            axes[row, 0].set_xlabel(r"x ($\mu$m)", fontsize=14)
            axes[row, 0].set_ylabel(r"y ($\mu$m)", fontsize=14)
            axes[row, 0].set_title(
                "Exp. Image\nTime: {:.2f} s".format(time),
                fontdict={"fontsize": 10},
                pad=10,
            )
            axes[row, 1].imshow(
                dF_sim, cmap="viridis", vmin=0, vmax=vmax, extent=extent
            )
            axes[row, 1].set_xlabel(r"x ($\mu$m)", fontsize=14)
            axes[row, 1].set_ylabel(r"y ($\mu$m)", fontsize=14)
            axes[row, 1].set_title(
                "2D Point-Clark Fit\n$I_m$: {:.1e} | Beta: {:.1e} | $\gamma$: {:.1e} | RMSE: {:.1f}".format(
                    Emax, beta, gamma, rmse
                ),
                fontdict={"fontsize": 10},
                pad=10,
            )
            I_line_roi_exp = self.line_average(image)
            r_centers, I_ring_roi_exp, std_ring_roi_exp = self.radial_average(image)
            I_line_roi_fit = self.line_average(dF_sim)
            r_centers, I_ring_roi_fit, std_ring_roi_fit = self.radial_average(dF_sim)
            if i == 0:
                lineROI_zero_max = np.max(I_line_roi_exp)
                ringROI_zero_max = np.max(I_ring_roi_exp)
            # r_ex_mask2 = (r_ex[r_ex_mask] < r_stim) & (r_ex[r_ex_mask] > -r_stim)
            axes[row, 2].fill_between(
                [-self.r_stim, self.r_stim],
                1.5 * lineROI_zero_max,
                label="STIM",
                color="r",
                alpha=0.5,
            )
            axes[row, 2].plot(
                r_ex, I_line_roi_exp, linewidth=2, label="Exp.", color="grey"
            )
            axes[row, 2].plot(
                r_ex, I_line_roi_fit, linewidth=4, label="from Fit", color="k"
            )
            axes[row, 2].set_title("Line ROI", fontdict={"fontsize": 10}, pad=10)
            axes[row, 2].set_xlabel(r"Position ($\mu$m)", fontsize=14)
            axes[row, 2].set_ylabel(r"$\Delta F$", fontsize=14)
            axes[row, 2].set_ylim((0, 1.25 * lineROI_zero_max))
            axes[row, 2].legend(loc=0, fontsize=10)
            # Radial averages (Ring ROI)
            axes[row, 3].fill_between(
                [0, self.r_stim],
                1.2 * ringROI_zero_max,
                label="STIM",
                color="r",
                alpha=0.5,
            )
            axes[row, 3].plot(
                r_centers,
                I_ring_roi_exp,
                linewidth=2,
                label="Exp.",
                linestyle="",
                marker="8",
                markersize=10,
                color="grey",
            )
            # r_centers, fit_r_avg, fit_r_std = self.radial_average(r_img, dF_sim)
            axes[row, 3].plot(
                r_centers,
                I_ring_roi_fit,
                linewidth=4,
                label="from Fit",
                linestyle="--",
                color="k",
            )
            axes[row, 3].set_title("Ring ROI", fontdict={"fontsize": 10}, pad=10)
            axes[row, 3].set_xlabel(r"Distance ($\mu$m)", fontsize=14)
            axes[row, 3].set_ylabel(r"$\Delta F$", fontsize=14)
            axes[row, 3].set_ylim((0, 1.1 * ringROI_zero_max))
            axes[row, 3].legend(loc=0, fontsize=10)
            row += 1
        f.add_subplot(111, frameon=False)
        f.subplots_adjust(wspace=0.35)
        f.subplots_adjust(hspace=0.55)
        # hide tick and tick label of the big axes
        plt.tick_params(
            labelcolor="none",
            top="off",
            bottom="off",
            left="off",
            right="off",
            length=0,
        )
        plt.grid(False)
        plt.title("Step 1 - 2D Point-Clark fits over time", pad=40)
        plt.tight_layout()
        if saveas is not None:
            plt.savefig(saveas)
        return

    @property
    def fitting_parameters(self):
        """The intensity model fitting paramaters from step 1."""
        t_v = self.times[self._idx_fitted_frames]
        E_vals = self._fitting_parameters[:, 0]
        beta_vals = self._fitting_parameters[:, 1]
        gamma_vals = self._fitting_parameters[:, -1]
        RMSE = self._fitting_scores[:, 0]
        RSSE = self._fitting_scores[:, 1]
        fp_vals = list()
        for i in range(len(t_v)):
            fp_vals.append(
                {
                    "Time": t_v[i],
                    "Imax": E_vals[i],
                    "Beta": beta_vals[i],
                    "Gamma": gamma_vals[i],
                    "Gamma^2": gamma_vals[i] ** 2,
                    "RMSE": RMSE[i],
                    "RSSE": RSSE[i],
                }
            )
        return pd.DataFrame(fp_vals)

    def estimate_loss_rate(self):
        t_v = self.times[self._idx_fitted_frames]
        beta_vals = self._fitting_parameters[:, 1]
        loss_rate_data = _estimate_loss_rate(t_v, beta_vals)
        self._loss_rate_data = loss_rate_data
        self._loss_rate = loss_rate_data[0]
        return self._loss_rate


class AnisotropicGaussianFit(DiffusionFitBase):
    @staticmethod
    def intensity_model(x_distance, y_distance, E, gamma_x, gamma_y):
        """Gaussian diffusion function."""
        return models.anisotropic_gaussian(x_distance, y_distance, E, gamma_x, gamma_y)

    def _set_n_params(self):
        self._n_params = 3
        return

    def _fit_intensity(self, image, signal):
        """Non-linear fit of the images."""
        rmask = self.fitting_mask
        x = self.xv - self._diffusion_center[1]
        y = self.yv - self._diffusion_center[0]

        def cost(theta):
            if (theta < 0).any():
                return np.inf
            I_fit = self.intensity_model(x[rmask], y[rmask], *theta)
            I_exp = image[rmask]
            sse = measure.ss_error(I_exp, I_fit)
            return sse

        initial_guess = list()
        initial_guess.append(signal)
        for i in range(1, self._n_params):
            initial_guess.append(100)
        initial_guess = np.array(initial_guess)
        opt_res = minimize(cost, initial_guess, method="Nelder-Mead")
        # Sum of squared error from the minimized cost fucntion.
        sse = opt_res.fun
        # Root mean squared error.
        rmse = measure.sse_to_rmse(sse, self.n_fitted_pixels)
        return opt_res.x, sse, rmse

    # override the rsse function
    def rsse(self, image, theta):
        rmask = self.fitting_mask
        x = self.xv - self._diffusion_center[1]
        y = self.yv - self._diffusion_center[0]
        I_fit = self.intensity_model(x[rmask], y[rmask], *theta)
        I_exp = image[rmask]
        sse = measure.ss_error(I_exp, I_fit)
        return np.sqrt(sse)

    def fit(
        self,
        start=None,
        end=None,
        interval=1,
        verbose=False,
        apply_step1_threshold=True,
        step1_threshold=3,
        threshold_on="image",
    ):
        if start is None:
            start = self._idx_zero_time
        if end is None:
            end = self.n_frames
        if threshold_on not in self._threshold_on_options:
            warnings.warn(
                "------threshold_on = "
                + str(threshold_on)
                + " is not a valid option. ------\n------ Options are:"
                + str(self._threshold_on_options)
                + " ------\n------ Setting to default: image ------",
                RuntimeWarning,
            )
            threshold_on = "image"
        self._set_n_params()
        self._idx_fitted_frames = list()
        self._fitted_times = list()
        self._fitting_parameters = list()
        self._fitting_scores = list()
        r_peak = self.r_stim + 5 * self.pixel_width
        x_line = self._line
        r_line = np.abs(x_line)
        r_noise = np.max(r_line) - 10 * self.pixel_width
        gf_sigma = 5 * self.pixel_width
        for f in range(start, end, interval):
            img = self.images[f] - self.background

            peak_mask = (self.r > (self.r_stim + self.pixel_width)) & (self.r < r_peak)
            peak = img[peak_mask].mean()
            peak_std = img[peak_mask].std()
            n_peak = np.prod(img[peak_mask].shape)
            noise_mask = self.r > r_noise  # & (self.r < np.max(r_line))

            fit_parms, sse, rmse = self._fit_intensity(img, peak)

            if threshold_on == "image":
                tail_mean = img[noise_mask].mean()
                tail_std = img[noise_mask].std()
                n_tail = np.prod(img[noise_mask].shape)
            elif threshold_on == "fit":
                img_fit = self.intensity_model(self.r, *fit_parms)
                peak = img_fit[peak_mask].mean()
                n_peak = np.prod(img_fit[peak_mask].shape)
                tail_mean = img_fit[noise_mask].mean()
                tail_std = img_fit[noise_mask].std()
                n_tail = np.prod(img_fit[noise_mask].shape)
            elif threshold_on == "line":
                I_line = self.line_average(img)
                # Estimate the peak and tail peaks
                peak_mask = (r_line > (self.r_stim + self.pixel_width)) & (
                    r_line < r_peak
                )
                peak = I_line[peak_mask].mean()
                n_peak = len(I_line[peak_mask])
                noise_mask = r_line > r_noise
                tail_mean = I_line[noise_mask].mean()
                tail_std = I_line[noise_mask].std()
                n_tail = len(I_line[noise_mask])
            elif threshold_on == "filter":
                img_gf = gaussian_filter(img, sigma=4)
                peak = img_gf[peak_mask].mean()
                peak_std = img_gf[peak_mask].std()
                tail_mean = img_gf[noise_mask].mean()
                tail_std = img_gf[noise_mask].std()

            if apply_step1_threshold and (
                peak <= (tail_mean + step1_threshold * tail_std)
            ):
                if verbose:
                    print(
                        "stopping at frame {} time {} peak-signl {} <= tail-signal {} + {}x tail-std {}".format(
                            f,
                            self.times[f],
                            signal,
                            tail_mean,
                            step1_threshold,
                            tail_std,
                        )
                    )
                break

            rsse = self.rsse(img, fit_parms)
            if verbose:
                print(
                    "frame {} time {} peak-signal {} tail-signal {} tail-std {} fit_parms {} RMSE {} RSSE {:.1f}".format(
                        f,
                        self.times[f],
                        signal,
                        tail_mean,
                        tail_std,
                        fit_parms,
                        rmse,
                        rsse,
                    )
                )
            self._idx_fitted_frames.append(f)
            self._fitting_parameters.append(fit_parms)
            self._fitting_scores.append([rmse, rsse])
        if len(self._fitting_parameters) == 0:
            return np.nan
        self._fitting_parameters = np.array(self._fitting_parameters)
        self._fitting_scores = np.array(self._fitting_scores)
        linr_res_x, Ds_x, t0_x = self._fit_diffusion(
            self.times[self._idx_fitted_frames], self._fitting_parameters[:, 1]
        )
        linr_res_y, Ds_y, t0_y = self._fit_diffusion(
            self.times[self._idx_fitted_frames], self._fitting_parameters[:, -1]
        )
        self._linr_res_x = linr_res_x
        self._Ds_x = Ds_x * 1e-8  # 1e-8 converts from um^2/s to cm^2/s
        self._t0_x = t0_x
        self._linr_res_y = linr_res_y
        self._Ds_y = Ds_y * 1e-8  # 1e-8 converts from um^2/s to cm^2/s
        self._t0_y = t0_y
        self._Ds = np.array([self._Ds_x, self._Ds_y])
        return self._Ds

    @property
    def time_resolved_diffusion(self):
        """The time-resolved estimate of the diffusion coefficients."""
        t_v = self.fit_times
        gamma_x_vals = self._fitting_parameters[:, 1]
        lfit = self._leg_filter(t_v, gamma_x_vals ** 2)
        deriv = np.gradient(lfit, t_v)
        tr_dc_x = 0.25 * deriv * 1e-8  # 1e-8 converts from um^2/s to cm^2/s
        gamma_y_vals = self._fitting_parameters[:, -1]
        lfit = self._leg_filter(t_v, gamma_y_vals ** 2)
        deriv = np.gradient(lfit, t_v)
        tr_dc_y = 0.25 * deriv * 1e-8  # 1e-8 converts from um^2/s to cm^2/s
        return t_v, tr_dc_x, tr_dc_y

    def display_linear_fit(self, saveas=None):
        f, axes = plt.subplots(1, 2, figsize=(7, 4), sharex=False, sharey=True)
        t_v = self.times[self._idx_fitted_frames]

        gamma_vals = self._fitting_parameters[:, 1]
        R2_fit = self._linr_res_x.rvalue ** 2
        Ds_fit = self._Ds_x
        t0_fit = self._t0_x
        tspan = np.linspace(0, np.max(t_v) * 1.1, 500)
        # Generate the plot for the gamma^2 linear fit - IOI step 2 fitting
        axes[0].plot(
            t_v, gamma_vals ** 2, marker="o", linestyle="", label=None, color="grey"
        )
        axes[0].plot(
            tspan,
            self.diffusion_model(tspan, Ds_fit * 1e8, t0_fit),
            linestyle="--",
            label="Fit",
            color="k",
        )
        axes[0].set_ylabel(r"$\gamma_x^2$")
        axes[0].set_xlabel("Time (s)")
        axes[0].legend(loc=0, frameon=False)
        axes[0].set_title(
            "$R^2$={:.3f} | $D_x$={:.1f} x$10^{{-7}}$ cm$^2$/s | $t_0$={:.2f} s".format(
                R2_fit, Ds_fit * 1e7, t0_fit
            ),
            pad=10,
        )

        gamma_vals = self._fitting_parameters[:, -1]
        R2_fit = self._linr_res_y.rvalue ** 2
        Ds_fit = self._Ds_y
        t0_fit = self._t0_y
        axes[1].plot(
            t_v, gamma_vals ** 2, marker="o", linestyle="", label=None, color="grey"
        )
        axes[1].plot(
            tspan,
            self.diffusion_model(tspan, Ds_fit * 1e8, t0_fit),
            linestyle="--",
            label="Fit",
            color="k",
        )
        axes[1].set_ylabel(r"$\gamma_y^2$")
        axes[1].set_xlabel("Time (s)")
        axes[1].legend(loc=0, frameon=False)
        axes[1].set_title(
            "$R^2$={:.3f} | $D_y$={:.1f} x$10^{{-7}}$ cm$^2$/s | $t_0$={:.2f} s".format(
                R2_fit, Ds_fit * 1e7, t0_fit
            ),
            pad=10,
        )

        plt.suptitle(
            "Step 2 - linear fit of $\gamma^2$ vs. $t$ \n $N_t$={} | Effective Time={:.1f} s".format(
                len(self.fit_times), self.effective_time
            )
        )
        plt.tight_layout()
        sns.despine()
        if saveas is not None:
            plt.savefig(saveas)

    def display_time_resolved_dc(self, saveas=None):

        t_v, d_c_x, d_c_y = self.time_resolved_diffusion
        d_c_x *= 1e7  # x10-7 cm^2/s
        d_c_y *= 1e7  # x10-7 cm^2/s
        print("d_c_x: ", np.mean(d_c_x))
        print("d_c_y: ", np.mean(d_c_y))
        f, axes = plt.subplots(1, 2, figsize=(7, 4), sharex=False, sharey=True)
        axes[0].plot(t_v, d_c_x, marker="o", linestyle="-", label=None, color="grey")
        axes[0].set_ylabel(r"$D_y(t)$ (x$10^{{-7}}$ cm$^2$/s)")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylim((1, 70))
        axes[0].set_title("x-dimension", pad=10)
        axes[1].plot(t_v, d_c_y, marker="o", linestyle="-", label=None, color="grey")
        axes[1].set_ylabel(r"$D_x(t)$ (x$10^{{-7}}$ cm$^2$/s)")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylim((1, 70))
        axes[1].set_title("y-dimension", pad=10)
        # plt.legend(loc=0, frameon=False)
        plt.suptitle("Time-Resolved Diffusion Coefficient")
        plt.tight_layout()
        sns.despine()
        if saveas is not None:
            plt.savefig(saveas)

    def export_to_csv(self, prefix):
        fp_df = self.fitting_parameters
        fp_df.to_csv(prefix + "_step_1_fits.csv", index=False)
        fp_df_step2 = fp_df[["Time", "Gamma_x^2", "Gamma_y^2"]]
        lin_fit = self.diffusion_model(
            fp_df["Time"].values, self._Ds_x * 1e8, self._t0_x
        )
        fp_df_step2 = fp_df_step2.assign(LinearFitX=lin_fit)
        lin_fit = self.diffusion_model(
            fp_df["Time"].values, self._Ds_y * 1e8, self._t0_y
        )
        fp_df_step2 = fp_df_step2.assign(LinearFitY=lin_fit)
        fp_df_step2.to_csv(prefix + "_step_2_fits.csv", index=False)

    def display_image_fits(self, n_rows=5, vmax=None, ring_roi_width=None, saveas=None):
        t_v = self.times[self._idx_fitted_frames]
        ntimes = len(t_v)
        rows = n_rows
        interval = int(ntimes / n_rows)  # + 1
        if interval == 0:
            interval = 1
        columns = 4
        counter = 0
        # print(rows, columns)
        f_height = 4 * rows
        f, axes = plt.subplots(
            rows, columns, figsize=(18, f_height), sharex=False, sharey=False
        )
        # x = np.linspace(-self.r_max, self.r_max, 201, endpoint=True)
        # r = np.abs(x)*1e-4
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
            if row >= n_rows:
                break
            time = t_v[i]
            # tcol = time_col[t_idx]
            E = self._fitting_parameters[i][0]
            gamma_x = self._fitting_parameters[i][1]
            gamma_y = self._fitting_parameters[i][2]
            rmse = self._fitting_scores[i][0]
            rsse = self._fitting_scores[i][1]
            dF_sim = self.intensity_model(
                self.xv - self._diffusion_center[1],
                self.yv - self._diffusion_center[0],
                *self._fitting_parameters[i]
            )
            image = self.images[self._idx_fitted_frames[i]] - self.background
            axes[row, 0].imshow(
                image, cmap="viridis", vmin=0, vmax=1.5 * vmax, extent=extent
            )
            axes[row, 0].set_xlabel(r"y ($\mu$m)", fontsize=14)
            axes[row, 0].set_ylabel(r"x ($\mu$m)", fontsize=14)
            axes[row, 0].set_title(
                "Exp. Image\nTime: {:.2f} s".format(time),
                fontdict={"fontsize": 10},
                pad=10,
            )
            axes[row, 1].imshow(
                dF_sim, cmap="viridis", vmin=0, vmax=vmax, extent=extent
            )
            axes[row, 1].set_xlabel(r"x ($\mu$m)", fontsize=14)
            axes[row, 1].set_ylabel(r"y ($\mu$m)", fontsize=14)
            axes[row, 1].set_title(
                "2D Anisotropic Gaussian Fit\nE: {:.1e} | $\gamma_x$: {:.1e} | $\gamma_y$: {:.1e} | RMSE: {:.1f} | RSSE: {:.1f}".format(
                    E, gamma_x, gamma_y, rmse, rsse
                ),
                fontdict={"fontsize": 10},
                pad=10,
            )
            I_line_roi_exp = self.line_average(image)
            r_centers, I_ring_roi_exp, std_ring_roi_exp = self.radial_average(image)
            I_line_roi_fit = self.line_average(dF_sim)
            r_centers, I_ring_roi_fit, std_ring_roi_fit = self.radial_average(dF_sim)
            if i == 0:
                lineROI_zero_max = np.max(I_line_roi_exp)
                ringROI_zero_max = np.max(I_ring_roi_exp)
            # r_ex_mask2 = (r_ex[r_ex_mask] < r_stim) & (r_ex[r_ex_mask] > -r_stim)
            axes[row, 2].fill_between(
                [-self.r_stim, self.r_stim],
                1.5 * lineROI_zero_max,
                label="STIM",
                color="r",
                alpha=0.5,
            )
            axes[row, 2].plot(
                r_ex, I_line_roi_exp, linewidth=2, label="Exp.", color="grey"
            )
            axes[row, 2].plot(
                r_ex, I_line_roi_fit, linewidth=4, label="from Fit", color="k"
            )
            axes[row, 2].set_title("Line ROI", fontdict={"fontsize": 10}, pad=10)
            axes[row, 2].set_xlabel(r"Position ($\mu$m)", fontsize=14)
            axes[row, 2].set_ylabel(r"$\Delta F$", fontsize=14)
            axes[row, 2].set_ylim((0, 1.25 * lineROI_zero_max))
            axes[row, 2].legend(loc=0, fontsize=10)
            # Radial averages (Ring ROI)
            axes[row, 3].fill_between(
                [0, self.r_stim],
                1.2 * ringROI_zero_max,
                label="STIM",
                color="r",
                alpha=0.5,
            )
            axes[row, 3].plot(
                r_centers,
                I_ring_roi_exp,
                linewidth=2,
                label="Exp.",
                linestyle="",
                marker="8",
                markersize=10,
                color="grey",
            )
            # r_centers, fit_r_avg, fit_r_std = self.radial_average(r_img, dF_sim)
            axes[row, 3].plot(
                r_centers,
                I_ring_roi_fit,
                linewidth=4,
                label="from Fit",
                linestyle="--",
                color="k",
            )
            axes[row, 3].set_title("Ring ROI", fontdict={"fontsize": 10}, pad=10)
            axes[row, 3].set_xlabel(r"Distance ($\mu$m)", fontsize=14)
            axes[row, 3].set_ylabel(r"$\Delta F$", fontsize=14)
            axes[row, 3].set_ylim((0, 1.1 * ringROI_zero_max))
            axes[row, 3].legend(loc=0, fontsize=10)
            row += 1
        f.add_subplot(111, frameon=False)
        f.subplots_adjust(wspace=0.35)
        f.subplots_adjust(hspace=0.55)
        # hide tick and tick label of the big axes
        plt.tick_params(
            labelcolor="none",
            top="off",
            bottom="off",
            left="off",
            right="off",
            length=0,
        )
        plt.grid(False)
        plt.title("Step 1 - 2D Anisotropic Gaussian fits over time", pad=40)
        plt.tight_layout()
        if saveas is not None:
            plt.savefig(saveas)
        return

    def display_image_fits_at_times(
        self, time_points, vmax=None, ring_roi_width=None, saveas=None
    ):
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
        # print(rows, columns)
        f_height = 4 * rows
        f, axes = plt.subplots(
            rows, columns, figsize=(18, f_height), sharex=False, sharey=False
        )
        # x = np.linspace(-self.r_max, self.r_max, 201, endpoint=True)
        # r = np.abs(x)*1e-4
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
            if row >= n_rows:
                break
            time = t_v[i]
            # tcol = time_col[t_idx]
            E = self._fitting_parameters[i][0]
            gamma_x = self._fitting_parameters[i][1]
            gamma_y = self._fitting_parameters[i][2]
            rmse = self._fitting_scores[i][0]
            rsse = self._fitting_scores[i][1]
            dF_sim = self.intensity_model(
                self.xv - self._diffusion_center[1],
                self.yv - self._diffusion_center[0],
                *self._fitting_parameters[i]
            )
            image = self.images[self._idx_fitted_frames[i]] - self.background
            axes[row, 0].imshow(
                image, cmap="viridis", vmin=0, vmax=1.5 * vmax, extent=extent
            )
            axes[row, 0].set_xlabel(r"x ($\mu$m)", fontsize=14)
            axes[row, 0].set_ylabel(r"y ($\mu$m)", fontsize=14)
            axes[row, 0].set_title(
                "Exp. Image\nTime: {:.2f} s".format(time),
                fontdict={"fontsize": 10},
                pad=10,
            )
            axes[row, 1].imshow(
                dF_sim, cmap="viridis", vmin=0, vmax=vmax, extent=extent
            )
            axes[row, 1].set_xlabel(r"y ($\mu$m)", fontsize=14)
            axes[row, 1].set_ylabel(r"x ($\mu$m)", fontsize=14)
            axes[row, 1].set_title(
                "2D Anisotropic Gaussian Fit\nE: {:.1e} | $\gamma_x$: {:.1e} | $\gamma_y$: {:.1e} | RMSE: {:.1f} | RSSE: {:.1f}".format(
                    E, gamma_x, gamma_y, rmse, rsse
                ),
                fontdict={"fontsize": 10},
                pad=10,
            )
            I_line_roi_exp = self.line_average(image)
            r_centers, I_ring_roi_exp, std_ring_roi_exp = self.radial_average(image)
            I_line_roi_fit = self.line_average(dF_sim)
            r_centers, I_ring_roi_fit, std_ring_roi_fit = self.radial_average(dF_sim)
            if i == 0:
                lineROI_zero_max = np.max(I_line_roi_exp)
                ringROI_zero_max = np.max(I_ring_roi_exp)
            # r_ex_mask2 = (r_ex[r_ex_mask] < r_stim) & (r_ex[r_ex_mask] > -r_stim)
            axes[row, 2].fill_between(
                [-self.r_stim, self.r_stim],
                1.5 * lineROI_zero_max,
                label="STIM",
                color="r",
                alpha=0.5,
            )
            axes[row, 2].plot(
                r_ex, I_line_roi_exp, linewidth=2, label="Exp.", color="grey"
            )
            axes[row, 2].plot(
                r_ex, I_line_roi_fit, linewidth=4, label="from Fit", color="k"
            )
            axes[row, 2].set_title("Line ROI", fontdict={"fontsize": 10}, pad=10)
            axes[row, 2].set_xlabel(r"Position ($\mu$m)", fontsize=14)
            axes[row, 2].set_ylabel(r"$\Delta F$", fontsize=14)
            axes[row, 2].set_ylim((0, 1.25 * lineROI_zero_max))
            axes[row, 2].legend(loc=0, fontsize=10)
            # Radial averages (Ring ROI)
            axes[row, 3].fill_between(
                [0, self.r_stim],
                1.2 * ringROI_zero_max,
                label="STIM",
                color="r",
                alpha=0.5,
            )
            axes[row, 3].plot(
                r_centers,
                I_ring_roi_exp,
                linewidth=2,
                label="Exp.",
                linestyle="",
                marker="8",
                markersize=10,
                color="grey",
            )
            # r_centers, fit_r_avg, fit_r_std = self.radial_average(r_img, dF_sim)
            axes[row, 3].plot(
                r_centers,
                I_ring_roi_fit,
                linewidth=4,
                label="from Fit",
                linestyle="--",
                color="k",
            )
            axes[row, 3].set_title("Ring ROI", fontdict={"fontsize": 10}, pad=10)
            axes[row, 3].set_xlabel(r"Distance ($\mu$m)", fontsize=14)
            axes[row, 3].set_ylabel(r"$\Delta F$", fontsize=14)
            axes[row, 3].set_ylim((0, 1.1 * ringROI_zero_max))
            axes[row, 3].legend(loc=0, fontsize=10)
            row += 1
        f.add_subplot(111, frameon=False)
        f.subplots_adjust(wspace=0.35)
        f.subplots_adjust(hspace=0.55)
        # hide tick and tick label of the big axes
        plt.tick_params(
            labelcolor="none",
            top="off",
            bottom="off",
            left="off",
            right="off",
            length=0,
        )
        plt.grid(False)
        plt.title("Step 1 - 2D Anisotropic Gaussian fits over time", pad=40)
        plt.tight_layout()
        if saveas is not None:
            plt.savefig(saveas)
        return

    @property
    def fitting_parameters(self):
        """The intensity model fitting paramaters from step 1."""
        t_v = self.times[self._idx_fitted_frames]
        E_vals = self._fitting_parameters[:, 0]
        gamma_x_vals = self._fitting_parameters[:, 1]
        gamma_y_vals = self._fitting_parameters[:, -1]
        RMSE = self._fitting_scores[:, 0]
        RSSE = self._fitting_scores[:, 1]
        fp_vals = list()
        for i in range(len(t_v)):
            fp_vals.append(
                {
                    "Time": t_v[i],
                    "Imax": E_vals[i],
                    "Gamma_x": gamma_x_vals[i],
                    "Gamma_y": gamma_y_vals[i],
                    "Gamma_x^2": gamma_x_vals[i] ** 2,
                    "Gamma_y^2": gamma_y_vals[i] ** 2,
                    "RMSE": RMSE[i],
                    "RSSE": RSSE[i],
                }
            )
        return pd.DataFrame(fp_vals)

    def estimate_loss_rate(self):
        t_v = self.times[self._idx_fitted_frames]
        E_vals = self._fitting_parameters[:, 0]
        loss_rate_data = _estimate_loss_rate(t_v, E_vals)
        self._loss_rate_data = loss_rate_data
        self._loss_rate = loss_rate_data[0]
        return self._loss_rate

    @property
    def step2_rsquared(self):
        return np.array([self._linr_res_x.rvalue ** 2, self._linr_res_y.rvalue ** 2])


class AsymmetricFit(GaussianFit):
    def __init__(self, *args, **kwargs):

        self._asymm_axis = kwargs.pop("asymm_axis", "x")
        super().__init__(*args, **kwargs)
        self._intensity_ratios = list()
        self._intensity_ratio = 1
        # print(self._asymm_axis, self._diffusion_center, self.img_center, self._idx_img_center)
        if self._asymm_axis == "y":
            self._asymm_mask_p = (
                self.yv - self._diffusion_center[0]
            ) > 0  # 0.5 * self.pixel_width
            self._asymm_mask_n = (
                self.yv - self._diffusion_center[0]
            ) < 0  # -0.5 * self.pixel_width
        elif self._asymm_axis == "x":
            self._asymm_mask_p = (
                self.xv - self._diffusion_center[1]
            ) > 0  # 0.5 * self.pixel_width
            self._asymm_mask_n = (
                self.xv - self._diffusion_center[1]
            ) < 0  # -0.5 * self.pixel_width
        return

    def fit(self, *args, **kwargs):
        Dfree = kwargs.pop("free_diffusion", 44e-7)
        super().fit(*args, **kwargs)
        for idx in self._idx_fitted_frames[-3:]:
            img = self.images[idx] - self.background
            # mask = self._asymm_mask_p & (img > 0.)
            p_tot = np.sum(img[self._asymm_mask_p])
            # p_tot = img[mask].shape[0] #* img[mask].shape[1]
            # mask = self._asymm_mask_n & (img > 0.)
            # n_tot = img[mask].shape[0] #* img[mask].shape[1]
            n_tot = np.sum(img[self._asymm_mask_n])
            # if (p_tot > 0) and (n_tot > 0):
            ratio = p_tot / n_tot
            # else:
            #    ratio = 1.
            # print(p_tot, n_tot, ratio, img.max(), img[self._asymm_mask_p].shape)
            self._intensity_ratios.append(ratio)
        self._intensity_ratios = np.array(self._intensity_ratios)
        ir_bar = self._intensity_ratios.mean()
        self._intensity_ratio = ir_bar

        return self.asymm_diffusion(Dfree)

    @property
    def intensity_ratio(self):
        return self._intensity_ratio

    def asymm_tortuosity(self, Dfree):
        ir_bar_sq = self.intensity_ratio ** 2
        lambda_p = np.sqrt((Dfree * (1 + 1 / ir_bar_sq)) / (2 * self._Ds))
        lambda_n = self.intensity_ratio * lambda_p
        return np.array([lambda_n, lambda_p])

    def asymm_diffusion(self, Dfree):
        tort = self.asymm_tortuosity(Dfree)
        Dside = Dfree / tort ** 2
        return Dside

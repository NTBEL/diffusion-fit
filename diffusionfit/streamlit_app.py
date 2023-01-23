import os
import io
import plotly.express as px
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from tifffile import imwrite as tiffwrite

# diffusionfit
from diffusionfit import GaussianFit, PointClarkFit
from diffusionfit import AnisotropicGaussianFit
from diffusionfit import AsymmetricFit

# Cached convert from:
# https://docs.streamlit.io/library/api-reference/widgets/st.download_button
@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")


model_options = {
    "Standard Gaussian": GaussianFit,
    "Anisotropic Gaussian": AnisotropicGaussianFit,
    "Point-Clark": PointClarkFit,
    "Asymmetric": AsymmetricFit,
}

model_equations = {
    "Standard Gaussian": r"I = I_{\textrm{max}} e^{-\left(r / \gamma \right)^2}",
    "Anisotropic Gaussian": r"I = I_{\textrm{max}} e^{-\left[(x-x_0)/\gamma_x\right]^2} e^{-\left[(y-y_0)/\gamma_y \right]^2}",
    "Point-Clark": r"I = \frac{I_{\textrm{max}}}{1 + \beta e^{-\left(r / \gamma \right)^2}}",
    "Asymmetric": r"I = I_{\textrm{max}} e^{-\left(r / \gamma \right)^2}, \, \frac{N_+}{N_-}=\sqrt{\frac{D^{*}_{+}}{D^{*}_{-}}}, \, D^*=(D^*_{+} + D^*_-)/2"
}

st.title("Diffusion Fitting App")
st.text(
    "Point source paradigm-based fitting\nof 2D fluorescence microscope images to\nestimate diffusion coeffients."
)
st.markdown("------")

imgfile = st.file_uploader("Choose a fluorescence image file:", type=["tif", "tiff"])
# images = skio.imread(imgfile, plugin='tifffile')
# st.write("Number of frames: ",len(images))
# fig, ax = plt.subplots()
# ax.imshow(images[0])
# st.pyplot(fig)
which_model = st.selectbox(
    "Choose Intensity Model:",
    ("Standard Gaussian", "Anisotropic Gaussian", "Point-Clark", "Asymmetric"),
)
model = model_options[which_model]
st.latex(model_equations[which_model])
st.markdown("------")

required_input_columns = st.columns(4)

timestep = required_input_columns[0].number_input("time step (s) :", 0.01, value=0.25)
pixel_size = required_input_columns[1].number_input("pixel size (um):", 0.1, value=1.0)
stim_frame = required_input_columns[2].number_input("stimulation frame:", 1)
d_stim = required_input_columns[3].number_input("stim. zone diameter (um):", 0.0)

subtract_background = st.checkbox(
    "Subtract background (average of frames prior to stimulation)", value=True
)
center = st.selectbox("Image center point for fitting:", ("image", "intensity"))
col1, col2 = st.columns(2)
apply_threshold = col1.checkbox("Apply threshold", value=True)
threshold_value = col2.number_input("Threshold value (signal/noise):", 1.0, 10.0, 3.0)

if which_model == 'Asymmetric':
    st.markdown("------")
    st.write("Asymmetric diffusion parameters")
    asymm_input_cols = st.columns(2)
    free_diffusion_coeff = asymm_input_cols[0].number_input("Free diffusion (x10^-7 cm2/s): ", 0.01, 100., 0.01)
    asymm_axis = asymm_input_cols[1].radio("Axis of asymmetry: ", ['x', 'y'])
st.markdown("------")

# def fit_the_model():
#     D = dfit.fit()
#     return D
# Idea to use session_state to store data from here:
# https://blog.devgenius.io/streamlit-python-tips-how-to-avoid-your-app-from-rerunning-on-every-widget-click-cae99c5189eb
if "fit_state" not in st.session_state:
    st.session_state.fit_state = False
    st.session_state.fitted_model = None

# fit_model_button = st.button('Fit the model')
if st.button("Fit the model"):
    if imgfile is not None:
        #     st.write("Fitting: ", imgfile.name)
        with st.spinner("Fitting..."):
            if which_model == 'Asymmetric':
                dfit = model(
                    imgfile,
                    stimulation_frame=stim_frame,
                    timestep=timestep,
                    pixel_width=pixel_size,
                    stimulation_radius=d_stim / 2,
                    center=center,
                    subtract_background=subtract_background,
                    asymm_axis = asymm_axis,
                )
            else:    
                dfit = model(
                    imgfile,
                    stimulation_frame=stim_frame,
                    timestep=timestep,
                    pixel_width=pixel_size,
                    stimulation_radius=d_stim / 2,
                    center=center,
                    subtract_background=subtract_background,
                )
            st.write(imgfile.name, " has ", dfit.n_frames, " frames")
            if which_model == 'Aymmetric':
                D = dfit.fit(
                    apply_step1_threshold=apply_threshold, step1_threshold=threshold_value,
                    free_diffusion = (free_diffusion_coeff * 1e-7),
                )
            else:    
                D = dfit.fit(
                    apply_step1_threshold=apply_threshold, step1_threshold=threshold_value
                )
            st.session_state.fit_state = True
            st.session_state.fitted_model = dfit
            st.session_state.diffco = D
    else:
        st.error("No image file selected!")
st.markdown("------")

if st.session_state.fit_state:
    dfit = st.session_state.fitted_model
    D = st.session_state.diffco
    st.text("\n\n")
    if which_model == "Anisotropic Gaussian":
        st.write(
            "Estimated Anisotropic diffusion coefficients ", r"(x10^{-7} cm^2/s): ", np.round(D * 1e7, 2)
        )        
    elif which_model == "Asymmetric":
        st.write(
            "Estimated Asymmetric diffusion coefficients ", r"(x10^{-7} cm^2/s): ", np.round(D * 1e7, 2)
        )          
    else:        
        st.write(
            "Estimated diffusion coefficient ", r"(x10^{-7} cm^2/s): ", np.round(D * 1e7, 2)
        )

    st.write("Effective Time (s) : ", np.round(dfit.effective_time, 2))
    nframes = len(dfit._idx_fitted_frames)
    # Downloads for fitting parameters/data.
    sample_name = os.path.splitext(imgfile.name)[0]
    step1_df, step2_df = dfit.export_to_df()
    step1_csv = convert_df(step1_df)
    step2_csv = convert_df(step2_df)

    st.download_button(
        "Download the step 1 fitting parameters (.csv)",
        data=step1_csv,
        file_name=sample_name + "_step1_fits.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download the step 2 fitting parameters (.csv)",
        data=step2_csv,
        file_name=sample_name + "_step2_fits.csv",
        mime="text/csv",
    )
    # Download for the step1 fits as imagej compatible tiff.
    trajectory = list()
    fps = 1 / dfit.timestep
    dx = dfit.pixel_width
    for fit_parm in dfit._fitting_parameters:
        dF_sim = dfit.intensity_model(dfit.r, *fit_parm)
        trajectory.append(dF_sim.astype(np.float32))
    step1_tiff = io.BytesIO()
    tiffwrite(
        step1_tiff,
        np.array(trajectory),
        imagej=True,
        metadata={"spacing": dx, "unit": "micron", "axes": "TYX", "fps": fps},
    )
    st.download_button(
        "Download step 1 intensity fits as ImageJ tiff file",
        data=step1_tiff,
        file_name=sample_name + "_step1_fits_imagej.tiff",
    )

    st.markdown("-----")
    st.subheader("Fit viewer:")
    frame = st.slider("Frame", 1, nframes)
    frame_idx = frame - 1
    st.write(
        "| Time: ",
        np.round(dfit.fit_times[frame_idx], 3),
        " seconds |",
        "RMSE: ",
        np.round(dfit.step1_rmse[frame_idx], 2),
        " | ",
    )
    fig1 = px.imshow(
        dfit.images[dfit._idx_fitted_frames[frame_idx]], binary_string=True
    )
    fit_parm = dfit._fitting_parameters[frame_idx]
    dF_sim = dfit.intensity_model(dfit.r, *fit_parm)
    fig2 = px.imshow(dF_sim, binary_string=True)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Image", "Fit"))
    fig.add_trace(fig1.data[0], row=1, col=1)
    fig.add_trace(fig2.data[0], row=1, col=2)
    # image = dfit.images[dfit._idx_fitted_frames[frame_idx]] - dfit.background
    # I_line_roi_exp = dfit.line_average(image)
    # r_centers, I_ring_roi_exp, std_ring_roi_exp = dfit.radial_average(image)
    # I_line_roi_fit = dfit.line_average(dF_sim)
    # r_centers, I_ring_roi_fit, std_ring_roi_fit = dfit.radial_average(dF_sim)
    # r_ex = dfit._line
    # #df_line = pd.DataFrame({"x":r_ex, "exp.":I_line_roi_exp, "from fit":I_line_roi_fit})
    # fig_roi = make_subplots(rows=1, cols=2, subplot_titles=("Line ROI", "Ring ROI"))
    # fig_line_exp = px.line(x=r_ex, y=I_line_roi_exp)
    # fig_line_fit = px.line(x=r_ex, y=I_line_roi_fit)
    # fig_line_fit.update_traces(line_color='#0000ff', line_width=5)
    # fig_roi.add_trace(fig_line_exp.data[0], row=1, col=1)
    # fig_roi.add_trace(fig_line_fit.data[0], row=1, col=1)
    # col1, col2 = st.columns(2)
    # col1.text("Image")
    # col2.text("Fit")
    st.plotly_chart(fig)
    # st.plotly_chart(fig_roi)

    st.write(
        "|    Linear fit with R-squared: ", np.round(dfit.step2_rsquared, 4), "    |"
    )
    t_v = dfit.times[dfit._idx_fitted_frames]
    gamma_vals = dfit._fitting_parameters[:, -1]
    R2_fit = dfit._linr_res.rvalue ** 2
    Ds_fit = dfit._Ds
    t0_fit = dfit._t0
    # Generate the plot for the gamma^2 linear fit - IOI step 2 fitting
    figb = px.scatter(
        x=t_v,
        y=gamma_vals ** 2,
        labels={"y": r"$ \gamma^2 $", "x": "Time (s)"},
        title=None,
    )
    tspan = np.linspace(0, np.max(t_v) * 1.1, 500)
    figb.add_trace(
        go.Line(
            x=tspan,
            y=dfit.diffusion_model(tspan, Ds_fit * 1e8, t0_fit),
            name="linear fit",
        )
    )
    current_t = [[dfit.fit_times[frame_idx]], [gamma_vals[frame_idx] ** 2]]
    figb.add_trace(
        go.Scatter(
            x=current_t[0],
            y=current_t[1],
            marker=dict(size=20, color=3),
            name="current time",
        )
    )
    # figb.update_layout(paper_bgcolor="white")
    st.plotly_chart(figb)

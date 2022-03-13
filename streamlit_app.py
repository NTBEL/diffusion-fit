import plotly.express as px
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# diffusionfit
from diffusionfit import GaussianFit, PointClarkFit, AnisotropicGaussianFit

model_options = {
    "Standard Gaussian": GaussianFit,
    "Anisotropic Gaussian": AnisotropicGaussianFit,
    "Point-Clark": PointClarkFit,
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
    "Choose Model:", ("Standard Gaussian", "Anisotropic Gaussian", "Point-Clark")
)
model = model_options[which_model]

required_input_columns = st.columns(4)

timestep = required_input_columns[0].number_input("time step:", 0.1)
pixel_size = required_input_columns[1].number_input("pixel size:", 0.1)
stim_frame = required_input_columns[2].number_input("stimulation frame:", 1)
d_stim = required_input_columns[3].number_input("stimulation zone diameter:", 0.0)

subtract_background = st.checkbox(
    "Subtract background (average of frames prior to stimulation)", value=True
)
center = st.selectbox("Image center point for fitting:", ("image", "intensity"))
col1, col2 = st.columns(2)
apply_threshold = col1.checkbox("Apply threshold", value=True)
threshold_value = col2.number_input("Threshold value (signal/noise):", 1.0, 10.0, 3.0)

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
if st.button("Fit the model") or (not st.session_state.fit_state):
    if imgfile is not None:
        #     st.write("Fitting: ", imgfile.name)
        with st.spinner("Fitting..."):
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
            D = dfit.fit(
                apply_step1_threshold=apply_threshold, step1_threshold=threshold_value
            )
            st.session_state.fit_state = True
            st.session_state.fitted_model = dfit
    else:
        st.error("No image file selected!")
st.markdown("------")

if st.session_state.fit_state:
    dfit = st.session_state.fitted_model
    D = dfit._Ds
    st.text("\n\n")
    st.write("Estimated diffusion coefficient (x10^-7 cm^2/s): ", np.round(D * 1e7, 2))
    st.write("Effective Time (s) : ", np.round(dfit.effective_time, 2))
    nframes = len(dfit._idx_fitted_frames)
    st.markdown("-----")
    st.subheader("Fit viewer:")
    frame = st.slider("Frame", 1, nframes)
    frame_idx = frame - 1
    st.write("Time: ", np.round(dfit.fit_times[frame_idx], 2), " seconds")
    fig1 = px.imshow(
        dfit.images[dfit._idx_fitted_frames[frame_idx]], binary_string=True
    )
    fit_parm = dfit._fitting_parameters[frame_idx]
    dF_sim = dfit.intensity_model(dfit.r, *fit_parm)
    fig2 = px.imshow(dF_sim, binary_string=True)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Image", "Fit"))
    fig.add_trace(fig1.data[0], row=1, col=1)
    fig.add_trace(fig2.data[0], row=1, col=2)
    # col1, col2 = st.columns(2)
    # col1.text("Image")
    # col2.text("Fit")
    st.plotly_chart(fig)

    # st.write("Linear fit:")
    t_v = dfit.times[dfit._idx_fitted_frames]
    gamma_vals = dfit._fitting_parameters[:, -1]
    R2_fit = dfit._linr_res.rvalue ** 2
    Ds_fit = dfit._Ds
    t0_fit = dfit._t0
    # Generate the plot for the gamma^2 linear fit - IOI step 2 fitting
    figb = px.scatter(
        x=t_v,
        y=gamma_vals ** 2,
        labels={"y": "gamma^2", "x": "Time (s)"},
        title="Linear fitting",
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

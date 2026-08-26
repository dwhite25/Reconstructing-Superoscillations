
# Reconstructing Superoscillations Buried Deeply in Noise

Code and experimental data accompanying the published paper:

**D. D. White, S. Zhang, B. Šoda, A. Kempf, D. C. Struppa,
A. N. Jordan, and J. C. Howell,**
*"Reconstructing superoscillations buried deeply in noise,"*
**Physical Review A 110, L061502 (2024).**

DOI: **10.1103/PhysRevA.110.L061502**

This repository contains the numerical waveform construction, Fourier-domain analysis, experimental oscilloscope data, and signal-reconstruction workflow used to study superoscillations buried deeply in noise.

Superoscillations are local regions of a bandlimited signal that oscillate faster than the signal's highest Fourier component. In this work, a finite frequency comb is used to construct a superoscillatory waveform whose target region approximates a higher-bandwidth sinc function. Because the complete signal occupies only a known set of discrete frequencies, broadband experimental noise can be strongly suppressed through spectral filtering.

## Key Results

- The target superoscillatory region was measured at approximately **17 dB below the experimental noise floor**
- Spectral filtering of **10 waveform cycles** reconstructed this region with a mean-squared error equal to **1.19% of the superregion energy**
- Using 99 cycles reduced the reported reconstruction error to **0.16%**
- The reconstructed waveform experimentally distinguished two scattering paths separated by **0.1875 inverse bandwidths**, below the conventional 0.5 inverse-bandwidth range-resolution criterion used for comparison in the paper

![Range Resolution Well Below the Inverse Bandwidth](https://github.com/user-attachments/assets/bef47bab-a1b0-4fb2-ab51-5bb6af78ca60)

*After spectrally filtering noise from the return signal, superoscillations 17dB below the noise floor are recovered with sufficient fidelity to resolve two point-like scatterers at separations well below the inverse bandwidth.*


## Method

The numerical and experimental workflow consists of four main steps.

1. **Frequency-comb construction**

   A target sinc function is approximated over a finite interval using a finite sum of equally spaced Fourier components, $\psi(t)=\sum_{k=0}^{K-1} A_k e^{i\omega_k t}$. The Fourier coefficients $A_k$ are determined numerically by solving the corresponding linear least-squares problem.

2. **Frequency-comb optimization**

   The notebook searches over the number, bandwidth, and placement of the frequency components to balance target-region fitting accuracy against the large sidelobe amplitudes associated with superoscillatory signals. The experimental waveform uses six discrete frequencies and a total bandwidth one quarter that of the target sinc waveform.

3. **Experimental spectral reconstruction**

   The constructed waveform was transmitted through a branched BNC-cable arrangement that produced two propagation paths of adjustable relative length. The return signal was sampled by an oscilloscope at 4 GS/s.

   For each reconstruction, repeated waveform cycles are transformed into the frequency domain using FFT. The known frequency-comb components are retained while the remaining spectrum is rejected, after which an inverse FFT reconstructs the filtered time-domain waveform.

4. **Sub-bandwidth scatterer discrimination**

   Changing the relative cable lengths emulates two pointlike scattering paths with different delays. Measurements at several path separations demonstrate that the reconstructed superoscillatory region is sensitive to delays substantially below the inverse bandwidth of the complete transmitted waveform.

## Repository Contents

### `reconstructing_superoscillations.ipynb`

Primary analysis notebook for the project. It includes:

- construction of the target sinc waveform
- numerical optimization of the frequency comb
- least-squares determination of Fourier coefficients
- time- and frequency-domain analysis of the superoscillatory waveform
- processing of experimental oscilloscope measurements
- FFT-based spectral filtering and signal reconstruction
- comparison of reconstructed signals for multiple scattering-path separations

### Experimental oscilloscope data

The repository includes the measured return signals used by the notebook:

- `6_10mV_8mV_0ft.csv`
- `6_10mV_8mV_10ft.csv`
- `6_10mV_8mV_20ft.csv`
- `6_10mV_8mV_30ft.csv`

These measurements correspond to increasing differences between the two propagation-path lengths in the experimental setup.

## Reproducing the Analysis

Clone or download the repository and run `reconstructing_superoscillations.ipynb` from the repository directory.

The analysis uses standard Python scientific-computing packages:

- NumPy
- SciPy
- pandas
- Matplotlib

The included oscilloscope data allow the experimental reconstruction and scatterer-separation analyses to be reproduced directly from the notebook.

## Techniques Demonstrated

**Signal processing**
- Fourier analysis and FFT-based filtering
- frequency-domain noise rejection
- time-series reconstruction
- signal-to-noise analysis

**Numerical methods**
- linear least squares
- parameter-grid optimization
- complex waveform synthesis
- numerical error analysis

**Experimental sensing**
- oscilloscope waveform processing
- coherent signal reconstruction
- propagation-delay estimation
- sub-bandwidth range discrimination

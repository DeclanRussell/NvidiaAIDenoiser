# NVidia AI Denoier command line tool

This is a simple implementation of NVidia AI denoiser. You can find a pre-built windows distribution either on my website [here](https://declanrussell.com/portfolio/nvidia-ai-denoiser/) or in the releases tab of this repro. To build you will need to download optix 5.+ and copy over the following to contrib/optix/bin as they were too big for the repo,
* optix.1.dll
* optix_denoiser.dll
* cudnn64_7.dll
* cudart64_90.dll

These libraries will also need to be copied to the bin directory or add them to your path to run the app. The scons build will copy the required dlls from the contrib to the bin for you. You may also need to update your driver as I did.

You will require an Nvidia driver of at least 390.xx or higher to use the optix denoiser.

## Usage
Command line parameters
* -i [string] : path to input image
* -o [string] : path to output image
* -a [string] : path to input albedo AOV
* -n [string] : path to input normal AOV
* -b [float] : blend amount
* -h/--help : Lists command line parameters

You need to at least have an input and output for the app to run. If you also have them, you can add an albedo AOV or albedo and normal AOVs to improve the denoising. All images should be the same resolutions, not meeting this requirement will lead to unexpected results (likely a crash).

## Examples
### Noisy image
<p align="center">
  <img src="https://github.com/DeclanRussell/NvidiaAIDenoiser/blob/master/images/test.png" alt="test"/>
</p>

### Denoised output
<p align="center">
  <img src="https://github.com/DeclanRussell/NvidiaAIDenoiser/blob/master/images/denoised_test.png" alt="denoise_test"/>
</p>

# Licence info
This has no licence, do whatever you want with it just don't sue me if it breaks something!

# Known issues
* DevIL seems to have a issue with saving float RGBA images as PNGs so unfortunately in the current version you will not be able to save as PNG.
* The version of DevIL I have doesn't seem to work with EXRs

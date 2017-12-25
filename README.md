# NVidia AI Denoier command line tool

This is a simple implementation of NVidia AI denoiser. Unfortunately this currently depends on Qt at the moment, in the future I may remove this dependency. To build you will need to download optix 5.+ and copy over the following to contrib/optix/bin as they were too big for the repo,
* optix.1.dll
* optix_denoiser.dll
* cudnn64_7.dll
* cudart64_90.dll

These libraries will also need to be copied to the bin directory or add them to your path to run the app.
## Usage
Command line parameters
-i [string] : path to input image
-a [string] : path to input albedo AOV
-n [string] : path to input normal AOV
-b [float] : blend amount
-h/--help : Lists command line parameters

## Examples
<p align="center">
  <img src="https://github.com/DeclanRussell/NvidiaAIDenoiser/blob/master/images/test.png" alt="test"/>
</p>

<p align="center">
  <img src="https://github.com/DeclanRussell/NvidiaAIDenoiser/blob/master/images/denoised_test.png" alt="denoise_test"/>
</p>


#include <optix_world.h>
#include <iostream>
#include <iomanip>
#include "OpenImageIO\imageio.h"
#include "OpenImageIO\imagebuf.h"
#include <stdio.h>
#include <exception>

#include <time.h>
#ifdef _WIN32
#include <thread>
#include <chrono>
#include <windows.h>
#include <winternl.h>
#endif

#define DENOISER_MAJOR_VERSION 3
#define DENOISER_MINOR_VERSION 0

// Our global image handles
OIIO::ImageBuf* input_beauty = nullptr;
OIIO::ImageBuf* input_albedo = nullptr;
OIIO::ImageBuf* input_normal = nullptr;

// Logging verbosity level
int verbosity = 2;

// Application start time
std::chrono::high_resolution_clock::time_point app_start_time;

std::string getTime()
{
    std::chrono::duration<double, std::milli> time_span = std::chrono::high_resolution_clock::now() - app_start_time;
    double milliseconds = time_span.count();
    int seconds = floor(milliseconds / 1000.0);
    int minutes = floor((float(seconds) / 60.f));
    milliseconds -= seconds * 1000.0;
    seconds -= minutes * 60;
    char s[9];
    sprintf(s, "%02d:%02d:%03d", minutes, seconds, (int)milliseconds);
    return std::string(s);
}

template<typename... Args>
void PrintInfo(char *c, Args... args)
{
    if (!verbosity)
        return;
    char buffer[256];
    sprintf(buffer, c, args...);
    std::cout<<getTime()<<"       | "<<buffer<<std::endl;
}

template<typename... Args>
void PrintError(char *c, Args... args)
{
    char buffer[256];
    sprintf(buffer, c, args...);
    std::cerr<<getTime()<<" ERROR | "<<buffer<<std::endl;
}

#ifdef _WIN32
int getSysOpType()
{
    int ret = 0;
    NTSTATUS(WINAPI *RtlGetVersion)(LPOSVERSIONINFOEXW);
    OSVERSIONINFOEXW osInfo;

    *(FARPROC*)&RtlGetVersion = GetProcAddress(GetModuleHandleA("ntdll"), "RtlGetVersion");

    if (NULL != RtlGetVersion)
    {
        osInfo.dwOSVersionInfoSize = sizeof(osInfo);
        RtlGetVersion(&osInfo);
        ret = osInfo.dwMajorVersion;
    }
    return ret;
}
#endif

void exitfunc(int exit_code)
{
#ifdef _WIN32
    if (getSysOpType() < 10)
    {
        HANDLE tmpHandle = OpenProcess(PROCESS_ALL_ACCESS, TRUE, GetCurrentProcessId());
        if (tmpHandle != NULL)
        {
            PrintInfo("terminating...");
            std::this_thread::sleep_for(std::chrono::seconds(1)); // delay 1s
            TerminateProcess(tmpHandle, 0);
        }
    }
#endif
	exit(exit_code);
}

void cleanup()
{
    if (input_beauty) delete input_beauty;
    if (input_albedo) delete input_albedo;
    if (input_normal) delete input_normal;
}

void printParams()
{
    // Always print parameters if needed
    int old_verbosity = verbosity;
    verbosity = 1;
    PrintInfo("Command line parameters");
    PrintInfo("-v [int]      : log verbosity level 0:disabled 1:simple 2:full (default 2)");
    PrintInfo("-i [string]   : path to input image");
    PrintInfo("-o [string]   : path to output image");
    PrintInfo("-a [string]   : path to input albedo AOV (optional)");
    PrintInfo("-n [string]   : path to input normal AOV (optional, requires albedo AOV)");
    PrintInfo("-b [float]    : blend amount (default 0)");
    PrintInfo("-hdr [int]    : Use HDR training data (default 1)");
    PrintInfo("-maxmem [int] : Maximum memory size used by the denoiser in MB");
    PrintInfo("-repeat [int] : Execute the denoiser N times. Useful for profiling.");
    verbosity = old_verbosity;
}

int main(int argc, char *argv[])
{
    app_start_time = std::chrono::high_resolution_clock::now();
    if (argc > 1)
    {
        for (int i=1; i<argc; i++)
        {
            if (strcmp(argv[i], "-v"))
                continue;
            i++;
            if (i >= argc)
            {
               PrintError("incorrect number of arguments for flag -v");
            }
            verbosity = std::stoi(std::string ( argv[i] ));
            break;
        }
    }
    PrintInfo("Launching Nvidia AI Denoiser command line app v%d.%d", DENOISER_MAJOR_VERSION, DENOISER_MINOR_VERSION);
    PrintInfo("Created by Declan Russell (25/12/2017 ~ Merry Christmas!)");

    bool b_loaded, n_loaded, a_loaded;
    b_loaded = n_loaded = a_loaded = false;

    // Pass our command line args
    std::string out_path;
    float blend = 0.f;
    unsigned int hdr = 1;
    unsigned int num_runs = 1;
    float maxmem = 0.f;
    if (argc == 1)
    {
        printParams();
        exitfunc(EXIT_SUCCESS);
    }
    for (int i=1; i<argc; i++)
    {
        const std::string arg( argv[i] );
        if (arg == "-i")
        {
            i++;
            std::string path( argv[i] );
            if (verbosity >= 2)
                PrintInfo("Input image: %s", path.c_str());
            input_beauty = new OIIO::ImageBuf(path);
            if (input_beauty->init_spec(path, 0, 0))
            {
                if (verbosity >= 2)
                    PrintInfo("Loaded successfully");
                b_loaded = true;
            }
            else
            {
                PrintError("Failed to load input image");
                PrintError("[OIIO]: %s", input_beauty->geterror().c_str());
                cleanup();
                exitfunc(EXIT_FAILURE);
            }
        }
        else if (arg == "-n")
        {
            i++;
            std::string path( argv[i] );
            if (verbosity >= 2)
                PrintInfo("Normal image: %s", path.c_str());
            input_normal = new OIIO::ImageBuf(path);
            if (input_normal->init_spec(path, 0, 0))
            {
                if (verbosity >= 2)
                    PrintInfo("Loaded successfully");
                n_loaded = true;
            }
            else
            {
                PrintError("Failed to load normal image");
                PrintError("[OIIO]: %s", input_normal->geterror().c_str());
                cleanup();
                exitfunc(EXIT_FAILURE);
            }
        }
        else if (arg == "-a")
        {
            i++;
            std::string path( argv[i] );
            if (verbosity >= 2)
                PrintInfo("Albedo image: %s", path.c_str());
            input_albedo = new OIIO::ImageBuf(path);
            if (input_albedo->init_spec(path, 0, 0))
            {
                if (verbosity >= 2)
                    PrintInfo("Loaded successfully");
                a_loaded = true;
            }
            else
            {
                PrintError("Failed to load albedo image");
                PrintError("[OIIO]: %s", input_albedo->geterror().c_str());
                cleanup();
                exitfunc(EXIT_FAILURE);
            }
        }
        else if(arg == "-o")
        {
            i++;
            out_path = std::string( argv[i] );
            if (verbosity >= 2)
                PrintInfo("Output image: %s", out_path.c_str());
        }
        else if (arg == "-b")
        {
            i++;
            std::string blend_string( argv[i] );
            blend = std::stof(blend_string);
            if (verbosity >= 2)
                PrintInfo("Blend amount: %f", blend);
        }
        else if (arg == "-hdr")
        {
            i++;
            std::string hdr_string( argv[i] );
            hdr = std::stoi(hdr_string);
            if (verbosity >= 2)
                PrintInfo((hdr) ? "HDR training data enabled" : "HDR training data disabled");
        }
        else if (arg == "-maxmem")
        {
            i++;
            std::string maxmem_string( argv[i] );
            maxmem = float(std::stoi(maxmem_string) * 1ULL<<20);
            if (verbosity >= 2)
                PrintInfo("Maximum denoiser memory set to %d", maxmem);
        }
        else if (arg == "-repeat")
        {
            i++;
            std::string repeat_string( argv[i] );
            num_runs = std::max(std::stoi(repeat_string), 1);
            if (verbosity >= 2)
                PrintInfo("Number of repeats set to %d", num_runs);
        }
        else if (arg == "-h" || arg == "--help")
        {
            printParams();
        }
    }

    // Check if a beauty has been loaded
    if (!b_loaded)
    {
        PrintError("No input image could be loaded");
        cleanup();
        exitfunc(EXIT_FAILURE);
    }

    // If a normal AOV is loaded then we also require an albedo AOV
    if (n_loaded && !a_loaded)
    {
        PrintError("You cannot use a normal AOV without an albedo");
        cleanup();
        exitfunc(EXIT_FAILURE);
    }

    // Check for a file extension
    int x = (int)out_path.find_last_of(".");
    x++;
    const char* ext_c = out_path.c_str()+x;
    std::string ext(ext_c);
    if (!ext.size())
    {
        PrintError("No output file extension");
        cleanup();
        exitfunc(EXIT_FAILURE);
    }

    OIIO::ROI beauty_roi, albedo_roi, normal_roi;
    beauty_roi = OIIO::get_roi_full(input_beauty->spec());
    int b_width = beauty_roi.width();
    int b_height = beauty_roi.height();
    if (a_loaded)
    {
        albedo_roi = OIIO::get_roi_full(input_albedo->spec());
        if (n_loaded)
            normal_roi = OIIO::get_roi_full(input_normal->spec());
    }

    // Check that our feature buffers are the same resolution as our beauty
    int a_width = (a_loaded) ? albedo_roi.width() : 0;
    int a_height = (a_loaded) ? albedo_roi.height() : 0;
    if (a_loaded)
    {
        if (a_width != b_width || a_height != b_height)
        {
            PrintError("Aldedo image not same resolution as beauty");
            cleanup();
            exitfunc(EXIT_FAILURE);
        }
    }

    int n_width = (n_loaded) ? normal_roi.width() : 0;
    int n_height = (n_loaded) ? normal_roi.height() : 0;
    if (n_loaded)
    {
        if (n_width != b_width || n_height != b_height)
        {
            PrintError("Normal image not same resolution as beauty");
            cleanup();
            exitfunc(EXIT_FAILURE);
        }
    }

    // Get our pixel data
    std::vector<float> beauty_pixels(b_width * b_height * beauty_roi.nchannels());
    input_beauty->get_pixels(beauty_roi, OIIO::TypeDesc::FLOAT, &beauty_pixels[0]);

    // Catch optix exceptions
    try
    {
        // Create our optix context and image buffers
        optix::Context optix_context = optix::Context::create();
        optix::Buffer beauty_buffer = optix_context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, b_width, b_height);
        optix::Buffer albedo_buffer = optix_context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, a_width, a_height);
        optix::Buffer normal_buffer = optix_context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, n_width, n_height);
        optix::Buffer out_buffer = optix_context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, b_width, b_height);


        float* device_ptr = (float*)beauty_buffer->map();
        unsigned int pixel_idx = 0;
        for(unsigned int y=0; y<b_height; y++)
        for(unsigned int x=0; x<b_width; x++)
        {
            memcpy(device_ptr, &beauty_pixels[pixel_idx], sizeof(float) * beauty_roi.nchannels());
            device_ptr += 4;
            pixel_idx += beauty_roi.nchannels();
        }
        beauty_buffer->unmap();
        device_ptr = 0;

        if (a_loaded)
        {
            std::vector<float> albedo_pixels(a_width * a_height * albedo_roi.nchannels());
            input_albedo->get_pixels(albedo_roi, OIIO::TypeDesc::FLOAT, &albedo_pixels[0]);

            device_ptr = (float*)albedo_buffer->map();
            pixel_idx = 0;
            for(unsigned int y=0; y<a_height; y++)
            for(unsigned int x=0; x<a_width; x++)
            {
                memcpy(device_ptr, &albedo_pixels[pixel_idx], sizeof(float) * albedo_roi.nchannels());
                device_ptr += 4;
                pixel_idx += albedo_roi.nchannels();
            }
            albedo_buffer->unmap();
            device_ptr = 0;
        }

        if (n_loaded)
        {
            std::vector<float> normal_pixels(n_width * n_height * normal_roi.nchannels());
            input_normal->get_pixels(normal_roi, OIIO::TypeDesc::FLOAT, &normal_pixels[0]);

            device_ptr = (float*)normal_buffer->map();
            pixel_idx = 0;
            for(unsigned int y=0; y<n_height; y++)
            for(unsigned int x=0; x<n_width; x++)
            {
                memcpy(device_ptr, &normal_pixels[pixel_idx], sizeof(float) * normal_roi.nchannels());
                device_ptr += 4;
                pixel_idx += normal_roi.nchannels();
            }
            normal_buffer->unmap();
            device_ptr = 0;
        }

        // Setup the optix denoiser post processing stage
        optix::PostprocessingStage denoiserStage = optix_context->createBuiltinPostProcessingStage("DLDenoiser");
        denoiserStage->declareVariable("input_buffer")->set(beauty_buffer);
        denoiserStage->declareVariable("output_buffer")->set(out_buffer);
        denoiserStage->declareVariable("blend")->setFloat(blend);
        denoiserStage->declareVariable("hdr")->setUint(hdr);
        if (maxmem) denoiserStage->declareVariable("maxmem")->setUint(maxmem);
        denoiserStage->declareVariable("input_albedo_buffer")->set(albedo_buffer);
        denoiserStage->declareVariable("input_normal_buffer")->set(normal_buffer);

        // Add the denoiser to the new optix command list
        optix::CommandList commandList= optix_context->createCommandList();
        commandList->appendPostprocessingStage(denoiserStage, b_width, b_height);
        commandList->finalize();

        // Compile our context. I'm not sure if this is needed given there is no megakernal?
        optix_context->validate();
        optix_context->compile();

        // Execute denoise
        int sum = 0;
        for (unsigned int i = 0; i < num_runs; i++)
        {
            PrintInfo("Denoising...");
            clock_t start = clock(), diff;
            commandList->execute();
            diff = clock() - start;
            int msec = diff * 1000 / CLOCKS_PER_SEC;
            if (num_runs > 1)
                PrintInfo("Denoising run %d complete in %d.%03d seconds", i, msec/1000, msec%1000);
            else
                PrintInfo("Denoising complete in %d.%03d seconds", msec/1000, msec%1000);
            sum += msec;
        }
        if (num_runs > 1)
        {
            sum /= num_runs;
            PrintInfo("Denoising avg of %d complete in %d.%03d seconds", num_runs, sum/1000, sum%1000);
        }


        // Copy denoised image back to the cpu
        device_ptr = (float*)out_buffer->map();
        pixel_idx = 0;
        for(unsigned int y=0; y<b_height; y++)
        for(unsigned int x=0; x<b_width; x++)
        {
            memcpy(&beauty_pixels[pixel_idx], device_ptr, sizeof(float) * beauty_roi.nchannels());
            device_ptr += 4;
            pixel_idx += beauty_roi.nchannels();
        }
        out_buffer->unmap();
        device_ptr = 0;

        // Remove our gpu buffers
        beauty_buffer->destroy();
        normal_buffer->destroy();
        albedo_buffer->destroy();
        out_buffer->destroy();
        optix_context->destroy();

    }
    catch (const std::exception &e)
    {
        PrintError("[OptiX]: %s", e.what());
        cleanup();
        exitfunc(EXIT_FAILURE);
    }


    // If the image already exists delete it
    remove(out_path.c_str());

    // Set our OIIO pixels
    if (!input_beauty->set_pixels(beauty_roi, OIIO::TypeDesc::FLOAT, &beauty_pixels[0]))
        PrintError("Something went wrong setting pixels");

    // Save the output image
    PrintInfo("Saving to: %s", out_path.c_str());
    if (input_beauty->write(out_path))
        PrintInfo("Done!");
    else
    {
        PrintError("Could not save file %s", out_path.c_str());
        PrintError("[OIIO]: %s", input_beauty->geterror().c_str());
    }

    cleanup();
    exitfunc(EXIT_SUCCESS);
}

